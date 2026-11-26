//! The typesetting engine held to account: what it reads, how it spaces and sizes,
//! that nothing it is given makes it panic, and that what it sets is kept.

use super::layout::{self, Bx, Chunk, Item};
use super::parse::{parse, Node};
use super::symbols::{Class, Face};
use super::*;
use std::sync::Arc;

/// A context with the fonts installed, as the application has it from its second pass.
fn context() -> egui::Context {
    let ctx = egui::Context::default();
    let _ = ctx.run_ui(egui::RawInput::default(), |ui| crate::app::apply_style(ui.ctx()));
    ctx
}

/// Run `f` in a pass of `ctx`.
fn in_pass<R>(ctx: &egui::Context, f: impl FnOnce(&egui::Context) -> R) -> R {
    let mut f = Some(f);
    let mut out = None;
    let _ = ctx.run_ui(egui::RawInput::default(), |ui| {
        if let Some(f) = f.take() {
            out = Some(f(ui.ctx()));
        }
    });
    out.expect("the pass ran")
}

/// A formula laid out, its words (`\text`) measured by an estimate.
fn layout(nodes: &[Node], display: bool) -> Vec<Chunk> {
    layout::layout(nodes, display, &layout::estimate(0.5, TEXT_SCALE))
}

/// A formula laid out on one line (in the style of running text).
fn boxed(src: &str) -> Bx {
    let nodes = parse(src).unwrap_or_else(|e| panic!("{src}: {e}")).nodes;
    let chunks = layout(&nodes, false);
    let mut bx = Bx::default();
    let mut x = 0.0;
    for (i, c) in chunks.iter().enumerate() {
        bx.height = bx.height.max(c.bx.height);
        bx.depth = bx.depth.max(c.bx.depth);
        bx.items.extend(c.bx.items.iter().cloned().map(|item| match item {
            Item::Glyph { face, ch, size, at, color } => Item::Glyph { face, ch, size, at: at + egui::vec2(x, 0.0), color },
            other => other,
        }));
        x += c.bx.width + if i + 1 < chunks.len() { c.glue } else { 0.0 };
    }
    bx.width = x;
    bx
}

fn width(src: &str) -> f32 {
    boxed(src).width
}

/// The advance of a glyph as a formula sets it (an italic letter's with its italic
/// correction), in ems.
fn advance(face: Face, c: char) -> f32 {
    let m = symbols::metrics(face, c).expect("a glyph");
    m.width + if face == Face::MathItalic { m.italic } else { 0.0 }
}

fn close(a: f32, b: f32) -> bool {
    (a - b).abs() < 1e-4
}

/// The glyphs of a box: character, where, and at what size.
fn glyphs(bx: &Bx) -> Vec<(char, egui::Vec2, f32)> {
    bx.items.iter().filter_map(|i| if let Item::Glyph { ch, at, size, .. } = i { Some((*ch, *at, *size)) } else { None }).collect()
}

#[test]
fn text_is_cut_into_words_and_formulas() {
    use std::borrow::Cow;
    assert_eq!(pieces("plain").unwrap(), [Piece::Text(Cow::Borrowed("plain"))]);
    assert_eq!(
        pieces("Two arrows, $a$ and $b$.").unwrap(),
        [Piece::Text("Two arrows, ".into()), Piece::Math("a"), Piece::Text(" and ".into()), Piece::Math("b"), Piece::Text(".".into())]
    );
    // a dollar sign written as \$, outside a formula and inside one
    assert_eq!(pieces(r"costs \$5, $x = \$3$").unwrap(), [Piece::Text("costs $5, ".into()), Piece::Math(r"x = \$3")]);
    // (a command's backslash does not end a formula: \\ is a row of a matrix)
    assert_eq!(pieces(r"$a \\ b$").unwrap(), [Piece::Math(r"a \\ b")]);
    assert!(pieces("a lone $ sign").is_err());
    assert!(pieces("empty $$ formula").is_err());
    assert!(pieces("ends in a backslash $x\\").is_err());
    assert!(!has_math("no dollars here") && has_math("$x$"));
}

/// What a person who knows LaTeX writes is read, and so is the Unicode the text was
/// written in before: each pair here is the same formula.
#[test]
fn the_unicode_forms_mean_what_the_commands_mean() {
    for (unicode, latex) in [
        ("x₁²", "x_1^2"),
        ("x²₁", "x_1^2"),
        ("A⁻¹", "A^{-1}"),
        ("Aᵀ", r"A^\top"),
        ("σ²ₖ₊₁", r"\sigma^2_{k+1}"),
        ("eᴬᵗ", "e^{At}"),
        ("î", r"\hat\imath"),
        ("ĵ", r"\hat{\jmath}"),
        ("x̂", r"\hat x"),
        ("v⃗", r"\vec v"),
        ("ā", r"\bar a"),
        ("ẋ", r"\dot x"),
        ("√2", r"\sqrt2"),
        ("a ≠ b", r"a \ne b"),
        ("a ≤ b", r"a \le b"),
        ("x ∉ S", r"x \notin S"),
        ("a·b", r"a \cdot b"),
        ("a − b", "a - b"),
        ("λ ∈ ℝ", r"\lambda \in \mathbb{R}"),
        ("Δx", r"\Delta x"),
        ("x′", "x'"),
        ("f″", "f''"),
        ("x'", r"x^\prime"),
        ("∑ᵢ", r"\sum_i"),
        ("‖x‖", r"\|x\|"),
        ("1 … n", r"1 \ldots n"),
    ] {
        let (a, b) = (parse(unicode).map(|p| p.nodes), parse(latex).map(|p| p.nodes));
        assert!(a.is_ok() && a == b, "{unicode} read as {a:?}, {latex} as {b:?}");
    }
    // letters are italic, each its own; digits and Greek capitals upright
    let nodes = parse(r"ab2\Gamma\gamma").unwrap().nodes;
    let faces: Vec<Face> = nodes.iter().map(|n| if let Node::Sym { face, .. } = n { *face } else { panic!("{n:?}") }).collect();
    assert_eq!(faces, [Face::MathItalic, Face::MathItalic, Face::Main, Face::Main, Face::MathItalic]);
    assert!(matches!(parse(r"\sin").unwrap().nodes[0], Node::Func { ref name, limits: false } if name == "sin"));
    assert!(matches!(parse(r"\max").unwrap().nodes[0], Node::Func { limits: true, .. }));
    // a number in the form computers write it is set as one
    assert_eq!(parse("2.5e-3").map(|p| p.nodes), parse(r"2.5 \times {10}^{-3}").map(|p| p.nodes));
    assert_eq!(parse("1e5").map(|p| p.nodes), parse(r"1 \times {10}^{5}").map(|p| p.nodes));
    // (but not a variable e after a number)
    assert_eq!(parse("2e^x").map(|p| p.nodes), parse("2 e^x").map(|p| p.nodes));
}

#[test]
fn what_cannot_be_read_is_an_error() {
    for bad in [
        r"\foo",
        r"x^{2",
        r"x}",
        r"x^2^3",
        r"x_1_2",
        r"\right)",
        r"\left( x",
        r"\begin{matrix} a & b",
        r"\begin{nonsense} a \end{nonsense}",
        r"\begin{pmatrix} a \end{bmatrix}",
        r"a & b",
        r"a \\ b",
        r"\frac{1}",
        r"\sqrt",
        r"x^",
        r"\textcolor{nocolour}{x}",
        r"\mathbb{x}",
        r"\text{\alpha}",
        r"\not x",
        "☃",
        r"\left x \right)",
    ] {
        assert!(parse(bad).is_err(), "{bad} was read: {:?}", parse(bad).map(|p| p.nodes));
        assert!(inspect(&format!("text ${bad}$")).error.is_some(), "{bad}");
    }
    assert_eq!(inspect("plain text, no dollars"), Findings::default());
    // words set in italic are found, functions with commands of their own named
    assert_eq!(inspect(r"$softmax(x) + \operatorname{relu}(x) + \text{word} + abc$").words, ["softmax"]);
    assert!(is_function("tanh") && !is_function("softmax"));
}

/// TeX's spaces: medium round a binary operator, thick round a relation, thin after a
/// function's name; none round a minus that has nothing before it; none of the
/// medium and thick ones in scripts. Spaces typed in a formula count for nothing.
#[test]
fn atoms_are_spaced_as_in_tex() {
    let (a, b, plus, eq, minus) =
        (advance(Face::MathItalic, 'a'), advance(Face::MathItalic, 'b'), advance(Face::Main, '+'), advance(Face::Main, '='), advance(Face::Main, '−'));
    let (thin, medium, thick) = (3.0 / 18.0, 4.0 / 18.0, 5.0 / 18.0);
    assert!(close(width("a+b"), a + plus + b + 2.0 * medium));
    assert!(close(width("a + b"), width("a+b")));
    assert!(close(width("a=b"), a + eq + b + 2.0 * thick));
    assert!(close(width("-b"), minus + b), "unary minus: {}", width("-b"));
    assert!(close(width("a=-b"), a + eq + minus + b + 2.0 * thick));
    assert!(close(width("(-b)"), advance(Face::Main, '(') + minus + b + advance(Face::Main, ')')));
    assert!(close(width(r"\sin x"), width(r"\sin") + thin + advance(Face::MathItalic, 'x')));
    assert!(close(width("a,b"), a + advance(Face::Main, ',') + thin + b));
    assert!(close(width(r"a\,b"), a + thin + b) && close(width(r"a\quad b"), a + 1.0 + b));
    // in a superscript, at seven tenths of the size, the binary operator has no space
    let sup = width("x^{a+b}") - width("x") - 0.05;
    assert!(close(sup, 0.7 * (a + plus + b)), "{sup}");
    // relations side by side: no space between them
    assert!(close(width(r"a\le=b"), a + advance(Face::Main, '≤') + eq + b + 2.0 * thick));
}

/// The boxes of the main constructs, against TeX's rules: scripts at their shifts,
/// a fraction's bar on the axis, a root's bar over exactly its radicand, brackets
/// that grow with what they hold, a matrix centred on the axis.
#[test]
fn boxes_have_the_sizes_tex_gives_them() {
    let rule_at = |bx: &Bx| bx.items.iter().find_map(|i| if let Item::Rule { min, max, .. } = i { Some((*min, *max)) } else { None }).expect("a rule");
    // superscript raised by sup2 (0.363 em), subscript lowered by sub1 (0.15), both at 0.7
    let x2 = glyphs(&boxed("x^2"));
    assert!(close(x2[1].1.y, 0.363) && close(x2[1].2, 0.7), "{x2:?}");
    let x1 = glyphs(&boxed("x_1"));
    assert!(close(x1[1].1.y, -0.15), "{x1:?}");
    // with both, the subscript drops to at least sub2 (0.247), and the two are stacked
    let both = glyphs(&boxed("x_1^2"));
    assert!(both[2].1.y <= -0.247 + 1e-4 && close(both[1].1.x, both[2].1.x), "{both:?}");
    // a subscript is tucked under a leaning letter, by its italic correction
    let f = glyphs(&boxed("f_1"));
    let fm = symbols::metrics(Face::MathItalic, 'f').unwrap();
    assert!(fm.italic > 0.0 && close(f[1].1.x, fm.width), "{f:?}");
    // script of script at half the size
    assert!(glyphs(&boxed("x^{y^z}")).iter().any(|g| close(g.2, 0.5)));
    // a fraction: its bar on the axis (0.25 em), as wide as the wider part
    let frac = boxed(r"\frac{a}{bbb}");
    let (min, max) = rule_at(&frac);
    assert!(close((min.y + max.y) * 0.5, 0.25) && close(max.x - min.x, 3.0 * advance(Face::MathItalic, 'b') * 0.7), "{min:?} {max:?}");
    // a root: its bar exactly over the radicand, clear of it
    let root = boxed(r"\sqrt{x^2 + 1}");
    let (min, max) = rule_at(&root);
    let radicand = width("x^2 + 1");
    assert!(close(max.x - min.x, radicand) && close(root.width - min.x, radicand), "{min:?} {max:?} against {radicand}");
    let top = glyphs(&root)
        .iter()
        .map(|(c, at, size)| at.y + symbols::metrics(if c.is_ascii_digit() || *c == '+' { Face::Main } else { Face::MathItalic }, *c).unwrap().height * size)
        .fold(0.0f32, f32::max);
    assert!(min.y > top, "the bar at {} clears the radicand's top at {top}", min.y);
    // brackets grow with what they hold, and stay centred on the axis
    let small = boxed(r"\left( x \right)");
    let big = boxed(r"\left( \frac{\frac{a}{b}}{\frac{c}{d}} \right)");
    let tall = boxed(r"\left( \begin{matrix} a \\ b \\ c \end{matrix} \right)");
    assert!(small.height + small.depth < big.height + big.depth && big.height + big.depth < tall.height + tall.depth);
    for bx in [&small, &big, &tall] {
        assert!(((bx.height - bx.depth) * 0.5 - 0.25).abs() < 0.12, "off the axis: {} {}", bx.height, bx.depth);
    }
    // a bracket taller than the fonts have is drawn
    assert!(tall.items.iter().any(|i| matches!(i, Item::Stroke { .. })));
    // a matrix centred on the axis
    let m = boxed(r"\begin{matrix} a & b \\ c & d \end{matrix}");
    assert!(close((m.height - m.depth) * 0.5, 0.25), "{} {}", m.height, m.depth);
    // a large operator in a display takes its limits above and below
    let display = layout(&parse(r"\sum_{i=1}^n").unwrap().nodes, true);
    let placed = glyphs(&display[0].bx);
    let n = placed.iter().find(|g| g.0 == 'n').unwrap();
    let i = placed.iter().find(|g| g.0 == 'i').unwrap();
    assert!(n.1.y > 0.8 && i.1.y < -0.8, "{placed:?}");
}

/// Colour set by `\textcolor` reaches the parts it covers, and only those; a colour
/// switched on by `\color` lasts to the end of its group.
#[test]
fn colour_reaches_what_it_covers() {
    let pink = symbols::color("pink");
    let colours = |bx: &Bx| bx.items.iter().filter_map(|i| if let Item::Glyph { color, .. } = i { Some(*color) } else { None }).collect::<Vec<_>>();
    assert_eq!(colours(&boxed(r"\textcolor{pink}{a_1} + b")), [pink, pink, None, None]);
    // the colour does not make the relation an ordinary symbol: its spaces stay
    assert!(close(width(r"a \textcolor{pink}{=} b"), width("a = b")));
    assert_eq!(colours(&boxed(r"{a \color{#ff0000} b} c")), [None, Some(egui::Color32::from_rgb(255, 0, 0)), None]);
    assert_eq!(symbols::color("#12345678"), Some(egui::Color32::from_rgba_unmultiplied(0x12, 0x34, 0x56, 0x78)));
    assert_eq!(symbols::color("#1234"), None);
}

/// Line breaks inside a formula come after a relation or a binary operator, at its
/// top level only.
#[test]
fn formulas_may_break_after_relations_and_operators() {
    let chunks = |src: &str| layout(&parse(src).unwrap().nodes, false).iter().map(|c| c.breakable).collect::<Vec<_>>();
    assert_eq!(chunks("a = b + c"), [true, true, false]);
    assert_eq!(chunks("-a = b"), [true, false]);
    assert_eq!(chunks("{(a = b)}"), [false]);
    assert_eq!(chunks("a := b"), [true, false]);
    assert_eq!(chunks(r"\frac{a = b}{c}"), [false]);
}

/// A few thousand strings, random and malformed, of the characters and commands the
/// markup is made of: none makes the engine panic, and whatever is laid out is finite.
#[test]
fn nothing_makes_it_panic() {
    let parts = [
        "x",
        "y",
        "2",
        "+",
        "-",
        "=",
        "^",
        "_",
        "{",
        "}",
        "(",
        ")",
        "[",
        "]",
        "|",
        "&",
        "\\\\",
        "\\",
        "$",
        " ",
        "'",
        "²",
        "₁",
        "ᵀ",
        "√",
        "î",
        "\u{302}",
        "≠",
        "λ",
        "∑",
        r"\frac",
        r"\sqrt",
        r"\left(",
        r"\right)",
        r"\left.",
        r"\right|",
        r"\middle|",
        r"\begin{pmatrix}",
        r"\end{pmatrix}",
        r"\begin{cases}",
        r"\end{cases}",
        r"\text{",
        r"\operatorname{",
        r"\mathbf{",
        r"\mathbb{",
        r"\hat",
        r"\vec",
        r"\overline",
        r"\sum",
        r"\int",
        r"\lim",
        r"\limits",
        r"\textcolor{pink}{",
        r"\color{red}",
        r"\,",
        r"\quad",
        r"\big(",
        r"\not",
        r"\alpha",
        r"\displaystyle",
        r"\binom",
        r"[3]",
        "1e-5",
        "\n",
        "€",
    ];
    let mut rng = crate::math::rng::Rng::new(17);
    let ctx = context();
    for n in 0..4000 {
        let len = 1 + (rng.range(0.0, 24.0) as usize);
        let mut s = String::new();
        for _ in 0..len {
            s.push_str(parts[(rng.range(0.0, parts.len() as f32) as usize).min(parts.len() - 1)]);
        }
        let _ = pieces(&s);
        let _ = inspect(&s);
        let _ = plain(&s);
        assert!(estimate_width(&s, 12.0, 0.47).is_finite(), "{s:?}");
        if let Ok(parsed) = parse(&s) {
            for display in [false, true] {
                for chunk in layout(&parsed.nodes, display) {
                    let b = &chunk.bx;
                    assert!(b.width.is_finite() && b.height.is_finite() && b.depth.is_finite() && chunk.glue.is_finite(), "{s:?}");
                }
            }
        }
        // and every tenth all the way to a galley, wrapped narrow
        if n % 10 == 0 {
            let text = format!("words ${s}$ and ${s}$");
            let set = in_pass(&ctx, |ctx| paragraph(ctx, &[Span::new(&text, 12.0, egui::Color32::WHITE)], Mode::Wrap(60.0)));
            let set = set.expect("fonts in place");
            assert!(set.galley.size().is_finite() && set.galley.rows.iter().all(|r| r.pos.is_finite()), "{text:?}");
        }
    }
    // the nesting is limited, so a deep one is an error, not an overflowed stack
    assert!(parse(&"{".repeat(5000)).is_err());
    assert!(parse(&"x^".repeat(5000)).is_err());
}

/// A glyph is drawn where the layout puts it: the formula's baseline is the text's,
/// and a variable stands as tall as the letters beside it.
#[test]
fn formulas_sit_on_the_baseline_of_the_text() {
    let ctx = context();
    let ink = |row: &egui::epaint::text::PlacedRow| {
        let mut r = egui::Rect::NOTHING;
        for v in &row.visuals.mesh.vertices {
            r.extend_with(v.pos + row.pos.to_vec2());
        }
        r
    };
    for size in [12.0f32, 15.0, 31.0] {
        let set = in_pass(&ctx, |ctx| label(ctx, "$x$ x", size)).expect("fonts in place");
        let (math, text) = (ink(&set.galley.rows[0]), ink(set.galley.rows.last().expect("rows")));
        // (to within the pixel the rasteriser's edge adds)
        assert!((math.bottom() - text.bottom()).abs() <= 1.01, "size {size}: math x ends at {}, text x at {}", math.bottom(), text.bottom());
        assert!((math.height() - text.height()).abs() <= 2.01, "size {size}: math x {} tall, text x {}", math.height(), text.height());
    }
}

/// The ink of everything a paragraph draws.
fn ink(set: &Paragraph) -> egui::Rect {
    let mut r = egui::Rect::NOTHING;
    for row in &set.galley.rows {
        for v in &row.visuals.mesh.vertices {
            r.extend_with(v.pos + row.pos.to_vec2());
        }
    }
    r
}

/// A formula is set a little larger than the words round it (`TEXT_SCALE`), as a
/// compromise: its digits no more than a tenth taller than the words' (at 1.2 they were
/// 15 % taller, and read as a size larger), its letters no more than a tenth shorter.
/// Beside a monospaced read-out, its digits are exactly as tall as the read-out's own.
#[test]
fn formulas_are_the_size_of_the_words_round_them() {
    let ctx = context();
    let size = 100.0;
    let height = |text: &str, mono: bool| {
        let set = in_pass(&ctx, |ctx| paragraph(ctx, &[Span { mono, ..Span::new(text, size, egui::Color32::WHITE) }], Mode::Line)).expect("fonts in place");
        ink(&set).height()
    };
    // (a 1: flat on top and at the foot in every one of the fonts, so the ink is the
    // digits' height, with no overshoot)
    let digits = height("$1$", false) / height("1", false);
    assert!((1.0..=1.1).contains(&digits), "a formula's digits are {digits} times as tall as the words'");
    let letters = height("$x$", false) / height("x", false);
    assert!((0.9..=1.0).contains(&letters), "a formula's x is {letters} times as tall as the words'");
    let (mono, math) = (height("1", true), height("$1$", true));
    assert!((mono - math).abs() <= 1.01, "at {size} points a read-out's digits are {mono} tall, and a formula's in it {math}");
}

/// Words in a formula (`\text`) are set as the words round it are: in the interface's
/// font at the text's size (the monospaced one in a monospaced read-out), on the
/// formula's baseline, with scripts on them; a function's name stays in Computer
/// Modern, being mathematics.
#[test]
fn words_in_a_formula_are_set_as_the_words_round_it() {
    let ctx = context();
    let size = 30.0;
    let set = |text: &str, mono: bool| {
        in_pass(&ctx, |ctx| paragraph(ctx, &[Span { mono, ..Span::new(text, size, egui::Color32::WHITE) }], Mode::Line)).expect("fonts in place")
    };
    for mono in [false, true] {
        for words in ["dot product", "pixel", "Ag (y)"] {
            let (plain, math) = (set(words, mono), set(&format!(r"$\text{{{words}}}$"), mono));
            let (a, b) = (ink(&plain), ink(&math));
            assert!(
                (a.width() - b.width()).abs() <= 1.01 && (a.height() - b.height()).abs() <= 1.01,
                "{words:?} (monospaced: {mono}): {a:?} as words, {b:?} in a formula"
            );
            assert!((plain.galley.size().x - math.galley.size().x).abs() <= 0.5, "{words:?}: as wide as the words themselves");
        }
    }
    // on the baseline beside a variable, and with a script on it
    let base = |text: &str| ink(&set(text, false)).bottom();
    assert!((base(r"$x\,\text{x}$") - base("$x$")).abs() <= 1.01, "the words sit on the formula's baseline");
    let length = set(r"$\text{length}^2$", false);
    let word = ink(&set(r"$\text{length}$", false));
    let two = ink(&length);
    assert!(two.width() > word.width() + 5.0 && two.top() < word.top(), "the square stands up beside the word: {two:?} against {word:?}");
    // a function's name is not words: Computer Modern, as wide as its metrics say
    let det = set(r"$\det$", false).galley.size().x;
    let em = size * TEXT_SCALE;
    assert!((det - (advance(Face::Main, 'd') + advance(Face::Main, 'e') + advance(Face::Main, 't')) * em).abs() < 1.0, "\\det is {det} wide");
    assert!((set(r"$\text{det}$", false).galley.size().x - set("det", false).galley.size().x).abs() <= 0.5);
    // bold words are a hair wider than plain ones, and still in the interface's font
    let (plain, bold) = (set(r"$\text{word}$", false).galley.size().x, set(r"$\textbf{word}$", false).galley.size().x);
    assert!(bold > plain && bold < plain + 0.1 * size, "{bold} against {plain}");
}

/// A search looks in what the visitor sees: a formula's symbols as they are drawn, and
/// a Greek letter by its name too, never the markup's commands or braces.
#[test]
fn a_search_sees_what_is_shown() {
    let s = searchable(r"The eigenvalue $\lambda$ of $\frac{a}{b}$ in $\mathbb{R}^n$, and $\rho^n$; $\Phi$");
    for found in ["λ", "lambda", "eigenvalue", "ℝ", "ρ", "rho", "phi", "the"] {
        assert!(s.contains(found), "{found:?} not in {s:?}");
    }
    for not in ["\\", "{", "}", "$", "frac", "mathbb"] {
        assert!(!s.contains(not), "{not:?} in {s:?}");
    }
    assert!(searchable("typed: Λ and ϕ").contains("lambda") && searchable("ϕ").contains("phi"));
    assert_eq!(searchable("Plain words, with no formula"), "plain words, with no formula");
}

/// What is set is kept: asked for again in the same pass or the next, it is the
/// same; not asked for in a pass, it is dropped; another width is another entry. A
/// text that does not parse is set as it is written.
#[test]
fn what_is_set_is_kept() {
    let ctx = context();
    let get = |ctx: &egui::Context, text: &str, width: f32| {
        paragraph(ctx, &[Span::new(text, 12.0, egui::Color32::WHITE)], Mode::Wrap(width)).expect("fonts in place")
    };
    let text = r"the angle $\theta$, and $a \cdot b = |a|\,|b| \cos\theta$";
    let (first, again, other) = in_pass(&ctx, |ctx| (get(ctx, text, 200.0), get(ctx, text, 200.0), get(ctx, text, 120.0)));
    assert!(Arc::ptr_eq(&first, &again) && !Arc::ptr_eq(&first, &other));
    assert!(other.galley.size().y > first.galley.size().y, "narrower is taller");
    let next = in_pass(&ctx, |ctx| get(ctx, text, 200.0));
    assert!(Arc::ptr_eq(&first, &next), "kept into the next pass");
    in_pass(&ctx, |_| ());
    let later = in_pass(&ctx, |ctx| get(ctx, text, 200.0));
    assert!(!Arc::ptr_eq(&first, &later), "dropped after a pass without it");
    // a different text is never taken for another
    let (a, b) = in_pass(&ctx, |ctx| (get(ctx, "$a$", 200.0), get(ctx, "$b$", 200.0)));
    assert!(!Arc::ptr_eq(&a, &b));
    // markup that does not parse: the text as written, dollars and all
    let raw = in_pass(&ctx, |ctx| get(ctx, "broken $x^{2$ here", 400.0));
    assert!(raw.galley.rows.iter().any(|r| r.glyphs.iter().any(|g| g.chr == '$')));
    // before the fonts are in place, nothing (the caller shows the text as it is)
    let bare = egui::Context::default();
    assert!(in_pass(&bare, |ctx| label(ctx, "$x$", 12.0)).is_none());
}

/// Lines of a paragraph are spaced as plain text's are, except that a line holding a
/// formula taller than the words is moved down by just so much that the formula is as
/// far from the line above as the words' ascenders are, and no further (and the line
/// after one deeper than they are, likewise). A root's bar, a superscript, a fraction
/// in a line no longer reach into the line above.
#[test]
fn lines_make_room_for_tall_formulas_and_no_more() {
    let ctx = context();
    let size = 24.0;
    let set = |text: &str| in_pass(&ctx, |ctx| paragraph(ctx, &[Span::new(text, size, egui::Color32::WHITE)], Mode::Wrap(800.0))).expect("fonts in place");
    // (the ink: not the soft edges of a drawn stroke, which fade to nothing)
    let ink_of = |row: &egui::epaint::text::PlacedRow| {
        let mut r = egui::Rect::NOTHING;
        for v in row.visuals.mesh.vertices.iter().filter(|v| v.color.a() > 0) {
            r.extend_with(v.pos + row.pos.to_vec2());
        }
        r
    };
    // each line is "dl" and perhaps a formula: the baselines of the words, the top of
    // their ink, and the ink of the formula on the middle line
    let lines = |text: &str| {
        let set = set(text);
        let mut words = Vec::new();
        let mut formula = egui::Rect::NOTHING;
        for row in &set.galley.rows {
            match row.glyphs.first() {
                Some(g) if g.chr == 'd' => words.push((row.pos.y + g.pos.y, ink_of(row).top())),
                _ => formula = formula.union(ink_of(row)),
            }
        }
        (words, formula, set)
    };
    let (plain, _, set) = lines("dl\ndl\ndl");
    assert_eq!(plain.len(), 3);
    assert!(set.rise == 0.0 && set.drop == 0.0 && set.over == 0.0);
    let pitch = plain[1].0 - plain[0].0;
    assert!((plain[2].0 - plain[1].0 - pitch).abs() < 0.01);
    // how far below the line above the words' ascenders reach
    let clear = plain[1].1 - plain[0].0;
    // a formula no taller than the words: spaced as plain text
    for f in ["a", "(x, y)", "A_k", r"\lambda", r"\hat x", "|v|", r"2 \times 3"] {
        let (words, _, _) = lines(&format!("dl\ndl ${f}$\ndl"));
        assert!(words.iter().zip(&plain).all(|(a, b)| (a.0 - b.0).abs() < 0.01), "${f}$ moved the lines: {words:?} against {plain:?}");
    }
    // a tall one: room above it, as much as the words' ascenders have and no more (to
    // within a pixel, and the little that a formula's brackets stand above the words'
    // ascenders, which is let through)
    let slack = 0.06 * size + 1.01;
    for f in [r"\sqrt{x}", "x^2", r"\frac{1}{2}", r"\sqrt{\frac{a}{b}}", r"A^{-1}", r"e^{x^2}", r"\hat h", r"\hat A", r"\tilde{f}"] {
        let (words, formula, _) = lines(&format!("dl\ndl ${f}$\ndl"));
        let gap = formula.top() - words[0].0;
        assert!(gap >= clear - slack, "${f}$ reaches {gap} below the line above, the words' ascenders {clear}");
        assert!(words[1].0 - words[0].0 >= pitch - 0.01, "${f}$: lines closer than plain text's");
        if words[1].0 - words[0].0 > pitch + 0.5 {
            assert!(gap <= clear + 1.51, "${f}$ was given more room than it needs: {gap} against {clear}");
        }
        assert!(words[2].0 - words[1].0 >= pitch - 0.01, "${f}$: the line after it closer than plain text's");
    }
    // and one deeper than the words: the next line moves down
    let (words, formula, _) = lines("dl\ndl $\\frac{1}{\\sqrt{2}}$\ndl");
    assert!(words[2].0 - words[1].0 > pitch + 0.5 && words[2].1 - formula.bottom() >= 1.0, "the line after a deep formula: {words:?} {formula:?}");
    // what a paragraph says it needs beyond plain text, for lines set by hand
    let one = in_pass(&ctx, |ctx| label(ctx, r"$\sqrt{x^2}$", size)).expect("fonts in place");
    assert!(one.rise > 0.0 && one.drop == 0.0, "{} {}", one.rise, one.drop);
}

/// A display too wide for its room is broken where a mathematician would break it:
/// between two formulas set side by side, else before a relation, else before an
/// operator (which begins the new line), never before its first relation; its writer
/// may break it with `\\`; and every line is centred.
#[test]
fn displays_break_where_a_mathematician_would() {
    use layout::Break::{Between, None as Not, Operator, Relation};
    let kinds = |src: &str| layout(&parse(src).unwrap().nodes, true).iter().map(|c| c.kind).collect::<Vec<_>>();
    assert_eq!(kinds("a = b + c"), [Relation, Operator, Not]);
    assert_eq!(kinds("a := b - c"), [Relation, Operator, Not]);
    assert_eq!(kinds(r"x, \quad y = 1"), [Between, Relation, Not]);
    assert_eq!(kinds("-a = {b + c}"), [Relation, Not]);
    assert_eq!(kinds(r"\frac{a = b}{c}"), [Not]);
    // (in running text, after them, as before)
    assert_eq!(layout(&parse("a = b + c").unwrap().nodes, false).iter().map(|c| c.kind).collect::<Vec<_>>(), [Relation, Operator, Not]);
    assert_eq!(display_lines(r"a = b \\ c = d"), ["a = b ", " c = d"]);
    assert_eq!(display_lines(r"\begin{pmatrix} a \\ b \end{pmatrix} \\ c"), [r"\begin{pmatrix} a \\ b \end{pmatrix} ", " c"]);
    assert_eq!(display_lines(r"{a \\ b} \\ \{ c \\"), [r"{a \\ b} ", r" \{ c "]);
    assert_eq!(display_lines("x"), ["x"]);
    // the lines of a display: the glyphs of each, left to right, and where its ink starts and ends
    let ctx = context();
    let lines = |src: &str, width: f32| {
        let set = in_pass(&ctx, |ctx| paragraph(ctx, &[Span::new(src, 12.5, egui::Color32::WHITE)], Mode::Display(width))).expect("fonts in place");
        // (a script is a few points off its line's baseline; lines are twenty apart)
        let mut glyphs: Vec<(f32, f32, char)> =
            set.galley.rows.iter().flat_map(|row| row.glyphs.iter().map(move |g| (row.pos.y + g.pos.y, row.pos.x + g.pos.x, g.chr))).collect();
        glyphs.sort_by(|a, b| a.0.total_cmp(&b.0));
        let mut lines: Vec<(f32, Vec<(f32, char)>)> = Vec::new();
        for (y, x, c) in glyphs {
            match lines.last_mut() {
                Some((last, line)) if y - *last < 9.0 => {
                    *last = y;
                    line.push((x, c));
                }
                _ => lines.push((y, vec![(x, c)])),
            }
        }
        let ink = ink(&set);
        let lines: Vec<String> = lines
            .into_iter()
            .map(|(_, mut glyphs)| {
                glyphs.sort_by(|a, b| a.0.total_cmp(&b.0));
                glyphs.into_iter().map(|g| g.1).collect()
            })
            .collect();
        (lines, ink, set.galley.size().x)
    };
    // two formulas side by side: broken between them, each line centred
    let (two, ink, width) = lines(r"X = aZ + \sigma_X \varepsilon_X, \qquad Y = bX + cZ + \sigma_Y \varepsilon_Y", 220.0);
    assert_eq!(two.len(), 2, "{two:?}");
    assert!(two[0].ends_with(',') && two[1].starts_with('Y'), "{two:?}");
    assert!((ink.left() - (width - ink.right())).abs() < 30.0, "the wider line is centred: {ink:?} in {width}");
    // a long sum: before its operators, each line after the first beginning with one
    let (sum, _, _) =
        lines(r"\lVert A - A_k \rVert_F^2 = \sigma_{k+1}^2 + \sigma_{k+2}^2 + \sigma_{k+3}^2 + \sigma_{k+4}^2 + \sigma_{k+5}^2 + \cdots + \sigma_n^2", 160.0);
    assert!(sum.len() >= 3, "{sum:?}");
    assert!(sum[1..].iter().all(|l| l.starts_with('+')) && sum.iter().all(|l| !l.ends_with('+')), "{sum:?}");
    // relations before operators, and never before the first relation
    let (sigma, _, _) = lines(r"\sigma(x) = \frac{1}{1 + e^{-x}} = \frac{e^x}{e^x + 1} = 1 - \sigma(-x) = \frac{1}{2} + \frac{1}{2} \tanh\frac{x}{2}", 200.0);
    // (the lines with a relation in them: those of the fractions' parts have none)
    let main: Vec<&String> = sigma.iter().filter(|l| l.contains('=')).collect();
    assert!(main.len() >= 2 && main[0].starts_with("σ(x)=") && main[1..].iter().all(|l| l.starts_with('=')), "{sigma:?}");
    // the writer's own breaks, however wide the room
    let (written, _, _) = lines(r"(a + b)^2 = a^2 + 2ab + b^2 \\ (a - b)^2 = a^2 - 2ab + b^2", 1000.0);
    assert_eq!(written.len(), 2, "{written:?}");
    // centred, each of them
    let set = in_pass(&ctx, |ctx| paragraph(ctx, &[Span::new(r"a \\ a + a + a + a", 12.5, egui::Color32::WHITE)], Mode::Display(300.0))).unwrap();
    let xs: Vec<f32> = set.galley.rows.iter().filter_map(|r| r.glyphs.first().map(|g| r.pos.x + g.pos.x)).collect();
    let short = xs.first().copied().unwrap_or(0.0);
    let long = xs.iter().copied().skip(1).fold(f32::MAX, f32::min);
    assert!(short > long + 20.0 && (short - 150.0).abs() < 10.0, "the short line {short}, the long one from {long}");
}

/// A paragraph keeps to its width, breaking between words, and inside a formula only
/// where the formula would not fit on a line of its own.
#[test]
fn paragraphs_keep_to_their_width() {
    let ctx = context();
    let text = r"Least squares settles for the point $p$ of the plane nearest to $b$, and the error $e = b - p$ then meets the plane at a right angle: $a_1 \cdot e = 0$ and $a_2 \cdot e = 0$.";
    for width in [120.0f32, 200.0, 290.0] {
        let set = in_pass(&ctx, |ctx| paragraph(ctx, &[Span::new(text, 12.0, egui::Color32::WHITE)], Mode::Wrap(width))).unwrap();
        assert!(set.galley.size().x <= width + 0.5, "{} wide in {width}", set.galley.size().x);
    }
    // a formula that fits on the next line goes there whole rather than broken
    for words in 5..12 {
        let text = format!("{}$a + b = c$", "word ".repeat(words));
        let set = in_pass(&ctx, |ctx| paragraph(ctx, &[Span::new(&text, 12.0, egui::Color32::WHITE)], Mode::Wrap(250.0))).unwrap();
        let ys: Vec<f32> = set.galley.rows.iter().filter(|r| r.glyphs.iter().any(|g| "abc+=".contains(g.chr))).map(|r| r.pos.y).collect();
        assert!(ys.windows(2).all(|w| (w[0] - w[1]).abs() < 4.0), "the formula was broken after {words} words: {ys:?}");
    }
    // a display that does not fit is broken after a relation, every line within the width
    let set = in_pass(&ctx, |ctx| {
        paragraph(
            ctx,
            &[Span::new(r"\sigma(x) = \frac{1}{1 + e^{-x}} = \frac{e^x}{e^x + 1} = 1 - \sigma(-x)", 12.5, egui::Color32::WHITE)],
            Mode::Display(150.0),
        )
    })
    .unwrap();
    // (on two lines or more: one line of display fractions is under 35 points tall)
    assert!(set.galley.size().y > 45.0 && set.galley.size().x <= 150.5, "{:?}", set.galley.size());
}

/// The estimate with no fonts at hand counts a formula's own width, not its markup's.
#[test]
fn the_estimate_measures_formulas_not_markup() {
    let plain = estimate_width("lambda", 12.0, 0.47);
    let math = estimate_width(r"$\lambda$", 12.0, 0.47);
    assert!(math < plain * 0.5, "{math} against {plain}");
    assert!(close(estimate_width("$x$", 10.0, 0.47), advance(Face::MathItalic, 'x') * 10.0 * TEXT_SCALE));
    let classes: Vec<Class> = "=+(),".chars().map(|c| if let Some(symbols::Cmd::Sym(_, _, class)) = symbols::typed(c) { class } else { Class::Ord }).collect();
    assert_eq!(classes, [Class::Rel, Class::Bin, Class::Open, Class::Close, Class::Punct]);
}

/// Every command and every typed character the tables name has its glyph in the
/// fonts (a missing one would be drawn as a box).
#[test]
fn every_symbol_has_its_glyph() {
    let mut missing = Vec::new();
    for c in (0x20u32..0x2B00).filter_map(char::from_u32) {
        match symbols::typed(c) {
            Some(symbols::Cmd::Sym(ch, face, _)) if symbols::metrics(face, ch).is_none() => missing.push(c),
            Some(symbols::Cmd::Neg(ch)) if symbols::metrics(Face::Main, ch).is_none() => missing.push(c),
            Some(symbols::Cmd::BigOp(ch, _)) if symbols::metrics(Face::Size1, ch).is_none() || symbols::metrics(Face::Size2, ch).is_none() => missing.push(c),
            _ => {}
        }
        if let Some(text) = symbols::superscript(c).or(symbols::subscript(c)) {
            if parse(text).is_err() {
                missing.push(c);
            }
        }
    }
    assert!(missing.is_empty(), "{missing:?}");
    for name in
        ["alpha", "Omega", "imath", "cdot", "le", "mapsto", "lVert", "langle", "lfloor", "ldots", "cdots", "infty", "partial", "nabla", "top", "lesssim"]
    {
        assert!(parse(&format!("\\{name}")).is_ok(), "\\{name}");
    }
    for function in ["det", "tr", "rank", "dim", "ker", "span", "cos", "sin", "tan", "exp", "ln", "log", "max", "min", "arg", "sup", "inf", "lim"] {
        assert!(parse(&format!("\\{function} x")).is_ok(), "\\{function}");
    }
}

/// Prints the table of `ink.rs` (run with --ignored --nocapture after a change of
/// fonts or of egui, and paste it in).
#[test]
#[ignore]
fn print_the_ink_table() {
    let ctx = context();
    in_pass(&ctx, |ctx| {
        ctx.fonts_mut(|fonts| {
            for code in 32u8..127 {
                let ch = code as char;
                let (a, b) = (paint::drawn_ink(fonts, ch, false), paint::drawn_ink(fonts, ch, true));
                println!("    [[{:?}, {:?}], [{:?}, {:?}]], // {:?}", a[0], a[1], b[0], b[1], ch);
            }
        });
    });
}

/// The table says what drawing the letter says.
#[test]
fn the_ink_table_is_what_egui_draws() {
    let ctx = context();
    in_pass(&ctx, |ctx| {
        ctx.fonts_mut(|fonts| {
            for code in 32u8..127 {
                let ch = code as char;
                for mono in [false, true] {
                    assert_eq!(super::ink::known(ch, mono), Some(paint::drawn_ink(fonts, ch, mono)), "{ch:?}, monospace {mono}");
                }
            }
        });
    });
}

/// The panel of a scene opened a second time, a paragraph scrolled out of sight and
/// in again: what was set is still there, and is not set again.
#[test]
fn a_paragraph_that_comes_back_is_not_set_again() {
    let ctx = context();
    let text = r"The level $f$ and the spread $g$ of $Y = f(X) + g(X) N$, with $N$ independent of $X$, and a few more words to fill two lines.";
    let set = |ctx: &egui::Context| paragraph(ctx, &[Span::new(text, 13.0, egui::Color32::WHITE)], Mode::Wrap(260.0)).expect("the fonts are there");
    let first = in_pass(&ctx, set);
    take_built();
    // a dozen passes in which it is not asked for (another scene is being shown)
    for _ in 0..12 {
        in_pass(&ctx, |ctx| label(ctx, "something else", 12.0));
    }
    take_built();
    let again = in_pass(&ctx, set);
    assert!(Arc::ptr_eq(&first, &again), "the same paragraph, kept");
    assert_eq!(take_built().0, 0, "nothing was set afresh");
}

/// But not for ever: a text that changes in every frame leaves one paragraph behind
/// each time, and those go once there are more than a set number.
#[test]
fn what_is_kept_is_bounded() {
    let ctx = context();
    let mut most = 0;
    for frame in 0..(3 * paint::KEPT_AT_MOST) {
        in_pass(&ctx, |ctx| {
            label(ctx, &format!("step {frame}"), 12.0);
            label(ctx, "a caption that stays", 12.0);
        });
        most = most.max(paint::kept(&ctx));
    }
    assert!(most <= paint::KEPT_AT_MOST + 2, "{most} kept at once");
    // and what is drawn in every frame was never among what went
    take_built();
    in_pass(&ctx, |ctx| label(ctx, "a caption that stays", 12.0));
    assert_eq!(take_built().0, 0);
}

/// What the builder remembers from one paragraph to the next (the widths of words, a
/// font's measures, a formula's letters laid out) changes nothing in what it builds.
#[test]
fn a_paragraph_is_set_the_same_whatever_was_set_before_it() {
    let text =
        r"With $\hat f$ and $\hat g$ fitted, $(Y - \hat f(X)) / \hat g(X)$ is the noise $N$ itself, \text{but for} the fits' own errors: $\sum_j f_j d_j$.";
    let set = |ctx: &egui::Context| paragraph(ctx, &[Span::new(text, 12.5, egui::Color32::WHITE)], Mode::Wrap(244.0)).expect("the fonts are there");
    let alone = in_pass(&context(), set);
    let ctx = context();
    for k in 0..40 {
        in_pass(&ctx, |ctx| {
            paragraph(
                ctx,
                &[Span::new(&format!(r"Other words {k}, the fits and the noise, with $\hat g(X) = {k}$ and $N$."), 12.5, egui::Color32::WHITE)],
                Mode::Wrap(244.0),
            );
        });
    }
    let after = in_pass(&ctx, set);
    assert_eq!(alone.galley.rect, after.galley.rect);
    assert_eq!(alone.galley.rows.len(), after.galley.rows.len());
    for (a, b) in alone.galley.rows.iter().zip(&after.galley.rows) {
        assert_eq!(a.pos, b.pos);
        assert_eq!(a.row.glyphs.len(), b.row.glyphs.len());
        for (x, y) in a.row.glyphs.iter().zip(&b.row.glyphs) {
            assert_eq!((x.chr, x.pos), (y.chr, y.pos));
        }
    }
    assert_eq!((alone.over, alone.rise, alone.drop), (after.over, after.rise, after.drop));
}

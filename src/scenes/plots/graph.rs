//! Graph any function.

use super::*;

pub const DEF: SceneDef = SceneDef {
    id: "graph",
    title: "Graph any function",
    group: "Plotting",
    blurb: "Several curves at once, with sliders and time.",
    make: |g| Box::new(Graph::new(g)),
};

const GRAPH_VARS: &[&str] = &["x", "t", "a", "b"];

/// One of the "things to try": the curves, the sliders, and whether the tangent under
/// the pointer is shown.
struct Trial {
    label: &'static str,
    look_for: &'static str,
    curves: &'static [&'static str],
    a: f32,
    b: f32,
    slope: bool,
}

const TRIALS: [Trial; 3] = [
    Trial {
        label: "a wave that travels",
        look_for: r"The crests of $\sin(a x - t)$ move right by $1/a$ for every second of $t$: half a unit a second at $a = 2$. Set $a$ to $1$: the waves are twice as long, and travel twice as fast.",
        curves: &["sin(a x - t)"],
        a: 2.0,
        b: 1.0,
        slope: false,
    },
    Trial {
        label: "$b$ is the bell's width",
        look_for: r"Drag $b$ and the bell widens and narrows with it, but at $x = \pm b$ it is always 0.61 of its height (point there to read it): $b$ measures the width.",
        curves: &["exp(-x^2 / (2 b^2))"],
        a: 2.0,
        b: 1.0,
        slope: false,
    },
    Trial {
        label: "where a cubic turns",
        look_for: r"Move the pointer along the curve and watch the tangent: it lies flat (slope 0) exactly where the curve turns, at $x = \pm 1.83$. The turns are at $\pm\sqrt{10b/3}$: drag $b$ below $0$ and they are gone.",
        curves: &["0.1 x^3 - b x"],
        a: 2.0,
        b: 1.0,
        slope: true,
    },
];

/// A formula as a label on the picture: not longer than the picture has room for.
/// Typeset where it can be read as it is meant (`typeset`: the formula compiles, and
/// [`markup`] can write it); otherwise as typed, like the formula box.
fn label(text: &str, typeset: bool) -> String {
    let plain = format!("y = {}", text.trim());
    if plain.chars().count() > 32 {
        return format!("{}…", plain.chars().take(31).collect::<String>());
    }
    if typeset {
        if let Some(body) = markup(text, GRAPH_VARS) {
            let label = format!("$y = {body}$");
            let found = crate::typeset::inspect(&label);
            if found.error.is_none() && found.words.is_empty() {
                return label;
            }
        }
    }
    plain
}

/// The functions a formula may call (those of `math::expr`), and how each is written:
/// what goes before its argument and what after.
fn function(name: &str) -> Option<(String, &'static str)> {
    Some(match name {
        "sin" | "cos" | "tan" | "sinh" | "cosh" | "tanh" | "exp" | "ln" | "log" | "min" | "max" => (format!(r"\{name}("), ")"),
        "asin" | "acos" | "atan" => (format!(r"\arc{}(", &name[1..]), ")"),
        "sqrt" => (r"\sqrt{".to_string(), "}"),
        "abs" => (r"\left|".to_string(), r"\right|"),
        "floor" => (r"\left\lfloor ".to_string(), r"\right\rfloor"),
        "ceil" => (r"\left\lceil ".to_string(), r"\right\rceil"),
        "log10" => (r"\log_{10}(".to_string(), ")"),
        "log2" => (r"\log_2(".to_string(), ")"),
        "sign" | "relu" | "sigmoid" | "gauss" | "pow" | "atan2" => (format!(r"\operatorname{{{name}}}("), ")"),
        _ => return None,
    })
}

/// A formula as typed, as markup that reads as the program reads it: names upright
/// where they are functions, powers raised, `*` a dot, and names written side by side
/// (`ax`) taken apart as the program takes them apart (the longest of the variables
/// and constants that fits, a function only at the end before its bracket). `None`
/// where it is not sure of that: a character or name it does not know, a number
/// followed by an e (`2e-1` is a number to the program), brackets that do not close.
fn markup(text: &str, vars: &[&str]) -> Option<String> {
    #[derive(Clone, PartialEq)]
    enum Tok {
        Num(String),
        Name(String),
        Sym(char),
    }
    // the characters, as the program's lexer reads them
    let chars: Vec<char> = text
        .chars()
        .map(|c| match c {
            '−' => '-',
            '×' | '·' | '⋅' => '*',
            c => c,
        })
        .collect();
    let mut raw = Vec::new();
    let mut i = 0;
    while i < chars.len() {
        let c = chars[i];
        let start = i;
        if c.is_whitespace() {
            i += 1;
        } else if c.is_ascii_digit() || c == '.' {
            while i < chars.len() && (chars[i].is_ascii_digit() || chars[i] == '.') {
                i += 1;
            }
            if chars.get(i).is_some_and(|c| *c == 'e' || *c == 'E') {
                return None;
            }
            raw.push(Tok::Num(chars[start..i].iter().collect()));
        } else if c.is_ascii_alphabetic() {
            while i < chars.len() && chars[i].is_ascii_alphanumeric() {
                i += 1;
            }
            raw.push(Tok::Name(chars[start..i].iter().collect()));
        } else if c == 'π' {
            raw.push(Tok::Name("pi".into()));
            i += 1;
        } else if "+-*/^(),".contains(c) {
            raw.push(Tok::Sym(c));
            i += 1;
        } else {
            return None;
        }
    }
    // names written side by side, taken apart
    let constants = ["pi", "tau", "e"];
    let mut t = Vec::new();
    for (k, tok) in raw.iter().enumerate() {
        let Tok::Name(name) = tok else {
            t.push(tok.clone());
            continue;
        };
        let call = raw.get(k + 1) == Some(&Tok::Sym('('));
        if vars.contains(&name.as_str()) || constants.contains(&name.as_str()) || (call && function(name).is_some()) {
            t.push(tok.clone());
            continue;
        }
        let mut rest = name.as_str();
        while !rest.is_empty() {
            let last = |n: &str| call && n.len() == rest.len() && function(n).is_some();
            let fits = vars.iter().chain(&constants).copied().filter(|n| rest.starts_with(n)).chain(Some(rest).filter(|n| last(n)));
            let n = fits.max_by_key(|n| n.len())?;
            t.push(Tok::Name(n.to_string()));
            rest = &rest[n.len()..];
        }
    }

    struct Writer<'a> {
        t: &'a [Tok],
        i: usize,
        out: String,
    }
    impl Writer<'_> {
        /// The next piece of markup, a space before it unless it closes something or
        /// follows an opening.
        fn put(&mut self, piece: &str) {
            let opened = self.out.is_empty() || self.out.ends_with(['(', '{', '|', ' ']);
            let closes = piece.starts_with([')', '}', ',', '^']) || piece.starts_with(r"\right");
            if !opened && !closes {
                self.out.push(' ');
            }
            self.out.push_str(piece);
        }

        /// Up to a closing bracket (left for the caller) or the end.
        fn run(&mut self) -> Option<()> {
            while let Some(tok) = self.t.get(self.i) {
                match tok {
                    Tok::Sym(')') => break,
                    Tok::Sym('^') => {
                        self.i += 1;
                        self.power()?;
                    }
                    Tok::Sym('*') => {
                        self.put(r"\cdot");
                        self.i += 1;
                    }
                    Tok::Sym(c @ ('+' | '-' | '/' | ',')) => {
                        self.put(&c.to_string());
                        self.i += 1;
                    }
                    _ => self.unit()?,
                }
            }
            Some(())
        }

        /// What follows a `^`, raised: as the program reads it, a sign, then one
        /// number, name, call or bracket, and that raised in turn. A bracket that is
        /// the whole of the power loses its brackets (`x^(2a)` is $x^{2a}$).
        fn power(&mut self) -> Option<()> {
            self.put("^{");
            let signed = matches!(self.t.get(self.i), Some(Tok::Sym('-' | '+')));
            if let Some(Tok::Sym(c)) = self.t.get(self.i).filter(|_| signed) {
                self.put(&c.to_string());
                self.i += 1;
            }
            let (from, bracket) = (self.out.len(), self.t.get(self.i) == Some(&Tok::Sym('(')));
            self.unit()?;
            if self.t.get(self.i) == Some(&Tok::Sym('^')) {
                self.i += 1;
                self.power()?;
            } else if bracket && !signed {
                // (what was written since `from` is "(", the inside, ")")
                let inside = self.out.get(from + 1..self.out.len() - 1)?.to_string();
                self.out.truncate(from);
                self.out.push_str(&inside);
            }
            self.put("}");
            Some(())
        }

        /// A number, a name, a call, or a bracket.
        fn unit(&mut self) -> Option<()> {
            match self.t.get(self.i)? {
                Tok::Num(n) => {
                    self.put(n);
                    self.i += 1;
                }
                Tok::Sym('(') => {
                    self.i += 1;
                    self.put("(");
                    self.run()?;
                    self.close()?;
                    self.put(")");
                }
                Tok::Name(name) => {
                    self.i += 1;
                    if self.t.get(self.i) == Some(&Tok::Sym('(')) {
                        let (open, close) = function(name)?;
                        self.put(&open);
                        self.i += 1;
                        self.run()?;
                        self.close()?;
                        self.put(close);
                    } else {
                        self.put(match name.as_str() {
                            "pi" => r"\pi",
                            "tau" => r"\tau",
                            n => n,
                        });
                    }
                }
                Tok::Sym(_) => return None,
            }
            Some(())
        }

        fn close(&mut self) -> Option<()> {
            (self.t.get(self.i) == Some(&Tok::Sym(')'))).then(|| self.i += 1)
        }
    }

    let mut w = Writer { t: &t, i: 0, out: String::new() };
    w.run()?;
    // (a closing bracket with nothing open is not read here)
    (w.i == t.len() && !t.is_empty()).then_some(w.out)
}

pub struct Graph {
    rows: Vec<ExprField>,
    a: f32,
    b: f32,
    show_slope: bool,
    /// The time the picture was last drawn at, for the read-out.
    t: f32,
    xs: Vec<f32>,
    ys: Vec<f32>,
    batch: Batch,
    /// Each curve's label on the picture, and the formula it was made from (made
    /// again only when that changes).
    labels: Vec<(Option<String>, String)>,
}

impl Graph {
    pub fn new(_: &Gfx) -> Self {
        Self::create()
    }

    /// Set up one of the experiments in `TRIALS`.
    fn set_up(&mut self, k: usize) {
        let t = &TRIALS[k];
        self.rows = t.curves.iter().map(|c| ExprField::new(c, GRAPH_VARS)).collect();
        (self.a, self.b, self.show_slope) = (t.a, t.b, t.slope);
    }

    /// (Nothing here needs the graphics device, so the tests can make one too.)
    fn create() -> Self {
        Graph {
            rows: vec![
                ExprField::new("sin(a x - t) / (1 + 0.1 x^2)", GRAPH_VARS),
                ExprField::new("exp(-x^2 / (2 b^2))", GRAPH_VARS),
                ExprField::new("0.1 x^3 - b x", GRAPH_VARS),
            ],
            a: 2.0,
            b: 1.0,
            show_slope: false,
            t: 0.0,
            xs: Vec::new(),
            ys: Vec::new(),
            batch: Batch::default(),
            labels: Vec::new(),
        }
    }
}

impl Scene for Graph {
    fn view(&self) -> View {
        View::Flat { center: Vec2::ZERO, half_height: 3.2 }
    }

    fn animated(&self) -> bool {
        self.rows.iter().any(|r| r.expr.uses(1))
    }

    fn settings(&self, s: &mut ViewSettings) {
        s.grid_alpha = 0.22;
    }

    fn ui(&mut self, ui: &mut egui::Ui, _: &Gfx) {
        ui::note(ui, "Plot anything: each curve is $y$ written as a formula in $x$, drawn for every $x$ in view. Formulas can also use the sliders $a$ and $b$, and $t$ for time. Move the pointer over the plot to read each curve's value there.");
        ui::section(ui, "Things to try");
        for (k, trial) in TRIALS.iter().enumerate() {
            if ui::experiment(ui, trial.label, trial.look_for) {
                self.set_up(k);
            }
        }
        ui::section(ui, "Curves");
        ui::note(ui, "Functions: sin cos tan exp ln sqrt abs tanh sigmoid relu floor min max and more; write 2x or x^2 as you would on paper.");
        let mut remove = None;
        let many = self.rows.len() > 1;
        for (i, row) in self.rows.iter_mut().enumerate() {
            ui.horizontal(|ui| {
                ui.label(egui::RichText::new("●").color(pal::SERIES[i % 8].egui()));
                ui.label(egui::RichText::new(format!("curve {}", i + 1)).size(12.0).color(pal::DIM.egui()));
                if many && ui.small_button("remove").clicked() {
                    remove = Some(i);
                }
            });
            row.ui(ui, "y =");
        }
        if let Some(i) = remove {
            self.rows.remove(i);
        }
        if self.rows.len() < 6 && ui.button("add a curve").clicked() {
            self.rows.push(ExprField::new("cos(x)", GRAPH_VARS));
        }
        ui::section(ui, "Parameters");
        ui::slider(ui, "$a$", &mut self.a, -5.0..=5.0);
        ui::slider(ui, "$b$", &mut self.b, -5.0..=5.0);
        ui::note(ui, "Numbers the formulas can use: drag them and watch the curves change.");
        ui.checkbox(&mut self.show_slope, "show the tangent under the pointer");
        ui::section(ui, "Reading the picture");
        let mut key: Vec<(Color, String)> = self.rows.iter().enumerate().map(|(i, _)| (pal::SERIES[i % 8], format!("curve {}, named at the right-hand edge", i + 1))).collect();
        key.push((pal::DIM, "the line under the pointer: each curve's value there is written beside it (and its slope, with the tangent shown)".to_string()));
        ui::key(ui, &key.iter().map(|(c, s)| (*c, s.as_str())).collect::<Vec<_>>());
        if self.animated() {
            ui::section(ui, "Numbers");
            ui::readout(ui, "$t$", format!("{:.1}", self.t));
            ui::note(ui, "$t$ is the scene's clock: it stops while the scene is paused and runs faster or slower with the time speed (under View).");
        }
        ui::see_also(ui, &[("calc1", "slopes, areas and Taylor polynomials of one curve"), ("parametric", "surfaces from formulas"), ("fourier", "a curve drawn by circles")]);
    }

    fn frame(&mut self, f: &mut Frame) {
        let (lo, hi) = f.cam.flat_bounds();
        self.t = f.t;
        let n = 1200;
        self.xs.clear();
        self.xs.extend((0..=n).map(|i| lo.x + (hi.x - lo.x) * i as f32 / n as f32));
        self.ys.resize(self.xs.len(), 0.0);
        let mouse = f.mouse();
        if let Some(m) = mouse {
            f.line((m.x, lo.y), (m.x, hi.y), 1.0, pal::DIM.a(0.6));
        }
        for (i, row) in self.rows.iter().enumerate() {
            let color = pal::SERIES[i % 8];
            row.expr.eval_batch(&[&self.xs, &[f.t], &[self.a], &[self.b]], &mut self.ys, &mut self.batch);
            plot_samples(f, &self.xs, &self.ys, 2.8, color, false);
            // read-out under the pointer
            if let Some(m) = mouse {
                let vars = [m.x, f.t, self.a, self.b];
                let (y, slope, _) = row.expr.grad2(&vars, 0, 9);
                if y.is_finite() && y > lo.y && y < hi.y {
                    if self.show_slope && slope.is_finite() {
                        let w = (hi.x - lo.x) * 0.12;
                        f.line((m.x - w, y - slope * w), (m.x + w, y + slope * w), 1.6, color.a(0.8));
                    }
                    f.dot((m.x, y), 9.0, color);
                    // (a slope that is not a number is written as it always was)
                    let number = |v: f32| if v.is_finite() { format!("${}$", fmt(v)) } else { fmt(v) };
                    let text = if self.show_slope { format!("{}   slope {}", number(y), number(slope)) } else { number(y) };
                    f.text_off((m.x, y), Vec2::new(10.0, -10.0), text, 13.0, color, Align2::LEFT_BOTTOM);
                }
            }
        }
        if let Some(m) = mouse {
            f.text_off((m.x, lo.y), Vec2::new(6.0, -6.0), format!("$x = {}$", fmt(m.x)), 12.0, pal::DIM, Align2::LEFT_BOTTOM);
        }
        // name each curve by its formula, near the right-hand edge and above it, on a
        // plate (other curves pass there), one under another where they would meet
        let (size, xl) = (12.0, hi.x - 0.03 * (hi.x - lo.x));
        self.labels.resize(self.rows.len(), (None, String::new()));
        for (row, (from, text)) in self.rows.iter().zip(&mut self.labels) {
            if from.as_deref() != Some(row.text.as_str()) {
                *text = label(&row.text, row.error.is_none());
                *from = Some(row.text.clone());
            }
        }
        let mut labels = Vec::new();
        for (i, row) in self.rows.iter().enumerate() {
            let y = row.expr.eval(&[xl, f.t, self.a, self.b]);
            if let Some(at) = on_screen(f, Vec3::new(xl, y, 0.0)).filter(|_| y > lo.y && y < hi.y) {
                labels.push((at - Vec2::new(0.0, 10.0), self.labels.get(i).map_or_else(String::new, |l| l.1.clone()), pal::SERIES[i % 8]));
            }
        }
        labels.sort_by(|p, q| p.0.y.total_cmp(&q.0.y));
        let mut below = f32::NEG_INFINITY;
        for (at, text, color) in labels {
            let y = at.y.max(below + size + 10.0);
            f.caption_px(Vec2::new(at.x, y), text, size, color, Align2::RIGHT_BOTTOM);
            below = y;
        }
        // name the axes at their far ends, while they are in view; on plates, because
        // any curve may pass there
        let view = f.view_size();
        let origin = Vec2::new(-lo.x / (hi.x - lo.x) * view.x, hi.y / (hi.y - lo.y) * view.y);
        if origin.y > 0.0 && origin.y < view.y {
            f.caption_px(Vec2::new(view.x - 14.0, origin.y - 9.0), "$x$", 13.0, pal::DIM, Align2::RIGHT_BOTTOM);
        }
        if origin.x > 0.0 && origin.x < view.x {
            f.caption_px(Vec2::new(origin.x - 12.0, 10.0), "$y$", 13.0, pal::DIM, Align2::RIGHT_TOP);
        }
        f.axes2();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The captions of the "things to try", held to account.
    #[test]
    fn the_experiments_show_what_their_captions_say() {
        let mut g = Graph::create();
        // a travelling wave: crests move 1/a a second, and are 2π/a apart
        g.set_up(0);
        let crest = |g: &Graph, a: f32, t: f32| {
            // (the crest between 0 and one wavelength, found on a fine grid)
            let xs = (0..=20_000).map(|i| i as f32 * TAU / a / 20_000.0);
            xs.max_by(|&p, &q| g.rows[0].expr.eval(&[p, t, a, g.b]).total_cmp(&g.rows[0].expr.eval(&[q, t, a, g.b]))).unwrap()
        };
        assert!(g.rows.len() == 1 && g.rows[0].expr.uses(1) && g.animated());
        assert!((crest(&g, g.a, 1.0) - crest(&g, g.a, 0.0) - 0.5).abs() < 1e-3);
        assert!((crest(&g, 1.0, 1.0) - crest(&g, 1.0, 0.0) - 1.0).abs() < 1e-3);
        // the bell: 0.61 of its height at x = ±b, whatever b
        g.set_up(1);
        for b in [0.3, 1.0, 2.5, -1.5] {
            for x in [b, -b] {
                assert!((g.rows[0].expr.eval(&[x, 0.0, g.a, b]) - 0.6065).abs() < 1e-4);
            }
        }
        assert!(!g.animated());
        // the cubic: the tangent is flat at ±1.83, and nowhere once b is below 0
        g.set_up(2);
        assert!(g.show_slope);
        let turn = (10.0 * g.b / 3.0).sqrt();
        assert!((turn - 1.83).abs() < 0.005);
        for x in [turn, -turn] {
            assert!(g.rows[0].expr.grad2(&[x, 0.0, g.a, g.b], 0, 9).1.abs() < 1e-5);
        }
        assert!((-500..=500).all(|i| g.rows[0].expr.grad2(&[i as f32 * 0.01, 0.0, g.a, -0.5], 0, 9).1 > 0.4));
    }

    /// The label is the formula, typeset, or as typed when it is long (cut short) or
    /// does not compile.
    #[test]
    fn a_curve_is_named_by_its_formula() {
        assert_eq!(label(" cos(x) ", true), r"$y = \cos(x)$");
        assert_eq!(label(" cos(x) ", false), "y = cos(x)");
        let long = label("sin(a x - t) / (1 + 0.1 x^2) + exp(-x^2)", true);
        assert!(long.chars().count() == 32 && long.ends_with('…'));
    }

    /// Typed formulas written as markup: as the program reads them, or not at all.
    #[test]
    fn typed_formulas_are_written_as_they_are_read() {
        for (typed, written) in [
            // the scene's own
            ("sin(a x - t) / (1 + 0.1 x^2)", r"\sin(a x - t ) / (1 + 0.1 x ^{2} ) "),
            ("exp(-x^2 / (2 b^2))", r"\exp(- x ^{2} / (2 b ^{2} ) ) "),
            ("0.1 x^3 - b x", r"0.1 x ^{3} - b x"),
            ("sin(a x - t)", r"\sin(a x - t )"),
            // names side by side, taken apart as the program takes them
            ("2ax + tb", r"2 a x + t b"),
            ("xsin(x)", r"x \sin(x )"),
            ("pix", r"\pi x"),
            // powers: a sign, one thing, right to left; brackets dropped as a whole power
            ("x^-1", r"x ^{-1 }"),
            ("x^(2a+1)", r"x ^{2 a + 1 }"),
            ("2^3^2", r"2 ^{3 ^{2 } }"),
            ("x^-(a+1)", r"x^{-(a + 1)}"),
            ("x^(a+1)^2", r"x^{(a + 1)^{2}}"),
            ("(a+1)^2", r"(a + 1)^{2}"),
            ("e^sin(x)", r"e ^{\sin(x ) }"),
            ("x^2a", r"x ^{2 } a"),
            // functions without a name of their own in print
            ("sqrt(abs(x))", r"\sqrt{\left|x \right|}"),
            ("relu(x) * 2", r"\operatorname{relu}(x ) \cdot 2"),
            ("asin(x)", r"\arcsin(x )"),
            ("π x", r"\pi x"),
        ] {
            // (spaces aside, which count for nothing in a formula)
            let bare = |m: &str| m.split_whitespace().collect::<String>();
            let made = markup(typed, GRAPH_VARS).unwrap_or_default();
            assert_eq!(bare(&made), bare(written), "{typed}");
            // and markup that sets as it should: no word read as a product of letters
            let found = crate::typeset::inspect(&format!("${made}$"));
            assert!(found.error.is_none() && found.words.is_empty(), "{typed}: {found:?}");
        }
        // what it is not sure of: as typed
        for typed in ["2e-1 x", "1.5E3", "x_1", "foo(x)", "sin(x", "x)", "q x", "x²", ""] {
            assert_eq!(markup(typed, GRAPH_VARS), None, "{typed}");
        }
        // every label of the things to try typesets
        for trial in &TRIALS {
            for curve in trial.curves {
                assert!(label(curve, true).starts_with('$'), "{curve}");
            }
        }
    }
}

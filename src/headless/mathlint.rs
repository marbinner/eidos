//! `eidos mathlint [scene...]`: what is left to typeset.
//!
//! Every text a scene shows is collected: its title and blurb, its panel as it opens
//! and after each of its "things to try" (pressed one after another), and the labels
//! on its picture meanwhile; and the glossary's entries, as a scene of their own
//! (`glossary`). Each is read for
//!
//! - (a) markup that does not parse: a lone `$`, an unclosed brace, an unknown
//!   command. Such a text is drawn as it is written, dollars and all;
//! - (b) mathematics left outside dollars: `^`, `_`, `√`, superscript and subscript
//!   characters, Greek letters, `=`, `≤`, `×`, `·` between symbols, `[[ … ]]`, a
//!   number glued to a letter (`2x`), a single letter standing alone (a variable:
//!   not `a` or `I`), a function applied (`f(x)`), a point (`(1, 0)`), `|x|`, `x*`,
//!   `1/2`, î;
//! - (c) inside dollars, runs of four or more italic letters: a word set as if it
//!   were a product of variables (`$softmax$`; it wants `\operatorname{softmax}`, or
//!   `\text`).
//!
//! Texts that differ only in their numbers are taken as one. (b) has false alarms by
//! nature: a text that has been looked at and is right as it is can be marked as
//! checked in `mathlint-checked.txt` beside this file. The command fails if (a) is
//! found anywhere, and prints how many of (b) and (c) each scene has.

use super::{input, known_scene, Res, Stage};
use crate::typeset::{self, Piece};

/// The window the scenes are shown in (a desktop one, with the panels beside the picture).
const WINDOW: [f32; 2] = [1100.0, 700.0];
/// Frames as the scene opens, and after each of its "things to try".
const FRAMES: u32 = 60;

/// Texts marked as checked: `scene: part of the text` (or `*: …` for every scene).
const CHECKED: &str = include_str!("mathlint-checked.txt");

/// What a text was found to have.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum Kind {
    /// (a) markup that does not parse
    Broken,
    /// (b) mathematics outside dollars
    Outside,
    /// (c) a word set in italic inside them
    Italic,
}

impl Kind {
    fn letter(self) -> char {
        match self {
            Kind::Broken => 'a',
            Kind::Outside => 'b',
            Kind::Italic => 'c',
        }
    }
}

/// One finding: what kind, in which text, and what was seen.
#[derive(Debug)]
pub struct Finding {
    pub kind: Kind,
    pub text: String,
    pub what: Vec<String>,
}

/// Every text `scene` shows, as it opens and after each of its things to try (texts
/// that differ only in their numbers once).
pub fn texts(scene: &str) -> Res<Vec<String>> {
    crate::clock::set_reproducible(true);
    crate::scenes::start_afresh();
    let mut stage = Stage::showing(scene)?;
    stage.app.set_dry(true);
    let mut texts = Texts::default();
    if let Some(def) = crate::scenes::registry().into_iter().find(|d| d.id == scene) {
        texts.add(def.title);
        texts.add(def.blurb);
    }
    typeset::collect(true);
    let (mut frame_no, mut press, mut experiments) = (0u64, None, 0usize);
    loop {
        for i in 0..FRAMES {
            crate::ui::collect_panel(true);
            if let (2, Some(k)) = (i, press) {
                crate::ui::press_experiment(k);
            }
            let out = stage.frame(input(WINDOW, frame_no, Vec::new()));
            stage.done(&out.textures_delta);
            frame_no += 1;
            experiments = experiments.max(crate::ui::collect_panel(false).experiments);
            for text in typeset::collect(true) {
                texts.add(&text);
            }
            for text in stage.app.drawn_labels() {
                texts.add(text);
            }
        }
        let next = press.map_or(0, |k| k + 1);
        if next >= experiments {
            break;
        }
        press = Some(next);
    }
    typeset::collect(false);
    drop(stage);
    crate::scenes::start_afresh();
    Ok(texts.all)
}

/// Texts gathered once each, texts that differ only in their numbers as one.
#[derive(Default)]
struct Texts {
    all: Vec<String>,
    shapes: std::collections::HashSet<String>,
}

impl Texts {
    fn add(&mut self, text: &str) {
        if text.trim().is_empty() {
            return;
        }
        let shape: String = text.chars().map(|c| if c.is_ascii_digit() { '0' } else { c }).collect();
        if self.shapes.insert(shape) {
            self.all.push(text.to_string());
        }
    }
}

/// The glossary's texts.
pub fn glossary_texts() -> Vec<String> {
    let mut texts = Texts::default();
    for (_, entries) in crate::glossary::SECTIONS {
        for entry in *entries {
            texts.add(entry.term);
            texts.add(entry.meaning);
        }
    }
    texts.all
}

/// Is `text` marked as checked for `scene`? (`scene: part of it`, or `scene: =all of it`.)
fn checked(scene: &str, text: &str) -> bool {
    CHECKED.lines().map(str::trim).filter(|l| !l.is_empty() && !l.starts_with('#')).any(|line| {
        let (who, part) = line.split_once(':').unwrap_or(("", line));
        let part = part.trim();
        let matches = match part.strip_prefix('=') {
            Some(whole) => text.trim() == whole,
            None => text.contains(part),
        };
        (who.trim() == scene || who.trim() == "*") && matches
    })
}

/// What is wrong with one text of `scene`.
pub fn lint(scene: &str, text: &str) -> Vec<Finding> {
    let found = typeset::inspect(text);
    if let Some(error) = found.error {
        return vec![Finding { kind: Kind::Broken, text: text.to_string(), what: vec![error] }];
    }
    if checked(scene, text) {
        return Vec::new();
    }
    let mut out = Vec::new();
    let mut outside = Vec::new();
    match typeset::pieces(text) {
        Ok(parts) => {
            for part in parts {
                if let Piece::Text(words) = part {
                    outside.extend(mathematics_in(&words));
                }
            }
        }
        Err(_) => outside.extend(mathematics_in(text)),
    }
    outside.dedup();
    if !outside.is_empty() {
        out.push(Finding { kind: Kind::Outside, text: text.to_string(), what: outside });
    }
    if !found.words.is_empty() {
        // (a function that has a command of its own: say so)
        let what = found.words.into_iter().map(|w| if typeset::is_function(&w) { format!("{w} (\\{w})") } else { w }).collect();
        out.push(Finding { kind: Kind::Italic, text: text.to_string(), what });
    }
    out
}

/// The mathematics in a piece of plain text: each piece of it found, with a little of
/// what surrounds it.
pub fn mathematics_in(text: &str) -> Vec<String> {
    let chars: Vec<char> = text.chars().collect();
    let n = chars.len();
    let at = |i: usize| chars.get(i).copied().unwrap_or(' ');
    let before = |i: usize| if i == 0 { ' ' } else { chars[i - 1] };
    let letter = |c: char| c.is_alphabetic();
    let mut hits: Vec<(usize, usize)> = Vec::new();
    // the word (run of letters) starting at i, and where it ends
    let word_at = |i: usize| {
        let mut j = i;
        while j < n && chars[j].is_ascii_alphabetic() {
            j += 1;
        }
        j
    };
    let mut i = 0;
    while i < n {
        let c = chars[i];
        let next = at(i + 1);
        let greek = ('Α'..='Ω').contains(&c) || ('α'..='ω').contains(&c) || "ϑϕϖϱϵ".contains(c);
        let operator = "^√=≤≥≠≈≡∝∈∉⊂⊆⊃⊇×±∓∑∏∫∂∇∞∀∃⊥∥‖⟨⟩⌊⌋⌈⌉<>⇒⇔↦→′".contains(c);
        let script = crate::typeset::script_char(c);
        let accent = matches!(c, 'î' | 'ĵ') || crate::typeset::combining_char(c);
        if greek || operator || script || accent {
            hits.push((i, i + 1));
        } else if c == '_' && (next.is_alphanumeric() || next == '{' || next == '(') {
            hits.push((i, i + 2));
        } else if c == '·' {
            // between two symbols (a · b), not as a separator between phrases
            let left = chars[..i].iter().rev().skip_while(|c| **c == ' ').take_while(|c| c.is_alphanumeric()).count();
            let right = chars[i + 1..].iter().skip_while(|c| **c == ' ').take_while(|c| c.is_alphanumeric()).count();
            if (1..=2).contains(&left) && (1..=2).contains(&right) {
                hits.push((i, i + 1));
            }
        } else if c == '[' && next == '[' {
            hits.push((i, i + 2));
        } else if (c == '*' && (before(i).is_alphanumeric() || next.is_alphanumeric())) || (c == '|' && next != ' ' && chars[i + 1..].contains(&'|')) {
            // x*, and |x|
            hits.push((i, i + 1));
        } else if c.is_ascii_digit() && next.is_ascii_alphabetic() {
            // a number glued to a letter (2x, 12k), but not 3D, 1st, 10th
            let start = i - chars[..i].iter().rev().take_while(|c| c.is_ascii_digit() || **c == '.').count();
            let end = word_at(i + 1);
            if end == i + 2 && next != 'D' && !before(start).is_alphanumeric() {
                hits.push((start, end));
            }
        } else if c == '/' && before(i).is_alphanumeric() && (next.is_alphanumeric() || next == '(') {
            // a ratio of numbers or single letters (1/2, x/2), not and/or
            let left = chars[..i].iter().rev().take_while(|c| c.is_alphanumeric() || **c == '.').count();
            let right = chars[i + 1..].iter().take_while(|c| c.is_alphanumeric() || **c == '.').count();
            let short = |k: usize, s: &[char]| k == 1 || s.iter().all(|c| c.is_ascii_digit() || *c == '.');
            if short(left, &chars[i - left..i]) && (next == '(' || short(right, &chars[i + 1..i + 1 + right])) {
                hits.push((i - left, i + 1 + right));
            }
        } else if c == '(' && !(i == 0 && text.trim_end().ends_with(')') && !text[1..].contains('(')) {
            // a point or a tuple: (1, 0), (x, y); but not a read-out's value that is
            // nothing but numbers in brackets
            if let Some(close) = chars[i + 1..].iter().take(40).position(|c| *c == ')') {
                let inside: String = chars[i + 1..i + 1 + close].iter().collect();
                let items: Vec<&str> = inside.split(',').map(str::trim).collect();
                let number = |s: &str| {
                    let s = s.trim_start_matches(['-', '−']);
                    !s.is_empty() && s.chars().all(|c| c.is_ascii_digit() || c == '.')
                };
                let single = |s: &str| s.chars().count() == 1 && s.chars().all(|c| c.is_alphabetic());
                if items.len() >= 2 && items.iter().all(|s| number(s) || single(s)) {
                    hits.push((i, i + 2 + close));
                }
            }
        }
        if c.is_ascii_alphabetic() && !before(i).is_alphanumeric() && before(i) != '\'' && before(i) != '’' {
            let end = word_at(i);
            let word: String = chars[i..end].iter().collect();
            let after = at(end);
            // a function applied: f(x), sin(x), tanh(6x)
            const FUNCTIONS: [&str; 18] =
                ["sin", "cos", "tan", "exp", "log", "ln", "tanh", "sinh", "cosh", "sigmoid", "sqrt", "abs", "det", "tr", "max", "min", "softmax", "relu"];
            if after == '(' && (word.len() == 1 || FUNCTIONS.contains(&word.to_lowercase().as_str())) {
                hits.push((i, end + 1));
            } else if word.eq_ignore_ascii_case("sqrt") {
                hits.push((i, end));
            } else if word.len() == 1
                && !after.is_alphanumeric()
                && after != '\''
                && after != '’'
                && !(after == '.' && letter(at(end + 1)))
                && !(before(i) == '.' && i >= 2 && letter(chars[i - 2]))
            {
                // a letter standing alone is a variable; but not "a", "I", or "A" starting a sentence
                let first = chars[..i].iter().rev().find(|c| **c != ' ').is_none_or(|c| ".!?:(".contains(*c));
                if !(c == 'a' || c == 'I' || (c == 'A' && first)) {
                    hits.push((i, end));
                }
            }
            i = end.max(i + 1);
            continue;
        }
        i += 1;
    }
    let mut out: Vec<String> = Vec::new();
    for (from, to) in hits {
        let piece: String = chars[from..to.min(n)].iter().collect();
        if !out.contains(&piece) {
            out.push(piece);
        }
    }
    out
}

/// `eidos mathlint [scene...]`: print what was found; an error if anything does not parse.
/// `eidos texts <scene>...`: every text the scene shows, one to a line, in the order
/// they first appear (for reading a scene's words without reading its code).
pub fn print_texts(args: &[String]) -> Res<()> {
    if args.is_empty() {
        return Err("usage: eidos texts <scene>...".into());
    }
    for id in args {
        known_scene(id)?;
        println!("== {id}");
        for text in texts(id)? {
            println!("{}", text.replace('\n', " ⏎ "));
        }
    }
    Ok(())
}

pub fn mathlint(args: &[String]) -> Res<()> {
    for id in args.iter().filter(|a| *a != "glossary") {
        known_scene(id)?;
    }
    let mut scenes: Vec<String> = if args.is_empty() { crate::scenes::registry().iter().map(|d| d.id.to_string()).collect() } else { args.to_vec() };
    if args.is_empty() {
        scenes.push("glossary".to_string());
    }
    println!("What is left to typeset: (a) markup that does not parse, (b) mathematics outside dollars,");
    println!("(c) words set in italic inside them (see src/headless/mathlint.rs).\n");
    let mut table = Vec::new();
    for scene in &scenes {
        let texts = if scene == "glossary" { glossary_texts() } else { texts(scene)? };
        let findings: Vec<Finding> = texts.iter().flat_map(|t| lint(scene, t)).collect();
        for f in &findings {
            let short: String = f.text.chars().take(110).collect();
            let more = if f.text.chars().count() > 110 { "…" } else { "" };
            println!("{scene:<15} ({})  {}\n{:<20}«{short}{more}»", f.kind.letter(), f.what.join("   "), "");
        }
        let count = |kind: Kind| findings.iter().filter(|f| f.kind == kind).count();
        table.push((scene.clone(), texts.len(), count(Kind::Broken), count(Kind::Outside), count(Kind::Italic)));
    }
    println!("\n{:<15} {:>6} {:>5} {:>5} {:>5}", "scene", "texts", "(a)", "(b)", "(c)");
    for (scene, texts, a, b, c) in &table {
        println!("{scene:<15} {texts:>6} {a:>5} {b:>5} {c:>5}");
    }
    let sum = |k: usize| table.iter().map(|row| [row.1, row.2, row.3, row.4][k]).sum::<usize>();
    println!("{:<15} {:>6} {:>5} {:>5} {:>5}", "total", sum(0), sum(1), sum(2), sum(3));
    if sum(1) > 0 {
        return Err(format!("{} texts have markup that does not parse", sum(1)).into());
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// What the lint finds outside dollars, and what it leaves alone.
    #[test]
    fn mathematics_outside_dollars_is_found() {
        let finds = |t: &str, all: &[&str]| {
            let found = mathematics_in(t);
            assert!(all.iter().all(|a| found.iter().any(|f| f == a)), "{t}: {found:?}, not all of {all:?}");
        };
        let nothing = |t: &str| assert_eq!(mathematics_in(t), Vec::<String>::new(), "{t}");
        finds("the arrow î = (1, 0) lands", &["î", "=", "(1, 0)"]);
        finds("x₁² + y", &["₁", "²", "y"]);
        finds("a 2x2 matrix, 2x faster, 12k", &["2x", "12k"]);
        finds("the matrix A moves", &["A"]);
        finds("sin(x) and f(x) and tanh(6x)", &["sin(", "f(", "tanh(", "6x"]);
        finds("and/or 1/2 x/2", &["1/2", "x/2"]);
        finds("|e| and x* and [[0, 1], [1, 0]]", &["|", "*", "[["]);
        finds("a · b", &["·"]);
        finds("λ and σ² and x^2 and x_1 and √n and sqrt(n)", &["λ", "σ", "²", "^", "_1", "√", "sqrt("]);
        nothing("in 3D, the 1st and 10th, 3 million people, after 30 steps");
        nothing("e.g. a curve, i.e. the line; I think so. A sentence");
        nothing("Calculus · 2 of 5, and/or, the dot's colour");
    }

    /// Markup that does not parse is (a); words in italic are (c); a text marked as
    /// checked is not looked at for (b) and (c).
    #[test]
    fn each_kind_is_told_apart() {
        let kinds = |t: &str| lint("nowhere", t).iter().map(|f| f.kind).collect::<Vec<_>>();
        assert_eq!(kinds("all $x^{2$ wrong"), [Kind::Broken]);
        assert_eq!(kinds("a lone $ sign"), [Kind::Broken]);
        assert_eq!(kinds("$\\mathrm{loss} = softmax(z)$"), [Kind::Italic]);
        assert_eq!(kinds("$a \\cdot b$ is the dot product"), Vec::<Kind>::new());
        assert_eq!(kinds("where x = 2"), [Kind::Outside]);
    }
}

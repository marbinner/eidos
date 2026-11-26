//! Typeset mathematics: text with formulas between dollar signs, as in LaTeX, set in
//! the KaTeX fonts (Computer Modern) the way TeX sets them.
//!
//! `"Two arrows, $a$ and $b$, and their dot product $a \cdot b = |a|\,|b| \cos\theta$."`
//!
//! Text stays a plain string everywhere: every function that puts words on the screen
//! (`ui::note`, `f.text`, ...) takes it, and text without a dollar sign is drawn exactly
//! as it always was. What may go between the dollars is in README.md ("Writing
//! mathematics"); `eidos mathlint` finds what does not parse, and mathematics left
//! outside them.
//!
//! The parts: [`parse`] reads a formula into a tree (LaTeX, and the Unicode the text was
//! written in before: x₁², Aᵀ, ≤, √); [`layout`] lays it out as TeX does (appendix G of
//! The TeXbook, by way of KaTeX), in boxes measured from the fonts' own metrics
//! ([`metrics`], written from the font files by `tools/math-metrics.py`); [`paint`]
//! turns a paragraph of text and formulas into one egui `Galley`, wrapped to a width
//! and kept from frame to frame.

mod ink;
mod layout;
mod metrics;
mod paint;
mod parse;
mod symbols;

pub use paint::{label, measure, paragraph, take_built, Mode, Paragraph, Span};
pub use symbols::{is_function, Face};

use std::borrow::Cow;

/// How much larger than the text around it a formula is set: a compromise between two
/// sizes that each look wrong. Matched by x-height (the interface font's 0.517 em to
/// Computer Modern's 0.431, 1.2, which is what KaTeX does), a variable stands as tall as
/// the letters beside it, but the formula's digits and capitals stand a ninth taller
/// than the words' (Computer Modern's are short for its x-height), and "slope 1.36 ±
/// 0.02" reads as words and then larger numbers. Matched by the digits (the interface
/// font's 0.693 em to 0.666, 1.04), the numbers agree and the variables look a size too
/// small. At 1.1 the digits are 6 % taller than the words' and the x-height 8 %
/// shorter, and neither is seen as another size (chosen on the same screens against
/// 1.2, 1.12 and 1.06).
pub const TEXT_SCALE: f32 = 1.1;
/// The same beside the monospaced font of the read-outs, by the height of the digits
/// alone (Hack's 0.729 em to Computer Modern's 0.666), since read-outs are numbers: a
/// formula's digits there are exactly as tall as the read-out's own.
pub const MONO_SCALE: f32 = (1493.0 / 2048.0) / 0.666;

/// Does the text have any mathematics in it (or anything that must be read as markup)?
/// Text without a dollar sign is drawn as it always was.
#[inline]
pub fn has_math(text: &str) -> bool {
    text.as_bytes().contains(&b'$')
}

/// A part of a marked-up text: words, or a formula (what stood between the dollars).
#[derive(Clone, Debug, PartialEq)]
pub enum Piece<'a> {
    Text(Cow<'a, str>),
    Math(&'a str),
}

/// The text cut into words and formulas. `\$` outside a formula is a dollar sign; a
/// formula that is not closed, or is empty, is an error.
pub fn pieces(text: &str) -> Result<Vec<Piece<'_>>, String> {
    let bytes = text.as_bytes();
    let mut out = Vec::new();
    fn words<'a>(out: &mut Vec<Piece<'a>>, words: &'a str, escapes: bool) {
        if !words.is_empty() {
            out.push(Piece::Text(if escapes { Cow::Owned(words.replace("\\$", "$")) } else { Cow::Borrowed(words) }));
        }
    }
    // (dollars and backslashes are single bytes, never inside a longer character)
    let (mut start, mut escapes, mut i) = (0, false, 0);
    while i < bytes.len() {
        match bytes[i] {
            b'\\' if bytes.get(i + 1) == Some(&b'$') => {
                escapes = true;
                i += 2;
            }
            b'$' => {
                words(&mut out, &text[start..i], escapes);
                // the formula runs to the next dollar that is not part of a command
                let mut j = i + 1;
                let end = loop {
                    match bytes.get(j) {
                        None => return Err("a $ is not closed".into()),
                        Some(b'\\') => j += 2,
                        Some(b'$') => break j,
                        Some(_) => j += 1,
                    }
                };
                let formula = &text[i + 1..end];
                if formula.trim().is_empty() {
                    return Err("an empty formula ($$)".into());
                }
                out.push(Piece::Math(formula));
                (start, escapes, i) = (end + 1, false, end + 1);
            }
            _ => i += 1,
        }
    }
    words(&mut out, &text[start..], escapes);
    Ok(out)
}

/// What the lint and the scene check want to know of a text: whether its markup
/// parses, and the words set in italic as if they were products of variables.
#[derive(Debug, Default, PartialEq)]
pub struct Findings {
    pub error: Option<String>,
    pub words: Vec<String>,
}

/// Read every formula in `text`.
pub fn inspect(text: &str) -> Findings {
    let mut found = Findings::default();
    if !has_math(text) {
        return found;
    }
    match pieces(text) {
        Err(e) => found.error = Some(e),
        Ok(parts) => {
            for part in parts {
                if let Piece::Math(formula) = part {
                    match parse::parse(formula) {
                        Ok(parsed) => found.words.extend(parsed.words),
                        Err(e) => {
                            found.error = Some(format!("{e} in ${formula}$"));
                            return found;
                        }
                    }
                }
            }
        }
    }
    found
}

/// The text with its formulas written out in plain Unicode, near enough to read: for
/// search, for screen readers, and where a formula has to be cut short.
pub fn plain(text: &str) -> String {
    if !has_math(text) {
        return text.to_string();
    }
    let Ok(parts) = pieces(text) else { return text.to_string() };
    let mut out = String::new();
    for part in parts {
        match part {
            Piece::Text(words) => out.push_str(&words),
            Piece::Math(formula) => match parse::parse(formula) {
                Ok(parsed) => paint::plain_nodes(&parsed.nodes, &mut out),
                Err(_) => {
                    out.push('$');
                    out.push_str(formula);
                    out.push('$');
                }
            },
        }
    }
    out
}

/// A display's lines as its writer broke them: the formula cut at each `\\` that is not
/// inside braces or an environment (where it is a matrix's next row). Lines with nothing
/// in them are left out.
pub fn display_lines(formula: &str) -> Vec<&str> {
    let bytes = formula.as_bytes();
    let (mut braces, mut environments, mut start, mut i) = (0i32, 0i32, 0, 0);
    let mut lines = Vec::new();
    // (braces, backslashes and the letters of `begin` and `end` are single bytes)
    while i < bytes.len() {
        match bytes[i] {
            b'{' => braces += 1,
            b'}' => braces -= 1,
            b'\\' => {
                let rest = &bytes[i + 1..];
                if rest.starts_with(b"begin") {
                    environments += 1;
                } else if rest.starts_with(b"end") {
                    environments -= 1;
                } else if rest.first() == Some(&b'\\') && braces == 0 && environments == 0 {
                    lines.push(&formula[start..i]);
                    start = i + 2;
                }
                // (the character after a backslash is the command's: `\{` opens nothing)
                i += 2;
                continue;
            }
            _ => {}
        }
        i += 1;
    }
    lines.push(formula.get(start..).unwrap_or(""));
    lines.retain(|line| !line.trim().is_empty());
    lines
}

/// What a search looks in for `text`: what the visitor sees of it (its formulas
/// written out as they are drawn, λ for `\lambda`, and never the markup's commands and
/// braces), lower-cased, with the name of every Greek letter in it after it, so that
/// "lambda" finds λ as "λ" does.
pub fn searchable(text: &str) -> String {
    let shown = plain(text);
    let mut out = shown.to_lowercase();
    let mut named: Vec<char> = Vec::new();
    for c in shown.chars() {
        if named.contains(&c) {
            continue;
        }
        if let Some(name) = greek_name(c) {
            named.push(c);
            out.push(' ');
            out.push_str(&name);
        }
    }
    out
}

/// The name of a Greek letter, as a visitor would type it ("lambda" for λ and Λ, "phi"
/// for φ and ϕ).
fn greek_name(c: char) -> Option<String> {
    const NAMES: [&str; 41] = [
        "alpha",
        "beta",
        "gamma",
        "delta",
        "epsilon",
        "varepsilon",
        "zeta",
        "eta",
        "theta",
        "vartheta",
        "iota",
        "kappa",
        "lambda",
        "mu",
        "nu",
        "xi",
        "omicron",
        "pi",
        "varpi",
        "rho",
        "varrho",
        "sigma",
        "varsigma",
        "tau",
        "upsilon",
        "phi",
        "varphi",
        "chi",
        "psi",
        "omega",
        "Gamma",
        "Delta",
        "Theta",
        "Lambda",
        "Xi",
        "Pi",
        "Sigma",
        "Upsilon",
        "Phi",
        "Psi",
        "Omega",
    ];
    if !('\u{370}'..='\u{3FF}').contains(&c) {
        return None;
    }
    NAMES
        .iter()
        .find(|name| matches!(symbols::command(name), Some(symbols::Cmd::Sym(ch, ..)) if ch == c))
        .map(|name| name.trim_start_matches("var").to_lowercase())
}

/// A rough width of `text` at `size` with no fonts at hand (as in unit tests, or
/// deciding a layout before anything is drawn): formulas exactly, from the fonts'
/// metrics; words by their number of characters, as `kit::text_width` counts them.
pub fn estimate_width(text: &str, size: f32, per_char: f32) -> f32 {
    let Ok(parts) = pieces(text) else { return text.chars().count() as f32 * size * per_char };
    let mut width = 0.0;
    for part in parts {
        match part {
            Piece::Text(words) => width += words.chars().count() as f32 * size * per_char,
            Piece::Math(formula) => match parse::parse(formula) {
                Ok(parsed) => {
                    let chunks = layout::layout(&parsed.nodes, false, &layout::estimate(per_char, TEXT_SCALE));
                    let em: f32 = chunks.iter().enumerate().map(|(i, c)| c.bx.width + if i + 1 < chunks.len() { c.glue } else { 0.0 }).sum();
                    width += em * size * TEXT_SCALE;
                }
                Err(_) => width += (formula.chars().count() + 2) as f32 * size * per_char,
            },
        }
    }
    width
}

/// Is `c` a superscript or subscript character (², ₁, ᵀ)?
pub fn script_char(c: char) -> bool {
    symbols::superscript(c).is_some() || symbols::subscript(c).is_some()
}

/// Is `c` an accent that combines with the letter before it (x̂, v⃗)?
pub fn combining_char(c: char) -> bool {
    symbols::combining(c).is_some()
}

thread_local! {
    static SEEN: std::cell::RefCell<Option<Vec<String>>> = const { std::cell::RefCell::new(None) };
}

/// Note a text that is being put on the screen, if anyone is collecting (the lint, and
/// the scene check, which want every text a scene shows).
pub fn seen(text: &str) {
    SEEN.with(|seen| {
        if let Some(texts) = seen.borrow_mut().as_mut() {
            if !texts.iter().any(|t| t == text) {
                texts.push(text.to_string());
            }
        }
    });
}

/// Start (or stop) collecting the texts put on the screen, and return those collected
/// so far.
pub fn collect(on: bool) -> Vec<String> {
    SEEN.with(|seen| seen.replace(on.then(Vec::new)).unwrap_or_default())
}

#[cfg(test)]
mod tests;

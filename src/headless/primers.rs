//! `eidos primers`: what is wrong with a scene's primer (the mathematics behind it,
//! written out: see `crate::primer`).
//!
//! A primer is read by someone who does not know the subject yet, so the check is
//! strict about what it can judge: the three parts in their order, a length that can be
//! read in a few minutes, formulas that set, mathematics inside dollars and words
//! outside them (the same reading `eidos mathlint` gives a scene's texts), related scenes
//! that exist, and nothing of Markdown beyond the little the primers use. Whether it
//! is true and clear, only a reader can say.

use super::mathlint::Kind;
use super::{known_scene, Res};
use crate::primer::{self, Block};

/// The parts every primer has, in order.
const PARTS: [&str; 3] = ["The idea", "The mathematics", "In the picture"];
/// The fewest and the most words (formulas on lines of their own not counted).
const WORDS: std::ops::RangeInclusive<usize> = 280..=700;

/// Everything found wrong with the primer `text` of scene `id` (nothing, if it is fine).
pub fn problems(id: &str, text: &str) -> Vec<String> {
    let mut found = Vec::new();
    if !crate::scenes::planned(id) {
        found.push(format!("\"{id}\" is not a scene"));
    }
    let blocks = primer::parse(text);
    if blocks.is_empty() {
        found.push("nothing is written".to_string());
        return found;
    }

    // the parts
    let headings: Vec<&str> = blocks.iter().filter_map(|b| if let Block::Heading(h) = b { Some(h.as_str()) } else { None }).collect();
    if headings != PARTS {
        found.push(format!("its parts should be {PARTS:?}, in that order; they are {headings:?}"));
    }
    if !matches!(blocks.first(), Some(Block::Heading(_))) {
        found.push("it should start with the heading \"# The idea\"".to_string());
    }
    let part_of = |k: usize| blocks[..=k].iter().rev().find_map(|b| if let Block::Heading(h) = b { Some(h.as_str()) } else { None }).unwrap_or("");
    let in_picture: Vec<&Block> = blocks.iter().enumerate().filter(|(k, _)| part_of(*k) == PARTS[2]).map(|(_, b)| b).collect();
    if headings.contains(&PARTS[2]) {
        if !in_picture.iter().any(|b| matches!(b, Block::List { .. })) {
            found.push("\"In the picture\" should have a list tying what is drawn to the mathematics".to_string());
        }
        if in_picture.iter().filter(|b| matches!(b, Block::Aside(_))).count() != 1 {
            found.push("\"In the picture\" should end with one line set apart (\"> \") saying what to try first".to_string());
        }
    }
    if !blocks.iter().any(|b| matches!(b, Block::Display(_))) {
        found.push("no formula on a line of its own (\"$$ … $$\"): the key formulas should stand out".to_string());
    }

    // the related scenes, last
    match blocks.last() {
        Some(Block::See(ids)) => {
            if ids.is_empty() || ids.len() > 4 {
                found.push(format!("\"see:\" should name one to four scenes, and names {}", ids.len()));
            }
            for (k, other) in ids.iter().enumerate() {
                if other == id {
                    found.push("\"see:\" names the scene itself".to_string());
                } else if !crate::scenes::planned(other) {
                    found.push(format!("\"see:\" names \"{other}\", which is not a scene"));
                }
                if ids[..k].contains(other) {
                    found.push(format!("\"see:\" names \"{other}\" twice"));
                }
            }
        }
        _ => found.push("its last line should be \"see: \" and one to four scene ids".to_string()),
    }
    if blocks.iter().filter(|b| matches!(b, Block::See(_))).count() > 1 {
        found.push("more than one \"see:\" line".to_string());
    }

    // the length
    let words = primer::word_count(&blocks);
    if !WORDS.contains(&words) {
        found.push(format!("{words} words: it should have {} to {}", WORDS.start(), WORDS.end()));
    }

    // every text: formulas that set, mathematics in dollars, words out of them, no other markup
    if text.lines().filter(|l| l.contains("$$")).map(|l| l.matches("$$").count()).sum::<usize>() % 2 == 1 {
        found.push("a \"$$\" is never closed".to_string());
    }
    let read = |found: &mut Vec<String>, what: &str, text: &str, formula: bool| {
        let shown: String = text.chars().take(60).collect();
        if formula {
            let reading = crate::typeset::inspect(&format!("${text}$"));
            if let Some(error) = reading.error {
                found.push(format!("{what} \"{shown}\": the mathematics does not set: {error}"));
            }
            if !reading.words.is_empty() {
                found.push(format!("{what} \"{shown}\": words inside a formula (put them in \\text{{…}}): {:?}", reading.words));
            }
            return;
        }
        // (the reading `eidos mathlint` gives every text of a scene)
        for finding in super::mathlint::lint("primer", text) {
            let kind = match finding.kind {
                Kind::Broken => "the mathematics does not set",
                Kind::Outside => "mathematics outside dollars",
                Kind::Italic => "words inside a formula (put them outside the dollars, or in \\text{…})",
            };
            found.push(format!("{what} \"{shown}\": {kind}: {:?}", finding.what));
        }
        for (mark, name) in [("**", "bold"), ("`", "code"), ("](", "a link"), ("__", "emphasis"), ("<br", "HTML")] {
            if text.contains(mark) {
                found.push(format!("{what} \"{shown}\": {name} ({mark}) is not part of the format"));
            }
        }
        if text.starts_with('|') || text.starts_with("### ") {
            found.push(format!("{what} \"{shown}\": tables and smaller headings are not part of the format"));
        }
    };
    for block in &blocks {
        match block {
            Block::Heading(t) => read(&mut found, "the heading", t, false),
            Block::Sub(t) => read(&mut found, "the heading", t, false),
            Block::Paragraph(t) => read(&mut found, "the paragraph", t, false),
            Block::Aside(t) => read(&mut found, "the line set apart", t, false),
            Block::List { items, .. } => items.iter().for_each(|t| read(&mut found, "the point", t, false)),
            Block::Display(t) => {
                if t.trim().is_empty() {
                    found.push("an empty formula (\"$$ $$\")".to_string());
                }
                for line in crate::typeset::display_lines(t) {
                    read(&mut found, "the formula", line, true);
                }
            }
            Block::See(_) => {}
        }
    }
    found
}

pub fn primers(args: &[String]) -> Res<()> {
    for id in args {
        known_scene(id)?;
    }
    let all: Vec<&str> = primer::IDS.to_vec();
    let asked: Vec<&str> = if args.is_empty() { all } else { args.iter().map(String::as_str).collect() };
    let (mut written, mut wrong) = (0, 0);
    for id in asked {
        let Some(text) = primer::of(id) else {
            if !args.is_empty() {
                println!("{id:<15} not written yet");
            }
            continue;
        };
        written += 1;
        let found = problems(id, text);
        let words = primer::word_count(&primer::parse(text));
        println!("{id:<15} {words:>4} words  {}", if found.is_empty() { "fine".to_string() } else { format!("{} problem(s)", found.len()) });
        for problem in &found {
            println!("    {problem}");
        }
        wrong += !found.is_empty() as usize;
    }
    println!("\n{written} primers read, {wrong} with problems; {} scenes have none yet", primer::IDS.iter().filter(|id| primer::of(id).is_none()).count());
    if wrong > 0 {
        return Err(format!("{wrong} primer(s) with problems").into());
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Every primer that has been written passes the check.
    #[test]
    fn every_written_primer_is_well_formed() {
        for id in primer::IDS {
            if let Some(text) = primer::of(id) {
                let found = problems(id, text);
                assert!(found.is_empty(), "{id}:\n  {}", found.join("\n  "));
            }
        }
    }

    #[test]
    fn the_check_finds_what_is_wrong() {
        let good = primer::of("vectors").expect("the model primer is written");
        assert!(problems("vectors", good).is_empty());
        let has = |text: &str, part: &str| problems("vectors", text).iter().any(|p| p.contains(part));
        assert!(has("", "nothing is written"));
        assert!(has(&good.replace("# In the picture", "# On the screen"), "its parts should be"));
        assert!(has(&good.replace("see: span, det, lsq", "see: span, nosuchscene"), "not a scene"));
        assert!(has(&good.replace("see: span, det, lsq", "see: vectors"), "the scene itself"));
        assert!(has(&good.replace("see: span, det, lsq", ""), "its last line"));
        assert!(has(&good.replace("$$ |a| = \\sqrt{a_1^2 + a_2^2 + a_3^2} $$", "$$ |a| = \\sqrt{a_1^2 + $$"), "does not set"));
        assert!(has(&good.replace("Multiply the arrows", "**Multiply** the arrows"), "bold"));
        assert!(has(&good.replace("> Try first", "Try first"), "one line set apart"));
        let short: String = good.lines().take(12).collect::<Vec<_>>().join("\n");
        assert!(has(&short, "words: it should have"));
    }
}

//! The mathematics behind each scene, written out: a short text a visitor can read
//! before the picture (its "primer"), shown over the application by a button at the top
//! of the scene's panel.
//!
//! One file a scene, `primers/<scene id>.md`, compiled in. A primer is plain text in a
//! small subset of Markdown with the application's own mathematics markup (`$…$`, see
//! `typeset`): headings (`# `, `## `), paragraphs, formulas on a line of their own
//! (`$$ … $$`), lists (`- `, `1. `), a line set apart (`> `), and a last line naming
//! related scenes (`see: a, b`). [`parse`] reads one into [`Block`]s and [`show`] sets
//! them; `eidos primers` (see `headless/primers.rs`) says what is wrong with one. A file
//! that is empty has no primer yet, and its scene shows no button.

use crate::gfx::color::pal;
use egui::RichText;

macro_rules! primers {
    ($($id:literal),* $(,)?) => {
        /// The scenes that have a primer file, written or still empty.
        pub const IDS: &[&str] = &[$($id),*];

        fn file(id: &str) -> Option<&'static str> {
            match id {
                $($id => Some(include_str!(concat!("primers/", $id, ".md"))),)*
                _ => None,
            }
        }
    };
}

primers!(
    "vectors",
    "span",
    "linmap2",
    "linmap3",
    "compose",
    "det",
    "basis",
    "rotations",
    "elim",
    "subspaces",
    "gram",
    "lsq",
    "overdet",
    "underdet",
    "condition",
    "eigen",
    "powers",
    "expm",
    "quadform",
    "svd",
    "lowrank",
    "pca",
    "dft",
    "highdim",
    "sensing",
    "packing",
    "special",
    "patterns",
    "symskew",
    "project",
    "commute",
    "factor",
    "spectral",
    "cholesky",
    "polar",
    "pinv",
    "schur",
    "jordan",
    "expskew",
    "expsum",
    "matfun",
    "modes",
    "calc1",
    "surface",
    "descent",
    "field",
    "fourier",
    "ftc",
    "jacobian",
    "chain",
    "hessian",
    "lagrange",
    "changevar",
    "autodiff",
    "backprop1",
    "backpropmat",
    "lossgrad",
    "actgrad",
    "gradflow",
    "normgrad",
    "residual",
    "attngrad",
    "tfgrad",
    "archlab",
    "attractor",
    "pendulum",
    "waves",
    "chance",
    "distributions",
    "moments",
    "lln",
    "clt",
    "gauss2",
    "describe",
    "estimators",
    "likelihood",
    "intervals",
    "bootstrap",
    "bayes",
    "mcmc",
    "testing",
    "pvalues",
    "twogroups",
    "anova",
    "chisquare",
    "correlation",
    "regress",
    "multireg",
    "logistic",
    "modelselect",
    "entropy",
    "confound",
    "dosee",
    "junctions",
    "adjust",
    "counterfactual",
    "direction",
    "lingam",
    "locscale",
    "discover",
    "pc",
    "hermite",
    "hermite_orth",
    "hermite_expand",
    "hermite_noise",
    "hermite_dirs",
    "mlp",
    "race",
    "particles",
    "signal",
    "fold",
    "fit1d",
    "landscape",
    "superposition",
    "sae",
    "probe",
    "grokking",
    "transformer",
    "lyapunov",
    "coherence",
    "belief",
    "graph",
    "parametric",
    "shingles",
    "switch",
);

/// The primer of scene `id`, if one has been written.
pub fn of(id: &str) -> Option<&'static str> {
    #[cfg(not(target_arch = "wasm32"))]
    if let Some(text) = from_disk(id) {
        return Some(text).filter(|text| !text.trim().is_empty());
    }
    file(id).filter(|text| !text.trim().is_empty())
}

/// For whoever is writing primers: with `EIDOS_PRIMERS` set to a directory, a scene's
/// primer is read from `<directory>/<scene id>.md` if that file is there, in place of
/// the one compiled in, so that a primer can be checked (`eidos primers`) and looked at
/// (`eidos shot … --primer`) without building. Read once a run.
#[cfg(not(target_arch = "wasm32"))]
fn from_disk(id: &str) -> Option<&'static str> {
    use std::collections::HashMap;
    use std::sync::{Mutex, OnceLock};
    static READ: OnceLock<Mutex<HashMap<String, Option<&'static str>>>> = OnceLock::new();
    let directory = std::env::var_os("EIDOS_PRIMERS")?;
    file(id)?;
    let mut read = READ.get_or_init(Default::default).lock().ok()?;
    *read.entry(id.to_string()).or_insert_with(|| {
        let text = std::fs::read_to_string(std::path::Path::new(&directory).join(format!("{id}.md"))).ok()?;
        Some(&*Box::leak(text.into_boxed_str()))
    })
}

/// One piece of a primer.
#[derive(Clone, Debug, PartialEq)]
pub enum Block {
    /// A part's heading (`# `).
    Heading(String),
    /// A smaller heading inside a part (`## `).
    Sub(String),
    Paragraph(String),
    /// A formula on a line of its own, without its dollars.
    Display(String),
    List {
        numbered: bool,
        items: Vec<String>,
    },
    /// A line set apart (`> `).
    Aside(String),
    /// The related scenes named on the last line.
    See(Vec<String>),
}

/// A primer's text as blocks. Anything it does not know is a paragraph: nothing is
/// lost, and `eidos primers` flags what should not be there.
pub fn parse(text: &str) -> Vec<Block> {
    let mut blocks: Vec<Block> = Vec::new();
    // what is being gathered: a paragraph's lines, or a formula's
    let mut paragraph: Vec<&str> = Vec::new();
    let mut formula: Option<Vec<&str>> = None;
    fn close(paragraph: &mut Vec<&str>, blocks: &mut Vec<Block>) {
        if !paragraph.is_empty() {
            blocks.push(Block::Paragraph(paragraph.join(" ")));
            paragraph.clear();
        }
    }
    for raw in text.lines() {
        let line = raw.trim_end();
        if let Some(lines) = &mut formula {
            match line.trim().strip_suffix("$$") {
                Some(last) => {
                    lines.push(last);
                    blocks.push(Block::Display(lines.join(" ").trim().to_string()));
                    formula = None;
                }
                None => lines.push(line.trim()),
            }
            continue;
        }
        let trimmed = line.trim_start();
        let indented = line.len() - trimmed.len() >= 2;
        let in_list = matches!(blocks.last(), Some(Block::List { .. })) && paragraph.is_empty();
        if trimmed.is_empty() {
            close(&mut paragraph, &mut blocks);
        } else if let Some(rest) = trimmed.strip_prefix("$$") {
            close(&mut paragraph, &mut blocks);
            match rest.strip_suffix("$$") {
                Some(inside) => blocks.push(Block::Display(inside.trim().to_string())),
                None => formula = Some(vec![rest.trim()]),
            }
        } else if let Some(rest) = trimmed.strip_prefix("## ") {
            close(&mut paragraph, &mut blocks);
            blocks.push(Block::Sub(rest.trim().to_string()));
        } else if let Some(rest) = trimmed.strip_prefix("# ") {
            close(&mut paragraph, &mut blocks);
            blocks.push(Block::Heading(rest.trim().to_string()));
        } else if let Some(rest) = trimmed.strip_prefix("> ") {
            close(&mut paragraph, &mut blocks);
            match blocks.last_mut() {
                Some(Block::Aside(words)) => {
                    words.push(' ');
                    words.push_str(rest.trim());
                }
                _ => blocks.push(Block::Aside(rest.trim().to_string())),
            }
        } else if let Some(rest) = trimmed.strip_prefix("see:") {
            close(&mut paragraph, &mut blocks);
            blocks.push(Block::See(rest.split(',').map(|id| id.trim().to_string()).filter(|id| !id.is_empty()).collect()));
        } else if indented && in_list {
            // an indented line goes on with the list's last point
            if let Some(Block::List { items, .. }) = blocks.last_mut() {
                if let Some(last) = items.last_mut() {
                    last.push(' ');
                    last.push_str(trimmed);
                }
            }
        } else if let Some((numbered, rest)) = item(trimmed) {
            close(&mut paragraph, &mut blocks);
            match blocks.last_mut() {
                Some(Block::List { numbered: n, items }) if *n == numbered => items.push(rest.to_string()),
                _ => blocks.push(Block::List { numbered, items: vec![rest.to_string()] }),
            }
        } else {
            paragraph.push(trimmed);
        }
    }
    if let Some(lines) = formula {
        // (a formula never closed: shown as it stands, and flagged by the check)
        blocks.push(Block::Display(lines.join(" ").trim().to_string()));
    }
    close(&mut paragraph, &mut blocks);
    blocks
}

/// A line that starts a point of a list: whether it is numbered, and its words.
fn item(line: &str) -> Option<(bool, &str)> {
    if let Some(rest) = line.strip_prefix("- ") {
        return Some((false, rest.trim()));
    }
    let digits = line.chars().take_while(char::is_ascii_digit).count();
    let rest = line[digits..].strip_prefix(". ")?;
    (digits > 0 && digits <= 2).then_some((true, rest.trim()))
}

/// How many words a primer has, its formulas on lines of their own left out.
pub fn word_count(blocks: &[Block]) -> usize {
    let words = |text: &str| text.split_whitespace().count();
    blocks
        .iter()
        .map(|b| match b {
            Block::Heading(t) | Block::Sub(t) | Block::Paragraph(t) | Block::Aside(t) => words(t),
            Block::List { items, .. } => items.iter().map(|i| words(i)).sum(),
            Block::Display(_) | Block::See(_) => 0,
        })
        .sum()
}

/// The sizes a primer is set in (points).
const BODY: f32 = 13.5;
const FORMULA: f32 = 14.5;

/// Set a primer's blocks in `ui`, one under another. Returns the scene a visitor
/// asked for by clicking one of the related scenes, if any.
pub fn show(ui: &mut egui::Ui, blocks: &[Block]) -> Option<String> {
    let mut wanted = None;
    ui.spacing_mut().item_spacing.y = 7.0;
    for (k, block) in blocks.iter().enumerate() {
        match block {
            Block::Heading(words) => {
                ui.add_space(if k == 0 { 2.0 } else { 14.0 });
                ui.label(RichText::new(words).size(18.0).strong().color(pal::WHITE.egui()));
                let line = ui.available_rect_before_wrap();
                ui.painter().hline(line.x_range(), line.top(), egui::Stroke::new(1.0, pal::DIM.a(0.25).egui()));
                ui.add_space(3.0);
            }
            Block::Sub(words) => {
                ui.add_space(8.0);
                crate::ui::words_unnoted(ui, words, 14.5, pal::YELLOW);
            }
            Block::Paragraph(words) => {
                crate::ui::words_unnoted(ui, words, BODY, pal::TEXT);
            }
            Block::Display(formula) => {
                crate::ui::equation_at(ui, formula, FORMULA);
            }
            Block::List { numbered, items } => {
                for (n, words) in items.iter().enumerate() {
                    ui.horizontal_top(|ui| {
                        ui.spacing_mut().item_spacing.x = 0.0;
                        let mark = if *numbered { format!("{}.", n + 1) } else { "•".to_string() };
                        let (rect, _) = ui.allocate_exact_size(egui::vec2(22.0, BODY + 4.0), egui::Sense::hover());
                        ui.painter().text(
                            rect.right_top() + egui::vec2(-8.0, 1.0),
                            egui::Align2::RIGHT_TOP,
                            mark,
                            egui::FontId::proportional(BODY),
                            pal::DIM.egui(),
                        );
                        ui.vertical(|ui| {
                            ui.set_width(ui.available_width());
                            crate::ui::words_unnoted(ui, words, BODY, pal::TEXT);
                        });
                    });
                }
            }
            Block::Aside(words) => {
                ui.add_space(2.0);
                let frame = egui::Frame::new().fill(egui::Color32::from_rgb(20, 26, 40)).inner_margin(egui::Margin { left: 14, right: 10, top: 8, bottom: 8 });
                let shown = frame.show(ui, |ui| {
                    ui.set_width(ui.available_width());
                    crate::ui::words_unnoted(ui, words, BODY, pal::TEXT);
                });
                let bar = shown.response.rect;
                ui.painter().rect_filled(egui::Rect::from_min_max(bar.left_top(), egui::pos2(bar.left() + 3.0, bar.bottom())), 0.0, pal::CYAN.a(0.8).egui());
            }
            Block::See(ids) => {
                let scenes = crate::scenes::registry();
                let there: Vec<&crate::scenes::SceneDef> = ids.iter().filter_map(|id| scenes.iter().find(|d| d.id == id)).collect();
                if there.is_empty() {
                    continue;
                }
                ui.add_space(12.0);
                ui.label(RichText::new("RELATED SCENES").size(10.5).strong().color(pal::MUTED.egui()));
                ui.horizontal_wrapped(|ui| {
                    for d in there {
                        let text = format!("→ {}", d.title);
                        if crate::ui::link_label(ui, crate::ui::widget_text(ui, &text, 12.5, Some(pal::LINK), false)).clicked() {
                            wanted = Some(d.id.to_string());
                        }
                    }
                });
            }
        }
    }
    wanted
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_primer_is_read_into_its_blocks() {
        let text = "# The idea\n\nTwo lines\nof one paragraph, with $a$.\n\n## Smaller\n$$ a = b $$\n$$ a\n= b \\\\ c $$\n- one\n- two\n  and more of two\n1. first\n2. second\n\n> Try this.\n> And that.\n\nsee: span, det\n";
        let blocks = parse(text);
        assert_eq!(
            blocks,
            vec![
                Block::Heading("The idea".into()),
                Block::Paragraph("Two lines of one paragraph, with $a$.".into()),
                Block::Sub("Smaller".into()),
                Block::Display("a = b".into()),
                Block::Display("a = b \\\\ c".into()),
                Block::List { numbered: false, items: vec!["one".into(), "two and more of two".into()] },
                Block::List { numbered: true, items: vec!["first".into(), "second".into()] },
                Block::Aside("Try this. And that.".into()),
                Block::See(vec!["span".into(), "det".into()]),
            ]
        );
        assert_eq!(word_count(&blocks), 2 + 7 + 1 + 6 + 2 + 4);
        // nothing is lost of a text that is not a primer at all
        assert_eq!(parse("just words\n$$ never closed"), vec![Block::Paragraph("just words".into()), Block::Display("never closed".into())]);
        assert!(parse("").is_empty());
    }

    #[test]
    fn every_primer_file_belongs_to_a_scene() {
        for id in IDS {
            assert!(crate::scenes::planned(id), "{id}: no such scene");
        }
        // and every scene of the catalogue has a file (written or still empty)
        for d in crate::scenes::registry() {
            assert!(IDS.contains(&d.id) || d.id == "showcase", "{}: no primer file", d.id);
        }
    }
}

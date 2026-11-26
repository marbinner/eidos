//! Causal inference: what data can and cannot say about what causes what.
//!
//! The first five scenes are the ideas (confounding, interventions, the three
//! junctions, what to adjust for, counterfactuals); the last four are causal discovery:
//! finding the arrows from the data, above all by fitting one variable from another
//! and asking whether what is left over is independent of the input. The last two are a
//! case study of a question still open: whether vaccination against shingles lowers the
//! risk of dementia.
//!
//! The statistics these scenes share are in `crate::math::causal`.

use crate::scenes::prelude::*;
use crate::ui;

/// The heading these scenes are listed under.
pub const GROUP: &str = "Causal inference";

// The scenes of this topic, in the order they appear in the catalogue.
crate::scenes::topic!(confound, dosee, junctions, adjust, counterfactual, direction, lingam, locscale, discover, pc);

// Two scenes about particular studies, made with this topic's tools and listed apart,
// near the end of the catalogue (`scenes::CASE_STUDIES`): they are about what one body
// of published work can and cannot show, not about a piece of mathematics.
mod shingles;
mod switch;

/// The case studies, in catalogue order.
pub fn case_studies() -> Vec<SceneDef> {
    vec![shingles::DEF, switch::DEF]
}

// One colour means one thing in every scene of the topic.
/// The variable asked about as a cause (a treatment, an input).
pub const CAUSE: Color = pal::CYAN;
/// The variable asked about as an effect (an outcome).
pub const EFFECT: Color = pal::ORANGE;
/// A third variable: a common cause, a go-between, a common effect.
pub const THIRD: Color = pal::PURPLE;
/// What is really so: the true causal effect, the true arrow.
pub const TRUTH: Color = pal::GREEN;
/// What the raw data suggest when read naively.
pub const NAIVE: Color = pal::PINK;

/// One variable of a causal graph drawn on the screen: where its disc is (logical
/// pixels), its name and its colour.
#[derive(Clone, Copy)]
pub struct Node {
    pub at: Vec2,
    pub name: &'static str,
    pub color: Color,
}

/// How the link between two variables is drawn.
#[derive(Clone, Copy, PartialEq)]
pub enum Link {
    /// The first causes the second: an arrow.
    Arrow,
    /// Connected, direction not known: a plain line.
    Line,
    /// An arrow that an intervention has cut: dashed, with a stroke across it.
    Cut,
}

/// A causal graph in screen space: each variable a ringed disc of the given `radius`
/// with its name inside, each link from rim to rim in its own colour and `width`.
/// A link that names a variable that is not there, or joins a variable to itself, is
/// left out.
pub fn hud_graph(f: &mut Frame, nodes: &[Node], links: &[(usize, usize, Link, Color, f32)], radius: f32) {
    for &(from, to, link, color, width) in links {
        let (Some(a), Some(b)) = (nodes.get(from), nodes.get(to)) else { continue };
        let along = (b.at - a.at).normalize_or_zero();
        if along == Vec2::ZERO {
            continue;
        }
        // from rim to rim, with a little air
        let (p, q) = (a.at + along * (radius + 3.0), b.at - along * (radius + 4.0));
        if (q - p).dot(along) <= 0.0 {
            continue;
        }
        match link {
            Link::Arrow => hud_arrow(f, p, q, width, color),
            Link::Line => hud_line(f, p, q, width, color),
            Link::Cut => {
                let (length, dash) = ((q - p).length(), 6.0);
                let count = (length / (2.0 * dash)).ceil().max(1.0) as usize;
                for k in 0..count {
                    let (s, e) = (k as f32 * 2.0 * dash, (k as f32 * 2.0 * dash + dash).min(length));
                    hud_line(f, p + along * s, p + along * e, width, color.a(0.6));
                }
                // (still an arrow, to show which way it pointed before it was cut)
                let (head, side) = (4.0 + 2.5 * width, Vec2::new(-along.y, along.x));
                hud_line(f, q, q - along * head + side * head * 0.5, width, color.a(0.6));
                hud_line(f, q, q - along * head - side * head * 0.5, width, color.a(0.6));
                let (mid, across) = ((p + q) * 0.5, side);
                hud_line(f, mid - (across + along * 0.6) * 7.0, mid + (across + along * 0.6) * 7.0, width + 0.6, color);
            }
        }
    }
    for node in nodes {
        hud_circle(f, node.at, radius, 1.8, node.color);
        f.text_px(node.at, node.name, (radius * 0.95).max(11.0), node.color, Align2::CENTER_CENTER);
    }
}

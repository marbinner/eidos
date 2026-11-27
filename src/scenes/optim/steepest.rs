//! Which way is steepest? It depends on the ruler (planned; not written yet).

use super::*;

pub const DEF: SceneDef = SceneDef {
    id: "steepest",
    title: "Which way is steepest? It depends on the ruler",
    group: crate::scenes::UNWRITTEN,
    blurb: "Measure a step by its length, by its largest entry or by the curvature of the bowl, and the steepest way down becomes the gradient, its signs, or Newton's step.",
    make: |_| Box::new(crate::scenes::Unwritten),
};

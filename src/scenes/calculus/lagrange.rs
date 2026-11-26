//! Lagrange multipliers (planned; not written yet).

use super::*;

pub const DEF: SceneDef = SceneDef {
    id: "lagrange",
    title: "Lagrange multipliers",
    group: crate::scenes::UNWRITTEN,
    blurb: "The highest point of a path on a hillside: where the path runs along a contour, and what the multiplier is the price of.",
    make: |_| Box::new(crate::scenes::Unwritten),
};

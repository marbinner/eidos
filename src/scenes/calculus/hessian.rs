//! Curvature: the Hessian and Newton's method (planned; not written yet).

use super::*;

pub const DEF: SceneDef = SceneDef {
    id: "hessian",
    title: "Curvature: the Hessian and Newton's method",
    group: crate::scenes::UNWRITTEN,
    blurb: "The second derivatives of a surface as a matrix: bowls, domes and saddles, and the step that jumps to the bottom of the local bowl.",
    make: |_| Box::new(crate::scenes::Unwritten),
};

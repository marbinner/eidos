//! Step size and curvature (planned; not written yet).

use super::*;

pub const DEF: SceneDef = SceneDef {
    id: "stepsize",
    title: "Step size and curvature",
    group: crate::scenes::UNWRITTEN,
    blurb: "Gradient descent on a bowl, one direction at a time: the sharpest direction sets the longest safe step, and the flattest one sets how long the descent takes.",
    make: |_| Box::new(crate::scenes::Unwritten),
};

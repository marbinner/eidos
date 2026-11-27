//! The edge of stability (planned; not written yet).

use super::*;

pub const DEF: SceneDef = SceneDef {
    id: "eos",
    title: "The edge of stability",
    group: crate::scenes::UNWRITTEN,
    blurb: "Train a network by full-batch gradient descent and watch its sharpness, the largest curvature of the loss, against the largest that the step size can bear.",
    make: |_| Box::new(crate::scenes::Unwritten),
};

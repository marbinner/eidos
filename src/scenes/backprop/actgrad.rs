//! Activation functions as gates (planned; not written yet).

use super::*;

pub const DEF: SceneDef = SceneDef {
    id: "actgrad",
    title: "Activation functions as gates",
    group: crate::scenes::UNWRITTEN,
    blurb: "Each unit lets the gradient through in proportion to its slope: saturated tanh, dead ReLUs, and what the smooth modern ones change.",
    make: |_| Box::new(crate::scenes::Unwritten),
};

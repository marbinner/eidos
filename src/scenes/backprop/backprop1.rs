//! Backpropagation in the smallest network (planned; not written yet).

use super::*;

pub const DEF: SceneDef = SceneDef {
    id: "backprop1",
    title: "Backpropagation in the smallest network",
    group: crate::scenes::UNWRITTEN,
    blurb: "Two layers, a handful of weights and every number shown: the forward pass, the loss, and the gradient handed back step by step.",
    make: |_| Box::new(crate::scenes::Unwritten),
};

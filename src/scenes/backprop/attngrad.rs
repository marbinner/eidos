//! Backpropagation through attention (planned; not written yet).

use super::*;

pub const DEF: SceneDef = SceneDef {
    id: "attngrad",
    title: "Backpropagation through attention",
    group: crate::scenes::UNWRITTEN,
    blurb: "One attention head has two roads for the gradient, through what is moved and through where it looks, and the second closes when attention is sure.",
    make: |_| Box::new(crate::scenes::Unwritten),
};

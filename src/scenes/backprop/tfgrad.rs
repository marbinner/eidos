//! Gradients in a transformer (planned; not written yet).

use super::*;

pub const DEF: SceneDef = SceneDef {
    id: "tfgrad",
    title: "Gradients in a transformer",
    group: crate::scenes::UNWRITTEN,
    blurb: "A small modern transformer, layer by layer and token by token: where the gradient of the loss goes, through which parts, and what pre-norm and skip connections change.",
    make: |_| Box::new(crate::scenes::Unwritten),
};

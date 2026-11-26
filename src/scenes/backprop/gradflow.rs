//! Gradients through depth: vanishing and exploding (planned; not written yet).

use super::*;

pub const DEF: SceneDef = SceneDef {
    id: "gradflow",
    title: "Gradients through depth: vanishing and exploding",
    group: crate::scenes::UNWRITTEN,
    blurb: "Fifty layers multiply fifty Jacobians: how the scale of the weights and the activation decide whether the gradient dies, explodes or arrives whole.",
    make: |_| Box::new(crate::scenes::Unwritten),
};

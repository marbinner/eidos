//! The chain rule: Jacobians multiply (planned; not written yet).

use super::*;

pub const DEF: SceneDef = SceneDef {
    id: "chain",
    title: "The chain rule: Jacobians multiply",
    group: crate::scenes::UNWRITTEN,
    blurb: "One map after another: small changes pass through each in turn, forwards as columns and backwards as rows, which is all that backpropagation is.",
    make: |_| Box::new(crate::scenes::Unwritten),
};

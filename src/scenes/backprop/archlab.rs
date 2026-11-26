//! Architecture playground (planned; not written yet).

use super::*;

pub const DEF: SceneDef = SceneDef {
    id: "archlab",
    title: "Architecture playground",
    group: crate::scenes::UNWRITTEN,
    blurb: "Build a network, choose its depth, width, activation, normalisation, skip connections and starting weights, and watch signals go forward, gradients come back, and training itself.",
    make: |_| Box::new(crate::scenes::Unwritten),
};

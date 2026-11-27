//! The noise scale and the critical batch size (planned; not written yet).

use super::*;

pub const DEF: SceneDef = SceneDef {
    id: "noisescale",
    title: "The noise scale and the critical batch size",
    group: crate::scenes::UNWRITTEN,
    blurb: "Bigger batches allow longer steps until curvature, not noise, is the limit: how many steps a larger batch saves, and the batch size beyond which it saves none.",
    make: |_| Box::new(crate::scenes::Unwritten),
};

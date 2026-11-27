//! Optimiser playground (planned; not written yet).

use super::*;

pub const DEF: SceneDef = SceneDef {
    id: "optlab",
    title: "Optimiser playground",
    group: crate::scenes::UNWRITTEN,
    blurb: "Two optimisers on the same network and the same data, with every knob: the loss, the updates, the sharpness and the noise scale side by side.",
    make: |_| Box::new(crate::scenes::Unwritten),
};

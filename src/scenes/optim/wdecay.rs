//! Weight decay and the effective learning rate (planned; not written yet).

use super::*;

pub const DEF: SceneDef = SceneDef {
    id: "wdecay",
    title: "Weight decay and the effective learning rate",
    group: crate::scenes::UNWRITTEN,
    blurb: "For weights that feed a normalisation layer only their direction matters: their length sets the real step size, and weight decay sets their length.",
    make: |_| Box::new(crate::scenes::Unwritten),
};

//! Momentum: a ball with weight (planned; not written yet).

use super::*;

pub const DEF: SceneDef = SceneDef {
    id: "momentum",
    title: "Momentum: a ball with weight",
    group: crate::scenes::UNWRITTEN,
    blurb: "Keeping part of the last step turns a crawl along the valley floor into a glide: which steps and momenta converge, which ring, and which fly off.",
    make: |_| Box::new(crate::scenes::Unwritten),
};

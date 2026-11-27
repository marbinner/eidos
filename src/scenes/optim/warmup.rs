//! Warm-up and decay: the learning-rate schedule (planned; not written yet).

use super::*;

pub const DEF: SceneDef = SceneDef {
    id: "warmup",
    title: "Warm-up and decay: the learning-rate schedule",
    group: crate::scenes::UNWRITTEN,
    blurb: "Why training starts with short steps and ends with short steps: curvature too high for a long step at the start, and noise that only a shorter step can quiet at the end.",
    make: |_| Box::new(crate::scenes::Unwritten),
};

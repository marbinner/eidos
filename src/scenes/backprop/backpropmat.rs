//! Backpropagation, a layer at a time (planned; not written yet).

use super::*;

pub const DEF: SceneDef = SceneDef {
    id: "backpropmat",
    title: "Backpropagation, a layer at a time",
    group: crate::scenes::UNWRITTEN,
    blurb: "The same sweep in matrices: each layer hands its error back through the transpose of its weights, and a weight's gradient is an outer product.",
    make: |_| Box::new(crate::scenes::Unwritten),
};

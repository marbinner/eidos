//! Derivatives by machine: the computation graph (planned; not written yet).

use super::*;

pub const DEF: SceneDef = SceneDef {
    id: "autodiff",
    title: "Derivatives by machine: the computation graph",
    group: crate::scenes::UNWRITTEN,
    blurb: "A formula as a graph of small steps, each with a slope of its own: one sweep forwards for the value, one sweep backwards for every derivative at once.",
    make: |_| Box::new(crate::scenes::Unwritten),
};

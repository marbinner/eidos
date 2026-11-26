//! The fundamental theorem of calculus (planned; not written yet).

use super::*;

pub const DEF: SceneDef = SceneDef {
    id: "ftc",
    title: "The fundamental theorem of calculus",
    group: crate::scenes::UNWRITTEN,
    blurb: "The area so far, as a function: its slope is the curve you started with, and that is why integrals are found by undoing derivatives.",
    make: |_| Box::new(crate::scenes::Unwritten),
};

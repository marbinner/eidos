//! Changing variables in an integral (planned; not written yet).

use super::*;

pub const DEF: SceneDef = SceneDef {
    id: "changevar",
    title: "Changing variables in an integral",
    group: crate::scenes::UNWRITTEN,
    blurb: "Why polar coordinates bring an extra r: a map stretches each small patch by the determinant of its Jacobian.",
    make: |_| Box::new(crate::scenes::Unwritten),
};

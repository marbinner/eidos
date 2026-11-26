//! The Jacobian: a curved map up close (planned; not written yet).

use super::*;

pub const DEF: SceneDef = SceneDef {
    id: "jacobian",
    title: "The Jacobian: a curved map up close",
    group: crate::scenes::UNWRITTEN,
    blurb: "Zoom in on any smooth map of the plane and it becomes a linear map: its matrix of partial derivatives, and what its determinant says.",
    make: |_| Box::new(crate::scenes::Unwritten),
};

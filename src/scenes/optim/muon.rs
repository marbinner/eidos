//! Muon: an orthogonal step (planned; not written yet).

use super::*;

pub const DEF: SceneDef = SceneDef {
    id: "muon",
    title: "Muon: an orthogonal step",
    group: crate::scenes::UNWRITTEN,
    blurb: "Muon replaces a layer's update by the nearest matrix that stretches nothing: every singular value pushed to one by a short polynomial iteration, and what that does to learning.",
    make: |_| Box::new(crate::scenes::Unwritten),
};

//! Muon's relatives (planned; not written yet).

use super::*;

pub const DEF: SceneDef = SceneDef {
    id: "muonkin",
    title: "Muon's relatives",
    group: crate::scenes::UNWRITTEN,
    blurb: "Sign descent, Shampoo, SOAP and Muon's own variants at work on one gradient matrix: what each does to its singular values, and what they share.",
    make: |_| Box::new(crate::scenes::Unwritten),
};

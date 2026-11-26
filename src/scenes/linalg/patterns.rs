//! A zoo of larger matrices (planned; not written yet).

use super::*;

pub const DEF: SceneDef = SceneDef {
    id: "patterns",
    title: "A zoo of larger matrices",
    group: crate::scenes::UNWRITTEN,
    blurb: "Banded, circulant, companion, stochastic and more: matrices known by their pattern, what each pattern buys, and where its eigenvalues lie.",
    make: |_| Box::new(crate::scenes::Unwritten),
};

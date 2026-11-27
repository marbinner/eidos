//! Normalised steps in noise (planned; not written yet).

use super::*;

pub const DEF: SceneDef = SceneDef {
    id: "signnoise",
    title: "Normalised steps in noise",
    group: crate::scenes::UNWRITTEN,
    blurb: "Sign descent, Adam and Muon throw away the size of the gradient: what is left when the gradient is mostly noise, and how that changes the batch size worth paying for.",
    make: |_| Box::new(crate::scenes::Unwritten),
};

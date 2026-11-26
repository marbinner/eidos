//! Normalisation and the gradient (planned; not written yet).

use super::*;

pub const DEF: SceneDef = SceneDef {
    id: "normgrad",
    title: "Normalisation and the gradient",
    group: crate::scenes::UNWRITTEN,
    blurb: "Batch, layer and RMS normalisation: what each does on the way forward, and how it reshapes the gradient on the way back.",
    make: |_| Box::new(crate::scenes::Unwritten),
};

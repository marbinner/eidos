//! Muon against Adam on one network (planned; not written yet).

use super::*;

pub const DEF: SceneDef = SceneDef {
    id: "muonadam",
    title: "Muon against Adam on one network",
    group: crate::scenes::UNWRITTEN,
    blurb: "The same small network trained by both: the loss, the shape of each update, the learning rates each tolerates, and what changes with the batch size.",
    make: |_| Box::new(crate::scenes::Unwritten),
};

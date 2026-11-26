//! Where the gradient starts: losses (planned; not written yet).

use super::*;

pub const DEF: SceneDef = SceneDef {
    id: "lossgrad",
    title: "Where the gradient starts: losses",
    group: crate::scenes::UNWRITTEN,
    blurb: "Softmax with cross-entropy hands back the probabilities minus the truth; other pairings of output and loss hand back less, and stall.",
    make: |_| Box::new(crate::scenes::Unwritten),
};

//! A mini-batch gradient: signal and noise (planned; not written yet).

use super::*;

pub const DEF: SceneDef = SceneDef {
    id: "minibatch",
    title: "A mini-batch gradient: signal and noise",
    group: crate::scenes::UNWRITTEN,
    blurb: "The gradient of a few examples is the true gradient plus noise that shrinks as the batch grows: the batch size at which the two are equal is the noise scale.",
    make: |_| Box::new(crate::scenes::Unwritten),
};

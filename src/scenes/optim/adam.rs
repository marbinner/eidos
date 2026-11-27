//! Adam, taken apart (planned; not written yet).

use super::*;

pub const DEF: SceneDef = SceneDef {
    id: "adam",
    title: "Adam, taken apart",
    group: crate::scenes::UNWRITTEN,
    blurb: "Two running averages and a division: what each piece of Adam does to a stream of noisy gradients, and when the whole behaves like the sign of the gradient.",
    make: |_| Box::new(crate::scenes::Unwritten),
};

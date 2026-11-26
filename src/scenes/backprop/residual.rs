//! Skip connections (planned; not written yet).

use super::*;

pub const DEF: SceneDef = SceneDef {
    id: "residual",
    title: "Skip connections",
    group: crate::scenes::UNWRITTEN,
    blurb: "Adding each layer's output to its input gives the gradient a straight road back: paths of every length, and why depth stops hurting.",
    make: |_| Box::new(crate::scenes::Unwritten),
};

//! Too sharp to stay: two weights (planned; not written yet).

use super::*;

pub const DEF: SceneDef = SceneDef {
    id: "eostoy",
    title: "Too sharp to stay: two weights",
    group: crate::scenes::UNWRITTEN,
    blurb: "A valley whose walls are steeper at one end: gradient descent cannot settle where the curvature is more than two over its step, and is thrown along to flatter ground.",
    make: |_| Box::new(crate::scenes::Unwritten),
};

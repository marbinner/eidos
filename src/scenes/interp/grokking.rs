//! Grokking: a network that memorises an arithmetic table and, much later, finds the
//! rule. The scene is the playground's arithmetic task opened directly (the code is
//! in `neural/table.rs`).

use super::*;

pub const DEF: SceneDef = SceneDef {
    id: "grokking",
    title: "Grokking a table",
    group: GROUP,
    blurb: r"A network memorises $a + b \;\mathrm{mod}\; p$, then suddenly understands it. See the algorithm it found.",
    make: |g| crate::scenes::neural::arithmetic(g),
};

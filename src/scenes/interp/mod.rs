//! Interpretability: opening a trained network up to see how it does what it does.

use crate::scenes::prelude::*;

/// The heading these scenes are listed under.
pub const GROUP: &str = "Interpretability";

// The scenes of this topic, in the order they appear in the catalogue.
// (Superposition, which opens this heading, is the last scene of the neural topic.)
crate::scenes::topic!(sae, probe, grokking, transformer);

// (what the scenes are built on)
mod autoencoder;

/// How fast this topic's models train in this build (appended to the networks' figures).
pub fn training_benchmark() -> String {
    transformer::benchmark()
}

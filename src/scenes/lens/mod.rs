//! The Jacobian lens: what averaging a network's Jacobians keeps, after Ostrow, Qian,
//! Liu and Fiete, "A Mathematical Theory of the Jacobian Lens" (2026).

use crate::scenes::prelude::*;

/// The heading these scenes are listed under.
pub const GROUP: &str = "The Jacobian lens";

// The scenes of this topic, in the order they appear in the catalogue.
crate::scenes::topic!(lyapunov, coherence, belief);

// Not scenes: the layout of a page of charts; a deep random network and its Lyapunov
// exponents; and the ring hidden Markov model, its Bayes filter and the lens's factors.
mod deep;
mod page;
mod ring;

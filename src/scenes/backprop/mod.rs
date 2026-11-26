//! Backpropagation: how the gradient of a loss is handed back through a network, from a
//! formula's computation graph to a transformer, and what each design choice (the
//! activation, the normalisation, the skip connections, the starting weights) does to it.

use crate::scenes::prelude::*;

// The scenes of this topic, in the order they appear in the catalogue: a course, to be
// read from the top.
crate::scenes::topic!(autodiff, backprop1, backpropmat, lossgrad, actgrad, gradflow, normgrad, residual, attngrad, tfgrad, archlab);

/// The heading the scenes of this topic are listed under.
#[allow(dead_code)]
pub const GROUP: &str = "Backpropagation";

// The colours of the course, one meaning each in every scene of it.
/// What goes forward: inputs, activations, the signal.
#[allow(dead_code)]
pub const FORWARD: Color = pal::CYAN;
/// What comes back: the gradient of the loss with respect to whatever it is drawn on.
#[allow(dead_code)]
pub const BACKWARD: Color = pal::PINK;
/// A weight or a bias, and its gradient where the two are shown together.
#[allow(dead_code)]
pub const WEIGHT: Color = pal::YELLOW;
/// The loss, and the target it is measured against.
#[allow(dead_code)]
pub const LOSS: Color = pal::ORANGE;
/// A skip connection: the road that goes round a layer.
#[allow(dead_code)]
pub const SKIP: Color = pal::GREEN;
/// A normalisation layer.
#[allow(dead_code)]
pub const NORMED: Color = pal::PURPLE;

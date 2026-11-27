//! Optimisation: why the rules that train networks work. From gradient descent on a
//! bowl, through momentum, mini-batch noise and Adam, to Muon and its relatives; then
//! what the curvature of the loss does during training (the edge of stability, warm-up,
//! weight decay), and a playground that sets any two rules side by side.
//!
//! What the course says about published work, and each rule's exact form, comes from
//! the sources listed in `SOURCES.md` beside this file.

use crate::scenes::prelude::*;

// The scenes of this topic, in the order they appear in the catalogue: a course, to be
// read from the top.
crate::scenes::topic!(
    stepsize, momentum, steepest, adam, minibatch, noisefloor, noisescale, muon, muonkin, signnoise, muonadam, eostoy, eos, warmup, wdecay, optlab
);

/// The heading the scenes of this topic are listed under.
pub const GROUP: &str = "Optimisation";

// The colours of the course, one meaning each in every scene of it. A rule keeps the
// colour it has in the scenes "descent" and "race" (`pal::SERIES` in the order of
// `nn::Optim::ALL`), so that a visitor who has met it there knows it here.
/// Plain gradient descent, with the whole data set or with mini-batches (SGD).
pub const SGD: Color = pal::SERIES[0];
/// Gradient descent with momentum (the heavy ball).
pub const MOMENTUM: Color = pal::SERIES[1];
/// Nesterov's momentum.
pub const NESTEROV: Color = pal::SERIES[2];
/// AdaGrad.
pub const ADAGRAD: Color = pal::SERIES[3];
/// RMSProp.
pub const RMSPROP: Color = pal::SERIES[4];
/// Adam and AdamW.
pub const ADAM: Color = pal::SERIES[5];
/// Muon.
pub const MUON: Color = pal::SERIES[6];

/// The true gradient: that of the whole data set, the signal a mini-batch estimates.
pub const SIGNAL: Color = pal::WHITE;
/// Noise: what a mini-batch's gradient adds to the true one.
pub const NOISE: Color = pal::RED;
/// Curvature: an eigenvalue of the Hessian, and the sharpness (the largest of them).
pub const CURVATURE: Color = pal::GREEN;
/// A limit that a step size sets, such as the largest curvature gradient descent can bear.
pub const LIMIT: Color = pal::YELLOW;

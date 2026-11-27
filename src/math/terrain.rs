//! What the Optimisation scenes measure of the ground a training run crosses, one way
//! for every scene: the loss and its gradient on chosen examples ([`Problem`]), the
//! curvature of the loss (Hessian-vector products [`hvp`], the whole Hessian of a small
//! model [`Hessian`], its largest eigenvalues by Lanczos's method [`Lanczos`], the
//! curvature of mini-batches [`BatchSharpness`]), the noise of mini-batch gradients
//! (per-example gradients and their statistics [`PerExample`], [`Noise`], the noise
//! scale with the Hessian [`FullNoise`], the estimator from two batch sizes
//! [`two_batch_estimates`]), the model of one step of McCandlish and co-authors
//! ([`OneStep`]) and the noisy quadratic model of Zhang and co-authors ([`Nqm`]).
//!
//! How a scene uses it:
//!
//! ```text
//!     let problem = Problem::spirals(64, 3, &mut Rng::new(seed));
//!     let mut model = Model::mlp(&spec, &mut Rng::new(seed));
//!     let mut tape = Tape::new();
//!     let mut batches = Sampler::new(problem.len(), 8, Draw::WithoutReplacement, Rng::new(seed));
//!
//!     problem.grad(&mut model, &mut tape, batches.deal());   // a training step's gradient
//!     model.step_sgd(0.1);
//!
//!     let mut lanczos = Lanczos::new(&model, 1, seed);        // the sharpness, a slice a frame
//!     lanczos.work(&problem, &mut model, &mut tape, &budget);
//!     lanczos.sharpness(); lanczos.residual();
//!     // ... after more training steps:
//!     lanczos.restart();                                     // warm: from the last estimate
//! ```
//!
//! **What things cost.** A *pass* is one forward pass of the network and its backward
//! sweep. An MLP reads a whole batch in one pass (its cost grows with the batch); a
//! transformer reads one sequence per pass. So a mean gradient over the examples `idx`
//! is one pass of `idx.len()` examples for an MLP and `idx.len()` passes for a
//! transformer; a loss alone is the forward half of that (for small networks the saving
//! is small, as the timings below show: recording the pass is much of its cost); a
//! Hessian-vector product is two gradients. Work that takes more than a few passes is
//! an object with `advance(…, n)` (n more units of work, returns whether it is done) and
//! `work(…, budget)` (units until the frame's [`Budget`] is spent, at least one, returns
//! whether it is done), to be read when it is done. Measured costs are in the test
//! `the_cost_of_measuring` (ignored; it prints them). In a `--profile fast` build on a
//! busy shared server (natively; the browser was not measured): for a tanh MLP with 3
//! hidden layers 16 wide (643 parameters) on 64 spiral points, a gradient took 0.10 ms
//! and the loss alone 0.08 ms, a Hessian-vector product 0.20 ms, the 64 per-example
//! gradients 0.27 ms, the sharpness to 10⁻⁴ from cold 1.4 ms (6 products) and warm after
//! a step 0.8 ms (4 products); for a transformer of 2 blocks 16 wide (6976 parameters),
//! a gradient on one sequence of 12 tokens 0.15 ms and a Hessian-vector product on it
//! 0.34 ms.
//!
//! **Where the definitions come from.** Every quantity that has a name in the
//! literature is defined as its source defines it, and its documentation names the
//! source and the place in it (with the section of the course's dossier, `noise.md` or
//! `curvature.md`, where it was checked). What is not in a source is derived in the
//! documentation and marked "derived here".
//!
//! **Conventions.** Everything is f64. Flat vectors of parameters list every group's
//! values end to end, in the order of `Model::groups`, each group row-major ([`get`],
//! [`spans`]). Nothing here panics on what a slider can produce: an empty list of
//! examples gives a loss of 0 and a zero gradient (a mean over nothing is 0, as in
//! `autograd`), one example works, and numbers that are not finite are carried through
//! (a NaN in gives NaN out). A model that does not fit a problem (an MLP with the wrong
//! number of inputs, a transformer for points) gives NaN.
//!
//! **Batch normalisation.** With `Norm::Batch` an MLP's output for one example depends
//! on the others in its batch, so the loss of a batch is not the mean of its examples'
//! own losses, and per-example gradients (each example a batch of its own, which batch
//! normalisation reduces to nothing) do not describe the noise of a batch. The noise
//! statistics here assume a loss that is a mean of per-example losses: use networks
//! without batch normalisation for them.

// (Nothing uses this yet: the Optimisation scenes are its users.)
#![allow(dead_code)]

use std::ops::Range;

use super::arch::{self, Loss, Model};
use super::autograd::Tape;
use super::linalg;
use super::rng::Rng;
use crate::clock::Budget;

// ----- small vector helpers ---------------------------------------------------------------

fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

fn norm(a: &[f64]) -> f64 {
    dot(a, a).sqrt()
}

/// y += k x
fn axpy(k: f64, x: &[f64], y: &mut [f64]) {
    for (y, x) in y.iter_mut().zip(x) {
        *y += k * x;
    }
}

// ----- flat views of a model ------------------------------------------------------------------

/// The number of parameters of `model`: every group's entries together, the length of
/// the flat vectors of this module.
pub fn n_params(model: &Model) -> usize {
    model.n_params()
}

/// Every group's values end to end, in the order of `Model::groups`, each group
/// row-major, into `out` (cleared first).
pub fn get(model: &Model, out: &mut Vec<f64>) {
    out.clear();
    for g in model.groups() {
        out.extend_from_slice(&g.value);
    }
}

/// Set every group's values from a flat vector laid out as [`get`] lays it out.
/// Numbers beyond `n_params` are ignored; a shorter vector sets only the parameters it
/// reaches and leaves the rest as they were.
pub fn set(model: &mut Model, w: &[f64]) {
    let mut at = 0;
    for g in model.groups_mut() {
        let end = (at + g.value.len()).min(w.len());
        if at < end {
            g.value[..end - at].copy_from_slice(&w[at..end]);
        }
        at += g.value.len();
    }
}

/// The accumulated gradients (`Group::grad`) end to end, laid out as [`get`] lays out
/// the values, into `out` (cleared first).
pub fn grad_of(model: &Model, out: &mut Vec<f64>) {
    out.clear();
    for g in model.groups() {
        out.extend_from_slice(&g.grad);
    }
}

/// Where each group lies in the flat vector, in the order of `Model::groups`: group k
/// is entries `spans(model)[k]`.
pub fn spans(model: &Model) -> Vec<Range<usize>> {
    let mut at = 0;
    model
        .groups()
        .iter()
        .map(|g| {
            let r = at..at + g.value.len();
            at = r.end;
            r
        })
        .collect()
}

/// Where the group of this name ("layer 2 weights", "block 1 head 1 query", ...) lies
/// in the flat vector.
pub fn span(model: &Model, name: &str) -> Option<Range<usize>> {
    let k = model.index_of(name)?;
    spans(model).into_iter().nth(k)
}

// ----- a problem: data and a loss ---------------------------------------------------------

#[derive(Clone, Debug)]
enum Data {
    /// Examples for an MLP: `inputs` numbers each in `x`, and in `y` one class index
    /// (cross-entropy) or `per` numbers (squared error) each.
    Points { x: Vec<f64>, y: Vec<f64>, inputs: usize, per: usize, loss: Loss, classes: usize },
    /// Sequences for a transformer, `len` tokens each, with their targets.
    Sequences { tokens: Vec<usize>, targets: Vec<usize>, len: usize },
}

/// A data set and its loss, measured one way for every scene: the loss on all of it or
/// on chosen examples, and the MEAN gradient over chosen examples.
///
/// What "the loss of an example" is, exactly (the losses of `arch`): for points with
/// squared error, the mean over the example's outputs of (output − target)² (a mean,
/// not a sum, and no factor ½); with cross-entropy, minus the log of the probability
/// the softmax of the outputs gives the example's class; for a sequence (an "example"
/// of [`Problem::repeats`] is one sequence), the mean over the positions that have a
/// target of the cross-entropy of the next token. The loss on a set of examples is the
/// mean over them of their losses, each example (each sequence) weighing the same; an
/// index listed twice counts twice. For an MLP this is exactly the loss `arch` computes
/// for the batch in one pass (but see the module's note on batch normalisation).
#[derive(Clone, Debug)]
pub struct Problem {
    data: Data,
    n: usize,
}

impl Problem {
    /// `n` points on `classes` spiral arms (`arch::spirals`: 2 numbers per point, sizes
    /// taken as at least 1 and `classes` as 2 or 3), cross-entropy. For an MLP with 2
    /// inputs and at least `classes` outputs.
    pub fn spirals(n: usize, classes: usize, rng: &mut Rng) -> Problem {
        let (x, y) = arch::spirals(n, classes, rng);
        Problem::from_points(x, y, 2, 1, Loss::CrossEntropy, classes.clamp(2, 3))
    }

    /// `n` examples of `inputs` standard normal numbers and the outputs of
    /// `arch::teacher`'s fixed network as targets, mean squared error (sizes taken as at
    /// least 1). For an MLP with `inputs` inputs and 1 output.
    pub fn teacher(n: usize, inputs: usize, rng: &mut Rng) -> Problem {
        let (x, y) = arch::teacher(n, inputs, rng);
        Problem::from_points(x, y, inputs.max(1), 1, Loss::Mse, 0)
    }

    /// `sequences` sequences of `arch::repeats` (a random run of tokens and the same run
    /// again; sizes taken as at least 1), each scored on every position that has a next
    /// token: next-token cross-entropy for a transformer whose vocabulary has at least
    /// `vocab` tokens. An example is one sequence.
    pub fn repeats(sequences: usize, context: usize, vocab: usize, rng: &mut Rng) -> Problem {
        let n = sequences.max(1);
        let len = context.max(1);
        let (mut tokens, mut targets) = (Vec::with_capacity(n * len), Vec::with_capacity(n * len));
        for _ in 0..n {
            let (t, y) = arch::repeats(len, vocab, rng);
            tokens.extend(t);
            targets.extend(y);
        }
        Problem { data: Data::Sequences { tokens, targets, len }, n }
    }

    /// The caller's own points for an MLP: `n` examples, `inputs` holding
    /// `inputs.len() / n` numbers for each, one example after another; `targets` one
    /// class index per example for [`Loss::CrossEntropy`] (0.0, 1.0, ...) or
    /// `targets.len() / n` numbers per example for [`Loss::Mse`]. An example whose class
    /// is not a whole number from 0 up (negative, fractional, NaN) is left out, so
    /// `len()` can be less than `n`. `n` of 0 gives an empty problem.
    pub fn points(inputs: &[f64], targets: &[f64], n: usize, loss: Loss) -> Problem {
        if n == 0 {
            return Problem::from_points(Vec::new(), Vec::new(), 0, 0, loss, 0);
        }
        let dims = inputs.len() / n;
        let per = match loss {
            Loss::CrossEntropy => 1,
            Loss::Mse => targets.len() / n,
        };
        let (mut x, mut y) = (Vec::new(), Vec::new());
        let mut classes = 0usize;
        for k in 0..n {
            let (xs, ys) = (&inputs[k * dims..(k + 1) * dims], &targets[(k * per).min(targets.len())..((k + 1) * per).min(targets.len())]);
            if ys.len() < per {
                break;
            }
            if loss == Loss::CrossEntropy {
                let t = ys[0];
                if !(t >= 0.0 && t.fract() == 0.0) {
                    continue;
                }
                classes = classes.max((t as usize).saturating_add(1));
            }
            x.extend_from_slice(xs);
            y.extend_from_slice(ys);
        }
        Problem::from_points(x, y, dims, per, loss, classes)
    }

    fn from_points(x: Vec<f64>, y: Vec<f64>, inputs: usize, per: usize, loss: Loss, classes: usize) -> Problem {
        let n = y.len().checked_div(per).unwrap_or(0);
        Problem { data: Data::Points { x, y, inputs, per, loss, classes }, n }
    }

    /// The number of examples (points, or sequences).
    pub fn len(&self) -> usize {
        self.n
    }

    /// Are there no examples?
    pub fn is_empty(&self) -> bool {
        self.n == 0
    }

    /// Every example's index, 0 to `len() − 1`: the list that means "all the data" where
    /// a function takes a list of examples.
    pub fn all(&self) -> Vec<usize> {
        (0..self.n).collect()
    }

    /// The loss of an MLP problem; `None` for sequences (always next-token
    /// cross-entropy).
    pub fn loss_kind(&self) -> Option<Loss> {
        match &self.data {
            Data::Points { loss, .. } => Some(*loss),
            Data::Sequences { .. } => None,
        }
    }

    /// Example `k` of an MLP problem: its inputs and its targets (one class index, or
    /// the outputs wanted).
    pub fn point(&self, k: usize) -> Option<(&[f64], &[f64])> {
        match &self.data {
            Data::Points { x, y, inputs, per, .. } if k < self.n => Some((&x[k * inputs..(k + 1) * inputs], &y[k * per..(k + 1) * per])),
            _ => None,
        }
    }

    /// Sequence `k` of a sequence problem: its tokens and the target at each position
    /// (`NO_TARGET` where there is none).
    pub fn sequence(&self, k: usize) -> Option<(&[usize], &[usize])> {
        match &self.data {
            Data::Sequences { tokens, targets, len } if k < self.n => Some((&tokens[k * len..(k + 1) * len], &targets[k * len..(k + 1) * len])),
            _ => None,
        }
    }

    /// Does `model` fit this problem: an MLP with as many inputs as an example has
    /// numbers and (squared error) as many outputs as an example has targets, or
    /// (cross-entropy) at least as many outputs as there are classes; a transformer for
    /// sequences. A model that does not fit gets NaN for every loss and gradient.
    pub fn fits(&self, model: &Model) -> bool {
        match &self.data {
            Data::Points { inputs, per, loss, classes, .. } => model.mlp_spec().is_some_and(|s| {
                s.inputs == *inputs
                    && match loss {
                        Loss::Mse => s.outputs == *per,
                        Loss::CrossEntropy => s.outputs >= *classes,
                    }
            }),
            Data::Sequences { .. } => model.transformer_spec().is_some(),
        }
    }

    /// The mean loss over `idx` (all examples for `None`); with `grad`, also the mean
    /// gradient, flat, into it (which must hold `n_params` numbers). Indices beyond the
    /// data are skipped.
    fn run(&self, model: &Model, tape: &mut Tape, idx: Option<&[usize]>, mut grad: Option<&mut [f64]>) -> f64 {
        if let Some(g) = grad.as_deref_mut() {
            g.fill(0.0);
        }
        let count = idx.map_or(self.n, |i| i.iter().filter(|&&k| k < self.n).count());
        if count == 0 {
            return 0.0;
        }
        if !self.fits(model) {
            if let Some(g) = grad {
                g.fill(f64::NAN);
            }
            return f64::NAN;
        }
        match &self.data {
            Data::Points { x, y, inputs, per, loss, .. } => {
                let (xb, yb): (Vec<f64>, Vec<f64>);
                let (xs, ys): (&[f64], &[f64]) = match idx {
                    None => (x, y),
                    Some(i) => {
                        let keep = || i.iter().copied().filter(|&k| k < self.n);
                        xb = keep().flat_map(|k| x[k * inputs..(k + 1) * inputs].iter().copied()).collect();
                        yb = keep().flat_map(|k| y[k * per..(k + 1) * per].iter().copied()).collect();
                        (&xb, &yb)
                    }
                };
                let trace = model.loss_mlp(tape, xs, ys, *loss);
                let value = tape.value(trace.loss).first().copied().unwrap_or(f64::NAN);
                if let Some(g) = grad {
                    tape.backward(trace.loss);
                    let mut at = 0;
                    for &v in &trace.params {
                        let gv = tape.grad(v);
                        let end = (at + gv.len()).min(g.len());
                        if at < end {
                            g[at..end].copy_from_slice(&gv[..end - at]);
                        }
                        at += gv.len();
                    }
                }
                value
            }
            Data::Sequences { tokens, targets, len } => {
                let weight = 1.0 / count as f64;
                let list: Vec<usize> = match idx {
                    None => (0..self.n).collect(),
                    Some(i) => i.iter().copied().filter(|&k| k < self.n).collect(),
                };
                let mut total = 0.0;
                for k in list {
                    let trace = model.loss_transformer(tape, &tokens[k * len..(k + 1) * len], &targets[k * len..(k + 1) * len]);
                    total += tape.value(trace.loss).first().copied().unwrap_or(f64::NAN);
                    if let Some(g) = grad.as_deref_mut() {
                        tape.backward(trace.loss);
                        let mut at = 0;
                        for &v in &trace.params {
                            let gv = tape.grad(v);
                            let end = (at + gv.len()).min(g.len());
                            if at < end {
                                axpy(weight, &gv[..end - at], &mut g[at..end]);
                            }
                            at += gv.len();
                        }
                    }
                }
                total * weight
            }
        }
    }

    /// The loss on all the data: the mean over every example of its loss (see
    /// [`Problem`]). Cost: the forward half of a pass over all the examples (an MLP: one
    /// batch; a transformer: one pass per sequence).
    pub fn loss(&self, model: &Model, tape: &mut Tape) -> f64 {
        self.run(model, tape, None, None)
    }

    /// The mean loss over the examples `idx` (an index listed twice counts twice; one
    /// beyond the data is skipped; none at all gives 0). Cost: the forward half of a
    /// pass over them.
    pub fn loss_on(&self, model: &Model, tape: &mut Tape, idx: &[usize]) -> f64 {
        self.run(model, tape, Some(idx), None)
    }

    /// Zero the model's accumulated gradients, then put into its groups' `grad` the MEAN
    /// gradient over the examples `idx` (each example, each sequence, weighing the
    /// same; for sequences, the mean over the sequences of the gradient of each
    /// sequence's own mean over its scored positions), and return their mean loss. No
    /// examples: zero gradient and loss 0. Cost: one pass of `idx.len()` examples for an
    /// MLP, `idx.len()` passes for a transformer. The values and Adam's state are not
    /// touched.
    pub fn grad(&self, model: &mut Model, tape: &mut Tape, idx: &[usize]) -> f64 {
        self.grad_into_model(model, tape, Some(idx))
    }

    /// [`Problem::grad`] over all the data.
    pub fn full_grad(&self, model: &mut Model, tape: &mut Tape) -> f64 {
        self.grad_into_model(model, tape, None)
    }

    fn grad_into_model(&self, model: &mut Model, tape: &mut Tape, idx: Option<&[usize]>) -> f64 {
        let mut flat = vec![0.0; model.n_params()];
        let loss = self.run(model, tape, idx, Some(&mut flat));
        let mut at = 0;
        for g in model.groups_mut() {
            let n = g.grad.len();
            g.grad.copy_from_slice(&flat[at..at + n]);
            at += n;
        }
        loss
    }

    /// The same mean gradient as [`Problem::grad`], flat, into `out` (resized to
    /// `n_params`), leaving the model untouched; returns the mean loss. Same cost.
    pub fn gradient(&self, model: &Model, tape: &mut Tape, idx: &[usize], out: &mut Vec<f64>) -> f64 {
        out.clear();
        out.resize(model.n_params(), 0.0);
        self.run(model, tape, Some(idx), Some(out))
    }

    fn gradient_on(&self, model: &Model, tape: &mut Tape, idx: Option<&[usize]>, out: &mut Vec<f64>) -> f64 {
        out.clear();
        out.resize(model.n_params(), 0.0);
        self.run(model, tape, idx, Some(out))
    }
}

// ----- mini-batches ---------------------------------------------------------------------

/// How a [`Sampler`] draws the examples of a batch.
///
/// The covariance of a mini-batch gradient differs between the two (`noise.md` D1,
/// derived there and checked numerically). With g₁ … g_N the per-example gradients, G
/// their mean, Σ = (1/N) Σₙ (gₙ − G)(gₙ − G)ᵀ their covariance over the data set
/// (divisor N) and Σₛ = N/(N − 1) Σ (divisor N − 1), the mean gradient of a batch of B
/// has mean G and covariance
///
/// - with replacement (B independent uniform draws): Σ / B, exactly (McCandlish and
///   co-authors 2018, Eq. 2.2, for independent draws);
/// - without replacement (a uniformly random subset of B distinct examples): (1/B −
///   1/N) Σₛ = (N − B) / (B (N − 1)) · Σ, exactly; zero at B = N (Jastrzębski and
///   co-authors 2018, Eqs. 6 to 7; McCandlish's footnote 4 "scales like (1/B − 1/D)").
///
/// [`noise_factor`] is the number that multiplies Σ in each case, and
/// [`effective_batch`] its reciprocal: the batch drawn with replacement that is as
/// noisy.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Draw {
    /// Epoch by epoch: the data shuffled, dealt out in batches of B distinct examples,
    /// reshuffled for the next epoch. Each single batch is a uniformly random subset of
    /// size B; successive batches of one epoch are disjoint, so not independent.
    WithoutReplacement,
    /// Every example of every batch drawn independently and uniformly from all of them
    /// (an example can appear twice in a batch).
    WithReplacement,
}

/// The number that multiplies the per-example covariance Σ (divisor N) in the
/// covariance of the mean gradient of a batch of `b` drawn from `n` examples (see
/// [`Draw`]): 1/b with replacement; (n − b) / (b (n − 1)) without, 0 once b ≥ n (and
/// for n = 1). `b` is taken as at least 1; for n = 0, NaN.
pub fn noise_factor(b: usize, n: usize, draw: Draw) -> f64 {
    if n == 0 {
        return f64::NAN;
    }
    let b = b.max(1);
    match draw {
        Draw::WithReplacement => 1.0 / b as f64,
        Draw::WithoutReplacement => {
            if b >= n {
                0.0
            } else {
                (n - b) as f64 / (b as f64 * (n - 1) as f64)
            }
        }
    }
}

/// The batch size that, drawn with replacement, is as noisy as a batch of `b` drawn
/// `draw` from `n` examples: 1 / [`noise_factor`] (b with replacement, b (n − 1) / (n −
/// b) without, infinite once b ≥ n). Every formula below written for "B independent
/// examples" is exact for a batch drawn without replacement with this B in its place
/// (derived here from `noise.md` D1: only the covariance enters them).
pub fn effective_batch(b: usize, n: usize, draw: Draw) -> f64 {
    1.0 / noise_factor(b, n, draw)
}

/// Deals out mini-batches (lists of example indices) of a given size from a seeded
/// [`Rng`], drawn as [`Draw`] says. Without replacement, an epoch deals ⌊N / B⌋ batches
/// of exactly B distinct examples from a fresh shuffle, and the N mod B examples left
/// over sit out that epoch (they are reshuffled with the rest for the next), so that
/// every batch is a uniformly random subset of size B and the formula of [`Draw`] holds
/// for it exactly. The batch size is taken as at least 1 and (without replacement) at
/// most N; with no examples every batch is empty.
#[derive(Clone, Debug)]
pub struct Sampler {
    n: usize,
    batch: usize,
    draw: Draw,
    rng: Rng,
    order: Vec<usize>,
    at: usize,
    epochs: u64,
    out: Vec<usize>,
}

impl Sampler {
    /// Batches of `batch` from `n` examples, drawn `draw`, from `rng`.
    pub fn new(n: usize, batch: usize, draw: Draw, rng: Rng) -> Sampler {
        let mut s = Sampler { n, batch, draw, rng, order: (0..n).collect(), at: 0, epochs: 0, out: Vec::new() };
        s.rng.shuffle(&mut s.order);
        s
    }

    /// The size of the batches dealt: `batch` taken as at least 1, at most N without
    /// replacement; 0 with no examples.
    pub fn size(&self) -> usize {
        match (self.n, self.draw) {
            (0, _) => 0,
            (_, Draw::WithReplacement) => self.batch.max(1),
            (n, Draw::WithoutReplacement) => self.batch.max(1).min(n),
        }
    }

    /// How the batches are drawn.
    pub fn draw(&self) -> Draw {
        self.draw
    }

    /// The number of examples drawn from.
    pub fn examples(&self) -> usize {
        self.n
    }

    /// Change the batch size (and start a new epoch).
    pub fn set_batch(&mut self, batch: usize) {
        self.batch = batch;
        self.at = self.n;
    }

    /// Epochs begun since the start, counting from 0 (without replacement).
    pub fn epoch(&self) -> u64 {
        self.epochs
    }

    /// Deal the next batch.
    pub fn deal(&mut self) -> &[usize] {
        let b = self.size();
        self.out.clear();
        match self.draw {
            Draw::WithReplacement => {
                for _ in 0..b {
                    let k = self.rng.below(self.n);
                    self.out.push(k);
                }
            }
            Draw::WithoutReplacement => {
                if self.at + b > self.n {
                    self.rng.shuffle(&mut self.order);
                    self.at = 0;
                    self.epochs += 1;
                }
                self.out.extend_from_slice(&self.order[self.at..self.at + b]);
                self.at += b;
            }
        }
        &self.out
    }
}

// ----- curvature -----------------------------------------------------------------------------

/// The radius r of the central difference in [`hvp`] for weights `w` and a direction of
/// length `vn`: the weights move by r |v| = ε^(1/3) (1 + rms(w)) along v, with ε the
/// spacing of doubles at 1 and rms(w) the root mean square of the weights. See [`hvp`].
fn fd_radius(w: &[f64], vn: f64) -> f64 {
    let rms = if w.is_empty() { 0.0 } else { (dot(w, w) / w.len() as f64).sqrt() };
    f64::EPSILON.cbrt() * (1.0 + rms) / vn
}

/// The Hessian of the mean loss over the examples `idx` times `v`, into `out` (resized
/// to `n_params`), by a central difference of two gradients: (g(w + r v) − g(w − r v)) /
/// 2r. The model is left exactly as it was (its values are restored from a copy, not by
/// subtraction; its accumulated gradients and Adam's state are not touched). Cost: two
/// gradients over `idx` (two passes of `idx.len()` examples for an MLP, 2 `idx.len()`
/// passes for a transformer); none for v = 0, whose product is 0. A shorter `v` is
/// taken as padded with zeros.
///
/// The radius: the weights move by r |v| = ε^(1/3) (1 + rms(w)) along v, where ε =
/// 2.2·10⁻¹⁶ (so ε^(1/3) = 6.1·10⁻⁶) and rms(w) is the root mean square of the weights.
/// A central difference is wrong by about (r |v|)² times the third derivative along v
/// (truncation) plus about ε |g| / (r |v|) (the rounding of the two gradients); the cube
/// root of ε balances the two when the loss's third derivative is of the size of its
/// first, and the factor keeps the step a fixed fraction of the typical weight. For a
/// loss that is exactly quadratic the truncation is nil. Measured (test
/// `hvp_accuracy_against_an_exact_hessian`): against the Hessian written out by hand
/// for a tanh network of one unit, the largest error is 7·10⁻¹¹ of the largest entry;
/// against the exact Hessian of an exactly quadratic loss (an MLP's read-out under
/// squared error), 6·10⁻¹²; on a tanh MLP of 123 parameters against a Richardson-
/// extrapolated difference (ignored test `hvp_radius_experiment`), 2 to 6·10⁻¹¹ of |Hv|,
/// within a factor of 2 of the best fixed step tried (steps from 10⁻⁷ to 3·10⁻⁴).
///
/// Where the loss has a kink (ReLU, leaky ReLU at 0) the Hessian is defined only away
/// from it: a difference whose two points straddle a unit's kink is off by about the
/// jump in the gradient divided by 2r. Smooth activations (tanh, GELU, SiLU) avoid that.
pub fn hvp(problem: &Problem, model: &mut Model, tape: &mut Tape, idx: &[usize], v: &[f64], out: &mut Vec<f64>) {
    hvp_on(problem, model, tape, Some(idx), v, out);
}

fn hvp_on(problem: &Problem, model: &mut Model, tape: &mut Tape, idx: Option<&[usize]>, v: &[f64], out: &mut Vec<f64>) {
    let p = model.n_params();
    out.clear();
    out.resize(p, 0.0);
    let v = &v[..v.len().min(p)];
    let vn = norm(v);
    if vn == 0.0 {
        return;
    }
    let mut w = Vec::with_capacity(p);
    get(model, &mut w);
    let r = fd_radius(&w, vn);
    let mut moved = w.clone();
    axpy(r, v, &mut moved);
    set(model, &moved);
    let mut up = Vec::new();
    problem.gradient_on(model, tape, idx, &mut up);
    moved.copy_from_slice(&w);
    axpy(-r, v, &mut moved);
    set(model, &moved);
    problem.gradient_on(model, tape, idx, out);
    set(model, &w);
    for (o, u) in out.iter_mut().zip(&up) {
        *o = (u - *o) / (2.0 * r);
    }
}

/// The curvature of the mean loss over `idx` along `v`: vᵀ H v / |v|² (the second
/// derivative of the loss along the unit vector of v). NaN for v = 0. Cost: one
/// Hessian-vector product (two gradients over `idx`).
pub fn curvature_along(problem: &Problem, model: &mut Model, tape: &mut Tape, idx: &[usize], v: &[f64]) -> f64 {
    let mut hv = Vec::new();
    hvp(problem, model, tape, idx, v, &mut hv);
    let v = &v[..v.len().min(hv.len())];
    dot(v, &hv[..v.len()]) / dot(v, v)
}

/// The mean loss over `idx` along a line through the present weights w: for each t of
/// `ts`, the loss at w + t `dir`, into `out` (cleared first). With `dir` a unit
/// eigenvector of the Hessian (from [`Lanczos`]), t is the distance along it. The model
/// is left exactly as it was. Cost: the forward half of a pass over `idx` per t.
pub fn directional(problem: &Problem, model: &mut Model, tape: &mut Tape, idx: &[usize], dir: &[f64], ts: &[f64], out: &mut Vec<f64>) {
    out.clear();
    let mut w = Vec::new();
    get(model, &mut w);
    let dir = &dir[..dir.len().min(w.len())];
    let mut moved = w.clone();
    for &t in ts {
        moved.copy_from_slice(&w);
        axpy(t, dir, &mut moved);
        set(model, &moved);
        out.push(problem.loss_on(model, tape, idx));
    }
    set(model, &w);
}

/// The whole Hessian of the mean loss (over all the data, or the examples given), for
/// small models: one column per Hessian-vector product ([`hvp`] along each unit vector),
/// made symmetric ((H + Hᵀ) / 2), and its eigenvalues and eigenvectors by
/// `linalg::eig_sym`. For tests, and for scenes with tiny networks.
///
/// Cost: one product (two gradients) per parameter, which [`Hessian::advance`] does a
/// few at a time; then, once, the eigen-decomposition by Jacobi rotations (measured: 68
/// ms at 123 parameters, 0.7 s at 300; it cannot be sliced) and n² numbers.
#[derive(Clone, Debug)]
pub struct Hessian {
    p: usize,
    examples: Option<Vec<usize>>,
    /// Raw products: row j is H e_j.
    columns: Vec<f64>,
    done: usize,
    matrix: Vec<f64>,
    values: Vec<f64>,
    vectors: Vec<f64>,
    asymmetry: f64,
}

impl Hessian {
    /// To be computed for `model` on all the data, a column at a time.
    pub fn new(model: &Model) -> Hessian {
        let p = model.n_params();
        Hessian { p, examples: None, columns: Vec::new(), done: 0, matrix: Vec::new(), values: Vec::new(), vectors: Vec::new(), asymmetry: f64::NAN }
    }

    /// The same for the mean loss over the examples `idx` only.
    pub fn on_examples(mut self, idx: &[usize]) -> Hessian {
        self.examples = Some(idx.to_vec());
        self
    }

    /// The whole Hessian of the loss on all the data, at once (see [`Hessian`] for the
    /// cost).
    pub fn dense(problem: &Problem, model: &mut Model, tape: &mut Tape) -> Hessian {
        let mut h = Hessian::new(model);
        h.advance(problem, model, tape, usize::MAX);
        h
    }

    /// The whole Hessian of the mean loss over the examples `idx`, at once.
    pub fn dense_on(problem: &Problem, model: &mut Model, tape: &mut Tape, idx: &[usize]) -> Hessian {
        let mut h = Hessian::new(model).on_examples(idx);
        h.advance(problem, model, tape, usize::MAX);
        h
    }

    /// Compute up to `products` more columns (one Hessian-vector product each), and when
    /// the last is done, symmetrise and decompose. Returns whether it is done. (A model
    /// with a different number of parameters starts it over.)
    pub fn advance(&mut self, problem: &Problem, model: &mut Model, tape: &mut Tape, products: usize) -> bool {
        if model.n_params() != self.p {
            *self = Hessian { examples: self.examples.take(), ..Hessian::new(model) };
        }
        let p = self.p;
        let mut e = vec![0.0; p];
        let mut col = Vec::new();
        let mut left = products;
        while self.done < p && left > 0 {
            e.fill(0.0);
            e[self.done] = 1.0;
            hvp_on(problem, model, tape, self.examples.as_deref(), &e, &mut col);
            self.columns.extend_from_slice(&col);
            self.done += 1;
            left -= 1;
        }
        if self.done == p && self.matrix.len() != p * p {
            self.finish();
        }
        self.is_done()
    }

    /// Columns until the budget is spent (at least one) or all are done; returns whether
    /// it is done.
    pub fn work(&mut self, problem: &Problem, model: &mut Model, tape: &mut Tape, budget: &Budget) -> bool {
        loop {
            let done = self.advance(problem, model, tape, 1);
            if done || budget.spent() {
                return done;
            }
        }
    }

    fn finish(&mut self) {
        let p = self.p;
        let raw = &self.columns;
        let (mut gap, mut size) = (0.0f64, 0.0f64);
        let mut m = vec![0.0; p * p];
        for i in 0..p {
            for j in 0..p {
                // raw[j * p + i] is H_ij
                let (a, b) = (raw[j * p + i], raw[i * p + j]);
                gap = gap.max((a - b).abs());
                size = size.max(a.abs());
                m[i * p + j] = 0.5 * (a + b);
            }
        }
        self.asymmetry = if size > 0.0 { gap / size } else { gap };
        let (values, vectors) = if m.iter().all(|x| x.is_finite()) { linalg::eig_sym(&m, p) } else { (vec![f64::NAN; p], vec![f64::NAN; p * p]) };
        self.matrix = m;
        self.values = values;
        self.vectors = vectors;
    }

    /// Have all the columns been computed (and the decomposition done)?
    pub fn is_done(&self) -> bool {
        self.done == self.p && self.matrix.len() == self.p * self.p
    }

    /// The number of parameters (rows and columns).
    pub fn size(&self) -> usize {
        self.p
    }

    /// The symmetrised Hessian, row-major, `size()` by `size()` (empty until done).
    pub fn matrix(&self) -> &[f64] {
        &self.matrix
    }

    /// Its eigenvalues, largest (most positive) first (empty until done).
    pub fn values(&self) -> &[f64] {
        &self.values
    }

    /// Its unit eigenvector `k` (in the order of [`Hessian::values`]), a flat vector of
    /// parameters (empty until done or out of range).
    pub fn vector(&self, k: usize) -> &[f64] {
        let p = self.p;
        self.vectors.get(k * p..(k + 1) * p).unwrap_or(&[])
    }

    /// How far the raw matrix of products was from symmetric before it was made so: the
    /// largest |H_ij − H_ji| over the largest |H_ij| (NaN until done). The exact Hessian
    /// is symmetric; this measures the products' error.
    pub fn asymmetry(&self) -> f64 {
        self.asymmetry
    }
}

/// Eigenvalues (largest first) and unit eigenvectors (laid out as `linalg::eig_sym` lays
/// them out: eigenvector k is the k-th run of n numbers) of the symmetric `n` by `n`
/// matrix `a`, by
/// Householder reduction to tridiagonal form and the implicit QL method with Wilkinson's
/// shift: the procedures tred2 and tql2 of Bowdler, Martin, Reinsch and Wilkinson
/// (Handbook for Automatic Computation II, as in EISPACK and JAMA). About 9 n³
/// operations, several times fewer than Jacobi's rotations, which matters because the
/// Lanczos estimate decomposes its projected matrix often. A matrix with an entry that is
/// not finite gives NaN; at most 60 iterations per eigenvalue.
fn sym_eigen(a: &[f64], n: usize) -> (Vec<f64>, Vec<f64>) {
    if n == 0 {
        return (Vec::new(), Vec::new());
    }
    if !a[..n * n].iter().all(|x| x.is_finite()) {
        return (vec![f64::NAN; n], vec![f64::NAN; n * n]);
    }
    // v[i * n + j] is V[i][j]; the eigenvectors end up as V's columns
    let mut v = a[..n * n].to_vec();
    let mut d = vec![0.0; n];
    let mut e = vec![0.0; n];
    // tred2: Householder reduction to tridiagonal form
    d.copy_from_slice(&v[(n - 1) * n..n * n]);
    for i in (1..n).rev() {
        let scale: f64 = d[..i].iter().map(|x| x.abs()).sum();
        let mut h = 0.0;
        if scale == 0.0 {
            e[i] = d[i - 1];
            for j in 0..i {
                d[j] = v[(i - 1) * n + j];
                v[i * n + j] = 0.0;
                v[j * n + i] = 0.0;
            }
        } else {
            for k in 0..i {
                d[k] /= scale;
                h += d[k] * d[k];
            }
            let mut f = d[i - 1];
            let mut g = if f > 0.0 { -h.sqrt() } else { h.sqrt() };
            e[i] = scale * g;
            h -= f * g;
            d[i - 1] = f - g;
            e[..i].fill(0.0);
            for j in 0..i {
                f = d[j];
                v[j * n + i] = f;
                g = e[j] + v[j * n + j] * f;
                for k in j + 1..i {
                    g += v[k * n + j] * d[k];
                    e[k] += v[k * n + j] * f;
                }
                e[j] = g;
            }
            f = 0.0;
            for j in 0..i {
                e[j] /= h;
                f += e[j] * d[j];
            }
            let hh = f / (h + h);
            for j in 0..i {
                e[j] -= hh * d[j];
            }
            for j in 0..i {
                f = d[j];
                g = e[j];
                for k in j..i {
                    v[k * n + j] -= f * e[k] + g * d[k];
                }
                d[j] = v[(i - 1) * n + j];
                v[i * n + j] = 0.0;
            }
        }
        d[i] = h;
    }
    // accumulate the transformations
    for i in 0..n - 1 {
        v[(n - 1) * n + i] = v[i * n + i];
        v[i * n + i] = 1.0;
        let h = d[i + 1];
        if h != 0.0 {
            for k in 0..=i {
                d[k] = v[k * n + i + 1] / h;
            }
            for j in 0..=i {
                let g: f64 = (0..=i).map(|k| v[k * n + i + 1] * v[k * n + j]).sum();
                for k in 0..=i {
                    v[k * n + j] -= g * d[k];
                }
            }
        }
        for k in 0..=i {
            v[k * n + i + 1] = 0.0;
        }
    }
    for j in 0..n {
        d[j] = v[(n - 1) * n + j];
        v[(n - 1) * n + j] = 0.0;
    }
    v[(n - 1) * n + n - 1] = 1.0;
    // tql2: the implicit QL method on the tridiagonal matrix (diagonal d, off-diagonal e)
    for i in 1..n {
        e[i - 1] = e[i];
    }
    e[n - 1] = 0.0;
    let (mut f, mut tst1) = (0.0, 0.0f64);
    let eps = f64::EPSILON;
    for l in 0..n {
        tst1 = tst1.max(d[l].abs() + e[l].abs());
        // the first negligible off-diagonal entry at or after l (e[n − 1] is 0)
        let mut m = l;
        while m < n - 1 && e[m].abs() > eps * tst1 {
            m += 1;
        }
        if m > l {
            for _ in 0..60 {
                let g = d[l];
                let mut p = (d[l + 1] - g) / (2.0 * e[l]);
                let mut r = p.hypot(1.0);
                if p < 0.0 {
                    r = -r;
                }
                d[l] = e[l] / (p + r);
                d[l + 1] = e[l] * (p + r);
                let dl1 = d[l + 1];
                let mut h = g - d[l];
                for x in &mut d[l + 2..] {
                    *x -= h;
                }
                f += h;
                p = d[m];
                let (mut c, mut c2, mut c3) = (1.0, 1.0, 1.0);
                let el1 = e[l + 1];
                let (mut s, mut s2) = (0.0, 0.0);
                for i in (l..m).rev() {
                    c3 = c2;
                    c2 = c;
                    s2 = s;
                    let g = c * e[i];
                    h = c * p;
                    r = p.hypot(e[i]);
                    e[i + 1] = s * r;
                    s = e[i] / r;
                    c = p / r;
                    p = c * d[i] - s * g;
                    d[i + 1] = h + s * (c * g + s * d[i]);
                    for k in 0..n {
                        let h = v[k * n + i + 1];
                        v[k * n + i + 1] = s * v[k * n + i] + c * h;
                        v[k * n + i] = c * v[k * n + i] - s * h;
                    }
                }
                p = -s * s2 * c3 * el1 * e[l] / dl1;
                e[l] = s * p;
                d[l] = c * p;
                if e[l].abs() <= eps * tst1 || e[l].is_nan() {
                    break;
                }
            }
        }
        d[l] += f;
        e[l] = 0.0;
    }
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&i, &j| d[j].total_cmp(&d[i]));
    let values = order.iter().map(|&i| d[i]).collect();
    let vectors = order.iter().flat_map(|&i| (0..n).map(move |k| (k, i))).map(|(k, i)| v[k * n + i]).collect();
    (values, vectors)
}

/// The part of Lanczos's method that does not know what the operator is: Rayleigh–Ritz
/// on a growing orthonormal basis with the operator's images of every basis vector kept,
/// thick restarts.
#[derive(Clone, Debug)]
struct Krylov {
    p: usize,
    want: usize,
    room: usize,
    basis: Vec<f64>,
    images: Vec<f64>,
    /// The projected matrix Vᵀ (A V) as computed (not yet symmetrised), kept up to date as
    /// the basis grows: entry (a, b) at a · room + b.
    t: Vec<f64>,
    j: usize,
    next: Vec<f64>,
    pending: Vec<Vec<f64>>,
    values: Vec<f64>,
    vectors: Vec<f64>,
    residuals: Vec<f64>,
    scale: f64,
    products: usize,
    fresh: bool,
    rng: Rng,
}

impl Krylov {
    fn new(p: usize, want: usize, seed: u64) -> Krylov {
        let want = want.max(1).min(p);
        let room = (2 * want + 20).min(p);
        Krylov {
            p,
            want,
            room,
            basis: Vec::new(),
            images: Vec::new(),
            t: vec![0.0; room * room],
            j: 0,
            next: Vec::new(),
            pending: Vec::new(),
            values: Vec::new(),
            vectors: Vec::new(),
            residuals: Vec::new(),
            scale: f64::NAN,
            products: 0,
            fresh: false,
            rng: Rng::new(seed),
        }
    }

    /// Take out of `x` its components along the basis (twice: classical Gram–Schmidt
    /// repeated is as good as modified), and return what is left's length.
    fn orthogonalise(&self, x: &mut [f64]) -> f64 {
        let p = self.p;
        for _ in 0..2 {
            for a in 0..self.j {
                let b = &self.basis[a * p..(a + 1) * p];
                let d = dot(b, x);
                axpy(-d, b, x);
            }
        }
        norm(x)
    }

    /// A random unit vector orthogonal to the basis, as the next direction.
    fn random_next(&mut self) -> bool {
        self.next.clear();
        if self.j >= self.p {
            return false;
        }
        for _ in 0..4 {
            let mut x: Vec<f64> = (0..self.p).map(|_| self.rng.normal() as f64).collect();
            let before = norm(&x);
            let left = self.orthogonalise(&mut x);
            if left > 1e-6 * before && left.is_finite() {
                x.iter_mut().for_each(|x| *x /= left);
                self.next = x;
                return true;
            }
        }
        false
    }

    /// One product: extend the basis by the next direction and keep its image. Returns
    /// false when nothing more can be done (the basis spans everything).
    fn step(&mut self, op: &mut impl FnMut(&[f64], &mut Vec<f64>)) -> bool {
        if self.p == 0 || self.j >= self.p {
            return false;
        }
        if self.j >= self.room {
            self.restart_thick();
        }
        let q = loop {
            if let Some(mut y) = self.pending.pop() {
                let before = norm(&y);
                let left = self.orthogonalise(&mut y);
                if left > 1e-6 * before && left.is_finite() {
                    y.iter_mut().for_each(|x| *x /= left);
                    break y;
                }
                continue;
            }
            if self.next.len() != self.p && !self.random_next() {
                return false;
            }
            break std::mem::take(&mut self.next);
        };
        let mut w = Vec::with_capacity(self.p);
        op(&q, &mut w);
        w.resize(self.p, 0.0);
        self.products += 1;
        self.basis.extend_from_slice(&q);
        self.images.extend_from_slice(&w);
        let (jj, p, room) = (self.j, self.p, self.room);
        for a in 0..=jj {
            self.t[a * room + jj] = dot(&self.basis[a * p..(a + 1) * p], &w);
            self.t[jj * room + a] = dot(&q, &self.images[a * p..(a + 1) * p]);
        }
        self.j += 1;
        self.fresh = false;
        if self.pending.is_empty() {
            // the Krylov direction: the image, less what the basis already holds
            let before = norm(&w);
            let left = self.orthogonalise(&mut w);
            if left > 1e-10 * before && left.is_finite() {
                w.iter_mut().for_each(|x| *x /= left);
                self.next = w;
            } else {
                self.next.clear();
            }
        }
        true
    }

    /// The Ritz pairs of the present basis: the eigen-decomposition of the projected
    /// matrix Vᵀ (A V), the top `count` vectors and images. Returns (all values, vectors,
    /// images).
    fn rotate(&self, count: usize) -> (Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>) {
        let (j, p, room) = (self.j, self.p, self.room);
        let mut t = vec![0.0; j * j];
        for a in 0..j {
            for b in 0..j {
                t[a * j + b] = 0.5 * (self.t[a * room + b] + self.t[b * room + a]);
            }
        }
        let count = count.min(j);
        if !t.iter().all(|x| x.is_finite()) {
            return (vec![f64::NAN; j], vec![f64::NAN; j * j], vec![f64::NAN; count * p], vec![f64::NAN; count * p]);
        }
        let (values, s) = sym_eigen(&t, j);
        let mut y = vec![0.0; count * p];
        let mut hy = vec![0.0; count * p];
        for k in 0..count {
            for a in 0..j {
                let c = s[k * j + a];
                axpy(c, &self.basis[a * p..(a + 1) * p], &mut y[k * p..(k + 1) * p]);
                axpy(c, &self.images[a * p..(a + 1) * p], &mut hy[k * p..(k + 1) * p]);
            }
        }
        (values, s, y, hy)
    }

    fn ritz(&mut self) {
        if self.fresh {
            return;
        }
        self.fresh = true;
        let p = self.p;
        if self.j == 0 {
            self.values.clear();
            self.vectors.clear();
            self.residuals.clear();
            self.scale = f64::NAN;
            return;
        }
        let (values, _, y, hy) = self.rotate(self.want);
        let count = self.want.min(self.j);
        self.residuals = (0..count)
            .map(|k| {
                let (y, hy) = (&y[k * p..(k + 1) * p], &hy[k * p..(k + 1) * p]);
                hy.iter().zip(y).map(|(a, b)| (a - values[k] * b).powi(2)).sum::<f64>().sqrt()
            })
            .collect();
        self.scale = values[0].abs().max(values[self.j - 1].abs());
        self.values = values[..count].to_vec();
        self.vectors = y;
    }

    /// Keep only the best Ritz vectors (half the room, at least those wanted) and their
    /// images, and go on from the same next direction (orthogonal to all of them).
    fn restart_thick(&mut self) {
        let keep = (self.room / 2).max(self.want).min(self.j.saturating_sub(1)).max(1);
        let (_, s, y, hy) = self.rotate(keep);
        // the new projected matrix, Sᵀ T S over the kept columns of S (from the matrix as
        // computed, so that it stays exactly Yᵀ (A Y) for the images kept)
        let (j, room) = (self.j, self.room);
        let mut ts = vec![0.0; j * keep];
        for a in 0..j {
            for c in 0..keep {
                ts[a * keep + c] = (0..j).map(|b| self.t[a * room + b] * s[c * j + b]).sum();
            }
        }
        for r in 0..keep {
            for c in 0..keep {
                self.t[r * room + c] = (0..j).map(|a| s[r * j + a] * ts[a * keep + c]).sum();
            }
        }
        self.basis = y;
        self.images = hy;
        self.j = keep;
        self.fresh = false;
    }

    fn converged(&self, tolerance: f64) -> bool {
        if self.p == 0 {
            return true;
        }
        if !self.fresh || self.values.len() < self.want {
            return false;
        }
        if self.j >= self.p {
            return self.values.iter().all(|v| v.is_finite());
        }
        self.residuals.iter().all(|&r| r <= tolerance * self.scale)
    }

    /// Start again from the present Ritz vectors (a warm start): from their sum, a single
    /// vector, so that the basis stays a Krylov space (a block of them, expanded along one
    /// of them only, would leave parts of the others' images out of the space for good,
    /// and the residuals would stall). From a random vector if there are none.
    fn warm(&mut self) {
        let p = self.p;
        let have = self.vectors.len() == self.values.len() * p && !self.values.is_empty() && self.vectors.iter().all(|x| x.is_finite());
        self.pending.clear();
        if have {
            let mut start = vec![0.0; p];
            for y in self.vectors.chunks_exact(p.max(1)) {
                axpy(1.0, y, &mut start);
            }
            self.pending.push(start);
        }
        self.clear_basis();
    }

    fn cold(&mut self) {
        self.pending.clear();
        self.clear_basis();
    }

    fn clear_basis(&mut self) {
        self.basis.clear();
        self.images.clear();
        self.j = 0;
        self.next.clear();
        self.products = 0;
        self.fresh = false;
    }
}

/// An estimate of the largest (most positive) eigenvalues of the Hessian of the loss
/// and their eigenvectors, from Hessian-vector products, that can be advanced a few
/// products at a time and warm-started from its last estimate when the weights have
/// moved a little (a scene tracks the sharpness along a training run this way).
///
/// **The sharpness** is the largest eigenvalue of the Hessian of the training loss, "the
/// maximum eigenvalue of the training loss Hessian" (Cohen and co-authors 2021, §1 and
/// footnote 2; `curvature.md` B1): the most positive one, not the largest in absolute
/// value. When the most negative eigenvalue is larger in size (a saddle, a network
/// early in training), power iteration would converge to that one; Lanczos's method
/// does not: it approximates both ends of the spectrum at once, and this takes the most
/// positive Ritz value, whose residual says when it has converged. A wide negative end
/// only slows its convergence (the rate depends on the gap below λ₁ relative to the
/// whole spread). If every eigenvalue is negative, the sharpness is negative.
///
/// **Preconditioned.** Given positive numbers p (one per parameter, what an adaptive
/// optimiser divides its step by, which the optimiser library hands out), the same for
/// P^(−1/2) H P^(−1/2) with P = diag(p): the preconditioned sharpness, "the largest
/// positive eigenvalue of a preconditioned Hessian", λ₁(P⁻¹H) = λ₁(P^(−1/2) H
/// P^(−1/2)) (Cohen and co-authors 2022, §2; `curvature.md` B2), whose stability
/// threshold for Adam is (2 + 2β₁) / ((1 − β₁) η), 38/η for β₁ = 0.9. Its eigenvectors
/// are then those of P^(−1/2) H P^(−1/2) (multiply by P^(−1/2) for a direction in
/// weight space). An entry of p that is not positive has no square root and makes the
/// estimate NaN.
///
/// **The method.** Lanczos's method with full reorthogonalisation (every new direction
/// is made orthogonal to the whole basis, twice) and Rayleigh–Ritz extraction: the
/// image H q of every basis vector q is kept, the Ritz values are the eigenvalues of
/// the projected matrix Vᵀ H V (made symmetric, by `linalg::eig_sym`), and the residual
/// |H y − θ y| of each wanted Ritz pair (θ, y) is computed from the kept images, so it
/// is the true residual of the products used, not a value the three-term recurrence
/// assumes. For a symmetric matrix, some eigenvalue lies within the residual of θ, and
/// a converged θ is usually far closer (about the residual squared over the gap to the
/// next eigenvalue). The basis holds at most 2·`want` + 20 vectors (`n_params` at most);
/// when it is full, it restarts thickly: it keeps the best half of the Ritz vectors
/// with their images (no products) and goes on along the same Krylov direction.
///
/// **Warm start.** [`Lanczos::restart`] starts the next estimate from the sum of the
/// present wanted Ritz vectors (for the sharpness alone: the top eigenvector). After a
/// small move of the weights that vector lies nearly in the wanted eigenvectors' span,
/// so the estimate converges in fewer products than from a random start (a single
/// starting vector keeps the basis a Krylov space; starting from the whole block, then
/// growing it along one vector, stalls the residuals).
///
/// Cost: one Hessian-vector product (two gradients over the examples, all the data by
/// default) per step, and 2 j dot products of `n_params` numbers to extend the projected
/// matrix (j the size of the basis); at the end of each [`Lanczos::advance`] (and every
/// fourth product in [`Lanczos::work`]) the Ritz pairs: an eigen-decomposition of the
/// projected matrix (at most 2·`want` + 20 square, by Householder reduction and the QL
/// method) and the wanted Ritz vectors (measured: 57 µs for a basis of 21 and 643
/// parameters, against 200 µs for one product there). Memory: twice the basis, 2
/// (2·`want` + 20) `n_params` numbers. Measured on a tanh MLP of 123 parameters
/// (test `lanczos_agrees_with_the_dense_hessian`): the top three eigenvalues to 10⁻⁷ in
/// 12 to 15 products from cold, agreeing with the dense Hessian's to 10⁻¹⁰; after one
/// step of 0.05 or 0.3 (top eigenvalue 3.1, the next 1.6) warm starts took 6 or 7
/// products against 9 for the sharpness, and 8 against 12 for the top three; after five
/// steps at the edge of stability (top eigenvalues 2.59, 2.15, 1.73, the top vector
/// turned to a cosine of 0.43 with the old one) 12 against 13.
#[derive(Clone, Debug)]
pub struct Lanczos {
    core: Krylov,
    /// Converged when every wanted Ritz pair's residual is at most this times the
    /// largest Ritz value in size (the estimate of |H|). 10⁻⁴ by default: then each
    /// wanted eigenvalue is certainly within 10⁻⁴ |H| of a true one, and usually much
    /// closer.
    pub tolerance: f64,
    examples: Option<Vec<usize>>,
    /// 1/√p for the preconditioned version.
    precond: Option<Vec<f64>>,
    scratch: Vec<f64>,
}

impl Lanczos {
    /// To estimate the `want` largest eigenvalues (at least 1, at most `n_params`) of
    /// the Hessian of the loss on all the data for `model`, the first direction drawn
    /// from `seed`.
    pub fn new(model: &Model, want: usize, seed: u64) -> Lanczos {
        Lanczos::of_size(model.n_params(), want, seed)
    }

    /// The same for an operator of `p` dimensions given to [`Lanczos::advance_with`]
    /// (a matrix, a model the caller differentiates).
    pub fn of_size(p: usize, want: usize, seed: u64) -> Lanczos {
        Lanczos { core: Krylov::new(p, want, seed), tolerance: 1e-4, examples: None, precond: None, scratch: Vec::new() }
    }

    /// Of the Hessian of the mean loss over the examples `idx` rather than all the data.
    pub fn on_examples(mut self, idx: &[usize]) -> Lanczos {
        self.examples = Some(idx.to_vec());
        self
    }

    /// Of P^(−1/2) H P^(−1/2) with P = diag(`p`) (the preconditioned sharpness).
    pub fn preconditioned(mut self, p: &[f64]) -> Lanczos {
        self.precond = Some(p.iter().map(|&p| 1.0 / p.sqrt()).collect());
        self
    }

    /// A new preconditioner (an adaptive optimiser's changes every step): the operator
    /// has changed, so this also restarts, warm.
    pub fn set_preconditioner(&mut self, p: &[f64]) {
        let d = self.precond.get_or_insert_with(Vec::new);
        d.clear();
        d.extend(p.iter().map(|&p| 1.0 / p.sqrt()));
        self.restart();
    }

    /// Start a new estimate from the present one (after the weights have moved): its
    /// first direction is the sum of the wanted Ritz vectors. The count of products
    /// starts again from 0; the old values stay readable until the first new Ritz
    /// values (at the end of the next `advance` or `work`).
    pub fn restart(&mut self) {
        self.core.warm();
    }

    /// Start a new estimate from a random direction.
    pub fn reset(&mut self) {
        self.core.cold();
    }

    fn sync(&mut self, model: &Model) {
        if model.n_params() != self.core.p {
            let seed = self.core.rng.next_u64();
            self.core = Krylov::new(model.n_params(), self.core.want, seed);
        }
    }

    fn step(&mut self, problem: &Problem, model: &mut Model, tape: &mut Tape) -> bool {
        let examples = self.examples.as_deref();
        let precond = self.precond.as_deref();
        let scratch = &mut self.scratch;
        self.core.step(&mut |v: &[f64], out: &mut Vec<f64>| match precond {
            None => hvp_on(problem, model, tape, examples, v, out),
            Some(d) => {
                scratch.clear();
                scratch.extend(v.iter().zip(d.iter().chain(std::iter::repeat(&f64::NAN))).map(|(v, d)| v * d));
                hvp_on(problem, model, tape, examples, scratch, out);
                for (o, d) in out.iter_mut().zip(d.iter().chain(std::iter::repeat(&f64::NAN))) {
                    *o *= d;
                }
            }
        })
    }

    /// Up to `products` more Hessian-vector products (none once converged), then the
    /// Ritz values. Returns whether the estimate has converged. (A model with a
    /// different number of parameters starts it over, cold.)
    pub fn advance(&mut self, problem: &Problem, model: &mut Model, tape: &mut Tape, products: usize) -> bool {
        self.sync(model);
        if self.converged() {
            return true;
        }
        for _ in 0..products {
            if !self.step(problem, model, tape) {
                break;
            }
        }
        self.core.ritz();
        self.converged()
    }

    /// Products until converged or the budget is spent (at least one, unless already
    /// converged), checking for convergence every fourth product. Returns whether it has
    /// converged.
    pub fn work(&mut self, problem: &Problem, model: &mut Model, tape: &mut Tape, budget: &Budget) -> bool {
        self.sync(model);
        if self.converged() {
            return true;
        }
        loop {
            let more = self.step(problem, model, tape);
            if !more || budget.spent() {
                break;
            }
            if self.core.products.is_multiple_of(4) {
                self.core.ritz();
                if self.converged() {
                    return true;
                }
            }
        }
        self.core.ritz();
        self.converged()
    }

    /// Up to `products` more applications of any symmetric operator of `p` dimensions
    /// (the one given to [`Lanczos::of_size`]): `op(v, out)` puts A v into `out`. The
    /// examples and preconditioner are ignored. Returns whether it has converged.
    pub fn advance_with(&mut self, products: usize, mut op: impl FnMut(&[f64], &mut Vec<f64>)) -> bool {
        if self.converged() {
            return true;
        }
        for _ in 0..products {
            if !self.core.step(&mut op) {
                break;
            }
        }
        self.core.ritz();
        self.converged()
    }

    /// The sharpness: the largest Ritz value (NaN before the first product).
    pub fn sharpness(&self) -> f64 {
        self.core.values.first().copied().unwrap_or(f64::NAN)
    }

    /// The wanted Ritz values, largest first (fewer while the basis is smaller than
    /// that).
    pub fn values(&self) -> &[f64] {
        &self.core.values
    }

    /// The unit Ritz vector of value `k`, a flat vector of parameters (empty if there is
    /// none).
    pub fn vector(&self, k: usize) -> &[f64] {
        let p = self.core.p;
        self.core.vectors.get(k * p..(k + 1) * p).unwrap_or(&[])
    }

    /// The residual |H y − θ y| of each wanted Ritz pair: some eigenvalue lies within it
    /// of θ.
    pub fn residuals(&self) -> &[f64] {
        &self.core.residuals
    }

    /// How far the estimate is from converged: the largest residual of the wanted
    /// pairs over the largest Ritz value in size (compare with `tolerance`). 0 once the
    /// basis spans every direction; NaN before the first product.
    pub fn residual(&self) -> f64 {
        if self.core.j >= self.core.p && self.core.fresh {
            return 0.0;
        }
        if self.core.residuals.is_empty() {
            return f64::NAN;
        }
        let r = self.core.residuals.iter().fold(0.0f64, |a, &b| if a.is_nan() || b.is_nan() { f64::NAN } else { a.max(b) });
        r / self.core.scale
    }

    /// Products since the estimate was (re)started.
    pub fn products(&self) -> usize {
        self.core.products
    }

    /// Has the estimate converged (see `tolerance`)?
    pub fn converged(&self) -> bool {
        self.core.converged(self.tolerance)
    }
}

/// The curvature of mini-batches along their own gradients: Andreyev and Beneventano's
/// "batch sharpness" (2024/2025, Def. 3, with the expectation outside, as their v5 and
/// their experiments have it; `curvature.md` C3),
///
///   E over mini-batches B of [ ∇L_B(θ)ᵀ H(L_B) ∇L_B(θ) / |∇L_B(θ)|² ],
///
/// "the expected directional curvature of mini-batch Hessians along their corresponding
/// stochastic gradients": for each batch, the Hessian of that batch's own mean loss
/// along that batch's own gradient, averaged over batches drawn as training draws them
/// (the [`Sampler`] it is given). Its source finds that it, not the full-batch sharpness,
/// reaches 2/η and hovers there under SGD, for image classification with squared error.
///
/// Estimated by the mean over sampled batches (with its standard error), advanced a
/// few batches at a time. A batch whose gradient is exactly zero has no direction and is
/// skipped (counted in `skipped`). Cost per batch: three gradients over the batch (its
/// gradient, and a Hessian-vector product).
#[derive(Clone, Debug)]
pub struct BatchSharpness {
    sampler: Sampler,
    count: usize,
    skipped: usize,
    mean: f64,
    m2: f64,
    idx: Vec<usize>,
    g: Vec<f64>,
    hg: Vec<f64>,
}

impl BatchSharpness {
    /// To average over the batches `sampler` deals.
    pub fn new(sampler: Sampler) -> BatchSharpness {
        BatchSharpness { sampler, count: 0, skipped: 0, mean: 0.0, m2: 0.0, idx: Vec::new(), g: Vec::new(), hg: Vec::new() }
    }

    /// Measure `batches` more batches. Returns the number measured so far.
    pub fn advance(&mut self, problem: &Problem, model: &mut Model, tape: &mut Tape, batches: usize) -> usize {
        for _ in 0..batches {
            self.idx.clear();
            self.idx.extend_from_slice(self.sampler.deal());
            problem.gradient(model, tape, &self.idx, &mut self.g);
            let gg = dot(&self.g, &self.g);
            if gg == 0.0 {
                self.skipped += 1;
                continue;
            }
            hvp(problem, model, tape, &self.idx, &self.g, &mut self.hg);
            let x = dot(&self.g, &self.hg) / gg;
            // Welford's running mean and sum of squared deviations
            self.count += 1;
            let d = x - self.mean;
            self.mean += d / self.count as f64;
            self.m2 += d * (x - self.mean);
        }
        self.count
    }

    /// Batches until the budget is spent (at least one). Returns the number measured.
    pub fn work(&mut self, problem: &Problem, model: &mut Model, tape: &mut Tape, budget: &Budget) -> usize {
        loop {
            self.advance(problem, model, tape, 1);
            if budget.spent() {
                return self.count;
            }
        }
    }

    /// The estimate: the mean over the batches measured (NaN before the first).
    pub fn mean(&self) -> f64 {
        if self.count == 0 {
            f64::NAN
        } else {
            self.mean
        }
    }

    /// The standard error of the mean: the spread of the batches' values over √count
    /// (NaN with fewer than two). Successive batches of one epoch without replacement are
    /// not independent, so this is approximate for them.
    pub fn standard_error(&self) -> f64 {
        if self.count < 2 {
            f64::NAN
        } else {
            (self.m2 / (self.count - 1) as f64 / self.count as f64).sqrt()
        }
    }

    /// Batches measured.
    pub fn count(&self) -> usize {
        self.count
    }

    /// Batches skipped for a zero gradient.
    pub fn skipped(&self) -> usize {
        self.skipped
    }

    /// Forget the batches measured (after the weights have moved).
    pub fn reset(&mut self) {
        self.count = 0;
        self.skipped = 0;
        self.mean = 0.0;
        self.m2 = 0.0;
    }
}

// ----- noise -----------------------------------------------------------------------------

/// The gradient of every example on its own, kept as rows (one flat vector of
/// parameters per example), at the weights the model has while it is measured: the
/// data of everything about mini-batch noise. One pass of one example each, advanced a
/// slice at a time; the model is not changed. Restart it (`reset`) after the weights
/// move. Memory: `len() · n_params` numbers.
#[derive(Clone, Debug)]
pub struct PerExample {
    n: usize,
    p: usize,
    rows: Vec<f64>,
    losses: Vec<f64>,
}

impl PerExample {
    /// To measure every example of `problem` for `model`.
    pub fn new(problem: &Problem, model: &Model) -> PerExample {
        PerExample { n: problem.len(), p: model.n_params(), rows: Vec::new(), losses: Vec::new() }
    }

    /// Forget the rows, to measure again (at new weights).
    pub fn reset(&mut self) {
        self.rows.clear();
        self.losses.clear();
    }

    /// Up to `passes` more examples (one pass each). Returns whether every example is
    /// done. (A model with a different number of parameters starts it over.)
    pub fn advance(&mut self, problem: &Problem, model: &Model, tape: &mut Tape, passes: usize) -> bool {
        if model.n_params() != self.p || problem.len() != self.n {
            *self = PerExample::new(problem, model);
        }
        let p = self.p;
        let mut left = passes;
        while self.len() < self.n && left > 0 {
            let k = self.len();
            let at = self.rows.len();
            self.rows.resize(at + p, 0.0);
            let loss = problem.run(model, tape, Some(&[k]), Some(&mut self.rows[at..at + p]));
            self.losses.push(loss);
            left -= 1;
        }
        self.is_done()
    }

    /// Examples until the budget is spent (at least one) or all are done.
    pub fn work(&mut self, problem: &Problem, model: &Model, tape: &mut Tape, budget: &Budget) -> bool {
        loop {
            let done = self.advance(problem, model, tape, 1);
            if done || budget.spent() {
                return done;
            }
        }
    }

    /// Has every example been measured?
    pub fn is_done(&self) -> bool {
        self.len() == self.n
    }

    /// Examples measured so far.
    pub fn len(&self) -> usize {
        self.losses.len()
    }

    /// None measured yet?
    pub fn is_empty(&self) -> bool {
        self.losses.is_empty()
    }

    /// The number of parameters (the length of a row).
    pub fn width(&self) -> usize {
        self.p
    }

    /// The gradient of example `k` alone (empty if not measured).
    pub fn row(&self, k: usize) -> &[f64] {
        self.rows.get(k * self.p..(k + 1) * self.p).unwrap_or(&[])
    }

    /// The loss of example `k` alone (NaN if not measured).
    pub fn loss(&self, k: usize) -> f64 {
        self.losses.get(k).copied().unwrap_or(f64::NAN)
    }

    /// The statistics of the rows measured so far, over every parameter (once all are
    /// done: of the data set). Cost: two sweeps over the rows, no passes.
    pub fn noise(&self) -> Noise {
        self.noise_in(0..self.p)
    }

    /// The same for the parameters `range` alone (one group: [`span`]), for the scenes
    /// that ask them of a single matrix.
    pub fn noise_in(&self, range: Range<usize>) -> Noise {
        let range = range.start.min(self.p)..range.end.min(self.p).max(range.start.min(self.p));
        let (n, d) = (self.len(), range.len());
        let mut mean = vec![0.0; d];
        for k in 0..n {
            axpy(1.0, &self.row(k)[range.clone()], &mut mean);
        }
        if n > 0 {
            mean.iter_mut().for_each(|m| *m /= n as f64);
        }
        let mut variances = vec![0.0; d];
        for k in 0..n {
            for ((v, &x), &m) in variances.iter_mut().zip(&self.row(k)[range.clone()]).zip(&mean) {
                *v += (x - m) * (x - m);
            }
        }
        if n > 0 {
            variances.iter_mut().for_each(|v| *v /= n as f64);
        }
        let g2 = dot(&mean, &mean);
        let trace = variances.iter().sum();
        Noise { n, mean, variances, g2, trace }
    }

    /// The mean of the rows `batch` (positions of examples; one not measured is skipped)
    /// over the parameters `range`, into `out`: the mini-batch gradient of that batch
    /// there, without a pass. Empty batch: zeros.
    pub fn batch_mean(&self, batch: &[usize], range: Range<usize>, out: &mut Vec<f64>) {
        let range = range.start.min(self.p)..range.end.min(self.p).max(range.start.min(self.p));
        out.clear();
        out.resize(range.len(), 0.0);
        let mut count = 0;
        for &k in batch {
            if k < self.len() {
                axpy(1.0, &self.row(k)[range.clone()], out);
                count += 1;
            }
        }
        if count > 0 {
            out.iter_mut().for_each(|x| *x /= count as f64);
        }
    }

    /// Draw `batches` mini-batches of `b` as `draw` says (from `rng`) from the rows and
    /// measure, over the parameters `range`, the mean and standard error of |g_B|² and of
    /// the cosine between g_B and G (the mean of all rows): the sampled counterparts of
    /// [`Noise::expected_sq_norm`] and [`Noise::expected_cosine`]. No passes; about
    /// `batches · b · range.len()` operations. A batch with g_B = 0 has cosine NaN and is
    /// left out of the cosine's mean.
    pub fn sample_in(&self, range: Range<usize>, b: usize, draw: Draw, batches: usize, rng: Rng) -> Sampled {
        let g = self.noise_in(range.clone()).mean;
        let gn = norm(&g);
        let mut sampler = Sampler::new(self.len(), b, draw, rng);
        let (mut sq, mut cos) = (Running::default(), Running::default());
        let (mut gb, mut batch) = (Vec::new(), Vec::new());
        for _ in 0..batches {
            batch.clear();
            batch.extend_from_slice(sampler.deal());
            self.batch_mean(&batch, range.clone(), &mut gb);
            let s = dot(&gb, &gb);
            sq.push(s);
            let c = dot(&gb, &g) / (s.sqrt() * gn);
            if !c.is_nan() {
                cos.push(c);
            }
        }
        Sampled { sq_norm: sq.mean(), sq_norm_error: sq.error(), cosine: cos.mean(), cosine_error: cos.error(), batches: sq.n }
    }

    /// [`PerExample::sample_in`] over every parameter.
    pub fn sample(&self, b: usize, draw: Draw, batches: usize, rng: Rng) -> Sampled {
        self.sample_in(0..self.p, b, draw, batches, rng)
    }
}

/// A running mean and sum of squared deviations (Welford).
#[derive(Clone, Copy, Debug, Default)]
struct Running {
    n: usize,
    mean: f64,
    m2: f64,
}

impl Running {
    fn push(&mut self, x: f64) {
        self.n += 1;
        let d = x - self.mean;
        self.mean += d / self.n as f64;
        self.m2 += d * (x - self.mean);
    }

    fn mean(&self) -> f64 {
        if self.n == 0 {
            f64::NAN
        } else {
            self.mean
        }
    }

    fn error(&self) -> f64 {
        if self.n < 2 {
            f64::NAN
        } else {
            (self.m2 / (self.n - 1) as f64 / self.n as f64).sqrt()
        }
    }
}

/// What [`PerExample::sample`] measured over sampled mini-batches.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Sampled {
    /// The mean of |g_B|².
    pub sq_norm: f64,
    /// Its standard error (the spread over √batches; approximate without replacement,
    /// where the batches of one epoch are not independent).
    pub sq_norm_error: f64,
    /// The mean cosine between g_B and G.
    pub cosine: f64,
    /// Its standard error.
    pub cosine_error: f64,
    /// Batches drawn.
    pub batches: usize,
}

/// The statistics of per-example gradients over a set of parameters (all, or one
/// group), from [`PerExample::noise`]: the mean gradient G, its squared length, the
/// per-example covariance's trace and diagonal, and what McCandlish and co-authors
/// (2018) build from them.
///
/// **Divisor.** The covariance here is Σ = (1/N) Σₙ (gₙ − G)(gₙ − G)ᵀ, divisor N: the
/// covariance of one example's gradient when the example is drawn uniformly from the
/// data set, which is McCandlish's Σ ("cov_{x∼ρ}", Eq. 2.3; `noise.md` §1.1) when ρ is
/// the data set itself, the distribution a mini-batch is drawn from. With that Σ the
/// covariance of a batch drawn with replacement is exactly Σ/B. The divisor N − 1
/// ([`Noise::trace_sample`]) is the one with which the without-replacement formula
/// takes the form (1/B − 1/N) Σₛ, and the unbiased estimate of the covariance of a
/// larger population the N examples were drawn from; the formulas here use whichever is
/// exact (see [`Draw`]).
#[derive(Clone, Debug, PartialEq)]
pub struct Noise {
    /// Examples (rows) the statistics are of.
    pub n: usize,
    /// G: the mean of the per-example gradients (the full gradient of the data set).
    pub mean: Vec<f64>,
    /// Each coordinate's variance over the examples, divisor N.
    pub variances: Vec<f64>,
    /// |G|².
    pub g2: f64,
    /// tr Σ: the sum of the variances, divisor N.
    pub trace: f64,
}

impl Noise {
    /// tr Σₛ = N/(N − 1) tr Σ: the trace with divisor N − 1 (NaN for fewer than 2
    /// examples).
    pub fn trace_sample(&self) -> f64 {
        if self.n < 2 {
            f64::NAN
        } else {
            self.trace * self.n as f64 / (self.n - 1) as f64
        }
    }

    /// The simple noise scale B_simple = tr Σ / |G|² (McCandlish and co-authors 2018,
    /// Eq. 2.9, `noise.md` §1.3: "the sum of the variances of the individual gradient
    /// components, divided by the global norm of the gradient", the norm squared), with
    /// Σ of divisor N (see [`Noise`]). It is a batch size: the one at which, with
    /// replacement, the expected squared length of a mini-batch gradient is half signal
    /// and half noise (`noise.md` D3; their Eq. 2.10 with the ratio 1). Their source
    /// derives it from the full noise scale ([`FullNoise`]) under "the (unrealistic)
    /// assumption ... that the Hessian is a multiple of the identity matrix". NaN when G
    /// and Σ are both zero; infinite when only G is.
    pub fn simple_noise_scale(&self) -> f64 {
        self.trace / self.g2
    }

    /// The batch size at which noise and signal are equal in E |g_B|², drawn `draw` from
    /// the N examples: B_simple with replacement; without, (1/B_simple' + 1/N)⁻¹ with
    /// B_simple' = tr Σₛ / |G|², which is smaller (`noise.md` D3, derived there).
    pub fn half_noise_batch(&self, draw: Draw) -> f64 {
        match draw {
            Draw::WithReplacement => self.simple_noise_scale(),
            Draw::WithoutReplacement => 1.0 / (self.g2 / self.trace_sample() + 1.0 / self.n as f64),
        }
    }

    /// The expected squared length of the mean gradient of a batch of `b` drawn `draw`
    /// from the N examples: |G|² + tr Σ / B with replacement (McCandlish and co-authors
    /// 2018, Eq. A.1), |G|² + (1/B − 1/N) tr Σₛ without (`noise.md` D1); `b` taken as
    /// at least 1 (and at most N without replacement, where the noise is then 0). Exact.
    pub fn expected_sq_norm(&self, b: usize, draw: Draw) -> f64 {
        self.g2 + noise_factor(b, self.n, draw) * self.trace
    }

    /// The expected cosine between a mini-batch gradient g_B (batch of `b`, drawn
    /// `draw`) and G, approximately: |G| / √(E |g_B|²) = 1 / √(1 + tr Cov(g_B) / |G|²),
    /// with replacement 1 / √(1 + B_simple / B). Derived here: g_B = G + ξ with E ξ = 0,
    /// so E[Gᵀ g_B] = |G|² exactly and E |g_B|² = |G|² + tr Cov(ξ) exactly; the
    /// approximation replaces |g_B| in the cosine Gᵀ g_B / (|G| |g_B|) by √(E |g_B|²). It
    /// is good when |g_B| varies little from batch to batch, i.e. when the noise is
    /// spread over many directions and many examples, and it is not a bound either way.
    /// Measured against sampling (test `the_cosine_approximation_against_sampling`, a tanh
    /// MLP of 643 parameters on 64 spiral points, at its start, B_simple 57): within 0.04
    /// over all parameters and within 0.07 for one 16 × 16 weight matrix (B_simple 15),
    /// too high for small batches (one layer at B = 2: 0.34 against a sampled 0.27) and
    /// too low for large ones; after training, with B_simple near 5·10⁴ and every cosine
    /// below 0.07, within 0.02, but at B = 1 one layer's sampled mean cosine was
    /// negative (−0.013) where this gives +0.007. For the value itself by sampling (no
    /// passes) use [`PerExample::sample`].
    pub fn expected_cosine(&self, b: usize, draw: Draw) -> f64 {
        (self.g2 / self.expected_sq_norm(b, draw)).sqrt()
    }

    /// One coordinate's own simple noise scale, its variance over its squared mean,
    /// σᵢ² / Gᵢ² (the batch size above which its mini-batch mean has the sign of Gᵢ
    /// with some reliability; `noise.md` D5). NaN out of range.
    pub fn coordinate_noise_scale(&self, i: usize) -> f64 {
        match (self.variances.get(i), self.mean.get(i)) {
            (Some(v), Some(m)) => v / (m * m),
            _ => f64::NAN,
        }
    }
}

/// The estimates of |G|² and tr Σ from the squared lengths of two mini-batch gradients
/// of sizes `b_small` and `b_big` (McCandlish and co-authors 2018, App. A.1, Eq. A.2;
/// `noise.md` §1.5), as plain numbers:
///
///   |𝒢|² = (B_big |G_big|² − B_small |G_small|²) / (B_big − B_small),
///   𝒮 = (|G_small|² − |G_big|²) / (1/B_small − 1/B_big).
///
/// Returns (|𝒢|², 𝒮). Each is unbiased, E |𝒢|² = |G|² and E 𝒮 = tr Σ, when the examples
/// of the batches are drawn independently (with replacement, or B ≪ N), because then E
/// |G_B|² = |G|² + tr Σ / B (their Eq. A.1). Their ratio 𝒮 / |𝒢|² estimates B_simple but
/// "is not an unbiased estimator" (their footnote 10); they average 𝒮 and |𝒢|²
/// separately over steps and divide the averages. Equal batch sizes give NaN.
pub fn two_batch_estimates(b_small: f64, sq_small: f64, b_big: f64, sq_big: f64) -> (f64, f64) {
    let g2 = (b_big * sq_big - b_small * sq_small) / (b_big - b_small);
    let s = (sq_small - sq_big) / (1.0 / b_small - 1.0 / b_big);
    (g2, s)
}

/// The full gradient noise scale B_noise = tr(H Σ) / (Gᵀ H G) of McCandlish and
/// co-authors (2018, Eq. 2.8; `noise.md` §1.2), with H the Hessian of the loss on all
/// the data, G its gradient and Σ the per-example covariance (divisor N, as [`Noise`]),
/// measured exactly from Hessian-vector products: Gᵀ H G from one product, and tr(H Σ)
/// = (1/N) Σₙ dₙᵀ H dₙ, dₙ = gₙ − G, from one product per example. It holds its own
/// [`PerExample`] and measures that first.
///
/// What it is: in their model of one SGD step ([`OneStep`]), the batch size at which the
/// best possible improvement in one step is half the full-batch maximum; "independent of
/// the size of the full training set". It equals B_simple when H is a multiple of the
/// identity. When Gᵀ H G is not positive (the loss is flat or curves down along the
/// gradient: a saddle, a concave stretch), the one-step model has no largest useful step
/// along the full gradient (its εmax = |G|² / Gᵀ H G is negative or infinite) and B_noise
/// is not a batch size: it is returned as computed (negative, or infinite), and
/// [`OneStep`] still answers from the raw quantities where they make sense.
///
/// Cost: one pass of one example per example (the rows), then N + 1 Hessian-vector
/// products on all the data (two gradients each). [`FullNoise::advance`] counts both
/// kinds of unit alike.
#[derive(Clone, Debug)]
pub struct FullNoise {
    per: PerExample,
    noise: Option<Noise>,
    ghg: f64,
    sum: f64,
    k: usize,
    started: bool,
}

impl FullNoise {
    /// To measure `problem` for `model` at its present weights.
    pub fn new(problem: &Problem, model: &Model) -> FullNoise {
        FullNoise { per: PerExample::new(problem, model), noise: None, ghg: f64::NAN, sum: 0.0, k: 0, started: false }
    }

    /// Up to `units` more units of work: per-example passes first, then products.
    /// Returns whether it is done. (The weights must not move in between.)
    pub fn advance(&mut self, problem: &Problem, model: &mut Model, tape: &mut Tape, units: usize) -> bool {
        let mut left = units;
        while left > 0 && !self.per.is_done() {
            self.per.advance(problem, model, tape, 1);
            left -= 1;
        }
        if !self.per.is_done() {
            return false;
        }
        let noise = self.noise.get_or_insert_with(|| self.per.noise()).clone();
        let mut hv = Vec::new();
        let mut d = vec![0.0; noise.mean.len()];
        while left > 0 && !self.is_done() {
            if !self.started {
                hvp_on(problem, model, tape, None, &noise.mean, &mut hv);
                self.ghg = dot(&noise.mean, &hv);
                self.started = true;
            } else {
                d.copy_from_slice(self.per.row(self.k));
                axpy(-1.0, &noise.mean, &mut d);
                hvp_on(problem, model, tape, None, &d, &mut hv);
                self.sum += dot(&d, &hv);
                self.k += 1;
            }
            left -= 1;
        }
        self.is_done()
    }

    /// Units until the budget is spent (at least one) or it is done.
    pub fn work(&mut self, problem: &Problem, model: &mut Model, tape: &mut Tape, budget: &Budget) -> bool {
        loop {
            let done = self.advance(problem, model, tape, 1);
            if done || budget.spent() {
                return done;
            }
        }
    }

    /// Done: every example's row and every product?
    pub fn is_done(&self) -> bool {
        self.per.is_done() && self.started && self.k == self.per.len()
    }

    /// The per-example gradients it measured (for [`Noise`] and sampling).
    pub fn per_example(&self) -> &PerExample {
        &self.per
    }

    /// Gᵀ H G (NaN until measured).
    pub fn ghg(&self) -> f64 {
        self.ghg
    }

    /// tr(H Σ), Σ of divisor N (NaN until done).
    pub fn trace_hs(&self) -> f64 {
        if self.is_done() && self.k > 0 {
            self.sum / self.k as f64
        } else if self.is_done() {
            0.0
        } else {
            f64::NAN
        }
    }

    /// B_noise = tr(H Σ) / (Gᵀ H G) (NaN until done).
    pub fn noise_scale(&self) -> f64 {
        self.trace_hs() / self.ghg
    }

    /// The one-step model at these weights (NaN until done).
    pub fn one_step(&self) -> OneStep {
        let g2 = if self.is_done() { self.noise.as_ref().map_or(f64::NAN, |n| n.g2) } else { f64::NAN };
        OneStep { g2, ghg: self.ghg, trhs: self.trace_hs() }
    }
}

/// McCandlish and co-authors' model of one SGD step (2018, §2.2, Eqs. 2.4 to 2.8;
/// `noise.md` §1.2, with the derivation written out there), as plain functions of three
/// numbers measured at the present weights: the step θ → θ − ε G_est with G_est the mean
/// gradient of a batch of B independent examples, the loss replaced by its second-order
/// expansion, so that
///
///   E[L(θ − ε G_est)] − L(θ) = −ε |G|² + ½ ε² (Gᵀ H G + tr(H Σ) / B)   (Eq. 2.5).
///
/// Its assumptions, as the paper lists them (§2.4): a greedy, strictly local picture;
/// the quadratic approximation; plain SGD (no momentum, no preconditioner); the best
/// step taken at each batch size; H, G, Σ at one point; only the mean and covariance of
/// the noise matter (no Gaussianity is needed). "Despite the many unfounded assumptions
/// in the above derivation, we will find that equations 2.7 and 2.8 provide a helpful
/// guide". For a batch drawn without replacement, put [`effective_batch`] for B (exact,
/// since only the covariance enters). Where Gᵀ H G + tr(H Σ)/B is not positive (possible
/// only where the Hessian is not positive semi-definite), the expected change has no
/// minimum: the model's loss falls without bound as the step grows, and the best step
/// and gain are infinite.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct OneStep {
    /// |G|².
    pub g2: f64,
    /// Gᵀ H G.
    pub ghg: f64,
    /// tr(H Σ), Σ the per-example covariance (divisor N).
    pub trhs: f64,
}

impl OneStep {
    /// The expected change of the loss from one step of `lr` with a batch of `b`
    /// (Eq. 2.5): −lr |G|² + ½ lr² (Gᵀ H G + tr(H Σ)/b). Negative is an improvement.
    pub fn change(&self, lr: f64, b: f64) -> f64 {
        -lr * self.g2 + 0.5 * lr * lr * (self.ghg + self.trhs / b)
    }

    /// The curvature term c(B) = Gᵀ H G + tr(H Σ)/B.
    fn c(&self, b: f64) -> f64 {
        self.ghg + self.trhs / b
    }

    /// The best step with the true gradient, εmax = |G|² / Gᵀ H G (§2.2). Infinite when
    /// Gᵀ H G ≤ 0 and |G| > 0; 0 when G = 0.
    pub fn max_step(&self) -> f64 {
        self.best_step(f64::INFINITY)
    }

    /// The largest one-step improvement, ΔLmax = ½ |G|⁴ / Gᵀ H G (Eq. 2.7), positive.
    pub fn max_gain(&self) -> f64 {
        self.best_gain(f64::INFINITY)
    }

    /// The full noise scale B_noise = tr(H Σ) / Gᵀ H G (Eq. 2.8), as computed (see
    /// [`FullNoise`] for what a denominator that is not positive means).
    pub fn noise_scale(&self) -> f64 {
        self.trhs / self.ghg
    }

    /// The best step at batch size `b`, εopt(B) = |G|² / (Gᵀ H G + tr(H Σ)/B) =
    /// εmax / (1 + B_noise / B) (Eq. 2.6). Infinite where the denominator is not
    /// positive; 0 when G = 0. A step larger than twice it makes the expected loss rise.
    pub fn best_step(&self, b: f64) -> f64 {
        let c = self.c(b);
        if self.g2 == 0.0 {
            0.0
        } else if c > 0.0 {
            self.g2 / c
        } else if c.is_nan() || self.g2.is_nan() {
            f64::NAN
        } else {
            f64::INFINITY
        }
    }

    /// The best one-step improvement at batch size `b` (a decrease, positive),
    /// ΔLopt(B) = ½ |G|⁴ / (Gᵀ H G + tr(H Σ)/B) = ΔLmax / (1 + B_noise / B) (Eq. 2.7).
    pub fn best_gain(&self, b: f64) -> f64 {
        0.5 * self.g2 * self.best_step(b)
    }

    /// The steps needed at batch size `b` over the fewest possible (at a batch far above
    /// the noise scale), 1 + B_noise / B, if the noise scale stayed what it is here for
    /// the whole run: [`steps_over_min`] with B_crit = B_noise (McCandlish's model
    /// predicts B_crit ≈ B_noise suitably averaged over training, §2.3 and App. D).
    pub fn steps_ratio(&self, b: f64) -> f64 {
        steps_over_min(b, self.noise_scale())
    }

    /// The examples needed at batch size `b` over the fewest possible (at a batch far
    /// below the noise scale), 1 + B / B_noise, likewise.
    pub fn examples_ratio(&self, b: f64) -> f64 {
        examples_over_min(b, self.noise_scale())
    }
}

/// The steps a run needs at batch size `b`, over the fewest possible S_min, in
/// McCandlish and co-authors' model of a whole run: S / S_min = 1 + B_crit / B, with
/// B_crit = E_min / S_min their critical batch size (2018, Eqs. 2.11 to 2.12 and App. D,
/// for a batch size fixed through the run; derived in `noise.md` §1.4 and D4). With
/// [`examples_over_min`] it satisfies (S/S_min − 1)(E/E_min − 1) = 1; at B = B_crit both
/// are 2.
pub fn steps_over_min(b: f64, b_crit: f64) -> f64 {
    1.0 + b_crit / b
}

/// The examples a run needs at batch size `b` over the fewest possible E_min: E / E_min
/// = 1 + B / B_crit (as [`steps_over_min`]).
pub fn examples_over_min(b: f64, b_crit: f64) -> f64 {
    1.0 + b / b_crit
}

// ----- the noisy quadratic model ---------------------------------------------------------------

/// The dimension of the noisy quadratic model's standard spectrum in its source:
/// d = 10⁴ (Zhang and co-authors 2019, §3.5; `noise.md` §4.1).
pub const NQM_DIM: usize = 10_000;

/// The target risk of its source's simulations (§4.3: "target risk 0.01").
pub const NQM_TARGET: f64 = 0.01;

/// One update rule of the noisy quadratic model, per direction i with curvature hᵢ:
///
///   m(t+1) = β m(t) + hᵢ θ(t) + √(cᵢ/B) ε(t),
///   θ(t+1) = θ(t) − α hᵢ^(−p) m(t+1),
///   θ̃(t+1) = γ θ̃(t) + (1 − γ) θ(t+1),
///
/// with ε(t) independent, mean 0, variance 1; m(0) = 0 and θ̃(0) = θ(0), and the loss
/// evaluated at θ̃ (which is θ itself when γ = 0). With β = p = γ = 0 this is SGD (Zhang
/// and co-authors 2019, Eq. 2); with β, their heavy-ball momentum in the convention v ←
/// βv + g with the step outside (Eq. 5); with p, their preconditioner P = H^p, 0 ≤ p ≤ 1
/// (Eq. 8: "θ(t+1) = θ(t) − α P⁻¹[Hθ + ε]"); with γ, their exponential moving average of
/// the iterates, used for evaluation (Eq. 11). Combinations (momentum with a
/// preconditioner, momentum with averaging) are derived here by composing the source's
/// updates, the preconditioner applied to the momentum (for a fixed P the same as
/// applying it to the gradient); θ̃(0) = θ(0) is decided here (the source does not say).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Rule {
    /// α, the learning rate.
    pub lr: f64,
    /// β, heavy-ball momentum (0: none), 0 ≤ β < 1.
    pub momentum: f64,
    /// p, the power of the curvature that preconditions (0: none), 0 ≤ p ≤ 1.
    pub power: f64,
    /// γ, the coefficient of the average of the iterates (0: none), 0 ≤ γ < 1.
    pub average: f64,
}

impl Rule {
    /// Plain SGD with learning rate `lr`.
    pub fn sgd(lr: f64) -> Rule {
        Rule { lr, momentum: 0.0, power: 0.0, average: 0.0 }
    }

    /// Heavy-ball momentum β = `momentum`.
    pub fn heavy_ball(lr: f64, momentum: f64) -> Rule {
        Rule { lr, momentum, ..Rule::sgd(lr) }
    }

    /// SGD preconditioned by H^`power`.
    pub fn preconditioned(lr: f64, power: f64) -> Rule {
        Rule { lr, power, ..Rule::sgd(lr) }
    }

    /// SGD evaluated at the moving average of its iterates with coefficient `average`.
    pub fn averaged(lr: f64, average: f64) -> Rule {
        Rule { lr, average, ..Rule::sgd(lr) }
    }
}

/// The best rule found for a target by [`Nqm::best_lr`] or [`Tuner`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Tuning {
    /// The learning rate.
    pub lr: f64,
    /// The momentum it was found at.
    pub momentum: f64,
    /// The steps it takes to reach the target ([`Nqm::steps_to`]).
    pub steps: u64,
    /// The same with the last step interpolated (the search's objective): steps − 1
    /// plus the fraction of the last step at which the expected loss, interpolated in
    /// its logarithm, reaches the target.
    pub fractional: f64,
}

type Mat<const N: usize> = [[f64; N]; N];

fn mat_mul<const N: usize>(a: &Mat<N>, b: &Mat<N>) -> Mat<N> {
    let mut c = [[0.0; N]; N];
    for i in 0..N {
        for k in 0..N {
            let x = a[i][k];
            for j in 0..N {
                c[i][j] += x * b[k][j];
            }
        }
    }
    c
}

/// a s aᵀ + w
fn mat_map<const N: usize>(a: &Mat<N>, s: &Mat<N>, w: &Mat<N>) -> Mat<N> {
    let as_ = mat_mul(a, s);
    let mut c = *w;
    for i in 0..N {
        for j in 0..N {
            let mut x = 0.0;
            for k in 0..N {
                x += as_[i][k] * a[j][k];
            }
            c[i][j] += x;
        }
    }
    c
}

/// (a s aᵀ + w)[e][e]
fn mat_map_entry<const N: usize>(a: &Mat<N>, s: &Mat<N>, w: &Mat<N>, e: usize) -> f64 {
    let mut x = w[e][e];
    for i in 0..N {
        let mut y = 0.0;
        for j in 0..N {
            y += s[i][j] * a[e][j];
        }
        x += a[e][i] * y;
    }
    x
}

/// One direction of the model as a linear system: x(t+1) = A x(t) + b ε(t), its
/// second moments S(t) = E[x xᵀ] obeying S(t+1) = A S(t) Aᵀ + W with W = b bᵀ (exact:
/// the noise is independent of the state), the loss weight × S[e][e].
#[derive(Clone, Copy, Debug)]
struct Dir<const N: usize> {
    a: Mat<N>,
    w: Mat<N>,
    s0: Mat<N>,
    weight: f64,
    e: usize,
}

impl<const N: usize> Dir<N> {
    fn loss(&self, s: &Mat<N>) -> f64 {
        self.weight * s[self.e][self.e]
    }
}

/// The exact expected loss after `t` steps: the map applied t times by doubling.
fn nqm_loss_at<const N: usize>(dirs: &[Dir<N>], t: u64) -> f64 {
    dirs.iter()
        .map(|d| {
            let (mut a, mut w, mut s) = (d.a, d.w, d.s0);
            let mut r = t;
            while r > 0 {
                if r & 1 == 1 {
                    s = mat_map(&a, &s, &w);
                }
                r >>= 1;
                if r > 0 {
                    w = mat_map(&a, &w, &w);
                    a = mat_mul(&a, &a);
                }
            }
            d.loss(&s)
        })
        .sum()
}

/// The exact expected loss at steps 0 to `steps`, step by step.
fn nqm_curve<const N: usize>(dirs: &[Dir<N>], steps: usize) -> Vec<f64> {
    let mut out = vec![0.0; steps + 1];
    for d in dirs {
        let mut s = d.s0;
        out[0] += d.loss(&s);
        for o in out.iter_mut().skip(1) {
            s = mat_map(&d.a, &s, &d.w);
            *o += d.loss(&s);
        }
    }
    out
}

/// The limit of the expected loss: W(2^64) by 64 doublings (exact to rounding for any
/// stable system not within about 10⁻¹⁹ of the edge of stability).
fn nqm_limit<const N: usize>(dirs: &[Dir<N>]) -> f64 {
    dirs.iter()
        .map(|d| {
            let (mut a, mut w) = (d.a, d.w);
            for _ in 0..64 {
                w = mat_map(&a, &w, &w);
                a = mat_mul(&a, &a);
            }
            d.loss(&w)
        })
        .sum()
}

/// The fraction of a step at which a loss going from `above` to `below` reaches
/// `target`, interpolated in the logarithm (linearly if a number is not positive).
fn fraction(above: f64, below: f64, target: f64) -> f64 {
    let f =
        if above > 0.0 && below > 0.0 && target > 0.0 { (above.ln() - target.ln()) / (above.ln() - below.ln()) } else { (above - target) / (above - below) };
    if f.is_nan() {
        1.0
    } else {
        f.clamp(0.0, 1.0)
    }
}

/// The first step at which the expected loss is at or below `target` (and the step
/// interpolated), within `max_steps`, by doubling then halving; see [`Nqm::steps_to`].
fn nqm_search<const N: usize>(dirs: &[Dir<N>], target: f64, max_steps: u64) -> Option<(u64, f64)> {
    let total = |f: &dyn Fn(usize) -> f64| -> f64 { (0..dirs.len()).map(f).sum() };
    // (a NaN loss counts as above the target)
    let above = |l: f64| !matches!(l.partial_cmp(&target), Some(std::cmp::Ordering::Less | std::cmp::Ordering::Equal));
    let loss0 = total(&|i| dirs[i].loss(&dirs[i].s0));
    if !above(loss0) {
        return Some((0, 0.0));
    }
    if max_steps == 0 {
        return None;
    }
    // powers[k][i] = (A^(2^k), W(2^k)) for direction i
    let mut powers: Vec<Vec<(Mat<N>, Mat<N>)>> = vec![dirs.iter().map(|d| (d.a, d.w)).collect()];
    let mut k = 0usize;
    let mut last = loss0;
    let found = loop {
        let lvl = &powers[k];
        let l = total(&|i| dirs[i].weight * mat_map_entry(&lvl[i].0, &dirs[i].s0, &lvl[i].1, dirs[i].e));
        if !above(l) {
            break Some((k, l));
        }
        last = l;
        let t = 1u64 << k;
        if k >= 62 || t > max_steps / 2 {
            break None;
        }
        let next = lvl.iter().map(|(a, w)| (mat_mul(a, a), mat_map(a, w, w))).collect();
        powers.push(next);
        k += 1;
    };
    let (lo, hi, lo_loss, hi_loss, mut states): (u64, u64, f64, f64, Vec<Mat<N>>) = match found {
        Some((0, l)) => return Some((1, fraction(loss0, l, target))),
        Some((kk, l)) => {
            let lvl = &powers[kk - 1];
            let states = dirs.iter().zip(lvl).map(|(d, (a, w))| mat_map(a, &d.s0, w)).collect();
            (1u64 << (kk - 1), 1u64 << kk, last, l, states)
        }
        None => {
            // not reached by 2^k (the last level): try max_steps itself
            let lo = 1u64 << k;
            let lvl = &powers[k];
            let states: Vec<Mat<N>> = dirs.iter().zip(lvl).map(|(d, (a, w))| mat_map(a, &d.s0, w)).collect();
            let rest = max_steps - lo;
            let mut end = states.clone();
            for (b, lvl) in powers.iter().enumerate() {
                if (rest >> b) & 1 == 1 {
                    for (s, (a, w)) in end.iter_mut().zip(lvl) {
                        *s = mat_map(a, s, w);
                    }
                }
            }
            let l = total(&|i| dirs[i].loss(&end[i]));
            if above(l) || rest == 0 {
                return None;
            }
            (lo, max_steps, last, l, states)
        }
    };
    // binary lifting in [lo, hi): the loss is above the target at t and at or below it at `upper`
    let (mut t, mut t_loss, mut upper, mut up_loss) = (lo, lo_loss, hi, hi_loss);
    let gap = hi - lo;
    let top = 63 - gap.leading_zeros() as usize;
    for kk in (0..=top.min(powers.len() - 1)).rev() {
        let cand = t + (1u64 << kk);
        if cand >= upper {
            continue;
        }
        let lvl = &powers[kk];
        let l = total(&|i| dirs[i].weight * mat_map_entry(&lvl[i].0, &states[i], &lvl[i].1, dirs[i].e));
        if above(l) {
            for (s, (a, w)) in states.iter_mut().zip(lvl) {
                *s = mat_map(a, s, w);
            }
            t = cand;
            t_loss = l;
        } else {
            upper = cand;
            up_loss = l;
        }
    }
    Some((t + 1, t as f64 + fraction(t_loss, up_loss, target)))
}

/// The noisy quadratic model of Zhang, Li, Nado, Martens, Sachdeva, Dahl, Shallue and
/// Grosse (2019, "Which algorithmic choices matter at which batch sizes?", §3;
/// `noise.md` §4): the loss L(θ) = ½ Σᵢ hᵢ θᵢ² (curvature hᵢ in independent directions,
/// the minimum at 0), a gradient query hᵢ θᵢ + noise of variance cᵢ independent of θ, and
/// a mini-batch of B queries averaging B independent ones, so noise of variance cᵢ / B
/// (the online setting: "observations are drawn i.i.d. in every training iteration").
/// Its standard spectrum and noise ([`Nqm::standard`]): hᵢ = 1/i, d = 10⁴, C = H (cᵢ =
/// hᵢ), θ(0) drawn from N(0, I); target risk 0.01.
///
/// Everything is computed from exact recursions of the second moments of each
/// direction's state (θ; with momentum also m; with averaging also θ̃): the noise is
/// independent of the state, so S(t+1) = A S(t) Aᵀ + W exactly, with A the rule's update
/// matrix and W = b bᵀ its noise; the expected loss is ½ Σᵢ hᵢ E[θ̃ᵢ(t)²]. Only the
/// second moments E[θᵢ(0)²] of the start enter (`start`; 1 for N(0, I)). For plain SGD
/// this is the source's Eq. 3; for each rule the stationary value is the source's
/// steady-state term ([`Nqm::stationary_loss`]). A direction with hᵢ = 0 adds nothing to
/// the loss, whatever its θ does; a negative curvature makes the loss unbounded below
/// (and with a preconditioner its power is NaN).
#[derive(Clone, Debug, PartialEq)]
pub struct Nqm {
    /// hᵢ, the curvature of each direction.
    pub h: Vec<f64>,
    /// cᵢ, the variance of one gradient query's noise in each direction.
    pub c: Vec<f64>,
    /// E[θᵢ(0)²], the start's second moment in each direction.
    pub start: Vec<f64>,
}

impl Nqm {
    /// A model with these curvatures, noise variances and starting second moments (all
    /// three cut to the shortest).
    pub fn new(mut h: Vec<f64>, mut c: Vec<f64>, mut start: Vec<f64>) -> Nqm {
        let d = h.len().min(c.len()).min(start.len());
        h.truncate(d);
        c.truncate(d);
        start.truncate(d);
        Nqm { h, c, start }
    }

    /// The source's standard model with `d` directions ([`NQM_DIM`] in the source): hᵢ
    /// = 1/i for i = 1 to d, cᵢ = hᵢ (C = H), θ(0) from N(0, I) (E θᵢ(0)² = 1).
    pub fn standard(d: usize) -> Nqm {
        let h: Vec<f64> = (1..=d).map(|i| 1.0 / i as f64).collect();
        Nqm { c: h.clone(), start: vec![1.0; d], h }
    }

    /// The number of directions.
    pub fn dim(&self) -> usize {
        self.h.len()
    }

    /// The expected loss at the start, ½ Σ hᵢ E[θᵢ(0)²].
    pub fn initial_loss(&self) -> f64 {
        self.h.iter().zip(&self.start).map(|(h, s)| 0.5 * h * s).sum()
    }

    /// The largest stable learning rate for momentum β and preconditioner power p: every
    /// direction with hᵢ > 0 stays bounded iff 0 < α hᵢ^(1−p) < 2 + 2β (with −1 < β < 1;
    /// `curvature.md` A2, the Jury conditions; for β = 0, 2/h_max). Infinite if no
    /// curvature is positive.
    pub fn max_lr(&self, momentum: f64, power: f64) -> f64 {
        let top = self.h.iter().filter(|&&h| h > 0.0).map(|&h| h.powf(1.0 - power)).fold(0.0f64, f64::max);
        (2.0 + 2.0 * momentum) / top
    }

    /// Does every direction (with hᵢ ≠ 0) stay bounded under `rule`: −1 < β < 1, −1 < γ <
    /// 1 and 0 < α hᵢ^(1−p) < 2 + 2β?
    pub fn is_stable(&self, rule: &Rule) -> bool {
        let (al, be, ga) = (rule.lr, rule.momentum, rule.average);
        be > -1.0
            && be < 1.0
            && ga > -1.0
            && ga < 1.0
            && self.h.iter().filter(|&&h| h != 0.0).all(|&h| {
                let x = al * h.powf(1.0 - rule.power);
                x > 0.0 && x < 2.0 + 2.0 * be
            })
    }

    fn dirs<const N: usize>(&self, rule: &Rule, b: f64, sel: [usize; N], e: usize) -> Vec<Dir<N>> {
        let (al, be, ga) = (rule.lr, rule.momentum, rule.average);
        (0..self.dim())
            .filter(|&i| self.h[i] != 0.0)
            .map(|i| {
                let (h, sigma2, s0) = (self.h[i], self.c[i] / b, self.start[i]);
                let q = h.powf(-rule.power);
                let ht = h * q;
                // the full state (θ, m, θ̃)
                let a3 = [[1.0 - al * ht, -al * q * be, 0.0], [h, be, 0.0], [(1.0 - ga) * (1.0 - al * ht), -(1.0 - ga) * al * q * be, ga]];
                let b3 = [-al * q, 1.0, -(1.0 - ga) * al * q];
                let s3 = [[s0, 0.0, s0], [0.0, 0.0, 0.0], [s0, 0.0, s0]];
                let mut d = Dir { a: [[0.0; N]; N], w: [[0.0; N]; N], s0: [[0.0; N]; N], weight: 0.5 * h, e };
                for (r, &si) in sel.iter().enumerate() {
                    for (c, &sj) in sel.iter().enumerate() {
                        d.a[r][c] = a3[si][sj];
                        d.w[r][c] = sigma2 * b3[si] * b3[sj];
                        d.s0[r][c] = s3[si][sj];
                    }
                }
                d
            })
            .collect()
    }

    /// Run `f` on the smallest system that holds the rule: θ alone; θ and m; θ and θ̃;
    /// or all three.
    fn with_system<R>(&self, rule: &Rule, b: f64, f1: impl FnOnce(&[Dir<1>]) -> R, f2: impl FnOnce(&[Dir<2>]) -> R, f3: impl FnOnce(&[Dir<3>]) -> R) -> R {
        match (rule.momentum != 0.0, rule.average != 0.0) {
            (false, false) => f1(&self.dirs(rule, b, [0], 0)),
            (true, false) => f2(&self.dirs(rule, b, [0, 1], 0)),
            (false, true) => f2(&self.dirs(rule, b, [0, 2], 1)),
            (true, true) => f3(&self.dirs(rule, b, [0, 1, 2], 2)),
        }
    }

    /// The exact expected loss after `t` steps of `rule` at batch size `b` (noise
    /// variance cᵢ / b). Cost: about 2 log₂ t small matrix products per direction (1×1 to
    /// 3×3).
    pub fn loss_at(&self, rule: &Rule, b: f64, t: u64) -> f64 {
        self.with_system(rule, b, |d| nqm_loss_at(d, t), |d| nqm_loss_at(d, t), |d| nqm_loss_at(d, t))
    }

    /// The exact expected loss at every step from 0 to `steps` (`steps` + 1 numbers),
    /// step by step: about `steps` small products per direction.
    pub fn losses(&self, rule: &Rule, b: f64, steps: usize) -> Vec<f64> {
        self.with_system(rule, b, |d| nqm_curve(d, steps), |d| nqm_curve(d, steps), |d| nqm_curve(d, steps))
    }

    /// The stationary expected loss of `rule` at batch size `b`, the limit of the
    /// expected loss as the steps go on; `None` if some direction is unstable
    /// ([`Nqm::is_stable`]). In closed form where there is one, per direction (sums over
    /// i, h̃ = hᵢ^(1−p)):
    ///
    /// - SGD: α cᵢ / (2B (2 − α hᵢ)) (Zhang and co-authors' steady-state risk, Eq. 4;
    ///   derived exactly in `noise.md` D2);
    /// - heavy ball: (1 + β) α cᵢ / (2B (2β + 2 − α hᵢ)(1 − β)) (the steady-state term of
    ///   their Theorem 1, which is stated as a bound; the dossier checked that it is the
    ///   exact stationary value, D2 and §4.2);
    /// - preconditioned: α cᵢ hᵢ^(−p) / (2B (2 − α h̃)) (the steady-state term of Eq. 9);
    /// - with both momentum and a preconditioner: (1 + β) α cᵢ hᵢ^(−p) / (2B (2β + 2 − α
    ///   h̃)(1 − β)) (derived here: the substitution that turns preconditioned SGD into
    ///   SGD on curvature h̃ and noise cᵢ hᵢ^(−2p), with the loss still weighted by hᵢ);
    /// - averaging the iterates (no momentum): the SGD value (with h̃ for hᵢ) times (1 −
    ///   γ)(1 + (1 − α h̃) γ) / ((1 + γ)(1 − (1 − α h̃) γ)) (Theorem 2; checked exact in
    ///   the dossier, §4.2);
    /// - momentum with averaging (no closed form in the source): the exact limit of the
    ///   recursion, computed by doubling (derived here).
    ///
    /// Closed forms are used for 0 ≤ β < 1 and 0 ≤ γ < 1, the source's ranges; outside
    /// them, the limit by doubling. The test `the_stationary_closed_forms_are_the_limits`
    /// checks each closed form against the limit.
    pub fn stationary_loss(&self, rule: &Rule, b: f64) -> Option<f64> {
        if !self.is_stable(rule) {
            return None;
        }
        let (al, be, ga, p) = (rule.lr, rule.momentum, rule.average, rule.power);
        let source_ranges = (0.0..1.0).contains(&be) && (0.0..1.0).contains(&ga);
        if !source_ranges || (be != 0.0 && ga != 0.0) {
            return Some(self.with_system(rule, b, nqm_limit, nqm_limit, nqm_limit));
        }
        let mut total = 0.0;
        for i in 0..self.dim() {
            let (h, c) = (self.h[i], self.c[i]);
            if h == 0.0 {
                continue;
            }
            let q = h.powf(-p);
            let ht = h * q;
            let base = (1.0 + be) * al * c * q / (2.0 * b * (2.0 * be + 2.0 - al * ht) * (1.0 - be));
            let factor = if ga == 0.0 {
                1.0
            } else {
                let r = 1.0 - al * ht;
                (1.0 - ga) * (1.0 + r * ga) / ((1.0 + ga) * (1.0 - r * ga))
            };
            total += base * factor;
        }
        Some(total)
    }

    /// The number of steps of `rule` at batch size `b` until the expected loss is first
    /// at or below `target`: the smallest t ≤ `max_steps` with E L(t) ≤ target (0 if
    /// the start already is), or `None` if it is not reached by `max_steps`.
    ///
    /// Found by doubling (the exact loss at t = 1, 2, 4, ... until one is at or below
    /// the target) and then halving within the last doubling, which finds a step t with
    /// the loss above the target at t − 1 and not above it at t. When the expected loss
    /// falls monotonically (plain SGD and preconditioned SGD from a start above their
    /// floors: each direction's second moment then moves monotonically to its floor) that
    /// is exactly the first such step. With momentum or averaging the expected loss can
    /// rise and fall, and a brief earlier dip below the target that lies between two of
    /// the trial points could be missed; the tests compare it with stepping one step at a
    /// time. Cost: about 4 log₂ T small matrix products per direction for an answer T,
    /// and memory for log₂ T of them per direction (2 numbers each for SGD, 8 with
    /// momentum or averaging, 18 with both: for 10⁴ directions and T near 6·10⁴, 3 MB
    /// and 24 MB). Measured on the standard model (10⁴ directions, B = 100, target
    /// 0.01, about 6·10⁴ steps): 3 ms for SGD, 12 ms with momentum, 28 ms with momentum
    /// and averaging.
    pub fn steps_to(&self, target: f64, rule: &Rule, b: f64, max_steps: u64) -> Option<u64> {
        self.search(target, rule, b, max_steps).map(|(t, _)| t)
    }

    fn search(&self, target: f64, rule: &Rule, b: f64, max_steps: u64) -> Option<(u64, f64)> {
        self.with_system(rule, b, |d| nqm_search(d, target, max_steps), |d| nqm_search(d, target, max_steps), |d| nqm_search(d, target, max_steps))
    }

    /// The best learning rate for reaching `target` at batch size `b` with the given
    /// momentum, preconditioner power and averaging: the one with the fewest
    /// [`Nqm::steps_to`] (within `max_steps`), with those steps; `None` if no rate
    /// reaches it. A [`Tuner`] with one momentum, run to the end; see there for the
    /// search and its accuracy. Cost: about 40 searches of [`Nqm::steps_to`].
    pub fn best_lr(&self, target: f64, b: f64, momentum: f64, power: f64, average: f64, max_steps: u64) -> Option<Tuning> {
        let mut t = Tuner::new(target, b, &[momentum], power, average, max_steps);
        while !t.advance(self, usize::MAX) {}
        t.best()
    }

    /// The best learning rate and momentum (from `momenta`, a grid the caller chooses;
    /// [`MOMENTA`] is one) for `target` at batch size `b`. About 40 searches per
    /// momentum.
    pub fn best_lr_momentum(&self, target: f64, b: f64, momenta: &[f64], power: f64, average: f64, max_steps: u64) -> Option<Tuning> {
        let mut t = Tuner::new(target, b, momenta, power, average, max_steps);
        while !t.advance(self, usize::MAX) {}
        t.best()
    }

    /// One run of `rule` at batch size `b` simulated with noise: θᵢ(0) drawn from a
    /// normal distribution with second moment `start`ᵢ, ε(t) standard normal from
    /// `rng`, and the loss ½ Σ hᵢ θ̃ᵢ² of that run at every step from 0 to `steps`. The
    /// mean over many runs approaches [`Nqm::losses`]. Cost: d normal draws per step.
    pub fn simulate(&self, rule: &Rule, b: f64, steps: usize, rng: &mut Rng) -> Vec<f64> {
        let (al, be, ga, p) = (rule.lr, rule.momentum, rule.average, rule.power);
        let live: Vec<usize> = (0..self.dim()).filter(|&i| self.h[i] != 0.0).collect();
        let q: Vec<f64> = live.iter().map(|&i| self.h[i].powf(-p)).collect();
        let sigma: Vec<f64> = live.iter().map(|&i| (self.c[i] / b).sqrt()).collect();
        let mut theta: Vec<f64> = live.iter().map(|&i| self.start[i].sqrt() * rng.normal() as f64).collect();
        let mut m = vec![0.0; live.len()];
        let mut avg = theta.clone();
        let loss = |avg: &[f64]| live.iter().zip(avg).map(|(&i, x)| 0.5 * self.h[i] * x * x).sum::<f64>();
        let mut out = Vec::with_capacity(steps + 1);
        out.push(loss(&avg));
        for _ in 0..steps {
            for (k, &i) in live.iter().enumerate() {
                let eps = rng.normal() as f64;
                m[k] = be * m[k] + self.h[i] * theta[k] + sigma[k] * eps;
                theta[k] -= al * q[k] * m[k];
                avg[k] = ga * avg[k] + (1.0 - ga) * theta[k];
            }
            out.push(loss(&avg));
        }
        out
    }
}

/// A grid of momenta for [`Nqm::best_lr_momentum`] (chosen here; the source tuned β by a
/// grid search it does not list): 0, 0.5, 0.7, 0.8, 0.9, 0.95, 0.98, 0.99, 0.995, 0.999.
pub const MOMENTA: [f64; 10] = [0.0, 0.5, 0.7, 0.8, 0.9, 0.95, 0.98, 0.99, 0.995, 0.999];

/// Points of the grid over the learning rate, and the decades it spans below the
/// largest stable rate.
const GRID: usize = 25;
const DECADES: f64 = 6.0;
/// The width (in the logarithm of the rate) at which the golden-section search stops.
const LR_TOL: f64 = 1e-3;

#[derive(Clone, Copy, Debug)]
enum Phase {
    Grid { i: usize, best: (f64, f64, usize) },
    Golden { a: f64, b: f64, x1: f64, x2: f64, f1: Option<f64>, f2: Option<f64> },
    Next,
    Done,
}

/// The search for the best learning rate (and momentum, from a grid) for a target at a
/// batch size in the noisy quadratic model, advanced one search of [`Nqm::steps_to`] at
/// a time (each is a few milliseconds for 10⁴ directions with SGD, some tens with
/// momentum: see [`Nqm::steps_to`]; [`Nqm::best_lr`] runs one to the end, 66 ms for
/// SGD on the standard model at B = 100).
///
/// For each momentum β (in the order given): a grid of 25 learning rates evenly spaced
/// in the logarithm over 6 decades below the largest stable rate ([`Nqm::max_lr`], times
/// 0.999999), then a golden-section search on the logarithm of the rate between the
/// grid's neighbours of its best point, until that bracket is narrower than 10⁻³ (the
/// rate known to 0.1%). The objective is the fractional number of steps
/// ([`Tuning::fractional`]), which unlike the whole number is continuous in the rate,
/// so the golden-section search is not stalled by ties. Accuracy: the rate is located
/// to 0.1% of the minimiser of that objective in the bracket, and the best of all the
/// rates tried is returned; if the objective has a second, better basin outside the
/// bracket (it did not in the tests), it is missed. The test
/// `the_best_lr_against_a_fine_grid` compares the steps with those of the best of 2000
/// rates.
#[derive(Clone, Debug)]
pub struct Tuner {
    target: f64,
    b: f64,
    power: f64,
    average: f64,
    max_steps: u64,
    momenta: Vec<f64>,
    at: usize,
    phase: Phase,
    best: Option<Tuning>,
    searches: usize,
}

impl Tuner {
    /// To find the best rate for `target` at batch `b`, over the momenta `momenta` (an
    /// empty list means momentum 0), with preconditioner power `power` and averaging
    /// `average`, within `max_steps`.
    pub fn new(target: f64, b: f64, momenta: &[f64], power: f64, average: f64, max_steps: u64) -> Tuner {
        let momenta = if momenta.is_empty() { vec![0.0] } else { momenta.to_vec() };
        Tuner { target, b, power, average, max_steps, momenta, at: 0, phase: Phase::Grid { i: 0, best: (f64::NAN, f64::INFINITY, 0) }, best: None, searches: 0 }
    }

    fn grid_x(&self, nqm: &Nqm, i: usize) -> f64 {
        let top = (nqm.max_lr(self.momenta[self.at], self.power) * 0.999999).ln();
        let span = DECADES * std::f64::consts::LN_10;
        top - span + span * i as f64 / (GRID - 1) as f64
    }

    fn evaluate(&mut self, nqm: &Nqm, x: f64) -> f64 {
        self.searches += 1;
        let beta = self.momenta[self.at];
        let rule = Rule { lr: x.exp(), momentum: beta, power: self.power, average: self.average };
        match nqm.search(self.target, &rule, self.b, self.max_steps) {
            Some((steps, fractional)) => {
                if self.best.is_none_or(|b| fractional < b.fractional) {
                    self.best = Some(Tuning { lr: rule.lr, momentum: beta, steps, fractional });
                }
                fractional
            }
            None => f64::INFINITY,
        }
    }

    /// Up to `searches` more searches of [`Nqm::steps_to`]. Returns whether it is done.
    pub fn advance(&mut self, nqm: &Nqm, searches: usize) -> bool {
        const PHI: f64 = 0.618_033_988_749_895;
        let mut left = searches;
        loop {
            match self.phase {
                Phase::Done => return true,
                Phase::Next => {
                    self.at += 1;
                    self.phase = if self.at < self.momenta.len() { Phase::Grid { i: 0, best: (f64::NAN, f64::INFINITY, 0) } } else { Phase::Done };
                    continue;
                }
                _ => {}
            }
            if left == 0 {
                return false;
            }
            left -= 1;
            match self.phase {
                Phase::Grid { i, best } => {
                    if !nqm.max_lr(self.momenta[self.at], self.power).is_finite() {
                        self.phase = Phase::Next;
                        continue;
                    }
                    let x = self.grid_x(nqm, i);
                    let f = self.evaluate(nqm, x);
                    let best = if f < best.1 { (x, f, i) } else { best };
                    self.phase = if i + 1 < GRID {
                        Phase::Grid { i: i + 1, best }
                    } else if best.1.is_finite() {
                        let a = self.grid_x(nqm, best.2.saturating_sub(1));
                        let b = self.grid_x(nqm, (best.2 + 1).min(GRID - 1));
                        Phase::Golden { a, b, x1: b - PHI * (b - a), x2: a + PHI * (b - a), f1: None, f2: None }
                    } else {
                        Phase::Next
                    };
                }
                Phase::Golden { a, b, x1, x2, f1, f2 } => {
                    self.phase = match (f1, f2) {
                        (None, _) => Phase::Golden { a, b, x1, x2, f1: Some(self.evaluate(nqm, x1)), f2 },
                        (Some(f1), None) => Phase::Golden { a, b, x1, x2, f1: Some(f1), f2: Some(self.evaluate(nqm, x2)) },
                        (Some(f1), Some(f2)) => {
                            left += 1; // (no search was done)
                            let (a, b, x1, x2, f1, f2) = if f1 < f2 {
                                let b = x2;
                                (a, b, b - PHI * (b - a), x1, None, Some(f1))
                            } else {
                                let a = x1;
                                (a, b, x2, a + PHI * (b - a), Some(f2), None)
                            };
                            if b - a < LR_TOL {
                                Phase::Next
                            } else {
                                Phase::Golden { a, b, x1, x2, f1, f2 }
                            }
                        }
                    };
                }
                Phase::Next | Phase::Done => {}
            }
        }
    }

    /// Searches until the budget is spent (at least one) or it is done.
    pub fn work(&mut self, nqm: &Nqm, budget: &Budget) -> bool {
        loop {
            let done = self.advance(nqm, 1);
            if done || budget.spent() {
                return done;
            }
        }
    }

    /// The best rule found so far (`None` if no rate tried reached the target).
    pub fn best(&self) -> Option<Tuning> {
        self.best
    }

    /// Searches of [`Nqm::steps_to`] done so far.
    pub fn searches(&self) -> usize {
        self.searches
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::math::arch::{Init, MlpSpec, Norm, Place, Stage, TransformerSpec};
    use crate::math::autograd::Act;

    fn mlp(inputs: usize, width: usize, depth: usize, outputs: usize, act: Act, seed: u64) -> Model {
        let spec =
            MlpSpec { inputs, width, depth, outputs, act, norm: Norm::None, place: Place::Pre, residual: false, init: Init::Xavier, ..MlpSpec::default() };
        Model::mlp(&spec, &mut Rng::new(seed))
    }

    /// The spiral problem and a tanh MLP of 123 parameters (2 → 8 → 8 → 3).
    fn spiral_setup(n: usize, seed: u64) -> (Problem, Model) {
        (Problem::spirals(n, 3, &mut Rng::new(seed)), mlp(2, 8, 2, 3, Act::Tanh, seed + 1))
    }

    fn small_transformer(seed: u64) -> Model {
        let spec = TransformerSpec { vocab: 6, context: 6, d_model: 8, heads: 2, layers: 1, d_mlp: 12, ..TransformerSpec::default() };
        Model::transformer(&spec, &mut Rng::new(seed))
    }

    fn random_vector(p: usize, rng: &mut Rng) -> Vec<f64> {
        (0..p).map(|_| rng.normal() as f64).collect()
    }

    /// A few SGD steps on the whole data.
    fn train(problem: &Problem, model: &mut Model, tape: &mut Tape, steps: usize, lr: f64) {
        for _ in 0..steps {
            problem.full_grad(model, tape);
            model.step_sgd(lr);
        }
    }

    fn close(a: f64, b: f64, rel: f64, abs: f64) -> bool {
        (a - b).abs() <= rel * a.abs().max(b.abs()) + abs
    }

    /// Advance an estimate until it converges, failing (not hanging) after `cap` products.
    fn converge(l: &mut Lanczos, cap: usize, mut step: impl FnMut(&mut Lanczos) -> bool) {
        while !step(l) {
            assert!(l.products() < cap, "no convergence after {} products: residual {:.2e}, values {:?}", l.products(), l.residual(), l.values());
        }
    }

    #[test]
    fn flat_views_round_trip() {
        let mut model = mlp(2, 3, 1, 2, Act::Tanh, 1);
        let p = n_params(&model);
        assert_eq!(p, 2 * 3 + 3 + 3 * 2 + 2);
        let spans = spans(&model);
        assert_eq!(spans.len(), model.groups().len());
        assert_eq!(spans.last().unwrap().end, p);
        assert_eq!(span(&model, "read-out bias"), Some(p - 2..p));
        assert_eq!(span(&model, "no such group"), None);
        let w: Vec<f64> = (0..p).map(|i| i as f64).collect();
        set(&mut model, &w);
        let mut back = Vec::new();
        get(&model, &mut back);
        assert_eq!(back, w);
        assert_eq!(model.group("layer 1 bias").unwrap().value, vec![6.0, 7.0, 8.0]);
        // a short vector sets what it reaches, a long one is cut
        set(&mut model, &[-1.0, -2.0]);
        get(&model, &mut back);
        assert_eq!(&back[..3], &[-1.0, -2.0, 2.0]);
        set(&mut model, &vec![5.0; p + 7]);
        get(&model, &mut back);
        assert!(back.iter().all(|&x| x == 5.0) && back.len() == p);
        model.groups_mut()[1].grad[2] = 3.5;
        grad_of(&model, &mut back);
        assert_eq!(back[spans[1].start + 2], 3.5);
    }

    /// `grad` over all examples (and over a subset) agrees with central differences of
    /// `loss` (and `loss_on`), for an MLP with cross-entropy, one with squared error, and
    /// a transformer; and the loss it returns is the loss.
    #[test]
    fn grad_agrees_with_differences_of_the_loss() {
        let mut rng = Rng::new(3);
        let cases: Vec<(&str, Problem, Model)> = vec![
            ("spirals", Problem::spirals(20, 3, &mut rng), mlp(2, 5, 2, 3, Act::Tanh, 4)),
            ("teacher", Problem::teacher(15, 3, &mut rng), mlp(3, 4, 2, 1, Act::Gelu, 5)),
            ("repeats", Problem::repeats(4, 6, 6, &mut rng), small_transformer(6)),
        ];
        for (name, problem, mut model) in cases {
            let mut tape = Tape::new();
            assert!(problem.fits(&model), "{name}");
            for idx in [problem.all(), vec![1, 3, 3, 0]] {
                let loss = problem.grad(&mut model, &mut tape, &idx);
                assert!(close(loss, problem.loss_on(&model, &mut tape, &idx), 1e-14, 0.0), "{name}: the loss returned");
                let mut g = Vec::new();
                grad_of(&model, &mut g);
                let mut flat = Vec::new();
                problem.gradient(&model, &mut tape, &idx, &mut flat);
                assert_eq!(g, flat, "{name}: gradient and grad");
                let largest = g.iter().fold(0.0f64, |a, b| a.max(b.abs()));
                let mut w = Vec::new();
                get(&model, &mut w);
                for e in [0, g.len() / 3, g.len() / 2, g.len() - 1, rng.below(g.len())] {
                    let h = 1e-5 * (1.0 + w[e].abs());
                    let mut moved = w.clone();
                    moved[e] += h;
                    set(&mut model, &moved);
                    let up = problem.loss_on(&model, &mut tape, &idx);
                    moved[e] -= 2.0 * h;
                    set(&mut model, &moved);
                    let down = problem.loss_on(&model, &mut tape, &idx);
                    set(&mut model, &w);
                    let fd = (up - down) / (2.0 * h);
                    assert!(close(g[e], fd, 1e-6, 1e-9 * largest), "{name} entry {e}: grad {} differences {fd}", g[e]);
                }
            }
            // the whole data through `full_grad` and `loss`
            let loss = problem.full_grad(&mut model, &mut tape);
            assert!(close(loss, problem.loss(&model, &mut tape), 1e-14, 0.0));
        }
    }

    /// The mean of the per-example gradients is the full gradient (for an MLP, whose
    /// full gradient comes from one batch pass, and for a transformer).
    #[test]
    fn the_mean_of_the_per_example_gradients_is_the_full_gradient() {
        let mut rng = Rng::new(9);
        let cases: Vec<(Problem, Model)> = vec![
            (Problem::spirals(30, 3, &mut rng), mlp(2, 6, 2, 3, Act::Relu, 10)),
            (Problem::teacher(12, 2, &mut rng), mlp(2, 4, 1, 1, Act::Tanh, 11)),
            (Problem::repeats(5, 6, 6, &mut rng), small_transformer(12)),
        ];
        for (problem, mut model) in cases {
            let mut tape = Tape::new();
            let mut per = PerExample::new(&problem, &model);
            assert!(!per.advance(&problem, &model, &mut tape, 2));
            assert_eq!(per.len(), 2);
            assert!(per.advance(&problem, &model, &mut tape, usize::MAX));
            let noise = per.noise();
            problem.full_grad(&mut model, &mut tape);
            let mut g = Vec::new();
            grad_of(&model, &mut g);
            let largest = g.iter().fold(0.0f64, |a, b| a.max(b.abs()));
            for (a, b) in noise.mean.iter().zip(&g) {
                assert!(close(*a, *b, 1e-12, 1e-14 * largest), "{a} vs {b}");
            }
            // the per-example losses average to the loss
            let mean_loss = (0..per.len()).map(|k| per.loss(k)).sum::<f64>() / per.len() as f64;
            assert!(close(mean_loss, problem.loss(&model, &mut tape), 1e-12, 0.0));
        }
    }

    /// The central difference with an explicit step length δ = r |v|, to compare radii.
    fn hvp_with_step(problem: &Problem, model: &mut Model, tape: &mut Tape, v: &[f64], delta: f64) -> Vec<f64> {
        let r = delta / norm(v);
        let mut w = Vec::new();
        get(model, &mut w);
        let mut moved = w.clone();
        axpy(r, v, &mut moved);
        set(model, &moved);
        let (mut up, mut down) = (Vec::new(), Vec::new());
        problem.gradient_on(model, tape, None, &mut up);
        moved.copy_from_slice(&w);
        axpy(-r, v, &mut moved);
        set(model, &moved);
        problem.gradient_on(model, tape, None, &mut down);
        set(model, &w);
        up.iter().zip(&down).map(|(u, d)| (u - d) / (2.0 * r)).collect()
    }

    /// The tanh network of one unit on one example, whose Hessian is written out by hand
    /// below: L = (w2 tanh(w1 x + b1) + b2 − y)².
    fn one_unit() -> (Problem, Model, Vec<f64>) {
        let (x, y) = (0.9, 0.4);
        let problem = Problem::points(&[x], &[y], 1, Loss::Mse);
        let mut model = mlp(1, 1, 1, 1, Act::Tanh, 1);
        let (w1, b1, w2, b2) = (0.7, -0.3, 1.2, 0.1);
        set(&mut model, &[w1, b1, w2, b2]);
        let t = (w1 * x + b1).tanh();
        let (t1, f) = (1.0 - t * t, w2 * t + b2 - y);
        let t2 = -2.0 * t * t1;
        let df = [w2 * t1 * x, w2 * t1, t, 1.0];
        let ddf = [[w2 * t2 * x * x, w2 * t2 * x, t1 * x, 0.0], [w2 * t2 * x, w2 * t2, t1, 0.0], [t1 * x, t1, 0.0, 0.0], [0.0; 4]];
        let h = (0..16).map(|k| 2.0 * (df[k / 4] * df[k % 4] + f * ddf[k / 4][k % 4])).collect();
        (problem, model, h)
    }

    /// The accuracy of `hvp` against a Hessian written out by hand (a tanh unit), and
    /// against an exactly quadratic loss (the read-out of an MLP under squared error, whose
    /// Hessian is (2/N) Σ zₙ zₙᵀ with zₙ the hidden activations and a 1).
    #[test]
    fn hvp_accuracy_against_an_exact_hessian() {
        let (problem, mut model, exact) = one_unit();
        let mut tape = Tape::new();
        let dense = Hessian::dense(&problem, &mut model, &mut tape);
        let size = exact.iter().fold(0.0f64, |a, b| a.max(b.abs()));
        let err = dense.matrix().iter().zip(&exact).map(|(a, b)| (a - b).abs()).fold(0.0, f64::max) / size;
        println!("one tanh unit: largest error of the dense Hessian {err:.2e} of its largest entry (asymmetry {:.1e})", dense.asymmetry());
        assert!(err < 1e-8, "{err}");
        // the read-out block of an MLP with squared error is exactly quadratic
        let mut rng = Rng::new(21);
        let n = 12;
        let x: Vec<f64> = (0..n * 3).map(|_| rng.normal() as f64).collect();
        let y: Vec<f64> = (0..n).map(|_| rng.normal() as f64).collect();
        let problem = Problem::points(&x, &y, n, Loss::Mse);
        let mut model = mlp(3, 4, 1, 1, Act::Tanh, 22);
        let trace = model.forward_mlp(&mut tape, &x);
        let hidden = tape.value(trace.get(Stage::Output, 1).unwrap()).to_vec();
        let block = span(&model, "read-out weights").unwrap().start..span(&model, "read-out bias").unwrap().end;
        let k = block.len();
        let mut exact = vec![0.0; k * k];
        for e in 0..n {
            let z: Vec<f64> = hidden[e * 4..e * 4 + 4].iter().copied().chain([1.0]).collect();
            for a in 0..k {
                for b in 0..k {
                    exact[a * k + b] += 2.0 / n as f64 * z[a] * z[b];
                }
            }
        }
        let dense = Hessian::dense(&problem, &mut model, &mut tape);
        let p = dense.size();
        let size = exact.iter().fold(0.0f64, |a, b| a.max(b.abs()));
        let mut err = 0.0f64;
        for a in 0..k {
            for b in 0..k {
                err = err.max((dense.matrix()[(block.start + a) * p + block.start + b] - exact[a * k + b]).abs());
            }
        }
        println!("exactly quadratic read-out: largest error {:.2e} of the largest entry", err / size);
        assert!(err / size < 1e-8, "{}", err / size);
    }

    /// Not a check: how the error of the central difference depends on its step, on a
    /// tanh MLP of 123 parameters against a Richardson-extrapolated reference.
    #[test]
    #[ignore]
    fn hvp_radius_experiment() {
        let (problem, mut model) = spiral_setup(64, 5);
        let mut tape = Tape::new();
        train(&problem, &mut model, &mut tape, 50, 0.5);
        let mut w = Vec::new();
        get(&model, &mut w);
        let rms = (dot(&w, &w) / w.len() as f64).sqrt();
        println!("|w| = {:.3}, rms(w) = {rms:.3}, {} parameters", norm(&w), w.len());
        let mut rng = Rng::new(7);
        for trial in 0..3 {
            let v = random_vector(w.len(), &mut rng);
            let d1 = hvp_with_step(&problem, &mut model, &mut tape, &v, 2e-3);
            let d2 = hvp_with_step(&problem, &mut model, &mut tape, &v, 1e-3);
            let reference: Vec<f64> = d1.iter().zip(&d2).map(|(a, b)| (4.0 * b - a) / 3.0).collect();
            let scale = norm(&reference);
            for delta in [1e-7, 3e-7, 1e-6, 3e-6, 1e-5, 3e-5, 1e-4, 3e-4] {
                let d = hvp_with_step(&problem, &mut model, &mut tape, &v, delta);
                let err = d.iter().zip(&reference).map(|(a, b)| (a - b).powi(2)).sum::<f64>().sqrt() / scale;
                println!("trial {trial}: step {delta:.0e}: relative error {err:.2e}");
            }
            let mut out = Vec::new();
            hvp(&problem, &mut model, &mut tape, &problem.all(), &v, &mut out);
            let err = out.iter().zip(&reference).map(|(a, b)| (a - b).powi(2)).sum::<f64>().sqrt() / scale;
            println!("trial {trial}: hvp's own step {:.1e}: relative error {err:.2e}", f64::EPSILON.cbrt() * (1.0 + rms));
        }
    }

    /// `hvp` agrees with second differences of the loss: vᵀ H v with (L(w + hv) − 2 L(w) +
    /// L(w − hv)) / h², and uᵀ H v with the mixed difference; and the model is left
    /// exactly as it was.
    #[test]
    fn hvp_agrees_with_second_differences_of_the_loss() {
        let mut rng = Rng::new(31);
        let cases: Vec<(Problem, Model)> =
            vec![(Problem::spirals(40, 3, &mut rng), mlp(2, 8, 2, 3, Act::Tanh, 32)), (Problem::repeats(3, 6, 6, &mut rng), small_transformer(33))];
        for (problem, mut model) in cases {
            let mut tape = Tape::new();
            problem.grad(&mut model, &mut tape, &[0]);
            let before = model.clone();
            let p = n_params(&model);
            let (u, v) = (random_vector(p, &mut rng), random_vector(p, &mut rng));
            let mut hv = Vec::new();
            let all = problem.all();
            hvp(&problem, &mut model, &mut tape, &all, &v, &mut hv);
            for (a, b) in model.groups().iter().zip(before.groups()) {
                assert_eq!(a.value, b.value, "values restored");
                assert_eq!(a.grad, b.grad, "gradients untouched");
            }
            let mut w = Vec::new();
            get(&model, &mut w);
            let mut loss_at = |du: f64, dv: f64| {
                let moved: Vec<f64> = w.iter().zip(&u).zip(&v).map(|((w, u), v)| w + du * u + dv * v).collect();
                set(&mut model, &moved);
                let l = problem.loss(&model, &mut tape);
                set(&mut model, &w);
                l
            };
            let h = 1e-4;
            let vhv = (loss_at(0.0, h) - 2.0 * loss_at(0.0, 0.0) + loss_at(0.0, -h)) / (h * h);
            let uhv = (loss_at(h, h) - loss_at(h, -h) - loss_at(-h, h) + loss_at(-h, -h)) / (4.0 * h * h);
            let (vhv_p, uhv_p) = (dot(&v, &hv), dot(&u, &hv));
            println!("vᵀHv {vhv_p:.8} vs second difference {vhv:.8}; uᵀHv {uhv_p:.8} vs {uhv:.8}");
            let scale = vhv.abs().max(uhv.abs()).max(1e-3);
            assert!((vhv_p - vhv).abs() < 1e-5 * scale, "{vhv_p} vs {vhv}");
            assert!((uhv_p - uhv).abs() < 1e-5 * scale, "{uhv_p} vs {uhv}");
            // and `curvature_along` is vᵀHv / |v|²
            let c = curvature_along(&problem, &mut model, &mut tape, &all, &v);
            assert!(close(c, vhv_p / dot(&v, &v), 1e-9, 0.0));
        }
    }

    /// The dense Hessian is symmetric to rounding, agrees with `hvp` along any vector,
    /// and its eigen-decomposition reproduces it.
    #[test]
    fn the_dense_hessian_is_symmetric_and_agrees_with_hvp() {
        let (problem, mut model) = spiral_setup(48, 41);
        let mut tape = Tape::new();
        train(&problem, &mut model, &mut tape, 30, 0.5);
        let mut h = Hessian::new(&model);
        let mut slices = 0;
        while !h.advance(&problem, &mut model, &mut tape, 10) {
            slices += 1;
        }
        assert_eq!(slices, n_params(&model) / 10);
        let p = h.size();
        println!("dense Hessian of {p} parameters: asymmetry {:.2e}", h.asymmetry());
        assert!(h.asymmetry() < 1e-6, "{}", h.asymmetry());
        let mut rng = Rng::new(42);
        let v = random_vector(p, &mut rng);
        let mut hv = Vec::new();
        hvp(&problem, &mut model, &mut tape, &problem.all(), &v, &mut hv);
        let mv: Vec<f64> = (0..p).map(|i| dot(&h.matrix()[i * p..(i + 1) * p], &v)).collect();
        let err = mv.iter().zip(&hv).map(|(a, b)| (a - b).powi(2)).sum::<f64>().sqrt() / norm(&hv);
        assert!(err < 1e-7, "{err}");
        // Σ λ_k (v_kᵀ v)² = vᵀ H v
        let quad: f64 = (0..p).map(|k| h.values()[k] * dot(h.vector(k), &v).powi(2)).sum();
        assert!(close(quad, dot(&v, &mv), 1e-8, 1e-10));
        assert!(h.values().windows(2).all(|w| w[0] >= w[1]));
    }

    /// Lanczos's largest eigenvalues agree with the dense Hessian's on an MLP of 123
    /// parameters, at the start and after some training; after a small move of the
    /// weights a warm start converges in fewer products than a cold one, for the top three
    /// and for the sharpness alone.
    #[test]
    fn lanczos_agrees_with_the_dense_hessian() {
        let (problem, mut model) = spiral_setup(64, 51);
        let mut tape = Tape::new();
        let mut lanczos = Lanczos::new(&model, 3, 7);
        lanczos.tolerance = 1e-7;
        for stage in ["at the start", "after 200 steps"] {
            if stage != "at the start" {
                train(&problem, &mut model, &mut tape, 200, 1.0);
                lanczos.reset();
            }
            let dense = Hessian::dense(&problem, &mut model, &mut tape);
            converge(&mut lanczos, 400, |l| l.advance(&problem, &mut model, &mut tape, 5));
            println!(
                "{stage}: dense top {:?}, bottom {:.4}; Lanczos {:?} in {} products, residual {:.1e}",
                &dense.values()[..3],
                dense.values()[dense.size() - 1],
                lanczos.values(),
                lanczos.products(),
                lanczos.residual()
            );
            for k in 0..3 {
                assert!(close(lanczos.values()[k], dense.values()[k], 1e-6, 1e-9), "{stage}: eigenvalue {k}: {} vs {}", lanczos.values()[k], dense.values()[k]);
            }
            assert_eq!(lanczos.sharpness(), lanczos.values()[0]);
            // the top eigenvector, up to sign
            let c = dot(lanczos.vector(0), dense.vector(0)).abs();
            assert!(c > 1.0 - 1e-6, "{stage}: cosine of the top eigenvectors {c}");
        }
        // warm against cold, after small moves of a network trained at a stable step (its
        // top eigenvalue well apart), and (printed only) after 5 steps at the edge of
        // stability, where the top eigenvalues crowd together and their vectors turn fast
        let (problem, mut stable) = spiral_setup(64, 51);
        train(&problem, &mut stable, &mut tape, 200, 0.3);
        for (start, steps, lr, check) in [(&stable, 1, 0.05, true), (&stable, 1, 0.3, true), (&model, 5, 1.0, false)] {
            for want in [1, 3] {
                let mut m = start.clone();
                let mut l = Lanczos::new(&m, want, 8);
                l.tolerance = 1e-7;
                converge(&mut l, 400, |l| l.advance(&problem, &mut m, &mut tape, 1));
                let old = l.vector(0).to_vec();
                train(&problem, &mut m, &mut tape, steps, lr);
                let mut cold = l.clone();
                l.restart();
                cold.reset();
                converge(&mut l, 400, |l| l.advance(&problem, &mut m, &mut tape, 1));
                converge(&mut cold, 400, |l| l.advance(&problem, &mut m, &mut tape, 1));
                let turn = dot(&old, l.vector(0)).abs();
                println!(
                    "{steps} step(s) of {lr}, top {want} {:?}: old and new top vectors' cosine {turn:.4}; warm {} products, cold {}",
                    l.values(),
                    l.products(),
                    cold.products()
                );
                if check {
                    assert!(l.products() < cold.products());
                }
                assert!(close(l.sharpness(), cold.sharpness(), 1e-6, 0.0));
            }
        }
    }

    /// The small eigen-solver of the Lanczos estimate agrees with `linalg::eig_sym` and
    /// reproduces its matrix, on random symmetric matrices of 1 to 40 rows and on
    /// degenerate ones (zero, the identity, rank one, repeated eigenvalues, diagonal).
    #[test]
    fn the_small_eigen_solver_agrees_with_eig_sym() {
        let mut rng = Rng::new(151);
        let mut cases: Vec<(usize, Vec<f64>)> = Vec::new();
        for n in 1..=40 {
            let x = random_vector(n * n, &mut rng);
            cases.push((n, (0..n * n).map(|k| x[k] + x[(k % n) * n + k / n]).collect()));
        }
        let n = 7;
        cases.push((n, vec![0.0; n * n]));
        cases.push((n, linalg::identity(n)));
        let u = random_vector(n, &mut rng);
        cases.push((n, (0..n * n).map(|k| u[k / n] * u[k % n]).collect()));
        cases.push((n, (0..n * n).map(|k| if k / n == k % n { [3.0, -1.0, 3.0, 0.5, 3.0, -1.0, 2.0][k % n] } else { 0.0 }).collect()));
        // repeated eigenvalues in a random basis: Q diag(2, 2, 2, −1, −1, 5, 0) Qᵀ
        let mut q = random_vector(n * n, &mut rng);
        linalg::orthonormalise(&mut q, n, n);
        let lam = [2.0, 2.0, 2.0, -1.0, -1.0, 5.0, 0.0];
        cases.push((n, (0..n * n).map(|k| (0..n).map(|m| q[(k / n) * n + m] * lam[m] * q[(k % n) * n + m]).sum()).collect()));
        for (n, a) in cases {
            let (values, vectors) = sym_eigen(&a, n);
            let (reference, _) = linalg::eig_sym(&a, n);
            let size = a.iter().fold(1e-300f64, |m, x| m.max(x.abs()));
            for (x, y) in values.iter().zip(&reference) {
                assert!((x - y).abs() < 1e-12 * size + 1e-300, "n = {n}: {values:?} vs {reference:?}");
            }
            for i in 0..n {
                for j in 0..n {
                    let rebuilt: f64 = (0..n).map(|k| values[k] * vectors[k * n + i] * vectors[k * n + j]).sum();
                    assert!((rebuilt - a[i * n + j]).abs() < 1e-12 * size, "n = {n}: entry ({i}, {j})");
                    let gram: f64 = (0..n).map(|k| vectors[i * n + k] * vectors[j * n + k]).sum();
                    assert!((gram - if i == j { 1.0 } else { 0.0 }).abs() < 1e-12, "n = {n}: orthonormal");
                }
            }
        }
        assert!(sym_eigen(&[f64::NAN, 1.0, 1.0, 0.0], 2).0.iter().all(|x| x.is_nan()));
        assert!(sym_eigen(&[], 0).0.is_empty());
    }

    /// The sharpness is the most positive eigenvalue even when the most negative one is
    /// far larger in size; thick restarts (a basis of 22 for a space of 200) keep it
    /// converging; and every eigenvalue negative gives a negative sharpness.
    #[test]
    fn lanczos_takes_the_most_positive_eigenvalue() {
        let p = 200;
        let mut eig: Vec<f64> = (0..p).map(|i| 1.0 + i as f64 / p as f64).collect();
        eig[0] = -50.0;
        eig[1] = 3.0;
        let op = |eig: &Vec<f64>| {
            let e = eig.clone();
            move |v: &[f64], out: &mut Vec<f64>| {
                out.clear();
                out.extend(v.iter().zip(&e).map(|(v, e)| v * e));
            }
        };
        let mut l = Lanczos::of_size(p, 1, 3);
        l.tolerance = 1e-9;
        converge(&mut l, 2000, |l| l.advance_with(10, op(&eig)));
        println!("sharpness {} (most negative −50) in {} products", l.sharpness(), l.products());
        assert!((l.sharpness() - 3.0).abs() < 1e-8);
        assert!((l.vector(0)[1].abs() - 1.0).abs() < 1e-6);
        let neg: Vec<f64> = eig.iter().map(|e| -e.abs() - 1.0).collect();
        let mut l = Lanczos::of_size(p, 2, 4);
        l.tolerance = 1e-9;
        converge(&mut l, 2000, |l| l.advance_with(10, op(&neg)));
        let mut sorted = neg.clone();
        sorted.sort_by(|a, b| b.total_cmp(a));
        assert!((l.values()[0] - sorted[0]).abs() < 1e-7 && (l.values()[1] - sorted[1]).abs() < 1e-7, "{:?} vs {:?}", l.values(), &sorted[..2]);
    }

    /// The preconditioned sharpness agrees with the largest eigenvalue of P^(−1/2) H
    /// P^(−1/2) formed from the dense Hessian.
    #[test]
    fn the_preconditioned_sharpness_agrees_with_the_dense_one() {
        let (problem, mut model) = spiral_setup(40, 61);
        let mut tape = Tape::new();
        train(&problem, &mut model, &mut tape, 100, 1.0);
        let dense = Hessian::dense(&problem, &mut model, &mut tape);
        let n = dense.size();
        let mut rng = Rng::new(62);
        let pre: Vec<f64> = (0..n).map(|_| 0.01 + rng.f32() as f64).collect();
        let mut m = dense.matrix().to_vec();
        for i in 0..n {
            for j in 0..n {
                m[i * n + j] /= (pre[i] * pre[j]).sqrt();
            }
        }
        let (values, _) = linalg::eig_sym(&m, n);
        let mut l = Lanczos::new(&model, 2, 5).preconditioned(&pre);
        l.tolerance = 1e-8;
        converge(&mut l, 1000, |l| l.advance(&problem, &mut model, &mut tape, 10));
        println!("preconditioned: dense {:?}, Lanczos {:?} in {} products", &values[..2], l.values(), l.products());
        assert!(close(l.values()[0], values[0], 1e-6, 0.0) && close(l.values()[1], values[1], 1e-6, 0.0));
        // a new preconditioner restarts it, warm, and it converges again to the new value
        let pre2: Vec<f64> = pre.iter().map(|p| p * 1.1).collect();
        l.set_preconditioner(&pre2);
        converge(&mut l, 1000, |l| l.advance(&problem, &mut model, &mut tape, 10));
        assert!(close(l.values()[0], values[0] / 1.1, 1e-6, 0.0), "{} vs {}", l.values()[0], values[0] / 1.1);
    }

    /// Batch sharpness: each batch's value is its own gradient's curvature under its own
    /// Hessian (replayed with the same seed), and with the whole data set as the batch it
    /// is the curvature of the full loss along the full gradient.
    #[test]
    fn batch_sharpness_is_its_definition() {
        let (problem, mut model) = spiral_setup(40, 71);
        let mut tape = Tape::new();
        train(&problem, &mut model, &mut tape, 50, 1.0);
        let mut bs = BatchSharpness::new(Sampler::new(40, 8, Draw::WithoutReplacement, Rng::new(72)));
        bs.advance(&problem, &mut model, &mut tape, 12);
        let mut replay = Sampler::new(40, 8, Draw::WithoutReplacement, Rng::new(72));
        let mut sum = 0.0;
        for _ in 0..12 {
            let idx = replay.deal().to_vec();
            let mut g = Vec::new();
            problem.gradient(&model, &mut tape, &idx, &mut g);
            sum += curvature_along(&problem, &mut model, &mut tape, &idx, &g);
        }
        assert_eq!(bs.count(), 12);
        assert!(close(bs.mean(), sum / 12.0, 1e-12, 0.0));
        assert!(bs.standard_error() > 0.0);
        let mut full = BatchSharpness::new(Sampler::new(40, 40, Draw::WithoutReplacement, Rng::new(73)));
        full.advance(&problem, &mut model, &mut tape, 3);
        let mut g = Vec::new();
        problem.gradient(&model, &mut tape, &problem.all(), &mut g);
        let along = curvature_along(&problem, &mut model, &mut tape, &problem.all(), &g);
        println!("batch sharpness at B = 8: {:.4} ± {:.4}; at B = N: {:.4} (full: {along:.4})", bs.mean(), bs.standard_error(), full.mean());
        assert!(close(full.mean(), along, 1e-9, 0.0));
    }

    /// Without replacement an epoch deals disjoint batches of B distinct examples; with
    /// replacement any; sizes as documented.
    #[test]
    fn the_sampler_deals_as_documented() {
        let mut s = Sampler::new(10, 3, Draw::WithoutReplacement, Rng::new(1));
        assert_eq!(s.size(), 3);
        let mut seen = Vec::new();
        for _ in 0..3 {
            seen.extend_from_slice(s.deal());
        }
        assert_eq!(s.epoch(), 0);
        let mut sorted = seen.clone();
        sorted.sort();
        sorted.dedup();
        assert_eq!(sorted.len(), 9, "one epoch: 9 distinct examples");
        s.deal();
        assert_eq!(s.epoch(), 1, "the tenth example sits out; a new epoch");
        assert_eq!(Sampler::new(5, 9, Draw::WithoutReplacement, Rng::new(2)).size(), 5);
        assert_eq!(Sampler::new(5, 0, Draw::WithReplacement, Rng::new(2)).size(), 1);
        let mut r = Sampler::new(4, 50, Draw::WithReplacement, Rng::new(3));
        assert_eq!(r.deal().len(), 50);
        assert!(r.deal().iter().all(|&k| k < 4));
        let mut empty = Sampler::new(0, 5, Draw::WithoutReplacement, Rng::new(4));
        assert!(empty.deal().is_empty());
        let mut empty = Sampler::new(0, 5, Draw::WithReplacement, Rng::new(4));
        assert!(empty.deal().is_empty());
    }

    /// The per-example gradients of a tanh MLP of 643 parameters (2 → 16 → 16 → 16 → 3) on
    /// 64 spiral points, after `steps` steps of gradient descent of 0.5.
    fn noise_setup(seed: u64, steps: usize) -> (Problem, Model, PerExample) {
        let problem = Problem::spirals(64, 3, &mut Rng::new(seed));
        let mut model = mlp(2, 16, 3, 3, Act::Tanh, seed + 1);
        let mut tape = Tape::new();
        train(&problem, &mut model, &mut tape, steps, 0.5);
        let mut per = PerExample::new(&problem, &model);
        per.advance(&problem, &model, &mut tape, usize::MAX);
        (problem, model, per)
    }

    /// The sampled mean of |g_B|² over many mini-batches agrees with the formula, with
    /// and without replacement, over all parameters and over one group. Tolerance: 4
    /// standard errors of the sampled mean (for without replacement, the batches of one
    /// epoch are disjoint and so negatively correlated, which makes the true error of
    /// their mean smaller than this estimate).
    #[test]
    fn the_sampled_squared_length_agrees_with_the_formula() {
        let (_, model, per) = noise_setup(81, 100);
        let n = per.len();
        let layer = span(&model, "layer 2 weights").unwrap();
        for range in [0..per.width(), layer] {
            let noise = per.noise_in(range.clone());
            for draw in [Draw::WithReplacement, Draw::WithoutReplacement] {
                for b in [1, 4, 16, 48, 64] {
                    let s = per.sample_in(range.clone(), b, draw, 20_000, Rng::new(b as u64 * 7 + 1));
                    let formula = noise.expected_sq_norm(b, draw);
                    let z = (s.sq_norm - formula) / s.sq_norm_error;
                    println!("{draw:?} B = {b}: sampled {:.6} ± {:.6}, formula {formula:.6} ({z:+.2} s.e.)", s.sq_norm, s.sq_norm_error);
                    if draw == Draw::WithoutReplacement && b == n {
                        assert!(close(s.sq_norm, noise.g2, 1e-9, 0.0), "the whole data set: no noise");
                    } else {
                        assert!(z.abs() < 4.0, "{draw:?} B = {b}: {z} standard errors");
                    }
                }
            }
        }
    }

    /// McCandlish's two-batch estimates are unbiased: their means over many pairs of
    /// batches (with replacement) are |G|² and tr Σ, within 4 standard errors.
    #[test]
    fn the_two_batch_estimator_is_unbiased() {
        let (_, _, per) = noise_setup(91, 100);
        let noise = per.noise();
        let mut rng = Rng::new(92);
        for (small, big) in [(1, 8), (4, 32), (2, 64)] {
            let mut gs = Running::default();
            let mut ss = Running::default();
            let mut sampler_small = Sampler::new(per.len(), small, Draw::WithReplacement, Rng::new(rng.next_u64()));
            let mut sampler_big = Sampler::new(per.len(), big, Draw::WithReplacement, Rng::new(rng.next_u64()));
            let mut g = Vec::new();
            for _ in 0..40_000 {
                per.batch_mean(sampler_small.deal(), 0..per.width(), &mut g);
                let sq_small = dot(&g, &g);
                per.batch_mean(sampler_big.deal(), 0..per.width(), &mut g);
                let sq_big = dot(&g, &g);
                let (g2, s) = two_batch_estimates(small as f64, sq_small, big as f64, sq_big);
                gs.push(g2);
                ss.push(s);
            }
            let (zg, zs) = ((gs.mean() - noise.g2) / gs.error(), (ss.mean() - noise.trace) / ss.error());
            println!(
                "B {small} and {big}: |G|² {:.5} ± {:.5} (true {:.5}), tr Σ {:.5} ± {:.5} (true {:.5})",
                gs.mean(),
                gs.error(),
                noise.g2,
                ss.mean(),
                ss.error(),
                noise.trace
            );
            assert!(zg.abs() < 4.0 && zs.abs() < 4.0, "{zg} {zs}");
        }
    }

    /// The approximation of the expected cosine against sampling, over all parameters and
    /// one group, with and without replacement: at the start of training (where the
    /// noise scale is some tens and the cosines spread over a wide range) held to 0.08
    /// (measured: at most 0.04 over all parameters, 0.07 for the layer), and after 100
    /// steps (noise scale in the tens of thousands, every cosine below 0.07) to 0.03; the
    /// differences are printed.
    #[test]
    fn the_cosine_approximation_against_sampling() {
        for (steps, tolerance) in [(0, 0.08), (100, 0.03)] {
            let (_, model, per) = noise_setup(101, steps);
            let layer = span(&model, "layer 2 weights").unwrap();
            let mut worst = 0.0f64;
            for (name, range) in [("all", 0..per.width()), ("layer 2 weights", layer)] {
                let noise = per.noise_in(range.clone());
                println!("after {steps} steps, {name}: B_simple {:.1}", noise.simple_noise_scale());
                for draw in [Draw::WithReplacement, Draw::WithoutReplacement] {
                    for b in [1, 2, 4, 8, 16, 32, 48] {
                        let s = per.sample_in(range.clone(), b, draw, 20_000, Rng::new(b as u64 + 5));
                        let approx = noise.expected_cosine(b, draw);
                        println!("  {draw:?} B = {b}: sampled {:.4} ± {:.4}, approximation {approx:.4}", s.cosine, s.cosine_error);
                        worst = worst.max((s.cosine - approx).abs());
                    }
                }
            }
            println!("after {steps} steps: largest difference {worst:.4}");
            assert!(worst < tolerance, "{worst}");
        }
    }

    /// The full noise scale's parts agree with the dense Hessian, Gᵀ H G and tr(H Σ) =
    /// (1/N) Σ dₙᵀ H dₙ, at the start and after 40 steps (when Gᵀ H G had turned slightly
    /// negative); and the one-step model's identities hold.
    #[test]
    fn the_full_noise_scale_agrees_with_the_dense_hessian() {
        let (problem, mut model) = spiral_setup(24, 111);
        let mut tape = Tape::new();
        for trained in [0, 40] {
            train(&problem, &mut model, &mut tape, trained, 0.5);
            let mut full = FullNoise::new(&problem, &model);
            let mut slices = 0;
            while !full.advance(&problem, &mut model, &mut tape, 7) {
                slices += 1;
            }
            assert_eq!(slices, (24 + 25usize).div_ceil(7) - 1, "24 rows and 25 products, 7 a slice");
            let dense = Hessian::dense(&problem, &mut model, &mut tape);
            let p = dense.size();
            let per = full.per_example();
            let noise = per.noise();
            let hx = |x: &[f64]| -> Vec<f64> { (0..p).map(|i| dot(&dense.matrix()[i * p..(i + 1) * p], x)).collect() };
            let ghg = dot(&noise.mean, &hx(&noise.mean));
            let trhs: f64 = (0..per.len())
                .map(|k| {
                    let d: Vec<f64> = per.row(k).iter().zip(&noise.mean).map(|(g, m)| g - m).collect();
                    dot(&d, &hx(&d))
                })
                .sum::<f64>()
                / per.len() as f64;
            println!(
                "after {trained} steps: Gᵀ H G {:.6e} (dense {ghg:.6e}), tr(H Σ) {:.6e} (dense {trhs:.6e}), B_noise {:.2}, B_simple {:.2}",
                full.ghg(),
                full.trace_hs(),
                full.noise_scale(),
                noise.simple_noise_scale()
            );
            assert!(close(full.ghg(), ghg, 1e-6, 1e-9 * noise.g2) && close(full.trace_hs(), trhs, 1e-6, 0.0));
            let one = full.one_step();
            assert!(close(one.g2, noise.g2, 1e-15, 0.0) && one.ghg == full.ghg() && one.trhs == full.trace_hs());
        }
        // the one-step model's identities (Eqs. 2.5 to 2.7, D4), on numbers with Gᵀ H G > 0
        let one = OneStep { g2: 2.0, ghg: 3.0, trhs: 50.0 };
        assert!(close(one.best_step(8.0), one.max_step() / (1.0 + one.noise_scale() / 8.0), 1e-12, 0.0));
        assert!(close(one.best_gain(8.0), one.max_gain() / (1.0 + one.noise_scale() / 8.0), 1e-12, 0.0));
        assert!(close(one.change(one.best_step(8.0), 8.0), -one.best_gain(8.0), 1e-12, 0.0));
        assert!(one.change(2.0 * one.best_step(8.0), 8.0).abs() < 1e-12, "twice the best step: no change");
        let b = one.noise_scale();
        assert!(close(one.best_gain(b), 0.5 * one.max_gain(), 1e-12, 0.0), "half the largest gain at B = B_noise");
        assert!(close(one.steps_ratio(b), 2.0, 1e-12, 0.0) && close(one.examples_ratio(b), 2.0, 1e-12, 0.0));
        assert!(close((steps_over_min(3.0, 7.0) - 1.0) * (examples_over_min(3.0, 7.0) - 1.0), 1.0, 1e-15, 0.0));
    }

    /// On an exactly quadratic problem (the read-out of an MLP under squared error, the
    /// hidden layer held fixed: linear regression on fixed features through
    /// `Problem::points`), the one-step model predicts the measured mean change of the
    /// loss after one SGD step of the read-out, for several steps and batch sizes, with
    /// and without replacement (B replaced by the effective batch). Tolerance: 4 standard
    /// errors of the measured mean.
    #[test]
    fn the_one_step_model_predicts_an_exactly_quadratic_problem() {
        let mut rng = Rng::new(121);
        let n = 32;
        let x: Vec<f64> = (0..n * 3).map(|_| rng.normal() as f64).collect();
        let y: Vec<f64> = x.chunks(3).map(|x| x[0] - 0.5 * x[1] + 0.3 * x[2] + 0.4 * rng.normal() as f64).collect();
        let problem = Problem::points(&x, &y, n, Loss::Mse);
        let mut model = mlp(3, 4, 1, 1, Act::Tanh, 122);
        let mut tape = Tape::new();
        let block = span(&model, "read-out weights").unwrap().start..span(&model, "read-out bias").unwrap().end;
        let p = n_params(&model);
        let mut per = PerExample::new(&problem, &model);
        per.advance(&problem, &model, &mut tape, usize::MAX);
        let noise = per.noise_in(block.clone());
        let pad = |x: &[f64]| -> Vec<f64> {
            let mut v = vec![0.0; p];
            v[block.clone()].copy_from_slice(x);
            v
        };
        let all = problem.all();
        let mut hv = Vec::new();
        hvp(&problem, &mut model, &mut tape, &all, &pad(&noise.mean), &mut hv);
        let ghg = dot(&noise.mean, &hv[block.clone()]);
        let mut trhs = 0.0;
        for k in 0..n {
            let d: Vec<f64> = per.row(k)[block.clone()].iter().zip(&noise.mean).map(|(g, m)| g - m).collect();
            hvp(&problem, &mut model, &mut tape, &all, &pad(&d), &mut hv);
            trhs += dot(&d, &hv[block.clone()]) / n as f64;
        }
        let one = OneStep { g2: noise.g2, ghg, trhs };
        println!("B_noise {:.2}, best step at B = 1 {:.4}, εmax {:.4}", one.noise_scale(), one.best_step(1.0), one.max_step());
        let loss0 = problem.loss(&model, &mut tape);
        let mut w = Vec::new();
        get(&model, &mut w);
        let mut g = Vec::new();
        for draw in [Draw::WithReplacement, Draw::WithoutReplacement] {
            for b in [1, 4, 16] {
                for lr in [0.5 * one.best_step(b as f64), one.best_step(b as f64), 1.8 * one.best_step(b as f64)] {
                    let mut sampler = Sampler::new(n, b, draw, Rng::new(b as u64 * 31 + 3));
                    let mut change = Running::default();
                    let mut moved = w.clone();
                    for _ in 0..4000 {
                        per.batch_mean(sampler.deal(), block.clone(), &mut g);
                        moved[block.clone()].copy_from_slice(&w[block.clone()]);
                        axpy(-lr, &g, &mut moved[block.clone()]);
                        set(&mut model, &moved);
                        change.push(problem.loss(&model, &mut tape) - loss0);
                    }
                    set(&mut model, &w);
                    let predicted = one.change(lr, effective_batch(b, n, draw));
                    let z = (change.mean() - predicted) / change.error();
                    println!("{draw:?} B = {b} η = {lr:.4}: measured {:.6} ± {:.6}, predicted {predicted:.6} ({z:+.2})", change.mean(), change.error());
                    assert!(z.abs() < 4.0, "{z}");
                }
            }
        }
    }

    /// The exact recursions against plain simulation with noise: the mean loss of 4000
    /// runs (independent seeds within one stream), at several steps, for every kind of
    /// rule, within 4.5 standard errors of the mean of those runs (estimated from them).
    /// The mean of 4000 independent runs is within 4.5 standard errors of its expectation
    /// with probability 1 − 7·10⁻⁶ (normal approximation), so 30 such checks fail by
    /// chance with probability about 2·10⁻⁴; the seeds are fixed in any case.
    #[test]
    fn the_noisy_quadratic_model_agrees_with_simulation() {
        let h = vec![2.0, 1.0, 0.5, 0.25, 0.1, 0.05];
        let nqm = Nqm::new(h.clone(), h.iter().map(|h| 0.7 * h).collect(), vec![1.0, 2.0, 1.0, 0.5, 1.0, 1.0]);
        let rules = [
            Rule::sgd(0.6),
            Rule::heavy_ball(0.2, 0.8),
            Rule::preconditioned(0.3, 0.5),
            Rule::averaged(0.6, 0.9),
            Rule { lr: 0.1, momentum: 0.7, power: 0.0, average: 0.8 },
            Rule { lr: 0.1, momentum: 0.5, power: 0.3, average: 0.5 },
        ];
        let mut rng = Rng::new(131);
        for rule in rules {
            let b = 4.0;
            let steps = 60;
            let exact = nqm.losses(&rule, b, steps);
            let mut at = [Running::default(); 5];
            let checks = [0, 1, 5, 20, 60];
            for _ in 0..4000 {
                let run = nqm.simulate(&rule, b, steps, &mut rng);
                for (r, &t) in at.iter_mut().zip(&checks) {
                    r.push(run[t]);
                }
            }
            for (r, &t) in at.iter().zip(&checks) {
                let z = (r.mean() - exact[t]) / r.error();
                println!("{rule:?} t = {t}: simulated {:.5} ± {:.5}, exact {:.5} ({z:+.2})", r.mean(), r.error(), exact[t]);
                assert!(z.abs() < 4.5, "{rule:?} at {t}: {z}");
            }
        }
    }

    /// The SGD recursion is the source's Eq. 4 in closed form; `loss_at` (doubling) agrees
    /// with stepping for every kind of rule.
    #[test]
    fn the_recursions_are_the_source_and_doubling_agrees_with_stepping() {
        let nqm = Nqm::standard(50);
        let (rule, b) = (Rule::sgd(1.3), 7.0);
        let curve = nqm.losses(&rule, b, 300);
        for t in [0, 1, 10, 300] {
            let eq4: f64 = (0..nqm.dim())
                .map(|i| {
                    let (h, c) = (nqm.h[i], nqm.c[i]);
                    let r = (1.0 - rule.lr * h).powi(2 * t as i32);
                    r * 0.5 * h * nqm.start[i] + (1.0 - r) * rule.lr * c / (2.0 * b * (2.0 - rule.lr * h))
                })
                .sum();
            assert!(close(curve[t], eq4, 1e-12, 0.0), "t = {t}: {} vs {eq4}", curve[t]);
        }
        for rule in [
            Rule::sgd(1.0),
            Rule::heavy_ball(0.3, 0.9),
            Rule::preconditioned(0.5, 0.7),
            Rule::averaged(1.0, 0.95),
            Rule { lr: 0.2, momentum: 0.6, power: 0.5, average: 0.9 },
        ] {
            let curve = nqm.losses(&rule, 3.0, 1000);
            for t in [0u64, 1, 2, 3, 17, 256, 999, 1000] {
                let d = nqm.loss_at(&rule, 3.0, t);
                assert!(close(d, curve[t as usize], 1e-10, 0.0), "{rule:?} t = {t}: {d} vs {}", curve[t as usize]);
            }
        }
    }

    /// Each closed form of the stationary loss is the limit of the recursion (by
    /// doubling) and of a long run of it.
    #[test]
    fn the_stationary_closed_forms_are_the_limits() {
        let nqm = Nqm::new(vec![3.0, 1.0, 0.4, 0.05], vec![1.0, 0.5, 2.0, 0.3], vec![1.0; 4]);
        for rule in [
            Rule::sgd(0.6),
            Rule::heavy_ball(0.3, 0.85),
            Rule::preconditioned(0.4, 0.5),
            Rule { lr: 0.2, momentum: 0.5, power: 0.6, average: 0.0 },
            Rule::averaged(0.5, 0.7),
            Rule { lr: 0.3, momentum: 0.0, power: 0.4, average: 0.9 },
        ] {
            let closed = nqm.stationary_loss(&rule, 5.0).unwrap();
            let limit = nqm.with_system(&rule, 5.0, nqm_limit, nqm_limit, nqm_limit);
            let long = nqm.loss_at(&rule, 5.0, 1 << 40);
            println!("{rule:?}: closed form {closed:.10}, limit {limit:.10}");
            assert!(close(closed, limit, 1e-10, 0.0) && close(closed, long, 1e-10, 0.0), "{rule:?}: {closed} {limit} {long}");
        }
        assert_eq!(nqm.stationary_loss(&Rule::sgd(0.7), 1.0), None, "3 × 0.7 > 2: unstable");
        assert!(nqm.stationary_loss(&Rule::heavy_ball(0.7, 0.5), 1.0).is_some(), "3 × 0.7 < 3: stable with momentum");
        let both = Rule { lr: 0.2, momentum: 0.5, power: 0.0, average: 0.5 };
        assert!(close(nqm.stationary_loss(&both, 5.0).unwrap(), nqm.loss_at(&both, 5.0, 1 << 40), 1e-10, 0.0));
    }

    /// `steps_to` against stepping one step at a time, on the standard spectrum (200
    /// directions), for several rules, batch sizes and targets.
    #[test]
    fn steps_to_agrees_with_stepping() {
        let nqm = Nqm::standard(200);
        let rules = [
            Rule::sgd(1.5),
            Rule::heavy_ball(0.5, 0.9),
            Rule::preconditioned(0.5, 0.5),
            Rule::averaged(1.5, 0.9),
            Rule { lr: 0.3, momentum: 0.8, power: 0.0, average: 0.5 },
        ];
        for rule in rules {
            for b in [1.0, 30.0, 1000.0] {
                let curve = nqm.losses(&rule, b, 20_000);
                for target in [0.5, 0.05, 0.01] {
                    let first = curve.iter().position(|&l| l <= target).map(|t| t as u64);
                    let found = nqm.steps_to(target, &rule, b, 20_000);
                    assert_eq!(found, first, "{rule:?} B = {b} target {target}");
                }
            }
        }
        // the start already at the target, and a target never reached
        assert_eq!(nqm.steps_to(1e9, &Rule::sgd(1.0), 1.0, 10), Some(0));
        assert_eq!(nqm.steps_to(1e-9, &Rule::sgd(1.0), 1.0, 1000), None);
        assert_eq!(nqm.steps_to(0.01, &Rule::sgd(1.0), 1.0, 0), None);
    }

    /// The best learning rate found by the search against the best of 2000 rates evenly
    /// spaced in the logarithm over the same range: never more than one step worse.
    #[test]
    fn the_best_lr_against_a_fine_grid() {
        let nqm = Nqm::standard(100);
        for (b, momentum, power) in [(4.0, 0.0, 0.0), (64.0, 0.0, 0.0), (64.0, 0.9, 0.0), (16.0, 0.0, 0.5)] {
            let target = 0.02;
            let best = nqm.best_lr(target, b, momentum, power, 0.0, 1 << 30).unwrap();
            let top = (nqm.max_lr(momentum, power) * 0.999999).ln();
            let span = DECADES * std::f64::consts::LN_10;
            let fine = (0..2000)
                .filter_map(|i| nqm.steps_to(target, &Rule { lr: (top - span + span * i as f64 / 1999.0).exp(), momentum, power, average: 0.0 }, b, 1 << 30))
                .min()
                .unwrap();
            println!("B = {b}, β = {momentum}, p = {power}: search {} steps at lr {:.4}; grid of 2000: {fine}", best.steps, best.lr);
            assert!(best.steps <= fine + 1, "{} vs {fine}", best.steps);
            assert_eq!(nqm.steps_to(target, &Rule { lr: best.lr, momentum, power, average: 0.0 }, b, 1 << 30), Some(best.steps));
        }
        // and with momentum chosen from a grid: no worse than without
        let without = nqm.best_lr(0.02, 256.0, 0.0, 0.0, 0.0, 1 << 30).unwrap();
        let with = nqm.best_lr_momentum(0.02, 256.0, &MOMENTA, 0.0, 0.0, 1 << 30).unwrap();
        println!("B = 256: SGD {} steps; momentum {} at β {}: {} steps", without.steps, with.lr, with.momentum, with.steps);
        assert!(with.steps <= without.steps);
    }

    /// Nothing panics on an empty list of examples, one example, an empty problem, a
    /// model that does not fit, the smallest model `arch` builds (two parameters: `arch`
    /// has no model of one), operators of one and of no dimensions, NaN weights, and odd
    /// numbers for the noisy quadratic model.
    #[test]
    fn nothing_panics_on_edge_inputs() {
        let mut tape = Tape::new();
        let mut rng = Rng::new(141);
        // an empty list of examples, and indices beyond the data
        let (problem, mut model) = spiral_setup(5, 142);
        assert_eq!(problem.grad(&mut model, &mut tape, &[]), 0.0);
        assert!(model.groups().iter().all(|g| g.grad.iter().all(|&x| x == 0.0)));
        assert_eq!(problem.loss_on(&model, &mut tape, &[99, 100]), 0.0);
        let mut hv = Vec::new();
        let v = random_vector(n_params(&model), &mut rng);
        hvp(&problem, &mut model, &mut tape, &[], &v, &mut hv);
        assert!(hv.iter().all(|&x| x == 0.0));
        hvp(&problem, &mut model, &mut tape, &problem.all(), &[], &mut hv);
        assert!(hv.iter().all(|&x| x == 0.0) && hv.len() == n_params(&model));
        assert!(curvature_along(&problem, &mut model, &mut tape, &problem.all(), &[]).is_nan());
        let mut out = Vec::new();
        directional(&problem, &mut model, &mut tape, &[], &[1.0], &[], &mut out);
        assert!(out.is_empty());
        // one example
        let one = Problem::spirals(1, 2, &mut rng);
        let mut per = PerExample::new(&one, &model);
        assert!(per.advance(&one, &model, &mut tape, 10));
        let noise = per.noise();
        assert_eq!(noise.trace, 0.0);
        assert!(noise.trace_sample().is_nan());
        assert_eq!(noise.simple_noise_scale(), 0.0);
        assert_eq!(noise.expected_sq_norm(5, Draw::WithoutReplacement), noise.g2);
        let mut full = FullNoise::new(&one, &model);
        assert!(full.advance(&one, &mut model, &mut tape, 10));
        assert_eq!(full.trace_hs(), 0.0);
        let s = per.sample(3, Draw::WithReplacement, 10, Rng::new(1));
        assert!(s.cosine.is_finite() && s.sq_norm.is_finite());
        // an empty problem
        let empty = Problem::points(&[], &[], 0, Loss::Mse);
        assert!(empty.is_empty());
        assert_eq!(empty.loss(&model, &mut tape), 0.0);
        let mut per = PerExample::new(&empty, &model);
        assert!(per.advance(&empty, &model, &mut tape, 1));
        assert!(per.noise().simple_noise_scale().is_nan());
        assert!(per.row(0).is_empty() && per.loss(3).is_nan());
        let mut full = FullNoise::new(&empty, &model);
        assert!(full.advance(&empty, &mut model, &mut tape, 5));
        let mut l = Lanczos::new(&model, 2, 1);
        assert!(l.advance(&empty, &mut model, &mut tape, 30));
        assert_eq!(l.sharpness(), 0.0);
        let mut bs = BatchSharpness::new(Sampler::new(0, 4, Draw::WithoutReplacement, Rng::new(1)));
        bs.advance(&empty, &mut model, &mut tape, 3);
        assert_eq!(bs.skipped(), 3);
        assert!(bs.mean().is_nan());
        // models that do not fit
        let transformer = small_transformer(143);
        assert!(!problem.fits(&transformer));
        assert!(problem.loss(&transformer, &mut tape).is_nan());
        let mut wrong = mlp(3, 4, 1, 3, Act::Tanh, 144);
        assert!(problem.grad(&mut wrong, &mut tape, &[0, 1]).is_nan());
        let seqs = Problem::repeats(2, 6, 6, &mut rng);
        assert!(seqs.loss(&model, &mut tape).is_nan());
        let bad = Problem::points(&[1.0, 2.0, 3.0], &[0.0, -1.0, 2.5], 3, Loss::CrossEntropy);
        assert_eq!(bad.len(), 1, "classes that are not whole numbers from 0 are left out");
        let short = Problem::points(&[1.0, 2.0, 3.0, 4.0], &[0.0], 2, Loss::CrossEntropy);
        assert_eq!(short.len(), 1);
        // the smallest model: one weight in, one out, no biases
        let spec = MlpSpec { inputs: 1, width: 1, depth: 1, outputs: 1, bias: false, act: Act::Tanh, ..MlpSpec::default() };
        let mut tiny = Model::mlp(&spec, &mut rng);
        assert_eq!(n_params(&tiny), 2);
        let tp = Problem::points(&[0.5, -1.0], &[0.2, 0.1], 2, Loss::Mse);
        let h = Hessian::dense(&tp, &mut tiny, &mut tape);
        assert_eq!(h.values().len(), 2);
        let mut l = Lanczos::new(&tiny, 5, 2);
        assert!(l.advance(&tp, &mut tiny, &mut tape, 10));
        assert!(close(l.values()[0], h.values()[0], 1e-6, 1e-9) && l.values().len() == 2);
        assert_eq!(l.residual(), 0.0);
        // operators of one and of no dimensions
        let mut l1 = Lanczos::of_size(1, 3, 0);
        assert!(l1.advance_with(5, |v, out| {
            out.clear();
            out.push(3.0 * v[0]);
        }));
        assert!((l1.sharpness() - 3.0).abs() < 1e-12);
        let mut l0 = Lanczos::of_size(0, 1, 0);
        assert!(l0.advance_with(5, |_, out| out.clear()));
        assert!(l0.sharpness().is_nan());
        l0.restart();
        assert!(!Lanczos::of_size(4, 1, 0).advance_with(0, |_, _| {}));
        // NaN weights: carried, not unwrapped
        let mut nan = model.clone();
        nan.groups_mut()[0].value[0] = f64::NAN;
        assert!(problem.full_grad(&mut nan, &mut tape).is_nan());
        let mut l = Lanczos::new(&nan, 1, 3);
        assert!(!l.advance(&problem, &mut nan, &mut tape, 3));
        assert!(l.sharpness().is_nan());
        let mut h = Hessian::new(&nan);
        h.advance(&problem, &mut nan, &mut tape, usize::MAX);
        assert!(h.values().iter().all(|v| v.is_nan()));
        let mut per = PerExample::new(&problem, &nan);
        per.advance(&problem, &nan, &mut tape, usize::MAX);
        assert!(per.noise().simple_noise_scale().is_nan());
        // a preconditioner that is not positive
        let mut l = Lanczos::new(&model, 1, 4).preconditioned(&vec![-1.0; n_params(&model)]);
        l.advance(&problem, &mut model, &mut tape, 3);
        assert!(l.sharpness().is_nan());
        // plain numbers
        assert!(noise_factor(3, 0, Draw::WithReplacement).is_nan());
        assert_eq!(noise_factor(0, 5, Draw::WithReplacement), 1.0);
        assert_eq!(noise_factor(9, 5, Draw::WithoutReplacement), 0.0);
        assert_eq!(effective_batch(5, 5, Draw::WithoutReplacement), f64::INFINITY);
        assert!(two_batch_estimates(4.0, 1.0, 4.0, 1.0).0.is_nan());
        let zero = OneStep { g2: 0.0, ghg: 0.0, trhs: 0.0 };
        assert_eq!(zero.best_step(4.0), 0.0);
        assert_eq!(zero.best_gain(4.0), 0.0);
        let concave = OneStep { g2: 1.0, ghg: -2.0, trhs: 1.0 };
        assert_eq!(concave.best_step(1.0), f64::INFINITY);
        assert!(concave.noise_scale() < 0.0);
        assert_eq!(concave.best_step(0.5), f64::INFINITY, "c = −2 + 1/0.5 = 0: no minimum");
        assert!(close(concave.best_step(0.25), 0.5, 1e-15, 0.0), "c = −2 + 4 = 2 > 0: a minimum again");
        // the noisy quadratic model with odd numbers
        let none = Nqm::standard(0);
        assert_eq!(none.loss_at(&Rule::sgd(1.0), 1.0, 10), 0.0);
        assert_eq!(none.steps_to(0.01, &Rule::sgd(1.0), 1.0, 10), Some(0));
        assert!(none.best_lr(0.01, 1.0, 0.0, 0.0, 0.0, 10).is_none_or(|t| t.steps == 0));
        let nqm = Nqm::new(vec![1.0, 0.0, -0.5], vec![1.0, 1.0, 1.0], vec![1.0, 1.0]);
        assert_eq!(nqm.dim(), 2);
        assert!(nqm.loss_at(&Rule::sgd(f64::NAN), 1.0, 5).is_nan());
        assert!(nqm.loss_at(&Rule::sgd(0.5), 0.0, 5).is_infinite());
        assert_eq!(nqm.stationary_loss(&Rule::sgd(0.5), 1.0).map(|l| l > 0.0), Some(true));
        assert_eq!(nqm.steps_to(0.0, &Rule::sgd(5.0), 1.0, 1 << 20), None);
        let neg = Nqm::new(vec![-1.0], vec![1.0], vec![1.0]);
        assert!(neg.stationary_loss(&Rule::sgd(0.1), 1.0).is_none());
        assert!(neg.loss_at(&Rule::preconditioned(0.1, 0.5), 1.0, 3).is_nan());
        let mut sim_rng = Rng::new(5);
        assert_eq!(nqm.simulate(&Rule::sgd(0.1), 1.0, 0, &mut sim_rng).len(), 1);
        let mut t = Tuner::new(0.01, 1.0, &[], 0.0, 0.0, 100);
        while !t.advance(&nqm, 7) {}
        assert!(t.searches() > 0);
    }

    /// Timings in the build the tests run in (`--profile fast`): one gradient on 64
    /// spiral points for a 3-layer MLP 16 wide and on one sequence of 12 tokens for a
    /// 2-block transformer 16 wide; one Hessian-vector product; a sharpness estimate from
    /// cold and from a warm start; the per-example gradients; a search of the noisy
    /// quadratic model. Ignored: run it to print them.
    #[test]
    #[ignore]
    fn the_cost_of_measuring() {
        fn time(mut f: impl FnMut(), reps: usize) -> (f64, f64) {
            for _ in 0..reps.div_ceil(5) {
                f();
            }
            let mut runs: Vec<f64> = (0..5)
                .map(|_| {
                    let start = std::time::Instant::now();
                    for _ in 0..reps {
                        f();
                    }
                    start.elapsed().as_secs_f64() * 1e6 / reps as f64
                })
                .collect();
            runs.sort_by(f64::total_cmp);
            (runs[0], runs[2])
        }
        let mut tape = Tape::new();
        let problem = Problem::spirals(64, 3, &mut Rng::new(1));
        let mut model = mlp(2, 16, 3, 3, Act::Tanh, 2);
        train(&problem, &mut model, &mut tape, 100, 0.5);
        let all = problem.all();
        let p = n_params(&model);
        let (best, median) = time(|| _ = problem.grad(&mut model, &mut tape, &all), 300);
        println!("MLP 2 → 16 × 3 → 3, tanh ({p} parameters), 64 points: grad {best:.0} us (median {median:.0})");
        let (best, median) = time(|| _ = problem.loss(&model, &mut tape), 300);
        println!("  loss alone {best:.0} us (median {median:.0})");
        let v = random_vector(p, &mut Rng::new(3));
        let mut hv = Vec::new();
        let (best, median) = time(|| hvp(&problem, &mut model, &mut tape, &all, &v, &mut hv), 300);
        println!("  hvp {best:.0} us (median {median:.0})");
        let mut per = PerExample::new(&problem, &model);
        let (best, median) = time(
            || {
                per.reset();
                per.advance(&problem, &model, &mut tape, usize::MAX);
            },
            20,
        );
        println!("  per-example gradients of all 64: {best:.0} us (median {median:.0})");
        let mut lanczos = Lanczos::new(&model, 1, 4);
        let start = std::time::Instant::now();
        converge(&mut lanczos, 2000, |l| l.advance(&problem, &mut model, &mut tape, 1));
        let cold = (start.elapsed().as_secs_f64() * 1e3, lanczos.products());
        train(&problem, &mut model, &mut tape, 1, 0.5);
        lanczos.restart();
        let start = std::time::Instant::now();
        converge(&mut lanczos, 2000, |l| l.advance(&problem, &mut model, &mut tape, 1));
        let warm = (start.elapsed().as_secs_f64() * 1e3, lanczos.products());
        println!(
            "  sharpness {:.3} (tolerance 1e-4): cold {:.1} ms in {} products, warm (after one step of 0.5) {:.1} ms in {} products",
            lanczos.sharpness(),
            cold.0,
            cold.1,
            warm.0,
            warm.1
        );
        let mut l = Lanczos::new(&model, 1, 4);
        l.advance(&problem, &mut model, &mut tape, 21);
        let (best, median) = time(
            || {
                l.core.fresh = false;
                l.core.ritz();
            },
            100,
        );
        println!("  the Ritz values of a basis of 21: {best:.0} us (median {median:.0})");
        let (small_problem, mut small) = spiral_setup(64, 8);
        let start = std::time::Instant::now();
        let dense = Hessian::dense(&small_problem, &mut small, &mut tape);
        println!("dense Hessian of the 123-parameter MLP on 64 points: {:.1} ms", start.elapsed().as_secs_f64() * 1e3);
        let start = std::time::Instant::now();
        let _ = linalg::eig_sym(dense.matrix(), dense.size());
        println!("  its eig_sym alone: {:.1} ms", start.elapsed().as_secs_f64() * 1e3);
        let x = random_vector(300 * 300, &mut Rng::new(9));
        let sym: Vec<f64> = (0..300 * 300).map(|k| x[k] + x[(k % 300) * 300 + k / 300]).collect();
        let start = std::time::Instant::now();
        let _ = linalg::eig_sym(&sym, 300);
        println!("  eig_sym of a 300 by 300 matrix: {:.0} ms", start.elapsed().as_secs_f64() * 1e3);
        let spec = TransformerSpec { vocab: 10, context: 12, d_model: 16, heads: 2, layers: 2, d_mlp: 64, ..TransformerSpec::default() };
        let mut tf = Model::transformer(&spec, &mut Rng::new(5));
        let seqs = Problem::repeats(8, 12, 10, &mut Rng::new(6));
        let (best, median) = time(|| _ = seqs.grad(&mut tf, &mut tape, &[0]), 300);
        println!("transformer 2 blocks × 16 ({} parameters), one sequence of 12: grad {best:.0} us (median {median:.0})", n_params(&tf));
        let (best, median) = time(|| _ = seqs.grad(&mut tf, &mut tape, &seqs.all()), 50);
        println!("  8 sequences: grad {best:.0} us (median {median:.0})");
        let v = random_vector(n_params(&tf), &mut Rng::new(7));
        let (best, median) = time(|| hvp(&seqs, &mut tf, &mut tape, &[0], &v, &mut hv), 300);
        println!("  hvp on one sequence {best:.0} us (median {median:.0})");
        for (name, rule) in [
            ("SGD", Rule::sgd(0.3)),
            ("momentum", Rule::heavy_ball(0.03, 0.9)),
            ("momentum and averaging", Rule { lr: 0.03, momentum: 0.9, power: 0.0, average: 0.9 }),
        ] {
            let nqm = Nqm::standard(NQM_DIM);
            let start = std::time::Instant::now();
            let steps = nqm.steps_to(NQM_TARGET, &rule, 100.0, 1 << 30);
            println!("noisy quadratic model, 10⁴ directions, {name}: steps_to {steps:?} in {:.1} ms", start.elapsed().as_secs_f64() * 1e3);
        }
        let nqm = Nqm::standard(NQM_DIM);
        let start = std::time::Instant::now();
        let best = nqm.best_lr(NQM_TARGET, 100.0, 0.0, 0.0, 0.0, 1 << 30);
        println!("  best_lr for SGD at B = 100: {best:?} in {:.0} ms", start.elapsed().as_secs_f64() * 1e3);
    }
}

//! A mini-batch gradient: signal and noise. The loss is a mean over examples, so its
//! gradient G is the mean of one gradient per example; a mini-batch averages only B of
//! them, and its gradient is G plus noise. This scene looks closely at one such
//! gradient on a real small network: 512 points in the plane in three overlapping
//! classes, and a tanh network of 371 weights, so that each example's gradient is a
//! list of 371 numbers.
//!
//! Three pictures. (a) A window of two directions in that 371-dimensional space: along
//! G, and across it the direction in which the examples' gradients spread most. Every
//! example's gradient is a faint dot there, G an arrow, and forty mini-batch gradients
//! of the chosen size bright dots, with the ellipse of their exact covariance. (b) The
//! squared length of a mini-batch gradient against B on logarithmic axes: the signal
//! |G|², the noise tr Σ / B (exactly, with and without replacement), the noise of the
//! sampled batches, and where the two parts are equal: the simple noise scale of
//! McCandlish and co-authors (2018). (c) A run of SGD, measured every ten steps: the
//! noise scale, |G|² and tr Σ against the step. While it trains, the window is measured
//! again every fifty steps, and the picture is a film that follows the run one such
//! interval behind, drawing each dot straight between two measurements (see
//! [`MiniBatch::roll`]), so that it moves evenly whatever the frame rate.
//!
//! Every number comes from the library: the per-example gradients and their
//! statistics (`terrain::PerExample`, `Noise`), the second direction of the window
//! (`terrain::Lanczos` on the operator P Σ P), the training rule (`optim::Optimizer`,
//! plain SGD). The batches of the picture are nested (see [`Lists`]), so that each
//! one can be followed as it grows.

use super::*;
use crate::clock::{self, Budget};
use crate::math::arch::{Init, Loss, MlpSpec, Model};
use crate::math::autograd::{Act, Tape};
use crate::math::optim::{Hyper, Optimizer, Rule};
use crate::math::terrain::{self, Draw, Lanczos, Noise, PerExample, Problem, Sampler};
use crate::scenes::stats::plot::{self, Glide, Plot, Rect, AXES, MARKS};

pub const DEF: SceneDef = SceneDef {
    id: "minibatch",
    title: "A mini-batch gradient: signal and noise",
    group: GROUP,
    blurb: "The gradient of a few examples is the true gradient plus noise that shrinks as the batch grows: the batch size at which the two are equal is the noise scale.",
    make: |_| Box::new(MiniBatch::create()),
};

// ----- the network and its data ------------------------------------------------------------

/// The examples: points in the plane in three classes.
const N: usize = 512;
/// The three clouds' centres are this far from the origin, a third of a turn apart, and
/// each is a round normal cloud of this standard deviation: they overlap.
const RADIUS: f64 = 1.0;
const SPREAD: f64 = 0.8;
/// The hidden layers: two of 16 tanh units.
const WIDTH: usize = 16;
const DEPTH: usize = 2;
/// The seeds of the data, of the network's starting weights, of the mini-batches drawn
/// for the picture, and of the batches the training run deals.
const DATA_SEED: u64 = 5;
const NET_SEED: u64 = 202;
const BATCH_SEED: u64 = 31;
const TRAIN_SEED: u64 = 7;
/// The mini-batches drawn for the picture, for each way of drawing.
const K: usize = 40;

/// The data set: N points, class k of point i being i mod 3, its cloud centred at angle
/// 90° + 120° k.
fn data() -> Problem {
    let mut rng = Rng::new(DATA_SEED);
    let (mut x, mut y) = (Vec::with_capacity(2 * N), Vec::with_capacity(N));
    for i in 0..N {
        let c = i % 3;
        let a = std::f64::consts::FRAC_PI_2 + std::f64::consts::TAU * c as f64 / 3.0;
        x.push(RADIUS * a.cos() + SPREAD * rng.normal() as f64);
        x.push(RADIUS * a.sin() + SPREAD * rng.normal() as f64);
        y.push(c as f64);
    }
    Problem::points(&x, &y, N, Loss::CrossEntropy)
}

/// The network as it starts: 2 inputs, two hidden layers of 16 tanh units, 3 outputs
/// (one score per class, a softmax and cross-entropy), Xavier's start.
fn network() -> Model {
    let spec = MlpSpec { inputs: 2, width: WIDTH, depth: DEPTH, outputs: 3, act: Act::Tanh, init: Init::Xavier, ..MlpSpec::default() };
    Model::mlp(&spec, &mut Rng::new(NET_SEED))
}

// ----- small vector helpers ----------------------------------------------------------------

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

// ----- the mini-batches of the picture -------------------------------------------------------

/// The two ways of drawing, in the order the controls list them.
const DRAWS: [Draw; 2] = [Draw::WithReplacement, Draw::WithoutReplacement];

fn draw_index(d: Draw) -> usize {
    match d {
        Draw::WithReplacement => 0,
        Draw::WithoutReplacement => 1,
    }
}

/// The mini-batches the picture shows: K for each way of drawing, nested, so that batch
/// k of size B is the first B examples of a list fixed once and for all. With
/// replacement the list is N independent uniform draws; without, a random order of all
/// N examples. Each batch of size B is then exactly a batch of its kind (B independent
/// draws; a uniformly random subset of B), and as B grows a batch gains examples and
/// keeps the ones it had, so that the picture can follow each batch as it grows.
/// (Batches of different sizes are therefore not independent of each other; the K
/// batches of one size are.)
#[derive(Clone, Debug)]
struct Lists {
    lists: [Vec<Vec<usize>>; 2],
}

impl Lists {
    fn new(seed: u64) -> Lists {
        let mut rng = Rng::new(seed);
        let with = (0..K).map(|_| (0..N).map(|_| rng.below(N)).collect()).collect();
        let without = (0..K)
            .map(|_| {
                let mut order: Vec<usize> = (0..N).collect();
                rng.shuffle(&mut order);
                order
            })
            .collect();
        Lists { lists: [with, without] }
    }

    /// Batch k of size b (taken as 1 to N), drawn as `draw` says.
    fn batch(&self, draw: Draw, k: usize, b: usize) -> &[usize] {
        let list = &self.lists[draw_index(draw)][k.min(K - 1)];
        &list[..b.clamp(1, N)]
    }
}

// ----- what is measured at one moment of training -------------------------------------------

/// Everything the picture shows of the gradients at one set of weights: the statistics
/// of the per-example gradients over all the parameters (the library's [`Noise`]: the
/// true gradient G, |G|², the trace of the per-example covariance Σ, divisor N), and
/// the window of two directions the dots are drawn in.
#[derive(Clone, Debug)]
struct Snapshot {
    /// Training steps taken to these weights.
    step: u64,
    /// The loss on all the data.
    loss: f64,
    noise: Noise,
    /// The window's two unit directions: e₁ along G, e₂ across it (see [`WindowJob`]).
    e: [Vec<f64>; 2],
    /// Each example's gradient seen in the window: (gₙ · e₁, gₙ · e₂).
    proj: Vec<[f64; 2]>,
    /// The per-example covariance in the window (divisor N): e₁ᵀΣe₁, e₁ᵀΣe₂, e₂ᵀΣe₂.
    cov: [f64; 3],
    /// Each example's gradient's length over |G|, sorted.
    lengths: Vec<f64>,
    /// The mean over the examples of the cosine between gₙ and G: the expected cosine of
    /// a batch of one, exactly.
    cos_one: f64,
}

impl Snapshot {
    /// |G|. (The tests hold the picture's numbers to these.)
    #[cfg(test)]
    fn g(&self) -> f64 {
        self.noise.g2.sqrt()
    }

    /// The share of tr Σ that the window shows: (e₁ᵀΣe₁ + e₂ᵀΣe₂) / tr Σ.
    fn shown(&self) -> f64 {
        (self.cov[0] + self.cov[2]) / self.noise.trace
    }

    /// The median example's gradient length over |G|.
    fn median_length(&self) -> f64 {
        self.lengths.get(self.lengths.len() / 2).copied().unwrap_or(f64::NAN)
    }

    /// The window position of batch k of size b drawn as `draw` says: the mean of its
    /// examples' positions (the window is linear).
    #[cfg(test)]
    fn batch_at(&self, lists: &Lists, draw: Draw, k: usize, b: usize) -> [f64; 2] {
        batch_mean(&self.proj, lists.batch(draw, k, b))
    }

    /// The covariance in the window of a batch's gradient of size b: the per-example
    /// covariance there times the library's [`terrain::noise_factor`] (1/B with
    /// replacement, (N − B)/(B(N − 1)) without; exact).
    #[cfg(test)]
    fn batch_cov(&self, draw: Draw, b: usize) -> [f64; 3] {
        let k = terrain::noise_factor(b, N, draw);
        self.cov.map(|c| c * k)
    }

    /// The noise of a batch of b, exactly: E|ξ|² = tr Cov(g_B).
    fn noise_at(&self, draw: Draw, b: f64) -> f64 {
        noise_line(&self.noise, draw, b)
    }

    /// Where noise and signal are equal, drawn as `draw` says (the library's
    /// `half_noise_batch`: B_simple with replacement, (1/B_simple' + 1/N)⁻¹ without).
    fn crossing(&self, draw: Draw) -> f64 {
        self.noise.half_noise_batch(draw)
    }
}

/// E|ξ|² for a batch of b (a real number, for the chart's lines) drawn from the N
/// examples: tr Σ / b with replacement, (1/b − 1/N) tr Σₛ = (N − b)/(b (N − 1)) tr Σ
/// without (0 from b = N on). The same as `Noise::expected_sq_norm(b) − |G|²` at whole
/// b (a test holds them together).
fn noise_line(noise: &Noise, draw: Draw, b: f64) -> f64 {
    let n = noise.n as f64;
    match draw {
        Draw::WithReplacement => noise.trace / b,
        Draw::WithoutReplacement if b >= n => 0.0,
        Draw::WithoutReplacement => noise.trace * (n - b) / (b * (n - 1.0)),
    }
}

/// The mean of the points `batch` of `points` (NaN for a point not there).
fn batch_mean(points: &[[f64; 2]], batch: &[usize]) -> [f64; 2] {
    let mut s = [0.0; 2];
    for &i in batch {
        let q = points.get(i).copied().unwrap_or([f64::NAN; 2]);
        s[0] += q[0];
        s[1] += q[1];
    }
    let n = batch.len().max(1) as f64;
    [s[0] / n, s[1] / n]
}

/// The noise of one batch, measured over every parameter: ξ = g_B − G.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
struct BatchNoise {
    /// |ξ|².
    sq: f64,
    /// G · ξ.
    along: f64,
}

/// The noise of the picture's batches at every size, over every parameter: for each way
/// of drawing, for each batch k, its |ξ|² and G·ξ at sizes 1 to N. Measured a batch at
/// a time ([`Samples::more`]).
#[derive(Clone, Debug, Default)]
struct Samples {
    of: [Vec<Vec<BatchNoise>>; 2],
}

impl Samples {
    /// Batch k's noise at size b, drawn as `draw` says.
    fn get(&self, draw: Draw, k: usize, b: usize) -> Option<BatchNoise> {
        self.of[draw_index(draw)].get(k)?.get(b.checked_sub(1)?).copied()
    }

    /// Is every batch of both ways measured?
    fn complete(&self) -> bool {
        self.of.iter().all(|v| v.len() == K)
    }

    /// The mean over the K batches of size b of |ξ|², and its standard error.
    fn mean_sq(&self, draw: Draw, b: usize) -> Option<(f64, f64)> {
        mean_se((0..K).map(|k| self.get(draw, k, b).map(|s| s.sq)))
    }

    /// The cosine between batch k's gradient G + ξ and G.
    fn cosine(&self, g2: f64, draw: Draw, k: usize, b: usize) -> Option<f64> {
        let s = self.get(draw, k, b)?;
        let len = (g2 + 2.0 * s.along + s.sq).max(0.0).sqrt();
        let c = (g2 + s.along) / (len * g2.sqrt());
        c.is_finite().then_some(c)
    }

    /// The mean over the K batches of size b of that cosine, and its standard error.
    fn mean_cosine(&self, g2: f64, draw: Draw, b: usize) -> Option<(f64, f64)> {
        mean_se((0..K).map(|k| self.cosine(g2, draw, k, b)))
    }

    /// More batches, until the budget is spent (at least one) or all are measured: N p
    /// operations for each batch of each way of drawing, its running sum growing one
    /// example at a time. Returns whether all are measured.
    fn more(&mut self, per: &PerExample, noise: &Noise, budget: &Budget) -> bool {
        if per.len() < N || noise.mean.len() != per.width() {
            // (nothing to measure them with: no batch has a number)
            self.of = [vec![Vec::new(); K], vec![Vec::new(); K]];
            return true;
        }
        let lists = Lists::new(BATCH_SEED);
        let mut sum = vec![0.0; per.width()];
        loop {
            let Some(d) = (0..2).find(|&d| self.of[d].len() < K) else { return true };
            let k = self.of[d].len();
            sum.iter_mut().for_each(|x| *x = 0.0);
            let mut out = Vec::with_capacity(N);
            for (b, &i) in lists.batch(DRAWS[d], k, N).iter().enumerate() {
                axpy(1.0, per.row(i), &mut sum);
                // ξ = sum / B − G
                let inv = 1.0 / (b + 1) as f64;
                let (mut sq, mut along) = (0.0, 0.0);
                for (s, g) in sum.iter().zip(&noise.mean) {
                    let x = s * inv - g;
                    sq += x * x;
                    along += x * g;
                }
                out.push(BatchNoise { sq, along });
            }
            self.of[d].push(out);
            if budget.spent() {
                return self.complete();
            }
        }
    }
}

/// The mean of some numbers and its standard error (None if any is missing or there
/// are fewer than two).
fn mean_se(values: impl Iterator<Item = Option<f64>>) -> Option<(f64, f64)> {
    let v: Vec<f64> = values.collect::<Option<Vec<f64>>>()?;
    if v.len() < 2 {
        return None;
    }
    let n = v.len() as f64;
    let mean = v.iter().sum::<f64>() / n;
    let var = v.iter().map(|x| (x - mean) * (x - mean)).sum::<f64>() / (n - 1.0);
    Some((mean, (var / n).sqrt()))
}

/// The most Lanczos products spent on the window's second direction (it converges in
/// about eight here: see the tests).
const MOST_PRODUCTS: usize = 60;

/// The window's second direction: the one across G in which the examples' gradients
/// spread most, that is the top eigenvector of P Σ P, with P the projection that
/// removes the part along e₁ = G/|G| and Σ the per-example covariance (divisor N). It
/// is found by the library's Lanczos iteration on that operator (2 N p operations a
/// product). Its sign is chosen to agree with `like` (the last window's), if given,
/// else so that the examples' third moment along it is positive. Where |G| is 0 or not
/// a number, e₁ is 0 and e₂ the top eigenvector of Σ itself.
struct WindowJob {
    e1: Vec<f64>,
    lanczos: Lanczos,
    products: usize,
    like: Option<Vec<f64>>,
}

impl WindowJob {
    fn new(noise: &Noise, like: Option<Vec<f64>>) -> WindowJob {
        let p = noise.mean.len();
        let g = noise.g2.sqrt();
        let e1 = if g > 0.0 && g.is_finite() { noise.mean.iter().map(|x| x / g).collect() } else { vec![0.0; p] };
        WindowJob { e1, lanczos: Lanczos::of_size(p, 1, BATCH_SEED), products: 0, like }
    }

    /// Up to `products` more products; true when converged (or out of products).
    fn advance(&mut self, per: &PerExample, noise: &Noise, products: usize) -> bool {
        if self.done() {
            return true;
        }
        let e1 = &self.e1;
        let mean = &noise.mean;
        let n = per.len().max(1) as f64;
        let mut w = Vec::new();
        let left = products.min(MOST_PRODUCTS - self.products);
        let converged = self.lanczos.advance_with(left, |v: &[f64], out: &mut Vec<f64>| {
            // out = P Σ P v, with Σ = (1/N) Σₙ dₙ dₙᵀ and dₙ = gₙ − G
            w.clear();
            w.extend_from_slice(v);
            let a = dot(e1, &w);
            axpy(-a, e1, &mut w);
            out.clear();
            out.resize(v.len(), 0.0);
            let gw = dot(mean, &w);
            let mut total = 0.0;
            for k in 0..per.len() {
                let c = dot(per.row(k), &w) - gw;
                axpy(c, per.row(k), out);
                total += c;
            }
            axpy(-total, mean, out);
            out.iter_mut().for_each(|x| *x /= n);
            let b = dot(e1, out);
            axpy(-b, e1, out);
        });
        self.products += left;
        converged || self.products >= MOST_PRODUCTS
    }

    fn done(&self) -> bool {
        self.lanczos.converged() || self.products >= MOST_PRODUCTS
    }

    /// The two directions, once done.
    fn directions(&self, per: &PerExample, noise: &Noise) -> [Vec<f64>; 2] {
        let p = self.e1.len();
        let mut e2: Vec<f64> = self.lanczos.vector(0).to_vec();
        if e2.len() != p || !e2.iter().all(|x| x.is_finite()) {
            e2 = vec![0.0; p];
        }
        // (kept exactly across e₁, and of length 1)
        let a = dot(&self.e1, &e2);
        axpy(-a, &self.e1, &mut e2);
        let len = norm(&e2);
        if len > 0.0 {
            e2.iter_mut().for_each(|x| *x /= len);
        }
        let flip = match &self.like {
            Some(old) if old.len() == p => dot(old, &e2) < 0.0,
            _ => {
                let ge = dot(&noise.mean, &e2);
                let third: f64 = (0..per.len()).map(|k| (dot(per.row(k), &e2) - ge).powi(3)).sum();
                third < 0.0
            }
        };
        if flip {
            e2.iter_mut().for_each(|x| *x = -*x);
        }
        [self.e1.clone(), e2]
    }
}

/// What a measurement gives: always a mark for the training record; with the window,
/// also a snapshot for the picture.
struct Measured {
    mark: Mark,
    snapshot: Option<Snapshot>,
}

/// The measurement of the present weights, a slice at a time: the per-example
/// gradients (one pass of one example each), their statistics, and (if wanted) the
/// window and the projections.
struct Measure {
    step: u64,
    per: PerExample,
    noise: Option<Noise>,
    window: Option<WindowJob>,
    want_window: bool,
    like: Option<Vec<f64>>,
}

impl Measure {
    fn new(problem: &Problem, model: &Model, step: u64, want_window: bool, like: Option<Vec<f64>>) -> Measure {
        Measure { step, per: PerExample::new(problem, model), noise: None, window: None, want_window, like }
    }

    /// Work until the budget is spent (at least one unit); what was measured, once
    /// done. (The weights must not change in between.)
    fn work(&mut self, problem: &Problem, model: &Model, tape: &mut Tape, budget: &Budget) -> Option<Measured> {
        loop {
            if !self.per.is_done() {
                self.per.advance(problem, model, tape, 16);
            } else if self.noise.is_none() {
                let noise = self.per.noise();
                if !self.want_window {
                    let mark = self.mark(&noise);
                    self.noise = Some(noise);
                    return Some(Measured { mark, snapshot: None });
                }
                self.window = Some(WindowJob::new(&noise, self.like.take()));
                self.noise = Some(noise);
            } else if let (Some(w), Some(noise)) = (&mut self.window, &self.noise) {
                if w.advance(&self.per, noise, 4) {
                    let snapshot = self.finish();
                    return Some(Measured { mark: self.mark(&snapshot.noise), snapshot: Some(snapshot) });
                }
            } else {
                // (measured already)
                let noise = self.noise.clone().unwrap_or_else(|| self.per.noise());
                return Some(Measured { mark: self.mark(&noise), snapshot: None });
            }
            if budget.spent() {
                return None;
            }
        }
    }

    fn loss(&self) -> f64 {
        let n = self.per.len().max(1);
        (0..self.per.len()).map(|k| self.per.loss(k)).sum::<f64>() / n as f64
    }

    fn mark(&self, noise: &Noise) -> Mark {
        Mark { step: self.step, g2: noise.g2, trace: noise.trace, loss: self.loss() }
    }

    fn finish(&self) -> Snapshot {
        let noise = self.noise.clone().unwrap_or_else(|| self.per.noise());
        let e = match &self.window {
            Some(w) => w.directions(&self.per, &noise),
            None => [vec![0.0; noise.mean.len()], vec![0.0; noise.mean.len()]],
        };
        let n = self.per.len();
        let proj: Vec<[f64; 2]> = (0..n).map(|k| [dot(self.per.row(k), &e[0]), dot(self.per.row(k), &e[1])]).collect();
        let m = [proj.iter().map(|p| p[0]).sum::<f64>() / n.max(1) as f64, proj.iter().map(|p| p[1]).sum::<f64>() / n.max(1) as f64];
        let mut cov = [0.0; 3];
        for p in &proj {
            let (a, b) = (p[0] - m[0], p[1] - m[1]);
            cov[0] += a * a;
            cov[1] += a * b;
            cov[2] += b * b;
        }
        cov.iter_mut().for_each(|c| *c /= n.max(1) as f64);
        let g = noise.g2.sqrt();
        let mut lengths: Vec<f64> = (0..n).map(|k| norm(self.per.row(k)) / g).collect();
        lengths.sort_by(|a, b| a.total_cmp(b));
        let cos_one = (0..n).map(|k| dot(self.per.row(k), &noise.mean) / (norm(self.per.row(k)) * g)).map(|c| if c.is_finite() { c } else { 0.0 }).sum::<f64>() / n.max(1) as f64;
        Snapshot { step: self.step, loss: self.loss(), noise, e, proj, cov, lengths, cos_one }
    }

    /// The per-example gradients it measured.
    fn rows(self) -> PerExample {
        self.per
    }
}

/// A snapshot with what more is needed to draw it: its per-example gradients, and the
/// noise of the picture's batches over every parameter (measured once the scene is at
/// rest).
#[derive(Clone, Debug)]
struct Shown {
    snap: Snapshot,
    rows: PerExample,
    samples: Samples,
}

impl Shown {
    /// Measure the weights at once (the scene as it opens): all but the batches' noise
    /// over every parameter, which the frames measure a slice at a time.
    fn now(problem: &Problem, model: &Model, step: u64) -> Shown {
        let mut tape = Tape::new();
        let mut m = Measure::new(problem, model, step, true, None);
        let budget = Budget::new(f64::INFINITY);
        let snap = loop {
            if let Some(Measured { snapshot: Some(s), .. }) = m.work(problem, model, &mut tape, &budget) {
                break s;
            }
        };
        Shown { snap, rows: m.rows(), samples: Samples::default() }
    }
}

// ----- training --------------------------------------------------------------------------------

/// One point of the training run's record.
#[derive(Clone, Copy, Debug, PartialEq)]
struct Mark {
    step: u64,
    g2: f64,
    trace: f64,
    loss: f64,
}

impl Mark {
    /// The simple noise scale at this step's weights.
    fn noise_scale(&self) -> f64 {
        self.trace / self.g2
    }
}

/// Steps between two measurements during training, and between two pictures of the
/// window.
const EVERY: u64 = 10;
const WINDOW_EVERY: u64 = 50;
/// The measurements a smoothed noise scale is taken over: the last hundred steps.
const SMOOTH: usize = 10;
/// Training steps a second of the scene's time (at most: measuring takes time too).
const RATE: f64 = 400.0;

/// The ratio of the averages of tr Σ and of |G|² over the last `SMOOTH` marks up to mark
/// i (fewer at the start): the way McCandlish and co-authors (2018, App. A.1) turn
/// noisy measurements of the two into a noise scale, averaging each separately and
/// dividing the averages (theirs are moving averages over steps).
fn smoothed(marks: &[Mark], i: usize) -> f64 {
    if marks.is_empty() {
        return f64::NAN;
    }
    let i = i.min(marks.len() - 1);
    let w = &marks[(i + 1).saturating_sub(SMOOTH)..=i];
    let (t, g) = w.iter().fold((0.0, 0.0), |(t, g), m| (t + m.trace, g + m.g2));
    t / g
}

/// A stretch of the run with one setting: from which step, the step size, the batch.
#[derive(Clone, Copy, Debug, PartialEq)]
struct Stretch {
    from: u64,
    lr: f64,
    batch: usize,
}

/// The training run: plain SGD (the library's `Rule::Sgd`, θ ← θ − η g) with batches
/// dealt an epoch at a time without replacement (the library's `Sampler`), measured
/// every `EVERY` steps.
struct Run {
    model: Model,
    opt: Optimizer,
    sampler: Sampler,
    steps: u64,
    target: u64,
    running: bool,
    marks: Vec<Mark>,
    stretches: Vec<Stretch>,
    pending: Option<Measure>,
    /// Steps owed by the scene's time.
    owed: f64,
    /// The weights are no longer finite numbers.
    diverged: bool,
}

impl Run {
    fn new(start: &Mark) -> Run {
        let model = network();
        let opt = Optimizer::new(&model, Rule::Sgd, Hyper::default_for(Rule::Sgd));
        Run {
            model,
            opt,
            sampler: Sampler::new(N, 16, Draw::WithoutReplacement, Rng::new(TRAIN_SEED)),
            steps: 0,
            target: 0,
            running: false,
            marks: vec![*start],
            stretches: Vec::new(),
            pending: None,
            owed: 0.0,
            diverged: false,
        }
    }

    /// One step at the setting of the last stretch.
    fn step(&mut self, problem: &Problem, tape: &mut Tape) {
        let Some(s) = self.stretches.last().copied() else { return };
        if self.sampler.size() != s.batch.clamp(1, N) {
            self.sampler.set_batch(s.batch);
        }
        let idx = self.sampler.deal().to_vec();
        problem.grad(&mut self.model, tape, &idx);
        self.opt.step(&mut self.model, s.lr);
        self.steps += 1;
    }

    /// Train `steps` more at this setting.
    fn more(&mut self, steps: u64, lr: f64, batch: usize) {
        if self.diverged {
            return;
        }
        let s = Stretch { from: self.steps, lr, batch };
        match self.stretches.last_mut() {
            Some(last) if last.from == self.steps => *last = s,
            Some(last) if last.lr == lr && last.batch == batch => {}
            _ => self.stretches.push(s),
        }
        self.target = self.steps + steps;
        self.running = true;
    }

    /// The smoothed noise scale at the last mark.
    fn smoothed_now(&self) -> f64 {
        smoothed(&self.marks, self.marks.len().saturating_sub(1))
    }
}

// ----- the settings and the experiments -------------------------------------------------------

/// What the picture and the training depend on.
#[derive(Clone, Copy, Debug, PartialEq)]
struct Setup {
    /// The mini-batch size of the picture.
    batch: usize,
    draw: Draw,
    /// The training run's step size and batch, and the steps a press of "train" takes.
    lr: f64,
    train_batch: usize,
    steps: u64,
}

const MOST_LR: f64 = 3.0;
const LEAST_LR: f64 = 0.01;
const MOST_STEPS: u64 = 5000;

impl Setup {
    fn base() -> Setup {
        Setup { batch: 1, draw: Draw::WithReplacement, lr: 0.3, train_batch: 16, steps: 1000 }
    }

    /// Kept to the controls' ranges (a value that is not a number takes the base's).
    fn kept(self) -> Setup {
        let base = Setup::base();
        Setup {
            batch: self.batch.clamp(1, N),
            lr: if self.lr.is_finite() { self.lr.max(LEAST_LR).min(MOST_LR) } else { base.lr },
            train_batch: self.train_batch.clamp(1, N),
            steps: self.steps.clamp(EVERY, MOST_STEPS),
            ..self
        }
    }
}

struct Trial {
    button: &'static str,
    caption: &'static str,
}

const TRIALS: [Trial; 5] = [
    Trial {
        button: "one example at a time",
        caption: r"$B = 1$: each bright dot is one example's gradient, forty drawn from the 512. A typical one (the median) is 2.47 times as long as the true gradient $G$, and over all 512 the mean of its cosine with $G$ is 0.354: each example pulls its own way, and only their mean is $G$. In squared length its noise is on average 5.91 times the signal: that number is the noise scale $B_{\text{simple}}$. The formula $1/\sqrt{1 + B_{\text{simple}}/B}$ gives 0.380 for the cosine: an approximation, a little high here.",
    },
    Trial {
        button: "a batch the size of the noise scale",
        caption: r"$B = 6$, about $B_{\text{simple}} = 5.91$: the noise's expected squared length, $\operatorname{tr}\Sigma/B = 0.976$, is within 2 % of the signal's, $|G|^2 = 0.991$, so about half of $\mathbb{E}\,|g_B|^2$ is noise. The cloud of batches has drawn in round the arrow's tip, $\sqrt 6 \approx 2.45$ times narrower than the examples' own. The forty batches' mean cosine with $G$ is $0.75 \pm 0.05$ (two standard errors); the formula gives 0.71.",
    },
    Trial {
        button: "a batch ten times the noise scale",
        caption: r"$B = 59$: the noise is a tenth of the signal (0.0993 against 0.991), and the batches sit close round the tip: their mean cosine with $G$ is $0.957 \pm 0.008$, the formula's 0.953. Drawn without replacement (the button under the window) the noise is smaller again, by the factor $(N - B)/(N - 1) = 0.89$: a batch of 59 is no longer a small part of 512.",
    },
    Trial {
        button: "the whole data set, without replacement",
        caption: r"$B = 512$ drawn without replacement: every batch holds every example once, in some order, so its gradient is $G$ exactly, and every bright dot is on the tip of the arrow; on the chart the noise falls away to nothing at $N$. Drawn with replacement, a batch of 512 repeats some examples and misses others, and keeps a noise of $\operatorname{tr}\Sigma/512 = 0.0114$, 1.2 % of the signal.",
    },
    Trial {
        button: "train, then look again",
        caption: r"From the start, 1000 steps of SGD with $\eta = 0.3$ and batches of 16 dealt an epoch at a time, all 512 examples measured every 10 steps. The loss falls from 1.17 to 0.61 in the first 100 steps, then only to 0.58. Over steps 200 to 1000, $|G|^2$ is on average 29 times smaller than at the start and $\operatorname{tr}\Sigma$ two thirds of what it was, so the noise scale (the ratio of their averages) is 115, 20 times the start's 5.91. From one measurement to the next it jumps, between 30 and 4 900: at a constant step the weights keep moving, as SGD's do about the bottom of a bowl (scene noisefloor), and $|G|$ depends on where they are. A batch of 16, the run's own, is now mostly noise. McCandlish and co-authors (2018) report that the noise scale grew as the loss fell in their runs: in their table its average over a run is 5 to 18 times its start for four image classifiers, and the same as at the start for an autoencoder.",
    },
];

/// The note under the training controls (its numbers are measured: see the tests).
const NOTE_TRAINING: &str = r"Plain SGD, $\theta \leftarrow \theta - \eta\,g_B$, with batches dealt an epoch at a time; every 10 steps all 512 examples' gradients are measured, and every 50 the window is drawn again. A new setting takes effect from the next step. The smoothed line divides the average of $\operatorname{tr}\Sigma$ over the last 100 steps by that of $|G|^2$, as McCandlish and co-authors (2018) do with moving averages: in large runs they estimate the two from the squared lengths of gradients of two batch sizes, estimates that are each unbiased while their ratio is not. They also report that the noise scale measured on a run depends on its step size: lowering $\eta/B$ 16-fold raised it about 16-fold (an image classifier trained by SGD, a language model by Adam), so they measure it on runs with a well-tuned step. Here, over steps 200 to 1000, it is 433 at $\eta = 0.1$ against 115 at $\eta = 0.3$.";

/// The scene opens as the first experiment sets it up.
const OPENING: usize = 0;

/// The settings of experiment k, and whether it trains from the start.
fn trial(k: usize) -> (Setup, bool) {
    let base = Setup::base();
    match k {
        1 => (Setup { batch: 6, ..base }, false),
        2 => (Setup { batch: 59, ..base }, false),
        3 => (Setup { batch: N, draw: Draw::WithoutReplacement, ..base }, false),
        4 => (Setup { batch: 16, ..base }, true),
        _ => (base, false),
    }
}

// ----- numbers as the scene writes them ----------------------------------------------------

/// A number with three significant digits (0.0129, 0.354, 5.91, 123), with a true minus
/// sign; thousands set apart.
fn sig(v: f64) -> String {
    if !v.is_finite() {
        return "?".to_string();
    }
    if v == 0.0 {
        return "0".to_string();
    }
    let a = v.abs();
    if a >= 1000.0 {
        let n = count(a.round() as usize);
        return if v < 0.0 { format!("−{n}") } else { n };
    }
    let decimals = (2 - a.log10().floor() as i32).clamp(0, 12) as usize;
    num(v, decimals)
}

/// A setting as the labels write it: as few digits as it needs (0.3, 0.025, 1.6).
fn short(v: f64) -> String {
    let text = sig(v);
    if text.contains('.') {
        text.trim_end_matches('0').trim_end_matches('.').to_string()
    } else {
        text
    }
}

/// A batch size or noise scale as the picture writes it: two significant digits under
/// 10 (5.9), whole numbers above (123, 1 700).
fn batch_text(v: f64) -> String {
    if !v.is_finite() {
        "?".to_string()
    } else if v < 10.0 {
        num(v, 1)
    } else {
        count(v.round() as usize)
    }
}

/// How a way of drawing is named.
fn draw_name(d: Draw) -> &'static str {
    match d {
        Draw::WithReplacement => "with replacement",
        Draw::WithoutReplacement => "without replacement",
    }
}

// ----- the scene -----------------------------------------------------------------------------

/// What is drawn, gliding to where it is going. Lengths in the window are in the
/// gradient's own units; on the charts, logarithms.
#[derive(Clone, Debug, Default)]
struct Look {
    /// The window's range: x from and to, y from and to (made to the window's shape
    /// when drawn).
    window: Glide,
    /// Each example's dot (x, y each).
    examples: Glide,
    /// Each batch's dot (x, y each).
    batches: Glide,
    /// |G|; the examples' covariance in the window, and a batch's (three numbers each).
    shape: Glide,
    /// Chart (b): log₁₀ |G|², log₁₀ tr Σ, its x range's top, its y range; log₁₀ B; the
    /// crossing's log₁₀ B.
    chart: Glide,
    /// Chart (b): each batch's log₁₀ |ξ|² at the chosen size, and whether they are
    /// measured (for the moment the picture shows).
    noise: Glide,
    noise_seen: bool,
    /// Chart (c): the steps across, the y range of the noise scale and of the strip.
    history: Glide,
    moving: bool,
}

// The chart's numbers, in the order the glide keeps them.
const C_G2: usize = 0;
const C_TRACE: usize = 1;
const C_XHI: usize = 2;
const C_YLO: usize = 3;
const C_YHI: usize = 4;
const C_B: usize = 5;
const C_CROSS: usize = 6;
const C_CROSS_OTHER: usize = 7;

pub struct MiniBatch {
    problem: Problem,
    tape: Tape,
    setup: Setup,
    /// The picture's mini-batches.
    lists: Lists,
    /// The network as it starts, measured once.
    start: Shown,
    /// The last moment measured with its window.
    shown: Shown,
    run: Run,
    /// The snapshots the picture goes through, in order: from the last one at or before
    /// the film's step on.
    reel: Vec<Snapshot>,
    /// The step the picture shows. While the run goes on it follows the run at `RATE`
    /// steps a second, a window's interval (`WINDOW_EVERY` steps) behind the last
    /// snapshot, so that there is always one on either side of it to draw between.
    film: f64,
    /// The ranges of the run's charts, held while a run is being drawn (so that its
    /// axes do not move in fits as it reaches higher or lower).
    held: Option<[f32; 4]>,
    look: Look,
    /// The window last drawn in (a new one is drawn at once, without gliding).
    canvas: Option<Vec2>,
    /// The batch size is being dragged on the chart.
    dragging: bool,
    placed: Vec<Placed>,
    labels_moving: bool,
}

impl MiniBatch {
    /// (Nothing here needs the graphics device, so the tests can make one too.)
    fn create() -> MiniBatch {
        let problem = data();
        let start = Shown::now(&problem, &network(), 0);
        let mut s = MiniBatch {
            problem,
            tape: Tape::new(),
            setup: Setup::base(),
            lists: Lists::new(BATCH_SEED),
            reel: vec![start.snap.clone()],
            shown: start.clone(),
            run: Run::new(&start.mark()),
            start,
            film: 0.0,
            held: None,
            look: Look::default(),
            canvas: None,
            dragging: false,
            placed: Vec::new(),
            labels_moving: false,
        };
        s.set_up(OPENING);
        s
    }

    /// Set up experiment k (and start its training, if it has any).
    fn set_up(&mut self, k: usize) {
        let (setup, train) = trial(k);
        self.setup = setup;
        self.back_to_start();
        if train {
            self.train();
        }
    }

    /// The network as it started, its record cleared.
    fn back_to_start(&mut self) {
        self.shown = self.start.clone();
        self.run = Run::new(&self.start.mark());
        self.reel = vec![self.start.snap.clone()];
        self.film = 0.0;
        self.held = None;
    }

    /// Train a press's worth of steps from where the run is.
    fn train(&mut self) {
        self.setup = self.setup.kept();
        let s = self.setup;
        self.run.more(s.steps, s.lr, s.train_batch);
        if self.run.running {
            self.held = Some(self.history_fit(true));
        }
    }

    /// Pause the run (the weights where it stopped are measured, for the picture).
    fn pause(&mut self) {
        self.run.running = false;
        self.run.owed = 0.0;
    }

    /// The step of the last snapshot.
    fn last_step(&self) -> f64 {
        self.reel.last().map_or(0.0, |s| s.step as f64)
    }

    /// Is anything still to be computed, or the film still to reach the last snapshot?
    fn busy(&self) -> bool {
        self.run.running || self.run.pending.is_some() || !self.shown.samples.complete() || self.film < self.last_step() || self.paused_unmeasured()
    }

    /// The run stopped at weights the picture has not measured.
    fn paused_unmeasured(&self) -> bool {
        !self.run.running && !self.run.diverged && self.shown.snap.step != self.run.steps
    }

    /// This frame's share of the work: training paced by the scene's time (`dt`), the
    /// measurements it needs, and then, at rest, the noise of the picture's batches.
    fn work(&mut self, dt: f32, budget: &Budget) {
        if self.run.running && dt > 0.0 {
            self.run.owed = (self.run.owed + RATE * dt as f64).min(RATE * 0.1 + 2.0);
        }
        loop {
            if let Some(m) = self.run.pending.as_mut() {
                match m.work(&self.problem, &self.run.model, &mut self.tape, budget) {
                    Some(done) => {
                        let m = self.run.pending.take();
                        self.take(done, m);
                    }
                    None => return,
                }
            } else if self.run.running && self.run.steps < self.run.target {
                if self.run.owed < 1.0 {
                    return;
                }
                self.run.step(&self.problem, &mut self.tape);
                self.run.owed -= 1.0;
                let t = self.run.steps;
                if t.is_multiple_of(EVERY) || t == self.run.target {
                    let window = t.is_multiple_of(WINDOW_EVERY) || t == self.run.target;
                    let like = Some(self.shown.snap.e[1].clone());
                    self.run.pending = Some(Measure::new(&self.problem, &self.run.model, t, window, like));
                }
            } else if self.run.running {
                self.pause();
            } else if self.paused_unmeasured() {
                let like = Some(self.shown.snap.e[1].clone());
                self.run.pending = Some(Measure::new(&self.problem, &self.run.model, self.run.steps, true, like));
            } else if !self.shown.samples.complete() {
                let Shown { snap, rows, samples } = &mut self.shown;
                if samples.more(rows, &snap.noise, budget) && snap.step == 0 && !self.start.samples.complete() {
                    // (the start's, kept for the next return to it)
                    self.start.samples = samples.clone();
                }
            } else {
                return;
            }
            if budget.spent() {
                return;
            }
        }
    }

    /// A measurement has come in.
    fn take(&mut self, done: Measured, m: Option<Measure>) {
        let mark = done.mark;
        let finite = mark.g2.is_finite() && mark.trace.is_finite() && mark.loss.is_finite();
        // (a pause just after a measurement measures the same weights again)
        if self.run.marks.last().is_none_or(|last| last.step != mark.step) {
            self.run.marks.push(mark);
        }
        if let (Some(snap), Some(m)) = (done.snapshot, m) {
            if finite {
                if self.reel.last().is_some_and(|last| last.step == snap.step) {
                    self.reel.pop();
                }
                self.reel.push(snap.clone());
                self.shown = Shown { snap, rows: m.rows(), samples: Samples::default() };
            }
        }
        if !finite {
            self.run.diverged = true;
            self.pause();
        }
    }

    /// Move the film on by `dt` seconds of the scene's time.
    fn roll(&mut self, dt: f32) {
        let first = self.reel.first().map_or(0.0, |s| s.step as f64);
        let lag = if self.run.running { WINDOW_EVERY as f64 } else { 0.0 };
        let limit = (self.last_step() - lag).max(first);
        if dt > 0.0 {
            self.film += RATE * dt as f64;
        }
        self.film = self.film.max(first).min(limit);
        // (the snapshots the film has passed are let go)
        while self.reel.len() >= 2 && self.reel[1].step as f64 <= self.film {
            self.reel.remove(0);
        }
        // (the run drawn to its end: its charts fit what it did)
        if !self.run.running && self.film >= self.last_step() {
            self.held = None;
        }
    }

    /// The two snapshots on either side of the film, and how far it is from the first to
    /// the second (0 to 1).
    fn between(&self) -> (&Snapshot, &Snapshot, f64) {
        let a = &self.reel[0];
        let b = self.reel.get(1).unwrap_or(a);
        let span = (b.step as f64 - a.step as f64).max(1e-9);
        let u = if b.step > a.step { ((self.film - a.step as f64) / span).clamp(0.0, 1.0) } else { 0.0 };
        (a, b, u)
    }

    /// Everything done at once, the film at its end (the tests).
    #[cfg(test)]
    fn finish(&mut self) {
        let budget = Budget::new(f64::INFINITY);
        let mut guard = 0;
        while self.busy() {
            self.work(1.0, &budget);
            self.roll(1.0);
            guard += 1;
            assert!(guard < 100_000, "the work does not end");
        }
    }
}

impl Shown {
    /// The training record's point for this moment.
    fn mark(&self) -> Mark {
        Mark { step: self.snap.step, g2: self.snap.noise.g2, trace: self.snap.noise.trace, loss: self.snap.loss }
    }
}

// ----- the layout ------------------------------------------------------------------------

/// Where everything goes in a window (logical pixels from its top-left corner).
#[derive(Clone, Copy, Debug, PartialEq)]
struct Layout {
    view: Vec2,
    narrow: bool,
    text: f32,
    small: f32,
    /// The window's square and its title's place.
    window: Rect,
    window_head: Vec2,
    /// The buttons under the window: with replacement, without.
    draw_buttons: [Rect; 2],
    /// Chart (b) and its title's place.
    chart: Rect,
    chart_head: Vec2,
    /// Chart (c), its strip, its title's place, and its buttons: train (or pause), back
    /// to the start.
    history: Rect,
    strip: Rect,
    history_head: Vec2,
    train_buttons: [Rect; 2],
    /// Only the window: there is no room for the rest.
    compact: bool,
}

impl Layout {
    fn of(view: Vec2, inset: f32) -> Layout {
        let view = view.max(Vec2::splat(1.0));
        let narrow = view.x < 700.0;
        let (text, small) = if narrow { (12.0, 10.5) } else { (13.5, 11.5) };
        let side = if narrow { 10.0 } else { 22.0 };
        let top = inset.max(0.0) + if narrow { 6.0 } else { 12.0 };
        let r = |x0: f32, y0: f32, x1: f32, y1: f32| (Vec2::new(x0, y0), Vec2::new(x1.max(x0 + 1.0), y1.max(y0 + 1.0)));
        let none = r(0.0, 0.0, 0.0, 0.0);
        let head = text + 10.0;
        let button_h = if narrow { 24.0 } else { 26.0 };
        let mut lay = Layout {
            view,
            narrow,
            text,
            small,
            window: none,
            window_head: Vec2::new(side, top),
            draw_buttons: [none; 2],
            chart: none,
            chart_head: Vec2::ZERO,
            history: none,
            strip: none,
            history_head: Vec2::ZERO,
            train_buttons: [none; 2],
            compact: false,
        };
        // (the charts' rectangles leave room for their tick numbers at the left and below)
        let left = if narrow { 40.0 } else { 48.0 };
        let two = view.x >= 860.0;
        let stack_needs = head + 150.0 + button_h + 16.0 + head + 110.0 + 40.0 + head + 70.0 + 36.0 + 40.0 + button_h;
        if view.x < 250.0 || (!two && view.y - top < stack_needs) || (two && view.y - top < 520.0) {
            lay.compact = true;
            let s = (view.x - 2.0 * side).min(view.y - top - head - 8.0).max(1.0);
            let x0 = (view.x - s) * 0.5;
            lay.window = r(x0, top + head, x0 + s, top + head + s);
            return lay;
        }
        // the history: title, chart, strip, buttons; from y0 to y1
        let history_in = |lay: &mut Layout, x0: f32, x1: f32, y0: f32, y1: f32| {
            lay.history_head = Vec2::new(x0, y0 + head - 4.0);
            let buttons_y = y1 - button_h - 4.0;
            let row = plot::even_row(2, x0, x1, buttons_y, button_h, 8.0);
            lay.train_buttons = [row[0], row[1]];
            let strip_h = ((buttons_y - y0 - head) * 0.25).clamp(32.0, 90.0);
            let bottom = buttons_y - 32.0;
            lay.strip = r(x0 + left, bottom - strip_h, x1 - 8.0, bottom);
            lay.history = r(x0 + left, y0 + head + 8.0, x1 - 8.0, lay.strip.0.y - 18.0);
        };
        if two {
            // the window and the chart side by side, the run under them
            // (the run's chart is given at most so much height; the rest goes to the top row)
            let below = ((view.y - top) * 0.44).min(460.0);
            let s = ((view.x - 3.0 * side) * 0.48).min(view.y - top - below - head - button_h - 14.0).max(1.0);
            lay.window_head = Vec2::new(side, top);
            lay.window = r(side, top + head, side + s, top + head + s);
            let row = plot::even_row(2, side, side + s, lay.window.1.y + 8.0, button_h, 8.0);
            lay.draw_buttons = [row[0], row[1]];
            let bottom = lay.draw_buttons[0].1.y.max(view.y - below - 12.0);
            let x0 = 2.0 * side + s;
            lay.chart_head = Vec2::new(x0, top + lay.text);
            lay.chart = r(x0 + left, top + head + 8.0, view.x - side - 8.0, bottom - 32.0);
            history_in(&mut lay, side, view.x - side, bottom + 12.0, view.y - 4.0);
        } else {
            // one under another
            let width = view.x - 2.0 * side;
            let s = width.min((view.y - top) * if view.y < 700.0 { 0.31 } else { 0.36 }).max(1.0);
            let x0 = (view.x - s) * 0.5;
            lay.window_head = Vec2::new(side, top);
            lay.window = r(x0, top + head, x0 + s, top + head + s);
            let row = plot::even_row(2, side, view.x - side, lay.window.1.y + 8.0, button_h, 8.0);
            lay.draw_buttons = [row[0], row[1]];
            let y0 = lay.window.1.y + 8.0 + button_h + 4.0;
            let split = y0 + (view.y - y0) * 0.45;
            lay.chart_head = Vec2::new(side, y0 + head - 4.0);
            lay.chart = r(side + left, y0 + head + 8.0, view.x - side - 8.0, split - 32.0);
            history_in(&mut lay, side, view.x - side, split, view.y - 4.0);
        }
        lay
    }
}

/// The height of the window's key: its lines and a margin (none where the window is
/// all there is room for).
fn key_height(lay: &Layout) -> f32 {
    if lay.compact {
        return 0.0;
    }
    let lines = if lay.narrow { 4.0 } else { 5.0 };
    lines * (lay.small + 6.0) + 8.0
}

// ----- what is drawn, gliding ----------------------------------------------------------------

/// How far the window reaches round the examples' cloud, in its standard deviations.
const REACH: f64 = 2.6;
/// A value that is not a number glides as this and is not drawn.
const NOWHERE: f32 = -30.0;
/// How many points each line of the charts is drawn with.
const CURVE_N: usize = 120;

/// What the picture shows at the film's step: the snapshots on either side of it,
/// mixed (each example's dot straight from one place to the other; the squared lengths
/// and the variances on a logarithmic scale).
struct Moment {
    step: f64,
    g2: f64,
    trace: f64,
    /// The examples' covariance in the window.
    cov: [f64; 3],
    proj: Vec<[f64; 2]>,
}

impl Moment {
    fn g(&self) -> f64 {
        self.g2.max(0.0).sqrt()
    }

    /// As the library's `Noise`, for the formulas (without the vectors).
    fn noise(&self) -> Noise {
        Noise { n: N, mean: Vec::new(), variances: Vec::new(), g2: self.g2, trace: self.trace }
    }

    fn shown(&self) -> f64 {
        (self.cov[0] + self.cov[2]) / self.trace
    }
}

impl MiniBatch {
    /// The moment the film is at.
    fn moment(&self) -> Moment {
        let (a, b, u) = self.between();
        let mix = |x: f64, y: f64| x + (y - x) * u;
        let mix_log = |x: f64, y: f64| if x > 0.0 && y > 0.0 { (x.ln() + (y.ln() - x.ln()) * u).exp() } else { mix(x, y) };
        Moment {
            step: self.film,
            g2: mix_log(a.noise.g2, b.noise.g2),
            trace: mix_log(a.noise.trace, b.noise.trace),
            cov: [mix_log(a.cov[0], b.cov[0]), mix(a.cov[1], b.cov[1]), mix_log(a.cov[2], b.cov[2])],
            proj: a.proj.iter().zip(&b.proj).map(|(p, q)| [mix(p[0], q[0]), mix(p[1], q[1])]).collect(),
        }
    }

    /// The window's range as it is to be: round the examples' cloud (and the origin).
    fn window_goal(m: &Moment) -> [f32; 4] {
        let g = m.g();
        let (sx, sy) = (m.cov[0].max(0.0).sqrt(), m.cov[2].max(0.0).sqrt());
        let spread = sx.max(sy).max(g * 0.5);
        let (sx, sy) = (sx.max(spread * 0.4), sy.max(spread * 0.4));
        let x0 = (g - REACH * sx).min(-0.15 * spread);
        let x1 = (g + REACH * sx).max(g + 0.15 * spread);
        let (y0, y1) = (-REACH * sy, REACH * sy);
        let ok = [x0, x1, y0, y1].iter().all(|v| v.is_finite()) && x1 > x0 && y1 > y0;
        if ok {
            [x0 as f32, x1 as f32, y0 as f32, y1 as f32]
        } else {
            [-1.0, 1.0, -1.0, 1.0]
        }
    }

    /// Chart (b)'s numbers as they are to be.
    fn chart_goal(&self, m: &Moment) -> Vec<f32> {
        let noise = m.noise();
        let (g2, trace) = (m.g2, m.trace);
        let cross = noise.half_noise_batch(Draw::WithReplacement);
        let cross_without = 1.0 / (g2 / (trace * N as f64 / (N - 1) as f64) + 1.0 / N as f64);
        // across: from 1 to N, or further where the noise scale lies beyond N
        let x_hi = (N as f64).max(if cross.is_finite() { cross * 2.0 } else { 0.0 }).min(1e7).log10();
        // up: from under the signal and the noise at the right edge to over the noise at B = 1
        let lo = g2.min(trace / 10f64.powf(x_hi)).max(1e-30);
        let hi = g2.max(trace).max(1e-30);
        let (mut y_lo, mut y_hi) = (lo.log10() - 0.8, hi.log10() + 0.9);
        if !(y_lo.is_finite() && y_hi.is_finite()) {
            (y_lo, y_hi) = (-2.0, 1.0);
        }
        if y_hi - y_lo < 2.0 {
            y_hi = y_lo + 2.0;
        }
        let f = |v: f64| if v.is_finite() { v as f32 } else { NOWHERE };
        vec![
            f(g2.max(1e-30).log10()),
            f(trace.max(1e-30).log10()),
            x_hi as f32,
            y_lo as f32,
            y_hi as f32,
            (self.setup.batch.max(1) as f32).log10(),
            f(cross.max(1e-30).log10()),
            f(cross_without.max(1e-30).log10()),
        ]
    }

    /// The ranges of the run's charts that fit its record (log₁₀): the noise scale's,
    /// then the strip's. `ahead`: wide enough for the run about to be drawn (the noise
    /// scale up to 3 000 times its start, |G|² down to a ten-thousandth), so that its
    /// axes need not move while it is.
    fn history_fit(&self, ahead: bool) -> [f32; 4] {
        let marks = &self.run.marks;
        let (mut lo, mut hi) = (f64::INFINITY, 0.0f64);
        let (mut slo, mut shi) = (f64::INFINITY, 0.0f64);
        for m in marks {
            let b = m.noise_scale();
            if b.is_finite() && b > 0.0 {
                lo = lo.min(b);
                hi = hi.max(b);
            }
            for v in [m.g2, m.trace] {
                if v.is_finite() && v > 0.0 {
                    slo = slo.min(v);
                    shi = shi.max(v);
                }
            }
        }
        if ahead {
            if let Some(m) = marks.last() {
                let b = m.noise_scale();
                if b.is_finite() && b > 0.0 {
                    lo = lo.min(b / 2.0);
                    hi = hi.max(b * 3000.0);
                }
                if m.g2.is_finite() && m.g2 > 0.0 && m.trace.is_finite() {
                    slo = slo.min(m.g2 * 1e-4);
                    shi = shi.max(m.trace * 3.0);
                }
            }
        }
        let range = |lo: f64, hi: f64, least: f64| {
            if !(lo.is_finite() && hi.is_finite() && lo > 0.0 && hi >= lo) {
                return (0.0, 1.0);
            }
            let (a, mut b) = (lo.log10() - 0.15, hi.log10() + 0.15);
            if b - a < least {
                b = a + least;
            }
            (a as f32, b as f32)
        };
        let (y0, y1) = range(lo, hi, 1.0);
        let (s0, s1) = range(slo, shi, 1.0);
        // (room above the lines for the loss)
        let s1 = s1 + 0.35 * (s1 - s0);
        [y0, y1, s0, s1]
    }

    /// The steps across the run's charts: to the run's end, or to where a press of train
    /// would take it.
    fn history_steps(&self) -> f64 {
        self.run.target.max(self.run.steps).max(self.setup.kept().steps) as f64
    }

    /// Chart (c)'s ranges as they are to be: the steps across, the noise scale's range,
    /// the strip's range.
    fn history_goal(&self) -> [f32; 5] {
        let fit = self.held.unwrap_or_else(|| self.history_fit(false));
        [self.history_steps() as f32, fit[0], fit[1], fit[2], fit[3]]
    }

    /// Bring the picture `dt` seconds nearer to what it shows. (`snap`: there at once.)
    fn follow(&mut self, dt: f32, snap: bool) {
        let m = self.moment();
        let (b, draw) = (self.setup.batch, self.setup.draw);
        let keep = |v: f64| if v.is_finite() { v as f32 } else { NOWHERE };
        let window = Self::window_goal(&m);
        let examples: Vec<f32> = m.proj.iter().flat_map(|p| [keep(p[0]), keep(p[1])]).collect();
        let batches: Vec<f32> = (0..K).flat_map(|k| batch_mean(&m.proj, self.lists.batch(draw, k, b))).map(keep).collect();
        let factor = terrain::noise_factor(b, N, draw);
        let shape: Vec<f32> = [m.g(), m.cov[0], m.cov[1], m.cov[2], factor * m.cov[0], factor * m.cov[1], factor * m.cov[2]].map(keep).to_vec();
        let chart = self.chart_goal(&m);
        // (the batches' own noise is measured for the last snapshot, once the film is
        // there; while it is not, their dots keep their places, unseen, and glide on from
        // there once it is; no noise at all is far under the chart)
        let there = (self.film - self.shown.snap.step as f64).abs() < 1e-9;
        let seen = self.shown.samples.complete() && there;
        let floor = keep((m.g2 * 1e-12).max(1e-300).log10());
        let noise: Vec<f32> = if seen {
            (0..K)
                .map(|k| match self.shown.samples.get(draw, k, b) {
                    Some(n) if n.sq > 0.0 => keep(n.sq.log10()),
                    _ => floor,
                })
                .collect()
        } else if self.look.noise.shown.len() == K {
            self.look.noise.shown.clone()
        } else {
            vec![NOWHERE; K]
        };
        let history = self.history_goal();
        let look = &mut self.look;
        if snap {
            look.window.snap(&window);
            look.examples.snap(&examples);
            look.batches.snap(&batches);
            look.shape.snap(&shape);
            look.chart.snap(&chart);
            look.noise.snap(&noise);
            look.history.snap(&history);
            look.moving = false;
            return;
        }
        let mut moving = look.window.follow(&window, dt, false);
        moving |= look.examples.follow(&examples, dt, false);
        moving |= look.batches.follow(&batches, dt, false);
        moving |= look.shape.follow(&shape, dt, false);
        moving |= look.chart.follow(&chart, dt, false);
        // (the dots placed for the first time appear where they are)
        if look.noise.shown.len() != noise.len() || look.noise.shown.contains(&NOWHERE) {
            look.noise.snap(&noise);
        } else {
            moving |= look.noise.follow(&noise, dt, false);
        }
        look.noise_seen = seen;
        moving |= look.history.follow(&history, dt, false);
        look.moving = moving;
    }
}

// ----- drawing ---------------------------------------------------------------------------------

// Lines whose number of points may change, each in a batch of its own.
const W_AXES: u8 = plot::OWN;
const W_RING: u8 = plot::OWN + 1;
const W_BATCH_RING: u8 = plot::OWN + 2;
const W_ARROW: u8 = plot::OWN + 3;
const B_SIGNAL: u8 = plot::OWN + 4;
const B_NOISE: u8 = plot::OWN + 5;
const B_OTHER: u8 = plot::OWN + 6;
const B_CROSS: u8 = plot::OWN + 7;
const B_CHOSEN: u8 = plot::OWN + 8;
const B_MEANS: u8 = plot::OWN + 9;
const B_BEYOND: u8 = plot::OWN + 10;
const H_SMOOTH: u8 = plot::OWN + 11;
const H_G2: u8 = plot::OWN + 12;
const H_TRACE: u8 = plot::OWN + 13;
const H_NOW: u8 = plot::OWN + 14;
const H_STRETCH: u8 = plot::OWN + 15;
const B_TICKS_Y: u8 = plot::OWN + 16;
const B_TICKS_X: u8 = plot::OWN + 17;
const H_TICKS_Y: u8 = plot::OWN + 18;
const LAST_KIND: u8 = H_TICKS_Y;

/// The point batches: examples, batches, the chart's batch dots, the history's dots, the
/// crossings.
const DOTS: usize = 5;

fn dot_batch<'b>(f: &'b mut Frame<'_>, k: usize) -> &'b mut crate::gfx::draw::PointBatch {
    f.dl.points_with(PointKind::Overlay, PointStyle { outline_px: 0.0, size_scale: 1.0 + 1e-4 * (k + 1) as f32, ..Default::default() })
}

/// Every batch the scene draws in, made in the same order in every frame, so that one
/// that is empty in a frame does not move the later ones up a place (see `eidos motion`).
fn hold_batches(f: &mut Frame) {
    let corner = f.hud(Vec2::ZERO);
    for kind in [AXES, MARKS, plot::BUTTONS].into_iter().chain(W_AXES..=LAST_KIND) {
        plot::apart(f, kind, false).segment(corner, corner, 1.0, Color::CLEAR);
    }
    for kind in [B_OTHER, B_CROSS, H_STRETCH] {
        plot::apart(f, kind, true).segment(corner, corner, 1.0, Color::CLEAR);
    }
    for k in 0..DOTS {
        dot_batch(f, k);
    }
}

/// Dots in screen space, in point batch k: every one of `pts`, those outside `rect` (or
/// not numbers) unseen at its edge, so that a dot that leaves does not shift the others
/// along the batch (`eidos motion` follows a point by its place in it).
fn dots_in(f: &mut Frame, k: usize, pts: &[Vec2], (min, max): Rect, size: f32, color: Color) {
    let at: Vec<(Vec3, Color)> = pts
        .iter()
        .map(|p| {
            let seen = p.is_finite() && plot::within(*p, (min, max), 0.5);
            let q = if p.is_finite() { p.max(min).min(max) } else { min };
            (f.hud(q), if seen { color } else { Color::CLEAR })
        })
        .collect();
    let b = dot_batch(f, k);
    for (p, c) in at {
        b.push(p, size, c);
    }
}

/// The ticks of a logarithmic axis of `pl` (up the left side, or along the foot with
/// `across`) at powers of ten (and with `fives`, 2 and 5 times them), in batch `kind`:
/// every one of a fixed set of them, those outside the chart unseen at its edge, so
/// that while the range glides each tick keeps its place in the batch. `grid`: a faint
/// line across the chart at each. Returns the ticks in the chart.
fn log_axis(f: &mut Frame, kind: u8, pl: &Plot, across: bool, fives: bool, grid: bool) -> Vec<f64> {
    let (min, max) = pl.rect;
    let mut inside = Vec::new();
    for d in -20..=20 {
        for m in [1.0f64, 2.0, 5.0] {
            if m > 1.0 && !fives {
                continue;
            }
            let v = d as f64 + m.log10();
            let (lo, hi) = if across { (pl.x[0], pl.x[1]) } else { (pl.y[0], pl.y[1]) };
            let seen = v >= lo - 1e-6 && v <= hi + 1e-6;
            let c = |a: f32| if seen { pal::DIM.a(a) } else { Color::CLEAR };
            let long = if m == 1.0 { 4.0 } else { 2.0 };
            if across {
                let x = pl.px_x(v).max(min.x).min(max.x);
                plot::line_of(f, kind, Vec2::new(x, max.y), Vec2::new(x, max.y + long), 1.0, c(0.8));
            } else {
                let y = pl.px_y(v).max(min.y).min(max.y);
                plot::line_of(f, kind, Vec2::new(min.x - long, y), Vec2::new(min.x, y), 1.0, c(0.8));
                if grid {
                    plot::line_of(f, kind, Vec2::new(min.x, y), Vec2::new(max.x, y), 1.0, c(0.1));
                }
            }
            if seen {
                inside.push(v);
            }
        }
    }
    inside
}

/// An arrow in screen space, in its own batch.
fn arrow_px(f: &mut Frame, kind: u8, a: Vec2, b: Vec2, width: f32, color: Color) {
    let d = (b - a).normalize_or_zero();
    let n = Vec2::new(-d.y, d.x);
    let head = (5.0 + 2.5 * width).min((b - a).length() * 0.6);
    plot::line_of(f, kind, a, b, width, color);
    plot::line_of(f, kind, b, b - d * head + n * head * 0.5, width, color);
    plot::line_of(f, kind, b, b - d * head - n * head * 0.5, width, color);
}

/// The ellipse of two standard deviations of a covariance (xx, xy, yy) round a centre,
/// in the data's units, as points: the circle of radius 2 through the covariance's
/// Cholesky factor L (L Lᵀ = the covariance), which changes smoothly with it, so that
/// the points of the drawn ellipse do not jump round it when its axes swap.
fn ellipse(centre: [f64; 2], c: [f64; 3]) -> Vec<(f64, f64)> {
    let a = c[0].max(0.0);
    let l11 = a.sqrt();
    let l21 = if l11 > 0.0 { c[1] / l11 } else { 0.0 };
    let l22 = (c[2] - l21 * l21).max(0.0).sqrt();
    (0..=72)
        .map(|k| {
            let (sn, cs) = (k as f64 / 72.0 * std::f64::consts::TAU).sin_cos();
            let (u, v) = (2.0 * cs, 2.0 * sn);
            (centre[0] + l11 * u, centre[1] + l21 * u + l22 * v)
        })
        .collect()
}

/// log₁₀ ticks: the whole decades in a range, and (where there are few) 2 and 5 of each.
fn log_ticks(lo: f64, hi: f64) -> Vec<(f64, bool)> {
    let mut v = Vec::new();
    if !(lo.is_finite() && hi.is_finite() && hi > lo) {
        return v;
    }
    let few = hi - lo < 1.6;
    let mut d = lo.floor() as i32;
    while (d as f64) <= hi + 1e-9 && v.len() < 80 {
        for (m, main) in [(1.0f64, true), (2.0, false), (5.0, false)] {
            let x = d as f64 + m.log10();
            if x >= lo - 1e-9 && x <= hi + 1e-9 && (main || few) {
                v.push((x, main || few));
            }
        }
        d += 1;
    }
    v
}

/// The ticks of a logarithmic axis `pixels` long, every decade where they have room for
/// their numbers (`size` high), else every second or third decade.
fn axis_ticks(lo: f64, hi: f64, pixels: f32, size: f32) -> Vec<(f64, bool)> {
    let ticks = log_ticks(lo, hi);
    let per = pixels as f64 / (hi - lo).max(1e-9);
    let named = ticks.iter().filter(|t| t.1).count().max(1) as f32;
    if pixels / named >= size + 4.0 {
        return ticks;
    }
    let stride = ((size as f64 + 6.0) / per).ceil().max(1.0) as i64;
    ticks.into_iter().filter(|&(x, _)| (x - x.round()).abs() < 1e-6 && (x.round() as i64).rem_euclid(stride) == 0).collect()
}

/// A power of ten as an axis writes it: 1, 10, 100, 1 000 across; $10^{-2}$ up.
fn power_name(x: f64, up: bool) -> String {
    let v = 10f64.powf(x);
    let d = x.round() as i32;
    if up && (x - d as f64).abs() < 1e-6 {
        return match d {
            0 => "1".to_string(),
            1 => "10".to_string(),
            _ => format!("$10^{{{d}}}$"),
        };
    }
    if v >= 1.0 {
        count(v.round() as usize)
    } else {
        sig(v)
    }
}

impl MiniBatch {
    /// The window's plot: its range made to the square's shape, the same scale both ways.
    fn window_plot(&self, lay: &Layout) -> Plot {
        let g = |i: usize| self.look.window.get(i) as f64;
        let (x0, x1, y0, y1) = (g(0), g(1), g(2), g(3));
        // (the dots stay above the key at the foot of the window)
        let (min, max) = lay.window;
        let rect = (min, Vec2::new(max.x, (max.y - key_height(lay)).max(min.y + 1.0)));
        let (w, h) = ((rect.1.x - rect.0.x).max(1.0) as f64, (rect.1.y - rect.0.y).max(1.0) as f64);
        let per_px = ((x1 - x0) / w).max((y1 - y0) / h).max(1e-30);
        let (cx, cy) = (0.5 * (x0 + x1), 0.5 * (y0 + y1));
        Plot::new(rect, [cx - 0.5 * w * per_px, cx + 0.5 * w * per_px], [cy - 0.5 * h * per_px, cy + 0.5 * h * per_px])
    }

    fn chart_plot(&self, lay: &Layout) -> Plot {
        let g = |i: usize| self.look.chart.get(i) as f64;
        // (from a little under 1, so that the batches of one are clear of the axis)
        Plot::new(lay.chart, [-0.08, g(C_XHI).max(0.3)], [g(C_YLO), g(C_YHI)])
    }

    fn history_plots(&self, lay: &Layout) -> (Plot, Plot) {
        let g = |i: usize| self.look.history.get(i) as f64;
        (Plot::new(lay.history, [0.0, g(0).max(1.0)], [g(1), g(2)]), Plot::new(lay.strip, [0.0, g(0).max(1.0)], [g(3), g(4)]))
    }

    /// The buttons on the picture; true if anything changed.
    fn buttons(&mut self, f: &mut Frame, lay: &Layout) -> bool {
        let size = lay.small;
        let mut changed = false;
        let names = if lay.narrow { ["with replacement", "without (an epoch)"] } else { ["drawn with replacement", "without (as an epoch deals)"] };
        for (k, d) in DRAWS.into_iter().enumerate() {
            if plot::button(f, lay.draw_buttons[k], names[k], size, self.setup.draw == d) && self.setup.draw != d {
                self.setup.draw = d;
                changed = true;
            }
        }
        let busy = self.run.running;
        let train = if busy {
            "❚❚ pause".to_string()
        } else if self.run.diverged {
            "training diverged".to_string()
        } else {
            format!("▶ train {} steps", count(self.setup.kept().steps as usize))
        };
        if plot::button(f, lay.train_buttons[0], &train, size, busy) && !self.run.diverged {
            if busy {
                self.pause();
            } else {
                self.train();
            }
            changed = true;
        }
        if plot::button(f, lay.train_buttons[1], "back to the start", size, false) {
            self.back_to_start();
            changed = true;
        }
        changed
    }

    /// The batch size, dragged along the chart's noise line.
    fn drag(&mut self, f: &mut Frame, lay: &Layout) {
        let p = self.chart_plot(lay);
        if p.tiny() {
            return;
        }
        let c = &self.look.chart;
        let x = c.get(C_B) as f64;
        let shaped = Noise { n: N, mean: Vec::new(), variances: Vec::new(), g2: 10f64.powf(c.get(C_G2) as f64), trace: 10f64.powf(c.get(C_TRACE) as f64) };
        let y = noise_line(&shaped, self.setup.draw, 10f64.powf(x)).max(1e-300).log10().max(p.y[0]).min(p.y[1]);
        if let Some((x, _)) = p.handle(f, 1, x, y, NOISE) {
            if x.is_finite() {
                self.setup.batch = (10f64.powf(x).round() as usize).clamp(1, N);
                self.dragging = true;
            }
        }
    }

    fn draw_window(&self, f: &mut Frame, lay: &Layout) {
        let p = self.window_plot(lay);
        if p.tiny() {
            return;
        }
        let s = self.moment();
        let (min, max) = p.rect;
        let size = lay.small;
        hud_panel(f, lay.window.0 - Vec2::splat(8.0), lay.window.1 + Vec2::splat(8.0));
        let step = s.step.round() as usize;
        let moment = if step == 0 { "at the start".to_string() } else { format!("after {} steps", count(step)) };
        let room = lay.window.1.x - lay.window_head.x;
        let title = [
            format!("{N} examples' gradients ({} numbers each) {moment}, seen in two directions", self.start.rows.width()),
            format!("{N} examples' gradients {moment}, in two directions"),
            format!("{N} examples' gradients {moment}"),
        ]
        .into_iter()
        .find(|t| f.text_size(t, lay.text).x <= room)
        .unwrap_or_default();
        f.text_px(lay.window_head + Vec2::new(0.0, lay.text), title, lay.text, pal::TEXT, Align2::LEFT_BOTTOM);
        // the axes through the origin, and their names
        let o = p.at(0.0, 0.0);
        plot::line_of(f, W_AXES, Vec2::new(min.x, o.y.max(min.y).min(max.y)), Vec2::new(max.x, o.y.max(min.y).min(max.y)), 1.0, pal::DIM.a(0.45));
        plot::line_of(f, W_AXES, Vec2::new(o.x.max(min.x).min(max.x), min.y), Vec2::new(o.x.max(min.x).min(max.x), max.y), 1.0, pal::DIM.a(0.45));
        let (along, across) = if lay.narrow { ("along $G$", "across $G$") } else { (r"$e_1$: along the true gradient", r"$e_2$: across it, where the examples differ most") };
        f.caption_px(Vec2::new(max.x - 4.0, o.y.max(min.y).min(max.y) - 3.0), along, size, pal::DIM, Align2::RIGHT_BOTTOM);
        f.caption_px(Vec2::new(o.x.max(min.x).min(max.x) + 5.0, min.y + 4.0), across, size, pal::DIM, Align2::LEFT_TOP);

        // the examples
        let ex = &self.look.examples.shown;
        let inside = |q: &Vec2| plot::within(*q, p.rect, 0.0);
        let pts: Vec<Vec2> = (0..ex.len() / 2).map(|i| p.at(ex[2 * i] as f64, ex[2 * i + 1] as f64)).collect();
        dots_in(f, 0, &pts, p.rect, if lay.narrow { 2.4 } else { 3.0 }, EXAMPLE);
        // their spread, and a batch's
        let sh = &self.look.shape;
        let g = sh.get(0) as f64;
        let c1 = [sh.get(1) as f64, sh.get(2) as f64, sh.get(3) as f64];
        let cb = [sh.get(4) as f64, sh.get(5) as f64, sh.get(6) as f64];
        p.path(f, W_RING, ellipse([g, 0.0], c1), 1.0, EXAMPLE.a(0.5));
        p.path(f, W_BATCH_RING, ellipse([g, 0.0], cb), 1.5, NOISE.a(0.85));
        // the batches
        let bs = &self.look.batches.shown;
        let pts: Vec<Vec2> = (0..bs.len() / 2).map(|i| p.at(bs[2 * i] as f64, bs[2 * i + 1] as f64)).collect();
        dots_in(f, 1, &pts, p.rect, if lay.narrow { 4.2 } else { 5.2 }, NOISE);
        // the true gradient
        let tip = p.at(g, 0.0);
        if inside(&o) && inside(&tip) {
            arrow_px(f, W_ARROW, o, tip, 2.4, SIGNAL);
        }
        f.text_px(Vec2::new(0.5 * (o.x + tip.x), o.y - 3.0), "$G$", lay.text, SIGNAL, Align2::CENTER_BOTTOM);

        // the key, at the foot
        let b = self.setup.batch;
        let mut key: Vec<(Color, String)> = vec![
            (SIGNAL, if lay.narrow { format!(r"$G$: the true gradient, $|G| = {}$", sig(s.g())) } else { format!(r"arrow $G$: the true gradient, the mean of all {N}; $|G| = {}$", sig(s.g())) }),
            (EXAMPLE, if lay.narrow { "an example's gradient".to_string() } else { format!("faint: one example's gradient ({N})") }),
            (
                NOISE,
                if lay.narrow {
                    format!("{K} batches of $B = {b}$")
                } else {
                    format!("bright: {K} mini-batch gradients, $B = {b}$, {}", draw_name(self.setup.draw))
                },
            ),
        ];
        if !lay.narrow {
            key.push((NOISE.a(0.85), r"ring: their spread, $\Sigma/B$ in the window (two standard deviations)".to_string()));
        }
        let share = num(100.0 * s.shown(), 0);
        key.push((
            pal::DIM,
            if lay.narrow { format!(r"the window shows {share} % of $\operatorname{{tr}}\Sigma$") } else { format!(r"the window shows {share} % of $\operatorname{{tr}}\Sigma$; the rest lies in directions it does not show") },
        ));
        let foot = lay.window.1.y;
        if lay.compact {
            key.clear();
        }
        for (k, (c, text)) in key.iter().enumerate() {
            let y = foot - 2.0 - (key.len() - 1 - k) as f32 * (size + 6.0);
            f.caption_px(Vec2::new(min.x + 2.0, y), text.clone(), size, *c, Align2::LEFT_BOTTOM);
        }
    }

    fn draw_chart(&mut self, f: &mut Frame, lay: &Layout) {
        let p = self.chart_plot(lay);
        if p.tiny() {
            return;
        }
        let m = self.moment();
        let (min, max) = p.rect;
        let size = lay.small;
        let draw = self.setup.draw;
        p.backing(f, if lay.narrow { 40.0 } else { 48.0 }, 34.0, 8.0, 8.0);
        let title = if lay.narrow { r"squared length: $|G|^2$ and the noise" } else { r"a mini-batch gradient's squared length, $\mathbb{E}\,|g_B|^2 = |G|^2 + \operatorname{tr}\Sigma/B$" };
        f.text_px(lay.chart_head, title, lay.text, pal::TEXT, Align2::LEFT_BOTTOM);
        // the axes: powers of ten
        plot::line_of(f, AXES, Vec2::new(min.x, max.y), max, 1.0, pal::DIM.a(0.8));
        plot::line_of(f, AXES, min, Vec2::new(min.x, max.y), 1.0, pal::DIM.a(0.8));
        let goal = self.chart_goal(&m);
        let xs = log_axis(f, B_TICKS_X, &p, true, true, false);
        for (x, named) in log_ticks(0.0, goal[C_XHI] as f64) {
            if named && x >= 0.0 && xs.iter().any(|v| (v - x).abs() < 1e-6) {
                f.text_px(Vec2::new(p.px_x(x), max.y + 6.0), power_name(x, false), size, pal::DIM, Align2::CENTER_TOP);
            }
        }
        f.text_px(Vec2::new(max.x, max.y + 8.0 + size), "batch size $B$", size, pal::DIM, Align2::RIGHT_TOP);
        log_axis(f, B_TICKS_Y, &p, false, false, true);
        for (y, named) in axis_ticks(goal[C_YLO] as f64, goal[C_YHI] as f64, p.height(), size) {
            if y < p.y[0] - 1e-6 || y > p.y[1] + 1e-6 || !named {
                continue;
            }
            f.text_px(Vec2::new(min.x - 6.0, p.px_y(y)), power_name(y, true), size, pal::DIM, Align2::RIGHT_CENTER);
        }

        let c = &self.look.chart;
        let (g2, trace) = (10f64.powf(c.get(C_G2) as f64), 10f64.powf(c.get(C_TRACE) as f64));
        let shaped = Noise { n: N, mean: Vec::new(), variances: Vec::new(), g2, trace };
        let across = |k: usize| p.x[1] * k as f64 / (CURVE_N - 1) as f64;
        // beyond N, where only batches drawn with replacement go
        let xn = (N as f64).log10();
        // (at N, or held at the right edge while N is beyond it: it comes in smoothly)
        let at_n = p.px_x(xn).min(max.x);
        let beyond = p.x[1] > xn + 0.01;
        plot::line_of(f, B_BEYOND, Vec2::new(at_n, min.y), Vec2::new(at_n, max.y), 1.0, if beyond { pal::DIM.a(0.6) } else { Color::CLEAR });
        if beyond {
            let name = format!("more than the {N} examples");
            if at_n + 8.0 + f.text_size(&name, size).x < max.x {
                f.text_px(Vec2::new(at_n + 4.0, min.y + 4.0), name, size, pal::DIM, Align2::LEFT_TOP);
            } else {
                f.text_px(Vec2::new(at_n - 4.0, min.y + 4.0), name, size, pal::DIM, Align2::RIGHT_TOP);
            }
        }
        // the signal
        p.path(f, B_SIGNAL, [(0.0, g2.log10()), (p.x[1], g2.log10())], 2.2, SIGNAL);
        // the noise: the way in use solid, the other dashed and faint
        for d in DRAWS {
            let pts: Vec<(f64, f64)> = (0..CURVE_N)
                .map(|k| {
                    let x = across(k);
                    let v = noise_line(&shaped, d, 10f64.powf(x));
                    if v > 0.0 && (d == Draw::WithReplacement || 10f64.powf(x) < N as f64) {
                        (x, v.log10().max(p.y[0] - 1.0))
                    } else {
                        (x, p.y[0] - 1.0)
                    }
                })
                .collect();
            if d == draw {
                p.path(f, B_NOISE, pts, 2.2, NOISE);
            } else {
                let at: Vec<Vec2> = pts.iter().map(|&(x, y)| p.at(x, y)).collect();
                for pair in plot::clip(&at, p.rect).windows(2) {
                    if pair[0].is_finite() && pair[1].is_finite() {
                        plot::dashed_of(f, B_OTHER, pair[0], pair[1], 1.3, NOISE.a(0.45));
                    }
                }
            }
        }
        // where noise and signal are equal: the noise scale
        let cross = c.get(C_CROSS) as f64;
        let mut labels: Vec<ChartLabel> = Vec::new();
        let cross_at = p.at(cross, g2.log10());
        let mut crossings = [Vec2::NAN; 2];
        let seen = plot::within(cross_at, p.rect, 0.5);
        // (held inside the chart when it is not in it, unseen, so that it comes in smoothly)
        let held = if cross_at.is_finite() { cross_at.max(min).min(max) } else { max };
        plot::dashed_of(f, B_CROSS, held, Vec2::new(held.x, max.y), 1.2, if seen { pal::TEXT.a(0.6) } else { Color::CLEAR });
        if seen {
            crossings[0] = cross_at;
            let text = format!(r"noise scale $B_{{\text{{simple}}}} = {}$", batch_text(m.noise().half_noise_batch(Draw::WithReplacement)));
            let (at, align) = beside(f, cross_at, &text, size, max.x);
            labels.push(ChartLabel { at, align, text, color: pal::TEXT });
        }
        if draw == Draw::WithoutReplacement {
            let other = p.at(c.get(C_CROSS_OTHER) as f64, g2.log10());
            if plot::within(other, p.rect, 0.5) && (other.x - cross_at.x).abs() > 6.0 {
                crossings[1] = other;
                let text = format!("without replacement: {}", batch_text(10f64.powf(goal[C_CROSS_OTHER] as f64)));
                let (at, align) = if other.x - 9.0 - f.text_size(&text, size).x > min.x { (other + Vec2::new(-5.0, 4.0), Align2::RIGHT_TOP) } else { beside(f, other, &text, size, max.x) };
                labels.push(ChartLabel { at, align, text, color: NOISE });
            }
        }
        dots_in(f, 4, &crossings, p.rect, 7.0, pal::TEXT);
        // the measured means, at powers of two
        let samples = &self.shown.samples;
        let mut means: Vec<Vec2> = Vec::new();
        let mut b = 1usize;
        while b <= N {
            if let Some((m, _)) = samples.mean_sq(draw, b) {
                if m > 0.0 {
                    let at = p.at((b as f64).log10(), m.log10());
                    if plot::within(at, p.rect, -3.0) {
                        means.push(at);
                    }
                }
            }
            b *= 2;
        }
        let ring = |at: Vec2| (0..=16).map(move |i| at + Vec2::from_angle(i as f32 / 16.0 * TAU) * 4.0);
        let rings: Vec<Vec2> = means.iter().flat_map(|a| ring(*a).chain(std::iter::once(Vec2::NAN))).collect();
        plot::path_of(f, B_MEANS, &rings, 1.3, NOISE.mix(pal::WHITE, 0.35));
        // the chosen batch size, and its batches' noise
        let xb = c.get(C_B) as f64;
        let at_b = p.px_x(xb);
        plot::line_of(f, B_CHOSEN, Vec2::new(at_b, min.y), Vec2::new(at_b, max.y), 1.0, NOISE.a(0.4));
        let noise = &self.look.noise.shown;
        let pts: Vec<Vec2> = noise.iter().map(|v| if *v == NOWHERE { Vec2::NAN } else { p.at(xb, *v as f64) }).collect();
        dots_in(f, 2, &pts, p.rect, if lay.narrow { 3.4 } else { 4.2 }, if self.look.noise_seen { NOISE } else { Color::CLEAR });
        // the names of the lines
        labels.push(ChartLabel {
            at: Vec2::new(max.x - 4.0, p.px_y(g2.log10()) - 4.0),
            align: Align2::RIGHT_BOTTOM,
            text: r"signal $|G|^2$".to_string(),
            color: SIGNAL,
        });
        let noise_name = match draw {
            Draw::WithReplacement => r"noise $\operatorname{tr}\Sigma/B$",
            Draw::WithoutReplacement => r"noise $(1/B - 1/N)\operatorname{tr}\Sigma_s$",
        };
        let x_name = (0.4 * cross.max(0.0)).min(p.x[1] * 0.25).max(0.12);
        labels.push(ChartLabel { at: p.at(x_name, noise_line(&shaped, draw, 10f64.powf(x_name)).max(1e-300).log10()) + Vec2::new(4.0, -5.0), align: Align2::LEFT_BOTTOM, text: noise_name.to_string(), color: NOISE });
        let b = self.setup.batch;
        let batch_note = if !samples.complete() {
            "measuring the batches…".to_string()
        } else if draw == Draw::WithoutReplacement && b >= N {
            format!("{K} batches of {b}: each is the whole set, no noise")
        } else {
            format!("{K} batches of {b}")
        };
        let right = at_b + 9.0 + f.text_size(&batch_note, size).x < max.x;
        let (x, align) = if right { (at_b + 5.0, Align2::LEFT_BOTTOM) } else { (at_b - 5.0, Align2::RIGHT_BOTTOM) };
        labels.push(ChartLabel { at: Vec2::new(x, max.y - 4.0), align, text: batch_note, color: NOISE });
        self.labels_moving |= place(f, labels, size, p.rect, &mut self.placed);
    }

    fn draw_history(&self, f: &mut Frame, lay: &Layout) {
        let (p, q) = self.history_plots(lay);
        if p.tiny() {
            return;
        }
        let size = lay.small;
        let (min, max) = p.rect;
        hud_panel(f, min - Vec2::new(if lay.narrow { 40.0 } else { 48.0 }, 8.0), Vec2::new(max.x + 8.0, q.rect.1.y + 30.0));
        let stretch = self.run.stretches.last().copied().unwrap_or(Stretch { from: 0, lr: self.setup.lr, batch: self.setup.train_batch });
        let title = if lay.narrow {
            format!(r"noise scale along a run of SGD, $\eta = {}$, $B = {}$", short(stretch.lr), stretch.batch)
        } else {
            format!(r"the noise scale along a run of SGD, $\eta = {}$, batches of {} (an epoch at a time)", short(stretch.lr), stretch.batch)
        };
        f.text_px(lay.history_head, title, lay.text, pal::TEXT, Align2::LEFT_BOTTOM);
        // axes
        for pl in [&p, &q] {
            let (a, b) = pl.rect;
            plot::line_of(f, AXES, Vec2::new(a.x, b.y), b, 1.0, pal::DIM.a(0.8));
            plot::line_of(f, AXES, a, Vec2::new(a.x, b.y), 1.0, pal::DIM.a(0.8));
        }
        let goal = self.history_goal();
        let ticks = plot::Ticks::of(0.0, goal[0] as f64, q.width());
        for v in ticks.values() {
            if v > q.x[1] + 1e-6 {
                continue;
            }
            let x = q.px_x(v);
            plot::line_of(f, AXES, Vec2::new(x, q.rect.1.y), Vec2::new(x, q.rect.1.y + 4.0), 1.0, pal::DIM.a(0.8));
            f.text_px(Vec2::new(x, q.rect.1.y + 6.0), count(v.round() as usize), size, pal::DIM, Align2::CENTER_TOP);
        }
        f.text_px(Vec2::new(q.rect.1.x, q.rect.1.y + 8.0 + size), "step", size, pal::DIM, Align2::RIGHT_TOP);
        for (pl, lo, hi) in [(&p, goal[1], goal[2]), (&q, goal[3], goal[4])] {
            log_axis(f, H_TICKS_Y, pl, false, false, true);
            for (y, named) in axis_ticks(lo as f64, hi as f64, pl.height(), size) {
                if y < pl.y[0] - 1e-6 || y > pl.y[1] + 1e-6 || !named {
                    continue;
                }
                f.text_px(Vec2::new(pl.rect.0.x - 6.0, pl.px_y(y)), power_name(y, true), size, pal::DIM, Align2::RIGHT_CENTER);
            }
        }
        // the marks the film has reached: the noise scale at each, and the smoothed line;
        // |G|² and tr Σ. (Each line gains a point as the film reaches a mark, 40 a second:
        // nothing on them moves, they grow.)
        let marks = &self.run.marks;
        let film = self.film;
        let reached = marks.partition_point(|m| m.step as f64 <= film + 1e-9).max(1).min(marks.len());
        let at = |pl: &Plot, step: u64, v: f64| if v > 0.0 && v.is_finite() { pl.at(step as f64, v.log10()) } else { Vec2::NAN };
        let shown = &marks[..reached];
        let dots: Vec<Vec2> = shown.iter().map(|m| at(&p, m.step, m.noise_scale())).collect();
        dots_in(f, 3, &dots, p.rect, if lay.narrow { 2.6 } else { 3.2 }, SGD.a(0.55));
        let smooth: Vec<Vec2> = (0..reached).map(|i| at(&p, marks[i].step, smoothed(marks, i))).collect();
        plot::path_of(f, H_SMOOTH, &plot::clip(&smooth, p.rect), 2.0, SGD);
        let g2: Vec<Vec2> = shown.iter().map(|m| at(&q, m.step, m.g2)).collect();
        let tr: Vec<Vec2> = shown.iter().map(|m| at(&q, m.step, m.trace)).collect();
        plot::path_of(f, H_G2, &plot::clip(&g2, q.rect), 1.6, SIGNAL);
        plot::path_of(f, H_TRACE, &plot::clip(&tr, q.rect), 1.6, NOISE);
        // before any training, what the chart is for
        if marks.len() == 1 && !self.run.running && p.height() > 60.0 {
            let hint = if lay.narrow { "“train” runs SGD and measures as it goes" } else { "“train” runs SGD from this start, measuring every 10 steps" };
            f.text_px(Vec2::new(0.5 * (min.x + max.x), min.y + 0.65 * p.height()), hint, size, pal::DIM, Align2::CENTER_CENTER);
        }
        // where the settings changed
        let mut changes = false;
        for st in self.run.stretches.iter().skip(1) {
            let x = p.px_x(st.from as f64);
            plot::dashed_of(f, H_STRETCH, Vec2::new(x, min.y), Vec2::new(x, q.rect.1.y), 1.0, pal::DIM.a(0.6));
            changes = true;
        }
        if !changes {
            plot::dashed_of(f, H_STRETCH, max, max, 1.0, Color::CLEAR);
        }
        // the moment the window shows
        let x = p.px_x(film);
        plot::line_of(f, H_NOW, Vec2::new(x, min.y), Vec2::new(x, q.rect.1.y), 1.0, pal::TEXT.a(0.35));
        // names
        let last = marks.get(reached - 1);
        let now = smoothed(marks, reached - 1);
        let name = match last {
            Some(m) if reached > 1 => format!(r"$B_{{\text{{simple}}}}$: {}, smoothed {}", batch_text(m.noise_scale()), batch_text(now)),
            Some(m) => format!(r"$B_{{\text{{simple}}}} = {}$ at the start: train to see it change", batch_text(m.noise_scale())),
            None => String::new(),
        };
        f.caption_px(Vec2::new(min.x + 6.0, min.y + 4.0), name, size, SGD, Align2::LEFT_TOP);
        if self.run.diverged {
            f.caption_px(Vec2::new(max.x - 6.0, min.y + 4.0), format!(r"the weights flew off at $\eta = {}$", short(stretch.lr)), size, LIMIT, Align2::RIGHT_TOP);
        }
        // the strip's lines named in their colours, and the loss
        let (a, b) = q.rect;
        f.caption_px(Vec2::new(a.x + 4.0, a.y + 2.0), r"$\textcolor{white}{|G|^2}$ and $\textcolor{red}{\operatorname{tr}\Sigma}$", size, pal::DIM, Align2::LEFT_TOP);
        if let Some(m) = last {
            f.caption_px(Vec2::new(b.x - 4.0, a.y + 2.0), format!("loss {}", sig(m.loss)), size, pal::DIM, Align2::RIGHT_TOP);
        }
    }
}

/// Where a label goes beside a point on a chart: to its right and below if there is
/// room before `right`, else to its left.
fn beside(f: &Frame, at: Vec2, text: &str, size: f32, right: f32) -> (Vec2, Align2) {
    if at.x + 9.0 + f.text_size(text, size).x < right {
        (at + Vec2::new(5.0, 4.0), Align2::LEFT_TOP)
    } else {
        (at + Vec2::new(-5.0, 4.0), Align2::RIGHT_TOP)
    }
}

/// A label of the chart: where it wants to be (the point its alignment anchors),
/// what it says, its colour.
struct ChartLabel {
    at: Vec2,
    align: Align2,
    text: String,
    color: Color,
}

/// Where a label of the chart was put: its words, the slot it took (above or below
/// its own place, in its heights) and how far from its place it is drawn now (gliding
/// to that slot when it changes).
#[derive(Clone, Debug, Default)]
struct Placed {
    text: String,
    slot: i32,
    drawn: f32,
}

/// The labels on plates, each at its own place if that is clear of those placed before
/// it, else the nearest clear place above or below it, inside the rectangle (in the
/// order given: the first keep their places). A label keeps the slot it had while that
/// is still clear, and glides to a new one (`memo` remembers them; true while one
/// glides). (The same as the noise floor's.)
fn place(f: &mut Frame, tags: Vec<ChartLabel>, size: f32, (min, max): Rect, memo: &mut Vec<Placed>) -> bool {
    let mut taken: Vec<Rect> = Vec::new();
    let mut now: Vec<Placed> = Vec::with_capacity(tags.len());
    let ease = 1.0 - (-f.real_dt.max(0.0) / 0.08).exp();
    let mut gliding = false;
    for t in tags {
        let w = f.text_size(&t.text, size) + Vec2::new(8.0, 4.0);
        let corner = |at: Vec2| {
            let x = match t.align.x() {
                egui::Align::Min => at.x,
                egui::Align::Center => at.x - 0.5 * w.x,
                egui::Align::Max => at.x - w.x,
            };
            let y = match t.align.y() {
                egui::Align::Min => at.y,
                egui::Align::Center => at.y - 0.5 * w.y,
                egui::Align::Max => at.y - w.y,
            };
            Vec2::new(x, y)
        };
        let fits = |c: Vec2| c.x >= min.x - 1.0 && c.y >= min.y - 1.0 && c.x + w.x <= max.x + 1.0 && c.y + w.y <= max.y + 1.0;
        let clear = |c: Vec2, taken: &[Rect]| taken.iter().all(|(a, b)| c.x + w.x <= a.x || b.x <= c.x || c.y + w.y <= a.y || b.y <= c.y);
        let step = w.y + 1.0;
        let before = memo.iter().find(|m| m.text == t.text).cloned();
        let slots = before.as_ref().map(|b| b.slot).into_iter().chain((0..13).map(|k| if k % 2 == 1 { -(k + 1) / 2 } else { k / 2 }));
        let found = slots.into_iter().find(|&k| {
            let c = corner(t.at + Vec2::new(0.0, k as f32 * step));
            fits(c) && clear(c, &taken)
        });
        let Some(slot) = found else { continue };
        let c = corner(t.at + Vec2::new(0.0, slot as f32 * step));
        taken.push((c, c + w));
        let goal = slot as f32 * step;
        let drawn = match before {
            Some(b) if (b.drawn - goal).abs() > 0.5 => {
                gliding = true;
                b.drawn + (goal - b.drawn) * ease
            }
            _ => goal,
        };
        f.caption_px(t.at + Vec2::new(0.0, drawn), t.text.clone(), size, t.color, t.align);
        now.push(Placed { text: t.text, slot, drawn });
    }
    *memo = now;
    gliding
}

/// One example's gradient: faint.
const EXAMPLE: Color = pal::DIM;

impl Scene for MiniBatch {
    fn view(&self) -> View {
        View::Flat { center: Vec2::ZERO, half_height: 1.0 }
    }

    fn settings(&self, s: &mut ViewSettings) {
        s.grid = false;
        s.bloom = s.bloom.min(0.2);
    }

    fn animated(&self) -> bool {
        self.busy() || self.look.moving || self.dragging || self.labels_moving
    }

    fn pointer_hint(&self) -> Option<&'static str> {
        Some("red dot on the chart: drag the batch size    buttons: click")
    }

    fn ui(&mut self, ui: &mut egui::Ui, _: &Gfx) {
        ui::note(
            ui,
            r"The loss is the mean of one loss per example, so its gradient, the true gradient $G$, is the mean of one gradient per example (scene backpropmat). Here there are 512 examples, points in the plane in three overlapping classes, and a network of 371 weights, so each example's gradient $g_n$ is a list of 371 numbers. A mini-batch of $B$ examples averages only $B$ of them: its gradient $g_B$ is right on average and off by noise. For $B$ examples drawn independently,",
        );
        ui::equation(ui, r"\mathbb{E}\,|g_B|^2 = |G|^2 + \frac{\operatorname{tr}\Sigma}{B}");
        ui::note(
            ui,
            r"where $\operatorname{tr}\Sigma$ is the total variance of one example's gradient: the mean squared distance of a $g_n$ from $G$. The two parts are equal at the noise scale $B_{\text{simple}} = \operatorname{tr}\Sigma/|G|^2$ (McCandlish and co-authors, 2018): well below it a mini-batch gradient is mostly noise, well above it, nearly $G$. What follows from that, how large a batch is worth its cost and what floor a constant step cannot get under, is for the scenes noisescale and noisefloor.",
        );

        ui::section(ui, "Things to try");
        for (k, t) in TRIALS.iter().enumerate() {
            if ui::experiment(ui, t.button, t.caption) {
                self.set_up(k);
            }
        }

        ui::section(ui, "The mini-batch");
        let s = &mut self.setup;
        ui::slider_int_log(ui, "batch size $B$", &mut s.batch, 1..=N);
        let mut d = draw_index(s.draw);
        if ui::choice(ui, &["with replacement", "without replacement"], &mut d) {
            s.draw = DRAWS[d.min(1)];
        }
        ui::note(
            ui,
            r"With replacement each of the $B$ examples is drawn from all 512 independently, so one can come twice, and the noise is exactly $\operatorname{tr}\Sigma/B$. Without, a batch is $B$ different examples, as when an epoch deals the data out in batches; then the noise is $(1/B - 1/N)\operatorname{tr}\Sigma_s$, smaller, and 0 for the whole set. ($\Sigma_s$ divides by $N - 1$ where $\Sigma$ divides by $N$.)",
        );

        ui::section(ui, "Training");
        let mut lr = s.lr as f32;
        if ui::slider_log(ui, r"step size $\eta$", &mut lr, LEAST_LR as f32..=MOST_LR as f32) {
            s.lr = lr as f64;
        }
        ui::slider_int_log(ui, "its batch size", &mut s.train_batch, 1..=N);
        let mut steps = s.steps as usize;
        if ui::slider_int_log(ui, "steps for each press of train", &mut steps, EVERY as usize..=MOST_STEPS as usize) {
            s.steps = steps as u64;
        }
        ui::note(
            ui,
            NOTE_TRAINING,
        );
        ui.horizontal_wrapped(|ui| {
            let label = if self.run.running { "pause" } else { "train" };
            if ui.button(ui::label_text(ui, label, 12.5, None, false)).clicked() {
                if self.run.running {
                    self.pause();
                } else {
                    self.train();
                }
            }
            if ui.button(ui::label_text(ui, "back to the start", 12.5, None, false)).clicked() {
                self.back_to_start();
            }
        });
        self.setup = self.setup.kept();

        ui::section(ui, "Reading the picture");
        ui::key(
            ui,
            &[
                (SIGNAL, r"white: the true gradient $G$ (the arrow) and its squared length $|G|^2$, the signal"),
                (EXAMPLE, "grey dots: one example's gradient each"),
                (NOISE, r"red: mini-batch gradients (dots), their spread (ring), the noise $\operatorname{tr}\Sigma/B$ (line) and each batch's $|g_B - G|^2$ (dots on the chart)"),
                (NOISE.mix(pal::WHITE, 0.35), "rings on the chart: the mean of the forty batches' noise, measured"),
                (SGD, "cyan: the noise scale measured along the run of SGD (dots), and smoothed over 100 steps (line)"),
            ],
        );
        ui::section(ui, "Words");
        ui::term(ui, r"Per-example gradient $g_n$:", "the gradient of one example's own loss with respect to all 371 weights.");
        ui::term(ui, r"True gradient $G$:", r"the gradient of the loss on all 512 examples, the mean of the $g_n$.");
        ui::term(ui, r"Mini-batch, $B$:", r"the examples a step averages; its gradient $g_B$ is their mean.");
        ui::term(ui, r"Noise $\xi$:", r"what a mini-batch gradient adds to the true one, $\xi = g_B - G$; on average 0.");
        ui::term(ui, r"Covariance $\Sigma$, trace:", r"$\Sigma = \frac1N \sum_n (g_n - G)(g_n - G)^\top$, a 371 by 371 matrix; its trace, the sum of its diagonal, is the mean of $|g_n - G|^2$.");
        ui::term(ui, r"Noise scale $B_{\text{simple}}$:", r"$\operatorname{tr}\Sigma/|G|^2$, a number of examples: the batch size at which noise and signal are equal in squared length. McCandlish and co-authors call it the simple noise scale and derive it from a fuller one by assuming the Hessian is a multiple of the identity, which they call unrealistic (scene noisescale). Not the “noise scale” of Smith and Le (2018), $\epsilon(N/B - 1)$ with $\epsilon$ the step size, which is measured in the units of a step size and not in examples.");
        ui::term(ui, "Cosine:", r"of two gradients, $a \cdot b/(|a|\,|b|)$: 1 when they point the same way, 0 at right angles.");
        ui::term(ui, "Window:", "a plane through 0 in the space of all 371 weights, spanned by two unit directions; a gradient is drawn at its two components along them. The numbers and the chart use all 371.");
        ui::term(ui, "Epoch:", "one pass through the data, here in batches of different examples.");
        ui::term(ui, r"SGD, step size $\eta$:", "gradient descent with a mini-batch's gradient; the number the gradient is multiplied by in a step.");

        ui::section(ui, "Numbers");
        self.numbers(ui);
        ui::see_also(
            ui,
            &[
                ("backpropmat", "a batch's gradient, one example at a time"),
                ("clt", "why the spread of a mean falls as one over its root"),
                ("noisefloor", "what the noise does to a constant step"),
                ("noisescale", "how large a batch is worth it"),
            ],
        );
    }

    fn frame(&mut self, f: &mut Frame) {
        hold_batches(f);
        let lay = Layout::of(f.view_size(), f.top_inset);
        // (a canvas squeezed to a sliver is one the controls have opened over, on a phone)
        let tiny = lay.view.x < 160.0 || lay.view.y < 120.0;
        self.dragging = false;
        if !tiny && !lay.compact {
            self.buttons(f, &lay);
            self.drag(f, &lay);
        }
        self.setup = self.setup.kept();
        let budget = clock::frame_budget(f.real_dt, 0.004);
        self.work(f.dt, &budget);
        self.roll(f.dt);
        let snap = self.canvas != Some(lay.view) && !tiny;
        if snap {
            self.canvas = Some(lay.view);
        }
        self.follow(f.real_dt, snap);
        if tiny {
            return;
        }
        self.labels_moving = false;
        self.draw_window(f, &lay);
        if !lay.compact {
            self.draw_chart(f, &lay);
            self.draw_history(f, &lay);
        }
    }
}

impl MiniBatch {
    /// The read-outs.
    fn numbers(&self, ui: &mut egui::Ui) {
        let s = &self.shown.snap;
        let n = &s.noise;
        let (b, draw) = (self.setup.batch, self.setup.draw);
        ui::readout(ui, "examples $N$, weights", format!("{N}, {}", s.e[0].len()));
        ui::readout(ui, "the moment shown", if s.step == 0 { "the start".to_string() } else { format!("after {} steps", count(s.step as usize)) });
        ui::readout(ui, "loss on all the data", sig(s.loss));
        ui::readout_colored(ui, r"signal $|G|^2$", sig(n.g2), SIGNAL);
        ui::readout_colored(ui, r"$\operatorname{tr}\Sigma$", sig(n.trace), NOISE);
        ui::readout(ui, r"noise scale $B_{\text{simple}}$", sig(n.simple_noise_scale()));
        ui::readout(ui, "  noise equals signal, without replacement, at", sig(s.crossing(Draw::WithoutReplacement)));
        ui::readout(ui, r"an example's length over $|G|$: median", format!("{} times", sig(s.median_length())));
        ui::readout(ui, "the window shows", format!("{} % of the variance", num(100.0 * s.shown(), 0)));
        ui::readout(ui, &format!("at $B = {b}$, {}:", draw_name(draw)), "");
        ui::readout_colored(ui, r"  noise, exactly", sig(s.noise_at(draw, b as f64)), NOISE);
        // (what is a rounding error of |G|² is written 0)
        let small = |v: f64| if v.abs() <= 1e-12 * n.g2 { "0".to_string() } else { sig(v) };
        if let Some((m, se)) = self.shown.samples.mean_sq(draw, b) {
            ui::readout(ui, &format!("  noise of the {K} batches"), format!(r"${} \pm {}$", small(m), small(2.0 * se)));
        }
        ui::readout(ui, r"  cosine with $G$, the formula", sig(n.expected_cosine(b, draw)));
        if let Some((m, se)) = self.shown.samples.mean_cosine(n.g2, draw, b) {
            let se = if se <= 1e-12 { "0".to_string() } else { sig(2.0 * se) };
            ui::readout(ui, &format!("  cosine of the {K} batches"), format!(r"${} \pm {se}$", sig(m)));
        }
        if b == 1 {
            ui::readout(ui, "  mean cosine over all 512 examples", sig(s.cos_one));
        }
        if self.run.marks.len() > 1 {
            let last = self.run.marks[self.run.marks.len() - 1];
            ui::readout(ui, "training: steps taken", count(self.run.steps as usize));
            ui::readout_colored(ui, "  noise scale now, smoothed", sig(self.run.smoothed_now()), SGD);
            ui::readout(ui, "  loss now", sig(last.loss));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn close(a: f64, b: f64, rel: f64) -> bool {
        (a - b).abs() <= rel * b.abs().max(1e-300)
    }

    /// An experiment as its button sets it up, with everything it needs measured.
    fn run(k: usize) -> MiniBatch {
        let mut s = MiniBatch::create();
        s.set_up(k);
        s.finish();
        s
    }

    /// The noise scale of the marks from step `from` on: the ratio of the averages of
    /// tr Σ and |G|² over them.
    fn settled(marks: &[Mark], from: u64) -> f64 {
        let w: Vec<&Mark> = marks.iter().filter(|m| m.step >= from).collect();
        w.iter().map(|m| m.trace).sum::<f64>() / w.iter().map(|m| m.g2).sum::<f64>()
    }

    /// The exact noise of the chart's lines is the library's, and the forty batches'
    /// measured mean lies within four standard errors of it, at every size tried, both
    /// ways of drawing; without replacement at N it is 0.
    #[test]
    fn the_batches_follow_the_formula() {
        let s = run(0);
        let sn = &s.shown.snap;
        let n = &sn.noise;
        let mut worst = 0.0f64;
        for d in DRAWS {
            for b in [1usize, 2, 3, 4, 6, 9, 16, 32, 59, 128, 300, 511] {
                let exact = sn.noise_at(d, b as f64);
                assert!(close(exact, n.expected_sq_norm(b, d) - n.g2, 1e-9), "{d:?} {b}");
                let (m, se) = s.shown.samples.mean_sq(d, b).unwrap();
                let z = (m - exact).abs() / se;
                worst = worst.max(z);
                assert!(z < 4.0, "{d:?} B = {b}: {m} ± {se} against {exact}");
            }
        }
        assert!(worst > 0.5, "suspiciously close: {worst}");
        // the whole set without replacement: no noise at all, every batch is G
        assert_eq!(sn.noise_at(Draw::WithoutReplacement, N as f64), 0.0);
        for k in 0..K {
            let at = sn.batch_at(&s.lists, Draw::WithoutReplacement, k, N);
            assert!((at[0] - sn.g()).abs() < 1e-12 && at[1].abs() < 1e-12, "{at:?}");
            assert!(s.shown.samples.get(Draw::WithoutReplacement, k, N).unwrap().sq < 1e-25 * n.g2);
        }
        // (and the two ways' factors: 1/B, and (N − B)/(B (N − 1)))
        assert!(close(terrain::noise_factor(59, N, Draw::WithoutReplacement) * 59.0, (N - 59) as f64 / (N - 1) as f64, 1e-12));
    }

    /// The window's dots and the chart's are the same batches: a batch's window position
    /// is its mean gradient's two components, and its noise is |g_B − G|², both as
    /// computed directly from the rows.
    #[test]
    fn the_running_sums_are_the_batches() {
        let s = run(0);
        let (sn, rows) = (&s.shown.snap, &s.shown.rows);
        let lists = Lists::new(BATCH_SEED);
        for d in DRAWS {
            for (k, b) in [(0usize, 1usize), (3, 7), (17, 59), (39, 512)] {
                let mut mean = Vec::new();
                rows.batch_mean(lists.batch(d, k, b), 0..rows.width(), &mut mean);
                let at = sn.batch_at(&lists, d, k, b);
                assert!((dot(&mean, &sn.e[0]) - at[0]).abs() < 1e-12 && (dot(&mean, &sn.e[1]) - at[1]).abs() < 1e-12);
                let xi: Vec<f64> = mean.iter().zip(&sn.noise.mean).map(|(a, g)| a - g).collect();
                let got = s.shown.samples.get(d, k, b).unwrap();
                assert!(close(got.sq, dot(&xi, &xi), 1e-9) || dot(&xi, &xi) < 1e-25);
                assert!((got.along - dot(&xi, &sn.noise.mean)).abs() < 1e-9 * sn.noise.g2.max(1.0));
            }
        }
        // with replacement a batch can hold an example twice; without, never
        let with = lists.batch(Draw::WithReplacement, 0, N);
        let without = lists.batch(Draw::WithoutReplacement, 0, N);
        let distinct = |v: &[usize]| {
            let mut w = v.to_vec();
            w.sort();
            w.dedup();
            w.len()
        };
        assert!(distinct(with) < N && distinct(without) == N);
    }

    /// The window: e₁ = G/|G|, e₂ of length 1 across it, and e₂ the direction across G
    /// of the largest spread (against 400 steps of the power method on P Σ P, and
    /// against random directions across G); its covariance and share as computed from
    /// the rows. The Lanczos estimate converges in a handful of products.
    #[test]
    fn the_window_is_the_widest_view_across_g() {
        let problem = data();
        for steps in [0u64, 300] {
            let mut s = MiniBatch::create();
            if steps > 0 {
                s.setup.steps = steps;
                s.train();
                s.finish();
            }
            let (sn, rows) = (&s.shown.snap, &s.shown.rows);
            let (e1, e2) = (&sn.e[0], &sn.e[1]);
            let g = sn.g();
            assert!(e1.iter().zip(&sn.noise.mean).all(|(e, m)| (e - m / g).abs() < 1e-12));
            assert!((norm(e2) - 1.0).abs() < 1e-12 && dot(e1, e2).abs() < 1e-12);
            let n = N as f64;
            let p_sigma_p = |v: &[f64]| -> Vec<f64> {
                let mut w = v.to_vec();
                axpy(-dot(e1, &w), e1, &mut w);
                let mut out = vec![0.0; w.len()];
                for k in 0..N {
                    let d: Vec<f64> = rows.row(k).iter().zip(&sn.noise.mean).map(|(a, m)| a - m).collect();
                    axpy(dot(&d, &w) / n, &d, &mut out);
                }
                axpy(-dot(e1, &out), e1, &mut out);
                out
            };
            let var = |v: &[f64]| dot(v, &p_sigma_p(v));
            // the power method, long
            let mut v: Vec<f64> = (0..e1.len()).map(|i| ((i * 7919) % 13) as f64 - 6.0).collect();
            for _ in 0..400 {
                v = p_sigma_p(&v);
                let l = norm(&v);
                v.iter_mut().for_each(|x| *x /= l);
            }
            let top = var(&v);
            assert!(close(var(e2), top, 1e-4), "after {steps}: {} against {top}", var(e2));
            let mut rng = Rng::new(3);
            for _ in 0..50 {
                let mut r: Vec<f64> = (0..e1.len()).map(|_| rng.normal() as f64).collect();
                axpy(-dot(e1, &r), e1, &mut r);
                let l = norm(&r);
                r.iter_mut().for_each(|x| *x /= l);
                assert!(var(&r) < var(e2));
            }
            // the window's covariance and its share of tr Σ
            assert!(close(sn.cov[2], var(e2), 1e-9));
            let along: f64 = (0..N).map(|k| (dot(rows.row(k), e1) - g).powi(2)).sum::<f64>() / n;
            assert!(close(sn.cov[0], along, 1e-9));
            assert!(close(sn.shown(), (along + var(e2)) / sn.noise.trace, 1e-9));
            // (the Lanczos products it took)
            let mut m = Measure::new(&problem, &s.run.model, 0, true, None);
            let mut tape = Tape::new();
            let mut done = None;
            while done.is_none() {
                done = m.work(&problem, &s.run.model, &mut tape, &Budget::new(f64::INFINITY));
            }
            let products = m.window.as_ref().unwrap().products;
            assert!(products <= 16, "{products} products");
        }
    }

    /// The library's approximate expected cosine against 4 000 sampled batches at the
    /// start, for the sizes the experiments use: within 0.045 (measured: at most 0.03).
    #[test]
    fn the_cosine_formula_against_sampling() {
        let s = run(0);
        let (sn, rows) = (&s.shown.snap, &s.shown.rows);
        let mut worst = 0.0f64;
        for d in DRAWS {
            for b in [1usize, 2, 6, 16, 59] {
                let sampled = rows.sample(b, d, 4000, Rng::new(b as u64 + 11));
                let formula = sn.noise.expected_cosine(b, d);
                println!("{d:?} B = {b}: sampled {:.4} ± {:.4}, formula {formula:.4}", sampled.cosine, sampled.cosine_error);
                worst = worst.max((sampled.cosine - formula).abs());
            }
        }
        println!("worst {worst:.4}");
        assert!(worst < 0.045, "{worst}");
        // at B = 1 the sampled mean and the exact mean over all examples agree
        let one = rows.sample(1, Draw::WithReplacement, 20_000, Rng::new(5));
        assert!((one.cosine - sn.cos_one).abs() < 4.0 * one.cosine_error);
    }

    /// The numbers each experiment's caption says, from the scene as the button sets it up.
    #[test]
    fn the_experiments_show_what_their_captions_say() {
        let says = |k: usize, what: &str| assert!(TRIALS[k].caption.contains(what), "experiment {k} should say \"{what}\": {}", TRIALS[k].caption);
        // one example at a time
        let s0 = run(0);
        let sn = &s0.shown.snap;
        let n = &sn.noise;
        assert_eq!((s0.setup.batch, s0.setup.draw), (1, Draw::WithReplacement));
        let bs = n.simple_noise_scale();
        says(0, &format!("the median) is {} times as long", num(sn.median_length(), 2)));
        says(0, &format!("the mean of its cosine with $G$ is {}", num(sn.cos_one, 3)));
        says(0, &format!("on average {} times the signal", num(n.trace / n.g2, 2)));
        says(0, &format!("gives {} for the cosine", num(n.expected_cosine(1, Draw::WithReplacement), 3)));
        assert!(n.expected_cosine(1, Draw::WithReplacement) > sn.cos_one, "a little high");
        assert_eq!(N, 512);
        // the noise scale
        let s1 = run(1);
        assert_eq!(s1.setup.batch, bs.round() as usize);
        let tr6 = n.trace / 6.0;
        says(1, &format!(r"B_{{\text{{simple}}}} = {}$", num(bs, 2)));
        says(1, &format!(r"$\operatorname{{tr}}\Sigma/B = {}$", num(tr6, 3)));
        says(1, &format!("$|G|^2 = {}$", num(n.g2, 3)));
        assert!((tr6 - n.g2).abs() < 0.02 * n.g2, "within 2 %");
        // (the ring of the batches is √6 times narrower than the examples', exactly)
        let c = s1.shown.snap.batch_cov(Draw::WithReplacement, 6);
        assert!(close(c[0] * 6.0, sn.cov[0], 1e-12) && close(c[2] * 6.0, sn.cov[2], 1e-12));
        says(1, r"$\sqrt 6 \approx 2.45$");
        let (cos6, se6) = s1.shown.samples.mean_cosine(n.g2, Draw::WithReplacement, 6).unwrap();
        says(1, &format!(r"mean cosine with $G$ is ${} \pm {}$ (two standard errors); the formula gives {}", num(cos6, 2), num(2.0 * se6, 2), num(n.expected_cosine(6, Draw::WithReplacement), 2)));
        // ten times
        let s2 = run(2);
        assert_eq!(s2.setup.batch, (10.0 * bs).round() as usize);
        let tr59 = n.trace / 59.0;
        says(2, &format!("tenth of the signal ({} against {})", num(tr59, 4), num(n.g2, 3)));
        assert!((tr59 / n.g2 - 0.1).abs() < 0.002);
        let (cos59, se59) = s2.shown.samples.mean_cosine(n.g2, Draw::WithReplacement, 59).unwrap();
        says(2, &format!(r"mean cosine with $G$ is ${} \pm {}$, the formula's {}", num(cos59, 3), num(2.0 * se59, 3), num(n.expected_cosine(59, Draw::WithReplacement), 3)));
        says(2, &format!("$(N - B)/(N - 1) = {}$", num((N - 59) as f64 / (N - 1) as f64, 2)));
        // the whole set
        let s3 = run(3);
        assert_eq!((s3.setup.batch, s3.setup.draw), (N, Draw::WithoutReplacement));
        for k in 0..K {
            let at = s3.shown.snap.batch_at(&s3.lists, Draw::WithoutReplacement, k, N);
            assert!((at[0] - sn.g()).abs() < 1e-12 && at[1].abs() < 1e-12);
        }
        says(3, &format!(r"$\operatorname{{tr}}\Sigma/512 = {}$, {} % of the signal", num(n.trace / 512.0, 4), num(100.0 * n.trace / 512.0 / n.g2, 1)));
        // training
        let s4 = run(4);
        let marks = &s4.run.marks;
        assert_eq!(marks.len(), 101);
        assert!(marks.iter().enumerate().all(|(i, m)| m.step == 10 * i as u64));
        assert_eq!(s4.shown.snap.step, 1000);
        let first = marks[0];
        let at = |t: u64| marks.iter().find(|m| m.step == t).unwrap();
        says(4, &format!("falls from {} to {} in the first 100 steps, then only to {}", num(first.loss, 2), num(at(100).loss, 2), num(at(1000).loss, 2)));
        let late: Vec<&Mark> = marks.iter().filter(|m| m.step >= 200).collect();
        let mean_g2 = late.iter().map(|m| m.g2).sum::<f64>() / late.len() as f64;
        let mean_tr = late.iter().map(|m| m.trace).sum::<f64>() / late.len() as f64;
        says(4, &format!("on average {} times smaller", num(first.g2 / mean_g2, 0)));
        assert!((mean_tr / first.trace - 2.0 / 3.0).abs() < 0.03, "two thirds: {}", mean_tr / first.trace);
        let b200 = settled(marks, 200);
        says(4, &format!("is {}, {} times the start's {}", num(b200, 0), num(b200 / first.noise_scale(), 0), num(first.noise_scale(), 2)));
        let (lo, hi) = late.iter().map(|m| m.noise_scale()).fold((f64::INFINITY, 0.0f64), |(a, b), v| (a.min(v), b.max(v)));
        says(4, &format!("between {} and {}", num(lo, 0), count((hi / 100.0).round() as usize * 100)));
        assert!((lo - lo.round()).abs() < 0.5 && (hi / 100.0).round() * 100.0 - hi < 60.0, "{lo} {hi}");
        // a batch of 16 is mostly noise: its noise is many times the signal
        assert!(s4.shown.snap.noise.simple_noise_scale() / 16.0 > 10.0);
        // and the smoothed line ends where the panel says it is
        assert!(close(s4.run.smoothed_now(), smoothed(marks, marks.len() - 1), 1e-15));
    }

    /// The training note's two numbers: the noise scale over steps 200 to 1000 at the two
    /// step sizes, all else as the experiment sets it.
    #[test]
    fn a_smaller_step_settles_at_a_larger_noise_scale() {
        let says = |what: &str| assert!(NOTE_TRAINING.contains(what), "the note should say \"{what}\"");
        let mut fast = run(4);
        let at_03 = settled(&fast.run.marks, 200);
        fast.set_up(4);
        fast.setup.lr = 0.1;
        fast.back_to_start();
        fast.train();
        fast.finish();
        let at_01 = settled(&fast.run.marks, 200);
        says(&format!("it is {} at $\\eta = 0.1$ against {} at $\\eta = 0.3$", num(at_01, 0), num(at_03, 0)));
        assert!(at_01 > 3.0 * at_03);
    }

    /// The run is the same every time, measured every ten steps; pressing train again
    /// goes on from where it is; back to the start forgets it.
    #[test]
    fn training_is_repeatable_and_goes_on() {
        let mut a = MiniBatch::create();
        a.setup.steps = 120;
        a.train();
        a.finish();
        let mut b = MiniBatch::create();
        b.setup.steps = 60;
        b.train();
        b.finish();
        b.train();
        b.finish();
        assert_eq!(a.run.steps, 120);
        assert_eq!(a.run.marks.len(), 13);
        assert_eq!(a.run.marks, b.run.marks);
        assert_eq!(a.shown.snap.step, 120);
        assert_eq!(a.run.stretches.len(), 1, "one setting all along");
        // a new setting is a new stretch
        b.setup.lr = 0.05;
        b.train();
        assert_eq!(b.run.stretches.len(), 2);
        b.back_to_start();
        assert_eq!((b.run.steps, b.run.marks.len(), b.shown.snap.step), (0, 1, 0));
        assert!(!b.run.running);
    }

    /// The scene opens as the first experiment sets it up.
    #[test]
    fn the_scene_opens_on_the_first_experiment() {
        let opened = MiniBatch::create();
        let mut pressed = MiniBatch::create();
        pressed.set_up(3);
        pressed.set_up(OPENING);
        assert_eq!(opened.setup, pressed.setup);
        assert_eq!((opened.run.steps, opened.shown.snap.step), (pressed.run.steps, pressed.shown.snap.step));
    }

    /// Edges: settings at and beyond their ends and not numbers, weights that are not
    /// numbers (training that blew up): every number drawn stays finite, and the scene
    /// says what happened.
    #[test]
    fn degenerate_settings_are_told_honestly() {
        let cases = [
            Setup { batch: 0, draw: Draw::WithReplacement, lr: f64::NAN, train_batch: 0, steps: 0 },
            Setup { batch: N, draw: Draw::WithReplacement, lr: MOST_LR, train_batch: N, steps: MOST_STEPS },
            Setup { batch: 10 * N, draw: Draw::WithoutReplacement, lr: -1.0, train_batch: 10 * N, steps: u64::MAX },
            Setup { batch: 1, draw: Draw::WithoutReplacement, lr: f64::INFINITY, train_batch: 1, steps: 10 },
        ];
        for setup in cases {
            let mut s = MiniBatch::create();
            s.setup = setup.kept();
            let k = s.setup;
            assert!(k.batch >= 1 && k.batch <= N && k.lr.is_finite() && k.lr > 0.0 && k.train_batch >= 1 && k.steps >= EVERY && k.steps <= MOST_STEPS, "{k:?}");
            s.setup.steps = 30;
            s.train();
            s.finish();
            s.follow(0.016, true);
            for g in [&s.look.window, &s.look.examples, &s.look.batches, &s.look.shape, &s.look.chart, &s.look.noise, &s.look.history] {
                assert!(g.shown.iter().all(|v| v.is_finite()), "{k:?}: {:?}", g.shown);
            }
        }
        // weights that are not numbers: the run stops and says so; the picture keeps the
        // last finite moment
        let mut s = MiniBatch::create();
        s.setup.steps = 20;
        s.train();
        s.finish();
        let mut w = Vec::new();
        terrain::get(&s.run.model, &mut w);
        w[0] = f64::NAN;
        terrain::set(&mut s.run.model, &w);
        s.train();
        s.finish();
        assert!(s.run.diverged && !s.run.running);
        assert!(s.shown.snap.noise.g2.is_finite());
        s.train();
        assert!(!s.run.running, "no more training once diverged");
        s.follow(0.016, true);
        assert!(s.look.history.shown.iter().all(|v| v.is_finite()));
        // the smallest batch and the largest, both ways
        let s0 = run(0);
        for d in DRAWS {
            for b in [1, N] {
                let c = s0.shown.snap.batch_cov(d, b);
                assert!(c.iter().all(|v| v.is_finite() && *v >= -1e-15 || c[1].is_finite()));
                assert!(ellipse([0.0, 0.0], c).iter().all(|p| p.0.is_finite() && p.1.is_finite()));
            }
        }
        assert!(ellipse([0.0, 0.0], [0.0, 0.0, 0.0]).iter().all(|p| p.0 == 0.0 && p.1 == 0.0));
    }

    /// The layout keeps everything in the window and apart, from a phone to a desktop.
    #[test]
    fn the_layout_fits_every_window() {
        for (w, h) in [(1060.0, 1000.0), (1600.0, 1000.0), (1220.0, 700.0), (900.0, 600.0), (390.0, 800.0), (360.0, 640.0), (700.0, 900.0), (300.0, 600.0), (500.0, 400.0), (240.0, 180.0)] {
            let lay = Layout::of(Vec2::new(w, h), 40.0);
            let inside = |(a, b): Rect| a.x >= 0.0 && a.y >= 0.0 && b.x <= w + 0.5 && b.y <= h + 0.5 && a.is_finite() && b.is_finite();
            let apart = |(a, b): Rect, (c, d): Rect| b.x <= c.x || d.x <= a.x || b.y <= c.y || d.y <= a.y;
            assert!(inside(lay.window), "{w}x{h}: {:?}", lay.window);
            if lay.compact {
                continue;
            }
            let boxes: Vec<Rect> = [vec![lay.window, lay.chart, lay.history, lay.strip], lay.draw_buttons.to_vec(), lay.train_buttons.to_vec()].concat();
            for (i, a) in boxes.iter().enumerate() {
                assert!(inside(*a), "{w}x{h}: box {i} {a:?}");
                for b in &boxes[i + 1..] {
                    assert!(apart(*a, *b), "{w}x{h}: {a:?} and {b:?} overlap");
                }
            }
            assert!(lay.window.1.y - lay.window.0.y - key_height(&lay) > 100.0, "{w}x{h}: the window {:?}", lay.window);
            let least = if h < 700.0 { 70.0 } else { 100.0 };
            assert!(lay.chart.1.y - lay.chart.0.y > least && lay.chart.1.x - lay.chart.0.x > 200.0, "{w}x{h}: {:?}", lay.chart);
            assert!(lay.history.1.y - lay.history.0.y > 0.5 * least && lay.strip.1.y - lay.strip.0.y >= 32.0, "{w}x{h}: {:?} {:?}", lay.history, lay.strip);
        }
    }

    /// The picture's glides come to rest after a change.
    #[test]
    fn glides_arrive_and_stop() {
        let mut s = run(0);
        s.follow(0.016, true);
        for k in [1, 2, 3, 0] {
            s.set_up(k);
            s.finish();
            let mut frames = 0;
            s.follow(1.0 / 60.0, false);
            while s.look.moving {
                s.follow(1.0 / 60.0, false);
                frames += 1;
                assert!(frames < 300, "experiment {k} still moving");
            }
            assert!(frames > 3, "experiment {k}: {frames}");
        }
    }

    /// Training is paced by the scene's time: none while it stands still, about RATE
    /// steps a second of it otherwise (measuring as it goes).
    #[test]
    fn training_follows_the_scenes_time() {
        let mut s = MiniBatch::create();
        s.train();
        let budget = Budget::new(f64::INFINITY);
        for _ in 0..30 {
            s.work(0.0, &budget);
        }
        assert_eq!(s.run.steps, 0);
        for _ in 0..60 {
            s.work(1.0 / 60.0, &budget);
        }
        assert!((s.run.steps as f64 - RATE).abs() <= 2.0 + EVERY as f64, "{}", s.run.steps);
    }
}

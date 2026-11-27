//! Muon, taken apart on one layer of a small network. Muon is a rule for a network's
//! weight matrices: it keeps the usual momentum of the gradient, replaces that matrix
//! by one with the same singular vectors and every singular value pushed towards 1
//! (by a short odd polynomial applied a few times, which acts on each singular value
//! alone), multiplies by a factor that depends on the matrix's shape, and steps along
//! that.
//!
//! The network is an MLP (2 inputs, three tanh layers of 16, 3 classes) trained for
//! 300 steps on spiral data with Muon as the reference repository has it (Jordan and
//! co-authors 2024; the course notes, muon.md §2.2), at this scene's learning rates.
//! Its next step on a chosen layer is shown four ways: (a) the singular values of the
//! gradient, of the momentum handed to the iteration, and after each round (a film);
//! (b) the polynomial and where one singular value goes round by round; (c) how far
//! the step moves the layer's output for a unit input along each principal direction
//! of its inputs, under a gradient step and under Muon's; and (d) either a single
//! linear layer fitted by gradient descent, Adam and Muon, the loss along each input
//! direction, or this network trained with the reference's iteration, the exact
//! polar factor and the classical cubic.
//!
//! The rules are the library's (`math::optim`): every scene of the course means the
//! same thing by "Muon" and "Adam".

use super::*;
use crate::clock::{self, Budget};
use crate::math::arch::{Init, Model, MlpSpec, Stage};
use crate::math::autograd::{Act, Tape};
use crate::math::optim::{self, Aux, Groups, Hyper, Iteration, Odd5, Optimizer, Ortho, Rule};
use crate::math::terrain::{Draw, Problem, Sampler};
use crate::scenes::stats::plot::{self, Glide, Plot, Rect, AXES, BUTTONS, CURVES, MARKS};

pub const DEF: SceneDef = SceneDef {
    id: "muon",
    title: "Muon: an orthogonal step",
    group: GROUP,
    blurb: "Muon replaces a layer's momentum by a matrix with the same singular vectors and every singular value pushed towards one, by a short polynomial iteration, and steps along it: what that does to a real layer's step.",
    make: |_| Box::new(MuonScene::create()),
};

// ----- the network whose layers are taken apart ------------------------------------------

/// The data: points on three spiral arms, one class an arm.
const POINTS: usize = 192;
/// Units in each of the three hidden layers.
const WIDTH: usize = 16;
const DEPTH: usize = 3;
/// The batch the network is trained with.
const BATCH: usize = 32;
/// Steps of training before the layer is looked at.
const TRAINED: usize = 300;
/// Muon's learning rate here, and the other parameters' Adam's. (The reference's
/// defaults are 0.02 and 0.0003; with them this small network learns slowly: the test
/// `the_notes_say_what_is_measured`.)
const LR: f64 = 0.05;
const AUX_LR: f64 = 0.01;
/// Muon's momentum, the reference's β.
const BETA: f64 = 0.95;
const DATA_SEED: u64 = 1;
const MODEL_SEED: u64 = 2;
const BATCH_SEED: u64 = 3;
/// The order the examples of the gradient looked at are taken in (a batch of B is the
/// first B of it, so that a larger batch holds every smaller one).
const PICK_SEED: u64 = 4;
/// The most examples in that gradient: all of them.
const MOST_BATCH: usize = POINTS;
/// The most rounds of the iteration a control allows.
const MOST_ROUNDS: usize = 10;
/// Where every singular value lands after enough rounds of the reference's quintic,
/// and after five from any start of at least 0.01 (computed in the course notes,
/// muon.md §8.3, and held by the library's test
/// `where_the_reference_polynomial_sends_the_unit_interval`).
const BAND: [f64; 2] = [0.6818, 1.1344];

fn spec() -> MlpSpec {
    MlpSpec { inputs: 2, width: WIDTH, depth: DEPTH, outputs: 3, act: Act::Tanh, init: Init::Xavier, ..MlpSpec::default() }
}

/// Muon as the reference repository has it now (momentum 0.95 in its averaging form,
/// Nesterov's mix, five rounds of its quintic, the factor √max(1, fan-out / fan-in)),
/// on the hidden layers, with Adam for the rest: at this scene's learning rates, with
/// the orthogonalisation given.
fn hyper(ortho: Ortho) -> Hyper {
    Hyper { lr: LR, ortho, aux: Aux { lr: AUX_LR, ..Aux::REFERENCE }, ..Hyper::muon_reference() }
}

/// The four weight matrices of the network.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
enum Layer {
    /// The first hidden layer, which reads the data: 16 by 2.
    Input,
    /// The second and third hidden layers: 16 by 16.
    Second,
    Third,
    /// The read-out: 3 by 16.
    Readout,
}

impl Layer {
    const ALL: [Layer; 4] = [Layer::Input, Layer::Second, Layer::Third, Layer::Readout];

    /// The group's name in the model.
    fn group(self) -> &'static str {
        match self {
            Layer::Input => "layer 1 weights",
            Layer::Second => "layer 2 weights",
            Layer::Third => "layer 3 weights",
            Layer::Readout => "read-out weights",
        }
    }

    /// What the picture calls it.
    fn name(self) -> &'static str {
        match self {
            Layer::Input => "layer 1",
            Layer::Second => "layer 2",
            Layer::Third => "layer 3",
            Layer::Readout => "the read-out",
        }
    }

    /// Its index among the stages of a pass (whose input it reads).
    fn depth(self) -> usize {
        match self {
            Layer::Input => 1,
            Layer::Second => 2,
            Layer::Third => 3,
            Layer::Readout => DEPTH + 1,
        }
    }

    /// Rows (fan-out) and columns (fan-in).
    fn shape(self) -> (usize, usize) {
        match self {
            Layer::Input => (WIDTH, 2),
            Layer::Second | Layer::Third => (WIDTH, WIDTH),
            Layer::Readout => (3, WIDTH),
        }
    }

    /// Whether the reference gives it to Muon (the hidden layers' matrices) or to Adam.
    fn muons(self) -> bool {
        matches!(self, Layer::Second | Layer::Third)
    }

    /// The reference's shape factor √max(1, fan-out / fan-in).
    fn factor(self) -> f64 {
        let (r, c) = self.shape();
        (r as f64 / c as f64).max(1.0).sqrt()
    }

    fn index(self) -> usize {
        Layer::ALL.iter().position(|&l| l == self).unwrap_or(1)
    }
}

/// The network after `TRAINED` steps of Muon, with what is needed to take a step's
/// gradient apart: the data, the weights, and for every matrix the momentum Muon keeps
/// (for the layers the reference gives Adam, the momentum Muon would have kept had it
/// had them: the same running average of their gradients).
#[derive(Clone, Debug)]
struct Net {
    problem: Problem,
    model: Model,
    /// The momentum M of each layer of `Layer::ALL`, after the training.
    momentum: [Vec<f64>; 4],
    /// The optimiser's state after the training (the tests hold the scene's own
    /// momentum and step to it).
    #[cfg(test)]
    opt: Optimizer,
    /// The order the examples of the looked-at gradient come in.
    order: Vec<usize>,
    /// The loss on all the data, at the start and now.
    loss_start: f64,
    loss: f64,
}

impl Net {
    fn new() -> Net {
        let problem = Problem::spirals(POINTS, 3, &mut Rng::new(DATA_SEED));
        let mut model = Model::mlp(&spec(), &mut Rng::new(MODEL_SEED));
        let mut opt = Optimizer::new(&model, Rule::Muon, hyper(Ortho::Polynomial(Iteration::JORDAN)));
        let mut tape = Tape::new();
        let mut batches = Sampler::new(problem.len(), BATCH, Draw::WithoutReplacement, Rng::new(BATCH_SEED));
        let index = Layer::ALL.map(|l| model.index_of(l.group()).unwrap_or(0));
        let mut momentum = index.map(|i| vec![0.0; model.groups()[i].value.len()]);
        let loss_start = problem.loss(&model, &mut tape);
        for _ in 0..TRAINED {
            let idx: Vec<usize> = batches.deal().to_vec();
            problem.grad(&mut model, &mut tape, &idx);
            for (m, &i) in momentum.iter_mut().zip(&index) {
                for (m, g) in m.iter_mut().zip(&model.groups()[i].grad) {
                    *m = BETA * *m + (1.0 - BETA) * g;
                }
            }
            opt.step(&mut model, LR);
        }
        let loss = problem.loss(&model, &mut tape);
        let mut order: Vec<usize> = (0..problem.len()).collect();
        Rng::new(PICK_SEED).shuffle(&mut order);
        Net {
            problem,
            model,
            momentum,
            #[cfg(test)]
            opt,
            order,
            loss_start,
            loss,
        }
    }
}

// ----- the iteration ------------------------------------------------------------------------

/// How the momentum is made orthogonal.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
enum Method {
    /// The reference's quintic (3.4445, −4.7750, 2.0315).
    Quintic,
    /// The classical cubic 1.5x − 0.5x³.
    Cubic,
    /// The exact polar factor U Vᵀ.
    Exact,
}

impl Method {
    const ALL: [Method; 3] = [Method::Quintic, Method::Cubic, Method::Exact];

    fn poly(self) -> Option<Odd5> {
        match self {
            Method::Quintic => Some(Odd5::JORDAN),
            Method::Cubic => Some(Odd5::CUBIC),
            Method::Exact => None,
        }
    }

    /// The iteration of `rounds` rounds (None for the exact factor).
    fn iteration(self, rounds: usize) -> Option<Iteration> {
        match self {
            Method::Quintic => Some(Iteration::JORDAN.with_rounds(rounds)),
            Method::Cubic => Some(Iteration::CUBIC.with_rounds(rounds)),
            Method::Exact => None,
        }
    }

    fn ortho(self, rounds: usize) -> Ortho {
        self.iteration(rounds).map_or(Ortho::Exact, Ortho::Polynomial)
    }

    fn color(self) -> Color {
        match self {
            Method::Quintic => MUON,
            Method::Cubic => CUBIC,
            Method::Exact => EXACT,
        }
    }

    fn name(self) -> &'static str {
        match self {
            Method::Quintic => "the reference's quintic",
            Method::Cubic => "the classical cubic",
            Method::Exact => "the exact polar factor",
        }
    }
}

/// What `rounds` rounds of `p` do to one number.
fn rounds_of(p: Odd5, s: f64, rounds: usize) -> f64 {
    (0..rounds).fold(s, |x, _| p.at(x))
}

/// The smallest start (a singular value over the Frobenius norm) from which `rounds`
/// rounds of `p` reach `goal`: the starts below it end below the goal (found by
/// bisection on the log of the start, from the top of the interval where the map
/// first fails). None if every start in [1e-9, 1] gets there, or none does.
fn least_start(p: Odd5, rounds: usize, goal: f64) -> Option<f64> {
    // the largest start in [1e-9, 1] that ends below the goal, on a fine grid, then bisected
    let n = 4000;
    let at = |k: usize| 10f64.powf(-9.0 + 9.0 * k as f64 / n as f64);
    let fails = |s: f64| rounds_of(p, s, rounds) < goal;
    let last = (0..=n).rev().find(|&k| fails(at(k)))?;
    if last == n {
        return None;
    }
    let (mut lo, mut hi) = (at(last).ln(), at(last + 1).ln());
    for _ in 0..60 {
        let mid = 0.5 * (lo + hi);
        if fails(mid.exp()) {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    Some(hi.exp())
}

/// Accurate singular values of `a` (`rows` by `cols`), largest first: the eigenvalues
/// of the symmetric factor of its polar decomposition, a = P H (H = Pᵀ a for a tall or
/// square matrix, a Pᵀ for a wide one). The polar factor P is found by one-sided
/// rotations, so a singular value is right to about the rounding unit times the
/// largest, where those taken from a Gram matrix are lost below about 1e-8 of it; and
/// a direction the polar factor counts as rounding gives exactly 0.
fn accurate_singular_values(a: &[f64], rows: usize, cols: usize) -> Vec<f64> {
    let k = rows.min(cols);
    let n = rows * cols;
    if a.len() < n || k == 0 {
        return vec![f64::NAN; k];
    }
    let p = optim::polar_factor(a, rows, cols);
    let mut h = vec![0.0; k * k];
    for i in 0..k {
        for j in 0..k {
            h[i * k + j] = if rows >= cols {
                // (Pᵀ a)_ij = Σ_r p_ri a_rj
                (0..rows).map(|r| p[r * cols + i] * a[r * cols + j]).sum()
            } else {
                // (a Pᵀ)_ij = Σ_c a_ic p_jc
                (0..cols).map(|c| a[i * cols + c] * p[j * cols + c]).sum()
            };
        }
    }
    // (symmetric to rounding: the average of the two halves)
    for i in 0..k {
        for j in i + 1..k {
            let s = 0.5 * (h[i * k + j] + h[j * k + i]);
            h[i * k + j] = s;
            h[j * k + i] = s;
        }
    }
    let (vals, _) = optim::sym_eigen(&h, k);
    // (what is left in a direction the polar factor counts as rounding is rounding too:
    // the same threshold, 4 max(rows, cols) ε σ_max)
    let largest = vals.iter().fold(0.0f64, |m, &v| if v.is_nan() { f64::NAN } else { m.max(v) });
    let zero = 4.0 * rows.max(cols) as f64 * f64::EPSILON * largest;
    vals.iter().map(|&v| if v.is_nan() || largest.is_nan() { f64::NAN } else if v > zero { v } else { 0.0 }).collect()
}

/// Everything the picture of the layer depends on.
#[derive(Clone, Copy, Debug, PartialEq)]
struct Setup {
    layer: Layer,
    /// Examples in the gradient looked at.
    batch: usize,
    /// The momentum empty: the first step (else, the one after `TRAINED` steps).
    fresh: bool,
    method: Method,
    rounds: usize,
}

impl Setup {
    fn base() -> Setup {
        Setup { layer: Layer::Second, batch: BATCH, fresh: false, method: Method::Quintic, rounds: 5 }
    }

    fn kept(self) -> Setup {
        Setup { batch: self.batch.clamp(1, MOST_BATCH), rounds: self.rounds.clamp(1, MOST_ROUNDS), ..self }
    }

    /// The rounds the film shows: the iteration's, or one for the exact factor.
    fn iterations(&self) -> usize {
        match self.method {
            Method::Exact => 1,
            _ => self.rounds,
        }
    }

    /// The film's stages: the gradient, the momentum, each round, the step.
    fn stages(&self) -> usize {
        self.iterations() + 3
    }
}

/// One step of Muon on the chosen layer, taken apart.
#[derive(Clone, Debug)]
struct Snapshot {
    setup: Setup,
    rows: usize,
    cols: usize,
    /// The gradient of the batch, G, and the matrix handed to the iteration, U (the
    /// momentum mixed with the gradient as Nesterov's form has it).
    g: Vec<f64>,
    u: Vec<f64>,
    /// The bars of each stage: G's singular values, U's (each divided by its matrix's
    /// Frobenius norm; U's by ‖U‖_F + 1e-7, the reference's normaliser), then what each
    /// round makes of U's values (the polynomial applied to each in turn: the matrix
    /// iteration does exactly that, keeping the singular vectors), then the step's
    /// (times the shape factor). From stage 1 on, bar i is U's i-th singular direction.
    bars: Vec<Vec<f64>>,
    /// The matrix of each stage (G, U normalised, the rounds, the step), for the
    /// picture of the inputs.
    mats: Vec<Vec<f64>>,
    /// The ranks of G and of U (their singular values above rounding).
    rank_g: usize,
    rank: usize,
    /// How far the layer's inputs reach (root mean square) along each of their
    /// principal directions over all the data, largest first.
    reach: Vec<f64>,
    /// For each stage, how far a unit input along each principal direction moves the
    /// output under that stage's matrix, drawn at the spectral norm of the step (the
    /// step's own row is exact; G's is the gradient step scaled to the same largest
    /// stretch).
    moved: Vec<Vec<f64>>,
}

/// The largest absolute value, or 0.
fn top(v: &[f64]) -> f64 {
    v.iter().fold(0.0f64, |m, x| m.max(x.abs()))
}

impl Snapshot {
    fn new(net: &Net, setup: Setup, tape: &mut Tape) -> Snapshot {
        let setup = setup.kept();
        let layer = setup.layer;
        let (rows, cols) = layer.shape();
        let n = rows * cols;
        let gi = net.model.index_of(layer.group()).unwrap_or(0);
        // the gradient of the first `batch` examples of the order
        let idx = &net.order[..setup.batch.min(net.order.len())];
        let mut model = net.model.clone();
        net.problem.grad(&mut model, tape, idx);
        let mut g = model.groups()[gi].grad.clone();
        g.resize(n, 0.0);
        // the momentum's update and Nesterov's mix, in the reference's averaging form
        let old = &net.momentum[layer.index()];
        let u: Vec<f64> = (0..n)
            .map(|i| {
                let m = if setup.fresh { 0.0 } else { old.get(i).copied().unwrap_or(0.0) };
                let m = BETA * m + (1.0 - BETA) * g[i];
                (1.0 - BETA) * g[i] + BETA * m
            })
            .collect();
        let g_f = optim::frobenius(&g);
        let g_s: Vec<f64> = accurate_singular_values(&g, rows, cols).iter().map(|s| if g_f > 0.0 { s / g_f } else { 0.0 }).collect();
        let d = optim::frobenius(&u) + 1e-7;
        let u_s: Vec<f64> = accurate_singular_values(&u, rows, cols).iter().map(|s| s / d).collect();
        let rank_g = g_s.iter().filter(|&&s| s > 0.0).count();
        let rank = u_s.iter().filter(|&&s| s > 0.0).count();
        // the stages
        let mut bars = vec![g_s, u_s.clone()];
        let mut mats = vec![g.clone(), u.iter().map(|x| x / d).collect::<Vec<f64>>()];
        match setup.method.poly() {
            Some(p) => {
                let mut x = u_s.clone();
                for r in 1..=setup.rounds {
                    x.iter_mut().for_each(|v| *v = p.at(*v));
                    bars.push(x.clone());
                    let mut m = u.clone();
                    if let Some(it) = setup.method.iteration(r) {
                        it.run(&mut m, rows, cols, &mut optim::Scratch::new());
                    }
                    mats.push(m);
                }
            }
            None => {
                bars.push(u_s.iter().map(|&s| if s > 0.0 { 1.0 } else { 0.0 }).collect());
                mats.push(optim::polar_factor(&u, rows, cols));
            }
        }
        let f = layer.factor();
        let last = bars.last().cloned().unwrap_or_default();
        bars.push(last.iter().map(|v| f * v).collect());
        let step: Vec<f64> = mats.last().map(|m| m.iter().map(|x| f * x).collect()).unwrap_or_default();
        mats.push(step);
        // the layer's inputs over all the data, and their principal directions
        let mut x = Vec::with_capacity(net.problem.len() * 2);
        for k in 0..net.problem.len() {
            if let Some((p, _)) = net.problem.point(k) {
                x.extend_from_slice(p);
            }
        }
        let trace = net.model.forward_mlp(tape, &x);
        let a = trace.get(Stage::Input, layer.depth()).map(|v| tape.value(v).to_vec()).unwrap_or_default();
        let count = net.problem.len().max(1);
        let mut second = vec![0.0; cols * cols];
        if a.len() >= count * cols {
            for k in 0..count {
                let row = &a[k * cols..(k + 1) * cols];
                for i in 0..cols {
                    for j in 0..cols {
                        second[i * cols + j] += row[i] * row[j] / count as f64;
                    }
                }
            }
        }
        let (vals, dirs) = optim::sym_eigen(&second, cols);
        let reach: Vec<f64> = vals.iter().map(|v| v.max(0.0).sqrt()).collect();
        // how far a unit input along each direction moves the output, each stage's
        // matrix drawn at the step's largest stretch
        let along = |m: &[f64], j: usize| -> f64 {
            let v = &dirs[j * cols..(j + 1) * cols];
            (0..rows).map(|i| (0..cols).map(|c| m[i * cols + c] * v[c]).sum::<f64>().powi(2)).sum::<f64>().sqrt()
        };
        let target = top(bars.last().map_or(&[][..], |b| &b[..]));
        let moved: Vec<Vec<f64>> = mats
            .iter()
            .zip(&bars)
            .enumerate()
            .map(|(k, (m, b))| {
                // (the spectral norm: G's and U's own, a round's from its values)
                let norm = if k < 2 { optim::spectral_norm(m, rows, cols) } else { top(b) };
                let scale = if norm > 0.0 && norm.is_finite() { target / norm } else { 0.0 };
                (0..cols).map(|j| scale * along(m, j)).collect()
            })
            .collect();
        Snapshot { setup, rows, cols, g, u, bars, mats, rank_g, rank, reach, moved }
    }

    /// The step's singular values in units of the learning rate.
    fn step(&self) -> &[f64] {
        self.bars.last().map_or(&[][..], |b| &b[..])
    }

    /// The values after the last round (before the shape factor).
    fn after(&self) -> &[f64] {
        let k = self.bars.len();
        if k >= 2 {
            &self.bars[k - 2]
        } else {
            &[]
        }
    }

    /// How many of the values after the last round lie in the band [0.68, 1.13]
    /// (to the rounding of the band's ends: 0.6818 and 1.1344).
    fn in_band(&self) -> usize {
        self.after().iter().filter(|&&v| (BAND[0] - 1e-4..=BAND[1] + 1e-4).contains(&v)).count()
    }

    /// The root mean square of the step's entries, in units of the learning rate.
    fn step_rms(&self) -> f64 {
        let m = self.mats.last().map_or(&[][..], |m| &m[..]);
        (optim::frobenius(m).powi(2) / (self.rows * self.cols).max(1) as f64).sqrt()
    }
}

/// The floating-point operations of the rounds and of the layer's forward and backward
/// pass (Jordan and co-authors 2024, "Runtime analysis"; the course notes, muon.md §8.6):
/// a round on an m by n matrix (m ≤ n) is 4m²n + 2m³; the layer's passes on a batch of
/// B are 6mnB. Returns (rounds, passes).
fn flops(rows: usize, cols: usize, rounds: usize, batch: usize) -> (f64, f64) {
    let (m, n) = (rows.min(cols) as f64, rows.max(cols) as f64);
    (rounds as f64 * (4.0 * m * m * n + 2.0 * m * m * m), 6.0 * m * n * batch as f64)
}

/// What it costs here, measured once for each picture of a layer (microseconds, on
/// this computer, now): the rounds, the exact polar factor, the batch's gradient.
#[derive(Clone, Copy, Debug, Default)]
struct Cost {
    rounds: f64,
    exact: f64,
    gradient: f64,
}

impl Cost {
    fn measure(net: &Net, s: &Snapshot, tape: &mut Tape) -> Cost {
        let reps = 5;
        let (rows, cols) = (s.rows, s.cols);
        let mut scratch = optim::Scratch::new();
        let it = Iteration::JORDAN.with_rounds(s.setup.rounds.max(1));
        let mut m = s.u.clone();
        let t0 = clock::now();
        for _ in 0..reps {
            m.copy_from_slice(&s.u);
            it.run(&mut m, rows, cols, &mut scratch);
        }
        let t1 = clock::now();
        let mut out = vec![0.0; rows * cols];
        for _ in 0..reps {
            optim::polar_into(&s.u, rows, cols, &mut out, &mut scratch);
        }
        let t2 = clock::now();
        let idx = &net.order[..s.setup.batch.min(net.order.len())];
        let mut model = net.model.clone();
        for _ in 0..reps {
            net.problem.grad(&mut model, tape, idx);
        }
        let t3 = clock::now();
        let us = |a: f64, b: f64| (b - a) * 1e6 / reps as f64;
        Cost { rounds: us(t0, t1), exact: us(t1, t2), gradient: us(t2, t3) }
    }
}

// ----- one linear layer --------------------------------------------------------------------

/// Directions of the inputs of the linear layer, and its outputs.
const LIN_N: usize = 8;
/// Steps of each run.
const LIN_STEPS: usize = 200;
/// The target's seed.
const LIN_SEED: u64 = 7;
/// The ratios of the largest variance of the inputs to the smallest.
const RATIOS: [f64; 5] = [1.0, 10.0, 100.0, 1000.0, 10_000.0];
/// The rules raced on it, each at its best step from `GRID`.
const LIN_RULES: [Rule; 3] = [Rule::Sgd, Rule::Adam, Rule::Muon];
/// The step sizes tried: 10^(−4 + 0.2k) for k = 0 to 24 (`grid_lr`).
#[cfg(test)]
const GRID: usize = 25;

fn grid_lr(k: usize) -> f64 {
    10f64.powf(-4.0 + 0.2 * k as f64)
}

/// The step from `GRID` with which each rule (`LIN_RULES`) has the lowest loss after
/// `LIN_STEPS` steps, for each ratio (`RATIOS`), directions turned (first) or lined up
/// with the coordinates: computed beforehand, and checked by the ignored test
/// `the_linear_layer_s_best_steps_are_the_grid_s`.
const BEST: [[[usize; 3]; 5]; 2] = [[[19, 16, 11], [21, 13, 11], [21, 17, 12], [21, 18, 11], [21, 18, 13]], [[19, 16, 11], [21, 16, 11], [21, 14, 12], [21, 14, 11], [21, 14, 12]]];

/// A linear layer y = W x fitted to the outputs of a fixed W* (8 by 8, entries drawn
/// with variance 1/8), the inputs x having variance λ_j along the direction v_j (λ from 1
/// down to 1/ratio, evenly on a log scale; the directions a random turn of the axes, or
/// the axes themselves). Its loss is the expected squared error, exactly: L = ½ E‖(W −
/// W*) x‖² = ½ Σ_j λ_j ‖(W − W*) v_j‖², a sum of one part per input direction, and its
/// gradient (W − W*) Σ with Σ = Σ_j λ_j v_j v_jᵀ.
#[derive(Clone, Debug)]
struct LinProblem {
    lam: [f64; LIN_N],
    /// v_j as column j (row-major).
    v: Vec<f64>,
    sigma: Vec<f64>,
    target: Vec<f64>,
}

impl LinProblem {
    fn new(ratio: f64, turned: bool) -> LinProblem {
        let n = LIN_N;
        let mut rng = Rng::new(LIN_SEED);
        let lam = std::array::from_fn(|j| ratio.powf(-(j as f64) / (n - 1) as f64));
        let mut v: Vec<f64> = (0..n * n).map(|_| rng.normal() as f64).collect();
        if turned {
            crate::math::linalg::orthonormalise(&mut v, n, n);
        } else {
            v = (0..n * n).map(|k| if k % (n + 1) == 0 { 1.0 } else { 0.0 }).collect();
        }
        let mut sigma = vec![0.0; n * n];
        for a in 0..n {
            for b in 0..n {
                sigma[a * n + b] = (0..n).map(|j| lam[j] * v[a * n + j] * v[b * n + j]).sum();
            }
        }
        let target: Vec<f64> = (0..n * n).map(|_| rng.normal() as f64 / (n as f64).sqrt()).collect();
        LinProblem { lam, v, sigma, target }
    }

    /// The gradient (W − W*) Σ into `out`.
    fn grad(&self, w: &[f64], out: &mut [f64]) {
        let n = LIN_N;
        for i in 0..n {
            for b in 0..n {
                out[i * n + b] = (0..n).map(|a| (w[i * n + a] - self.target[i * n + a]) * self.sigma[a * n + b]).sum();
            }
        }
    }

    /// The loss along each input direction, ½ λ_j ‖(W − W*) v_j‖².
    fn parts(&self, w: &[f64]) -> [f64; LIN_N] {
        let n = LIN_N;
        std::array::from_fn(|j| {
            let e2: f64 = (0..n).map(|i| (0..n).map(|a| (w[i * n + a] - self.target[i * n + a]) * self.v[a * n + j]).sum::<f64>().powi(2)).sum();
            0.5 * self.lam[j] * e2
        })
    }
}

/// One rule's run on the linear layer: the weights live in a model of one hidden layer
/// (its first matrix; the other groups get no gradient and never move), so that the
/// rule is the library's own.
#[derive(Clone, Debug)]
struct LinRun {
    rule: Rule,
    lr: f64,
    model: Model,
    opt: Optimizer,
    /// The loss along each direction after each step, over its value at the start.
    curves: Vec<[f64; LIN_N]>,
    start: [f64; LIN_N],
    grad: Vec<f64>,
}

impl LinRun {
    fn new(p: &LinProblem, rule: Rule, lr: f64) -> LinRun {
        let spec = MlpSpec { inputs: LIN_N, width: LIN_N, depth: 1, outputs: LIN_N, act: Act::Identity, ..MlpSpec::default() };
        let mut model = Model::mlp(&spec, &mut Rng::new(1));
        model.groups_mut().iter_mut().for_each(|g| g.value.fill(0.0));
        let h = match rule {
            // (the reference's Muon, on every matrix: this model's first layer would
            // otherwise count as its input layer and go to Adam)
            Rule::Muon => Hyper { groups: Groups::Matrices, ..Hyper::muon_reference() },
            r => Hyper::default_for(r),
        };
        let opt = Optimizer::new(&model, rule, h);
        let start = p.parts(&model.groups()[0].value);
        let mut curves = Vec::with_capacity(LIN_STEPS + 1);
        curves.push([1.0; LIN_N]);
        LinRun { rule, lr, model, opt, curves, start, grad: vec![0.0; LIN_N * LIN_N] }
    }

    fn done(&self) -> bool {
        self.curves.len() > LIN_STEPS
    }

    fn step(&mut self, p: &LinProblem) {
        if self.done() {
            return;
        }
        p.grad(&self.model.groups()[0].value, &mut self.grad);
        for g in self.model.groups_mut() {
            g.grad.fill(0.0);
        }
        self.model.groups_mut()[0].grad.copy_from_slice(&self.grad);
        self.opt.step(&mut self.model, self.lr);
        let parts = p.parts(&self.model.groups()[0].value);
        let start = self.start;
        self.curves.push(std::array::from_fn(|j| if start[j] > 0.0 { parts[j] / start[j] } else { 0.0 }));
    }

    /// The whole loss after the last step, over its start.
    fn total(&self) -> f64 {
        let last = self.curves.last().copied().unwrap_or([1.0; LIN_N]);
        let s: f64 = self.start.iter().sum();
        if s > 0.0 {
            last.iter().zip(&self.start).map(|(r, s0)| r * s0).sum::<f64>() / s
        } else {
            0.0
        }
    }
}

/// The three rules on one linear layer.
#[derive(Clone, Debug)]
struct LinRace {
    ratio: usize,
    turned: bool,
    problem: LinProblem,
    runs: Vec<LinRun>,
}

impl LinRace {
    fn new(ratio: usize, turned: bool) -> LinRace {
        let ratio = ratio.min(RATIOS.len() - 1);
        let problem = LinProblem::new(RATIOS[ratio], turned);
        let best = BEST[!turned as usize][ratio];
        let runs = LIN_RULES.iter().zip(best).map(|(&r, k)| LinRun::new(&problem, r, grid_lr(k))).collect();
        LinRace { ratio, turned, problem, runs }
    }

    fn done(&self) -> bool {
        self.runs.iter().all(|r| r.done())
    }

    /// Steps, round the runs, until done or the budget is spent (at least one).
    fn work(&mut self, budget: &Budget) {
        let mut first = true;
        while !self.done() && (first || !budget.spent()) {
            for r in &mut self.runs {
                r.step(&self.problem);
            }
            first = false;
        }
    }

    /// The first step after which the loss along direction j is below `share` of its
    /// start and stays there, for run k (None if not within the run).
    fn below_from(&self, k: usize, j: usize, share: f64) -> Option<usize> {
        let c = &self.runs.get(k)?.curves;
        let last_above = (0..c.len()).rev().find(|&t| c[t][j] >= share);
        match last_above {
            Some(t) if t + 1 < c.len() => Some(t + 1),
            Some(_) => None,
            None => Some(0),
        }
    }
}

// ----- this network, trained three ways ----------------------------------------------------

/// Steps of each training run, how often its loss is noted, and the starts.
const RACE_STEPS: usize = 600;
const RACE_EVERY: usize = 10;
const RACE_SEEDS: [u64; 3] = [11, 12, 13];

/// One training run of the network with one orthogonalisation and one start.
#[derive(Clone, Debug)]
struct RaceRun {
    method: Method,
    model: Model,
    opt: Optimizer,
    batches: Sampler,
    /// The loss on all the data every `RACE_EVERY` steps, from step 0.
    losses: Vec<f64>,
    t: usize,
}

impl RaceRun {
    fn new(net: &Net, method: Method, seed: u64, tape: &mut Tape) -> RaceRun {
        let model = Model::mlp(&spec(), &mut Rng::new(seed));
        let opt = Optimizer::new(&model, Rule::Muon, hyper(method.ortho(5)));
        let batches = Sampler::new(net.problem.len(), BATCH, Draw::WithoutReplacement, Rng::new(seed + 100));
        let first = net.problem.loss(&model, tape);
        RaceRun { method, model, opt, batches, losses: vec![first], t: 0 }
    }

    fn done(&self) -> bool {
        self.t >= RACE_STEPS
    }

    fn step(&mut self, net: &Net, tape: &mut Tape) {
        if self.done() {
            return;
        }
        let idx: Vec<usize> = self.batches.deal().to_vec();
        net.problem.grad(&mut self.model, tape, &idx);
        self.opt.step(&mut self.model, LR);
        self.t += 1;
        if self.t.is_multiple_of(RACE_EVERY) {
            self.losses.push(net.problem.loss(&self.model, tape));
        }
    }

    /// The loss at step t (a multiple of `RACE_EVERY`), once reached.
    fn loss_at(&self, t: usize) -> Option<f64> {
        self.losses.get(t / RACE_EVERY).copied()
    }
}

/// The network trained with each orthogonalisation from each start, the same batches
/// for every orthogonalisation of a start.
#[derive(Clone, Debug, Default)]
struct Race {
    runs: Vec<RaceRun>,
}

impl Race {
    fn started(&self) -> bool {
        !self.runs.is_empty()
    }

    fn start(&mut self, net: &Net, tape: &mut Tape) {
        if self.started() {
            return;
        }
        for &seed in &RACE_SEEDS {
            for &method in &[Method::Quintic, Method::Exact, Method::Cubic] {
                self.runs.push(RaceRun::new(net, method, seed, tape));
            }
        }
    }

    fn done(&self) -> bool {
        self.started() && self.runs.iter().all(|r| r.done())
    }

    /// Steps, round the runs, until done or the budget is spent (at least one round).
    fn work(&mut self, net: &Net, tape: &mut Tape, budget: &Budget) {
        let mut first = true;
        while self.started() && !self.done() && (first || !budget.spent()) {
            for r in &mut self.runs {
                r.step(net, tape);
            }
            first = false;
        }
    }

    /// The losses of one orthogonalisation's runs at step t, one a start.
    fn at(&self, method: Method, t: usize) -> Vec<f64> {
        self.runs.iter().filter(|r| r.method == method).filter_map(|r| r.loss_at(t)).collect()
    }
}

// ----- numbers as the scene writes them ------------------------------------------------------

/// A number with three significant digits (0.0129, 0.635, 1.96, 152), with a true minus
/// sign; tiny ones as powers of ten.
fn sig(v: f64) -> String {
    if !v.is_finite() {
        return "?".to_string();
    }
    if v == 0.0 {
        return "0".to_string();
    }
    let a = v.abs();
    if a < 1e-3 {
        let e = a.log10().floor() as i32;
        let m = v / 10f64.powi(e);
        return format!(r"{} \times 10^{{{}}}", num(m, 1), e);
    }
    if a >= 1000.0 {
        return count(a.round() as usize);
    }
    let decimals = (2 - a.log10().floor() as i32).clamp(0, 12) as usize;
    num(v, decimals)
}

/// A number with two significant digits (0.34, 0.088, 1.2), as the captions write
/// most of theirs.
#[cfg(test)]
fn sig2(v: f64) -> String {
    if !v.is_finite() {
        return "?".to_string();
    }
    if v == 0.0 {
        return "0".to_string();
    }
    let decimals = (1 - v.abs().log10().floor() as i32).clamp(0, 12) as usize;
    num(v, decimals)
}

/// The same, between dollars when it is a power of ten.
fn sig_text(v: f64) -> String {
    let s = sig(v);
    if s.contains('\\') {
        format!("${s}$")
    } else {
        s
    }
}

/// A step size as a label writes it: two significant digits.
fn rate(v: f64) -> String {
    if !(v.is_finite() && v > 0.0) {
        return "?".to_string();
    }
    let decimals = (1 - v.log10().floor() as i32).clamp(0, 9) as usize;
    num(v, decimals)
}

/// A power of ten as an axis writes it.
fn power(d: i32) -> String {
    match d {
        0 => "1".to_string(),
        1 => "10".to_string(),
        _ => format!("$10^{{{d}}}$"),
    }
}

/// A ratio as markup (without dollars): 1, 10, 100, 1000, 10⁴.
fn ratio_math(k: usize) -> &'static str {
    match k {
        0 => "1",
        1 => "10",
        2 => "100",
        3 => "1000",
        _ => "10^4",
    }
}

/// A ratio as the controls write it.
fn ratio_name(k: usize) -> String {
    match k {
        4 => "$10^4$".to_string(),
        _ => ratio_math(k).to_string(),
    }
}

// ----- colours ---------------------------------------------------------------------------------

/// The gradient, and a gradient step.
const GRAD: Color = SGD;
/// The momentum handed to the iteration.
const MOMENT: Color = MOMENTUM;
/// The exact polar factor, and the classical cubic.
const EXACT: Color = Color::hex(0xB9C4FF);
const CUBIC: Color = Color::hex(0xB8BFCC);
/// The inputs' reach along their directions.
const REACH: Color = pal::DIM;
/// The band the quintic's values land in.
const BAND_COLOR: Color = pal::FAINT;

/// The colour of stage k of the film.
fn stage_color(s: &Setup, k: usize) -> Color {
    match k {
        0 => GRAD,
        1 => MOMENT,
        _ => s.method.color(),
    }
}

// ----- the experiments -------------------------------------------------------------------------

/// What the lower right chart shows.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Lower {
    /// One linear layer, three rules.
    Linear,
    /// This network trained with three orthogonalisations.
    Training,
}

/// Which chart a narrow window shows under the spectrum.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Focus {
    Poly,
    Inputs,
    Lower,
}

struct Trial {
    button: &'static str,
    caption: &'static str,
}

const TRIALS: [Trial; 6] = [
    Trial {
        button: "the rounds one at a time",
        caption: r"Layer 2's gradient on a batch of 32, after 300 steps of training (cyan; each singular value over the matrix's Frobenius norm): three values stand out, 0.94, 0.30 and 0.14, and the rest fall from 0.011 to $3.9 \times 10^{-9}$. Muon works on its momentum mixed with it, $\tilde M$ (pink), whose values are more even: 0.71, 0.52, 0.45, then 0.10 down to $9.9 \times 10^{-5}$. Each round sends every value $s$ to $p(s) = 3.4445s - 4.775s^3 + 2.0315s^5$ and keeps the singular vectors: a small value grows about 3.44 times, while values near 1 are thrown about between 0.68 and 1.2. After five rounds 11 of the 16 are in $[0.68, 1.13]$; the five that started below 0.0015 end at 0.39 or less. The value followed on the chart of $p$ starts at 0.0027 and needs all five rounds: 0.0093, 0.032, 0.11, 0.37, 1.05.",
    },
    Trial {
        button: "a gradient from one example",
        caption: r"One example, and the momentum empty (the first step). A layer's gradient from one example is its error times its input, $\delta x^\top$ (scene backpropmat): rank 1, one singular value, the other 15 exactly 0. So $\tilde M$ is a multiple of $G$, the iteration has one value to work on, 1 after the division, and the zeros stay 0: five rounds give 0.70, 1.11, 0.72, 1.09, 0.70. Muon's step is then the gradient's own direction with spectral norm $0.70\eta$: it only rescales the gradient, and on the chart of the inputs it moves the output exactly as a gradient step does. It is not an orthogonal matrix: 15 of its singular values are 0. A batch of $B$ gives rank at most $B$: try the batch slider.",
    },
    Trial {
        button: "the classical cubic instead",
        caption: r"The same momentum, five rounds of the classical cubic $1.5s - 0.5s^3$, which converges to 1 from any start between 0 and $\sqrt 3$, but slowly from small ones: its slope at 0 is 1.5, so a small value grows 1.5 times a round, 7.6 times in five. Here the three largest reach 1.000, the fourth 0.67, and the rest stay below 0.36; the value followed goes from 0.0027 only to 0.021. To reach 0.68 in five rounds the cubic needs a start of at least 0.105, the reference's quintic 0.0015. That is the trade the quintic's authors made: they chose its coefficients to make the slope at 0 'as large as possible', accepting values that end anywhere in about $[0.7, 1.3]$ (Jordan and co-authors, 2024). Later versions of the NanoGPT speed-run code use other polynomials, a different one each round.",
    },
    Trial {
        button: "the exact polar factor, in training",
        caption: r"The exact polar factor $UV^\top$ in place of the polynomial: every value of $\tilde M$ that is not 0 becomes exactly 1. In the training chart, this network trained from three starts with each, at the same step size and on the same batches. Here the exact factor is ahead early and the cubic behind, from every start: at step 300 the losses are 0.34, 0.088 and 0.14 with the exact factor, 0.48, 0.14 and 0.26 with five rounds of the quintic, 0.94, 0.56 and 0.65 with five of the cubic. By step 600 all nine runs are between 0.0037 and 0.073, and their order differs from start to start. On GPT-2-sized language models, Amsel and co-authors (2025) report that more than six rounds, or an exact decomposition, did not lower the final validation loss; Liu and co-authors at Moonshot AI (2025), that ten rounds instead of five did not give better results.",
    },
    Trial {
        button: "inputs spread a thousand to one",
        caption: r"One linear layer $y = Wx$ fitted to a fixed $W^* x$, its inputs varying 1000 times more along one direction than along another (variances $\lambda$ from 1 to $1/1000$ along 8 directions, turned away from the coordinates). Its loss is one part per direction, $\tfrac12\lambda_j \|(W - W^*)v_j\|^2$, and gradient descent shrinks each part by exactly $(1 - \eta\lambda_j)^2$ a step. Each rule at its best step of a grid (the lowest loss after 200 steps): gradient descent gets the widest direction below a tenth of its start in 3 steps and leaves the two narrowest at 0.18 and 0.53 after 200; Adam gets every direction below a tenth by step 132, Muon by step 182, Muon the most evenly (from 54 to 182 steps, Adam from 13 to 132). Muon's step keeps its size however small the error: its spectral norm stays between $0.68\eta$ and $1.2\eta$ (the most the quintic gives any value), where gradient descent's shrinks with the gradient.",
    },
    Trial {
        button: "the shape factor on a tall layer",
        caption: r"Layer 1 reads the 2 numbers of a point into 16 units: a tall matrix, 16 by 2 (the reference gives it to Adam; it is shown here as Muon would treat it). Five rounds take its 2 singular values to 0.91 and 0.75; the reference then multiplies by $\sqrt{\max(1, 16/2)} = 2.83$, and the step's singular values are $2.57\eta$ and $2.12\eta$. So its spectral norm is not $\eta$: the reference's 'learning rate, in units of spectral norm per update' holds only when fan-out $\le$ fan-in. What the factor keeps is the size of each entry's change: with the exact polar factor and a matrix of full rank, the root mean square of the step's entries is $\eta/\sqrt{\text{fan-in}}$ for every shape, here $0.71\eta$ (with the five rounds, $0.59\eta$). Bernstein (2025) derives $\sqrt{\text{fan-out}/\text{fan-in}}$, without the max; Liu and co-authors at Moonshot AI (2025) use $0.2\sqrt{\max(\text{fan-out}, \text{fan-in})}$.",
    },
];

/// The scene opens as the first experiment sets it up.
const OPENING: usize = 0;
/// The singular direction picked to follow, unless the visitor picks another: U's
/// ninth (an index into its values, largest first).
const PICKED: usize = 8;

/// Everything an experiment sets.
#[derive(Clone, Copy, Debug, PartialEq)]
struct Look {
    setup: Setup,
    lower: Lower,
    ratio: usize,
    turned: bool,
    focus: Focus,
    /// The stage the film opens at, and whether it then plays.
    stage: usize,
    play: bool,
    /// The singular direction followed (an index into U's values, largest first).
    pick: usize,
}

fn trial(k: usize) -> Look {
    let base = Look { setup: Setup::base(), lower: Lower::Linear, ratio: 2, turned: true, focus: Focus::Poly, stage: 0, play: true, pick: PICKED };
    match k {
        1 => Look { setup: Setup { batch: 1, fresh: true, ..Setup::base() }, pick: 0, ..base },
        2 => Look { setup: Setup { method: Method::Cubic, ..Setup::base() }, ..base },
        3 => Look { setup: Setup { method: Method::Exact, ..Setup::base() }, lower: Lower::Training, focus: Focus::Lower, ..base },
        4 => Look { ratio: 3, focus: Focus::Lower, stage: Setup::base().stages() - 1, play: false, ..base },
        5 => Look { setup: Setup { layer: Layer::Input, ..Setup::base() }, focus: Focus::Inputs, pick: 1, ..base },
        _ => base,
    }
}

// ----- the film ------------------------------------------------------------------------------

/// Seconds a stage of the film is shown before the next.
const STAGE_TIME: f32 = 1.1;
/// The time constant of the ease from one stage to the next.
const STAGE_EASE: f32 = 0.16;

#[derive(Clone, Copy, Debug, Default)]
struct Film {
    /// The stage shown (whole) and the one the picture is easing towards.
    stage: usize,
    /// Where the picture is, between stages.
    shown: f32,
    playing: bool,
    /// Time on this stage while playing.
    clock: f32,
}

impl Film {
    fn at(stage: usize, play: bool) -> Film {
        Film { stage, shown: stage as f32, playing: play, clock: 0.0 }
    }

    /// Move on by `dt` seconds of the scene's time, in a film of `stages` stages.
    fn advance(&mut self, dt: f32, stages: usize) {
        let last = stages.saturating_sub(1);
        self.stage = self.stage.min(last);
        if self.playing {
            self.clock += dt.max(0.0);
            if self.clock >= STAGE_TIME {
                self.clock = 0.0;
                if self.stage < last {
                    self.stage += 1;
                }
            }
            if self.stage >= last && (self.shown - last as f32).abs() < 1e-3 {
                self.playing = false;
            }
        }
        let goal = self.stage as f32;
        let k = 1.0 - (-dt.max(0.0) / STAGE_EASE).exp();
        self.shown += (goal - self.shown) * k;
        if (self.shown - goal).abs() < 2e-3 {
            self.shown = goal;
        }
    }

    fn moving(&self) -> bool {
        self.playing || self.shown != self.stage as f32
    }
}

// ----- the scene -------------------------------------------------------------------------------

/// The pick: one of U's singular directions, or a value of the visitor's own (dragged
/// on the chart of the polynomial).
#[derive(Clone, Copy, Debug, PartialEq)]
enum Pick {
    Bar(usize),
    Free(f64),
}

pub struct MuonScene {
    net: Net,
    tape: Tape,
    look: Look,
    snap: Snapshot,
    cost: Cost,
    film: Film,
    pick: Pick,
    lin: LinRace,
    race: Race,
    /// What is drawn, gliding: the spectrum's bars (log₁₀), its top; the inputs' bars;
    /// the linear layer's curves (log₁₀).
    bars: Glide,
    moved: Glide,
    curves: Glide,
    /// The window last drawn in (a new one is drawn at once, without gliding).
    canvas: Option<Vec2>,
    gliding: bool,
}

impl MuonScene {
    /// (Nothing here needs the graphics device, so the tests can make one too.)
    fn create() -> MuonScene {
        let net = Net::new();
        let mut tape = Tape::new();
        let look = trial(OPENING);
        let snap = Snapshot::new(&net, look.setup, &mut tape);
        let mut s = MuonScene {
            cost: Cost::default(),
            snap,
            net,
            tape,
            look,
            film: Film::default(),
            pick: Pick::Bar(PICKED),
            lin: LinRace::new(look.ratio, look.turned),
            race: Race::default(),
            bars: Glide::default(),
            moved: Glide::default(),
            curves: Glide::default(),
            canvas: None,
            gliding: false,
        };
        s.set_up(OPENING);
        s
    }

    /// Set up experiment k, and play it.
    fn set_up(&mut self, k: usize) {
        let l = trial(k);
        self.look = l;
        self.pick = Pick::Bar(l.pick);
        self.refresh();
        self.film = Film::at(l.stage.min(self.look.setup.stages() - 1), l.play);
    }

    /// Bring what depends on the settings up to date with them.
    fn refresh(&mut self) {
        self.look.setup = self.look.setup.kept();
        self.look.ratio = self.look.ratio.min(RATIOS.len() - 1);
        if self.snap.setup != self.look.setup {
            self.snap = Snapshot::new(&self.net, self.look.setup, &mut self.tape);
            self.cost = Cost::default();
            let last = self.look.setup.stages() - 1;
            if self.film.stage > last {
                self.film.stage = last;
            }
            self.film.shown = self.film.shown.min(last as f32);
        }
        if (self.lin.ratio, self.lin.turned) != (self.look.ratio, self.look.turned) {
            self.lin = LinRace::new(self.look.ratio, self.look.turned);
        }
        if self.look.lower == Lower::Training && !self.race.started() {
            self.race.start(&self.net, &mut self.tape);
        }
        if let Pick::Bar(i) = self.pick {
            if i >= self.snap.bars[1].len() {
                self.pick = Pick::Bar(self.snap.bars[1].len().saturating_sub(1));
            }
        }
    }

    /// The value the picked direction starts from (U's, over its norm).
    fn pick_start(&self) -> f64 {
        match self.pick {
            Pick::Bar(i) => self.snap.bars[1].get(i).copied().unwrap_or(0.0),
            Pick::Free(s) => s,
        }
    }

    /// Work owed: the training runs, a slice at a time.
    fn work(&mut self, budget: &Budget) {
        if !self.lin.done() {
            self.lin.work(budget);
        }
        if self.race.started() && !self.race.done() && !budget.spent() {
            self.race.work(&self.net, &mut self.tape, budget);
        }
    }

    fn busy(&self) -> bool {
        !self.lin.done() || (self.race.started() && !self.race.done())
    }
}

// ----- the layout --------------------------------------------------------------------------

/// Where everything goes in a window (logical pixels from its top-left corner).
#[derive(Clone, Copy, Debug, PartialEq)]
struct Layout {
    view: Vec2,
    narrow: bool,
    text: f32,
    small: f32,
    /// The buttons that choose the layer.
    layers: [Rect; 4],
    /// The spectrum: its plot, and under it the film's stages (one button each, up to
    /// 14) and its play button.
    spectrum: Rect,
    stages: Rect,
    play: Rect,
    /// The polynomial (a square plot).
    poly: Rect,
    /// The inputs: the strip of their reach, and the bars of how far the output moves.
    reach: Rect,
    inputs: Rect,
    /// The lower right chart's two buttons and its area.
    modes: [Rect; 2],
    lower: Rect,
    /// A narrow window's tabs (polynomial, inputs, the lower chart), and whether it
    /// shows the spectrum alone (too small for more).
    tabs: [Rect; 3],
    compact: bool,
}

impl Layout {
    fn of(view: Vec2, inset: f32, focus: Focus) -> Layout {
        let view = view.max(Vec2::splat(1.0));
        let wide = view.x >= 900.0 && view.y >= 560.0;
        let narrow = !wide;
        let (text, small) = if narrow { (12.0, 10.5) } else { (13.5, 11.5) };
        let side = if narrow { 10.0 } else { 22.0 };
        let top = inset.max(0.0) + if narrow { 6.0 } else { 10.0 };
        let r = |x0: f32, y0: f32, x1: f32, y1: f32| (Vec2::new(x0, y0), Vec2::new(x1.max(x0 + 1.0), y1.max(y0 + 1.0)));
        let none = r(-100.0, -100.0, -99.0, -99.0);
        let button_h = if narrow { 24.0 } else { 26.0 };
        let mut lay = Layout {
            view,
            narrow,
            text,
            small,
            layers: [none; 4],
            spectrum: none,
            stages: none,
            play: none,
            poly: none,
            reach: none,
            inputs: none,
            modes: [none; 2],
            lower: none,
            tabs: [none; 3],
            compact: false,
        };
        let head = text + 8.0;
        let full = view.x - 2.0 * side;
        // the layer buttons along the top
        let layer_row = |lay: &mut Layout, x0: f32, x1: f32, y: f32| {
            for (k, b) in plot::even_row(4, x0, x1, y, button_h, 6.0).into_iter().enumerate().take(4) {
                lay.layers[k] = b;
            }
        };
        if view.x < 250.0 || view.y < 300.0 {
            // the spectrum alone
            lay.compact = true;
            let y0 = top + head + 6.0;
            lay.spectrum = r(side + 30.0, y0, view.x - side, (view.y - 34.0).max(y0 + 1.0));
            return lay;
        }
        if wide {
            // two columns: the spectrum over the inputs on the left, the polynomial
            // over the lower chart on the right
            let gap = 34.0;
            let left_w = (full - gap) * 0.56;
            let x_mid = side + left_w + gap;
            layer_row(&mut lay, side, side + left_w, top);
            let row1 = top + button_h + 14.0;
            let height = view.y - row1;
            let upper_h = (height * 0.52).max(200.0);
            // spectrum: title above, axis labels left and below, the stages under it
            lay.spectrum = r(side + 44.0, row1 + head + 8.0, side + left_w, row1 + upper_h - 2.0 * button_h - 28.0);
            let y = lay.spectrum.1.y + 26.0;
            lay.play = r(side + 44.0, y, side + 44.0 + 86.0, y + button_h);
            lay.stages = r(lay.play.1.x + 8.0, y, side + left_w, y + button_h);
            let row2 = row1 + upper_h + 6.0;
            lay.reach = r(side + 44.0, row2 + head + 8.0, side + left_w, row2 + head + 8.0 + 38.0);
            lay.inputs = r(side + 44.0, lay.reach.1.y + 14.0, side + left_w, view.y - 40.0);
            // the polynomial: square, as large as the upper row allows
            let right_w = view.x - side - x_mid;
            let size = (upper_h - head - 50.0).min(right_w - 50.0).max(60.0);
            let x0 = x_mid + 44.0 + ((right_w - 44.0 - size) * 0.5).max(0.0);
            lay.poly = r(x0, row1 + head + 8.0, x0 + size, row1 + head + 8.0 + size);
            for (k, b) in plot::even_row(2, x_mid, view.x - side, row2, button_h, 6.0).into_iter().enumerate().take(2) {
                lay.modes[k] = b;
            }
            lay.lower = r(x_mid, row2 + button_h + 8.0, view.x - side, view.y - 8.0);
            return lay;
        }
        // one column: the layers, the spectrum and its film, the tabs, one chart (or,
        // where that does not fit, the spectrum alone)
        layer_row(&mut lay, side, view.x - side, top);
        let row1 = top + button_h + 10.0;
        let rows = head + 4.0 + 22.0 + button_h + 10.0 + button_h + 8.0;
        let rest = 150.0;
        let avail = view.y - row1;
        if avail < rows + 90.0 + rest {
            lay.compact = true;
            lay.layers = [none; 4];
            let y0 = top + head + 6.0;
            lay.spectrum = r(side + 30.0, y0, view.x - side, (view.y - 34.0).max(y0 + 1.0));
            return lay;
        }
        let spec_h = ((avail - rows) * 0.4).clamp(90.0, 300.0).min(avail - rows - rest);
        lay.spectrum = r(side + 34.0, row1 + head + 4.0, view.x - side, row1 + head + 4.0 + spec_h);
        let y = lay.spectrum.1.y + 22.0;
        lay.play = r(side, y, side + 64.0, y + button_h);
        lay.stages = r(side + 70.0, y, view.x - side, y + button_h);
        let y = lay.play.1.y + 10.0;
        for (k, b) in plot::even_row(3, side, view.x - side, y, button_h, 6.0).into_iter().enumerate().take(3) {
            lay.tabs[k] = b;
        }
        let y0 = y + button_h + 8.0;
        match focus {
            Focus::Poly => {
                let size = (view.y - y0 - head - 44.0).min(full - 40.0).max(40.0);
                let x0 = side + 36.0 + ((full - 36.0 - size) * 0.5).max(0.0);
                lay.poly = r(x0, y0 + head + 4.0, x0 + size, y0 + head + 4.0 + size);
            }
            Focus::Inputs => {
                lay.reach = r(side + 34.0, y0 + head + 4.0, view.x - side, y0 + head + 4.0 + 30.0);
                lay.inputs = r(side + 34.0, lay.reach.1.y + 12.0, view.x - side, view.y - 36.0);
            }
            Focus::Lower => {
                for (k, b) in plot::even_row(2, side, view.x - side, y0, button_h, 6.0).into_iter().enumerate().take(2) {
                    lay.modes[k] = b;
                }
                lay.lower = r(side, y0 + button_h + 6.0, view.x - side, view.y - 6.0);
            }
        }
        lay
    }

    /// Whether a chart is shown.
    fn shows(&self, rect: Rect) -> bool {
        rect.0.x >= 0.0 && rect.1.x - rect.0.x > 20.0 && rect.1.y - rect.0.y > 20.0
    }
}

// ----- drawing ---------------------------------------------------------------------------------

// Lines whose number of points changes, each in a batch of its own.
const STAIR: u8 = plot::OWN;
/// (one batch for each chart of the linear layer, one for each training run)
const LINEAR: u8 = plot::OWN + 1;
const RACE: u8 = LINEAR + 3;
const LAST_KIND: u8 = RACE + 9;

/// Every batch the scene draws in, made in the same order in every frame, so that one
/// that is empty in a frame does not move the later ones up a place.
fn hold_batches(f: &mut Frame) {
    let corner = f.hud(Vec2::ZERO);
    for kind in [AXES, CURVES, MARKS, BUTTONS].into_iter().chain(STAIR..=LAST_KIND) {
        plot::apart(f, kind, false).segment(corner, corner, 1.0, Color::CLEAR);
    }
    for kind in [AXES, CURVES, MARKS].into_iter().chain(RACE..=LAST_KIND) {
        plot::apart(f, kind, true).segment(corner, corner, 1.0, Color::CLEAR);
    }
}

/// log₁₀, with 0 (and anything not positive) taken to `floor`.
fn lg(v: f64, floor: f64) -> f64 {
    if v > 0.0 && v.is_finite() {
        v.log10().max(floor)
    } else {
        floor
    }
}

/// The spectrum's floor: values below 10⁻⁶ of the Frobenius norm are drawn there.
const FLOOR: f64 = -6.0;

/// The ticks of a log₁₀ axis from `lo` to `hi`: every decade, or every second one when
/// there are many for the room.
fn decades(lo: f64, hi: f64, room: f32) -> Vec<i32> {
    if !(lo.is_finite() && hi.is_finite() && hi > lo) {
        return Vec::new();
    }
    let all: Vec<i32> = (lo.ceil() as i32..=hi.floor() as i32).collect();
    let every = if all.len() as f32 * 18.0 > room { 2 } else { 1 };
    all.into_iter().filter(|d| d % every == 0).collect()
}

/// A plot's left axis of powers of ten (its bottom one written "≤ …" with `floor`: the
/// values below it are drawn there).
fn log_axis(f: &mut Frame, p: &Plot, size: f32, grid: bool, floor: bool) {
    let (min, max) = p.rect;
    plot::line_of(f, AXES, min, Vec2::new(min.x, max.y), 1.0, pal::DIM.a(0.8));
    let bottom = p.y[0].ceil() as i32;
    for d in decades(p.y[0], p.y[1], p.height()) {
        let y = p.px_y(d as f64);
        plot::line_of(f, AXES, Vec2::new(min.x - 4.0, y), Vec2::new(min.x, y), 1.0, pal::DIM.a(0.8));
        if grid {
            plot::line_of(f, AXES, Vec2::new(min.x, y), Vec2::new(max.x, y), 1.0, pal::DIM.a(0.1));
        }
        let name = if floor && d == bottom { format!(r"$\le 10^{{{d}}}$") } else { power(d) };
        f.text_px(Vec2::new(min.x - 6.0, y), name, size, pal::DIM, Align2::RIGHT_CENTER);
    }
}

/// A dashed polyline in the data's units, kept inside the plot.
fn dashed_path(f: &mut Frame, p: &Plot, kind: u8, points: impl IntoIterator<Item = (f64, f64)>, width: f32, color: Color) {
    let pts: Vec<Vec2> = points.into_iter().map(|(x, y)| p.at(x, y)).collect();
    for piece in plot::clip(&pts, p.rect).split(|q| !q.is_finite()).filter(|piece| piece.len() >= 2) {
        let points: Vec<Vec3> = piece.iter().map(|q| f.hud(*q)).collect();
        plot::apart(f, kind, true).polyline(points, width, color);
    }
}

/// The words of a text, a formula between dollars kept whole.
fn words_of(text: &str) -> Vec<&str> {
    let mut out = Vec::new();
    let (mut start, mut inside) = (0, false);
    for (i, c) in text.char_indices() {
        match c {
            '$' => inside = !inside,
            ' ' if !inside => {
                if i > start {
                    out.push(&text[start..i]);
                }
                start = i + 1;
            }
            _ => {}
        }
    }
    if start < text.len() {
        out.push(&text[start..]);
    }
    out
}

/// A text in screen space broken into lines no wider than `width`, from `at` (its
/// top-left); returns the height it took.
fn wrap_px(f: &mut Frame, at: Vec2, width: f32, text: &str, size: f32, color: Color) -> f32 {
    let mut lines: Vec<String> = Vec::new();
    let mut line = String::new();
    for w in words_of(text) {
        let trial = if line.is_empty() { w.to_string() } else { format!("{line} {w}") };
        if !line.is_empty() && f.text_size(&trial, size).x > width {
            lines.push(std::mem::take(&mut line));
            line = w.to_string();
        } else {
            line = trial;
        }
    }
    if !line.is_empty() {
        lines.push(line);
    }
    let step = size + 5.0;
    for (k, l) in lines.iter().enumerate() {
        f.text_px(at + Vec2::new(0.0, k as f32 * step), l.as_str(), size, color, Align2::LEFT_TOP);
    }
    lines.len() as f32 * step
}

impl MuonScene {
    /// The film's stage names: G, U, the rounds (or "exact"), the step.
    fn stage_names(&self) -> Vec<String> {
        let s = &self.look.setup;
        let mut v = vec![r"$G$".to_string(), r"$\tilde M$".to_string()];
        match s.method {
            Method::Exact => v.push("exact".to_string()),
            _ => v.extend((1..=s.rounds).map(|r| r.to_string())),
        }
        v.push("step".to_string());
        v
    }

    /// What stage k is, in words.
    fn stage_words(&self, k: usize, narrow: bool) -> String {
        let s = &self.look.setup;
        let last = s.stages() - 1;
        let (rows, cols) = s.layer.shape();
        if k == 0 {
            format!("the gradient $G$ of a batch of {}", s.batch)
        } else if k == 1 {
            if s.fresh {
                r"$\tilde M$: the momentum, empty before this step, mixed with $G$".to_string()
            } else if narrow {
                r"$\tilde M$: the momentum, mixed with $G$".to_string()
            } else {
                format!(r"$\tilde M$: the momentum of {TRAINED} steps, mixed with $G$ (Nesterov)")
            }
        } else if k == last {
            format!(r"the step, in units of $\eta$: times $\sqrt{{\max(1, {rows}/{cols})}} = {}$", num(s.layer.factor(), 2))
        } else if s.method == Method::Exact {
            r"the exact polar factor: every value not 0 set to 1".to_string()
        } else {
            format!("after round {} of {}", k - 1, s.rounds)
        }
    }

    /// The bars as the film shows them now (log₁₀, at the floor for 0), with the top
    /// of the axis.
    fn bars_goal(&self) -> Vec<f32> {
        let b = &self.snap.bars;
        let last = b.len() - 1;
        let t = self.film.shown.clamp(0.0, last as f32);
        let (i, frac) = (t.floor() as usize, t - t.floor());
        let j = (i + 1).min(last);
        let k = b[i].len();
        let mut out: Vec<f32> = (0..k)
            .map(|n| {
                let (a, c) = (lg(b[i][n], FLOOR), lg(b[j][n], FLOOR));
                (a + (c - a) * frac as f64) as f32
            })
            .collect();
        // the axis reaches above the tallest bar of any stage
        let tallest = b.iter().flat_map(|s| s.iter()).fold(1.0f64, |m, &v| m.max(v));
        out.push(lg(tallest * 3.2, FLOOR) as f32);
        out
    }

    /// The inputs' bars as the film shows them now: the gradient step's, then the
    /// stage's, then the top of the axis.
    fn moved_goal(&self) -> Vec<f32> {
        let m = &self.snap.moved;
        let last = m.len() - 1;
        let t = self.film.shown.clamp(0.0, last as f32);
        let (i, frac) = (t.floor() as usize, t - t.floor());
        let j = (i + 1).min(last);
        let k = m[0].len();
        let mut out: Vec<f32> = m[0].iter().map(|&v| v as f32).collect();
        out.extend((0..k).map(|n| (m[i][n] + (m[j][n] - m[i][n]) * frac as f64) as f32));
        let tallest = m.iter().flat_map(|s| s.iter()).fold(0.0f64, |a, &v| a.max(v));
        out.push((tallest * HEADROOM).max(1e-6) as f32);
        out
    }

    /// The linear layer's curves (log₁₀ of the share of each direction's loss left, at
    /// `LIN_POINTS` steps), as far as they have got.
    fn curves_goal(&self) -> Vec<f32> {
        let mut out = Vec::with_capacity(3 * LIN_N * LIN_POINTS);
        for run in &self.lin.runs {
            for j in 0..LIN_N {
                for p in 0..LIN_POINTS {
                    let t = p * LIN_STEPS / (LIN_POINTS - 1);
                    let c = &run.curves;
                    let v = c.get(t.min(c.len() - 1)).map_or(1.0, |c| c[j]);
                    out.push(lg(v, LIN_FLOOR).min(LIN_TOP) as f32);
                }
            }
        }
        out
    }

    /// Bring the gliding numbers `dt` seconds nearer to where they are going.
    fn follow(&mut self, dt: f32, snap: bool) {
        let (b, m, c) = (self.bars_goal(), self.moved_goal(), self.curves_goal());
        if snap {
            self.bars.snap(&b);
            self.moved.snap(&m);
            self.curves.snap(&c);
            self.gliding = false;
            return;
        }
        let mut moving = self.bars.follow(&b, dt, false);
        moving |= self.moved.follow(&m, dt, false);
        moving |= self.curves.follow(&c, dt, false);
        self.gliding = moving;
    }
}

/// How far the chart of the inputs reaches above the step's spectral norm, for its key.
const HEADROOM: f64 = 1.45;

/// Points per curve of the linear layer, and its chart's floor and top (log₁₀; the
/// highest any run reaches is 4.3 times its start, 10^0.63).
const LIN_POINTS: usize = 101;
const LIN_FLOOR: f64 = -8.0;
const LIN_TOP: f64 = 0.8;

impl MuonScene {
    fn buttons(&mut self, f: &mut Frame, lay: &Layout) -> bool {
        let size = lay.small;
        let mut changed = false;
        if !lay.compact {
            for (k, layer) in Layer::ALL.iter().enumerate() {
                let (r, c) = layer.shape();
                let name = match layer {
                    Layer::Readout => "read-out".to_string(),
                    _ => format!("layer {}", k + 1),
                };
                let word = if lay.narrow {
                    name
                } else {
                    let who = if layer.muons() { "Muon" } else { "Adam" };
                    format!(r"{name}: ${r} \times {c}$, {who}")
                };
                if plot::button(f, lay.layers[k], &word, size, self.look.setup.layer == *layer) && self.look.setup.layer != *layer {
                    self.look.setup.layer = *layer;
                    changed = true;
                }
            }
        }
        // the film
        let playing = self.film.playing;
        let last = self.look.setup.stages() - 1;
        let word = if playing {
            "❚❚ pause"
        } else if self.film.stage >= last {
            "▶ again"
        } else {
            "▶ play"
        };
        if plot::button(f, lay.play, word, size, false) {
            if playing {
                self.film.playing = false;
            } else {
                if self.film.stage >= last {
                    self.film.stage = 0;
                }
                self.film.playing = true;
                self.film.clock = 0.0;
            }
        }
        let names = self.stage_names();
        for (k, rect) in plot::even_row(names.len(), lay.stages.0.x, lay.stages.1.x, lay.stages.0.y, lay.stages.1.y - lay.stages.0.y, 4.0).into_iter().enumerate() {
            if plot::button(f, rect, &names[k], size, self.film.stage == k) {
                self.film.stage = k;
                self.film.playing = false;
            }
        }
        // a narrow window's tabs
        if lay.narrow && !lay.compact {
            for (k, (word, focus)) in [("polynomial", Focus::Poly), ("inputs", Focus::Inputs), ("training", Focus::Lower)].into_iter().enumerate() {
                if plot::button(f, lay.tabs[k], word, size, self.look.focus == focus) && self.look.focus != focus {
                    self.look.focus = focus;
                    changed = true;
                }
            }
        }
        // the lower chart's modes
        if lay.shows(lay.lower) {
            for (k, (word, lower)) in [("one linear layer", Lower::Linear), ("this network, trained", Lower::Training)].into_iter().enumerate() {
                if plot::button(f, lay.modes[k], word, size, self.look.lower == lower) && self.look.lower != lower {
                    self.look.lower = lower;
                    changed = true;
                }
            }
        }
        changed
    }

    fn spectrum_plot(&self, lay: &Layout) -> Plot {
        let k = self.snap.bars[0].len().max(1);
        let top = self.bars.get(k) as f64;
        Plot::new(lay.spectrum, [0.0, k as f64], [FLOOR, top.max(0.2)])
    }

    /// The spectrum: a bar for each singular value, on a log scale.
    fn draw_spectrum(&mut self, f: &mut Frame, lay: &Layout) {
        let p = self.spectrum_plot(lay);
        if p.tiny() {
            return;
        }
        let s = self.look.setup;
        let (min, max) = p.rect;
        let size = lay.small;
        p.backing(f, if lay.narrow { 34.0 } else { 44.0 }, 22.0, lay.text + 14.0, 8.0);
        let (rows, cols) = s.layer.shape();
        let stage = (self.film.shown.round() as usize).min(s.stages() - 1);
        let t = self.film.shown;
        let color = {
            let (a, b) = (stage_color(&s, t.floor() as usize), stage_color(&s, t.ceil() as usize));
            a.mix(b, t - t.floor())
        };
        let left = if lay.narrow { 30.0 } else { 40.0 };
        if lay.narrow {
            // (the title says what the stage is)
            let words = format!("{}: {}", s.layer.name(), self.stage_words(stage, true));
            f.text_px(Vec2::new(min.x - left, min.y - 8.0), words, lay.text, color, Align2::LEFT_BOTTOM);
        } else {
            let title = format!("the singular values of {}'s matrices ({rows} by {cols}), each over its matrix's Frobenius norm", s.layer.name());
            f.text_px(Vec2::new(min.x - left, min.y - 8.0), title, lay.text, pal::TEXT, Align2::LEFT_BOTTOM);
        }
        log_axis(f, &p, size, true, !lay.narrow);
        plot::line_of(f, AXES, Vec2::new(min.x, max.y), max, 1.0, pal::DIM.a(0.8));
        // the band and the line at 1
        if s.method == Method::Quintic {
            let (y0, y1) = (p.px_y(BAND[1].log10()), p.px_y(BAND[0].log10()));
            plot::fill(f, (Vec2::new(min.x, y0.max(min.y)), Vec2::new(max.x, y1.min(max.y))), BAND_COLOR.a(0.35));
        }
        let one = p.px_y(0.0);
        plot::dashed_of(f, MARKS, Vec2::new(min.x, one), Vec2::new(max.x, one), 1.0, pal::DIM.a(0.7));
        // the bars
        let k = self.snap.bars[0].len();
        let picked = match self.pick {
            Pick::Bar(i) if t >= 0.5 => Some(i),
            _ => None,
        };
        let gap = if p.width() / k.max(1) as f32 > 10.0 { 2.0 } else { 0.5 };
        for i in 0..k {
            let h = self.bars.get(i) as f64;
            let c = if picked == Some(i) { color.mix(pal::WHITE, 0.45) } else { color };
            if h <= FLOOR + 1e-6 {
                // a value at or below the floor: a stub
                p.bar(f, i as f64, (i + 1) as f64, FLOOR, FLOOR + 0.06 * (p.y[1] - FLOOR), gap, c.a(0.35));
            } else {
                p.bar(f, i as f64, (i + 1) as f64, FLOOR, h, gap, c.a(0.85));
            }
        }
        // once the rounds begin, U's values stay as pink ticks to compare with
        let faded = ((t - 1.0).clamp(0.0, 1.0) * 0.95).min(0.95);
        let stub = FLOOR + 0.06 * (p.y[1] - FLOOR);
        for (i, &u) in self.snap.bars[1].iter().enumerate() {
            let h = if u > 0.0 { lg(u, FLOOR).max(stub) } else { stub };
            let y = p.px_y(h.min(p.y[1]));
            let (x0, x1) = (p.px_x(i as f64) + gap, p.px_x((i + 1) as f64) - gap);
            plot::line_of(f, MARKS, Vec2::new(x0, y), Vec2::new(x1.max(x0), y), 2.0, MOMENT.a(faded));
        }
        // what the stage is, and how many values are where (at the top right, where
        // the bars are the shortest)
        let bars = &self.snap.bars[stage.min(self.snap.bars.len() - 1)];
        let unit = if stage + 1 == s.stages() { s.layer.factor() } else { 1.0 };
        let zero = bars.iter().filter(|&&v| v == 0.0).count();
        let mut line = Vec::new();
        if stage >= 2 && s.method == Method::Quintic {
            let inside = bars.iter().filter(|&&v| (BAND[0] - 1e-4..=BAND[1] + 1e-4).contains(&(v / unit))).count();
            line.push(format!("{inside} of {k} in [0.68, 1.13]"));
        }
        if zero > 0 {
            line.push(format!("{zero} of {k} are 0"));
        }
        let mut y = min.y + 4.0;
        if !lay.narrow {
            f.caption_px(Vec2::new(max.x - 6.0, y), self.stage_words(stage, false), size, color, Align2::RIGHT_TOP);
            y += size + 7.0;
        }
        if !line.is_empty() {
            f.caption_px(Vec2::new(max.x - 6.0, y), line.join("; "), size, pal::TEXT, Align2::RIGHT_TOP);
        }
        let below = if lay.narrow { "each over its Frobenius norm, largest first" } else { "one bar a singular value, largest first; a stub: below a millionth, or 0" };
        f.text_px(Vec2::new(min.x, max.y + 6.0), below, size, pal::DIM, Align2::LEFT_TOP);
    }

    /// Clicks on the spectrum pick a bar.
    fn pick_bar(&mut self, f: &mut Frame, lay: &Layout) {
        let p = self.spectrum_plot(lay);
        if p.tiny() {
            return;
        }
        if let Some((x, _)) = p.under_pointer(f) {
            f.handles.claim_clicks();
            if f.ptr.clicked {
                let k = self.snap.bars[1].len();
                let i = (x.floor().max(0.0) as usize).min(k.saturating_sub(1));
                self.pick = Pick::Bar(i);
                f.ptr.clicked = false;
            }
        }
    }

    fn poly_plot(&self, lay: &Layout) -> Plot {
        Plot::new(lay.poly, [POLY_LO, POLY_HI], [POLY_LO, POLY_HI])
    }

    /// The polynomial, the diagonal, the band, and the staircase of the picked value.
    fn draw_poly(&mut self, f: &mut Frame, lay: &Layout) {
        let p = self.poly_plot(lay);
        if p.tiny() {
            return;
        }
        let s = self.look.setup;
        let (min, max) = p.rect;
        let size = lay.small;
        p.backing(f, if lay.narrow { 34.0 } else { 44.0 }, 30.0, lay.text + 14.0, 8.0);
        let title = if lay.narrow { "one round sends each value $s$ to $p(s)$" } else { "one round of the iteration sends each value $s$ to $p(s)$" };
        f.text_px(Vec2::new(min.x - if lay.narrow { 30.0 } else { 40.0 }, min.y - 8.0), title, lay.text, pal::TEXT, Align2::LEFT_BOTTOM);
        // axes: powers of ten both ways
        log_axis(f, &p, size, true, false);
        plot::line_of(f, AXES, Vec2::new(min.x, max.y), max, 1.0, pal::DIM.a(0.8));
        for d in decades(p.x[0], p.x[1], p.width()) {
            let x = p.px_x(d as f64);
            plot::line_of(f, AXES, Vec2::new(x, max.y), Vec2::new(x, max.y + 4.0), 1.0, pal::DIM.a(0.8));
            plot::line_of(f, AXES, Vec2::new(x, min.y), Vec2::new(x, max.y), 1.0, pal::DIM.a(0.1));
            f.text_px(Vec2::new(x, max.y + 5.0), power(d), size, pal::DIM, Align2::CENTER_TOP);
        }
        f.text_px(Vec2::new(max.x, max.y + 7.0 + size), "$s$, before the round", size, pal::DIM, Align2::RIGHT_TOP);
        // the band
        let (y0, y1) = (p.px_y(BAND[1].log10()), p.px_y(BAND[0].log10()));
        plot::fill(f, (Vec2::new(min.x, y0), Vec2::new(max.x, y1)), BAND_COLOR.a(0.35));
        // the diagonal
        p.path(f, AXES, [(POLY_LO, POLY_LO), (POLY_HI, POLY_HI)], 1.0, pal::DIM.a(0.8));
        // the curves: the other polynomial faint, this one bright
        let curve = |poly: Odd5| (0..=160).map(move |i| {
            let x = POLY_LO + (POLY_HI - POLY_LO) * i as f64 / 160.0;
            let y = poly.at(10f64.powf(x));
            (x, if y > 0.0 { y.log10() } else { f64::NAN })
        });
        let other = if s.method == Method::Cubic { Odd5::JORDAN } else { Odd5::CUBIC };
        let other_color = if s.method == Method::Cubic { MUON } else { CUBIC };
        if other == Odd5::CUBIC {
            dashed_path(f, &p, CURVES, curve(other), 1.3, other_color.a(0.7));
        } else {
            p.path(f, CURVES, curve(other), 1.2, other_color.a(0.45));
        }
        match s.method.poly() {
            Some(poly) if poly == Odd5::CUBIC => dashed_path(f, &p, CURVES, curve(poly), 2.2, s.method.color()),
            Some(poly) => p.path(f, CURVES, curve(poly), 2.2, s.method.color()),
            None => p.path(f, CURVES, [(POLY_LO, 0.0), (POLY_HI, 0.0)], 2.2, EXACT),
        }
        // the curves' names, in the empty triangle under the diagonal
        let (name, oname) = match s.method {
            Method::Quintic => ("$p$: the quintic, slope 3.4445 at 0", "the cubic, slope 1.5"),
            Method::Cubic => ("$p$: the cubic, slope 1.5 at 0", "the quintic, slope 3.4445"),
            Method::Exact => ("exact: every value not 0 to 1", "the cubic"),
        };
        let x = max.x - 6.0;
        let mut y = max.y - 6.0;
        for (text, c) in [("the diagonal: $p(s) = s$", pal::DIM), (oname, other_color.a(0.8)), (name, s.method.color())] {
            f.caption_px(Vec2::new(x, y), text, size, c, Align2::RIGHT_BOTTOM);
            y -= size + 7.0;
        }
        // the staircase of the picked value, as far as the film has got
        let rounds_shown = (self.film.shown - 1.0).clamp(0.0, s.iterations() as f32);
        let s0 = self.pick_start();
        let mut pts: Vec<(f64, f64)> = Vec::new();
        let mut values = vec![s0];
        let step = |x: f64| match s.method.poly() {
            Some(poly) => poly.at(x),
            None => {
                if x > 0.0 {
                    1.0
                } else {
                    0.0
                }
            }
        };
        if s0 > 0.0 && self.film.shown >= 0.5 {
            let mut x = s0;
            pts.push((lg(x, POLY_LO), lg(x, POLY_LO)));
            let whole = rounds_shown.floor() as usize;
            for r in 0..=whole {
                let y = step(x);
                let part = if r < whole { 1.0 } else { (rounds_shown - whole as f32) as f64 };
                if part <= 0.0 {
                    break;
                }
                // up (or down) to the curve, then across to the diagonal
                let (lx, ly) = (lg(x, POLY_LO), lg(y, POLY_LO));
                let up = (part * 2.0).min(1.0);
                pts.push((lx, lx + (ly - lx) * up));
                if part > 0.5 {
                    let across = (part - 0.5) * 2.0;
                    pts.push((lx + (ly - lx) * across, ly));
                }
                if r < whole {
                    values.push(y);
                }
                x = y;
            }
        }
        let c = s.method.color().mix(pal::WHITE, 0.45);
        p.path(f, STAIR, pts.iter().copied(), 1.6, c);
        if let Some(&(x, y)) = pts.first() {
            p.dot(f, x, y, 6.0, c);
        }
        if let Some(&(x, y)) = pts.last() {
            p.dot(f, x, y, 4.0, c);
        }
        // the values it passes through
        if !values.is_empty() && self.film.shown >= 0.5 {
            let list: Vec<String> = values.iter().map(|&v| sig(v)).collect();
            let text = format!("${}$", list.join(r" \to "));
            f.caption_px(Vec2::new(min.x + 6.0, min.y + 4.0), text, size, c, Align2::LEFT_TOP);
        }
        // the start can be dragged along the diagonal
        if !lay.compact {
            let x0 = lg(s0.max(10f64.powf(POLY_LO)), POLY_LO);
            if let Some((x, _)) = p.handle(f, 1, x0, x0, c) {
                if x.is_finite() {
                    self.pick = Pick::Free(10f64.powf(x.clamp(POLY_LO, 0.0)));
                }
            }
        }
    }

    fn draw_inputs(&mut self, f: &mut Frame, lay: &Layout) {
        let s = self.look.setup;
        let k = self.snap.reach.len().max(1);
        let size = lay.small;
        let left = if lay.narrow { 34.0 } else { 44.0 };
        // the reach strip (log)
        let rp = Plot::new(lay.reach, [0.0, k as f64], [-6.0, 0.5]);
        if !rp.tiny() {
            let (min, max) = rp.rect;
            rp.backing(f, left, 4.0, lay.text + 14.0, 8.0);
            let title = if lay.narrow {
                r"how far a step moves the output, per unit input ($\times\eta$)"
            } else {
                "how far a step moves the layer's output, for a unit input along each principal direction of its inputs"
            };
            f.text_px(Vec2::new(min.x - left + 4.0, min.y - 8.0), title, lay.text, pal::TEXT, Align2::LEFT_BOTTOM);
            for (i, &v) in self.snap.reach.iter().enumerate() {
                rp.bar(f, i as f64, (i + 1) as f64, -6.0, lg(v, -6.0), 1.0, REACH.a(0.6));
            }
            plot::line_of(f, AXES, Vec2::new(min.x, max.y), max, 1.0, pal::DIM.a(0.6));
            f.text_px(Vec2::new(min.x - 6.0, (min.y + max.y) * 0.5), "reach", size, REACH, Align2::RIGHT_CENTER);
            let words = if lay.narrow { "the inputs' reach (log)" } else { "how far the inputs reach along it (log, 1 to a millionth)" };
            f.caption_px(Vec2::new(max.x - 4.0, min.y + 2.0), words, size, REACH, Align2::RIGHT_TOP);
        }
        let top = self.moved.get(2 * k) as f64;
        let p = Plot::new(lay.inputs, [0.0, k as f64], [0.0, top.max(1e-6)]);
        if p.tiny() {
            return;
        }
        let (min, max) = p.rect;
        p.backing(f, left, 30.0, 6.0, 8.0);
        p.y_axis(f, "", size);
        plot::line_of(f, AXES, Vec2::new(min.x, max.y), max, 1.0, pal::DIM.a(0.8));
        let t = self.film.shown;
        let color = {
            let (a, b) = (stage_color(&s, t.floor() as usize), stage_color(&s, t.ceil() as usize));
            a.mix(b, t - t.floor())
        };
        for i in 0..k {
            let (g, m) = (self.moved.get(i) as f64, self.moved.get(k + i) as f64);
            let x = i as f64;
            p.bar(f, x + 0.08, x + 0.5, 0.0, g, 0.5, GRAD.a(0.85));
            p.bar(f, x + 0.5, x + 0.92, 0.0, m, 0.5, color.a(0.85));
        }
        // the bound: the step's spectral norm
        let bound = top / HEADROOM;
        let y = p.px_y(bound);
        plot::dashed_of(f, MARKS, Vec2::new(min.x, y), Vec2::new(max.x, y), 1.2, pal::DIM);
        let names = if lay.narrow {
            ["dashed: the most any unit input can move", "cyan: a gradient step, at the same spectral norm", "the film's matrix; at its end, Muon's step"]
        } else {
            ["dashed: the most any unit input can move, the step's spectral norm", "cyan: a gradient step, scaled to the same spectral norm", "the film's stage at that norm; at its end, Muon's step"]
        };
        let mut at = min.y + 4.0;
        for (text, c) in [(names[0], pal::DIM), (names[1], GRAD), (names[2], color)] {
            f.caption_px(Vec2::new(max.x - 4.0, at), text, size, c, Align2::RIGHT_TOP);
            at += size + 6.0;
        }
        f.text_px(Vec2::new(min.x, max.y + 6.0), "the inputs' principal directions, the widest first", size, pal::DIM, Align2::LEFT_TOP);
        if !lay.narrow {
            f.text_px(Vec2::new(min.x - left + 4.0, min.y - 2.0), r"$\|\Delta W v\| / \eta$", size, pal::DIM, Align2::LEFT_BOTTOM);
        }
    }

    fn draw_lower(&mut self, f: &mut Frame, lay: &Layout) {
        match self.look.lower {
            Lower::Linear => self.draw_linear(f, lay),
            Lower::Training => self.draw_training(f, lay),
        }
    }

    fn draw_linear(&mut self, f: &mut Frame, lay: &Layout) {
        let (min, max) = lay.lower;
        let size = lay.small;
        let ratio = ratio_math(self.lin.ratio);
        let title = if lay.narrow {
            format!("one linear layer, inputs of variance 1 to ${{1/{ratio}}}$ along 8 directions: the loss along each, over its start")
        } else {
            format!("one linear layer, its inputs of variance 1 down to ${{1/{ratio}}}$ along 8 directions: the loss along each direction, over its start")
        };
        let left = if lay.narrow { 34.0 } else { 40.0 };
        let gap = 10.0;
        let w = ((max.x - min.x - left - 2.0 * gap) / 3.0).max(1.0);
        if max.y - min.y < 120.0 || w < 30.0 {
            return;
        }
        hud_panel(f, min, max);
        let used = wrap_px(f, Vec2::new(min.x + 6.0, min.y + 6.0), max.x - min.x - 12.0, &title, lay.text, pal::TEXT);
        let top = min.y + 12.0 + used + if lay.narrow { 2.0 * size + 12.0 } else { size + 8.0 };
        let bottom = max.y - 34.0;
        if bottom - top < 30.0 {
            return;
        }
        for (k, run) in self.lin.runs.iter().enumerate() {
            let x0 = min.x + left + k as f32 * (w + gap);
            let p = Plot::new((Vec2::new(x0, top), Vec2::new(x0 + w, bottom)), [0.0, LIN_STEPS as f64], [LIN_FLOOR, LIN_TOP]);
            let (pmin, pmax) = p.rect;
            if k == 0 {
                log_axis(f, &p, size, true, false);
            } else {
                for d in decades(p.y[0], p.y[1], p.height()) {
                    let y = p.px_y(d as f64);
                    plot::line_of(f, AXES, Vec2::new(pmin.x, y), Vec2::new(pmax.x, y), 1.0, pal::DIM.a(0.1));
                }
                plot::line_of(f, AXES, pmin, Vec2::new(pmin.x, pmax.y), 1.0, pal::DIM.a(0.5));
            }
            plot::line_of(f, AXES, Vec2::new(pmin.x, pmax.y), pmax, 1.0, pal::DIM.a(0.8));
            for t in [0usize, 100, 200] {
                let x = p.px_x(t as f64);
                plot::line_of(f, AXES, Vec2::new(x, pmax.y), Vec2::new(x, pmax.y + 4.0), 1.0, pal::DIM.a(0.8));
                f.text_px(Vec2::new(x, pmax.y + 5.0), t.to_string(), size, pal::DIM, if t == 0 { Align2::LEFT_TOP } else if t == 200 { Align2::RIGHT_TOP } else { Align2::CENTER_TOP });
            }
            let color = match run.rule {
                Rule::Sgd => SGD,
                Rule::Adam => ADAM,
                _ => MUON,
            };
            let name = match run.rule {
                Rule::Sgd => "gradient descent",
                Rule::Adam => "Adam",
                _ => "Muon",
            };
            if lay.narrow {
                // (a narrow chart: the name over its step)
                f.text_px(Vec2::new(pmin.x, pmin.y - 6.0 - size), name, size, color, Align2::LEFT_BOTTOM);
                f.text_px(Vec2::new(pmin.x, pmin.y - 4.0), format!(r"$\eta = {}$", rate(run.lr)), size, color, Align2::LEFT_BOTTOM);
            } else {
                f.text_px(Vec2::new(pmin.x, pmin.y - 4.0), format!(r"{name}, $\eta = {}$", rate(run.lr)), size, color, Align2::LEFT_BOTTOM);
            }
            for j in 0..LIN_N {
                let alpha = 1.0 - 0.75 * j as f32 / (LIN_N - 1) as f32;
                let base = (k * LIN_N + j) * LIN_POINTS;
                let pts = (0..LIN_POINTS).map(|i| (i as f64 * LIN_STEPS as f64 / (LIN_POINTS - 1) as f64, self.curves.get(base + i) as f64));
                p.path(f, LINEAR + k as u8, pts, 1.4, color.a(alpha));
            }
        }
        let words = if lay.narrow { "step · bright: the widest direction" } else { "step  ·  bright: the widest direction, faint: the narrowest" };
        f.text_px(Vec2::new(max.x - 4.0, max.y - 4.0), words, size, pal::DIM, Align2::RIGHT_BOTTOM);
    }

    fn draw_training(&mut self, f: &mut Frame, lay: &Layout) {
        let (min, max) = lay.lower;
        let size = lay.small;
        let left = if lay.narrow { 34.0 } else { 44.0 };
        if max.y - min.y < 120.0 || max.x - min.x < 120.0 {
            return;
        }
        hud_panel(f, min, max);
        let title = format!("this network trained from {} starts, each method on the same batches: the loss on all the data", RACE_SEEDS.len());
        let used = wrap_px(f, Vec2::new(min.x + 6.0, min.y + 6.0), max.x - min.x - 12.0, &title, lay.text, pal::TEXT);
        let p = Plot::new((Vec2::new(min.x + left, min.y + 18.0 + used), Vec2::new(max.x - 10.0, max.y - 34.0)), [0.0, RACE_STEPS as f64], [-3.0, 0.3]);
        if p.tiny() {
            return;
        }
        let (pmin, pmax) = p.rect;
        log_axis(f, &p, size, true, false);
        p.x_axis(f, "step", size);
        for (k, run) in self.race.runs.iter().enumerate() {
            let pts = run.losses.iter().enumerate().map(|(i, &l)| ((i * RACE_EVERY) as f64, lg(l, -3.0)));
            if run.method == Method::Cubic {
                dashed_path(f, &p, RACE + k as u8, pts, 1.6, run.method.color());
            } else {
                p.path(f, RACE + k as u8, pts, if run.method == Method::Quintic { 1.8 } else { 1.5 }, run.method.color().a(0.95));
            }
        }
        let mut y = pmin.y + 6.0;
        for m in [Method::Quintic, Method::Exact, Method::Cubic] {
            let word = match m {
                Method::Quintic => "five rounds of the reference's quintic",
                Method::Exact => "the exact polar factor",
                Method::Cubic => "five rounds of the classical cubic",
            };
            f.caption_px(Vec2::new(pmax.x - 6.0, y), word, size, m.color(), Align2::RIGHT_TOP);
            y += size + 6.0;
        }
        if !self.race.done() {
            f.caption_px(Vec2::new(pmin.x + 6.0, pmax.y - 6.0), format!("training: step {}", self.race.runs.first().map_or(0, |r| r.t)), size, pal::DIM, Align2::LEFT_BOTTOM);
        }
    }
}


impl MuonScene {
    /// The read-outs.
    fn numbers(&self, ui: &mut egui::Ui) {
        let sn = &self.snap;
        let s = sn.setup;
        let (rows, cols) = s.layer.shape();
        ui::readout(ui, "the network's loss on all the data", format!("{} at the start, {} after {TRAINED} steps", sig(self.net.loss_start), sig(self.net.loss)));
        ui::readout(ui, "the layer", format!("{} by {}, {} in the reference", rows, cols, if s.layer.muons() { "Muon's" } else { "Adam's" }));
        ui::readout(ui, r"rank of $G$, of $\tilde M$", format!("{}, {} (of {})", sn.rank_g, sn.rank, rows.min(cols)));
        ui::readout(ui, r"stable rank of $G$, of $\tilde M$", format!("{}, {}", sig(optim::stable_rank(&sn.g, rows, cols)), sig(optim::stable_rank(&sn.u, rows, cols))));
        if s.method == Method::Quintic {
            ui::readout(ui, "after the rounds, in $[0.68, 1.13]$", format!("{} of {}", sn.in_band(), rows.min(cols)));
        }
        if let Some(p) = s.method.poly() {
            let least = least_start(p, s.rounds, 0.68);
            ui::readout(ui, &format!("the least start that reaches 0.68 in {} rounds", s.rounds), least.map_or("none in [10⁻⁹, 1]".to_string(), sig_text));
        }
        let step = sn.step();
        ui::readout(ui, r"the step's spectral norm", format!(r"${}\,\eta$", sig(top(step))));
        let exact_rms = s.layer.factor() * (sn.rank as f64 / (rows * cols) as f64).sqrt();
        ui::readout(ui, r"the root mean square of its entries", format!(r"${}\,\eta$ (exact factor: ${}\,\eta$)", sig(sn.step_rms()), sig(exact_rms)));
        let (fr, fp) = flops(rows, cols, s.iterations(), s.batch);
        if s.method != Method::Exact {
            ui::readout(ui, "the rounds' arithmetic over the layer's passes", sig(fr / fp));
        }
        if self.cost.gradient > 0.0 {
            ui::readout(ui, "measured now: five rounds of the quintic", format!("{} µs", sig(self.cost.rounds)));
            ui::readout(ui, "  the exact polar factor", format!("{} µs", sig(self.cost.exact)));
            ui::readout(ui, "  the whole network's gradient on the batch", format!("{} µs", sig(self.cost.gradient)));
        }
        if self.lin.done() {
            for (k, run) in self.lin.runs.iter().enumerate() {
                let times: Vec<Option<usize>> = (0..LIN_N).map(|j| self.lin.below_from(k, j, 0.1)).collect();
                let reached: Vec<usize> = times.iter().flatten().copied().collect();
                let text = if reached.len() == LIN_N {
                    format!("steps {} to {}", reached.iter().min().copied().unwrap_or(0), reached.iter().max().copied().unwrap_or(0))
                } else {
                    format!("{} of {} within {}", reached.len(), LIN_N, LIN_STEPS)
                };
                ui::readout(ui, &format!(r"{}, $\eta = {}$: each direction below a tenth", run.rule.name(), rate(run.lr)), text);
                ui::readout(ui, "  the loss after 200 steps, over its start", sig_text(run.total()));
            }
        }
        if self.race.started() {
            let t = self.race.runs.first().map_or(0, |r| r.t);
            for m in [Method::Quintic, Method::Exact, Method::Cubic] {
                let at = self.race.at(m, t - t % RACE_EVERY);
                let list: Vec<String> = at.iter().map(|&v| sig(v)).collect();
                ui::readout(ui, &format!("{} at step {}", m.name(), t - t % RACE_EVERY), list.join(", "));
            }
        }
    }
}

/// The polynomial's chart reaches from 10⁻⁴ to 1.6 both ways (log₁₀).
const POLY_LO: f64 = -4.0;
const POLY_HI: f64 = 0.204_119_982_655_924_8;

impl Scene for MuonScene {
    fn view(&self) -> View {
        View::Flat { center: Vec2::ZERO, half_height: 1.0 }
    }

    fn settings(&self, s: &mut ViewSettings) {
        s.grid = false;
        s.bloom = s.bloom.min(0.2);
    }

    fn animated(&self) -> bool {
        self.film.moving() || self.gliding || self.busy()
    }

    fn pointer_hint(&self) -> Option<&'static str> {
        Some("bars: click one to follow it    dot on the curve's chart: drag    buttons: click")
    }

    fn ui(&mut self, ui: &mut egui::Ui, _: &Gfx) {
        ui::note(
            ui,
            r"Muon is a rule for a network's weight matrices (Jordan and co-authors, 2024). Each step it takes the usual momentum of the gradient, replaces that matrix by one with the same singular vectors and every singular value pushed towards 1, and steps along that. As its reference code has it, for a weight matrix $W$ with gradient $G$ (rows: the units it feeds, the fan-out; columns: its inputs, the fan-in):",
        );
        ui::equation(ui, r"M \leftarrow \beta M + (1 - \beta)\,G, \quad \tilde M = (1 - \beta)\,G + \beta M");
        ui::equation(ui, r"X_0 = \frac{\tilde M}{\|\tilde M\|_F + 10^{-7}}, \quad X_{r+1} = aX_r + b\,(X_r X_r^\top) X_r + c\,(X_r X_r^\top)^2 X_r");
        ui::equation(ui, r"W \leftarrow W - \eta \sqrt{\max(1, \text{fan-out}/\text{fan-in})}\; X_5");
        ui::note(
            ui,
            r"with $\beta = 0.95$ and $(a, b, c) = (3.4445, -4.7750, 2.0315)$. Dividing by the Frobenius norm puts every singular value in $[0, 1]$. A round then applies $p(s) = as + bs^3 + cs^5$ to every singular value and keeps the singular vectors, since for $X = USV^\top$ the odd powers $(XX^\top)^k X = US^{2k+1}V^\top$ share them (scene svd). The reference gives Muon only the matrices of hidden layers: 'scalar and vector parameters of the network, as well as the input and output layers, should be optimized by a standard method such as AdamW'. For the embedding its authors say this 'follows from the modular norm theory'; for the output layer it 'is instead driven by empirics'. Here every part is shown on a small network (2 inputs, three tanh layers of 16, 3 classes of points on spirals) after 300 steps of training with Muon.",
        );

        ui::section(ui, "Things to try");
        for (k, t) in TRIALS.iter().enumerate() {
            if ui::experiment(ui, t.button, t.caption) {
                self.set_up(k);
            }
        }

        ui::section(ui, "The layer and its gradient");
        let mut layer = self.look.setup.layer.index();
        if ui::choice(ui, &["layer 1", "layer 2", "layer 3", "read-out"], &mut layer) {
            self.look.setup.layer = Layer::ALL[layer.min(3)];
        }
        ui::note(
            ui,
            r"Layers 2 and 3 (16 by 16) are the hidden matrices the reference gives Muon. Layer 1 (16 by 2, which reads the data) and the read-out (3 by 16) it gives Adam; they are shown here as Muon would treat them. The read-out's gradient has rank 2 at most whatever the batch: the error it gets from the loss, $P - Y$, adds up to 0 along each example's row.",
        );
        ui::slider_int_log(ui, "examples in the gradient $B$", &mut self.look.setup.batch, 1..=MOST_BATCH);
        ui::note(ui, r"A batch's gradient is a sum of one outer product per example, so its rank is at most $B$ (scene backpropmat). The network was trained with batches of 32.");
        ui::checkbox(ui, &mut self.look.setup.fresh, "the momentum empty: the first step");

        ui::section(ui, "The iteration");
        let mut method = Method::ALL.iter().position(|&m| m == self.look.setup.method).unwrap_or(0);
        if ui::choice(ui, &["quintic", "cubic", "exact"], &mut method) {
            self.look.setup.method = Method::ALL[method.min(2)];
        }
        if self.look.setup.method != Method::Exact {
            ui::slider_int(ui, "rounds", &mut self.look.setup.rounds, 1..=MOST_ROUNDS);
        }
        ui::note(
            ui,
            r"The reference's quintic, the classical cubic $1.5s - 0.5s^3$, or the exact polar factor $UV^\top$, found here by plane rotations: the reference runs five rounds in bfloat16, and every number here is in double precision. With the quintic no number of rounds converges: its two fixed points other than 0, 0.868 and 1.264, both repel, and after enough rounds every value above 0 lands in $[0.6818, 1.1344]$. After five rounds a start of at least 0.01 is in $[0.682, 1.134]$, one of at least 0.003 in $[0.682, 1.202]$.",
        );

        ui::section(ui, "One linear layer");
        let names: Vec<String> = (0..RATIOS.len()).map(ratio_name).collect();
        let labels: Vec<&str> = names.iter().map(|s| s.as_str()).collect();
        ui::choice(ui, &labels, &mut self.look.ratio);
        ui::note(ui, "The ratio of the largest variance of the inputs to the smallest.");
        let mut lined = !self.look.turned;
        if ui::checkbox(ui, &mut lined, "directions lined up with the coordinates").changed() {
            self.look.turned = !lined;
        }
        ui::note(
            ui,
            r"Turned or lined up is the same thing to gradient descent: each direction's part falls by $(1 - \eta\lambda_j)^2$ a step either way. Adam scales each entry of the gradient by its own running size, and here, lined up at 1000 to 1, it gets every direction below a tenth by step 14. Each rule takes the step size from $10^{-4}$ to 6.3 (25 of them, evenly on a log scale) whose loss is lowest after 200 steps; Adam with Kingma and Ba's defaults, Muon the reference's.",
        );

        ui::section(ui, "Why a step like this");
        ui::note(
            ui,
            r"The mathematics. Measure a matrix by how much it can stretch a vector, its spectral norm $\|\Delta W\|$, the largest singular value. Of all steps with $\|\Delta W\| \le \eta$, the one that lowers the loss most to first order (the least $\langle G, \Delta W\rangle$) is $-\eta UV^\top$, for $G = USV^\top$ of full rank: steepest descent under the spectral norm, in its form with a fixed length (scene steepest; Bernstein, 2025; Kovalev, 2025). With the size added as a penalty $\tfrac{\lambda}{2}\|\Delta W\|^2$ instead, the step is $-(\|G\|_{\mathrm{nuc}}/\lambda)\,UV^\top$, its length the sum of the singular values over $\lambda$ (Bernstein and Newhouse, 2024). Muon takes the fixed length, and applies it to the momentum. A step of spectral norm $\eta$ moves no input's output by more than $\eta$ times the input's length, $\|\Delta W x\| \le \eta \|x\|$. Here (the chart of the inputs, layer 2) Muon's step moves the output by at least $0.2\eta$ for a unit input along each of the inputs' 16 principal directions; a gradient step of the same spectral norm, by less than $0.04\eta$ along 13 of them.",
        );
        ui::note(
            ui,
            r"Its authors' second reason is a speculation from looking at updates: those of SGD with momentum and of Adam for a transformer's matrices 'typically have very high condition number. That is, they are almost low-rank matrices', and they 'speculate that orthogonalization effectively increases the scale of other “rare directions”' (Jordan and co-authors, 2024). Here layer 2's gradient has three large singular values and thirteen below 0.011 of its norm; the momentum mix $\tilde M$ that the rounds are given is more even, and five rounds give 11 of its 16 a size between 0.68 and 1.12.",
        );
        ui::note(
            ui,
            r"What its authors report: with Muon, CIFAR-10 to 94 % accuracy in 2.6 instead of 3.3 A100-seconds, every record of the NanoGPT speed-run (a race to train a small GPT-2) from October to December 2024, and a transformer of 1.5 billion parameters at GPT-2 XL's HellaSwag score in 10 hours on 8 H100s, where AdamW took 13.3, both tuned (Jordan and co-authors, 2024). A study that tuned every optimiser separately at 0.1 to 1.2 billion parameters found its speed-up over AdamW at most about 1.4 times, falling to about 1.1 at 1.2 billion (Wen and co-authors, 2025). More, and the explanations offered, in scene muonadam.",
        );

        ui::section(ui, "What it costs");
        let (rows, cols) = self.look.setup.layer.shape();
        let (m, b) = (rows.min(cols), self.look.setup.batch);
        ui::note(
            ui,
            &format!(
                r"Memory: one matrix a layer, the momentum, as for SGD with momentum. Arithmetic: its authors count $T$ rounds on an $n$ by $m$ matrix ($m \le n$) as at most $Tm/B$ times that layer's forward and backward pass on $B$ inputs (tokens), about 0.7 % for the NanoGPT record ($m = 768$, $B = 524\,288$) and 'below 1%' in typical language-model training; a count of operations, not of time (Jordan and co-authors, 2024). Here the batch is small: for this layer ($m = {m}$) and $B = {b}$, $Tm/B = {}$. The times measured now are under Numbers.",
                sig(self.look.setup.rounds as f64 * m as f64 / b as f64)
            ),
        );
        self.refresh();

        ui::section(ui, "Reading the picture");
        ui::key(
            ui,
            &[
                (GRAD, "cyan: the gradient $G$ of a batch, and a gradient step"),
                (MOMENT, r"pink: $\tilde M$, the momentum mixed with the gradient, which the iteration is given"),
                (MUON, "blue: after each round of the reference's quintic, and Muon's step"),
                (EXACT, r"light blue: the exact polar factor $UV^\top$"),
                (CUBIC, "grey: the classical cubic"),
                (ADAM, "orange: Adam, on the linear layer"),
                (BAND_COLOR.mix(pal::WHITE, 0.3), r"shaded: $[0.68, 1.13]$, where the quintic's values land"),
                (REACH, "dim bars: how far the inputs reach along each of their directions"),
            ],
        );

        ui::section(ui, "Words");
        ui::term(ui, "Singular values:", r"$\sigma_1 \ge \sigma_2 \ge \dots \ge 0$ in $G = USV^\top$: how much $G$ stretches each of a set of perpendicular directions (scene svd). Here each is divided by the matrix's Frobenius norm.");
        ui::term(ui, r"Frobenius norm $\|G\|_F$:", "the root of the sum of the squares of the entries, which is also that of the singular values.");
        ui::term(ui, r"Spectral norm $\|G\|$:", "the largest singular value: the most the matrix stretches any vector.");
        ui::term(
            ui,
            r"Polar factor $UV^\top$:",
            r"$G$ with every singular value set to 1 (scene polar). For $G$ of full rank it is the matrix with orthonormal rows (or columns) nearest to $G$, measured by the Frobenius norm (Bernstein and Newhouse, 2024). If $G$ has lost rank, $r < \min(\text{rows}, \text{columns})$, the nearest such matrix is not unique, and the iteration aims instead at $U_r V_r^\top$: $r$ singular values of 1 and the rest 0, so not itself orthogonal.",
        );
        ui::term(ui, "Rank:", "how many singular values are not 0.");
        ui::term(ui, "Stable rank:", r"$\|G\|_F^2 / \|G\|^2$: about how many singular values are near the largest.");
        ui::term(ui, r"Momentum, $\beta$:", r"$M \leftarrow \beta M + (1 - \beta)G$, a running average of the gradients; with $\beta = 0.95$ a gradient's share halves in about 14 steps.");
        ui::term(ui, "Nesterov's mix:", r"$\tilde M = (1 - \beta)G + \beta M$: the average, leaning further towards the newest gradient.");
        ui::term(ui, "Principal directions of the inputs:", r"the eigenvectors of the mean of $xx^\top$ over the data, for $x$ what the layer receives. How far the inputs reach along one: the root mean square of their component along it.");
        ui::term(ui, "Shape factor:", r"$\sqrt{\max(1, \text{fan-out}/\text{fan-in})}$, by which the reference multiplies the orthogonalised matrix.");
        ui::term(
            ui,
            r"Step size $\eta$:",
            "the learning rate. Here Muon's is 0.05 and Adam's, for the other parameters, 0.01. With the reference's own, 0.02 and 0.0003, this network's loss after 300 steps would be 1.0 instead of 0.44.",
        );

        ui::section(ui, "Numbers");
        self.numbers(ui);
        ui::see_also(ui, &[("polar", "the polar factor"), ("steepest", "steepest descent under a norm"), ("muonkin", "Muon's relatives"), ("muonadam", "Muon against Adam")]);
    }

    fn frame(&mut self, f: &mut Frame) {
        hold_batches(f);
        let lay = Layout::of(f.view_size(), f.top_inset, self.look.focus);
        let tiny = lay.view.x < 160.0 || lay.view.y < 120.0;
        if !tiny && self.buttons(f, &lay) {
            self.refresh();
        }
        if !tiny {
            self.pick_bar(f, &lay);
        }
        self.refresh();
        if self.cost.gradient == 0.0 {
            self.cost = Cost::measure(&self.net, &self.snap, &mut self.tape);
        }
        let budget = clock::frame_budget(f.real_dt, 0.004);
        self.work(&budget);
        self.film.advance(f.dt, self.look.setup.stages());
        let snap = self.canvas != Some(lay.view) && !tiny;
        if snap {
            self.canvas = Some(lay.view);
        }
        self.follow(f.real_dt, snap);
        if tiny {
            return;
        }
        self.draw_spectrum(f, &lay);
        if lay.compact {
            return;
        }
        if lay.shows(lay.poly) {
            self.draw_poly(f, &lay);
        }
        if lay.shows(lay.inputs) {
            self.draw_inputs(f, &lay);
        }
        if lay.shows(lay.lower) {
            self.draw_lower(f, &lay);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn close(a: f64, b: f64, rel: f64) -> bool {
        (a - b).abs() <= rel * b.abs().max(1e-300)
    }

    /// The caption of experiment k says `what`.
    fn says(k: usize, what: &str) {
        assert!(TRIALS[k].caption.contains(what), "experiment {k} should say \"{what}\": {}", TRIALS[k].caption);
    }

    /// A matrix with the given singular values (min(rows, cols) of them) and random
    /// singular vectors.
    fn with_singular_values(rows: usize, cols: usize, s: &[f64], rng: &mut Rng) -> Vec<f64> {
        let k = rows.min(cols);
        let mut u: Vec<f64> = (0..rows * k).map(|_| rng.normal() as f64).collect();
        crate::math::linalg::orthonormalise(&mut u, rows, k);
        let mut v: Vec<f64> = (0..cols * k).map(|_| rng.normal() as f64).collect();
        crate::math::linalg::orthonormalise(&mut v, cols, k);
        let mut a = vec![0.0; rows * cols];
        for i in 0..rows {
            for j in 0..cols {
                a[i * cols + j] = (0..k).map(|p| u[i * k + p] * s[p] * v[j * k + p]).sum();
            }
        }
        a
    }

    /// The singular values found from the polar factor are right to the rounding unit
    /// times the largest, far below where those from a Gram matrix stop (about 1e-8 of
    /// the largest), and a matrix that has lost rank gives exact zeros.
    #[test]
    fn accurate_singular_values_are_accurate() {
        let mut rng = Rng::new(9);
        for (rows, cols) in [(16, 16), (16, 2), (3, 16), (7, 5)] {
            let k = rows.min(cols);
            let want: Vec<f64> = (0..k).map(|i| 10f64.powf(-12.0 * i as f64 / (k - 1).max(1) as f64)).collect();
            let a = with_singular_values(rows, cols, &want, &mut rng);
            let got = accurate_singular_values(&a, rows, cols);
            for (g, w) in got.iter().zip(&want) {
                assert!((g - w).abs() < 1e-14, "{rows} by {cols}: {g} against {w}");
            }
        }
        // rank 3 of 16 by 16: three values, thirteen exact zeros
        let mut s = vec![0.0; 16];
        s[..3].copy_from_slice(&[2.0, 0.5, 1e-6]);
        let a = with_singular_values(16, 16, &s, &mut rng);
        let got = accurate_singular_values(&a, 16, 16);
        assert!(close(got[0], 2.0, 1e-12) && close(got[1], 0.5, 1e-12) && close(got[2], 1e-6, 1e-8), "{got:?}");
        assert!(got[3..].iter().all(|&v| v == 0.0), "{got:?}");
        assert_eq!(accurate_singular_values(&[0.0; 6], 2, 3), vec![0.0, 0.0]);
        assert!(accurate_singular_values(&[f64::NAN, 1.0, 0.0, 1.0], 2, 2).iter().all(|v| v.is_nan()));
    }

    /// The scene's momentum, its mix and its step are the library's Muon: for the layers
    /// the reference gives Muon, the matrix the optimiser hands to its iteration is U, and
    /// its step is −η times the last stage's matrix; the first step from an empty
    /// momentum hands it (1 − β)(1 + β) G.
    #[test]
    fn the_momentum_and_the_step_are_the_library_s() {
        let net = Net::new();
        let mut tape = Tape::new();
        // the reference gives Muon the hidden layers' matrices, and the rest to Adam
        for layer in Layer::ALL {
            let gi = net.model.index_of(layer.group()).unwrap();
            assert_eq!(net.opt.as_matrix(gi), layer.muons(), "{layer:?}");
            assert_eq!(net.opt.rule_of(gi), if layer.muons() { Rule::Muon } else { Rule::Adam });
        }
        for layer in [Layer::Second, Layer::Third] {
            for batch in [1, 7, BATCH, POINTS] {
                for fresh in [false, true] {
                    let sn = Snapshot::new(&net, Setup { layer, batch, fresh, ..Setup::base() }, &mut tape);
                    let mut model = net.model.clone();
                    let mut opt = if fresh { Optimizer::new(&model, Rule::Muon, hyper(Ortho::Polynomial(Iteration::JORDAN))) } else { net.opt.clone() };
                    net.problem.grad(&mut model, &mut tape, &net.order[..batch]);
                    opt.step(&mut model, LR);
                    let gi = model.index_of(layer.group()).unwrap();
                    let scale = top(opt.ortho_in(gi)).max(1e-300);
                    for (a, b) in opt.ortho_in(gi).iter().zip(&sn.u) {
                        assert!((a - b).abs() <= 1e-12 * scale, "{layer:?} {batch} {fresh}: U");
                    }
                    for (a, b) in opt.update(gi).iter().zip(sn.mats.last().unwrap()) {
                        assert!((a + LR * b).abs() <= 1e-12, "{layer:?} {batch} {fresh}: the step");
                    }
                    if fresh {
                        let c = (1.0 - BETA) * (1.0 + BETA);
                        assert!(sn.u.iter().zip(&sn.g).all(|(u, g)| (u - c * g).abs() <= 1e-15 * scale));
                    }
                }
            }
        }
    }

    /// Each stage's bars are what the matrix iteration does to the matrix: the singular
    /// values of each round's matrix are the polynomial applied to U's (sorted), and the
    /// exact factor's are 1 over U's rank.
    #[test]
    fn the_bars_are_the_matrix_iteration_s() {
        let net = Net::new();
        let mut tape = Tape::new();
        for method in Method::ALL {
            for layer in Layer::ALL {
                for (batch, rounds) in [(BATCH, 5), (3, 8), (POINTS, 1)] {
                    let sn = Snapshot::new(&net, Setup { layer, batch, method, rounds, fresh: false }, &mut tape);
                    let (r, c) = (sn.rows, sn.cols);
                    assert_eq!(sn.bars.len(), sn.setup.stages());
                    for k in 2..sn.bars.len() - 1 {
                        let mut want = sn.bars[k].clone();
                        want.sort_by(|a, b| b.total_cmp(a));
                        let got = accurate_singular_values(&sn.mats[k], r, c);
                        for (g, w) in got.iter().zip(&want) {
                            assert!((g - w).abs() <= 1e-9 * want[0].max(1e-300), "{method:?} {layer:?} {batch} round {k}: {got:?} against {want:?}");
                        }
                    }
                    // the step: the last round times the shape factor
                    let f = layer.factor();
                    let n = sn.bars.len();
                    assert!(sn.bars[n - 1].iter().zip(&sn.bars[n - 2]).all(|(a, b)| close(*a, f * b, 1e-14) || *a == 0.0));
                }
            }
        }
    }

    /// The numbers the panel's notes give.
    #[test]
    fn the_notes_say_what_is_measured() {
        let net = Net::new();
        let mut tape = Tape::new();
        // the reference's own rates learn slowly here: 1.0 against 0.44 after 300 steps
        let mut model = Model::mlp(&spec(), &mut Rng::new(MODEL_SEED));
        let mut opt = Optimizer::new(&model, Rule::Muon, Hyper::muon_reference());
        let mut batches = Sampler::new(net.problem.len(), BATCH, Draw::WithoutReplacement, Rng::new(BATCH_SEED));
        for _ in 0..TRAINED {
            let idx: Vec<usize> = batches.deal().to_vec();
            net.problem.grad(&mut model, &mut tape, &idx);
            opt.step(&mut model, Hyper::muon_reference().lr);
        }
        let slow = net.problem.loss(&model, &mut tape);
        assert_eq!((sig2(slow), sig2(net.loss)), ("1.0".to_string(), "0.44".to_string()), "{slow} {}", net.loss);
        // layer 2: three large values and thirteen below 0.011; five rounds give 11 a
        // size between 0.68 and 1.12
        let sn = Snapshot::new(&net, Setup::base(), &mut tape);
        assert!(sn.bars[0][2] > 0.1 && sn.bars[0][3] < 0.011, "{:?}", sn.bars[0]);
        assert_eq!(sn.in_band(), 11);
        let kept: Vec<f64> = sn.after().iter().copied().filter(|&v| v >= BAND[0] - 1e-4).collect();
        assert!(kept.len() == 11 && kept.iter().all(|&v| (0.68..=1.12).contains(&v)), "{kept:?}");
        // Muon's step moves the output by at least 0.2η along every principal direction;
        // a gradient step of the same norm by less than 0.04η along 13 of them
        let step = sn.moved.last().unwrap();
        assert!(step.iter().all(|&v| v >= 0.2), "{step:?}");
        assert_eq!(sn.moved[0].iter().filter(|&&v| v < 0.04).count(), 13, "{:?}", sn.moved[0]);
        // five rounds from at least 0.01 land in [0.682, 1.134], from 0.003 in [0.682, 1.202]
        for (from, lo, hi) in [(0.01, 0.682, 1.134), (0.003, 0.682, 1.202)] {
            let (mut a, mut b) = (f64::INFINITY, 0.0f64);
            for i in 0..=200_000 {
                let s0 = from + (1.0 - from) * i as f64 / 200_000.0;
                let v = rounds_of(Odd5::JORDAN, s0, 5);
                a = a.min(v);
                b = b.max(v);
            }
            assert!((a - lo).abs() < 5e-4 && (b - hi).abs() < 5e-4, "{from}: [{a}, {b}]");
        }
        // a gradient's share of the momentum halves in about 14 steps
        assert_eq!((0.5f64.ln() / BETA.ln()).round(), 14.0);
        // the read-out's gradient and momentum have rank 2 at most, whatever the batch
        for batch in [1, 2, 5, BATCH, POINTS] {
            let sn = Snapshot::new(&net, Setup { layer: Layer::Readout, batch, ..Setup::base() }, &mut tape);
            assert!(sn.rank_g <= 2 && sn.rank <= 2 && sn.rank_g == batch.min(2), "batch {batch}: {} {}", sn.rank_g, sn.rank);
        }
        // a batch of B gives rank at most B
        for batch in [1, 2, 4, 8, 15] {
            let sn = Snapshot::new(&net, Setup { batch, fresh: true, ..Setup::base() }, &mut tape);
            assert_eq!((sn.rank_g, sn.rank), (batch, batch));
        }
        // the quintic's fixed points other than 0 repel; it never leaves [0, 1.2024] from [0, 1]
        let p = Odd5::JORDAN;
        for x0 in [0.8680258, 1.26372907] {
            assert!((p.at(x0) - x0).abs() < 1e-6);
            let slope = (p.at(x0 + 1e-7) - p.at(x0 - 1e-7)) / 2e-7;
            assert!(slope.abs() > 1.0, "{slope}");
        }
        let most = (0..=100_000).map(|i| p.at(i as f64 / 100_000.0)).fold(0.0f64, f64::max);
        assert!((most - 1.2024).abs() < 1e-4);
        // the slopes at 0, and the growth in five rounds
        assert!(close(p.a, 3.4445, 1e-12) && close(Odd5::CUBIC.a, 1.5, 1e-12) && num(1.5f64.powi(5), 1) == "7.6");
        // lined up at 1000 to 1, Adam gets every direction below a tenth by step 14
        let mut lin = LinRace::new(3, false);
        lin.work(&Budget::new(100.0));
        let adam: Vec<usize> = (0..LIN_N).map(|j| lin.below_from(1, j, 0.1).unwrap()).collect();
        assert_eq!(adam.iter().max(), Some(&14), "{adam:?}");
        // the cost of five rounds on layer 2 against its passes on a batch of 32: Tm/B
        let (rounds, passes) = flops(16, 16, 5, 32);
        assert!(close(rounds / passes, 5.0 * 16.0 / 32.0, 1e-12));
        // (never above Tm/B, m the shorter side)
        for (r, c) in [(16, 2), (3, 16), (768, 3072)] {
            let (a, b) = flops(r, c, 5, 64);
            assert!(a / b <= 5.0 * r.min(c) as f64 / 64.0 + 1e-12);
        }
    }

    /// The least start that reaches 0.68 in five rounds: the course notes' 0.0015 for
    /// the quintic; 0.105 for the cubic.
    #[test]
    fn the_least_starts() {
        let q = least_start(Odd5::JORDAN, 5, 0.68).unwrap();
        assert!((q - 0.0015).abs() < 0.00005, "{q}");
        assert!(rounds_of(Odd5::JORDAN, q * 0.999, 5) < 0.68 && rounds_of(Odd5::JORDAN, q * 1.001, 5) >= 0.68);
        let c = least_start(Odd5::CUBIC, 5, 0.68).unwrap();
        assert!((c - 0.1045).abs() < 0.0005, "{c}");
        assert!(least_start(Odd5::JORDAN, 5, 1e-30).is_none());
    }

    /// Gradient descent on the linear layer shrinks the loss along each direction by
    /// exactly (1 − ηλ)² a step.
    #[test]
    fn gradient_descent_shrinks_each_direction_by_its_own_factor() {
        for (ratio, turned) in [(2, true), (3, false), (4, true)] {
            let mut lin = LinRace::new(ratio, turned);
            lin.work(&Budget::new(100.0));
            let run = &lin.runs[0];
            assert_eq!(run.rule, Rule::Sgd);
            for j in 0..LIN_N {
                let a = (1.0 - run.lr * lin.problem.lam[j]).powi(2);
                for t in [1usize, 5, 30, 200] {
                    let want = a.powi(t as i32);
                    // (until the error reaches the rounding of the weights)
                    if want > 1e-20 {
                        assert!(close(run.curves[t][j], want, 1e-8), "direction {j}, step {t}: {} against {want}", run.curves[t][j]);
                    }
                }
            }
        }
    }

    /// An experiment as its button sets it up, with its runs done and its film at the end.
    fn run(k: usize) -> MuonScene {
        let mut s = MuonScene::create();
        s.set_up(k);
        s.refresh();
        s.lin.work(&Budget::new(100.0));
        s.race.work(&s.net.clone(), &mut s.tape, &Budget::new(100.0));
        s.film = Film::at(s.look.setup.stages() - 1, false);
        s
    }

    /// The numbers each experiment's caption says, from the scene as its button sets it up.
    #[test]
    fn the_experiments_show_what_their_captions_say() {
        // the rounds one at a time
        let s = run(0);
        let sn = &s.snap;
        let (g, u) = (&sn.bars[0], &sn.bars[1]);
        says(0, &format!("three values stand out, {}, {} and {}", num(g[0], 2), num(g[1], 2), num(g[2], 2)));
        says(0, &format!("the rest fall from {} to ${}$", num(g[3], 3), sig_pow(g[15])));
        says(0, &format!("more even: {}, {}, {}, then {} down to ${}$", num(u[0], 2), num(u[1], 2), num(u[2], 2), num(u[3], 2), sig_pow(u[15])));
        says(0, &format!("After five rounds {} of the 16 are in $[0.68, 1.13]$", sn.in_band()));
        let least = least_start(Odd5::JORDAN, 5, 0.68).unwrap();
        let short: Vec<usize> = (0..16).filter(|&i| u[i] < least).collect();
        assert_eq!(short.len(), 5);
        assert!(short.iter().all(|&i| sn.after()[i] < BAND[0]) && (0..16).filter(|i| !short.contains(i)).all(|i| sn.after()[i] >= BAND[0] - 1e-4));
        let worst = short.iter().map(|&i| sn.after()[i]).fold(0.0f64, f64::max);
        says(0, &format!("the five that started below {} end at {} or less", sig2(least), sig2(worst + 0.005)));
        assert!(worst <= 0.39);
        let x0 = u[PICKED];
        // (two significant digits, and two decimals at or above 1)
        let path: Vec<String> = (1..=5).map(|r| rounds_of(Odd5::JORDAN, x0, r)).map(|v| if v >= 1.0 { num(v, 2) } else { sig2(v) }).collect();
        says(0, &format!("starts at {} and needs all five rounds: {}", sig2(x0), path.join(", ")));
        assert!(rounds_of(Odd5::JORDAN, x0, 4) < BAND[0] && rounds_of(Odd5::JORDAN, x0, 5) >= BAND[0]);
        // a gradient from one example
        let s = run(1);
        let sn = &s.snap;
        assert_eq!((sn.rank_g, sn.rank), (1, 1));
        assert!(sn.bars.iter().all(|b| b[1..].iter().all(|&v| v == 0.0)));
        let path: Vec<String> = (2..7).map(|k| num(sn.bars[k][0], 2)).collect();
        says(1, &format!("five rounds give {}", path.join(", ")));
        says(1, &format!(r"spectral norm ${}\eta$", num(top(sn.step()), 2)));
        assert!(sn.moved[0].iter().zip(sn.moved.last().unwrap()).all(|(a, b)| (a - b).abs() < 1e-9 * top(&sn.moved[0])));
        assert_eq!(sn.step().iter().filter(|&&v| v == 0.0).count(), 15);
        says(1, "15 of its singular values are 0");
        // the classical cubic
        let s = run(2);
        let after = s.snap.after();
        assert!(after[..3].iter().all(|v| num(*v, 3) == "1.000"));
        says(2, &format!("the fourth {}, and the rest stay below {}", sig2(after[3]), num(after[4] + 0.0005, 2)));
        assert!(after[4..].iter().all(|&v| v < 0.36));
        let x0 = s.snap.bars[1][PICKED];
        says(2, &format!("goes from {} only to {}", sig2(x0), sig2(rounds_of(Odd5::CUBIC, x0, 5))));
        // (a least start is rounded up, so that the number named does reach 0.68)
        let cubic = (least_start(Odd5::CUBIC, 5, 0.68).unwrap() * 1000.0).ceil() / 1000.0;
        says(2, &format!("needs a start of at least {cubic:.3}, the reference's quintic {}", sig2(least)));
        // the exact polar factor, in training
        let s = run(3);
        assert!(s.race.done());
        assert!(s.snap.after().iter().all(|&v| v == 1.0));
        let at = |m: Method, t: usize| s.race.at(m, t);
        let list = |v: Vec<f64>| format!("{}, {} and {}", sig2(v[0]), sig2(v[1]), sig2(v[2]));
        says(3, &format!("at step 300 the losses are {} with the exact factor, {} with five rounds of the quintic, {} with five of the cubic", list(at(Method::Exact, 300)), list(at(Method::Quintic, 300)), list(at(Method::Cubic, 300))));
        for t in (100..=400).step_by(RACE_EVERY) {
            let (e, q, c) = (at(Method::Exact, t), at(Method::Quintic, t), at(Method::Cubic, t));
            for i in 0..3 {
                assert!(e[i] < q[i] && q[i] < c[i], "step {t}, start {i}: {} {} {}", e[i], q[i], c[i]);
            }
        }
        let end: Vec<f64> = [Method::Quintic, Method::Exact, Method::Cubic].iter().flat_map(|&m| at(m, RACE_STEPS)).collect();
        let (lo, hi) = end.iter().fold((f64::INFINITY, 0.0f64), |(a, b), &v| (a.min(v), b.max(v)));
        says(3, &format!("By step 600 all nine runs are between {} and {}", sig2(lo), sig2(hi)));
        let best: Vec<Method> = (0..3)
            .map(|i| *[Method::Quintic, Method::Exact, Method::Cubic].iter().min_by(|&&a, &&b| at(a, RACE_STEPS)[i].total_cmp(&at(b, RACE_STEPS)[i])).unwrap())
            .collect();
        assert!(best.iter().any(|&m| m != best[0]), "{best:?}");
        // inputs spread a thousand to one
        let s = run(4);
        let lin = &s.lin;
        assert_eq!((lin.ratio, lin.turned), (3, true));
        let times = |k: usize| (0..LIN_N).map(|j| lin.below_from(k, j, 0.1)).collect::<Vec<_>>();
        let (gd, adam, muon) = (times(0), times(1), times(2));
        says(4, &format!("gets the widest direction below a tenth of its start in {} steps", gd[0].unwrap()));
        let last = lin.runs[0].curves[LIN_STEPS];
        says(4, &format!("leaves the two narrowest at {} and {} after 200", sig2(last[6]), sig2(last[7])));
        assert!(gd[6].is_none() && gd[7].is_none() && gd[..6].iter().all(|t| t.is_some()));
        let span = |v: &[Option<usize>]| {
            let w: Vec<usize> = v.iter().map(|t| t.unwrap()).collect();
            (*w.iter().min().unwrap(), *w.iter().max().unwrap())
        };
        let (a0, a1) = span(&adam);
        let (m0, m1) = span(&muon);
        says(4, &format!("Adam gets every direction below a tenth by step {a1}, Muon by step {m1}"));
        says(4, &format!("(from {m0} to {m1} steps, Adam from {a0} to {a1})"));
        assert!((m1 as f64 / m0 as f64) < (a1 as f64 / a0 as f64));
        // Muon's step keeps its spectral norm between 0.68η and 1.2η at every step (the
        // largest value, at least 1/√8 of the norm, lands in [0.68, 1.13]; a smaller one
        // can land as high as the quintic's peak, 1.2024);
        // gradient descent's falls with the gradient
        let mut muon = LinRun::new(&lin.problem, Rule::Muon, lin.runs[2].lr);
        let mut gd = LinRun::new(&lin.problem, Rule::Sgd, lin.runs[0].lr);
        let mut gd_norms = Vec::new();
        for _ in 0..LIN_STEPS {
            muon.step(&lin.problem);
            gd.step(&lin.problem);
            let n = optim::spectral_norm(muon.opt.update(0), LIN_N, LIN_N) / muon.lr;
            assert!((0.68..=1.2025).contains(&n), "{n}");
            gd_norms.push(optim::spectral_norm(gd.opt.update(0), LIN_N, LIN_N));
        }
        assert!(gd_norms[LIN_STEPS - 1] < 0.05 * gd_norms[0]);
        says(4, r"its spectral norm stays between $0.68\eta$ and $1.2\eta$");
        // the shape factor on a tall layer
        let s = run(5);
        let sn = &s.snap;
        assert_eq!((sn.rows, sn.cols), (16, 2));
        says(5, &format!("Five rounds take its 2 singular values to {} and {}", sig2(sn.after()[0]), sig2(sn.after()[1])));
        says(5, &format!(r"$\sqrt{{\max(1, 16/2)}} = {}$", num(Layer::Input.factor(), 2)));
        says(5, &format!(r"the step's singular values are ${}\eta$ and ${}\eta$", num(sn.step()[0], 2), num(sn.step()[1], 2)));
        let exact = Layer::Input.factor() * (2.0f64 / 32.0).sqrt();
        assert!(close(exact, 1.0 / 2f64.sqrt(), 1e-12));
        says(5, &format!(r"here ${}\eta$ (with the five rounds, ${}\eta$)", num(exact, 2), num(sn.step_rms(), 2)));
    }

    /// A number as a power of ten with one decimal, as the captions write it.
    fn sig_pow(v: f64) -> String {
        sig(v)
    }

    /// The scene opens as the first experiment sets it up.
    #[test]
    fn the_scene_opens_on_the_first_experiment() {
        let opened = MuonScene::create();
        let mut pressed = MuonScene::create();
        pressed.set_up(3);
        pressed.set_up(OPENING);
        assert_eq!(opened.look, pressed.look);
        assert_eq!(opened.pick, pressed.pick);
        assert!(opened.film.playing && opened.film.stage == 0);
    }

    /// Edges: every layer, batch and method, rounds at their ends, the momentum empty or
    /// not, a pick of its own: every number drawn is finite, and the film and glides come
    /// to rest.
    #[test]
    fn degenerate_settings_are_told_honestly() {
        let mut s = MuonScene::create();
        for layer in Layer::ALL {
            for method in Method::ALL {
                for (batch, rounds, fresh) in [(1, 1, true), (1, MOST_ROUNDS, false), (MOST_BATCH, 3, true), (0, 0, false), (usize::MAX, usize::MAX, false)] {
                    s.look.setup = Setup { layer, batch, method, rounds, fresh };
                    s.pick = Pick::Bar(999);
                    s.refresh();
                    let k = s.look.setup;
                    assert!(k.batch >= 1 && k.batch <= MOST_BATCH && k.rounds >= 1 && k.rounds <= MOST_ROUNDS);
                    assert!(s.pick_start().is_finite());
                    for b in s.snap.bars.iter().chain(&s.snap.moved).chain(std::iter::once(&s.snap.reach)) {
                        assert!(b.iter().all(|v| v.is_finite() && *v >= 0.0), "{k:?}: {b:?}");
                    }
                    s.film = Film::at(99, true);
                    for _ in 0..2000 {
                        s.film.advance(1.0 / 60.0, k.stages());
                        s.follow(1.0 / 60.0, false);
                        if !s.film.moving() && !s.gliding {
                            break;
                        }
                    }
                    assert!(!s.film.moving() && !s.gliding, "{k:?}");
                    for g in [&s.bars, &s.moved, &s.curves] {
                        assert!(g.shown.iter().all(|v| v.is_finite()), "{k:?}");
                    }
                }
            }
        }
        // a free pick at the ends of its range, and none
        for v in [1e-9, 1.0, 0.0] {
            s.pick = Pick::Free(v);
            assert!(s.pick_start().is_finite());
        }
        // a zero matrix gives zero bars and a zero step
        let net = Net::new();
        let mut zero = net.clone();
        zero.momentum = Default::default();
        zero.momentum.iter_mut().zip(Layer::ALL).for_each(|(m, l)| *m = vec![0.0; l.shape().0 * l.shape().1]);
        zero.problem = Problem::points(&[], &[], 0, crate::math::arch::Loss::CrossEntropy);
        let sn = Snapshot::new(&zero, Setup::base(), &mut Tape::new());
        assert!(sn.bars.iter().all(|b| b.iter().all(|&v| v == 0.0)) && sn.rank == 0, "{:?}", sn.bars);
        assert!(sn.moved.iter().all(|b| b.iter().all(|v| v.is_finite())));
    }

    /// The film: it plays from the gradient to the step and stops, the picture then at
    /// rest; every stage's bars glide there and arrive.
    #[test]
    fn the_film_plays_to_the_end_and_rests() {
        let mut s = MuonScene::create();
        s.follow(1.0 / 60.0, true);
        let last = s.look.setup.stages() - 1;
        let mut frames = 0;
        let mut seen = 0.0f32;
        while s.animated() {
            s.film.advance(1.0 / 60.0, s.look.setup.stages());
            s.work(&Budget::new(1.0));
            s.follow(1.0 / 60.0, false);
            assert!(s.film.shown >= seen - 1e-6, "the film went back");
            seen = s.film.shown;
            frames += 1;
            assert!(frames < 60 * 30, "still moving");
        }
        assert_eq!((s.film.stage, s.film.shown), (last, last as f32));
        // about a stage's time a stage
        let want = STAGE_TIME * last as f32 * 60.0;
        assert!(frames as f32 >= want && (frames as f32) < want + 90.0, "{frames} frames against {want}");
        // pause and a stage chosen: the picture glides there and rests
        s.film.stage = 2;
        let mut frames = 0;
        while s.film.moving() || s.gliding {
            s.film.advance(1.0 / 120.0, s.look.setup.stages());
            s.follow(1.0 / 120.0, false);
            frames += 1;
            assert!(frames < 600);
        }
        assert!(frames > 5);
    }

    /// The layout keeps everything in the window and apart, from a phone to a desktop.
    #[test]
    fn the_layout_fits_every_window() {
        for (w, h) in [(1220.0, 1000.0), (1600.0, 1000.0), (1100.0, 700.0), (900.0, 560.0), (390.0, 800.0), (360.0, 640.0), (700.0, 500.0), (300.0, 600.0), (500.0, 400.0), (240.0, 180.0), (1600.0, 400.0)] {
            for focus in [Focus::Poly, Focus::Inputs, Focus::Lower] {
                let lay = Layout::of(Vec2::new(w, h), 40.0, focus);
                let inside = |(a, b): Rect| a.x >= 0.0 && a.y >= 0.0 && b.x <= w + 0.5 && b.y <= h + 0.5;
                let apart = |(a, b): Rect, (c, d): Rect| b.x <= c.x || d.x <= a.x || b.y <= c.y || d.y <= a.y;
                assert!(inside(lay.spectrum), "{w}x{h}: {:?}", lay.spectrum);
                if lay.compact {
                    continue;
                }
                let mut boxes = vec![lay.spectrum, lay.stages, lay.play];
                boxes.extend(lay.layers);
                for r in [lay.poly, lay.reach, lay.inputs, lay.lower] {
                    if lay.shows(r) {
                        boxes.push(r);
                    }
                }
                if lay.shows(lay.lower) {
                    boxes.extend(lay.modes);
                }
                if lay.narrow {
                    boxes.extend(lay.tabs);
                }
                for (i, a) in boxes.iter().enumerate() {
                    assert!(inside(*a), "{w}x{h} {focus:?}: box {i} {a:?}");
                    for b in &boxes[i + 1..] {
                        assert!(apart(*a, *b), "{w}x{h} {focus:?}: {a:?} and {b:?} overlap");
                    }
                }
                // the chart the narrow window shows is there and of a usable size
                let shown = match (lay.narrow, focus) {
                    (false, _) => lay.poly,
                    (true, Focus::Poly) => lay.poly,
                    (true, Focus::Inputs) => lay.inputs,
                    (true, Focus::Lower) => lay.lower,
                };
                assert!(lay.shows(shown), "{w}x{h} {focus:?}: {shown:?}");
                assert!(lay.spectrum.1.y - lay.spectrum.0.y > 100.0, "{w}x{h}");
            }
        }
    }

    /// The best step of each rule on the linear layer, from the grid, for every ratio,
    /// turned and lined up: what `BEST` holds (prints the table).
    fn best_steps() -> [[[usize; 3]; 5]; 2] {
        let mut out = [[[0usize; 3]; 5]; 2];
        for (t, turned) in [true, false].into_iter().enumerate() {
            for ratio in 0..RATIOS.len() {
                let problem = LinProblem::new(RATIOS[ratio], turned);
                for (r, &rule) in LIN_RULES.iter().enumerate() {
                    let mut best = (f64::INFINITY, 0);
                    for k in 0..GRID {
                        let mut run = LinRun::new(&problem, rule, grid_lr(k));
                        while !run.done() {
                            run.step(&problem);
                        }
                        let total = run.total();
                        // (the first of equal losses: the smaller step)
                        if total < best.0 {
                            best = (total, k);
                        }
                    }
                    out[t][ratio][r] = best.1;
                }
            }
        }
        out
    }

    #[test]
    #[ignore]
    fn the_linear_layer_s_best_steps_are_the_grid_s() {
        let best = best_steps();
        println!("const BEST: [[[usize; 3]; 5]; 2] = {best:?};");
        assert_eq!(best, BEST);
    }
}




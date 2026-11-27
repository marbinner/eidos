//! Weight decay and the effective learning rate. Weights that feed a normalisation can
//! be scaled without changing the loss, so their gradient is at right angles to them
//! and shrinks as they grow. A step of gradient descent then always lengthens them (by
//! Pythagoras) and turns them by an angle set by η/|w|², the effective learning rate.
//! Without decay the length only grows and the turning slows by itself; with decay the
//! length settles where growth and shrinkage balance, and the turn per step settles at
//! arccos(1 − ηλ), whatever the gradient.
//!
//! Two pictures. Two weights in a made-up loss that depends only on their direction,
//! with noise added to each step's gradient as a mini-batch would add it (the "toy"):
//! the path from above, and its length and turn per step. And a real network with batch
//! normalisation trained on mini-batches: each hidden layer's length and turn per step
//! against the formulas; then the two kinds of decay under Adam, and Muon's weights
//! inside the ball that decay keeps them in.
//!
//! The facts (course notes, curvature.md Part F and muon.md §6.1): the identities of a
//! scale-invariant weight (F1, derived there and tested here on the network's layers);
//! the balance of SGD with decay, |w|⁴ = ηL/(λ(2 − ηλ)) and cos Δ = 1 − ηλ (F6, the
//! exact deterministic form of Wan and co-authors' Theorems 1 and 3, 2021); van
//! Laarhoven's effective learning rate η/|w|² (2017, F2); the decoupled decay of AdamW
//! (Loshchilov and Hutter 2019, F8); the bound of Chen, Li and Liu's Proposition 1
//! (2025; muon.md §6.1), with the reference polynomial's largest output 1.2024 (muon.md
//! §8.3).

use super::*;
use crate::clock::{self, Budget};
use crate::math::arch::{Init, Model, MlpSpec, Norm, Place};
use crate::math::autograd::{Act, Tape};
use crate::math::optim::{self, Hyper, Optimizer, Rule, WeightDecay};
use crate::math::terrain::{Draw, Problem, Sampler};
use crate::scenes::stats::plot::{self, Glide, Plot, Rect, AXES, MARKS};

pub const DEF: SceneDef = SceneDef {
    id: "wdecay",
    title: "Weight decay and the effective learning rate",
    group: GROUP,
    blurb: "For weights that feed a normalisation layer only their direction matters: their length sets the real step size, and weight decay sets their length.",
    make: |_| Box::new(Decay::create()),
};

const PI64: f64 = std::f64::consts::PI;

// ----- numbers as the scene writes them ------------------------------------------------------

/// A number with three significant digits (0.0447, 1.50, 152), with a true minus sign.
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

/// A setting as the labels write it: as few digits as it needs (0.1, 0.005, 0.02).
fn short(v: f64) -> String {
    if !v.is_finite() {
        return "?".to_string();
    }
    let text = sig(v);
    if text.contains('.') {
        text.trim_end_matches('0').trim_end_matches('.').to_string()
    } else {
        text
    }
}

/// An angle in radians, written in degrees with two significant digits (2.6°, 0.72°,
/// 0.0041°; 12° and up whole).
fn degrees(rad: f64) -> String {
    if !rad.is_finite() {
        return "?".to_string();
    }
    let d = rad.to_degrees();
    let a = d.abs();
    let decimals = if a >= 10.0 {
        0
    } else if a > 0.0 {
        ((1.0 - a.log10().floor()) as usize).min(6)
    } else {
        0
    };
    format!("{}°", num(d, decimals))
}

/// The round number (1, 1.2, 1.5, 2, 2.5, 3, 4, 5, 6 or 8 times a power of ten) at or
/// above x.
fn nice_up(x: f64) -> f64 {
    if !(x.is_finite() && x > 0.0) {
        return 1.0;
    }
    let p = 10f64.powf(x.log10().floor());
    let lead = x / p;
    let up = [1.0, 1.2, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0].into_iter().find(|&m| lead <= m * (1.0 + 1e-9)).unwrap_or(10.0);
    up * p
}

// ----- the two weights (the toy) -------------------------------------------------------------

/// The direction in which the toy's loss is lowest: 60° from the first axis.
const BEST: f64 = PI64 / 3.0;
/// Where the toy starts: half a unit from 0, 160° round from the best direction.
const TOY_START_LEN: f64 = 0.5;
const TOY_START_TURN: f64 = 160.0 * PI64 / 180.0;
/// Steps in a run of the toy.
const TOY_STEPS: usize = 3000;
/// The standard deviation of the noise in each step's gradient (in units of the loss's
/// slope along the circle: see [`ToyRun`]).
const TOY_NOISE: f64 = 1.0;
/// The seed of the toy's noise (both of its runs draw the same).
const TOY_SEED: u64 = 3;
/// The ranges of the toy's controls.
const TOY_LEAST_ETA: f64 = 0.003;
const TOY_MOST_ETA: f64 = 1.0;
const TOY_LEAST_LAMBDA: f64 = 1e-4;
const TOY_MOST_LAMBDA: f64 = 0.1;

/// The toy's loss: 1 − cos(θ − θ*), with θ the direction of w and θ* = 60°. It depends
/// on the direction alone, so its lines of equal loss are rays from 0.
fn toy_loss(theta: f64) -> f64 {
    1.0 - (theta - BEST).cos()
}

/// The toy's settings: the step η and the decay λ.
#[derive(Clone, Copy, Debug, PartialEq)]
struct ToySetup {
    eta: f64,
    lambda: f64,
}

impl ToySetup {
    fn base() -> ToySetup {
        ToySetup { eta: 0.1, lambda: 0.01 }
    }

    /// Kept to the controls' ranges (a value that is not a number takes the base's).
    fn kept(self) -> ToySetup {
        let b = ToySetup::base();
        let eta = if self.eta.is_finite() { self.eta.max(TOY_LEAST_ETA).min(TOY_MOST_ETA) } else { b.eta };
        let lambda = if self.lambda.is_nan() {
            b.lambda
        } else if self.lambda > 0.0 {
            self.lambda.max(TOY_LEAST_LAMBDA).min(TOY_MOST_LAMBDA)
        } else {
            0.0
        };
        ToySetup { eta, lambda }
    }
}

/// One run of the toy, whole. Each step, with r = |w| and e the unit vector along the
/// circle (w turned a quarter turn, over r), the gradient is (u/r) e: u = sin(θ − θ*) is
/// the loss's slope along the circle of radius 1, plus a random part of standard
/// deviation [`TOY_NOISE`], new at each step. (A mini-batch's loss depends on the
/// direction alone too, so its gradient is also along the circle and also shrinks as
/// 1/r.) Then w ← (1 − ηλ) w − η (u/r) e.
#[derive(Clone, Debug)]
struct ToyRun {
    setup: ToySetup,
    /// The weights after each step (entry 0: the start).
    w: Vec<[f64; 2]>,
    /// Running sums over the first t steps (entry t) of u² and of the angle turned,
    /// squared: a mean over any stretch costs two subtractions.
    unit2: Vec<f64>,
    turn2: Vec<f64>,
    /// The largest relative difference over the run between |w'|² and (1 − ηλ)²|w|² +
    /// η²|g|²: Pythagoras, held to rounding.
    pythagoras: f64,
}

impl ToyRun {
    fn of(s: ToySetup) -> ToyRun {
        let n = TOY_STEPS;
        let mut rng = Rng::new(TOY_SEED);
        let start = BEST + TOY_START_TURN;
        let mut w = [TOY_START_LEN * start.cos(), TOY_START_LEN * start.sin()];
        let mut run = ToyRun { setup: s, w: Vec::with_capacity(n + 1), unit2: Vec::with_capacity(n + 1), turn2: Vec::with_capacity(n + 1), pythagoras: 0.0 };
        run.w.push(w);
        run.unit2.push(0.0);
        run.turn2.push(0.0);
        let keep = 1.0 - s.eta * s.lambda;
        let (mut su, mut st) = (0.0, 0.0);
        for _ in 0..n {
            let r = (w[0] * w[0] + w[1] * w[1]).sqrt();
            let theta = w[1].atan2(w[0]);
            let u = (theta - BEST).sin() + TOY_NOISE * rng.normal() as f64;
            let along = [-theta.sin(), theta.cos()];
            let g = [u / r * along[0], u / r * along[1]];
            let next = [keep * w[0] - s.eta * g[0], keep * w[1] - s.eta * g[1]];
            let n2 = next[0] * next[0] + next[1] * next[1];
            let want = keep * keep * r * r + s.eta * s.eta * (g[0] * g[0] + g[1] * g[1]);
            if want > 0.0 {
                run.pythagoras = run.pythagoras.max(((n2 - want) / want).abs());
            }
            let turn = (w[0] * next[1] - w[1] * next[0]).abs().atan2(w[0] * next[0] + w[1] * next[1]);
            su += u * u;
            st += turn * turn;
            run.unit2.push(su);
            run.turn2.push(st);
            w = next;
            run.w.push(w);
        }
        run
    }

    fn steps(&self) -> usize {
        self.w.len() - 1
    }

    /// |w| after step t.
    fn len(&self, t: usize) -> f64 {
        let w = self.w[t.min(self.steps())];
        (w[0] * w[0] + w[1] * w[1]).sqrt()
    }

    /// The angle turned per step over the steps before step t ([`window`]), root mean
    /// square, in radians.
    fn turn(&self, t: usize) -> f64 {
        rms_window(&self.turn2, t)
    }

    /// The length at which growth and decay balance for the gradient of the last
    /// relaxation time's steps before step t (None without decay).
    fn balance(&self, t: usize) -> Option<f64> {
        let s = self.setup;
        balance_len(s.eta, s.lambda, mean_window(&self.unit2, t, relaxation(s.eta, s.lambda)))
    }

    /// Where the weights are at a step between two whole ones.
    fn at(&self, s: f64) -> [f64; 2] {
        let n = self.steps();
        let s = s.max(0.0).min(n as f64);
        let i = (s.floor() as usize).min(n);
        let j = (i + 1).min(n);
        let f = s - i as f64;
        [self.w[i][0] + (self.w[j][0] - self.w[i][0]) * f, self.w[i][1] + (self.w[j][1] - self.w[i][1]) * f]
    }
}

/// The steps a mean "over recent steps" at step t takes: the last quarter (at least
/// one), so that a curve is as smooth at step 30 as at step 3000 on a logarithmic axis.
fn window(t: usize) -> usize {
    (t / 4).max(1)
}

/// The root mean square over the window before step t, from running sums.
fn rms_window(sums: &[f64], t: usize) -> f64 {
    mean_window(sums, t, window(t)).max(0.0).sqrt()
}

/// The mean over the last `w` steps before step t (fewer if there are fewer), from
/// running sums. NaN at step 0.
fn mean_window(sums: &[f64], t: usize, w: usize) -> f64 {
    let t = t.min(sums.len().saturating_sub(1));
    let w = w.min(t);
    if w == 0 {
        return f64::NAN;
    }
    (sums[t] - sums[t - w]) / w as f64
}

/// The steps the squared length takes to relax to its balance: 1/(4ηλ) (the linearised
/// map of the course notes, curvature.md F6), at least one.
fn relaxation(eta: f64, lambda: f64) -> usize {
    let x = 4.0 * eta * lambda;
    if x > 0.0 && x.is_finite() {
        ((1.0 / x).ceil() as usize).clamp(1, 1_000_000)
    } else {
        1
    }
}

/// The length at which a step's growth and its decay balance, for weights whose loss
/// does not depend on their length, stepped by gradient descent at η with decay λ, for a
/// mean squared "unit gradient" L = |g|²|w|²: |w|⁴ = ηL/(λ(2 − ηλ)). (The fixed point of
/// |w'|² = (1 − ηλ)²|w|² + η²L/|w|²; the course notes, curvature.md F6, derived there as
/// the exact deterministic form of Wan and co-authors' Theorem 1.) None without decay.
fn balance_len(eta: f64, lambda: f64, unit_sq: f64) -> Option<f64> {
    let d = lambda * (2.0 - eta * lambda);
    (lambda > 0.0 && d > 0.0 && unit_sq.is_finite() && unit_sq >= 0.0).then(|| (eta * unit_sq / d).powf(0.25))
}

/// The angle per step at a steady length: arccos(1 − ηλ), exactly (with g at right
/// angles to w, w'·w = (1 − ηλ)|w|², and if |w'| = |w| that is the cosine times |w|²).
/// About √(2ηλ).
fn balance_turn(eta: f64, lambda: f64) -> Option<f64> {
    let c = 1.0 - eta * lambda;
    (lambda > 0.0 && c > -1.0 && c < 1.0).then(|| c.acos())
}

/// Without decay and with a steady mean squared unit gradient, |w|⁴ grows by about 2η²L
/// a step, and the angle per step falls as 1/√(2t) radians once that growth outweighs
/// the start, whatever η (the course notes, curvature.md F1 item 8).
fn slowing_turn(t: f64) -> f64 {
    1.0 / (2.0 * t.max(1e-9)).sqrt()
}

// ----- the network -----------------------------------------------------------------------------

/// The network: three hidden layers of 16 tanh units, each tanh(γ ⊙ BN(W x)), with
/// batch normalisation, a learnt gain γ and no biases; then a read-out of three units.
/// On 120 points of three spiral arms, with the cross-entropy loss and batches of 32.
const WIDTH: usize = 16;
const DEPTH: usize = 3;
const POINTS: usize = 120;
const BATCH: usize = 32;
/// Steps in a run of the network.
const NET_STEPS: usize = 4000;

/// The network's specification.
fn net_spec() -> MlpSpec {
    MlpSpec {
        inputs: 2,
        width: WIDTH,
        depth: DEPTH,
        outputs: 3,
        act: Act::Tanh,
        norm: Norm::Batch,
        place: Place::Post,
        residual: false,
        residual_scale: 1.0,
        init: Init::Lecun,
        gain: 1.0,
        bias: false,
    }
}

/// The rules the network can be trained with.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Opt {
    Sgd,
    Adam,
    Muon,
}

impl Opt {
    const ALL: [Opt; 3] = [Opt::Sgd, Opt::Adam, Opt::Muon];

    fn rule(self) -> Rule {
        match self {
            Opt::Sgd => Rule::Sgd,
            Opt::Adam => Rule::Adam,
            Opt::Muon => Rule::Muon,
        }
    }

    fn color(self) -> Color {
        match self {
            Opt::Sgd => SGD,
            Opt::Adam => ADAM,
            Opt::Muon => MUON,
        }
    }

    /// The step sizes a slider offers for this rule.
    fn etas(self) -> (f64, f64) {
        match self {
            Opt::Sgd => (0.01, 3.0),
            Opt::Adam => (1e-4, 0.1),
            Opt::Muon => (1e-3, 0.2),
        }
    }

    /// The decays a slider offers (and 0).
    fn lambdas(self) -> (f64, f64) {
        match self {
            Opt::Sgd => (1e-4, 0.1),
            Opt::Adam | Opt::Muon => (1e-3, 2.0),
        }
    }
}

/// Everything a run of the network depends on.
#[derive(Clone, Copy, Debug, PartialEq)]
struct NetSetup {
    opt: Opt,
    eta: f64,
    lambda: f64,
    /// Adam only: the decay added to the gradient (L2) instead of taken off the weights
    /// (AdamW). For gradient descent the two are one rule, θ − η(g + λθ); Muon's is
    /// taken off the weights, as in its reference.
    coupled: bool,
    /// The decay also on the weights that feed no normalisation: the read-out and the
    /// gains (gradient descent and Adam).
    all: bool,
    seed: u64,
}

impl NetSetup {
    fn sgd() -> NetSetup {
        NetSetup { opt: Opt::Sgd, eta: 0.2, lambda: 0.005, coupled: false, all: false, seed: 1 }
    }

    /// The usual settings of a rule: the ones its experiment uses.
    fn usual(opt: Opt) -> NetSetup {
        match opt {
            Opt::Sgd => NetSetup::sgd(),
            Opt::Adam => NetSetup { opt, eta: 0.01, lambda: 0.1, ..NetSetup::sgd() },
            Opt::Muon => NetSetup { opt, eta: 0.02, lambda: 0.2, ..NetSetup::sgd() },
        }
    }

    /// Kept to the controls' ranges (a value that is not a number takes the rule's
    /// usual one), and ηλ at most a half.
    fn kept(self) -> NetSetup {
        let (lo, hi) = self.opt.etas();
        let (llo, lhi) = self.opt.lambdas();
        let usual = NetSetup::usual(self.opt);
        let eta = if self.eta.is_finite() { self.eta.max(lo).min(hi) } else { usual.eta };
        let lambda = if self.lambda.is_nan() {
            usual.lambda
        } else if self.lambda > 0.0 {
            self.lambda.max(llo).min(lhi).min(0.5 / eta)
        } else {
            0.0
        };
        NetSetup { eta, lambda, coupled: self.coupled && self.opt == Opt::Adam, all: self.all && self.opt != Opt::Muon, ..self }
    }

    /// The rule's name, with its kind of decay.
    fn name(&self) -> &'static str {
        match (self.opt, self.coupled, self.lambda > 0.0) {
            (Opt::Sgd, _, _) => "gradient descent",
            (Opt::Adam, false, true) => "AdamW",
            (Opt::Adam, true, true) => "Adam + L2",
            (Opt::Adam, _, false) => "Adam",
            (Opt::Muon, _, _) => "Muon",
        }
    }

    /// How the run is named on the charts.
    fn label(&self) -> String {
        if self.lambda > 0.0 {
            format!(r"{}, $\eta = {}$, $\lambda = {}$", self.name(), short(self.eta), short(self.lambda))
        } else {
            format!(r"{}, $\eta = {}$, no decay", self.name(), short(self.eta))
        }
    }
}

/// The largest singular value Muon's polynomial can give, in exact arithmetic: the
/// reference coefficients (3.4445, −4.7750, 2.0315) map [0, 1.2024] into itself, and
/// the input is first divided by its Frobenius norm (course notes, muon.md §8.3).
const MUON_MOST: f64 = 1.2024;

/// A run of the network, step by step, with what each step did to each hidden layer.
struct NetRun {
    setup: NetSetup,
    model: Model,
    opt: Optimizer,
    problem: Problem,
    sampler: Sampler,
    tape: Tape,
    /// The groups of the three hidden layers' weights, and of the read-out's.
    layers: [usize; DEPTH],
    readout: usize,
    /// Which groups this scene decays itself (gradient descent and Adam; Muon's decay
    /// is the optimiser's own).
    decayed: Vec<bool>,
    /// Steps taken.
    t: usize,
    /// The length of each hidden layer's weights after each step (entry 0: the start),
    /// then the read-out's.
    len: [Vec<f64>; DEPTH + 1],
    /// Running sums over the first t steps of the angle each hidden layer turned,
    /// squared, and of its unit gradient |g|²|W|².
    turn2: [Vec<f64>; DEPTH],
    unit2: [Vec<f64>; DEPTH],
    /// Muon only: each hidden layer's largest singular value after each step.
    top: [Vec<f64>; DEPTH],
    /// Running sums of the mini-batches' losses.
    loss: Vec<f64>,
    /// The largest |cos| of the angle between a hidden layer's weights and their
    /// gradient over the run (0 for a gradient at right angles).
    tilt: f64,
    /// The weights a step started from (each hidden layer's), to measure its turn.
    before: [Vec<f64>; DEPTH],
    batch: Vec<usize>,
    sv: Vec<f64>,
    scratch: optim::Scratch,
    /// Something stopped being a number: the run stops there.
    broken: bool,
}

impl NetRun {
    fn new(setup: NetSetup) -> NetRun {
        let s = setup.kept();
        let model = Model::mlp(&net_spec(), &mut Rng::new(s.seed));
        let problem = Problem::spirals(POINTS, 3, &mut Rng::new(1000 + s.seed));
        let sampler = Sampler::new(problem.len(), BATCH, Draw::WithoutReplacement, Rng::new(2000 + s.seed));
        let mut hyper = match s.opt {
            Opt::Muon => Hyper::muon_reference(),
            o => Hyper::default_for(o.rule()),
        };
        hyper.lr = s.eta;
        if s.opt == Opt::Muon && s.lambda > 0.0 {
            hyper.weight_decay = WeightDecay::Decoupled(s.lambda);
        }
        let opt = Optimizer::new(&model, s.opt.rule(), hyper);
        let index = |name: &str| model.index_of(name).unwrap_or(0);
        let layers = [index("layer 1 weights"), index("layer 2 weights"), index("layer 3 weights")];
        let readout = index("read-out weights");
        let decayed = (0..model.groups().len()).map(|i| s.opt != Opt::Muon && s.lambda > 0.0 && (s.all || layers.contains(&i))).collect();
        let cap = NET_STEPS + 1;
        let start = |g: usize| norm(&model.groups()[g].value);
        let with = |v: f64| {
            let mut x = Vec::with_capacity(cap);
            x.push(v);
            x
        };
        let len = [with(start(layers[0])), with(start(layers[1])), with(start(layers[2])), with(start(readout))];
        let mut run = NetRun {
            setup: s,
            model,
            opt,
            problem,
            sampler,
            tape: Tape::new(),
            layers,
            readout,
            decayed,
            t: 0,
            len,
            turn2: [with(0.0), with(0.0), with(0.0)],
            unit2: [with(0.0), with(0.0), with(0.0)],
            top: [Vec::new(), Vec::new(), Vec::new()],
            loss: with(0.0),
            tilt: 0.0,
            before: [Vec::new(), Vec::new(), Vec::new()],
            batch: Vec::with_capacity(BATCH),
            sv: Vec::new(),
            scratch: optim::Scratch::new(),
            broken: false,
        };
        if s.opt == Opt::Muon {
            for k in 0..DEPTH {
                let v = run.largest(k);
                run.top[k].reserve(cap);
                run.top[k].push(v);
            }
        }
        run
    }

    fn done(&self) -> bool {
        self.t >= NET_STEPS || self.broken
    }

    /// Layer k's largest singular value now.
    fn largest(&mut self, k: usize) -> f64 {
        let g = &self.model.groups()[self.layers[k]];
        optim::singular_values_into(&g.value, g.rows, g.cols, &mut self.sv, &mut self.scratch);
        self.sv.first().copied().unwrap_or(0.0)
    }

    /// One step: a mini-batch's gradient, the decay, the rule's step; and what it did.
    fn step(&mut self) {
        if self.done() {
            return;
        }
        let s = self.setup;
        self.batch.clear();
        self.batch.extend_from_slice(self.sampler.deal());
        let loss = self.problem.grad(&mut self.model, &mut self.tape, &self.batch);
        for k in 0..DEPTH {
            let g = &self.model.groups()[self.layers[k]];
            let (w2, g2, wg) = (dot(&g.value, &g.value), dot(&g.grad, &g.grad), dot(&g.value, &g.grad));
            let unit = w2 * g2;
            let last = self.unit2[k].last().copied().unwrap_or(0.0);
            self.unit2[k].push(last + unit);
            if unit > 0.0 {
                self.tilt = self.tilt.max((wg / unit.sqrt()).abs());
            }
            self.before[k].clear();
            self.before[k].extend_from_slice(&g.value);
        }
        // the decay, on the groups it acts on (Muon's is the optimiser's own)
        if s.lambda > 0.0 && s.opt != Opt::Muon {
            let shrink = s.eta * s.lambda;
            for (group, &on) in self.model.groups_mut().iter_mut().zip(&self.decayed) {
                if !on {
                    continue;
                }
                if s.opt == Opt::Sgd || s.coupled {
                    // added to the gradient, g + λθ: PyTorch's `weight_decay` (for
                    // gradient descent the same as taking ηλθ off the weights)
                    for (g, v) in group.grad.iter_mut().zip(&group.value) {
                        *g += s.lambda * v;
                    }
                } else {
                    // taken off the weights, θ ← θ − ηλθ (AdamW), before the rule's step,
                    // which does not depend on θ
                    for v in group.value.iter_mut() {
                        *v -= shrink * *v;
                    }
                }
            }
        }
        self.opt.step(&mut self.model, s.eta);
        self.t += 1;
        let last = self.loss.last().copied().unwrap_or(0.0);
        self.loss.push(last + loss);
        let mut fine = loss.is_finite();
        for k in 0..DEPTH {
            let now = &self.model.groups()[self.layers[k]].value;
            let b = &self.before[k];
            let (d, n1, n0) = (dot(now, b), norm(now), norm(b));
            let turn = if n1 > 0.0 && n0 > 0.0 { (d / (n1 * n0)).clamp(-1.0, 1.0).acos() } else { 0.0 };
            fine &= n1.is_finite() && turn.is_finite();
            self.len[k].push(n1);
            let last = self.turn2[k].last().copied().unwrap_or(0.0);
            self.turn2[k].push(last + turn * turn);
        }
        let r = norm(&self.model.groups()[self.readout].value);
        fine &= r.is_finite();
        self.len[DEPTH].push(r);
        if s.opt == Opt::Muon {
            for k in 0..DEPTH {
                let v = self.largest(k);
                self.top[k].push(v);
            }
        }
        if !fine {
            self.broken = true;
        }
    }

    /// Layer k's length after step t (k = DEPTH: the read-out's).
    fn len_at(&self, k: usize, t: usize) -> f64 {
        let v = &self.len[k.min(DEPTH)];
        v.get(t.min(v.len().saturating_sub(1))).copied().unwrap_or(f64::NAN)
    }

    /// Layer k's angle turned per step over recent steps before step t (radians, root
    /// mean square).
    fn turn(&self, k: usize, t: usize) -> f64 {
        rms_window(&self.turn2[k.min(DEPTH - 1)], t)
    }

    /// Layer k's largest singular value after step t (Muon only).
    fn top_at(&self, k: usize, t: usize) -> f64 {
        let v = &self.top[k.min(DEPTH - 1)];
        v.get(t.min(v.len().saturating_sub(1))).copied().unwrap_or(f64::NAN)
    }

    /// The mean loss of the mini-batches of recent steps before step t.
    fn batch_loss(&self, t: usize) -> f64 {
        mean_window(&self.loss, t, window(t))
    }

    /// For gradient descent with decay, the length at which layer k's growth and decay
    /// balance, for its unit gradient over the last relaxation time's steps before t.
    fn balance(&self, k: usize, t: usize) -> Option<f64> {
        let s = self.setup;
        if s.opt != Opt::Sgd {
            return None;
        }
        balance_len(s.eta, s.lambda, mean_window(&self.unit2[k.min(DEPTH - 1)], t, relaxation(s.eta, s.lambda)))
    }

    /// The loss on all the data now (the whole set normalised as one batch).
    #[cfg(test)]
    fn whole_loss(&mut self) -> f64 {
        self.problem.loss(&self.model, &mut self.tape)
    }
}

fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

fn norm(a: &[f64]) -> f64 {
    dot(a, a).sqrt()
}

/// For Muon with decay λ at step η, from a largest singular value σ₀: the radius of the
/// ball its square matrices are kept in, b/λ with b = [`MUON_MOST`] the largest
/// singular value a step's matrix can have (its shape factor is 1); and the bound after
/// t steps, b/λ + (1 − ηλ)^t (σ₀ − b/λ) (Chen, Li and Liu 2025, Proposition 1, which
/// needs ηλ ≤ 1).
fn muon_bound(eta: f64, lambda: f64, start: f64, t: f64) -> Option<(f64, f64)> {
    if !(lambda > 0.0 && eta * lambda <= 1.0 && eta > 0.0) {
        return None;
    }
    let radius = MUON_MOST / lambda;
    Some((radius, radius + (1.0 - eta * lambda).powf(t) * (start - radius)))
}

// ----- the experiments ---------------------------------------------------------------------------

/// Which picture is shown.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Tab {
    Toy,
    Net,
}

struct Trial {
    button: &'static str,
    caption: &'static str,
}

const TRIALS: [Trial; 5] = [
    Trial { button: "two weights: a spiral and a circle", caption: r"Gradient descent with $\eta = 0.1$ on two weights whose loss depends on their direction alone, each step with a fresh random push. Each gradient step is at right angles to $w$, so it lengthens $w$. Without decay (grey) the path spirals out, to a length of 2.80 at step 3 000 and still growing, and its turn per step falls, to 0.80°, close to the $1/\sqrt{2t} = 0.74°$ that a gradient of steady size gives, whatever $\eta$. With $\lambda = 0.01$ (cyan) the length settles at 1.52, by the circle where growth and decay balance for the gradient measured (1.54), and the turn stays at 2.6°: the $\arccos(1 - \eta\lambda)$ of a steady length." },
    Trial { button: "a network, with and without decay", caption: r"Each hidden layer feeds a batch normalisation, so the same holds for it: here its gradient is at right angles to it to within 0.03°. Gradient descent, $\eta = 0.2$. Without decay (grey) every layer only lengthens, from about 4.1 to 7.32, 12.1 and 9.45 by step 4 000, and its turn per step falls to between 0.73° and 0.89°, near the $1/\sqrt{2t}$ line: a learning-rate decay nobody set. With $\lambda = 0.005$ the lengths settle within 1.4 % of where growth and decay balance (dashed: 3.21, 5.25 and 4.40), and the layers turn 2.56° to 2.61° a step against $\arccos(1 - \eta\lambda) = 2.56°$: the decay keeps their effective learning rate up. Zhang, Wang, Xu and Grosse (2019) named this as the way weight decay acts on layers that feed batch normalisation, under SGD and Adam. For this seed the run without decay ends with the lower loss on all the data, 0.27 against 0.47." },
    Trial { button: "half the step, twice the decay", caption: r"$\eta = 0.1$ and $\lambda = 0.01$ against $\eta = 0.2$ and $\lambda = 0.005$ (grey): the same product $\eta\lambda$. The layers settle shorter, at 0.73 to 0.75 times the grey run's lengths (the balance, $|W|^4 \propto \eta/\lambda$, gives $1/\sqrt2 = 0.71$), so their effective learning rates $\eta/|W|^2$ end within 12 % of the grey run's, and they turn 2.60° to 2.63° a step against 2.56° to 2.61°: within 3 %. For weights whose loss ignores their length the step and the decay are one knob: from weights $c$ times as long, with $c^2\eta$ and $\lambda/c^2$, gradient descent turns them exactly as before. The read-out and the gains feed none, and here learn with half the step: the loss on all the data ends at 0.56 against 0.47." },
    Trial { button: "Adam: two kinds of decay", caption: r"Adam, $\eta = 0.01$, $\lambda = 0.1$. Taken off the weights (AdamW, orange), the decay multiplies each weight by exactly $1 - \eta\lambda = 0.999$ a step, by 0.905 over 100 steps, whatever Adam's own step does; layer 2 is at 4.09 after them, from 4.14. Added to the gradient (L2, grey), the decay goes through Adam's division by $\sqrt{\hat v}$ with the rest, and here acts far more strongly: layer 2 falls to 0.751 in the same 100 steps. The runs then settle at different lengths (layer 2: 3.65 against 1.15) and turns (0.47° against 1.00° a step). For gradient descent the two kinds are one rule." },
    Trial { button: "Muon: inside a ball", caption: r"Muon, $\eta = 0.02$, with $\lambda = 0.2$ taken off its matrices. Each step adds $\eta$ times a matrix whose singular values are at most 1.2024 (the peak of Muon's polynomial), so $\sigma_{\max}' \le (1 - \eta\lambda)\,\sigma_{\max} + 1.2024\,\eta$, and the largest singular value can never pass $1.2024/\lambda = 6.01$ (yellow; dashed curve: how far it can have got by each step). Layers 2 and 3 stay well inside, at 3.17 and 2.44: steps whose directions change add up to less than the bound allows. Without decay (grey) they grow to 33.1 and 17.1. Chen, Li and Liu (2025) showed that Muon with exact orthogonalisation, with steps that shrink over training and further conditions, converges to points that meet the conditions for the lowest loss inside the ball of radius $1/\lambda$. Xie and Li (2024) showed the like for full-batch AdamW, if it converges at all (with $\beta_1 \le \beta_2$, and steps that never grow but add up to infinity), for the box in which every weight is at most $1/\lambda$ in size." },
];

/// The scene opens as the first experiment sets it up.
const OPENING: usize = 0;

/// Everything experiment k sets: the picture, the toy, the network's run and the run
/// drawn faint beside it.
fn trial(k: usize) -> (Tab, ToySetup, NetSetup, Option<NetSetup>) {
    let toy = ToySetup::base();
    let sgd = NetSetup::sgd();
    match k {
        1 => (Tab::Net, toy, sgd, Some(NetSetup { lambda: 0.0, ..sgd })),
        2 => (Tab::Net, toy, NetSetup { eta: sgd.eta / 2.0, lambda: sgd.lambda * 2.0, ..sgd }, Some(sgd)),
        3 => {
            let adamw = NetSetup::usual(Opt::Adam);
            (Tab::Net, toy, adamw, Some(NetSetup { coupled: true, ..adamw }))
        }
        4 => {
            let muon = NetSetup::usual(Opt::Muon);
            (Tab::Net, toy, muon, Some(NetSetup { lambda: 0.0, ..muon }))
        }
        _ => (Tab::Toy, toy, sgd, Some(NetSetup { lambda: 0.0, ..sgd })),
    }
}

// ----- the film ------------------------------------------------------------------------------------

/// A run played as a film on a logarithmic clock: the step shown is
/// (T + 1)^(clock/length) − 1, so the first steps come one by one and each decade of
/// steps takes the same time.
#[derive(Clone, Copy, Debug, Default)]
struct Film {
    clock: f64,
    playing: bool,
    /// Play once the picture has glided to the new run.
    waiting: bool,
}

impl Film {
    fn start() -> Film {
        Film { clock: 0.0, playing: false, waiting: true }
    }

    /// The same with the clock at the first step: the network's charts begin there (a
    /// logarithmic axis has no step 0), so the film begins there too.
    fn from_first(steps: usize) -> Film {
        Film { clock: Self::clock_of(1.0, steps), ..Film::start() }
    }

    /// How long a whole run takes to play, in seconds.
    fn length(steps: usize) -> f64 {
        2.0 + 0.85 * ((steps + 1) as f64).ln()
    }

    /// The step shown at a time of the clock.
    fn step_at(clock: f64, steps: usize) -> f64 {
        let n = (steps + 1) as f64;
        (n.powf((clock / Self::length(steps)).clamp(0.0, 1.0)) - 1.0).clamp(0.0, steps as f64)
    }

    /// The time of the clock at which a step is shown.
    fn clock_of(step: f64, steps: usize) -> f64 {
        let n = (steps + 1) as f64;
        Self::length(steps) * (step.max(0.0) + 1.0).ln() / n.ln()
    }

    fn busy(&self) -> bool {
        self.playing || self.waiting
    }
}

// ----- the scene -----------------------------------------------------------------------------------

/// Points each curve of a chart is drawn with.
const CURVE_N: usize = 120;

pub struct Decay {
    tab: Tab,
    toy: ToySetup,
    toy_run: ToyRun,
    /// The same toy without decay, drawn faint (None when the toy has no decay).
    toy_plain: Option<ToyRun>,
    toy_film: Film,
    /// The step the toy's picture shows (a fraction between two whole ones).
    toy_shown: f64,
    net: NetSetup,
    run: NetRun,
    /// The run drawn faint beside the network's, if any.
    compare: Option<NetSetup>,
    compared: Option<NetRun>,
    net_film: Film,
    net_shown: f64,
    look: Look,
    /// The window last drawn in (a new one is drawn at once, without gliding).
    canvas: Option<Vec2>,
    /// Where the charts' labels were put, and whether one is gliding to a new place.
    placed: Vec<Placed>,
    labels_moving: bool,
    /// The lines as drawn, gliding to where they are going (in the order they are drawn
    /// in a frame: `glide_at` counts them), and the toy's heads; whether one is moving.
    glides: Vec<Glide>,
    glide_at: usize,
    heads: Glide,
    glides_moving: bool,
    /// This frame's time step for the glides, and whether they are to be there at once.
    dt_now: f32,
    snap_now: bool,
    /// The picture shown has changed (the other tab): the next frame draws the new one
    /// at once, without gliding from the old.
    cut: bool,
}

/// The ranges of the charts, gliding to where they are going.
#[derive(Clone, Debug, Default)]
struct Look {
    frame: Glide,
    moving: bool,
}

// The ranges, in the order the glide keeps them: the toy's plane (half its width), the
// top of its length chart, its turn chart's bottom and top (log₁₀ of degrees); the
// network's size chart's, turn chart's and loss chart's bottoms and tops (log₁₀).
const PLANE: usize = 0;
const TOY_LEN: usize = 1;
const TOY_TURN: usize = 2;
const NET_SIZE: usize = 4;
const NET_TURN: usize = 6;
const NET_LOSS: usize = 8;
const RANGES: usize = 10;

impl Decay {
    /// (Nothing here needs the graphics device, so the tests can make one too.)
    fn create() -> Decay {
        let (tab, toy, net, compare) = trial(OPENING);
        let mut s = Decay {
            tab,
            toy,
            toy_run: ToyRun::of(toy),
            toy_plain: None,
            toy_film: Film::start(),
            toy_shown: 0.0,
            net,
            run: NetRun::new(net),
            compare,
            compared: None,
            net_film: Film::start(),
            net_shown: 0.0,
            look: Look::default(),
            canvas: None,
            placed: Vec::new(),
            labels_moving: false,
            glides: Vec::new(),
            glide_at: 0,
            heads: Glide::default(),
            glides_moving: false,
            dt_now: 0.0,
            snap_now: true,
            cut: true,
        };
        s.set_up(OPENING);
        s
    }

    /// Set up experiment k, and play it.
    fn set_up(&mut self, k: usize) {
        let (tab, toy, net, compare) = trial(k);
        self.cut |= self.tab != tab;
        self.tab = tab;
        self.toy = toy;
        self.net = net;
        self.compare = compare;
        self.restart_toy();
        self.restart_net();
    }

    /// The toy's runs for its settings, played from the start.
    fn restart_toy(&mut self) {
        self.toy = self.toy.kept();
        self.toy_run = ToyRun::of(self.toy);
        self.toy_plain = (self.toy.lambda > 0.0).then(|| ToyRun::of(ToySetup { lambda: 0.0, ..self.toy }));
        self.toy_film = Film::start();
        self.toy_shown = 0.0;
    }

    /// The network's runs for its settings, started again.
    fn restart_net(&mut self) {
        self.net = self.net.kept();
        self.compare = self.compare.map(|c| c.kept()).filter(|c| *c != self.net);
        self.run = NetRun::new(self.net);
        self.compared = self.compare.map(NetRun::new);
        self.net_film = Film::from_first(NET_STEPS);
        self.net_shown = 0.0;
    }

    /// Bring the runs up to date with the settings (a change starts a run again).
    fn refresh(&mut self) {
        if self.toy.kept() != self.toy_run.setup {
            self.restart_toy();
        }
        let kept = self.net.kept();
        if kept != self.run.setup || self.compare.map(|c| c.kept()) != self.compared.as_ref().map(|c| c.setup) {
            self.restart_net();
        }
    }

    /// Halve the network's step and double its decay; the run until now stays, faint.
    fn halve_and_double(&mut self) {
        self.compare = Some(self.net);
        self.net = NetSetup { eta: self.net.eta / 2.0, lambda: self.net.lambda * 2.0, ..self.net };
        self.restart_net();
    }

    /// The film of the picture shown.
    fn film(&self) -> &Film {
        match self.tab {
            Tab::Toy => &self.toy_film,
            Tab::Net => &self.net_film,
        }
    }

    fn film_mut(&mut self) -> &mut Film {
        match self.tab {
            Tab::Toy => &mut self.toy_film,
            Tab::Net => &mut self.net_film,
        }
    }

    /// Has the picture shown played to its end?
    fn at_end(&self) -> bool {
        match self.tab {
            Tab::Toy => self.toy_shown >= TOY_STEPS as f64 - 1e-9,
            Tab::Net => self.net_shown >= self.net_end() - 1e-9,
        }
    }

    /// The last step the network's film can show: its runs' end (or where one broke).
    fn net_end(&self) -> f64 {
        let mut end = if self.run.broken { self.run.t } else { NET_STEPS };
        if let Some(c) = &self.compared {
            if c.broken {
                end = end.min(c.t);
            }
        }
        end as f64
    }

    /// Move the film of the picture shown on by `dt` seconds of the scene's time,
    /// training the network as far as it shows (within `budget`).
    fn advance(&mut self, dt: f32, budget: &Budget) {
        let glided = !self.look.moving;
        let film = self.film_mut();
        if film.waiting && glided {
            film.waiting = false;
            film.playing = true;
        }
        if film.playing && dt > 0.0 {
            film.clock += dt as f64;
        }
        match self.tab {
            Tab::Toy => {
                let want = Film::step_at(self.toy_film.clock, TOY_STEPS);
                self.toy_shown = want;
                if self.toy_film.playing && want >= TOY_STEPS as f64 - 1e-9 {
                    self.toy_film.playing = false;
                }
            }
            Tab::Net => {
                let want = Film::step_at(self.net_film.clock, NET_STEPS);
                let goal = (want.ceil() as usize).min(NET_STEPS);
                // (at least one step a frame for each run, so that even a slow machine
                // sees the film go on)
                let mut first = true;
                loop {
                    let behind = |r: &NetRun| r.t < goal && !r.done();
                    let main = behind(&self.run);
                    let other = self.compared.as_ref().is_some_and(behind);
                    if !(main || other) || (!first && budget.spent()) {
                        break;
                    }
                    if main {
                        self.run.step();
                    }
                    if other {
                        if let Some(c) = self.compared.as_mut() {
                            c.step();
                        }
                    }
                    first = false;
                }
                // the picture shows no more than both runs have done; the clock waits
                let done = self.compared.as_ref().map_or(self.run.t, |c| c.t.min(self.run.t).max(if c.broken { self.run.t } else { 0 }));
                let done = if self.run.broken { self.compared.as_ref().map_or(self.run.t, |c| c.t) } else { done };
                let shown = want.min(done as f64).min(self.net_end());
                if shown < want {
                    self.net_film.clock = Film::clock_of(shown, NET_STEPS);
                }
                self.net_shown = shown;
                if self.net_film.playing && shown >= self.net_end() - 1e-9 && self.run.done() && self.compared.as_ref().is_none_or(|c| c.done()) {
                    self.net_film.playing = false;
                }
            }
        }
    }

    /// Play from where the film is (from the start, if it is at the end).
    fn play(&mut self) {
        if self.at_end() {
            match self.tab {
                Tab::Toy => self.restart_toy(),
                Tab::Net => {
                    // (the runs are kept: only the film starts again)
                    self.net_film = Film::from_first(NET_STEPS);
                    self.net_shown = 0.0;
                }
            }
        } else {
            self.film_mut().playing = true;
        }
    }
}

// ----- the layout ------------------------------------------------------------------------------

/// Where everything goes in a window (logical pixels from its top-left corner).
#[derive(Clone, Copy, Debug, PartialEq)]
struct Layout {
    view: Vec2,
    narrow: bool,
    /// Text sizes: titles, small print.
    text: f32,
    small: f32,
    /// The two tabs; the film's buttons (play or pause, from the start) and the
    /// network's "halve η, double λ".
    tabs: [Rect; 2],
    buttons: [Rect; 3],
    /// The toy: its plane, its title, and its two charts (length, turn).
    plane: Rect,
    plane_head: Vec2,
    toy_charts: [Rect; 2],
    /// The network: its three charts (size, turn, loss).
    net_charts: [Rect; 3],
    /// Only the plane, or only the first chart: the window is too small for the rest.
    compact: bool,
}

/// Room round a chart's rectangle for its tick numbers and names: left, below, above.
const CHART_LEFT: f32 = 50.0;
const CHART_BELOW: f32 = 30.0;
const CHART_ABOVE: f32 = 24.0;

impl Layout {
    fn of(view: Vec2, inset: f32) -> Layout {
        let view = view.max(Vec2::splat(1.0));
        let narrow = view.x < 700.0;
        let (text, small) = if narrow { (12.0, 10.5) } else { (13.5, 11.5) };
        let side = if narrow { 10.0 } else { 22.0 };
        let top = inset.max(0.0) + if narrow { 6.0 } else { 12.0 };
        let r = |x0: f32, y0: f32, x1: f32, y1: f32| (Vec2::new(x0, y0), Vec2::new(x1.max(x0 + 1.0), y1.max(y0 + 1.0)));
        let none = r(0.0, 0.0, 0.0, 0.0);
        let button_h = if narrow { 24.0 } else { 26.0 };
        let width = view.x - 2.0 * side;
        let mut lay = Layout {
            view,
            narrow,
            text,
            small,
            tabs: [none; 2],
            buttons: [none; 3],
            plane: none,
            plane_head: Vec2::ZERO,
            toy_charts: [none; 2],
            net_charts: [none; 3],
            compact: false,
        };
        // the buttons: the tabs on the left, the film's on the right (on a phone, two rows)
        let mut y = top;
        if narrow {
            let tabs = plot::even_row(2, side, view.x - side, y, button_h, 6.0);
            lay.tabs = [tabs[0], tabs[1]];
            y += button_h + 6.0;
            let row = plot::even_row(3, side, view.x - side, y, button_h, 6.0);
            lay.buttons = [row[0], row[1], row[2]];
        } else {
            let half = (width - 12.0) * 0.5;
            let tabs = plot::even_row(2, side, side + half.min(420.0), y, button_h, 6.0);
            lay.tabs = [tabs[0], tabs[1]];
            let row = plot::even_row(3, view.x - side - half.min(460.0), view.x - side, y, button_h, 6.0);
            lay.buttons = [row[0], row[1], row[2]];
        }
        y += button_h + if narrow { 10.0 } else { 14.0 };
        let bottom = view.y - if narrow { 8.0 } else { 14.0 };
        let avail = bottom - y;
        let gap = if narrow { 8.0 } else { 12.0 };
        if avail < 160.0 || width < 230.0 || (avail - 2.0 * gap) / 3.0 - CHART_ABOVE - CHART_BELOW < 50.0 {
            // one picture only: the plane, or the first chart
            lay.compact = true;
            let side_len = (avail - lay.text - 8.0).min(width).max(1.0);
            let x0 = side + (width - side_len) * 0.5;
            lay.plane_head = Vec2::new(x0, y + lay.text);
            lay.plane = r(x0, y + lay.text + 6.0, x0 + side_len, y + lay.text + 6.0 + side_len);
            lay.net_charts[0] = r(side + CHART_LEFT, y + CHART_ABOVE, view.x - side - 8.0, bottom - CHART_BELOW);
            return lay;
        }
        // the network: three charts one under another
        let h = (avail - 2.0 * gap) / 3.0;
        for k in 0..3 {
            let y0 = y + k as f32 * (h + gap);
            lay.net_charts[k] = r(side + CHART_LEFT, y0 + CHART_ABOVE, view.x - side - 8.0, y0 + h - CHART_BELOW);
        }
        // the toy: the plane beside its charts on a wide window, above them on a tall one
        let wide = view.x >= 1.2 * view.y && view.x >= 800.0;
        let head = lay.text + 8.0;
        if wide {
            let side_len = (avail - head).min(width * 0.5).max(1.0);
            lay.plane_head = Vec2::new(side, y + lay.text);
            lay.plane = r(side, y + head, side + side_len, y + head + side_len);
            let x0 = side + side_len + 24.0;
            let ch = (avail - gap) * 0.5;
            for k in 0..2 {
                let y0 = y + k as f32 * (ch + gap);
                lay.toy_charts[k] = r(x0 + CHART_LEFT, y0 + CHART_ABOVE, view.x - side - 8.0, y0 + ch - CHART_BELOW);
            }
        } else {
            let charts = (avail * 0.42).max(2.0 * 120.0).min(avail - head - 120.0);
            let side_len = (avail - head - charts - gap).min(width).max(1.0);
            let x0 = side + (width - side_len) * 0.5;
            lay.plane_head = Vec2::new(side, y + lay.text);
            lay.plane = r(x0, y + head, x0 + side_len, y + head + side_len);
            let y1 = y + head + side_len + gap;
            let ch = (bottom - y1 - gap) * 0.5;
            for k in 0..2 {
                let y0 = y1 + k as f32 * (ch + gap);
                lay.toy_charts[k] = r(side + CHART_LEFT, y0 + CHART_ABOVE, view.x - side - 8.0, y0 + ch - CHART_BELOW);
            }
        }
        lay
    }
}

// ----- what is drawn ------------------------------------------------------------------------------

// The batches of lines, by kind (`plot::apart`): each line whose number of points
// changes from frame to frame in a kind of its own.
const PATH_MAIN: u8 = plot::OWN;
const PATH_FAINT: u8 = plot::OWN + 1;
const CURVES_MAIN: u8 = plot::OWN + 2;
const CURVES_FAINT: u8 = plot::OWN + 3;
const FORMULA: u8 = plot::OWN + 4;
const BOUND: u8 = plot::OWN + 5;
const NOW: u8 = plot::OWN + 6;

/// Every batch the scene draws in, made in the same order in every frame, so that one
/// that is empty in a frame does not move the later ones up a place (see `eidos motion`).
fn hold_batches(f: &mut Frame) {
    let corner = f.hud(Vec2::ZERO);
    for kind in [AXES, MARKS, plot::BUTTONS, PATH_MAIN, PATH_FAINT, CURVES_MAIN, CURVES_FAINT, NOW] {
        plot::apart(f, kind, false).segment(corner, corner, 1.0, Color::CLEAR);
    }
    for kind in [FORMULA, BOUND] {
        plot::apart(f, kind, true).segment(corner, corner, 1.0, Color::CLEAR);
    }
    dots(f);
}

/// The point batch of the heads of the paths.
fn dots<'b>(f: &'b mut Frame<'_>) -> &'b mut crate::gfx::draw::PointBatch {
    f.dl.points_with(PointKind::Overlay, PointStyle { outline_px: 0.0, ..Default::default() })
}

/// The colour of hidden layer k (0, 1, 2) of a run of this colour: lighter further in.
fn shade(c: Color, k: usize) -> Color {
    c.mix(pal::WHITE, [0.0, 0.33, 0.62][k.min(2)])
}

/// The read-out's colour.
const READOUT: Color = pal::TEXT;

/// The steps across a chart: log₁₀ of the step, from step 1 to the last.
fn across(k: usize, steps: usize) -> f64 {
    (steps.max(2) as f64).log10() * k as f64 / (CURVE_N - 1) as f64
}

/// A value at a step between two whole ones, straight between the two.
fn between(s: f64, f: impl Fn(usize) -> f64) -> f64 {
    let i = s.max(0.0).floor() as usize;
    let frac = s - i as f64;
    let a = f(i);
    if frac <= 0.0 {
        return a;
    }
    let b = f(i + 1);
    if a.is_finite() && b.is_finite() {
        a + (b - a) * frac
    } else {
        a
    }
}

/// A curve's points on a chart, `CURVE_N` of them, from step 1 to the last: the value at
/// each step (`value`, None where it has none), drawn as far as step `head` (the points
/// past it wait at the head, so that the number of points never changes; points before
/// the first value wait at it). Kept inside the chart. With `log`, the chart's y is
/// log₁₀ of the value. With no value at all, every point at the chart's bottom left,
/// and false.
fn curve(p: &Plot, steps: usize, head: f64, log: bool, value: impl Fn(f64) -> Option<f64>) -> (Vec<Vec2>, bool) {
    let head = head.max(1.0);
    let (min, max) = p.rect;
    let pts: Vec<Option<Vec2>> = (0..CURVE_N)
        .map(|k| {
            let s = 10f64.powf(across(k, steps)).min(head);
            let x = s.log10();
            value(s).and_then(|v| {
                let y = if log { (v > 0.0).then(|| v.log10()) } else { Some(v) };
                y.filter(|y| y.is_finite()).map(|y| p.at(x, y).max(min).min(max))
            })
        })
        .collect();
    let Some(first) = pts.iter().flatten().next().copied() else { return (vec![Vec2::new(min.x, max.y); CURVE_N], false) };
    let mut last = first;
    let pts = pts
        .into_iter()
        .map(|q| {
            last = q.unwrap_or(last);
            last
        })
        .collect();
    (pts, true)
}

/// The first step from which the balance is drawn: before it, the mean of the gradient
/// it is worked out from has too few steps in it to say much.
const BALANCE_FROM: f64 = 20.0;

/// A polyline in screen space in the batch of its kind, one colour.
fn line(f: &mut Frame, kind: u8, dashed: bool, pts: &[Vec2], width: f32, color: Color) {
    if pts.len() < 2 || pts.iter().any(|p| !p.is_finite()) {
        return;
    }
    let pts: Vec<Vec3> = pts.iter().map(|p| f.hud(*p)).collect();
    plot::apart(f, kind, dashed).polyline(pts, width, color);
}

/// A line to be drawn once it has glided ([`Decay::emit`]).
struct Stroke {
    kind: u8,
    dashed: bool,
    pts: Vec<Vec2>,
    width: f32,
    color: Color,
}

/// A line for [`Decay::emit`] to draw.
fn ink(strokes: &mut Vec<Stroke>, kind: u8, dashed: bool, pts: &[Vec2], width: f32, color: Color) {
    strokes.push(Stroke { kind, dashed, pts: pts.to_vec(), width, color });
}

/// The log₁₀ ticks of a range: the whole decades, and 2 and 5 of each where there are few.
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

/// A number at a tick of a logarithmic axis: 1, 10, 1 000; 0.2, 0.05.
fn tick_name(x: f64) -> String {
    let v = 10f64.powf(x);
    if v >= 1.0 {
        count(v.round() as usize)
    } else {
        short(v)
    }
}

/// A chart's backing, title, axes and ticks: across, log₁₀ of the step from 1 to the
/// last; up, `y` (log₁₀ of the value with `log`), each tick named by `name`.
#[allow(clippy::too_many_arguments)]
fn chart(f: &mut Frame, lay: &Layout, p: &Plot, title: &str, steps: usize, log: bool, name: impl Fn(f64) -> String) {
    if p.tiny() {
        return;
    }
    let (min, max) = p.rect;
    let size = lay.small;
    p.backing(f, CHART_LEFT - 4.0, CHART_BELOW - 4.0, CHART_ABOVE - 2.0, 6.0);
    f.text_px(Vec2::new(min.x - CHART_LEFT + 10.0, min.y - 6.0), title, lay.small + 0.5, pal::TEXT, Align2::LEFT_BOTTOM);
    plot::line_of(f, AXES, Vec2::new(min.x, max.y), max, 1.0, pal::DIM.a(0.8));
    plot::line_of(f, AXES, min, Vec2::new(min.x, max.y), 1.0, pal::DIM.a(0.8));
    for (x, named) in log_ticks(0.0, (steps.max(2) as f64).log10()) {
        let at = p.px_x(x);
        plot::line_of(f, AXES, Vec2::new(at, max.y), Vec2::new(at, max.y + if named { 4.0 } else { 2.0 }), 1.0, pal::DIM.a(0.8));
        if named {
            f.text_px(Vec2::new(at, max.y + 5.0), tick_name(x), size, pal::DIM, Align2::CENTER_TOP);
        }
    }
    f.text_px(Vec2::new(max.x, max.y + 5.0 + size + 1.0), "step", size, pal::DIM, Align2::RIGHT_TOP);
    let ticks: Vec<f64> = if log {
        log_ticks(p.y[0], p.y[1]).into_iter().filter(|t| t.1).map(|t| t.0).collect()
    } else {
        plot::Ticks::of(p.y[0], p.y[1], p.height() * 1.4).values().collect()
    };
    for y in ticks {
        if y < p.y[0] - 1e-9 || y > p.y[1] + 1e-9 {
            continue;
        }
        let at = p.px_y(y);
        plot::line_of(f, AXES, Vec2::new(min.x - 4.0, at), Vec2::new(min.x, at), 1.0, pal::DIM.a(0.8));
        plot::line_of(f, AXES, Vec2::new(min.x, at), Vec2::new(max.x, at), 1.0, pal::DIM.a(0.1));
        f.text_px(Vec2::new(min.x - 6.0, at), name(y), size, pal::DIM, Align2::RIGHT_CENTER);
    }
}

/// A label of a chart: where it wants to be (the point its alignment anchors), what it
/// says, its colour.
struct ChartLabel {
    at: Vec2,
    align: Align2,
    text: String,
    color: Color,
}

/// Where a label was put: its chart, its words, the slot it took (above or below its
/// own place, in its heights) and how far from its place it is drawn now.
#[derive(Clone, Debug, Default)]
struct Placed {
    chart: u8,
    text: String,
    slot: i32,
    drawn: f32,
}

/// The labels on plates, each at its own place if that is clear of those placed before
/// it, else the nearest clear place above or below it, inside the rectangle (in the order
/// given: the first keep their places). A label keeps the slot it had while that is
/// still clear, and glides to a new one (`memo` remembers them; true while one glides).
/// (As the noise floor's scene places its labels.)
#[allow(clippy::too_many_arguments)]
fn place(f: &mut Frame, chart: u8, tags: Vec<ChartLabel>, size: f32, (min, max): Rect, memo: &[Placed], now: &mut Vec<Placed>) -> bool {
    let mut taken: Vec<Rect> = Vec::new();
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
        let before = memo.iter().find(|m| m.chart == chart && m.text == t.text).cloned();
        // (its own place first, whenever that is clear; else the slot it had)
        let slots = std::iter::once(0).chain(before.as_ref().map(|b| b.slot)).chain((1..13).map(|k| if k % 2 == 1 { -(k + 1) / 2 } else { k / 2 }));
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
        now.push(Placed { chart, text: t.text, slot, drawn });
    }
    gliding
}

impl Decay {
    /// The ranges as they are to be (see `RANGES`): the toy's from its whole runs, the
    /// network's from what has been shown of its runs.
    fn goal(&self) -> Vec<f32> {
        let mut g = vec![0.0f32; RANGES];
        // the toy
        let toys: Vec<&ToyRun> = std::iter::once(&self.toy_run).chain(self.toy_plain.as_ref()).collect();
        let n = TOY_STEPS;
        fn samples(r: &ToyRun, n: usize) -> impl Iterator<Item = (f64, f64)> + '_ {
            (0..CURVE_N).map(move |k| (10f64.powf(across(k, n)).round() as usize).min(n)).map(move |t| (r.len(t), r.turn(t.max(1))))
        }
        let mut most = TOY_START_LEN;
        let (mut lo, mut hi) = (f64::INFINITY, 0.0f64);
        for r in &toys {
            most = (0..=n).step_by(5).map(|t| r.len(t)).fold(most, f64::max).max(r.len(n));
            for (_, turn) in samples(r, n) {
                if turn > 0.0 && turn.is_finite() {
                    lo = lo.min(turn);
                    hi = hi.max(turn);
                }
            }
            if let Some(b) = r.balance(n) {
                most = most.max(b);
            }
        }
        if let Some(b) = balance_turn(self.toy.eta, self.toy.lambda) {
            lo = lo.min(b);
            hi = hi.max(b);
        }
        if self.toy_plain.is_some() || self.toy.lambda == 0.0 {
            lo = lo.min(slowing_turn(n as f64));
        }
        g[PLANE] = ((most * 1.08 * 2.0).ceil() / 2.0).max(1.0) as f32;
        g[TOY_LEN] = nice_up(most * 1.1) as f32;
        let (a, b) = log_range(lo.to_degrees(), hi.to_degrees(), 1.0);
        g[TOY_TURN] = a as f32;
        g[TOY_TURN + 1] = b as f32;
        // the network, as far as it is shown
        let head = self.net_shown.max(1.0);
        let runs: Vec<&NetRun> = std::iter::once(&self.run).chain(self.compared.as_ref()).collect();
        let (mut top, mut bottom, mut lo, mut hi, mut llo, mut lhi) = (0.0f64, f64::INFINITY, f64::INFINITY, 0.0f64, f64::INFINITY, 0.0f64);
        let muon = self.net.opt == Opt::Muon;
        for r in &runs {
            for k in 0..CURVE_N {
                let t = (10f64.powf(across(k, NET_STEPS)).min(head).round() as usize).max(1);
                if t > r.t {
                    break;
                }
                for l in 0..DEPTH {
                    let size = if muon { r.top_at(l, t) } else { r.len_at(l, t) };
                    if !(muon && l == 0) && size > 0.0 {
                        top = top.max(size);
                        bottom = bottom.min(size);
                    }
                    if let Some(b) = r.balance(l, t).filter(|b| *b > 0.0 && t as f64 >= BALANCE_FROM) {
                        top = top.max(b);
                        bottom = bottom.min(b);
                    }
                    let turn = r.turn(l, t);
                    if turn > 0.0 && turn.is_finite() && !(muon && l == 0) {
                        lo = lo.min(turn);
                        hi = hi.max(turn);
                    }
                }
                if r.setup.all && r.len_at(DEPTH, t) > 0.0 {
                    top = top.max(r.len_at(DEPTH, t));
                    bottom = bottom.min(r.len_at(DEPTH, t));
                }
                let loss = r.batch_loss(t);
                if loss > 0.0 && loss.is_finite() {
                    llo = llo.min(loss);
                    lhi = lhi.max(loss);
                }
            }
            if r.setup.opt == Opt::Sgd {
                if let Some(b) = balance_turn(r.setup.eta, r.setup.lambda) {
                    lo = lo.min(b);
                    hi = hi.max(b);
                }
            }
        }
        if muon {
            if let Some((radius, _)) = muon_bound(self.net.eta, self.net.lambda, 0.0, 0.0) {
                top = top.max(radius);
                bottom = bottom.min(1.0 / self.net.lambda);
            }
        }
        let (a, b) = log_range(bottom, top, 0.5);
        g[NET_SIZE] = a as f32;
        g[NET_SIZE + 1] = b as f32;
        let (a, b) = log_range(lo.to_degrees(), hi.to_degrees(), 1.0);
        g[NET_TURN] = a as f32;
        g[NET_TURN + 1] = b as f32;
        let (a, b) = log_range(llo, lhi.max(1.2), 0.6);
        g[NET_LOSS] = a as f32;
        g[NET_LOSS + 1] = b as f32;
        g
    }

    /// Bring the ranges `dt` seconds nearer to where they are going. (`snap`: there at once.)
    fn follow(&mut self, dt: f32, snap: bool) {
        let goal = self.goal();
        if snap {
            self.look.frame.snap(&goal);
            self.look.moving = false;
        } else {
            self.look.moving = self.look.frame.follow(&goal, dt, false);
        }
    }
}

/// A logarithmic range (log₁₀) holding the values from `lo` to `hi`, its ends on half
/// decades, at least `least` decades wide (and at most four); (0, 1) if there are none.
fn log_range(lo: f64, hi: f64, least: f64) -> (f64, f64) {
    if !(lo.is_finite() && hi.is_finite() && lo > 0.0 && hi >= lo) {
        return (0.0, 1.0);
    }
    let mut a = (2.0 * lo.log10() - 0.15).floor() / 2.0;
    let mut b = (2.0 * hi.log10() + 0.15).ceil() / 2.0;
    if b - a < least {
        let mid = 0.5 * (a + b);
        a = ((mid - 0.5 * least) * 2.0).floor() / 2.0;
        b = a + least.max(0.5);
    }
    if b - a > 4.0 {
        a = b - 4.0;
    }
    (a, b)
}

// ----- drawing ---------------------------------------------------------------------------------

impl Decay {
    /// The tabs and the film's buttons; true if anything changed.
    fn buttons(&mut self, f: &mut Frame, lay: &Layout) -> bool {
        let size = lay.small;
        let mut changed = false;
        for (k, (name, tab)) in [("two weights", Tab::Toy), ("a network", Tab::Net)].into_iter().enumerate() {
            if plot::button(f, lay.tabs[k], name, size, self.tab == tab) && self.tab != tab {
                self.tab = tab;
                self.cut = true;
                changed = true;
            }
        }
        let busy = self.film().busy();
        let play = if busy {
            "❚❚ pause"
        } else if self.at_end() {
            "▶ again"
        } else {
            "▶ play"
        };
        if plot::button(f, lay.buttons[0], play, size, false) {
            if busy {
                let film = self.film_mut();
                film.playing = false;
                film.waiting = false;
            } else {
                self.play();
            }
            changed = true;
        }
        if plot::button(f, lay.buttons[1], if lay.narrow { "restart" } else { "from the start" }, size, false) {
            match self.tab {
                Tab::Toy => self.restart_toy(),
                Tab::Net => self.restart_net(),
            }
            changed = true;
        }
        if self.tab == Tab::Net {
            let words = if lay.narrow { r"$\eta/2$, $2\lambda$" } else { r"halve $\eta$, double $\lambda$" };
            if plot::button(f, lay.buttons[2], words, size, false) {
                self.halve_and_double();
                changed = true;
            }
        }
        changed
    }

    fn draw_plane(&mut self, f: &mut Frame, lay: &Layout) -> Vec<Placed> {
        let half = (self.look.frame.get(PLANE) as f64).max(0.5);
        let p = Plot::new(lay.plane, [-half, half], [-half, half]);
        if p.tiny() {
            return Vec::new();
        }
        let (min, max) = p.rect;
        let size = lay.small;
        p.backing(f, 6.0, 6.0, 6.0, 6.0);
        let mut strokes = Vec::new();
        let title = if lay.narrow { "two weights; the loss depends on their direction alone" } else { r"two weights $w = (w_1, w_2)$, whose loss depends on their direction alone: gradient descent from above" };
        f.text_px(lay.plane_head, title, lay.text, pal::TEXT, Align2::LEFT_BOTTOM);
        // the lines of equal loss: rays from 0, every 15°, with the loss on them
        let edge = |a: f64| {
            let (s, c) = a.sin_cos();
            let k = half / c.abs().max(s.abs()).max(1e-9);
            (k * c, k * s)
        };
        for k in 0..24 {
            let a = BEST + k as f64 * PI64 / 12.0;
            let shade = (toy_loss(a) / 2.0) as f32;
            let c = if k == 0 { pal::TEXT.a(0.65) } else { pal::DIM.a(0.10 + 0.22 * shade) };
            p.path(f, AXES, [(0.0, 0.0), edge(a)], if k == 0 { 1.4 } else { 1.0 }, c);
        }
        // the axes, lightly
        p.path(f, AXES, [(-half, 0.0), (half, 0.0)], 1.0, pal::DIM.a(0.35));
        p.path(f, AXES, [(0.0, -half), (0.0, half)], 1.0, pal::DIM.a(0.35));
        f.text_px(Vec2::new(max.x - 4.0, p.px_y(0.0) + 3.0), "$w_1$", size, pal::DIM, Align2::RIGHT_TOP);
        f.text_px(Vec2::new(p.px_x(0.0) + 5.0, min.y + 3.0), "$w_2$", size, pal::DIM, Align2::LEFT_TOP);
        for v in [1.0, 2.0, 3.0, 4.0, 5.0] {
            if v < half * 0.95 {
                let x = p.px_x(v);
                plot::line_of(f, AXES, Vec2::new(x, p.px_y(0.0) - 3.0), Vec2::new(x, p.px_y(0.0) + 3.0), 1.0, pal::DIM.a(0.6));
                f.text_px(Vec2::new(x, p.px_y(0.0) + 4.0), short(v), size - 1.0, pal::DIM.a(0.8), Align2::CENTER_TOP);
            }
        }
        // the names of the lowest and highest loss, at the edge
        let mut tags = Vec::new();
        let name_at = |a: f64, inset: f64| {
            let (x, y) = edge(a);
            p.at(x * inset, y * inset)
        };
        tags.push(ChartLabel { at: name_at(BEST, 0.86), align: Align2::CENTER_CENTER, text: "lowest loss, 0".to_string(), color: pal::TEXT });
        tags.push(ChartLabel { at: name_at(BEST + PI64, 0.86), align: Align2::CENTER_CENTER, text: "highest, 2".to_string(), color: pal::DIM });
        // the length at which growth and decay balance, now
        let head = self.toy_shown;
        let color = SGD;
        let circle = between(head.max(BALANCE_FROM), |t| self.toy_run.balance(t).unwrap_or(f64::NAN));
        if let Some(b) = (self.toy.lambda > 0.0 && circle.is_finite()).then_some(circle) {
            if b.is_finite() && b < half * 1.5 {
                let pts: Vec<Vec2> = (0..=96).map(|i| (i as f64 / 96.0 * std::f64::consts::TAU).sin_cos()).map(|(s, c)| p.at(b * c, b * s)).collect();
                ink(&mut strokes, FORMULA, true, &pts, 1.3, color.mix(pal::WHITE, 0.35));
                tags.push(ChartLabel {
                    at: p.at(b * (BEST + 0.55 * PI64).cos(), b * (BEST + 0.55 * PI64).sin()) + Vec2::new(0.0, -2.0),
                    align: Align2::CENTER_BOTTOM,
                    text: format!(r"balance $|w| = {}$", sig(b)),
                    color: color.mix(pal::WHITE, 0.35),
                });
            }
        }
        // the heads, gliding to where the runs are (each step is random: drawn as they
        // come, they would move in starts)
        let mut goal: Vec<f32> = Vec::with_capacity(4);
        for run in self.toy_plain.iter().chain(std::iter::once(&self.toy_run)) {
            let w = run.at(head);
            let at = p.at(w[0], w[1]).max(min).min(max);
            goal.extend([at.x, at.y]);
        }
        if self.snap_now || self.heads.shown.len() != goal.len() {
            self.heads.snap(&goal);
        } else {
            self.glides_moving |= self.heads.follow(&goal, self.dt_now, false);
        }
        let glided: Vec<Vec2> = self.heads.shown.chunks_exact(2).map(|c| Vec2::new(c[0], c[1])).collect();
        // the paths: faded the further back they are
        let plain = self.toy_plain.is_some();
        for (run, kind, c, width, k) in [(self.toy_plain.as_ref(), PATH_FAINT, pal::MUTED, 1.0, 0), (Some(&self.toy_run), PATH_MAIN, color, 1.3, plain as usize)] {
            let Some(run) = run else {
                // (an empty path, so that the batches keep their places)
                continue;
            };
            let end = (head.floor() as usize).min(run.steps());
            let span = ((head + 1.0).ln()).max(1e-9);
            let mut pts: Vec<(Vec3, Color)> = Vec::with_capacity(end + 2);
            for s in 0..=end {
                let w = run.w[s];
                let age = ((s as f64 + 1.0).ln() / span).clamp(0.0, 1.0);
                let at = p.at(w[0], w[1]).max(min).min(max);
                pts.push((f.hud(at), c.a((0.18 + 0.82 * age * age) as f32)));
            }
            if let Some(end) = glided.get(k) {
                pts.push((f.hud(*end), c));
            }
            let batch = plot::apart(f, kind, false);
            for (q, col) in pts {
                batch.push(q, width, col);
            }
            batch.brk();
        }
        // the heads, and their names
        let mut heads = Vec::new();
        if plain {
            heads.push((glided.first().copied().unwrap_or(min), pal::MUTED.mix(pal::WHITE, 0.3), "no decay".to_string()));
        }
        let name = if self.toy.lambda > 0.0 { format!(r"decay $\lambda = {}$", short(self.toy.lambda)) } else { "no decay".to_string() };
        heads.push((glided.last().copied().unwrap_or(min), color, name));
        let dot = if lay.narrow { 6.0 } else { 7.0 };
        for (at, c, _) in &heads {
            let h = f.hud(at.max(min).min(max));
            dots(f).push(h, dot, *c);
        }
        for (at, c, name) in heads.into_iter().rev() {
            tags.push(ChartLabel { at: at.max(min).min(max) + Vec2::new(8.0, -6.0), align: Align2::LEFT_BOTTOM, text: name, color: c });
        }
        let mut now = Vec::new();
        self.labels_moving |= place(f, 0, tags, size, (min, max), &self.placed, &mut now);
        self.emit(f, strokes);
        now
    }

    /// Draw the lines, each gliding to where it is going (lines are matched from frame
    /// to frame by the order they are drawn in: a new picture's glide from the old).
    fn emit(&mut self, f: &mut Frame, strokes: Vec<Stroke>) {
        for s in strokes {
            let k = self.glide_at;
            self.glide_at += 1;
            if self.glides.len() <= k {
                self.glides.push(Glide::default());
            }
            let flat: Vec<f32> = s.pts.iter().flat_map(|p| [p.x, p.y]).collect();
            let g = &mut self.glides[k];
            if self.snap_now {
                g.snap(&flat);
            } else {
                self.glides_moving |= g.follow(&flat, self.dt_now, false);
            }
            let pts: Vec<Vec2> = g.shown.chunks_exact(2).map(|c| Vec2::new(c[0], c[1])).collect();
            line(f, s.kind, s.dashed, &pts, s.width, s.color);
        }
    }
}

impl Decay {
    /// A vertical line at the step shown.
    fn now_line(f: &mut Frame, p: &Plot, head: f64) {
        if head >= 1.0 {
            let x = p.px_x(head.log10());
            if x >= p.rect.0.x && x <= p.rect.1.x {
                plot::line_of(f, NOW, Vec2::new(x, p.rect.0.y), Vec2::new(x, p.rect.1.y), 1.0, pal::TEXT.a(0.3));
            }
        }
    }

    fn draw_toy_charts(&mut self, f: &mut Frame, lay: &Layout, now: &mut Vec<Placed>) {
        let n = TOY_STEPS;
        let head = self.toy_shown;
        let x = [0.0, (n as f64).log10()];
        let size = lay.small;
        let color = SGD;
        let faint = pal::MUTED;
        let soft = color.mix(pal::WHITE, 0.35);
        let runs: Vec<(&ToyRun, bool)> = self.toy_plain.iter().map(|r| (r, false)).chain(std::iter::once((&self.toy_run, true))).collect();
        let mut strokes = Vec::new();
        // the length
        let top = (self.look.frame.get(TOY_LEN) as f64).max(0.5);
        let p = Plot::new(lay.toy_charts[0], x, [0.0, top]);
        if !p.tiny() {
            chart(f, lay, &p, r"the length $|w|$", n, false, short);
            let mut tags = Vec::new();
            for (run, main) in &runs {
                let (pts, _) = curve(&p, n, head, false, |s| Some(between(s, |t| run.len(t))));
                ink(&mut strokes, if *main { CURVES_MAIN } else { CURVES_FAINT }, false, &pts, if *main { 2.0 } else { 1.5 }, if *main { color } else { faint });
                if let Some(end) = pts.last() {
                    let text = if *main && self.toy.lambda > 0.0 { format!(r"$\lambda = {}$", short(self.toy.lambda)) } else { "no decay".to_string() };
                    tags.push(ChartLabel { at: *end + Vec2::new(-4.0, -5.0), align: Align2::RIGHT_BOTTOM, text, color: if *main { color } else { faint.mix(pal::WHITE, 0.3) } });
                }
            }
            if self.toy.lambda > 0.0 {
                let run = &self.toy_run;
                let (pts, any) = curve(&p, n, head, false, |s| if s < BALANCE_FROM { None } else { Some(between(s, |t| run.balance(t).unwrap_or(f64::NAN))).filter(|v| v.is_finite()) });
                ink(&mut strokes, FORMULA, true, &pts, 1.3, if any { soft } else { Color::CLEAR });
                if let (true, Some(end)) = (any, pts.last()) {
                    let text = if lay.narrow { "balance" } else { "where growth and decay balance" };
                    tags.push(ChartLabel { at: *end + Vec2::new(-4.0, 5.0), align: Align2::RIGHT_TOP, text: text.to_string(), color: soft });
                }
            }
            Self::now_line(f, &p, head);
            self.labels_moving |= place(f, 1, tags, size, p.rect, &self.placed, now);
        }
        // the turn
        let (lo, hi) = (self.look.frame.get(TOY_TURN) as f64, self.look.frame.get(TOY_TURN + 1) as f64);
        let p = Plot::new(lay.toy_charts[1], x, [lo, hi]);
        if !p.tiny() {
            let title = if lay.narrow { "the angle turned per step" } else { "the angle turned per step (root mean square over the last quarter of the steps)" };
            chart(f, lay, &p, title, n, true, |y| format!("{}°", short(10f64.powf(y))));
            let mut tags = Vec::new();
            for (run, main) in &runs {
                let (pts, _) = curve(&p, n, head, true, |s| Some(between(s, |t| run.turn(t.max(1)).to_degrees())));
                ink(&mut strokes, if *main { CURVES_MAIN } else { CURVES_FAINT }, false, &pts, if *main { 2.0 } else { 1.5 }, if *main { color } else { faint });
            }
            if let Some(b) = balance_turn(self.toy.eta, self.toy.lambda) {
                let y = b.to_degrees().log10();
                let (a, c) = (p.at(x[0], y), p.at(x[1], y));
                ink(&mut strokes, FORMULA, true, &[a, c], 1.3, soft);
                tags.push(ChartLabel { at: Vec2::new(c.x - 4.0, c.y - 3.0), align: Align2::RIGHT_BOTTOM, text: format!(r"$\arccos(1 - \eta\lambda) = {}$", degrees(b)), color: soft });
            }
            if self.toy_plain.is_some() || self.toy.lambda == 0.0 {
                let (pts, _) = curve(&p, n, n as f64, true, |s| Some(slowing_turn(s).to_degrees()).filter(|d| d.log10() <= p.y[1]));
                ink(&mut strokes, FORMULA, true, &pts, 1.2, faint.mix(pal::WHITE, 0.3));
                if let Some(end) = pts.last() {
                    tags.push(ChartLabel { at: *end + Vec2::new(-4.0, 4.0), align: Align2::RIGHT_TOP, text: r"$1/\sqrt{2t}$".to_string(), color: faint.mix(pal::WHITE, 0.3) });
                }
            }
            Self::now_line(f, &p, head);
            self.labels_moving |= place(f, 2, tags, size, p.rect, &self.placed, now);
        }
        self.emit(f, strokes);
    }

    /// The name of the faint run, as the charts give it.
    fn compare_name(&self) -> String {
        match self.compare {
            Some(c) if c.opt == self.net.opt && c.lambda == 0.0 && self.net.lambda > 0.0 => "no decay".to_string(),
            Some(c) if c.opt == Opt::Adam && self.net.opt == Opt::Adam && c.coupled != self.net.coupled && c.eta == self.net.eta && c.lambda == self.net.lambda => {
                c.name().to_string()
            }
            Some(c) => c.label(),
            None => String::new(),
        }
    }

    fn draw_net(&mut self, f: &mut Frame, lay: &Layout, now: &mut Vec<Placed>) {
        let n = NET_STEPS;
        let head = self.net_shown;
        let x = [0.0, (n as f64).log10()];
        let size = lay.small;
        let s = self.net;
        let muon = s.opt == Opt::Muon;
        // (under Muon, layer 1 takes Adam, as in Muon's reference: it is left out)
        let shown: Vec<usize> = if muon { vec![1, 2] } else { vec![0, 1, 2] };
        let runs: Vec<(&NetRun, bool)> = self.compared.iter().map(|r| (r, false)).chain(std::iter::once((&self.run, true))).collect();
        let mut strokes = Vec::new();
        let tone = |r: &NetRun, main: bool, l: usize| {
            let c = shade(r.setup.opt.color(), l);
            if main {
                c
            } else {
                c.mix(pal::MUTED, 0.55).a(0.7)
            }
        };
        // the size of each layer's weights
        let (lo, hi) = (self.look.frame.get(NET_SIZE) as f64, self.look.frame.get(NET_SIZE + 1) as f64);
        let p = Plot::new(lay.net_charts[0], x, [lo, hi]);
        if !p.tiny() {
            let title = match (muon, lay.narrow) {
                (true, false) => r"the largest singular value $\sigma_{\max}(W)$ of each of Muon's matrices",
                (true, true) => r"$\sigma_{\max}(W)$ of Muon's matrices",
                (false, false) => r"the length $|W|$ of each hidden layer's weights",
                (false, true) => r"the length $|W|$ of each layer's weights",
            };
            chart(f, lay, &p, title, n, true, |y| short(10f64.powf(y)));
            let mut tags = Vec::new();
            for (r, main) in &runs {
                let kind = if *main { CURVES_MAIN } else { CURVES_FAINT };
                for &l in &shown {
                    let (pts, _) = curve(&p, n, head, true, |st| {
                        let v = between(st, |t| if muon { r.top_at(l, t) } else { r.len_at(l, t) });
                        v.is_finite().then_some(v)
                    });
                    ink(&mut strokes, kind, false, &pts, if *main { 2.2 } else { 1.4 }, tone(r, *main, l));
                }
                if r.setup.all {
                    let (pts, _) = curve(&p, n, head, true, |st| Some(between(st, |t| r.len_at(DEPTH, t))));
                    ink(&mut strokes, kind, false, &pts, if *main { 1.6 } else { 1.2 }, if *main { READOUT } else { READOUT.a(0.4) });
                    if *main {
                        if let Some(end) = pts.last() {
                            tags.push(ChartLabel { at: *end + Vec2::new(-4.0, 4.0), align: Align2::RIGHT_TOP, text: "read-out".to_string(), color: READOUT });
                        }
                    }
                }
            }
            // where growth and decay balance (gradient descent with decay)
            if s.opt == Opt::Sgd && s.lambda > 0.0 {
                let r = &self.run;
                for &l in &shown {
                    let (pts, any) = curve(&p, n, head, true, |st| if st < BALANCE_FROM { None } else { Some(between(st, |t| r.balance(l, t).unwrap_or(f64::NAN))).filter(|v| v.is_finite()) });
                    ink(&mut strokes, FORMULA, true, &pts, 1.2, if any { shade(SGD, l).mix(pal::WHITE, 0.25).a(0.8) } else { Color::CLEAR });
                }
            }
            // Muon's ball: the radius, and how far out the start lets it be by step t
            if muon {
                if let Some((radius, _)) = muon_bound(s.eta, s.lambda, 0.0, 0.0) {
                    let y = p.px_y(radius.log10());
                    ink(&mut strokes, BOUND, true, &[Vec2::new(p.rect.0.x, y), Vec2::new(p.rect.1.x, y)], 1.4, LIMIT);
                    tags.push(ChartLabel { at: Vec2::new(p.rect.1.x - 4.0, y - 3.0), align: Align2::RIGHT_BOTTOM, text: format!(r"radius $1.2024/\lambda = {}$", sig(radius)), color: LIMIT });
                    let exact = 1.0 / s.lambda;
                    let ye = p.px_y(exact.log10());
                    ink(&mut strokes, BOUND, true, &[Vec2::new(p.rect.0.x, ye), Vec2::new(p.rect.1.x, ye)], 1.0, LIMIT.a(0.5));
                    tags.push(ChartLabel { at: Vec2::new(p.rect.0.x + 6.0, ye + 3.0), align: Align2::LEFT_TOP, text: format!(r"$1/\lambda = {}$, with exact orthogonalisation", sig(exact)), color: LIMIT.a(0.8) });
                    let start = shown.iter().map(|&l| self.run.top_at(l, 0)).fold(0.0, f64::max);
                    let (pts, _) = curve(&p, n, n as f64, true, |st| muon_bound(s.eta, s.lambda, start, st).map(|b| b.1));
                    ink(&mut strokes, BOUND, true, &pts, 1.2, LIMIT.a(0.7));
                }
            }
            // which run is which
            let mut key = vec![(s.label(), s.opt.color())];
            if let Some(c) = self.compare {
                key.push((format!("grey: {}", self.compare_name()), shade(c.opt.color(), 0).mix(pal::MUTED, 0.55)));
            }
            let rows = key.len();
            for (k, (text, c)) in key.into_iter().enumerate() {
                tags.insert(k, ChartLabel { at: Vec2::new(p.rect.0.x + 6.0, p.rect.0.y + 4.0 + k as f32 * (size + 7.0)), align: Align2::LEFT_TOP, text, color: c });
            }
            // the layers by their shades, in a row under the runs' names
            let mut x0 = p.rect.0.x + 6.0;
            let y0 = p.rect.0.y + 4.0 + rows as f32 * (size + 7.0);
            for (k, &l) in shown.iter().enumerate() {
                let text = format!("layer {}", l + 1);
                let w = f.text_size(&text, size).x + 14.0;
                tags.insert(rows + k, ChartLabel { at: Vec2::new(x0, y0), align: Align2::LEFT_TOP, text, color: shade(s.opt.color(), l) });
                x0 += w;
            }
            Self::now_line(f, &p, head);
            self.labels_moving |= place(f, 3, tags, size, p.rect, &self.placed, now);
        }
        // the angle turned per step
        let (lo, hi) = (self.look.frame.get(NET_TURN) as f64, self.look.frame.get(NET_TURN + 1) as f64);
        let p = Plot::new(lay.net_charts[1], x, [lo, hi]);
        if !p.tiny() {
            let title = if lay.narrow { "the angle each layer turns per step" } else { "the angle each layer turns per step (root mean square over the last quarter of the steps)" };
            chart(f, lay, &p, title, n, true, |y| format!("{}°", short(10f64.powf(y))));
            let mut tags = Vec::new();
            for (r, main) in &runs {
                let kind = if *main { CURVES_MAIN } else { CURVES_FAINT };
                for &l in &shown {
                    let (pts, _) = curve(&p, n, head, true, |st| Some(between(st, |t| r.turn(l, t.max(1)).to_degrees())));
                    ink(&mut strokes, kind, false, &pts, if *main { 2.0 } else { 1.4 }, tone(r, *main, l));
                }
            }
            let mut levels: Vec<f64> = Vec::new();
            for (r, main) in &runs {
                let rs = r.setup;
                if rs.opt != Opt::Sgd {
                    continue;
                }
                let soft = SGD.mix(pal::WHITE, 0.35).a(if *main { 1.0 } else { 0.6 });
                if let Some(b) = balance_turn(rs.eta, rs.lambda) {
                    if levels.iter().any(|l| (l - b).abs() < 1e-12) {
                        continue;
                    }
                    levels.push(b);
                    let y = b.to_degrees().log10();
                    let (a, c) = (p.at(x[0], y), p.at(x[1], y));
                    ink(&mut strokes, FORMULA, true, &[a, c], 1.3, soft);
                    tags.push(ChartLabel { at: Vec2::new(c.x - 4.0, c.y - 3.0), align: Align2::RIGHT_BOTTOM, text: format!(r"$\arccos(1 - \eta\lambda) = {}$", degrees(b)), color: soft });
                } else {
                    let (pts, _) = curve(&p, n, n as f64, true, |st| Some(slowing_turn(st).to_degrees()).filter(|d| d.log10() <= p.y[1]));
                    ink(&mut strokes, FORMULA, true, &pts, 1.2, soft);
                    if let Some(end) = pts.last() {
                        tags.push(ChartLabel { at: *end + Vec2::new(-4.0, 4.0), align: Align2::RIGHT_TOP, text: r"$1/\sqrt{2t}$".to_string(), color: soft });
                    }
                }
            }
            Self::now_line(f, &p, head);
            self.labels_moving |= place(f, 4, tags, size, p.rect, &self.placed, now);
        }
        // the loss
        let (lo, hi) = (self.look.frame.get(NET_LOSS) as f64, self.look.frame.get(NET_LOSS + 1) as f64);
        let p = Plot::new(lay.net_charts[2], x, [lo, hi]);
        if !p.tiny() {
            let title = if lay.narrow { "the loss of the mini-batches" } else { "the loss of the mini-batches (mean over the last quarter of the steps)" };
            chart(f, lay, &p, title, n, true, |y| short(10f64.powf(y)));
            for (r, main) in &runs {
                let (pts, _) = curve(&p, n, head, true, |st| Some(between(st, |t| r.batch_loss(t.max(1)))));
                ink(&mut strokes, if *main { CURVES_MAIN } else { CURVES_FAINT }, false, &pts, if *main { 2.0 } else { 1.4 }, tone(r, *main, 0));
            }
            Self::now_line(f, &p, head);
        }
        self.emit(f, strokes);
    }
}

impl Scene for Decay {
    fn view(&self) -> View {
        View::Flat { center: Vec2::ZERO, half_height: 1.0 }
    }

    fn settings(&self, s: &mut ViewSettings) {
        s.grid = false;
        s.bloom = s.bloom.min(0.2);
    }

    fn animated(&self) -> bool {
        self.film().busy() || self.look.moving || self.labels_moving || self.glides_moving
    }

    fn pointer_hint(&self) -> Option<&'static str> {
        Some("buttons above the picture: click")
    }

    fn ui(&mut self, ui: &mut egui::Ui, _: &Gfx) {
        ui::note(
            ui,
            r"Weights that feed a normalisation layer, with nothing added in between, can be scaled without changing the loss (scene normgrad): only their direction matters. So their gradient $g$ is at right angles to them, $w \cdot g = 0$, and shrinks as they grow, $|g| \propto 1/|w|$. A step of gradient descent with weight decay $\lambda$, $w' = (1 - \eta\lambda)\,w - \eta g$, then changes their length by Pythagoras,",
        );
        ui::equation(ui, r"|w'|^2 = (1 - \eta\lambda)^2\,|w|^2 + \eta^2 |g|^2");
        ui::note(
            ui,
            r"and turns them by an angle $\Delta$ with $\tan\Delta = \eta|g| / ((1 - \eta\lambda)|w|)$. As $|g|$ goes as $1/|w|$, the turn goes as $\eta/|w|^2$: the effective learning rate. Without decay the length only grows, so the turning slows by itself. With decay the length settles where growth and shrinkage balance, and wherever the length is steady, exactly,",
        );
        ui::equation(ui, r"\cos\Delta = 1 - \eta\lambda");
        ui::note(ui, "whatever the gradient: for such weights the step and the decay act together, through their product.");

        ui::section(ui, "Things to try");
        for (k, t) in TRIALS.iter().enumerate() {
            if ui::experiment(ui, t.button, t.caption) {
                self.set_up(k);
            }
        }

        match self.tab {
            Tab::Toy => self.toy_controls(ui),
            Tab::Net => self.net_controls(ui),
        }
        self.refresh();

        ui::section(ui, "Reading the picture");
        ui::key(
            ui,
            &[
                (SGD, "cyan: gradient descent"),
                (ADAM, "orange: Adam"),
                (MUON, "blue: Muon"),
                (pal::MUTED, "grey: the run compared with it, named on the chart (on the plane: the same random pushes without decay)"),
                (shade(SGD, 1), "lighter further in: layers 1, 2 and 3 of the network"),
                (SGD.mix(pal::WHITE, 0.35), "dashed: what the formulas say: where growth and decay balance, the turn there, and without decay the turn's slowing"),
                (LIMIT, "yellow dashed: the ball the decay keeps Muon's matrices in"),
                (READOUT, "white: the read-out's length, when the decay acts on it too"),
                (pal::DIM, "rays on the plane: the toy's lines of equal loss; the brightest, where it is lowest"),
            ],
        );
        ui::section(ui, "Words");
        ui::term(ui, "Normalisation layer:", "a layer that rescales what passes through it to a standard size; here batch normalisation, each unit standardised over the examples of the mini-batch (scene normgrad).");
        ui::term(ui, "Weight decay:", r"a shrinking of the weights at each step: $\lambda w$ added to the gradient, or $\eta\lambda w$ taken off the weights. For gradient descent the two are the same rule.");
        ui::term(
            ui,
            "Scale-free weights:",
            r"weights whose length does not change the loss: here each hidden layer's $W$ (with batch normalisation, each of its rows on its own too). Not the gains $\gamma$, which multiply the normalisation's output, nor the read-out. A bias between $W$ and a batch normalisation would not spoil this, as the normalisation takes the mean off; with layer normalisation only the whole matrix, with no bias, is scale-free.",
        );
        ui::term(ui, "Effective learning rate:", r"$\eta/|w|^2$: the step that the direction of $w$ takes, as if $w$ had length 1.");
        ui::term(ui, "Turn per step:", r"the angle between the weights before a step and after it; the charts show its root mean square over the last quarter of the steps.");
        ui::term(ui, "Unit gradient:", r"$|g|\,|w|$, the gradient's length the weights would have at length 1. The balance is where $|w|^4 = \eta L / (\lambda(2 - \eta\lambda))$, with $L$ its mean square.");
        ui::term(ui, "AdamW:", r"Adam with the decay taken off the weights, $\theta \leftarrow \theta - \eta\lambda\theta$, instead of added to the gradient, where Adam would divide it by $\sqrt{\hat v}$ with the rest (Loshchilov and Hutter, 2019). Here, as in PyTorch, the decay is multiplied by the learning rate $\eta$.");
        ui::term(ui, "Largest singular value:", r"$\sigma_{\max}(W)$, the most $W$ can stretch a vector (scene svd). Each of Muon's steps here has a largest singular value of at most $1.2024\,\eta$.");
        ui::term(ui, r"Guard $\varepsilon$:", r"$10^{-5}$, added to each unit's variance in the normalisation so that it never divides by 0. It makes the network's layers scale-free only nearly: their gradient is at right angles to them to within the angle the numbers give.");
        ui::term(ui, "The network:", r"3 hidden layers of 16 units, each $\tanh(\gamma \odot \operatorname{BN}(W x))$ with a learnt gain $\gamma$ and no biases, then a read-out of 3 units; 120 points of three spiral arms, cross-entropy loss, batches of 32, 4 000 steps.");

        ui::section(ui, "Numbers");
        self.numbers(ui);
        ui::see_also(
            ui,
            &[
                ("normgrad", "why the gradient is at right angles to such weights"),
                ("adam", "the two kinds of decay on two weights"),
                ("muon", "Muon's step, and its polynomial"),
                ("eos", "how large a step a network bears"),
            ],
        );
    }

    fn frame(&mut self, f: &mut Frame) {
        hold_batches(f);
        let lay = Layout::of(f.view_size(), f.top_inset);
        // (a canvas squeezed to a sliver is one the controls have opened over, on a phone)
        let tiny = lay.view.x < 160.0 || lay.view.y < 120.0;
        if !tiny && self.buttons(f, &lay) {
            self.refresh();
        }
        self.refresh();
        let budget = clock::frame_budget(f.real_dt, 0.004);
        self.advance(f.dt, &budget);
        let snap = (self.canvas != Some(lay.view) || self.cut) && !tiny;
        if snap {
            self.canvas = Some(lay.view);
        }
        self.follow(f.real_dt, snap);
        if tiny {
            return;
        }
        self.labels_moving = false;
        self.glides_moving = false;
        self.glide_at = 0;
        self.dt_now = f.real_dt;
        self.snap_now = snap;
        let mut now = Vec::new();
        match self.tab {
            Tab::Toy => {
                now = self.draw_plane(f, &lay);
                if !lay.compact {
                    self.draw_toy_charts(f, &lay, &mut now);
                }
            }
            Tab::Net => self.draw_net(f, &lay, &mut now),
        }
        self.glides.truncate(self.glide_at);
        self.placed = now;
        self.cut = false;
    }
}

impl Decay {
    fn toy_controls(&mut self, ui: &mut egui::Ui) {
        ui::section(ui, "The two weights");
        let t = &mut self.toy;
        let mut eta = t.eta as f32;
        if ui::slider_log(ui, r"step size $\eta$", &mut eta, TOY_LEAST_ETA as f32..=TOY_MOST_ETA as f32) {
            t.eta = eta as f64;
        }
        let mut lambda = t.lambda as f32;
        if ui::slider_log_or_zero(ui, r"weight decay $\lambda$", &mut lambda, TOY_LEAST_LAMBDA as f32, TOY_MOST_LAMBDA as f32) {
            t.lambda = lambda as f64;
        }
        ui::note(
            ui,
            r"The loss is $1 - \cos(\theta - 60°)$, $\theta$ the direction of $w$. Each step's gradient also gets a random push along the circle, new at each step, as a mini-batch's would: the slope along the circle of radius 1 plus noise of standard deviation 1. The grey run has the same pushes and no decay.",
        );
    }

    fn net_controls(&mut self, ui: &mut egui::Ui) {
        ui::section(ui, "The network");
        let mut which = Opt::ALL.iter().position(|o| *o == self.net.opt).unwrap_or(0);
        if ui::choice(ui, &["gradient descent", "Adam", "Muon"], &mut which) {
            let opt = Opt::ALL[which.min(2)];
            self.net = NetSetup { seed: self.net.seed, ..NetSetup::usual(opt) };
            self.compare = None;
        }
        let s = &mut self.net;
        let (lo, hi) = s.opt.etas();
        let mut eta = s.eta as f32;
        if ui::slider_log(ui, r"step size $\eta$", &mut eta, lo as f32..=hi as f32) {
            s.eta = eta as f64;
        }
        let (llo, lhi) = s.opt.lambdas();
        let mut lambda = s.lambda as f32;
        if ui::slider_log_or_zero(ui, r"weight decay $\lambda$", &mut lambda, llo as f32, lhi as f32) {
            s.lambda = lambda as f64;
        }
        match s.opt {
            Opt::Sgd => ui::note(ui, r"For gradient descent the decay added to the gradient and the decay taken off the weights are one rule: $w - \eta(g + \lambda w)$."),
            Opt::Adam => {
                let mut kind = s.coupled as usize;
                if ui::choice(ui, &["taken off the weights (AdamW)", "added to the gradient (L2)"], &mut kind) {
                    s.coupled = kind == 1;
                }
                ui::note(ui, r"Adam with Kingma and Ba's defaults ($\beta_1 = 0.9$, $\beta_2 = 0.999$, $\varepsilon = 10^{-8}$).");
            }
            Opt::Muon => ui::note(
                ui,
                r"Muon as its reference code has it, with the decay taken off its matrices, $W \leftarrow (1 - \eta\lambda) W$. It takes layers 2 and 3; layer 1, which reads the data, and the read-out and gains take Adam, without decay, as in the reference: the charts leave layer 1 out.",
            ),
        }
        if s.opt != Opt::Muon {
            ui::checkbox(ui, &mut s.all, "decay the read-out and the gains too");
            ui::note(ui, r"The read-out and the gains feed no normalisation: on them the decay does pull towards small weights, and changes what the network computes. With the second experiment's settings, it ends the read-out 3.90 long instead of 4.89, and the loss on all the data at 0.58 instead of 0.47 (this seed).");
        }
        ui.horizontal_wrapped(|ui| {
            if ui.button(ui::label_text(ui, "new start and batches", 12.5, None, false)).clicked() {
                self.net.seed = self.net.seed.wrapping_add(1);
                if let Some(c) = self.compare.as_mut() {
                    c.seed = self.net.seed;
                }
            }
            if self.compare.is_some() && ui.button(ui::label_text(ui, "forget the grey run", 12.5, None, false)).clicked() {
                self.compare = None;
            }
        });
    }

    /// The read-outs.
    fn numbers(&mut self, ui: &mut egui::Ui) {
        match self.tab {
            Tab::Toy => {
                let t = self.toy_shown.round() as usize;
                let r = &self.toy_run;
                ui::readout(ui, "step", count(t));
                let label = if self.toy.lambda > 0.0 { format!(r"with decay $\lambda = {}$: $|w|$, turn per step", short(self.toy.lambda)) } else { r"no decay: $|w|$, turn per step".to_string() };
                ui::readout_colored(ui, &label, format!("{}, {}", sig(r.len(t)), if t > 0 { degrees(r.turn(t)) } else { "–".to_string() }), SGD);
                if let (Some(b), Some(a)) = (r.balance(t.max(1)), balance_turn(self.toy.eta, self.toy.lambda)) {
                    ui::readout(ui, r"  the balance: $|w|$, $\arccos(1 - \eta\lambda)$", format!("{}, {}", sig(b), degrees(a)));
                }
                ui::readout(ui, r"  effective learning rate $\eta/|w|^2$", sig(self.toy.eta / r.len(t).powi(2)));
                if let Some(p) = &self.toy_plain {
                    ui::readout_colored(ui, r"no decay: $|w|$, turn per step", format!("{}, {}", sig(p.len(t)), if t > 0 { degrees(p.turn(t)) } else { "–".to_string() }), pal::MUTED.mix(pal::WHITE, 0.3));
                    if t > 0 {
                        ui::readout(ui, r"  $1/\sqrt{2t}$", degrees(slowing_turn(t as f64)));
                    }
                }
                ui::readout(ui, r"Pythagoras: $|w'|^2$ against the formula, over the run, to within", format!("${}$", fmt(r.pythagoras.max(1e-18) as f32)));
            }
            Tab::Net => {
                let t = (self.net_shown.round() as usize).min(self.run.t);
                let r = &self.run;
                let muon = r.setup.opt == Opt::Muon;
                ui::readout(ui, "step", count(t));
                for l in 0..DEPTH {
                    let c = shade(r.setup.opt.color(), l);
                    let size = if muon { format!(r"$\sigma_{{\max}}$ {}", sig(r.top_at(l, t))) } else { format!(r"$|W|$ {}", sig(r.len_at(l, t))) };
                    let turn = if t > 0 { degrees(r.turn(l, t)) } else { "–".to_string() };
                    ui::readout_colored(ui, &format!("layer {}", l + 1), format!("{size}, turn {turn}"), c);
                    if let Some(b) = r.balance(l, t.max(1)) {
                        ui::readout(ui, "  where growth and decay balance", sig(b));
                    }
                }
                if let Some(a) = (r.setup.opt == Opt::Sgd).then(|| balance_turn(r.setup.eta, r.setup.lambda)).flatten() {
                    ui::readout(ui, r"$\arccos(1 - \eta\lambda)$", degrees(a));
                }
                if muon {
                    if let Some((radius, _)) = muon_bound(r.setup.eta, r.setup.lambda, 0.0, 0.0) {
                        ui::readout_colored(ui, r"the ball's radius $1.2024/\lambda$", sig(radius), LIMIT);
                    }
                }
                ui::readout_colored(ui, "read-out's length", sig(r.len_at(DEPTH, t)), READOUT);
                ui::readout(ui, "loss of recent mini-batches", if t > 0 { sig(r.batch_loss(t)) } else { "–".to_string() });
                let tilt = r.tilt.min(1.0).asin().to_degrees();
                ui::readout(ui, "gradient at right angles to the layers, within", if r.t > 0 { format!("{}°", num(tilt, 3)) } else { "–".to_string() });
                if let Some(c) = &self.compared {
                    let tc = t.min(c.t);
                    ui::readout_colored(ui, &format!("grey ({}): loss", self.compare_name()), if tc > 0 { sig(c.batch_loss(tc)) } else { "–".to_string() }, shade(c.setup.opt.color(), 0).mix(pal::MUTED, 0.55));
                }
                if r.broken {
                    ui::readout_colored(ui, "the run", "stopped: its numbers are no longer finite", pal::RED);
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Experiment k as its button sets it up, with every run played to its end.
    fn run(k: usize) -> Decay {
        let mut s = Decay::create();
        s.set_up(k);
        while !s.run.done() {
            s.run.step();
        }
        if let Some(c) = s.compared.as_mut() {
            while !c.done() {
                c.step();
            }
        }
        s.net_shown = NET_STEPS as f64;
        s.toy_shown = TOY_STEPS as f64;
        s
    }

    fn says(k: usize, what: &str) {
        assert!(TRIALS[k].caption.contains(what), "experiment {k} should say \"{what}\": {}", TRIALS[k].caption);
    }

    /// Degrees with two decimals, as the captions write a turn.
    fn deg2(rad: f64) -> String {
        num(rad.to_degrees(), 2)
    }

    /// The mean over the last quarter of a run of a layer's turn per step: the end of
    /// its curve.
    fn late(r: &NetRun, l: usize) -> f64 {
        r.turn(l, r.t)
    }

    /// The toy, step by step: the gradient is at right angles to w, and so |w'|² =
    /// (1 − ηλ)²|w|² + η²|g|² to rounding, and cos Δ = (1 − ηλ)|w|/|w'| exactly.
    #[test]
    fn the_toy_steps_by_pythagoras() {
        for s in [ToySetup::base(), ToySetup { lambda: 0.0, ..ToySetup::base() }, ToySetup { eta: 1.0, lambda: 0.1 }, ToySetup { eta: 0.003, lambda: 1e-4 }] {
            let r = ToyRun::of(s);
            assert!(r.pythagoras < 1e-13, "{s:?}: {}", r.pythagoras);
            for t in [0usize, 1, 17, 500, 2999] {
                let (a, b) = (r.w[t], r.w[t + 1]);
                let cos = (a[0] * b[0] + a[1] * b[1]) / (r.len(t) * r.len(t + 1));
                let want = (1.0 - s.eta * s.lambda) * r.len(t) / r.len(t + 1);
                assert!((cos - want).abs() < 1e-12, "{s:?} step {t}: {cos} {want}");
                let turn2 = r.turn2[t + 1] - r.turn2[t];
                assert!((turn2.sqrt() - cos.min(1.0).acos()).abs() < 1e-7, "{s:?} step {t}");
            }
            // without decay the length never falls
            if s.lambda == 0.0 {
                assert!((0..TOY_STEPS).all(|t| r.len(t + 1) >= r.len(t)));
            }
        }
    }

    /// The balance: the fixed point of |w'|² = (1 − ηλ)²|w|² + η²L/|w|², reached by
    /// iterating it; there cos Δ = 1 − ηλ, so Δ ≈ √(2ηλ); and without decay the turn
    /// falls as 1/√(2t) whatever η.
    #[test]
    fn the_balance_formulas_hold() {
        for (eta, lambda, l) in [(0.1, 0.01, 1.0), (0.2, 0.005, 36.0), (0.02, 0.5, 4.0), (1.0, 0.1, 0.3)] {
            let k = 1.0 - eta * lambda;
            let mut x: f64 = 0.3;
            for _ in 0..200_000 {
                x = k * k * x + eta * eta * l / x;
            }
            let b = balance_len(eta, lambda, l).unwrap();
            assert!((x.sqrt() - b).abs() < 1e-9 * b, "{eta} {lambda}: {} {b}", x.sqrt());
            // the angle there: tan Δ = η√L / (k|w|²)
            let turn = (eta * l.sqrt() / (k * x)).atan();
            assert!((turn - balance_turn(eta, lambda).unwrap()).abs() < 1e-9, "{eta} {lambda}");
            assert!((balance_turn(eta, lambda).unwrap() / (2.0 * eta * lambda).sqrt() - 1.0).abs() < 0.01 + eta * lambda);
            // the squared length relaxes as (2k² − 1)^t: about 1/(4ηλ) steps
            let rate = 2.0 * k * k - 1.0;
            assert!(((1.0 - rate) * relaxation(eta, lambda) as f64 - 1.0).abs() < 0.02 + eta * lambda * 2.0, "{eta} {lambda}");
        }
        for eta in [0.01, 0.1, 1.0] {
            let (l, mut x) = (2.0f64, 0.25f64);
            let mut last = 0.0;
            for _ in 0..1_000_000 {
                last = (eta * l.sqrt() / x).atan();
                x += eta * eta * l / x;
            }
            assert!((last / slowing_turn(1e6) - 1.0).abs() < 0.01, "{eta}: {last}");
        }
        assert!(balance_len(0.1, 0.0, 1.0).is_none() && balance_turn(0.1, 0.0).is_none() && balance_len(0.1, 0.01, f64::NAN).is_none());
    }

    /// The network's hidden layers feed batch normalisation: scaling one leaves the loss
    /// as it was and divides its gradient by the factor, and the gradient is at right
    /// angles to it (to within what the guard ε in the normalisation changes). The
    /// read-out feeds none, and is not so.
    #[test]
    fn the_hidden_layers_are_scale_free() {
        let mut r = NetRun::new(NetSetup::sgd());
        for _ in 0..300 {
            r.step();
        }
        assert!(r.tilt < 1e-3, "{}", r.tilt);
        let idx = r.problem.all();
        let loss = r.problem.grad(&mut r.model, &mut r.tape, &idx);
        for g in r.layers.into_iter().chain([r.readout]) {
            let before: Vec<f64> = r.model.groups()[g].grad.clone();
            let w = r.model.groups()[g].value.clone();
            let cos = dot(&w, &before) / (norm(&w) * norm(&before));
            let mut m = r.model.clone();
            m.groups_mut()[g].value.iter_mut().for_each(|v| *v *= 2.0);
            let scaled = r.problem.grad(&mut m, &mut r.tape, &idx);
            let after = &m.groups()[g].grad;
            let halved = after.iter().zip(&before).map(|(a, b)| (2.0 * a - b).abs()).fold(0.0, f64::max) / before.iter().fold(0.0f64, |a, b| a.max(b.abs()));
            if g == r.readout {
                assert!((scaled - loss).abs() > 0.01 * loss, "the read-out: {loss} {scaled}");
            } else {
                assert!(cos.abs() < 3e-4, "group {g}: {cos}");
                assert!((scaled - loss).abs() < 5e-5 * loss, "group {g}: {loss} {scaled}");
                assert!(halved < 2e-3, "group {g}: {halved}");
            }
        }
    }

    /// For gradient descent the decay added to the gradient is the decay taken off the
    /// weights: one step of each, from the same weights, agrees to rounding. The scene's
    /// own decay is the library's (all groups decayed): for gradient descent the coupled
    /// form, for Adam the decoupled one.
    #[test]
    fn the_scene_decays_as_the_library_does() {
        let base = NetRun::new(NetSetup::sgd());
        let mut idx_model = base.model.clone();
        let mut tape = Tape::new();
        let idx: Vec<usize> = (0..BATCH).collect();
        base.problem.grad(&mut idx_model, &mut tape, &idx);
        let lambda = 0.05;
        for (opt, wd, eta) in [(Opt::Sgd, WeightDecay::Coupled(lambda), 0.2), (Opt::Sgd, WeightDecay::Decoupled(lambda), 0.2), (Opt::Adam, WeightDecay::Decoupled(lambda), 0.01)] {
            let mut lib = idx_model.clone();
            let mut hyper = Hyper::default_for(opt.rule());
            hyper.lr = eta;
            hyper.weight_decay = wd;
            Optimizer::new(&lib, opt.rule(), hyper).step(&mut lib, eta);
            // the scene's way: every group decayed
            let mut mine = NetRun::new(NetSetup { opt, eta, lambda, all: true, ..NetSetup::sgd() });
            mine.model = idx_model.clone();
            let s = mine.setup;
            for (group, &on) in mine.model.groups_mut().iter_mut().zip(&mine.decayed) {
                assert!(on);
                if s.opt == Opt::Sgd || s.coupled {
                    for (g, v) in group.grad.iter_mut().zip(&group.value) {
                        *g += s.lambda * v;
                    }
                } else {
                    for v in group.value.iter_mut() {
                        *v -= s.eta * s.lambda * *v;
                    }
                }
            }
            mine.opt.step(&mut mine.model, eta);
            for (a, b) in lib.groups().iter().zip(mine.model.groups()) {
                for (x, y) in a.value.iter().zip(&b.value) {
                    assert!((x - y).abs() <= 1e-15 * x.abs().max(1.0), "{opt:?} {wd:?} {}: {x} {y}", a.name);
                }
            }
        }
    }

    /// Muon's matrices never leave the ball: the largest singular value stays under the
    /// bound of Chen, Li and Liu's Proposition 1 at every step, and every step's matrix
    /// has singular values of at most 1.2024.
    #[test]
    fn muon_stays_inside_its_ball() {
        let mut r = NetRun::new(NetSetup::usual(Opt::Muon));
        let s = r.setup;
        let mut most: f64 = 0.0;
        for t in 1..=1500 {
            r.step();
            for l in [1, 2] {
                let o = r.opt.ortho_out(r.layers[l]);
                let g = &r.model.groups()[r.layers[l]];
                most = most.max(optim::spectral_norm(o, g.rows, g.cols));
                let (_, bound) = muon_bound(s.eta, s.lambda, r.top_at(l, 0), t as f64).unwrap();
                assert!(r.top_at(l, t) <= bound * (1.0 + 1e-12), "layer {} step {t}: {} {bound}", l + 1, r.top_at(l, t));
            }
        }
        assert!(most <= MUON_MOST + 1e-9 && most > 1.0, "{most}");
        // (the square layers: Keller's shape factor is 1)
        assert_eq!((WIDTH, WIDTH), (r.model.groups()[r.layers[1]].rows, r.model.groups()[r.layers[1]].cols));
    }

    #[test]
    fn experiment_0_the_toy() {
        let s = run(0);
        assert_eq!(s.tab, Tab::Toy);
        let (r, p) = (&s.toy_run, s.toy_plain.as_ref().unwrap());
        let n = TOY_STEPS;
        assert_eq!((s.toy.eta, s.toy.lambda), (0.1, 0.01));
        says(0, &format!("to a length of {} at step 3 000 and still growing", sig(p.len(n))));
        assert!(p.len(n) > p.len(n - 300) && p.len(n - 300) > p.len(n - 600));
        says(0, &format!("its turn per step falls, to {}°, close to the $1/\\sqrt{{2t}} = {}°$", deg2(p.turn(n)), deg2(slowing_turn(n as f64))));
        assert!((p.turn(n) / slowing_turn(n as f64) - 1.0).abs() < 0.1);
        says(0, &format!("the length settles at {}, by the circle where growth and decay balance for the gradient measured ({})", sig(r.len(n)), sig(r.balance(n).unwrap())));
        // settled: from step 1 000 on within 5 % of where it ends, while the run without
        // decay grows by a third
        let worst = (1000..=n).map(|t| (r.len(t) / r.len(n) - 1.0).abs()).fold(0.0, f64::max);
        assert!(worst < 0.05, "not settled: {worst}");
        assert!(p.len(n) > 1.3 * p.len(1000));
        let b = balance_turn(0.1, 0.01).unwrap();
        says(0, &format!("the turn stays at {}", degrees(r.turn(n))));
        assert_eq!(degrees(r.turn(n)), degrees(b));
        assert!((r.turn(n) / b - 1.0).abs() < 0.03, "{} {b}", r.turn(n));
    }

    #[test]
    fn experiment_1_a_network_with_and_without_decay() {
        let s = run(1);
        assert_eq!(s.tab, Tab::Net);
        let (r, c) = (&s.run, s.compared.as_ref().unwrap());
        assert_eq!((r.setup.eta, r.setup.lambda, c.setup.lambda), (0.2, 0.005, 0.0));
        let tilt = r.tilt.max(c.tilt).asin().to_degrees();
        assert!(tilt < 0.03, "{tilt}");
        says(1, "to within 0.03°");
        let ends: Vec<String> = (0..DEPTH).map(|l| sig(c.len_at(l, NET_STEPS))).collect();
        says(1, &format!("from about 4.1 to {}, {} and {} by step 4 000", ends[0], ends[1], ends[2]));
        assert!((0..DEPTH).all(|l| (c.len_at(l, 0) - 4.1).abs() < 0.07 && (500..NET_STEPS).all(|t| c.len_at(l, t + 1) >= c.len_at(l, t))));
        let turns: Vec<f64> = (0..DEPTH).map(|l| late(c, l)).collect();
        let (lo, hi) = (turns.iter().cloned().fold(f64::INFINITY, f64::min), turns.iter().cloned().fold(0.0, f64::max));
        says(1, &format!("falls to between {}° and {}°", deg2(lo), deg2(hi)));
        // near the line: within 45 % of it at the end, from four times it at step 100
        assert!(turns.iter().all(|t| *t / slowing_turn(NET_STEPS as f64) < 1.45 && *t > slowing_turn(NET_STEPS as f64)));
        let worst = (0..DEPTH).map(|l| (r.len_at(l, NET_STEPS) / r.balance(l, NET_STEPS).unwrap() - 1.0).abs()).fold(0.0, f64::max);
        assert!(worst < 0.014 && worst > 0.01, "{worst}");
        says(1, "within 1.4 % of where growth and decay balance");
        let b: Vec<String> = (0..DEPTH).map(|l| sig(r.balance(l, NET_STEPS).unwrap())).collect();
        says(1, &format!("(dashed: {}, {} and {})", b[0], b[1], b[2]));
        let turns: Vec<f64> = (0..DEPTH).map(|l| late(r, l)).collect();
        let (lo, hi) = (turns.iter().cloned().fold(f64::INFINITY, f64::min), turns.iter().cloned().fold(0.0, f64::max));
        let want = balance_turn(0.2, 0.005).unwrap();
        says(1, &format!("turn {}° to {}° a step against $\\arccos(1 - \\eta\\lambda) = {}°$", deg2(lo), deg2(hi), deg2(want)));
        assert!(turns.iter().all(|t| (t / want - 1.0).abs() < 0.025));
        let (mut r, mut c) = (run(1).run, run(1).compared.unwrap());
        let (lr, lc) = (r.whole_loss(), c.whole_loss());
        says(1, &format!("ends with the lower loss on all the data, {} against {}", num(lc, 2), num(lr, 2)));
        assert!(lc < lr);
    }

    #[test]
    fn experiment_2_half_the_step_twice_the_decay() {
        let s = run(2);
        let (r, c) = (&s.run, s.compared.as_ref().unwrap());
        assert_eq!((r.setup.eta, r.setup.lambda, c.setup.eta, c.setup.lambda), (0.1, 0.01, 0.2, 0.005));
        assert!((r.setup.eta * r.setup.lambda - c.setup.eta * c.setup.lambda).abs() < 1e-15);
        let ratios: Vec<f64> = (0..DEPTH).map(|l| r.len_at(l, NET_STEPS) / c.len_at(l, NET_STEPS)).collect();
        let (lo, hi) = (ratios.iter().cloned().fold(f64::INFINITY, f64::min), ratios.iter().cloned().fold(0.0, f64::max));
        says(2, &format!("at {} to {} times the grey run's lengths", num(lo, 2), num(hi, 2)));
        says(2, &format!("gives $1/\\sqrt2 = {}$", num(0.5f64.sqrt(), 2)));
        let eff = (0..DEPTH).map(|l| (r.setup.eta / r.len_at(l, NET_STEPS).powi(2)) / (c.setup.eta / c.len_at(l, NET_STEPS).powi(2))).map(|q| (q - 1.0).abs()).fold(0.0, f64::max);
        assert!(eff < 0.12 && eff > 0.1, "{eff}");
        says(2, "within 12 % of the grey run's");
        let (a, b): (Vec<f64>, Vec<f64>) = (0..DEPTH).map(|l| (late(r, l), late(c, l))).unzip();
        let span = |v: &[f64]| (v.iter().cloned().fold(f64::INFINITY, f64::min), v.iter().cloned().fold(0.0, f64::max));
        let ((alo, ahi), (blo, bhi)) = (span(&a), span(&b));
        says(2, &format!("they turn {}° to {}° a step against {}° to {}°: within 3 %", deg2(alo), deg2(ahi), deg2(blo), deg2(bhi)));
        assert!(a.iter().zip(&b).all(|(x, y)| (x / y - 1.0).abs() < 0.03));
        // the exact statement: from weights c times as long, with c²η and λ/c², a step of
        // gradient descent on a layer whose loss ignores its length gives c times the step
        let mut m = c.model.clone();
        let mut tape = Tape::new();
        let idx: Vec<usize> = (0..BATCH).collect();
        c.problem.grad(&mut m, &mut tape, &idx);
        let g = c.layers[1];
        let (w, grad) = (m.groups()[g].value.clone(), m.groups()[g].grad.clone());
        let k = 2.0f64.sqrt();
        let mut scaled = m.clone();
        scaled.groups_mut()[g].value.iter_mut().for_each(|v| *v *= k);
        c.problem.grad(&mut scaled, &mut tape, &idx);
        let (eta, lambda) = (0.2, 0.005);
        for i in [0usize, 7, 100, 255] {
            let one = (1.0 - eta * lambda) * w[i] - eta * grad[i];
            let other = (1.0 - k * k * eta * lambda / (k * k)) * (k * w[i]) - k * k * eta * scaled.groups()[g].grad[i];
            assert!((other - k * one).abs() < 2e-5 * one.abs().max(1e-3), "{i}: {one} {other}");
        }
        let (mut r, mut c) = (run(2).run, run(2).compared.unwrap());
        says(2, &format!("ends at {} against {}", num(r.whole_loss(), 2), num(c.whole_loss(), 2)));
    }

    #[test]
    fn experiment_3_adam_two_kinds() {
        let s = run(3);
        let (r, c) = (&s.run, s.compared.as_ref().unwrap());
        assert!(!r.setup.coupled && c.setup.coupled && r.setup.opt == Opt::Adam && (r.setup.eta, r.setup.lambda) == (0.01, 0.1));
        says(3, &format!("by {} over 100 steps", num(0.999f64.powi(100), 3)));
        says(3, &format!("layer 2 is at {} after them, from {}", sig(r.len_at(1, 100)), sig(r.len_at(1, 0))));
        says(3, &format!("layer 2 falls to {} in the same 100 steps", sig(c.len_at(1, 100))));
        says(3, &format!("(layer 2: {} against {})", sig(r.len_at(1, NET_STEPS)), sig(c.len_at(1, NET_STEPS))));
        says(3, &format!("({}° against {}° a step)", deg2(late(r, 1)), deg2(late(c, 1))));
        // every layer: L2 shorter and turning faster
        assert!((0..DEPTH).all(|l| c.len_at(l, NET_STEPS) < 0.5 * r.len_at(l, NET_STEPS) && late(c, l) > 1.5 * late(r, l)));
    }

    #[test]
    fn experiment_4_muon_inside_a_ball() {
        let s = run(4);
        let (r, c) = (&s.run, s.compared.as_ref().unwrap());
        assert!(r.setup.opt == Opt::Muon && (r.setup.eta, r.setup.lambda) == (0.02, 0.2) && c.setup.lambda == 0.0);
        let (radius, _) = muon_bound(0.02, 0.2, 0.0, 0.0).unwrap();
        says(4, &format!("can never pass $1.2024/\\lambda = {}$", sig(radius)));
        says(4, &format!("at {} and {}", sig(r.top_at(1, NET_STEPS)), sig(r.top_at(2, NET_STEPS))));
        assert!(r.top_at(1, NET_STEPS) < 0.6 * radius && r.top_at(2, NET_STEPS) < 0.6 * radius);
        says(4, &format!("they grow to {} and {}", sig(c.top_at(1, NET_STEPS)), sig(c.top_at(2, NET_STEPS))));
        assert!(c.top_at(1, NET_STEPS) > radius && c.top_at(2, NET_STEPS) > radius);
    }

    /// The button on the picture halves the step and doubles the decay, keeping their
    /// product, and keeps the run until then, grey.
    #[test]
    fn halving_and_doubling_keeps_the_product() {
        let mut s = Decay::create();
        s.set_up(1);
        let before = s.net;
        s.halve_and_double();
        assert_eq!(s.compare, Some(before));
        assert!((s.net.eta * s.net.lambda - before.eta * before.lambda).abs() < 1e-15 && s.net.eta == before.eta / 2.0);
        assert_eq!(s.run.setup, s.net);
        assert_eq!(s.compared.as_ref().map(|c| c.setup), Some(before));
    }

    /// The note under "decay the read-out and the gains too".
    #[test]
    fn decay_on_what_feeds_no_normalisation() {
        let mut out = Vec::new();
        for all in [false, true] {
            let mut r = NetRun::new(NetSetup { all, ..trial(1).2 });
            while !r.done() {
                r.step();
            }
            out.push((r.len_at(DEPTH, NET_STEPS), r.whole_loss()));
        }
        let mut probe = Decay::create();
        probe.net = NetSetup { all: true, ..NetSetup::sgd() };
        let note = format!("it ends the read-out {} long instead of {}, and the loss on all the data at {} instead of {}", sig(out[1].0), sig(out[0].0), num(out[1].1, 2), num(out[0].1, 2));
        let text = include_str!("wdecay.rs");
        assert!(text.contains(&note), "the note should say: {note}");
        assert!(out[1].0 < out[0].0 && out[1].1 > out[0].1);
    }

    #[test]
    fn the_scene_opens_on_the_first_experiment() {
        let opened = Decay::create();
        let mut pressed = Decay::create();
        pressed.set_up(3);
        pressed.set_up(OPENING);
        assert_eq!((opened.tab, opened.toy, opened.net, opened.compare), (pressed.tab, pressed.toy, pressed.net, pressed.compare));
    }

    /// Edges: every setting at its ends, no decay, numbers that are not numbers; the
    /// runs go on and every range stays finite.
    #[test]
    fn degenerate_settings_are_safe() {
        let toys = [ToySetup { eta: f64::NAN, lambda: f64::INFINITY }, ToySetup { eta: 0.0, lambda: -1.0 }, ToySetup { eta: 9.0, lambda: 9.0 }, ToySetup { eta: TOY_LEAST_ETA, lambda: 0.0 }];
        for t in toys {
            let k = t.kept();
            assert!(k.eta.is_finite() && k.lambda.is_finite() && k.eta * k.lambda <= 0.1, "{k:?}");
            let r = ToyRun::of(k);
            assert!(r.w.iter().all(|w| w[0].is_finite() && w[1].is_finite()));
        }
        let nets = [
            NetSetup { eta: f64::NAN, lambda: f64::NAN, ..NetSetup::sgd() },
            NetSetup { eta: 3.0, lambda: 0.1, all: true, ..NetSetup::sgd() },
            NetSetup { opt: Opt::Adam, eta: 0.1, lambda: 2.0, coupled: true, all: true, seed: 9 },
            NetSetup { opt: Opt::Muon, eta: 0.2, lambda: 2.0, ..NetSetup::sgd() },
            NetSetup { opt: Opt::Muon, eta: 1e-9, lambda: 0.0, ..NetSetup::sgd() },
            NetSetup { opt: Opt::Adam, eta: -1.0, lambda: f64::INFINITY, ..NetSetup::sgd() },
        ];
        for n in nets {
            let k = n.kept();
            assert!(k.eta > 0.0 && k.eta.is_finite() && k.lambda >= 0.0 && k.eta * k.lambda <= 0.5 + 1e-12, "{k:?}");
            let mut s = Decay::create();
            s.tab = Tab::Net;
            s.net = n;
            s.compare = Some(NetSetup { lambda: 0.0, ..n });
            s.refresh();
            for _ in 0..200 {
                s.run.step();
                if let Some(c) = s.compared.as_mut() {
                    c.step();
                }
            }
            s.net_shown = 200.0;
            s.follow(0.016, true);
            assert!(s.look.frame.shown.iter().all(|v| v.is_finite()), "{n:?}: {:?}", s.look.frame.shown);
            let g = s.goal();
            assert!(g[NET_SIZE + 1] > g[NET_SIZE] && g[NET_TURN + 1] > g[NET_TURN] && g[NET_LOSS + 1] > g[NET_LOSS], "{n:?}: {g:?}");
        }
        assert_eq!(log_range(f64::NAN, 1.0, 1.0), (0.0, 1.0));
        assert!(muon_bound(0.1, 0.0, 1.0, 1.0).is_none() && muon_bound(2.0, 1.0, 1.0, 1.0).is_none());
    }

    /// The films play to their ends and the picture comes to rest.
    #[test]
    fn the_films_play_to_the_end_and_rest() {
        for steps in [TOY_STEPS, NET_STEPS] {
            let len = Film::length(steps);
            assert_eq!(Film::step_at(0.0, steps), 0.0);
            assert!((Film::step_at(len, steps) - steps as f64).abs() < 1e-9);
            for c in [0.3, 1.7, len * 0.8] {
                assert!((Film::clock_of(Film::step_at(c, steps), steps) - c).abs() < 1e-9);
            }
        }
        for k in [0, 1] {
            let mut s = Decay::create();
            s.set_up(k);
            let mut frames = 0;
            let mut last = 0.0;
            while s.animated() {
                s.advance(1.0 / 60.0, &Budget::new(1.0));
                s.follow(1.0 / 60.0, false);
                let shown = if s.tab == Tab::Toy { s.toy_shown } else { s.net_shown };
                assert!(shown >= last, "the film went back");
                last = shown;
                frames += 1;
                assert!(frames < 60 * 40, "still playing");
            }
            assert_eq!(last, if k == 0 { TOY_STEPS as f64 } else { NET_STEPS as f64 });
        }
    }

    /// The layout keeps everything in the window and apart, from a phone to a desktop.
    #[test]
    fn the_layout_fits_every_window() {
        for (w, h) in [(1220.0, 1000.0), (1080.0, 1000.0), (1600.0, 1000.0), (1600.0, 700.0), (390.0, 800.0), (360.0, 640.0), (700.0, 500.0), (300.0, 600.0), (500.0, 400.0), (240.0, 180.0)] {
            let lay = Layout::of(Vec2::new(w, h), 40.0);
            let inside = |(a, b): Rect| a.x >= 0.0 && a.y >= 0.0 && b.x <= w + 0.5 && b.y <= h + 0.5 && a.is_finite() && b.is_finite();
            let apart = |(a, b): Rect, (c, d): Rect| b.x <= c.x || d.x <= a.x || b.y <= c.y || d.y <= a.y;
            let (a, b) = lay.plane;
            assert!(inside(lay.plane), "{w}x{h}: {:?}", lay.plane);
            assert!(((b.x - a.x) - (b.y - a.y)).abs() < 1.0, "{w}x{h}: the plane is not square");
            let mut boxes: Vec<Rect> = [lay.tabs.to_vec(), lay.buttons.to_vec()].concat();
            boxes.push(lay.plane);
            if !lay.compact {
                boxes.extend(lay.toy_charts.iter().map(|&(a, b)| (a - Vec2::new(CHART_LEFT, CHART_ABOVE), b + Vec2::new(0.0, CHART_BELOW))));
            }
            for (i, a) in boxes.iter().enumerate() {
                assert!(inside(*a), "{w}x{h}: box {i} {a:?}");
                for b in &boxes[i + 1..] {
                    assert!(apart(*a, *b), "{w}x{h}: {a:?} and {b:?} overlap");
                }
            }
            if !lay.compact {
                let charts: Vec<Rect> = lay.net_charts.iter().map(|&(a, b)| (a - Vec2::new(CHART_LEFT, CHART_ABOVE), b + Vec2::new(0.0, CHART_BELOW))).collect();
                for (i, a) in charts.iter().enumerate() {
                    assert!(inside(*a), "{w}x{h}: chart {i} {a:?}");
                    for b in charts[i + 1..].iter().chain(lay.tabs.iter()).chain(lay.buttons.iter()) {
                        assert!(apart(*a, *b), "{w}x{h}: {a:?} and {b:?} overlap");
                    }
                }
                assert!(lay.net_charts.iter().all(|c| c.1.y - c.0.y > 40.0 && c.1.x - c.0.x > 150.0), "{w}x{h}");
                assert!(lay.toy_charts.iter().all(|c| c.1.y - c.0.y > 40.0), "{w}x{h}: {:?}", lay.toy_charts);
            }
        }
    }

    /// The ranges' glides come to rest after a change.
    #[test]
    fn glides_arrive_and_stop() {
        let mut s = Decay::create();
        s.follow(0.016, true);
        for k in [1, 0, 4, 3, 2] {
            s.set_up(k);
            let mut frames = 0;
            s.follow(1.0 / 60.0, false);
            while s.look.moving {
                s.follow(1.0 / 60.0, false);
                frames += 1;
                assert!(frames < 300, "experiment {k} still moving");
            }
        }
    }

    /// The numbers behind the captions (run with --ignored --nocapture).
    #[test]
    #[ignore]
    fn explore() {
        let s = run(0);
        let (r, p) = (&s.toy_run, s.toy_plain.as_ref().unwrap());
        let n = TOY_STEPS;
        println!("toy: balance {:?} len end {} plain end {} turn end {} plain turn end {} arccos {:?} slow {} pyth {}", r.balance(n), r.len(n), p.len(n), r.turn(n).to_degrees(), p.turn(n).to_degrees(), balance_turn(0.1, 0.01).map(|a| a.to_degrees()), slowing_turn(n as f64).to_degrees(), r.pythagoras);
        for t in [1usize, 10, 30, 100, 300, 1000, 2000, 3000] {
            println!("   t {t}: len {:.3} {:.3} turn {:.4} {:.4} balance {:?}", r.len(t), p.len(t), r.turn(t).to_degrees(), p.turn(t).to_degrees(), r.balance(t));
        }
        for k in 1..TRIALS.len() {
            let mut s = run(k);
            for (name, r) in [("main", &mut s.run)].into_iter().chain(s.compared.as_mut().map(|c| ("faint", c))) {
                let t = r.t;
                let whole = r.whole_loss();
                let most = r.layers.iter().map(|&g| r.model.groups()[g].value.iter().fold(0.0f64, |a, b| a.max(b.abs()))).fold(0.0, f64::max);
                println!("exp {k} {name} {:?} t {t} tilt {:.2e} loss batches {:.4} whole {:.4} readout {:.3} max entry {most:.3}", r.setup, r.tilt, r.batch_loss(t), whole, r.len_at(DEPTH, t));
                for l in 0..DEPTH {
                    println!(
                        "   layer {}: len start {:.3} at100 {:.3} end {:.3} balance {:?} turn end {:.4}° top {:.3} top0 {:.3} | at 1000 len {:.3} turn {:.4}°",
                        l + 1,
                        r.len_at(l, 0),
                        r.len_at(l, 100),
                        r.len_at(l, t),
                        r.balance(l, t),
                        r.turn(l, t).to_degrees(),
                        r.top_at(l, t),
                        r.top_at(l, 0),
                        r.len_at(l, 1000),
                        r.turn(l, 1000).to_degrees()
                    );
                }
            }
        }
        for all in [false, true] {
            let mut r = NetRun::new(NetSetup { all, ..NetSetup::sgd() });
            while !r.done() {
                r.step();
            }
            println!("all {all}: readout {:.3} whole {:.4} gains {:?}", r.len_at(DEPTH, NET_STEPS), r.whole_loss(), r.model.groups().iter().filter(|g| g.name.contains("gain")).map(|g| norm(&g.value)).collect::<Vec<_>>());
        }
        println!("slow at 4000: {}", slowing_turn(4000.0).to_degrees());
    }
}

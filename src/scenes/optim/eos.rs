//! The edge of stability. A small network is trained by full-batch gradient descent
//! (or heavy-ball momentum, or Adam, or with mini-batches), and at every step its
//! sharpness, the largest eigenvalue of the loss's Hessian, is estimated by Lanczos's
//! method from the last step's estimate (`math::terrain`). Three charts share one step
//! axis: the loss, the sharpness against the level 2/η (or the rule's own level) that
//! the step size sets, and the step size or Adam's sharpness without its
//! preconditioner. Beside them the loss along the sharpest direction, through the
//! midpoint of the last two steps, with the last few iterates on it, the parabola of
//! the sharpness's curvature and, dashed, one exactly as sharp as the step can bear.
//!
//! The task is like the smallest Cohen and co-authors (2021, Appendix L; course notes,
//! curvature.md B1) used: a network with one hidden layer of tanh units fitted, with
//! squared error, to a Chebyshev polynomial at twenty points (here spaced evenly on
//! [−1, 1], 32 units). Every outcome the scene states is this network's, measured for
//! the stated seed and settings, and held by the tests at the end of the file.

use super::*;
use crate::clock::{self, Budget};
use crate::math::arch::{Init, Loss, MlpSpec, Model, Norm, Place};
use crate::math::autograd::{Act, Tape};
use crate::math::optim::{self, Hyper, Optimizer};
use crate::math::terrain::{self, BatchSharpness, Draw, Lanczos, Problem, Sampler};
use crate::scenes::stats::plot::{self, Glide, Plot, Rect, AXES, MARKS};
use std::collections::VecDeque;

pub const DEF: SceneDef = SceneDef {
    id: "eos",
    title: "The edge of stability",
    group: GROUP,
    blurb: "Train a network by full-batch gradient descent and watch its sharpness, the largest curvature of the loss, against the largest that the step size can bear.",
    make: |_| Box::new(Eos::create()),
};

// ----- the task ----------------------------------------------------------------------------

/// The examples: points spaced evenly on [−1, 1].
const POINTS: usize = 20;

/// The Chebyshev polynomial of degree k at x, by its recurrence T₀ = 1, T₁ = x,
/// T_{k+1} = 2x T_k − T_{k−1}.
fn chebyshev(k: usize, x: f64) -> f64 {
    let (mut a, mut b) = (1.0, x);
    if k == 0 {
        return a;
    }
    for _ in 1..k {
        (a, b) = (b, 2.0 * x * b - a);
    }
    b
}

/// The inputs: `POINTS` points spaced evenly on [−1, 1].
fn inputs() -> Vec<f64> {
    (0..POINTS).map(|i| -1.0 + 2.0 * i as f64 / (POINTS - 1) as f64).collect()
}

/// The data set: the inputs labelled by the Chebyshev polynomial of degree `degree`,
/// with squared error (the mean over the points of (output − target)², no ½).
fn task(degree: usize) -> Problem {
    let x = inputs();
    let y: Vec<f64> = x.iter().map(|&x| chebyshev(degree, x)).collect();
    Problem::points(&x, &y, POINTS, Loss::Mse)
}

// ----- a run -------------------------------------------------------------------------------

/// The rule a run trains with.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Rule {
    /// Plain gradient descent: θ ← θ − η g.
    Gd,
    /// Heavy-ball momentum, v ← β v + g, θ ← θ − η v (PyTorch's `SGD` with dampening 0).
    Momentum,
    /// Adam with Kingma and Ba's defaults (β₁ = 0.9, β₂ = 0.999, ε = 10⁻⁸, bias
    /// correction).
    Adam,
}

impl Rule {
    fn color(self) -> Color {
        match self {
            Rule::Gd => SGD,
            Rule::Momentum => MOMENTUM,
            Rule::Adam => ADAM,
        }
    }
}

/// Adam's β₁ (Kingma and Ba's default, the library's).
const BETA1: f64 = 0.9;
/// What a cut divides the step by.
const CUT: f64 = 4.0;

/// The ranges of the controls.
const LEAST_ETA: f64 = 0.005;
const MOST_ETA: f64 = 1.0;
const LEAST_ADAM_ETA: f64 = 1e-4;
const MOST_ADAM_ETA: f64 = 0.1;
const MOST_BETA: f64 = 0.99;
const LEAST_WIDTH: usize = 4;
const MOST_WIDTH: usize = 64;
const MOST_DEPTH: usize = 3;
const LEAST_STEPS: usize = 200;
const MOST_STEPS: usize = 10_000;
const DEGREES: [usize; 3] = [3, 4, 5];

/// Everything a run depends on.
#[derive(Clone, Copy, Debug, PartialEq)]
struct Setup {
    rule: Rule,
    eta: f64,
    /// Heavy-ball momentum β (the rule Momentum only).
    beta: f64,
    /// Examples per step: `POINTS` is the whole data set.
    batch: usize,
    width: usize,
    depth: usize,
    /// The Chebyshev polynomial's degree.
    degree: usize,
    seed: u64,
    steps: usize,
    /// From this step on the step is divided by `CUT`.
    cut: Option<usize>,
}

impl Setup {
    /// The opening experiment's run.
    fn base() -> Setup {
        Setup { rule: Rule::Gd, eta: 0.2, beta: 0.9, batch: POINTS, width: 32, depth: 1, degree: 4, seed: 1, steps: 4000, cut: None }
    }

    /// Kept to the controls' ranges (a value that is not a number takes the base's).
    fn kept(self) -> Setup {
        let base = Setup::base();
        let fix = |v: f64, lo: f64, hi: f64, or: f64| if v.is_finite() { v.max(lo).min(hi) } else { or };
        let (lo, hi) = if self.rule == Rule::Adam { (LEAST_ADAM_ETA, MOST_ADAM_ETA) } else { (LEAST_ETA, MOST_ETA) };
        let steps = self.steps.clamp(LEAST_STEPS, MOST_STEPS);
        Setup {
            eta: fix(self.eta, lo, hi, base.eta.max(lo).min(hi)),
            beta: fix(self.beta, 0.0, MOST_BETA, base.beta),
            batch: self.batch.clamp(1, POINTS),
            width: self.width.clamp(LEAST_WIDTH, MOST_WIDTH),
            depth: self.depth.clamp(1, MOST_DEPTH),
            degree: if DEGREES.contains(&self.degree) { self.degree } else { base.degree },
            steps,
            cut: self.cut.map(|c| c.clamp(1, steps)),
            ..self
        }
    }

    /// The step size used for step t (from the weights after t steps).
    fn eta_at(&self, t: usize) -> f64 {
        match self.cut {
            Some(c) if t >= c => self.eta / CUT,
            _ => self.eta,
        }
    }

    /// The level that the step size sets at step t: the largest curvature of a
    /// quadratic on which the rule does not blow up (for Adam, with its preconditioner
    /// held fixed, on the preconditioned sharpness). Gradient descent 2/η; heavy ball
    /// (2 + 2β)/η; Adam (2 + 2β₁)/((1 − β₁)η), 38/η (Cohen and co-authors 2022, §4;
    /// course notes, curvature.md A2, A4, B2).
    fn level_at(&self, t: usize) -> f64 {
        let eta = self.eta_at(t);
        match self.rule {
            Rule::Gd => 2.0 / eta,
            Rule::Momentum => (2.0 + 2.0 * self.beta) / eta,
            Rule::Adam => (2.0 + 2.0 * BETA1) / ((1.0 - BETA1) * eta),
        }
    }

    /// The level's formula, for a label: "2/η", "3.8/η", "38/η".
    fn level_name(&self) -> String {
        match self.rule {
            Rule::Gd => r"2/\eta".to_string(),
            Rule::Momentum => format!(r"{}/\eta", short(2.0 + 2.0 * self.beta)),
            Rule::Adam => r"38/\eta".to_string(),
        }
    }

    fn full_batch(&self) -> bool {
        self.batch >= POINTS
    }

    /// Whether the sharpness held against the level is the preconditioned one.
    fn preconditioned(&self) -> bool {
        self.rule == Rule::Adam
    }

    fn spec(&self) -> MlpSpec {
        MlpSpec {
            inputs: 1,
            width: self.width,
            depth: self.depth,
            outputs: 1,
            act: Act::Tanh,
            norm: Norm::None,
            place: Place::Pre,
            residual: false,
            residual_scale: 1.0,
            init: Init::Xavier,
            gain: 1.0,
            bias: true,
        }
    }

    fn params(&self) -> usize {
        // (1 input: w + w; hidden layers w² + w each; the read-out w + 1)
        let w = self.width;
        2 * w + (self.depth - 1) * (w * w + w) + w + 1
    }

    fn optimizer(&self, model: &Model) -> Optimizer {
        let (rule, hyper) = match self.rule {
            Rule::Gd => (optim::Rule::Sgd, Hyper::default_for(optim::Rule::Sgd)),
            Rule::Momentum => (optim::Rule::Momentum, Hyper { momentum: self.beta, ..Hyper::default_for(optim::Rule::Momentum) }),
            Rule::Adam => (optim::Rule::Adam, Hyper { beta1: BETA1, ..Hyper::default_for(optim::Rule::Adam) }),
        };
        Optimizer::new(model, rule, Hyper { lr: self.eta, ..hyper })
    }

    /// What tells this run from `other` on a chart: the first setting in which they
    /// differ (the whole label if there is no other).
    fn tag(&self, other: Option<&Setup>) -> String {
        let Some(o) = other else { return self.label() };
        if self.rule != o.rule || (self.rule == Rule::Momentum && self.beta != o.beta) {
            return match self.rule {
                Rule::Gd => "gradient descent".to_string(),
                Rule::Momentum => format!(r"momentum $\beta = {}$", short(self.beta)),
                Rule::Adam => "Adam".to_string(),
            };
        }
        if self.eta != o.eta {
            return format!(r"$\eta = {}$", short(self.eta));
        }
        if self.cut != o.cut {
            return self.cut.map_or("no cut".to_string(), |c| format!("cut at {}", count(c)));
        }
        if self.batch != o.batch {
            return if self.full_batch() { format!("all {POINTS} points") } else { format!("batches of {}", self.batch) };
        }
        if self.width != o.width || self.depth != o.depth {
            return format!(r"{} units $\times$ {}", self.width, self.depth);
        }
        if self.degree != o.degree {
            return format!("$T_{}$", self.degree);
        }
        if self.seed != o.seed {
            return format!("start {}", self.seed);
        }
        self.label()
    }

    /// The run named as the charts name it.
    fn label(&self) -> String {
        let mut s = format!(r"$\eta = {}$", short(self.eta));
        match self.rule {
            Rule::Gd => {}
            Rule::Momentum => s = format!(r"momentum $\beta = {}$, {s}", short(self.beta)),
            Rule::Adam => s = format!("Adam, {s}"),
        }
        if let Some(c) = self.cut {
            s += &format!(", cut at {}", count(c));
        }
        if !self.full_batch() {
            s += &format!(", batches of {}", self.batch);
        }
        s
    }
}

/// A loss above this has blown up: the run stops there.
const BLOWN: f64 = 1e6;
/// The most Hessian-vector products one estimate of the sharpness may take.
const MOST_PRODUCTS: usize = 60;
/// A slice is recorded every this many steps (an even number: at the edge of
/// stability the iterates flip across the valley every step, and slices an even
/// number of steps apart catch them on the same side).
const SLICE_EVERY: usize = 10;
/// Points along a slice.
const SLICE_N: usize = 49;
/// The iterates drawn on a slice: the present one and the ones before it.
const TRAIL: usize = 8;
/// With mini-batches, the batch sharpness is estimated every this many steps, from
/// this many batches.
const BATCH_EVERY: usize = 20;
const BATCH_DRAWS: usize = 16;

/// The loss along the sharpest direction through the midpoint of the last two steps.
#[derive(Clone, Debug, PartialEq)]
struct Slice {
    step: usize,
    /// The slice reaches from −range to range (in the direction's units: the weights'
    /// own, or Adam's preconditioned ones).
    range: f64,
    /// The loss at `SLICE_N` points evenly across.
    values: Vec<f64>,
    /// The present iterate and the ones before it (newest first): where each is along
    /// the direction, and the loss on the slice there.
    iterates: Vec<(f64, f64)>,
    /// The curvature the step bears there.
    level: f64,
    /// The curvature of the slice at its centre: the sharpness (or the preconditioned
    /// one), as estimated.
    curvature: f64,
}

/// One training run and everything measured along it.
struct Run {
    setup: Setup,
    problem: Problem,
    model: Model,
    opt: Optimizer,
    tape: Tape,
    lanczos: Lanczos,
    /// Adam's: of the preconditioned Hessian.
    pre: Option<Lanczos>,
    sampler: Sampler,
    batch_sharpness: Option<BatchSharpness>,
    /// Steps taken.
    t: usize,
    /// At each iterate w_t (t = 0 … steps): the whole data's loss, the sharpness, its
    /// estimate's residual (relative to the largest Ritz value) and its products; for
    /// Adam the preconditioned sharpness and its residual (NaN at t = 0, before the
    /// first preconditioner).
    loss: Vec<f64>,
    sharp: Vec<f64>,
    resid: Vec<f64>,
    products: Vec<u32>,
    psharp: Vec<f64>,
    presid: Vec<f64>,
    /// With mini-batches: (step, batch sharpness, its standard error).
    batch: Vec<(usize, f64, f64)>,
    slices: Vec<Slice>,
    /// The last weights, newest first.
    recent: VecDeque<Vec<f64>>,
    /// The direction of the last slice, to keep its sign from slice to slice.
    last_dir: Vec<f64>,
    /// The step at which the loss blew up, if it did.
    blown: Option<usize>,
    pbuf: Vec<f64>,
}

impl Run {
    fn new(setup: Setup) -> Run {
        let setup = setup.kept();
        let problem = task(setup.degree);
        let model = Model::mlp(&setup.spec(), &mut Rng::new(setup.seed));
        let opt = setup.optimizer(&model);
        let lanczos = Lanczos::new(&model, 1, setup.seed ^ 0x5eed);
        let pre = setup.preconditioned().then(|| Lanczos::new(&model, 1, setup.seed ^ 0x0ada));
        let sampler = Sampler::new(POINTS, setup.batch, Draw::WithoutReplacement, Rng::new(setup.seed.wrapping_mul(31).wrapping_add(7)));
        let batch_sharpness = (!setup.full_batch()).then(|| BatchSharpness::new(Sampler::new(POINTS, setup.batch, Draw::WithoutReplacement, Rng::new(setup.seed.wrapping_mul(97).wrapping_add(3)))));
        let n = setup.steps + 1;
        Run {
            setup,
            problem,
            model,
            opt,
            tape: Tape::new(),
            lanczos,
            pre,
            sampler,
            batch_sharpness,
            t: 0,
            loss: Vec::with_capacity(n),
            sharp: Vec::with_capacity(n),
            resid: Vec::with_capacity(n),
            products: Vec::with_capacity(n),
            psharp: Vec::with_capacity(n),
            presid: Vec::with_capacity(n),
            batch: Vec::new(),
            slices: Vec::new(),
            recent: VecDeque::new(),
            last_dir: Vec::new(),
            blown: None,
            pbuf: Vec::new(),
        }
    }

    fn done(&self) -> bool {
        self.loss.len() > self.setup.steps || self.blown.is_some()
    }

    /// The iterates measured so far (w_0 … w_{n−1}).
    fn measured(&self) -> usize {
        self.loss.len()
    }

    /// The sharpness held against the level at iterate t: the preconditioned one for
    /// Adam.
    fn main(&self, t: usize) -> f64 {
        let v = if self.setup.preconditioned() { &self.psharp } else { &self.sharp };
        v.get(t).copied().unwrap_or(f64::NAN)
    }

    /// Measure the iterate w_t, then (unless the run is over) take step t.
    fn step(&mut self) {
        if self.done() {
            return;
        }
        let t = self.t;
        let s = self.setup;
        let loss = if s.full_batch() {
            self.problem.full_grad(&mut self.model, &mut self.tape)
        } else {
            let loss = self.problem.loss(&self.model, &mut self.tape);
            let idx = self.sampler.deal().to_vec();
            self.problem.grad(&mut self.model, &mut self.tape, &idx);
            loss
        };
        self.loss.push(loss);
        if !(loss.is_finite() && loss < BLOWN) {
            self.blown = Some(t);
            for v in [&mut self.sharp, &mut self.resid, &mut self.psharp, &mut self.presid] {
                v.push(f64::NAN);
            }
            self.products.push(0);
            return;
        }
        // the weights now, kept for the slice
        let mut w = self.recent.len().gt(&TRAIL).then(|| self.recent.pop_back()).flatten().unwrap_or_default();
        terrain::get(&self.model, &mut w);
        self.recent.push_front(w);
        // the sharpness at w_t, started from the last estimate (the Hessian-vector
        // products leave the gradient just computed as it is)
        if t > 0 {
            self.lanczos.restart();
        }
        while !self.lanczos.advance(&self.problem, &mut self.model, &mut self.tape, 3) && self.lanczos.products() < MOST_PRODUCTS {}
        self.sharp.push(self.lanczos.sharpness());
        self.resid.push(self.lanczos.residual());
        self.products.push(self.lanczos.products() as u32);
        // Adam's preconditioned sharpness, with the preconditioner of the last step
        let mut pre_done = false;
        if let Some(pre) = &mut self.pre {
            if t > 0 && self.opt.preconditioner(&mut self.pbuf) {
                pre.set_preconditioner(&self.pbuf);
                while !pre.advance(&self.problem, &mut self.model, &mut self.tape, 3) && pre.products() < MOST_PRODUCTS {}
                self.psharp.push(pre.sharpness());
                self.presid.push(pre.residual());
                pre_done = true;
            }
        }
        if !pre_done {
            self.psharp.push(f64::NAN);
            self.presid.push(f64::NAN);
        }
        if let Some(bs) = &mut self.batch_sharpness {
            if t.is_multiple_of(BATCH_EVERY) {
                bs.reset();
                bs.advance(&self.problem, &mut self.model, &mut self.tape, BATCH_DRAWS);
                self.batch.push((t, bs.mean(), bs.standard_error()));
            }
        }
        // (Adam's direction needs a preconditioner: its first slice is at step 1)
        if t.is_multiple_of(SLICE_EVERY) || t == s.steps || (t == 1 && self.pre.is_some()) {
            self.record_slice(t);
        }
        if t >= s.steps {
            return;
        }
        // (the gradient is still in the model: Lanczos's products restore the values
        // and do not touch the accumulated gradients)
        self.opt.step(&mut self.model, s.eta_at(t));
        self.t += 1;
    }

    /// The slice at iterate t.
    fn record_slice(&mut self, t: usize) {
        let s = self.setup;
        let p = self.model.n_params();
        // the direction in the weights, d, and the scale its coordinate is read in:
        // the plain one, or (Adam) the preconditioner's, so that the curvature of the
        // slice is the preconditioned sharpness
        let (mut dir, root): (Vec<f64>, Vec<f64>) = match &self.pre {
            Some(pre) => {
                let y = pre.vector(0);
                if !(t > 0 && self.pbuf.len() == p && y.len() == p) {
                    return;
                }
                let root: Vec<f64> = self.pbuf.iter().map(|v| v.sqrt()).collect();
                (y.iter().zip(&root).map(|(y, r)| y / r).collect(), root)
            }
            None => {
                let u = self.lanczos.vector(0);
                if u.len() != p {
                    return;
                }
                (u.to_vec(), vec![1.0; p])
            }
        };
        if self.last_dir.len() == p && dir.iter().zip(&self.last_dir).map(|(a, b)| a * b).sum::<f64>() < 0.0 {
            dir.iter_mut().for_each(|v| *v = -*v);
        }
        self.last_dir.clone_from(&dir);
        let Some(now) = self.recent.front().cloned() else { return };
        let centre: Vec<f64> = match self.recent.get(1) {
            Some(before) => now.iter().zip(before).map(|(a, b)| 0.5 * (a + b)).collect(),
            None => now.clone(),
        };
        // where each iterate is along it: x with w = centre + x d + (something
        // across d, in the scale's inner product)
        let unit: Vec<f64> = dir.iter().zip(&root).map(|(d, r)| d * r * r).collect();
        let along = |w: &[f64]| w.iter().zip(&centre).zip(&unit).map(|((w, c), u)| (w - c) * u).sum::<f64>();
        let at: Vec<f64> = self.recent.iter().map(|w| along(w)).collect();
        let level = s.level_at(t);
        let loss = self.loss.last().copied().unwrap_or(1.0).max(1e-12);
        let spread = at.iter().fold(0.0f64, |m, v| m.max(v.abs()));
        // (wide enough for the iterates, and at least where a valley as sharp as the
        // level rises by a twentieth of the loss)
        let range = (2.5 * spread).max((0.1 * loss / level).sqrt());
        let ts: Vec<f64> = (0..SLICE_N).map(|i| -range + 2.0 * range * i as f64 / (SLICE_N - 1) as f64).chain(at.iter().copied()).collect();
        terrain::set(&mut self.model, &centre);
        let all = self.problem.all();
        let mut out = Vec::new();
        terrain::directional(&self.problem, &mut self.model, &mut self.tape, &all, &dir, &ts, &mut out);
        terrain::set(&mut self.model, &now);
        let values = out[..SLICE_N.min(out.len())].to_vec();
        let mut iterates: Vec<(f64, f64)> = at.iter().zip(out.iter().skip(SLICE_N)).map(|(a, l)| (*a, *l)).collect();
        // (always as many, the oldest repeated in the first steps, so that one slice
        // can glide into the next)
        if let Some(&oldest) = iterates.last() {
            iterates.resize(TRAIL + 1, oldest);
        }
        self.slices.push(Slice { step: t, range, values, iterates, level, curvature: self.main(t) });
    }
}

// ----- what a run shows, measured -----------------------------------------------------------

/// After first reaching its level, the sharpness is followed from this many steps
/// later (the first swings are over by then).
const SETTLE: usize = 200;

/// What a run did over its iterates `from .. to`, measured.
#[derive(Clone, Copy, Debug, PartialEq)]
struct Stats {
    /// The sharpness (held against the level) at `from`, the largest, and where.
    start: f64,
    peak: f64,
    peak_at: usize,
    /// The first iterate at which it was at or over the level.
    cross: Option<usize>,
    /// Iterates before that at which it had fallen since the one before.
    falls_before: usize,
    /// From `cross + SETTLE` on: the sharpness over the level, its mean, least and
    /// largest, and over how many iterates.
    mean: f64,
    lo: f64,
    hi: f64,
    stretch: usize,
    /// From `cross` on (or over the whole stretch, if it never crosses): steps after
    /// which the loss was higher than before them, and of how many.
    rises: usize,
    of: usize,
    /// The loss at `from`, at the crossing and at the last iterate.
    loss_from: f64,
    loss_cross: f64,
    loss_end: f64,
}

impl Run {
    /// What it did over iterates `from .. to` (as far as measured).
    fn stats(&self, from: usize, to: usize) -> Stats {
        let n = to.min(self.measured());
        let from = from.min(n.saturating_sub(1));
        let s = &self.setup;
        let mut st = Stats {
            start: self.main(from),
            peak: f64::NAN,
            peak_at: from,
            cross: None,
            falls_before: 0,
            mean: f64::NAN,
            lo: f64::NAN,
            hi: f64::NAN,
            stretch: 0,
            rises: 0,
            of: 0,
            loss_from: self.loss.get(from).copied().unwrap_or(f64::NAN),
            loss_cross: f64::NAN,
            loss_end: n.checked_sub(1).and_then(|i| self.loss.get(i)).copied().unwrap_or(f64::NAN),
        };
        for t in from..n {
            let v = self.main(t);
            // (the first number, or a larger one: the peak starts as NaN)
            if v.is_finite() && (st.peak.is_nan() || v > st.peak) {
                st.peak = v;
                st.peak_at = t;
            }
            if st.cross.is_none() {
                if v >= s.level_at(t) {
                    st.cross = Some(t);
                } else if t > from && v < self.main(t - 1) {
                    st.falls_before += 1;
                }
            }
        }
        if let Some(c) = st.cross {
            st.loss_cross = self.loss[c];
            let (mut sum, mut k, mut lo, mut hi) = (0.0, 0, f64::INFINITY, f64::NEG_INFINITY);
            for t in (c + SETTLE)..n {
                let r = self.main(t) / s.level_at(t);
                if r.is_finite() {
                    sum += r;
                    k += 1;
                    lo = lo.min(r);
                    hi = hi.max(r);
                }
            }
            if k > 0 {
                (st.mean, st.lo, st.hi, st.stretch) = (sum / k as f64, lo, hi, k);
            }
        }
        let first = st.cross.unwrap_or(from);
        for t in (first + 1)..n {
            st.of += 1;
            if self.loss[t] > self.loss[t - 1] {
                st.rises += 1;
            }
        }
        st
    }

    /// The mean loss over iterates `a .. b`. (The tests hold a caption to it.)
    #[cfg(test)]
    fn mean_loss(&self, a: usize, b: usize) -> f64 {
        let b = b.min(self.measured());
        if b <= a {
            return f64::NAN;
        }
        self.loss[a..b].iter().sum::<f64>() / (b - a) as f64
    }

    /// The largest relative residual of the sharpness estimates so far.
    fn worst_residual(&self) -> f64 {
        let v = if self.setup.preconditioned() { &self.presid } else { &self.resid };
        v.iter().copied().filter(|r| r.is_finite()).fold(0.0, f64::max)
    }

    /// The products an estimate took, on average.
    fn mean_products(&self) -> f64 {
        if self.products.is_empty() {
            return f64::NAN;
        }
        self.products.iter().map(|&p| p as f64).sum::<f64>() / self.products.len() as f64
    }
}

// ----- numbers as the scene writes them ----------------------------------------------------

/// A number with three significant digits (0.0129, 0.00635, 1.96, 152), with a true
/// minus sign; thousands set apart.
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

/// A setting as the labels write it: as few digits as it needs (0.1, 0.025, 1.6).
fn short(v: f64) -> String {
    let text = sig(v);
    if text.contains('.') {
        text.trim_end_matches('0').trim_end_matches('.').to_string()
    } else {
        text
    }
}

/// A loss as the read-outs write it: three significant digits, in powers of ten when
/// small ($2.07 \times 10^{-4}$).
fn small(v: f64) -> String {
    if v.is_finite() && v != 0.0 && v.abs() < 0.001 {
        let e = v.abs().log10().floor() as i32;
        let m = v / 10f64.powi(e);
        // (rounding can make the mantissa 10.0)
        let (m, e) = if m.abs() >= 9.995 { (m / 10.0, e + 1) } else { (m, e) };
        format!(r"${} \times 10^{{{}}}$", num(m, 2), e)
    } else {
        sig(v)
    }
}

/// A share in per cent, whole.
fn percent(part: usize, of: usize) -> String {
    format!("{} %", num(100.0 * part as f64 / of.max(1) as f64, 0))
}

// ----- the experiments ---------------------------------------------------------------------

struct Trial {
    button: &'static str,
    caption: &'static str,
}

const TRIALS: [Trial; 7] = [
    Trial {
        button: "a short step",
        caption: r"Step $\eta = 0.05$: the level $2/\eta$ is 40. The sharpness rises from 6.05 to 17.4, its largest at the last of the 4000 steps, and never comes near its level. The loss falls at every step, to 0.197. In the slice the green parabola is far wider than the yellow one, and the iterates do not flip across the valley: they move one way along it. This is the first phase only: the step is too short to reach the second within this run.",
    },
    Trial {
        button: "a longer step",
        caption: r"$\eta = 0.2$, level 10. The sharpness rises from 6.05 to 10 at step 607, the loss falling at every step. It goes on rising, and the iterates start to flip across the sharpest direction, each flip about 1.5 times the last: the sharpness is about 12.5 by then, and on a bowl that sharp the factor is $|1 - 0.2 \times 12.5| = 1.5$. At step 760 the loss leaps to 2.84 and the sharpness drops to 7.16. After four such leaps, the last at step 1722, the sharpness stays near its level: from step 1800 on, 1.03 times it on average (from 0.96 to 1.16). The loss rises at 42 % of those steps, yet its mean over each 500 steps is lower than over the 500 before, down to 0.0815.",
    },
    Trial {
        button: "the two side by side",
        caption: r"$\eta = 0.2$ (bright) and $\eta = 0.1$ (faint), each with its own level. Until it reaches its level the shorter step's run is the longer one's at half the pace: its sharpness at step $2t$ is within 2 % of the other's at step $t$. So the longer step meets its level, 10, at step 607, while the shorter goes on rising to its own, 20, at step 2287, and its first leap of the loss comes at step 2576. Over the last 1000 steps each averages 1.02 times its own level: here, where the sharpness settles is set by the step.",
    },
    Trial {
        button: "a step cut by four",
        caption: r"$\eta = 0.3$, cut to 0.075 at step 1000 (bright), against no cut (faint). Up to the cut both are at the edge of stability, near the level 6.67, with two leaps of the loss. After the cut the level is 26.7, and the sharpness rises at once: 6.71 at the cut, 16.6 1500 steps later, back at the new level at step 5010. Until then the loss falls at every step; from there it goes up and down again. Without the cut the sharpness stays between 0.98 and 1.05 times 6.67 from step 2000 on. Cohen and co-authors (2021) cut the step of a network trained by full-batch gradient descent (from $2/200$ to $2/300$) and saw the same: the sharpness rose until it was back at the edge of stability.",
    },
    Trial {
        button: "momentum 0.9",
        caption: r"Heavy-ball momentum $\beta = 0.9$ at $\eta = 0.1$ (pink), against gradient descent at the same step (faint). Written $v \leftarrow \beta v + g$, $\theta \leftarrow \theta - \eta v$, momentum bears curvatures up to $(2 + 2\beta)/\eta = 38$, not 20. The sharpness reaches 38 at step 481 and goes on to 43.2; the loss leaps to 17.5 and the sharpness drops to 14.0. From step 623 the loss falls at every step, and the sharpness climbs back, to 36.1 at the end (0.95 of its level), with the loss down to $4.58 \times 10^{-4}$. Cohen and co-authors (2021) found that with full-batch momentum the sharpness rises to this level and then stays just above it, or swings about it; within this run it came back only to just under it.",
    },
    Trial {
        button: "Adam",
        caption: r"Adam at $\eta = 0.01$. Its level, $38/\eta = 3800$, is for the preconditioned sharpness: the largest eigenvalue of $P^{-1}H$, where $P$ holds the numbers Adam divides each parameter's step by. It starts far above (69 408 at step 1, after one gradient); from step 1000 on it averages 0.99 of 3800, between 0.93 and 1.04 times it, and the loss rises at 42 % of the steps. The sharpness itself (lowest chart) has no level here: it climbs from 6.05 to 43.2 by step 2000, and is 43.3 at the end. Cohen and co-authors (2022) report, for full-batch Adam on image classifiers, the preconditioned sharpness at $38/\eta$ and the plain one usually still rising; here the plain one had almost stopped rising by step 2000. (Their main runs left out Adam's bias correction; this one keeps it.)",
    },
    Trial {
        button: "mini-batches",
        caption: r"Batches of 10 of the 20 points at $\eta = 0.2$ (bright), against all 20 (faint). The whole data's sharpness first falls, to 2.32 at step 283, near the least it can be (2); then it rises, but reaches $2/\eta$ at only 7 steps, between 2292 and 2335, and over the last 1000 steps averages 0.89 of it. The red dots are the batch sharpness, each batch's own curvature along its own gradient (16 batches every 20 steps): over the last 1000 steps 0.80 of $2/\eta$ on average. Andreyev and Beneventano (a 2024 preprint) report that for image classifiers trained by plain mini-batch SGD (no momentum, no Adam) with squared error this settles at $2/\eta$; in this run its average stayed lower. With batches of 5 at this step, all five seeds tried ended with every weight near zero: the network gives one constant, and the sharpness is 2, the least it can be on this loss.",
    },
];

/// The scene opens as this experiment sets it up.
const OPENING: usize = 1;

/// The run, and the one drawn faint beside it, of experiment k.
fn trial(k: usize) -> (Setup, Option<Setup>) {
    let base = Setup::base();
    match k {
        0 => (Setup { eta: 0.05, ..base }, None),
        2 => (base, Some(Setup { eta: 0.1, ..base })),
        3 => {
            let s = Setup { eta: 0.3, steps: 6000, ..base };
            (Setup { cut: Some(1000), ..s }, Some(s))
        }
        4 => (Setup { rule: Rule::Momentum, eta: 0.1, beta: 0.9, ..base }, Some(Setup { eta: 0.1, ..base })),
        5 => (Setup { rule: Rule::Adam, eta: 0.01, ..base }, None),
        6 => (Setup { batch: 10, ..base }, Some(base)),
        _ => (base, None),
    }
}

// ----- the film ----------------------------------------------------------------------------

/// A whole run plays in this many seconds at the speed 1.
const FILM_SECONDS: f64 = 12.0;

/// The run played as a film: the step shown moves on at a steady rate, as far as the
/// runs have been computed.
#[derive(Clone, Copy, Debug, Default)]
struct Film {
    /// The step shown (fractional: the picture goes smoothly between steps).
    shown: f64,
    playing: bool,
    /// Play once the picture has glided to the new run's start, or after `FILM_WAIT`
    /// seconds; how long it has waited.
    waiting: bool,
    waited: f32,
}

/// The longest the film waits for a glide, in seconds.
const FILM_WAIT: f32 = 0.2;

// ----- the scene -----------------------------------------------------------------------------

pub struct Eos {
    setup: Setup,
    /// A run drawn faint beside this one.
    compare: Option<Setup>,
    runs: Vec<Run>,
    film: Film,
    /// The film's speed: 1 plays a run in `FILM_SECONDS`.
    speed: f32,
    look: Look,
    /// The window last drawn in (a new one is drawn at once, without gliding).
    canvas: Option<Vec2>,
    /// Where the charts' labels were put.
    placed: Vec<Placed>,
    labels_moving: bool,
    /// What the third chart showed last, kept while it glides out.
    third_was: Third,
    /// The ticks across the three charts, and up the sharpness's and the third's.
    sticky: [Sticky; 5],
}

impl Eos {
    /// (Nothing here needs the graphics device, so the tests can make one too.)
    fn create() -> Eos {
        let mut s = Eos {
            setup: Setup::base(),
            compare: None,
            runs: Vec::new(),
            film: Film::default(),
            speed: 1.0,
            look: Look::default(),
            canvas: None,
            placed: Vec::new(),
            labels_moving: false,
            third_was: Third::None,
            sticky: [Sticky::default(); 5],
        };
        s.set_up(OPENING);
        s
    }

    /// Set up experiment k, and play it.
    fn set_up(&mut self, k: usize) {
        let (setup, compare) = trial(k);
        self.setup = setup;
        self.compare = compare;
        self.speed = 1.0;
        self.restart();
    }

    /// Start the runs again from their start, and the film once the picture has
    /// glided there.
    fn restart(&mut self) {
        self.setup = self.setup.kept();
        self.compare = self.compare.map(Setup::kept);
        self.runs = std::iter::once(self.setup).chain(self.compare).map(Run::new).collect();
        self.film = Film { shown: 0.0, playing: false, waiting: true, waited: 0.0 };
        // (the film waits until the picture has glided to the new start: until the
        // next glide says it is there)
        self.look.moving = true;
    }

    /// Bring the runs up to date with the settings: a changed setting starts them again.
    fn refresh(&mut self) {
        let s = self.setup.kept();
        let c = self.compare.map(Setup::kept);
        let same = self.runs.first().is_some_and(|r| r.setup == s) && self.runs.get(1).map(|r| r.setup) == c && self.runs.len() == 1 + c.is_some() as usize;
        self.setup = s;
        self.compare = c;
        if !same {
            self.restart();
        }
    }

    /// The longest of the runs, in steps (the charts reach across it).
    fn steps(&self) -> usize {
        self.runs.iter().map(|r| r.setup.steps).max().unwrap_or(self.setup.steps)
    }

    /// The last step the film can show: as far as every run has been measured (a run
    /// that blew up, or is over, holds nothing back).
    fn ready(&self) -> f64 {
        let end = self.steps() as f64;
        self.runs
            .iter()
            .enumerate()
            .map(|(k, r)| match (r.done(), k) {
                (true, _) => end,
                // (the slice drawn is the first run's, through its slices round the step
                // shown: not past the last but one recorded)
                (false, 0) => r.slices.len().checked_sub(2).map_or(0.0, |j| r.slices[j].step as f64).min(r.measured().saturating_sub(1) as f64),
                (false, _) => r.measured().saturating_sub(1) as f64,
            })
            .fold(f64::INFINITY, f64::min)
            .min(end)
    }

    /// Is the film playing (or about to)?
    fn busy(&self) -> bool {
        self.film.playing || self.film.waiting
    }

    /// Steps per second of the scene's time.
    fn rate(&self) -> f64 {
        self.steps() as f64 / FILM_SECONDS * self.speed.max(0.01) as f64
    }

    /// Move the film on by `dt` seconds of the scene's time, computing the runs as far
    /// as it shows (within `budget`: the film waits for the computation).
    fn advance(&mut self, dt: f32, budget: &Budget) {
        // (the film starts once the picture has glided to the new start, or a moment
        // after the change if the glide is a long one: in either case while things still
        // move, so that the glide and the film are one movement)
        if self.film.waiting {
            self.film.waited += dt.max(0.0);
            if !self.look.moving || self.film.waited >= FILM_WAIT {
                self.film.waiting = false;
                self.film.playing = true;
            }
        }
        let end = self.steps() as f64;
        let want = if self.film.playing { (self.film.shown + self.rate() * dt.max(0.0) as f64).min(end) } else { self.film.shown };
        // what the picture needs: the step it shows, and the next two slices after it
        let need = (want.floor() as usize + 2 * SLICE_EVERY + 1).min(self.steps());
        let mut first = true;
        // (the run that is furthest behind goes first; at least one step a frame)
        while let Some(r) = self.runs.iter_mut().filter(|r| !r.done() && r.measured() <= need).min_by_key(|r| r.measured()) {
            if !first && budget.spent() {
                break;
            }
            r.step();
            first = false;
        }
        self.film.shown = want.min(self.ready()).max(0.0);
        if self.film.playing && self.film.shown >= end - 1e-9 {
            self.film.playing = false;
        }
    }

    /// Play from where the film is (from the start, if it is at the end).
    fn play(&mut self) {
        if self.film.shown >= self.steps() as f64 - 1e-9 {
            self.film.shown = 0.0;
        }
        self.film.playing = true;
        self.film.waiting = false;
    }

    /// Keep this run as the one drawn faint for comparison.
    fn keep(&mut self) {
        if self.compare != Some(self.setup) {
            self.compare = Some(self.setup);
            self.refresh();
        }
    }
}

// ----- the layout ------------------------------------------------------------------------

/// Where everything goes in a window (logical pixels from its top-left corner).
#[derive(Clone, Copy, Debug, PartialEq)]
struct Layout {
    view: Vec2,
    narrow: bool,
    /// Text sizes: titles, small print.
    text: f32,
    small: f32,
    /// The buttons: play (or pause), from the start, keep this run.
    buttons: [Rect; 3],
    /// The charts' rectangles (their plots), the third only if it is shown.
    loss: Rect,
    sharp: Rect,
    third: Option<Rect>,
    /// The slice's plot, and where its key starts and how wide it may be.
    slice: Rect,
    key: (Vec2, f32),
    /// Only the sharpness chart: the window is too small for the rest.
    compact: bool,
}

/// Room round a chart's plot: for the numbers at its left, its title above and the
/// step numbers below.
const LEFT: f32 = 46.0;
const ABOVE: f32 = 26.0;
const BELOW: f32 = 22.0;

impl Layout {
    /// `third` is how far the third chart has come in: 0 none, 1 all of it (it glides
    /// in and out as experiments change, and the other two make room for it).
    fn of(view: Vec2, inset: f32, third: f32) -> Layout {
        let third = if third.is_finite() { third.max(0.0).min(1.0) } else { 0.0 };
        let view = view.max(Vec2::splat(1.0));
        let narrow = view.x < 700.0;
        let (text, small) = if narrow { (12.0, 10.5) } else { (13.5, 11.5) };
        let side = if narrow { 10.0 } else { 20.0 };
        let top = inset.max(0.0) + if narrow { 6.0 } else { 10.0 };
        let r = |x0: f32, y0: f32, x1: f32, y1: f32| (Vec2::new(x0, y0), Vec2::new(x1.max(x0 + 1.0), y1.max(y0 + 1.0)));
        let none = r(0.0, 0.0, 0.0, 0.0);
        let left = if narrow { 40.0 } else { LEFT };
        let button_h = if narrow { 24.0 } else { 26.0 };
        let mut lay = Layout { view, narrow, text, small, buttons: [none; 3], loss: none, sharp: none, third: None, slice: none, key: (Vec2::ZERO, 1.0), compact: false };
        if view.y - top < 330.0 || view.x < 250.0 {
            // the sharpness alone
            lay.compact = true;
            lay.sharp = r(side + left, top + ABOVE, view.x - side - 6.0, view.y - BELOW - 8.0);
            return lay;
        }
        let row = plot::even_row(3, side, side + if narrow { view.x - 2.0 * side } else { 420.0f32.min(view.x * 0.5) }, top, button_h, 6.0);
        for (k, b) in row.into_iter().enumerate().take(3) {
            lay.buttons[k] = b;
        }
        let y0 = top + button_h + 8.0;
        // the charts, one under another, from `a` to `b` down and `x0` to `x1` across
        let charts = |lay: &mut Layout, x0: f32, x1: f32, a: f32, b: f32| {
            let (lx, rx) = (x0 + left, x1 - 6.0);
            let gap = ABOVE + BELOW + 6.0;
            let shares = [0.47 - 0.09 * third, 0.53 - 0.13 * third, 0.22 * third];
            let h = (b - a - gap * (2.0 + third)).max(3.0);
            let mut y = a + ABOVE;
            let mut rects = Vec::new();
            for (k, share) in shares.iter().enumerate() {
                let rh = h * share;
                rects.push(r(lx, y, rx, y + rh));
                y += rh + gap * if k == 1 { third } else { 1.0 };
            }
            lay.loss = rects[0];
            lay.sharp = rects[1];
            lay.third = (third > 0.0).then_some(rects[2]);
        };
        let wide = view.x >= 640.0 && view.x >= 0.95 * view.y;
        if wide {
            // the charts on the left, the slice on the right with its key under it
            let split = side + (view.x - 2.0 * side) * 0.6;
            charts(&mut lay, side, split, y0, view.y - 6.0);
            let (sx0, sx1) = (split + 14.0 + left, view.x - side - 6.0);
            let key = key_lines(false, sx1 - sx0 + left - 6.0, small) * (small + 4.0) + small + 14.0;
            let h = (view.y - y0 - ABOVE - BELOW - key - 10.0).min((sx1 - sx0) * 1.3).max(3.0);
            lay.slice = r(sx0, y0 + ABOVE, sx1, y0 + ABOVE + h);
            lay.key = (Vec2::new(sx0 - left + 6.0, lay.slice.1.y + BELOW + small + 2.0), sx1 - sx0 + left - 6.0);
        } else {
            // the slice on top, its key, the charts under it
            let h = ((view.y - y0) * if narrow { 0.28 } else { 0.3 }).min(view.x * 0.6);
            lay.slice = r(side + left, y0 + ABOVE, view.x - side - 6.0, y0 + ABOVE + h);
            let width = view.x - 2.0 * side - 6.0;
            let key = key_lines(narrow, width, small) * (small + 4.0) + if narrow { 4.0 } else { small + 8.0 };
            lay.key = (Vec2::new(side + 6.0, lay.slice.1.y + BELOW + if narrow { 0.0 } else { small + 2.0 }), width);
            let below = lay.key.0.y + key;
            charts(&mut lay, side, view.x - side, below, view.y - 4.0);
        }
        if lay.loss.1.y - lay.loss.0.y < 60.0 || lay.slice.1.y - lay.slice.0.y < 60.0 {
            // too little room for all of it: the sharpness alone
            return Layout { compact: true, sharp: r(side + left, top + ABOVE, view.x - side - 6.0, view.y - BELOW - 8.0), ..lay };
        }
        lay
    }
}

/// The slice's key: what each colour on it is.
fn slice_key(narrow: bool, level: &str, dots: Color, preconditioned: bool) -> [(Color, String); 4] {
    let sharpness = if preconditioned { r"the preconditioned sharpness $\lambda_1(P^{-1}H)$" } else { "the sharpness $S$" };
    if narrow {
        [
            (pal::TEXT, "white: the loss along the direction".to_string()),
            (CURVATURE, format!("green: a parabola whose curvature is {sharpness}")),
            (LIMIT, format!("yellow dashed: curvature ${level}$, the most this step bears")),
            (dots, format!("dots: this iterate (largest) and the {TRAIL} before it")),
        ]
    } else {
        [
            (pal::TEXT, "white: the loss along the sharpest direction, through the midpoint of the last two steps".to_string()),
            (CURVATURE, format!("green: the parabola that matches it at the middle, whose curvature is {sharpness}")),
            (LIMIT, format!("yellow dashed: a parabola of curvature ${level}$, the sharpest valley this step can bear")),
            (dots, format!("dots: this iterate (largest) and the {TRAIL} before it, placed by how far along the direction they are")),
        ]
    }
}

/// How many lines the slice's key takes in a column `width` pixels wide.
fn key_lines(narrow: bool, width: f32, size: f32) -> f32 {
    slice_key(narrow, r"38/\eta", pal::TEXT, true).iter().map(|(_, t)| (text_width(t, size) / width.max(50.0)).ceil().max(1.0)).sum()
}

/// Words broken into lines no wider than `width` pixels (a formula is never broken),
/// drawn from `at` down, each line `size + 4` below the last. Returns the height used.
fn key_px(f: &mut Frame, at: Vec2, width: f32, text: &str, size: f32, color: Color) -> f32 {
    let mut words: Vec<&str> = Vec::new();
    let (mut start, mut inside) = (0, false);
    for (i, c) in text.char_indices() {
        match c {
            '$' => inside = !inside,
            ' ' if !inside => {
                words.push(&text[start..i]);
                start = i + 1;
            }
            _ => {}
        }
    }
    words.push(&text[start..]);
    let mut lines: Vec<String> = Vec::new();
    let mut line = String::new();
    for w in words {
        let candidate = if line.is_empty() { w.to_string() } else { format!("{line} {w}") };
        if !line.is_empty() && f.text_size(&candidate, size).x > width {
            lines.push(std::mem::replace(&mut line, w.to_string()));
        } else {
            line = candidate;
        }
    }
    lines.push(line);
    let mut y = at.y;
    for l in lines {
        f.text_px(Vec2::new(at.x, y), l, size, color, Align2::LEFT_TOP);
        y += size + 4.0;
    }
    y - at.y
}

// ----- what is drawn, gliding ----------------------------------------------------------------

/// What is drawn, gliding to where it is going.
#[derive(Clone, Debug, Default)]
struct Look {
    /// The charts' ranges: across (to), the loss's (log₁₀, from and to), the
    /// sharpness's top, the third chart's top.
    frame: Glide,
    /// The slice, in its own scale (see `Eos::slice_goal`): the logarithms of its reach
    /// and its height, the two parabolas' bottom (across, up) and the logarithms of their
    /// curvatures, then `SLICE_N` heights and the iterates (across, up each).
    slice: Glide,
    /// How far the third chart has come in (0 to 1).
    third: Glide,
    moving: bool,
}

/// The numbers the slice's glide keeps before its `SLICE_N` heights (see `slice_goal`).
const SLICE_G: usize = 6;

/// The slice glides this many times as fast as `Glide` does by itself.
const SLICE_PACE: f32 = 0.4;

// The charts' ranges, in the order the glide keeps them.
const X_HI: usize = 0;
const LOSS_LO: usize = 1;
const LOSS_HI: usize = 2;
const SHARP_HI: usize = 3;
const THIRD_HI: usize = 4;

/// What the third chart shows.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Third {
    /// Nothing: there is no third chart.
    None,
    /// The step size (a run has a cut).
    Step,
    /// Adam's sharpness without its preconditioner.
    Plain,
}

impl Eos {
    fn third(&self) -> Third {
        if self.runs.iter().any(|r| r.setup.preconditioned()) {
            Third::Plain
        } else if self.runs.iter().any(|r| r.setup.cut.is_some()) {
            Third::Step
        } else {
            Third::None
        }
    }

    /// Whether run k is drawn on the sharpness chart: its sharpness is of the same kind
    /// as the first run's (both preconditioned, or neither).
    fn on_sharp_chart(&self, k: usize) -> bool {
        self.runs.get(k).is_some_and(|r| r.setup.preconditioned() == self.setup.preconditioned())
    }

    /// The charts' ranges as they are to be, from what the film has shown so far.
    fn frame_goal(&self) -> Vec<f32> {
        let shown = self.film.shown.floor() as usize;
        let (mut lo, mut hi) = (f64::INFINITY, 0.0f64);
        let (mut top, mut third) = (0.0f64, 0.0f64);
        for (k, r) in self.runs.iter().enumerate() {
            let n = (shown + 1).min(r.measured());
            for t in 0..n {
                let l = r.loss[t];
                if l.is_finite() && l > 0.0 {
                    lo = lo.min(l);
                    hi = hi.max(l);
                }
            }
            if self.on_sharp_chart(k) {
                // (the level over the whole run, and what the sharpness has reached, up
                // to two and a half times the level: Adam's first steps are far above it)
                let most = (0..=r.setup.steps).map(|t| r.setup.level_at(t)).fold(0.0, f64::max);
                top = top.max(most);
                for t in 0..n {
                    let v = r.main(t);
                    if v.is_finite() {
                        top = top.max(v.min(2.5 * most));
                    }
                }
            }
            match self.third() {
                Third::Plain => {
                    for t in 0..n {
                        if r.sharp[t].is_finite() {
                            third = third.max(r.sharp[t]);
                        }
                    }
                }
                Third::Step => third = third.max(r.setup.eta),
                Third::None => {}
            }
        }
        if !(lo.is_finite() && lo > 0.0) {
            lo = 0.1;
        }
        if hi.is_nan() || hi <= 0.0 {
            hi = 1.0;
        }
        // (from under the lowest loss to over the highest, at least a decade and a half)
        let (mut y0, y1) = ((lo.log10() - 0.25).min(-1.5), hi.log10() + 0.2);
        if y1 - y0 < 1.5 {
            y0 = y1 - 1.5;
        }
        let mut g = vec![0.0f32; 5];
        g[X_HI] = self.steps().max(1) as f32;
        g[LOSS_LO] = y0 as f32;
        g[LOSS_HI] = y1 as f32;
        // (the tops as logarithms: from one experiment to the next they can change a
        // hundredfold, and glide evenly so)
        g[SHARP_HI] = (top.max(1.0) * 1.12).ln() as f32;
        g[THIRD_HI] = (third.max(1e-6) * 1.15).ln() as f32;
        g
    }

    /// The slice as it is to be: through run 0's slices round the step shown, a curve
    /// with no corners (Catmull and Rom's, through the slices before and after), so
    /// that it moves smoothly while the film plays: the slice's reach on a logarithmic
    /// scale, where its iterates are as shares of it, and the losses.
    fn slice_goal(&self) -> Vec<f32> {
        let Some(r) = self.runs.first() else { return Vec::new() };
        let s = self.film.shown;
        let sl = &r.slices;
        if sl.is_empty() {
            return Vec::new();
        }
        // (before the first slice, the first)
        let i = sl.iter().rposition(|x| x.step as f64 <= s + 1e-9).unwrap_or(0);
        let at = |j: isize| &sl[j.clamp(0, sl.len() as isize - 1) as usize];
        let (p0, p1, p2, p3) = (at(i as isize - 1), at(i as isize), at(i as isize + 1), at(i as isize + 2));
        let u = if p2.step > p1.step { ((s - p1.step as f64) / (p2.step - p1.step) as f64).clamp(0.0, 1.0) } else { 0.0 };
        let cr = |a: f64, b: f64, c: f64, d: f64| 0.5 * (2.0 * b + (c - a) * u + (2.0 * a - 5.0 * b + 4.0 * c - d) * u * u + (3.0 * b - a - 3.0 * c + d) * u * u * u);
        let each = |f: &dyn Fn(&Slice) -> f64| cr(f(p0), f(p1), f(p2), f(p3));
        let range = each(&|x| x.range.max(1e-300).ln()).exp();
        let values: Vec<f64> = (0..SLICE_N).map(|k| each(&|x| x.values[k])).collect();
        let iterates: Vec<(f64, f64)> = (0..=TRAIL).map(|k| (range * each(&|x| x.iterates[k].0 / x.range), each(&|x| x.iterates[k].1))).collect();
        let level = p1.level + (p2.level - p1.level) * u;
        // (on a logarithmic scale: from one slice to the next it can change a
        // hundredfold, Adam's in its first steps, and a cubic through such numbers would
        // dip below zero between them)
        let curvature = each(&|x| x.curvature.max(1e-300).ln()).exp();
        // the two parabolas share a bottom: that of the parabola which matches the
        // slice at its middle (its value and slope there, its curvature the sharpness)
        let h = 2.0 * range / (SLICE_N - 1) as f64;
        let m = SLICE_N / 2;
        let slope = (values[m + 1] - values[m - 1]) / (2.0 * h);
        let (mut vx, mut vy) = (0.0, values[m]);
        if curvature > 0.0 {
            vx = (-slope / curvature).max(-range).min(range);
            vy = values[m] + slope * vx + 0.5 * curvature * vx * vx;
        }
        // the picture, in its own scale: across, the distance over the reach (−1 to 1);
        // up, the loss above the slice's lowest point over the height of its highest
        // (0 to 1); the scales themselves as logarithms. So what glides is a shape of
        // fixed size and two scales, smoothly even when a leap of the loss changes the
        // scale a hundredfold in a few frames.
        let low = values.iter().copied().fold(vy, f64::min);
        let height = (values.iter().copied().fold(f64::NEG_INFINITY, f64::max) - low).max(1e-300);
        let k = |c: f64| (c * range * range / height).max(1e-300).ln();
        let mut g: Vec<f32> = vec![range.ln() as f32, height.ln() as f32, (vx / range) as f32, ((vy - low) / height) as f32, k(curvature) as f32, k(level) as f32];
        g.extend(values.iter().map(|v| ((v - low) / height) as f32));
        for (x, y) in &iterates {
            g.push((x / range) as f32);
            g.push(((y - low) / height) as f32);
        }
        if g.iter().any(|v| !v.is_finite()) {
            return Vec::new();
        }
        g
    }

    /// The slice's numbers as they are now (not glided: what the captions write): its
    /// reach, its lowest loss and the height above it, its curvature and its level.
    fn slice_now(&self) -> Option<(f64, f64, f64, f64, f64)> {
        let r = self.runs.first()?;
        let s = self.film.shown;
        let a = r.slices.iter().rev().find(|x| x.step as f64 <= s + 1e-9).or(r.slices.first())?;
        let low = a.values.iter().copied().fold(f64::INFINITY, f64::min);
        let height = a.values.iter().copied().fold(f64::NEG_INFINITY, f64::max) - low;
        Some((a.range, low, height, a.curvature, a.level))
    }

    /// Bring the picture `dt` seconds nearer to what it shows. (`snap`: there at once.)
    fn follow(&mut self, dt: f32, snap: bool) {
        let frame = self.frame_goal();
        let slice = self.slice_goal();
        let third = [if self.third() == Third::None { 0.0 } else { 1.0 }];
        let look = &mut self.look;
        if snap {
            look.frame.snap(&frame);
            look.slice.snap(&slice);
            look.third.snap(&third);
            look.moving = false;
            return;
        }
        let mut moving = look.frame.follow(&frame, dt, false);
        moving |= look.third.follow(&third, dt, false);
        // (the slice more slowly than the glide's own pace: a leap of the loss swings it
        // through a hundredfold change of scale in a few frames, which it follows as one
        // smooth movement)
        moving |= look.slice.follow(&slice, dt * SLICE_PACE, false);
        look.moving = moving;
    }
}

// ----- drawing ---------------------------------------------------------------------------------

// Lines whose number of points changes, each in a batch of its own.
const CURVE: u8 = plot::OWN;
/// (the loss, the sharpness and the third chart's, for two runs each)
const LEVEL: u8 = CURVE + 6;
const NOW: u8 = LEVEL + 1;
const SLICE: u8 = NOW + 1;
const VALLEY: u8 = SLICE + 1;
const OSCULATING: u8 = VALLEY + 1;
const TRAIL_LINE: u8 = OSCULATING + 1;
/// Each chart's axes and ticks in a batch of their own: one chart's ticks can come and
/// go while another's do too, and in one batch the rest would seem to jump.
const LOSS_AXES: u8 = TRAIL_LINE + 1;
const SHARP_AXES: u8 = LOSS_AXES + 1;
const THIRD_AXES: u8 = SHARP_AXES + 1;
const SLICE_AXES: u8 = THIRD_AXES + 1;
const LAST: u8 = SLICE_AXES;

/// The ticks of an axis whose range glides, keeping their spacing while it stays
/// between `CLOSEST` and `FURTHEST` pixels: a new spacing then gives a different number
/// of ticks, so that no tick seems to jump to where another was (as it would if the
/// spacing changed and the count did not).
#[derive(Clone, Copy, Debug, Default)]
struct Sticky {
    step: f64,
}

const CLOSEST: f32 = 28.0;
const FURTHEST: f32 = 140.0;

impl Sticky {
    /// The values of the ticks from `lo` to `hi` over `pixels`, and how many decimals
    /// their names need.
    fn ticks(&mut self, lo: f64, hi: f64, pixels: f32) -> (Vec<f64>, usize) {
        let span = hi - lo;
        if !(span.is_finite() && span > 0.0 && pixels > 1.0) {
            return (Vec::new(), 0);
        }
        let apart = |step: f64| pixels * (step / span) as f32;
        if !(self.step > 0.0 && (CLOSEST..=FURTHEST).contains(&apart(self.step))) {
            self.step = plot::round_step(span / (pixels as f64 / 60.0).max(1.0));
        }
        let step = self.step;
        let (first, last) = ((lo / step - 1e-9).ceil() as i64, ((hi / step + 1e-9).floor() as i64).min((lo / step).ceil() as i64 + 60));
        let decimals = (-(step * (1.0 + 1e-9)).log10().floor()).max(0.0).min(6.0) as usize;
        ((first..=last).map(|k| k as f64 * step).collect(), decimals)
    }
}

/// The point batches: the iterates on the slice, the batch sharpness.
fn dot_batch<'b>(f: &'b mut Frame<'_>, k: usize) -> &'b mut crate::gfx::draw::PointBatch {
    f.dl.points_with(PointKind::Overlay, PointStyle { outline_px: 0.0, size_scale: 1.0 + 1e-4 * (k + 1) as f32, ..Default::default() })
}

/// Every batch the scene draws in, made in the same order in every frame, so that one
/// that is empty in a frame does not move the later ones up a place (see `eidos motion`).
fn hold_batches(f: &mut Frame) {
    let corner = f.hud(Vec2::ZERO);
    for kind in [AXES, MARKS, plot::BUTTONS].into_iter().chain(CURVE..=LAST) {
        plot::apart(f, kind, false).segment(corner, corner, 1.0, Color::CLEAR);
    }
    for kind in [LEVEL, VALLEY] {
        plot::apart(f, kind, true).segment(corner, corner, 1.0, Color::CLEAR);
    }
    for k in 0..3 {
        dot_batch(f, k);
    }
}

/// Dots in screen space, in batch k.
fn dots_px(f: &mut Frame, k: usize, pts: &[(Vec2, f32, Color)]) {
    let at: Vec<(Vec3, f32, Color)> = pts.iter().filter(|p| p.0.is_finite()).map(|p| (f.hud(p.0), p.1, p.2)).collect();
    let b = dot_batch(f, k);
    for (p, size, color) in at {
        b.push(p, size, color);
    }
}

/// A run's values over steps `0 ..= shown`, as points of a chart `pixels` wide: where
/// several steps share a column of pixels, the lowest and the highest of them, in the
/// order they came, so that a curve that swings from step to step is drawn as the band
/// it fills. Only whole columns are drawn: the curve grows a column at a time, and
/// what it has drawn never moves (the film's place is the line across the charts).
fn envelope(values: &[f64], shown: f64, steps: usize, pixels: f32, map: impl Fn(f64) -> f64) -> Vec<(f64, f64)> {
    let last = (shown.max(0.0).floor() as usize).min(values.len().saturating_sub(1));
    if values.is_empty() {
        return Vec::new();
    }
    let columns = (pixels.max(8.0) as usize).max(1);
    let per = steps.max(1) as f64 / columns as f64;
    let at = |v: f64| if v.is_finite() { v } else { f64::NAN };
    if per <= 2.0 {
        return (0..=last).map(|t| (t as f64, at(map(values[t])))).collect();
    }
    // the columns whose steps have all been shown
    let whole = ((last + 1) as f64 / per).floor() as usize;
    let mut out: Vec<(f64, f64)> = Vec::with_capacity(2 * whole + 2);
    for column in 0..whole {
        let (a, b) = ((column as f64 * per).ceil() as usize, (((column + 1) as f64 * per).ceil() as usize).min(last + 1));
        let (mut lo, mut hi) = ((a, f64::INFINITY), (a, f64::NEG_INFINITY));
        for (i, v) in values.iter().enumerate().take(b).skip(a) {
            let v = map(*v);
            if v.is_finite() {
                if v < lo.1 {
                    lo = (i, v);
                }
                if v > hi.1 {
                    hi = (i, v);
                }
            }
        }
        if lo.1.is_finite() {
            let (p, q) = if lo.0 <= hi.0 { (lo, hi) } else { (hi, lo) };
            out.push((p.0 as f64, p.1));
            out.push((q.0 as f64, q.1));
        } else {
            out.push((a as f64, f64::NAN));
            out.push((a as f64, f64::NAN));
        }
    }
    out
}

/// A run's step size over its steps, as the corners of a line of steps.
fn eta_points(s: &Setup) -> Vec<(f64, f64)> {
    let end = s.steps as f64;
    match s.cut {
        Some(c) if c < s.steps => vec![(0.0, s.eta_at(0)), (c as f64, s.eta_at(0)), (c as f64, s.eta_at(c)), (end, s.eta_at(c))],
        _ => vec![(0.0, s.eta), (end, s.eta)],
    }
}

/// A run's level over its steps, as the corners of a line of steps.
fn level_points(s: &Setup) -> Vec<(f64, f64)> {
    let end = s.steps as f64;
    match s.cut {
        Some(c) if c < s.steps => vec![(0.0, s.level_at(0)), (c as f64, s.level_at(0)), (c as f64, s.level_at(c)), (end, s.level_at(c))],
        _ => vec![(0.0, s.level_at(0)), (end, s.level_at(0))],
    }
}

impl Eos {
    fn loss_plot(&self, lay: &Layout) -> Plot {
        let g = |i: usize| self.look.frame.get(i) as f64;
        Plot::new(lay.loss, [0.0, g(X_HI)], [g(LOSS_LO), g(LOSS_HI)])
    }

    fn sharp_plot(&self, lay: &Layout) -> Plot {
        let g = |i: usize| self.look.frame.get(i) as f64;
        Plot::new(lay.sharp, [0.0, g(X_HI)], [0.0, g(SHARP_HI).exp()])
    }

    fn third_plot(&self, lay: &Layout) -> Option<Plot> {
        let g = |i: usize| self.look.frame.get(i) as f64;
        lay.third.map(|r| Plot::new(r, [0.0, g(X_HI)], [0.0, g(THIRD_HI).exp()]))
    }

    /// The buttons over the picture; true if anything changed.
    fn buttons(&mut self, f: &mut Frame, lay: &Layout) -> bool {
        let size = lay.small;
        let mut changed = false;
        let busy = self.busy();
        let at_end = self.film.shown >= self.steps() as f64 - 1e-9;
        let play = if busy {
            "❚❚ pause"
        } else if at_end {
            "▶ again"
        } else {
            "▶ play"
        };
        if plot::button(f, lay.buttons[0], play, size, false) {
            if busy {
                self.film.playing = false;
                self.film.waiting = false;
            } else {
                self.play();
            }
            changed = true;
        }
        if plot::button(f, lay.buttons[1], if lay.narrow { "restart" } else { "from the start" }, size, false) {
            self.restart();
            changed = true;
        }
        if plot::button(f, lay.buttons[2], if lay.narrow { "keep" } else { "keep this run" }, size, false) {
            self.keep();
            changed = true;
        }
        changed
    }

    /// The step axis under a chart: its ticks, and (`named`) its name.
    fn step_axis(f: &mut Frame, kind: u8, sticky: &mut Sticky, p: &Plot, size: f32, named: bool) {
        let (min, max) = p.rect;
        plot::line_of(f, kind, Vec2::new(min.x, max.y), max, 1.0, pal::DIM.a(0.8));
        let (values, _) = sticky.ticks(0.0, p.x[1], p.width() * 0.8);
        for v in values {
            let x = p.px_x(v);
            if x > max.x + 0.5 {
                continue;
            }
            plot::line_of(f, kind, Vec2::new(x, max.y), Vec2::new(x, max.y + 3.0), 1.0, pal::DIM.a(0.8));
            f.text_px(Vec2::new(x, max.y + 4.0), count(v.round().max(0.0) as usize), size, pal::DIM, Align2::CENTER_TOP);
        }
        if named {
            f.text_px(Vec2::new(max.x, max.y + 5.0 + size), "step", size, pal::DIM, Align2::RIGHT_TOP);
        }
    }

    /// Where the film is, across a chart.
    fn now_line(&self, f: &mut Frame, p: &Plot) {
        // (drawn from the start, at step 0 too, so that its batch never gains a line)
        let x = self.film.shown;
        if x >= 0.0 && x <= p.x[1] {
            let (min, max) = p.rect;
            let at = p.px_x(x);
            plot::line_of(f, NOW, Vec2::new(at, min.y), Vec2::new(at, max.y), 1.0, pal::TEXT.a(0.3));
        }
    }

    /// Run k's colour on a chart: its own, faint for the run compared.
    fn shade(k: usize, c: Color) -> Color {
        if k == 0 {
            c
        } else {
            c.a(0.42)
        }
    }

    fn draw_loss(&mut self, f: &mut Frame, lay: &Layout) {
        let p = self.loss_plot(lay);
        if p.tiny() {
            return;
        }
        let (min, max) = p.rect;
        let size = lay.small;
        p.backing(f, if lay.narrow { 38.0 } else { 44.0 }, BELOW - 2.0, ABOVE - 2.0, 6.0);
        let title = if lay.narrow { "loss (on all 20 points)" } else { "the loss on all 20 points, at every step (logarithmic)" };
        f.text_px(Vec2::new(min.x - if lay.narrow { 34.0 } else { 40.0 }, min.y - 6.0), title, lay.text, pal::TEXT, Align2::LEFT_BOTTOM);
        plot::line_of(f, LOSS_AXES, min, Vec2::new(min.x, max.y), 1.0, pal::DIM.a(0.8));
        // powers of ten up the side
        let (y0, y1) = (p.y[0], p.y[1]);
        let mut d = y0.ceil() as i32;
        while d as f64 <= y1 + 1e-9 && d < 40 {
            let at = p.px_y(d as f64);
            plot::line_of(f, LOSS_AXES, Vec2::new(min.x - 3.0, at), Vec2::new(min.x, at), 1.0, pal::DIM.a(0.8));
            plot::line_of(f, LOSS_AXES, Vec2::new(min.x, at), Vec2::new(max.x, at), 1.0, pal::DIM.a(0.1));
            let name = match d {
                0 => "1".to_string(),
                _ => format!("$10^{{{d}}}$"),
            };
            f.text_px(Vec2::new(min.x - 5.0, at), name, size, pal::DIM, Align2::RIGHT_CENTER);
            d += 1;
        }
        Self::step_axis(f, LOSS_AXES, &mut self.sticky[0], &p, size, false);
        let shown = self.film.shown;
        for (k, r) in self.runs.iter().enumerate().rev() {
            let c = Self::shade(k, r.setup.rule.color());
            let pts = envelope(&r.loss, shown, self.steps(), p.width(), |v| if v > 0.0 { v.log10() } else { f64::NAN });
            p.path(f, CURVE + k.min(1) as u8, pts.iter().copied(), if k == 0 { 1.6 } else { 1.2 }, c);
        }
        // which run is which (in every chart), in the corner: a key that stays put
        let mut y = min.y + 4.0;
        for (k, r) in self.runs.iter().enumerate() {
            let c = Self::shade(k, r.setup.rule.color());
            let other = self.runs.get(1 - k.min(1)).filter(|_| self.runs.len() > 1).map(|o| &o.setup);
            let mut text = r.setup.tag(other);
            if self.runs.len() > 1 {
                text = format!("{}: {text}", if k == 0 { "bright" } else { "faint" });
            }
            if r.blown.is_some_and(|b| (b as f64) <= shown) {
                text += ", blew up";
            }
            f.caption_px(Vec2::new(max.x - 6.0, y), text, size, if k == 0 { c } else { c.a(0.8) }, Align2::RIGHT_TOP);
            y += size + 7.0;
        }
        self.now_line(f, &p);
    }

    fn draw_sharp(&mut self, f: &mut Frame, lay: &Layout, labels: &mut Vec<ChartLabel>) {
        let p = self.sharp_plot(lay);
        if p.tiny() {
            return;
        }
        let (min, max) = p.rect;
        let size = lay.small;
        p.backing(f, if lay.narrow { 38.0 } else { 44.0 }, BELOW - 2.0, ABOVE - 2.0, 6.0);
        let pre = self.setup.preconditioned();
        let title = match (pre, lay.narrow) {
            (false, false) => "the sharpness: the Hessian's largest eigenvalue, at every step",
            (false, true) => "sharpness",
            (true, false) => r"the preconditioned sharpness: the largest eigenvalue of $P^{-1}H$",
            (true, true) => "preconditioned sharpness",
        };
        f.text_px(Vec2::new(min.x - if lay.narrow { 34.0 } else { 40.0 }, min.y - 6.0), title, lay.text, pal::TEXT, Align2::LEFT_BOTTOM);
        plot::line_of(f, SHARP_AXES, min, Vec2::new(min.x, max.y), 1.0, pal::DIM.a(0.8));
        let (values, decimals) = self.sticky[3].ticks(0.0, p.y[1], p.height() * 1.2);
        for v in values {
            let at = p.px_y(v);
            if at < min.y - 0.5 {
                continue;
            }
            plot::line_of(f, SHARP_AXES, Vec2::new(min.x - 3.0, at), Vec2::new(min.x, at), 1.0, pal::DIM.a(0.8));
            plot::line_of(f, SHARP_AXES, Vec2::new(min.x, at), Vec2::new(max.x, at), 1.0, pal::DIM.a(0.1));
            f.text_px(Vec2::new(min.x - 5.0, at), if v >= 1000.0 { count(v.round() as usize) } else { num(v, decimals) }, size, pal::DIM, Align2::RIGHT_CENTER);
        }
        Self::step_axis(f, SHARP_AXES, &mut self.sticky[1], &p, size, lay.third.is_none() || lay.compact);
        let shown = self.film.shown;
        for (k, r) in self.runs.iter().enumerate().rev() {
            if !self.on_sharp_chart(k) {
                continue;
            }
            let s = &r.setup;
            // the level, over the whole run
            let lvl = level_points(s);
            let lc = Self::shade(k, LIMIT);
            let pts: Vec<Vec2> = lvl.iter().map(|&(x, y)| p.at(x, y)).collect();
            for w in pts.windows(2) {
                if let Some((a, b)) = clip_segment(w[0], w[1], p.rect) {
                    plot::dashed_of(f, LEVEL, a, b, if k == 0 { 1.5 } else { 1.1 }, lc);
                }
            }
            let last = lvl.last().copied().unwrap_or((0.0, 0.0));
            let at_level = p.at(last.0, last.1);
            if at_level.y > min.y + 4.0 {
                let name = format!(r"${} = {}$", s.level_name(), short(last.1));
                labels.push(ChartLabel { at: Vec2::new(max.x - 4.0, at_level.y - 3.0), align: Align2::RIGHT_BOTTOM, text: name, color: lc, area: 1 });
            }
            // the sharpness
            let c = Self::shade(k, CURVATURE);
            let values = if s.preconditioned() { &r.psharp } else { &r.sharp };
            let pts = envelope(values, shown, self.steps(), p.width(), |v| v);
            p.path(f, CURVE + 2 + k.min(1) as u8, pts.iter().copied(), if k == 0 { 1.6 } else { 1.2 }, c);
            // the batch sharpness, as dots
            if k == 0 && !r.batch.is_empty() {
                let dots: Vec<(Vec2, f32, Color)> = r
                    .batch
                    .iter()
                    .filter(|(t, v, _)| (*t as f64) <= shown && v.is_finite())
                    .map(|(t, v, _)| (p.at(*t as f64, *v), if lay.narrow { 2.6 } else { 3.2 }, NOISE.a(0.9)))
                    .filter(|d| plot::within(d.0, p.rect, 0.5))
                    .collect();
                dots_px(f, 1, &dots);
            }
        }
        self.now_line(f, &p);
    }

    fn draw_third(&mut self, f: &mut Frame, lay: &Layout, labels: &mut Vec<ChartLabel>) {
        let Some(p) = self.third_plot(lay) else { return };
        if p.tiny() {
            return;
        }
        let (min, max) = p.rect;
        let size = lay.small;
        let third = match self.third() {
            Third::None => self.third_was,
            t => t,
        };
        self.third_was = third;
        p.backing(f, if lay.narrow { 38.0 } else { 44.0 }, BELOW - 2.0, ABOVE - 2.0, 6.0);
        let title = match (third, lay.narrow) {
            (Third::Plain, false) => "the sharpness without the preconditioner: the Hessian's largest eigenvalue",
            (Third::Plain, true) => "sharpness, not preconditioned",
            (_, false) => r"the step size $\eta$",
            (_, true) => r"step size $\eta$",
        };
        f.text_px(Vec2::new(min.x - if lay.narrow { 34.0 } else { 40.0 }, min.y - 6.0), title, lay.text, pal::TEXT, Align2::LEFT_BOTTOM);
        plot::line_of(f, THIRD_AXES, min, Vec2::new(min.x, max.y), 1.0, pal::DIM.a(0.8));
        let (values, decimals) = self.sticky[4].ticks(0.0, p.y[1], p.height() * 1.4);
        for v in values {
            let at = p.px_y(v);
            if at < min.y - 0.5 {
                continue;
            }
            plot::line_of(f, THIRD_AXES, Vec2::new(min.x - 3.0, at), Vec2::new(min.x, at), 1.0, pal::DIM.a(0.8));
            f.text_px(Vec2::new(min.x - 5.0, at), num(v, decimals), size, pal::DIM, Align2::RIGHT_CENTER);
        }
        Self::step_axis(f, THIRD_AXES, &mut self.sticky[2], &p, size, true);
        let shown = self.film.shown;
        for (k, r) in self.runs.iter().enumerate().rev() {
            let s = &r.setup;
            match third {
                Third::Plain => {
                    let c = Self::shade(k, CURVATURE);
                    let pts = envelope(&r.sharp, shown, self.steps(), p.width(), |v| v);
                    p.path(f, CURVE + 4 + k.min(1) as u8, pts.iter().copied(), if k == 0 { 1.6 } else { 1.2 }, c);
                }
                _ => {
                    let c = Self::shade(k, s.rule.color());
                    let pts = eta_points(s);
                    p.path(f, CURVE + 4 + k.min(1) as u8, pts.iter().copied(), if k == 0 { 1.6 } else { 1.2 }, c);
                    if let Some(&(x, y)) = pts.last() {
                        let at = p.at(x, y);
                        labels.push(ChartLabel { at: Vec2::new(at.x - 4.0, at.y - 3.0), align: Align2::RIGHT_BOTTOM, text: format!(r"$\eta = {}$", short(y)), color: c, area: 2 });
                    }
                }
            }
        }
        self.now_line(f, &p);
    }

    fn draw_slice(&mut self, f: &mut Frame, lay: &Layout) {
        let (min, max) = lay.slice;
        let g = &self.look.slice.shown;
        if g.len() < SLICE_G + SLICE_N || lay.compact {
            return;
        }
        // (the picture in the slice's own scale: see `slice_goal`; the scales' numbers
        // are the glided ones, so that the ticks say what the picture shows)
        let (range, height) = ((g[0] as f64).exp(), (g[1] as f64).exp());
        let Some((_, low, _, curvature, level)) = self.slice_now() else { return };
        let p = Plot::new(lay.slice, [-1.0, 1.0], [-0.08, 1.08]);
        if p.tiny() {
            return;
        }
        let size = lay.small;
        let r0 = &self.runs[0];
        let s = r0.setup;
        p.backing(f, if lay.narrow { 38.0 } else { 44.0 }, BELOW + if lay.narrow { 0.0 } else { size + 4.0 }, ABOVE - 2.0, 6.0);
        let step = r0.slices.iter().rev().find(|x| x.step as f64 <= self.film.shown + 1e-9).map_or(0, |x| x.step);
        let title = match (s.preconditioned(), lay.narrow) {
            (false, false) => format!("the loss along the sharpest direction, near step {}", count(step)),
            (false, true) => format!("along the sharpest direction, step {}", count(step)),
            (true, false) => format!("along the sharpest direction after preconditioning, near step {}", count(step)),
            (true, true) => format!("sharpest direction, preconditioned, step {}", count(step)),
        };
        f.text_px(Vec2::new(min.x - if lay.narrow { 34.0 } else { 40.0 }, min.y - 6.0), title, lay.text, pal::TEXT, Align2::LEFT_BOTTOM);
        // the axes: the loss up the side, the distance along the direction across
        plot::line_of(f, SLICE_AXES, min, Vec2::new(min.x, max.y), 1.0, pal::DIM.a(0.8));
        plot::line_of(f, SLICE_AXES, Vec2::new(min.x, max.y), max, 1.0, pal::DIM.a(0.8));
        // the ticks stay where they are and their numbers change with the scale, which
        // glides with every change of the slice (ticks at round numbers would jump from
        // one set to another as it does): up the side, the loss above the lowest point
        // of the slice; across, the distance along the direction
        for k in 0..=2 {
            let at = p.px_y(k as f64 / 2.0);
            plot::line_of(f, SLICE_AXES, Vec2::new(min.x - 3.0, at), Vec2::new(min.x, at), 1.0, pal::DIM.a(0.8));
            let name = if k == 0 { "0".to_string() } else { format!("+{} %", short(100.0 * height * k as f64 / 2.0 / low.max(1e-300))) };
            f.text_px(Vec2::new(min.x - 5.0, at), name, size, pal::DIM, Align2::RIGHT_CENTER);
        }
        for k in -2i32..=2 {
            let at = p.px_x(k as f64 / 2.0);
            plot::line_of(f, SLICE_AXES, Vec2::new(at, max.y), Vec2::new(at, max.y + 3.0), 1.0, pal::DIM.a(0.8));
            if k % 2 == 0 || !lay.narrow {
                let name = if k == 0 { "0".to_string() } else { short(range * k as f64 / 2.0) };
                f.text_px(Vec2::new(at, max.y + 4.0), name, size, pal::DIM, Align2::CENTER_TOP);
            }
        }
        // two parabolas with one bottom: the curvature the step bears (dashed), and the
        // sharpness, the slice's own curvature at its middle
        let (vx, vy, kc, kl) = (g[2] as f64, g[3] as f64, (g[4] as f64).exp(), (g[5] as f64).exp());
        // (each as far as it stays in the picture, through as many points in every frame,
        // so that where it leaves the picture glides with it)
        let parabola = |k: f64| -> Vec<Vec2> {
            let reach = (2.0 * (p.y[1] - vy).max(0.0) / k).sqrt();
            let (left, right) = (reach.min(vx + 1.0).max(0.0), reach.min(1.0 - vx).max(0.0));
            let half = 30;
            (0..=2 * half)
                .map(|i| {
                    let u = (i as f64 - half as f64) / half as f64;
                    let d = if u < 0.0 { u * left } else { u * right };
                    p.at(vx + d, vy + 0.5 * k * d * d)
                })
                .collect()
        };
        let valley = parabola(kl);
        for pair in valley.windows(2) {
            plot::dashed_of(f, VALLEY, pair[0], pair[1], 1.5, LIMIT);
        }
        plot::path_of(f, OSCULATING, &parabola(kc), 1.3, CURVATURE);
        // the slice
        let values = &g[SLICE_G..SLICE_G + SLICE_N];
        let pts = values.iter().enumerate().map(|(i, v)| (-1.0 + 2.0 * i as f64 / (SLICE_N - 1) as f64, *v as f64));
        p.path(f, SLICE, pts, 3.0, pal::TEXT.a(0.9));
        // the iterates, newest brightest, joined in the order they came
        let it = &g[SLICE_G + SLICE_N..];
        let color = s.rule.color();
        let n = it.len() / 2;
        let at: Vec<Vec2> = (0..n).map(|i| p.at(it[2 * i] as f64, it[2 * i + 1] as f64)).collect();
        if n >= 2 {
            let pts: Vec<Vec2> = at.iter().rev().copied().collect();
            for w in plot::clip(&pts, p.rect).split(|q| !q.is_finite()).filter(|w| w.len() >= 2) {
                plot::path_of(f, TRAIL_LINE, w, 1.0, color.a(0.55));
            }
        }
        let dots: Vec<(Vec2, f32, Color)> = at
            .iter()
            .enumerate()
            .filter(|(_, q)| plot::within(**q, p.rect, 0.5))
            .map(|(i, q)| (*q, if i == 0 { 7.0 } else { 4.5 }, color.a(if i == 0 { 1.0 } else { 0.85 - 0.08 * i as f32 })))
            .collect();
        dots_px(f, 0, &dots);
        // the numbers: the slice's curvature against the level
        let (sym, lvl) = if s.preconditioned() { (r"\lambda_1(P^{-1}H)", s.level_name()) } else { ("S", s.level_name()) };
        let line = format!(r"curvature here ${sym} = {}$,  ${lvl} = {}$", sig(curvature), sig(level));
        f.caption_px(Vec2::new(min.x + 6.0, min.y + 4.0), line, size, pal::TEXT, Align2::LEFT_TOP);
        f.caption_px(Vec2::new(min.x + 6.0, min.y + 8.0 + size + 2.0), format!("up: how much higher than the lowest loss here, {}", small(low)), size, pal::DIM, Align2::LEFT_TOP);
        // the key, under the slice
        let key = slice_key(lay.narrow, &s.level_name(), color, s.preconditioned());
        let (mut at, width) = lay.key;
        for (c, text) in key {
            at.y += key_px(f, at, width, &text, size, c);
        }
        if !lay.narrow {
            f.text_px(Vec2::new(max.x, max.y + 4.0 + size + 1.0), "distance along the direction", size, pal::DIM, Align2::RIGHT_TOP);
        }
    }
}

/// The part of a segment inside a rectangle, if any.
fn clip_segment(a: Vec2, b: Vec2, rect: Rect) -> Option<(Vec2, Vec2)> {
    let pts = plot::clip(&[a, b], rect);
    match pts.as_slice() {
        [p, q] if p.is_finite() && q.is_finite() => Some((*p, *q)),
        _ => None,
    }
}

/// A label of a chart: where it wants to be (the point its alignment anchors), what it
/// says, its colour, and which chart it is on (labels keep to their own chart).
struct ChartLabel {
    at: Vec2,
    align: Align2,
    text: String,
    color: Color,
    area: usize,
}

/// Where a label of a chart was put: its words, the slot it took (above or below its
/// own place, in its heights) and how far from its place it is drawn now (gliding to
/// that slot when it changes).
#[derive(Clone, Debug, Default)]
struct Placed {
    text: String,
    slot: i32,
    drawn: f32,
}

/// The labels on plates, each at its own place if that is clear of those placed before
/// it, else the nearest clear place above or below it, inside its chart's rectangle (in
/// the order given: the first keep their places). A label keeps the slot it had while
/// that is still clear, and glides to a new one (`memo` remembers them; true while one
/// glides). (As the noise floor scene places its labels.)
fn place(f: &mut Frame, tags: Vec<ChartLabel>, size: f32, rects: &[Rect], memo: &mut Vec<Placed>) -> bool {
    let mut taken: Vec<Rect> = Vec::new();
    let mut now: Vec<Placed> = Vec::with_capacity(tags.len());
    let ease = 1.0 - (-f.real_dt.max(0.0) / 0.08).exp();
    let mut gliding = false;
    for t in tags {
        let Some(&(min, max)) = rects.get(t.area) else { continue };
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

impl Scene for Eos {
    fn view(&self) -> View {
        View::Flat { center: Vec2::ZERO, half_height: 1.0 }
    }

    fn settings(&self, s: &mut ViewSettings) {
        s.grid = false;
        s.bloom = s.bloom.min(0.2);
    }

    fn animated(&self) -> bool {
        self.busy() || self.look.moving || self.labels_moving
    }

    fn pointer_hint(&self) -> Option<&'static str> {
        Some("buttons over the charts: click")
    }

    fn ui(&mut self, ui: &mut egui::Ui, _: &Gfx) {
        ui::note(
            ui,
            r"On a bowl, gradient descent with step $\eta$ blows up along any direction whose curvature is over $2/\eta$ (scenes stepsize and eostoy). A network's loss is not a bowl: its sharpness $S$, the largest eigenvalue of its Hessian, changes as it trains. Cohen and co-authors (2021) trained many networks by full-batch gradient descent and typically found two phases. First the sharpness rises while it is under $2/\eta$ (they call this progressive sharpening). Then, unless training finishes first, it reaches $2/\eta$ and hovers right at, or just above, it: the edge of stability. There the loss goes up and down from step to step, yet keeps falling over long stretches.",
        );
        ui::note(
            ui,
            r"It is an observation, not a theorem, and it is for full-batch training: image classifiers (fully connected and convolutional), a transformer on text and small regressions, with squared error and with cross-entropy. One of the caveats they list: for shallow or wide networks, or on simple data, the sharpness does not rise that much. Here a network with one hidden layer of 32 tanh units is fitted, with squared error, to the Chebyshev polynomial $T_4$ at 20 points, a task like their smallest, and its sharpness is estimated at every step. This small network shows both phases.",
        );

        ui::section(ui, "Things to try");
        for (k, t) in TRIALS.iter().enumerate() {
            if ui::experiment(ui, t.button, t.caption) {
                self.set_up(k);
            }
        }

        ui::section(ui, "The run");
        let s = &mut self.setup;
        let mut rule = match s.rule {
            Rule::Gd => 0,
            Rule::Momentum => 1,
            Rule::Adam => 2,
        };
        if ui::choice(ui, &["gradient descent", "momentum", "Adam"], &mut rule) {
            let new = [Rule::Gd, Rule::Momentum, Rule::Adam][rule.min(2)];
            // (Adam's steps are a hundred times shorter for the same progress)
            if (new == Rule::Adam) != (s.rule == Rule::Adam) {
                s.eta = if new == Rule::Adam { 0.01 } else { 0.2 };
            }
            s.rule = new;
        }
        let (lo, hi) = if s.rule == Rule::Adam { (LEAST_ADAM_ETA, MOST_ADAM_ETA) } else { (LEAST_ETA, MOST_ETA) };
        let mut eta = s.eta as f32;
        if ui::slider_log(ui, r"step size $\eta$", &mut eta, lo as f32..=hi as f32) {
            s.eta = eta as f64;
        }
        ui::note(ui, &format!(r"Its level now: ${} = {}$.", s.level_name(), sig(s.level_at(0))));
        if s.rule == Rule::Momentum {
            let mut beta = s.beta as f32;
            if ui::slider(ui, r"momentum $\beta$", &mut beta, 0.0..=MOST_BETA as f32) {
                s.beta = beta as f64;
            }
            ui::note(ui, r"Heavy ball, $v \leftarrow \beta v + g$, then $\theta \leftarrow \theta - \eta v$. On a bowl it bears curvatures up to $(2 + 2\beta)/\eta$.");
        }
        let mut cut = s.cut.is_some();
        if ui::checkbox(ui, &mut cut, "cut the step by four").changed() {
            s.cut = cut.then_some(s.steps / 2);
        }
        if let Some(c) = &mut s.cut {
            let most = s.steps;
            ui::slider_int(ui, "at step", c, 1..=most);
        }
        ui::slider_int(ui, "batch size", &mut s.batch, 1..=POINTS);
        ui::note(ui, "20 is the whole data set: full-batch training. Fewer: each step's gradient is that of a random batch, drawn without putting back. Cohen and co-authors (2021) report that with mini-batches the sharpness does not always settle at any value, let alone one the step predicts, that smaller batches tend to give lower sharpness, and that with large batches it behaves as with the whole data.");
        ui::slider_int_log(ui, "units in each hidden layer", &mut s.width, LEAST_WIDTH..=MOST_WIDTH);
        ui::slider_int(ui, "hidden layers", &mut s.depth, 1..=MOST_DEPTH);
        let mut degree = DEGREES.iter().position(|&d| d == s.degree).unwrap_or(1);
        if ui::choice(ui, &["$T_3$", "$T_4$", "$T_5$"], &mut degree) {
            s.degree = DEGREES[degree.min(2)];
        }
        ui::note(ui, r"The target: the Chebyshev polynomial of that degree, $T_k(\cos\theta) = \cos k\theta$, with $k$ zeros in $[-1, 1]$.");
        ui::slider_int_log(ui, "steps in the run", &mut s.steps, LEAST_STEPS..=MOST_STEPS);
        ui::slider_log(ui, "film speed", &mut self.speed, 0.25..=4.0);
        ui::note(ui, &format!("At 1 a run plays in {} seconds, if this device can compute it that fast.", FILM_SECONDS));
        ui.horizontal_wrapped(|ui| {
            if ui.button("new starting weights").clicked() {
                self.setup.seed = self.setup.seed.wrapping_add(1);
            }
            if ui.button("keep this run").clicked() {
                self.keep();
            }
            if self.compare.is_some() && ui.button("forget the kept run").clicked() {
                self.compare = None;
            }
        });
        self.refresh();

        ui::section(ui, "Reading the picture");
        ui::key(
            ui,
            &[
                (SGD, "cyan: the loss under gradient descent (pink with momentum, orange with Adam)"),
                (CURVATURE, "green: the sharpness, estimated at every step; on the slice, a parabola of that curvature"),
                (LIMIT, r"yellow dashed: the level the step sets, $2/\eta$ for gradient descent; on the slice, a valley of that curvature"),
                (pal::TEXT, "white: the loss along the sharpest direction"),
                (NOISE, "red dots: the batch sharpness, with mini-batches"),
                (SGD.a(0.42), "faint: the run kept for comparison"),
            ],
        );
        ui::section(ui, "Words");
        ui::term(ui, "Sharpness:", r"the largest eigenvalue of the Hessian of the training loss, $S$: the curvature along the sharpest direction.");
        ui::term(ui, "Level:", r"the largest curvature of a bowl on which the rule does not blow up: $2/\eta$ for gradient descent, $(2 + 2\beta)/\eta$ for heavy-ball momentum written $v \leftarrow \beta v + g$, and for Adam $38/\eta$ on the preconditioned curvature (with $\beta_1 = 0.9$ and its preconditioner held fixed).");
        ui::term(ui, "Progressive sharpening:", "the sharpness rising during training while it is under the level.");
        ui::term(ui, "Edge of stability:", "the phase in which the sharpness stays at about the level and the loss goes up and down from step to step. Theory for it exists only in special settings or under assumptions; Cohen, Damian and co-authors (2025) derive, by informal reasoning, a central flow that follows a full-batch run of gradient descent (or RMSProp; not yet with momentum) averaged over its oscillations, along which the loss never rises.");
        ui::term(ui, "Full batch:", "every step uses the gradient of the whole data set.");
        ui::term(ui, "Preconditioned sharpness:", r"for Adam, which divides each parameter's step by its own number $p_i$: the largest eigenvalue of $P^{-1}H$, $P$ the diagonal matrix of the $p_i$.");
        ui::term(ui, "Batch sharpness:", "for each mini-batch, the curvature of its own loss along its own gradient, averaged over batches.");
        ui::term(ui, "Residual:", r"how far Lanczos's estimate $\mu$ with its unit vector $y$ is from satisfying $Hy = \mu y$, $|Hy - \mu y|$: some eigenvalue lies within it of $\mu$.");

        ui::section(ui, "Numbers");
        self.numbers(ui);
        ui::see_also(
            ui,
            &[
                ("stepsize", r"the level $2/\eta$ on a bowl"),
                ("eostoy", "the same in a loss of two weights"),
                ("hessian", "the Hessian and its eigenvalues"),
                ("warmup", "warm-up: the step and the sharpness at the start"),
            ],
        );
    }

    fn frame(&mut self, f: &mut Frame) {
        hold_batches(f);
        let lay = Layout::of(f.view_size(), f.top_inset, self.look.third.get(0));
        // (a canvas squeezed to a sliver is one the controls have opened over, on a phone)
        let tiny = lay.view.x < 160.0 || lay.view.y < 120.0;
        if !tiny && !lay.compact && self.buttons(f, &lay) {
            self.refresh();
        }
        self.refresh();
        let budget = clock::frame_budget(f.real_dt, 0.006);
        self.advance(f.dt, &budget);
        let snap = self.canvas != Some(lay.view) && !tiny;
        if snap {
            self.canvas = Some(lay.view);
        }
        self.follow(f.real_dt, snap);
        if tiny {
            return;
        }
        let mut labels = Vec::new();
        if !lay.compact {
            self.draw_loss(f, &lay);
            self.draw_third(f, &lay, &mut labels);
            self.draw_slice(f, &lay);
        }
        self.draw_sharp(f, &lay, &mut labels);
        let rects = [lay.loss, lay.sharp, lay.third.unwrap_or(lay.loss)];
        self.labels_moving = place(f, labels, lay.small, &rects, &mut self.placed);
    }
}

impl Eos {
    /// The read-outs: what run 0 has done, as far as the film has shown it.
    fn numbers(&self, ui: &mut egui::Ui) {
        let Some(r) = self.runs.first() else { return };
        let s = &r.setup;
        let shown = (self.film.shown.floor() as usize + 1).min(r.measured());
        // (Adam's preconditioned sharpness from step 1: it needs a preconditioner)
        let from = if s.preconditioned() { 1 } else { 0 };
        let st = r.stats(from, shown);
        let name = if s.preconditioned() { "preconditioned sharpness" } else { "sharpness" };
        ui::readout(ui, "step", format!("{} of {}", count(shown.saturating_sub(1)), count(s.steps)));
        ui::readout(ui, "parameters", count(s.params()));
        ui::readout_colored(ui, &format!("{name} at step {from}"), sig(st.start), CURVATURE);
        ui::readout_colored(ui, &format!("{name} now"), sig(r.main(shown.saturating_sub(1))), CURVATURE);
        ui::readout_colored(ui, &format!("level ${}$", s.level_name()), sig(s.level_at(0)), LIMIT);
        if let Some(c) = s.cut {
            ui::readout_colored(ui, &format!("  from step {}", count(c)), sig(s.level_at(c)), LIMIT);
        }
        if let Some(b) = r.blown.filter(|&b| b < shown) {
            ui::readout(ui, "the loss blew up at step", count(b));
        }
        match st.cross {
            Some(c) => {
                if c == from {
                    ui::readout(ui, "at the level", "over it from the start");
                } else {
                    ui::readout(ui, "first at the level: step", count(c));
                    ui::readout(ui, "  before that, steps at which it fell", count(st.falls_before));
                }
                if st.stretch > 0 {
                    ui::readout(ui, &format!("  steps {} to {}: over the level", count(c + SETTLE), count(shown - 1)), format!("mean {}, from {} to {}", sig(st.mean), sig(st.lo), sig(st.hi)));
                }
                ui::readout(ui, "  steps since at which the loss rose", format!("{} of {} ({})", count(st.rises), count(st.of), percent(st.rises, st.of)));
                ui::readout(ui, "  loss at the crossing", small(st.loss_cross));
            }
            None => {
                ui::readout(ui, "at the level", "not yet");
                ui::readout(ui, "  largest so far", format!("{} at step {}", sig(st.peak), count(st.peak_at)));
                ui::readout(ui, "  steps at which the loss rose", format!("{} of {}", count(st.rises), count(st.of)));
            }
        }
        if let Some(c) = s.cut.filter(|&c| c < shown) {
            let after = r.stats(c, shown);
            match after.cross {
                Some(x) => ui::readout(ui, "after the cut, back at the level: step", count(x)),
                None => ui::readout(ui, "after the cut, at the level", format!("not yet ({} now)", sig(r.main(shown - 1)))),
            }
        }
        ui::readout(ui, "loss now", small(st.loss_end));
        if s.preconditioned() {
            ui::readout_colored(ui, "sharpness without the preconditioner now", sig(r.sharp.get(shown.saturating_sub(1)).copied().unwrap_or(f64::NAN)), CURVATURE);
        }
        if let Some(&(t, m, se)) = r.batch.iter().rev().find(|b| b.0 < shown) {
            ui::readout_colored(ui, &format!("batch sharpness at step {}", count(t)), format!(r"${} \pm {}$", sig(m), sig(2.0 * se)), NOISE);
        }
        ui::readout(ui, "sharpness estimated", "at every step, from the last");
        ui::readout(ui, "  Hessian-vector products, mean", num(r.mean_products(), 1));
        ui::readout(ui, "  residual, largest so far", format!(r"${}$ of $|H|$", small(r.worst_residual()).trim_matches('$')));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// An experiment as its button sets it up, with its runs computed to the end.
    fn run(k: usize) -> Eos {
        let mut s = Eos::create();
        s.set_up(k);
        for r in &mut s.runs {
            while !r.done() {
                r.step();
            }
        }
        s.film.shown = s.steps() as f64;
        s
    }

    /// The stretches in which the loss was over `over`: (first step, step after the
    /// last, highest loss, the largest sharpness in the 150 steps before, the least in
    /// the 100 from its start).
    fn leaps(r: &Run, over: f64) -> Vec<(usize, usize, f64, f64, f64)> {
        let n = r.measured();
        let mut out = Vec::new();
        let mut t = 1;
        while t < n {
            if r.loss[t] > over {
                let a = t;
                while t < n && r.loss[t] > over {
                    t += 1;
                }
                let before = (a.saturating_sub(150)..a).map(|t| r.main(t)).fold(f64::MIN, f64::max);
                let after = (a..(a + 100).min(n)).map(|t| r.main(t)).fold(f64::MAX, f64::min);
                let top = (a..t).map(|t| r.loss[t]).fold(0.0, f64::max);
                out.push((a, t, top, before, after));
            }
            t += 1;
        }
        out
    }

    /// The mean loss over each `k` steps, from the start.
    fn block_means(r: &Run, k: usize) -> Vec<f64> {
        (0..r.measured() / k).map(|b| r.mean_loss(k * b, k * (b + 1))).collect()
    }

    fn finished(setup: Setup) -> Run {
        let mut r = Run::new(setup);
        while !r.done() {
            r.step();
        }
        r
    }

    #[test]
    fn chebyshev_is_cos_k_arccos() {
        for k in 0..7 {
            for i in 0..=20 {
                let x = -1.0 + i as f64 / 10.0;
                assert!((chebyshev(k, x) - (k as f64 * x.acos()).cos()).abs() < 1e-12, "{k} {x}");
            }
        }
        // the inputs: 20 points from −1 to 1, evenly spaced
        let x = inputs();
        assert_eq!(x.len(), 20);
        assert!((x[0] + 1.0).abs() < 1e-15 && (x[19] - 1.0).abs() < 1e-15 && ((x[1] - x[0]) - 2.0 / 19.0).abs() < 1e-15);
    }

    /// The levels are the exact limits of the rules on a quadratic f = ½ h θ²: the
    /// spectral radius of one step's linear map is below 1 just under the level and
    /// above 1 just over it. Heavy ball (v ← βv + g, θ ← θ − ηv) has the map [[1 − ηh,
    /// −ηβ], [h, β]]; Adam with its preconditioner p held fixed, m ← β₁m + (1 − β₁)g,
    /// θ ← θ − ηm/p, the map [[1 − η(1 − β₁)h/p, −ηβ₁/p], [(1 − β₁)h, β₁]], whose limit is
    /// on h/p.
    #[test]
    fn the_levels_are_the_quadratic_limits() {
        let radius = |a: [[f64; 2]; 2]| {
            let (tr, det) = (a[0][0] + a[1][1], a[0][0] * a[1][1] - a[0][1] * a[1][0]);
            let disc = tr * tr - 4.0 * det;
            if disc >= 0.0 {
                ((tr.abs() + disc.sqrt()) * 0.5).max((tr.abs() - disc.sqrt()).abs() * 0.5)
            } else {
                det.sqrt()
            }
        };
        for eta in [0.01, 0.2, 0.3] {
            for (rule, beta) in [(Rule::Gd, 0.0), (Rule::Momentum, 0.5), (Rule::Momentum, 0.9), (Rule::Adam, BETA1)] {
                let s = Setup { rule, eta, beta, ..Setup::base() };
                let level = s.level_at(0);
                let map = |h: f64| match rule {
                    Rule::Gd => [[1.0 - eta * h, 0.0], [0.0, 0.0]],
                    Rule::Momentum => [[1.0 - eta * h, -eta * beta], [h, beta]],
                    Rule::Adam => [[1.0 - eta * (1.0 - BETA1) * h, -eta * BETA1], [(1.0 - BETA1) * h, BETA1]],
                };
                assert!(radius(map(level * 0.999)) < 1.0, "{rule:?} {eta} {beta}");
                assert!(radius(map(level * 1.001)) > 1.0, "{rule:?} {eta} {beta}");
            }
        }
        // the numbers the scene writes
        assert!((Setup { rule: Rule::Momentum, eta: 0.1, beta: 0.9, ..Setup::base() }.level_at(0) - 38.0).abs() < 1e-9);
        assert!((Setup { rule: Rule::Adam, eta: 0.01, ..Setup::base() }.level_at(0) - 3800.0).abs() < 1e-6);
        let cut = Setup { eta: 0.3, cut: Some(1000), ..Setup::base() };
        assert!((cut.level_at(999) - 2.0 / 0.3).abs() < 1e-12 && (cut.level_at(1000) - 2.0 / 0.075).abs() < 1e-9);
        assert_eq!(Setup { rule: Rule::Momentum, beta: 0.9, ..Setup::base() }.level_name(), r"3.8/\eta");
    }

    /// The rules are the library's, in the conventions the levels are for: heavy ball
    /// summed (v ← βv + g) with no dampening, Adam with β₁ = 0.9; and two steps of each
    /// are what those formulas give.
    #[test]
    fn the_rules_are_the_conventions_of_the_levels() {
        let s = Setup { rule: Rule::Momentum, eta: 0.1, beta: 0.9, ..Setup::base() };
        let model = Model::mlp(&s.spec(), &mut Rng::new(3));
        let opt = s.optimizer(&model);
        let h = opt.hyper();
        assert_eq!((h.convention, h.dampening, h.momentum, h.nesterov), (optim::Convention::Sum, 0.0, 0.9, false));
        // two steps by hand: θ₂ = θ₁ − η(β g₁ + g₂) with g the full gradient
        let problem = task(4);
        let (mut model, mut tape) = (model, Tape::new());
        let mut opt = opt;
        let mut w0 = Vec::new();
        terrain::get(&model, &mut w0);
        let mut g = Vec::new();
        problem.full_grad(&mut model, &mut tape);
        terrain::grad_of(&model, &mut g);
        opt.step(&mut model, 0.1);
        let g1 = g.clone();
        problem.full_grad(&mut model, &mut tape);
        terrain::grad_of(&model, &mut g);
        let mut w1 = Vec::new();
        terrain::get(&model, &mut w1);
        opt.step(&mut model, 0.1);
        let mut w2 = Vec::new();
        terrain::get(&model, &mut w2);
        for i in 0..w0.len() {
            assert!((w1[i] - (w0[i] - 0.1 * g1[i])).abs() < 1e-14);
            assert!((w2[i] - (w1[i] - 0.1 * (0.9 * g1[i] + g[i]))).abs() < 1e-13);
        }
        let a = Setup { rule: Rule::Adam, ..Setup::base() };
        let h = *a.optimizer(&model).hyper();
        assert!(h.beta1 == 0.9 && h.beta2 == 0.999 && h.eps == 1e-8 && h.bias_correction && !h.amsgrad);
        // the parameter count
        for (width, depth) in [(32, 1), (4, 1), (16, 3), (64, 2)] {
            let s = Setup { width, depth, ..Setup::base() };
            assert_eq!(s.params(), Model::mlp(&s.spec(), &mut Rng::new(1)).n_params());
        }
    }

    /// The sharpness the runs record is the dense Hessian's largest eigenvalue, to the
    /// estimate's tolerance, at the start and at the edge of stability; and Adam's
    /// preconditioned sharpness is that of P^(−1/2) H P^(−1/2) with the preconditioner of
    /// the last step.
    #[test]
    fn the_sharpness_is_the_dense_hessians() {
        let mut r = Run::new(Setup { width: 8, steps: 400, ..Setup::base() });
        for stop in [0usize, 150, 399] {
            while r.t < stop {
                r.step();
            }
            r.step();
            let t = r.measured() - 1;
            // (the model has taken step t since: go back to w_t)
            let now = r.recent.front().cloned().unwrap();
            let mut model = r.model.clone();
            terrain::set(&mut model, &now);
            let dense = terrain::Hessian::dense(&r.problem, &mut model, &mut Tape::new());
            let top = dense.values()[0];
            assert!((r.sharp[t] - top).abs() <= 2e-4 * top.abs(), "step {t}: {} against {top}", r.sharp[t]);
            assert!(r.resid[t] <= 1e-4 + 1e-12, "step {t}: residual {}", r.resid[t]);
        }
        let mut a = Run::new(Setup { rule: Rule::Adam, eta: 0.01, width: 8, steps: 100, ..Setup::base() });
        while a.measured() < 60 {
            a.step();
        }
        let t = a.measured() - 1;
        let now = a.recent.front().cloned().unwrap();
        let mut model = a.model.clone();
        terrain::set(&mut model, &now);
        let h = terrain::Hessian::dense(&a.problem, &mut model, &mut Tape::new());
        let n = h.size();
        let d: Vec<f64> = a.pbuf.iter().map(|p| 1.0 / p.sqrt()).collect();
        let m: Vec<f64> = (0..n * n).map(|k| d[k / n] * h.matrix()[k] * d[k % n]).collect();
        let (vals, _) = optim::sym_eigen(&m, n);
        let top = vals.iter().copied().fold(f64::MIN, f64::max);
        assert!((a.psharp[t] - top).abs() <= 2e-4 * top.abs(), "{} against {top}", a.psharp[t]);
    }

    /// No point of this loss has a sharpness under 2: along the read-out's bias the
    /// loss's second derivative is exactly 2 (the mean of (output − target)², and the
    /// output moves one for one with the bias), so the largest eigenvalue is at least 2.
    #[test]
    fn the_sharpness_is_never_under_two() {
        let s = Setup::base();
        let problem = task(4);
        for seed in [1, 2, 7] {
            let mut model = Model::mlp(&s.spec(), &mut Rng::new(seed));
            let bias = terrain::span(&model, "read-out bias").unwrap();
            let mut e = vec![0.0; model.n_params()];
            e[bias.start] = 1.0;
            let c = terrain::curvature_along(&problem, &mut model, &mut Tape::new(), &problem.all(), &e);
            assert!((c - 2.0).abs() < 1e-6, "{c}");
        }
        let r = finished(Setup { batch: 5, steps: 600, ..s });
        assert!(r.sharp.iter().all(|&v| v >= 2.0 - 1e-6), "{:?}", r.sharp.iter().copied().fold(f64::MAX, f64::min));
    }

    /// The slice is what it says: its curvature at the centre (a second difference of
    /// the loss along it) is the recorded sharpness, or for Adam the preconditioned one;
    /// its iterates lie on it; at the edge of stability the present iterate and the one
    /// before are on either side of its centre, at the same distance.
    #[test]
    fn the_slice_curves_as_the_sharpness() {
        for setup in [Setup { steps: 1500, ..Setup::base() }, Setup { rule: Rule::Adam, eta: 0.01, steps: 600, ..Setup::base() }] {
            let r = finished(setup);
            assert!(r.slices.len() >= 50);
            // (where the iterates have flown far up the walls the slice reaches far, and
            // its samples are too coarse for a second difference: those are left out)
            let mut checked = 0;
            for sl in r.slices.iter().filter(|s| s.step >= 10) {
                let h = 2.0 * sl.range / (SLICE_N - 1) as f64;
                let m = SLICE_N / 2;
                let rise = 0.5 * (sl.values[0] + sl.values[SLICE_N - 1]) - sl.values[m];
                if rise < 0.2 * sl.values[m] {
                    let second = (sl.values[m + 1] - 2.0 * sl.values[m] + sl.values[m - 1]) / (h * h);
                    assert!((second / sl.curvature - 1.0).abs() < 0.02, "{setup:?} step {}: {second} against {}", sl.step, sl.curvature);
                    checked += 1;
                }
                assert_eq!(sl.iterates.len(), TRAIL + 1);
                assert!(sl.values.iter().all(|v| v.is_finite()) && sl.range > 0.0);
                // the newest two straddle the centre (it is their midpoint)
                let (a, b) = (sl.iterates[0].0, sl.iterates[1].0);
                assert!((a + b).abs() <= 1e-9 * (a.abs() + b.abs()).max(1e-300) + 1e-15, "{} {a} {b}", sl.step);
            }
            assert!(checked * 10 >= 8 * r.slices.len(), "{checked} of {}", r.slices.len());
        }
    }

    /// The scene opens on the opening experiment, and its first 700 steps do what the
    /// caption says of them: the sharpness rises from 6.05 to the level 10 at step 607,
    /// the loss falling at every step until then.
    #[test]
    fn the_opening_run_reaches_its_level() {
        let opened = Eos::create();
        let mut pressed = Eos::create();
        pressed.set_up(4);
        pressed.set_up(OPENING);
        assert_eq!((opened.setup, opened.compare), (pressed.setup, pressed.compare));
        assert_eq!(opened.setup, Setup::base());
        let mut r = Run::new(Setup { steps: 700, ..Setup::base() });
        while !r.done() {
            r.step();
        }
        let st = r.stats(0, 701);
        assert_eq!(st.cross, Some(607));
        assert_eq!(sig(st.start), "6.05");
        assert!((1..=607).all(|t| r.loss[t] <= r.loss[t - 1]));
        assert!(TRIALS[OPENING].caption.contains("rises from 6.05 to 10 at step 607, the loss falling at every step"));
    }

    /// The numbers each experiment's caption says, from the scene as the button sets it
    /// up (slow: every run computed to its end).
    #[test]
    #[ignore]
    fn the_experiments_show_what_their_captions_say() {
        let says = |k: usize, what: &str| assert!(TRIALS[k].caption.contains(what), "experiment {k} should say \"{what}\": {}", TRIALS[k].caption);
        let ratio = |r: &Run, a: usize, b: usize| {
            let v: Vec<f64> = (a..b.min(r.measured())).map(|t| r.main(t) / r.setup.level_at(t)).collect();
            let (lo, hi) = v.iter().fold((f64::MAX, f64::MIN), |(l, h), x| (l.min(*x), h.max(*x)));
            (v.iter().sum::<f64>() / v.len() as f64, lo, hi)
        };
        let rises = |r: &Run, a: usize, b: usize| (a + 1..b.min(r.measured())).filter(|&t| r.loss[t] > r.loss[t - 1]).count();
        // a short step: never near the level, the loss falls at every step
        let e = run(0);
        let r = &e.runs[0];
        let n = r.measured();
        let st = r.stats(0, n);
        assert!(st.cross.is_none() && st.peak < 0.5 * r.setup.level_at(0));
        says(0, &format!("rises from {} to {}, its largest at the last of the 4000 steps", sig(st.start), sig(st.peak)));
        assert_eq!(st.peak_at, 4000);
        assert_eq!(st.rises, 0);
        says(0, &format!("The loss falls at every step, to {}", sig(st.loss_end)));
        let last = r.slices.last().unwrap();
        assert!(last.curvature < 0.5 * last.level);
        let d: Vec<f64> = last.iterates.windows(2).map(|w| w[0].0 - w[1].0).collect();
        assert!(d.iter().all(|v| v.signum() == d[0].signum()), "{:?}", last.iterates);
        // a longer step
        let e = run(1);
        let r = &e.runs[0];
        let n = r.measured();
        let st = r.stats(0, n);
        let c = st.cross.unwrap();
        says(1, &format!("from {} to 10 at step {c}", sig(st.start)));
        assert!((1..=c).all(|t| r.loss[t] <= r.loss[t - 1]));
        // the flips grow by |1 − ηS| a step: from step 730 to 760, as the slices measure them
        let amp = |t: usize| r.slices.iter().find(|s| s.step == t).map(|s| s.iterates[0].0.abs()).unwrap();
        let factor = (amp(760) / amp(730)).powf(1.0 / 30.0);
        assert!((factor - 1.5).abs() < 0.05, "{factor}");
        let (sa, sb) = (730..760).map(|t| r.sharp[t]).fold((f64::MAX, f64::MIN), |(l, h), x| (l.min(x), h.max(x)));
        assert!(sa > 12.2 && sb < 12.8, "{sa} {sb}");
        assert!((1.0 - 0.2 * 12.5f64).abs() == 1.5);
        let lp = leaps(r, 1.0);
        assert_eq!(lp.len(), 4, "{lp:?}");
        says(1, &format!("At step {} the loss leaps to {} and the sharpness drops to {}", lp[0].0, sig(lp[0].2), sig(lp[0].4)));
        says(1, &format!("the last at step {}", lp[3].0));
        assert!(lp.iter().all(|l| l.2 > 2.0));
        let (m, lo, hi) = ratio(r, 1800, n);
        says(1, &format!("from step 1800 on, {} times it on average (from {} to {})", num(m, 2), num(lo, 2), num(hi, 2)));
        says(1, &format!("The loss rises at {} of those steps", percent(rises(r, 1800, n), n - 1801)));
        let blocks = block_means(r, 500);
        assert!(blocks.windows(2).all(|w| w[1] < w[0]), "{blocks:?}");
        says(1, &format!("down to {}", sig(*blocks.last().unwrap())));
        // at the end, in the slice, the sharpness is at the level and the iterates flip
        let last = r.slices.last().unwrap();
        assert!((last.curvature / last.level - 1.0).abs() < 0.05);
        assert!(last.iterates.windows(2).all(|w| w[0].0.signum() == -w[1].0.signum()), "{:?}", last.iterates);
        // side by side
        let e = run(2);
        let (a, b) = (&e.runs[0], &e.runs[1]);
        let (sa, sb) = (a.stats(0, a.measured()), b.stats(0, b.measured()));
        let ca = sa.cross.unwrap();
        let worst = (1..=ca).map(|t| (b.sharp[2 * t] / a.sharp[t] - 1.0).abs()).fold(0.0, f64::max);
        assert!(worst < 0.02, "{worst}");
        says(2, &format!("meets its level, 10, at step {ca}"));
        says(2, &format!("to its own, 20, at step {}, and its first leap of the loss comes at step {}", sb.cross.unwrap(), leaps(b, 1.0)[0].0));
        let (ma, mb) = (ratio(a, 3001, 4001).0, ratio(b, 3001, 4001).0);
        assert_eq!(num(ma, 2), "1.02");
        assert_eq!(num(mb, 2), "1.02");
        says(2, "each averages 1.02 times its own level");
        // a cut
        let e = run(3);
        let (r, plain) = (&e.runs[0], &e.runs[1]);
        let n = r.measured();
        let before = r.stats(0, 1000);
        assert!(before.cross.is_some_and(|c| c < 300) && (before.mean - 1.0).abs() < 0.05, "{before:?}");
        assert_eq!(leaps(r, 1.0).iter().filter(|l| l.0 < 1000).count(), 2);
        says(3, &format!("near the level {}", sig(r.setup.level_at(0))));
        says(3, &format!("After the cut the level is {}", sig(r.setup.level_at(1000))));
        let back = r.stats(1000, n).cross.unwrap();
        says(3, &format!("{} at the cut, {} 1500 steps later, back at the new level at step {back}", sig(r.sharp[1000]), sig(r.sharp[2500])));
        assert!((1000..back).all(|t| r.sharp[t] < r.setup.level_at(t)));
        // (it rises at once, and nearly all the way: it falls at few of the steps)
        assert!(r.sharp[1050] > r.sharp[1000] && r.sharp[1100] > r.sharp[1050]);
        let falls = (1001..back).filter(|&t| r.sharp[t] < r.sharp[t - 1]).count();
        assert!(falls * 20 < back - 1000, "{falls}");
        assert_eq!(rises(r, 1000, back + 1), 0);
        assert!(rises(r, back, n) > 20);
        let (_, lo, hi) = ratio(plain, 2000, n);
        says(3, &format!("between {} and {} times 6.67 from step 2000 on", num(lo, 2), num(hi, 2)));
        // momentum
        let e = run(4);
        let r = &e.runs[0];
        let n = r.measured();
        let st = r.stats(0, n);
        says(4, &format!("reaches 38 at step {} and goes on to {}", st.cross.unwrap(), sig(st.peak)));
        let lp = leaps(r, 1.0);
        says(4, &format!("the loss leaps to {} and the sharpness drops to {}", sig(lp[0].2), sig(lp[0].4)));
        let quiet = (1..n).rev().find(|&t| r.loss[t] > r.loss[t - 1]).unwrap() + 1;
        says(4, &format!("From step {quiet} the loss falls at every step"));
        says(4, &format!("to {} at the end ({} of its level), with the loss down to {}", sig(r.sharp[n - 1]), num(r.sharp[n - 1] / 38.0, 2), small(r.loss[n - 1])));
        assert!((quiet..n).all(|t| r.sharp[t] < 38.0));
        // (it climbs back: from step 1000 on it rises at every step but a few)
        let falls = (1001..n).filter(|&t| r.sharp[t] < r.sharp[t - 1]).count();
        assert!(falls * 20 < n - 1000 && r.sharp[n - 1] > r.sharp[1000], "{falls}");
        // Adam
        let e = run(5);
        let r = &e.runs[0];
        let n = r.measured();
        says(5, &format!("({} at step 1, after one gradient)", sig(r.psharp[1])));
        assert!((2..n).all(|t| r.psharp[t] < r.psharp[1]));
        let (m, lo, hi) = ratio(r, 1000, n);
        says(5, &format!("from step 1000 on it averages {} of 3800, between {} and {} times it, and the loss rises at {} of the steps", num(m, 2), num(lo, 2), num(hi, 2), percent(rises(r, 1000, n), n - 1001)));
        says(5, &format!("it climbs from {} to {} by step 2000, and is {} at the end", sig(r.sharp[0]), sig(r.sharp[2000]), sig(r.sharp[n - 1])));
        // with mini-batches
        let e = run(6);
        let r = &e.runs[0];
        let n = r.measured();
        let (dip_at, dip) = (0..1000).map(|t| (t, r.sharp[t])).fold((0, f64::MAX), |a, b| if b.1 < a.1 { b } else { a });
        says(6, &format!("first falls, to {} at step {dip_at}, near the least it can be (2)", sig(dip)));
        let over: Vec<usize> = (0..n).filter(|&t| r.sharp[t] >= 10.0).collect();
        says(6, &format!("at only {} steps, between {} and {}", over.len(), over[0], over.last().unwrap()));
        says(6, &format!("over the last 1000 steps averages {} of it", num(ratio(r, 3001, n).0, 2)));
        let bs: Vec<f64> = r.batch.iter().filter(|b| b.0 > n - 1001).map(|b| b.1 / 10.0).collect();
        assert_eq!(bs.len(), 50);
        says(6, &format!(r"over the last 1000 steps {} of $2/\eta$", num(bs.iter().sum::<f64>() / bs.len() as f64, 2)));
        // (each estimate is from 16 batches and scatters: the average is what is claimed)
        assert!(bs.iter().sum::<f64>() / bs.len() as f64 <= 0.9);
        for seed in 1..=5 {
            let r = finished(Setup { batch: 5, seed, ..Setup::base() });
            let n = r.measured();
            for g in r.model.groups().iter().filter(|g| g.name != "read-out bias") {
                let norm = g.value.iter().map(|v| v * v).sum::<f64>().sqrt();
                assert!(norm < 0.05, "seed {seed}: {} {norm}", g.name);
            }
            assert!((r.sharp[n - 1] - 2.0).abs() < 0.01, "seed {seed}: {}", r.sharp[n - 1]);
        }
        says(6, "all five seeds tried ended with every weight near zero");
    }

    /// Edges: a step far over every level, the smallest and largest settings, numbers
    /// that are not numbers: nothing panics, every run ends (or stops where it blew up),
    /// and everything drawn stays finite.
    #[test]
    fn degenerate_settings_are_told_honestly() {
        let base = Setup::base();
        let cases = [
            Setup { eta: MOST_ETA, ..base },
            Setup { rule: Rule::Adam, eta: MOST_ADAM_ETA, ..base },
            Setup { rule: Rule::Momentum, eta: MOST_ETA, beta: MOST_BETA, ..base },
            Setup { eta: LEAST_ETA, width: LEAST_WIDTH, depth: MOST_DEPTH, batch: 1, ..base },
            Setup { width: MOST_WIDTH, depth: 2, degree: 5, cut: Some(1), ..base },
            Setup { rule: Rule::Adam, eta: LEAST_ADAM_ETA, batch: 2, cut: Some(10_000), ..base },
            Setup { eta: f64::NAN, beta: f64::INFINITY, batch: 0, width: 0, depth: 0, degree: 9, seed: u64::MAX, steps: 0, cut: Some(0), ..base },
        ];
        for c in cases {
            let mut scene = Eos::create();
            scene.setup = Setup { steps: LEAST_STEPS, ..c };
            scene.compare = Some(Setup { eta: 0.3, steps: 300, ..base });
            scene.refresh();
            let k = scene.setup;
            assert!(k.eta.is_finite() && k.beta.is_finite() && k.batch >= 1 && k.width >= LEAST_WIDTH && k.depth >= 1 && k.steps >= LEAST_STEPS && DEGREES.contains(&k.degree), "{k:?}");
            for r in &mut scene.runs {
                while !r.done() {
                    r.step();
                }
                assert!(r.blown.is_some() || r.measured() == r.setup.steps + 1, "{:?}", r.setup);
            }
            for shown in [0.0, 0.5, 37.3, scene.steps() as f64] {
                scene.film.shown = shown;
                scene.follow(0.016, true);
                for g in [&scene.look.frame, &scene.look.slice] {
                    assert!(g.shown.iter().all(|v| v.is_finite()), "{k:?}: {:?}", g.shown);
                }
                let fr = scene.frame_goal();
                assert!(fr[X_HI] > 0.0 && fr[LOSS_HI] > fr[LOSS_LO] && fr[SHARP_HI].is_finite() && fr[THIRD_HI].is_finite(), "{k:?}: {fr:?}");
                let st = scene.runs[0].stats(0, shown as usize + 1);
                assert!(st.loss_from.is_finite() || scene.runs[0].blown == Some(0), "{k:?}");
            }
        }
        // a step far over the level blows up, and the run says where
        let r = finished(Setup { eta: MOST_ETA, ..base });
        assert!(r.blown.is_some_and(|b| b < 200), "{:?}", r.blown);
    }

    /// The film: it waits for the glide, plays at its pace, comes to the end, stops,
    /// and the picture comes to rest; and it never shows a step not yet computed.
    #[test]
    fn the_film_plays_to_the_end_and_rests() {
        let mut s = Eos::create();
        s.setup = Setup { steps: LEAST_STEPS, ..Setup::base() };
        s.compare = Some(Setup { eta: 0.1, steps: LEAST_STEPS, ..Setup::base() });
        s.refresh();
        s.follow(0.016, true);
        let (mut frames, mut last) = (0, 0.0);
        while s.animated() {
            s.advance(1.0 / 60.0, &Budget::new(1.0));
            s.follow(1.0 / 60.0, false);
            assert!(s.film.shown >= last, "the film went back");
            assert!(s.film.shown <= s.ready() + 1e-9);
            last = s.film.shown;
            frames += 1;
            assert!(frames < 60 * 60, "still playing");
        }
        assert_eq!(s.film.shown, LEAST_STEPS as f64);
        assert!(s.runs.iter().all(|r| r.done()));
        // about as long as it says (and a little for the glides to come to rest)
        let secs = frames as f64 / 60.0 - FILM_SECONDS;
        assert!(secs > 0.0 && secs < 1.6, "{frames}");
        // at rest, a frame changes nothing
        let g = s.look.slice.shown.clone();
        s.advance(1.0 / 60.0, &Budget::new(1.0));
        s.follow(1.0 / 60.0, false);
        assert!(!s.animated() && s.look.slice.shown == g);
        // paused, nothing is computed beyond what is shown and the two slices after it
        let mut p = Eos::create();
        p.follow(0.016, true);
        p.film = Film { shown: 0.0, playing: false, waiting: false, waited: 0.0 };
        for _ in 0..30 {
            p.advance(1.0 / 60.0, &Budget::new(1.0));
        }
        assert!(p.runs.iter().all(|r| r.measured() <= 2 * SLICE_EVERY + 2), "{:?}", p.runs.iter().map(|r| r.measured()).collect::<Vec<_>>());
    }

    /// The layout keeps everything in the window and apart, from a phone to a desktop.
    #[test]
    fn the_layout_fits_every_window() {
        for third in [false, true] {
            for (w, h) in [(1220.0, 1000.0), (1100.0, 700.0), (1600.0, 1000.0), (1600.0, 700.0), (1060.0, 1000.0), (390.0, 800.0), (360.0, 640.0), (700.0, 500.0), (300.0, 600.0), (500.0, 400.0), (240.0, 180.0)] {
                let lay = Layout::of(Vec2::new(w, h), 40.0, if third { 1.0 } else { 0.0 });
                let inside = |(a, b): Rect| a.x >= 0.0 && a.y >= 0.0 && b.x <= w + 0.5 && b.y <= h + 0.5 && a.is_finite() && b.is_finite();
                let apart = |(a, b): Rect, (c, d): Rect| b.x <= c.x || d.x <= a.x || b.y <= c.y || d.y <= a.y;
                assert!(inside(lay.sharp), "{w}x{h}: {:?}", lay.sharp);
                if lay.compact {
                    continue;
                }
                // (each chart with the room round it for its title and numbers)
                let room = |(a, b): Rect| (a - Vec2::new(LEFT - 8.0, ABOVE - 4.0), b + Vec2::new(0.0, BELOW - 4.0));
                let mut boxes: Vec<Rect> = vec![room(lay.loss), room(lay.sharp), room(lay.slice)];
                boxes.extend(lay.third.map(room));
                boxes.extend(lay.buttons);
                for (i, a) in boxes.iter().enumerate() {
                    assert!(inside(*a), "{w}x{h} {third}: box {i} {a:?}");
                    for b in &boxes[i + 1..] {
                        assert!(apart(*a, *b), "{w}x{h} {third}: {a:?} and {b:?} overlap");
                    }
                }
                assert_eq!(lay.third.is_some(), third);
                for r in [lay.loss, lay.sharp, lay.slice] {
                    assert!(r.1.y - r.0.y > 60.0 && r.1.x - r.0.x > 150.0, "{w}x{h}: {r:?}");
                }
                // the slice's key between the slice and what is under it
                assert!(lay.key.0.y > lay.slice.1.y && lay.key.1 > 100.0, "{w}x{h}: {:?}", lay.key);
            }
        }
    }

    /// The picture's glides come to rest after a change, and the film then plays (and,
    /// if the glide is a long one, starts after `FILM_WAIT` while it goes on).
    #[test]
    fn glides_arrive_and_stop() {
        let mut s = Eos::create();
        s.advance(1.0 / 60.0, &Budget::new(1.0));
        s.follow(0.016, true);
        for k in [3, 0, 5, 1] {
            // (part of the way through the run first, so that the new one glides from it)
            s.film = Film { shown: 300.0f64.min(s.ready()), playing: false, waiting: false, waited: 0.0 };
            s.follow(0.016, true);
            s.set_up(k);
            let mut frames = 0;
            // (the first frame computes the new runs' first steps; the film then waits)
            s.advance(1.0 / 60.0, &Budget::new(1.0));
            s.follow(1.0 / 60.0, false);
            assert!(!s.film.playing && s.film.waiting, "experiment {k}");
            while s.look.moving {
                s.follow(1.0 / 60.0, false);
                frames += 1;
                assert!(frames < 300, "experiment {k} still moving");
            }
            assert!(frames > 3, "experiment {k}: {frames}");
            s.advance(1.0 / 60.0, &Budget::new(1.0));
            assert!(s.film.playing, "experiment {k}");
        }
    }

    /// The numbers behind the captions (run with --ignored --nocapture).
    #[test]
    #[ignore]
    fn explore() {
        let ratio = |r: &Run, a: usize, b: usize| {
            let v: Vec<f64> = (a..b.min(r.measured())).map(|t| r.main(t) / r.setup.level_at(t)).filter(|v| v.is_finite()).collect();
            let mean = v.iter().sum::<f64>() / v.len().max(1) as f64;
            let (lo, hi) = v.iter().fold((f64::MAX, f64::MIN), |(l, h), x| (l.min(*x), h.max(*x)));
            (mean, lo, hi)
        };
        let rises = |r: &Run, a: usize, b: usize| (a + 1..b.min(r.measured())).filter(|&t| r.loss[t] > r.loss[t - 1]).count();
        for k in 0..TRIALS.len() {
            let e = run(k);
            for (i, r) in e.runs.iter().enumerate() {
                let n = r.measured();
                let st = r.stats(0, n);
                println!("== {k} run {i}: {}", r.setup.label());
                println!("   S0 {} end S {} P {} loss end {} | cross {:?} falls before {} peak {} at {}", sig(r.sharp[0]), sig(r.sharp[n - 1]), sig(r.psharp[n - 1]), sig(r.loss[n - 1]), st.cross, st.falls_before, sig(st.peak), st.peak_at);
                println!("   leaps over 1: {:?}", leaps(r, 1.0).iter().map(|l| format!("{}..{} top {} S {}->{}", l.0, l.1, sig(l.2), sig(l.3), sig(l.4))).collect::<Vec<_>>());
                for a in [1000, 1800, 2000, 3000] {
                    if a < n {
                        let (m, lo, hi) = ratio(r, a, n);
                        println!("   from {a}: ratio {} [{}, {}], rises {} of {} = {}", num(m, 3), num(lo, 3), num(hi, 3), rises(r, a, n), n - a - 1, percent(rises(r, a, n), n - a - 1));
                    }
                }
                let quiet = (1..n).rev().find(|&t| r.loss[t] > r.loss[t - 1]).map_or(0, |t| t + 1);
                println!("   loss falls at every step from {quiet}; block means {:?}", block_means(r, 500).iter().map(|v| sig(*v)).collect::<Vec<_>>());
                if let Some(c) = r.setup.cut {
                    let after = r.stats(c, n);
                    let back = after.cross;
                    let (m, lo, hi) = ratio(r, st.cross.unwrap_or(0) + SETTLE, c);
                    println!("   before the cut ratio {} [{}, {}]; S at cut {} at +1500 {}; back {:?}; rises cut..back {}", num(m, 3), num(lo, 3), num(hi, 3), sig(r.sharp[c]), sig(r.sharp[c + 1500]), back, back.map_or(0, |b| rises(r, c, b + 1)));
                }
                if !r.batch.is_empty() {
                    let bs: Vec<f64> = r.batch.iter().filter(|b| b.0 >= n - 1000).map(|b| b.1 / r.setup.level_at(b.0)).collect();
                    let over: Vec<usize> = (0..n).filter(|&t| r.sharp[t] >= r.setup.level_at(t)).collect();
                    println!("   batch sharpness last 1000 {} ; over level at {:?}", num(bs.iter().sum::<f64>() / bs.len() as f64, 3), over);
                }
                let amp: Vec<String> = r.slices.iter().filter(|s| s.step >= 600 && s.step <= 780).map(|s| format!("{}:{:.1e}", s.step, s.iterates[0].0.abs())).collect();
                println!("   amplitude {amp:?}");
            }
        }
    }
}

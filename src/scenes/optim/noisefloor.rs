//! SGD on a bowl: the noise floor. A mini-batch's gradient is the true gradient plus
//! noise, so on a quadratic bowl a constant step does not take the weights to the
//! bottom: it leaves them in a cloud whose expected loss is a floor set by the step and
//! the batch. Four hundred runs from one start are drawn as a swarm, with the ellipse
//! the exact recursion of their moments predicts; beside them, the expected loss
//! against the step (or the examples used), exact as a line and measured on the swarm
//! as dots.
//!
//! The bowl is L = ½(λ₂x² + λ₁y²) with the flat direction across the picture and the
//! sharp one up it. One example's gradient is the true gradient plus a Gaussian random
//! part, independent from example to example and from step to step (every example is
//! fresh: the "online" setting of the noisy quadratic model, Zhang and co-authors
//! 2019); a batch of B averages B of them, so its random part has variance σ²/B along
//! a direction in which one example's has σ². Drawing that average directly from its
//! own Gaussian is exact. Each direction is then on its own, and with heavy-ball
//! momentum v ← βv + g, θ ← θ − ηv and an exponential average a ← γa + (1 − γ)θ of the
//! weights, its state (θ, v, a) moves by a fixed linear map plus noise: its mean and
//! covariance follow exactly, step by step ([`Moments`]), which is what the lines and
//! the ellipses are. The dossier's closed forms for the floors ([`floor_one`],
//! [`averaging_factor`]) are tested against the stationary state of that recursion.

use super::*;
use crate::clock::{self, Budget};
use crate::scenes::stats::plot::{self, Glide, Plot, Rect, AXES, MARKS};

pub const DEF: SceneDef = SceneDef {
    id: "noisefloor",
    title: "SGD on a bowl: the noise floor",
    group: GROUP,
    blurb: "With noisy gradients a constant step stops making progress at a floor set by the step and the batch: one reason learning rates decay, and what averaging the weights buys.",
    make: |_| Box::new(NoiseFloor::create()),
};

// ----- the bowl and its noise ------------------------------------------------------------

/// The bowl's curvatures, the eigenvalues of its Hessian: across the picture (x) the
/// flat direction, λ₂; up it (y) the sharp one, λ₁.
const LAMBDA: [f64; 2] = [0.25, 1.0];
/// Which of the two is the sharp direction (it sets the longest step).
const SHARP: usize = 1;
/// Where every run starts, unless the start is dragged.
const START: [f64; 2] = [-2.8, 1.4];
/// How far the picture of the bowl reaches from its bottom, across and up.
const BOWL: [f64; 2] = [3.4, 1.7];
/// The runs in the swarm.
const RUNS: usize = 400;
/// What a drop divides the step by (or multiplies the batch by).
const DROP: f64 = 10.0;
/// The ranges of the controls.
const MOST_STEPS: usize = 10_000;
const LEAST_STEPS: usize = 10;
const MOST_BATCH: usize = 1024;
const MOST_ETA: f64 = 3.0;
const LEAST_ETA: f64 = 0.001;
const MOST_BETA: f64 = 0.99;
const MOST_WINDOW: usize = 2000;
/// A comparison run computes at most this many steps (enough to reach across the chart).
const MOST_COMPARED: usize = 20_000;
/// An expected loss above this is off every chart: the run has blown up.
const BLOWN: f64 = 1e12;
/// The share of a two-dimensional Gaussian cloud inside the ellipse of two standard
/// deviations: 1 − e⁻². (The tests hold the drawn ring to it.)
#[cfg(test)]
const INSIDE: f64 = 0.864_664_716_763_387_3;

/// How one example's gradient noise is spread over the two directions.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Noise {
    /// The same variance along both: σ² = 1.
    Even,
    /// In proportion to the curvature, as the noisy quadratic model assumes (its
    /// covariance a multiple of the Hessian), with the same total.
    Curved,
}

impl Noise {
    /// The variance σ² of one example's gradient along each direction (x, y). Both
    /// shapes have the same total, 2.
    fn per_example(self) -> [f64; 2] {
        match self {
            Noise::Even => [1.0, 1.0],
            Noise::Curved => {
                let s = 2.0 / (LAMBDA[0] + LAMBDA[1]);
                [s * LAMBDA[0], s * LAMBDA[1]]
            }
        }
    }
}

/// How the step (or the batch) changes during a run.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Schedule {
    Constant,
    /// From a chosen step on, the step is divided by 10 (or the batch multiplied by 10).
    Drop,
    /// η_t = ½η(1 + cos(πt/T)) over the T steps of the run, from η down to 0.
    Cosine,
}

/// Everything a run depends on but its mini-batches.
#[derive(Clone, Copy, Debug, PartialEq)]
struct Setup {
    eta: f64,
    batch: usize,
    beta: f64,
    schedule: Schedule,
    /// The first step taken with the dropped step (or the larger batch).
    drop_at: usize,
    /// At the drop the batch grows tenfold, instead of the step falling tenfold.
    drop_batch: bool,
    /// The averaged weights are shown; they average over about `window` steps.
    averaging: bool,
    window: usize,
    noise: Noise,
    steps: usize,
    start: [f64; 2],
}

impl Setup {
    /// The first experiment's: the run the scene opens with.
    fn base() -> Setup {
        Setup {
            eta: 0.1,
            batch: 4,
            beta: 0.0,
            schedule: Schedule::Constant,
            drop_at: 400,
            drop_batch: false,
            averaging: false,
            window: 100,
            noise: Noise::Even,
            steps: 2000,
            start: START,
        }
    }

    /// Kept to the controls' ranges (a value that is not a number takes the base's).
    fn kept(self) -> Setup {
        let base = Setup::base();
        let fix = |v: f64, lo: f64, hi: f64, or: f64| if v.is_finite() { v.max(lo).min(hi) } else { or };
        let steps = self.steps.clamp(LEAST_STEPS, MOST_STEPS);
        Setup {
            eta: fix(self.eta, LEAST_ETA, MOST_ETA, base.eta),
            batch: self.batch.clamp(1, MOST_BATCH),
            beta: fix(self.beta, 0.0, MOST_BETA, 0.0),
            drop_at: self.drop_at.clamp(1, steps),
            window: self.window.clamp(2, MOST_WINDOW),
            steps,
            start: [fix(self.start[0], -BOWL[0] * 0.97, BOWL[0] * 0.97, START[0]), fix(self.start[1], -BOWL[1] * 0.95, BOWL[1] * 0.95, START[1])],
            ..self
        }
    }

    /// The step used for step t (t = 0 is the first).
    fn eta_at(&self, t: usize) -> f64 {
        match self.schedule {
            Schedule::Constant => self.eta,
            Schedule::Drop if !self.drop_batch && t >= self.drop_at => self.eta / DROP,
            Schedule::Drop => self.eta,
            Schedule::Cosine => 0.5 * self.eta * (1.0 + (std::f64::consts::PI * t as f64 / self.steps.max(1) as f64).cos()),
        }
    }

    /// The batch used for step t.
    fn batch_at(&self, t: usize) -> f64 {
        let b = self.batch as f64;
        if self.schedule == Schedule::Drop && self.drop_batch && t >= self.drop_at {
            b * DROP
        } else {
            b
        }
    }

    fn dropping(&self) -> bool {
        self.schedule == Schedule::Drop
    }

    /// The share kept by the average at each step: γ = 1 − 1/W.
    fn gamma(&self) -> f64 {
        1.0 - 1.0 / self.window.max(2) as f64
    }

    /// The longest step the sharp direction bears: 2/λ₁, or (2 + 2β)/λ₁ with momentum.
    fn limit(&self) -> f64 {
        (2.0 + 2.0 * self.beta) / LAMBDA[SHARP]
    }

    /// Momentum's colour, or plain SGD's.
    fn color(&self) -> Color {
        if self.beta > 0.0 {
            MOMENTUM
        } else {
            SGD
        }
    }

    /// The stretches of the run with a constant step and batch, and the floor each
    /// would settle at if it went on: (first step, last step + 1, floor). None for a
    /// cosine schedule, whose step never stays.
    fn phases(&self) -> Vec<(usize, usize, Option<f64>)> {
        let floor = |eta: f64, batch: f64| floor_of(eta, batch, self.beta, self.noise);
        match self.schedule {
            Schedule::Constant => vec![(0, self.steps, floor(self.eta, self.batch as f64))],
            Schedule::Drop => {
                let at = self.drop_at.min(self.steps);
                let after = floor(self.eta_at(at), self.batch_at(at));
                vec![(0, at, floor(self.eta, self.batch as f64)), (at, self.steps, after)]
            }
            Schedule::Cosine => Vec::new(),
        }
    }

    /// The floor the run ends on (its last stretch's), if it has one.
    fn floor(&self) -> Option<f64> {
        self.phases().last().and_then(|p| p.2)
    }

    /// The loss at the start.
    fn start_loss(&self) -> f64 {
        loss_at(self.start)
    }

    /// The run named as the chart names it.
    fn label(&self) -> String {
        let mut s = format!(r"$\eta = {}$, $B = {}$", short(self.eta), self.batch);
        if self.beta > 0.0 {
            s += &format!(r", $\beta = {}$", short(self.beta));
        }
        match self.schedule {
            Schedule::Drop if self.drop_batch => s += &format!(r", batch $\times 10$ at {}", count(self.drop_at)),
            Schedule::Drop => s += &format!(r", $\eta/10$ at {}", count(self.drop_at)),
            Schedule::Cosine => s += ", cosine",
            Schedule::Constant => {}
        }
        s
    }
}

/// The loss at a point of the bowl.
fn loss_at(z: [f64; 2]) -> f64 {
    0.5 * (LAMBDA[0] * z[0] * z[0] + LAMBDA[1] * z[1] * z[1])
}

/// The floor of one direction of curvature λ: the expected loss in that direction
/// that a constant step η settles at, with batches of B examples whose gradients have
/// noise of variance c along it, and heavy-ball momentum β. With β = 0 it is
/// ηc/(2B(2 − ηλ)) (`noise.md` D2; Zhang and co-authors' Eq. 4); with momentum,
/// (1 + β)ηc/(2B(2β + 2 − ηλ)(1 − β)) (the steady-state term of their Theorem 1,
/// which the dossier checked is exact). None where the direction does not settle:
/// ηλ ≥ 2 + 2β.
fn floor_one(eta: f64, batch: f64, beta: f64, lambda: f64, c: f64) -> Option<f64> {
    let x = eta * lambda;
    if !(x > 0.0 && x < 2.0 + 2.0 * beta && beta < 1.0 && batch > 0.0) {
        return None;
    }
    Some((1.0 + beta) * eta * c / (2.0 * batch * (2.0 * beta + 2.0 - x) * (1.0 - beta)))
}

/// The floor of the whole bowl: the sum over its two directions.
fn floor_of(eta: f64, batch: f64, beta: f64, noise: Noise) -> Option<f64> {
    let c = noise.per_example();
    Some(floor_one(eta, batch, beta, LAMBDA[0], c[0])? + floor_one(eta, batch, beta, LAMBDA[1], c[1])?)
}

/// The floor's small-step form: η tr Σ / (4B(1 − β)) (`noise.md` D2, first order).
fn small_step_floor(eta: f64, batch: f64, beta: f64, noise: Noise) -> f64 {
    let c = noise.per_example();
    eta * (c[0] + c[1]) / (4.0 * batch * (1.0 - beta))
}

/// What an exponential average of the iterates, γ kept a step, multiplies plain SGD's
/// floor by in a direction of curvature λ: (1 − γ)(1 + aγ)/((1 + γ)(1 − aγ)) with
/// a = 1 − ηλ (Zhang and co-authors' Theorem 2; exact, `noise.md` D2). (The tests hold
/// the recursion's floor to it.)
#[cfg(test)]
fn averaging_factor(eta: f64, lambda: f64, gamma: f64) -> f64 {
    let a = 1.0 - eta * lambda;
    (1.0 - gamma) * (1.0 + a * gamma) / ((1.0 + gamma) * (1.0 - a * gamma))
}

// ----- the exact moments -----------------------------------------------------------------

/// The mean and the covariance of one direction's state (the weight z, the momentum
/// buffer v, the average a) over all the ways the mini-batches can come out.
#[derive(Clone, Copy, Debug)]
struct Moments {
    mean: [f64; 3],
    cov: [[f64; 3]; 3],
}

/// The linear map of one step on (z, v, a), and how the gradient's noise enters it:
/// v ← βv + λz + ξ, z ← z − ηv, a ← γa + (1 − γ)z.
fn step_map(lambda: f64, eta: f64, beta: f64, gamma: f64) -> ([[f64; 3]; 3], [f64; 3]) {
    let g = 1.0 - gamma;
    let a = [[1.0 - eta * lambda, -eta * beta, 0.0], [lambda, beta, 0.0], [g * (1.0 - eta * lambda), -g * eta * beta, gamma]];
    (a, [-eta, 1.0, -g * eta])
}

impl Moments {
    /// Every run at z (its average starts there too).
    fn at(z: f64) -> Moments {
        Moments { mean: [z, 0.0, z], cov: [[0.0; 3]; 3] }
    }

    /// One step, with gradient noise of variance q: the mean goes through the map,
    /// and the covariance becomes A P Aᵀ + q b bᵀ.
    fn step(&mut self, lambda: f64, eta: f64, beta: f64, gamma: f64, q: f64) {
        let (a, b) = step_map(lambda, eta, beta, gamma);
        let m = self.mean;
        self.mean = [0, 1, 2].map(|i| a[i][0] * m[0] + a[i][1] * m[1] + a[i][2] * m[2]);
        let p = self.cov;
        let mut ap = [[0.0; 3]; 3];
        for i in 0..3 {
            for j in 0..3 {
                ap[i][j] = a[i][0] * p[0][j] + a[i][1] * p[1][j] + a[i][2] * p[2][j];
            }
        }
        for i in 0..3 {
            for j in 0..3 {
                self.cov[i][j] = ap[i][0] * a[j][0] + ap[i][1] * a[j][1] + ap[i][2] * a[j][2] + q * b[i] * b[j];
            }
        }
    }

    /// The expected square of part k (0 the weight, 2 the average).
    fn second(&self, k: usize) -> f64 {
        self.mean[k] * self.mean[k] + self.cov[k][k]
    }
}

/// The stationary expected losses (of the weight, of its average) in one direction with
/// a constant step: the covariance that one step maps to itself, P = A P Aᵀ + q b bᵀ,
/// found by solving that linear equation for P's six entries. Exact for any β and γ;
/// None if the step does not settle (the map has an eigenvalue of size 1 or more).
fn stationary(lambda: f64, eta: f64, beta: f64, gamma: f64, q: f64) -> Option<(f64, f64)> {
    // (z and v alone settle when 0 < ηλ < 2 + 2β; the average only follows them)
    if !(eta * lambda > 0.0 && eta * lambda < 2.0 + 2.0 * beta && beta.abs() < 1.0 && gamma.abs() < 1.0) {
        return None;
    }
    let (a, b) = step_map(lambda, eta, beta, gamma);
    let pairs = [(0, 0), (0, 1), (0, 2), (1, 1), (1, 2), (2, 2)];
    let index = |i: usize, j: usize| pairs.iter().position(|&(p, r)| (p, r) == (i.min(j), i.max(j))).unwrap_or(0);
    let mut m = [[0.0; 7]; 6];
    for (row, &(i, j)) in pairs.iter().enumerate() {
        m[row][row] += 1.0;
        for k in 0..3 {
            for l in 0..3 {
                m[row][index(k, l)] -= a[i][k] * a[j][l];
            }
        }
        m[row][6] = q * b[i] * b[j];
    }
    let p = solve6(m)?;
    Some((0.5 * lambda * p[0], 0.5 * lambda * p[5]))
}

/// Gaussian elimination with partial pivoting on six equations (the last column is
/// the right-hand side).
fn solve6(mut m: [[f64; 7]; 6]) -> Option<[f64; 6]> {
    for c in 0..6 {
        let pivot = (c..6).max_by(|&i, &j| m[i][c].abs().total_cmp(&m[j][c].abs()))?;
        let size = m[pivot][c].abs();
        if size.is_nan() || size <= 1e-300 {
            return None;
        }
        m.swap(c, pivot);
        for r in 0..6 {
            if r != c {
                let k = m[r][c] / m[c][c];
                for col in c..7 {
                    m[r][col] -= k * m[c][col];
                }
            }
        }
    }
    let x = [0, 1, 2, 3, 4, 5].map(|i| m[i][6] / m[i][i]);
    x.iter().all(|v| v.is_finite()).then_some(x)
}

/// The exact expected course of a run: after each step t = 0 ..= steps.
#[derive(Clone, Debug)]
struct Exact {
    setup: Setup,
    /// The expected loss of the weights, and of their average.
    loss: Vec<f64>,
    avg: Vec<f64>,
    /// The mean and the variance of each coordinate (x, y), of the weights and of
    /// their average.
    mean: Vec<[f64; 2]>,
    var: Vec<[f64; 2]>,
    avg_mean: Vec<[f64; 2]>,
    avg_var: Vec<[f64; 2]>,
    /// The examples used in the first t steps.
    examples: Vec<f64>,
    /// The first step whose expected loss is off every chart, if it blows up.
    blown: Option<usize>,
}

impl Exact {
    /// The run of `s`, for `steps` steps (its own, or more for a comparison drawn across
    /// a longer chart: its schedule is still that of its own `steps`).
    fn of(s: &Setup, steps: usize) -> Exact {
        let c = s.noise.per_example();
        let gamma = s.gamma();
        let mut m = [Moments::at(s.start[0]), Moments::at(s.start[1])];
        let n = steps + 1;
        let mut e = Exact {
            setup: *s,
            loss: Vec::with_capacity(n),
            avg: Vec::with_capacity(n),
            mean: Vec::with_capacity(n),
            var: Vec::with_capacity(n),
            avg_mean: Vec::with_capacity(n),
            avg_var: Vec::with_capacity(n),
            examples: Vec::with_capacity(n),
            blown: None,
        };
        let mut examples = 0.0;
        for t in 0..=steps {
            if t > 0 {
                let (eta, batch) = (s.eta_at(t - 1), s.batch_at(t - 1));
                examples += batch;
                for i in 0..2 {
                    m[i].step(LAMBDA[i], eta, s.beta, gamma, c[i] / batch);
                }
            }
            let loss = 0.5 * (LAMBDA[0] * m[0].second(0) + LAMBDA[1] * m[1].second(0));
            let avg = 0.5 * (LAMBDA[0] * m[0].second(2) + LAMBDA[1] * m[1].second(2));
            if e.blown.is_none() && !(loss <= BLOWN && avg <= BLOWN * 1e3) {
                e.blown = Some(t);
            }
            if e.blown.is_some() {
                // (what is beyond every chart is not followed further: it would overflow)
                for v in [&mut e.loss, &mut e.avg] {
                    v.push(f64::INFINITY);
                }
                for v in [&mut e.mean, &mut e.var, &mut e.avg_mean, &mut e.avg_var] {
                    v.push([f64::INFINITY; 2]);
                }
            } else {
                e.loss.push(loss);
                e.avg.push(avg);
                e.mean.push([m[0].mean[0], m[1].mean[0]]);
                e.var.push([m[0].cov[0][0], m[1].cov[0][0]]);
                e.avg_mean.push([m[0].mean[2], m[1].mean[2]]);
                e.avg_var.push([m[0].cov[2][2], m[1].cov[2][2]]);
            }
            e.examples.push(examples);
        }
        e
    }

    fn steps(&self) -> usize {
        self.loss.len().saturating_sub(1)
    }

    /// From which step on the expected loss stays within a tenth of `floor` to the end
    /// of the run (None if it is not there at the end).
    fn settled_from(&self, floor: f64, from: usize) -> Option<usize> {
        let near = |l: f64| (l - floor).abs() <= 0.1 * floor;
        let last = self.loss.len().checked_sub(1)?;
        if !near(self.loss[last]) {
            return None;
        }
        let mut t = last;
        while t > from && near(self.loss[t - 1]) {
            t -= 1;
        }
        Some(t)
    }

    /// The mean expected loss over steps `from ..= to`.
    fn mean_over(&self, from: usize, to: usize, avg: bool) -> f64 {
        let v = if avg { &self.avg } else { &self.loss };
        let (a, b) = (from.min(v.len()), (to + 1).min(v.len()));
        if b <= a {
            return f64::NAN;
        }
        v[a..b].iter().sum::<f64>() / (b - a) as f64
    }
}

// ----- the swarm -------------------------------------------------------------------------

/// What the swarm's runs came to at one step: their mean loss and its standard error
/// (the runs' spread over √RUNS), for the weights and for their averages.
#[derive(Clone, Copy, Debug, PartialEq)]
struct Mark {
    step: usize,
    loss: f64,
    se: f64,
    avg: f64,
    avg_se: f64,
}

/// The steps at which the swarm's mean loss is noted: ten a decade, and the last.
fn checkpoints(steps: usize) -> Vec<usize> {
    let mut v: Vec<usize> = Vec::new();
    let mut k = 0;
    loop {
        let t = 10f64.powf(k as f64 / 10.0).round() as usize;
        if t > steps {
            break;
        }
        if v.last() != Some(&t) {
            v.push(t);
        }
        k += 1;
    }
    if v.last() != Some(&steps) {
        v.push(steps);
    }
    v
}

/// RUNS independent runs of the setup from its start, each with its own mini-batches
/// (all drawn from one seeded generator, run after run, step after step).
#[derive(Clone, Debug)]
struct Swarm {
    setup: Setup,
    seed: u64,
    rng: Rng,
    /// Steps taken.
    t: usize,
    z: Vec<[f64; 2]>,
    v: Vec<[f64; 2]>,
    a: Vec<[f64; 2]>,
    /// Where each run and its average were one step before (the picture is drawn
    /// between the two).
    z_before: Vec<[f64; 2]>,
    a_before: Vec<[f64; 2]>,
    checks: Vec<usize>,
    marks: Vec<Mark>,
    /// Over the last quarter of the run, each run's summed loss (weights, average):
    /// the plateau as measured.
    tail_from: usize,
    tail: Vec<[f64; 2]>,
    /// A run has flown out of all reach: the swarm stops there.
    diverged: bool,
}

impl Swarm {
    fn new(setup: Setup, seed: u64) -> Swarm {
        let at = vec![setup.start; RUNS];
        Swarm {
            setup,
            seed,
            rng: Rng::new(seed.wrapping_mul(0x9E37_79B9).wrapping_add(17)),
            t: 0,
            z: at.clone(),
            v: vec![[0.0; 2]; RUNS],
            a: at.clone(),
            z_before: at.clone(),
            a_before: at,
            checks: checkpoints(setup.steps),
            marks: Vec::new(),
            tail_from: setup.steps - setup.steps / 4,
            tail: vec![[0.0; 2]; RUNS],
            diverged: false,
        }
    }

    fn done(&self) -> bool {
        self.t >= self.setup.steps || self.diverged
    }

    /// One step of every run.
    fn step(&mut self) {
        if self.done() {
            return;
        }
        let s = self.setup;
        let (eta, batch) = (s.eta_at(self.t), s.batch_at(self.t));
        let c = s.noise.per_example();
        let sd = [(c[0] / batch).sqrt(), (c[1] / batch).sqrt()];
        let (beta, gamma) = (s.beta, s.gamma());
        self.z_before.copy_from_slice(&self.z);
        self.a_before.copy_from_slice(&self.a);
        let mut far = false;
        for r in 0..RUNS {
            let noise = [self.rng.normal() as f64, self.rng.normal() as f64];
            let (z, v, a) = (&mut self.z[r], &mut self.v[r], &mut self.a[r]);
            for i in 0..2 {
                let g = LAMBDA[i] * z[i] + sd[i] * noise[i];
                v[i] = beta * v[i] + g;
                z[i] -= eta * v[i];
                a[i] = gamma * a[i] + (1.0 - gamma) * z[i];
            }
            far |= !(z[0].abs() < 1e7 && z[1].abs() < 1e7);
        }
        self.t += 1;
        if far {
            self.diverged = true;
            return;
        }
        if self.t > self.tail_from {
            for r in 0..RUNS {
                self.tail[r][0] += loss_at(self.z[r]);
                self.tail[r][1] += loss_at(self.a[r]);
            }
        }
        if self.checks.binary_search(&self.t).is_ok() {
            let (loss, se) = Self::mean_se(self.z.iter().map(|z| loss_at(*z)));
            let (avg, avg_se) = Self::mean_se(self.a.iter().map(|a| loss_at(*a)));
            self.marks.push(Mark { step: self.t, loss, se, avg, avg_se });
        }
    }

    /// The mean of RUNS numbers and its standard error.
    fn mean_se(values: impl Iterator<Item = f64>) -> (f64, f64) {
        let (mut sum, mut sq) = (0.0, 0.0);
        for v in values {
            sum += v;
            sq += v * v;
        }
        let n = RUNS as f64;
        let mean = sum / n;
        let var = ((sq - n * mean * mean) / (n - 1.0)).max(0.0);
        (mean, (var / n).sqrt())
    }

    /// The plateau as measured, once the run is over: the mean over the runs of each
    /// run's mean loss over the last quarter of its steps, with its standard error
    /// (weights; with `avg`, their averages). The steps are `tail_from + 1 ..= steps`.
    fn plateau(&self, avg: bool) -> Option<(f64, f64)> {
        if self.t < self.setup.steps || self.diverged {
            return None;
        }
        let n = (self.setup.steps - self.tail_from).max(1) as f64;
        let k = avg as usize;
        Some(Self::mean_se(self.tail.iter().map(|t| t[k] / n)))
    }

    /// The share of the runs inside the ellipse of two standard deviations round the
    /// exact mean, with the exact variances (the weights; with `avg`, their averages).
    fn inside(&self, e: &Exact, avg: bool) -> f64 {
        let t = self.t.min(e.steps());
        let (mean, var) = if avg { (e.avg_mean[t], e.avg_var[t]) } else { (e.mean[t], e.var[t]) };
        let pts = if avg { &self.a } else { &self.z };
        if !(var[0] > 0.0 && var[1] > 0.0) {
            return f64::NAN;
        }
        let k = pts
            .iter()
            .filter(|p| {
                let (dx, dy) = (p[0] - mean[0], p[1] - mean[1]);
                dx * dx / var[0] + dy * dy / var[1] <= 4.0
            })
            .count();
        k as f64 / RUNS as f64
    }

    /// Where the runs (or their averages) are at the shown step `s`, between the last
    /// step and the one before.
    fn shown(&self, s: f64, avg: bool) -> impl Iterator<Item = [f64; 2]> + '_ {
        let frac = if self.t == 0 { 1.0 } else { (s - (self.t - 1) as f64).clamp(0.0, 1.0) };
        let (now, before) = if avg { (&self.a, &self.a_before) } else { (&self.z, &self.z_before) };
        now.iter().zip(before).map(move |(n, b)| [b[0] + (n[0] - b[0]) * frac, b[1] + (n[1] - b[1]) * frac])
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

/// A ratio written with two significant digits after the point where it is near 1
/// (1.02, 0.97), else three significant digits.
fn ratio(v: f64) -> String {
    if v.is_finite() && (0.9..1.1).contains(&v) {
        num(v, 3)
    } else {
        sig(v)
    }
}

// ----- the experiments ---------------------------------------------------------------------

/// Which axis the chart is drawn against.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Axis {
    Steps,
    Examples,
}

struct Trial {
    button: &'static str,
    caption: &'static str,
}

const TRIALS: [Trial; 7] = [
    Trial {
        button: "a constant step",
        caption: r"Step $\eta = 0.1$, batches of 4. The runs slide down together while the bowl's pull is far larger than the noise (the descent), then stop getting closer and spread into a cloud that stays (the plateau). Their expected loss (the line) settles on the floor the formula gives, 0.0129, 152 times below the start, and is within a tenth of it from step 131 on; the runs' mean (the dots) follows it within its chance error. After that the steps only stir the cloud.",
    },
    Trial {
        button: "half the step",
        caption: r"$\eta = 0.05$ against 0.1 (faint). The floor is 0.00635, a little under half of 0.0129: the formula also divides by $2 - \eta\lambda$, which grows as $\eta$ shrinks. It is reached later: within a tenth from step 292 instead of 131. The longer step is lower until step 198, the shorter one from step 199 on: faster in the descent, worse on the plateau.",
    },
    Trial {
        button: "twice the batch instead",
        caption: r"$\eta = 0.1$ with batches of 8, against the first run and against half the step. The floor is 0.00645, within 2 % of half the step's 0.00635 (a little higher: the step is longer), and it comes at the first run's pace per step: within a tenth from step 145. But each step costs 8 examples: against the examples used (the button on the chart), this run and half the step's lie nearly on one line.",
    },
    Trial {
        button: "step and batch together",
        caption: r"Against the examples used: $\eta = 0.025$ with batches of 1, then step and batch both times 4, 16 and 64. Per example the first three nearly agree (floors 0.0126, 0.0129, 0.0144). The last, the swarm, has $\eta = 1.6$, near the sharp direction's limit $2/\lambda_1 = 2$: each step flips its runs across the valley, and its floor is 0.0391, 3.1 times the first. In the sharp direction the exact floor divides by $2 - \eta\lambda = 0.4$ where the small-step form has 2: five times as high. Scaled by 80, the step would be 2 and that direction would never settle. Goyal and co-authors (2017) used this rule for ResNet-50 on ImageNet (momentum SGD, with a warm-up): it held up to batches of about 8 000 images; beyond, accuracy fell, and beyond 64 000 training diverged.",
    },
    Trial {
        button: "a drop of the step",
        caption: r"The first run, with its step divided by 10 at step 400: the cloud shrinks, and the loss goes down to the new floor, 0.00125, within a tenth of it 763 steps after the drop: a short step settles slowly. The faint line that falls steeply at the drop multiplies the batch by 10 instead: floor 0.00129, reached 75 steps after the drop, using 40 examples a step: about as many in all (3000) as the drop of the step needs (3052). Smith and co-authors (2018) did this with image classifiers trained by SGD, with momentum, and by Adam: the curves per epoch matched those of decaying the step, with fewer steps; they grew the batch only to about a tenth of the data set.",
    },
    Trial {
        button: "averaged weights",
        caption: r"The first run, and the average of each run's weights over about the last 100 steps (white): a mean of the points the run visits, which lies nearer the bottom than they do. Its floor is 0.00237, 5.4 times lower, as Zhang and co-authors' formula gives. But the average trails behind the weights during the descent, and is only below them from step 294.",
    },
    Trial {
        button: "momentum",
        caption: r"Momentum $\beta = 0.9$ at the first run's step: each gradient goes on acting for about $1/(1 - \beta) = 10$ steps, and so does its noise. The cloud is far wider and the floor is 0.127, 9.8 times plain SGD's (faint cyan). With the step cut to a tenth, 0.01 (faint pink), the floor is 0.0125, 3 % below plain SGD's 0.0129: momentum at $\eta(1 - \beta)$ is close to plain SGD at $\eta$, as the floor's small-step form, $\eta\sigma^2/(4B(1 - \beta))$, says. Shallue and co-authors (2019), tuning both at each batch size on many networks, found that at small batches momentum took no fewer steps than plain SGD.",
    },
];

/// The scene opens as the first experiment sets it up.
const OPENING: usize = 0;
/// The seed of the swarm each experiment shows.
const SEED: u64 = 1;

/// The setup, the runs drawn faint beside it, and the chart's axis, of experiment k.
fn trial(k: usize) -> (Setup, Vec<Setup>, Axis) {
    let base = Setup::base();
    match k {
        1 => (Setup { eta: 0.05, ..base }, vec![base], Axis::Steps),
        2 => (Setup { batch: 8, ..base }, vec![base, Setup { eta: 0.05, ..base }], Axis::Steps),
        3 => {
            let at = |k: usize, steps: usize| Setup { eta: 0.025 * k as f64, batch: k, steps, ..base };
            (at(64, 125), vec![at(1, 8000), at(4, 2000), at(16, 500)], Axis::Examples)
        }
        4 => {
            let drop = Setup { schedule: Schedule::Drop, drop_at: 400, ..base };
            (drop, vec![base, Setup { drop_batch: true, ..drop }], Axis::Steps)
        }
        5 => (Setup { averaging: true, window: 100, ..base }, Vec::new(), Axis::Steps),
        6 => (Setup { beta: 0.9, ..base }, vec![base, Setup { beta: 0.9, eta: 0.01, ..base }], Axis::Steps),
        _ => (base, Vec::new(), Axis::Steps),
    }
}

// ----- the film ------------------------------------------------------------------------------

/// The run played as a film on a logarithmic clock: the step shown is
/// (T + 1)^(clock/length) − 1, so the first steps come one by one and the plateau
/// goes by quickly, each decade of steps taking the same time.
#[derive(Clone, Copy, Debug, Default)]
struct Film {
    clock: f64,
    playing: bool,
    /// Play once the picture has glided to the new run's start.
    waiting: bool,
}

impl Film {
    /// How long the whole run takes to play, in seconds.
    fn length(steps: usize) -> f64 {
        2.5 + 0.9 * ((steps + 1) as f64).ln()
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
}

// ----- the scene -----------------------------------------------------------------------------

/// How many points each curve of the chart is drawn with.
const CURVE_N: usize = 160;
/// The runs drawn faint for comparison, at most.
const MOST_COMPARE: usize = 4;

pub struct NoiseFloor {
    setup: Setup,
    /// Runs drawn faint beside this one.
    compare: Vec<Setup>,
    axis: Axis,
    seed: u64,
    exact: Exact,
    compared: Vec<Exact>,
    swarm: Swarm,
    film: Film,
    /// The step shown (the film's, as far as the swarm has got).
    shown: f64,
    look: Look,
    /// The start is being dragged.
    dragging: bool,
    /// The window last drawn in (a new one is drawn at once, without gliding).
    canvas: Option<Vec2>,
    /// Where the chart's labels were put, and whether one is gliding to a new place.
    placed: Vec<Placed>,
    labels_moving: bool,
    /// How far the strip's name is drawn from its line (up is negative).
    strip_label: f32,
    /// The chart's ranges as they are to be, and what they were worked out for.
    goal: Option<(GoalKey, Vec<f32>)>,
}

/// What the chart's ranges depend on: the run, the runs compared with it, the axis.
type GoalKey = (Setup, Vec<Setup>, Axis);

/// What is drawn, gliding to where it is going.
#[derive(Clone, Debug, Default)]
struct Look {
    /// The chart's ranges: log₁₀ of x from and to, of the loss from and to; the
    /// strip's top.
    frame: Glide,
    /// Each curve's points (log₁₀ of the loss at `CURVE_N` places across), and the
    /// strip's (the step size there): this run's weights, their average, then the
    /// comparisons'.
    curves: Vec<Glide>,
    strips: Vec<Glide>,
    /// Where each curve has a value (not glided).
    masks: Vec<Vec<bool>>,
    /// The floors' heights (log₁₀) and where they start and end across (log₁₀ of x).
    floors: Glide,
    /// The runs' dots and their averages' (x, y each), and the two ellipses (centre
    /// and half-axes).
    dots: Glide,
    rings: Glide,
    moving: bool,
}

impl NoiseFloor {
    /// (Nothing here needs the graphics device, so the tests can make one too.)
    fn create() -> NoiseFloor {
        let setup = Setup::base();
        let mut s = NoiseFloor {
            setup,
            compare: Vec::new(),
            axis: Axis::Steps,
            seed: SEED,
            exact: Exact::of(&setup, setup.steps),
            compared: Vec::new(),
            swarm: Swarm::new(setup, SEED),
            film: Film::default(),
            shown: 0.0,
            look: Look::default(),
            dragging: false,
            canvas: None,
            placed: Vec::new(),
            labels_moving: false,
            strip_label: 0.0,
            goal: None,
        };
        s.set_up(OPENING);
        s
    }

    /// Set up experiment k, and play it.
    fn set_up(&mut self, k: usize) {
        let (setup, compare, axis) = trial(k);
        self.setup = setup;
        self.compare = compare;
        self.axis = axis;
        self.seed = SEED;
        self.restart();
    }

    /// Start the film again from the start, once the picture has glided there.
    fn restart(&mut self) {
        self.setup = self.setup.kept();
        self.swarm = Swarm::new(self.setup, self.seed);
        self.film = Film { clock: 0.0, playing: false, waiting: true };
        self.shown = 0.0;
        self.refresh();
    }

    /// Bring what depends on the settings up to date with them (a changed setup
    /// starts the film again).
    fn refresh(&mut self) {
        self.setup = self.setup.kept();
        let s = self.setup;
        if self.exact.setup != s || self.exact.steps() != s.steps {
            self.exact = Exact::of(&s, s.steps);
        }
        if self.swarm.setup != s || self.swarm.seed != self.seed {
            self.swarm = Swarm::new(s, self.seed);
            self.film = Film { clock: 0.0, playing: false, waiting: true };
            self.shown = 0.0;
        }
        self.compare.truncate(MOST_COMPARE);
        // the comparisons are drawn across the whole chart: as many steps as that takes
        let reach = self.reach();
        let wanted: Vec<(Setup, usize)> = self.compare.iter().map(|c| (c.kept(), self.compared_steps(&c.kept(), reach))).collect();
        let same = self.compared.len() == wanted.len() && self.compared.iter().zip(&wanted).all(|(e, (c, n))| e.setup == *c && e.steps() == *n);
        if !same {
            self.compared = wanted.iter().map(|(c, n)| Exact::of(c, *n)).collect();
        }
    }

    /// How far the chart reaches across: this run's last step, or the examples it uses.
    fn reach(&self) -> f64 {
        match self.axis {
            Axis::Steps => self.setup.steps as f64,
            Axis::Examples => self.exact.examples.last().copied().unwrap_or(1.0),
        }
    }

    /// The steps a comparison is computed for, to reach across the chart (at least its own).
    fn compared_steps(&self, c: &Setup, reach: f64) -> usize {
        let n = match self.axis {
            Axis::Steps => reach.ceil() as usize,
            Axis::Examples => {
                // (the batch may grow at a drop: the steps up to it, then the rest at the larger batch)
                let before = if c.dropping() && c.drop_batch { c.drop_at as f64 * c.batch as f64 } else { f64::INFINITY };
                if reach <= before {
                    (reach / c.batch as f64).ceil() as usize
                } else {
                    c.drop_at + ((reach - before) / (c.batch as f64 * DROP)).ceil() as usize
                }
            }
        };
        n.max(c.steps).clamp(1, MOST_COMPARED)
    }

    /// Is the film playing (or about to)?
    fn busy(&self) -> bool {
        self.film.playing || self.film.waiting
    }

    /// Move the film on by `dt` seconds of the scene's time, stepping the swarm as far
    /// as it shows (within `budget`).
    fn advance(&mut self, dt: f32, budget: &Budget) {
        if self.film.waiting && !self.look.moving && !self.dragging {
            self.film.waiting = false;
            self.film.playing = true;
        }
        let steps = self.setup.steps;
        if self.film.playing && dt > 0.0 {
            self.film.clock += dt as f64;
        }
        let want = Film::step_at(self.film.clock, steps);
        let goal = (want.ceil() as usize).min(steps);
        // (at least one step a frame, so that even a slow machine sees the film go on)
        let mut first = true;
        while self.swarm.t < goal && !self.swarm.diverged && (first || !budget.spent()) {
            self.swarm.step();
            first = false;
        }
        // the picture shows no more than the swarm has done; the clock waits for it
        let shown = want.min(self.swarm.t as f64);
        if shown < want {
            self.film.clock = Film::clock_of(shown, steps);
        }
        self.shown = shown;
        // (a swarm that has flown off ends the film where it stopped)
        let end = if self.swarm.diverged { self.swarm.t as f64 } else { steps as f64 };
        if self.film.playing && self.swarm.done() && shown >= end - 1e-9 {
            self.film.playing = false;
        }
    }

    /// Play from where the film is (from the start, if it is at the end).
    fn play(&mut self) {
        if self.swarm.done() {
            self.restart();
        } else {
            self.film.playing = true;
        }
    }

    /// The exact mean and variance along x and y, of the weights and of their average,
    /// at a step between two whole ones (straight between the two).
    fn exact_at(&self, s: f64) -> ([f64; 2], [f64; 2], [f64; 2], [f64; 2]) {
        let e = &self.exact;
        let n = e.steps();
        let (i, frac) = if s <= 0.0 { (0, 0.0) } else { ((s.ceil() as usize).min(n), s - (s.ceil() - 1.0)) };
        let j = i.saturating_sub(1);
        let mix = |v: &[[f64; 2]]| {
            let (a, b) = (v.get(j).copied().unwrap_or([0.0; 2]), v.get(i).copied().unwrap_or([0.0; 2]));
            if i == 0 {
                b
            } else {
                [a[0] + (b[0] - a[0]) * frac, a[1] + (b[1] - a[1]) * frac]
            }
        };
        (mix(&e.mean), mix(&e.var), mix(&e.avg_mean), mix(&e.avg_var))
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
    /// The bowl's plot and its title's place.
    bowl: Rect,
    bowl_head: Vec2,
    /// The buttons: play (or pause), again, steps, examples, keep this run.
    buttons: [Rect; 5],
    /// The chart's title, the strip of the step size and the chart of the loss (their
    /// plots' rectangles).
    chart_head: Vec2,
    strip: Rect,
    chart: Rect,
    /// Only the bowl: the window is too small for the rest.
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
        let mut lay = Layout {
            view,
            narrow,
            text,
            small,
            bowl: none,
            bowl_head: Vec2::new(side, top),
            buttons: [none; 5],
            chart_head: Vec2::ZERO,
            strip: none,
            chart: none,
            compact: false,
        };
        let width = view.x - 2.0 * side;
        let bowl_at = |x0: f32, y0: f32, w: f32, h: f32| {
            // (two units across for one up: the bowl's own proportions)
            let h = h.min(w * 0.5).max(1.0);
            let x0 = x0 + (w - 2.0 * h) * 0.5;
            r(x0, y0, x0 + 2.0 * h, y0 + h)
        };
        let button_h = if narrow { 24.0 } else { 26.0 };
        // (what the chart needs under the bowl: the buttons, a title, the strip, the
        // chart itself and its numbers)
        let strip_h = if narrow { 36.0 } else { 44.0 };
        let below = 12.0 + button_h + 14.0 + head + strip_h + 16.0 + 130.0 + 40.0;
        let wide = view.x >= 900.0 && view.x >= 1.45 * view.y;
        if (view.y - top - head - below < 80.0 && !wide) || view.x < 250.0 {
            // the bowl alone
            lay.compact = true;
            lay.bowl = bowl_at(side, top + head, width, (view.y - top - head - 24.0).max(1.0));
            return lay;
        }
        let buttons_at = |lay: &mut Layout, x0: f32, x1: f32, y: f32| {
            let row = plot::even_row(5, x0, x1, y, button_h, 6.0);
            for (k, b) in row.into_iter().enumerate().take(5) {
                lay.buttons[k] = b;
            }
        };
        let chart_in = |lay: &mut Layout, x0: f32, x1: f32, y0: f32, y1: f32| {
            lay.chart_head = Vec2::new(x0, y0 + head - 4.0);
            let (left, right) = (x0 + if narrow { 40.0 } else { 46.0 }, x1 - 8.0);
            lay.strip = r(left, y0 + head + 6.0, right, y0 + head + 6.0 + strip_h);
            lay.chart = r(left, lay.strip.1.y + 16.0, right, y1 - 38.0);
        };
        if wide {
            // the bowl on the left with the buttons under it, the charts on the right
            let column = (view.x - 3.0 * side) * 0.5;
            let bowl_h = (view.y - top - head - button_h - 40.0).min(column * 0.5);
            lay.bowl = bowl_at(side, top + head, column, bowl_h);
            let y = lay.bowl.1.y + 14.0;
            buttons_at(&mut lay, side, side + column, y);
            let x0 = 2.0 * side + column;
            chart_in(&mut lay, x0, view.x - side, top, view.y);
        } else {
            let bowl_h = (view.y - top - head - below).min(view.y * if narrow { 0.3 } else { 0.42 });
            lay.bowl = bowl_at(side, top + head, width, bowl_h);
            let y = lay.bowl.1.y + 12.0;
            buttons_at(&mut lay, side, view.x - side, y);
            chart_in(&mut lay, side, view.x - side, y + button_h + 10.0, view.y);
        }
        lay
    }
}

// ----- what is drawn, gliding ----------------------------------------------------------------

// The chart's ranges, in the order the glide keeps them.
const X_LO: usize = 0;
const X_HI: usize = 1;
const Y_LO: usize = 2;
const Y_HI: usize = 3;
const STRIP_HI: usize = 4;

/// A value that is not a number glides as this (off the bottom of every chart) and is
/// not drawn.
const NOWHERE: f32 = -30.0;

/// log₁₀ of x at point k of `CURVE_N` across a range.
fn across(k: usize, lo: f64, hi: f64) -> f64 {
    lo + (hi - lo) * k as f64 / (CURVE_N - 1) as f64
}

/// The (fractional) step at which a run has got to x along the axis (None before its
/// first step or after its last).
fn step_for(e: &Exact, axis: Axis, x: f64) -> Option<f64> {
    let n = e.steps();
    match axis {
        Axis::Steps => (x >= 1.0 - 1e-9 && x <= n as f64 + 1e-9).then_some(x.min(n as f64)),
        Axis::Examples => {
            let ex = &e.examples;
            if n == 0 || !(x >= ex[1] - 1e-9 && x <= ex[n] + 1e-9) {
                return None;
            }
            let i = ex.partition_point(|&v| v < x).clamp(1, n);
            let (a, b) = (ex[i - 1], ex[i]);
            let frac = if b > a { ((x - a) / (b - a)).clamp(0.0, 1.0) } else { 1.0 };
            Some((i - 1) as f64 + frac)
        }
    }
}

/// log₁₀ of a run's expected loss (of the weights, or of their average) at a
/// fractional step: straight between the steps on the logarithmic scale.
fn log_loss_at(e: &Exact, t: f64, avg: bool) -> f64 {
    let v = if avg { &e.avg } else { &e.loss };
    let n = e.steps();
    let i = (t.floor().max(0.0) as usize).min(n);
    let j = (i + 1).min(n);
    let frac = t - i as f64;
    let (a, b) = (v[i].max(1e-300).log10(), v[j].max(1e-300).log10());
    a + (b - a) * frac
}

/// Where the swarm's dot for a step goes across the chart (log₁₀ of x).
fn x_of(e: &Exact, axis: Axis, step: usize) -> f64 {
    match axis {
        Axis::Steps => (step.max(1) as f64).log10(),
        Axis::Examples => e.examples.get(step.min(e.steps())).copied().unwrap_or(1.0).max(1.0).log10(),
    }
}

/// The same for a step between two whole ones (the film's), straight between them.
fn x_between(e: &Exact, axis: Axis, step: f64) -> f64 {
    match axis {
        Axis::Steps => step.max(1.0).log10(),
        Axis::Examples => {
            let n = e.steps();
            let i = (step.floor().max(0.0) as usize).min(n);
            let j = (i + 1).min(n);
            let (a, b) = (e.examples[i], e.examples[j]);
            (a + (b - a) * (step - i as f64).clamp(0.0, 1.0)).max(1.0).log10()
        }
    }
}

/// A curve's points across the chart's range, and where it has a value.
fn curve_points(e: &Exact, axis: Axis, lo: f64, hi: f64, avg: bool) -> (Vec<f32>, Vec<bool>) {
    let mut pts = Vec::with_capacity(CURVE_N);
    let mut mask = Vec::with_capacity(CURVE_N);
    for k in 0..CURVE_N {
        let x = 10f64.powf(across(k, lo, hi));
        match step_for(e, axis, x) {
            Some(t) => {
                let i = (t.ceil() as usize).min(e.steps());
                let v = if avg { e.avg[i] } else { e.loss[i] };
                if v.is_finite() {
                    pts.push(log_loss_at(e, t, avg) as f32);
                    mask.push(true);
                } else {
                    // (blown up: off the top)
                    pts.push(-NOWHERE);
                    mask.push(false);
                }
            }
            None => {
                pts.push(f32::NAN);
                mask.push(false);
            }
        }
    }
    fill_gaps(&mut pts);
    (pts, mask)
}

/// The step size across the chart's range (for the strip).
fn strip_points(e: &Exact, axis: Axis, lo: f64, hi: f64) -> (Vec<f32>, Vec<bool>) {
    let mut pts = Vec::with_capacity(CURVE_N);
    let mut mask = Vec::with_capacity(CURVE_N);
    for k in 0..CURVE_N {
        let x = 10f64.powf(across(k, lo, hi));
        match step_for(e, axis, x) {
            Some(t) => {
                pts.push(e.setup.eta_at((t.ceil() as usize).saturating_sub(1)) as f32);
                mask.push(true);
            }
            None => {
                pts.push(f32::NAN);
                mask.push(false);
            }
        }
    }
    fill_gaps(&mut pts);
    (pts, mask)
}

/// Points that are not numbers take their nearest neighbour's value (so that a glide
/// to or from them stays on the chart), or `NOWHERE` if there is none.
fn fill_gaps(pts: &mut [f32]) {
    let Some(first) = pts.iter().position(|v| v.is_finite()) else {
        pts.iter_mut().for_each(|v| *v = NOWHERE);
        return;
    };
    let mut last = pts[first];
    for v in pts.iter_mut() {
        if v.is_finite() {
            last = *v;
        } else {
            *v = last;
        }
    }
}

impl NoiseFloor {
    /// The chart's ranges as they are to be: across, the run's steps (or examples)
    /// from the first; up, from under the lowest floor or curve to over the start.
    fn frame_goal(&self) -> Vec<f32> {
        let s = &self.setup;
        let x_lo = match self.axis {
            Axis::Steps => 0.0,
            Axis::Examples => {
                let first = |e: &Exact| e.examples.get(1).copied().unwrap_or(1.0);
                self.compared.iter().map(first).fold(first(&self.exact), f64::min).max(1.0).log10()
            }
        };
        let x_hi = self.reach().max(2.0).log10().max(x_lo + 0.3);
        let mut lo = f64::INFINITY;
        let mut hi: f64 = 0.0;
        let mut see = |e: &Exact, avg: bool| {
            let v = if avg { &e.avg } else { &e.loss };
            for (t, &l) in v.iter().enumerate().skip(1) {
                let x = match self.axis {
                    Axis::Steps => t as f64,
                    Axis::Examples => e.examples[t],
                };
                if x <= 10f64.powf(x_hi) * 1.000_001 && l.is_finite() && l > 0.0 {
                    lo = lo.min(l);
                    hi = hi.max(l);
                }
            }
            hi = hi.max(e.loss.first().copied().unwrap_or(0.0));
        };
        see(&self.exact, false);
        if s.averaging {
            see(&self.exact, true);
        }
        for e in &self.compared {
            see(e, false);
        }
        for (_, _, floor) in s.phases() {
            if let Some(f) = floor {
                lo = lo.min(f);
            }
        }
        if let (true, Some(a)) = (s.averaging, self.averaged_floor()) {
            lo = lo.min(a);
        }
        let start = s.start_loss();
        if !(lo.is_finite() && lo > 0.0) {
            lo = (start * 0.01).max(1e-6);
        }
        if hi.is_nan() || hi <= 0.0 {
            hi = start.max(lo * 10.0);
        }
        // (a run that blows up is followed only a little way above its start)
        let cap = start.max(lo * 10.0) * 300.0;
        let (y_lo, mut y_hi) = (lo.log10() - 0.45, hi.min(cap).log10() + 0.3);
        if y_hi - y_lo < 1.2 {
            y_hi = y_lo + 1.2;
        }
        // the strip: up to a little over the largest step, and up to the limit if it is near
        let most = std::iter::once(s).chain(self.compare.iter()).map(|c| c.eta).fold(0.0, f64::max);
        let limit = s.limit();
        // (the line at a little over half the height, its name above it)
        let strip_hi = if most * 2.0 > limit { limit.max(most) * 1.45 } else { most * 1.8 };
        let mut g = vec![0.0f32; 5];
        g[X_LO] = x_lo as f32;
        g[X_HI] = x_hi as f32;
        g[Y_LO] = y_lo as f32;
        g[Y_HI] = y_hi as f32;
        g[STRIP_HI] = strip_hi as f32;
        g
    }

    /// The same, worked out again only when what it depends on changes (it reads
    /// every step of every run).
    fn goal(&mut self) -> Vec<f32> {
        let key = (self.setup, self.compare.clone(), self.axis);
        if self.goal.as_ref().is_none_or(|(k, _)| *k != key) {
            self.goal = Some((key, self.frame_goal()));
        }
        self.goal.as_ref().map(|(_, g)| g.clone()).unwrap_or_default()
    }

    /// The runs' dots as they are to be (x, y each): the weights, then their averages.
    fn dots_goal(&self) -> Vec<f32> {
        let mut v = Vec::with_capacity(4 * RUNS);
        for avg in [false, true] {
            for p in self.swarm.shown(self.shown, avg) {
                // (a run flown far off waits just beyond the picture's edge)
                let keep = |a: f64, edge: f64| if a.is_finite() { a.max(-3.0 * edge).min(3.0 * edge) } else { 3.0 * edge };
                v.push(keep(p[0], BOWL[0]) as f32);
                v.push(keep(p[1], BOWL[1]) as f32);
            }
        }
        v
    }

    /// The two ellipses as they are to be: centre and half-axes (two standard
    /// deviations), of the weights and of their averages.
    fn rings_goal(&self) -> Vec<f32> {
        let (mean, var, avg_mean, avg_var) = self.exact_at(self.shown);
        let mut v = Vec::with_capacity(8);
        for (m, s) in [(mean, var), (avg_mean, avg_var)] {
            let keep = |a: f64, edge: f64| if a.is_finite() { a.max(-3.0 * edge).min(3.0 * edge) } else { 0.0 };
            v.push(keep(m[0], BOWL[0]) as f32);
            v.push(keep(m[1], BOWL[1]) as f32);
            v.push(keep(2.0 * s[0].max(0.0).sqrt(), BOWL[0]) as f32);
            v.push(keep(2.0 * s[1].max(0.0).sqrt(), BOWL[1]) as f32);
        }
        v
    }

    /// The floors as they are to be: for this run's two stretches (one, if it has one
    /// floor), log₁₀ of the floor and where across it starts and ends.
    fn floors_goal(&self, frame: &[f32]) -> Vec<f32> {
        let e = &self.exact;
        let mut v = vec![NOWHERE; 6];
        for (k, (from, to, floor)) in self.setup.phases().into_iter().enumerate().take(2) {
            if let Some(f) = floor {
                v[3 * k] = f.log10() as f32;
                v[3 * k + 1] = if from == 0 { frame[X_LO] } else { x_of(e, self.axis, from) as f32 };
                v[3 * k + 2] = if to >= self.setup.steps { frame[X_HI] } else { x_of(e, self.axis, to) as f32 };
            }
        }
        v
    }

    /// Bring the picture `dt` seconds nearer to what it shows. (`snap`: there at once.)
    fn follow(&mut self, dt: f32, snap: bool) {
        let frame = self.goal();
        let (lo, hi) = (frame[X_LO] as f64, frame[X_HI] as f64);
        let main = curve_points(&self.exact, self.axis, lo, hi, false);
        let avg = if self.setup.averaging { curve_points(&self.exact, self.axis, lo, hi, true) } else { (main.0.clone(), vec![false; CURVE_N]) };
        let mut curves: Vec<(Vec<f32>, Vec<bool>)> = vec![main, avg];
        for e in &self.compared {
            curves.push(curve_points(e, self.axis, lo, hi, false));
        }
        let mut strips = vec![strip_points(&self.exact, self.axis, lo, hi)];
        for e in &self.compared {
            strips.push(strip_points(e, self.axis, lo, hi));
        }
        let floors = self.floors_goal(&frame);
        let dots = self.dots_goal();
        let rings = self.rings_goal();
        let look = &mut self.look;
        look.masks = curves.iter().map(|c| c.1.clone()).chain(strips.iter().map(|c| c.1.clone())).collect();
        look.curves.resize_with(curves.len(), Glide::default);
        look.strips.resize_with(strips.len(), Glide::default);
        if snap {
            look.frame.snap(&frame);
            for (g, c) in look.curves.iter_mut().zip(&curves) {
                g.snap(&c.0);
            }
            for (g, c) in look.strips.iter_mut().zip(&strips) {
                g.snap(&c.0);
            }
            look.floors.snap(&floors);
            look.dots.snap(&dots);
            look.rings.snap(&rings);
            look.moving = false;
            return;
        }
        let mut moving = look.frame.follow(&frame, dt, false);
        for (g, c) in look.curves.iter_mut().zip(&curves) {
            moving |= g.follow(&c.0, dt, false);
        }
        for (g, c) in look.strips.iter_mut().zip(&strips) {
            moving |= g.follow(&c.0, dt, false);
        }
        moving |= look.floors.follow(&floors, dt, false);
        // (the runs follow their steps through the glide too: a step that turns them
        // round, across the valley, is then a curve and not a corner)
        moving |= look.dots.follow(&dots, dt, false);
        moving |= look.rings.follow(&rings, dt, false);
        look.moving = moving;
    }
}

// ----- drawing ---------------------------------------------------------------------------------

// Lines whose number of points changes, each in a batch of its own.
const CURVE: u8 = plot::OWN;
/// (this run's weights, its average, then up to `MOST_COMPARE` comparisons)
const STRIP: u8 = CURVE + 2 + MOST_COMPARE as u8;
const NOW: u8 = STRIP + 1 + MOST_COMPARE as u8;
const RING: u8 = NOW + 1;
const FLOOR: u8 = RING + 2;
const LIMIT_LINE: u8 = FLOOR + 1;

/// The point batches: the runs, their averages, the chart's dots for each.
fn dot_batch<'b>(f: &'b mut Frame<'_>, k: usize) -> &'b mut crate::gfx::draw::PointBatch {
    f.dl.points_with(PointKind::Overlay, PointStyle { outline_px: 0.0, size_scale: 1.0 + 1e-4 * (k + 1) as f32, ..Default::default() })
}

/// Every batch the scene draws in, made in the same order in every frame, so that one
/// that is empty in a frame does not move the later ones up a place (see `eidos motion`).
fn hold_batches(f: &mut Frame) {
    let corner = f.hud(Vec2::ZERO);
    for kind in [AXES, MARKS, plot::BUTTONS].into_iter().chain(CURVE..=LIMIT_LINE) {
        plot::apart(f, kind, false).segment(corner, corner, 1.0, Color::CLEAR);
    }
    for kind in [FLOOR, LIMIT_LINE, MARKS] {
        plot::apart(f, kind, true).segment(corner, corner, 1.0, Color::CLEAR);
    }
    for k in 0..4 {
        dot_batch(f, k);
    }
}

/// Dots in screen space, in batch k.
fn dots_px(f: &mut Frame, k: usize, pts: &[Vec2], size: f32, color: Color) {
    let at: Vec<Vec3> = pts.iter().filter(|p| p.is_finite()).map(|p| f.hud(*p)).collect();
    let b = dot_batch(f, k);
    for p in at {
        b.push(p, size, color);
    }
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

/// A power of ten as an axis writes it: 1, 10, 100, 1 000 across; 10⁻² up.
fn power_name(x: f64, up: bool) -> String {
    let v = 10f64.powf(x);
    if up {
        let d = x.round() as i32;
        if (x - d as f64).abs() < 1e-6 {
            return match d {
                0 => "1".to_string(),
                1 => "10".to_string(),
                _ => format!("$10^{{{d}}}$"),
            };
        }
        return sig(v);
    }
    if v >= 1.0 {
        count(v.round() as usize)
    } else {
        sig(v)
    }
}

impl NoiseFloor {
    fn bowl_plot(lay: &Layout) -> Plot {
        Plot::new(lay.bowl, [-BOWL[0], BOWL[0]], [-BOWL[1], BOWL[1]])
    }

    fn chart_plot(&self, lay: &Layout) -> Plot {
        let g = |i: usize| self.look.frame.get(i) as f64;
        Plot::new(lay.chart, [g(X_LO), g(X_HI)], [g(Y_LO), g(Y_HI)])
    }

    fn strip_plot(&self, lay: &Layout) -> Plot {
        let g = |i: usize| self.look.frame.get(i) as f64;
        Plot::new(lay.strip, [g(X_LO), g(X_HI)], [0.0, g(STRIP_HI).max(1e-6)])
    }

    /// The buttons over the picture; true if anything changed.
    fn buttons(&mut self, f: &mut Frame, lay: &Layout) -> bool {
        let size = lay.small;
        let mut changed = false;
        let busy = self.busy();
        let play = if busy {
            "❚❚ pause"
        } else if self.swarm.done() {
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
        for (k, (name, axis)) in [("steps", Axis::Steps), ("examples", Axis::Examples)].into_iter().enumerate() {
            if plot::button(f, lay.buttons[2 + k], name, size, self.axis == axis) && self.axis != axis {
                self.axis = axis;
                changed = true;
            }
        }
        if plot::button(f, lay.buttons[4], if lay.narrow { "keep" } else { "keep this run" }, size, false) {
            self.keep();
            changed = true;
        }
        changed
    }

    /// Keep this run's line, faint, for comparison (the oldest kept goes if there are many).
    fn keep(&mut self) {
        let s = self.setup;
        if !self.compare.contains(&s) {
            if self.compare.len() >= MOST_COMPARE {
                self.compare.remove(0);
            }
            self.compare.push(s);
        }
    }

    /// The start, dragged on the bowl.
    fn drag(&mut self, f: &mut Frame, lay: &Layout) {
        let bowl = Self::bowl_plot(lay);
        let s = self.setup.start;
        if let Some((x, y)) = bowl.handle(f, 1, s[0], s[1], pal::WHITE) {
            if x.is_finite() && y.is_finite() {
                self.setup.start = [x, y];
                self.dragging = true;
            }
        }
    }

    fn draw_bowl(&self, f: &mut Frame, lay: &Layout) {
        let p = Self::bowl_plot(lay);
        if p.tiny() {
            return;
        }
        let s = &self.setup;
        let (min, max) = p.rect;
        p.backing(f, 8.0, 8.0, 8.0, 8.0);
        let title = if lay.narrow {
            format!("{RUNS} runs of SGD from one start")
        } else {
            format!("{RUNS} runs of SGD on one bowl, from one start, each with its own mini-batches")
        };
        f.text_px(lay.bowl_head + Vec2::new(0.0, lay.text), title, lay.text, pal::TEXT, Align2::LEFT_BOTTOM);
        // the bowl: lines of equal loss
        for level in [0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0] {
            let (a, b) = ((2.0 * level / LAMBDA[0]).sqrt(), (2.0 * level / LAMBDA[1]).sqrt());
            p.path(f, AXES, (0..=72).map(|k| (k as f64 / 72.0 * std::f64::consts::TAU).sin_cos()).map(|(sn, cs)| (a * cs, b * sn)), 1.0, pal::DIM.a(0.22));
        }
        // its two directions and their curvatures
        p.path(f, AXES, [(-BOWL[0], 0.0), (BOWL[0], 0.0)], 1.0, CURVATURE.a(0.3));
        p.path(f, AXES, [(0.0, -BOWL[1]), (0.0, BOWL[1])], 1.0, CURVATURE.a(0.3));
        let names = if lay.narrow {
            [r"$\lambda_2 = 0.25$", r"$\lambda_1 = 1$"]
        } else {
            [r"flat: curvature $\lambda_2 = 0.25$", r"sharp: curvature $\lambda_1 = 1$"]
        };
        f.text_px(Vec2::new(max.x - 6.0, p.px_y(0.0) - 3.0), names[0], lay.small, CURVATURE, Align2::RIGHT_BOTTOM);
        f.text_px(Vec2::new(p.px_x(0.0) + 6.0, min.y + 4.0), names[1], lay.small, CURVATURE, Align2::LEFT_TOP);
        // the bottom
        let o = p.at(0.0, 0.0);
        for d in [Vec2::new(4.0, 4.0), Vec2::new(4.0, -4.0)] {
            plot::line_of(f, AXES, o - d, o + d, 1.2, pal::DIM);
        }

        // the runs, and their averages; the ellipses the formula gives
        let color = s.color();
        let dots = &self.look.dots.shown;
        let size = if lay.narrow { 2.6 } else { 3.4 };
        let at = |i: usize| p.at(dots[2 * i] as f64, dots[2 * i + 1] as f64);
        let inside = |q: &Vec2| plot::within(*q, p.rect, 0.0);
        if dots.len() >= 4 * RUNS {
            let runs: Vec<Vec2> = (0..RUNS).map(at).filter(inside).collect();
            dots_px(f, 0, &runs, size, color.a(if s.averaging { 0.45 } else { 0.85 }));
            if s.averaging {
                let avgs: Vec<Vec2> = (RUNS..2 * RUNS).map(at).filter(inside).collect();
                dots_px(f, 1, &avgs, size, AVERAGE.a(0.9));
            }
        }
        let rings = &self.look.rings.shown;
        if rings.len() >= 8 {
            for (k, c) in [(0usize, color), (1, AVERAGE)] {
                if k == 1 && !s.averaging {
                    continue;
                }
                let r = &rings[4 * k..4 * k + 4];
                let (cx, cy, rx, ry) = (r[0] as f64, r[1] as f64, r[2] as f64, r[3] as f64);
                if rx * p.width() as f64 / (2.0 * BOWL[0]) > 1.0 || ry * p.height() as f64 / (2.0 * BOWL[1]) > 1.0 {
                    p.path(
                        f,
                        RING + k as u8,
                        (0..=72).map(|i| (i as f64 / 72.0 * std::f64::consts::TAU).sin_cos()).map(|(sn, cs)| (cx + rx * cs, cy + ry * sn)),
                        1.6,
                        c.mix(pal::WHITE, 0.25),
                    );
                }
            }
        }
        // the step, the mean loss now, what the formula says
        let t = self.swarm.t;
        let measured = self.swarm.z.iter().map(|z| loss_at(*z)).sum::<f64>() / RUNS as f64;
        let exact = self.exact.loss.get(self.swarm.t.min(self.exact.steps())).copied().unwrap_or(f64::NAN);
        let mut lines = vec![format!("step {} of {}", count(t), count(s.steps))];
        if self.swarm.diverged || self.exact.blown.is_some_and(|b| b <= self.swarm.t.max(1)) {
            lines.push(format!(r"flown off: $\eta = {}$ is over the limit {}", short(s.eta), short(s.limit())));
        } else if self.swarm.t > 0 {
            lines.push(format!("their mean loss {}, the formula's {}", sig(measured), sig(exact)));
        }
        let size = lay.small;
        for (k, line) in lines.iter().rev().enumerate() {
            f.caption_px(Vec2::new(min.x + 8.0, max.y - 6.0 - k as f32 * (size + 6.0)), line.clone(), size, pal::TEXT, Align2::LEFT_BOTTOM);
        }
        // a key to the dots and the ring
        if !lay.compact {
            let mut key: Vec<(Color, String)> = vec![(color, if lay.narrow { "a run".to_string() } else { format!("one run (of {RUNS})") })];
            key.push((color.mix(pal::WHITE, 0.25), if lay.narrow { "86 % inside, says the formula".to_string() } else { "ring: where the formula puts 86 % of the runs".to_string() }));
            if s.averaging {
                key.push((AVERAGE, format!("its average over about {} steps", count(s.window))));
            }
            let x = max.x - 8.0;
            for (k, (c, text)) in key.iter().enumerate() {
                let y = max.y - 6.0 - (key.len() - 1 - k) as f32 * (size + 6.0);
                f.caption_px(Vec2::new(x, y), text.clone(), size, *c, Align2::RIGHT_BOTTOM);
            }
        }
    }

    /// The strip of the step size over the run, above the chart.
    fn draw_strip(&mut self, f: &mut Frame, lay: &Layout) {
        let p = self.strip_plot(lay);
        if p.tiny() {
            return;
        }
        let s = self.setup;
        let (min, max) = p.rect;
        p.backing(f, 40.0, 4.0, 6.0, 6.0);
        plot::line_of(f, AXES, Vec2::new(min.x, max.y), max, 1.0, pal::DIM.a(0.8));
        plot::line_of(f, AXES, min, Vec2::new(min.x, max.y), 1.0, pal::DIM.a(0.8));
        f.text_px(Vec2::new(min.x - 6.0, (min.y + max.y) * 0.5), r"$\eta$", lay.small + 1.0, pal::DIM, Align2::RIGHT_CENTER);
        let (lo, hi) = (p.x[0], p.x[1]);
        let n = self.look.strips.len();
        for (k, g) in self.look.strips.iter().enumerate().rev() {
            let mask = self.look.masks.get(self.look.curves.len() + k);
            let main = k == 0;
            let c = if main { s.color() } else { self.compare.get(k - 1).map_or(SGD, |c| c.color()).a(0.4) };
            let pts: Vec<(f64, f64)> =
                (0..CURVE_N).map(|i| if mask.is_some_and(|m| m[i]) { (across(i, lo, hi), g.get(i) as f64) } else { (f64::NAN, f64::NAN) }).collect();
            p.path(f, STRIP + k.min(n) as u8, pts, if main { 1.8 } else { 1.2 }, c);
        }
        // the limit, if it is near: its name above it, at the right
        let limit = s.limit();
        let near = limit <= p.y[1];
        if near {
            let y = p.px_y(limit);
            plot::dashed_of(f, LIMIT_LINE, Vec2::new(min.x, y), Vec2::new(max.x, y), 1.2, LIMIT);
            let name = if s.beta > 0.0 { format!(r"limit $(2 + 2\beta)/\lambda_1 = {}$", short(limit)) } else { format!(r"limit $2/\lambda_1 = {}$", short(limit)) };
            f.caption_px(Vec2::new(max.x - 4.0, y - 1.0), name, lay.small, LIMIT, Align2::RIGHT_BOTTOM);
        }
        // the step's name at the left, above its line where there is room, else under
        // it (decided by where the strip is going, and glided to)
        let label = format!(r"step size $\eta = {}$", short(s.eta));
        // (kept to the strip while its range glides to one that holds the line)
        let y = p.px_y(s.eta).max(min.y).min(max.y);
        let goal = self.goal();
        let to = Plot::new(lay.strip, [p.x[0], p.x[1]], [0.0, goal[STRIP_HI].max(1e-6) as f64]);
        let room_above = to.px_y(s.eta) - if limit <= to.y[1] { to.px_y(limit) } else { min.y - 6.0 };
        let half = 0.5 * lay.small + 3.0;
        let want = if room_above >= lay.small + 4.0 { -half } else { half };
        let ease = 1.0 - (-f.real_dt.max(0.0) / 0.08).exp();
        self.strip_label = if (self.strip_label - want).abs() > 0.5 && self.strip_label.abs() > 0.0 { self.strip_label + (want - self.strip_label) * ease } else { want };
        self.labels_moving |= self.strip_label != want;
        f.caption_px(Vec2::new(min.x + 6.0, y + self.strip_label), label, lay.small, s.color(), Align2::LEFT_CENTER);
        // where the film is
        let x = x_between(&self.exact, self.axis, self.shown);
        if self.shown >= 1.0 && x >= lo && x <= hi {
            plot::line_of(f, NOW, Vec2::new(p.px_x(x), min.y), Vec2::new(p.px_x(x), max.y), 1.0, pal::TEXT.a(0.35));
        }
    }

    fn draw_chart(&mut self, f: &mut Frame, lay: &Layout) {
        let p = self.chart_plot(lay);
        if p.tiny() {
            return;
        }
        let s = self.setup;
        let (min, max) = p.rect;
        let size = lay.small;
        p.backing(f, if lay.narrow { 40.0 } else { 46.0 }, 34.0, 8.0, 8.0);
        let title = match (self.axis, lay.narrow) {
            (Axis::Steps, false) => "the expected loss against the step: the formula's line, and the swarm's mean as dots",
            (Axis::Steps, true) => "expected loss against the step",
            (Axis::Examples, false) => "the expected loss against the examples used (B a step): the formula's line, the swarm's mean as dots",
            (Axis::Examples, true) => "expected loss against the examples used",
        };
        f.text_px(lay.chart_head, title, lay.text, pal::TEXT, Align2::LEFT_BOTTOM);
        // the axes: powers of ten
        plot::line_of(f, AXES, Vec2::new(min.x, max.y), max, 1.0, pal::DIM.a(0.8));
        plot::line_of(f, AXES, min, Vec2::new(min.x, max.y), 1.0, pal::DIM.a(0.8));
        let goal = self.goal();
        for (x, named) in log_ticks(goal[X_LO] as f64, goal[X_HI] as f64) {
            if x < p.x[0] - 1e-6 || x > p.x[1] + 1e-6 {
                continue;
            }
            let at = p.px_x(x);
            plot::line_of(f, AXES, Vec2::new(at, max.y), Vec2::new(at, max.y + if named { 4.0 } else { 2.0 }), 1.0, pal::DIM.a(0.8));
            if named {
                f.text_px(Vec2::new(at, max.y + 6.0), power_name(x, false), size, pal::DIM, Align2::CENTER_TOP);
            }
        }
        let across_name = match self.axis {
            Axis::Steps => "step",
            Axis::Examples => "examples used",
        };
        f.text_px(Vec2::new(max.x, max.y + 8.0 + size), across_name, size, pal::DIM, Align2::RIGHT_TOP);
        for (y, named) in log_ticks(goal[Y_LO] as f64, goal[Y_HI] as f64) {
            if y < p.y[0] - 1e-6 || y > p.y[1] + 1e-6 || !named {
                continue;
            }
            let at = p.px_y(y);
            plot::line_of(f, AXES, Vec2::new(min.x - 4.0, at), Vec2::new(min.x, at), 1.0, pal::DIM.a(0.8));
            plot::line_of(f, AXES, Vec2::new(min.x, at), Vec2::new(max.x, at), 1.0, pal::DIM.a(0.1));
            f.text_px(Vec2::new(min.x - 6.0, at), power_name(y, true), size, pal::DIM, Align2::RIGHT_CENTER);
        }
        f.text_px(Vec2::new(min.x + 6.0, min.y + 4.0), "loss", size, pal::DIM, Align2::LEFT_TOP);

        // the floors
        let fl = &self.look.floors.shown;
        let mut floors: Vec<ChartLabel> = Vec::new();
        for (k, (_, _, floor)) in s.phases().into_iter().enumerate().take(2) {
            let (Some(floor), Some(seg)) = (floor, fl.get(3 * k..3 * k + 3)) else { continue };
            let (y, a, b) = (seg[0] as f64, seg[1] as f64, seg[2] as f64);
            if !(y > p.y[0] && y < p.y[1]) {
                continue;
            }
            let (xa, xb) = (p.px_x(a.max(p.x[0])), p.px_x(b.min(p.x[1])));
            plot::dashed_of(f, FLOOR, Vec2::new(xa, p.px_y(y)), Vec2::new(xb, p.px_y(y)), 1.4, NOISE);
            let name = if k == 0 && s.dropping() { format!("floor before the drop {}", sig(floor)) } else { format!("floor {}", sig(floor)) };
            let name = if k == 1 { format!("floor after {}", sig(floor)) } else { name };
            floors.push(ChartLabel { at: Vec2::new(xa + 6.0, p.px_y(y) + 3.0), align: Align2::LEFT_TOP, text: name, color: NOISE });
        }
        // the averaged weights' floor (a constant step's)
        if let (true, Some(a)) = (s.averaging, self.averaged_floor()) {
            let y = a.log10();
            if y > p.y[0] && y < p.y[1] {
                let at = p.px_y(y);
                plot::dashed_of(f, FLOOR, Vec2::new(min.x, at), Vec2::new(max.x, at), 1.2, AVERAGE.a(0.7));
                floors.push(ChartLabel { at: Vec2::new(min.x + 6.0, at + 3.0), align: Align2::LEFT_TOP, text: format!("the averaged weights' floor {}", sig(a)), color: AVERAGE });
            }
        }

        // the comparisons, faint; this run's average; this run
        let (lo, hi) = (p.x[0], p.x[1]);
        let mut ends: Vec<ChartLabel> = Vec::new();
        for k in (0..self.look.curves.len()).rev() {
            let Some(mask) = self.look.masks.get(k) else { continue };
            let g = &self.look.curves[k];
            let (c, width, name) = match k {
                0 => (s.color(), 2.2, s.label()),
                1 => (AVERAGE, 2.0, "the averaged weights".to_string()),
                _ => {
                    let Some(c) = self.compare.get(k - 2) else { continue };
                    (c.color().a(0.42), 1.4, c.label())
                }
            };
            if k == 1 && !s.averaging {
                continue;
            }
            let pts: Vec<(f64, f64)> = (0..CURVE_N).map(|i| if mask[i] { (across(i, lo, hi), g.get(i) as f64) } else { (f64::NAN, f64::NAN) }).collect();
            p.path(f, CURVE + k as u8, pts, width, c);
            if let Some(last) = (0..CURVE_N).rev().find(|&i| mask[i]) {
                let at = p.at(across(last, lo, hi), (g.get(last) as f64).max(p.y[0]).min(p.y[1]));
                ends.push(ChartLabel { at: Vec2::new(at.x.min(max.x - 4.0), at.y - 4.0), align: Align2::RIGHT_BOTTOM, text: name, color: if k >= 2 { c.a(0.8) } else { c } });
            }
        }

        // the swarm's mean, as dots on its line
        let shown = self.shown;
        let dot = if lay.narrow { 4.0 } else { 5.0 };
        let mut on: Vec<Vec2> = Vec::new();
        let mut on_avg: Vec<Vec2> = Vec::new();
        for m in self.swarm.marks.iter().filter(|m| (m.step as f64) <= shown + 1e-9) {
            let x = x_of(&self.exact, self.axis, m.step);
            for (v, list) in [(m.loss, &mut on), (m.avg, &mut on_avg)] {
                if v > 0.0 {
                    let at = p.at(x, v.log10());
                    if plot::within(at, p.rect, 0.5) {
                        list.push(at);
                    }
                }
            }
        }
        dots_px(f, 2, &on, dot, s.color().mix(pal::WHITE, 0.15));
        if s.averaging {
            dots_px(f, 3, &on_avg, dot, AVERAGE);
        }
        // where the film is
        let x = x_between(&self.exact, self.axis, shown);
        if shown >= 1.0 && x >= lo && x <= hi {
            plot::line_of(f, NOW, Vec2::new(p.px_x(x), min.y), Vec2::new(p.px_x(x), max.y), 1.0, pal::TEXT.a(0.35));
        }

        // the names: the floors' at their left ends, the runs' at their right ends
        // (this run's first), none over another
        ends.reverse();
        floors.extend(ends);
        self.labels_moving |= place(f, floors, size, p.rect, &mut self.placed);
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
/// is still clear, and glides to a new one (`memo` remembers them; true while one glides).
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

/// The averaged weights' colour.
const AVERAGE: Color = pal::TEXT;

impl Scene for NoiseFloor {
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
        Some("white dot: drag the start    buttons under the bowl: click")
    }

    fn ui(&mut self, ui: &mut egui::Ui, _: &Gfx) {
        ui::note(
            ui,
            r"A mini-batch's gradient is the true gradient plus noise (scene minibatch). Here 400 runs of SGD go down one bowl from one start, each with its own mini-batches. Far up the walls the bowl's pull outweighs the noise and they slide down together; near the bottom only the noise is left, and they spread into a cloud that a constant step never shrinks. Along a direction of curvature $\lambda$, with one example's gradient noise of variance $\sigma^2$ there and a batch of $B$, the expected squared distance $z^2$ from the bottom obeys exactly",
        );
        ui::equation(ui, r"\mathbb{E}\,z_{t+1}^2 = (1 - \eta\lambda)^2\,\mathbb{E}\,z_t^2 + \frac{\eta^2\sigma^2}{B}");
        ui::note(ui, r"The first part forgets the start geometrically; the second is added at every step. Where the two balance, the loss $\tfrac12\lambda z^2$ in that direction has its floor:");
        ui::equation(ui, r"\frac{\eta\,\sigma^2}{2B\,(2 - \eta\lambda)} \approx \frac{\eta\,\sigma^2}{4B}");
        ui::note(
            ui,
            r"The rule is exact here because the noise is the same wherever the weights are. A network's is not: it changes as the weights do. Where a model can fit every example exactly, every example's gradient is zero at the bottom, the noise vanishes there, and a small enough constant step does converge (Ma, Bassily and Belkin, 2018, proved this for least squares).",
        );

        ui::section(ui, "Things to try");
        for (k, t) in TRIALS.iter().enumerate() {
            if ui::experiment(ui, t.button, t.caption) {
                self.set_up(k);
            }
        }

        ui::section(ui, "The run");
        let s = &mut self.setup;
        let mut eta = s.eta as f32;
        if ui::slider_log(ui, r"step size $\eta$", &mut eta, LEAST_ETA as f32..=MOST_ETA as f32) {
            s.eta = eta as f64;
        }
        ui::note(ui, r"What the gradient is multiplied by in a step (the learning rate). Over $2/\lambda_1 = 2$ (with momentum $\beta$, over $(2 + 2\beta)/\lambda_1$) the sharp direction flies off.");
        ui::slider_int_log(ui, "batch size $B$", &mut s.batch, 1..=MOST_BATCH);
        let mut beta = s.beta as f32;
        if ui::slider(ui, r"momentum $\beta$", &mut beta, 0.0..=MOST_BETA as f32) {
            s.beta = beta as f64;
        }
        ui::note(ui, r"Heavy ball: $v \leftarrow \beta v + g$, then $\theta \leftarrow \theta - \eta v$. A gradient goes on acting for about $1/(1 - \beta)$ steps.");
        let mut schedule = match s.schedule {
            Schedule::Constant => 0,
            Schedule::Drop => 1,
            Schedule::Cosine => 2,
        };
        if ui::choice(ui, &["constant", "a drop", "cosine"], &mut schedule) {
            s.schedule = [Schedule::Constant, Schedule::Drop, Schedule::Cosine][schedule.min(2)];
        }
        match s.schedule {
            Schedule::Drop => {
                let most = s.steps;
                ui::slider_int(ui, "the drop at step", &mut s.drop_at, 1..=most);
                let mut what = s.drop_batch as usize;
                if ui::choice(ui, &[r"step to $\eta/10$", r"batch to $10B$"], &mut what) {
                    s.drop_batch = what == 1;
                }
            }
            Schedule::Cosine => ui::note(ui, r"$\eta_t = \tfrac12\eta\,(1 + \cos(\pi t/T))$: from $\eta$ down to 0 over the $T$ steps of the run."),
            Schedule::Constant => {}
        }
        ui::note(
            ui,
            "On this bowl a decay of the step helps for one exact reason: the floor. Why it helps real networks is not settled. One proposal, a conjecture about the pre-training of language models (Wen and co-authors, 2024), is that their loss is a long valley along which a large step makes progress while it rattles from side to side, and that decaying the step stops the rattling and shows the progress.",
        );
        ui::checkbox(ui, &mut s.averaging, "show the averaged weights");
        if s.averaging {
            ui::slider_int_log(ui, "averaged over about $W$ steps", &mut s.window, 2..=MOST_WINDOW);
            ui::note(ui, r"Each run keeps $\bar\theta \leftarrow \gamma\bar\theta + (1 - \gamma)\theta$ with $\gamma = 1 - 1/W$: a moving average of its weights.");
        }
        let mut noise = (s.noise == Noise::Curved) as usize;
        if ui::choice(ui, &["noise even", "noise as curvature"], &mut noise) {
            s.noise = if noise == 1 { Noise::Curved } else { Noise::Even };
        }
        ui::note(
            ui,
            r"One example's gradient noise: variance 1 along both directions, or in proportion to the curvature ($0.4$ and $1.6$). With a short step, the first gives a cloud shaped like the bowl's lines of equal loss, the second a nearly round one. The second is what the noisy quadratic model of Zhang and co-authors (2019) assumes, since near a good fit the gradients' covariance resembles the Hessian; they report that this held closely for a convolutional network and moderately well for a transformer.",
        );
        ui::slider_int_log(ui, "steps in the run $T$", &mut s.steps, LEAST_STEPS..=MOST_STEPS);
        ui.horizontal_wrapped(|ui| {
            if ui.button("new mini-batches").clicked() {
                self.seed = self.seed.wrapping_add(1);
            }
            if ui.button("keep this run").clicked() {
                self.keep();
            }
            if !self.compare.is_empty() && ui.button("forget the kept runs").clicked() {
                self.compare.clear();
            }
        });
        self.refresh();

        ui::section(ui, "Reading the picture");
        ui::key(
            ui,
            &[
                (SGD, "cyan dots on the bowl: the runs of SGD; on the chart, their mean loss at a step"),
                (SGD.mix(pal::WHITE, 0.25), "cyan ring and line: what the formula says, exactly: the ellipse that holds 86 % of the runs on average, and the expected loss"),
                (MOMENTUM, "pink: the same, with momentum"),
                (AVERAGE, "white: the averaged weights, when shown"),
                (NOISE, "red dashed: the floor"),
                (CURVATURE, r"green: the bowl's two directions and their curvatures $\lambda$"),
                (LIMIT, "yellow: the longest step the sharp direction bears, when it is near"),
                (SGD.a(0.42), "faint lines: other runs, kept for comparison"),
            ],
        );
        ui::section(ui, "Words");
        ui::term(ui, "SGD:", "stochastic gradient descent: gradient descent with the gradient of a mini-batch in place of the whole data's.");
        ui::term(ui, r"Mini-batch, batch size $B$:", r"the examples whose gradients a step averages. Here every example is fresh, so the noise has variance $\sigma^2/B$ exactly. Drawn without putting back from a fixed set of $N$, it would be smaller by the factor $(N - B)/(N - 1)$: none at all for the whole set.");
        ui::term(ui, "Curvature:", r"how fast the slope grows along one of the bowl's own directions: an eigenvalue $\lambda$ of its Hessian.");
        ui::term(ui, "Expected loss:", "the mean over all the ways the mini-batches could come out; the dots estimate it from 400 of them.");
        ui::term(ui, "Noise floor:", "the expected loss a constant step settles at.");
        ui::term(ui, "Schedule:", "how the step changes during a run: here constant, a drop by 10 at a chosen step, or cosine.");
        ui::term(ui, "Averaged weights:", "a moving average of a run's weights over its recent steps, used in place of the weights themselves.");
        ui::term(ui, "Examples used:", r"the gradients of single examples computed so far: $B$ a step.");
        ui::term(ui, "Standard error:", r"how far the mean of 400 runs may stray from the expected loss by chance; the read-outs give $\pm$ two of them.");

        ui::section(ui, "Numbers");
        self.numbers(ui);
        ui::see_also(
            ui,
            &[
                ("stepsize", r"the factor $1 - \eta\lambda$, without noise"),
                ("minibatch", "a mini-batch gradient: signal and noise"),
                ("noisescale", "how large a batch is worth it"),
                ("warmup", "schedules on a real network"),
            ],
        );
    }

    fn frame(&mut self, f: &mut Frame) {
        hold_batches(f);
        let lay = Layout::of(f.view_size(), f.top_inset);
        // (a canvas squeezed to a sliver is one the controls have opened over, on a phone)
        let tiny = lay.view.x < 160.0 || lay.view.y < 120.0;
        self.dragging = false;
        if !tiny {
            if !lay.compact && self.buttons(f, &lay) {
                self.refresh();
            }
            self.drag(f, &lay);
        }
        if self.dragging {
            self.film = Film { clock: 0.0, playing: false, waiting: true };
        }
        self.refresh();
        let budget = clock::frame_budget(f.real_dt, 0.004);
        self.advance(f.dt, &budget);
        let snap = self.canvas != Some(lay.view) && !tiny;
        if snap {
            self.canvas = Some(lay.view);
        }
        self.follow(f.real_dt, snap);
        if tiny {
            return;
        }
        self.labels_moving = false;
        self.draw_bowl(f, &lay);
        if !lay.compact {
            self.draw_strip(f, &lay);
            self.draw_chart(f, &lay);
        }
    }
}

impl NoiseFloor {
    /// The averaged weights' floor with a constant step (None for a schedule, or where
    /// the step does not settle): the stationary state of the moments.
    fn averaged_floor(&self) -> Option<f64> {
        let s = &self.setup;
        if s.schedule != Schedule::Constant {
            return None;
        }
        let c = s.noise.per_example();
        (0..2).map(|i| stationary(LAMBDA[i], s.eta, s.beta, s.gamma(), c[i] / s.batch as f64).map(|v| v.1)).sum()
    }

    /// The read-outs.
    fn numbers(&self, ui: &mut egui::Ui) {
        let s = &self.setup;
        let e = &self.exact;
        ui::readout(ui, "loss at the start", sig(s.start_loss()));
        match s.schedule {
            Schedule::Cosine => ui::readout(ui, "floor", "none: the step falls to 0"),
            _ => {
                for (k, (from, _, floor)) in s.phases().into_iter().enumerate() {
                    let name = match (s.dropping(), k) {
                        (true, 0) => "floor before the drop",
                        (true, _) => "floor after the drop",
                        _ => "floor (the formula)",
                    };
                    match floor {
                        Some(fl) => {
                            ui::readout_colored(ui, name, sig(fl), NOISE);
                            if let Some(t) = e.settled_from(fl, from) {
                                if k + 1 == s.phases().len() {
                                    ui::readout(ui, "  within a tenth of it from step", count(t));
                                }
                            } else if k + 1 == s.phases().len() {
                                ui::readout(ui, "  within a tenth of it", "not in this run");
                            }
                        }
                        None => ui::readout_colored(ui, name, format!(r"none: $\eta\lambda_1 \ge {}$", short(2.0 + 2.0 * s.beta)), LIMIT),
                    }
                }
                if !s.dropping() {
                    ui::readout(ui, r"  small-step form $\eta(\sigma_1^2 + \sigma_2^2)/(4B(1 - \beta))$", sig(small_step_floor(s.eta, s.batch as f64, s.beta, s.noise)));
                }
            }
        }
        if s.beta > 0.0 && !s.dropping() && s.schedule == Schedule::Constant {
            let plain = floor_of(s.eta, s.batch as f64, 0.0, s.noise);
            let matched = floor_of(s.eta * (1.0 - s.beta), s.batch as f64, s.beta, s.noise);
            ui::readout(ui, "plain SGD at the same step: floor", plain.map_or("none".to_string(), sig));
            ui::readout(ui, r"this momentum at $\eta(1 - \beta)$: floor", matched.map_or("none".to_string(), sig));
        }
        if let (true, Some(a), Some(w)) = (s.averaging, self.averaged_floor(), s.floor()) {
            ui::readout_colored(ui, "averaged weights' floor", sig(a), AVERAGE);
            ui::readout(ui, "  lower than the weights' by", format!("{} times", ratio(w / a)));
        }
        if let Some((m, se)) = self.swarm.plateau(false) {
            let from = self.swarm.tail_from + 1;
            ui::readout(ui, &format!("the swarm's mean, steps {}–{}", count(from), count(s.steps)), format!(r"${} \pm {}$", sig(m), sig(2.0 * se)));
            ui::readout(ui, "  the formula's, over the same steps", sig(e.mean_over(from, s.steps, false)));
            if s.averaging {
                if let Some((m, se)) = self.swarm.plateau(true) {
                    ui::readout_colored(ui, "  their averages'", format!(r"${} \pm {}$", sig(m), sig(2.0 * se)), AVERAGE);
                    ui::readout(ui, "  the formula's for the averages", sig(e.mean_over(from, s.steps, true)));
                }
            }
        }
        if self.swarm.t > 0 && !self.swarm.diverged {
            let inside = self.swarm.inside(e, false);
            if inside.is_finite() {
                ui::readout(ui, "runs inside the ring now", format!("{} % (the formula: 86 %)", num(100.0 * inside, 0)));
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn close(a: f64, b: f64, rel: f64) -> bool {
        (a - b).abs() <= rel * b.abs().max(1e-300)
    }

    /// The swarm of a setup, run to its end.
    fn swarm(s: Setup, seed: u64) -> Swarm {
        let mut w = Swarm::new(s, seed);
        while !w.done() {
            w.step();
        }
        w
    }

    /// An experiment as its button sets it up, with its film run to the end.
    fn run(k: usize) -> NoiseFloor {
        let mut s = NoiseFloor::create();
        s.set_up(k);
        while !s.swarm.done() {
            s.swarm.step();
        }
        s.shown = s.setup.steps as f64;
        s
    }

    /// Plain SGD: the recursion of the moments is NQM's Eq. 4 per direction, the
    /// transient and the floor exactly: E ℓ(t) = a^t ℓ(0) + (1 − a^t) F, a = (1 − ηλ)².
    #[test]
    fn the_moments_are_the_closed_form() {
        for (eta, batch, noise) in [(0.1, 4usize, Noise::Even), (0.05, 1, Noise::Curved), (1.6, 64, Noise::Even), (1.9, 2, Noise::Curved)] {
            let s = Setup { eta, batch, noise, ..Setup::base() };
            let e = Exact::of(&s, 500);
            let c = noise.per_example();
            for t in [0usize, 1, 7, 60, 500] {
                let want: f64 = (0..2)
                    .map(|i| {
                        let a = (1.0 - eta * LAMBDA[i]).powi(2);
                        let start = 0.5 * LAMBDA[i] * s.start[i] * s.start[i];
                        let floor = floor_one(eta, batch as f64, 0.0, LAMBDA[i], c[i]).unwrap();
                        a.powi(t as i32) * start + (1.0 - a.powi(t as i32)) * floor
                    })
                    .sum();
                assert!(close(e.loss[t], want, 1e-10), "{s:?} t {t}: {} against {want}", e.loss[t]);
            }
            // and the mean is (1 − ηλ)^t times the start
            for i in 0..2 {
                assert!(close(e.mean[60][i], (1.0 - eta * LAMBDA[i]).powi(60) * s.start[i], 1e-9) || e.mean[60][i].abs() < 1e-12);
            }
        }
    }

    /// The dossier's floors (with momentum, NQM Theorem 1's steady state; the average,
    /// Theorem 2) are the stationary state of the recursion, solved exactly; and the
    /// recursion run long goes there.
    #[test]
    fn the_floors_are_the_stationary_state() {
        for (eta, beta, gamma, lambda) in [(0.1, 0.0, 0.99, 1.0), (0.1, 0.0, 0.9, 0.25), (0.1, 0.9, 0.99, 1.0), (0.01, 0.9, 0.5, 0.25), (1.6, 0.0, 0.95, 1.0), (3.5, 0.9, 0.99, 1.0)] {
            let q = 0.25;
            let (w, a) = stationary(lambda, eta, beta, gamma, q).unwrap();
            let f = floor_one(eta, 1.0, beta, lambda, q).unwrap();
            assert!(close(w, f, 1e-9), "{eta} {beta} {lambda}: {w} against {f}");
            if beta == 0.0 {
                let want = f * averaging_factor(eta, lambda, gamma);
                assert!(close(a, want, 1e-9), "{eta} {gamma} {lambda}: {a} against {want}");
            }
            // the moments themselves get there
            let mut m = Moments::at(1.0);
            for _ in 0..20_000 {
                m.step(lambda, eta, beta, gamma, q);
            }
            assert!(close(0.5 * lambda * m.second(0), w, 1e-6) && close(0.5 * lambda * m.second(2), a, 1e-6), "{eta} {beta} {gamma} {lambda}");
        }
        // beyond the limit there is no floor, and at it neither
        assert!(floor_one(2.0, 1.0, 0.0, 1.0, 1.0).is_none() && stationary(1.0, 2.0, 0.0, 0.5, 1.0).is_none());
        assert!(floor_one(3.79, 1.0, 0.9, 1.0, 1.0).is_some() && floor_one(3.8, 1.0, 0.9, 1.0, 1.0).is_none());
        // the small-step form is the first-order limit
        let s = Setup::base();
        let exact = floor_of(1e-4, 4.0, 0.0, s.noise).unwrap();
        assert!(close(exact, small_step_floor(1e-4, 4.0, 0.0, s.noise), 1e-4));
        // doubling the batch halves the floor exactly; halving the step, a little more than halves it
        let f = |eta: f64, b: f64| floor_of(eta, b, 0.0, Noise::Even).unwrap();
        assert!(close(f(0.1, 8.0), 0.5 * f(0.1, 4.0), 1e-12));
        assert!(f(0.05, 4.0) < 0.5 * f(0.1, 4.0));
    }

    /// The swarm's mean loss lies within sampling error of the exact curve at every
    /// noted step, for every kind of run (four standard errors: 40-odd steps each).
    #[test]
    fn the_swarm_follows_the_exact_curve() {
        let base = Setup::base();
        let setups = [
            base,
            Setup { eta: 0.05, ..base },
            Setup { beta: 0.9, ..base },
            Setup { beta: 0.9, eta: 0.01, ..base },
            Setup { schedule: Schedule::Drop, ..base },
            Setup { schedule: Schedule::Drop, drop_batch: true, ..base },
            Setup { schedule: Schedule::Cosine, ..base },
            Setup { averaging: true, ..base },
            Setup { noise: Noise::Curved, eta: 1.6, batch: 64, steps: 125, ..base },
            Setup { start: [0.0, 0.0], ..base },
        ];
        for (n, s) in setups.into_iter().enumerate() {
            let w = swarm(s, 11 + n as u64);
            let e = Exact::of(&s, s.steps);
            assert_eq!(w.marks.len(), checkpoints(s.steps).len());
            let mut worst = 0.0f64;
            for m in &w.marks {
                for (got, se, want) in [(m.loss, m.se, e.loss[m.step]), (m.avg, m.avg_se, e.avg[m.step])] {
                    let z = (got - want).abs() / se.max(1e-300);
                    worst = worst.max(z);
                    assert!(z < 4.0, "{s:?} at step {}: {got} ± {se} against {want}", m.step);
                }
            }
            assert!(worst > 0.3, "{s:?}: suspiciously close ({worst})");
            // and over the plateau, time-averaged
            let (m, se) = w.plateau(false).unwrap();
            let want = e.mean_over(w.tail_from + 1, s.steps, false);
            assert!((m - want).abs() < 4.0 * se, "{s:?}: plateau {m} ± {se} against {want}");
        }
    }

    /// The cloud is Gaussian: on the plateau, over many steps, the share of runs inside
    /// the ring of two standard deviations is 1 − e⁻² = 0.865.
    #[test]
    fn the_ring_holds_what_the_formula_says() {
        for s in [Setup::base(), Setup { beta: 0.9, ..Setup::base() }, Setup { averaging: true, ..Setup::base() }] {
            let mut w = Swarm::new(s, 5);
            let e = Exact::of(&s, s.steps);
            let (mut sum, mut sum_avg, mut k) = (0.0, 0.0, 0);
            while !w.done() {
                w.step();
                if w.t > 400 && w.t.is_multiple_of(40) {
                    sum += w.inside(&e, false);
                    sum_avg += w.inside(&e, true);
                    k += 1;
                }
            }
            // (40 steps apart the clouds are nearly independent: k·RUNS draws)
            let se = (INSIDE * (1.0 - INSIDE) / (k * RUNS) as f64).sqrt();
            let (share, share_avg) = (sum / k as f64, sum_avg / k as f64);
            assert!((share - INSIDE).abs() < 4.0 * se, "{s:?}: {share}");
            // (the averages move slowly: their samples are not independent; a looser check)
            assert!((share_avg - INSIDE).abs() < 0.03, "{s:?}: {share_avg}");
        }
        assert!(close(INSIDE, 1.0 - (-2.0f64).exp(), 1e-15));
    }

    /// The numbers each experiment's caption says, from the scene as the button sets it up.
    #[test]
    fn the_experiments_show_what_their_captions_say() {
        let says = |k: usize, what: &str| assert!(TRIALS[k].caption.contains(what), "experiment {k} should say \"{what}\": {}", TRIALS[k].caption);
        let floor = |s: &NoiseFloor| s.setup.floor().unwrap();
        let settled = |s: &NoiseFloor| s.exact.settled_from(floor(s), 0).unwrap();
        // a constant step: the floor, how far below the start, from which step
        let s0 = run(0);
        let f0 = floor(&s0);
        says(0, &format!("the formula gives, {}", sig(f0)));
        says(0, &format!("{} times below the start", num(s0.setup.start_loss() / f0, 0)));
        says(0, &format!("from step {} on", settled(&s0)));
        let (m, se) = s0.swarm.plateau(false).unwrap();
        assert!((m - f0).abs() < 3.0 * se + 0.003 * f0, "{m} ± {se} against {f0}");
        // half the step: a little under half the floor, reached later; the crossing
        let s1 = run(1);
        let f1 = floor(&s1);
        assert!(f1 < 0.5 * f0 && f1 > 0.49 * f0);
        says(1, &format!("The floor is {}, a little under half of {}", sig(f1), sig(f0)));
        says(1, &format!("from step {} instead of {}", settled(&s1), settled(&s0)));
        let cross = (1..s1.setup.steps).find(|&t| s1.exact.loss[t] < s0.exact.loss[t]).unwrap();
        assert!((cross..=s1.setup.steps).all(|t| s1.exact.loss[t] <= s0.exact.loss[t]));
        says(1, &format!("lower until step {}, the shorter one from step {cross} on", cross - 1));
        // twice the batch: the same floor as half the step within 2 %, at the first run's pace
        let s2 = run(2);
        let f2 = floor(&s2);
        assert!(f2 > f1 && f2 < 1.02 * f1, "{f2} {f1}");
        says(2, &format!("The floor is {}, within 2 % of half the step's {} (a little higher", sig(f2), sig(f1)));
        says(2, &format!("within a tenth from step {}", settled(&s2)));
        let half = &s2.compared[1];
        assert_eq!(half.setup, Setup { eta: 0.05, ..Setup::base() });
        // against examples, the two lie nearly on one line: within 4 % everywhere after the first 80 examples
        let worst = (10..=s2.setup.steps / 2).map(|t| (s2.exact.loss[t] / half.loss[2 * t] - 1.0).abs()).fold(0.0, f64::max);
        assert!(worst < 0.04, "{worst}");
        // step and batch together: floors, ratio, the factor 5 at ηλ = 1.6, times 80 reaches 2
        let s3 = run(3);
        assert_eq!(s3.axis, Axis::Examples);
        let floors: Vec<f64> = s3.compared.iter().map(|e| e.setup.floor().unwrap()).collect();
        says(3, &format!("(floors {}, {}, {})", sig(floors[0]), sig(floors[1]), sig(floors[2])));
        let f3 = floor(&s3);
        says(3, &format!("its floor is {}, {} times the first", sig(f3), short(f3 / floors[0])));
        assert!(close(2.0 / (2.0 - 1.6), 5.0, 1e-12));
        assert!(close(0.025 * 80.0 * LAMBDA[SHARP], 2.0, 1e-12) && floor_of(0.025 * 80.0, 80.0, 0.0, Noise::Even).is_none());
        // the swarm flips: its mean height changes sign every step
        assert!(s3.exact.mean[1][1] < 0.0 && s3.exact.mean[2][1] > 0.0 && s3.exact.mean[3][1] < 0.0);
        // per example the first three nearly agree (times 4 within 3 %, times 16 within
        // 15 %, all along); the last is a quarter higher or more all along
        let at_examples = |e: &Exact, x: f64| step_for(e, Axis::Examples, x).map(|t| 10f64.powf(log_loss_at(e, t, false)));
        for x in (1..=125).map(|k| 64.0 * k as f64) {
            let a = at_examples(&s3.compared[0], x).unwrap();
            let (b, c) = (at_examples(&s3.compared[1], x).unwrap(), at_examples(&s3.compared[2], x).unwrap());
            assert!((b / a - 1.0).abs() < 0.03 && (c / a - 1.0).abs() < 0.15, "{x}: {a} {b} {c}");
            assert!(at_examples(&s3.exact, x).unwrap() > 1.25 * a, "{x}");
        }
        // a drop: the new floor, how long after the drop; the batch instead
        let s4 = run(4);
        let after = floor(&s4);
        says(4, &format!("the new floor, {}", sig(after)));
        let t = s4.exact.settled_from(after, 400).unwrap();
        says(4, &format!("within a tenth of it {} steps after the drop", t - 400));
        let batch = &s4.compared[1];
        assert!(batch.setup.drop_batch);
        let fb = batch.setup.floor().unwrap();
        says(4, &format!("floor {}, reached {} steps after the drop", sig(fb), batch.settled_from(fb, 400).unwrap() - 400));
        // (the same examples: 763 × 4 against 75 × 40, within a few per cent)
        let (te, tb) = ((t - 400) as f64 * 4.0, (batch.settled_from(fb, 400).unwrap() - 400) as f64 * 40.0);
        assert!((te / tb - 1.0).abs() < 0.05, "{te} {tb}");
        // averaged weights: their floor, the ratio, when they get below
        let s5 = run(5);
        let c = Noise::Even.per_example();
        let (w, a) = (0..2).map(|i| stationary(LAMBDA[i], 0.1, 0.0, s5.setup.gamma(), c[i] / 4.0).unwrap()).fold((0.0, 0.0), |acc, v| (acc.0 + v.0, acc.1 + v.1));
        assert!(close(w, f0, 1e-9));
        says(5, &format!("Its floor is {}, {} times lower", sig(a), num(w / a, 1)));
        let below = (1..=2000).find(|&t| s5.exact.avg[t] < s5.exact.loss[t]).unwrap();
        assert!((below..=2000).all(|t| s5.exact.avg[t] < s5.exact.loss[t]));
        says(5, &format!("below them from step {below}"));
        let (ma, sea) = s5.swarm.plateau(true).unwrap();
        assert!((ma - s5.exact.mean_over(s5.swarm.tail_from + 1, 2000, true)).abs() < 4.0 * sea);
        // momentum: 9.8 times; at a tenth of the step, 3 % below
        let s6 = run(6);
        let fm = floor(&s6);
        says(6, &format!("the floor is {}, {} times plain SGD's", sig(fm), num(fm / f0, 1)));
        let matched = s6.compared[1].setup.floor().unwrap();
        assert!(close(s6.compared[1].setup.eta, 0.01, 1e-12));
        says(6, &format!("the floor is {}, {} % below plain SGD's {}", sig(matched), num(100.0 * (1.0 - matched / f0), 0), sig(f0)));
    }

    /// The scene opens as the first experiment sets it up.
    #[test]
    fn the_scene_opens_on_the_first_experiment() {
        let opened = NoiseFloor::create();
        let mut pressed = NoiseFloor::create();
        pressed.set_up(3);
        pressed.set_up(OPENING);
        assert_eq!((opened.setup, opened.compare.clone(), opened.axis, opened.seed), (pressed.setup, pressed.compare.clone(), pressed.axis, pressed.seed));
    }

    /// Edges: a step over the limit, the smallest and largest settings, a start at the
    /// bottom, a drop at the first and the last step: every number drawn stays finite.
    #[test]
    fn degenerate_settings_are_told_honestly() {
        let base = Setup::base();
        let cases = [
            Setup { eta: 2.4, ..base },
            Setup { eta: MOST_ETA, beta: MOST_BETA, ..base },
            Setup { eta: LEAST_ETA, batch: MOST_BATCH, steps: LEAST_STEPS, ..base },
            Setup { batch: 1, beta: MOST_BETA, averaging: true, window: MOST_WINDOW, ..base },
            Setup { start: [0.0, 0.0], ..base },
            Setup { schedule: Schedule::Drop, drop_at: 1, ..base },
            Setup { schedule: Schedule::Drop, drop_at: 2000, drop_batch: true, ..base },
            Setup { schedule: Schedule::Cosine, steps: LEAST_STEPS, averaging: true, window: 2, ..base },
            Setup { eta: f64::NAN, batch: 0, beta: -1.0, steps: 0, window: 0, start: [f64::INFINITY, f64::NAN], ..base },
        ];
        for s in cases {
            let mut scene = NoiseFloor::create();
            scene.setup = s;
            scene.compare = vec![Setup { batch: 1, ..base }];
            scene.axis = Axis::Examples;
            scene.refresh();
            let k = scene.setup;
            assert!(k.eta.is_finite() && k.batch >= 1 && k.steps >= LEAST_STEPS && k.start.iter().all(|v| v.is_finite()), "{k:?}");
            while !scene.swarm.done() {
                scene.swarm.step();
            }
            scene.shown = scene.swarm.t as f64;
            scene.follow(0.016, true);
            let look = &scene.look;
            for g in [&look.frame, &look.floors, &look.dots, &look.rings].into_iter().chain(look.curves.iter()).chain(look.strips.iter()) {
                assert!(g.shown.iter().all(|v| v.is_finite()), "{k:?}: {:?}", g.shown);
            }
            let fr = scene.frame_goal();
            assert!(fr[X_HI] > fr[X_LO] && fr[Y_HI] > fr[Y_LO] && fr[STRIP_HI] > 0.0, "{k:?}: {fr:?}");
        }
        // over the limit: no floor, the swarm flies off and stops, the curve is blown
        let mut s = NoiseFloor::create();
        s.setup = Setup { eta: 2.4, ..base };
        s.refresh();
        assert!(s.setup.floor().is_none() && s.exact.blown.is_some());
        while !s.swarm.done() {
            s.swarm.step();
        }
        assert!(s.swarm.diverged && s.swarm.t < 2000);
        // at the bottom: the loss starts at 0 and rises to the floor
        let e = Exact::of(&Setup { start: [0.0, 0.0], ..base }, 2000);
        assert!(e.loss[0] == 0.0 && e.loss[1] > 0.0 && close(e.loss[2000], base.floor().unwrap(), 1e-9));
    }

    /// The film: its clock shows step 0 at 0 and the last step at its end, one decade
    /// in the same time as another; it stops at the end and the picture comes to rest.
    #[test]
    fn the_film_plays_to_the_end_and_rests() {
        for steps in [10usize, 125, 2000, 10_000] {
            let len = Film::length(steps);
            assert_eq!(Film::step_at(0.0, steps), 0.0);
            assert!(close(Film::step_at(len, steps), steps as f64, 1e-12));
            for c in [0.3, 1.7, len * 0.8] {
                assert!(close(Film::clock_of(Film::step_at(c, steps), steps), c, 1e-9));
            }
        }
        let mut s = NoiseFloor::create();
        let mut frames = 0;
        let mut last = 0.0;
        while s.animated() {
            s.advance(1.0 / 60.0, &Budget::new(1.0));
            s.follow(1.0 / 60.0, false);
            assert!(s.shown >= last, "the film went back");
            last = s.shown;
            frames += 1;
            assert!(frames < 60 * 30, "still playing");
        }
        assert_eq!(s.swarm.t, 2000);
        assert_eq!(s.shown, 2000.0);
        // about as long as it says
        assert!((frames as f64 / 60.0 - Film::length(2000)).abs() < 1.0, "{frames}");
    }

    /// The layout keeps everything in the window and apart, from a phone to a desktop.
    #[test]
    fn the_layout_fits_every_window() {
        for (w, h) in [(1220.0, 1000.0), (1100.0, 700.0), (1600.0, 1000.0), (1600.0, 700.0), (390.0, 800.0), (360.0, 640.0), (700.0, 500.0), (300.0, 600.0), (500.0, 400.0), (240.0, 180.0)] {
            let lay = Layout::of(Vec2::new(w, h), 40.0);
            let inside = |(a, b): Rect| a.x >= 0.0 && a.y >= 0.0 && b.x <= w + 0.5 && b.y <= h + 0.5 && a.is_finite() && b.is_finite();
            let apart = |(a, b): Rect, (c, d): Rect| b.x <= c.x || d.x <= a.x || b.y <= c.y || d.y <= a.y;
            assert!(inside(lay.bowl), "{w}x{h}: {:?}", lay.bowl);
            let (a, b) = lay.bowl;
            assert!(((b.x - a.x) - 2.0 * (b.y - a.y)).abs() < 1.0, "{w}x{h}: the bowl is not two by one");
            if lay.compact {
                continue;
            }
            let boxes: Vec<Rect> = [vec![lay.bowl, lay.strip, lay.chart], lay.buttons.to_vec()].concat();
            for (i, a) in boxes.iter().enumerate() {
                assert!(inside(*a), "{w}x{h}: box {i} {a:?}");
                for b in &boxes[i + 1..] {
                    assert!(apart(*a, *b), "{w}x{h}: {a:?} and {b:?} overlap");
                }
            }
            assert!(lay.chart.1.y - lay.chart.0.y > 120.0 && lay.chart.1.x - lay.chart.0.x > 200.0, "{w}x{h}: {:?}", lay.chart);
            assert!(lay.bowl.1.x - lay.bowl.0.x > 200.0, "{w}x{h}: {:?}", lay.bowl);
        }
    }

    /// The picture's glides come to rest after a change, and the film then plays.
    #[test]
    fn glides_arrive_and_stop() {
        let mut s = NoiseFloor::create();
        s.follow(0.016, true);
        for k in [3, 0, 5, 4, 6] {
            s.set_up(k);
            let mut frames = 0;
            s.follow(1.0 / 60.0, false);
            while s.look.moving {
                s.follow(1.0 / 60.0, false);
                frames += 1;
                assert!(frames < 300, "experiment {k} still moving");
            }
            assert!(frames > 3, "experiment {k}: {frames}");
            let budget = Budget::new(1.0);
            s.advance(1.0 / 60.0, &budget);
            assert!(s.film.playing, "experiment {k}");
        }
    }

    /// The numbers behind the captions (run with --ignored --nocapture).
    #[test]
    #[ignore]
    fn explore() {
        for k in 0..TRIALS.len() {
            let s = run(k);
            let fl = s.setup.floor();
            println!("{k} {:<28} floor {:?} start {} settled {:?}", TRIALS[k].button, fl, s.setup.start_loss(), fl.and_then(|f| s.exact.settled_from(f, 0)));
            for e in &s.compared {
                let f = e.setup.floor();
                println!("    compare {:<40} floor {:?} settled {:?}", e.setup.label(), f, f.and_then(|f| e.settled_from(f, 0)));
            }
            println!("    plateau {:?} avg {:?}", s.swarm.plateau(false), s.swarm.plateau(true));
        }
    }
}

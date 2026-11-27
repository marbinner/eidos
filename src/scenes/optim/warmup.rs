//! Warm-up and decay: the learning-rate schedule. One small network, trained twice (a
//! bright run and a pale one, with schedules of the visitor's choosing) and measured as
//! it trains: its loss on the whole data, the sharpness of that loss (the largest eigenvalue of
//! its Hessian; for Adam, of the Hessian preconditioned by Adam's divisors) with the
//! ceiling 2/S it sets on the step, and, with mini-batches, the ceiling that the
//! gradient's noise sets in McCandlish and co-authors' model of one step.
//!
//! The two ends of a schedule have two different reasons, and the charts show both:
//! at the start the curvature is high and a long step is unstable; at the end the
//! curvature no longer binds, but the noise of a mini-batch holds a constant step's
//! loss up, and only a shorter step lets it go lower.
//!
//! Everything is measured on this network, for its fixed data and starting weights
//! (and, with mini-batches, for a stated order of the batches): the captions say what
//! happened here, and the tests at the end of the file hold them.

use super::*;
use crate::clock::{self, Budget};
use crate::math::arch::{self, Init, Loss, MlpSpec, Model, Norm, Place};
use crate::math::autograd::{Act, Tape};
use crate::math::optim::{Hyper, Optimizer, Rule, Schedule};
use crate::math::terrain::{effective_batch, Draw, FullNoise, Lanczos, Problem, Sampler};
use crate::scenes::stats::plot::{self, Glide, Plot, Rect, AXES};

pub const DEF: SceneDef = SceneDef {
    id: "warmup",
    title: "Warm-up and decay: the learning-rate schedule",
    group: GROUP,
    blurb: "Why training starts with short steps and ends with short steps: curvature too high for a long step at the start, and noise that only a shorter step can quiet at the end.",
    make: |_| Box::new(Warmup::create()),
};

// ----- the task and the network ----------------------------------------------------------

/// Examples in the data set.
const N: usize = 128;
/// Numbers per example.
const INPUTS: usize = 2;
/// The standard deviation of the noise added to every target. No network can fit it on
/// new points, and on these it keeps the gradients of single examples from all
/// vanishing together: the noise of a mini-batch does not die out near the best fit.
const LABEL_NOISE: f64 = 0.3;
/// The network: three hidden tanh layers of 16 and one output (609 weights and biases),
/// every weight matrix drawn as LeCun's start times `GAIN`. (A smooth activation: the
/// curvature is measured by differences of gradients, which a ReLU's kink would spoil.)
const WIDTH: usize = 16;
const DEPTH: usize = 3;
const GAIN: f64 = 2.0;
/// The seeds of the data and of the starting weights (the same in every run).
const DATA_SEED: u64 = 1;
const WEIGHT_SEED: u64 = 1;

/// The data: `N` points of the plane with standard normal coordinates, each with the
/// output of a fixed small tanh network (`arch::teacher`) plus normal noise of
/// standard deviation `LABEL_NOISE`; the loss is the mean squared error.
fn problem() -> Problem {
    let mut rng = Rng::new(DATA_SEED);
    let (x, mut y) = arch::teacher(N, INPUTS, &mut rng);
    for v in &mut y {
        *v += LABEL_NOISE * rng.normal() as f64;
    }
    Problem::points(&x, &y, N, Loss::Mse)
}

fn spec() -> MlpSpec {
    MlpSpec {
        inputs: INPUTS,
        width: WIDTH,
        depth: DEPTH,
        outputs: 1,
        act: Act::Tanh,
        norm: Norm::None,
        place: Place::Pre,
        residual: false,
        residual_scale: 1.0,
        init: Init::Lecun,
        gain: GAIN,
        bias: true,
    }
}

/// The starting weights, the same for every run.
fn start_model() -> Model {
    Model::mlp(&spec(), &mut Rng::new(WEIGHT_SEED))
}

// ----- a run -------------------------------------------------------------------------------

/// The rule a run trains with.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Method {
    /// Plain gradient descent, θ ← θ − η g: with the whole data, or with mini-batches (SGD).
    Gd,
    /// Adam with Kingma and Ba's defaults (β1 0.9, β2 0.999, ε 1e-8), as `math::optim` has it.
    Adam,
}

impl Method {
    fn rule(self) -> Rule {
        match self {
            Method::Gd => Rule::Sgd,
            Method::Adam => Rule::Adam,
        }
    }

    /// What multiplies 1/η in the stability limit the scene draws. For gradient
    /// descent 2: on a quadratic it is stable exactly while the sharpness is under 2/η
    /// (the course notes, curvature.md A1). For Adam with its divisors held fixed
    /// ("frozen Adam"), (2 + 2β1)/(1 − β1) = 38 at β1 = 0.9, a limit on the
    /// preconditioned sharpness (Cohen and co-authors 2022, §4; curvature.md A4, B2).
    fn limit_factor(self) -> f64 {
        match self {
            Method::Gd => 2.0,
            Method::Adam => {
                let b1 = Hyper::default_for(Rule::Adam).beta1;
                (2.0 + 2.0 * b1) / (1.0 - b1)
            }
        }
    }

    fn color(self) -> Color {
        match self {
            Method::Gd => SGD,
            Method::Adam => ADAM,
        }
    }
}

/// What the step does after the warm-up.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Decay {
    Constant,
    Linear,
    Cosine,
}

/// Everything a run depends on.
#[derive(Clone, Copy, Debug, PartialEq)]
struct Setup {
    method: Method,
    /// The step the warm-up rises to (the learning rate at its peak).
    peak: f64,
    /// Steps of linear warm-up from 0 (0: none).
    warmup: u64,
    decay: Decay,
    /// Examples a step's gradient averages (`N`: the whole data, gradient descent).
    batch: usize,
    /// Steps in the run.
    steps: u64,
    /// The seed of the order of the mini-batches.
    seed: u64,
}

/// The controls' ranges.
const LEAST_PEAK: f64 = 1e-3;
const MOST_PEAK: f64 = 1.0;
const MOST_WARMUP: u64 = 1000;
const LEAST_STEPS: u64 = 50;
const MOST_STEPS: u64 = 3000;

impl Setup {
    /// Gradient descent with the whole data at `peak`, no warm-up, constant, 150 steps.
    fn gd(peak: f64) -> Setup {
        Setup { method: Method::Gd, peak, warmup: 0, decay: Decay::Constant, batch: N, steps: 150, seed: 1 }
    }

    /// Kept to the controls' ranges (a value that is not a number takes 0.01).
    fn kept(self) -> Setup {
        let peak = if self.peak.is_finite() { self.peak.max(LEAST_PEAK).min(MOST_PEAK) } else { 0.01 };
        let steps = self.steps.clamp(LEAST_STEPS, MOST_STEPS);
        Setup { peak, steps, warmup: self.warmup.min(MOST_WARMUP).min(steps), batch: self.batch.clamp(1, N), ..self }
    }

    /// The learning rate at every step, as `math::optim::Schedule` defines it: a linear
    /// warm-up from 0 over `warmup` steps (η_t = peak · t / warmup), then the decay over
    /// the steps that are left, to 0.
    fn schedule(&self) -> Schedule {
        let after = self.steps.saturating_sub(self.warmup);
        let s = match self.decay {
            Decay::Constant => Schedule::constant(self.peak),
            Decay::Linear => Schedule::linear(self.peak, 0.0, after),
            Decay::Cosine => Schedule::cosine(self.peak, 0.0, after),
        };
        s.with_warmup(self.warmup, 0.0)
    }

    fn full_batch(&self) -> bool {
        self.batch >= N
    }

    /// The sharpness (or Adam's preconditioned sharpness) is re-estimated every this
    /// many steps: with the whole data, every step in a run of up to 300 steps and
    /// about 300 times a longer one; with mini-batches, about 100 times a run.
    fn sharp_every(&self) -> u64 {
        let times = if self.full_batch() { 300 } else { 100 };
        self.steps.div_ceil(times).max(1)
    }

    /// The whole data's loss is measured every this many steps (with the whole data
    /// as the batch, at every step: the gradient gives it).
    fn loss_every(&self) -> u64 {
        if self.full_batch() {
            1
        } else {
            self.steps.div_ceil(600).max(1)
        }
    }

    /// The noise ceiling is measured every this many steps, with mini-batches only:
    /// nine times a run, from the start to the end.
    fn noise_every(&self) -> u64 {
        self.steps.div_ceil(NOISE_MARKS - 1).max(1)
    }

    /// The run as the picture names it.
    fn label(&self, short: bool) -> String {
        let rule = match (self.method, self.full_batch(), short) {
            (Method::Gd, true, _) => "gradient descent".to_string(),
            (Method::Gd, false, false) => format!("SGD, batches of {}", self.batch),
            (Method::Gd, false, true) => format!("SGD, $B = {}$", self.batch),
            (Method::Adam, true, _) => "Adam".to_string(),
            (Method::Adam, false, false) => format!("Adam, batches of {}", self.batch),
            (Method::Adam, false, true) => format!("Adam, $B = {}$", self.batch),
        };
        let step = match (self.warmup, short) {
            (0, _) => format!(r"$\eta = {}$ at once", short_num(self.peak)),
            (w, false) => format!(r"$\eta$ up to {} over {} steps", short_num(self.peak), count(w as usize)),
            (w, true) => format!(r"$\eta \to {}$ over {}", short_num(self.peak), count(w as usize)),
        };
        let decay = match (self.decay, short) {
            (Decay::Constant, _) => "",
            (Decay::Linear, false) => ", then a linear decay to 0",
            (Decay::Cosine, false) => ", then a cosine decay to 0",
            (Decay::Linear, true) => ", linear decay",
            (Decay::Cosine, true) => ", cosine decay",
        };
        format!("{rule}, {step}{decay}, {} steps", count(self.steps as usize))
    }
}

/// How many times a run with mini-batches has its noise ceiling measured.
const NOISE_MARKS: u64 = 9;
/// A loss above this has run away: the run stops there.
const BLOWN: f64 = 1e5;
/// The most Hessian-vector products one estimate of the sharpness may take.
const MOST_PRODUCTS: usize = 60;
/// How exactly the sharpness is estimated: until the Lanczos residual is a thousandth
/// of the estimate (warm-started from the last estimate, it takes two or three
/// products a step).
const TOLERANCE: f64 = 1e-3;

/// One estimate of the sharpness, at the weights of step `t`: the largest eigenvalue
/// of the Hessian of the whole data's loss (for Adam, of P^(−1/2) H P^(−1/2), P the
/// numbers the next step divides by), and the Lanczos residual over the estimate.
#[derive(Clone, Copy, Debug, PartialEq)]
struct Sharp {
    t: u64,
    value: f64,
    residual: f64,
}

/// One measurement of the noise ceiling at the weights of step `t`: McCandlish and
/// co-authors' one-step model there (|G|², Gᵀ H G and tr(H Σ), `terrain::FullNoise`,
/// exact), and the batch it is for.
#[derive(Clone, Copy, Debug, PartialEq)]
struct NoiseMark {
    t: u64,
    g2: f64,
    ghg: f64,
    trhs: f64,
    /// The batch drawn with replacement that is as noisy as the run's, drawn without
    /// (`terrain::effective_batch`): the one-step model is written for that.
    batch: f64,
}

impl NoiseMark {
    /// The longest step that lowers the loss on average in the one-step model: twice
    /// its best step, 2|G|² / (Gᵀ H G + tr(H Σ)/B) (McCandlish and co-authors 2018,
    /// Eqs. 2.5 and 2.6: the expected change −ε|G|² + ½ε²(Gᵀ H G + tr(H Σ)/B) is
    /// negative for ε between 0 and that). None where the model has no such step (its
    /// curvature term is not positive) or a number is missing.
    fn ceiling(&self) -> Option<f64> {
        let c = self.ghg + self.trhs / self.batch;
        let v = 2.0 * self.g2 / c;
        (c > 0.0 && v.is_finite()).then_some(v)
    }

    /// The share of the curvature term that the noise makes, tr(H Σ)/B over the whole.
    fn noise_share(&self) -> f64 {
        let n = self.trhs / self.batch;
        n / (self.ghg + n)
    }
}

/// Where the work on the present step has got.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Phase {
    Gradient,
    Sharpness,
    Step,
    Done,
}

/// A measurement of the noise ceiling waiting to be done, on a copy of the weights of
/// step `t` (so that the run need not wait for it).
#[derive(Clone, Debug)]
struct NoiseJob {
    t: u64,
    model: Model,
    noise: FullNoise,
}

/// A run of training from the scene's starting weights, measured as it goes, a slice
/// at a time.
#[derive(Clone, Debug)]
struct Run {
    setup: Setup,
    schedule: Schedule,
    model: Model,
    opt: Optimizer,
    sampler: Sampler,
    tape: Tape,
    lanczos: Lanczos,
    jobs: Vec<NoiseJob>,
    /// Adam's divisors for the next step.
    divisors: Vec<f64>,
    /// Whether to measure the sharpness and the noise (tests that want only the loss
    /// leave it off).
    measure: bool,
    /// The step being worked on (the model has its weights), and where that work is.
    t: u64,
    phase: Phase,
    /// The learning rate of every step taken (index t).
    lr: Vec<f64>,
    /// The whole data's loss at the weights of step t (NaN where it was not measured),
    /// from t = 0.
    loss: Vec<f64>,
    sharp: Vec<Sharp>,
    noise_marks: Vec<NoiseMark>,
    /// The step at whose weights the loss had run away, if it did.
    blown: Option<u64>,
}

impl Run {
    fn new(setup: Setup, measure: bool) -> Run {
        let setup = setup.kept();
        let model = start_model();
        let opt = Optimizer::new(&model, setup.method.rule(), Hyper::default_for(setup.method.rule()));
        let mut lanczos = Lanczos::new(&model, 1, 7);
        lanczos.tolerance = TOLERANCE;
        Run {
            setup,
            schedule: setup.schedule(),
            sampler: Sampler::new(N, setup.batch, Draw::WithoutReplacement, Rng::new(setup.seed.wrapping_mul(0x9E37_79B9_7F4A_7C15).wrapping_add(3))),
            model,
            opt,
            tape: Tape::new(),
            lanczos,
            jobs: Vec::new(),
            divisors: Vec::new(),
            measure,
            t: 0,
            phase: Phase::Gradient,
            lr: Vec::new(),
            loss: Vec::new(),
            sharp: Vec::new(),
            noise_marks: Vec::new(),
            blown: None,
        }
    }

    /// Its steps are all taken (or it has run away).
    fn done(&self) -> bool {
        self.phase == Phase::Done
    }

    /// Its steps taken and every measurement of them made.
    fn complete(&self) -> bool {
        self.done() && self.jobs.is_empty()
    }

    /// The last step whose weights have been measured (their loss, and the sharpness
    /// where it is due): what the picture may show of this run.
    fn recorded(&self) -> f64 {
        match self.phase {
            Phase::Done => self.loss.len().saturating_sub(1) as f64,
            _ => self.t as f64 - 1.0,
        }
    }

    fn due(&self, every: u64) -> bool {
        self.t.is_multiple_of(every) || self.t == self.setup.steps
    }

    /// The whole run at once, with every measurement.
    #[cfg(test)]
    fn finish(&mut self, problem: &Problem) {
        let budget = Budget::new(f64::INFINITY);
        while !self.done() {
            self.unit(problem);
        }
        while !self.jobs.is_empty() {
            self.noise_unit(problem, &budget);
        }
    }

    /// One piece of the work on the present step: its gradient, its sharpness (a slice
    /// of products, within the budget), or the step itself.
    fn unit(&mut self, problem: &Problem) {
        let s = self.setup;
        match self.phase {
            Phase::Done => {}
            Phase::Gradient => {
                let loss = if s.full_batch() {
                    problem.full_grad(&mut self.model, &mut self.tape)
                } else {
                    let whole = if self.due(s.loss_every()) { problem.loss(&self.model, &mut self.tape) } else { f64::NAN };
                    let batch = self.sampler.deal().to_vec();
                    let on_batch = problem.grad(&mut self.model, &mut self.tape, &batch);
                    // (a batch whose loss has run away means the run's has)
                    if on_batch.is_finite() && on_batch < BLOWN {
                        whole
                    } else {
                        f64::INFINITY
                    }
                };
                self.loss.push(loss);
                if !(loss.is_nan() || loss < BLOWN) {
                    self.blown = Some(self.t);
                    self.phase = Phase::Done;
                    return;
                }
                if self.measure && !s.full_batch() && self.due(s.noise_every()) {
                    self.jobs.push(NoiseJob { t: self.t, model: self.model.clone(), noise: FullNoise::new(problem, &self.model) });
                }
                self.phase = if self.measure && self.due(s.sharp_every()) {
                    if s.method == Method::Adam {
                        next_divisors(&self.opt, &self.model, &mut self.divisors);
                        self.lanczos.set_preconditioner(&self.divisors);
                    } else {
                        self.lanczos.restart();
                    }
                    Phase::Sharpness
                } else {
                    Phase::Step
                };
            }
            Phase::Sharpness => {
                // (two products at a time, so that a unit of work stays short whatever
                // the budget: one that will not converge stops at `MOST_PRODUCTS`)
                let done = self.lanczos.advance(problem, &mut self.model, &mut self.tape, 2);
                if done || self.lanczos.products() >= MOST_PRODUCTS {
                    let (value, residual) = (self.lanczos.sharpness(), self.lanczos.residual());
                    self.sharp.push(Sharp { t: self.t, value, residual });
                    self.phase = Phase::Step;
                }
            }
            Phase::Step => {
                if self.t >= s.steps {
                    self.phase = Phase::Done;
                    return;
                }
                let lr = self.schedule.at(self.t);
                self.lr.push(lr);
                self.opt.step(&mut self.model, lr);
                self.t += 1;
                self.phase = Phase::Gradient;
            }
        }
    }

    /// A slice of the oldest measurement of the noise ceiling still to do.
    fn noise_unit(&mut self, problem: &Problem, budget: &Budget) {
        let Some(job) = self.jobs.first_mut() else { return };
        if job.noise.work(problem, &mut job.model, &mut self.tape, budget) {
            let one = job.noise.one_step();
            let batch = effective_batch(self.setup.batch, N, Draw::WithoutReplacement);
            self.noise_marks.push(NoiseMark { t: job.t, g2: one.g2, ghg: one.ghg, trhs: one.trhs, batch });
            self.jobs.remove(0);
        }
    }

    /// The step of the oldest measurement of the noise ceiling still to do.
    fn next_job(&self) -> Option<u64> {
        self.jobs.first().map(|j| j.t)
    }

    /// The largest whole-data loss measured, and the step it was at (before any
    /// running away).
    fn highest(&self) -> (f64, u64) {
        self.loss.iter().enumerate().filter(|(_, l)| l.is_finite()).fold((f64::NAN, 0), |a, (t, &l)| if a.0.is_nan() || l > a.0 { (l, t as u64) } else { a })
    }

    /// The whole data's loss at the end (None if the run ran away or is not over).
    fn final_loss(&self) -> Option<f64> {
        (self.done() && self.blown.is_none()).then(|| self.loss.last().copied()).flatten()
    }

    /// The ceiling the estimate at step t sets on the step: 2/S (for Adam 38/S̃).
    fn ceiling(&self, s: &Sharp) -> Option<f64> {
        let c = self.setup.method.limit_factor() / s.value;
        (s.value > 0.0 && c.is_finite()).then_some(c)
    }

    /// The level the step of step t sets on the sharpness: 2/η (for Adam 38/η);
    /// infinite where η is 0.
    fn level(&self, t: u64) -> f64 {
        self.setup.method.limit_factor() / self.schedule.at(t)
    }
}

/// The numbers Adam's next step will divide each parameter's step by, P in θ ← θ − η
/// P⁻¹ m, from its state and the gradient the model holds: with v' = β2 v + (1 − β2) g²
/// and t' the next step's count, P = (1 − β1^t′)(√(v′/(1 − β2^t′)) + ε) (Cohen and
/// co-authors 2022, Eq. 2 as corrected in their second version; curvature.md B2). It is
/// what `Optimizer::preconditioner` reports after that step, found before it, so that
/// the preconditioned sharpness at the weights of step t is that of the step about to
/// be taken from them (the first one included).
fn next_divisors(opt: &Optimizer, model: &Model, out: &mut Vec<f64>) {
    let h = *opt.hyper();
    let t = (opt.steps() + 1).min(i32::MAX as u64) as i32;
    let (c1, c2) = (1.0 - h.beta1.powi(t), 1.0 - h.beta2.powi(t));
    out.clear();
    for (i, g) in model.groups().iter().enumerate() {
        let v = opt.second_moment(i);
        for (k, &gk) in g.grad.iter().enumerate() {
            let vn = h.beta2 * v.get(k).copied().unwrap_or(0.0) + (1.0 - h.beta2) * gk * gk;
            out.push(c1 * ((vn / c2).sqrt() + h.eps));
        }
    }
}

/// The sharpness of the loss at the starting weights (for Adam, of the step its first
/// update takes there with the whole data), from a cold start, to the scene's
/// tolerance.
fn start_sharpness(problem: &Problem, method: Method) -> f64 {
    let mut run = Run::new(Setup { method, ..Setup::gd(0.01) }, true);
    while run.sharp.is_empty() && !run.done() {
        run.unit(problem);
    }
    run.sharp.first().map_or(f64::NAN, |s| s.value)
}

// ----- numbers as the scene writes them ----------------------------------------------------

/// A number with three significant digits (0.0227, 16.5, 88.3, 3 184), with a true minus
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

/// A setting as the labels write it: as few digits as it needs (0.1, 0.09, 0.27).
fn short_num(v: f64) -> String {
    let text = sig(v);
    if text.contains('.') {
        text.trim_end_matches('0').trim_end_matches('.').to_string()
    } else {
        text
    }
}

/// A ratio with two significant digits (4.0, 7.9, 12).
fn times(v: f64) -> String {
    if !v.is_finite() {
        return "?".to_string();
    }
    if v.abs() >= 10.0 {
        num(v, 0)
    } else {
        num(v, 1)
    }
}

// ----- the experiments ---------------------------------------------------------------------

struct Trial {
    button: &'static str,
    caption: &'static str,
}

const TRIALS: [Trial; 6] = [
    Trial {
        button: "a long step from the start",
        caption: r"Both start at once above the ceiling $2/S_0 = 0.0227$. The bright run, at 0.09 (4.0 times it): the loss jumps from 2.64 to 16.5 at the first step while the sharpness falls, to 16.8 by step 2, under the level $2/\eta = 22.2$; by step 3 the loss is 0.639, and the run trains on. The pale run, at 0.18 (7.9 times): its loss passes 100 000 at step 12. Tried here without a warm-up: at 0.03 the loss never rose; from 0.05 to 0.14 (6.2 times the ceiling) it jumped and recovered; from 0.15 (6.6 times) it ran away. Lewkowycz and co-authors (2020) called such a jump a catapult; in deep, wide ReLU networks trained by SGD on squared error, they found the largest step that trained near 6 times this ceiling ($12/\lambda_0$ against $2/\lambda_0$, $\lambda_0$ the starting curvature).",
    },
    Trial {
        button: "the same peak, warmed up",
        caption: r"The same peak, 0.18, at once (bright) and after a warm-up of 50 steps (pale). The bright run runs away at step 12. In the pale one the sharpness first falls by itself while the step is short (from 88.3 to 67.3 by step 5). The step passes its ceiling at step 9; the sharpness stays over the level $2/\eta$, by up to 2.6 times (step 21), until the loss jumps (8.03 at step 23) and the sharpness drops to 14.7; a smaller jump (1.92 at step 43) takes it to 12.5. From step 50 on it stays between 10.5 and 14.1, round $2/\eta = 11.1$. Gilmer and co-authors (2021), with large networks and SGD with momentum, saw warm-up push the sharpness down; Kalra and Barkeshli (2024), where training lowers the sharpness by itself, as here, saw the warm-up go by one or more separate catapults, or ride that fall.",
    },
    Trial {
        button: "a warm-up too short",
        caption: r"A peak of 0.27 (12 times the starting ceiling), reached over 10 steps (bright) or 100 (pale). The bright run runs away at step 18; as the pale one's step climbs, its loss jumps over 1 four times, each within 10 steps of the step passing its ceiling (the highest, 7.90 at step 28), and it trains on, to 0.0754. Of the warm-ups tried here, the shortest that trained was 15 steps for a peak of 0.18, 50 for 0.27 and 150 for 0.36. But the larger peaks did not end lower: in 300 steps, 0.09 with no warm-up ended at 0.0602, under every run warmed up to 0.18, 0.27 or 0.36 (0.0671 to 0.101). Kalra and Barkeshli (2024) found the main gain from warm-up to be that a larger peak step could train, and the test performance then reached to depend mostly on that peak, little on the warm-up's length.",
    },
    Trial {
        button: "a peak under the ceiling",
        caption: r"A step of 0.01, under the starting ceiling of 0.0227, at once (bright) and after a warm-up of 50 steps (pale). Neither loss ever rises and the step stays under its ceiling all along: the sharpness falls in both, to 55.0 and 56.3. The pale run ends a little higher, 0.0986 against 0.0959: its first 50 steps were shorter. Here, where the step never meets its ceiling, the warm-up only costs time. Gilmer and co-authors (2021) found that some widely used networks (a ResNet-50 and a wide ResNet, with batch normalisation) did not gain from warm-up at the steps they tried.",
    },
    Trial {
        button: "Adam, with and without",
        caption: r"Adam at 0.1, at once (bright) or after a warm-up of 20 steps (pale). For Adam the curvature that counts is the preconditioned sharpness $\tilde S$; with Adam's divisors held fixed its limit is $38/\eta$ (Cohen and co-authors 2022). Adam's first divisors are only $1 - \beta_1 = 0.1$ times the gradient's size, so $\tilde S$ starts high: 3 184, 8.4 times the bright run's limit of 380. Its loss jumps from 2.64 to 17.8 at step 1, then recovers. The pale run's $\tilde S$ stays under a quarter of its limit through the warm-up, and its loss never goes over its start (it does climb from 0.472 at step 3 to 2.11 at step 6). At step 150 the bright run is at 0.0440 and the pale at 0.0466: the warm-up spared it the jump, and did not end lower. Kalra and Barkeshli (2024) also found $\tilde S$ starting very high, and Adam then training badly rather than diverging: here, at a step of 1.0 in runs of 300 steps, no warm-up ended at 0.364 and a warm-up of 100 steps at 0.0430.",
    },
    Trial {
        button: "the end: constant against cosine",
        caption: r"Batches of 8 (SGD) and a warm-up of 50 steps to 0.1; then the bright run stays there and the pale one falls to 0 along a cosine. At step 0 the two ceilings agree: 0.0227 from the curvature, 0.0217 from the noise. After the warm-up the curvature's has risen to between 0.15 and 0.19, far over the step, while the red dots, the noise ceiling, lie round the bright run's step, from 0.0103 to 0.133: a step of 0.1 is now over it, now under it, and the loss wanders on a floor. Under the pale run's cosine the noise ceiling comes down with the step, and the loss goes lower: it ends at 0.0450, the bright run at 0.0522. Over 8 orders of the batches the constant step ended at 0.0814 on average (0.0479 to 0.147) and the cosine at 0.0473 (0.0450 to 0.0501), lower in 7 of the 8. With batches of 32 they ended level, at 0.0405 and 0.0411.",
    },
];

/// The scene opens on this experiment: the same peak with and without a warm-up.
const OPENING: usize = 1;

/// The bright and the pale run of experiment k.
fn trial(k: usize) -> [Setup; 2] {
    let gd = Setup::gd;
    match k {
        0 => [gd(0.09), gd(0.18)],
        2 => [Setup { warmup: 10, steps: 300, ..gd(0.27) }, Setup { warmup: 100, steps: 300, ..gd(0.27) }],
        3 => [Setup { steps: 300, ..gd(0.01) }, Setup { warmup: 50, steps: 300, ..gd(0.01) }],
        4 => [Setup { method: Method::Adam, ..gd(0.1) }, Setup { method: Method::Adam, warmup: 20, ..gd(0.1) }],
        5 => {
            let sgd = Setup { batch: 8, steps: 2000, warmup: 50, ..gd(0.1) };
            [sgd, Setup { decay: Decay::Cosine, ..sgd }]
        }
        _ => [gd(0.18), Setup { warmup: 50, ..gd(0.18) }],
    }
}

// ----- the film ----------------------------------------------------------------------------

/// How long the film of the runs takes, in seconds of the scene's time: the step axis
/// is crossed at an even pace, whatever the number of steps.
const FILM: f64 = 5.0;
/// The time over which the step shown eases to the one the film is at (seconds).
const FILM_EASE: f64 = 0.12;

/// The runs shown as a film: the step shown is `clock / FILM` of the way along the axis,
/// and the clock waits for the runs where they have not got that far.
#[derive(Clone, Copy, Debug, Default)]
struct Film {
    clock: f64,
    playing: bool,
    /// Play once the axes have glided to the new runs' ranges.
    waiting: bool,
}

// ----- the layout --------------------------------------------------------------------------

/// Where everything goes in a window (logical pixels from its top-left corner).
#[derive(Clone, Copy, Debug, PartialEq)]
struct Layout {
    view: Vec2,
    narrow: bool,
    text: f32,
    small: f32,
    /// The runs' names, at the top left.
    names: Vec2,
    /// The button that plays the film again (or pauses it).
    button: Rect,
    /// The charts: the step, the sharpness, the loss (their data rectangles).
    charts: [Rect; 3],
    /// How many of the charts there is room for (from the loss up: 1 only shows the
    /// loss).
    shown: usize,
}

impl Layout {
    fn of(view: Vec2, inset: f32) -> Layout {
        let view = view.max(Vec2::splat(1.0));
        let narrow = view.x < 700.0;
        let (text, small) = if narrow { (12.0, 10.5) } else { (13.5, 11.5) };
        let side = if narrow { 8.0 } else { 18.0 };
        let top = inset.max(0.0) + if narrow { 6.0 } else { 10.0 };
        let r = |x0: f32, y0: f32, x1: f32, y1: f32| (Vec2::new(x0, y0), Vec2::new(x1.max(x0 + 1.0), y1.max(y0 + 1.0)));
        let button_w = if narrow { 74.0 } else { 108.0 };
        let button = r(view.x - side - button_w, top, view.x - side, top + if narrow { 22.0 } else { 24.0 });
        let names = Vec2::new(side, top);
        // (the names take two lines; below them the charts, each with its title above it)
        let first = top + 2.0 * (small + 6.0) + 10.0;
        let left = side + if narrow { 36.0 } else { 46.0 };
        let right = view.x - side - 6.0;
        let head = text + 12.0;
        let bottom = view.y - if narrow { 30.0 } else { 34.0 };
        let room = bottom - first;
        let gap = 10.0;
        // as many charts as have room for 70 pixels each, the loss first
        let shown = (1..=3).rev().find(|&k| room - k as f32 * head - (k - 1) as f32 * gap >= 70.0 * k as f32).unwrap_or(1);
        let each = ((room - shown as f32 * head - (shown - 1) as f32 * gap) / shown as f32).max(1.0);
        let mut charts = [r(0.0, 0.0, 0.0, 0.0); 3];
        for (slot, k) in (3 - shown..3).enumerate() {
            let y0 = first + slot as f32 * (head + each + gap) + head;
            charts[k] = r(left, y0, right, y0 + each);
        }
        Layout { view, narrow, text, small, names, button, charts, shown }
    }

    /// Whether chart k is drawn.
    fn has(&self, k: usize) -> bool {
        k + self.shown >= 3
    }
}

// ----- what is drawn, gliding --------------------------------------------------------------

// The charts' ranges, in the order the glide keeps them.
const X_HI: usize = 0;
const STEP_HI: usize = 1;
const S_LO: usize = 2;
const S_HI: usize = 3;
const L_LO: usize = 4;
const L_HI: usize = 5;

/// Lines whose number of points changes, each in a batch of its own: for each run, the
/// step, its ceiling, the sharpness, its level, the loss.
const LINES: u8 = plot::OWN;
const PER_RUN: u8 = 5;
const NOW: u8 = LINES + 2 * PER_RUN;
const BLOWN_MARK: u8 = NOW + 1;
/// Each run's noise ceilings, joined.
const NOISE_LINE: u8 = BLOWN_MARK + 1;
/// The ticks of the axes, in one of two batches: every tick of the ranges the axes are
/// gliding to is drawn in every frame (transparent where it is off its chart), so that
/// the batch keeps its count while they glide; and a new set of ranges goes in the
/// other batch, so that it is never taken for the old one moved.
const TICKS: u8 = NOISE_LINE + 2;

/// What the two runs are called: by how they are drawn.
const RUN_NAMES: [&str; 2] = ["bright", "pale"];

/// The point batches: each run's noise ceilings.
fn dot_batch<'b>(f: &'b mut Frame<'_>, k: usize) -> &'b mut crate::gfx::draw::PointBatch {
    f.dl.points_with(PointKind::Overlay, PointStyle { outline_px: 0.0, size_scale: 1.0 + 1e-4 * (k + 1) as f32, ..Default::default() })
}

/// Every batch the scene draws in, made in the same order in every frame, so that one
/// that is empty in a frame does not move the later ones up a place.
fn hold_batches(f: &mut Frame) {
    let corner = f.hud(Vec2::ZERO);
    for kind in [AXES, plot::MARKS, plot::BUTTONS].into_iter().chain(LINES..=TICKS + 1) {
        plot::apart(f, kind, false).segment(corner, corner, 1.0, Color::CLEAR);
    }
    plot::apart(f, AXES, true).segment(corner, corner, 1.0, Color::CLEAR);
    for k in 0..2 {
        dot_batch(f, k);
    }
}

/// Run k's colour for a thing of colour `c`: the first has it as it is, the second
/// paler.
fn shade(c: Color, k: usize) -> Color {
    if k == 0 {
        c
    } else {
        c.mix(pal::WHITE, 0.4)
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

/// A power of ten (or 2 or 5 times one) as a log axis writes it.
fn power_name(x: f64) -> String {
    let d = x.round() as i32;
    if (x - d as f64).abs() < 1e-6 && !(-2..=3).contains(&d) {
        return format!("$10^{{{d}}}$");
    }
    short_num(10f64.powf(x))
}

/// The scene.
pub struct Warmup {
    problem: Problem,
    setups: [Setup; 2],
    runs: [Run; 2],
    /// Which run the controls set.
    editing: usize,
    /// The sharpness at the starting weights, for gradient descent and for Adam.
    start: [f64; 2],
    film: Film,
    /// The step the picture shows, along the axis.
    shown: f64,
    look: Glide,
    /// The ranges as they are to be, and what they were worked out for.
    goal: Option<(GoalKey, Vec<f32>)>,
    /// The window last drawn in (a new one is drawn at once, without gliding).
    canvas: Option<Vec2>,
    /// The ranges the ticks are drawn for, and which of the two batches they are in.
    ticks_for: Vec<f32>,
    tick_batch: u8,
}

/// What the ranges depend on: the setups, and how much of each run has been measured.
type GoalKey = ([Setup; 2], [usize; 2], [usize; 2]);

impl Warmup {
    /// (Nothing here needs the graphics device, so the tests can make one too.)
    fn create() -> Warmup {
        let problem = problem();
        let start = [start_sharpness(&problem, Method::Gd), start_sharpness(&problem, Method::Adam)];
        let setups = trial(OPENING);
        let mut s = Warmup {
            runs: [Run::new(setups[0], true), Run::new(setups[1], true)],
            problem,
            setups,
            editing: 0,
            start,
            film: Film::default(),
            shown: 0.0,
            look: Glide::default(),
            goal: None,
            canvas: None,
            ticks_for: Vec::new(),
            tick_batch: TICKS,
        };
        s.set_up(OPENING);
        s
    }

    /// Set up experiment k, and play it.
    fn set_up(&mut self, k: usize) {
        self.setups = trial(k);
        self.editing = 0;
        self.restart();
    }

    /// Start both runs again from the start, and the film once the axes are there.
    fn restart(&mut self) {
        self.setups = self.setups.map(Setup::kept);
        self.runs = [Run::new(self.setups[0], true), Run::new(self.setups[1], true)];
        self.film = Film { clock: 0.0, playing: false, waiting: true };
        self.shown = 0.0;
    }

    /// Bring the runs up to date with the settings (a changed setup starts again).
    fn refresh(&mut self) {
        let kept = self.setups.map(Setup::kept);
        if kept != self.setups || kept[0] != self.runs[0].setup || kept[1] != self.runs[1].setup {
            self.setups = kept;
            self.restart();
        }
    }

    /// The axis's length: the longer run's steps.
    fn reach(&self) -> f64 {
        self.setups.iter().map(|s| s.steps).max().unwrap_or(1).max(1) as f64
    }

    /// Work on the runs within the budget: first whatever the film is waiting for, then
    /// the noise ceilings the film has passed, then the rest.
    fn work(&mut self, budget: &Budget) {
        let target = self.target();
        loop {
            let behind = (0..2).filter(|&k| !self.runs[k].done() && self.runs[k].recorded() < target.min(self.runs[k].setup.steps as f64)).min_by(|&a, &b| {
                let (x, y) = (self.runs[a].recorded() / self.runs[a].setup.steps as f64, self.runs[b].recorded() / self.runs[b].setup.steps as f64);
                x.total_cmp(&y)
            });
            let job = (0..2).filter_map(|k| self.runs[k].next_job().map(|t| (k, t))).min_by_key(|&(_, t)| t);
            let ahead = (0..2).filter(|&k| !self.runs[k].done()).min_by(|&a, &b| self.runs[a].recorded().total_cmp(&self.runs[b].recorded()));
            if let Some(k) = behind {
                self.runs[k].unit(&self.problem);
            } else if let Some((k, _)) = job {
                self.runs[k].noise_unit(&self.problem, budget);
            } else if let Some(k) = ahead {
                self.runs[k].unit(&self.problem);
            } else {
                return;
            }
            if budget.spent() {
                return;
            }
        }
    }

    /// The step the film wants to show now.
    fn target(&self) -> f64 {
        (self.film.clock / FILM).clamp(0.0, 1.0) * self.reach()
    }

    /// As far along the axis as both runs have been measured (a run that is over, or
    /// shorter, is measured as far as it goes).
    fn measured(&self) -> f64 {
        let reach = self.reach();
        self.runs.iter().map(|r| if r.done() { reach } else { r.recorded().max(0.0) }).fold(reach, f64::min)
    }

    /// Move the film on by `dt` seconds of the scene's time (it waits for the runs).
    fn advance(&mut self, dt: f32, moving: bool) {
        if self.film.waiting && !moving {
            self.film.waiting = false;
            self.film.playing = true;
        }
        if self.film.playing && dt > 0.0 {
            self.film.clock += dt as f64;
        }
        let reach = self.reach();
        let want = self.target();
        let have = self.measured();
        if want > have {
            self.film.clock = have / reach * FILM;
        }
        // (the picture follows through an ease: while the runs are being worked out,
        // what is measured comes in pieces of uneven size)
        let to = want.min(have);
        let next = if to < self.shown { to } else { self.shown + (to - self.shown) * (1.0 - (-(dt.max(0.0) as f64) / FILM_EASE).exp()) };
        self.shown = if (to - next).abs() < 1e-3 * reach { to } else { next };
        if self.film.playing && self.shown >= reach - 1e-9 && self.runs.iter().all(Run::complete) {
            self.film.playing = false;
        }
    }

    fn busy(&self) -> bool {
        self.film.playing || self.film.waiting || !self.runs.iter().all(Run::complete)
    }

    /// The charts' ranges as they are to be: across, the steps; the step up to a
    /// little over the largest peak and starting ceiling; the sharpness and the loss
    /// (logarithmic) round what the runs have measured.
    fn frame_goal(&self) -> Vec<f32> {
        let mut step_hi: f64 = 0.0;
        let (mut s_lo, mut s_hi) = (f64::INFINITY, 0.0f64);
        let (mut l_lo, mut l_hi) = (f64::INFINITY, 0.0f64);
        // the peaks and the starting ceilings, and the ceilings measured later up to
        // two and a half times as high
        let start = |k: usize| self.runs[k].sharp.first().map_or(self.start[(self.setups[k].method == Method::Adam) as usize], |x| x.value);
        let base = (0..2).map(|k| self.setups[k].peak.max(self.setups[k].method.limit_factor() / start(k))).filter(|v| v.is_finite()).fold(0.0, f64::max);
        for r in &self.runs {
            let later = r.sharp.iter().filter_map(|x| r.ceiling(x)).chain(r.noise_marks.iter().filter_map(NoiseMark::ceiling));
            step_hi = later.fold(step_hi, |m, c| m.max(c.min(2.5 * base)));
        }
        step_hi = step_hi.max(base);
        for (k, r) in self.runs.iter().enumerate() {
            let s = &self.setups[k];
            let s0 = start(k);
            if s0 > 0.0 && s0.is_finite() {
                s_hi = s_hi.max(s0);
                s_lo = s_lo.min(s0);
            }
            // the level at the peak, and what was measured (a run that ran away is
            // followed only a little way up)
            s_lo = s_lo.min(s.method.limit_factor() / s.peak);
            for x in &r.sharp {
                if x.value.is_finite() && x.value > 0.0 {
                    s_lo = s_lo.min(x.value);
                    s_hi = s_hi.max(x.value.min(s0 * 3.0));
                }
            }
            for &l in &r.loss {
                if l.is_finite() && l > 0.0 {
                    l_lo = l_lo.min(l);
                    l_hi = l_hi.max(l.min(100.0));
                }
            }
        }
        if !(s_lo.is_finite() && s_lo > 0.0) {
            s_lo = 1.0;
        }
        if s_hi.is_nan() || s_hi <= s_lo {
            s_hi = s_lo * 10.0;
        }
        if !(l_lo.is_finite() && l_lo > 0.0) {
            l_lo = 0.01;
        }
        if l_hi.is_nan() || l_hi <= l_lo {
            l_hi = l_lo * 100.0;
        }
        let (s_lo, mut s_hi) = (s_lo.log10() - 0.2, s_hi.log10() + 0.25);
        if s_hi - s_lo < 1.0 {
            s_hi = s_lo + 1.0;
        }
        let (l_lo, mut l_hi) = (l_lo.log10() - 0.2, l_hi.log10() + 0.2);
        if l_hi - l_lo < 1.0 {
            l_hi = l_lo + 1.0;
        }
        let mut g = vec![0.0f32; 6];
        g[X_HI] = self.reach() as f32;
        g[STEP_HI] = (step_hi.max(1e-6) * 1.12) as f32;
        g[S_LO] = s_lo as f32;
        g[S_HI] = s_hi as f32;
        g[L_LO] = l_lo as f32;
        g[L_HI] = l_hi as f32;
        g
    }

    /// The same, worked out again only when what it depends on changes.
    fn goal(&mut self) -> Vec<f32> {
        let key = (self.setups, [self.runs[0].sharp.len(), self.runs[1].sharp.len()], [self.runs[0].loss.len(), self.runs[1].loss.len()]);
        if self.goal.as_ref().is_none_or(|(k, _)| *k != key) {
            self.goal = Some((key, self.frame_goal()));
        }
        self.goal.as_ref().map(|(_, g)| g.clone()).unwrap_or_default()
    }

    /// Bring the axes `dt` seconds nearer to their goal (`snap`: there at once). Says
    /// whether they are still on their way.
    fn follow(&mut self, dt: f32, snap: bool) -> bool {
        let goal = self.goal();
        if snap || self.look.shown.len() != goal.len() {
            self.look.snap(&goal);
            return false;
        }
        self.look.follow(&goal, dt, false)
    }
}

// ----- drawing ---------------------------------------------------------------------------------

/// The part of the segment from `a` to `b` inside the rectangle, as the range of its
/// parameter (Liang and Barsky's way), if any of it is inside.
fn cut(a: Vec2, b: Vec2, (min, max): Rect) -> Option<(f32, f32)> {
    let d = b - a;
    let (mut t0, mut t1) = (0.0f32, 1.0f32);
    for (p, q) in [(-d.x, a.x - min.x), (d.x, max.x - a.x), (-d.y, a.y - min.y), (d.y, max.y - a.y)] {
        if p == 0.0 {
            if q < 0.0 {
                return None;
            }
        } else {
            let r = q / p;
            if p < 0.0 {
                t0 = t0.max(r);
            } else {
                t1 = t1.min(r);
            }
        }
    }
    (t0 <= t1).then_some((t0, t1))
}

/// A run's line through `pts` (step, value; with `log` the value's log₁₀ is drawn), cut
/// to the chart, as far as the film has got (`shown`): every point of it is drawn in
/// every frame, in its place, and the ones the film has not reached are transparent,
/// the segment it is crossing fading in. So the line is revealed without any of its
/// points moving (a tip drawn where the film is would race along a steep jump).
#[allow(clippy::too_many_arguments)]
fn reveal(f: &mut Frame, p: &Plot, kind: u8, pts: &[(f64, f64)], log: bool, shown: f64, width: f32, color: Color) {
    let rect = (p.rect.0 - Vec2::splat(0.5), p.rect.1 + Vec2::splat(0.5));
    // (a long run is drawn with one point in so many)
    let stride = pts.len().div_ceil(1500).max(1);
    let mut verts: Vec<(Vec2, f32)> = Vec::with_capacity(pts.len() / stride + 2);
    let mut before = f64::NEG_INFINITY;
    for (i, &(t, v)) in pts.iter().enumerate() {
        if !(i % stride == 0 || i + 1 == pts.len()) {
            continue;
        }
        let y = if !log {
            v
        } else if v > 0.0 {
            v.log10()
        } else {
            f64::NAN
        };
        let alpha = if t <= shown {
            1.0
        } else if before <= shown && t > before {
            ((shown - before) / (t - before)).clamp(0.0, 1.0)
        } else {
            0.0
        };
        verts.push((p.at(t, y), alpha as f32));
        before = t;
    }
    let mut out: Vec<Option<(Vec3, Color)>> = Vec::with_capacity(verts.len() + 8);
    let mut open = false;
    for w in verts.windows(2) {
        let ((a, aa), (b, ab)) = (w[0], w[1]);
        let piece = if a.is_finite() && b.is_finite() { cut(a, b, rect) } else { None };
        let Some((t0, t1)) = piece else {
            open = false;
            continue;
        };
        let at = |t: f32| (a + (b - a) * t, aa + (ab - aa) * t);
        if !(open && t0 == 0.0) {
            if !out.is_empty() {
                out.push(None);
            }
            let (q, al) = at(t0);
            out.push(Some((f.hud(q), color.a(color.0[3] as f32 / 255.0 * al))));
        }
        let (q, al) = at(t1);
        out.push(Some((f.hud(q), color.a(color.0[3] as f32 / 255.0 * al))));
        open = t1 == 1.0;
    }
    let batch = plot::apart(f, kind, false);
    for v in out {
        match v {
            Some((q, c)) => batch.push(q, width, c),
            None => batch.brk(),
        }
    }
    batch.brk();
}


impl Warmup {
    fn plot(&self, lay: &Layout, k: usize) -> Plot {
        let g = |i: usize| self.look.get(i) as f64;
        let x = [0.0, g(X_HI).max(1.0)];
        match k {
            0 => Plot::new(lay.charts[0], x, [0.0, g(STEP_HI).max(1e-6)]),
            1 => Plot::new(lay.charts[1], x, [g(S_LO), g(S_HI)]),
            _ => Plot::new(lay.charts[2], x, [g(L_LO), g(L_HI)]),
        }
    }

    /// The steps of the axis, its ticks and their numbers (numbers under the lowest
    /// chart only), and the line of the step the film is at.
    fn x_axis(&self, f: &mut Frame, p: &Plot, numbers: bool, size: f32) {
        let (min, max) = p.rect;
        plot::line_of(f, AXES, Vec2::new(min.x, max.y), max, 1.0, pal::DIM.a(0.8));
        let ticks = plot::Ticks::of(0.0, self.goal_value(X_HI).max(1.0), p.width());
        for v in ticks.values() {
            let x = p.px_x(v);
            let on = v >= p.x[0] - 1e-9 && v <= p.x[1] + 1e-9;
            plot::line_of(f, self.tick_batch, Vec2::new(x, max.y), Vec2::new(x, max.y + 3.0), 1.0, pal::DIM.a(if on { 0.8 } else { 0.0 }));
            if numbers && on {
                f.text_px(Vec2::new(x, max.y + 5.0), count(v.round().max(0.0) as usize), size, pal::DIM, Align2::CENTER_TOP);
            }
        }
        if numbers {
            f.text_px(Vec2::new(max.x, max.y + 7.0 + size), "step", size, pal::DIM, Align2::RIGHT_TOP);
        }
    }

    /// The value the glide of range `i` is going to.
    fn goal_value(&self, i: usize) -> f64 {
        self.goal.as_ref().and_then(|g| g.1.get(i)).map_or(self.look.get(i), |v| *v) as f64
    }

    /// The step chart's y axis: linear, its ticks those of the range it is gliding to.
    fn step_axis(&self, f: &mut Frame, p: &Plot, size: f32) {
        let (min, max) = p.rect;
        plot::line_of(f, AXES, min, Vec2::new(min.x, max.y), 1.0, pal::DIM.a(0.8));
        let ticks = plot::Ticks::of(0.0, self.goal_value(STEP_HI).max(1e-9), p.height() * 1.6);
        for v in ticks.values() {
            let y = p.px_y(v);
            let on = v >= p.y[0] - 1e-12 && v <= p.y[1] + 1e-12;
            plot::line_of(f, self.tick_batch, Vec2::new(min.x - 4.0, y), Vec2::new(min.x, y), 1.0, pal::DIM.a(if on { 0.8 } else { 0.0 }));
            if on {
                f.text_px(Vec2::new(min.x - 6.0, y), ticks.name(v), size, pal::DIM, Align2::RIGHT_CENTER);
            }
        }
    }

    /// A logarithmic y axis (`lo` and `hi` its ranges' places in the glide), its ticks
    /// those of the range it is gliding to.
    fn log_axis(&self, f: &mut Frame, p: &Plot, lo: usize, hi: usize, size: f32) {
        let (min, max) = p.rect;
        plot::line_of(f, AXES, min, Vec2::new(min.x, max.y), 1.0, pal::DIM.a(0.8));
        for (y, named) in log_ticks(self.goal_value(lo), self.goal_value(hi)) {
            if !named {
                continue;
            }
            let at = p.px_y(y);
            let on = y >= p.y[0] - 1e-9 && y <= p.y[1] + 1e-9;
            let a = if on { 1.0 } else { 0.0 };
            plot::line_of(f, self.tick_batch, Vec2::new(min.x - 4.0, at), Vec2::new(min.x, at), 1.0, pal::DIM.a(0.8 * a));
            plot::line_of(f, self.tick_batch, Vec2::new(min.x, at), Vec2::new(max.x, at), 1.0, pal::DIM.a(0.1 * a));
            if on {
                f.text_px(Vec2::new(min.x - 6.0, at), power_name(y), size, pal::DIM, Align2::RIGHT_CENTER);
            }
        }
    }

    /// A chart's title, and its key at the right: names in their colours.
    fn heading(f: &mut Frame, lay: &Layout, p: &Plot, title: &str, key: &[(Color, String)]) {
        let (min, max) = p.rect;
        let y = min.y - 6.0;
        let left = min.x - if lay.narrow { 34.0 } else { 44.0 };
        f.text_px(Vec2::new(left, y), title, lay.text, pal::TEXT, Align2::LEFT_BOTTOM);
        // the key beside the title where there is room, else in the chart's top corner
        let widths: Vec<f32> = key.iter().map(|(_, name)| f.text_size(name, lay.small).x).collect();
        let room = widths.iter().map(|w| w + 12.0).sum::<f32>() + f.text_size(title, lay.text).x + 16.0 <= max.x - left;
        let mut x = if room { max.x } else { max.x - 6.0 };
        for ((c, name), w) in key.iter().zip(&widths).rev() {
            if room {
                f.text_px(Vec2::new(x, y), name.clone(), lay.small, *c, Align2::RIGHT_BOTTOM);
            } else {
                f.caption_px(Vec2::new(x, min.y + 4.0), name.clone(), lay.small, *c, Align2::RIGHT_TOP);
            }
            x -= w + if room { 12.0 } else { 16.0 };
        }
    }

    fn draw_step_chart(&self, f: &mut Frame, lay: &Layout) {
        let p = self.plot(lay, 0);
        if p.tiny() {
            return;
        }
        p.backing(f, if lay.narrow { 36.0 } else { 46.0 }, 8.0, lay.text + 12.0, 6.0);
        self.x_axis(f, &p, !lay.has(1) && !lay.has(2), lay.small);
        self.step_axis(f, &p, lay.small);
        let noisy = self.setups.iter().any(|s| !s.full_batch());
        let mut key: Vec<(Color, String)> = vec![(self.setups[0].method.color(), r"step $\eta$".to_string()), (LIMIT, r"ceiling $2/S$".to_string())];
        if self.setups.iter().any(|s| s.method == Method::Adam) {
            key[1] = (LIMIT, if self.setups.iter().all(|s| s.method == Method::Adam) { r"ceiling $38/\tilde S$".to_string() } else { r"ceilings $2/S$, $38/\tilde S$".to_string() });
        }
        if noisy {
            key.push((NOISE, "noise ceiling".to_string()));
        }
        let title = if lay.narrow { r"the step $\eta$ and its ceilings" } else { r"the step size $\eta$, and the longest steps the curvature and the noise allow" };
        Self::heading(f, lay, &p, title, &key);
        for (k, r) in self.runs.iter().enumerate().rev() {
            // the ceiling the curvature sets
            let ceil: Vec<(f64, f64)> = r.sharp.iter().filter_map(|s| r.ceiling(s).map(|c| (s.t as f64, c))).collect();
            reveal(f, &p, LINES + PER_RUN * k as u8 + 1, &ceil, false, self.shown, if k == 0 { 1.6 } else { 1.3 }, shade(LIMIT, k));
            // the step
            let steps: Vec<(f64, f64)> = r.lr.iter().enumerate().map(|(t, &v)| (t as f64, v)).collect();
            reveal(f, &p, LINES + PER_RUN * k as u8, &steps, false, self.shown, if k == 0 { 2.2 } else { 1.8 }, shade(r.setup.method.color(), k));
            // the noise ceilings, as dots joined by a thin line, those the film has not
            // reached transparent
            let size = if lay.narrow { 5.0 } else { 6.5 } * if k == 0 { 1.0 } else { 0.8 };
            let marks: Vec<(f64, f64)> = r.noise_marks.iter().filter_map(|m| m.ceiling().map(|c| (m.t as f64, c))).collect();
            reveal(f, &p, NOISE_LINE + k as u8, &marks, false, self.shown, 1.0, shade(NOISE, k).a(0.45));
            let dots: Vec<(Vec3, f32)> = marks
                .iter()
                .map(|&(t, c)| (p.at(t, c), if t <= self.shown + 1e-9 { 1.0 } else { 0.0 }))
                .filter(|(q, _)| plot::within(*q, p.rect, 0.5))
                .map(|(q, a)| (f.hud(q), a))
                .collect();
            let b = dot_batch(f, k);
            for (q, a) in dots {
                b.push(q, size, shade(NOISE, k).a(a));
            }
        }
    }

    fn draw_sharp_chart(&self, f: &mut Frame, lay: &Layout) {
        let p = self.plot(lay, 1);
        if p.tiny() {
            return;
        }
        p.backing(f, if lay.narrow { 36.0 } else { 46.0 }, 8.0, lay.text + 12.0, 6.0);
        self.x_axis(f, &p, !lay.has(2), lay.small);
        self.log_axis(f, &p, S_LO, S_HI, lay.small);
        let adam = self.setups.iter().map(|s| s.method == Method::Adam).collect::<Vec<_>>();
        let (name, level) = match (adam[0], adam[1]) {
            (false, false) => (r"sharpness $S$", r"level $2/\eta$"),
            (true, true) => (r"preconditioned sharpness $\tilde S$", r"level $38/\eta$"),
            _ => (r"$S$ and $\tilde S$", r"levels $2/\eta$, $38/\eta$"),
        };
        let title = match (lay.narrow, adam[0] || adam[1]) {
            (true, _) => "the sharpness, and the level the step sets",
            (false, false) => r"the sharpness $S$ (the Hessian's largest eigenvalue), and the most the step allows, $2/\eta$",
            (false, true) => r"the (preconditioned) sharpness, and the most the step allows",
        };
        Self::heading(f, lay, &p, title, &[(CURVATURE, name.to_string()), (LIMIT, level.to_string())]);
        for (k, r) in self.runs.iter().enumerate().rev() {
            // the level the step sets (where the step is 0, off the top)
            let top = 10f64.powf(p.y[1] + 1.0);
            let levels: Vec<(f64, f64)> = (0..r.lr.len().max(1)).map(|t| (t as f64, r.level(t as u64).min(top))).collect();
            reveal(f, &p, LINES + PER_RUN * k as u8 + 3, &levels, true, self.shown, if k == 0 { 1.6 } else { 1.3 }, shade(LIMIT, k));
            let sharp: Vec<(f64, f64)> = r.sharp.iter().map(|s| (s.t as f64, s.value)).collect();
            reveal(f, &p, LINES + PER_RUN * k as u8 + 2, &sharp, true, self.shown, if k == 0 { 2.2 } else { 1.8 }, shade(CURVATURE, k));
        }
    }

    fn draw_loss_chart(&self, f: &mut Frame, lay: &Layout) {
        let p = self.plot(lay, 2);
        if p.tiny() {
            return;
        }
        p.backing(f, if lay.narrow { 36.0 } else { 46.0 }, 30.0, lay.text + 12.0, 6.0);
        self.x_axis(f, &p, true, lay.small);
        self.log_axis(f, &p, L_LO, L_HI, lay.small);
        let title = if lay.narrow { "the loss on the whole data" } else { "the loss on the whole data (mean squared error)" };
        Self::heading(f, lay, &p, title, &[]);
        let mut gone: Vec<(usize, u64)> = Vec::new();
        for (k, r) in self.runs.iter().enumerate().rev() {
            let mut loss: Vec<(f64, f64)> = r.loss.iter().enumerate().filter(|(_, l)| l.is_finite()).map(|(t, &l)| (t as f64, l)).collect();
            if let Some(b) = r.blown {
                // (it leaves through the top of the chart)
                loss.push((b as f64, 10f64.powf(p.y[1] + 2.0)));
                if b as f64 <= self.shown + 1e-9 {
                    gone.push((k, b));
                }
            }
            reveal(f, &p, LINES + PER_RUN * k as u8 + 4, &loss, true, self.shown, if k == 0 { 2.2 } else { 1.8 }, shade(r.setup.method.color(), k));
        }
        // where a run ran away, at the top
        for (k, b) in gone {
            let x = p.px_x(b as f64);
            let (min, _) = p.rect;
            plot::line_of(f, BLOWN_MARK, Vec2::new(x, min.y), Vec2::new(x, min.y + 8.0), 1.4, NOISE);
            let name = format!("{} run: runs away at step {b}", RUN_NAMES[k]);
            let align = if x > (p.rect.0.x + p.rect.1.x) * 0.5 { Align2::RIGHT_TOP } else { Align2::LEFT_TOP };
            f.caption_px(Vec2::new(x + if align == Align2::LEFT_TOP { 4.0 } else { -4.0 }, min.y + 3.0 + k as f32 * (lay.small + 6.0)), name, lay.small, NOISE, align);
        }
    }

    /// The runs' names, and the film's button.
    fn draw_names(&mut self, f: &mut Frame, lay: &Layout) -> bool {
        for (k, s) in self.setups.iter().enumerate() {
            let at = lay.names + Vec2::new(0.0, k as f32 * (lay.small + 6.0));
            let text = format!("{}: {}", RUN_NAMES[k], s.label(lay.narrow));
            f.text_px(at, text, lay.small, shade(s.method.color(), k), Align2::LEFT_TOP);
        }
        let busy = self.film.playing || self.film.waiting;
        let label = if busy {
            "❚❚ pause"
        } else if self.shown >= self.reach() - 1e-9 {
            "▶ again"
        } else {
            "▶ play"
        };
        if plot::button(f, lay.button, label, lay.small, false) {
            if busy {
                self.film.playing = false;
                self.film.waiting = false;
            } else if self.shown >= self.reach() - 1e-9 {
                self.film = Film { clock: 0.0, playing: true, waiting: false };
                self.shown = 0.0;
            } else {
                self.film.playing = true;
            }
            return true;
        }
        false
    }

    /// The line of the step the film is at, through every chart.
    fn draw_now(&self, f: &mut Frame, lay: &Layout) {
        if !(self.shown > 0.0 && self.shown < self.reach() - 1e-9) {
            return;
        }
        for k in 0..3 {
            if lay.has(k) {
                let p = self.plot(lay, k);
                p.vline(f, NOW, self.shown, 1.0, pal::TEXT.a(0.3), false);
            }
        }
    }
}

impl Scene for Warmup {
    fn view(&self) -> View {
        View::Flat { center: Vec2::ZERO, half_height: 1.0 }
    }

    fn settings(&self, s: &mut ViewSettings) {
        s.grid = false;
        s.bloom = s.bloom.min(0.2);
    }

    fn animated(&self) -> bool {
        self.busy() || self.look.shown != self.goal.as_ref().map(|g| g.1.clone()).unwrap_or_default()
    }

    fn pointer_hint(&self) -> Option<&'static str> {
        Some("the button at the top right: play the runs again")
    }

    fn ui(&mut self, ui: &mut egui::Ui, _: &Gfx) {
        ui::note(
            ui,
            r"Nearly every training run starts with steps that grow from almost nothing (a warm-up) and ends with steps that shrink to almost nothing (a decay). The two ends have different reasons. Here one small network (three tanh layers of 16, 609 weights, fitted to 128 noisy points by squared error) is trained twice from the same starting weights, a bright run and a pale one with settings of their own, and measured as it goes. On a quadratic, gradient descent with step $\eta$ is stable only while every curvature is under $2/\eta$ (scene stepsize), so the sharpness $S$, the largest eigenvalue of the loss's Hessian, sets a ceiling on the step:",
        );
        ui::equation(ui, r"\eta < \frac{2}{S}");
        ui::note(ui, r"At the starting weights $S_0 = 88.3$: a ceiling of 0.0227. A network is not a quadratic, and the sharpness moves as it trains; the charts follow both. With mini-batches the gradient's noise sets a second ceiling, which matters late in training (scene noisefloor).");

        ui::section(ui, "Things to try");
        for (k, t) in TRIALS.iter().enumerate() {
            if ui::experiment(ui, t.button, t.caption) {
                self.set_up(k);
            }
        }

        ui::section(ui, "The runs");
        let mut which = self.editing;
        if ui::choice(ui, &["set the bright run", "set the pale run"], &mut which) {
            self.editing = which.min(1);
        }
        let k = self.editing;
        let s = &mut self.setups[k];
        let mut rule = (s.method == Method::Adam) as usize;
        if ui::choice(ui, &["gradient descent", "Adam"], &mut rule) {
            s.method = if rule == 1 { Method::Adam } else { Method::Gd };
        }
        let mut peak = s.peak as f32;
        if ui::slider_log(ui, r"peak step $\eta$", &mut peak, LEAST_PEAK as f32..=MOST_PEAK as f32) {
            s.peak = peak as f64;
        }
        ui::note(ui, "The step the warm-up climbs to and the decay starts from. The numbers below give it as a multiple of the starting ceiling.");
        let mut warm = s.warmup as usize;
        if ui::slider_int(ui, "warm-up steps", &mut warm, 0..=MOST_WARMUP as usize) {
            s.warmup = warm as u64;
        }
        ui::note(ui, r"Over the warm-up the step climbs in a straight line from 0 to the peak; 0 starts at the peak.");
        let mut decay = match s.decay {
            Decay::Constant => 0,
            Decay::Linear => 1,
            Decay::Cosine => 2,
        };
        if ui::choice(ui, &["then constant", "linear decay", "cosine decay"], &mut decay) {
            s.decay = [Decay::Constant, Decay::Linear, Decay::Cosine][decay.min(2)];
        }
        ui::slider_int_log(ui, "batch size $B$", &mut s.batch, 1..=N);
        ui::note(
            ui,
            r"128 is the whole data: gradient descent. Fewer is SGD, whose gradient is noisy (scene minibatch). Its limit is then not exactly $2/\eta$: Cohen and co-authors (2021) found that with mini-batches the sharpness stays lower, the more so the smaller the batch.",
        );
        let mut steps = s.steps as usize;
        if ui::slider_int_log(ui, "steps in the run", &mut steps, LEAST_STEPS as usize..=MOST_STEPS as usize) {
            s.steps = steps as u64;
        }
        ui.horizontal_wrapped(|ui| {
            if ui.button(ui::label_text(ui, "another order of the batches", 12.5, None, false)).clicked() {
                self.setups[k].seed = self.setups[k].seed.wrapping_add(1);
            }
            if ui.button(ui::label_text(ui, "copy to the other run", 12.5, None, false)).clicked() {
                self.setups[1 - k] = self.setups[k];
            }
        });
        self.refresh();

        ui::section(ui, "Reading the picture");
        ui::key(
            ui,
            &[
                (SGD, "cyan: gradient descent (SGD with mini-batches): its step, and its loss"),
                (ADAM, "orange: Adam"),
                (shade(SGD, 1), "bright and pale: the two runs, each with settings of its own"),
                (CURVATURE, r"green: the sharpness $S$ (for Adam the preconditioned $\tilde S$)"),
                (LIMIT, r"yellow: a limit that the step and the sharpness set each other: the ceiling $2/S$ over the step, the level $2/\eta$ over the sharpness"),
                (NOISE, "red dots: the noise ceiling, with mini-batches"),
            ],
        );
        ui::section(ui, "Words");
        ui::term(ui, r"Step, learning rate $\eta$:", "what the gradient is multiplied by in a step.");
        ui::term(
            ui,
            r"Sharpness $S$:",
            "the largest eigenvalue of the Hessian of the loss on the whole data: the curvature along the most curved direction. Estimated here by Lanczos's method from Hessian-vector products, each estimate starting from the last, until its residual is a thousandth of it.",
        );
        ui::term(ui, r"Ceiling $2/S$, level $2/\eta$:", r"one limit seen from each side: the longest step the sharpness allows, and the largest sharpness the step allows.");
        ui::term(ui, "Warm-up:", r"the step climbing in a straight line from 0 to its peak: $\eta_t = \eta_{\text{peak}}\, t/T_w$ over the first $T_w$ steps.");
        ui::term(
            ui,
            "Decay:",
            r"after the warm-up, the step falling to 0 by the last step: in a straight line, or along half a cosine, $\tfrac12\eta_{\text{peak}}(1 + \cos(\pi s/T))$, $s$ steps after the warm-up, $T$ the steps after it (Loshchilov and Hutter's cosine, without their restarts).",
        );
        ui::term(ui, "Catapult:", r"a jump of the loss during which the sharpness falls, after which training goes on under the ceiling (Lewkowycz and co-authors' name).");
        ui::term(ui, "Runs away:", "here, the loss passes 100 000; the run stops there.");
        ui::term(ui, "SGD, mini-batch:", "gradient descent with the gradient of a few examples, drawn afresh each step (here without putting back, epoch after epoch).");
        ui::term(
            ui,
            r"Preconditioned sharpness $\tilde S$:",
            r"for Adam, the largest eigenvalue of $P^{-1}H$, $P$ the numbers its next step divides each parameter's step by. Adam with $P$ held fixed is stable on a quadratic while $\tilde S < 38/\eta$ (at $\beta_1 = 0.9$).",
        );
        ui::term(
            ui,
            "Noise ceiling:",
            r"in McCandlish and co-authors' model of one SGD step, a step $\varepsilon$ with batch $B$ changes the loss on average by $-\varepsilon|G|^2 + \tfrac12\varepsilon^2(G^\top HG + \operatorname{tr}(H\Sigma)/B)$, with $G$ and $H$ the gradient and Hessian on the whole data and $\Sigma$ the covariance of single examples' gradients. It lowers the loss only for $\varepsilon < 2|G|^2/(G^\top HG + \operatorname{tr}(H\Sigma)/B)$: that is the ceiling, measured exactly nine times a run.",
        );
        ui::term(ui, "Noise floor:", "the level about which a constant step's loss wanders once the noise ceiling has come down to the step.");

        ui::section(ui, "Numbers");
        self.numbers(ui);
        ui::see_also(
            ui,
            &[
                ("eos", "the sharpness at the edge of stability, step by step"),
                ("noisefloor", "why a constant step leaves the loss on a floor"),
                ("stepsize", r"the factor $1 - \eta\lambda$ on a quadratic"),
                ("adam", "Adam's divisors"),
            ],
        );
    }

    fn frame(&mut self, f: &mut Frame) {
        hold_batches(f);
        let lay = Layout::of(f.view_size(), f.top_inset);
        let tiny = lay.view.x < 160.0 || lay.view.y < 120.0;
        if !tiny && self.draw_names(f, &lay) {
            self.refresh();
        }
        self.refresh();
        let budget = clock::frame_budget(f.real_dt, 0.004);
        self.work(&budget);
        let snap = self.canvas != Some(lay.view) && !tiny;
        if snap {
            self.canvas = Some(lay.view);
        }
        let moving = self.follow(f.real_dt, snap);
        self.advance(f.dt, moving);
        let goal = self.goal();
        if goal != self.ticks_for {
            self.ticks_for = goal;
            self.tick_batch = if self.tick_batch == TICKS { TICKS + 1 } else { TICKS };
        }
        if tiny {
            return;
        }
        if lay.has(0) {
            self.draw_step_chart(f, &lay);
        }
        if lay.has(1) {
            self.draw_sharp_chart(f, &lay);
        }
        self.draw_loss_chart(f, &lay);
        self.draw_now(f, &lay);
    }
}

impl Warmup {
    /// The read-outs.
    fn numbers(&self, ui: &mut egui::Ui) {
        ui::readout_colored(ui, r"sharpness at the start $S_0$", sig(self.start[0]), CURVATURE);
        ui::readout_colored(ui, r"the ceiling there $2/S_0$", sig(2.0 / self.start[0]), LIMIT);
        ui::readout_colored(ui, r"Adam's first step: $\tilde S_0$", sig(self.start[1]), CURVATURE);
        ui::readout_colored(ui, r"its ceiling $38/\tilde S_0$", sig(Method::Adam.limit_factor() / self.start[1]), LIMIT);
        for (k, r) in self.runs.iter().enumerate() {
            let s = &r.setup;
            let adam = s.method == Method::Adam;
            let s0 = r.sharp.first().map_or(self.start[adam as usize], |x| x.value);
            ui::readout_series(ui, &format!("the {} run", RUN_NAMES[k]), "", shade(s.method.color(), k));
            ui::readout(ui, "  peak over the starting ceiling", format!("{} times", times(s.peak * s0 / s.method.limit_factor())));
            match (r.blown, r.final_loss()) {
                (Some(b), _) => ui::readout_colored(ui, "  ran away", format!("at step {b}"), NOISE),
                (None, Some(l)) => {
                    let (hi, at) = r.highest();
                    ui::readout(ui, "  highest loss", format!("{} at step {at}", sig(hi)));
                    ui::readout(ui, "  loss at the end", sig(l));
                    if let Some(last) = r.sharp.last() {
                        let name = if adam { r"  $\tilde S$ at the end, and $38/\eta$" } else { r"  $S$ at the end, and $2/\eta$" };
                        let level = r.level(s.steps.saturating_sub(1));
                        let level = if level.is_finite() { sig(level) } else { "none".to_string() };
                        ui::readout(ui, name, format!("{}, {level}", sig(last.value)));
                    }
                }
                _ => ui::readout(ui, "  loss at the end", "still running"),
            }
            if r.blown.is_none() {
                let every = s.sharp_every();
                let worst = r.sharp.iter().map(|x| x.residual).fold(0.0, f64::max);
                ui::readout(ui, "  sharpness measured every", if every == 1 { "step".to_string() } else { format!("{every} steps") });
                ui::readout(ui, "  largest residual of an estimate", format!("{} %", num(100.0 * worst, 2)));
            }
            if let Some(m) = r.noise_marks.last() {
                let at = m.ceiling().map_or("none".to_string(), sig);
                ui::readout(ui, &format!("  noise ceiling at step {}", count(m.t as usize)), at);
                ui::readout(ui, r"  the noise's share of $G^\top HG + \operatorname{tr}(H\Sigma)/B$", format!("{} %", num(100.0 * m.noise_share(), 0)));
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

    /// A run with only its loss, to its end.
    fn quick(p: &Problem, s: Setup) -> Run {
        let mut r = Run::new(s, false);
        r.finish(p);
        r
    }

    /// Experiment k as its button sets it up, both runs measured to their ends.
    fn experiment(k: usize) -> Warmup {
        let mut s = Warmup::create();
        s.set_up(k);
        let Warmup { problem, runs, .. } = &mut s;
        for r in runs.iter_mut() {
            r.finish(problem);
        }
        s
    }

    fn says(k: usize, what: &str) {
        assert!(TRIALS[k].caption.contains(what), "experiment {k} should say \"{what}\": {}", TRIALS[k].caption);
    }

    /// The sharpness estimated at the weights of step t.
    fn sharp_at(r: &Run, t: u64) -> f64 {
        r.sharp.iter().find(|s| s.t == t).map_or(f64::NAN, |s| s.value)
    }

    /// The steps at which the loss is a peak above `above`: higher than the step before
    /// and not lower than the step after.
    fn jumps(r: &Run, above: f64) -> Vec<(usize, f64)> {
        let l = &r.loss;
        (1..l.len().saturating_sub(1)).filter(|&t| l[t] > l[t - 1] && l[t] >= l[t + 1] && l[t] > above).map(|t| (t, l[t])).collect()
    }

    /// The schedule is `math::optim::Schedule` as the controls describe it.
    #[test]
    fn the_schedule_climbs_and_falls_as_described() {
        let s = Setup { warmup: 50, decay: Decay::Cosine, steps: 250, ..Setup::gd(0.2) };
        let f = s.schedule();
        assert_eq!(f.at(0), 0.0);
        assert!(close(f.at(10), 0.2 * 10.0 / 50.0, 1e-12) && close(f.at(50), 0.2, 1e-12));
        assert!(close(f.at(150), 0.5 * 0.2 * (1.0 + (std::f64::consts::PI * 100.0 / 200.0).cos()), 1e-12));
        assert_eq!(f.at(250), 0.0);
        let lin = Setup { decay: Decay::Linear, ..s }.schedule();
        assert!(close(lin.at(150), 0.1, 1e-12) && lin.at(250) == 0.0);
        let flat = Setup::gd(0.2).schedule();
        assert!((0..300).all(|t| flat.at(t) == 0.2));
        // a warm-up as long as the run climbs all the way; the controls' ends are kept
        let long = Setup { warmup: 5000, steps: 40, peak: f64::NAN, batch: 0, ..Setup::gd(0.1) }.kept();
        assert_eq!((long.steps, long.warmup, long.batch), (LEAST_STEPS, LEAST_STEPS, 1));
        assert!(long.peak == 0.01 && long.schedule().at(LEAST_STEPS - 1) < long.peak);
    }

    /// What `next_divisors` finds before a step of Adam is what the step divided by:
    /// the library's preconditioner after it.
    #[test]
    fn adams_next_divisors_are_what_its_step_divides_by() {
        let p = problem();
        let mut run = Run::new(Setup { method: Method::Adam, batch: 16, ..Setup::gd(0.05) }, false);
        let mut predicted = Vec::new();
        let mut reported = Vec::new();
        for _ in 0..6 {
            // the gradient, then the divisors it will be divided by, then the step
            while run.phase != Phase::Step {
                run.unit(&p);
            }
            next_divisors(&run.opt, &run.model, &mut predicted);
            run.unit(&p);
            assert!(run.opt.preconditioner(&mut reported));
            assert_eq!(predicted.len(), reported.len());
            for (a, b) in predicted.iter().zip(&reported) {
                assert!(close(*a, *b, 1e-12), "{a} against {b}");
            }
        }
        // the first divisors are (1 − β1) times the first gradient's size
        let mut fresh = Run::new(Setup { method: Method::Adam, ..Setup::gd(0.05) }, false);
        fresh.unit(&p);
        next_divisors(&fresh.opt, &fresh.model, &mut predicted);
        let g: Vec<f64> = fresh.model.groups().iter().flat_map(|g| g.grad.iter().copied()).collect();
        for (d, g) in predicted.iter().zip(&g) {
            assert!(close(*d, 0.1 * (g.abs() + 1e-8), 1e-9), "{d} against {g}");
        }
    }

    /// The sharpness the scene draws is the largest eigenvalue, to its tolerance:
    /// against an estimate a million times tighter from another start.
    #[test]
    fn the_sharpness_is_the_largest_eigenvalue() {
        let p = problem();
        let mut model = start_model();
        let mut tape = Tape::new();
        let mut tight = Lanczos::new(&model, 1, 99);
        tight.tolerance = 1e-9;
        while !tight.advance(&p, &mut model, &mut tape, 4) && tight.products() < 400 {}
        let s = Warmup::create();
        assert!(close(s.start[0], tight.sharpness(), 1e-3), "{} against {}", s.start[0], tight.sharpness());
        // and Adam's, of P^(−1/2) H P^(−1/2) with the first step's divisors
        let mut run = Run::new(Setup { method: Method::Adam, ..Setup::gd(0.1) }, false);
        run.unit(&p);
        let mut d = Vec::new();
        next_divisors(&run.opt, &run.model, &mut d);
        let mut pre = Lanczos::new(&model, 1, 98).preconditioned(&d);
        pre.tolerance = 1e-9;
        while !pre.advance(&p, &mut model, &mut tape, 4) && pre.products() < 400 {}
        assert!(close(s.start[1], pre.sharpness(), 1e-3), "{} against {}", s.start[1], pre.sharpness());
    }

    /// The noise ceiling is twice McCandlish's best step, the step at which the one-step
    /// model's expected change is 0.
    #[test]
    fn the_noise_ceiling_is_where_the_expected_change_is_zero() {
        let p = problem();
        let mut model = start_model();
        let mut tape = Tape::new();
        let mut fnz = FullNoise::new(&p, &model);
        while !fnz.advance(&p, &mut model, &mut tape, 64) {}
        let one = fnz.one_step();
        let b = effective_batch(8, N, Draw::WithoutReplacement);
        let mark = NoiseMark { t: 0, g2: one.g2, ghg: one.ghg, trhs: one.trhs, batch: b };
        let c = mark.ceiling().unwrap();
        assert!(close(c, 2.0 * one.best_step(b), 1e-12));
        assert!(one.change(c, b).abs() < 1e-9 * one.change(0.5 * c, b).abs());
        assert!(one.change(0.9 * c, b) < 0.0 && one.change(1.1 * c, b) > 0.0);
        // no positive curvature term: no ceiling
        assert!(NoiseMark { ghg: -1.0, trhs: 0.0, ..mark }.ceiling().is_none());
    }

    #[test]
    fn a_long_step_from_the_start() {
        let s = experiment(0);
        let s0 = s.start[0];
        let ceiling = 2.0 / s0;
        says(0, &format!("$2/S_0 = {}$", sig(ceiling)));
        let (a, b) = (&s.runs[0], &s.runs[1]);
        assert_eq!((a.setup.peak, b.setup.peak), (0.09, 0.18));
        says(0, &format!("The bright run, at 0.09 ({} times it)", times(0.09 / ceiling)));
        says(0, &format!("jumps from {} to {} at the first step", sig(a.loss[0]), sig(a.loss[1])));
        assert_eq!(a.highest().1, 1);
        let s1 = sharp_at(a, 1);
        let s2 = sharp_at(a, 2);
        assert!(s1 < sharp_at(a, 0) && s2 < s1 && s2 < a.level(2));
        says(0, &format!("to {} by step 2, under the level $2/\\eta = {}$", sig(s2), sig(a.level(2))));
        says(0, &format!("by step 3 the loss is {}", sig(a.loss[3])));
        assert!(a.final_loss().unwrap() < a.loss[3] && a.blown.is_none());
        says(0, &format!("The pale run, at 0.18 ({} times)", times(0.18 / ceiling)));
        assert_eq!(b.blown, Some(12));
        assert!(b.loss[11] < BLOWN);
        says(0, "passes 100 000 at step 12");
        // the steps tried without a warm-up
        let p = problem();
        let start = a.loss[0];
        let r = quick(&p, Setup::gd(0.03));
        assert!(r.blown.is_none() && r.highest().1 == 0, "0.03");
        for k in 5..=14 {
            let r = quick(&p, Setup::gd(k as f64 / 100.0));
            assert!(r.blown.is_none() && r.highest().0 > start, "{k}");
        }
        for k in [15, 16, 17, 18, 20] {
            assert!(quick(&p, Setup::gd(k as f64 / 100.0)).blown.is_some(), "{k}");
        }
        says(0, &format!("from 0.05 to 0.14 ({} times the ceiling)", times(0.14 / ceiling)));
        says(0, &format!("from 0.15 ({} times) it ran away", times(0.15 / ceiling)));
    }

    #[test]
    fn the_same_peak_warmed_up() {
        let s = experiment(1);
        let (a, b) = (&s.runs[0], &s.runs[1]);
        assert_eq!((a.setup.warmup, b.setup.warmup, a.setup.peak, b.setup.peak), (0, 50, 0.18, 0.18));
        assert_eq!(a.blown, Some(12));
        says(1, "The bright run runs away at step 12");
        // the sharpness falls by itself while the step is short (under its ceiling)
        assert!((0..=5).all(|t| b.lr[t] < b.ceiling(&b.sharp[t]).unwrap()));
        says(1, &format!("(from {} to {} by step 5)", sig(sharp_at(b, 0)), sig(sharp_at(b, 5))));
        // the step passes its ceiling
        let cross = b.sharp.iter().find(|x| b.lr[x.t as usize] > b.ceiling(x).unwrap()).unwrap().t;
        says(1, &format!("passes its ceiling at step {cross}"));
        let (worst, at) = b.sharp.iter().filter(|x| x.t >= 1 && x.t < 50).map(|x| (x.value / b.level(x.t), x.t)).fold((0.0, 0), |m, v| if v.0 > m.0 { v } else { m });
        says(1, &format!("by up to {} times (step {at})", times(worst)));
        let j = jumps(b, 1.0);
        let (t1, l1) = j[0];
        assert!((cross..t1 as u64 + 1).all(|t| sharp_at(b, t) > b.level(t)), "over the level until the jump");
        assert!(sharp_at(b, t1 as u64 + 1) < b.level(t1 as u64 + 1));
        says(1, &format!("until the loss jumps ({} at step {t1}) and the sharpness drops to {}", sig(l1), sig(sharp_at(b, t1 as u64 + 1))));
        let (t2, l2) = j[1];
        says(1, &format!("a smaller jump ({} at step {t2}) takes it to {}", sig(l2), sig(sharp_at(b, t2 as u64 + 1))));
        assert!(l2 < l1 && j.iter().all(|&(t, _)| t <= t2 || b.loss[t] < l2), "{j:?}");
        let later: Vec<f64> = b.sharp.iter().filter(|x| x.t >= 50).map(|x| x.value).collect();
        let (lo, hi) = (later.iter().copied().fold(f64::INFINITY, f64::min), later.iter().copied().fold(0.0, f64::max));
        says(1, &format!("From step 50 on it stays between {} and {}, round $2/\\eta = {}$", sig(lo), sig(hi), sig(b.level(50))));
        assert!(lo < b.level(50) && hi > b.level(50));
    }

    #[test]
    fn a_warm_up_too_short() {
        let s = experiment(2);
        let (a, b) = (&s.runs[0], &s.runs[1]);
        says(2, &format!("A peak of 0.27 ({} times the starting ceiling)", times(0.27 * s.start[0] / 2.0)));
        assert_eq!((a.setup.warmup, b.setup.warmup), (10, 100));
        says(2, &format!("The bright run runs away at step {}", a.blown.unwrap()));
        let (hi, at) = b.highest();
        assert!(at < 100 && b.blown.is_none());
        // the jumps over 1, each within 10 steps of the step passing its ceiling
        let j = jumps(b, 1.0);
        let over = |t: usize| (t.saturating_sub(10)..=t).any(|u| b.lr.get(u).is_some_and(|&lr| lr > b.ceiling(&b.sharp[u]).unwrap()));
        assert!(j.len() == 4 && j.iter().all(|&(t, _)| over(t)), "{j:?}");
        assert!(b.sharp.iter().enumerate().all(|(t, x)| x.t == t as u64));
        says(2, "its loss jumps over 1 four times, each within 10 steps of the step passing its ceiling");
        says(2, &format!("(the highest, {} at step {at})", sig(hi)));
        says(2, &format!("and it trains on, to {}", sig(b.final_loss().unwrap())));
    }

    /// The warm-ups tried at three peaks: the shortest that trained, and where the runs
    /// that trained ended.
    #[test]
    fn the_warm_ups_tried() {
        let p = problem();
        let tried = [0u64, 5, 10, 15, 20, 25, 30, 40, 50, 75, 100, 150, 200];
        let mut finals = Vec::new();
        for (peak, shortest) in [(0.18, 15u64), (0.27, 50), (0.36, 150)] {
            for &w in &tried {
                let r = quick(&p, Setup { warmup: w, steps: 300, ..Setup::gd(peak) });
                assert_eq!(r.blown.is_none(), w >= shortest, "peak {peak}, warm-up {w}");
                finals.extend(r.final_loss());
            }
        }
        says(2, "the shortest that trained was 15 steps for a peak of 0.18, 50 for 0.27 and 150 for 0.36");
        let none = quick(&p, Setup { steps: 300, ..Setup::gd(0.09) }).final_loss().unwrap();
        let (lo, hi) = (finals.iter().copied().fold(f64::INFINITY, f64::min), finals.iter().copied().fold(0.0, f64::max));
        assert!(none < lo);
        says(2, &format!("0.09 with no warm-up ended at {}, under every run warmed up to 0.18, 0.27 or 0.36 ({} to {})", sig(none), sig(lo), sig(hi)));
    }

    #[test]
    fn a_peak_under_the_ceiling() {
        let s = experiment(3);
        let (a, b) = (&s.runs[0], &s.runs[1]);
        assert!(0.01 < 2.0 / s.start[0]);
        for r in [a, b] {
            assert!(r.loss.windows(2).all(|w| w[1] <= w[0]), "the loss rose");
            assert!(r.sharp.iter().all(|x| r.lr.get(x.t as usize).is_none_or(|&lr| lr < r.ceiling(x).unwrap())), "over the ceiling");
        }
        says(3, &format!("the sharpness falls in both, to {} and {}", sig(a.sharp.last().unwrap().value), sig(b.sharp.last().unwrap().value)));
        let (fa, fb) = (a.final_loss().unwrap(), b.final_loss().unwrap());
        assert!(fb > fa);
        says(3, &format!("The pale run ends a little higher, {} against {}", sig(fb), sig(fa)));
    }

    #[test]
    fn adam_with_and_without() {
        let s = experiment(4);
        let (a, b) = (&s.runs[0], &s.runs[1]);
        let s0 = sharp_at(a, 0);
        assert!(close(s0, s.start[1], 1e-9));
        let limit = a.level(0);
        assert!(close(Method::Adam.limit_factor(), 38.0, 1e-12) && close(1.0 - Hyper::default_for(Rule::Adam).beta1, 0.1, 1e-12));
        says(4, &format!("$\\tilde S$ starts high: {}, {} times the bright run's limit of {}", sig(s0), times(s0 / limit), sig(limit)));
        says(4, &format!("Its loss jumps from {} to {} at step 1, then recovers", sig(a.loss[0]), sig(a.loss[1])));
        assert!(a.highest().1 == 1 && a.final_loss().unwrap() < a.loss[0]);
        let worst = b.sharp.iter().filter(|x| x.t >= 1 && x.t <= 20).map(|x| x.value / b.level(x.t)).fold(0.0, f64::max);
        assert!(worst < 0.25, "{worst}");
        assert_eq!(b.highest().1, 0);
        assert!(b.loss[3..=6].windows(2).all(|w| w[1] > w[0]));
        says(4, &format!("it does climb from {} at step 3 to {} at step 6", sig(b.loss[3]), sig(b.loss[6])));
        let (fa, fb) = (a.final_loss().unwrap(), b.final_loss().unwrap());
        assert!(fb > fa && a.setup.steps == 150);
        says(4, &format!("At step 150 the bright run is at {} and the pale at {}", sig(fa), sig(fb)));
        // a peak of 1.0
        let p = problem();
        let none = quick(&p, Setup { method: Method::Adam, steps: 300, ..Setup::gd(1.0) });
        let warm = quick(&p, Setup { method: Method::Adam, warmup: 100, steps: 300, ..Setup::gd(1.0) });
        says(4, &format!("in runs of 300 steps, no warm-up ended at {} and a warm-up of 100 steps at {}", sig(none.final_loss().unwrap()), sig(warm.final_loss().unwrap())));
    }

    #[test]
    fn the_end_constant_against_cosine() {
        let s = experiment(5);
        let (a, b) = (&s.runs[0], &s.runs[1]);
        assert_eq!((a.setup.batch, a.setup.decay, b.setup.decay, a.setup.warmup), (8, Decay::Constant, Decay::Cosine, 50));
        let first = a.noise_marks.first().unwrap();
        assert_eq!(first.t, 0);
        says(5, &format!("{} from the curvature, {} from the noise", sig(a.ceiling(&a.sharp[0]).unwrap()), sig(first.ceiling().unwrap())));
        // the curvature's ceiling after the warm-up
        let ceil: Vec<f64> = a.sharp.iter().filter(|x| x.t > 50).filter_map(|x| a.ceiling(x)).collect();
        let (lo, hi) = (ceil.iter().copied().fold(f64::INFINITY, f64::min), ceil.iter().copied().fold(0.0, f64::max));
        assert!(lo > 0.1);
        says(5, &format!("risen to between {} and {}", num(lo, 2), num(hi, 2)));
        // the bright run's noise ceilings after the warm-up, round its step
        let after: Vec<f64> = a.noise_marks.iter().filter(|m| m.t > 50).filter_map(|m| m.ceiling()).collect();
        assert_eq!(after.len() as u64, NOISE_MARKS - 1);
        let (lo, hi) = (after.iter().copied().fold(f64::INFINITY, f64::min), after.iter().copied().fold(0.0, f64::max));
        assert!(lo < 0.1 && hi > 0.1);
        says(5, &format!("from {} to {}", sig(lo), sig(hi)));
        // and the loss wanders on a floor: as high over the last quarter as over the
        // second, on average; the cosine ends under it
        let mean_over = |r: &Run, from: usize, to: usize| {
            let v: Vec<f64> = r.loss[from..=to].iter().copied().filter(|l| l.is_finite()).collect();
            v.iter().sum::<f64>() / v.len() as f64
        };
        let (second, last) = (mean_over(a, 500, 1000), mean_over(a, 1500, 2000));
        assert!(last / second > 0.75 && last / second < 1.33, "{second} {last}");
        assert!(b.final_loss().unwrap() < last);
        says(5, "the loss wanders on a floor");
        // the pale run's come down with its step
        let bs: Vec<f64> = b.noise_marks.iter().filter(|m| m.t > 50).filter_map(|m| m.ceiling()).collect();
        let n = bs.len();
        assert!(bs[n - 3..].iter().sum::<f64>() < 0.1 * bs[..2].iter().sum::<f64>(), "{bs:?}");
        says(5, &format!("it ends at {}, the bright run at {}", sig(b.final_loss().unwrap()), sig(a.final_loss().unwrap())));
    }

    /// Eight orders of the batches, at batches of 8 (and, slowly, of 32).
    fn orders(batch: usize) -> (Vec<f64>, Vec<f64>) {
        let p = problem();
        let base = trial(5);
        let (mut a, mut b) = (Vec::new(), Vec::new());
        for seed in 1..=8 {
            a.push(quick(&p, Setup { seed, batch, ..base[0] }).final_loss().unwrap());
            b.push(quick(&p, Setup { seed, batch, ..base[1] }).final_loss().unwrap());
        }
        (a, b)
    }

    fn mean(v: &[f64]) -> f64 {
        v.iter().sum::<f64>() / v.len() as f64
    }

    #[test]
    fn eight_orders_of_the_batches() {
        let (a, b) = orders(8);
        let range = |v: &[f64]| (v.iter().copied().fold(f64::INFINITY, f64::min), v.iter().copied().fold(0.0, f64::max));
        let ((alo, ahi), (blo, bhi)) = (range(&a), range(&b));
        says(5, &format!("the constant step ended at {} on average ({} to {}) and the cosine at {} ({} to {})", sig(mean(&a)), sig(alo), sig(ahi), sig(mean(&b)), sig(blo), sig(bhi)));
        let lower = (0..8).filter(|&i| b[i] < a[i]).count();
        says(5, &format!("lower in {lower} of the 8"));
    }

    /// (Slow: run with --ignored.)
    #[test]
    #[ignore]
    fn eight_orders_of_larger_batches() {
        let (a, b) = orders(32);
        says(5, &format!("With batches of 32 they ended level, at {} and {}", sig(mean(&a)), sig(mean(&b))));
        assert!((mean(&a) / mean(&b) - 1.0).abs() < 0.03);
    }

    /// The scene opens on its opening experiment, and its texts say the measured start.
    #[test]
    fn the_scene_opens_on_its_opening_experiment() {
        let opened = Warmup::create();
        let mut pressed = Warmup::create();
        pressed.set_up(3);
        pressed.set_up(OPENING);
        assert_eq!(opened.setups, pressed.setups);
        assert_eq!(opened.setups, trial(OPENING));
        // the numbers of the panel's opening paragraph
        assert_eq!((sig(opened.start[0]), sig(2.0 / opened.start[0])), ("88.3".to_string(), "0.0227".to_string()));
        assert_eq!(start_model().n_params(), 609);
        assert_eq!(problem().len(), N);
    }

    /// Edges of the controls: every number drawn stays finite, and nothing panics.
    #[test]
    fn degenerate_settings_are_told_honestly() {
        let gd = Setup::gd;
        let cases = [
            gd(MOST_PEAK),
            Setup { method: Method::Adam, ..gd(MOST_PEAK) },
            Setup { peak: f64::NAN, batch: 0, steps: 0, warmup: u64::MAX, ..gd(0.1) },
            Setup { batch: 1, steps: LEAST_STEPS, decay: Decay::Linear, warmup: LEAST_STEPS, ..gd(LEAST_PEAK) },
            Setup { method: Method::Adam, batch: 1, steps: 120, decay: Decay::Cosine, warmup: 7, ..gd(0.5) },
            Setup { peak: f64::INFINITY, ..gd(0.1) },
        ];
        for c in cases {
            let mut s = Warmup::create();
            s.setups = [c, gd(0.05)];
            s.refresh();
            let Warmup { problem, runs, .. } = &mut s;
            for r in runs.iter_mut() {
                r.finish(problem);
            }
            let k = s.setups[0];
            assert!(k.peak.is_finite() && k.batch >= 1 && k.steps >= LEAST_STEPS && k.warmup <= k.steps, "{k:?}");
            s.follow(0.016, true);
            let g = s.frame_goal();
            assert!(g.iter().all(|v| v.is_finite()) && g[STEP_HI] > 0.0 && g[S_HI] > g[S_LO] && g[L_HI] > g[L_LO], "{k:?}: {g:?}");
            for r in &s.runs {
                assert!(r.lr.iter().all(|v| v.is_finite()));
                assert!(r.sharp.iter().all(|x| x.value.is_finite() || r.blown.is_some()), "{k:?}");
            }
            s.shown = s.reach();
            let lay = Layout::of(Vec2::new(900.0, 700.0), 0.0);
            for chart in 0..3 {
                let p = s.plot(&lay, chart);
                assert!(p.x.iter().chain(p.y.iter()).all(|v| v.is_finite()));
            }
        }
        // the largest step runs away at once
        let p = problem();
        assert!(quick(&p, gd(MOST_PEAK)).blown.is_some_and(|t| t < 10));
    }

    /// The film: it waits for the axes, plays across the axis in about `FILM` seconds,
    /// never goes back, and comes to rest at the end.
    #[test]
    fn the_film_plays_to_the_end_and_rests() {
        let mut s = Warmup::create();
        s.follow(0.016, true);
        let mut frames = 0;
        let mut last = 0.0;
        while s.animated() {
            s.work(&Budget::new(1.0));
            let moving = s.follow(1.0 / 60.0, false);
            s.advance(1.0 / 60.0, moving);
            assert!(s.shown >= last, "the film went back");
            last = s.shown;
            frames += 1;
            assert!(frames < 60 * 40, "still playing");
        }
        assert_eq!(s.shown, s.reach());
        assert!(s.runs.iter().all(Run::complete));
        assert!((frames as f64 / 60.0 - FILM).abs() < 1.5, "{frames}");
    }

    /// The layout keeps the charts in the window and apart, from a phone to a desktop.
    #[test]
    fn the_layout_fits_every_window() {
        for (w, h) in [(1220.0, 1000.0), (1100.0, 700.0), (1600.0, 1000.0), (390.0, 800.0), (360.0, 640.0), (700.0, 500.0), (500.0, 400.0), (300.0, 300.0), (240.0, 180.0)] {
            let lay = Layout::of(Vec2::new(w, h), 40.0);
            let inside = |(a, b): Rect| a.x >= 0.0 && a.y >= 0.0 && b.x <= w + 0.5 && b.y <= h + 0.5 && a.is_finite() && b.is_finite();
            let apart = |(a, b): Rect, (c, d): Rect| b.x <= c.x || d.x <= a.x || b.y <= c.y || d.y <= a.y;
            assert!(lay.shown >= 1 && lay.has(2));
            let charts: Vec<Rect> = (0..3).filter(|&k| lay.has(k)).map(|k| lay.charts[k]).collect();
            for (i, a) in charts.iter().enumerate() {
                if h >= 300.0 {
                    assert!(inside(*a), "{w}x{h}: chart {i} {a:?}");
                    assert!(a.1.y - a.0.y >= 60.0 && a.1.x - a.0.x > 150.0, "{w}x{h}: {a:?}");
                }
                for b in &charts[i + 1..] {
                    assert!(apart(*a, *b), "{w}x{h}: {a:?} and {b:?} overlap");
                }
                assert!(apart(*a, lay.button), "{w}x{h}: the button");
            }
            if h >= 640.0 {
                assert_eq!(lay.shown, 3, "{w}x{h}");
            }
        }
    }

    /// After each experiment the axes glide to their new ranges and stop, and the film
    /// then plays.
    #[test]
    fn glides_arrive_and_stop() {
        let mut s = Warmup::create();
        s.follow(0.016, true);
        for k in [5, 0, 4, 3] {
            s.set_up(k);
            s.work(&Budget::new(2.0));
            let mut frames = 0;
            while s.follow(1.0 / 60.0, false) {
                frames += 1;
                assert!(frames < 300, "experiment {k} still gliding");
            }
            s.advance(1.0 / 60.0, false);
            assert!(s.film.playing, "experiment {k}");
        }
    }
}

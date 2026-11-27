//! The optimiser playground: any two rules, A and B, on one network, with what the
//! course has measured drawn side by side as they train.
//!
//! Everything measured is the libraries' (`math::optim`, `math::terrain`, `math::arch`).
//! A design is a rule of `optim` with every knob it has (`Hyper`), a schedule and a
//! batch size. Both designs train the same network from the same starting weights on
//! the same data set, and deal their batches from the same seed, so that two designs
//! with the same batch size see the same batches. After every step the loss on all the
//! data is worked out. Every `EVERY` steps a copy of each design's weights is taken and
//! measured in slices, in the time that training leaves over: the sharpness (for the
//! rules that divide by running sizes, the preconditioned sharpness) by Lanczos's
//! method, started from the last estimate, and the simple noise scale from every
//! example's own gradient (a measurement that falls due while another is under way is
//! left out). The singular values of one matrix's present update and the size of every
//! weight matrix come from the optimiser's record of its last step and from the weights
//! themselves. The charts' ranges, and the charts of the update and the weights, glide
//! to what was measured; the curves along the steps grow a point a step.

use super::*;
use crate::clock::{self, Pace, Rest};
use crate::math::arch::{Init, Kind, MlpSpec, Model, Norm, Place, TransformerSpec};
use crate::math::autograd::{Act, Tape};
use crate::math::optim::{self, AuxRule, Convention, Groups, Hyper, Iteration, Optimizer, Ortho, Rule, Scale, Schedule, Scratch, WeightDecay};
use crate::math::terrain::{Draw, Lanczos, PerExample, Problem, Sampler};
use crate::scenes::stats::plot::{self, Glide, Plot, Rect, AXES, CURVES, MARKS, OWN};

pub const DEF: SceneDef = SceneDef {
    id: "optlab",
    title: "Optimiser playground",
    group: GROUP,
    blurb: "Two optimisers on the same network and the same data, with every knob: the loss, the updates, the sharpness and the noise scale side by side.",
    make: |_| Box::new(Lab::create()),
};

/// The two designs' colours, in every chart: the scene's own, since A and B may be
/// the same rule (they lie in the two widest gaps between the rules' colours).
const A_COLOR: Color = Color::hex(0xE879F9);
const B_COLOR: Color = Color::hex(0xA3E635);

fn color_of(k: usize) -> Color {
    if k == 0 {
        A_COLOR
    } else {
        B_COLOR
    }
}

fn letter(k: usize) -> &'static str {
    if k == 0 {
        "$A$"
    } else {
        "$B$"
    }
}

// ----- the task ----------------------------------------------------------------------------

/// What the network is asked to learn (shared by A and B).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Task {
    /// An MLP sorting points of the plane into three spiral arms (cross-entropy).
    Spirals,
    /// An MLP imitating a fixed small network on four numbers (squared error).
    Teacher,
    /// A transformer predicting the next token of runs of tokens said twice.
    Repeats,
}

impl Task {
    const ALL: [Task; 3] = [Task::Spirals, Task::Teacher, Task::Repeats];

    fn mlp(self) -> bool {
        self != Task::Repeats
    }

    fn index(self) -> usize {
        Task::ALL.iter().position(|t| *t == self).unwrap_or(0)
    }
}

/// The network and its data, shared by both designs.
#[derive(Clone, Copy, Debug, PartialEq)]
struct Net {
    task: Task,
    /// Hidden layers (an MLP) or blocks (a transformer).
    depth: usize,
    /// Units in a hidden layer, or the width of the transformer's stream.
    width: usize,
    norm: Norm,
}

/// Examples in the MLPs' data sets, and sequences in the transformer's: sizes at which
/// a step of both designs and their measurements fit a frame's allowance.
const POINTS: usize = 256;
const SEQUENCES: usize = 16;
/// The transformer's sequences: tokens read, and different tokens.
const CONTEXT: usize = 8;
const VOCAB: usize = 8;
const TEACHER_INPUTS: usize = 4;
/// The widths offered.
const WIDTHS: [usize; 3] = [8, 16, 32];
/// The data set is the same for every seed: a seed draws the weights and the batches.
const DATA_SEED: u64 = 11;

impl Net {
    /// The same with every number one the controls offer.
    fn kept(self) -> Net {
        let most = if self.task.mlp() { 4 } else { 3 };
        let width = WIDTHS.iter().copied().min_by_key(|w: &usize| w.abs_diff(self.width)).unwrap_or(16);
        let norm = if !self.task.mlp() && self.norm == Norm::Batch { Norm::Layer } else { self.norm };
        Net { depth: self.depth.clamp(1, most), width, norm, ..self }
    }

    fn problem(&self) -> Problem {
        let mut rng = Rng::new(DATA_SEED);
        match self.task {
            Task::Spirals => Problem::spirals(POINTS, 3, &mut rng),
            Task::Teacher => Problem::teacher(POINTS, TEACHER_INPUTS, &mut rng),
            Task::Repeats => Problem::repeats(SEQUENCES, CONTEXT, VOCAB, &mut rng),
        }
    }

    /// The network at its start, drawn from `seed`. Its activations are smooth (tanh,
    /// GELU): a Hessian-vector product by differences is wrong across a ReLU's kink.
    fn build(&self, seed: u64) -> Model {
        let mut rng = Rng::new(seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) ^ 0x0A7B);
        if self.task.mlp() {
            let (inputs, outputs) = if self.task == Task::Teacher { (TEACHER_INPUTS, 1) } else { (2, 3) };
            let spec = MlpSpec {
                inputs,
                width: self.width,
                depth: self.depth,
                outputs,
                act: Act::Tanh,
                norm: self.norm,
                place: Place::Post,
                residual: false,
                residual_scale: 1.0,
                init: Init::Xavier,
                gain: 1.0,
                bias: true,
            };
            Model::mlp(&spec, &mut rng)
        } else {
            let spec = TransformerSpec {
                vocab: VOCAB,
                context: CONTEXT,
                d_model: self.width,
                heads: 2,
                layers: self.depth,
                d_mlp: 2 * self.width,
                act: Act::Gelu,
                norm: self.norm,
                place: Place::Pre,
                residual: true,
                init_gain: 1.0,
                tied_unembed: false,
            };
            Model::transformer(&spec, &mut rng)
        }
    }

    /// The loss of a network that knows nothing: ln 3 for three classes, ln 8 for
    /// eight tokens, and for the regression the mean square of the targets about their
    /// mean (always answering the mean).
    fn chance(&self, problem: &Problem) -> f64 {
        match self.task {
            Task::Spirals => 3f64.ln(),
            Task::Repeats => (VOCAB as f64).ln(),
            Task::Teacher => {
                let ys: Vec<f64> = (0..problem.len()).filter_map(|k| problem.point(k).map(|(_, y)| y[0])).collect();
                let mean = ys.iter().sum::<f64>() / ys.len().max(1) as f64;
                ys.iter().map(|y| (y - mean) * (y - mean)).sum::<f64>() / ys.len().max(1) as f64
            }
        }
    }

    /// The network in words, for the picture.
    fn summary(&self) -> String {
        let norm = match self.norm {
            Norm::None => String::new(),
            n => format!(", {} normalisation", n.name()),
        };
        match self.task {
            Task::Spirals => format!("an MLP, {} tanh layers of {}{norm}, sorting {POINTS} points into three spiral arms", self.depth, self.width),
            Task::Teacher => format!("an MLP, {} tanh layers of {}{norm}, imitating a fixed network on {POINTS} examples", self.depth, self.width),
            Task::Repeats => format!("a transformer, {} blocks {} wide{norm}, on {SEQUENCES} runs of tokens said twice", self.depth, self.width),
        }
    }
}

// ----- a design: a rule and its knobs ------------------------------------------------------

/// What the step size does after its warm-up.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Shape {
    Constant,
    Linear,
    Cosine,
}

/// One of the two optimisers being compared: the rule, every knob of it (the
/// library's `Hyper`: its `lr` is the step at the schedule's peak), the schedule and
/// the batch size.
#[derive(Clone, Copy, Debug, PartialEq)]
struct Design {
    rule: Rule,
    hyper: Hyper,
    warmup: u64,
    shape: Shape,
    batch: usize,
}

/// The longest run.
const MOST_STEPS: u64 = 2000;

impl Design {
    /// The rule at its source's defaults, at step `lr`, constant, batches of `batch`.
    fn of(rule: Rule, lr: f64, batch: usize) -> Design {
        Design { rule, hyper: Hyper { lr, ..Hyper::default_for(rule) }, warmup: 0, shape: Shape::Constant, batch }
    }

    /// The same with every number in the range its control offers, and finite.
    fn kept(self, examples: usize) -> Design {
        let h = self.hyper;
        let fin = |v: f64, lo: f64, hi: f64, or: f64| if v.is_finite() { v.max(lo).min(hi) } else { or };
        let decay = match h.weight_decay {
            WeightDecay::Coupled(l) => WeightDecay::Coupled(fin(l, 0.0, 1.0, 0.0)),
            WeightDecay::Decoupled(l) => WeightDecay::Decoupled(fin(l, 0.0, 1.0, 0.0)),
            WeightDecay::Scheduled(l) => WeightDecay::Scheduled(fin(l, 0.0, 1.0, 0.0)),
            WeightDecay::None => WeightDecay::None,
        };
        let ortho = match h.ortho {
            Ortho::Polynomial(it) => Ortho::Polynomial(it.with_rounds(it.rounds.clamp(1, 12))),
            Ortho::Exact => Ortho::Exact,
        };
        let hyper = Hyper {
            lr: fin(h.lr, 0.0, 10.0, 1e-3),
            weight_decay: decay,
            clip: if h.clip.is_nan() || h.clip <= 0.0 || h.clip > 100.0 { f64::INFINITY } else { h.clip.max(0.01) },
            momentum: fin(h.momentum, 0.0, 0.9999, 0.9),
            beta1: fin(h.beta1, 0.0, 0.9999, 0.9),
            beta2: fin(h.beta2, 0.0, 0.99999, 0.999),
            alpha: fin(h.alpha, 0.0, 0.99999, 0.99),
            eps: fin(h.eps, 0.0, 1.0, 1e-8),
            ortho,
            damping: fin(h.damping, 0.0, 1.0, 1e-12),
            exponent: fin(h.exponent, 0.0, 1.0, 0.25),
            precondition_every: h.precondition_every.clamp(1, 1000),
            aux: optim::Aux { lr: fin(h.aux.lr, 0.0, 10.0, 3e-4), ..h.aux },
            ..h
        };
        Design { hyper, warmup: self.warmup.min(MOST_STEPS), batch: self.batch.clamp(1, examples.max(1)), ..self }
    }

    /// The step size at each step of a run `total` steps long: the warm-up from 0, then
    /// constant or a decay to 0 at the run's last step.
    fn schedule(&self, total: u64) -> Schedule {
        let peak = self.hyper.lr;
        let after = total.saturating_sub(self.warmup);
        let s = match self.shape {
            Shape::Constant => Schedule::constant(peak),
            Shape::Linear => Schedule::linear(peak, 0.0, after),
            Shape::Cosine => Schedule::cosine(peak, 0.0, after),
        };
        s.with_warmup(self.warmup, 0.0)
    }

    /// Whether the rule divides each weight's step by a running size of its own, so
    /// that its sharpness is the preconditioned one.
    fn preconditioned(&self) -> bool {
        matches!(self.rule, Rule::Adam | Rule::RmsProp | Rule::AdaGrad)
    }

    /// The number c such that, on a quadratic, the rule's step at rate η is stable
    /// while the curvature (for the rules that divide by running sizes, the
    /// curvature after that division, the preconditioner held fixed) is below c / η,
    /// and grows without end above it; `None` for a rule whose step is not a fixed
    /// multiple of the gradient (a sign, an orthogonalisation), and for Shampoo and SOAP,
    /// which have such a level on a sharpness after their matrices, not measured here. (Goh
    /// 2017; Cohen and co-authors 2021, App. B, and 2022, §2 and Table 1; the course
    /// notes, curvature.md A1 to A4. Tested below against the library's own steps.)
    /// Weight decay, which adds about λ to every curvature, is left out.
    fn level(&self) -> Option<f64> {
        let h = &self.hyper;
        let mu = h.momentum;
        let ema = h.convention == Convention::Ema;
        match self.rule {
            Rule::Sgd | Rule::AdaGrad => Some(2.0),
            Rule::RmsProp => Some(2.0 + 2.0 * mu),
            Rule::Momentum => Some(if ema { (2.0 + 2.0 * mu) / (1.0 - mu) } else { 2.0 + 2.0 * mu }),
            Rule::Nesterov => Some(if ema { (2.0 + 2.0 * mu) / ((1.0 - mu) * (1.0 + 2.0 * mu)) } else { (2.0 + 2.0 * mu) / (1.0 + 2.0 * mu) }),
            Rule::Adam => Some((2.0 + 2.0 * h.beta1) / (1.0 - h.beta1)),
            _ => None,
        }
        .filter(|c: &f64| c.is_finite())
    }

    /// The rule's name, with AdamW for Adam with its decay taken off the weights.
    fn name(&self) -> &'static str {
        match (self.rule, self.hyper.weight_decay) {
            (Rule::Adam, WeightDecay::Decoupled(l)) if l > 0.0 => "AdamW",
            (r, _) => r.name(),
        }
    }

    /// The design in a line: "momentum · step 0.1, β = 0.9 · all the data".
    fn summary(&self, examples: usize) -> String {
        let h = &self.hyper;
        let mut s = format!("{} · step {}", self.name(), trim(h.lr));
        let knobs = match self.rule {
            Rule::Momentum | Rule::Nesterov | Rule::Signum => format!(r", $\beta = {}$", trim(h.momentum)),
            Rule::RmsProp if h.momentum > 0.0 => format!(r", $\alpha = {}$, momentum {}", trim(h.alpha), trim(h.momentum)),
            Rule::RmsProp => format!(r", $\alpha = {}$", trim(h.alpha)),
            Rule::Adam | Rule::Lion | Rule::Soap => format!(r", $\beta_1 = {}$, $\beta_2 = {}$", trim(h.beta1), trim(h.beta2)),
            r if r.is_matrix_rule() && r != Rule::Shampoo && r != Rule::Soap => {
                let ortho = match h.ortho {
                    Ortho::Exact => "exact".to_string(),
                    Ortho::Polynomial(it) => format!("{} rounds", it.rounds),
                };
                format!(", momentum {}, {ortho}", trim(h.momentum))
            }
            _ => String::new(),
        };
        s.push_str(&knobs);
        if self.warmup > 0 {
            s.push_str(&format!(" · warm-up {}", self.warmup));
        }
        match self.shape {
            Shape::Constant => {}
            Shape::Linear => s.push_str(" · linear decay"),
            Shape::Cosine => s.push_str(" · cosine decay"),
        }
        match h.weight_decay {
            WeightDecay::Coupled(l) if l > 0.0 => s.push_str(&format!(r" · decay $\lambda = {}$ added to the gradient", trim(l))),
            WeightDecay::Decoupled(l) | WeightDecay::Scheduled(l) if l > 0.0 => s.push_str(&format!(r" · decay $\lambda = {}$ off the weights", trim(l))),
            _ => {}
        }
        if h.clip.is_finite() {
            s.push_str(&format!(" · clipped at {}", trim(h.clip)));
        }
        if self.batch >= examples {
            s.push_str(" · all the data each step");
        } else {
            s.push_str(&format!(" · batch {}", self.batch));
        }
        s
    }
}

/// A number with no more digits than it needs (three significant ones at most), with a
/// true minus sign.
fn trim(v: f64) -> String {
    if !v.is_finite() {
        return "?".to_string();
    }
    if v == 0.0 {
        return "0".to_string();
    }
    let digits = (2 - v.abs().log10().floor() as i32).clamp(0, 8) as usize;
    let mut t = format!("{:.*}", digits, v);
    if t.contains('.') {
        t = t.trim_end_matches('0').trim_end_matches('.').to_string();
    }
    if let Some(rest) = t.strip_prefix('-') {
        t = format!("−{rest}");
    }
    t
}

/// A positive number for a label: plain between 0.01 and 10 000, otherwise a power of
/// ten in a formula.
fn sci(v: f64) -> String {
    if !v.is_finite() {
        return "?".to_string();
    }
    if v == 0.0 {
        return "0".to_string();
    }
    let a = v.abs();
    if (0.01..10_000.0).contains(&a) {
        return trim(v);
    }
    let e = a.log10().floor();
    let m = v / 10f64.powf(e);
    let (m, e) = if (m.abs() * 10.0).round() >= 100.0 { (m / 10.0, e + 1.0) } else { (m, e) };
    format!("${} \\times 10^{{{}}}$", num(m, 1), num(e, 0))
}

// ----- a run -----------------------------------------------------------------------------

/// A measurement of the sharpness or the noise scale, at the weights of one step.
#[derive(Clone, Copy, Debug, PartialEq)]
struct Mark {
    step: u64,
    value: f64,
    /// The estimate's own error, as the library gives it (the sharpness's relative
    /// residual), 0 where there is none.
    residual: f64,
}

/// A measurement under way: a copy of the weights at one step, and how far the
/// estimates of their sharpness and noise scale have got.
#[derive(Clone, Debug)]
struct Probe {
    step: u64,
    model: Model,
    products: usize,
    sharp_done: bool,
    per: PerExample,
    noise_done: bool,
}

/// The most Hessian-vector products one estimate of the sharpness takes: one that has
/// not converged by then is drawn with its residual.
const MOST_PRODUCTS: usize = 40;
/// How many examples' gradients a piece of a measurement takes.
const EXAMPLES_A_PIECE: usize = 32;
/// The weights are measured every so many steps (and at the run's last step).
const EVERY: u64 = 10;

/// What the charts of updates and weights show of a run, worked out from its last step.
#[derive(Clone, Debug, Default, PartialEq)]
struct Seen {
    /// The step it is of.
    step: u64,
    /// The singular values of the chosen matrix's update at that step, largest first.
    sv: Vec<f64>,
    /// Every weight matrix's Frobenius norm and largest singular value.
    frob: Vec<f64>,
    spec: Vec<f64>,
}

/// One design training on the task.
#[derive(Clone, Debug)]
struct Run {
    design: Design,
    seed: u64,
    model: Model,
    opt: Optimizer,
    sampler: Sampler,
    schedule: Schedule,
    steps: u64,
    /// The loss on all the data at the start and after each step.
    losses: Vec<f64>,
    /// The step size each step took, and the size of its change of the weights over
    /// the size of the weights.
    lrs: Vec<f64>,
    rels: Vec<f64>,
    /// The step after which the loss or a weight stopped being a finite number, or the
    /// weights grew past `BLOWN`.
    diverged: Option<u64>,
    sharp: Vec<Mark>,
    noise: Vec<Mark>,
    lanczos: Lanczos,
    probe: Option<Probe>,
    seen: Seen,
    before: Vec<f64>,
    pre: Vec<f64>,
    batch: Vec<usize>,
}

/// The largest the weights may grow (their length, all together) before a run counts
/// as having blown up.
const BLOWN: f64 = 1e8;

impl Run {
    fn new(net: &Net, problem: &Problem, design: Design, seed: u64, total: u64, tape: &mut Tape) -> Run {
        let model = net.build(seed);
        let opt = Optimizer::new(&model, design.rule, design.hyper);
        // (the same seed for both designs: equal batch sizes deal equal batches)
        let sampler = Sampler::new(problem.len(), design.batch, Draw::WithoutReplacement, Rng::new(seed ^ 0x0BA7_C4E5));
        let lanczos = Lanczos::new(&model, 1, seed ^ 0x1A2C);
        let start = problem.loss(&model, tape);
        Run {
            design,
            seed,
            model,
            opt,
            sampler,
            schedule: design.schedule(total),
            steps: 0,
            losses: vec![start],
            lrs: Vec::new(),
            rels: Vec::new(),
            diverged: (!start.is_finite()).then_some(0),
            sharp: Vec::new(),
            noise: Vec::new(),
            lanczos,
            probe: None,
            seen: Seen::default(),
            before: Vec::new(),
            pre: Vec::new(),
            batch: Vec::new(),
        }
    }

    fn done(&self, total: u64) -> bool {
        self.steps >= total || self.diverged.is_some()
    }

    /// One training step on the next batch, then the loss on all the data.
    fn step(&mut self, problem: &Problem, tape: &mut Tape) {
        if self.diverged.is_some() {
            return;
        }
        let lr = self.schedule.at(self.steps);
        self.batch.clear();
        self.batch.extend_from_slice(self.sampler.deal());
        problem.grad(&mut self.model, tape, &self.batch);
        crate::math::terrain::get(&self.model, &mut self.before);
        self.opt.step(&mut self.model, lr);
        let (mut moved, mut size) = (0.0, 0.0);
        let mut at = 0;
        for g in self.model.groups() {
            for (v, b) in g.value.iter().zip(&self.before[at.min(self.before.len())..]) {
                moved += (v - b) * (v - b);
                size += v * v;
            }
            at += g.value.len();
        }
        self.steps += 1;
        let loss = problem.loss(&self.model, tape);
        self.losses.push(loss);
        self.lrs.push(lr);
        self.rels.push(moved.sqrt() / size.sqrt());
        if !(loss.is_finite() && size.is_finite() && size.sqrt() < BLOWN) {
            self.diverged = Some(self.steps);
        }
    }

    /// Start measuring the present weights, unless a measurement is under way.
    fn begin_probe(&mut self, problem: &Problem) {
        if self.probe.is_some() || self.diverged.is_some() {
            return;
        }
        if self.design.preconditioned() {
            // (before the first step there is no preconditioner: the plain sharpness)
            if !self.opt.preconditioner(&mut self.pre) {
                self.pre.clear();
                self.pre.resize(self.model.n_params(), 1.0);
            }
            self.lanczos.set_preconditioner(&self.pre);
        } else {
            self.lanczos.restart();
        }
        // (with batch normalisation one example's gradient depends on the others of its
        // batch, and the per-example statistics describe nothing)
        let noise_done = self.model.mlp_spec().is_some_and(|s| s.norm == Norm::Batch);
        self.probe = Some(Probe { step: self.steps, model: self.model.clone(), products: 0, sharp_done: false, per: PerExample::new(problem, &self.model), noise_done });
    }

    /// A piece of the measurement under way: one Hessian-vector product, or a few
    /// examples' gradients. Returns whether there was anything to do. (One product at a
    /// time, the Ritz values after each: an estimate comes out the same however its
    /// products fall into frames.)
    fn probe_unit(&mut self, problem: &Problem, tape: &mut Tape) -> bool {
        let Some(p) = self.probe.as_mut() else { return false };
        if !p.sharp_done {
            let converged = self.lanczos.advance(problem, &mut p.model, tape, 1);
            p.products += 1;
            if converged || p.products >= MOST_PRODUCTS {
                p.sharp_done = true;
                self.sharp.push(Mark { step: p.step, value: self.lanczos.sharpness(), residual: self.lanczos.residual() });
            }
            return true;
        }
        if !p.noise_done {
            if p.per.advance(problem, &p.model, tape, EXAMPLES_A_PIECE) {
                p.noise_done = true;
                self.noise.push(Mark { step: p.step, value: p.per.noise().simple_noise_scale(), residual: 0.0 });
            }
            return true;
        }
        self.probe = None;
        false
    }

    /// The group indices of the weight matrices (every group but biases, gains and
    /// normalisations' biases), in the model's order.
    fn matrices(model: &Model) -> Vec<usize> {
        model.groups().iter().enumerate().filter(|(_, g)| !matches!(g.kind, Kind::Bias | Kind::Gain | Kind::Shift)).map(|(i, _)| i).collect()
    }

    /// Work out what the charts of updates and weights show, for matrix `chosen` (a
    /// group index).
    fn look_again(&mut self, chosen: usize, scratch: &mut Scratch) {
        let mut seen = Seen { step: self.steps, ..Seen::default() };
        let groups = self.model.groups();
        let mut sv = Vec::new();
        for &i in &Run::matrices(&self.model) {
            let g = &groups[i];
            seen.frob.push(optim::frobenius(&g.value));
            optim::singular_values_into(&g.value, g.rows, g.cols, &mut sv, scratch);
            seen.spec.push(sv.first().copied().unwrap_or(0.0));
        }
        if let Some(g) = groups.get(chosen) {
            let u = self.opt.update(chosen);
            if u.len() == g.rows * g.cols {
                optim::singular_values_into(u, g.rows, g.cols, &mut seen.sv, scratch);
            }
        }
        self.seen = seen;
    }
}

/// The stable rank of a list of singular values: the sum of their squares over the
/// largest's square (between 1 and their number; 0 for none or all zero).
fn stable_rank(sv: &[f64]) -> f64 {
    let top = sv.iter().fold(0.0f64, |a, b| a.max(*b));
    if top.is_nan() || top <= 0.0 {
        return 0.0;
    }
    sv.iter().map(|s| s * s).sum::<f64>() / (top * top)
}

// ----- the experiments -------------------------------------------------------------------

/// The charts, in the order they are placed.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Chart {
    Loss,
    Step,
    Sharpness,
    Noise,
    Update,
    Weights,
}

impl Chart {
    const ALL: [Chart; 6] = [Chart::Loss, Chart::Step, Chart::Sharpness, Chart::Noise, Chart::Update, Chart::Weights];

    fn tab(self) -> &'static str {
        match self {
            Chart::Loss => "loss",
            Chart::Step => "steps",
            Chart::Sharpness => "sharpness",
            Chart::Noise => "noise",
            Chart::Update => "update",
            Chart::Weights => "weights",
        }
    }
}

struct Trial {
    button: &'static str,
    caption: &'static str,
    net: Net,
    a: Design,
    b: Design,
    total: u64,
    /// The chart the lesson is read from, and the matrix whose update is shown.
    fore: Chart,
    layer: &'static str,
    speed: f32,
}

/// The seed every experiment is measured at.
const SEED: u64 = 1;

const TEACHER: Net = Net { task: Task::Teacher, depth: 3, width: 16, norm: Norm::None };
const SPIRALS: Net = Net { task: Task::Spirals, depth: 3, width: 16, norm: Norm::None };
const TRANSFORMER: Net = Net { task: Task::Repeats, depth: 2, width: 16, norm: Norm::Layer };

/// The experiments, in the order of the course's scenes they recall.
const TRIALS: usize = 5;

fn trial(k: usize) -> Trial {
    match k {
        1 => Trial {
            button: "batch 8 against batch 128",
            caption: CAPTIONS[1],
            net: SPIRALS,
            a: Design::of(Rule::Sgd, 0.02, 8),
            b: Design::of(Rule::Sgd, 0.32, 128),
            total: 1000,
            fore: Chart::Noise,
            layer: "layer 2 weights",
            speed: 100.0,
        },
        2 => Trial {
            button: "Muon against Adam on the transformer",
            caption: CAPTIONS[2],
            net: TRANSFORMER,
            a: Design::of(Rule::Muon, 0.02, 8),
            b: Design::of(Rule::Adam, 1e-3, 8),
            total: 500,
            fore: Chart::Update,
            layer: "block 1 MLP in weights",
            speed: 60.0,
        },
        3 => Trial {
            button: "a long step, with and without warm-up",
            caption: CAPTIONS[3],
            net: TEACHER,
            a: Design { warmup: 50, ..Design::of(Rule::Sgd, 0.4, POINTS) },
            b: Design::of(Rule::Sgd, 0.4, POINTS),
            total: 300,
            fore: Chart::Sharpness,
            layer: "layer 2 weights",
            speed: 40.0,
        },
        4 => {
            let adam = Design::of(Rule::Adam, 0.01, 32);
            Trial {
                button: "AdamW against Adam with decay",
                caption: CAPTIONS[4],
                net: Net { norm: Norm::Layer, ..SPIRALS },
                a: Design { hyper: Hyper { weight_decay: WeightDecay::Decoupled(0.01), ..adam.hyper }, ..adam },
                b: Design { hyper: Hyper { weight_decay: WeightDecay::Coupled(0.01), ..adam.hyper }, ..adam },
                total: 1000,
                fore: Chart::Weights,
                layer: "layer 2 weights",
                speed: 100.0,
            }
        }
        _ => Trial {
            button: "plain descent against momentum",
            caption: CAPTIONS[0],
            net: TEACHER,
            a: Design::of(Rule::Momentum, 0.1, POINTS),
            b: Design::of(Rule::Sgd, 0.1, POINTS),
            total: 500,
            fore: Chart::Sharpness,
            layer: "layer 2 weights",
            speed: 60.0,
        },
    }
}

/// The scene opens as the first experiment sets it up, training.
const OPENING: usize = 0;

// ----- the scene -------------------------------------------------------------------------

/// The steps a second training goes at, unless the visitor asks for another speed.
const SPEED: f32 = 60.0;

pub struct Lab {
    net: Net,
    designs: [Design; 2],
    total: u64,
    seed: u64,
    problem: Problem,
    chance: f64,
    runs: [Run; 2],
    running: bool,
    speed: f32,
    pace: Pace,
    /// The rest after a slice of measurement.
    rest: Rest,
    tape: Tape,
    scratch: Scratch,
    /// Which design the panel's controls change.
    editing: usize,
    /// The matrix (a place in `Run::matrices`) whose update is shown.
    layer: usize,
    /// The chart the experiment's lesson is read from, and the chart chosen in the cell
    /// with tabs (where the window has room for only some).
    fore: Chart,
    tab: Chart,
    /// Seconds since the charts of updates and weights were worked out.
    since_seen: f32,
    look: Look,
    canvas: Option<Vec2>,
}

impl Lab {
    fn create() -> Lab {
        let t = trial(OPENING);
        let problem = t.net.problem();
        let mut tape = Tape::new();
        let runs = [Run::new(&t.net, &problem, t.a, SEED, t.total, &mut tape), Run::new(&t.net, &problem, t.b, SEED, t.total, &mut tape)];
        let mut lab = Lab {
            net: t.net,
            designs: [t.a, t.b],
            total: t.total,
            seed: SEED,
            chance: t.net.chance(&problem),
            problem,
            runs,
            running: false,
            speed: SPEED,
            pace: Pace::default(),
            rest: Rest::default(),
            tape,
            scratch: Scratch::new(),
            editing: 0,
            layer: 0,
            fore: t.fore,
            tab: t.fore,
            since_seen: 0.0,
            look: Look::default(),
            canvas: None,
        };
        lab.set_up(OPENING);
        lab
    }

    /// Set up experiment k: every setting it depends on, both runs at their start, and
    /// training.
    fn set_up(&mut self, k: usize) {
        let t = trial(k);
        self.net = t.net;
        self.designs = [t.a, t.b];
        self.total = t.total;
        self.seed = SEED;
        self.speed = t.speed;
        self.fore = t.fore;
        if t.fore != Chart::Loss {
            self.tab = t.fore;
        }
        self.rebuild();
        let model = &self.runs[0].model;
        self.layer = Run::matrices(model).iter().position(|&i| model.groups()[i].name == t.layer).unwrap_or(0);
        self.look_again();
        self.running = true;
    }

    /// Both runs again from the start, as the settings now are.
    fn rebuild(&mut self) {
        self.net = self.net.kept();
        self.total = self.total.clamp(1, MOST_STEPS);
        let problem = self.net.problem();
        self.chance = self.net.chance(&problem);
        self.problem = problem;
        let n = self.problem.len();
        for k in 0..2 {
            self.designs[k] = self.designs[k].kept(n);
            self.runs[k] = Run::new(&self.net, &self.problem, self.designs[k], self.seed, self.total, &mut self.tape);
            self.runs[k].begin_probe(&self.problem);
        }
        let matrices = Run::matrices(&self.runs[0].model).len();
        self.layer = self.layer.min(matrices.saturating_sub(1));
        self.pace.reset();
        if clock::reproducible() {
            self.finish_probes();
        }
        self.look_again();
    }

    /// The settings the panel holds differ from the runs': start again.
    fn apply(&mut self) {
        let n = self.problem.len();
        let same = self.net.kept() == self.runs[0].model_net()
            && (0..2).all(|k| self.designs[k].kept(n) == self.runs[k].design && self.runs[k].seed == self.seed && self.runs[k].schedule == self.designs[k].kept(n).schedule(self.total));
        if !same {
            self.rebuild();
            self.running = true;
        }
    }

    /// Both back to their start (the same seed).
    fn start_again(&mut self) {
        self.rebuild();
    }

    fn steps(&self) -> u64 {
        self.runs[0].steps.max(self.runs[1].steps)
    }

    fn finished(&self) -> bool {
        self.runs.iter().all(|r| r.done(self.total))
    }

    /// One step of both designs (of those still training), and the measurements that
    /// fall due: begun, and with `finish` done at once. Returns whether anything trained.
    fn step_both(&mut self, finish: bool) -> bool {
        let mut any = false;
        for k in 0..2 {
            let r = &mut self.runs[k];
            if r.done(self.total) {
                continue;
            }
            r.step(&self.problem, &mut self.tape);
            any = true;
            if r.steps.is_multiple_of(EVERY) || r.steps == self.total {
                r.begin_probe(&self.problem);
                if finish {
                    while r.probe_unit(&self.problem, &mut self.tape) {}
                }
            }
        }
        any
    }

    /// One step of both, at once (the "one step" button).
    fn one_step(&mut self) {
        self.running = false;
        self.step_both(clock::reproducible());
        self.look_again();
    }

    /// Every measurement under way, done now.
    fn finish_probes(&mut self) {
        for r in &mut self.runs {
            while r.probe_unit(&self.problem, &mut self.tape) {}
        }
    }

    /// Training's share of this frame: so many steps a second of the scene's time,
    /// within the frame's allowance.
    fn train(&mut self, dt: f32, real_dt: f32) {
        if !self.running || dt <= 0.0 {
            return;
        }
        let finish = clock::reproducible();
        let mut turn = self.pace.begin(self.speed, dt, real_dt, 0.003);
        while turn.next() {
            if !self.step_both(finish) {
                self.running = false;
                break;
            }
        }
        self.pace.end(turn);
        if self.finished() {
            self.running = false;
        }
        if !self.running {
            self.pace.reset();
        }
    }

    /// The measurements under way, a slice at a time in what is left of the frame, the
    /// older first; each slice is followed by a rest twice as long as it took, so that
    /// measuring never takes more than a third of the time.
    fn measure(&mut self, real_dt: f32) {
        if self.runs.iter().all(|r| r.probe.is_none()) || !self.rest.over() {
            return;
        }
        let start = clock::now();
        let budget = clock::frame_budget(real_dt, 0.002);
        loop {
            let older = (0..2).filter(|&k| self.runs[k].probe.is_some()).min_by_key(|&k| self.runs[k].probe.as_ref().map_or(u64::MAX, |p| p.step));
            let Some(k) = older else { break };
            self.runs[k].probe_unit(&self.problem, &mut self.tape);
            if budget.spent() {
                break;
            }
        }
        self.rest = Rest::after(clock::now() - start);
    }

    /// Work out the charts of updates and weights again for both.
    fn look_again(&mut self) {
        let chosen = Run::matrices(&self.runs[0].model).get(self.layer).copied().unwrap_or(0);
        for r in &mut self.runs {
            r.look_again(chosen, &mut self.scratch);
        }
        self.since_seen = 0.0;
    }

    /// The same, when they are out of date: at once while nothing trains, and a few
    /// times a second while it does.
    fn look_due(&mut self, real_dt: f32) {
        self.since_seen += real_dt.max(0.0);
        let stale = self.runs.iter().any(|r| r.seen.step != r.steps);
        if stale && (!self.running || clock::reproducible() || self.since_seen >= 0.1) {
            self.look_again();
        }
    }

    fn measuring(&self) -> bool {
        self.runs.iter().any(|r| r.probe.is_some())
    }
}

impl Run {
    /// The network this run trains (as `Net` describes it).
    fn model_net(&self) -> Net {
        match self.model.arch() {
            crate::math::arch::Arch::Mlp(s) => Net { task: if s.inputs == 2 { Task::Spirals } else { Task::Teacher }, depth: s.depth, width: s.width, norm: s.norm },
            crate::math::arch::Arch::Transformer(s) => Net { task: Task::Repeats, depth: s.layers, width: s.d_model, norm: s.norm },
        }
    }
}

// ----- the layout ------------------------------------------------------------------------

/// Where everything goes in a window (logical pixels from the top-left of the picture).
#[derive(Clone, Debug, PartialEq)]
struct Layout {
    view: Vec2,
    narrow: bool,
    /// Text sizes: titles, words, small print.
    big: f32,
    text: f32,
    small: f32,
    /// The two designs' lines, the network's, and the row of the run's buttons.
    lines: [Vec2; 3],
    room: f32,
    buttons: Vec2,
    /// The charts' cells; the one with tabs shows the chart chosen among the rest.
    cells: Vec<(Rect, Option<Chart>)>,
    tabbed: Vec<Chart>,
}

impl Layout {
    fn of(view: Vec2, inset: f32) -> Layout {
        let view = view.max(Vec2::splat(1.0));
        let narrow = view.x < 720.0;
        let (big, text, small) = if narrow { (12.5, 11.5, 10.5) } else { (13.5, 12.5, 11.0) };
        let side = if narrow { 8.0 } else { 14.0 };
        let top = inset.max(0.0) + if narrow { 6.0 } else { 10.0 };
        let (x0, x1) = (side, view.x - side);
        // the header: on a phone the buttons first, then the lines; otherwise the lines
        // with the buttons beside the first
        let step = text + 7.0;
        let (buttons, first) = if narrow { (Vec2::new(x0, top), top + 30.0) } else { (Vec2::new(x1, top), top + 2.0) };
        let lines = [Vec2::new(x0, first), Vec2::new(x0, first + step), Vec2::new(x0, first + 2.0 * step)];
        let room = if narrow { x1 - x0 } else { (x1 - x0 - 330.0).max(80.0) };
        let grid_top = lines[2].y + step + if narrow { 4.0 } else { 8.0 };
        let (gw, gh) = (x1 - x0, view.y - 8.0 - grid_top);
        let cols = ((gw / 330.0).floor() as usize).clamp(1, 3);
        let rows = ((gh / 215.0).floor() as usize).max(1);
        let n = Chart::ALL.len();
        let slots = cols * rows;
        let (own, tabbed): (Vec<Chart>, Vec<Chart>) = if slots >= n {
            (Chart::ALL.to_vec(), Vec::new())
        } else {
            // the loss always, then the rest in order; the last cell has tabs
            let fixed = slots.saturating_sub(1);
            let own: Vec<Chart> = Chart::ALL.iter().copied().take(fixed).collect();
            let rest: Vec<Chart> = Chart::ALL.iter().copied().filter(|c| !own.contains(c)).collect();
            (own, rest)
        };
        let mut order: Vec<Option<Chart>> = own.iter().map(|c| Some(*c)).collect();
        if !tabbed.is_empty() {
            // (on a phone the chart chosen comes first, the loss under it)
            if narrow {
                order.insert(0, None);
            } else {
                order.push(None);
            }
        }
        let shown = order.len();
        let rows_used = shown.div_ceil(cols).max(1);
        let gap = if narrow { 8.0 } else { 12.0 };
        let cw = (gw - gap * (cols - 1) as f32) / cols as f32;
        let ch = ((gh - gap * (rows_used - 1) as f32) / rows_used as f32).min(if narrow { 360.0 } else { 440.0 });
        let r = |a: f32, b: f32, c: f32, d: f32| (Vec2::new(a, b), Vec2::new(c.max(a + 1.0), d.max(b + 1.0)));
        let cells = order
            .into_iter()
            .enumerate()
            .map(|(k, c)| {
                let (i, j) = (k / cols, k % cols);
                let (cx, cy) = (x0 + j as f32 * (cw + gap), grid_top + i as f32 * (ch + gap));
                let wide = if k + 1 == shown && j + 1 < cols { (cols - j) as f32 * (cw + gap) - gap } else { cw };
                (r(cx, cy, cx + wide, cy + ch), c)
            })
            .collect();
        Layout { view, narrow, big, text, small, lines, room, buttons, cells, tabbed }
    }

    /// Too small a window to show more than words.
    fn tiny(&self) -> bool {
        self.view.x < 200.0 || self.view.y < 200.0
    }
}

// ----- what is drawn, gliding --------------------------------------------------------------

/// The lowest power of ten the charts reach: anything smaller is drawn on the floor.
const FLOOR: f64 = -14.0;

fn log10_or_floor(v: f64) -> f32 {
    if v > 0.0 && v.is_finite() {
        v.log10().max(FLOOR) as f32
    } else {
        FLOOR as f32
    }
}

#[derive(Clone, Debug, Default)]
struct Look {
    /// Each chart's ranges, [x lo, x hi, y lo, y hi] (y as powers of ten), in the order
    /// of `Chart::ALL`.
    ranges: Vec<Glide>,
    /// Per design: log₁₀ of the update's singular values, and of the matrices' sizes
    /// (the Frobenius norms, then the largest singular values).
    sv: [Glide; 2],
    sizes: [Glide; 2],
    /// Where the ranges are going.
    goals: Vec<Vec<f32>>,
    moving: bool,
}

/// Whole decades round the positive finite numbers given (at least two).
fn decades(vals: impl Iterator<Item = f64>) -> (f32, f32) {
    let (mut lo, mut hi) = (f64::INFINITY, f64::NEG_INFINITY);
    for v in vals {
        if v > 0.0 && v.is_finite() {
            let l = v.log10().max(FLOOR);
            lo = lo.min(l);
            hi = hi.max(l);
        }
    }
    if !(lo.is_finite() && hi.is_finite()) {
        return (-1.0, 1.0);
    }
    let (mut a, mut b) = ((lo - 0.1).floor(), (hi + 0.1).ceil());
    if b - a < 2.0 {
        let mid = 0.5 * (lo + hi);
        a = (mid - 1.0).floor();
        b = a + 2.0;
        if b < hi {
            b = hi.ceil();
        }
    }
    (a as f32, b as f32)
}

impl Lab {
    /// Where each chart's ranges are going.
    fn range_goals(&self) -> Vec<Vec<f32>> {
        let t = self.total as f32;
        let both = || self.runs.iter();
        let loss = decades(both().flat_map(|r| r.losses.iter().copied()).chain([self.chance]));
        let rates = both().flat_map(|r| (0..self.total).step_by((self.total as usize / 200).max(1)).map(|s| r.schedule.at(s)));
        let step = decades(rates.chain(both().flat_map(|r| r.rels.iter().copied())));
        let levels = both().filter_map(|r| r.design.level().map(|c| c / r.design.hyper.lr));
        let sharp = decades(both().flat_map(|r| r.sharp.iter().map(|m| m.value)).chain(levels));
        let batch = both().map(|r| r.sampler.size() as f64);
        let noise = decades(both().flat_map(|r| r.noise.iter().map(|m| m.value)).chain(batch));
        let k = both().map(|r| r.seen.sv.len()).max().unwrap_or(1).max(1) as f32;
        // (singular values below 10⁻⁸ of the largest are rounding: the chart's floor)
        let sv = decades(both().flat_map(|r| {
            let top = r.seen.sv.first().copied().unwrap_or(0.0);
            r.seen.sv.iter().copied().filter(move |v| *v >= 1e-8 * top)
        }));
        let m = both().map(|r| r.seen.frob.len()).max().unwrap_or(1).max(1) as f32;
        let sizes = decades(both().flat_map(|r| r.seen.frob.iter().chain(&r.seen.spec).copied()));
        vec![
            vec![0.0, t, loss.0, loss.1],
            vec![0.0, t, step.0, step.1],
            vec![0.0, t, sharp.0, sharp.1],
            vec![0.0, t, noise.0, noise.1],
            vec![0.5, k + 0.5, sv.0, sv.1],
            vec![-0.5, m - 0.5, sizes.0, sizes.1],
        ]
    }

    /// Bring what is drawn nearer to what was measured.
    fn follow(&mut self, dt: f32, snap: bool) {
        // (while training, new numbers come every step: they are followed more slowly,
        // so that the picture moves on evenly between them)
        let dt = if snap {
            1e6
        } else if self.running {
            dt * 0.3
        } else {
            dt
        };
        let mut moving = false;
        for k in 0..2 {
            let s = &self.runs[k].seen;
            let sv: Vec<f32> = s.sv.iter().map(|v| log10_or_floor(*v)).collect();
            moving |= self.look.sv[k].follow(&sv, dt, false);
            let sizes: Vec<f32> = s.frob.iter().chain(&s.spec).map(|v| log10_or_floor(*v)).collect();
            moving |= self.look.sizes[k].follow(&sizes, dt, false);
        }
        let goals = self.range_goals();
        if self.look.ranges.len() != goals.len() {
            self.look.ranges = vec![Glide::default(); goals.len()];
        }
        for (g, to) in self.look.ranges.iter_mut().zip(&goals) {
            moving |= g.follow(to, dt, false);
        }
        self.look.goals = goals;
        self.look.moving = moving;
    }

    fn range_now(&self, chart: Chart) -> [f64; 4] {
        let i = Chart::ALL.iter().position(|c| *c == chart).unwrap_or(0);
        let g = self.look.ranges.get(i);
        [0, 1, 2, 3].map(|k| g.map_or(k as f64 % 2.0, |g| g.get(k) as f64))
    }

    fn range_goal(&self, chart: Chart) -> [f64; 4] {
        let i = Chart::ALL.iter().position(|c| *c == chart).unwrap_or(0);
        let g = self.look.goals.get(i);
        [0, 1, 2, 3].map(|k| g.and_then(|g| g.get(k)).map_or(k as f64 % 2.0, |v| *v as f64))
    }
}

// ----- drawing -----------------------------------------------------------------------------

/// Lines whose number of points changes, each kind in a batch of its own (A's, then
/// B's): the loss, the update's relative size, the step size (dashed), the levels
/// (dashed), the sharpness and noise scale measured, the batch sizes (dashed), the
/// update's singular values, the matrices' sizes (solid: Frobenius; dashed: largest
/// singular value), and the band on the matrix chosen.
const LOSS: u8 = OWN;
const REL: u8 = OWN + 2;
const RATE: u8 = OWN + 4;
const LEVEL: u8 = OWN + 6;
const SHARP: u8 = OWN + 8;
const NOISE_LINE: u8 = OWN + 10;
const BATCH: u8 = OWN + 12;
const SV: u8 = OWN + 14;
const SIZE: u8 = OWN + 16;
const BAND: u8 = OWN + 18;

/// The point batches: the sharpness's dots (A, B), the noise scale's, the update's,
/// the sizes'.
fn dot_batch<'b>(f: &'b mut Frame<'_>, k: usize) -> &'b mut crate::gfx::draw::PointBatch {
    f.dl.points_with(PointKind::Overlay, PointStyle { outline_px: 0.0, size_scale: 1.0 + 1e-4 * (k + 1) as f32, ..Default::default() })
}

/// Every batch the scene draws in, made in the same order in every frame, so that one
/// that is empty in a frame does not move the later ones up a place (`eidos motion`).
fn hold_batches(f: &mut Frame) {
    let corner = f.hud(Vec2::ZERO);
    for kind in [AXES, CURVES, MARKS, plot::BUTTONS].into_iter().chain(OWN..=BAND) {
        plot::apart(f, kind, false).segment(corner, corner, 1.0, Color::CLEAR);
    }
    for kind in [MARKS, RATE, RATE + 1, LEVEL, LEVEL + 1, BATCH, BATCH + 1, SIZE, SIZE + 1] {
        plot::apart(f, kind, true).segment(corner, corner, 1.0, Color::CLEAR);
    }
    for k in 0..8 {
        dot_batch(f, k);
    }
}

/// Dots in screen space, in point batch k.
fn dots_px(f: &mut Frame, k: usize, pts: &[Vec2], size: f32, color: Color) {
    let at: Vec<Vec3> = pts.iter().filter(|p| p.is_finite()).map(|p| f.hud(*p)).collect();
    let b = dot_batch(f, k);
    for p in at {
        b.push(p, size, color);
    }
}

/// A polyline in the batch of its kind; a point that is not a number breaks it. (Not
/// cut at the chart's edge: the points are kept inside it, so that a line has as many
/// points as it has values, and one that gains none from one frame to the next is
/// compared point for point by `eidos motion`.)
fn lines_in(f: &mut Frame, kind: u8, dashed: bool, pts: &[Vec2], width: f32, color: Color) {
    for piece in pts.split(|p| !p.is_finite()).filter(|piece| piece.len() >= 2) {
        let points: Vec<Vec3> = piece.iter().map(|p| f.hud(*p)).collect();
        plot::apart(f, kind, dashed).polyline(points, width, color);
    }
}

/// Words cut to fit `room` pixels, with an ellipsis (never inside a formula).
fn fit(f: &Frame, text: &str, size: f32, room: f32) -> String {
    if f.text_size(text, size).x <= room {
        return text.to_string();
    }
    let mut cuts = Vec::new();
    let mut inside = false;
    for (i, c) in text.char_indices() {
        if !inside {
            cuts.push(i);
        }
        if c == '$' {
            inside = !inside;
        }
    }
    let (mut lo, mut hi) = (0usize, cuts.len().saturating_sub(1));
    while lo < hi {
        let mid = (lo + hi).div_ceil(2);
        if f.text_size(&format!("{}…", &text[..cuts[mid]]), size).x <= room {
            lo = mid;
        } else {
            hi = mid - 1;
        }
    }
    format!("{}…", &text[..cuts.get(lo).copied().unwrap_or(0)])
}

/// Ticks at the powers of ten of a logarithmic y axis, chosen from where the range is
/// going (so that their number does not change while it glides there), drawn where it
/// is now; one that is off the chart waits unseen at its edge.
fn log_axis(f: &mut Frame, lay: &Layout, plot: &Plot, goal: [f64; 4]) {
    let (min, max) = plot.rect;
    plot::line_of(f, AXES, min, Vec2::new(min.x, max.y), 1.0, pal::DIM.a(0.8));
    let (lo, hi) = (goal[2].ceil() as i64, goal[3].floor() as i64);
    let span = (hi - lo).max(1);
    let room = plot.height() / span as f32;
    let every = if room > 24.0 { 1 } else if room > 12.0 { 2 } else { (span / 4).max(3) };
    for e in (FLOOR as i64 - 2)..=31 {
        let y = plot.px_y(e as f64);
        let inside = y >= min.y - 0.5 && y <= max.y + 0.5 && e >= lo && e <= hi;
        let y = if y.is_finite() { y.max(min.y).min(max.y) } else { max.y };
        let (tick, grid) = if inside { (pal::DIM.a(0.8), pal::DIM.a(0.12)) } else { (Color::CLEAR, Color::CLEAR) };
        plot::line_of(f, AXES, Vec2::new(min.x - 4.0, y), Vec2::new(min.x, y), 1.0, tick);
        plot::line_of(f, AXES, Vec2::new(min.x, y), Vec2::new(max.x, y), 1.0, grid);
        if inside && e % every == 0 {
            let text = match e {
                0 => "1".to_string(),
                1 => "10".to_string(),
                _ => format!("$10^{{{}}}$", num(e as f64, 0)),
            };
            f.text_px(Vec2::new(min.x - 6.0, y), text, lay.small, pal::DIM, Align2::RIGHT_CENTER);
        }
    }
}

/// Whole-number ticks along the x axis, chosen from where the range is going.
fn x_ticks(f: &mut Frame, lay: &Layout, plot: &Plot, goal: [f64; 4], name: &str) {
    if plot.tiny() {
        return;
    }
    let (min, max) = plot.rect;
    plot::line_of(f, AXES, Vec2::new(min.x, max.y), max, 1.0, pal::DIM.a(0.8));
    let ticks = plot::Ticks::of(goal[0].ceil().max(0.0), goal[1], plot.width() * 0.8);
    let step = ticks.step.max(1.0);
    let ticks = plot::Ticks::of(goal[0].ceil().max(0.0), goal[1], plot.width() * 0.8 * (ticks.step / step) as f32);
    for v in ticks.values() {
        let x = plot.px_x(v);
        let inside = x >= min.x - 0.5 && x <= max.x + 0.5;
        let x = x.max(min.x).min(max.x);
        plot::line_of(f, AXES, Vec2::new(x, max.y), Vec2::new(x, max.y + 4.0), 1.0, if inside { pal::DIM.a(0.8) } else { Color::CLEAR });
        if inside {
            f.text_px(Vec2::new(x, max.y + 5.0), ticks.name(v), lay.small, pal::DIM, Align2::CENTER_TOP);
        }
    }
    if !name.is_empty() {
        f.text_px(Vec2::new(max.x, max.y + 5.0 + lay.small + 2.0), name, lay.small, pal::DIM, Align2::RIGHT_TOP);
    }
}

/// How many lines of words go under a chart.
const LINES_BELOW: usize = 2;

impl Lab {
    /// The rectangle a chart's data is drawn in, inside its cell.
    fn plot_rect(lay: &Layout, (min, max): Rect, chart: Chart, tabbed: bool) -> Rect {
        let top = min.y + if tabbed { 30.0 } else { 0.0 } + lay.big + if chart == Chart::Update { 34.0 } else { 24.0 };
        let below = 18.0 + lay.small + LINES_BELOW as f32 * (lay.small + 5.0);
        let (left, right) = (min.x + 44.0, max.x - 12.0);
        (Vec2::new(left, top), Vec2::new(right.max(left + 1.0), (max.y - below).max(top + 1.0)))
    }

    /// The run's buttons, the tabs, and the update chart's arrows.
    fn buttons(&mut self, f: &mut Frame, lay: &Layout) {
        let size = lay.small;
        let words = [if self.running { "❚❚ pause" } else { "▶ train" }, "one step", "↺ start again", "new seed"];
        let widest = ["❚❚ pause", "one step", "↺ start again", "new seed"];
        let mut rects = plot::button_row(f, &widest, lay.buttons.x, lay.buttons.y, 22.0, size, 6.0);
        let first = f.text_size("▶ train", size).x.max(f.text_size("❚❚ pause", size).x) + 18.0;
        let shift = first - (rects[0].1.x - rects[0].0.x);
        for (k, r) in rects.iter_mut().enumerate() {
            if k == 0 {
                r.1.x = r.0.x + first;
            } else {
                r.0.x += shift;
                r.1.x += shift;
            }
        }
        if !lay.narrow {
            // (right-aligned on a desktop)
            let width = rects.last().map_or(0.0, |r| r.1.x) - lay.buttons.x;
            for r in &mut rects {
                r.0.x -= width;
                r.1.x -= width;
            }
        }
        if plot::button(f, rects[0], words[0], size, self.running) {
            if self.running {
                self.running = false;
            } else {
                if self.finished() {
                    self.start_again();
                }
                self.running = true;
            }
        }
        if plot::button(f, rects[1], words[1], size, false) {
            self.one_step();
        }
        if plot::button(f, rects[2], words[2], size, false) {
            self.start_again();
            self.running = true;
        }
        if plot::button(f, rects[3], words[3], size, false) {
            self.seed = self.seed.wrapping_add(1);
            self.rebuild();
            self.running = true;
        }
        for (cell, chart) in lay.cells.clone() {
            let tabbed = chart.is_none();
            let chart = chart.unwrap_or(self.tab);
            if tabbed {
                let names: Vec<&str> = lay.tabbed.iter().map(|c| c.tab()).collect();
                let rects = plot::even_row(names.len(), cell.0.x + 6.0, cell.1.x - 6.0, cell.0.y + 6.0, 22.0, 4.0);
                for (k, r) in rects.into_iter().enumerate() {
                    let c = lay.tabbed[k];
                    if plot::button(f, r, names[k], size, self.tab == c) {
                        self.tab = c;
                    }
                }
            }
            if chart == Chart::Update {
                let y = cell.0.y + if tabbed { 36.0 } else { 6.0 };
                let m = Run::matrices(&self.runs[0].model).len();
                let right = cell.1.x - 8.0;
                let next = (Vec2::new(right - 24.0, y), Vec2::new(right, y + 20.0));
                let back = (Vec2::new(right - 52.0, y), Vec2::new(right - 28.0, y + 20.0));
                if plot::button(f, back, "◀", size, false) && m > 0 {
                    self.layer = (self.layer + m - 1) % m;
                    self.look_again();
                }
                if plot::button(f, next, "▶", size, false) && m > 0 {
                    self.layer = (self.layer + 1) % m;
                    self.look_again();
                }
            }
        }
    }

    /// A click on the weights chart chooses the matrix whose update is shown.
    fn clicks(&mut self, f: &mut Frame, lay: &Layout) {
        let Some(at) = plot::pointer(f) else { return };
        for (cell, chart) in &lay.cells {
            let tabbed = chart.is_none();
            if chart.unwrap_or(self.tab) != Chart::Weights {
                continue;
            }
            let p = Lab::plot_rect(lay, *cell, Chart::Weights, tabbed);
            if !plot::within(at, p, 6.0) {
                continue;
            }
            f.handles.claim_clicks();
            if f.ptr.clicked {
                let x = self.range_now(Chart::Weights);
                let pl = Plot::new(p, [x[0], x[1]], [0.0, 1.0]);
                let m = Run::matrices(&self.runs[0].model).len();
                let k = pl.x_at(at.x).round().max(0.0) as usize;
                self.layer = k.min(m.saturating_sub(1));
                f.ptr.clicked = false;
                self.look_again();
            }
        }
    }

    /// The two designs' lines and the network's.
    fn draw_header(&self, f: &mut Frame, lay: &Layout) {
        let n = self.problem.len();
        for k in 0..2 {
            let at = lay.lines[k];
            let c = color_of(k);
            hud_rect(f, Vec2::new(at.x, at.y + 2.0), Vec2::new(at.x + 9.0, at.y + 11.0), c);
            f.text_px(Vec2::new(at.x + 14.0, at.y), letter(k), lay.text, c, Align2::LEFT_TOP);
            let r = &self.runs[k];
            let state = match r.diverged {
                Some(s) => format!(" · diverged at step {s}"),
                None => String::new(),
            };
            let text = fit(f, &format!("{}{state}", r.design.summary(n)), lay.text, lay.room - 32.0);
            f.text_px(Vec2::new(at.x + 32.0, at.y), text, lay.text, pal::TEXT, Align2::LEFT_TOP);
        }
        let s = self.steps();
        let state = if self.running {
            String::new()
        } else if self.finished() {
            ": done".to_string()
        } else {
            ": paused".to_string()
        };
        let text = format!("Both: {} · step {s} of {}{state}", self.net.summary(), self.total);
        let room = if lay.narrow { lay.room } else { lay.view.x - 2.0 * lay.lines[2].x };
        f.text_px(lay.lines[2], fit(f, &text, lay.small, room), lay.small, pal::DIM, Align2::LEFT_TOP);
    }

    fn draw_chart(&self, f: &mut Frame, lay: &Layout, cell: Rect, chart: Chart, tabbed: bool) {
        hud_panel(f, cell.0, cell.1);
        // (the chart the experiment's lesson is read from is outlined)
        let edge = [cell.0, Vec2::new(cell.1.x, cell.0.y), cell.1, Vec2::new(cell.0.x, cell.1.y), cell.0];
        plot::path_of(f, AXES, &edge, 1.2, if chart == self.fore && lay.tabbed.is_empty() { pal::LINK.a(0.55) } else { Color::CLEAR });
        let rect = Lab::plot_rect(lay, cell, chart, tabbed);
        let title_y = cell.0.y + if tabbed { 36.0 } else { 6.0 } + 10.0;
        let r = self.range_now(chart);
        let plot = Plot::new(rect, [r[0], r[1]], [r[2], r[3]]);
        let title = match chart {
            Chart::Loss => "The loss on all the data".to_string(),
            Chart::Step => "The step size, and how far each step went".to_string(),
            Chart::Sharpness => "The sharpness, and the level the step sets".to_string(),
            Chart::Noise => "The noise scale, and the batch".to_string(),
            Chart::Update => "The update of one matrix".to_string(),
            Chart::Weights => "The size of each weight matrix".to_string(),
        };
        let room = cell.1.x - cell.0.x - if chart == Chart::Update { 76.0 } else { 20.0 };
        f.text_px(Vec2::new(cell.0.x + 10.0, title_y), fit(f, &title, lay.big, room), lay.big, pal::TEXT, Align2::LEFT_CENTER);
        if chart == Chart::Update {
            let model = &self.runs[0].model;
            if let Some(g) = Run::matrices(model).get(self.layer).and_then(|&i| model.groups().get(i)) {
                let name = format!("{}, ${} \\times {}$", g.name, g.rows, g.cols);
                f.text_px(Vec2::new(cell.0.x + 10.0, title_y + lay.big - 1.0), fit(f, &name, lay.small, room), lay.small, pal::DIM, Align2::LEFT_CENTER);
            }
        }
        if plot.tiny() {
            return;
        }
        match chart {
            Chart::Loss => self.draw_loss(f, lay, &plot),
            Chart::Step => self.draw_step(f, lay, &plot),
            Chart::Sharpness => self.draw_sharpness(f, lay, &plot),
            Chart::Noise => self.draw_noise(f, lay, &plot),
            Chart::Update => self.draw_update(f, lay, &plot),
            Chart::Weights => self.draw_weights(f, lay, &plot),
        }
    }

    /// The lines of words under a chart, one per design (or one for both).
    fn below(f: &mut Frame, lay: &Layout, plot: &Plot, k: usize, text: &str, color: Color) {
        let (min, max) = plot.rect;
        let y = max.y + 18.0 + lay.small + k as f32 * (lay.small + 5.0);
        let room = max.x - min.x + 40.0;
        f.text_px(Vec2::new(min.x - 36.0, y), fit(f, text, lay.small, room), lay.small, color, Align2::LEFT_TOP);
    }

    /// Points along the step axis, (step, value) with the value on a logarithmic scale,
    /// kept inside the chart (a value that is not a number breaks the line).
    fn along(plot: &Plot, pts: impl Iterator<Item = (f64, f64)>) -> Vec<Vec2> {
        let (lo, hi) = (plot.y[0], plot.y[1]);
        pts.map(|(s, v)| if v.is_nan() { Vec2::NAN } else { plot.at(s, if v > 0.0 { v.log10().max(lo).min(hi) } else { lo }) }).collect()
    }

    fn draw_loss(&self, f: &mut Frame, lay: &Layout, plot: &Plot) {
        let goal = self.range_goal(Chart::Loss);
        log_axis(f, lay, plot, goal);
        x_ticks(f, lay, plot, goal, "step");
        // the loss of a network that knows nothing
        let chance = self.chance.log10();
        plot.hline(f, MARKS, chance, 1.0, pal::DIM.a(0.7), true);
        let words = if self.net.task == Task::Teacher { "always the mean" } else { "knows nothing" };
        plot.text(f, plot.x[0], chance, Vec2::new(4.0, -3.0), words, lay.small - 0.5, pal::DIM, Align2::LEFT_BOTTOM);
        for k in [1, 0] {
            let r = &self.runs[k];
            let pts = Lab::along(plot, r.losses.iter().enumerate().map(|(s, v)| (s as f64, *v)));
            lines_in(f, LOSS + k as u8, false, &pts, if k == 0 { 1.8 } else { 1.6 }, color_of(k));
        }
        for k in 0..2 {
            let r = &self.runs[k];
            let text = match (r.diverged, r.losses.last()) {
                (Some(s), _) => format!("{}: diverged at step {s}: its loss or its weights stopped being finite numbers", letter(k)),
                (None, Some(l)) => format!("{}: {} at step {}", letter(k), sci(*l), r.steps),
                _ => String::new(),
            };
            Lab::below(f, lay, plot, k, &text, color_of(k));
        }
    }

    fn draw_step(&self, f: &mut Frame, lay: &Layout, plot: &Plot) {
        let goal = self.range_goal(Chart::Step);
        log_axis(f, lay, plot, goal);
        x_ticks(f, lay, plot, goal, "step");
        let t = self.total.max(1);
        let n = 200u64.min(t);
        for k in [1, 0] {
            let r = &self.runs[k];
            // the whole schedule, ahead of the run
            let pts = Lab::along(plot, (0..=n).map(|i| {
                let s = (i * t / n).min(t.saturating_sub(1));
                (s as f64, r.schedule.at(s))
            }));
            lines_in(f, RATE + k as u8, true, &pts, 1.2, color_of(k).a(0.8));
            let pts = Lab::along(plot, r.rels.iter().enumerate().map(|(s, v)| ((s + 1) as f64, *v)));
            lines_in(f, REL + k as u8, false, &pts, if k == 0 { 1.8 } else { 1.6 }, color_of(k));
        }
        Lab::below(f, lay, plot, 0, "solid: how far each step moved the weights, over their size", pal::DIM);
        Lab::below(f, lay, plot, 1, r"dashed: the step size $\eta$ the schedule gives", pal::DIM);
    }

    fn draw_sharpness(&self, f: &mut Frame, lay: &Layout, plot: &Plot) {
        let goal = self.range_goal(Chart::Sharpness);
        log_axis(f, lay, plot, goal);
        x_ticks(f, lay, plot, goal, "step");
        let t = self.total.max(1);
        let n = 200u64.min(t);
        let last = |r: &Run| r.design.level().map(|c| (c, c / r.schedule.at(t.saturating_sub(1))));
        let same = last(&self.runs[0]).is_some() && last(&self.runs[0]) == last(&self.runs[1]);
        for k in [1, 0] {
            let r = &self.runs[k];
            if let Some(c) = r.design.level() {
                let pts = Lab::along(plot, (0..=n).map(|i| {
                    let s = (i * t / n).min(t.saturating_sub(1));
                    let lr = r.schedule.at(s);
                    (s as f64, if lr > 0.0 { c / lr } else { f64::NAN })
                }));
                lines_in(f, LEVEL + k as u8, true, &pts, 1.3, LIMIT.a(0.85));
                // its name at the right-hand end, A's above the line and B's under it
                let lr = r.schedule.at(t.saturating_sub(1));
                let y = if lr > 0.0 { (c / lr).log10() } else { f64::NAN };
                let name = if same { format!(r"${}/\eta$, both", trim(c)) } else { format!(r"{}: ${}/\eta$", letter(k), trim(c)) };
                if !(k == 1 && same) && y.is_finite() && y >= plot.y[0] && y <= plot.y[1] {
                    let at = plot.at(plot.x[1], y);
                    let dy = if k == 0 { -3.0 } else { 3.0 };
                    let align = if k == 0 { Align2::RIGHT_BOTTOM } else { Align2::RIGHT_TOP };
                    f.caption_px(at + Vec2::new(-4.0, dy), name, lay.small - 0.5, LIMIT, align);
                }
            }
            let pts = Lab::along(plot, r.sharp.iter().map(|m| (m.step as f64, m.value)));
            lines_in(f, SHARP + k as u8, false, &pts, 1.4, color_of(k));
            dots_px(f, k, &pts, 4.5, color_of(k));
        }
        for k in 0..2 {
            let r = &self.runs[k];
            let what = if r.design.preconditioned() { " (preconditioned)" } else { "" };
            let mut text = match r.sharp.last() {
                Some(m) => {
                    let ago = r.steps.saturating_sub(m.step);
                    let age = match ago {
                        0 => String::new(),
                        1 => ", 1 step ago".to_string(),
                        _ => format!(", {ago} steps ago"),
                    };
                    format!("{}{what}: {} at step {}{age}", letter(k), sci(m.value), m.step)
                }
                None => format!("{}: measuring…", letter(k)),
            };
            match r.design.level() {
                Some(c) => {
                    let lr = r.schedule.at(r.steps.min(t.saturating_sub(1)));
                    if lr > 0.0 {
                        text.push_str(&format!(" · level {}", sci(c / lr)));
                    }
                }
                None if matches!(r.design.rule, Rule::Shampoo | Rule::Soap) => {
                    text.push_str(&format!(" · no level drawn: {} divides by matrices, and the sharpness it meets is not measured here", r.design.name()));
                }
                None => text.push_str(&format!(" · no level drawn: the step of {} is not a fixed multiple of the gradient", r.design.name())),
            }
            Lab::below(f, lay, plot, k, &text, color_of(k));
        }
    }

    fn draw_noise(&self, f: &mut Frame, lay: &Layout, plot: &Plot) {
        let goal = self.range_goal(Chart::Noise);
        log_axis(f, lay, plot, goal);
        x_ticks(f, lay, plot, goal, "step");
        let same = self.runs[0].sampler.size() == self.runs[1].sampler.size();
        for k in [1, 0] {
            let r = &self.runs[k];
            let b = r.sampler.size() as f64;
            let y = b.log10();
            let inside = y >= plot.y[0] && y <= plot.y[1];
            let yc = y.max(plot.y[0]).min(plot.y[1]);
            let (a, e) = (plot.at(plot.x[0], yc), plot.at(plot.x[1], yc));
            let c = if inside && !(k == 1 && same) { color_of(k).a(0.8) } else { Color::CLEAR };
            lines_in(f, BATCH + k as u8, true, &[a, e], 1.3, c);
            if inside && !(k == 1 && same) {
                let name = if same { format!("batch {}, both", r.sampler.size()) } else { format!("{}'s batch, {}", letter(k), r.sampler.size()) };
                let (dy, align) = if k == 0 { (-3.0, Align2::LEFT_BOTTOM) } else { (3.0, Align2::LEFT_TOP) };
                f.caption_px(a + Vec2::new(4.0, dy), name, lay.small - 0.5, color_of(k), align);
            }
            let pts = Lab::along(plot, r.noise.iter().map(|m| (m.step as f64, m.value)));
            lines_in(f, NOISE_LINE + k as u8, false, &pts, 1.4, color_of(k));
            dots_px(f, 2 + k, &pts, 4.5, color_of(k));
        }
        for k in 0..2 {
            let r = &self.runs[k];
            let text = if self.net.task.mlp() && self.net.norm == Norm::Batch {
                format!("{}: none with batch normalisation: an example's gradient depends on the rest of its batch", letter(k))
            } else {
                match r.noise.last() {
                    Some(m) => format!("{}: {} at step {} · batch {}", letter(k), sci(m.value), m.step, r.sampler.size()),
                    None => format!("{}: measuring…", letter(k)),
                }
            };
            Lab::below(f, lay, plot, k, &text, color_of(k));
        }
    }

    fn draw_update(&self, f: &mut Frame, lay: &Layout, plot: &Plot) {
        let goal = self.range_goal(Chart::Update);
        log_axis(f, lay, plot, goal);
        x_ticks(f, lay, plot, goal, "largest first");
        for k in [1, 0] {
            let ys = &self.look.sv[k].shown;
            let pts: Vec<Vec2> = ys.iter().enumerate().map(|(i, y)| plot.at((i + 1) as f64, (*y as f64).max(plot.y[0]).min(plot.y[1]))).collect();
            let c = color_of(k);
            lines_in(f, SV + k as u8, false, &pts, 1.4, c);
            dots_px(f, 4 + k, &pts, if k == 0 { 5.0 } else { 4.0 }, c);
        }
        for k in 0..2 {
            let r = &self.runs[k];
            let sv = &r.seen.sv;
            let text = if r.steps == 0 {
                format!("{}: no step yet", letter(k))
            } else if sv.iter().all(|v| *v == 0.0) {
                format!("{}: the last step did not move this matrix", letter(k))
            } else {
                let g = Run::matrices(&r.model).get(self.layer).copied().unwrap_or(0);
                // (a matrix rule's other matrices are its fall-back's)
                let rule = if !r.design.rule.is_matrix_rule() || r.opt.as_matrix(g) { r.design.name() } else { r.opt.rule_of(g).name() };
                format!("{}: stable rank {} of {} · largest {} · by {rule}", letter(k), trim(stable_rank(sv)), sv.len(), sci(sv[0]))
            };
            Lab::below(f, lay, plot, k, &text, color_of(k));
        }
    }

    fn draw_weights(&self, f: &mut Frame, lay: &Layout, plot: &Plot) {
        let goal = self.range_goal(Chart::Weights);
        log_axis(f, lay, plot, goal);
        let (min, max) = plot.rect;
        plot::line_of(f, AXES, Vec2::new(min.x, max.y), max, 1.0, pal::DIM.a(0.8));
        let model = &self.runs[0].model;
        let matrices = Run::matrices(model);
        // the matrices named under the axis: an MLP's by layer, a transformer's by block
        let mut last = None;
        for (i, &g) in matrices.iter().enumerate() {
            let gr = &model.groups()[g];
            let name = match (self.net.task.mlp(), gr.kind) {
                (true, Kind::Weight) if gr.part == crate::math::arch::Part::Readout => "out".to_string(),
                (true, _) => gr.layer.to_string(),
                (false, Kind::Embedding | Kind::Position) => "in".to_string(),
                (false, Kind::Unembedding) => "out".to_string(),
                (false, _) => format!("block {}", gr.layer),
            };
            if last.as_ref() == Some(&name) {
                continue;
            }
            let x = plot.px_x(i as f64);
            if x >= min.x - 0.5 && x <= max.x + 0.5 {
                plot::line_of(f, AXES, Vec2::new(x, max.y), Vec2::new(x, max.y + 4.0), 1.0, pal::DIM.a(0.8));
                f.text_px(Vec2::new(x, max.y + 5.0), &name, lay.small, pal::DIM, if self.net.task.mlp() { Align2::CENTER_TOP } else { Align2::LEFT_TOP });
            }
            last = Some(name);
        }
        // the matrix whose update is shown
        let x = plot.px_x(self.layer as f64);
        if x >= min.x - 1.0 && x <= max.x + 1.0 {
            hud_rect_at(f, 1, Vec2::new(x - 5.0, min.y), Vec2::new(x + 5.0, max.y), pal::WHITE.a(0.07));
            plot::line_of(f, BAND, Vec2::new(x, min.y), Vec2::new(x, max.y), 1.0, pal::WHITE.a(0.35));
        } else {
            plot::line_of(f, BAND, min, min, 1.0, Color::CLEAR);
        }
        let m = matrices.len();
        for k in [1, 0] {
            let ys = &self.look.sizes[k].shown;
            let c = color_of(k);
            let at = |i: usize, y: f32| plot.at(i as f64, (y as f64).max(plot.y[0]).min(plot.y[1]));
            let frob: Vec<Vec2> = ys.iter().take(m).enumerate().map(|(i, y)| at(i, *y)).collect();
            let spec: Vec<Vec2> = ys.iter().skip(m).take(m).enumerate().map(|(i, y)| at(i, *y)).collect();
            lines_in(f, SIZE + k as u8, false, &frob, 1.4, c);
            lines_in(f, SIZE + k as u8, true, &spec, 1.2, c.a(0.85));
            let mut dots = frob;
            dots.extend(spec);
            dots_px(f, 6 + k, &dots, if k == 0 { 5.0 } else { 4.0 }, c);
        }
        Lab::below(f, lay, plot, 0, "solid: the Frobenius norm · dashed: the largest singular value", pal::DIM);
        Lab::below(f, lay, plot, 1, "click a matrix to see its update", pal::DIM);
    }
}

// ----- the scene ---------------------------------------------------------------------------

impl Scene for Lab {
    fn view(&self) -> View {
        View::Flat { center: Vec2::ZERO, half_height: 1.0 }
    }

    fn settings(&self, s: &mut ViewSettings) {
        s.grid = false;
        s.bloom = s.bloom.min(0.2);
    }

    fn animated(&self) -> bool {
        self.running || self.look.moving || self.measuring() || self.runs.iter().any(|r| r.seen.step != r.steps)
    }

    fn pointer_hint(&self) -> Option<&'static str> {
        Some("buttons on the picture: click    the weights chart: click a matrix to see its update")
    }

    fn ui(&mut self, ui: &mut egui::Ui, _: &Gfx) {
        self.panel(ui);
        self.apply();
    }

    fn frame(&mut self, f: &mut Frame) {
        hold_batches(f);
        let lay = Layout::of(f.view_size(), f.top_inset);
        if !lay.tabbed.is_empty() && !lay.tabbed.contains(&self.tab) {
            self.tab = lay.tabbed[0];
        }
        let tiny = lay.tiny();
        if !tiny {
            self.buttons(f, &lay);
            self.clicks(f, &lay);
        }
        self.train(f.dt, f.real_dt);
        self.measure(f.real_dt);
        self.look_due(f.real_dt);
        let snap = self.canvas != Some(lay.view) && !tiny;
        if snap {
            self.canvas = Some(lay.view);
        }
        self.follow(f.real_dt, snap);
        if tiny {
            // (a sliver, on a phone whose controls are open over the picture, shows nothing)
            let words = "too small a window for the charts";
            if f.text_size(words, 11.0).x + 16.0 <= lay.view.x {
                let at = Vec2::new(lay.view.x * 0.5, (lay.view.y + f.top_inset) * 0.5);
                f.text_px(at, words, 11.0, pal::DIM, Align2::CENTER_CENTER);
            }
            return;
        }
        self.draw_header(f, &lay);
        for (cell, chart) in lay.cells.clone() {
            self.draw_chart(f, &lay, cell, chart.unwrap_or(self.tab), chart.is_none());
        }
    }
}

// ----- the panel ---------------------------------------------------------------------------

/// What each rule does, in a line.
fn rule_note(rule: Rule) -> &'static str {
    match rule {
        Rule::Sgd => r"$\theta \leftarrow \theta - \eta g$: every weight moves by the step size times its gradient.",
        Rule::Momentum => r"Heavy ball: $v \leftarrow \beta v + g$, then $\theta \leftarrow \theta - \eta v$. Written as an average, $v \leftarrow \beta v + (1 - \beta) g$, it takes the same steps at a step size $1/(1 - \beta)$ times larger.",
        Rule::Nesterov => r"$v \leftarrow \beta v + g$, then $\theta \leftarrow \theta - \eta\,(g + \beta v)$.",
        Rule::AdaGrad => r"Each weight's step is divided by the root of the sum of all its squared gradients so far, so that each weight's step size only shrinks.",
        Rule::RmsProp => r"Each weight's step is divided by the root of a running average (coefficient $\alpha$) of its squared gradient.",
        Rule::Adam => r"A running average of the gradient ($\beta_1$) divided, weight by weight, by the root of a running average of its square ($\beta_2$), both corrected for starting at 0, with $\varepsilon$ added to the root.",
        Rule::SignSgd => r"$\theta \leftarrow \theta - \eta \operatorname{sign}(g)$: every weight moves by exactly $\eta$.",
        Rule::Signum => r"The sign of a running average of the gradient: every weight moves by $\eta$.",
        Rule::Lion => r"The sign of a mix of the running average ($\beta_1$) and the new gradient; the average itself is kept with $\beta_2$.",
        Rule::Muon => r"On each matrix it takes: the momentum, orthogonalised (its singular vectors kept, its singular values pushed towards 1), times a factor for the matrix's shape. The other parameters take Adam.",
        Rule::NorMuon => r"Muon, with each row of the orthogonalised matrix divided by a running size of its own, then scaled to a root mean square of $0.2\,\eta$.",
        Rule::AdaMuon => r"Muon on the signs of the momentum, each entry then divided by a running size of its own, then scaled to a root mean square of $0.2\,\eta$.",
        Rule::SpectralDescent => r"Each matrix's gradient replaced by its exact orthogonal factor, times the sum of its singular values over its number of entries; no momentum.",
        Rule::Scion => r"The exact orthogonal factor of a running average of the gradient, with the weights pulled towards a ball of fixed spectral norm.",
        Rule::Shampoo => r"Each matrix's gradient $G$ multiplied on the left and on the right by inverse roots of the running sums of $G G^\top$ and $G^\top G$.",
        Rule::Soap => r"Adam, run in the coordinates of the eigenvectors of running averages of $G G^\top$ and $G^\top G$.",
    }
}

/// A slider for an f64 setting.
fn knob(ui: &mut egui::Ui, label: &str, v: &mut f64, range: std::ops::RangeInclusive<f32>, log: bool) -> bool {
    let mut x = *v as f32;
    let changed = if log { ui::slider_log(ui, label, &mut x, range) } else { ui::slider(ui, label, &mut x, range) };
    if changed {
        *v = x as f64;
    }
    changed
}

/// The same, with 0 at its left end (off).
fn knob_or_zero(ui: &mut egui::Ui, label: &str, v: &mut f64, smallest: f32, largest: f32) -> bool {
    let mut x = *v as f32;
    let changed = ui::slider_log_or_zero(ui, label, &mut x, smallest, largest);
    if changed {
        *v = x as f64;
    }
    changed
}

/// The shape factors a Muon-family rule can take, with their names (A rows of fan-out
/// by B columns of fan-in written m by n here, since A and B name the designs).
const SCALES: [(Scale, &str); 8] = [
    (Scale::One, "1"),
    (Scale::Keller, r"$\sqrt{\max(1, m/n)}$"),
    (Scale::Bernstein, r"$\sqrt{m/n}$"),
    (Scale::MatchAdamW(0.2), r"$0.2\sqrt{\max(m, n)}$"),
    (Scale::Rms(0.2), "root mean square 0.2"),
    (Scale::Nuclear, "nuclear norm"),
    (Scale::NuclearPerEntry, "nuclear norm per entry"),
    (Scale::Scion(1.0), r"Scion's, $\rho = 1$"),
];

impl Lab {
    fn panel(&mut self, ui: &mut egui::Ui) {
        ui::note(
            ui,
            "Two optimisers, $A$ and $B$, train the same network from the same starting weights on the same data, and, where their batch sizes agree, on the same batches. As they train, what the course has measured is measured for both: the loss, the size of the steps, the sharpness, the noise of the batches, the shape of an update and the size of the weights. Any rule and any setting can be given to either.",
        );
        ui::section(ui, "Things to try");
        for k in 0..TRIALS {
            let t = trial(k);
            if ui::experiment(ui, t.button, t.caption) {
                self.set_up(k);
            }
        }

        ui::section(ui, "The network, for both");
        let net = &mut self.net;
        let mut task = net.task.index();
        if ui::choice(ui, &["MLP: spirals", "MLP: a regression", "transformer"], &mut task) {
            let t = Task::ALL[task.min(2)];
            *net = match t {
                Task::Repeats => TRANSFORMER,
                Task::Teacher => Net { task: t, ..TEACHER },
                Task::Spirals => Net { task: t, ..SPIRALS },
            };
        }
        ui::note(
            ui,
            match net.task {
                Task::Spirals => "An MLP of tanh layers sorts 256 points of the plane into three spiral arms (cross-entropy).",
                Task::Teacher => "An MLP of tanh layers learns to imitate a fixed small network on four numbers, from 256 examples (squared error).",
                Task::Repeats => "A transformer (two heads, GELU) reads 16 sequences of 8 tokens, a run of 4 tokens said twice, and predicts each next token (cross-entropy).",
            },
        );
        let most = if net.task.mlp() { 4 } else { 3 };
        ui::slider_int(ui, if net.task.mlp() { "hidden layers" } else { "blocks" }, &mut net.depth, 1..=most);
        let mut w = WIDTHS.iter().position(|v| *v == net.width).unwrap_or(1);
        if ui::choice(ui, &["8 wide", "16 wide", "32 wide"], &mut w) {
            net.width = WIDTHS[w.min(2)];
        }
        let norms: Vec<Norm> = if net.task.mlp() { Norm::ALL.to_vec() } else { vec![Norm::None, Norm::Layer, Norm::Rms] };
        let names: Vec<&str> = norms.iter().map(|n| if *n == Norm::None { "no normalisation" } else { n.name() }).collect();
        let mut k = norms.iter().position(|n| *n == net.norm).unwrap_or(0);
        if ui::choice(ui, &names, &mut k) {
            net.norm = norms[k.min(norms.len() - 1)];
        }
        ui::note(ui, "The activations are smooth (tanh, GELU), so that the curvature can be measured: a ReLU's kink spoils the differences the Hessian is measured by.");

        ui::section(ui, "The run");
        let mut total = self.total as usize;
        if ui::slider_int_log(ui, "steps in the run", &mut total, 10..=MOST_STEPS as usize) {
            self.total = total as u64;
        }
        ui::slider_log(ui, "steps a second", &mut self.speed, 1.0..=1000.0);
        ui::note(ui, "As many as a frame has time for: the transformer trains more slowly than asked.");
        ui.horizontal_wrapped(|ui| {
            if ui.button(if self.running { "❚❚ pause" } else { "▶ train" }).clicked() {
                if self.running {
                    self.running = false;
                } else {
                    if self.finished() {
                        self.start_again();
                    }
                    self.running = true;
                }
            }
            if ui.button("one step").clicked() {
                self.one_step();
            }
            if ui.button("↺ start again").clicked() {
                self.start_again();
                self.running = true;
            }
            if ui.button(format!("new seed (now {})", self.seed)).clicked() {
                self.seed = self.seed.wrapping_add(1);
            }
        });
        ui::note(ui, "A seed draws the starting weights and the order of the batches; the data stay the same. The captions are for seed 1.");

        ui::section(ui, "The designs");
        let mut editing = self.editing;
        if ui::choice(ui, &["change $A$", "change $B$"], &mut editing) {
            self.editing = editing.min(1);
        }
        ui.horizontal_wrapped(|ui| {
            if ui.button(ui::label_text(ui, "make $B$ a copy of $A$", 12.5, None, false)).clicked() {
                self.designs[1] = self.designs[0];
            }
            if ui.button(ui::label_text(ui, "swap $A$ and $B$", 12.5, None, false)).clicked() {
                self.designs.swap(0, 1);
            }
        });
        let examples = self.problem.len();
        let total = self.total;
        let e = self.editing.min(1);
        design_knobs(ui, &mut self.designs[e], examples, total);

        ui::section(ui, "Reading the picture");
        ui::key(
            ui,
            &[
                (A_COLOR, "orchid: design $A$, in every chart"),
                (B_COLOR, "lime: design $B$"),
                (LIMIT, "yellow, dashed: the level each design's step sets, on the sharpness chart"),
                (pal::WHITE, "white band: the matrix whose update is shown"),
            ],
        );
        ui::note(ui, "The loss: the mean over all the examples of each one's loss, after each step (the rules themselves see only their batches).");
        ui::note(ui, r"The steps: dashed, the step size $\eta$ the schedule gives at each step; solid, how far each step moved all the weights together, over their length, $|\Delta\theta|/|\theta|$ (“Warm-up and decay: the learning-rate schedule”).");
        ui::note(ui, r"The sharpness: the largest eigenvalue of the Hessian of the loss on all the data, measured every 10 steps (less often while training outruns the measuring, which takes the time training leaves over); for AdaGrad, RMSProp and Adam, the largest of $P^{-1}H$, with $P$ what the rule divides each weight's step by. The level is $c/\eta$: the sharpness above which the rule's step, its divisor held fixed, would grow without end on a quadratic (“Step size and curvature”, “Momentum: a ball with weight”, “The edge of stability”).");
        ui::note(ui, r"The noise scale: $B_\text{simple} = \operatorname{tr}\Sigma/|G|^2$, from every example's own gradient, measured with the sharpness: the batch at which, for examples drawn independently, a mini-batch gradient's expected squared length is half noise (“The noise scale and the critical batch size”).");
        ui::note(ui, "The update: the singular values of the chosen matrix's change at the last step, weight decay apart (“Muon: an orthogonal step”, “Muon against Adam on one network”).");
        ui::note(ui, "The weights: the size of each weight matrix, whole and in its most stretched direction (“Weight decay and the effective learning rate”).");

        ui::section(ui, "Words");
        ui::term(ui, "Design:", "a rule with all its settings, a schedule of step sizes and a batch size.");
        ui::term(ui, r"Step size $\eta$:", "the learning rate: what each rule multiplies its step by. Its schedule may start with a warm-up, a straight rise from 0, and may decay to 0 at the run's last step.");
        ui::term(ui, "Batch:", "the examples whose mean gradient a step uses; all of them, here 256 or 16, is full-batch training.");
        ui::term(ui, "Sharpness:", "the largest eigenvalue of the Hessian of the loss: how sharply it curves in its most curved direction.");
        ui::term(ui, "Preconditioned sharpness:", r"the same for $P^{-1}H$, the curvature a rule that divides each weight's step by $P$ meets.");
        ui::term(ui, "Level:", r"$2/\eta$ for plain descent, $(2 + 2\beta)/\eta$ for heavy ball with $v \leftarrow \beta v + g$, $(2 + 2\beta_1)/((1 - \beta_1)\eta)$ for Adam with its divisors $P$ held fixed, on $P^{-1}H$ (AdaGrad and RMSProp likewise, at $2/\eta$ and $(2 + 2\beta)/\eta$; Nesterov's is $(2 + 2\beta)/((1 + 2\beta)\eta)$, and a momentum written as an average has $1/(1 - \beta)$ times its level): on a quadratic, the curvature beyond which the rule's step grows without end. Weight decay, which adds about $\lambda$ to every curvature, is left out.");
        ui::term(ui, "Simple noise scale:", r"the total variance of the examples' gradients over the squared length of their mean, $\operatorname{tr}\Sigma/|G|^2$, a batch size. McCandlish and co-authors' (2018) noise scale weights the noise by the curvature; this simple one is what theirs becomes when the Hessian is a multiple of the identity.");
        ui::term(ui, "Singular values:", "how much a matrix stretches each of its own directions, largest first.");
        ui::term(ui, "Stable rank:", "the sum of the squared singular values over the largest one's square: about how many directions a matrix stretches nearly as much as its most stretched one.");
        ui::term(ui, "Frobenius norm:", "the root of the sum of the squares of a matrix's entries, which is also that of its squared singular values.");
        ui::term(ui, "Weight decay:", r"taken off the weights, $\theta \leftarrow \theta - \eta\lambda\theta$ each step (AdamW's); or added to the gradient, $g + \lambda\theta$, before the rule sees it.");
        ui::term(ui, "Orthogonalised:", "a matrix with its singular vectors kept and every nonzero singular value set to 1, or, by Muon's polynomial, pushed towards 1.");
        ui::term(ui, "Diverged:", "a run whose loss or weights stopped being finite numbers, or whose weights grew past a length of $10^8$: it stops.");

        ui::section(ui, "Numbers");
        ui::readout(ui, "steps", format!("{} of {}", self.steps(), self.total));
        for k in 0..2 {
            let r = &self.runs[k];
            let name = format!("{}: loss on all the data", letter(k));
            let value = match (r.diverged, r.losses.last()) {
                (Some(s), _) => format!("diverged at step {s}"),
                (None, Some(l)) => sci(*l),
                _ => "?".to_string(),
            };
            ui::readout_series(ui, &name, value, color_of(k));
            let what = if r.design.preconditioned() { "  preconditioned sharpness" } else { "  sharpness" };
            if let Some(m) = r.sharp.last() {
                ui::readout(ui, what, format!("{} at step {}", sci(m.value), m.step));
            }
            if let Some(c) = r.design.level() {
                let lr = r.schedule.at(r.steps.min(self.total.saturating_sub(1)));
                ui::readout(ui, &format!(r"  its level ${}/\eta$ now", trim(c)), if lr > 0.0 { sci(c / lr) } else { "none: the step is 0".to_string() });
            }
            if let Some(m) = r.noise.last() {
                ui::readout(ui, "  simple noise scale", format!("{} at step {}", sci(m.value), m.step));
            }
            ui::readout(ui, "  batch", count(r.sampler.size()));
            if !r.seen.sv.is_empty() && r.steps > 0 {
                ui::readout(ui, "  update's stable rank", format!("{} of {}", trim(stable_rank(&r.seen.sv)), r.seen.sv.len()));
            }
        }
        ui::readout(ui, "loss of a network that knows nothing", sci(self.chance));
        ui::readout(ui, "weights and biases", count(self.runs[0].model.n_params()));
        ui::see_also(
            ui,
            &[
                ("muonadam", "Muon and Adam set to the same work, with tables over seeds"),
                ("eos", "the sharpness of a network trained by full-batch descent"),
                ("noisescale", "how large a batch is worth it"),
                ("archlab", "the same kind of bench for the network's design"),
                ("race", "seven rules racing on points of the plane"),
            ],
        );
    }
}

/// The controls of one design: the rule, and the settings that rule has.
fn design_knobs(ui: &mut egui::Ui, d: &mut Design, examples: usize, total: u64) {
    let names: Vec<&str> = Rule::ALL.iter().map(|r| r.name()).collect();
    let mut k = Rule::ALL.iter().position(|r| *r == d.rule).unwrap_or(0);
    if ui::choice(ui, &names, &mut k) {
        let rule = Rule::ALL[k.min(Rule::ALL.len() - 1)];
        // (the rule's own defaults; the clipping, schedule and batch stay)
        let clip = d.hyper.clip;
        *d = Design { rule, hyper: Hyper { clip, ..Hyper::default_for(rule) }, ..*d };
    }
    ui::note(ui, rule_note(d.rule));
    let h = &mut d.hyper;
    knob(ui, r"step size $\eta$ (at its peak)", &mut h.lr, 1e-5..=10.0, true);
    match d.rule {
        Rule::Momentum | Rule::Nesterov => {
            knob(ui, r"momentum $\beta$", &mut h.momentum, 0.0..=0.999, false);
            let mut c = (h.convention == Convention::Ema) as usize;
            if ui::choice(ui, &[r"$v \leftarrow \beta v + g$", r"$v \leftarrow \beta v + (1 - \beta) g$"], &mut c) {
                h.convention = if c == 1 { Convention::Ema } else { Convention::Sum };
            }
        }
        Rule::RmsProp => {
            knob(ui, r"average $\alpha$", &mut h.alpha, 0.0..=0.9999, false);
            knob(ui, r"momentum $\beta$", &mut h.momentum, 0.0..=0.99, false);
        }
        Rule::Adam | Rule::Lion | Rule::Soap => {
            knob(ui, r"$\beta_1$", &mut h.beta1, 0.0..=0.999, false);
            knob(ui, r"$\beta_2$", &mut h.beta2, 0.0..=0.9999, false);
            if d.rule == Rule::Soap {
                let mut every = h.precondition_every as usize;
                if ui::slider_int(ui, "new eigenvectors every so many steps", &mut every, 1..=100) {
                    h.precondition_every = every as u32;
                }
            }
        }
        Rule::Signum => {
            knob(ui, r"momentum $\beta$", &mut h.momentum, 0.0..=0.999, false);
        }
        Rule::Muon | Rule::NorMuon | Rule::AdaMuon | Rule::SpectralDescent | Rule::Scion => {
            knob(ui, r"momentum $\beta$", &mut h.momentum, 0.0..=0.999, false);
            if d.rule == Rule::Muon {
                ui::checkbox(ui, &mut h.nesterov, "Nesterov: orthogonalise $(1 - \\beta) g + \\beta v$");
            }
            if matches!(d.rule, Rule::NorMuon | Rule::AdaMuon) {
                knob(ui, r"$\beta_2$, the running size's", &mut h.beta2, 0.0..=0.9999, false);
            }
            let mut names: Vec<&str> = Iteration::ALL.iter().map(|i| i.name).collect();
            names.push("exact");
            let mut o = match h.ortho {
                Ortho::Exact => names.len() - 1,
                Ortho::Polynomial(it) => Iteration::ALL.iter().position(|i| i.name == it.name).unwrap_or(0),
            };
            ui::note(ui, "How the matrix is orthogonalised: a named polynomial, applied for a number of rounds, or exactly.");
            if ui::choice(ui, &names, &mut o) {
                h.ortho = Iteration::ALL.get(o).map_or(Ortho::Exact, |it| Ortho::Polynomial(*it));
            }
            if let Ortho::Polynomial(it) = h.ortho {
                let mut rounds = it.rounds;
                if ui::slider_int(ui, "rounds", &mut rounds, 1..=12) {
                    h.ortho = Ortho::Polynomial(it.with_rounds(rounds));
                }
            }
            let mut s = SCALES.iter().position(|(s, _)| std::mem::discriminant(s) == std::mem::discriminant(&h.scale)).unwrap_or(0);
            let labels: Vec<&str> = SCALES.iter().map(|(_, n)| *n).collect();
            ui::note(ui, "The factor for the matrix's shape, $m$ rows by $n$ columns:");
            if ui::choice(ui, &labels, &mut s) {
                h.scale = SCALES[s.min(SCALES.len() - 1)].0;
            }
        }
        Rule::Shampoo => {
            let mut p = (h.exponent > 0.375) as usize;
            if ui::choice(ui, &[r"roots $p = 1/4$", r"$p = 1/2$"], &mut p) {
                h.exponent = if p == 1 { 0.5 } else { 0.25 };
            }
            knob_or_zero(ui, r"damping $\varepsilon$", &mut h.damping, 1e-12, 1.0);
        }
        _ => {}
    }
    if matches!(d.rule, Rule::AdaGrad | Rule::RmsProp | Rule::Adam | Rule::Soap | Rule::NorMuon | Rule::AdaMuon) {
        knob_or_zero(ui, r"$\varepsilon$", &mut h.eps, 1e-12, 1.0);
    }
    if d.rule.is_matrix_rule() {
        let mut g = match h.groups {
            Groups::Hidden => 0,
            Groups::HiddenAndInput => 1,
            Groups::Matrices => 2,
        };
        ui::note(ui, "Which matrices it takes: the hidden layers' (the reference Muon's choice), those and an MLP's first layer, or every matrix.");
        if ui::choice(ui, &["hidden", "hidden and the first", "every matrix"], &mut g) {
            h.groups = [Groups::Hidden, Groups::HiddenAndInput, Groups::Matrices][g.min(2)];
        }
        let what = match (d.rule, h.aux.rule) {
            (Rule::Soap, _) => "the other parameters' step (SOAP without rotation)",
            (_, AuxRule::ScaledSign) => "the other parameters' step (a scaled sign)",
            _ => "the other parameters' step (Adam)",
        };
        knob(ui, what, &mut h.aux.lr, 1e-5..=1.0, true);
    }
    let (mut lambda, mut added) = match h.weight_decay {
        WeightDecay::None => (0.0, false),
        WeightDecay::Coupled(l) => (l, true),
        WeightDecay::Decoupled(l) | WeightDecay::Scheduled(l) => (l, false),
    };
    let mut changed = knob_or_zero(ui, r"weight decay $\lambda$", &mut lambda, 1e-5, 1.0);
    if lambda > 0.0 {
        let mut c = added as usize;
        if ui::choice(ui, &["taken off the weights", "added to the gradient"], &mut c) {
            added = c == 1;
            changed = true;
        }
    }
    if changed {
        h.weight_decay = match (lambda > 0.0, added) {
            (false, _) => WeightDecay::None,
            (true, true) => WeightDecay::Coupled(lambda),
            (true, false) => WeightDecay::Decoupled(lambda),
        };
    }
    let mut clip = if h.clip.is_finite() { h.clip } else { 0.0 };
    if knob_or_zero(ui, "clip the gradient's length at", &mut clip, 0.01, 100.0) {
        h.clip = if clip > 0.0 { clip } else { f64::INFINITY };
    }
    ui::note(ui, "At 0, no clipping; otherwise a longer gradient is shortened to this length first.");
    let mut warmup = d.warmup as usize;
    if ui::slider_int(ui, "warm-up steps", &mut warmup, 0..=total as usize) {
        d.warmup = warmup as u64;
    }
    let mut shape = match d.shape {
        Shape::Constant => 0,
        Shape::Linear => 1,
        Shape::Cosine => 2,
    };
    if ui::choice(ui, &["then constant", "linear decay", "cosine decay"], &mut shape) {
        d.shape = [Shape::Constant, Shape::Linear, Shape::Cosine][shape.min(2)];
    }
    ui::slider_int_log(ui, "batch size", &mut d.batch, 1..=examples.max(1));
}

// ----- the captions ------------------------------------------------------------------------

/// What each experiment shows, measured at its own settings and seed (the tests at the
/// end of the file hold every number).
const CAPTIONS: [&str; TRIALS] = [
    r"Plain descent ($B$) and heavy-ball momentum $\beta = 0.9$ ($A$), both at step 0.1 on the regression, with all the data each step. On a steady slope momentum's steps add up to $1/(1 - \beta) = 10$ plain ones: $A$'s loss is under 0.01 from step 108 on, $B$'s from step 428. Yet a plain step ten times as long, 1, blows up at step 10: the sharpness at the start is 7.96, over that step's level $2/\eta = 2$. At 0.1 momentum's level is $3.8/\eta = 38$ and plain descent's 20, and both runs stay under theirs (14.1 and 10.4 at most). Momentum's loss also rises on the way, from 0.075 at step 2 to 0.27 at step 3 (“Momentum: a ball with weight”).",
    r"Plain descent on the spirals: $A$ with batches of 8 at step 0.02, $B$ with batches of 128 at a step 16 times as long, 0.32. The simple noise scale starts at 167, above both batches. Count examples, not steps: $A$ after 1000 steps (8000 examples) is at 1.0247, $B$ after 63 steps (8064 examples) at 1.0246, so here $B$'s 63 steps did the work of $A$'s 1000. Per step $B$ is far ahead (0.18 at step 1000). As $B$ learns its noise scale falls, below its batch at 24 of its 50 measurements after step 500; in McCandlish and co-authors' model (2018), far above the noise scale (their curvature-weighted one, for which the simple one stands in here) doubling the batch barely helps (“The noise scale and the critical batch size”).",
    r"Muon at the reference's step 0.02 ($A$) and Adam at Kingma and Ba's 0.001 ($B$), batches of 8 sequences. The update of block 1's first MLP matrix (32 by 16) at step 500: fifteen of Muon's sixteen singular values lie between 0.020 and 0.032, its step times its shape factor $\sqrt{32/16}$ times 0.69 to 1.13, inside the band (about 0.68 to 1.2) where five rounds of its polynomial leave them (“Muon: an orthogonal step”); Adam's fall from $1.2 \times 10^{-3}$ to $3.7 \times 10^{-5}$ (stable ranks 9.3 and 2.1). Muon takes the hidden matrices only; the rest goes to Adam at 0.0003, as in the reference code. Here Muon's loss falls faster at first (0.42 against 1.51 at step 50) and Adam's is lower at step 500 (0.135 against 0.154): one seed, each rule at its source's step.",
    r"Full-batch descent on the regression, peak step 0.4. The sharpness at the start is 7.96, so on a quadratic with that curvature plain descent would bear only steps under $2/7.96 = 0.25$. $B$ starts at 0.4: its loss falls, then jumps (0.33 at step 1, 0.62 at step 2), and by step 10 its sharpness is 4.0, under its level $2/\eta = 5$; from step 100 on it stays about the level (4.4 to 6.1). $A$ warms up over 50 steps: its loss falls at every step meanwhile, but its sharpness does not follow the falling level down: it is 8.5 at step 70, against 5, and 4.6 at step 80. At a peak of 0.8 $B$ blows up at step 12, $A$ at step 53, and with 200 steps of warm-up at step 138. Gilmer and co-authors (2021) saw warm-up lower the sharpness as the step grew, in image classifiers and transformers far larger than this network (“The edge of stability”).",
    r"Adam at step 0.01 with decay $\lambda = 0.01$ (PyTorch's default for AdamW) on the spirals, with layer normalisation after each hidden layer and batches of 32. $A$ takes the decay off the weights, $\theta \leftarrow \theta - \eta\lambda\theta$ (AdamW); $B$ adds $\lambda\theta$ to the gradient, where Adam divides it, weight by weight, by the root of its running average of squared gradients. For plain descent the two are the same steps; for Adam they are not. After 1000 steps $A$'s hidden matrices are 2.5 to 4.8 in size, $B$'s 1.9 to 2.4, and $B$'s loss is 0.075 against $A$'s 0.0017. Here, scaling a hidden layer's weights and bias together leaves the output as it was, the normalisation's small guard $\varepsilon$ aside: their size only sets how far a step turns them (“Weight decay and the effective learning rate”).",
];

#[cfg(test)]
mod tests {
    use super::*;

    /// An experiment as its button sets it up, trained to its end with every
    /// measurement made at once.
    fn run(k: usize) -> Lab {
        let mut lab = Lab::create();
        lab.set_up(k);
        lab.finish_probes();
        while lab.step_both(true) {}
        lab.running = false;
        lab.look_again();
        lab
    }

    fn print_run(lab: &Lab) {
        for k in 0..2 {
            let r = &lab.runs[k];
            println!(" {} {} div {:?}", letter(k), r.design.summary(lab.problem.len()), r.diverged);
            let at: Vec<String> = [0usize, 1, 2, 3, 4, 5, 6, 8, 10, 15, 20, 30, 50, 62, 63, 100, 150, 200, 250, 300, 400, 500, 700, 1000]
                .iter()
                .filter_map(|&t| r.losses.get(t).map(|v| format!("{t}:{v:.4}")))
                .collect();
            println!("   loss {}", at.join(" "));
            let s: Vec<String> = r.sharp.iter().map(|m| format!("{}:{:.3}", m.step, m.value)).collect();
            println!("   sharp {}", s.join(" "));
            let n: Vec<String> = r.noise.iter().map(|m| format!("{}:{:.0}", m.step, m.value)).collect();
            println!("   noise {}", n.join(" "));
            let first = |x: f64| r.losses.iter().position(|v| *v <= x);
            println!("   first <= 0.1 {:?}, 0.03 {:?}, 0.01 {:?}, 0.003 {:?}", first(0.1), first(0.03), first(0.01), first(0.003));
            println!("   update sv {:?} stable {:.3}", r.seen.sv.iter().map(|v| format!("{v:.3e}")).collect::<Vec<_>>(), stable_rank(&r.seen.sv));
            println!("   frob {:?} spec {:?}", r.seen.frob.iter().map(|v| format!("{v:.3}")).collect::<Vec<_>>(), r.seen.spec.iter().map(|v| format!("{v:.3}")).collect::<Vec<_>>());
            println!("   rel first {:?} last {:?}", r.rels.iter().take(3).map(|v| format!("{v:.2e}")).collect::<Vec<_>>(), r.rels.last());
        }
    }

    /// Every experiment's numbers, printed: where the captions' numbers come from, to
    /// write them again after a change. (Run with `--ignored --nocapture`.)
    #[test]
    #[ignore]
    fn explore_trials() {
        for k in 0..TRIALS {
            let lab = run(k);
            println!("== {}", trial(k).button);
            print_run(&lab);
        }
    }

    fn within(v: f64, lo: f64, hi: f64, what: &str) {
        assert!(v >= lo && v <= hi, "{what}: {v} is not in [{lo}, {hi}]");
    }

    /// The first step at which the loss is at or below `x`, and whether it stays there.
    fn under_from(r: &Run, x: f64) -> Option<(usize, bool)> {
        let t = r.losses.iter().position(|v| *v <= x)?;
        Some((t, r.losses[t..].iter().all(|v| *v <= x)))
    }

    /// Experiment k as its button sets it up, trained `steps` steps, measured at the
    /// start and (with `measure`) every 10 steps.
    fn run_to(k: usize, steps: u64, measure: bool) -> Lab {
        let mut lab = Lab::create();
        lab.set_up(k);
        lab.finish_probes();
        for _ in 0..steps {
            lab.step_both(measure);
        }
        lab.running = false;
        lab.look_again();
        lab
    }

    /// The first steps of every experiment, as its caption tells them (the whole of
    /// each is held by the ignored test named after it).
    #[test]
    fn the_experiments_begin_as_their_captions_say() {
        // plain descent against momentum
        let lab = run_to(0, 3, false);
        let (a, b) = (&lab.runs[0], &lab.runs[1]);
        within(a.sharp[0].value, 7.955, 7.965, "the sharpness at the start");
        within(a.losses[2], 0.074, 0.076, "A at step 2");
        within(a.losses[3], 0.26, 0.275, "A at step 3");
        within(a.design.level().unwrap() / 0.1, 37.99, 38.01, "momentum's level");
        within(b.design.level().unwrap() / 0.1, 19.99, 20.01, "plain descent's level");
        let lab = changed(0, 1, |d| d.hyper.lr = 1.0);
        assert_eq!(lab.runs[1].diverged, Some(10));
        // batch 8 against batch 128
        let lab = run_to(1, 1000, false);
        let (a, b) = (&lab.runs[0], &lab.runs[1]);
        within(a.noise[0].value, 166.5, 167.5, "the noise scale at the start");
        within(a.losses[1000], 1.0245, 1.0249, "A after 1000 steps");
        within(b.losses[63], 1.0244, 1.0248, "B after 63 steps");
        within(b.losses[1000], 0.175, 0.185, "B after 1000 steps");
        // Muon against Adam
        let lab = run_to(2, 50, false);
        within(lab.runs[0].losses[50], 0.415, 0.425, "Muon at step 50");
        within(lab.runs[1].losses[50], 1.505, 1.52, "Adam at step 50");
        // a long step, with and without warm-up
        let lab = run_to(3, 10, true);
        let b = &lab.runs[1];
        within(b.losses[1], 0.33, 0.34, "B at step 1");
        within(b.losses[2], 0.61, 0.62, "B at step 2");
        within(sharp_at(b, 10), 3.99, 4.01, "B's sharpness at step 10");
        assert_eq!(changed(3, 1, |d| d.hyper.lr = 0.8).runs[1].diverged, Some(12));
        assert_eq!(changed(3, 0, |d| d.hyper.lr = 0.8).runs[0].diverged, Some(53));
        // AdamW against Adam with decay
        let lab = run_to(4, 0, false);
        assert_eq!((lab.runs[0].design.name(), lab.runs[1].design.name()), ("AdamW", "Adam"));
    }

    fn sharp_at(r: &Run, step: u64) -> f64 {
        r.sharp.iter().find(|m| m.step == step).map_or(f64::NAN, |m| m.value)
    }

    /// Experiment k with design `which` changed, trained to its end without measuring.
    fn changed(k: usize, which: usize, change: impl Fn(&mut Design)) -> Lab {
        let mut lab = Lab::create();
        lab.set_up(k);
        change(&mut lab.designs[which]);
        lab.apply();
        while lab.step_both(false) {}
        lab
    }

    /// (Slow: run with `--ignored`; the start of it is held by
    /// `the_experiments_begin_as_their_captions_say`.)
    #[test]
    #[ignore]
    fn plain_descent_against_momentum() {
        let lab = run(0);
        let (a, b) = (&lab.runs[0], &lab.runs[1]);
        assert_eq!(under_from(a, 0.01), Some((108, true)));
        assert_eq!(under_from(b, 0.01), Some((428, true)));
        within(a.losses[500], 0.00075, 0.00085, "A at step 500");
        within(b.losses[500], 0.0080, 0.0086, "B at step 500");
        within(a.sharp[0].value, 7.955, 7.965, "the sharpness at the start");
        assert_eq!(a.sharp[0].value, b.sharp[0].value, "the same start");
        // the levels: 3.8/η and 2/η at η = 0.1
        within(a.design.level().unwrap() / 0.1, 37.99, 38.01, "momentum's level");
        within(b.design.level().unwrap() / 0.1, 19.99, 20.01, "plain descent's level");
        let top = |r: &Run| r.sharp.iter().map(|m| m.value).fold(0.0, f64::max);
        within(top(a), 14.0, 14.15, "A's sharpness at most");
        within(top(b), 10.35, 10.45, "B's sharpness at most");
        assert!(a.sharp.iter().all(|m| m.value < 38.0) && b.sharp.iter().all(|m| m.value < 20.0));
        // after a first dip, a rise
        assert!(sharp_at(a, 20) < sharp_at(a, 0) && sharp_at(b, 10) < sharp_at(b, 0));
        assert!(a.sharp.windows(2).filter(|w| w[0].step >= 20).all(|w| w[1].value > w[0].value), "A's sharpness rises from step 20");
        assert!(b.sharp.windows(2).filter(|w| w[0].step >= 10).all(|w| w[1].value > w[0].value), "B's rises from step 10");
        // momentum's loss goes up on the way
        within(a.losses[2], 0.074, 0.076, "A at step 2");
        within(a.losses[3], 0.26, 0.275, "A at step 3");
        // a plain step ten times as long blows up, at step 10
        let lab = changed(0, 1, |d| d.hyper.lr = 1.0);
        assert_eq!(lab.runs[1].diverged, Some(10));
        assert!(lab.runs[0].diverged.is_none());
    }

    /// (Slow: run with `--ignored`; the start of it is held by
    /// `the_experiments_begin_as_their_captions_say`.)
    #[test]
    #[ignore]
    fn batch_8_against_batch_128() {
        let lab = run(1);
        let (a, b) = (&lab.runs[0], &lab.runs[1]);
        within(a.noise[0].value, 166.5, 167.5, "the noise scale at the start");
        assert_eq!(a.noise[0].value, b.noise[0].value);
        assert!(a.noise.iter().all(|m| m.value >= 166.9), "A's noise scale stays at or above its start");
        within(a.losses[1000], 1.0245, 1.0249, "A after 1000 steps");
        within(b.losses[63], 1.0244, 1.0248, "B after 63 steps");
        assert_eq!((1000 * 8, 63 * 128), (8000, 8064));
        within(b.losses[1000], 0.175, 0.185, "B after 1000 steps");
        assert_eq!(b.noise.iter().find(|m| m.value < 128.0).map(|m| m.step), Some(270));
        let late: Vec<&Mark> = b.noise.iter().filter(|m| m.step > 500).collect();
        assert_eq!((late.len(), late.iter().filter(|m| m.value < 128.0).count()), (50, 24));
    }

    /// (Slow: run with `--ignored`; the start of it is held by
    /// `the_experiments_begin_as_their_captions_say`.)
    #[test]
    #[ignore]
    fn muon_against_adam() {
        let lab = run(2);
        let (a, b) = (&lab.runs[0], &lab.runs[1]);
        let model = &a.model;
        let g = &model.groups()[Run::matrices(model)[lab.layer]];
        assert_eq!((g.name.as_str(), g.rows, g.cols), ("block 1 MLP in weights", 32, 16));
        let unit = 0.02 * 2f64.sqrt();
        let sv = &a.seen.sv;
        assert_eq!(sv.len(), 16);
        within(sv[0] / unit, 1.125, 1.135, "Muon's largest, over its step and shape factor");
        within(sv[14] / unit, 0.69, 0.695, "Muon's fifteenth");
        within(sv[0], 0.0315, 0.0322, "Muon's largest");
        within(sv[14], 0.0195, 0.0197, "Muon's fifteenth");
        assert!(sv[15] / unit < 0.2);
        within(stable_rank(sv), 9.3, 9.4, "Muon's stable rank");
        let sb = &b.seen.sv;
        within(sb[0], 1.2e-3, 1.3e-3, "Adam's largest");
        within(sb[15], 3.6e-5, 3.8e-5, "Adam's smallest");
        within(stable_rank(sb), 2.1, 2.16, "Adam's stable rank");
        within(a.losses[50], 0.415, 0.425, "Muon at step 50");
        within(b.losses[50], 1.505, 1.52, "Adam at step 50");
        within(a.losses[500], 0.150, 0.158, "Muon at step 500");
        within(b.losses[500], 0.132, 0.138, "Adam at step 500");
        // Muon takes the hidden matrices alone; the rest is Adam's at 0.0003
        assert_eq!(a.design.hyper.aux.lr, 3e-4);
        for (i, g) in model.groups().iter().enumerate() {
            let hidden = matches!(g.kind, Kind::Query | Kind::Key | Kind::Value | Kind::HeadOut) || (g.kind == Kind::Weight);
            assert_eq!(a.opt.as_matrix(i), hidden, "{}", g.name);
            if !hidden {
                assert_eq!(a.opt.rule_of(i), Rule::Adam, "{}", g.name);
            }
        }
    }

    /// (Slow: run with `--ignored`; the start of it is held by
    /// `the_experiments_begin_as_their_captions_say`.)
    #[test]
    #[ignore]
    fn a_long_step_with_and_without_warm_up() {
        let lab = run(3);
        let (a, b) = (&lab.runs[0], &lab.runs[1]);
        within(b.sharp[0].value, 7.955, 7.965, "the sharpness at the start");
        within(2.0 / b.sharp[0].value, 0.25, 0.252, "the longest step it bears");
        within(b.losses[1], 0.33, 0.34, "B at step 1");
        within(b.losses[2], 0.61, 0.62, "B at step 2");
        within(sharp_at(b, 10), 3.99, 4.01, "B's sharpness at step 10");
        let later: Vec<f64> = b.sharp.iter().filter(|m| m.step >= 100).map(|m| m.value).collect();
        within(later.iter().copied().fold(f64::INFINITY, f64::min), 4.36, 4.4, "B's least from step 100");
        within(later.iter().copied().fold(0.0, f64::max), 6.05, 6.15, "B's most from step 100");
        assert!(a.losses[..=50].windows(2).all(|w| w[1] <= w[0]), "A's loss falls through its warm-up");
        within(sharp_at(a, 70), 8.45, 8.55, "A's sharpness at step 70");
        within(a.schedule.at(70), 0.4, 0.4, "A's step by then");
        within(sharp_at(a, 80), 4.55, 4.6, "A's sharpness at step 80");
        assert!(a.diverged.is_none() && b.diverged.is_none());
        // at a peak of 0.8
        let both = |lab: &Lab| (lab.runs[0].diverged, lab.runs[1].diverged);
        let lab = changed(3, 0, |d| d.hyper.lr = 0.8);
        assert_eq!(lab.runs[0].diverged, Some(53));
        let lab = changed(3, 1, |d| d.hyper.lr = 0.8);
        assert_eq!(lab.runs[1].diverged, Some(12));
        let lab = changed(3, 0, |d| {
            d.hyper.lr = 0.8;
            d.warmup = 200;
        });
        assert_eq!(both(&lab).0, Some(138));
    }

    /// (Slow: run with `--ignored`; the start of it is held by
    /// `the_experiments_begin_as_their_captions_say`.)
    #[test]
    #[ignore]
    fn adamw_against_adam_with_decay() {
        let lab = run(4);
        let (a, b) = (&lab.runs[0], &lab.runs[1]);
        let hidden = |r: &Run| -> (f64, f64) { (r.seen.frob[..3].iter().copied().fold(f64::INFINITY, f64::min), r.seen.frob[..3].iter().copied().fold(0.0, f64::max)) };
        let (lo, hi) = hidden(a);
        within(lo, 2.45, 2.5, "A's smallest hidden matrix");
        within(hi, 4.8, 4.85, "A's largest");
        let (lo, hi) = hidden(b);
        within(lo, 1.88, 1.92, "B's smallest");
        within(hi, 2.4, 2.45, "B's largest");
        within(a.losses[1000], 0.0016, 0.0018, "A after 1000 steps");
        within(b.losses[1000], 0.074, 0.077, "B after 1000 steps");
        assert_eq!(a.design.name(), "AdamW");
    }

    /// With layer normalisation after a hidden layer, scaling its weights and bias
    /// together leaves the network's output as it was (to the normalisation's guard ε).
    #[test]
    fn a_normalised_layer_does_not_see_its_size() {
        let lab = run_to(4, 200, false);
        let problem = &lab.problem;
        let mut model = lab.runs[0].model.clone();
        let mut tape = Tape::new();
        let before = problem.loss(&model, &mut tape);
        for name in ["layer 2 weights", "layer 2 bias"] {
            if let Some(g) = model.group_mut(name) {
                g.value.iter_mut().for_each(|v| *v *= 3.0);
            }
        }
        let after = problem.loss(&model, &mut tape);
        // (the guard, 10⁻⁵ beside a variance near 1, moves it by about that share)
        assert!((after - before).abs() < 3e-5 * before, "{before} against {after}");
        // and the weights alone do change it
        let mut alone = lab.runs[0].model.clone();
        if let Some(g) = alone.group_mut("layer 2 weights") {
            g.value.iter_mut().for_each(|v| *v *= 3.0);
        }
        assert!((problem.loss(&alone, &mut tape) - before).abs() > 10.0 * (after - before).abs());
    }

    /// For plain descent, decay taken off the weights and decay added to the gradient
    /// are the same steps (the library's decoupled decay is multiplied by the step size).
    #[test]
    fn for_plain_descent_the_two_decays_are_the_same() {
        let mut lab = Lab::create();
        lab.set_up(4);
        for k in 0..2 {
            lab.designs[k].rule = Rule::Sgd;
            lab.designs[k].hyper = Hyper { weight_decay: if k == 0 { WeightDecay::Decoupled(0.01) } else { WeightDecay::Coupled(0.01) }, ..Hyper::default_for(Rule::Sgd) };
            lab.designs[k].hyper.lr = 0.1;
        }
        lab.apply();
        for _ in 0..50 {
            lab.step_both(false);
        }
        let (mut wa, mut wb) = (Vec::new(), Vec::new());
        crate::math::terrain::get(&lab.runs[0].model, &mut wa);
        crate::math::terrain::get(&lab.runs[1].model, &mut wb);
        let most = wa.iter().zip(&wb).map(|(x, y)| (x - y).abs()).fold(0.0, f64::max);
        assert!(most < 1e-12, "{most}");
    }

    /// The level of each rule is where its step on a quadratic stops shrinking and starts
    /// to grow, held against the library's own steps: every parameter in a bowl of
    /// curvature h (its gradient h θ), a little under and a little over c / η.
    #[test]
    fn the_levels_are_where_the_steps_become_unstable() {
        let lr = 0.1;
        let base = |rule: Rule, mu: f64, conv: Convention| Design { hyper: Hyper { momentum: mu, convention: conv, ..Design::of(rule, lr, 1).hyper }, ..Design::of(rule, lr, 1) };
        let cases = [
            base(Rule::Sgd, 0.0, Convention::Sum),
            base(Rule::Momentum, 0.9, Convention::Sum),
            base(Rule::Momentum, 0.5, Convention::Ema),
            base(Rule::Nesterov, 0.9, Convention::Sum),
            base(Rule::Nesterov, 0.7, Convention::Ema),
        ];
        for d in cases {
            let c = d.level().unwrap();
            for (factor, grows) in [(0.97, false), (1.03, true)] {
                let h = factor * c / lr;
                let mut model = TEACHER.build(1);
                let mut opt = Optimizer::new(&model, d.rule, d.hyper);
                let size = |m: &Model| m.groups().iter().map(|g| optim::frobenius(&g.value).powi(2)).sum::<f64>().sqrt();
                let start = size(&model);
                for _ in 0..3000 {
                    for g in model.groups_mut() {
                        for (gr, v) in g.grad.iter_mut().zip(&g.value) {
                            *gr = h * v;
                        }
                    }
                    opt.step(&mut model, lr);
                }
                let end = size(&model);
                // (one that has grown past the largest double has grown)
                assert_eq!(end > start || end.is_nan(), grows, "{} ({:?}, μ {}): {start} to {end} at {factor} of the level", d.rule.name(), d.hyper.convention, d.hyper.momentum);
            }
        }
        // Adam with its divisor held fixed is heavy ball written as an average, on P⁻¹H;
        // RMSProp with momentum is heavy ball as a sum on it; AdaGrad is plain descent
        let adam = Design::of(Rule::Adam, lr, 1);
        assert_eq!(adam.level(), base(Rule::Momentum, adam.hyper.beta1, Convention::Ema).level());
        let rms = Design { hyper: Hyper { momentum: 0.5, ..Design::of(Rule::RmsProp, lr, 1).hyper }, ..Design::of(Rule::RmsProp, lr, 1) };
        assert_eq!(rms.level(), base(Rule::Momentum, 0.5, Convention::Sum).level());
        assert_eq!(Design::of(Rule::AdaGrad, lr, 1).level(), Some(2.0));
        within(adam.level().unwrap(), 37.99, 38.01, "Adam's c at β₁ = 0.9");
        for rule in [Rule::SignSgd, Rule::Signum, Rule::Lion, Rule::Muon, Rule::NorMuon, Rule::AdaMuon, Rule::SpectralDescent, Rule::Scion, Rule::Shampoo, Rule::Soap] {
            assert_eq!(Design::of(rule, lr, 1).level(), None);
        }
    }

    /// Two designs with the same batch size deal the same batches.
    #[test]
    fn equal_batch_sizes_see_the_same_batches() {
        let mut lab = Lab::create();
        lab.set_up(4);
        lab.designs[1] = Design::of(Rule::Sgd, 0.1, lab.designs[0].batch);
        lab.apply();
        for _ in 0..30 {
            lab.step_both(false);
            assert_eq!(lab.runs[0].batch, lab.runs[1].batch);
        }
        lab.designs[1].batch = 31;
        lab.apply();
        lab.step_both(false);
        assert_ne!(lab.runs[0].batch, lab.runs[1].batch);
    }

    #[test]
    fn the_scene_opens_on_the_first_experiment() {
        let lab = Lab::create();
        let t = trial(OPENING);
        assert_eq!(lab.designs, [t.a.kept(POINTS), t.b.kept(POINTS)]);
        assert_eq!((lab.net, lab.total, lab.seed, lab.fore), (t.net.kept(), t.total, SEED, t.fore));
        assert!(lab.running);
        for k in 0..TRIALS {
            let t = trial(k);
            assert!(!t.caption.is_empty() && t.caption == CAPTIONS[k]);
        }
    }

    /// What is drawn is finite, whatever was measured, and a glide that has arrived rests.
    fn drawn_finite(lab: &mut Lab) {
        lab.look_again();
        lab.follow(1e6, true);
        for g in lab.look.sv.iter().chain(&lab.look.sizes).chain(&lab.look.ranges) {
            assert!(g.shown.iter().all(|v| v.is_finite()), "a glide holds {:?}", g.shown);
        }
        lab.follow(1.0 / 60.0, false);
        assert!(!lab.look.moving);
    }

    /// Every rule on every task, with its settings at the ends of their controls: nothing
    /// panics, and every number drawn is finite.
    #[test]
    fn the_ends_of_the_controls_are_fine() {
        let mut lab = Lab::create();
        let ends = |rule: Rule, e: usize| -> Design {
            let d = Design::of(rule, 0.01, 4);
            let h = d.hyper;
            match e {
                0 => Design { hyper: Hyper { lr: 0.0, momentum: 0.0, beta1: 0.0, beta2: 0.0, eps: 0.0, alpha: 0.0, clip: 1e-9, ..h }, batch: 1, ..d },
                1 => Design {
                    hyper: Hyper { lr: 10.0, momentum: 0.9999, beta1: 0.9999, beta2: 0.99999, weight_decay: WeightDecay::Coupled(1.0), ortho: Ortho::Exact, scale: Scale::Nuclear, groups: Groups::Matrices, ..h },
                    batch: 9999,
                    warmup: 9999,
                    shape: Shape::Cosine,
                    ..d
                },
                _ => Design {
                    hyper: Hyper { lr: f64::NAN, momentum: f64::INFINITY, eps: -1.0, weight_decay: WeightDecay::Decoupled(f64::NAN), clip: f64::NAN, aux: optim::Aux { lr: f64::INFINITY, ..h.aux }, ..h },
                    batch: 0,
                    shape: Shape::Linear,
                    ..d
                },
            }
        };
        // (every rule at every end, each time on another task and normalisation, so
        // that every task and every end meets many rules)
        let norms = [Norm::None, Norm::Batch, Norm::Rms];
        for (i, rule) in Rule::ALL.into_iter().enumerate() {
            for e in 0..3 {
                let task = Task::ALL[(i + e) % 3];
                let norm = norms[(i + 2 * e) % 3];
                lab.net = Net { task, depth: if e == 1 { 9 } else { 1 }, width: if e == 1 { 99 } else { 1 }, norm };
                lab.total = if e == 2 { 0 } else { 20 };
                lab.designs = [ends(rule, e), ends(Rule::ALL[(i + 5 * e + 3) % 16], e)];
                lab.layer = 99;
                lab.apply();
                for _ in 0..3 {
                    lab.step_both(true);
                }
                drawn_finite(&mut lab);
            }
        }
    }

    /// A step that blows a run up stops it and says when; the other goes on.
    #[test]
    fn a_run_that_blows_up_stops_and_says_so() {
        let mut lab = changed(0, 0, |d| d.hyper.lr = 50.0);
        let s = lab.runs[0].diverged.expect("a step of 50 blows up");
        assert!(s <= 5, "{s}");
        assert_eq!(lab.runs[0].steps, s);
        assert_eq!(lab.runs[1].steps, 500);
        lab.finish_probes();
        drawn_finite(&mut lab);
        assert!(lab.finished());
    }

    /// In every window, from a phone's to one far too small, the cells lie in the view,
    /// do not overlap, and every chart has a place.
    #[test]
    fn the_layout_fits_every_window() {
        for (w, h) in [(1600.0, 1000.0), (1270.0, 940.0), (1100.0, 700.0), (770.0, 650.0), (570.0, 650.0), (390.0, 800.0), (390.0, 600.0), (240.0, 180.0), (1.0, 1.0), (3000.0, 300.0)] {
            let lay = Layout::of(Vec2::new(w, h), 40.0);
            let shown: Vec<Chart> = lay.cells.iter().filter_map(|c| c.1).chain(lay.tabbed.iter().copied()).collect();
            assert_eq!(shown.len(), Chart::ALL.len(), "{w} x {h}: every chart has a place");
            if lay.tiny() {
                continue;
            }
            for (i, (a, _)) in lay.cells.iter().enumerate() {
                assert!(a.0.x >= 0.0 && a.1.x <= w + 0.5 && a.1.y <= h + 0.5, "{w} x {h}: cell {i} {a:?}");
                for (b, _) in &lay.cells[i + 1..] {
                    let apart = a.1.x <= b.0.x + 0.5 || b.1.x <= a.0.x + 0.5 || a.1.y <= b.0.y + 0.5 || b.1.y <= a.0.y + 0.5;
                    assert!(apart, "{w} x {h}: cells overlap");
                }
            }
        }
    }


}


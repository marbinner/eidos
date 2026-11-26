//! The architecture playground: build a network from a specification and watch its
//! signal go forward, its gradient come back and its training, beside a second design
//! kept for comparison.
//!
//! Everything is the library's (`math::arch`, `math::autograd`): a design is an
//! `MlpSpec` or a `TransformerSpec` and a way of training it; its network is built
//! from a fixed seed, trained step by step on the same examples as the design kept
//! beside it, and measured by the library's `Probe` on a fixed batch, so that the two
//! are measured alike. What the charts show is worked out at the present weights, a few
//! times a second while training runs (each measurement rests twice as long as it took,
//! so it never takes more than a third of the time), and drawn through glides, so that
//! numbers that arrive in steps move evenly.

use super::*;
use crate::clock::{Pace, Rest};
use crate::math::arch::{self, Init, Kind, Loss, Model, MlpSpec, Norm, Place, Probe, Span, Stage, TransformerSpec};
use crate::math::autograd::{Act, Tape, NO_TARGET};
use crate::scenes::stats::plot::{self, Glide, Plot, Rect, AXES, CURVES, MARKS, OWN};

pub const DEF: SceneDef = SceneDef {
    id: "archlab",
    title: "Architecture playground",
    group: GROUP,
    blurb: "Build a network, choose its depth, width, activation, normalisation, skip connections and starting weights, and watch signals go forward, gradients come back, and training itself.",
    make: |_| Box::new(Lab::create()),
};

/// The design kept for comparison, wherever it is drawn: grey, and dashed.
const KEPT: Color = Color::hex(0xB4BCCB);

// ----- what can be built ---------------------------------------------------------------

/// What the network is asked to learn.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Task {
    /// An MLP sorting points of the plane into three interleaved spiral arms.
    Spirals,
    /// An MLP imitating a fixed small network on four numbers (a regression).
    Teacher,
    /// A transformer predicting the second copy of a repeated run of tokens.
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

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Optim {
    Sgd,
    Adam,
}

/// A design: the network and how it is trained. Every number here is one a control
/// sets; `kept` brings each into the range the controls offer.
#[derive(Clone, Copy, Debug, PartialEq)]
struct Design {
    task: Task,
    /// Hidden layers (an MLP) or blocks (a transformer).
    depth: usize,
    /// Units in every hidden layer, or the width of the transformer's stream.
    width: usize,
    /// The transformer's heads, and how many tokens it reads.
    heads: usize,
    context: usize,
    act: Act,
    norm: Norm,
    place: Place,
    residual: bool,
    /// The factor on every residual branch (MLP).
    scale: f64,
    /// How the MLP's weights are drawn at the start, and the factor on them all.
    init: Init,
    gain: f64,
    optim: Optim,
    lr: f64,
    /// Examples (MLP) or sequences (transformer) in each training step.
    batch: usize,
    /// The longest the whole gradient may be (0: no clipping).
    clip: f64,
}

/// The sizes offered: what keeps one training step of one design (a forward and a
/// backward pass) to a few milliseconds in a browser. (Natively, a step of the largest
/// MLP takes about 3 ms; a transformer's step is split into one piece per sequence,
/// each under about 1.5 ms.)
const MLP_DEPTH: usize = 32;
const MLP_WIDTH: usize = 32;
const MLP_BATCH: usize = 32;
const TF_DEPTH: usize = 8;
const TF_WIDTH: usize = 32;
const TF_HEADS: usize = 4;
const TF_CONTEXT: usize = 12;
const TF_BATCH: usize = 8;
/// Different tokens in the repeated runs (a run is at most 6 long, so its tokens differ).
const VOCAB: usize = 10;
/// Inputs of the teacher's regression.
const TEACHER_INPUTS: usize = 4;

/// The design the controls start from, and that the experiments vary.
const BASE: Design = Design {
    task: Task::Spirals,
    depth: 20,
    width: 32,
    heads: 2,
    context: 12,
    act: Act::Tanh,
    norm: Norm::None,
    place: Place::Pre,
    residual: false,
    scale: 1.0,
    init: Init::Lecun,
    gain: 1.0,
    optim: Optim::Adam,
    lr: 0.001,
    batch: 32,
    clip: 0.0,
};

impl Design {
    /// The same with every number in the range its control offers.
    fn kept(self) -> Design {
        let mlp = self.task.mlp();
        let (depth, width, batch) = if mlp { (MLP_DEPTH, MLP_WIDTH, MLP_BATCH) } else { (TF_DEPTH, TF_WIDTH, TF_BATCH) };
        let finite = |v: f64, lo: f64, hi: f64, or: f64| if v.is_finite() { v.max(lo).min(hi) } else { or };
        Design {
            depth: self.depth.clamp(1, depth),
            width: self.width.clamp(1, width),
            heads: self.heads.clamp(1, TF_HEADS),
            context: self.context.clamp(2, TF_CONTEXT),
            batch: self.batch.clamp(1, batch),
            scale: finite(self.scale, 0.0, 1.0, 1.0),
            gain: finite(self.gain, 0.0, 3.0, 1.0),
            lr: finite(self.lr, 0.0, 1.0, 0.001),
            clip: finite(self.clip, 0.0, 100.0, 0.0),
            ..self
        }
    }

    fn mlp_spec(&self) -> MlpSpec {
        let (inputs, outputs) = match self.task {
            Task::Teacher => (TEACHER_INPUTS, 1),
            _ => (2, 3),
        };
        MlpSpec {
            inputs,
            width: self.width,
            depth: self.depth,
            outputs,
            act: self.act,
            norm: self.norm,
            place: self.place,
            residual: self.residual,
            residual_scale: self.scale,
            init: self.init,
            gain: self.gain,
            bias: true,
        }
    }

    fn transformer_spec(&self) -> TransformerSpec {
        TransformerSpec {
            vocab: VOCAB,
            context: self.context,
            d_model: self.width,
            heads: self.heads,
            layers: self.depth,
            d_mlp: 2 * self.width,
            act: self.act,
            norm: self.norm,
            place: self.place,
            residual: self.residual,
            init_gain: self.gain,
            tied_unembed: false,
        }
    }

    /// The network at its start. Every design draws its weights from the same seed, so
    /// that two designs that differ in one setting differ in as little else as can be.
    fn build(&self) -> Model {
        let mut rng = Rng::new(WEIGHT_SEED);
        if self.task.mlp() {
            Model::mlp(&self.mlp_spec(), &mut rng)
        } else {
            Model::transformer(&self.transformer_spec(), &mut rng)
        }
    }

    /// The number of points along the charts' layer axis: from what enters the first
    /// layer (0) to the read-out (depth + 1).
    fn stops(&self) -> usize {
        self.depth + 2
    }

    /// The design in a line: "20 tanh layers of 32 · LeCun ×0.6 · Adam 0.001".
    fn summary(&self, short: bool) -> String {
        let norm = match (self.norm, self.place) {
            (Norm::None, _) => String::new(),
            (n, p) => format!(", {} norm ({})", n.name(), p.name()),
        };
        let skips = if self.residual {
            if self.task.mlp() && self.scale != 1.0 {
                format!(", skips ×{}", trim(self.scale, 2))
            } else {
                ", skips".to_string()
            }
        } else {
            String::new()
        };
        let body = if self.task.mlp() {
            format!("{} {} layers of {}{}{}", self.depth, self.act.name(), self.width, norm, skips)
        } else {
            format!("{} blocks of {}, {} heads, {}{}{}", self.depth, self.width, self.heads, self.act.name(), norm, skips)
        };
        let optim = match self.optim {
            Optim::Sgd => "descent",
            Optim::Adam => "Adam",
        };
        if short {
            return format!("{body} · {optim} {}", trim(self.lr, 4));
        }
        let start = if self.task.mlp() { format!("{} ×{}", self.init.name(), trim(self.gain, 2)) } else { format!("gain {}", trim(self.gain, 2)) };
        let clip = if self.clip > 0.0 { format!(", clip {}", trim(self.clip, 2)) } else { String::new() };
        format!("{body} · {start} · {optim} {}, batch {}{clip}", trim(self.lr, 4), self.batch)
    }
}

/// A number with no more digits than it needs, up to `most` decimals.
fn trim(v: f64, most: usize) -> String {
    let mut t = format!("{:.*}", most, v);
    if t.contains('.') {
        t = t.trim_end_matches('0').trim_end_matches('.').to_string();
    }
    if t.starts_with('-') {
        t = format!("−{}", &t[1..]);
    }
    t
}

const WEIGHT_SEED: u64 = 7;
const DATA_SEED: u64 = 11;
const BATCH_SEED: u64 = 0x5EED_BA7C;
/// Training examples of the MLP tasks, and how many of them the probe looks at.
const TRAIN_N: usize = 300;
const PROBE_N: usize = 32;
/// Sequences the probe looks at.
const PROBE_SEQS: usize = 4;

/// The data of a task: for the MLP tasks a training set (inputs row-major, targets as
/// `Model::loss_mlp` takes them); the transformer's sequences are drawn afresh for each
/// step, and the probe's are fixed.
#[derive(Clone, Debug)]
struct Data {
    task: Task,
    x: Vec<f64>,
    y: Vec<f64>,
    inputs: usize,
    probe_seqs: Vec<(Vec<usize>, Vec<usize>)>,
    context: usize,
    /// The loss of a network that knows nothing: ln 3 for three classes, ln 10 for ten
    /// tokens, and for the regression the mean square of the targets about their mean.
    chance: f64,
}

impl Data {
    fn new(task: Task, context: usize) -> Data {
        let mut rng = Rng::new(DATA_SEED);
        let (x, y, inputs) = match task {
            Task::Spirals => {
                let (x, y) = arch::spirals(TRAIN_N, 3, &mut rng);
                (x, y, 2)
            }
            Task::Teacher => {
                let (x, y) = arch::teacher(TRAIN_N, TEACHER_INPUTS, &mut rng);
                (x, y, TEACHER_INPUTS)
            }
            Task::Repeats => (Vec::new(), Vec::new(), 0),
        };
        let probe_seqs = if task == Task::Repeats { (0..PROBE_SEQS).map(|_| sequence(context, &mut rng)).collect() } else { Vec::new() };
        let chance = match task {
            Task::Spirals => 3f64.ln(),
            Task::Repeats => (VOCAB as f64).ln(),
            Task::Teacher => {
                let mean = y.iter().sum::<f64>() / y.len().max(1) as f64;
                y.iter().map(|v| (v - mean) * (v - mean)).sum::<f64>() / y.len().max(1) as f64
            }
        };
        Data { task, x, y, inputs, probe_seqs, context, chance }
    }

    fn loss(&self) -> Loss {
        if self.task == Task::Teacher {
            Loss::Mse
        } else {
            Loss::CrossEntropy
        }
    }

    /// The examples of training step `step`: `batch` different ones of the training
    /// set, the same for every design, so that two designs see the same data.
    fn batch(&self, step: u64, batch: usize, order: &mut Vec<usize>, x: &mut Vec<f64>, y: &mut Vec<f64>) {
        let n = self.y.len();
        let mut rng = Rng::new(BATCH_SEED ^ step.wrapping_mul(0x9E37_79B9_7F4A_7C15));
        order.clear();
        order.extend(0..n);
        let take = batch.min(n);
        for i in 0..take {
            let j = i + rng.below(n - i);
            order.swap(i, j);
        }
        x.clear();
        y.clear();
        for &k in &order[..take] {
            x.extend_from_slice(&self.x[k * self.inputs..(k + 1) * self.inputs]);
            y.push(self.y[k]);
        }
    }

    /// Sequence `k` of training step `step`.
    fn sequence_of(&self, step: u64, k: usize) -> (Vec<usize>, Vec<usize>) {
        let mut rng = Rng::new(BATCH_SEED ^ step.wrapping_mul(0x9E37_79B9_7F4A_7C15) ^ (k as u64 + 1).wrapping_mul(0xD1B5_4A32_D192_ED03));
        sequence(self.context, &mut rng)
    }
}

/// A repeated run of tokens, scored on its second copy alone: the positions whose next
/// token the first copy makes predictable. (The first copy is random: nothing can
/// predict it, and a loss that counted it could not fall to 0.)
fn sequence(context: usize, rng: &mut Rng) -> (Vec<usize>, Vec<usize>) {
    let (tokens, mut targets) = arch::repeats(context, VOCAB, rng);
    let run = arch::run_length(context);
    for t in targets.iter_mut().take(run) {
        *t = NO_TARGET;
    }
    (tokens, targets)
}

/// How many scored positions have their target's logit the largest, of how many.
fn right_tokens(logits: &[f64], vocab: usize, targets: &[usize]) -> (usize, usize) {
    let (mut right, mut scored) = (0, 0);
    for (row, &t) in logits.chunks_exact(vocab.max(1)).zip(targets) {
        if t >= vocab {
            continue;
        }
        scored += 1;
        if (0..vocab).all(|k| k == t || row[k] < row[t]) {
            right += 1;
        }
    }
    (right, scored)
}

// ----- one design being trained ----------------------------------------------------------

/// A design and its network as training leaves it.
#[derive(Clone, Debug)]
struct Run {
    design: Design,
    model: Model,
    /// Training steps taken.
    steps: u64,
    /// The training loss and the share right of each step, in histories that keep a
    /// point for every so many steps.
    losses: History,
    rights: History,
    /// The last step's loss and share right (NaN for the regression).
    last: Option<(f64, f64)>,
    /// The step at which the loss or a weight stopped being a number, if one has.
    diverged: Option<u64>,
    /// For each parameter group, the square of the length of its last step and of its
    /// weights after it.
    moved: Vec<(f64, f64)>,
    /// Halfway through a transformer's step: the sequences done, their loss, and their
    /// right and scored positions.
    partial: (usize, f64, usize, usize),
    /// Room for a step's examples and for the weights before it.
    order: Vec<usize>,
    bx: Vec<f64>,
    by: Vec<f64>,
    before: Vec<f64>,
}

/// Points in the loss charts' histories.
const HISTORY: usize = 240;

impl Run {
    fn new(design: Design) -> Run {
        let design = design.kept();
        Run {
            design,
            model: design.build(),
            steps: 0,
            losses: History::new(HISTORY),
            rights: History::new(HISTORY),
            last: None,
            diverged: None,
            moved: Vec::new(),
            partial: (0, 0.0, 0, 0),
            order: Vec::new(),
            bx: Vec::new(),
            by: Vec::new(),
            before: Vec::new(),
        }
    }

    /// Pieces of work a training step is made of: the whole batch at once for an MLP
    /// (batch normalisation needs it whole), one sequence each for a transformer.
    fn units(&self) -> usize {
        if self.design.task.mlp() {
            1
        } else {
            self.design.batch
        }
    }

    /// One piece of the next training step; true when the step is complete. A network
    /// whose numbers have stopped being numbers is not trained further.
    fn unit(&mut self, data: &Data, tape: &mut Tape) -> bool {
        let step = self.steps;
        if self.diverged.is_some() {
            self.steps += 1;
            return true;
        }
        let d = self.design;
        if d.task.mlp() {
            self.model.zero_grads();
            data.batch(step, d.batch, &mut self.order, &mut self.bx, &mut self.by);
            let trace = self.model.loss_mlp(tape, &self.bx, &self.by, data.loss());
            tape.backward(trace.loss);
            self.model.accumulate(tape, &trace);
            let loss = tape.value(trace.loss).first().copied().unwrap_or(f64::NAN);
            let right = if data.loss() == Loss::CrossEntropy { arch::share_right(tape.value(trace.output), 3, &self.by) } else { f64::NAN };
            self.update(loss, right);
            return true;
        }
        if self.partial.0 == 0 {
            self.model.zero_grads();
        }
        let (tokens, targets) = data.sequence_of(step, self.partial.0);
        let trace = self.model.loss_transformer(tape, &tokens, &targets);
        tape.backward(trace.loss);
        self.model.accumulate(tape, &trace);
        let (right, scored) = right_tokens(tape.value(trace.output), VOCAB, &targets);
        let loss = tape.value(trace.loss).first().copied().unwrap_or(f64::NAN);
        self.partial = (self.partial.0 + 1, self.partial.1 + loss, self.partial.2 + right, self.partial.3 + scored);
        if self.partial.0 < d.batch {
            return false;
        }
        let (n, total, right, scored) = std::mem::replace(&mut self.partial, (0, 0.0, 0, 0));
        // (the gradient of the mean over the sequences)
        let k = 1.0 / n.max(1) as f64;
        for g in self.model.groups_mut() {
            g.grad.iter_mut().for_each(|v| *v *= k);
        }
        self.update(total * k, right as f64 / scored.max(1) as f64);
        true
    }

    /// Clip, step, and note what the step did.
    fn update(&mut self, loss: f64, right: f64) {
        let d = self.design;
        if d.clip > 0.0 {
            self.model.clip(d.clip);
        }
        self.before.clear();
        for g in self.model.groups() {
            self.before.extend_from_slice(&g.value);
        }
        match d.optim {
            Optim::Sgd => self.model.step_sgd(d.lr),
            Optim::Adam => self.model.step_adam(d.lr),
        }
        let mut at = 0;
        self.moved.clear();
        let mut finite = loss.is_finite();
        for g in self.model.groups() {
            let n = g.value.len();
            let old = &self.before[at..at + n];
            at += n;
            let moved: f64 = old.iter().zip(&g.value).map(|(a, b)| (a - b) * (a - b)).sum();
            let size: f64 = g.value.iter().map(|v| v * v).sum();
            finite &= size.is_finite();
            self.moved.push((moved, size));
        }
        self.steps += 1;
        self.last = Some((loss, right));
        self.losses.push(if loss.is_finite() { loss as f32 } else { f32::NAN });
        if right.is_finite() {
            self.rights.push(right as f32);
        }
        if !finite && self.diverged.is_none() {
            self.diverged = Some(self.steps);
        }
    }

    /// A whole training step, every piece of it.
    #[cfg(test)]
    fn step(&mut self, data: &Data, tape: &mut Tape) {
        while !self.unit(data, tape) {}
    }
}

// ----- what is measured ----------------------------------------------------------------

/// One layer's numbers, from the probe.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
struct LayerStat {
    /// The root mean square and the mean of what the layer hands on, and the root mean
    /// square of the gradient of the loss there.
    rms: f64,
    mean: f64,
    grad: f64,
    /// The share of the activation's entries whose slope is under 0.01 in size (shut
    /// or saturated), and of its units for which that is so for every example (dead);
    /// none for the data and the read-out.
    shut: Option<f64>,
    dead: Option<f64>,
}

/// One group of weights (a weight matrix; in a transformer, the embeddings, a block's
/// attention or its MLP taken together): where it is, the length of its gradient on the
/// probe's batch and of its weights.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
struct GroupStat {
    /// Its place along the layer axis, and which part of a block (0 attention or the
    /// only part, 1 a transformer's MLP).
    layer: usize,
    part: u8,
    grad: f64,
    weights: f64,
    /// The indices of the model's groups it is made of.
    first: usize,
    count: usize,
}

/// Bins of a histogram.
const BINS: usize = 32;

/// A histogram from −`half` to `half` in `BINS` bins, each the share of the entries in
/// it (those beyond the range counted in the end bins).
#[derive(Clone, Debug, Default, PartialEq)]
struct Hist {
    half: f64,
    shares: Vec<f64>,
}

impl Hist {
    fn of(values: &[f64], half: f64) -> Hist {
        let half = if half.is_finite() && half > 0.0 { half } else { 1.0 };
        let mut shares = vec![0.0; BINS];
        let mut n = 0usize;
        for &v in values.iter().filter(|v| v.is_finite()) {
            n += 1;
            let at = ((v + half) / (2.0 * half) * BINS as f64).floor();
            let k = if at < 0.0 { 0 } else { (at as usize).min(BINS - 1) };
            shares[k] += 1.0;
        }
        let total = n.max(1) as f64;
        shares.iter_mut().for_each(|s| *s /= total);
        Hist { half, shares }
    }

    /// The half-width for values of this root mean square and largest size: three
    /// times the root mean square, but no wider than the largest.
    fn half_for(values: &[f64]) -> f64 {
        let n = values.iter().filter(|v| v.is_finite()).count().max(1) as f64;
        let rms = (values.iter().filter(|v| v.is_finite()).map(|v| v * v).sum::<f64>() / n).sqrt();
        let most = values.iter().filter(|v| v.is_finite()).fold(0.0f64, |a, b| a.max(b.abs()));
        let h = (3.0 * rms).min(most * 1.02);
        if h > 1e-300 && h.is_finite() {
            h
        } else {
            1.0
        }
    }
}

/// A layer's histograms: the input of its activation (or, where it has none, what it
/// hands on) and the gradient at what it hands on.
#[derive(Clone, Debug, Default, PartialEq)]
struct LayerHists {
    values: Hist,
    /// The activation whose input `values` is, if it is one.
    gate: Option<Act>,
    grads: Hist,
}

/// Everything the charts show for one design at its present weights, on the probe's
/// fixed batch.
#[derive(Clone, Debug, Default)]
struct Measure {
    /// For an MLP: what enters the first hidden layer (0), the hidden layers 1 to L,
    /// the read-out (L + 1). For a transformer: the stream after the embeddings (0)
    /// and after each block, and the logits.
    layers: Vec<LayerStat>,
    groups: Vec<GroupStat>,
    /// The histograms of the layer asked for.
    hists: LayerHists,
    loss: f64,
}

fn sum_sq(v: &[f64]) -> f64 {
    v.iter().map(|x| x * x).sum()
}

impl Measure {
    /// The probe of an MLP on the first `PROBE_N` examples of the training set.
    fn of_mlp(model: &Model, data: &Data, tape: &mut Tape, want: usize) -> Measure {
        let Some(spec) = model.mlp_spec().copied() else { return Measure::default() };
        let n = PROBE_N.min(data.y.len());
        let trace = model.loss_mlp(tape, &data.x[..n * data.inputs], &data.y[..n], data.loss());
        tape.backward(trace.loss);
        let probe = Probe::new(tape, &trace);
        let top = spec.depth + 1;
        let mut layers = Vec::with_capacity(top + 1);
        let mut hists = LayerHists::default();
        for l in 0..=top {
            // what layer l hands on: the data or the input map's output, a hidden
            // layer's output, the read-out's
            let stage = if l == 0 && !spec.residual { Stage::Input } else { Stage::Output };
            let (Some(var), Some(p)) = (trace.get(stage, l), probe.of(&trace, stage, l)) else { continue };
            let gate = trace.entry(Stage::PreAct, l);
            let gp = probe.of(&trace, Stage::PreAct, l);
            layers.push(LayerStat { rms: p.rms, mean: p.mean, grad: p.grad_rms, shut: gp.and_then(|g| g.shut), dead: gp.and_then(|g| g.dead) });
            if l == want {
                let values = tape.value(gate.map_or(var, |e| e.var));
                let grads = tape.grad(var);
                hists = LayerHists {
                    values: Hist::of(values, Hist::half_for(values)),
                    gate: gate.and_then(|e| e.gate.map(|g| g.0)),
                    grads: Hist::of(grads, Hist::half_for(grads)),
                };
            }
        }
        let groups = model
            .groups()
            .iter()
            .zip(&probe.groups)
            .enumerate()
            .filter(|(_, (g, _))| g.kind == Kind::Weight)
            .map(|(i, (g, p))| GroupStat { layer: g.layer, part: 0, grad: p.grad_norm, weights: p.value_norm, first: i, count: 1 })
            .collect();
        Measure { layers, groups, hists, loss: tape.value(trace.loss).first().copied().unwrap_or(f64::NAN) }
    }

    /// The probe of a transformer on the probe's sequences: each layer's numbers over
    /// all of them (a unit is dead if it is shut at every position of every one), and
    /// the gradients those of their mean loss.
    fn of_transformer(model: &Model, data: &Data, tape: &mut Tape, want: usize) -> Measure {
        let Some(spec) = model.transformer_spec().copied() else { return Measure::default() };
        let top = spec.layers + 1;
        let k = data.probe_seqs.len().max(1) as f64;
        // per layer: sum of squares, sum, sum of squared gradients, entries, shut entries,
        // gated entries
        let mut sums = vec![[0.0f64; 6]; top + 1];
        let mut alive: Vec<Vec<bool>> = vec![Vec::new(); top + 1];
        let mut group_sq = vec![0.0; model.groups().len()];
        let (mut want_values, mut want_grads) = (Vec::new(), Vec::new());
        let mut gate = None;
        let mut loss = 0.0;
        let mut acc = model.clone();
        acc.zero_grads();
        for (tokens, targets) in &data.probe_seqs {
            let trace = model.loss_transformer(tape, tokens, targets);
            tape.backward(trace.loss);
            loss += tape.value(trace.loss).first().copied().unwrap_or(f64::NAN) / k;
            acc.accumulate(tape, &trace);
            for l in 0..=top {
                let stage = if l == top { Stage::Logits } else { Stage::Stream };
                let Some(var) = trace.get(stage, l) else { continue };
                let (v, g) = (tape.value(var), tape.grad(var));
                let s = &mut sums[l];
                s[0] += sum_sq(v);
                s[1] += v.iter().sum::<f64>();
                s[2] += sum_sq(g) / (k * k);
                s[3] += v.len() as f64;
                let pre = trace.entry(Stage::MlpPre, l);
                if let Some(e) = pre {
                    let z = tape.value(e.var);
                    let units = tape.cols(e.var).max(1);
                    let act = e.gate.map_or(spec.act, |g| g.0);
                    let closed = |x: f64| act.df(x).abs() < 0.01;
                    s[4] += z.iter().filter(|&&x| closed(x)).count() as f64;
                    s[5] += z.len() as f64;
                    if alive[l].is_empty() {
                        alive[l] = vec![false; units];
                    }
                    for (j, a) in alive[l].iter_mut().enumerate() {
                        *a |= z.iter().skip(j).step_by(units).any(|&x| !closed(x));
                    }
                }
                if l == want {
                    match pre {
                        Some(e) => {
                            want_values.extend_from_slice(tape.value(e.var));
                            gate = Some(e.gate.map_or(spec.act, |g| g.0));
                        }
                        None => want_values.extend_from_slice(v),
                    }
                    want_grads.extend(g.iter().map(|x| x / k));
                }
            }
        }
        // the gradient of the mean loss for each group: the sum over the sequences, over
        // their number
        for (sq, g) in group_sq.iter_mut().zip(acc.groups()) {
            *sq = sum_sq(&g.grad) / (k * k);
        }
        let layers = (0..=top)
            .map(|l| {
                let s = sums[l];
                let n = s[3].max(1.0);
                let gated = s[5] > 0.0;
                let units = alive[l].len().max(1) as f64;
                LayerStat {
                    rms: (s[0] / n).sqrt(),
                    mean: s[1] / n,
                    grad: (s[2] / n).sqrt(),
                    shut: gated.then(|| s[4] / s[5]),
                    dead: gated.then(|| alive[l].iter().filter(|a| !**a).count() as f64 / units),
                }
            })
            .collect();
        // the weights by place and part: the embeddings, each block's attention and its
        // MLP, the unembedding
        let mut groups: Vec<GroupStat> = Vec::new();
        for (i, (g, &gsq)) in model.groups().iter().zip(&group_sq).enumerate() {
            let part = match g.kind {
                Kind::Query | Kind::Key | Kind::Value | Kind::HeadOut | Kind::Embedding | Kind::Position | Kind::Unembedding => 0,
                Kind::Weight => 1,
                _ => continue,
            };
            let wsq = sum_sq(&g.value);
            match groups.iter_mut().find(|s| s.layer == g.layer && s.part == part) {
                Some(s) => {
                    s.grad += gsq;
                    s.weights += wsq;
                    s.count = i + 1 - s.first;
                }
                None => groups.push(GroupStat { layer: g.layer, part, grad: gsq, weights: wsq, first: i, count: 1 }),
            }
        }
        for s in &mut groups {
            s.grad = s.grad.sqrt();
            s.weights = s.weights.sqrt();
        }
        let hists = LayerHists {
            values: Hist::of(&want_values, Hist::half_for(&want_values)),
            gate,
            grads: Hist::of(&want_grads, Hist::half_for(&want_grads)),
        };
        Measure { layers, groups, hists, loss }
    }

    fn of(model: &Model, data: &Data, tape: &mut Tape, want: usize) -> Measure {
        if model.mlp_spec().is_some() {
            Measure::of_mlp(model, data, tape, want)
        } else {
            Measure::of_transformer(model, data, tape, want)
        }
    }
}

/// With batch normalisation, the batch the Jacobian is worked out in (each example's
/// map depends on the others of its batch).
const JACOBIAN_BATCH: usize = 8;

/// The singular values of the Jacobian of an MLP's hidden layers 2 to L (how a small
/// change in what layer 1 hands on moves what layer L hands on), at the training set's
/// first example, largest first. Empty for a transformer, for a single hidden layer,
/// and where a number is not finite.
fn singular_values(model: &Model, data: &Data) -> Vec<f64> {
    let Some(spec) = model.mlp_spec() else { return Vec::new() };
    if spec.depth < 2 || data.y.is_empty() {
        return Vec::new();
    }
    let n = if spec.norm == Norm::Batch { JACOBIAN_BATCH.min(data.y.len()) } else { 1 };
    let (jac, rows, cols) = arch::input_output_jacobian(model, &data.x[..n * data.inputs], Span::Layers(2, spec.depth));
    if rows == 0 || cols == 0 || jac.iter().any(|v| !v.is_finite()) {
        return Vec::new();
    }
    crate::math::linalg::svd(&jac, rows, cols).1
}

/// The size of each group's step relative to its weights: the last step taken, or
/// before the first, the one the first would take on the probe's batch (plain descent:
/// the rate times the gradient; Adam's first step: the rate times g / (|g| + 1e-8),
/// entry by entry).
fn relative_steps(run: &Run, m: &Measure) -> Vec<f64> {
    if !run.moved.is_empty() {
        return m
            .groups
            .iter()
            .map(|g| {
                let (moved, size) = run.moved[g.first.min(run.moved.len())..(g.first + g.count).min(run.moved.len())]
                    .iter()
                    .fold((0.0, 0.0), |(a, b), (m, s)| (a + m, b + s));
                moved.sqrt() / size.sqrt()
            })
            .collect();
    }
    let d = run.design;
    m.groups
        .iter()
        .map(|g| match d.optim {
            Optim::Sgd => {
                let total = m.groups.iter().map(|g| g.grad * g.grad).sum::<f64>().sqrt();
                let k = if d.clip > 0.0 && total > d.clip { d.clip / total } else { 1.0 };
                d.lr * k * g.grad / g.weights
            }
            // (computed from the group's entries at the probe below; here, its share)
            Optim::Adam => f64::NAN,
        })
        .collect()
}

// ----- the experiments -------------------------------------------------------------------

struct Trial {
    button: &'static str,
    caption: &'static str,
    a: Design,
    b: Design,
    /// The steps it trains for, the layer it opens, and the chart in the slot that
    /// changes (where the window has room for only some of them).
    steps: u64,
    layer: usize,
    tab: Chart,
}

const DEEP_TANH: Design = Design { depth: 20, act: Act::Tanh, init: Init::Lecun, gain: 0.6, ..BASE };
const DEEP_RELU: Design = Design { depth: 30, act: Act::Relu, init: Init::He, gain: 1.0, ..BASE };
const POST_NORM: Design = Design {
    task: Task::Repeats,
    depth: 8,
    width: 16,
    heads: 2,
    context: 12,
    act: Act::Gelu,
    norm: Norm::Layer,
    place: Place::Post,
    residual: true,
    gain: 1.0,
    lr: 0.01,
    batch: 4,
    ..BASE
};
const TANH_8: Design = Design { depth: 8, act: Act::Tanh, init: Init::Xavier, gain: 1.0, lr: 0.003, ..BASE };
const RELU_6: Design = Design { depth: 6, act: Act::Relu, init: Init::He, gain: 1.0, lr: 0.003, ..BASE };

const TRIALS: [Trial; 5] = [
    Trial {
        button: "a deep network that will not train",
        caption: r"$B$ is twenty tanh layers that start too small (LeCun's start times 0.6); $A$ is the same with skip connections. Before training, look at the gradient: in $B$ it shrinks by about 0.6 at each layer on the way back, from $2.4 \times 10^{-3}$ at layer 20 to $1.7 \times 10^{-7}$ at layer 1, and the signal going forward dies away the same way. In $A$ the skips hand it back undiminished. Press “train”: after 300 steps $A$ gets every example of its batch right (loss 0.005), while $B$ is at 1.05, hardly below the 1.10 of a network that knows nothing. Without skips, a start at gain 1 mends it too, more slowly (0.003 after 1000 steps); layer normalisation alone does not (still above 1 after 1000). Explained in “Gradients through depth” and “Skip connections”.",
        a: Design { residual: true, ..DEEP_TANH },
        b: DEEP_TANH,
        steps: 300,
        layer: 1,
        tab: Chart::Gradient,
    },
    Trial {
        button: "thirty ReLU layers, normalised",
        caption: r"$B$ is thirty ReLU layers with He's start; $A$ adds batch normalisation after each linear layer. Before training, $A$'s signal leaves every layer at about the same size (root mean square 0.71), where $B$'s wanders between 0.26 and 0.65: going forward, normalisation does what it promises. Going back it does not: $A$'s gradient grows from $5.7 \times 10^{-3}$ at layer 30 to 4.1 at layer 1, about 700 times, while $B$'s stays within a factor of 6. Train: after 300 steps $B$ is at 0.09, $A$ still at 1.12. With skip connections normalisation does help: give $A$ layer normalisation, pre, and skips, and it reaches 0.004. Explained in “Normalisation and the gradient”.",
        a: Design { norm: Norm::Batch, place: Place::Post, ..DEEP_RELU },
        b: DEEP_RELU,
        steps: 300,
        layer: 1,
        tab: Chart::Gradient,
    },
    Trial {
        button: "post-norm against pre-norm",
        caption: r"Eight transformer blocks with layer normalisation, trained by Adam at rate 0.01. $B$ normalises after each sum (post-norm, as the first transformers did), $A$ before each attention and MLP, inside the branch (pre-norm, as most do now): in $A$'s diagram the skip road has nothing on it. Train, and watch the gradient: within 20 steps $B$'s falls at block 1 from $1.4 \times 10^{-2}$ to about $6 \times 10^{-6}$ while block 8 keeps $7 \times 10^{-3}$, so $B$'s lower blocks all but stop learning (by step 300 their steps are more than a hundred times smaller than block 8's), and after 300 steps $B$ is at 2.29, no better than a network that knows nothing (2.30). $A$ predicts every scored token of its last batch (loss 0.003). Explained in “Gradients in a transformer”.",
        a: Design { place: Place::Pre, ..POST_NORM },
        b: POST_NORM,
        steps: 300,
        layer: 1,
        tab: Chart::Weights,
    },
    Trial {
        button: "sigmoids everywhere",
        caption: r"Eight layers with Xavier's start: $B$ uses tanh, $A$ the sigmoid. A sigmoid's slope is at most $1/4$ (tanh's is 1), so on the way back each of $A$'s layers cuts the gradient by about 4: from $6.8 \times 10^{-3}$ at layer 8 to $2.5 \times 10^{-7}$ at layer 1, while $B$'s stays near $6 \times 10^{-3}$. And a sigmoid's outputs are not centred on 0: their mean is about 0.5 at every layer (the signal chart's “mean”). Train: after 1000 steps $B$ has learnt (loss 0.001) and $A$ has not (1.11). Explained in “Activation functions as gates”.",
        a: Design { act: Act::Sigmoid, ..TANH_8 },
        b: TANH_8,
        steps: 1000,
        layer: 1,
        tab: Chart::Gradient,
    },
    Trial {
        button: "a rate too large",
        caption: r"Six ReLU layers trained by Adam: $B$ at rate 0.003, $A$ at 0.1. Adam's first step moves every weight by the rate, whatever its gradient: the weights chart (“step”) shows that step against each matrix's size, 30 to 40 % for most of $A$'s, about 1 % for $B$'s. Train: $A$'s loss leaps (to 16 at step 2) and its ReLUs shut: after 10 steps three units in four are dead, shut for every example, and after 300 no more than one of the 32 units of each of layers 2 to 6 is alive. A dead unit hands no gradient back: at the outputs of $A$'s layers 1 to 5 it is exactly 0. $A$ stays at 1.10; $B$ reaches 0.003. Explained in “Optimisers on a loss surface”.",
        a: Design { lr: 0.1, ..RELU_6 },
        b: RELU_6,
        steps: 300,
        layer: 3,
        tab: Chart::Weights,
    },
];

/// The scene opens as the first experiment sets it up, before it trains.
const OPENING: usize = 0;

// ----- the scene -------------------------------------------------------------------------

/// The charts, in the order they are placed.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Chart {
    Signal,
    Gradient,
    Training,
    Weights,
    Jacobian,
    Layer,
}

impl Chart {
    fn tab(self) -> &'static str {
        match self {
            Chart::Signal => "signal",
            Chart::Gradient => "gradient",
            Chart::Training => "training",
            Chart::Weights => "weights",
            Chart::Jacobian => "Jacobian",
            Chart::Layer => "layer",
        }
    }
}

/// One design with what has been measured of it.
struct Side {
    run: Run,
    measure: Measure,
    /// The relative step of each of `measure.groups`.
    steps: Vec<f64>,
    sv: Vec<f64>,
    /// What the measurements are for: the run's step and the layer of the histograms
    /// (None: not measured since the design was built).
    measured: Option<(u64, usize)>,
    sv_at: Option<u64>,
}

impl Side {
    fn new(design: Design) -> Side {
        Side { run: Run::new(design), measure: Measure::default(), steps: Vec::new(), sv: Vec::new(), measured: None, sv_at: None }
    }

    fn restart(&mut self) {
        *self = Side::new(self.run.design);
    }

    fn stale(&self, layer: usize) -> bool {
        self.measured != Some((self.run.steps, layer))
    }

    fn measure(&mut self, data: &Data, tape: &mut Tape, layer: usize) {
        self.measure = Measure::of(&self.run.model, data, tape, layer);
        self.steps = relative_steps(&self.run, &self.measure);
        if self.run.moved.is_empty() && self.run.design.optim == Optim::Adam {
            // Adam's first step on the probe's gradient: every entry moves by the rate
            // times g / (|g| + 1e-8)
            self.steps = adam_first_steps(&self.run, data, tape, &self.measure);
        }
        self.measured = Some((self.run.steps, layer));
    }

    fn measure_sv(&mut self, data: &Data) {
        self.sv = singular_values(&self.run.model, data);
        self.sv_at = Some(self.run.steps);
    }
}

/// The relative size of Adam's first step for each group, on the probe's gradient.
fn adam_first_steps(run: &Run, data: &Data, tape: &mut Tape, m: &Measure) -> Vec<f64> {
    let model = &run.model;
    let lr = run.design.lr;
    // the probe's gradient for every group again (the measure keeps only its length)
    let mut acc = model.clone();
    acc.zero_grads();
    if model.mlp_spec().is_some() {
        let n = PROBE_N.min(data.y.len());
        let trace = model.loss_mlp(tape, &data.x[..n * data.inputs], &data.y[..n], data.loss());
        tape.backward(trace.loss);
        acc.accumulate(tape, &trace);
    } else {
        let k = data.probe_seqs.len().max(1) as f64;
        for (tokens, targets) in &data.probe_seqs {
            let trace = model.loss_transformer(tape, tokens, targets);
            tape.backward(trace.loss);
            acc.accumulate(tape, &trace);
        }
        acc.groups_mut().iter_mut().for_each(|g| g.grad.iter_mut().for_each(|v| *v /= k));
    }
    m.groups
        .iter()
        .map(|g| {
            let parts = &acc.groups()[g.first.min(acc.groups().len())..(g.first + g.count).min(acc.groups().len())];
            let moved: f64 = parts.iter().flat_map(|p| &p.grad).map(|d| (lr * d / (d.abs() + 1e-8)).powi(2)).sum();
            moved.sqrt() / g.weights
        })
        .collect()
}

/// While training: how often the charts are measured again, and the Jacobian (seconds).
const PROBE_EVERY: f32 = 0.2;
const SV_EVERY: f32 = 0.5;

/// The steps a second training goes at, unless the visitor asks for another speed.
const SPEED: f32 = 100.0;

pub struct Lab {
    /// A's design as the controls set it.
    design: Design,
    data: Data,
    a: Side,
    b: Option<Side>,
    /// The design kept as B for each task.
    kept: [Option<Design>; 3],
    running: bool,
    /// Where training stops by itself (an experiment's length), if anywhere.
    stop_at: Option<u64>,
    speed: f32,
    pace: Pace,
    /// Pieces of the present step done so far (A's first, then B's).
    unit: usize,
    tape: Tape,
    /// The layer whose histograms are shown, and the chart in the slot that changes.
    layer: usize,
    tab: Chart,
    /// The signal chart's size or mean; the weight chart's gradients or steps.
    show_mean: bool,
    show_steps: bool,
    /// Seconds since the last measurement and the last Jacobian, and how long each
    /// rests after the last one.
    since_probe: f32,
    since_sv: f32,
    rest: Rest,
    sv_rest: Rest,
    look: Look,
    canvas: Option<Vec2>,
}

impl Lab {
    fn create() -> Lab {
        let t = &TRIALS[OPENING];
        let mut lab = Lab {
            design: t.a,
            data: Data::new(t.a.task, t.a.context),
            a: Side::new(t.a),
            b: Some(Side::new(t.b)),
            kept: [None; 3],
            running: false,
            stop_at: Some(t.steps),
            speed: SPEED,
            pace: Pace::default(),
            unit: 0,
            tape: Tape::new(),
            layer: t.layer,
            tab: t.tab,
            show_mean: false,
            show_steps: false,
            since_probe: 0.0,
            since_sv: 0.0,
            rest: Rest::default(),
            sv_rest: Rest::default(),
            look: Look::default(),
            canvas: None,
        };
        lab.kept[t.a.task.index()] = Some(t.b);
        lab.measure_all();
        lab
    }

    /// Set up an experiment: both designs at their start, ready to train for as many
    /// steps as the experiment's caption speaks of.
    fn set_up(&mut self, k: usize) {
        let t = &TRIALS[k.min(TRIALS.len() - 1)];
        self.design = t.a.kept();
        self.kept[t.a.task.index()] = Some(t.b.kept());
        self.rebuild();
        self.layer = t.layer;
        self.tab = t.tab;
        self.show_mean = false;
        self.show_steps = matches!(t.tab, Chart::Weights);
        self.stop_at = Some(t.steps);
        self.speed = SPEED;
        // (it waits at its first step: the captions say what to see before training)
        self.running = false;
    }

    /// A's design has changed (or training starts again): both start again from their
    /// first step, so that the two are always compared at the same step.
    fn rebuild(&mut self) {
        self.design = self.design.kept();
        let task = self.design.task;
        if self.data.task != task || self.data.context != self.design.context {
            self.data = Data::new(task, self.design.context);
        }
        self.a = Side::new(self.design);
        self.b = self.kept[task.index()].map(Side::new);
        self.unit = 0;
        self.pace.reset();
        self.layer = self.layer.min(self.design.stops() - 1);
        if !task.mlp() && self.tab == Chart::Jacobian {
            self.tab = Chart::Signal;
        }
        if self.stop_at.is_some_and(|s| s == 0) {
            self.stop_at = None;
        }
    }

    /// Both designs back to their start.
    fn start_again(&mut self) {
        self.a.restart();
        if let Some(b) = &mut self.b {
            b.restart();
        }
        self.unit = 0;
        self.pace.reset();
    }

    fn step_count(&self) -> u64 {
        self.a.run.steps
    }

    /// Pieces of work in one step of both designs.
    fn units(&self) -> usize {
        self.a.run.units() + self.b.as_ref().map_or(0, |b| b.run.units())
    }

    /// The next piece of training; true when a whole step of both is complete.
    fn work_unit(&mut self) -> bool {
        let first = self.a.run.units();
        if self.unit < first {
            self.a.run.unit(&self.data, &mut self.tape);
        } else if let Some(b) = &mut self.b {
            b.run.unit(&self.data, &mut self.tape);
        }
        self.unit += 1;
        if self.unit >= self.units() {
            self.unit = 0;
            return true;
        }
        false
    }

    /// One whole step of both designs, at once.
    fn one_step(&mut self) {
        while !self.work_unit() {}
    }

    /// Training's share of this frame: so many steps a second of the scene's time,
    /// within the frame's allowance.
    fn train(&mut self, dt: f32, real_dt: f32) {
        if !self.running || dt <= 0.0 {
            return;
        }
        let units = self.units().max(1) as f32;
        let mut turn = self.pace.begin(self.speed * units, dt, real_dt, 0.004);
        while turn.next() {
            if self.work_unit() && self.stop_at.is_some_and(|s| self.step_count() >= s) {
                self.running = false;
                break;
            }
        }
        self.pace.end(turn);
        if !self.running {
            self.pace.reset();
        }
    }

    /// Is anything measured out of date?
    fn stale(&self) -> bool {
        self.a.stale(self.layer) || self.b.as_ref().is_some_and(|b| b.stale(self.layer))
    }

    fn sv_stale(&self) -> bool {
        self.design.task.mlp() && (self.a.sv_at != Some(self.a.run.steps) || self.b.as_ref().is_some_and(|b| b.sv_at != Some(b.run.steps)))
    }

    /// Measure both now (the tests, and the scene as it opens).
    fn measure_all(&mut self) {
        let layer = self.layer;
        self.a.measure(&self.data, &mut self.tape, layer);
        self.a.measure_sv(&self.data);
        if let Some(b) = &mut self.b {
            b.measure(&self.data, &mut self.tape, layer);
            b.measure_sv(&self.data);
        }
    }

    /// Measure what is out of date, if it is time to: between steps; while training,
    /// every so often (`PROBE_EVERY`, and the Jacobian `SV_EVERY`) and only after resting
    /// twice as long as the last measurement took, so that measuring never takes more
    /// than a third of the time; otherwise at once. A design at a time where time is
    /// short: the one further behind.
    fn measure_due(&mut self, real_dt: f32) {
        self.since_probe += real_dt.max(0.0);
        self.since_sv += real_dt.max(0.0);
        let free = crate::clock::reproducible();
        // (a design is measured between its own steps: A before its part of the
        // present step or after it, B before its part; where the pictures must be the
        // same every time, between whole steps)
        let first = self.a.run.units();
        if free && self.unit != 0 {
            return;
        }
        let (a_free, b_free) = (self.unit == 0 || self.unit >= first, self.unit <= first);
        let layer = self.layer;
        let a_due = a_free && self.a.stale(layer);
        let b_due = b_free && self.b.as_ref().is_some_and(|b| b.stale(layer));
        if (a_due || b_due) && (!self.running || self.since_probe >= PROBE_EVERY) && (free || self.rest.over()) {
            let start = crate::clock::now();
            let a_behind = a_due && (!b_due || self.a.measured.map(|m| m.0) <= self.b.as_ref().and_then(|b| b.measured.map(|m| m.0)));
            if a_due && (a_behind || free) {
                self.a.measure(&self.data, &mut self.tape, layer);
            }
            if b_due && (!a_behind || free) {
                if let Some(b) = self.b.as_mut() {
                    b.measure(&self.data, &mut self.tape, layer);
                }
            }
            self.rest = Rest::after(crate::clock::now() - start);
            if !self.stale() {
                self.since_probe = 0.0;
            }
        }
        let a_sv = a_free && self.design.task.mlp() && self.a.sv_at != Some(self.a.run.steps);
        let b_sv = b_free && self.design.task.mlp() && self.b.as_ref().is_some_and(|b| b.sv_at != Some(b.run.steps));
        if (a_sv || b_sv) && (!self.running || self.since_sv >= SV_EVERY) && (free || (self.rest.over() && self.sv_rest.over())) {
            let start = crate::clock::now();
            if a_sv {
                self.a.measure_sv(&self.data);
            }
            if let Some(b) = self.b.as_mut().filter(|_| b_sv) {
                b.measure_sv(&self.data);
            }
            // (the Jacobian rests longer: it is the costliest number shown)
            self.sv_rest = Rest::after(2.0 * (crate::clock::now() - start));
            if !self.sv_stale() {
                self.since_sv = 0.0;
            }
        }
    }

    /// Keep A's design as B (for this task), or forget B.
    fn keep(&mut self, on: bool) {
        let task = self.design.task.index();
        self.kept[task] = on.then_some(self.design.kept());
        self.rebuild();
    }

    /// The panel has changed A's design: rebuild if so.
    fn apply(&mut self) {
        if self.design.kept() != self.a.run.design {
            if self.design.task != self.a.run.design.task {
                self.layer = self.layer.min(self.design.kept().stops() - 1);
            }
            self.stop_at = None;
            self.rebuild();
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
    /// A's diagram, B's beneath it, and the button that keeps A as B.
    diagram: Rect,
    diagram_b: Rect,
    keep: Rect,
    /// The charts' cells; the one with tabs shows the chart chosen among the rest.
    cells: Vec<(Rect, Option<Chart>)>,
    tabbed: Vec<Chart>,
}

impl Layout {
    fn of(view: Vec2, inset: f32, charts: &[Chart]) -> Layout {
        let view = view.max(Vec2::splat(1.0));
        let narrow = view.x < 640.0;
        let (big, text, small) = if narrow { (12.5, 11.5, 10.5) } else { (13.5, 12.0, 11.0) };
        let side = if narrow { 8.0 } else { 16.0 };
        let top = inset.max(0.0) + if narrow { 6.0 } else { 12.0 };
        let r = |x0: f32, y0: f32, x1: f32, y1: f32| (Vec2::new(x0, y0), Vec2::new(x1.max(x0 + 1.0), y1.max(y0 + 1.0)));
        let (x0, x1) = (side, view.x - side);
        // the diagrams: A's tall enough for its boxes and its line of words, B's lower
        let keep_w = if narrow { 0.0 } else { 150.0 };
        let a_h = if narrow { 78.0 } else { 96.0 };
        let b_h = if narrow { 20.0 } else { 52.0 };
        let diagram = r(x0, top, x1 - keep_w, top + a_h);
        let diagram_b = r(x0, diagram.1.y + 4.0, x1 - keep_w, diagram.1.y + 4.0 + b_h);
        let keep = if narrow { r(x1 - 108.0, diagram_b.0.y - 2.0, x1, diagram_b.0.y + b_h + 2.0) } else { r(x1 - keep_w + 12.0, top + 8.0, x1, top + 40.0) };
        let grid_top = diagram_b.1.y + if narrow { 8.0 } else { 14.0 };
        let (gw, gh) = (x1 - x0, view.y - 8.0 - grid_top);
        let cols = ((gw / 330.0).floor() as usize).clamp(1, 3);
        let rows = ((gh / 215.0).floor() as usize).max(1);
        let n = charts.len();
        let slots = cols * rows;
        let shown = if slots >= n { n } else { slots };
        let rows_used = shown.div_ceil(cols).max(1);
        let gap = if narrow { 8.0 } else { 12.0 };
        let cw = (gw - gap * (cols - 1) as f32) / cols as f32;
        let ch = ((gh - gap * (rows_used - 1) as f32) / rows_used as f32).min(if narrow { 340.0 } else { 420.0 });
        // which charts get a cell of their own: the training chart always, then the rest
        // in order; where there is not room for all, the last cell has tabs
        let mut own: Vec<Chart> = Vec::new();
        let mut tabbed: Vec<Chart> = Vec::new();
        if shown >= n {
            own = charts.to_vec();
        } else {
            let fixed = shown.saturating_sub(1);
            let mut rest: Vec<Chart> = charts.iter().copied().filter(|c| *c != Chart::Training).collect();
            own.push(Chart::Training);
            while own.len() < fixed && !rest.is_empty() {
                own.push(rest.remove(0));
            }
            if fixed == 0 {
                own.clear();
                rest = charts.iter().copied().filter(|c| *c != Chart::Training).collect();
                rest.insert(0, Chart::Training);
            }
            tabbed = rest;
        }
        // in the reading order of the charts, the tabbed cell where its first chart is
        let mut order: Vec<Option<Chart>> = Vec::new();
        for c in charts {
            if own.contains(c) {
                order.push(Some(*c));
            } else if !order.contains(&None) && tabbed.contains(c) {
                order.push(None);
            }
        }
        // on a phone the training chart goes last, under the chart chosen
        if narrow {
            if let Some(i) = order.iter().position(|c| *c == Some(Chart::Training)) {
                let t = order.remove(i);
                order.push(t);
            }
        }
        let count = order.len();
        let cells = order
            .into_iter()
            .enumerate()
            .map(|(k, c)| {
                let (i, j) = (k / cols, k % cols);
                let (cx, cy) = (x0 + j as f32 * (cw + gap), grid_top + i as f32 * (ch + gap));
                // (the last cell of a row left short takes the rest of the row)
                let wide = if k + 1 == count && j + 1 < cols { (cols - j) as f32 * (cw + gap) - gap } else { cw };
                (r(cx, cy, cx + wide, cy + ch), c)
            })
            .collect();
        Layout { view, narrow, big, text, small, diagram, diagram_b, keep, cells, tabbed }
    }

    /// Too small a window to show more than words.
    fn tiny(&self) -> bool {
        self.view.x < 200.0 || self.view.y < 200.0
    }
}

/// The charts for a task.
fn charts_for(task: Task) -> Vec<Chart> {
    let mut c = vec![Chart::Signal, Chart::Gradient, Chart::Training, Chart::Weights, Chart::Jacobian, Chart::Layer];
    if !task.mlp() {
        c.retain(|c| *c != Chart::Jacobian);
    }
    c
}

// ----- what is drawn, gliding --------------------------------------------------------------

/// The lowest power of ten the logarithmic charts reach: anything smaller (a gradient of
/// exactly 0, a singular value lost in rounding) is drawn on the floor.
const FLOOR: f64 = -14.0;

fn log10_or_floor(v: f64) -> f32 {
    if v > 0.0 && v.is_finite() {
        v.log10().max(FLOOR) as f32
    } else if v.is_nan() || v.is_infinite() {
        NOT_A_NUMBER
    } else {
        ZERO
    }
}

/// What a logarithmic chart is given for a number that is exactly 0 (drawn on the
/// chart's bottom edge, which is then labelled 0) and for one that is not a number
/// (not drawn). Both far outside any range a chart shows, and finite, so that they
/// can glide.
const ZERO: f32 = -40.0;
const NOT_A_NUMBER: f32 = 40.0;

fn finite_or(v: f64, or: f32) -> f32 {
    if v.is_finite() {
        (v as f32).clamp(-1e30, 1e30)
    } else {
        or
    }
}

/// What the charts draw for one design, as it glides.
#[derive(Clone, Debug, Default)]
struct Shown {
    /// Per layer: log10 of the size, the mean, log10 of the gradient, the share shut.
    rms: Glide,
    mean: Glide,
    grad: Glide,
    shut: Glide,
    /// Per weight group: log10 of the gradient's length and of the relative step.
    w_grad: Glide,
    w_step: Glide,
    /// log10 of the singular values.
    sv: Glide,
    /// The histograms' halves and shares.
    halves: Glide,
    values: Glide,
    grads: Glide,
}

impl Shown {
    fn follow(&mut self, side: &Side, dt: f32) -> bool {
        let m = &side.measure;
        let mut moving = false;
        let mut go = |g: &mut Glide, to: Vec<f32>| moving |= g.follow(&to, dt, false);
        go(&mut self.rms, m.layers.iter().map(|l| log10_or_floor(l.rms)).collect());
        go(&mut self.mean, m.layers.iter().map(|l| finite_or(l.mean, 0.0)).collect());
        go(&mut self.grad, m.layers.iter().map(|l| log10_or_floor(l.grad)).collect());
        go(&mut self.shut, m.layers.iter().map(|l| finite_or(l.shut.unwrap_or(0.0), 0.0)).collect());
        go(&mut self.w_grad, m.groups.iter().map(|g| log10_or_floor(g.grad)).collect());
        go(&mut self.w_step, side.steps.iter().map(|s| log10_or_floor(*s)).collect());
        go(&mut self.sv, side.sv.iter().map(|s| log10_or_floor(*s)).collect());
        let h = &m.hists;
        go(&mut self.halves, vec![finite_or(h.values.half, 1.0), finite_or(h.grads.half, 1.0)]);
        go(&mut self.values, h.values.shares.iter().map(|v| finite_or(*v, 0.0)).collect());
        go(&mut self.grads, h.grads.shares.iter().map(|v| finite_or(*v, 0.0)).collect());
        moving
    }
}

#[derive(Clone, Debug, Default)]
struct Look {
    a: Shown,
    b: Shown,
    /// Each chart's ranges: [x lo, x hi, y lo, y hi], in the order of `Chart`, and where
    /// they are going.
    ranges: Vec<Glide>,
    goals: Vec<Vec<f32>>,
    moving: bool,
}

/// Lines whose number of points changes: the loss curves (A's, B's), the tab and toggle
/// outlines, the histograms' outlines.
const LOSS_A: u8 = OWN;
const LOSS_B: u8 = OWN + 1;
const DIAGRAM: u8 = OWN + 2;
const HIST: u8 = OWN + 3;
const BAND: u8 = OWN + 4;

/// Every kind of line this scene draws, asked for in the same order at the start of
/// every frame, each with one unseen point: a batch not drawn in a frame would drop
/// out and move every later one up a place (and `eidos motion` would take one for
/// another).
fn hold_batches(f: &mut Frame) {
    let corner = f.hud(Vec2::ZERO);
    for kind in [AXES, CURVES, MARKS, plot::BUTTONS, LOSS_A, LOSS_B, DIAGRAM, HIST, BAND] {
        plot::apart(f, kind, false).segment(corner, corner, 1.0, Color::CLEAR);
    }
    for kind in [CURVES, MARKS, LOSS_B, HIST] {
        plot::apart(f, kind, true).segment(corner, corner, 1.0, Color::CLEAR);
    }
    plot::disc(f, Vec2::ZERO, 1.0, Color::CLEAR);
}

impl Scene for Lab {
    fn view(&self) -> View {
        View::Flat { center: Vec2::ZERO, half_height: 1.0 }
    }

    fn settings(&self, s: &mut ViewSettings) {
        s.grid = false;
        s.bloom = s.bloom.min(0.2);
    }

    fn animated(&self) -> bool {
        self.running || self.look.moving || self.stale() || self.sv_stale()
    }

    fn pointer_hint(&self) -> Option<&'static str> {
        Some("buttons on the picture: click    a chart of layers: click a layer to open it")
    }

    fn ui(&mut self, ui: &mut egui::Ui, _: &Gfx) {
        self.panel(ui);
        self.apply();
    }

    fn frame(&mut self, f: &mut Frame) {
        hold_batches(f);
        let charts = charts_for(self.design.task);
        let lay = Layout::of(f.view_size(), f.top_inset, &charts);
        if !lay.tabbed.is_empty() && !lay.tabbed.contains(&self.tab) {
            self.tab = lay.tabbed[0];
        }
        let tiny = lay.tiny();
        if !tiny {
            self.buttons(f, &lay);
            self.clicks(f, &lay);
        }
        self.train(f.dt, f.real_dt);
        self.measure_due(f.real_dt);
        let snap = self.canvas != Some(lay.view);
        if snap && !tiny {
            self.canvas = Some(lay.view);
        }
        self.follow(f.real_dt, snap);
        if tiny {
            let at = Vec2::new(lay.view.x * 0.5, (lay.view.y + f.top_inset) * 0.5);
            f.text_px(at, "too small a window for the charts", 11.0, pal::DIM, Align2::CENTER_CENTER);
            return;
        }
        self.draw_diagrams(f, &lay);
        for (cell, chart) in lay.cells.clone() {
            let chart = chart.unwrap_or(self.tab);
            let tabs = if lay.cells.iter().any(|c| c.0 == cell && c.1.is_none()) { lay.tabbed.clone() } else { Vec::new() };
            self.draw_chart(f, &lay, cell, chart, &tabs);
        }
    }
}

// ----- the panel ---------------------------------------------------------------------------

impl Lab {
    fn panel(&mut self, ui: &mut egui::Ui) {
        ui::note(
            ui,
            "Build a network, and see what happens inside it: the signal going forward through its layers, the gradient of the loss coming back, and the network learning. A second design, $B$, can be kept beside it for comparison: both start from the same seed, see the same examples and are measured on the same fixed batch.",
        );
        ui::section(ui, "Things to try");
        for (k, trial) in TRIALS.iter().enumerate() {
            if ui::experiment(ui, trial.button, trial.caption) {
                self.set_up(k);
            }
        }

        ui::section(ui, "Design $A$: the network");
        let d = &mut self.design;
        let mut task = d.task.index();
        if ui::choice(ui, &["MLP: spirals", "MLP: a regression", "transformer"], &mut task) {
            let t = Task::ALL[task];
            *d = match t {
                Task::Repeats => Design { task: t, depth: 2, width: 16, heads: 2, context: 12, act: Act::Gelu, norm: Norm::Layer, place: Place::Pre, residual: true, gain: 1.0, lr: 0.003, batch: 4, ..*d },
                _ if !d.task.mlp() => Design { task: t, depth: 6, width: 32, act: Act::Tanh, norm: Norm::None, residual: false, init: Init::Xavier, gain: 1.0, lr: 0.003, batch: 32, ..*d },
                _ => Design { task: t, ..*d },
            };
        }
        ui::note(
            ui,
            match d.task {
                Task::Spirals => "An MLP sorts 300 points of the plane into three spiral arms (cross-entropy).",
                Task::Teacher => "An MLP learns to imitate a fixed small network on four numbers (squared error): a regression it can learn exactly.",
                Task::Repeats => "A transformer predicts each next token of a run of tokens said twice (cross-entropy).",
            },
        );
        let mlp = d.task.mlp();
        let (most_depth, most_width) = if mlp { (MLP_DEPTH, MLP_WIDTH) } else { (TF_DEPTH, TF_WIDTH) };
        ui::slider_int(ui, if mlp { "hidden layers" } else { "blocks" }, &mut d.depth, 1..=most_depth);
        ui::slider_int(ui, if mlp { "units per layer" } else { "width of the stream" }, &mut d.width, 1..=most_width);
        if !mlp {
            ui::slider_int(ui, "heads", &mut d.heads, 1..=TF_HEADS);
            ui::slider_int(ui, "tokens read", &mut d.context, 2..=TF_CONTEXT);
            let run = arch::run_length(d.context);
            ui::note(ui, &format!("It reads a run of {run} random tokens (of {VOCAB}) and the same run again, and is scored on the second copy, which the first makes predictable."));
            ui::note(ui, "Each block: attention, then an MLP twice as wide as the stream.");
        }
        ui::note(
            ui,
            if mlp {
                "Up to 32 layers of 32 units, and 32 examples a step: sizes at which one training step stays quick in a browser."
            } else {
                "Up to 8 blocks 32 wide, 12 tokens and 8 sequences a step: sizes at which training stays quick in a browser."
            },
        );
        let names: Vec<&str> = Act::ALL.iter().map(|a| a.name()).collect();
        let mut act = Act::ALL.iter().position(|a| *a == d.act).unwrap_or(0);
        if ui::choice(ui, &names, &mut act) {
            d.act = Act::ALL[act];
        }
        ui::note(ui, "The activation of every hidden unit. On the way back each unit multiplies the gradient by the activation's slope there.");
        let mut norm = Norm::ALL.iter().position(|n| *n == d.norm).unwrap_or(0);
        if ui::choice(ui, &["no normalisation", "batch", "layer", "RMS"], &mut norm) {
            d.norm = Norm::ALL[norm];
        }
        let mut place = (d.place == Place::Post) as usize;
        if ui::choice(ui, &["pre: before each layer", "post: after it"], &mut place) {
            d.place = if place == 1 { Place::Post } else { Place::Pre };
        }
        ui::note(ui, "Where the normalisation goes, as the diagram draws it. With skip connections, pre puts it inside the branch and leaves the straight road clear; post puts it on the road, after the sum.");
        ui::checkbox(ui, &mut d.residual, "skip connections: each layer adds to its input");
        ui::note(ui, "On the way back a skip hands the gradient down unchanged, beside what comes back through the layer.");
        if mlp && d.residual {
            let mut s = d.scale as f32;
            if ui::slider(ui, "the branch's factor", &mut s, 0.0..=1.0) {
                d.scale = s as f64;
            }
            if ui.button(ui::label_text(ui, r"factor $1/\sqrt{\text{layers}}$", 12.5, None, false)).clicked() {
                d.scale = 1.0 / (d.depth.max(1) as f64).sqrt();
            }
            ui::note(ui, r"Each layer adds this times its branch to the stream. At 1 the stream grows with depth; $1/\sqrt{\text{layers}}$ keeps it from growing.");
        }
        if mlp {
            let mut init = Init::ALL.iter().position(|i| *i == d.init).unwrap_or(0);
            if ui::choice(ui, &["LeCun", "Xavier", "He", "orthogonal"], &mut init) {
                d.init = Init::ALL[init];
            }
        }
        let mut gain = d.gain as f32;
        if ui::slider(ui, "gain on the starting weights", &mut gain, 0.0..=3.0) {
            d.gain = gain as f64;
        }
        ui::note(
            ui,
            if mlp {
                r"Starting weights are drawn with variance $\text{gain}^2$ times $1/n_\text{in}$ (LeCun), $2/(n_\text{in} + n_\text{out})$ (Xavier) or $2/n_\text{in}$ (He), or are an orthogonal matrix times the gain."
            } else {
                r"The transformer's weights start with variance $\text{gain}^2/n_\text{in}$, $n_\text{in}$ the number of inputs of each."
            },
        );

        ui::section(ui, "Training");
        let mut optim = (d.optim == Optim::Adam) as usize;
        if ui::choice(ui, &["plain gradient descent", "Adam"], &mut optim) {
            d.optim = if optim == 1 { Optim::Adam } else { Optim::Sgd };
        }
        let mut lr = d.lr as f32;
        if ui::slider_log_or_zero(ui, "learning rate", &mut lr, 1e-4, 1.0) {
            d.lr = lr as f64;
        }
        ui::note(ui, "Plain descent moves each weight by the rate times its gradient; Adam moves each typically by the rate or less, whatever the gradient's size.");
        let most_batch = if mlp { MLP_BATCH } else { TF_BATCH };
        ui::slider_int(ui, if mlp { "examples per step" } else { "sequences per step" }, &mut d.batch, 1..=most_batch);
        let mut clip = d.clip as f32;
        if ui::slider_log_or_zero(ui, "clip the gradient at", &mut clip, 0.01, 100.0) {
            d.clip = clip as f64;
        }
        ui::note(ui, "At 0, no clipping; otherwise a gradient longer than this is shortened to it before the step.");
        ui::slider_log(ui, "steps a second", &mut self.speed, 1.0..=1000.0);
        ui::note(ui, "As many as a frame has time for: large networks train more slowly than asked.");
        ui.horizontal_wrapped(|ui| {
            if ui.button(if self.running { "❚❚ pause" } else { "▶ train" }).clicked() {
                self.running = !self.running;
                if self.stop_at.is_some_and(|s| self.a.run.steps >= s) {
                    self.stop_at = None;
                }
            }
            if ui.button("one step").clicked() {
                self.one_step();
            }
            if ui.button("↺ start again").clicked() {
                self.start_again();
            }
        });

        ui::section(ui, "Design $B$: kept for comparison");
        let has_b = self.kept[self.design.task.index()].is_some();
        ui.horizontal_wrapped(|ui| {
            if ui.button(ui::label_text(ui, "keep $A$ as $B$", 12.5, None, false)).clicked() {
                self.keep(true);
            }
            if has_b && ui.button(ui::label_text(ui, "forget $B$", 12.5, None, false)).clicked() {
                self.keep(false);
            }
        });
        ui::note(ui, "$B$ is drawn in grey beside whatever $A$ becomes next. Every change to $A$ starts both again from their first step.");

        ui::section(ui, "Reading the picture");
        ui::key(
            ui,
            &[
                (FORWARD, "cyan: the signal going forward, the size of what each layer hands on"),
                (BACKWARD, "pink: the gradient of the loss coming back, its size at each layer"),
                (WEIGHT, "yellow: the weights: linear layers in the diagram, and for each weight matrix its gradient or its step"),
                (LOSS, "orange: the loss as training goes"),
                (NORMED, "purple: a normalisation"),
                (SKIP, "green: a skip connection and its sum"),
                (KEPT, "grey, dashed: design $B$, kept for comparison"),
                (pal::WHITE, "a white band: the layer opened (click another)"),
            ],
        );
        ui::section(ui, "Words");
        ui::term(ui, "Root mean square:", r"the square root of the mean of the squares, $\sqrt{(x_1^2 + \dots + x_n^2)/n}$: a typical size, over every unit and every example of the batch.");
        ui::term(ui, "Shut:", "a unit whose activation's slope is under 0.01 in size for an example: it passes back almost none of the gradient. A ReLU below 0 is shut; a tanh or a sigmoid far from 0 is saturated, which is shut too.");
        ui::term(ui, "Dead:", "a unit shut for every example of the batch.");
        ui::term(ui, "Normalisation:", "rescaling what passes through to a standard size: each unit over the batch (batch), each example over its units (layer), or each example divided by its root mean square (RMS).");
        ui::term(ui, "Skip connection:", "a layer that adds its result to its input, so that the input also goes round it.");
        ui::term(ui, "Gain:", "a factor on all the starting weights.");
        ui::term(ui, "Jacobian:", "the matrix of how each output of a map moves with each input; its singular values say how much it stretches each direction.");
        ui::term(ui, "Step relative to the weights:", "how far one training step moves a weight matrix, divided by the matrix's own size.");
        ui::term(ui, "Clipping:", "shortening the whole gradient to a set length when it is longer.");
        ui::term(ui, "The probe's batch:", "the examples every chart is measured on, the same for $A$ and $B$ at every step: the first 32 of the training set, or four fixed sequences.");

        ui::section(ui, "Numbers");
        ui::readout(ui, "training steps", self.a.run.steps.to_string());
        for (name, side, color) in [("$A$", Some(&self.a), LOSS), ("$B$", self.b.as_ref(), KEPT)] {
            let Some(side) = side else { continue };
            let r = &side.run;
            let text = match (r.diverged, r.last) {
                (Some(s), _) => format!("not a number since step {s}"),
                (None, Some((loss, right))) if right.is_finite() => format!("{} · {} % right", num(loss, 3), num(100.0 * right, 0)),
                (None, Some((loss, _))) => num(loss, 3),
                (None, None) => format!("{}, before training", num(side.measure.loss, 3)),
            };
            ui::readout_series(ui, &format!("{name}: loss"), text, color);
            ui::readout(ui, &format!("  {name}: weights and biases"), count(r.model.n_params()));
        }
        ui::readout(ui, "loss of a network that knows nothing", num(self.data.chance, 3));
        let layer = self.layer;
        for (name, side) in [("$A$", Some(&self.a)), ("$B$", self.b.as_ref())] {
            let Some(l) = side.and_then(|s| s.measure.layers.get(layer)) else { continue };
            ui::readout(ui, &format!("{name}, layer {layer}: size, gradient"), format!("{}, {}", sci(l.rms), sci(l.grad)));
            if let (Some(shut), Some(dead)) = (l.shut, l.dead) {
                ui::readout(ui, &format!("  {name}: shut, dead"), format!("{} %, {} %", num(100.0 * shut, 0), num(100.0 * dead, 0)));
            }
        }
        ui::see_also(
            ui,
            &[
                ("gradflow", "why a deep plain network's gradient dies or explodes"),
                ("normgrad", "what a normalisation does to the gradient"),
                ("residual", "skip connections, and the roads round each layer"),
                ("tfgrad", "the gradient inside a transformer"),
                ("mlp", "the other playground: a network learning points of the plane"),
            ],
        );
    }
}

/// A positive number as a formula, `2.4 \times 10^{-3}` (a plain number between 0.01 and
/// 1000), for read-outs and labels.
fn sci(v: f64) -> String {
    if !v.is_finite() {
        return "?".to_string();
    }
    if v == 0.0 {
        return "0".to_string();
    }
    let a = v.abs();
    if (0.01..1000.0).contains(&a) {
        return num(v, if a < 0.1 { 3 } else if a < 10.0 { 2 } else { 1 });
    }
    let e = a.log10().floor();
    let m = v / 10f64.powf(e);
    let (m, e) = if (m.abs() * 10.0).round() >= 100.0 { (m / 10.0, e + 1.0) } else { (m, e) };
    format!("${} \\times 10^{{{}}}$", num(m, 1), num(e, 0))
}

// ----- buttons, clicks and glides ----------------------------------------------------------

impl Lab {
    /// The buttons over the picture.
    fn buttons(&mut self, f: &mut Frame, lay: &Layout) {
        let size = lay.small;
        let has_b = self.b.is_some();
        let keep_text = if lay.narrow {
            if has_b {
                "keep $A$ as $B$"
            } else {
                "keep as $B$"
            }
        } else {
            "keep $A$ as $B$"
        };
        let (k0, k1) = lay.keep;
        let (keep, forget) = if has_b {
            let mid = if lay.narrow { (k0.x + k1.x) * 0.5 } else { (k0.y + k1.y) * 0.5 };
            if lay.narrow {
                ((k0, Vec2::new(mid - 3.0, k1.y)), Some((Vec2::new(mid + 3.0, k0.y), k1)))
            } else {
                ((k0, Vec2::new(k1.x, k1.y)), Some((Vec2::new(k0.x, k1.y + 6.0), Vec2::new(k1.x, k1.y + 6.0 + (k1.y - k0.y)))))
            }
        } else {
            ((k0, k1), None)
        };
        if plot::button(f, keep, keep_text, size, false) {
            self.keep(true);
        }
        if let Some(r) = forget {
            if plot::button(f, r, "forget $B$", size, false) {
                self.keep(false);
            }
        }
        for (cell, chart) in lay.cells.clone() {
            let tabbed = chart.is_none();
            let chart = chart.unwrap_or(self.tab);
            let mut y = cell.0.y + 6.0;
            if tabbed {
                let names: Vec<&str> = lay.tabbed.iter().map(|c| c.tab()).collect();
                let rects = plot::even_row(names.len(), cell.0.x + 6.0, cell.1.x - 6.0, y, 22.0, 4.0);
                for (k, r) in rects.into_iter().enumerate() {
                    let c = lay.tabbed[k];
                    if plot::button(f, r, names[k], size, self.tab == c) {
                        self.tab = c;
                    }
                }
                y += 28.0;
            }
            match chart {
                Chart::Training => {
                    let words = [if self.running { "❚❚ pause" } else { "▶ train" }, "one step", "↺ start again"];
                    // (the first as wide as either of its words, so that nothing moves when it changes)
                    let rects = plot::button_row(f, &["❚❚ pause", "one step", "↺ start again"], cell.0.x + 10.0, y + 22.0, 24.0, size, 6.0);
                    let first = f.text_size("▶ train", size).x.max(f.text_size("❚❚ pause", size).x) + 18.0;
                    let shift = first - (rects[0].1.x - rects[0].0.x);
                    let rects: Vec<Rect> = rects
                        .iter()
                        .enumerate()
                        .map(|(k, r)| if k == 0 { (r.0, Vec2::new(r.0.x + first, r.1.y)) } else { (r.0 + Vec2::new(shift, 0.0), r.1 + Vec2::new(shift, 0.0)) })
                        .collect();
                    if plot::button(f, rects[0], words[0], size, self.running) {
                        self.running = !self.running;
                        if self.stop_at.is_some_and(|s| self.a.run.steps >= s) {
                            self.stop_at = None;
                        }
                    }
                    if plot::button(f, rects[1], words[1], size, false) {
                        self.running = false;
                        self.one_step();
                    }
                    if plot::button(f, rects[2], words[2], size, false) {
                        self.start_again();
                    }
                }
                Chart::Signal | Chart::Weights => {
                    let words = if chart == Chart::Signal { ["size", "mean"] } else { ["gradient", "step"] };
                    let on = if chart == Chart::Signal { self.show_mean } else { self.show_steps };
                    let right = cell.1.x - 8.0;
                    let w0 = f.text_size(words[0], size).x + 14.0;
                    let w1 = f.text_size(words[1], size).x + 14.0;
                    let r1 = (Vec2::new(right - w1, y), Vec2::new(right, y + 20.0));
                    let r0 = (Vec2::new(right - w1 - 4.0 - w0, y), Vec2::new(right - w1 - 4.0, y + 20.0));
                    let mut pick = None;
                    if plot::button(f, r0, words[0], size, !on) {
                        pick = Some(false);
                    }
                    if plot::button(f, r1, words[1], size, on) {
                        pick = Some(true);
                    }
                    if let Some(p) = pick {
                        if chart == Chart::Signal {
                            self.show_mean = p;
                        } else {
                            self.show_steps = p;
                        }
                    }
                }
                _ => {}
            }
        }
    }

    /// A click on a chart of layers opens that layer.
    fn clicks(&mut self, f: &mut Frame, lay: &Layout) {
        let Some(at) = plot::pointer(f) else { return };
        for (cell, chart) in &lay.cells {
            let chart = chart.unwrap_or(self.tab);
            if !matches!(chart, Chart::Signal | Chart::Gradient | Chart::Weights) {
                continue;
            }
            let tabbed = lay.cells.iter().any(|c| c.0 == *cell && c.1.is_none());
            let p = self.plot_rect(lay, *cell, chart, tabbed);
            if !plot::within(at, p, 6.0) {
                continue;
            }
            f.handles.claim_clicks();
            if f.ptr.clicked {
                let x = self.range_now(chart);
                let pl = Plot::new(p, [x[0] as f64, x[1] as f64], [0.0, 1.0]);
                let layer = pl.x_at(at.x).round().max(0.0) as usize;
                self.layer = layer.min(self.design.stops() - 1);
                f.ptr.clicked = false;
                if !lay.tabbed.is_empty() && lay.tabbed.contains(&Chart::Layer) && self.tab != Chart::Layer && lay.cells.iter().all(|c| c.1 != Some(Chart::Layer)) {
                    self.tab = Chart::Layer;
                }
            }
        }
    }

    /// Where a chart's ranges are going.
    fn range_goal(&self, chart: Chart) -> [f32; 4] {
        let g = self.look.goals.get(chart as usize);
        [0, 1, 2, 3].map(|k| g.and_then(|g| g.get(k).copied()).unwrap_or(k as f32 % 2.0))
    }

    /// Whether a logarithmic chart has a row of zeros along its bottom.
    fn zero_row(&self, chart: Chart) -> bool {
        self.look.goals.get(chart as usize).and_then(|g| g.get(4)).is_some_and(|z| *z > 0.5)
    }

    /// The x range a chart has now (as drawn).
    fn range_now(&self, chart: Chart) -> [f32; 4] {
        let i = chart as usize;
        let g = self.look.ranges.get(i);
        [g.map_or(0.0, |g| g.get(0)), g.map_or(1.0, |g| g.get(1)), g.map_or(0.0, |g| g.get(2)), g.map_or(1.0, |g| g.get(3))]
    }

    /// Bring what is drawn nearer to what was measured.
    fn follow(&mut self, dt: f32, snap: bool) {
        // (while training, new numbers come every `PROBE_EVERY` seconds: they are
        // followed more slowly, so that the picture moves on evenly between them)
        let dt = if snap {
            1e6
        } else if self.running {
            dt * 0.3
        } else {
            dt
        };
        let mut moving = self.look.a.follow(&self.a, dt);
        if let Some(b) = &self.b {
            moving |= self.look.b.follow(b, dt);
        } else {
            self.look.b = Shown::default();
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

    /// Where each chart's ranges are going: whole decades round what is shown.
    fn range_goals(&self) -> Vec<Vec<f32>> {
        let sides: Vec<&Side> = std::iter::once(&self.a).chain(self.b.as_ref()).collect();
        // whole decades round the numbers, and one more below for a row of zeros
        let decades = |vals: &mut dyn Iterator<Item = f64>| {
            let (mut lo, mut hi, mut zero) = (f64::INFINITY, f64::NEG_INFINITY, false);
            for v in vals {
                let l = v.log10();
                if l.is_finite() {
                    lo = lo.min(l.max(FLOOR));
                    hi = hi.max(l);
                }
                zero |= v == 0.0;
            }
            if !(lo.is_finite() && hi.is_finite()) {
                return (-1.0f32, 1.0f32, zero);
            }
            let (mut a, mut b) = ((lo - 0.15).floor(), (hi + 0.15).ceil());
            if b - a < 2.0 {
                a = (a - 1.0).max(FLOOR - 1.0);
                b = a + 2.0f64.max(b - a);
            }
            if zero {
                a -= 1.0;
            }
            (a as f32, b as f32, zero)
        };
        let flag = |z: bool| if z { 1.0 } else { 0.0 };
        let stops = sides.iter().map(|s| s.run.design.stops()).max().unwrap_or(2);
        let xs = [-0.4f32, (stops - 1) as f32 + 0.4];
        let mut out = Vec::new();
        // signal: size (log) or mean
        let y = if self.show_mean {
            let m = sides.iter().flat_map(|s| s.measure.layers.iter().map(|l| l.mean)).filter(|v| v.is_finite()).fold(0.0f64, |a, b| a.max(b.abs()));
            let m = if m > 0.0 { plot::round_step(m * 1.1) } else { 1.0 };
            (-(m as f32), m as f32, false)
        } else {
            decades(&mut sides.iter().flat_map(|s| s.measure.layers.iter().map(|l| l.rms)))
        };
        out.push(vec![xs[0], xs[1], y.0, y.1, flag(y.2)]);
        let y = decades(&mut sides.iter().flat_map(|s| s.measure.layers.iter().map(|l| l.grad)));
        out.push(vec![xs[0], xs[1], y.0, y.1, flag(y.2)]);
        // training: steps and the loss (log)
        let steps = self.a.run.steps.max(self.stop_at.unwrap_or(0)).max(10) as f32;
        let y = decades(&mut sides.iter().flat_map(|s| s.run.losses.points().iter().map(|v| *v as f64)).chain([self.data.chance]));
        out.push(vec![0.0, steps, y.0, y.1, flag(y.2)]);
        // weights
        let y = if self.show_steps {
            decades(&mut sides.iter().flat_map(|s| s.steps.iter().copied()))
        } else {
            decades(&mut sides.iter().flat_map(|s| s.measure.groups.iter().map(|g| g.grad)))
        };
        out.push(vec![xs[0], xs[1], y.0, y.1, flag(y.2)]);
        // the Jacobian's singular values
        let n = sides.iter().map(|s| s.sv.len()).max().unwrap_or(1).max(2) as f32;
        let y = decades(&mut sides.iter().flat_map(|s| s.sv.iter().copied()));
        out.push(vec![0.5, n + 0.5, y.0, y.1, flag(y.2)]);
        // the layer's histograms: their ranges and tallest bin
        let tall = sides.iter().flat_map(|s| s.measure.hists.values.shares.iter().chain(&s.measure.hists.grads.shares)).fold(0.0f64, |a, b| a.max(*b));
        out.push(vec![0.0, 1.0, 0.0, (tall.max(0.05) * 1.1) as f32, 0.0]);
        out
    }
}

// ----- drawing ---------------------------------------------------------------------------

/// One piece of a diagram, left to right.
#[derive(Clone, Debug, PartialEq)]
enum Piece {
    /// A box: its words, its colour, and the activation whose curve it shows.
    Block(String, Color, Option<Act>),
    /// The sum where a skip connection rejoins the road.
    Plus,
}

/// A design drawn as boxes: the pieces, the skip connections (from the road before
/// piece `.0` to the sum at piece `.1`) and the stretch that repeats (pieces `.0` to
/// `.1`, so many times).
#[derive(Clone, Debug, PartialEq)]
struct Diagram {
    pieces: Vec<Piece>,
    skips: Vec<(usize, usize)>,
    repeat: (usize, usize, usize),
}

fn norm_word(n: Norm, short: bool) -> String {
    match (n, short) {
        (Norm::Batch, false) => "batch norm",
        (Norm::Layer, false) => "layer norm",
        (Norm::Rms, false) => "RMS norm",
        (Norm::Batch, true) => "BN",
        (Norm::Layer, true) => "LN",
        (Norm::Rms, true) => "RMS",
        (Norm::None, _) => "",
    }
    .to_string()
}

fn diagram_of(d: &Design, short: bool) -> Diagram {
    let block = |t: &str, c: Color| Piece::Block(t.to_string(), c, None);
    let mut pieces = Vec::new();
    let mut skips = Vec::new();
    let normed = d.norm != Norm::None;
    let norm = || Piece::Block(norm_word(d.norm, short), NORMED, None);
    let act = || Piece::Block(d.act.name().to_string(), FORWARD, Some(d.act));
    if d.task.mlp() {
        pieces.push(block(if short { "x" } else { "data" }, FORWARD));
        if d.residual {
            pieces.push(block(if short { "map" } else { "input map" }, WEIGHT));
        }
        let start = pieces.len();
        let linear = || Piece::Block(if short { "lin" } else { "linear" }.to_string(), WEIGHT, None);
        match (d.residual, d.place) {
            (false, Place::Post) => {
                pieces.push(linear());
                if normed {
                    pieces.push(norm());
                }
                pieces.push(act());
            }
            (false, Place::Pre) => {
                if normed {
                    pieces.push(norm());
                }
                pieces.push(linear());
                pieces.push(act());
            }
            (true, place) => {
                if normed && place == Place::Pre {
                    pieces.push(norm());
                }
                pieces.push(linear());
                pieces.push(act());
                if d.scale != 1.0 {
                    pieces.push(block(&format!("×{}", trim(d.scale, 2)), SKIP));
                }
                pieces.push(Piece::Plus);
                skips.push((start, pieces.len() - 1));
                if normed && place == Place::Post {
                    pieces.push(norm());
                }
            }
        }
        let end = pieces.len() - 1;
        pieces.push(block(if short { "out" } else { "read-out" }, WEIGHT));
        pieces.push(block(
            match (d.task, short) {
                (Task::Teacher, false) => "squared error",
                (_, false) => "cross-entropy",
                _ => "loss",
            },
            LOSS,
        ));
        return Diagram { pieces, skips, repeat: (start, end, d.depth) };
    }
    pieces.push(block(if short { "tok" } else { "tokens" }, FORWARD));
    pieces.push(block(if short { "emb" } else { "embedding" }, WEIGHT));
    let start = pieces.len();
    for attention in [true, false] {
        let from = pieces.len();
        if normed && d.place == Place::Pre {
            pieces.push(norm());
        }
        if attention {
            let text = if short { "attn".to_string() } else { format!("attention, {} heads", d.heads) };
            pieces.push(Piece::Block(text, WEIGHT, None));
        } else {
            let text = if short { "MLP".to_string() } else { format!("MLP, {}", d.act.name()) };
            pieces.push(Piece::Block(text, WEIGHT, Some(d.act)));
        }
        if d.residual {
            pieces.push(Piece::Plus);
            skips.push((from, pieces.len() - 1));
        }
        if normed && d.place == Place::Post {
            pieces.push(norm());
        }
    }
    let end = pieces.len() - 1;
    if normed && d.place == Place::Pre {
        pieces.push(Piece::Block(if short { norm_word(d.norm, true) } else { format!("final {}", norm_word(d.norm, false)) }, NORMED, None));
    }
    pieces.push(block(if short { "out" } else { "unembedding" }, WEIGHT));
    pieces.push(block(if short { "loss" } else { "cross-entropy" }, LOSS));
    Diagram { pieces, skips, repeat: (start, end, d.depth) }
}

impl Lab {
    fn draw_diagrams(&mut self, f: &mut Frame, lay: &Layout) {
        let (min, max) = lay.diagram;
        let a_title = fit(f, &format!("$A$   {}", self.design.summary(lay.narrow)), lay.text, max.x - min.x);
        f.text_px(min, a_title, lay.text, pal::TEXT, Align2::LEFT_TOP);
        let box_h = if lay.narrow { 26.0 } else { 34.0 };
        let top = min.y + lay.text + 22.0;
        draw_diagram(f, &self.design, (Vec2::new(min.x, top), Vec2::new(max.x, top + box_h)), if lay.narrow { lay.small } else { lay.text }, true);
        let (bmin, bmax) = lay.diagram_b;
        match &self.b {
            Some(b) => {
                let d = b.run.design;
                let words = format!("$B$   {}", d.summary(lay.narrow));
                if lay.narrow {
                    let room = lay.keep.0.x - bmin.x - 6.0;
                    let text = fit(f, &words, lay.small, room);
                    f.text_px(Vec2::new(bmin.x, (bmin.y + bmax.y) * 0.5), text, lay.small, KEPT, Align2::LEFT_CENTER);
                } else {
                    let words = fit(f, &words, lay.small, bmax.x - bmin.x);
                    f.text_px(bmin, words, lay.small, KEPT, Align2::LEFT_TOP);
                    let y = bmin.y + lay.small + 16.0;
                    draw_diagram(f, &d, (Vec2::new(bmin.x, y), Vec2::new(bmax.x, y + 20.0)), lay.small - 1.0, false);
                }
            }
            None => {
                let text = if lay.narrow { "$B$   none kept" } else { "$B$   none kept: “keep $A$ as $B$” draws this design in grey beside the next" };
                f.text_px(Vec2::new(bmin.x, if lay.narrow { (bmin.y + bmax.y) * 0.5 } else { bmin.y }), text, lay.small, pal::DIM, if lay.narrow { Align2::LEFT_CENTER } else { Align2::LEFT_TOP });
            }
        }
    }
}

/// Words cut to fit `room` pixels, with an ellipsis (never inside a formula).
fn fit(f: &Frame, text: &str, size: f32, room: f32) -> String {
    if f.text_size(text, size).x <= room {
        return text.to_string();
    }
    // the places it may be cut: between characters, outside dollars
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

/// A design's diagram in the rectangle (its boxes as tall as it is), in its colours or
/// all in grey; it shrinks its words to fit, and failing that uses short ones.
fn draw_diagram(f: &mut Frame, d: &Design, (min, max): Rect, size: f32, coloured: bool) {
    let gap = if size < 11.0 { 12.0 } else { 16.0 };
    let pad = if size < 11.0 { 8.0 } else { 12.0 };
    let width_of = |f: &Frame, dg: &Diagram, size: f32| -> Vec<f32> {
        dg.pieces
            .iter()
            .map(|p| match p {
                Piece::Block(t, _, act) => f.text_size(t, size).x + pad + if act.is_some() { size * 1.6 } else { 0.0 },
                Piece::Plus => size + 6.0,
            })
            .collect()
    };
    let room = max.x - min.x;
    let mut dg = diagram_of(d, false);
    let total = |w: &[f32]| w.iter().sum::<f32>() + gap * (w.len().saturating_sub(1)) as f32;
    let full = size;
    let mut size = size;
    let mut widths = width_of(f, &dg, size);
    // smaller words first, down to 10 pixels; then short words, from the full size down
    while total(&widths) > room && size > 10.0 {
        size -= 0.5;
        widths = width_of(f, &dg, size);
    }
    if total(&widths) > room {
        dg = diagram_of(d, true);
        size = full;
        widths = width_of(f, &dg, size);
        while total(&widths) > room && size > 8.0 {
            size -= 0.5;
            widths = width_of(f, &dg, size);
        }
    }
    let gap = if total(&widths) > room { ((room - widths.iter().sum::<f32>()) / (widths.len().saturating_sub(1)).max(1) as f32).max(2.0) } else { gap };
    let h = max.y - min.y;
    let mid = min.y + h * 0.5;
    let grey = |c: Color| if coloured { c } else { KEPT.a(0.8) };
    let mut x = min.x;
    let mut lefts = Vec::with_capacity(widths.len());
    for (k, (p, w)) in dg.pieces.iter().zip(&widths).enumerate() {
        lefts.push(x);
        match p {
            Piece::Block(text, color, act) => {
                let c = grey(*color);
                let (lo, hi) = (Vec2::new(x, min.y), Vec2::new(x + w, max.y));
                hud_rect(f, lo, hi, c.a(if coloured { 0.16 } else { 0.08 }));
                let edge = [lo, Vec2::new(hi.x, lo.y), hi, Vec2::new(lo.x, hi.y), lo];
                plot::path_of(f, DIAGRAM, &edge, if coloured { 1.3 } else { 1.0 }, c.a(0.9));
                let mut tx = x + w * 0.5;
                if let Some(a) = act {
                    // the activation's curve, small, at the left of its name
                    let cw = size * 1.3;
                    let (c0, c1) = (Vec2::new(x + pad * 0.5, mid - h * 0.32), Vec2::new(x + pad * 0.5 + cw, mid + h * 0.32));
                    let pts: Vec<Vec2> = (0..=16)
                        .map(|i| {
                            let t = -3.0 + 6.0 * i as f64 / 16.0;
                            let y = a.f(t);
                            let y = (y / 3.0).clamp(-1.0, 1.0) as f32;
                            Vec2::new(c0.x + (c1.x - c0.x) * i as f32 / 16.0, mid - y * (c1.y - c0.y) * 0.5)
                        })
                        .collect();
                    plot::path_of(f, DIAGRAM, &pts, 1.2, c.mix(pal::WHITE, 0.3));
                    tx += cw * 0.5;
                }
                f.text_px(Vec2::new(tx, mid), text.as_str(), size, if coloured { c.mix(pal::WHITE, 0.45) } else { KEPT }, Align2::CENTER_CENTER);
            }
            Piece::Plus => {
                let c = grey(SKIP);
                let centre = Vec2::new(x + w * 0.5, mid);
                hud_circle(f, centre, w * 0.5, 1.4, c);
                hud_line(f, centre - Vec2::new(w * 0.25, 0.0), centre + Vec2::new(w * 0.25, 0.0), 1.4, c);
                hud_line(f, centre - Vec2::new(0.0, w * 0.25), centre + Vec2::new(0.0, w * 0.25), 1.4, c);
            }
        }
        if k + 1 < dg.pieces.len() {
            let a = Vec2::new(x + w + 1.0, mid);
            let b = Vec2::new(x + w + gap - 1.0, mid);
            if b.x - a.x > 4.0 {
                hud_arrow(f, a, b, 1.0, grey(pal::DIM));
            }
        }
        x += w + gap;
    }
    // skip connections: from the road before the first piece of the branch, over the
    // branch, down into the sum
    for &(from, to) in &dg.skips {
        let (Some(&l), Some(&r), Some(&rw)) = (lefts.get(from), lefts.get(to), widths.get(to)) else { continue };
        let lift = h * 0.5 + if coloured { 9.0 } else { 6.0 };
        let x0 = l - gap * 0.5;
        let x1 = r + rw * 0.5;
        let pts = [Vec2::new(x0, mid), Vec2::new(x0, mid - lift), Vec2::new(x1, mid - lift), Vec2::new(x1, mid - rw * 0.5 - 1.0)];
        plot::path_of(f, DIAGRAM, &pts, if coloured { 1.5 } else { 1.0 }, grey(SKIP));
    }
    // the stretch that repeats
    let (s, e, times) = dg.repeat;
    if let (Some(&l), Some(&r), Some(&rw)) = (lefts.get(s), lefts.get(e), widths.get(e)) {
        let y = max.y + 5.0;
        let (x0, x1) = (l - 3.0, r + rw + 3.0);
        let pts = [Vec2::new(x0, y - 3.0), Vec2::new(x0, y), Vec2::new(x1, y), Vec2::new(x1, y - 3.0)];
        plot::path_of(f, DIAGRAM, &pts, 1.0, grey(pal::DIM));
        let what = if d.task.mlp() { "layers" } else { "blocks" };
        f.text_px(Vec2::new((x0 + x1) * 0.5, y + 2.0), format!("{times} {what}"), size, if coloured { pal::DIM } else { KEPT.a(0.8) }, Align2::CENTER_TOP);
    }
}

impl Lab {
    /// The rectangle a chart's data is drawn in, inside its cell.
    fn plot_rect(&self, lay: &Layout, (min, max): Rect, chart: Chart, tabbed: bool) -> Rect {
        let top = min.y + if tabbed { 28.0 } else { 0.0 } + lay.big + 30.0;
        let top = if chart == Chart::Training { top + 26.0 } else { top };
        let bottom = max.y - 34.0 - if chart == Chart::Signal && !self.show_mean { 26.0 } else { 0.0 };
        let bottom = if chart == Chart::Training { bottom - 2.0 * (lay.small + 4.0) } else { bottom };
        let left = min.x + 46.0;
        let right = max.x - 12.0;
        (Vec2::new(left, top), Vec2::new(right.max(left + 1.0), bottom.max(top + 1.0)))
    }

    fn draw_chart(&mut self, f: &mut Frame, lay: &Layout, cell: Rect, chart: Chart, tabs: &[Chart]) {
        let tabbed = !tabs.is_empty();
        hud_panel(f, cell.0, cell.1);
        let rect = self.plot_rect(lay, cell, chart, tabbed);
        let title_y = cell.0.y + if tabbed { 34.0 } else { 6.0 };
        let r = self.range_now(chart);
        let title = match chart {
            Chart::Signal => {
                if self.show_mean {
                    "Forward: the signal's mean"
                } else {
                    "Forward: the signal's size"
                }
            }
            Chart::Gradient => "Backward: the gradient's size",
            Chart::Training => "Training: the loss",
            Chart::Weights => {
                if self.show_steps {
                    "Weights: their relative step"
                } else {
                    "Weights: their gradient"
                }
            }
            Chart::Jacobian => "Stretching: the Jacobian of layers 2 to $L$",
            Chart::Layer => "",
        };
        if chart != Chart::Layer {
            let room = if matches!(chart, Chart::Signal | Chart::Weights) { cell.1.x - cell.0.x - 150.0 } else { cell.1.x - cell.0.x - 16.0 };
            let t = fit(f, title, lay.big, room);
            f.text_px(Vec2::new(cell.0.x + 10.0, title_y + 10.0), t, lay.big, pal::TEXT, Align2::LEFT_CENTER);
        }
        let plot = Plot::new(rect, [r[0] as f64, r[1] as f64], [r[2] as f64, r[3] as f64]);
        match chart {
            Chart::Signal => self.draw_signal(f, lay, &plot),
            Chart::Gradient => self.draw_layers(f, lay, &plot, Chart::Gradient),
            Chart::Training => self.draw_training(f, lay, &plot, cell),
            Chart::Weights => self.draw_weights(f, lay, &plot),
            Chart::Jacobian => self.draw_jacobian(f, lay, &plot),
            Chart::Layer => self.draw_layer(f, lay, cell, title_y),
        }
    }

    /// The axis of layers, with the chosen one marked.
    fn layer_axis(&self, f: &mut Frame, lay: &Layout, plot: &Plot, goal: [f32; 4]) {
        x_ticks(f, lay, plot, goal);
        let (min, max) = plot.rect;
        let what = if self.design.task.mlp() { "layer" } else { "block" };
        f.text_px(Vec2::new(max.x, max.y + 8.0 + lay.small), what, lay.small, pal::DIM, Align2::RIGHT_TOP);
        let x = plot.px_x(self.layer as f64);
        if x >= min.x - 1.0 && x <= max.x + 1.0 {
            hud_rect_at(f, 1, Vec2::new(x - 5.0, min.y), Vec2::new(x + 5.0, max.y), pal::WHITE.a(0.07));
            line_band(f, Vec2::new(x, min.y), Vec2::new(x, max.y));
        }
    }

    /// Ticks at the powers of ten of a logarithmic y axis (see `x_ticks` for why they
    /// are chosen from the range's goal).
    fn log_axis(f: &mut Frame, lay: &Layout, plot: &Plot, goal: [f32; 4], zero: bool) {
        let (min, max) = plot.rect;
        plot::line_of(f, AXES, min, Vec2::new(min.x, max.y), 1.0, pal::DIM.a(0.8));
        let (lo, hi) = ((goal[2] as f64).ceil() as i64, (goal[3] as f64).floor() as i64);
        let span = (hi - lo).max(1);
        let room = plot.height() / span as f32;
        let every = if room > 26.0 { 1 } else if room > 13.0 { 2 } else { (span / 4).max(3) };
        // (a tick for every power of ten there can be, each always in the same place in
        // the batch: one that is not on the chart waits unseen at its edge)
        for e in (FLOOR as i64 - 2)..=31 {
            let y = plot.px_y(e as f64);
            let inside = y >= min.y - 0.5 && y <= max.y + 0.5 && e >= lo && e <= hi;
            let y = if y.is_finite() { y.max(min.y).min(max.y) } else { max.y };
            let (tick, grid) = if inside { (pal::DIM.a(0.8), pal::DIM.a(0.12)) } else { (Color::CLEAR, Color::CLEAR) };
            plot::line_of(f, AXES, Vec2::new(min.x - 4.0, y), Vec2::new(min.x, y), 1.0, tick);
            plot::line_of(f, AXES, Vec2::new(min.x, y), Vec2::new(max.x, y), 1.0, grid);
            if inside && zero && e == lo {
                // (the bottom row is where the zeros are: no power of ten)
                f.text_px(Vec2::new(min.x - 6.0, y), "0", lay.small, pal::DIM, Align2::RIGHT_CENTER);
                plot::line_of(f, AXES, Vec2::new(min.x - 4.0, y - 4.0), Vec2::new(min.x + 4.0, y - 8.0), 1.0, pal::DIM.a(0.8));
            } else if inside && e % every == 0 && !(zero && e == lo + 1 && every > 1) {
                let text = match e {
                    0 => "1".to_string(),
                    1 => "10".to_string(),
                    _ => format!("$10^{{{}}}$", num(e as f64, 0)),
                };
                f.text_px(Vec2::new(min.x - 6.0, y), text, lay.small, pal::DIM, Align2::RIGHT_CENTER);
            }
        }
    }

    /// A linear y axis, its ticks chosen from the range's goal.
    fn lin_axis(f: &mut Frame, lay: &Layout, plot: &Plot, goal: [f32; 4]) {
        let (min, max) = plot.rect;
        plot::line_of(f, AXES, min, Vec2::new(min.x, max.y), 1.0, pal::DIM.a(0.8));
        let ticks = plot::Ticks::of(goal[2] as f64, goal[3] as f64, plot.height() * 1.6);
        for v in ticks.values() {
            let y = plot.px_y(v);
            let inside = y >= min.y - 0.5 && y <= max.y + 0.5;
            let y = y.max(min.y).min(max.y);
            let (tick, grid) = if inside { (pal::DIM.a(0.8), pal::DIM.a(0.12)) } else { (Color::CLEAR, Color::CLEAR) };
            plot::line_of(f, AXES, Vec2::new(min.x - 4.0, y), Vec2::new(min.x, y), 1.0, tick);
            plot::line_of(f, AXES, Vec2::new(min.x, y), Vec2::new(max.x, y), 1.0, grid);
            if inside {
                f.text_px(Vec2::new(min.x - 6.0, y), ticks.name(v), lay.small, pal::DIM, Align2::RIGHT_CENTER);
            }
        }
    }

    /// The legend in a chart's corner: what A and B are drawn in.
    fn legend(&self, f: &mut Frame, lay: &Layout, plot: &Plot, a: Color) {
        let (min, max) = plot.rect;
        // (over the chart's top right corner, above the data)
        let at = Vec2::new(max.x - 4.0, min.y - 12.0);
        let mut x = at.x;
        if self.b.is_some() {
            let w = f.text_size("$B$", lay.small).x;
            f.text_px(Vec2::new(x, at.y), "$B$", lay.small, KEPT, Align2::RIGHT_CENTER);
            x -= w + 4.0;
            plot::dashed_of(f, MARKS, Vec2::new(x - 16.0, at.y), Vec2::new(x, at.y), 1.5, KEPT);
            x -= 26.0;
        }
        let w = f.text_size("$A$", lay.small).x;
        f.text_px(Vec2::new(x, at.y), "$A$", lay.small, a, Align2::RIGHT_CENTER);
        x -= w + 4.0;
        plot::line_of(f, MARKS, Vec2::new((x - 16.0).max(min.x), at.y), Vec2::new(x, at.y), 2.0, a);
    }

    /// A series along the layer axis: a line and dots, A's solid and in colour, B's dashed
    /// and grey.
    /// (A number that is not one is left out, and breaks the line; a 0 sits on the
    /// bottom edge.)
    fn series(f: &mut Frame, plot: &Plot, xs: &[f32], ys: &[f32], a: bool, color: Color) {
        let (lo, hi) = (plot.y[0], plot.y[1]);
        let pts: Vec<Vec2> =
            xs.iter().zip(ys).map(|(x, y)| if *y >= NOT_A_NUMBER - 1.0 { Vec2::NAN } else { plot.at(*x as f64, (*y as f64).max(lo).min(hi)) }).collect();
        let c = if a { color } else { KEPT };
        for piece in pts.split(|p| !p.is_finite()).filter(|piece| piece.len() >= 2) {
            let points: Vec<Vec3> = piece.iter().map(|p| f.hud(*p)).collect();
            plot::apart(f, CURVES, !a).polyline(points, if a { 1.8 } else { 1.4 }, c);
        }
        for p in pts.into_iter().filter(|p| p.is_finite()) {
            plot::disc(f, p, if a { 5.5 } else { 4.0 }, c);
        }
    }

    fn draw_signal(&mut self, f: &mut Frame, lay: &Layout, plot: &Plot) {
        if self.show_mean {
            Lab::lin_axis(f, lay, plot, self.range_goal(Chart::Signal));
            plot.hline(f, MARKS, 0.0, 1.0, pal::DIM.a(0.5), false);
        } else {
            Lab::log_axis(f, lay, plot, self.range_goal(Chart::Signal), self.zero_row(Chart::Signal));
        }
        self.layer_axis(f, lay, plot, self.range_goal(Chart::Signal));
        for (shown, a) in [(&self.look.b, false), (&self.look.a, true)] {
            if !a && self.b.is_none() {
                continue;
            }
            let ys = if self.show_mean { &shown.mean.shown } else { &shown.rms.shown };
            let xs: Vec<f32> = (0..ys.len()).map(|i| i as f32).collect();
            Lab::series(f, plot, &xs, ys, a, FORWARD);
        }
        self.legend(f, lay, plot, FORWARD);
        // the share shut, as bars under the chart
        if !self.show_mean {
            let (min, max) = plot.rect;
            let strip = Plot::new((Vec2::new(min.x, max.y + 30.0 + lay.small), Vec2::new(max.x, max.y + 48.0 + lay.small)), plot.x, [0.0, 1.0]);
            plot::line_of(f, AXES, Vec2::new(strip.rect.0.x, strip.rect.1.y), strip.rect.1, 1.0, pal::DIM.a(0.5));
            f.text_px(Vec2::new(min.x - 6.0, (strip.rect.0.y + strip.rect.1.y) * 0.5), "shut", lay.small, FORWARD.a(0.8), Align2::RIGHT_CENTER);
            for (shown, a) in [(&self.look.a, true), (&self.look.b, false)] {
                if !a && self.b.is_none() {
                    continue;
                }
                let side = if a { Some(&self.a) } else { self.b.as_ref() };
                let Some(side) = side else { continue };
                for (i, s) in shown.shut.shown.iter().enumerate() {
                    if side.measure.layers.get(i).is_none_or(|l| l.shut.is_none()) {
                        continue;
                    }
                    let (x0, x1) = if a { (i as f64 - 0.38, i as f64 + 0.02) } else { (i as f64 - 0.02, i as f64 + 0.38) };
                    strip.bar(f, x0, x1, 0.0, (*s as f64).clamp(0.0, 1.0), 0.0, if a { FORWARD.a(0.7) } else { KEPT.a(0.55) });
                }
            }
        }
    }

    fn draw_layers(&mut self, f: &mut Frame, lay: &Layout, plot: &Plot, chart: Chart) {
        Lab::log_axis(f, lay, plot, self.range_goal(chart), self.zero_row(chart));
        self.layer_axis(f, lay, plot, self.range_goal(chart));
        for (shown, a) in [(&self.look.b, false), (&self.look.a, true)] {
            if !a && self.b.is_none() {
                continue;
            }
            let ys = &shown.grad.shown;
            let xs: Vec<f32> = (0..ys.len()).map(|i| i as f32).collect();
            Lab::series(f, plot, &xs, ys, a, BACKWARD);
        }
        self.legend(f, lay, plot, BACKWARD);
        if plot.height() > 100.0 {
            let words = "← the gradient comes back from the loss";
            f.text_px(Vec2::new(plot.rect.1.x, plot.rect.0.y + 4.0), words, lay.small - 0.5, pal::DIM, Align2::RIGHT_TOP);
        }
    }

    fn draw_weights(&mut self, f: &mut Frame, lay: &Layout, plot: &Plot) {
        Lab::log_axis(f, lay, plot, self.range_goal(Chart::Weights), self.zero_row(Chart::Weights));
        self.layer_axis(f, lay, plot, self.range_goal(Chart::Weights));
        for (side, shown, a) in [(self.b.as_ref(), &self.look.b, false), (Some(&self.a), &self.look.a, true)] {
            let Some(side) = side else { continue };
            let ys = if self.show_steps { &shown.w_step.shown } else { &shown.w_grad.shown };
            for part in [0u8, 1] {
                let pick: Vec<(f32, f32)> = side
                    .measure
                    .groups
                    .iter()
                    .zip(ys)
                    .filter(|(g, _)| g.part == part)
                    .map(|(g, y)| (g.layer as f32 + if side.run.design.task.mlp() { 0.0 } else if part == 0 { -0.15 } else { 0.15 }, *y))
                    .collect();
                if pick.is_empty() {
                    continue;
                }
                let (xs, ys): (Vec<f32>, Vec<f32>) = pick.into_iter().unzip();
                let color = if part == 0 { WEIGHT } else { WEIGHT.mix(pal::ORANGE, 0.45) };
                Lab::series(f, plot, &xs, &ys, a, color);
            }
        }
        self.legend(f, lay, plot, WEIGHT);
        if !self.design.task.mlp() {
            let (min, max) = plot.rect;
            f.text_px(Vec2::new(min.x + 4.0, max.y - 4.0), "light: attention    darker: the MLP", lay.small - 0.5, WEIGHT.a(0.85), Align2::LEFT_BOTTOM);
        }
    }

    fn draw_jacobian(&mut self, f: &mut Frame, lay: &Layout, plot: &Plot) {
        Lab::log_axis(f, lay, plot, self.range_goal(Chart::Jacobian), self.zero_row(Chart::Jacobian));
        x_ticks(f, lay, plot, self.range_goal(Chart::Jacobian));
        let (_, max) = plot.rect;
        f.text_px(Vec2::new(max.x, max.y + 8.0 + lay.small), "largest first", lay.small, pal::DIM, Align2::RIGHT_TOP);
        if self.a.sv.is_empty() && self.b.as_ref().is_none_or(|b| b.sv.is_empty()) {
            let words = if self.design.depth < 2 { "needs two hidden layers or more" } else { "not a number: the network has blown up" };
            f.text_px((plot.rect.0 + plot.rect.1) * 0.5, words, lay.text, pal::DIM, Align2::CENTER_CENTER);
            return;
        }
        for (shown, a) in [(&self.look.b, false), (&self.look.a, true)] {
            if !a && self.b.is_none() {
                continue;
            }
            let ys = &shown.sv.shown;
            let xs: Vec<f32> = (0..ys.len()).map(|i| i as f32 + 1.0).collect();
            Lab::series(f, plot, &xs, ys, a, BACKWARD);
        }
        self.legend(f, lay, plot, BACKWARD);
        plot.hline(f, MARKS, 0.0, 1.0, pal::DIM.a(0.6), true);
    }

    fn draw_training(&mut self, f: &mut Frame, lay: &Layout, plot: &Plot, cell: Rect) {
        Lab::log_axis(f, lay, plot, self.range_goal(Chart::Training), self.zero_row(Chart::Training));
        x_ticks(f, lay, plot, self.range_goal(Chart::Training));
        let (min, max) = plot.rect;
        f.text_px(Vec2::new(max.x, max.y + 8.0 + lay.small), "step", lay.small, pal::DIM, Align2::RIGHT_TOP);
        // the loss of a network that knows nothing
        let chance = self.data.chance.log10();
        plot.hline(f, MARKS, chance, 1.0, pal::DIM.a(0.7), true);
        let words = if self.design.task == Task::Teacher { "always the mean" } else { "knows nothing" };
        plot.text(f, plot.x[0], chance, Vec2::new(4.0, -3.0), words, lay.small - 0.5, pal::DIM, Align2::LEFT_BOTTOM);
        for (side, kind, a) in [(self.b.as_ref(), LOSS_B, false), (Some(&self.a), LOSS_A, true)] {
            let Some(side) = side else { continue };
            let pts = side.run.losses.points();
            let every = side.run.losses.every() as f64;
            let p: Vec<Vec2> = pts
                .iter()
                .enumerate()
                .map(|(i, v)| {
                    let y = if v.is_finite() && *v > 0.0 { (*v as f64).log10() } else if v.is_finite() { plot.y[0] } else { plot.y[1] };
                    plot.at((i as f64 + 0.5) * every, y.max(plot.y[0]).min(plot.y[1]))
                })
                .collect();
            if p.len() >= 2 {
                let points: Vec<Vec3> = p.iter().map(|q| f.hud(*q)).collect();
                plot::apart(f, kind, !a).polyline(points, if a { 1.8 } else { 1.4 }, if a { LOSS } else { KEPT });
            }
        }
        self.legend(f, lay, plot, LOSS);
        // where training stands, under the chart
        let mut y = max.y + 30.0;
        let step = self.a.run.steps;
        let status = match self.stop_at {
            Some(s) if !self.running && step >= s => format!("step {step}: done"),
            _ if self.running || step == 0 => format!("step {step}"),
            _ => format!("step {step}: paused"),
        };
        let row = cell.0.y + if lay.cells.iter().any(|c| c.0 == cell && c.1.is_none()) { 28.0 } else { 0.0 } + 6.0 + 22.0 + 12.0;
        f.text_px(Vec2::new(cell.1.x - 10.0, row), status, lay.small, pal::DIM, Align2::RIGHT_CENTER);
        for (name, side, color) in [("$A$", Some(&self.a), LOSS), ("$B$", self.b.as_ref(), KEPT)] {
            let Some(side) = side else { continue };
            let r = &side.run;
            let text = match (r.diverged, r.last) {
                (Some(s), _) => format!("{name}: its numbers stopped being numbers at step {s}"),
                (None, Some((loss, right))) if right.is_finite() => format!("{name}: loss {} · {} % of this step's examples right", num(loss, 3), num(100.0 * right, 0)),
                (None, Some((loss, _))) => format!("{name}: loss {}", num(loss, 3)),
                (None, None) => format!("{name}: loss {} on the probe's batch, before training", num(side.measure.loss, 3)),
            };
            let text = fit(f, &text, lay.small, cell.1.x - min.x);
            f.text_px(Vec2::new(min.x - 36.0, y), text, lay.small, color, Align2::LEFT_TOP);
            y += lay.small + 4.0;
        }
        let batch_one = self.a.run.design.norm == Norm::Batch && self.a.run.design.batch == 1 && self.design.task.mlp();
        if batch_one {
            f.text_px((min + max) * 0.5, "batch normalisation of one example makes every unit 0", lay.small, NORMED, Align2::CENTER_CENTER);
        }
    }

    fn draw_layer(&mut self, f: &mut Frame, lay: &Layout, cell: Rect, title_y: f32) {
        let l = self.layer;
        let mlp = self.design.task.mlp();
        let top = self.design.stops() - 1;
        let name = match (l, mlp) {
            (0, true) => "what enters layer 1".to_string(),
            (0, false) => "the stream after the embeddings".to_string(),
            (k, true) if k == top => "the read-out".to_string(),
            (k, false) if k == top => "the logits".to_string(),
            (k, true) => format!("layer {k}"),
            (k, false) => format!("block {k}"),
        };
        let title = fit(f, &format!("Inside {name}"), lay.big, cell.1.x - cell.0.x - 16.0);
        f.text_px(Vec2::new(cell.0.x + 10.0, title_y + 10.0), title, lay.big, pal::TEXT, Align2::LEFT_CENTER);
        let gate = self.a.measure.hists.gate;
        let (x0, x1) = (cell.0.x + 14.0, cell.1.x - 14.0);
        let y0 = title_y + lay.big + 34.0;
        let rows = 1 + self.b.is_some() as usize;
        let y1 = cell.1.y - (rows + 1) as f32 * (lay.small + 5.0) - 26.0;
        let mid = (x0 + x1) * 0.5;
        let ymax = self.range_now(Chart::Layer)[3] as f64;
        for (k, (lo, hi)) in [(x0, mid - 8.0), (mid + 8.0, x1)].into_iter().enumerate() {
            let values = k == 0;
            let half_a = self.look.a.halves.get(k) as f64;
            let half_b = if self.b.is_some() { self.look.b.halves.get(k) as f64 } else { 0.0 };
            let half = half_a.max(half_b).max(1e-300);
            let p = Plot::new((Vec2::new(lo, y0), Vec2::new(hi, y1)), [-half, half], [0.0, ymax]);
            if p.tiny() {
                continue;
            }
            plot::line_of(f, AXES, Vec2::new(p.rect.0.x, p.rect.1.y), p.rect.1, 1.0, pal::DIM.a(0.8));
            plot::line_of(f, AXES, Vec2::new(p.px_x(0.0), p.rect.1.y), Vec2::new(p.px_x(0.0), p.rect.1.y + 4.0), 1.0, pal::DIM.a(0.8));
            f.text_px(Vec2::new(p.px_x(0.0), p.rect.1.y + 5.0), "0", lay.small, pal::DIM, Align2::CENTER_TOP);
            for (edge, align) in [(-half, Align2::LEFT_TOP), (half, Align2::RIGHT_TOP)] {
                f.text_px(Vec2::new(p.px_x(edge), p.rect.1.y + 5.0), sci(edge), lay.small - 1.0, pal::DIM, align);
            }
            let what = match (values, gate.is_some()) {
                (true, true) => "input of the activation",
                (true, false) => "the numbers it hands on",
                (false, _) => "the gradient there",
            };
            f.text_px(Vec2::new(lo, y0 - 6.0), what, lay.small, if values { FORWARD } else { BACKWARD }, Align2::LEFT_BOTTOM);
            // A's bars, filled; B's outline, dashed
            let color = if values { FORWARD } else { BACKWARD };
            let shares_a = if values { &self.look.a.values.shown } else { &self.look.a.grads.shown };
            let bin_a = 2.0 * half_a / BINS as f64;
            for (i, s) in shares_a.iter().enumerate() {
                let x = -half_a + i as f64 * bin_a;
                p.bar(f, x, x + bin_a, 0.0, (*s as f64).min(ymax), 0.0, color.a(0.55));
            }
            if self.b.is_some() {
                let shares_b = if values { &self.look.b.values.shown } else { &self.look.b.grads.shown };
                let bin_b = 2.0 * half_b / BINS as f64;
                let mut pts = Vec::with_capacity(2 * BINS + 2);
                for (i, s) in shares_b.iter().enumerate() {
                    let x = -half_b + i as f64 * bin_b;
                    let y = (*s as f64).min(ymax);
                    pts.push(p.at(x, y));
                    pts.push(p.at(x + bin_b, y));
                }
                let points: Vec<Vec3> = pts.iter().map(|q| f.hud(*q)).collect();
                if points.len() >= 2 {
                    plot::apart(f, HIST, true).polyline(points, 1.2, KEPT);
                }
            }
            // over the activation's input, its slope: where it is near 0 the unit is shut
            if let (true, Some(act)) = (values, gate) {
                // (slope 1, or a sigmoid's 1/4, at four fifths of the height: GELU's and
                // SiLU's go a little above 1)
                let (unit, word) = if act == Act::Sigmoid { (0.25, "1/4") } else { (1.0, "1") };
                let height = p.rect.1.y - p.rect.0.y;
                let y_of = |s: f64| p.rect.1.y - height * (0.8 * s / unit).clamp(-0.05, 1.0) as f32;
                let n = 64;
                let pts: Vec<Vec2> = (0..=n)
                    .map(|i| {
                        let x = -half + 2.0 * half * i as f64 / n as f64;
                        Vec2::new(p.px_x(x), y_of(act.df(x)))
                    })
                    .collect();
                plot::path_of(f, HIST, &pts, 1.2, pal::WHITE.a(0.65));
                plot::dashed_of(f, HIST, Vec2::new(p.rect.1.x - 8.0, y_of(unit)), Vec2::new(p.rect.1.x, y_of(unit)), 1.0, pal::WHITE.a(0.6));
                f.text_px(Vec2::new(p.rect.1.x, y_of(unit) - 3.0), format!("slope {word}"), lay.small - 1.0, pal::WHITE.a(0.8), Align2::RIGHT_BOTTOM);
            }
        }
        // the numbers of this layer, as a table
        let y = y1 + 24.0;
        let heads = ["size", "mean", "gradient", "shut, dead"];
        let lead = 22.0;
        let col = (x1 - x0 - lead) / heads.len() as f32;
        let at = |k: usize| x0 + lead + col * (k as f32 + 0.5);
        for (k, h) in heads.iter().enumerate() {
            f.text_px(Vec2::new(at(k), y), *h, lay.small, pal::DIM, Align2::CENTER_TOP);
        }
        let mut y = y + lay.small + 5.0;
        for (name, side, color) in [("$A$", Some(&self.a), pal::TEXT), ("$B$", self.b.as_ref(), KEPT)] {
            let Some(st) = side.and_then(|s| s.measure.layers.get(l)) else { continue };
            f.text_px(Vec2::new(x0, y), name, lay.small, color, Align2::LEFT_TOP);
            let shares = match (st.shut, st.dead) {
                (Some(s), Some(d)) => format!("{} %, {} %", num(100.0 * s, 0), num(100.0 * d, 0)),
                _ => "–".to_string(),
            };
            let cells = [sci(st.rms), sci_signed(st.mean), sci(st.grad), shares];
            for (k, c) in cells.into_iter().enumerate() {
                f.text_px(Vec2::new(at(k), y), c, lay.small, color, Align2::CENTER_TOP);
            }
            y += lay.small + 5.0;
        }
    }
}

/// `sci` for a number of either sign.
fn sci_signed(v: f64) -> String {
    if v < 0.0 {
        let s = sci(-v);
        if let Some(rest) = s.strip_prefix('$') {
            format!("$-{rest}")
        } else {
            format!("−{s}")
        }
    } else {
        sci(v)
    }
}

/// Ticks along a chart's x axis. They are chosen from where its range is going, not
/// where it is, so that their number does not change while it glides there (`eidos
/// motion` would take that for a jerk); drawn where the range puts them now, kept to
/// the chart's edge and unseen while they are beyond it.
fn x_ticks(f: &mut Frame, lay: &Layout, plot: &Plot, goal: [f32; 4]) {
    if plot.tiny() {
        return;
    }
    let (min, max) = plot.rect;
    plot::line_of(f, AXES, Vec2::new(min.x, max.y), max, 1.0, pal::DIM.a(0.8));
    let ticks = plot::Ticks::of(goal[0].ceil().max(0.0) as f64, goal[1] as f64, plot.width());
    // (whole numbers only: layers, steps, the singular values' places)
    let step = ticks.step.max(1.0);
    let ticks = plot::Ticks::of(goal[0].ceil().max(0.0) as f64, goal[1] as f64, plot.width() * (ticks.step / step) as f32);
    for v in ticks.values() {
        let x = plot.px_x(v);
        let inside = x >= min.x - 0.5 && x <= max.x + 0.5;
        let x = x.max(min.x).min(max.x);
        plot::line_of(f, AXES, Vec2::new(x, max.y), Vec2::new(x, max.y + 4.0), 1.0, if inside { pal::DIM.a(0.8) } else { Color::CLEAR });
        if inside {
            f.text_px(Vec2::new(x, max.y + 6.0), ticks.name(v), lay.small, pal::DIM, Align2::CENTER_TOP);
        }
    }
}

/// The white line of the layer opened.
fn line_band(f: &mut Frame, a: Vec2, b: Vec2) {
    plot::line_of(f, BAND, a, b, 1.0, pal::WHITE.a(0.35));
}

#[cfg(test)]
mod tests {
    use super::*;

    /// An experiment as its button sets it up, measured at its start.
    fn start(k: usize) -> Lab {
        let mut lab = Lab::create();
        lab.set_up(k);
        lab.measure_all();
        lab
    }

    /// The same after `steps` training steps of both designs.
    fn after(k: usize, steps: u64) -> Lab {
        let mut lab = start(k);
        for _ in 0..steps {
            lab.one_step();
        }
        lab.measure_all();
        lab
    }

    /// An experiment trained to its end, as "train" runs it.
    fn trained(k: usize) -> Lab {
        let mut lab = start(k);
        lab.running = true;
        while lab.running {
            lab.train(0.05, 0.05);
        }
        lab.measure_all();
        lab
    }

    fn b(lab: &Lab) -> &Side {
        lab.b.as_ref().unwrap()
    }

    fn grads(side: &Side) -> Vec<f64> {
        side.measure.layers.iter().map(|l| l.grad).collect()
    }

    fn last(side: &Side) -> (f64, f64) {
        side.run.last.unwrap()
    }

    fn within(v: f64, lo: f64, hi: f64, what: &str) {
        assert!(v >= lo && v <= hi, "{what}: {v} is not in [{lo}, {hi}]");
    }

    #[test]
    fn a_deep_network_that_will_not_train() {
        let lab = start(0);
        let (gb, ga) = (grads(b(&lab)), grads(&lab.a));
        within(gb[20], 2.2e-3, 2.6e-3, "B's gradient at layer 20");
        within(gb[1], 1.5e-7, 1.9e-7, "B's gradient at layer 1");
        within((gb[1] / gb[20]).powf(1.0 / 19.0), 0.55, 0.65, "B's factor per layer");
        // the signal dies away too
        let rb: Vec<f64> = b(&lab).measure.layers.iter().map(|l| l.rms).collect();
        assert!(rb[20] < 1e-3 * rb[1], "B's signal {} at layer 20 against {} at layer 1", rb[20], rb[1]);
        // in A the gradient comes back undiminished: no smaller at layer 1 than at 20
        assert!(ga[1] >= ga[20], "A's gradient {} at layer 1, {} at layer 20", ga[1], ga[20]);
        let lab = trained(0);
        assert_eq!(lab.step_count(), 300);
        let (la, lb) = (last(&lab.a), last(b(&lab)));
        within(la.0, 0.004, 0.006, "A's loss after 300 steps");
        assert_eq!(la.1, 1.0, "A gets every example of its batch right");
        within(lb.0, 1.03, 1.07, "B's loss after 300 steps");
        within(lab.data.chance, 1.098, 1.099, "the loss of a network that knows nothing");
    }

    #[test]
    fn thirty_relu_layers_normalised() {
        let lab = start(1);
        let ra: Vec<f64> = lab.a.measure.layers[1..=30].iter().map(|l| l.rms).collect();
        let rb: Vec<f64> = b(&lab).measure.layers[1..=30].iter().map(|l| l.rms).collect();
        assert!(ra.iter().all(|r| (0.66..=0.75).contains(r)), "A's sizes {ra:?}");
        within(rb.iter().copied().fold(f64::INFINITY, f64::min), 0.25, 0.27, "B's smallest size");
        within(rb.iter().copied().fold(0.0, f64::max), 0.64, 0.66, "B's largest size");
        let (ga, gb) = (grads(&lab.a), grads(b(&lab)));
        within(ga[30], 5.5e-3, 5.9e-3, "A's gradient at layer 30");
        within(ga[1], 3.9, 4.3, "A's gradient at layer 1");
        within(ga[1] / ga[30], 650.0, 760.0, "A's growth");
        let hidden = &gb[1..=30];
        let spread = hidden.iter().copied().fold(0.0, f64::max) / hidden.iter().copied().fold(f64::INFINITY, f64::min);
        within(spread, 1.0, 6.0, "B's spread");
        let lab = trained(1);
        within(last(b(&lab)).0, 0.08, 0.1, "B's loss after 300 steps");
        within(last(&lab.a).0, 1.1, 1.14, "A's loss after 300 steps");
        // with skips, pre-norm and layer normalisation
        let mut lab = start(1);
        lab.design = Design { norm: Norm::Layer, place: Place::Pre, residual: true, ..lab.design };
        lab.apply();
        for _ in 0..300 {
            lab.one_step();
        }
        within(last(&lab.a).0, 0.003, 0.005, "A with skips after 300 steps");
    }

    #[test]
    fn post_norm_against_pre_norm() {
        let lab = start(2);
        within(grads(b(&lab))[1], 1.3e-2, 1.5e-2, "B's gradient at block 1 at the start");
        let lab = after(2, 20);
        let g = grads(b(&lab));
        within(g[1], 4e-6, 8e-6, "B's gradient at block 1 after 20 steps");
        within(g[8], 6.5e-3, 7.5e-3, "B's gradient at block 8 after 20 steps");
        let lab = trained(2);
        assert_eq!(lab.step_count(), 300);
        // the steps of block 1's attention and MLP, against block 8's
        let steps = |block: usize| -> Vec<f64> { b(&lab).measure.groups.iter().zip(&b(&lab).steps).filter(|(g, _)| g.layer == block).map(|(_, s)| *s).collect() };
        let (low, high) = (steps(1), steps(8));
        assert_eq!((low.len(), high.len()), (2, 2));
        assert!(low.iter().all(|l| high.iter().all(|h| 100.0 * l < *h)), "block 1 {low:?}, block 8 {high:?}");
        within(last(b(&lab)).0, 2.27, 2.31, "B's loss after 300 steps");
        within(lab.data.chance, 2.302, 2.303, "ln 10");
        let la = last(&lab.a);
        within(la.0, 0.002, 0.004, "A's loss after 300 steps");
        assert_eq!(la.1, 1.0, "A predicts every scored token");
    }

    #[test]
    fn sigmoids_everywhere() {
        let lab = start(3);
        let ga = grads(&lab.a);
        within(ga[8], 6.5e-3, 7.1e-3, "A's gradient at layer 8");
        within(ga[1], 2.3e-7, 2.7e-7, "A's gradient at layer 1");
        within((ga[8] / ga[1]).powf(1.0 / 7.0), 3.8, 4.8, "A's factor per layer");
        assert!(grads(b(&lab))[1..=8].iter().all(|g| (5e-3..=6.5e-3).contains(g)), "B's gradients {:?}", grads(b(&lab)));
        assert!(lab.a.measure.layers[1..=8].iter().all(|l| (0.45..=0.55).contains(&l.mean)), "A's means");
        let lab = trained(3);
        assert_eq!(lab.step_count(), 1000);
        within(last(b(&lab)).0, 0.0, 0.0015, "B's loss after 1000 steps");
        within(last(&lab.a).0, 1.09, 1.13, "A's loss after 1000 steps");
    }

    #[test]
    fn a_rate_too_large() {
        let lab = start(4);
        let (sa, sb) = (&lab.a.steps, &b(&lab).steps);
        // the groups: layer 1, layers 2 to 6, the read-out
        assert!(sa[1..].iter().all(|s| (0.28..=0.42).contains(s)), "A's first steps {sa:?}");
        assert!(sb[1..].iter().all(|s| (0.008..=0.013).contains(s)), "B's first steps {sb:?}");
        let lab = after(4, 2);
        within(last(&lab.a).0, 15.0, 17.5, "A's loss at step 2");
        let lab = after(4, 10);
        let dead: Vec<f64> = lab.a.measure.layers.iter().filter_map(|l| l.dead).collect();
        within(dead.iter().sum::<f64>() / dead.len() as f64, 0.7, 0.8, "A's dead share after 10 steps");
        let lab = trained(4);
        let layers = &lab.a.measure.layers;
        assert!(layers[2..=6].iter().all(|l| l.dead.unwrap() >= 31.0 / 32.0), "dead after 300 steps");
        assert!(layers[1..=5].iter().all(|l| l.grad == 0.0), "the gradient at layers 1 to 5");
        within(last(&lab.a).0, 1.09, 1.12, "A's loss after 300 steps");
        within(last(b(&lab)).0, 0.002, 0.004, "B's loss after 300 steps");
    }

    // ----- every gradient drawn, against differences of the same loss ---------------------

    /// The loss of `model` on the probe's batch (the MLP tasks).
    fn probe_loss(model: &Model, data: &Data) -> f64 {
        let n = PROBE_N.min(data.y.len());
        let mut tape = Tape::new();
        let trace = model.loss_mlp(&mut tape, &data.x[..n * data.inputs], &data.y[..n], data.loss());
        tape.value(trace.loss)[0]
    }

    /// Layers `from + 1` to the read-out of an MLP (whose arrangement is post, or
    /// residual) as a network of their own: fed what layer `from` hands on, it computes
    /// what the whole network computes from there. (A residual network's input map is
    /// made the identity.)
    fn tail(model: &Model, from: usize) -> Model {
        let spec = *model.mlp_spec().unwrap();
        let depth = spec.depth - from;
        let n = spec.width;
        let identity: Vec<f64> = (0..n * n).map(|k| if k % (n + 1) == 0 { 1.0 } else { 0.0 }).collect();
        if depth == 0 {
            // (the read-out alone, behind a layer that hands on what it is given)
            let plain = MlpSpec { inputs: n, depth: 1, act: Act::Identity, norm: Norm::None, place: Place::Post, residual: false, ..spec };
            let mut t = Model::mlp(&plain, &mut Rng::new(1));
            t.group_mut("layer 1 weights").unwrap().value = identity;
            t.group_mut("layer 1 bias").unwrap().value.fill(0.0);
            for w in ["read-out weights", "read-out bias"] {
                t.group_mut(w).unwrap().value = model.group(w).unwrap().value.clone();
            }
            return t;
        }
        let mut t = Model::mlp(&MlpSpec { inputs: spec.width, depth, ..spec }, &mut Rng::new(1));
        let names = |l: usize| ["weights", "bias", "norm gain", "norm bias"].map(|w| format!("layer {l} {w}"));
        for k in 1..=depth {
            for (to, from) in names(k).iter().zip(names(k + from)) {
                if let (Some(src), Some(dst)) = (model.group(&from).map(|g| g.value.clone()), t.group_mut(to)) {
                    dst.value = src;
                }
            }
        }
        for w in ["read-out weights", "read-out bias"] {
            t.group_mut(w).unwrap().value = model.group(w).unwrap().value.clone();
        }
        if spec.residual {
            t.group_mut("input map weights").unwrap().value = identity;
            t.group_mut("input map bias").unwrap().value.fill(0.0);
        }
        t
    }

    /// For small MLPs of several arrangements: the gradient at every layer (its root
    /// mean square, as the backward chart draws it), the length of every weight
    /// matrix's gradient (the weights chart), and the singular values of the Jacobian of
    /// layers 2 to L, each against central differences of the same loss or map.
    #[test]
    fn the_mlp_gradients_drawn_are_the_differences_of_the_loss() {
        let small = Design { depth: 4, width: 5, ..BASE };
        for d in [
            Design { act: Act::Tanh, norm: Norm::None, place: Place::Post, ..small },
            Design { act: Act::Relu, norm: Norm::Batch, place: Place::Post, init: Init::He, ..small },
            Design { act: Act::Gelu, norm: Norm::Layer, place: Place::Pre, residual: true, scale: 0.5, ..small },
            Design { task: Task::Teacher, act: Act::Sigmoid, norm: Norm::Rms, place: Place::Post, residual: true, ..small },
        ] {
            let data = Data::new(d.task, d.context);
            let mut run = Run::new(d);
            let mut tape = Tape::new();
            // (a few steps first, so that no bias or gain sits at its start)
            for _ in 0..5 {
                run.step(&data, &mut tape);
            }
            let model = &run.model;
            let m = Measure::of(model, &data, &mut tape, 1);
            let n = PROBE_N.min(data.y.len());
            let h = 1e-6;
            // each hidden layer's gradient, from a network made of the layers above it
            for l in 1..=d.depth {
                let trace = model.forward_mlp(&mut tape, &data.x[..n * data.inputs]);
                let out = tape.value(trace.get(Stage::Output, l).unwrap()).to_vec();
                let t = tail(model, l);
                let sub = Data { x: out.clone(), inputs: d.width, ..data.clone() };
                assert!((probe_loss(&t, &sub) - m.loss).abs() < 1e-12, "the tail from layer {l} computes the same loss");
                let mut sq = 0.0;
                for k in 0..out.len() {
                    let (mut up, mut down) = (sub.clone(), sub.clone());
                    up.x[k] += h;
                    down.x[k] -= h;
                    let g = (probe_loss(&t, &up) - probe_loss(&t, &down)) / (2.0 * h);
                    sq += g * g;
                }
                let fd = (sq / out.len() as f64).sqrt();
                let drawn = m.layers[l].grad;
                assert!((fd - drawn).abs() <= 1e-6 * drawn + 1e-12, "{}: layer {l}: drawn {drawn}, differences {fd}", d.summary(true));
            }
            // each weight matrix's gradient
            for g in &m.groups {
                let mut sq = 0.0;
                let len = model.groups()[g.first].value.len();
                for k in 0..len {
                    let (mut up, mut down) = (model.clone(), model.clone());
                    up.groups_mut()[g.first].value[k] += h;
                    down.groups_mut()[g.first].value[k] -= h;
                    let fd = (probe_loss(&up, &data) - probe_loss(&down, &data)) / (2.0 * h);
                    sq += fd * fd;
                }
                assert!((sq.sqrt() - g.grad).abs() <= 1e-6 * g.grad + 1e-12, "{}: weights of layer {}: drawn {}, differences {}", d.summary(true), g.layer, g.grad, sq.sqrt());
            }
            // the Jacobian of layers 2 to L at the first example (with batch
            // normalisation, in a batch of `JACOBIAN_BATCH`)
            let sv = singular_values(model, &data);
            let b = if d.norm == Norm::Batch { JACOBIAN_BATCH } else { 1 };
            let x1 = {
                let trace = model.forward_mlp(&mut tape, &data.x[..b * data.inputs]);
                tape.value(trace.get(Stage::Output, 1).unwrap()).to_vec()
            };
            let t = tail(model, 1);
            let top = |x: &[f64], tape: &mut Tape| {
                let trace = t.forward_mlp(tape, x);
                tape.value(trace.get(Stage::Output, d.depth - 1).unwrap())[..d.width].to_vec()
            };
            let w = d.width;
            let mut jac = vec![0.0; w * w];
            for c in 0..w {
                let (mut up, mut down) = (x1.clone(), x1.clone());
                up[c] += h;
                down[c] -= h;
                let (a, z) = (top(&up, &mut tape), top(&down, &mut tape));
                for r in 0..w {
                    jac[r * w + c] = (a[r] - z[r]) / (2.0 * h);
                }
            }
            let fd = crate::math::linalg::svd(&jac, w, w).1;
            for (a, z) in sv.iter().zip(&fd) {
                assert!((a - z).abs() <= 1e-6 * sv[0], "{}: singular values {sv:?} against {fd:?}", d.summary(true));
            }
        }
    }

    /// For a small transformer: the length of each group's gradient (the weights chart)
    /// against differences of the probe's mean loss, and the gradient at the stream
    /// after each block (the backward chart): its root mean square is made from the
    /// tape's gradients as the chart's is, and the sum of those gradients over the
    /// positions is the derivative of the loss with respect to a number added at every
    /// position (a bias that is added there, or the same change to every row of the
    /// position embedding).
    #[test]
    fn the_transformer_gradients_drawn_are_the_differences_of_the_loss() {
        let d = Design { depth: 2, width: 8, heads: 2, context: 6, act: Act::Gelu, norm: Norm::Layer, place: Place::Pre, residual: true, batch: 2, ..POST_NORM };
        let data = Data::new(d.task, d.context);
        let mut run = Run::new(d);
        let mut tape = Tape::new();
        for _ in 0..3 {
            run.step(&data, &mut tape);
        }
        let model = run.model.clone();
        let m = Measure::of(&model, &data, &mut tape, 1);
        let k = data.probe_seqs.len() as f64;
        let mean_loss = |model: &Model| {
            let mut tape = Tape::new();
            let mut total = 0.0;
            for (t, y) in &data.probe_seqs {
                let trace = model.loss_transformer(&mut tape, t, y);
                total += tape.value(trace.loss)[0];
            }
            total / k
        };
        assert!((mean_loss(&model) - m.loss).abs() < 1e-12);
        let h = 1e-6;
        let fd_of = |change: &dyn Fn(&mut Model, f64)| {
            let (mut up, mut down) = (model.clone(), model.clone());
            change(&mut up, h);
            change(&mut down, -h);
            (mean_loss(&up) - mean_loss(&down)) / (2.0 * h)
        };
        for g in &m.groups {
            let mut sq = 0.0;
            for gi in g.first..g.first + g.count {
                let kind = model.groups()[gi].kind;
                if !matches!(kind, Kind::Query | Kind::Key | Kind::Value | Kind::HeadOut | Kind::Embedding | Kind::Position | Kind::Unembedding | Kind::Weight) {
                    continue;
                }
                for e in 0..model.groups()[gi].value.len() {
                    let fd = fd_of(&|m: &mut Model, h| m.groups_mut()[gi].value[e] += h);
                    sq += fd * fd;
                }
            }
            assert!((sq.sqrt() - g.grad).abs() <= 1e-6 * g.grad + 1e-12, "group at {} part {}: drawn {}, differences {}", g.layer, g.part, g.grad, sq.sqrt());
        }
        // the stream's gradients, sequence by sequence, as the chart has them
        let w = d.width;
        let mut sums = vec![vec![0.0; w]; d.depth + 1];
        let mut sq = vec![0.0; d.depth + 1];
        let mut count = vec![0.0; d.depth + 1];
        for (tokens, targets) in &data.probe_seqs {
            let trace = model.loss_transformer(&mut tape, tokens, targets);
            tape.backward(trace.loss);
            for b in 0..=d.depth {
                let g = tape.grad(trace.get(Stage::Stream, b).unwrap());
                for (i, v) in g.iter().enumerate() {
                    sums[b][i % w] += v / k;
                    sq[b] += (v / k) * (v / k);
                }
                count[b] += g.len() as f64;
            }
        }
        for b in 0..=d.depth {
            let rms = (sq[b] / count[b]).sqrt();
            assert!((rms - m.layers[b].grad).abs() <= 1e-12 * rms, "block {b}: the chart's root mean square");
            for j in 0..w {
                let fd = if b == 0 {
                    let pos = model.index_of("position embedding").unwrap();
                    fd_of(&|m: &mut Model, h| m.groups_mut()[pos].value.iter_mut().skip(j).step_by(w).for_each(|v| *v += h))
                } else {
                    let bias = model.index_of(&format!("block {b} MLP out bias")).unwrap();
                    fd_of(&|m: &mut Model, h| m.groups_mut()[bias].value[j] += h)
                };
                assert!((fd - sums[b][j]).abs() <= 1e-6 * fd.abs().max(1e-3), "block {b} unit {j}: tape {}, differences {fd}", sums[b][j]);
            }
        }
    }

    // ----- degenerate settings ---------------------------------------------------------

    /// What is drawn is finite, whatever the numbers measured.
    fn drawn_finite(lab: &mut Lab) {
        lab.follow(1e6, true);
        for shown in [&lab.look.a, &lab.look.b] {
            for g in [&shown.rms, &shown.mean, &shown.grad, &shown.shut, &shown.w_grad, &shown.w_step, &shown.sv, &shown.halves, &shown.values, &shown.grads] {
                assert!(g.shown.iter().all(|v| v.is_finite()), "{}: a glide holds {:?}", lab.design.summary(false), g.shown);
            }
        }
        for g in &lab.look.ranges {
            assert!(g.shown.iter().all(|v| v.is_finite()), "a range {:?}", g.shown);
        }
        // and a glide that has arrived rests
        assert!(!lab.look.moving || {
            lab.follow(1e6, false);
            !lab.look.moving
        });
    }

    /// Every setting at the ends of its control, alone and together: a width, depth and
    /// batch of 1, a gain of 0, a rate of 0, the largest sizes, the largest rate; every
    /// normalisation in both places, with and without skips. Nothing panics, every
    /// number drawn is finite, and a network that blows up is said to have.
    #[test]
    fn the_ends_of_the_controls_are_fine() {
        let ends = [
            Design { depth: 1, width: 1, batch: 1, gain: 0.0, lr: 0.0, ..BASE },
            Design { depth: 1, width: 1, batch: 1, gain: 3.0, lr: 1.0, clip: 0.01, optim: Optim::Sgd, ..BASE },
            Design { depth: 2, width: 3, batch: 1, gain: 3.0, lr: 1.0, heads: 4, context: 2, ..BASE },
            Design { depth: 99, width: 99, batch: 99, heads: 99, context: 99, gain: f64::NAN, lr: f64::INFINITY, clip: -1.0, scale: f64::NAN, ..BASE },
        ];
        let mut lab = Lab::create();
        for task in Task::ALL {
            for (e, end) in ends.iter().enumerate() {
                for act in [Act::Relu, Act::Gelu] {
                    for norm in Norm::ALL {
                        for place in Place::ALL {
                            for residual in [false, true] {
                                // (the largest sizes once for each norm: they are slow)
                                if e == 3 && (act != Act::Gelu || place == Place::Post || residual) {
                                    continue;
                                }
                                lab.keep(false);
                                lab.design = Design { task, act, norm, place, residual, ..*end };
                                lab.apply();
                                for _ in 0..2 {
                                    lab.one_step();
                                }
                                lab.layer = 99;
                                lab.apply();
                                lab.measure_all();
                                drawn_finite(&mut lab);
                                lab.keep(true);
                                lab.one_step();
                                lab.measure_all();
                                drawn_finite(&mut lab);
                            }
                        }
                    }
                }
            }
        }
        // a rate that blows a network up: it stops, and says when
        let mut lab = Lab::create();
        lab.design = Design { depth: 3, optim: Optim::Sgd, lr: 1.0, gain: 3.0, act: Act::Identity, ..BASE };
        lab.apply();
        for _ in 0..200 {
            lab.one_step();
        }
        assert!(lab.a.run.diverged.is_some(), "{:?}", lab.a.run.last);
        lab.measure_all();
        drawn_finite(&mut lab);
    }

    /// In every window, from a phone's to one far too small, the cells lie in the view
    /// and do not overlap, and on a phone the training chart is the last.
    #[test]
    fn the_layout_fits_every_window() {
        for task in Task::ALL {
            let charts = charts_for(task);
            for (w, h) in [(1600.0, 1000.0), (1270.0, 940.0), (770.0, 650.0), (570.0, 650.0), (390.0, 800.0), (390.0, 600.0), (240.0, 180.0), (1.0, 1.0), (3000.0, 300.0)] {
                let lay = Layout::of(Vec2::new(w, h), 40.0, &charts);
                let shown: Vec<Chart> = lay.cells.iter().filter_map(|c| c.1).chain(lay.tabbed.iter().copied()).collect();
                assert_eq!(shown.len(), charts.len(), "{w} x {h}: every chart has a place");
                if lay.tiny() {
                    continue;
                }
                for (i, (a, _)) in lay.cells.iter().enumerate() {
                    assert!(a.0.x >= 0.0 && a.1.x <= w + 0.5, "{w} x {h}: cell {i} across");
                    for (b, _) in &lay.cells[i + 1..] {
                        let apart = a.1.x <= b.0.x + 0.5 || b.1.x <= a.0.x + 0.5 || a.1.y <= b.0.y + 0.5 || b.1.y <= a.0.y + 0.5;
                        assert!(apart, "{w} x {h}: cells overlap");
                    }
                }
                if lay.narrow && lay.cells.len() > 1 {
                    assert_eq!(lay.cells.last().unwrap().1, Some(Chart::Training), "{w} x {h}");
                }
            }
        }
    }

    /// The other two changes the first experiment's caption speaks of, each on its own
    /// in place of the skips, trained 1000 steps. (Slow: run with `--ignored`.)
    #[test]
    #[ignore]
    fn a_deep_network_mended_otherwise() {
        for (change, lo, hi) in [(Design { gain: 1.0, ..DEEP_TANH }, 0.002, 0.004), (Design { norm: Norm::Layer, ..DEEP_TANH }, 1.0, 1.2), (Design { norm: Norm::Layer, place: Place::Post, ..DEEP_TANH }, 1.0, 1.2)] {
            let mut lab = start(0);
            lab.design = change;
            lab.apply();
            for _ in 0..1000 {
                lab.one_step();
            }
            within(last(&lab.a).0, lo, hi, &change.summary(false));
        }
    }
}

//! Backpropagation, a layer at a time: the sweep of `backprop1` for a network of real
//! shape and a batch of examples, as it is computed, in matrices. Every matrix has one
//! row per example; a weight matrix has one row per unit of its layer, so a layer is
//! Z = A Wᵀ + b. The loss hands back D = (P − Y)/n; each layer passes D W back through
//! the same weights and multiplies it by its units' slopes; and the slope of every
//! weight matrix is Dᵀ A, a sum of one outer product per example.

use super::*;
use crate::math::arch::{self, Init, Loss, MlpSpec, Model, Norm, Place, Stage};
use crate::math::autograd::{Act, Tape, Var};
use crate::math::linalg;
use crate::scenes::stats::plot::{self, Rect};

pub const DEF: SceneDef = SceneDef {
    id: "backpropmat",
    title: "Backpropagation, a layer at a time",
    group: GROUP,
    blurb: "The same sweep in matrices: each layer hands its error back through the transpose of its weights, and a weight's gradient is a sum of outer products, one per example.",
    make: |_| Box::new(BackpropMat::new()),
};

// ----- the network and its numbers ------------------------------------------------------

/// Numbers per example: points in the plane.
const INPUTS: usize = 2;
/// Classes: the read-out's units.
const CLASSES: usize = 3;
/// Hidden layers. Layer 0 is the data, layers 1 and 2 are hidden, layer 3 the read-out.
const DEPTH: usize = 2;
/// The read-out's layer number.
const TOP: usize = DEPTH + 1;
/// The most examples in a batch, and the most units in a hidden layer.
const MOST_BATCH: usize = 8;
const MOST_WIDTH: usize = 6;
/// The seeds of the starting weights and of the data. (Chosen so that with ReLU every
/// example has a unit open in each hidden layer, at every width: a network that is
/// dead for some example from the start would show a different lesson.)
const WEIGHT_SEED: u64 = 144;
const DATA_SEED: u64 = 3;
/// A singular value counts as 0 when it is below this share of the largest.
const RANK_TOL: f64 = 1e-10;

/// What the visitor sets.
#[derive(Clone, Copy, Debug, PartialEq)]
struct Setup {
    batch: usize,
    width: usize,
    act: Act,
    /// Whether one unit of layer 1 (`shut_unit`) is pushed below 0 for every example.
    shut: bool,
}

impl Default for Setup {
    fn default() -> Self {
        Setup { batch: 4, width: 4, act: Act::Relu, shut: false }
    }
}

impl Setup {
    fn clamped(self) -> Setup {
        Setup { batch: self.batch.clamp(1, MOST_BATCH), width: self.width.clamp(1, MOST_WIDTH), ..self }
    }
}

/// The unit of layer 1 that `Setup::shut` pushes below 0 (from 0): the second, or the
/// only one.
fn shut_unit(width: usize) -> usize {
    1.min(width.max(1) - 1)
}

/// A matrix, row-major: entry (i, j) of a `rows` by `cols` one is `v[i * cols + j]`.
#[derive(Clone, Debug, Default, PartialEq)]
struct Mat {
    rows: usize,
    cols: usize,
    v: Vec<f64>,
}

impl Mat {
    /// (Missing numbers are NaN: a mistake shows instead of ending the program.)
    fn new(rows: usize, cols: usize, v: &[f64]) -> Mat {
        let mut v = v.to_vec();
        v.resize(rows * cols, f64::NAN);
        Mat { rows, cols, v }
    }

    fn zeros(rows: usize, cols: usize) -> Mat {
        Mat { rows, cols, v: vec![0.0; rows * cols] }
    }

    /// Entry (i, j); NaN outside.
    fn at(&self, i: usize, j: usize) -> f64 {
        if i < self.rows && j < self.cols {
            self.v[i * self.cols + j]
        } else {
            f64::NAN
        }
    }

    /// The largest entry in size (0 for none).
    fn largest(&self) -> f64 {
        self.v.iter().filter(|x| x.is_finite()).fold(0.0f64, |m, x| m.max(x.abs()))
    }
}

/// One forward and backward pass of the scene's network on its batch. Every number is
/// the tape's, but for the probabilities P (the softmax of the tape's Z₃) and the gates
/// (the activation's slope at the tape's Z), worked out by the same functions.
#[derive(Clone, Debug)]
struct Net {
    setup: Setup,
    /// The units of layers 0 to 3: 2, the width twice, 3.
    units: [usize; TOP + 1],
    model: Model,
    /// The data, one row per example, and each example's class.
    x: Vec<f64>,
    targets: Vec<f64>,
    /// For each layer l from 1 to 3 (index 0 unused): the weights W_l (units of l by
    /// units of l − 1) and the bias b_l.
    w: [Mat; TOP + 1],
    /// Z_l = A_(l−1) W_lᵀ + b_l, for l from 1 to 3 (Z_3: the logits).
    z: [Mat; TOP + 1],
    /// A_0 = X, A_1, A_2 the hidden layers' outputs, and (at 3) P, the softmax of Z_3.
    a: [Mat; TOP + 1],
    /// φ'(Z_l), the gates, for l = 1 and 2.
    gate: [Mat; TOP + 1],
    /// The targets as rows of 0s and a 1.
    y: Mat,
    /// The slope of the loss with respect to A_l (l = 1, 2): D_(l+1) W_(l+1).
    back: [Mat; TOP + 1],
    /// D_l, the slope of the loss with respect to Z_l (l = 1 to 3).
    d: [Mat; TOP + 1],
    /// The slopes of the loss with respect to W_l and b_l (b as a row).
    dw: [Mat; TOP + 1],
    db: [Mat; TOP + 1],
    loss: f64,
    /// The singular values of each ∂L/∂W_l, largest first.
    sv: [Vec<f64>; TOP + 1],
}

/// The names of layer l's weights and bias among the model's groups.
fn names(l: usize) -> (String, String) {
    if l == TOP {
        ("read-out weights".into(), "read-out bias".into())
    } else {
        (format!("layer {l} weights"), format!("layer {l} bias"))
    }
}

/// The batch's points and classes: the first `n` of eight points on three spiral arms
/// (classes 0, 1, 2, 0, 1, ... in turn).
fn data(n: usize) -> (Vec<f64>, Vec<f64>) {
    let (x, y) = arch::spirals(MOST_BATCH, CLASSES, &mut Rng::new(DATA_SEED));
    let n = n.clamp(1, MOST_BATCH);
    (x[..n * INPUTS].to_vec(), y[..n].to_vec())
}

/// The model the scene starts from for a setup: He's start, biases 0, and (with `shut`)
/// one unit of layer 1 given a bias that keeps its Z at −1 or below for every one of the
/// eight points.
fn model_for(s: Setup) -> Model {
    let spec = MlpSpec {
        inputs: INPUTS,
        width: s.width,
        depth: DEPTH,
        outputs: CLASSES,
        act: s.act,
        norm: Norm::None,
        place: Place::Pre,
        residual: false,
        residual_scale: 1.0,
        init: Init::He,
        gain: 1.0,
        bias: true,
    };
    let mut model = Model::mlp(&spec, &mut Rng::new(WEIGHT_SEED));
    if s.shut {
        let u = shut_unit(s.width);
        let (all, _) = data(MOST_BATCH);
        let w = model.group(&names(1).0).map(|g| g.value.clone()).unwrap_or_default();
        let weight = |k: usize| w.get(u * INPUTS + k).copied().unwrap_or(0.0);
        let most = all.chunks_exact(INPUTS).map(|x| (0..INPUTS).map(|k| weight(k) * x[k]).sum::<f64>()).fold(f64::MIN, f64::max);
        if let Some(b) = model.group_mut(&names(1).1).and_then(|g| g.value.get_mut(u)) {
            *b = -1.0 - most;
        }
    }
    model
}

/// The softmax of each row.
fn softmax_rows(z: &Mat) -> Mat {
    let mut p = Mat::zeros(z.rows, z.cols);
    for i in 0..z.rows {
        let row = &z.v[i * z.cols..(i + 1) * z.cols];
        let top = row.iter().fold(f64::NEG_INFINITY, |m, &v| m.max(v));
        let sum: f64 = row.iter().map(|v| (v - top).exp()).sum();
        for j in 0..z.cols {
            p.v[i * z.cols + j] = (row[j] - top).exp() / sum;
        }
    }
    p
}

impl Net {
    fn new(setup: Setup) -> Net {
        let s = setup.clamped();
        let model = model_for(s);
        let (x, targets) = data(s.batch);
        let n = targets.len();
        let units = [INPUTS, s.width, s.width, CLASSES];
        let mut tape = Tape::new();
        let tr = model.loss_mlp(&mut tape, &x, &targets, Loss::CrossEntropy);
        tape.backward(tr.loss);
        let take = |v: Option<Var>, rows: usize, cols: usize, grad: bool| match v {
            Some(v) => Mat::new(rows, cols, if grad { tape.grad(v) } else { tape.value(v) }),
            None => Mat::new(rows, cols, &[]),
        };
        let mut net = Net {
            setup: s,
            units,
            model: model.clone(),
            x: x.clone(),
            targets: targets.clone(),
            w: Default::default(),
            z: Default::default(),
            a: Default::default(),
            gate: Default::default(),
            y: Mat::zeros(n, CLASSES),
            back: Default::default(),
            d: Default::default(),
            dw: Default::default(),
            db: Default::default(),
            loss: tape.value(tr.loss).first().copied().unwrap_or(f64::NAN),
            sv: Default::default(),
        };
        for (i, &c) in targets.iter().enumerate() {
            if c >= 0.0 && (c as usize) < CLASSES {
                net.y.v[i * CLASSES + c as usize] = 1.0;
            }
        }
        net.a[0] = Mat::new(n, INPUTS, &x);
        for l in 1..=TOP {
            let (wn, bn) = names(l);
            let (rows, cols) = (units[l], units[l - 1]);
            let wv = model.index_of(&wn).and_then(|i| tr.params.get(i).copied());
            let bv = model.index_of(&bn).and_then(|i| tr.params.get(i).copied());
            net.w[l] = take(wv, rows, cols, false);
            net.dw[l] = take(wv, rows, cols, true);
            net.db[l] = take(bv, 1, rows, true);
            if l < TOP {
                net.z[l] = take(tr.get(Stage::PreAct, l), n, rows, false);
                net.d[l] = take(tr.get(Stage::PreAct, l), n, rows, true);
                net.a[l] = take(tr.get(Stage::PostAct, l), n, rows, false);
                net.back[l] = take(tr.get(Stage::PostAct, l), n, rows, true);
                let z = &net.z[l];
                net.gate[l] = Mat { rows: z.rows, cols: z.cols, v: z.v.iter().map(|&v| s.act.df(v)).collect() };
            } else {
                net.z[l] = take(Some(tr.output), n, rows, false);
                net.d[l] = take(Some(tr.output), n, rows, true);
                net.a[l] = softmax_rows(&net.z[l]);
            }
            net.sv[l] = linalg::svd(&net.dw[l].v, rows.max(1), cols.max(1)).1;
        }
        net
    }

    fn n(&self) -> usize {
        self.a[0].rows
    }

    /// How many entries of Z₁ and Z₂ are exactly 0 where the activation has a corner
    /// there (ReLU, leaky ReLU): at a corner the slope is not defined, and the tape
    /// takes the slope of the piece below it, as a difference of the loss need not.
    /// (It happens when a unit's inputs are all 0 and its bias is 0: every unit of
    /// layer 1 shut for an example leaves that example's Z₂ at 0.)
    fn corners(&self) -> usize {
        if !matches!(self.setup.act, Act::Relu | Act::LeakyRelu) {
            return 0;
        }
        (1..=DEPTH).map(|l| self.z[l].v.iter().filter(|z| **z == 0.0).count()).sum()
    }

    /// The number of singular values of ∂L/∂W_l that are not 0 (above `RANK_TOL` of the
    /// largest).
    fn rank(&self, l: usize) -> usize {
        let sv = &self.sv[l.min(TOP)];
        let top = sv.first().copied().unwrap_or(0.0);
        if top.is_nan() || top <= 0.0 {
            return 0;
        }
        sv.iter().filter(|&&s| s > RANK_TOL * top).count()
    }

    /// What a weight slope's rank can be at most: its smaller side, or the batch.
    fn rank_bound(&self, l: usize) -> usize {
        let l = l.clamp(1, TOP);
        self.units[l].min(self.units[l - 1]).min(self.n())
    }

    /// Multiply-adds of the matrix products: the forward pass (one product a layer)
    /// and the backward pass (two a layer, D W and Dᵀ A, but no D₁ W₁: the slope with
    /// respect to the data is not wanted).
    fn work(&self) -> (usize, usize) {
        let n = self.n();
        let layer = |l: usize| n * self.units[l] * self.units[l - 1];
        let forward: usize = (1..=TOP).map(layer).sum();
        (forward, 2 * forward - layer(1))
    }

    /// The loss of the same batch with Z_l replaced by `z` and everything after it
    /// worked out again by hand (in f64, from the network's own weights): what the
    /// slopes D_l are checked against.
    fn loss_from_z(&self, l: usize, z: &Mat) -> f64 {
        let l = l.clamp(1, TOP);
        let mut z = z.clone();
        for next in l + 1..=TOP {
            let a: Vec<f64> = z.v.iter().map(|&v| self.setup.act.f(v)).collect();
            let (w, b) = (&self.w[next], self.model.group(&names(next).1).map(|g| g.value.clone()).unwrap_or_default());
            let mut out = Mat::zeros(z.rows, w.rows);
            for i in 0..z.rows {
                for j in 0..w.rows {
                    out.v[i * w.rows + j] = (0..w.cols).map(|k| a[i * z.cols + k] * w.at(j, k)).sum::<f64>() + b.get(j).copied().unwrap_or(0.0);
                }
            }
            z = out;
        }
        let p = softmax_rows(&z);
        let n = z.rows.max(1) as f64;
        (0..z.rows).map(|i| -p.at(i, self.targets[i] as usize).ln()).sum::<f64>() / n
    }

    /// The model's loss with one entry of a parameter group moved by `by`.
    fn loss_moved(&self, group: &str, entry: usize, by: f64) -> f64 {
        let mut m = self.model.clone();
        if let Some(v) = m.group_mut(group).and_then(|g| g.value.get_mut(entry)) {
            *v += by;
        }
        let mut tape = Tape::new();
        let tr = m.loss_mlp(&mut tape, &self.x, &self.targets, Loss::CrossEntropy);
        tape.value(tr.loss).first().copied().unwrap_or(f64::NAN)
    }

    /// The slope of the loss with respect to the picked number, by a central difference
    /// of the loss (h a millionth of the number's size, or of 1).
    fn by_difference(&self, pick: Pick) -> f64 {
        let step = |x: f64| 1e-6 * x.abs().max(1.0);
        match pick {
            Pick::Weight(l, j, k) => {
                let h = step(self.w[l].at(j, k));
                let name = names(l).0;
                (self.loss_moved(&name, j * self.units[l - 1] + k, h) - self.loss_moved(&name, j * self.units[l - 1] + k, -h)) / (2.0 * h)
            }
            Pick::Bias(l, j) => {
                let name = names(l).1;
                let h = 1e-6;
                (self.loss_moved(&name, j, h) - self.loss_moved(&name, j, -h)) / (2.0 * h)
            }
            Pick::Back(l, i, k) => {
                let z = &self.z[l];
                let h = step(z.at(i, k));
                let moved = |by: f64| {
                    let mut z = z.clone();
                    if let Some(v) = z.v.get_mut(i * z.cols + k) {
                        *v += by;
                    }
                    self.loss_from_z(l, &z)
                };
                (moved(h) - moved(-h)) / (2.0 * h)
            }
        }
    }

    /// The outer product of example i's slopes at layer l with its inputs to it:
    /// d_iᵀ a_i, units of l by units of l − 1. (The slope ∂L/∂W_l is their sum over i.)
    fn outer(&self, l: usize, i: usize) -> Mat {
        let (d, a) = (&self.d[l], &self.a[l - 1]);
        let mut m = Mat::zeros(d.cols, a.cols);
        for j in 0..d.cols {
            for k in 0..a.cols {
                m.v[j * a.cols + k] = d.at(i, j) * a.at(i, k);
            }
        }
        m
    }
}

// ----- the picture's parts ------------------------------------------------------------

/// The blocks of the picture: for each layer its output (`Out`), what is beside it in
/// the middle row (`Mid`: the gates of a hidden layer, the targets for the read-out),
/// its D (`Back`) and the column sums of D, the bias's slope (`Bias`); between layers
/// the weights (`W`) and their slope (`Dw`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Slot {
    Out(usize),
    Mid(usize),
    Back(usize),
    Bias(usize),
    W(usize),
    Dw(usize),
}

const SLOTS: [Slot; 19] = [
    Slot::Out(0),
    Slot::Out(1),
    Slot::Out(2),
    Slot::Out(3),
    Slot::Mid(1),
    Slot::Mid(2),
    Slot::Mid(3),
    Slot::Back(1),
    Slot::Back(2),
    Slot::Back(3),
    Slot::Bias(1),
    Slot::Bias(2),
    Slot::Bias(3),
    Slot::W(1),
    Slot::W(2),
    Slot::W(3),
    Slot::Dw(1),
    Slot::Dw(2),
    Slot::Dw(3),
];

fn slot_index(s: Slot) -> usize {
    SLOTS.iter().position(|&t| t == s).unwrap_or(0)
}

/// What the picked entry is: an entry (j, k) of ∂L/∂W_l, an entry j of ∂L/∂b_l, or an
/// entry (i, k) of D_l (example i, unit k). Counted from 0.
#[derive(Clone, Copy, Debug, PartialEq)]
enum Pick {
    Weight(usize, usize, usize),
    Bias(usize, usize),
    Back(usize, usize, usize),
}

impl Pick {
    fn layer(self) -> usize {
        match self {
            Pick::Weight(l, ..) | Pick::Bias(l, _) | Pick::Back(l, ..) => l,
        }
    }

    /// The same pick, kept inside a network's shapes.
    fn within(self, net: &Net) -> Pick {
        let u = net.units;
        let last = |n: usize, i: usize| i.min(n.max(1) - 1);
        match self {
            Pick::Weight(l, j, k) => {
                let l = l.clamp(1, TOP);
                Pick::Weight(l, last(u[l], j), last(u[l - 1], k))
            }
            Pick::Bias(l, j) => {
                let l = l.clamp(1, TOP);
                Pick::Bias(l, last(u[l], j))
            }
            Pick::Back(l, i, k) => {
                let l = l.clamp(1, TOP);
                Pick::Back(l, last(net.n(), i), last(u[l], k))
            }
        }
    }

    /// The step of the film after which its number is there to be shown.
    fn ready_at(self) -> usize {
        match self {
            Pick::Weight(l, ..) | Pick::Bias(l, _) => made_at(Slot::Dw(l)),
            Pick::Back(l, ..) if l >= TOP => made_at(Slot::Back(TOP)),
            Pick::Back(l, ..) => made_at(Slot::Back(l)) - 1,
        }
    }

    /// The number picked.
    fn value(self, net: &Net) -> f64 {
        match self.within(net) {
            Pick::Weight(l, j, k) => net.dw[l].at(j, k),
            Pick::Bias(l, j) => net.db[l].at(0, j),
            Pick::Back(l, i, k) => net.d[l].at(i, k),
        }
    }
}

// ----- the film -----------------------------------------------------------------------

/// Steps of the film: three forward, eight back.
const STEPS: usize = 11;

/// The step after which a block is filled (0: there from the start). A hidden layer's
/// D is filled a step earlier with D_(l+1) W_(l+1), and becomes D_l at this step.
fn made_at(s: Slot) -> usize {
    match s {
        Slot::Out(0) | Slot::W(_) | Slot::Mid(3) => 0,
        Slot::Out(l) | Slot::Mid(l) => l.min(TOP),
        Slot::Back(3) => 4,
        Slot::Dw(3) | Slot::Bias(3) => 5,
        Slot::Back(2) => 7,
        Slot::Dw(2) | Slot::Bias(2) => 8,
        Slot::Back(_) => 10,
        Slot::Dw(_) | Slot::Bias(_) => 11,
    }
}

/// What step `s` works out (its main block) and the two blocks it uses.
fn step_blocks(s: usize) -> (Slot, [Slot; 2]) {
    use Slot::*;
    match s {
        1 => (Out(1), [Out(0), W(1)]),
        2 => (Out(2), [Out(1), W(2)]),
        3 => (Out(3), [Out(2), W(3)]),
        4 => (Back(3), [Out(3), Mid(3)]),
        5 => (Dw(3), [Back(3), Out(2)]),
        6 => (Back(2), [Back(3), W(3)]),
        7 => (Back(2), [Mid(2), Mid(2)]),
        8 => (Dw(2), [Back(2), Out(1)]),
        9 => (Back(1), [Back(2), W(2)]),
        10 => (Back(1), [Mid(1), Mid(1)]),
        11 => (Dw(1), [Back(1), Out(0)]),
        _ => (Out(0), [Out(0), Out(0)]),
    }
}

/// The numbers a block shows at film position `s` (none while it is not yet worked out).
fn block_values(net: &Net, slot: Slot, s: usize) -> Option<&Mat> {
    let filled = s >= made_at(slot);
    match slot {
        Slot::Back(l) if l <= DEPTH && !filled && s + 1 == made_at(slot) => Some(&net.back[l]),
        _ if !filled => None,
        Slot::Out(l) => Some(&net.a[l]),
        Slot::Mid(l) if l == TOP => Some(&net.y),
        Slot::Mid(l) => Some(&net.gate[l]),
        Slot::Back(l) => Some(&net.d[l]),
        Slot::Bias(l) => Some(&net.db[l]),
        Slot::W(l) => Some(&net.w[l]),
        Slot::Dw(l) => Some(&net.dw[l]),
    }
}

/// The rows and columns of a block.
fn shape(net: &Net, slot: Slot) -> (usize, usize) {
    let (n, u) = (net.n(), net.units);
    match slot {
        Slot::Out(l) | Slot::Mid(l) | Slot::Back(l) => (n, u[l.min(TOP)]),
        Slot::Bias(l) => (1, u[l.min(TOP)]),
        Slot::W(l) | Slot::Dw(l) => (u[l.clamp(1, TOP)], u[l.clamp(1, TOP) - 1]),
    }
}

/// What a block's squares are measured against: its largest entry (for a hidden
/// layer's D, the larger of D and D_(l+1) W_(l+1), which it is made from); 1 for the
/// probabilities, the targets, and gates no larger than 1.
fn scale(net: &Net, slot: Slot) -> f64 {
    let m = match slot {
        Slot::Out(TOP) | Slot::Mid(TOP) => 1.0,
        Slot::Mid(l) => net.gate[l].largest().max(1.0),
        Slot::Back(l) if l <= DEPTH => net.d[l].largest().max(net.back[l].largest()),
        Slot::Out(l) => net.a[l].largest(),
        Slot::Back(l) => net.d[l].largest(),
        Slot::Bias(l) => net.db[l].largest(),
        Slot::W(l) => net.w[l].largest(),
        Slot::Dw(l) => net.dw[l].largest(),
    };
    if m > 0.0 && m.is_finite() {
        m
    } else {
        1.0
    }
}

/// The colour of what a block holds.
fn color_of(slot: Slot) -> Color {
    match slot {
        Slot::Out(_) => FORWARD,
        Slot::Mid(TOP) => LOSS,
        Slot::Mid(_) => GATE,
        Slot::Back(_) => BACKWARD,
        Slot::Bias(_) | Slot::W(_) | Slot::Dw(_) => WEIGHT,
    }
}

/// The gates: each unit's slope at what it received on the way forward.
const GATE: Color = Color::hex(0xC3CADB);
/// The step being worked out, and the entry picked.
const CURSOR: Color = pal::WHITE;

/// A block's name, as markup (`short`: for a narrow block on a narrow canvas).
fn title(slot: Slot, s: usize, short: bool) -> String {
    match slot {
        Slot::Out(0) if short => "$X$".into(),
        Slot::Out(0) => "$X = A_0$".into(),
        Slot::Out(l) if l == TOP => "$P$".into(),
        Slot::Out(l) => format!("$A_{l}$"),
        Slot::Mid(l) if l == TOP => "$Y$".into(),
        Slot::Mid(l) => format!(r"$\varphi'(Z_{l})$"),
        Slot::Back(l) if l <= DEPTH && s + 1 == made_at(slot) => format!("$D_{} W_{}$", l + 1, l + 1),
        Slot::Back(l) => format!("$D_{l}$"),
        Slot::Bias(l) => format!(r"$\partial L/\partial b_{l}$"),
        Slot::W(l) => format!("$W_{l}$"),
        Slot::Dw(l) if short => format!(r"$\partial W_{l}$"),
        Slot::Dw(l) => format!(r"$\partial L/\partial W_{l}$"),
    }
}

/// "4 × 3", as markup.
fn dims(rows: usize, cols: usize) -> String {
    format!(r"{rows} \times {cols}")
}

/// A number as a table shows it: 4 decimals, an exact 0 as "0", and one that would read
/// 0.0000 in two figures and a power of ten. Markup.
fn four(v: f64) -> String {
    if !v.is_finite() {
        "?".into()
    } else if v == 0.0 {
        "0".into()
    } else if v.abs() < 0.00005 {
        let e = v.abs().log10().floor() as i32;
        let m = v.abs() / 10f64.powi(e);
        let (m, e) = if m >= 9.95 { (1.0, e + 1) } else { (m, e) };
        format!(r"{}{} \times 10^{{{e}}}", if v < 0.0 { "-" } else { "" }, num(m, 1))
    } else {
        num(v, 4)
    }
}

/// The step's line of arithmetic and the line of shapes under it, as markup.
fn step_lines(net: &Net, s: usize, narrow: bool) -> (String, String) {
    let (n, u) = (net.n(), net.units);
    let act = net.setup.act;
    let relu = act == Act::Relu;
    let name = act.name();
    match s {
        0 => (
            format!("the data $A_0 = X$: {n} {} (rows) of 2 numbers; the weights $W_l$ are drawn at random", if n == 1 { "example" } else { "examples" }),
            "next ▶ runs the forward pass, a layer at a time".into(),
        ),
        1 | 2 => {
            let l = s;
            (
                format!(r"forward, layer {l}: $Z_{l} = A_{} W_{l}^\top + b_{l}$, then $A_{l} = \varphi(Z_{l})$ entry by entry ($\varphi$ = {name})", l - 1),
                format!(
                    r"$({})({}) \to {}$; the gates $\varphi'(Z_{l})$ are kept for the way back",
                    dims(n, u[l - 1]),
                    dims(u[l - 1], u[l]),
                    dims(n, u[l])
                ),
            )
        }
        3 => (
            format!(r"the read-out: $Z_3 = A_2 W_3^\top + b_3$, $P = \operatorname{{softmax}}(Z_3)$ row by row; the loss $L = {}$", num(net.loss, 3)),
            format!(r"$({})({}) \to {}$; $L$ is the mean over the rows of $-\ln P_{{iy_i}}$, $y_i$ the row's class", dims(n, u[2]), dims(u[2], 3), dims(n, 3)),
        ),
        4 => (
            r"backward: $D_3 = (P - Y)/n$, the slope of $L$ with respect to $Z_3$".into(),
            format!(r"${}$, the shape of $Z_3$; each row adds up to 0", dims(n, 3)),
        ),
        5 | 8 | 11 => {
            let l = TOP - (s - 5) / 3;
            let per = if n == 1 { "one outer product, for the one example".to_string() } else { format!("a sum of {n} outer products, one per example") };
            (
                format!(r"$\partial L/\partial W_{l} = D_{l}^\top A_{}$; and $\partial L/\partial b_{l}$, the column sums of $D_{l}$", l - 1),
                format!(r"$({})({}) \to {}$, the shape of $W_{l}$: {per}", dims(u[l], n), dims(n, u[l - 1]), dims(u[l], u[l - 1])),
            )
        }
        6 | 9 => {
            let l = if s == 6 { 2 } else { 1 };
            (
                format!(r"$D_{} W_{}$: the slope with respect to $A_{l}$, back through the same weights", l + 1, l + 1),
                if narrow {
                    format!(r"$({})({}) \to {}$", dims(n, u[l + 1]), dims(u[l + 1], u[l]), dims(n, u[l]))
                } else {
                    format!(
                        r"$({})({}) \to {}$; forward it was $A_{l} W_{}^\top$: back, the transpose of what forward used",
                        dims(n, u[l + 1]),
                        dims(u[l + 1], u[l]),
                        dims(n, u[l]),
                        l + 1
                    )
                },
            )
        }
        7 | 10 => {
            let l = if s == 7 { 2 } else { 1 };
            (
                format!(r"$D_{l} = (D_{} W_{}) \odot \varphi'(Z_{l})$: each entry times its unit's gate", l + 1, l + 1),
                if relu {
                    format!(r"${}$; a ReLU gate is 1 or 0: where it is 0 (the unit was shut), nothing passes", dims(n, u[l]))
                } else {
                    format!(r"${}$; each gate is {name}'s slope at what the unit received", dims(n, u[l]))
                },
            )
        }
        _ => (String::new(), String::new()),
    }
}

// ----- a glide over many numbers ------------------------------------------------------

/// Numbers drawn following the numbers worked out, through two eases one after the
/// other (`tau` seconds each), so that a picture whose numbers change in a step moves
/// evenly and comes exactly to rest (as `plot::Glide`, with an ease of one's choosing).
#[derive(Clone, Debug, Default)]
struct Follow {
    between: Vec<f32>,
    shown: Vec<f32>,
}

impl Follow {
    /// Bring the numbers `dt` seconds nearer to `to` (a list of another length starts
    /// where it is going). Says whether anything is still on its way.
    fn follow(&mut self, to: &[f32], dt: f32, tau: f32) -> bool {
        if self.shown.len() != to.len() {
            self.snap(to);
            return false;
        }
        if self.shown == to {
            return false;
        }
        let k = 1.0 - (-dt.max(0.0) / tau.max(1e-3)).exp();
        let (mut far, mut moved) = (false, false);
        for ((b, s), t) in self.between.iter_mut().zip(self.shown.iter_mut()).zip(to) {
            let (b0, s0) = (*b, *s);
            *b += (t - *b) * k;
            *s += (*b - *s) * k;
            if !(b.is_finite() && s.is_finite()) {
                (*b, *s) = (*t, *t);
            }
            moved |= *b != b0 || *s != s0;
            let near = 2e-4 * t.abs().max(1.0);
            far |= !((t - *s).abs() <= near && (t - *b).abs() <= near);
        }
        if !far || !moved {
            self.snap(to);
            return false;
        }
        true
    }

    fn snap(&mut self, to: &[f32]) {
        self.shown = to.to_vec();
        self.between = to.to_vec();
    }

    fn get(&self, i: usize) -> f32 {
        self.shown.get(i).copied().unwrap_or(0.0)
    }
}

// ----- where things go ----------------------------------------------------------------

/// Where everything is on the screen (logical pixels from the top-left corner).
#[derive(Clone, Debug, Default)]
struct Layout {
    narrow: bool,
    /// Sizes of the titles, of the small print, and of the step's lines.
    text: f32,
    small: f32,
    /// The film's buttons: start, back, next, play, end; and where its count goes.
    buttons: Vec<Rect>,
    status: Option<Vec2>,
    /// Where the value under the pointer is written, and where the key to the squares
    /// goes (its right end, or its left end under the blocks).
    hover: Option<Vec2>,
    legend: Option<(Vec2, bool)>,
    /// The side of a cell, and each block's top-left corner (in `SLOTS` order).
    cell: f32,
    blocks: [Vec2; 19],
    /// The first y below the blocks and their labels, and (on a phone, where the
    /// singular values go under it) the room the picked entry's sum takes.
    below: f32,
    table_h: f32,
    /// The step's two lines: their top-left corner and the room across.
    lines: Option<(Vec2, f32)>,
    /// The picked entry's sum, the outer products, and the singular values.
    table: Option<Rect>,
    terms: Option<Rect>,
    chart: Option<Rect>,
    view: Vec2,
}

/// The words on the film's buttons, the longest each can say.
const BUTTON_WORDS: [&str; 5] = ["↺ start", "◀ back", "next ▶", "❚❚ pause", "end ⇥"];
const BUTTON_SHORT: [&str; 5] = ["↺", "◀", "▶", "❚❚", "⇥"];

impl Layout {
    /// The blocks laid out in a row, the layers from left to right (the data, W₁, A₁,
    /// W₂, A₂, W₃, P), each layer's column three blocks deep: its output, its gates (or
    /// the targets), its D with the bias's slope under it; a weight's slope under the
    /// weight. Then the parts below, from `below` down.
    fn of(f: &Frame, view: Vec2, inset: f32, net: &Net, pick: Pick) -> Layout {
        let view = view.max(Vec2::ONE);
        let narrow = view.x < 620.0;
        let (text, small) = if narrow { (11.0, 10.0) } else { (13.0, 11.5) };
        let side = if narrow { 8.0 } else { 16.0 };
        let top = inset.max(0.0) + if narrow { 6.0 } else { 10.0 };
        let mut lay = Layout { narrow, text, small, view, ..Default::default() };
        if view.x < 200.0 || view.y < 160.0 {
            lay.below = view.y;
            return lay;
        }
        // the buttons
        let h = if narrow { 26.0 } else { 28.0 };
        lay.buttons = if narrow {
            plot::even_row(5, side, (view.x - side).min(side + 5.0 * 48.0), top, h, 5.0)
        } else {
            plot::button_row(f, &BUTTON_WORDS, side, top, h, text - 0.5, 8.0)
        };
        let after = lay.buttons.last().map_or(side, |r| r.1.x) + 12.0;
        lay.status = Some(Vec2::new(after, top + h * 0.5));
        if !narrow {
            let hover = after + f.text_size("step 11 of 11: backward", small).x + 28.0;
            lay.hover = Some(Vec2::new(hover, top + h * 0.5));
            // the key to the squares at the top right, if the value under the pointer
            // leaves it room
            if view.x - side - legend_width(f, small - 0.5) > hover + 220.0 {
                lay.legend = Some((Vec2::new(view.x - side, top + h * 0.5), true));
            }
        }
        // the blocks: their sizes in cells
        let n = net.n() as f32;
        let w = net.units[1] as f32;
        let across = (INPUTS + INPUTS + CLASSES) as f32 + 4.0 * w;
        let top_cells = n.max(w).max(CLASSES as f32);
        let bottom_cells = (n + 1.0).max(w).max(CLASSES as f32);
        let deep = top_cells + n + bottom_cells;
        let gap = if narrow { 7.0 } else { 22.0 };
        let labels = if narrow { 12.0 } else { 18.0 };
        let right = if narrow { 0.0 } else { 70.0 };
        let title_h = text + if narrow { 6.0 } else { 9.0 };
        let band_gap = if narrow { 4.0 } else { 8.0 };
        let bias_gap = if narrow { 3.0 } else { 5.0 };
        let bias_label = if narrow { 0.0 } else { small + 6.0 };
        let y0 = top + h + if narrow { 8.0 } else { 14.0 };
        // (the parts below take what is left, but no more than two fifths of a short canvas)
        let lower = ((view.y - y0) * 0.42).max(150.0).min(if narrow { 300.0 } else { 330.0 });
        let room_x = view.x - 2.0 * side - labels - right - 6.0 * gap;
        let room_y = view.y - y0 - lower - 3.0 * title_h - 2.0 * band_gap - bias_gap - bias_label;
        let cell = (room_x / across).min(room_y / deep).min(40.0).floor();
        if cell < 4.0 {
            lay.below = view.y;
            return lay;
        }
        lay.cell = cell;
        let widths = [INPUTS as f32, INPUTS as f32, w, w, w, w, CLASSES as f32];
        let mut x = side + labels;
        let mut xs = [0.0f32; 7];
        for (c, wc) in widths.iter().enumerate() {
            xs[c] = x;
            x += wc * cell + gap;
        }
        let band = [y0 + title_h, y0 + title_h + top_cells * cell + band_gap + title_h];
        let bottom_y = band[1] + n * cell + band_gap + title_h;
        for (k, slot) in SLOTS.iter().enumerate() {
            lay.blocks[k] = match *slot {
                Slot::Out(l) => Vec2::new(xs[2 * l.min(TOP)], band[0]),
                Slot::Mid(l) => Vec2::new(xs[2 * l.min(TOP)], band[1]),
                Slot::Back(l) => Vec2::new(xs[2 * l.min(TOP)], bottom_y),
                Slot::Bias(l) => Vec2::new(xs[2 * l.min(TOP)], bottom_y + n * cell + bias_gap),
                Slot::W(l) => Vec2::new(xs[2 * l.clamp(1, TOP) - 1], band[0]),
                Slot::Dw(l) => Vec2::new(xs[2 * l.clamp(1, TOP) - 1], bottom_y),
            };
        }
        lay.below = bottom_y + bottom_cells * cell + bias_label + 4.0;
        // (the rows the picked entry's sum takes, its heading among them)
        let rows = match pick {
            Pick::Weight(..) | Pick::Bias(..) => n + 3.0,
            Pick::Back(l, ..) if l >= TOP => 4.0,
            Pick::Back(l, ..) => net.units[l + 1] as f32 + 5.0,
        };
        lay.table_h = rows * (small + if narrow { 5.0 } else { 7.0 }) + 6.0;
        lay
    }

    /// The parts under the blocks, from `below` (which glides) down.
    fn lower(&mut self) {
        let (view, narrow) = (self.view, self.narrow);
        if self.cell < 4.0 {
            return;
        }
        let side = if narrow { 8.0 } else { 16.0 };
        let mut y = self.below + if narrow { 4.0 } else { 8.0 };
        let width = view.x - 2.0 * side;
        if self.legend.is_none_or(|(_, right)| !right) {
            self.legend = Some((Vec2::new(side, y + self.small * 0.5), false));
            y += self.small + 8.0;
        }
        self.lines = Some((Vec2::new(side, y), width));
        let lines_h = if narrow { 3.0 * (self.small + 5.0) } else { 2.0 * (self.text + 8.0) };
        let y = y + lines_h + if narrow { 8.0 } else { 16.0 };
        let left = view.y - y - 6.0;
        if left < 60.0 {
            return;
        }
        let chart_in = |x0: f32, x1: f32, top: f32| -> Option<Rect> {
            let (a, b) = (Vec2::new(x0 + 30.0, top + 46.0), Vec2::new(x1, (top + 46.0 + 240.0).min(view.y - 30.0)));
            (b.y - a.y >= 50.0 && b.x - a.x >= 60.0).then_some((a, b))
        };
        if view.x < 500.0 {
            // a phone: one under another, the sum, then the singular values with the
            // outer products beside them
            let table_h = self.table_h.min(left * 0.6);
            self.table = Some((Vec2::new(side, y), Vec2::new(view.x - side, y + table_h)));
            let y = y + table_h + 10.0;
            let rest = view.y - y - 6.0;
            if rest >= 90.0 {
                let chart_w = (width * 0.5).min(220.0);
                self.chart = Some((Vec2::new(side + 30.0, y + 36.0), Vec2::new(side + chart_w, view.y - 26.0)));
                self.terms = Some((Vec2::new(side + chart_w + 14.0, y), Vec2::new(view.x - side, view.y - 6.0)));
            }
        } else if view.x < 900.0 {
            // two columns: the sum with the outer products under it, the singular values
            let half = (width * 0.5).floor();
            let table_h = self.table_h.min(left);
            self.table = Some((Vec2::new(side, y), Vec2::new(side + half - 12.0, y + table_h)));
            self.chart = chart_in(side + half + 12.0, view.x - side, y);
            let below_table = y + table_h + 8.0;
            if view.y - below_table >= 70.0 {
                self.terms = Some((Vec2::new(side, below_table), Vec2::new(side + half - 12.0, view.y - 6.0)));
            }
        } else {
            // three columns: the sum, the outer products, the singular values
            let table_w = 320.0f32.min(width * 0.34);
            let chart_w = 250.0f32.min(width * 0.28);
            self.table = Some((Vec2::new(side, y), Vec2::new(side + table_w, view.y - 8.0)));
            let chart_x = view.x - side - chart_w;
            self.chart = chart_in(chart_x, view.x - side, y);
            let terms_x = side + table_w + 24.0;
            if chart_x - 24.0 - terms_x >= 120.0 {
                self.terms = Some((Vec2::new(terms_x, y), Vec2::new(chart_x - 24.0, view.y - 8.0)));
            }
        }
    }

    /// The numbers that glide when the layout changes: the cell, the first y below the
    /// blocks, the room for the sum, and each block's corner.
    fn moving_parts(&self) -> Vec<f32> {
        let mut v = Vec::with_capacity(3 + 2 * SLOTS.len());
        v.push(self.cell);
        v.push(self.below);
        v.push(self.table_h);
        for b in &self.blocks {
            v.extend([b.x, b.y]);
        }
        v
    }

    fn set_moving_parts(&mut self, v: &[f32]) {
        if v.len() != 3 + 2 * SLOTS.len() {
            return;
        }
        self.cell = v[0];
        self.below = v[1];
        self.table_h = v[2];
        for (k, b) in self.blocks.iter_mut().enumerate() {
            *b = Vec2::new(v[3 + 2 * k], v[4 + 2 * k]);
        }
    }

    /// A block's rectangle, for its shape.
    fn rect(&self, net: &Net, slot: Slot) -> Rect {
        let (rows, cols) = shape(net, slot);
        let at = self.blocks[slot_index(slot)];
        (at, at + Vec2::new(cols as f32, rows as f32) * self.cell)
    }

    /// A cell's rectangle.
    fn cell_rect(&self, slot: Slot, i: usize, j: usize) -> Rect {
        let at = self.blocks[slot_index(slot)] + Vec2::new(j as f32, i as f32) * self.cell;
        (at, at + Vec2::splat(self.cell))
    }
}

// ----- the experiments ----------------------------------------------------------------

struct Trial {
    label: &'static str,
    look_for: &'static str,
    setup: Setup,
    /// Where the film starts, and whether it plays.
    at: usize,
    play: bool,
    pick: Pick,
}

const TRIALS: [Trial; 5] = [
    Trial {
        label: "forwards, then back, a layer at a time",
        look_for: r"Four examples, ReLU. Forwards, the cyan blocks fill left to right, one matrix product a layer, $Z_l = A_{l-1} W_l^\top + b_l$, then $\varphi$ entry by entry. The loss hands back $D_3 = (P - Y)/n$: pink, under $P$ and $Y$. Then right to left: under each $W_l$ its slope $D_l^\top A_{l-1}$ (yellow), and under $A_2$ first $D_3 W_3$, back through the same weights, then that times the gates: $D_2$; and so on down. The way back multiplies matrices of the forward pass's sizes, twice for each layer but the first ($D_1 W_1$, the slope for the data, is not wanted): 256 multiply-adds against 144.",
        setup: Setup { batch: 4, width: 4, act: Act::Relu, shut: false },
        at: 0,
        play: true,
        pick: Pick::Weight(2, 3, 1),
    },
    Trial {
        label: "one example: every weight's slope is an outer product",
        look_for: r"One example, tanh, six units a layer. Every weight's slope is one column times one row, $\partial L/\partial W_l = d^\top a$: $d$ the example's row of $D_l$, $a$ its row of $A_{l-1}$. So every row of $\partial L/\partial W_2$ is a multiple of the same row $a$, and its rank is 1: of its six singular values one is not 0 (the chart), the other five are below $10^{-10}$ of it. The same for $W_1$ and $W_3$.",
        setup: Setup { batch: 1, width: 6, act: Act::Tanh, shut: false },
        at: STEPS,
        play: false,
        pick: Pick::Weight(2, 0, 0),
    },
    Trial {
        label: "a batch is a sum of outer products",
        look_for: r"Three examples: $\partial L/\partial W_2 = d_1^\top a_1 + d_2^\top a_2 + d_3^\top a_3$, the three outer products drawn small at the foot of the picture, adding up to it. Each has rank 1, so the sum has rank at most 3: of its six singular values, three are not 0. However many units, a weight's slope has rank at most the batch size. $\partial L/\partial W_3$ has rank 2, not 3: each row of $P - Y$ adds up to 0.",
        setup: Setup { batch: 3, width: 6, act: Act::Tanh, shut: false },
        at: STEPS,
        play: false,
        pick: Pick::Weight(2, 3, 2),
    },
    Trial {
        label: "a unit that is shut hands nothing back",
        look_for: r"ReLU. Unit 2 of layer 1 has its bias set so low that its $Z$ is $-1$ or less for every example: it gives 0 and its gate is 0 for all four. So column 2 of $D_1$ is 0, whatever reached it from above; row 2 of $\partial L/\partial W_1$ and its bias's slope are 0, so no gradient step on these examples can open it again. And having sent 0 forward, it gives column 2 of $\partial L/\partial W_2$ zeros too: the weights out of it get no slope either.",
        setup: Setup { batch: 4, width: 4, act: Act::Relu, shut: true },
        at: STEPS,
        play: false,
        pick: Pick::Weight(1, 1, 0),
    },
    Trial {
        label: "the same weights carry the error back",
        look_for: r"Forward, unit 3 of layer 2 reached the read-out along column 3 of $W_3$: in $Z_3 = A_2 W_3^\top$, read-out unit $j$ takes $(W_3)_{j3}$ times it. Back, the slope reaching it comes along the same column: $(D_3 W_3)_{13} = \sum_j (D_3)_{1j} (W_3)_{j3}$, the three products in the table. That is the product with $W_3$ where forward had $W_3^\top$: the transpose. Next ▶ multiplies it by the unit's gate.",
        setup: Setup { batch: 4, width: 4, act: Act::Relu, shut: false },
        at: 6,
        play: false,
        pick: Pick::Back(2, 0, 2),
    },
];

/// The entry picked as the scene opens.
const OPENING_PICK: Pick = Pick::Weight(2, 3, 1);

// ----- the scene ----------------------------------------------------------------------

/// Eases: the cells as they fill or change, the layout, the marks.
const VALUES_EASE: f32 = 0.09;
const LAYOUT_EASE: f32 = 0.08;
const MARKS_EASE: f32 = 0.07;

/// Kinds of line on the screen with a batch each (see `plot::apart`): the blocks'
/// frames, the step's marks, the pick's rings, the chart's dashed line, and the rules
/// between each D and its column sums.
const FRAMES: u8 = plot::OWN;
const MARKS: u8 = plot::OWN + 1;
const RINGS: u8 = plot::OWN + 2;
const TOLERANCE: u8 = plot::OWN + 3;
const SIGNS: u8 = plot::OWN + 4;

fn in_order(f: &mut Frame) {
    for kind in [plot::AXES, plot::BUTTONS, FRAMES, MARKS, RINGS, SIGNS] {
        plot::apart(f, kind, false);
    }
    plot::apart(f, TOLERANCE, true);
}

pub struct BackpropMat {
    setup: Setup,
    net: Net,
    /// The film: steps done (0 to `STEPS`), whether it plays, steps a second, and the
    /// part of a step owed.
    s: usize,
    playing: bool,
    rate: f32,
    owed: f32,
    pick: Pick,
    /// The pick's slope by a difference of the loss, for the pick and setup it was made
    /// for.
    check: Option<(Pick, Setup, f64)>,
    /// What is drawn, following what was worked out.
    layout: Follow,
    values: Follow,
    marks: Follow,
    moving: bool,
    /// The cell under the pointer.
    hovered: Option<(Slot, usize, usize)>,
}

impl BackpropMat {
    fn new() -> BackpropMat {
        let setup = Setup::default();
        let net = Net::new(setup);
        let mut scene = BackpropMat {
            setup,
            net,
            s: STEPS,
            playing: false,
            rate: 1.0,
            owed: 0.0,
            pick: OPENING_PICK,
            check: None,
            layout: Follow::default(),
            values: Follow::default(),
            marks: Follow::default(),
            moving: false,
            hovered: None,
        };
        scene.values.snap(&scene.value_targets());
        scene
    }

    /// A new setup: the network is made afresh, the pick kept inside its shapes.
    fn set_setup(&mut self, setup: Setup) {
        let setup = setup.clamped();
        if setup != self.setup || self.net.setup != setup {
            self.setup = setup;
            self.net = Net::new(setup);
        }
        self.pick = self.pick.within(&self.net);
    }

    /// Set experiment `i` up.
    fn set_up(&mut self, i: usize) {
        let t = &TRIALS[i.min(TRIALS.len() - 1)];
        self.set_setup(t.setup);
        self.pick = t.pick.within(&self.net);
        self.s = t.at.min(STEPS);
        (self.playing, self.owed) = (t.play, 0.0);
        self.rate = 1.0;
    }

    fn next(&mut self) {
        self.playing = false;
        self.s = (self.s + 1).min(STEPS);
    }

    fn back(&mut self) {
        self.playing = false;
        self.s = self.s.saturating_sub(1);
    }

    fn toggle_play(&mut self) {
        if self.playing {
            self.playing = false;
        } else {
            if self.s >= STEPS {
                self.s = 0;
            }
            (self.playing, self.owed) = (true, 0.0);
        }
    }

    /// Every cell's square's side as a share of the cell (0 for a block not yet worked
    /// out), signed as its number is, block after block in `SLOTS` order, and then how
    /// filled each block is. (A square's area is its number's size beside its block's
    /// scale, so its side is the square root of that: and it is the side that glides,
    /// so that a square shrinks evenly in pixels and leaves no dot lingering.)
    fn value_targets(&self) -> Vec<f32> {
        let net = &self.net;
        let mut v = Vec::with_capacity(400);
        for slot in SLOTS {
            let (rows, cols) = shape(net, slot);
            let k = scale(net, slot);
            match block_values(net, slot, self.s) {
                Some(m) => v.extend((0..rows * cols).map(|e| (m.at(e / cols.max(1), e % cols.max(1)) / k) as f32)),
                None => v.extend(std::iter::repeat_n(0.0, rows * cols)),
            }
        }
        for slot in SLOTS {
            v.push(if block_values(net, slot, self.s).is_some() { 1.0 } else { 0.0 });
        }
        let side = |x: f32| x.signum() * x.abs().min(1.0).sqrt();
        let n = v.len() - SLOTS.len();
        v.iter().enumerate().map(|(e, x)| if !x.is_finite() { 0.0 } else if e < n { side(*x) } else { *x }).collect()
    }

    /// Where each block's values start in `value_targets`.
    fn offsets(&self) -> [usize; 20] {
        let mut out = [0; 20];
        for (k, slot) in SLOTS.iter().enumerate() {
            let (r, c) = shape(&self.net, *slot);
            out[k + 1] = out[k] + r * c;
        }
        out
    }

    /// Move the film on by `dt` seconds of the scene's time.
    fn advance(&mut self, dt: f32) {
        if !self.playing {
            return;
        }
        self.owed = (self.owed + self.rate.max(0.0) * dt.max(0.0)).min(2.0);
        while self.owed >= 1.0 && self.s < STEPS {
            self.s += 1;
            self.owed -= 1.0;
        }
        if self.s >= STEPS {
            (self.playing, self.owed) = (false, 0.0);
        }
    }

    /// The pick's check by differences, worked out once for each pick and setup.
    fn checked(&mut self) -> f64 {
        let pick = self.pick.within(&self.net);
        match self.check {
            Some((p, s, v)) if p == pick && s == self.setup => v,
            _ => {
                let v = self.net.by_difference(pick);
                self.check = Some((pick, self.setup, v));
                v
            }
        }
    }
}

impl Scene for BackpropMat {
    fn view(&self) -> View {
        View::Flat { center: Vec2::ZERO, half_height: 1.0 }
    }

    fn settings(&self, s: &mut ViewSettings) {
        s.grid = false;
        s.bloom = s.bloom.min(0.2);
    }

    fn animated(&self) -> bool {
        self.playing || self.moving
    }

    fn pointer_hint(&self) -> Option<&'static str> {
        Some("buttons over the picture: click    a cell: click to see what it is made of")
    }

    fn ui(&mut self, ui: &mut egui::Ui, _: &Gfx) {
        ui::note(ui, r"A network finds its gradient a layer at a time, for a whole batch of examples at once, with matrix products. Here 2 numbers go in, through two hidden layers, to 3 classes. Every matrix has one row per example, and a weight matrix $W_l$ one row per unit of its layer, so a layer is $A_{l-1} W_l^\top$. Forwards, left to right:");
        ui::equation(ui, r"Z_l = A_{l-1} W_l^\top + b_l, \quad A_l = \varphi(Z_l)");
        ui::note(ui, r"The loss hands back $D_3 = (P - Y)/n$, its slope with respect to the read-out's $Z_3$. Then, right to left:");
        ui::equation(ui, r"D_l = (D_{l+1} W_{l+1}) \odot \varphi'(Z_l), \quad \frac{\partial L}{\partial W_l} = D_l^\top A_{l-1}");
        ui::note(ui, r"($\odot$: entry by entry.) Every slope has the shape of what it is the slope of. The way back needs every $A_l$ and gate of the way forward, kept; and it never builds a Jacobian, only products like $D_{l+1} W_{l+1}$. Step through with the buttons over the picture; click a cell to see the sum it is.");

        ui::section(ui, "Things to try");
        for (i, t) in TRIALS.iter().enumerate() {
            if ui::experiment(ui, t.label, t.look_for) {
                self.set_up(i);
            }
        }

        ui::section(ui, "The film");
        if let Some(b) = ui::buttons(ui, &["↺ start", "◀ back", "next ▶", if self.playing { "❚❚ pause" } else { "▶▶ play" }, "end ⇥"]) {
            match b {
                0 => (self.s, self.playing) = (0, false),
                1 => self.back(),
                2 => self.next(),
                3 => self.toggle_play(),
                _ => (self.s, self.playing) = (STEPS, false),
            }
        }
        ui::slider_log(ui, "steps a second", &mut self.rate, 0.25..=4.0);

        ui::section(ui, "The network");
        let mut setup = self.setup;
        let mut changed = ui::slider_int(ui, "examples in the batch, $n$", &mut setup.batch, 1..=MOST_BATCH);
        changed |= ui::slider_int(ui, "units in each hidden layer", &mut setup.width, 1..=MOST_WIDTH);
        let names: Vec<&str> = Act::ALL.iter().map(|a| a.name()).collect();
        let mut chosen = Act::ALL.iter().position(|a| *a == setup.act).unwrap_or(1);
        if ui::choice(ui, &names, &mut chosen) {
            setup.act = Act::ALL[chosen];
            changed = true;
        }
        ui::note(ui, r"The activation $\varphi$ of both hidden layers. The read-out has none: its $Z_3$ goes to a softmax.");
        let unit = shut_unit(setup.width) + 1;
        changed |= ui::checkbox(ui, &mut setup.shut, &format!("push unit {unit} of layer 1 below 0")).changed();
        ui::note(ui, r"Gives it a bias low enough that its $Z$ is $-1$ or less for every example (with ReLU: shut for all of them).");
        if changed {
            self.set_setup(setup);
        }

        ui::section(ui, "Reading the picture");
        ui::key(
            ui,
            &[
                (FORWARD, r"cyan: what goes forward: the data $X = A_0$, each hidden layer's output $A_l$, and the probabilities $P$"),
                (GATE, r"grey: the gates $\varphi'(Z_l)$, each unit's slope at what it received"),
                (LOSS, r"orange: the targets $Y$, a 1 at each example's class; and the loss $L$"),
                (BACKWARD, r"pink: $D_l$, the slope of $L$ with respect to $Z_l$ (for the step before it, $D_{l+1} W_{l+1}$, the slope with respect to $A_l$)"),
                (WEIGHT, r"yellow: the weights $W_l$ (top) and their slopes (bottom); under each $D_l$ its column sums, the slope of the bias $b_l$"),
                (pal::DIM, "a filled square: a positive number; an outline: a negative one; its area: the number's size beside the largest in its block; an empty cell: 0"),
                (CURSOR, "white frame: what the step works out; thinner frames, in the blocks' colours: what it uses; white ring: the entry picked, with rings round what it is made of"),
            ],
        );

        ui::section(ui, "Words");
        ui::term(ui, "Batch:", "the examples worked on together: here the rows of every matrix but the weights.");
        ui::term(ui, "Unit:", r"one neuron: a column of $A_l$, $Z_l$ and $D_l$, a row of $W_l$.");
        ui::term(ui, "Slope, gradient:", r"$\partial L/\partial W$, how fast the loss changes as each entry of $W$ is nudged: a matrix of $W$'s shape.");
        ui::term(ui, "Gate:", r"$\varphi'(z)$, the slope of the activation at the value $z$ the unit had on the way forward. On the way back each unit multiplies what reaches it by its gate.");
        ui::term(ui, "Softmax, cross-entropy:", r"$P_{ij} = e^{z_{ij}} / \sum_k e^{z_{ik}}$, with $z$ the read-out's $Z_3$; the loss is the mean over the examples of $-\ln P_{iy_i}$, $y_i$ the example's class (natural logarithm).");
        ui::term(ui, "Outer product:", r"a column times a row, $u^\top v$ for rows $u$ and $v$: a matrix every row of which is a multiple of $v$. Its rank is 1 (or 0).");
        ui::term(ui, "Singular values, rank:", r"the stretches of a matrix's SVD; the number that are not 0 is its rank. Here one counts as 0 below $10^{-10}$ times the largest.");
        let (n, w) = (self.net.n(), self.net.units[2]);
        ui::term(
            ui,
            "Vector–Jacobian product:",
            &format!(r"$D_3 W_3$ is what the Jacobian of $Z_3$ with respect to $A_2$ (every output of every example against every input) does to $D_3$, without that Jacobian being built: it would be ${} \times {}$, with 0 wherever the output and the input belong to different examples, since examples do not mix.", 3 * n, w * n),
        );

        ui::section(ui, "Numbers");
        ui::readout_colored(ui, "loss $L$", num(self.net.loss, 4), LOSS);
        let pick = self.pick.within(&self.net);
        let fd = self.checked();
        ui::readout(ui, &format!("picked: {}", pick_name(pick)), format!("${}$", four(pick.value(&self.net))));
        ui::readout(ui, "the same, by a difference of the loss", format!("${}$", four(fd)));
        if self.net.corners() > 0 {
            ui::note(ui, r"Some unit here has $Z$ exactly 0 (all its inputs 0, its bias 0), at ReLU's corner, where it has no slope: the tape takes the slope of the piece below (that is the gate shown), and a difference of the loss, which takes both sides, can disagree with it.");
        }
        for l in 1..=TOP {
            ui::readout(ui, &format!(r"rank of $\partial L/\partial W_{l}$"), format!("{} (at most {})", self.net.rank(l), self.net.rank_bound(l)));
        }
        let (fw, bw) = self.net.work();
        ui::readout(ui, "multiply-adds, forward and back", format!("{fw}, {bw}"));
        ui::see_also(
            ui,
            &[
                ("backprop1", "the same sweep with every number on the wires"),
                ("chain", "the chain rule it all comes from"),
                ("lossgrad", r"where $P - Y$ comes from"),
                ("lowrank", "a matrix as a sum of outer products"),
                ("actgrad", "the gates of each activation"),
            ],
        );
    }

    fn frame(&mut self, f: &mut Frame) {
        let view = f.view_size();
        // (a canvas squeezed to a sliver is one the controls have opened over, on a phone)
        let covered = view.x < 160.0;
        let target = Layout::of(f, view, f.top_inset, &self.net, self.pick.within(&self.net));
        if !covered {
            f.handles.capture();
            f.handles.claim_clicks();
            self.buttons(f, &target);
            self.click(f, &target);
        }
        self.advance(f.dt);
        let mut moving = self.values.follow(&self.value_targets(), f.real_dt, VALUES_EASE);
        if covered || target.cell < 4.0 {
            // (nothing to draw: the layout is left as it was, and met again as it was)
            self.moving = moving;
            return;
        }
        // a layout coming from none (the first frame, a canvas that was too small) is
        // there at once; any other glides from the old to the new
        let fresh = self.layout.get(0) < 4.0;
        if fresh {
            self.layout.snap(&target.moving_parts());
        }
        moving |= self.layout.follow(&target.moving_parts(), f.real_dt, LAYOUT_EASE);
        let mut lay = target.clone();
        lay.set_moving_parts(&self.layout.shown);
        lay.lower();
        let marks = self.mark_targets(&lay);
        if fresh {
            self.marks.snap(&marks);
        }
        moving |= self.marks.follow(&marks, f.real_dt, MARKS_EASE);
        self.moving = moving;
        in_order(f);
        self.hovered = plot::pointer(f).and_then(|p| self.cell_at(&lay, p));
        self.blocks(f, &lay);
        self.marks(f);
        self.labels(f, &lay);
        self.lines(f, &lay);
        self.table(f, &lay);
        self.terms(f, &lay);
        self.chart(f, &lay);
    }
}

/// The picked entry's name, as markup.
fn pick_name(p: Pick) -> String {
    match p {
        Pick::Weight(l, j, k) => format!(r"$(\partial L/\partial W_{l})_{{{}{}}}$", j + 1, k + 1),
        Pick::Bias(l, j) => format!(r"$(\partial L/\partial b_{l})_{{{}}}$", j + 1),
        Pick::Back(l, i, k) => format!("$(D_{l})_{{{}{}}}$", i + 1, k + 1),
    }
}

impl BackpropMat {
    /// The film's buttons over the picture.
    fn buttons(&mut self, f: &mut Frame, lay: &Layout) {
        let size = lay.text - 0.5;
        let words: [&str; 5] = if lay.narrow {
            [BUTTON_SHORT[0], BUTTON_SHORT[1], BUTTON_SHORT[2], if self.playing { "❚❚" } else { "▶▶" }, BUTTON_SHORT[4]]
        } else {
            [BUTTON_WORDS[0], BUTTON_WORDS[1], BUTTON_WORDS[2], if self.playing { "❚❚ pause" } else { "▶▶ play" }, BUTTON_WORDS[4]]
        };
        for (k, r) in lay.buttons.clone().iter().enumerate() {
            if plot::button(f, *r, words[k], size, k == 3 && self.playing) {
                match k {
                    0 => (self.s, self.playing) = (0, false),
                    1 => self.back(),
                    2 => self.next(),
                    3 => self.toggle_play(),
                    _ => (self.s, self.playing) = (STEPS, false),
                }
            }
        }
        if let Some(at) = lay.status {
            let phase = match self.s {
                0 => "the start",
                1..=3 => "forward",
                _ => "backward",
            };
            let text = if lay.narrow { format!("{} of {STEPS}", self.s) } else { format!("step {} of {STEPS}: {phase}", self.s) };
            f.text_px(at, text, lay.small, pal::DIM, Align2::LEFT_CENTER);
        }
    }

    /// The block and cell under a point of the screen.
    fn cell_at(&self, lay: &Layout, p: Vec2) -> Option<(Slot, usize, usize)> {
        if lay.cell < 4.0 {
            return None;
        }
        SLOTS.iter().find_map(|&slot| {
            let (min, max) = lay.rect(&self.net, slot);
            if !plot::within(p, (min, max), 0.0) {
                return None;
            }
            let (rows, cols) = shape(&self.net, slot);
            let j = (((p.x - min.x) / lay.cell).floor().max(0.0) as usize).min(cols.max(1) - 1);
            let i = (((p.y - min.y) / lay.cell).floor().max(0.0) as usize).min(rows.max(1) - 1);
            Some((slot, i, j))
        })
    }

    /// A click on a cell picks it, or what it belongs to.
    fn click(&mut self, f: &mut Frame, lay: &Layout) {
        if !f.ptr.clicked {
            return;
        }
        let Some(p) = plot::pointer(f) else { return };
        let Some((slot, i, j)) = self.cell_at(lay, p) else { return };
        let pick = match slot {
            Slot::W(l) | Slot::Dw(l) => Some(Pick::Weight(l, i, j)),
            Slot::Bias(l) => Some(Pick::Bias(l, j)),
            Slot::Out(l) | Slot::Mid(l) | Slot::Back(l) if l >= 1 => Some(Pick::Back(l, i, j)),
            _ => None,
        };
        if let Some(p) = pick {
            self.pick = p.within(&self.net);
        }
    }

    /// Where the step's marks and the pick's rings go: rectangles (corners), the step's
    /// output, its two inputs, then five for the pick.
    fn mark_targets(&self, lay: &Layout) -> Vec<f32> {
        let net = &self.net;
        let mut rects: Vec<Rect> = Vec::with_capacity(8);
        let pad = |(a, b): Rect, by: f32| (a - Vec2::splat(by), b + Vec2::splat(by));
        let (out, ins) = step_blocks(self.s);
        rects.push(pad(lay.rect(net, out), 4.0));
        for slot in ins {
            rects.push(pad(lay.rect(net, slot), 2.0));
        }
        for r in self.pick_rects(lay) {
            rects.push(r);
        }
        rects.iter().flat_map(|(a, b)| [a.x, a.y, b.x, b.y]).collect()
    }

    /// The pick's five rings: the entry itself, two cells it goes with, and the two
    /// rows or columns it is made from (each also where it is only to coincide).
    fn pick_rects(&self, lay: &Layout) -> [Rect; 5] {
        let net = &self.net;
        let c = lay.cell;
        let column = |slot: Slot, k: usize| {
            let (rows, _) = shape(net, slot);
            let at = lay.blocks[slot_index(slot)] + Vec2::new(k as f32 * c, 0.0);
            (at - Vec2::splat(1.5), at + Vec2::new(c, rows as f32 * c) + Vec2::splat(1.5))
        };
        let row = |slot: Slot, i: usize| {
            let (_, cols) = shape(net, slot);
            let at = lay.blocks[slot_index(slot)] + Vec2::new(0.0, i as f32 * c);
            (at - Vec2::splat(1.5), at + Vec2::new(cols as f32 * c, c) + Vec2::splat(1.5))
        };
        let cell = |slot: Slot, i: usize, j: usize| {
            let (a, b) = lay.cell_rect(slot, i, j);
            (a - Vec2::splat(1.0), b + Vec2::splat(1.0))
        };
        match self.pick.within(net) {
            Pick::Weight(l, j, k) => [cell(Slot::Dw(l), j, k), cell(Slot::W(l), j, k), cell(Slot::Bias(l), 0, j), column(Slot::Back(l), j), column(Slot::Out(l - 1), k)],
            Pick::Bias(l, j) => [cell(Slot::Bias(l), 0, j), cell(Slot::Bias(l), 0, j), cell(Slot::Bias(l), 0, j), column(Slot::Back(l), j), column(Slot::Back(l), j)],
            Pick::Back(l, i, k) if l >= TOP => [cell(Slot::Back(l), i, k), cell(Slot::Out(l), i, k), cell(Slot::Mid(l), i, k), row(Slot::Out(l), i), row(Slot::Mid(l), i)],
            Pick::Back(l, i, k) => [cell(Slot::Back(l), i, k), cell(Slot::Mid(l), i, k), cell(Slot::Out(l), i, k), row(Slot::Back(l + 1), i), column(Slot::W(l + 1), k)],
        }
    }

    /// The colours of the pick's five rings.
    fn pick_colors(&self) -> [Color; 5] {
        match self.pick {
            Pick::Weight(..) => [CURSOR, WEIGHT, WEIGHT, BACKWARD, FORWARD],
            Pick::Bias(..) => [CURSOR, CURSOR, CURSOR, BACKWARD, BACKWARD],
            Pick::Back(l, ..) if l >= TOP => [CURSOR, FORWARD, LOSS, FORWARD, LOSS],
            Pick::Back(..) => [CURSOR, GATE, FORWARD, BACKWARD, WEIGHT],
        }
    }

    /// The blocks: each a grid of cells with a square in each, filled for a positive
    /// number and outlined for a negative one.
    fn blocks(&self, f: &mut Frame, lay: &Layout) {
        let net = &self.net;
        let offsets = self.offsets();
        let fills_at = offsets[SLOTS.len()];
        let c = lay.cell;
        for (k, slot) in SLOTS.iter().enumerate() {
            let (rows, cols) = shape(net, *slot);
            let filled = self.values.get(fills_at + k).clamp(0.0, 1.0);
            let color = color_of(*slot);
            let (min, max) = lay.rect(net, *slot);
            let backing = Color::hex(0x0C101B).mix(Color::hex(0x161C2B), filled);
            let gap = if c >= 10.0 { 1.0 } else { 0.0 };
            for i in 0..rows {
                for j in 0..cols {
                    let at = min + Vec2::new(j as f32, i as f32) * c;
                    hud_rect_at(f, 0, at, at + Vec2::splat(c - gap), backing);
                    let v = self.values.get(offsets[k] + i * cols + j);
                    let side = (c - 3.0).max(1.0) * v.abs().min(1.0);
                    if side < 0.6 {
                        continue;
                    }
                    square(f, at + Vec2::splat((c - gap) * 0.5), side, v > 0.0, color, 0.14);
                }
            }
            // the frame, brighter once the block is worked out
            let edge = [min - Vec2::splat(1.0), Vec2::new(max.x + 1.0, min.y - 1.0), max + Vec2::splat(1.0), Vec2::new(min.x - 1.0, max.y + 1.0), min - Vec2::splat(1.0)];
            plot::path_of(f, FRAMES, &edge, 1.0, color.a(0.25 + 0.35 * filled));
            // the bias's slope is a sum: a line between D and it
            if let Slot::Bias(_) = slot {
                let y = min.y - if lay.narrow { 1.5 } else { 2.5 };
                plot::line_of(f, SIGNS, Vec2::new(min.x, y), Vec2::new(max.x, y), 1.0, WEIGHT.a(0.3 + 0.5 * filled));
            }
        }
    }

    /// The step's frames and the pick's rings.
    fn marks(&self, f: &mut Frame) {
        let r = |k: usize| -> Rect { (Vec2::new(self.marks.get(4 * k), self.marks.get(4 * k + 1)), Vec2::new(self.marks.get(4 * k + 2), self.marks.get(4 * k + 3))) };
        let outline = |(a, b): Rect| [a, Vec2::new(b.x, a.y), b, Vec2::new(a.x, b.y), a];
        let started = if self.s == 0 { 0.0 } else { 1.0 };
        let (_, ins) = step_blocks(self.s);
        plot::path_of(f, MARKS, &outline(r(0)), 2.2, CURSOR.a(0.9 * started));
        for (k, slot) in ins.iter().enumerate() {
            plot::path_of(f, MARKS, &outline(r(1 + k)), 1.3, color_of(*slot).mix(pal::WHITE, 0.3).a(0.75 * started));
        }
        // (the pick's rings once what it picked is worked out; the table says when)
        let colors = self.pick_colors();
        let alpha = if self.s >= self.pick.within(&self.net).ready_at() { 0.95 } else { 0.0 };
        for (k, color) in colors.iter().enumerate() {
            let width = if k == 0 { 2.2 } else { 1.6 };
            plot::path_of(f, RINGS, &outline(r(3 + k)), width, color.a(alpha));
        }
    }

    /// The blocks' titles, the rows' numbers, and the loss.
    fn labels(&self, f: &mut Frame, lay: &Layout) {
        let net = &self.net;
        let size = lay.text;
        for slot in SLOTS {
            if let Slot::Bias(_) = slot {
                if !lay.narrow {
                    let (min, max) = lay.rect(net, slot);
                    f.text_px(Vec2::new((min.x + max.x) * 0.5, max.y + 3.0), title(slot, self.s, false), lay.small - 0.5, WEIGHT.a(0.85), Align2::CENTER_TOP);
                }
                continue;
            }
            let (min, max) = lay.rect(net, slot);
            let (rows, cols) = shape(net, slot);
            let room = (max.x - min.x) + if lay.narrow { 6.0 } else { 20.0 };
            let full = title(slot, self.s, false);
            let mut text = full.clone();
            if !lay.narrow {
                let with = format!(r"{full}  ${}$", dims(rows, cols));
                if f.text_size(&with, size).x <= room {
                    text = with;
                }
            }
            if f.text_size(&text, size).x > room {
                text = title(slot, self.s, true);
            }
            let filled = block_values(net, slot, self.s).is_some();
            let color = if filled { color_of(slot).mix(pal::WHITE, 0.25) } else { pal::DIM };
            // (the data's title starts at its block, leaving the rows' numbers their room)
            if slot == Slot::Out(0) {
                f.text_px(Vec2::new(min.x, min.y - 4.0), text, size, color, Align2::LEFT_BOTTOM);
            } else {
                f.text_px(Vec2::new((min.x + max.x) * 0.5, min.y - 4.0), text, size, color, Align2::CENTER_BOTTOM);
            }
        }
        // the examples' numbers, left of the data
        let (min, _) = lay.rect(net, Slot::Out(0));
        let small = lay.small - if lay.narrow { 1.0 } else { 0.0 };
        if lay.cell >= 9.0 {
            for i in 0..net.n() {
                f.text_px(Vec2::new(min.x - 4.0, min.y + (i as f32 + 0.5) * lay.cell), format!("{}", i + 1), small, pal::DIM, Align2::RIGHT_CENTER);
            }
            f.text_px(Vec2::new(min.x - 4.0, min.y - 4.0), "$i$", small, pal::DIM, Align2::RIGHT_BOTTOM);
        }
        if let Some((at, right)) = lay.legend {
            legend(f, at, right, lay.small - 0.5);
        }
        // the loss, beside the targets
        if !lay.narrow {
            let (_, max) = lay.rect(net, Slot::Mid(TOP));
            let (min_p, _) = lay.rect(net, Slot::Mid(TOP));
            let y = (min_p.y + max.y) * 0.5;
            let text = if self.s >= made_at(Slot::Out(TOP)) { format!("$L = {}$", num(net.loss, 3)) } else { "$L$".into() };
            f.text_px(Vec2::new(max.x + 10.0, y), text, size, LOSS, Align2::LEFT_CENTER);
        }
        // the cell under the pointer
        if let (Some(at), Some((slot, i, j))) = (lay.hover, self.hovered) {
            if let Some(m) = block_values(net, slot, self.s) {
                let name = match slot {
                    Slot::Out(0) => "X".to_string(),
                    Slot::Out(l) if l == TOP => "P".into(),
                    Slot::Out(l) => format!("A_{l}"),
                    Slot::Mid(l) if l == TOP => "Y".into(),
                    Slot::Mid(l) => format!(r"\varphi'(Z_{l})"),
                    Slot::Back(l) if l <= DEPTH && self.s + 1 == made_at(slot) => format!("D_{} W_{}", l + 1, l + 1),
                    Slot::Back(l) => format!("D_{l}"),
                    Slot::Bias(l) => format!(r"\partial L/\partial b_{l}"),
                    Slot::W(l) => format!("W_{l}"),
                    Slot::Dw(l) => format!(r"\partial L/\partial W_{l}"),
                };
                let index = if let Slot::Bias(_) = slot { format!("{}", j + 1) } else { format!("{}{}", i + 1, j + 1) };
                f.text_px(at, format!("$({name})_{{{index}}} = {}$", four(m.at(i, j))), lay.small, pal::TEXT, Align2::LEFT_CENTER);
            }
        }
    }

    /// The step's line of arithmetic and its shapes.
    fn lines(&self, f: &mut Frame, lay: &Layout) {
        let Some((at, width)) = lay.lines else { return };
        let (first, second) = step_lines(&self.net, self.s, lay.narrow);
        if lay.narrow {
            // on a phone each may need two lines
            let mut y = at.y;
            for (text, color, size) in [(first, pal::TEXT, lay.small + 0.5), (second, pal::DIM, lay.small)] {
                y += fit(f, Vec2::new(at.x, y), &text, size, width, color, 2);
            }
            return;
        }
        fit(f, at, &first, lay.text + 0.5, width, pal::TEXT, 1);
        fit(f, at + Vec2::new(0.0, lay.text + 9.0), &second, lay.small + 0.5, width, pal::DIM, 1);
    }

    /// The picked entry, worked out: the sum it is, one term per row.
    fn table(&self, f: &mut Frame, lay: &Layout) {
        let Some((min, max)) = lay.table else { return };
        let net = &self.net;
        let pick = self.pick.within(net);
        let small = lay.small;
        let row_h = small + if lay.narrow { 5.0 } else { 7.0 };
        let heading = match pick {
            Pick::Weight(1, j, k) => format!("{}: unit {} of layer 1, from column {} of the data", pick_name(pick), j + 1, k + 1),
            Pick::Weight(l, j, k) => format!("{}: unit {} of layer {l}, from unit {} of layer {}", pick_name(pick), j + 1, k + 1, l - 1),
            Pick::Bias(l, j) => format!("{}: unit {} of layer {l}", pick_name(pick), j + 1),
            Pick::Back(l, i, k) => format!("{}: example {}, unit {} of layer {l}", pick_name(pick), i + 1, k + 1),
        };
        fit(f, min, &heading, small + 0.5, max.x - min.x, pal::TEXT, 1);
        let y0 = min.y + row_h + 4.0;
        if self.s < pick.ready_at() {
            let text = format!("worked out at step {} of {STEPS}", pick.ready_at());
            f.text_px(Vec2::new(min.x, y0), text, small, pal::DIM, Align2::LEFT_TOP);
            return;
        }
        let width = max.x - min.x;
        let cols = [min.x + 4.0, min.x + width * 0.14, min.x + width * 0.42, min.x + width * 0.7];
        let rows_left = ((max.y - y0) / row_h).floor().max(0.0) as usize;
        let line = |f: &mut Frame, r: usize, cells: [String; 4], colors: [Color; 4]| {
            if r >= rows_left {
                return;
            }
            let y = y0 + r as f32 * row_h;
            for (c, (text, color)) in cells.iter().zip(colors).enumerate() {
                if !text.is_empty() {
                    f.text_px(Vec2::new(cols[c], y), text.as_str(), small, color, Align2::LEFT_TOP);
                }
            }
        };
        let dim = pal::DIM;
        match pick {
            Pick::Weight(l, j, k) => {
                let (jj, kk) = (j + 1, k + 1);
                line(f, 0, ["$i$".into(), format!("$(D_{l})_{{i{jj}}}$"), format!("$(A_{})_{{i{kk}}}$", l - 1), "product".into()], [dim, BACKWARD, FORWARD, dim]);
                let n = net.n();
                // (the header, a row for each example and the sum; if they do not all
                // fit, as many examples as leave room for a line saying how many more)
                let shown = if n + 2 <= rows_left { n } else { rows_left.saturating_sub(3) };
                for i in 0..shown {
                    let (d, a) = (net.d[l].at(i, j), net.a[l - 1].at(i, k));
                    line(f, 1 + i, [format!("{}", i + 1), format!("${}$", four(d)), format!("${}$", four(a)), format!("${}$", four(d * a))], [dim, pal::TEXT, pal::TEXT, pal::TEXT]);
                }
                let r = 1 + shown;
                if shown < n {
                    line(f, r, [String::new(), format!("({} more)", n - shown), String::new(), String::new()], [dim; 4]);
                }
                let r = r + usize::from(shown < n);
                line(f, r, [String::new(), String::new(), "the sum:".into(), format!("${}$", four(net.dw[l].at(j, k)))], [dim, dim, dim, WEIGHT]);
            }
            Pick::Bias(l, j) => {
                let jj = j + 1;
                line(f, 0, ["$i$".into(), format!("$(D_{l})_{{i{jj}}}$"), String::new(), String::new()], [dim, BACKWARD, dim, dim]);
                let n = net.n();
                let shown = if n + 2 <= rows_left { n } else { rows_left.saturating_sub(3) };
                for i in 0..shown {
                    line(f, 1 + i, [format!("{}", i + 1), format!("${}$", four(net.d[l].at(i, j))), String::new(), String::new()], [dim, pal::TEXT, dim, dim]);
                }
                if shown < n {
                    line(f, 1 + shown, [String::new(), format!("({} more)", n - shown), String::new(), String::new()], [dim; 4]);
                }
                let r = 1 + shown + usize::from(shown < n);
                line(f, r, [String::new(), String::new(), "the sum:".into(), format!("${}$", four(net.db[l].at(0, j)))], [dim, dim, dim, WEIGHT]);
            }
            Pick::Back(l, i, k) if l >= TOP => {
                let (ii, kk) = (i + 1, k + 1);
                line(f, 0, [String::new(), format!("$P_{{{ii}{kk}}}$"), format!("$Y_{{{ii}{kk}}}$"), "$(P - Y)/n$".into()], [dim, FORWARD, LOSS, dim]);
                let (p, y) = (net.a[TOP].at(i, k), net.y.at(i, k));
                line(f, 1, [String::new(), format!("${}$", four(p)), format!("${}$", four(y)), format!("${}$", four(net.d[TOP].at(i, k)))], [dim, pal::TEXT, pal::TEXT, BACKWARD]);
                line(f, 2, [String::new(), format!("$n = {}$", net.n()), String::new(), String::new()], [dim; 4]);
            }
            Pick::Back(l, i, k) => {
                let (ii, kk) = (i + 1, k + 1);
                let up = l + 1;
                line(f, 0, ["$j$".into(), format!("$(D_{up})_{{{ii}j}}$"), format!("$(W_{up})_{{j{kk}}}$"), "product".into()], [dim, BACKWARD, WEIGHT, dim]);
                let m = net.units[up];
                let shown = if m + 4 <= rows_left { m } else { rows_left.saturating_sub(4) };
                for j in 0..shown {
                    let (d, w) = (net.d[up].at(i, j), net.w[up].at(j, k));
                    line(f, 1 + j, [format!("{}", j + 1), format!("${}$", four(d)), format!("${}$", four(w)), format!("${}$", four(d * w))], [dim, pal::TEXT, pal::TEXT, pal::TEXT]);
                }
                let r = 1 + shown;
                let g = net.back[l].at(i, k);
                line(f, r, [String::new(), String::new(), format!("$(D_{up} W_{up})_{{{ii}{kk}}}$:"), format!("${}$", four(g))], [dim, dim, dim, BACKWARD]);
                if self.s >= made_at(Slot::Back(l)) {
                    let gate = net.gate[l].at(i, k);
                    line(f, r + 1, [String::new(), String::new(), format!(r"times $\varphi'(Z_{l})_{{{ii}{kk}}}$:"), format!("${}$", four(gate))], [dim, dim, dim, GATE]);
                    line(f, r + 2, [String::new(), String::new(), format!("$(D_{l})_{{{ii}{kk}}}$:"), format!("${}$", four(net.d[l].at(i, k)))], [dim, dim, dim, BACKWARD]);
                }
            }
        }
    }

    /// The picked layer's weight slope as the sum of one outer product per example.
    fn terms(&self, f: &mut Frame, lay: &Layout) {
        let Some((min, max)) = lay.terms else { return };
        let net = &self.net;
        let l = self.pick.within(net).layer();
        if self.s < made_at(Slot::Dw(l)) {
            return;
        }
        let small = lay.small;
        let n = net.n();
        let heading = if lay.narrow {
            format!(r"$\partial L/\partial W_{l} = \sum_i d_i^\top a_i$")
        } else if n == 1 {
            format!(r"one example: $\partial L/\partial W_{l} = d_1^\top a_1$, one outer product")
        } else {
            format!(r"$\partial L/\partial W_{l} = \sum_i d_i^\top a_i$: one outer product per example")
        };
        fit(f, min, &heading, small + 0.5, max.x - min.x, pal::TEXT, 1);
        if !lay.narrow {
            fit(f, min + Vec2::new(0.0, small + 8.0), &format!("$d_i$: row $i$ of $D_{l}$; $a_i$: row $i$ of $A_{}$", l - 1), small, max.x - min.x, pal::DIM, 1);
        }
        let (rows, cols) = shape(net, Slot::Dw(l));
        let pieces: Vec<Mat> = (0..n).map(|i| net.outer(l, i)).collect();
        let k = pieces.iter().map(Mat::largest).fold(net.dw[l].largest(), f64::max);
        let k = if k > 0.0 { k } else { 1.0 };
        let top = min.y + if lay.narrow { small + 10.0 } else { 2.0 * small + 20.0 };
        let sign_w = if lay.narrow { 10.0 } else { 16.0 };
        // the pieces in a row (two rows if they do not fit one): n of them, then the sum
        let count = n + 1;
        let across = max.x - min.x;
        // (in one, two or three lines, whichever lets them be largest)
        let cap = if lay.narrow { 8.0 } else { 15.0 };
        let (mut cell, mut per_row) = (0.0f32, count);
        for lines in 1..=3usize {
            let per = count.div_ceil(lines).max(1);
            let lines = count.div_ceil(per);
            let by_width = (across - per as f32 * sign_w) / (per as f32 * cols as f32);
            let by_height = (max.y - top - lines as f32 * 16.0) / (lines as f32 * rows as f32);
            let c = by_width.min(by_height).min(cap).floor();
            if c > cell {
                (cell, per_row) = (c, per);
            }
        }
        if cell < 4.0 {
            return;
        }
        let w = cols as f32 * cell;
        let h = rows as f32 * cell;
        for (idx, m) in pieces.iter().chain(std::iter::once(&net.dw[l])).enumerate() {
            let (rrow, rcol) = (idx / per_row, idx % per_row);
            let at = Vec2::new(min.x + rcol as f32 * (w + sign_w), top + rrow as f32 * (h + 16.0));
            hud_rect_at(f, 0, at - Vec2::splat(1.0), at + Vec2::new(w, h) + Vec2::splat(1.0), Color::hex(0x161C2B));
            for i in 0..rows {
                for j in 0..cols {
                    let v = m.at(i, j) / k;
                    let side = (cell - 1.0).max(0.5) * (v.abs().min(1.0).sqrt() as f32);
                    if side < 0.5 || !side.is_finite() {
                        continue;
                    }
                    square(f, at + Vec2::new(j as f32 + 0.5, i as f32 + 0.5) * cell, side, v > 0.0, WEIGHT, 0.2);
                }
            }
            let sign = if idx + 1 == count { "" } else if idx + 2 == count { "$=$" } else { "$+$" };
            // (at the end of a line too, as a sum broken across lines is written)
            if !sign.is_empty() {
                f.text_px(Vec2::new(at.x + w + sign_w * 0.5, at.y + h * 0.5), sign, small + 1.0, pal::DIM, Align2::CENTER_CENTER);
            }
            let name = if idx < n { format!("$i = {}$", idx + 1) } else { "sum".into() };
            if cell * cols as f32 >= 18.0 {
                f.text_px(Vec2::new(at.x + w * 0.5, at.y + h + 3.0), name, small - 1.0, pal::DIM, Align2::CENTER_TOP);
            }
        }
    }

    /// The singular values of the picked layer's weight slope, beside the largest, on a
    /// logarithmic scale.
    fn chart(&self, f: &mut Frame, lay: &Layout) {
        let Some((min, max)) = lay.chart else { return };
        let net = &self.net;
        let l = self.pick.within(net).layer();
        let small = lay.small;
        let ready = self.s >= made_at(Slot::Dw(l));
        let title_y = min.y - if lay.narrow { 38.0 } else { 44.0 };
        f.text_px(Vec2::new(min.x - 30.0, title_y), format!(r"singular values $\sigma_k$ of $\partial L/\partial W_{l}$"), small + 0.5, pal::TEXT, Align2::LEFT_TOP);
        let sv = &net.sv[l];
        let biggest = sv.first().copied().unwrap_or(0.0);
        let rank = net.rank(l);
        let note = if !ready {
            format!("worked out at step {}", made_at(Slot::Dw(l)))
        } else if biggest.is_nan() || biggest <= 0.0 {
            "all 0: the slope is 0".to_string()
        } else {
            format!("rank {rank}: those above the dashed line (at most {})", net.rank_bound(l))
        };
        fit(f, Vec2::new(min.x - 30.0, title_y + small + 6.0), &note, small, max.x - min.x + 30.0, pal::DIM, 1);
        let (lo, hi) = (-17.0f64, 0.0f64);
        let px_y = |v: f64| max.y - (max.y - min.y) * ((v - lo) / (hi - lo)) as f32;
        // the axis and its marks, the dashed line among them: below it a value counts as 0
        plot::line_of(f, plot::AXES, Vec2::new(min.x, min.y), Vec2::new(min.x, max.y), 1.0, pal::DIM.a(0.8));
        plot::line_of(f, plot::AXES, Vec2::new(min.x, max.y), Vec2::new(max.x, max.y), 1.0, pal::DIM.a(0.8));
        for (e, name) in [(0.0, "1"), (-5.0, "10^{-5}"), (-10.0, "10^{-10}"), (-15.0, "10^{-15}")] {
            let y = px_y(e);
            plot::line_of(f, plot::AXES, Vec2::new(min.x - 4.0, y), Vec2::new(min.x, y), 1.0, pal::DIM.a(0.8));
            f.text_px(Vec2::new(min.x - 6.0, y), format!("${name}$"), small - 1.5, pal::DIM, Align2::RIGHT_CENTER);
        }
        let ty = px_y(RANK_TOL.log10());
        plot::dashed_of(f, TOLERANCE, Vec2::new(min.x, ty), Vec2::new(max.x, ty), 1.0, pal::DIM.a(0.8));
        f.text_px(Vec2::new(min.x + 4.0, min.y - 3.0), r"$\sigma_k / \sigma_1$", small - 1.0, pal::DIM, Align2::LEFT_BOTTOM);
        let count = sv.len().max(1);
        let slot_w = (max.x - min.x) / count as f32;
        for (k, &s) in sv.iter().enumerate() {
            let x0 = min.x + k as f32 * slot_w + slot_w * 0.2;
            let x1 = min.x + (k + 1) as f32 * slot_w - slot_w * 0.2;
            let centre = (x0 + x1) * 0.5;
            if ready && biggest > 0.0 {
                let r = s / biggest;
                if r > 1e-17 {
                    let y = px_y(r.log10().max(lo));
                    hud_rect(f, Vec2::new(x0, y), Vec2::new(x1, max.y), WEIGHT.a(if r > RANK_TOL { 0.9 } else { 0.35 }));
                } else {
                    f.text_px(Vec2::new(centre, max.y - 3.0), "0", small - 1.0, pal::DIM, Align2::CENTER_BOTTOM);
                }
            }
            f.text_px(Vec2::new(centre, max.y + 3.0), format!("${}$", k + 1), small - 1.0, pal::DIM, Align2::CENTER_TOP);
        }
    }
}

/// One number's square, `side` pixels across round `centre`: filled for a positive
/// number, an outline (its thickness a share `edge` of the side, at least a pixel) for a
/// negative one.
fn square(f: &mut Frame, centre: Vec2, side: f32, positive: bool, color: Color, edge: f32) {
    let (lo, hi) = (centre - Vec2::splat(side * 0.5), centre + Vec2::splat(side * 0.5));
    if positive {
        hud_rect(f, lo, hi, color.a(0.92));
    } else {
        let t = (side * edge).max(1.0).min(side * 0.5);
        hud_rect(f, lo, Vec2::new(hi.x, lo.y + t), color);
        hud_rect(f, Vec2::new(lo.x, hi.y - t), hi, color);
        hud_rect(f, Vec2::new(lo.x, lo.y + t), Vec2::new(lo.x + t, hi.y - t), color);
        hud_rect(f, Vec2::new(hi.x - t, lo.y + t), Vec2::new(hi.x, hi.y - t), color);
    }
}

/// The key to the squares, its words and marks: a filled square, an outline, words.
const LEGEND: [(i8, &str); 4] = [(1, "positive"), (-1, "negative"), (0, "area: size"), (0, "empty cell: 0")];

fn legend_width(f: &Frame, size: f32) -> f32 {
    LEGEND.iter().map(|(mark, text)| f.text_size(text, size).x + if *mark != 0 { size + 4.0 } else { 0.0 } + 16.0).sum::<f32>()
}

/// The key to the squares on one line, from `at` (its left end, or with `right` its
/// right end), centred on its y.
fn legend(f: &mut Frame, at: Vec2, right: bool, size: f32) {
    let mut x = if right { at.x - legend_width(f, size) + 16.0 } else { at.x };
    for (mark, text) in LEGEND {
        let sq = size - 2.0;
        let box_at = Vec2::new(x, at.y - sq * 0.5);
        match mark {
            1 => hud_rect(f, box_at, box_at + Vec2::splat(sq), pal::DIM),
            -1 => hud_outline(f, box_at + Vec2::splat(0.5), box_at + Vec2::splat(sq - 0.5), 1.2, pal::DIM),
            _ => {}
        }
        if mark != 0 {
            x += size + 4.0;
        }
        f.text_px(Vec2::new(x, at.y), text, size, pal::DIM, Align2::LEFT_CENTER);
        x += f.text_size(text, size).x + 16.0;
    }
}

/// A line of text no wider than `width`: drawn smaller if it would be wider, and broken
/// in two (at most `lines` lines) on a narrow canvas. Returns the height it took.
fn fit(f: &mut Frame, at: Vec2, text: &str, size: f32, width: f32, color: Color, lines: usize) -> f32 {
    if text.is_empty() {
        return 0.0;
    }
    let w = f.text_size(text, size).x;
    if w <= width || lines <= 1 {
        let s = if w > width && w > 0.0 { (size * width / w).max(7.0) } else { size };
        f.text_px(at, text, s, color, Align2::LEFT_TOP);
        return s + 5.0;
    }
    // break at the space nearest the middle that is not inside a formula
    let mut best: Option<usize> = None;
    let mut inside = false;
    let mid = text.len() / 2;
    let bytes = text.as_bytes();
    let mut i = 0;
    while i < bytes.len() {
        match bytes[i] {
            b'\\' => i += 1,
            b'$' => inside = !inside,
            b' ' if !inside && best.is_none_or(|b: usize| (i as i64 - mid as i64).abs() < (b as i64 - mid as i64).abs()) => best = Some(i),
            _ => {}
        }
        i += 1;
    }
    let Some(cut) = best else {
        let s = (size * width / w).max(7.0);
        f.text_px(at, text, s, color, Align2::LEFT_TOP);
        return s + 5.0;
    };
    let (a, b) = (&text[..cut], &text[cut + 1..]);
    let wa = f.text_size(a, size).x.max(f.text_size(b, size).x);
    let s = if wa > width && wa > 0.0 { (size * width / wa).max(7.0) } else { size };
    f.text_px(at, a, s, color, Align2::LEFT_TOP);
    f.text_px(at + Vec2::new(0.0, s + 4.0), b, s, color, Align2::LEFT_TOP);
    2.0 * (s + 4.0) + 1.0
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Settings that cover the controls' ends and middles: every activation, batches of
    /// 1, 4 and 8, widths of 1, 4 and 6, the unit pushed below 0 or not.
    fn setups() -> Vec<Setup> {
        let mut out = Vec::new();
        for act in Act::ALL {
            for (batch, width) in [(1, 1), (4, 4), (8, 6), (1, 6), (8, 1), (3, 2)] {
                for shut in [false, true] {
                    out.push(Setup { batch, width, act, shut });
                }
            }
        }
        out
    }

    fn close(a: f64, b: f64, rel: f64, floor: f64) -> bool {
        (a - b).abs() <= rel * a.abs().max(b.abs()) + floor
    }

    /// Every slope the scene shows (each ∂L/∂W_l, ∂L/∂b_l and D_l, from the tape) is
    /// the slope of a difference of the same loss: the weights' and biases' by moving
    /// the model's own numbers, each D_l by moving one entry of Z_l and working the rest
    /// of the network out again by hand. (Central differences, to a part in a million,
    /// with a floor of 2e-9.)
    #[test]
    fn every_slope_shown_is_a_difference_of_the_loss() {
        let mut cornered = 0;
        for setup in setups() {
            let net = Net::new(setup);
            if net.corners() > 0 {
                // (a Z exactly 0 at ReLU's corner comes only from inputs all 0 and a bias 0)
                for l in 1..=DEPTH {
                    for i in 0..net.n() {
                        if (0..net.units[l]).any(|k| net.z[l].at(i, k) == 0.0) {
                            assert!((0..net.units[l - 1]).all(|k| net.a[l - 1].at(i, k) == 0.0), "{setup:?}");
                        }
                    }
                }
                cornered += 1;
                continue;
            }
            // (a difference of a loss near 1 with steps of a millionth is good to about
            // 1e-10 for rounding: the floor; the tables show four places)
            let floor = 2e-9;
            for l in 1..=TOP {
                let (rows, cols) = (net.units[l], net.units[l - 1]);
                for j in 0..rows {
                    for k in 0..cols {
                        let p = Pick::Weight(l, j, k);
                        let (tape, fd) = (net.dw[l].at(j, k), net.by_difference(p));
                        assert!(close(tape, fd, 1e-6, floor), "{setup:?}: ∂L/∂W{l}[{j},{k}] tape {tape} differences {fd}");
                    }
                    let (tape, fd) = (net.db[l].at(0, j), net.by_difference(Pick::Bias(l, j)));
                    assert!(close(tape, fd, 1e-6, floor), "{setup:?}: ∂L/∂b{l}[{j}] tape {tape} differences {fd}");
                }
                for i in 0..net.n() {
                    for k in 0..rows {
                        let (tape, fd) = (net.d[l].at(i, k), net.by_difference(Pick::Back(l, i, k)));
                        assert!(close(tape, fd, 1e-6, floor), "{setup:?}: D{l}[{i},{k}] tape {tape} differences {fd}");
                    }
                }
            }
        }
        assert!(cornered <= 8, "{cornered} setups at a corner");
        // the scene's own settings never sit on a corner unless a unit is pushed below 0
        // or a layer is one unit wide
        for batch in 1..=MOST_BATCH {
            for width in 2..=MOST_WIDTH {
                for act in [Act::Relu, Act::LeakyRelu] {
                    assert_eq!(Net::new(Setup { batch, width, act, shut: false }).corners(), 0, "{batch} {width} {act:?}");
                }
            }
        }
    }

    /// What a hidden layer's D shows for the step before it is D, D_(l+1) W_(l+1), is
    /// the slope of the loss with respect to A_l: against differences of the loss with
    /// one entry of A_l moved and the rest worked out by hand.
    #[test]
    fn the_slope_with_respect_to_a_layers_output_is_a_difference_of_the_loss() {
        for setup in setups() {
            let net = Net::new(setup);
            if net.corners() > 0 {
                continue;
            }
            for l in 1..=DEPTH {
                let a = &net.a[l];
                let (w, b) = (&net.w[l + 1], net.model.group(&names(l + 1).1).map(|g| g.value.clone()).unwrap_or_default());
                let loss_with = |a: &Mat| {
                    let mut z = Mat::zeros(a.rows, w.rows);
                    for i in 0..a.rows {
                        for j in 0..w.rows {
                            z.v[i * w.rows + j] = (0..w.cols).map(|k| a.at(i, k) * w.at(j, k)).sum::<f64>() + b[j];
                        }
                    }
                    net.loss_from_z(l + 1, &z)
                };
                for i in 0..a.rows {
                    for k in 0..a.cols {
                        let h = 1e-6 * a.at(i, k).abs().max(1.0);
                        let moved = |by: f64| {
                            let mut m = a.clone();
                            m.v[i * a.cols + k] += by;
                            loss_with(&m)
                        };
                        let fd = (moved(h) - moved(-h)) / (2.0 * h);
                        let tape = net.back[l].at(i, k);
                        assert!(close(tape, fd, 1e-6, 2e-9), "{setup:?}: D{} W{} [{i},{k}] tape {tape} differences {fd}", l + 1, l + 1);
                    }
                }
            }
        }
    }

    /// The formulas the scene writes, held against the tape's numbers: the forward pass
    /// by hand, D_3 = (P − Y)/n, D_l = (D_(l+1) W_(l+1)) ⊙ φ'(Z_l), ∂L/∂W_l = D_lᵀ A_(l−1)
    /// as the sum of the examples' outer products, ∂L/∂b_l the column sums of D_l.
    #[test]
    fn the_blocks_are_what_the_formulas_say() {
        for setup in setups() {
            let net = Net::new(setup);
            let n = net.n();
            let tol = |v: f64| 1e-12 * (1.0 + v.abs());
            // the forward pass by hand gives the tape's loss, from the data's Z₁ on
            assert!(close(net.loss_from_z(1, &net.z[1]), net.loss, 1e-12, 1e-12), "{setup:?}");
            for l in 1..=TOP {
                let (rows, cols) = (net.units[l], net.units[l - 1]);
                let b = net.model.group(&names(l).1).map(|g| g.value.clone()).unwrap_or_default();
                for i in 0..n {
                    for j in 0..rows {
                        let z: f64 = (0..cols).map(|k| net.a[l - 1].at(i, k) * net.w[l].at(j, k)).sum::<f64>() + b[j];
                        assert!((z - net.z[l].at(i, j)).abs() <= tol(z), "{setup:?}: Z{l}");
                        if l < TOP {
                            assert!((setup.act.f(z) - net.a[l].at(i, j)).abs() <= tol(z), "{setup:?}: A{l}");
                        }
                    }
                }
                // the slope of the weights: the sum of one outer product per example
                let mut sum = Mat::zeros(rows, cols);
                for i in 0..n {
                    let o = net.outer(l, i);
                    for (s, v) in sum.v.iter_mut().zip(&o.v) {
                        *s += v;
                    }
                }
                for (a, b) in sum.v.iter().zip(&net.dw[l].v) {
                    assert!((a - b).abs() <= 1e-12 * (1.0 + net.dw[l].largest()), "{setup:?}: ∂L/∂W{l}");
                }
                for j in 0..rows {
                    let column: f64 = (0..n).map(|i| net.d[l].at(i, j)).sum();
                    assert!((column - net.db[l].at(0, j)).abs() <= 1e-12, "{setup:?}: ∂L/∂b{l}");
                }
            }
            for i in 0..n {
                for k in 0..CLASSES {
                    let want = (net.a[TOP].at(i, k) - net.y.at(i, k)) / n as f64;
                    assert!((want - net.d[TOP].at(i, k)).abs() <= 1e-12, "{setup:?}: D3");
                }
                for l in 1..=DEPTH {
                    for k in 0..net.units[l] {
                        let back: f64 = (0..net.units[l + 1]).map(|j| net.d[l + 1].at(i, j) * net.w[l + 1].at(j, k)).sum();
                        assert!((back - net.back[l].at(i, k)).abs() <= 1e-12, "{setup:?}: D{} W{}", l + 1, l + 1);
                        let d = back * net.gate[l].at(i, k);
                        assert!((d - net.d[l].at(i, k)).abs() <= 1e-12, "{setup:?}: D{l}");
                    }
                }
            }
            // every number finite, and each row of P − Y adding up to 0
            for l in 0..=TOP {
                for m in [&net.a[l], &net.z[l], &net.d[l], &net.dw[l], &net.db[l]] {
                    assert!(m.v.iter().all(|v| v.is_finite()), "{setup:?}: layer {l}");
                }
            }
            for i in 0..n {
                assert!((0..CLASSES).map(|k| net.d[TOP].at(i, k)).sum::<f64>().abs() < 1e-15);
            }
        }
    }

    /// The rank of a weight's slope is at most the batch size and its smaller side, and
    /// the read-out's is at most 2 (each row of P − Y adds up to 0); with tanh, nothing
    /// shut, it is exactly the smallest of those.
    #[test]
    fn a_weights_slope_has_rank_at_most_the_batch() {
        for batch in 1..=MOST_BATCH {
            for width in 1..=MOST_WIDTH {
                for act in [Act::Tanh, Act::Relu] {
                    let net = Net::new(Setup { batch, width, act, shut: false });
                    for l in 1..=TOP {
                        assert!(net.rank(l) <= net.rank_bound(l), "{batch} {width} {act:?} layer {l}");
                    }
                    assert!(net.rank(TOP) <= 2);
                    if act == Act::Tanh {
                        assert_eq!(net.rank(1), batch.min(width).min(INPUTS), "{batch} {width} layer 1");
                        assert_eq!(net.rank(2), batch.min(width), "{batch} {width} layer 2");
                        assert_eq!(net.rank(TOP), batch.min(width).min(2), "{batch} {width} layer 3");
                    }
                }
            }
        }
    }

    /// The film: every block is filled by the end, none before its step, a hidden
    /// layer's D shows D_(l+1) W_(l+1) the step before it becomes D_l, and each step's
    /// main block is one it fills.
    #[test]
    fn the_film_fills_each_block_at_its_step() {
        let net = Net::new(Setup::default());
        for slot in SLOTS {
            assert!(block_values(&net, slot, STEPS).is_some(), "{slot:?} at the end");
            let at = made_at(slot);
            assert!(at <= STEPS);
            if at > 0 {
                let early = block_values(&net, slot, at - 1);
                match slot {
                    Slot::Back(l) if l <= DEPTH => assert_eq!(early, Some(&net.back[l])),
                    _ => assert!(early.is_none(), "{slot:?}"),
                }
            }
        }
        for s in 1..=STEPS {
            let (out, ins) = step_blocks(s);
            assert!(block_values(&net, out, s).is_some() && block_values(&net, out, s - 1) != block_values(&net, out, s), "step {s}");
            for i in ins {
                assert!(block_values(&net, i, s - 1).is_some(), "step {s} uses {i:?} before it is there");
            }
            let (a, b) = step_lines(&net, s, false);
            assert!(!a.is_empty() && !b.is_empty());
        }
        // played at one step a second, it gets to the end in eleven seconds, and stops
        let mut scene = BackpropMat::new();
        scene.set_up(0);
        assert!(scene.s == 0 && scene.playing);
        let mut t = 0.0;
        while scene.playing {
            scene.advance(1.0 / 60.0);
            t += 1.0 / 60.0;
            assert!(t < 20.0);
        }
        assert_eq!(scene.s, STEPS);
        assert!((t - STEPS as f32).abs() < 0.1, "{t}");
    }

    /// The scene as an experiment's button leaves it.
    fn trial(i: usize) -> BackpropMat {
        let mut scene = BackpropMat::new();
        scene.set_up(i);
        scene
    }

    #[test]
    fn forwards_then_back_does_what_its_caption_says() {
        let scene = trial(0);
        assert_eq!(scene.setup, Setup { batch: 4, width: 4, act: Act::Relu, shut: false });
        assert_eq!(scene.net.work(), (144, 256));
        // twice a layer the sizes of the forward product, but for D₁ W₁
        let n = scene.net.n();
        let u = scene.net.units;
        assert_eq!(scene.net.work().1, 2 * 144 - n * u[1] * u[0]);
        assert!(TRIALS[0].look_for.contains("256 multiply-adds against 144"));
        // the steps: 3 forward, then the loss's D, then each layer's two moves
        assert_eq!((1..=3).map(step_blocks).map(|b| b.0).collect::<Vec<_>>(), vec![Slot::Out(1), Slot::Out(2), Slot::Out(3)]);
        assert_eq!(step_blocks(4).0, Slot::Back(3));
    }

    #[test]
    fn one_example_does_what_its_caption_says() {
        let scene = trial(1);
        let net = &scene.net;
        assert_eq!((net.n(), net.units[1], net.setup.act), (1, 6, Act::Tanh));
        for l in 1..=TOP {
            assert_eq!(net.rank(l), 1, "layer {l}");
        }
        // of ∂L/∂W₂'s six singular values, one is not 0 and five are below 1e-10 of it
        let sv = &net.sv[2];
        assert_eq!(sv.len(), 6);
        assert!(sv[0] > 0.0 && sv[1..].iter().all(|s| *s < RANK_TOL * sv[0]), "{sv:?}");
        // every row is a multiple of the example's row of A₁
        let a = &net.a[1];
        for j in 0..6 {
            for k in 0..6 {
                assert!((net.dw[2].at(j, k) - net.d[2].at(0, j) * a.at(0, k)).abs() < 1e-15);
            }
        }
        // with tanh nothing is shut: every entry of the slope is there to see
        assert!(net.dw[2].v.iter().all(|v| v.abs() > 1e-4), "{:?}", net.dw[2].v);
    }

    #[test]
    fn a_batch_does_what_its_caption_says() {
        let scene = trial(2);
        let net = &scene.net;
        assert_eq!((net.n(), net.units[1], net.setup.act), (3, 6, Act::Tanh));
        assert_eq!(net.sv[2].len(), 6);
        assert_eq!(net.rank(2), 3);
        assert_eq!(net.rank(TOP), 2);
        assert_eq!(net.rank_bound(TOP), 3);
        // each of the three outer products has rank 1
        for i in 0..3 {
            let o = net.outer(2, i);
            let sv = linalg::svd(&o.v, 6, 6).1;
            assert!(sv[0] > 0.0 && sv[1] < RANK_TOL * sv[0], "{sv:?}");
        }
    }

    #[test]
    fn a_shut_unit_does_what_its_caption_says() {
        let scene = trial(3);
        let net = &scene.net;
        assert_eq!((net.n(), net.units[1], net.setup.act, net.setup.shut), (4, 4, Act::Relu, true));
        let u = shut_unit(4);
        assert_eq!(u, 1, "unit 2");
        for i in 0..4 {
            assert!(net.z[1].at(i, u) <= -1.0 + 1e-12, "example {i}: {}", net.z[1].at(i, u));
            assert_eq!(net.a[1].at(i, u), 0.0);
            assert_eq!(net.gate[1].at(i, u), 0.0);
            assert_eq!(net.d[1].at(i, u), 0.0);
        }
        // something did reach it from above
        assert!((0..4).any(|i| net.back[1].at(i, u).abs() > 1e-3));
        for k in 0..INPUTS {
            assert_eq!(net.dw[1].at(u, k), 0.0);
        }
        assert_eq!(net.db[1].at(0, u), 0.0);
        for j in 0..4 {
            assert_eq!(net.dw[2].at(j, u), 0.0);
        }
        // and no other unit of either layer is shut for all four
        for l in 1..=DEPTH {
            for k in 0..4 {
                if !(l == 1 && k == u) {
                    assert!((0..4).any(|i| net.gate[l].at(i, k) > 0.0), "layer {l} unit {k}");
                }
            }
        }
        // the bias is low enough for every one of the eight points
        let all = Net::new(Setup { batch: MOST_BATCH, ..net.setup });
        assert!((0..MOST_BATCH).all(|i| all.z[1].at(i, u) <= -1.0 + 1e-12));
    }

    #[test]
    fn the_same_weights_do_what_their_caption_says() {
        let mut scene = trial(4);
        assert_eq!((scene.s, scene.pick), (6, Pick::Back(2, 0, 2)));
        let net = &scene.net;
        // forward, unit 3 of layer 2 reaches read-out unit j through (W₃)_j3
        let b3 = net.model.group(&names(TOP).1).map(|g| g.value.clone()).unwrap_or_default();
        for j in 0..CLASSES {
            let z: f64 = (0..4).map(|k| net.a[2].at(0, k) * net.w[TOP].at(j, k)).sum::<f64>() + b3[j];
            assert!((z - net.z[TOP].at(0, j)).abs() < 1e-12);
        }
        // back, its slope comes along the same column: three products
        let three: Vec<f64> = (0..CLASSES).map(|j| net.d[TOP].at(0, j) * net.w[TOP].at(j, 2)).collect();
        assert!((three.iter().sum::<f64>() - net.back[2].at(0, 2)).abs() < 1e-12);
        // what the D₂ slot shows at this step is D₃ W₃; the next step gates it
        assert_eq!(block_values(net, Slot::Back(2), 6), Some(&net.back[2]));
        assert_eq!(net.gate[2].at(0, 2), 1.0);
        scene.next();
        assert_eq!(block_values(&scene.net, Slot::Back(2), scene.s), Some(&scene.net.d[2]));
    }

    /// The pick is kept inside the network as its sizes change, and its check by
    /// differences agrees with it.
    #[test]
    fn the_pick_follows_the_sizes() {
        let mut scene = BackpropMat::new();
        scene.pick = Pick::Weight(2, 5, 5);
        scene.set_setup(Setup { width: 2, ..Setup::default() });
        assert_eq!(scene.pick, Pick::Weight(2, 1, 1));
        scene.pick = Pick::Back(2, 7, 3);
        scene.set_setup(Setup { batch: 1, width: 3, ..Setup::default() });
        assert_eq!(scene.pick, Pick::Back(2, 0, 2));
        let v = scene.pick.value(&scene.net);
        let fd = scene.checked();
        assert!(close(v, fd, 1e-6, 1e-12), "{v} {fd}");
    }

    /// A glide comes exactly to rest, at every display rate.
    #[test]
    fn a_glide_comes_to_rest() {
        for hz in [30.0, 60.0, 120.0, 240.0] {
            let mut g = Follow::default();
            g.snap(&[0.0, 300.0]);
            let mut frames = 0;
            while g.follow(&[1.0, 40.0], 1.0 / hz, VALUES_EASE) {
                frames += 1;
                assert!(frames < 2000);
            }
            assert_eq!(g.shown, vec![1.0, 40.0]);
        }
    }

    /// The numbers the tables write: four places, an exact 0 as "0", tiny ones in powers.
    #[test]
    fn numbers_are_written_plainly() {
        assert_eq!(four(0.0), "0");
        assert_eq!(four(-0.24361), "−0.2436");
        assert_eq!(four(8.2e-6), r"8.2 \times 10^{-6}");
        assert_eq!(four(-9.97e-6), r"-1.0 \times 10^{-5}");
        assert_eq!(four(f64::NAN), "?");
    }
}

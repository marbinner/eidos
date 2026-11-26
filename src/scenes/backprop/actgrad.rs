//! Activation functions as gates.
//!
//! On the way back through a network, a unit multiplies the gradient arriving at its
//! output by the slope of its activation at the input it had on the way forward: the
//! slope is the unit's gate. What the gate can be (at most a quarter, exactly 0 or 1, a
//! little below 0) decides how much of the gradient gets through a deep network. The
//! scene shows it four ways:
//!
//! - the activation and its slope over one axis, with the inputs a unit receives drawn
//!   as a bell whose centre and width are handles;
//! - the gates those inputs meet, as a histogram, with the average gate and the average
//!   of its square (the factor on the gradient's mean square), worked out by
//!   integrating against the bell: Simpson's rule, broken at the kink at 0, and the
//!   shares from the normal distribution function between the places where the slope
//!   crosses 0.01 and 0 (found once for each activation);
//! - ten layers built with `math::arch`, the same weights for every activation, with
//!   the root mean square of the gradient at each layer, pooled over a few starts;
//! - a small network learning the spirals by plain gradient descent, with every unit's
//!   gate for every point and the share of its units that are shut or dead.

use super::*;
use crate::clock::Pace;
use crate::math::arch::{self, Init, Kind, Loss, MlpSpec, Model, Norm, Part, Place, Stage, Trace};
use crate::math::autograd::{Act, Tape};
use crate::math::stats::normal_cdf;
use crate::scenes::stats::plot::{self, Glide, Plot, Rect};

pub const DEF: SceneDef = SceneDef {
    id: "actgrad",
    title: "Activation functions as gates",
    group: GROUP,
    blurb: "Each unit lets the gradient through in proportion to its slope: saturated tanh, dead ReLUs, and what the smooth modern ones change.",
    make: |_| Box::new(ActGrad::create()),
};

// ----- the gate, averaged over a bell of inputs --------------------------------------

/// A gate smaller than this counts as shut, as `arch::Probe` counts it, so that the
/// shares of the bell and of the training network are the same thing. (Leaky ReLU's
/// 0.01 is not smaller.)
const SHUT: f64 = 0.01;

/// The axis of the activation's two charts.
const Z_RANGE: [f64; 2] = [-6.0, 6.0];
/// The bell's centre goes from −5 to 5 and its width from 0 (every input the same) to 4.
const MU_MOST: f64 = 5.0;
const SIGMA_MOST: f64 = 4.0;

/// The places where something holds of the input z, as intervals (an end may be
/// infinite): found by looking along z from −60 to 60 in steps of 1/64 for where it
/// starts or stops holding, each such place then pinned down by bisection. (None of the
/// slopes here changes its mind twice within 1/64, and beyond ±60 each is constant to
/// the last digit, so what holds at an end holds out to infinity.)
fn intervals(holds: impl Fn(f64) -> bool) -> Vec<(f64, f64)> {
    const REACH: f64 = 60.0;
    const STEP: f64 = 1.0 / 64.0;
    let n = (2.0 * REACH / STEP) as usize;
    let at = |k: usize| -REACH + k as f64 * STEP;
    // the place between a and b where `holds` changes from `was`
    let edge = |mut a: f64, mut b: f64, was: bool| {
        for _ in 0..60 {
            let m = 0.5 * (a + b);
            if holds(m) == was {
                a = m;
            } else {
                b = m;
            }
        }
        0.5 * (a + b)
    };
    let mut out = Vec::new();
    let mut was = holds(at(0));
    let mut start = was.then_some(f64::NEG_INFINITY);
    for k in 1..=n {
        let now = holds(at(k));
        if now != was {
            let e = edge(at(k - 1), at(k), was);
            match start.take() {
                Some(s) => out.push((s, e)),
                None => start = Some(e),
            }
            was = now;
        }
    }
    if let Some(s) = start {
        out.push((s, f64::INFINITY));
    }
    out
}

/// Where an activation's gate is shut, and where it is below 0 (the gradient then comes
/// back with its sign turned round).
#[derive(Clone, Debug)]
struct Sets {
    shut: Vec<(f64, f64)>,
    negative: Vec<(f64, f64)>,
}

impl Sets {
    fn of(act: Act) -> Sets {
        Sets { shut: intervals(|z| act.df(z).abs() < SHUT), negative: intervals(|z| act.df(z) < 0.0) }
    }
}

/// A bell too narrow to spread its inputs: every one of them is at its centre (and a
/// width that is not a number is taken as that too).
fn no_width(sigma: f64) -> bool {
    sigma.is_nan() || sigma <= 1e-9
}

/// The share of a bell of centre `mu` and width `sigma` that falls in the intervals.
/// (A bell of no width is all at its centre: `at_centre` says whether that is in them.)
fn share(set: &[(f64, f64)], mu: f64, sigma: f64, at_centre: bool) -> f64 {
    if no_width(sigma) {
        return if at_centre { 1.0 } else { 0.0 };
    }
    let cdf = |z: f64| {
        if z == f64::INFINITY {
            1.0
        } else if z == f64::NEG_INFINITY {
            0.0
        } else {
            normal_cdf((z - mu) / sigma)
        }
    };
    set.iter().map(|&(a, b)| cdf(b) - cdf(a)).sum::<f64>().max(0.0).min(1.0)
}

/// What a unit's gate does to inputs spread like a bell: averages over the bell.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
struct Averages {
    /// The average gate E f'(z), and the average of its square E f'(z)²: the factor by
    /// which the gates multiply the gradient's mean square.
    gate: f64,
    gate_sq: f64,
    /// The share of inputs whose gate is shut, and whose gate is below 0.
    shut: f64,
    negative: f64,
    /// The average output E f(z): the forward side.
    output: f64,
}

/// How far out the averages go, in widths of the bell: beyond 9, less than 1e-18 of it.
const REACH_T: f64 = 9.0;
/// Simpson's intervals over the whole reach.
const SIMPSON: f64 = 1200.0;

/// The averages over the bell z = μ + σt, t standard normal. E f'(z), E f'(z)² and E f(z)
/// by Simpson's rule in t from −9 to 9, broken where z = 0: there the slopes of ReLU,
/// leaky ReLU and ELU have their kink (ReLU's jumps), each side is taken with its own
/// slope, and the rule keeps its order on both. The shares from the intervals.
fn averages(act: Act, sets: &Sets, mu: f64, sigma: f64) -> Averages {
    let shares = |a: &mut Averages| {
        a.shut = share(&sets.shut, mu, sigma, act.df(mu).abs() < SHUT);
        a.negative = share(&sets.negative, mu, sigma, act.df(mu) < 0.0);
    };
    if no_width(sigma) || !mu.is_finite() {
        let g = act.df(mu);
        let mut a = Averages { gate: g, gate_sq: g * g, output: act.f(mu), ..Default::default() };
        shares(&mut a);
        return a;
    }
    let kink = -mu / sigma;
    let mut cuts = vec![-REACH_T];
    if kink > -REACH_T && kink < REACH_T {
        cuts.push(kink);
    }
    cuts.push(REACH_T);
    let density = |t: f64| (-0.5 * t * t).exp() / std::f64::consts::TAU.sqrt();
    let mut total = [0.0f64; 3];
    for piece in cuts.windows(2) {
        let (a, b) = (piece[0], piece[1]);
        // each side of the kink with its own slope, its ends too
        let below = b <= kink;
        let side = |z: f64| if below { z.min(0.0) } else if a >= kink && kink > -REACH_T { z.max(f64::MIN_POSITIVE) } else { z };
        let n = ((((b - a) / (2.0 * REACH_T)) * SIMPSON).ceil() as usize).max(4).div_ceil(2) * 2;
        let h = (b - a) / n as f64;
        for i in 0..=n {
            let t = a + i as f64 * h;
            let z = side(mu + sigma * t);
            let weight = if i == 0 || i == n {
                1.0
            } else if i % 2 == 1 {
                4.0
            } else {
                2.0
            };
            let w = density(t) * weight * h / 3.0;
            let g = act.df(z);
            total[0] += w * g;
            total[1] += w * g * g;
            total[2] += w * act.f(z);
        }
    }
    let mut out = Averages { gate: total[0], gate_sq: total[1], output: total[2], ..Default::default() };
    shares(&mut out);
    out
}

/// The histogram's bins: 0.05 wide, centred on −0.2, −0.15, ..., 1.2 (every slope here
/// is inside; one outside would be counted in the end bin).
const BIN: f64 = 0.05;
const BIN_FIRST: f64 = -0.2;
const BINS: usize = 29;

fn bin_of(g: f64) -> usize {
    (((g - BIN_FIRST) / BIN).round().max(0.0) as usize).min(BINS - 1)
}

/// The standard normal's weight in each of the cells of t from −8.5 to 8.5, 0.005 wide:
/// the same for every bell, so worked out once.
struct Cells {
    mid: Vec<f64>,
    weight: Vec<f64>,
}

impl Cells {
    const REACH: f64 = 8.5;
    const N: usize = 3400;

    fn new() -> Cells {
        let h = 2.0 * Self::REACH / Self::N as f64;
        let edge = |i: usize| -Self::REACH + i as f64 * h;
        let cdf: Vec<f64> = (0..=Self::N).map(|i| normal_cdf(edge(i))).collect();
        Cells { mid: (0..Self::N).map(|i| edge(i) + 0.5 * h).collect(), weight: cdf.windows(2).map(|w| w[1] - w[0]).collect() }
    }
}

/// The share of inputs whose gate falls in each bin, split into the gates that are
/// open and those that are shut: each cell of the bell goes to the bin of the gate at
/// its middle. (Good to the weight of a cell, a few thousandths, at a kink: for the
/// eye. The numbers written are `averages`'.)
#[derive(Clone, Debug, PartialEq)]
struct Histogram {
    open: [f64; BINS],
    shut: [f64; BINS],
}

fn histogram(act: Act, mu: f64, sigma: f64, cells: &Cells) -> Histogram {
    let mut h = Histogram { open: [0.0; BINS], shut: [0.0; BINS] };
    let mut put = |g: f64, w: f64| {
        if g.is_finite() {
            let b = bin_of(g);
            if g.abs() < SHUT {
                h.shut[b] += w;
            } else {
                h.open[b] += w;
            }
        }
    };
    if no_width(sigma) {
        put(act.df(mu), 1.0);
        return h;
    }
    for (&t, &w) in cells.mid.iter().zip(&cells.weight) {
        put(act.df(mu + sigma * t), w);
    }
    h
}

// ----- the same ten layers, with each activation --------------------------------------

const STACK_DEPTH: usize = 10;
const STACK_WIDTH: usize = 32;
const STACK_BATCH: usize = 32;
const STACK_STARTS: u64 = 4;

/// One activation in the ten layers: the root mean square of the gradient with respect
/// to each layer's inputs z (what reaches its weights), of z itself, and the mean of
/// f'(z)², each over every example, unit and start.
#[derive(Clone, Copy, Debug, Default)]
struct Profile {
    grad: [f64; STACK_DEPTH],
    z: [f64; STACK_DEPTH],
    gate_sq: [f64; STACK_DEPTH],
}

/// Ten plain layers of 32 units, weights drawn with variance 1/32 (one over the number
/// of inputs: on their own they keep the gradient's mean square, on average), biases 0,
/// on 32 examples of 32 standard normal numbers. The loss hands the top layer's outputs
/// a gradient of standard normal numbers scaled to a root mean square of exactly 1. The
/// weights, the examples and that gradient are the same for every activation (the same
/// draws in the same order); four starts are pooled.
fn profile(act: Act) -> Profile {
    let spec = MlpSpec {
        inputs: STACK_WIDTH,
        width: STACK_WIDTH,
        depth: STACK_DEPTH,
        outputs: 1,
        act,
        norm: Norm::None,
        place: Place::Pre,
        residual: false,
        residual_scale: 1.0,
        init: Init::Lecun,
        gain: 1.0,
        bias: true,
    };
    let mut tape = Tape::new();
    let mut sums = [[0.0f64; 3]; STACK_DEPTH];
    let mut count = 0usize;
    let n = STACK_BATCH * STACK_WIDTH;
    for start in 0..STACK_STARTS {
        let mut rng = Rng::new(0x5_7AC4 + start);
        let model = Model::mlp(&spec, &mut rng);
        let x: Vec<f64> = (0..n).map(|_| rng.normal() as f64).collect();
        let mut seed: Vec<f64> = (0..n).map(|_| rng.normal() as f64).collect();
        let size = (seed.iter().map(|s| s * s).sum::<f64>() / n as f64).sqrt();
        seed.iter_mut().for_each(|s| *s /= size.max(1e-12));
        let trace = model.forward_mlp(&mut tape, &x);
        let Some(top) = trace.get(Stage::Output, STACK_DEPTH) else { continue };
        tape.backward_seeded(top, &seed);
        for (l, sum) in sums.iter_mut().enumerate() {
            let Some(z) = trace.get(Stage::PreAct, l + 1) else { continue };
            for (&g, &v) in tape.grad(z).iter().zip(tape.value(z)) {
                let s = act.df(v);
                sum[0] += g * g;
                sum[1] += v * v;
                sum[2] += s * s;
            }
        }
        count += n;
    }
    let mut p = Profile::default();
    let c = count.max(1) as f64;
    for (l, sum) in sums.iter().enumerate() {
        p.grad[l] = (sum[0] / c).sqrt();
        p.z[l] = (sum[1] / c).sqrt();
        p.gate_sq[l] = sum[2] / c;
    }
    p
}

/// Where an activation is in `Act::ALL`.
fn index_of(act: Act) -> usize {
    Act::ALL.iter().position(|&a| a == act).unwrap_or(0)
}

// ----- a network that learns -----------------------------------------------------------

/// Points of the spirals, hidden units in each of the hidden layers, hidden layers.
const POINTS: usize = 150;
const UNITS: usize = 16;
const LAYERS: usize = 3;
const ALL_UNITS: usize = UNITS * LAYERS;
/// A run is this many steps, taken at this many a second of the scene's time.
const STEPS: usize = 1500;
const STEPS_PER_SECOND: f32 = 50.0;
/// The start every run is drawn from, and the spirals' own.
const RUN_SEED: u64 = 1;
const DATA_SEED: u64 = 7;

/// A network of three layers of 16 units learning to tell the three arms of the
/// spirals apart (He's starting weights, cross-entropy, every point in every step), by
/// plain gradient descent: each weight moves by the rate times its gradient, so that the
/// size of the gradient is the size of the step.
struct Run {
    act: Act,
    rate: f64,
    bias: f64,
    model: Model,
    tape: Tape,
    trace: Trace,
    x: Vec<f64>,
    y: Vec<f64>,
    /// The points in the order the map shows them: arm by arm, from the centre out.
    order: Vec<usize>,
    /// Steps taken.
    step: usize,
    /// Where the activation's gate is shut (see `Sets`).
    shut_set: Vec<(f64, f64)>,
    /// At the present weights: the loss, the share of points classified right, which
    /// units are dead (shut for every point), and the share of all gates that are shut.
    loss: f64,
    right: f64,
    /// Whether every point gets the same output (then none is classified right: a tie).
    same: bool,
    dead: [bool; ALL_UNITS],
    shut: f64,
    /// Over the steps: the share of points right, of units dead, of gates shut.
    rights: History,
    deads: History,
    shuts: History,
}

impl Run {
    fn new(act: Act, rate: f64, bias: f64) -> Run {
        Run::seeded(act, rate, bias, RUN_SEED)
    }

    fn seeded(act: Act, rate: f64, bias: f64, seed: u64) -> Run {
        let (x, y) = arch::spirals(POINTS, 3, &mut Rng::new(DATA_SEED));
        let spec = MlpSpec { inputs: 2, width: UNITS, depth: LAYERS, outputs: 3, act, init: Init::He, ..MlpSpec::default() };
        let mut model = Model::mlp(&spec, &mut Rng::new(seed));
        for g in model.groups_mut() {
            if g.kind == Kind::Bias && g.part == Part::Layer {
                g.value.iter_mut().for_each(|b| *b = bias);
            }
        }
        let mut order: Vec<usize> = (0..POINTS).collect();
        let radius = |i: usize| x[2 * i].hypot(x[2 * i + 1]);
        order.sort_by(|&a, &b| y[a].total_cmp(&y[b]).then(radius(a).total_cmp(&radius(b))));
        let mut tape = Tape::new();
        let trace = model.forward_mlp(&mut tape, &[]);
        let mut run = Run {
            act,
            rate,
            bias,
            model,
            tape,
            trace,
            x,
            y,
            order,
            step: 0,
            shut_set: Sets::of(act).shut,
            loss: f64::NAN,
            right: 0.0,
            same: false,
            dead: [false; ALL_UNITS],
            shut: 0.0,
            rights: History::new(300),
            deads: History::new(300),
            shuts: History::new(300),
        };
        run.measure();
        run
    }

    /// A forward and a backward pass at the present weights, and what they show.
    fn measure(&mut self) {
        self.trace = self.model.loss_mlp(&mut self.tape, &self.x, &self.y, Loss::CrossEntropy);
        self.tape.backward(self.trace.loss);
        self.loss = self.tape.value(self.trace.loss).first().copied().unwrap_or(f64::NAN);
        let out = self.tape.value(self.trace.output);
        self.right = arch::share_right(out, 3, &self.y);
        self.same = out.len() >= 3 && out.chunks_exact(3).all(|row| row.iter().zip(&out[..3]).all(|(a, b)| (a - b).abs() <= 1e-12 * (1.0 + b.abs())));
        let mut shut = 0usize;
        for l in 0..LAYERS {
            let Some(z) = self.trace.get(Stage::PreAct, l + 1) else { continue };
            let z = self.tape.value(z);
            if z.len() != POINTS * UNITS {
                continue;
            }
            // (shut is told by where z is, which costs a comparison or two: the slope
            // itself is a library call for the smooth activations)
            let set = &self.shut_set;
            let closed = |v: f64| set.iter().any(|&(a, b)| v >= a && v <= b);
            let mut open = [false; UNITS];
            for row in z.chunks_exact(UNITS) {
                for (o, &v) in open.iter_mut().zip(row) {
                    let c = closed(v);
                    *o |= !c;
                    shut += c as usize;
                }
            }
            for (j, o) in open.iter().enumerate() {
                self.dead[l * UNITS + j] = !o;
            }
        }
        self.shut = shut as f64 / (ALL_UNITS * POINTS) as f64;
        self.rights.push(self.right as f32);
        self.deads.push(self.dead_count() as f32 / ALL_UNITS as f32);
        self.shuts.push(self.shut as f32);
    }

    /// One step down the gradient of the last pass, and a pass at the new weights.
    fn step(&mut self) {
        if self.finished() {
            return;
        }
        self.model.zero_grads();
        self.model.accumulate(&self.tape, &self.trace);
        self.model.step_sgd(self.rate);
        self.step += 1;
        self.measure();
    }

    fn finished(&self) -> bool {
        self.step >= STEPS
    }

    fn dead_count(&self) -> usize {
        self.dead.iter().filter(|&&d| d).count()
    }

    fn dead_in(&self, layer: usize) -> usize {
        self.dead[layer * UNITS..(layer + 1) * UNITS].iter().filter(|&&d| d).count()
    }

    /// Every unit's gate for every point at the present weights: unit by unit, the
    /// points in the map's order.
    fn gates(&self, out: &mut Vec<f32>) {
        out.clear();
        out.resize(ALL_UNITS * POINTS, f32::NAN);
        for l in 0..LAYERS {
            let Some(z) = self.trace.get(Stage::PreAct, l + 1) else { continue };
            let z = self.tape.value(z);
            if z.len() != POINTS * UNITS {
                continue;
            }
            for j in 0..UNITS {
                let u = l * UNITS + j;
                for (k, &i) in self.order.iter().enumerate() {
                    out[u * POINTS + k] = self.act.df(z[i * UNITS + j]) as f32;
                }
            }
        }
    }
}

// ----- the experiments -----------------------------------------------------------------

/// Which of the two lower pictures a canvas too small for all four shows.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Lower {
    Stack,
    Training,
}

struct Trial {
    button: &'static str,
    caption: &'static str,
    act: Act,
    /// The bell.
    mu: f64,
    sigma: f64,
    /// The run: its learning rate, and the bias every hidden unit starts with.
    rate: f64,
    bias: f64,
    lower: Lower,
}

/// The learning rate and the starting bias of a run, unless an experiment says otherwise.
const RATE: f64 = 0.3;
const BIAS: f64 = 0.0;

const TRIALS: [Trial; 5] = [
    Trial {
        button: "sigmoid passes at most a quarter",
        caption: r"The sigmoid's slope is at most $1/4$ (at $z = 0$), so its gate never passes more than a quarter of the gradient. For this bell $\mathbb{E}\,f'(z)^2 = 0.045$, so the gates of a layer whose inputs are spread like this divide the gradient's mean square by about 22; in the ten layers $4.5 \times 10^{-7}$ of it reaches the first. The network below is 37 % right after 1500 steps; with tanh instead, 95 % by step 475.",
        act: Act::Sigmoid,
        mu: 0.0,
        sigma: 1.0,
        rate: RATE,
        bias: BIAS,
        lower: Lower::Training,
    },
    Trial {
        button: "tanh shuts when its inputs are large",
        caption: r"tanh's slope is 1 at $z = 0$ but below 0.01 beyond $|z| = 2.99$: there it is saturated, its gate shut. With inputs this spread out, 32 % meet a shut gate and $\mathbb{E}\,f'(z)^2 = 0.17$; drag the right-hand dot in to $\sigma = 0.5$ and the shut share falls to 0.0 % while $\mathbb{E}\,f'(z)^2$ rises to 0.72. As the network below learns, its gates saturate: none is shut at the start, 11 % at step 1500.",
        act: Act::Tanh,
        mu: 0.0,
        sigma: 3.0,
        rate: RATE,
        bias: BIAS,
        lower: Lower::Training,
    },
    Trial {
        button: "ReLU is open or shut, and can die",
        caption: r"ReLU's gate is exactly 1 or exactly 0. Below, every hidden unit's bias starts at $-1$, and 38 of the 48 units are shut for every point (red ticks): dead. No gradient reaches their weights, so these stay as they are, and in 1500 steps none reopens; the whole third layer is dead, every point gets the same output, and the loss stays at $\ln 3 = 1.10$. From the same start ELU is 95 % right by step 522; leaky ReLU loses no unit, but nearly all its gates are 0.01, passing a hundredth of the gradient, and after 1500 steps it is 33 % right.",
        act: Act::Relu,
        mu: -1.0,
        sigma: 0.5,
        rate: RATE,
        bias: -1.0,
        lower: Lower::Training,
    },
    Trial {
        button: "GELU and SiLU leak a little",
        caption: r"GELU's slope is 0 only at $z = -0.75$. Below that it is a little negative (down to $-0.13$), so the gradient comes back slightly reversed; above $z = 0.75$ it is a little over 1 (up to 1.13). For this bell 23 % of the gates are below 0, and as many above 1. SiLU's slope is alike, below 0 under $z = -1.28$. From the start that killed ReLU's units, no GELU unit dies, and the network learns, slowly: 74 % right after 1500 steps.",
        act: Act::Gelu,
        mu: 0.0,
        sigma: 1.0,
        rate: RATE,
        bias: -1.0,
        lower: Lower::Training,
    },
    Trial {
        button: "the same ten layers with each activation",
        caption: r"The same weights, inputs and gradient (of size 1, handed to the top layer) for each activation, four starts pooled. The weights, of variance $1/32$, on their own keep the gradient's mean square on average, so the gates decide: at each layer it is multiplied by about $\mathbb{E}\,f'(z)^2$ of that layer's inputs. ReLU's is about $1/2$, and 0.027 reaches the first layer. GELU's inputs shrink on the way up, to a root mean square of 0.003 at the tenth, and near 0 its slope is $1/2$ (this narrow bell: $\mathbb{E}\,f'(z)^2 = 0.26$), so each of the upper layers about quarters the mean square: 0.002 reaches the first. tanh and ELU keep a quarter; the sigmoid, $4.5 \times 10^{-7}$.",
        act: Act::Gelu,
        mu: 0.0,
        sigma: 0.15,
        rate: RATE,
        bias: BIAS,
        lower: Lower::Stack,
    },
];

// ----- the scene -----------------------------------------------------------------------

/// The points of each curve: 121 from −6 up to 0, and 121 from just above 0 up to 6, so
/// that a slope that jumps at 0 is drawn jumping there.
const HALF: usize = 121;
const CURVE_N: usize = 2 * HALF;

fn curve_z(k: usize) -> f64 {
    let span = Z_RANGE[1] / (HALF - 1) as f64;
    if k < HALF {
        Z_RANGE[0] + k as f64 * span
    } else {
        (k - HALF) as f64 * span
    }
}

/// The slope at a point of the curve: just above 0 on the right half.
fn curve_slope(act: Act, k: usize) -> f64 {
    let z = curve_z(k);
    act.df(if k >= HALF { z.max(f64::MIN_POSITIVE) } else { z })
}

/// The tops the histogram's axis can have: the first that the tallest bar fits under.
const TOPS: [f64; 7] = [0.1, 0.2, 0.3, 0.4, 0.6, 0.8, 1.1];

fn histogram_top(h: &Histogram) -> f64 {
    let tallest = h.open.iter().zip(&h.shut).map(|(a, b)| a + b).fold(0.0, f64::max);
    TOPS.into_iter().find(|&t| tallest * 1.08 <= t).unwrap_or(1.1)
}

// The numbers the bars' glide keeps after the bars themselves.
const B_TOP: usize = 2 * BINS;
const B_GATE: usize = 2 * BINS + 1;
const B_OUTPUT: usize = 2 * BINS + 2;
const B_LEN: usize = 2 * BINS + 3;

/// What is drawn, gliding to what is computed.
#[derive(Default)]
struct Look {
    /// f and f' at the curve's points.
    curves: Glide,
    /// The bars (open, then shut), the histogram's top, the average gate, the mean
    /// output.
    bars: Glide,
    /// The bell's centre and width.
    bell: Glide,
    /// How bright each activation's curve of the ten layers is (1: the chosen one).
    stack: Glide,
    moving: bool,
}

pub struct ActGrad {
    act: Act,
    mu: f64,
    sigma: f64,
    /// Which side of the centre the width's handle is on: 1 right, −1 left.
    side: f64,
    sets: Vec<Sets>,
    cells: Cells,
    avg: Averages,
    hist: Histogram,
    /// What `avg` and `hist` were worked out for.
    worked: Option<(Act, u64, u64)>,
    /// The ten layers with each activation (in the order of `Act::ALL`), worked out a
    /// few at a time in the first frames.
    profiles: Vec<Option<Profile>>,
    run: Run,
    rate: f64,
    bias: f64,
    running: bool,
    pace: Pace,
    lower: Lower,
    look: Look,
    dragging: bool,
    /// The window last drawn in (a new one is drawn at once, without gliding).
    canvas: Option<Vec2>,
    map: Option<Image>,
    /// What the map was painted for: the run's number and step.
    painted: Option<(u64, usize)>,
    runs: u64,
    /// Room for the run's gates, painted into the map, and the map's colours.
    gates: Vec<f32>,
    tints: Vec<Color>,
    /// The activation and its slope at the curves' points, for the activation they are of.
    curves: Option<(Act, Vec<f32>)>,
}

impl ActGrad {
    /// (Nothing here needs the graphics device, so the tests can make one too.)
    fn create() -> ActGrad {
        let act = Act::Relu;
        let mut s = ActGrad {
            act,
            mu: 0.0,
            sigma: 1.0,
            side: 1.0,
            sets: Act::ALL.iter().map(|&a| Sets::of(a)).collect(),
            cells: Cells::new(),
            avg: Averages::default(),
            hist: Histogram { open: [0.0; BINS], shut: [0.0; BINS] },
            worked: None,
            profiles: vec![None; Act::ALL.len()],
            run: Run::new(act, RATE, BIAS),
            rate: RATE,
            bias: BIAS,
            running: true,
            pace: Pace::default(),
            lower: Lower::Training,
            look: Look::default(),
            dragging: false,
            canvas: None,
            map: None,
            painted: None,
            runs: 0,
            gates: Vec::new(),
            tints: gate_tints(),
            curves: None,
        };
        s.refresh();
        s
    }

    /// Set up one of the experiments in `TRIALS`.
    fn set_up(&mut self, k: usize) {
        let Some(t) = TRIALS.get(k) else { return };
        (self.act, self.mu, self.sigma, self.rate, self.bias, self.lower) = (t.act, t.mu, t.sigma, t.rate, t.bias, t.lower);
        self.side = 1.0;
        self.restart();
    }

    /// A new run from the start, with the present settings.
    fn restart(&mut self) {
        self.run = Run::new(self.act, self.rate, self.bias);
        self.runs += 1;
        self.running = true;
        self.pace.reset();
        self.refresh();
    }

    /// Bring what depends on the settings up to date with them (a change of activation
    /// also starts the run again).
    fn refresh(&mut self) {
        self.mu = if self.mu.is_finite() { self.mu.max(-MU_MOST).min(MU_MOST) } else { 0.0 };
        self.sigma = if self.sigma.is_finite() { self.sigma.max(0.0).min(SIGMA_MOST) } else { 1.0 };
        if self.run.act != self.act || self.run.rate != self.rate || self.run.bias != self.bias {
            self.run = Run::new(self.act, self.rate, self.bias);
            self.runs += 1;
            self.running = true;
            self.pace.reset();
        }
        let key = (self.act, self.mu.to_bits(), self.sigma.to_bits());
        if self.worked != Some(key) {
            let sets = &self.sets[index_of(self.act)];
            self.avg = averages(self.act, sets, self.mu, self.sigma);
            self.hist = histogram(self.act, self.mu, self.sigma, &self.cells);
            self.worked = Some(key);
        }
    }

    fn training(&self) -> bool {
        self.running && !self.run.finished()
    }

    /// Work out the ten layers for the activations not done yet, the chosen one first,
    /// for at most `seconds` (but at least one).
    fn ensure_profiles(&mut self, seconds: f64) {
        let budget = crate::clock::Budget::new(seconds);
        let chosen = index_of(self.act);
        let mut done = 0;
        for i in std::iter::once(chosen).chain(0..Act::ALL.len()) {
            if self.profiles[i].is_none() {
                if done > 0 && budget.spent() {
                    return;
                }
                self.profiles[i] = Some(profile(Act::ALL[i]));
                done += 1;
            }
        }
    }

    fn profiles_done(&self) -> bool {
        self.profiles.iter().all(Option::is_some)
    }

    /// The steps of this frame: so many a second of the scene's time, within the
    /// frame's allowance (none while the scene is paused).
    fn work(&mut self, dt: f32, real_dt: f32) {
        if !self.training() || dt <= 0.0 {
            return;
        }
        let mut turn = self.pace.begin(STEPS_PER_SECOND, dt, real_dt, 0.0015);
        while !self.run.finished() && turn.next() {
            self.run.step();
        }
        self.pace.end(turn);
    }

    /// What the glides go to (the curves are kept: the smooth activations' are a
    /// library call a point).
    fn goals(&mut self) -> (Vec<f32>, Vec<f32>, [f32; 2], Vec<f32>) {
        let act = self.act;
        if self.curves.as_ref().is_none_or(|c| c.0 != act) {
            let mut curves: Vec<f32> = (0..CURVE_N).map(|k| act.f(curve_z(k)) as f32).collect();
            curves.extend((0..CURVE_N).map(|k| curve_slope(act, k) as f32));
            self.curves = Some((act, curves));
        }
        let curves = self.curves.as_ref().map(|c| c.1.clone()).unwrap_or_default();
        let mut bars: Vec<f32> = self.hist.open.iter().chain(&self.hist.shut).map(|&v| v as f32).collect();
        bars.extend([histogram_top(&self.hist) as f32, self.avg.gate as f32, self.avg.output as f32]);
        let stack = Act::ALL.iter().map(|&a| if a == act { 1.0 } else { 0.0 }).collect();
        (curves, bars, [self.mu as f32, self.sigma as f32], stack)
    }

    /// Bring the picture `dt` seconds nearer to what it shows. (`snap`: there at once.)
    fn follow(&mut self, dt: f32, snap: bool) {
        let (curves, bars, bell, stack) = self.goals();
        let look = &mut self.look;
        if snap {
            look.curves.snap(&curves);
            look.bars.snap(&bars);
            look.bell.snap(&bell);
            look.stack.snap(&stack);
            look.moving = false;
            return;
        }
        if self.dragging {
            // (the bell answers the hand at once)
            look.bell.snap(&bell);
        }
        let mut moving = look.curves.follow(&curves, dt, false);
        moving |= look.bars.follow(&bars, dt, false);
        moving |= look.bell.follow(&bell, dt, false);
        moving |= look.stack.follow(&stack, dt, false);
        look.moving = moving;
    }
}

// ----- the layout ----------------------------------------------------------------------

/// Where everything goes in a window.
#[derive(Clone, Debug, PartialEq)]
struct Layout {
    view: Vec2,
    /// Room for all four pictures at once.
    wide: bool,
    /// A phone held upright: one picture under another, smaller.
    narrow: bool,
    /// Text sizes: headings and words, small print.
    text: f32,
    small: f32,
    /// "activation:" and the nine buttons.
    lead: Option<Vec2>,
    acts: Vec<Rect>,
    /// The activation and its slope (the plots' rectangles).
    act_plot: Option<Rect>,
    gate_plot: Option<Rect>,
    /// The histogram, and where its numbers go (top-left, width).
    hist: Option<Rect>,
    numbers: Option<(Vec2, f32)>,
    /// The two buttons that choose the lower picture (when there is room for one).
    tabs: Vec<Rect>,
    /// The ten layers; and the network that learns: its map, its chart, its buttons.
    stack: Option<Rect>,
    map: Option<Rect>,
    chart: Option<Rect>,
    run_buttons: Vec<Rect>,
}

/// The names of the activations, on their buttons.
fn act_words() -> [&'static str; 9] {
    Act::ALL.map(|a| a.name())
}

/// Rectangles for buttons side by side from (x, y), each as wide as its words need.
fn button_row(words: &[&str], x: f32, y: f32, h: f32, size: f32, gap: f32) -> Vec<Rect> {
    let mut left = x;
    words
        .iter()
        .map(|w| {
            let width = text_width(w, size) + 16.0;
            let r = (Vec2::new(left, y), Vec2::new(left + width, y + h));
            left += width + gap;
            r
        })
        .collect()
}

fn rect(x0: f32, y0: f32, x1: f32, y1: f32) -> Option<Rect> {
    (x1 - x0 >= 24.0 && y1 - y0 >= 18.0 && x0.is_finite() && y1.is_finite()).then(|| (Vec2::new(x0, y0), Vec2::new(x1, y1)))
}

impl Layout {
    fn of(view: Vec2, inset: f32, lower: Lower) -> Layout {
        let view = view.max(Vec2::ONE);
        let narrow = view.x < 620.0;
        let wide = view.x >= 1040.0 && view.y >= 660.0;
        let (text, small) = if narrow { (11.5, 10.5) } else { (13.0, 11.5) };
        let side = if narrow { 10.0 } else { 20.0 };
        let top = inset.max(0.0) + if narrow { 6.0 } else { 12.0 };
        let mut lay = Layout {
            view,
            wide,
            narrow,
            text,
            small,
            lead: None,
            acts: Vec::new(),
            act_plot: None,
            gate_plot: None,
            hist: None,
            numbers: None,
            tabs: Vec::new(),
            stack: None,
            map: None,
            chart: None,
            run_buttons: Vec::new(),
        };
        if view.x < 200.0 || view.y < 200.0 {
            return lay;
        }
        // the buttons: on one row after "activation:" if they fit, else in two that
        // fill the width
        let h = if narrow { 24.0 } else { 26.0 };
        let words = act_words();
        let lead_w = text_width("activation:", text) + 10.0;
        let one_row = button_row(&words, side + lead_w, top, h, small, 5.0);
        let below_buttons = if one_row.last().is_some_and(|r| r.1.x <= view.x - side) {
            lay.lead = Some(Vec2::new(side, top + h * 0.5));
            lay.acts = one_row;
            top + h
        } else {
            lay.acts = plot::even_row(5, side, view.x - side, top, h, 4.0);
            lay.acts.extend(plot::even_row(4, side, view.x - side, top + h + 4.0, h, 4.0));
            top + 2.0 * h + 4.0
        };
        let ticks_w = if narrow { 30.0 } else { 34.0 };
        let title_h = if narrow { 20.0 } else { 24.0 };
        let axis_h = if narrow { 30.0 } else { 34.0 };
        let gap = if narrow { 12.0 } else { 18.0 };
        if wide {
            let y1 = below_buttons + gap + title_h;
            let b1 = y1 + (view.y - y1) * 0.50;
            let a1 = side + ticks_w + 0.31 * view.x;
            let split = y1 + (b1 - y1 - title_h) * 0.38;
            lay.act_plot = rect(side + ticks_w, y1, a1, split);
            lay.gate_plot = rect(side + ticks_w, split + title_h + 6.0, a1, b1);
            let h0 = a1 + 34.0 + ticks_w;
            let h1 = h0 + 0.21 * view.x;
            lay.hist = rect(h0, y1, h1, b1 - 4.0 * (small + 7.0) - 8.0 - axis_h);
            lay.numbers = Some((Vec2::new(h0 - ticks_w + 4.0, b1 - 4.0 * (small + 7.0) + 2.0), h1 - h0 + ticks_w));
            lay.stack = rect(h1 + 40.0 + ticks_w, y1 + small + 4.0, view.x - side - 116.0, b1);
            Layout::training(&mut lay, side, b1 + axis_h + gap + title_h, view.y - axis_h, ticks_w);
            return lay;
        }
        // the gate's two pictures on top (side by side, or one under the other on a
        // phone), and under them one of the other two, chosen by two buttons
        let room = view.y - below_buttons;
        if narrow {
            let y1 = below_buttons + gap + title_h;
            let a_h = (room * 0.12).max(40.0);
            let g_h = (room * 0.17).max(60.0);
            lay.act_plot = rect(side + ticks_w, y1, view.x - side - 4.0, y1 + a_h);
            let g0 = y1 + a_h + title_h + 6.0;
            lay.gate_plot = rect(side + ticks_w, g0, view.x - side - 4.0, g0 + g_h);
            let hist0 = g0 + g_h + axis_h + title_h;
            let hist_h = (room * 0.10).max(36.0);
            let numbers_w = 150.0f32.min((view.x - 2.0 * side) * 0.42);
            lay.hist = rect(side + ticks_w, hist0, view.x - side - numbers_w - 8.0, hist0 + hist_h);
            lay.numbers = Some((Vec2::new(view.x - side - numbers_w, hist0 - 4.0), numbers_w));
            let tabs_y = hist0 + hist_h + axis_h + 4.0;
            Layout::lower(&mut lay, lower, side, tabs_y, ticks_w, title_h, axis_h, gap, h);
        } else {
            let y1 = below_buttons + gap + title_h;
            let b1 = y1 + room * 0.40;
            let a1 = side + ticks_w + (view.x - 2.0 * side) * 0.55;
            let split = y1 + (b1 - y1 - title_h) * 0.38;
            lay.act_plot = rect(side + ticks_w, y1, a1, split);
            lay.gate_plot = rect(side + ticks_w, split + title_h + 6.0, a1, b1);
            let h0 = a1 + 30.0 + ticks_w;
            lay.hist = rect(h0, y1, view.x - side - 4.0, b1 - 4.0 * (small + 7.0) - 8.0 - axis_h);
            lay.numbers = Some((Vec2::new(h0 - ticks_w + 4.0, b1 - 4.0 * (small + 7.0) + 2.0), view.x - side - h0 + ticks_w));
            Layout::lower(&mut lay, lower, side, b1 + axis_h + 6.0, ticks_w, title_h, axis_h, gap, h);
        }
        lay
    }

    /// The two buttons at `y` and, under them, the lower picture chosen.
    #[allow(clippy::too_many_arguments)]
    fn lower(lay: &mut Layout, lower: Lower, side: f32, y: f32, ticks_w: f32, title_h: f32, axis_h: f32, gap: f32, h: f32) {
        let view = lay.view;
        if view.y - y < 160.0 {
            return;
        }
        let tab_w = (view.x - 2.0 * side).min(420.0);
        lay.tabs = plot::even_row(2, side, side + tab_w, y, h, 5.0);
        let y0 = y + h + gap + title_h.max(h + 4.0);
        let bottom = view.y - axis_h - if lay.narrow { 8.0 } else { 0.0 };
        match lower {
            Lower::Stack => lay.stack = rect(side + ticks_w, y0 + lay.small + 4.0, view.x - side - if lay.narrow { 96.0 } else { 116.0 }, bottom),
            Lower::Training => Layout::training(lay, side, y0, bottom, ticks_w),
        }
    }

    /// The network that learns, from `y0` (the top of its pictures) to `y1`.
    fn training(lay: &mut Layout, side: f32, y0: f32, y1: f32, ticks_w: f32) {
        let view = lay.view;
        let label_w = if lay.narrow { 20.0 } else { 58.0 };
        let h = if lay.narrow { 24.0 } else { 26.0 };
        // the buttons on the heading's line, at the right
        let first = ["pause", "go on", "run again"].iter().map(|w| text_width(w, lay.small)).fold(0.0, f32::max) + 16.0;
        let second = text_width("start again", lay.small) + 16.0;
        let x = view.x - side - first - second - 5.0;
        let y = y0 - h - 4.0;
        lay.run_buttons = vec![(Vec2::new(x, y), Vec2::new(x + first, y + h)), (Vec2::new(x + first + 5.0, y), Vec2::new(x + first + 5.0 + second, y + h))];
        // the chart's own heading
        let ch = lay.small + 12.0;
        if lay.narrow {
            // the map over the chart
            let map_h = ((y1 - y0) * 0.42).max(40.0);
            lay.map = rect(side + label_w, y0, view.x - side, y0 + map_h);
            lay.chart = rect(side + ticks_w, y0 + map_h + 20.0 + ch, view.x - side - 4.0, y1);
        } else {
            let split = side + label_w + (view.x - 2.0 * side) * 0.46;
            lay.map = rect(side + label_w, y0, split, y1 - 20.0);
            lay.chart = rect(split + 30.0 + ticks_w, y0 + ch, view.x - side - 4.0, y1);
        }
    }
}

// ----- drawing -------------------------------------------------------------------------

/// Shut gates, in the histogram and in the chart of shares.
const SHUT_GREY: Color = pal::DIM;
/// Dead units.
const DEAD: Color = pal::RED;
/// A gate below 0, in the map.
const NEGATIVE: Color = pal::BLUE;
/// The share of points the network classifies right.
const RIGHT: Color = pal::TEXT;
/// The map's pink, a gate of 1: the course's pink, a little darker, so that a map of
/// thousands of open gates does not outshine the rest of the picture.
const OPEN: Color = Color::hex(0xC23A6E);
/// The map's black: a shut gate.
const DARK: Color = Color::hex(0x0B0E15);

// Kinds of line, each a batch of its own (see `plot::apart`).
/// The activation itself, which leaves its chart where it is large.
const F_CURVE: u8 = plot::OWN;
/// The bell's outline, which can reach past the charts' ends; and its spike, when it
/// has no width.
const BELL: u8 = plot::OWN + 1;
const SPIKE: u8 = plot::OWN + 2;
/// The mean output, which can leave its chart.
const MEAN_OUT: u8 = plot::OWN + 3;
/// The chart of the run, whose lines grow.
const RUN_RIGHT: u8 = plot::OWN + 4;
const RUN_DEAD: u8 = plot::OWN + 5;
const RUN_SHUT: u8 = plot::OWN + 6;
/// The ten layers' curves, the chosen one bright, over the others.
const STACK_TOP: u8 = plot::OWN + 7;
/// The dashes that are always there.
const DASHED: u8 = plot::MARKS;

fn in_order(f: &mut Frame) {
    for kind in [plot::AXES, plot::CURVES, plot::MARKS, plot::BUTTONS, F_CURVE, BELL, SPIKE, RUN_RIGHT, RUN_DEAD, RUN_SHUT, STACK_TOP] {
        plot::apart(f, kind, false);
    }
    for kind in [DASHED, MEAN_OUT] {
        plot::apart(f, kind, true);
    }
}

/// A batch of lines with nothing in it this frame is given a point that cannot be seen:
/// one left empty is dropped, and every batch after it would be compared, point for
/// point, with the one before it in the last frame.
fn keep_batches(f: &mut Frame) {
    let corner = f.hud(Vec2::ZERO);
    for (kind, dashed) in [
        (plot::AXES, false),
        (plot::CURVES, false),
        (plot::MARKS, false),
        (plot::BUTTONS, false),
        (F_CURVE, false),
        (BELL, false),
        (SPIKE, false),
        (RUN_RIGHT, false),
        (RUN_DEAD, false),
        (RUN_SHUT, false),
        (STACK_TOP, false),
        (DASHED, true),
        (MEAN_OUT, true),
    ] {
        let batch = plot::apart(f, kind, dashed);
        if batch.pts.len() <= 1 {
            batch.segment(corner, corner, 0.0, Color::CLEAR);
        }
    }
}

/// The first of the texts that fits in `room` pixels (the last if none does).
fn fitting<'t>(texts: &[&'t str], room: f32, size: f32) -> &'t str {
    texts.iter().copied().find(|t| text_width(t, size) <= room).or(texts.last().copied()).unwrap_or("")
}

/// A share as a percentage, to `decimals` places.
fn percent(v: f64, decimals: usize) -> String {
    format!("{} %", num(100.0 * v, decimals))
}

/// A positive number with two significant figures, as markup: 0.82, 0.022, 1.8 × 10⁻³.
fn sig2(v: f64) -> String {
    if !(v.is_finite() && v > 0.0) {
        return num(v, 2);
    }
    let e = v.log10().floor() as i32;
    if e >= -2 {
        num(v, (1 - e).max(0) as usize)
    } else {
        let m = v / 10f64.powi(e);
        // (a mantissa that rounds up to 10 is 1.0 of the next power)
        let (m, e) = if (m * 10.0).round() >= 100.0 { (1.0, e + 1) } else { (m, e) };
        format!(r"{} \times 10^{{{}}}", num(m, 1), num(e as f64, 0))
    }
}

/// The map's colours for gates from `TINT_LO` to `TINT_HI`, worked out once: a colour
/// mixed in linear light costs nine powers, and the map has thousands of gates.
const TINT_N: usize = 512;
const TINT_LO: f32 = -0.5;
const TINT_HI: f32 = 1.2;

fn gate_tints() -> Vec<Color> {
    (0..TINT_N).map(|i| gate_color(TINT_LO + (TINT_HI - TINT_LO) * i as f32 / (TINT_N - 1) as f32)).collect()
}

/// A gate's colour from the table (shut and not-a-number exactly as `gate_color` says).
fn gate_tint(tints: &[Color], g: f32) -> Color {
    if !g.is_finite() || g.abs() < SHUT as f32 {
        return gate_color(g);
    }
    let i = ((g - TINT_LO) / (TINT_HI - TINT_LO) * (TINT_N - 1) as f32).round().max(0.0) as usize;
    tints.get(i.min(TINT_N - 1)).copied().unwrap_or(DARK)
}

/// The colour of a gate in the map: black where it is shut, more pink the more it
/// passes, whiter above 1, blue below 0.
fn gate_color(g: f32) -> Color {
    if !g.is_finite() {
        pal::FAINT
    } else if g.abs() < SHUT as f32 {
        DARK
    } else if g < 0.0 {
        // (on the scale of the pink, doubled: GELU's least slope, −0.13, would be all
        // but black)
        DARK.mix(NEGATIVE, (-2.0 * g).min(1.0))
    } else if g <= 1.0 {
        DARK.mix(OPEN, g)
    } else {
        OPEN.mix(pal::WHITE, ((g - 1.0) / 0.13).min(1.0) * 0.6)
    }
}

/// The bell's height over the gate's chart: from its floor up to 0.62 of the chart.
fn bell_height(plot: &Plot) -> (f64, f64) {
    let (lo, hi) = (plot.y[0], plot.y[1]);
    (lo, 0.62 * (hi - lo))
}

fn bell_at(z: f64, mu: f64, sigma: f64) -> f64 {
    if sigma > 1e-6 {
        (-0.5 * ((z - mu) / sigma).powi(2)).exp()
    } else {
        0.0
    }
}

/// The activation's chart and the gate's.
fn act_plot(r: Rect) -> Plot {
    Plot::new(r, Z_RANGE, [-1.5, 3.0])
}

fn gate_plot(r: Rect) -> Plot {
    Plot::new(r, Z_RANGE, [-0.3, 1.3])
}

/// The ten layers' chart: across, the layers from the top one (10) at the left back to
/// the first at the right; up, the base-10 logarithm of the gradient's size.
fn stack_plot(r: Rect) -> Plot {
    Plot::new(r, [0.6, STACK_DEPTH as f64 + 0.4], [-7.2, 0.5])
}

/// Where layer `l` (from 1) is across the ten layers' chart.
fn stack_x(l: usize) -> f64 {
    (STACK_DEPTH + 1 - l) as f64
}

impl Scene for ActGrad {
    fn view(&self) -> View {
        View::Flat { center: Vec2::ZERO, half_height: 1.0 }
    }

    fn settings(&self, s: &mut ViewSettings) {
        s.grid = false;
        s.bloom = s.bloom.min(0.2);
    }

    fn animated(&self) -> bool {
        self.look.moving || self.training() || self.dragging || !self.profiles_done()
    }

    fn pointer_hint(&self) -> Option<&'static str> {
        Some("the bell's two dots: drag its centre and its width    buttons over the picture: click")
    }

    fn ui(&mut self, ui: &mut egui::Ui, _: &Gfx) {
        ui::note(
            ui,
            r"Going back through a network, each unit multiplies the gradient that reaches its output by the slope of its activation $f$ at the input $z$ it had on the way forward. The slope is the unit's gate: at 1 the gradient goes through whole, at 0 nothing does. A deep network passes the gradient through one layer of gates after another, so the gates, with the weights between them, decide how much of it reaches the first layers.",
        );
        ui::equation(ui, r"\frac{\partial L}{\partial z} = f'(z)\,\frac{\partial L}{\partial a}, \qquad a = f(z)");

        ui::section(ui, "Things to try");
        for (k, trial) in TRIALS.iter().enumerate() {
            if ui::experiment(ui, trial.button, trial.caption) {
                self.set_up(k);
            }
        }

        ui::section(ui, "The activation");
        let names = act_words();
        let mut chosen = index_of(self.act);
        if ui::choice(ui, &names, &mut chosen) {
            self.act = Act::ALL[chosen];
            self.refresh();
        }
        ui::section(ui, "The inputs");
        let mut mu = self.mu as f32;
        if ui::slider(ui, r"centre $\mu$", &mut mu, -MU_MOST as f32..=MU_MOST as f32) {
            self.mu = mu as f64;
            self.refresh();
        }
        let mut sigma = self.sigma as f32;
        if ui::slider(ui, r"width $\sigma$", &mut sigma, 0.0..=SIGMA_MOST as f32) {
            self.sigma = sigma as f64;
            self.refresh();
        }
        ui::note(ui, r"The bell stands for the inputs $z$ one unit receives over many examples (or a layer's units over one). Its two dots on the picture can be dragged.");
        ui::section(ui, "The network that learns");
        let mut rate = self.rate as f32;
        if ui::slider_log(ui, "learning rate", &mut rate, 0.03..=3.0) {
            self.rate = rate as f64;
            self.refresh();
        }
        let mut bias = self.bias as f32;
        if ui::slider(ui, "starting bias", &mut bias, -2.0..=1.0) {
            self.bias = bias as f64;
            self.refresh();
        }
        ui.horizontal(|ui| {
            let pause = if self.training() { "pause" } else if self.run.finished() { "run again" } else { "go on" };
            if ui.button(ui::label_text(ui, pause, 12.5, None, false)).clicked() {
                if self.run.finished() {
                    self.restart();
                } else {
                    self.running = !self.running;
                }
            }
            if ui.button(ui::label_text(ui, "start again", 12.5, None, false)).clicked() {
                self.restart();
            }
        });
        ui::note(ui, "Plain gradient descent: every weight moves by the rate times its gradient, so a small gradient makes a small step. The starting bias is what every hidden unit's bias starts at (usually 0). An optimiser such as Adam divides each weight's step by the size of its recent gradients, which makes up for much of what small gates take away, though not for gates that are exactly 0.");

        ui::section(ui, "Reading the picture");
        ui::key(
            ui,
            &[
                (FORWARD, r"cyan: what goes forward: the activation $f(z)$, the bell of inputs $z$, the mean output"),
                (BACKWARD, r"pink: what comes back: the gate $f'(z)$, the gates the inputs meet, the gradient at each of the ten layers, an open gate in the map"),
                (SHUT_GREY, "grey: gates that are shut"),
                (NEGATIVE, "blue (in the map): a gate below 0, which sends the gradient back with its sign turned round"),
                (DEAD, "red: dead units"),
                (RIGHT, "white: the share of points the network that learns classifies right"),
            ],
        );
        ui::section(ui, "Words");
        ui::term(ui, "Gate:", r"the slope $f'(z)$ of a unit's activation at its input $z$. On the way back the gradient is multiplied by it.");
        ui::term(ui, "Shut:", r"a gate smaller than 0.01 in size: next to nothing gets back through it.");
        ui::term(ui, "Saturated:", r"a smooth activation (tanh, the sigmoid) far out where it is nearly flat, so that its gate is nearly 0.");
        ui::term(ui, "Dead:", "a unit whose gate is shut for every point of the data, so that next to no gradient reaches its weights (for ReLU, none at all).");
        ui::term(ui, r"$\mathbb{E}$:", r"the average over the inputs, spread as the bell says. $\mathbb{E}\,f'(z)^2$ is about the factor by which the gates multiply the gradient's mean square (exactly, when what reaches them is unrelated to their inputs).");
        ui::term(ui, "Root mean square:", "the square root of the average of the squares: a typical size, whatever the signs.");
        ui::term(ui, "Leaky ReLU:", r"$z$ above 0 and $0.01\,z$ below: ReLU with a slope of 0.01 where ReLU has none.");
        ui::term(ui, "ELU:", r"$z$ above 0 and $e^z - 1$ below, whose slope $e^z$ falls off gradually.");
        ui::term(ui, "GELU:", r"$z\,\Phi(z)$, with $\Phi$ the share of a standard bell below $z$: the activation of most transformers.");
        ui::term(ui, "SiLU:", r"$z/(1 + e^{-z})$, $z$ times the sigmoid (also called swish): a cousin of GELU, used in many transformers too.");
        ui::term(ui, "Softplus:", r"$\ln(1 + e^z)$, a smoothed ReLU, whose slope is the sigmoid.");

        ui::section(ui, "Numbers");
        let a = &self.avg;
        ui::readout(ui, r"average gate $\mathbb{E}\,f'(z)$", num(a.gate, 3));
        ui::readout(ui, r"average squared gate $\mathbb{E}\,f'(z)^2$", num(a.gate_sq, 3));
        ui::readout(ui, "inputs meeting a shut gate", percent(a.shut, 1));
        ui::readout(ui, "inputs meeting a gate below 0", percent(a.negative, 1));
        ui::readout(ui, r"mean output $\mathbb{E}\,f(z)$", num(a.output, 3));
        let r = &self.run;
        ui::readout(ui, "training step", format!("{} of {STEPS}", r.step));
        ui::readout(ui, "loss", num(r.loss, 3));
        ui::readout(ui, "points classified right", percent(r.right, 0));
        ui::readout(ui, "dead units", format!("{} of {ALL_UNITS} ({} in the layers)", r.dead_count(), (0..LAYERS).map(|l| r.dead_in(l).to_string()).collect::<Vec<_>>().join(", ")));
        ui::readout(ui, "gates shut", percent(r.shut, 0));
        ui::see_also(
            ui,
            &[
                ("backpropmat", "the gates as a diagonal matrix in the backward pass"),
                ("gradflow", "the weights' scale: what decides vanishing and exploding"),
                ("signal", "the same averages, from the forward side"),
                ("fit1d", "what each activation's shape builds"),
            ],
        );
    }

    fn frame(&mut self, f: &mut Frame) {
        let view = f.view_size();
        // (a canvas squeezed to a sliver is one the controls have opened over, on a phone)
        let covered = view.x < 160.0 || view.y < 160.0;
        let lay = Layout::of(view, f.top_inset, self.lower);
        self.dragging = false;
        if !covered {
            self.buttons(f, &lay);
            self.hands(f, &lay);
        }
        self.refresh();
        self.ensure_profiles(0.002);
        self.work(f.dt, f.real_dt);
        let snap = self.canvas != Some(lay.view);
        if snap && !covered {
            self.canvas = Some(lay.view);
        }
        self.follow(f.real_dt, snap);
        if covered {
            return;
        }
        in_order(f);
        self.draw_activation(f, &lay);
        self.draw_histogram(f, &lay);
        self.draw_stack(f, &lay);
        self.draw_training(f, &lay);
        keep_batches(f);
    }
}

impl ActGrad {
    /// The buttons over the picture: the activations, the choice of lower picture, and
    /// the run's.
    fn buttons(&mut self, f: &mut Frame, lay: &Layout) {
        if let Some(at) = lay.lead {
            f.text_px(at, "activation:", lay.text, pal::DIM, Align2::LEFT_CENTER);
        }
        for (k, r) in lay.acts.iter().enumerate() {
            let act = Act::ALL[k.min(Act::ALL.len() - 1)];
            if plot::button(f, *r, act.name(), lay.small, act == self.act) && act != self.act {
                self.act = act;
            }
        }
        for (r, (name, which)) in lay.tabs.iter().zip([("the same ten layers", Lower::Stack), ("a network that learns", Lower::Training)]) {
            if plot::button(f, *r, name, lay.small, self.lower == which) {
                self.lower = which;
            }
        }
        if lay.map.is_some() {
            let pause = if self.training() {
                "pause"
            } else if self.run.finished() {
                "run again"
            } else {
                "go on"
            };
            if let Some(r) = lay.run_buttons.first() {
                if plot::button(f, *r, pause, lay.small, false) {
                    if self.run.finished() {
                        self.restart();
                    } else {
                        self.running = !self.running;
                    }
                }
            }
            if let Some(r) = lay.run_buttons.get(1) {
                if plot::button(f, *r, "start again", lay.small, false) {
                    self.restart();
                }
            }
        }
    }

    /// The bell's two dots: its centre, on the floor of the gate's chart, and its width,
    /// on its flank.
    fn hands(&mut self, f: &mut Frame, lay: &Layout) {
        let Some(r) = lay.gate_plot else { return };
        let plot = gate_plot(r);
        let (mu, sigma) = (self.look.bell.get(0) as f64, self.look.bell.get(1) as f64);
        let (floor, height) = bell_height(&plot);
        if let Some((x, _)) = plot.handle(f, 1, mu, floor, FORWARD) {
            if x.is_finite() {
                self.mu = x.max(-MU_MOST).min(MU_MOST);
                self.dragging = true;
            }
        }
        // the width's dot on whichever flank is on the chart
        if mu + sigma > Z_RANGE[1] - 0.25 && mu - sigma > Z_RANGE[0] + 0.25 {
            self.side = -1.0;
        } else if mu - sigma < Z_RANGE[0] + 0.25 && mu + sigma < Z_RANGE[1] - 0.25 {
            self.side = 1.0;
        }
        let at = mu + self.side * sigma;
        if let Some((x, _)) = plot.handle(f, 2, at, floor + height * (-0.5f64).exp(), FORWARD) {
            if x.is_finite() {
                let d = x - self.mu;
                if d.abs() > 1e-9 {
                    self.side = d.signum();
                }
                self.sigma = d.abs().min(SIGMA_MOST);
                self.dragging = true;
            }
        }
    }

    fn draw_activation(&mut self, f: &mut Frame, lay: &Layout) {
        let (Some(ar), Some(gr)) = (lay.act_plot, lay.gate_plot) else { return };
        let (a, g) = (act_plot(ar), gate_plot(gr));
        let (text, small) = (lay.text, lay.small);
        hud_panel(f, ar.0 - Vec2::new(36.0, 30.0), gr.1 + Vec2::new(10.0, 32.0));
        let curves = &self.look.curves.shown;
        let (mu, sigma) = (self.look.bell.get(0) as f64, self.look.bell.get(1) as f64);
        let room = a.width() + 30.0;
        // the activation, over the bell of its inputs
        let title = fitting(&[r"the activation $a = f(z)$, and the inputs $z$ a unit receives: a bell", r"the activation $a = f(z)$ and its inputs $z$", r"$a = f(z)$"], room, text);
        f.text_px(Vec2::new(ar.0.x - 30.0, ar.0.y - 8.0), title, text, pal::TEXT, Align2::LEFT_BOTTOM);
        for plot in [&a, &g] {
            let (floor, height) = bell_height(plot);
            let lift = |z: f64| floor + height * bell_at(z, mu, sigma);
            if sigma > 0.02 {
                plot.area(f, mu - 4.5 * sigma, mu + 4.5 * sigma, floor, 160, FORWARD.a(0.16), lift);
            }
            let pts: Vec<(f64, f64)> = (0..161).map(|i| Z_RANGE[0] + (Z_RANGE[1] - Z_RANGE[0]) * i as f64 / 160.0).map(|z| (z, lift(z))).collect();
            plot.path(f, BELL, pts, 1.2, FORWARD.a(0.5));
            if sigma <= 0.02 && (Z_RANGE[0]..=Z_RANGE[1]).contains(&mu) {
                let (p, q) = (plot.at(mu, floor), plot.at(mu, floor + height));
                plot::line_of(f, SPIKE, p, q, 2.0, FORWARD.a(0.8));
            }
        }
        a.y_grid(f);
        a.y_axis(f, "", small);
        line_of_axis(f, &a);
        if curves.len() == 2 * CURVE_N {
            a.path(f, F_CURVE, (0..CURVE_N).map(|k| (curve_z(k), curves[k] as f64)), 2.4, FORWARD);
        }
        // the mean output, the forward side
        let out = self.look.bars.get(B_OUTPUT) as f64;
        if out >= a.y[0] && out <= a.y[1] {
            let (p, q) = (a.at(a.x[0], out), a.at(a.x[1], out));
            plot::dashed_of(f, MEAN_OUT, p, q, 1.2, FORWARD.a(0.75));
            let label = format!(r"mean output $\mathbb{{E}}\,f(z) = {}$", num(self.avg.output, 2));
            let above = out < a.y[1] - 0.35 * (a.y[1] - a.y[0]);
            let at = a.at(a.x[0], out) + Vec2::new(6.0, if above { -4.0 } else { 4.0 });
            f.text_px(at, label, small, FORWARD, if above { Align2::LEFT_BOTTOM } else { Align2::LEFT_TOP });
        }

        // the gate
        let title = fitting(
            &[
                r"its slope $f'(z)$: the gate, by which the gradient is multiplied on the way back",
                r"its slope $f'(z)$: the gate the gradient passes",
                r"the gate $f'(z)$",
            ],
            room,
            text,
        );
        f.text_px(Vec2::new(gr.0.x - 30.0, gr.0.y - 8.0), title, text, pal::TEXT, Align2::LEFT_BOTTOM);
        g.y_axis(f, "", small);
        g.x_axis(f, "", small);
        f.text_px(Vec2::new(gr.1.x, gr.1.y + 8.0 + small), "$z$", small, pal::DIM, Align2::RIGHT_TOP);
        for level in [0.0, 1.0] {
            g.hline(f, DASHED, level, 1.0, pal::DIM.a(0.45), true);
        }
        if curves.len() == 2 * CURVE_N {
            g.path(f, plot::CURVES, (0..CURVE_N).map(|k| (curve_z(k), curves[CURVE_N + k] as f64)), 2.4, BACKWARD);
        }
        // the dots' names
        let (floor, height) = bell_height(&g);
        let centre = g.at(mu.max(g.x[0]).min(g.x[1]), floor);
        f.text_px(centre + Vec2::new(8.0, -6.0), r"$\mu$", small, FORWARD, Align2::LEFT_BOTTOM);
        let w = (mu + self.side * sigma).max(g.x[0]).min(g.x[1]);
        let flank = g.at(w, floor + height * (-0.5f64).exp());
        let align = if self.side > 0.0 { Align2::LEFT_BOTTOM } else { Align2::RIGHT_BOTTOM };
        f.text_px(flank + Vec2::new(8.0 * self.side as f32, -6.0), r"$\sigma$", small, FORWARD, align);
    }

    fn draw_histogram(&mut self, f: &mut Frame, lay: &Layout) {
        let Some(hr) = lay.hist else { return };
        let (text, small) = (lay.text, lay.small);
        let bars = &self.look.bars.shown;
        if bars.len() != B_LEN {
            return;
        }
        let top = (bars[B_TOP] as f64).max(0.05);
        let p = Plot::new(hr, [BIN_FIRST - 0.5 * BIN, BIN_FIRST + (BINS as f64 - 0.5) * BIN], [0.0, top]);
        let numbers_h = lay.numbers.map_or(0.0, |(at, _)| at.y + 4.0 * (small + 7.0) - hr.1.y);
        hud_panel(f, hr.0 - Vec2::new(36.0, 30.0), hr.1 + Vec2::new(10.0, 34.0 + numbers_h.max(0.0)));
        let title = fitting(&["the gates these inputs meet: share of inputs", "the gates these inputs meet", "the gates they meet", "the gates"], p.width() + 30.0, text);
        f.text_px(Vec2::new(hr.0.x - 30.0, hr.0.y - 8.0), title, text, pal::TEXT, Align2::LEFT_BOTTOM);
        p.y_grid(f);
        p.y_axis(f, "", small);
        p.x_axis(f, "", small);
        f.text_px(Vec2::new(hr.1.x, hr.1.y + 8.0 + small), "$f'(z)$", small, pal::DIM, Align2::RIGHT_TOP);
        for b in 0..BINS {
            let (x0, x1) = (BIN_FIRST + (b as f64 - 0.5) * BIN, BIN_FIRST + (b as f64 + 0.5) * BIN);
            let (open, shut) = (bars[b] as f64, bars[BINS + b] as f64);
            p.bar(f, x0, x1, 0.0, shut, 1.0, SHUT_GREY.a(0.85));
            p.bar(f, x0, x1, shut, shut + open, 1.0, BACKWARD.a(0.8));
        }
        let shut_bar = bars[BINS + bin_of(0.0)] as f64;
        if p.px_y(0.0) - p.px_y(shut_bar.min(top)) > small {
            p.text(f, -0.5 * BIN, 0.5 * shut_bar.min(top), Vec2::new(-3.0, 0.0), "shut", small, SHUT_GREY, Align2::RIGHT_CENTER);
        }
        let mean = bars[B_GATE] as f64;
        p.vline(f, DASHED, mean.max(p.x[0]).min(p.x[1]), 1.2, BACKWARD, true);
        // the numbers
        let Some((at, width)) = lay.numbers else { return };
        let a = &self.avg;
        let line = small + 7.0;
        let lines = [
            (
                fitting(&["average gate", "average", ""], width - text_width(r"$\mathbb{E}\,f'(z) = 0.000$", small) - 12.0, small).to_string(),
                format!(r"$\mathbb{{E}}\,f'(z) = {}$", num(a.gate, 3)),
                BACKWARD,
            ),
            (
                fitting(&["the factor on the mean square", "on the mean square", ""], width - text_width(r"$\mathbb{E}\,f'(z)^2 = 0.000$", small) - 12.0, small).to_string(),
                format!(r"$\mathbb{{E}}\,f'(z)^2 = {}$", num(a.gate_sq, 3)),
                BACKWARD,
            ),
            ("".to_string(), format!("shut: {}", percent(a.shut, 1)), SHUT_GREY),
            ("".to_string(), format!("below 0: {}", percent(a.negative, 1)), NEGATIVE.mix(pal::TEXT, 0.3)),
        ];
        for (k, (words, value, color)) in lines.iter().enumerate() {
            let y = at.y + k as f32 * line;
            f.text_px(Vec2::new(at.x, y), value, small, *color, Align2::LEFT_TOP);
            if !words.is_empty() {
                f.text_px(Vec2::new(at.x + width, y), words, small, pal::DIM, Align2::RIGHT_TOP);
            }
        }
    }

    fn draw_stack(&mut self, f: &mut Frame, lay: &Layout) {
        let Some(sr) = lay.stack else { return };
        let (text, small) = (lay.text, lay.small);
        let p = stack_plot(sr);
        hud_panel(f, sr.0 - Vec2::new(40.0, 30.0 + small + 4.0), sr.1 + Vec2::new(if lay.narrow { 92.0 } else { 112.0 }, 32.0));
        // the heading, and under it what is drawn
        let room = p.width() + 130.0;
        let title = fitting(&["the same ten layers with each activation: the gradient at each", "the gradient at each of ten layers", "ten layers"], room, text);
        let sub = fitting(&["its root mean square over four starts, log scale; 1 is handed to the top", "root mean square over four starts, log scale", "log scale"], room, small);
        f.text_px(Vec2::new(sr.0.x - 34.0, sr.0.y - 10.0 - small - 4.0), title, text, pal::TEXT, Align2::LEFT_BOTTOM);
        f.text_px(Vec2::new(sr.0.x - 34.0, sr.0.y - 10.0), sub, small, pal::DIM, Align2::LEFT_BOTTOM);
        // the axes: decades up the side, layers along the bottom
        let (min, max) = p.rect;
        plot::line_of(f, plot::AXES, min, Vec2::new(min.x, max.y), 1.0, pal::DIM.a(0.8));
        plot::line_of(f, plot::AXES, Vec2::new(min.x, max.y), max, 1.0, pal::DIM.a(0.8));
        for e in -7..=0 {
            let y = p.px_y(e as f64);
            plot::line_of(f, plot::AXES, Vec2::new(min.x - 4.0, y), Vec2::new(min.x, y), 1.0, pal::DIM.a(0.8));
            plot::line_of(f, plot::AXES, Vec2::new(min.x, y), Vec2::new(max.x, y), 1.0, pal::DIM.a(0.12));
            if e % 2 == 0 {
                let name = if e == 0 { "1".to_string() } else { format!("$10^{{{}}}$", num(e as f64, 0)) };
                f.text_px(Vec2::new(min.x - 6.0, y), name, small, pal::DIM, Align2::RIGHT_CENTER);
            }
        }
        for l in 1..=STACK_DEPTH {
            let x = p.px_x(stack_x(l));
            plot::line_of(f, plot::AXES, Vec2::new(x, max.y), Vec2::new(x, max.y + 4.0), 1.0, pal::DIM.a(0.8));
            f.text_px(Vec2::new(x, max.y + 6.0), l.to_string(), small, pal::DIM, Align2::CENTER_TOP);
        }
        let axis = fitting(&["layer: the gradient goes back from the top (10) to the first", "layer, from the top back to the first", "layer"], p.width(), small);
        f.text_px(Vec2::new((min.x + max.x) * 0.5, max.y + 8.0 + small), axis, small, pal::DIM, Align2::CENTER_TOP);
        // what the top layer is handed
        p.hline(f, DASHED, 0.0, 1.0, pal::DIM.a(0.5), true);
        // the nine, the chosen one bright and on top
        let alpha = &self.look.stack.shown;
        // (every curve faint, in an order that never changes, and each again on top
        // in pink as bright as it is chosen)
        for kind in [plot::CURVES, STACK_TOP] {
            for (i, prof) in self.profiles.iter().enumerate() {
                let Some(prof) = prof else { continue };
                let t = alpha.get(i).copied().unwrap_or(0.0).clamp(0.0, 1.0);
                let pts: Vec<(f64, f64)> = (1..=STACK_DEPTH).map(|l| (stack_x(l), prof.grad[l - 1].max(1e-300).log10())).collect();
                if kind == plot::CURVES {
                    p.path(f, kind, pts.iter().copied(), 1.1, pal::DIM.a(0.45));
                } else {
                    p.path(f, kind, pts.iter().copied(), 2.6, BACKWARD.a(t));
                    for &(x, y) in &pts {
                        p.dot(f, x, y, 2.0 + 4.5 * t, pal::DIM.a(0.6).mix(BACKWARD, t));
                    }
                }
            }
        }
        // their names, at the first layer, kept apart
        let line = small + 3.0;
        let mut names: Vec<(f32, usize, f64)> = self.profiles.iter().enumerate().filter_map(|(i, prof)| prof.map(|q| (p.px_y(q.grad[0].max(1e-300).log10()), i, q.grad[0]))).collect();
        names.sort_by(|a, b| a.0.total_cmp(&b.0));
        let ys = spread(&names.iter().map(|n| n.0).collect::<Vec<_>>(), line, min.y - 6.0, max.y + 6.0);
        for ((_, i, first), y) in names.iter().zip(ys) {
            let t = alpha.get(*i).copied().unwrap_or(0.0).clamp(0.0, 1.0);
            let color = pal::DIM.mix(BACKWARD, t);
            let label = format!("{} ${}$", Act::ALL[*i].name(), sig2(*first));
            f.text_px(Vec2::new(max.x + 10.0, y), label, small, color, Align2::LEFT_CENTER);
        }
    }

    fn draw_training(&mut self, f: &mut Frame, lay: &Layout) {
        let (Some(mr), Some(cr)) = (lay.map, lay.chart) else { return };
        let (text, small) = (lay.text, lay.small);
        let title_y = mr.0.y - 8.0;
        // the backing, round the map (a picture is drawn on the plane, behind any panel)
        let outer = (Vec2::new(mr.0.x - if lay.narrow { 16.0 } else { 56.0 }, title_y - text - 10.0), Vec2::new(lay.view.x - if lay.narrow { 6.0 } else { 14.0 }, cr.1.y + 30.0));
        panel_round(f, outer, (mr.0 - Vec2::ONE, mr.1 + Vec2::ONE));
        let r = &self.run;
        let name = r.act.name();
        let head = format!("three layers of 16 {name} units learning the spirals: each unit's gate (row) for each point (column)");
        let shorter = format!("three layers of 16 {name} units learning the spirals");
        let shortest = format!("{name} units learning the spirals");
        let buttons_x = lay.run_buttons.first().map_or(lay.view.x, |b| b.0.x);
        let left = outer.0.x + 6.0;
        let title = fitting(&[&head, &shorter, &shortest], buttons_x - left - 10.0, text).to_string();
        f.text_px(Vec2::new(left, title_y), title, text, pal::TEXT, Align2::LEFT_BOTTOM);
        // the map
        self.paint(f);
        let rows = (ALL_UNITS + LAYERS - 1) as f32;
        let row_h = (mr.1.y - mr.0.y) / rows;
        if let Some(image) = &mut self.map {
            let (lo, hi) = world_rect(f, mr.0, mr.1);
            f.image(image, lo, hi, 0.0);
        }
        hud_outline(f, mr.0 - Vec2::ONE, mr.1 + Vec2::ONE, 1.0, pal::DIM.a(0.5));
        let r = &self.run;
        for l in 0..LAYERS {
            let y0 = mr.0.y + (l * (UNITS + 1)) as f32 * row_h;
            let y1 = y0 + UNITS as f32 * row_h;
            let label = if lay.narrow { format!("{}", l + 1) } else { format!("layer {}", l + 1) };
            f.text_px(Vec2::new(mr.0.x - 9.0, (y0 + y1) * 0.5), label, small, pal::DIM, Align2::RIGHT_CENTER);
            for j in 0..UNITS {
                if r.dead[l * UNITS + j] {
                    let y = y0 + j as f32 * row_h;
                    hud_rect(f, Vec2::new(mr.0.x - 6.0, y), Vec2::new(mr.0.x - 2.0, y + row_h.max(1.0)), DEAD);
                }
            }
        }
        // the chart: points right, gates shut, units dead, over the steps
        let cp = Plot::new(cr, [0.0, STEPS as f64], [0.0, 1.0]);
        // (the heading names the three lines in their colours, rather than labels at
        // their ends, which would leap about with the share right; plain words, which
        // cost less to set than formulas, since the numbers change at every step)
        let right = if r.same { "right: none, the same output for every point".to_string() } else { format!("right {}", percent(r.right, 0)) };
        let dead = r.dead_count();
        let shut = percent(r.shut, 0);
        let long = if r.loss.is_finite() {
            vec![
                (format!("step {} of {STEPS}, loss {}: ", r.step, num(r.loss, 3)), pal::TEXT),
                (right.clone(), RIGHT),
                (", ".to_string(), pal::TEXT),
                (format!("gates shut {shut}"), SHUT_GREY),
                (", ".to_string(), pal::TEXT),
                (format!("units dead {dead} of {ALL_UNITS}"), DEAD),
            ]
        } else {
            vec![(format!("step {} of {STEPS}: the loss has blown up (the steps are too large)", r.step), pal::TEXT)]
        };
        let short = if r.loss.is_finite() {
            vec![(right, RIGHT), (", ".to_string(), pal::TEXT), (format!("shut {shut}"), SHUT_GREY), (", ".to_string(), pal::TEXT), (format!("dead {dead}"), DEAD)]
        } else {
            vec![("the loss has blown up".to_string(), pal::TEXT)]
        };
        let width = |parts: &[(String, Color)]| parts.iter().map(|(t, _)| f.text_size(t, small).x).sum::<f32>();
        let parts = if width(&long) <= cp.width() + 30.0 { long } else { short };
        let mut x = cr.0.x - 30.0;
        for (t, c) in parts {
            let w = f.text_size(&t, small).x;
            f.text_px(Vec2::new(x, cr.0.y - 6.0), t, small, c, Align2::LEFT_BOTTOM);
            x += w;
        }
        let (min, max) = cp.rect;
        plot::line_of(f, plot::AXES, min, Vec2::new(min.x, max.y), 1.0, pal::DIM.a(0.8));
        for v in [0.0, 0.5, 1.0] {
            let y = cp.px_y(v);
            plot::line_of(f, plot::AXES, Vec2::new(min.x - 4.0, y), Vec2::new(min.x, y), 1.0, pal::DIM.a(0.8));
            plot::line_of(f, plot::AXES, Vec2::new(min.x, y), Vec2::new(max.x, y), 1.0, pal::DIM.a(0.12));
            f.text_px(Vec2::new(min.x - 6.0, y), percent(v, 0), small, pal::DIM, Align2::RIGHT_CENTER);
        }
        cp.x_axis(f, "", small);
        f.text_px(Vec2::new(max.x, max.y + 8.0 + small), "steps", small, pal::DIM, Align2::RIGHT_TOP);
        for (history, kind, color) in [(&r.rights, RUN_RIGHT, RIGHT), (&r.shuts, RUN_SHUT, SHUT_GREY), (&r.deads, RUN_DEAD, DEAD)] {
            let every = history.every() as f64;
            cp.path(f, kind, history.points().iter().enumerate().map(|(k, &v)| (k as f64 * every, v as f64)), 1.8, color);
        }
        // how to read the map, under it
        let key = fitting(
            &["pink: the gate is open, black: shut, blue: below 0; red: a dead unit", "pink: open, black: shut, blue: below 0", "pink open, black shut", ""],
            mr.1.x - mr.0.x + 10.0,
            small,
        );
        f.text_px(Vec2::new(mr.0.x, mr.1.y + 5.0), key, small, pal::DIM, Align2::LEFT_TOP);
    }

    /// Paint the map of the run's gates, if it has changed since it was last painted.
    fn paint(&mut self, f: &mut Frame) {
        let key = (self.runs, self.run.step);
        let (w, h) = (POINTS as u32, (ALL_UNITS + LAYERS - 1) as u32);
        let image = self.map.get_or_insert_with(|| Image::new(f.gfx, w, h));
        if image.size() != [w, h] {
            image.ensure_size(f.gfx, w, h);
            self.painted = None;
        }
        // (every other step while the run goes on: it changes little from one to the
        // next, and painting is half of what a frame costs)
        let behind = match self.painted {
            Some((run, step)) => run != key.0 || key.1 >= step + 2 || (key.1 != step && self.run.finished()),
            None => true,
        };
        if image.size() != [w, h] || !behind {
            return;
        }
        self.run.gates(&mut self.gates);
        let pixels = image.pixels_mut();
        pixels.fill(Color::CLEAR);
        for u in 0..ALL_UNITS {
            // (from the top: layer 1 first, a row of gap between layers; the pixels go
            // from the bottom)
            let row = u + u / UNITS;
            let from_bottom = h as usize - 1 - row;
            for k in 0..POINTS {
                if let (Some(p), Some(&g)) = (pixels.get_mut(from_bottom * POINTS + k), self.gates.get(u * POINTS + k)) {
                    *p = gate_tint(&self.tints, g);
                }
            }
        }
        self.painted = Some(key);
    }
}

/// A panel's backing over `outer`, but for the hole (a picture is drawn on the plane,
/// behind any panel, and would be covered).
fn panel_round(f: &mut Frame, outer: Rect, hole: Rect) {
    let (o0, o1) = outer;
    let (h0, h1) = (hole.0.max(o0), hole.1.min(o1));
    if !(h1.x > h0.x && h1.y > h0.y) {
        hud_panel(f, o0, o1);
        return;
    }
    hud_panel(f, o0, Vec2::new(o1.x, h0.y));
    hud_panel(f, Vec2::new(o0.x, h1.y), o1);
    hud_panel(f, Vec2::new(o0.x, h0.y), Vec2::new(h0.x, h1.y));
    hud_panel(f, Vec2::new(h1.x, h0.y), Vec2::new(o1.x, h1.y));
}

/// The world rectangle under a rectangle of the screen (logical pixels).
fn world_rect(f: &Frame, min: Vec2, max: Vec2) -> (Vec2, Vec2) {
    let (lo, hi) = f.cam.flat_bounds();
    let view = f.view_size().max(Vec2::ONE);
    let to = |p: Vec2| Vec2::new(lo.x + p.x / view.x * (hi.x - lo.x), hi.y - p.y / view.y * (hi.y - lo.y));
    let (a, b) = (to(min), to(max));
    (Vec2::new(a.x, b.y), Vec2::new(b.x, a.y))
}

/// The z axis of the activation's chart, along its zero (the chart's own x axis is the
/// gate's, under it).
fn line_of_axis(f: &mut Frame, a: &Plot) {
    let y = a.px_y(0.0);
    plot::line_of(f, plot::AXES, Vec2::new(a.rect.0.x, y), Vec2::new(a.rect.1.x, y), 1.0, pal::DIM.a(0.5));
    let x = a.px_x(0.0);
    plot::line_of(f, plot::AXES, Vec2::new(x, a.rect.0.y), Vec2::new(x, a.rect.1.y), 1.0, pal::DIM.a(0.3));
}

/// Places for labels wanted at `ys` (in order, top to bottom), at least `gap` apart and
/// inside `lo` to `hi`, each moved as little as it can be.
fn spread(ys: &[f32], gap: f32, lo: f32, hi: f32) -> Vec<f32> {
    let n = ys.len();
    let mut out: Vec<f32> = ys.to_vec();
    if n == 0 {
        return out;
    }
    // clusters that overlap are pushed apart about their middle, until none overlap
    for _ in 0..n * n + 1 {
        let mut moved = false;
        for i in 1..n {
            if out[i] - out[i - 1] < gap - 1e-3 {
                // the run of labels packed up against this one, centred on its wish
                let (mut a, mut b) = (i - 1, i);
                while a > 0 && out[a] - out[a - 1] < gap + 1e-3 {
                    a -= 1;
                }
                while b + 1 < n && out[b + 1] - out[b] < gap + 1e-3 {
                    b += 1;
                }
                let wish = ys[a..=b].iter().sum::<f32>() / (b - a + 1) as f32;
                let first = wish - 0.5 * gap * (b - a) as f32;
                for (k, y) in out[a..=b].iter_mut().enumerate() {
                    *y = first + k as f32 * gap;
                }
                moved = true;
            }
        }
        if !moved {
            break;
        }
    }
    // kept inside, pushed in from the ends
    let total = gap * (n - 1) as f32;
    let shift_down = (lo - out[0]).max(0.0);
    let shift_up = (out[n - 1] - hi).max(0.0);
    if total <= hi - lo {
        let d = shift_down - shift_up;
        out.iter_mut().for_each(|y| *y += d);
        for i in 1..n {
            out[i] = out[i].max(out[i - 1] + gap);
        }
        for i in (0..n - 1).rev() {
            out[i] = out[i].min(out[i + 1] - gap);
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Averages of the bell by drawing from it: (E f', E f'², shut, below 0, E f) and
    /// the standard errors of the first two and the last.
    fn sampled(act: Act, mu: f64, sigma: f64, n: usize, seed: u64) -> ([f64; 5], [f64; 3]) {
        let mut rng = Rng::new(seed);
        let (mut s, mut s2) = ([0.0f64; 5], [0.0f64; 3]);
        for _ in 0..n {
            // (two uniform draws through Box and Muller in double precision)
            let (u, v) = (rng.next_u64() as f64 / u64::MAX as f64, rng.next_u64() as f64 / u64::MAX as f64);
            let t = (-2.0 * u.max(1e-300).ln()).sqrt() * (std::f64::consts::TAU * v).cos();
            let z = mu + sigma * t;
            let g = act.df(z);
            let vals = [g, g * g, (g.abs() < SHUT) as u8 as f64, (g < 0.0) as u8 as f64, act.f(z)];
            for k in 0..5 {
                s[k] += vals[k];
            }
            for (k, &i) in [0usize, 1, 4].iter().enumerate() {
                s2[k] += vals[i] * vals[i];
            }
        }
        let m = s.map(|v| v / n as f64);
        let se = [0, 1, 4].map(|i| i).iter().enumerate().fold([0.0; 3], |mut acc, (k, &i)| {
            acc[k] = ((s2[k] / n as f64 - m[i] * m[i]).max(0.0) / n as f64).sqrt();
            acc
        });
        (m, se)
    }

    #[test]
    fn the_averages_agree_with_drawing_from_the_bell() {
        let cells = Cells::new();
        for &act in Act::ALL.iter() {
            let sets = Sets::of(act);
            for &(mu, sigma) in &[(0.0, 1.0), (-1.0, 0.5), (0.0, 3.0), (2.0, 0.7), (-4.0, 2.5), (0.3, 0.05)] {
                let a = averages(act, &sets, mu, sigma);
                let (m, se) = sampled(act, mu, sigma, 200_000, 11);
                let name = act.name();
                for (k, (got, want, err)) in [(a.gate, m[0], se[0]), (a.gate_sq, m[1], se[1]), (a.output, m[4], se[2])].into_iter().enumerate() {
                    assert!((got - want).abs() <= 5.0 * err + 1e-9, "{name} at ({mu}, {sigma}), average {k}: {got} by integration, {want} ± {err} by drawing");
                }
                for (got, want, what) in [(a.shut, m[2], "shut"), (a.negative, m[3], "below 0")] {
                    let p = got.max(want);
                    let err = (p * (1.0 - p) / 200_000.0).sqrt() + 2.0 / 200_000.0;
                    assert!((got - want).abs() <= 5.0 * err, "{name} at ({mu}, {sigma}): {what} {got} against {want}");
                }
                // the histogram holds every input, and its shut part is the shut share
                let h = histogram(act, mu, sigma, &cells);
                let total: f64 = h.open.iter().chain(&h.shut).sum();
                assert!((total - 1.0).abs() < 1e-9, "{name}: the histogram holds {total}");
                let shut: f64 = h.shut.iter().sum();
                assert!((shut - a.shut).abs() < 0.004, "{name} at ({mu}, {sigma}): shut {shut} in the bars, {} by the intervals", a.shut);
            }
        }
    }

    /// Where the closed forms are known, the integration gives them.
    #[test]
    fn the_averages_have_their_closed_forms() {
        let close = |a: f64, b: f64| (a - b).abs() < 1e-9;
        for &(mu, sigma) in &[(0.0, 1.0), (-1.0, 0.5), (1.5, 2.0), (0.0, 0.0), (-0.7, 0.0)] {
            let id = averages(Act::Identity, &Sets::of(Act::Identity), mu, sigma);
            assert!(close(id.gate, 1.0) && close(id.gate_sq, 1.0) && id.shut == 0.0 && close(id.output, mu), "{id:?}");
            let r = averages(Act::Relu, &Sets::of(Act::Relu), mu, sigma);
            let open = if sigma > 0.0 { normal_cdf(mu / sigma) } else { (mu > 0.0) as u8 as f64 };
            let mean = if sigma > 0.0 { mu * open + sigma * (-0.5 * (mu / sigma).powi(2)).exp() / std::f64::consts::TAU.sqrt() } else { mu.max(0.0) };
            assert!(close(r.gate, open) && close(r.gate_sq, open) && close(r.shut, 1.0 - open) && close(r.output, mean), "ReLU at ({mu}, {sigma}): {r:?}");
        }
        // GELU's and SiLU's slopes at z and −z add up to 1, so a centred bell averages 1/2
        for act in [Act::Gelu, Act::Silu] {
            for sigma in [0.3, 1.0, 3.0] {
                let a = averages(act, &Sets::of(act), 0.0, sigma);
                assert!(close(a.gate, 0.5), "{}: {}", act.name(), a.gate);
            }
        }
        // the sigmoid's gate is at most a quarter, and that is reached only at 0
        let s = Sets::of(Act::Sigmoid);
        assert!((averages(Act::Sigmoid, &s, 0.0, 0.0).gate - 0.25).abs() < 1e-15);
        for &(mu, sigma) in &[(0.0, 0.01), (0.0, 1.0), (3.0, 1.0)] {
            assert!(averages(Act::Sigmoid, &s, mu, sigma).gate < 0.25);
        }
    }

    #[test]
    fn the_shut_and_negative_sets_are_where_they_should_be() {
        let near = |a: f64, b: f64| (a - b).abs() < 1e-9;
        let relu = Sets::of(Act::Relu);
        assert!(relu.shut.len() == 1 && relu.shut[0].0 == f64::NEG_INFINITY && near(relu.shut[0].1, 0.0) && relu.negative.is_empty(), "{relu:?}");
        assert!(Sets::of(Act::LeakyRelu).shut.is_empty() && Sets::of(Act::Identity).shut.is_empty());
        // tanh: 1 − tanh² z < 0.01 beyond atanh √0.99
        let edge = 0.99f64.sqrt().atanh();
        let tanh = Sets::of(Act::Tanh);
        assert!(tanh.shut.len() == 2 && near(tanh.shut[0].1, -edge) && near(tanh.shut[1].0, edge) && tanh.shut[1].1 == f64::INFINITY, "{tanh:?}");
        // the sigmoid: σ(1 − σ) < 0.01 beyond ln((1 + √0.96) / (1 − √0.96))
        let edge = ((1.0 + 0.96f64.sqrt()) / (1.0 - 0.96f64.sqrt())).ln();
        let sig = Sets::of(Act::Sigmoid);
        assert!(sig.shut.len() == 2 && near(sig.shut[1].0, edge), "{sig:?}");
        // GELU's slope is below 0 from far out up to −0.7518, and shut near that root
        let gelu = Sets::of(Act::Gelu);
        let last = gelu.negative.last().copied().unwrap_or((0.0, 0.0));
        assert!((last.1 + 0.7518).abs() < 1e-3, "{gelu:?}");
        assert!(gelu.shut.iter().any(|&(a, b)| a < -0.7518 && b > -0.7518 && b - a < 0.1), "{gelu:?}");
    }

    /// The gate is the slope: what comes back to a unit's input is its slope times what
    /// reaches its output, entry by entry, in the network that learns and in the stack.
    #[test]
    fn the_gradient_through_a_gate_is_the_slope_times_what_reaches_it() {
        for &act in Act::ALL.iter() {
            let run = Run::new(act, RATE, BIAS);
            for l in 1..=LAYERS {
                let (Some(z), Some(a)) = (run.trace.get(Stage::PreAct, l), run.trace.get(Stage::PostAct, l)) else { panic!("no layer {l}") };
                for ((&gz, &ga), &v) in run.tape.grad(z).iter().zip(run.tape.grad(a)).zip(run.tape.value(z)) {
                    assert!((gz - act.df(v) * ga).abs() <= 1e-12 * (1.0 + ga.abs()), "{}: {gz} against {} × {ga}", act.name(), act.df(v));
                }
            }
        }
    }

    /// The loss of the network that learns, with one parameter moved by `d`.
    fn loss_moved(run: &Run, group: usize, entry: usize, d: f64) -> f64 {
        let mut model = run.model.clone();
        model.groups_mut()[group].value[entry] += d;
        let mut tape = Tape::new();
        let trace = model.loss_mlp(&mut tape, &run.x, &run.y, Loss::CrossEntropy);
        tape.value(trace.loss)[0]
    }

    /// The gradient the network that learns steps down is the loss's: central
    /// differences of the same loss agree with it, entry by entry, for a sample of every
    /// group's parameters.
    #[test]
    fn the_steps_go_down_the_gradient_of_the_loss() {
        for &act in Act::ALL.iter() {
            for &bias in &[BIAS, -1.0] {
                let mut run = Run::new(act, RATE, bias);
                for _ in 0..5 {
                    run.step();
                }
                let mut model = run.model.clone();
                model.zero_grads();
                model.accumulate(&run.tape, &run.trace);
                for (gi, g) in model.groups().iter().enumerate() {
                    for e in (0..g.value.len()).step_by((g.value.len() / 5).max(1)) {
                        let h = 1e-5;
                        let numeric = (loss_moved(&run, gi, e, h) - loss_moved(&run, gi, e, -h)) / (2.0 * h);
                        let exact = g.grad[e];
                        assert!((numeric - exact).abs() <= 1e-6 + 1e-4 * exact.abs(), "{} {}: {} entry {e}: {exact} against {numeric}", act.name(), bias, g.name);
                    }
                }
            }
        }
    }

    /// The ten layers' gradient at a layer's inputs is the derivative of the same loss:
    /// moving one input z of layer l by ±h and carrying it through the layers above gives
    /// the same number.
    #[test]
    fn the_ten_layers_gradient_is_the_loss_s() {
        for &act in Act::ALL.iter() {
            let spec = MlpSpec {
                inputs: STACK_WIDTH,
                width: STACK_WIDTH,
                depth: STACK_DEPTH,
                outputs: 1,
                act,
                norm: Norm::None,
                place: Place::Pre,
                residual: false,
                residual_scale: 1.0,
                init: Init::Lecun,
                gain: 1.0,
                bias: true,
            };
            let mut rng = Rng::new(0x5_7AC4);
            let model = Model::mlp(&spec, &mut rng);
            let n = STACK_BATCH * STACK_WIDTH;
            let x: Vec<f64> = (0..n).map(|_| rng.normal() as f64).collect();
            let seed: Vec<f64> = (0..n).map(|_| rng.normal() as f64).collect();
            let mut tape = Tape::new();
            let trace = model.forward_mlp(&mut tape, &x);
            tape.backward_seeded(trace.get(Stage::Output, STACK_DEPTH).unwrap(), &seed);
            // the loss Σ seed · (the top layer's outputs), from layer l's inputs up
            let w = |l: usize| &model.group(&format!("layer {l} weights")).unwrap().value;
            let loss_from = |l: usize, z: &[f64]| -> f64 {
                let mut rows: Vec<Vec<f64>> = z.chunks(STACK_WIDTH).map(|r| r.iter().map(|&v| act.f(v)).collect()).collect();
                for up in l + 1..=STACK_DEPTH {
                    let wv = w(up);
                    rows = rows.iter().map(|a| (0..STACK_WIDTH).map(|i| act.f((0..STACK_WIDTH).map(|j| wv[i * STACK_WIDTH + j] * a[j]).sum::<f64>())).collect()).collect();
                }
                rows.iter().flatten().zip(&seed).map(|(a, s)| a * s).sum()
            };
            for l in [1, 4, STACK_DEPTH] {
                let zv = trace.get(Stage::PreAct, l).unwrap();
                let (z, grad) = (tape.value(zv).to_vec(), tape.grad(zv).to_vec());
                for e in [0, 37, 500, n - 1] {
                    let h = 1e-6 * (1.0 + z[e].abs());
                    let (mut up, mut down) = (z.clone(), z.clone());
                    up[e] += h;
                    down[e] -= h;
                    let numeric = (loss_from(l, &up) - loss_from(l, &down)) / (2.0 * h);
                    // (ReLU's slope jumps at 0: an input that close to it is not differentiable)
                    if z[e].abs() < 10.0 * h && matches!(act, Act::Relu | Act::LeakyRelu | Act::Elu) {
                        continue;
                    }
                    assert!((numeric - grad[e]).abs() <= 1e-6 * (1.0 + grad[e].abs()), "{} layer {l} entry {e}: {} against {numeric}", act.name(), grad[e]);
                }
            }
        }
    }

    #[test]
    fn nothing_breaks_at_the_ends_of_the_settings() {
        let mut s = ActGrad::create();
        for &act in Act::ALL.iter() {
            for &(mu, sigma) in &[(-MU_MOST, 0.0), (MU_MOST, SIGMA_MOST), (0.0, 0.0), (f64::NAN, f64::NAN), (1e9, -3.0), (MU_MOST, 0.0)] {
                (s.act, s.mu, s.sigma) = (act, mu, sigma);
                s.refresh();
                let a = s.avg;
                for v in [a.gate, a.gate_sq, a.shut, a.negative, a.output] {
                    assert!(v.is_finite(), "{}: {a:?} at ({mu}, {sigma})", act.name());
                }
                s.follow(0.016, false);
                assert!(s.look.bars.shown.iter().chain(&s.look.curves.shown).all(|v| v.is_finite()));
            }
        }
        // every window lays out without a number that is not finite
        for &(w, h) in &[(1600.0, 1000.0), (1220.0, 960.0), (1100.0, 700.0), (720.0, 700.0), (390.0, 800.0), (390.0, 400.0), (240.0, 180.0), (160.0, 900.0), (2400.0, 300.0)] {
            for lower in [Lower::Stack, Lower::Training] {
                let lay = Layout::of(Vec2::new(w, h), 40.0, lower);
                let rects = [lay.act_plot, lay.gate_plot, lay.hist, lay.stack, lay.map, lay.chart];
                for r in rects.iter().flatten().chain(&lay.acts).chain(&lay.tabs).chain(&lay.run_buttons) {
                    assert!(r.0.is_finite() && r.1.is_finite() && r.1.x > r.0.x && r.1.y > r.0.y, "{w} by {h}: {r:?}");
                }
            }
        }
        // a run at the largest rate, from the worst start, stays a run
        let mut run = Run::new(Act::Elu, 3.0, -2.0);
        for _ in 0..200 {
            run.step();
        }
        assert!(run.dead_count() <= ALL_UNITS);
    }

    /// The scene as an experiment sets it up, with the ten layers worked out.
    fn experiment(k: usize) -> ActGrad {
        let mut s = ActGrad::create();
        s.set_up(k);
        s.ensure_profiles(f64::INFINITY);
        s
    }

    /// A run to its end: the first step at which it is 95 % right, if any, and whether
    /// any unit ever came back from the dead (in each layer).
    fn to_the_end(run: &mut Run) -> (Option<usize>, [usize; LAYERS]) {
        let mut first = (run.right >= 0.95).then_some(0);
        let mut was = run.dead;
        let mut revived = [0; LAYERS];
        while !run.finished() {
            run.step();
            for u in 0..ALL_UNITS {
                if was[u] && !run.dead[u] {
                    revived[u / UNITS] += 1;
                }
            }
            was = run.dead;
            if first.is_none() && run.right >= 0.95 {
                first = Some(run.step);
            }
        }
        (first, revived)
    }

    fn within(v: f64, lo: f64, hi: f64) -> bool {
        v >= lo && v <= hi
    }

    fn profile_of(s: &ActGrad, act: Act) -> Profile {
        s.profiles[index_of(act)].expect("worked out")
    }

    #[test]
    #[ignore] // (about ten seconds: five runs of 1500 steps)
    fn the_sigmoid_passes_at_most_a_quarter() {
        let mut s = experiment(0);
        assert_eq!((s.act, s.mu, s.sigma, s.lower), (Act::Sigmoid, 0.0, 1.0, Lower::Training));
        let sets = Sets::of(Act::Sigmoid);
        assert_eq!(averages(Act::Sigmoid, &sets, 0.0, 0.0).gate, 0.25);
        assert!(within(s.avg.gate_sq, 0.0445, 0.0455) && within(1.0 / s.avg.gate_sq, 21.5, 22.5), "{:?}", s.avg);
        assert!(within(profile_of(&s, Act::Sigmoid).grad[0], 4.45e-7, 4.55e-7));
        to_the_end(&mut s.run);
        assert!(within(s.run.right, 0.365, 0.375), "the sigmoid's run: {}", s.run.right);
        let (tanh, _) = to_the_end(&mut Run::new(Act::Tanh, s.rate, s.bias));
        assert_eq!(tanh, Some(475));
    }

    #[test]
    #[ignore] // (a few seconds)
    fn tanh_shuts_when_its_inputs_are_large() {
        let mut s = experiment(1);
        assert_eq!((s.act, s.mu, s.sigma, s.lower), (Act::Tanh, 0.0, 3.0, Lower::Training));
        let edge = s.sets[index_of(Act::Tanh)].shut[1].0;
        assert!((edge - 2.99).abs() < 0.005, "{edge}");
        assert!(within(s.avg.shut, 0.315, 0.325) && within(s.avg.gate_sq, 0.165, 0.175), "{:?}", s.avg);
        s.sigma = 0.5;
        s.refresh();
        assert!(s.avg.shut < 5e-4 && within(s.avg.gate_sq, 0.715, 0.725), "{:?}", s.avg);
        assert_eq!(s.run.shut, 0.0);
        let (first, _) = to_the_end(&mut s.run);
        assert_eq!(first, Some(475));
        assert!(within(s.run.shut, 0.105, 0.115), "{}", s.run.shut);
    }

    #[test]
    #[ignore] // (a few seconds)
    fn relu_is_open_or_shut_and_can_die() {
        let mut s = experiment(2);
        assert_eq!((s.act, s.mu, s.sigma, s.bias, s.lower), (Act::Relu, -1.0, 0.5, -1.0, Lower::Training));
        assert!(within(s.avg.shut, 0.975, 0.985), "{:?}", s.avg);
        let per_layer = |r: &Run| (0..LAYERS).map(|l| r.dead_in(l)).collect::<Vec<_>>();
        assert_eq!(per_layer(&s.run), [9, 13, 16]);
        let start = s.run.dead;
        // a dead unit's weights and bias get no gradient at all
        let mut model = s.run.model.clone();
        model.zero_grads();
        model.accumulate(&s.run.tape, &s.run.trace);
        for l in 0..LAYERS {
            let (w, b) = (model.group(&format!("layer {} weights", l + 1)).unwrap(), model.group(&format!("layer {} bias", l + 1)).unwrap());
            for j in 0..UNITS {
                if start[l * UNITS + j] {
                    assert!(b.grad[j] == 0.0 && w.grad[j * w.cols..(j + 1) * w.cols].iter().all(|&g| g == 0.0));
                }
            }
        }
        let (first, revived) = to_the_end(&mut s.run);
        assert_eq!((first, revived, s.run.dead), (None, [0; LAYERS], start));
        assert!(s.run.same && (s.run.loss - 3f64.ln()).abs() < 1e-3 && (3f64.ln() - 1.10).abs() < 0.005, "{}", s.run.loss);
        let (elu, _) = to_the_end(&mut Run::new(Act::Elu, s.rate, s.bias));
        assert_eq!(elu, Some(522));
        // leaky ReLU: nearly all its gates are 0.01, at the start and at the end
        let mut leaky = Run::new(Act::LeakyRelu, s.rate, s.bias);
        let leaking = |r: &Run| {
            let mut g = Vec::new();
            r.gates(&mut g);
            g.iter().filter(|&&g| g == 0.01).count() as f64 / g.len() as f64
        };
        let at_start = leaking(&leaky);
        to_the_end(&mut leaky);
        assert!(leaky.dead_count() == 0 && within(leaky.right, 0.325, 0.335), "{}", leaky.right);
        assert!(at_start > 0.95 && leaking(&leaky) > 0.95, "{at_start} {}", leaking(&leaky));
    }

    #[test]
    #[ignore] // (a few seconds)
    fn gelu_and_silu_leak_a_little() {
        let mut s = experiment(3);
        assert_eq!((s.act, s.mu, s.sigma, s.bias, s.lower), (Act::Gelu, 0.0, 1.0, -1.0, Lower::Training));
        // the root, the least slope and the most
        let gelu = &s.sets[index_of(Act::Gelu)];
        let root = gelu.negative.last().unwrap().1;
        assert!((root + 0.75).abs() < 0.005, "{root}");
        let (least, most) = (Act::Gelu.df(-std::f64::consts::SQRT_2), Act::Gelu.df(std::f64::consts::SQRT_2));
        assert!((least + 0.13).abs() < 0.005 && (most - 1.13).abs() < 0.005, "{least} {most}");
        for k in 0..2000 {
            let z = -6.0 + 12.0 * k as f64 / 1999.0;
            assert!(Act::Gelu.df(z) >= least - 1e-12 && Act::Gelu.df(z) <= most + 1e-12);
            // 0 only at the root, slightly below 0 below it, above 1 beyond 0.75
            assert!((Act::Gelu.df(z) < 0.0) == (z < root) || (z - root).abs() < 1e-6);
            assert!((Act::Gelu.df(z) > 1.0) == (z > -root) || (z + root).abs() < 1e-6);
        }
        assert!(within(s.avg.negative, 0.225, 0.235), "{:?}", s.avg);
        // as many above 1: the slopes at z and −z add up to 1, and the bell is centred
        let above = intervals(|z| Act::Gelu.df(z) > 1.0);
        assert!((share(&above, 0.0, 1.0, false) - s.avg.negative).abs() < 1e-9);
        let silu = Sets::of(Act::Silu);
        assert!((silu.negative.last().unwrap().1 + 1.28).abs() < 0.005);
        let (_, revived) = to_the_end(&mut s.run);
        assert!(s.run.dead_count() == 0 && revived == [0; LAYERS] && within(s.run.right, 0.735, 0.745), "{}", s.run.right);
        let (first, _) = to_the_end(&mut Run::new(Act::Silu, s.rate, s.bias));
        assert_eq!(first, Some(1324));
    }

    #[test]
    fn the_same_ten_layers_with_each_activation() {
        let s = experiment(4);
        assert_eq!((s.act, s.mu, s.sigma, s.lower), (Act::Gelu, 0.0, 0.15, Lower::Stack));
        for &act in Act::ALL.iter() {
            let p = profile_of(&s, act);
            // the mean square, layer by layer, goes as the gates' average square
            for l in 0..STACK_DEPTH - 1 {
                let ratio = (p.grad[l] / p.grad[l + 1]).powi(2);
                assert!((ratio / p.gate_sq[l] - 1.0).abs() < 0.2, "{} layer {}: {ratio} against {}", act.name(), l + 1, p.gate_sq[l]);
            }
        }
        let relu = profile_of(&s, Act::Relu);
        assert!(relu.gate_sq.iter().all(|&g| within(g, 0.44, 0.53)) && within(relu.grad[0], 0.0265, 0.0275), "{relu:?}");
        let gelu = profile_of(&s, Act::Gelu);
        assert!(within(gelu.z[0], 0.975, 0.985) && within(gelu.z[STACK_DEPTH - 1], 0.0025, 0.0035) && within(gelu.grad[0], 0.0015, 0.0025), "{gelu:?}");
        for l in 4..STACK_DEPTH - 1 {
            assert!(within((gelu.grad[l] / gelu.grad[l + 1]).powi(2), 0.23, 0.28), "GELU layer {}", l + 1);
        }
        assert!(within(s.avg.gate_sq, 0.255, 0.265), "{:?}", s.avg);
        let (tanh, elu, sig) = (profile_of(&s, Act::Tanh), profile_of(&s, Act::Elu), profile_of(&s, Act::Sigmoid));
        assert!(within(tanh.grad[0], 0.24, 0.27) && within(elu.grad[0], 0.24, 0.27) && within(sig.grad[0], 4.45e-7, 4.55e-7));
        // the slope of each at 0
        assert!(Act::Tanh.df(0.0) == 1.0 && Act::Elu.df(0.0) == 1.0 && Act::Gelu.df(0.0) == 0.5);
    }

    /// The opening: ReLU's two bars, and a run that learns.
    #[test]
    fn the_opening_is_relu_open_or_shut() {
        let s = ActGrad::create();
        assert_eq!((s.act, s.mu, s.sigma, s.rate, s.bias), (Act::Relu, 0.0, 1.0, RATE, BIAS));
        assert!((s.avg.shut - 0.5).abs() < 1e-12 && (s.avg.gate - 0.5).abs() < 1e-9);
        let (open, shut) = (s.hist.open[bin_of(1.0)], s.hist.shut[bin_of(0.0)]);
        assert!((open - 0.5).abs() < 0.003 && (shut - 0.5).abs() < 0.003, "{open} {shut}");
    }

    /// The numbers behind the captions (run with --ignored --nocapture).
    #[test]
    #[ignore]
    fn explore() {
        let mut s = ActGrad::create();
        s.ensure_profiles(f64::INFINITY);
        for (i, p) in s.profiles.iter().enumerate() {
            let Some(p) = p else { continue };
            let factors: Vec<String> = (0..STACK_DEPTH - 1).map(|l| format!("{:.3}/{:.3}", (p.grad[l] / p.grad[l + 1]).powi(2), p.gate_sq[l])).collect();
            println!("{:<10} first {:.2e}, inputs' RMS {:.3} to {:.3}; mean square kept / gates' E f'² by layer: {}", Act::ALL[i].name(), p.grad[0], p.z[0], p.z[STACK_DEPTH - 1], factors.join(" "));
        }
        for &act in Act::ALL.iter() {
            let sets = Sets::of(act);
            for &(mu, sigma) in &[(0.0, 1.0), (0.0, 3.0), (0.0, 0.5), (-1.0, 0.5), (0.0, 0.15)] {
                let a = averages(act, &sets, mu, sigma);
                println!("{:<10} ({mu}, {sigma}): E f' {:.4}, E f'² {:.4}, shut {:.4}, below 0 {:.4}, E f {:.4}", act.name(), a.gate, a.gate_sq, a.shut, a.negative, a.output);
            }
        }
        for &bias in &[BIAS, -1.0] {
            for &act in Act::ALL.iter() {
                let mut run = Run::new(act, RATE, bias);
                let start: Vec<usize> = (0..LAYERS).map(|l| run.dead_in(l)).collect();
                let (first, revived) = to_the_end(&mut run);
                let end: Vec<usize> = (0..LAYERS).map(|l| run.dead_in(l)).collect();
                println!("{:<10} bias {bias:>4}: 95 % right at step {first:?}; at the end {:.3} right, loss {:.4}, shut {:.3}; dead {start:?} to {end:?}, revived {revived:?}", act.name(), run.right, run.loss, run.shut);
            }
        }
    }
}

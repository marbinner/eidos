//! Normalisation and the gradient: what batch, layer and RMS normalisation do to the
//! signal on the way forward, and how each reshapes the gradient on the way back.
//!
//! Four pictures, drawn on the screen, every number in them worked out by the library's
//! own operations (`math::autograd`) and networks (`math::arch`):
//!
//! - one example of two units through RMS normalisation, up close: the gradient that
//!   arrives at the output loses its part along the input on the way back, and the rest
//!   is divided by the input's size;
//! - a batch of five such examples through batch normalisation: a gradient that arrives
//!   at one example's output comes back to all five, and moving another example changes
//!   it;
//! - a plain ReLU network of twenty layers, with and without a normalisation: the size of
//!   its signal and of its gradient at every layer, as the gain of its starting weights
//!   is turned;
//! - one layer of that network with its weights scaled by c: the outputs stay as they
//!   were, and the gradient with respect to the weights is divided by c and stays at
//!   right angles to them.
//!
//! On a desktop the four are shown at once; on a phone, two at a time on two pages.

use super::*;
use crate::math::arch::{self, Init, Loss, MlpSpec, Model, Norm, Place, Stage, EPS};
use crate::math::autograd::{Act, Tape};
use crate::scenes::stats::plot::{self, Plot, Rect};
use glam::DVec2;
use std::collections::HashMap;
use std::f64::consts::SQRT_2;

pub const DEF: SceneDef = SceneDef {
    id: "normgrad",
    title: "Normalisation and the gradient",
    group: GROUP,
    blurb: "Batch, layer and RMS normalisation: what each does on the way forward, and how it reshapes the gradient on the way back.",
    make: |_| Box::new(NormGrad::new()),
};

// ----- one example of two units through RMS normalisation ------------------------------

/// The input of the first picture is kept between these lengths: at x = 0 there is no
/// direction to normalise, and the gradient would be divided by next to nothing (the
/// guard ε would stop it at about 316 times).
const X_LEAST: f64 = 0.7;
const X_MOST: f64 = 2.4;
/// The longest a gradient arriving at an output may be dragged.
const G_MOST: f64 = 1.2;

/// One example x of two units through RMS normalisation, y = x / rms(x), and a gradient
/// g arriving at y, worked out on the library's own operation.
#[derive(Clone, Copy, Debug)]
struct One {
    x: DVec2,
    /// rms(x) = √((x₁² + x₂²)/2 + ε): what x is divided by.
    rms: f64,
    y: DVec2,
    g: DVec2,
    /// g's part along x, and the rest, at right angles to x.
    along: DVec2,
    rest: DVec2,
    /// What comes back to x: the tape's gradient.
    back: DVec2,
}

fn pair(v: &[f64]) -> DVec2 {
    DVec2::new(v.first().copied().unwrap_or(f64::NAN), v.get(1).copied().unwrap_or(f64::NAN))
}

fn one(x: DVec2, g: DVec2, tape: &mut Tape) -> One {
    tape.clear();
    let xv = tape.param(1, 2, &[x.x, x.y]);
    let yv = tape.rms_norm(xv, EPS);
    tape.backward_seeded(yv, &[g.x, g.y]);
    let (y, back) = (pair(tape.value(yv)), pair(tape.grad(xv)));
    let u = x.normalize_or_zero();
    let along = u * g.dot(u);
    One { x, rms: (0.5 * x.length_squared() + EPS).sqrt(), y, g, along, rest: g - along, back }
}

/// x kept between `X_LEAST` and `X_MOST` long (in the direction of `before` if it has
/// none of its own), and a number in place of what is not one.
fn kept_x(x: DVec2, before: DVec2) -> DVec2 {
    let x = if x.is_finite() { x } else { before };
    let len = x.length();
    let dir = if len > 1e-9 { x / len } else { before.normalize_or(DVec2::X) };
    dir * len.max(X_LEAST).min(X_MOST)
}

/// A gradient kept to at most `G_MOST` long, and 0 for one that is not a number.
fn kept_g(g: DVec2) -> DVec2 {
    if !g.is_finite() {
        return DVec2::ZERO;
    }
    let len = g.length();
    if len > G_MOST {
        g * (G_MOST / len)
    } else {
        g
    }
}

// ----- a batch of five examples through batch normalisation ----------------------------

const MANY: usize = 5;

/// Five examples of two units through batch normalisation (each unit standardised over
/// the five), and a gradient g arriving at the first example's output alone.
#[derive(Clone, Copy, Debug)]
struct Many {
    x: [DVec2; MANY],
    y: [DVec2; MANY],
    /// Each unit's mean over the batch, and √(variance + ε), what it is divided by.
    mean: DVec2,
    sd: DVec2,
    g: DVec2,
    /// What comes back to each example.
    back: [DVec2; MANY],
}

fn many(x: &[DVec2; MANY], g: DVec2, tape: &mut Tape) -> Many {
    tape.clear();
    let data: Vec<f64> = x.iter().flat_map(|p| [p.x, p.y]).collect();
    let xv = tape.param(MANY, 2, &data);
    let yv = tape.batch_norm(xv, EPS);
    let mut seed = [0.0; 2 * MANY];
    (seed[0], seed[1]) = (g.x, g.y);
    tape.backward_seeded(yv, &seed);
    let (vals, grads) = (tape.value(yv), tape.grad(xv));
    let y = std::array::from_fn(|i| pair(vals.get(2 * i..).unwrap_or(&[])));
    let back = std::array::from_fn(|i| pair(grads.get(2 * i..).unwrap_or(&[])));
    let mean = x.iter().fold(DVec2::ZERO, |s, p| s + *p) / MANY as f64;
    let var = x.iter().fold(DVec2::ZERO, |s, p| s + (*p - mean) * (*p - mean)) / MANY as f64;
    Many { x: *x, y, mean, sd: (var + DVec2::splat(EPS)).sqrt(), g, back }
}

/// The part of the plane the five examples may be dragged in, and the part their
/// outputs are shown in (the same size, so that arrows in the two can be compared).
const IN_LO: DVec2 = DVec2::new(-0.4, -0.4);
const IN_HI: DVec2 = DVec2::new(4.0, 4.0);
const OUT_LO: DVec2 = DVec2::new(-2.2, -2.2);
const OUT_HI: DVec2 = DVec2::new(2.2, 2.2);

fn kept_point(p: DVec2, before: DVec2) -> DVec2 {
    let p = if p.is_finite() { p } else { before };
    p.max(IN_LO + DVec2::splat(0.15)).min(IN_HI - DVec2::splat(0.15))
}

// ----- the deep network ----------------------------------------------------------------

/// Hidden layers, their units, and the examples of the batch.
const DEPTH: usize = 20;
const WIDTH: usize = 24;
const EXAMPLES: usize = 24;
const NET_SEED: u64 = 5;
const DATA_SEED: u64 = 9;
/// The weights the fourth picture scales: layer 10's.
const SCALED_NAME: &str = "layer 10 weights";

/// The gain of the starting weights, on its grid: √2 · 2^(k/8), k from −8 to 8, so from
/// √2/2 to 2√2 with the critical gain for ReLU, √2, in the middle.
const GAIN_STEPS: i32 = 8;
fn gain_of(k: i32) -> f64 {
    SQRT_2 * 2f64.powf(k as f64 / 8.0)
}

/// The step of the grid nearest to a gain (a number that is not one: the middle).
fn gain_step(gain: f64) -> i32 {
    let k = (8.0 * (gain / SQRT_2).log2()).round();
    if k.is_finite() {
        (k as i32).clamp(-GAIN_STEPS, GAIN_STEPS)
    } else {
        0
    }
}

/// The factor on one layer's weights, on its grid: 2^(k/8), k from −16 to 16.
const FACTOR_STEPS: i32 = 16;
fn factor_of(k: i32) -> f64 {
    2f64.powf(k as f64 / 8.0)
}

fn factor_step(c: f64) -> i32 {
    let k = (8.0 * c.log2()).round();
    if k.is_finite() {
        (k as i32).clamp(-FACTOR_STEPS, FACTOR_STEPS)
    } else {
        0
    }
}

/// The three kinds a visitor can choose; `Norm::None` is always drawn beside the one
/// chosen.
const KINDS: [Norm; 3] = [Norm::Batch, Norm::Layer, Norm::Rms];

fn kind_index(n: Norm) -> usize {
    Norm::ALL.iter().position(|m| *m == n).unwrap_or(0)
}

/// The kind's name in a sentence: "layer normalisation".
fn kind_long(n: Norm) -> &'static str {
    match n {
        Norm::None => "no normalisation",
        Norm::Batch => "batch normalisation",
        Norm::Layer => "layer normalisation",
        Norm::Rms => "RMS normalisation",
    }
}

/// Twenty ReLU layers, each ReLU(norm(W x)) (the library's plain "post" arrangement), no
/// biases, starting weights of variance gain² / (inputs); the same draw of weights for
/// every normalisation, which has no random numbers of its own.
fn spec(norm: Norm, gain: f64) -> MlpSpec {
    MlpSpec { inputs: 2, width: WIDTH, depth: DEPTH, outputs: 3, act: Act::Relu, norm, place: Place::Post, residual: false, residual_scale: 1.0, init: Init::Lecun, gain, bias: false }
}

/// The batch: points of three spiral arms in the plane, and which arm each is on.
#[derive(Clone, Debug)]
struct Data {
    x: Vec<f64>,
    y: Vec<f64>,
}

impl Data {
    fn new() -> Data {
        let (x, y) = arch::spirals(EXAMPLES, 3, &mut Rng::new(DATA_SEED));
        Data { x, y }
    }
}

/// What one forward and backward pass of the network shows.
#[derive(Clone, Debug)]
struct Profile {
    /// The root mean square of each hidden layer's output (the signal), and of the
    /// gradient of the loss there, layers 1 to `DEPTH`.
    signal: Vec<f64>,
    grad: Vec<f64>,
    loss: f64,
    /// The network's outputs, every example's.
    output: Vec<f64>,
    /// Layer `SCALED`'s weights: their length, their gradient's, and the dot product of
    /// the two.
    w: f64,
    w_grad: f64,
    w_dot: f64,
}

fn rms(v: &[f64]) -> f64 {
    if v.is_empty() {
        0.0
    } else {
        (v.iter().map(|a| a * a).sum::<f64>() / v.len() as f64).sqrt()
    }
}

fn pass(model: &Model, data: &Data, tape: &mut Tape) -> Profile {
    let tr = model.loss_mlp(tape, &data.x, &data.y, Loss::CrossEntropy);
    tape.backward(tr.loss);
    let at = |l: usize| tr.get(Stage::Output, l);
    let signal = (1..=DEPTH).map(|l| at(l).map_or(f64::NAN, |v| rms(tape.value(v)))).collect();
    let grad = (1..=DEPTH).map(|l| at(l).map_or(f64::NAN, |v| rms(tape.grad(v)))).collect();
    let (mut w, mut w_grad, mut w_dot) = (f64::NAN, f64::NAN, f64::NAN);
    if let Some(&v) = model.index_of(SCALED_NAME).and_then(|i| tr.params.get(i)) {
        let (a, b) = (tape.value(v), tape.grad(v));
        w = a.iter().map(|x| x * x).sum::<f64>().sqrt();
        w_grad = b.iter().map(|x| x * x).sum::<f64>().sqrt();
        w_dot = a.iter().zip(b).map(|(x, y)| x * y).sum();
    }
    Profile { signal, grad, loss: tape.value(tr.loss).first().copied().unwrap_or(f64::NAN), output: tape.value(tr.output).to_vec(), w, w_grad, w_dot }
}

/// The network with normalisation `norm` (or none) at the gain of step `k`, and layer
/// `SCALED`'s weights times c.
fn network(norm: Norm, k: i32, c: f64) -> Model {
    let mut model = Model::mlp(&spec(norm, gain_of(k)), &mut Rng::new(NET_SEED));
    if c != 1.0 {
        if let Some(g) = model.group_mut(SCALED_NAME) {
            g.value.iter_mut().for_each(|v| *v *= c);
        }
    }
    model
}

/// What each layer multiplies `v` by, on average: e to the slope of the straight line
/// fitted (least squares) to the logarithms of `v` against the layer. (Entries that are
/// not positive numbers are left out; with fewer than two left, 1.)
fn per_layer(v: &[f64]) -> f64 {
    let pts: Vec<(f64, f64)> = v.iter().enumerate().filter(|(_, x)| x.is_finite() && **x > 0.0).map(|(i, x)| (i as f64, x.ln())).collect();
    if pts.len() < 2 {
        return 1.0;
    }
    let n = pts.len() as f64;
    let (mx, my) = (pts.iter().map(|p| p.0).sum::<f64>() / n, pts.iter().map(|p| p.1).sum::<f64>() / n);
    let sxy: f64 = pts.iter().map(|p| (p.0 - mx) * (p.1 - my)).sum();
    let sxx: f64 = pts.iter().map(|p| (p.0 - mx) * (p.0 - mx)).sum();
    (sxy / sxx).exp()
}

/// The largest over the smallest of the positive numbers in `v` (1 if there are none).
#[cfg(test)]
fn spread(v: &[f64]) -> f64 {
    let (lo, hi) = v.iter().filter(|x| x.is_finite() && **x > 0.0).fold((f64::MAX, 0.0f64), |(lo, hi), &x| (lo.min(x), hi.max(x)));
    if hi > 0.0 {
        hi / lo
    } else {
        1.0
    }
}

/// The step of gradient descent drawn in the fourth picture, relative to W at c = 1: the
/// rate is chosen so that at c = 1 the step is this long (and it is the same rate for
/// every c).
const STEP_AT_ONE: f64 = 0.4;

/// The fourth picture's numbers: layer `SCALED` of the network with the kind chosen, its
/// weights times c, against the same network at c = 1.
#[derive(Clone, Debug)]
struct Scaled {
    c: f64,
    at: Profile,
    /// The largest change of any output from c = 1, and the loss's.
    out_change: f64,
    loss_change: f64,
    /// |∇W L| at c = 1, and the step of gradient descent, in units of |W| at c = 1.
    grad_at_one: f64,
    step: f64,
}

impl Scaled {
    fn new(c: f64, at: Profile, one: &Profile) -> Scaled {
        let out_change = at.output.iter().zip(&one.output).map(|(a, b)| (a - b).abs()).fold(0.0, f64::max);
        let step = if one.w_grad > 0.0 { STEP_AT_ONE * at.w_grad / one.w_grad } else { 0.0 };
        Scaled { c, out_change, loss_change: at.loss - one.loss, grad_at_one: one.w_grad, step: if step.is_finite() { step } else { 0.0 }, at }
    }

    /// The angle between the weights and their gradient, in degrees.
    fn angle(&self) -> f64 {
        let cos = self.at.w_dot / (self.at.w * self.at.w_grad);
        if cos.is_finite() {
            cos.max(-1.0).min(1.0).acos().to_degrees()
        } else {
            90.0
        }
    }

    /// How far one step of gradient descent turns the weights, in degrees: the step is
    /// at right angles to W, `step` long against W's `c`.
    fn turn(&self) -> f64 {
        let t = (self.step / self.c).atan().to_degrees();
        if t.is_finite() {
            t
        } else {
            0.0
        }
    }
}

// ----- easing ---------------------------------------------------------------------------

/// How fast a change glides: each of two eases one after the other covers about two
/// thirds of what is left in this many seconds. Quick for what answers a button, slow
/// for the moves the things to try play.
const QUICK: f64 = 0.07;
const SLOW: f64 = 0.28;

/// Numbers as asked for and as drawn: the drawn ones follow through `between`, two eases
/// one after the other, so that they set off gently and arrive exactly.
#[derive(Clone, Debug)]
struct Ease {
    to: Vec<f64>,
    between: Vec<f64>,
    shown: Vec<f64>,
    tau: f64,
}

impl Ease {
    fn new(v: &[f64]) -> Ease {
        Ease { to: v.to_vec(), between: v.to_vec(), shown: v.to_vec(), tau: QUICK }
    }

    fn arrived(&self) -> bool {
        self.shown == self.to && self.between == self.to
    }

    fn aim(&mut self, i: usize, v: f64) {
        if let Some(t) = self.to.get_mut(i) {
            *t = v;
        }
    }

    fn aim_all(&mut self, v: &[f64]) {
        if v.len() != self.to.len() {
            *self = Ease { tau: self.tau, ..Ease::new(v) };
        } else {
            self.to.copy_from_slice(v);
        }
    }

    /// At once, where a hand puts it.
    fn put(&mut self, i: usize, v: f64) {
        for list in [&mut self.to, &mut self.between, &mut self.shown] {
            if let Some(t) = list.get_mut(i) {
                *t = v;
            }
        }
    }

    /// Bring the drawn numbers `dt` seconds nearer; says whether they are still on
    /// their way.
    fn step(&mut self, dt: f64) -> bool {
        if self.arrived() {
            return false;
        }
        let k = 1.0 - (-dt.max(0.0) / self.tau).exp();
        let mut far = false;
        for ((b, s), t) in self.between.iter_mut().zip(self.shown.iter_mut()).zip(&self.to) {
            *b += (t - *b) * k;
            *s += (*b - *s) * k;
            let near = 1e-4 * t.abs().max(1.0);
            far |= !((t - *s).abs() <= near && (t - *b).abs() <= near);
        }
        // (exactly there in the end: the picture then stops changing)
        if !far {
            self.between.clone_from(&self.to);
            self.shown.clone_from(&self.to);
            self.tau = QUICK;
            return false;
        }
        true
    }

    fn get(&self, i: usize) -> f64 {
        self.shown.get(i).copied().unwrap_or(0.0)
    }

    fn vec(&self, i: usize) -> DVec2 {
        DVec2::new(self.get(i), self.get(i + 1))
    }

    fn asked(&self, i: usize) -> DVec2 {
        DVec2::new(self.to.get(i).copied().unwrap_or(0.0), self.to.get(i + 1).copied().unwrap_or(0.0))
    }
}

// ----- the things to try ----------------------------------------------------------------

/// Where each thing moved by hand sits among the numbers of `NormGrad::hands`: x and g
/// of the first picture, then the five examples and the gradient at the first one's
/// output, of the second.
const H_X: usize = 0;
const H_G: usize = 2;
const H_PTS: usize = 4;
const H_GB: usize = H_PTS + 2 * MANY;
const HANDS: usize = H_GB + 2;

/// The first picture as it opens.
const X0: DVec2 = DVec2::new(0.76, 0.38);
const G0: DVec2 = DVec2::new(0.25, 0.95);
/// The second: the five examples and the gradient at the first one's output.
const PTS0: [DVec2; MANY] = [DVec2::new(1.0, 1.65), DVec2::new(1.75, 1.15), DVec2::new(0.35, 0.9), DVec2::new(1.65, 2.45), DVec2::new(2.4, 1.55)];
const GB0: DVec2 = DVec2::new(0.45, 0.75);
/// The fourth thing to try: where example 2 goes.
const PT2_TO: DVec2 = DVec2::new(1.75, 3.8);
/// The gain the network opens at (step 4: gain 2), above the critical √2, and the
/// third thing to try's (step −4: gain 1), below it.
const OPEN_GAIN: i32 = 4;
const LOW_GAIN: i32 = -4;

/// The slow move a thing to try plays.
#[derive(Clone, Copy, Debug, PartialEq)]
enum Then {
    /// The hand at this index of `NormGrad::hands` to this place.
    Hand(usize, DVec2),
    /// The factor on layer 10's weights to this step.
    Factor(i32),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Page {
    /// One layer, up close: the first two pictures.
    Close,
    /// The deep network: the last two.
    Deep,
}

struct Trial {
    label: &'static str,
    look_for: &'static str,
}

const TRIALS: [Trial; 5] = [
    Trial {
        label: "what comes back has lost its part along the input",
        look_for: r"The gradient $g$ arrives at $y$, 0.98 long. Its part along $x$ (dashed), 0.65, is dropped: moving $x$ along its own line does not move $y$, so the loss has no slope that way. What is left, 0.74, is divided by $\operatorname{rms}(x) = 0.60$ and comes back to $x$ as $\partial L/\partial x$, 1.23 long, at right angles to $x$. Drag the tip of $g$ round: pointing along $x$, nothing comes back at all.",
    },
    Trial {
        label: "twice as long, half the gradient",
        look_for: r"$x$ grows from 1.00 to 2.00 long in the same direction. $y$ does not move, and the same $g$ arrives there, but what comes back is divided by $\operatorname{rms}(x)$, which doubles, so it halves: from 1.04 to 0.52 (faint: $x$ as it was, and what came back to it). The larger the input, the smaller its gradient.",
    },
    Trial {
        label: "a deep network at a bad gain, rescued",
        look_for: r"Gain 1, below the critical $\sqrt{2}$. Without normalisation (dashed) each layer keeps about 0.69 of the signal it is given: from layer 1 to layer 20 it falls to a 1400th. The gradient fades as fast on its way back, to a 1700th at layer 1. With layer normalisation (solid) the signal leaves every layer at 0.63 to 0.77, and across the 20 layers the gradient varies by less than a factor of 4. Drag the gain: the solid signal does not move. Choose batch normalisation: the signal is as level, but the gradient is not: it grows on its way back, by about 1.3 a layer.",
    },
    Trial {
        label: "batch normalisation ties the examples together",
        look_for: r"A gradient arrives at example 1's output alone, yet something comes back to all five examples: their mean and spread are in every output. Now example 2 moves up, and nothing else: example 1's output moves too, and what comes back to it shrinks from 1.21 to 0.74 and turns by 16°. Unit 1 is standardised on its own, so the part along unit 1 does not change. With layer or RMS normalisation each example is normalised by itself, and nothing reaches the others.",
    },
    Trial {
        label: "scale the weights: the same output, a smaller gradient",
        look_for: r"Layer 10's weights $W$ become four times as large. Layer normalisation divides out their size, so the network's outputs and its loss stay the same (to 14 decimal places). The gradient $\nabla_W L$ becomes a quarter as long, 0.178 instead of 0.711, and stays at right angles to $W$. A step of gradient descent at the same rate is then a sixteenth as long against $W$: it turns $W$ by 1.4° instead of 21.8°. A larger weight learns more slowly.",
    },
];

// ----- where things go --------------------------------------------------------------------

/// A plane drawn on the screen: a rectangle of it, and the part of the plane it shows
/// (the two of the same shape).
#[derive(Clone, Copy, Debug)]
struct Plane {
    rect: Rect,
    lo: DVec2,
    hi: DVec2,
}

impl Plane {
    fn px(&self, p: DVec2) -> Vec2 {
        let (min, max) = self.rect;
        let t = (p - self.lo) / (self.hi - self.lo);
        Vec2::new(min.x + (max.x - min.x) * t.x as f32, max.y - (max.y - min.y) * t.y as f32)
    }

    fn at(&self, q: Vec2) -> DVec2 {
        let (min, max) = self.rect;
        let t = DVec2::new(((q.x - min.x) / (max.x - min.x).max(1e-6)) as f64, ((max.y - q.y) / (max.y - min.y).max(1e-6)) as f64);
        self.lo + t * (self.hi - self.lo)
    }

    /// Pixels per unit.
    fn scale(&self) -> f32 {
        (self.rect.1.x - self.rect.0.x) / (self.hi.x - self.lo.x) as f32
    }
}

/// The box of the first picture, and of the fourth.
const ONE_LO: DVec2 = DVec2::new(-2.7, -2.7);
const ONE_HI: DVec2 = DVec2::new(2.7, 2.7);
const W_LO: DVec2 = DVec2::new(-0.3, -0.45);
const W_HI: DVec2 = DVec2::new(4.65, 1.72);

#[derive(Clone, Copy, Debug)]
struct OneLay {
    head: Vec2,
    plane: Plane,
    lines: Vec2,
    width: f32,
}

#[derive(Clone, Copy, Debug)]
struct ManyLay {
    head: Vec2,
    planes: [Plane; 2],
    titles: [Vec2; 2],
    lines: Vec2,
    width: f32,
}

#[derive(Clone, Copy, Debug)]
struct DeepLay {
    head: Vec2,
    /// Where "normalisation:" is written, and its three buttons.
    lead: Vec2,
    kinds: [Rect; 3],
    chart: Rect,
    /// The gain's track: the line it runs along (its two ends), and where its name goes.
    track: (Vec2, Vec2),
    track_name: Vec2,
    lines: Vec2,
    width: f32,
}

#[derive(Clone, Copy, Debug)]
struct WeightsLay {
    head: Vec2,
    plane: Plane,
    lines: Vec2,
    width: f32,
}

#[derive(Clone, Debug)]
struct Layout {
    narrow: bool,
    /// The size of the headings, and of everything else written.
    text: f32,
    small: f32,
    /// The two tabs, where the pictures come two at a time.
    tabs: Option<[Rect; 2]>,
    one: Option<OneLay>,
    many: Option<ManyLay>,
    deep: Option<DeepLay>,
    weights: Option<WeightsLay>,
}

/// How far apart lines of text are.
fn line_h(size: f32) -> f32 {
    size + 7.0
}

const ONE_LINES: usize = 3;
const MANY_LINES: usize = 2;
/// (the legend on one line or two, and the factors)
const DEEP_LINES: usize = 3;
const W_LINES: usize = 3;

fn size_of(r: Rect) -> Vec2 {
    r.1 - r.0
}

fn one_in(r: Rect, text: f32, small: f32) -> Option<OneLay> {
    let (w, h) = (size_of(r).x, size_of(r).y);
    let head_h = text + 10.0;
    let lines_h = ONE_LINES as f32 * line_h(small) + 6.0;
    let side = w.min(h - head_h - lines_h).floor();
    if side < 90.0 {
        return None;
    }
    let x0 = r.0.x + ((w - side) * 0.5).floor();
    let top = r.0.y + head_h;
    let rect = (Vec2::new(x0, top), Vec2::new(x0 + side, top + side));
    Some(OneLay { head: r.0, plane: Plane { rect, lo: ONE_LO, hi: ONE_HI }, lines: Vec2::new(r.0.x, top + side + 6.0), width: w })
}

fn many_in(r: Rect, text: f32, small: f32) -> Option<ManyLay> {
    let (w, h) = (size_of(r).x, size_of(r).y);
    let head_h = text + 10.0;
    let title_h = small + 8.0;
    let lines_h = MANY_LINES as f32 * line_h(small) + 6.0;
    let gap = 22.0;
    let side = ((w - gap) * 0.5).min(h - head_h - title_h - lines_h).floor();
    if side < 70.0 {
        return None;
    }
    let x0 = r.0.x + ((w - 2.0 * side - gap) * 0.5).floor();
    let top = r.0.y + head_h + title_h;
    let x1 = x0 + side + gap;
    let planes = [
        Plane { rect: (Vec2::new(x0, top), Vec2::new(x0 + side, top + side)), lo: IN_LO, hi: IN_HI },
        Plane { rect: (Vec2::new(x1, top), Vec2::new(x1 + side, top + side)), lo: OUT_LO, hi: OUT_HI },
    ];
    Some(ManyLay { head: r.0, planes, titles: [Vec2::new(x0, top - 4.0), Vec2::new(x1, top - 4.0)], lines: Vec2::new(r.0.x, top + side + 6.0), width: w })
}

fn deep_in(r: Rect, text: f32, small: f32) -> Option<DeepLay> {
    let w = size_of(r).x;
    let head_h = text + 10.0;
    let row_h = 26.0;
    let lead_w = text_width("normalisation:", small) + 10.0;
    let words = KINDS.map(|k| k.name());
    let mut left = r.0.x + lead_w;
    let top = r.0.y + head_h;
    let kinds = words.map(|word| {
        let wd = text_width(word, small) + 22.0;
        let b = (Vec2::new(left, top), Vec2::new(left + wd, top + row_h));
        left += wd + 6.0;
        b
    });
    let lines_h = DEEP_LINES as f32 * line_h(small) + 4.0;
    let track_h = 40.0;
    let chart = (Vec2::new(r.0.x + 46.0, top + row_h + 16.0), Vec2::new(r.1.x - 10.0, r.1.y - lines_h - track_h - 30.0));
    if size_of(chart).y < 50.0 || size_of(chart).x < 120.0 || left > r.1.x + 1.0 {
        return None;
    }
    let ty = chart.1.y + 30.0 + 15.0;
    let name_w = text_width("gain of the starting weights:", small) + 14.0;
    let (t0, t1) = if w - name_w >= 220.0 { (Vec2::new(r.0.x + name_w + 6.0, ty), Vec2::new(r.1.x - 24.0, ty)) } else { (Vec2::new(r.0.x + 20.0, ty + 4.0), Vec2::new(r.1.x - 24.0, ty + 4.0)) };
    let track_name = if w - name_w >= 220.0 { Vec2::new(r.0.x, ty) } else { Vec2::new(r.0.x, ty - 16.0) };
    Some(DeepLay { head: r.0, lead: Vec2::new(r.0.x, top + row_h * 0.5), kinds, chart, track: (t0, t1), track_name, lines: Vec2::new(r.0.x, chart.1.y + 30.0 + track_h), width: w })
}

fn weights_in(r: Rect, text: f32, small: f32) -> Option<WeightsLay> {
    let (w, h) = (size_of(r).x, size_of(r).y);
    let head_h = text + 10.0;
    let lines_h = W_LINES as f32 * line_h(small) + 6.0;
    let aspect = ((W_HI.x - W_LO.x) / (W_HI.y - W_LO.y)) as f32;
    let pw = w.min((h - head_h - lines_h) * aspect).floor();
    let ph = (pw / aspect).floor();
    if ph < 60.0 {
        return None;
    }
    let x0 = r.0.x + ((w - pw) * 0.5).floor();
    let top = r.0.y + head_h;
    let rect = (Vec2::new(x0, top), Vec2::new(x0 + pw, top + ph));
    Some(WeightsLay { head: r.0, plane: Plane { rect, lo: W_LO, hi: W_HI }, lines: Vec2::new(r.0.x, top + ph + 6.0), width: w })
}

impl Layout {
    fn of(view: Vec2, inset: f32, page: Page) -> Layout {
        let view = view.max(Vec2::ONE);
        let narrow = view.x < 620.0;
        let (text, small) = if narrow { (12.0, 11.0) } else { (13.0, 12.0) };
        let mut lay = Layout { narrow, text, small, tabs: None, one: None, many: None, deep: None, weights: None };
        if view.x < 200.0 || view.y < 160.0 {
            return lay;
        }
        let side = if narrow { 10.0 } else { 16.0 };
        let top = inset.max(0.0) + if narrow { 8.0 } else { 12.0 };
        let bottom = view.y - 8.0;
        if view.x >= 900.0 && view.y >= 640.0 {
            // all four at once: the two pictures of one layer on the left, the network on
            // the right; the upper row taller, for the plane and the chart
            let gap = 30.0;
            let mid_x = (view.x * 0.5).floor();
            let avail = bottom - top - gap;
            let lower = (avail * 0.4).max(300.0).min(avail * 0.5).floor();
            let mid_y = bottom - lower;
            let cells = [
                (Vec2::new(side, top), Vec2::new(mid_x - gap * 0.5, mid_y - gap)),
                (Vec2::new(side, mid_y), Vec2::new(mid_x - gap * 0.5, bottom)),
                (Vec2::new(mid_x + gap * 0.5, top), Vec2::new(view.x - side, mid_y - gap)),
                (Vec2::new(mid_x + gap * 0.5, mid_y), Vec2::new(view.x - side, bottom)),
            ];
            lay.one = one_in(cells[0], text, small);
            lay.many = many_in(cells[1], text, small);
            lay.deep = deep_in(cells[2], text, small);
            lay.weights = weights_in(cells[3], text, small);
            return lay;
        }
        // two at a time, the page chosen by the tabs
        let tabs = plot::even_row(2, side, view.x - side, top, 26.0, 8.0);
        lay.tabs = Some([tabs[0], tabs[1]]);
        let area = (Vec2::new(side, top + 26.0 + 12.0), Vec2::new(view.x - side, bottom));
        let s = size_of(area);
        let gap = 22.0;
        // (the chart takes more of the height than the plane under it)
        let share = if page == Page::Deep { 0.58 } else { 0.5 };
        let [a, b] = if s.y >= 0.9 * s.x {
            let first = ((s.y - gap) * share).floor();
            [(area.0, Vec2::new(area.1.x, area.0.y + first)), (Vec2::new(area.0.x, area.0.y + first + gap), area.1)]
        } else {
            let first = ((s.x - gap) * 0.5).floor();
            [(area.0, Vec2::new(area.0.x + first, area.1.y)), (Vec2::new(area.0.x + first + gap, area.0.y), area.1)]
        };
        match page {
            Page::Close => {
                lay.one = one_in(a, text, small);
                lay.many = many_in(b, text, small);
            }
            Page::Deep => {
                lay.deep = deep_in(a, text, small);
                lay.weights = weights_in(b, text, small);
            }
        }
        lay
    }
}

fn inside_rect(p: Vec2, (min, max): Rect) -> bool {
    p.cmpge(min).all() && p.cmple(max).all()
}

// ----- drawing helpers --------------------------------------------------------------------

/// The line kinds, each a batch of its own (see `plot::apart`), so that from one frame to
/// the next `eidos motion` can follow each picture's lines point for point: the frames
/// and axes of the planes; the first picture's lines (circle, ray) and arrows and its
/// faint copy (comes and goes); the second's lines and arrows; the chart's ticks (as
/// many as its range has), its curves, its track; the fourth's lines and arrows.
const ONE_LINES_K: u8 = plot::OWN;
const ONE_ARROWS: u8 = plot::OWN + 1;
const ONE_GHOST: u8 = plot::OWN + 2;
const MANY_LINES_K: u8 = plot::OWN + 3;
const MANY_ARROWS: u8 = plot::OWN + 4;
const DEEP_TICKS: u8 = plot::OWN + 5;
const DEEP_CURVES: u8 = plot::OWN + 6;
const DEEP_TRACK: u8 = plot::OWN + 7;
const W_LINES_K: u8 = plot::OWN + 8;
const W_ARROWS: u8 = plot::OWN + 9;
/// Dashed: the first picture's part of g along x and the ray, the second's boxes, the
/// chart's curves without normalisation, the fourth's ray.
const ONE_DASHED: u8 = plot::OWN + 10;
const MANY_DASHED: u8 = plot::OWN + 11;
const DEEP_DASHED: u8 = plot::OWN + 12;
const W_DASHED: u8 = plot::OWN + 13;

const SOLID_KINDS: [u8; 10] = [ONE_LINES_K, ONE_ARROWS, ONE_GHOST, MANY_LINES_K, MANY_ARROWS, DEEP_TICKS, DEEP_CURVES, DEEP_TRACK, W_LINES_K, W_ARROWS];
const DASHED_KINDS: [u8; 4] = [ONE_DASHED, MANY_DASHED, DEEP_DASHED, W_DASHED];

/// Every batch, made in the same order in every frame.
fn in_order(f: &mut Frame) {
    for kind in SOLID_KINDS {
        plot::apart(f, kind, false);
    }
    for kind in DASHED_KINDS {
        plot::apart(f, kind, true);
    }
    for panel in 0..4 {
        dots(f, panel);
    }
}

/// A batch with nothing in it this frame is given a point that cannot be seen: one left
/// empty is dropped, and every batch after it would be compared with the wrong one.
fn keep_batches(f: &mut Frame) {
    for (kind, dashed) in SOLID_KINDS.iter().map(|k| (*k, false)).chain(DASHED_KINDS.iter().map(|k| (*k, true))) {
        let batch = plot::apart(f, kind, dashed);
        if batch.pts.len() <= 1 {
            batch.segment(Vec3::ZERO, Vec3::ZERO, 0.0, Color::CLEAR);
        }
    }
    for panel in 0..4 {
        let batch = dots(f, panel);
        if batch.pts.is_empty() {
            batch.push(Vec3::ZERO, 0.0, Color::CLEAR);
        }
    }
}

/// The dots of one picture: a batch of their own (by a hair of their size).
fn dots<'b>(f: &'b mut Frame<'_>, panel: u8) -> &'b mut crate::gfx::draw::PointBatch {
    f.dl.points_with(PointKind::Overlay, PointStyle { outline_px: 0.0, size_scale: 1.0 + 1e-4 * (panel as f32 + 1.0), ..Default::default() })
}

fn dot(f: &mut Frame, panel: u8, at: Vec2, size: f32, color: Color) {
    if at.is_finite() {
        let p = f.hud(at);
        dots(f, panel).push(p, size, color);
    }
}

/// An arrow on the screen from a to b, its shaft in the batch of `kind` and its head a
/// filled triangle of a size fixed on the screen. One too short to see is still drawn,
/// invisibly, so that the batch keeps its number of points.
fn arrow(f: &mut Frame, kind: u8, a: Vec2, b: Vec2, width: f32, color: Color) {
    if !(a.is_finite() && b.is_finite()) {
        plot::line_of(f, kind, Vec2::ZERO, Vec2::ZERO, 0.0, Color::CLEAR);
        return;
    }
    let d = b - a;
    let len = d.length();
    if len < 1.5 {
        plot::line_of(f, kind, a, a, 0.0, Color::CLEAR);
        return;
    }
    let dir = d / len;
    let head = (width * 2.6 + 6.0).min(len * 0.55);
    let base = b - dir * head;
    let side = Vec2::new(-dir.y, dir.x) * head * 0.42;
    plot::line_of(f, kind, a, b - dir * head * 0.6, width, color);
    let (p, q, r) = (f.hud_at(2, b), f.hud_at(2, base + side), f.hud_at(2, base - side));
    f.dl.tris_alpha.tri(p, q, r, color);
}

/// The end of an arrow from `a` to `b` kept inside the rectangle (an arrow longer than
/// the picture is cut at its edge).
fn kept_in(a: Vec2, b: Vec2, (min, max): Rect) -> Vec2 {
    if !(a.is_finite() && b.is_finite()) || inside_rect(b, (min, max)) {
        return b;
    }
    let d = b - a;
    let mut t = 1.0f32;
    for (p, lo, hi) in [(d.x, min.x - a.x, max.x - a.x), (d.y, min.y - a.y, max.y - a.y)] {
        if p > 0.0 {
            t = t.min(hi / p);
        } else if p < 0.0 {
            t = t.min(lo / p);
        }
    }
    a + d * t.max(0.0)
}

/// Text that is made smaller to fit `width` if it would not.
fn fitted(f: &mut Frame, at: Vec2, text: &str, size: f32, width: f32, color: Color) {
    let w = f.text_size(text, size).x;
    let s = if w > width && w > 0.0 { (size * width / w).max(8.0) } else { size };
    f.text_px(at, text, s, color, Align2::LEFT_TOP);
}

/// A label beside a point of the screen, on whichever side is clear (see `Room`).
#[allow(clippy::too_many_arguments)]
fn label(room: &mut Room, f: &mut Frame, tag: impl Into<Tag>, at: Vec2, text: &str, size: f32, color: Color, towards: Vec2) {
    if let Some(world) = f.cam.pick_ground(at * f.px).filter(|w| w.is_finite()) {
        room.label(f, tag, world, text, size, color, towards, 7.0, false);
    }
}

/// Room for labels inside `keep` alone.
fn room_in(f: &Frame, (min, max): Rect) -> Room {
    let mut room = Room::new(f);
    let view = f.view_size();
    let far = view + Vec2::splat(100.0);
    room.wall(Vec2::splat(-100.0), Vec2::new(far.x, min.y));
    room.wall(Vec2::new(-100.0, max.y), far);
    room.wall(Vec2::splat(-100.0), Vec2::new(min.x, far.y));
    room.wall(Vec2::new(max.x, -100.0), far);
    room
}

/// A number written for a formula: 2.4 × 10⁻¹⁵ style when it is very small.
fn tiny(v: f64) -> String {
    if v == 0.0 {
        "0".into()
    } else if v.abs() < 1e-3 {
        let e = v.abs().log10().floor();
        let m = v / 10f64.powf(e);
        format!(r"{} \times 10^{{{}}}", num(m, 1), e as i32)
    } else {
        num(v, 4)
    }
}

// ----- the scene ------------------------------------------------------------------------

pub struct NormGrad {
    /// The normalisation chosen, the gain's step and the factor's step.
    kind: Norm,
    gain_k: i32,
    c_k: i32,
    /// What is moved by hand, as asked for and as drawn (see `H_X` and the rest).
    hands: Ease,
    /// A move a thing to try plays slowly once what it set up has arrived.
    then: Option<Then>,
    /// x as it was before the second thing to try doubled it, drawn faint.
    ghost: Option<DVec2>,
    /// The network's passes worked out so far, by (kind, gain step).
    profiles: HashMap<(usize, i32), Profile>,
    /// The fourth picture's numbers, for (kind, gain step, factor step).
    scaled: Option<((usize, i32, i32), Scaled)>,
    /// The chart as drawn: four curves of `DEPTH` logarithms each, the range shown (lowest
    /// and highest power of ten), and the gain (log₂ of gain / √2).
    chart: Ease,
    /// The fourth picture as drawn: log₂ c and log₂ of the step.
    weights: Ease,
    page: Page,
    moving: bool,
    data: Data,
    tape: Tape,
}

const CHART_LEN: usize = 4 * DEPTH + 3;

impl NormGrad {
    fn new() -> NormGrad {
        let mut hands = vec![0.0; HANDS];
        for (k, p) in [(H_X, X0), (H_G, G0), (H_GB, GB0)] {
            (hands[k], hands[k + 1]) = (p.x, p.y);
        }
        for (i, p) in PTS0.iter().enumerate() {
            (hands[H_PTS + 2 * i], hands[H_PTS + 2 * i + 1]) = (p.x, p.y);
        }
        let mut scene = NormGrad {
            kind: Norm::Layer,
            gain_k: OPEN_GAIN,
            c_k: 0,
            hands: Ease::new(&hands),
            then: None,
            ghost: None,
            profiles: HashMap::new(),
            scaled: None,
            chart: Ease::new(&[0.0; CHART_LEN]),
            weights: Ease::new(&[0.0, STEP_AT_ONE.log2()]),
            page: Page::Close,
            moving: false,
            data: Data::new(),
            tape: Tape::new(),
        };
        // (opening, there is nothing to glide from)
        scene.work(usize::MAX);
        if let Some(t) = scene.chart_target() {
            scene.chart = Ease::new(&t);
        }
        if let Some(t) = scene.weights_target() {
            scene.weights = Ease::new(&t);
        }
        scene
    }

    /// Set up thing to try `k`.
    fn set_up(&mut self, k: usize) {
        self.then = None;
        self.ghost = None;
        match k {
            0 | 1 => {
                self.page = Page::Close;
                // (the second from x 1 long, in the same direction)
                let (x, g) = if k == 0 { (X0, G0) } else { (X0.normalize(), G0) };
                self.aim_hand(H_X, x);
                self.aim_hand(H_G, g);
                if k == 1 {
                    self.ghost = Some(x);
                    self.then = Some(Then::Hand(H_X, x * 2.0));
                }
            }
            3 => {
                self.page = Page::Close;
                for (i, p) in PTS0.iter().enumerate() {
                    self.aim_hand(H_PTS + 2 * i, *p);
                }
                self.aim_hand(H_GB, GB0);
                self.then = Some(Then::Hand(H_PTS + 2, PT2_TO));
            }
            2 => {
                self.page = Page::Deep;
                self.kind = Norm::Layer;
                self.set_gain(LOW_GAIN);
                self.set_factor(0);
            }
            _ => {
                self.page = Page::Deep;
                self.kind = Norm::Layer;
                self.set_gain(0);
                // (from c = 1, slowly, once the network is there)
                self.c_k = 0;
                self.weights.tau = QUICK;
                self.then = Some(Then::Factor(FACTOR_STEPS));
            }
        }
    }

    fn aim_hand(&mut self, at: usize, p: DVec2) {
        self.hands.aim(at, p.x);
        self.hands.aim(at + 1, p.y);
    }

    /// A hand moves a thing: at once, and whatever was playing stops.
    fn put_hand(&mut self, at: usize, p: DVec2) {
        self.hands.put(at, p.x);
        self.hands.put(at + 1, p.y);
        self.then = None;
        self.ghost = None;
    }

    fn set_gain(&mut self, k: i32) {
        self.gain_k = k.clamp(-GAIN_STEPS, GAIN_STEPS);
    }

    fn set_factor(&mut self, k: i32) {
        self.c_k = k.clamp(-FACTOR_STEPS, FACTOR_STEPS);
    }

    fn set_kind(&mut self, n: Norm) {
        if n != Norm::None {
            self.kind = n;
        }
    }

    /// The network's passes the settings asked for need, at most `most` of them now.
    fn work(&mut self, most: usize) {
        let mut done = 0;
        let ki = kind_index(self.kind);
        for (n, idx) in [(Norm::None, 0), (self.kind, ki)] {
            if done < most && !self.profiles.contains_key(&(idx, self.gain_k)) {
                let p = pass(&network(n, self.gain_k, 1.0), &self.data, &mut self.tape);
                self.profiles.insert((idx, self.gain_k), p);
                done += 1;
            }
        }
        let key = (ki, self.gain_k, self.c_k);
        if self.scaled.as_ref().map(|s| s.0) != Some(key) {
            if let Some(at_one) = self.profiles.get(&(ki, self.gain_k)).cloned() {
                let c = factor_of(self.c_k);
                if self.c_k == 0 {
                    self.scaled = Some((key, Scaled::new(c, at_one.clone(), &at_one)));
                } else if done < most {
                    let at = pass(&network(self.kind, self.gain_k, c), &self.data, &mut self.tape);
                    self.scaled = Some((key, Scaled::new(c, at, &at_one)));
                }
            }
        }
    }

    fn profile_without(&self) -> Option<&Profile> {
        self.profiles.get(&(0, self.gain_k))
    }

    fn profile_with(&self) -> Option<&Profile> {
        self.profiles.get(&(kind_index(self.kind), self.gain_k))
    }

    fn scaled_now(&self) -> Option<&Scaled> {
        self.scaled.as_ref().filter(|s| s.0 == (kind_index(self.kind), self.gain_k, self.c_k)).map(|s| &s.1)
    }

    /// The chart the settings ask for, once their passes are worked out.
    fn chart_target(&self) -> Option<Vec<f64>> {
        let (a, b) = (self.profile_without()?, self.profile_with()?);
        let log = |v: f64| if v > 0.0 && v.is_finite() { v.log10().max(-30.0).min(30.0) } else { -30.0 };
        let mut out: Vec<f64> = [&a.signal, &a.grad, &b.signal, &b.grad].iter().flat_map(|s| s.iter().map(|v| log(*v))).collect();
        // the range: whole powers of ten round everything, at least three of them
        let real = out.iter().copied().filter(|v| *v > -30.0);
        let (lo, hi) = real.fold((f64::MAX, f64::MIN), |(lo, hi), v| (lo.min(v), hi.max(v)));
        let (mut lo, mut hi) = if lo <= hi { ((lo - 0.15).floor(), (hi + 0.15).ceil()) } else { (-3.0, 1.0) };
        while hi - lo < 3.0 {
            lo -= 1.0;
            hi += 1.0;
        }
        out.extend([lo, hi, (gain_of(self.gain_k) / SQRT_2).log2()]);
        Some(out)
    }

    /// The fourth picture the settings ask for.
    fn weights_target(&self) -> Option<Vec<f64>> {
        let s = self.scaled_now()?;
        Some(vec![s.c.log2(), s.step.max(1e-6).log2()])
    }

    /// Move everything on by `dt` seconds of the clock's (a change of setting glides by
    /// the clock, so that it is seen through even while the scene's time stands still).
    fn advance(&mut self, dt: f64) {
        let dt = if dt.is_finite() { dt.max(0.0) } else { 0.0 };
        // (two passes at most in a frame: a third waits for the next)
        self.work(2);
        let mut moving = self.hands.step(dt);
        match self.then {
            // (the factor's move once the network at c = 1 is drawn)
            Some(Then::Factor(k)) if self.chart.arrived() && self.weights.arrived() && self.scaled_now().is_some() => {
                self.set_factor(k);
                self.weights.tau = SLOW;
                self.then = None;
            }
            Some(Then::Hand(at, to)) if self.hands.arrived() => {
                self.aim_hand(at, to);
                self.hands.tau = SLOW;
                self.then = None;
            }
            _ => {}
        }
        moving |= self.then.is_some();
        if let Some(t) = self.chart_target() {
            self.chart.aim_all(&t);
        }
        moving |= self.chart.step(dt);
        if let Some(t) = self.weights_target() {
            self.weights.aim_all(&t);
        }
        moving |= self.weights.step(dt);
        self.moving = moving;
    }

    fn at_rest(&self) -> bool {
        !self.moving
            && self.then.is_none()
            && self.hands.arrived()
            && self.chart.arrived()
            && self.weights.arrived()
            && self.profile_with().is_some()
            && self.profile_without().is_some()
            && self.scaled_now().is_some()
    }

    /// The first picture as asked for, and as drawn.
    fn one_asked(&mut self) -> One {
        let (x, g) = (self.hands.asked(H_X), self.hands.asked(H_G));
        one(x, g, &mut self.tape)
    }

    fn one_shown(&mut self) -> One {
        let (x, g) = (self.hands.vec(H_X), self.hands.vec(H_G));
        one(x, g, &mut self.tape)
    }

    fn points(&self, asked: bool) -> [DVec2; MANY] {
        std::array::from_fn(|i| if asked { self.hands.asked(H_PTS + 2 * i) } else { self.hands.vec(H_PTS + 2 * i) })
    }

    fn many_asked(&mut self) -> Many {
        let (x, g) = (self.points(true), self.hands.asked(H_GB));
        many(&x, g, &mut self.tape)
    }

    fn many_shown(&mut self) -> Many {
        let (x, g) = (self.points(false), self.hands.vec(H_GB));
        many(&x, g, &mut self.tape)
    }
}

impl Scene for NormGrad {
    fn view(&self) -> View {
        View::Flat { center: Vec2::ZERO, half_height: 1.0 }
    }

    fn settings(&self, s: &mut ViewSettings) {
        s.grid = false;
        s.bloom = s.bloom.min(0.2);
    }

    fn animated(&self) -> bool {
        !self.at_rest()
    }

    fn pointer_hint(&self) -> Option<&'static str> {
        Some("dots: drag them    buttons on the picture: click")
    }

    fn ui(&mut self, ui: &mut egui::Ui, _: &Gfx) {
        ui::note(ui, r"A normalisation layer rescales what passes through it to a standard size. RMS normalisation divides each example by the root mean square of its units. Layer normalisation first takes off their mean, then divides by their standard deviation. Batch normalisation does that to each unit over the examples of the batch. Going forward, the signal then leaves the layer at the same size, whatever the size of the weights before it. Going back, the layer's Jacobian takes out of the gradient its part along the input and divides the rest by the input's size (layer and batch normalisation also take out its part along the all-ones direction, and batch normalisation does all this to each unit, across the batch). For RMS normalisation, $y = x / \operatorname{rms}(x)$, with $g = \partial L / \partial y$ the gradient that arrives at $y$ and $u = x / |x|$ the direction of $x$:");
        ui::equation(ui, r"\frac{\partial L}{\partial x} = \frac{g - (g \cdot u)\,u}{\operatorname{rms}(x)}");
        ui::note(ui, r"So what comes back is at right angles to $x$: a change of $x$'s size changes nothing, so the loss has no slope that way. And it is half as large for an $x$ twice as large. The same goes for the weights that feed a normalisation: the loss does not depend on their size, and their gradient shrinks as they grow.");

        ui::section(ui, "Things to try");
        for (k, trial) in TRIALS.iter().enumerate() {
            if ui::experiment(ui, trial.label, trial.look_for) {
                self.set_up(k);
            }
        }

        ui::section(ui, "The network");
        let mut k = KINDS.iter().position(|n| *n == self.kind).unwrap_or(1);
        if ui::choice(ui, &["batch", "layer", "RMS"], &mut k) {
            self.set_kind(KINDS[k.min(2)]);
        }
        let mut gain = gain_of(self.gain_k) as f32;
        if ui::slider_log(ui, "gain of the starting weights", &mut gain, gain_of(-GAIN_STEPS) as f32..=gain_of(GAIN_STEPS) as f32) {
            self.set_gain(gain_step(gain as f64));
        }
        ui::note(ui, r"Each weight starts with variance $\text{gain}^2 / n$, $n$ the number of its layer's inputs, in steps of an eighth of an octave. Without a normalisation, $\text{gain} = \sqrt{2}$ keeps a ReLU network's signal the same size from layer to layer on average when the layers are very wide (scene gradflow); in this one, 24 units wide, each layer multiplies it by about 0.97.");
        let mut c = factor_of(self.c_k) as f32;
        if ui::slider_log(ui, "factor $c$ on layer 10's weights", &mut c, 0.25..=4.0) {
            self.set_factor(factor_step(c as f64));
        }
        ui::note(ui, "Only the fourth picture uses it: the network of the third, with layer 10's weights scaled.");

        ui::section(ui, "Reading the picture");
        ui::key(
            ui,
            &[
                (FORWARD, r"cyan: what goes forward: the input $x$ and the five examples $x_i$ (drag them), and in the chart the signal, each layer's output"),
                (NORMED, r"purple: the normalisation's output: $y$, on the circle where $\operatorname{rms}(y) = 1$, and the five $y_i$"),
                (BACKWARD, r"pink: gradients: $g$, arriving at an output (drag its tip), and what comes back to the input; dashed, the part of $g$ along $x$, which is dropped. In the chart, the gradient at each layer's output. In the fourth picture, a step of gradient descent at a fixed rate $\eta$, chosen so that at $c = 1$ the step is 0.4 times as long as $W$"),
                (WEIGHT, r"yellow: layer 10's weights $W$, times $c$ (drag the tip along its line), and the gain's knob"),
                (pal::DIM, r"in the chart, solid: with the normalisation chosen; dashed: without it. Dashed in the planes: the line of $x$, along which $y$ stays put, and each unit's mean $\pm$ one standard deviation"),
            ],
        );

        ui::section(ui, "Words");
        ui::term(ui, "Root mean square:", r"$\operatorname{rms}(x) = \sqrt{(x_1^2 + \dots + x_n^2)/n}$, the size of a vector per unit: its length over $\sqrt{n}$.");
        ui::term(ui, "RMS normalisation:", r"$y = x / \operatorname{rms}(x)$, the $n$ units of one example together, then a learnt gain per unit; nothing is taken off. Many current transformers use it.");
        ui::term(ui, "Layer normalisation:", r"the mean of one example's units taken off, then divided by their standard deviation $s$; then a learnt gain and bias per unit. On the way back the gradient also loses its mean: it becomes $(g - \bar g\,\mathbf{1} - (g \cdot v)\,v)/s$, with $\bar g$ the mean of $g$'s entries and $v$ the direction of the input with its mean taken off.");
        ui::term(ui, "Batch normalisation:", "each unit standardised over the examples of the batch, with their mean and standard deviation, so that one example's output, and its gradient, depend on the others. That is in training. Afterwards a network is usually run with fixed statistics (averages kept during training), and then each example is normalised on its own.");
        ui::term(ui, "Gradient:", r"$g = \partial L / \partial y$, how fast the loss $L$ changes with each entry of $y$. A layer hands back $g$ times its Jacobian, the matrix of its outputs' derivatives with respect to its inputs (scenes backpropmat, jacobian).");
        ui::term(ui, "Gain:", "the factor on the starting weights. Without a normalisation it decides whether the signal and the gradient grow or fade from layer to layer; with one, the signal after each normalisation does not depend on it.");
        ui::term(ui, r"Guard $\varepsilon$:", r"$10^{-5}$, added under the square root, so that an input that is all the same is not divided by 0. With layer and RMS normalisation it makes what is said here hold to about one part in $10^4$ instead of exactly. With batch normalisation it acts on each unit apart, and what it changes grows from layer to layer: the fourth picture's outputs then change by up to 0.006, and the gradient's length is $1/c$ times its length at $c = 1$ only to within 0.7%.");
        ui::term(ui, "The network:", r"20 layers of 24 units, each $\operatorname{ReLU}(\operatorname{norm}(W x))$ with no skip connections and no biases, on 24 points of three spiral arms in the plane, with the cross-entropy loss. Where the normalisation goes matters once there are skip connections: see the next term.");
        ui::term(ui, "Pre and post:", r"with a skip connection round each layer, the normalisation can go before the layer, inside the skip connection (pre: $x + f(\operatorname{norm}(x))$), or after the sum (post: $\operatorname{norm}(x + f(x))$). The next scene, residual, shows why that matters.");

        ui::section(ui, "Numbers");
        let o = self.one_asked();
        ui::readout_colored(ui, r"$|x|$, $\operatorname{rms}(x)$", format!("{}, {}", num(o.x.length(), 2), num(o.rms, 2)), FORWARD);
        ui::readout_colored(ui, r"$|g|$, its part along $x$", format!("{}, {}", num(o.g.length(), 2), num(o.g.dot(o.x.normalize_or_zero()), 2)), BACKWARD);
        ui::readout_colored(ui, r"$|\partial L / \partial x|$", num(o.back.length(), 2), BACKWARD);
        let m = self.many_asked();
        ui::readout_colored(ui, r"back to example 1, to the others", format!("{}, {}", num(m.back[0].length(), 2), m.back[1..].iter().map(|b| num(b.length(), 2)).collect::<Vec<_>>().join(", ")), BACKWARD);
        ui::readout(ui, "gain", num(gain_of(self.gain_k), 3));
        if let (Some(a), Some(b)) = (self.profile_without(), self.profile_with()) {
            ui::readout(ui, "each layer, without: signal, gradient", format!(r"$\times {}$, $\times {}$", num(per_layer(&a.signal), 2), num(1.0 / per_layer(&a.grad), 2)));
            ui::readout(ui, &format!("with {}", kind_long(self.kind)), format!(r"$\times {}$, $\times {}$", num(per_layer(&b.signal), 2), num(1.0 / per_layer(&b.grad), 2)));
            ui::note(ui, "(the gradient's factor counted on its way back, from layer 20 to layer 1)");
        }
        if let Some(s) = self.scaled_now() {
            ui::readout_colored(ui, r"$c$, $|W|$", format!("{}, {}", num(s.c, 3), num(s.at.w, 3)), WEIGHT);
            ui::readout_colored(ui, r"$|\nabla_W L|$, angle to $W$", format!("{}, {}°", num(s.at.w_grad, 3), num(s.angle(), 1)), BACKWARD);
            if self.c_k != 0 {
                ui::readout(ui, "largest change of an output", format!("${}$", tiny(s.out_change)));
            }
        }
        ui::see_also(
            ui,
            &[
                ("gradflow", "gradients through depth, without a normalisation"),
                ("residual", "skip connections, and where the normalisation goes"),
                ("tfgrad", "the gradient through a transformer"),
                ("archlab", "try the choices together"),
            ],
        );
    }

    fn frame(&mut self, f: &mut Frame) {
        let view = f.view_size();
        // (a canvas squeezed to a sliver is one the controls have opened over, on a phone)
        let covered = view.x < 160.0;
        let lay = Layout::of(view, f.top_inset, self.page);
        if !covered {
            // the picture is for its buttons and its dots: no moving the view
            f.handles.capture();
            f.handles.claim_clicks();
            self.buttons(f, &lay);
            self.hands_frame(f, &lay);
        }
        self.advance(f.real_dt as f64);
        if covered {
            return;
        }
        in_order(f);
        if let Some(l) = lay.one {
            self.draw_one(f, &lay, &l);
        }
        if let Some(l) = lay.many {
            self.draw_many(f, &lay, &l);
        }
        if let Some(l) = lay.deep {
            self.draw_deep(f, &lay, &l);
        }
        if let Some(l) = lay.weights {
            self.draw_weights(f, &lay, &l);
        }
        keep_batches(f);
    }
}

impl NormGrad {
    /// The buttons on the picture: the two pages', and the kinds of normalisation.
    fn buttons(&mut self, f: &mut Frame, lay: &Layout) {
        if let Some(tabs) = lay.tabs {
            for (k, (r, page)) in tabs.iter().zip([Page::Close, Page::Deep]).enumerate() {
                let words = if k == 0 { "one layer, up close" } else { "a deep network" };
                if plot::button(f, *r, words, lay.small, self.page == page) {
                    self.page = page;
                }
            }
        }
        if let Some(d) = lay.deep {
            f.text_px(d.lead, "normalisation:", lay.small, pal::DIM, Align2::LEFT_CENTER);
            for (r, n) in d.kinds.iter().zip(KINDS) {
                if plot::button(f, *r, n.name(), lay.small, self.kind == n) {
                    self.set_kind(n);
                }
            }
        }
    }

    /// The dots that can be dragged.
    fn hands_frame(&mut self, f: &mut Frame, lay: &Layout) {
        if let Some(l) = lay.one {
            let o = self.one_shown();
            let p = l.plane;
            if let Some(to) = plot::screen_handle(f, 1, p.px(o.x), FORWARD) {
                let x = kept_x(p.at(to), o.x);
                self.put_hand(H_X, x);
            }
            if let Some(to) = plot::screen_handle(f, 2, p.px(o.y + o.g), BACKWARD) {
                let y = self.one_shown().y;
                self.put_hand(H_G, kept_g(p.at(to) - y));
            }
        }
        if let Some(l) = lay.many {
            let pts = self.points(false);
            for (i, x) in pts.iter().enumerate() {
                if let Some(to) = plot::screen_handle(f, 3 + i as u32, l.planes[0].px(*x), FORWARD) {
                    self.put_hand(H_PTS + 2 * i, kept_point(l.planes[0].at(to), *x));
                }
            }
            let m = self.many_shown();
            if let Some(to) = plot::screen_handle(f, 8, l.planes[1].px(m.y[0] + m.g), BACKWARD) {
                let y = self.many_shown().y[0];
                self.put_hand(H_GB, kept_g(l.planes[1].at(to) - y));
            }
        }
        if let Some(d) = lay.deep {
            let (a, b) = d.track;
            let t = (self.chart.get(CHART_LEN - 1) + 1.0) * 0.5;
            if let Some(to) = plot::screen_handle(f, 9, a + (b - a) * t as f32, WEIGHT) {
                let t = ((to.x - a.x) / (b.x - a.x).max(1.0)) as f64;
                self.set_gain(((t * 2.0 - 1.0) * 8.0).round() as i32);
                self.then = None;
            }
        }
        if let Some(w) = lay.weights {
            let c = 2f64.powf(self.weights.get(0));
            if let Some(to) = plot::screen_handle(f, 10, w.plane.px(DVec2::new(c, 0.0)), WEIGHT) {
                let c = w.plane.at(to).x;
                self.set_factor(factor_step(c.max(factor_of(-FACTOR_STEPS))));
                self.then = None;
                self.weights.tau = QUICK;
            }
        }
    }

    /// The first picture: RMS normalisation of one example of two units.
    fn draw_one(&mut self, f: &mut Frame, lay: &Layout, l: &OneLay) {
        let p = l.plane;
        let small = lay.small;
        fitted(f, l.head, r"RMS normalisation of one example of two units, $y = x / \operatorname{rms}(x)$", lay.text, l.width, pal::TEXT);
        let o = self.one_shown();
        let (min, max) = p.rect;
        // the plane's frame and its two axes, the units
        let edge = [min, Vec2::new(max.x, min.y), max, Vec2::new(min.x, max.y), min];
        plot::path_of(f, ONE_LINES_K, &edge, 1.0, pal::FAINT.a(0.7));
        let o0 = p.px(DVec2::ZERO);
        plot::line_of(f, ONE_LINES_K, Vec2::new(min.x, o0.y), Vec2::new(max.x, o0.y), 1.0, pal::FAINT.a(0.8));
        plot::line_of(f, ONE_LINES_K, Vec2::new(o0.x, min.y), Vec2::new(o0.x, max.y), 1.0, pal::FAINT.a(0.8));
        // (the axes' names where the plane has room for them)
        let named = max.x - min.x >= 170.0;
        if named {
            f.text_px(Vec2::new(max.x - 4.0, o0.y + 3.0), "unit 1", small - 1.0, pal::DIM, Align2::RIGHT_TOP);
            f.text_px(Vec2::new(o0.x + 4.0, min.y + 3.0), "unit 2", small - 1.0, pal::DIM, Align2::LEFT_TOP);
        }
        // the circle where rms(y) = 1, and the line of x, along which y does not move
        let r = std::f64::consts::SQRT_2;
        let ring: Vec<Vec2> = (0..=96).map(|i| p.px(DVec2::from_angle(i as f64 / 96.0 * std::f64::consts::TAU) * r)).collect();
        plot::path_of(f, ONE_LINES_K, &ring, 1.8, NORMED.a(0.85));
        let u = o.x.normalize_or(DVec2::X);
        let reach = (ONE_HI - ONE_LO).length();
        let far = kept_in(o0, p.px(u * reach), p.rect);
        plot::dashed_of(f, ONE_DASHED, o0, far, 1.2, FORWARD.a(0.45));
        // the faint copy: x as it was, and what came back to it
        if let Some(gx) = self.ghost {
            let before = one(gx, o.g, &mut self.tape);
            arrow(f, ONE_GHOST, o0, p.px(gx), 2.0, FORWARD.a(0.3));
            let a = p.px(gx);
            arrow(f, ONE_GHOST, a, kept_in(a, p.px(gx + before.back), p.rect), 2.0, BACKWARD.a(0.35));
        }
        // x, y, g with its part along x, and what comes back
        let (px, py) = (p.px(o.x), p.px(o.y));
        arrow(f, ONE_ARROWS, o0, px, 2.6, FORWARD);
        plot::dashed_of(f, ONE_DASHED, py, p.px(o.y + o.along), 2.0, BACKWARD.a(0.9));
        arrow(f, ONE_ARROWS, py, p.px(o.y + o.rest), 1.3, BACKWARD.a(0.6));
        arrow(f, ONE_ARROWS, py, p.px(o.y + o.g), 2.6, BACKWARD);
        let tip = kept_in(px, p.px(o.x + o.back), p.rect);
        arrow(f, ONE_ARROWS, px, tip, 2.6, BACKWARD);
        dot(f, 0, py, 9.0, NORMED);
        dot(f, 0, o0, 5.0, pal::DIM);
        // the names, kept clear of what is drawn
        let mut room = room_in(f, p.rect);
        if named {
            let w = f.text_size("unit 2", small - 1.0);
            room.wall(Vec2::new(max.x - 4.0 - w.x, o0.y), Vec2::new(max.x, o0.y + 5.0 + w.y));
            room.wall(Vec2::new(o0.x, min.y), Vec2::new(o0.x + 6.0 + w.x, min.y + 5.0 + w.y));
        }
        for w in ring.windows(2).step_by(4) {
            room.segment_px(w[0], w[1]);
        }
        for (a, b) in [(o0, px), (py, p.px(o.y + o.g)), (py, p.px(o.y + o.along)), (px, tip), (o0, far)] {
            room.segment_px(a, b);
        }
        let screen = |v: DVec2| Vec2::new(v.x as f32, -v.y as f32);
        let size = if lay.narrow { 13.0 } else { 15.0 };
        label(&mut room, f, "x", px, "$x$", size, FORWARD, screen(u));
        label(&mut room, f, "y", py, "$y$", size, NORMED, screen(-o.g.normalize_or_zero()));
        label(&mut room, f, "g", p.px(o.y + o.g), "$g$", size, BACKWARD, screen(o.g.normalize_or_zero()));
        if o.back.length() * p.scale() as f64 > 14.0 {
            label(&mut room, f, "back", tip, r"$\partial L/\partial x$", size - 1.0, BACKWARD, screen(o.back.normalize_or_zero()));
        }
        let ring_at = p.px(DVec2::from_angle(-2.4) * r);
        label(&mut room, f, "ring", ring_at, r"$\operatorname{rms}(y) = 1$", small, NORMED, Vec2::new(-1.0, 1.0));
        if let Some(gx) = self.ghost {
            label(&mut room, f, "ghost", p.px(gx), r"$x$ before", small, FORWARD.a(0.6), screen(-u.perp()));
        }
        // the numbers
        let along = o.g.dot(u);
        let lines = [
            (format!(r"$|x| = {}$, $\operatorname{{rms}}(x) = |x|/\sqrt{{2}} = {}$; $y = x / {}$ is on the circle", num(o.x.length(), 2), num(o.rms, 2), num(o.rms, 2)), FORWARD),
            (format!(r"$g$ is ${}$ long; its part along $x$ (dashed), ${}$, is dropped, the rest is ${}$", num(o.g.length(), 2), num(along, 2), num(o.rest.length(), 2)), BACKWARD),
            (format!(r"what comes back: ${} / {} = {}$ long, at right angles to $x$", num(o.rest.length(), 2), num(o.rms, 2), num(o.back.length(), 2)), BACKWARD),
        ];
        for (k, (text, color)) in lines.iter().enumerate() {
            fitted(f, l.lines + Vec2::new(0.0, k as f32 * line_h(small)), text, small, l.width, *color);
        }
    }

    /// The second picture: batch normalisation of five examples.
    fn draw_many(&mut self, f: &mut Frame, lay: &Layout, l: &ManyLay) {
        let small = lay.small;
        fitted(f, l.head, "Batch normalisation of five examples: each unit standardised over the five", lay.text, l.width, pal::TEXT);
        let m = self.many_shown();
        let [pin, pout] = l.planes;
        // (each plane's title over it, made smaller where the plane is narrower than it)
        let side = size_of(pin.rect).x;
        for (at, text, color) in [(l.titles[0], r"their inputs $x_i$, and what comes back", FORWARD), (l.titles[1], r"outputs $y_i$; $g$ arrives at $y_1$", NORMED)] {
            let w = f.text_size(text, small).x;
            let size = if w > side && w > 0.0 { (small * side / w).max(8.0) } else { small };
            f.text_px(at, text, size, color, Align2::LEFT_BOTTOM);
        }
        for p in [pin, pout] {
            let (min, max) = p.rect;
            let edge = [min, Vec2::new(max.x, min.y), max, Vec2::new(min.x, max.y), min];
            plot::path_of(f, MANY_LINES_K, &edge, 1.0, pal::FAINT.a(0.7));
            let o0 = p.px(DVec2::ZERO);
            plot::line_of(f, MANY_LINES_K, Vec2::new(min.x, o0.y), Vec2::new(max.x, o0.y), 1.0, pal::FAINT.a(0.8));
            plot::line_of(f, MANY_LINES_K, Vec2::new(o0.x, min.y), Vec2::new(o0.x, max.y), 1.0, pal::FAINT.a(0.8));
        }
        // each unit's mean ± one standard deviation, and where that goes: the square ±1
        let boxes = [(pin, m.mean - m.sd, m.mean + m.sd, FORWARD), (pout, DVec2::splat(-1.0), DVec2::splat(1.0), NORMED)];
        for (p, lo, hi, c) in boxes {
            let (a, b) = (p.px(lo), p.px(hi));
            let corners = [a, Vec2::new(b.x, a.y), b, Vec2::new(a.x, b.y)];
            for k in 0..4 {
                plot::dashed_of(f, MANY_DASHED, corners[k], corners[(k + 1) % 4], 1.1, c.a(0.5));
            }
        }
        let mean = pin.px(m.mean);
        plot::line_of(f, MANY_LINES_K, mean - Vec2::new(5.0, 0.0), mean + Vec2::new(5.0, 0.0), 1.4, FORWARD.a(0.7));
        plot::line_of(f, MANY_LINES_K, mean - Vec2::new(0.0, 5.0), mean + Vec2::new(0.0, 5.0), 1.4, FORWARD.a(0.7));
        // the gradient at y₁, and what comes back to every example
        let y1 = pout.px(m.y[0]);
        arrow(f, MANY_ARROWS, y1, kept_in(y1, pout.px(m.y[0] + m.g), pout.rect), 2.4, BACKWARD);
        let mut tips = [Vec2::ZERO; MANY];
        for i in 0..MANY {
            let a = pin.px(m.x[i]);
            tips[i] = kept_in(a, pin.px(m.x[i] + m.back[i]), pin.rect);
            arrow(f, MANY_ARROWS, a, tips[i], if i == 0 { 2.4 } else { 1.8 }, BACKWARD);
        }
        for i in 0..MANY {
            dot(f, 1, pout.px(m.y[i]), 8.0, NORMED);
        }
        // the examples' numbers, beside their dots in both planes
        let size = small;
        for (k, p) in [pin, pout].iter().enumerate() {
            let mut room = room_in(f, p.rect);
            for i in 0..MANY {
                let at = if k == 0 { p.px(m.x[i]) } else { p.px(m.y[i]) };
                room.segment_px(at, if k == 0 { tips[i] } else { at });
                room.boxed(at - Vec2::splat(6.0), at + Vec2::splat(6.0));
            }
            room.segment_px(y1, pout.px(m.y[0] + m.g));
            for i in 0..MANY {
                let at = if k == 0 { p.px(m.x[i]) } else { p.px(m.y[i]) };
                let away = if k == 0 { -(m.back[i]) } else { m.y[i] - DVec2::ZERO };
                let towards = Vec2::new(away.x as f32, -away.y as f32);
                let color = if k == 0 { FORWARD } else { NORMED };
                label(&mut room, f, ("ex", (k * MANY + i) as u32), at, &format!("{}", i + 1), size, color, towards);
            }
        }
        // the numbers
        let others: Vec<String> = m.back[1..].iter().map(|b| num(b.length(), 2)).collect();
        let sum = m.back.iter().fold(DVec2::ZERO, |s, b| s + *b);
        let flat = m.sd.min_element() < 1e-2;
        let lines = [
            if flat {
                ("a unit with the same value in all five has nothing to standardise: the guard $\\varepsilon$ keeps it at 0".to_string(), pal::DIM)
            } else {
                (format!(r"back to example 1: ${}$; to the others: ${}$; together ${}$", num(m.back[0].length(), 2), others.join(r",\ "), tuple(&[sum.x, sum.y], 2)), BACKWARD)
            },
            (r"dashed: each unit's mean $\pm$ one standard deviation, which becomes $\pm 1$".to_string(), pal::DIM),
        ];
        for (k, (text, color)) in lines.iter().enumerate() {
            fitted(f, l.lines + Vec2::new(0.0, k as f32 * line_h(small)), text, small, l.width, *color);
        }
    }

    /// The third picture: the size of the signal and of the gradient at every layer.
    fn draw_deep(&mut self, f: &mut Frame, lay: &Layout, l: &DeepLay) {
        let small = lay.small;
        let head = if lay.narrow {
            r"20 layers $\operatorname{ReLU}(\operatorname{norm}(Wx))$: root mean square at each"
        } else {
            r"A plain network of 20 layers, each $\operatorname{ReLU}(\operatorname{norm}(Wx))$: root mean square at each layer"
        };
        fitted(f, l.head, head, lay.text, l.width, pal::TEXT);
        let (lo, hi) = (self.chart.get(4 * DEPTH), self.chart.get(4 * DEPTH + 1).max(self.chart.get(4 * DEPTH) + 1.0));
        let chart = Plot::new(l.chart, [0.5, DEPTH as f64 + 0.5], [lo, hi]);
        if chart.tiny() {
            return;
        }
        let (min, max) = l.chart;
        // the powers of ten up the side, as many as the range holds
        let (want_lo, want_hi) = self.chart.to.get(4 * DEPTH..4 * DEPTH + 2).map_or((lo, hi), |v| (v[0], v[1]));
        let span = want_hi - want_lo;
        let every = if span <= 7.0 { 1 } else if span <= 14.0 { 2 } else { 3 };
        let first = (want_lo / every as f64).ceil() as i32 * every;
        let mut e = first;
        while e as f64 <= want_hi + 1e-9 {
            let y = chart.px_y(e as f64);
            if y >= min.y - 0.5 && y <= max.y + 0.5 {
                plot::line_of(f, DEEP_TICKS, Vec2::new(min.x - 4.0, y), Vec2::new(max.x, y), 1.0, pal::DIM.a(0.14));
                f.text_px(Vec2::new(min.x - 6.0, y), format!("$10^{{{e}}}$"), small - 1.0, pal::DIM, Align2::RIGHT_CENTER);
            }
            e += every;
        }
        plot::line_of(f, DEEP_TICKS, min, Vec2::new(min.x, max.y), 1.0, pal::DIM.a(0.8));
        plot::line_of(f, DEEP_TICKS, Vec2::new(min.x, max.y), max, 1.0, pal::DIM.a(0.8));
        for layer in [1usize, 5, 10, 15, 20] {
            let x = chart.px_x(layer as f64);
            plot::line_of(f, DEEP_TICKS, Vec2::new(x, max.y), Vec2::new(x, max.y + 4.0), 1.0, pal::DIM.a(0.8));
            f.text_px(Vec2::new(x, max.y + 5.0), layer.to_string(), small - 1.0, pal::DIM, Align2::CENTER_TOP);
        }
        f.text_px(Vec2::new(max.x, max.y + 7.0 + small), "layer", small - 1.0, pal::DIM, Align2::RIGHT_TOP);
        // the four curves: kept inside the chart while its range glides, so that none is
        // cut at its edge and the batches keep their numbers of points
        let curve = |k: usize| -> Vec<Vec2> { (0..DEPTH).map(|i| chart.at((i + 1) as f64, self.chart.get(k * DEPTH + i).max(lo).min(hi))).collect() };
        for (k, color) in [(0, FORWARD), (1, BACKWARD)] {
            let hud: Vec<Vec3> = curve(k).iter().map(|p| f.hud(*p)).collect();
            plot::apart(f, DEEP_DASHED, true).polyline(hud, 1.6, color.a(0.75));
        }
        for (k, color) in [(2, FORWARD), (3, BACKWARD)] {
            let pts = curve(k);
            plot::path_of(f, DEEP_CURVES, &pts, 2.4, color);
            for p in pts {
                dot(f, 2, p, 5.5, color);
            }
        }
        // the gain's track
        let (a, b) = l.track;
        f.text_px(l.track_name, "gain of the starting weights:", small, pal::DIM, Align2::LEFT_CENTER);
        plot::line_of(f, DEEP_TRACK, a, b, 2.0, pal::DIM.a(0.6));
        for k in [-8, -4, 0, 4, 8] {
            let x = a.x + (b.x - a.x) * ((k as f32 / 8.0) + 1.0) * 0.5;
            let critical = k == 0;
            plot::line_of(f, DEEP_TRACK, Vec2::new(x, a.y - 5.0), Vec2::new(x, a.y + 5.0), if critical { 2.0 } else { 1.0 }, if critical { WEIGHT.a(0.8) } else { pal::DIM.a(0.8) });
            let name = match k {
                -8 => "0.71".to_string(),
                -4 => "1".to_string(),
                0 => r"$\sqrt{2}$, critical".to_string(),
                4 => "2".to_string(),
                _ => "2.83".to_string(),
            };
            f.text_px(Vec2::new(x, a.y + 8.0), name, small - 1.0, if critical { WEIGHT } else { pal::DIM }, Align2::CENTER_TOP);
        }
        // the legend and the factors
        let kind = format!("with {}", kind_long(self.kind));
        let entries = [(FORWARD, false, "signal"), (BACKWARD, false, "gradient"), (pal::DIM, false, kind.as_str()), (pal::DIM, true, "without")];
        let sw = 22.0;
        let widths: Vec<f32> = entries.iter().map(|e| sw + 5.0 + f.text_size(e.2, small).x + 16.0).collect();
        // (on one line where it fits, else the colours on one and the strokes on the next)
        let one_line = widths.iter().sum::<f32>() <= l.width;
        let (mut x, mut y0) = (l.lines.x, l.lines.y);
        for (k, (color, dashed, text)) in entries.into_iter().enumerate() {
            if k == 2 && !one_line {
                (x, y0) = (l.lines.x, y0 + line_h(small));
            }
            let mid = y0 + small * 0.55;
            if dashed {
                plot::dashed_of(f, DEEP_DASHED, Vec2::new(x, mid), Vec2::new(x + sw, mid), 1.6, color);
            } else {
                plot::line_of(f, DEEP_TRACK, Vec2::new(x, mid), Vec2::new(x + sw, mid), 2.4, color);
            }
            f.text_px(Vec2::new(x + sw + 5.0, y0), text, small, pal::TEXT.a(0.85), Align2::LEFT_TOP);
            x += widths[k];
        }
        if let (Some(w), Some(n)) = (self.profile_without(), self.profile_with()) {
            let text = format!(
                r"each layer, without: signal $\times {}$, gradient $\times {}$ on its way back; with: $\times {}$ and $\times {}$",
                num(per_layer(&w.signal), 2),
                num(1.0 / per_layer(&w.grad), 2),
                num(per_layer(&n.signal), 2),
                num(1.0 / per_layer(&n.grad), 2)
            );
            fitted(f, Vec2::new(l.lines.x, y0 + line_h(small)), &text, small, l.width, pal::DIM);
        }
    }

    /// The fourth picture: layer 10's weights times c.
    fn draw_weights(&mut self, f: &mut Frame, lay: &Layout, l: &WeightsLay) {
        let small = lay.small;
        let p = l.plane;
        let head = format!(r"Layer 10's weights $W$ times $c$, in the same network, with {}", kind_long(self.kind));
        fitted(f, l.head, &head, lay.text, l.width, pal::TEXT);
        let (c, s) = (2f64.powf(self.weights.get(0)), 2f64.powf(self.weights.get(1)));
        let o0 = p.px(DVec2::ZERO);
        let right = Vec2::new(p.rect.1.x - 4.0, o0.y);
        // the line of W: every cW with c > 0 has the same loss
        plot::dashed_of(f, W_DASHED, o0, right, 1.2, WEIGHT.a(0.45));
        for k in [-16, -8, 0, 8, 16] {
            let x = p.px(DVec2::new(factor_of(k), 0.0)).x;
            plot::line_of(f, W_LINES_K, Vec2::new(x, o0.y + 3.0), Vec2::new(x, o0.y + 8.0), 1.0, pal::DIM.a(0.8));
            let name = match k {
                -16 => r"$\tfrac14$",
                -8 => r"$\tfrac12$",
                0 => "1",
                8 => "2",
                _ => "4",
            };
            f.text_px(Vec2::new(x, o0.y + 10.0), name, small - 1.0, pal::DIM, Align2::CENTER_TOP);
        }
        f.text_px(Vec2::new(p.rect.1.x - 2.0, o0.y + 10.0), "$c$", small, pal::DIM, Align2::RIGHT_TOP);
        // W, the step, where W points after it, the right angle and the angle turned
        let (w_tip, s_tip) = (p.px(DVec2::new(c, 0.0)), p.px(DVec2::new(c, s)));
        plot::dashed_of(f, W_DASHED, o0, s_tip, 1.0, WEIGHT.a(0.5));
        arrow(f, W_ARROWS, o0, w_tip, 3.0, WEIGHT);
        arrow(f, W_ARROWS, w_tip, s_tip, 2.4, BACKWARD);
        let corner = (0.6 * s.min(c)).min(0.1) * p.scale() as f64;
        let k = corner as f32;
        let mark = [w_tip - Vec2::new(k, 0.0), w_tip - Vec2::new(k, k), w_tip - Vec2::new(0.0, k)];
        plot::path_of(f, W_LINES_K, &mark, 1.0, pal::TEXT.a(0.6));
        let theta = (s / c).atan();
        let r = (0.75 * c).min(0.9);
        let arc: Vec<Vec2> = (0..=24).map(|i| p.px(DVec2::from_angle(theta * i as f64 / 24.0) * r)).collect();
        plot::path_of(f, W_LINES_K, &arc, 1.4, pal::TEXT.a(0.7));
        dot(f, 3, o0, 5.0, pal::DIM);
        // names
        let mut room = room_in(f, p.rect);
        room.segment_px(o0, right);
        room.segment_px(w_tip, s_tip);
        room.segment_px(o0, s_tip);
        for w in arc.windows(2) {
            room.segment_px(w[0], w[1]);
        }
        let size = if lay.narrow { 13.0 } else { 14.0 };
        label(&mut room, f, "w", w_tip, "$cW$", size, WEIGHT, Vec2::new(0.4, 1.0));
        label(&mut room, f, "step", s_tip, r"$-\eta\,\nabla_W L$", size - 1.0, BACKWARD, Vec2::new(1.0, -0.3));
        // the numbers
        if let Some(sc) = self.scaled_now() {
            let same = if self.c_k == 0 {
                r"drag the yellow tip along its line: the network's outputs will not change".to_string()
            } else if sc.out_change < 1e-9 {
                format!(r"the network's {} outputs and its loss: as at $c = 1$, to within ${}$", sc.at.output.len(), tiny(sc.out_change.max(sc.loss_change.abs())))
            } else {
                format!(r"the network's {} outputs: within ${}$ of those at $c = 1$ (the guard $\varepsilon$)", sc.at.output.len(), tiny(sc.out_change))
            };
            let lines = [
                (same, pal::DIM),
                if self.c_k == 0 {
                    (format!(r"$|\nabla_W L| = {}$, at ${}°$ to $W$", num(sc.at.w_grad, 3), num(sc.angle(), 1)), BACKWARD)
                } else {
                    // (with batch normalisation the guard can show in the third digit)
                    let is = if num(sc.at.w_grad, 3) == num(sc.grad_at_one / sc.c, 3) { "which is" } else { "close to" };
                    (format!(r"$|\nabla_W L| = {}$, {is} ${} / c$, at ${}°$ to $W$", num(sc.at.w_grad, 3), num(sc.grad_at_one, 3), num(sc.angle(), 1)), BACKWARD)
                },
                if self.c_k == 0 {
                    (format!(r"a step of gradient descent at a rate $\eta$ turns $W$ by ${}°$", num(sc.turn(), 1)), WEIGHT)
                } else {
                    (format!(r"a step at the same rate $\eta$ turns $W$ by ${}°$ (${}°$ at $c = 1$)", num(sc.turn(), 1), num(STEP_AT_ONE.atan().to_degrees(), 1)), WEIGHT)
                },
            ];
            for (k, (text, color)) in lines.iter().enumerate() {
                fitted(f, l.lines + Vec2::new(0.0, k as f32 * line_h(small)), text, small, l.width, *color);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::math::autograd::{Var, NO_TARGET};

    /// A central difference of `loss` at `x` along its entry `k`.
    fn difference(x: &[f64], k: usize, loss: &dyn Fn(&[f64]) -> f64) -> f64 {
        let h = 1e-6 * (1.0 + x[k].abs());
        let mut moved = x.to_vec();
        moved[k] = x[k] + h;
        let up = loss(&moved);
        moved[k] = x[k] - h;
        (up - loss(&moved)) / (2.0 * h)
    }

    /// The scene as thing to try `k` leaves it, from where another left it, run at 60
    /// frames a second until it has come to rest.
    fn run(k: usize) -> NormGrad {
        let mut scene = NormGrad::new();
        scene.set_up((k + 2) % TRIALS.len());
        for _ in 0..40 {
            scene.advance(1.0 / 60.0);
        }
        scene.set_up(k);
        for _ in 0..60 * 20 {
            scene.advance(1.0 / 60.0);
            if scene.at_rest() {
                break;
            }
        }
        assert!(scene.at_rest(), "thing to try {k} has not come to rest");
        scene
    }

    // ----- the arithmetic ---------------------------------------------------------------

    /// What comes back through RMS normalisation is the difference quotient of the same
    /// loss (g · y), is the formula of the panel (g less its part along x, over rms(x)),
    /// is at right angles to x, and halves when x doubles; y is on the circle |y| = √2.
    #[test]
    fn rms_normalisation_of_one_example_is_as_the_panel_says() {
        let mut tape = Tape::new();
        let mut rng = Rng::new(3);
        for _ in 0..200 {
            let x = kept_x(DVec2::new(rng.normal() as f64, rng.normal() as f64) * 1.5, DVec2::X);
            let g = DVec2::new(rng.normal() as f64, rng.normal() as f64);
            let o = one(x, g, &mut tape);
            let loss = |v: &[f64]| {
                let mut t = Tape::new();
                let y = one(DVec2::new(v[0], v[1]), DVec2::ZERO, &mut t).y;
                g.dot(y)
            };
            for k in 0..2 {
                let fd = difference(&[x.x, x.y], k, &loss);
                assert!((fd - o.back[k]).abs() < 1e-7 * (1.0 + fd.abs()), "{x} {g}: entry {k}: {fd} against {}", o.back[k]);
            }
            let u = x.normalize();
            let formula = (g - u * g.dot(u)) / (x.length() / SQRT_2);
            // (the formula is for ε = 0: with the guard, a few parts in 10⁵ off)
            let size = g.length() / o.rms;
            assert!((o.back - formula).length() < 1e-4 * size, "{x} {g}");
            assert!(o.back.dot(u).abs() < 1e-4 * size);
            assert!((o.y.length() - SQRT_2).abs() < 1e-4 && (o.y.normalize() - u).length() < 1e-12);
            assert!((o.along + o.rest - g).length() < 1e-12 && o.rest.dot(u).abs() < 1e-12);
            let twice = one(x * 2.0, g, &mut tape);
            assert!((twice.back * 2.0 - o.back).length() < 1e-4 * size, "{x}");
            assert!((twice.y - o.y).length() < 1e-4);
        }
        // a gradient along x: nothing comes back
        let o = one(X0, X0.normalize() * 0.9, &mut tape);
        assert!(o.back.length() < 1e-4, "{}", o.back);
    }

    /// What the panel says of layer normalisation, on the library's operation for an
    /// example of six units: what comes back is the difference quotient of the loss, is
    /// at right angles to the input and to the all-ones direction (RMS normalisation's is
    /// not, to the latter), is the formula of the words, and halves when the input
    /// doubles; a shift of the input changes nothing.
    #[test]
    fn layer_normalisation_also_drops_the_mean() {
        let mut rng = Rng::new(8);
        let back_of = |x: &[f64], g: &[f64], layer: bool| {
            let mut t = Tape::new();
            let xv = t.param(1, x.len(), x);
            let yv = if layer { t.layer_norm(xv, EPS) } else { t.rms_norm(xv, EPS) };
            t.backward_seeded(yv, g);
            (t.value(yv).to_vec(), t.grad(xv).to_vec())
        };
        let dot = |a: &[f64], b: &[f64]| a.iter().zip(b).map(|(x, y)| x * y).sum::<f64>();
        let mut kept_mean = 0;
        for _ in 0..50 {
            // (spread out, so that the guard ε stays out of the way: here it acts in
            // the fifth digit)
            let x: Vec<f64> = (0..6).map(|_| 1.0 + 2.0 * rng.normal() as f64).collect();
            let g: Vec<f64> = (0..6).map(|_| rng.normal() as f64).collect();
            let (y, back) = back_of(&x, &g, true);
            let loss = |v: &[f64]| dot(&g, &back_of(v, &g, true).0);
            for k in 0..6 {
                let fd = difference(&x, k, &loss);
                assert!((fd - back[k]).abs() < 1e-6 * (1.0 + fd.abs()), "{fd} against {}", back[k]);
            }
            let scale = dot(&back, &back).sqrt();
            assert!(back.iter().sum::<f64>().abs() < 1e-9 * scale && dot(&back, &x).abs() < 1e-4 * scale * dot(&x, &x).sqrt());
            // (g − ḡ1 − (g·v) v)/s, v the direction of x with its mean taken off
            let m = x.iter().sum::<f64>() / 6.0;
            let c: Vec<f64> = x.iter().map(|v| v - m).collect();
            let s = (dot(&c, &c) / 6.0).sqrt();
            let v: Vec<f64> = c.iter().map(|a| a / (6.0 * s * s).sqrt()).collect();
            let gbar = g.iter().sum::<f64>() / 6.0;
            let gv = dot(&g, &v);
            for k in 0..6 {
                let formula = (g[k] - gbar - gv * v[k]) / s;
                assert!((formula - back[k]).abs() < 1e-4 * dot(&g, &g).sqrt() / s);
            }
            let (y2, back2) = back_of(&x.iter().map(|v| 2.0 * v - 3.0).collect::<Vec<_>>(), &g, true);
            for k in 0..6 {
                assert!((y2[k] - y[k]).abs() < 1e-4 && (2.0 * back2[k] - back[k]).abs() < 1e-4 * scale.max(1e-3), "{x:?} {g:?}");
            }
            // RMS normalisation keeps the gradient's mean
            let (_, rms_back) = back_of(&x, &g, false);
            assert!(dot(&rms_back, &x).abs() < 1e-4 * dot(&rms_back, &rms_back).sqrt() * dot(&x, &x).sqrt());
            kept_mean += (rms_back.iter().sum::<f64>().abs() > 1e-3) as usize;
        }
        assert!(kept_mean >= 45, "{kept_mean}");
    }

    /// What comes back through batch normalisation is the difference quotient of the
    /// same loss (g · y₁) for every example; it reaches all five, adds up to nothing,
    /// and each unit's part is at right angles to that unit's outputs. With layer or RMS
    /// normalisation (one example at a time) nothing reaches the others.
    #[test]
    fn batch_normalisation_ties_the_examples_and_the_others_do_not() {
        let mut tape = Tape::new();
        let m = many(&PTS0, GB0, &mut tape);
        let flat: Vec<f64> = PTS0.iter().flat_map(|p| [p.x, p.y]).collect();
        let loss = |v: &[f64]| {
            let pts: [DVec2; MANY] = std::array::from_fn(|i| DVec2::new(v[2 * i], v[2 * i + 1]));
            let mut t = Tape::new();
            GB0.dot(many(&pts, DVec2::ZERO, &mut t).y[0])
        };
        for k in 0..2 * MANY {
            let fd = difference(&flat, k, &loss);
            let got = m.back[k / 2][k % 2];
            assert!((fd - got).abs() < 1e-7 * (1.0 + fd.abs()), "entry {k}: {fd} against {got}");
        }
        assert!(m.back.iter().all(|b| b.length() > 0.05), "{:?}", m.back);
        let sum = m.back.iter().fold(DVec2::ZERO, |s, b| s + *b);
        assert!(sum.length() < 1e-12);
        for j in 0..2 {
            let dot: f64 = (0..MANY).map(|i| m.back[i][j] * m.y[i][j]).sum();
            // (but for the guard ε, a few parts in 10⁵)
            assert!(dot.abs() < 1e-4, "unit {j}: {dot}");
            let mean: f64 = (0..MANY).map(|i| m.y[i][j]).sum::<f64>() / MANY as f64;
            let square: f64 = (0..MANY).map(|i| m.y[i][j] * m.y[i][j]).sum::<f64>() / MANY as f64;
            assert!(mean.abs() < 1e-12 && (square - 1.0).abs() < 1e-4);
        }
        // the same five, one example at a time: the others get nothing back
        for norm in [Norm::Layer, Norm::Rms] {
            let mut t = Tape::new();
            let x = t.param(MANY, 2, &flat);
            let y = if norm == Norm::Layer { t.layer_norm(x, EPS) } else { t.rms_norm(x, EPS) };
            let mut seed = [0.0; 2 * MANY];
            (seed[0], seed[1]) = (GB0.x, GB0.y);
            t.backward_seeded(y, &seed);
            assert!(t.grad(x)[2..].iter().all(|v| *v == 0.0) && t.grad(x)[..2].iter().any(|v| *v != 0.0), "{norm:?}");
        }
    }

    /// The network as the library builds it, recorded again by hand from its weights
    /// with a matrix added to layer `at`'s output: the loss as a function of that matrix.
    fn loss_with(model: &Model, data: &Data, norm: Norm, at: usize, added: &[f64]) -> f64 {
        let mut t = Tape::new();
        let mut x = t.constant(EXAMPLES, 2, &data.x);
        for l in 1..=DEPTH {
            let w = model.group(&format!("layer {l} weights")).expect("weights");
            let wv = t.constant(w.rows, w.cols, &w.value);
            let z = t.matmul_t(x, wv);
            let z = match norm {
                Norm::None => z,
                Norm::Batch => t.batch_norm(z, EPS),
                Norm::Layer => t.layer_norm(z, EPS),
                Norm::Rms => t.rms_norm(z, EPS),
            };
            x = t.act(z, Act::Relu);
            if l == at {
                let d = t.constant(EXAMPLES, WIDTH, added);
                x = t.add(x, d);
            }
        }
        let w = model.group("read-out weights").expect("read-out");
        let wv = t.constant(w.rows, w.cols, &w.value);
        let out = t.matmul_t(x, wv);
        let targets: Vec<usize> = data.y.iter().map(|&c| if c >= 0.0 { c as usize } else { NO_TARGET }).collect();
        let loss = t.cross_entropy(out, &targets);
        t.value(loss)[0]
    }

    /// Every gradient the chart and the fourth picture show is the derivative of the
    /// same loss: the gradient at a layer's output against differences of the loss with
    /// a small matrix added there, and layer 10's weights' against differences of the
    /// loss in those weights; with each normalisation and none, at three gains.
    #[test]
    fn the_networks_gradients_are_those_of_its_loss() {
        let data = Data::new();
        let mut rng = Rng::new(17);
        for norm in Norm::ALL {
            for k in [-6, 0, 5] {
                let model = network(norm, k, 1.0);
                let mut tape = Tape::new();
                let tr = model.loss_mlp(&mut tape, &data.x, &data.y, Loss::CrossEntropy);
                tape.backward(tr.loss);
                let zero = vec![0.0; EXAMPLES * WIDTH];
                assert!((loss_with(&model, &data, norm, 0, &zero) - tape.value(tr.loss)[0]).abs() < 1e-12 * (1.0 + tape.value(tr.loss)[0].abs()));
                let largest = (1..=DEPTH).flat_map(|l| tape.grad(tr.get(Stage::Output, l).unwrap()).to_vec()).fold(0.0f64, |a, b| a.max(b.abs()));
                for l in [1, 7, DEPTH] {
                    let v: Var = tr.get(Stage::Output, l).unwrap();
                    for _ in 0..3 {
                        let e = rng.below(EXAMPLES * WIDTH);
                        let fd = difference(&zero, e, &|d| loss_with(&model, &data, norm, l, d));
                        let got = tape.grad(v)[e];
                        // (the floor: what rounding leaves of a difference of losses near 1)
                        assert!((fd - got).abs() <= 1e-5 * fd.abs().max(got.abs()) + 1e-7 * largest + 1e-9, "{norm:?} gain {k} layer {l} entry {e}: {fd} against {got}");
                    }
                }
                let i = model.index_of(SCALED_NAME).unwrap();
                let w = model.groups()[i].value.clone();
                let grad = tape.grad(tr.params[i]).to_vec();
                let wl = |v: &[f64]| {
                    let mut m = model.clone();
                    m.groups_mut()[i].value = v.to_vec();
                    let mut t = Tape::new();
                    let tr = m.loss_mlp(&mut t, &data.x, &data.y, Loss::CrossEntropy);
                    t.value(tr.loss)[0]
                };
                let big = grad.iter().fold(0.0f64, |a, b| a.max(b.abs()));
                for _ in 0..3 {
                    let e = rng.below(w.len());
                    let fd = difference(&w, e, &wl);
                    assert!((fd - grad[e]).abs() <= 1e-5 * fd.abs().max(grad[e].abs()) + 1e-7 * big + 1e-9, "{norm:?} gain {k} weight {e}: {fd} against {}", grad[e]);
                }
                // and the profile is what the pass gives
                let p = pass(&model, &data, &mut tape);
                assert!((p.w_grad - grad.iter().map(|g| g * g).sum::<f64>().sqrt()).abs() < 1e-12 * (1.0 + p.w_grad));
            }
        }
    }

    /// Going forward, with each normalisation the signal at every hidden layer is the
    /// same whatever the gain: with layer and RMS normalisation to 3 parts in 10⁴, with batch
    /// normalisation to within a percent (the guard ε acts on each unit apart there, and
    /// what it changes grows from layer to layer); without one it is not.
    #[test]
    fn the_signal_after_a_normalisation_does_not_depend_on_the_gain() {
        let data = Data::new();
        let mut tape = Tape::new();
        for norm in KINDS {
            let first = pass(&network(norm, -GAIN_STEPS, 1.0), &data, &mut tape);
            for k in -GAIN_STEPS + 1..=GAIN_STEPS {
                let p = pass(&network(norm, k, 1.0), &data, &mut tape);
                let most = if norm == Norm::Batch { 1e-2 } else { 3e-4 };
                for (a, b) in p.signal.iter().zip(&first.signal) {
                    assert!((a / b - 1.0).abs() < most, "{norm:?} at step {k}: {a} against {b}");
                }
            }
        }
        let (a, b) = (pass(&network(Norm::None, -4, 1.0), &data, &mut tape), pass(&network(Norm::None, 4, 1.0), &data, &mut tape));
        assert!(b.signal[DEPTH - 1] / a.signal[DEPTH - 1] > 1e5);
    }

    /// The fourth picture's facts, at every gain and every other factor the knobs reach,
    /// for each normalisation: the outputs stay (to 1e-12 with layer and RMS
    /// normalisation, to the guard's 0.006 with batch normalisation), the gradient times
    /// c stays (to 0.7% with batch normalisation), and it is at right angles to the
    /// weights; the step relative to W goes as 1/c².
    #[test]
    fn scaled_weights_keep_the_output_and_divide_the_gradient() {
        let data = Data::new();
        let mut tape = Tape::new();
        for norm in KINDS {
            let (most, off) = if norm == Norm::Batch { (6e-3, 7e-3) } else { (1e-12, 1e-12) };
            for g in -GAIN_STEPS..=GAIN_STEPS {
                let at_one = pass(&network(norm, g, 1.0), &data, &mut tape);
                for k in (-FACTOR_STEPS..=FACTOR_STEPS).step_by(2) {
                    let c = factor_of(k);
                    let s = Scaled::new(c, pass(&network(norm, g, c), &data, &mut tape), &at_one);
                    assert!(s.out_change < most && s.loss_change.abs() < most, "{norm:?} gain {g} c {c}: {} {}", s.out_change, s.loss_change);
                    assert!((c * s.at.w_grad / at_one.w_grad - 1.0).abs() < off, "{norm:?} gain {g} c {c}");
                    assert!((s.angle() - 90.0).abs() < 0.01, "{norm:?} gain {g} c {c}: {}", s.angle());
                    assert!((s.at.w - c * at_one.w).abs() < 1e-9 * s.at.w);
                    assert!((s.step / s.c * c * c / STEP_AT_ONE - 1.0).abs() < off.max(1e-12) + 1e-12);
                }
            }
        }
    }

    // ----- the things to try -------------------------------------------------------------

    /// What the primer says of the gradient, at every gain: with layer and RMS
    /// normalisation it stays within a factor of 4 of level; with batch normalisation it
    /// grows by 1.3 to 1.4 a layer on its way back.
    #[test]
    fn the_gradient_with_each_normalisation_is_as_the_primer_says() {
        let data = Data::new();
        let mut tape = Tape::new();
        for k in -GAIN_STEPS..=GAIN_STEPS {
            for norm in [Norm::Layer, Norm::Rms] {
                let p = pass(&network(norm, k, 1.0), &data, &mut tape);
                assert!(spread(&p.grad) < 4.0, "{norm:?} at step {k}: {}", spread(&p.grad));
            }
            let p = pass(&network(Norm::Batch, k, 1.0), &data, &mut tape);
            let up = 1.0 / per_layer(&p.grad);
            assert!((1.25..1.45).contains(&up), "batch at step {k}: {up}");
        }
    }

    /// The note under the gain: at √2, without normalisation, each layer multiplies the
    /// signal by about 0.97 in this network.
    #[test]
    fn the_critical_gain_is_as_the_note_says() {
        let p = pass(&network(Norm::None, 0, 1.0), &Data::new(), &mut Tape::new());
        assert_eq!(num(per_layer(&p.signal), 2), "0.97");
        assert_eq!(num(gain_of(0), 4), num(SQRT_2, 4));
    }

    /// The scene opens on the first thing to try, with the network at gain 2 with layer
    /// normalisation, and c = 1.
    #[test]
    fn the_scene_opens_as_described() {
        let mut opened = NormGrad::new();
        assert!(opened.at_rest() && opened.page == Page::Close && opened.kind == Norm::Layer && opened.c_k == 0);
        assert_eq!(num(gain_of(opened.gain_k), 3), "2.000");
        let first = run(0);
        assert!(opened.hands.asked(H_X) == first.hands.asked(H_X) && opened.hands.asked(H_G) == first.hands.asked(H_G));
        // at gain 2 without normalisation, the dashed curves fan out
        let (a, b) = (opened.profile_without().unwrap().clone(), opened.profile_with().unwrap().clone());
        assert!(a.signal[DEPTH - 1] / a.signal[0] > 300.0 && a.grad[0] / a.grad[DEPTH - 1] > 300.0);
        assert!(spread(&b.signal) < 1.3 && spread(&b.grad) < 4.0);
        let _ = opened.one_asked();
    }

    #[test]
    fn the_first_thing_to_try_does_what_its_caption_says() {
        let mut scene = run(0);
        let o = scene.one_asked();
        let u = o.x.normalize();
        assert_eq!([num(o.g.length(), 2), num(o.g.dot(u), 2), num(o.rest.length(), 2), num(o.rms, 2), num(o.back.length(), 2)], ["0.98", "0.65", "0.74", "0.60", "1.23"]);
        // (but for the guard ε: a few parts in 10⁵)
        assert!(o.back.dot(u).abs() < 1e-4 * o.back.length());
        assert!(TRIALS[0].look_for.contains("0.98 long") && TRIALS[0].look_for.contains("(dashed), 0.65") && TRIALS[0].look_for.contains("0.74") && TRIALS[0].look_for.contains("= 0.60") && TRIALS[0].look_for.contains("1.23 long"));
    }

    #[test]
    fn the_second_thing_to_try_does_what_its_caption_says() {
        let mut scene = run(1);
        let o = scene.one_asked();
        let before = one(scene.ghost.expect("the faint copy"), o.g, &mut scene.tape);
        assert_eq!([num(before.x.length(), 2), num(o.x.length(), 2)], ["1.00", "2.00"]);
        assert!((o.x.normalize() - before.x.normalize()).length() < 1e-12 && (o.y - before.y).length() < 1e-4);
        assert_eq!([num(before.back.length(), 2), num(o.back.length(), 2)], ["1.04", "0.52"]);
        assert!((before.back.length() / o.back.length() - 2.0).abs() < 1e-4);
        assert!(TRIALS[1].look_for.contains("from 1.04 to 0.52"));
    }

    #[test]
    fn the_third_thing_to_try_does_what_its_caption_says() {
        let scene = run(2);
        assert!(scene.kind == Norm::Layer && scene.page == Page::Deep && num(gain_of(scene.gain_k), 3) == "1.000");
        let (a, b) = (scene.profile_without().unwrap(), scene.profile_with().unwrap());
        // without: about 0.69 a layer; a 1400th left after 20; the gradient a 1700th at layer 1
        assert_eq!(num(per_layer(&a.signal), 2), "0.69");
        assert_eq!(((a.signal[0] / a.signal[DEPTH - 1]) / 100.0).round() * 100.0, 1400.0);
        assert_eq!(((a.grad[DEPTH - 1] / a.grad[0]) / 100.0).round() * 100.0, 1700.0);
        // with layer normalisation: the signal 0.63 to 0.77, the gradient within a factor of 4
        let (lo, hi) = b.signal.iter().fold((f64::MAX, 0.0f64), |(lo, hi), &v| (lo.min(v), hi.max(v)));
        assert_eq!([num(lo, 2), num(hi, 2)], ["0.63", "0.77"]);
        assert!(spread(&b.grad) < 4.0 && spread(&b.grad) > 3.0, "{}", spread(&b.grad));
        // batch normalisation: as level a signal, a gradient that grows by about 1.3 a layer on its way back
        let mut tape = Tape::new();
        let batch = pass(&network(Norm::Batch, scene.gain_k, 1.0), &scene.data, &mut tape);
        assert!(spread(&batch.signal) < 1.1);
        assert_eq!(num(1.0 / per_layer(&batch.grad), 1), "1.3");
        assert!(TRIALS[2].look_for.contains("about 0.69") && TRIALS[2].look_for.contains("1400th") && TRIALS[2].look_for.contains("1700th") && TRIALS[2].look_for.contains("0.63 to 0.77") && TRIALS[2].look_for.contains("by about 1.3 a layer"));
    }

    #[test]
    fn the_fourth_thing_to_try_does_what_its_caption_says() {
        let mut scene = run(3);
        let now = scene.many_asked();
        assert!(now.x[1] == PT2_TO && (0..MANY).filter(|&i| i != 1).all(|i| now.x[i] == PTS0[i]) && now.g == GB0);
        let before = many(&PTS0, GB0, &mut scene.tape);
        assert!(before.back.iter().all(|b| b.length() > 0.2), "something comes back to all five");
        assert!((now.y[0] - before.y[0]).length() > 0.5, "example 1's output moves");
        assert_eq!([num(before.back[0].length(), 2), num(now.back[0].length(), 2)], ["1.21", "0.74"]);
        let turn = (before.back[0].to_angle() - now.back[0].to_angle()).to_degrees();
        assert_eq!(num(turn, 0), "16");
        assert!((before.back[0].x - now.back[0].x).abs() < 1e-12, "unit 1's part stays");
        assert!(TRIALS[3].look_for.contains("from 1.21 to 0.74") && TRIALS[3].look_for.contains("by 16°"));
    }

    #[test]
    fn the_fifth_thing_to_try_does_what_its_caption_says() {
        let scene = run(4);
        assert!(scene.kind == Norm::Layer && scene.gain_k == 0 && num(factor_of(scene.c_k), 2) == "4.00");
        let s = scene.scaled_now().unwrap();
        assert!(s.out_change < 5e-15 && s.loss_change.abs() < 5e-15, "the same to 14 decimal places: {} {}", s.out_change, s.loss_change);
        assert_eq!([num(s.at.w_grad, 3), num(s.grad_at_one, 3), num(s.angle(), 1)], ["0.178", "0.711", "90.0"]);
        assert_eq!([num(s.turn(), 1), num(STEP_AT_ONE.atan().to_degrees(), 1)], ["1.4", "21.8"]);
        assert!(((s.step / s.c) / STEP_AT_ONE - 1.0 / 16.0).abs() < 1e-5);
        assert!(TRIALS[4].look_for.contains("0.178 instead of 0.711") && TRIALS[4].look_for.contains("1.4° instead of 21.8°") && TRIALS[4].look_for.contains("14 decimal places"));
    }

    // ----- what may go wrong ----------------------------------------------------------------

    /// Whatever the hands do (ends, the middle, on top of one another, numbers that are
    /// not numbers) and whatever the settings, every number drawn stays finite and the
    /// scene comes to rest.
    #[test]
    fn degenerate_input_leaves_everything_finite() {
        let mut scene = NormGrad::new();
        let odd = [DVec2::ZERO, DVec2::NAN, DVec2::new(1e9, -1e9), DVec2::new(0.0, 1e-12), DVec2::new(-2.0, 0.5)];
        for (n, p) in odd.iter().enumerate() {
            let before = scene.hands.vec(H_X);
            scene.put_hand(H_X, kept_x(*p, before));
            scene.put_hand(H_G, kept_g(*p));
            for i in 0..MANY {
                let b = scene.hands.vec(H_PTS + 2 * i);
                scene.put_hand(H_PTS + 2 * i, kept_point(if n % 2 == 0 { *p } else { DVec2::new(1.0, 1.0) }, b));
            }
            scene.put_hand(H_GB, kept_g(*p * 3.0));
            scene.set_kind(KINDS[n % 3]);
            scene.set_gain([i32::MIN, i32::MAX, 0, -3, 8][n]);
            scene.set_factor([i32::MAX, i32::MIN, 3, 0, -16][n]);
            for _ in 0..60 * 10 {
                scene.advance(1.0 / 60.0);
                if scene.at_rest() {
                    break;
                }
            }
            assert!(scene.at_rest(), "{p}");
            let o = scene.one_asked();
            assert!(o.x.length() >= X_LEAST - 1e-12 && o.x.length() <= X_MOST + 1e-12);
            assert!(o.back.is_finite() && o.y.is_finite() && o.rest.is_finite() && o.g.length() <= G_MOST + 1e-12, "{p}: {o:?}");
            let m = scene.many_asked();
            assert!(m.back.iter().chain(&m.y).all(|v| v.is_finite()) && m.sd.is_finite(), "{p}: {m:?}");
            assert!(scene.chart.shown.iter().chain(&scene.weights.shown).all(|v| v.is_finite()));
            let s = scene.scaled_now().unwrap();
            assert!(s.turn().is_finite() && s.angle().is_finite() && s.step.is_finite());
            for text in [num(o.back.length(), 2), num(m.back[0].length(), 2), tiny(s.out_change), num(s.turn(), 1)] {
                assert!(!text.contains('?') && !text.contains("NaN") && !text.contains("inf"), "{text}");
            }
        }
        // five examples on one point: nothing to standardise, and still finite
        let m = many(&[DVec2::ONE; MANY], GB0, &mut Tape::new());
        assert!(m.back.iter().chain(&m.y).all(|v| v.is_finite()) && m.y.iter().all(|y| y.length() == 0.0));
    }

    /// The moves the things to try play go smoothly: no drawn point jumps between two
    /// frames, at 60 and at 240 frames a second.
    #[test]
    fn the_moves_glide() {
        for hz in [60.0, 240.0] {
            for k in 0..TRIALS.len() {
                let mut scene = NormGrad::new();
                scene.set_up(k);
                let mut last = scene.hands.shown.clone();
                let mut last_c = scene.weights.shown.clone();
                for _ in 0..(hz as usize) * 8 {
                    scene.advance(1.0 / hz);
                    let most = scene.hands.shown.iter().zip(&last).map(|(a, b)| (a - b).abs()).fold(0.0, f64::max);
                    let most_c = scene.weights.shown.iter().zip(&last_c).map(|(a, b)| (a - b).abs()).fold(0.0, f64::max);
                    assert!(most < 6.0 / hz && most_c < 6.0 / hz, "thing to try {k} at {hz} Hz: {most} {most_c}");
                    (last, last_c) = (scene.hands.shown.clone(), scene.weights.shown.clone());
                }
                assert!(scene.at_rest(), "thing to try {k} at {hz} Hz");
            }
        }
    }

    /// The pictures fit the canvas, from a phone to a desktop, inside it and clear of
    /// one another and of the buttons.
    #[test]
    fn the_layout_fits_every_canvas() {
        for (w, h) in [(1078.0, 1000.0), (578.0, 700.0), (390.0, 800.0), (390.0, 560.0), (800.0, 390.0), (1600.0, 1000.0), (240.0, 180.0), (620.0, 480.0), (1300.0, 700.0), (2400.0, 1400.0), (900.0, 640.0)] {
            for inset in [0.0, 46.0] {
                for page in [Page::Close, Page::Deep] {
                    let view = Vec2::new(w, h);
                    let lay = Layout::of(view, inset, page);
                    let within = |r: Rect| r.0.cmpge(Vec2::splat(-0.5)).all() && r.1.cmple(view + Vec2::splat(0.5)).all() && r.0.y >= inset;
                    let mut boxes: Vec<Rect> = Vec::new();
                    if let Some(tabs) = lay.tabs {
                        assert!(tabs.iter().all(|r| within(*r)), "{w}x{h}: tabs");
                        boxes.extend(tabs);
                    }
                    if let Some(l) = lay.one {
                        let r = (l.head, Vec2::new(l.head.x + l.width, l.lines.y + ONE_LINES as f32 * line_h(lay.small)));
                        assert!(within(l.plane.rect) && within(r), "{w}x{h}: the first picture {r:?}");
                        assert!((size_of(l.plane.rect).x - size_of(l.plane.rect).y).abs() < 0.5);
                        boxes.push(r);
                    }
                    if let Some(l) = lay.many {
                        let r = (l.head, Vec2::new(l.head.x + l.width, l.lines.y + MANY_LINES as f32 * line_h(lay.small)));
                        assert!(within(l.planes[0].rect) && within(l.planes[1].rect) && within(r) && l.planes[0].rect.1.x < l.planes[1].rect.0.x, "{w}x{h}: the second picture");
                        boxes.push(r);
                    }
                    if let Some(l) = lay.deep {
                        let r = (l.head, Vec2::new(l.head.x + l.width, l.lines.y + DEEP_LINES as f32 * line_h(lay.small)));
                        assert!(within(l.chart) && within(r) && l.kinds.iter().all(|k| within(*k) && k.1.y < l.chart.0.y), "{w}x{h}: the chart {r:?}");
                        assert!(l.track.0.y > l.chart.1.y + 20.0 && l.track.1.x > l.track.0.x + 80.0 && l.lines.y > l.track.0.y + 20.0);
                        boxes.push(r);
                    }
                    if let Some(l) = lay.weights {
                        let r = (l.head, Vec2::new(l.head.x + l.width, l.lines.y + W_LINES as f32 * line_h(lay.small)));
                        assert!(within(l.plane.rect) && within(r), "{w}x{h}: the fourth picture {r:?}");
                        boxes.push(r);
                    }
                    for (i, a) in boxes.iter().enumerate() {
                        for b in &boxes[..i] {
                            assert!(a.1.x <= b.0.x || b.1.x <= a.0.x || a.1.y <= b.0.y || b.1.y <= a.0.y, "{w}x{h} {page:?}: {a:?} and {b:?} overlap");
                        }
                    }
                    // on a desktop all four, on a phone the page's two
                    if w >= 900.0 && h >= 640.0 {
                        assert!(lay.one.is_some() && lay.many.is_some() && lay.deep.is_some() && lay.weights.is_some(), "{w}x{h}");
                    } else if w >= 390.0 && h >= 560.0 {
                        let shown = [lay.one.is_some(), lay.many.is_some(), lay.deep.is_some(), lay.weights.is_some()];
                        assert_eq!(shown, if page == Page::Close { [true, true, false, false] } else { [false, false, true, true] }, "{w}x{h} {page:?}");
                    }
                }
            }
        }
    }

    #[test]
    #[ignore]
    fn explore_guard() {
        let data = Data::new();
        let mut tape = Tape::new();
        for norm in KINDS {
            let mut most: (f64, i32, i32) = (0.0, 0, 0);
            let mut grad_most = 0.0f64;
            for k in -GAIN_STEPS..=GAIN_STEPS {
                let at_one = pass(&network(norm, k, 1.0), &data, &mut tape);
                for ck in -FACTOR_STEPS..=FACTOR_STEPS {
                    let c = factor_of(ck);
                    let s = Scaled::new(c, pass(&network(norm, k, c), &data, &mut tape), &at_one);
                    if s.out_change > most.0 {
                        most = (s.out_change, k, ck);
                    }
                    grad_most = grad_most.max((c * s.at.w_grad / at_one.w_grad - 1.0).abs()).max((s.angle() - 90.0).abs() / 90.0);
                }
            }
            println!("{norm:?}: largest change of an output {:.3e} at gain step {} factor step {}; gradient off by {:.2e}", most.0, most.1, most.2, grad_most);
        }
        let mut worst = 0.0f64;
        for i in 0..2000 {
            let x = DVec2::from_angle(i as f64) * (X_LEAST + (X_MOST - X_LEAST) * (i % 7) as f64 / 6.0);
            let o = one(x, DVec2::new(0.3, 1.0), &mut tape);
            worst = worst.max(o.back.dot(x.normalize()).abs() / o.back.length());
        }
        println!("one: worst cos {worst:.2e}");
    }

    /// The numbers behind the captions (run with --ignored --nocapture).
    #[test]
    #[ignore]
    fn explore() {
        let mut tape = Tape::new();
        let data = Data::new();
        for norm in Norm::ALL {
            for k in [-8, -4, 0, 4, 8] {
                let p = pass(&network(norm, k, 1.0), &data, &mut tape);
                println!(
                    "{} gain {:.3}: loss {:.4}; signal x{:.3}/layer ({:.3} .. {:.3e}, spread {:.2}), grad x{:.3}/layer back ({:.3e} at {DEPTH}, {:.3e} at 1, spread {:.2}); |W10| {:.4} |dW10| {:.5}",
                    norm.name(),
                    gain_of(k),
                    p.loss,
                    per_layer(&p.signal),
                    p.signal[0],
                    p.signal[DEPTH - 1],
                    spread(&p.signal),
                    1.0 / per_layer(&p.grad),
                    p.grad[DEPTH - 1],
                    p.grad[0],
                    spread(&p.grad),
                    p.w,
                    p.w_grad
                );
            }
        }
        for k in 0..TRIALS.len() {
            let mut s = run(k);
            let o = s.one_asked();
            let m = s.many_asked();
            println!("trial {k}: one {o:?} |g| {:.4} along {:.4} rest {:.4} back {:.4}; many back1 {:.4} {:.2} deg", o.g.length(), o.g.dot(o.x.normalize()), o.rest.length(), o.back.length(), m.back[0].length(), m.back[0].to_angle().to_degrees());
            if let Some(sc) = s.scaled_now() {
                println!("   c {} |dW| {:.5} at one {:.5} angle {:.3} turn {:.3} out {:.2e} loss {:.2e}", sc.c, sc.at.w_grad, sc.grad_at_one, sc.angle(), sc.turn(), sc.out_change, sc.loss_change);
            }
        }
    }
}

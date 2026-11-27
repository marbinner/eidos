//! Adam, taken apart. One weight is fed a stream of gradients (a steady part, noise the
//! visitor sets, and on request a sudden change), and each piece of Adam is drawn at
//! work on it: the running average of the gradients, the running average of their
//! squares, the corrections for their start at zero, ε, and the step that comes out.
//! Under it, two weights seen from above: in a long valley, lined up with the axes and
//! turned, and under the two kinds of weight decay.
//!
//! Everything is worked out at once, in f64, with the rules exactly as their sources
//! give them (Kingma and Ba 2015, Algorithm 1; AdaGrad, RMSProp and AdamW as the PyTorch
//! 2.15 documentation writes them; course notes, noise.md §7.1, §8 and classics.md),
//! and the picture glides to what was worked out.

use super::*;
use crate::scenes::stats::plot::{self, Glide, Plot, Rect, AXES, BUTTONS, CURVES, MARKS, OWN};

pub const DEF: SceneDef = SceneDef {
    id: "adam",
    title: "Adam, taken apart",
    group: GROUP,
    blurb: "Two running averages and a division: what each piece of Adam does to a stream of noisy gradients, and when the whole behaves like the sign of the gradient.",
    make: |_| Box::new(Apart::create()),
};

// ----- the rules, in f64 ---------------------------------------------------------------

/// Adam's constants, as in Kingma and Ba (2015), Algorithm 1. `corrected`: divide the two
/// running averages by 1 − βᵗ, the corrections for their start at zero.
#[derive(Clone, Copy, Debug, PartialEq)]
struct Consts {
    beta1: f64,
    beta2: f64,
    eps: f64,
    corrected: bool,
}

/// Kingma and Ba's defaults ("good default settings for the tested machine learning
/// problems"), which are PyTorch's too.
const DEFAULTS: Consts = Consts { beta1: 0.9, beta2: 0.999, eps: 1e-8, corrected: true };

/// What Adam remembers of one weight: the two running averages and the step count.
#[derive(Clone, Copy, Debug, Default)]
struct Moments {
    m: f64,
    v: f64,
    t: i32,
}

/// What one step of Adam works out from one gradient. The step itself is −η `u`.
#[derive(Clone, Copy, Debug)]
struct Pieces {
    m_hat: f64,
    v_hat: f64,
    u: f64,
}

impl Moments {
    /// Take in the gradient `g` and work out the step (Algorithm 1, element by element).
    fn take(&mut self, g: f64, c: &Consts) -> Pieces {
        self.t = self.t.saturating_add(1);
        self.m = c.beta1 * self.m + (1.0 - c.beta1) * g;
        self.v = c.beta2 * self.v + (1.0 - c.beta2) * g * g;
        let (m_hat, v_hat) = if c.corrected { (self.m / (1.0 - c.beta1.powi(self.t)), self.v / (1.0 - c.beta2.powi(self.t))) } else { (self.m, self.v) };
        Pieces { m_hat, v_hat, u: m_hat / (v_hat.max(0.0).sqrt() + c.eps) }
    }
}

/// RMSProp's `alpha` as PyTorch sets it by default.
const RMS_ALPHA: f64 = 0.99;

/// One step of RMSProp (PyTorch, without momentum, not centred), in units of its
/// learning rate. It has no correction for the start of `v` at zero.
fn rmsprop(v: &mut f64, g: f64, eps: f64) -> f64 {
    *v = RMS_ALPHA * *v + (1.0 - RMS_ALPHA) * g * g;
    g / (v.sqrt() + eps)
}

/// One step of AdaGrad (PyTorch, no decay of the rate, accumulator from 0), in units
/// of its learning rate: the squares are summed and never forgotten.
fn adagrad(sum: &mut f64, g: f64, eps: f64) -> f64 {
    *sum += g * g;
    g / (sum.sqrt() + eps)
}

/// The two kinds of weight decay with Adam.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Decay {
    /// λw added to the gradient, which Adam then averages and divides (an L2 penalty
    /// of λw²/2 on the loss: PyTorch's `Adam` with `weight_decay`).
    Coupled,
    /// λw taken off the weight directly, w ← w − ηλw, before Adam's step (PyTorch's
    /// `AdamW`: the decay per step is the learning rate times λ).
    Decoupled,
}

/// One step of Adam with weight decay on one weight, whose loss gradient is `g`.
fn decayed_step(w: &mut f64, mo: &mut Moments, g: f64, eta: f64, lambda: f64, decay: Decay, c: &Consts) {
    match decay {
        Decay::Coupled => {
            let p = mo.take(g + lambda * *w, c);
            *w -= eta * p.u;
        }
        Decay::Decoupled => {
            let p = mo.take(g, c);
            *w = *w * (1.0 - eta * lambda) - eta * p.u;
        }
    }
}

// ----- the stream of gradients ---------------------------------------------------------

/// The most steps the stream runs for.
const MOST_STEPS: usize = 3000;
const LEAST_STEPS: usize = 10;

/// A sudden change of the gradient, a quarter of the way along the stream.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Event {
    None,
    /// The steady part is ten times larger from then on.
    Jump,
    /// The steady part changes sign.
    Flip,
    /// One gradient a hundred times the usual size.
    Spike,
}

const EVENTS: [Event; 4] = [Event::None, Event::Jump, Event::Flip, Event::Spike];
const JUMP: f64 = 10.0;
const SPIKE: f64 = 100.0;

impl Event {
    fn button(self, narrow: bool) -> &'static str {
        match (self, narrow) {
            (Event::None, _) => "no change",
            (Event::Jump, _) => "a jump",
            (Event::Flip, false) => "a change of sign",
            (Event::Flip, true) => "a sign change",
            (Event::Spike, _) => "a spike",
        }
    }
}

/// Which picture the two squares show.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Pair {
    Valley,
    Decay,
}

/// What the stream depends on: the steady part, the noise, the scale, the steps shown, the
/// sudden change, the seed, β₁, β₂ and ε (their bits) and the corrections.
type StreamKey = (f64, f64, f64, usize, Event, u64, [u64; 3], bool);

/// Everything the scene shows depends on, and nothing else.
#[derive(Clone, Copy, Debug, PartialEq)]
struct Setting {
    /// The steady part of the gradient, and the standard deviation of its noise.
    mean: f64,
    noise: f64,
    /// Everything about the gradients is multiplied by this (a loss so many times larger).
    scale: f64,
    /// Steps of the stream shown.
    steps: usize,
    event: Event,
    seed: u64,
    c: Consts,
    /// Beside Adam's step: gradient descent's, AdaGrad's and RMSProp's.
    sgd: bool,
    adagrad: bool,
    rmsprop: bool,
    pair: Pair,
    /// The right-hand valley's turn, in degrees.
    angle: f64,
    /// The learning rate of the two weights, and the weight decay.
    eta: f64,
    lambda: f64,
}

impl Setting {
    /// The same, with every number inside the range its control allows.
    fn kept(mut self) -> Setting {
        let keep = |v: f64, lo: f64, hi: f64, or: f64| if v.is_finite() { v.max(lo).min(hi) } else { or };
        self.mean = keep(self.mean, -3.0, 3.0, 1.0);
        self.noise = keep(self.noise, 0.0, 5.0, 0.0);
        self.scale = keep(self.scale, 1e-12, 1e4, 1.0);
        self.steps = self.steps.clamp(LEAST_STEPS, MOST_STEPS);
        self.c.beta1 = keep(self.c.beta1, 0.0, 0.999, 0.9);
        self.c.beta2 = keep(self.c.beta2, 0.0, 0.9999, 0.999);
        self.c.eps = keep(self.c.eps, 1e-12, 1.0, 1e-8);
        self.angle = keep(self.angle, 0.0, 180.0, 45.0);
        self.eta = keep(self.eta, 0.001, 0.5, 0.05);
        self.lambda = keep(self.lambda, 0.0, 1.0, 0.0);
        self
    }

    /// The step (counted from 1) at which the sudden change comes.
    fn event_step(&self) -> usize {
        (self.steps / 4).max(2)
    }

    /// The usual size of a gradient, before scaling: the root of the mean square of the
    /// steady part and the noise (1 if both are 0, so that a spike is still a spike).
    fn usual(&self) -> f64 {
        let r = (self.mean * self.mean + self.noise * self.noise).sqrt();
        if r > 0.0 {
            r
        } else {
            1.0
        }
    }

    /// The steady part at step `t`, before scaling.
    fn steady_at(&self, t: usize) -> f64 {
        let after = t >= self.event_step();
        match self.event {
            Event::Jump if after => self.mean * JUMP,
            Event::Flip if after => -self.mean,
            _ => self.mean,
        }
    }

    /// The gradient at step `t`, before scaling, with the noise's standard normal `z`.
    fn gradient_at(&self, t: usize, z: f64) -> f64 {
        if self.event == Event::Spike && t == self.event_step() {
            return SPIKE * self.usual() * if self.mean < 0.0 { -1.0 } else { 1.0 };
        }
        self.steady_at(t) + self.noise * z
    }

    /// What the stream depends on (the squares do not).
    fn stream_key(&self) -> StreamKey {
        (self.mean, self.noise, self.scale, self.steps, self.event, self.seed, [self.c.beta1.to_bits(), self.c.beta2.to_bits(), self.c.eps.to_bits()], self.c.corrected)
    }

    /// What the squares depend on.
    fn pair_key(&self) -> (Pair, [u64; 6], bool) {
        let c = &self.c;
        (self.pair, [self.angle.to_bits(), self.eta.to_bits(), self.lambda.to_bits(), c.beta1.to_bits(), c.beta2.to_bits(), c.eps.to_bits()], c.corrected)
    }
}

/// The stream's noise: standard normal numbers, the same for the same seed.
fn noise_of(seed: u64) -> Vec<f64> {
    let mut rng = Rng::new(seed);
    (0..MOST_STEPS).map(|_| rng.normal() as f64).collect()
}

/// The stream worked out, one entry a step (step `t` at index `t − 1`). The gradients
/// and Adam's two averages are divided by the scale, so that a change of scale leaves
/// them where they are (only the numbers on the axis change); the steps are in units
/// of each rule's own learning rate.
#[derive(Clone, Debug, Default)]
struct Stream {
    g: Vec<f64>,
    m_hat: Vec<f64>,
    /// √v̂ + ε, Adam's divisor.
    size: Vec<f64>,
    u: Vec<f64>,
    /// Adam's step with the corrections the other way (off if they are on).
    other: Vec<f64>,
    /// Gradient descent's step ηg, in units of η: the gradient itself (scaled).
    sgd: Vec<f64>,
    adagrad: Vec<f64>,
    rmsprop: Vec<f64>,
}

impl Stream {
    fn of(s: &Setting, z: &[f64]) -> Stream {
        let n = MOST_STEPS.min(z.len());
        let c = s.c;
        let other = Consts { corrected: !c.corrected, ..c };
        let scale = s.scale;
        let (mut a, mut b) = (Moments::default(), Moments::default());
        let (mut rms, mut sum) = (0.0, 0.0);
        let mut out = Stream::default();
        for (i, z) in z.iter().enumerate().take(n) {
            let g = s.gradient_at(i + 1, *z);
            let actual = g * scale;
            let p = a.take(actual, &c);
            let q = b.take(actual, &other);
            out.g.push(g);
            out.m_hat.push(p.m_hat / scale);
            out.size.push((p.v_hat.max(0.0).sqrt() + c.eps) / scale);
            out.u.push(p.u);
            out.other.push(q.u);
            out.sgd.push(actual);
            out.adagrad.push(adagrad(&mut sum, actual, c.eps));
            out.rmsprop.push(rmsprop(&mut rms, actual, c.eps));
        }
        out
    }

    /// Adam's steps over the first `n` steps.
    fn shown(&self, n: usize) -> &[f64] {
        &self.u[..n.min(self.u.len())]
    }
}

/// The numbers the captions and the read-outs quote about the stream's first `steps` steps.
#[derive(Clone, Copy, Debug)]
struct Summary {
    first: f64,
    /// The largest step in size, and its step number.
    largest: (f64, usize),
    /// The mean and standard deviation of the steps after the first 100 (after the
    /// first half, if fewer than 200 are shown), and the step that window starts at.
    from: usize,
    mean_late: f64,
    sd_late: f64,
    /// After the sudden change: how many steps are below half of η in the direction
    /// of the steady part.
    small_after: usize,
}

fn summary(s: &Setting, st: &Stream) -> Summary {
    let u = st.shown(s.steps);
    let first = u.first().copied().unwrap_or(0.0);
    let largest = u.iter().enumerate().fold((0.0f64, 1usize), |best, (i, &x)| if x.abs() > best.0.abs() { (x, i + 1) } else { best });
    let from = 101.min(u.len() / 2 + 1);
    let late = &u[(from - 1).min(u.len())..];
    let k = late.len().max(1) as f64;
    let mean_late = late.iter().sum::<f64>() / k;
    let sd_late = (late.iter().map(|x| (x - mean_late) * (x - mean_late)).sum::<f64>() / k).sqrt();
    let t0 = s.event_step();
    let way = if s.steady_at(u.len().max(1)) < 0.0 { -1.0 } else { 1.0 };
    let small_after = if s.event == Event::None { 0 } else { u.iter().skip(t0).filter(|x| **x * way < 0.5).count() };
    Summary { first, largest, from, mean_late, sd_late, small_after }
}

/// The mean step once the averages have settled, ε left out: the steady part over the
/// root mean square of the gradient, between −1 and 1 (Kingma and Ba 2015, Sec. 2.1;
/// Balles and Hennig 2018, Eqs. 7–9; course notes, noise.md D6).
fn settled_ratio(mean: f64, noise: f64) -> f64 {
    let r = (mean * mean + noise * noise).sqrt();
    if r > 0.0 {
        mean / r
    } else {
        0.0
    }
}

// ----- the two weights -----------------------------------------------------------------

/// The valley: ½(ALONG a² + ACROSS b²), with a measured along its floor and b across it.
const ALONG: f64 = 1.0;
const ACROSS: f64 = 100.0;
/// Where the walks start, in the valley's own coordinates (along, across).
const VALLEY_START: [f64; 2] = [-3.0, 0.5];
/// The fixed step that suits this valley best for gradient descent, 2/(smallest +
/// largest curvature): it shrinks the worst error fastest.
const GD_RATE: f64 = 2.0 / (ALONG + ACROSS);
/// How far the loss must fall (as a share of where it started) to count as arrived.
const ARRIVED: f64 = 1e-4;
/// Steps the two weights are run for.
const PAIR_STEPS: usize = 2000;
/// The sizes of the two weights' gradients in the weight-decay picture.
const DECAY_SIZES: [f64; 2] = [100.0, 1.0];

/// A walk of two weights: where they were after each step (the start first), and the
/// loss there.
#[derive(Clone, Debug, Default)]
struct Walk {
    at: Vec<[f64; 2]>,
    loss: Vec<f64>,
}

impl Walk {
    /// The first step at which the loss is at most `share` of where it started.
    fn steps_to(&self, share: f64) -> Option<usize> {
        let start = *self.loss.first()?;
        self.loss.iter().position(|&l| l <= share * start)
    }

    /// The first step at which weight `i` is at most half of where it started.
    fn half_life(&self, i: usize) -> Option<usize> {
        let start = self.at.first()?[i];
        self.at.iter().position(|w| w[i] <= 0.5 * start)
    }
}

/// The valley turned by `angle` degrees: its direction along the floor and across it.
fn valley_axes(angle: f64) -> ([f64; 2], [f64; 2]) {
    let (s, c) = angle.to_radians().sin_cos();
    ([c, s], [-s, c])
}

fn valley_loss(w: [f64; 2], along: [f64; 2], across: [f64; 2]) -> (f64, [f64; 2]) {
    let a = w[0] * along[0] + w[1] * along[1];
    let b = w[0] * across[0] + w[1] * across[1];
    let (ga, gb) = (ALONG * a, ACROSS * b);
    (0.5 * (ALONG * a * a + ACROSS * b * b), [ga * along[0] + gb * across[0], ga * along[1] + gb * across[1]])
}

/// Gradient descent at its best fixed step and Adam at `eta`, from the same start
/// (turned with the valley), for `PAIR_STEPS` steps.
fn valley_walks(angle: f64, eta: f64, c: &Consts) -> [Walk; 2] {
    let (along, across) = valley_axes(angle);
    let start = [VALLEY_START[0] * along[0] + VALLEY_START[1] * across[0], VALLEY_START[0] * along[1] + VALLEY_START[1] * across[1]];
    let mut walks = [Walk::default(), Walk::default()];
    for (k, walk) in walks.iter_mut().enumerate() {
        let mut w = start;
        let mut mo = [Moments::default(); 2];
        for step in 0..=PAIR_STEPS {
            let (loss, g) = valley_loss(w, along, across);
            walk.at.push(w);
            walk.loss.push(loss);
            if step == PAIR_STEPS {
                break;
            }
            for i in 0..2 {
                w[i] -= if k == 0 { GD_RATE * g[i] } else { eta * mo[i].take(g[i], c).u };
            }
        }
    }
    walks
}

/// The two weights' loss gradient at step `t`: it flips sign at every step, so that on
/// average it pushes neither way.
fn flipping(t: usize, size: f64) -> f64 {
    if t % 2 == 1 {
        size
    } else {
        -size
    }
}

/// The two weights from (1, 1) under each kind of decay: [coupled, decoupled].
fn decay_walks(eta: f64, lambda: f64, c: &Consts) -> [Walk; 2] {
    [Decay::Coupled, Decay::Decoupled].map(|decay| {
        let mut w = [1.0, 1.0];
        let mut mo = [Moments::default(); 2];
        let mut walk = Walk::default();
        walk.at.push(w);
        for t in 1..=PAIR_STEPS {
            for i in 0..2 {
                decayed_step(&mut w[i], &mut mo[i], flipping(t, DECAY_SIZES[i]), eta, lambda, decay, c);
            }
            walk.at.push(w);
        }
        walk
    })
}

/// The squares' walks: [left first, left second, right first, right second]. A valley:
/// gradient descent and Adam, lined up and turned. Weight decay: the coupled kind on
/// the left and the decoupled on the right (each twice, the first copy unseen).
fn pair_walks(s: &Setting) -> [Walk; 4] {
    match s.pair {
        Pair::Valley => {
            let [a, b] = valley_walks(0.0, s.eta, &s.c);
            let [c, d] = valley_walks(s.angle, s.eta, &s.c);
            [a, b, c, d]
        }
        Pair::Decay => {
            let [coupled, decoupled] = decay_walks(s.eta, s.lambda, &s.c);
            [coupled.clone(), coupled, decoupled.clone(), decoupled]
        }
    }
}

// ----- the experiments -----------------------------------------------------------------

struct Trial {
    button: &'static str,
    caption: &'static str,
    setting: Setting,
}

const OPENING: Setting = Setting {
    mean: 1.0,
    noise: 1.0,
    scale: 1.0,
    steps: 300,
    event: Event::None,
    seed: 1,
    c: DEFAULTS,
    sgd: false,
    adagrad: false,
    rmsprop: false,
    pair: Pair::Valley,
    angle: 45.0,
    eta: 0.05,
    lambda: 0.1,
};

const TRIALS: [Trial; 7] = [
    Trial {
        button: "a steady gradient",
        caption: r"The gradient is about 1 at every step, with a little noise. Adam's first step is exactly $\eta$: with the corrections, $\hat m_1 = g_1$ and $\hat v_1 = g_1^2$, so it is $\eta\, g_1/|g_1|$, whatever $g_1$ is. The later steps stay between 0.94 and 1.04 of $\eta$: with little noise $\hat m/\sqrt{\hat v}$ is close to 1, and Adam moves the weight by about $\eta$ in the direction of the gradient's sign, which is sign descent. AdaGrad (green) adds up the squares and never forgets them: with a steady gradient its step $t$ is $1/\sqrt t$ of its learning rate, a tenth of it by step 100.",
        setting: Setting { mean: 1.0, noise: 0.1, steps: 200, adagrad: true, ..OPENING },
    },
    Trial {
        button: "a thousand times larger",
        caption: r"The same gradients times 1000, as from a loss a thousand times larger. $\hat m$ and $\sqrt{\hat v}$ grow a thousandfold (see the numbers on the upper chart's axis), their ratio does not, and Adam's steps are what they were at scale 1 to within one part in $10^{7}$. Gradient descent's steps (cyan, $\eta g$) are a thousand times larger and leave the chart. Only $\varepsilon$ does not scale: slide the scale down to $10^{-10}$, where the gradients are about a hundredth of $\varepsilon = 10^{-8}$. The divisor is then almost $\varepsilon$ alone, and the steps shrink to about a hundredth of $\eta$: Adam has become gradient descent with momentum, with the step $\eta/\varepsilon$.",
        setting: Setting { mean: 1.0, noise: 1.0, scale: 1000.0, steps: 200, sgd: true, ..OPENING },
    },
    Trial {
        button: "mostly noise",
        caption: r"The steady part is 0.1 and the noise 1. $\sqrt{\hat v}$ settles near the root mean square of the gradients, $\sqrt{0.1^2 + 1^2} \approx 1$, while $\hat m$ wanders about the steady part, so over steps 101 to 3000 of this run the steps average 0.097 of $\eta$ (in a long run, $\bar g/\sqrt{\bar g^2 + \sigma^2} = 0.0995$), with a spread of 0.22 about that. Sign descent would take a full step of $\eta$ every time, 46 % of them the wrong way. Adam brakes a weight whose gradients are mostly noise: its average step is about the steady part divided by the noise.",
        setting: Setting { mean: 0.1, noise: 1.0, steps: 3000, ..OPENING },
    },
    Trial {
        button: "one huge spike",
        caption: r"At step 500 one gradient is 100 times the usual size. The step at the spike is 2.1 times $\eta$. Within 50 steps $\hat m$ is back within 5 % of the steady part, but $\sqrt{\hat v}$, with its memory of about $1/(1-\beta_2) = 1000$ steps, still remembers the spike: 1372 of the 1500 steps after it are under half of $\eta$. With $\beta_2 = 0.99$ it is 339 steps; with $\beta_1$ at 0.5 or 0.99 instead of 0.9 it stays between 1365 and 1386. The longer memory decides.",
        setting: Setting { mean: 1.0, noise: 0.1, steps: 2000, event: Event::Spike, ..OPENING },
    },
    Trial {
        button: "the corrections switched off",
        caption: r"A steady gradient of 1, no noise, and no corrections. $m$ and $v$ start at 0 and grow towards their true values, $\sqrt v$ more slowly than $m$, so the steps come out too large: the first is $(1-\beta_1)/\sqrt{1-\beta_2} = 3.16$ times $\eta$, they peak at 6.57 at step 12, and at step 100 they are still 3.24 (the faint line: with the corrections, exactly $\eta$ from the first). Step $t$ is $(1-\beta_1^t)/\sqrt{1-\beta_2^t}$. RMSProp (purple) has no correction either: its first step is $1/\sqrt{1-\alpha} = 10$ times its learning rate, with PyTorch's $\alpha = 0.99$.",
        setting: Setting { mean: 1.0, noise: 0.0, steps: 100, c: Consts { corrected: false, ..DEFAULTS }, rmsprop: true, ..OPENING },
    },
    Trial {
        button: "a valley, lined up and turned",
        caption: r"A long valley, a hundred times steeper across than along: on the left lined up with the axes, on the right turned by 45°. Gradient descent (cyan, at the fixed step that suits this valley best, $2/101$) takes the same path in both, turned, and needs 231 steps to bring the loss to a ten-thousandth of where it started. Adam ($\eta = 0.05$) divides each coordinate by its own $\sqrt{\hat v}$. Lined up, that treats the steep and the shallow direction apart, and it needs 101 steps; turned, each coordinate sees both directions mixed, and it needs 379 (at 90° it is lined up again: 101). A ruler that rescales each coordinate (scene steepest) fits a valley lined up with the axes, not one turned. Drag the blue dot to turn the valley.",
        setting: Setting { pair: Pair::Valley, angle: 45.0, eta: 0.05, ..OPENING },
    },
    Trial {
        button: "two kinds of weight decay",
        caption: r"Two weights whose gradients flip sign at every step, of size 100 for $w_1$ and 1 for $w_2$: on their own they would stay within 0.02 of where they start. Decay $\lambda = 0.1$, $\eta = 0.01$. Left, the decay is added to the gradient (an L2 penalty), and Adam divides it by $\sqrt{\hat v}$ with the rest: $w_2$ halves in 673 steps, while $w_1$, whose gradients are 100 times larger, ends at 0.96 after 2000 steps, against 0.98 with no decay at all. Right, AdamW takes the decay off the weights: both halve in 675 steps, and the path runs straight down the diagonal. For plain gradient descent without momentum the two kinds are the same step.",
        setting: Setting { pair: Pair::Decay, eta: 0.01, lambda: 0.1, ..OPENING },
    },
];

// ----- the scene -----------------------------------------------------------------------

pub struct Apart {
    setting: Setting,
    /// The stream's noise, for `setting.seed`.
    noise: (u64, Vec<f64>),
    /// What was worked out, and for which settings.
    stream: Stream,
    stream_for: Option<StreamKey>,
    walks: [Walk; 4],
    walks_for: Option<(Pair, [u64; 6], bool)>,
    /// What the picture glides to (worked out again only when what it shows changes).
    goal: Goals,
    look: Look,
    /// Is the right-hand valley being turned by hand?
    turning: bool,
    /// The window last drawn in (a new one is drawn at once, without gliding).
    canvas: Option<Vec2>,
}

impl Apart {
    /// (Nothing here needs the graphics device, so the tests can make one too.)
    fn create() -> Apart {
        let mut s = Apart {
            setting: OPENING,
            noise: (OPENING.seed, noise_of(OPENING.seed)),
            stream: Stream::default(),
            stream_for: None,
            walks: Default::default(),
            walks_for: None,
            goal: Goals::default(),
            look: Look::default(),
            turning: false,
            canvas: None,
        };
        s.refresh();
        s
    }

    /// Set up one of the experiments in `TRIALS`.
    fn set_up(&mut self, k: usize) {
        if let Some(t) = TRIALS.get(k) {
            self.setting = t.setting;
            self.refresh();
        }
    }

    /// Bring what is worked out up to date with the settings.
    fn refresh(&mut self) {
        self.setting = self.setting.kept();
        let s = self.setting;
        if self.noise.0 != s.seed {
            self.noise = (s.seed, noise_of(s.seed));
        }
        if self.stream_for != Some(s.stream_key()) {
            self.stream = Stream::of(&s, &self.noise.1);
            self.stream_for = Some(s.stream_key());
            self.goal.stream = stream_goal(&self.stream);
            self.goal.setting = None;
        }
        if self.walks_for != Some(s.pair_key()) {
            self.walks = pair_walks(&s);
            self.walks_for = Some(s.pair_key());
            self.goal.walks = walks_goal(&self.walks);
        }
        if self.goal.setting != Some(s) {
            self.goal.frame = frame_goal(&s, &self.stream);
            self.goal.setting = Some(s);
        }
    }

    fn summary(&self) -> Summary {
        summary(&self.setting, &self.stream)
    }
}

// ----- the layout ----------------------------------------------------------------------

/// Room under a square for its numbers and two lines about it.
const BELOW_SQUARE: f32 = 58.0;

/// Where everything goes in a window.
#[derive(Clone, Copy, Debug, PartialEq)]
struct Layout {
    view: Vec2,
    narrow: bool,
    /// Only the two charts of the stream: the window is too small for the rest.
    compact: bool,
    big: f32,
    text: f32,
    small: f32,
    /// The two charts of the stream (their plots' rectangles).
    stream: Rect,
    steps: Rect,
    events: [Rect; 4],
    beside: [Rect; 3],
    pairs: [Rect; 2],
    squares: [Rect; 2],
    /// Room for a few lines on the two weights, right of the squares (a wide window).
    notes: Option<Rect>,
}

impl Layout {
    fn of(view: Vec2, inset: f32) -> Layout {
        let view = view.max(Vec2::splat(1.0));
        let narrow = view.x < 700.0;
        let (big, text, small) = if narrow { (13.0, 11.5, 10.5) } else { (15.0, 12.5, 11.5) };
        let side = if narrow { 10.0 } else { 22.0 };
        let top = inset.max(0.0) + if narrow { 8.0 } else { 12.0 };
        let compact = view.y - top < if narrow { 560.0 } else { 600.0 } || view.x < 300.0;
        let r = |x0: f32, y0: f32, x1: f32, y1: f32| (Vec2::new(x0, y0), Vec2::new(x1.max(x0 + 1.0), y1.max(y0 + 1.0)));
        let none = r(0.0, 0.0, 0.0, 0.0);
        // room left of a chart for its numbers, and right of it for the names of its lines
        let (left, right) = if narrow { (34.0, 6.0) } else { (46.0, 96.0) };
        let (x0, x1) = (side + left, view.x - side - right);
        let button_h = if narrow { 24.0 } else { 26.0 };
        let mut lay = Layout {
            view,
            narrow,
            compact,
            big,
            text,
            small,
            stream: none,
            steps: none,
            events: [none; 4],
            beside: [none; 3],
            pairs: [none; 2],
            squares: [none; 2],
            notes: None,
        };
        if compact && view.y - top < 330.0 {
            // the step alone: the window is too short for more
            lay.steps = r(x0, top + 30.0, x1, view.y - 24.0);
            return lay;
        }
        if compact {
            // the two charts and nothing else, sharing the height
            let room = (view.y - top - 44.0 - 26.0 - 44.0 - 26.0).max(40.0);
            let h1 = room * 0.62;
            lay.stream = r(x0, top + 44.0, x1, top + 44.0 + h1);
            lay.steps = r(x0, lay.stream.1.y + 26.0 + 44.0, x1, lay.stream.1.y + 26.0 + 44.0 + room - h1);
            return lay;
        }
        // above the first chart: the buttons for a sudden change, then its title
        let row = |n: usize, x0: f32, x1: f32, y: f32| plot::even_row(n, x0, x1, y, button_h, 6.0);
        let events = if narrow { row(4, side, view.x - side, top) } else { row(4, side, (side + 4.0 * 150.0).min(view.x - side), top) };
        for (k, e) in events.into_iter().enumerate().take(4) {
            lay.events[k] = e;
        }
        let head = if narrow { 30.0 } else { 32.0 };
        let axis = if narrow { 22.0 } else { 26.0 };
        let y = top + button_h + 6.0 + head;
        // the heights left for the two charts and the squares, shared out
        let fixed = (y - top) + axis + head + axis + 10.0 + button_h + 8.0 + 20.0 + BELOW_SQUARE + 4.0;
        let room = (view.y - top - fixed).max(120.0);
        let width_room = if narrow { (view.x - 2.0 * side - 2.0 * 30.0 - 14.0) / 2.0 } else { ((view.x - 2.0 * side) * 0.62 - 2.0 * 40.0 - 30.0) / 2.0 };
        let square = (room * if narrow { 0.42 } else { 0.4 }).min(width_room).min(if narrow { 260.0 } else { 340.0 }).max(60.0);
        let charts = (room - square).max(80.0);
        let h1 = charts * 0.64;
        lay.stream = r(x0, y, x1, y + h1);
        let y = lay.stream.1.y + axis + head;
        lay.steps = r(x0, y, x1, y + charts - h1);
        // the buttons for what to draw beside Adam's step, at the right of its title line
        let widths = if narrow { [104.0, 60.0, 66.0] } else { [118.0, 80.0, 88.0] };
        let mut bx = view.x - side;
        for k in (0..3).rev() {
            lay.beside[k] = r(bx - widths[k], y - head + 2.0, bx, y - head + 2.0 + button_h - 2.0);
            bx -= widths[k] + 6.0;
        }
        // the squares, with the buttons choosing what they show above them
        let y = lay.steps.1.y + axis + 10.0;
        let pairs = if narrow { row(2, side, view.x - side, y) } else { row(2, side, side + 2.0 * 160.0, y) };
        for (k, p) in pairs.into_iter().enumerate().take(2) {
            lay.pairs[k] = p;
        }
        let y = y + button_h + 8.0 + 20.0;
        let left = if narrow { 30.0 } else { 40.0 };
        let gap = if narrow { 14.0 } else { 30.0 };
        let a = side + left;
        lay.squares[0] = r(a, y, a + square, y + square);
        let b = a + square + gap + left;
        lay.squares[1] = r(b, y, b + square, y + square);
        if !narrow {
            let n0 = lay.squares[1].1.x + 30.0;
            if view.x - side - n0 > 200.0 {
                lay.notes = Some(r(n0, y - 20.0, view.x - side, y + square + 30.0));
            }
        }
        lay
    }
}

// ----- what is drawn, gliding ----------------------------------------------------------

// The picture's own numbers, in the order the glide keeps them.
/// Steps shown (the charts' x range).
const FR_STEPS: usize = 0;
/// Half the height of the upper chart, in its units (divided by the scale), and the
/// upper end of the lower one.
const FR_Y1: usize = 1;
const FR_Y2: usize = 2;
/// How strongly each line beside Adam's step is drawn (0 to 1).
const FR_SGD: usize = 3;
const FR_ADAGRAD: usize = 4;
const FR_RMS: usize = 5;
const FR_OTHER: usize = 6;
/// The sudden change: where, and how strongly its mark is drawn.
const FR_EVENT: usize = 7;
const FR_EVENT_ON: usize = 8;
/// ε divided by the scale (where its line is on the upper chart).
const FR_EPS: usize = 9;
/// The squares' range (the same across and up), the right valley's turn (degrees), and
/// how strongly the valley's level lines and gradient descent are drawn.
const FR_LO: usize = 10;
const FR_HI: usize = 11;
const FR_ANGLE: usize = 12;
const FR_VALLEY: usize = 13;
/// The lower end of the lower chart (`FR_Y2` is its upper end).
const FR_Y2_LO: usize = 14;
/// The average step over the window the read-outs average over, and how strongly its
/// line is drawn.
const FR_MEAN: usize = 15;
const FR_MEAN_ON: usize = 16;
/// The steady part before the sudden change and after it.
const FR_STEADY: usize = 17;
const FR_STEADY_AFTER: usize = 18;
const FR_LEN: usize = 19;

/// The stream's arrays in the glide, one after another, `MOST_STEPS` each.
const S_G: usize = 0;
const S_M: usize = 1;
const S_SIZE: usize = 2;
const S_U: usize = 3;
const S_OTHER: usize = 4;
const S_SGD: usize = 5;
const S_ADAGRAD: usize = 6;
const S_RMS: usize = 7;
const S_LEN: usize = 8;

/// What the picture glides to, kept while the settings stay the same.
#[derive(Clone, Debug, Default)]
struct Goals {
    setting: Option<Setting>,
    frame: Vec<f32>,
    stream: Vec<f32>,
    walks: Vec<f32>,
}

#[derive(Clone, Debug, Default)]
struct Look {
    frame: Glide,
    stream: Glide,
    /// The four walks of the squares, (x, y) after (x, y), `PAIR_STEPS + 1` points each.
    walks: Glide,
    moving: bool,
}

/// The largest size among values (and 0 if there are none).
fn reach(values: impl Iterator<Item = f64>) -> f64 {
    values.filter(|v| v.is_finite()).fold(0.0, |m, v| m.max(v.abs()))
}

fn frame_goal(s: &Setting, st: &Stream) -> Vec<f32> {
    let n = s.steps.min(st.u.len());
    let spike = (s.event == Event::Spike).then(|| s.event_step());
    // the upper chart: the gradients (but for a spike, which may leave it) and Adam's divisor
    let y1 = reach(st.g[..n].iter().enumerate().filter(|(i, _)| Some(i + 1) != spike).map(|(_, g)| *g)).max(reach(st.size[..n].iter().copied()));
    let y1 = if y1 > 0.0 { 1.1 * y1 } else { 1.0 };
    // the lower: the steps drawn (gradient descent's only while they are not huge),
    // from −1.25 to 1.25 at least
    let mut drawn: Vec<&[f64]> = vec![&st.u[..n]];
    if !s.c.corrected {
        drawn.push(&st.other[..n]);
    }
    if s.adagrad {
        drawn.push(&st.adagrad[..n]);
    }
    if s.rmsprop {
        drawn.push(&st.rmsprop[..n]);
    }
    if s.sgd && reach(st.sgd[..n].iter().copied()) <= 12.0 {
        drawn.push(&st.sgd[..n]);
    }
    let (mut low, mut high) = (0.0f64, 0.0f64);
    for list in drawn {
        for v in list.iter().filter(|v| v.is_finite()) {
            (low, high) = (low.min(*v), high.max(*v));
        }
    }
    let (y2_lo, y2) = ((1.1 * low).min(-1.25).max(-12.0), (1.1 * high).max(1.25).min(12.0));
    let sum = summary(s, st);
    let on = |b: bool| if b { 1.0 } else { 0.0 };
    let (lo, hi) = match s.pair {
        Pair::Valley => (-3.6, 3.6),
        Pair::Decay => (-0.15, 1.15),
    };
    let mut g = vec![0.0f32; FR_LEN];
    g[FR_STEPS] = s.steps as f32;
    g[FR_Y1] = y1 as f32;
    g[FR_Y2] = y2 as f32;
    g[FR_SGD] = on(s.sgd);
    g[FR_ADAGRAD] = on(s.adagrad);
    g[FR_RMS] = on(s.rmsprop);
    g[FR_OTHER] = on(!s.c.corrected);
    g[FR_EVENT] = s.event_step() as f32;
    g[FR_EVENT_ON] = on(s.event != Event::None);
    g[FR_EPS] = (s.c.eps / s.scale).min(1e30) as f32;
    g[FR_LO] = lo;
    g[FR_HI] = hi;
    g[FR_ANGLE] = s.angle as f32;
    g[FR_VALLEY] = on(s.pair == Pair::Valley);
    g[FR_Y2_LO] = y2_lo as f32;
    g[FR_MEAN] = held(sum.mean_late);
    // (an average worth a line: the stream steady, corrected and noisy)
    g[FR_MEAN_ON] = on(s.event == Event::None && s.c.corrected && s.noise > 0.0 && s.steps >= 40);
    g[FR_STEADY] = s.steady_at(1) as f32;
    g[FR_STEADY_AFTER] = s.steady_at(s.event_step()) as f32;
    g
}

/// A number for the glide: what f32 can hold, and 0 for what is not a number.
fn held(v: f64) -> f32 {
    if v.is_finite() {
        v.max(-1e30).min(1e30) as f32
    } else {
        0.0
    }
}

fn stream_goal(st: &Stream) -> Vec<f32> {
    let mut out = vec![0.0f32; S_LEN * MOST_STEPS];
    for (k, list) in [&st.g, &st.m_hat, &st.size, &st.u, &st.other, &st.sgd, &st.adagrad, &st.rmsprop].into_iter().enumerate() {
        for (i, v) in list.iter().enumerate().take(MOST_STEPS) {
            out[k * MOST_STEPS + i] = held(*v);
        }
    }
    out
}

fn walks_goal(walks: &[Walk; 4]) -> Vec<f32> {
    let n = PAIR_STEPS + 1;
    let mut out = vec![0.0f32; 4 * 2 * n];
    for (k, walk) in walks.iter().enumerate() {
        for (i, w) in walk.at.iter().enumerate().take(n) {
            out[(k * n + i) * 2] = held(w[0]);
            out[(k * n + i) * 2 + 1] = held(w[1]);
        }
    }
    out
}

// ----- colours and kinds of line -------------------------------------------------------

/// The gradients' dots: noise, drawn softly so that the averages over them stand out.
fn dots_colour() -> Color {
    NOISE.mix(pal::DIM, 0.35).a(0.75)
}

/// The handle that turns the right-hand valley.
const HANDLE: Color = pal::LINK;

/// Adam's divisor, √v̂ + ε: a lighter Adam.
fn size_colour() -> Color {
    ADAM.mix(pal::WHITE, 0.55)
}

// Lines whose number of points changes as the charts' range glides, each in a batch of
// its own (see `plot::apart`).
const STEADY: u8 = OWN;
const MHAT: u8 = OWN + 1;
const SIZE_UP: u8 = OWN + 2;
const SIZE_DOWN: u8 = OWN + 3;
const EPS_LINE: u8 = OWN + 4;
const MEMORY: u8 = OWN + 5;
const STEP_LINE: u8 = OWN + 6;
const OTHER_LINE: u8 = OWN + 7;
const SGD_LINE: u8 = OWN + 8;
const ADAGRAD_LINE: u8 = OWN + 9;
const RMS_LINE: u8 = OWN + 10;
const EVENT_LINE: u8 = OWN + 11;
/// The level lines of the two valleys (`CONTOURS` and `CONTOURS + 1`).
const CONTOURS: u8 = OWN + 12;
/// The four walks (`WALKS` to `WALKS + 3`).
const WALKS: u8 = OWN + 14;
const HALF_LINE: u8 = OWN + 18;
const ARROWS: u8 = OWN + 19;
const MEAN_LINE: u8 = OWN + 20;

/// Every kind of line and point this scene draws, asked for in the same order at the
/// start of every frame, each with one unseen point: a batch not drawn in a frame
/// would drop out and move every later one up a place, and `eidos motion` would then
/// take one batch for another.
fn hold_batches(f: &mut Frame) {
    let corner = f.hud(Vec2::ZERO);
    let solid = [AXES, CURVES, MARKS, BUTTONS, MHAT, SIZE_UP, SIZE_DOWN, MEMORY, STEP_LINE, SGD_LINE, ADAGRAD_LINE, RMS_LINE, CONTOURS, CONTOURS + 1, WALKS, WALKS + 1, WALKS + 2, WALKS + 3, ARROWS];
    for kind in solid {
        plot::apart(f, kind, false).segment(corner, corner, 1.0, Color::CLEAR);
    }
    for kind in [MARKS, STEADY, EPS_LINE, OTHER_LINE, EVENT_LINE, HALF_LINE, MEAN_LINE] {
        plot::apart(f, kind, true).segment(corner, corner, 1.0, Color::CLEAR);
    }
    plot::disc(f, Vec2::ZERO, 1.0, Color::CLEAR);
}

/// A number to two significant figures, for "about so much" (in a formula).
fn rough(v: f64) -> String {
    if !v.is_finite() || v == 0.0 {
        return sci(v);
    }
    let e = v.abs().log10().floor();
    let unit = 10f64.powf(e - 1.0);
    sci((v / unit).round() * unit)
}

/// A number in a formula: plain where that is short, else as a power of ten.
fn sci(v: f64) -> String {
    if !v.is_finite() {
        return "—".to_string();
    }
    let a = v.abs();
    if a == 0.0 || (0.01..10_000.0).contains(&a) {
        let decimals = if a >= 100.0 {
            0
        } else if a >= 10.0 {
            1
        } else if a >= 1.0 {
            2
        } else {
            3
        };
        return num(v, decimals);
    }
    let e = a.log10().floor();
    let lead = v / 10f64.powf(e);
    if (lead.abs() - 1.0).abs() < 5e-3 {
        format!("{}10^{{{}}}", if v < 0.0 { "-" } else { "" }, num(e, 0))
    } else {
        // (with the noughts at the end left off: 5 × 10⁻⁹, 2.5 × 10⁻⁹)
        let mut lead = num(lead, 2);
        if lead.contains('.') {
            lead = lead.trim_end_matches('0').trim_end_matches('.').to_string();
        }
        format!(r"{lead} \times 10^{{{}}}", num(e, 0))
    }
}

/// A line in a chart drawn through `points` (in the data's units), each kept inside the
/// chart (a value beyond it runs along its edge), so that it has as many points in
/// every frame whatever it shows: `eidos motion` follows a batch's points by their
/// places in it.
fn kept_line(f: &mut Frame, p: &Plot, kind: u8, points: &[(f64, f64)], width: f32, color: Color, dashed: bool) {
    let (min, max) = p.rect;
    let pts: Vec<Vec3> = points
        .iter()
        .map(|&(x, y)| {
            let q = p.at(x, y);
            let q = if q.is_finite() { q } else { min };
            f.hud(q.max(min).min(max))
        })
        .collect();
    if pts.len() >= 2 {
        plot::apart(f, kind, dashed).polyline(pts, width, color);
    }
}

/// A line of the stream (a point for every step there can be) drawn as far as step
/// `last`: the points beyond it wait unseen at the chart's right-hand edge, each at its
/// own height, so that every point stays the same step from frame to frame, as the
/// range glides, and the line keeps its number of points (see `kept_line`). The last
/// seen segment fades out on its way to the first unseen point, which stands where
/// the line leaves the chart. A point beyond the chart's other edges is unseen too, at
/// the edge, so that a line that leaves the chart there fades out on its way off it.
/// (With `last` infinite, it is any line that may leave its chart: a walk in a square.)
#[allow(clippy::too_many_arguments)]
fn kept_series(f: &mut Frame, p: &Plot, kind: u8, points: &[(f64, f64)], last: f64, width: f32, color: Color, dashed: bool) {
    let (min, max) = p.rect;
    let (lo, hi) = (p.y[0], p.y[1]);
    let pts: Vec<(Vec3, Color)> = points
        .iter()
        .map(|&(x, y)| {
            let q = p.at(x.min(last), y);
            let q = if q.is_finite() { q } else { min };
            let seen = x <= last && x >= p.x[0] && x <= p.x[1] && y >= lo && y <= hi;
            (f.hud(q.max(min).min(max)), if seen { color } else { Color::CLEAR })
        })
        .collect();
    let batch = plot::apart(f, kind, dashed);
    batch.brk();
    for (q, c) in pts {
        batch.push(q, width, c);
    }
    batch.brk();
}

/// A dot of the stream, in a batch of its own (the same number of them in every frame).
fn stream_dot(f: &mut Frame, at: Vec2, size: f32, color: Color) {
    if at.is_finite() {
        let p = f.hud(at);
        f.dl.points_with(PointKind::Overlay, PointStyle { outline_px: 0.0, intensity: 0.999, ..Default::default() }).push(p, size, color);
    }
}

impl Scene for Apart {
    fn view(&self) -> View {
        View::Flat { center: Vec2::ZERO, half_height: 1.0 }
    }

    fn settings(&self, s: &mut ViewSettings) {
        s.grid = false;
        s.bloom = s.bloom.min(0.2);
    }

    fn animated(&self) -> bool {
        self.look.moving || self.turning
    }

    fn pointer_hint(&self) -> Option<&'static str> {
        Some("buttons over the picture: click    blue dot on the right-hand valley: drag to turn it")
    }

    fn ui(&mut self, ui: &mut egui::Ui, _: &Gfx) {
        ui::note(
            ui,
            r"Adam (Kingma and Ba, 2015) keeps, for each weight, two running averages of that weight's gradients $g_t$: $m$ of the gradients and $v$ of their squares. It corrects both for having started at zero, and steps by the one divided by the root of the other, times the learning rate $\eta$:",
        );
        ui::equation(ui, r"m_t = \beta_1 m_{t-1} + (1-\beta_1)\, g_t, \quad v_t = \beta_2 v_{t-1} + (1-\beta_2)\, g_t^2, \\ \hat m_t = \frac{m_t}{1 - \beta_1^t}, \quad \hat v_t = \frac{v_t}{1 - \beta_2^t}, \quad \Delta w_t = -\eta\, \frac{\hat m_t}{\sqrt{\hat v_t} + \varepsilon}");
        ui::note(ui, r"Above, one weight is fed a stream of gradients, a steady part plus noise, with each piece drawn at work on it; below, two weights seen from above.");

        ui::section(ui, "Things to try");
        for (k, trial) in TRIALS.iter().enumerate() {
            if ui::experiment(ui, trial.button, trial.caption) {
                self.set_up(k);
            }
        }

        let s = &mut self.setting;
        ui::section(ui, "The gradients");
        let mut mean = s.mean as f32;
        if ui::slider(ui, r"steady part $\bar g$", &mut mean, -3.0..=3.0) {
            s.mean = mean as f64;
        }
        let mut noise = s.noise as f32;
        if ui::slider(ui, r"noise $\sigma$", &mut noise, 0.0..=5.0) {
            s.noise = noise as f64;
        }
        let mut scale = s.scale as f32;
        if ui::slider_log(ui, "scale", &mut scale, 1e-12..=1e4) {
            s.scale = scale as f64;
        }
        ui::note(ui, r"Every gradient is multiplied by the scale, as by a loss that many times larger. Adam's step does not change, until the gradients come near $\varepsilon$.");
        ui::slider_int_log(ui, "steps shown", &mut s.steps, LEAST_STEPS..=MOST_STEPS);
        let mut event = EVENTS.iter().position(|e| *e == s.event).unwrap_or(0);
        if ui::choice(ui, &["no change", "a jump", "a change of sign", "a spike"], &mut event) {
            s.event = EVENTS[event.min(3)];
        }
        ui::note(ui, "A quarter of the way along: the steady part ten times larger from then on, or the other way round, or one gradient 100 times the usual size.");
        if ui.button("new noise").clicked() {
            s.seed = s.seed.wrapping_add(1);
        }

        ui::section(ui, "Adam");
        let memory = |beta: f64| 1.0 / (1.0 - beta).max(1e-6);
        let mut m1 = memory(s.c.beta1) as f32;
        if ui::slider_log(ui, r"$\beta_1$'s memory $1/(1-\beta_1)$", &mut m1, 1.0..=1000.0) {
            s.c.beta1 = 1.0 - 1.0 / (m1 as f64).max(1.0);
        }
        let mut m2 = memory(s.c.beta2) as f32;
        if ui::slider_log(ui, r"$\beta_2$'s memory $1/(1-\beta_2)$", &mut m2, 1.0..=10_000.0) {
            s.c.beta2 = 1.0 - 1.0 / (m2 as f64).max(1.0);
        }
        ui::note(ui, r"A running average with coefficient $\beta$ remembers about the last $1/(1-\beta)$ values. The defaults, $\beta_1 = 0.9$ and $\beta_2 = 0.999$ (Kingma and Ba's, and PyTorch's), remember about 10 and 1000 steps.");
        let mut eps = s.c.eps as f32;
        if ui::slider_log(ui, r"$\varepsilon$", &mut eps, 1e-12..=1.0) {
            s.c.eps = eps as f64;
        }
        ui::checkbox(ui, &mut s.c.corrected, "corrections for the start at zero");
        ui::note(ui, r"How large can a step be? Kingma and Ba (2015) call it approximately bounded by $\eta$: in the common case $|\hat m/\sqrt{\hat v}|$ is about 1 at most. Their bound for the sparsest case, a gradient after a long run of zeros, is $(1-\beta_1)/\sqrt{1-\beta_2}$ times $\eta$: 3.16 with the defaults. A gradient that grows can pass it: 'a jump' with 3000 steps shown, steady part 1 and noise 0.1 gives 4.20, and the most any gradients can give, long after the start, is $(1-\beta_1)/\sqrt{(1-\beta_2)(1-\beta_1^2/\beta_2)} = 7.27$.");
        ui::note(ui, r"Some code, and Kingma and Ba's own second form, fold the corrections into the step size and add $\varepsilon$ to $\sqrt{v}$ before correcting: the same steps, except while $\sqrt{\hat v}$ is not far above $\varepsilon$ early on.");

        ui::section(ui, "Beside Adam's step");
        ui::checkbox(ui, &mut s.sgd, r"gradient descent: $\eta g$");
        ui::checkbox(ui, &mut s.adagrad, "AdaGrad");
        ui::checkbox(ui, &mut s.rmsprop, "RMSProp");
        ui::note(ui, r"As PyTorch writes them, each in units of its own learning rate: AdaGrad divides by the root of the sum of all the squares so far; RMSProp by the root of their running average with $\alpha = 0.99$, uncorrected and without momentum.");

        ui::section(ui, "Two weights");
        let mut pair = (s.pair == Pair::Decay) as usize;
        if ui::choice(ui, &["a valley", "weight decay"], &mut pair) {
            s.pair = if pair == 1 { Pair::Decay } else { Pair::Valley };
        }
        let mut angle = s.angle as f32;
        if ui::slider(ui, "the right valley's turn (degrees)", &mut angle, 0.0..=180.0) {
            s.angle = angle as f64;
        }
        let mut eta = s.eta as f32;
        if ui::slider_log(ui, r"Adam's learning rate $\eta$", &mut eta, 0.001..=0.5) {
            s.eta = eta as f64;
        }
        let mut lambda = s.lambda as f32;
        if ui::slider_log_or_zero(ui, r"weight decay $\lambda$", &mut lambda, 0.001, 1.0) {
            s.lambda = lambda as f64;
        }
        ui::note(ui, r"AdamW here is PyTorch's: each step takes $\eta\lambda w$ off the weight. In Loshchilov and Hutter's paper (2019), where AdamW is defined, the decay is multiplied by the schedule's factor but not by $\eta$, so the same $\lambda$ means a different decay there.");
        self.refresh();

        ui::section(ui, "Reading the picture");
        let corrected = self.setting.c.corrected;
        ui::key(
            ui,
            &[
                (dots_colour(), r"red dots: the gradient $g_t$ at each step $t$, its steady part plus noise"),
                (SIGNAL, "white dashes: the steady part alone"),
                (ADAM, if corrected { r"orange: $\hat m_t$, the running average of the gradients; below, Adam's step $\hat m_t/(\sqrt{\hat v_t} + \varepsilon)$ in units of $\eta$" } else { r"orange: $m_t$, the running average of the gradients; below, Adam's step $m_t/(\sqrt{v_t} + \varepsilon)$ in units of $\eta$" }),
                (size_colour(), if corrected { r"light orange: $\pm(\sqrt{\hat v_t} + \varepsilon)$, the running size of the gradients that Adam divides by" } else { r"light orange: $\pm(\sqrt{v_t} + \varepsilon)$, the running size of the gradients that Adam divides by" }),
                (pal::DIM, r"grey dashes: a step of $\pm\eta$ below; on the upper chart, $\varepsilon$; bars over it, how many steps each average remembers"),
                (SGD, r"cyan: gradient descent, its step $\eta g_t$ and, in the squares, its path"),
                (ADAGRAD, "green: AdaGrad's step, in units of its learning rate"),
                (RMSPROP, "purple: RMSProp's step, in units of its learning rate"),
                (pal::WHITE, "white ring: where the two weights start"),
                (HANDLE, "blue dot on the right-hand valley: drag it to turn the valley"),
            ],
        );
        ui::section(ui, "Words");
        ui::term(ui, "Gradient:", "the slope of the loss along a weight; a step of training moves the weight against it.");
        ui::term(ui, "Learning rate:", r"$\eta$, how large the steps are. Adam's steps are mostly about $\eta$ at most, whatever the gradient's size (the note under Adam says when they are larger).");
        ui::term(ui, "Running average:", r"$m_t = \beta m_{t-1} + (1-\beta) g_t$: each new value counts $1 - \beta$, and older ones fade by a factor $\beta$ a step.");
        ui::term(ui, "Corrections for the start at zero:", r"$m$ and $v$ start at 0, which leaves them too small early on by the factor $1 - \beta^t$; dividing by it gives $\hat m$ and $\hat v$, exactly right for a steady gradient.");
        ui::term(ui, r"$\varepsilon$:", r"a small number added to the divisor so that it is never 0; $10^{-8}$ by default.");
        ui::term(ui, "Sign descent:", "moving every weight by the same amount in the direction of the sign of its gradient, whatever the gradient's size (scene steepest).");
        ui::term(ui, "Signal-to-noise ratio:", r"Kingma and Ba's name, 'with a slight abuse of terminology', for $\hat m/\sqrt{\hat v}$. It averages about $\bar g/\sqrt{\bar g^2 + \sigma^2}$, which lies between $-1$ and 1, and not $\bar g/\sigma$.");
        ui::term(ui, "Weight decay:", r"shrinking every weight towards 0 a little at each step, by $\lambda$ times the weight.");
        ui::term(ui, "L2 penalty:", r"$\tfrac12 \lambda w^2$ added to the loss, which adds $\lambda w$ to the gradient.");
        ui::term(ui, "AdaGrad, RMSProp:", "two rules that divide by the size of past gradients as Adam does, without a running average of the gradients themselves (as PyTorch writes them, without momentum).");
        ui::term(
            ui,
            "Balles and Hennig's reading:",
            r"(2018) the step, $\varepsilon$ left out, is the sign of $\hat m$ times $1/\sqrt{1 + (\hat v - \hat m^2)/\hat m^2}$, where $(\hat v - \hat m^2)/\hat m^2$ estimates the gradient's variance next to its squared mean. The factor shrinks as that grows: the brake in 'mostly noise'.",
        );

        ui::section(ui, "Numbers");
        let s = self.setting;
        let sum = self.summary();
        let word = if s.c.corrected { "" } else { " (no corrections)" };
        ui::readout(ui, &format!("first step{word}"), format!(r"${}\,\eta$", num(sum.first, 3)));
        ui::readout(ui, "largest step", format!(r"${}\,\eta$ at step {}", num(sum.largest.0, 3), sum.largest.1));
        ui::readout(ui, &format!("mean step, steps {} to {}", sum.from, s.steps), format!(r"${}\,\eta$", num(sum.mean_late, 3)));
        ui::readout(ui, "  spread about it", format!(r"${}\,\eta$", num(sum.sd_late, 3)));
        if s.event == Event::None {
            ui::readout(ui, r"$\bar g/\sqrt{\bar g^2 + \sigma^2}$", num(settled_ratio(s.mean, s.noise), 4));
        } else {
            ui::readout(ui, "steps under $\\eta/2$ after the change", count(sum.small_after));
        }
        let last = s.steps.min(self.stream.u.len()).max(1) - 1;
        if let (Some(m), Some(size), Some(g)) = (self.stream.m_hat.get(last), self.stream.size.get(last), self.stream.sgd.get(last)) {
            ui::readout(ui, &format!(r"at step {}: $\hat m$", last + 1), format!("${}$", sci(m * s.scale)));
            ui::readout(ui, r"  $\sqrt{\hat v} + \varepsilon$", format!("${}$", sci(size * s.scale)));
            ui::readout(ui, r"  gradient descent's step $\eta g$", format!(r"${}\,\eta$", sci(*g)));
        }
        match s.pair {
            Pair::Valley => {
                let steps = |w: &Walk| w.steps_to(ARRIVED).map_or("not in 2000 steps".to_string(), count);
                ui::readout(ui, "to a ten-thousandth, lined up: GD", steps(&self.walks[0]));
                ui::readout(ui, "  Adam", steps(&self.walks[1]));
                ui::readout(ui, &format!("turned by {}°: GD", num(s.angle, 0)), steps(&self.walks[2]));
                ui::readout(ui, "  Adam", steps(&self.walks[3]));
            }
            Pair::Decay => {
                let half = |w: &Walk, i: usize| w.half_life(i).map_or_else(|| format!("not in 2000 steps ({} left)", num(w.at.last().map_or(1.0, |p| p[i]), 3)), |n| format!("{} steps", count(n)));
                ui::readout(ui, r"L2: $w_1$ halves in", half(&self.walks[1], 0));
                ui::readout(ui, r"  $w_2$", half(&self.walks[1], 1));
                ui::readout(ui, r"AdamW: $w_1$ halves in", half(&self.walks[3], 0));
                ui::readout(ui, r"  $w_2$", half(&self.walks[3], 1));
            }
        }
        ui::see_also(
            ui,
            &[
                ("steepest", "sign descent and other rulers"),
                ("signnoise", "sign descent and Adam with mini-batch noise"),
                ("wdecay", "what weight decay does in a network"),
                ("descent", "seven optimisers on surfaces of two variables"),
            ],
        );
    }

    fn frame(&mut self, f: &mut Frame) {
        hold_batches(f);
        let lay = Layout::of(f.view_size(), f.top_inset);
        // (a canvas squeezed to a sliver is one the controls have opened over, on a phone)
        let tiny = lay.view.x < 160.0 || lay.view.y < 160.0;
        let was_turning = self.turning;
        self.turning = false;
        if !tiny && !lay.compact {
            self.buttons(f, &lay);
            self.turn(f, &lay);
        }
        self.refresh();
        let snap = self.canvas != Some(lay.view);
        if snap && !tiny {
            self.canvas = Some(lay.view);
        }
        self.follow(f.real_dt, snap, was_turning || self.turning);
        if tiny {
            return;
        }
        self.draw_stream(f, &lay);
        self.draw_steps(f, &lay);
        if !lay.compact {
            self.draw_squares(f, &lay);
        }
    }
}

impl Apart {
    /// The buttons over the picture.
    fn buttons(&mut self, f: &mut Frame, lay: &Layout) {
        let size = lay.small;
        let s = &mut self.setting;
        for (k, e) in EVENTS.into_iter().enumerate() {
            if plot::button(f, lay.events[k], e.button(lay.narrow), size, s.event == e) {
                s.event = e;
            }
        }
        let names = ["gradient descent", "AdaGrad", "RMSProp"];
        let on = [&mut s.sgd, &mut s.adagrad, &mut s.rmsprop];
        for (k, flag) in on.into_iter().enumerate() {
            if plot::button(f, lay.beside[k], names[k], size, *flag) {
                *flag = !*flag;
            }
        }
        for (k, (name, pair)) in [("a valley", Pair::Valley), ("weight decay", Pair::Decay)].into_iter().enumerate() {
            if plot::button(f, lay.pairs[k], name, size, s.pair == pair) {
                s.pair = pair;
            }
        }
    }

    /// The plot of square `k`, with the range it is drawn with now.
    fn square_plot(&self, lay: &Layout, k: usize) -> Option<Plot> {
        let g = &self.look.frame;
        if g.shown.len() < FR_LEN {
            return None;
        }
        let (lo, hi) = (g.get(FR_LO) as f64, g.get(FR_HI) as f64);
        let p = Plot::new(lay.squares[k.min(1)], [lo, hi], [lo, hi]);
        (!p.tiny()).then_some(p)
    }

    /// Where the handle that turns the right-hand valley is, in its units.
    const HANDLE_R: f64 = 3.25;

    /// The right-hand valley, turned by dragging the blue dot at the end of its floor.
    fn turn(&mut self, f: &mut Frame, lay: &Layout) {
        if self.setting.pair != Pair::Valley {
            return;
        }
        let Some(p) = self.square_plot(lay, 1) else { return };
        let angle = self.look.frame.get(FR_ANGLE) as f64;
        let (along, _) = valley_axes(angle);
        // the valley's floor, out to the handle
        kept_line(f, &p, EVENT_LINE, &[(0.0, 0.0), (Self::HANDLE_R * along[0], Self::HANDLE_R * along[1])], 1.0, HANDLE.a(0.5), true);
        if let Some((x, y)) = p.handle(f, 7, Self::HANDLE_R * along[0], Self::HANDLE_R * along[1], HANDLE) {
            if x.is_finite() && y.is_finite() && (x * x + y * y) > 0.04 {
                let mut a = y.atan2(x).to_degrees();
                // a valley turned by 180° is the same valley
                if a < 0.0 {
                    a += 180.0;
                }
                self.setting.angle = a.max(0.0).min(180.0);
                self.turning = true;
            }
        }
    }

    /// Bring the picture `dt` seconds nearer to what it shows. (`snap`: there at once.)
    fn follow(&mut self, dt: f32, snap: bool, held: bool) {
        let s = self.setting;
        let mut frame = self.goal.frame.clone();
        let (stream, walks) = (&self.goal.stream, &self.goal.walks);
        let look = &mut self.look;
        if snap {
            look.frame.snap(&frame);
            look.stream.snap(stream);
            look.walks.snap(walks);
            look.moving = false;
            return;
        }
        // (the turn answers the hand at once, and the walks with it; the rest glides on
        // from where it is)
        if held {
            let mut at = look.frame.shown.clone();
            if at.len() == FR_LEN && frame.len() == FR_LEN {
                at[FR_ANGLE] = s.angle as f32;
                frame[FR_ANGLE] = s.angle as f32;
                look.frame.snap(&at);
            }
            let mut moving = look.frame.follow(&frame, dt, false);
            look.walks.snap(walks);
            moving |= look.stream.follow(stream, dt, false);
            look.moving = moving;
            return;
        }
        let mut moving = look.frame.follow(&frame, dt, false);
        moving |= look.stream.follow(stream, dt, false);
        moving |= look.walks.follow(walks, dt, false);
        look.moving = moving;
    }

    /// A shown value of the stream: array `k`, step `t` (from 1).
    fn shown(&self, k: usize, t: usize) -> f64 {
        self.look.stream.get(k * MOST_STEPS + t.clamp(1, MOST_STEPS) - 1) as f64
    }

    /// Array `k` of the stream, a point for every step there can be (`sign` turns it
    /// upside down: the lower edge of Adam's divisor).
    fn series(&self, k: usize, sign: f64) -> Vec<(f64, f64)> {
        (1..=MOST_STEPS).map(|t| (t as f64, sign * self.shown(k, t))).collect()
    }

    /// The stream's upper chart: the gradients, the steady part, m̂ and ±(√v̂ + ε), and ε,
    /// with a strip along its top for how long each average remembers.
    fn draw_stream(&self, f: &mut Frame, lay: &Layout) {
        let g = &self.look.frame;
        if g.shown.len() < FR_LEN || self.look.stream.shown.len() < S_LEN * MOST_STEPS || self.goal.frame.len() < FR_LEN {
            return;
        }
        let s = &self.setting;
        let n_shown = (g.get(FR_STEPS) as f64).max(1.0);
        let y1 = (g.get(FR_Y1) as f64).max(1e-30);
        let whole = Plot::new(lay.stream, [0.0, n_shown], [-y1, y1]);
        let strip = 2.0 * (lay.small + 5.0) + 6.0;
        let rect = (lay.stream.0 + Vec2::new(0.0, strip), lay.stream.1);
        let p = Plot::new(rect, [0.0, n_shown], [-y1, y1]);
        if whole.tiny() || p.tiny() {
            return;
        }
        whole.backing(f, if lay.narrow { 34.0 } else { 46.0 }, if lay.narrow { 22.0 } else { 26.0 }, 28.0, if lay.narrow { 6.0 } else { 96.0 });
        let corrected = s.c.corrected;
        let title = match (lay.narrow, corrected) {
            (true, true) => r"the gradients, $\hat m$ and $\pm(\sqrt{\hat v} + \varepsilon)$".to_string(),
            (true, false) => r"the gradients, $m$ and $\pm(\sqrt{v} + \varepsilon)$".to_string(),
            (false, true) => r"one weight's gradients $g_t$, their running average $\hat m_t$, and their running size $\pm(\sqrt{\hat v_t} + \varepsilon)$".to_string(),
            (false, false) => r"one weight's gradients $g_t$, their running average $m_t$, and their running size $\pm(\sqrt{v_t} + \varepsilon)$, uncorrected".to_string(),
        };
        f.text_px(Vec2::new(whole.rect.0.x - 30.0, whole.rect.0.y - 8.0), title, lay.text, pal::TEXT, Align2::LEFT_BOTTOM);
        let goal = self.goal.frame[FR_Y1] as f64;
        Self::value_axis(f, &p, [-goal, goal], s.scale, lay.small);
        let n = (n_shown.floor() as usize).clamp(1, MOST_STEPS);
        Self::step_axis(f, &p, s.steps as f64, lay);
        let (x0, x1) = (0.0, n_shown);
        kept_line(f, &p, AXES, &[(x0, 0.0), (x1, 0.0)], 1.0, pal::DIM.a(0.35), false);

        // the sudden change, a mark up the chart
        let on = g.get(FR_EVENT_ON).clamp(0.0, 1.0);
        let t0 = g.get(FR_EVENT) as f64;
        kept_line(f, &p, EVENT_LINE, &[(t0, -y1), (t0, y1)], 1.0, pal::DIM.a(0.7 * on), true);

        // the gradients as dots, every step in every frame (those not shown unseen at the
        // edge), and the spike as an arrow at the edge if it is beyond it
        let dot = if n as f32 > p.width() * 0.6 {
            2.2
        } else if n as f32 > p.width() * 0.2 {
            3.0
        } else {
            4.0
        };
        let colour = dots_colour();
        let spike = (s.event == Event::Spike).then(|| s.event_step());
        let mut off: Option<(f64, f64)> = None;
        let (min, max) = p.rect;
        for t in 1..=MOST_STEPS {
            let v = self.shown(S_G, t);
            let seen = t <= n && v.abs() <= y1;
            if t <= n && v.abs() > y1 && Some(t) == spike {
                off = Some((t as f64, v));
            }
            let at = p.at(t as f64, v).max(min).min(max);
            stream_dot(f, at, dot, if seen { colour } else { Color::CLEAR });
        }
        // (the arrow stands at the sudden change in every frame, seen or not, so that it
        // never jumps there from somewhere else)
        let up = off.is_none_or(|(_, v)| v > 0.0);
        let x = p.px_x(t0).max(min.x).min(max.x);
        let (y, d) = if up { (min.y + 2.0, -1.0) } else { (max.y - 2.0, 1.0) };
        let arrow = [(Vec2::new(x, y - 16.0 * d), Vec2::new(x, y)), (Vec2::new(x - 5.0, y - 6.0 * d), Vec2::new(x, y)), (Vec2::new(x + 5.0, y - 6.0 * d), Vec2::new(x, y))];
        for (a, b) in arrow {
            plot::line_of(f, ARROWS, a, b, 1.6, colour.a(if off.is_some() { 1.0 } else { 0.0 }));
        }
        if let Some((_, v)) = off {
            let text = format!("a gradient of ${}$", sci(v * s.scale));
            f.text_px(Vec2::new(x - 8.0, y - 9.0 * d), text, lay.small, colour.a(1.0), if up { Align2::RIGHT_TOP } else { Align2::RIGHT_BOTTOM });
        }

        // the steady part (before and after a sudden change), ε, and Adam's two averages
        let (before, after) = (g.get(FR_STEADY) as f64, g.get(FR_STEADY_AFTER) as f64);
        kept_line(f, &p, STEADY, &[(1.0, before), (t0, before), (t0, after), (n_shown.max(t0), after)], 1.4, SIGNAL.a(0.8), true);
        let eps = g.get(FR_EPS) as f64;
        let eps_seen = eps > 0.03 * y1 && eps < y1;
        kept_line(f, &p, EPS_LINE, &[(x0, eps), (x1, eps)], 1.0, pal::DIM.a(if eps_seen { 0.9 } else { 0.0 }), true);
        kept_series(f, &p, SIZE_UP, &self.series(S_SIZE, 1.0), n_shown, 1.6, size_colour(), false);
        kept_series(f, &p, SIZE_DOWN, &self.series(S_SIZE, -1.0), n_shown, 1.6, size_colour(), false);
        kept_series(f, &p, MHAT, &self.series(S_M, 1.0), n_shown, 2.4, ADAM, false);

        // what each average remembers, as bars in the chart's own steps
        self.memory_bars(f, &whole, lay);

        // the names of the lines, at their right-hand ends
        let at = |v: f64| p.px_y(v.max(-y1).min(y1));
        let m_name = if corrected { r"$\hat m$" } else { "$m$" };
        let size_name = if corrected { r"$\sqrt{\hat v} + \varepsilon$" } else { r"$\sqrt{v} + \varepsilon$" };
        let minus_name = if corrected { r"$-(\sqrt{\hat v} + \varepsilon)$" } else { r"$-(\sqrt{v} + \varepsilon)$" };
        let size = self.shown(S_SIZE, n);
        let mut names = vec![(at(self.shown(S_M, n)), m_name.to_string(), ADAM), (at(size), size_name.to_string(), size_colour()), (at(-size), minus_name.to_string(), size_colour())];
        if eps_seen {
            names.push((at(eps), r"$\varepsilon$".to_string(), pal::DIM));
        }
        Self::end_names(f, &p, names, lay);
    }

    /// The upper chart's axis of values: round numbers in the gradients' own units (the
    /// drawn ones times the scale), written as powers of ten where they are very small
    /// or large, with faint lines across at them. The round numbers are those of the
    /// range the chart is gliding to (`goal`, drawn units), so that they keep their
    /// places on the way; one beyond the chart is not drawn.
    fn value_axis(f: &mut Frame, p: &Plot, goal: [f64; 2], scale: f64, size: f32) {
        let (min, max) = p.rect;
        plot::line_of(f, AXES, min, Vec2::new(min.x, max.y), 1.0, pal::DIM.a(0.8));
        // (about one every 24 pixels, so that a short chart still has a few)
        let ticks = plot::Ticks::of(goal[0] * scale, goal[1] * scale, p.height() * 2.5);
        let plain = ticks.step >= 1e-3 && ticks.step < 1e5;
        for v in ticks.values() {
            let y = p.px_y(v / scale);
            let inside = y >= min.y - 0.5 && y <= max.y + 0.5;
            let y = y.max(min.y).min(max.y);
            let a = if inside { 1.0 } else { 0.0 };
            plot::line_of(f, AXES, Vec2::new(min.x - 4.0, y), Vec2::new(min.x, y), 1.0, pal::DIM.a(0.8 * a));
            plot::line_of(f, AXES, Vec2::new(min.x, y), Vec2::new(max.x, y), 1.0, pal::DIM.a(0.14 * a));
            if inside {
                let name = if plain {
                    ticks.name(v)
                } else if v == 0.0 {
                    "0".to_string()
                } else {
                    format!("${}$", sci(v))
                };
                f.text_px(Vec2::new(min.x - 6.0, y), name, size, pal::DIM, Align2::RIGHT_CENTER);
            }
        }
    }

    /// The x axis of a chart of steps: its round values chosen from where its range is
    /// going, so that they keep their places while it glides there.
    fn step_axis(f: &mut Frame, p: &Plot, goal: f64, lay: &Layout) {
        let (min, max) = p.rect;
        plot::line_of(f, AXES, Vec2::new(min.x, max.y), max, 1.0, pal::DIM.a(0.8));
        let ticks = plot::Ticks::of(0.0, goal, p.width());
        for v in ticks.values() {
            let inside = v >= p.x[0] - 1e-9 && v <= p.x[1] + 1e-9;
            let x = p.px_x(v).max(min.x).min(max.x);
            plot::line_of(f, AXES, Vec2::new(x, max.y), Vec2::new(x, max.y + 4.0), 1.0, pal::DIM.a(if inside { 0.8 } else { 0.0 }));
            if inside {
                f.text_px(Vec2::new(x, max.y + 6.0), ticks.name(v), lay.small, pal::DIM, Align2::CENTER_TOP);
            }
        }
        if !lay.narrow {
            f.text_px(Vec2::new(max.x + 8.0, max.y + 6.0), "step $t$", lay.small, pal::DIM, Align2::LEFT_TOP);
        }
    }

    /// Names at the right-hand ends of lines (`(y, name, colour)`), kept apart.
    fn end_names(f: &mut Frame, p: &Plot, mut names: Vec<(f32, String, Color)>, lay: &Layout) {
        if lay.narrow {
            return;
        }
        let gap = lay.small + 4.0;
        names.sort_by(|a, b| a.0.total_cmp(&b.0));
        // push down where two would overlap, then up again where that ran off the chart
        for i in 1..names.len() {
            if names[i].0 < names[i - 1].0 + gap {
                names[i].0 = names[i - 1].0 + gap;
            }
        }
        let bottom = p.rect.1.y + 4.0;
        if let Some(last) = names.last().map(|n| n.0) {
            if last > bottom {
                let up = last - bottom;
                for n in names.iter_mut() {
                    n.0 -= up;
                }
            }
        }
        for (y, name, colour) in names {
            f.text_px(Vec2::new(p.rect.1.x + 8.0, y), name, lay.small, colour, Align2::LEFT_CENTER);
        }
    }

    /// Bars at the top of the upper chart as long as the memory of each average, in the
    /// chart's own steps: a name, then the bar (with an arrow where it is longer than
    /// the chart is wide).
    fn memory_bars(&self, f: &mut Frame, p: &Plot, lay: &Layout) {
        let c = &self.setting.c;
        let (min, max) = p.rect;
        let per_step = p.width() / (p.x[1] - p.x[0]).max(1e-9) as f32;
        let corrected = c.corrected;
        let rows = [(c.beta1, if corrected { r"$\hat m$" } else { "$m$" }, ADAM), (c.beta2, if corrected { r"$\hat v$" } else { "$v$" }, size_colour())];
        for (k, (beta, name, colour)) in rows.into_iter().enumerate() {
            let memory = 1.0 / (1.0 - beta).max(1e-9);
            let y = min.y + 9.0 + k as f32 * (lay.small + 5.0);
            let steps = count(memory.round() as usize);
            let text = if lay.narrow { format!("{name}: {steps} steps") } else { format!("{name} remembers about {steps} steps") };
            let w = f.text_size(&text, lay.small).x;
            let x = min.x + 8.0;
            f.text_px(Vec2::new(x, y), text, lay.small, colour, Align2::LEFT_CENTER);
            let from = x + w + 8.0;
            let room = (max.x - 8.0 - from).max(0.0);
            let full = memory as f32 * per_step;
            let len = full.min(room).max(1.0);
            plot::line_of(f, MEMORY, Vec2::new(from, y), Vec2::new(from + len, y), 3.0, colour.a(0.9));
            if full > room {
                let end = from + len;
                plot::line_of(f, MEMORY, Vec2::new(end - 6.0, y - 4.0), Vec2::new(end, y), 2.0, colour.a(0.9));
                plot::line_of(f, MEMORY, Vec2::new(end - 6.0, y + 4.0), Vec2::new(end, y), 2.0, colour.a(0.9));
            } else {
                // (the same number of points in every frame: an arrowhead of no length)
                plot::line_of(f, MEMORY, Vec2::new(from + len, y), Vec2::new(from + len, y), 2.0, Color::CLEAR);
                plot::line_of(f, MEMORY, Vec2::new(from + len, y), Vec2::new(from + len, y), 2.0, Color::CLEAR);
            }
        }
    }

    /// The stream's lower chart: Adam's step in units of η, with what is drawn beside it.
    fn draw_steps(&self, f: &mut Frame, lay: &Layout) {
        let g = &self.look.frame;
        if g.shown.len() < FR_LEN || self.look.stream.shown.len() < S_LEN * MOST_STEPS {
            return;
        }
        let s = &self.setting;
        let n_shown = (g.get(FR_STEPS) as f64).max(1.0);
        let (lo, hi) = ((g.get(FR_Y2_LO) as f64).min(-0.5), (g.get(FR_Y2) as f64).max(0.5));
        let p = Plot::new(lay.steps, [0.0, n_shown], [lo, hi]);
        if p.tiny() || self.goal.frame.len() < FR_LEN {
            return;
        }
        p.backing(f, if lay.narrow { 34.0 } else { 46.0 }, if lay.narrow { 22.0 } else { 26.0 }, 28.0, if lay.narrow { 6.0 } else { 96.0 });
        let corrected = s.c.corrected;
        let title = match (lay.narrow, corrected) {
            (true, _) => r"the step $/\, \eta$".to_string(),
            (false, true) => r"Adam's step $\hat m_t / (\sqrt{\hat v_t} + \varepsilon)$, in units of $\eta$".to_string(),
            (false, false) => r"Adam's step $m_t / (\sqrt{v_t} + \varepsilon)$, uncorrected, in units of $\eta$".to_string(),
        };
        f.text_px(Vec2::new(p.rect.0.x - 30.0, p.rect.0.y - 8.0), title, lay.text, pal::TEXT, Align2::LEFT_BOTTOM);
        let goal = [self.goal.frame[FR_Y2_LO] as f64, self.goal.frame[FR_Y2] as f64];
        Self::value_axis(f, &p, goal, 1.0, lay.small);
        kept_line(f, &p, AXES, &[(0.0, 0.0), (n_shown, 0.0)], 1.0, pal::DIM.a(0.35), false);
        for v in [1.0, -1.0] {
            kept_line(f, &p, MARKS, &[(0.0, v), (n_shown, v)], 1.0, pal::DIM.a(0.8), true);
        }
        Self::step_axis(f, &p, s.steps as f64, lay);
        let n = (n_shown.floor() as usize).clamp(1, MOST_STEPS);
        let on = g.get(FR_EVENT_ON).clamp(0.0, 1.0);
        let t0 = g.get(FR_EVENT) as f64;
        kept_line(f, &p, EVENT_LINE, &[(t0, lo), (t0, hi)], 1.0, pal::DIM.a(0.7 * on), true);

        let a = |k: usize| g.get(k).clamp(0.0, 1.0);
        // the lines beside Adam's, each drawn while it is to be seen at all (gradient
        // descent's steps may be far beyond the chart: its line fades out there)
        for (k, alpha, kind, colour, width, dashed) in [
            (S_OTHER, 0.55 * a(FR_OTHER), OTHER_LINE, ADAM, 1.4, true),
            (S_SGD, 0.9 * a(FR_SGD), SGD_LINE, SGD, 1.4, false),
            (S_ADAGRAD, 0.95 * a(FR_ADAGRAD), ADAGRAD_LINE, ADAGRAD, 1.6, false),
            (S_RMS, 0.95 * a(FR_RMS), RMS_LINE, RMSPROP, 1.6, false),
        ] {
            if alpha > 0.0 {
                kept_series(f, &p, kind, &self.series(k, 1.0), n_shown, width, colour.a(alpha), dashed);
            }
        }
        // the average step over the window the read-outs average over
        let sum = self.summary();
        let mean_on = a(FR_MEAN_ON);
        let mean = g.get(FR_MEAN) as f64;
        kept_line(f, &p, MEAN_LINE, &[(sum.from as f64, mean), (n_shown, mean)], 1.6, pal::WHITE.a(0.85 * mean_on), true);
        kept_series(f, &p, STEP_LINE, &self.series(S_U, 1.0), n_shown, 2.0, ADAM, false);

        // the first step, and the largest where it stands out
        let first = self.shown(S_U, 1);
        p.dot(f, 1.0, first, 6.0, ADAM);
        let size = lay.small;
        let label = ADAM.mix(pal::WHITE, 0.3);
        if p.width() > 200.0 {
            let text = format!("step 1: ${}$", num(sum.first, 2));
            let y = p.px_y(first.max(lo).min(hi));
            let up = first < lo + 0.3 * (hi - lo);
            f.text_px(Vec2::new(p.px_x(1.0) + 7.0, y + if up { -8.0 } else { 8.0 }), text, size, label, if up { Align2::LEFT_BOTTOM } else { Align2::LEFT_TOP });
        }
        let (big, at) = sum.largest;
        if big.abs() > 1.05 && at > 1 && at <= n {
            let shown = self.shown(S_U, at);
            p.dot(f, at as f64, shown, 6.0, ADAM);
            let text = format!("largest: ${}$ at step {}", num(big, 2), at);
            let y = p.px_y(shown.max(lo).min(hi));
            let x = (p.px_x(at as f64) + 8.0).min(p.rect.1.x - f.text_size(&text, size).x - 4.0);
            let below = shown > 0.0;
            f.text_px(Vec2::new(x, y + if below { 6.0 } else { -6.0 }), text, size, label, if below { Align2::LEFT_TOP } else { Align2::LEFT_BOTTOM });
        }

        // the names of the lines at their right-hand ends
        let at_end = |k: usize| p.px_y(self.shown(k, n).max(lo).min(hi));
        let mut names = vec![(at_end(S_U), "Adam".to_string(), ADAM)];
        if mean_on > 0.5 {
            names.push((p.px_y(mean.max(lo).min(hi)), format!("average ${}$", num(sum.mean_late, 3)), pal::WHITE));
        }
        if a(FR_OTHER) > 0.5 {
            names.push((at_end(S_OTHER), "with the corrections".to_string(), ADAM.a(0.7)));
        }
        let gd_off = a(FR_SGD) > 0.5 && self.stream.sgd[..n.min(self.stream.sgd.len())].iter().filter(|v| **v >= lo && **v <= hi).count() * 2 < n;
        if a(FR_SGD) > 0.5 && !gd_off {
            names.push((at_end(S_SGD), "gradient descent".to_string(), SGD));
        }
        if a(FR_ADAGRAD) > 0.5 {
            names.push((at_end(S_ADAGRAD), "AdaGrad".to_string(), ADAGRAD));
        }
        if a(FR_RMS) > 0.5 {
            names.push((at_end(S_RMS), "RMSProp".to_string(), RMSPROP));
        }
        Self::end_names(f, &p, names, lay);
        // gradient descent's steps when most are off the chart: their usual size, in the chart
        if gd_off {
            let mut sizes: Vec<f64> = self.stream.sgd[..n.min(self.stream.sgd.len())].iter().map(|v| v.abs()).collect();
            sizes.sort_by(|a, b| a.total_cmp(b));
            let typical = sizes.get(sizes.len() / 2).copied().unwrap_or(0.0);
            let text = if typical > hi { format!(r"gradient descent's steps $\eta g$: about ${}\,\eta$, off the chart", rough(typical)) } else { "gradient descent's steps: off the chart".to_string() };
            f.text_px(Vec2::new(p.rect.1.x - 6.0, p.rect.0.y + 1.0), text, size, SGD, Align2::RIGHT_TOP);
        }
        if !lay.narrow {
            f.text_px(Vec2::new(lay.beside[0].0.x - 8.0, (lay.beside[0].0.y + lay.beside[0].1.y) * 0.5), "beside it:", size, pal::DIM, Align2::RIGHT_CENTER);
        }
    }

    /// The two squares: the valley lined up and turned, or the two kinds of decay.
    fn draw_squares(&self, f: &mut Frame, lay: &Layout) {
        let g = &self.look.frame;
        let n = PAIR_STEPS + 1;
        if g.shown.len() < FR_LEN || self.look.walks.shown.len() < 8 * n {
            return;
        }
        let s = &self.setting;
        let valley = g.get(FR_VALLEY).clamp(0.0, 1.0);
        let size = lay.small;
        for k in 0..2 {
            let Some(p) = self.square_plot(lay, k) else { continue };
            let (min, max) = p.rect;
            p.backing(f, if lay.narrow { 30.0 } else { 40.0 }, if lay.narrow { 34.0 } else { 40.0 }, 22.0, 8.0);
            // the axes through 0, with ticks where they help
            p.hline(f, AXES, 0.0, 1.0, pal::DIM.a(0.45), false);
            p.vline(f, AXES, 0.0, 1.0, pal::DIM.a(0.45), false);
            p.x_axis(f, "", size);
            p.y_axis(f, "", size);
            f.text_px(Vec2::new(max.x - 2.0, max.y - 4.0), "$w_1$", size, pal::DIM, Align2::RIGHT_BOTTOM);
            f.text_px(Vec2::new(min.x + 4.0, min.y + 2.0), "$w_2$", size, pal::DIM, Align2::LEFT_TOP);

            // the valley's level lines, turned as it is
            let angle = if k == 0 { 0.0 } else { g.get(FR_ANGLE) as f64 };
            let (along, across) = valley_axes(angle);
            // (the largest stays inside the square however the valley is turned)
            for level in [0.02, 0.08, 0.3, 1.0, 3.0, 6.0] {
                let (ra, rb) = ((2.0 * level / ALONG).sqrt(), (2.0 * level / ACROSS).sqrt());
                let ring = (0..=96).map(|i| {
                    let (sn, cs) = (TAU as f64 * i as f64 / 96.0).sin_cos();
                    let (a, b) = (ra * cs, rb * sn);
                    (a * along[0] + b * across[0], a * along[1] + b * across[1])
                });
                if valley > 0.0 {
                    p.path(f, CONTOURS + k as u8, ring, 1.0, pal::DIM.a(0.5 * valley));
                }
            }
            // in the decay picture, where each weight is half of where it started
            let half = pal::DIM.a(0.8 * (1.0 - valley));
            kept_line(f, &p, HALF_LINE, &[(p.x[0], 0.5), (p.x[1], 0.5)], 1.0, half, true);
            kept_line(f, &p, HALF_LINE, &[(0.5, p.y[0]), (0.5, p.y[1])], 1.0, half, true);

            // the two walks (in the decay picture the first is not drawn)
            for j in 0..2 {
                let w = 2 * k + j;
                let point = |i: usize| (self.look.walks.get((w * n + i) * 2) as f64, self.look.walks.get((w * n + i) * 2 + 1) as f64);
                let colour = if j == 0 { SGD.a(0.75 * valley) } else { ADAM };
                // (gradient descent's walk is drawn while it is to be seen at all)
                if j == 1 || valley > 0.0 {
                    let walk: Vec<(f64, f64)> = (0..n).map(point).collect();
                    kept_series(f, &p, WALKS + w as u8, &walk, f64::INFINITY, if j == 0 { 1.0 } else { 1.8 }, colour, false);
                }
                // a dot at each step while they are apart (the valley), or every 100 steps (the decay)
                if valley > 0.5 {
                    for i in (0..n).take(160) {
                        let (x, y) = point(i);
                        p.dot(f, x, y, if j == 0 { 2.4 } else { 3.6 }, colour);
                    }
                } else if j == 1 {
                    for i in (0..n).step_by(100) {
                        let (x, y) = point(i);
                        p.dot(f, x, y, 4.0, colour);
                    }
                }
            }
            // the start, a white ring
            let start = p.at(self.look.walks.get(2 * k * n * 2) as f64, self.look.walks.get(2 * k * n * 2 + 1) as f64);
            if plot::within(start, p.rect, 0.5) {
                let ring: Vec<Vec2> = (0..=24).map(|i| start + 5.0 * Vec2::from_angle(TAU * i as f32 / 24.0)).collect();
                plot::path_of(f, MARKS, &ring, 1.8, pal::WHITE);
            }

            // the title above and the counts below
            let title = match (s.pair, k) {
                (Pair::Valley, 0) => "a valley lined up with the axes".to_string(),
                (Pair::Valley, _) => format!("the same valley turned by {}°", num(s.angle, 0)),
                (Pair::Decay, 0) => if lay.narrow { "L2: decay in the gradient".to_string() } else { "Adam + L2: decay added to the gradient".to_string() },
                (Pair::Decay, _) => if lay.narrow { "AdamW: decay off the weights".to_string() } else { "AdamW: decay taken off the weights".to_string() },
            };
            f.text_px(Vec2::new(min.x - if lay.narrow { 28.0 } else { 38.0 }, min.y - 6.0), title, lay.text, pal::TEXT, Align2::LEFT_BOTTOM);
            let below = match s.pair {
                Pair::Valley => {
                    let steps = |w: &Walk| w.steps_to(ARRIVED).map_or("over 2000".to_string(), count);
                    vec![(format!("gradient descent: {} steps", steps(&self.walks[2 * k])), SGD), (format!("Adam: {} steps", steps(&self.walks[2 * k + 1])), ADAM)]
                }
                Pair::Decay => {
                    let walk = &self.walks[2 * k + 1];
                    let half = |i: usize| walk.half_life(i).map_or_else(|| format!("{} left after 2000", num(walk.at.last().map_or(1.0, |p| p[i]), 2)), |n| format!("halves in {}", count(n)));
                    vec![(format!("$w_1$ {}", half(0)), ADAM), (format!("$w_2$ {}", half(1)), ADAM)]
                }
            };
            let y0 = max.y + if lay.narrow { 20.0 } else { 24.0 };
            for (i, (text, colour)) in below.into_iter().enumerate() {
                f.text_px(Vec2::new(min.x - if lay.narrow { 28.0 } else { 38.0 }, y0 + i as f32 * (size + 4.0)), text, size, colour, Align2::LEFT_TOP);
            }
        }
        if let Some((min, max)) = lay.notes {
            let text = match s.pair {
                Pair::Valley => format!(
                    r"The loss is $\tfrac12(a^2 + 100\,b^2)$, with $a$ measured along the valley's floor and $b$ across it. Both walks start at the white ring and run for 2000 steps; the counts are the steps until the loss is a ten-thousandth of where it started. Gradient descent steps $2/101$ times the gradient, the fixed step that suits this valley best; Adam has $\eta = {}$.",
                    sci(s.eta)
                ),
                Pair::Decay => format!(
                    r"Each weight's gradient is $\pm 100$ for $w_1$ and $\pm 1$ for $w_2$, flipping sign at every step. With $\eta = {}$ and $\lambda = {}$, for 2000 steps from $(1, 1)$; a dot every 100 steps. The dashed lines are where a weight has halved.",
                    sci(s.eta),
                    sci(s.lambda)
                ),
            };
            plot_note(f, min, max.x - min.x, &text, size, pal::DIM);
        }
    }
}

/// Text broken into lines no wider than `width`, from `at` down, in logical pixels of
/// the screen (`kit::wrapped` takes the plane's own units).
fn plot_note(f: &mut Frame, at: Vec2, width: f32, text: &str, size: f32, color: Color) {
    let ppu = pixels_per_unit(f).max(1e-6);
    let Some(world) = f.cam.pick_ground(at * f.px) else { return };
    wrapped(f, world.truncate(), width / ppu, text, size, color);
}

#[cfg(test)]
mod tests {
    use super::*;

    fn close(a: f64, b: f64, tol: f64) -> bool {
        (a - b).abs() <= tol
    }

    /// The scene as an experiment sets it up, worked out.
    fn trial(k: usize) -> Apart {
        let mut s = Apart::create();
        s.set_up(k);
        s
    }

    /// Algorithm 1, written out by hand for three steps.
    #[test]
    fn adam_is_kingma_and_ba_algorithm_one() {
        let c = DEFAULTS;
        let gs = [0.3, -1.2, 2.5];
        let mut mo = Moments::default();
        let (mut m, mut v) = (0.0f64, 0.0f64);
        for (i, &g) in gs.iter().enumerate() {
            let t = (i + 1) as i32;
            m = 0.9 * m + 0.1 * g;
            v = 0.999 * v + 0.001 * g * g;
            let want = (m / (1.0 - 0.9f64.powi(t))) / ((v / (1.0 - 0.999f64.powi(t))).sqrt() + 1e-8);
            let got = mo.take(g, &c).u;
            assert!(close(got, want, 1e-12), "step {t}: {got} against {want}");
        }
        // without the corrections the averages are used as they stand
        let mut mo = Moments::default();
        let p = mo.take(2.0, &Consts { corrected: false, ..c });
        assert!(close(p.m_hat, 0.2, 1e-15) && close(p.v_hat, 0.004, 1e-15));
    }

    /// With the corrections, the first step is the sign of the first gradient, exactly
    /// (but for ε), whatever its size; for a steady gradient, every step is.
    #[test]
    fn the_first_step_is_eta_whatever_the_gradient() {
        for g in [1e-6, 0.37, -4.0, 1e5] {
            let p = Moments::default().take(g, &DEFAULTS);
            assert!(close(p.m_hat, g, 1e-12 * g.abs()) && close(p.v_hat, g * g, 1e-12 * g * g), "{g}");
            assert!(close(p.u.abs(), 1.0, 1e-8 / g.abs() + 1e-12) && p.u.signum() == g.signum(), "{g}: {}", p.u);
        }
        let mut mo = Moments::default();
        for _ in 0..500 {
            assert!(close(mo.take(2.0, &DEFAULTS).u, 1.0, 1e-8));
        }
    }

    /// Without the corrections a steady gradient gives the steps (1 − β₁ᵗ)/√(1 − β₂ᵗ):
    /// 3.16 at the first, a peak of 6.57 at step 12, 3.24 at step 100.
    #[test]
    fn without_the_corrections_the_first_steps_are_too_large() {
        let c = Consts { corrected: false, ..DEFAULTS };
        let mut mo = Moments::default();
        let us: Vec<f64> = (0..3000).map(|_| mo.take(1.0, &c).u).collect();
        for (i, u) in us.iter().enumerate() {
            let t = (i + 1) as i32;
            let want = (1.0 - 0.9f64.powi(t)) / ((1.0 - 0.999f64.powi(t)).sqrt() + 1e-8);
            assert!(close(*u, want, 1e-9), "step {t}");
        }
        assert!(close(us[0], 0.1 / 0.001f64.sqrt(), 1e-6) && close(us[0], 3.162, 5e-4));
        let (peak, at) = us.iter().enumerate().fold((0.0, 0), |b, (i, &u)| if u > b.0 { (u, i + 1) } else { b });
        assert!(at == 12 && close(peak, 6.57, 0.005), "{peak} at {at}");
        assert!(close(us[99], 3.24, 0.005), "{}", us[99]);
    }

    /// RMSProp's first step is 1/√(1 − α) = 10 times its learning rate; AdaGrad's t-th
    /// step on a steady gradient is 1/√t.
    #[test]
    fn rmsprop_and_adagrad_as_pytorch_has_them() {
        let mut v = 0.0;
        assert!(close(rmsprop(&mut v, 0.7, 1e-8), 10.0, 1e-5));
        let mut sum = 0.0;
        for t in 1..=400 {
            let u = adagrad(&mut sum, 3.0, 1e-10);
            assert!(close(u, 1.0 / (t as f64).sqrt(), 1e-9), "{t}");
        }
    }

    /// Both kinds of decay, one step each, by hand; and for plain gradient descent the
    /// two kinds are the same step.
    #[test]
    fn the_two_kinds_of_decay() {
        let c = DEFAULTS;
        let (w0, g, eta, lambda) = (2.0, 0.5, 0.01, 0.1);
        let mut w = w0;
        decayed_step(&mut w, &mut Moments::default(), g, eta, lambda, Decay::Coupled, &c);
        let gl = g + lambda * w0;
        assert!(close(w, w0 - eta * gl / (gl.abs() + 1e-8), 1e-12));
        let mut w = w0;
        decayed_step(&mut w, &mut Moments::default(), g, eta, lambda, Decay::Decoupled, &c);
        assert!(close(w, w0 * (1.0 - eta * lambda) - eta * g / (g.abs() + 1e-8), 1e-12));
        // gradient descent: w − η(g + λw) and w(1 − ηλ) − ηg are one number
        assert!(close(w0 - eta * (g + lambda * w0), w0 * (1.0 - eta * lambda) - eta * g, 1e-15));
    }

    /// The paper's second way of computing the step (the corrections folded into the
    /// step size, ε added before correcting) gives the same steps exactly when its ε is
    /// ε√(1 − β₂ᵗ), and differs early on when the two are the same number.
    #[test]
    fn the_two_orders_of_computing_it() {
        let c = DEFAULTS;
        let mut mo = Moments::default();
        let (mut m, mut v) = (0.0f64, 0.0f64);
        for t in 1..=50 {
            let g = 1e-8 * (1.0 + 0.5 * (t as f64).sin());
            let u = mo.take(g, &c).u;
            m = 0.9 * m + 0.1 * g;
            v = 0.999 * v + 0.001 * g * g;
            let fold = (1.0 - 0.999f64.powi(t)).sqrt() / (1.0 - 0.9f64.powi(t));
            let matched = fold * m / (v.sqrt() + 1e-8 * (1.0 - 0.999f64.powi(t)).sqrt());
            let same = fold * m / (v.sqrt() + 1e-8);
            assert!(close(u, matched, 1e-9 * u.abs().max(1.0)), "step {t}");
            if t < 10 {
                assert!(same < 0.5 * u, "step {t}: {same} against {u}");
            }
        }
    }

    /// The two limits, by simulation over long runs (the averages settled, the first
    /// 5000 steps left out): with noise a tenth of the steady part the mean step is
    /// within 1 % of 1/√1.01, and the step's spread about 0.1 × 0.23; with noise ten
    /// times the steady part it is within 10 % of 0.0995, its spread near 0.23.
    #[test]
    fn the_two_limits() {
        for (mean, noise, tol, spread) in [(1.0, 0.1, 0.01, 0.023), (0.1, 1.0, 0.1, 0.229)] {
            let mut rng = Rng::new(5);
            let mut mo = Moments::default();
            let (mut sum, mut sq, mut k) = (0.0, 0.0, 0.0);
            for t in 0..105_000 {
                let u = mo.take(mean + noise * rng.normal() as f64, &DEFAULTS).u;
                if t >= 5000 {
                    sum += u;
                    sq += u * u;
                    k += 1.0;
                }
            }
            let avg = sum / k;
            let sd = (sq / k - avg * avg).sqrt();
            let want = settled_ratio(mean, noise);
            assert!(close(avg, want, tol * want), "{mean}, {noise}: mean step {avg} against {want}");
            assert!(close(sd, spread, 0.2 * spread), "{mean}, {noise}: spread {sd}");
        }
    }

    /// Kingma and Ba's bound for the sparsest case, (1 − β₁)/√(1 − β₂) = 3.16: a gradient
    /// after a long run of zeros comes close to it. It can be passed: a gradient that
    /// grows tenfold and stays there gives 5.12 seventeen steps later, and the most
    /// any gradients can give, late in a run, is (1 − β₁)/√((1 − β₂)(1 − β₁²/β₂)) =
    /// 7.27 (Cauchy and Schwarz, reached by a gradient that grows by β₂/β₁ a step).
    #[test]
    fn the_largest_step() {
        let c = DEFAULTS;
        let stated = (1.0 - c.beta1) / (1.0 - c.beta2).sqrt();
        assert!(close(stated, 3.162, 5e-4));
        let mut mo = Moments::default();
        for _ in 0..20_000 {
            mo.take(0.0, &c);
        }
        let u = mo.take(1.0, &c).u;
        assert!(close(u, stated, 1e-3), "{u}");
        let mut mo = Moments::default();
        for _ in 0..3000 {
            mo.take(1.0, &c);
        }
        let most = (0..100).map(|_| mo.take(10.0, &c).u).fold(0.0, f64::max);
        assert!(close(most, 5.12, 0.005), "{most}");
        // the scene's own jump, with 3000 steps shown and noise 0.1: 4.20
        let mut s = Apart::create();
        s.setting = Setting { mean: 1.0, noise: 0.1, steps: 3000, event: Event::Jump, ..OPENING };
        s.refresh();
        assert!(close(s.summary().largest.0, 4.20, 0.005), "{:?}", s.summary().largest);
        let sup = (1.0 - c.beta1) / ((1.0 - c.beta2) * (1.0 - c.beta1 * c.beta1 / c.beta2)).sqrt();
        assert!(close(sup, 7.27, 0.005));
        // (zeros for a long while, so that the corrections are all but 1, then the growth)
        let mut mo = Moments::default();
        for _ in 0..7700 {
            mo.take(0.0, &c);
        }
        let mut g = 1.0;
        let mut last = 0.0;
        for _ in 0..300 {
            last = mo.take(g, &c).u;
            g *= c.beta2 / c.beta1;
        }
        assert!(last < sup && last > sup - 0.01, "{last} against {sup}");
    }

    /// The captions of the stream's experiments, held to what the scene works out.
    #[test]
    fn the_stream_experiments_show_what_their_captions_say() {
        // a steady gradient: the first step exactly η, the later ones between 0.94 and 1.04
        let s = trial(0);
        let u = s.stream.shown(s.setting.steps);
        assert!(close(u[0], 1.0, 2e-8));
        let (lo, hi) = u[1..].iter().fold((f64::MAX, f64::MIN), |b, &x| (b.0.min(x), b.1.max(x)));
        assert!(close(lo, 0.94, 0.005) && close(hi, 1.04, 0.005), "{lo} to {hi}");
        // AdaGrad's step a tenth by step 100
        assert!(close(s.stream.adagrad[99], 0.1, 0.01), "{}", s.stream.adagrad[99]);

        // a thousand times larger: the same steps; gradient descent's a thousand times
        let big = trial(1);
        let mut one = trial(1);
        one.setting.scale = 1.0;
        one.refresh();
        for t in 0..big.setting.steps {
            assert!(close(big.stream.u[t], one.stream.u[t], 1e-7 * one.stream.u[t].abs().max(1.0)), "step {}", t + 1);
            assert!(close(big.stream.sgd[t], 1000.0 * one.stream.sgd[t], 1e-9 * big.stream.sgd[t].abs().max(1.0)));
        }
        // and at 10⁻¹⁰ the divisor is almost ε: the steps are m̂/ε, about a hundredth
        let mut tiny = trial(1);
        tiny.setting.scale = 1e-10;
        tiny.refresh();
        let n = tiny.setting.steps;
        for t in 0..n {
            let m = tiny.stream.m_hat[t] * 1e-10;
            assert!(close(tiny.stream.u[t], m / 1e-8, 0.02 * (m / 1e-8).abs() + 1e-12), "step {}", t + 1);
        }
        let avg = tiny.stream.u[..n].iter().sum::<f64>() / n as f64;
        let usual = one.stream.u[..n].iter().sum::<f64>() / n as f64;
        assert!(avg > 0.008 && avg < 0.012 && usual > 50.0 * avg, "{avg} against {usual}");

        // mostly noise: the mean step 0.097 over steps 101 to 3000, spread 0.22; 46 % of
        // the signs wrong
        let s = trial(2);
        let sum = s.summary();
        assert!(sum.from == 101 && close(sum.mean_late, 0.097, 0.0006), "{}", sum.mean_late);
        assert!(close(sum.sd_late, 0.22, 0.003), "{}", sum.sd_late);
        assert!(close(settled_ratio(0.1, 1.0), 0.0995, 5e-5));
        // (the sign of a normal number of mean 0.1 and spread 1 is wrong with chance Φ(−0.1))
        let wrong = 0.5 * erfc(0.1 / 2f64.sqrt());
        assert!(close(wrong, 0.46, 0.002), "{wrong}");

        // one huge spike: at step 500, the step 2.1; 1372 of the 1500 steps after it under
        // η/2; 339 with β₂ = 0.99; between 1365 and 1386 with β₁ = 0.5 or 0.99
        let s = trial(3);
        assert_eq!(s.setting.event_step(), 500);
        let at = s.stream.u[499];
        assert!(close(at, 2.1, 0.05), "{at}");
        let small = s.summary().small_after;
        assert_eq!(small, 1372);
        assert_eq!(s.setting.steps - s.setting.event_step(), 1500);
        // m̂ back within 5 % of the steady part 50 steps after the spike
        assert!(close(s.stream.m_hat[549], 1.0, 0.05), "{}", s.stream.m_hat[549]);
        let with = |b1: f64, b2: f64| {
            let mut t = trial(3);
            t.setting.c.beta1 = b1;
            t.setting.c.beta2 = b2;
            t.refresh();
            t.summary().small_after
        };
        assert_eq!(with(0.9, 0.99), 339);
        assert_eq!((with(0.5, 0.999), with(0.99, 0.999)), (1365, 1386));

        // the corrections off: 3.16 first, 6.57 at step 12, 3.24 at step 100; RMSProp 10
        let s = trial(4);
        let u = &s.stream.u;
        assert!(close(u[0], 3.16, 0.005) && close(u[11], 6.57, 0.005) && close(u[99], 3.24, 0.005));
        assert_eq!(s.summary().largest.1, 12);
        assert!(s.stream.other[..100].iter().all(|x| close(*x, 1.0, 1e-7)));
        assert!(close(s.stream.rmsprop[0], 10.0, 1e-6));
    }

    /// The complementary error function for a small `x`, from the Taylor series of erf
    /// (for the chance of a wrong sign above: Φ(−x√2) = erfc(x)/2).
    fn erfc(x: f64) -> f64 {
        let mut term = x;
        let mut sum = x;
        for n in 1..40 {
            term *= -x * x / n as f64;
            sum += term / (2 * n + 1) as f64;
        }
        1.0 - 2.0 / std::f64::consts::PI.sqrt() * sum
    }

    /// Balles and Hennig's reading is an identity: m̂/√v̂ is the sign of m̂ divided by
    /// √(1 + (v̂ − m̂²)/m̂²), for any m̂ ≠ 0 and v̂ > 0; and the factor is 1/√(1 + σ²/ḡ²)
    /// when v̂ = ḡ² + σ² and m̂ = ḡ.
    #[test]
    fn the_sign_times_a_factor() {
        for (m, v) in [(0.3f64, 0.5f64), (-1.2, 2.0), (2.0, 4.0), (-0.01, 1.0)] {
            let factor = 1.0 / (1.0 + (v - m * m) / (m * m)).sqrt();
            assert!(close(m / v.sqrt(), m.signum() * factor, 1e-12), "{m}, {v}");
        }
        let (g, sigma) = (0.1, 1.0);
        assert!(close(settled_ratio(g, sigma), 1.0 / (1.0 + sigma * sigma / (g * g)).sqrt(), 1e-12));
    }

    /// The captions of the two weights' experiments.
    #[test]
    fn the_two_weights_show_what_their_captions_say() {
        let s = trial(5);
        let [gd0, adam0, gd1, adam1] = &s.walks;
        assert_eq!(gd0.steps_to(ARRIVED), Some(231));
        assert_eq!(gd1.steps_to(ARRIVED), Some(231));
        assert_eq!(adam0.steps_to(ARRIVED), Some(101));
        assert_eq!(adam1.steps_to(ARRIVED), Some(379));
        let [_, right_angle] = valley_walks(90.0, 0.05, &DEFAULTS);
        assert_eq!(right_angle.steps_to(ARRIVED), Some(101));
        // gradient descent does not care how the valley is turned
        for angle in [0.0, 17.0, 45.0, 90.0, 133.0] {
            let [gd, _] = valley_walks(angle, 0.05, &DEFAULTS);
            for (a, b) in gd.loss.iter().zip(&gd0.loss) {
                assert!(close(*a, *b, 1e-9 * b.max(1e-12)), "{angle}°");
            }
        }

        let s = trial(6);
        let (coupled, decoupled) = (&s.walks[1], &s.walks[3]);
        assert_eq!(coupled.half_life(0), None);
        let kept = coupled.at[PAIR_STEPS][0];
        assert!(close(kept, 0.96, 0.005), "{kept}");
        assert_eq!(coupled.half_life(1), Some(673));
        assert_eq!(decoupled.half_life(0), Some(675));
        assert_eq!(decoupled.half_life(1), Some(675));
        // without decay the two weights stay within 0.02 of where they start (0.018 at
        // most), and end at 0.98
        let [still, _] = decay_walks(0.01, 0.0, &DEFAULTS);
        let most = still.at.iter().fold(0.0f64, |m, w| m.max((w[0] - 1.0).abs()).max((w[1] - 1.0).abs()));
        assert!(most > 0.015 && most < 0.02, "{most}");
        assert!(close(still.at[PAIR_STEPS][0], 0.98, 0.005) && close(still.at[PAIR_STEPS][1], 0.98, 0.005));
        // AdamW's walk runs down the diagonal: both weights the same at every step (but
        // for ε, which is a different share of the two gradients)
        assert!(decoupled.at.iter().all(|w| close(w[0], w[1], 1e-9)));
    }

    /// Degenerate settings: everything worked out stays finite.
    #[test]
    fn the_ends_of_the_controls_are_fine() {
        let mut s = Apart::create();
        for mean in [0.0, -3.0, 3.0] {
            for noise in [0.0, 5.0] {
                for scale in [1e-12, 1e4] {
                    for (b1, b2) in [(0.0, 0.0), (0.999, 0.9999), (0.0, 0.9999)] {
                        for eps in [1e-12, 1.0] {
                            for event in EVENTS {
                                s.setting = Setting { mean, noise, scale, event, c: Consts { beta1: b1, beta2: b2, eps, corrected: mean >= 0.0 }, steps: if noise > 0.0 { 10 } else { 3000 }, ..OPENING };
                                s.refresh();
                                let st = &s.stream;
                                for list in [&st.g, &st.m_hat, &st.size, &st.u, &st.other, &st.sgd, &st.adagrad, &st.rmsprop] {
                                    assert!(list.iter().all(|v| v.is_finite()), "{:?}", s.setting);
                                }
                                assert!(frame_goal(&s.setting, st).iter().all(|v| v.is_finite()));
                            }
                        }
                    }
                }
            }
        }
        for pair in [Pair::Valley, Pair::Decay] {
            for eta in [0.001, 0.5] {
                for lambda in [0.0, 1.0] {
                    s.setting = Setting { pair, eta, lambda, angle: 180.0, ..OPENING };
                    s.refresh();
                    assert!(s.walks.iter().all(|w| w.at.iter().all(|p| p[0].is_finite() && p[1].is_finite())));
                }
            }
        }
        // out of range and not a number: brought back
        s.setting = Setting { mean: f64::NAN, scale: 0.0, steps: 0, angle: -5.0, ..OPENING };
        s.refresh();
        assert!(s.setting.mean == 1.0 && s.setting.scale == 1e-12 && s.setting.steps == LEAST_STEPS && s.setting.angle == 0.0);
    }

    /// The layout in windows from a phone's to a desktop's, and far too small ones.
    #[test]
    fn the_layout_fits_every_window() {
        for (w, h) in [(1100.0, 1000.0), (1075.0, 1000.0), (390.0, 800.0), (390.0, 640.0), (700.0, 500.0), (240.0, 180.0), (1.0, 1.0), (2000.0, 300.0)] {
            let lay = Layout::of(Vec2::new(w, h), 40.0);
            for r in [lay.stream, lay.steps].into_iter().chain(lay.squares).chain(lay.events).chain(lay.beside).chain(lay.pairs) {
                assert!(r.0.is_finite() && r.1.is_finite() && r.1.x >= r.0.x && r.1.y >= r.0.y, "{w}x{h}: {r:?}");
            }
            if !lay.compact {
                // nothing runs off the bottom
                let bottom = lay.squares[0].1.y + BELOW_SQUARE;
                assert!(bottom <= h + 1.0, "{w}x{h}: down to {bottom}");
            }
        }
    }

    /// The numbers behind the captions (run with --ignored --nocapture).
    #[test]
    #[ignore]
    fn explore() {
        for (k, t) in TRIALS.iter().enumerate() {
            let s = trial(k);
            let sum = s.summary();
            println!("{:<32} first {:.4} largest {:.4} at {} mean_late {:.4} sd_late {:.4} small_after {}", t.button, sum.first, sum.largest.0, sum.largest.1, sum.mean_late, sum.sd_late, sum.small_after);
            let u = s.stream.shown(s.setting.steps);
            let (lo, hi) = u[1..].iter().fold((f64::MAX, f64::MIN), |b, &x| (b.0.min(x), b.1.max(x)));
            println!("    later steps from {lo:.4} to {hi:.4}; u[99] {:.4}; at event {:.4}", u.get(99).copied().unwrap_or(0.0), u.get(s.setting.event_step() - 1).copied().unwrap_or(0.0));
            println!("    walks: {:?}", s.walks.iter().map(|w| (w.steps_to(ARRIVED), w.half_life(0), w.half_life(1), w.at.last().copied())).collect::<Vec<_>>());
        }
        for (b1, b2) in [(0.9, 0.999), (0.9, 0.99), (0.5, 0.999), (0.99, 0.999), (0.0, 0.999)] {
            let mut t = trial(3);
            t.setting.c.beta1 = b1;
            t.setting.c.beta2 = b2;
            t.refresh();
            println!("spike with {b1}, {b2}: {} small, at it {:.3}", t.summary().small_after, t.stream.u[499]);
        }
        for scale in [1e-8, 1e-9, 1e-10, 1e-11] {
            let mut t = trial(1);
            t.setting.scale = scale;
            t.refresh();
            let n = t.setting.steps;
            let avg = t.stream.u[..n].iter().sum::<f64>() / n as f64;
            println!("scale {scale}: mean step {avg:.4}");
        }
        for (mean, noise) in [(1.0, 0.0), (1.0, 0.1), (1.0, 1.0)] {
            let mut t = Apart::create();
            t.setting = Setting { mean, noise, steps: 3000, event: Event::Jump, ..OPENING };
            t.refresh();
            let sum = t.summary();
            println!("jump ×10 at {} with noise {noise}: largest {:.4} at {}", t.setting.event_step(), sum.largest.0, sum.largest.1);
        }
        for angle in [0.0, 10.0, 20.0, 30.0, 45.0, 60.0, 90.0] {
            let [gd, adam] = valley_walks(angle, 0.05, &DEFAULTS);
            println!("angle {angle}: gd {:?} adam {:?}", gd.steps_to(ARRIVED), adam.steps_to(ARRIVED));
        }
    }
}

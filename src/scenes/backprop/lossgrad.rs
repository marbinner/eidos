//! Where the gradient starts: the slope of the loss with respect to a network's raw
//! outputs, its logits, for four pairings of output function and loss side by side.
//!
//! The backward sweep of a network begins here. The first thing it hands back is
//! ∂L/∂s, the loss's slope with respect to the logits s, and every gradient behind it
//! is that vector carried back through the layers. Which output function turns s
//! into a prediction, and which loss scores the prediction against the truth y,
//! decides that first slope:
//!
//! - softmax with cross-entropy: p = softmax(s/T), L = −Σ y ln p, slope (p − y)/T;
//! - a sigmoid on each logit with the squared error, L = ½ Σ (σ(s) − y)²: slope
//!   (σ − y) σ (1 − σ), which is nearly 0 wherever the sigmoid is flat, wrong or not;
//! - the same sigmoid with its own cross-entropy: slope σ − y;
//! - the logit itself with the squared error (an output for numbers): slope s − y.
//!
//! Everything is worked out in f64 from those closed forms, and each is tested against
//! differences of its loss. A race runs plain gradient steps on the logits under each
//! pairing from the same start, and counts the steps each needs to get the answer right.

use super::*;
use crate::scenes::stats::plot::{self, Glide, Plot, Rect, AXES, CURVES, MARKS, OWN};

pub const DEF: SceneDef = SceneDef {
    id: "lossgrad",
    title: "Where the gradient starts: losses",
    group: GROUP,
    blurb: "Softmax with cross-entropy hands back the probabilities minus the truth; other pairings of output and loss hand back less, and stall.",
    make: |_| Box::new(LossGrad::create()),
};

// ----- the mathematics -------------------------------------------------------------------

/// The most classes, and the fewest.
const MOST: usize = 5;
const FEWEST: usize = 3;
/// How far a logit can be dragged either side of 0.
const LOGIT_MOST: f64 = 6.0;
/// The temperature's range, label smoothing's largest, and the race's step sizes.
const TEMP_LO: f64 = 0.25;
const TEMP_HI: f64 = 8.0;
const SMOOTH_MOST: f64 = 0.5;
const ETA_LO: f64 = 0.05;
const ETA_HI: f64 = 1.5;

/// A value for each class (those beyond the number of classes are unused, and 0).
type Vector = [f64; MOST];

/// An output function and the loss it is paired with.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Pairing {
    SoftmaxCe,
    SigmoidSe,
    SigmoidCe,
    LinearSe,
}

const PAIRINGS: [Pairing; 4] = [Pairing::SoftmaxCe, Pairing::SigmoidSe, Pairing::SigmoidCe, Pairing::LinearSe];

/// σ(x), without overflow on either side.
fn sigmoid(x: f64) -> f64 {
    if x >= 0.0 {
        1.0 / (1.0 + (-x).exp())
    } else {
        let e = x.exp();
        e / (1.0 + e)
    }
}

/// ln(1 + eˣ), exactly enough for any x (the cross-entropy of a sigmoid is made of it).
fn softplus(x: f64) -> f64 {
    x.max(0.0) + (-x.abs()).exp().ln_1p()
}

/// ln Σ e^z, shifted by the largest so that nothing overflows.
fn log_sum_exp(z: &[f64]) -> f64 {
    let top = z.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    if !top.is_finite() {
        return top;
    }
    top + z.iter().map(|v| (v - top).exp()).sum::<f64>().ln()
}

/// The logits divided by the temperature: what the softmax sees.
fn tempered(s: &[f64], temp: f64) -> Vector {
    let mut z = [0.0; MOST];
    for (z, v) in z.iter_mut().zip(s) {
        *z = v / temp;
    }
    z
}

/// The softmax of the first `k` entries of z.
fn softmax(z: &[f64]) -> Vector {
    let lse = log_sum_exp(z);
    let mut p = [0.0; MOST];
    for (p, v) in p.iter_mut().zip(z) {
        *p = (v - lse).exp();
    }
    p
}

impl Pairing {
    fn name(self) -> &'static str {
        match self {
            Pairing::SoftmaxCe => "softmax + cross-entropy",
            Pairing::SigmoidSe => "sigmoid + squared error",
            Pairing::SigmoidCe => "sigmoid + cross-entropy",
            Pairing::LinearSe => "linear + squared error",
        }
    }

    /// The name where room is short.
    fn short(self) -> &'static str {
        match self {
            Pairing::SoftmaxCe => "softmax + CE",
            Pairing::SigmoidSe => "sigmoid + SE",
            Pairing::SigmoidCe => "sigmoid + CE",
            Pairing::LinearSe => "linear + SE",
        }
    }

    /// Its colour in the charts (and its name's, over its row).
    fn color(self) -> Color {
        match self {
            Pairing::SoftmaxCe => pal::WHITE,
            Pairing::SigmoidSe => pal::RED,
            Pairing::SigmoidCe => pal::BLUE,
            Pairing::LinearSe => pal::DIM.mix(pal::WHITE, 0.25),
        }
    }

    /// The slope, as its row writes it.
    fn slope_formula(self, temp: f64) -> &'static str {
        match self {
            Pairing::SoftmaxCe if (temp - 1.0).abs() > 1e-9 => r"$(p_k - y_k)/T$",
            Pairing::SoftmaxCe => r"$p_k - y_k$",
            Pairing::SigmoidSe => r"$(\sigma_k - y_k)\,\sigma_k(1 - \sigma_k)$",
            Pairing::SigmoidCe => r"$\sigma_k - y_k$",
            Pairing::LinearSe => r"$s_k - y_k$",
        }
    }

    /// What the output function makes of the logits: the softmax's probabilities, each
    /// logit's sigmoid, or the logits themselves.
    fn outputs(self, s: &[f64], temp: f64) -> Vector {
        let k = s.len().min(MOST);
        let mut out = [0.0; MOST];
        match self {
            Pairing::SoftmaxCe => out = softmax(&tempered(s, temp)[..k]),
            Pairing::SigmoidSe | Pairing::SigmoidCe => {
                for (o, v) in out.iter_mut().zip(s) {
                    *o = sigmoid(*v);
                }
            }
            Pairing::LinearSe => out[..k].copy_from_slice(&s[..k]),
        }
        out
    }

    /// The loss at the logits `s` against the truth `y` (natural logarithms).
    fn loss(self, s: &[f64], y: &[f64], temp: f64) -> f64 {
        let k = s.len().min(y.len()).min(MOST);
        let (s, y) = (&s[..k], &y[..k]);
        match self {
            Pairing::SoftmaxCe => {
                // −Σ y ln p, with ln p_k = z_k − ln Σ e^z
                let z = tempered(s, temp);
                let lse = log_sum_exp(&z[..k]);
                y.iter().zip(&z).map(|(y, z)| y * (lse - z)).sum()
            }
            Pairing::SigmoidSe => 0.5 * s.iter().zip(y).map(|(s, y)| (sigmoid(*s) - y).powi(2)).sum::<f64>(),
            // −[y ln σ(s) + (1 − y) ln(1 − σ(s))], with ln σ(s) = −softplus(−s)
            Pairing::SigmoidCe => s.iter().zip(y).map(|(s, y)| y * softplus(-s) + (1.0 - y) * softplus(*s)).sum(),
            Pairing::LinearSe => 0.5 * s.iter().zip(y).map(|(s, y)| (s - y).powi(2)).sum::<f64>(),
        }
    }

    /// The slope of the loss with respect to each logit, by its closed form.
    fn slope(self, s: &[f64], y: &[f64], temp: f64) -> Vector {
        let k = s.len().min(y.len()).min(MOST);
        let mut g = [0.0; MOST];
        match self {
            Pairing::SoftmaxCe => {
                // (p Σy − y)/T, which is (p − y)/T for a truth that adds up to 1
                let p = self.outputs(&s[..k], temp);
                let total: f64 = y[..k].iter().sum();
                for i in 0..k {
                    g[i] = (p[i] * total - y[i]) / temp;
                }
            }
            Pairing::SigmoidSe => {
                for i in 0..k {
                    let q = sigmoid(s[i]);
                    g[i] = (q - y[i]) * q * (1.0 - q);
                }
            }
            Pairing::SigmoidCe => {
                for i in 0..k {
                    g[i] = sigmoid(s[i]) - y[i];
                }
            }
            Pairing::LinearSe => {
                for i in 0..k {
                    g[i] = s[i] - y[i];
                }
            }
        }
        g
    }
}

/// The softmax's own Jacobian, ∂p/∂s = (diag(p) − p pᵀ)/T, at the probabilities p.
fn softmax_jacobian(p: &[f64], temp: f64) -> [[f64; MOST]; MOST] {
    let mut j = [[0.0; MOST]; MOST];
    for (a, row) in j.iter_mut().enumerate().take(p.len().min(MOST)) {
        for (b, cell) in row.iter_mut().enumerate().take(p.len().min(MOST)) {
            *cell = ((if a == b { p[a] } else { 0.0 }) - p[a] * p[b]) / temp;
        }
    }
    j
}

/// The truth: 1 − ε for the target, ε shared evenly among the rest.
fn truth(k: usize, target: usize, smooth: f64) -> Vector {
    let k = k.clamp(1, MOST);
    let rest = if k > 1 { smooth / (k - 1) as f64 } else { 0.0 };
    let mut y = [0.0; MOST];
    for (i, y) in y.iter_mut().enumerate().take(k) {
        *y = if i == target { 1.0 - smooth } else { rest };
    }
    y
}

/// The target's logit less the largest of the others: above 0, the network's answer
/// (its largest logit) is the target.
fn gap(s: &[f64], target: usize) -> f64 {
    let others = s.iter().enumerate().filter(|(i, _)| *i != target).map(|(_, v)| *v).fold(f64::NEG_INFINITY, f64::max);
    s.get(target).copied().unwrap_or(0.0) - others
}

/// Where softmax with cross-entropy against a softened truth has no slope: the gap at
/// which p = y, T ln((1 − ε)(K − 1)/ε). (None for the hard truth: there is no such place.)
fn reachable_gap(k: usize, smooth: f64, temp: f64) -> Option<f64> {
    (smooth > 0.0 && k > 1).then(|| temp * ((1.0 - smooth) * (k - 1) as f64 / smooth).ln())
}

/// Everything the loss and its slope depend on.
#[derive(Clone, Copy, Debug, PartialEq)]
struct Setting {
    k: usize,
    s: Vector,
    target: usize,
    temp: f64,
    smooth: f64,
}

impl Setting {
    /// Kept to the controls' ranges, with nothing that is not a number.
    fn kept(mut self) -> Setting {
        self.k = self.k.clamp(FEWEST, MOST);
        self.target = self.target.min(self.k - 1);
        let k = self.k;
        for (i, v) in self.s.iter_mut().enumerate() {
            *v = if i >= k || !v.is_finite() { 0.0 } else { v.max(-LOGIT_MOST).min(LOGIT_MOST) };
        }
        self.temp = if self.temp.is_finite() { self.temp.max(TEMP_LO).min(TEMP_HI) } else { 1.0 };
        self.smooth = if self.smooth.is_finite() { self.smooth.max(0.0).min(SMOOTH_MOST) } else { 0.0 };
        self
    }

    fn logits(&self) -> &[f64] {
        &self.s[..self.k]
    }

    fn truth(&self) -> Vector {
        truth(self.k, self.target, self.smooth)
    }

    fn outputs(&self, p: Pairing) -> Vector {
        p.outputs(self.logits(), self.temp)
    }

    fn loss(&self, p: Pairing) -> f64 {
        p.loss(self.logits(), &self.truth()[..self.k], self.temp)
    }

    fn slope(&self, p: Pairing) -> Vector {
        p.slope(self.logits(), &self.truth()[..self.k], self.temp)
    }

    fn gap(&self) -> f64 {
        gap(self.logits(), self.target)
    }
}

// ----- the race ----------------------------------------------------------------------------

/// How many steps a race runs.
const RACE_STEPS: usize = 10_000;
/// How many steps of each lane are worked out in one frame (the race is ready within
/// ten frames, long before the film shows its end).
const RACE_CHUNK: usize = 1000;
/// How fast the film of a race plays: tenfolds of steps (on its logarithmic clock) a
/// second.
const FILM_RATE: f32 = 0.9;

/// One pairing's run: the logits as they now are, the gap after each step, and the
/// first step after which the answer was right.
#[derive(Clone, Debug)]
struct Lane {
    pairing: Pairing,
    y: Vector,
    s: Vector,
    gaps: Vec<f64>,
    right: Option<usize>,
}

impl Lane {
    fn new(pairing: Pairing, y: Vector, from: &Setting) -> Lane {
        let g = from.gap();
        let mut gaps = Vec::with_capacity(RACE_STEPS + 1);
        gaps.push(g);
        Lane { pairing, y, s: from.s, gaps, right: (g > 0.0).then_some(0) }
    }
}

/// Plain gradient steps on the logits, s ← s − η ∂L/∂s, from the same start under each
/// pairing; and, against a softened truth, softmax with cross-entropy against the hard
/// one beside them (the last lane), to show what the softening changes.
#[derive(Clone, Debug)]
struct Race {
    from: Setting,
    eta: f64,
    lanes: Vec<Lane>,
    /// Steps worked out so far (each lane has one more gap than this).
    done: usize,
}

impl Race {
    fn new(from: Setting, eta: f64) -> Race {
        let from = from.kept();
        let eta = if eta.is_finite() { eta.max(ETA_LO).min(ETA_HI) } else { 0.5 };
        let y = from.truth();
        let mut lanes: Vec<Lane> = PAIRINGS.iter().map(|&p| Lane::new(p, y, &from)).collect();
        if from.smooth > 0.0 {
            lanes.push(Lane::new(Pairing::SoftmaxCe, truth(from.k, from.target, 0.0), &from));
        }
        Race { from, eta, lanes, done: 0 }
    }

    /// Is there a lane for the hard truth beside the softened one?
    #[cfg(test)]
    fn ghost(&self) -> bool {
        self.lanes.len() > PAIRINGS.len()
    }

    fn finished(&self) -> bool {
        self.done >= RACE_STEPS
    }

    /// Work out up to `most` more steps of every lane.
    fn advance(&mut self, most: usize) {
        let (k, t, temp, eta) = (self.from.k, self.from.target, self.from.temp, self.eta);
        let until = (self.done + most).min(RACE_STEPS);
        for lane in &mut self.lanes {
            for n in self.done..until {
                let g = lane.pairing.slope(&lane.s[..k], &lane.y[..k], temp);
                for (s, g) in lane.s.iter_mut().zip(g).take(k) {
                    *s -= eta * g;
                }
                let d = gap(&lane.s[..k], t);
                lane.gaps.push(d);
                if lane.right.is_none() && d > 0.0 {
                    lane.right = Some(n + 1);
                }
            }
        }
        self.done = until;
    }

    /// All of it at once (the tests).
    #[cfg(test)]
    fn finish(&mut self) {
        self.advance(RACE_STEPS);
    }

    /// A lane's gap after `n` steps, between whole steps where `n` falls between them.
    fn gap_at(&self, lane: usize, n: f64) -> f64 {
        let Some(gaps) = self.lanes.get(lane).map(|l| &l.gaps) else { return 0.0 };
        if gaps.is_empty() {
            return 0.0;
        }
        let n = n.max(0.0).min((gaps.len() - 1) as f64);
        let i = (n.floor() as usize).min(gaps.len() - 1);
        let j = (i + 1).min(gaps.len() - 1);
        gaps[i] + (gaps[j] - gaps[i]) * (n - i as f64)
    }

    /// The lowest and highest gap worked out so far.
    fn range(&self) -> (f64, f64) {
        let mut lo = f64::INFINITY;
        let mut hi = f64::NEG_INFINITY;
        for g in self.lanes.iter().flat_map(|l| &l.gaps).filter(|g| g.is_finite()) {
            lo = lo.min(*g);
            hi = hi.max(*g);
        }
        if lo.is_finite() {
            (lo, hi)
        } else {
            (-1.0, 1.0)
        }
    }
}

/// The steps a race's curves are drawn through: every one of the first forty, then
/// spaced evenly on the logarithmic clock.
fn race_samples() -> Vec<usize> {
    let mut out: Vec<usize> = (0..=40).collect();
    let mut x = 40.0f64;
    while (x as usize) < RACE_STEPS {
        x *= 1.035;
        let n = (x.round() as usize).min(RACE_STEPS);
        if out.last().is_none_or(|l| n > *l) {
            out.push(n);
        }
    }
    out
}

/// Where step `n` lies on the race's clock: log₁₀(1 + n).
fn clock(n: f64) -> f64 {
    (1.0 + n.max(0.0)).log10()
}

/// The film's last moment: the race's last step.
fn film_end() -> f32 {
    clock(RACE_STEPS as f64) as f32
}

// ----- numbers as the picture writes them ------------------------------------------------

/// A number of a size from 0 to a few: two decimals, more for one under a tenth (two
/// significant digits, to a ten-thousandth), and a power of ten below that.
fn brief(v: f64) -> String {
    if !v.is_finite() {
        return "?".to_string();
    }
    let a = v.abs();
    if a == 0.0 {
        return "0".to_string();
    }
    if a >= 0.0995 {
        return num(v, 2);
    }
    if a >= 0.009_95 {
        return num(v, 3);
    }
    if a >= 0.000_099_5 {
        return num(v, 4);
    }
    let mut e = a.log10().floor() as i32;
    let mut m = (a / 10f64.powi(e)).round();
    if m >= 10.0 {
        (m, e) = (1.0, e + 1);
    }
    format!(r"${}{} \times 10^{{{}}}$", if v < 0.0 { "-" } else { "" }, m as i64, e)
}

/// A number with no more digits than it needs (to two decimals): −4, 0.5, 1.25.
fn plain(v: f64) -> String {
    let text = num(v, 2);
    if text.contains('.') {
        text.trim_end_matches('0').trim_end_matches('.').to_string()
    } else {
        text
    }
}

/// A step count as the race writes it.
fn steps_text(n: usize) -> String {
    if n == 1 {
        "1 step".to_string()
    } else {
        format!("{} steps", count(n))
    }
}

// ----- the experiments ---------------------------------------------------------------------

struct Trial {
    button: &'static str,
    caption: &'static str,
    s: [f64; 4],
    temp: f64,
    smooth: f64,
    /// Which chart a phone shows (see `Tab`).
    tab: Tab,
}

/// Every experiment has four classes, class 1 as the target, and races with η = 0.5.
const TRIALS: [Trial; 5] = [
    Trial {
        button: "probabilities minus the truth",
        caption: r"Class 1 is the target. The softmax gives it $p_1 = 0.24$, so the slope on its logit is $0.24 - 1 = -0.76$; on each other logit the slope is that class's own probability, 0.64, 0.09 and 0.03. The four add up to 0 (raising every logit by the same amount changes no probability), and none can be larger than 1 in size. A step against the slope raises the target's logit and lowers the others, class 2's the most, since it took the most probability.",
        s: [1.0, 2.0, 0.0, -1.0],
        temp: 1.0,
        smooth: 0.0,
        tab: Tab::Slope,
    },
    Trial {
        button: "confident and wrong: one shouts, one whispers",
        caption: r"The softmax gives class 2 a probability of 0.95 and the target 0.0003. Softmax with cross-entropy pulls the target's logit up with slope −1.00, as hard as it can (the size is $1 - p_1$, which never quite reaches 1). The sigmoid with the squared error pulls with −0.017: its slope carries the factor $\sigma(1 - \sigma) = 0.018$, nearly 0 where the sigmoid is flat (saturated). In the chart of the loss its curve levels off towards the left, and a level loss has no slope. The same sigmoid with cross-entropy pulls with −0.98: it is the pairing that matters, not the sigmoid.",
        s: [-4.0, 4.0, 1.0, -1.0],
        temp: 1.0,
        smooth: 0.0,
        tab: Tab::Loss,
    },
    Trial {
        button: "the race from a wrong start",
        caption: r"Plain gradient steps on the logits, $s \leftarrow s - \eta\,\partial L/\partial s$ with $\eta = 0.5$, from a start that gives class 2 a probability of 0.995, under each pairing. Softmax with cross-entropy has the answer right after 14 steps, and so has the sigmoid with cross-entropy. The sigmoid with the squared error needs 850: its slope on the target's logit starts at −0.0025, small but not 0, and in 100 steps the gap closes only from −12 to −11.74. The linear output with the squared error is right after 4 (its slope is the whole error, −7 on the target's logit at the start), but its outputs are numbers, not probabilities.",
        s: [-6.0, 6.0, 0.0, 0.0],
        temp: 1.0,
        smooth: 0.0,
        tab: Tab::Race,
    },
    Trial {
        button: "temperature",
        caption: r"$T = 3$: the softmax sees the logits divided by 3. The probabilities flatten, from $(0.24, 0.64, 0.09, 0.03)$ to $(0.28, 0.38, 0.20, 0.14)$, and the slope is $(p - y)/T$: on the target's logit −0.24 instead of −0.76, and never more than $1/3$ in size. In the race the softmax lane now needs 6 steps instead of 2. Below 1 the temperature sharpens instead: at $T = 0.5$ class 2 gets 0.86.",
        s: [1.0, 2.0, 0.0, -1.0],
        temp: 3.0,
        smooth: 0.0,
        tab: Tab::Slope,
    },
    Trial {
        button: "a softened target that can be reached",
        caption: r"With $\varepsilon = 0.1$ the truth is 0.9 for the target and 0.033 for each other class. Softmax with cross-entropy then has slope 0 at finite logits, where $p = y$: with the target's logit $\ln 27 = 3.30$ above each of the others. In the race its gap levels off there (the dashed line) and the logits stop growing. Against the hard truth (the dashed curve, $\varepsilon = 0$) the slope shrinks but never reaches 0: that gap is 9.83 after 10 000 steps and still climbing, 2.4 more than after 1000.",
        s: [-4.0, 4.0, 1.0, -1.0],
        temp: 1.0,
        smooth: 0.1,
        tab: Tab::Race,
    },
];

/// The scene opens as the second experiment sets it up.
const OPENING: usize = 1;
/// The race's step size in every experiment.
const TRIAL_ETA: f64 = 0.5;

// ----- the scene ---------------------------------------------------------------------------

/// Which chart a narrow window shows under the logits (a wide one shows all three).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Tab {
    Loss,
    Slope,
    Race,
}

const TABS: [(Tab, &str); 3] = [(Tab::Loss, "the loss"), (Tab::Slope, "the slope"), (Tab::Race, "the race")];

pub struct LossGrad {
    set: Setting,
    eta: f64,
    race: Race,
    /// How far the film of the race has got, on its logarithmic clock (log₁₀(1 + n)),
    /// and whether it is playing.
    film: f32,
    playing: bool,
    tab: Tab,
    look: Look,
    curves: Curves,
    /// The logit being dragged, if one is.
    dragging: Option<usize>,
    /// The window last drawn in (a new one is drawn at once, without gliding).
    canvas: Option<Vec2>,
}

impl LossGrad {
    /// (Nothing here needs the graphics device, so the tests can make one too.)
    fn create() -> LossGrad {
        let set = Setting { k: 4, s: [0.0; MOST], target: 0, temp: 1.0, smooth: 0.0 };
        let mut scene = LossGrad {
            set,
            eta: TRIAL_ETA,
            race: Race::new(set, TRIAL_ETA),
            film: 0.0,
            playing: false,
            tab: Tab::Loss,
            look: Look::default(),
            curves: Curves::default(),
            dragging: None,
            canvas: None,
        };
        scene.set_up(OPENING);
        scene
    }

    /// Set up one of the experiments in `TRIALS`, and run its race.
    fn set_up(&mut self, k: usize) {
        let t = &TRIALS[k];
        let mut s = [0.0; MOST];
        s[..4].copy_from_slice(&t.s);
        self.set = Setting { k: 4, s, target: 0, temp: t.temp, smooth: t.smooth }.kept();
        self.eta = TRIAL_ETA;
        self.tab = t.tab;
        self.start_race();
    }

    /// A new race from the setting as it is, its film from the start.
    fn start_race(&mut self) {
        self.set = self.set.kept();
        self.race = Race::new(self.set, self.eta);
        self.film = 0.0;
        self.playing = true;
    }

    /// Is the race shown one from the setting as it now is?
    fn race_is_current(&self) -> bool {
        self.race.from == self.set.kept() && self.race.eta == self.eta
    }

    /// The race's work for this frame, and its film's.
    fn work(&mut self, dt: f32) {
        if !self.race.finished() {
            self.race.advance(RACE_CHUNK);
        }
        if self.playing {
            self.film = (self.film + FILM_RATE * dt.max(0.0)).min(film_end());
            if self.film >= film_end() {
                self.playing = false;
            }
        }
    }

    /// The step the film has reached (a fraction between two whole steps), within what
    /// has been worked out.
    fn film_step(&self) -> f64 {
        (10f64.powf(self.film as f64) - 1.0).max(0.0).min(self.race.done as f64)
    }
}

// ----- what is drawn, gliding --------------------------------------------------------------

// The picture's numbers in the order the glide keeps them: the logits, the truth, the
// logarithm of the temperature, and the target's column.
const LOOK_S: usize = 0;
const LOOK_Y: usize = MOST;
const LOOK_TEMP: usize = 2 * MOST;
const LOOK_TARGET: usize = 2 * MOST + 1;
const LOOK_LEN: usize = 2 * MOST + 2;

// The charts' ranges: the loss chart's top, the slope chart's top, the race's bottom and top.
const RANGE_LOSS: usize = 0;
const RANGE_SLOPE: usize = 1;
const RANGE_LO: usize = 2;
const RANGE_HI: usize = 3;

#[derive(Clone, Debug, Default)]
struct Look {
    state: Glide,
    ranges: Glide,
    moving: bool,
}

/// The state the picture is to show.
fn look_goal(set: &Setting) -> Vec<f32> {
    let y = set.truth();
    let mut g = vec![0.0f32; LOOK_LEN];
    for i in 0..MOST {
        g[LOOK_S + i] = set.s[i] as f32;
        g[LOOK_Y + i] = y[i] as f32;
    }
    g[LOOK_TEMP] = set.temp.ln() as f32;
    g[LOOK_TARGET] = set.target as f32;
    g
}

/// The setting as the picture shows it now (between two settings, while it glides).
fn shown_setting(set: &Setting, look: &Glide) -> (Setting, Vector) {
    let mut s = [0.0; MOST];
    let mut y = [0.0; MOST];
    for i in 0..MOST {
        s[i] = look.get(LOOK_S + i) as f64;
        y[i] = look.get(LOOK_Y + i) as f64;
    }
    let temp = (look.get(LOOK_TEMP) as f64).exp();
    let shown = Setting { s, temp, ..*set };
    (shown, y)
}

/// The margin charts' curves: the loss, and the size of the slope on the target's
/// logit, as the target's logit goes from −6 to 6 with the other logits as shown.
#[derive(Clone, Debug, Default)]
struct Curves {
    key: Vec<u64>,
    loss: [Vec<f64>; 4],
    slope: [Vec<f64>; 4],
}

const CURVE_N: usize = 121;

fn curve_x(i: usize) -> f64 {
    -LOGIT_MOST + 2.0 * LOGIT_MOST * i as f64 / (CURVE_N - 1) as f64
}

impl Curves {
    fn update(&mut self, shown: &Setting, y: &Vector) {
        let k = shown.k;
        let mut key: Vec<u64> = shown.s[..k].iter().chain(&y[..k]).map(|v| v.to_bits()).collect();
        key.extend([shown.temp.to_bits(), k as u64, shown.target as u64]);
        if key == self.key && self.loss[0].len() == CURVE_N {
            return;
        }
        self.key = key;
        let t = shown.target;
        for (j, p) in PAIRINGS.iter().enumerate() {
            let (loss, slope) = (&mut self.loss[j], &mut self.slope[j]);
            loss.clear();
            slope.clear();
            let mut s = shown.s;
            for i in 0..CURVE_N {
                s[t] = curve_x(i);
                loss.push(p.loss(&s[..k], &y[..k], shown.temp));
                slope.push(p.slope(&s[..k], &y[..k], shown.temp)[t].abs());
            }
        }
    }

    /// The top of the loss chart: a little above the highest of the classifiers'
    /// curves (the linear output's squared error grows far beyond them and is cut off).
    fn loss_top(&self) -> f64 {
        let top = self.loss[..3].iter().flatten().copied().filter(|v| v.is_finite()).fold(0.0, f64::max);
        (top * 1.08).max(1.0)
    }
}

/// The ranges the charts glide to.
fn range_goal(set: &Setting, curves: &Curves, race: &Race) -> Vec<f32> {
    let (lo, hi) = race.range();
    let (lo, hi) = (lo.min(0.0), hi.max(0.0));
    let pad = 0.08 * (hi - lo).max(1.0);
    let mut g = vec![0.0f32; 4];
    g[RANGE_LOSS] = curves.loss_top() as f32;
    g[RANGE_SLOPE] = (1.2 * (1.0 / set.temp).max(1.0)) as f32;
    g[RANGE_LO] = (lo - pad) as f32;
    g[RANGE_HI] = (hi + pad) as f32;
    g
}

impl LossGrad {
    /// Bring the picture `dt` seconds nearer to what it shows (`snap`: there at once).
    fn follow(&mut self, dt: f32, snap: bool) {
        let goal = look_goal(&self.set);
        let look = &mut self.look;
        if snap || look.state.shown.len() != LOOK_LEN {
            look.state.snap(&goal);
        } else {
            look.state.follow(&goal, dt, false);
        }
        // (a logit in the hand is where the hand is)
        if let Some(i) = self.dragging {
            if let Some(v) = look.state.shown.get_mut(LOOK_S + i) {
                *v = goal[LOOK_S + i];
            }
        }
        let still = look.state.shown == goal;
        let (shown, y) = shown_setting(&self.set, &look.state);
        self.curves.update(&shown, &y);
        let ranges = range_goal(&shown, &self.curves, &self.race);
        let ranges_moving = if snap || look.ranges.shown.len() != ranges.len() {
            look.ranges.snap(&ranges);
            false
        } else {
            look.ranges.follow(&ranges, dt, false)
        };
        look.moving = !still || ranges_moving;
    }
}

// ----- the layout --------------------------------------------------------------------------

/// Where everything goes in a window.
#[derive(Clone, Copy, Debug, PartialEq)]
struct Layout {
    view: Vec2,
    /// Everything at once, in two columns.
    wide: bool,
    /// Only the logits and the rows: the window is too small for a chart.
    compact: bool,
    /// Text sizes: headings, words, small print.
    big: f32,
    text: f32,
    small: f32,
    /// The heading's top-left corner, and whether it has a second line.
    head: Vec2,
    hint: bool,
    /// The logits' chart, and the row of buttons that choose the target under it.
    logits: Rect,
    targets: Rect,
    /// The cells of each pairing's row (its name above, numbers below), and the
    /// height of a value of 1 in them.
    rows: [Rect; 4],
    unit: f32,
    row_head: f32,
    numbers: bool,
    /// The key to the pairings' colours, over the charts.
    legend: Rect,
    loss: Rect,
    slope: Rect,
    /// The race: its heading, the list of how each pairing did, and its chart.
    race_head: Vec2,
    race_list: Rect,
    race: Rect,
    race_button: Rect,
    tabs: [Rect; 3],
    /// The line that says what the rows' colours are (its top-left corner), and, where
    /// there is room under the rows, the softmax's Jacobian.
    key: Vec2,
    jacobian: Rect,
}

/// How far the rows' cells reach either way, in units of the slope.
const CLIP: f64 = 1.15;

impl Layout {
    /// (`lanes`: how many lanes the race has, whose results are listed over it.)
    fn of(view: Vec2, inset: f32, tab: Tab, lanes: usize) -> Layout {
        let view = view.max(Vec2::splat(1.0));
        let r = |x0: f32, y0: f32, x1: f32, y1: f32| (Vec2::new(x0, y0), Vec2::new(x1.max(x0 + 1.0), y1.max(y0 + 1.0)));
        let none = r(0.0, 0.0, 0.0, 0.0);
        let mut lay = Layout {
            view,
            wide: false,
            compact: false,
            big: 13.0,
            text: 11.5,
            small: 10.5,
            head: Vec2::ZERO,
            hint: false,
            logits: none,
            targets: none,
            rows: [none; 4],
            unit: 10.0,
            row_head: 16.0,
            numbers: true,
            legend: none,
            loss: none,
            slope: none,
            race_head: Vec2::ZERO,
            race_list: none,
            race: none,
            race_button: none,
            tabs: [none; 3],
            key: Vec2::ZERO,
            jacobian: none,
        };
        // what a row needs beside its cells: its name, its numbers, the gap to the next
        let wide_top = inset.max(0.0) + 14.0;
        let wide_fixed = 46.0 + 8.0 + 24.0 + 42.0 + 4.0 * (20.0 + 16.0 + 12.0);
        let wide_unit = (view.y - wide_top - 16.0 - wide_fixed) / (4.6 + 4.0 * 2.0 * CLIP as f32);
        if view.x >= 760.0 && wide_unit >= 20.0 {
            lay.wide = true;
            (lay.big, lay.text, lay.small) = (15.0, 12.5, 11.5);
            let (side, top) = (22.0, wide_top);
            let u = wide_unit.min(32.0);
            let (x0, x1) = (side + 30.0, (view.x * 0.53).min(620.0));
            lay.head = Vec2::new(side, top);
            lay.hint = true;
            lay.unit = u;
            lay.row_head = 20.0;
            let mut y = top + 46.0;
            lay.logits = r(x0, y, x1, y + 4.6 * u);
            y = lay.logits.1.y + 8.0;
            lay.targets = r(x0, y, x1, y + 24.0);
            lay.key = Vec2::new(side, lay.targets.1.y + 10.0);
            y = lay.targets.1.y + 22.0 + 20.0;
            for row in lay.rows.iter_mut() {
                y += 20.0;
                *row = r(x0, y, x1, y + 2.0 * CLIP as f32 * u);
                y = row.1.y + 16.0 + 12.0;
            }
            if view.y - 20.0 - y >= 150.0 {
                lay.jacobian = r(side, y + 6.0, x1, view.y - 20.0);
            }
            // the charts, in a column at the right
            let (c0, c1) = (x1 + 66.0, view.x - side - 10.0);
            let fixed = 36.0 + 2.0 * (26.0 + 26.0 + 12.0) + (40.0 + 5.0 * (lay.small + 5.0) + 26.0 + 36.0);
            let free = (view.y - top - 16.0 - fixed).max(120.0);
            let (hc, hr) = (0.3 * free, 0.4 * free);
            lay.legend = r(c0 - 30.0, top, c1, top + 30.0);
            let mut y = top + 36.0 + 26.0;
            lay.loss = r(c0, y, c1, y + hc);
            y = lay.loss.1.y + 26.0 + 12.0 + 26.0;
            lay.slope = r(c0, y, c1, y + hc);
            y = lay.slope.1.y + 26.0 + 12.0;
            lay.race_head = Vec2::new(c0 - 30.0, y);
            let list = 5.0 * (lay.small + 5.0);
            lay.race_list = r(c0 - 30.0, y + 40.0, c1 - 125.0, y + 40.0 + list);
            lay.race = r(c0, lay.race_list.1.y + 26.0, c1, (lay.race_list.1.y + 26.0 + hr).min(view.y - 46.0));
            lay.race_button = r(c1 - 115.0, lay.race_list.0.y, c1, lay.race_list.0.y + 24.0);
            return lay;
        }
        // one column: the logits and the rows, and under them one chart at a time
        let (side, top) = (10.0, inset.max(0.0) + 8.0);
        let avail = view.y - top - 10.0;
        // the heading, the target buttons, and each row's name and gap; the numbers
        // under the rows, where there is room for them; the logits' chart is 4 units
        // high and each row 2.3
        let fixed = 18.0 + 6.0 + 22.0 + 30.0 + 4.0 * (16.0 + 8.0);
        let numbers = 4.0 * 13.0;
        let per_unit = 4.0 + 4.0 * 2.0 * CLIP as f32;
        let charts = 240.0;
        let fit = |fixed: f32, charts: f32| (avail - fixed - charts) / per_unit;
        lay.numbers = fit(fixed + numbers, charts) >= 11.0 || (fit(fixed, charts) < 10.0 && fit(fixed + numbers, 0.0) >= 9.0);
        let fixed = if lay.numbers { fixed + numbers } else { fixed };
        let unit = fit(fixed, charts).min(20.0);
        lay.compact = unit < 10.0 || view.x < 240.0;
        let u = if lay.compact { fit(fixed, 0.0).min(20.0).max(6.0) } else { unit };
        lay.unit = u;
        let (x0, x1) = (side + 24.0, view.x - side);
        lay.head = Vec2::new(side, top);
        let mut y = top + 24.0;
        lay.logits = r(x0, y, x1, y + 4.0 * u);
        y = lay.logits.1.y + 6.0;
        lay.targets = r(x0, y, x1, y + 22.0);
        lay.key = Vec2::new(side, lay.targets.1.y + 8.0);
        y = lay.targets.1.y + 14.0 + 16.0;
        for row in lay.rows.iter_mut() {
            y += 16.0;
            *row = r(x0, y, x1, y + 2.0 * CLIP as f32 * u);
            y = row.1.y + if lay.numbers { 13.0 } else { 0.0 } + 8.0;
            // (in a window too small for them all, the rows that do not fit are left out)
            if row.1.y > view.y - 4.0 {
                *row = none;
            }
        }
        if lay.compact {
            return lay;
        }
        // the tabs, and the chart they choose
        y += 2.0;
        let tabs = plot::even_row(3, side, x1, y, 26.0, 6.0);
        for (t, r) in lay.tabs.iter_mut().zip(tabs) {
            *t = r;
        }
        y += 26.0 + 10.0;
        let bottom = view.y - 10.0 - 30.0;
        match tab {
            Tab::Loss | Tab::Slope => {
                // (the rows' names are in the pairings' colours: a key of its own over
                // the chart only where it leaves the chart room)
                let lines = if x1 - side >= 460.0 { 1.0 } else { 2.0 };
                let key = lines * (lay.small + 5.0);
                let top = if bottom - y - key - 28.0 >= 130.0 {
                    lay.legend = r(side, y, x1, y + key);
                    y + key + 6.0 + 22.0
                } else {
                    y + 22.0
                };
                let chart = r(x0 + 12.0, top, x1 - 6.0, bottom);
                if tab == Tab::Loss {
                    lay.loss = chart;
                } else {
                    lay.slope = chart;
                }
            }
            Tab::Race => {
                lay.race_head = Vec2::new(side, y);
                lay.race_button = r(x1 - 112.0, y - 4.0, x1, y + 18.0);
                // how each lane did, in two columns
                let list = lanes.div_ceil(2).max(1) as f32 * (lay.small + 3.0);
                lay.race_list = r(side, y + 22.0, x1, y + 22.0 + list);
                lay.race = r(x0 + 12.0, lay.race_list.1.y + 20.0, x1 - 6.0, bottom);
            }
        }
        lay
    }

    /// The columns of the classes, across the logits' chart and the rows.
    fn column(&self, i: usize, k: usize) -> (f32, f32) {
        let (a, b) = (self.logits.0.x, self.logits.1.x);
        let w = (b - a) / k.max(1) as f32;
        (a + w * i as f32, a + w * (i + 1) as f32)
    }
}

// ----- colours and kinds of line -----------------------------------------------------------

/// The truth's ticks in the rows, the ends of bars cut off, the legend's swatches.
const TICKS: u8 = OWN;
const CUT: u8 = OWN + 1;
const LEGEND: u8 = OWN + 2;
/// Where the target's logit passes the largest other, and the rows' lines at ±1.
const EDGES: u8 = OWN + 3;
/// The race: its marks where each lane comes right, and the line where softmax's
/// slope is 0 against a softened truth.
const RIGHTS: u8 = OWN + 4;
const PLATEAU: u8 = OWN + 5;
/// Each pairing's loss curve, its slope curve, and each lane of the race (each its own
/// kind: a curve that leaves its chart is cut there, and grows and loses points).
const LOSS_CURVES: u8 = OWN + 6;
const SLOPE_CURVES: u8 = OWN + 10;
const LANES: u8 = OWN + 14;

/// The colour of a lane of the race.
fn lane_color(race: &Race, lane: usize) -> Color {
    match race.lanes.get(lane) {
        Some(l) if lane < PAIRINGS.len() => l.pairing.color(),
        _ => pal::WHITE.a(0.5),
    }
}

/// Every kind of line and point this scene draws, asked for in the same order at the
/// start of every frame, each with one unseen point: a batch not drawn in a frame would
/// drop out and move every later one up a place, and `eidos motion` would then take one
/// batch for another.
fn hold_batches(f: &mut Frame) {
    let corner = f.hud(Vec2::ZERO);
    let mut kinds = vec![AXES, CURVES, MARKS, plot::BUTTONS, TICKS, CUT, LEGEND, EDGES, RIGHTS, PLATEAU];
    kinds.extend((0..4).map(|j| LOSS_CURVES + j));
    kinds.extend((0..4).map(|j| SLOPE_CURVES + j));
    kinds.extend((0..5).map(|j| LANES + j));
    for kind in kinds {
        plot::apart(f, kind, false).segment(corner, corner, 1.0, Color::CLEAR);
    }
    for kind in [EDGES, PLATEAU] {
        plot::apart(f, kind, true).segment(corner, corner, 1.0, Color::CLEAR);
    }
    plot::disc(f, Vec2::ZERO, 1.0, Color::CLEAR);
}

/// A dashed polyline in screen space, kept inside a rectangle: as solid dashes in the
/// batch of its kind, the pattern measured along the line (a dashed batch starts its
/// pattern afresh at every segment, and a curve of short segments comes out solid).
fn dashed_path(f: &mut Frame, kind: u8, pts: &[Vec2], rect: Rect, width: f32, color: Color) {
    let cut = plot::clip(pts, rect);
    let view = f.view_size();
    let points: Vec<(Vec3, Option<Vec2>)> = cut.iter().map(|p| (f.hud(*p), p.is_finite().then_some(*p))).collect();
    plot::apart(f, kind, false).dashes(points, 9.0, 5.0, view, width, color);
}

impl Scene for LossGrad {
    fn view(&self) -> View {
        View::Flat { center: Vec2::ZERO, half_height: 1.0 }
    }

    fn settings(&self, s: &mut ViewSettings) {
        s.grid = false;
        s.bloom = s.bloom.min(0.2);
    }

    fn animated(&self) -> bool {
        self.look.moving || self.playing || !self.race.finished() || self.dragging.is_some()
    }

    fn pointer_hint(&self) -> Option<&'static str> {
        Some("dots on the logits: drag them up and down    buttons over the picture: click")
    }

    fn ui(&mut self, ui: &mut egui::Ui, _: &Gfx) {
        ui::note(
            ui,
            r"The backward sweep of a network starts at the loss. The first thing it hands back is the loss's slope with respect to the network's raw scores, its logits $s_k$, one for each class: every gradient further back is this one carried back by the chain rule, and grows or shrinks with it. The slope depends on a pairing: the output function that turns the logits into a prediction, and the loss that scores the prediction against the truth $y$. For softmax with cross-entropy, the usual choice for classes, it is the probabilities minus the truth:",
        );
        ui::equation(ui, r"\frac{\partial L}{\partial s_k} = p_k - y_k");
        ui::note(ui, r"Four pairings are worked out here side by side for the same logits, each with its own slope. Logarithms are natural ($\ln$).");

        ui::section(ui, "Things to try");
        for (k, trial) in TRIALS.iter().enumerate() {
            if ui::experiment(ui, trial.button, trial.caption) {
                self.set_up(k);
            }
        }

        ui::section(ui, "The logits and the truth");
        let mut k = self.set.k;
        if ui::slider_int(ui, "classes $K$", &mut k, FEWEST..=MOST) {
            self.set.k = k;
        }
        let names: Vec<String> = (1..=self.set.k).map(|i| format!("class {i}")).collect();
        let names: Vec<&str> = names.iter().map(String::as_str).collect();
        let mut target = self.set.target;
        if ui::choice(ui, &names, &mut target) {
            self.set.target = target;
        }
        ui::note(ui, "The target class: the right answer for this example.");
        let mut temp = self.set.temp as f32;
        if ui::slider_log(ui, "temperature $T$", &mut temp, TEMP_LO as f32..=TEMP_HI as f32) {
            self.set.temp = temp as f64;
        }
        ui::note(ui, r"The softmax's input is $s/T$ (only the first row has a softmax): above 1 the probabilities flatten, below 1 they sharpen.");
        let mut smooth = self.set.smooth as f32;
        if ui::slider(ui, r"label smoothing $\varepsilon$", &mut smooth, 0.0..=SMOOTH_MOST as f32) {
            self.set.smooth = smooth as f64;
        }
        ui::note(ui, r"The truth becomes $1 - \varepsilon$ for the target and $\varepsilon/(K - 1)$ for each other class, in every row.");
        if ui.button(ui::label_text(ui, "all logits to 0", 12.5, None, false)).clicked() {
            self.set.s = [0.0; MOST];
        }

        ui::section(ui, "The race");
        let mut eta = self.eta as f32;
        if ui::slider_log(ui, r"step size $\eta$", &mut eta, ETA_LO as f32..=ETA_HI as f32) {
            self.eta = eta as f64;
        }
        ui::note(ui, r"Each step is $s \leftarrow s - \eta\,\partial L/\partial s$, on the logits themselves (as if the network could set them freely), with the same $\eta$ and the same start for every pairing.");
        let label = if self.race_is_current() { "race again" } else { "race from these logits" };
        if ui.button(ui::label_text(ui, label, 12.5, None, false)).clicked() {
            self.start_race();
        }
        self.set = self.set.kept();

        ui::section(ui, "Reading the picture");
        ui::key(
            ui,
            &[
                (FORWARD, r"cyan: forward. The logits $s_k$ (drag the dots), and in each row what its output function makes of them"),
                (LOSS, r"orange: the truth $y_k$, a tick at its height in each row; the target's column; each pairing's loss $L$"),
                (BACKWARD, r"pink: the slope $\partial L/\partial s_k$ on each logit, the first gradient of the backward sweep"),
                (Pairing::SoftmaxCe.color(), "white curves: softmax + cross-entropy"),
                (Pairing::SigmoidSe.color(), "red: sigmoid + squared error"),
                (Pairing::SigmoidCe.color(), "blue: sigmoid + cross-entropy"),
                (Pairing::LinearSe.color(), "grey: linear output + squared error"),
                (pal::DIM, "dashed upright line in the charts: where the target's logit passes the largest other; to its left the answer is wrong"),
            ],
        );
        ui::section(ui, "Words");
        ui::term(ui, "Logit:", "the network's raw score for one class, any number, before an output function turns the scores into a prediction.");
        ui::term(ui, "Softmax:", r"$p_k = e^{s_k} / \sum_j e^{s_j}$: positive numbers that add up to 1, the largest logit taking the largest share.");
        ui::term(ui, "Sigmoid:", r"$\sigma(s) = 1/(1 + e^{-s})$, from 0 to 1, for each logit on its own (the outputs need not add up to 1). Its slope is $\sigma(1 - \sigma)$, at most $1/4$.");
        ui::term(
            ui,
            "Cross-entropy (CE):",
            r"$L = -\sum_k y_k \ln p_k$; with the truth 1 for the target and 0 elsewhere, $-\ln$ of the target's probability. For sigmoids each class is a yes-or-no question: $-\sum_k [y_k \ln \sigma_k + (1 - y_k) \ln(1 - \sigma_k)]$.",
        );
        ui::term(ui, "Squared error (SE):", r"$L = \tfrac12 \sum_k (\text{output}_k - y_k)^2$.");
        ui::term(ui, "Saturated:", "a sigmoid far from its middle is nearly flat, so nearly nothing passes back through it.");
        ui::term(ui, "Gap:", r"the target's logit less the largest other, $s_t - \max_{k \ne t} s_k$: above 0, the network's answer is the target.");
        ui::term(ui, "Label smoothing:", r"training against a softened truth, $1 - \varepsilon$ and $\varepsilon/(K - 1)$, which softmax can reach at finite logits.");

        ui::section(ui, "Numbers");
        let set = self.set;
        for p in PAIRINGS {
            ui::readout_series(ui, p.name(), format!("$L = {}$", num(set.loss(p), 3)), p.color());
        }
        let g = set.slope(Pairing::SoftmaxCe);
        ui::readout(ui, "softmax slopes", kit_tuple(&g[..set.k]));
        ui::readout(ui, "their sum", num(g[..set.k].iter().sum::<f64>(), 3));
        ui::readout(ui, r"gap $s_t - \max_{k \ne t} s_k$", num(set.gap(), 2));
        if let Some(d) = reachable_gap(set.k, set.smooth, set.temp) {
            ui::readout(ui, "gap where softmax's slope is 0", num(d, 2));
        }
        ui::note(ui, r"The softmax's own Jacobian, how each probability moves with each logit. Its rows add up to 0, and the slope $\partial L/\partial s$ is this matrix times the slope of $L = -\sum_k y_k \ln p_k$ with respect to each $p_k$, which is $-y_k/p_k$:");
        let p = set.outputs(Pairing::SoftmaxCe);
        let j = softmax_jacobian(&p[..set.k], set.temp);
        let rows: Vec<Vec<f64>> = (0..set.k).map(|a| j[a][..set.k].to_vec()).collect();
        let lhs = if (set.temp - 1.0).abs() > 1e-9 { r"\frac{\partial p}{\partial s} = \frac{\operatorname{diag}(p) - p\,p^\top}{T}" } else { r"\frac{\partial p}{\partial s} = \operatorname{diag}(p) - p\,p^\top" };
        ui::equation(ui, &format!("{lhs} = {}", bmatrix(&rows, 2)));
        if self.race.finished() {
            for (lane, l) in self.race.lanes.iter().enumerate().take(PAIRINGS.len()) {
                let text = match l.right {
                    Some(n) => format!("right after {}", steps_text(n)),
                    None => format!("not right after {}", steps_text(RACE_STEPS)),
                };
                ui::readout_colored(ui, l.pairing.name(), text, lane_color(&self.race, lane));
            }
        }
        ui::see_also(
            ui,
            &[
                ("backpropmat", "what the network does with this slope, a layer at a time"),
                ("actgrad", "the next gates back: activation functions"),
                ("attngrad", "the softmax's Jacobian again, inside attention"),
                ("entropy", "entropy and cross-entropy"),
            ],
        );
    }

    fn frame(&mut self, f: &mut Frame) {
        hold_batches(f);
        let lay = Layout::of(f.view_size(), f.top_inset, self.tab, self.race.lanes.len());
        // (a canvas squeezed to a sliver is one the controls have opened over, on a phone)
        let tiny = lay.view.x < 160.0 || lay.view.y < 160.0;
        self.dragging = None;
        if !tiny {
            self.buttons(f, &lay);
            self.handles(f, &lay);
        }
        self.set = self.set.kept();
        self.work(f.dt);
        let snap = self.canvas != Some(lay.view) || self.look.state.shown.len() != LOOK_LEN;
        if snap && !tiny {
            self.canvas = Some(lay.view);
        }
        self.follow(f.real_dt, snap);
        if tiny {
            return;
        }
        self.draw_logits(f, &lay);
        draw_key(f, &lay);
        self.draw_rows(f, &lay);
        self.draw_jacobian(f, &lay);
        if !lay.compact {
            self.draw_margins(f, &lay);
            self.draw_race(f, &lay);
        }
    }
}

/// Numbers in round brackets, as the panel writes them (to three decimals).
fn kit_tuple(v: &[f64]) -> String {
    let parts: Vec<String> = v.iter().map(|x| num(*x, 3)).collect();
    format!("({})", parts.join(", "))
}

impl LossGrad {
    /// The buttons over the picture: the target, the race, and (narrow) the tabs.
    fn buttons(&mut self, f: &mut Frame, lay: &Layout) {
        let k = self.set.k;
        let (min, max) = lay.targets;
        for i in 0..k {
            let (a, b) = lay.column(i, k);
            let wide_enough = b - a >= 64.0;
            let text = if wide_enough { format!("class {}", i + 1) } else { format!("{}", i + 1) };
            let rect = (Vec2::new(a + 3.0, min.y), Vec2::new(b - 3.0, max.y));
            if plot::button(f, rect, &text, lay.small, self.set.target == i) {
                self.set.target = i;
            }
        }
        if !lay.compact {
            if lay.race_button.1.x - lay.race_button.0.x > 4.0 {
                let current = self.race_is_current();
                let text = if current { "race again" } else { "race these logits" };
                if plot::button(f, lay.race_button, text, lay.small, !current) {
                    self.start_race();
                }
            }
            if !lay.wide {
                for (rect, (tab, name)) in lay.tabs.iter().zip(TABS) {
                    if plot::button(f, *rect, name, lay.small, self.tab == tab) {
                        self.tab = tab;
                    }
                }
            }
        }
    }

    /// The logits' dots, dragged up and down.
    fn handles(&mut self, f: &mut Frame, lay: &Layout) {
        let k = self.set.k;
        let chart = Plot::new(lay.logits, [0.0, k as f64], [-LOGIT_MOST, LOGIT_MOST]);
        if chart.tiny() {
            return;
        }
        for i in 0..k {
            let shown = self.look.state.get(LOOK_S + i) as f64;
            if let Some((_, y)) = chart.handle(f, 1 + i as u32, i as f64 + 0.5, shown, FORWARD) {
                if y.is_finite() {
                    self.set.s[i] = y.max(-LOGIT_MOST).min(LOGIT_MOST);
                    self.dragging = Some(i);
                }
            }
        }
    }

    /// The heading, the logits as bars, and the target's column.
    fn draw_logits(&self, f: &mut Frame, lay: &Layout) {
        let k = self.set.k;
        let (shown, _) = shown_setting(&self.set, &self.look.state);
        let chart = Plot::new(lay.logits, [0.0, k as f64], [-LOGIT_MOST, LOGIT_MOST]);
        // the heading
        let head = if lay.wide { r"the logits $s_k$: the network's raw scores, one for each class" } else { r"the logits $s_k$, one for each class" };
        f.text_px(lay.head, head, if lay.wide { lay.big } else { lay.text }, pal::TEXT, Align2::LEFT_TOP);
        if lay.hint {
            f.text_px(lay.head + Vec2::new(0.0, lay.big + 7.0), "drag a dot to change a logit; the orange column is the target (click a class to choose another)", lay.small, pal::DIM, Align2::LEFT_TOP);
        }
        if chart.tiny() {
            return;
        }
        // the target's column, from the logits down through every row
        let t = self.look.state.get(LOOK_TARGET);
        let w = (lay.logits.1.x - lay.logits.0.x) / k as f32;
        let x = lay.logits.0.x + w * t;
        let last = lay.rows.iter().rev().find(|(a, b)| b.y - a.y > 2.0).map_or(lay.logits.1.y, |r| r.1.y);
        let bottom = last + if lay.numbers { 14.0 } else { 2.0 };
        plot::fill(f, (Vec2::new(x + 1.0, lay.logits.0.y - 4.0), Vec2::new(x + w - 1.0, bottom)), LOSS.a(0.025));

        chart.backing(f, 30.0, 8.0, 8.0, 6.0);
        chart.y_axis(f, "", lay.small);
        plot::line_of(f, AXES, Vec2::new(lay.logits.0.x, chart.px_y(0.0)), Vec2::new(lay.logits.1.x, chart.px_y(0.0)), 1.0, pal::DIM.a(0.7));
        let decimals = if lay.wide { 2 } else { 1 };
        for i in 0..k {
            let s = shown.s[i];
            chart.bar(f, i as f64 + 0.32, i as f64 + 0.68, 0.0, s, 0.0, FORWARD.a(0.45));
            // the value beside the dot, on the side away from the bar
            let at = chart.at(i as f64 + 0.5, s);
            let size = lay.small;
            let text = if lay.wide { format!("$s_{} = {}$", i + 1, num(s, decimals)) } else { format!("${}$", num(s, decimals)) };
            let dx = w * 0.18 + 6.0;
            f.text_px(Vec2::new(at.x + dx, at.y), text, size, FORWARD.mix(pal::WHITE, 0.3), Align2::LEFT_CENTER);
        }
    }

    /// Each pairing's row: its outputs, the truth, and the slope on each logit.
    fn draw_rows(&self, f: &mut Frame, lay: &Layout) {
        let k = self.set.k;
        let (shown, y) = shown_setting(&self.set, &self.look.state);
        let u = lay.unit;
        let small = lay.small;
        for (j, p) in PAIRINGS.iter().enumerate() {
            let (min, max) = lay.rows[j];
            if max.y - min.y < 4.0 {
                continue;
            }
            let mid = (min.y + max.y) * 0.5;
            let at = |v: f64| mid - (v.max(-CLIP).min(CLIP) as f32) * u;
            // the row's name, its slope, and its loss
            let name_y = min.y - lay.row_head + 1.0;
            let name = if lay.view.x < 330.0 { p.short() } else { p.name() };
            f.text_px(Vec2::new(lay.head.x, name_y), name, lay.text, p.color(), Align2::LEFT_TOP);
            let w_name = f.text_size(name, lay.text).x;
            let formula = format!("slope {}", p.slope_formula(shown.temp));
            let loss = format!("$L = {}$", num(p.loss(&shown.s[..k], &y[..k], shown.temp), 2));
            let w_loss = f.text_size(&loss, small).x;
            let room = max.x - (lay.head.x + w_name + 12.0) - w_loss - 12.0;
            if f.text_size(&formula, small).x <= room {
                f.text_px(Vec2::new(lay.head.x + w_name + 12.0, name_y + 1.0), formula, small, pal::DIM, Align2::LEFT_TOP);
            }
            f.text_px(Vec2::new(max.x, name_y + 1.0), loss, small, LOSS, Align2::RIGHT_TOP);
            // the scale: lines at 1 and −1, and 0
            plot::line_of(f, AXES, Vec2::new(min.x, mid), Vec2::new(max.x, mid), 1.0, pal::DIM.a(0.55));
            for v in [1.0, -1.0] {
                plot::dashed_of(f, EDGES, Vec2::new(min.x, at(v)), Vec2::new(max.x, at(v)), 1.0, pal::DIM.a(0.25));
            }
            if u >= 9.0 {
                for (v, name) in [(1.0, "1"), (0.0, "0"), (-1.0, "−1")] {
                    f.text_px(Vec2::new(min.x - 5.0, at(v)), name, small - 1.0, pal::DIM, Align2::RIGHT_CENTER);
                }
            }
            let out = p.outputs(&shown.s[..k], shown.temp);
            let g = p.slope(&shown.s[..k], &y[..k], shown.temp);
            for i in 0..k {
                let (a, b) = lay.column(i, k);
                let w = b - a;
                let (o0, o1) = (a + 0.12 * w, a + 0.44 * w);
                let (g0, g1) = (a + 0.56 * w, a + 0.88 * w);
                // the output, and the truth's tick across it
                bar_px(f, o0, o1, mid, at(out[i]), FORWARD.a(0.5));
                plot::line_of(f, TICKS, Vec2::new(a + 0.06 * w, at(y[i])), Vec2::new(a + 0.5 * w, at(y[i])), 2.2, LOSS);
                // the slope
                bar_px(f, g0, g1, mid, at(g[i]), BACKWARD.a(0.85));
                for (v, x0, x1, color) in [(out[i], o0, o1, FORWARD), (g[i], g0, g1, BACKWARD)] {
                    if v.abs() > CLIP {
                        // cut off: a chevron at the end says it goes on
                        let end = at(v);
                        let d = if v > 0.0 { 1.0 } else { -1.0 };
                        let c = (x0 + x1) * 0.5;
                        let half = ((x1 - x0) * 0.35).min(6.0);
                        plot::line_of(f, CUT, Vec2::new(c - half, end + d * half), Vec2::new(c, end), 1.6, color.mix(pal::WHITE, 0.3));
                        plot::line_of(f, CUT, Vec2::new(c + half, end + d * half), Vec2::new(c, end), 1.6, color.mix(pal::WHITE, 0.3));
                    }
                }
                if lay.numbers {
                    let ny = max.y + 3.0;
                    f.text_px(Vec2::new((o0 + o1) * 0.5, ny), brief(out[i]), small - 1.0, FORWARD.mix(pal::WHITE, 0.25), Align2::CENTER_TOP);
                    f.text_px(Vec2::new((g0 + g1) * 0.5, ny), brief(g[i]), small - 1.0, BACKWARD.mix(pal::WHITE, 0.3), Align2::CENTER_TOP);
                }
            }
        }
    }

    /// The loss and the size of the slope on the target's logit, against that logit.
    fn draw_margins(&self, f: &mut Frame, lay: &Layout) {
        let (shown, y) = shown_setting(&self.set, &self.look.state);
        let t = shown.target;
        let k = shown.k;
        let others = shown.s[..k].iter().enumerate().filter(|(i, _)| *i != t).map(|(_, v)| *v).fold(f64::NEG_INFINITY, f64::max);
        let now = shown.s[t];
        let r = &self.look.ranges;
        // the key to the colours
        let (lmin, lmax) = lay.legend;
        if lmax.x - lmin.x > 40.0 {
            let per_line = if lmax.y - lmin.y < 1.5 * (lay.small + 5.0) { 4 } else { 2 };
            let col = (lmax.x - lmin.x) / per_line as f32;
            for (j, p) in PAIRINGS.iter().enumerate() {
                let x = lmin.x + col * (j % per_line) as f32;
                let y = lmin.y + (j / per_line) as f32 * (lay.small + 5.0) + lay.small * 0.5;
                plot::line_of(f, LEGEND, Vec2::new(x, y), Vec2::new(x + 18.0, y), 2.4, p.color());
                let name = if lay.wide { p.name() } else { p.short() };
                f.text_px(Vec2::new(x + 24.0, y), name, lay.small, pal::TEXT, Align2::LEFT_CENTER);
            }
        }
        let charts = [
            (lay.loss, format!(r"the loss $L$ as the target's logit $s_{}$ moves, the others staying", t + 1), r.get(RANGE_LOSS) as f64, true),
            (lay.slope, format!(r"how hard the target's logit is pulled: $|\partial L/\partial s_{}|$", t + 1), r.get(RANGE_SLOPE) as f64, false),
        ];
        for (rect, title, top, is_loss) in charts {
            let chart = Plot::new(rect, [-LOGIT_MOST, LOGIT_MOST], [0.0, top.max(0.1)]);
            if chart.tiny() {
                continue;
            }
            chart.backing(f, 38.0, 30.0, 30.0, 10.0);
            let narrow_title = if is_loss { format!(r"the loss $L$ against $s_{}$", t + 1) } else { format!(r"the pull $|\partial L/\partial s_{}|$ against $s_{}$", t + 1, t + 1) };
            let title = if f.text_size(&title, lay.text).x <= rect.1.x - rect.0.x + 30.0 { title } else { narrow_title };
            f.text_px(Vec2::new(rect.0.x - 30.0, rect.0.y - 8.0), title, lay.text, pal::TEXT, Align2::LEFT_BOTTOM);
            chart.y_grid(f);
            chart.y_axis(f, "", lay.small);
            chart.x_axis(f, "", lay.small);
            f.text_px(Vec2::new(rect.1.x, rect.1.y + 6.0 + lay.small + 4.0), format!("$s_{}$", t + 1), lay.small + 1.0, pal::TEXT, Align2::RIGHT_TOP);
            // where the answer turns right, and where the target's logit is now
            if others.is_finite() && others > -LOGIT_MOST && others < LOGIT_MOST {
                let x = chart.px_x(others);
                plot::dashed_of(f, EDGES, Vec2::new(x, rect.0.y), Vec2::new(x, rect.1.y), 1.2, pal::DIM.a(0.8));
                if is_loss {
                    let y = rect.0.y + 4.0;
                    f.text_px(Vec2::new(x - 5.0, y), "wrong", lay.small, pal::DIM, Align2::RIGHT_TOP);
                    f.text_px(Vec2::new(x + 5.0, y), "right", lay.small, pal::DIM, Align2::LEFT_TOP);
                }
            }
            plot::line_of(f, MARKS, Vec2::new(chart.px_x(now), rect.0.y), Vec2::new(chart.px_x(now), rect.1.y), 1.4, FORWARD.a(0.7));
            for (j, p) in PAIRINGS.iter().enumerate().rev() {
                let values = if is_loss { &self.curves.loss[j] } else { &self.curves.slope[j] };
                if values.len() != CURVE_N {
                    continue;
                }
                let kind = if is_loss { LOSS_CURVES } else { SLOPE_CURVES } + j as u8;
                pinned_curve(f, kind, &chart, (0..CURVE_N).map(|i| (curve_x(i), values[i])), 2.2, p.color());
                // the pairing's point now (in every frame, unseen when off the chart)
                let v = if is_loss { p.loss(&shown.s[..k], &y[..k], shown.temp) } else { p.slope(&shown.s[..k], &y[..k], shown.temp)[t].abs() };
                let at = chart.at(now, v);
                let inside = plot::within(at, rect, 0.5);
                plot::disc(f, at.max(rect.0).min(rect.1), 7.0, if inside { p.color() } else { Color::CLEAR });
            }
        }
    }

    /// The race: each lane's gap against the steps, on a logarithmic clock, as far as
    /// the film has got.
    fn draw_race(&self, f: &mut Frame, lay: &Layout) {
        let chart = Plot::new(lay.race, [0.0, film_end() as f64], [self.look.ranges.get(RANGE_LO) as f64, self.look.ranges.get(RANGE_HI) as f64]);
        if chart.tiny() {
            return;
        }
        let race = &self.race;
        let from = &race.from;
        let t = from.target;
        // the heading
        let logits: Vec<String> = from.s[..from.k].iter().map(|v| plain(*v)).collect();
        let start = format!(r"from $s = ({})$, $\eta = {}$", logits.join(", "), plain(race.eta));
        if lay.wide {
            f.text_px(lay.race_head, r"the race: plain gradient steps on the logits, $s \leftarrow s - \eta\,\partial L/\partial s$", lay.text, pal::TEXT, Align2::LEFT_TOP);
            f.text_px(lay.race_head + Vec2::new(0.0, lay.text + 6.0), format!("{start}, the same start and step for every pairing"), lay.small, pal::DIM, Align2::LEFT_TOP);
        } else {
            f.text_px(lay.race_head, format!("gradient steps {start}"), lay.small, pal::TEXT, Align2::LEFT_TOP);
        }
        // how each did, so far as the film has shown
        let n = self.film_step();
        let line = lay.small + if lay.wide { 5.0 } else { 3.0 };
        let (columns, col_w) = if lay.wide { (1, 0.0) } else { (2, (lay.race_list.1.x - lay.race_list.0.x) * 0.5) };
        for (lane, l) in race.lanes.iter().enumerate() {
            let y = lay.race_list.0.y + (lane / columns) as f32 * line;
            let x = lay.race_list.0.x + (lane % columns) as f32 * col_w;
            let name = if lay.wide || lane >= PAIRINGS.len() { l.pairing.name() } else { l.pairing.short() };
            let name = match (lane >= PAIRINGS.len(), lay.wide) {
                (true, true) => format!(r"{name}, $\varepsilon = 0$ (dashed)"),
                (true, false) => r"softmax, $\varepsilon = 0$".to_string(),
                _ => name.to_string(),
            };
            let shown = l.right.filter(|r| (*r as f64) <= n + 1e-9);
            let over = self.film >= film_end() && race.finished();
            let verdict = match (shown, over, lay.wide) {
                (Some(r), _, true) => format!("right after {}", steps_text(r)),
                (Some(r), _, false) => steps_text(r),
                (None, true, true) => format!("not right after {}", steps_text(RACE_STEPS)),
                (None, true, false) => "not right".to_string(),
                (None, false, _) => "…".to_string(),
            };
            plot::line_of(f, LEGEND, Vec2::new(x, y + lay.small * 0.5), Vec2::new(x + 14.0, y + lay.small * 0.5), 2.4, lane_color(race, lane));
            f.text_px(Vec2::new(x + 20.0, y), format!("{name}: {verdict}"), lay.small, pal::TEXT, Align2::LEFT_TOP);
        }
        chart.backing(f, 38.0, 34.0, 8.0, 10.0);
        // the axes: the gap up the side, the steps along the bottom on the clock
        chart.y_grid(f);
        chart.y_axis(f, "", lay.small);
        let (min, max) = chart.rect;
        plot::line_of(f, AXES, Vec2::new(min.x, max.y), max, 1.0, pal::DIM.a(0.8));
        for (n, name) in [(0usize, "0"), (1, "1"), (10, "10"), (100, "100"), (1000, "1000"), (10_000, "10 000")] {
            let x = chart.px_x(clock(n as f64));
            plot::line_of(f, AXES, Vec2::new(x, max.y), Vec2::new(x, max.y + 4.0), 1.0, pal::DIM.a(0.8));
            if n != 1 || chart.width() > 260.0 {
                f.text_px(Vec2::new(x, max.y + 6.0), name, lay.small, pal::DIM, Align2::CENTER_TOP);
            }
        }
        f.text_px(Vec2::new(max.x, max.y + 8.0 + lay.small), "steps (on a logarithmic scale)", lay.small, pal::DIM, Align2::RIGHT_TOP);
        f.text_px(Vec2::new(min.x - 30.0, min.y - 4.0), format!(r"the gap $s_{} - \max_{{k \ne {}}} s_k$: above 0 the answer is right", t + 1, t + 1), lay.small, pal::DIM, Align2::LEFT_BOTTOM);
        // 0: right above, wrong below
        let zero = chart.px_y(0.0);
        plot::line_of(f, MARKS, Vec2::new(min.x, zero), Vec2::new(max.x, zero), 1.4, pal::DIM);
        // where softmax's slope is 0, against a softened truth
        if let Some(d) = reachable_gap(from.k, from.smooth, from.temp) {
            let y = chart.px_y(d);
            if y >= min.y && y <= max.y {
                plot::dashed_of(f, PLATEAU, Vec2::new(min.x, y), Vec2::new(max.x, y), 1.2, pal::WHITE.a(0.6));
                f.text_px(Vec2::new(max.x - 4.0, y - 3.0), format!(r"softmax: slope 0 at {}, $p = y$", num(d, 2)), lay.small, pal::WHITE.a(0.8), Align2::RIGHT_BOTTOM);
            }
        }
        // the lanes, as far as the film has got
        let samples = race_samples();
        for lane in (0..race.lanes.len()).rev() {
            let mut pts: Vec<Vec2> = samples.iter().take_while(|&&s| (s as f64) < n).map(|&s| chart.at(clock(s as f64), race.gap_at(lane, s as f64))).collect();
            pts.push(chart.at(clock(n), race.gap_at(lane, n)));
            let color = lane_color(race, lane);
            let kind = LANES + lane as u8;
            if lane >= PAIRINGS.len() {
                dashed_path(f, kind, &pts, chart.rect, 1.8, color);
            } else {
                plot::path_of(f, kind, &plot::clip(&pts, chart.rect), 2.2, color);
            }
            // where it came right
            if let Some(r) = race.lanes[lane].right.filter(|r| (*r as f64) <= n + 1e-9 && lane < PAIRINGS.len()) {
                let x = chart.px_x(clock(r as f64));
                plot::line_of(f, RIGHTS, Vec2::new(x, zero - 6.0), Vec2::new(x, zero + 6.0), 2.0, color);
            }
            // the head of the lane (in every frame, unseen when off the chart)
            let head = chart.at(clock(n), race.gap_at(lane, n));
            let inside = plot::within(head, chart.rect, 0.5);
            plot::disc(f, head.max(min).min(max), 7.0, if inside { color } else { Color::CLEAR });
        }
    }
}

/// A curve through points in a chart's units, kept inside the chart without being cut
/// at its edge: a point beyond the top or the bottom is drawn on that edge, unseen
/// unless it is next to a point inside. Every point is drawn in every frame, so that
/// the curve keeps its number of points. (Cut where it crosses the edge, a curve that
/// moves has its end sliding along the edge, faster and slower as it passes each of
/// its points, and gains and loses points: `eidos motion` sees the batch now and not
/// the next frame, and the picture's speed seems to jump to and fro.)
fn pinned_curve(f: &mut Frame, kind: u8, chart: &Plot, points: impl Iterator<Item = (f64, f64)>, width: f32, color: Color) {
    let (lo, hi) = (chart.y[0], chart.y[1]);
    let pts: Vec<(Vec3, bool)> = points
        .map(|(x, y)| {
            let inside = y >= lo && y <= hi;
            let kept = if y.is_nan() { hi } else { y.max(lo).min(hi) };
            (f.hud(chart.at(x, kept)), inside)
        })
        .collect();
    let batch = plot::apart(f, kind, false);
    batch.brk();
    for (i, (p, inside)) in pts.iter().enumerate() {
        let near = *inside || (i > 0 && pts[i - 1].1) || pts.get(i + 1).is_some_and(|q| q.1);
        batch.push(*p, width, if near { color } else { color.a(0.0) });
    }
    batch.brk();
}

/// What the rows' colours are, in a line over them.
fn draw_key(f: &mut Frame, lay: &Layout) {
    let entries: [(Color, &str); 3] = if lay.wide {
        [
            (FORWARD, "the output"),
            (LOSS, r"the truth $y_k$"),
            (BACKWARD, r"the slope $\partial L/\partial s_k$: the first gradient handed back"),
        ]
    } else {
        [(FORWARD, "output"), (LOSS, r"truth $y_k$"), (BACKWARD, r"slope $\partial L/\partial s_k$")]
    };
    let size = lay.small;
    // (no key for rows that are not there: a window too small for any)
    let (min, max) = lay.rows[0];
    if max.y - min.y <= 2.0 {
        return;
    }
    let mut x = lay.key.x;
    for (color, text) in entries {
        if x + size + f.text_size(text, size).x > lay.logits.1.x + 4.0 {
            break;
        }
        plot::fill(f, (Vec2::new(x, lay.key.y + 1.0), Vec2::new(x + size - 2.0, lay.key.y + size - 1.0)), color);
        f.text_px(Vec2::new(x + size + 4.0, lay.key.y + size * 0.5), text, size, pal::DIM.mix(pal::TEXT, 0.4), Align2::LEFT_CENTER);
        x += size + 4.0 + f.text_size(text, size).x + 16.0;
    }
}

/// Text broken into lines no wider than `width`, from `at` down, in logical pixels of
/// the screen (`kit::wrapped` takes the plane's own units). Returns the height taken.
fn wrapped_px(f: &mut Frame, at: Vec2, width: f32, text: &str, size: f32, color: Color) -> f32 {
    let ppu = pixels_per_unit(f).max(1e-6);
    let Some(world) = f.cam.pick_ground(at * f.px) else { return 0.0 };
    let w = width / ppu;
    wrapped(f, world.truncate(), w, text, size, color);
    wrapped_height(f, w, text, size)
}

impl LossGrad {
    /// Under the rows, where there is room: the softmax's own Jacobian, with its numbers.
    fn draw_jacobian(&self, f: &mut Frame, lay: &Layout) {
        let (min, max) = lay.jacobian;
        if max.y - min.y < 100.0 || max.x - min.x < 300.0 {
            return;
        }
        let (shown, _) = shown_setting(&self.set, &self.look.state);
        let k = shown.k;
        let p = shown.outputs(Pairing::SoftmaxCe);
        let j = softmax_jacobian(&p[..k], shown.temp);
        let head = if (shown.temp - 1.0).abs() > 1e-6 {
            r"inside the first row: the softmax's own Jacobian, $\partial p/\partial s = (\operatorname{diag}(p) - p\,p^\top)/T$"
        } else {
            r"inside the first row: the softmax's own Jacobian, $\partial p/\partial s = \operatorname{diag}(p) - p\,p^\top$"
        };
        f.text_px(min, head, lay.text, pal::TEXT, Align2::LEFT_TOP);
        let size = lay.small;
        let rows: Vec<Vec<(String, Color)>> = (0..k).map(|a| (0..k).map(|b| (num(j[a][b], 2), if a == b { pal::TEXT } else { pal::DIM.mix(pal::TEXT, 0.5) })).collect()).collect();
        let top = min.y + lay.text + 14.0;
        let total = hud_matrix(f, Vec2::new(min.x + 8.0, top), &rows, 5, size, pal::DIM);
        let x = min.x + 8.0 + total.x + 18.0;
        let note = r"How each probability moves as each logit does. Its rows add up to 0 (raising every logit alike moves nothing), and the slope $\partial L/\partial s$ is this matrix times the slope of $L = -\sum_k y_k \ln p_k$ with respect to each $p_k$, which is $-y_k/p_k$: the chain rule, once.";
        wrapped_px(f, Vec2::new(x, top), (max.x - x).max(80.0), note, size, pal::DIM);
    }
}

/// A bar in screen space from `base` to `top` (either way up), between `x0` and `x1`.
fn bar_px(f: &mut Frame, x0: f32, x1: f32, base: f32, top: f32, color: Color) {
    let (a, b) = (base.min(top), base.max(top));
    if b - a > 0.05 && x1 > x0 {
        plot::fill(f, (Vec2::new(x0, a), Vec2::new(x1, b)), color);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn close(a: f64, b: f64, tol: f64) -> bool {
        (a - b).abs() <= tol
    }

    /// An experiment as its button sets it up, with its race run to the end.
    fn run(k: usize) -> LossGrad {
        let mut s = LossGrad::create();
        s.set_up(k);
        s.race.finish();
        s
    }

    /// Settings of every shape the controls reach, from a seed.
    fn random_setting(rng: &mut Rng) -> Setting {
        let k = FEWEST + (rng.range(0.0, 2.999) as usize);
        let mut s = [0.0; MOST];
        for v in s.iter_mut().take(k) {
            *v = rng.range(-6.0, 6.0) as f64;
        }
        let target = (rng.range(0.0, k as f32 - 0.001) as usize).min(k - 1);
        let temp = (rng.range((TEMP_LO as f32).ln(), (TEMP_HI as f32).ln()) as f64).exp();
        let smooth = if rng.bool(0.3) { 0.0 } else { rng.range(0.0, 0.5) as f64 };
        Setting { k, s, target, temp, smooth }.kept()
    }

    /// Every slope the scene shows is the slope of its own loss: central differences of
    /// the loss, logit by logit, for every pairing over settings of every shape.
    #[test]
    fn every_slope_is_the_difference_of_its_loss() {
        let mut rng = Rng::new(7);
        let h = 1e-5;
        for _ in 0..400 {
            let set = random_setting(&mut rng);
            let (k, y) = (set.k, set.truth());
            for p in PAIRINGS {
                let g = set.slope(p);
                for i in 0..k {
                    let (mut up, mut down) = (set.s, set.s);
                    up[i] += h;
                    down[i] -= h;
                    let diff = (p.loss(&up[..k], &y[..k], set.temp) - p.loss(&down[..k], &y[..k], set.temp)) / (2.0 * h);
                    assert!(close(g[i], diff, 1e-7 * (1.0 + diff.abs())), "{p:?} at {set:?}, logit {i}: {} against {diff}", g[i]);
                }
            }
        }
        // and the margin charts' slopes are the same slopes
        let mut s = LossGrad::create();
        s.follow(0.016, true);
        let (shown, y) = shown_setting(&s.set, &s.look.state);
        for (j, p) in PAIRINGS.iter().enumerate() {
            for i in [0, 30, 60, 90, 120] {
                let mut at = shown.s;
                at[shown.target] = curve_x(i);
                assert!(close(s.curves.slope[j][i], p.slope(&at[..4], &y[..4], shown.temp)[0].abs(), 1e-12));
                assert!(close(s.curves.loss[j][i], p.loss(&at[..4], &y[..4], shown.temp), 1e-12));
            }
        }
    }

    /// Softmax with cross-entropy: the slope is (p − y)/T, its entries add up to 0 and
    /// none is larger than 1/T; the largest pull on the target is when it is
    /// confidently wrong. The sigmoid's squared error has its slope times σ(1 − σ).
    #[test]
    fn the_slopes_are_what_the_rows_say() {
        let mut rng = Rng::new(11);
        for _ in 0..300 {
            let set = random_setting(&mut rng);
            let (k, y) = (set.k, set.truth());
            let p = set.outputs(Pairing::SoftmaxCe);
            let g = set.slope(Pairing::SoftmaxCe);
            assert!(close(p[..k].iter().sum::<f64>(), 1.0, 1e-12));
            assert!(close(g[..k].iter().sum::<f64>(), 0.0, 1e-12), "{set:?}");
            for i in 0..k {
                assert!(close(g[i], (p[i] - y[i]) / set.temp, 1e-12));
                assert!(g[i].abs() <= 1.0 / set.temp + 1e-12);
                let q = sigmoid(set.s[i]);
                assert!(close(set.slope(Pairing::SigmoidSe)[i], (q - y[i]) * q * (1.0 - q), 1e-15));
                assert!(close(set.slope(Pairing::SigmoidCe)[i], q - y[i], 1e-15));
                assert!(close(set.slope(Pairing::LinearSe)[i], set.s[i] - y[i], 1e-15));
            }
        }
        // the target's pull, with the hard truth and T = 1: grows as its logit falls,
        // towards 1 for the cross-entropies, and towards 0 for the squared error
        let at = |p: Pairing, x: f64| {
            let set = Setting { k: 4, s: [x, 4.0, 1.0, -1.0, 0.0], target: 0, temp: 1.0, smooth: 0.0 };
            set.slope(p)[0].abs()
        };
        for p in [Pairing::SoftmaxCe, Pairing::SigmoidCe] {
            assert!(at(p, -6.0) > at(p, 0.0) && at(p, 0.0) > at(p, 6.0) && at(p, -6.0) > 0.99, "{p:?}");
        }
        assert!(at(Pairing::SigmoidSe, -6.0) < 0.003 && at(Pairing::SigmoidSe, -6.0) > 0.002);
        // (its largest is 4/27 at σ = 1/3, a little left of the middle)
        let most = (0..=1200).map(|i| at(Pairing::SigmoidSe, -6.0 + i as f64 * 0.01)).fold(0.0, f64::max);
        assert!(close(most, 4.0 / 27.0, 1e-4), "{most}");
    }

    /// The softmax's Jacobian is diag(p) − p pᵀ (over T): the differences of the
    /// probabilities, rows that add up to 0, and the chain rule through it gives p − y.
    #[test]
    fn the_softmax_jacobian_is_the_differences_of_the_probabilities() {
        let mut rng = Rng::new(5);
        for _ in 0..100 {
            let set = random_setting(&mut rng);
            let k = set.k;
            let p = set.outputs(Pairing::SoftmaxCe);
            let j = softmax_jacobian(&p[..k], set.temp);
            let h = 1e-6;
            for b in 0..k {
                let (mut up, mut down) = (set.s, set.s);
                up[b] += h;
                down[b] -= h;
                let (pu, pd) = (Pairing::SoftmaxCe.outputs(&up[..k], set.temp), Pairing::SoftmaxCe.outputs(&down[..k], set.temp));
                for a in 0..k {
                    assert!(close(j[a][b], (pu[a] - pd[a]) / (2.0 * h), 1e-8), "{set:?}: ({a}, {b})");
                }
            }
            let y = set.truth();
            let g = set.slope(Pairing::SoftmaxCe);
            for a in 0..k {
                assert!(close(j[a][..k].iter().sum::<f64>(), 0.0, 1e-12));
                // Σ_b ∂p_b/∂s_a · (−y_b/p_b): the matrix is symmetric
                let chain: f64 = (0..k).map(|b| j[b][a] * (-y[b] / p[b])).sum();
                assert!(close(chain, g[a], 1e-9 * (1.0 + g[a].abs())), "{set:?}: {chain} against {}", g[a]);
            }
        }
    }

    /// What each experiment's caption says, from the scene as its button sets it up.
    #[test]
    fn the_experiments_show_what_their_captions_say() {
        let says = |k: usize, what: &str| assert!(TRIALS[k].caption.contains(what), "experiment {k} should say \"{what}\": {}", TRIALS[k].caption);
        let right = |s: &LossGrad, lane: usize| s.race.lanes[lane].right;
        // probabilities minus the truth
        let s = run(0);
        let p = s.set.outputs(Pairing::SoftmaxCe);
        let g = s.set.slope(Pairing::SoftmaxCe);
        says(0, &format!("$p_1 = {}$", num(p[0], 2)));
        says(0, &format!("${} - 1 = {}$", num(p[0], 2), num(g[0], 2).replace('−', "-")));
        says(0, &format!("probability, {}, {} and {}.", num(g[1], 2), num(g[2], 2), num(g[3], 2)));
        assert!(close(g[..4].iter().sum::<f64>(), 0.0, 1e-12) && g.iter().all(|v| v.abs() <= 1.0));
        // a step against the slope lowers class 2's logit the most
        assert!(g[0] < 0.0 && g[1] > g[2] && g[1] > g[3] && g[2] > 0.0 && g[3] > 0.0);
        // confident and wrong
        let s = run(1);
        let p = s.set.outputs(Pairing::SoftmaxCe);
        says(1, &format!("class 2 a probability of {} and the target {}.", num(p[1], 2), num(p[0], 4)));
        let soft = s.set.slope(Pairing::SoftmaxCe)[0];
        let se = s.set.slope(Pairing::SigmoidSe)[0];
        let ce = s.set.slope(Pairing::SigmoidCe)[0];
        let q = sigmoid(s.set.s[0]);
        says(1, &format!("with slope {}, as hard", num(soft, 2)));
        says(1, &format!("pulls with {}: its slope", num(se, 3)));
        says(1, &format!(r"$\sigma(1 - \sigma) = {}$", num(q * (1.0 - q), 3)));
        says(1, &format!("cross-entropy pulls with {}:", num(ce, 2)));
        // "as hard as it ever pulls": the target's pull is below 1 everywhere and
        // within a thousandth of it here
        assert!(soft > -1.0 && soft < -0.999);
        // its loss levels off to the left: over the chart's left quarter (the target's
        // logit from −6 to −3) the squared error's loss changes by less than a fiftieth
        // of the cross-entropy's
        let mut c = Curves::default();
        let (shown, y) = (s.set, s.set.truth());
        c.update(&shown, &y);
        assert_eq!(curve_x(CURVE_N / 4), -3.0);
        let drop = |j: usize| c.loss[j][0] - c.loss[j][CURVE_N / 4];
        assert!(drop(1) < drop(0) / 50.0 && drop(1) > 0.0, "{} against {}", drop(1), drop(0));
        // the race from a wrong start
        let s = run(2);
        let p = s.set.outputs(Pairing::SoftmaxCe);
        says(2, &format!("probability of {}, under", num(p[1], 3)));
        says(2, &format!(r"$\eta = {}$", num(s.eta, 1)));
        let (soft, se, ce, lin) = (right(&s, 0), right(&s, 1), right(&s, 2), right(&s, 3));
        assert!(soft.is_some() && soft == ce);
        says(2, &format!("right after {} steps, and so has", soft.unwrap_or(0)));
        says(2, &format!("needs {}:", se.unwrap_or(0)));
        says(2, &format!("starts at {}, small", brief(s.set.slope(Pairing::SigmoidSe)[0])));
        let gaps = &s.race.lanes[1].gaps;
        says(2, &format!("only from {} to {}.", num(gaps[0], 0), num(gaps[100], 2)));
        says(2, &format!("right after {} (its", lin.unwrap_or(0)));
        says(2, &format!("error, {} on", num(s.set.slope(Pairing::LinearSe)[0], 0)));
        // temperature
        let s = run(3);
        let cold = Setting { temp: 1.0, ..s.set };
        let tuple = |v: &[f64]| format!("$({})$", v.iter().map(|x| num(*x, 2)).collect::<Vec<_>>().join(", "));
        says(3, &format!("${} = {}$", "T", num(s.set.temp, 0)));
        says(3, &format!("from {} to {}", tuple(&cold.outputs(Pairing::SoftmaxCe)[..4]), tuple(&s.set.outputs(Pairing::SoftmaxCe)[..4])));
        says(3, &format!("logit {} instead of {}", num(s.set.slope(Pairing::SoftmaxCe)[0], 2), num(cold.slope(Pairing::SoftmaxCe)[0], 2)));
        let mut cold_race = Race::new(cold, TRIAL_ETA);
        cold_race.finish();
        says(3, &format!("needs {} steps instead of {}", right(&s, 0).unwrap_or(0), cold_race.lanes[0].right.unwrap_or(0)));
        let sharp = Setting { temp: 0.5, ..s.set };
        says(3, &format!(r"$T = 0.5$ class 2 gets {}.", num(sharp.outputs(Pairing::SoftmaxCe)[1], 2)));
        // a softened target that can be reached
        let s = run(4);
        let y = s.set.truth();
        says(4, &format!("the truth is {} for the target and {} for each", num(y[0], 1), num(y[1], 3)));
        let d = reachable_gap(4, s.set.smooth, s.set.temp).unwrap_or(f64::NAN);
        assert!(close(d, 27f64.ln(), 1e-12));
        says(4, &format!(r"$\ln 27 = {}$", num(d, 2)));
        let soft = &s.race.lanes[0];
        assert!(close(soft.gaps[RACE_STEPS], d, 1e-6), "{}", soft.gaps[RACE_STEPS]);
        // the logits stop growing: the last thousand steps move them by nothing to speak of
        assert!((soft.gaps[RACE_STEPS] - soft.gaps[RACE_STEPS - 1000]).abs() < 1e-6);
        assert!(s.set.slope(Pairing::SoftmaxCe).iter().any(|g| g.abs() > 0.1));
        let at_end = Pairing::SoftmaxCe.slope(&soft.s[..4], &y[..4], 1.0);
        assert!(at_end.iter().all(|g| g.abs() < 1e-9), "{at_end:?}");
        let ghost = &s.race.lanes[4];
        assert!(s.race.ghost() && ghost.pairing == Pairing::SoftmaxCe);
        says(4, &format!("that gap is {} after 10 000 steps", num(ghost.gaps[RACE_STEPS], 2)));
        let decade = ghost.gaps[RACE_STEPS] - ghost.gaps[1000];
        assert!(decade > 2.3 && decade < 2.45, "{decade}");
        says(4, &format!("climbing, {} more than after 1000.", num(decade, 1)));
        assert!(ghost.gaps[RACE_STEPS] > ghost.gaps[RACE_STEPS - 1]);
    }

    /// The opening race, from the second experiment's start: the stall of the
    /// sigmoid with the squared error, beside the others.
    #[test]
    fn the_scene_opens_on_the_second_experiment() {
        let opened = LossGrad::create();
        let mut pressed = LossGrad::create();
        pressed.set_up(OPENING);
        assert_eq!((opened.set, opened.eta, opened.tab), (pressed.set, pressed.eta, pressed.tab));
        assert!(opened.playing && opened.film == 0.0);
        let s = run(OPENING);
        let counts: Vec<Option<usize>> = s.race.lanes.iter().map(|l| l.right).collect();
        assert_eq!(counts, vec![Some(10), Some(140), Some(10), Some(4)]);
    }

    /// Against a softened truth softmax with cross-entropy comes to rest at the gap
    /// T ln((1 − ε)(K − 1)/ε), for every number of classes and temperature; against the
    /// hard one it never does.
    #[test]
    fn a_softened_target_is_reached_and_a_hard_one_is_not() {
        for k in FEWEST..=MOST {
            for (temp, smooth) in [(1.0, 0.1), (2.0, 0.2), (0.5, 0.05), (1.0, 0.5)] {
                let mut s = [0.0; MOST];
                s[1] = 2.0;
                let set = Setting { k, s, target: 0, temp, smooth }.kept();
                let mut race = Race::new(set, 0.5);
                race.finish();
                let lane = &race.lanes[0];
                let d = reachable_gap(k, smooth, temp).unwrap_or(f64::NAN);
                assert!(close(lane.gaps[RACE_STEPS], d, 1e-4), "{k} classes, T {temp}, ε {smooth}: {} against {d}", lane.gaps[RACE_STEPS]);
                // where it rests, p = y
                let p = Pairing::SoftmaxCe.outputs(&lane.s[..k], temp);
                let y = set.truth();
                assert!((0..k).all(|i| close(p[i], y[i], 1e-5)), "{p:?} against {y:?}");
                // and the hard truth's gap is still growing at the end
                let hard = &race.lanes[4];
                assert!(hard.gaps[RACE_STEPS] > hard.gaps[RACE_STEPS - 100] && hard.gaps[RACE_STEPS] > d);
            }
        }
        assert_eq!(reachable_gap(4, 0.0, 1.0), None);
    }

    /// Edges: equal logits, logits at the ends, the temperature and the smoothing and
    /// the step at their ends, every number of classes: everything finite, nothing
    /// that panics, and the race's numbers finite to its end.
    #[test]
    fn degenerate_settings_stay_finite() {
        let ends = [-LOGIT_MOST, LOGIT_MOST, 0.0];
        for k in FEWEST..=MOST {
            for &a in &ends {
                for &b in &ends {
                    for temp in [TEMP_LO, 1.0, TEMP_HI] {
                        for smooth in [0.0, SMOOTH_MOST] {
                            let mut s = [b; MOST];
                            s[0] = a;
                            let set = Setting { k, s, target: k - 1, temp, smooth }.kept();
                            for p in PAIRINGS {
                                assert!(set.loss(p).is_finite() && set.loss(p) >= 0.0, "{p:?} {set:?}");
                                assert!(set.slope(p).iter().chain(&set.outputs(p)).all(|v| v.is_finite()));
                            }
                            for eta in [ETA_LO, ETA_HI] {
                                let mut race = Race::new(set, eta);
                                race.finish();
                                assert!(race.lanes.iter().all(|l| l.gaps.len() == RACE_STEPS + 1 && l.gaps.iter().all(|g| g.is_finite())), "{set:?} η {eta}");
                                let (lo, hi) = race.range();
                                assert!(lo.is_finite() && hi.is_finite() && lo <= hi);
                            }
                        }
                    }
                }
            }
        }
        // what is not a number is kept out
        let wild = Setting { k: 9, s: [f64::NAN, f64::INFINITY, -1e9, 3.0, 2.0], target: 7, temp: f64::NAN, smooth: -3.0 }.kept();
        assert!(wild.k == MOST && wild.target == MOST - 1 && wild.temp == 1.0 && wild.smooth == 0.0);
        assert!(wild.s.iter().all(|v| v.is_finite() && v.abs() <= LOGIT_MOST));
        let few = Setting { k: 0, ..wild }.kept();
        assert!(few.k == FEWEST && few.target < FEWEST && few.s[FEWEST..].iter().all(|v| *v == 0.0));
        // a race worked out in pieces is the race worked out at once
        let mut whole = Race::new(few, 0.7);
        whole.finish();
        let mut pieces = Race::new(few, 0.7);
        while !pieces.finished() {
            pieces.advance(RACE_CHUNK);
        }
        assert_eq!(whole.lanes.iter().map(|l| l.gaps.clone()).collect::<Vec<_>>(), pieces.lanes.iter().map(|l| l.gaps.clone()).collect::<Vec<_>>());
        assert!(close(pieces.gap_at(0, 2.5), (pieces.lanes[0].gaps[2] + pieces.lanes[0].gaps[3]) * 0.5, 1e-12));
        assert!(pieces.gap_at(9, 1.0) == 0.0 && pieces.gap_at(0, 1e9).is_finite());
    }

    /// Numbers as the picture writes them.
    #[test]
    fn numbers_are_written_briefly() {
        assert_eq!(brief(-0.763), "−0.76");
        assert_eq!(brief(-0.01735), "−0.017");
        assert_eq!(brief(0.0025), "0.0025");
        assert_eq!(brief(0.000_317), "0.0003");
        assert_eq!(brief(6.1e-6), r"$6 \times 10^{-6}$");
        assert_eq!(brief(-9.96e-7), r"$-1 \times 10^{-6}$");
        assert_eq!((brief(0.0), brief(f64::NAN)), ("0".to_string(), "?".to_string()));
        assert_eq!((plain(-4.0), plain(0.5), plain(1.25), plain(-0.004)), ("−4".to_string(), "0.5".to_string(), "1.25".to_string(), "0".to_string()));
        assert_eq!((steps_text(1), steps_text(850), steps_text(10_000)), ("1 step".to_string(), "850 steps".to_string(), "10\u{2009}000 steps".to_string()));
        let samples = race_samples();
        assert!(samples.windows(2).all(|w| w[0] < w[1]) && samples.first() == Some(&0) && samples.last() == Some(&RACE_STEPS) && samples.len() < 260);
    }

    /// The layout keeps everything in the window and apart, from a phone to a desktop.
    #[test]
    fn the_layout_fits_every_window() {
        for (w, h) in [(1078.0, 1000.0), (844.0, 768.0), (578.0, 700.0), (1600.0, 1000.0), (390.0, 800.0), (360.0, 640.0), (700.0, 500.0), (300.0, 520.0), (500.0, 400.0), (240.0, 180.0)] {
            for tab in [Tab::Loss, Tab::Slope, Tab::Race] {
                let lay = Layout::of(Vec2::new(w, h), 46.0, tab, 5);
                let inside = |(a, b): Rect| a.x >= 0.0 && a.y >= 0.0 && b.x <= w + 0.5 && b.y <= h + 0.5 && a.is_finite() && b.is_finite();
                let apart = |(a, b): Rect, (c, d): Rect| b.x <= c.x || d.x <= a.x || b.y <= c.y || d.y <= a.y;
                let real = |(a, b): &Rect| b.x - a.x > 2.0 && b.y - a.y > 2.0;
                let mut boxes: Vec<Rect> = vec![lay.logits, lay.targets];
                boxes.extend(lay.rows);
                if !lay.compact {
                    boxes.extend([lay.loss, lay.slope, lay.race, lay.race_list, lay.legend, lay.race_button]);
                    boxes.extend(lay.tabs);
                }
                let boxes: Vec<Rect> = boxes.into_iter().filter(real).collect();
                for (i, a) in boxes.iter().enumerate() {
                    assert!(inside(*a), "{w}x{h} {tab:?}: box {i} {a:?}");
                    for b in &boxes[i + 1..] {
                        assert!(apart(*a, *b), "{w}x{h} {tab:?}: {a:?} and {b:?} overlap");
                    }
                }
                if !lay.compact {
                    let shown = if lay.wide { vec![lay.loss, lay.slope, lay.race] } else { vec![[lay.loss, lay.slope, lay.race].into_iter().find(real).unwrap_or(lay.loss)] };
                    let least = if lay.wide { 100.0 } else { 70.0 };
                    for c in shown {
                        assert!(c.1.y - c.0.y > least && c.1.x - c.0.x > 200.0, "{w}x{h} {tab:?}: {c:?}");
                    }
                }
                if w >= 1000.0 && h >= 900.0 {
                    assert!(lay.wide && lay.unit >= 30.0, "{w}x{h}");
                }
                if (w, h) == (390.0, 800.0) || (w, h) == (360.0, 640.0) {
                    assert!(!lay.wide && !lay.compact, "{w}x{h}");
                }
            }
        }
    }

    /// The picture's glides come to rest after a change, and the film of a race ends.
    #[test]
    fn glides_arrive_and_stop() {
        let mut s = LossGrad::create();
        s.follow(0.016, true);
        assert!(!s.look.moving);
        for k in [0, 2, 3, 4, 1] {
            s.set_up(k);
            let mut frames = 0;
            loop {
                s.work(1.0 / 60.0);
                s.follow(1.0 / 60.0, false);
                frames += 1;
                if !(s.look.moving || s.playing || !s.race.finished()) {
                    break;
                }
                assert!(frames < 600, "experiment {k} still moving");
            }
            // (the film takes its time: four tenfolds at 0.9 a second)
            assert!(frames > 200, "experiment {k}: {frames}");
        }
        // a drag answers at once
        s.dragging = Some(2);
        s.set.s[2] = 5.0;
        s.follow(1.0 / 60.0, false);
        assert_eq!(s.look.state.get(LOOK_S + 2), 5.0);
    }
}

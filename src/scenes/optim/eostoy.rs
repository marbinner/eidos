//! Too sharp to stay: gradient descent on a loss of two weights whose valley floor is a
//! whole curve of minima, flat in the middle and sharper and sharper out along its arms.

use super::*;
use crate::scenes::stats::plot::{self, Plot};
use glam::DVec2;

pub const DEF: SceneDef = SceneDef {
    id: "eostoy",
    title: "Too sharp to stay: two weights",
    group: GROUP,
    blurb: "A valley whose walls grow steeper out along its arms: gradient descent cannot settle where the curvature is more than two over its step, and is thrown along to flatter ground.",
    make: |_| Box::new(EosToy::new()),
};

// ----- the two losses, in double precision -------------------------------------------

/// The two losses. Each depends on the weights only through their product p = ab, as
/// L = ℓ(p), so that its minima are whole curves: the valley floor ab = 1 (and, for the
/// second, ab = −1 as well).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Loss {
    /// ½(ab − 1)²: a network of two weights in a row, a and b, fitted to the target 1.
    Two,
    /// ¼(1 − a²b²)²: four weights in a row, ½(1 − xyzw)², started with z = x and w = y,
    /// which gradient descent keeps equal.
    Four,
}

impl Loss {
    /// ℓ(p).
    fn ell(self, p: f64) -> f64 {
        match self {
            Loss::Two => 0.5 * (p - 1.0) * (p - 1.0),
            Loss::Four => 0.25 * (1.0 - p * p) * (1.0 - p * p),
        }
    }

    /// ℓ′(p): how the loss changes with the product.
    fn slope(self, p: f64) -> f64 {
        match self {
            Loss::Two => p - 1.0,
            Loss::Four => p * (p * p - 1.0),
        }
    }

    /// ℓ″(p).
    fn bend(self, p: f64) -> f64 {
        match self {
            Loss::Two => 1.0,
            Loss::Four => 3.0 * p * p - 1.0,
        }
    }

    fn value(self, z: DVec2) -> f64 {
        self.ell(z.x * z.y)
    }

    /// ∇L = ℓ′(ab) (b, a).
    fn gradient(self, z: DVec2) -> DVec2 {
        let g = self.slope(z.x * z.y);
        DVec2::new(g * z.y, g * z.x)
    }

    /// The Hessian as [∂²L/∂a², ∂²L/∂a∂b, ∂²L/∂b²]: ℓ″(p) (b, a)(b, a)ᵀ, the curvature
    /// across the level lines, plus ℓ′(p) on the two off-diagonal places, from the
    /// product's own second derivative ∂²(ab)/∂a∂b = 1.
    fn hessian(self, z: DVec2) -> [f64; 3] {
        let p = z.x * z.y;
        let (g, h) = (self.slope(p), self.bend(p));
        [h * z.y * z.y, h * p + g, h * z.x * z.x]
    }

    /// The sharpness: the Hessian's larger eigenvalue.
    fn sharpness(self, z: DVec2) -> f64 {
        largest(self.hessian(z))
    }

    /// ℓ″ on the floor (where ab = ±1): the sharpness there is this times a² + b², and
    /// a² + b² = √((a² − b²)² + 4a²b²) = √((a² − b²)² + 4).
    fn scale(self) -> f64 {
        match self {
            Loss::Two => 1.0,
            Loss::Four => 2.0,
        }
    }

    /// The sharpness of the floor where a² − b² = d.
    fn floor_sharpness(self, d: f64) -> f64 {
        self.scale() * (d * d + 4.0).sqrt()
    }

    /// The flattest a minimum gets: at a = b = ±1.
    fn flattest(self) -> f64 {
        2.0 * self.scale()
    }

    /// The products, from 0 up, at which the loss is `level` (each a pair of level lines).
    fn level_products(self, level: f64) -> Vec<f64> {
        let mut out = Vec::new();
        match self {
            Loss::Two => {
                let r = (2.0 * level).sqrt();
                out.extend([1.0 - r, 1.0 + r]);
            }
            Loss::Four => {
                let r = 2.0 * level.sqrt();
                for sq in [1.0 - r, 1.0 + r] {
                    if sq >= 0.0 {
                        out.extend([sq.sqrt(), -sq.sqrt()]);
                    }
                }
            }
        }
        out
    }

    fn index(self) -> usize {
        self as usize
    }

    fn of(index: usize) -> Loss {
        if index == 1 {
            Loss::Four
        } else {
            Loss::Two
        }
    }
}

/// How far out along a floor of scale `scale` (`Loss::scale`), in |a² − b²|, its
/// sharpness stays at or under `s` (None if nowhere: `s` below the flattest).
fn reach(scale: f64, s: f64) -> Option<f64> {
    let r = s / scale;
    (r >= 2.0).then(|| (r * r - 4.0).max(0.0).sqrt())
}

/// The larger eigenvalue of the symmetric matrix [[x, y], [y, z]].
fn largest([x, y, z]: [f64; 3]) -> f64 {
    0.5 * (x + z) + (0.5 * (x - z)).hypot(y)
}

/// The point where a² − b² = d and ab = p, on the side of `side`: the path of gradient
/// flow through `side` is the curve a² − b² = d, along which a keeps its sign if d > 0
/// and b keeps its sign if d < 0. None at the origin, which every such curve with d = 0
/// and p = 0 runs into.
fn on_flow_curve(d: f64, p: f64, side: DVec2) -> Option<DVec2> {
    let r = (d * d + 4.0 * p * p).sqrt();
    if d >= 0.0 {
        let a = (0.5 * (d + r)).sqrt();
        (a > 0.0).then(|| {
            let a = a.copysign(if side.x == 0.0 { side.y } else { side.x });
            DVec2::new(a, p / a)
        })
    } else {
        let b = (0.5 * (r - d)).sqrt().copysign(side.y);
        Some(DVec2::new(p / b, b))
    }
}

/// Where gradient flow (infinitely small steps) from z comes to rest, if at a minimum:
/// on the floor with the same a² − b² as z, since the flow keeps a² − b² fixed. None
/// where it never reaches one: from the line a = −b the first loss's flow runs into the
/// saddle at the origin, and on the axes the second loss's slope is 0.
fn landing(loss: Loss, z: DVec2) -> Option<DVec2> {
    let (p, d) = (z.x * z.y, z.x * z.x - z.y * z.y);
    if !(p.is_finite() && d.is_finite()) {
        return None;
    }
    let q = match loss {
        Loss::Two if d == 0.0 && p <= 0.0 => return None,
        Loss::Two => 1.0,
        Loss::Four if p == 0.0 => return None,
        Loss::Four => p.signum(),
    };
    on_flow_curve(d, q, z)
}

/// The sharpness of the floor beneath z: where gradient flow from z comes to rest
/// (NaN where it does not come to rest at a minimum).
fn beneath(loss: Loss, z: DVec2) -> f64 {
    match landing(loss, z) {
        Some(_) => loss.floor_sharpness(z.x * z.x - z.y * z.y),
        None => f64::NAN,
    }
}

/// Points along gradient flow's path from z: the curve a² − b² = const from z to where
/// the flow comes to rest (the origin, for the first loss from the line a = −b; z
/// alone, for the second from an axis).
fn flow_path(loss: Loss, z: DVec2, points: usize) -> Vec<DVec2> {
    let (p, d) = (z.x * z.y, z.x * z.x - z.y * z.y);
    let end = match landing(loss, z) {
        Some(w) => w.x * w.y,
        None if loss == Loss::Two => 0.0,
        None => return vec![z; points.max(2)],
    };
    let n = points.max(2);
    // (only the origin has no point on its curve: it is where such a path ends)
    (0..n).map(|i| on_flow_curve(d, p + (end - p) * i as f64 / (n - 1) as f64, z).unwrap_or(DVec2::ZERO)).collect()
}

/// One step of gradient descent.
fn step(loss: Loss, eta: f64, z: DVec2) -> DVec2 {
    z - loss.gradient(z) * eta
}

// ----- a run of gradient descent ------------------------------------------------------

/// The most steps a run is given.
const MOST_STEPS: usize = 6000;
/// A weight further out than this has been thrown off.
const FAR: f64 = 50.0;
/// A loss under this is at a minimum.
const SETTLED: f64 = 1e-14;

/// How a run ended.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Ending {
    /// At a minimum: the loss under `SETTLED`, on a part of the floor flat enough to stay.
    Settled,
    /// Where the slope is 0 and the loss is not, or on a minimum too sharp to stay at
    /// that it was started on exactly: nothing moves it, and it stays.
    Stuck,
    /// Bouncing for ever round a cycle of this many points.
    Cycle(usize),
    /// Still moving after `MOST_STEPS` steps.
    Restless,
    /// Thrown off: a weight grew past `FAR`.
    FlewOff,
}

/// Gradient descent from `start` until it ends, calling `visit` with each iterate after
/// the start. Returns how it ended, the last iterate, and how many steps were taken.
fn descend(loss: Loss, eta: f64, start: DVec2, mut visit: impl FnMut(DVec2)) -> (Ending, DVec2, usize) {
    let mut z = start;
    if !z.is_finite() || z.abs().max_element() > FAR {
        return (Ending::FlewOff, z, 0);
    }
    // the last few iterates, to see a cycle by
    let mut ring = [z; RING];
    for n in 1..=MOST_STEPS {
        if loss.value(z) < SETTLED && loss.floor_sharpness(z.x * z.x - z.y * z.y) * eta < 2.0 + 1e-9 {
            return (Ending::Settled, z, n - 1);
        }
        let next = step(loss, eta, z);
        let moved = (next - z).abs().max_element();
        // (nothing moves it, or nothing will: the slope is 0, or dying away at a point
        // that is not a minimum, a saddle the iterates run into)
        if moved == 0.0 || (moved < 1e-16 * (1.0 + z.abs().max_element()) && loss.value(z) > 1e-10) {
            return (Ending::Stuck, z, n - 1);
        }
        z = next;
        visit(z);
        if !z.is_finite() || z.abs().max_element() > FAR {
            return (Ending::FlewOff, z, n);
        }
        ring[n % RING] = z;
        // a cycle: the last two iterates each where it was a whole cycle before (the
        // shortest such cycle, and not a point standing still, which settles or sticks)
        if n >= 3 * RING && loss.value(z) > 1e-12 {
            let tol = 1e-12 * (1.0 + z.abs().max_element());
            let back = |k: usize| ring[(n - k) % RING];
            let apart = (back(0) - back(1)).abs().max_element() > 1e-9 * (1.0 + z.abs().max_element());
            for period in [2usize, 4, 8] {
                if apart && (back(0) - back(period)).abs().max_element() <= tol && (back(1) - back(period + 1)).abs().max_element() <= tol {
                    return (Ending::Cycle(period), z, n);
                }
            }
        }
    }
    (Ending::Restless, z, MOST_STEPS)
}

/// How many iterates are kept to see a cycle by: enough for the longest looked for, 8,
/// and one more.
const RING: usize = 10;

/// A whole run, kept for drawing: every iterate, and at each the loss, the sharpness
/// there, and the sharpness of the floor beneath.
#[derive(Clone, Debug)]
struct Run {
    loss: Loss,
    eta: f64,
    z: Vec<DVec2>,
    value: Vec<f64>,
    sharp: Vec<f64>,
    under: Vec<f64>,
    ending: Ending,
}

impl Run {
    fn new(loss: Loss, eta: f64, start: DVec2) -> Run {
        let mut z = vec![start];
        let (ending, ..) = descend(loss, eta, start, |w| z.push(w));
        let value = z.iter().map(|w| loss.value(*w)).collect();
        let sharp = z.iter().map(|w| loss.sharpness(*w)).collect();
        let under = z.iter().map(|w| beneath(loss, *w)).collect();
        Run { loss, eta, z, value, sharp, under, ending }
    }

    /// The number of steps taken.
    fn steps(&self) -> usize {
        self.z.len() - 1
    }

    fn last(&self) -> DVec2 {
        self.z[self.steps()]
    }

    /// The most steps in a row at which the loss went up, and how many in all.
    fn rises(&self) -> (usize, usize) {
        let (mut longest, mut now, mut all) = (0, 0, 0);
        for w in self.value.windows(2) {
            if w[1] > w[0] {
                now += 1;
                all += 1;
                longest = longest.max(now);
            } else {
                now = 0;
            }
        }
        (longest, all)
    }

    /// The sharpness of the minimum it settled at.
    fn end_sharpness(&self) -> Option<f64> {
        (self.ending == Ending::Settled).then(|| self.under[self.steps()])
    }
}

// ----- where runs from one start end, for every step -----------------------------------

/// The steps tried, spaced evenly on a logarithmic scale.
const SWEEP: usize = 81;
const ETA_LO: f64 = 0.01;
const ETA_HI: f64 = 2.0;

fn sweep_eta(i: usize) -> f64 {
    ETA_LO * (ETA_HI / ETA_LO).powf(i as f64 / (SWEEP - 1) as f64)
}

/// Where one run ended.
#[derive(Clone, Copy, Debug, PartialEq)]
enum Outcome {
    /// At a minimum of this sharpness.
    Ends(f64),
    /// Anywhere else (see `Ending`).
    Not(Ending),
}

fn outcome(loss: Loss, eta: f64, start: DVec2) -> Outcome {
    let (ending, last, _) = descend(loss, eta, start, |_| {});
    match ending {
        Ending::Settled => Outcome::Ends(loss.floor_sharpness(last.x * last.x - last.y * last.y)),
        other => Outcome::Not(other),
    }
}


/// All the steps of the sweep, once: where a run from `start` ends for each.
struct Sweep {
    loss: Loss,
    start: DVec2,
    ends: Vec<Outcome>,
}

impl Sweep {
    fn new(loss: Loss, start: DVec2) -> Sweep {
        Sweep { loss, start, ends: Vec::with_capacity(SWEEP) }
    }

    fn done(&self) -> bool {
        self.ends.len() >= SWEEP
    }

    /// Go on with the runs until they are all done or the allowance is spent (at least
    /// one run each time, so that it always gets on).
    fn work(&mut self, budget: &crate::clock::Budget) {
        while !self.done() {
            let i = self.ends.len();
            self.ends.push(outcome(self.loss, sweep_eta(i), self.start));
            if budget.spent() {
                break;
            }
        }
    }

    /// The longest step of the sweep that settles, if any does.
    fn longest_settling(&self) -> Option<f64> {
        self.ends.iter().enumerate().rev().find(|(_, o)| matches!(o, Outcome::Ends(_))).map(|(i, _)| sweep_eta(i))
    }
}

// ----- the film -----------------------------------------------------------------------

/// The film shows this many steps a second as it begins, and more and more: after t
/// seconds it has shown R τ (e^{t/τ} − 1) of them. So the first bounces are seen one by
/// one, a run of thousands of steps is over in a quarter of a minute, and on the charts'
/// logarithmic step axis the film moves along at a nearly even pace.
const FILM_RATE: f64 = 6.0;
const FILM_TAU: f64 = 2.5;

/// How far the film has got `dt` seconds after it was at `s` steps.
fn film_on(s: f64, dt: f64) -> f64 {
    let k = FILM_RATE * FILM_TAU;
    (s + k) * (dt.max(0.0) / FILM_TAU).exp() - k
}

/// The point `s` steps along the path (a fraction of a step is that much of the
/// straight line to the next iterate).
fn along(z: &[DVec2], s: f64) -> DVec2 {
    let Some(&last) = z.last() else { return DVec2::ZERO };
    let n = z.len() - 1;
    let s = s.max(0.0).min(n as f64);
    let i = s.floor() as usize;
    if i >= n {
        last
    } else {
        z[i].lerp(z[i + 1], s - i as f64)
    }
}

/// The two dots `s` steps into the film: where the point is after an even and after an
/// odd number of steps, each gliding from one iterate of its kind to the next, two
/// steps on. While the point bounces from wall to wall each dot keeps to its own wall
/// and moves smoothly along it (the point itself, jumping across, would flicker); the
/// path's growing end jumps from one to the other; and at the end they are the last
/// two iterates.
fn dots(z: &[DVec2], s: f64) -> [DVec2; 2] {
    let Some(&last) = z.last() else { return [DVec2::ZERO; 2] };
    let n = z.len() - 1;
    let s = s.max(0.0).min(n as f64);
    [0usize, 1].map(|k| {
        // (before the first odd iterate, the odd dot goes with the path from the start)
        if s < k as f64 {
            return z[0].lerp(z[k.min(n)], s);
        }
        let lo = (k + 2 * ((s - k as f64) / 2.0).floor() as usize).min(n);
        match z.get(lo + 2) {
            Some(hi) => z[lo].lerp(*hi, (s - lo as f64) / 2.0),
            None if lo == n => last,
            None => z[lo],
        }
    })
}

// ----- experiments --------------------------------------------------------------------

struct Trial {
    label: &'static str,
    look_for: &'static str,
    loss: Loss,
    eta: f64,
    start: [f64; 2],
}

/// A start a little above the floor far out along its arm, where it is sharp, and one
/// near its flat middle.
const SHARP: [f64; 2] = [3.0, 0.35];
const MIDDLE: [f64; 2] = [1.3, 0.9];

const TRIALS: [Trial; 5] = [
    Trial {
        label: "a short step",
        look_for: r"The start is a little above the floor, far out along its arm, where the floor beneath it has $S = 9.10$. The step $\eta = 0.2$ allows up to $2/\eta = 10$, so the yellow stretch reaches past it. Gradient descent comes down onto the floor nearly beneath the start, where gradient flow (dashed) ends, swinging from one side to the other with smaller and smaller swings (across the valley each step multiplies the distance by about $1 - \eta S = -0.82$), and settles on a sharp minimum, $S = 9.10$, after 65 steps. The loss falls at every step.",
        loss: Loss::Two,
        eta: 0.2,
        start: SHARP,
    },
    Trial {
        label: "a longer step",
        look_for: r"The same start with $\eta = 0.25$: now $2/\eta = 8$, the yellow stretch ends short of the start, and the floor beneath it is too sharp to stay on. Each step throws the point further across the valley: the loss rises 15 steps in a row, from 0.00125 to 0.37, while the two dots, one on each wall, slide along the valley towards its middle. Once the floor beneath them is under 8 the swings die away, and it settles after 57 steps on a much flatter minimum, $S = 6.71$, well inside the stretch. Gradient flow (dashed) ends at $S = 9.10$.",
        loss: Loss::Two,
        eta: 0.25,
        start: SHARP,
    },
    Trial {
        label: "longer still",
        look_for: r"$\eta = 0.3$: the swings are larger (the loss rises 8 steps in a row, up to 1.95) and it is carried further, to $S = 3.17$ after 15 steps, less than half of $2/\eta = 6.67$. The lowest chart shows where a run from this start ends for every step: up to about 0.22 ($2/9.10$), where gradient flow ends; from there on, the longer the step the flatter the end, down to the flattest minimum, $S = 2.00$, at 0.36. Beyond that it lands here and there, some runs across the middle on the other arm, some on the floor's other half, where $a$ and $b$ are both negative; and with any step over 0.44 it is thrown off.",
        loss: Loss::Two,
        eta: 0.3,
        start: SHARP,
    },
    Trial {
        label: "the other loss",
        look_for: r"The second loss from the same start, with the step halved to $\eta = 0.125$. Its floor is twice as sharp everywhere, $S = 2\sqrt{(a^2 - b^2)^2 + 4}$, so the yellow stretch is the one of the second experiment, and the floor beneath the start, $S = 18.20$, is too sharp by the same factor. This time the swing stops growing while it is small (the loss rises at most 3 steps in a row), and the dots creep along the valley for hundreds of steps. It settles after 2105 steps at $S = 15.94$: barely inside the end of the stretch, $2/\eta = 16$, where the first loss went far inside. In the lowest chart, every step too long for gradient flow's end that still settles ends a little under $2/\eta$, within 1 %.",
        loss: Loss::Four,
        eta: 0.125,
        start: SHARP,
    },
    Trial {
        label: "too long for any minimum",
        look_for: r"From near the flat middle, where a step of 0.25 would settle beside gradient flow at $S = 2.18$. But $\eta = 1.1$ asks for $S < 2/\eta = 1.82$, and the flattest minimum has $S = 2$: no part of the floor is flat enough, and the stretch is gone. It is not thrown off. The point moves to the line $a = b$ and ends bouncing for ever between $(0.81, 0.81)$ and $(1.12, 1.12)$, one on each side of the floor, its loss 0.030 and 0.057 by turns. From the sharp start the same step throws it off after 4 steps.",
        loss: Loss::Two,
        eta: 1.1,
        start: MIDDLE,
    },
];

/// How long a hand must rest before what waits for it is done (seconds).
const HAND_REST: f64 = 0.25;
/// How fast the dots follow the film: each of two eases covers two thirds of the way in
/// this many seconds (fast enough to keep with the film, slow enough that a run
/// changed by hand, which may end on another arm, is not a jump).
const DOTS_FOLLOW: f64 = 0.035;

/// The experiment the scene opens with: the bounce.
const OPENING: usize = 1;

/// A start kept to where the weights are drawn, and a number for one that is not one.
fn within_start(z: DVec2) -> DVec2 {
    let keep = |v: f64| if v.is_finite() { v.max(-START_MAX).min(START_MAX) } else { 0.0 };
    DVec2::new(keep(z.x), keep(z.y))
}

const START_MAX: f64 = 6.0;

fn within_eta(eta: f64) -> f64 {
    if eta.is_finite() {
        eta.max(ETA_LO).min(ETA_HI)
    } else {
        TRIALS[OPENING].eta
    }
}

// ----- what glides ----------------------------------------------------------------------

/// The start, the step (as its logarithm, so that it glides evenly on the slider's
/// scale) and the loss (0 the first, 1 the second, in between while one becomes the
/// other).
#[derive(Clone, Copy, Debug, PartialEq)]
struct Setting {
    start: DVec2,
    ln_eta: f64,
    mix: f64,
}

impl Setting {
    fn of(loss: Loss, eta: f64, start: DVec2) -> Setting {
        Setting { start, ln_eta: eta.ln(), mix: loss.index() as f64 }
    }

    fn numbers(&self) -> [f64; 4] {
        [self.start.x, self.start.y, self.ln_eta, self.mix]
    }

    fn from_numbers(v: [f64; 4]) -> Setting {
        Setting { start: DVec2::new(v[0], v[1]), ln_eta: v[2], mix: v[3] }
    }

    fn eta(&self) -> f64 {
        self.ln_eta.exp()
    }

    /// The floor's scale, `Loss::scale`, between the two losses' while one becomes the
    /// other: on a logarithmic scale, as the step glides, so that where the two keep
    /// 2/η over the scale the same (the fourth experiment), the stretch stays put.
    fn scale(&self) -> f64 {
        2f64.powf(self.mix.clamp(0.0, 1.0))
    }
}

/// How fast a change of setting is glided to: each of the two eases one after the
/// other covers two thirds of what is left in this many seconds.
const FOLLOW: f64 = 0.11;

/// A setting as asked for, and as drawn: the drawn one follows it through `between`,
/// two eases one after the other, so that it sets off gently and arrives exactly.
#[derive(Clone, Copy, Debug)]
struct Glide {
    to: Setting,
    between: Setting,
    shown: Setting,
}

impl Glide {
    fn at(s: Setting) -> Glide {
        Glide { to: s, between: s, shown: s }
    }

    fn arrived(&self) -> bool {
        self.shown == self.to && self.between == self.to
    }

    /// Change part of the setting at once, in all three: what a hand moves.
    fn now(&mut self, change: impl Fn(&mut Setting)) {
        for s in [&mut self.to, &mut self.between, &mut self.shown] {
            change(s);
        }
    }

    fn step(&mut self, dt: f64) {
        if self.arrived() || dt <= 0.0 {
            return;
        }
        let k = 1.0 - (-dt / FOLLOW).exp();
        let (to, mut between, mut shown) = (self.to.numbers(), self.between.numbers(), self.shown.numbers());
        let mut apart = 0.0f64;
        for i in 0..4 {
            between[i] += (to[i] - between[i]) * k;
            shown[i] += (between[i] - shown[i]) * k;
            apart = apart.max((to[i] - between[i]).abs()).max((to[i] - shown[i]).abs());
        }
        // (exactly there in the end: the picture then stops changing)
        if apart < 2e-4 || !apart.is_finite() {
            *self = Glide::at(self.to);
        } else {
            (self.between, self.shown) = (Setting::from_numbers(between), Setting::from_numbers(shown));
        }
    }
}

// ----- where things go ------------------------------------------------------------------

const PAD: f32 = 12.0;
const BUTTON_H: f32 = 26.0;
/// A chart's title line, and the room left of it for its numbers and below it for its
/// step numbers.
const TITLE_H: f32 = 22.0;
const LEFT: f32 = 38.0;
const BELOW: f32 = 18.0;
/// Between one chart and the next.
const CHART_GAP: f32 = 10.0;
/// The key's line at the foot of the panel on a phone.
const KEY_H: f32 = 18.0;

/// The panel drawn on the picture: the buttons that choose the loss and play the film,
/// and the charts. In logical pixels.
#[derive(Clone, Copy, Debug)]
struct Board {
    min: Vec2,
    max: Vec2,
    buttons: [plot::Rect; 3],
    /// Each chart's title (its top-left corner) and rectangle: the loss, the sharpness,
    /// and where runs end (None: no room for it).
    charts: [Option<(Vec2, plot::Rect)>; 3],
    /// The key, on a phone, where the picture has no room for it.
    key: Option<Vec2>,
}

#[derive(Clone, Copy, Debug)]
struct Layout {
    /// A phone held upright: the panel goes below the picture.
    narrow: bool,
    /// The part of the canvas the picture has to itself.
    picture: (Vec2, Vec2),
    /// None on a canvas too small to spare the room (the numbers are in the side panel).
    board: Option<Board>,
}

impl Layout {
    fn of(view: Vec2, inset: f32) -> Layout {
        let narrow = view.x < 640.0 && view.y >= 1.15 * view.x && view.y >= 480.0;
        let whole = Layout { narrow, picture: (Vec2::new(0.0, inset), view), board: None };
        // what a chart takes besides its own height: its title, its step numbers and the
        // gap to the next (less on a phone)
        let (title_h, below, gap) = if narrow { (15.0, 16.0, 5.0) } else { (TITLE_H, BELOW, CHART_GAP) };
        let extra = title_h + below + gap;
        let place = |min: Vec2, width: f32, heights: &[f32], key: bool| {
            let (x, right) = (min.x + PAD, min.x + width - PAD);
            let mut y = min.y + PAD;
            let buttons = plot::even_row(3, x, right, y, BUTTON_H, 8.0);
            y += BUTTON_H + 10.0;
            let mut charts = [None; 3];
            for (k, h) in heights.iter().enumerate() {
                let title = Vec2::new(x, y);
                let top = y + title_h;
                charts[k] = Some((title, (Vec2::new(x + LEFT, top), Vec2::new(right - 6.0, top + h))));
                y = top + h + below + gap;
            }
            let key = key.then(|| Vec2::new(x, y - gap + 2.0));
            let bottom = y - gap + if key.is_some() { KEY_H } else { 0.0 } + PAD;
            Board { min, max: Vec2::new(min.x + width, bottom), buttons: [buttons[0], buttons[1], buttons[2]], charts, key }
        };
        let top_part = PAD + BUTTON_H + 10.0;
        if narrow {
            // the picture keeps at least 46 % of the canvas; the charts share the rest
            let room = view.y - (view.y * 0.46).max(inset + 220.0) - top_part - KEY_H - PAD + gap;
            for n in [3usize, 2] {
                let h = (room / n as f32 - extra).min(112.0);
                if h >= 54.0 {
                    let used = top_part + n as f32 * (h + extra) - gap + KEY_H + PAD;
                    let board = place(Vec2::new(0.0, view.y - used), view.x, &vec![h; n], true);
                    return Layout { narrow, picture: (Vec2::new(0.0, inset), Vec2::new(view.x, board.min.y)), board: Some(board) };
                }
            }
            return whole;
        }
        if view.x < 600.0 || view.y < 380.0 {
            return whole;
        }
        let width = (view.x * 0.31).clamp(290.0, 420.0);
        let top = inset + 12.0;
        let room = view.y - 12.0 - top - top_part - PAD + gap;
        for n in [3usize, 2] {
            let h = (room / n as f32 - extra).min(0.72 * width);
            if h >= 64.0 {
                // (against the right edge: a strip of the picture beside it would show lines
                // running out from under it)
                let board = place(Vec2::new(view.x - width, top), width, &vec![h; n], false);
                return Layout { narrow, picture: (Vec2::new(0.0, inset), Vec2::new(board.min.x - 8.0, view.y)), board: Some(board) };
            }
        }
        whole
    }

    /// The part of the canvas the plane's frame is fitted into: the picture's, less the
    /// heading at its top and the caption at its foot.
    fn frame_room(&self) -> (Vec2, Vec2) {
        let (min, max) = self.picture;
        let foot = if self.narrow { 10.0 } else { 52.0 };
        (min + Vec2::new(14.0, 58.0), max - Vec2::new(14.0, foot))
    }

    /// The part of the plane the view is fitted to: the first quarter, with both arms of
    /// the floor out to where the experiments' stretches end; on a phone, the one arm
    /// where the experiments happen, and the middle.
    fn frame(&self) -> (Vec2, Vec2) {
        if self.narrow {
            (Vec2::new(2.05, 0.6), Vec2::new(1.35, 0.85))
        } else {
            (Vec2::new(1.8, 1.225), Vec2::new(1.7, 1.775))
        }
    }
}

/// A loss for the panel: to three places, or as a power of ten where it is smaller.
fn loss_text(v: f64) -> String {
    if !v.is_finite() {
        "?".into()
    } else if v == 0.0 || v >= 0.01 {
        num(v, 3)
    } else {
        format!("${:.2e}$", v)
    }
}

/// A number to at most `decimals` places, without the zeros at its end: 8, 0.25, 6.67.
fn short(v: f64, decimals: usize) -> String {
    let s = num(v, decimals);
    if s.contains('.') {
        s.trim_end_matches('0').trim_end_matches('.').to_string()
    } else {
        s
    }
}

fn inside_rect(p: Vec2, (min, max): (Vec2, Vec2)) -> bool {
    p.cmpge(min).all() && p.cmple(max).all()
}

fn v3(p: DVec2) -> Vec3 {
    Vec3::new(p.x as f32, p.y as f32, 0.0)
}

// ----- what is drawn in batches of its own ----------------------------------------------

/// The parts of the plane drawn as lines, each in a batch of its own (different from
/// the usual one by a hair of its width), so that `eidos motion` can follow each part
/// point for point from frame to frame: the parts whose number of points changes (the
/// path as the film draws it, the dashes of gradient flow) do not upset the others.
#[derive(Clone, Copy, PartialEq)]
enum Part {
    Levels = 1,
    Floor = 2,
    Stretch = 3,
    Flow = 4,
    Old = 5,
    Path = 6,
}

const PARTS: [Part; 6] = [Part::Levels, Part::Stretch, Part::Floor, Part::Flow, Part::Old, Part::Path];

fn lines_for<'a>(f: &'a mut Frame, part: Part) -> &'a mut crate::gfx::draw::LineBatch {
    f.dl.lines_with(LineKind::Blend, LineStyle { width_scale: 1.0 + 1e-4 * part as u8 as f32, ..Default::default() })
}

/// A batch of lines that has nothing in it this frame is given a point that cannot be
/// seen: a batch left empty is dropped, and every batch drawn after it would then be
/// compared with the one before it in the frame before.
fn keep_batch(batch: &mut crate::gfx::draw::LineBatch) {
    if batch.pts.len() <= 1 {
        batch.segment(Vec3::ZERO, Vec3::ZERO, 0.0, Color::CLEAR);
    }
}

/// The dots: each iterate's, which grow in number as the film plays, and the two that
/// move, which do not.
fn iterate_dots<'a>(f: &'a mut Frame) -> &'a mut crate::gfx::draw::PointBatch {
    f.dl.points_with(PointKind::Disc, PointStyle { size_scale: 1.0001, ..Default::default() })
}

fn moving_dots<'a>(f: &'a mut Frame) -> &'a mut crate::gfx::draw::PointBatch {
    f.dl.points_with(PointKind::Overlay, PointStyle { outline_px: 1.5, ..Default::default() })
}

/// The line kinds of the charts (see `plot::apart`): the curves that grow as the film
/// plays, each in a kind of its own.
const LOSS_CURVE: u8 = plot::OWN;
const AT_POINT: u8 = plot::OWN + 1;
const UNDER: u8 = plot::OWN + 2;
const FADING: u8 = plot::OWN + 3;
const SWEEP_MARKS: u8 = plot::OWN + 4;
const CHART_KINDS: [u8; 9] = [plot::AXES, plot::CURVES, plot::MARKS, plot::BUTTONS, LOSS_CURVE, AT_POINT, UNDER, FADING, SWEEP_MARKS];

/// The sharpnesses the lowest chart spans (logarithmically).
const S_LO: f64 = 1.0;
const S_HI: f64 = 100.0;

/// The loss's level lines drawn: on a logarithmic scale, as the loss chart is.
const LEVELS: [f64; 5] = [0.01, 0.1, 0.5, 2.0, 8.0];

/// The colour of the floor where its sharpness is `s` (`flattest` where it is least):
/// brighter where it is sharper, on a scale of doublings; `seen` of it.
fn floor_color(s: f64, flattest: f64, seen: f32) -> Color {
    let t = ((s / flattest).max(1.0).log2() / 5.0).min(1.0) as f32;
    CURVATURE.a((0.42 + 0.58 * t) * seen)
}

/// x on the step charts: log₁₀(1 + n), so that the first steps and the thousandth
/// both have room.
fn step_x(n: f64) -> f64 {
    (1.0 + n.max(0.0)).log10()
}

/// A run's values at each step, `values[i]` at x = step_x(i), as a step chart draws
/// them: one point for each column of pixels across the chart, kept inside it. Where
/// the steps are further apart than a column, the point is on the straight line from
/// one step to the next; where several steps fall in one column, far along the
/// logarithmic axis, the column has four, the first, lowest, highest and last of them,
/// so that a value that swings from step to step shows as a band as wide as its
/// swings; and past the run's last step the line stays at its last value. So the
/// curve has the same number of points whatever the run, and when a hand changes the
/// run its points move with it instead of being dealt out afresh. Only the columns up
/// to step `upto` are drawn (a column that steps after it still fall in, only to its
/// first: drawn whole it would change as they came). A value that is not a number
/// breaks the line.
fn columns(chart: &Plot, values: &[f64], upto: usize) -> Vec<Vec2> {
    let mut out: Vec<Vec2> = Vec::new();
    let Some(&end) = values.last() else { return out };
    let last = values.len() - 1;
    let (left, width) = (chart.rect.0.x, chart.width());
    if !(width.is_finite() && width >= 1.0) {
        return out;
    }
    let value = |i: usize| if i <= last { values[i] } else { end };
    let y = |v: f64| if v.is_finite() { chart.px_y(v).max(chart.rect.0.y).min(chart.rect.1.y) } else { f32::NAN };
    let column = |i: usize| (chart.px_x(step_x(i as f64)) - left).floor();
    let cols = width.ceil() as usize;
    // the first step at or after each column; steps past the run's end count, with its
    // last value, so that the columns are the same for every run
    let mut i = 0usize;
    for c in 0..cols {
        let cf = c as f32;
        while column(i) < cf {
            i += 1;
        }
        let x = left + cf + 0.5;
        if column(i) > cf {
            // no step in this column: on the line between the steps either side
            if i > upto {
                break;
            }
            let (x0, x1) = (step_x((i - 1) as f64), step_x(i as f64));
            let t = (chart.x_at(x) - x0) / (x1 - x0);
            out.push(Vec2::new(x, y(value(i - 1) + (value(i) - value(i - 1)) * t)));
            continue;
        }
        // the steps in this column
        let mut j = i;
        while column(j + 1) <= cf {
            j += 1;
        }
        if i > upto {
            break;
        }
        if j > upto {
            out.push(Vec2::new(x, y(value(i))));
            break;
        }
        let (mut lo, mut hi) = (f64::INFINITY, f64::NEG_INFINITY);
        for k in i..=j {
            lo = lo.min(value(k));
            hi = hi.max(value(k));
        }
        out.extend([value(i), lo, hi, value(j)].map(|v| Vec2::new(x, y(v))));
        i = j + 1;
    }
    out
}

/// A run's values at each step on a step chart, as far as the film has got: the
/// steps it has passed, and the next fading in as it comes to it (the same as on the
/// plane: a line whose end ran up and down after a swinging value would jerk).
#[allow(clippy::too_many_arguments)]
fn film_curve(f: &mut Frame, chart: &Plot, kind: u8, values: &[f64], film: f64, width: f32, color: Color) {
    let last = values.len().saturating_sub(1);
    let n = (film.max(0.0).floor() as usize).min(last);
    // (the whole width once the film is done)
    let upto = if n >= last { usize::MAX } else { n };
    let pts = columns(chart, values, upto);
    plot::path_of(f, kind, &pts, width, color);
    let next = (film - n as f64) as f32;
    if n < last && next > 0.0 {
        let (a, b) = (chart.at(step_x(n as f64), values[n]), chart.at(step_x((n + 1) as f64), values[n + 1]));
        // (only where the steps are apart: where they crowd, the column takes them)
        if (b.x - a.x).abs() >= 1.0 {
            if let Some((p, q)) = plot::clip(&[a, b], chart.rect).windows(2).find(|w| w[0].is_finite() && w[1].is_finite()).map(|w| (w[0], w[1])) {
                plot::line_of(f, kind, p, q, width, color.a(next));
            }
        }
    }
}

/// Ticks for a logarithmic axis: the round values 1, 2 and 5 times a power of ten
/// between `lo` and `hi`, keeping only 1 and those of 1 and 5 where there is no room.
fn log_ticks(lo: f64, hi: f64, pixels: f32) -> Vec<f64> {
    let decades = (hi / lo).log10().max(1e-9);
    let per_decade = pixels as f64 / decades;
    let leads: &[f64] = if per_decade >= 150.0 {
        &[1.0, 2.0, 5.0]
    } else if per_decade >= 70.0 {
        &[1.0, 3.0]
    } else {
        &[1.0]
    };
    let mut out = Vec::new();
    let mut power = 10f64.powf(lo.log10().floor() - 1.0);
    while power <= hi * 1.0001 && out.len() < 40 {
        for lead in leads {
            let v = lead * power;
            if v >= lo * 0.9999 && v <= hi * 1.0001 {
                out.push(v);
            }
        }
        power *= 10.0;
    }
    out
}

/// A tick's number on a logarithmic axis: as few digits as it needs.
fn log_tick_text(v: f64) -> String {
    if v >= 1.0 {
        num(v, 0)
    } else {
        let decimals = (-v.log10()).ceil().max(0.0) as usize;
        num(v, decimals)
    }
}

// ----- the scene ------------------------------------------------------------------------

pub struct EosToy {
    /// What is asked for: the loss, the step and the start; and the run they make.
    loss: Loss,
    eta: f64,
    start: DVec2,
    run: Run,
    /// Where runs from the same start end, for every step of the sweep.
    sweep: Sweep,
    glide: Glide,
    /// How far the film has got, in steps (with a fraction of the next).
    film: f64,
    playing: bool,
    /// Play once the glide to a new setting has arrived.
    play_when_there: bool,
    /// The last run, fading away while a new setting glides in.
    old: Option<Fading>,
    /// The step charts' ranges as drawn: [log₁₀(1 + steps), the loss's lowest and
    /// highest power of ten, the top of the sharpness], gliding to a new run's.
    ranges: plot::Glide,
    ranges_moving: bool,
    /// The scene's own clock (seconds of real time it has been drawn for), and when a
    /// hand last moved the start or the step: what a hand moves is followed at once,
    /// and what costs more (the sweep) or would jump (the charts' ranges) waits until
    /// the hand has rested a moment.
    clock: f64,
    hand_at: f64,
    /// The sweep is for another start than the one asked for: done again once the
    /// hand rests (and drawn faint till then).
    sweep_stale: bool,
    /// The two dots as drawn, following where the film has them through two eases (a
    /// run changed by hand can end somewhere else entirely).
    drawn: [DVec2; 2],
    between: [DVec2; 2],
    /// Where the name of the run's end goes (from its point), kept while it is not
    /// yet written.
    end_place: Option<Vec2>,
    /// The canvas last drawn on and how much of its top the application's buttons
    /// cover: they decide where the panel goes, and so where the picture is.
    canvas: Vec2,
    inset: f32,
    measured: bool,
    new_view: bool,
}

/// A run that is fading away, with how far its film had got, how much of it is still
/// seen, and the step charts' ranges it was drawn on.
struct Fading {
    run: Run,
    film: f64,
    seen: f32,
    ranges: [f32; 4],
}

/// The ranges of the step charts for a run.
fn ranges_of(run: &Run) -> [f32; 4] {
    let x_hi = step_x(run.steps().max(10) as f64);
    // (what is thrown off goes up without end: the chart stops at 10⁴)
    let (mut lo, mut hi) = (f64::MAX, f64::MIN);
    for v in run.value.iter().filter(|v| v.is_finite() && **v > 0.0) {
        lo = lo.min(v.log10());
        hi = hi.max(v.log10());
    }
    if lo > hi {
        (lo, hi) = (-1.0, 0.0);
    }
    // (down to 10⁻¹²: a run stops when its loss is under 10⁻¹⁴, and the last steps, which
    // depend on where exactly it stopped, stay under the chart's floor)
    let lo = lo.max(-12.0).floor();
    let hi = hi.min(4.0).ceil().max(lo + 2.0);
    // the sharpness: up to the floor beneath, and to 2/η where that is not far above
    let first = run.under[0];
    let under_most = run.under.iter().copied().filter(|v| v.is_finite()).fold(0.0f64, f64::max).min(4.0 * first.max(2.0 / run.eta).max(1.0));
    let limit = 2.0 / run.eta;
    let top = under_most.max(if limit <= 2.2 * under_most.max(1.0) { limit } else { 0.0 }).max(run.loss.flattest()) * 1.1;
    let step = plot::round_step(top / 6.0);
    let top = (top / step).ceil() * step;
    [x_hi as f32, lo as f32, hi as f32, top as f32]
}

impl EosToy {
    fn new() -> Self {
        let t = &TRIALS[OPENING];
        let (loss, eta, start) = (t.loss, t.eta, DVec2::from(t.start));
        let run = Run::new(loss, eta, start);
        let mut ranges = plot::Glide::default();
        ranges.snap(&ranges_of(&run));
        EosToy {
            loss,
            eta,
            start,
            run,
            sweep: Sweep::new(loss, start),
            glide: Glide::at(Setting::of(loss, eta, start)),
            film: 0.0,
            playing: true,
            play_when_there: false,
            old: None,
            ranges,
            ranges_moving: false,
            clock: 0.0,
            hand_at: f64::NEG_INFINITY,
            sweep_stale: false,
            drawn: [start; 2],
            between: [start; 2],
            end_place: None,
            canvas: Vec2::new(1100.0, 1000.0),
            inset: 0.0,
            measured: false,
            new_view: false,
        }
    }

    /// Set up experiment `k` of `TRIALS`.
    fn set_up(&mut self, k: usize) {
        let t = &TRIALS[k.min(TRIALS.len() - 1)];
        self.ask(t.loss, t.eta, DVec2::from(t.start));
    }

    /// Ask for a new setting (an experiment, the other loss): the path drawn fades, the
    /// start, the step and the loss glide to the new ones, and then the new run's film
    /// plays from its start.
    fn ask(&mut self, loss: Loss, eta: f64, start: DVec2) {
        let (eta, start) = (within_eta(eta), within_start(start));
        if self.film > 0.0 || self.old.is_none() {
            self.old = Some(self.fading());
        }
        (self.loss, self.eta, self.start) = (loss, eta, start);
        self.run = Run::new(loss, eta, start);
        if self.sweep.loss != loss || self.sweep.start != start {
            self.sweep = Sweep::new(loss, start);
        }
        self.sweep_stale = false;
        self.glide.to = Setting::of(loss, eta, start);
        self.film = 0.0;
        (self.playing, self.play_when_there) = (false, true);
    }

    /// The start or the step moved by hand: at once, and the whole run shown (the
    /// film at its end); play shows it again from the start.
    fn hand(&mut self, eta: f64, start: DVec2) {
        let (eta, start) = (within_eta(eta), within_start(start));
        (self.eta, self.start) = (eta, start);
        self.glide.now(|s| {
            s.start = start;
            s.ln_eta = eta.ln();
        });
        self.run = Run::new(self.loss, eta, start);
        self.sweep_stale |= self.sweep.start != start || self.sweep.loss != self.loss;
        self.film = self.run.steps() as f64;
        (self.playing, self.play_when_there) = (false, false);
        self.old = None;
        self.hand_at = self.clock;
    }

    /// Has a hand moved the start or the step in the last moment?
    fn hand_moving(&self) -> bool {
        self.clock - self.hand_at < HAND_REST
    }

    /// Play from the start, or pause, or play on from where the film was paused.
    fn play(&mut self) {
        if self.playing || self.play_when_there {
            (self.playing, self.play_when_there) = (false, false);
        } else if self.film >= self.run.steps() as f64 {
            self.old = Some(self.fading());
            self.film = 0.0;
            self.play_when_there = true;
        } else {
            self.playing = true;
        }
    }

    /// The run as it is drawn now, to fade away.
    fn fading(&self) -> Fading {
        Fading { run: self.run.clone(), film: self.film, seen: 1.0, ranges: [0, 1, 2, 3].map(|i| self.ranges.get(i)) }
    }

    fn busy(&self) -> bool {
        self.playing || self.play_when_there
    }

    /// Move everything on: `dt` seconds of the scene's time (the film) and `real_dt` of
    /// the clock's (a change of setting glides by the clock, so that it is seen through
    /// even while the scene's time stands still).
    fn advance(&mut self, dt: f32, real_dt: f32) {
        self.glide.step(real_dt as f64);
        if let Some(old) = &mut self.old {
            old.seen *= (-real_dt.max(0.0) / 0.1).exp();
            if old.seen < 0.02 {
                self.old = None;
            }
        }
        if self.play_when_there && self.old.is_none() && self.glide.arrived() {
            (self.play_when_there, self.playing) = (false, true);
        }
        let end = self.run.steps() as f64;
        if self.playing {
            self.film = film_on(self.film, dt as f64).min(end);
            if self.film >= end {
                self.playing = false;
            }
        }
        self.clock += real_dt.max(0.0) as f64;
        let resting = !self.hand_moving();
        self.ranges_moving = resting && self.ranges.follow(&ranges_of(&self.run), real_dt, false);
        if self.sweep_stale && resting {
            (self.sweep, self.sweep_stale) = (Sweep::new(self.loss, self.start), false);
        }
        if !self.sweep_stale {
            self.sweep.work(&crate::clock::Budget::new(0.002));
        }
        // the dots follow where the film has them
        let to = dots(&self.run.z, self.film);
        let k = 1.0 - (-(real_dt.max(0.0) as f64) / DOTS_FOLLOW).exp();
        let mut apart = 0.0f64;
        for i in 0..2 {
            self.between[i] += (to[i] - self.between[i]) * k;
            self.drawn[i] += (self.between[i] - self.drawn[i]) * k;
            apart = apart.max((to[i] - self.between[i]).length()).max((to[i] - self.drawn[i]).length());
        }
        // (exactly there in the end; and where a run went far, at once)
        if apart < 1e-4 || !apart.is_finite() || !self.drawn.iter().all(|d| d.abs().max_element() < 1e3) {
            (self.drawn, self.between) = (to, to);
        }
    }

    fn dots_moving(&self) -> bool {
        self.drawn != dots(&self.run.z, self.film) || self.between != self.drawn
    }

    fn at_rest(&self) -> bool {
        self.glide.arrived() && !self.busy() && self.old.is_none() && !self.ranges_moving && self.sweep.done() && !self.sweep_stale && !self.dots_moving()
    }

    /// The film's step, as a whole number.
    fn step_now(&self) -> usize {
        (self.film.floor().max(0.0) as usize).min(self.run.steps())
    }

    fn film_done(&self) -> bool {
        !self.busy() && self.film >= self.run.steps() as f64
    }

    /// The canonical view: the plane's frame in the middle of the part of the canvas
    /// the picture has to itself.
    fn the_view(&self) -> View {
        let lay = Layout::of(self.canvas, self.inset);
        let (min, max) = lay.frame_room();
        let (centre, half) = lay.frame();
        let room = (max - min).max(Vec2::splat(60.0));
        let per_unit = (room.x / (2.0 * half.x)).min(room.y / (2.0 * half.y)).max(1e-3);
        // (screen y runs down, the scene's up)
        let off = ((min + max) * 0.5 - self.canvas * 0.5) / per_unit;
        View::Fit { center: Vec2::new(centre.x - off.x, centre.y + off.y), half: self.canvas.max(Vec2::ONE) * (0.5 / per_unit) }
    }

    /// The heading's second line: what the film has got to.
    fn status(&self) -> String {
        let run = &self.run;
        let limit = 2.0 / self.eta;
        if self.play_when_there {
            return String::new();
        }
        if !self.film_done() {
            let k = self.step_now();
            if k == 0 {
                return "step 0".into();
            }
            let rising = run.value[k] > run.value[k - 1];
            return format!("step {k}: loss {}", if rising { "rising" } else { "falling" });
        }
        let n = run.steps();
        match run.ending {
            Ending::Settled => {
                let s = run.under[n];
                format!(r"settled after {n} steps where $S = {} < 2/\eta = {}$", num(s, 2), short(limit, 2))
            }
            // (a start exactly on a minimum too sharp to stay at: the slope is 0 there, so
            // it stays, though the least nudge would grow)
            Ending::Stuck if n == 0 && run.value[0] < SETTLED => format!(r"it never moves: the start is a minimum, too sharp for the step ($S = {} > 2/\eta$), but the slope there is 0", num(run.under[0], 2)),
            Ending::Stuck if n == 0 => "it never moves: the slope is 0 where it starts, which is not a minimum".into(),
            Ending::Stuck => format!("after {n} steps stuck where the slope is 0, not at a minimum"),
            Ending::Cycle(k) => format!("after {n} steps it bounces for ever round {k} points"),
            Ending::Restless => format!("still moving after {n} steps"),
            Ending::FlewOff => format!("thrown off after {n} steps"),
        }
    }

    /// What the run came to, in a few words, for the panel.
    fn ending_words(&self) -> String {
        let run = &self.run;
        let n = run.steps();
        match run.ending {
            Ending::Settled => format!(r"settled, ${}$, after {n} steps", tuple(&[run.last().x, run.last().y], 2)),
            Ending::Stuck => "stuck: the slope is 0".into(),
            Ending::Cycle(k) => format!("bounces round {k} points"),
            Ending::Restless => format!("still moving after {n}"),
            Ending::FlewOff => format!("thrown off after {n} steps"),
        }
    }
}

impl Scene for EosToy {
    fn view(&self) -> View {
        self.the_view()
    }

    fn take_view(&mut self) -> Option<View> {
        std::mem::take(&mut self.new_view).then(|| self.the_view())
    }

    fn settings(&self, s: &mut ViewSettings) {
        s.grid_alpha = 0.08;
        s.axis_alpha = 0.3;
    }

    fn animated(&self) -> bool {
        !self.at_rest()
    }

    fn ui(&mut self, ui: &mut egui::Ui, _: &Gfx) {
        ui::note(ui, r"Two weights, $a$ and $b$, and a loss that depends only on their product. Its minima, all of loss 0, form a curved valley floor, $ab = 1$ (green): flattest in the middle, where the sharpness $S$ is 2 (for the first loss), and sharper and sharper out along its arms. Gradient descent with step $\eta$ can come to rest only where $S < 2/\eta$ (yellow). Started above a part too sharp for it, it is not necessarily thrown off: unless the step is far too long, it bounces from wall to wall, and each bounce carries it along the valley. One step, $a \to a - \eta g b$ and $b \to b - \eta g a$, changes $a^2 - b^2$ by an exact factor:");
        ui::equation(ui, r"(a - \eta g b)^2 - (b - \eta g a)^2 \\ = (1 - \eta^2 g^2)(a^2 - b^2)");
        ui::note(ui, r"where $g = \partial L/\partial(ab)$ says how far the product is from fitting. Gradient flow keeps $a^2 - b^2$ fixed. A bounce, where $g$ is large, shrinks it, and on the floor a smaller $a^2 - b^2$ is flatter: $S = \sqrt{(a^2 - b^2)^2 + 4}$ for the first loss.");

        ui::section(ui, "Things to try");
        for (k, trial) in TRIALS.iter().enumerate() {
            if ui::experiment(ui, trial.label, trial.look_for) {
                self.set_up(k);
            }
        }

        ui::section(ui, "Loss");
        let mut which = self.loss.index();
        if ui::choice(ui, &[r"$\tfrac12(ab - 1)^2$", r"$\tfrac14(1 - a^2b^2)^2$"], &mut which) {
            self.ask(Loss::of(which), self.eta, self.start);
        }
        ui::note(ui, r"The first is a network of two weights in a row, fitted to the target 1 with squared error. The second is a published minimal example: four weights in a row, $\tfrac12(1 - w_1 w_2 w_3 w_4)^2$, started with $w_3 = w_1$ and $w_4 = w_2$, which gradient descent keeps equal; with $a = w_1$ and $b = w_2$ it is gradient descent on $\tfrac14(1 - a^2b^2)^2$, whose floor is $ab = \pm 1$ and twice as sharp.");

        ui::section(ui, "Step");
        let mut eta = self.eta as f32;
        if ui::slider_log(ui, r"step $\eta$", &mut eta, ETA_LO as f32..=ETA_HI as f32) {
            self.hand(eta as f64, self.start);
        }
        ui::note(ui, r"The floor is flat enough where $S < 2/\eta$. Pressing on the lowest chart chooses the step too.");
        ui.horizontal(|ui| {
            let text = if self.busy() { "❚❚ pause" } else if self.film_done() { "▶ play again" } else { "▶ play on" };
            if ui.button(ui::label_text(ui, text, 12.5, None, false)).clicked() {
                self.play();
            }
        });
        let mut start = [self.start.x, self.start.y];
        let mut moved = false;
        ui.horizontal(|ui| {
            for (k, name) in [r"start $a$", "$b$"].into_iter().enumerate() {
                ui.label(ui::label_text(ui, name, 12.5, Some(pal::DIM), false));
                moved |= ui.add(egui::DragValue::new(&mut start[k]).speed(0.01).fixed_decimals(2).range(-START_MAX..=START_MAX)).changed();
            }
        });
        if moved {
            self.hand(self.eta, DVec2::from(start));
        }
        ui::note(ui, "Or drag the cyan dot in the picture.");

        ui::section(ui, "Reading the picture");
        ui::key(
            ui,
            &[
                (SGD, "cyan: gradient descent's path from the start (drag it); the two dots are where it is after an even and after an odd number of steps, one on each wall while it bounces"),
                (SGD.a(0.55), "dashed: gradient flow from the same start, with infinitely small steps: straight down onto the floor"),
                (CURVATURE, r"green: the floor, brighter where it is sharper; on the charts, the sharpness $S$"),
                (LIMIT, r"yellow: the stretch of floor flat enough for this step, $S < 2/\eta$; on the charts, the level $2/\eta$"),
                (pal::DIM, "grey: level lines of the loss"),
            ],
        );
        ui::note(ui, r"The charts: the loss at each step (logarithmic); the sharpness at the point (faint) and of the floor beneath it (bold); and, for every step $\eta$ from 0.01 to 2, where a run from this start ends (thrown off along the top, never settling along the foot).");

        ui::section(ui, "Words");
        ui::term(ui, "Sharpness $S$:", "the larger eigenvalue of the Hessian, the matrix of second derivatives: the curvature in the direction where the loss curves most. On the floor the other eigenvalue is 0, since along the floor the loss does not change.");
        ui::term(ui, "Valley floor:", r"the minima, the curve $ab = 1$ (and $ab = -1$ for the second loss). There $S = a^2 + b^2 = \sqrt{(a^2 - b^2)^2 + 4}$ for the first loss and twice that for the second: least where $|a| = |b|$, and growing without bound out along each arm.");
        ui::term(ui, "Flat enough:", r"$S < 2/\eta$. Near a minimum the loss is nearly a bowl of curvature $S$ across the valley, and on a bowl gradient descent settles only if $\eta S < 2$ (scene stepsize).");
        ui::term(ui, "Gradient flow:", r"gradient descent with infinitely small steps: a path straight down the slope. Here it keeps $a^2 - b^2$ fixed, so it comes down onto the floor at the point with its start's $a^2 - b^2$ (for the first loss from the line $a = -b$, and for the second from an axis, it never reaches the floor).");
        ui::term(ui, "Floor beneath:", r"where gradient flow from the point would end. Its sharpness depends only on $a^2 - b^2$.");
        ui::term(ui, r"$g$:", r"$\partial L/\partial(ab)$: $ab - 1$ for the first loss, $ab(a^2b^2 - 1)$ for the second; $\nabla L = g\,(b, a)$.");
        ui::term(ui, "Catapult:", "a name from published work for a run whose loss grows for a while as the curvature falls, and that then settles on a flatter minimum.");

        ui::section(ui, "Numbers");
        let run = &self.run;
        ui::readout_colored(ui, r"$2/\eta$", short(2.0 / self.eta, 3), LIMIT);
        ui::readout_colored(ui, "flattest minimum", short(self.loss.flattest(), 2), CURVATURE);
        ui::readout_colored(ui, "$S$ at the start", num(run.sharp[0], 2), CURVATURE);
        ui::readout_colored(ui, "floor beneath the start", num(run.under[0], 2), CURVATURE);
        let k = self.step_now();
        ui::readout(ui, "step", format!("{k} of {}", run.steps()));
        ui::readout_colored(ui, "loss", loss_text(run.value[k]), SGD);
        ui::readout_colored(ui, "$S$ at the point", num(run.sharp[k], 2), CURVATURE);
        ui::readout_colored(ui, "floor beneath it", num(run.under[k], 2), CURVATURE);
        let z = run.z[k];
        ui::readout(ui, "$a^2 - b^2$", num(z.x * z.x - z.y * z.y, 3));
        ui::readout(ui, "end", self.ending_words());
        if let Some(s) = run.end_sharpness() {
            ui::readout_colored(ui, "$S$ at the end", num(s, 2), CURVATURE);
        }
        let rises = match run.rises().0 {
            0 => "never".to_string(),
            1 => "1 step at a time".to_string(),
            n => format!("{n} steps in a row"),
        };
        ui::readout(ui, "the loss rose", rises);
        if self.sweep.done() && !self.sweep_stale {
            let text = self.sweep.longest_settling().map_or("none".to_string(), |e| num(e, 3));
            ui::readout(ui, "longest step in the lowest chart that settles", text);
        }

        ui::section(ui, "What published work reports");
        ui::note(ui, r"Lewkowycz and co-authors (2020) studied a model of the first kind: a wide network with one linear hidden layer, squared loss and one example with target 0. They wrote down exactly what a step of gradient descent does to it, and analysed it for large widths. With a learning rate between $2/\lambda_0$ and $4/\lambda_0$, $\lambda_0$ the starting curvature, the loss first grows while the curvature falls below $2/\eta$, and then training converges: the catapult. In their experiments, deep ReLU networks trained with mini-batches and squared loss, they found about $12/\lambda_0$ a good predictor of the largest learning rate that still trained. Measured here, from the experiments' sharp start: every step from 0.01 to 0.4395 settles and every longer one up to 2 is thrown off (tried at intervals of 0.0005); 4 over the sharpness of the floor beneath the start is $4/9.10 = 0.4396$.");
        ui::note(ui, r"Zhu and co-authors (2022) built the second loss as a minimal example. They proved that for small enough steps gradient descent from a region of starts converges to a minimum with sharpness between $2/\eta - 20\eta/3$ and $2/\eta$, and found that losses of degree 2, like the first, end at minima significantly flatter. Measured here, far outside the steps the proof covers, from 60 starts along the floor's arm and the lowest chart's 81 steps: of the runs that started too sharp and settled, all 589 on the second loss ended inside that interval (more than half of them above 0.99 of $2/\eta$), while the 542 on the first ended between 0.20 and 0.995 of $2/\eta$, half of them below 0.70.");
        ui::note(ui, r#"Damian, Nichani and Lee (2023) proposed this mechanism for networks, and derived it under assumptions; they call it self-stabilisation: "as the iterates diverge in direction of the top eigenvector of the Hessian due to instability, the cubic term in the local Taylor expansion of the loss function causes the curvature to decrease until stability is restored". Here that push can be followed exactly: a swing that makes $g$ large shrinks $a^2 - b^2$ by the factor $1 - \eta^2 g^2$, and with it the sharpness of the floor beneath."#);
        ui::note(ui, r"Cohen and co-authors (2021) observed that when full-batch gradient descent trains a neural network, the sharpness typically rises to about $2/\eta$ and then stays right at, or a little above, it, unless training finishes first. That resembles the second loss, which ends at the edge, more than the first, which ends well inside it. It is a finding from experiments, not a theorem (scene eos).");

        ui::see_also(ui, &[("stepsize", r"on a bowl, why the step must stay under $2/\lambda$"), ("hessian", "curvature, and the Hessian's eigenvalues"), ("eos", "the same limit in a real network's training"), ("warmup", "why training often starts with short steps")]);
    }

    fn frame(&mut self, f: &mut Frame) {
        // the panel's place follows the canvas, and the picture's place follows the panel's
        let view = f.view_size();
        let lay = Layout::of(view, f.top_inset);
        let before = Layout::of(self.canvas, self.inset);
        // (a canvas squeezed to a sliver is one the controls have opened over, on a
        // phone: the picture is not seen then, and is left as it was for when they close)
        let covered = view.x < 160.0;
        let changed = !covered && (lay.narrow != before.narrow || lay.board.is_some() != before.board.is_some() || (view - self.canvas).abs().max_element() > 24.0);
        if !covered && (changed || !self.measured) {
            self.new_view |= changed;
            (self.canvas, self.measured) = (view, true);
        }
        if !covered {
            self.new_view |= (f.top_inset - self.inset).abs() > 1.0;
            self.inset = f.top_inset;
        }

        let pointer = plot::pointer(f);
        if let Some(b) = lay.board.as_ref().filter(|_| !covered) {
            if pointer.is_some_and(|p| inside_rect(p, (b.min, b.max))) {
                // the panel is for its own buttons and charts: no moving the view from there
                f.handles.capture();
                f.handles.claim_clicks();
            }
            if let Some(p) = pointer.filter(|_| f.ptr.clicked) {
                if inside_rect(p, b.buttons[0]) && self.loss != Loss::Two {
                    self.ask(Loss::Two, self.eta, self.start);
                } else if inside_rect(p, b.buttons[1]) && self.loss != Loss::Four {
                    self.ask(Loss::Four, self.eta, self.start);
                } else if inside_rect(p, b.buttons[2]) {
                    self.play();
                }
            }
            // pressing on the chart of where runs end chooses the step
            if let (Some((_, rect)), Some(p)) = (b.charts[2], pointer) {
                if f.ptr.down && !f.handles.dragging() && inside_rect(p, (rect.0 - Vec2::new(0.0, 4.0), rect.1 + Vec2::new(0.0, 4.0))) {
                    let chart = Plot::new(rect, [ETA_LO.log10(), ETA_HI.log10()], [0.0, 1.0]);
                    let eta = 10f64.powf(chart.x_at(p.x));
                    if (eta - self.eta).abs() > 1e-9 * eta {
                        self.hand(eta, self.start);
                    }
                }
            }
        }

        // the start, a handle (not where the panel covers it)
        let mut grip = v3(self.glide.shown.start).truncate();
        let under = lay.board.as_ref().is_some_and(|b| on_screen(f, grip.extend(0.0)).is_some_and(|q| inside_rect(q, (b.min, b.max))));
        if (!under || f.handles.dragging()) && !covered && f.drag2(1, &mut grip, SGD) {
            self.hand(self.eta, grip.as_dvec2());
        }
        self.advance(f.dt, f.real_dt);

        let labels_before = f.dl.labels.len();
        let plates = self.picture(f, &lay, covered);
        let numbers_from = f.dl.labels.len();
        f.axes2();
        // no number of the axes is written over the heading or the caption
        let mut k = 0;
        f.dl.labels.retain(|l| {
            k += 1;
            k <= numbers_from || !plates.iter().flatten().any(|plate| overlap(label_box(l), *plate))
        });
        if covered {
            // (the sliver left beside the controls is the picture alone)
            f.dl.labels.truncate(labels_before);
            return;
        }
        // nothing written into the picture may show through the panel
        if let Some(b) = &lay.board {
            let (lo, hi) = (b.min - Vec2::splat(2.0), b.max + Vec2::splat(2.0));
            let mut k = 0;
            f.dl.labels.retain(|l| {
                k += 1;
                k <= labels_before || l.backed || !inside_rect(l.pos, (lo, hi))
            });
            self.board(f, b);
        }
    }
}

/// The box a label takes on the screen, nearly: its text as wide as it is reckoned to be.
fn label_box(l: &crate::gfx::draw::Label) -> (Vec2, Vec2) {
    let size = Vec2::new(text_width(&l.text, l.size), l.size + 2.0);
    let share = |a: egui::Align| match a {
        egui::Align::Min => 0.0,
        egui::Align::Center => 0.5,
        egui::Align::Max => 1.0,
    };
    let min = l.pos - size * Vec2::new(share(l.align.x()), share(l.align.y()));
    (min, min + size)
}

/// Whether two boxes on the screen meet.
fn overlap(a: (Vec2, Vec2), b: (Vec2, Vec2)) -> bool {
    a.0.x < b.1.x && b.0.x < a.1.x && a.0.y < b.1.y && b.0.y < a.1.y
}

/// A name for the scene's point `at`, where there is room: beside it, on the side
/// `towards` (a direction on the screen) or the other; failing that further out that
/// way or the other, joined to it by a thin line; and failing that, wherever is
/// clearest. (Where the picture is crowded, as it is round a bouncing path, a name
/// beside its point would sit on the path.) `lines`: whether a line may be drawn to
/// it; `shown`: whether it is written, or only its place kept; `hold`: keep it where it
/// was even where it is crowded (while a hand moves things through it).
#[allow(clippy::too_many_arguments)]
fn leader_label(f: &mut Frame, room: &mut Room, name: &'static str, at: Vec3, text: &str, size: f32, color: Color, towards: Vec2, lines: bool, shown: bool, hold: bool) {
    let mut kept = f.places.chosen(name, 0);
    leader_label_kept(f, room, name, at, text, size, color, towards, lines, shown, hold, &mut kept);
}

/// The same, keeping its place in `kept` (an offset from its point) instead of where
/// the frame remembers what it drew: for a name not shown yet, whose place must hold
/// still all the same, or the names round it would move.
#[allow(clippy::too_many_arguments)]
fn leader_label_kept(f: &mut Frame, room: &mut Room, name: &'static str, at: Vec3, text: &str, size: f32, color: Color, towards: Vec2, lines: bool, shown: bool, hold: bool, place: &mut Option<Vec2>) {
    // (a point off the picture has no name on it)
    let Some(p) = on_screen(f, at).filter(|p| p.cmpge(Vec2::ZERO).all() && p.cmple(f.view_size()).all()) else { return };
    let half = f.text_size(text, size) * 0.5 + Vec2::new(4.0, 2.0);
    let dir = towards.normalize_or(Vec2::new(0.0, -1.0));
    // (the box's nearest point to p about `gap` away along the direction)
    let centre = |d: Vec2, gap: f32| p + d * gap + Vec2::new(d.x * half.x, d.y * half.y);
    // where it was last frame, for as long as there is still a little room there (a
    // name that looked afresh in every frame would hop between two places as good)
    // (and, with `hold`, while a hand moves the picture, wherever it can still be seen)
    let enough = if hold { -0.5 } else { 3.0 };
    let kept = place.filter(|off| off.is_finite() && room.room_for(p, p + *off, half) >= enough);
    let mut choice = kept.map(|off| {
        let d = off.normalize_or(dir);
        let gap = if off.length() > half.length() + 24.0 { 48.0 } else { 12.0 };
        (p + off, d, gap)
    });
    'out: for gap in [12.0, 48.0, 80.0, 112.0] {
        if choice.is_some() {
            break;
        }
        for d in [dir, -dir] {
            let c = centre(d, gap);
            if room.room_for(p, c, half) >= 5.0 {
                choice = Some((c, d, gap));
                break 'out;
            }
        }
    }
    let Some((c, d, gap)) = choice else {
        if shown {
            room.label(f, name, at, text, size, color, towards, 12.0, false);
        }
        *place = None;
        return;
    };
    *place = Some(c - p);
    room.boxed(c - half, c + half);
    // the line to a name further out, stopping short of both (its way kept free even
    // while it is not drawn)
    let line = (gap > 12.0).then(|| (p + d * 7.0, c - Vec2::new(d.x * half.x, d.y * half.y) - d * 2.0));
    if let Some((from, to)) = line {
        room.segment_px(from, to);
    }
    // (a name not shown yet keeps its place free, so that the others need not move
    // out of its way when it comes)
    if !shown {
        return;
    }
    // (the line only once the name has its place: drawn to a place the name is still
    // gliding to, it would jump there ahead of it)
    if let (Some((from, to)), true, true) = (line, kept.is_some(), lines) {
        hud_line(f, from, to, 1.0, color.a(0.55));
    }
    f.text_placed(name, 0, p, c - p, text, size, color);
}

/// Points of the hyperbola ab = p on the side of the signs (sa, sb): a = sa √|p| eᵘ,
/// b = sb √|p| e⁻ᵘ, for u across [−spread, spread]. For p = 0, the two half-axes.
fn hyperbola(p: f64, sa: f64, sb: f64, spread: f64, n: usize) -> Vec<Vec3> {
    let r = p.abs().sqrt();
    (0..n)
        .map(|i| {
            let u = -spread + 2.0 * spread * i as f64 / (n - 1).max(1) as f64;
            if r < 1e-9 {
                // (the level line through p = 0 is the two axes: half of each)
                let t = 12.0 * (u / spread);
                if t < 0.0 {
                    v3(DVec2::new(0.0, -t * sb))
                } else {
                    v3(DVec2::new(t * sa, 0.0))
                }
            } else {
                v3(DVec2::new(sa * r * u.exp(), sb * r * (-u).exp()))
            }
        })
        .collect()
}

impl EosToy {
    /// Everything drawn in the plane, and the names written beside it. Returns the boxes
    /// of the heading and of the caption, which are written over it.
    fn picture(&mut self, f: &mut Frame, lay: &Layout, covered: bool) -> [Option<(Vec2, Vec2)>; 2] {
        for part in PARTS {
            lines_for(f, part);
        }
        iterate_dots(f);
        moving_dots(f);
        let shown = self.glide.shown;
        let mix = shown.mix.clamp(0.0, 1.0);
        let eta = shown.eta();
        let scale = shown.scale();
        let wpp = {
            let (lo, hi) = f.cam.flat_bounds();
            ((hi.y - lo.y) / f.view_size().y.max(1.0)) as f64
        };

        // the loss's level lines, the one loss's fading as the other's comes
        for (loss, seen) in [(Loss::Two, 1.0 - mix), (Loss::Four, mix)] {
            for (k, level) in LEVELS.iter().enumerate() {
                let products = loss.level_products(*level);
                // (always four products, so that the batch keeps its number of points)
                for i in 0..4 {
                    let p = products.get(i).copied();
                    let color = pal::DIM.a(if p.is_some() { (0.34 - 0.03 * k as f32) * seen as f32 } else { 0.0 });
                    let p = p.unwrap_or(1.0);
                    let sign = if p < 0.0 { -1.0 } else { 1.0 };
                    for (sa, sb) in [(1.0, sign), (-1.0, -sign)] {
                        let pts = hyperbola(p, sa, sb, 3.2, 96);
                        lines_for(f, Part::Levels).polyline(pts, 1.0, color);
                    }
                }
            }
        }

        // the floor, coloured by its sharpness: ab = 1, and for the second loss ab = −1
        let flattest = 2.0 * scale;
        for (q, seen) in [(1.0, 1.0), (-1.0, mix)] {
            for side in [1.0, -1.0] {
                let batch = lines_for(f, Part::Floor);
                batch.brk();
                for i in 0..160 {
                    let u = -2.6 + 5.2 * i as f64 / 159.0;
                    let z = DVec2::new(side * u.exp(), side * q * (-u).exp());
                    let s = scale * (z.x * z.x + z.y * z.y);
                    batch.push(v3(z), 2.6, floor_color(s, flattest, seen as f32));
                }
                batch.brk();
            }
        }

        // the stretch flat enough for this step: along the floor out to where S = 2/η
        let reach = reach(scale, 2.0 / eta);
        // (on the floor a² − b² = 2 sinh 2u, for a = eᵘ)
        let u_end = reach.map_or(0.0, |d| 0.5 * (0.5 * d).asinh());
        let mut ends: Vec<(DVec2, DVec2)> = Vec::new();
        for (q, seen) in [(1.0, 1.0f32), (-1.0, mix as f32)] {
            for side in [1.0, -1.0] {
                let at = |u: f64| DVec2::new(side * u.exp(), side * q * (-u).exp());
                let color = LIMIT.a(if reach.is_some() { 0.2 * seen } else { 0.0 });
                let pts: Vec<Vec3> = (0..48).map(|i| v3(at(-u_end + 2.0 * u_end * i as f64 / 47.0))).collect();
                lines_for(f, Part::Stretch).polyline(pts, 10.0, color);
                // a tick across the floor at each end
                for u in [-u_end, u_end] {
                    let z = at(u);
                    let across = DVec2::new(z.y, z.x).normalize_or_zero() * (13.0 * wpp);
                    let tick = LIMIT.a(if reach.is_some() { 0.95 * seen } else { 0.0 });
                    lines_for(f, Part::Floor).segment(v3(z - across), v3(z + across), 2.6, tick);
                    if q > 0.0 && side > 0.0 {
                        ends.push((z, DVec2::new(z.y, z.x).normalize_or_zero()));
                    }
                }
            }
        }

        // gradient flow from the start, dashed
        // (dashes of a fixed share of the path each, so that they glide with it as the
        // start moves instead of being cut afresh)
        let flow = flow_path(self.loss, shown.start, 81);
        for dash in flow.chunks(4) {
            lines_for(f, Part::Flow).polyline(dash.iter().take(3).map(|p| v3(*p)), 1.6, SGD.a(0.6));
        }
        let landing_at = landing(self.loss, shown.start);

        // the last run, fading
        if let Some(Fading { run, film, seen, .. }) = &self.old {
            let n = (film.floor() as usize).min(run.steps());
            let mut pts: Vec<Vec3> = run.z[..=n].iter().map(|p| v3(*p).clamp(Vec3::splat(-60.0), Vec3::splat(60.0))).collect();
            pts.push(v3(along(&run.z, *film)).clamp(Vec3::splat(-60.0), Vec3::splat(60.0)));
            lines_for(f, Part::Old).polyline(pts, 1.8, SGD.a(0.9 * seen));
        }

        // gradient descent's path as far as the film has got, with a dot at each step
        let shows_run = !self.play_when_there;
        let run = &self.run;
        let film = self.film;
        let mut drawn_path: Vec<Vec3> = Vec::new();
        if shows_run {
            let n = self.step_now();
            let far = |p: Vec3| p.clamp(Vec3::splat(-60.0), Vec3::splat(60.0));
            drawn_path.extend(run.z[..=n].iter().map(|p| far(v3(*p))));
            lines_for(f, Part::Path).polyline(drawn_path.iter().copied(), 2.0, SGD);
            // the next step fades in as the film comes to it: a bounce lights up across
            // the valley, while the two dots glide along its walls (a point running
            // across and back would jerk to a stop at every turn)
            let next = (film - n as f64) as f32;
            if n < run.steps() && next > 0.0 {
                lines_for(f, Part::Path).segment(far(v3(run.z[n])), far(v3(run.z[n + 1])), 2.0, SGD.a(next));
                iterate_dots(f).push(far(v3(run.z[n + 1])), 4.5, SGD.a(0.75 * next));
            }
            // (a dot for each iterate, but not on top of one another where they crowd)
            let mut last: Option<Vec2> = None;
            for p in &drawn_path {
                let Some(at) = on_screen(f, *p) else { continue };
                if last.is_none_or(|l| l.distance(at) >= 4.0) {
                    iterate_dots(f).push(*p, 4.5, SGD.a(0.75));
                    last = Some(at);
                }
            }
            for d in self.drawn {
                moving_dots(f).push(v3(d).clamp(Vec3::splat(-60.0), Vec3::splat(60.0)), 11.0, SGD);
            }
        }
        // (the batches that may have had nothing in them, kept)
        for part in PARTS {
            keep_batch(lines_for(f, part));
        }
        if iterate_dots(f).pts.is_empty() {
            iterate_dots(f).push(Vec3::ZERO, 0.0, Color::CLEAR);
        }
        if moving_dots(f).pts.is_empty() {
            moving_dots(f).push(Vec3::ZERO, 0.0, Color::CLEAR);
        }

        // the heading, at the top of the picture, and the caption at its foot
        let heading_at = lay.picture.0 + Vec2::new(16.0, 12.0);
        let formula = if self.loss == Loss::Two { r"$L = \tfrac12(ab - 1)^2$" } else { r"$L = \tfrac14(1 - a^2b^2)^2$" };
        let heading = if reach.is_some() {
            format!(r"{formula},  step $\eta = {}$,  flat enough where $S < 2/\eta = {}$", short(eta, 3), short(2.0 / eta, 2))
        } else {
            format!(r"{formula},  step $\eta = {}$,  no minimum flat enough: the flattest has $S = {} > 2/\eta = {}$", short(eta, 3), short(flattest, 2), short(2.0 / eta, 2))
        };
        let status = self.status();
        let size = if lay.narrow { 13.0 } else { 15.0 };
        let wide = f.text_size(&heading, size).x.max(f.text_size(&status, 13.0).x);
        let room_across = lay.picture.1.x - lay.picture.0.x - 32.0;
        let (heading, size) = if wide > room_across { (format!(r"{formula},  $\eta = {}$", short(eta, 3)), 13.0) } else { (heading, size) };
        f.caption_px(heading_at, heading.clone(), size, pal::TEXT, Align2::LEFT_TOP);
        if !status.is_empty() {
            f.caption_px(heading_at + Vec2::new(0.0, size + 8.0), status.clone(), 13.0, pal::DIM, Align2::LEFT_TOP);
        }
        let wide = f.text_size(&heading, size).x.max(f.text_size(&status, 13.0).x);
        let heading_box = (heading_at - Vec2::splat(6.0), heading_at + Vec2::new(wide + 12.0, size + 32.0));
        let mut plates = [Some(heading_box), None];
        if !lay.narrow && f.view_size().y >= 300.0 {
            let parts = ["cyan: gradient descent", "dashed: gradient flow", "green: the floor, brighter where sharper", r"yellow: flat enough, $S < 2/\eta$"];
            let (left, bottom) = (lay.picture.0.x + 18.0, lay.picture.1.y - 16.0);
            let line = parts.join("      ");
            let w = f.text_size(&line, 12.0).x;
            let room = lay.picture.1.x - lay.picture.0.x - 36.0;
            if w <= room {
                f.caption_px(Vec2::new(left, bottom), line, 12.0, pal::DIM, Align2::LEFT_BOTTOM);
                plates[1] = Some((Vec2::new(left - 8.0, bottom - 20.0), Vec2::new(left + w + 8.0, bottom + 6.0)));
            } else {
                let (one, two) = (parts[..2].join("      "), parts[2..].join("      "));
                let w = f.text_size(&one, 12.0).x.max(f.text_size(&two, 12.0).x);
                if w <= room {
                    f.caption_px(Vec2::new(left, bottom - 19.0), one, 12.0, pal::DIM, Align2::LEFT_BOTTOM);
                    f.caption_px(Vec2::new(left, bottom), two, 12.0, pal::DIM, Align2::LEFT_BOTTOM);
                    plates[1] = Some((Vec2::new(left - 8.0, bottom - 39.0), Vec2::new(left + w + 8.0, bottom + 6.0)));
                }
            }
        }
        if covered {
            return plates;
        }

        // the names, keeping clear of what is drawn and of one another
        let mut room = Room::new(f);
        if let Some(b) = &lay.board {
            room.wall(b.min - Vec2::splat(4.0), b.max + Vec2::splat(4.0));
        }
        for plate in plates.iter().flatten() {
            room.wall(plate.0, plate.1);
        }
        // the strips where the axes' numbers go (as `Frame::axes2` places them)
        {
            let (lo, hi) = f.cam.flat_bounds();
            let wpp = (hi.y - lo.y) / f.view_size().y.max(1.0);
            let margin = 16.0 * wpp;
            let y = 0.0f32.max(lo.y + margin * 1.6).min(hi.y - margin * 0.4);
            let x = 0.0f32.max(lo.x + margin * 0.4).min(hi.x - margin * 2.4);
            if let (Some(along), Some(up)) = (on_screen(f, Vec3::new(0.0, y, 0.0)), on_screen(f, Vec3::new(x, 0.0, 0.0))) {
                room.wall(Vec2::new(lay.picture.0.x, along.y + 3.0), Vec2::new(lay.picture.1.x, along.y + 21.0));
                room.wall(Vec2::new(up.x + 3.0, lay.picture.0.y), Vec2::new(up.x + 40.0, lay.picture.1.y));
            }
        }
        // (the whole path, not only as far as the film has got: so that the names find
        // their places once, and keep them while the film draws it)
        {
            let every = (run.z.len() / 500).max(1);
            let pts: Vec<Vec3> = run.z.iter().step_by(every).map(|p| v3(*p).clamp(Vec3::splat(-60.0), Vec3::splat(60.0))).collect();
            for w in pts.windows(2) {
                room.line(f, w[0], w[1]);
            }
        }
        for w in flow.windows(2).step_by(4) {
            room.line(f, v3(w[0]), v3(w[1]));
        }
        for i in 0..40 {
            let at = |u: f64| v3(DVec2::new(u.exp(), (-u).exp()));
            let (u0, u1) = (-2.6 + 5.2 * i as f64 / 40.0, -2.6 + 5.2 * (i + 1) as f64 / 40.0);
            room.line(f, at(u0), at(u1));
        }
        let away = |d: DVec2| Vec2::new(d.x as f32, -(d.y as f32));
        // the names, the most needed first, so that it is the less needed that give way:
        // what the run came to (on the side of the floor towards the valley's middle),
        // the ends of the stretch (on the other side), the start, and where gradient
        // flow from it ends (where that is not the same place)
        let mut named: Vec<DVec2> = Vec::new();
        // (names only while nothing glides: a name's text and place change as the
        // setting glides, and it would jump about; they come back once it has arrived)
        let still = self.glide.arrived();
        let (hold, lines) = (self.hand_moving(), still && !self.hand_moving());
        {
            // (its place is found from the start of the film, and kept free until the
            // film gets there)
            let last = run.last();
            let text = match run.ending {
                Ending::Settled => Some(format!("ends: $S = {}$", num(run.under[run.steps()], 2))),
                Ending::Cycle(_) => Some("bounces for ever".to_string()),
                Ending::Restless => Some("never settles".to_string()),
                Ending::Stuck => Some("stuck".to_string()),
                Ending::FlewOff => None,
            };
            if let Some(text) = text {
                // (at the dot that ends there, which glides where a hand changes the run;
                // till the film gets there, the end itself)
                let at = if self.film_done() { self.drawn[run.steps() % 2] } else { last };
                let shown = still && shows_run && self.film_done();
                // (while it is not shown, the scene keeps its place; once it is, the frame)
                let mut place = if shown { f.places.chosen("end", 0).or(self.end_place) } else { self.end_place };
                leader_label_kept(f, &mut room, "end", v3(at), &text, 13.0, pal::TEXT, -away(DVec2::new(last.y, last.x)), lines, shown, hold, &mut place);
                self.end_place = place;
                named.push(last);
            }
        }
        if reach.is_some() {
            for (k, (z, out)) in ends.iter().enumerate() {
                let text = format!(r"$S = 2/\eta = {}$", short(2.0 / eta, 2));
                // (out across the floor, and leaning on out along its arm: not over the
                // stretch the name belongs to)
                let along = if k == 0 { DVec2::new(-z.x, z.y) } else { DVec2::new(z.x, -z.y) };
                let towards = away(*out) + 0.8 * away(along.normalize_or_zero());
                leader_label(f, &mut room, if k == 0 { "stretch 0" } else { "stretch 1" }, v3(*z), &text, 12.5, LIMIT, towards, lines, still, hold);
            }
        } else {
            // (the heading says the rest)
            let text = format!(r"flattest: $S = {}$", short(flattest, 2));
            leader_label(f, &mut room, "flattest", v3(DVec2::ONE), &text, 12.5, LIMIT, Vec2::new(-1.0, -1.0), lines, still, hold);
        }
        leader_label(f, &mut room, "start", v3(shown.start), "start", 13.0, SGD, Vec2::new(1.0, -1.0), lines, still, hold);
        named.push(shown.start);
        if let Some(w) = landing_at {
            // (not beside the start or the end, where it would only crowd them)
            let apart = |z: &DVec2| on_screen(f, v3(*z)).zip(on_screen(f, v3(w))).is_none_or(|(p, q)| p.distance(q) > 40.0);
            if still && named.iter().all(apart) {
                let text = format!("gradient flow ends: $S = {}$", num(self.loss.floor_sharpness(w.x * w.x - w.y * w.y), 2));
                room.label(f, "landing", v3(w), &text, 12.0, SGD.a(0.8), -away(DVec2::new(w.y, w.x)), 10.0, true);
            }
        }
        // the axes' names, near the picture's edge
        let world = |px: Vec2| {
            let ((lo, hi), view) = (f.cam.flat_bounds(), f.view_size().max(Vec2::ONE));
            Vec2::new(lo.x + (hi.x - lo.x) * px.x / view.x, hi.y - (hi.y - lo.y) * px.y / view.y)
        };
        let (right, top) = (world(lay.picture.1 - Vec2::new(28.0, 0.0)).x, world(lay.picture.0 + Vec2::new(0.0, 70.0)).y);
        room.label(f, "a axis", Vec3::new(right, 0.0, 0.0), "$a$", 15.0, pal::DIM, Vec2::new(0.0, -1.0), 8.0, true);
        room.label(f, "b axis", Vec3::new(0.0, top, 0.0), "$b$", 15.0, pal::DIM, Vec2::new(1.0, 0.0), 8.0, true);
        plates
    }

    /// The panel on the picture: the buttons, and the three charts.
    fn board(&mut self, f: &mut Frame, b: &Board) {
        for kind in CHART_KINDS {
            plot::apart(f, kind, false);
        }
        plot::apart(f, plot::CURVES, true);
        hud_panel(f, b.min, b.max);
        let busy = self.busy();
        plot::button(f, b.buttons[0], r"$\tfrac12(ab - 1)^2$", 12.5, self.loss == Loss::Two);
        plot::button(f, b.buttons[1], r"$\tfrac14(1 - a^2b^2)^2$", 12.5, self.loss == Loss::Four);
        let play = if busy { "❚❚ pause" } else if self.film_done() { "▶ again" } else { "▶ play" };
        plot::button(f, b.buttons[2], play, 12.5, false);
        if let Some((title, rect)) = b.charts[0] {
            self.loss_chart(f, title, rect);
        }
        if let Some((title, rect)) = b.charts[1] {
            self.sharpness_chart(f, title, rect);
        }
        if let Some((title, rect)) = b.charts[2] {
            self.sweep_chart(f, title, rect);
        }
        if let Some(at) = b.key {
            let line = r"cyan: descent   dashed: flow   green: floor   yellow: $S < 2/\eta$";
            f.text_px(at, line, 11.0, pal::DIM, Align2::LEFT_TOP);
        }
        for kind in CHART_KINDS {
            keep_batch(plot::apart(f, kind, false));
        }
        keep_batch(plot::apart(f, plot::CURVES, true));
    }

    /// The step numbers along a step chart's foot: 0, 1, 10, 100, ...
    fn step_axis(&self, f: &mut Frame, chart: &Plot) {
        let (min, max) = chart.rect;
        plot::line_of(f, plot::AXES, Vec2::new(min.x, max.y), max, 1.0, pal::DIM.a(0.8));
        let mut last_x = f32::MIN;
        for n in [0.0, 1.0, 3.0, 10.0, 30.0, 100.0, 300.0, 1000.0, 3000.0, 10000.0] {
            let x = chart.px_x(step_x(n));
            if x > max.x + 0.5 || x - last_x < 22.0 {
                continue;
            }
            last_x = x;
            plot::line_of(f, plot::AXES, Vec2::new(x, max.y), Vec2::new(x, max.y + 4.0), 1.0, pal::DIM.a(0.8));
            f.text_px(Vec2::new(x, max.y + 5.0), num(n, 0), 10.5, pal::DIM, Align2::CENTER_TOP);
        }
    }

    /// The loss at each step, on a logarithmic scale.
    fn loss_chart(&self, f: &mut Frame, title: Vec2, rect: plot::Rect) {
        let [x_hi, lo, hi, _] = [0, 1, 2, 3].map(|i| self.ranges.get(i) as f64);
        let chart = Plot::new(rect, [0.0, x_hi], [lo, hi]);
        f.text_px(title, "loss $L$ at each step (logarithmic scales)", 11.5, pal::TEXT, Align2::LEFT_TOP);
        if chart.tiny() {
            return;
        }
        // powers of ten up the side, as many as there is room for
        let (min, max) = chart.rect;
        plot::line_of(f, plot::AXES, min, Vec2::new(min.x, max.y), 1.0, pal::DIM.a(0.8));
        let every = ((hi - lo) / (chart.height() as f64 / 22.0).max(1.0)).ceil().max(1.0) as i64;
        let mut k = lo.ceil() as i64;
        while k as f64 <= hi + 1e-9 {
            if k.rem_euclid(every) == 0 {
                let y = chart.px_y(k as f64);
                plot::line_of(f, plot::AXES, Vec2::new(min.x, y), Vec2::new(max.x, y), 1.0, pal::DIM.a(0.12));
                f.text_px(Vec2::new(min.x - 5.0, y), format!("$10^{{{k}}}$"), 10.5, pal::DIM, Align2::RIGHT_CENTER);
            }
            k += 1;
        }
        self.step_axis(f, &chart);
        let log_loss: Vec<f64> = self.run.value.iter().map(|v| v.max(1e-300).log10()).collect();
        if let Some(Fading { run, film, seen, ranges }) = &self.old {
            // (on its own chart's scales, fading where it was)
            let then = Plot::new(rect, [0.0, ranges[0] as f64], [ranges[1] as f64, ranges[2] as f64]);
            let old: Vec<f64> = run.value.iter().map(|v| v.max(1e-300).log10()).collect();
            film_curve(f, &then, FADING, &old, *film, 1.4, SGD.a(0.8 * seen));
        }
        if !self.play_when_there {
            film_curve(f, &chart, LOSS_CURVE, &log_loss, self.film, 1.6, SGD);
            if !self.film_done() {
                let x = chart.px_x(step_x(self.film));
                plot::line_of(f, plot::MARKS, Vec2::new(x, min.y), Vec2::new(x, max.y), 1.0, pal::WHITE.a(0.25));
            }
        }
    }

    /// The sharpness at each step: at the point, and of the floor beneath it; and 2/η.
    fn sharpness_chart(&self, f: &mut Frame, title: Vec2, rect: plot::Rect) {
        let [x_hi, _, _, top] = [0, 1, 2, 3].map(|i| self.ranges.get(i) as f64);
        let chart = Plot::new(rect, [0.0, x_hi], [0.0, top]);
        f.text_px(title, r"sharpness $S$: of the floor beneath (bold), at the point (faint)", 11.5, pal::TEXT, Align2::LEFT_TOP);
        if chart.tiny() {
            return;
        }
        chart.y_grid(f);
        chart.y_axis(f, "", 10.5);
        self.step_axis(f, &chart);
        let (min, max) = chart.rect;
        let limit = 2.0 / self.glide.shown.eta();
        chart.hline(f, plot::CURVES, limit, 1.6, LIMIT, false);
        if limit <= top {
            f.text_px(Vec2::new(max.x - 2.0, chart.px_y(limit) - 2.0), r"$2/\eta$", 11.0, LIMIT, Align2::RIGHT_BOTTOM);
        } else {
            f.text_px(Vec2::new(max.x - 2.0, min.y + 2.0), format!(r"$2/\eta = {}$, above", num(limit, 1)), 11.0, LIMIT, Align2::RIGHT_TOP);
        }
        if let Some(Fading { run, film, seen, ranges }) = &self.old {
            let then = Plot::new(rect, [0.0, ranges[0] as f64], [0.0, ranges[3] as f64]);
            film_curve(f, &then, FADING, &run.under, *film, 1.4, CURVATURE.a(0.8 * seen));
        }
        if !self.play_when_there {
            let run = &self.run;
            film_curve(f, &chart, AT_POINT, &run.sharp, self.film, 1.0, CURVATURE.a(0.45));
            film_curve(f, &chart, UNDER, &run.under, self.film, 2.2, CURVATURE);
            if !self.film_done() {
                let x = chart.px_x(step_x(self.film));
                plot::line_of(f, plot::MARKS, Vec2::new(x, min.y), Vec2::new(x, max.y), 1.0, pal::WHITE.a(0.25));
            }
        }
    }

    /// Where a run from this start ends, for each step of the sweep; 2/η; gradient
    /// flow's end; and the step chosen.
    fn sweep_chart(&self, f: &mut Frame, title: Vec2, rect: plot::Rect) {
        let chart = Plot::new(rect, [ETA_LO.log10(), ETA_HI.log10()], [S_LO.log10(), S_HI.log10()]);
        f.text_px(title, r"where a run from this start ends, for each step $\eta$", 11.5, pal::TEXT, Align2::LEFT_TOP);
        if chart.tiny() {
            return;
        }
        let (min, max) = chart.rect;
        plot::line_of(f, plot::AXES, min, Vec2::new(min.x, max.y), 1.0, pal::DIM.a(0.8));
        plot::line_of(f, plot::AXES, Vec2::new(min.x, max.y), max, 1.0, pal::DIM.a(0.8));
        for v in log_ticks(S_LO, S_HI, chart.height() * 1.3) {
            let y = chart.px_y(v.log10());
            plot::line_of(f, plot::AXES, Vec2::new(min.x, y), Vec2::new(max.x, y), 1.0, pal::DIM.a(0.12));
            f.text_px(Vec2::new(min.x - 5.0, y), log_tick_text(v), 10.5, pal::DIM, Align2::RIGHT_CENTER);
        }
        for v in log_ticks(ETA_LO, ETA_HI, chart.width()) {
            let x = chart.px_x(v.log10());
            plot::line_of(f, plot::AXES, Vec2::new(x, max.y), Vec2::new(x, max.y + 4.0), 1.0, pal::DIM.a(0.8));
            f.text_px(Vec2::new(x, max.y + 5.0), log_tick_text(v), 10.5, pal::DIM, Align2::CENTER_TOP);
        }
        // 2/η, and where gradient flow ends (whatever the step)
        chart.curve(f, plot::CURVES, 48, 1.6, LIMIT, |x| (2.0 / 10f64.powf(x)).log10());
        // (as drawn: following the start and the loss while they glide)
        let shown = self.glide.shown;
        let flow_end = landing(self.loss, shown.start).map_or(f64::NAN, |_| shown.scale() * (shown.start.x.powi(2) - shown.start.y.powi(2)).hypot(2.0));
        if flow_end.is_finite() {
            let y = chart.px_y(flow_end.log10());
            if y >= min.y && y <= max.y {
                plot::dashed_of(f, plot::CURVES, Vec2::new(min.x, y), Vec2::new(max.x, y), 1.2, SGD.a(0.7));
                f.text_px(Vec2::new(min.x + 4.0, y - 2.0), "gradient flow", 10.5, SGD.a(0.85), Align2::LEFT_BOTTOM);
            }
        }
        let label_at = chart.at(1.0f64.log10() - 0.55, (2.0 / 10f64.powf(-0.55)).log10());
        if inside_rect(label_at, chart.rect) {
            f.text_px(label_at + Vec2::new(6.0, -2.0), r"$2/\eta$", 11.0, LIMIT, Align2::LEFT_BOTTOM);
        }
        // each run's end: a dot where it settled, a mark along the top where it was
        // thrown off and along the foot where it never settled
        let mut thrown: Option<f32> = None;
        // (faint while it is for a start the hand has moved from, till it is done again)
        let seen = if self.sweep_stale { 0.3 } else { 1.0 };
        for (i, o) in self.sweep.ends.iter().enumerate() {
            let x = chart.px_x(sweep_eta(i).log10());
            match o {
                Outcome::Ends(s) => chart.dot(f, sweep_eta(i).log10(), s.log10().max(S_LO.log10()).min(S_HI.log10()), 5.0, CURVATURE.a(seen)),
                Outcome::Not(Ending::FlewOff) => {
                    plot::line_of(f, SWEEP_MARKS, Vec2::new(x, min.y), Vec2::new(x, min.y + 6.0), 2.0, pal::DIM.a(seen));
                    thrown.get_or_insert(x);
                }
                Outcome::Not(_) => plot::line_of(f, SWEEP_MARKS, Vec2::new(x, max.y - 6.0), Vec2::new(x, max.y), 2.0, pal::DIM.a(seen)),
            }
        }
        if let Some(x) = thrown {
            f.text_px(Vec2::new(x.min(max.x - 52.0), min.y + 8.0), "thrown off", 10.5, pal::DIM, Align2::LEFT_TOP);
        }
        // the step chosen, and its run's end
        let eta = self.glide.shown.eta();
        chart.vline(f, plot::MARKS, eta.log10(), 1.4, SGD.a(0.8), false);
        if self.glide.arrived() {
            if let Some(s) = self.run.end_sharpness() {
                chart.dot(f, eta.log10(), s.log10().max(S_LO.log10()).min(S_HI.log10()), 9.0, pal::WHITE);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Points all over the plane of the two weights, and the awkward ones: on the
    /// floor, on the axes, at the origin, on the diagonals, far out.
    fn points() -> Vec<DVec2> {
        let mut rng = Rng::new(11);
        let mut out: Vec<DVec2> = (0..300).map(|k| DVec2::new(rng.normal() as f64, rng.normal() as f64) * [0.3, 1.0, 2.0][k % 3]).collect();
        out.extend([
            DVec2::new(2.0, 0.5),
            DVec2::new(1.0, 1.0),
            DVec2::new(-1.0, -1.0),
            DVec2::new(1.0, -1.0),
            DVec2::new(0.0, 0.0),
            DVec2::new(0.0, 1.5),
            DVec2::new(-2.0, 0.0),
            DVec2::new(5.0, 0.2),
            DVec2::new(0.7, -0.7),
            DVec2::from(SHARP),
            DVec2::from(MIDDLE),
        ]);
        out
    }

    const LOSSES: [Loss; 2] = [Loss::Two, Loss::Four];

    /// The gradient is the loss's slope and the Hessian the gradient's, by central
    /// differences.
    #[test]
    fn the_hessian_in_closed_form_is_the_second_derivative() {
        for loss in LOSSES {
            for z in points() {
                let h = 1e-5;
                let (ea, eb) = (DVec2::new(h, 0.0), DVec2::new(0.0, h));
                let slope = DVec2::new(loss.value(z + ea) - loss.value(z - ea), loss.value(z + eb) - loss.value(z - eb)) / (2.0 * h);
                let g = loss.gradient(z);
                let scale = 1.0 + z.length().powi(4);
                assert!((slope - g).abs().max_element() < 1e-6 * scale, "{loss:?} at {z:?}: {slope:?} against {g:?}");
                let da = (loss.gradient(z + ea) - loss.gradient(z - ea)) / (2.0 * h);
                let db = (loss.gradient(z + eb) - loss.gradient(z - eb)) / (2.0 * h);
                let [haa, hab, hbb] = loss.hessian(z);
                let tol = 1e-6 * (1.0 + z.length().powi(6));
                assert!((da.x - haa).abs() < tol && (da.y - hab).abs() < tol && (db.x - hab).abs() < tol && (db.y - hbb).abs() < tol, "{loss:?} at {z:?}");
            }
        }
    }

    /// The sharpness is the larger eigenvalue, as the library's symmetric
    /// eigensolver finds it.
    #[test]
    fn the_sharpness_is_the_larger_eigenvalue() {
        for loss in LOSSES {
            for z in points() {
                let [x, y, w] = loss.hessian(z);
                let (values, _) = crate::math::linalg::eig_sym(&[x, y, y, w], 2);
                let most = values.iter().copied().fold(f64::MIN, f64::max);
                assert!((loss.sharpness(z) - most).abs() < 1e-9 * (1.0 + most.abs()), "{loss:?} at {z:?}");
            }
        }
    }

    /// On the floor the Hessian's eigenvalues are 0 (along the floor) and the
    /// sharpness, c (a² + b²) = c √((a² − b²)² + 4): least, 2c, at a = b, and growing
    /// without bound along each arm. `reach` finds where it is a given value.
    #[test]
    fn on_the_floor_the_sharpness_grows_out_along_each_arm() {
        for loss in LOSSES {
            let c = loss.scale();
            let mut last = 0.0;
            for i in 0..=60 {
                let u = i as f64 * 0.05;
                // (the floor: ab = 1, and for the second loss ab = −1 too)
                for q in if loss == Loss::Two { &[1.0][..] } else { &[1.0, -1.0][..] } {
                    for side in [1.0, -1.0] {
                        let z = DVec2::new(side * u.exp(), side * q * (-u).exp());
                        assert!(loss.value(z) < 1e-28 && loss.gradient(z).length() < 1e-12);
                        let [x, y, w] = loss.hessian(z);
                        let (det, tr) = (x * w - y * y, x + w);
                        assert!(det.abs() < 1e-9 * (1.0 + tr * tr), "{loss:?} at {z:?}: the other eigenvalue is not 0");
                        let d = z.x * z.x - z.y * z.y;
                        assert!((loss.sharpness(z) - c * (z.x * z.x + z.y * z.y)).abs() < 1e-9 * tr && (loss.floor_sharpness(d) - loss.sharpness(z)).abs() < 1e-9 * tr);
                        assert!((beneath(loss, z) - loss.sharpness(z)).abs() < 1e-9 * tr);
                    }
                }
                let s = loss.floor_sharpness(2.0 * (2.0 * u).sinh());
                assert!(s >= last, "{loss:?}: not growing along the arm");
                last = s;
                if let Some(d) = reach(loss.scale(), s) {
                    assert!((loss.floor_sharpness(d) - s).abs() < 1e-9 * s);
                }
            }
            assert_eq!(loss.flattest(), loss.sharpness(DVec2::ONE));
            assert!(reach(loss.scale(), loss.flattest() * 0.999).is_none() && reach(loss.scale(), loss.flattest()) == Some(0.0));
            assert!(last > 100.0 * loss.flattest());
        }
        assert_eq!([Loss::Two.flattest(), Loss::Four.flattest()], [2.0, 4.0]);
    }

    /// The level lines are where the loss has their level.
    #[test]
    fn the_level_lines_are_at_their_levels() {
        for loss in LOSSES {
            for level in LEVELS {
                for p in loss.level_products(level) {
                    assert!((loss.ell(p) - level).abs() < 1e-12, "{loss:?} {level} {p}");
                    for z in hyperbola(p, 1.0, if p < 0.0 { -1.0 } else { 1.0 }, 3.2, 20) {
                        let z = DVec2::new(z.x as f64, z.y as f64);
                        assert!((loss.value(z) - level).abs() < 1e-5 * (1.0 + level), "{loss:?} {level} at {z:?}");
                    }
                }
            }
        }
    }

    /// One step of gradient descent multiplies a² − b² by exactly 1 − η²g², where
    /// g = ℓ′(ab); and a² + b² becomes (a² + b²)(1 + η²g²) − 4ηg ab.
    #[test]
    fn a_step_multiplies_a2_minus_b2_by_one_minus_eta2_g2() {
        for loss in LOSSES {
            for z in points() {
                for eta in [0.01, 0.25, 0.7, 1.9] {
                    let w = step(loss, eta, z);
                    let g = loss.slope(z.x * z.y);
                    let (d, d2) = (z.x * z.x - z.y * z.y, w.x * w.x - w.y * w.y);
                    let want = (1.0 - eta * eta * g * g) * d;
                    assert!((d2 - want).abs() <= 1e-9 * (1.0 + d.abs() * (1.0 + eta * eta * g * g)), "{loss:?} {eta} at {z:?}");
                    let (l, l2) = (z.length_squared(), w.length_squared());
                    let want = l * (1.0 + eta * eta * g * g) - 4.0 * eta * g * z.x * z.y;
                    assert!((l2 - want).abs() <= 1e-9 * (1.0 + l2.abs() + want.abs()));
                }
            }
        }
    }

    /// Gradient flow, followed with tiny steps, keeps a² − b² as it was, comes to rest
    /// where `landing` says, and runs along the path `flow_path` draws.
    #[test]
    fn gradient_flow_keeps_a2_minus_b2_and_lands_where_it_is_said_to() {
        for loss in LOSSES {
            for z0 in [DVec2::from(SHARP), DVec2::from(MIDDLE), DVec2::new(2.0, 1.6), DVec2::new(0.4, -0.3), DVec2::new(-1.2, 0.3), DVec2::new(0.3, 2.5)] {
                // Runge and Kutta's fourth-order steps of dz/dt = −∇L(z)
                let field = |z: DVec2| -loss.gradient(z);
                let mut z = z0;
                for _ in 0..400_000 {
                    // (steps short against the curvature where it is: the flow is stiff)
                    let h = 0.01 / (1.0 + loss.sharpness(z).abs());
                    let k1 = field(z);
                    let k2 = field(z + k1 * (h / 2.0));
                    let k3 = field(z + k2 * (h / 2.0));
                    let k4 = field(z + k3 * h);
                    z += (k1 + k2 * 2.0 + k3 * 2.0 + k4) * (h / 6.0);
                    if loss.gradient(z).length() < 1e-13 {
                        break;
                    }
                }
                let d0 = z0.x * z0.x - z0.y * z0.y;
                assert!((z.x * z.x - z.y * z.y - d0).abs() < 1e-8 * (1.0 + d0.abs()), "{loss:?} from {z0:?}");
                let w = landing(loss, z0).expect("a minimum");
                assert!((w - z).length() < 1e-6, "{loss:?} from {z0:?}: {z:?} against {w:?}");
                let path = flow_path(loss, z0, 80);
                assert!(path.len() == 80 && (path[0] - z0).length() < 1e-9 && (path[79] - w).length() < 1e-9);
                for p in &path {
                    assert!((p.x * p.x - p.y * p.y - d0).abs() < 1e-9 * (1.0 + d0.abs()));
                }
            }
        }
        // from the line a = −b the first loss's flow runs into the saddle at the origin;
        // on an axis the second loss's does not move
        assert!(landing(Loss::Two, DVec2::new(0.7, -0.7)).is_none() && (flow_path(Loss::Two, DVec2::new(0.7, -0.7), 20)[19]).length() < 1e-12);
        assert!(landing(Loss::Four, DVec2::new(0.0, 1.5)).is_none() && flow_path(Loss::Four, DVec2::new(0.0, 1.5), 20).iter().all(|p| *p == DVec2::new(0.0, 1.5)));
    }

    /// Gradient descent on the network of four weights in a row, ½(1 − xyzw)², from
    /// z = x and w = y, keeps them equal, and is gradient descent on the second loss.
    #[test]
    fn the_second_loss_is_four_weights_tied_in_pairs() {
        // (runs that settle: in one thrown off or never settling, the two ways of
        // rounding part at once, as any two nearby runs there do)
        for (x0, y0, eta) in [(3.0, 0.35, 0.05), (3.0, 0.35, 0.125), (1.3, 0.9, 0.3), (0.5, -1.7, 0.05), (0.5, -1.7, 0.2)] {
            assert_eq!(Run::new(Loss::Four, eta, DVec2::new(x0, y0)).ending, Ending::Settled);
            let mut w: [f64; 4] = [x0, y0, x0, y0];
            let mut z = DVec2::new(x0, y0);
            for _ in 0..300 {
                // ∂/∂wᵢ of ½(1 − xyzw)²: −(1 − xyzw) times the other three (the pairs
                // multiplied in the same order, so that equal weights stay equal)
                let r = 1.0 - w[0] * w[1] * w[2] * w[3];
                let g = [w[1] * w[3] * w[2], w[0] * w[2] * w[3], w[1] * w[3] * w[0], w[0] * w[2] * w[1]].map(|v| -r * v);
                for i in 0..4 {
                    w[i] -= eta * g[i];
                }
                z = step(Loss::Four, eta, z);
                assert!(w[0] == w[2] && w[1] == w[3]);
                assert!((DVec2::new(w[0], w[1]) - z).length() < 1e-9 * (1.0 + z.length()), "{x0} {y0} {eta}");
            }
        }
    }

    /// The film's two dots move without jumps: each from one iterate of its kind to the
    /// next, the path's end meets one of them at every whole step, and at the end they
    /// are the last two iterates.
    #[test]
    fn the_film_moves_without_jumps() {
        let run = Run::new(Loss::Two, 0.25, DVec2::from(SHARP));
        let n = run.steps();
        let longest = run.z.windows(2).map(|w| (w[1] - w[0]).length()).fold(0.0, f64::max);
        let mut before = dots(&run.z, 0.0);
        assert!(before.iter().all(|d| *d == run.z[0]));
        let ds = 0.01;
        let mut s = 0.0;
        while s < n as f64 {
            s += ds;
            let now = dots(&run.z, s);
            for k in 0..2 {
                assert!((now[k] - before[k]).length() <= longest * ds + 1e-12, "a dot jumps at {s}");
            }
            before = now;
        }
        for k in 0..=n {
            let d = dots(&run.z, k as f64);
            assert!((d[k % 2] - run.z[k]).length() < 1e-12 && (along(&run.z, k as f64) - run.z[k]).length() < 1e-12);
        }
        let end = dots(&run.z, n as f64);
        assert!(end.contains(&run.z[n]) && end.contains(&run.z[n - 1]));
        // and the film's pace: R steps a second at first, faster and faster
        assert!((film_on(0.0, 1e-4) - FILM_RATE * 1e-4).abs() < 1e-6 && (film_on(film_on(0.0, 1.0), 1.0) - film_on(0.0, 2.0)).abs() < 1e-9);
        assert!(film_on(0.0, 12.0) > 1500.0 && film_on(0.0, 3.0) < 40.0);
    }

    /// The scene as an experiment's button leaves it, run at 60 frames a second until
    /// it has come to rest.
    fn run(k: usize) -> EosToy {
        let mut scene = EosToy::new();
        for _ in 0..200 {
            scene.advance(1.0 / 60.0, 1.0 / 60.0);
        }
        scene.set_up(k);
        for _ in 0..60 * 40 {
            scene.advance(1.0 / 60.0, 1.0 / 60.0);
            if scene.at_rest() {
                break;
            }
        }
        assert!(scene.at_rest(), "experiment {k} has not come to rest");
        let t = &TRIALS[k];
        assert!(scene.loss == t.loss && scene.eta == t.eta && scene.start == DVec2::from(t.start) && scene.film == scene.run.steps() as f64);
        scene
    }

    /// The sweep's outcome at the grid step nearest `eta`.
    fn swept(scene: &EosToy, eta: f64) -> (f64, Outcome) {
        let i = ((eta / ETA_LO).ln() / (ETA_HI / ETA_LO).ln() * (SWEEP - 1) as f64).round() as usize;
        (sweep_eta(i), scene.sweep.ends[i])
    }

    #[test]
    fn a_short_step_does_what_its_caption_says() {
        let scene = run(0);
        let r = &scene.run;
        assert_eq!(num(r.under[0], 2), "9.10");
        assert!(2.0 / scene.eta > r.under[0] && num(2.0 / scene.eta, 0) == "10");
        assert!(r.ending == Ending::Settled && r.steps() == 65 && num(r.end_sharpness().unwrap_or(0.0), 2) == "9.10");
        // it ends nearly beneath the start, where gradient flow ends
        let w = landing(Loss::Two, DVec2::from(SHARP)).unwrap_or(DVec2::ZERO);
        assert!((r.last() - w).length() < 0.005, "{:?} {w:?}", r.last());
        assert_eq!(r.rises(), (0, 0));
        // swinging from side to side, each swing about −0.82 times the last
        assert_eq!(num(1.0 - scene.eta * r.under[0], 2), "−0.82");
        let off = |i: usize| r.z[i].x * r.z[i].y - 1.0;
        for i in 20..40 {
            assert!((off(i + 1) / off(i) + 0.82).abs() < 0.01, "{}", off(i + 1) / off(i));
        }
    }

    #[test]
    fn a_longer_step_does_what_its_caption_says() {
        let scene = run(1);
        let r = &scene.run;
        assert!(r.under[0] > 2.0 / scene.eta && num(2.0 / scene.eta, 0) == "8");
        // the loss rises 15 steps in a row, from the start, to its highest
        assert_eq!(r.rises().0, 15);
        assert!((1..=15).all(|i| r.value[i] > r.value[i - 1]) && r.value[16] < r.value[15]);
        let most = r.value.iter().copied().fold(0.0, f64::max);
        assert!(most == r.value[15] && num(r.value[0], 5) == "0.00125" && num(most, 2) == "0.37");
        // while the floor beneath falls; it settles after 57 steps at S = 6.71
        assert!(r.under.windows(2).all(|w| w[1] <= w[0] + 1e-12));
        assert!(r.ending == Ending::Settled && r.steps() == 57 && num(r.end_sharpness().unwrap_or(0.0), 2) == "6.71");
        // the swings die away once the floor beneath is under 8: the loss falls from
        // then on
        let under_at = r.under.iter().position(|s| *s < 8.0).unwrap_or(0);
        assert!(under_at > 10 && r.value[under_at + 2..].windows(2).all(|w| w[1] < w[0]), "{under_at}");
        assert_eq!(num(landing(Loss::Two, DVec2::from(SHARP)).map_or(0.0, |w| beneath(Loss::Two, w)), 2), "9.10");
    }

    #[test]
    fn longer_still_does_what_its_caption_says() {
        let scene = run(2);
        let r = &scene.run;
        assert_eq!(r.rises().0, 8);
        assert!((1..=8).all(|i| r.value[i] > r.value[i - 1]));
        assert_eq!(num(r.value.iter().copied().fold(0.0, f64::max), 2), "1.95");
        let end = r.end_sharpness().unwrap_or(0.0);
        assert!(r.steps() == 15 && num(end, 2) == "3.17" && end < 0.5 * 2.0 / scene.eta && num(2.0 / scene.eta, 2) == "6.67");
        // the lowest chart: up to 0.2 where gradient flow ends; then the longer the
        // step the flatter the end, down to S = 2.00 at 0.36; beyond, scattered, some
        // across the middle on the other arm, some on the floor's other half; and from
        // 0.44 on thrown off
        let gf = r.under[0];
        let ends: Vec<(f64, Outcome)> = (0..SWEEP).map(|i| (sweep_eta(i), scene.sweep.ends[i])).collect();
        for (eta, o) in &ends {
            match o {
                Outcome::Ends(s) if *eta <= 0.2 => assert!((s - gf).abs() < 0.005, "{eta}: {s}"),
                Outcome::Ends(_) => {}
                _ => assert!(*eta > 0.44, "{eta}: {o:?}"),
            }
            if *eta > 0.44 {
                assert_eq!(*o, Outcome::Not(Ending::FlewOff), "{eta}");
            }
        }
        let falling: Vec<f64> = ends.iter().filter(|(e, _)| *e > 0.2 && *e < 0.36).map(|(_, o)| if let Outcome::Ends(s) = o { *s } else { f64::NAN }).collect();
        assert!(falling.len() >= 8 && falling.windows(2).all(|w| w[1] < w[0]), "{falling:?}");
        let (eta, o) = swept(&scene, 0.36);
        assert!(num(eta, 2) == "0.36" && matches!(o, Outcome::Ends(s) if num(s, 2) == "2.00"), "{eta} {o:?}");
        let beyond: Vec<DVec2> = ends.iter().filter(|(e, _)| *e > 0.36 && *e < 0.44).map(|(e, _)| Run::new(Loss::Two, *e, DVec2::from(SHARP)).last()).collect();
        assert!(beyond.iter().any(|z| z.x > 0.0 && z.y > z.x + 0.1) && beyond.iter().any(|z| z.x < 0.0 && z.y < 0.0), "{beyond:?}");
    }

    #[test]
    fn the_other_loss_does_what_its_caption_says() {
        let scene = run(3);
        let r = &scene.run;
        assert!(scene.loss == Loss::Four && num(r.under[0], 2) == "18.20" && num(2.0 / scene.eta, 0) == "16");
        // the stretch is the second experiment's
        let (one, two) = (reach(Loss::Two.scale(), 2.0 / TRIALS[1].eta).unwrap_or(0.0), reach(Loss::Four.scale(), 2.0 / scene.eta).unwrap_or(1.0));
        assert!((one - two).abs() < 1e-12 && r.under[0] / (2.0 / scene.eta) == Run::new(Loss::Two, TRIALS[1].eta, DVec2::from(SHARP)).under[0] / 8.0);
        assert_eq!(r.rises().0, 3);
        // the dots creep along the valley for hundreds of steps before the floor
        // beneath them is flat enough
        let flat_at = r.under.iter().position(|s| *s < 16.0).unwrap_or(0);
        assert!(flat_at > 200 && r.under[..flat_at].windows(2).all(|w| w[1] <= w[0] + 1e-9), "{flat_at}");
        let end = r.end_sharpness().unwrap_or(0.0);
        assert!(r.steps() == 2105 && num(end, 2) == "15.94" && end < 16.0);
        // in the lowest chart, every step too long for where gradient flow ends that
        // still settles ends just under 2/η (within 1 %)
        let mut seen = 0;
        for (i, o) in scene.sweep.ends.iter().enumerate() {
            let limit = 2.0 / sweep_eta(i);
            if let (Outcome::Ends(s), true) = (o, r.under[0] > limit) {
                assert!(*s < limit && *s > 0.99 * limit, "{}: {s} against {limit}", sweep_eta(i));
                seen += 1;
            }
        }
        assert!(seen >= 10, "{seen}");
    }

    #[test]
    fn too_long_for_any_minimum_does_what_its_caption_says() {
        let scene = run(4);
        let r = &scene.run;
        assert!(num(2.0 / scene.eta, 2) == "1.82" && reach(Loss::Two.scale(), 2.0 / scene.eta).is_none());
        assert_eq!(num(Run::new(Loss::Two, 0.25, DVec2::from(MIDDLE)).end_sharpness().unwrap_or(0.0), 2), "2.18");
        assert_eq!(r.ending, Ending::Cycle(2));
        let n = r.steps();
        let (p, q) = (r.z[n], r.z[n - 1]);
        let (lo, hi) = if p.x < q.x { (p, q) } else { (q, p) };
        assert!((lo.x - lo.y).abs() < 1e-9 && (hi.x - hi.y).abs() < 1e-9);
        assert_eq!([num(lo.x, 2), num(hi.x, 2)], ["0.81", "1.12"]);
        assert!(lo.x * lo.y < 1.0 && hi.x * hi.y > 1.0);
        let (l1, l2) = (Loss::Two.value(lo).min(Loss::Two.value(hi)), Loss::Two.value(lo).max(Loss::Two.value(hi)));
        assert_eq!([num(l1, 3), num(l2, 3)], ["0.030", "0.057"]);
        let thrown = Run::new(Loss::Two, scene.eta, DVec2::from(SHARP));
        assert!(thrown.ending == Ending::FlewOff && thrown.steps() == 4, "{:?} {}", thrown.ending, thrown.steps());
    }

    /// What the panel says was measured: from the sharp start, the first loss settles
    /// for every step up to 0.4395 and is thrown off for every longer one, and 4 over
    /// the sharpness of the floor beneath the start is 0.4396.
    #[test]
    fn the_longest_step_that_settles_is_four_over_the_floor_beneath() {
        let start = DVec2::from(SHARP);
        let s0 = beneath(Loss::Two, start);
        assert_eq!(num(4.0 / s0, 4), "0.4396");
        for i in 0..=859 {
            let eta = 0.01 + 0.0005 * i as f64;
            let o = outcome(Loss::Two, eta, start);
            assert!(matches!(o, Outcome::Ends(_)), "{eta}: {o:?}");
        }
        assert!(matches!(outcome(Loss::Two, 0.4395, start), Outcome::Ends(_)));
        for i in 0..=3120 {
            let eta = 0.4396 + 0.0005 * i as f64;
            assert_eq!(outcome(Loss::Two, eta, start), Outcome::Not(Ending::FlewOff), "{eta}");
        }
    }

    /// The panel's count over a grid of starts: of the runs that started too sharp
    /// for their step (the floor beneath the start over 2/η) and settled, every one of
    /// the 589 on the second loss ended between 2/η − 20η/3 and 2/η (more than half
    /// above 0.99 of 2/η); the 542 on the first between 0.20 and 0.995 of 2/η, half below 0.70.
    #[test]
    fn over_a_grid_of_starts_the_two_losses_end_differently() {
        let starts: Vec<DVec2> = (0..15).flat_map(|i| [0.9, 0.97, 1.03, 1.1].map(move |p| (1.2 + 0.2 * i as f64, p))).map(|(a, p)| DVec2::new(a, p / a)).collect();
        assert_eq!(starts.len(), 60);
        for loss in LOSSES {
            let mut shares = Vec::new();
            let mut inside = true;
            for start in &starts {
                for i in 0..SWEEP {
                    let eta = sweep_eta(i);
                    if beneath(loss, *start) <= 2.0 / eta {
                        continue;
                    }
                    if let Outcome::Ends(s) = outcome(loss, eta, *start) {
                        shares.push(s * eta / 2.0);
                        inside &= s < 2.0 / eta && s > 2.0 / eta - 20.0 * eta / 3.0;
                    }
                }
            }
            shares.sort_by(f64::total_cmp);
            let median = shares[shares.len() / 2];
            println!("{loss:?}: {} runs settled; shares of 2/eta from {:.5} to {:.5}, median {:.5}; all inside the interval: {inside}", shares.len(), shares[0], shares[shares.len() - 1], median);
            match loss {
                Loss::Two => assert!(shares.len() == 542 && num(shares[0], 2) == "0.20" && num(shares[shares.len() - 1], 3) == "0.995" && median < 0.70 && !inside),
                Loss::Four => assert!(inside && shares.len() == 589 && median > 0.99),
            }
        }
    }

    /// Whatever is asked, by buttons, typed numbers or hands, every number drawn stays
    /// finite, the words say nothing that is not a number, and the scene comes to rest.
    #[test]
    fn degenerate_input_leaves_everything_finite_and_comes_to_rest() {
        let starts = [
            DVec2::ZERO,
            DVec2::new(0.0, 1.5),
            DVec2::new(1.5, 0.0),
            DVec2::new(2.0, 0.5),
            DVec2::new(1.0, 1.0),
            DVec2::new(0.8, -0.8),
            DVec2::new(-1.0, 1.0),
            DVec2::new(6.0, 6.0),
            DVec2::new(-6.0, 0.01),
            DVec2::new(f64::NAN, 1.0),
            DVec2::new(1e30, -1e30),
        ];
        for loss in LOSSES {
            for start in starts {
                for eta in [ETA_LO, 0.3, ETA_HI, f64::NAN, 1e9] {
                    let mut scene = EosToy::new();
                    scene.ask(loss, eta, start);
                    for _ in 0..60 * 40 {
                        scene.advance(1.0 / 60.0, 1.0 / 60.0);
                        if scene.at_rest() {
                            break;
                        }
                    }
                    assert!(scene.at_rest(), "{loss:?} {start:?} {eta}");
                    let r = &scene.run;
                    assert!(scene.eta.is_finite() && scene.start.is_finite() && scene.start.abs().max_element() <= START_MAX);
                    assert!(r.z.iter().all(|z| z.is_finite()) && r.value.iter().all(|v| !v.is_nan()) && r.sharp.iter().all(|v| !v.is_nan()));
                    let flow = flow_path(loss, scene.start, 80);
                    assert!(flow.len() == 80 && flow.iter().all(|p| p.is_finite()));
                    for s in [0.0, 0.5, scene.film] {
                        assert!(dots(&r.z, s).iter().all(|d| d.is_finite()) && along(&r.z, s).is_finite());
                    }
                    assert!((0..4).all(|i| scene.ranges.get(i).is_finite()));
                    let words = [scene.status(), scene.ending_words()].join(" ");
                    assert!(!words.contains("NaN") && !words.contains("inf") && !words.contains('?'), "{words}");
                    // a hand on the start, then let go: at rest again, the whole run shown
                    scene.hand(scene.eta, start * 0.5 + DVec2::new(0.1, 0.0));
                    for _ in 0..60 * 5 {
                        scene.advance(1.0 / 60.0, 1.0 / 60.0);
                    }
                    assert!(scene.at_rest() && scene.film_done());
                }
            }
        }
        // a start exactly on a minimum too sharp to stay at: the slope is 0 there
        let stuck = Run::new(Loss::Two, 0.5, DVec2::new(2.0, 0.5));
        assert!(stuck.ending == Ending::Stuck && stuck.steps() == 0);
        let mut scene = EosToy::new();
        scene.hand(0.5, DVec2::new(2.0, 0.5));
        assert!(scene.status().contains("the start is a minimum, too sharp") && scene.run.under[0] * 0.5 > 2.0, "{}", scene.status());
        scene.hand(0.5, DVec2::ZERO);
        assert!(scene.status().contains("which is not a minimum"), "{}", scene.status());
        // and one exactly on a minimum flat enough has settled where it is
        scene.hand(0.5, DVec2::new(1.25, 0.8));
        assert!(scene.run.ending == Ending::Settled && scene.run.steps() == 0 && scene.status().starts_with("settled after 0 steps"));
        // the first loss from the line a = −b runs into the saddle at the origin
        let saddle = Run::new(Loss::Two, 0.1, DVec2::new(0.8, -0.8));
        assert!(saddle.ending == Ending::Stuck && saddle.last().length() < 1e-6, "{:?} {:?}", saddle.ending, saddle.last());
    }

    /// A new setting glides there, the old path fading, and the film plays only once
    /// it has arrived: the start never jumps.
    #[test]
    fn a_new_setting_glides_and_the_film_waits_for_it() {
        let mut scene = EosToy::new();
        for _ in 0..100 {
            scene.advance(1.0 / 60.0, 1.0 / 60.0);
        }
        scene.set_up(4);
        let mut last = scene.glide.shown.start;
        let mut played = false;
        for _ in 0..600 {
            scene.advance(1.0 / 60.0, 1.0 / 60.0);
            let now = scene.glide.shown.start;
            assert!((now - last).length() < 0.15, "{now:?} after {last:?}");
            if scene.film > 0.0 && !played {
                assert!(scene.glide.arrived() && scene.old.is_none());
                played = true;
            }
            last = now;
        }
        assert!(played && scene.start == DVec2::from(MIDDLE));
    }

    /// The panel and the picture fit the canvas, from a phone to a desktop, and the
    /// charts and the buttons stay inside the panel.
    #[test]
    fn the_layout_fits_every_canvas() {
        for (w, h) in [(1080.0, 1000.0), (1078.0, 1000.0), (580.0, 700.0), (390.0, 800.0), (390.0, 560.0), (800.0, 600.0), (1600.0, 1000.0), (240.0, 180.0), (620.0, 480.0), (1300.0, 700.0), (700.0, 1000.0)] {
            for inset in [0.0, 46.0] {
                let view = Vec2::new(w, h);
                let lay = Layout::of(view, inset);
                let (min, max) = lay.picture;
                assert!(min.cmpge(Vec2::ZERO).all() && max.cmple(view).all() && (max - min).min_element() > 50.0, "{w}x{h}: {lay:?}");
                if let Some(b) = &lay.board {
                    let within = |q: Vec2| q.cmpge(b.min - Vec2::splat(0.5)).all() && q.cmple(b.max + Vec2::splat(0.5)).all();
                    assert!(b.min.cmpge(Vec2::ZERO).all() && b.max.cmple(view + Vec2::splat(0.5)).all(), "{w}x{h}: {b:?}");
                    assert!(b.buttons.iter().all(|r| within(r.0) && within(r.1)), "{w}x{h}: {b:?}");
                    for (title, rect) in b.charts.iter().flatten() {
                        assert!(within(*title) && within(rect.0 - Vec2::new(LEFT, 0.0)) && within(rect.1 + Vec2::new(0.0, BELOW)), "{w}x{h}: {b:?}");
                        assert!(rect.1.x - rect.0.x > 100.0 && rect.1.y - rect.0.y >= 50.0, "{w}x{h}: {rect:?}");
                    }
                    assert!(b.min.y >= inset || lay.narrow, "{w}x{h}: under the buttons");
                    assert!(b.charts[0].is_some() && b.charts[1].is_some());
                }
            }
        }
    }
}


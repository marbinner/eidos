//! Step size and curvature: gradient descent on a bowl, one of the bowl's own directions
//! at a time. Along a direction of curvature λ each step multiplies the distance left by
//! 1 − ηλ, and everything else follows from that one factor: the sharpest direction sets
//! the longest step that is safe, and the flattest sets how long the descent takes.

use super::*;
use crate::scenes::stats::plot::{self, Plot, Rect};
use glam::DVec2;

pub const DEF: SceneDef = SceneDef {
    id: "stepsize",
    title: "Step size and curvature",
    group: GROUP,
    blurb: "Gradient descent on a bowl, one direction at a time: the sharpest direction sets the longest safe step, and the flattest one sets how long the descent takes.",
    make: |_| Box::new(StepSize::new()),
};

// ----- the bowl, in double precision ---------------------------------------------------

/// The fewest and the most directions a bowl may have.
const FEWEST: usize = 2;
const MOST: usize = 12;
/// The range of the sharpest curvature (the flattest goes from 0 up to it).
const SHARPEST: [f64; 2] = [0.1, 4.0];
/// The smallest curvature above 0 the flattest one's slider offers.
const FLATTEST_LEAST: f64 = 0.001;
/// How far from the bottom the start may be put, each way, in the picture's units.
const REACH: DVec2 = DVec2::new(2.45, 1.7);
/// Where the start is along the directions the picture does not show: this far from the
/// bottom along each.
const OTHERS_START: f64 = 1.0;
/// The step's scale runs from 0 to this many times 1/λmax: a quarter beyond the limit.
const KNOB_END: f64 = 2.5;
/// How much lower the loss must be to count as "down a thousandfold".
const FALL: f64 = 1000.0;

/// How the curvatures between the sharpest and the flattest are spread.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Spread {
    Even,
    Log,
    /// A few large ones and a bulk near the flattest, as measured Hessians of trained
    /// classifiers are reported to look (see the panel's note).
    FewSharp,
}

const SPREADS: [(Spread, &str); 3] = [(Spread::Even, "evenly"), (Spread::Log, "evenly on a log scale"), (Spread::FewSharp, "a few sharp, the rest flat")];

/// The curvatures of a bowl of `n` directions, sharpest first: `lmax` and `lmin` at the
/// two ends, the others spread between them.
fn curvatures(lmax: f64, lmin: f64, n: usize, spread: Spread) -> Vec<f64> {
    let n = n.clamp(FEWEST, MOST);
    let lmax = if lmax.is_finite() { lmax.max(0.0) } else { 1.0 };
    let lmin = if lmin.is_finite() { lmin.max(0.0).min(lmax) } else { 0.0 };
    let share = |i: usize, of: usize| if of > 1 { i as f64 / (of - 1) as f64 } else { 0.0 };
    let mut out: Vec<f64> = match spread {
        Spread::Even => (0..n).map(|i| lmax + (lmin - lmax) * share(i, n)).collect(),
        Spread::Log => {
            // (0 has no place on a log scale: the others go down to a thousandth of the sharpest)
            let low = lmin.max(lmax * 1e-3);
            (0..n).map(|i| if lmax > 0.0 { lmax * (low / lmax).powf(share(i, n)) } else { 0.0 }).collect()
        }
        Spread::FewSharp => {
            // a quarter of them large, between λmax and half of it; the rest in the lowest
            // twentieth of the range
            let sharp = ((n as f64 / 4.0).round() as usize).clamp(1, n - 1);
            let top = lmin + 0.05 * (lmax - lmin);
            let low_sharp = (0.5 * lmax).max(top);
            (0..n)
                .map(|i| if i < sharp { lmax * (low_sharp / lmax.max(1e-300)).powf(share(i, sharp)) } else { top + (lmin - top) * share(i - sharp, n - sharp) })
                .collect()
        }
    };
    out.sort_by(|a, b| b.total_cmp(a));
    out[0] = lmax;
    out[n - 1] = lmin;
    out
}

/// The bowl's sharpest and flattest directions in the picture: the flattest at the angle
/// `turn` from the x axis, the sharpest a right angle further round.
fn axes(turn: f64) -> (DVec2, DVec2) {
    let flat = DVec2::from_angle(turn);
    (flat.perp(), flat)
}

/// What a run of gradient descent comes to in the end.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Fate {
    /// The start is the bottom: the gradient is 0 and nothing moves.
    AtBottom,
    /// Every direction that holds any loss shrinks: the loss goes to 0.
    Converges,
    /// A direction that holds loss neither shrinks nor grows (its factor is 1 or −1),
    /// and none grows: the loss levels off above 0.
    Stays,
    /// A direction that holds loss grows: in the end the loss climbs for ever.
    Grows,
}

/// How near 1 the size of a factor must be to count as 1.
const ONE: f64 = 1e-12;

/// Gradient descent with the step η on the bowl L = ½ Σ λᵢ xᵢ², where xᵢ is the distance
/// from the bottom along the bowl's i-th own direction. The gradient along that direction
/// is λᵢxᵢ, so a step takes xᵢ to (1 − ηλᵢ)xᵢ: everything here is in closed form.
#[derive(Clone, Debug, PartialEq)]
struct Descent {
    /// The curvatures, sharpest first.
    lam: Vec<f64>,
    /// The start's distance from the bottom along each direction, with its sign.
    start: Vec<f64>,
    eta: f64,
}

impl Descent {
    fn n(&self) -> usize {
        self.lam.len().min(self.start.len())
    }

    /// The factor that one step multiplies direction i's distance by.
    fn factor(&self, i: usize) -> f64 {
        1.0 - self.eta * self.lam[i]
    }

    /// Direction i's part of the loss at the start, ½λc².
    fn share(&self, i: usize) -> f64 {
        0.5 * self.lam[i] * self.start[i] * self.start[i]
    }

    fn loss0(&self) -> f64 {
        (0..self.n()).map(|i| self.share(i)).sum()
    }

    /// The directions that hold any loss at the start.
    fn loaded(&self) -> impl Iterator<Item = usize> + '_ {
        (0..self.n()).filter(|&i| self.share(i) > 0.0)
    }

    /// The natural logarithm of direction i's loss after t steps, ½λc²(1 − ηλ)^{2t}
    /// (−∞ for none). In logarithms, so that neither a tiny loss nor a huge one is lost.
    fn ln_share(&self, i: usize, t: f64) -> f64 {
        let s = self.share(i);
        if s.is_nan() || s <= 0.0 {
            return f64::NEG_INFINITY;
        }
        if t <= 0.0 {
            return s.ln();
        }
        let f = self.factor(i).abs();
        if f == 0.0 {
            return f64::NEG_INFINITY;
        }
        s.ln() + 2.0 * t * f.ln()
    }

    /// The natural logarithm of the whole loss after t steps (−∞ for none).
    fn ln_loss(&self, t: f64) -> f64 {
        let most = (0..self.n()).map(|i| self.ln_share(i, t)).fold(f64::NEG_INFINITY, f64::max);
        if !most.is_finite() {
            return most;
        }
        most + (0..self.n()).map(|i| (self.ln_share(i, t) - most).exp()).sum::<f64>().ln()
    }

    fn fate(&self) -> Fate {
        if self.loaded().next().is_none() {
            return Fate::AtBottom;
        }
        if self.loaded().any(|i| self.factor(i).abs() > 1.0 + ONE) {
            Fate::Grows
        } else if self.loaded().any(|i| self.factor(i).abs() >= 1.0 - ONE) {
            Fate::Stays
        } else {
            Fate::Converges
        }
    }

    /// The worst factor: the largest |1 − ηλ| among the directions with any curvature
    /// (a direction with none holds no loss, whatever the start), and which it is.
    fn worst(&self) -> Option<(usize, f64)> {
        (0..self.n()).filter(|&i| self.lam[i] > 0.0).map(|i| (i, self.factor(i).abs())).fold(None, |best: Option<(usize, f64)>, (i, r)| match best {
            Some((_, b)) if b >= r => best,
            _ => Some((i, r)),
        })
    }

    /// From any start, the loss is `fall` times lower after at most this many steps:
    /// every direction's loss shrinks at least by r² per step, r the worst factor, so
    /// after ln(fall)/(−2 ln r) steps the whole loss has too. None if r ≥ 1.
    fn bound(&self, fall: f64) -> Option<f64> {
        let (_, r) = self.worst()?;
        (r < 1.0).then(|| if r == 0.0 { 0.0 } else { fall.ln() / (-2.0 * r.ln()) })
    }

    /// The same as a whole number of steps (at least one).
    fn bound_steps(&self, fall: f64) -> Option<u64> {
        self.bound(fall).filter(|b| b.is_finite()).map(|b| (b.ceil().min(1e15) as u64).max(1))
    }

    /// The first step after which the loss is `fall` times lower than at the start, if
    /// it ever is.
    fn steps_to(&self, fall: f64) -> Option<u64> {
        let l0 = self.ln_loss(0.0);
        if !l0.is_finite() || fall.is_nan() || fall <= 1.0 {
            return None;
        }
        let target = l0 - fall.ln();
        let below = |t: u64| self.ln_loss(t as f64) <= target;
        let whole = |x: f64| (x.is_finite() && x >= 0.0).then(|| x.ceil().min(1e15) as u64);
        // a step by which the loss is surely low enough if it ever is, and up to which it
        // only falls
        let end = match self.fate() {
            Fate::AtBottom => return None,
            Fate::Grows => self.lowest()?,
            Fate::Converges => {
                let q = self.loaded().map(|i| self.factor(i).abs()).fold(0.0f64, f64::max);
                if q == 0.0 {
                    1
                } else {
                    whole(fall.ln() / (-2.0 * q.ln()))?.max(1)
                }
            }
            Fate::Stays => {
                // the loss levels off at what the directions that keep their size hold
                let kept: f64 = self.loaded().filter(|&i| self.factor(i).abs() >= 1.0 - ONE).map(|i| self.share(i)).sum();
                let total = self.loss0();
                let room = total / fall - kept;
                if room.is_nan() || room <= 0.0 {
                    return None;
                }
                let q = self.loaded().map(|i| self.factor(i).abs()).filter(|r| *r < 1.0 - ONE).fold(0.0f64, f64::max);
                if q == 0.0 {
                    1
                } else {
                    whole(((total - kept) / room).max(1.0).ln() / (-2.0 * q.ln()))?.max(1)
                }
            }
        };
        if !below(end) {
            return None;
        }
        // (the loss at the start is above the target, and it only falls up to `end`)
        let (mut lo, mut hi) = (0u64, end);
        while hi - lo > 1 {
            let mid = lo + (hi - lo) / 2;
            if below(mid) {
                hi = mid;
            } else {
                lo = mid;
            }
        }
        Some(hi)
    }

    /// For a run that grows in the end: the step at which the loss is lowest. Before it
    /// the loss falls at every step, after it it climbs at every step: a sum of
    /// exponentials in t, sᵢ(1 − ηλᵢ)^{2t}, has a single lowest point.
    fn lowest(&self) -> Option<u64> {
        if self.fate() != Fate::Grows {
            return None;
        }
        let (first, start) = (self.ln_loss(1.0), self.ln_loss(0.0));
        if first.is_nan() || start.is_nan() || first >= start {
            return Some(0);
        }
        let low = self.lowest_between()?;
        let a = low.floor() as u64;
        Some(if self.ln_loss((a + 1) as f64) < self.ln_loss(a as f64) { a + 1 } else { a })
    }

    /// Where the loss, as a smooth function of t (each direction's part a geometric
    /// curve through its values at the steps), is lowest: for a run that grows.
    fn lowest_between(&self) -> Option<f64> {
        if self.fate() != Fate::Grows {
            return None;
        }
        // the slope of the loss against t is Σ sᵢ ln(fᵢ²) fᵢ^{2t}: its sign, each term
        // divided by the largest
        let slope = |t: f64| {
            let terms: Vec<(f64, f64)> =
                self.loaded().filter(|&i| self.factor(i) != 0.0).map(|i| (self.ln_share(i, t), 2.0 * self.factor(i).abs().ln())).collect();
            let most = terms.iter().map(|p| p.0).fold(f64::NEG_INFINITY, f64::max);
            terms.iter().map(|(a, k)| k * (a - most).exp()).sum::<f64>()
        };
        let mut hi = 1.0f64;
        while slope(hi).is_nan() || slope(hi) <= 0.0 {
            hi *= 2.0;
            if hi > 1e15 {
                return None;
            }
        }
        let mut lo = 0.0f64;
        for _ in 0..200 {
            if hi - lo < 1e-7 {
                break;
            }
            let mid = 0.5 * (lo + hi);
            if slope(mid) > 0.0 {
                hi = mid;
            } else {
                lo = mid;
            }
        }
        Some(0.5 * (lo + hi))
    }

    /// Where the loss, as a smooth function of t, first comes down to a `fall`th of the
    /// start: between the step before `steps_to` and that step (where the chart's curve
    /// crosses the line, and a point that moves smoothly as the step size does).
    fn crossing(&self, fall: f64) -> Option<f64> {
        let n = self.steps_to(fall)?;
        let target = self.ln_loss(0.0) - fall.ln();
        let (mut lo, mut hi) = (n.saturating_sub(1) as f64, n as f64);
        for _ in 0..60 {
            let mid = 0.5 * (lo + hi);
            if self.ln_loss(mid) <= target {
                hi = mid;
            } else {
                lo = mid;
            }
        }
        Some(hi)
    }

    /// The first step after `from` at which the loss is `rise` times what it was at the
    /// start (for a run that grows).
    fn steps_up(&self, from: u64, rise: f64) -> Option<u64> {
        let target = self.ln_loss(0.0) + rise.ln();
        let above = |t: u64| self.ln_loss(t as f64) >= target;
        let mut step = 1u64;
        while !above(from + step) {
            step = step.checked_mul(2)?;
            if step > 1 << 50 {
                return None;
            }
        }
        let (mut lo, mut hi) = (from + step / 2, from + step);
        if above(lo) {
            return Some(lo.max(from));
        }
        while hi - lo > 1 {
            let mid = lo + (hi - lo) / 2;
            if above(mid) {
                hi = mid;
            } else {
                lo = mid;
            }
        }
        Some(hi)
    }
}

/// The fewest and most steps the chart shows.
const SHORTEST: f64 = 12.0;
const LONGEST: f64 = 3000.0;

/// How many steps the chart and the path show: enough for the loss to come down a
/// millionfold, or to climb a hundredfold.
fn horizon(d: &Descent) -> f64 {
    let steps = match d.fate() {
        Fate::AtBottom => SHORTEST,
        Fate::Converges | Fate::Stays => d.steps_to(1e6).map_or(100.0, |t| t as f64),
        Fate::Grows => d.lowest().and_then(|low| d.steps_up(low, 100.0)).map_or(LONGEST, |t| t as f64),
    };
    nice_span((steps * 1.08 + 2.0).max(SHORTEST)).min(LONGEST)
}

/// The least round number at or above `t` that a fifth of is round too: 1, 1.5, 2, 3,
/// 4, 5, 6 or 8 times a power of ten. The chart's steps run from 0 to such a number,
/// with a tick at every fifth of it, so that its ticks keep their number and only slide
/// while the span glides from one to another.
fn nice_span(t: f64) -> f64 {
    if !t.is_finite() || t <= 0.0 {
        return SHORTEST;
    }
    let power = 10f64.powf(t.log10().floor());
    [1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0].iter().map(|m| m * power).find(|v| *v >= t * (1.0 - 1e-12)).unwrap_or(10.0 * power)
}

// ----- what glides ------------------------------------------------------------------------

/// Everything the picture is drawn from.
#[derive(Clone, Debug, PartialEq)]
struct Setting {
    /// The curvatures, sharpest first (at least two).
    lam: Vec<f64>,
    /// The angle of the flattest direction from the x axis.
    turn: f64,
    /// The start, in the plane of the sharpest and the flattest direction.
    start: DVec2,
    eta: f64,
}

impl Setting {
    fn sharpest(&self) -> f64 {
        self.lam.first().copied().unwrap_or(1.0)
    }

    fn flattest(&self) -> f64 {
        self.lam.last().copied().unwrap_or(0.0)
    }

    /// The start along the sharpest and the flattest direction.
    fn along(&self) -> (f64, f64) {
        let (sharp, flat) = axes(self.turn);
        (self.start.dot(sharp), self.start.dot(flat))
    }

    fn descent(&self) -> Descent {
        let n = self.lam.len();
        let mut start = vec![OTHERS_START; n];
        if n >= 2 {
            let (sharp, flat) = self.along();
            start[0] = sharp;
            start[n - 1] = flat;
        }
        Descent { lam: self.lam.clone(), start, eta: self.eta }
    }

    /// Where the point is after t steps, in the picture (the sharpest and the flattest
    /// direction only: the others are not in the plane).
    fn point(&self, t: u64) -> DVec2 {
        let (sharp, flat) = axes(self.turn);
        let (a, b) = self.along();
        let t = t.min(i32::MAX as u64) as i32;
        let factor = |l: f64| 1.0 - self.eta * l;
        sharp * a * factor(self.sharpest()).powi(t) + flat * b * factor(self.flattest()).powi(t)
    }

    fn numbers(&self) -> Vec<f64> {
        let mut v = vec![self.turn, self.start.x, self.start.y, self.eta];
        v.extend_from_slice(&self.lam);
        v
    }

    fn set_numbers(&mut self, v: &[f64]) {
        (self.turn, self.start.x, self.start.y, self.eta) = (v[0], v[1], v[2], v[3]);
        self.lam.copy_from_slice(&v[4..]);
    }
}

/// How fast a change is glided to: each of two eases, one after the other, covers two
/// thirds of what is left in this many seconds.
const FOLLOW: f64 = 0.09;

/// A setting as asked for, and as drawn: the drawn one follows it through `between`, two
/// eases one after the other, so that it sets off gently and arrives exactly.
#[derive(Clone, Debug)]
struct Glide {
    to: Setting,
    between: Setting,
    shown: Setting,
}

impl Glide {
    fn at(s: Setting) -> Glide {
        Glide { to: s.clone(), between: s.clone(), shown: s }
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
        if self.arrived() || dt.is_nan() || dt <= 0.0 {
            return;
        }
        // a bowl of another number of directions: its sharpest and flattest glide from
        // where they are, the ones between appear where they are going
        for s in [&mut self.between, &mut self.shown] {
            if s.lam.len() != self.to.lam.len() {
                let (first, last) = (s.sharpest(), s.flattest());
                s.lam.clone_from(&self.to.lam);
                let n = s.lam.len();
                (s.lam[0], s.lam[n - 1]) = (first, last);
            }
        }
        let k = 1.0 - (-dt / FOLLOW).exp();
        let to = self.to.numbers();
        let (mut between, mut shown) = (self.between.numbers(), self.shown.numbers());
        let mut apart = 0.0f64;
        for i in 0..to.len() {
            between[i] += (to[i] - between[i]) * k;
            shown[i] += (between[i] - shown[i]) * k;
            apart = apart.max((to[i] - between[i]).abs().max((to[i] - shown[i]).abs()) / to[i].abs().max(1.0));
        }
        // (exactly there in the end: the picture then stops changing)
        if apart < 2e-5 || !apart.is_finite() {
            *self = Glide::at(self.to.clone());
        } else {
            self.between.set_numbers(&between);
            self.shown.set_numbers(&shown);
        }
    }
}

// ----- experiments ------------------------------------------------------------------------

/// The step an experiment takes.
#[derive(Clone, Copy, Debug)]
enum Step {
    At(f64),
    /// The best fixed step, 2/(λmax + λmin).
    Best,
    /// This many times the limit 2/λmax.
    OfLimit(f64),
}

struct Trial {
    label: &'static str,
    look_for: &'static str,
    lmax: f64,
    lmin: f64,
    n: usize,
    spread: Spread,
    step: Step,
    /// The start along the sharpest and the flattest direction.
    start: [f64; 2],
}

/// How every experiment turns the bowl, and where it starts (along the sharpest and the
/// flattest direction), unless it says otherwise.
const TURN_DEGREES: f64 = 25.0;
const START: [f64; 2] = [0.8, 1.6];

const TRIALS: [Trial; 5] = [
    Trial {
        label: "a short step",
        look_for: r"$\eta = 0.2$, a tenth of the limit, on a bowl of four directions. Every factor is between $0$ and $1$ (from $0.80$ to $0.98$), so every direction creeps in and none overshoots. The loss is a thousand times lower after 127 steps. The sharper directions are gone early; what is left is the flattest, whose loss shrinks by a factor $(1 - \eta\lambda_{\min})^2 = 0.9604$ per step: on its own it would need $\ln 1000 / (-\ln 0.9604) = 171$ steps, but it starts with only 16% of the loss. No step up to $1/\lambda_{\max}$, the first mark on the scale, has a factor below $0$.",
        lmax: 1.0,
        lmin: 0.1,
        n: 4,
        spread: Spread::Log,
        step: Step::At(0.2),
        start: START,
    },
    Trial {
        label: "the best fixed step",
        look_for: r"$\eta = 2/(\lambda_{\max} + \lambda_{\min}) = 1.82$. The sharpest direction's factor is $-0.82$ and the flattest's $+0.82$: one flips sides at every step, the other creeps, and both shrink at the same rate, $(\kappa - 1)/(\kappa + 1)$ with $\kappa = 10$. On the chart their two lines are parallel. A longer step slows the sharpest, a shorter one the flattest, so no fixed step has a smaller worst factor: from any start the loss is a thousand times lower after at most 18 steps, here after 18. (A step that suits one start better can do a little better there: from this start $\eta = 1.6$ takes 17.)",
        lmax: 1.0,
        lmin: 0.1,
        n: 2,
        spread: Spread::Log,
        step: Step::Best,
        start: START,
    },
    Trial {
        label: "just under the limit",
        look_for: r"$\eta = 1.96$, 98% of the limit $2/\lambda_{\max}$. The sharpest direction's factor is $-0.96$: it flips sides at every step and shrinks by only 4% each time, so the path zigzags across the valley for a long time, while the flattest direction's loss (factor $0.80$) is a thousand times smaller after 16 steps. The whole loss is a thousand times lower after 81 steps, against 18 at the best step: now the sharpest direction sets the time.",
        lmax: 1.0,
        lmin: 0.1,
        n: 2,
        spread: Spread::Log,
        step: Step::OfLimit(0.98),
        start: START,
    },
    Trial {
        label: "just over the limit",
        look_for: r"$\eta = 2.1$, 5% over the limit. The sharpest direction's factor is $-1.10$: it flips sides and grows by a tenth at every step. The start is almost on the valley's floor, so at first the other three directions, which still shrink, carry the loss down: it falls for 11 steps, to 0.5% of where it started. By then the sharpest direction has grown from almost nothing, and from there on the loss climbs at every step, for ever: one direction ruins the descent while all the others converge.",
        lmax: 1.0,
        lmin: 0.1,
        n: 4,
        spread: Spread::Log,
        step: Step::OfLimit(1.05),
        start: [0.02, 1.6],
    },
    Trial {
        label: "a long bowl",
        look_for: r"$\lambda_{\min} = 0.01$: the curvatures are a hundred times apart, $\kappa = 100$, so the level lines are ten times longer than wide. At its best step, $\eta = 1.98$, both extreme directions shrink by a factor $(\kappa - 1)/(\kappa + 1) = 0.980$ per step, about $1 - 2/\kappa$: the loss is a thousand times lower after 173 steps, against 18 for $\kappa = 10$. Ten times $\kappa$, about ten times the steps: each fall of the distance by a factor $e$ takes about $\kappa/2 = 50$ steps.",
        lmax: 1.0,
        lmin: 0.01,
        n: 2,
        spread: Spread::Log,
        step: Step::Best,
        start: START,
    },
];

/// The step a trial takes, on the bowl with these extreme curvatures.
fn step_of(step: Step, lmax: f64, lmin: f64) -> f64 {
    match step {
        Step::At(eta) => eta,
        Step::Best => 2.0 / (lmax + lmin),
        Step::OfLimit(share) => share * 2.0 / lmax,
    }
}

/// A start given along the sharpest and the flattest direction, in the picture.
fn start_at([sharp, flat]: [f64; 2], turn: f64) -> DVec2 {
    let (u_sharp, u_flat) = axes(turn);
    u_sharp * sharp + u_flat * flat
}

/// The scene as it opens: the bowl of the experiments, two directions, a step between
/// 1/λmax and the limit.
const OPENING_STEP: f64 = 1.6;

// ----- where things go ------------------------------------------------------------------------

/// The part of the plane the picture is fitted to (half its size, centred on the bottom).
const FRAME_HALF: Vec2 = Vec2::new(2.5, 1.75);
const PAD: f32 = 12.0;
/// The width of the board on a wide canvas.
const BOARD_W: f32 = 392.0;
/// The height of the step's scale (its line runs across the middle).
const KNOB_H: f32 = 20.0;
/// The space a row of the factors' chart takes, on a wide canvas and on a phone.
const ROW_WIDE: f32 = 21.0;
const ROW_NARROW: f32 = 13.0;

/// The panel drawn on the picture: the step's scale, each direction's factor, and the
/// loss against the step. In logical pixels.
#[derive(Clone, Debug, PartialEq)]
struct Board {
    min: Vec2,
    max: Vec2,
    text: f32,
    small: f32,
    knob_title: Vec2,
    /// The step's scale: the line runs across the middle of this rectangle.
    knob: Rect,
    rows_title: Vec2,
    /// The factors' chart: the rows fill it (the names of its parts above it, the
    /// numbers of its axis below it, the curvatures to its left).
    rows: Rect,
    chart_title: Vec2,
    chart: Rect,
}

#[derive(Clone, Debug, PartialEq)]
struct Layout {
    /// A phone held upright: the board goes below the picture.
    narrow: bool,
    /// The part of the canvas the bowl has to itself.
    picture: (Vec2, Vec2),
    /// None on a canvas too small to spare the room (the numbers are in the side panel).
    board: Option<Board>,
}

impl Layout {
    /// The layout for a canvas of `view` logical pixels, `inset` of its top under the
    /// application's buttons, with `rows` rows of factors (gliding between whole numbers
    /// when the bowl gains or loses directions).
    fn of(view: Vec2, inset: f32, rows: f32) -> Layout {
        let view = view.max(Vec2::ONE);
        let rows = if rows.is_finite() { rows.max(FEWEST as f32).min(MOST as f32) } else { FEWEST as f32 };
        let narrow = view.x < 620.0 && view.y >= 560.0;
        let whole = Layout { narrow, picture: (Vec2::new(0.0, inset), view), board: None };
        let (text, small) = if narrow { (12.0, 10.5) } else { (12.5, 11.0) };
        let pad = if narrow { 10.0 } else { PAD };
        let pitch = if narrow { ROW_NARROW } else { ROW_WIDE };
        // from the board's top: the step's title, the names over its scale, the scale,
        // the names under it; the factors' title, the names of its parts, its rows, its
        // numbers; the chart's title and key, the chart, its numbers
        let knob_h = text + 2.0 + small + 2.0 + KNOB_H + small + 4.0;
        let rows_h = text + 6.0 + small + 6.0 + rows * pitch + small + 10.0;
        // (on a phone the chart's key is left to the panel)
        let chart_above = if narrow { text + 12.0 } else { 2.0 * (text + 4.0) + 8.0 };
        let chart_below = small + 22.0;
        let place = |min: Vec2, width: f32, chart_h: f32| {
            let (x, right) = (min.x + pad, min.x + width - pad);
            let mut y = min.y + pad;
            let knob_title = Vec2::new(x, y);
            y += text + 2.0 + small + 2.0;
            // (tall enough to hold its handle: a chart lower than 18 pixels takes none)
            let knob = (Vec2::new(x + 8.0, y), Vec2::new(right - 8.0, y + KNOB_H));
            y += KNOB_H + small + 4.0 + 8.0;
            let rows_title = Vec2::new(x, y);
            y += text + 6.0 + small + 6.0;
            // (room at the left for the curvatures, at the right for the factors' values)
            let factors = (Vec2::new(x + if narrow { 60.0 } else { 70.0 }, y), Vec2::new(right - if narrow { 44.0 } else { 50.0 }, y + rows * pitch));
            y += rows * pitch + small + 10.0 + 6.0;
            let chart_title = Vec2::new(x, y);
            y += chart_above;
            let chart = (Vec2::new(x + 34.0, y), Vec2::new(right - 8.0, y + chart_h));
            Board { min, max: Vec2::new(min.x + width, chart.1.y + chart_below + pad), text, small, knob_title, knob, rows_title, rows: factors, chart_title, chart }
        };
        let fixed = 2.0 * pad + knob_h + 8.0 + rows_h + 6.0 + chart_above + chart_below;
        if narrow {
            // below the picture, as tall as its rows need: the picture keeps the rest
            // (and is fitted again when the rows change)
            let chart_h = (view.y * 0.58 - fixed).min(140.0);
            let height = fixed + chart_h;
            let top = view.y - height;
            if chart_h < 60.0 || top - inset < 150.0 {
                return whole;
            }
            let mut board = place(Vec2::new(0.0, top), view.x, chart_h);
            board.max = view;
            return Layout { narrow, picture: (Vec2::new(0.0, inset), Vec2::new(view.x, top)), board: Some(board) };
        }
        let width = BOARD_W.min(view.x * 0.42);
        let top = inset + 12.0;
        let chart_h = (view.y - 12.0 - top - fixed).min(380.0);
        if view.x < 640.0 || width < 300.0 || chart_h < 70.0 {
            return whole;
        }
        let board = place(Vec2::new(view.x - width - 12.0, top), width, chart_h);
        let picture = (Vec2::new(0.0, inset), Vec2::new(board.min.x - 8.0, view.y));
        Layout { narrow, picture, board: Some(board) }
    }

    /// The part of the canvas the scene's frame is fitted into: the picture's, less the
    /// heading at its top and the caption at its foot.
    fn frame_room(&self) -> (Vec2, Vec2) {
        let (min, max) = self.picture;
        let foot = if self.narrow || self.board.is_none() { 8.0 } else { 40.0 };
        (min + Vec2::new(12.0, 52.0), max - Vec2::new(12.0, foot))
    }
}

fn inside_rect(p: Vec2, (min, max): (Vec2, Vec2)) -> bool {
    p.cmpge(min).all() && p.cmple(max).all()
}

// ----- drawing helpers ------------------------------------------------------------------------

/// The parts of the picture drawn in a batch of their own (different from the usual one by
/// a hair of its width): the path, which always has as many points but not always as many
/// seen, so that it and the rest can each be followed point for point from one frame to the
/// next (`eidos motion`).
#[derive(Clone, Copy, PartialEq)]
enum Part {
    Path = 1,
}

fn lines_for<'a>(f: &'a mut Frame, part: Part) -> &'a mut crate::gfx::draw::LineBatch {
    f.dl.lines_with(LineKind::Blend, LineStyle { width_scale: 1.0 + 1e-4 * part as u8 as f32, ..Default::default() })
}

/// The dots of the path's steps, in a batch of their own for the same reason.
fn step_dots<'a>(f: &'a mut Frame) -> &'a mut crate::gfx::draw::PointBatch {
    f.dl.points_with(PointKind::Disc, PointStyle { size_scale: 1.0001, ..Default::default() })
}

/// The line kinds of the board's charts (see `plot::apart`), each apart. (Each keeps its
/// number of points from frame to frame: what is not needed is put off the screen.)
const TOTAL: u8 = plot::OWN;
const SHARES: u8 = plot::OWN + 1;
const ROWS: u8 = plot::OWN + 2;
const COUNTS: u8 = plot::OWN + 3;

/// Every batch the scene draws into, asked for in the same order in every frame and
/// never left empty (an empty batch is dropped, and every batch after it would then be
/// compared with the wrong one).
fn hold_batches(f: &mut Frame) {
    let corner = f.hud(Vec2::ZERO);
    lines_for(f, Part::Path).segment(Vec3::ZERO, Vec3::ZERO, 0.0, Color::CLEAR);
    step_dots(f).push(corner, 0.0, Color::CLEAR);
    for kind in [plot::AXES, plot::CURVES, plot::MARKS, TOTAL, SHARES, ROWS, COUNTS] {
        plot::apart(f, kind, false).segment(corner, corner, 0.0, Color::CLEAR);
    }
    plot::apart(f, plot::MARKS, true).segment(corner, corner, 0.0, Color::CLEAR);
    plot::disc(f, Vec2::ZERO, 0.0, Color::CLEAR);
}

fn v3(p: DVec2) -> Vec3 {
    Vec3::new(p.x as f32, p.y as f32, 0.0)
}

/// The colour of direction i of n: the sharpest in the colour of curvature, the flatter
/// ones paler.
fn shade(i: usize, n: usize) -> Color {
    let t = if n > 1 { i as f32 / (n - 1) as f32 } else { 0.0 };
    CURVATURE.mix(pal::TEXT, 0.55 * t)
}

/// A number with two decimals and a true minus sign.
fn n2(v: f64) -> String {
    num(v, 2)
}

/// The points of the path drawn, and the dots on it (one per step).
const PATH_POINTS: usize = LONGEST as usize + 1;
const DOTS: usize = 150;

/// The level lines drawn in the bowl: these losses (each twice the one before).
const LEVELS: [f64; 8] = [0.02, 0.04, 0.08, 0.16, 0.32, 0.64, 1.28, 2.56];
/// Points on each half of a level line.
const HALF_POINTS: usize = 61;
/// How far out a level line or a direction is drawn (beyond any view of the frame).
const FAR: f64 = 14.0;
/// How far from the bottom the path is followed: past the frame's corners.
const SEEN: f64 = 3.6;

/// The y axis of the loss chart: powers of ten of the loss as a share of the start.
const LOG_LO: f64 = -7.0;
const LOG_HI: f64 = 2.5;

// ----- the scene --------------------------------------------------------------------------------

pub struct StepSize {
    glide: Glide,
    /// What the controls ask for: the extreme curvatures, how many directions and how the
    /// others are spread.
    lmax: f64,
    lmin: f64,
    n: usize,
    spread: Spread,
    /// The chart's last step as drawn, and the number of rows as drawn: both follow what
    /// the setting asked for needs, through two eases (the first number of each pair is
    /// the one between).
    horizon: f64,
    horizon_between: f64,
    rows: f64,
    rows_between: f64,
    /// The canvas last drawn on and how much of its top the application's buttons cover.
    canvas: Vec2,
    inset: f32,
    measured: bool,
    new_view: bool,
    /// The rows the picture was last fitted for (on a phone the board below it grows
    /// with them).
    fitted_rows: f32,
    /// Whether the step's dot on the board is being dragged.
    knob_held: bool,
}

impl StepSize {
    fn new() -> Self {
        let turn = TURN_DEGREES.to_radians();
        let lam = curvatures(1.0, 0.1, 2, Spread::Log);
        let s = Setting { lam, turn, start: start_at(START, turn), eta: OPENING_STEP };
        let horizon = horizon(&s.descent());
        StepSize { glide: Glide::at(s), lmax: 1.0, lmin: 0.1, n: 2, spread: Spread::Log, horizon, horizon_between: horizon, rows: 2.0, rows_between: 2.0, canvas: Vec2::new(1080.0, 1000.0), inset: 0.0, measured: false, new_view: false, fitted_rows: 2.0, knob_held: false }
    }

    /// Set up experiment `k` of `TRIALS`: the picture glides there.
    fn set_up(&mut self, k: usize) {
        let t = &TRIALS[k];
        (self.lmax, self.lmin, self.n, self.spread) = (t.lmax, t.lmin, t.n, t.spread);
        let turn = TURN_DEGREES.to_radians();
        self.glide.to = Setting { lam: curvatures(t.lmax, t.lmin, t.n, t.spread), turn, start: start_at(t.start, turn), eta: step_of(t.step, t.lmax, t.lmin) };
    }

    /// The controls of the bowl have changed: the picture follows at once.
    fn bowl_changed(&mut self) {
        self.lmax = if self.lmax.is_finite() { self.lmax.max(SHARPEST[0]).min(SHARPEST[1]) } else { 1.0 };
        self.lmin = if self.lmin.is_finite() { self.lmin.max(0.0).min(self.lmax) } else { 0.0 };
        self.n = self.n.clamp(FEWEST, MOST);
        let lam = curvatures(self.lmax, self.lmin, self.n, self.spread);
        let most = KNOB_END / self.lmax;
        self.glide.now(|s| {
            s.lam.clone_from(&lam);
            s.eta = s.eta.min(most);
        });
    }

    /// A hand on the step: at once.
    fn set_step(&mut self, eta: f64) {
        let eta = if eta.is_finite() { eta.max(0.0).min(KNOB_END / self.lmax) } else { 0.0 };
        self.glide.now(|s| s.eta = eta);
    }

    /// A hand on the start: at once.
    fn set_start(&mut self, p: DVec2) {
        if p.is_finite() {
            let p = p.clamp(-REACH, REACH);
            self.glide.now(|s| s.start = p);
        }
    }

    fn set_turn(&mut self, turn: f64) {
        if turn.is_finite() {
            self.glide.now(|s| s.turn = turn);
        }
    }

    fn shown(&self) -> &Setting {
        &self.glide.shown
    }

    /// Move everything that glides `dt` seconds on.
    fn advance(&mut self, dt: f32) {
        let dt = dt as f64;
        self.glide.step(dt);
        let k = 1.0 - (-dt.max(0.0) / FOLLOW).exp();
        // (two eases one after the other, as the glide's: off gently, there exactly)
        let follow = |shown: &mut f64, between: &mut f64, to: f64, near: f64| {
            if *shown == to && *between == to {
                return;
            }
            *between += (to - *between) * k;
            *shown += (*between - *shown) * k;
            if ((*shown - to).abs().max((*between - to).abs()) < near) || !shown.is_finite() || !between.is_finite() {
                (*shown, *between) = (to, to);
            }
        };
        let to = self.horizon_to();
        follow(&mut self.horizon, &mut self.horizon_between, to, 1e-4 * to.max(1.0));
        let rows = self.glide.to.lam.len() as f64;
        follow(&mut self.rows, &mut self.rows_between, rows, 1e-4);
    }

    /// The chart's span the setting asked for needs.
    fn horizon_to(&self) -> f64 {
        horizon(&self.glide.to.descent())
    }

    fn at_rest(&self) -> bool {
        self.glide.arrived() && self.horizon == self.horizon_to() && self.horizon_between == self.horizon && self.rows == self.glide.to.lam.len() as f64 && self.rows_between == self.rows
    }

    fn the_view(&self) -> View {
        let lay = Layout::of(self.canvas, self.inset, self.fitted_rows);
        let (min, max) = lay.frame_room();
        let room = (max - min).max(Vec2::splat(60.0));
        let per_unit = (room.x / (2.0 * FRAME_HALF.x)).min(room.y / (2.0 * FRAME_HALF.y)).max(1e-3);
        // (screen y runs down, the scene's up)
        let off = ((min + max) * 0.5 - self.canvas * 0.5) / per_unit;
        View::Fit { center: Vec2::new(-off.x, off.y), half: self.canvas.max(Vec2::ONE) * (0.5 / per_unit) }
    }

    /// The heading over the picture: what the run comes to.
    fn verdict(&self) -> String {
        let s = self.shown();
        let d = s.descent();
        match d.fate() {
            Fate::AtBottom => r"the start is the bottom: the gradient is $0$ and nothing moves".to_string(),
            // (with no curvature along the flattest direction the point stops on that
            // line, wherever it started along it)
            Fate::Converges if s.flattest() == 0.0 && s.along().1 != 0.0 => match d.steps_to(FALL) {
                Some(t) => format!("comes to rest on the floor line, not at the bottom: the loss is a thousand times lower after {}", steps_text(t)),
                None => "comes to rest on the floor line, not at the bottom".to_string(),
            },
            Fate::Converges => match d.steps_to(FALL) {
                Some(t) => format!("converges: the loss is a thousand times lower after {}", steps_text(t)),
                None => "converges".to_string(),
            },
            Fate::Stays if s.eta == 0.0 => r"$\eta = 0$: no step, and nothing moves".to_string(),
            Fate::Stays => "a direction flips sides for ever, neither shrinking nor growing: the loss levels off".to_string(),
            Fate::Grows => match d.lowest() {
                Some(0) | None => "the step is too long: the loss climbs from the first step on".to_string(),
                Some(t) => format!("the step is too long: the loss falls for {}, then climbs for ever", steps_text(t)),
            },
        }
    }
}

/// "1 step", "17 steps", "1,234 steps".
fn steps_text(t: u64) -> String {
    if t == 1 {
        "1 step".to_string()
    } else {
        format!("{} steps", count(t.min(usize::MAX as u64) as usize))
    }
}

impl Scene for StepSize {
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

    fn pointer_hint(&self) -> Option<&'static str> {
        Some("white dot: drag the start    cyan dot on the step's scale: drag the step")
    }

    fn ui(&mut self, ui: &mut egui::Ui, _: &Gfx) {
        ui::note(
            ui,
            r"Gradient descent steps against the gradient, $w \leftarrow w - \eta\, \nabla L(w)$, with a fixed step size $\eta$ (the learning rate). Here the loss is a bowl, $L = \tfrac12 \sum_i \lambda_i x_i^2$: $x_i$ is the distance from the bottom along the bowl's $i$-th own direction (an eigenvector of its Hessian), and $\lambda_i$ is the curvature along it. The gradient along that direction is $\lambda_i x_i$, so each direction goes its own way, and every step multiplies the distance left along it by the same factor:",
        );
        ui::equation(ui, r"x_i \;\leftarrow\; (1 - \eta\lambda_i)\, x_i");
        ui::note(ui, r"A factor between $0$ and $1$: the direction creeps in. At $0$: it is done in one step. Between $-1$ and $0$: it overshoots, flips sides at every step, and still shrinks. Below $-1$, which is $\eta > 2/\lambda_i$: it grows. The bowl from above shows its sharpest and flattest directions; the board, every direction's factor, and the loss step by step.");

        ui::section(ui, "Things to try");
        for (k, trial) in TRIALS.iter().enumerate() {
            if ui::experiment(ui, trial.label, trial.look_for) {
                self.set_up(k);
            }
        }

        ui::section(ui, "The step");
        let mut eta = self.glide.to.eta as f32;
        if ui::slider(ui, r"step $\eta$", &mut eta, 0.0..=(KNOB_END / self.lmax) as f32) {
            self.set_step(eta as f64);
        }
        ui::note(ui, r"Or drag the cyan dot on the scale over the picture. Its marks: $1/\lambda_{\max}$, the longest step with no factor below $0$ (nothing overshoots); the best fixed step; and the limit $2/\lambda_{\max}$, beyond which the sharpest direction grows.");

        ui::section(ui, "The bowl");
        let mut lmax = self.lmax as f32;
        let mut changed = ui::slider(ui, r"sharpest curvature $\lambda_{\max}$", &mut lmax, SHARPEST[0] as f32..=SHARPEST[1] as f32);
        if changed {
            self.lmax = lmax as f64;
            self.lmin = self.lmin.min(self.lmax);
        }
        let mut lmin = self.lmin as f32;
        if ui::slider_log_or_zero(ui, r"flattest curvature $\lambda_{\min}$", &mut lmin, FLATTEST_LEAST as f32, self.lmax as f32) {
            self.lmin = lmin as f64;
            changed = true;
        }
        ui::note(ui, r"At $\lambda_{\min} = 0$ the loss does not change along the flattest direction: the gradient has no part along it, so the point never moves along it, and every point of that line is a minimum. At $\lambda_{\min} = \lambda_{\max}$ the bowl is round, and every direction is one of its own.");
        changed |= ui::slider_int(ui, "directions $n$", &mut self.n, FEWEST..=MOST);
        if self.n > 2 {
            let mut k = SPREADS.iter().position(|(s, _)| *s == self.spread).unwrap_or(0);
            let names: Vec<&str> = SPREADS.iter().map(|(_, name)| *name).collect();
            if ui::choice(ui, &names, &mut k) {
                self.spread = SPREADS[k.min(SPREADS.len() - 1)].0;
                changed = true;
            }
            ui::note(ui, "How the curvatures between the sharpest and the flattest are spread. The directions not drawn in the bowl each start 1 from the bottom.");
            if self.spread == Spread::FewSharp {
                ui::note(ui, "Sagun and co-authors (2017) and Papyan (2018) measured the Hessians of trained classifiers and found a bulk of eigenvalues near zero and a few large ones, roughly one per class; Ghorbani and co-authors (2019) found the large ones almost absent in networks with batch normalisation. This bowl has that shape; it is not a network.");
            }
        }
        if changed {
            self.bowl_changed();
        }
        let mut turn = self.glide.to.turn.to_degrees() as f32;
        if ui::slider(ui, "turn the bowl", &mut turn, 0.0..=180.0) {
            self.set_turn((turn as f64).to_radians());
        }
        ui::note(ui, "A bowl's own directions need not be the axes. Turning it moves them under a start that stays put, and the path changes with them; drag the white dot to start elsewhere.");

        ui::section(ui, "Reading the picture");
        ui::key(
            ui,
            &[
                (pal::DIM, "grey: level lines of the loss, each for twice the loss of the one inside it"),
                (CURVATURE, r"green: the bowl's own directions and their curvatures $\lambda$, the sharpest bright, the flatter paler; on the chart, each direction's part of the loss"),
                (SGD, "cyan: gradient descent: its path, with a dot at each step, and the whole loss on the chart; the cyan dot on the scale is the step"),
                (pal::WHITE, "white dot: the start (drag it)"),
                (LIMIT, r"yellow: the limit $2/\lambda_{\max}$, and the factors below $-1$"),
            ],
        );

        ui::section(ui, "Words");
        ui::term(ui, r"Step size $\eta$:", "how far a step goes for a given slope (the learning rate).");
        ui::term(ui, r"Curvature $\lambda$:", r"how fast the slope changes along a direction. Along a bowl's own direction the loss is $\tfrac12 \lambda x^2$, and $\lambda$ is the Hessian's eigenvalue for it.");
        ui::term(ui, "Own directions:", r"the eigenvectors of the Hessian, at right angles to one another: the axes of the level ellipses. At a point on one of them the gradient points straight away from the bottom, so a step goes along that line, towards the bottom (and past it, if the step is too long).");
        ui::term(ui, "Factor:", r"$1 - \eta\lambda$, what one step multiplies the distance along a direction by: its size says how fast that direction shrinks (or grows), its sign whether the point stays on its side or flips.");
        ui::term(ui, r"Sharpness $\lambda_{\max}$:", r"the largest curvature. On a bowl a step longer than $2/\lambda_{\max}$ makes the sharpest direction grow. On a loss that is not a bowl the same holds near a minimum, approximately, since there the loss is close to a bowl whose curvatures are the Hessian's eigenvalues; and if the Hessian's largest eigenvalue is at most $\ell$ everywhere, every step shorter than $2/\ell$ lowers the loss (unless the gradient is $0$). But the curvature changes from place to place: Cohen and co-authors (2021) trained neural networks with full-batch gradient descent and found that the sharpness typically rises to about $2/\eta$ (unless training ends first) and then hovers right at, or just above, it, while the loss goes up and down from step to step but keeps falling over longer stretches (the scenes on the edge of stability).");
        ui::term(ui, r"Condition number $\kappa$:", r"$\lambda_{\max}/\lambda_{\min}$. The bowl's level lines are $\sqrt\kappa$ times longer than wide.");
        ui::term(ui, "Best fixed step:", r"$2/(\lambda_{\max} + \lambda_{\min})$: the step whose worst factor is smallest, $(\kappa - 1)/(\kappa + 1)$, the two extreme directions shrinking at that same rate.");

        ui::section(ui, "Numbers");
        let s = self.shown();
        let d = s.descent();
        let (lmax, lmin) = (s.sharpest(), s.flattest());
        ui::readout_colored(ui, r"step $\eta$", n2(s.eta), SGD);
        ui::readout_colored(ui, r"limit $2/\lambda_{\max}$", n2(2.0 / lmax), LIMIT);
        if lmin > 0.0 {
            ui::readout(ui, r"best fixed step $2/(\lambda_{\max} + \lambda_{\min})$", n2(2.0 / (lmax + lmin)));
            let kappa = lmax / lmin;
            ui::readout(ui, r"$\kappa = \lambda_{\max}/\lambda_{\min}$", if kappa < 100.0 { num(kappa, 1) } else { count(kappa.round().min(1e15) as usize) });
            ui::readout(ui, r"its worst factor $(\kappa - 1)/(\kappa + 1)$", num((kappa - 1.0) / (kappa + 1.0), 3));
        } else {
            ui::readout(ui, r"$\kappa = \lambda_{\max}/\lambda_{\min}$", r"none: $\lambda_{\min} = 0$");
        }
        if let Some((i, r)) = d.worst() {
            let which = if lmax - lmin <= 1e-9 * lmax {
                "every direction"
            } else if i == 0 {
                "the sharpest"
            } else if i + 1 == d.n() {
                "the flattest"
            } else {
                "a middle direction"
            };
            ui::readout(ui, r"worst factor now, $\max |1 - \eta\lambda|$", format!("{}, {which}", num(r, 3)));
        }
        match d.fate() {
            Fate::AtBottom => ui::readout(ui, "loss at the start", "0: the start is the bottom"),
            fate => {
                ui::readout(ui, "loss at the start", num(d.loss0(), 3));
                let to = d.steps_to(FALL).map_or_else(|| "never".to_string(), |t| count(t.min(1 << 50) as usize));
                ui::readout(ui, "steps until it is a thousand times lower", to);
                if let Some(b) = d.bound_steps(FALL) {
                    ui::readout(ui, "at most, from any start", count(b.min(1 << 50) as usize));
                }
                if fate == Fate::Grows {
                    if let Some(t) = d.lowest() {
                        ui::readout(ui, "steps for which the loss falls", count(t.min(1 << 50) as usize));
                    }
                }
            }
        }
        ui::see_also(ui, &[("quadform", "the bowl, and its own directions"), ("hessian", "the Hessian, whose eigenvalues are the curvatures"), ("momentum", "next: a step that keeps part of the last one"), ("eostoy", "a loss that is not a bowl: too sharp to stay")]);
    }

    fn frame(&mut self, f: &mut Frame) {
        hold_batches(f);
        let view = f.view_size();
        let lay = Layout::of(view, f.top_inset, self.rows as f32);
        let before = Layout::of(self.canvas, self.inset, self.fitted_rows);
        // (a canvas squeezed to a sliver is one the controls have opened over, on a
        // phone: the picture is not seen then, and is left as it was for when they close)
        let covered = view.x < 160.0 || view.y < 120.0;
        let changed = !covered && (lay.narrow != before.narrow || lay.board.is_some() != before.board.is_some() || (view - self.canvas).abs().max_element() > 24.0);
        if !covered && (changed || !self.measured) {
            self.new_view |= changed;
            (self.canvas, self.measured) = (view, true);
        }
        if !covered {
            self.new_view |= (f.top_inset - self.inset).abs() > 1.0;
            self.inset = f.top_inset;
            // on a phone the board below the picture grows and shrinks with its rows,
            // and the picture is fitted again (the camera glides there)
            let rows = self.rows as f32;
            if lay.narrow && rows != self.fitted_rows && ((rows - self.fitted_rows).abs() > 0.02 || rows == rows.round()) {
                self.new_view = true;
            }
            self.fitted_rows = rows;
        }
        let board = lay.board.as_ref().filter(|_| !covered);
        if let Some(b) = board {
            if plot::pointer(f).is_some_and(|p| inside_rect(p, (b.min, b.max))) {
                // the board is for its own handle: no moving the view from there
                f.handles.capture();
            }
        }

        // the hands: the start, and the step on its scale
        let mut start = self.shown().start.as_vec2();
        let under = board.is_some_and(|b| on_screen(f, start.extend(0.0)).is_some_and(|q| inside_rect(q, (b.min - Vec2::splat(4.0), b.max + Vec2::splat(4.0)))));
        if !covered && (!under || f.handles.dragging()) && f.drag2(1, &mut start, pal::WHITE) {
            self.set_start(start.as_dvec2());
        }
        self.knob_held = false;
        if let Some(b) = board {
            let knob = self.knob_plot(b);
            if let Some((x, _)) = knob.handle(f, 2, self.shown().eta, 0.0, SGD) {
                self.set_step(x);
                self.knob_held = true;
            }
        }
        self.advance(f.real_dt);

        let labels_before = f.dl.labels.len();
        let mut plates = self.picture(f, &lay, covered);
        let numbers_from = f.dl.labels.len();
        // no number of the axes over the heading, the caption or a name in the picture
        plates.extend(f.dl.labels[labels_before..].iter().map(|l| {
            let (lo, hi) = label_box(l);
            (lo - Vec2::splat(3.0), hi + Vec2::splat(3.0))
        }));
        f.axes2();
        let mut k = 0;
        f.dl.labels.retain(|l| {
            k += 1;
            k <= numbers_from || !plates.iter().any(|plate| overlap(label_box(l), *plate))
        });
        if covered {
            f.dl.labels.truncate(labels_before);
            return;
        }
        if let Some(b) = board {
            // nothing written into the picture may show through the board
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

impl StepSize {
    /// The step's scale on the board, from 0 to a quarter past the limit.
    fn knob_plot(&self, b: &Board) -> Plot {
        Plot::new(b.knob, [0.0, KNOB_END / self.shown().sharpest().max(1e-6)], [-1.0, 1.0])
    }

    /// The bowl from above, and the names written in it. Returns the boxes of the
    /// heading and the caption, which are written over it.
    fn picture(&mut self, f: &mut Frame, lay: &Layout, covered: bool) -> Vec<(Vec2, Vec2)> {
        let s = self.shown().clone();
        let n = s.lam.len();
        let (sharp, flat) = axes(s.turn);
        let (lmax, lmin) = (s.sharpest(), s.flattest());
        let round = lmax - lmin <= 1e-9 * lmax.max(1e-12);

        // the level lines: ellipses ½(λmax a² + λmin b²) = L, drawn as two halves, each
        // by its point along the flattest direction (a long ellipse drawn by angle would
        // have most of its points far outside the picture); with λmin = 0, two lines
        for (k, &level) in LEVELS.iter().enumerate() {
            let a = (2.0 * level / lmax.max(1e-12)).sqrt();
            let b = if lmin > 0.0 { (2.0 * level / lmin).sqrt() } else { f64::INFINITY };
            let reach = b.min(FAR);
            let color = pal::DIM.a(if k % 2 == 0 { 0.42 } else { 0.26 });
            for side in [1.0, -1.0] {
                let pts: Vec<Vec3> = (0..HALF_POINTS)
                    .map(|j| {
                        let y = reach * (std::f64::consts::PI * (j as f64 / (HALF_POINTS - 1) as f64 - 0.5)).sin();
                        let x = side * a * (1.0 - (y / b).powi(2)).max(0.0).sqrt();
                        v3(sharp * x + flat * y)
                    })
                    .collect();
                f.path(pts, 1.2, color);
            }
        }

        // the bowl's own directions through the bottom
        f.dashed(v3(-flat * FAR), v3(flat * FAR), 1.4, shade(n - 1, n).a(0.75));
        f.dashed(v3(-sharp * FAR), v3(sharp * FAR), 1.4, shade(0, n).a(0.85));
        f.circle(Vec3::ZERO, Vec3::Z, 0.045, 1.6, pal::WHITE.a(0.8));

        // the path, with a dot at each of its first steps: drawn up to its first point
        // outside the frame (the line to which fades: after it, a path that grows would
        // only cross and cross the picture). Always as many points and dots, those not
        // seen unseen, so that from one frame to the next they can be compared point for
        // point (`eidos motion`); and a point far outside the picture is put where the
        // line to it leaves a circle well outside it, which keeps it finite and makes it
        // move smoothly as the step does
        let mut path = Vec::with_capacity(PATH_POINTS);
        let batch = lines_for(f, Part::Path);
        batch.brk();
        for t in 0..PATH_POINTS {
            let p = s.point(t as u64);
            let far = p.length();
            let p = if far > FAR { p * (FAR / far) } else if p.is_finite() { p } else { DVec2::ZERO };
            let seen = path.len() == t && path.last().is_none_or(|q: &Vec3| q.length() as f64 <= SEEN);
            batch.push(v3(p), 2.0, if seen { SGD.a(0.85) } else { Color::CLEAR });
            if seen {
                path.push(v3(p));
            }
        }
        batch.brk();
        // a dot at each of the first steps (as many in every frame: those the path does
        // not reach, far off the picture)
        for t in 1..=DOTS {
            let (p, c) = match path.get(t) {
                Some(p) if p.length() < 6.0 => (*p, SGD.a(0.9)),
                _ => (FAR_AWAY, Color::CLEAR),
            };
            step_dots(f).push(p, 5.0, c);
        }

        // the heading: what the run comes to; under it, which directions are drawn
        let mut plates = Vec::new();
        let heading_at = lay.picture.0 + Vec2::new(16.0, 14.0);
        let heading = self.verdict();
        let size = if lay.narrow { 13.0 } else { 14.5 };
        let room = lay.picture.1.x - lay.picture.0.x - 32.0;
        let lines = wrap_line(f, &heading, size, room);
        // (a picture too small for the heading's lines goes without it: the panel says it)
        let lines = if lines.len() <= 3 && lay.picture.1.y - lay.picture.0.y >= 220.0 { lines } else { Vec::new() };
        let mut y = heading_at.y;
        for line in &lines {
            let w = f.text_size(line, size).x;
            f.caption_px(Vec2::new(heading_at.x, y), line.as_str(), size, pal::TEXT, Align2::LEFT_TOP);
            plates.push((Vec2::new(heading_at.x - 6.0, y - 5.0), Vec2::new(heading_at.x + w + 6.0, y + size + 6.0)));
            y += size + 7.0;
        }
        if n > 2 {
            let sub = format!("the plane of the sharpest and the flattest of its {n} directions");
            let w = f.text_size(&sub, 11.5).x;
            if w <= room {
                f.caption_px(Vec2::new(heading_at.x, y), sub.as_str(), 11.5, pal::DIM, Align2::LEFT_TOP);
                plates.push((Vec2::new(heading_at.x - 6.0, y - 5.0), Vec2::new(heading_at.x + w + 6.0, y + 17.0)));
            }
        }
        if !lay.narrow && lay.board.is_some() && f.view_size().y >= 300.0 {
            let line = "grey: level lines of the loss      dashed green: the bowl's own directions      cyan: the path, a dot per step";
            let (left, bottom) = (lay.picture.0.x + 18.0, lay.picture.1.y - 14.0);
            let w = f.text_size(line, 12.0).x;
            if w <= lay.picture.1.x - lay.picture.0.x - 36.0 {
                f.caption_px(Vec2::new(left, bottom), line, 12.0, pal::DIM, Align2::LEFT_BOTTOM);
                plates.push((Vec2::new(left - 8.0, bottom - 20.0), Vec2::new(left + w + 8.0, bottom + 6.0)));
            }
        }
        if covered {
            return plates;
        }

        // the names, clear of what is drawn and of one another
        let mut room = Room::new(f);
        if let Some(b) = &lay.board {
            room.wall(b.min - Vec2::splat(4.0), b.max + Vec2::splat(4.0));
        }
        for plate in &plates {
            room.wall(plate.0, plate.1);
        }
        for pair in path.windows(2).take(400) {
            room.line(f, pair[0], pair[1]);
        }
        let view = f.view_size();
        let seen = |f: &Frame, p: DVec2| on_screen(f, v3(p)).is_some_and(|q| inside_rect(q, (lay.picture.0 + Vec2::splat(30.0), lay.picture.1 - Vec2::splat(30.0))) && q.x < view.x);
        // each direction named near where its line leaves the picture, on the side that
        // is up on the screen
        let (lo, hi) = lay.frame_room();
        let name_at = |f: &Frame, dir: DVec2| -> Option<DVec2> {
            let dir = if dir.y < 0.0 || (dir.y == 0.0 && dir.x < 0.0) { -dir } else { dir };
            let mut best = None;
            for i in (1..=60).rev() {
                let p = dir * (FAR * i as f64 / 60.0);
                if on_screen(f, v3(p)).is_some_and(|q| inside_rect(q, (lo + Vec2::new(40.0, 10.0), hi - Vec2::new(40.0, 10.0)))) {
                    best = Some(p);
                    break;
                }
            }
            best
        };
        let screen_dir = |d: DVec2| Vec2::new(d.x as f32, -(d.y as f32));
        if round {
            if let Some(p) = name_at(f, flat) {
                room.label(f, "flat", v3(p), &format!(r"a round bowl: every direction is its own, $\lambda = {}$", n2(lmax)), 12.5, shade(0, n), screen_dir(sharp), 8.0, true);
            }
        } else {
            if let Some(p) = name_at(f, flat) {
                let text = if lmin > 0.0 { format!(r"flattest, $\lambda_{{\min}} = {}$", n2(lmin)) } else { r"flattest, $\lambda_{\min} = 0$: no slope".to_string() };
                room.label(f, "flat", v3(p), &text, 12.5, shade(n - 1, n), screen_dir(sharp), 8.0, true);
            }
            if let Some(p) = name_at(f, sharp) {
                room.label(f, "sharp", v3(p), &format!(r"sharpest, $\lambda_{{\max}} = {}$", n2(lmax)), 12.5, shade(0, n), screen_dir(flat), 8.0, true);
            }
        }
        if seen(f, s.start) {
            let away = Vec2::new(s.start.x as f32, -(s.start.y as f32));
            room.label(f, "start", v3(s.start), "start", 12.5, pal::WHITE, away, 12.0, true);
        }
        plates
    }

    /// The board: the step's scale, each direction's factor, the loss against the step.
    /// (Every batch of lines and dots on it has as many points in every frame, the ones
    /// not needed unseen, so that `eidos motion` can follow them from frame to frame.)
    fn board(&self, f: &mut Frame, b: &Board) {
        hud_panel(f, b.min, b.max);
        let s = self.shown();
        let d = s.descent();
        let n = d.n();
        let (lmax, lmin) = (s.sharpest(), s.flattest());
        let (text, small) = (b.text, b.small);

        // the step's scale: 0 to a quarter past the limit, with its three marks
        let knob = self.knob_plot(b);
        let (k0, k1) = knob.rect;
        let mid = 0.5 * (k0.y + k1.y);
        f.text_px(b.knob_title, format!(r"step $\eta = {}$", n2(s.eta)), text, pal::TEXT, Align2::LEFT_TOP);
        let limit = 2.0 / lmax;
        plot::fill(f, (Vec2::new(knob.px_x(limit), mid - 5.0), Vec2::new(k1.x, mid + 5.0)), LIMIT.a(0.015));
        plot::line_of(f, plot::AXES, Vec2::new(k0.x, mid), Vec2::new(k1.x, mid), 1.4, pal::DIM.a(0.8));
        plot::line_of(f, plot::AXES, Vec2::new(k0.x, mid - 4.0), Vec2::new(k0.x, mid + 4.0), 1.0, pal::DIM.a(0.8));
        f.text_px(Vec2::new(k0.x, mid + 8.0), "0", small, pal::DIM, Align2::CENTER_TOP);
        let mark = |f: &mut Frame, x: f64, color: Color| {
            let px = knob.px_x(x);
            plot::line_of(f, plot::AXES, Vec2::new(px, mid - 7.0), Vec2::new(px, mid + 7.0), 1.6, color);
            px
        };
        let one = mark(f, 1.0 / lmax, pal::DIM);
        f.text_px(Vec2::new(one, mid + 8.0), r"$1/\lambda_{\max}$", small, pal::DIM, Align2::CENTER_TOP);
        let lim = mark(f, limit, LIMIT);
        f.text_px(Vec2::new(lim, mid + 8.0), r"$2/\lambda_{\max}$", small, LIMIT, Align2::CENTER_TOP);
        // (with no flattest curvature there is no best step: its mark unseen)
        let best = mark(f, 2.0 / (lmax + lmin), if lmin > 0.0 { pal::TEXT } else { Color::CLEAR });
        if lmin > 0.0 {
            f.text_px(Vec2::new(best, mid - 8.0), "best", small, pal::TEXT, Align2::CENTER_BOTTOM);
        }
        // the step itself, the dot to drag (the handle's own dot is behind the board)
        let at = Vec2::new(knob.px_x(s.eta).max(k0.x).min(k1.x), mid);
        let hot = plot::pointer(f).is_some_and(|p| (p - at).length() < 15.0) || self.knob_held;
        plot::disc(f, at, if hot { 17.0 } else { 13.0 }, SGD.mix(pal::WHITE, if hot { 0.25 } else { 0.0 }));
        plot::disc(f, at, if hot { 7.0 } else { 5.0 }, pal::WHITE.a(0.9));

        // each direction's factor
        let (r0, r1) = b.rows;
        f.text_px(b.rows_title, r"each direction's factor per step, $1 - \eta\lambda$", text, pal::TEXT, Align2::LEFT_TOP);
        let rows = Plot::new(b.rows, [-1.5, 1.0], [0.0, 1.0]);
        let x_of = |v: f64| rows.px_x(v);
        plot::fill(f, (Vec2::new(x_of(-1.5), r0.y), Vec2::new(x_of(-1.0), r1.y)), LIMIT.a(0.012));
        plot::fill(f, (Vec2::new(x_of(-1.0), r0.y), Vec2::new(x_of(1.0), r1.y)), pal::FAINT.a(0.16));
        for (v, c, w) in [(-1.0, LIMIT, 1.4), (0.0, pal::DIM.a(0.7), 1.0), (1.0, pal::DIM.a(0.7), 1.0)] {
            plot::line_of(f, plot::AXES, Vec2::new(x_of(v), r0.y), Vec2::new(x_of(v), r1.y), w, c);
        }
        // the names of the parts, over them
        let names_y = r0.y - 4.0;
        let names: [(&str, &str, f64, f64, Color); 3] = [("grows", "grows", -1.5, -1.0, LIMIT), ("flips sides, shrinks", "flips", -1.0, 0.0, pal::DIM), ("creeps in", "creeps", 0.0, 1.0, pal::DIM)];
        for (long, short, a, b2, c) in names {
            let width = x_of(b2) - x_of(a);
            let name = if f.text_size(long, small).x <= width - 4.0 { long } else { short };
            f.text_px(Vec2::new(0.5 * (x_of(a) + x_of(b2)), names_y), name, small, c, Align2::CENTER_BOTTOM);
        }
        for v in [-1.5, -1.0, -0.5, 0.0, 0.5, 1.0] {
            plot::line_of(f, plot::AXES, Vec2::new(x_of(v), r1.y), Vec2::new(x_of(v), r1.y + 3.0), 1.0, pal::DIM.a(0.8));
            f.text_px(Vec2::new(x_of(v), r1.y + 4.0), format!("${}$", num(v, if v == v.round() { 0 } else { 1 })), small, pal::DIM, Align2::CENTER_TOP);
        }
        // the rows: as many as the most directions there can be, in the order of
        // `drawing_order` (the sharpest and the flattest first, which stay when the
        // bowl gains or loses directions), those the bowl has not parked off the screen;
        // the flattest's row glides down as the rows' block grows
        let pitch = (r1.y - r0.y) / (self.rows as f32).max(1.0);
        let spread = if n > 1 { (self.rows as f32 - 1.0) / (n - 1) as f32 } else { 0.0 };
        for i in drawing_order(n) {
            let here = i < n;
            let y = if here { r0.y + (0.5 + i as f32 * spread) * pitch } else { PARKED.y };
            let factor = if here { d.factor(i) } else { 1.0 };
            let c = if here { shade(i, n) } else { Color::CLEAR };
            let at = if here { x_of(factor.max(-1.5)) } else { PARKED.x };
            plot::line_of(f, ROWS, Vec2::new(if here { x_of(1.0) } else { PARKED.x }, y), Vec2::new(at, y), 1.6, c.a(if here { 0.45 } else { 0.0 }));
            plot::disc(f, Vec2::new(at, y), if pitch > 16.0 { 8.0 } else { 6.5 }, c);
            // off the scale: an arrow at its end
            let off = if here && factor < -1.5 { c } else { Color::CLEAR };
            plot::line_of(f, ROWS, Vec2::new(at - 9.0, y), Vec2::new(at - 3.0, y - 3.5), 1.4, off);
            plot::line_of(f, ROWS, Vec2::new(at - 9.0, y), Vec2::new(at - 3.0, y + 3.5), 1.4, off);
            if !here {
                continue;
            }
            let name = format!(r"$\lambda = {}$", n2(d.lam[i]));
            f.text_px(Vec2::new(r0.x - 8.0, y), name, small, c, Align2::RIGHT_CENTER);
            if pitch >= 10.0 {
                // (in a column of their own at the right)
                let value = format!("${}$", n2(factor));
                f.text_px(Vec2::new(b.max.x - 12.0, y), value, small, if factor < -1.0 { LIMIT } else { pal::TEXT.a(0.85) }, Align2::RIGHT_CENTER);
            }
        }

        self.loss_chart(f, b, &d);
    }

    /// The loss against the step, as a share of where it started, on a logarithmic axis:
    /// each direction's part is a straight line there, and the whole loss is their sum.
    fn loss_chart(&self, f: &mut Frame, b: &Board, d: &Descent) {
        let (text, small) = (b.text, b.small);
        f.text_px(b.chart_title, "the loss against the step (log scale)", text, pal::TEXT, Align2::LEFT_TOP);
        if b.chart.0.y - b.chart_title.y > 2.0 * (text + 4.0) {
            let key = "cyan: the whole loss      green: each direction's part";
            f.text_px(b.chart_title + Vec2::new(0.0, text + 4.0), key, small, pal::DIM, Align2::LEFT_TOP);
        }
        let horizon = self.horizon.max(1.0);
        let chart = Plot::new(b.chart, [0.0, horizon], [LOG_LO, LOG_HI]);
        if chart.tiny() {
            return;
        }
        let (min, max) = chart.rect;
        // the decades, and the start's level
        for k in [2.0, 0.0, -2.0, -4.0, -6.0] {
            let y = chart.px_y(k);
            plot::line_of(f, plot::AXES, Vec2::new(min.x, y), Vec2::new(max.x, y), 1.0, pal::DIM.a(if k == 0.0 { 0.35 } else { 0.14 }));
            let name = if k == 0.0 { "1".to_string() } else { format!("$10^{{{}}}$", num(k, 0)) };
            f.text_px(Vec2::new(min.x - 6.0, y), name, small, pal::DIM, Align2::RIGHT_CENTER);
        }
        plot::line_of(f, plot::AXES, Vec2::new(min.x, min.y), Vec2::new(min.x, max.y), 1.0, pal::DIM.a(0.8));
        // the steps along the bottom: a tick at every fifth of the span (a round number
        // when the span has come to rest; see `nice_span`)
        plot::line_of(f, plot::AXES, Vec2::new(min.x, max.y), max, 1.0, pal::DIM.a(0.8));
        for j in 0..=5 {
            let v = horizon * j as f64 / 5.0;
            let x = chart.px_x(v);
            plot::line_of(f, plot::AXES, Vec2::new(x, max.y), Vec2::new(x, max.y + 4.0), 1.0, pal::DIM.a(0.8));
            f.text_px(Vec2::new(x, max.y + 6.0), num(v, if horizon < 20.0 { 1 } else { 0 }), small, pal::DIM, Align2::CENTER_TOP);
        }
        f.text_px(Vec2::new(0.5 * (min.x + max.x), max.y + 8.0 + small), "step", small, pal::DIM, Align2::CENTER_TOP);

        let l0 = d.ln_loss(0.0);
        let log10 = std::f64::consts::LN_10;
        let corner = Vec2::new(min.x, max.y);
        // each direction's part: a straight line, its logarithm falling by −2 ln|1 − ηλ|
        // a step (one for each of the most directions there can be, unseen where there
        // is none)
        for i in drawing_order(d.n()) {
            let s = if i < d.n() { d.share(i) } else { 0.0 };
            let line = if s > 0.0 && l0.is_finite() {
                let y0 = (s.ln() - l0) / log10;
                let f_abs = d.factor(i).abs();
                // (a factor of 0: the part is gone after the first step)
                let (t1, y1) = if f_abs > 0.0 { (horizon, y0 + 2.0 * horizon * f_abs.ln() / log10) } else { (1.0, LOG_LO - 10.0) };
                let cut = plot::clip(&[chart.at(0.0, y0), chart.at(t1, y1)], chart.rect);
                (cut.len() == 2).then(|| (cut[0], cut[1]))
            } else {
                None
            };
            let (a, z, c) = match line {
                Some((a, z)) => (a, z, shade(i, d.n()).a(0.6)),
                None => (PARKED, PARKED, Color::CLEAR),
            };
            plot::line_of(f, SHARES, a, z, 1.3, c);
        }
        // the whole loss, at as many points in every frame: those beyond the chart's
        // edges put on the edge and unseen
        let total: Vec<(Vec2, bool)> = (0..SAMPLES)
            .map(|j| {
                let t = horizon * j as f64 / (SAMPLES - 1) as f64;
                let y = if l0.is_finite() { (d.ln_loss(t) - l0) / log10 } else { f64::NAN };
                let p = chart.at(t, if y.is_finite() { y } else { LOG_LO - 10.0 });
                let p = if p.is_finite() { p } else { corner };
                (p.max(min).min(max), y.is_finite() && (LOG_LO..=LOG_HI).contains(&y))
            })
            .collect();
        let points: Vec<(Vec3, bool)> = total.iter().map(|(p, seen)| (f.hud(*p), *seen)).collect();
        let batch = plot::apart(f, TOTAL, false);
        batch.brk();
        for (p, seen) in points {
            batch.push(p, 2.2, if seen { SGD } else { Color::CLEAR });
        }
        batch.brk();
        if !l0.is_finite() {
            f.text_px((min + max) * 0.5, "no loss: the start is the bottom", small, pal::DIM, Align2::CENTER_CENTER);
        }

        // a thousand times lower: when
        let thousandth = -FALL.log10();
        let level = chart.px_y(thousandth);
        chart.hline(f, plot::MARKS, thousandth, 1.0, SGD.a(0.5), true);
        f.text_px(Vec2::new(max.x - 4.0, level - 3.0), r"$1000 \times$ lower", small, SGD.a(0.8), Align2::RIGHT_BOTTOM);
        // (the mark where the curve crosses the line, between two steps: it moves smoothly
        // as the step size does; the number is the step after which it is below)
        let reached = d.steps_to(FALL).zip(d.crossing(FALL)).filter(|(_, x)| *x <= horizon);
        let x = reached.map_or(PARKED.x, |(_, t)| chart.px_x(t));
        let y = if reached.is_some() { level } else { PARKED.y };
        plot::line_of(f, COUNTS, Vec2::new(x, y), Vec2::new(x, if reached.is_some() { max.y } else { PARKED.y }), 1.4, SGD.a(if reached.is_some() { 0.8 } else { 0.0 }));
        plot::disc(f, Vec2::new(x, y), 6.0, if reached.is_some() { SGD } else { Color::CLEAR });
        if let Some((t, _)) = reached {
            let right = x < min.x + 0.7 * (max.x - min.x);
            let (dx, align) = if right { (9.0, Align2::LEFT_TOP) } else { (-9.0, Align2::RIGHT_TOP) };
            f.caption_px(Vec2::new(x + dx, level + 5.0), steps_text(t), small, SGD, align);
        }
        // the lowest point of a run that climbs in the end
        let lowest = d
            .lowest()
            .zip(d.lowest_between())
            .filter(|(t, x)| *t > 0 && *x <= horizon)
            .map(|(t, x)| (t, x, (d.ln_loss(x) - l0) / log10))
            .filter(|(_, _, y)| *y > LOG_LO);
        let p = lowest.map_or(PARKED, |(_, x, y)| chart.at(x, y));
        plot::disc(f, p, 7.0, if lowest.is_some() { pal::WHITE } else { Color::CLEAR });
        if let Some((t, _, _)) = lowest {
            f.caption_px(p + Vec2::new(0.0, 9.0), format!("lowest, after {}", steps_text(t)), small, pal::TEXT, Align2::CENTER_TOP);
        }
    }
}

/// The points the whole loss is drawn through.
const SAMPLES: usize = 241;

/// Where the board puts a row, a line or a mark it has no use for in this frame: off
/// the screen, so that every batch keeps its number of points from frame to frame.
const PARKED: Vec2 = Vec2::new(-4000.0, -4000.0);
/// The same in the plane of the bowl.
const FAR_AWAY: Vec3 = Vec3::new(1e4, 1e4, 0.0);

/// The order the board draws its rows, lines and marks for a bowl of `n` directions in:
/// the sharpest, the flattest, the ones between, then the places of the most there can
/// be. So the sharpest and the flattest keep their places in the batch when the bowl
/// gains or loses directions, and can be followed from one frame to the next.
fn drawing_order(n: usize) -> impl Iterator<Item = usize> {
    let n = n.clamp(1, MOST);
    let flattest = (n > 1).then_some(n - 1);
    std::iter::once(0).chain(flattest).chain(1..n.saturating_sub(1)).chain(n..MOST)
}

/// A heading broken into lines no wider than `room` pixels: after its colon, and then
/// wherever its words need.
fn wrap_line(f: &Frame, text: &str, size: f32, room: f32) -> Vec<String> {
    if f.text_size(text, size).x <= room {
        return vec![text.to_string()];
    }
    // a line break after the colon if there is one, and then wherever the words need it
    let (head, rest) = match text.split_once(": ") {
        Some((a, b)) => (Some(format!("{a}:")), b),
        None => (None, text),
    };
    let mut lines: Vec<String> = head.into_iter().collect();
    let mut line = String::new();
    for word in rest.split(' ') {
        let longer = if line.is_empty() { word.to_string() } else { format!("{line} {word}") };
        if f.text_size(&longer, size).x > room && !line.is_empty() {
            lines.push(std::mem::replace(&mut line, word.to_string()));
        } else {
            line = longer;
        }
    }
    if !line.is_empty() {
        lines.push(line);
    }
    lines
}

/// The box a label takes on the screen, nearly.
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

fn overlap(a: (Vec2, Vec2), b: (Vec2, Vec2)) -> bool {
    a.0.x < b.1.x && b.0.x < a.1.x && a.0.y < b.1.y && b.0.y < a.1.y
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The scene as an experiment's button leaves it, once everything has glided there
    /// (from the scene as it opens, or from where `from` left it).
    fn run_from(mut scene: StepSize, k: usize) -> StepSize {
        scene.set_up(k);
        for _ in 0..60 * 20 {
            scene.advance(1.0 / 60.0);
            if scene.at_rest() {
                break;
            }
        }
        assert!(scene.at_rest(), "experiment {k} has not come to rest");
        scene
    }

    fn run(k: usize) -> StepSize {
        run_from(StepSize::new(), k)
    }

    /// A descent of `n` directions, spread on a log scale, from a start of these
    /// distances (the ones not given are 1).
    fn descent(lmax: f64, lmin: f64, n: usize, eta: f64, start: &[f64]) -> Descent {
        let lam = curvatures(lmax, lmin, n, Spread::Log);
        let mut s = vec![1.0; n];
        for (i, v) in start.iter().enumerate().take(n) {
            s[i] = *v;
        }
        Descent { lam, start: s, eta }
    }

    /// The loss after each of `steps` steps, by stepping: no closed forms.
    fn losses(d: &Descent, steps: usize) -> Vec<f64> {
        let mut x = d.start.clone();
        let mut out = Vec::with_capacity(steps + 1);
        for t in 0..=steps {
            if t > 0 {
                for (xi, l) in x.iter_mut().zip(&d.lam) {
                    *xi -= d.eta * l * *xi;
                }
            }
            out.push(x.iter().zip(&d.lam).map(|(xi, l)| 0.5 * l * xi * xi).sum());
        }
        out
    }

    /// Random bowls and starts, and the awkward ones.
    fn descents() -> Vec<Descent> {
        let mut rng = Rng::new(11);
        let mut out = Vec::new();
        for k in 0..300 {
            let n = 2 + k % 11;
            let lmax = 0.1 + 3.9 * rng.range(0.0, 1.0) as f64;
            let lmin = lmax * [0.0, 1e-3, 0.01, 0.1, 0.5, 1.0][k % 6];
            let spread = SPREADS[k % 3].0;
            let lam = curvatures(lmax, lmin, n, spread);
            let start: Vec<f64> = (0..n).map(|_| 2.0 * rng.normal() as f64).collect();
            let eta = rng.range(0.0, 1.0) as f64 * KNOB_END / lmax;
            out.push(Descent { lam, start, eta });
        }
        out
    }

    /// A step multiplies the distance along each of the bowl's own directions by
    /// 1 − ηλ: gradient descent run in the coordinates of the plane, on the matrix of
    /// the bowl turned, comes where the closed form says; and in many directions, on a
    /// bowl whose own directions are random, likewise.
    #[test]
    fn each_direction_is_multiplied_by_its_factor() {
        // the plane: H = λmax s sᵀ + λmin u uᵀ, the sharpest s and the flattest u
        for (turn, lmax, lmin, eta) in [(0.4, 1.0, 0.1, 1.6), (2.0, 3.0, 0.0, 0.5), (1.0, 0.5, 0.5, 2.5), (0.0, 1.0, 0.01, 2.1)] {
            let s = Setting { lam: vec![lmax, lmin], turn, start: DVec2::new(1.2, -0.7), eta };
            let (sharp, flat) = axes(turn);
            let hess = |w: DVec2| sharp * (lmax * sharp.dot(w)) + flat * (lmin * flat.dot(w));
            let mut w = s.start;
            for t in 0..60 {
                assert!((s.point(t) - w).length() < 1e-9 * w.length().max(1.0), "turn {turn}, step {t}: {:?} against {w:?}", s.point(t));
                w -= hess(w) * eta;
            }
        }
        // n directions along a random orthonormal basis
        let mut rng = Rng::new(3);
        for n in [3, 7, 12] {
            let mut basis: Vec<Vec<f64>> = Vec::new();
            while basis.len() < n {
                let mut v: Vec<f64> = (0..n).map(|_| rng.normal() as f64).collect();
                for b in &basis {
                    let along: f64 = v.iter().zip(b).map(|(a, c)| a * c).sum();
                    v.iter_mut().zip(b).for_each(|(a, c)| *a -= along * c);
                }
                let size = v.iter().map(|a| a * a).sum::<f64>().sqrt();
                if size > 1e-6 {
                    basis.push(v.iter().map(|a| a / size).collect());
                }
            }
            let d = descent(2.0, 0.05, n, 0.9, &(0..n).map(|i| 0.3 * i as f64 - 1.0).collect::<Vec<_>>());
            let mut w: Vec<f64> = (0..n).map(|j| (0..n).map(|i| d.start[i] * basis[i][j]).sum()).collect();
            for t in 0..40 {
                for i in 0..n {
                    let along: f64 = w.iter().zip(&basis[i]).map(|(a, b)| a * b).sum();
                    let want = d.start[i] * d.factor(i).powi(t);
                    assert!((along - want).abs() < 1e-9, "n {n}, step {t}, direction {i}: {along} against {want}");
                }
                // the gradient H w, with H = Σ λᵢ bᵢ bᵢᵀ
                let grad: Vec<f64> = (0..n).map(|j| (0..n).map(|i| d.lam[i] * basis[i][j] * w.iter().zip(&basis[i]).map(|(a, b)| a * b).sum::<f64>()).sum()).collect();
                w.iter_mut().zip(&grad).for_each(|(a, g)| *a -= d.eta * g);
            }
        }
    }

    /// The curvatures of every spread: as many as asked, the sharpest first, the two
    /// ends exactly λmax and λmin, none negative.
    #[test]
    fn the_curvatures_keep_their_ends_and_their_order() {
        for spread in SPREADS.map(|s| s.0) {
            for n in FEWEST..=MOST {
                for (lmax, lmin) in [(1.0, 0.1), (4.0, 0.0), (0.1, 0.1), (2.0, 1.9), (1.0, 1e-3), (3.0, 2.0)] {
                    let lam = curvatures(lmax, lmin, n, spread);
                    assert_eq!(lam.len(), n);
                    assert!(lam[0] == lmax && lam[n - 1] == lmin, "{spread:?} {n}: {lam:?}");
                    assert!(lam.windows(2).all(|w| w[0] >= w[1]) && lam.iter().all(|l| l.is_finite() && *l >= 0.0), "{spread:?} {n}: {lam:?}");
                }
            }
        }
        // a few sharp, the rest flat: the bulk in the lowest twentieth of the range
        let lam = curvatures(1.0, 0.01, 12, Spread::FewSharp);
        assert_eq!(lam.iter().filter(|l| **l >= 0.5).count(), 3, "{lam:?}");
        assert!(lam[3..].iter().all(|l| *l <= 0.01 + 0.05 * 0.99 + 1e-12), "{lam:?}");
        // and what the controls cannot give is mended
        assert_eq!(curvatures(f64::NAN, 2.0, 1, Spread::Log), vec![1.0, 1.0]);
        assert_eq!(curvatures(1.0, -3.0, 40, Spread::Even).len(), MOST);
    }

    /// Converges exactly when every factor is between −1 and 1, that is when the step
    /// is shorter than 2/λmax (with λmin above 0): beyond it the sharpest direction
    /// grows, the others still shrink; at it, it flips with the same size for ever.
    #[test]
    fn the_limit_is_two_over_the_sharpest_curvature() {
        for (lmax, lmin) in [(1.0, 0.1), (2.5, 0.3), (0.4, 0.4)] {
            for k in 1..250 {
                let eta = k as f64 * 0.01 * KNOB_END / lmax;
                let d = descent(lmax, lmin, 4, eta, &[0.7, 1.0, 1.0, -1.3]);
                let limit = 2.0 / lmax;
                if eta < limit * (1.0 - 1e-9) {
                    assert_eq!(d.fate(), Fate::Converges, "{lmax} {lmin} at {eta}");
                    assert!(d.steps_to(FALL).is_some());
                } else if eta > limit * (1.0 + 1e-9) {
                    assert_eq!(d.fate(), Fate::Grows, "{lmax} {lmin} at {eta}");
                    // (the sharpest grows; every direction with ηλ < 2 still shrinks)
                    assert!(d.factor(0) < -1.0 && (1..4).filter(|&i| eta * d.lam[i] < 2.0 - 1e-9).all(|i| d.factor(i).abs() < 1.0));
                    let l = losses(&d, 400);
                    assert!(l[400] > l[0], "{lmax} {lmin} at {eta}: the loss has not grown");
                }
            }
            // at the limit: the sharpest flips at its size for ever
            let d = descent(lmax, lmin, 2, 2.0 / lmax, &[0.5, 1.0]);
            assert!((d.factor(0) + 1.0).abs() < 1e-12);
            if lmin < lmax {
                assert_eq!(d.fate(), Fate::Stays);
                let l = losses(&d, 2000);
                assert!((l[2000] - d.share(0)).abs() < 1e-9 * l[0], "{lmax}: {} against {}", l[2000], d.share(0));
            }
        }
    }

    /// The best fixed step: no step has a smaller worst factor, and its worst factor is
    /// (κ − 1)/(κ + 1), the two extreme directions shrinking at that rate, one on each
    /// side of 0. A longer step slows the sharpest, a shorter one the flattest.
    #[test]
    fn the_best_fixed_step_is_best() {
        for (lmax, lmin) in [(1.0, 0.1), (1.0, 0.01), (3.0, 1.0), (0.7, 0.6)] {
            let best = 2.0 / (lmax + lmin);
            let kappa = lmax / lmin;
            let at = |eta: f64| descent(lmax, lmin, 5, eta, &[]);
            let worst = |eta: f64| at(eta).worst().map_or(f64::NAN, |w| w.1);
            let rate = (kappa - 1.0) / (kappa + 1.0);
            assert!((worst(best) - rate).abs() < 1e-12, "{lmax} {lmin}");
            assert!((at(best).factor(0) + rate).abs() < 1e-12 && (at(best).factor(4) - rate).abs() < 1e-12);
            for k in 1..2000 {
                let eta = k as f64 * 1e-3 * 2.0 / lmax;
                assert!(worst(eta) >= rate - 1e-12, "{lmax} {lmin}: {eta} beats the best step");
            }
            for (a, b) in [(best, best * 1.05), (best * 1.05, best * 1.1)] {
                assert!(at(b).factor(0).abs() > at(a).factor(0).abs() && at(b).factor(4).abs() < at(a).factor(4).abs());
            }
            for (a, b) in [(best * 0.9, best * 0.95), (best * 0.95, best)] {
                assert!(at(a).factor(4).abs() > at(b).factor(4).abs());
            }
        }
    }

    /// The counts: the closed forms against stepping. The first step after which the
    /// loss is a thousand times lower; where the curve crosses that line; the lowest
    /// step of a run that grows; the bound, which no start exceeds.
    #[test]
    fn the_counts_agree_with_stepping() {
        for d in descents() {
            let l = losses(&d, 3000);
            let l0 = l[0];
            if l0 == 0.0 {
                assert_eq!(d.fate(), Fate::AtBottom);
                continue;
            }
            let first = l.iter().position(|v| *v <= l0 / FALL);
            match d.fate() {
                Fate::Converges | Fate::Stays => {
                    if let Some(t) = first {
                        assert_eq!(d.steps_to(FALL), Some(t as u64), "{d:?}");
                        let x = d.crossing(FALL).unwrap_or(f64::NAN);
                        assert!(x > t as f64 - 1.0 - 1e-9 && x <= t as f64 + 1e-9 && (d.ln_loss(x) - (l0 / FALL).ln()).abs() < 1e-6, "{d:?}: {x}");
                    } else if d.fate() == Fate::Converges {
                        // (beyond what was stepped)
                        assert!(d.steps_to(FALL).is_some_and(|t| t > 3000), "{d:?}");
                    }
                }
                Fate::Grows => {
                    let low = d.lowest().expect("a lowest step") as usize;
                    if low < 2999 {
                        // (after it, as far as the loss stays a number)
                        let climbs = l[low..].iter().take(300).take_while(|v| v.is_finite() && **v < 1e300).copied().collect::<Vec<f64>>();
                        assert!(l[..=low].windows(2).all(|w| w[1] < w[0]) && climbs.windows(2).all(|w| w[1] > w[0]), "{d:?}: lowest {low}");
                        assert_eq!(d.steps_to(FALL), first.filter(|t| *t <= low).map(|t| t as u64), "{d:?}");
                    }
                }
                Fate::AtBottom => unreachable!(),
            }
            if let (Some(t), Some(b)) = (d.steps_to(FALL), d.bound_steps(FALL)) {
                assert!(t <= b, "{d:?}: {t} steps, more than the bound {b}");
            }
            // the closed form of the loss, at the steps
            for t in [0usize, 1, 7, 50, 400] {
                let want = l[t];
                let got = d.ln_loss(t as f64).exp();
                assert!((got - want).abs() <= 1e-9 * want.max(1e-300) || (want < 1e-280 && got < 1e-270), "{d:?} at {t}: {got} against {want}");
            }
        }
        // with two directions at the best step, the bound is the count, whatever the start
        for start in [[0.8, 1.6], [0.01, 2.0], [2.0, 0.01], [-1.0, 0.3]] {
            for (lmax, lmin) in [(1.0, 0.1), (1.0, 0.01), (2.0, 0.5)] {
                let d = descent(lmax, lmin, 2, 2.0 / (lmax + lmin), &start);
                assert_eq!(d.steps_to(FALL), d.bound_steps(FALL), "{d:?}");
            }
        }
    }

    /// The captions of the "things to try", held to account.
    #[test]
    fn a_short_step_does_what_its_caption_says() {
        let scene = run(0);
        let s = scene.shown();
        let d = s.descent();
        assert_eq!((d.n(), n2(s.eta)), (4, "0.20".to_string()));
        assert!((s.eta - 0.1 * 2.0 / s.sharpest()).abs() < 1e-12);
        let factors: Vec<String> = (0..4).map(|i| n2(d.factor(i))).collect();
        assert!(factors[0] == "0.80" && factors[3] == "0.98" && (0..4).all(|i| d.factor(i) > 0.0 && d.factor(i) < 1.0), "{factors:?}");
        assert_eq!(d.steps_to(FALL), Some(127));
        let q = d.factor(3).powi(2);
        assert_eq!(num(q, 4), "0.9604");
        assert_eq!(num(FALL.ln() / -q.ln(), 0), "171");
        assert_eq!(num(100.0 * d.share(3) / d.loss0(), 0), "16");
        // the sharper ones are gone early: at step 127 the flattest holds nearly all
        let left = |i: usize| (d.ln_share(i, 127.0) - d.ln_loss(127.0)).exp();
        assert!(left(3) > 0.99 && left(0) < 1e-10, "{}", left(3));
        // no step up to 1/λmax has a factor below 0, and every one beyond it has
        for k in 0..=100 {
            let eta = k as f64 * 0.01 / s.sharpest();
            assert!(Descent { eta, ..d.clone() }.factor(0) >= -1e-12);
            assert!(Descent { eta: eta + 1.0001 / s.sharpest(), ..d.clone() }.factor(0) < 0.0);
        }
    }

    #[test]
    fn the_best_fixed_step_does_what_its_caption_says() {
        let scene = run(1);
        let s = scene.shown();
        let d = s.descent();
        assert_eq!((d.n(), n2(s.eta)), (2, "1.82".to_string()));
        assert_eq!([n2(d.factor(0)), n2(d.factor(1))], ["−0.82", "0.82"]);
        let kappa = s.sharpest() / s.flattest();
        assert!((kappa - 10.0).abs() < 1e-12 && (d.factor(1) - (kappa - 1.0) / (kappa + 1.0)).abs() < 1e-12);
        // the two parts fall at the same rate: parallel lines on the chart
        let slope = |i: usize| d.ln_share(i, 10.0) - d.ln_share(i, 0.0);
        assert!((slope(0) - slope(1)).abs() < 1e-12);
        assert_eq!((d.bound_steps(FALL), d.steps_to(FALL)), (Some(18), Some(18)));
        // from this start, the opening step does it in one fewer
        assert_eq!(Descent { eta: OPENING_STEP, ..d.clone() }.steps_to(FALL), Some(17));
        assert_eq!(StepSize::new().shown().descent().steps_to(FALL), Some(17));
    }

    #[test]
    fn just_under_the_limit_does_what_its_caption_says() {
        let scene = run(2);
        let s = scene.shown();
        let d = s.descent();
        assert_eq!((n2(s.eta), num(100.0 * s.eta * s.sharpest() / 2.0, 0)), ("1.96".to_string(), "98".to_string()));
        assert_eq!([n2(d.factor(0)), n2(d.factor(1))], ["−0.96", "0.80"]);
        assert_eq!(num(100.0 * (1.0 - d.factor(0).abs()), 0), "4");
        // the flattest's own loss, a thousand times smaller
        let own = (FALL.ln() / (-2.0 * d.factor(1).abs().ln())).ceil();
        assert_eq!(own, 16.0);
        assert_eq!(d.steps_to(FALL), Some(81));
        assert_eq!(d.worst().map(|w| w.0), Some(0));
    }

    #[test]
    fn just_over_the_limit_does_what_its_caption_says() {
        let scene = run(3);
        let s = scene.shown();
        let d = s.descent();
        assert_eq!((d.n(), n2(s.eta), num(100.0 * (s.eta * s.sharpest() / 2.0 - 1.0), 0)), (4, "2.10".to_string(), "5".to_string()));
        assert_eq!(n2(d.factor(0)), "−1.10");
        assert!((1..4).all(|i| d.factor(i).abs() < 1.0));
        // almost on the floor: a fiftieth from it
        assert!((s.along().0 - 0.02).abs() < 1e-12);
        assert_eq!(d.fate(), Fate::Grows);
        assert_eq!(d.lowest(), Some(11));
        let l = losses(&d, 300);
        assert!(l[..=11].windows(2).all(|w| w[1] < w[0]) && l[11..].windows(2).all(|w| w[1] > w[0]));
        assert_eq!(num(100.0 * l[11] / l[0], 1), "0.5");
        // by then the sharpest direction holds most of what is left
        assert!(d.share(0) * d.factor(0).powi(22) > 0.5 * l[11]);
    }

    #[test]
    fn a_long_bowl_does_what_its_caption_says() {
        let scene = run(4);
        let s = scene.shown();
        let d = s.descent();
        let kappa = s.sharpest() / s.flattest();
        assert!((kappa - 100.0).abs() < 1e-9);
        // the level lines: ½(λmax a² + λmin b²) = L has b/a = √κ
        assert_eq!(num(kappa.sqrt(), 0), "10");
        assert_eq!(n2(s.eta), "1.98");
        let rate = (kappa - 1.0) / (kappa + 1.0);
        assert_eq!(num(rate, 3), "0.980");
        assert!((d.factor(1) - rate).abs() < 1e-12 && (d.factor(0) + rate).abs() < 1e-12);
        assert!((rate - (1.0 - 2.0 / kappa)).abs() < 2e-4);
        assert_eq!(d.steps_to(FALL), Some(173));
        assert_eq!(run(1).shown().descent().steps_to(FALL), Some(18));
        assert_eq!(num(-1.0 / rate.ln(), 0), "50");
        assert_eq!(num(kappa / 2.0, 0), "50");
    }

    /// The panel's words: on a loss that is not a bowl, if the curvature is at most ℓ
    /// everywhere, every step shorter than 2/ℓ lowers the loss, by at least
    /// η(2 − ηℓ)/2 |∇L|². The loss here is ln cosh 2x + 0.1 y² + 0.25 cos x cos y, whose
    /// Hessian is [[4 sech² 2x − c, s], [s, 0.2 − c]] with c = 0.25 cos x cos y and
    /// s = 0.25 sin x sin y: by Gershgorin its largest eigenvalue is at most 4.5.
    #[test]
    fn the_descent_lemma_holds_on_a_loss_that_is_not_a_bowl() {
        let loss = |x: f64, y: f64| (2.0 * x).cosh().ln() + 0.1 * y * y + 0.25 * x.cos() * y.cos();
        let grad = |x: f64, y: f64| DVec2::new(2.0 * (2.0 * x).tanh() - 0.25 * x.sin() * y.cos(), 0.2 * y - 0.25 * x.cos() * y.sin());
        let ell = 4.5;
        let mut rng = Rng::new(5);
        let mut fell = 0;
        for _ in 0..2000 {
            let (x, y) = (rng.range(-3.0, 3.0) as f64, rng.range(-3.0, 3.0) as f64);
            // (the Hessian's largest eigenvalue here, by finite differences, is within ℓ)
            let h = 1e-4;
            let (gxx, gxy, gyy) = ((grad(x + h, y).x - grad(x - h, y).x) / (2.0 * h), (grad(x, y + h).x - grad(x, y - h).x) / (2.0 * h), (grad(x, y + h).y - grad(x, y - h).y) / (2.0 * h));
            let top = 0.5 * (gxx + gyy) + (0.25 * (gxx - gyy).powi(2) + gxy * gxy).sqrt();
            assert!(top <= ell, "{top} at ({x}, {y})");
            let g = grad(x, y);
            for k in 1..20 {
                let eta = k as f64 / 20.0 * 2.0 / ell;
                let after = loss(x - eta * g.x, y - eta * g.y);
                let promised = loss(x, y) - eta * (2.0 - eta * ell) / 2.0 * g.length_squared();
                assert!(after <= promised + 1e-12, "({x}, {y}) with {eta}: {after} above {promised}");
                fell += (after < loss(x, y)) as usize;
            }
        }
        assert!(fell > 37_000);
    }

    /// The words about the own directions and a curvature of 0: at a point on an own
    /// direction the gradient points straight away from the bottom; with λmin = 0 the
    /// point never moves along the flattest direction, and the loss there is that of
    /// its distance across alone.
    #[test]
    fn the_words_about_directions_are_true() {
        for turn in [0.0, 0.6, 2.1] {
            let (sharp, flat) = axes(turn);
            for (lmax, lmin) in [(1.0, 0.1), (2.0, 0.0)] {
                let grad = |w: DVec2| sharp * (lmax * sharp.dot(w)) + flat * (lmin * flat.dot(w));
                for along in [sharp * 1.3, flat * -0.7] {
                    let g = grad(along);
                    assert!(g.perp_dot(along).abs() < 1e-12 && g.dot(along) >= 0.0);
                }
            }
            let s = Setting { lam: vec![2.0, 0.0], turn, start: DVec2::new(0.9, -1.1), eta: 0.7 };
            let (_, b) = s.along();
            for t in [1, 5, 40] {
                assert!((s.point(t).dot(flat) - b).abs() < 1e-12);
            }
            let d = s.descent();
            assert!(d.share(1) == 0.0 && d.fate() == Fate::Converges);
        }
    }

    /// Whatever the hands and the controls do, and at the ends of every slider, every
    /// number drawn or written is finite, nothing panics, and the scene comes to rest.
    #[test]
    fn degenerate_input_is_normal_input() {
        type Case = Box<dyn Fn(&mut StepSize)>;
        let mut cases: Vec<Case> = vec![
            Box::new(|s| s.set_start(DVec2::ZERO)),
            Box::new(|s| s.set_start(DVec2::new(f64::NAN, 1.0))),
            Box::new(|s| s.set_start(DVec2::new(1e9, -1e9))),
            Box::new(|s| s.set_step(0.0)),
            Box::new(|s| s.set_step(f64::INFINITY)),
            Box::new(|s| s.set_step(-3.0)),
            Box::new(|s| s.set_step(2.0)),
            Box::new(|s| s.set_turn(f64::NAN)),
            Box::new(|s| {
                s.lmin = 0.0;
                s.bowl_changed();
            }),
            Box::new(|s| {
                (s.lmax, s.lmin) = (0.1, 0.1);
                s.bowl_changed();
                s.set_step(10.0);
            }),
            Box::new(|s| {
                (s.lmax, s.lmin, s.n, s.spread) = (4.0, 0.0, 12, Spread::FewSharp);
                s.bowl_changed();
                s.set_step(0.5);
            }),
            Box::new(|s| {
                (s.lmax, s.lmin, s.n) = (f64::NAN, f64::INFINITY, 99);
                s.bowl_changed();
            }),
            Box::new(|s| {
                (s.lmax, s.lmin, s.n, s.spread) = (4.0, 0.001, 12, Spread::Log);
                s.bowl_changed();
                s.set_step(0.0005);
            }),
        ];
        for k in 0..TRIALS.len() {
            cases.push(Box::new(move |s| s.set_up(k)));
        }
        for (c, case) in cases.iter().enumerate() {
            let mut scene = StepSize::new();
            case(&mut scene);
            for _ in 0..60 * 20 {
                scene.advance(1.0 / 60.0);
                if scene.at_rest() {
                    break;
                }
            }
            assert!(scene.at_rest(), "case {c}");
            let s = scene.shown();
            let d = s.descent();
            assert!(s.start.is_finite() && s.eta.is_finite() && s.eta >= 0.0 && s.turn.is_finite() && s.lam.iter().all(|l| l.is_finite() && *l >= 0.0), "case {c}: {s:?}");
            assert!(scene.horizon.is_finite() && scene.horizon >= SHORTEST && scene.horizon <= LONGEST, "case {c}");
            for t in [0, 1, 2, 10, 1000, 3000] {
                assert!(s.point(t).is_finite(), "case {c}");
                let l = d.ln_loss(t as f64);
                assert!(!l.is_nan(), "case {c} at {t}");
            }
            let words = [scene.verdict(), steps_text(d.steps_to(FALL).unwrap_or(0)), steps_text(d.lowest().unwrap_or(0))].join(" ");
            assert!(!words.contains("NaN") && !words.contains("inf") && !words.contains('?'), "case {c}: {words}");
            let _ = (d.crossing(FALL), d.lowest_between(), d.bound(FALL), horizon(&d));
        }
    }

    /// The board and the picture fit every canvas, from a phone to a desktop, with any
    /// number of rows; on a desktop the picture's place does not depend on the rows.
    #[test]
    fn the_layout_fits_every_canvas() {
        for (w, h) in [(1080.0, 1000.0), (580.0, 700.0), (390.0, 800.0), (390.0, 560.0), (800.0, 600.0), (1600.0, 1000.0), (240.0, 180.0), (620.0, 480.0), (1300.0, 700.0), (700.0, 1400.0)] {
            for inset in [0.0, 46.0] {
                let mut desk = None;
                for rows in [2.0, 3.5, 7.0, 12.0] {
                    let view = Vec2::new(w, h);
                    let lay = Layout::of(view, inset, rows);
                    let (min, max) = lay.picture;
                    assert!(min.cmpge(Vec2::ZERO).all() && max.cmple(view).all() && (max - min).min_element() > 50.0, "{w}x{h} {rows}: {lay:?}");
                    if let Some(b) = &lay.board {
                        let inside = |q: Vec2| q.cmpge(b.min - Vec2::splat(0.5)).all() && q.cmple(b.max + Vec2::splat(0.5)).all();
                        assert!(b.min.cmpge(Vec2::ZERO).all() && b.max.cmple(view + Vec2::splat(0.5)).all(), "{w}x{h} {rows}: {b:?}");
                        for r in [b.knob, b.rows, b.chart] {
                            assert!(inside(r.0) && inside(r.1) && r.1.x > r.0.x && r.1.y > r.0.y, "{w}x{h} {rows}: {b:?}");
                        }
                        assert!(inside(b.chart.1 + Vec2::new(0.0, 28.0)) && b.knob.1.y < b.rows_title.y && b.rows.1.y < b.chart_title.y && b.chart_title.y < b.chart.0.y, "{w}x{h} {rows}: {b:?}");
                        assert!(b.min.y >= inset || lay.narrow, "{w}x{h}: under the buttons");
                        assert!(!Plot::new(b.knob, [0.0, 1.0], [0.0, 1.0]).tiny(), "the step's scale must take its handle");
                        if !lay.narrow {
                            assert!(desk.is_none_or(|p| p == lay.picture), "{w}x{h}: the picture moved with the rows");
                            desk = Some(lay.picture);
                        }
                    }
                }
            }
        }
    }

    /// The span of the chart is a round number with round fifths.
    #[test]
    fn the_chart_spans_round_numbers() {
        for (t, want) in [(12.0, 15.0), (44.5, 50.0), (51.0, 60.0), (343.0, 400.0), (100.0, 100.0), (1.0, 1.0), (2999.0, 3000.0)] {
            assert_eq!(nice_span(t), want, "{t}");
        }
        assert_eq!(nice_span(f64::NAN), SHORTEST);
        for d in descents() {
            let h = horizon(&d);
            assert!((SHORTEST..=LONGEST).contains(&h) && nice_span(h) == h, "{h}");
        }
        // the batches keep their order: the sharpest and the flattest first
        assert_eq!(drawing_order(4).collect::<Vec<_>>(), vec![0, 3, 1, 2, 4, 5, 6, 7, 8, 9, 10, 11]);
        assert_eq!(drawing_order(2).count(), MOST);
    }

    /// The numbers behind the captions (run with --ignored --nocapture).
    #[test]
    #[ignore]
    fn explore() {
        let report = |name: &str, d: &Descent| {
            let total = d.loss0();
            let shares: Vec<String> = (0..d.n()).map(|i| format!("{:.4}", d.share(i) / total)).collect();
            let factors: Vec<String> = (0..d.n()).map(|i| format!("{:.4}", d.factor(i))).collect();
            println!(
                "{name}: eta {:.4} lam {:?} factors {factors:?} shares {shares:?} fate {:?} N {:?} bound {:?} lowest {:?} horizon {:.1}",
                d.eta,
                d.lam,
                d.fate(),
                d.steps_to(FALL),
                d.bound(FALL),
                d.lowest(),
                horizon(d)
            );
            if let Some(t) = d.lowest() {
                println!("   lowest loss share {:.5}", (d.ln_loss(t as f64) - d.ln_loss(0.0)).exp());
            }
        };
        report("opening", &StepSize::new().shown().descent());
        for k in 0..TRIALS.len() {
            report(TRIALS[k].label, &run(k).shown().descent());
        }
    }
}

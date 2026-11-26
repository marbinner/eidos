//! Fingerprints of the noise: why, for straight-line effects, the shape of a cloud of
//! points shows which way the arrow points (LiNGAM), and why a bell-shaped noise hides it.
//!
//! X causes Y: X = N_X, Y = bX + N_Y, the two noises independent. The cloud of (X, Y)
//! is the cloud of (N_X, N_Y) sheared: every point moved up by b times its X. A shear
//! keeps upright lines upright, so a square of flat noise becomes a parallelogram with
//! two sides still upright. Had Y caused X, two sides would be level. A round cloud of
//! bell-shaped noise becomes an ellipse, which has no sides and fits either way.
//!
//! Below the story, two ways of reading the fingerprint from the points alone: the
//! straight-line fits both ways and what each leaves over (one way a rectangle, the
//! other a slanted diamond), and the turn that finds the noises' directions without
//! being told them (independent component analysis): round the cloud off to the same
//! spread every way, turn a line through it, and find where its shadow is furthest
//! from a bell's.

use super::*;
use crate::math::causal::{self, Noise};
use glam::DVec2;
use std::f64::consts::{FRAC_PI_2, PI as PI64};

pub const DEF: SceneDef = SceneDef {
    id: "lingam",
    title: "Fingerprints of the noise",
    group: GROUP,
    blurb: "Why noise that is not bell-shaped shows which way an arrow points, and bell-shaped noise hides it.",
    make: |_| Box::new(Lingam::create()),
};

/// The most points there can be, the fewest, and how many the scene opens with.
const N_MAX: usize = 2000;
const N_MIN: usize = 20;
const N_START: usize = 600;
/// The largest |b| the slider and the handle allow.
const B_MAX: f64 = 2.0;
/// Where flat noise ends: ±√3 (its variance is 1).
const EDGE: f64 = 1.732_050_807_568_877_2;

/// The shape of a cloud is told from a bell's when its two coordinates at the best
/// turn are, in root mean square, further from normal (in kurtosis) than n points of
/// bell-shaped noise get one time in 40. That height times √n, for these numbers of
/// points n (measured: 8000 samples of bell-shaped noise at each; see `calibrate` in
/// the tests). In between, it is read off a straight line in the logarithm of n.
const GATE: [(f64, f64); 8] = [(20.0, 10.0), (30.0, 11.6), (50.0, 12.3), (100.0, 12.7), (200.0, 12.4), (400.0, 11.8), (1000.0, 11.5), (2000.0, 11.4)];
/// Found sides within this many degrees of upright and of level are both: then
/// neither variable moves the other, as far as the points can show.
const BOTH_WAYS: f64 = 6.0;

/// Seconds: a shear from the square to its value, the way back to the square, the
/// pause on the square between them, a glide to a new sample, and the time in which
/// each of a glide's two eases covers two thirds of what is left of the way.
const SHEAR_TIME: f64 = 1.6;
const BACK_TIME: f64 = 0.7;
const HOLD_TIME: f64 = 0.35;
const CHANGE_TIME: f64 = 0.9;
const EASE: f64 = 0.08;
/// Seconds for the bell to be blended in, in the experiment that does it, and for the
/// line to be turned through the round cloud once.
const MORPH_TIME: f64 = 4.0;
const SWEEP_TIME: f64 = 6.0;

// The colours: one meaning each (the section's for cause and effect).
/// The points.
const DOTS: Color = Color::hex(0xD9E2F2);
/// The sides that the turn finds from the points alone.
const FOUND: Color = pal::WHITE;
/// How far from normal the shadow is, as the line turns.
const MEASURE: Color = pal::YELLOW;
/// What a straight-line fit leaves over, and its spread.
const LEFT: Color = pal::PURPLE;

/// The seed of the sample the scene opens with, and every experiment's.
const SEED: u64 = 11;

// ----- the statistics -----------------------------------------------------------------

/// What turning the cloud finds: the cloud rescaled to the same spread every way
/// (whitened), the fourth moments of that round cloud, and from them how far from
/// normal its shadow is on a line at any angle.
#[derive(Clone, Debug)]
struct Turn {
    mean: DVec2,
    /// The covariance's square root and its inverse (symmetric: [a, b, d] of [[a, b], [b, d]]).
    root: [f64; 3],
    white: [f64; 3],
    /// The averages of z₁⁴, z₁³z₂, z₁²z₂², z₁z₂³, z₂⁴ over the whitened cloud.
    m4: [f64; 5],
    /// The turn (radians, 0 to a quarter turn) whose two coordinates are furthest from
    /// normal, in root mean square of their kurtoses.
    best: f64,
}

/// The symmetric square root of [[a, b], [b, d]] and of its inverse, or None if it is
/// not safely positive (the points lie on a line, or there is no spread at all).
fn roots(a: f64, b: f64, d: f64) -> Option<([f64; 3], [f64; 3])> {
    let (mid, half) = ((a + d) * 0.5, (((a - d) * 0.5).powi(2) + b * b).sqrt());
    let (l1, l2) = (mid + half, mid - half);
    if !(l1.is_finite() && l2 > 1e-9 * l1.max(1e-300)) {
        return None;
    }
    // the unit eigenvector of the larger eigenvalue, and the one across it
    let v = if b.abs() > 1e-300 * l1 { DVec2::new(b, l1 - a).normalize_or_zero() } else if a >= d { DVec2::X } else { DVec2::Y };
    if v == DVec2::ZERO {
        return None;
    }
    let w = DVec2::new(-v.y, v.x);
    let make = |f1: f64, f2: f64| [f1 * v.x * v.x + f2 * w.x * w.x, f1 * v.x * v.y + f2 * w.x * w.y, f1 * v.y * v.y + f2 * w.y * w.y];
    Some((make(l1.sqrt(), l2.sqrt()), make(1.0 / l1.sqrt(), 1.0 / l2.sqrt())))
}

/// A symmetric 2 × 2 matrix [a, b, d] times a vector.
fn apply(m: [f64; 3], p: DVec2) -> DVec2 {
    DVec2::new(m[0] * p.x + m[1] * p.y, m[1] * p.x + m[2] * p.y)
}

impl Turn {
    /// None for fewer than four points, or points with no spread across some line.
    fn of(points: &[DVec2]) -> Option<Turn> {
        let n = points.len();
        if n < 4 {
            return None;
        }
        let mean = points.iter().copied().sum::<DVec2>() / n as f64;
        let (mut a, mut b, mut d) = (0.0, 0.0, 0.0);
        for p in points {
            let q = *p - mean;
            (a, b, d) = (a + q.x * q.x, b + q.x * q.y, d + q.y * q.y);
        }
        let (root, white) = roots(a / n as f64, b / n as f64, d / n as f64)?;
        let mut m4 = [0.0; 5];
        for p in points {
            let z = apply(white, *p - mean);
            let (z1, z2) = (z.x, z.y);
            m4[0] += z1 * z1 * z1 * z1;
            m4[1] += z1 * z1 * z1 * z2;
            m4[2] += z1 * z1 * z2 * z2;
            m4[3] += z1 * z2 * z2 * z2;
            m4[4] += z2 * z2 * z2 * z2;
        }
        for m in &mut m4 {
            *m /= n as f64;
        }
        if !m4.iter().all(|m| m.is_finite()) {
            return None;
        }
        let mut t = Turn { mean, root, white, m4, best: 0.0 };
        // the best turn: the best of a grid of half degrees, then closer by golden sections
        let steps = 180;
        let at = |k: usize| FRAC_PI_2 * k as f64 / steps as f64;
        let k = (0..steps).max_by(|i, j| t.score(at(*i)).total_cmp(&t.score(at(*j)))).unwrap_or(0);
        let (mut lo, mut hi) = (at(k) - FRAC_PI_2 / steps as f64, at(k) + FRAC_PI_2 / steps as f64);
        let g = 0.5 * (5f64.sqrt() - 1.0);
        for _ in 0..40 {
            let (c, e) = (hi - g * (hi - lo), lo + g * (hi - lo));
            if t.score(c) > t.score(e) {
                hi = e;
            } else {
                lo = c;
            }
        }
        t.best = (0.5 * (lo + hi)).rem_euclid(FRAC_PI_2);
        Some(t)
    }

    /// The kurtosis of the round cloud's shadow on the line at angle `a` (radians):
    /// 0 for a bell.
    fn kurtosis(&self, a: f64) -> f64 {
        let (s, c) = a.sin_cos();
        let m = &self.m4;
        c.powi(4) * m[0] + 4.0 * c.powi(3) * s * m[1] + 6.0 * c * c * s * s * m[2] + 4.0 * c * s.powi(3) * m[3] + s.powi(4) * m[4] - 3.0
    }

    /// How far from normal the two coordinates of the turn `a` are, together: the root
    /// mean square of their kurtoses.
    fn score(&self, a: f64) -> f64 {
        ((self.kurtosis(a).powi(2) + self.kurtosis(a + FRAC_PI_2).powi(2)) * 0.5).sqrt()
    }

    /// The directions in the plane of the data along which the two noises found move
    /// a point, for the turn `a`, as they are scaled: the cloud's sides. (A point is
    /// the mean plus ŝ₁ times the first plus ŝ₂ times the second, ŝ₁ and ŝ₂ the found
    /// noises, each of variance 1.)
    fn mixing_at(&self, a: f64) -> [DVec2; 2] {
        let (s, c) = a.sin_cos();
        [apply(self.root, DVec2::new(c, s)), apply(self.root, DVec2::new(-s, c))]
    }

    /// The same as unit vectors.
    fn sides_at(&self, a: f64) -> [DVec2; 2] {
        self.mixing_at(a).map(|v| v.normalize_or_zero())
    }

    /// How far (degrees) the most upright side is from upright, and the most level
    /// one from level.
    fn tilts(&self) -> (f64, f64) {
        let from_up = |v: DVec2| v.x.abs().atan2(v.y.abs()).to_degrees();
        let [a, b] = self.sides_at(self.best);
        let (ua, ub) = (from_up(a), from_up(b));
        (ua.min(ub), (90.0 - ua).min(90.0 - ub))
    }

    /// A point of the data, rescaled to the round cloud.
    fn whiten(&self, p: DVec2) -> DVec2 {
        apply(self.white, p - self.mean)
    }

    /// The found noises of a point, at the turn `a`.
    fn unmix(&self, p: DVec2, a: f64) -> DVec2 {
        let z = self.whiten(p);
        let (s, c) = a.sin_cos();
        DVec2::new(c * z.x + s * z.y, -s * z.x + c * z.y)
    }
}

/// What the points alone say.
#[derive(Clone, Copy, Debug, PartialEq)]
enum Verdict {
    /// Too few points, or none apart from a line.
    Nothing,
    /// The cloud's shape is not told from a bell's: no sides to read.
    Round,
    /// Sides upright and level at once: neither variable moves the other.
    Apart,
    /// An upright side: X causes Y.
    Forwards,
    /// A level side: Y causes X.
    Backwards,
}

/// How far from normal n points of bell-shaped noise get one time in 40, at their
/// best turn (see `GATE`).
fn gate(n: usize) -> f64 {
    let n = n.max(1) as f64;
    let c = match GATE.iter().position(|(m, _)| *m >= n) {
        Some(0) => GATE[0].1,
        Some(i) => {
            let (a, b) = (GATE[i - 1], GATE[i]);
            a.1 + (b.1 - a.1) * (n.ln() - a.0.ln()) / (b.0.ln() - a.0.ln())
        }
        None => GATE[GATE.len() - 1].1,
    };
    c / n.sqrt()
}

/// The verdict on a cloud of `n` points from what its turn found.
fn verdict(turn: Option<&Turn>, n: usize) -> Verdict {
    let Some(t) = turn else { return Verdict::Nothing };
    if t.score(t.best) <= gate(n) {
        return Verdict::Round;
    }
    let (up, level) = t.tilts();
    if up < BOTH_WAYS && level < BOTH_WAYS {
        Verdict::Apart
    } else if up < level {
        Verdict::Forwards
    } else {
        Verdict::Backwards
    }
}

/// The correlation of the squares of two samples (each about its mean): 0 for
/// independent ones, and not, in general, for ones that are only uncorrelated.
fn square_correlation(a: &[f64], b: &[f64]) -> f64 {
    let (ma, mb) = (causal::mean(a), causal::mean(b));
    let sa: Vec<f64> = a.iter().map(|v| (v - ma).powi(2)).collect();
    let sb: Vec<f64> = b.iter().map(|v| (v - mb).powi(2)).collect();
    causal::correlation(&sa, &sb)
}

/// How what a fit leaves over spreads along its input: at `points` places from the
/// 2nd to the 98th percentile of the input, the root mean square of what is left
/// there, each point weighted by how near its input is (within half the input's
/// standard deviation, the nearer the more).
fn spread_along(input: &[f64], left: &[f64], points: usize) -> Vec<(f64, f64)> {
    let n = input.len().min(left.len());
    if n < 8 || points < 2 {
        return Vec::new();
    }
    let mut sorted = input[..n].to_vec();
    sorted.sort_by(f64::total_cmp);
    let (lo, hi) = (sorted[n / 50], sorted[n - 1 - n / 50]);
    let h = 0.5 * causal::variance(&input[..n]).sqrt();
    if !(hi > lo && h > 0.0 && h.is_finite()) {
        return Vec::new();
    }
    (0..points)
        .map(|k| {
            let at = lo + (hi - lo) * k as f64 / (points - 1) as f64;
            let (mut top, mut bottom) = (0.0, 0.0);
            for i in 0..n {
                let u = (input[i] - at) / h;
                let w = (1.0 - u * u).max(0.0);
                top += w * left[i] * left[i];
                bottom += w;
            }
            (at, if bottom > 0.0 { (top / bottom).sqrt() } else { 0.0 })
        })
        .collect()
}

/// The value below which a share `p` of the samples lie.
fn quantile(sorted: &[f64], p: f64) -> f64 {
    if sorted.is_empty() {
        return 0.0;
    }
    sorted[((sorted.len() - 1) as f64 * p.max(0.0).min(1.0)).round() as usize]
}

// ----- the model ----------------------------------------------------------------------

/// The shear as drawn: [v, h], with Y moved by v X and X by h Y. Only one is ever
/// not 0: v = b when X causes Y, h = b when Y causes X.
type Shear = [f64; 2];

fn shear_of(b: f64, forwards: bool) -> Shear {
    if forwards {
        [b, 0.0]
    } else {
        [0.0, b]
    }
}

/// Where the shear takes the point with noises `n` = (N_X, N_Y).
fn sheared(s: Shear, n: DVec2) -> DVec2 {
    DVec2::new(n.x + s[1] * n.y, n.y + s[0] * n.x)
}

/// The noise of each point, blended from its draw of the shape and its draw of a bell.
fn blend(shape: DVec2, bell: DVec2, w: f64) -> DVec2 {
    let w = w.max(0.0).min(1.0);
    shape * (1.0 - w).sqrt() + bell * w.sqrt()
}

/// N_MAX draws of a shape of noise, two to a point (X's and Y's), from a seed: the
/// same draws for the same seed and shape whatever else has happened.
fn draws(noise: Noise, seed: u64) -> Vec<DVec2> {
    let which = Noise::ALL.iter().position(|n| *n == noise).unwrap_or(0) as u64;
    let mut rng = Rng::new(seed.wrapping_mul(0x9E37_79B9).wrapping_add(17 * which + 1));
    (0..N_MAX).map(|_| DVec2::new(noise.sample(&mut rng), noise.sample(&mut rng))).collect()
}

/// The bell draws of a seed (they do not depend on the shape).
fn bells(seed: u64) -> Vec<DVec2> {
    draws(Noise::Normal, seed ^ 0xB311)
}

/// Half the side of the square of noise values a panel shows, for a shape blended
/// with a bell: room for most of the points (the few beyond are left out).
fn reach(noise: Noise, w: f64) -> f64 {
    let own = match noise {
        Noise::Uniform => 2.15,
        Noise::TwoHumps => 2.5,
        Noise::Normal => 2.8,
        Noise::Laplace => 3.3,
    };
    own + (2.8 - own) * w.max(0.0).min(1.0).sqrt()
}

fn smoothstep(t: f64) -> f64 {
    let t = t.max(0.0).min(1.0);
    t * t * (3.0 - 2.0 * t)
}

/// A value drawn gliding to where it should be: two eases one after the other (what
/// is drawn follows a value between, which follows the goal), so that it sets off
/// and arrives at no speed, and is exactly there in the end.
#[derive(Clone, Copy, Debug, PartialEq)]
struct Glide {
    drawn: f64,
    between: f64,
}

impl Glide {
    fn at(v: f64) -> Glide {
        Glide { drawn: v, between: v }
    }

    /// `dt` seconds nearer to `goal`: there once both are within `near` of it.
    fn toward(&mut self, goal: f64, dt: f64, near: f64) {
        let k = 1.0 - (-dt.max(0.0) / EASE).exp();
        self.between += (goal - self.between) * k;
        self.drawn += (self.between - self.drawn) * k;
        if !(self.drawn.is_finite() && self.between.is_finite()) || ((goal - self.drawn).abs() < near && (goal - self.between).abs() < near) {
            *self = Glide::at(goal);
        }
    }

    fn there(&self, goal: f64) -> bool {
        self.drawn == goal && self.between == goal
    }
}

/// One stretch of the shear's animation: to `to`, taking `time` seconds, eased in and out.
#[derive(Clone, Copy, Debug)]
struct Leg {
    to: Shear,
    time: f64,
}

/// What the lower part of the picture shows on a narrow canvas (a wide one shows both).
#[derive(Clone, Copy, Debug, PartialEq)]
enum Lower {
    /// The straight-line fits both ways, and what each leaves over.
    Fits,
    /// The turn through the round cloud.
    Turning,
}

/// Where the line through the round cloud is.
#[derive(Clone, Copy, Debug, PartialEq)]
enum TurnAt {
    /// At the best turn.
    Best,
    /// Where the visitor put it (radians).
    Held(f64),
    /// Being turned once round: this share of the way (below 0: not started yet).
    Sweep(f64),
}

/// One lesson's settings.
#[derive(Clone, Copy, Debug)]
struct Setup {
    noise: Noise,
    /// How bell-shaped the noise is (0: the shape, 1: a bell); with `morph`, the blend
    /// is played from 0 to this.
    bell: f64,
    morph: bool,
    b: f64,
    forwards: bool,
    n: usize,
    lower: Lower,
    sweep: bool,
}

/// The experiments: button, what to look for, settings.
const TRIALS: [(&str, &str, Setup); 5] = [
    (
        r"flat noise: $X$ causes $Y$",
        r"Each noise is flat: the two together fill a square. Watch $b$ grow from 0 to 1: every point moves straight up, by $b$ times its $X$, so the square leans while its upright sides stay upright (orange). From the points alone, the turn finds a side upright to within a degree: $X \to Y$.",
        Setup { noise: Noise::Uniform, bell: 0.0, morph: false, b: 1.0, forwards: true, n: N_START, lower: Lower::Fits, sweep: false },
    ),
    (
        "turn the arrow round",
        r"The same noises and the same $b$, but now $Y$ causes $X$: every point moves sideways, by $b$ times its $Y$, and the sides that stay as they were are the level ones. The fingerprint has moved with the arrow, and the verdict with it: $Y \to X$.",
        Setup { noise: Noise::Uniform, bell: 0.0, morph: false, b: 1.0, forwards: false, n: N_START, lower: Lower::Fits, sweep: false },
    ),
    (
        "blend in a bell",
        r"The flat noise is blended with a bell until it is one (its $w$ is written by the noises). The square's corners round off; at $w = 0.34$ the cloud's shape can no longer be told from a bell's, and the verdict becomes: cannot tell. What is left is an ellipse, and the ellipse that $X \to Y$ makes with $b = 1$ is made exactly by $Y \to X$ too, with $c = 0.5$ and noises of variance 2 and 0.5.",
        Setup { noise: Noise::Uniform, bell: 1.0, morph: true, b: 1.0, forwards: true, n: N_START, lower: Lower::Fits, sweep: false },
    ),
    (
        "peaked noise",
        r"Peaked (Laplace) noise: most values near 0, a few far out. No sharp sides now, but arms: the far-out values of each noise stretch the cloud along that noise's own direction, one arm upright. Only those few points carry the fingerprint, so it takes more of them: with these 2000 the turn finds a side 1.6° from upright, with the first 600 of them 3.3°. Either way, $X \to Y$.",
        Setup { noise: Noise::Laplace, bell: 0.0, morph: false, b: 1.0, forwards: true, n: N_MAX, lower: Lower::Fits, sweep: false },
    ),
    (
        "turn and measure",
        r"How the sides are found without being told. The cloud is rescaled to the same spread every way, which brings back the noises' own cloud, only turned: here four clumps, as each noise has two humps. A line is turned through it once round, and the yellow curve is how far from normal the cloud's shadow on the line is. It peaks at two angles a right angle apart, where the shadow is one noise alone; between them the shadow mixes the two, and a mix is nearer to normal. The best turn puts the two lines on the peaks; carried back to the data, they are the white dashed lines.",
        Setup { noise: Noise::TwoHumps, bell: 0.0, morph: false, b: 1.0, forwards: true, n: N_START, lower: Lower::Turning, sweep: true },
    ),
];

/// Everything worked out from the points as they are drawn.
struct Seen {
    /// The points' noises and their (X, Y).
    noise: Vec<DVec2>,
    data: Vec<DVec2>,
    turn: Option<Turn>,
    verdict: Verdict,
    /// For each found noise, at the best turn: from its 1st to its 99th percentile
    /// among the points (how far the cloud reaches along its side).
    span: [(f64, f64); 2],
    /// The straight-line fits, Y from X and X from Y: (slope, intercept) each, what
    /// each leaves over, and how that spreads along the input.
    forth: (f64, f64),
    back: (f64, f64),
    forth_left: Vec<f64>,
    back_left: Vec<f64>,
    forth_spread: Vec<(f64, f64)>,
    back_spread: Vec<(f64, f64)>,
    /// The points rescaled to the round cloud.
    round: Vec<DVec2>,
    /// The shadow's |kurtosis| at every degree from 0 to 180.
    curve: Vec<f64>,
    /// The points' X and Y.
    x: Vec<f64>,
    y: Vec<f64>,
    /// For the fits Y from X and X from Y: the correlation of what is left with the
    /// input, and of their squares; and how far each panel reaches (the input, what
    /// is left and its band all in view).
    left_correlation: [f64; 2],
    square_correlation: [f64; 2],
    fit_reach: [f64; 2],
}

impl Seen {
    fn of(noise: Vec<DVec2>, shear: Shear) -> Seen {
        let data: Vec<DVec2> = noise.iter().map(|n| sheared(shear, *n)).collect();
        let turn = Turn::of(&data);
        let verdict = verdict(turn.as_ref(), data.len());
        let span = turn.as_ref().map_or([(0.0, 0.0); 2], |t| {
            let found: Vec<DVec2> = data.iter().map(|p| t.unmix(*p, t.best)).collect();
            let mut s1: Vec<f64> = found.iter().map(|s| s.x).collect();
            let mut s2: Vec<f64> = found.iter().map(|s| s.y).collect();
            s1.sort_by(f64::total_cmp);
            s2.sort_by(f64::total_cmp);
            [(quantile(&s1, 0.01), quantile(&s1, 0.99)), (quantile(&s2, 0.01), quantile(&s2, 0.99))]
        });
        let x: Vec<f64> = data.iter().map(|p| p.x).collect();
        let y: Vec<f64> = data.iter().map(|p| p.y).collect();
        let (forth, back) = (causal::line(&x, &y), causal::line(&y, &x));
        let forth_left: Vec<f64> = data.iter().map(|p| p.y - forth.0 * p.x - forth.1).collect();
        let back_left: Vec<f64> = data.iter().map(|p| p.x - back.0 * p.y - back.1).collect();
        let (forth_spread, back_spread) = (spread_along(&x, &forth_left, 25), spread_along(&y, &back_left, 25));
        let round = turn.as_ref().map_or_else(Vec::new, |t| data.iter().map(|p| t.whiten(*p)).collect());
        let curve = turn.as_ref().map_or_else(Vec::new, |t| (0..=180).map(|d| t.kurtosis((d as f64).to_radians()).abs()).collect());
        let left_correlation = [causal::correlation(&forth_left, &x), causal::correlation(&back_left, &y)];
        let square_correlation = [square_correlation(&forth_left, &x), square_correlation(&back_left, &y)];
        let reach = |input: &[f64], left: &[f64], spread: &[(f64, f64)]| {
            let mut all: Vec<f64> = input.iter().chain(left.iter()).map(|v| v.abs()).collect();
            all.sort_by(f64::total_cmp);
            let band = spread.iter().fold(0.0f64, |m, (_, sd)| m.max(2.0 * sd));
            (quantile(&all, 0.995) * 1.1).max(band * 1.08).max(1e-6)
        };
        let fit_reach = [reach(&x, &forth_left, &forth_spread), reach(&y, &back_left, &back_spread)];
        Seen {
            noise,
            data,
            turn,
            verdict,
            span,
            forth,
            back,
            forth_left,
            back_left,
            forth_spread,
            back_spread,
            round,
            curve,
            x,
            y,
            left_correlation,
            square_correlation,
            fit_reach,
        }
    }

    /// The tilts of the found sides from upright and from level (degrees).
    fn tilts(&self) -> Option<(f64, f64)> {
        self.turn.as_ref().map(Turn::tilts)
    }

    /// How far from normal the best turn's coordinates are, and the gate it is held to.
    fn score(&self) -> Option<(f64, f64)> {
        self.turn.as_ref().map(|t| (t.score(t.best), gate(self.data.len())))
    }

    /// Does the spread of what the fit leaves over change along its input? Told by
    /// the correlation of their squares, against what independent samples of this many
    /// points give: beyond three times 1/√n. (None: too few points to say.)
    fn spread_changes(&self, forth: bool) -> Option<bool> {
        let n = self.data.len();
        if n < N_MIN {
            return None;
        }
        Some(self.square_correlation[usize::from(!forth)].abs() > 3.0 / (n as f64).sqrt())
    }

    fn inputs(&self, forth: bool) -> &[f64] {
        if forth {
            &self.x
        } else {
            &self.y
        }
    }
}

pub struct Lingam {
    noise: Noise,
    /// How bell-shaped the noise is: 0 the shape alone, 1 a bell.
    bell: f64,
    b: f64,
    forwards: bool,
    n: usize,
    seed: u64,
    lower: Lower,
    /// Each point's draws: of the shape, and of a bell (N_MAX each).
    shape_draws: Vec<DVec2>,
    bell_draws: Vec<DVec2>,
    /// After a change of shape or sample, the noises the points glide from, and how
    /// far they have come (1: there).
    old: Vec<DVec2>,
    change: f64,
    /// The blend as drawn, and its animation: (from, to, seconds so far, seconds in all).
    bell_shown: f64,
    morph: Option<(f64, f64, f64, f64)>,
    /// The shear as drawn, where its present leg started, the legs still to go, and
    /// the time into the first.
    shear: Shear,
    from: Shear,
    legs: Vec<Leg>,
    leg_time: f64,
    /// The size of the square of noise the panels show, as drawn (it follows the shape).
    reach_shown: Glide,
    /// How far the cloud spreads across and up, in units of that square, as drawn (it
    /// follows b, but not while a handle is held).
    stretch_shown: [Glide; 2],
    /// The line through the round cloud, and the angle it is drawn at (radians).
    turn_at: TurnAt,
    turn: Option<Glide>,
    /// The best turn as the found sides over the data are drawn at: it glides, and
    /// keeps its lines in order (the pair is the same a quarter turn on).
    found: Option<Glide>,
    /// Which top corner of the data's panel its key is in (0 left, 1 right: the one the
    /// cloud leaves free), as drawn.
    key_side: Glide,
    seen: Seen,
    /// What `seen` was worked out from.
    seen_key: Option<[u64; 7]>,
}

impl Lingam {
    /// (Nothing here needs the graphics device, so the tests can make one too.)
    fn create() -> Self {
        let mut s = Lingam {
            noise: Noise::Uniform,
            bell: 0.0,
            b: 1.0,
            forwards: true,
            n: N_START,
            seed: SEED,
            lower: Lower::Fits,
            shape_draws: draws(Noise::Uniform, SEED),
            bell_draws: bells(SEED),
            old: Vec::new(),
            change: 1.0,
            bell_shown: 0.0,
            morph: None,
            shear: [0.0, 0.0],
            from: [0.0, 0.0],
            legs: Vec::new(),
            leg_time: 0.0,
            reach_shown: Glide::at(reach(Noise::Uniform, 0.0)),
            stretch_shown: [Glide::at(1.0), Glide::at(2.0)],
            turn_at: TurnAt::Best,
            turn: None,
            found: None,
            key_side: Glide::at(0.0),
            seen: Seen::of(Vec::new(), [0.0, 0.0]),
            seen_key: None,
        };
        s.set_up(0);
        s.look();
        s
    }

    /// Set up one of the experiments in `TRIALS`.
    fn set_up(&mut self, k: usize) {
        let Some((_, _, setup)) = TRIALS.get(k) else { return };
        self.apply(*setup);
    }

    fn apply(&mut self, s: Setup) {
        self.change_draws(s.noise, SEED);
        self.n = s.n.clamp(N_MIN, N_MAX);
        self.lower = s.lower;
        if s.morph {
            // (begun after the shear, so that the square is seen leaning first)
            self.bell = s.bell.max(0.0).min(1.0);
            self.morph = Some((0.0, self.bell, -(BACK_TIME + HOLD_TIME + SHEAR_TIME + 0.4), MORPH_TIME));
            self.bell_shown = 0.0;
        } else {
            self.set_bell(s.bell);
        }
        (self.b, self.forwards) = (s.b.max(-B_MAX).min(B_MAX), s.forwards);
        // the shear is played again from the square: back to it first, if need be
        self.legs.clear();
        (self.from, self.leg_time) = (self.shear, 0.0);
        if self.shear != [0.0, 0.0] {
            self.legs.push(Leg { to: [0.0, 0.0], time: BACK_TIME });
        }
        self.legs.push(Leg { to: [0.0, 0.0], time: HOLD_TIME });
        self.legs.push(Leg { to: shear_of(self.b, self.forwards), time: SHEAR_TIME });
        let wait = self.legs.iter().map(|l| l.time).sum::<f64>();
        self.turn_at = if s.sweep { TurnAt::Sweep(-wait / SWEEP_TIME) } else { TurnAt::Best };
    }

    /// Glide to the draws of another shape, or another sample (nothing moves if they
    /// are the ones already drawn).
    fn change_draws(&mut self, noise: Noise, seed: u64) {
        if noise == self.noise && seed == self.seed {
            return;
        }
        self.old = self.noise_now();
        self.shape_draws = draws(noise, seed);
        if seed != self.seed {
            self.bell_draws = bells(seed);
        }
        (self.noise, self.seed, self.change) = (noise, seed, 0.0);
    }

    /// The blend, at once (the slider), with no animation of it pending.
    fn set_bell(&mut self, w: f64) {
        self.bell = w.max(0.0).min(1.0);
        self.bell_shown = self.bell;
        self.morph = None;
    }

    /// The cause turned round: the shear goes back to the square and out the other way.
    fn set_forwards(&mut self, forwards: bool) {
        if forwards == self.forwards {
            return;
        }
        self.forwards = forwards;
        self.legs.clear();
        (self.from, self.leg_time) = (self.shear, 0.0);
        if self.shear != [0.0, 0.0] {
            self.legs.push(Leg { to: [0.0, 0.0], time: BACK_TIME });
        }
        self.legs.push(Leg { to: shear_of(self.b, forwards), time: SHEAR_TIME * 0.7 });
    }

    /// A new b, at once (the slider, the handle).
    fn set_b(&mut self, b: f64) {
        if !b.is_finite() {
            return;
        }
        self.b = b.max(-B_MAX).min(B_MAX);
        self.legs.clear();
        self.shear = shear_of(self.b, self.forwards);
        self.from = self.shear;
    }

    /// Each point's noise as drawn now (all N_MAX of them).
    fn noise_now(&self) -> Vec<DVec2> {
        let s = smoothstep(self.change);
        (0..N_MAX)
            .map(|i| {
                let to = blend(self.shape_draws.get(i).copied().unwrap_or_default(), self.bell_draws.get(i).copied().unwrap_or_default(), self.bell_shown);
                match self.old.get(i) {
                    Some(from) if s < 1.0 => *from + (to - *from) * s,
                    _ => to,
                }
            })
            .collect()
    }

    /// Is anything still on its way?
    fn moving(&self) -> bool {
        !self.legs.is_empty()
            || self.change < 1.0
            || self.morph.is_some()
            || matches!(self.turn_at, TurnAt::Sweep(_))
            || !self.reach_shown.there(reach(self.noise, self.bell_shown))
            || !self.stretch_shown[0].there(self.stretch_goal()[0])
            || !self.stretch_shown[1].there(self.stretch_goal()[1])
            || !self.key_side.there(self.key_goal())
            || self.turn.map(|t| t.drawn) != self.turn_goal()
            || self.turn.is_some_and(|t| t.drawn != t.between)
            || self.found.map(|g| g.drawn) != self.found_goal()
            || self.found.is_some_and(|g| g.drawn != g.between)
    }

    /// The corner of the data's panel the cloud leaves free: the top left when it
    /// leans to the right (b ≥ 0), the top right when it leans to the left.
    fn key_goal(&self) -> f64 {
        if self.b >= 0.0 {
            0.0
        } else {
            1.0
        }
    }

    /// The spread of the cloud the panels make room for, in units of the noise square:
    /// across and up.
    fn stretch_goal(&self) -> [f64; 2] {
        let s = shear_of(self.b, self.forwards);
        [1.0 + s[1].abs(), 1.0 + s[0].abs()]
    }

    /// Where the line through the round cloud should be drawn (None: no line). Turned
    /// once round, it goes on to the best turn and stops there, slowing down to it.
    fn turn_goal(&self) -> Option<f64> {
        let t = self.seen.turn.as_ref()?;
        Some(match self.turn_at {
            TurnAt::Best => match self.turn {
                // (the pair of lines is the same a quarter turn on: the nearest way round)
                Some(g) => g.drawn + (t.best - g.drawn + FRAC_PI_2 * 0.5).rem_euclid(FRAC_PI_2) - FRAC_PI_2 * 0.5,
                None => t.best,
            },
            TurnAt::Held(a) => a,
            TurnAt::Sweep(s) => (PI64 + t.best) * smoothstep(s),
        })
    }

    /// The best turn, the nearest way round from where the found sides are drawn.
    fn found_goal(&self) -> Option<f64> {
        let t = self.seen.turn.as_ref()?;
        Some(match self.found {
            Some(g) => g.drawn + (t.best - g.drawn + FRAC_PI_2 * 0.5).rem_euclid(FRAC_PI_2) - FRAC_PI_2 * 0.5,
            None => t.best,
        })
    }

    /// How far round the line has been turned, for the curve drawn behind it (degrees,
    /// up to 180: all of it).
    fn reached(&self) -> f64 {
        match (self.turn_at, self.turn) {
            (TurnAt::Sweep(s), Some(g)) if s > 0.0 => g.drawn.to_degrees().max(0.0).min(180.0),
            (TurnAt::Sweep(_), _) => 0.0,
            _ => 180.0,
        }
    }

    /// Move the lesson on by `dt` seconds of the scene's time: the shear, the bell
    /// blended in, the line turned once round.
    fn advance(&mut self, dt: f64) {
        if dt <= 0.0 {
            return;
        }
        self.leg_time += dt;
        while let Some(leg) = self.legs.first().copied() {
            if self.leg_time < leg.time {
                let s = smoothstep(self.leg_time / leg.time.max(1e-9));
                self.shear = [self.from[0] + (leg.to[0] - self.from[0]) * s, self.from[1] + (leg.to[1] - self.from[1]) * s];
                break;
            }
            self.leg_time -= leg.time;
            (self.shear, self.from) = (leg.to, leg.to);
            self.legs.remove(0);
        }
        if self.legs.is_empty() {
            self.leg_time = 0.0;
        }
        if let Some((from, to, t, time)) = self.morph {
            let t = t + dt;
            self.bell_shown = from + (to - from) * smoothstep(t / time);
            self.morph = (t < time).then_some((from, to, t, time));
            if self.morph.is_none() {
                self.bell_shown = to;
            }
        }
        if let TurnAt::Sweep(s) = self.turn_at {
            let s = s + dt / SWEEP_TIME;
            self.turn_at = if s >= 1.0 { TurnAt::Best } else { TurnAt::Sweep(s) };
        }
    }

    /// Bring what follows the visitor's changes `dt` seconds of real time nearer: the
    /// points to a new sample or shape, the framing to the shape and to b (not while
    /// a handle is held: the scale would slide under the pointer).
    fn follow(&mut self, dt: f64, holding: bool) {
        let dt = dt.max(0.0);
        if self.change < 1.0 {
            self.change = (self.change + dt / CHANGE_TIME).min(1.0);
            if self.change >= 1.0 {
                self.old.clear();
            }
        }
        self.reach_shown.toward(reach(self.noise, self.bell_shown), dt, 1e-4);
        self.key_side.toward(self.key_goal(), dt, 1e-3);
        if !holding {
            let goal = self.stretch_goal();
            for (shown, goal) in self.stretch_shown.iter_mut().zip(goal) {
                shown.toward(goal, dt, 1e-4);
            }
        }
    }

    /// Work out what the points as drawn say (only when they have changed).
    fn look(&mut self) {
        let key = [
            self.n as u64,
            self.seed,
            Noise::ALL.iter().position(|n| *n == self.noise).unwrap_or(0) as u64,
            self.change.to_bits(),
            self.bell_shown.to_bits(),
            self.shear[0].to_bits(),
            self.shear[1].to_bits(),
        ];
        if self.seen_key == Some(key) {
            return;
        }
        let mut noise = self.noise_now();
        noise.truncate(self.n);
        self.seen = Seen::of(noise, self.shear);
        self.seen_key = Some(key);
    }

    /// Bring the line through the round cloud `dt` seconds nearer to where it goes
    /// (straight there while it is held, or being turned once round).
    fn turn_on(&mut self, dt: f64) {
        let goal = self.turn_goal();
        let glides = match self.turn_at {
            TurnAt::Best => true,
            TurnAt::Held(_) => false,
            TurnAt::Sweep(s) => s <= 0.0,
        };
        self.turn = match (self.turn, goal) {
            (Some(mut g), Some(goal)) if glides => {
                g.toward(goal, dt, 1e-4);
                Some(g)
            }
            (_, goal) => goal.map(Glide::at),
        };
        self.found = match (self.found, self.found_goal()) {
            (Some(mut g), Some(goal)) => {
                g.toward(goal, dt, 1e-4);
                Some(g)
            }
            (_, goal) => goal.map(Glide::at),
        };
    }

    /// The angle the line through the round cloud is drawn at.
    fn turn_drawn(&self) -> Option<f64> {
        self.turn.map(|g| g.drawn)
    }

    /// One frame's worth of everything that moves, without drawing (the tests run the
    /// scene with this).
    fn step(&mut self, dt: f64, real_dt: f64, holding: bool) {
        self.advance(dt);
        self.follow(real_dt, holding);
        self.look();
        self.turn_on(real_dt);
    }
}

// ----- the layout ---------------------------------------------------------------------

/// A rectangle on the screen, logical pixels from the top-left corner.
#[derive(Clone, Copy, Debug, PartialEq)]
struct Rect {
    min: Vec2,
    max: Vec2,
}

impl Rect {
    fn new(min: Vec2, size: Vec2) -> Rect {
        Rect { min, max: min + size.max(Vec2::splat(1.0)) }
    }

    fn size(&self) -> Vec2 {
        self.max - self.min
    }

    fn mid(&self) -> Vec2 {
        (self.min + self.max) * 0.5
    }

    fn contains(&self, p: Vec2) -> bool {
        p.cmpge(self.min).all() && p.cmple(self.max).all()
    }
}

/// Where everything goes on a canvas: on a wide one the story along the top (noise,
/// shear, data, the two candidate models) and both ways of reading it below; on a
/// narrow one the story down the screen, and one way of reading it at the foot, with
/// tabs to choose which.
#[derive(Clone, Copy, Debug, PartialEq)]
struct Layout {
    narrow: bool,
    /// Text sizes: titles, labels, small print.
    big: f32,
    text: f32,
    small: f32,
    noise: Rect,
    data: Rect,
    /// The shear's arrow, from and to.
    arrow: (Vec2, Vec2),
    /// The two candidate models' cards.
    cards: [Rect; 2],
    /// Where the verdict is written: top-left corner and width.
    verdict: (Vec2, f32),
    /// The tabs for the lower part (narrow only): top-left corner and the size of one.
    tabs: Option<(Vec2, Vec2)>,
    /// The fits both ways, the round cloud, the curve (None: not shown).
    forth: Option<Rect>,
    back: Option<Rect>,
    round: Option<Rect>,
    curve: Option<Rect>,
}

impl Layout {
    fn of(view: Vec2, inset: f32, lower: Lower) -> Layout {
        let view = view.max(Vec2::splat(1.0));
        if view.x < 760.0 {
            Layout::narrow(view, inset, lower)
        } else {
            Layout::wide(view, inset)
        }
    }

    fn wide(view: Vec2, inset: f32) -> Layout {
        let (big, text, small) = (13.5, 12.0, 11.0);
        let side = 18.0;
        let title = big + 12.0;
        let verdict_h = 2.0 * (text + 6.0) + 4.0;
        let top = inset.max(0.0) + 6.0 + title;
        // the height left for the data's side and the lower row's
        let room = (view.y - top - verdict_h - 26.0 - title - small - 18.0).max(80.0);
        let gap = 22.0;
        let low_w = ((view.x - 2.0 * side - 4.6 * gap) / 4.4).max(30.0);
        let low = low_w.min(270.0).min((room / 3.0).max(110.0));
        let arrow_w = 168.0;
        let cards_w = (view.x * 0.17).clamp(150.0, 230.0);
        // the cards beside the data, or under the noise (whichever lets the data be larger)
        let beside = ((view.x - 2.0 * side - arrow_w - 20.0 - cards_w) / 1.5).min(room - low);
        let under = ((view.x - 2.0 * side - arrow_w) / 1.5).min(room - low);
        let below = under > beside && (0.5 * under - 3.0 * 14.0) * 0.5 >= 80.0;
        let s = if below { under } else { beside }.max(40.0);
        // the lower row as tall as the room left lets it, and what is over shared out
        let low = low_w.min(270.0).min(room - s).max(30.0);
        let top = top + ((room - s - low) * 0.4).max(0.0);
        let noise = Rect::new(Vec2::new(side, top + if below { 0.0 } else { 0.25 * s }), Vec2::splat(0.5 * s));
        let data = Rect::new(Vec2::new(side + 0.5 * s + arrow_w, top), Vec2::splat(s));
        let arrow = (Vec2::new(noise.max.x + 12.0, noise.mid().y), Vec2::new(data.min.x - 12.0, noise.mid().y));
        let cards = if below {
            let h = ((data.max.y - noise.max.y - 3.0 * 14.0) * 0.5).min(160.0);
            let card = |k: f32| Rect::new(Vec2::new(side, noise.max.y + 28.0 + k * (h + 14.0)), Vec2::new(0.5 * s, h));
            [card(0.0), card(1.0)]
        } else {
            let h = ((s - 14.0) * 0.5).min(160.0);
            let card = |k: f32| Rect::new(Vec2::new(data.max.x + 20.0, top + k * (h + 14.0)), Vec2::new(cards_w, h));
            [card(0.0), card(1.0)]
        };
        let verdict = (Vec2::new(data.min.x, data.max.y + 8.0), (view.x - side - data.min.x).max(40.0));
        let low_top = data.max.y + verdict_h + 26.0 + title - 12.0;
        let forth = Rect::new(Vec2::new(side, low_top), Vec2::splat(low));
        let back = Rect::new(Vec2::new(forth.max.x + gap, low_top), Vec2::splat(low));
        let round = Rect::new(Vec2::new(back.max.x + 1.8 * gap, low_top), Vec2::splat(low));
        let curve_x = round.max.x + gap;
        let curve = Rect::new(Vec2::new(curve_x, low_top), Vec2::new((view.x - side - curve_x).min(2.0 * low), low));
        let fits = low >= 80.0 && curve.size().x >= 100.0 && low_top + low + small + 6.0 <= view.y;
        Layout {
            narrow: false,
            big,
            text,
            small,
            noise,
            data,
            arrow,
            cards,
            verdict,
            tabs: None,
            forth: fits.then_some(forth),
            back: fits.then_some(back),
            round: fits.then_some(round),
            curve: fits.then_some(curve),
        }
    }

    fn narrow(view: Vec2, inset: f32, lower: Lower) -> Layout {
        let (big, text, small) = (12.5, 11.0, 10.0);
        let side = 12.0;
        let title = big + 9.0;
        let top = inset.max(0.0) + 4.0 + title;
        let w = (view.x - 2.0 * side).max(10.0);
        // the noise with the cards beside it, the arrow, the data, the verdict, the
        // tabs, and the pair at the foot
        let verdict_h = 2.0 * (text + 5.0) + 2.0;
        let tabs_h = 26.0;
        let fixed = 26.0 + title + verdict_h + 8.0 + tabs_h + 8.0 + title + small + 10.0;
        let room = (view.y - top - fixed).max(60.0);
        let low = (room * 0.28).min((w - 18.0) * 0.5).max(30.0);
        let noise_side = (room * 0.24).min(w * 0.42).max(30.0);
        let data_h = (room - low - noise_side).min(w * 1.1).max(30.0);
        let noise = Rect::new(Vec2::new(side, top), Vec2::splat(noise_side));
        let data = Rect::new(Vec2::new(side, noise.max.y + 26.0 + title), Vec2::new(w, data_h));
        let arrow = (Vec2::new(noise.mid().x, noise.max.y + 5.0), Vec2::new(noise.mid().x, data.min.y - title + 1.0));
        let cards_x = noise.max.x + 14.0;
        let card_h = ((noise.max.y - (top - title) - 8.0) * 0.5).max(20.0);
        let card = |k: f32| Rect::new(Vec2::new(cards_x, top - title + k * (card_h + 8.0)), Vec2::new(view.x - side - cards_x, card_h));
        let verdict = (Vec2::new(side, data.max.y + 6.0), w);
        let tabs_top = data.max.y + verdict_h + 8.0;
        let tabs = (Vec2::new(side, tabs_top), Vec2::new((w - 6.0) * 0.5, tabs_h));
        let low_top = tabs_top + tabs_h + 8.0 + title;
        let left = Rect::new(Vec2::new(side, low_top), Vec2::splat(low));
        let right = Rect::new(Vec2::new(view.x - side - low, low_top), Vec2::splat(low));
        let fits = low_top + low <= view.y + 2.0 && low >= 60.0;
        let (forth, back, round, curve) = match (fits, lower) {
            (false, _) => (None, None, None, None),
            (true, Lower::Fits) => (Some(left), Some(right), None, None),
            (true, Lower::Turning) => (None, None, Some(left), Some(right)),
        };
        Layout { narrow: true, big, text, small, noise, data, arrow, cards: [card(0.0), card(1.0)], verdict, tabs: fits.then_some(tabs), forth, back, round, curve }
    }
}

/// From a panel's plane to the screen: the point `centre` of the plane at the middle of
/// `rect`, `ppu` logical pixels to a unit, y up.
#[derive(Clone, Copy, Debug)]
struct Plot {
    rect: Rect,
    centre: DVec2,
    ppu: f64,
}

impl Plot {
    fn at(&self, p: DVec2) -> Vec2 {
        let mid = self.rect.mid();
        let d = (p - self.centre) * self.ppu;
        Vec2::new(mid.x + d.x as f32, mid.y - d.y as f32)
    }

    fn back(&self, s: Vec2) -> DVec2 {
        let d = (s - self.rect.mid()).as_dvec2();
        self.centre + DVec2::new(d.x, -d.y) / self.ppu.max(1e-9)
    }

    /// The part of the plane the panel shows.
    fn seen(&self) -> (DVec2, DVec2) {
        let half = self.rect.size().as_dvec2() * 0.5 / self.ppu.max(1e-9);
        (self.centre - half, self.centre + half)
    }
}

/// The part of the segment from `a` to `b` inside the box `lo`–`hi` (None if none).
fn clip(a: DVec2, b: DVec2, (lo, hi): (DVec2, DVec2)) -> Option<(DVec2, DVec2)> {
    let (mut t0, mut t1) = (0.0f64, 1.0f64);
    let d = b - a;
    for (p, q) in [(-d.x, a.x - lo.x), (d.x, hi.x - a.x), (-d.y, a.y - lo.y), (d.y, hi.y - a.y)] {
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
    (t0 <= t1 && t0.is_finite() && t1.is_finite()).then(|| (a + d * t0, a + d * t1))
}

// ----- drawing on the screen ------------------------------------------------------------

/// A dashed line in screen space, on top of everything.
fn hud_dashed(f: &mut Frame, a: Vec2, b: Vec2, width: f32, color: Color) {
    let (a, b) = (f.hud(a), f.hud(b));
    f.dl.lines_with(LineKind::Overlay, LineStyle { dash: 6.0, duty: 0.55, depth_bias: 0.0, ..Default::default() }).segment(a, b, width, color);
}

/// Dots in screen space, on top of the panels. A dot outside `rect` is drawn clear at
/// its edge, so that there are as many from frame to frame.
fn hud_dots(f: &mut Frame, rect: Rect, points: impl Iterator<Item = Vec2>, size: f32, color: Color) {
    let placed: Vec<(Vec3, bool)> = points.map(|p| (f.hud_at(2, p.max(rect.min).min(rect.max)), rect.contains(p) && p.is_finite())).collect();
    let batch = f.dl.points_with(PointKind::Overlay, PointStyle::default());
    for (at, seen) in placed {
        batch.push(at, size, if seen { color } else { color.a(0.0) });
    }
}

/// The line `a`–`b` of a panel's plane, as much of it as the panel shows.
fn plot_line(f: &mut Frame, plot: &Plot, a: DVec2, b: DVec2, width: f32, color: Color, dashed: bool) {
    if let Some((p, q)) = clip(a, b, plot.seen()) {
        let (p, q) = (plot.at(p), plot.at(q));
        if dashed {
            hud_dashed(f, p, q, width, color);
        } else {
            hud_line(f, p, q, width, color);
        }
    }
}

/// A panel's backing and frame.
fn backing(f: &mut Frame, r: Rect) {
    hud_panel(f, r.min, r.max);
    hud_outline(f, r.min, r.max, 1.0, pal::FAINT.a(0.7));
}

/// The first of `options` that fits in `width` logical pixels at `size`.
fn pick<'a>(f: &Frame, options: &[&'a str], width: f32, size: f32) -> &'a str {
    options.iter().copied().find(|t| f.text_size(t, size).x <= width).unwrap_or(options.last().copied().unwrap_or(""))
}

/// A button over the picture: true when clicked.
fn hud_tab(f: &mut Frame, min: Vec2, max: Vec2, text: &str, on: bool, pointer: Option<Vec2>, size: f32) -> bool {
    let over = pointer.is_some_and(|p| p.cmpge(min).all() && p.cmple(max).all());
    let fill = match (on, over) {
        (true, _) => Color::hex(0x2B3B60),
        (false, true) => Color::hex(0x2A3349),
        (false, false) => Color::hex(0x161C2B),
    };
    hud_rect(f, min, max, fill);
    if on {
        hud_line(f, Vec2::new(min.x, max.y - 1.0), Vec2::new(max.x, max.y - 1.0), 2.0, pal::LINK);
    }
    f.text_px((min + max) * 0.5, text, size, if on { pal::TEXT } else { pal::DIM }, Align2::CENTER_CENTER);
    over && f.ptr.clicked
}

/// The colours of X's and Y's roles: the cause cyan, the effect orange.
fn roles(forwards: bool) -> (Color, Color) {
    if forwards {
        (CAUSE, EFFECT)
    } else {
        (EFFECT, CAUSE)
    }
}

/// A draggable dot at a point of the screen: where the pointer has taken it, if it
/// is being dragged.
fn screen_handle(f: &mut Frame, id: u32, at: Vec2, color: Color) -> Option<Vec2> {
    let world = f.cam.pick_ground(at * f.px)?;
    let mut grip = world.truncate();
    if f.drag2(id, &mut grip, color) {
        on_screen(f, grip.extend(0.0))
    } else {
        None
    }
}

impl Lingam {
    /// The scale the noise and the data are drawn at (logical pixels to a unit), the
    /// same in both so that the shear is seen as it is.
    fn scale(&self, lay: &Layout) -> f64 {
        let q = self.reach_shown.drawn.max(0.1);
        let [sx, sy] = self.stretch_shown.map(|g| g.drawn);
        let (n, d) = (lay.noise.size().as_dvec2(), lay.data.size().as_dvec2());
        (n.x / (2.0 * q)).min(n.y / (2.0 * q)).min(d.x / (2.0 * q * sx.max(1.0))).min(d.y / (2.0 * q * sy.max(1.0))).max(1e-3)
    }

    /// The square of flat noise, ±√3 each way, as the shear `s` takes it: the sides
    /// along which only N_Y varies in Y's colour, those along which only N_X varies in X's.
    fn draw_square(&self, f: &mut Frame, plot: &Plot, s: Shear) {
        let (cx, cy) = roles(self.forwards);
        let corner = |a: f64, b: f64| sheared(s, DVec2::new(a * EDGE, b * EDGE));
        for k in [-1.0, 1.0] {
            plot_line(f, plot, corner(k, -1.0), corner(k, 1.0), 2.0, cy.a(0.95), false);
            plot_line(f, plot, corner(-1.0, k), corner(1.0, k), 2.0, cx.a(0.95), false);
        }
    }

    /// The axes of a panel, faint, with their names on plates at the ends.
    fn axes(f: &mut Frame, plot: &Plot, names: (&str, &str), colors: (Color, Color), size: f32) {
        let (lo, hi) = plot.seen();
        plot_line(f, plot, DVec2::new(lo.x, 0.0), DVec2::new(hi.x, 0.0), 1.0, pal::DIM.a(0.45), false);
        plot_line(f, plot, DVec2::new(0.0, lo.y), DVec2::new(0.0, hi.y), 1.0, pal::DIM.a(0.45), false);
        let r = plot.rect;
        let y0 = plot.at(DVec2::ZERO);
        if y0.y > r.min.y && y0.y < r.max.y - size - 8.0 {
            f.caption_px(Vec2::new(r.max.x - 4.0, y0.y + 4.0), names.0, size, colors.0, Align2::RIGHT_TOP);
        }
        if y0.x > r.min.x && y0.x < r.max.x - 30.0 {
            f.caption_px(Vec2::new(y0.x + 5.0, r.min.y + 4.0), names.1, size, colors.1, Align2::LEFT_TOP);
        }
    }

    fn draw_noise(&self, f: &mut Frame, lay: &Layout) {
        let r = lay.noise;
        backing(f, r);
        let plot = Plot { rect: r, centre: DVec2::ZERO, ppu: self.scale(lay) };
        let (cx, cy) = roles(self.forwards);
        Lingam::axes(f, &plot, ("$N_X$", "$N_Y$"), (cx, cy), lay.text);
        self.draw_square(f, &plot, [0.0, 0.0]);
        let size = if self.n > 1000 { 2.2 } else { 2.8 };
        hud_dots(f, r, self.seen.noise.iter().map(|p| plot.at(*p)), size, DOTS.a(0.8));
        let shape = match self.noise {
            Noise::Normal => "bell-shaped",
            Noise::Uniform => "flat",
            Noise::Laplace => "peaked",
            Noise::TwoHumps => "two-humped",
        };
        let what = if self.bell_shown > 0.0 && self.noise != Noise::Normal {
            format!("{shape}, blended with a bell: $w = {}$", num(self.bell_shown, 2))
        } else {
            format!("{shape} noise")
        };
        if lay.narrow {
            let long = format!("the two noises: {what}");
            let title = pick(f, &[&long, &what, "the noises"], lay.cards[0].min.x - r.min.x - 8.0, lay.big);
            f.text_px(Vec2::new(r.min.x, r.min.y - 6.0), title, lay.big, pal::TEXT, Align2::LEFT_BOTTOM);
        } else {
            let title = pick(f, &["the two noises, drawn independently", "the two noises", "noises"], r.size().x + 60.0, lay.big);
            f.text_px(Vec2::new(r.min.x, r.min.y - 6.0), title, lay.big, pal::TEXT, Align2::LEFT_BOTTOM);
            f.text_px(Vec2::new(r.min.x, r.max.y + 5.0), what, lay.small, pal::DIM, Align2::LEFT_TOP);
        }
    }

    fn draw_arrow(&self, f: &mut Frame, lay: &Layout) {
        let (a, b) = lay.arrow;
        hud_arrow(f, a, b, 2.0, pal::TEXT.a(0.85));
        let (cx, cy) = roles(self.forwards);
        let (cause, effect, c_col, e_col) = if self.forwards { ("X", "Y", cx, cy) } else { ("Y", "X", cy, cx) };
        let b_now = if self.forwards { self.shear[0] } else { self.shear[1] };
        let shear = format!("shear by $b = {}$", num(b_now, 2));
        let made = format!(r"made by ${} \to {}$", tint(c_col, cause), tint(e_col, effect));
        if lay.narrow {
            // the arrow points down: the words beside it
            let at = Vec2::new(a.x + 12.0, (a.y + b.y) * 0.5);
            f.text_px(at, &shear, lay.text, pal::TEXT, Align2::LEFT_CENTER);
            let w = f.text_size(&shear, lay.text).x;
            f.text_px(at + Vec2::new(w + 16.0, 0.0), &made, lay.text, TRUTH, Align2::LEFT_CENTER);
            return;
        }
        let moves = if self.forwards { "each point moves up by $bX$" } else { "each point moves across by $bY$" };
        let cause_eq = format!("${} = {}$", tint(c_col, cause), tint(c_col, &format!("N_{cause}")));
        let effect_eq = format!(r"${} = b\,{} + {}$", tint(e_col, effect), tint(c_col, cause), tint(e_col, &format!("N_{effect}")));
        let mid = (a + b) * 0.5;
        f.text_px(mid - Vec2::new(0.0, 8.0), &shear, lay.text, pal::TEXT, Align2::CENTER_BOTTOM);
        f.text_px(mid - Vec2::new(0.0, 8.0 + lay.text + 14.0), &made, lay.text, TRUTH, Align2::CENTER_BOTTOM);
        f.text_px(mid + Vec2::new(0.0, 8.0), moves, lay.small, pal::DIM, Align2::CENTER_TOP);
        let eq_top = mid.y + 8.0 + lay.small + 16.0;
        f.text_px(Vec2::new(mid.x, eq_top), &cause_eq, lay.text, pal::TEXT, Align2::CENTER_TOP);
        f.text_px(Vec2::new(mid.x, eq_top + lay.text + 10.0), &effect_eq, lay.text, pal::TEXT, Align2::CENTER_TOP);
    }

    fn draw_data(&mut self, f: &mut Frame, lay: &Layout) {
        let r = lay.data;
        backing(f, r);
        let plot = Plot { rect: r, centre: DVec2::ZERO, ppu: self.scale(lay) };
        Lingam::axes(f, &plot, ("$X$", "$Y$"), (pal::TEXT, pal::TEXT), lay.text);
        self.draw_square(f, &plot, self.shear);
        let size = if self.n > 1000 { 2.2 } else { 2.8 };
        hud_dots(f, r, self.seen.data.iter().map(|p| plot.at(*p)), size, DOTS.a(0.8));
        // the sides found from the points alone: each noise's line through the middle
        if let (Some(t), Some(a)) = (self.seen.turn.as_ref(), self.found.map(|g| g.drawn)) {
            let alpha = match self.seen.verdict {
                Verdict::Round | Verdict::Nothing => 0.3,
                _ => 0.95,
            };
            // (the drawn angle is a whole number q of quarter turns from the best turn,
            // give or take what is left of its glide; a quarter turn makes the first
            // line the second, and the second the first one reversed)
            let q = ((a - t.best) / FRAC_PI_2).round().rem_euclid(4.0) as usize;
            let [s0, s1] = self.seen.span;
            let flip = |(lo, hi): (f64, f64)| (-hi, -lo);
            let spans = match q {
                0 => [s0, s1],
                1 => [s1, flip(s0)],
                2 => [flip(s0), flip(s1)],
                _ => [flip(s1), s0],
            };
            for (m, (lo, hi)) in t.mixing_at(a).into_iter().zip(spans) {
                plot_line(f, &plot, t.mean + m * lo * 1.15, t.mean + m * hi * 1.15, 1.4, FOUND.a(alpha), true);
            }
        }
        // the handle: where the shear takes the end of the cause's middle line
        let tip = sheared(self.shear, if self.forwards { DVec2::new(EDGE, 0.0) } else { DVec2::new(0.0, EDGE) });
        let at = plot.at(tip);
        if r.contains(at) {
            if let Some(to) = screen_handle(f, 2, at, CAUSE) {
                let p = plot.back(to);
                let b = if self.forwards { p.y / EDGE } else { p.x / EDGE };
                self.set_b(b);
            }
        }
        let title = pick(f, &["the data $(X, Y)$: all that an observer sees", "the data $(X, Y)$"], r.size().x, lay.big);
        f.text_px(Vec2::new(r.min.x, r.min.y - 6.0), title, lay.big, pal::TEXT, Align2::LEFT_BOTTOM);
        self.data_key(f, lay);
    }

    /// What the lines over the data are, in the corner the cloud leaves free.
    fn data_key(&self, f: &mut Frame, lay: &Layout) {
        let r = lay.data;
        let size = lay.small;
        let (own, other) = if self.forwards { ("$N_Y$", "$N_X$") } else { ("$N_X$", "$N_Y$") };
        let long = [
            (EFFECT, format!("only {own}, the effect's own noise, varies along these"), false),
            (CAUSE, format!("only {other}, the cause's noise, varies along these"), false),
            (FOUND, "the two noises' lines, found from the points alone".to_string(), true),
        ];
        let short = [
            (EFFECT, format!("only {own} varies along these"), false),
            (CAUSE, format!("only {other} varies along these"), false),
            (FOUND, "found from the points alone".to_string(), true),
        ];
        let wide = |rows: &[(Color, String, bool)], f: &Frame| rows.iter().map(|(_, t, _)| f.text_size(t, size).x).fold(0.0, f32::max) + 34.0;
        let rows = if wide(&long, f) + 16.0 <= r.size().x * 0.62 { long } else { short };
        let width = wide(&rows, f);
        let row = size + 8.0;
        let height = rows.len() as f32 * row + 8.0;
        if width + 16.0 > r.size().x * 0.62 || height + 16.0 > r.size().y * 0.3 {
            return;
        }
        let x = r.min.x + 8.0 + (r.size().x - width - 16.0) * self.key_side.drawn as f32;
        let at = Vec2::new(x, r.min.y + 8.0);
        hud_rect_at(f, 1, at, at + Vec2::new(width, height), Color::hex(0x080B14).a(0.85));
        for (k, (color, text, dashed)) in rows.iter().enumerate() {
            let y = at.y + 4.0 + (k as f32 + 0.5) * row;
            let (a, b) = (Vec2::new(at.x + 8.0, y), Vec2::new(at.x + 24.0, y));
            if *dashed {
                hud_dashed(f, a, b, 1.4, *color);
            } else {
                hud_line(f, a, b, 2.0, *color);
            }
            f.text_px(Vec2::new(at.x + 30.0, y), text.as_str(), size, pal::TEXT, Align2::LEFT_CENTER);
        }
    }

    fn draw_verdict(&self, f: &mut Frame, lay: &Layout) {
        let (at, width) = lay.verdict;
        let words = match self.seen.verdict {
            Verdict::Nothing => "too few points to say anything.",
            Verdict::Round => "the shape could be a bell's: cannot tell.",
            Verdict::Apart => "sides upright and level: no arrow that these points can show.",
            Verdict::Forwards => r"an upright side, so $X \to Y$.",
            Verdict::Backwards => r"a level side, so $Y \to X$.",
        };
        let long = format!("From the points alone: {words}");
        let line = pick(f, &[&long, words], width, lay.text);
        f.text_px(at, line, lay.text, pal::TEXT, Align2::LEFT_TOP);
        if let Some((up, level)) = self.seen.tilts() {
            let long = format!("the found sides: one {}° from upright, one {}° from level", num(up, 1), num(level, 1));
            let short = format!("found: {}° from upright, {}° from level", num(up, 1), num(level, 1));
            let detail = pick(f, &[&long, &short], width, lay.small);
            f.text_px(at + Vec2::new(0.0, lay.text + 6.0), detail, lay.small, pal::DIM, Align2::LEFT_TOP);
        }
    }

    fn draw_cards(&self, f: &mut Frame, lay: &Layout) {
        for (k, r) in lay.cards.iter().enumerate() {
            if r.size().y < 46.0 || r.size().x < 90.0 {
                continue;
            }
            let forwards = k == 0;
            let fits = match self.seen.verdict {
                Verdict::Forwards => Some(forwards),
                Verdict::Backwards => Some(!forwards),
                _ => None,
            };
            hud_panel(f, r.min, r.max);
            let lit = fits == Some(true);
            hud_outline(f, r.min, r.max, if lit { 1.8 } else { 1.0 }, if lit { pal::TEXT } else { pal::FAINT.a(0.7) });
            // the model as a graph
            let (cx, cy) = roles(forwards);
            let radius = (r.size().y * 0.12).clamp(9.0, 13.0);
            let y = r.min.y + radius + 9.0;
            let x0 = r.min.x + radius + 10.0;
            let nodes = [Node { at: Vec2::new(x0, y), name: "$X$", color: cx }, Node { at: Vec2::new(x0 + 3.6 * radius + 10.0, y), name: "$Y$", color: cy }];
            let link = if forwards { (0, 1, Link::Arrow, pal::TEXT, 1.6) } else { (1, 0, Link::Arrow, pal::TEXT, 1.6) };
            hud_graph(f, &nodes, &[link], radius);
            // the sides it predicts, as a small parallelogram: the effect's own noise
            // along the sides that stay upright (or level)
            let half = (r.size().y * 0.24).min(r.size().x * 0.11).min(28.0);
            let mid = Vec2::new(r.max.x - 1.7 * half - 12.0, r.mid().y);
            let lean = 0.6;
            let corners = if forwards {
                [Vec2::new(-1.0, -1.0 - lean), Vec2::new(1.0, -1.0 + lean), Vec2::new(1.0, 1.0 + lean), Vec2::new(-1.0, 1.0 - lean)]
            } else {
                [Vec2::new(-1.0 - lean, -1.0), Vec2::new(1.0 - lean, -1.0), Vec2::new(1.0 + lean, 1.0), Vec2::new(-1.0 + lean, 1.0)]
            }
            .map(|c| mid + Vec2::new(c.x, -c.y) * half);
            for i in 0..4 {
                let (a, b) = (corners[i], corners[(i + 1) % 4]);
                let own = if forwards { (a.x - b.x).abs() < 0.5 } else { (a.y - b.y).abs() < 0.5 };
                hud_line(f, a, b, 1.8, if own { EFFECT } else { CAUSE });
            }
            let says = if forwards { "predicts two sides upright" } else { "predicts two sides level" };
            let status = match (fits, self.seen.verdict) {
                (Some(true), _) => ("fits the points", pal::TEXT),
                (Some(false), _) => ("does not fit them", pal::DIM),
                (None, Verdict::Round) => ("cannot tell", pal::DIM),
                (None, Verdict::Apart) => ("fits, with $b$ near 0", pal::DIM),
                _ => ("", pal::DIM),
            };
            let (size, big) = if lay.narrow { (lay.small, lay.text) } else { (lay.text, lay.big) };
            let short = if forwards { "sides upright" } else { "sides level" };
            // the words under the graph, or (in a card too shallow for that) beside it
            let under = r.min.y + 2.0 * radius + 18.0;
            let beside = x0 + 4.6 * radius + 24.0;
            let place = if under + size + big + 8.0 <= r.max.y {
                Some((Vec2::new(r.min.x + 10.0, under), r.size().x - 3.4 * half - 30.0))
            } else {
                let room = r.max.x - 3.4 * half - 20.0 - beside;
                (room >= 80.0 && size + big + 7.0 <= r.size().y - 8.0).then(|| (Vec2::new(beside, r.mid().y - 0.5 * (size + big + 7.0)), room))
            };
            if let Some((at, room)) = place {
                let says = pick(f, &[says, short], room, size);
                f.text_px(at, says, size, pal::DIM, Align2::LEFT_TOP);
                f.text_px(at + Vec2::new(0.0, size + 7.0), status.0, big, status.1, Align2::LEFT_TOP);
            }
        }
    }

    /// What a straight-line fit leaves over, against its input.
    fn draw_fit(&self, f: &mut Frame, lay: &Layout, r: Rect, forth: bool) {
        backing(f, r);
        let input = self.seen.inputs(forth);
        let (left, spread) = if forth { (&self.seen.forth_left, &self.seen.forth_spread) } else { (&self.seen.back_left, &self.seen.back_spread) };
        // a scale of its own, the same both ways: room for the input and what is left
        let reach = self.seen.fit_reach[usize::from(!forth)];
        let ppu = (r.size().x.min(r.size().y) as f64 * 0.5 / reach).max(1e-3);
        let plot = Plot { rect: r, centre: DVec2::ZERO, ppu };
        let (input_name, left_name) = if forth { ("$X$", r"$Y - \hat b X$") } else { ("$Y$", r"$X - \hat c Y$") };
        Lingam::axes(f, &plot, (input_name, left_name), (pal::TEXT, LEFT), lay.small);
        hud_dots(f, r, input.iter().zip(left.iter()).map(|(i, l)| plot.at(DVec2::new(*i, *l))), 2.4, DOTS.a(0.75));
        if spread.len() >= 2 {
            for sign in [-2.0, 2.0] {
                let pts: Vec<Vec2> = spread.iter().map(|(at, sd)| plot.at(DVec2::new(*at, sign * sd))).map(|p| p.max(r.min).min(r.max)).collect();
                hud_path(f, pts.into_iter(), 2.0, LEFT);
            }
        }
        let title = if forth {
            pick(f, &[r"fit $Y$ from $X$: what is left, against $X$", r"$Y$ from $X$: what is left", r"$Y$ from $X$"], r.size().x + 8.0, lay.text)
        } else {
            pick(f, &[r"fit $X$ from $Y$: what is left, against $Y$", r"$X$ from $Y$: what is left", r"$X$ from $Y$"], r.size().x + 8.0, lay.text)
        };
        f.text_px(Vec2::new(r.min.x, r.min.y - 6.0), title, lay.text, pal::TEXT, Align2::LEFT_BOTTOM);
        if let Some(changes) = self.seen.spread_changes(forth) {
            let words = match (changes, forth) {
                (false, true) => [r"the same spread all along $X$", "the same spread all along"],
                (false, false) => [r"the same spread all along $Y$", "the same spread all along"],
                (true, true) => [r"its spread changes along $X$", "its spread changes"],
                (true, false) => [r"its spread changes along $Y$", "its spread changes"],
            };
            let line = pick(f, &words, r.size().x + 8.0, lay.small);
            f.text_px(Vec2::new(r.min.x, r.max.y + 5.0), line, lay.small, if changes { LEFT } else { pal::DIM }, Align2::LEFT_TOP);
        }
    }

    /// The round cloud and the line turned through it.
    fn draw_round(&mut self, f: &mut Frame, lay: &Layout, r: Rect) {
        backing(f, r);
        let ppu = (r.size().x.min(r.size().y) as f64 * 0.5 / 3.0).max(1e-3);
        let plot = Plot { rect: r, centre: DVec2::ZERO, ppu };
        hud_dots(f, r, self.seen.round.iter().map(|p| plot.at(*p)), 2.4, DOTS.a(0.75));
        let title = pick(f, &["rescaled to the same spread every way", "the same spread every way", "rescaled"], r.size().x + 8.0, lay.text);
        f.text_px(Vec2::new(r.min.x, r.min.y - 6.0), title, lay.text, pal::TEXT, Align2::LEFT_BOTTOM);
        let hint = pick(f, &["drag the yellow dot to turn the line", "drag the yellow dot"], r.size().x + 8.0, lay.small);
        f.text_px(Vec2::new(r.min.x, r.max.y + 5.0), hint, lay.small, pal::DIM, Align2::LEFT_TOP);
        let Some(a) = self.turn_drawn() else { return };
        let (u, v) = (DVec2::new(a.cos(), a.sin()), DVec2::new(-a.sin(), a.cos()));
        plot_line(f, &plot, -u * 10.0, u * 10.0, 2.0, MEASURE.a(0.95), false);
        plot_line(f, &plot, -v * 10.0, v * 10.0, 1.4, MEASURE.a(0.7), false);
        // the handle: the end of the line, to turn it by hand
        let tip = plot.at(u * 2.5);
        if let Some(to) = screen_handle(f, 1, tip, MEASURE) {
            let d = plot.back(to);
            // (the nearest way round from where it was: the handle stays on its end of
            // the line, under the pointer, all the way round)
            let angle = a + (d.y.atan2(d.x) - a + PI64).rem_euclid(2.0 * PI64) - PI64;
            if angle.is_finite() && d.length() > 1e-6 {
                self.turn_at = TurnAt::Held(angle);
                self.turn = Some(Glide::at(angle));
            }
        }
    }

    /// How far from normal the shadow is, at every angle of the line.
    fn draw_curve(&self, f: &mut Frame, lay: &Layout, r: Rect) {
        backing(f, r);
        let curve = &self.seen.curve;
        let top = curve.iter().fold(1.5f64, |m, v| m.max(*v)) * 1.15;
        let (x0, x1) = (r.min.x + 12.0, r.max.x - 12.0);
        let (y0, y1) = (r.max.y - lay.small - 10.0, r.min.y + 10.0);
        let x_of = |deg: f64| x0 + (x1 - x0) * (deg.max(-360.0).min(540.0) / 180.0) as f32;
        let y_of = |k: f64| y0 + (y1 - y0) * (k / top).max(0.0).min(1.0) as f32;
        hud_line(f, Vec2::new(x0, y0), Vec2::new(x1, y0), 1.0, pal::DIM.a(0.8));
        for deg in [0.0, 90.0, 180.0] {
            f.text_px(Vec2::new(x_of(deg), y0 + 3.0), format!("{deg:.0}°"), lay.small, pal::DIM, Align2::CENTER_TOP);
        }
        f.text_px(Vec2::new(x0, y0 + 3.0 + lay.small * 0.5), "", lay.small, pal::DIM, Align2::LEFT_TOP);
        // the gate: how far from normal a bell's points get, one time in 40
        if let Some((_, gate)) = self.seen.score() {
            let y = y_of(gate);
            hud_dashed(f, Vec2::new(x0, y), Vec2::new(x1, y), 1.0, pal::DIM);
        }
        // (while the line is turned once round, the curve is drawn as far as it has
        // come: every point not reached yet is drawn where it has got to, so that the
        // curve has as many points from frame to frame)
        let reached = self.reached();
        if curve.len() >= 2 {
            let at = |deg: f64| {
                let i = (deg.floor().max(0.0) as usize).min(curve.len() - 2);
                let t = (deg - i as f64).max(0.0).min(1.0);
                Vec2::new(x_of(deg), y_of(curve[i] + (curve[i + 1] - curve[i]) * t))
            };
            let pts: Vec<Vec2> = (0..curve.len()).map(|d| at(d as f64)).collect();
            hud_path(f, pts.iter().copied(), 1.2, MEASURE.a(0.25));
            hud_path(f, (0..curve.len()).map(|d| at((d as f64).min(reached))), 2.2, MEASURE);
        }
        // the two lines of the turn, on the curve: each drawn at its angle and at that
        // less a half turn and a whole one (the same line), the one between 0° and
        // 180° seen, so that a marker leaving at one end comes back at the other
        // without leaping across
        if let (Some(a), Some(t)) = (self.turn_drawn(), self.seen.turn.as_ref()) {
            let deg = a.to_degrees().rem_euclid(360.0);
            for line in [deg, deg + 90.0] {
                for copy in [line, line - 180.0, line - 360.0] {
                    let k = t.kurtosis(copy.to_radians()).abs();
                    let seen = if (-1e-9..=180.0 + 1e-9).contains(&copy) { 1.0 } else { 0.0 };
                    hud_line(f, Vec2::new(x_of(copy), y0), Vec2::new(x_of(copy), y_of(k)), 1.0, MEASURE.a(0.5 * seen));
                    hud_circle(f, Vec2::new(x_of(copy), y_of(k)), 3.5, 1.8, MEASURE.a(seen));
                }
            }
        }
        let title = pick(f, &["how far from normal the shadow on the line is", "how far from normal", "|kurtosis|"], r.size().x + 8.0, lay.text);
        f.text_px(Vec2::new(r.min.x, r.min.y - 6.0), title, lay.text, pal::TEXT, Align2::LEFT_BOTTOM);
        let under = pick(f, &["dots: the two lines of the turn.  dashed: what the dots must top", "dashed: what the dots must top", "dashed: the gate"], r.size().x + 8.0, lay.small);
        f.text_px(Vec2::new(r.min.x, r.max.y + 5.0), under, lay.small, pal::DIM, Align2::LEFT_TOP);
    }
}

impl Scene for Lingam {
    fn view(&self) -> View {
        View::Flat { center: Vec2::ZERO, half_height: 1.0 }
    }

    fn settings(&self, s: &mut ViewSettings) {
        s.grid = false;
        s.bloom = s.bloom.min(0.2);
    }

    /// Moving while the shear plays, the points glide, the bell is blended in or the
    /// line is turned.
    fn animated(&self) -> bool {
        self.moving()
    }

    fn pointer_hint(&self) -> Option<&'static str> {
        Some("Drag the cyan dot to change $b$, and the yellow dot to turn the line through the round cloud.    On a narrow screen, the tabs choose what the foot of the picture shows.")
    }

    fn ui(&mut self, ui: &mut egui::Ui, _: &Gfx) {
        ui::note(ui, r"Suppose $X$ causes $Y$ along a straight line: $Y = bX + N_Y$, where $X = N_X$ and the two noises are independent of each other. Then the cloud of points $(X, Y)$ is the cloud of the noises $(N_X, N_Y)$ sheared: every point moved up by $b$ times its $X$. Moving straight up keeps upright lines upright. So if the noises are flat and fill a square, the data fill a parallelogram with two sides still upright: the fingerprint of $X \to Y$. Had $Y$ caused $X$, two sides would be level. Bell-shaped noise fills a round blob, the data an ellipse, and an ellipse has no sides: it fits either arrow.");
        ui::note(ui, "Under the story, two ways of reading the fingerprint from the points alone: a straight line fitted each way, and what it leaves over; and a turn that finds the noises' directions without being told them. Both rest on what the model assumes: straight-line effects, noises independent of each other (so no hidden common cause), and noises that are not bell-shaped.");

        ui::section(ui, "Things to try");
        for (k, (label, look_for, _)) in TRIALS.iter().enumerate() {
            if ui::experiment(ui, label, look_for) {
                self.set_up(k);
            }
        }

        ui::section(ui, "The model");
        let mut way = usize::from(!self.forwards);
        if ui::choice(ui, &[r"$X$ causes $Y$", r"$Y$ causes $X$"], &mut way) {
            self.set_forwards(way == 0);
        }
        let mut b = self.b as f32;
        if ui::slider(ui, "how strongly, $b$", &mut b, -B_MAX as f32..=B_MAX as f32) {
            self.set_b(b as f64);
        }
        ui::note(ui, "$b$: how much the effect moves for each unit of the cause. At 0 neither moves the other. The cyan dot in the picture sets it too.");
        ui::section(ui, "The noise");
        let names: Vec<&str> = Noise::ALL.iter().map(|n| n.name()).collect();
        let mut shape = Noise::ALL.iter().position(|n| *n == self.noise).unwrap_or(0);
        if ui::choice(ui, &names, &mut shape) {
            self.change_draws(Noise::ALL[shape], self.seed);
        }
        let mut w = self.bell as f32;
        if ui::slider(ui, "how bell-shaped, $w$", &mut w, 0.0..=1.0) {
            self.set_bell(w as f64);
        }
        ui::note(ui, r"Each noise is $\sqrt{1 - w}$ times a draw of the shape plus $\sqrt{w}$ times a draw of a bell: mean 0 and variance 1 always, the shape at $w = 0$, a bell at $w = 1$.");
        ui::section(ui, "The sample");
        let mut n = self.n;
        if ui::slider_int_log(ui, "points", &mut n, N_MIN..=N_MAX) {
            self.n = n.clamp(N_MIN, N_MAX);
        }
        if ui.button("new sample").clicked() {
            self.change_draws(self.noise, self.seed.wrapping_add(1));
        }
        ui::note(ui, "With fewer than about 150 points of flat noise ($b = 1$) the shape cannot be told from a bell's. Peaked noise is told with fewer, but wrongly more often: with 50 to 150 points, about 1 time in 25.");
        ui::section(ui, "Under the picture");
        let mut lower = usize::from(self.lower == Lower::Turning);
        if ui::choice(ui, &["the fits both ways", "the turn"], &mut lower) {
            self.lower = if lower == 0 { Lower::Fits } else { Lower::Turning };
        }
        ui::note(ui, "A narrow screen has room for one of the two at a time; a wide one shows both.");
        if ui.button("put the line on the best turn").clicked() {
            self.turn_at = TurnAt::Best;
        }

        ui::section(ui, "Reading the picture");
        ui::key(
            ui,
            &[
                (DOTS, "dots: the points, each one unit (a person, a day): on the left at its two noises, on the right at its $(X, Y)$"),
                (CAUSE, "cyan: the cause, and the sides along which only the cause's noise varies"),
                (EFFECT, "orange: the effect, and the sides along which only the effect's own noise varies: upright when $Y$ is the effect"),
                (FOUND, "white, dashed: the directions of the two noises, found from the points alone"),
                (LEFT, "purple: the axis of what a straight-line fit leaves over, and a band twice its spread either side, at each value of the input"),
                (MEASURE, "yellow: the line turned through the round cloud, and how far from normal its shadow is"),
                (pal::DIM, "grey dashed, on the curve: the yellow dots of the best turn, taken together, must be above it for the cloud to be told from a bell's (bell-shaped noise gets there 1 time in 40)"),
                (pal::TEXT, "a white frame round a model: the points fit it"),
                (TRUTH, "green: how the data were really made"),
            ],
        );
        ui::section(ui, "Words");
        ui::term(ui, "Independent:", "knowing one says nothing about the other: not its average, not its spread, nothing.");
        ui::term(ui, "Shear:", r"a map that slides every point along one direction, in proportion to how far it is along another: here $(x, y) \mapsto (x, y + bx)$.");
        ui::term(ui, "Normal, bell-shaped, Gaussian:", "three names for the spread of values the bell curve describes.");
        ui::term(ui, "Kurtosis:", "how far a spread of values is from a bell's in its tails and peak: 0 for a bell, negative for flatter ones (−1.2 for flat noise), positive for peaked ones with long tails (3 for Laplace noise). The yellow curve is its size, whichever its sign.");
        ui::term(ui, "Residual:", "what a fit leaves over: each value less the fitted line's value.");
        ui::term(ui, "LiNGAM:", "a linear, non-Gaussian, acyclic model: each variable a straight-line function of its causes plus a noise of its own; the noises independent of each other (so no hidden common cause) and not bell-shaped; no loops. Shimizu and others (2006) showed that the arrows of such a model can be found from the data alone, given enough of it.");
        ui::term(ui, "Whitening:", "rescaling a cloud of points so that it has the same spread along every line through its middle.");
        ui::term(ui, "Independent component analysis:", "finding, from mixtures alone, the independent sources that were mixed: here by whitening the cloud and turning it until its coordinates are as far from normal as they get.");

        ui::section(ui, "Numbers");
        let seen = &self.seen;
        ui::readout(ui, r"$\hat b$, fitting $Y$ from $X$", format!("${}$", num(seen.forth.0, 3)));
        ui::readout(ui, r"$\hat c$, fitting $X$ from $Y$", format!("${}$", num(seen.back.0, 3)));
        ui::readout(ui, r"what is left, with $X$", format!("${}$", num(seen.left_correlation[0], 3)));
        ui::readout(ui, r"what is left, with $Y$", format!("${}$", num(seen.left_correlation[1], 3)));
        ui::readout(ui, r"its square, with $X^2$", format!("${}$", num(seen.square_correlation[0], 3)));
        ui::readout(ui, r"its square, with $Y^2$", format!("${}$", num(seen.square_correlation[1], 3)));
        ui::note(ui, r"Correlations: of what the fit of $Y$ from $X$ leaves with $X$, of what the fit of $X$ from $Y$ leaves with $Y$, and of their squares. The first two are 0 for any straight-line fit. The squares' are 0 too when what is left is independent of the input; beyond $3/\sqrt{n}$ either way, its spread changes along the input.");
        if let Some((score, gate)) = seen.score() {
            ui::readout(ui, "the best turn, from normal", format!("${}$", num(score, 3)));
            ui::readout(ui, "a bell's, 1 time in 40", format!("${}$", num(gate, 3)));
            ui::note(ui, "How far the best turn's two coordinates are from normal, together (the root mean square of their kurtoses), and how far bell-shaped noise with this many points gets 1 time in 40: the grey dashed line.");
        }
        ui::see_also(ui, &[("direction", "cause or effect, by what a fit leaves over"), ("discover", "more than two variables"), ("clt", "why mixtures are nearer to normal"), ("pca", "the directions of most spread")]);
    }

    fn frame(&mut self, f: &mut Frame) {
        self.step(f.dt as f64, f.real_dt as f64, f.handles.dragging());
        let view = f.view_size();
        if view.x < 120.0 || view.y < 120.0 {
            return;
        }
        let lay = Layout::of(view, f.top_inset, self.lower);
        let pointer = f.ptr.pos.map(|p| p / f.px);
        self.draw_noise(f, &lay);
        self.draw_arrow(f, &lay);
        self.draw_data(f, &lay);
        self.draw_verdict(f, &lay);
        self.draw_cards(f, &lay);
        if let Some(r) = lay.forth {
            self.draw_fit(f, &lay, r, true);
        }
        if let Some(r) = lay.back {
            self.draw_fit(f, &lay, r, false);
        }
        if let Some(r) = lay.round {
            self.draw_round(f, &lay, r);
        }
        if let Some(r) = lay.curve {
            self.draw_curve(f, &lay, r);
        }
        if let Some((at, size)) = lay.tabs {
            if pointer.is_some_and(|p| Rect::new(at, Vec2::new(2.0 * size.x + 6.0, size.y)).contains(p)) {
                f.handles.claim_clicks();
            }
            for (k, (name, which)) in [("the fits both ways", Lower::Fits), ("the turn", Lower::Turning)].into_iter().enumerate() {
                let min = at + Vec2::new(k as f32 * (size.x + 6.0), 0.0);
                if hud_tab(f, min, min + size, name, self.lower == which, pointer, lay.text) {
                    self.lower = which;
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const FRAME: f64 = 1.0 / 60.0;

    /// An experiment as its button sets it up, run to rest at 60 frames a second;
    /// `watch` sees the scene after every frame.
    fn run(k: usize, mut watch: impl FnMut(&Lingam)) -> Lingam {
        let mut s = Lingam::create();
        s.set_up(k);
        for _ in 0..3000 {
            s.step(FRAME, FRAME, false);
            watch(&s);
            if !s.moving() {
                return s;
            }
        }
        panic!("experiment {k} still moving after 50 seconds");
    }

    fn settled(k: usize) -> Lingam {
        run(k, |_| {})
    }

    /// A sample of the scene's model, as the scene makes it, worked out.
    fn seen_of(noise: Noise, w: f64, b: f64, forwards: bool, n: usize, seed: u64) -> Seen {
        let (shape, bell) = (draws(noise, seed), bells(seed));
        let pts: Vec<DVec2> = (0..n.min(N_MAX)).map(|i| blend(shape[i], bell[i], w)).collect();
        Seen::of(pts, shear_of(b, forwards))
    }

    /// The correlation of two samples' values.
    fn corr(a: impl Iterator<Item = f64>, b: impl Iterator<Item = f64>) -> f64 {
        let (a, b): (Vec<f64>, Vec<f64>) = (a.collect(), b.collect());
        causal::correlation(&a, &b)
    }

    /// The first experiment: the square leans as b grows, every point moving straight
    /// up, so its upright sides stay upright; the turn finds a side upright to within
    /// a degree; what the fits leave is level one way and changes the other.
    #[test]
    fn flat_noise_keeps_two_sides_upright() {
        let mut last = 0.0;
        let s = run(0, |s| {
            assert!(s.shear[1] == 0.0 && s.shear[0] >= last - 1e-12, "{:?} after {last}", s.shear);
            last = s.shear[0];
            assert!(s.seen.noise.iter().zip(&s.seen.data).all(|(n, d)| n.x == d.x), "a point moved sideways");
        });
        assert_eq!((s.shear, s.noise, s.n), ([1.0, 0.0], Noise::Uniform, 600));
        let (up, level) = s.seen.tilts().unwrap();
        assert!(s.seen.verdict == Verdict::Forwards && up < 1.0 && level > 40.0, "{:?}: {up}° from upright, {level}° from level", s.seen.verdict);
        assert_eq!((s.seen.spread_changes(true), s.seen.spread_changes(false)), (Some(false), Some(true)));
        // the found sides are the true ones: upright, and along (1, b)
        let t = s.seen.turn.as_ref().unwrap();
        let sides = t.sides_at(t.best);
        let slanted = DVec2::new(1.0, 1.0).normalize();
        assert!(sides.iter().any(|v| v.perp_dot(slanted).abs() < (2f64).to_radians().sin()), "{sides:?}");
    }

    /// Turned round: every point moves sideways, the level sides stay level, and the
    /// verdict turns with the arrow.
    #[test]
    fn turned_round_the_level_sides_stay() {
        let s = run(1, |s| {
            assert!(s.seen.noise.iter().zip(&s.seen.data).all(|(n, d)| n.y == d.y) || s.shear[0] != 0.0, "a point moved up while the arrow was Y to X");
        });
        assert_eq!(s.shear, [0.0, 1.0]);
        let (up, level) = s.seen.tilts().unwrap();
        assert!(s.seen.verdict == Verdict::Backwards && level < 3.0 && up > 40.0, "{:?}: {up}° {level}°", s.seen.verdict);
        assert!(s.seen.noise.iter().zip(&s.seen.data).all(|(n, d)| n.y == d.y));
    }

    /// Blending in a bell: the verdict holds until w reaches 0.33 and is "cannot tell"
    /// from then on; at the end, an ellipse, which X → Y with b = 1 and Y → X with
    /// c = 0.5 (noises of variance 2 and 0.5) make alike.
    #[test]
    fn a_bell_hides_the_fingerprint() {
        let (mut last_told, mut first_round) = (None, None);
        let s = run(2, |s| {
            // (from when the blend begins: the shear before it passes through b = 0)
            if s.bell_shown == 0.0 {
                return;
            }
            match s.seen.verdict {
                Verdict::Forwards => {
                    assert!(first_round.is_none(), "told again at w = {} after cannot-tell at {first_round:?}", s.bell_shown);
                    last_told = Some(s.bell_shown);
                }
                Verdict::Round => first_round = first_round.or(Some(s.bell_shown)),
                v => panic!("{v:?} at w = {}", s.bell_shown),
            }
        });
        let (told, round) = (last_told.unwrap(), first_round.unwrap());
        println!("told until w = {told}, cannot tell from w = {round}");
        assert!(told > 0.33 && round < 0.345, "told until w = {told}, cannot tell from w = {round}");
        assert_eq!((s.bell_shown, s.seen.verdict), (1.0, Verdict::Round));
        // the fits both ways leave what looks independent of the input
        assert_eq!((s.seen.spread_changes(true), s.seen.spread_changes(false)), (Some(false), Some(false)));
        assert!((s.seen.back.0 - 0.5).abs() < 0.03, "the slope fitting X from Y: {}", s.seen.back.0);
        // the two models' covariances, exactly: X → Y with b = 1 and noises of variance 1,
        // and Y → X with c = 0.5, Y's noise of variance 2 and X's of 0.5
        let (b, c, vy, vx) = (1.0f64, 0.5f64, 2.0f64, 0.5f64);
        let forwards = [1.0, b, b * b + 1.0];
        let backwards = [c * c * vy + vx, c * vy, vy];
        assert_eq!(forwards, backwards);
        // the square's corners round off: points leave it as the bell comes in
        let outside = |w: f64| seen_of(Noise::Uniform, w, 1.0, true, N_START, SEED).noise.iter().filter(|p| p.x.abs().max(p.y.abs()) > EDGE).count();
        assert!(outside(0.0) == 0 && outside(0.1) > 0 && outside(0.1) < outside(0.5) && outside(0.5) < outside(1.0), "{} {} {} {}", outside(0.0), outside(0.1), outside(0.5), outside(1.0));
    }

    /// Peaked noise: arms instead of sides, still read right with 2000 points (1.6°
    /// from upright), less precisely with 600 of them (3.3°).
    #[test]
    fn peaked_noise_has_arms() {
        let s = settled(3);
        assert_eq!((s.noise, s.n), (Noise::Laplace, 2000));
        let (up, level) = s.seen.tilts().unwrap();
        assert!(s.seen.verdict == Verdict::Forwards && (up - 1.6).abs() < 0.1 && level > 40.0, "{:?} {up}° {level}°", s.seen.verdict);
        let fewer = seen_of(Noise::Laplace, 0.0, 1.0, true, 600, SEED);
        let (up600, _) = fewer.tilts().unwrap();
        assert!(fewer.verdict == Verdict::Forwards && (up600 - 3.3).abs() < 0.1, "{up600}°");
        // most values near 0, a few far out: as many within 0.5 of 0 as flat noise has
        // within 0.9, and some beyond 3, where flat noise has none
        let near = s.seen.noise.iter().filter(|p| p.x.abs() < 0.5).count();
        let far = s.seen.noise.iter().filter(|p| p.x.abs() > 3.0).count();
        assert!(near > 1000 && far > 10, "{near} near, {far} far");
    }

    /// Turning the round cloud: the noises' own cloud comes back, turned; the measure
    /// peaks at two angles a right angle apart, each where the shadow is one noise
    /// alone, and dips between; the best turn's lines are drawn through the peaks.
    #[test]
    fn turning_finds_the_noises() {
        let mut drawn = 0;
        let s = run(4, |s| {
            if let TurnAt::Sweep(t) = s.turn_at {
                if t > 0.0 {
                    drawn += 1;
                }
            }
        });
        assert!(drawn > 300, "the line was turned once round in {drawn} frames");
        assert_eq!((s.noise, s.lower, s.turn_at), (Noise::TwoHumps, Lower::Turning, TurnAt::Best));
        let t = s.seen.turn.as_ref().unwrap();
        // the found noises are the true ones (each up to its sign)
        let found: Vec<DVec2> = s.seen.data.iter().map(|p| t.unmix(*p, t.best)).collect();
        for k in 0..2 {
            let best = [0usize, 1].map(|j| corr(found.iter().map(|f| if k == 0 { f.x } else { f.y }), s.seen.noise.iter().map(|n| if j == 0 { n.x } else { n.y })).abs());
            assert!(best[0].max(best[1]) > 0.995, "found noise {k}: correlations {best:?}");
        }
        // the curve's peaks
        let c = &s.seen.curve;
        let peaks: Vec<usize> = (0..180).filter(|&d| c[d] > c[(d + 179) % 180] && c[d] >= c[(d + 1) % 180]).collect();
        assert_eq!(peaks.len(), 2, "{peaks:?}");
        let apart = (peaks[1] - peaks[0]) as f64;
        assert!((apart - 90.0).abs() <= 3.0, "{peaks:?}");
        let dip = (peaks[0]..peaks[1]).map(|d| c[d]).fold(f64::MAX, f64::min);
        assert!(dip < 0.6 * c[peaks[0]].min(c[peaks[1]]), "dip {dip} between {} and {}", c[peaks[0]], c[peaks[1]]);
        for &d in &peaks {
            let a = (d as f64).to_radians();
            let shadow: Vec<f64> = s.seen.data.iter().map(|p| t.whiten(*p).dot(DVec2::new(a.cos(), a.sin()))).collect();
            let one = [0usize, 1].map(|j| corr(shadow.iter().copied(), s.seen.noise.iter().map(|n| if j == 0 { n.x } else { n.y })).abs());
            assert!(one[0].max(one[1]) > 0.99, "at {d}°: {one:?}");
            let line = s.turn_drawn().unwrap().to_degrees();
            let off = [line, line + 90.0].map(|l| ((l - d as f64).rem_euclid(180.0)).min((d as f64 - l).rem_euclid(180.0))).into_iter().fold(f64::MAX, f64::min);
            assert!(off < 2.0, "a line of the turn is {off}° from the peak at {d}°");
        }
    }

    /// The line drawn where bell-shaped noise reaches one time in 40 is passed by it
    /// about that often, whatever the number of points (two thousand samples at each).
    #[test]
    fn a_bell_passes_the_gate_one_time_in_40() {
        for (n, seed) in [(20usize, 1u64), (50, 2), (100, 3), (300, 4), (600, 5), (2000, 6)] {
            let trials = 2000;
            let mut rng = Rng::new(seed * 7_777);
            let passed = (0..trials)
                .filter(|_| {
                    let pts: Vec<DVec2> = (0..n).map(|_| DVec2::new(Noise::Normal.sample(&mut rng), Noise::Normal.sample(&mut rng))).collect();
                    verdict(Turn::of(&pts).as_ref(), n) != Verdict::Round
                })
                .count();
            let rate = passed as f64 / trials as f64;
            assert!(rate > 1.0 / 70.0 && rate < 1.0 / 25.0, "{n} points: 1 in {:.0}", 1.0 / rate.max(1e-9));
        }
        // and the gate falls as the points grow in number
        assert!(gate(20) > gate(100) && gate(100) > gate(600) && gate(600) > gate(2000) && gate(5000).is_finite() && gate(0).is_finite());
    }

    /// The turn finds the noises' directions for any b and either arrow.
    #[test]
    fn the_turn_finds_the_true_sides() {
        for noise in [Noise::Uniform, Noise::TwoHumps, Noise::Laplace] {
            for b in [-1.5, -0.4, 0.5, 1.0, 2.0] {
                for forwards in [true, false] {
                    let s = seen_of(noise, 0.0, b, forwards, N_MAX, 5);
                    let t = s.turn.as_ref().unwrap();
                    let truth = if forwards { [DVec2::new(1.0, b), DVec2::Y] } else { [DVec2::X, DVec2::new(b, 1.0)] }.map(|v| v.normalize());
                    for v in truth {
                        let off = t.sides_at(t.best).iter().map(|f| f.perp_dot(v).abs().asin().to_degrees()).fold(f64::MAX, f64::min);
                        assert!(off < 2.5, "{noise:?} b = {b} forwards {forwards}: a true side is {off}° from the found ones");
                    }
                    assert_eq!(s.verdict, if forwards { Verdict::Forwards } else { Verdict::Backwards }, "{noise:?} b = {b} forwards {forwards}");
                }
            }
        }
    }

    /// What the fits leave over is uncorrelated with the input both ways; against the
    /// arrow its square is not (for flat noise and b = 1, −3/7 for many points), and
    /// its spread shrinks to the ends: a diamond.
    #[test]
    fn the_fits_leave_a_rectangle_one_way_and_a_diamond_the_other() {
        let s = seen_of(Noise::Uniform, 0.0, 1.0, true, N_MAX, SEED);
        let (x, y) = (s.inputs(true), s.inputs(false));
        assert!(causal::correlation(&s.forth_left, x).abs() < 1e-9 && causal::correlation(&s.back_left, y).abs() < 1e-9);
        assert!(s.left_correlation.iter().all(|c| c.abs() < 1e-9));
        let (forth, back) = (square_correlation(&s.forth_left, x), square_correlation(&s.back_left, y));
        assert_eq!([forth, back], s.square_correlation);
        assert!(forth.abs() < 3.0 / (N_MAX as f64).sqrt() && (back + 3.0 / 7.0).abs() < 0.03, "{forth} {back}");
        let spread = |v: &[(f64, f64)]| (v[0].1, v[v.len() / 2].1, v[v.len() - 1].1);
        let (a, m, z) = spread(&s.forth_spread);
        assert!((a / m - 1.0).abs() < 0.15 && (z / m - 1.0).abs() < 0.15, "level: {a} {m} {z}");
        let (a, m, z) = spread(&s.back_spread);
        assert!(a < 0.5 * m && z < 0.5 * m, "a diamond: {a} {m} {z}");
    }

    /// No arrow at all (b = 0): sides upright and level at once. A little arrow
    /// (b = 0.1) is seen with this many flat points.
    #[test]
    fn no_arrow_is_no_arrow() {
        assert_eq!(seen_of(Noise::Uniform, 0.0, 0.0, true, N_START, SEED).verdict, Verdict::Apart);
        assert_eq!(seen_of(Noise::Uniform, 0.0, 0.0, false, N_START, SEED).verdict, Verdict::Apart);
        assert_eq!(seen_of(Noise::Uniform, 0.0, 0.1, true, N_START, SEED).verdict, Verdict::Forwards);
        let mut s = Lingam::create();
        s.set_b(0.0);
        s.step(FRAME, FRAME, false);
        assert_eq!(s.seen.verdict, Verdict::Apart);
    }

    /// Few points, a cloud on a line, nothing at all: nothing panics, every number
    /// is finite, and the verdict says what has happened.
    #[test]
    fn degenerate_clouds_are_said_to_be_so() {
        assert!(Turn::of(&[]).is_none() && Turn::of(&[DVec2::ONE; 3]).is_none() && Turn::of(&[DVec2::ONE; 50]).is_none());
        let line: Vec<DVec2> = (0..50).map(|i| DVec2::new(i as f64, 2.0 * i as f64)).collect();
        assert!(Turn::of(&line).is_none());
        assert_eq!(verdict(None, 50), Verdict::Nothing);
        let empty = Seen::of(Vec::new(), [1.0, 0.0]);
        assert!(empty.verdict == Verdict::Nothing && empty.tilts().is_none() && empty.spread_changes(true).is_none());
        // twenty flat points: too few to tell the shape from a bell's
        assert_eq!(seen_of(Noise::Uniform, 0.0, 1.0, true, N_MIN, SEED).verdict, Verdict::Round);
        // the scene at every corner of its controls
        for noise in Noise::ALL {
            for w in [0.0, 0.5, 1.0] {
                for b in [-B_MAX, -0.01, 0.0, 1.0, B_MAX] {
                    for n in [N_MIN, 21, N_START, N_MAX] {
                        let s = seen_of(noise, w, b, b >= 0.0, n, 3);
                        let finite = s.data.iter().chain(&s.round).all(|p| p.is_finite())
                            && s.curve.iter().chain(&s.forth_left).chain(&s.back_left).all(|v| v.is_finite())
                            && s.forth_spread.iter().chain(&s.back_spread).all(|(a, b)| a.is_finite() && b.is_finite())
                            && s.span.iter().all(|(a, b)| a.is_finite() && b.is_finite())
                            && s.tilts().is_some_and(|(a, b)| a.is_finite() && b.is_finite());
                        assert!(finite, "{noise:?} w {w} b {b} n {n}");
                    }
                }
            }
        }
        assert!(spread_along(&[1.0; 40], &[0.5; 40], 10).is_empty() && quantile(&[], 0.5) == 0.0);
    }

    /// What the note under the number of points says: below about 150 points of flat
    /// noise the shape cannot be told from a bell's, from 200 it is told right; peaked
    /// noise is told with fewer, and with 50 to 150 points the wrong way about 1 time
    /// in 25 (four hundred samples of each).
    #[test]
    fn few_points_say_little() {
        let count = |noise: Noise, n: usize| {
            let mut c = [0usize; 3];
            for k in 0..400 {
                c[match seen_of(noise, 0.0, 1.0, true, n, 500 + k).verdict {
                    Verdict::Forwards => 0,
                    Verdict::Backwards => 1,
                    _ => 2,
                }] += 1;
            }
            c
        };
        for n in [20, 50, 100] {
            let [right, wrong, none] = count(Noise::Uniform, n);
            assert!(none >= if n == 100 { 300 } else { 395 } && wrong == 0 && right + none == 400, "flat, {n} points: {right} right, {none} cannot tell");
        }
        assert_eq!(count(Noise::Uniform, 200), [400, 0, 0]);
        let (mut wrong, mut told) = (0, 0);
        for n in [50, 100, 150] {
            let [r, w, _] = count(Noise::Laplace, n);
            (wrong, told) = (wrong + w, told + r + w);
        }
        let rate = wrong as f64 / told as f64;
        assert!(rate > 1.0 / 32.0 && rate < 1.0 / 18.0 && told > 900, "peaked, 50 to 150 points: wrong {wrong} of {told}");
        let [_, w, _] = count(Noise::Laplace, 20);
        assert!(w > 0, "peaked, 20 points: never wrong?");
    }

    /// Every experiment comes to rest, and the picture with it.
    #[test]
    fn the_scene_comes_to_rest() {
        for k in 0..TRIALS.len() {
            let mut s = settled(k);
            let before = (s.shear, s.bell_shown, s.turn, s.found, s.reach_shown, s.stretch_shown, s.key_side);
            s.step(FRAME, FRAME, false);
            assert_eq!(before, (s.shear, s.bell_shown, s.turn, s.found, s.reach_shown, s.stretch_shown, s.key_side), "experiment {k}");
        }
        // and the arrow turned round, at rest or half-way through a shear
        let mut s = Lingam::create();
        for _ in 0..30 {
            s.step(FRAME, FRAME, false);
        }
        s.set_forwards(false);
        for _ in 0..600 {
            s.step(FRAME, FRAME, false);
            assert!(s.shear[0] == 0.0 || s.shear[1] == 0.0, "{:?}", s.shear);
        }
        assert!(!s.moving() && s.shear == [0.0, 1.0]);
    }

    /// The layout keeps everything on the canvas and apart, from a desktop to a phone,
    /// and stays finite on canvases far too small.
    #[test]
    fn the_layout_fits() {
        for (view, inset) in [(Vec2::new(1075.0, 1000.0), 0.0), (Vec2::new(1300.0, 700.0), 0.0), (Vec2::new(1600.0, 1000.0), 46.0), (Vec2::new(390.0, 800.0), 46.0), (Vec2::new(580.0, 700.0), 0.0), (Vec2::new(760.0, 520.0), 0.0)] {
            for lower in [Lower::Fits, Lower::Turning] {
                let lay = Layout::of(view, inset, lower);
                let canvas = Rect { min: Vec2::new(0.0, inset), max: view };
                let all = [Some(lay.noise), Some(lay.data), Some(lay.cards[0]), Some(lay.cards[1]), lay.forth, lay.back, lay.round, lay.curve];
                for r in all.iter().flatten() {
                    assert!(canvas.contains(r.min) && canvas.contains(r.max), "{view} {lower:?}: {r:?}");
                }
                assert!(lay.forth.is_some() || lay.round.is_some(), "{view}: no room below");
                let boxes: Vec<Rect> = all.iter().flatten().copied().collect();
                for (i, a) in boxes.iter().enumerate() {
                    for b in &boxes[..i] {
                        let apart = a.max.x <= b.min.x || b.max.x <= a.min.x || a.max.y <= b.min.y || b.max.y <= a.min.y;
                        assert!(apart, "{view}: {a:?} and {b:?} overlap");
                    }
                }
                assert!(lay.data.size().min_element() >= 200.0, "{view}: the data's panel is {:?}", lay.data.size());
            }
        }
        for view in [Vec2::new(240.0, 180.0), Vec2::new(1.0, 1.0), Vec2::new(10_000.0, 60.0), Vec2::new(60.0, 10_000.0)] {
            let lay = Layout::of(view, 46.0, Lower::Fits);
            let points = [lay.noise.min, lay.noise.max, lay.data.min, lay.data.max, lay.arrow.0, lay.arrow.1, lay.verdict.0];
            assert!(points.iter().all(|p| p.is_finite()), "{view}: {lay:?}");
        }
    }

    /// The numbers behind the gate (run with --ignored --nocapture): how far from
    /// normal bell-shaped noise gets at its best turn, times √n, at the 97.5th
    /// percentile of 8000 samples, for each number of points in `GATE`.
    #[test]
    #[ignore]
    fn calibrate() {
        for n in [20usize, 30, 50, 75, 100, 150, 200, 300, 400, 600, 1000, 1500, 2000] {
            let trials = 8000;
            let mut v: Vec<f64> = (0..trials)
                .filter_map(|k| {
                    let mut rng = Rng::new(900_000 + k as u64 * 7 + n as u64 * 1_000_003);
                    let pts: Vec<DVec2> = (0..n).map(|_| DVec2::new(rng.normal() as f64, rng.normal() as f64)).collect();
                    Turn::of(&pts).map(|t| t.score(t.best) * (n as f64).sqrt())
                })
                .collect();
            v.sort_by(f64::total_cmp);
            println!("n={n:5}: 90% {:.2} 95% {:.2} 97.5% {:.2} 99% {:.2}", quantile(&v, 0.9), quantile(&v, 0.95), quantile(&v, 0.975), quantile(&v, 0.99));
        }
    }

    /// How often the verdict is right, with few points (run with --ignored --nocapture).
    #[test]
    #[ignore]
    fn explore() {
        for noise in [Noise::Uniform, Noise::Laplace, Noise::TwoHumps] {
            for n in [20usize, 50, 100, 150, 200, 300, 600] {
                let mut counts = [0usize; 5];
                for k in 0..400 {
                    let v = seen_of(noise, 0.0, 1.0, true, n, 500 + k).verdict;
                    counts[match v {
                        Verdict::Nothing => 0,
                        Verdict::Round => 1,
                        Verdict::Apart => 2,
                        Verdict::Forwards => 3,
                        Verdict::Backwards => 4,
                    }] += 1;
                }
                println!("{noise:?} n={n}: of 400 [nothing, cannot tell, apart, X to Y, Y to X] {counts:?}");
            }
        }
        let s = Lingam::create();
        println!("opening: {:?}", s.seen.verdict);
    }
}

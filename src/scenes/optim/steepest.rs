//! Which way is steepest? "Steepest" means: of all the steps of one size, the one that
//! lowers the loss most to first order. What "size" means is a ruler (a norm), and each
//! ruler has its own answer: the lowest point of its unit shape on the tilted plane the
//! gradient makes. A circle gives the gradient's own direction, a square its signs, a
//! diamond its largest entry alone, and an ellipse shaped like the bowl, Newton's step.
//! For a weight matrix the same three questions give G itself, the signs of its entries,
//! and UVᵀ.

use super::*;
use crate::math::linalg;
use crate::scenes::stats::plot::{self, Plot};
use glam::{DMat2, DVec2};
use std::f64::consts::PI as PI64;

pub const DEF: SceneDef = SceneDef {
    id: "steepest",
    title: "Which way is steepest? It depends on the ruler",
    group: GROUP,
    blurb: "Measure a step by its length, by its largest entry or by the curvature of the bowl, and the steepest way down becomes the gradient, its signs, or Newton's step.",
    make: |_| Box::new(Steepest::new()),
};

// ----- the mathematics, in double precision -------------------------------------------

/// Five ways of saying how big a step Δ of the two weights is.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Ruler {
    /// Its length, √(Δ₁² + Δ₂²): the unit shape is a circle.
    Round,
    /// Its largest entry, max(|Δ₁|, |Δ₂|): a square.
    Square,
    /// The sum of its entries' sizes, |Δ₁| + |Δ₂|: a diamond.
    Diamond,
    /// The bowl's own curvature, √(ΔᵀHΔ) (scaled, below): an ellipse shaped like the
    /// bowl's level lines.
    Bowl,
    /// Each coordinate's curvature alone, √(ΔᵀDΔ) with D the diagonal of H: an ellipse
    /// lined up with the axes.
    Axes,
}

const RULERS: [Ruler; 5] = [Ruler::Round, Ruler::Square, Ruler::Diamond, Ruler::Bowl, Ruler::Axes];

impl Ruler {
    fn index(self) -> usize {
        self as usize
    }

    fn colour(self) -> Color {
        match self {
            Ruler::Round => SGD,
            // (with its moving averages switched off, Adam is the square's fixed-length step)
            Ruler::Square => ADAM,
            Ruler::Diamond => pal::PINK,
            Ruler::Bowl => CURVATURE,
            Ruler::Axes => pal::PURPLE,
        }
    }

    /// What it measures, as the board's buttons and the legends say it.
    fn name(self) -> &'static str {
        match self {
            Ruler::Round => "length",
            Ruler::Square => "largest entry",
            Ruler::Diamond => "sum of sizes",
            Ruler::Bowl => "curvature",
            Ruler::Axes => "curvature on axes",
        }
    }

    /// What its steepest step is called (in the paths' legend).
    fn rule(self) -> &'static str {
        match self {
            Ruler::Round => "gradient descent",
            Ruler::Square => "sign descent",
            Ruler::Diamond => "one coordinate at a time",
            Ruler::Bowl => "Newton's step",
            Ruler::Axes => "diagonal scaling",
        }
    }
}

/// The bowl's Hessian: curvature 1/κ along its floor, which runs at the angle θ, and 1
/// across it. The loss is f(w) = ½ wᵀHw, lowest at 0, and its gradient is g = Hw.
fn hessian(theta: f64, kappa: f64) -> DMat2 {
    let u = DVec2::from_angle(theta);
    let v = u.perp();
    outer(u, u) * (1.0 / kappa.max(1.0)) + outer(v, v)
}

fn outer(u: DVec2, v: DVec2) -> DMat2 {
    DMat2::from_cols(u * v.x, u * v.y)
}

fn loss(h: DMat2, w: DVec2) -> f64 {
    0.5 * w.dot(h * w)
}

/// The matrix P of an ellipse ruler, √(ΔᵀPΔ): H or its diagonal D, scaled to have
/// determinant 1, so that its unit shape has the circle's area. (A ruler's scale changes
/// how long its steepest step is, never which way it points; and the second form's step
/// does not depend on it at all.)
fn quadratic(r: Ruler, h: DMat2) -> DMat2 {
    match r {
        Ruler::Bowl => h * (1.0 / h.determinant().max(1e-300).sqrt()),
        Ruler::Axes => {
            let (a, b) = (h.x_axis.x, h.y_axis.y);
            DMat2::from_diagonal(DVec2::new(a, b)) * (1.0 / (a * b).max(1e-300).sqrt())
        }
        _ => DMat2::IDENTITY,
    }
}

/// The size of the step Δ under the ruler.
fn size(r: Ruler, h: DMat2, d: DVec2) -> f64 {
    match r {
        Ruler::Round => d.length(),
        Ruler::Square => d.x.abs().max(d.y.abs()),
        Ruler::Diamond => d.x.abs() + d.y.abs(),
        Ruler::Bowl | Ruler::Axes => d.dot(quadratic(r, h) * d).max(0.0).sqrt(),
    }
}

/// The dual size of the gradient: the most g · t over the steps t of size 1, which is
/// how much the loss falls, to first order, along the steepest step of size 1. For the
/// circle it is |g|; for the square |g₁| + |g₂| (the diamond's ruler); for the diamond
/// max(|g₁|, |g₂|) (the square's); for an ellipse √(gᵀP⁻¹g).
fn dual(r: Ruler, h: DMat2, g: DVec2) -> f64 {
    match r {
        Ruler::Round => g.length(),
        Ruler::Square => g.x.abs() + g.y.abs(),
        Ruler::Diamond => g.x.abs().max(g.y.abs()),
        Ruler::Bowl | Ruler::Axes => g.dot(quadratic(r, h).inverse() * g).max(0.0).sqrt(),
    }
}

/// L: the most the bowl curves over a step of size 1 under the ruler, the largest tᵀHt
/// with t of size 1. With it, f(w + Δ) ≤ f(w) + g · Δ + (L/2) size(Δ)² for every Δ, so a
/// step that makes the right-hand side least never goes up. (A convex function on a
/// polygon is largest at a corner: for the square at (±1, ±1), for the diamond at the
/// four vertices. For an ellipse it is the largest eigenvalue of P⁻¹H.)
fn bound(r: Ruler, h: DMat2) -> f64 {
    let (a, b, c) = (h.x_axis.x, h.y_axis.y, h.y_axis.x);
    match r {
        Ruler::Round => 0.5 * (a + b) + (0.25 * (a - b) * (a - b) + c * c).sqrt(),
        Ruler::Square => a + b + 2.0 * c.abs(),
        Ruler::Diamond => a.max(b),
        Ruler::Bowl => h.determinant().max(0.0).sqrt(),
        Ruler::Axes => (a * b).max(0.0).sqrt() + c.abs(),
    }
}

/// Below this share of the gradient's length two slopes count as equal, or a slope as
/// none: where the steepest step stops being one point.
const TIE: f64 = 1e-9;

/// The steepest steps of size 1: one step, or a whole edge of the unit shape (every
/// point of it is as steep as every other), or, when the gradient is 0, every step.
#[derive(Clone, Copy, Debug, PartialEq)]
enum Set {
    One(DVec2),
    Edge(DVec2, DVec2),
    All,
}

impl Set {
    /// The step taken where there is a choice: the middle of the edge (for the square,
    /// 0 in the coordinate whose slope is 0), and no step at all when every step is level.
    fn middle(&self) -> DVec2 {
        match *self {
            Set::One(d) => d,
            Set::Edge(a, b) => (a + b) * 0.5,
            Set::All => DVec2::ZERO,
        }
    }

    fn scaled(&self, t: f64) -> Set {
        match *self {
            Set::One(d) => Set::One(d * t),
            Set::Edge(a, b) => Set::Edge(a * t, b * t),
            Set::All => Set::All,
        }
    }
}

/// Of the steps of size 1 under the ruler, the ones that make g · Δ least: the points of
/// the unit shape that lie lowest on the tilted plane.
fn steepest(r: Ruler, h: DMat2, g: DVec2) -> Set {
    let n = g.length();
    // (a gradient that is no number has no steepest step either)
    if n.is_nan() || n <= 1e-12 {
        return Set::All;
    }
    let sign = |v: f64| if v > 0.0 { 1.0 } else { -1.0 };
    match r {
        Ruler::Round => Set::One(-g / n),
        Ruler::Square => {
            let free = |v: f64| v.abs() <= TIE * n;
            match (free(g.x), free(g.y)) {
                (false, false) => Set::One(DVec2::new(-sign(g.x), -sign(g.y))),
                (true, false) => Set::Edge(DVec2::new(-1.0, -sign(g.y)), DVec2::new(1.0, -sign(g.y))),
                (false, true) => Set::Edge(DVec2::new(-sign(g.x), -1.0), DVec2::new(-sign(g.x), 1.0)),
                (true, true) => Set::All,
            }
        }
        Ruler::Diamond => {
            let (x, y) = (g.x.abs(), g.y.abs());
            if (x - y).abs() <= TIE * n {
                Set::Edge(DVec2::new(-sign(g.x), 0.0), DVec2::new(0.0, -sign(g.y)))
            } else if x > y {
                Set::One(DVec2::new(-sign(g.x), 0.0))
            } else {
                Set::One(DVec2::new(0.0, -sign(g.y)))
            }
        }
        Ruler::Bowl | Ruler::Axes => {
            let q = quadratic(r, h).inverse() * g;
            Set::One(-q / g.dot(q).max(1e-300).sqrt())
        }
    }
}

/// The two forms of "the steepest step".
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Form {
    /// The step of size η that lowers the loss most to first order: always size η,
    /// whatever the gradient's size.
    Fixed,
    /// The step that makes g · Δ + (L/2) size(Δ)² least: the same direction, of size
    /// (dual size of g) / L.
    Scaled,
}

/// How big the step is under its ruler, in each form.
fn reach(r: Ruler, h: DMat2, g: DVec2, form: Form, eta: f64) -> f64 {
    match form {
        Form::Fixed => eta,
        Form::Scaled => dual(r, h, g) / bound(r, h).max(1e-300),
    }
}

fn step(r: Ruler, h: DMat2, g: DVec2, form: Form, eta: f64) -> DVec2 {
    steepest(r, h, g).middle() * reach(r, h, g, form, eta)
}

/// The most steps a run is followed for, and the share of its first loss it must get below.
const MOST_STEPS: usize = 2000;
const TARGET: f64 = 1e-6;

/// A run of second-form steps from a start, under one ruler.
#[derive(Clone, Debug, PartialEq)]
struct Run {
    pts: Vec<DVec2>,
    /// The steps it took to bring the loss below a millionth of its start (None: more
    /// than `MOST_STEPS`).
    steps: Option<usize>,
}

fn run(r: Ruler, h: DMat2, w0: DVec2) -> Run {
    let f0 = loss(h, w0);
    let mut pts = vec![w0];
    let mut w = w0;
    for n in 0..=MOST_STEPS {
        if loss(h, w) <= TARGET * f0 {
            return Run { pts, steps: Some(n) };
        }
        if n < MOST_STEPS {
            w += step(r, h, h * w, Form::Scaled, 0.0);
            pts.push(w);
        }
    }
    Run { pts, steps: None }
}

// ----- the same for a weight matrix -----------------------------------------------------

/// How many examples a 3×3 gradient is made from, for each of the matrix panel's buttons.
const EXAMPLES: [usize; 3] = [1, 2, 8];
/// The draw each button starts from.
const FIRST_DRAW: u64 = 1;

/// The gradient of a 3×3 linear layer over a batch: the average over the examples of
/// (the error at the outputs) times (the input), transposed, one outer product per
/// example (scene `backpropmat`). The inputs spread unequally over their three directions
/// (by 1, 0.35 and 0.12), and so do the errors (1, 0.6, 0.3), as a layer's often do:
/// the gradient is lopsided. Row-major.
fn gradient3(examples: usize, draw: u64) -> [f64; 9] {
    let mut rng = Rng::new(0x5EED_0000 + draw * 977 + examples as u64);
    let (xs, ds) = ([1.0, 0.35, 0.12], [1.0, 0.6, 0.3]);
    let mut g = [0.0; 9];
    let n = examples.max(1);
    for _ in 0..n {
        let x: [f64; 3] = std::array::from_fn(|j| rng.normal() as f64 * xs[j]);
        let d: [f64; 3] = std::array::from_fn(|i| rng.normal() as f64 * ds[i]);
        for i in 0..3 {
            for j in 0..3 {
                g[i * 3 + j] += d[i] * x[j] / n as f64;
            }
        }
    }
    g
}

/// The three steepest steps of size 1 for a 3×3 gradient G, and their singular values.
#[derive(Clone, Debug, PartialEq)]
struct Steps3 {
    g: [f64; 9],
    /// G/‖G‖: steepest under the root of the sum of the squared entries.
    frob: [f64; 9],
    /// The signs of G's entries: steepest under the largest entry.
    sign: [f64; 9],
    /// UVᵀ from G = USVᵀ, over the singular values that are not 0: steepest under the
    /// spectral norm (the most the matrix stretches a vector of length 1).
    ortho: [f64; 9],
    /// Singular values, largest first: of G, of sign G and of UVᵀ.
    sv: [[f64; 3]; 3],
    /// How many singular values of G are not 0 (where fewer than 3, UVᵀ is not the only
    /// steepest step: any size up to 1 will do in the directions G does not use).
    rank: usize,
    /// How many entries of G are exactly 0 (each is then free between −1 and 1 in sign G).
    zeros: usize,
    /// The dual sizes: ‖G‖ (root of the sum of squares), the sum of the entries' sizes,
    /// and the sum of the singular values.
    duals: [f64; 3],
}

fn singular_values(a: &[f64; 9]) -> [f64; 3] {
    let (_, s, _) = linalg::svd(a, 3, 3);
    [s[0], s[1], s[2]]
}

fn steps3(g: [f64; 9]) -> Steps3 {
    let (u, s, v) = linalg::svd(&g, 3, 3);
    let top = s[0];
    let rank = s.iter().filter(|x| top > 0.0 && **x > 1e-12 * top).count();
    let mut ortho = [0.0; 9];
    for k in 0..rank {
        for i in 0..3 {
            for j in 0..3 {
                ortho[i * 3 + j] += u[i * 3 + k] * v[j * 3 + k];
            }
        }
    }
    let frobenius = g.iter().map(|x| x * x).sum::<f64>().sqrt();
    let frob = g.map(|x| if frobenius > 0.0 { x / frobenius } else { 0.0 });
    let sign = g.map(|x| {
        if x > 0.0 {
            1.0
        } else if x < 0.0 {
            -1.0
        } else {
            0.0
        }
    });
    let sv = [[s[0], s[1], s[2]], singular_values(&sign), singular_values(&ortho)];
    let zeros = g.iter().filter(|x| **x == 0.0).count();
    let duals = [frobenius, g.iter().map(|x| x.abs()).sum(), s.iter().sum()];
    Steps3 { g, frob, sign, ortho, sv, rank, zeros, duals }
}

// ----- experiments ----------------------------------------------------------------------

/// Which rulers the main picture shows.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Choice {
    One(Ruler),
    /// The circle, the square and the diamond together.
    Three,
}

impl Choice {
    fn shows(self, r: Ruler) -> bool {
        match self {
            Choice::One(q) => q == r,
            Choice::Three => matches!(r, Ruler::Round | Ruler::Square | Ruler::Diamond),
        }
    }

    /// The board's six buttons, in order.
    const BUTTONS: [Choice; 6] =
        [Choice::One(Ruler::Round), Choice::One(Ruler::Square), Choice::One(Ruler::Diamond), Choice::One(Ruler::Bowl), Choice::One(Ruler::Axes), Choice::Three];

    fn label(self) -> &'static str {
        match self {
            Choice::One(r) => r.name(),
            Choice::Three => "the first three",
        }
    }
}

/// What plays after an experiment's button.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Film {
    /// The point goes once round a round bowl, so that the gradient turns, stopping at
    /// each direction where the square's or the diamond's answer is not one point.
    Round,
    /// The bowl turns through a quarter turn under the point.
    Turn,
}

/// The radius the point goes round at in the first film.
const ROUND_R: f64 = 1.7;
/// Seconds to turn through 45°, and to stay at each stop.
const MOVE_S: f64 = 1.3;
const HOLD_S: f64 = 1.0;
/// Where the first film starts and ends: half way between an axis and a diagonal.
const ROUND_FROM: f64 = PI64 / 8.0;
/// Seconds the bowl takes to turn in the second film, and how far it turns.
const TURN_S: f64 = 6.0;
const TURN_BY: f64 = PI64 / 2.0;

/// The angle of the point in the first film at `t` seconds: half a leg to the first
/// diagonal, then eight legs of 45°, each eased and followed by a stop, then half a leg
/// back to where it began. (None once it is over.)
fn round_angle(t: f64) -> Option<f64> {
    let half = 0.5 * MOVE_S;
    if t < half {
        return Some(ROUND_FROM + (PI64 / 8.0) * smooth(t / half));
    }
    let mut t = t - half;
    for k in 0..8 {
        let at = PI64 / 4.0 * (k + 1) as f64;
        if t < HOLD_S {
            return Some(at);
        }
        t -= HOLD_S;
        let span = if k < 7 { PI64 / 4.0 } else { PI64 / 8.0 };
        let time = if k < 7 { MOVE_S } else { half };
        if t < time {
            return Some(at + span * smooth(t / time));
        }
        t -= time;
    }
    None
}

struct Trial {
    label: &'static str,
    choice: Choice,
    /// The bowl's angle (degrees) and stretch.
    theta: f64,
    kappa: f64,
    w: [f64; 2],
    form: Form,
    /// The size of the shapes (and of the first form's steps).
    eta: f64,
    film: Option<Film>,
    /// The matrix panel's button, and whether a phone's board shows the matrix.
    examples: usize,
    matrix: bool,
}

const ETA: f64 = 0.8;

/// The point of the second and third experiments, before the third turns it.
const VALLEY_W: [f64; 2] = [-1.9, 0.7];

fn turned(p: [f64; 2], degrees: f64) -> [f64; 2] {
    let q = DVec2::from_angle(degrees.to_radians()).rotate(DVec2::from(p));
    [q.x, q.y]
}

const TRIALS: [Trial; 5] = [
    Trial {
        label: "turn the gradient",
        choice: Choice::Three,
        theta: 0.0,
        kappa: 1.0,
        w: [0.0, 0.0],
        form: Form::Fixed,
        eta: ETA,
        film: Some(Film::Round),
        examples: 2,
        matrix: false,
    },
    Trial {
        label: "a valley along an axis",
        choice: Choice::One(Ruler::Axes),
        theta: 0.0,
        kappa: 10.0,
        w: VALLEY_W,
        form: Form::Scaled,
        eta: 0.6,
        film: None,
        examples: 2,
        matrix: false,
    },
    Trial {
        label: "the same valley turned by 45°",
        choice: Choice::One(Ruler::Axes),
        theta: 45.0,
        kappa: 10.0,
        w: [0.0, 0.0],
        form: Form::Scaled,
        eta: 0.6,
        film: None,
        examples: 2,
        matrix: false,
    },
    Trial {
        label: "Newton's ellipse",
        choice: Choice::One(Ruler::Bowl),
        theta: 15.0,
        kappa: 10.0,
        w: [0.5, -2.0],
        form: Form::Scaled,
        eta: ETA,
        film: Some(Film::Turn),
        examples: 2,
        matrix: false,
    },
    Trial {
        label: "a weight matrix",
        choice: Choice::Three,
        theta: OPEN_THETA,
        kappa: OPEN_KAPPA,
        w: OPEN_W,
        form: Form::Fixed,
        eta: ETA,
        film: None,
        examples: 2,
        matrix: true,
    },
];

/// Where an experiment puts the point: the first film's start, the third experiment's
/// turned valley point, or as written.
fn trial_point(k: usize) -> DVec2 {
    let t = &TRIALS[k];
    match k {
        0 => DVec2::from_angle(ROUND_FROM) * ROUND_R,
        2 => DVec2::from(turned(VALLEY_W, t.theta)),
        _ => DVec2::from(t.w),
    }
}

/// How long the matrix panel is outlined after its experiment's button (seconds).
const ATTENTION_S: f64 = 2.5;

/// The scene as it opens.
const OPEN_THETA: f64 = -60.0;
const OPEN_KAPPA: f64 = 5.0;
const OPEN_W: [f64; 2] = [1.9, 0.3];

// ----- what glides ------------------------------------------------------------------------

/// The bowl, the point, the form (0 the first, 1 the second, between while it glides) and
/// the size η of the first form's steps.
#[derive(Clone, Copy, Debug, PartialEq)]
struct Setting {
    theta: f64,
    ln_kappa: f64,
    w: DVec2,
    form: f64,
    eta: f64,
}

impl Setting {
    fn numbers(&self) -> [f64; 6] {
        [self.theta, self.ln_kappa, self.w.x, self.w.y, self.form, self.eta]
    }

    fn from_numbers(v: [f64; 6]) -> Setting {
        Setting { theta: v[0], ln_kappa: v[1], w: DVec2::new(v[2], v[3]), form: v[4], eta: v[5] }
    }

    fn kappa(&self) -> f64 {
        self.ln_kappa.exp()
    }

    fn h(&self) -> DMat2 {
        hessian(self.theta, self.kappa())
    }

    fn form(&self) -> Form {
        if self.form >= 0.5 {
            Form::Scaled
        } else {
            Form::Fixed
        }
    }
}

/// How fast a change is glided to: each of the two eases covers two thirds of what is
/// left in this many seconds.
const FOLLOW: f64 = 0.09;

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

    fn within(&self, near: f64) -> bool {
        let (to, between, shown) = (self.to.numbers(), self.between.numbers(), self.shown.numbers());
        (0..6).all(|i| (to[i] - between[i]).abs() < near && (to[i] - shown[i]).abs() < near)
    }

    /// Change the setting at once, in all three: what a hand moves.
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
        for i in 0..6 {
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

fn smooth(t: f64) -> f64 {
    let t = t.clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

/// The limits of the controls.
const W_MAX: f64 = 3.0;
const KAPPA_MAX: f64 = 100.0;
const ETA_RANGE: [f64; 2] = [0.2, 1.0];

/// The setting with every number within its control's limits (and a number that is not
/// one replaced by where the control starts).
fn within_limits(s: Setting) -> Setting {
    let keep = |v: f64, lo: f64, hi: f64, or: f64| if v.is_finite() { v.max(lo).min(hi) } else { or };
    Setting {
        theta: keep(s.theta, -PI64, 2.0 * PI64, 0.0),
        ln_kappa: keep(s.ln_kappa, 0.0, KAPPA_MAX.ln(), 0.0),
        w: DVec2::new(keep(s.w.x, -W_MAX, W_MAX, 1.0), keep(s.w.y, -W_MAX, W_MAX, 1.0)),
        form: keep(s.form, 0.0, 1.0, 0.0),
        eta: keep(s.eta, ETA_RANGE[0], ETA_RANGE[1], ETA),
    }
}

/// The lines through the bottom along which the gradient points along an axis or a
/// diagonal (where the square's or the diamond's answer is an edge): the directions
/// H⁻¹u for u at 0°, 45°, 90° and 135°.
fn tie_lines(h: DMat2) -> [DVec2; 4] {
    let inv = h.inverse();
    std::array::from_fn(|k| (inv * DVec2::from_angle(PI64 / 4.0 * k as f64)).normalize_or(DVec2::X))
}

/// A point put down by hand, moved onto the bottom, or onto one of the lines where the
/// steepest step is not one point, if it is within `near` of it: so that those places,
/// each a line with no width, can be found by hand.
fn snapped(h: DMat2, w: DVec2, near: f64) -> DVec2 {
    if w.length() < near {
        return DVec2::ZERO;
    }
    let mut best = (near, w);
    for e in tie_lines(h) {
        let on = e * w.dot(e);
        let d = (w - on).length();
        if d < best.0 {
            best = (d, on);
        }
    }
    best.1
}

// ----- where things go -----------------------------------------------------------------------

/// The part of the plane the picture is fitted to.
const FRAME_HALF: f64 = 3.0;
/// The gradient is drawn at this share of the plane's scale (its units are loss per unit
/// of weight, not weight: any scale would do, and this one keeps it on the picture).
const ARROW: f64 = 0.75;

type Rect = plot::Rect;

const PAD: f32 = 12.0;
const LINE_H: f32 = 17.0;
const BUTTON_H: f32 = 25.0;
const BOARD_W: f32 = 384.0;
/// The sizes of the board's words.
const SMALL: f32 = 11.5;

/// The panel drawn on the picture: the ruler's buttons, the form's, a line on the step,
/// the paths down the bowl and the matrix panel (on a canvas short of room, one of the
/// two at a time, chosen by tabs).
#[derive(Clone, Debug)]
struct Board {
    min: Vec2,
    max: Vec2,
    rulers: Vec<Rect>,
    forms: Vec<Rect>,
    /// The line saying what the step is.
    formula: Vec2,
    /// Two tabs (paths, matrix) when only one of the two fits.
    tabs: Option<Vec<Rect>>,
    paths: Option<PathsBox>,
    matrix: Option<MatrixBox>,
}

#[derive(Clone, Debug)]
struct PathsBox {
    title: Vec2,
    chart: Rect,
    legend: Vec2,
    /// The legend's entries side by side in two columns (a phone), or one under another.
    columns: usize,
}

#[derive(Clone, Debug)]
struct MatrixBox {
    title: Vec2,
    chart: Rect,
    buttons: Vec<Rect>,
}

#[derive(Clone, Debug)]
struct Layout {
    narrow: bool,
    picture: (Vec2, Vec2),
    board: Option<Board>,
    /// Where the board has tabs: the same board with the matrix panel in place of the paths.
    matrix_board: Option<Board>,
}

/// The heights of the board's parts.
const RULER_ROWS_H: f32 = 2.0 * BUTTON_H + 6.0;
const FORMULA_H: f32 = 2.0 * LINE_H;
const PATHS_TITLE_H: f32 = LINE_H + 4.0;
const MATRIX_TITLE_H: f32 = 2.0 * LINE_H + 2.0;
/// Under the bars: the three steps' names, two lines.
const MATRIX_FOOT_H: f32 = 2.0 * LINE_H + 6.0;
const MATRIX_CHART_H: f32 = 92.0;

impl Layout {
    fn of(view: Vec2, inset: f32) -> Layout {
        let narrow = view.x < 640.0 && view.y >= 520.0;
        let whole = Layout { narrow, picture: (Vec2::new(0.0, inset), view), board: None, matrix_board: None };
        let pad = if narrow { 10.0 } else { PAD };
        let width = if narrow { view.x } else { BOARD_W };
        let columns = if narrow { 2 } else { 1 };
        let legend_h = 5usize.div_ceil(columns) as f32 * 16.0 + 4.0;
        let top_part = pad + RULER_ROWS_H + 8.0 + BUTTON_H + 8.0 + FORMULA_H + 6.0;
        let tabs_h = BUTTON_H + 8.0;
        let matrix_h = MATRIX_TITLE_H + MATRIX_CHART_H + MATRIX_FOOT_H + BUTTON_H + 6.0;
        let paths_fixed = PATHS_TITLE_H + 4.0 + legend_h;
        // the board's parts from its top: `parts` says whether the paths, the matrix panel
        // or both are on it, and `tabs` whether it has the tabs that choose between them
        let place = |min: Vec2, height: f32, tabs: bool, chart_h: f32, paths: bool, matrix: bool| -> Board {
            let x = min.x + pad;
            let right = min.x + width - pad;
            let mut y = min.y + pad;
            let rulers = [plot::even_row(3, x, right, y, BUTTON_H, 6.0), plot::even_row(3, x, right, y + BUTTON_H + 6.0, BUTTON_H, 6.0)].concat();
            y += RULER_ROWS_H + 8.0;
            let forms = plot::even_row(2, x, right, y, BUTTON_H, 6.0);
            y += BUTTON_H + 8.0;
            let formula = Vec2::new(x, y);
            y += FORMULA_H + 6.0;
            let tab_rects = tabs.then(|| {
                let r = plot::even_row(2, x, right, y, BUTTON_H, 6.0);
                y += tabs_h;
                r
            });
            let paths = paths.then(|| {
                let title = Vec2::new(x, y);
                y += PATHS_TITLE_H;
                let chart = (Vec2::new(x, y), Vec2::new(right, y + chart_h));
                y += chart_h + 6.0;
                let legend = Vec2::new(x, y);
                y += legend_h + 6.0;
                PathsBox { title, chart, legend, columns }
            });
            let matrix = matrix.then(|| {
                let title = Vec2::new(x, y);
                y += MATRIX_TITLE_H;
                let chart = (Vec2::new(x + 26.0, y), Vec2::new(right - 4.0, y + MATRIX_CHART_H));
                y += MATRIX_CHART_H + MATRIX_FOOT_H;
                let buttons = plot::even_row(4, x, right, y, BUTTON_H, 6.0);
                MatrixBox { title, chart, buttons }
            });
            Board { min, max: Vec2::new(min.x + width, min.y + height), rulers, forms, formula, tabs: tab_rects, paths, matrix }
        };
        let chart_most = width - 2.0 * pad;
        if narrow {
            // the board under the picture, which keeps at least half the canvas
            let room = view.y - (view.y * 0.5).max(inset + 150.0);
            let above = top_part + tabs_h;
            let chart_h = (room - above - paths_fixed - pad).min(chart_most * 0.4).min(140.0);
            if chart_h < 80.0 {
                return whole;
            }
            let height = (above + paths_fixed + chart_h + pad).max(above + matrix_h + pad);
            let min = Vec2::new(0.0, view.y - height);
            return Layout {
                narrow,
                picture: (Vec2::new(0.0, inset), Vec2::new(view.x, min.y)),
                board: Some(place(min, height, true, chart_h, true, false)),
                matrix_board: Some(place(min, height, true, chart_h, false, true)),
            };
        }
        if view.x < 760.0 {
            return whole;
        }
        let min = Vec2::new(view.x - width - 12.0, inset + 12.0);
        let room = view.y - 12.0 - min.y;
        let picture = (Vec2::new(0.0, inset), Vec2::new(min.x - 8.0, view.y));
        let both = top_part + paths_fixed + matrix_h + pad;
        let chart_h = (room - both).min(chart_most).min(300.0);
        if chart_h >= 140.0 {
            return Layout { narrow, picture, board: Some(place(min, both + chart_h, false, chart_h, true, true)), matrix_board: None };
        }
        // one of the two at a time
        let above = top_part + tabs_h;
        let chart_h = (room - above - paths_fixed - pad).min(chart_most).min(300.0);
        if chart_h < 90.0 {
            return whole;
        }
        let height = (above + paths_fixed + chart_h + pad).max(above + matrix_h + pad);
        Layout { narrow, picture, board: Some(place(min, height, true, chart_h, true, false)), matrix_board: Some(place(min, height, true, chart_h, false, true)) }
    }

    /// The board as it is with the matrix tab chosen or not.
    fn board_for(&self, matrix_tab: bool) -> Option<Board> {
        if matrix_tab && self.matrix_board.is_some() {
            return self.matrix_board.clone();
        }
        self.board.clone()
    }

    /// The part of the canvas the picture's frame is fitted into: the picture's, less the
    /// heading at its top and the caption at its foot.
    fn frame_room(&self) -> (Vec2, Vec2) {
        let (min, max) = self.picture;
        let foot = if self.narrow { 10.0 } else { 46.0 };
        (min + Vec2::new(12.0, 40.0), max - Vec2::new(12.0, foot))
    }
}

fn inside_rect(p: Vec2, (min, max): Rect) -> bool {
    p.cmpge(min).all() && p.cmple(max).all()
}

fn v3(p: DVec2) -> Vec3 {
    Vec3::new(p.x as f32, p.y as f32, 0.0)
}

/// Points round a unit shape: as many as make every corner one of them (a multiple of 8).
const RING: usize = 128;

/// The point of the ruler's unit shape in the direction of the `i`th of `RING` angles.
fn boundary(r: Ruler, h: DMat2, i: usize) -> DVec2 {
    let d = DVec2::from_angle(i as f64 / RING as f64 * 2.0 * PI64);
    d / size(r, h, d).max(1e-12)
}

/// The parts of the picture drawn as lines that come and go, each in a batch of its own
/// (told apart by a hair of width), so that `eidos motion` can follow the rest point for
/// point from frame to frame.
fn lines_for<'a>(f: &'a mut Frame, part: u8) -> &'a mut crate::gfx::draw::LineBatch {
    f.dl.lines_with(LineKind::Blend, LineStyle { width_scale: 1.0 + 1e-4 * part as f32, ..Default::default() })
}

/// A batch of lines with nothing in it this frame is given a point that cannot be seen:
/// an empty batch is dropped, and every batch after it would then be compared with the
/// wrong one.
fn keep_batch(batch: &mut crate::gfx::draw::LineBatch) {
    if batch.pts.len() <= 1 {
        batch.segment(Vec3::ZERO, Vec3::ZERO, 0.0, Color::CLEAR);
    }
}

/// The batches: one for each ruler's shape, one for each ruler's step, and the rest.
fn shape_part(r: Ruler) -> u8 {
    1 + r.index() as u8
}
fn step_part(r: Ruler) -> u8 {
    6 + r.index() as u8
}
fn edge_part(r: Ruler) -> u8 {
    11 + r.index() as u8
}
const TIE_PART: u8 = 16;
const LEVEL_PART: u8 = 17;
const PARTS: u8 = 18;

/// The kinds of line in the paths chart (see `plot::apart`).
const CHART_BOWL: u8 = plot::OWN;
fn chart_path(r: Ruler) -> u8 {
    plot::OWN + 1 + r.index() as u8
}

/// How many steps of each path are marked with a dot (the same number every frame).
const DOTS: usize = 40;

// ----- the scene -------------------------------------------------------------------------

pub struct Steepest {
    glide: Glide,
    choice: Choice,
    /// How much each ruler is seen, 0 to 1, easing towards what `choice` asks.
    seen: [f64; 5],
    /// The film playing (and how far into it), or waiting to play once the glide arrives.
    film: Option<(Film, f64)>,
    waiting: Option<Film>,
    /// Where the second film's bowl starts.
    turn_from: f64,
    /// The matrix panel: which button, which draw, and its steps.
    examples: usize,
    draw: u64,
    mats: Steps3,
    /// A phone's board: the matrix panel instead of the paths.
    matrix_tab: bool,
    /// Seconds left of the outline drawn round the matrix panel after its experiment.
    attention: f64,
    /// Each ruler's step as drawn, and its path as drawn: they follow the true ones
    /// through two short eases, so that a step that jumps (from one corner to the next)
    /// or a path that changes all at once (where the bowl turns through a tie) is seen
    /// moving, not flashing from one place to another.
    steps_drawn: [plot::Glide; 5],
    /// The lowest point of each shape, as drawn.
    touch_drawn: [plot::Glide; 5],
    paths_drawn: [plot::Glide; 5],
    /// The dots on each path's first `DOTS` steps, as drawn (always as many: a path
    /// shorter than that has the rest at its end).
    dots_drawn: [plot::Glide; 5],
    gliding: bool,
    /// Where a hand last put the point, and how far it was moved from there onto a line
    /// where the steepest step is not one point; and how much of that move is drawn so far.
    snap: Option<(DVec2, DVec2)>,
    snap_drawn: DVec2,
    /// A hand has moved the point since it took hold of it.
    gesture: bool,
    /// The paths, worked out for the setting they were worked out for.
    runs: Vec<Run>,
    runs_for: Option<[f64; 4]>,
    /// The runs' points and their first `DOTS`, flat, as the drawn ones follow them.
    targets: Vec<(Vec<f32>, Vec<f32>)>,
    /// Texts broken into lines for the board, by text and width.
    wrapped: Vec<((String, i32), Vec<String>)>,
    canvas: Vec2,
    inset: f32,
    measured: bool,
    new_view: bool,
}

impl Steepest {
    fn new() -> Self {
        let open = Setting { theta: OPEN_THETA.to_radians(), ln_kappa: OPEN_KAPPA.ln(), w: DVec2::from(OPEN_W), form: 0.0, eta: ETA };
        let mut scene = Steepest {
            glide: Glide::at(open),
            choice: Choice::Three,
            seen: [1.0, 1.0, 1.0, 0.0, 0.0],
            film: None,
            waiting: None,
            turn_from: 0.0,
            examples: 2,
            draw: FIRST_DRAW,
            mats: steps3(gradient3(EXAMPLES[2], FIRST_DRAW)),
            matrix_tab: false,
            attention: 0.0,
            steps_drawn: std::array::from_fn(|_| plot::Glide::default()),
            touch_drawn: std::array::from_fn(|_| plot::Glide::default()),
            paths_drawn: std::array::from_fn(|_| plot::Glide::default()),
            dots_drawn: std::array::from_fn(|_| plot::Glide::default()),
            gliding: false,
            snap: None,
            snap_drawn: DVec2::ZERO,
            gesture: false,
            runs: Vec::new(),
            runs_for: None,
            targets: Vec::new(),
            wrapped: Vec::new(),
            canvas: Vec2::new(1080.0, 1000.0),
            inset: 0.0,
            measured: false,
            new_view: false,
        };
        scene.advance(0.0, 0.0);
        scene
    }

    /// Set up experiment `k`: every setting its lesson depends on, glided to; its film
    /// plays once the glide has arrived.
    fn set_up(&mut self, k: usize) {
        let t = &TRIALS[k];
        self.set_examples(t.examples, FIRST_DRAW);
        self.matrix_tab = t.matrix;
        if t.matrix {
            // (the matrix panel's lesson: the plane is left as it is, and the panel is
            // outlined for a moment)
            self.attention = ATTENTION_S;
            return;
        }
        self.glide.to = Setting {
            theta: t.theta.to_radians(),
            ln_kappa: t.kappa.ln(),
            w: trial_point(k),
            form: if t.form == Form::Scaled { 1.0 } else { 0.0 },
            eta: t.eta,
        };
        self.choice = t.choice;
        self.film = None;
        self.waiting = t.film;
        self.snap = None;
        self.turn_from = t.theta.to_radians();
    }

    fn set_examples(&mut self, examples: usize, draw: u64) {
        self.examples = examples.min(EXAMPLES.len() - 1);
        self.draw = draw;
        self.mats = steps3(gradient3(EXAMPLES[self.examples], draw));
    }

    fn shown(&self) -> Setting {
        self.glide.shown
    }

    /// A hand on the point, at `raw`, moved onto `snapped` (see `snapped`): the film stops,
    /// the point is where the hand is at once, and glides from there the rest of the way.
    fn hand_snapped(&mut self, raw: DVec2, snapped: DVec2) {
        (self.film, self.waiting) = (None, None);
        let limit = |w: DVec2| within_limits(Setting { w, ..self.glide.to }).w;
        let (raw, snapped) = (limit(raw), limit(snapped));
        // (a new gesture starts from the point as drawn, with nothing of an earlier
        // snap still to glide)
        if !self.gesture {
            self.snap_drawn = DVec2::ZERO;
            self.gesture = true;
        }
        self.snap = Some((raw, snapped - raw));
        let w = raw + self.snap_drawn;
        self.glide.now(|s| s.w = w);
    }

    /// A control moved: the film stops, and the setting glides there.
    fn ask(&mut self, change: impl Fn(&mut Setting)) {
        (self.film, self.waiting) = (None, None);
        self.snap = None;
        change(&mut self.glide.to);
        self.glide.to = within_limits(self.glide.to);
    }

    /// Move everything on: the film by the scene's time, the glides by the clock's (a
    /// change of setting is seen through even while the scene's time stands still).
    fn advance(&mut self, dt: f64, real_dt: f64) {
        // a point moved onto a line by hand glides the last few pixels
        if let Some((raw, by)) = self.snap {
            if self.snap_drawn != by {
                let k = 1.0 - (-real_dt.max(0.0) / 0.05).exp();
                self.snap_drawn += (by - self.snap_drawn) * k;
                if (by - self.snap_drawn).length() < 1e-5 || !self.snap_drawn.is_finite() {
                    self.snap_drawn = by;
                }
                let w = raw + self.snap_drawn;
                self.glide.now(|s| s.w = w);
            }
        } else {
            self.snap_drawn = DVec2::ZERO;
        }
        self.glide.step(real_dt);
        if let Some(film) = self.waiting {
            if self.glide.within(0.004) {
                self.glide = Glide::at(self.glide.to);
                self.film = Some((film, 0.0));
                self.waiting = None;
            }
        }
        if let Some((film, t)) = self.film {
            let t = t + dt;
            match film {
                Film::Round => match round_angle(t) {
                    Some(a) => {
                        self.glide.now(|s| s.w = DVec2::from_angle(a) * ROUND_R);
                        self.film = Some((film, t));
                    }
                    None => {
                        self.glide.now(|s| s.w = DVec2::from_angle(ROUND_FROM) * ROUND_R);
                        self.film = None;
                    }
                },
                Film::Turn => {
                    let from = self.turn_from;
                    let s = smooth(t / TURN_S);
                    self.glide.now(|set| set.theta = from + TURN_BY * s);
                    self.film = if t >= TURN_S { None } else { Some((film, t)) };
                }
            }
        }
        if self.attention > 0.0 {
            self.attention = (self.attention - real_dt.max(0.0)).max(0.0);
        }
        // the rulers fade in and out
        let k = 1.0 - (-real_dt.max(0.0) / 0.12).exp();
        for r in RULERS {
            let want = if self.choice.shows(r) { 1.0 } else { 0.0 };
            let s = &mut self.seen[r.index()];
            if *s != want {
                *s += (want - *s) * k;
                if (*s - want).abs() < 2e-3 || !s.is_finite() {
                    *s = want;
                }
            }
        }
        self.update_runs();
        // the steps and paths as drawn follow the true ones
        let s = self.shown();
        let (h, form) = (s.h(), smooth(s.form));
        let g = h * s.w;
        let mut gliding = false;
        for r in RULERS {
            let size = (1.0 - form) * s.eta + form * reach_scaled(r, h, g);
            let unit = steepest(r, h, g).middle();
            let d = unit * size;
            gliding |= self.steps_drawn[r.index()].follow(&[d.x as f32, d.y as f32], real_dt as f32, false);
            let t = unit * s.eta;
            gliding |= self.touch_drawn[r.index()].follow(&[t.x as f32, t.y as f32], real_dt as f32, false);
            if let Some((path, dots)) = self.targets.get(r.index()) {
                gliding |= self.paths_drawn[r.index()].follow(path, real_dt as f32, true);
                gliding |= self.dots_drawn[r.index()].follow(dots, real_dt as f32, false);
            }
        }
        self.gliding = gliding;
    }

    /// The step of ruler `r` as drawn, from the point.
    fn step_drawn(&self, r: Ruler) -> DVec2 {
        let g = &self.steps_drawn[r.index()];
        DVec2::new(g.get(0) as f64, g.get(1) as f64)
    }

    /// The paths of the five rulers from the point, each in the second form, which
    /// reaches the bottom (worked out again only when the bowl or the point drawn has
    /// changed: 5 × 2000 steps at most, a fraction of a millisecond).
    fn update_runs(&mut self) {
        let s = self.shown();
        let key = [s.theta, s.ln_kappa, s.w.x, s.w.y];
        if self.runs_for == Some(key) {
            return;
        }
        let h = s.h();
        self.runs = RULERS.iter().map(|r| run(*r, h, s.w)).collect();
        self.runs_for = Some(key);
        // what the drawn paths and their dots follow, as the glides take them
        self.targets = self
            .runs
            .iter()
            .map(|run| {
                let path: Vec<f32> = run.pts.iter().flat_map(|p| [p.x as f32, p.y as f32]).collect();
                let last = run.pts.len().max(1) - 1;
                let dots: Vec<f32> = (0..DOTS).flat_map(|i| run.pts.get(i.min(last)).map_or([0.0; 2], |p| [p.x as f32, p.y as f32])).collect();
                (path, dots)
            })
            .collect();
    }

    fn at_rest(&self) -> bool {
        self.glide.arrived() && self.film.is_none() && self.waiting.is_none() && self.attention == 0.0 && !self.gliding && self.snap.is_none_or(|(_, by)| self.snap_drawn == by) && RULERS.iter().all(|r| self.seen[r.index()] == if self.choice.shows(*r) { 1.0 } else { 0.0 })
    }

    fn the_view(&self) -> View {
        let lay = Layout::of(self.canvas, self.inset);
        let (min, max) = lay.frame_room();
        let room = (max - min).max(Vec2::splat(60.0));
        let per_unit = (room.x / (2.0 * FRAME_HALF as f32)).min(room.y / (2.0 * FRAME_HALF as f32)).max(1e-3);
        let off = ((min + max) * 0.5 - self.canvas * 0.5) / per_unit;
        View::Fit { center: Vec2::new(-off.x, off.y), half: self.canvas.max(Vec2::ONE) * (0.5 / per_unit) }
    }

    /// What the picture's heading says.
    fn heading(&self) -> String {
        let s = self.shown();
        let g = s.h() * s.w;
        if g.length() <= 1e-12 {
            return r"At the bottom: $g = 0$, every step is level".into();
        }
        match self.choice {
            Choice::Three => {
                let h = s.h();
                match (steepest(Ruler::Square, h, g), steepest(Ruler::Diamond, h, g)) {
                    (Set::Edge(..), _) => "Three rulers: for the square, a whole edge is steepest".into(),
                    (_, Set::Edge(..)) => "Three rulers: for the diamond, a whole edge is steepest".into(),
                    _ => "The steepest step under three rulers".into(),
                }
            }
            Choice::One(r) => {
                let what = match (r, steepest(r, s.h(), g)) {
                    (_, Set::Edge(..)) => "a whole edge, not one point",
                    (Ruler::Round, _) => r"along $-g$",
                    (Ruler::Square, _) => "a corner",
                    (Ruler::Diamond, _) => "a vertex",
                    (Ruler::Bowl, _) => "straight at the bottom",
                    (Ruler::Axes, _) => r"along $-D^{-1} g$",
                };
                format!("Size by {}: the steepest step is {what}", r.name())
            }
        }
    }
}

/// A singular value to three significant figures, and 0 for one that is nothing.
fn sig(v: f64) -> String {
    if v.abs() < 1e-9 {
        return "0".into();
    }
    let decimals = (2.0 - v.abs().log10().floor()).clamp(0.0, 12.0) as usize;
    num(v, decimals)
}

/// A number for the board and the panel: two decimals, a true minus sign.
fn n2(v: f64) -> String {
    num(v, 2)
}

/// The step's formula for the board's line, in each form.
fn formula(r: Ruler, form: Form) -> &'static str {
    match (r, form) {
        (Ruler::Round, Form::Fixed) => r"step $= -\eta\, g / |g|$: size $\eta$ along $-g$",
        (Ruler::Round, Form::Scaled) => r"step $= -g / L$: gradient descent, $L$ the largest curvature",
        (Ruler::Square, Form::Fixed) => r"step $= -\eta \operatorname{sign} g$: sign descent",
        (Ruler::Square, Form::Scaled) => r"step $= -(|g_1| + |g_2|)/L \cdot \operatorname{sign} g$",
        (Ruler::Diamond, Form::Fixed) => r"only coordinate $i$ moves, by $-\eta \operatorname{sign} g_i$, for the larger $|g_i|$",
        (Ruler::Diamond, Form::Scaled) => r"only coordinate $i$ moves, by $-g_i / L$, for the larger $|g_i|$",
        (Ruler::Bowl, Form::Fixed) => r"step $\propto -H^{-1} g$: size $\eta$ towards the bottom",
        (Ruler::Bowl, Form::Scaled) => r"step $= -H^{-1} g$: Newton's step, to the bottom",
        (Ruler::Axes, Form::Fixed) => r"step $\propto -D^{-1} g$, $D$ the diagonal of $H$",
        (Ruler::Axes, Form::Scaled) => r"step $\propto -D^{-1} g$, of size $\|g\|_* / L$; $D$ the diagonal of $H$",
    }
}

impl Scene for Steepest {
    fn view(&self) -> View {
        self.the_view()
    }

    fn take_view(&mut self) -> Option<View> {
        std::mem::take(&mut self.new_view).then(|| self.the_view())
    }

    fn settings(&self, s: &mut ViewSettings) {
        s.grid_alpha = 0.08;
        s.axis_alpha = 0.35;
    }

    fn animated(&self) -> bool {
        !self.at_rest()
    }

    fn ui(&mut self, ui: &mut egui::Ui, _: &Gfx) {
        ui::note(ui, r"Gradient descent steps along $-g$, minus the gradient, and $-g$ is often called the steepest way down. Steepest means: of all the steps of one size, the one that lowers the loss most. Near the point $w$ the loss is, to first order, a tilted plane, $f(w + \Delta) \approx f(w) + g \cdot \Delta$, so the steepest step is the point of the shape of all steps of size at most $\eta$ that lies lowest on that plane:");
        ui::equation(ui, r"\Delta^* = \argmin_{\|\Delta\| \le \eta} \; g \cdot \Delta");
        ui::note(ui, r"What size means is a choice, a ruler $\|\cdot\|$, and each ruler gives its own answer. Several well-known optimisers are this one rule under different rulers (see a weight matrix, below).");

        ui::section(ui, "Things to try");
        for (k, trial) in TRIALS.iter().enumerate() {
            if ui::experiment(ui, trial.label, captions(k)) {
                self.set_up(k);
            }
        }

        ui::section(ui, "The ruler");
        let mut pick = Choice::BUTTONS.iter().position(|c| *c == self.choice).unwrap_or(0);
        let labels: Vec<&str> = Choice::BUTTONS.iter().map(|c| c.label()).collect();
        if ui::choice(ui, &labels, &mut pick) {
            self.choice = Choice::BUTTONS[pick.min(Choice::BUTTONS.len() - 1)];
        }
        ui::note(ui, "Or use the buttons on the picture. Drag the white point $w$, or the tip of the gradient's arrow.");

        ui::section(ui, "The step");
        let mut form = (self.glide.to.form >= 0.5) as usize;
        if ui::choice(ui, &[r"size $\eta$", r"size $\|g\|_* / L$"], &mut form) {
            self.ask(|s| s.form = form as f64);
        }
        ui::note(ui, r"The first form: the steepest step of size $\eta$, the same size however large $g$ is. Sign descent, $-\eta \operatorname{sign} g$, is this form under the largest entry. The second: the step that makes $g \cdot \Delta + \tfrac{L}{2}\|\Delta\|^2$ least. It points the same way, and its size is $\|g\|_* / L$, the gradient's dual size over $L$, so it shrinks as the gradient does. On this bowl it never raises the loss, and the paths chart takes it to the bottom. Under length it is gradient descent with step $1/L$.");
        let mut eta = self.glide.to.eta as f32;
        if ui::slider(ui, r"$\eta$", &mut eta, ETA_RANGE[0] as f32..=ETA_RANGE[1] as f32) {
            self.ask(|s| s.eta = eta as f64);
        }
        ui::note(ui, r"The size of the shapes, and of the first form's steps.");

        ui::section(ui, "The bowl");
        let mut theta = self.glide.to.theta.to_degrees().rem_euclid(180.0) as f32;
        if ui::slider(ui, "turn (degrees)", &mut theta, 0.0..=180.0) {
            self.ask(|s| s.theta = (theta as f64).to_radians());
        }
        let mut kappa = self.glide.to.kappa() as f32;
        if ui::slider_log(ui, r"stretch $\kappa$", &mut kappa, 1.0..=KAPPA_MAX as f32) {
            self.ask(|s| s.ln_kappa = (kappa as f64).max(1.0).ln());
        }
        ui::note(ui, r"The loss is $f(w) = \tfrac12 w^\top H w$, lowest at $0$: its floor runs at the angle chosen, with curvature $1/\kappa$ along it and $1$ across it.");

        ui::section(ui, "A weight matrix");
        if let Some(k) = ui::buttons(ui, &["1 example", "2 examples", "8 examples", "draw again"]) {
            if k < EXAMPLES.len() {
                self.set_examples(k, FIRST_DRAW);
            } else {
                self.set_examples(self.examples, self.draw + 1);
            }
        }
        ui::note(ui, r"A $3 \times 3$ layer's gradient $G = USV^\top$ (scene svd), with three rulers for a matrix: the root of the sum of its squared entries, whose steepest step points along $-G$; its largest entry, whose step is $-\eta \operatorname{sign} G$; and the spectral norm, the most it stretches a vector of length 1, whose step is $-\eta\, UV^\top$, with every singular value of $G$ set to 1. Where $G$ has rank 2 or less, $UV^\top$ is not the only steepest step: in the directions $G$ does not use, any singular value from 0 to 1 will do (the outlined bars), and $UV^\top$ takes 0.");
        ui::note(ui, r"Bernstein and Newhouse (2024) pointed out that, with their running averages switched off, Adam and Shampoo are each steepest descent under a ruler. Adam with $\beta_1 = \beta_2 = 0$ and $\epsilon$ left out steps by $-\eta \operatorname{sign} g$: the first form under the largest entry. Shampoo with its original power, $-\tfrac14$, using only the current gradient (no sum of past ones, $\epsilon = 0$), steps by $-\eta\, UV^\top$ when $G$ has full rank; so does Muon with its momentum switched off, $UV^\top$ computed exactly and no factor for the matrix's shape (scene muon). Muon as it is used takes the first form, applied to its momentum: how big its step is does not depend on how large the gradient is.");

        ui::section(ui, "Reading the picture");
        ui::key(
            ui,
            &[
                (SIGNAL, r"white: the point $w$ and the gradient $g$ there (drawn at $\tfrac34$ of the plane's scale; drag the point or the arrow's tip), and the faint straight lines, the level lines of the tilted plane $f(w) + g \cdot \Delta$"),
                (SGD, "cyan: size by length, a circle; its step in the second form is gradient descent's"),
                (ADAM, "orange: size by the largest entry, a square; its step in the first form is sign descent's"),
                (pal::PINK, "pink: size by the sum of the entries' sizes, a diamond; its step moves one coordinate"),
                (CURVATURE, "green: size by the bowl's curvature, an ellipse shaped like its level lines; its step in the second form is Newton's"),
                (pal::PURPLE, r"purple: size by each coordinate's curvature, an ellipse along the axes ($D$, the diagonal of $H$)"),
                (pal::DIM, "grey ellipses: the bowl's level lines; the cross: its bottom"),
                (ADAM, "orange or pink lines through the bottom, with the square or the diamond alone: where its step is a whole edge; as the point crosses one, the step jumps"),
                (MUON, r"blue, on the matrix panel: $UV^\top$"),
            ],
        );

        ui::section(ui, "Words");
        ui::term(ui, r"Ruler (norm) $\|\Delta\|$:", r"a size for every step, $0$ only for no step, multiplied by $|c|$ when the step is multiplied by $c$, and never more for a sum of two steps than their two sizes added. Its unit shape is the set of steps of size at most 1.");
        ui::term(ui, "Length, largest entry, sum of sizes:", r"$\sqrt{\Delta_1^2 + \Delta_2^2}$, $\max(|\Delta_1|, |\Delta_2|)$ and $|\Delta_1| + |\Delta_2|$. Their unit shapes are the circle, the square and the diamond.");
        ui::term(ui, "Curvature rulers:", r"$\sqrt{\Delta^\top H \Delta}$, with $H$ the bowl's Hessian, or $\sqrt{\Delta^\top D \Delta}$ with only its diagonal, each scaled so that its ellipse has the circle's area (the scale changes how big a step is, not which way it points).");
        ui::term(ui, r"Dual size $\|g\|_*$:", r"the most $g \cdot t$ can be for a step $t$ of size 1: how fast the loss can fall per unit of size. For length it is $|g|$; for the largest entry $|g_1| + |g_2|$; for the sum of sizes $\max(|g_1|, |g_2|)$; for an ellipse $\sqrt{g^\top P^{-1} g}$.");
        ui::term(ui, "$L$:", r"the most the bowl curves over a step of size 1, the largest $t^\top H t$ with $\|t\| = 1$; for length, the largest eigenvalue of $H$.");
        ui::term(ui, "Preconditioning:", r"changing the ruler. Under $\sqrt{\Delta^\top P \Delta}$ the step that makes $g \cdot \Delta + \tfrac{1}{2\eta} \Delta^\top P \Delta$ least is $-\eta P^{-1} g$. Adam's step, $-\eta\, \hat m / (\sqrt{\hat v} + \epsilon)$, has this form, with $P$ diagonal, $\sqrt{\hat v} + \epsilon$ along it, and its average $\hat m$ in place of $g$ (scene adam).");
        ui::term(ui, r"$\operatorname{sign} g$:", r"each entry replaced by $+1$ or $-1$, by its sign (and $0$ by $0$).");
        ui::term(ui, r"Spectral norm, $UV^\top$:", r"the largest singular value of a matrix. From $G = USV^\top$, $UV^\top$ keeps $G$'s singular vectors and sets every singular value to 1 (scene polar).");

        ui::section(ui, "Numbers");
        let s = self.shown();
        let h = s.h();
        let g = h * s.w;
        ui::readout_colored(ui, "$w$", format!("${}$", tuple(&[s.w.x, s.w.y], 2)), SIGNAL);
        ui::readout_colored(ui, "$g = Hw$", format!("${}$", tuple(&[g.x, g.y], 2)), SIGNAL);
        for r in RULERS.iter().filter(|r| self.choice.shows(**r)) {
            let d = step(*r, h, g, s.form(), s.eta);
            ui::readout_colored(ui, &format!("{}: step", r.name()), format!("${}$", tuple(&[d.x, d.y], 2)), r.colour());
            ui::readout_colored(ui, r"$\|g\|_*$ and slope $g \cdot \Delta$", format!("${}$ and ${}$", n2(dual(*r, h, g)), n2(g.dot(d))), r.colour());
        }
        for (r, run) in RULERS.iter().zip(&self.runs) {
            let value = match run.steps {
                Some(1) => "1 step".to_string(),
                Some(n) => format!("{} steps", count(n)),
                None => format!("over {} steps", count(MOST_STEPS)),
            };
            ui::readout_series(ui, r.rule(), value, r.colour());
        }
        let m = &self.mats;
        ui::readout(ui, "$G$", format!("${}$", bmatrix(&rows3(&m.g), 2)));
        ui::readout_colored(ui, r"$\operatorname{sign} G$", format!("${}$", bmatrix(&rows3(&m.sign), 0)), ADAM);
        ui::readout_colored(ui, r"$UV^\top$", format!("${}$", bmatrix(&rows3(&m.ortho), 2)), MUON);
        for (name, sv, c) in [("$G$", m.sv[0], SGD), (r"$\operatorname{sign} G$", m.sv[1], ADAM), (r"$UV^\top$", m.sv[2], MUON)] {
            ui::readout_colored(ui, &format!("singular values of {name}"), format!("$({})$", sv.map(sig).join(", ")), c);
        }
        ui::see_also(
            ui,
            &[
                ("hessian", r"the Hessian $H$ and Newton's step"),
                ("descent", "seven optimisers on surfaces of two variables"),
                ("adam", "Adam: a ruler for each coordinate, from the gradients themselves"),
                ("muon", r"Muon: the step $UV^\top$, for a network's weight matrices"),
                ("polar", r"$UV^\top$, the orthogonal factor of a matrix"),
            ],
        );
    }

    fn frame(&mut self, f: &mut Frame) {
        let view = f.view_size();
        let lay = Layout::of(view, f.top_inset);
        let before = Layout::of(self.canvas, self.inset);
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
        let board = lay.board_for(self.matrix_tab).filter(|_| !covered);

        let pointer = plot::pointer(f);
        if let Some(b) = &board {
            if pointer.is_some_and(|p| inside_rect(p, (b.min, b.max))) {
                f.handles.capture();
                f.handles.claim_clicks();
            }
            if let Some(p) = pointer.filter(|_| f.ptr.clicked) {
                self.board_click(b, p);
            }
        }

        // the handles: the point, and the tip of the gradient's arrow
        let s = self.shown();
        let h = s.h();
        let under = |f: &Frame, p: DVec2| board.as_ref().is_some_and(|b| on_screen(f, v3(p)).is_some_and(|q| inside_rect(q, (b.min, b.max))));
        let near = 5.0 / pixels_per_unit(f).max(1e-3) as f64;
        let mut tip = (s.w + h * s.w * ARROW).as_vec2();
        if (!under(f, s.w + h * s.w * ARROW) || f.handles.dragging()) && f.drag2(2, &mut tip, SIGNAL) {
            // the point whose gradient's arrow ends there: w + ARROW·Hw = tip
            let w = (DMat2::IDENTITY + h * ARROW).inverse() * tip.as_dvec2();
            self.hand_snapped(w, snapped(h, w, near));
        }
        let mut w = s.w.as_vec2();
        if (!under(f, s.w) || f.handles.dragging()) && f.drag2(1, &mut w, pal::WHITE) {
            self.hand_snapped(w.as_dvec2(), snapped(h, w.as_dvec2(), near));
        }
        if !f.handles.dragging() {
            self.gesture = false;
        }
        self.advance(f.dt as f64, f.real_dt as f64);

        let labels_before = f.dl.labels.len();
        let plates = self.picture(f, &lay, covered);
        let numbers_from = f.dl.labels.len();
        f.axes2();
        let mut k = 0;
        f.dl.labels.retain(|l| {
            k += 1;
            k <= numbers_from || !plates.iter().flatten().any(|plate| overlap(label_box(l), *plate))
        });
        if covered {
            f.dl.labels.truncate(labels_before);
            return;
        }
        if let Some(b) = &board {
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

/// The three rows of a 3×3 matrix stored row-major.
fn rows3(m: &[f64; 9]) -> [[f64; 3]; 3] {
    [[m[0], m[1], m[2]], [m[3], m[4], m[5]], [m[6], m[7], m[8]]]
}

impl Steepest {
    fn board_click(&mut self, b: &Board, p: Vec2) {
        for (i, r) in b.rulers.iter().enumerate() {
            if inside_rect(p, *r) {
                self.choice = Choice::BUTTONS[i.min(Choice::BUTTONS.len() - 1)];
            }
        }
        for (i, r) in b.forms.iter().enumerate() {
            if inside_rect(p, *r) {
                self.ask(|s| s.form = i as f64);
            }
        }
        if let Some(tabs) = &b.tabs {
            for (i, r) in tabs.iter().enumerate() {
                if inside_rect(p, *r) {
                    self.matrix_tab = i == 1;
                }
            }
        }
        if let Some(m) = &b.matrix {
            for (i, r) in m.buttons.iter().enumerate() {
                if inside_rect(p, *r) {
                    if i < EXAMPLES.len() {
                        self.set_examples(i, FIRST_DRAW);
                    } else {
                        self.set_examples(self.examples, self.draw + 1);
                    }
                }
            }
        }
    }

    /// Everything drawn in the plane, and the names beside it. Returns the boxes of the
    /// heading and of the caption, which are written over it.
    fn picture(&mut self, f: &mut Frame, lay: &Layout, covered: bool) -> [Option<(Vec2, Vec2)>; 2] {
        let s = self.shown();
        let h = s.h();
        let w = s.w;
        let g = h * w;
        let gn = g.length();
        let form = smooth(s.form);
        for part in 1..PARTS {
            lines_for(f, part);
        }

        // the bowl's level lines, and its bottom
        let u = DVec2::from_angle(s.theta);
        let kappa = s.kappa();
        for k in 1..=8 {
            let c = 0.08 * (k * k) as f64;
            let (a, b) = ((2.0 * c * kappa).sqrt(), (2.0 * c).sqrt());
            let pts: Vec<Vec3> = (0..96).map(|i| {
                let t = i as f64 / 96.0 * 2.0 * PI64;
                v3(u * (a * t.cos()) + u.perp() * (b * t.sin()))
            }).collect();
            f.dl.lines.closed(&pts, 1.0, pal::DIM.a(0.32));
        }
        let cross = 0.09;
        f.line(Vec3::new(-cross, -cross, 0.0), Vec3::new(cross, cross, 0.0), 1.6, pal::DIM);
        f.line(Vec3::new(-cross, cross, 0.0), Vec3::new(cross, -cross, 0.0), 1.6, pal::DIM);

        // the tilted plane round the point: its level lines, chords of a disc, a half
        // step of η apart
        let eta = s.eta;
        let ghat = if gn > 1e-12 { g / gn } else { DVec2::X };
        let reach = 2.3 * eta;
        {
            let batch = lines_for(f, LEVEL_PART);
            for k in -2..=4 {
                let along = -(k as f64) * 0.5 * eta;
                let half = (reach * reach - along * along).max(0.0).sqrt();
                let mid = w + ghat * along;
                let alpha = if gn > 1e-12 { 0.22 } else { 0.0 };
                batch.segment(v3(mid - ghat.perp() * half), v3(mid + ghat.perp() * half), 1.0, pal::WHITE.a(alpha));
            }
        }

        // the lines where the square's or the diamond's answer is an edge (shown with
        // that ruler alone)
        {
            let (lo, hi) = f.cam.flat_bounds();
            let far = (hi - lo).length() as f64 + 10.0;
            let lines = tie_lines(h);
            let square = self.seen[Ruler::Square.index()] * if self.choice == Choice::One(Ruler::Square) { 1.0 } else { 0.0 };
            let diamond = self.seen[Ruler::Diamond.index()] * if self.choice == Choice::One(Ruler::Diamond) { 1.0 } else { 0.0 };
            let batch = lines_for(f, TIE_PART);
            for (k, e) in lines.iter().enumerate() {
                let (c, a) = if k % 2 == 0 { (ADAM, square) } else { (pal::PINK, diamond) };
                if a > 0.01 {
                    batch.segment(v3(-*e * far), v3(*e * far), 1.0, c.a(0.35 * a as f32));
                }
            }
        }

        // each ruler seen: its shape of steps of size η round the point, the level line
        // of the tilted plane that just touches it, and its steepest step
        let mut tips: Vec<(Ruler, DVec2, Set)> = Vec::new();
        for r in RULERS {
            let seen = self.seen[r.index()];
            if seen <= 0.0 {
                continue;
            }
            let c = r.colour().a(seen as f32);
            let grow = 0.85 + 0.15 * seen;
            let shape: Vec<Vec3> = (0..RING).map(|i| v3(w + boundary(r, h, i) * (eta * grow))).collect();
            lines_for(f, shape_part(r)).closed(&shape, 2.0, c);
            let set = steepest(r, h, g);
            // the touching level line
            if gn > 1e-12 {
                let drop = eta * grow * dual(r, h, g) / gn;
                let mid = w - ghat * drop;
                let half = 1.25 * eta;
                lines_for(f, shape_part(r)).segment(v3(mid - ghat.perp() * half), v3(mid + ghat.perp() * half), 1.2, c.a(0.75 * seen as f32));
            }
            // the step as drawn (between the two forms' sizes while the form changes)
            let size_now = (1.0 - form) * eta + form * reach_scaled(r, h, g);
            let d = self.step_drawn(r) * grow;
            let wpp = f.cam.world_per_px(v3(w)) * f.px;
            let edge = !matches!(set, Set::One(_));
            arrow_into(lines_for(f, step_part(r)), wpp, w, w + d, if edge { 2.0 } else { 3.0 }, c);
            match set.scaled(size_now * grow) {
                Set::One(_) => {}
                Set::Edge(a, b) => {
                    // every step to the edge is as steep: the edge, bright, and the fan
                    let batch = lines_for(f, edge_part(r));
                    batch.segment(v3(w + a), v3(w + b), 4.5, c);
                    batch.segment(v3(w), v3(w + a), 1.2, c.a(0.6 * seen as f32));
                    batch.segment(v3(w), v3(w + b), 1.2, c.a(0.6 * seen as f32));
                    f.dl.tris_alpha.tri(v3(w), v3(w + a), v3(w + b), r.colour().a(0.18 * seen as f32));
                }
                Set::All => {
                    for i in 0..RING {
                        f.dl.tris_alpha.tri(v3(w), shape[i], shape[(i + 1) % RING], r.colour().a(0.1 * seen as f32));
                    }
                }
            }
            // the shape's lowest point, which the first form's step ends on (as drawn,
            // gliding from corner to corner; hidden where a whole edge or the whole
            // shape is as low)
            let touch = &self.touch_drawn[r.index()];
            let at = DVec2::new(touch.get(0) as f64, touch.get(1) as f64) * grow;
            f.dot(v3(w + at), 6.0, if matches!(set, Set::One(_)) { c } else { Color::CLEAR });
            if seen > 0.5 {
                tips.push((r, w + d, set));
            }
        }
        for r in RULERS {
            keep_batch(lines_for(f, shape_part(r)));
            keep_batch(lines_for(f, step_part(r)));
            keep_batch(lines_for(f, edge_part(r)));
        }
        keep_batch(lines_for(f, TIE_PART));
        keep_batch(lines_for(f, LEVEL_PART));

        // the gradient, white, at three quarters of the plane's scale
        let tip = w + g * ARROW;
        if gn > 1e-12 {
            f.arrow(v3(w), v3(tip), 2.2, SIGNAL);
        }
        f.dot(v3(w), 7.0, SIGNAL);

        // the heading, and the caption at the foot
        // (broken into lines where the picture is narrow)
        let heading_at = lay.picture.0 + Vec2::new(16.0, 12.0);
        let text = self.heading();
        let size = 15.0;
        let room_x = (lay.picture.1.x - lay.picture.0.x - 40.0).max(80.0);
        let lines = self.lines(f, &text, size, room_x);
        let mut wide: f32 = 0.0;
        for (i, line) in lines.iter().enumerate() {
            wide = wide.max(f.text_size(line, size).x);
            f.caption_px(heading_at + Vec2::new(0.0, i as f32 * (size + 6.0)), line.as_str(), size, pal::TEXT, Align2::LEFT_TOP);
        }
        let tall = lines.len().max(1) as f32 * (size + 6.0) - 6.0;
        let mut plates = [Some((heading_at - Vec2::splat(6.0), heading_at + Vec2::new(wide + 12.0, tall + 10.0))), None];
        if !lay.narrow && f.view_size().y >= 300.0 {
            let parts = [r"each shape: the steps of size $\eta$ under its ruler", "faint straight lines: the loss to first order, a tilted plane", "grey: the bowl"];
            let (left, bottom) = (lay.picture.0.x + 18.0, lay.picture.1.y - 16.0);
            let line = parts.join("      ");
            let wide = f.text_size(&line, 12.0).x;
            if wide <= lay.picture.1.x - lay.picture.0.x - 36.0 {
                f.caption_px(Vec2::new(left, bottom), line, 12.0, pal::DIM, Align2::LEFT_BOTTOM);
                plates[1] = Some((Vec2::new(left - 8.0, bottom - 20.0), Vec2::new(left + wide + 8.0, bottom + 6.0)));
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
        for r in RULERS.iter().filter(|r| self.seen[r.index()] > 0.5) {
            let pts: Vec<Vec3> = (0..RING).step_by(8).map(|i| v3(w + boundary(*r, h, i) * eta)).collect();
            for i in 0..pts.len() {
                room.line(f, pts[i], pts[(i + 1) % pts.len()]);
            }
        }
        for (_, at, _) in &tips {
            room.line(f, v3(w), v3(*at));
        }
        room.line(f, v3(w), v3(tip));
        let away = |p: DVec2| Vec2::new(p.x as f32, -(p.y as f32));
        if gn > 1e-12 {
            room.label(f, "g", v3(tip), "$g$", 15.0, SIGNAL, away(g), 8.0, false);
        }
        room.label(f, "w", v3(w), "$w$", 15.0, SIGNAL, away(-g), 10.0, false);
        // (on a phone, or a window too small for them, the small names are left out)
        let short = lay.narrow || f.view_size().x < 500.0 || f.view_size().y < 400.0;
        if gn > 1e-12 && !short {
            // which way the tilted plane goes down, and the level each shape reaches
            // (each at the same end of its line, an end that turns with g, so that the
            // label never has to cross over to the other)
            room.label(f, "level", v3(w - ghat.perp() * reach), "$f(w)$", 12.0, pal::DIM, away(-ghat), 4.0, true);
            for (r, _, _) in &tips {
                let drop = eta * dual(*r, h, g) / gn;
                let end = w - ghat * drop - ghat.perp() * (1.25 * eta);
                room.label(f, ("level", r.index()), v3(end), level_name(*r), 12.0, r.colour(), away(-ghat), 4.0, true);
            }
        }
        // the lines where the square's or the diamond's step is a whole edge
        for (k, e) in tie_lines(h).iter().enumerate() {
            let (ruler, name) = match k {
                0 => (Ruler::Square, "$g_2 = 0$"),
                1 => (Ruler::Diamond, "$g_1 = g_2$"),
                2 => (Ruler::Square, "$g_1 = 0$"),
                _ => (Ruler::Diamond, "$g_1 = -g_2$"),
            };
            if self.choice == Choice::One(ruler) && self.seen[ruler.index()] > 0.5 {
                let at = [v3(*e * 2.6), v3(-*e * 2.6)];
                room.label_any(f, ("tie", k), &at, name, 12.0, ruler.colour(), Vec2::new(e.y as f32, e.x as f32), 6.0, true);
            }
        }
        for (r, at, set) in &tips {
            let name = tip_name(*r, *set, s.form(), short);
            if !name.is_empty() {
                room.label(f, ("tip", r.index()), v3(*at), name, 13.0, r.colour(), away(*at - w), 8.0, true);
            }
        }
        plates
    }

    /// A text broken into lines no wider than `room` pixels on the screen, its formulas
    /// kept whole (worked out once for each text and width, and kept).
    fn lines(&mut self, f: &Frame, text: &str, size: f32, room: f32) -> Vec<String> {
        let key = (text.to_string(), room.round() as i32);
        if let Some((_, lines)) = self.wrapped.iter().find(|(k, _)| *k == key) {
            return lines.clone();
        }
        let mut words: Vec<String> = Vec::new();
        let (mut word, mut math) = (String::new(), false);
        for ch in text.chars() {
            if ch == '$' {
                math = !math;
            }
            if ch == ' ' && !math {
                if !word.is_empty() {
                    words.push(std::mem::take(&mut word));
                }
            } else {
                word.push(ch);
            }
        }
        if !word.is_empty() {
            words.push(word);
        }
        let mut lines: Vec<String> = Vec::new();
        for word in words {
            match lines.last_mut() {
                Some(last) if f.text_size(&format!("{last} {word}"), size).x <= room => {
                    last.push(' ');
                    last.push_str(&word);
                }
                _ => lines.push(word),
            }
        }
        if self.wrapped.len() > 32 {
            self.wrapped.clear();
        }
        self.wrapped.push((key, lines.clone()));
        lines
    }

    /// The panel on the picture.
    fn board(&mut self, f: &mut Frame, b: &Board) {
        hud_panel(f, b.min, b.max);
        for (i, r) in b.rulers.iter().enumerate() {
            let c = Choice::BUTTONS[i.min(Choice::BUTTONS.len() - 1)];
            plot::button(f, *r, c.label(), SMALL, self.choice == c);
            // a mark in the ruler's colour
            let colour = match c {
                Choice::One(r) => r.colour(),
                Choice::Three => pal::TEXT,
            };
            hud_rect_at(f, 2, Vec2::new(r.0.x + 3.0, r.1.y - 4.0), Vec2::new(r.1.x - 3.0, r.1.y - 2.0), colour.a(0.8));
        }
        let form = self.glide.to.form();
        plot::button(f, b.forms[0], r"step of size $\eta$", SMALL, form == Form::Fixed);
        plot::button(f, b.forms[1], r"size $\|g\|_* / L$", SMALL, form == Form::Scaled);
        let s = self.shown();
        let line = match self.choice {
            Choice::One(r) => formula(r, s.form()).to_string(),
            Choice::Three => match s.form() {
                Form::Fixed => r"the three steps of size $\eta$: along $-g$, to a corner, to a vertex".to_string(),
                Form::Scaled => r"each step of size $\|g\|_* / L$, measured by its own ruler".to_string(),
            },
        };
        let room = b.max.x - b.formula.x - PAD;
        for (i, text) in self.lines(f, &line, 12.0, room).iter().enumerate() {
            f.text_px(b.formula + Vec2::new(0.0, i as f32 * LINE_H), text.as_str(), 12.0, pal::TEXT, Align2::LEFT_TOP);
        }
        if let Some(tabs) = &b.tabs {
            plot::button(f, tabs[0], "paths down the bowl", SMALL, !self.matrix_tab);
            plot::button(f, tabs[1], "a weight matrix", SMALL, self.matrix_tab);
        }
        if let Some(p) = &b.paths {
            self.paths_chart(f, p);
        }
        if let Some(m) = &b.matrix {
            self.matrix_chart(f, m);
        }
    }

    /// The same bowl from above, and each ruler's path down it from the point.
    fn paths_chart(&self, f: &mut Frame, p: &PathsBox) {
        let s = self.shown();
        let title = r"From $w$ to the bottom, each ruler's steps of size $\|g\|_* / L$";
        f.text_px(p.title, title, SMALL, pal::TEXT, Align2::LEFT_TOP);
        let (min, max) = p.chart;
        let (wpx, hpx) = ((max.x - min.x).max(1.0) as f64, (max.y - min.y).max(1.0) as f64);
        let half = FRAME_HALF;
        let (hx, hy) = if wpx >= hpx { (half * wpx / hpx, half) } else { (half, half * hpx / wpx) };
        let chart = Plot::new(p.chart, [-hx, hx], [-hy, hy]);
        if chart.tiny() {
            return;
        }
        plot::fill(f, p.chart, Color::hex(0x0E1322));
        let edge = [min, Vec2::new(max.x, min.y), max, Vec2::new(min.x, max.y), min];
        plot::path_of(f, plot::AXES, &edge, 1.0, pal::DIM.a(0.4));
        // the bowl's level lines
        let u = DVec2::from_angle(s.theta);
        let kappa = s.kappa();
        // (everything here is kept to the chart by moving what lies outside onto its edge,
        // not by cutting it there: the lines then keep their points from frame to frame)
        let inside = |x: f64, y: f64| chart.at(x, y).max(min).min(max);
        for k in 1..=8 {
            let c = 0.08 * (k * k) as f64;
            let (a, b) = ((2.0 * c * kappa).sqrt(), (2.0 * c).sqrt());
            let pts: Vec<Vec2> = (0..=72)
                .map(|i| {
                    let t = i as f64 / 72.0 * 2.0 * PI64;
                    let q = u * (a * t.cos()) + u.perp() * (b * t.sin());
                    inside(q.x, q.y)
                })
                .collect();
            plot::path_of(f, CHART_BOWL, &pts, 1.0, pal::DIM.a(0.3));
        }
        let bottom = chart.at(0.0, 0.0);
        plot::line_of(f, plot::MARKS, bottom - Vec2::splat(4.0), bottom + Vec2::splat(4.0), 1.4, pal::DIM);
        plot::line_of(f, plot::MARKS, bottom + Vec2::new(-4.0, 4.0), bottom + Vec2::new(4.0, -4.0), 1.4, pal::DIM);
        // the paths, the chosen rulers' last so that they lie on top
        let mut order: Vec<Ruler> = RULERS.to_vec();
        order.sort_by_key(|r| self.choice.shows(*r));
        for r in order {
            let pts: Vec<(f64, f64)> = self.paths_drawn[r.index()].shown.chunks_exact(2).map(|q| (q[0] as f64, q[1] as f64)).collect();
            let Some(last) = pts.last().copied() else { continue };
            let bright = if self.choice.shows(r) { 1.0 } else { 0.55 };
            let line: Vec<Vec2> = pts.iter().map(|(x, y)| inside(*x, *y)).collect();
            plot::path_of(f, chart_path(r), &line, if bright == 1.0 { 1.8 } else { 1.2 }, r.colour().a(bright));
            let dots = &self.dots_drawn[r.index()].shown;
            for i in 0..DOTS {
                let (x, y) = (dots.get(2 * i).copied().unwrap_or(last.0 as f32), dots.get(2 * i + 1).copied().unwrap_or(last.1 as f32));
                plot::disc(f, inside(x as f64, y as f64), 4.0, r.colour().a(bright));
            }
        }
        plot::disc(f, inside(s.w.x, s.w.y), 7.0, SIGNAL);
        // the legend: each ruler's count
        let row = 16.0;
        let col_w = (max.x - min.x) / p.columns as f32;
        for (k, r) in RULERS.iter().enumerate() {
            let (cx, cy) = ((k % p.columns) as f32, (k / p.columns) as f32);
            let at = p.legend + Vec2::new(cx * col_w, cy * row);
            hud_rect(f, at + Vec2::new(0.0, 4.0), at + Vec2::new(9.0, 12.0), r.colour());
            let Some(run) = self.runs.get(k) else { continue };
            let what = match run.steps {
                Some(1) => "1 step".to_string(),
                Some(n) => format!("{} steps", count(n)),
                None => format!("over {} steps", count(MOST_STEPS)),
            };
            let text = if p.columns == 1 { format!("{}, {}: {what}", r.name(), r.rule()) } else { format!("{}: {what}", r.name()) };
            f.text_px(at + Vec2::new(14.0, 0.0), text, SMALL, pal::TEXT, Align2::LEFT_TOP);
        }
    }

    /// The singular values of the 3×3 gradient and of its three steepest steps.
    fn matrix_chart(&self, f: &mut Frame, m: &MatrixBox) {
        let x = m.title.x;
        f.text_px(m.title, r"A $3 \times 3$ weight matrix: the singular values of", SMALL, pal::TEXT, Align2::LEFT_TOP);
        f.text_px(m.title + Vec2::new(0.0, LINE_H), "its gradient and of the three steepest steps, each over its largest", SMALL, pal::TEXT, Align2::LEFT_TOP);
        let chart = Plot::new(m.chart, [0.0, 3.0], [0.0, 1.0]);
        if chart.tiny() {
            return;
        }
        chart.y_grid(f);
        chart.y_axis(f, "", 10.0);
        let mats = &self.mats;
        let groups = [(0usize, SGD), (1, ADAM), (2, MUON)];
        for (gi, c) in groups {
            let sv = mats.sv[gi];
            let top = sv[0].max(1e-300);
            for k in 0..3 {
                let x0 = gi as f64 + 0.12 + 0.26 * k as f64;
                let v = if sv[0] > 0.0 { sv[k] / top } else { 0.0 };
                chart.bar(f, x0, x0 + 0.22, 0.0, v, 1.0, c.a(0.85));
                // where the step is not one matrix: any size up to 1 in the directions
                // the gradient does not use
                if gi == 2 && k >= mats.rank {
                    let (a, b) = (chart.at(x0, 0.0), chart.at(x0 + 0.22, 1.0));
                    let pts = [Vec2::new(a.x + 1.0, a.y), Vec2::new(a.x + 1.0, b.y), Vec2::new(b.x - 1.0, b.y), Vec2::new(b.x - 1.0, a.y)];
                    plot::path_of(f, plot::MARKS, &pts, 1.0, c.a(0.6));
                }
            }
        }
        let (min, max) = m.chart;
        let third = (max.x - min.x) / 3.0;
        let names = [("$G$", "gradient descent"), (r"$\operatorname{sign} G$", "sign descent"), (r"$UV^\top$", "spectral norm")];
        for (i, (name, what)) in names.iter().enumerate() {
            let cx = min.x + third * (i as f32 + 0.5);
            f.text_px(Vec2::new(cx, max.y + 4.0), *name, 12.0, groups[i].1, Align2::CENTER_TOP);
            f.text_px(Vec2::new(cx, max.y + 4.0 + LINE_H), *what, 10.5, pal::DIM, Align2::CENTER_TOP);
        }
        if self.attention > 0.0 {
            // an outline that fades as the moment passes
            let fade = smooth(self.attention / ATTENTION_S) as f32;
            let end = m.buttons.last().map_or(m.chart.1, |r| r.1);
            let (lo, hi) = (m.title - Vec2::new(6.0, 6.0), end + Vec2::new(6.0, 6.0));
            let edge = [lo, Vec2::new(hi.x, lo.y), hi, Vec2::new(lo.x, hi.y), lo];
            plot::path_of(f, plot::OWN + 7, &edge, 2.0, pal::LINK.a(0.9 * fade));
        }
        keep_batch(plot::apart(f, plot::OWN + 7, false));
        let labels = ["1 example", "2 examples", "8 examples", "draw again"];
        for (i, r) in m.buttons.iter().enumerate() {
            plot::button(f, *r, labels[i], SMALL, i == self.examples);
        }
        let _ = x;
    }
}

/// The second form's size for the ruler (the first's is η).
fn reach_scaled(r: Ruler, h: DMat2, g: DVec2) -> f64 {
    reach(r, h, g, Form::Scaled, 0.0)
}

/// An arrow into a batch of its own, its head fixed in pixels (as `Frame::arrow` draws one).
fn arrow_into(batch: &mut crate::gfx::draw::LineBatch, wpp: f32, from: DVec2, to: DVec2, width: f32, c: Color) {
    let (a, b) = (v3(from), v3(to));
    let d = b - a;
    let len = d.length();
    if len < 1e-6 {
        batch.segment(a, a, 0.0, Color::CLEAR);
        batch.segment(a, a, 0.0, Color::CLEAR);
        batch.segment(a, a, 0.0, Color::CLEAR);
        return;
    }
    let dir = d / len;
    let head = ((width * 3.2 + 5.0) * wpp).min(len * 0.6);
    let side = Vec3::new(-dir.y, dir.x, 0.0) * head * 0.42;
    let base = b - dir * head;
    batch.segment(a, b - dir * (head * 0.8), width, c);
    batch.segment(b, base + side, width, c);
    batch.segment(b, base - side, width, c);
}

/// The level of the tilted plane that a ruler's shape just touches: the loss to first
/// order there, η times the gradient's dual size below f(w).
fn level_name(r: Ruler) -> &'static str {
    match r {
        Ruler::Round => r"$f - \eta\,|g|$",
        Ruler::Square => r"$f - \eta\,(|g_1| + |g_2|)$",
        Ruler::Diamond => r"$f - \eta \max(|g_1|, |g_2|)$",
        Ruler::Bowl | Ruler::Axes => r"$f - \eta\,\|g\|_*$",
    }
}

/// What the end of a ruler's step is called.
fn tip_name(r: Ruler, set: Set, form: Form, short: bool) -> &'static str {
    if short {
        return "";
    }
    match (r, set, form) {
        (_, Set::All, _) => "",
        (_, Set::Edge(..), _) => "the whole edge",
        (Ruler::Round, _, Form::Fixed) => r"$-\eta\, g/|g|$",
        (Ruler::Round, _, Form::Scaled) => r"$-g/L$",
        (Ruler::Square, _, Form::Fixed) => r"$-\eta \operatorname{sign} g$",
        (Ruler::Square, _, Form::Scaled) => r"$\propto -\operatorname{sign} g$",
        (Ruler::Diamond, _, Form::Fixed) => "a vertex",
        (Ruler::Diamond, _, Form::Scaled) => "one coordinate moves",
        (Ruler::Bowl, _, Form::Fixed) => "towards the bottom",
        (Ruler::Bowl, _, Form::Scaled) => r"$-H^{-1} g$",
        (Ruler::Axes, _, _) => r"$\propto -D^{-1} g$",
    }
}

/// The captions of the things to try (each held by a test below).
fn captions(k: usize) -> &'static str {
    const C: [&str; 5] = [
        r"The point goes once round a round bowl, so the gradient $g$ turns steadily. The cyan step, along $-g$, turns with it. The orange one stays on a corner of the square, $-\eta \operatorname{sign} g$, and the pink one on a vertex of the diamond, each until it jumps to the next. The point stops where they jump: on each axis, where one entry of $g$ is $0$ and a whole edge of the square is steepest, and on each diagonal, where $|g_1| = |g_2|$ and a whole edge of the diamond is.",
        r"A valley along the $x$ axis, 10 times as curved across as along. In the paths chart each ruler takes steps of size $\|g\|_* / L$ until the loss is below a millionth of its start: gradient descent needs 62, sign descent 31, the diamond 63, and both ellipses 1. The purple one along the axes is the green one here: for a bowl lined up with the axes, its diagonal is all of $H$.",
        r"The same valley and start, turned by 45°. Gradient descent takes the same 62 steps, turned with the picture: the circle looks the same turned. The green ellipse turns with the bowl and still needs 1. The purple one cannot turn: the bowl's diagonal is now the same in both coordinates, so it is a circle and takes gradient descent's 62. Sign descent now needs 63 and the diamond 31, the other way round: turned by 45°, a square is a diamond.",
        r"The green ellipse has the shape of the bowl's level lines. Its steepest step points straight at the bottom, and in the second form it is Newton's step, $-H^{-1} g$, which lands there: 1 step in the paths chart. The bowl now turns a quarter turn under the point: the ellipse turns with it, the gradient swings round, and the step still ends on the bottom all the way.",
        r"A layer's gradient $G$, the average over 8 examples of one outer product each. Its singular values are $0.37$, $0.08$ and $3 \times 10^{-4}$ (cyan, each over the largest): a gradient step moves the weights mostly along one direction, and hardly at all along the third. Under the largest entry the steepest step is $\operatorname{sign} G$ (orange), with $2.56$, $1.56$ and $0$: two directions, and none at all along the third. Under the spectral norm it is $UV^\top$ (blue): $1$, $1$ and $1$, every direction alike. With 1 example $G$ has rank 1, and so do its signs and $UV^\top$.",
    ];
    C[k.min(C.len() - 1)]
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

    fn bowls() -> Vec<DMat2> {
        let mut out = Vec::new();
        for theta in [0.0, 0.3, 45.0f64.to_radians(), 1.2, -0.7, PI64 / 2.0] {
            for kappa in [1.0, 2.0, 10.0, 100.0] {
                out.push(hessian(theta, kappa));
            }
        }
        out
    }

    fn gradients() -> Vec<DVec2> {
        let mut rng = Rng::new(3);
        let mut out: Vec<DVec2> = (0..60).map(|_| DVec2::new(rng.normal() as f64, rng.normal() as f64) * 1.3).collect();
        out.extend([DVec2::X, DVec2::Y, -DVec2::X, DVec2::new(1.0, 1.0), DVec2::new(-0.5, 0.5), DVec2::new(1e-3, 2.0), DVec2::new(3.0, -2.999)]);
        out
    }

    /// For every ruler, bowl and gradient: the steepest steps have size 1 and make g · Δ
    /// as low as any step of size 1 can (found by trying 7200 of them round the unit
    /// shape), and that lowest value is minus the dual size.
    #[test]
    fn the_steepest_step_is_the_lowest_point_of_the_unit_shape() {
        for h in bowls() {
            for g in gradients() {
                for r in RULERS {
                    let lowest = (0..7200)
                        .map(|i| {
                            let d = DVec2::from_angle(i as f64 / 7200.0 * 2.0 * PI64);
                            g.dot(d / size(r, h, d))
                        })
                        .fold(f64::MAX, f64::min);
                    let set = steepest(r, h, g);
                    let ends: Vec<DVec2> = match set {
                        Set::One(d) => vec![d],
                        Set::Edge(a, b) => vec![a, b, (a + b) * 0.5],
                        Set::All => panic!("{r:?} {g:?}"),
                    };
                    for d in ends {
                        assert!((size(r, h, d) - 1.0).abs() < 1e-12, "{r:?} {g:?} {d:?}");
                        assert!(g.dot(d) <= lowest + 1e-12, "{r:?} {g:?}: {} > {lowest}", g.dot(d));
                        assert!((g.dot(d) + dual(r, h, g)).abs() < 1e-12, "{r:?} {g:?}");
                    }
                    // and nothing on the shape is lower (to within the sampling's grain)
                    assert!(lowest >= -dual(r, h, g) - 1e-12 && lowest <= -dual(r, h, g) + 1e-4 * g.length(), "{r:?} {g:?}");
                }
            }
        }
    }

    /// The circle's answer is −g/|g|; the square's the signs; the diamond's the larger
    /// entry; the bowl's ellipse points at the bottom, along −H⁻¹g = −w.
    #[test]
    fn each_ruler_gives_the_step_its_caption_names() {
        for h in bowls() {
            for g in gradients() {
                let unit = |r: Ruler| steepest(r, h, g).middle();
                assert!((unit(Ruler::Round) + g / g.length()).length() < 1e-12);
                if g.x.abs() > 1e-6 && g.y.abs() > 1e-6 {
                    assert_eq!(unit(Ruler::Square), DVec2::new(-g.x.signum(), -g.y.signum()));
                }
                if (g.x.abs() - g.y.abs()).abs() > 1e-6 {
                    let d = unit(Ruler::Diamond);
                    assert!(if g.x.abs() > g.y.abs() { d == DVec2::new(-g.x.signum(), 0.0) } else { d == DVec2::new(0.0, -g.y.signum()) });
                }
                let w = h.inverse() * g;
                let b = unit(Ruler::Bowl);
                assert!(b.perp_dot(w).abs() < 1e-9 * w.length().max(1.0) && b.dot(w) < 0.0, "points at the bottom");
                let d = DVec2::new(h.x_axis.x, h.y_axis.y);
                let a = unit(Ruler::Axes);
                assert!(a.perp_dot(g / d).abs() < 1e-9 * (g / d).length() && a.dot(g / d) < 0.0);
            }
        }
    }

    /// Where the answer is not one point: a gradient along an axis (the square), or with
    /// |g₁| = |g₂| (the diamond), or 0 (every ruler).
    #[test]
    fn ties_give_a_whole_edge() {
        let h = hessian(0.3, 4.0);
        assert_eq!(steepest(Ruler::Square, h, DVec2::new(0.0, 2.0)), Set::Edge(DVec2::new(-1.0, -1.0), DVec2::new(1.0, -1.0)));
        assert_eq!(steepest(Ruler::Square, h, DVec2::new(-0.5, 1e-12)), Set::Edge(DVec2::new(1.0, -1.0), DVec2::new(1.0, 1.0)));
        assert_eq!(steepest(Ruler::Diamond, h, DVec2::new(0.7, -0.7)), Set::Edge(DVec2::new(-1.0, 0.0), DVec2::new(0.0, 1.0)));
        for r in RULERS {
            assert_eq!(steepest(r, h, DVec2::ZERO), Set::All);
            assert_eq!(step(r, h, DVec2::ZERO, Form::Scaled, 0.5), DVec2::ZERO);
            assert_eq!(steepest(r, h, DVec2::new(f64::NAN, 1.0)), Set::All);
        }
        // every point of an edge is as low as every other
        for g in [DVec2::new(0.0, 2.0), DVec2::new(1.5, 0.0)] {
            if let Set::Edge(a, b) = steepest(Ruler::Square, h, g) {
                for i in 0..=10 {
                    let d = a + (b - a) * (i as f64 / 10.0);
                    assert!((g.dot(d) + dual(Ruler::Square, h, g)).abs() < 1e-12 && (size(Ruler::Square, h, d) - 1.0).abs() < 1e-12);
                }
            }
        }
    }

    /// L is the most tᵀHt over the steps of size 1, and with it the quadratic model is
    /// never below the bowl; the second form's step makes that model least, and so never
    /// raises the loss.
    #[test]
    fn the_bound_and_the_second_form_are_right() {
        for h in bowls() {
            for r in RULERS {
                let most = (0..7200)
                    .map(|i| {
                        let d = DVec2::from_angle(i as f64 / 7200.0 * 2.0 * PI64);
                        let t = d / size(r, h, d);
                        t.dot(h * t)
                    })
                    .fold(0.0, f64::max);
                let l = bound(r, h);
                assert!(l >= most - 1e-12 && l <= most * (1.0 + 1e-5), "{r:?}: {l} against {most}");
                for g in gradients() {
                    let w = h.inverse() * g;
                    let d = step(r, h, g, Form::Scaled, 0.0);
                    let model = |d: DVec2| g.dot(d) + 0.5 * l * size(r, h, d).powi(2);
                    // least among many steps
                    for i in 0..720 {
                        for t in [0.1, 0.5, 1.0, 2.0] {
                            let e = DVec2::from_angle(i as f64 / 720.0 * 2.0 * PI64) * t * d.length().max(0.1);
                            assert!(model(d) <= model(e) + 1e-12, "{r:?}");
                        }
                    }
                    assert!(loss(h, w + d) <= loss(h, w) + 1e-12, "{r:?} raised the loss");
                    assert!((model(d) + 0.5 * dual(r, h, g).powi(2) / l).abs() < 1e-12 * (1.0 + dual(r, h, g).powi(2) / l));
                }
            }
        }
    }

    /// The second form under the circle is gradient descent with step 1/λ_max; under the
    /// bowl's ellipse it is Newton's step, which lands on the bottom; the first form's
    /// step is always of size η.
    #[test]
    fn the_second_form_is_gradient_descent_and_newton() {
        for h in bowls() {
            let top = bound(Ruler::Round, h);
            for g in gradients() {
                assert!((step(Ruler::Round, h, g, Form::Scaled, 0.0) + g / top).length() < 1e-12);
                let w = h.inverse() * g;
                assert!((w + step(Ruler::Bowl, h, g, Form::Scaled, 0.0)).length() < 1e-9 * w.length().max(1.0));
                for r in RULERS {
                    assert!((size(r, h, step(r, h, g, Form::Fixed, 0.37)) - 0.37).abs() < 1e-12);
                }
            }
        }
    }

    /// Gradient descent does not care how the picture is turned; Newton's step does not
    /// care how it is turned or stretched; and turned by 45°, the square is the diamond.
    #[test]
    fn turning_the_picture() {
        let w0 = DVec2::from(VALLEY_W);
        let flat = hessian(0.0, 10.0);
        let turn = DVec2::from_angle(PI64 / 4.0);
        let tilted = hessian(PI64 / 4.0, 10.0);
        let a = run(Ruler::Round, flat, w0);
        let b = run(Ruler::Round, tilted, turn.rotate(w0));
        assert_eq!(a.steps, b.steps);
        assert!(a.pts.iter().zip(&b.pts).all(|(p, q)| (turn.rotate(*p) - *q).length() < 1e-9));
        let sq = run(Ruler::Square, tilted, turn.rotate(w0));
        let di = run(Ruler::Diamond, flat, w0);
        assert_eq!(sq.steps, di.steps);
        assert!(sq.pts.iter().zip(&di.pts).all(|(p, q)| (*p - turn.rotate(*q)).length() < 1e-9));
        for h in bowls() {
            assert_eq!(run(Ruler::Bowl, h, DVec2::new(2.0, -1.0)).steps, Some(1));
        }
    }

    fn settle(scene: &mut Steepest) {
        for _ in 0..60 * 40 {
            scene.advance(1.0 / 60.0, 1.0 / 60.0);
            if scene.at_rest() {
                break;
            }
        }
        assert!(scene.at_rest());
    }

    /// The scene as an experiment's button leaves it, from another experiment's state.
    fn after(k: usize) -> Steepest {
        let mut scene = Steepest::new();
        scene.set_up((k + 2) % TRIALS.len());
        settle(&mut scene);
        scene.set_up(k);
        settle(&mut scene);
        scene
    }

    fn counts(scene: &Steepest) -> Vec<Option<usize>> {
        scene.runs.iter().map(|r| r.steps).collect()
    }

    #[test]
    fn turn_the_gradient_does_what_its_caption_says() {
        let mut scene = Steepest::new();
        scene.set_up(0);
        // the film starts once the glide is there
        let mut frames = 0;
        while scene.film.is_none() {
            scene.advance(1.0 / 60.0, 1.0 / 60.0);
            frames += 1;
            assert!(frames < 600);
        }
        let mut last: Option<[DVec2; 3]> = None;
        let (mut corner_jumps, mut vertex_jumps, mut edges_sq, mut edges_di) = (0, 0, 0, 0);
        let mut was_edge = [false, false];
        while scene.film.is_some() {
            scene.advance(1.0 / 60.0, 1.0 / 60.0);
            let s = scene.shown();
            let h = s.h();
            assert!((h - DMat2::IDENTITY).to_cols_array().iter().all(|v| v.abs() < 1e-12) && (s.w.length() - ROUND_R).abs() < 1e-9);
            let g = h * s.w;
            let now = [steepest(Ruler::Round, h, g), steepest(Ruler::Square, h, g), steepest(Ruler::Diamond, h, g)];
            // the circle's step turns with −g, never far in a frame
            assert!((now[0].middle() + g / g.length()).length() < 1e-12);
            let sq_edge = matches!(now[1], Set::Edge(..));
            let di_edge = matches!(now[2], Set::Edge(..));
            if sq_edge && !was_edge[0] {
                edges_sq += 1;
            }
            if di_edge && !was_edge[1] {
                edges_di += 1;
            }
            let was_edge_before = was_edge;
            was_edge = [sq_edge, di_edge];
            let mids = [now[0].middle(), now[1].middle(), now[2].middle()];
            if let Some(before) = last {
                assert!((mids[0] - before[0]).length() < 0.05);
                // a corner or a vertex changes only on its way onto an edge or off it
                if (mids[1] - before[1]).length() > 1e-9 {
                    corner_jumps += 1;
                    assert!(sq_edge != was_edge_before[0]);
                }
                if (mids[2] - before[2]).length() > 1e-9 {
                    vertex_jumps += 1;
                    assert!(di_edge != was_edge_before[1]);
                }
            }
            if !sq_edge {
                assert!(matches!(now[1], Set::One(d) if d.x.abs() == 1.0 && d.y.abs() == 1.0), "a corner");
            }
            if !di_edge {
                assert!(matches!(now[2], Set::One(d) if d.x.abs() + d.y.abs() == 1.0 && d.x * d.y == 0.0), "a vertex");
            }
            last = Some(mids);
        }
        // four stops on the axes, four on the diagonals; a corner changes twice at each
        // axis (onto the edge's middle and off it), a vertex at each diagonal
        assert_eq!((edges_sq, edges_di), (4, 4));
        assert_eq!((corner_jumps, vertex_jumps), (8, 8));
        assert!(scene.at_rest() || !scene.glide.arrived() || scene.film.is_none());
    }

    #[test]
    fn the_valley_along_an_axis_does_what_its_caption_says() {
        let scene = after(1);
        let s = scene.shown();
        assert!(s.theta == 0.0 && (s.kappa() - 10.0).abs() < 1e-12 && s.form() == Form::Scaled);
        assert_eq!(counts(&scene), vec![Some(62), Some(31), Some(63), Some(1), Some(1)]);
        let h = s.h();
        // the axes' ruler is the bowl's
        assert!((quadratic(Ruler::Axes, h) - quadratic(Ruler::Bowl, h)).to_cols_array().iter().all(|v| v.abs() < 1e-12));
    }

    #[test]
    fn the_turned_valley_does_what_its_caption_says() {
        let scene = after(2);
        let s = scene.shown();
        assert!((s.theta - PI64 / 4.0).abs() < 1e-12);
        assert_eq!(counts(&scene), vec![Some(62), Some(63), Some(31), Some(1), Some(62)]);
        let h = s.h();
        // the diagonal is the same in both coordinates: the purple ruler is the circle
        assert!((quadratic(Ruler::Axes, h) - DMat2::IDENTITY).to_cols_array().iter().all(|v| v.abs() < 1e-12));
        assert!(scene.runs[0].pts.iter().zip(&scene.runs[4].pts).all(|(p, q)| (*p - *q).length() < 1e-9));
    }

    #[test]
    fn newtons_ellipse_does_what_its_caption_says() {
        let mut scene = Steepest::new();
        scene.set_up(3);
        let mut seen_film = false;
        for _ in 0..60 * 30 {
            scene.advance(1.0 / 60.0, 1.0 / 60.0);
            seen_film |= scene.film.is_some();
            let s = scene.shown();
            if scene.film.is_some() {
                let h = s.h();
                // the step ends on the bottom all the way, and the ellipse is the bowl's
                let d = step(Ruler::Bowl, h, h * s.w, Form::Scaled, s.eta);
                assert!((s.w + d).length() < 1e-9);
                assert_eq!(scene.runs[3].steps, Some(1));
            }
            if seen_film && scene.at_rest() {
                break;
            }
        }
        assert!(seen_film && scene.at_rest());
        let s = scene.shown();
        assert!((s.theta - (15.0f64.to_radians() + TURN_BY)).abs() < 1e-9);
    }

    #[test]
    fn the_weight_matrix_does_what_its_caption_says() {
        let scene = after(4);
        let m = &scene.mats;
        assert_eq!(EXAMPLES[scene.examples], 8);
        assert_eq!([num(m.sv[0][0], 2), num(m.sv[0][1], 2), num(m.sv[0][2] * 1e4, 0)], ["0.37", "0.08", "3"], "{:?}", m.sv[0]);
        assert_eq!([num(m.sv[1][0], 2), num(m.sv[1][1], 2)], ["2.56", "1.56"], "{:?}", m.sv[1]);
        assert!(m.sv[2].iter().all(|v| (v - 1.0).abs() < 1e-9) && m.rank == 3);
        // (a step moves the weights along the directions of its singular values, by them)
        assert!(m.sv[0][1] / m.sv[0][0] < 0.25 && m.sv[0][2] / m.sv[0][0] < 1e-3 && m.sv[1][2] < 1e-12);
        // with 1 example: rank 1, its signs too, and UVᵀ
        let one = steps3(gradient3(EXAMPLES[0], FIRST_DRAW));
        assert_eq!(one.rank, 1);
        assert!(one.sv[1][1] < 1e-9 && one.sv[1][0] > 1.0 && one.sv[2][1] < 1e-9 && (one.sv[2][0] - 1.0).abs() < 1e-9, "{:?}", one.sv);
    }

    /// The matrix steps: each is steepest under its ruler (it makes ⟨G, T⟩ as large as any
    /// T of size 1 among many tried), its size under its ruler is 1, and ⟨G, T⟩ is the
    /// dual size.
    #[test]
    fn the_matrix_steps_are_steepest() {
        let mut rng = Rng::new(11);
        let inner = |a: &[f64; 9], b: &[f64; 9]| (0..9).map(|i| a[i] * b[i]).sum::<f64>();
        let spectral = |a: &[f64; 9]| singular_values(a)[0];
        for draw in 0..20 {
            let m = steps3(gradient3(EXAMPLES[draw % 3], draw as u64));
            assert!((inner(&m.g, &m.frob) - m.duals[0]).abs() < 1e-12);
            assert!((inner(&m.g, &m.sign) - m.duals[1]).abs() < 1e-12);
            assert!((inner(&m.g, &m.ortho) - m.duals[2]).abs() < 1e-9 * m.duals[2]);
            assert!((spectral(&m.ortho) - 1.0).abs() < 1e-9);
            for _ in 0..300 {
                let t: [f64; 9] = std::array::from_fn(|_| rng.normal() as f64);
                let fro = t.iter().map(|x| x * x).sum::<f64>().sqrt();
                let big = t.iter().fold(0.0f64, |a, x| a.max(x.abs()));
                assert!(inner(&m.g, &t.map(|x| x / fro)) <= m.duals[0] + 1e-12);
                assert!(inner(&m.g, &t.map(|x| x / big)) <= m.duals[1] + 1e-12);
                let sp = spectral(&t);
                assert!(inner(&m.g, &t.map(|x| x / sp)) <= m.duals[2] + 1e-9);
            }
            // UVᵀ keeps G's singular vectors: G Vᵀ... G·(UVᵀ)ᵀ is symmetric with no negative eigenvalue
            assert_eq!(m.rank, if EXAMPLES[draw % 3] == 1 { 1 } else if EXAMPLES[draw % 3] == 2 { 2 } else { 3 });
        }
    }

    /// For a gradient of rank r < 3, UᵣVᵣᵀ is one of many steepest steps under the
    /// spectral norm: adding any multiple up to 1 of the pairs of singular vectors G does
    /// not use changes neither ⟨G, T⟩ nor the size; with 1 example the gradient's signs
    /// are rank 1 too.
    #[test]
    fn a_gradient_of_short_rank_has_many_steepest_steps() {
        let inner = |a: &[f64; 9], b: &[f64; 9]| (0..9).map(|i| a[i] * b[i]).sum::<f64>();
        for examples in [1, 2] {
            for draw in 1..6 {
                let m = steps3(gradient3(examples, draw));
                assert_eq!(m.rank, examples);
                let (u, _, v) = linalg::svd(&m.g, 3, 3);
                for z in [0.0, 0.3, -0.7, 1.0] {
                    let mut t = m.ortho;
                    for k in m.rank..3 {
                        for i in 0..3 {
                            for j in 0..3 {
                                t[i * 3 + j] += z * u[i * 3 + k] * v[j * 3 + k];
                            }
                        }
                    }
                    assert!((inner(&m.g, &t) - m.duals[2]).abs() < 1e-9 * m.duals[2]);
                    assert!((singular_values(&t)[0] - 1.0).abs() < 1e-9);
                }
                if examples == 1 {
                    assert!(m.sv[1][1] < 1e-9 && m.zeros == 0);
                }
            }
        }
    }

    #[test]
    fn degenerate_input_leaves_everything_finite_and_comes_to_rest() {
        let mut scene = Steepest::new();
        let extremes = [DVec2::ZERO, DVec2::new(W_MAX, W_MAX), DVec2::new(-W_MAX, 0.0), DVec2::new(1e-14, 0.0), DVec2::new(f64::NAN, 1.0)];
        for w in extremes {
            for kappa in [1.0, KAPPA_MAX, 1e9, f64::NAN] {
                for choice in Choice::BUTTONS {
                    for form in [0.0, 1.0] {
                        scene.choice = choice;
                        scene.ask(|s| {
                            s.ln_kappa = kappa.ln();
                            s.form = form;
                            s.eta = ETA_RANGE[1] * 5.0;
                        });
                        scene.hand_snapped(w, w);
                        settle(&mut scene);
                        let s = scene.shown();
                        assert!(s.numbers().iter().all(|v| v.is_finite()));
                        let h = s.h();
                        for r in RULERS {
                            let d = step(r, h, h * s.w, s.form(), s.eta);
                            assert!(d.is_finite() && dual(r, h, h * s.w).is_finite() && bound(r, h).is_finite());
                            for i in 0..RING {
                                assert!(boundary(r, h, i).is_finite());
                            }
                        }
                        assert!(scene.runs.iter().all(|r| r.pts.iter().all(|p| p.is_finite())));
                        assert!(!scene.heading().contains("NaN"));
                    }
                }
            }
        }
    }

    #[test]
    fn the_layout_fits_every_canvas() {
        for (w, h) in [(1240.0, 1000.0), (1100.0, 700.0), (390.0, 800.0), (390.0, 560.0), (800.0, 600.0), (1600.0, 1000.0), (240.0, 180.0), (620.0, 480.0), (1300.0, 700.0)] {
            for inset in [0.0, 46.0] {
                let view = Vec2::new(w, h);
                let lay = Layout::of(view, inset);
                let (min, max) = lay.picture;
                assert!(min.cmpge(Vec2::ZERO).all() && max.cmple(view).all() && (max - min).min_element() > 50.0, "{w}x{h}: {lay:?}");
                for tab in [false, true] {
                    if let Some(b) = lay.board_for(tab) {
                        let within = |q: Vec2| q.cmpge(b.min - Vec2::splat(0.5)).all() && q.cmple(b.max + Vec2::splat(0.5)).all();
                        assert!(b.min.cmpge(Vec2::ZERO).all() && b.max.cmple(view + Vec2::splat(0.5)).all(), "{w}x{h}: {b:?}");
                        let mut rects: Vec<Rect> = b.rulers.iter().chain(&b.forms).copied().collect();
                        if let Some(t) = &b.tabs {
                            rects.extend(t.iter().copied());
                        }
                        if let Some(p) = &b.paths {
                            rects.push(p.chart);
                            rects.push((p.legend, p.legend + Vec2::new(10.0, 2.0 * 16.0)));
                        }
                        if let Some(m) = &b.matrix {
                            rects.push(m.chart);
                            rects.extend(m.buttons.iter().copied());
                        }
                        assert!(rects.iter().all(|r| within(r.0) && within(r.1)), "{w}x{h}: {b:?}");
                    }
                }
            }
        }
    }
}

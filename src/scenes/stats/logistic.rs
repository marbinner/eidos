//! Logistic regression: answers that are yes or no, the probability of yes as an S-shaped
//! curve of the input, fitted by maximum likelihood, and a threshold that turns the
//! probability into a decision.
//!
//! The data are made so that the truth is itself a logistic model: the inputs of the no's
//! are bell-shaped round −d/2, those of the yes's round +d/2, each of spread σ, and then
//! the log-odds of yes at x is exactly ln(share / (1 − share)) + (d / σ²) x. (In two
//! inputs the same along one direction; the ring is the exception, made to fail.)

use super::plot::{self, Glide, Plot, Rect, AXES, CURVES, MARKS, OWN};
use super::*;
use crate::math::stats::{normal, ranks, unit};

pub const DEF: SceneDef = SceneDef {
    id: "logistic",
    title: "Logistic regression",
    group: MODELS,
    blurb: "Fitting a probability: odds, the logistic curve, and a boundary between two classes found by maximum likelihood.",
    make: |_| Box::new(Logistic::create()),
};

// ----- the model ---------------------------------------------------------------------------

/// The logistic function: the probability of yes at log-odds `z`, worked out so that it
/// is exact in both tails.
fn prob(z: f64) -> f64 {
    if z >= 0.0 {
        1.0 / (1.0 + (-z).exp())
    } else {
        let e = z.exp();
        e / (1.0 + e)
    }
}

/// ln(1 + e^z), without overflow for a large z or a loss of digits for a small one.
fn softplus(z: f64) -> f64 {
    z.max(0.0) + (-z.abs()).exp().ln_1p()
}

/// The log-odds of a probability (±∞ at 1 and 0).
fn logit(p: f64) -> f64 {
    (p / (1.0 - p)).ln()
}

/// The log-odds at which the steepness handle sits: 2, where the probability is 0.88.
const STEEP_AT: f64 = 2.0;

/// The model's coefficients: the intercept a and one slope per input (the second is 0
/// with one input). The log-odds of yes at x is a + b·x.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
struct Coef {
    a: f64,
    b: [f64; 2],
}

impl Coef {
    /// One input: the curve whose probability is 1/2 at `mid` and whose log-odds rise by
    /// `slope` for each unit of x.
    fn through(mid: f64, slope: f64) -> Coef {
        Coef { a: -slope * mid, b: [slope, 0.0] }
    }

    fn score(&self, x: [f64; 2]) -> f64 {
        self.a + self.b[0] * x[0] + self.b[1] * x[1]
    }

    /// The share `t` of the way from here to `to`.
    fn toward(&self, to: &Coef, t: f64) -> Coef {
        Coef { a: self.a + (to.a - self.a) * t, b: [self.b[0] + (to.b[0] - self.b[0]) * t, self.b[1] + (to.b[1] - self.b[1]) * t] }
    }

    fn finite(&self) -> bool {
        self.a.is_finite() && self.b[0].is_finite() && self.b[1].is_finite()
    }

    /// The size of the slope (of the two slopes together).
    fn steepness(&self) -> f64 {
        self.b[0].hypot(self.b[1])
    }

    /// The largest change of a coefficient from `other`.
    fn apart(&self, other: &Coef) -> f64 {
        (self.a - other.a).abs().max((self.b[0] - other.b[0]).abs()).max((self.b[1] - other.b[1]).abs())
    }

    fn largest(&self) -> f64 {
        self.a.abs().max(self.b[0].abs()).max(self.b[1].abs())
    }
}

// ----- the data ----------------------------------------------------------------------------

/// The most points a sample has.
const MOST: usize = 400;

/// The direction along which the two clouds of two inputs lie apart.
const ALONG: [f64; 2] = [0.8, 0.6];

/// How the points of two inputs lie.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Kind {
    /// Two bell-shaped clouds, one of each class.
    Clouds,
    /// The yes's in the middle, the no's in a ring round them.
    Ring,
}

/// What the data are made from.
#[derive(Clone, Copy, Debug, PartialEq)]
struct Setup {
    /// One input or two.
    dims: usize,
    kind: Kind,
    n: usize,
    /// The share of the points that are yes's (exactly: the count is rounded).
    share: f64,
    /// The distance d between the centres of the two classes.
    apart: f64,
    /// The spread σ of each class about its centre.
    spread: f64,
    seed: u64,
}

impl Setup {
    /// How many of the points are yes's.
    fn yes_count(&self) -> usize {
        ((self.share * self.n as f64).round().max(0.0) as usize).min(self.n)
    }
}

/// The random numbers a sample is made from, for the most points there can be: changing
/// how many there are, or how far apart the classes are, moves the same points.
struct Draws {
    seed: u64,
    /// A bell-shaped pair for each point.
    z: Vec<[f64; 2]>,
    /// Which points are yes's: those with the smallest keys.
    key: Vec<f64>,
    /// Where in its row a point is drawn (0 to 1), and where round the ring it lies.
    jitter: Vec<f64>,
    turn: Vec<f64>,
}

impl Draws {
    fn new(seed: u64) -> Draws {
        let mut rng = Rng::new(seed);
        let z = (0..MOST).map(|_| [normal(&mut rng), normal(&mut rng)]).collect();
        let key = (0..MOST).map(|_| unit(&mut rng)).collect();
        let jitter = (0..MOST).map(|_| unit(&mut rng)).collect();
        let turn = (0..MOST).map(|_| unit(&mut rng)).collect();
        Draws { seed, z, key, jitter, turn }
    }
}

/// A sample: the inputs (the second 0 with one input) and the answers.
#[derive(Clone, Debug, Default)]
struct Data {
    x: Vec<[f64; 2]>,
    y: Vec<bool>,
    dims: usize,
    yes: usize,
}

impl Data {
    fn new(s: &Setup, draws: &Draws) -> Data {
        let n = s.n.clamp(1, MOST);
        let yes_count = s.yes_count().min(n);
        let mut order: Vec<usize> = (0..n).collect();
        order.sort_by(|&i, &j| draws.key[i].total_cmp(&draws.key[j]));
        let mut y = vec![false; n];
        for &i in order.iter().take(yes_count) {
            y[i] = true;
        }
        let x = (0..n)
            .map(|i| {
                let z = draws.z[i];
                let side = if y[i] { 0.5 } else { -0.5 } * s.apart;
                match (s.dims, s.kind) {
                    (1, _) => [side + s.spread * z[0], 0.0],
                    (_, Kind::Clouds) => [side * ALONG[0] + s.spread * z[0], side * ALONG[1] + s.spread * z[1]],
                    (_, Kind::Ring) if y[i] => [0.6 * s.spread * z[0], 0.6 * s.spread * z[1]],
                    (_, Kind::Ring) => {
                        let r = 1.0 + s.apart + 0.35 * s.spread * z[0];
                        let angle = std::f64::consts::TAU * draws.turn[i];
                        [r * angle.cos(), r * angle.sin()]
                    }
                }
            })
            .collect();
        Data { x, y, dims: if s.dims == 2 { 2 } else { 1 }, yes: yes_count }
    }

    fn len(&self) -> usize {
        self.y.len()
    }

    /// Every point a yes, or every point a no: nothing for a curve to tell apart.
    fn one_class(&self) -> bool {
        self.yes == 0 || self.yes == self.len()
    }

    /// The inputs do not spread out: with one input all the same, with two all on one
    /// line. Then the slopes are not all settled by the data.
    fn same_inputs(&self) -> bool {
        let n = self.len() as f64;
        let m = [0, 1].map(|k| self.x.iter().map(|x| x[k]).sum::<f64>() / n);
        let c = |i: usize, j: usize| self.x.iter().map(|x| (x[i] - m[i]) * (x[j] - m[j])).sum::<f64>() / n;
        if self.dims == 1 {
            c(0, 0) <= 1e-18
        } else {
            let (xx, yy, xy) = (c(0, 0), c(1, 1), c(0, 1));
            xx * yy - xy * xy <= 1e-12 * (xx + yy).powi(2) + 1e-30
        }
    }

    /// With one input: every yes on one side of every no (or the other way round), so
    /// that a steeper and steeper curve puts them ever more surely right.
    fn separated(&self) -> bool {
        if self.dims != 1 || self.one_class() {
            return false;
        }
        let ends = |which: bool| self.x.iter().zip(&self.y).filter(|(_, y)| **y == which).fold((f64::MAX, f64::MIN), |(lo, hi), (x, _)| (lo.min(x[0]), hi.max(x[0])));
        let ((yes_lo, yes_hi), (no_lo, no_hi)) = (ends(true), ends(false));
        no_hi < yes_lo || yes_hi < no_lo
    }
}

/// The true model, where the data have one: the log-odds of yes at x for a point drawn
/// as these are (none for the ring, or where it would be a step: no spread).
fn truth(s: &Setup, d: &Data) -> Option<Coef> {
    if s.dims == 2 && s.kind == Kind::Ring || d.one_class() || s.spread <= 0.0 {
        return None;
    }
    let slope = s.apart / (s.spread * s.spread);
    let a = logit(d.yes as f64 / d.len() as f64);
    let b = if s.dims == 1 { [slope, 0.0] } else { [slope * ALONG[0], slope * ALONG[1]] };
    Some(Coef { a, b })
}

// ----- the likelihood and Newton's method --------------------------------------------------

/// The log-likelihood: the sum over the points of the log of the probability the model
/// gave to what happened.
fn log_likelihood(c: &Coef, d: &Data) -> f64 {
    d.x.iter().zip(&d.y).map(|(x, &yes)| ln_given(c.score(*x), yes)).sum()
}

/// The log of the probability that log-odds `z` give to the answer `yes`.
fn ln_given(z: f64, yes: bool) -> f64 {
    if yes {
        -softplus(-z)
    } else {
        -softplus(z)
    }
}

/// What the fit makes as large as it can: the log-likelihood less the penalty on the
/// slopes' size, (λ/2)(b₁² + b₂²). (The intercept is not penalised: it only says where
/// the probability is one half.)
fn objective(c: &Coef, d: &Data, penalty: f64) -> f64 {
    log_likelihood(c, d) - 0.5 * penalty * (c.b[0] * c.b[0] + c.b[1] * c.b[1])
}

/// The slope of the objective along each coefficient (intercept, then slopes), and how
/// it bends: the gradient and minus the Hessian (the information).
fn gradient(c: &Coef, d: &Data, penalty: f64) -> ([f64; 3], [[f64; 3]; 3]) {
    let k = d.dims + 1;
    let (mut g, mut h) = ([0.0; 3], [[0.0; 3]; 3]);
    for (x, &yes) in d.x.iter().zip(&d.y) {
        let z = c.score(*x);
        let row = [1.0, x[0], x[1]];
        // (y − p and p(1 − p), each from the tail it is small in, so no digits are lost)
        let miss = if yes { prob(-z) } else { -prob(z) };
        let w = prob(z) * prob(-z);
        for i in 0..k {
            g[i] += miss * row[i];
            for j in 0..k {
                h[i][j] += w * row[i] * row[j];
            }
        }
    }
    for i in 1..k {
        g[i] -= penalty * c.b[i - 1];
        h[i][i] += penalty;
    }
    (g, h)
}

/// The k × k system h s = g, by elimination with the largest pivot. None if it is
/// singular, near enough.
fn eliminate(mut m: [[f64; 3]; 3], mut g: [f64; 3], k: usize, scale: f64) -> Option<[f64; 3]> {
    for c in 0..k {
        let p = (c..k).max_by(|&i, &j| m[i][c].abs().total_cmp(&m[j][c].abs()))?;
        let pivot = m[p][c].abs();
        if pivot.is_nan() || pivot <= 1e-13 * scale {
            return None;
        }
        m.swap(p, c);
        g.swap(p, c);
        for r in c + 1..k {
            let f = m[r][c] / m[c][c];
            for j in c..k {
                m[r][j] -= f * m[c][j];
            }
            g[r] -= f * g[c];
        }
    }
    let mut s = [0.0; 3];
    for c in (0..k).rev() {
        s[c] = (g[c] - (c + 1..k).map(|j| m[c][j] * s[j]).sum::<f64>()) / m[c][c];
    }
    s.iter().all(|v| v.is_finite()).then_some(s)
}

/// Solve for Newton's step. Where the information is singular (inputs that are all the
/// same, or on one line: the data then do not settle every coefficient), a touch of
/// damping picks the smallest step among the many that would do.
fn solve(h: [[f64; 3]; 3], g: [f64; 3], k: usize) -> Option<[f64; 3]> {
    let scale = (0..k).map(|i| h[i][i].abs()).fold(0.0, f64::max);
    if !(scale > 0.0 && scale.is_finite()) {
        return None;
    }
    [0.0, 1e-10, 1e-6].into_iter().find_map(|damping| {
        let mut m = h;
        for (i, row) in m.iter_mut().enumerate().take(k) {
            row[i] += damping * scale;
        }
        eliminate(m, g, k, scale)
    })
}

/// The longest step Newton's method takes (in the coefficients' units). From a curve far
/// too steep, far from the points, the likelihood hardly bends, and a full step would
/// fly off by thousands.
const LONGEST: f64 = 8.0;

/// One step of Newton's method on the objective, no longer than `LONGEST` and halved as
/// often as it takes for the objective not to fall (a full step can overshoot from a
/// curve far from the top). Where the likelihood is so flat that even that fails (every
/// point given a probability within a hair of 0 or 1), a step up the gradient instead.
/// None if no step can be worked out.
fn newton(c: &Coef, d: &Data, penalty: f64) -> Option<Coef> {
    let k = d.dims + 1;
    let (g, h) = gradient(c, d, penalty);
    let start = objective(c, d, penalty);
    let along = |s: [f64; 3]| -> Option<Coef> {
        let size = s.iter().take(k).fold(0.0f64, |m, v| m.max(v.abs()));
        if size == 0.0 {
            return Some(*c);
        }
        if !size.is_finite() {
            return None;
        }
        let mut t = if size > LONGEST { LONGEST / size } else { 1.0 };
        for _ in 0..60 {
            let to = Coef { a: c.a + t * s[0], b: [c.b[0] + t * s[1], if k > 2 { c.b[1] + t * s[2] } else { 0.0 }] };
            let v = objective(&to, d, penalty);
            if to.finite() && v.is_finite() && v >= start - 1e-12 * start.abs().max(1.0) {
                return Some(to);
            }
            t *= 0.5;
        }
        None
    };
    solve(h, g, k).and_then(along).or_else(|| along(g))
}

/// A step this small (beside the coefficients' own size) is the top.
fn arrived(from: &Coef, to: &Coef) -> bool {
    from.apart(to) <= 1e-9 * (1.0 + to.largest())
}

/// Whether there is a top, as far as the fit can tell.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Top {
    /// Newton's method came to rest: the top.
    At,
    /// It does not come to rest: the objective rises for ever (the classes are apart).
    Never,
    /// Every point the same answer: no curve to fit.
    OneClass,
}

/// The top of the objective for these data and this penalty.
#[derive(Clone, Copy, Debug)]
struct Best {
    coef: Coef,
    log_likelihood: f64,
    objective: f64,
    top: Top,
}

/// Steps the search for the top takes at most.
const SEARCH: usize = 200;

fn best(d: &Data, penalty: f64) -> Best {
    let share = d.yes as f64 / d.len().max(1) as f64;
    let flat = Coef { a: logit(share.max(1e-9).min(1.0 - 1e-9)), b: [0.0, 0.0] };
    let done = |coef: Coef, top: Top| Best { coef, log_likelihood: log_likelihood(&coef, d), objective: objective(&coef, d, penalty), top };
    if d.one_class() {
        return done(flat, Top::OneClass);
    }
    let mut c = flat;
    for _ in 0..SEARCH {
        let Some(next) = newton(&c, d, penalty) else { return done(c, Top::Never) };
        let there = arrived(&c, &next);
        c = next;
        if there {
            return done(c, Top::At);
        }
    }
    done(c, Top::Never)
}

// ----- decisions ---------------------------------------------------------------------------

/// The four outcomes of saying yes wherever the probability is at least the threshold.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
struct Table {
    /// Yes's said yes to, yes's said no to, no's said yes to, no's said no to.
    hits: usize,
    misses: usize,
    false_alarms: usize,
    right_nos: usize,
}

impl Table {
    fn of(probs: &[f64], y: &[bool], threshold: f64) -> Table {
        let mut t = Table::default();
        for (&p, &yes) in probs.iter().zip(y) {
            match (yes, p >= threshold) {
                (true, true) => t.hits += 1,
                (true, false) => t.misses += 1,
                (false, true) => t.false_alarms += 1,
                (false, false) => t.right_nos += 1,
            }
        }
        t
    }

    fn yes(&self) -> usize {
        self.hits + self.misses
    }

    fn no(&self) -> usize {
        self.false_alarms + self.right_nos
    }

    /// The share of the yes's found (None without yes's).
    fn hit_rate(&self) -> Option<f64> {
        (self.yes() > 0).then(|| self.hits as f64 / self.yes() as f64)
    }

    /// The share of the no's said yes to (None without no's).
    fn false_alarm_rate(&self) -> Option<f64> {
        (self.no() > 0).then(|| self.false_alarms as f64 / self.no() as f64)
    }

    fn right(&self) -> usize {
        self.hits + self.right_nos
    }

    fn accuracy(&self) -> f64 {
        self.right() as f64 / (self.yes() + self.no()).max(1) as f64
    }
}

/// The ROC curve: the false alarm rate and the hit rate at every threshold, from the
/// highest (nothing said yes to) down to the lowest (everything), and the area under it.
/// Points with the same probability are passed together, as one threshold does: a
/// slanting piece. None unless there are yes's and no's.
#[derive(Clone, Debug, PartialEq)]
struct Roc {
    points: Vec<[f64; 2]>,
    area: f64,
}

fn roc(scores: &[f64], y: &[bool]) -> Option<Roc> {
    let yes = y.iter().filter(|v| **v).count();
    let no = y.len() - yes;
    if yes == 0 || no == 0 {
        return None;
    }
    let mut order: Vec<usize> = (0..scores.len()).collect();
    order.sort_by(|&i, &j| scores[j].total_cmp(&scores[i]));
    let (mut hits, mut alarms, mut area) = (0usize, 0usize, 0.0);
    let mut points = vec![[0.0, 0.0]];
    let mut i = 0;
    while i < order.len() {
        let mut j = i;
        while j < order.len() && scores[order[j]] == scores[order[i]] {
            if y[order[j]] {
                hits += 1;
            } else {
                alarms += 1;
            }
            j += 1;
        }
        let last = points[points.len() - 1];
        let next = [alarms as f64 / no as f64, hits as f64 / yes as f64];
        area += (next[0] - last[0]) * (next[1] + last[1]) * 0.5;
        points.push(next);
        i = j;
    }
    Some(Roc { points, area })
}

/// The share of the pairs of a yes and a no in which the yes has the higher score (a
/// tie counting half), from the ranks of the scores: the Mann–Whitney count. (None
/// unless there are yes's and no's.)
fn ranked_right(scores: &[f64], y: &[bool]) -> Option<f64> {
    let yes = y.iter().filter(|v| **v).count() as f64;
    let no = y.len() as f64 - yes;
    if yes == 0.0 || no == 0.0 {
        return None;
    }
    let r = ranks(scores);
    let sum: f64 = r.iter().zip(y).filter(|(_, y)| **y).map(|(r, _)| r).sum();
    Some((sum - yes * (yes + 1.0) / 2.0) / (yes * no))
}

/// One slice of the data along the input, for the log-odds chart: where its points lie
/// on average, and the share of them that are yes's.
#[derive(Clone, Copy, Debug, PartialEq)]
struct Slice {
    x: f64,
    share: f64,
    n: usize,
}

/// The data cut into slices of (nearly) equal counts along the input: as many as give
/// each at least eight points, and at most eight.
fn slices(d: &Data) -> Vec<Slice> {
    let n = d.len();
    let k = (n / 8).clamp(1, 8);
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&i, &j| d.x[i][0].total_cmp(&d.x[j][0]));
    (0..k)
        .map(|s| {
            let part = &order[s * n / k..(s + 1) * n / k];
            let m = part.len().max(1) as f64;
            let x = part.iter().map(|&i| d.x[i][0]).sum::<f64>() / m;
            let share = part.iter().filter(|&&i| d.y[i]).count() as f64 / m;
            Slice { x, share, n: part.len() }
        })
        .collect()
}

/// The straight line least squares puts through the 0s and 1s: (intercept, slope).
fn least_squares(d: &Data) -> (f64, f64) {
    let x: Vec<f64> = d.x.iter().map(|x| x[0]).collect();
    let y: Vec<f64> = d.y.iter().map(|&y| if y { 1.0 } else { 0.0 }).collect();
    let (slope, intercept) = crate::math::causal::line(&x, &y);
    (intercept, slope)
}

// ----- the scene ---------------------------------------------------------------------------

/// Newton's steps a second, as the visitor watches them.
const RATE: f32 = 3.0;
/// The most steps one press of "fit" takes.
const CAP: usize = 30;
/// How long the model takes to glide to a new setting (seconds).
const JUMP: f32 = 0.6;
/// How long the threshold takes to slide to where an experiment puts it (seconds).
const SLIDE: f32 = 1.6;

/// The ranges the charts show: the input, the probability (with room above 1 and below 0
/// for the rows of points), the log-odds, the plane of two inputs, and the log-odds a
/// point of two inputs is given.
const X_RANGE: [f64; 2] = [-5.0, 5.0];
const P_RANGE: [f64; 2] = [-0.25, 1.25];
const LOGODDS: [f64; 2] = [-6.0, 6.0];
const PLANE: [f64; 2] = [-4.5, 4.5];
const SCORES: [f64; 2] = [-7.0, 7.0];

/// What the picture shows besides the model.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
struct Shows {
    least_squares: bool,
    odds: bool,
    truth: bool,
}

/// How a run of Newton's steps ended.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum End {
    /// At the top: the last step changed nothing.
    Top,
    /// It took the most steps a press allows without coming to rest.
    Cap,
    /// No step could be worked out.
    Stuck,
}

/// Newton's method as the visitor watches it.
#[derive(Clone, Copy, Debug, Default)]
struct Run {
    /// Steps still to take (0: not running).
    left: usize,
    /// Steps taken since the model was last set any other way.
    taken: usize,
    end: Option<End>,
    /// The step after the last, worked out ahead (the picture needs to know where the
    /// run goes next to pass smoothly through where it is going now).
    next: Option<Coef>,
}

/// The model as drawn: on its way from `from` to `to` (the model itself), `share` of the
/// way there. A new setting eases in and out. A run of Newton's steps is drawn along a
/// smooth path through the steps (a Catmull–Rom spline, through `before`, `from`, `to`
/// and `after`), each step in the time to the next, so that it moves in every frame
/// and its speed does not jerk where one step meets the next.
#[derive(Clone, Copy, Debug)]
struct Look {
    before: Coef,
    from: Coef,
    to: Coef,
    after: Coef,
    share: f32,
    newton: bool,
}

impl Look {
    fn at(c: Coef) -> Look {
        Look { before: c, from: c, to: c, after: c, share: 1.0, newton: false }
    }

    fn now(&self) -> Coef {
        if self.share >= 1.0 {
            return self.to;
        }
        let s = self.share.clamp(0.0, 1.0) as f64;
        if !self.newton {
            return self.from.toward(&self.to, s * s * (3.0 - 2.0 * s));
        }
        let (s2, s3) = (s * s, s * s * s);
        let w = [-0.5 * s3 + s2 - 0.5 * s, 1.5 * s3 - 2.5 * s2 + 1.0, -1.5 * s3 + 2.0 * s2 + 0.5 * s, 0.5 * s3 - 0.5 * s2];
        let knots = [self.before, self.from, self.to, self.after];
        let sum = |pick: fn(&Coef) -> f64| knots.iter().zip(w).map(|(k, w)| w * pick(k)).sum::<f64>();
        Coef { a: sum(|c| c.a), b: [sum(|c| c.b[0]), sum(|c| c.b[1])] }
    }

    fn moving(&self) -> bool {
        self.share < 1.0
    }
}

/// A point that follows where it should be through two eases one after the other, slower
/// than the kit's [`Glide`]: the decision's place on the ROC curve steps from one corner to
/// the next as the threshold passes each point, and drawn quickly the steps would show.
#[derive(Clone, Copy, Debug, Default)]
struct Follow {
    between: [f32; 2],
    shown: Option<[f32; 2]>,
}

impl Follow {
    const EASE: f32 = 0.12;

    /// `dt` seconds nearer to `to`; says whether it is still on its way.
    fn follow(&mut self, to: [f32; 2], dt: f32) -> bool {
        let Some(mut shown) = self.shown.filter(|s| s.iter().all(|v| v.is_finite())) else {
            (self.between, self.shown) = (to, Some(to));
            return false;
        };
        let k = 1.0 - (-dt.max(0.0) / Self::EASE).exp();
        let mut apart = 0.0f32;
        for i in 0..2 {
            self.between[i] += (to[i] - self.between[i]) * k;
            shown[i] += (self.between[i] - shown[i]) * k;
            apart = apart.max((to[i] - shown[i]).abs()).max((to[i] - self.between[i]).abs());
        }
        // (exactly there in the end: the picture then stops changing)
        if apart.is_nan() || apart <= 1e-4 {
            (self.between, shown) = (to, to);
        }
        self.shown = Some(shown);
        apart > 1e-4
    }
}

/// The threshold on its way to where an experiment puts it.
#[derive(Clone, Copy, Debug)]
struct Slide {
    from: f64,
    to: f64,
    share: f32,
}

/// How the model starts in an experiment.
#[derive(Clone, Copy, Debug)]
enum Start {
    /// At the top of the likelihood.
    Best,
    /// Set by hand: a midpoint and a slope.
    Hand(f64, f64),
}

/// An experiment: its button, what to look for, and every setting it depends on.
struct Trial {
    button: &'static str,
    caption: &'static str,
    setup: Setup,
    start: Start,
    /// Then Newton's method is run from there, as by "fit".
    fit: bool,
    threshold: f64,
    /// The threshold then slides to this.
    slide: Option<f64>,
    shows: Shows,
    penalty: f64,
}

/// Where the pair of inputs the odds are compared at starts: the first of them.
const ODDS_AT: f64 = -0.5;

/// The data the scene opens with, and most experiments show.
const OVERLAP: Setup = Setup { dims: 1, kind: Kind::Clouds, n: 40, share: 0.5, apart: 2.0, spread: 1.0, seed: 11 };
const APART: Setup = Setup { dims: 1, kind: Kind::Clouds, n: 30, share: 0.5, apart: 3.0, spread: 0.5, seed: 3 };
const RARE: Setup = Setup { dims: 1, kind: Kind::Clouds, n: 200, share: 0.05, apart: 2.0, spread: 1.0, seed: 9 };
const CLOUDS: Setup = Setup { dims: 2, kind: Kind::Clouds, n: 120, share: 0.5, apart: 2.5, spread: 1.0, seed: 3 };
const RING: Setup = Setup { dims: 2, kind: Kind::Ring, n: 120, share: 0.5, apart: 1.5, spread: 1.0, seed: 3 };

const NOTHING: Shows = Shows { least_squares: false, odds: false, truth: false };

const TRIALS: [Trial; 7] = [
    Trial {
        button: "a straight line",
        caption: r"Least squares puts a straight line (blue) through the points at 0 and at 1. Read as a probability it is impossible at both ends: below 0 for $x < -1.72$ and above 1 for $x > 2.06$ (where it turns pink), and 6 of the 40 points lie there. The logistic curve (yellow) bends to stay between 0 and 1; what it makes straight is the log-odds, in the chart below.",
        setup: OVERLAP,
        start: Start::Best,
        fit: false,
        threshold: 0.5,
        slide: None,
        shows: Shows { least_squares: true, ..NOTHING },
        penalty: 0.0,
    },
    Trial {
        button: "by hand, then by the method",
        caption: r"The curve starts badly set: log-likelihood $-22.81$, against a top of $-13.58$. Drag its two yellow handles, the midpoint and the steepness, to raise the log-likelihood as far as you can: the pink bars shorten where the curve gives more probability to what happened. Then press fit: Newton's method climbs to the top (from this start, in 7 steps). No setting by hand gets above it.",
        setup: OVERLAP,
        start: Start::Hand(1.8, 0.45),
        fit: false,
        threshold: 0.5,
        slide: None,
        shows: NOTHING,
        penalty: 0.0,
    },
    Trial { button: "odds", caption: r"The fitted slope is $b = 1.99$, so one more unit of $x$ multiplies the odds of yes by $e^{1.99} = 7.28$. At $x = -0.5$ the odds are 0.29 ($p = 0.23$); at $x = 0.5$ they are 2.13 ($p = 0.68$): 7.28 times as much. Drag the pair along the curve by its left dot: the probabilities change, the factor does not. In the log-odds chart, one unit across is $b$ up.", setup: OVERLAP, start: Start::Best, fit: false, threshold: 0.5, slide: None, shows: Shows { odds: true, ..NOTHING }, penalty: 0.0 },
    Trial {
        button: "classes apart",
        caption: r"Every yes lies to the right of every no. Then a steeper curve (its midpoint in the gap) always gives what happened more probability, and the likelihood has no top: Newton's method runs away. The slope is 6.24 after 5 steps, 13.15 after 10 and 28.64 after 20, while the log-likelihood creeps up towards 0; the scene stops it after 30. Yet the data were drawn from a curve of slope 12 (green). Set the penalty to 0.1 and press fit: now there is a top, at slope 3.99, an answer pulled towards flat.",
        setup: APART,
        start: Start::Hand(0.0, 1.0),
        fit: true,
        threshold: 0.5,
        slide: None,
        shows: Shows { truth: true, ..NOTHING },
        penalty: 0.0,
    },
    Trial { button: "a lower threshold", caption: r"The threshold slides from 0.5 down to 0.2, and more points are said yes to: misses fall from 3 to 0, false alarms rise from 2 to 7, and the white point climbs the ROC curve from $(0.10, 0.85)$ to $(0.35, 1)$. That curve is every threshold at once. The area under it, 0.92, is also the share of the (yes, no) pairs in which the yes gets the higher probability: 369 of the 400.", setup: OVERLAP, start: Start::Best, fit: false, threshold: 0.5, slide: Some(0.2), shows: NOTHING, penalty: 0.0 },
    Trial { button: "a rare yes", caption: r"10 yes's among 200. Saying no to every point would be right 95 % of the time, and find none of them. The fit, at threshold 0.5, is right 95 % of the time too, and finds 1 of the 10: accuracy cannot tell the two apart; the hit rate can. Lower the threshold to 0.1: it finds 8 of the 10 at 19 false alarms, and is right only 89.5 % of the time.", setup: RARE, start: Start::Best, fit: false, threshold: 0.5, slide: None, shows: NOTHING, penalty: 0.0 },
    Trial { button: "a ring", caption: r"Two inputs: the yes's in the middle, the no's in a ring round them. The boundary of this model is a straight line, and no straight line puts a ring on one side and its middle on the other. The fit comes out all but flat, every probability between 0.48 and 0.52, the area under the ROC curve 0.50: no better than chance. Given the squares of the inputs as two more inputs, the same model could draw a circle.", setup: RING, start: Start::Best, fit: false, threshold: 0.5, slide: None, shows: NOTHING, penalty: 0.0 },
];

/// The data sets the panel offers: (name, setup), with one input and with two.
const PRESETS_1: [(&str, Setup); 3] = [("overlapping", OVERLAP), ("classes apart", APART), ("a rare yes", RARE)];
const PRESETS_2: [(&str, Setup); 2] = [("two clouds", CLOUDS), ("a ring", RING)];

pub struct Logistic {
    setup: Setup,
    /// The penalty λ on the slopes' size.
    penalty: f64,
    threshold: f64,
    shows: Shows,
    /// Where the first of the two inputs the odds are compared at is (one input).
    odds_at: f64,
    /// The model.
    coef: Coef,
    draws: Draws,
    data: Data,
    /// The setup the data were made from, and the setup and penalty the top was found for.
    data_key: Option<Setup>,
    best_key: Option<(Setup, u64)>,
    best: Best,
    /// The least-squares line and the slices of the data (one input).
    line: (f64, f64),
    slices: Vec<Slice>,
    /// The highest objective reached by hand since the data or the penalty last changed.
    by_hand: Option<f64>,
    run: Run,
    pace: Pace,
    phase: Phase,
    look: Look,
    slide: Option<Slide>,
    /// The points as drawn, gliding to where a change of the data puts them: three
    /// numbers each, the inputs (the second 0 with one input) and the height in its row.
    points: Glide,
    /// Two inputs: the point the boundary's handle keeps near (it slides along the
    /// boundary to stay nearest to it).
    anchor: [f64; 2],
    /// A handle was held in the last frame, and the points were gliding.
    held: bool,
    gliding: bool,
    /// The steepness handle sits where the log-odds are +2 (1) or, where that is off
    /// the chart, −2 (−1); kept while a handle is held.
    steep_side: f64,
    /// Where the threshold puts the decision on the ROC curve, as drawn, and whether it
    /// is on its way.
    roc_point: Follow,
    roc_moving: bool,
}

impl Logistic {
    /// (Nothing here needs the graphics device, so the tests can make one too.)
    fn create() -> Logistic {
        let setup = OVERLAP;
        let draws = Draws::new(setup.seed);
        let data = Data::new(&setup, &draws);
        let best = best(&data, 0.0);
        let mut s = Logistic {
            setup,
            penalty: 0.0,
            threshold: 0.5,
            shows: NOTHING,
            odds_at: ODDS_AT,
            coef: best.coef,
            draws,
            data,
            data_key: None,
            best_key: None,
            best,
            line: (0.0, 0.0),
            slices: Vec::new(),
            by_hand: None,
            run: Run::default(),
            pace: Pace::default(),
            phase: Phase::default(),
            look: Look::at(best.coef),
            slide: None,
            points: Glide::default(),
            anchor: [0.0, 0.0],
            held: false,
            gliding: false,
            steep_side: 1.0,
            roc_point: Follow::default(),
            roc_moving: false,
        };
        s.refresh();
        s
    }

    /// Set up one of the experiments in `TRIALS`.
    fn set_up(&mut self, k: usize) {
        let Some(t) = TRIALS.get(k) else { return };
        let dims_before = self.setup.dims;
        self.setup = t.setup;
        self.penalty = t.penalty;
        self.shows = t.shows;
        self.odds_at = ODDS_AT;
        self.anchor = [0.0, 0.0];
        self.refresh();
        let coef = match t.start {
            Start::Best => self.best.coef,
            Start::Hand(mid, slope) => Coef::through(mid, slope),
        };
        self.set_model(coef, dims_before != self.setup.dims);
        self.by_hand = None;
        self.threshold = t.threshold;
        self.slide = t.slide.map(|to| Slide { from: t.threshold, to, share: 0.0 });
        if t.fit {
            self.start_run(CAP);
        }
    }

    /// Put the model at `coef`, gliding there from where it is drawn (or there at once).
    fn set_model(&mut self, coef: Coef, at_once: bool) {
        let from = self.look.now();
        self.coef = coef;
        self.run = Run::default();
        self.look = if at_once || !from.finite() { Look::at(coef) } else { Look { before: from, from, to: coef, after: coef, share: 0.0, newton: false } };
    }

    /// Make the data again if what they depend on has changed, and find the top again
    /// if the data or the penalty have.
    fn refresh(&mut self) {
        let s = &mut self.setup;
        s.dims = s.dims.clamp(1, 2);
        s.n = s.n.clamp(1, MOST);
        s.share = s.share.max(0.0).min(1.0);
        s.apart = s.apart.max(0.0).min(5.0);
        s.spread = s.spread.max(0.0).min(1.5);
        self.penalty = if self.penalty.is_finite() { self.penalty.max(0.0).min(10.0) } else { 0.0 };
        self.threshold = if self.threshold.is_finite() { self.threshold.max(0.0).min(1.0) } else { 0.5 };
        if self.draws.seed != self.setup.seed {
            self.draws = Draws::new(self.setup.seed);
        }
        if self.data_key != Some(self.setup) {
            let dims_changed = self.data_key.is_none_or(|k| k.dims != self.setup.dims);
            self.data = Data::new(&self.setup, &self.draws);
            self.data_key = Some(self.setup);
            self.line = least_squares(&self.data);
            self.slices = slices(&self.data);
            self.by_hand = None;
            if self.run.left > 0 {
                self.run = Run::default();
            }
            self.run.taken = 0;
            if dims_changed {
                // (a new kind of picture: the points start where they are)
                self.points = Glide::default();
                if self.setup.dims == 1 {
                    self.coef.b[1] = 0.0;
                    self.look = Look::at(self.coef);
                }
            }
        }
        let key = (self.setup, self.penalty.to_bits());
        if self.best_key != Some(key) {
            // (a step worked out ahead was for the old penalty)
            self.run.next = None;
            self.best = best(&self.data, self.penalty);
            self.best_key = Some(key);
            self.by_hand = None;
        }
    }

    /// Start Newton's method from the model as it is: the first step at once, then
    /// `RATE` a second, `steps` in all at most.
    fn start_run(&mut self, steps: usize) {
        if self.data.one_class() {
            return;
        }
        if self.run.left > 0 {
            // (a press while a run goes on lets it go on, as far as the press asks: a
            // step taken at once would cut short the one being drawn)
            self.run.left = self.run.left.max(steps);
            return;
        }
        if self.run.end.is_some() {
            // (a new press starts a new count)
            self.run.taken = 0;
        }
        self.run.left = steps;
        self.run.end = None;
        self.pace.reset();
        self.phase.reset();
        self.step();
    }

    /// One step of Newton's method, drawn from where the model is drawn now.
    fn step(&mut self) {
        // (a run already under way goes on from the step it was drawing, where the
        // picture now is near enough; a first step starts where the picture is)
        let (before, from) = if self.look.newton { (self.look.from, self.look.to) } else { (self.look.now(), self.look.now()) };
        let share = if self.look.newton { (self.look.share - 1.0).max(0.0) } else { 0.0 };
        let next = self.run.next.take().or_else(|| newton(&self.coef, &self.data, self.penalty));
        match next {
            Some(next) => {
                let there = arrived(&self.coef, &next);
                self.coef = next;
                self.run.taken += 1;
                self.run.left = self.run.left.saturating_sub(1);
                if there {
                    (self.run.left, self.run.end) = (0, Some(End::Top));
                } else if self.run.left == 0 && self.run.taken >= CAP {
                    self.run.end = Some(End::Cap);
                }
            }
            None => (self.run.left, self.run.end) = (0, Some(End::Stuck)),
        }
        // where the run goes after this, if it goes on (and drawn as stopping if not)
        self.run.next = if self.run.left > 0 { newton(&self.coef, &self.data, self.penalty) } else { None };
        let after = match self.run.next {
            Some(n) if !arrived(&self.coef, &n) => n,
            _ => self.coef,
        };
        self.look = Look { before, from, to: self.coef, after, share, newton: true };
    }

    /// The visitor has set the model by hand.
    fn by_hand_to(&mut self, coef: Coef) {
        if !coef.finite() {
            return;
        }
        self.coef = coef;
        self.run = Run::default();
        self.look = Look::at(coef);
        let v = objective(&coef, &self.data, self.penalty);
        if v.is_finite() {
            self.by_hand = Some(self.by_hand.map_or(v, |b| b.max(v)));
        }
    }

    /// Time passes: Newton's steps (by the scene's time, or by the clock while it stands
    /// still: they answer a press), the model's glide, the threshold's slide.
    fn advance(&mut self, dt: f32, real_dt: f32) {
        let t = if dt > 0.0 { dt } else { real_dt.max(0.0) };
        if self.run.left > 0 {
            let mut turn = scene_turn(&mut self.pace, RATE, t, real_dt);
            let mut taken = 0;
            while self.run.left > 0 && turn.next() {
                self.step();
                taken += 1;
            }
            self.pace.end(turn);
            let share = self.phase.after(RATE, t, taken);
            if self.look.newton {
                self.look.share = share;
            }
        } else if self.look.newton && self.look.moving() {
            self.look.share = self.phase.after(RATE, t, 0);
        } else if self.look.moving() {
            self.look.share = (self.look.share + real_dt.max(0.0) / JUMP).min(1.0);
        }
        if let Some(mut s) = self.slide {
            s.share = (s.share + real_dt.max(0.0) / SLIDE).min(1.0);
            let e = s.share as f64;
            let e = e * e * (3.0 - 2.0 * e);
            self.threshold = s.from + (s.to - s.from) * e;
            self.slide = (s.share < 1.0).then_some(s);
        }
    }

    /// Every motion to its end at once (the tests of this file).
    #[cfg(test)]
    fn settle(&mut self) {
        while self.run.left > 0 {
            self.step();
        }
        self.look = Look::at(self.coef);
        if let Some(s) = self.slide.take() {
            self.threshold = s.to;
        }
    }

    /// What the fit makes as large as it can, for the model as it is.
    fn value(&self) -> f64 {
        objective(&self.coef, &self.data, self.penalty)
    }
}

// ----- the layout --------------------------------------------------------------------------

const NO_RECT: Rect = (Vec2::ZERO, Vec2::ZERO);

/// The words on the buttons over the picture.
const BUTTON_WORDS: [&str; 4] = ["fit", "one step", "one input", "two inputs"];

/// Where everything goes on a canvas.
#[derive(Clone, Copy, Debug, PartialEq)]
struct Layout {
    view: Vec2,
    narrow: bool,
    /// Text sizes: words, small print.
    text: f32,
    small: f32,
    /// Room above a chart (its title and the name of its y axis) and below it (the
    /// numbers and name of its x axis), and a line of words under that.
    head: f32,
    foot: f32,
    line: f32,
    buttons: [Rect; 4],
    /// The probability against the input (with two inputs: the plane), the log-odds
    /// against the input (with two: the probability against the log-odds), the ROC
    /// curve, and the table of outcomes.
    main: Rect,
    second: Rect,
    roc: Rect,
    table: Rect,
    /// Room for the first chart alone.
    compact: bool,
    /// No room for anything.
    nothing: bool,
}

impl Layout {
    fn of(view: Vec2, inset: f32, dims: usize) -> Layout {
        let view = view.max(Vec2::splat(1.0));
        let nothing = view.x < 150.0 || view.y < 130.0;
        let narrow = view.x < 760.0;
        let (text, small) = if narrow { (11.5, 10.5) } else { (12.5, 11.5) };
        let side = if narrow { 8.0 } else { 20.0 };
        let numbers = if narrow { 30.0 } else { 36.0 };
        let top = inset.max(0.0) + if narrow { 6.0 } else { 10.0 };
        let row = 24.0;
        let (head, foot) = if narrow { (36.0, 34.0) } else { (44.0, 40.0) };
        let line = small + 6.0;
        // a square for the plane of two inputs, in a rectangle (left-aligned, or centred)
        let square = |(min, max): Rect, centred: bool| {
            let s = (max.x - min.x).min(max.y - min.y).max(0.0);
            let x = if centred { (min.x + max.x - s) * 0.5 } else { min.x };
            (Vec2::new(x, min.y), Vec2::new(x + s, min.y + s))
        };
        let fit = |r: Rect, centred: bool| if dims == 2 { square(r, centred) } else { r };
        let mut lay = Layout {
            view,
            narrow,
            text,
            small,
            head,
            foot,
            line,
            buttons: [NO_RECT; 4],
            main: NO_RECT,
            second: NO_RECT,
            roc: NO_RECT,
            table: NO_RECT,
            compact: false,
            nothing,
        };
        if narrow {
            let (x0, x1) = (side + numbers, view.x - side);
            let y0 = top + row + 8.0 + head;
            let below_main = foot + 2.0 * line + 4.0;
            let below_second = foot + line + 4.0;
            let below_roc = foot + 2.0 * line;
            let rest = view.y - y0 - below_main - head - below_second - head - below_roc - 2.0;
            lay.compact = rest < 250.0 || x1 - x0 < 170.0;
            if !lay.compact {
                let b = plot::even_row(4, side, view.x - side, top, row, 5.0);
                lay.buttons = [0, 1, 2, 3].map(|i| b.get(i).copied().unwrap_or(NO_RECT));
                let (h1, h2, h3) = (rest * 0.46, rest * 0.22, rest * 0.32);
                lay.main = fit((Vec2::new(x0, y0), Vec2::new(x1, y0 + h1)), true);
                let s0 = y0 + h1 + below_main + head;
                lay.second = (Vec2::new(x0, s0), Vec2::new(x1, s0 + h2));
                let r0 = s0 + h2 + below_second + head;
                let r = h3.min((x1 - x0) * 0.46);
                lay.roc = (Vec2::new(x0, r0), Vec2::new(x0 + r, r0 + r));
                lay.table = (Vec2::new(x0 + r + 16.0, r0 - head + 6.0), Vec2::new(x1, view.y - 6.0));
                return lay;
            }
        } else {
            let lx0 = side + numbers;
            let lx1 = side + (view.x - 2.0 * side) * 0.6;
            let rx0 = lx1 + 48.0 + numbers;
            let rx1 = view.x - side;
            let y0 = top + row + 12.0 + head;
            let below_main = foot + 2.0 * line + 4.0;
            let rest = view.y - y0 - below_main - head - foot - line - 10.0;
            lay.compact = rest < 260.0;
            if !lay.compact {
                let left = button_row(&BUTTON_WORDS[..2], lx0 - numbers, top, row, small, 6.0);
                let right = button_row(&BUTTON_WORDS[2..], 0.0, top, row, small, 6.0);
                let shift = rx1 - right.last().map_or(0.0, |r| r.1.x);
                let right: Vec<Rect> = right.iter().map(|r| (r.0 + Vec2::new(shift, 0.0), r.1 + Vec2::new(shift, 0.0))).collect();
                lay.buttons = [left[0], left[1], right[0], right[1]];
                let (h1, h2) = (rest * 0.6, rest * 0.4);
                lay.main = fit((Vec2::new(lx0, y0), Vec2::new(lx1, y0 + h1)), false);
                let s0 = y0 + h1 + below_main + head;
                lay.second = (Vec2::new(lx0, s0), Vec2::new(lx1, s0 + h2));
                let r = (rx1 - rx0).min(rest * 0.5).max(0.0);
                lay.roc = (Vec2::new(rx0, y0), Vec2::new(rx0 + r, y0 + r));
                let t0 = y0 + r + foot + 2.0 * line + 14.0;
                lay.table = (Vec2::new(rx0 - numbers, t0), Vec2::new(rx1, view.y - 10.0));
                return lay;
            }
        }
        // the first chart alone, as large as the canvas allows
        let (x0, x1) = (side + numbers, view.x - side);
        // (room above for the title and the axis's name, as every chart has)
        let y0 = top + if narrow { 30.0 } else { 44.0 };
        lay.main = fit((Vec2::new(x0, y0), Vec2::new(x1, (view.y - foot).max(y0 + 1.0))), true);
        lay
    }
}

/// Rectangles for buttons side by side from (x, y), each as wide as its words.
fn button_row(words: &[&str], x: f32, y: f32, h: f32, size: f32, gap: f32) -> Vec<Rect> {
    let mut left = x;
    words
        .iter()
        .map(|w| {
            let width = text_width(w, size) + 20.0;
            let r = (Vec2::new(left, y), Vec2::new(left + width, y + h));
            left += width + gap;
            r
        })
        .collect()
}

/// The charts of a layout.
struct Plots {
    main: Plot,
    second: Plot,
    roc: Plot,
}

impl Plots {
    fn of(lay: &Layout, dims: usize) -> Plots {
        let (main, second) = if dims == 2 {
            (Plot::new(lay.main, PLANE, PLANE), Plot::new(lay.second, SCORES, P_RANGE))
        } else {
            (Plot::new(lay.main, X_RANGE, P_RANGE), Plot::new(lay.second, X_RANGE, LOGODDS))
        };
        Plots { main, second, roc: Plot::new(lay.roc, [0.0, 1.0], [0.0, 1.0]) }
    }
}

// ----- lines of our own --------------------------------------------------------------------

/// The batches of lines this scene keeps apart, beyond the kit's: what changes its number
/// of points (see `plot.rs`).
const BARS: u8 = OWN;
const ROC_LINE: u8 = OWN + 1;
const CONTOURS: u8 = OWN + 2;
const DASHES: u8 = OWN + 3;
const ODDS: u8 = OWN + 4;
const ARROWS: u8 = OWN + 5;

/// A polyline on the screen as dashes along its length (the line batches dash each
/// segment afresh, which leaves a curve of short segments solid): `on` pixels drawn,
/// `off` left out, the pieces apart by a point that is not a number.
fn dashes(pts: &[Vec2], on: f32, off: f32) -> Vec<Vec2> {
    let mut out = Vec::new();
    let mut gone = 0.0f32;
    for piece in pts.split(|p| !p.is_finite()) {
        for w in piece.windows(2) {
            let (a, b) = (w[0], w[1]);
            let len = (b - a).length();
            let mut at = 0.0f32;
            while at < len {
                let phase = gone % (on + off);
                let (drawing, left) = if phase < on { (true, on - phase) } else { (false, on + off - phase) };
                let step = left.min(len - at);
                if drawing {
                    let (p, q) = (a + (b - a) * (at / len), a + (b - a) * ((at + step) / len));
                    if out.last().is_some_and(|l: &Vec2| (*l - p).length() < 1e-3) {
                        out.push(q);
                    } else {
                        out.extend([Vec2::NAN, p, q]);
                    }
                }
                at += step;
                gone += step;
            }
        }
        out.push(Vec2::NAN);
    }
    out
}

/// A dashed curve y = g(x) across a chart.
fn dashed_curve(f: &mut Frame, plot: &Plot, kind: u8, width: f32, color: Color, g: impl Fn(f64) -> f64) {
    let n = 120;
    let pts: Vec<Vec2> = (0..n).map(|i| plot.x[0] + (plot.x[1] - plot.x[0]) * i as f64 / (n - 1) as f64).map(|x| plot.at(x, g(x))).collect();
    let cut = plot::clip(&pts, plot.rect);
    plot::path_of(f, kind, &dashes(&cut, 7.0, 5.0), width, color);
}

/// The part of the line y = a + bx inside a chart, as its two ends (None if it misses the
/// chart). Drawn as two points, a turning line keeps its number of points in every frame;
/// sampled and cut at the edges, it would gain and lose them as the samples crossed.
fn line_in(plot: &Plot, a: f64, b: f64) -> Option<[(f64, f64); 2]> {
    let ([x0, x1], [y0, y1]) = (plot.x, plot.y);
    let (lo, hi) = if b.abs() > 1e-12 {
        let (xa, xb) = ((y0 - a) / b, (y1 - a) / b);
        (x0.max(xa.min(xb)), x1.min(xa.max(xb)))
    } else if a >= y0 && a <= y1 {
        (x0, x1)
    } else {
        return None;
    };
    (lo.is_finite() && hi.is_finite() && hi > lo).then_some([(lo, a + b * lo), (hi, a + b * hi)])
}

/// The two ends of the line {x : c·x = level} inside a chart of two inputs (None if it
/// misses the chart, or there is no such line).
fn across(plot: &Plot, c: &Coef, level: f64) -> Option<[[f64; 2]; 2]> {
    let (b0, b1) = (c.b[0], c.b[1]);
    if b0.hypot(b1) < 1e-12 {
        return None;
    }
    let [x0, x1] = plot.x;
    let [y0, y1] = plot.y;
    let mut ends: Vec<[f64; 2]> = Vec::with_capacity(4);
    // where the line crosses each edge of the chart
    if b1.abs() > 1e-12 {
        for x in [x0, x1] {
            let y = (level - c.a - b0 * x) / b1;
            if y >= y0 - 1e-9 && y <= y1 + 1e-9 {
                ends.push([x, y]);
            }
        }
    }
    if b0.abs() > 1e-12 {
        for y in [y0, y1] {
            let x = (level - c.a - b1 * y) / b0;
            if x >= x0 - 1e-9 && x <= x1 + 1e-9 {
                ends.push([x, y]);
            }
        }
    }
    let first = *ends.first()?;
    let far = ends.iter().copied().max_by(|p, q| ((p[0] - first[0]).hypot(p[1] - first[1])).total_cmp(&(q[0] - first[0]).hypot(q[1] - first[1])))?;
    ((far[0] - first[0]).hypot(far[1] - first[1]) > 1e-9).then_some([first, far])
}

/// The part of a chart of two inputs where the log-odds lie between `lo` and `hi`: a
/// polygon (the chart's rectangle cut by two lines).
fn strip(plot: &Plot, c: &Coef, lo: f64, hi: f64) -> Vec<[f64; 2]> {
    let [x0, x1] = plot.x;
    let [y0, y1] = plot.y;
    let mut poly = vec![[x0, y0], [x1, y0], [x1, y1], [x0, y1]];
    // keep where s(x) ≥ lo, then where s(x) ≤ hi
    for (level, above) in [(lo, true), (hi, false)] {
        if !level.is_finite() {
            continue;
        }
        let inside = |p: &[f64; 2]| {
            let s = c.score(*p) - level;
            if above {
                s >= 0.0
            } else {
                s <= 0.0
            }
        };
        let mut out = Vec::with_capacity(poly.len() + 2);
        for i in 0..poly.len() {
            let (p, q) = (poly[i], poly[(i + 1) % poly.len()]);
            let (pin, qin) = (inside(&p), inside(&q));
            if pin {
                out.push(p);
            }
            if pin != qin {
                let (sp, sq) = (c.score(p) - level, c.score(q) - level);
                let t = sp / (sp - sq);
                if t.is_finite() {
                    out.push([p[0] + (q[0] - p[0]) * t, p[1] + (q[1] - p[1]) * t]);
                }
            }
        }
        poly = out;
        if poly.len() < 3 {
            return Vec::new();
        }
    }
    poly
}

/// The colour for the probability of yes in the plane: orange for near 0, cyan for near
/// 1, faint about 1/2.
fn shade(p: f64) -> Color {
    let (color, sure) = if p >= 0.5 { (DATA, p) } else { (OTHER, 1.0 - p) };
    color.a((0.004 + 0.026 * ((sure - 0.5) * 2.0) as f32).max(0.0).min(0.03))
}

// ----- what is drawn -----------------------------------------------------------------------

/// Where a point is drawn in its row: just above 1 for a yes, just below 0 for a no,
/// spread a little up and down so that the points can be counted.
fn row_height(yes: bool, jitter: f64) -> f64 {
    let off = 0.035 + 0.12 * jitter;
    if yes {
        1.0 + off
    } else {
        -off
    }
}

/// The picture of a frame: the model as drawn, and everything that follows from it.
struct Shown {
    coef: Coef,
    /// Each point as drawn (gliding after a change of the data): its inputs and its
    /// height in its row; where it is along the probability chart (the input, or with
    /// two inputs its log-odds); and the probability the model gives it.
    x: Vec<[f64; 2]>,
    height: Vec<f64>,
    along: Vec<f64>,
    probs: Vec<f64>,
    /// At the true inputs: the outcomes at the threshold, the ROC curve, the share of
    /// pairs ranked right, and what the fit makes as large as it can.
    table: Table,
    roc: Option<Roc>,
    ranked: Option<f64>,
    value: f64,
}

impl Logistic {
    /// The points as drawn glide to where the data put them (in the time of the clock:
    /// a change of the data answers the visitor's hand). Says whether they are moving.
    fn follow_points(&mut self, real_dt: f32) -> bool {
        let to: Vec<f32> = (0..self.data.len())
            .flat_map(|i| {
                let (x, yes) = (self.data.x[i], self.data.y[i]);
                [x[0] as f32, x[1] as f32, row_height(yes, self.draws.jitter[i]) as f32]
            })
            .collect();
        self.points.follow(&to, real_dt, true)
    }

    fn shown(&self) -> Shown {
        let coef = self.look.now();
        let d = &self.data;
        let n = d.len();
        let glided = self.points.shown.len() == 3 * n;
        let (mut x, mut height) = (Vec::with_capacity(n), Vec::with_capacity(n));
        for i in 0..n {
            if glided {
                let p = &self.points.shown[3 * i..3 * i + 3];
                x.push([p[0] as f64, p[1] as f64]);
                height.push(p[2] as f64);
            } else {
                x.push(d.x[i]);
                height.push(row_height(d.y[i], self.draws.jitter[i]));
            }
        }
        let along: Vec<f64> = x.iter().map(|x| if d.dims == 1 { x[0] } else { coef.score(*x) }).collect();
        let probs: Vec<f64> = x.iter().map(|x| prob(coef.score(*x))).collect();
        let scores: Vec<f64> = d.x.iter().map(|x| coef.score(*x)).collect();
        let true_probs: Vec<f64> = scores.iter().map(|s| prob(*s)).collect();
        Shown {
            coef,
            x,
            height,
            along,
            probs,
            table: Table::of(&true_probs, &d.y, self.threshold),
            roc: roc(&scores, &d.y),
            ranked: ranked_right(&scores, &d.y),
            value: objective(&coef, d, self.penalty),
        }
    }

    /// The name of what the fit makes as large as it can.
    fn value_name(&self) -> &'static str {
        if self.penalty > 0.0 {
            r"log-likelihood $- \tfrac{\lambda}{2} b^2$"
        } else {
            "log-likelihood"
        }
    }

    /// A number of the likelihood's, with as many places as it needs to show anything.
    fn ll_text(v: f64) -> String {
        if !v.is_finite() {
            return "?".to_string();
        }
        if v != 0.0 && v.abs() < 0.005 {
            let e = v.abs().log10().floor();
            let m = v / 10f64.powf(e);
            return format!(r"{} \times 10^{{{}}}", num(m, 1), e as i64);
        }
        num(v, 2)
    }

    /// What the fit has come to, in a line: the fullest way of saying it first, then
    /// shorter ones for a narrower chart.
    fn status(&self, shown: &Shown) -> Vec<String> {
        let d = &self.data;
        let name = self.value_name();
        let now = format!("${}$", Self::ll_text(shown.value));
        let k = self.run.taken;
        if d.one_class() {
            return if d.len() == 1 {
                vec!["one point: there is no curve to fit".to_string()]
            } else {
                let (all, level) = if d.yes == 0 { ("no", 0) } else { ("yes", 1) };
                vec![format!("every point is a {all}: the best curve would be {level} everywhere, which no curve reaches"), format!("every point is a {all}: no curve to fit")]
            };
        }
        let steepness = if d.dims == 1 { format!("$b = {}$", num(shown.coef.b[0], 2)) } else { format!(r"$|b| = {}$", num(shown.coef.steepness(), 2)) };
        if self.run.left > 0 || self.look.newton && self.look.moving() {
            return vec![format!("Newton's method, step {k}: {name} {now}, {steepness}"), format!("step {k}: {name} {now}"), format!("step {k}: {now}")];
        }
        // (inputs that do not spread out leave a slope the data cannot settle)
        let unsettled = match (d.same_inputs(), d.dims) {
            (false, _) => "",
            (true, 1) => "; every input is the same, so the data cannot settle the slope",
            (true, _) => "; the inputs lie on one line, so the data cannot settle the slope across it",
        };
        let apart = if d.separated() { "every yes lies on one side of every no, so there is " } else { "there is " };
        let top = format!("${}$", Self::ll_text(self.best.objective));
        match (self.run.end, self.best.top) {
            (Some(End::Cap), Top::Never) => vec![
                format!("{apart}no top: after {k} steps {steepness} and still rising; {name} {now}, creeping up to 0"),
                format!("no top: after {k} steps {steepness} and still rising"),
                format!("no top: {steepness} after {k} steps"),
            ],
            (Some(End::Cap), _) => vec![format!("{name} {now}: not at the top after {k} steps (press fit again)"), format!("not at the top after {k} steps")],
            (Some(End::Stuck), _) => vec![format!("{name} {now}: no further step can be worked out"), "no further step can be worked out".to_string()],
            (Some(End::Top), _) => vec![format!("{name} {now}: the top, after {k} steps{unsettled}"), format!("{name} {now}: the top, after {k} steps"), format!("{now}: the top")],
            (None, Top::Never) => vec![format!("{name} {now}; {apart}no top: the steeper, the higher, for ever (press fit)"), format!("{name} {now}; no top (press fit)"), "no top (press fit)".to_string()],
            (None, _) => {
                if (shown.value - self.best.objective).abs() <= 1e-6 * self.best.objective.abs().max(1.0) {
                    vec![format!("{name} {now}: the top{unsettled}"), format!("{name} {now}: the top"), format!("{now}: the top")]
                } else {
                    vec![format!("{name} {now}; the top is {top}{unsettled}"), format!("{name} {now}; the top is {top}"), format!("{now}; the top {top}")]
                }
            }
        }
    }
}

/// The first of `options` that fits in `room` logical pixels at `size` (the last if none does).
fn pick(f: &Frame, options: &[String], room: f32, size: f32) -> String {
    options.iter().find(|t| f.text_size(t, size).x <= room).or(options.last()).cloned().unwrap_or_default()
}

// ----- the panel ---------------------------------------------------------------------------

impl Scene for Logistic {
    fn view(&self) -> View {
        View::Flat { center: Vec2::ZERO, half_height: 1.0 }
    }

    fn settings(&self, s: &mut ViewSettings) {
        s.grid = false;
        s.bloom = s.bloom.min(0.2);
    }

    fn animated(&self) -> bool {
        self.run.left > 0 || self.look.moving() || self.slide.is_some() || self.held || self.points.shown.len() != 3 * self.data.len() || self.points_moving() || self.roc_moving
    }

    fn pointer_hint(&self) -> Option<&'static str> {
        Some("yellow dots: drag to set the curve by hand    white dot: drag the threshold    buttons over the picture: click")
    }

    fn ui(&mut self, ui: &mut egui::Ui, _: &Gfx) {
        ui::note(ui, "When an answer is yes or no (ill or well, spam or not), what a model of it can give is the probability of yes. A straight line is the wrong shape for a probability: it runs below 0 and above 1. Logistic regression makes the log of the odds a straight line in the input instead, and that bends the probability into a smooth step from 0 to 1:");
        ui::equation(ui, r"p(x) = \frac{1}{1 + e^{-(a + bx)}}, \quad \ln\frac{p(x)}{1 - p(x)} = a + bx");
        ui::note(ui, r"One more unit of $x$ adds $b$ to the log-odds, so it multiplies the odds of yes by $e^b$. The curve is fitted by maximum likelihood: each point counts the probability the curve gave to what actually happened there, and the fit makes the sum of the logs of those probabilities, the log-likelihood, as large as it can.");

        ui::section(ui, "Things to try");
        for (k, trial) in TRIALS.iter().enumerate() {
            if ui::experiment(ui, trial.button, trial.caption) {
                self.set_up(k);
            }
        }

        ui::section(ui, "The data");
        let mut dims = self.setup.dims.clamp(1, 2) - 1;
        if ui::choice(ui, &["one input", "two inputs"], &mut dims) {
            self.switch_dims(dims + 1);
        }
        let presets: Vec<(&str, Setup)> = if self.setup.dims == 2 { PRESETS_2.to_vec() } else { PRESETS_1.to_vec() };
        let names: Vec<&str> = presets.iter().map(|p| p.0).collect();
        if let Some(i) = ui::buttons(ui, &names) {
            if let Some(&(_, setup)) = presets.get(i) {
                self.preset(setup);
            }
        }
        let mut changed = ui::slider_int_log(ui, "points", &mut self.setup.n, 1..=MOST);
        let mut share = self.setup.share as f32;
        if ui::slider(ui, "share of yes's", &mut share, 0.0..=1.0) {
            (self.setup.share, changed) = (share as f64, true);
        }
        let mut apart = self.setup.apart as f32;
        if ui::slider(ui, r"distance between the classes $d$", &mut apart, 0.0..=5.0) {
            (self.setup.apart, changed) = (apart as f64, true);
        }
        let mut spread = self.setup.spread as f32;
        if ui::slider(ui, r"spread within a class $\sigma$", &mut spread, 0.0..=1.5) {
            (self.setup.spread, changed) = (spread as f64, true);
        }
        if self.setup.dims == 2 && self.setup.kind == Kind::Ring {
            ui::note(ui, r"The yes's are bell-shaped round the middle; the no's lie on a ring of radius $1 + d$ round them.");
        } else {
            ui::note(ui, r"The no's are bell-shaped round $-d/2$ and the yes's round $+d/2$, each with spread $\sigma$ (with two inputs, along a slant). Then the true probability of yes is itself a logistic curve, of slope $d/\sigma^2$: the green one.");
        }
        if ui.button("new sample").clicked() {
            self.setup.seed = self.setup.seed.wrapping_add(1);
            changed = true;
        }
        if changed {
            self.refresh();
        }

        ui::section(ui, "The model");
        if let Some(i) = ui::buttons(ui, &["fit", "one step"]) {
            self.start_run(if i == 0 { CAP } else { 1 });
        }
        ui::note(ui, "Fit runs Newton's method from the curve as it is, three steps a second, up to 30 steps; or take one step at a time.");
        let mut penalty = self.penalty as f32;
        if ui::slider_log_or_zero(ui, r"penalty $\lambda$", &mut penalty, 0.001, 10.0) {
            self.penalty = penalty as f64;
            self.refresh();
        }
        ui::note(ui, r"The fit makes the log-likelihood minus $\tfrac{\lambda}{2} b^2$ (with two inputs, $\tfrac{\lambda}{2}(b_1^2 + b_2^2)$) as large as it can: any $\lambda$ above 0 keeps the slope from running away, and pulls it towards flat.");
        if self.setup.dims == 1 {
            ui::checkbox(ui, &mut self.shows.least_squares, "the least-squares straight line");
            ui::checkbox(ui, &mut self.shows.odds, "the odds at two inputs one apart");
        }
        ui::checkbox(ui, &mut self.shows.truth, "the truth the data came from");

        ui::section(ui, "The decision");
        let mut threshold = self.threshold as f32;
        if ui::slider(ui, "threshold", &mut threshold, 0.0..=1.0) {
            self.threshold = threshold as f64;
            self.slide = None;
        }
        ui::note(ui, "Say yes wherever the probability is at least this.");
        self.refresh();

        ui::section(ui, "Reading the picture");
        ui::key(
            ui,
            &[
                (DATA, "cyan dots: the yes's (answer 1), in a row above 1"),
                (OTHER, "orange dots: the no's (answer 0), in a row below 0"),
                (ESTIMATE, "yellow: the model (the curve, its straight log-odds, the boundary of two inputs), its two handles, and the ROC curve"),
                (ALARM, "pink bars: for each point, the probability the model did not give to what happened; pink rings: points the threshold gets wrong"),
                (pal::WHITE, "white: the threshold, and where it puts the decision on the ROC curve"),
                (pal::TEXT, "grey dots: the log-odds of the share of yes's in each slice of the data (an arrow: all yes's or all no's)"),
                (TRUTH, "green dashes: the true curve the data were drawn from"),
                (pal::BLUE, "blue: the straight line of least squares"),
            ],
        );
        ui::section(ui, "Words");
        ui::term(ui, "Odds:", r"$p / (1 - p)$, how many times as likely yes is as no: a probability of 0.8 is odds of 4.");
        ui::term(ui, "Log-odds (logit):", r"$\ln\bigl(p / (1 - p)\bigr)$: 0 at $p = 1/2$, and any number from $-\infty$ to $\infty$, so a straight line can be one.");
        ui::term(ui, "Logistic curve:", r"$p = 1/(1 + e^{-z})$, the smooth step from 0 to 1 that turns log-odds $z$ back into a probability.");
        ui::term(ui, "Odds ratio:", r"how many times the odds are multiplied when $x$ grows by one: $e^b$.");
        ui::term(ui, "Log-likelihood (log loss):", "the sum over the points of the log of the probability the model gave to what happened. Minus it is the log loss.");
        ui::term(ui, "Maximum likelihood:", "the coefficients with the highest log-likelihood: the top.");
        ui::term(ui, "Newton's method:", "steps towards the top, each to the top of the dome-shaped quadratic that slopes and bends as the log-likelihood does where the step starts (shortened here when that would overshoot).");
        ui::term(ui, "Decision threshold:", "say yes wherever the probability is at least this.");
        ui::term(ui, "Hit rate (sensitivity):", "the share of the yes's said yes to.");
        ui::term(ui, "False alarm rate:", "the share of the no's said yes to (one less the specificity).");
        ui::term(ui, "ROC curve:", "the hit rate against the false alarm rate, as the threshold goes down from 1 to 0.");
        ui::term(ui, "Separation:", "every yes on one side of every no (with two inputs, of a straight line). Then a steeper curve is always more likely, and the likelihood has no top.");
        ui::term(ui, r"Penalty $\lambda$:", r"a price on the slope's size, $\tfrac{\lambda}{2} b^2$, taken off the log-likelihood (as in ridge regression).");

        ui::section(ui, "Numbers");
        let c = self.coef;
        ui::readout(ui, r"intercept $a$", format!("${}$", num(c.a, 3)));
        if self.setup.dims == 1 {
            ui::readout(ui, r"slope $b$", format!("${}$", num(c.b[0], 3)));
            ui::readout(ui, r"odds ratio $e^b$", format!("${}$", Self::ll_text(c.b[0].exp())));
        } else {
            ui::readout(ui, r"slopes $b_1$, $b_2$", format!("${}$, ${}$", num(c.b[0], 3), num(c.b[1], 3)));
        }
        let name = self.value_name();
        ui::readout(ui, name, format!("${}$", Self::ll_text(self.value())));
        let top = match self.best.top {
            Top::At => format!("${}$", Self::ll_text(self.best.objective)),
            Top::Never => "none: it rises for ever".to_string(),
            Top::OneClass => "no curve to fit".to_string(),
        };
        ui::readout(ui, "the top", top);
        if self.penalty > 0.0 && self.best.top == Top::At {
            ui::readout(ui, "  its log-likelihood", format!("${}$", Self::ll_text(self.best.log_likelihood)));
        }
        ui::readout(ui, "your best by hand", self.by_hand.map_or("not yet".to_string(), |v| format!("${}$", Self::ll_text(v))));
        ui::readout(ui, "Newton's steps", format!("${}$", self.run.taken));
        let t = Table::of(&self.data.x.iter().map(|x| prob(c.score(*x))).collect::<Vec<f64>>(), &self.data.y, self.threshold);
        let rate = |r: Option<f64>| r.map_or("none to count".to_string(), |r| format!("${}$", num(r, 2)));
        ui::readout(ui, "hit rate", rate(t.hit_rate()));
        ui::readout(ui, "false alarm rate", rate(t.false_alarm_rate()));
        ui::readout(ui, "share right", format!("${}$", num(t.accuracy(), 3)));
        let scores: Vec<f64> = self.data.x.iter().map(|x| c.score(*x)).collect();
        ui::readout(ui, "area under the ROC curve", roc(&scores, &self.data.y).map_or("none".to_string(), |r| format!("${}$", num(r.area, 3))));
        ui::readout(ui, "(yes, no) pairs ranked right", ranked_right(&scores, &self.data.y).map_or("none".to_string(), |r| format!("${}$", num(r, 3))));
        ui::see_also(
            ui,
            &[
                ("likelihood", "maximum likelihood with one parameter"),
                ("chance", "hit rates, false alarms and base rates"),
                ("entropy", "the log loss is a cross-entropy"),
                ("mlp", "a network's last layer is this model"),
            ],
        );
    }

    fn frame(&mut self, f: &mut Frame) {
        let lay = Layout::of(f.view_size(), f.top_inset, self.setup.dims);
        let plots = Plots::of(&lay, self.setup.dims);
        if !lay.nothing {
            if !lay.compact && self.buttons(f, &lay) {
                self.refresh();
            }
            self.handles(f, &lay, &plots);
        } else {
            self.held = false;
        }
        self.advance(f.dt, f.real_dt);
        self.gliding = self.follow_points(f.real_dt);
        if lay.nothing {
            return;
        }
        let shown = self.shown();
        if let (Some(hit), Some(alarm)) = (shown.table.hit_rate(), shown.table.false_alarm_rate()) {
            self.roc_moving = self.roc_point.follow([alarm as f32, hit as f32], f.real_dt);
        } else {
            self.roc_moving = false;
        }
        if self.setup.dims == 1 {
            self.draw_curve_chart(f, &lay, &plots.main, &shown, true);
            if !lay.compact {
                self.draw_logodds(f, &lay, &plots.second, &shown);
            }
        } else {
            self.draw_plane(f, &lay, &plots.main, &shown);
            if !lay.compact {
                self.draw_curve_chart(f, &lay, &plots.second, &shown, false);
            }
        }
        if !lay.compact {
            self.draw_roc(f, &lay, &plots.roc, &shown);
            self.draw_table(f, &lay, &shown);
        }
    }
}

// ----- what the visitor does on the picture ------------------------------------------------

impl Logistic {
    fn points_moving(&self) -> bool {
        self.gliding
    }

    /// One input or two: the other's first data set, and the model fitted to it from a
    /// flat start, so that the fit is seen to find it.
    fn switch_dims(&mut self, dims: usize) {
        if dims == self.setup.dims || !(1..=2).contains(&dims) {
            return;
        }
        self.setup = if dims == 2 { CLOUDS } else { OVERLAP };
        self.shows.least_squares = false;
        self.shows.odds = false;
        self.anchor = [0.0, 0.0];
        self.refresh();
        let share = self.data.yes as f64 / self.data.len().max(1) as f64;
        let flat = Coef { a: logit(share.max(0.01).min(0.99)), b: [0.0, 0.0] };
        self.set_model(flat, true);
        self.by_hand = None;
        self.start_run(CAP);
    }

    /// One of the panel's data sets, with the model gliding to its top (where it has one).
    fn preset(&mut self, setup: Setup) {
        let dims_before = self.setup.dims;
        self.setup = setup;
        self.refresh();
        if self.best.top == Top::At {
            self.set_model(self.best.coef, dims_before != setup.dims);
        } else {
            self.run = Run::default();
        }
    }

    /// The buttons over the picture; true if the data changed.
    fn buttons(&mut self, f: &mut Frame, lay: &Layout) -> bool {
        let size = lay.small;
        let mut changed = false;
        if plot::button(f, lay.buttons[0], BUTTON_WORDS[0], size, self.run.left > 1) {
            self.start_run(CAP);
        }
        if plot::button(f, lay.buttons[1], BUTTON_WORDS[1], size, false) {
            self.start_run(1);
        }
        for k in 0..2 {
            let dims = k + 1;
            if plot::button(f, lay.buttons[2 + k], BUTTON_WORDS[2 + k], size, self.setup.dims == dims) && self.setup.dims != dims {
                self.switch_dims(dims);
                changed = true;
            }
        }
        changed
    }

    /// The handles: the model's two (its midpoint and its steepness), the threshold, and
    /// the pair of inputs the odds are compared at.
    fn handles(&mut self, f: &mut Frame, lay: &Layout, plots: &Plots) {
        let c = self.look.now();
        if c.finite() {
            if self.setup.dims == 1 {
                let main = &plots.main;
                if c.b[0].abs() > 1e-9 {
                    let mid = -c.a / c.b[0];
                    // (the side is chosen for the model itself, where the drawing is going:
                    // it changes as a new setting is made, not on the way there)
                    let to = self.coef;
                    if !f.handles.dragging() && to.b[0].abs() > 1e-9 {
                        let inside = |x: f64| (X_RANGE[0]..=X_RANGE[1]).contains(&x);
                        let m = -to.a / to.b[0];
                        self.steep_side = if !inside(m + STEEP_AT / to.b[0]) && inside(m - STEEP_AT / to.b[0]) { -1.0 } else { 1.0 };
                    }
                    let side = self.steep_side;
                    let steep = mid + side * STEEP_AT / c.b[0];
                    if let Some((x, _)) = main.handle(f, 1, mid, 0.5, ESTIMATE) {
                        self.by_hand_to(Coef::through(x, c.b[0]));
                    }
                    // (a curve so flat that its log-odds reach neither +2 nor −2 on the chart
                    // has no place for this handle)
                    let shown = (X_RANGE[0]..=X_RANGE[1]).contains(&steep);
                    let held = if shown { main.handle(f, 2, steep, prob(side * STEEP_AT), ESTIMATE) } else { None };
                    if let Some((x, _)) = held {
                        // (it stays on its side of the midpoint, no nearer than a tenth: the
                        // curve keeps its direction, and its slope stays below 20)
                        let away = side * c.b[0].signum();
                        let run = ((x - mid) * away).max(0.1) * away;
                        self.by_hand_to(Coef::through(mid, side * STEEP_AT / run));
                    }
                }
                if self.shows.odds {
                    let x0 = self.odds_at;
                    if let Some((x, _)) = main.handle(f, 4, x0, prob(c.score([x0, 0.0])), ESTIMATE) {
                        self.odds_at = x.max(X_RANGE[0]).min(X_RANGE[1] - 1.0);
                    }
                }
            } else {
                let main = &plots.main;
                let b2 = c.b[0] * c.b[0] + c.b[1] * c.b[1];
                if b2 > 1e-12 {
                    let anchor = [self.anchor[0].max(PLANE[0]).min(PLANE[1]), self.anchor[1].max(PLANE[0]).min(PLANE[1])];
                    let off = c.score(anchor) / b2;
                    let p0 = [anchor[0] - off * c.b[0], anchor[1] - off * c.b[1]];
                    let toward = |side: f64| [p0[0] + side * STEEP_AT * c.b[0] / b2, p0[1] + side * STEEP_AT * c.b[1] / b2];
                    let to = self.coef;
                    let to2 = to.b[0] * to.b[0] + to.b[1] * to.b[1];
                    if !f.handles.dragging() && to2 > 1e-12 {
                        let inside = |p: [f64; 2]| p.iter().all(|v| *v >= PLANE[0] && *v <= PLANE[1]);
                        let off = to.score(anchor) / to2;
                        let q = [anchor[0] - off * to.b[0], anchor[1] - off * to.b[1]];
                        let at = |side: f64| [q[0] + side * STEEP_AT * to.b[0] / to2, q[1] + side * STEEP_AT * to.b[1] / to2];
                        self.steep_side = if !inside(at(1.0)) && inside(at(-1.0)) { -1.0 } else { 1.0 };
                    }
                    let side = self.steep_side;
                    let steep = toward(side);
                    if p0[0].is_finite() && p0[1].is_finite() {
                        self.anchor = p0;
                    }
                    let shown = steep.iter().all(|v| *v >= PLANE[0] && *v <= PLANE[1]);
                    // (the two names on either side, along the line from one handle to the other)
                    let away = (main.at(steep[0], steep[1]) - main.at(p0[0], p0[1])).normalize_or(Vec2::X);
                    plot_label(f, main, p0, -away, "boundary", lay.small);
                    if shown {
                        plot_label(f, main, steep, away, "steepness", lay.small);
                    }
                    if let Some((x, y)) = main.handle(f, 1, p0[0], p0[1], ESTIMATE) {
                        self.anchor = [x, y];
                        self.by_hand_to(Coef { a: -(c.b[0] * x + c.b[1] * y), b: c.b });
                    }
                    let held = if shown { main.handle(f, 2, steep[0], steep[1], ESTIMATE) } else { None };
                    if let Some((x, y)) = held {
                        let v = [x - p0[0], y - p0[1]];
                        let len2 = (v[0] * v[0] + v[1] * v[1]).max(0.01);
                        let b = [side * STEEP_AT * v[0] / len2, side * STEEP_AT * v[1] / len2];
                        self.by_hand_to(Coef { a: -(b[0] * p0[0] + b[1] * p0[1]), b });
                    }
                }
            }
        }
        let chart = if self.setup.dims == 1 { Some(&plots.main) } else { (!lay.compact).then_some(&plots.second) };
        if let Some(chart) = chart {
            let x = chart.x[1] - 0.03 * (chart.x[1] - chart.x[0]);
            if let Some((_, y)) = chart.handle(f, 3, x, self.threshold, pal::WHITE) {
                self.threshold = y.max(0.0).min(1.0);
                self.slide = None;
            }
        }
        self.held = f.handles.dragging();
    }
}

// ----- drawing -----------------------------------------------------------------------------

/// The name of a handle of the plane, beside it on the side `toward` (a direction on the
/// screen), where it is drawn: kept inside the chart.
fn plot_label(f: &mut Frame, plot: &Plot, at: [f64; 2], toward: Vec2, text: &str, size: f32) {
    let p = plot.at(at[0], at[1]).max(plot.rect.0).min(plot.rect.1);
    let half = f.text_size(text, size) * 0.5;
    // (far enough that the text's box clears the handle, whichever way it lies)
    let reach = 10.0 + (toward.x.abs() * half.x).max(toward.y.abs() * half.y);
    if p.is_finite() && toward.is_finite() {
        f.text_px(p + toward * reach, text, size, ESTIMATE.a(0.85), Align2::CENTER_CENTER);
    }
}

/// A probability as a tick's number: 0, 0.25, 0.5, 0.75, 1.
fn tick_name(v: f64) -> String {
    let s = format!("{v:.2}");
    s.trim_end_matches('0').trim_end_matches('.').to_string()
}

/// Odds, or any positive number that may be very large or very small, to write.
fn odds_text(v: f64) -> String {
    if !v.is_finite() {
        return r"\infty".to_string();
    }
    if (0.01..1000.0).contains(&v) {
        num(v, 2)
    } else {
        Logistic::ll_text(v)
    }
}

impl Logistic {
    /// The probability of yes against the input (one input), or against the log-odds
    /// (two): the curve, the points in their rows with their bars, the threshold.
    fn draw_curve_chart(&self, f: &mut Frame, lay: &Layout, plot: &Plot, s: &Shown, one: bool) {
        if plot.tiny() {
            return;
        }
        let (text, small) = (lay.text, lay.small);
        let lines = if one { 2.0 } else { 1.0 };
        plot.backing(f, if lay.narrow { 34.0 } else { 40.0 }, lay.foot + lines * lay.line + 4.0, lay.head, 12.0);
        let (min, max) = plot.rect;
        let [x0, x1] = plot.x;
        let c = s.coef;
        let model = |u: f64| if one { c.a + c.b[0] * u } else { u };
        // the frame of the probabilities: faint lines at 0, 1/2 and 1, numbers from 0 to 1
        for p in [0.0, 0.5, 1.0] {
            plot.hline(f, AXES, p, 1.0, pal::DIM.a(if p == 0.5 { 0.16 } else { 0.32 }), false);
        }
        plot::line_of(f, AXES, Vec2::new(min.x, plot.px_y(0.0)), Vec2::new(min.x, plot.px_y(1.0)), 1.0, pal::DIM.a(0.8));
        for v in [0.0, 0.25, 0.5, 0.75, 1.0] {
            let y = plot.px_y(v);
            plot::line_of(f, AXES, Vec2::new(min.x - 4.0, y), Vec2::new(min.x, y), 1.0, pal::DIM.a(0.8));
            f.text_px(Vec2::new(min.x - 6.0, y), tick_name(v), small, pal::DIM, Align2::RIGHT_CENTER);
        }
        plot.x_axis(f, if one { "$x$" } else { r"log-odds $s = a + b_1 x_1 + b_2 x_2$" }, small);
        f.text_px(Vec2::new(min.x - 30.0, min.y - 6.0), "probability of yes", small, pal::DIM, Align2::LEFT_BOTTOM);
        let title = match (one, lay.narrow) {
            (true, false) => r"the data, and the model: $p(x) = 1/(1 + e^{-(a + bx)})$",
            (true, true) => r"the model $p(x) = 1/(1 + e^{-(a + bx)})$",
            (false, false) => r"the same along the slope's direction: $p = 1/(1 + e^{-s})$",
            (false, true) => r"along the slope: $p = 1/(1 + e^{-s})$",
        };
        plot.title(f, title, text);
        if one {
            f.text_px(Vec2::new(max.x, min.y - 6.0), format!("$a = {}$, $b = {}$", num(c.a, 2), num(c.b[0], 2)), small, ESTIMATE, Align2::RIGHT_BOTTOM);
        }

        // the straight line of least squares, and where it leaves 0 to 1
        if one && self.shows.least_squares {
            let (i, k) = self.line;
            let ls = |x: f64| i + k * x;
            // (blue while it could be a probability, pink beyond: each part a curve of
            // its own, NaN where it is the other's)
            let inside = |x: f64| (0.0..=1.0).contains(&ls(x));
            plot.curve(f, CURVES, 160, 2.2, pal::BLUE, |x| if inside(x) { ls(x) } else { f64::NAN });
            plot.curve(f, CURVES, 160, 2.2, ALARM, |x| if inside(x) { f64::NAN } else { ls(x) });
            if k.abs() > 1e-9 {
                let (at0, at1) = (-i / k, (1.0 - i) / k);
                let (lo, hi) = if k > 0.0 { (at0, at1) } else { (at1, at0) };
                if lo > x0 && lo < x1 {
                    plot.dot(f, lo, if k > 0.0 { 0.0 } else { 1.0 }, 7.0, pal::BLUE);
                }
                if hi > x0 && hi < x1 {
                    plot.dot(f, hi, if k > 0.0 { 1.0 } else { 0.0 }, 7.0, pal::BLUE);
                }
                let below = if k > 0.0 { format!(r"below 0 for $x < {}$", num(at0, 2)) } else { format!(r"below 0 for $x > {}$", num(at0, 2)) };
                let above = if k > 0.0 { format!(r"above 1 for $x > {}$", num(at1, 2)) } else { format!(r"above 1 for $x < {}$", num(at1, 2)) };
                let at = |x: f64, y: f64| plot.at(x.max(x0).min(x1), y);
                f.caption_px(at(at0, 0.0) + Vec2::new(8.0, -10.0), below, small, pal::BLUE, Align2::LEFT_BOTTOM);
                f.caption_px(at(at1, 1.0) + Vec2::new(-8.0, 10.0), above, small, pal::BLUE, Align2::RIGHT_TOP);
            }
        }
        // the truth the data came from
        if let (true, Some(t)) = (self.shows.truth, truth(&self.setup, &self.data)) {
            if one {
                dashed_curve(f, plot, DASHES, 1.8, TRUTH, |x| prob(t.score([x, 0.0])));
            }
        }
        // the model
        plot.curve(f, CURVES, 200, 2.6, ESTIMATE, |u| prob(model(u)));

        // each point: its bar (what the model did not give to what happened), whether the
        // threshold gets it wrong, and the point
        let d = &self.data;
        let along = |i: usize| s.along[i].max(x0).min(x1);
        for i in 0..d.len() {
            let target = if d.y[i] { 1.0 } else { 0.0 };
            plot.path(f, BARS, [(along(i), s.probs[i]), (along(i), target)], 1.6, ALARM.a(0.5));
        }
        for i in 0..d.len() {
            let wrong = (s.probs[i] >= self.threshold) != d.y[i];
            plot.dot(f, along(i), s.height[i], if lay.narrow { 10.0 } else { 11.5 }, if wrong { ALARM } else { Color::CLEAR });
        }
        for i in 0..d.len() {
            plot.dot(f, along(i), s.height[i], if lay.narrow { 5.5 } else { 6.5 }, if d.y[i] { DATA } else { OTHER });
        }
        plot.text(f, x0, 1.0 + 0.095, Vec2::new(6.0, 0.0), "yes's: 1", small, DATA, Align2::LEFT_CENTER);
        // the names of the model's two handles, on the side of each where the curve is not
        if one && c.b[0].abs() > 1e-9 {
            let mid = -c.a / c.b[0];
            let side = if c.b[0] > 0.0 { 1.0 } else { -1.0 };
            let align = if c.b[0] > 0.0 { Align2::LEFT_TOP } else { Align2::RIGHT_TOP };
            plot.text(f, mid, 0.5, Vec2::new(10.0 * side, 7.0), "midpoint", small, ESTIMATE.a(0.85), align);
            let up = self.steep_side;
            plot.text(f, mid + up * STEEP_AT / c.b[0], prob(up * STEEP_AT), Vec2::new(10.0 * side, if up > 0.0 { 7.0 } else { -7.0 }), "steepness", small, ESTIMATE.a(0.85), if up > 0.0 { align } else if c.b[0] > 0.0 { Align2::LEFT_BOTTOM } else { Align2::RIGHT_BOTTOM });
        }
        plot.text(f, x1, -0.095, Vec2::new(-6.0, 0.0), "no's: 0", small, OTHER, Align2::RIGHT_CENTER);

        // the threshold, and where the curve crosses it: said no on one side, yes on the other
        let t = self.threshold;
        plot.hline(f, MARKS, t, 1.3, pal::WHITE.a(0.8), true);
        plot.text(f, x1, t, Vec2::new(-22.0, -5.0), &format!("threshold ${}$", num(t, 2)), small, pal::WHITE, Align2::RIGHT_BOTTOM);
        let z = logit(t);
        let cross = if one { (c.b[0].abs() > 1e-12).then(|| (z - c.a) / c.b[0]) } else { Some(z) };
        if let Some(u) = cross.filter(|u| u.is_finite() && *u > x0 && *u < x1) {
            plot::dashed_of(f, MARKS, plot.at(u, 0.0), plot.at(u, t), 1.2, pal::WHITE.a(0.6));
            let rising = !one || c.b[0] > 0.0;
            let (left, right) = if rising { ("said no", "said yes") } else { ("said yes", "said no") };
            // (each where there is room for it between the line and the chart's edge)
            let at = plot.px_x(u);
            if at - min.x > f.text_size(left, small).x + 10.0 {
                plot.text(f, u, -0.205, Vec2::new(-6.0, 0.0), left, small, pal::DIM, Align2::RIGHT_CENTER);
            }
            if max.x - at > f.text_size(right, small).x + 10.0 {
                plot.text(f, u, -0.205, Vec2::new(6.0, 0.0), right, small, pal::DIM, Align2::LEFT_CENTER);
            }
        }

        // the odds at two inputs one apart
        if one && self.shows.odds {
            let (u0, u1) = (self.odds_at, self.odds_at + 1.0);
            let (p0, p1) = (prob(model(u0)), prob(model(u1)));
            let (o0, o1) = (model(u0).exp(), model(u1).exp());
            plot.dot(f, u1, p1, 9.0, ESTIMATE);
            for (u, p) in [(u0, p0), (u1, p1)] {
                plot::dashed_of(f, ODDS, plot.at(u, 0.0), plot.at(u, p), 1.0, ESTIMATE.a(0.7));
            }
            let left = format!(r"$p = {}$, odds ${}$", num(p0, 2), odds_text(o0));
            let right = format!(r"$p = {}$, odds ${}$", num(p1, 2), odds_text(o1));
            let rising = c.b[0] >= 0.0;
            f.caption_px(plot.at(u0, p0) + Vec2::new(-12.0, if rising { -4.0 } else { 4.0 }), left, small, ESTIMATE, if rising { Align2::RIGHT_BOTTOM } else { Align2::RIGHT_TOP });
            f.caption_px(plot.at(u1, p1) + Vec2::new(12.0, if rising { 4.0 } else { -4.0 }), right, small, ESTIMATE, if rising { Align2::LEFT_TOP } else { Align2::LEFT_BOTTOM });
            f.caption_px(plot.at(u0 + 0.5, 0.0) + Vec2::new(0.0, -8.0), format!(r"one unit: odds $\times\,{}$", odds_text(c.b[0].exp())), small, ESTIMATE, Align2::CENTER_BOTTOM);
        }

        // what the fit has come to, and what the bars are
        let mut y = max.y + lay.foot;
        // (as far as the canvas's edge on a phone, the column beside it on a desktop)
        let room = if lay.narrow { lay.view.x - 8.0 } else { max.x + 40.0 } - (min.x - 30.0);
        if one {
            let status = pick(f, &self.status(s), room, small);
            f.text_px(Vec2::new(min.x - 30.0, y), status, small, pal::TEXT, Align2::LEFT_TOP);
            y += lay.line;
        }
        let keys = [
            "pink bars: the probability the model did not give to what happened (the shorter, the likelier)".to_string(),
            "pink bars: the probability not given to what happened".to_string(),
            "pink bars: probability not given".to_string(),
        ];
        f.text_px(Vec2::new(min.x - 30.0, y), pick(f, &keys, room, small), small, ALARM, Align2::LEFT_TOP);
    }

    /// The same model as log-odds against the input: a straight line, with the log-odds
    /// of the share of yes's in slices of the data.
    fn draw_logodds(&self, f: &mut Frame, lay: &Layout, plot: &Plot, s: &Shown) {
        if plot.tiny() {
            return;
        }
        let (text, small) = (lay.text, lay.small);
        plot.backing(f, if lay.narrow { 34.0 } else { 40.0 }, lay.foot + lay.line + 4.0, lay.head, 12.0);
        let (min, max) = plot.rect;
        let [x0, x1] = plot.x;
        let c = s.coef;
        plot.y_grid(f);
        plot.y_axis(f, "log-odds", small);
        plot.x_axis(f, "$x$", small);
        let title = if lay.narrow { r"as log-odds, $\ln\bigl(p/(1-p)\bigr) = a + bx$" } else { r"the same model as log-odds, $\ln\bigl(p/(1-p)\bigr) = a + bx$: a straight line" };
        plot.title(f, title, text);
        // the threshold as log-odds
        let z = logit(self.threshold);
        if z.is_finite() {
            plot.hline(f, MARKS, z, 1.2, pal::WHITE.a(0.6), true);
        }
        if let (true, Some(t)) = (self.shows.truth, truth(&self.setup, &self.data)) {
            dashed_curve(f, plot, DASHES, 1.8, TRUTH, |x| t.score([x, 0.0]));
        }
        if let Some([p, q]) = line_in(plot, c.a, c.b[0]) {
            plot.path(f, CURVES, [p, q], 2.6, ESTIMATE);
        }
        // the slices of the data: a share of 0 or 1 is minus or plus infinity, an arrow at the edge
        let [y0, y1] = plot.y;
        let edge = (y1 - y0) * 0.04;
        for sl in &self.slices {
            let v = logit(sl.share);
            let x = sl.x.max(x0).min(x1);
            if v.is_finite() && v > y0 && v < y1 {
                plot.dot(f, x, v, 8.0, pal::TEXT);
                // (the arrows' batch keeps its count: a slice shown as a dot has an empty one)
                plot::path_of(f, ARROWS, &[Vec2::NAN], 1.4, pal::TEXT);
            } else {
                let up = v > 0.0;
                let tip = if up { y1 - edge * 0.3 } else { y0 + edge * 0.3 };
                let base = if up { y1 - edge * 2.2 } else { y0 + edge * 2.2 };
                plot.dot(f, x, base, 8.0, pal::TEXT);
                let (b, t) = (plot.at(x, base), plot.at(x, tip));
                plot::path_of(f, ARROWS, &[b, t, Vec2::NAN, t + Vec2::new(-4.0, if up { 5.0 } else { -5.0 }), t, t + Vec2::new(4.0, if up { 5.0 } else { -5.0 })], 1.4, pal::TEXT);
            }
        }
        // one unit more of x adds b to the log-odds
        if self.shows.odds {
            let u0 = self.odds_at;
            let s0 = c.a + c.b[0] * u0;
            let corner = (u0 + 1.0, s0);
            let pts = [plot.at(u0, s0), plot.at(corner.0, corner.1), plot.at(u0 + 1.0, s0 + c.b[0])];
            plot::path_of(f, ODDS, &plot::clip(&pts, plot.rect), 1.6, ESTIMATE.a(0.8));
            plot.text(f, u0 + 0.5, s0, Vec2::new(0.0, 4.0), "1", small, ESTIMATE, Align2::CENTER_TOP);
            plot.text(f, u0 + 1.0, s0 + c.b[0] * 0.5, Vec2::new(6.0, 0.0), &format!("$b = {}$", num(c.b[0], 2)), small, ESTIMATE, Align2::LEFT_CENTER);
        }
        let n = self.slices.first().map_or(0, |s| s.n);
        let keys = [
            format!("grey dots: the log-odds of the share of yes's in each slice of {n} points along $x$ (an arrow: all yes's, or none)"),
            format!("grey: the log-odds of the yes's share in slices of {n} points"),
            "grey: log-odds of slices of the data".to_string(),
        ];
        let room = if lay.narrow { lay.view.x - 8.0 } else { max.x + 40.0 } - (min.x - 30.0);
        f.text_px(Vec2::new(min.x - 30.0, max.y + lay.foot), pick(f, &keys, room, small), small, pal::TEXT, Align2::LEFT_TOP);
    }

    /// Two inputs: the points in the plane, shaded by the probability of yes, with the
    /// boundary where it is one half and the lines where it is 0.1, 0.25, 0.75 and 0.9.
    fn draw_plane(&self, f: &mut Frame, lay: &Layout, plot: &Plot, s: &Shown) {
        if plot.tiny() {
            return;
        }
        let (text, small) = (lay.text, lay.small);
        plot.backing(f, if lay.narrow { 34.0 } else { 40.0 }, lay.foot + 2.0 * lay.line + 4.0, lay.head, 12.0);
        let (min, max) = plot.rect;
        let c = s.coef;
        // the shading, strip by strip between the lines of equal probability
        let levels = [0.1, 0.25, 0.5, 0.75, 0.9].map(logit);
        let mut edges = vec![f64::NEG_INFINITY];
        edges.extend(levels);
        edges.push(f64::INFINITY);
        for w in edges.windows(2) {
            let (lo, hi) = (w[0], w[1]);
            let middle = if lo.is_finite() && hi.is_finite() { prob((lo + hi) * 0.5) } else if lo.is_finite() { 0.95 } else { 0.05 };
            let poly = strip(plot, &c, lo, hi);
            if poly.len() >= 3 {
                let pts: Vec<Vec3> = poly.iter().map(|p| f.hud_at(1, plot.at(p[0], p[1]))).collect();
                for i in 1..pts.len() - 1 {
                    f.dl.tris_alpha.tri(pts[0], pts[i], pts[i + 1], shade(middle));
                }
            }
        }
        plot.x_axis(f, "$x_1$", small);
        plot.y_axis(f, "$x_2$", small);
        let title = if lay.narrow { r"two inputs: $p = 1/2$ on a straight line" } else { r"two inputs: the boundary where $p = 1/2$ is a straight line" };
        plot.title(f, title, text);
        f.text_px(Vec2::new(max.x, min.y - 6.0), format!("$a = {}$, $b_1 = {}$, $b_2 = {}$", num(c.a, 2), num(c.b[0], 2), num(c.b[1], 2)), small, ESTIMATE, Align2::RIGHT_BOTTOM);
        // the lines of equal probability, the boundary bold, each named where it passes
        // nearest to a point a little way along the boundary from the middle: the names
        // lie in a row across the bands
        let b2 = c.b[0] * c.b[0] + c.b[1] * c.b[1];
        let along = if b2 > 1e-12 { [-c.b[1] / b2.sqrt(), c.b[0] / b2.sqrt()] } else { [1.0, 0.0] };
        let along = if along[0] - along[1] < 0.0 { [-along[0], -along[1]] } else { along };
        let reach = 0.3 * (plot.x[1] - plot.x[0]);
        let mid = [(plot.x[0] + plot.x[1]) * 0.5, (plot.y[0] + plot.y[1]) * 0.5];
        let near = [mid[0] + reach * along[0], mid[1] + reach * along[1]];
        for (k, &level) in levels.iter().enumerate() {
            if let Some([p, q]) = across(plot, &c, level) {
                let boundary = k == 2;
                plot.path(f, CONTOURS, [(p[0], p[1]), (q[0], q[1])], if boundary { 2.6 } else { 1.0 }, ESTIMATE.a(if boundary { 1.0 } else { 0.45 }));
                let name = ["0.1", "0.25", "1/2", "0.75", "0.9"][k];
                let off = (c.score(near) - level) / b2.max(1e-12);
                let at = [near[0] - off * c.b[0], near[1] - off * c.b[1]];
                let px = plot.at(at[0], at[1]);
                // (only the boundary's name where the lines are too close together for all)
                let gap = (plot.px_x(1.0) - plot.px_x(0.0)) as f64 * (levels[3] - levels[2]) / b2.sqrt().max(1e-12);
                if px.is_finite() && plot::within(px, plot.rect, -8.0) && (boundary || gap >= 20.0) {
                    f.caption_px(px, format!("${name}$"), small, ESTIMATE.a(if boundary { 1.0 } else { 0.8 }), Align2::CENTER_CENTER);
                }
            }
        }
        // the threshold, as a line of the plane
        let z = logit(self.threshold);
        if let Some([p, q]) = across(plot, &c, z).filter(|_| z.is_finite()) {
            let cut = [plot.at(p[0], p[1]), plot.at(q[0], q[1])];
            plot::path_of(f, DASHES, &dashes(&cut, 7.0, 5.0), 1.3, pal::WHITE.a(0.8));
        }
        if let (true, Some(t)) = (self.shows.truth, truth(&self.setup, &self.data)) {
            if let Some([p, q]) = across(plot, &t, 0.0) {
                let cut = [plot.at(p[0], p[1]), plot.at(q[0], q[1])];
                plot::path_of(f, DASHES, &dashes(&cut, 7.0, 5.0), 1.8, TRUTH);
            }
        }
        // the points
        let d = &self.data;
        let at = |i: usize| (s.x[i][0].max(PLANE[0]).min(PLANE[1]), s.x[i][1].max(PLANE[0]).min(PLANE[1]));
        for i in 0..d.len() {
            let wrong = (s.probs[i] >= self.threshold) != d.y[i];
            let (x, y) = at(i);
            plot.dot(f, x, y, if lay.narrow { 10.0 } else { 11.5 }, if wrong { ALARM } else { Color::CLEAR });
        }
        for i in 0..d.len() {
            let (x, y) = at(i);
            plot.dot(f, x, y, if lay.narrow { 5.5 } else { 6.5 }, if d.y[i] { DATA } else { OTHER });
        }
        let y = max.y + lay.foot;
        // (the plane is square: the words under it may run on, to the canvas's edge on a
        // phone, to the column beside it on a desktop)
        let end = if lay.narrow { lay.view.x - 8.0 } else { lay.table.0.x - 16.0 };
        let room = end - (min.x - 30.0);
        let status = pick(f, &self.status(s), room, small);
        f.text_px(Vec2::new(min.x - 30.0, y), status, small, pal::TEXT, Align2::LEFT_TOP);
        let keys = [
            "shading: the probability of yes, cyan near 1 and orange near 0; pink rings: what the threshold (white) gets wrong".to_string(),
            "shading: the probability of yes (cyan near 1, orange near 0)".to_string(),
            "shading: the probability of yes".to_string(),
        ];
        let key = pick(f, &keys, room, small);
        f.text_px(Vec2::new(min.x - 30.0, y + lay.line), key, small, pal::DIM, Align2::LEFT_TOP);
    }

    /// The hit rate against the false alarm rate at every threshold, the area under it,
    /// and where the threshold puts the decision.
    fn draw_roc(&self, f: &mut Frame, lay: &Layout, plot: &Plot, s: &Shown) {
        if plot.tiny() {
            return;
        }
        let (text, small) = (lay.text, lay.small);
        plot.backing(f, if lay.narrow { 34.0 } else { 40.0 }, lay.foot + 2.0 * lay.line + 4.0, lay.head, 12.0);
        let (min, max) = plot.rect;
        plot.y_grid(f);
        plot.x_axis(f, "false alarm rate", small);
        plot.y_axis(f, "hit rate", small);
        plot.title(f, if lay.narrow { "every threshold: ROC" } else { "every threshold at once: the ROC curve" }, text);
        plot::dashed_of(f, MARKS, plot.at(0.0, 0.0), plot.at(1.0, 1.0), 1.0, pal::DIM.a(0.7));
        plot.text(f, 0.5, 0.5, Vec2::new(8.0, 2.0), "chance", small, pal::DIM, Align2::LEFT_TOP);
        let Some(r) = &s.roc else {
            f.caption_px((min + max) * 0.5, "needs yes's and no's", small, pal::DIM, Align2::CENTER_CENTER);
            return;
        };
        for w in r.points.windows(2) {
            let (a, b) = (w[0], w[1]);
            if b[0] > a[0] {
                let c = [plot.at(a[0], 0.0), plot.at(a[0], a[1]), plot.at(b[0], b[1]), plot.at(b[0], 0.0)].map(|p| f.hud_at(1, p));
                f.dl.tris_alpha.quad(c[0], c[1], c[2], c[3], ESTIMATE.a(0.014));
            }
        }
        let pts: Vec<Vec2> = r.points.iter().map(|p| plot.at(p[0], p[1])).collect();
        plot::path_of(f, ROC_LINE, &pts, 2.2, ESTIMATE);
        // where the threshold is
        if let (Some([alarm, hit]), Some(_), Some(_)) = (self.roc_point.shown, s.table.hit_rate(), s.table.false_alarm_rate()) {
            // (drawn gliding: the counts step from one corner of the curve to the next)
            let (alarm, hit) = (alarm as f64, hit as f64);
            plot.dot(f, alarm, hit, 10.0, pal::WHITE);
            let right = alarm < 0.55;
            let at = plot.at(alarm, hit);
            let label = format!("threshold ${}$", num(self.threshold, 2));
            let w = f.text_size(&label, small).x;
            let offset = if right { Vec2::new(10.0 + w * 0.5, 12.0) } else { Vec2::new(-10.0 - w * 0.5, -12.0) };
            f.text_placed("roc point", 0, at, offset, label, small, pal::WHITE);
        }
        let y = max.y + lay.foot;
        let ranked = s.ranked.map_or("?".to_string(), |v| num(v, 3));
        // (on a phone the table is beside the chart; on a desktop, the canvas's edge)
        let room = if lay.narrow { lay.table.0.x - 8.0 } else { lay.view.x - 12.0 } - (min.x - 30.0);
        let area = [format!("area under the curve ${}$", num(r.area, 3)), format!("area ${}$", num(r.area, 3))];
        f.text_px(Vec2::new(min.x - 30.0, y), pick(f, &area, room, small), small, ESTIMATE, Align2::LEFT_TOP);
        let pairs = [format!("(yes, no) pairs whose yes is given more: ${ranked}$"), format!("pairs ranked right ${ranked}$"), format!("pairs right ${ranked}$")];
        f.text_px(Vec2::new(min.x - 30.0, y + lay.line), pick(f, &pairs, room, small), small, pal::TEXT, Align2::LEFT_TOP);
    }

    /// The four outcomes of the decision, and the rates made from them.
    fn draw_table(&self, f: &mut Frame, lay: &Layout, s: &Shown) {
        let (min, max) = lay.table;
        let (w, h) = (max.x - min.x, max.y - min.y);
        if w < 140.0 || h < 120.0 {
            return;
        }
        let (text, small) = (lay.text, lay.small);
        hud_panel(f, min - Vec2::new(6.0, 4.0), max + Vec2::new(6.0, 0.0));
        let t = &s.table;
        f.text_px(min + Vec2::new(0.0, 2.0), format!(r"saying yes where $p \ge {}$", num(self.threshold, 2)), text, pal::TEXT, Align2::LEFT_TOP);
        let top = min.y + text + 12.0;
        let first = (w * 0.26).min(80.0);
        let cell = ((w - first) / 2.0).min(140.0);
        let head = small + 8.0;
        let row = text + small + 16.0;
        let x = [min.x, min.x + first, min.x + first + cell];
        for (k, name) in ["said yes", "said no"].iter().enumerate() {
            f.text_px(Vec2::new(x[k + 1] + cell * 0.5, top + head * 0.5), *name, small, pal::DIM, Align2::CENTER_CENTER);
        }
        let cells = [[(t.hits, "hits", false), (t.misses, "misses", true)], [(t.false_alarms, "false alarms", true), (t.right_nos, "right no's", false)]];
        for (r, (label, color)) in [("was yes", DATA), ("was no", OTHER)].iter().enumerate() {
            let y = top + head + r as f32 * row;
            f.text_px(Vec2::new(x[0], y + row * 0.5), *label, small, *color, Align2::LEFT_CENTER);
            for (k, (count, name, wrong)) in cells[r].iter().enumerate() {
                let (a, b) = (Vec2::new(x[k + 1] + 2.0, y + 2.0), Vec2::new(x[k + 1] + cell - 2.0, y + row - 2.0));
                hud_rect(f, a, b, if *wrong { ALARM.a(0.07) } else { Color::hex(0x1A2235) });
                f.text_px(Vec2::new((a.x + b.x) * 0.5, a.y + 5.0), count.to_string(), text + 1.5, if *wrong { ALARM } else { pal::TEXT }, Align2::CENTER_TOP);
                f.text_px(Vec2::new((a.x + b.x) * 0.5, b.y - 4.0), *name, small, pal::DIM, Align2::CENTER_BOTTOM);
            }
        }
        let mut y = top + head + 2.0 * row + 8.0;
        let rate = |part: usize, whole: usize| if whole == 0 { "none to count".to_string() } else { format!("${part}/{whole} = {}$", num(part as f64 / whole as f64, 2)) };
        for line in [
            format!("hit rate {}", rate(t.hits, t.yes())),
            format!("false alarm rate {}", rate(t.false_alarms, t.no())),
            format!("right {} of the time", rate(t.right(), t.yes() + t.no())),
        ] {
            if y + small > max.y {
                break;
            }
            f.text_px(Vec2::new(x[0], y), line, small, pal::TEXT, Align2::LEFT_TOP);
            y += lay.line;
        }
        // the colours, where there is room for them (a desktop)
        let one = self.setup.dims == 1;
        let mut key: Vec<(Color, &str)> = vec![
            (DATA, "a yes (answer 1)"),
            (OTHER, "a no (answer 0)"),
            (ESTIMATE, "the model, its two handles, its ROC curve"),
            (ALARM, "the probability not given to what happened; a wrong decision"),
            (pal::WHITE, "the threshold"),
        ];
        if one {
            key.push((pal::TEXT, "the log-odds of a slice's share of yes's"));
        }
        if self.shows.truth && truth(&self.setup, &self.data).is_some() {
            key.push((TRUTH, "the truth the data were drawn from"));
        }
        if one && self.shows.least_squares {
            key.push((pal::BLUE, "the least-squares straight line"));
        }
        let y = y + 14.0;
        if !lay.narrow && y + key.len() as f32 * (small + 6.0) <= max.y {
            hud_key(f, Vec2::new(x[0], y), &key, small);
        }
    }
}
#[cfg(test)]
mod tests {
    use super::*;

    /// An experiment as its button sets it up, with every motion run to its end.
    fn run(k: usize) -> Logistic {
        let mut s = Logistic::create();
        s.set_up(k);
        s.settle();
        s
    }

    /// The outcomes at the scene's threshold, for the model as it is.
    fn table(s: &Logistic, threshold: f64) -> Table {
        let probs: Vec<f64> = s.data.x.iter().map(|x| prob(s.coef.score(*x))).collect();
        Table::of(&probs, &s.data.y, threshold)
    }

    fn scores(s: &Logistic) -> Vec<f64> {
        s.data.x.iter().map(|x| s.coef.score(*x)).collect()
    }

    /// The (yes, no) pairs in which the yes has the higher score, a tie counting half,
    /// counted one pair at a time.
    fn pairs_right(scores: &[f64], y: &[bool]) -> (f64, usize) {
        let (mut right, mut all) = (0.0, 0);
        for i in 0..y.len() {
            for j in 0..y.len() {
                if y[i] && !y[j] {
                    all += 1;
                    right += if scores[i] > scores[j] {
                        1.0
                    } else if scores[i] == scores[j] {
                        0.5
                    } else {
                        0.0
                    };
                }
            }
        }
        (right, all)
    }

    /// The things to try, held to their captions.
    #[test]
    fn the_experiments_show_what_their_captions_say() {
        // every experiment sets every setting it depends on: the same after any other
        for k in 0..TRIALS.len() {
            let fresh = run(k);
            for before in 0..TRIALS.len() {
                let mut s = run(before);
                s.penalty = 0.7;
                s.threshold = 0.9;
                s.shows = Shows { least_squares: true, odds: true, truth: true };
                s.set_up(k);
                s.settle();
                assert!(s.coef == fresh.coef && s.threshold == fresh.threshold && s.shows == fresh.shows && s.penalty == fresh.penalty && s.setup == fresh.setup, "{} after {}", TRIALS[k].button, TRIALS[before].button);
            }
        }

        // a straight line: below 0 for x < −1.72, above 1 for x > 2.06, where 6 of the 40 points lie
        let s = run(0);
        assert!(s.shows.least_squares && s.data.len() == 40);
        let (i, k) = s.line;
        assert_eq!((num(-i / k, 2), num((1.0 - i) / k, 2)), ("−1.72".to_string(), "2.06".to_string()));
        assert_eq!(s.data.x.iter().filter(|x| x[0] < -i / k || x[0] > (1.0 - i) / k).count(), 6);
        // (and the logistic curve stays inside, at every input of the chart)
        assert!((0..=100).map(|j| X_RANGE[0] + 0.1 * j as f64).all(|x| (0.0..=1.0).contains(&prob(s.coef.score([x, 0.0])))));

        // by hand, then by the method: −22.81 against a top of −13.58; fit reaches it in 7 steps
        let mut s = run(1);
        assert_eq!(num(s.value(), 2), "−22.81");
        assert!(s.best.top == Top::At && num(s.best.objective, 2) == "−13.58");
        s.start_run(CAP);
        s.settle();
        assert_eq!((s.run.taken, s.run.end), (7, Some(End::Top)));
        assert!((s.value() - s.best.objective).abs() < 1e-9);

        // odds: b = 1.99, e^b = 7.28; at −0.5, odds 0.29 (p = 0.23); at 0.5, 2.13 (p = 0.68)
        let s = run(2);
        let c = s.coef;
        assert!(s.shows.odds && s.odds_at == -0.5);
        assert_eq!((num(c.b[0], 2), num(c.b[0].exp(), 2)), ("1.99".to_string(), "7.28".to_string()));
        let (z0, z1) = (c.score([-0.5, 0.0]), c.score([0.5, 0.0]));
        assert_eq!((num(z0.exp(), 2), num(prob(z0), 2)), ("0.29".to_string(), "0.23".to_string()));
        assert_eq!((num(z1.exp(), 2), num(prob(z1), 2)), ("2.13".to_string(), "0.68".to_string()));

        // classes apart: separated; slope 6.24, 13.15, 28.64 after 5, 10, 20 steps; stopped
        // after 30 with the log-likelihood still below 0; the truth's slope is 12; with a
        // penalty of 0.1 a top at slope 3.99
        let mut s = Logistic::create();
        s.set_up(3);
        assert!(s.data.separated() && s.best.top == Top::Never && s.shows.truth);
        assert_eq!(truth(&s.setup, &s.data).map(|t| t.b[0]), Some(12.0));
        // (the first step is taken as the button is pressed: slopes[k] is after step k)
        let mut slopes = vec![f64::NAN, s.coef.b[0]];
        let mut last = log_likelihood(&s.coef, &s.data);
        while s.run.left > 0 {
            s.step();
            let ll = log_likelihood(&s.coef, &s.data);
            assert!(ll > last && ll < 0.0, "step {}: {ll} after {last}", s.run.taken);
            last = ll;
            slopes.push(s.coef.b[0]);
            assert_eq!(slopes.len(), s.run.taken + 1);
        }
        assert_eq!((s.run.taken, s.run.end), (CAP, Some(End::Cap)));
        assert_eq!([5, 10, 20].map(|i| num(slopes[i], 2)), ["6.24", "13.15", "28.64"].map(String::from));
        assert!(s.coef.finite() && slopes[30] > slopes[20] + 10.0);
        s.penalty = 0.1;
        s.refresh();
        s.start_run(CAP);
        s.settle();
        assert!(s.best.top == Top::At && s.run.end == Some(End::Top) && num(s.coef.b[0], 2) == "3.99", "{:?}", s.coef);

        // a lower threshold: from 0.5 to 0.2, misses 3 to 0 and false alarms 2 to 7; the
        // point from (0.10, 0.85) to (0.35, 1); area 0.92 = 369 of 400 pairs
        let s = run(4);
        assert!(s.threshold == 0.2 && TRIALS[4].threshold == 0.5);
        let (at5, at2) = (table(&s, 0.5), table(&s, 0.2));
        assert!((at5.misses, at5.false_alarms, at2.misses, at2.false_alarms) == (3, 2, 0, 7), "{at5:?} {at2:?}");
        assert!(at5.false_alarm_rate() == Some(0.1) && at5.hit_rate() == Some(0.85) && at2.false_alarm_rate() == Some(0.35) && at2.hit_rate() == Some(1.0));
        let sc = scores(&s);
        let area = roc(&sc, &s.data.y).map_or(0.0, |r| r.area);
        assert_eq!(pairs_right(&sc, &s.data.y), (369.0, 400));
        assert!(num(area, 2) == "0.92" && (area - 369.0 / 400.0).abs() < 1e-12);

        // a rare yes: 10 of 200; always no is right 95 %; the fit at 0.5 right 95 % finding
        // 1; at 0.1, 8 found at 19 false alarms, right 89.5 %
        let s = run(5);
        assert!(s.data.len() == 200 && s.data.yes == 10 && s.threshold == 0.5);
        let always_no = Table::of(&vec![0.0; 200], &s.data.y, 0.5);
        assert!(always_no.accuracy() == 0.95 && always_no.hits == 0);
        let at5 = table(&s, 0.5);
        assert!(at5.accuracy() == 0.95 && at5.hits == 1, "{at5:?}");
        let at1 = table(&s, 0.1);
        assert!(at1.hits == 8 && at1.false_alarms == 19 && at1.accuracy() == 0.895, "{at1:?}");

        // a ring: all but flat, probabilities 0.48 to 0.52, area 0.50
        let s = run(6);
        assert!(s.setup.dims == 2 && s.setup.kind == Kind::Ring && s.best.top == Top::At);
        let probs: Vec<f64> = s.data.x.iter().map(|x| prob(s.coef.score(*x))).collect();
        let (lo, hi) = (probs.iter().copied().fold(1.0, f64::min), probs.iter().copied().fold(0.0, f64::max));
        assert_eq!((num(lo, 2), num(hi, 2)), ("0.48".to_string(), "0.52".to_string()));
        assert_eq!(roc(&scores(&s), &s.data.y).map(|r| num(r.area, 2)), Some("0.50".to_string()));
    }

    /// The top: the gradient of the log-likelihood is zero there (the likelihood
    /// equations), and no setting near it or far from it is higher.
    #[test]
    fn the_fit_is_the_top() {
        let mut rng = Rng::new(5);
        for (k, base) in [OVERLAP, RARE, CLOUDS, RING].into_iter().enumerate() {
            for seed in [base.seed, 100 + k as u64] {
                for penalty in [0.0, 0.1, 2.0] {
                    let setup = Setup { seed, ..base };
                    let d = Data::new(&setup, &Draws::new(seed));
                    let b = best(&d, penalty);
                    assert_eq!(b.top, Top::At, "{setup:?} {penalty}");
                    let (g, _) = gradient(&b.coef, &d, penalty);
                    let size = g.iter().fold(0.0f64, |m, v| m.max(v.abs()));
                    assert!(size < 1e-8 * d.len() as f64, "{setup:?} {penalty}: gradient {g:?}");
                    // nudges in every direction, and settings anywhere, are lower
                    for _ in 0..400 {
                        let nudge = 10f64.powf(rng.range(-4.0, 1.0) as f64);
                        let mut c = b.coef;
                        c.a += nudge * rng.range(-1.0, 1.0) as f64;
                        c.b[0] += nudge * rng.range(-1.0, 1.0) as f64;
                        if d.dims == 2 {
                            c.b[1] += nudge * rng.range(-1.0, 1.0) as f64;
                        }
                        assert!(objective(&c, &d, penalty) <= b.objective, "{setup:?}: {c:?} beats {:?}", b.coef);
                    }
                    // (a hand setting is one of these: any midpoint, any slope the handles allow)
                    for _ in 0..400 {
                        let c = Coef::through(rng.range(-5.0, 5.0) as f64, rng.range(-20.0, 20.0) as f64);
                        let c = if d.dims == 2 { Coef { b: [c.b[0], rng.range(-20.0, 20.0) as f64], ..c } } else { c };
                        assert!(objective(&c, &d, penalty) <= b.objective);
                    }
                }
            }
        }
    }

    /// Newton's method gets to the top from any setting the handles allow, within the
    /// steps one press of fit takes.
    #[test]
    fn fit_gets_to_the_top_from_any_hand_setting() {
        let mut s = Logistic::create();
        for mid in [-5.0, -2.5, 0.0, 2.5, 5.0] {
            for slope in [-20.0, -4.0, -0.2, 0.2, 1.0, 4.0, 20.0] {
                s.by_hand_to(Coef::through(mid, slope));
                s.start_run(CAP);
                s.settle();
                assert_eq!(s.run.end, Some(End::Top), "from {mid}, {slope}: {} steps", s.run.taken);
                assert!((s.value() - s.best.objective).abs() < 1e-9);
            }
        }
    }

    /// A press of fit or one step while a run goes on lets it go on: no step is taken at
    /// once (it would cut short the one being drawn), and the run is as long as asked.
    #[test]
    fn a_press_during_a_run_lets_it_go_on() {
        let mut s = Logistic::create();
        s.set_up(3);
        let (taken, coef, look) = (s.run.taken, s.coef, s.look.now());
        assert!(taken == 1 && s.run.left == CAP - 1);
        s.start_run(1);
        assert!(s.run.taken == taken && s.coef == coef && s.look.now() == look && s.run.left == CAP - 1);
        s.start_run(CAP);
        assert_eq!(s.run.left, CAP);
        // and once a run has ended, a press starts a new count
        s.settle();
        assert_eq!(s.run.end, Some(End::Cap));
        s.start_run(1);
        assert_eq!((s.run.taken, s.run.left), (1, 0));
    }

    /// One more unit of the input multiplies the odds by e to the slope, wherever it is.
    #[test]
    fn odds_multiply_by_e_to_the_slope() {
        for c in [Coef::through(0.3, 1.7), Coef::through(-2.0, -0.4), Coef { a: 1.0, b: [0.0, 0.0] }] {
            for x in [-4.0, -1.0, 0.0, 0.5, 3.0] {
                let odds = |x: f64| {
                    let p = prob(c.score([x, 0.0]));
                    p / (1.0 - p)
                };
                let ratio = odds(x + 1.0) / odds(x);
                assert!((ratio / c.b[0].exp() - 1.0).abs() < 1e-12, "{c:?} at {x}: {ratio}");
                // and the log-odds are the straight line
                assert!((logit(prob(c.score([x, 0.0]))) - c.score([x, 0.0])).abs() < 1e-9);
            }
        }
    }

    /// With the classes apart there is no top: the log-likelihood rises with the slope
    /// without end, and only a penalty gives an answer. Newton's method does not overflow.
    #[test]
    fn classes_apart_have_no_top() {
        let d = Data::new(&APART, &Draws::new(APART.seed));
        assert!(d.separated());
        // the midpoint between the classes, and steeper and steeper
        let gap = |yes: bool| d.x.iter().zip(&d.y).filter(move |(_, y)| **y == yes).map(|(x, _)| x[0]);
        let mid = (gap(false).fold(f64::MIN, f64::max) + gap(true).fold(f64::MAX, f64::min)) * 0.5;
        let mut last = f64::NEG_INFINITY;
        for k in 0..12 {
            let ll = log_likelihood(&Coef::through(mid, 2f64.powi(k)), &d);
            assert!(ll > last && ll <= 0.0, "slope {}: {ll}", 2f64.powi(k));
            last = ll;
        }
        assert_eq!(best(&d, 0.0).top, Top::Never);
        for penalty in [0.001, 0.1, 1.0] {
            assert_eq!(best(&d, penalty).top, Top::At);
        }
        // a long run stays finite
        let mut c = Coef::through(0.0, 1.0);
        for _ in 0..SEARCH {
            match newton(&c, &d, 0.0) {
                Some(n) => c = n,
                None => break,
            }
            assert!(c.finite() && log_likelihood(&c, &d).is_finite());
        }
        // two inputs apart: the same
        let setup = Setup { apart: 5.0, spread: 0.4, ..CLOUDS };
        let d2 = Data::new(&setup, &Draws::new(setup.seed));
        assert_eq!(best(&d2, 0.0).top, Top::Never);
        assert_eq!(best(&d2, 0.1).top, Top::At);
    }

    /// The area under the ROC curve is the share of (yes, no) pairs in which the yes has
    /// the higher score: measured both ways, with ties, flat models and falling ones.
    #[test]
    fn the_roc_area_is_the_share_of_pairs_ranked_right() {
        for k in 0..TRIALS.len() {
            let s = run(k);
            for c in [s.coef, Coef { a: 0.3, b: [0.0, 0.0] }, Coef { a: s.coef.a, b: [-s.coef.b[0], -s.coef.b[1]] }] {
                let sc: Vec<f64> = s.data.x.iter().map(|x| c.score(*x)).collect();
                // (rounded scores: many ties)
                let tied: Vec<f64> = sc.iter().map(|v| (v * 2.0).round()).collect();
                for v in [sc, tied] {
                    let (right, all) = pairs_right(&v, &s.data.y);
                    let r = roc(&v, &s.data.y).expect("yes's and no's");
                    assert!((r.area - right / all as f64).abs() < 1e-12, "{}: {} against {}", TRIALS[k].button, r.area, right / all as f64);
                    assert!((ranked_right(&v, &s.data.y).unwrap_or(-1.0) - r.area).abs() < 1e-12);
                    assert!(r.points.first() == Some(&[0.0, 0.0]) && r.points.last() == Some(&[1.0, 1.0]));
                }
            }
        }
        // the flat model ranks every pair as a tie: the area is a half
        let s = run(0);
        let flat = vec![0.0; s.data.len()];
        assert_eq!(roc(&flat, &s.data.y).map(|r| r.area), Some(0.5));
    }

    /// Degenerate data are normal data: one point, one class, inputs all the same, the
    /// threshold at its ends; everything stays finite and nothing panics.
    #[test]
    fn degenerate_data_stay_finite() {
        let cases = [
            Setup { n: 1, ..OVERLAP },
            Setup { share: 0.0, ..OVERLAP },
            Setup { share: 1.0, ..OVERLAP },
            Setup { apart: 0.0, spread: 0.0, ..OVERLAP },
            Setup { apart: 3.0, spread: 0.0, ..OVERLAP },
            Setup { apart: 0.0, spread: 0.0, ..CLOUDS },
            Setup { apart: 3.0, spread: 0.0, ..CLOUDS },
            Setup { n: 2, ..RING },
            Setup { n: MOST, share: 0.003, ..RARE },
        ];
        for setup in cases {
            for penalty in [0.0, 0.5] {
                let mut s = Logistic::create();
                s.setup = setup;
                s.penalty = penalty;
                s.refresh();
                assert!(s.best.coef.finite() && s.best.objective.is_finite(), "{setup:?}");
                s.start_run(CAP);
                s.settle();
                assert!(s.coef.finite() && s.value().is_finite(), "{setup:?}: {:?}", s.coef);
                for t in [0.0, 1.0] {
                    s.threshold = t;
                    let shown = s.shown();
                    let tb = shown.table;
                    assert_eq!(tb.yes() + tb.no(), s.data.len());
                    assert!(shown.probs.iter().all(|p| p.is_finite()) && shown.value.is_finite());
                    assert!(!s.status(&shown).is_empty() && s.status(&shown).iter().all(|t| !t.is_empty()));
                }
                let one = s.data.one_class();
                assert_eq!(s.best.top == Top::OneClass, one, "{setup:?}");
                if one {
                    assert!(s.shown().roc.is_none() && s.status(&s.shown()).iter().all(|t| t.contains("no curve")));
                }
            }
        }
        // all inputs the same, with both answers: the fit settles on the share there
        let d = Data::new(&Setup { apart: 0.0, spread: 0.0, ..OVERLAP }, &Draws::new(1));
        assert!(d.same_inputs() && !d.one_class());
        let b = best(&d, 0.0);
        assert!(b.top == Top::At && (prob(b.coef.score(d.x[0])) - 0.5).abs() < 1e-9, "{:?}", b.coef);
        // two classes each at one place: apart, and no top
        let d = Data::new(&Setup { apart: 3.0, spread: 0.0, ..OVERLAP }, &Draws::new(1));
        assert!(d.separated() && best(&d, 0.0).top == Top::Never);
    }

    /// Every layout keeps its charts inside the canvas, at every size down to the
    /// smallest, with no number that is not finite.
    #[test]
    fn the_layouts_fit_their_canvases() {
        for (w, h) in [(1600.0, 1000.0), (1068.0, 1000.0), (578.0, 700.0), (390.0, 800.0), (390.0, 300.0), (240.0, 180.0), (120.0, 90.0), (2560.0, 1400.0)] {
            for dims in [1, 2] {
                for inset in [0.0, 46.0] {
                    let lay = Layout::of(Vec2::new(w, h), inset, dims);
                    for r in [lay.main, lay.second, lay.roc, lay.table] {
                        assert!(r.0.is_finite() && r.1.is_finite());
                        if r != NO_RECT {
                            assert!(r.0.x >= 0.0 && r.1.x <= w + 0.5 && r.0.y >= 0.0 && r.1.y <= h + 0.5, "{w}x{h} {dims}: {r:?}");
                        }
                    }
                    if dims == 2 && lay.main != NO_RECT {
                        let (a, b) = lay.main;
                        assert!(((b.x - a.x) - (b.y - a.y)).abs() < 0.01, "the plane is square");
                    }
                }
            }
        }
    }

    /// The drawing helpers: a dashed line keeps its length's share drawn; the line of a
    /// level crosses the chart where it should; the strips between levels tile it.
    #[test]
    fn the_drawing_helpers_do_what_they_say() {
        let pts = [Vec2::new(0.0, 0.0), Vec2::new(120.0, 0.0)];
        let d = dashes(&pts, 7.0, 5.0);
        let drawn: f32 = d.split(|p| !p.is_finite()).filter(|s| s.len() >= 2).map(|s| (s[s.len() - 1] - s[0]).length()).sum();
        assert!((drawn - 70.0).abs() < 1e-3, "{drawn}");
        let plot = Plot::new((Vec2::new(0.0, 0.0), Vec2::new(100.0, 100.0)), PLANE, PLANE);
        let c = Coef { a: 0.5, b: [1.0, 1.0] };
        let [p, q] = across(&plot, &c, 0.0).expect("a line across");
        assert!(c.score(p).abs() < 1e-12 && c.score(q).abs() < 1e-12);
        assert!(across(&plot, &Coef { a: 1.0, b: [0.0, 0.0] }, 0.0).is_none());
        let chart = Plot::new(plot.rect, X_RANGE, LOGODDS);
        let [(xa, ya), (xb, yb)] = line_in(&chart, 0.5, 3.0).expect("a line inside");
        assert!((ya - (0.5 + 3.0 * xa)).abs() < 1e-12 && (yb - (0.5 + 3.0 * xb)).abs() < 1e-12 && ya == LOGODDS[0] && yb == LOGODDS[1]);
        assert!(line_in(&chart, 9.0, 0.0).is_none() && line_in(&chart, 2.0, 0.0) == Some([(X_RANGE[0], 2.0), (X_RANGE[1], 2.0)]));
        // the strips' areas add up to the chart's
        let area = |poly: &[[f64; 2]]| (0..poly.len()).map(|i| poly[i][0] * poly[(i + 1) % poly.len()][1] - poly[(i + 1) % poly.len()][0] * poly[i][1]).sum::<f64>().abs() * 0.5;
        let edges = [f64::NEG_INFINITY, -2.0, 0.0, 1.0, f64::INFINITY];
        let total: f64 = edges.windows(2).map(|w| area(&strip(&plot, &c, w[0], w[1]))).sum();
        assert!((total - 81.0).abs() < 1e-9, "{total}");
    }

    /// The numbers behind the captions (run with --ignored --nocapture).
        #[test]
    #[ignore]
    fn measure() {
        for k in 0..TRIALS.len() {
            let s = run(k);
            let d = &s.data;
            let scores: Vec<f64> = d.x.iter().map(|x| s.coef.score(*x)).collect();
            let probs: Vec<f64> = scores.iter().map(|z| prob(*z)).collect();
            let t = Table::of(&probs, &d.y, s.threshold);
            let r = roc(&scores, &d.y);
            println!("{}: n {} yes {} coef {:?} ll {:.4} top {:?} best {:.4} steps {} end {:?} | t {} {:?} acc {:.4} | auc {:?} ranked {:?} | separated {}", TRIALS[k].button, d.len(), d.yes, s.coef, log_likelihood(&s.coef, d), s.best.top, s.best.objective, s.run.taken, s.run.end, s.threshold, t, t.accuracy(), r.map(|r| r.area), ranked_right(&scores, &d.y), d.separated());
        }
        // a straight line
        let s = run(0);
        let (i, k) = s.line;
        let (lo, hi) = (-i / k, (1.0 - i) / k);
        let out = s.data.x.iter().filter(|x| x[0] < lo || x[0] > hi).count();
        println!("least squares: intercept {i:.4} slope {k:.4}: below 0 for x < {lo:.4}, above 1 for x > {hi:.4}; {out} points there");
        // by hand
        let mut s = run(1);
        println!("hand start ll {:.4}", log_likelihood(&s.coef, &s.data));
        s.start_run(CAP);
        s.settle();
        println!("fit from the hand start: {} steps, end {:?}, ll {:.6}", s.run.taken, s.run.end, log_likelihood(&s.coef, &s.data));
        // odds
        let s = run(2);
        let c = s.coef;
        println!("odds: b {:.4} e^b {:.4}; x=0: p {:.4} odds {:.4}; x=1: p {:.4} odds {:.4}", c.b[0], c.b[0].exp(), prob(c.a), c.a.exp(), prob(c.a + c.b[0]), (c.a + c.b[0]).exp());
        // apart
        let mut s = Logistic::create();
        s.set_up(3);
        let mut line = String::new();
        while s.run.left > 0 {
            s.step();
            if [1, 2, 3, 5, 10, 15, 20, 25, 30].contains(&s.run.taken) {
                line += &format!(" {}: b {:.3} ll {:.3e} |", s.run.taken, s.coef.b[0], log_likelihood(&s.coef, &s.data));
            }
        }
        println!("apart:{line} end {:?} truth {:?}", s.run.end, truth(&s.setup, &s.data));
        for pen in [0.01, 0.1, 1.0] {
            s.penalty = pen;
            s.refresh();
            println!("  penalty {pen}: {:?} b {:.4} a {:.4}", s.best.top, s.best.coef.b[0], s.best.coef.a);
        }
        // a lower threshold: at 0.5 and at 0.2
        let s = run(4);
        let probs: Vec<f64> = s.data.x.iter().map(|x| prob(s.coef.score(*x))).collect();
        for t in [0.5, 0.2] {
            let tb = Table::of(&probs, &s.data.y, t);
            println!("threshold {t}: {tb:?} hit {:?} fa {:?}", tb.hit_rate(), tb.false_alarm_rate());
        }
        let mut right = 0.0;
        for i in 0..probs.len() {
            for j in 0..probs.len() {
                if s.data.y[i] && !s.data.y[j] {
                    right += if probs[i] > probs[j] { 1.0 } else if probs[i] == probs[j] { 0.5 } else { 0.0 };
                }
            }
        }
        println!("pairs ranked right: {right} of {}", s.data.yes * (s.data.len() - s.data.yes));
        // a rare yes at other thresholds
        let s = run(5);
        let probs: Vec<f64> = s.data.x.iter().map(|x| prob(s.coef.score(*x))).collect();
        for t in [0.5, 0.3, 0.2, 0.1, 0.05] {
            let tb = Table::of(&probs, &s.data.y, t);
            println!("rare, threshold {t}: {tb:?} acc {:.4}", tb.accuracy());
        }
        // a ring
        let s = run(6);
        let probs: Vec<f64> = s.data.x.iter().map(|x| prob(s.coef.score(*x))).collect();
        println!("ring: p from {:.4} to {:.4}", probs.iter().copied().fold(1.0, f64::min), probs.iter().copied().fold(0.0, f64::max));
    }
}

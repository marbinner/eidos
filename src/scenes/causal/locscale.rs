//! Noise that changes size: location-scale noise models, with and without a hidden
//! common cause. Fit the level and the spread of an effect from its cause, divide
//! what is left over by the fitted spread, and ask whether that is independent of the
//! input; and compare ways of fitting the two together.

use super::*;
use crate::clock::Pace;
use crate::math::causal::{mean, variance, Noise};
use std::sync::Arc;

pub const DEF: SceneDef = SceneDef {
    id: "locscale",
    title: "Noise that changes size",
    group: GROUP,
    blurb: "When a cause sets the spread of its effect as well as its level: fitting both, checking what is left over, and what a hidden common cause does to the check.",
    make: |_| Box::new(LocScale::create()),
};

// ----- the truth ------------------------------------------------------------------------------

/// Which graph makes the data.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Graph {
    /// X → Y, and nothing else.
    Direct,
    /// X ← U → Y: a third variable causes both, and neither causes the other.
    Common,
    /// U → X, U → Y and X → Y.
    Confounded,
}

impl Graph {
    /// Is there a third variable?
    fn third(self) -> bool {
        self != Graph::Direct
    }

    /// Does X cause Y?
    fn arrow(self) -> bool {
        self != Graph::Common
    }
}

/// How closely X follows the third variable, when there is one: X = 0.8 U + 0.6 E, with
/// U and X's own noise E each of spread 1 (so that X has spread 1 too).
const CLOSE: f64 = 0.8;

/// The level of the effect: f in Y = f(X) + g(X) N.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Curve {
    Line,
    Bend,
    Step,
    Parabola,
}

impl Curve {
    fn at(self, c: f64) -> f64 {
        match self {
            Curve::Line => c,
            // (a smooth hinge: level to the left, rising to the right)
            Curve::Bend => 0.9 * (1.6 * c).exp().ln_1p() - 0.9,
            Curve::Step => 1.6 * (1.4 * c).tanh(),
            Curve::Parabola => c * c - 1.0,
        }
    }
}

/// How the spread of the effect varies with its cause: g in Y = f(X) + σ g(X) N, each
/// shape scaled so that its mean square over the cause (a bell of spread 1) is 1, so
/// that σ is the noise's overall size whatever the shape.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Spread {
    Even,
    Rising,
    Pinched,
    Bulge,
}

impl Spread {
    fn at(self, c: f64) -> f64 {
        match self {
            Spread::Even => 1.0,
            // (e^{0.6c} has mean square e^{0.72}: dividing by e^{0.36} makes it 1)
            Spread::Rising => (0.6 * c - 0.36).exp(),
            // (0.04 + 0.96 c² has mean 1)
            Spread::Pinched => (0.04 + 0.96 * c * c).sqrt(),
            // ((0.3 + e^{-c²})² has mean 0.09 + 0.6/√3 + 1/√5)
            Spread::Bulge => (0.3 + (-c * c).exp()) / BULGE_RMS,
        }
    }
}

/// The root mean square of 0.3 + e^{-c²} over a bell of spread 1.
const BULGE_RMS: f64 = 0.940_012_637;

/// Everything the data depend on.
#[derive(Clone, Copy, Debug, PartialEq)]
struct Setup {
    graph: Graph,
    /// Is the third variable measured (and so in the fits, as Z)?
    seen: bool,
    curve: Curve,
    spread: Spread,
    noise: Noise,
    /// The noise's overall size, σ.
    sigma: f64,
    /// How strongly the third variable acts on Y's level (λ, through the same curve f)
    /// and on its spread (κ): Y = f(X) + λ f(U) + σ g(X) e^{κU − κ²} N, without f(X)
    /// and g(X) when X does not cause Y. (The e^{−κ²} keeps the mean square of e^{κU}
    /// at 1.)
    level: f64,
    scale: f64,
    /// How closely X follows the third variable: X = cU + √(1 − c²) E.
    close: f64,
    n: usize,
}

impl Setup {
    /// Y's true level and spread given all of its causes: X at `x`, and the third
    /// variable at `u` (if there is one).
    fn truth(&self, x: f64, u: f64) -> (f64, f64) {
        let third = self.graph.third();
        let (lambda, kappa) = if third { (self.level, self.scale) } else { (0.0, 0.0) };
        let (f, g) = if self.graph.arrow() { (self.curve.at(x), self.spread.at(x)) } else { (0.0, 1.0) };
        let through = if third { self.curve.at(u) } else { 0.0 };
        (f + lambda * through, self.sigma * g * (kappa * u - kappa * kappa).exp())
    }
}

// ----- drawing samples ----------------------------------------------------------------------------

/// Φ, the share of a bell (normal) below v.
fn bell_below(v: f64) -> f64 {
    // (the complementary error function to a few parts in ten million: Numerical Recipes' erfcc)
    let z = v.abs() / std::f64::consts::SQRT_2;
    let t = 1.0 / (1.0 + 0.5 * z);
    let tail = t
        * (-z * z - 1.265_512_23
            + t * (1.000_023_68
                + t * (0.374_091_96 + t * (0.096_784_18 + t * (-0.186_288_06 + t * (0.278_868_07 + t * (-1.135_203_98 + t * (1.488_515_87 + t * (-0.822_152_23 + t * 0.170_872_77)))))))))
            .exp();
    if v >= 0.0 {
        1.0 - 0.5 * tail
    } else {
        0.5 * tail
    }
}

/// The value below which a share `u` of a shape of noise lies. The same uniform number
/// gives the point of the same rank in every shape, so a change of shape moves each
/// point a little instead of dealing them all out afresh.
fn quantile(noise: Noise, u: f64) -> f64 {
    let u = u.max(1e-12).min(1.0 - 1e-12);
    let invert = |below: &dyn Fn(f64) -> f64| {
        let (mut lo, mut hi) = (-12.0f64, 12.0f64);
        for _ in 0..52 {
            let mid = 0.5 * (lo + hi);
            if below(mid) < u {
                lo = mid;
            } else {
                hi = mid;
            }
        }
        0.5 * (lo + hi)
    };
    match noise {
        Noise::Normal => invert(&bell_below),
        Noise::Uniform => (u - 0.5) * 12f64.sqrt(),
        Noise::Laplace => {
            let b = 0.5f64.sqrt();
            if u < 0.5 {
                b * (2.0 * u).ln()
            } else {
                -b * (2.0 * (1.0 - u)).ln()
            }
        }
        Noise::TwoHumps => {
            let s = (1.0f64 - 0.81).sqrt();
            invert(&|v| 0.5 * (bell_below((v - 0.9) / s) + bell_below((v + 0.9) / s)))
        }
    }
}

/// A uniform number in (0, 1), to double precision.
fn uniform(rng: &mut Rng) -> f64 {
    ((rng.next_u64() >> 11) as f64 + 0.5) / (1u64 << 53) as f64
}

/// The random part of a sample: for each point, the root cause (X itself, or the third
/// variable U), X's own noise E when X follows U, and the uniform number that makes
/// Y's noise N (in whatever shape is asked for).
#[derive(Clone)]
struct Draws {
    key: (u64, usize),
    root: Vec<f64>,
    own: Vec<f64>,
    noise: Vec<f64>,
}

impl Draws {
    fn new(seed: u64, n: usize) -> Draws {
        let mut rng = Rng::new(seed);
        let (mut root, mut own, mut noise) = (Vec::with_capacity(n), Vec::with_capacity(n), Vec::with_capacity(n));
        for _ in 0..n {
            root.push(quantile(Noise::Normal, uniform(&mut rng)));
            own.push(quantile(Noise::Normal, uniform(&mut rng)));
            noise.push(uniform(&mut rng));
        }
        Draws { key: (seed, n), root, own, noise }
    }

    /// The sample these draws make with `setup`: X, Y and the third variable (all 0 if
    /// there is none).
    fn sample(&self, s: &Setup) -> Sample {
        let n = self.root.len();
        let (mut x, mut y, mut u) = (Vec::with_capacity(n), Vec::with_capacity(n), Vec::with_capacity(n));
        let close = s.close.max(0.0).min(1.0);
        let other = (1.0 - close * close).sqrt();
        for i in 0..n {
            let (xi, ui) = if s.graph.third() { (close * self.root[i] + other * self.own[i], self.root[i]) } else { (self.root[i], 0.0) };
            let (level, spread) = s.truth(xi, ui);
            x.push(xi);
            y.push(level + spread * quantile(s.noise, self.noise[i]));
            u.push(ui);
        }
        Sample { x, y, u }
    }
}

/// One sample: X, Y, and the third variable U (0 when there is none).
#[derive(Clone, Debug, PartialEq)]
struct Sample {
    x: Vec<f64>,
    y: Vec<f64>,
    u: Vec<f64>,
}

// ----- smoothing ------------------------------------------------------------------------------------

/// The inputs a fit is made from: one column (the other variable) or two (it and the
/// measured third variable), each with its spread, by which distances along it are
/// measured.
#[derive(Clone, Debug)]
struct Inputs {
    cols: Vec<Vec<f64>>,
    scale: Vec<f64>,
}

impl Inputs {
    fn new(cols: Vec<Vec<f64>>) -> Inputs {
        let scale = cols
            .iter()
            .map(|c| {
                let sd = variance(c).sqrt();
                if sd > 0.0 && sd.is_finite() {
                    sd
                } else {
                    1.0
                }
            })
            .collect();
        Inputs { cols, scale }
    }

    fn len(&self) -> usize {
        self.cols.iter().map(Vec::len).min().unwrap_or(0)
    }

    fn dims(&self) -> usize {
        self.cols.len()
    }

    /// Point `i`, measured in spreads.
    fn point(&self, i: usize) -> [f64; 2] {
        let mut p = [0.0; 2];
        for (k, c) in self.cols.iter().enumerate().take(2) {
            p[k] = c[i] / self.scale[k];
        }
        p
    }

    /// The same columns at the points `rows` (a half of the sample).
    fn subset(&self, rows: &[usize]) -> Inputs {
        Inputs { cols: self.cols.iter().map(|c| rows.iter().map(|&i| c[i]).collect()).collect(), scale: self.scale.clone() }
    }
}

/// The kernel weights a fit gives each data point at each place it is evaluated: a
/// bell round the place, as wide as the usual width for this many points
/// (`0.6 n^{-1/5}` spreads for one input, `0.6 n^{-1/6}` for two) where the points are
/// dense, and wide enough to reach a twelfth of them where they are sparse (at the
/// ends of a bell-shaped cloud, a few points alone would swing the curve about).
struct Table {
    /// Places (rows) by data points (columns).
    w: Vec<f64>,
    data: usize,
    /// Each place, in spreads.
    at: Vec<[f64; 2]>,
}

impl Table {
    fn new(data: &Inputs, places: &[[f64; 2]]) -> Table {
        let n = data.len();
        let d = data.dims().clamp(1, 2);
        let least = 0.6 * (n.max(2) as f64).powf(if d == 1 { -0.2 } else { -1.0 / 6.0 });
        let reach = (n / 12).max(6).min(n.max(1)) - 1;
        let pts: Vec<[f64; 2]> = (0..n).map(|i| data.point(i)).collect();
        let mut w = vec![0.0; places.len() * n];
        let mut dist = vec![0.0; n];
        for (r, at) in places.iter().enumerate() {
            for (j, p) in pts.iter().enumerate() {
                dist[j] = (p[0] - at[0]).powi(2) + (p[1] - at[1]).powi(2);
            }
            let far = if n > 0 {
                let mut sorted = dist.clone();
                sorted.select_nth_unstable_by(reach.min(n - 1), f64::total_cmp).1.sqrt()
            } else {
                0.0
            };
            // (a bell this wide gives the furthest of them about a third of the weight)
            let h = least.max(0.67 * far);
            let inv = 0.5 / (h * h);
            for j in 0..n {
                w[r * n + j] = (-dist[j] * inv).exp();
            }
        }
        Table { w, data: n, at: places.to_vec() }
    }

    fn row(&self, r: usize) -> &[f64] {
        &self.w[r * self.data..(r + 1) * self.data]
    }
}

/// The weighted straight-line (or plane) fit of `y` near one place: the kernel weights
/// `k` (times `extra`, when given), and the fit's value there. A local average where
/// the points near do not settle a slope.
fn local_line(k: &[f64], extra: Option<&[f64]>, pts: &[[f64; 2]], at: [f64; 2], dims: usize, y: &[f64]) -> f64 {
    let m = dims.clamp(1, 2) + 1;
    let mut a = [0.0f64; 9];
    let mut b = [0.0f64; 3];
    for j in 0..y.len().min(k.len()).min(pts.len()) {
        let w = k[j] * extra.map_or(1.0, |e| e[j]);
        if w <= 0.0 {
            continue;
        }
        let t = [1.0, pts[j][0] - at[0], pts[j][1] - at[1]];
        for r in 0..m {
            b[r] += w * t[r] * y[j];
            for c in 0..m {
                a[r * m + c] += w * t[r] * t[c];
            }
        }
    }
    solve_small(&a, &b, m).map_or_else(|| if a[0] > 1e-300 { b[0] / a[0] } else { mean(y) }, |beta| beta[0])
}

/// A weighted local fit of the `tau`-quantile of `y` near one place (a straight line or
/// a plane, by the least sum of tilted absolute differences), by iteratively
/// reweighted least squares from the local straight-line fit.
#[allow(clippy::too_many_arguments)] // (each is a part of the same small fit)
fn local_quantile(k: &[f64], skip: Option<usize>, pts: &[[f64; 2]], at: [f64; 2], dims: usize, y: &[f64], tau: f64, tiny: f64) -> f64 {
    let m = dims.clamp(1, 2) + 1;
    let n = y.len().min(k.len()).min(pts.len());
    // start from the least-squares line
    let mut beta = [local_line(k, None, pts, at, dims, y), 0.0, 0.0];
    let mut a = [0.0f64; 9];
    let mut b = [0.0f64; 3];
    for _ in 0..QUANTILE_ROUNDS {
        a.fill(0.0);
        b.fill(0.0);
        for j in 0..n {
            if k[j] < 1e-12 || skip == Some(j) {
                continue;
            }
            let t = [1.0, pts[j][0] - at[0], pts[j][1] - at[1]];
            let r = y[j] - (beta[0] + beta[1] * t[1] + beta[2] * t[2]);
            let w = k[j] / (2.0 * r.abs().max(tiny));
            let tilt = (tau - 0.5) * k[j];
            for row in 0..m {
                b[row] += w * t[row] * y[j] + tilt * t[row];
                for c in 0..m {
                    a[row * m + c] += w * t[row] * t[c];
                }
            }
        }
        match solve_small(&a, &b, m) {
            Some(next) => {
                beta[..m].copy_from_slice(&next[..m]);
            }
            None => break,
        }
    }
    beta[0]
}

/// The weighted 25th and 75th percentiles of `v` (with `order` the indices that sort
/// it, and weights `k`, leaving out `skip`): where the running share of the weight
/// passes a quarter and three quarters, between neighbouring values by straight lines.
fn weighted_quartiles(k: &[f64], skip: Option<usize>, order: &[usize], v: &[f64]) -> [f64; 2] {
    let total: f64 = order.iter().filter(|&&j| Some(j) != skip).map(|&j| k.get(j).copied().unwrap_or(0.0)).sum();
    if total.is_nan() || total <= 0.0 {
        return [0.0, 0.0];
    }
    let mut out = [f64::NAN; 2];
    let mut below = 0.0;
    let mut last: Option<(f64, f64)> = None;
    for &j in order {
        if Some(j) == skip {
            continue;
        }
        let w = k.get(j).copied().unwrap_or(0.0);
        if w <= 0.0 {
            continue;
        }
        // (each point stands at the middle of its own share)
        let at = (below + 0.5 * w) / total;
        for (q, target) in [0.25, 0.75].into_iter().enumerate() {
            if out[q].is_nan() && at >= target {
                out[q] = match last {
                    Some((a, va)) if at > a => va + (v[j] - va) * (target - a) / (at - a),
                    _ => v[j],
                };
            }
        }
        last = Some((at, v[j]));
        below += w;
    }
    let end = last.map_or(0.0, |l| l.1);
    out.map(|q| if q.is_nan() { end } else { q })
}

/// Rounds of reweighting in a quantile fit.
const QUANTILE_ROUNDS: usize = 12;

/// Solve a 1×1, 2×2 or 3×3 system (row-major), or None if it does not settle the
/// answer (a pivot that is nothing beside the entries).
fn solve_small(a: &[f64; 9], b: &[f64; 3], m: usize) -> Option<[f64; 3]> {
    let mut a = *a;
    let mut b = *b;
    let size = a[..m * m].iter().fold(0.0f64, |s, v| s.max(v.abs()));
    if size.is_nan() || size <= 0.0 || !size.is_finite() {
        return None;
    }
    for col in 0..m {
        let piv = (col..m).max_by(|&p, &q| a[p * m + col].abs().total_cmp(&a[q * m + col].abs())).unwrap_or(col);
        if a[piv * m + col].abs() <= 1e-10 * size {
            return None;
        }
        if piv != col {
            for c in 0..m {
                a.swap(col * m + c, piv * m + c);
            }
            b.swap(col, piv);
        }
        for r in col + 1..m {
            let f = a[r * m + col] / a[col * m + col];
            for c in col..m {
                a[r * m + c] -= f * a[col * m + c];
            }
            b[r] -= f * b[col];
        }
    }
    let mut x = [0.0; 3];
    for r in (0..m).rev() {
        let mut acc = b[r];
        for c in r + 1..m {
            acc -= a[r * m + c] * x[c];
        }
        x[r] = acc / a[r * m + r];
    }
    x.iter().all(|v| v.is_finite()).then_some(x)
}

// ----- the ways of fitting ---------------------------------------------------------------------

/// How the level and the spread are fitted.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
enum Method {
    /// The level alone (a smooth curve), and what is left over taken as it is.
    Level,
    /// A smooth curve for the level, then a smooth curve through the squares of what is
    /// left over for the spread.
    TwoSteps,
    /// The same, then the level fitted again counting each point less where the spread
    /// is large, the spread again, and so on.
    Reweighted,
    /// Both at once, by making the data most probable under normal noise.
    Likelihood,
    /// The median for the level, and the distance between the quartiles for the spread.
    Quantiles,
}

const METHODS: [Method; 5] = [Method::Level, Method::TwoSteps, Method::Reweighted, Method::Likelihood, Method::Quantiles];

/// Rounds of refitting with weights (after the two steps).
const ROUNDS: usize = 3;

/// The smallest spread a fit may give, as a share of the overall spread of what is
/// left over: dividing by a spread near 0 would blow a point up to any size.
const FLOOR: f64 = 0.05;

/// A fitted level and spread at some places.
#[derive(Clone, Debug, Default, PartialEq)]
struct Fitted {
    level: Vec<f64>,
    spread: Vec<f64>,
}

/// The spread of what is left over (its root mean square), or 1 if there is none.
fn rms(v: &[f64]) -> f64 {
    let s = (v.iter().map(|r| r * r).sum::<f64>() / v.len().max(1) as f64).sqrt();
    if s > 0.0 && s.is_finite() {
        s
    } else {
        1.0
    }
}

/// One stage of a fit, done over a set of places (set 0: the data points themselves).
#[derive(Clone, Copy, Debug, PartialEq)]
enum Pass {
    /// The kernel weights at a set of places.
    Table(usize),
    /// The level at each place: a local straight line (weighted by the last spreads,
    /// if asked), or a local median.
    Level { set: usize, weighted: bool, median: bool },
    /// What is left over at the data points, from their levels.
    Leftover,
    /// The spread at each place: from the local mean of the squares of what is left
    /// over, from its local quartiles, or the same everywhere (the level alone).
    Spread { set: usize, how: Spreading },
    /// The weights for the next level: one over the square of each point's spread.
    Weights,
    /// The whole fit by likelihood (quick enough to make at once).
    Likelihood,
}

#[derive(Clone, Copy, Debug, PartialEq)]
enum Spreading {
    Squares,
    Quartiles,
    Even,
}

/// What a fit does, pass by pass, with `sets` sets of places.
fn plan(method: Method, sets: usize) -> Vec<Pass> {
    use Pass::*;
    let level = |set, weighted| Level { set, weighted, median: false };
    let mut p = Vec::new();
    if method == Method::Likelihood {
        p.push(Likelihood);
        return p;
    }
    let (median, how) = match method {
        Method::Quantiles => (true, Spreading::Quartiles),
        Method::Level => (false, Spreading::Even),
        _ => (false, Spreading::Squares),
    };
    p.extend([Table(0), Level { set: 0, weighted: false, median }, Leftover, Spread { set: 0, how }]);
    let weighted = method == Method::Reweighted;
    if weighted {
        for _ in 0..ROUNDS {
            p.extend([Weights, level(0, true), Leftover, Spread { set: 0, how }]);
        }
    }
    for set in 1..sets {
        p.extend([Table(set), Level { set, weighted, median }, Spread { set, how }]);
    }
    p
}

/// How much work a place takes in a pass, in units of one kernel weight.
fn place_cost(pass: Pass, n: usize) -> usize {
    n.max(1)
        * match pass {
            Pass::Table(_) => 4,
            Pass::Level { median: true, .. } => 2 * QUANTILE_ROUNDS,
            Pass::Level { .. } => 2,
            Pass::Spread { how: Spreading::Even, .. } => 0,
            _ => 1,
        }
}

/// A fit made a few places at a time: `step` does about so much work and says when
/// it is done. The kernel weights at the data points can be handed in, made once for
/// several fits.
struct Job {
    data: Inputs,
    pts: Vec<[f64; 2]>,
    y: Vec<f64>,
    sets: Vec<Vec<[f64; 2]>>,
    tables: Vec<Option<Arc<Table>>>,
    plan: Vec<Pass>,
    stage: usize,
    cursor: usize,
    /// The level and spread at each set of places, as far as they are done.
    level: Vec<Vec<f64>>,
    spread: Vec<Vec<f64>>,
    /// What is left over at the data points, the smallest spread allowed, the weights
    /// of the last level, and the order of what is left over (for its quartiles).
    left: Vec<f64>,
    squares: Vec<f64>,
    even: f64,
    floor: f64,
    weights: Option<Vec<f64>>,
    order: Vec<usize>,
    tiny: f64,
}

impl Job {
    /// A fit of `y` from `data`, giving the level and spread at the data points and at
    /// each of `places`. `tables` may hold kernel weights already made, by set.
    fn new(method: Method, data: Inputs, y: &[f64], places: Vec<Vec<[f64; 2]>>, tables: Vec<Option<Arc<Table>>>) -> Job {
        let n = data.len().min(y.len());
        let pts: Vec<[f64; 2]> = (0..n).map(|i| data.point(i)).collect();
        let mut sets = vec![pts.clone()];
        sets.extend(places);
        let mut tables = tables;
        tables.resize(sets.len(), None);
        let y = y[..n].to_vec();
        let centred: Vec<f64> = y.iter().map(|v| v - mean(&y)).collect();
        let tiny = 1e-6 * rms(&centred);
        let plan = plan(method, sets.len());
        Job {
            level: sets.iter().map(|s| vec![0.0; s.len()]).collect(),
            spread: sets.iter().map(|s| vec![1.0; s.len()]).collect(),
            data,
            pts,
            y,
            sets,
            tables,
            plan,
            stage: 0,
            cursor: 0,
            left: vec![0.0; n],
            squares: vec![0.0; n],
            even: 1.0,
            floor: 0.0,
            weights: None,
            order: Vec::new(),
            tiny,
        }
    }

    fn done(&self) -> bool {
        self.stage >= self.plan.len()
    }

    /// About `work` kernel weights' worth of the fit (at least one place). True when
    /// the fit is done.
    fn step(&mut self, work: usize) -> bool {
        let mut spent = 0;
        while !self.done() && (spent == 0 || spent < work) {
            let pass = self.plan[self.stage];
            let n = self.pts.len();
            match pass {
                Pass::Table(set) => {
                    if self.tables[set].is_none() {
                        self.tables[set] = Some(Arc::new(Table::new(&self.data, &self.sets[set])));
                        spent += place_cost(pass, n) * self.sets[set].len();
                    }
                    self.next();
                }
                Pass::Leftover => {
                    self.left = self.y.iter().zip(&self.level[0]).map(|(a, b)| a - b).collect();
                    self.squares = self.left.iter().map(|v| v * v).collect();
                    self.even = rms(&self.left);
                    if self.floor == 0.0 {
                        self.floor = FLOOR * rms(&self.left);
                    }
                    self.order = (0..n).collect();
                    let left = &self.left;
                    self.order.sort_by(|&a, &b| left[a].total_cmp(&left[b]));
                    spent += n;
                    self.next();
                }
                Pass::Weights => {
                    self.weights = Some(self.spread[0].iter().map(|s| 1.0 / (s * s)).collect());
                    spent += n;
                    self.next();
                }
                Pass::Likelihood => {
                    let model = Likelihood::fit(&self.data, &self.y);
                    self.floor = model.floor;
                    for set in 0..self.sets.len() {
                        let f = model.at_all(&self.sets[set]);
                        self.level[set] = f.level;
                        self.spread[set] = f.spread;
                    }
                    spent += 50 * n;
                    self.next();
                }
                Pass::Level { set, .. } | Pass::Spread { set, .. } => {
                    let Some(table) = self.tables.get(set).cloned().flatten() else {
                        self.next();
                        continue;
                    };
                    let count = self.sets[set].len();
                    let each = place_cost(pass, n);
                    while self.cursor < count && (spent == 0 || spent < work) {
                        let r = self.cursor;
                        self.place(pass, &table, r);
                        self.cursor += 1;
                        spent += each.max(1);
                    }
                    if self.cursor >= count {
                        self.next();
                    }
                }
            }
        }
        self.done()
    }

    fn next(&mut self) {
        self.stage += 1;
        self.cursor = 0;
    }

    /// One place of a level or spread pass.
    fn place(&mut self, pass: Pass, table: &Table, r: usize) {
        let dims = self.data.dims();
        let own = |set: usize| (set == 0).then_some(r);
        match pass {
            Pass::Level { set, weighted, median } => {
                let k = table.row(r);
                self.level[set][r] = if median {
                    // (a median fitted near a point that is itself in the fit, with the most
                    // weight, passes through it: each point's own fit is made without it)
                    local_quantile(k, own(set), &self.pts, table.at[r], dims, &self.y, 0.5, self.tiny)
                } else {
                    let extra = if weighted { self.weights.as_deref() } else { None };
                    local_line(k, extra, &self.pts, table.at[r], dims, &self.y)
                };
            }
            Pass::Spread { set, how } => {
                let floor = self.floor.max(1e-300);
                self.spread[set][r] = match how {
                    Spreading::Even => self.even,
                    // (a straight-line fit near each place, which does not lean towards the
                    // middle at the ends as a plain average does)
                    Spreading::Squares => local_line(table.row(r), None, &self.pts, table.at[r], dims, &self.squares).max(floor * floor).sqrt(),
                    Spreading::Quartiles => {
                        let [lo, hi] = weighted_quartiles(table.row(r), own(set), &self.order, &self.left);
                        // (the quartiles of a bell are 1.349 of its spreads apart)
                        ((hi - lo) / 1.349).max(floor)
                    }
                };
            }
            _ => {}
        }
    }

    /// The fitted level and spread at set `set` of places.
    fn fitted(&self, set: usize) -> Fitted {
        Fitted { level: self.level.get(set).cloned().unwrap_or_default(), spread: self.spread.get(set).cloned().unwrap_or_default() }
    }
}

/// Fit `y` from `data` by `method` all at once, and give the fitted level and spread at
/// the data points themselves and at each set of `places` (the tests).
#[cfg(test)]
fn fit(method: Method, data: &Inputs, y: &[f64], places: &[&[[f64; 2]]]) -> (Fitted, Vec<Fitted>) {
    let mut job = Job::new(method, data.clone(), y, places.iter().map(|p| p.to_vec()).collect(), Vec::new());
    while !job.step(usize::MAX) {}
    (job.fitted(0), (1..job.sets.len()).map(|s| job.fitted(s)).collect())
}

/// The fit by likelihood: the level a combination of smooth bumps (cubic B-splines)
/// along each input, the logarithm of the spread another, chosen together to make the
/// data most probable under normal noise: the least of
/// Σ log g(xᵢ) + (yᵢ − f(xᵢ))² / (2 g(xᵢ)²), plus a small penalty on bumps that differ
/// from their neighbours (so that a few points cannot bend the curves at will).
struct Likelihood {
    basis: Basis,
    level: Vec<f64>,
    log_spread: Vec<f64>,
    floor: f64,
}

/// Cubic B-splines along each input, on knots that cut its points into equal shares.
struct Basis {
    /// Each input's knots (in spreads), from its least to its greatest point.
    knots: Vec<Vec<f64>>,
}

/// Pieces of each input's B-spline curve.
const PIECES: usize = 6;
/// Functions per input: one more than the pieces, and two more for a cubic.
const PER_INPUT: usize = PIECES + 3;

impl Basis {
    fn new(data: &Inputs) -> Basis {
        let n = data.len();
        let knots = (0..data.dims())
            .map(|k| {
                let mut v: Vec<f64> = (0..n).map(|i| data.point(i)[k]).collect();
                v.sort_by(f64::total_cmp);
                if v.is_empty() {
                    return (0..=PIECES).map(|j| j as f64).collect();
                }
                let mut knots: Vec<f64> = (0..=PIECES).map(|j| v[(j * (n - 1)) / PIECES]).collect();
                // (knots that coincide, when many points share a value, are spread apart)
                for j in 1..knots.len() {
                    if knots[j] <= knots[j - 1] + 1e-9 {
                        knots[j] = knots[j - 1] + 1e-3;
                    }
                }
                knots
            })
            .collect();
        Basis { knots }
    }

    fn size(&self) -> usize {
        self.knots.len() * PER_INPUT
    }

    /// Every function's value at a point (in spreads).
    fn at(&self, p: [f64; 2], out: &mut Vec<f64>) {
        out.clear();
        for (k, knots) in self.knots.iter().enumerate() {
            let (lo, hi) = (knots[0], knots[PIECES]);
            // (beyond the ends the curve goes on level, as at the end)
            let x = p[k].max(lo).min(hi);
            let piece = knots.partition_point(|&t| t <= x).clamp(1, PIECES) - 1;
            let (a, b) = (knots[piece], knots[piece + 1]);
            let t = ((x - a) / (b - a).max(1e-12)).max(0.0).min(1.0);
            // the four uniform cubic B-splines that are not 0 on this piece (the knots
            // are evenly spaced in rank, not in value: each piece is mapped onto [0, 1])
            let s = 1.0 - t;
            let w = [s * s * s / 6.0, (3.0 * t * t * t - 6.0 * t * t + 4.0) / 6.0, (-3.0 * t * t * t + 3.0 * t * t + 3.0 * t + 1.0) / 6.0, t * t * t / 6.0];
            let start = out.len();
            out.resize(start + PER_INPUT, 0.0);
            for (j, v) in w.iter().enumerate() {
                out[start + piece + j] = *v;
            }
        }
    }
}

/// The penalty on neighbouring bumps: how much a curve's second differences cost,
/// against the data's own weight (a share of the weights' total per function).
const BEND_LEVEL: f64 = 0.02;
const BEND_SPREAD: f64 = 0.1;
/// Rounds of the likelihood fit, at most.
const LIKELIHOOD_ROUNDS: usize = 40;

impl Likelihood {
    fn fit(data: &Inputs, y: &[f64]) -> Likelihood {
        let basis = Basis::new(data);
        let n = data.len().min(y.len());
        let m = basis.size();
        let rows: Vec<Vec<f64>> = (0..n)
            .map(|i| {
                let mut v = Vec::with_capacity(m);
                basis.at(data.point(i), &mut v);
                v
            })
            .collect();
        // second differences within each input's functions
        let mut bend = vec![0.0; m * m];
        for k in 0..basis.knots.len() {
            for j in 0..PER_INPUT - 2 {
                let idx = [k * PER_INPUT + j, k * PER_INPUT + j + 1, k * PER_INPUT + j + 2];
                let c = [1.0, -2.0, 1.0];
                for a in 0..3 {
                    for b in 0..3 {
                        bend[idx[a] * m + idx[b]] += c[a] * c[b];
                    }
                }
            }
        }
        // a weighted least-squares fit of `target` on the functions, with the penalty
        let solve_weighted = |w: &[f64], target: &[f64], strength: f64, unit: f64| -> Option<Vec<f64>> {
            let mut a = vec![0.0; m * m];
            let mut b = vec![0.0; m];
            for i in 0..n {
                let r = &rows[i];
                for p in 0..m {
                    if r[p] == 0.0 {
                        continue;
                    }
                    b[p] += w[i] * r[p] * target[i];
                    for q in 0..m {
                        a[p * m + q] += w[i] * r[p] * r[q];
                    }
                }
            }
            // (the penalty is measured against what the points would weigh with weight
            // `unit` each: fixed, so that where the spread is large and the points weigh
            // little, the penalty counts for more, as it does in the likelihood)
            let trace = unit * n as f64 / m as f64;
            let trace = if trace > 0.0 && trace.is_finite() { trace } else { 1.0 };
            for p in 0..m {
                for q in 0..m {
                    a[p * m + q] += strength * trace * bend[p * m + q];
                }
                // (and a whisper of ridge, so that functions no point touches stay put)
                a[p * m + p] += 1e-6 * trace;
            }
            crate::math::linalg::solve(&mut a, &mut b, m)
        };
        let dot = |r: &[f64], c: &[f64]| r.iter().zip(c).map(|(a, b)| a * b).sum::<f64>();
        let ones = vec![1.0; n];
        let mut level = solve_weighted(&ones, &y[..n], BEND_LEVEL, 1.0).unwrap_or_else(|| vec![0.0; m]);
        let left: Vec<f64> = (0..n).map(|i| y[i] - dot(&rows[i], &level)).collect();
        let floor = FLOOR * rms(&left);
        let unit = 1.0 / (rms(&left) * rms(&left));
        // the spread starts even: every function at the log of the overall spread (B-splines add up to 1)
        let start = rms(&left).ln() / basis.knots.len().max(1) as f64;
        let mut log_spread = vec![start; m];
        let objective = |level: &[f64], log_spread: &[f64]| -> f64 {
            let mut total = 0.0;
            for i in 0..n {
                let s = dot(&rows[i], log_spread).max(floor.ln());
                let r = y[i] - dot(&rows[i], level);
                total += s + 0.5 * r * r * (-2.0 * s).exp();
            }
            total
        };
        let mut best = objective(&level, &log_spread);
        for _ in 0..LIKELIHOOD_ROUNDS {
            // the level, for this spread: weighted least squares, exactly
            let w: Vec<f64> = (0..n).map(|i| (-2.0 * dot(&rows[i], &log_spread).max(floor.ln())).exp()).collect();
            let next_level = solve_weighted(&w, &y[..n], BEND_LEVEL, unit).unwrap_or_else(|| level.clone());
            // the spread, for this level: a step of Fisher scoring on its logarithm
            // (the gradient of the objective, over its expected curvature 2 Σ ψψᵀ)
            let resid: Vec<f64> = (0..n).map(|i| y[i] - dot(&rows[i], &next_level)).collect();
            let target: Vec<f64> = (0..n)
                .map(|i| {
                    let s = dot(&rows[i], &log_spread);
                    // (the working response of the step: where it would put log g)
                    s + 0.5 * (resid[i] * resid[i] * (-2.0 * s).exp() - 1.0)
                })
                .collect();
            let proposal = solve_weighted(&ones, &target, BEND_SPREAD, 1.0).unwrap_or_else(|| log_spread.clone());
            // halve the step until the objective does not rise
            let mut step = 1.0;
            let mut accepted = false;
            for _ in 0..12 {
                let trial: Vec<f64> = log_spread.iter().zip(&proposal).map(|(a, b)| a + step * (b - a)).collect();
                let value = objective(&next_level, &trial);
                if value.is_finite() && value <= best + 1e-12 * best.abs() {
                    let change = best - value;
                    level = next_level.clone();
                    log_spread = trial;
                    best = value;
                    accepted = true;
                    if change < 1e-9 * n as f64 {
                        step = 0.0;
                    }
                    break;
                }
                step *= 0.5;
            }
            if !accepted || step == 0.0 {
                break;
            }
        }
        Likelihood { basis, level, log_spread, floor }
    }

    fn at_all(&self, places: &[[f64; 2]]) -> Fitted {
        let mut v = Vec::with_capacity(self.basis.size());
        let mut out = Fitted { level: Vec::with_capacity(places.len()), spread: Vec::with_capacity(places.len()) };
        for p in places {
            self.basis.at(*p, &mut v);
            let dot = |c: &[f64]| v.iter().zip(c).map(|(a, b)| a * b).sum::<f64>();
            out.level.push(dot(&self.level));
            out.spread.push(dot(&self.log_spread).exp().max(self.floor));
        }
        out
    }
}

// ----- the check ----------------------------------------------------------------------------------

/// A typical distance between two of the samples (the median of the distances among
/// up to 200 of them, evenly picked): the width of the bells the dependence measure
/// compares them with.
fn typical_distance(x: &[f64]) -> f64 {
    let step = x.len().div_ceil(200).max(1);
    let picked: Vec<f64> = x.iter().step_by(step).copied().collect();
    let mut d: Vec<f64> = Vec::with_capacity(picked.len() * picked.len() / 2);
    for i in 0..picked.len() {
        for j in 0..i {
            d.push((picked[i] - picked[j]).abs());
        }
    }
    d.retain(|v| v.is_finite() && *v > 0.0);
    if d.is_empty() {
        return 1.0;
    }
    let mid = d.len() / 2;
    *d.select_nth_unstable_by(mid, f64::total_cmp).1
}

/// The input side of the dependence measure (HSIC, as in `causal::Hsic`): the table of
/// likenesses between the inputs, centred. With two inputs, two points are alike when
/// they are alike in both (the product of a bell along each).
struct Likeness {
    n: usize,
    k: Vec<f64>,
}

impl Likeness {
    fn new(cols: &[&[f64]]) -> Likeness {
        let n = cols.iter().map(|c| c.len()).min().unwrap_or(0);
        let mut k = vec![1.0; n * n];
        for c in cols {
            let s = typical_distance(&c[..n]);
            for i in 0..n {
                for j in 0..n {
                    k[i * n + j] *= (-0.5 * ((c[i] - c[j]) / s).powi(2)).exp();
                }
            }
        }
        if n > 0 {
            let rows: Vec<f64> = (0..n).map(|i| k[i * n..(i + 1) * n].iter().sum::<f64>() / n as f64).collect();
            let all = rows.iter().sum::<f64>() / n as f64;
            for i in 0..n {
                for j in 0..n {
                    k[i * n + j] += all - rows[i] - rows[j];
                }
            }
        }
        Likeness { n, k }
    }

    /// The other side's table: likenesses between the values `v`.
    fn other(&self, v: &[f64]) -> Vec<f64> {
        let n = self.n.min(v.len());
        let s = typical_distance(&v[..n]);
        let mut l = vec![0.0; self.n * self.n];
        for i in 0..n {
            for j in 0..n {
                l[i * self.n + j] = (-0.5 * ((v[i] - v[j]) / s).powi(2)).exp();
            }
        }
        l
    }

    /// The measure for the pairing as it is.
    fn value(&self, l: &[f64]) -> f64 {
        let n = self.n;
        if n == 0 {
            return 0.0;
        }
        self.k.iter().zip(l).map(|(a, b)| a * b).sum::<f64>() / (n * n) as f64
    }

    /// The measure with the other side dealt out again at random.
    fn shuffled(&self, l: &[f64], rng: &mut Rng, order: &mut Vec<usize>) -> f64 {
        let n = self.n;
        if n == 0 {
            return 0.0;
        }
        order.clear();
        order.extend(0..n);
        rng.shuffle(order);
        let mut sum = 0.0;
        for i in 0..n {
            let (row, to) = (&self.k[i * n..(i + 1) * n], order[i] * n);
            for j in 0..n {
                sum += row[j] * l[to + order[j]];
            }
        }
        sum / (n * n) as f64
    }
}

/// Below this share of shuffles at least as dependent, what is left over is called dependent.
const LEVEL: f64 = 0.05;
/// How many shuffled pairings each dependence is judged against.
const SHUFFLES: usize = 99;
/// With this many shuffles reaching the real pairing, p is at least `LEVEL`.
const ENOUGH: usize = 4;

/// p, from how many of `shuffles` shuffled pairings were at least as dependent as the
/// real one (the real pairing counts as one of the pairings).
fn p_of(reached: usize, shuffles: usize) -> f64 {
    (reached + 1) as f64 / (shuffles + 1) as f64
}

// ----- the analysis of a sample ---------------------------------------------------------------------

/// How much work one piece of a fit is, in kernel weights (a tenth of a millisecond or so).
const WORK: usize = 40_000;

/// What a sample's analysis depends on.
#[derive(Clone, Copy, Debug, PartialEq)]
struct Key {
    setup: Setup,
    seed: u64,
    /// What is left over taken from points the fit has not seen (each half from the
    /// fit on the other half).
    held: bool,
}

/// Places along a column's input at which its curves are drawn.
const GRID: usize = 48;
/// The third variable's levels at which the curves are drawn when it is measured: its
/// 20th, 50th and 80th percentiles.
const LEVELS: [f64; 3] = [0.2, 0.5, 0.8];

/// The two halves of a sample: the points in even places and in odd ones (the points
/// are drawn independently, so this is a split at random).
fn halves(n: usize) -> [Vec<usize>; 2] {
    [(0..n).step_by(2).collect(), (1..n).step_by(2).collect()]
}

/// The `q`-quantile of sorted numbers (by straight lines between them).
fn percentile(sorted: &[f64], q: f64) -> f64 {
    if sorted.is_empty() {
        return 0.0;
    }
    let at = q.max(0.0).min(1.0) * (sorted.len() - 1) as f64;
    let (i, t) = (at.floor() as usize, at.fract());
    let j = (i + 1).min(sorted.len() - 1);
    sorted[i] + (sorted[j] - sorted[i]) * t
}

/// The smallest and largest of some numbers ((0, 0) for none).
fn extent(v: &[f64]) -> (f64, f64) {
    let lo = v.iter().copied().fold(f64::INFINITY, f64::min);
    let hi = v.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    if lo.is_finite() && hi.is_finite() {
        (lo, hi)
    } else {
        (0.0, 0.0)
    }
}

/// One direction of the check: the output fitted from the input (and the third
/// variable, when it is measured).
struct Column {
    data: Inputs,
    output: Vec<f64>,
    /// Where the curves are drawn (in spreads): `GRID` places across the input at each
    /// of the third variable's levels (one curve, if it is not measured); the input at
    /// each of those places, and the third variable's value at each level.
    places: Vec<[f64; 2]>,
    across: Vec<f64>,
    levels: [f64; 3],
    /// Where along the input each curve is drawn: where there are points with the
    /// third variable near its level (everywhere, without one).
    spans: [(f64, f64); 3],
    /// The kernel weights already made, by half (one, when the fit is on all the points)
    /// and by set of places, kept for the next fit of the same column.
    cache: Vec<Vec<Option<Arc<Table>>>>,
}

impl Column {
    /// `curves`: whether the places for the curves are wanted (the shown sample).
    fn new(input: &[f64], output: &[f64], third: Option<&[f64]>, held: bool, curves: bool) -> Column {
        let mut cols = vec![input.to_vec()];
        if let Some(u) = third {
            cols.push(u.to_vec());
        }
        let data = Inputs::new(cols);
        let (lo, hi) = extent(input);
        let across: Vec<f64> = if curves { (0..GRID).map(|g| lo + (hi - lo) * g as f64 / (GRID - 1) as f64).collect() } else { Vec::new() };
        let mut levels = [0.0; 3];
        let mut spans = [(f64::NEG_INFINITY, f64::INFINITY); 3];
        if let Some(u) = third {
            let mut sorted = u.to_vec();
            sorted.sort_by(f64::total_cmp);
            levels = LEVELS.map(|q| percentile(&sorted, q));
            let near = 0.5 * variance(u).sqrt();
            for (k, z) in levels.iter().enumerate() {
                let mut along: Vec<f64> = input.iter().zip(u).filter(|(_, v)| (*v - z).abs() <= near).map(|(a, _)| *a).collect();
                along.sort_by(f64::total_cmp);
                if along.len() >= 2 {
                    spans[k] = (percentile(&along, 0.05), percentile(&along, 0.95));
                }
            }
        }
        let places = if !curves {
            Vec::new()
        } else if third.is_some() {
            let (sx, sz) = (data.scale[0], data.scale.get(1).copied().unwrap_or(1.0));
            levels.iter().flat_map(|z| across.iter().map(move |a| [a / sx, z / sz])).collect()
        } else {
            across.iter().map(|a| [a / data.scale[0], 0.0]).collect()
        };
        Column { data, output: output.to_vec(), places, across, levels, spans, cache: vec![Vec::new(); if held { 2 } else { 1 }] }
    }

    fn len(&self) -> usize {
        self.data.len().min(self.output.len())
    }

    /// The jobs of one fit by `method`: on all the points, or on each half with the
    /// other half's points as places (and then the curves' places).
    fn jobs(&self, method: Method) -> Vec<Job> {
        let n = self.len();
        if self.cache.len() < 2 {
            let places = if self.places.is_empty() { Vec::new() } else { vec![self.places.clone()] };
            return vec![Job::new(method, self.data.clone(), &self.output[..n], places, self.cache[0].clone())];
        }
        halves(n)
            .iter()
            .enumerate()
            .map(|(h, from)| {
                let other = &halves(n)[1 - h];
                let mut places = vec![other.iter().map(|&i| self.data.point(i)).collect::<Vec<_>>()];
                if !self.places.is_empty() {
                    places.push(self.places.clone());
                }
                let y: Vec<f64> = from.iter().map(|&i| self.output[i]).collect();
                Job::new(method, self.data.subset(from), &y, places, self.cache.get(h).cloned().unwrap_or_default())
            })
            .collect()
    }

    /// Keep the kernel weights a finished job made, for the next fit.
    fn keep(&mut self, h: usize, job: &Job) {
        if let Some(cache) = self.cache.get_mut(h) {
            cache.resize(job.tables.len().max(cache.len()), None);
            for (kept, made) in cache.iter_mut().zip(&job.tables) {
                if kept.is_none() {
                    *kept = made.clone();
                }
            }
        }
    }

    /// One piece of the fit `jobs`; the analysis when it is the last.
    fn work(&mut self, jobs: &mut [Job]) -> Option<Analysis> {
        if let Some(h) = jobs.iter().position(|j| !j.done()) {
            if jobs[h].step(WORK) {
                self.keep(h, &jobs[h]);
            }
            return None;
        }
        Some(self.assemble(jobs))
    }

    /// What finished jobs come to.
    fn assemble(&self, jobs: &[Job]) -> Analysis {
        let n = self.len();
        let y = &self.output[..n];
        let (fitted, curves) = if jobs.len() < 2 {
            let job = jobs.first();
            (job.map(|j| j.fitted(0)).unwrap_or_default(), job.map(|j| j.fitted(1)).unwrap_or_default())
        } else {
            let mut fitted = Fitted { level: vec![0.0; n], spread: vec![1.0; n] };
            for (h, other) in [1usize, 0].iter().enumerate() {
                let to = &halves(n)[*other];
                let f = jobs[h].fitted(1);
                for (k, &i) in to.iter().enumerate() {
                    if let (Some(l), Some(s)) = (f.level.get(k), f.spread.get(k)) {
                        fitted.level[i] = *l;
                        fitted.spread[i] = *s;
                    }
                }
            }
            // (the curves drawn: the two halves' fits, averaged)
            let (a, b) = (jobs[0].fitted(2), jobs[1].fitted(2));
            let avg = |u: &[f64], v: &[f64]| u.iter().zip(v).map(|(p, q)| 0.5 * (p + q)).collect::<Vec<f64>>();
            (fitted, Fitted { level: avg(&a.level, &b.level), spread: avg(&a.spread, &b.spread) })
        };
        let raw: Vec<f64> = (0..n).map(|i| y[i] - fitted.level.get(i).copied().unwrap_or(0.0)).collect();
        let spread = variance(y).sqrt();
        let exact = spread.is_nan() || spread <= 0.0 || rms(&raw) <= 1e-7 * spread || raw.iter().all(|r| *r == 0.0);
        let z = if exact { vec![0.0; n] } else { (0..n).map(|i| raw[i] / fitted.spread.get(i).copied().unwrap_or(1.0).max(1e-300)).collect() };
        Analysis { fitted, raw, z, curves, exact }
    }
}

/// A finished fit of a column: the level and spread at the points (as used for what is
/// left over), what is left over as it is and divided by the spread, and the curves.
#[derive(Clone, Debug, Default)]
struct Analysis {
    fitted: Fitted,
    raw: Vec<f64>,
    z: Vec<f64>,
    curves: Fitted,
    /// Nothing is left over (to rounding): the output an exact function of the input.
    exact: bool,
}

/// One dependence being judged: the real pairing's, and shuffled ones as they come.
struct Side {
    l: Option<Vec<f64>>,
    value: f64,
    shuffles: Vec<f64>,
    rng: Rng,
    /// Nothing is left over to test (an exact fit): independent of anything.
    exact: bool,
}

impl Side {
    fn new(seed: u64) -> Side {
        Side { l: None, value: 0.0, shuffles: Vec::with_capacity(SHUFFLES), rng: Rng::new(seed), exact: false }
    }

    fn done(&self) -> bool {
        self.exact || self.shuffles.len() >= SHUFFLES
    }

    fn reached(&self) -> usize {
        self.shuffles.iter().filter(|s| **s >= self.value).count()
    }

    fn p(&self) -> f64 {
        if self.exact {
            1.0
        } else {
            p_of(self.reached(), self.shuffles.len())
        }
    }

    fn independent(&self) -> bool {
        self.p() >= LEVEL
    }

    /// One piece of the work: the table, or one shuffle.
    fn step(&mut self, like: &Likeness, z: &[f64], order: &mut Vec<usize>) {
        match &self.l {
            None => {
                let l = like.other(z);
                self.value = like.value(&l);
                self.l = Some(l);
            }
            Some(l) => {
                let s = like.shuffled(l, &mut self.rng, order);
                self.shuffles.push(s);
            }
        }
    }
}

/// The verdict on one sample.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Verdict {
    XCausesY,
    YCausesX,
    /// What is left over looks independent of the input both ways: both fit.
    Both,
    /// Dependent both ways: neither fits.
    Neither,
}

impl Verdict {
    fn of(forwards: bool, backwards: bool) -> Verdict {
        match (forwards, backwards) {
            (true, false) => Verdict::XCausesY,
            (false, true) => Verdict::YCausesX,
            (true, true) => Verdict::Both,
            (false, false) => Verdict::Neither,
        }
    }

    /// Its place in the counts: X → Y, both fit, neither fits, Y → X.
    fn slot(self) -> usize {
        match self {
            Verdict::XCausesY => 0,
            Verdict::Both => 1,
            Verdict::Neither => 2,
            Verdict::YCausesX => 3,
        }
    }
}

/// One method's fit of one column of the shown sample, and its check.
struct Slot {
    jobs: Vec<Job>,
    result: Option<Analysis>,
    side: Side,
}

/// The sample shown, its two columns, and each method's fit and check (made when
/// they are first wanted, and kept).
struct Shown {
    key: Key,
    sample: Sample,
    cols: [Column; 2],
    likes: [Option<Likeness>; 2],
    /// By column, then by method (in the order of `METHODS`).
    slots: [Vec<Slot>; 2],
    order: Vec<usize>,
}

impl Shown {
    fn new(key: Key, draws: &Draws) -> Shown {
        let sample = draws.sample(&key.setup);
        let third = (key.setup.seen && key.setup.graph.third()).then_some(sample.u.as_slice());
        let cols = [Column::new(&sample.x, &sample.y, third, key.held, true), Column::new(&sample.y, &sample.x, third, key.held, true)];
        let slots = [0u64, 1].map(|c| (0..METHODS.len()).map(|m| Slot { jobs: Vec::new(), result: None, side: Side::new(key.seed.wrapping_mul(0x9E37_79B9).wrapping_add(31 + 7 * c + m as u64)) }).collect());
        Shown { key, sample, cols, likes: [None, None], slots, order: Vec::new() }
    }

    /// Is there work left on these methods (the one shown first)?
    fn pending(&self, wanted: &[usize]) -> bool {
        wanted.iter().any(|&m| (0..2).any(|c| self.slots[c].get(m).is_none_or(|s| s.result.is_none() || !s.side.done())))
    }

    /// The fits of `m` are done in both columns.
    fn fitted(&self, m: usize) -> bool {
        (0..2).all(|c| self.slots[c].get(m).is_some_and(|s| s.result.is_some()))
    }

    /// The checks of `m` are done in both columns.
    fn judged(&self, m: usize) -> bool {
        (0..2).all(|c| self.slots[c].get(m).is_some_and(|s| s.result.is_some() && s.side.done()))
    }

    /// One piece of the work on `wanted` (in order): the fits first, then the tables
    /// of likenesses, then the checks, the column with the least done going next.
    fn step(&mut self, wanted: &[usize]) {
        for &m in wanted {
            for c in 0..2 {
                let Some(slot) = self.slots[c].get_mut(m) else { continue };
                if slot.result.is_some() {
                    continue;
                }
                if slot.jobs.is_empty() {
                    slot.jobs = self.cols[c].jobs(METHODS[m]);
                }
                if let Some(done) = self.cols[c].work(&mut slot.jobs) {
                    slot.side.exact = done.exact;
                    slot.result = Some(done);
                    slot.jobs = Vec::new();
                }
                return;
            }
        }
        for c in 0..2 {
            if self.likes[c].is_none() {
                let cols: Vec<&[f64]> = self.cols[c].data.cols.iter().map(|v| v.as_slice()).collect();
                self.likes[c] = Some(Likeness::new(&cols));
                return;
            }
        }
        for &m in wanted {
            let next = (0..2)
                .filter(|&c| self.slots[c].get(m).is_some_and(|s| !s.side.done()))
                .min_by_key(|&c| self.slots[c][m].side.shuffles.len() + self.slots[c][m].side.l.is_some() as usize);
            if let Some(c) = next {
                let (Some(like), Some(slot)) = (&self.likes[c], self.slots[c].get_mut(m)) else { return };
                if let Some(result) = &slot.result {
                    slot.side.step(like, &result.z, &mut self.order);
                }
                return;
            }
        }
    }

    /// The verdict of method `m`, once both its checks are done.
    fn verdict(&self, m: usize) -> Option<Verdict> {
        self.judged(m).then(|| Verdict::of(self.slots[0][m].side.independent(), self.slots[1][m].side.independent()))
    }
}

// ----- the scoreboard: every method over many new samples ----------------------------------------------

/// How many new samples the scoreboard counts over.
const RUNS: usize = 100;
/// Its rows: the methods, and then the truth itself (what is left over divided by the
/// true spread, which only the scene knows).
const ROWS: usize = METHODS.len() + 1;
const TRUTH_ROW: usize = METHODS.len();
/// Shuffles in one piece of the work on a sample of the count.
const CHUNK: usize = 6;

/// A method's counts over the samples so far.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
struct Row {
    /// How often each direction's check passed.
    pass: [usize; 2],
    /// The verdicts: X → Y, both fit, neither fits, Y → X.
    verdicts: [usize; 4],
    /// The sums of the level's and the spread's errors (see `errors`), over `measured` samples.
    level: f64,
    spread: f64,
    measured: usize,
}

/// How far a fit is from the truth, at the sample's points: the level's root mean
/// square distance from the true level, over the true spread's root mean square; and
/// the spread's root mean square distance from the true one on a scale of ratios
/// (logarithms), leaving out a factor common to all the points (which the check
/// cannot see, and which a fit from quartiles gets wrong for noise that is not a bell).
fn errors(fitted: &Fitted, truth: &[(f64, f64)]) -> (f64, f64) {
    let n = truth.len().min(fitted.level.len()).min(fitted.spread.len());
    if n == 0 {
        return (0.0, 0.0);
    }
    let typical = rms(&truth[..n].iter().map(|t| t.1).collect::<Vec<_>>());
    let level = rms(&(0..n).map(|i| fitted.level[i] - truth[i].0).collect::<Vec<_>>()) / typical;
    let logs: Vec<f64> = (0..n).map(|i| (fitted.spread[i].max(1e-300) / truth[i].1.max(1e-300)).ln()).collect();
    let m = mean(&logs);
    let centred: Vec<f64> = logs.iter().map(|v| v - m).collect();
    let spread = (centred.iter().map(|v| v * v).sum::<f64>() / n as f64).sqrt();
    (level, spread)
}

/// One direction's check on a sample of the count, done as briefly as its verdict
/// allows: independent as soon as `ENOUGH` shuffles reach the real pairing.
#[derive(Default)]
struct Quick {
    exact: bool,
    l: Option<Vec<f64>>,
    value: f64,
    done: usize,
    reached: usize,
}

impl Quick {
    fn decided(&self) -> bool {
        self.exact || self.reached >= ENOUGH || self.done >= SHUFFLES
    }

    fn independent(&self) -> bool {
        self.exact || self.reached >= ENOUGH
    }
}

/// One method's fit and check of one column, on a sample of the count.
#[derive(Default)]
struct RunSlot {
    wanted: bool,
    jobs: Vec<Job>,
    fitted: Option<Fitted>,
    z: Option<Vec<f64>>,
    quick: Quick,
}

/// A sample of the count, part way through.
struct Run {
    truth: Vec<(f64, f64)>,
    /// Is the truth of the first column's model known (no hidden third variable)?
    known: bool,
    cols: [Column; 2],
    likes: [Option<Likeness>; 2],
    slots: [Vec<RunSlot>; 2],
    rng: Rng,
    order: Vec<usize>,
}

impl Run {
    fn new(key: &Key, seed: u64) -> Run {
        let s = &key.setup;
        let sample = Draws::new(seed, s.n).sample(s);
        let third = (s.seen && s.graph.third()).then_some(sample.u.as_slice());
        let cols = [Column::new(&sample.x, &sample.y, third, key.held, false), Column::new(&sample.y, &sample.x, third, key.held, false)];
        let truth: Vec<(f64, f64)> = sample.x.iter().zip(&sample.u).map(|(x, u)| s.truth(*x, *u)).collect();
        let known = !s.graph.third() || s.seen;
        let slots = [0, 1].map(|c| (0..ROWS).map(|r| RunSlot { wanted: r < TRUTH_ROW || (c == 0 && known), ..Default::default() }).collect());
        Run { truth, known, cols, likes: [None, None], slots, rng: Rng::new(seed ^ 0x5DEE_CE66), order: Vec::new() }
    }

    /// One piece of the work; true when the sample is done.
    fn step(&mut self) -> bool {
        for c in 0..2 {
            if self.likes[c].is_none() {
                let cols: Vec<&[f64]> = self.cols[c].data.cols.iter().map(|v| v.as_slice()).collect();
                self.likes[c] = Some(Likeness::new(&cols));
                return false;
            }
        }
        for c in 0..2 {
            for r in 0..ROWS {
                let slot = &mut self.slots[c][r];
                if !slot.wanted || slot.quick.decided() && slot.z.is_some() {
                    continue;
                }
                if slot.z.is_none() {
                    if r == TRUTH_ROW {
                        let n = self.cols[c].len();
                        let y = &self.cols[c].output;
                        slot.z = Some((0..n).map(|i| (y[i] - self.truth[i].0) / self.truth[i].1.max(1e-300)).collect());
                        return false;
                    }
                    if slot.jobs.is_empty() {
                        slot.jobs = self.cols[c].jobs(METHODS[r]);
                    }
                    if let Some(done) = self.cols[c].work(&mut slot.jobs) {
                        slot.quick.exact = done.exact;
                        slot.z = Some(done.z);
                        slot.fitted = Some(done.fitted);
                        slot.jobs = Vec::new();
                    }
                    return false;
                }
                let (Some(like), Some(z)) = (&self.likes[c], &slot.z) else { continue };
                let q = &mut slot.quick;
                match &q.l {
                    None => {
                        let l = like.other(z);
                        q.value = like.value(&l);
                        q.l = Some(l);
                    }
                    Some(l) => {
                        for _ in 0..CHUNK {
                            if q.decided() {
                                break;
                            }
                            let s = like.shuffled(l, &mut self.rng, &mut self.order);
                            q.reached += (s >= q.value) as usize;
                            q.done += 1;
                        }
                        if q.decided() {
                            q.l = None;
                        }
                    }
                }
                return false;
            }
        }
        true
    }

    /// What each row makes of the finished sample.
    fn tally(&self, rows: &mut [Row; ROWS]) {
        for (r, row) in rows.iter_mut().enumerate() {
            let pass = [self.slots[0][r].quick.independent(), r < TRUTH_ROW && self.slots[1][r].quick.independent()];
            if !self.slots[0][r].wanted {
                continue;
            }
            row.pass[0] += pass[0] as usize;
            row.pass[1] += pass[1] as usize;
            if r < TRUTH_ROW {
                row.verdicts[Verdict::of(pass[0], pass[1]).slot()] += 1;
                if let (true, Some(fitted)) = (self.known, &self.slots[0][r].fitted) {
                    let (l, s) = errors(fitted, &self.truth);
                    row.level += l;
                    row.spread += s;
                    row.measured += 1;
                }
            }
        }
    }
}

/// Every method's verdicts over `RUNS` new samples like the one shown.
struct Board {
    key: Option<Key>,
    rows: [Row; ROWS],
    done: usize,
    run: Option<Run>,
}

impl Board {
    fn new() -> Board {
        Board { key: None, rows: [Row::default(); ROWS], done: 0, run: None }
    }

    fn restart(&mut self, key: Key) {
        *self = Board::new();
        self.key = Some(key);
    }

    fn pending(&self) -> bool {
        self.key.is_some() && self.done < RUNS
    }

    /// The seed of new sample `k` (none of them the one shown).
    fn seed_of(seed: u64, k: usize) -> u64 {
        seed.wrapping_mul(1_000_003).wrapping_add(7_919 * (k as u64 + 1))
    }

    fn step(&mut self) {
        let Some(key) = self.key else { return };
        if self.done >= RUNS {
            return;
        }
        let k = self.done;
        let run = self.run.get_or_insert_with(|| Run::new(&key, Board::seed_of(key.seed, k)));
        if run.step() {
            run.tally(&mut self.rows);
            self.done += 1;
            self.run = None;
        }
    }

    /// The whole count at once (the tests).
    #[cfg(test)]
    fn finish(&mut self) {
        while self.pending() {
            self.step();
        }
    }
}

// ----- the experiments ---------------------------------------------------------------------------------

/// An experiment: its button and caption, and every setting it depends on.
struct Trial {
    button: &'static str,
    caption: &'static str,
    setup: Setup,
    method: Method,
    scaled: bool,
    held: bool,
}

/// The data as the scene opens: X causes Y through a smooth step, with a spread that grows.
const GROWING: Setup = Setup {
    graph: Graph::Direct,
    seen: false,
    curve: Curve::Step,
    spread: Spread::Rising,
    noise: Noise::Normal,
    sigma: 0.4,
    level: 1.0,
    scale: 0.0,
    close: CLOSE,
    n: 200,
};

const TRIALS: [Trial; 5] = [
    Trial {
        button: "a spread that grows",
        caption: r"$X$ causes $Y$ through a smooth step, and the spread of $Y$ grows with $X$. With the spread fitted (two steps), what is left over divided by it is a level band on the left, independent of $X$ ($p = 0.59$), and not on the right ($p = 0.01$): $X \to Y$. Press level only: on the left the band becomes a funnel, narrow at the left and wide at the right, and the check fails both ways ($p = 0.01$ and $0.01$): fitting the level alone, no arrow fits. Over 100 new samples: level only, neither fits in all 100; two steps, $X \to Y$ in 99.",
        setup: GROWING,
        method: Method::TwoSteps,
        scaled: true,
        held: false,
    },
    Trial {
        button: "a hidden common cause",
        caption: r"Neither causes the other: a hidden $U$ sets $X$ (closely) and the level of $Y$ (through the smooth step). Fitting the level alone, the check fails with $X$ as the input ($p = 0.01$: the spread of $Y$ is widest in the middle, where the step is steep) and passes with $Y$ ($p = 0.62$): $Y \to X$, wrong, here and in 96 of 100 new samples. Fit the spread too, and both pass: both fit, in 77 of 100 with two steps ($Y \to X$ in 23).",
        setup: Setup { graph: Graph::Common, spread: Spread::Even, ..GROWING },
        method: Method::TwoSteps,
        scaled: false,
        held: false,
    },
    Trial {
        button: "measure the hidden cause",
        caption: r"The same, with $U$ measured (called $Z$: the dots take its colour) and put in every fit: $Y$ from $X$ and $Z$, $X$ from $Y$ and $Z$. Both checks pass ($p = 0.96$ and $0.87$): given $Z$, nothing is left for an arrow between $X$ and $Y$ to do, as it should be when neither causes the other. Over 100 new samples every way of fitting says both fit in 99 or 100, the level alone included: the wrong arrow is gone.",
        setup: Setup { graph: Graph::Common, seen: true, spread: Spread::Even, ..GROWING },
        method: Method::TwoSteps,
        scaled: true,
        held: false,
    },
    Trial {
        button: "a hidden cause the spread hides",
        caption: r"Now $U$ causes $X$ and $Y$, $X$ causes $Y$ as well, and $U$ also widens the spread of $Y$. Fitting the level alone, the check with $X$ as the input fails in 88 of 100 new samples: rightly, as no curve of $X$ plus independent noise makes these data. Fit the spread too, and it passes in 93 (two steps): the spread soaks up what $U$ does, and the verdict is $X \to Y$ in 78. The arrow is right, but the yellow level is not what setting $X$ would do (green): typically 0.48 away. A check that passes is not proof.",
        setup: Setup { graph: Graph::Confounded, spread: Spread::Even, level: 0.5, scale: 0.5, ..GROWING },
        method: Method::TwoSteps,
        scaled: true,
        held: false,
    },
    Trial {
        button: "peaked noise",
        caption: r"The growing spread again, with peaked noise (long tails). On the scoreboard the true level and spread pass the check in 96 of 100 new samples, as they should. Two steps pass in only 84 (against 100 with bell-shaped noise): its spread, fitted from squares, is pulled about by the far points (24 % off, against 19 %). Likelihood still finds $X \to Y$ in 96. Quantiles let the wrong direction pass in 79, as they did with bell-shaped noise (85): in these measurements no way of fitting wins everywhere.",
        setup: Setup { noise: Noise::Laplace, ..GROWING },
        method: Method::TwoSteps,
        scaled: true,
        held: false,
    },
];

/// What the counts showed for each way of fitting (each held by a test).
const FINDINGS: [&str; 7] = [
    r"Level only: wherever the spread varies, the check fails even with $X$ as the input (in all 100 with the growing spread). With an even spread it is right most often: $X \to Y$ in 97, against 84 for two steps, 87 for likelihood and 6 for quantiles. A spread fitted where none is needed lets the wrong direction pass more often.",
    r"Two steps: right in 99 at the opening settings. With peaked noise the true direction passes in only 84.",
    r"Reweighted: never clearly better than two steps; on the parabola its level was off by more than twice as much (0.96 against 0.42).",
    r"Likelihood: the smallest spread errors, except on the parabola, where it widened the spread instead of bending the level (level off 1.36, against 0.42 for two steps) and the true direction passed in only 52. Peaked noise did not mislead it (99).",
    r"Quantiles: the wrong direction passed in 85 at 200 points, and still in 55 at 300.",
    r"Few points: at 50, every way of fitting a spread let both directions pass in 85 or more.",
    r"The other half: what is left over from points the fit has not seen passed the true direction less often (two steps 77 instead of 100; 47 instead of 98 with the spread pinched in the middle). It failed the wrong direction more often only with quantiles (passing in 72 instead of 85). From the fitted points, the true direction passed about as often as the true noise does (100 against 97).",
];

/// The scene opens as the first experiment sets it up (without the count).
const OPENING: usize = 0;
/// The seed of the sample every experiment shows.
const FIRST_SEED: u64 = 4;

/// Pieces of work a second: on the sample shown (its fits, and its checks one shuffle
/// at a time, in about half a second), and on the scoreboard (as fast as the frame
/// allows: in a window the time allowance stops it long before this).
const SHOWN_RATE: f32 = 600.0;
const BOARD_RATE: f32 = 6000.0;
/// How many shuffles a strip waits for before it is drawn.
const FIRST_LOOK: usize = 12;

/// The names of the curves, spreads and graphs, in the order of the choices.
const CURVES: [(Curve, &str); 4] = [(Curve::Line, "straight"), (Curve::Bend, "gentle bend"), (Curve::Step, "smooth step"), (Curve::Parabola, "parabola")];
const SPREADS: [(Spread, &str); 4] = [(Spread::Even, "even"), (Spread::Rising, "growing"), (Spread::Pinched, "pinched in the middle"), (Spread::Bulge, "wide in the middle")];
const GRAPHS: [Graph; 3] = [Graph::Direct, Graph::Common, Graph::Confounded];

impl Method {
    fn name(self) -> &'static str {
        match self {
            Method::Level => "level only",
            Method::TwoSteps => "two steps",
            Method::Reweighted => "reweighted",
            Method::Likelihood => "likelihood",
            Method::Quantiles => "quantiles",
        }
    }

    fn index(self) -> usize {
        METHODS.iter().position(|m| *m == self).unwrap_or(0)
    }
}

/// The mean of `f` over a bell of spread 1 (to the precision of a fine sum).
fn bell_mean(f: impl Fn(f64) -> f64) -> f64 {
    let (steps, half) = (480, 7.0);
    let h = 2.0 * half / steps as f64;
    (0..steps)
        .map(|i| {
            let u = -half + (i as f64 + 0.5) * h;
            f(u) * (-0.5 * u * u).exp()
        })
        .sum::<f64>()
        * h
        / std::f64::consts::TAU.sqrt()
}

// ----- the scene -----------------------------------------------------------------------------------------

pub struct LocScale {
    setup: Setup,
    seed: u64,
    /// The way of fitting level and spread together that is shown.
    method: Method,
    /// Level and spread fitted (or the level alone).
    scaled: bool,
    held: bool,
    /// On a phone: the fit (0) or the scoreboard (1).
    tab: usize,
    /// Count over new samples (asked for once, it counts again after every change).
    counting: bool,
    draws: Draws,
    shown: Shown,
    board: Board,
    pace: Pace,
    board_pace: Pace,
    look: Look,
    canvas: Option<Vec2>,
    strip_goal: [[f32; 2]; 2],
    marked: [bool; 2],
    aimed: [bool; 2],
    /// The slot the strips were last aimed for.
    strip_slot: usize,
    ticks: [[Ticks; 3]; 2],
    drawn_last: (usize, usize),
}

impl LocScale {
    /// (Nothing here needs the graphics device, so the tests can make one too.)
    fn create() -> LocScale {
        let t = &TRIALS[OPENING];
        let key = Key { setup: t.setup, seed: FIRST_SEED, held: t.held };
        let draws = Draws::new(key.seed, key.setup.n);
        let mut shown = Shown::new(key, &draws);
        // (the fits of the opening sample at once, so that the first frame has them)
        let wanted = [t.method.index(), 0];
        while !wanted.iter().all(|&m| shown.fitted(m)) {
            shown.step(&wanted);
        }
        LocScale {
            setup: t.setup,
            seed: FIRST_SEED,
            method: t.method,
            scaled: t.scaled,
            held: t.held,
            tab: 0,
            counting: false,
            draws,
            shown,
            board: Board::new(),
            pace: Pace::default(),
            board_pace: Pace::default(),
            look: Look::default(),
            canvas: None,
            strip_goal: [[0.0; 2]; 2],
            marked: [false; 2],
            aimed: [false; 2],
            strip_slot: 0,
            ticks: [[Ticks::NONE; 3]; 2],
            drawn_last: (0, 0),
        }
    }

    /// Set up one of the experiments, and count every method over new samples.
    fn set_up(&mut self, k: usize) {
        let t = &TRIALS[k];
        self.setup = t.setup;
        self.method = t.method;
        self.scaled = t.scaled;
        self.held = t.held;
        self.seed = FIRST_SEED;
        self.counting = true;
        self.refresh();
    }

    fn key(&self) -> Key {
        Key { setup: self.setup, seed: self.seed, held: self.held }
    }

    /// The slot shown: the method's, or the level alone's.
    fn slot(&self) -> usize {
        if self.scaled {
            self.method.index()
        } else {
            0
        }
    }

    /// The slots worked on: the one shown, then the other of the switch.
    fn wanted(&self) -> [usize; 2] {
        let m = self.method.index();
        if self.scaled {
            [m, 0]
        } else {
            [0, m]
        }
    }

    /// Make the data again if what they depend on has changed, and start their
    /// analysis (and the count, if it is kept) afresh.
    fn refresh(&mut self) {
        let s = &mut self.setup;
        s.n = s.n.clamp(20, 300);
        s.sigma = s.sigma.max(0.0).min(1.5);
        s.level = s.level.max(0.0).min(1.5);
        s.scale = s.scale.max(0.0).min(0.8);
        s.close = s.close.max(0.0).min(1.0);
        if self.method == Method::Level {
            self.method = Method::TwoSteps;
        }
        if self.draws.key != (self.seed, self.setup.n) {
            self.draws = Draws::new(self.seed, self.setup.n);
        }
        let key = self.key();
        if self.shown.key != key {
            self.shown = Shown::new(key, &self.draws);
            self.pace.reset();
        }
        if self.counting && self.board.key != Some(key) {
            self.board.restart(key);
            self.board_pace.reset();
        }
    }

    /// The analysis of the sample shown, then the count, paced (when the scene's time
    /// stands still, by the clock: the work answers the visitor's hand).
    fn work(&mut self, dt: f32, real_dt: f32) {
        let dt = if dt > 0.0 { dt } else { real_dt };
        let wanted = self.wanted();
        if self.shown.pending(&wanted) {
            let mut turn = self.pace.begin(SHOWN_RATE, dt, real_dt, 0.004);
            while self.shown.pending(&wanted) && turn.next() {
                self.shown.step(&wanted);
            }
            self.pace.end(turn);
        } else if self.counting && self.board.pending() {
            let mut turn = self.board_pace.begin(BOARD_RATE, dt, real_dt, 0.004);
            while self.board.pending() && turn.next() {
                self.board.step();
            }
            self.board_pace.end(turn);
        }
    }

    /// The analysis of the sample shown to the end at once (the tests).
    #[cfg(test)]
    fn settle(&mut self) {
        self.refresh();
        let wanted = self.wanted();
        while self.shown.pending(&wanted) {
            self.shown.step(&wanted);
        }
    }

    /// The third variable is in the fits.
    fn seen(&self) -> bool {
        self.setup.seen && self.setup.graph.third()
    }

    /// The analysis drawn in column `c`: the shown slot's, or while that is being made,
    /// the other slot's.
    fn analysis(&self, c: usize) -> Option<&Analysis> {
        let [a, b] = self.wanted();
        self.shown.slots[c].get(a).and_then(|s| s.result.as_ref()).or_else(|| self.shown.slots[c].get(b).and_then(|s| s.result.as_ref()))
    }

    /// The level and spread the truth gives Y at input `x` along curve `k` (of the
    /// third variable's levels when it is measured); with a hidden third variable,
    /// the level that setting X gives, and no spread.
    fn truth_at(&self, x: f64, k: usize) -> (f64, f64) {
        let s = &self.setup;
        if !s.graph.third() {
            return s.truth(x, 0.0);
        }
        if self.seen() {
            return s.truth(x, self.shown.cols[0].levels[k.min(2)]);
        }
        let f = if s.graph.arrow() { s.curve.at(x) } else { 0.0 };
        (f + s.level * bell_mean(|u| s.curve.at(u)), 0.0)
    }

    /// What the picture glides to, on this layout (nothing while the fits it needs
    /// are being made: the picture then waits where it is).
    fn goals(&self, lay: &Layout) -> Option<Goals> {
        let s = &self.shown;
        let analyses = [self.analysis(0)?, self.analysis(1)?];
        let n = s.sample.x.len();
        let scaled = self.scaled && (0..2).all(|c| s.slots[c].get(self.slot()).is_some_and(|sl| sl.result.is_some()));
        let mut points = Vec::with_capacity(2 * n);
        let mut resid = Vec::with_capacity(2 * n);
        for i in 0..n {
            points.push(s.sample.x[i] as f32);
            points.push(s.sample.y[i] as f32);
            for a in &analyses {
                let r = if scaled { a.z.get(i) } else { a.raw.get(i) };
                resid.push(r.copied().unwrap_or(0.0) as f32);
            }
        }
        let reach = |v: &[f64]| {
            let (lo, hi) = extent(v);
            let pad = ((hi - lo) * 0.06).max(0.05);
            [(lo - pad) as f32, (hi + pad) as f32]
        };
        // (what is left over reaches to the furthest point, but no further than two and a
        // half times the 90th percentile of its size: the few points beyond, which a
        // spread from quartiles can leave, wait at the edge, and the band stays readable)
        let deep = |c: usize| {
            let mut sizes: Vec<f64> = (0..n).map(|i| resid[2 * i + c].abs() as f64).collect();
            sizes.sort_by(f64::total_cmp);
            let most = sizes.last().copied().unwrap_or(0.0).min(2.5 * percentile(&sizes, 0.9)) as f32;
            let most = if most > 1e-9 && most.is_finite() { most * 1.12 } else { 1.0 };
            [-most, most]
        };
        let ranges = [reach(&s.sample.x), reach(&s.sample.y), deep(0), deep(1)].concat();
        // three curves along each column's input (one, three times over, without a third variable)
        let curves = [0, 1].map(|c| {
            let col = &s.cols[c];
            let a = analyses[c];
            let g = col.across.len();
            let mut v = Vec::with_capacity(3 * g * 3);
            for k in 0..3 {
                let block = if a.curves.level.len() >= 3 * g { k * g } else { 0 };
                for j in 0..g {
                    let (l, sp) = (a.curves.level.get(block + j).copied().unwrap_or(0.0), a.curves.spread.get(block + j).copied().unwrap_or(0.0));
                    v.extend([col.across[j] as f32, l as f32, sp as f32]);
                }
            }
            v
        });
        let truth = {
            let col = &s.cols[0];
            let mut v = Vec::with_capacity(3 * col.across.len() * 3);
            for k in 0..3 {
                for &x in &col.across {
                    let (l, sp) = self.truth_at(x, k);
                    v.extend([x as f32, l as f32, sp as f32]);
                }
            }
            v
        };
        let seen = self.seen();
        let band_known = !self.setup.graph.third() || seen;
        let misc = vec![scaled as u8 as f32, seen as u8 as f32, band_known as u8 as f32];
        let slices = [0, 1].map(|c| {
            let input = if c == 0 { &s.sample.x } else { &s.sample.y };
            let r: Vec<f64> = (0..n).map(|i| resid[2 * i + c] as f64).collect();
            slices(input, &r, slice_count(n, lay.plot_width()))
        });
        let strips = self.strip_goal.concat();
        let b = &self.board;
        let planned = RUNS as f32;
        let bars = (0..ROWS).flat_map(|r| {
            let row = &b.rows[r];
            [row.verdicts[0], row.verdicts[1], row.verdicts[2], row.verdicts[3], row.pass[0], row.pass[1]].map(|v| v as f32 / planned)
        });
        Some(Goals { points, resid, ranges, curves, truth, misc, slices, strips, bars: bars.collect() })
    }

    /// Is strip `c` drawn: has the real pairing been measured, and enough shuffles to
    /// say how far the strip should reach?
    fn strip_ready(&self, c: usize) -> bool {
        self.shown.slots[c].get(self.slot()).is_some_and(|s| s.side.l.is_some() && (s.side.done() || s.side.shuffles.len() >= FIRST_LOOK))
    }

    /// How far each strip reaches, and where its real pairing is (as in `direction`:
    /// from the first dozen shuffles, half as far again as the furthest; once all are
    /// in, a little beyond the furthest of all).
    fn aim_strips(&mut self) {
        let slot = self.slot();
        if slot != self.strip_slot {
            self.aimed = [false; 2];
            self.strip_slot = slot;
        }
        for c in 0..2 {
            if !self.strip_ready(c) {
                self.aimed[c] = false;
                continue;
            }
            let Some(s) = self.shown.slots[c].get(slot).map(|s| &s.side) else { continue };
            let furthest = s.shuffles.iter().copied().fold(s.value, f64::max).max(1e-12);
            if s.done() {
                self.strip_goal[c] = [(furthest * 1.15).ln() as f32, s.value as f32];
            } else if !self.aimed[c] {
                self.strip_goal[c] = [(furthest * 1.5).ln() as f32, s.value as f32];
                self.aimed[c] = true;
            }
        }
    }
}

// ----- gliding ----------------------------------------------------------------------------------------

/// How long a glide takes: the time in which each of its two eases covers about two
/// thirds of what is left (seconds).
const FOLLOW: f32 = 0.075;
/// Closer than this (in the units of what glides), it is there.
const THERE: f32 = 2e-4;

/// Numbers drawn on their way to what they are: two eases one after the other (so
/// that a move sets off gently), and exactly there in the end (as in `direction`).
#[derive(Default)]
struct Glide {
    between: Vec<f32>,
    shown: Vec<f32>,
}

impl Glide {
    /// Bring the numbers `dt` seconds nearer to `to`. With `keep`, a list that has
    /// grown or shrunk keeps what it had in common; otherwise it starts where it is
    /// going. Says whether anything is still on its way.
    fn follow(&mut self, to: &[f32], dt: f32, keep: bool) -> bool {
        if self.shown.len() != to.len() {
            if keep && !self.shown.is_empty() {
                let n = self.shown.len().min(to.len());
                self.shown.truncate(n);
                self.between.truncate(n);
                self.shown.extend_from_slice(&to[n..]);
                self.between.extend_from_slice(&to[n..]);
            } else {
                self.shown = to.to_vec();
                self.between = to.to_vec();
            }
        }
        if self.shown == to {
            return false;
        }
        let k = 1.0 - (-dt.max(0.0) / FOLLOW).exp();
        let mut apart = 0.0f32;
        for ((b, s), t) in self.between.iter_mut().zip(self.shown.iter_mut()).zip(to) {
            *b += (t - *b) * k;
            *s += (*b - *s) * k;
            apart = apart.max((t - *s).abs()).max((t - *b).abs());
        }
        // (exactly there in the end, not nearly: the picture then stops changing)
        if apart.is_nan() || apart <= THERE {
            self.shown = to.to_vec();
            self.between = to.to_vec();
            return false;
        }
        true
    }

    fn snap(&mut self, to: &[f32]) {
        self.shown = to.to_vec();
        self.between = to.to_vec();
    }
}

/// Everything that is drawn gliding.
#[derive(Default)]
struct Look {
    /// The samples, (X, Y) pairs.
    points: Glide,
    /// What is left over in each column, by point: column 0's, column 1's.
    resid: Glide,
    /// X's range, Y's, and what is left over in each column.
    ranges: Glide,
    /// Each column's three curves: (input, level, spread) at each place.
    curves: [Glide; 2],
    /// The truth in the first column, likewise.
    truth: Glide,
    /// How much shows of: the fitted spread, the third variable's colours, the true spread.
    misc: Glide,
    slices: [Glide; 2],
    strips: Glide,
    bars: Glide,
    moving: bool,
}

struct Goals {
    points: Vec<f32>,
    resid: Vec<f32>,
    ranges: Vec<f32>,
    curves: [Vec<f32>; 2],
    truth: Vec<f32>,
    misc: Vec<f32>,
    slices: [Vec<f32>; 2],
    strips: Vec<f32>,
    bars: Vec<f32>,
}

impl Look {
    fn follow(&mut self, g: &Goals, dt: f32, snap: bool) {
        if snap {
            self.points.snap(&g.points);
            self.resid.snap(&g.resid);
            self.ranges.snap(&g.ranges);
            for c in 0..2 {
                self.curves[c].snap(&g.curves[c]);
                self.slices[c].snap(&g.slices[c]);
            }
            self.truth.snap(&g.truth);
            self.misc.snap(&g.misc);
            self.strips.snap(&g.strips);
            self.bars.snap(&g.bars);
            self.moving = false;
            return;
        }
        let mut moving = self.points.follow(&g.points, dt, true);
        moving |= self.resid.follow(&g.resid, dt, true);
        moving |= self.ranges.follow(&g.ranges, dt, false);
        for c in 0..2 {
            moving |= self.curves[c].follow(&g.curves[c], dt, false);
            moving |= self.slices[c].follow(&g.slices[c], dt, false);
        }
        moving |= self.truth.follow(&g.truth, dt, false);
        moving |= self.misc.follow(&g.misc, dt, false);
        moving |= self.strips.follow(&g.strips, dt, false);
        moving |= self.bars.follow(&g.bars, dt, false);
        self.moving = moving;
    }
}

// ----- slices -----------------------------------------------------------------------------------------

/// What is left over in one slice of the input: where the slice starts and ends, and
/// the residuals' 10th, 25th, 50th, 75th and 90th percentiles in it.
const PER_SLICE: usize = 7;
const MARKS: [f64; 5] = [0.1, 0.25, 0.5, 0.75, 0.9];

/// The input cut into `k` slices with as many points in each, and what is left over
/// in each: `PER_SLICE` numbers a slice, and then the same five percentiles for all of
/// it together.
fn slices(input: &[f64], residuals: &[f64], k: usize) -> Vec<f32> {
    let n = input.len().min(residuals.len());
    let k = k.max(1);
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&a, &b| input[a].total_cmp(&input[b]));
    let mut out = Vec::with_capacity(k * PER_SLICE + MARKS.len());
    for s in 0..k {
        let (from, to) = (s * n / k, ((s + 1) * n / k).max(s * n / k + 1).min(n));
        let part = order.get(from..to).unwrap_or(&[]);
        let (lo, hi) = match (part.first(), part.last()) {
            (Some(&a), Some(&b)) => (input[a], input[b]),
            _ => (0.0, 0.0),
        };
        // (each slice reaches halfway to the next, so that together they cover the cloud)
        let start = if s == 0 { lo } else { 0.5 * (lo + order.get(from.wrapping_sub(1)).map_or(lo, |&i| input[i])) };
        let end = if s + 1 == k { hi } else { 0.5 * (hi + order.get(to).map_or(hi, |&i| input[i])) };
        let mut r: Vec<f64> = part.iter().map(|&i| residuals[i]).collect();
        r.sort_by(f64::total_cmp);
        out.push(start as f32);
        out.push(end as f32);
        out.extend(MARKS.iter().map(|&q| percentile(&r, q) as f32));
    }
    let mut all: Vec<f64> = residuals[..n].to_vec();
    all.sort_by(f64::total_cmp);
    out.extend(MARKS.iter().map(|&q| percentile(&all, q) as f32));
    out
}

/// How many slices: about thirty points in each, and none narrower on the screen than
/// a finger's width on average.
fn slice_count(n: usize, width: f32) -> usize {
    ((n as f32 / 30.0).round() as usize).clamp(2, 7).min((width / 34.0).max(2.0) as usize)
}

// ----- the layout ------------------------------------------------------------------------------------

/// A rectangle on the screen (logical pixels): top-left, bottom-right.
type Rect = (Vec2, Vec2);

const NO_RECT: Rect = (Vec2::ZERO, Vec2::ZERO);

/// The words on the buttons over the picture.
const SWITCH: [&str; 2] = ["level only", "level and spread"];
const FROM: [&str; 2] = ["the fitted points", "the other half"];
const TABS: [&str; 2] = ["the fit", "the scoreboard"];
const FITTED: [Method; 4] = [Method::TwoSteps, Method::Reweighted, Method::Likelihood, Method::Quantiles];

/// Where everything goes on a canvas.
#[derive(Clone, Copy, Debug, PartialEq)]
struct Layout {
    view: Vec2,
    narrow: bool,
    /// Text sizes: headings, words, small print.
    big: f32,
    text: f32,
    small: f32,
    /// On a narrow canvas, the tabs of its two views; and which parts are drawn.
    tabs: Option<[Rect; 2]>,
    fit_view: bool,
    board_view: bool,
    /// The truth's graph, the buttons, the verdict and its reasons.
    truth: Rect,
    switch: [Rect; 2],
    methods: [Rect; 4],
    from: Option<[Rect; 2]>,
    /// Where the names of the rows of buttons go (a wide canvas only).
    labels: [Vec2; 3],
    verdict: Rect,
    words: Rect,
    /// Each column's heading (top-left corner) and its three pictures.
    head: [Vec2; 2],
    scatter: [Rect; 2],
    resid: [Rect; 2],
    strip: [Rect; 2],
    board: Rect,
    /// The radius of the variables' discs in the small graphs.
    node: f32,
    /// Too small a canvas for the pictures: the verdict alone.
    compact: bool,
}

/// A row of buttons from `x` at height `y`, each as wide as its words.
fn button_row(words: &[&str], x: f32, y: f32, h: f32, size: f32, gap: f32) -> Vec<Rect> {
    let mut at = x;
    words
        .iter()
        .map(|w| {
            let width = text_width(w, size) + 18.0;
            let r = (Vec2::new(at, y), Vec2::new(at + width, y + h));
            at += width + gap;
            r
        })
        .collect()
}

/// A row of `k` buttons sharing the width from `x0` to `x1` evenly.
fn even_row(k: usize, x0: f32, x1: f32, y: f32, h: f32, gap: f32) -> Vec<Rect> {
    let w = ((x1 - x0 - gap * (k as f32 - 1.0)) / k as f32).max(4.0);
    (0..k).map(|i| (Vec2::new(x0 + i as f32 * (w + gap), y), Vec2::new(x0 + i as f32 * (w + gap) + w, y + h))).collect()
}

impl Layout {
    fn of(view: Vec2, inset: f32, tab: usize) -> Layout {
        let view = view.max(Vec2::splat(1.0));
        let narrow = view.x < 960.0;
        let (big, text, small) = if narrow { (13.0, 11.5, 10.5) } else { (15.0, 12.5, 11.5) };
        let side = if narrow { 8.0 } else { 20.0 };
        let gap = if narrow { 12.0 } else { 46.0 };
        let numbers = if narrow { 24.0 } else { 34.0 };
        let top = inset.max(0.0) + if narrow { 6.0 } else { 12.0 };
        let node = if narrow { 12.0 } else { 15.0 };
        let width = ((view.x - 2.0 * side - gap) * 0.5).max(10.0);
        let head = big + small + 14.0;
        let line = small + 9.0;
        let strip_h = if narrow { 36.0 } else { 46.0 };
        let fit_view = !narrow || tab == 0;
        let board_view = !narrow || tab == 1;
        let to2 = |v: Vec<Rect>| [v.first().copied().unwrap_or(NO_RECT), v.get(1).copied().unwrap_or(NO_RECT)];
        let to4 = |v: Vec<Rect>| [0, 1, 2, 3].map(|i| v.get(i).copied().unwrap_or(NO_RECT));
        let (tabs, truth, switch, methods, from, labels, verdict, words, board, columns_top, columns_end);
        if narrow {
            let row = 24.0;
            tabs = Some(to2(button_row(&TABS, side, top, row, text, 6.0)));
            switch = to2(even_row(2, side, view.x - side, top + row + 8.0, row, 6.0));
            methods = to4(even_row(4, side, view.x - side, top + 2.0 * row + 14.0, row - 2.0, 4.0));
            from = None;
            labels = [Vec2::ZERO; 3];
            columns_top = top + 3.0 * row + 24.0;
            // the verdict and the truth side by side under the columns, the reasons under them
            let graphs = text + 2.0 * node + 14.0;
            let band = graphs + 4.0 + 3.0 * (small + 4.0) + 8.0;
            let band_top = view.y - band - 4.0;
            let mid = view.x * 0.5;
            verdict = (Vec2::new(side, band_top), Vec2::new(mid - 6.0, band_top + graphs));
            truth = if fit_view { (Vec2::new(mid + 6.0, band_top), Vec2::new(view.x - side, band_top + graphs)) } else { (Vec2::new(side, top + row + 10.0), Vec2::new(view.x - side, top + row + 10.0 + graphs + 14.0)) };
            words = (Vec2::new(side, band_top + graphs + 4.0), Vec2::new(view.x - side, view.y - 6.0));
            columns_end = band_top - 12.0;
            board = (Vec2::new(side, truth.1.y + 16.0), Vec2::new(view.x - side, view.y - 8.0));
        } else {
            let row = 22.0;
            let bar = 3.0 * row + 2.0 * 7.0 + 22.0;
            let method_words: Vec<&str> = FITTED.iter().map(|m| m.name()).collect();
            // the truth, the buttons and the verdict share the strip: on a canvas a little
            // too narrow for all three at their full widths, each gives some way
            let end = |rects: Vec<Rect>| rects.last().map_or(0.0, |r| r.1.x);
            let buttons_w = end(button_row(&SWITCH, 0.0, 0.0, row, small, 6.0)).max(end(button_row(&method_words, 0.0, 0.0, row, small, 6.0))).max(end(button_row(&FROM, 0.0, 0.0, row, small, 6.0)));
            let short = (2.0 * side + 210.0 + 28.0 + 104.0 + buttons_w + 18.0 + 330.0 - view.x).clamp(0.0, 70.0);
            let truth_w = 210.0 - 0.2 * short;
            let verdict_w = 330.0 - 0.57 * short;
            truth = (Vec2::new(side, top), Vec2::new(side + truth_w, top + bar));
            verdict = (Vec2::new(view.x - side - verdict_w, top), Vec2::new(view.x - side, top + 2.0 * node + text + 14.0));
            words = (Vec2::new(verdict.0.x, verdict.1.y + 2.0), Vec2::new(view.x - side, top + bar + 4.0));
            let x0 = side + truth_w + 28.0;
            let names = x0 + 104.0 - 0.23 * short;
            labels = [0.0, 1.0, 2.0].map(|r| Vec2::new(x0, top + r * (row + 7.0) + row * 0.5));
            switch = to2(button_row(&SWITCH, names, top, row, small, 6.0));
            methods = to4(button_row(&method_words, names, top + row + 7.0, row, small, 6.0));
            from = Some(to2(button_row(&FROM, names, top + 2.0 * (row + 7.0), row, small, 6.0)));
            tabs = None;
            columns_top = top + bar + 18.0;
            let board_h = (view.y * 0.2).clamp(150.0, 190.0);
            board = (Vec2::new(side, view.y - 8.0 - board_h), Vec2::new(view.x - side, view.y - 8.0));
            columns_end = board.0.y - 22.0;
        }
        // what is left for the two pictures of each column
        let rest = (columns_end - columns_top - head - strip_h - 5.0 * line - 2.0 * 8.0 - (small + 6.0)).max(20.0);
        let (scatter_h, resid_h) = (rest * 0.56, rest * 0.44);
        let compact = resid_h < 46.0 || width < 110.0;
        let mut heads = [Vec2::ZERO; 2];
        let (mut scatter, mut resid, mut strip) = ([NO_RECT; 2], [NO_RECT; 2], [NO_RECT; 2]);
        for c in 0..2 {
            let x0 = side + c as f32 * (width + gap);
            let (l, r) = (x0 + numbers, x0 + width);
            heads[c] = Vec2::new(x0, columns_top);
            let y = columns_top + head;
            scatter[c] = (Vec2::new(l, y), Vec2::new(r, y + scatter_h));
            let y = y + scatter_h + line + line;
            resid[c] = (Vec2::new(l, y), Vec2::new(r, y + resid_h));
            let y = y + resid_h + line + line + 8.0;
            strip[c] = (Vec2::new(l, y), Vec2::new(r, y + strip_h));
        }
        Layout { view, narrow, big, text, small, tabs, fit_view, board_view, truth, switch, methods, from, labels, verdict, words, head: heads, scatter, resid, strip, board, node, compact }
    }

    fn plot_width(&self) -> f32 {
        self.scatter[0].1.x - self.scatter[0].0.x
    }
}

/// From a picture's numbers to the screen: `rect` shows `across` (lo, hi) along and
/// `up` (lo, hi) up.
#[derive(Clone, Copy)]
struct Map {
    rect: Rect,
    across: [f32; 2],
    up: [f32; 2],
}

impl Map {
    fn to(&self, a: f32, b: f32) -> Vec2 {
        let (min, max) = self.rect;
        let t = (a - self.across[0]) / (self.across[1] - self.across[0]).max(1e-9);
        let s = (b - self.up[0]) / (self.up[1] - self.up[0]).max(1e-9);
        Vec2::new(min.x + t * (max.x - min.x), max.y - s * (max.y - min.y))
    }
}

/// Is the point inside the rectangle (with a margin)?
fn within(p: Vec2, (min, max): Rect, margin: f32) -> bool {
    p.x >= min.x - margin && p.x <= max.x + margin && p.y >= min.y - margin && p.y <= max.y + margin
}

/// The first of `options` that fits in `room` logical pixels at `size` (the last if none does).
fn pick(f: &Frame, options: &[String], room: f32, size: f32) -> String {
    options.iter().find(|t| f.text_size(t, size).x <= room).or(options.last()).cloned().unwrap_or_default()
}

/// The words of a text, each with its formulas whole.
fn words_of(text: &str) -> Vec<&str> {
    let (mut words, mut start, mut inside) = (Vec::new(), 0, false);
    let bytes = text.as_bytes();
    let mut i = 0;
    while i < bytes.len() {
        match bytes[i] {
            b'\\' => i += 1,
            b'$' => inside = !inside,
            b' ' if !inside => {
                words.push(&text[start..i]);
                start = i + 1;
            }
            _ => {}
        }
        i += 1;
    }
    words.push(&text[start.min(text.len())..]);
    words
}

/// Text broken into lines no wider than `width` logical pixels, from `at` down (screen
/// space), no lower than `bottom`. Returns the height taken.
fn wrap(f: &mut Frame, at: Vec2, width: f32, bottom: f32, text: &str, size: f32, color: Color) -> f32 {
    let mut lines: Vec<String> = vec![String::new()];
    for word in words_of(text) {
        let Some(last) = lines.last_mut() else { break };
        if last.is_empty() {
            last.push_str(word);
            continue;
        }
        let longer = format!("{last} {word}");
        if f.text_size(&longer, size).x > width {
            lines.push(word.to_string());
        } else {
            *last = longer;
        }
    }
    let step = size + 4.0;
    let mut used = 0.0;
    for line in &lines {
        if at.y + used + step > bottom + 2.0 {
            break;
        }
        f.text_px(at + Vec2::new(0.0, used), line.as_str(), size, color, Align2::LEFT_TOP);
        used += step;
    }
    used
}

/// A button over the picture, lit when `on`: true when it is clicked.
fn button(f: &mut Frame, (min, max): Rect, text: &str, size: f32, pointer: Option<Vec2>, on: bool) -> bool {
    if max.x - min.x < 2.0 {
        return false;
    }
    let over = pointer.is_some_and(|p| within(p, (min, max), 0.0));
    let fill_color = match (on, over) {
        (true, _) => Color::hex(0x2B3D66),
        (false, true) => Color::hex(0x2A3350),
        _ => Color::hex(0x161C2B),
    };
    hud_rect(f, min, max, fill_color);
    let edge = [min, Vec2::new(max.x, min.y), max, Vec2::new(min.x, max.y), min];
    path_of(f, BUTTONS, &edge, if on { 1.4 } else { 1.0 }, pal::LINK.a(if on { 0.95 } else if over { 0.7 } else { 0.35 }));
    f.text_px((min + max) * 0.5, text, size, if on { pal::WHITE } else { pal::TEXT.a(0.85) }, Align2::CENTER_CENTER);
    if over {
        f.handles.claim_clicks();
        f.handles.capture();
    }
    over && f.ptr.clicked
}

/// The names of the two variables in a formula, each in the colour of its part in a
/// column: the cause it supposes across (`input`), the effect up.
fn names(c: usize) -> (String, String) {
    let (input, output) = if c == 0 { ("X", "Y") } else { ("Y", "X") };
    (tint(CAUSE, input), tint(EFFECT, output))
}

/// The third variable's colour, from its lowest values (0) to its highest (1).
fn third_color(t: f32) -> Color {
    Color::hex(0x6B4FD0).mix(Color::hex(0xF4E6FF), t.max(0.0).min(1.0))
}

/// The round values along one axis that get a grid line and a number (as in
/// `direction`: chosen from where the axis is going, and kept while it glides there).
#[derive(Clone, Copy, Debug, PartialEq)]
struct Ticks {
    step: f32,
    first: i64,
    last: i64,
}

impl Ticks {
    const NONE: Ticks = Ticks { step: 1.0, first: 0, last: -1 };

    /// For an axis going from `lo` to `hi` over `pixels`: a round step about sixty
    /// pixels long.
    fn of(lo: f32, hi: f32, pixels: f32) -> Ticks {
        let span = hi - lo;
        if !span.is_finite() || span <= 0.0 || pixels.is_nan() || pixels <= 0.0 {
            return Ticks::NONE;
        }
        let step = nice_step((hi - lo) / (pixels / 60.0).max(1.0));
        let first = (lo / step).floor() as i64 - 3;
        let last = ((hi / step).ceil() as i64 + 3).min(first + 60);
        Ticks { step, first, last }
    }

    fn values(&self) -> impl Iterator<Item = f32> + '_ {
        (self.first..=self.last).map(|k| k as f32 * self.step)
    }
}

/// Grey: a verdict that does not pick a side; darker: one that says neither fits.
const UNDECIDED: Color = pal::DIM;
const NEITHER: Color = Color::hex(0x5B6479);
/// The samples, the fitted level and spread, what is left over slice by slice, and
/// the dependence of shuffled pairings and of the real one.
const DOT: Color = pal::TEXT;
const FIT: Color = pal::YELLOW;
const SLICE: Color = pal::BLUE;
const SHUFFLED: Color = pal::DIM;
const PAIRED: Color = pal::WHITE;

/// The kinds of line drawn in batches of their own (see [`apart`]).
const GRID_LINES: u8 = 0;
const FITTED_LINES: u8 = 1;
const SLICES: u8 = 2;
const STRIPS: u8 = 3;
const BUTTONS: u8 = 4;
const LEGEND: u8 = 5;
const DASHED: u8 = 6;
const TRUE_CURVE: u8 = 7;

/// A batch of lines on top of everything for one kind of line alone. `eidos motion`
/// follows a batch's points by their places in it, so a kind whose number of points
/// changes from frame to frame is kept apart from the others, whose points then keep
/// their places. (The fading is for trails: in a line batch it only tells one batch
/// from another.)
fn apart<'b>(f: &'b mut Frame<'_>, kind: u8, dash: bool) -> &'b mut crate::gfx::draw::LineBatch {
    let style = if dash { LineStyle { dash: 7.0, duty: 0.55, depth_bias: 0.0, fade_power: 30.0 + kind as f32, ..Default::default() } } else { LineStyle { depth_bias: 0.0, fade_power: 30.0 + kind as f32, ..Default::default() } };
    f.dl.lines_with(LineKind::Overlay, style)
}

/// A line in screen space, in the batch of its kind.
fn line_of(f: &mut Frame, kind: u8, a: Vec2, b: Vec2, width: f32, color: Color) {
    let (a, b) = (f.hud(a), f.hud(b));
    apart(f, kind, false).segment(a, b, width, color);
}

/// A polyline in screen space, in the batch of its kind.
fn path_of(f: &mut Frame, kind: u8, pts: &[Vec2], width: f32, color: Color) {
    let points: Vec<Vec3> = pts.iter().map(|p| f.hud(*p)).collect();
    apart(f, kind, false).polyline(points, width, color);
}

/// A translucent rectangle in screen space, over a panel's backing.
fn fill(f: &mut Frame, min: Vec2, max: Vec2, color: Color) {
    let c = [min, Vec2::new(max.x, min.y), max, Vec2::new(min.x, max.y)].map(|p| f.hud_at(1, p));
    f.dl.tris_alpha.quad(c[0], c[1], c[2], c[3], color);
}

/// A polyline kept inside a rectangle: a point outside it next to one inside is
/// brought back along the segment between them to the edge, and one with no
/// neighbour inside is left out (the line has a gap there).
fn clip(pts: &[Vec2], (min, max): Rect) -> Vec<Vec2> {
    let inside = |p: Vec2| p.is_finite() && p.x >= min.x - 0.5 && p.x <= max.x + 0.5 && p.y >= min.y - 0.5 && p.y <= max.y + 0.5;
    let edge = |a: Vec2, b: Vec2| {
        let d = b - a;
        let mut t = 1.0f32;
        if d.x != 0.0 {
            t = t.min(((if d.x > 0.0 { max.x } else { min.x }) - a.x) / d.x);
        }
        if d.y != 0.0 {
            t = t.min(((if d.y > 0.0 { max.y } else { min.y }) - a.y) / d.y);
        }
        a + d * t.max(0.0)
    };
    (0..pts.len())
        .map(|i| {
            let p = pts[i];
            if inside(p) || !p.is_finite() {
                return p;
            }
            let prev = i.checked_sub(1).map(|j| pts[j]).filter(|q| inside(*q));
            let next = pts.get(i + 1).copied().filter(|q| inside(*q));
            match (prev, next) {
                (Some(a), _) | (None, Some(a)) => edge(a, p),
                _ => Vec2::NAN,
            }
        })
        .collect()
}

/// A dashed polyline in screen space, kept inside a rectangle, in the batch of its kind.
fn dashed_px(f: &mut Frame, kind: u8, pts: &[Vec2], width: f32, color: Color, rect: Rect) {
    let pts = clip(pts, rect);
    let points: Vec<Vec3> = pts.iter().map(|p| f.hud(*p)).collect();
    apart(f, kind, true).polyline(points, width, color);
}

impl Scene for LocScale {
    fn view(&self) -> View {
        View::Flat { center: Vec2::ZERO, half_height: 1.0 }
    }

    fn settings(&self, s: &mut ViewSettings) {
        s.grid = false;
        s.bloom = s.bloom.min(0.2);
    }

    fn animated(&self) -> bool {
        self.look.moving || self.shown.pending(&self.wanted()) || (self.counting && self.board.pending())
    }

    fn pointer_hint(&self) -> Option<&'static str> {
        Some("buttons over the picture: click")
    }

    fn ui(&mut self, ui: &mut egui::Ui, _: &Gfx) {
        ui::note(ui, r"A cause can set how widely its effect scatters as well as where it lies: a dose that raises a response on average and also makes it more variable. Then the effect is a level that depends on the cause, plus noise whose size depends on it too:");
        ui::equation(ui, r"Y = f(X) + g(X)\,N, \quad N \text{ independent of } X");
        ui::note(ui, r"$f$ is the location (the level) and $g > 0$ the scale (the spread). Added noise, as in “Cause or effect?”, is the special case of an even spread: $g$ the same everywhere.");
        ui::note(ui, r"Fit the level alone, and what is left over, $Y - \hat f(X)$, is $g(X)\,N$: its spread follows $g$, so it depends on $X$ although $X$ causes $Y$. Divide it by a fitted spread, and $(Y - \hat f(X)) / \hat g(X)$ is the noise $N$ itself, but for the fits' own errors: independent of $X$. The other way round, in general no level and spread make what is left over independent of the input, so the direction can be told: a theorem, with exceptions in special cases (Immer and others, 2023; Strobl and Lasko, 2023). The familiar exception: a straight line, an even spread, and the cause and the noise both bell-shaped.");
        ui::note(ui, r"The check assumes that no hidden variable causes both. One that passes is not proof that the model is right; one that fails says only that this model does not fit this way round.");

        ui::section(ui, "Things to try");
        for (k, trial) in TRIALS.iter().enumerate() {
            if ui::experiment(ui, trial.button, trial.caption) {
                self.set_up(k);
            }
        }

        ui::section(ui, "The data");
        let mut graph = GRAPHS.iter().position(|g| *g == self.setup.graph).unwrap_or(0);
        if ui::choice(ui, &[r"$X \to Y$", r"$X \leftarrow U \to Y$", r"$U \to X,\ U \to Y,\ X \to Y$"], &mut graph) {
            self.setup.graph = GRAPHS[graph];
        }
        match self.setup.graph {
            Graph::Direct => ui::equation(ui, r"Y = f(X) + \sigma g(X)\,N"),
            Graph::Common => ui::equation(ui, r"X = cU + \sqrt{1 - c^2}\,E, \quad Y = \lambda f(U) + \sigma e^{\kappa U - \kappa^2} N"),
            Graph::Confounded => ui::equation(ui, r"X = cU + \sqrt{1 - c^2}\,E, \quad Y = f(X) + \lambda f(U) + \sigma g(X)\, e^{\kappa U - \kappa^2} N"),
        }
        ui::note(ui, r"$X$, $U$ and $X$'s own noise $E$ are bell-shaped with spread 1; $N$ has the shape chosen below. ($e^{-\kappa^2}$ keeps the noise's overall size at $\sigma$.)");
        if self.setup.graph.third() {
            let mut seen = self.setup.seen as usize;
            if ui::choice(ui, &[r"$U$ hidden", r"$U$ measured: then called $Z$, and in every fit"], &mut seen) {
                self.setup.seen = seen == 1;
            }
            let mut close = self.setup.close as f32;
            if ui::slider(ui, r"on $X$: $c$", &mut close, 0.0..=1.0) {
                self.setup.close = close as f64;
            }
            let mut level = self.setup.level as f32;
            if ui::slider(ui, r"on $Y$'s level: $\lambda$", &mut level, 0.0..=1.5) {
                self.setup.level = level as f64;
            }
            let mut scale = self.setup.scale as f32;
            if ui::slider(ui, r"on $Y$'s spread: $\kappa$", &mut scale, 0.0..=0.8) {
                self.setup.scale = scale as f64;
            }
            ui::note(ui, r"How strongly $U$ acts on $X$, on the level of $Y$ (through the same curve $f$) and on the spread of $Y$.");
        }
        let mut curve = CURVES.iter().position(|c| c.0 == self.setup.curve).unwrap_or(0);
        let curve_names: Vec<&str> = CURVES.iter().map(|c| c.1).collect();
        if ui::choice(ui, &curve_names, &mut curve) {
            self.setup.curve = CURVES[curve].0;
        }
        ui::note(ui, r"The level $f$.");
        let mut spread = SPREADS.iter().position(|s| s.0 == self.setup.spread).unwrap_or(0);
        let spread_names: Vec<&str> = SPREADS.iter().map(|s| s.1).collect();
        if ui::choice(ui, &spread_names, &mut spread) {
            self.setup.spread = SPREADS[spread].0;
        }
        ui::note(ui, r"The spread $g$ as $X$ grows. Pinched in the middle, it falls to a fifth of its typical size.");
        let mut noise = Noise::ALL.iter().position(|n| *n == self.setup.noise).unwrap_or(0);
        let noise_names: Vec<&str> = Noise::ALL.iter().map(|n| n.name()).collect();
        if ui::choice(ui, &noise_names, &mut noise) {
            self.setup.noise = Noise::ALL[noise];
        }
        ui::note(ui, r"The shape of the noise $N$ (mean 0, spread 1).");
        let mut sigma = self.setup.sigma as f32;
        if ui::slider(ui, r"noise $\sigma$", &mut sigma, 0.0..=1.5) {
            self.setup.sigma = sigma as f64;
        }
        ui::slider_int_log(ui, "points", &mut self.setup.n, 20..=300);
        if ui.button("new sample").clicked() {
            self.seed = self.seed.wrapping_add(1);
        }

        ui::section(ui, "The fit");
        let mut scaled = self.scaled as usize;
        if ui::choice(ui, &SWITCH, &mut scaled) {
            self.scaled = scaled == 1;
        }
        let mut method = FITTED.iter().position(|m| *m == self.method).unwrap_or(0);
        let method_names: Vec<&str> = FITTED.iter().map(|m| m.name()).collect();
        if ui::choice(ui, &method_names, &mut method) {
            self.method = FITTED[method];
            self.scaled = true;
        }
        ui::note(
            ui,
            match self.method {
                Method::TwoSteps => r"Two steps: a smooth curve through the points for the level (at each place, the straight line that best fits the points near it), then a smooth curve through the squares of what is left over, whose square root is the spread.",
                Method::Reweighted => r"Reweighted: two steps, then three rounds of fitting the level again with each point counted in proportion to $1/\hat g^2$ (less where the spread is large), and the spread again from what is then left over.",
                Method::Likelihood => r"Likelihood: the level a sum of nine smooth bumps along each input, the logarithm of the spread another, both chosen together to make the points most probable if the noise were bell-shaped: the least of $\sum_i \log g(x_i) + (y_i - f(x_i))^2 / (2 g(x_i)^2)$, plus a small penalty on bumps unlike their neighbours.",
                _ => r"Quantiles: the median for the level (a straight line near each place, by least absolute differences), and the distance between the quartiles of what is left over near each place (the values a quarter and three quarters of the way up) for the spread, divided by 1.349, the distance for a bell of spread 1. Each point's own fit leaves it out: a median fitted through a point passes through it.",
            },
        );
        let mut held = self.held as usize;
        if ui::choice(ui, &FROM, &mut held) {
            self.held = held == 1;
        }
        ui::note(ui, r"What is left over from the very points the fit was made from, or each half's from a fit made on the other half alone.");
        ui::note(ui, r"No fitted spread is allowed below a twentieth of the typical size of what is left over: dividing by a spread near 0 would blow a point up to any size.");
        if !self.counting && ui.button(format!("count over {RUNS} new samples")).clicked() {
            self.counting = true;
        }
        self.refresh();

        ui::section(ui, "What the counts showed");
        ui::note(ui, "Over 100 new samples each, at the opening settings unless said otherwise, the seed as the experiments set it. Other settings give other counts: count them. Each way of fitting is this scene's own plain version of it, so a count says how that version did here, not how good the idea is.");
        for line in FINDINGS {
            ui::note(ui, line);
        }

        ui::section(ui, "Reading the picture");
        ui::key(
            ui,
            &[
                (DOT, "dots: the samples; in each column the cause it supposes is across, the effect up"),
                (CAUSE, "cyan: the variable a column supposes is the cause"),
                (EFFECT, "orange: the variable it supposes is the effect"),
                (FIT, r"yellow: the fitted level $\hat f$ and, shaded, the band $\hat f \pm \hat g$; under it, the same laid flat"),
                (TRUTH, "green: the truth (the true level and band, the true arrows), and verdicts that are right"),
                (NAIVE, "pink: verdicts that are wrong"),
                (THIRD, r"purple: the third variable $U$ (dimmed while hidden); measured, the dots take its colour, from dark (low) to light (high)"),
                (SLICE, "blue boxes: what is left over in each slice of the input: the middle 80 % (light), the middle half (dark) and the median"),
                (SHUFFLED, "grey dots: how dependent what is left over looks when its pairing with the input is shuffled"),
                (PAIRED, "white line: how dependent it is as paired"),
            ],
        );
        ui::section(ui, "Words");
        ui::term(ui, "Location, scale:", r"the level $f$ and the spread $g$. A model with both is a location-scale (or heteroscedastic) noise model.");
        ui::term(ui, "Standardised residual:", r"what is left over divided by the fitted spread, $(Y - \hat f(X)) / \hat g(X)$.");
        ui::term(ui, "Causally sufficient:", "no variable left out of the model causes two of those in it. A hidden common cause makes a model insufficient.");
        ui::term(ui, "Independent:", "knowing one says nothing about the other: in every slice of one, the other has the same spread and shape.");
        ui::term(ui, "HSIC:", "a measure of dependence of any kind, 0 for independent variables but for sampling noise. With two inputs, two points count as alike when they are alike in both.");
        ui::term(ui, r"$p$:", "the share of 100 pairings (the real one and 99 shuffled) that look at least as dependent as the real one. Below 0.05: called dependent. Noise that really is independent passes 96 times in 100.");
        ui::term(ui, "Level off, spread off:", r"how far the fitted level is from the true one at the points, in units of the noise's typical size; and by how much the fitted spread is typically too large or too small, apart from a factor common to every point, which the check cannot see.");

        ui::section(ui, "Numbers");
        let m = self.slot();
        for (c, name) in [(0, r"with $X$ as the input"), (1, r"with $Y$ as the input")] {
            let p = match self.shown.slots[c].get(m) {
                Some(s) if s.side.exact => "nothing left over".to_string(),
                Some(s) if s.side.done() => format!("$p = {}$", num(s.side.p(), 2)),
                _ => "testing".to_string(),
            };
            ui::readout(ui, name, p);
        }
        let known = !self.setup.graph.third() || self.seen();
        if let (true, Some(Some(a))) = (known, self.shown.slots[0].get(m).map(|s| s.result.as_ref())) {
            let truth: Vec<(f64, f64)> = self.shown.sample.x.iter().zip(&self.shown.sample.u).map(|(x, u)| self.setup.truth(*x, *u)).collect();
            let (l, s) = errors(&a.fitted, &truth);
            ui::readout(ui, r"this fit of $Y$: level off", format!("${}$", num(l, 2)));
            ui::readout(ui, r"this fit of $Y$: spread off", format!("{} %", num(100.0 * (s.exp() - 1.0), 0)));
        }
        if self.counting {
            ui::readout(ui, "new samples counted", format!("{} of {RUNS}", self.board.done));
        }
        ui::see_also(ui, &[("direction", "added noise: the same check, with an even spread"), ("lingam", "straight lines and the shape of the noise"), ("confound", "what a hidden common cause does"), ("discover", "more than two variables")]);
    }

    fn frame(&mut self, f: &mut Frame) {
        let lay = Layout::of(f.view_size(), f.top_inset, self.tab);
        let pointer = f.ptr.pos.map(|p| p / f.px);
        // (a canvas squeezed to a sliver is one the controls have opened over, on a phone)
        let tiny = lay.view.x < 160.0 || lay.view.y < 160.0;
        if !tiny && !lay.compact && self.draw_buttons(f, &lay, pointer) {
            self.refresh();
        }
        self.work(f.dt, f.real_dt);
        self.aim_strips();
        let snap = self.canvas != Some(lay.view);
        if let Some(goals) = self.goals(&lay) {
            if snap && !tiny {
                self.canvas = Some(lay.view);
            }
            self.look.follow(&goals, f.real_dt, snap || self.look.points.shown.is_empty());
            // a strip that has just appeared is drawn where it is, not glided to
            for c in 0..2 {
                let now = self.strip_ready(c);
                if now && !self.marked[c] && self.look.strips.shown.len() == 4 {
                    for i in 2 * c..2 * c + 2 {
                        (self.look.strips.shown[i], self.look.strips.between[i]) = (goals.strips[i], goals.strips[i]);
                    }
                }
                self.marked[c] = now;
            }
        }
        let ticks = [0, 1].map(|c| {
            let r = &self.look.ranges.shown;
            if r.len() < 8 {
                return [Ticks::NONE; 3];
            }
            let (across, up) = if c == 0 { ((r[0], r[1]), (r[2], r[3])) } else { ((r[2], r[3]), (r[0], r[1])) };
            let (s, d) = (lay.scatter[c], lay.resid[c]);
            [Ticks::of(across.0, across.1, s.1.x - s.0.x), Ticks::of(up.0, up.1, (s.1.y - s.0.y) * 1.3), Ticks::of(r[4 + 2 * c], r[5 + 2 * c], (d.1.y - d.0.y) * 1.3)]
        });
        let new_ticks = ticks != self.ticks;
        self.ticks = ticks;
        if tiny {
            return;
        }
        if lay.compact {
            self.draw_compact(f, &lay);
            return;
        }
        self.draw_truth(f, &lay);
        if lay.fit_view {
            for c in 0..2 {
                self.draw_column(f, &lay, c);
            }
            self.draw_legend(f, &lay);
            self.draw_verdict(f, &lay);
        }
        if lay.board_view {
            self.draw_board(f, &lay, pointer);
        }
        // new round values on an axis: make sure the grid's batch and the labels are not
        // as many as in the last frame, so that nothing takes the new ones for the old
        // ones moved
        let count = |f: &mut Frame| (apart(f, GRID_LINES, false).pts.iter().filter(|p| p.width >= 0.0).count(), f.dl.labels.len());
        if new_ticks {
            let (grid, labels) = count(f);
            if grid == self.drawn_last.0 {
                let far = f.hud(Vec2::ZERO);
                apart(f, GRID_LINES, false).segment(far, far, 1.0, Color::CLEAR);
            }
            if labels == self.drawn_last.1 {
                f.text_px(Vec2::ZERO, "", 1.0, Color::CLEAR, Align2::LEFT_TOP);
            }
        }
        self.drawn_last = count(f);
    }
}

impl LocScale {
    /// The buttons over the picture; true if a setting changed.
    fn draw_buttons(&mut self, f: &mut Frame, lay: &Layout, pointer: Option<Vec2>) -> bool {
        let mut changed = false;
        let size = lay.small;
        if let Some(tabs) = lay.tabs {
            for (k, r) in tabs.iter().enumerate() {
                if button(f, *r, TABS[k], lay.text, pointer, self.tab == k) {
                    self.tab = k;
                }
            }
        }
        if !lay.narrow {
            for (k, text) in ["fit", "how", "left over from"].iter().enumerate() {
                f.text_px(lay.labels[k], *text, lay.small, pal::DIM, Align2::LEFT_CENTER);
            }
        }
        if lay.fit_view {
            for (k, r) in lay.switch.iter().enumerate() {
                if button(f, *r, SWITCH[k], size, pointer, self.scaled == (k == 1)) && self.scaled != (k == 1) {
                    self.scaled = k == 1;
                    changed = true;
                }
            }
            for (k, r) in lay.methods.iter().enumerate() {
                let on = self.scaled && self.method == FITTED[k];
                if button(f, *r, FITTED[k].name(), size, pointer, on) && !on {
                    self.method = FITTED[k];
                    self.scaled = true;
                    changed = true;
                }
            }
            if let Some(from) = lay.from {
                for (k, r) in from.iter().enumerate() {
                    if button(f, *r, FROM[k], size, pointer, self.held == (k == 1)) && self.held != (k == 1) {
                        self.held = k == 1;
                        changed = true;
                    }
                }
            }
        }
        changed
    }

    /// The settings the scoreboard counts at, in a line.
    fn settings_line(&self) -> String {
        let s = &self.setup;
        let curve = CURVES.iter().find(|c| c.0 == s.curve).map_or("", |c| c.1);
        let spread = SPREADS.iter().find(|c| c.0 == s.spread).map_or("", |c| c.1);
        let graph = match (s.graph, self.seen()) {
            (Graph::Direct, _) => r"$X \to Y$".to_string(),
            (Graph::Common, false) => r"$X \leftarrow U \to Y$, $U$ hidden".to_string(),
            (Graph::Common, true) => r"$X \leftarrow Z \to Y$, $Z$ measured".to_string(),
            (Graph::Confounded, false) => r"$U \to X$, $U \to Y$, $X \to Y$, $U$ hidden".to_string(),
            (Graph::Confounded, true) => r"$Z \to X$, $Z \to Y$, $X \to Y$, $Z$ measured".to_string(),
        };
        let from = if self.held { "left over from the other half" } else { "left over from the fitted points" };
        format!(r"{graph}; {curve}, spread {spread}, {} noise, $\sigma = {}$, {} points; {from}", s.noise.name(), num(s.sigma, 2), s.n)
    }

    /// The maps of column `c`'s two pictures, as drawn.
    fn maps(&self, lay: &Layout, c: usize) -> Option<(Map, Map)> {
        let r = &self.look.ranges.shown;
        if r.len() < 8 {
            return None;
        }
        let (x, y) = ([r[0], r[1]], [r[2], r[3]]);
        let (across, up) = if c == 0 { (x, y) } else { (y, x) };
        let deep = if c == 0 { [r[4], r[5]] } else { [r[6], r[7]] };
        Some((Map { rect: lay.scatter[c], across, up }, Map { rect: lay.resid[c], across, up: deep }))
    }

    /// The third variable's rank among the sample (0 lowest, 1 highest), by point.
    fn ranks(&self) -> Vec<f32> {
        let u = &self.shown.sample.u;
        let n = u.len();
        let mut order: Vec<usize> = (0..n).collect();
        order.sort_by(|&a, &b| u[a].total_cmp(&u[b]));
        let mut rank = vec![0.5; n];
        for (k, &i) in order.iter().enumerate() {
            rank[i] = k as f32 / (n.max(2) - 1) as f32;
        }
        rank
    }

    fn draw_column(&self, f: &mut Frame, lay: &Layout, c: usize) {
        let Some((scatter, resid)) = self.maps(lay, c) else { return };
        let (input, output) = names(c);
        let width = lay.scatter[c].1.x - lay.head[c].x;
        let misc = &self.look.misc.shown;
        let (band, seen, known) = (misc.first().copied().unwrap_or(0.0), misc.get(1).copied().unwrap_or(0.0), misc.get(2).copied().unwrap_or(0.0));
        let inputs = if self.seen() { format!("{input}, {}", tint(THIRD, "Z")) } else { input.clone() };

        // the heading
        let head = lay.head[c];
        let title = format!("if ${input}$ causes ${output}$");
        f.text_px(head, title.as_str(), lay.big, pal::TEXT, Align2::LEFT_TOP);
        let true_here = c == 0 && self.setup.graph.arrow();
        let w = f.text_size(&title, lay.big).x;
        f.text_px(head + Vec2::new(w + 10.0, 2.0), "(true)", lay.small, TRUTH.a(if true_here { 1.0 } else { 0.0 }), Align2::LEFT_TOP);
        let model = if self.scaled {
            [format!(r"fit ${output} = \hat f({inputs}) + \hat g({inputs})\,N$, $N$ independent of ${inputs}$"), format!(r"fit ${output} = \hat f({inputs}) + \hat g({inputs})\,N$"), format!(r"fit ${output}$ from ${inputs}$")]
        } else {
            [format!(r"fit ${output} = \hat f({inputs}) + \text{{noise}}$, the noise independent of ${inputs}$"), format!(r"fit ${output} = \hat f({inputs}) + \text{{noise}}$"), format!(r"fit ${output}$ from ${inputs}$")]
        };
        let line = pick(f, &model, width, lay.small);
        f.text_px(head + Vec2::new(0.0, lay.big + 5.0), line, lay.small, pal::DIM, Align2::LEFT_TOP);

        // the scatter
        let (min, max) = lay.scatter[c];
        hud_panel(f, min - Vec2::splat(4.0), max + Vec2::splat(4.0));
        self.axes(f, lay, &scatter, self.ticks[c][0], self.ticks[c][1], true);
        let pts = &self.look.points.shown;
        let n = pts.len() / 2;
        let size = if lay.narrow { 3.0 } else { 3.8 };
        let alpha = if n > 150 { 0.62 } else { 0.8 };
        let ranks = if seen > 0.0 { self.ranks() } else { Vec::new() };
        let keep = |p: Vec2, (lo, hi): Rect| p.max(lo).min(hi);
        let tone = |i: usize| -> Color {
            let base = DOT.a(alpha);
            match ranks.get(i) {
                Some(&t) if seen > 0.0 => base.mix(third_color(t), seen),
                _ => base,
            }
        };
        let size = size + 0.8 * seen;
        for i in 0..n {
            let (a, b) = if c == 0 { (pts[2 * i], pts[2 * i + 1]) } else { (pts[2 * i + 1], pts[2 * i]) };
            let p = keep(scatter.to(a, b), lay.scatter[c]);
            f.dot(f.hud_at(1, p), size, tone(i));
        }
        // the fitted band (the middle curve's), then the curves
        let cv = &self.look.curves[c].shown;
        let g = cv.len() / 9;
        // (which places of each curve are drawn is decided on the sample's own places,
        // not on where they are gliding, so that the same places are drawn throughout)
        let spans = self.shown.cols[c].spans;
        let grid = &self.shown.cols[c].across;
        let drawn = |k: usize, j: usize| grid.get(j).is_none_or(|a| (spans[k.min(2)].0..=spans[k.min(2)].1).contains(a));
        let curve = |k: usize, part: usize| -> Vec<Vec2> {
            (0..g)
                .filter(|&j| drawn(k, j))
                .map(|j| {
                    let at = (k * g + j) * 3;
                    let (a, l, s) = (cv[at], cv[at + 1], cv[at + 2]);
                    let v = match part {
                        0 => l,
                        1 => l - s,
                        _ => l + s,
                    };
                    scatter.to(a, v)
                })
                .collect()
        };
        if g > 1 && band > 0.0 {
            // (the shading kept inside the picture: what is beyond it is cut off at its edge)
            let inside = |v: Vec<Vec2>| v.into_iter().map(|p| p.max(min).min(max)).collect::<Vec<Vec2>>();
            let (lo, hi) = (inside(curve(1, 1)), inside(curve(1, 2)));
            for j in 0..lo.len().min(hi.len()).saturating_sub(1) {
                let quad = [lo[j], lo[j + 1], hi[j + 1], hi[j]];
                if quad.iter().all(|p| p.is_finite()) {
                    let q = quad.map(|p| f.hud_at(1, p));
                    f.dl.tris_alpha.quad(q[0], q[1], q[2], q[3], FIT.a(0.13 * band));
                }
            }
            for part in [1, 2] {
                path_of(f, FITTED_LINES, &clip(&curve(1, part), lay.scatter[c]), 1.0, FIT.a(0.55 * band));
            }
        }
        for k in 0..3 {
            if g < 2 {
                break;
            }
            let a = if k == 1 { 1.0 } else { seen };
            path_of(f, FITTED_LINES, &clip(&curve(k, 0), lay.scatter[c]), 2.4, FIT.a(a));
        }
        // (with the third variable measured, a disc at each curve's end in its colour)
        if seen > 0.0 && g > 1 {
            for (k, t) in LEVELS.iter().enumerate() {
                if let Some(end) = curve(k, 0).last().copied().filter(|p| within(*p, lay.scatter[c], 0.0)) {
                    f.dot(f.hud_at(2, end), 10.0, third_color(*t as f32).a(seen));
                }
            }
        }
        // the truth (in the first column only)
        if c == 0 {
            let t = &self.look.truth.shown;
            let gt = t.len() / 9;
            let truth = |k: usize, part: usize| -> Vec<Vec2> {
                (0..gt)
                    .filter(|&j| drawn(k, j))
                    .map(|j| {
                        let at = (k * gt + j) * 3;
                        let (a, l, s) = (t[at], t[at + 1], t[at + 2]);
                        scatter.to(a, l + [0.0, -s, s][part.min(2)])
                    })
                    .collect()
            };
            for k in 0..3 {
                let a = if k == 1 { 1.0 } else { seen };
                if gt > 1 {
                    dashed_px(f, TRUE_CURVE, &truth(k, 0), 2.0, TRUTH.a(0.9 * a), lay.scatter[c]);
                }
            }
            if gt > 1 {
                for part in [1, 2] {
                    dashed_px(f, TRUE_CURVE, &truth(1, part), 1.2, TRUTH.a(0.6 * known), lay.scatter[c]);
                }
            }
        }
        // names in the corner
        let mut key_at = min + Vec2::new(8.0, 6.0);
        let fitted_name = if band > 0.5 { r"$\hat f$ and $\hat f \pm \hat g$, fitted" } else { r"$\hat f$, fitted" };
        f.text_px(key_at, if lay.narrow { r"$\hat f$, fitted" } else { fitted_name }, lay.small, FIT, Align2::LEFT_TOP);
        key_at.y += lay.small + 5.0;
        if c == 0 {
            let name = if self.seen() {
                r"true, at three values of $Z$"
            } else if self.setup.graph.third() {
                r"what setting $X$ would do"
            } else {
                "true level and spread"
            };
            f.text_px(key_at, if lay.narrow { "true" } else { name }, lay.small, TRUTH, Align2::LEFT_TOP);
        }
        f.text_px(Vec2::new(max.x, max.y + 3.0), format!("${input}$"), lay.text, CAUSE, Align2::RIGHT_TOP);
        f.text_px(Vec2::new(min.x - 4.0, min.y - 2.0), format!("${output}$"), lay.text, EFFECT, Align2::RIGHT_TOP);

        // what is left over
        let (rmin, rmax) = lay.resid[c];
        let label = if self.scaled {
            [
                format!(r"left over, divided by the spread: $({output} - \hat f) / \hat g$, against ${input}$"),
                format!(r"$({output} - \hat f) / \hat g$ against ${input}$"),
                r"$(\cdot - \hat f) / \hat g$".to_string(),
            ]
        } else {
            [format!(r"what is left over, ${output} - \hat f({inputs})$, against ${input}$"), format!(r"left over: ${output} - \hat f$"), "left over".to_string()]
        };
        let label = pick(f, &label, width, lay.text);
        f.text_px(Vec2::new(lay.head[c].x, rmin.y - 8.0), label, lay.text, pal::TEXT, Align2::LEFT_BOTTOM);
        hud_panel(f, rmin - Vec2::splat(4.0), rmax + Vec2::splat(4.0));
        self.axes(f, lay, &resid, self.ticks[c][0], self.ticks[c][2], false);
        self.draw_slices(f, &resid, c);
        let r = &self.look.resid.shown;
        for i in 0..n.min(r.len() / 2) {
            let a = if c == 0 { pts[2 * i] } else { pts[2 * i + 1] };
            let p = keep(resid.to(a, r[2 * i + c]), lay.resid[c]);
            f.dot(f.hud_at(2, p), size, tone(i));
        }
        // the level laid flat, and the band (±1 once divided by the spread)
        line_of(f, SLICES, resid.to(resid.across[0], 0.0), resid.to(resid.across[1], 0.0), 1.6, FIT.a(0.85));
        for v in [-1.0, 1.0] {
            let y = resid.to(resid.across[0], v).y.max(rmin.y).min(rmax.y);
            line_of(f, SLICES, Vec2::new(rmin.x, y), Vec2::new(rmax.x, y), 1.0, FIT.a(0.35 * band));
        }

        self.draw_strip(f, lay, c);
    }

    /// One line under the pictures of what is left over, naming the marks in them.
    fn draw_legend(&self, f: &mut Frame, lay: &Layout) {
        let y = lay.resid[0].1.y + 8.0;
        let (x0, x1) = (lay.head[0].x, lay.scatter[1].1.x);
        let s = lay.small;
        let long = lay.view.x >= 1000.0;
        let parts: [(&str, u8); 3] = if long {
            [("box: the middle 80 % of what is left over in a slice (darker: the middle half)", 0), ("its median", 1), ("the middle 80 % of all slices together", 2)]
        } else {
            [("each slice's middle 80 %", 0), ("median", 1), ("all slices", 2)]
        };
        let mut x = x0;
        for (text, mark) in parts {
            let w = f.text_size(text, s).x;
            if x + 20.0 + w > x1 {
                break;
            }
            let swatch = Vec2::new(x, y + 2.0);
            match mark {
                0 => {
                    fill(f, swatch, swatch + Vec2::new(14.0, s - 2.0), Color::hex(0x1B3566).a(0.9));
                    line_of(f, LEGEND, swatch, swatch + Vec2::new(14.0, 0.0), 1.6, SLICE.mix(pal::WHITE, 0.2));
                    line_of(f, LEGEND, swatch + Vec2::new(0.0, s - 2.0), swatch + Vec2::new(14.0, s - 2.0), 1.6, SLICE.mix(pal::WHITE, 0.2));
                }
                1 => line_of(f, LEGEND, swatch + Vec2::new(2.0, s * 0.5 - 1.0), swatch + Vec2::new(12.0, s * 0.5 - 1.0), 2.0, pal::WHITE),
                _ => dashed_px(f, LEGEND, &[swatch + Vec2::new(0.0, s * 0.5 - 1.0), swatch + Vec2::new(16.0, s * 0.5 - 1.0)], 1.3, pal::WHITE.a(0.55), (Vec2::ZERO, lay.view)),
            }
            f.text_px(Vec2::new(x + 20.0, y), text, s, pal::DIM, Align2::LEFT_TOP);
            x += 20.0 + w + 22.0;
        }
    }

    /// A picture's grid and its numbers (as in `direction`).
    fn axes(&self, f: &mut Frame, lay: &Layout, map: &Map, across: Ticks, up: Ticks, under: bool) {
        let (min, max) = map.rect;
        for v in across.values() {
            let x = map.to(v, map.up[0]).x;
            let seen = x >= min.x - 0.5 && x <= max.x + 0.5;
            let x = x.max(min.x).min(max.x);
            line_of(f, GRID_LINES, Vec2::new(x, min.y), Vec2::new(x, max.y), 1.0, pal::FAINT.a(if seen { 0.35 } else { 0.0 }));
            if under {
                let text = if seen && x > min.x + 8.0 && x < max.x - 14.0 { fmt_tick(v, across.step) } else { String::new() };
                f.text_px(Vec2::new(x, max.y + 5.0), text, lay.small, pal::DIM, Align2::CENTER_TOP);
            }
        }
        for v in up.values() {
            let y = map.to(map.across[0], v).y;
            let seen = y >= min.y - 0.5 && y <= max.y + 0.5;
            let y = y.max(min.y).min(max.y);
            line_of(f, GRID_LINES, Vec2::new(min.x, y), Vec2::new(max.x, y), 1.0, pal::FAINT.a(if seen && v != 0.0 { 0.35 } else { 0.0 }));
            let text = if seen && y > min.y + 14.0 && y < max.y - 6.0 { fmt_tick(v, up.step) } else { String::new() };
            f.text_px(Vec2::new(min.x - 6.0, y), text, lay.small, pal::DIM, Align2::RIGHT_CENTER);
        }
    }

    /// Column `c`'s slices of what is left over (as in `direction`).
    fn draw_slices(&self, f: &mut Frame, map: &Map, c: usize) {
        let s = &self.look.slices[c].shown;
        if s.len() < MARKS.len() {
            return;
        }
        let k = (s.len() - MARKS.len()) / PER_SLICE;
        let (min, max) = map.rect;
        let keep = |p: Vec2| Vec2::new(p.x.max(min.x).min(max.x), p.y.max(min.y).min(max.y));
        let all = &s[k * PER_SLICE..];
        for q in [all[0], all[4]] {
            let y = keep(map.to(0.0, q)).y;
            dashed_px(f, DASHED, &[Vec2::new(min.x, y), Vec2::new(max.x, y)], 1.3, pal::WHITE.a(0.55), map.rect);
        }
        let edge = SLICE.mix(pal::WHITE, 0.2);
        for j in 0..k {
            let v = &s[j * PER_SLICE..(j + 1) * PER_SLICE];
            let (x0, x1) = (keep(map.to(v[0], 0.0)).x + 1.0, keep(map.to(v[1], 0.0)).x - 1.0);
            let x1 = x1.max(x0 + 1.0);
            let y = |q: f32| keep(map.to(v[0], q)).y;
            fill(f, Vec2::new(x0, y(v[6])), Vec2::new(x1, y(v[2])), Color::hex(0x1B3566).a(0.55));
            fill(f, Vec2::new(x0, y(v[5])), Vec2::new(x1, y(v[3])), Color::hex(0x24457F).a(0.5));
            for q in [v[2], v[6]] {
                line_of(f, SLICES, Vec2::new(x0, y(q)), Vec2::new(x1, y(q)), 2.2, edge);
            }
            let mid = 0.5 * (x0 + x1);
            let tick = ((x1 - x0) * 0.25).clamp(3.0, 10.0);
            line_of(f, SLICES, Vec2::new(mid - tick, y(v[4])), Vec2::new(mid + tick, y(v[4])), 2.0, pal::WHITE.a(0.9));
        }
    }

    /// Column `c`'s strip: the dependence as paired, against shuffled pairings.
    fn draw_strip(&self, f: &mut Frame, lay: &Layout, c: usize) {
        let (min, max) = lay.strip[c];
        let width = max.x - lay.head[c].x;
        let (input, _) = names(c);
        let inputs = if self.seen() { format!("${input}$ and ${}$", tint(THIRD, "Z")) } else { format!("${input}$") };
        let label = [format!(r"how much it depends on {inputs}: as paired (white), shuffled (grey)"), format!(r"dependence on {inputs}"), "dependence".to_string()];
        let label = pick(f, &label, width, lay.text);
        f.text_px(Vec2::new(lay.head[c].x, min.y - 6.0), label, lay.text, pal::TEXT, Align2::LEFT_BOTTOM);
        hud_panel(f, min - Vec2::splat(4.0), max + Vec2::splat(4.0));
        let Some(side) = self.shown.slots[c].get(self.slot()).map(|s| &s.side) else { return };
        let st = &self.look.strips.shown;
        let ready = self.strip_ready(c);
        let (top, value) = (st.get(2 * c).map_or(1.0, |v| v.exp()).max(1e-12), st.get(2 * c + 1).copied().unwrap_or(0.0));
        let x_of = |v: f32| min.x + (v / top).max(0.0).min(1.0) * (max.x - min.x);
        line_of(f, STRIPS, Vec2::new(min.x, max.y), Vec2::new(max.x, max.y), 1.0, pal::DIM.a(0.7));
        f.text_px(Vec2::new(min.x - 6.0, max.y), "0", lay.small, pal::DIM, Align2::RIGHT_CENTER);
        if !lay.narrow {
            f.text_px(Vec2::new(max.x, max.y + 5.0), "more dependent →", lay.small, pal::DIM, Align2::RIGHT_TOP);
        }
        let (row, h) = (lay.small + 4.0, max.y - min.y);
        for (k, s) in side.shuffles.iter().enumerate().filter(|_| ready) {
            let jitter = ((k as f32 + 1.0) * 0.618_034).fract();
            let p = Vec2::new(x_of(*s as f32), min.y + row + 2.0 + jitter * (h - row - 6.0).max(1.0));
            let at = f.hud_at(1, p);
            let reached = *s >= side.value;
            f.dl.points_with(PointKind::Disc, PointStyle { outline_px: 0.0, ..Default::default() }).push(at, 4.0, if reached { pal::TEXT.a(0.85) } else { SHUFFLED.a(0.7) });
        }
        let below = Vec2::new(lay.head[c].x, max.y + 6.0);
        if side.exact {
            f.text_px(below, "nothing left over: the fit is exact", lay.small, pal::DIM, Align2::LEFT_TOP);
            return;
        }
        if ready {
            let x = x_of(value);
            line_of(f, STRIPS, Vec2::new(x, min.y - 2.0), Vec2::new(x, max.y + 2.0), 2.4, PAIRED);
            let half = f.text_size("as paired", lay.small) * 0.5;
            let right = x + 6.0 + 2.0 * half.x < max.x;
            let offset = Vec2::new(if right { 6.0 + half.x } else { -6.0 - half.x }, row * 0.5);
            f.text_placed("as paired", c as u32, Vec2::new(x, min.y), offset, "as paired", lay.small, PAIRED);
        }
        let done = side.shuffles.len();
        let options = if !side.done() {
            vec![format!("testing: {done} of {SHUFFLES} shuffles"), "testing".to_string()]
        } else {
            let reached = side.reached();
            let p = num(side.p(), 2);
            let verdict = if side.independent() { "independent" } else { "dependent" };
            let how = match reached {
                0 => "no shuffle reaches it".to_string(),
                1 => "1 shuffle reaches it".to_string(),
                r => format!("{r} of {SHUFFLES} shuffles reach it"),
            };
            vec![format!("{verdict}: {how} ($p = {p}$)"), format!("{verdict} ($p = {p}$)"), verdict.to_string()]
        };
        let text = pick(f, &options, width, lay.small);
        f.text_px(below, text, lay.small, if side.done() { pal::TEXT } else { pal::DIM }, Align2::LEFT_TOP);
    }

    /// The graph of the truth, the third variable dimmed while it is hidden.
    fn draw_truth(&self, f: &mut Frame, lay: &Layout) {
        let (min, max) = lay.truth;
        hud_panel(f, min - Vec2::splat(6.0), max + Vec2::splat(4.0));
        f.text_px(min, "the truth", lay.text, pal::DIM, Align2::LEFT_TOP);
        let r = lay.node;
        let g = self.setup.graph;
        let seen = self.seen();
        let (left, right) = (min.x + r + 4.0, (max.x - r - 4.0).min(min.x + r + 4.0 + if lay.narrow { 96.0 } else { 120.0 }));
        let low = (max.y - r - 4.0).max(min.y + lay.text + 2.0 * r + 8.0);
        let (x, y) = (Vec2::new(left, low), Vec2::new(right, low));
        let u = Vec2::new(0.5 * (left + right), low - 2.6 * r);
        let (cx, cy) = if g.arrow() { (CAUSE, EFFECT) } else { (EFFECT, EFFECT) };
        let mut nodes = vec![Node { at: x, name: "$X$", color: cx }, Node { at: y, name: "$Y$", color: cy }];
        let mut links = Vec::new();
        if g.arrow() {
            links.push((0, 1, Link::Arrow, TRUTH, 2.2));
        }
        if g.third() {
            let strength = if seen { 1.0 } else { 0.45 };
            nodes.push(Node { at: u, name: if seen { "$Z$" } else { "$U$" }, color: THIRD.a(strength) });
            links.push((2, 0, Link::Arrow, TRUTH.a(strength), 2.0));
            links.push((2, 1, Link::Arrow, TRUTH.a(strength), 2.0));
        }
        if !g.third() {
            // (one row: X and Y alone)
            for node in nodes.iter_mut() {
                node.at.y = low - 1.3 * r;
            }
        }
        hud_graph(f, &nodes, &links, r);
        if g.third() {
            let note = if seen { "measured" } else { "hidden" };
            f.text_px(u + Vec2::new(r + 8.0, 0.0), note, lay.small, THIRD.a(if seen { 1.0 } else { 0.7 }), Align2::LEFT_CENTER);
        }
    }

    /// On a canvas too small for the pictures: the verdict alone.
    fn draw_compact(&self, f: &mut Frame, lay: &Layout) {
        let mid = lay.view * 0.5;
        let at = Vec2::new(mid.x - 60.0, mid.y.max(f.top_inset + 30.0));
        let verdict = self.shown.verdict(self.slot());
        let head = match verdict {
            None => "weighing…",
            Some(Verdict::XCausesY) => r"$X \to Y$",
            Some(Verdict::YCausesX) => r"$Y \to X$",
            Some(Verdict::Both) => "both fit",
            Some(Verdict::Neither) => "neither fits",
        };
        f.text_px(at, head, lay.text, pal::TEXT, Align2::LEFT_CENTER);
        f.text_px(at + Vec2::new(0.0, 20.0), "(a larger window shows the pictures)", lay.small, pal::DIM, Align2::LEFT_TOP);
    }

    fn draw_verdict(&self, f: &mut Frame, lay: &Layout) {
        let (min, max) = lay.verdict;
        let (wmin, wmax) = lay.words;
        hud_panel(f, min - Vec2::splat(6.0), Vec2::new(max.x, max.y.max(wmax.y)) + Vec2::splat(4.0));
        f.text_px(min, "verdict", lay.text, pal::DIM, Align2::LEFT_TOP);
        let m = self.slot();
        let s = &self.shown;
        let r = lay.node;
        let graph_y = min.y + lay.text + 8.0 + r;
        let gx = min.x + r + 4.0;
        let span = if lay.narrow { 64.0 } else { 84.0 };
        let verdict = s.verdict(m);
        let (cx, cy) = match verdict {
            Some(Verdict::XCausesY) => (CAUSE, EFFECT),
            Some(Verdict::YCausesX) => (EFFECT, CAUSE),
            _ => (pal::TEXT, pal::TEXT),
        };
        let nodes = [Node { at: Vec2::new(gx, graph_y), name: "$X$", color: cx }, Node { at: Vec2::new(gx + span, graph_y), name: "$Y$", color: cy }];
        let links: Vec<(usize, usize, Link, Color, f32)> = match verdict {
            Some(Verdict::XCausesY) => vec![(0, 1, Link::Arrow, PAIRED, 2.2)],
            Some(Verdict::YCausesX) => vec![(1, 0, Link::Arrow, PAIRED, 2.2)],
            Some(Verdict::Neither) | Some(Verdict::Both) => vec![(0, 1, Link::Line, pal::DIM, 1.6)],
            None => vec![],
        };
        hud_graph(f, &nodes, &links, r);
        let words_x = gx + span + r + 14.0;
        let head = match verdict {
            None => "weighing…",
            Some(Verdict::XCausesY) => r"$X \to Y$",
            Some(Verdict::YCausesX) => r"$Y \to X$",
            Some(Verdict::Both) => "both fit",
            Some(Verdict::Neither) => "neither fits",
        };
        f.text_px(Vec2::new(words_x, graph_y), head, lay.big + 2.0, if verdict.is_some() { pal::WHITE } else { pal::DIM }, Align2::LEFT_CENTER);
        let p = |c: usize| s.slots[c].get(m).map_or("1".to_string(), |sl| num(sl.side.p(), 2));
        let (pf, pb) = (p(0), p(1));
        let model = if self.scaled { "level and spread" } else { "level alone" };
        let flat = variance(&s.sample.x) == 0.0 || variance(&s.sample.y) == 0.0;
        let exact = (0..2).all(|c| s.slots[c].get(m).is_some_and(|sl| sl.side.exact));
        let why = match verdict {
            _ if flat => "One of the variables takes a single value: nothing varies with anything, and there is no direction to find.".to_string(),
            Some(_) if exact => "Nothing is left over either way: each variable is an exact function of the other, and the check has nothing to go on.".to_string(),
            None => String::new(),
            Some(Verdict::XCausesY) => format!(r"With the {model} fitted, only with $X$ as the input is what is left over independent of it ($p = {pf}$, against $p = {pb}$)."),
            Some(Verdict::YCausesX) => format!(r"With the {model} fitted, only with $Y$ as the input is what is left over independent of it ($p = {pb}$, against $p = {pf}$)."),
            Some(Verdict::Both) => format!(r"With the {model} fitted, what is left over looks independent of the input both ways ($p = {pf}$ and $p = {pb}$): the data cannot tell the directions apart."),
            Some(Verdict::Neither) => format!(r"With the {model} fitted, what is left over depends on the input both ways ($p = {pf}$ and $p = {pb}$): this model fits neither way round."),
        };
        wrap(f, wmin, wmax.x - wmin.x, wmax.y, &why, lay.small, pal::TEXT);
    }

    /// The scoreboard: every way of fitting over new samples like the one shown.
    fn draw_board(&mut self, f: &mut Frame, lay: &Layout, pointer: Option<Vec2>) {
        let (min, max) = lay.board;
        if max.y - min.y < 40.0 {
            return;
        }
        hud_panel(f, min - Vec2::splat(6.0), max + Vec2::splat(4.0));
        let width = max.x - min.x;
        if !self.counting {
            f.text_px(min, "how does each way of fitting do on new samples like this one?", lay.text, pal::DIM, Align2::LEFT_TOP);
            let size = Vec2::new(width.min(360.0), 30.0);
            let at = min + Vec2::new(0.0, lay.text + 12.0);
            if button(f, (at, at + size), &format!("count every way over {RUNS} new samples"), lay.text, pointer, false) {
                self.counting = true;
                self.refresh();
            }
            return;
        }
        let b = &self.board;
        let title = if b.done < RUNS {
            [format!("every way of fitting, over {RUNS} new samples like this one: how often each check passed ({} so far)", b.done), format!("{RUNS} new samples ({} so far)", b.done)]
        } else {
            [format!("every way of fitting, over {RUNS} new samples like this one: how often each check passed"), format!("{RUNS} new samples")]
        };
        let title = pick(f, &title, width, lay.text);
        f.text_px(min, title, lay.text, pal::DIM, Align2::LEFT_TOP);
        let bars = self.look.bars.shown.clone();
        if bars.len() < ROWS * 6 {
            return;
        }
        let known = !self.setup.graph.third() || self.seen();
        let arrow = self.setup.graph.arrow();
        let colors = [if arrow { TRUTH } else { NAIVE }, UNDECIDED, NEITHER, NAIVE];
        // (the bars a shade darker than the key, so that the counts on them read in white)
        let shade = |c: Color| c.mix(Color::hex(0x0B0E17), 0.42);
        let seen = self.seen();
        let (fw, bw) = if seen { (r"$Y$ from $X, Z$", r"$X$ from $Y, Z$") } else { (r"$X \to Y$", r"$Y \to X$") };
        let (fh, bh) = if seen { (fw.to_string(), bw.to_string()) } else { (format!("{fw} check"), format!("{bw} check")) };
        let s = lay.small;
        let narrow = lay.narrow;
        // the columns: name, level off, spread off, each check's passes, the verdicts
        let name_w = if narrow { 0.0 } else { 150.0 };
        let num_w = if narrow { 0.0 } else { 74.0 };
        let pass_w = if narrow { 0.0 } else { 104.0 };
        let x_level = min.x + name_w;
        let x_spread = x_level + num_w;
        let x_f = x_spread + num_w;
        let x_b = x_f + pass_w;
        let x_bar = if narrow { min.x } else { x_b + pass_w };
        // (on a phone, a row is two lines of words over a bar)
        let row_h = if narrow { (3.0 * s + 22.0).min((max.y - min.y - lay.text - 3.0 * s - 30.0) / ROWS as f32) } else { ((max.y - min.y - lay.text - 3.0 * s - 26.0) / ROWS as f32).min(24.0) };
        let mut y = min.y + lay.text + 6.0;
        let settings = self.settings_line();
        y += wrap(f, Vec2::new(min.x, y), width, max.y, &settings, s, pal::DIM) + 4.0;
        if !narrow {
            for (x, text) in [(x_level, "level off"), (x_spread, "spread off"), (x_f, fh.as_str()), (x_b, bh.as_str()), (x_bar, "verdicts")] {
                f.text_px(Vec2::new(x, y), text, s, pal::DIM, Align2::LEFT_TOP);
            }
            y += s + 6.0;
        }
        let show_rate = |count: usize| -> String { format!("{count}") };
        for row in 0..ROWS {
            let r = &b.rows[row];
            let truth_row = row == TRUTH_ROW;
            if truth_row && !known {
                continue;
            }
            let name = if truth_row { "the true level and spread" } else { METHODS[row].name() };
            let lit = !truth_row && ((self.scaled && METHODS[row] == self.method) || (!self.scaled && row == 0));
            let color = if lit { pal::WHITE } else { pal::TEXT.a(0.85) };
            let mid = y + 0.5 * row_h.max(s + 4.0);
            if narrow {
                // two lines of words over a bar
                f.text_px(Vec2::new(min.x, y), name, s, color, Align2::LEFT_TOP);
                if known && r.measured > 0 && !truth_row {
                    let off = format!("level off ${}$, spread off {} %", num(r.level / r.measured as f64, 2), num(100.0 * ((r.spread / r.measured as f64).exp() - 1.0), 0));
                    f.text_px(Vec2::new(max.x, y), off, s, pal::DIM, Align2::RIGHT_TOP);
                }
                let words = if truth_row { format!("checks passed: {fw} {}", show_rate(r.pass[0])) } else { format!("checks passed: {fw} {}, {bw} {}", show_rate(r.pass[0]), show_rate(r.pass[1])) };
                f.text_px(Vec2::new(min.x, y + s + 4.0), words, s, pal::DIM, Align2::LEFT_TOP);
            } else {
                f.text_px(Vec2::new(min.x, mid), name, s, color, Align2::LEFT_CENTER);
                let (level, spread) = if known && r.measured > 0 && !truth_row {
                    (format!("${}$", num(r.level / r.measured as f64, 2)), format!("{} %", num(100.0 * ((r.spread / r.measured as f64).exp() - 1.0), 0)))
                } else {
                    ("–".to_string(), "–".to_string())
                };
                f.text_px(Vec2::new(x_level, mid), level, s, pal::TEXT, Align2::LEFT_CENTER);
                f.text_px(Vec2::new(x_spread, mid), spread, s, pal::TEXT, Align2::LEFT_CENTER);
                f.text_px(Vec2::new(x_f, mid), show_rate(r.pass[0]), s, pal::TEXT, Align2::LEFT_CENTER);
                f.text_px(Vec2::new(x_b, mid), if truth_row { String::new() } else { show_rate(r.pass[1]) }, s, pal::TEXT, Align2::LEFT_CENTER);
            }
            // the verdicts as a bar (not for the truth, which is checked one way only)
            let (bar_top, bar_h) = if narrow { (y + 2.0 * s + 9.0, (row_h - 2.0 * s - 14.0).clamp(4.0, 14.0)) } else { (mid - 7.0, 14.0) };
            if !truth_row {
                hud_rect(f, Vec2::new(x_bar, bar_top), Vec2::new(max.x, bar_top + bar_h), Color::hex(0x161C2B));
                let mut x = x_bar;
                for k in 0..4 {
                    let share = bars[row * 6 + k].max(0.0).min(1.0);
                    let w = share * (max.x - x_bar);
                    if w > 0.5 {
                        hud_rect_at(f, 2, Vec2::new(x, bar_top), Vec2::new((x + w).min(max.x), bar_top + bar_h), shade(colors[k]));
                        if w > 24.0 && bar_h >= 12.0 {
                            f.text_px(Vec2::new(x + 0.5 * w, bar_top + 0.5 * bar_h), format!("{}", r.verdicts[k]), s, pal::WHITE, Align2::CENTER_CENTER);
                        }
                    }
                    x += w;
                }
            } else if !narrow {
                f.text_px(Vec2::new(x_bar, mid), "what the check gives on the noise itself", s, pal::DIM, Align2::LEFT_CENTER);
            }
            y += row_h.max(s + 4.0);
        }
        // the bar's key
        let names = [r"$X \to Y$", "both fit", "neither fits", r"$Y \to X$"];
        let mut x = x_bar;
        let ky = (y + 4.0).min(max.y - s);
        for k in 0..4 {
            hud_rect(f, Vec2::new(x, ky + 2.0), Vec2::new(x + 10.0, ky + s), shade(colors[k]));
            f.text_px(Vec2::new(x + 14.0, ky), names[k], s, pal::DIM, Align2::LEFT_TOP);
            x += 14.0 + f.text_size(names[k], s).x + 14.0;
        }
        if !known && !narrow {
            let note = "level and spread off: no truth to compare with while the third variable is hidden";
            f.text_px(Vec2::new(min.x, ky), pick(f, &[note.to_string(), String::new()], x_bar - min.x - 10.0, s), s, pal::DIM, Align2::LEFT_TOP);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[allow(clippy::too_many_arguments)]
    fn setup(graph: Graph, curve: Curve, spread: Spread, noise: Noise, sigma: f64, level: f64, scale: f64, n: usize) -> Setup {
        Setup { graph, seen: false, curve, spread, noise, sigma, level, scale, close: CLOSE, n }
    }

    /// What each method's fit costs (run with --ignored --nocapture).
    #[test]
    #[ignore]
    fn costs() {
        for (seen, n) in [(false, 100), (false, 200), (false, 400), (true, 200)] {
            let s = Setup { seen, ..setup(Graph::Confounded, Curve::Step, Spread::Rising, Noise::Normal, 0.4, 0.5, 0.5, n) };
            let sample = Draws::new(3, n).sample(&s);
            let cols = if seen { vec![sample.x.clone(), sample.u.clone()] } else { vec![sample.x.clone()] };
            let data = Inputs::new(cols);
            let grid: Vec<[f64; 2]> = (0..64).map(|k| [-2.5 + 5.0 * k as f64 / 63.0, 0.0]).collect();
            for &m in &METHODS {
                let t0 = std::time::Instant::now();
                let reps = 5;
                for _ in 0..reps {
                    std::hint::black_box(fit(m, &data, &sample.y, &[&grid]));
                }
                println!("n={n} seen={seen} {m:?}: {:.2} ms", t0.elapsed().as_secs_f64() * 1e3 / reps as f64);
            }
            let t0 = std::time::Instant::now();
            let like = Likeness::new(&data.cols.iter().map(|c| c.as_slice()).collect::<Vec<_>>());
            let t1 = t0.elapsed().as_secs_f64() * 1e3;
            let l = like.other(&sample.y);
            let t2 = t0.elapsed().as_secs_f64() * 1e3 - t1;
            let mut rng = Rng::new(1);
            let mut order = Vec::new();
            let t3 = std::time::Instant::now();
            for _ in 0..20 {
                std::hint::black_box(like.shuffled(&l, &mut rng, &mut order));
            }
            println!("n={n}: likeness {t1:.2} ms, other {t2:.2} ms, one shuffle {:.3} ms", t3.elapsed().as_secs_f64() * 1e3 / 20.0);
        }
    }

    /// An experiment as its button sets it up, with the analysis of the sample it shows
    /// done to the end (for the level alone and for the method it shows).
    fn run(k: usize) -> LocScale {
        let mut s = LocScale::create();
        s.set_up(k);
        s.settle();
        let wanted = [0, s.method.index()];
        while s.shown.pending(&wanted) {
            s.shown.step(&wanted);
        }
        s
    }

    /// The spreads (10th to 90th percentile) of what is left over in each slice of
    /// column `c`, by slot `m` (as it is for the level alone, divided by the spread
    /// otherwise), as a desktop picture draws them.
    fn spreads(s: &LocScale, c: usize, m: usize) -> Vec<f64> {
        let input = if c == 0 { &s.shown.sample.x } else { &s.shown.sample.y };
        let Some(a) = &s.shown.slots[c][m].result else { return Vec::new() };
        let r = if m == 0 { &a.raw } else { &a.z };
        let k = slice_count(input.len(), 480.0);
        let v = slices(input, r, k);
        (0..k).map(|j| (v[j * PER_SLICE + 6] - v[j * PER_SLICE + 2]) as f64).collect()
    }

    /// The widest slice's spread over the narrowest's.
    fn ratio(v: &[f64]) -> f64 {
        v.iter().copied().fold(0.0f64, f64::max) / v.iter().copied().fold(f64::MAX, f64::min)
    }

    /// The numbers behind the captions (run with --ignored --nocapture).
    #[test]
    #[ignore]
    fn explore_trials() {
        let all: Vec<(usize, LocScale)> = std::thread::scope(|scope| {
            let runs: Vec<_> = (0..TRIALS.len())
                .map(|k| {
                    scope.spawn(move || {
                        let mut s = run(k);
                        s.board.finish();
                        (k, s)
                    })
                })
                .collect();
            runs.into_iter().map(|r| r.join().expect("a trial")).collect()
        });
        for (k, s) in &all {
            println!("== {}: {:?} method {:?}", TRIALS[*k].button, s.setup, s.method);
            for m in [0, s.method.index()] {
                let p = |c: usize| s.shown.slots[c][m].side.p();
                println!(
                    "   {:?}: p {:.2} {:.2}, verdict {:?}, slice ratio {:.2} {:.2}, slices {:?} | {:?}",
                    METHODS[m],
                    p(0),
                    p(1),
                    s.shown.verdict(m),
                    ratio(&spreads(s, 0, m)),
                    ratio(&spreads(s, 1, m)),
                    spreads(s, 0, m).iter().map(|v| format!("{v:.2}")).collect::<Vec<_>>(),
                    spreads(s, 1, m).iter().map(|v| format!("{v:.2}")).collect::<Vec<_>>()
                );
                if let Some(a) = &s.shown.slots[0][m].result {
                    let truth: Vec<(f64, f64)> = s.shown.sample.x.iter().zip(&s.shown.sample.u).map(|(x, u)| s.setup.truth(*x, *u)).collect();
                    let causal: Vec<f64> = s.shown.sample.x.iter().map(|x| s.truth_at(*x, 1).0).collect();
                    let off = rms(&a.fitted.level.iter().zip(&causal).map(|(a, b)| a - b).collect::<Vec<_>>());
                    println!("      errors {:?}, level from what setting X does: {off:.3}", errors(&a.fitted, &truth));
                }
            }
            for (r, row) in s.board.rows.iter().enumerate() {
                let name = METHODS.get(r).map_or("truth".to_string(), |m| format!("{m:?}"));
                println!("   board {name:<11} pass {:?} verdicts {:?} level {:.3} spread {:.3} ({} measured)", row.pass, row.verdicts, row.level / row.measured.max(1) as f64, row.spread / row.measured.max(1) as f64, row.measured);
            }
        }
    }

    /// The settings the panel's and the report's measurements are made at, each as the
    /// scoreboard counts it (the seed every experiment shows).
    fn settings_measured() -> Vec<(&'static str, Setup, bool)> {
        let g = GROWING;
        let hidden = Setup { graph: Graph::Common, spread: Spread::Even, ..g };
        let confounded = Setup { graph: Graph::Confounded, spread: Spread::Even, level: 0.5, scale: 0.5, ..g };
        vec![
            ("step, growing", g, false),
            ("step, growing, other half", g, true),
            ("step, even", Setup { spread: Spread::Even, ..g }, false),
            ("step, even, other half", Setup { spread: Spread::Even, ..g }, true),
            ("straight, even", Setup { curve: Curve::Line, spread: Spread::Even, ..g }, false),
            ("straight, growing", Setup { curve: Curve::Line, ..g }, false),
            ("bend, growing", Setup { curve: Curve::Bend, ..g }, false),
            ("parabola, growing", Setup { curve: Curve::Parabola, ..g }, false),
            ("step, pinched", Setup { spread: Spread::Pinched, ..g }, false),
            ("step, pinched, other half", Setup { spread: Spread::Pinched, ..g }, true),
            ("step, wide in the middle", Setup { spread: Spread::Bulge, ..g }, false),
            ("step, growing, peaked", Setup { noise: Noise::Laplace, ..g }, false),
            ("step, growing, peaked, other half", Setup { noise: Noise::Laplace, ..g }, true),
            ("step, growing, flat", Setup { noise: Noise::Uniform, ..g }, false),
            ("step, growing, two humps", Setup { noise: Noise::TwoHumps, ..g }, false),
            ("step, growing, 30 points", Setup { n: 30, ..g }, false),
            ("step, growing, 50 points", Setup { n: 50, ..g }, false),
            ("step, growing, 100 points", Setup { n: 100, ..g }, false),
            ("step, growing, 300 points", Setup { n: 300, ..g }, false),
            ("step, growing, sigma 0.2", Setup { sigma: 0.2, ..g }, false),
            ("step, growing, sigma 0.7", Setup { sigma: 0.7, ..g }, false),
            ("U hidden: X <- U -> Y", hidden, false),
            ("U hidden: X <- U -> Y, other half", hidden, true),
            ("U hidden: X <- U -> Y, kappa 0.5", Setup { scale: 0.5, ..hidden }, false),
            ("U measured: X <- Z -> Y", Setup { seen: true, ..hidden }, false),
            ("U measured: X <- Z -> Y, kappa 0.5", Setup { seen: true, scale: 0.5, ..hidden }, false),
            ("U hidden: U -> X, U -> Y, X -> Y", confounded, false),
            ("U hidden: U -> X, U -> Y, X -> Y, other half", confounded, true),
            ("U measured: Z -> X, Z -> Y, X -> Y", Setup { seen: true, ..confounded }, false),
            ("U measured: Z -> X, Z -> Y, X -> Y, growing", Setup { seen: true, spread: Spread::Rising, ..confounded }, false),
            ("U hidden: U -> X, U -> Y, X -> Y, growing", Setup { spread: Spread::Rising, ..confounded }, false),
        ]
    }

    /// The scoreboard at each of `settings_measured` (run with --ignored --nocapture).
    #[test]
    #[ignore]
    fn explore_settings() {
        let list = settings_measured();
        let boards: Vec<Board> = std::thread::scope(|scope| {
            let runs: Vec<_> = list
                .iter()
                .map(|(_, setup, held)| {
                    let key = Key { setup: *setup, seed: FIRST_SEED, held: *held };
                    scope.spawn(move || {
                        let mut b = Board::new();
                        b.restart(key);
                        b.finish();
                        b
                    })
                })
                .collect();
            runs.into_iter().map(|r| r.join().expect("a board")).collect()
        });
        for ((name, _, _), b) in list.iter().zip(&boards) {
            println!("== {name}");
            for (r, row) in b.rows.iter().enumerate() {
                let label = METHODS.get(r).map_or("truth", |m| m.name());
                let errs = if row.measured > 0 { format!("level off {:.2}, spread off {:.0} %", row.level / row.measured as f64, 100.0 * ((row.spread / row.measured as f64).exp() - 1.0)) } else { String::new() };
                if r == TRUTH_ROW {
                    if b.rows[r].pass[0] > 0 {
                        println!("   {label:<11} X->Y passes {:>3}", row.pass[0]);
                    }
                } else {
                    println!("   {label:<11} passes {:>3} {:>3}   verdicts X->Y {:>3}, both {:>3}, neither {:>3}, Y->X {:>3}   {errs}", row.pass[0], row.pass[1], row.verdicts[0], row.verdicts[1], row.verdicts[2], row.verdicts[3]);
                }
            }
        }
    }

    /// The samples the "things to try" show, held to their captions.
    #[test]
    fn the_samples_shown_say_what_the_captions_say() {
        let all: Vec<LocScale> = (0..TRIALS.len()).map(run).collect();
        let p = |s: &LocScale, m: usize, c: usize| num(s.shown.slots[c][m].side.p(), 2);
        let two = Method::TwoSteps.index();
        for (k, s) in all.iter().enumerate() {
            assert!(s.counting && s.seed == FIRST_SEED && s.setup == TRIALS[k].setup && s.method == TRIALS[k].method && s.scaled == TRIALS[k].scaled, "{k}");
        }

        // a spread that grows: two steps, a level band on the left (p = 0.59) and not
        // on the right (0.01), X -> Y; the level alone, a funnel on the left, narrow at
        // the left and wide at the right, and the check failing both ways
        let g = &all[0];
        assert_eq!((g.setup.graph, g.setup.curve, g.setup.spread, g.setup.noise), (Graph::Direct, Curve::Step, Spread::Rising, Noise::Normal));
        assert_eq!((p(g, two, 0), p(g, two, 1)), ("0.59".to_string(), "0.01".to_string()));
        assert_eq!(g.shown.verdict(two), Some(Verdict::XCausesY));
        assert_eq!((p(g, 0, 0), p(g, 0, 1)), ("0.01".to_string(), "0.01".to_string()));
        assert_eq!(g.shown.verdict(0), Some(Verdict::Neither));
        let funnel = spreads(g, 0, 0);
        let last = funnel.len() - 1;
        assert!(funnel[0] < 0.2 * funnel[last] && funnel[0] + funnel[1] < 0.5 * (funnel[last] + funnel[last - 1]), "{funnel:?}");
        assert!(ratio(&spreads(g, 0, two)) < 1.6, "{:?}", spreads(g, 0, two));

        // a hidden common cause: the level alone, p = 0.01 with X as the input (the
        // spread widest in the middle) and 0.62 with Y: Y -> X; two steps, both fit
        let h = &all[1];
        assert_eq!((h.setup.graph, h.setup.seen, h.scaled), (Graph::Common, false, false));
        assert_eq!((p(h, 0, 0), p(h, 0, 1)), ("0.01".to_string(), "0.62".to_string()));
        assert_eq!(h.shown.verdict(0), Some(Verdict::YCausesX));
        let middle = spreads(h, 0, 0);
        let widest = (0..middle.len()).max_by(|&a, &b| middle[a].total_cmp(&middle[b])).unwrap_or(0);
        assert!(widest > 0 && widest + 1 < middle.len(), "{middle:?}");
        assert!(middle[widest] > 1.4 * middle[0] && middle[widest] > 1.4 * middle[middle.len() - 1], "{middle:?}");
        assert_eq!(h.shown.verdict(two), Some(Verdict::Both));

        // measured: both checks pass, p = 0.96 and 0.87
        let m = &all[2];
        assert!(m.seen() && m.shown.cols[0].data.dims() == 2 && m.shown.cols[1].data.dims() == 2);
        assert_eq!((p(m, two, 0), p(m, two, 1)), ("0.96".to_string(), "0.87".to_string()));
        assert_eq!(m.shown.verdict(two), Some(Verdict::Both));

        // a hidden cause the spread hides: X -> Y with the spread fitted, and the fitted
        // level typically 0.48 from what setting X would do
        let c = &all[3];
        assert_eq!((c.setup.graph, c.setup.seen), (Graph::Confounded, false));
        assert_eq!(c.shown.verdict(two), Some(Verdict::XCausesY));
        assert_eq!(c.shown.verdict(0), Some(Verdict::Neither));
        let fitted = &c.shown.slots[0][two].result.as_ref().expect("a fit").fitted.level;
        let setting: Vec<f64> = c.shown.sample.x.iter().map(|x| c.truth_at(*x, 1).0).collect();
        let away = rms(&fitted.iter().zip(&setting).map(|(a, b)| a - b).collect::<Vec<_>>());
        assert_eq!(num(away, 2), "0.48");

        // peaked noise
        let q = &all[4];
        assert_eq!((q.setup.noise, q.setup.spread), (Noise::Laplace, Spread::Rising));
        assert_eq!(q.shown.verdict(two), Some(Verdict::XCausesY));
    }

    /// The counts over new samples that the captions quote (a minute: run with --ignored).
    #[test]
    #[ignore]
    fn the_counts_in_the_captions_hold() {
        let boards: Vec<Board> = std::thread::scope(|scope| {
            let runs: Vec<_> = (0..TRIALS.len())
                .map(|k| {
                    scope.spawn(move || {
                        let mut s = run(k);
                        s.board.finish();
                        s.board
                    })
                })
                .collect();
            runs.into_iter().map(|r| r.join().expect("a count")).collect()
        });
        let [level, two, _, likelihood, quantiles, truth] = [0, 1, 2, 3, 4, 5];
        let spread_off = |b: &Board, r: usize| num(100.0 * ((b.rows[r].spread / b.rows[r].measured as f64).exp() - 1.0), 0);
        // (verdicts: X -> Y, both, neither, Y -> X)
        assert_eq!(boards[0].rows[level].verdicts[2], 100, "a spread that grows");
        assert_eq!(boards[0].rows[two].verdicts[0], 99);
        assert_eq!(boards[0].rows[two].pass[0], 100);
        assert_eq!(boards[0].rows[quantiles].pass[1], 85);
        assert_eq!(spread_off(&boards[0], two), "19");
        assert_eq!(boards[1].rows[level].verdicts[3], 96, "a hidden common cause");
        assert_eq!((boards[1].rows[two].verdicts[1], boards[1].rows[two].verdicts[3]), (77, 23));
        for r in 0..METHODS.len() {
            assert!(boards[2].rows[r].verdicts[1] >= 99, "measured: {:?}", boards[2].rows[r]);
        }
        assert_eq!(boards[2].rows[level].verdicts[1], 99);
        assert_eq!(boards[3].rows[level].pass[0], 12, "a hidden cause the spread hides");
        assert_eq!((boards[3].rows[two].pass[0], boards[3].rows[two].verdicts[0]), (93, 78));
        assert_eq!(boards[4].rows[truth].pass[0], 96, "peaked noise");
        assert_eq!(boards[4].rows[two].pass[0], 84);
        assert_eq!(spread_off(&boards[4], two), "24");
        assert_eq!(boards[4].rows[likelihood].verdicts[0], 96);
        assert_eq!(boards[4].rows[quantiles].pass[1], 79);
    }

    /// The counts the panel's findings quote (a minute: run with --ignored).
    #[test]
    #[ignore]
    fn the_counts_in_the_panel_hold() {
        let g = GROWING;
        let list = [
            (g, false),
            (g, true),
            (Setup { spread: Spread::Even, ..g }, false),
            (Setup { curve: Curve::Parabola, ..g }, false),
            (Setup { spread: Spread::Pinched, ..g }, false),
            (Setup { spread: Spread::Pinched, ..g }, true),
            (Setup { noise: Noise::Laplace, ..g }, false),
            (Setup { n: 50, ..g }, false),
            (Setup { n: 300, ..g }, false),
        ];
        let boards: Vec<Board> = std::thread::scope(|scope| {
            let runs: Vec<_> = list
                .iter()
                .map(|&(setup, held)| {
                    scope.spawn(move || {
                        let mut b = Board::new();
                        b.restart(Key { setup, seed: FIRST_SEED, held });
                        b.finish();
                        b
                    })
                })
                .collect();
            runs.into_iter().map(|r| r.join().expect("a count")).collect()
        });
        let [level, two, reweighted, likelihood, quantiles, truth] = [0, 1, 2, 3, 4, 5];
        let off = |b: &Board, r: usize| (num(b.rows[r].level / b.rows[r].measured as f64, 2), b.rows[r].spread / b.rows[r].measured as f64);
        let [grow, other, even, parabola, pinched, pinched_other, peaked, fifty, three] = [0, 1, 2, 3, 4, 5, 6, 7, 8].map(|k| &boards[k]);
        // level only
        assert_eq!(grow.rows[level].pass[0], 0);
        assert_eq!([level, two, likelihood, quantiles].map(|r| even.rows[r].verdicts[0]), [97, 84, 87, 6]);
        assert!(even.rows[level].verdicts[0] > even.rows[reweighted].verdicts[0]);
        // two steps
        assert_eq!(grow.rows[two].verdicts[0], 99);
        assert_eq!(peaked.rows[two].pass[0], 84);
        // reweighted: never clearly better (here, at the counts quoted), and the parabola
        for b in [grow, even, pinched, peaked, fifty, three] {
            assert!(b.rows[reweighted].verdicts[0] <= b.rows[two].verdicts[0] + 2, "{:?} against {:?}", b.rows[reweighted], b.rows[two]);
        }
        assert_eq!((off(parabola, reweighted).0, off(parabola, two).0), ("0.96".to_string(), "0.42".to_string()));
        // likelihood: the smallest spread errors except on the parabola
        for b in [grow, even, pinched, peaked, fifty, three] {
            for r in [two, reweighted, quantiles] {
                assert!(off(b, likelihood).1 < off(b, r).1, "{:?}", b.key);
            }
        }
        assert!(off(parabola, likelihood).1 > off(parabola, two).1);
        assert_eq!(off(parabola, likelihood).0, "1.36");
        assert_eq!(parabola.rows[likelihood].pass[0], 52);
        assert_eq!(peaked.rows[likelihood].pass[0], 99);
        // quantiles
        assert_eq!((grow.rows[quantiles].pass[1], three.rows[quantiles].pass[1]), (85, 55));
        // few points
        for r in [two, reweighted, likelihood, quantiles] {
            assert!(fifty.rows[r].verdicts[1] >= 85, "{:?}", fifty.rows[r]);
        }
        // the other half
        assert_eq!((other.rows[two].pass[0], grow.rows[two].pass[0]), (77, 100));
        assert_eq!((pinched_other.rows[two].pass[0], pinched.rows[two].pass[0]), (47, 98));
        for r in [two, reweighted, likelihood] {
            assert!(other.rows[r].pass[1] >= grow.rows[r].pass[1], "{:?}", other.rows[r]);
        }
        assert_eq!((other.rows[quantiles].pass[1], grow.rows[quantiles].pass[1]), (72, 85));
        assert_eq!(grow.rows[truth].pass[0], 97);
        // and the true noise passes about 96 times in 100, as a check at the 5 % level
        // with 99 shuffles should (it says dependent when at most 3 of 99 reach)
        let all: Vec<usize> = boards.iter().map(|b| b.rows[truth].pass[0]).collect();
        let mean = all.iter().sum::<usize>() as f64 / all.len() as f64;
        assert!(all.iter().all(|&p| p >= 90) && (94.0..=98.5).contains(&mean), "{all:?}");
    }

    /// Each shape of spread has mean square 1 over the cause (a bell of spread 1), so
    /// that σ is the noise's overall size whatever the shape; and the noise drawn
    /// through its quantiles has the shape asked for.
    #[test]
    fn the_spreads_and_the_noise_are_what_they_say() {
        for (spread, _) in SPREADS {
            let ms = bell_mean(|c| spread.at(c).powi(2));
            assert!((ms - 1.0).abs() < 1e-6, "{spread:?}: {ms}");
        }
        assert!(Spread::Pinched.at(0.0) < 0.21 && Spread::Rising.at(2.0) > 10.0 * Spread::Rising.at(-2.0));
        assert!((bell_mean(|c| c * c) - 1.0).abs() < 1e-6 && bell_mean(|c| Curve::Step.at(c)).abs() < 1e-9);
        for noise in Noise::ALL {
            let mut rng = Rng::new(9);
            let v: Vec<f64> = (0..100_000).map(|_| quantile(noise, uniform(&mut rng))).collect();
            assert!(mean(&v).abs() < 0.01 && (variance(&v) - 1.0).abs() < 0.02, "{noise:?}: mean {}, variance {}", mean(&v), variance(&v));
        }
        assert!(quantile(Noise::Normal, 0.0).is_finite() && quantile(Noise::Laplace, 1.0).is_finite() && quantile(Noise::TwoHumps, 0.5).abs() < 1e-6);
        assert!((bell_below(1.96) - 0.975).abs() < 1e-4);
    }

    /// With many points, every way of fitting finds the level and the spread it was
    /// made with; the likelihood fit makes the data more probable than where it starts.
    #[test]
    fn each_way_of_fitting_finds_the_truth_with_many_points() {
        let s = Setup { n: 1500, ..GROWING };
        let sample = Draws::new(21, s.n).sample(&s);
        let data = Inputs::new(vec![sample.x.clone()]);
        let truth: Vec<(f64, f64)> = sample.x.iter().map(|x| s.truth(*x, 0.0)).collect();
        for m in [Method::TwoSteps, Method::Reweighted, Method::Likelihood, Method::Quantiles] {
            let (own, _) = fit(m, &data, &sample.y, &[]);
            let (level, spread) = errors(&own, &truth);
            // (in units of the noise's typical size, which at the right end is three times that)
            assert!(level < 0.3 && spread < 0.15, "{m:?}: level off {level}, spread off {spread}");
        }
        // the quartiles of evenly weighted numbers, and with a point left out
        let v: Vec<f64> = (0..101).map(|i| i as f64).collect();
        let order: Vec<usize> = (0..101).collect();
        let [lo, hi] = weighted_quartiles(&[1.0; 101], None, &order, &v);
        assert!((lo - 25.0).abs() < 0.6 && (hi - 75.0).abs() < 0.6, "{lo} {hi}");
        let [lo, _] = weighted_quartiles(&[1.0; 101], Some(10), &order, &v);
        assert!(lo > 25.0 && lo < 26.5, "{lo}");
        assert_eq!(weighted_quartiles(&[0.0; 3], None, &[0, 1, 2], &[1.0, 2.0, 3.0]), [0.0, 0.0]);
        // the dependence measure with one input is the shared one
        let like = Likeness::new(&[&sample.x[..200]]);
        let ours = like.value(&like.other(&sample.y[..200]));
        assert!((ours - crate::math::causal::hsic(&sample.x[..200], &sample.y[..200])).abs() < 1e-12);
    }

    /// No spread is fitted below the floor, so what is left over divided by it stays
    /// finite and no more than twenty times its typical size; identical inputs, a
    /// single output value and the fewest points leave every number finite.
    #[test]
    fn the_floor_holds_and_degenerate_fits_stay_finite() {
        // half the points exactly on a line: nothing is left over there
        let x: Vec<f64> = (0..120).map(|i| i as f64 / 30.0 - 2.0).collect();
        let y: Vec<f64> = x.iter().enumerate().map(|(i, v)| if *v < 0.0 { *v } else { v + if i % 2 == 0 { 0.5 } else { -0.5 } }).collect();
        let data = Inputs::new(vec![x.clone()]);
        for m in METHODS {
            let mut job = Job::new(m, data.clone(), &y, Vec::new(), Vec::new());
            while !job.step(WORK) {}
            // (the floor is a twentieth of the typical size of what the first level leaves)
            assert!(job.floor > 0.0 && job.spread[0].iter().all(|s| s.is_finite() && *s >= job.floor), "{m:?}");
            let left: Vec<f64> = y.iter().zip(&job.level[0]).map(|(a, b)| a - b).collect();
            assert!(left.iter().zip(&job.spread[0]).all(|(l, s)| (l / s).is_finite()), "{m:?}");
        }
        let mut job = Job::new(Method::TwoSteps, data.clone(), &y, Vec::new(), Vec::new());
        while !job.step(WORK) {}
        assert!(job.spread[0].iter().all(|s| *s >= job.floor) && job.floor > 0.0);
        let most = job.left.iter().zip(&job.spread[0]).map(|(l, s)| (l / s).abs()).fold(0.0, f64::max);
        let bound = job.left.iter().map(|l| l.abs()).fold(0.0, f64::max) / job.floor;
        assert!(most <= bound + 1e-9, "{most} {bound}");
        // identical inputs; a single output value
        for (xs, ys) in [(vec![1.0; 40], (0..40).map(|i| (i % 7) as f64).collect::<Vec<f64>>()), ((0..40).map(|i| i as f64).collect(), vec![3.0; 40])] {
            let d = Inputs::new(vec![xs]);
            for m in METHODS {
                let (own, other) = fit(m, &d, &ys, &[&[[0.0, 0.0], [1e9, 0.0]]]);
                assert!(own.level.iter().chain(&own.spread).chain(&other[0].level).chain(&other[0].spread).all(|v| v.is_finite()), "{m:?}");
            }
        }
    }

    /// Every graph, the third variable hidden and measured, no noise and some, the
    /// fewest points and the most, every way of fitting: the analysis finishes, every
    /// number drawn is finite, and the verdict says what happened.
    #[test]
    fn degenerate_settings_are_told_honestly() {
        let lays = [Layout::of(Vec2::new(1075.0, 1000.0), 0.0, 0), Layout::of(Vec2::new(390.0, 756.0), 44.0, 0), Layout::of(Vec2::new(390.0, 756.0), 44.0, 1), Layout::of(Vec2::new(240.0, 180.0), 0.0, 0)];
        for graph in GRAPHS {
            for seen in [false, true] {
                for (sigma, n, spread) in [(0.0, 20, Spread::Pinched), (0.4, 20, Spread::Even), (0.4, 300, Spread::Pinched)] {
                    let mut s = LocScale::create();
                    s.setup = Setup { graph, seen, sigma, n, spread, level: 0.0, ..GROWING };
                    s.held = n == 20;
                    for m in FITTED {
                        s.method = m;
                        s.settle();
                        let all = [0, m.index()];
                        while s.shown.pending(&all) {
                            s.shown.step(&all);
                        }
                        assert!(s.shown.verdict(m.index()).is_some() && s.shown.verdict(0).is_some());
                        for c in 0..2 {
                            for slot in [0, m.index()] {
                                let a = s.shown.slots[c][slot].result.as_ref().expect("a fit");
                                assert!(a.z.iter().chain(&a.raw).chain(&a.fitted.spread).chain(&a.curves.level).chain(&a.curves.spread).all(|v| v.is_finite()), "{graph:?} {seen} {sigma} {n} {m:?}");
                                let p = s.shown.slots[c][slot].side.p();
                                assert!(p > 0.0 && p <= 1.0);
                            }
                        }
                        for lay in &lays {
                            let g = s.goals(lay).expect("goals");
                            let numbers = g.points.iter().chain(&g.resid).chain(&g.ranges).chain(&g.curves[0]).chain(&g.curves[1]).chain(&g.truth).chain(&g.misc).chain(&g.slices[0]).chain(&g.slices[1]).chain(&g.strips).chain(&g.bars);
                            assert!(numbers.clone().all(|v| v.is_finite()), "{graph:?} {seen} {sigma} {n} {m:?}");
                        }
                    }
                    // no noise and no curve to speak of: nothing is left over either way
                    if sigma == 0.0 && graph == Graph::Common {
                        assert!(s.shown.slots[0][0].side.exact && variance(&s.shown.sample.y) == 0.0);
                    }
                }
            }
        }
        // no relation at all: Y is noise, independent of X; both directions fit
        let mut s = LocScale::create();
        s.setup = Setup { graph: Graph::Common, level: 0.0, scale: 0.0, ..GROWING };
        s.settle();
        assert_eq!(s.shown.verdict(s.method.index()), Some(Verdict::Both));
        // an even spread: the two-step level is the level alone's, and its spread nearly even
        s.setup = Setup { spread: Spread::Even, ..GROWING };
        s.settle();
        let (level, two) = (s.shown.slots[0][0].result.as_ref().expect("a fit"), s.shown.slots[0][Method::TwoSteps.index()].result.as_ref().expect("a fit"));
        assert_eq!(level.fitted.level, two.fitted.level);
        let (lo, hi) = extent(&two.fitted.spread);
        assert!(hi < 2.5 * lo, "{lo} {hi}");
        assert!(s.shown.slots[0][0].side.independent() && s.shown.slots[0][Method::TwoSteps.index()].side.independent());
    }

    /// The layout of every window: finite, inside the canvas, the pictures of a column
    /// one under another, and the columns side by side, also on a phone.
    #[test]
    fn the_layout_fits_every_window() {
        for (w, h, inset) in [(1075.0, 1000.0, 0.0), (1400.0, 760.0, 0.0), (390.0, 756.0, 44.0), (580.0, 700.0, 0.0), (700.0, 500.0, 0.0), (240.0, 180.0, 0.0), (1.0, 1.0, 0.0), (3000.0, 2000.0, 0.0)] {
            for tab in [0, 1] {
                let lay = Layout::of(Vec2::new(w, h), inset, tab);
                let rects = [lay.scatter, lay.resid, lay.strip, lay.switch].concat();
                let all = rects.iter().chain(lay.methods.iter()).chain([&lay.verdict, &lay.truth, &lay.words, &lay.board]);
                assert!(all.clone().all(|(a, b)| a.is_finite() && b.is_finite()), "{w}x{h}");
                if w >= 390.0 && h >= 700.0 {
                    for c in 0..2 {
                        assert!(lay.scatter[c].1.y < lay.resid[c].0.y && lay.resid[c].1.y < lay.strip[c].0.y, "{w}x{h}");
                        assert!(lay.scatter[c].0.y >= inset && lay.scatter[c].1.y - lay.scatter[c].0.y > 90.0, "{w}x{h}");
                    }
                    assert!(lay.scatter[0].1.x < lay.scatter[1].0.x && lay.scatter[1].1.x <= w);
                    assert!(all.clone().all(|(a, b)| a.x >= 0.0 && a.y >= 0.0 && b.x <= w + 0.5 && b.y <= h + 0.5), "{w}x{h}");
                    // the strips above the verdict on a phone, above the scoreboard on a desktop
                    let below = if lay.narrow { lay.verdict.0.y } else { lay.board.0.y };
                    assert!(lay.strip[0].1.y + lay.small + 6.0 < below, "{w}x{h}");
                    if lay.board_view {
                        assert!(lay.board.1.y - lay.board.0.y > 140.0, "{w}x{h}: room for the scoreboard");
                    }
                    assert!(!lay.compact, "{w}x{h}");
                }
            }
        }
    }

    /// After a change, the picture glides to the new one and comes to rest exactly
    /// there, at any frame rate.
    #[test]
    fn glides_arrive_and_stop() {
        let lay = Layout::of(Vec2::new(1075.0, 1000.0), 0.0, 0);
        for hz in [30.0f32, 60.0, 144.0] {
            let mut s = LocScale::create();
            s.settle();
            let g = s.goals(&lay).expect("goals");
            s.look.follow(&g, 0.0, true);
            assert!(!s.look.moving);
            // level only: what is left over glides from divided to as it is
            s.scaled = false;
            s.settle();
            let mut frames = 0;
            loop {
                let g = s.goals(&lay).expect("goals");
                s.look.follow(&g, 1.0 / hz, false);
                frames += 1;
                if !s.look.moving {
                    assert!(s.look.resid.shown == g.resid && s.look.curves[0].shown == g.curves[0]);
                    break;
                }
                assert!((frames as f32) < 2.0 * hz, "still gliding after two seconds at {hz} Hz");
            }
            assert!(frames as f32 > 0.2 * hz, "arrived in {frames} frames at {hz} Hz: no glide");
        }
    }

    /// The work is in small pieces: a fit of the most points by the slowest way takes
    /// many steps, and a change of setting starts the analysis (and the count, once it
    /// is kept) afresh.
    #[test]
    fn the_work_comes_in_small_pieces() {
        let s = Setup { n: 300, ..GROWING };
        let sample = Draws::new(5, s.n).sample(&s);
        let col = Column::new(&sample.x, &sample.y, None, false, true);
        let mut jobs = col.jobs(Method::Quantiles);
        let mut steps = 0;
        while !jobs[0].step(WORK) {
            steps += 1;
        }
        // (a median at each of 300 points, each 24 passes over them: about 50 pieces)
        assert!(steps > 40, "{steps} steps");
        let mut scene = LocScale::create();
        scene.settle();
        assert!(!scene.shown.pending(&scene.wanted()) && !scene.counting);
        scene.setup.noise = Noise::Uniform;
        scene.refresh();
        assert!(scene.shown.pending(&scene.wanted()) && !scene.board.pending());
        scene.counting = true;
        scene.refresh();
        assert!(scene.board.pending() && scene.board.done == 0);
        scene.held = true;
        scene.refresh();
        assert!(scene.board.key.is_some_and(|k| k.held) && scene.shown.key.held);
        // the methods and the switch do not restart the count: it measures them all
        let key = scene.board.key;
        scene.method = Method::Quantiles;
        scene.scaled = false;
        scene.refresh();
        assert_eq!(scene.board.key, key);
    }
}

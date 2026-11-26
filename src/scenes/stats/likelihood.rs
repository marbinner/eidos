//! Maximum likelihood: how well each value of a parameter explains the data seen.
//!
//! Three charts. The first has the data and the model's density at the value being
//! tried, with a bar from each point up to the curve: those heights multiplied are the
//! likelihood of that value. The second adds their logarithms one point at a time: the
//! log-likelihood. The third has the log-likelihood of every value: its peak is the
//! estimate, the curvature there gives a standard error, and the values within 1.92 of
//! the peak are the 95 % likelihood interval, drawn beside the interval that the
//! parabola of the same curvature gives (the estimate ± 1.96 standard errors). With the
//! centre and the spread of a bell both unknown, the third chart is a map of the surface
//! over the two, drawn as level curves.

use super::plot::{self, Glide, Plot, Rect, AXES, CURVES, MARKS};
use super::*;
use crate::math::stats::{self, Dist};
use std::f64::consts::{LN_10, PI as PI64};
use std::sync::OnceLock;

pub const DEF: SceneDef = SceneDef {
    id: "likelihood",
    title: "Maximum likelihood",
    group: ESTIMATION,
    blurb: "How well each value of a parameter explains the data: the likelihood, its peak, and how the sharpness of the peak says how sure the estimate is.",
    make: |_| Box::new(Likelihood::create()),
};

/// The most points (or tosses) a sample may have.
const MOST: usize = 200;
/// How far below its peak the chart of the log-likelihood reaches.
const DEPTH: f64 = 10.0;
/// How many points a curve is drawn with.
const SAMPLES: usize = 241;
/// The least a rate's data may be (a waiting time of 0 would make every rate but an
/// infinite one impossible).
const LEAST_WAIT: f64 = 0.01;

/// How far below its peak the log-likelihood of a value may be for the value to be in
/// the 95 % likelihood interval: half the 95th percentile of a chi-square distribution
/// with one degree of freedom, 1.92. (Wilks's theorem: in large samples, twice the drop
/// from the peak to the true value has that distribution, so the values within this
/// drop cover the true one about 95 times in 100.)
fn drop95() -> f64 {
    static D: OnceLock<f64> = OnceLock::new();
    *D.get_or_init(|| Dist::ChiSquare { df: 1.0 }.quantile(0.95) / 2.0)
}

/// The 97.5th percentile of the standard normal distribution, 1.96: an estimate ± this
/// many standard errors is the usual 95 % interval.
fn z95() -> f64 {
    static Z: OnceLock<f64> = OnceLock::new();
    *Z.get_or_init(|| stats::normal_quantile(0.975))
}

// ----- the models ----------------------------------------------------------------------

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Model {
    /// Tosses of a coin, the chance of heads unknown.
    Coin,
    /// Values from a bell of known spread, its centre unknown.
    Mean,
    /// Values from a bell, its centre and its spread both unknown.
    Both,
    /// Waiting times from an exponential distribution, its rate unknown.
    Rate,
    /// Values from a Cauchy curve of width 1, its centre unknown.
    Centre,
}

const MODELS: [Model; 5] = [Model::Coin, Model::Mean, Model::Both, Model::Rate, Model::Centre];

impl Model {
    fn index(self) -> usize {
        MODELS.iter().position(|m| *m == self).unwrap_or(0)
    }

    /// The name on the panel's button.
    fn name(self) -> &'static str {
        match self {
            Model::Coin => "a coin",
            Model::Mean => "a bell's centre",
            Model::Both => "centre and spread",
            Model::Rate => "a rate",
            Model::Centre => "a Cauchy centre",
        }
    }

    /// The parameter's letter (the first of two, for the bell with both unknown).
    fn symbol(self) -> &'static str {
        match self {
            Model::Coin => "p",
            Model::Mean | Model::Both => r"\mu",
            Model::Rate => r"\lambda",
            Model::Centre => r"\theta",
        }
    }

}

/// What the data are drawn from.
#[derive(Clone, Copy, Debug, PartialEq)]
struct Truth {
    /// The true value of the parameter (for the bell with both unknown, its centre) ...
    a: f64,
    /// ... and that bell's true spread.
    b: f64,
    /// Drawn instead from a long right tail, which the model cannot be.
    tail: bool,
}

/// A long right tail: a log-normal distribution (the logarithm of a value is a bell of
/// centre 0 and spread 0.75).
const TAIL: Dist = Dist::LogNormal { mu: 0.0, sigma: 0.75 };

impl Truth {
    fn of(model: Model) -> Truth {
        let (a, b) = match model {
            Model::Coin => (0.6, 1.0),
            Model::Mean | Model::Both => (1.0, 1.0),
            Model::Rate => (1.5, 1.0),
            Model::Centre => (0.0, 1.0),
        };
        Truth { a, b, tail: false }
    }

    /// The distribution of one value (None for the coin, whose tosses are counted).
    fn dist(&self, model: Model, sd: f64) -> Option<Dist> {
        if model == Model::Coin {
            return None;
        }
        if self.tail {
            return Some(TAIL);
        }
        Some(match model {
            Model::Mean => Dist::Normal { mean: self.a, sd },
            Model::Both => Dist::Normal { mean: self.a, sd: self.b },
            Model::Rate => Dist::Exponential { rate: self.a },
            _ => Dist::Cauchy { centre: self.a, scale: 1.0 },
        })
    }

    /// The value that the estimate comes to with ever more data: the true one, or, from
    /// a long tail that the model cannot be, the value whose curve is closest to it,
    /// in the sense of the least Kullback–Leibler divergence. For a bell that is the
    /// one with the tail's own mean and spread, and for an exponential the one with its
    /// mean; for a Cauchy curve there is no formula (None).
    fn target(&self, model: Model) -> Option<[f64; 2]> {
        if !self.tail {
            return Some([self.a, self.b]);
        }
        let (mean, sd) = (TAIL.mean()?, TAIL.sd()?);
        match model {
            Model::Mean => Some([mean, 0.0]),
            Model::Both => Some([mean, sd]),
            Model::Rate => Some([1.0 / mean, 0.0]),
            _ => None,
        }
    }
}

/// `heads` heads among `n` tosses, in an order that looks like tossing: the same order
/// for the same `n`, so that one more head turns exactly one toss over.
fn pattern(heads: usize, n: usize) -> Vec<bool> {
    let mut rng = Rng::new(0x7055 + n as u64);
    let mut order: Vec<usize> = (0..n).collect();
    rng.shuffle(&mut order);
    let mut tosses = vec![false; n];
    for &i in order.iter().take(heads) {
        tosses[i] = true;
    }
    tosses
}

/// `n` draws from the truth, seeded (the first so many are the same whatever `n` is).
fn draw(model: Model, truth: Truth, sd: f64, n: usize, seed: u64) -> (Vec<f64>, Vec<bool>) {
    let mut rng = Rng::new(seed);
    match truth.dist(model, sd) {
        None => (Vec::new(), (0..n).map(|_| stats::unit(&mut rng) < truth.a).collect()),
        Some(d) => (d.draws(n, &mut rng).into_iter().map(|x| if model == Model::Rate { x.max(LEAST_WAIT) } else { x }).collect(), Vec::new()),
    }
}

// ----- the likelihood --------------------------------------------------------------------

/// The data as a model's likelihood sees them, with the sums it is made of.
#[derive(Clone, Debug, PartialEq)]
struct Obs {
    model: Model,
    /// The values (every model but the coin's).
    xs: Vec<f64>,
    /// The coin's heads and tails (not whole numbers while the picture glides from one
    /// count to another).
    heads: f64,
    tails: f64,
    /// The bell's spread, where it is known.
    sd: f64,
    sum: f64,
    mean: f64,
    /// The sum of the squared distances from the mean.
    ss: f64,
    lo: f64,
    hi: f64,
}

impl Obs {
    fn new(model: Model, xs: Vec<f64>, heads: f64, tails: f64, sd: f64) -> Obs {
        let sum: f64 = xs.iter().sum();
        let mean = if xs.is_empty() { 0.0 } else { sum / xs.len() as f64 };
        let ss = xs.iter().map(|x| (x - mean) * (x - mean)).sum();
        let lo = xs.iter().copied().fold(f64::INFINITY, f64::min);
        let hi = xs.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        Obs { model, xs, heads: heads.max(0.0), tails: tails.max(0.0), sd, sum, mean, ss, lo, hi }
    }

    fn n(&self) -> f64 {
        if self.model == Model::Coin {
            self.heads + self.tails
        } else {
            self.xs.len() as f64
        }
    }

    /// The log-likelihood of `th` (the second number is the spread, for the bell with
    /// both unknown): the sum over the data of the logarithm of the model's density at
    /// each. 0 with no data; −∞ where the data are impossible.
    fn ll(&self, th: [f64; 2]) -> f64 {
        let n = self.n();
        if n <= 0.0 {
            return 0.0;
        }
        match self.model {
            Model::Coin => {
                let p = th[0];
                if !(0.0..=1.0).contains(&p) {
                    return f64::NEG_INFINITY;
                }
                // (no tosses of a kind contribute nothing, even where that kind is impossible)
                let part = |k: f64, q: f64| if k > 0.0 { k * q.ln() } else { 0.0 };
                part(self.heads, p) + part(self.tails, 1.0 - p)
            }
            Model::Mean => normal_ll(n, self.mean, self.ss, th[0], self.sd),
            Model::Both => normal_ll(n, self.mean, self.ss, th[0], th[1]),
            Model::Rate => {
                let rate = th[0];
                if rate.is_nan() || rate <= 0.0 {
                    return f64::NEG_INFINITY;
                }
                n * rate.ln() - rate * self.sum
            }
            Model::Centre => -n * PI64.ln() - self.xs.iter().map(|x| ((x - th[0]) * (x - th[0])).ln_1p()).sum::<f64>(),
        }
    }
}

/// The log-likelihood of a bell of centre `mu` and spread `sd` for `n` values of this
/// mean whose squared distances from it add up to `ss`.
fn normal_ll(n: f64, mean: f64, ss: f64, mu: f64, sd: f64) -> f64 {
    if sd.is_nan() || sd <= 0.0 {
        return f64::NEG_INFINITY;
    }
    -n * (sd.ln() + 0.5 * (2.0 * PI64).ln()) - (ss + n * (mean - mu) * (mean - mu)) / (2.0 * sd * sd)
}

/// The model's distribution of one value at `th` (None for the coin, and for a value
/// that no distribution has).
fn dist_at(model: Model, th: [f64; 2], sd: f64) -> Option<Dist> {
    let d = match model {
        Model::Coin => return None,
        Model::Mean => Dist::Normal { mean: th[0], sd },
        Model::Both => Dist::Normal { mean: th[0], sd: th[1] },
        Model::Rate => Dist::Exponential { rate: th[0] },
        Model::Centre => Dist::Cauchy { centre: th[0], scale: 1.0 },
    };
    d.valid().then_some(d)
}

/// The logarithm of the model's density at one value `x` (for the coin, `x` is 1 for
/// heads and 0 for tails, and between the two while a toss is turning over).
fn ln_f(model: Model, x: f64, th: [f64; 2], sd: f64) -> f64 {
    match model {
        Model::Coin => (x * th[0] + (1.0 - x) * (1.0 - th[0])).ln(),
        Model::Mean => normal_ll(1.0, x, 0.0, th[0], sd),
        Model::Both => normal_ll(1.0, x, 0.0, th[0], th[1]),
        Model::Rate => {
            if th[0] > 0.0 && x >= 0.0 {
                th[0].ln() - th[0] * x
            } else {
                f64::NEG_INFINITY
            }
        }
        Model::Centre => -PI64.ln() - ((x - th[0]) * (x - th[0])).ln_1p(),
    }
}

/// The Fisher information of one value at `th`: how curved, on average, its
/// log-density is at its peak, over the data the model itself would give. (For the
/// bell with both unknown, the information about the centre and about the spread.)
fn fisher(model: Model, th: [f64; 2], sd: f64) -> [f64; 2] {
    match model {
        Model::Coin => [1.0 / (th[0] * (1.0 - th[0])), 0.0],
        Model::Mean => [1.0 / (sd * sd), 0.0],
        Model::Both => [1.0 / (th[1] * th[1]), 2.0 / (th[1] * th[1])],
        Model::Rate => [1.0 / (th[0] * th[0]), 0.0],
        Model::Centre => [0.5, 0.0],
    }
}

/// The curvature of the log-likelihood at `at`: −ℓ″, by its formula.
fn curvature(o: &Obs, at: f64) -> f64 {
    match o.model {
        Model::Coin => o.heads / (at * at) + o.tails / ((1.0 - at) * (1.0 - at)),
        Model::Mean => o.n() / (o.sd * o.sd),
        Model::Rate => o.n() / (at * at),
        Model::Centre => o
            .xs
            .iter()
            .map(|x| {
                let z2 = (x - at) * (x - at);
                2.0 * (1.0 - z2) / ((1.0 + z2) * (1.0 + z2))
            })
            .sum(),
        Model::Both => f64::NAN,
    }
}

// ----- the peak and what is read off it -------------------------------------------------

/// The peak of a log-likelihood of one parameter.
#[derive(Clone, Debug, PartialEq)]
struct Top {
    at: f64,
    ll: f64,
    /// At an end of the parameter's range, where the curve is still rising: no level top.
    edge: bool,
    /// The curvature there, −ℓ″ (where the top is level and curved).
    info: Option<f64>,
    /// The values within `drop95` of the peak: the likelihood interval, in as many
    /// pieces as it comes in.
    pieces: Vec<[f64; 2]>,
    /// How many peaks the curve has, and where they are (and how high).
    peaks: usize,
    tops: Vec<[f64; 2]>,
}

impl Top {
    fn se(&self) -> Option<f64> {
        self.info.map(|i| 1.0 / i.sqrt())
    }

    /// The estimate ± 1.96 standard errors: where the parabola of the same curvature
    /// is 1.92 below the peak.
    fn wald(&self) -> Option<[f64; 2]> {
        self.se().map(|s| [self.at - z95() * s, self.at + z95() * s])
    }

    /// From the lowest end of the likelihood interval to the highest.
    fn span(&self) -> [f64; 2] {
        let lo = self.pieces.first().map_or(self.at, |p| p[0]);
        let hi = self.pieces.last().map_or(self.at, |p| p[1]);
        [lo, hi]
    }
}

/// The peak of the surface over a bell's centre and spread.
#[derive(Clone, Debug, PartialEq)]
struct Top2 {
    mu: f64,
    sigma: f64,
    ll: f64,
    /// The curvatures across the centre and across the spread at the peak (along the
    /// two axes: there is none across both together there).
    info: [f64; 2],
    /// Each parameter's 95 % likelihood interval, the other chosen at its best for
    /// each value of it: the shadows of the region within 1.92 of the peak.
    mu_range: [f64; 2],
    sigma_range: [f64; 2],
}

impl Top2 {
    fn wald(&self) -> [[f64; 2]; 2] {
        let z = z95();
        let (a, b) = (z / self.info[0].sqrt(), z / self.info[1].sqrt());
        [[self.mu - a, self.mu + a], [self.sigma - b, self.sigma + b]]
    }
}

/// What a likelihood comes to.
#[derive(Clone, Debug, PartialEq)]
enum Fit {
    /// No data: the likelihood is the same for every value.
    Flat,
    /// No maximum: the likelihood grows without bound (the spread shrinking to 0 at the
    /// data's one value).
    Unbounded,
    One(Top),
    Two(Top2),
}

fn fit(o: &Obs) -> Fit {
    if o.model == Model::Both {
        fit2(o)
    } else {
        fit1(o).map_or(Fit::Flat, Fit::One)
    }
}

/// The highest point of `g` between `lo` and `hi` (on a logarithmic scale if `log`), by
/// golden-section search: where it is, and how high.
fn golden(g: impl Fn(f64) -> f64, lo: f64, hi: f64, log: bool) -> (f64, f64) {
    let to = |u: f64| if log { u.exp() } else { u };
    let (mut a, mut b) = if log { (lo.ln(), hi.ln()) } else { (lo, hi) };
    let r = (5f64.sqrt() - 1.0) / 2.0;
    let (mut c, mut d) = (b - r * (b - a), a + r * (b - a));
    let (mut gc, mut gd) = (g(to(c)), g(to(d)));
    for _ in 0..120 {
        if gc >= gd {
            (b, d, gd) = (d, c, gc);
            c = b - r * (b - a);
            gc = g(to(c));
        } else {
            (a, c, gc) = (c, d, gd);
            d = a + r * (b - a);
            gd = g(to(d));
        }
        if (b - a).abs() <= 1e-14 * (1.0 + a.abs()) {
            break;
        }
    }
    let x = to(0.5 * (a + b));
    (x, g(x))
}

/// The point between `inside` (where `g` is at least `level`) and `outside` (where it
/// is not) at which it crosses `level`, by halving.
fn crossing(g: impl Fn(f64) -> f64, level: f64, mut inside: f64, mut outside: f64) -> f64 {
    for _ in 0..80 {
        let m = 0.5 * (inside + outside);
        if g(m) >= level {
            inside = m;
        } else {
            outside = m;
        }
    }
    0.5 * (inside + outside)
}

/// Where a parameter is searched for: a share's whole range, and otherwise a range that
/// surely holds every peak and the interval round it.
fn domain(o: &Obs) -> [f64; 2] {
    match o.model {
        Model::Coin => [0.0, 1.0],
        Model::Mean => [o.lo - 8.0 * o.sd, o.hi + 8.0 * o.sd],
        Model::Rate => [1e-3 / o.hi.max(1e-9), 1e3 / o.lo.max(1e-9)],
        Model::Centre => [o.lo - 12.0, o.hi + 12.0],
        Model::Both => [o.lo, o.hi],
    }
}

/// The peak of a log-likelihood of one parameter, found numerically: the curve on a
/// fine grid, each of its local tops refined, the highest kept, and the interval found
/// by walking out from each top that is high enough. None with no data.
fn fit1(o: &Obs) -> Option<Top> {
    if o.n().is_nan() || o.n() <= 1e-9 {
        return None;
    }
    let [a, b] = domain(o);
    if !(a.is_finite() && b.is_finite() && b > a) {
        return None;
    }
    let log = o.model == Model::Rate;
    let k = match o.model {
        Model::Coin => 2001,
        Model::Centre => (((b - a) / 0.04) as usize).clamp(400, 1200),
        _ => 1601,
    };
    let node = |i: usize| {
        let t = i as f64 / (k - 1) as f64;
        if log {
            (a.ln() + (b.ln() - a.ln()) * t).exp()
        } else {
            a + (b - a) * t
        }
    };
    let xs: Vec<f64> = (0..k).map(node).collect();
    let ll = |x: f64| o.ll([x, 0.0]);
    let ys: Vec<f64> = xs.iter().map(|&x| ll(x)).collect();
    let mut tops: Vec<(usize, f64, f64)> = Vec::new();
    for i in 0..k {
        let y = ys[i];
        let left = if i == 0 { f64::NEG_INFINITY } else { ys[i - 1] };
        let right = if i + 1 == k { f64::NEG_INFINITY } else { ys[i + 1] };
        if !(y.is_finite() && y >= left && y > right) {
            continue;
        }
        // (a top at an end of the range stays there: the curve is still rising)
        let (x, v) = if i == 0 || i + 1 == k { (xs[i], y) } else { golden(ll, xs[i - 1], xs[i + 1], log) };
        tops.push((i, x, v.max(y)));
    }
    let &(gi, at, top) = tops.iter().max_by(|p, q| p.2.total_cmp(&q.2))?;
    let edge = gi == 0 || gi + 1 == k;
    let info = if edge { None } else { Some(curvature(o, at)).filter(|c| c.is_finite() && *c > 0.0) };
    let level = top - drop95();
    let mut pieces: Vec<[f64; 2]> = Vec::new();
    for &(_, x, v) in &tops {
        if v < level {
            continue;
        }
        // walk out from the top to the first grid point below the level each way
        let first_right = xs.partition_point(|&g| g <= x);
        let (mut inner, mut left) = (x, None);
        for j in (0..first_right).rev() {
            if ys[j] >= level {
                inner = xs[j];
            } else {
                left = Some(crossing(ll, level, inner, xs[j]));
                break;
            }
        }
        let (mut inner, mut right) = (x, None);
        for j in first_right..k {
            if ys[j] >= level {
                inner = xs[j];
            } else {
                right = Some(crossing(ll, level, inner, xs[j]));
                break;
            }
        }
        pieces.push([left.unwrap_or(a), right.unwrap_or(b)]);
    }
    pieces.sort_by(|p, q| p[0].total_cmp(&q[0]));
    let mut merged: Vec<[f64; 2]> = Vec::new();
    for p in pieces {
        match merged.last_mut() {
            Some(last) if p[0] <= last[1] => last[1] = last[1].max(p[1]),
            _ => merged.push(p),
        }
    }
    Some(Top { at, ll: top, edge, info, pieces: merged, peaks: tops.len(), tops: tops.iter().map(|&(_, x, v)| [x, v]).collect() })
}

/// The peak of the surface over a bell's centre and spread, found numerically: the
/// best of a grid, then the two in turn, each at its best for the other.
fn fit2(o: &Obs) -> Fit {
    let n = o.n();
    if n <= 0.0 {
        return Fit::Flat;
    }
    // one value, or all of them at one place: no spread, and no maximum
    let scale = 1.0 + o.mean.abs();
    let spread2 = o.ss / n;
    if n < 1.5 || spread2.is_nan() || spread2 <= (1e-9 * scale) * (1e-9 * scale) {
        return Fit::Unbounded;
    }
    let spread = (o.hi - o.lo).max(1e-9 * scale);
    let ll = |mu: f64, s: f64| o.ll([mu, s]);
    let (t0, t1) = ((spread * 1e-3).ln(), (spread * 2.0).ln());
    let (mut mu, mut sigma, mut best) = (o.mean, spread, f64::NEG_INFINITY);
    for i in 0..41 {
        for j in 0..41 {
            let (m, s) = (o.lo + spread * i as f64 / 40.0, (t0 + (t1 - t0) * j as f64 / 40.0).exp());
            let v = ll(m, s);
            if v > best {
                (mu, sigma, best) = (m, s, v);
            }
        }
    }
    for _ in 0..6 {
        mu = golden(|m| ll(m, sigma), mu - spread, mu + spread, false).0;
        sigma = golden(|s| ll(mu, s), sigma * (-3f64).exp(), sigma * 3f64.exp(), true).0;
    }
    let top = ll(mu, sigma);
    let level = top - drop95();
    // the centre's interval, the spread at its best for each centre; the spread's, the
    // centre at its best (which is the same for every spread)
    let profile = |m: f64| ll(m, (o.ss / n + (m - o.mean) * (m - o.mean)).sqrt());
    let out = |g: &dyn Fn(f64) -> f64, from: f64, step: f64| {
        let mut far = step;
        for _ in 0..60 {
            if g(from + far) < level {
                break;
            }
            far *= 2.0;
        }
        crossing(g, level, from, from + far)
    };
    let se = sigma / n.sqrt();
    let mu_range = [out(&profile, mu, -se), out(&profile, mu, se)];
    let along = |s: f64| ll(mu, s);
    let mut low = sigma * 0.5;
    for _ in 0..200 {
        if along(low) < level {
            break;
        }
        low *= 0.5;
    }
    let sigma_range = [crossing(along, level, sigma, low), out(&along, sigma, sigma / (2.0 * n).sqrt())];
    Fit::Two(Top2 { mu, sigma, ll: top, info: [n / (sigma * sigma), 2.0 * n / (sigma * sigma)], mu_range, sigma_range })
}

/// A fit kept for as long as the data it was made from stay the same.
#[derive(Default)]
struct Fitted {
    obs: Option<Obs>,
    fit: Option<Fit>,
}

impl Fitted {
    fn of(&mut self, o: &Obs) -> Fit {
        if self.obs.as_ref() != Some(o) || self.fit.is_none() {
            self.fit = Some(fit(o));
            self.obs = Some(o.clone());
        }
        self.fit.clone().unwrap_or(Fit::Flat)
    }
}

// ----- level curves ----------------------------------------------------------------------

/// How many points each half of a level curve of the bell's surface is drawn with.
const LEVEL_POINTS: usize = 120;

/// The curve on which the log-likelihood of a bell's centre and spread is `level`, up to
/// the spread `top` (the top of the chart), as `2 (LEVEL_POINTS + 1)` points (the same
/// number whatever the data, so that `eidos motion` can follow it): a closed loop if it
/// closes below `top`. For each spread σ at which the best centre, the mean m, reaches
/// the level, the two centres m ± d where the surface is at it:
/// d = σ √(2 (ℓ(m, σ) − level) / n).
fn level_curve(o: &Obs, level: f64, top: f64) -> Vec<(f64, f64)> {
    let (n, m) = (o.n(), o.mean);
    let along = |s: f64| o.ll([m, s]);
    // the spreads at which the best centre is at the level: below and above the best
    // spread (with no spread in the data, from 0 up)
    let best = if o.ss > 0.0 && n > 0.0 { (o.ss / n).sqrt() } else { 0.0 };
    let (lo, hi) = if best > 0.0 && along(best) >= level {
        let mut low = best * 0.5;
        for _ in 0..200 {
            if along(low) < level {
                break;
            }
            low *= 0.5;
        }
        let mut high = best * 2.0;
        for _ in 0..200 {
            if along(high) < level {
                break;
            }
            high *= 2.0;
        }
        (crossing(along, level, best, low), crossing(along, level, best, high))
    } else if best == 0.0 && n > 0.0 {
        let mut high = 1e-6;
        for _ in 0..200 {
            if along(high) < level {
                break;
            }
            high *= 2.0;
        }
        (0.0, crossing(along, level, high * 0.5, high))
    } else {
        (best, best)
    };
    let hi = hi.min(top.max(lo));
    let half = |k: usize| {
        let s = lo + (hi - lo) * 0.5 * (1.0 - (std::f64::consts::PI * k as f64 / LEVEL_POINTS as f64).cos());
        let d = if s > 0.0 { s * (2.0 * (along(s) - level).max(0.0) / n.max(1.0)).sqrt() } else { 0.0 };
        (s, if d.is_finite() { d } else { 0.0 })
    };
    let right = (0..=LEVEL_POINTS).map(half).map(|(s, d)| (m + d, s));
    let left = (0..=LEVEL_POINTS).rev().map(half).map(|(s, d)| (m - d, s));
    right.chain(left).collect()
}

// ----- numbers as they are written ------------------------------------------------------

/// A value to two decimals, or to two significant digits when it is smaller than a
/// tenth, with a true minus sign.
fn val(v: f64) -> String {
    figures(v, 2)
}

/// A value to `places` decimals, or to `places` significant digits (and no zeros after
/// them) when it is smaller than a tenth as it is written (0.0999 is 0.10, 0.0060 is
/// 0.006), with a true minus sign.
fn figures(v: f64, places: usize) -> String {
    let a = v.abs();
    let tenth = 0.1 - 0.5 * 10f64.powi(-(places as i32));
    if a >= 99.95 {
        num(v, 1)
    } else if a >= tenth || a == 0.0 || !a.is_finite() {
        num(v, places)
    } else {
        let text = num(v, ((places as f64 - 1.0 - a.log10().floor()) as usize).min(9));
        text.trim_end_matches('0').to_string()
    }
}

/// A log-likelihood, as markup (to go between dollars).
fn ell(v: f64) -> String {
    if v == f64::NEG_INFINITY {
        r"-\infty".to_string()
    } else {
        val(v)
    }
}

/// e to the power `v` (a log-likelihood), as markup: plainly when it is neither tiny
/// nor huge, otherwise as m × 10 to a power.
fn exp_text(v: f64) -> String {
    if v == f64::NEG_INFINITY {
        return "0".to_string();
    }
    if !v.is_finite() {
        return r"\infty".to_string();
    }
    let k = (v / LN_10).floor();
    if (-2.0..=2.0).contains(&k) {
        let x = v.exp();
        return num(x, (2.0 - k).max(0.0) as usize);
    }
    let (mut m, mut k) = ((v - k * LN_10).exp(), k as i64);
    if m >= 9.95 {
        (m, k) = (m / 10.0, k + 1);
    }
    format!(r"{} \times 10^{{{}}}", num(m, 1), k)
}

/// The likelihood interval's pieces, written out.
fn pieces_text(pieces: &[[f64; 2]]) -> String {
    let words: Vec<String> = pieces.iter().map(|p| format!("{} to {}", val(p[0]), val(p[1]))).collect();
    words.join(", and ")
}

// ----- the experiments -------------------------------------------------------------------

/// Experiments: the button, and what to look for. Each is set up by [`Likelihood::set_up`].
const TRIALS: [(&str, &str); 7] = [
    (
        "7 heads in 10",
        r"A coin tossed 10 times, 7 heads. Each bar is the chance the model gives that toss: $p$ for heads, $1 - p$ for tails; the likelihood of $p$ is their product. Drag the purple mark: the product is largest at $\hat p = 0.7$. The 95 % likelihood interval, the values whose log-likelihood is within 1.92 of the peak, runs from 0.39 to 0.92: a fair coin, 0.5, is well inside it.",
    ),
    (
        "70 heads in 100",
        r"The same share of heads in ten times the tosses. The peak is again at 0.7, ten times as sharply curved (476 against 48), so the standard error is $\sqrt{10} = 3.16$ times smaller: $\hat p \pm 1.96\,\text{SE}$ is 0.61 to 0.79, against 0.42 to 0.98. The likelihood interval narrows by 2.9: at 10 tosses it was lopsided. A fair coin is now 8.2 below the peak, far outside.",
    ),
    (
        "a bell's centre",
        r"Ten values from a bell of spread 1, its centre $\mu$ unknown. The peak is at the mean of the values, exactly, and not at the truth (green). Drag a value: the peak follows it a tenth of the way, as the mean does. For a bell the log-likelihood is a parabola, so the two intervals are the same: the mean $\pm 1.96/\sqrt{10}$.",
    ),
    (
        "centre and spread",
        r"Five values, the bell's centre and spread both unknown: the log-likelihood is a surface over the two, drawn as curves so far below its peak. The peak is at the mean, and at the spread that divides by $n$: $\hat\sigma$ is $\sqrt{4/5} = 0.89$ times $s$, the spread that divides by $n - 1$, and on average too low. Here the truth's spread, 1, lies between them. The yellow curve, 1.92 below the peak, reaches further up than down; the parabola's ellipse (orange) does not.",
    ),
    (
        "the wrong model",
        r"Sixty values from a long right tail (green), and a bell for a model. The peak is the bell with the sample's mean and spread (the value tried is set there): it puts 11 % of its chance below 0, where no value can be, and its top, at 1.32, is not where the truth's values crowd, near 0.57. Yet the surface is as tidy as for data from a bell, since it depends on the data only through their mean and spread: it gives no warning. With ever more data the peak comes to the bell that gives data from the truth the highest average log-likelihood (the green ring), not to the truth.",
    ),
    (
        "1 head in 10",
        r"The curve is lopsided: steep towards 0, gentle towards 1. The parabola at the peak is not, so $\hat p \pm 1.96\,\text{SE}$ runs from −0.086 to 0.29, below 0, where no $p$ can be. The likelihood interval, 0.006 to 0.37, stays inside, and leans the way the curve does.",
    ),
    (
        "three peaks",
        r"Three values far apart, and a Cauchy curve of width 1, its centre $\theta$ unknown. The log-likelihood has three peaks, one near each value. The highest, $\hat\theta = -1.03$, is the estimate; the parabola there knows nothing of the others. The values within 1.92 of the top come in two pieces, −6.67 to 2.04 and 3.35 to 5.42, while $\hat\theta \pm 1.96\,\text{SE}$ is −2.46 to 0.40.",
    ),
];

/// The scene opens as the first experiment sets it up.
const OPENING: usize = 0;

/// The three values of the experiment with three peaks.
const THREE: [f64; 3] = [-6.0, -1.0, 5.0];

// ----- the scene -------------------------------------------------------------------------

/// One model's data and settings (each model keeps its own).
#[derive(Clone, Debug, PartialEq)]
struct Set {
    xs: Vec<f64>,
    tosses: Vec<bool>,
    /// The value being tried: the purple mark.
    theta: [f64; 2],
    truth: Truth,
    /// The data are the draw from the truth, not changed by hand since: the truth is shown.
    drawn: bool,
    seed: u64,
    /// How many to draw.
    n: usize,
}

impl Set {
    fn new(model: Model, sd: f64) -> Set {
        let truth = Truth::of(model);
        let (theta, n, seed) = match model {
            Model::Coin => ([0.5, 0.0], 10, 1),
            Model::Mean => ([0.0, 0.0], 10, 11),
            Model::Both => ([0.0, 1.5], 10, 12),
            Model::Rate => ([0.8, 0.0], 10, 13),
            Model::Centre => ([1.0, 0.0], 5, 14),
        };
        let mut set = Set { xs: Vec::new(), tosses: Vec::new(), theta, truth, drawn: true, seed, n };
        set.redraw(model, sd);
        set
    }

    fn redraw(&mut self, model: Model, sd: f64) {
        (self.xs, self.tosses) = draw(model, self.truth, sd, self.n.min(MOST), self.seed);
        self.drawn = true;
    }
}

/// What the visitor's hand holds.
#[derive(Clone, Copy, Debug, PartialEq)]
enum Held {
    Nothing,
    /// The value being tried.
    Theta,
    /// A data point.
    Point(usize),
}

pub struct Likelihood {
    model: Model,
    sets: [Set; 5],
    /// The known spread of the bell whose centre alone is unknown.
    sd: f64,
    /// The likelihood itself is drawn, not its logarithm.
    plain: bool,
    /// What is drawn, gliding to where it has gone: the value tried; the charts'
    /// ranges and the coin's counts; the data (see [`Likelihood::goals`]).
    look_theta: Glide,
    look_range: Glide,
    look_data: Glide,
    moving: bool,
    /// The model and the canvas the glides were last given goals for (another starts
    /// afresh, without gliding).
    looked: Option<(Model, Vec2)>,
    held: Held,
    /// The point held is below its chart: let go, and it is taken away.
    removing: bool,
    /// The data chart's ranges, held still while a point is dragged.
    held_range: Option<[f32; 3]>,
    fit_true: Fitted,
    fit_shown: Fitted,
    /// The log-likelihood across the likelihood chart, kept with the data and the
    /// range it was worked out for (for the Cauchy curve each point is a sum over the
    /// data).
    curves: Vec<(Obs, [f64; 2], Vec<f64>)>,
    /// The round values on the axes in the last frame, and how many lines and labels it
    /// drew (see the end of [`Likelihood::frame`]).
    ticks: Vec<f64>,
    drawn_last: (usize, usize),
}

impl Likelihood {
    /// (Nothing here needs the graphics device, so the tests can make one too.)
    fn create() -> Likelihood {
        let sd = 1.0;
        let mut s = Likelihood {
            model: Model::Coin,
            sets: MODELS.map(|m| Set::new(m, sd)),
            sd,
            plain: false,
            look_theta: Glide::default(),
            look_range: Glide::default(),
            look_data: Glide::default(),
            moving: false,
            looked: None,
            held: Held::Nothing,
            removing: false,
            held_range: None,
            fit_true: Fitted::default(),
            fit_shown: Fitted::default(),
            curves: Vec::new(),
            ticks: Vec::new(),
            drawn_last: (0, 0),
        };
        s.set_up(OPENING);
        s
    }

    fn set(&self) -> &Set {
        &self.sets[self.model.index()]
    }

    fn set_mut(&mut self) -> &mut Set {
        &mut self.sets[self.model.index()]
    }

    /// Set up one of the experiments in `TRIALS`.
    fn set_up(&mut self, k: usize) {
        self.plain = false;
        self.sd = 1.0;
        let coin = |heads: usize, n: usize, theta: f64| {
            let mut s = Set::new(Model::Coin, 1.0);
            (s.tosses, s.n, s.theta, s.drawn) = (pattern(heads, n), n, [theta, 0.0], false);
            s
        };
        let (model, set) = match k {
            0 => (Model::Coin, coin(7, 10, 0.5)),
            1 => (Model::Coin, coin(70, 100, 0.5)),
            2 => {
                let mut s = Set::new(Model::Mean, 1.0);
                (s.n, s.seed, s.theta) = (10, 11, [0.0, 0.0]);
                s.redraw(Model::Mean, 1.0);
                (Model::Mean, s)
            }
            3 => {
                let mut s = Set::new(Model::Both, 1.0);
                (s.n, s.seed, s.theta) = (5, 4, [0.0, 1.6]);
                s.redraw(Model::Both, 1.0);
                (Model::Both, s)
            }
            4 => {
                let mut s = Set::new(Model::Both, 1.0);
                (s.n, s.seed, s.truth.tail) = (60, 21, true);
                s.redraw(Model::Both, 1.0);
                // the value tried is the peak: the purple curve is the fitted bell
                let o = Obs::new(Model::Both, s.xs.clone(), 0.0, 0.0, 1.0);
                if let Fit::Two(t) = fit(&o) {
                    s.theta = [t.mu, t.sigma];
                }
                (Model::Both, s)
            }
            5 => (Model::Coin, coin(1, 10, 0.3)),
            _ => {
                let mut s = Set::new(Model::Centre, 1.0);
                (s.xs, s.n, s.theta, s.drawn) = (THREE.to_vec(), 3, [3.0, 0.0], false);
                (Model::Centre, s)
            }
        };
        self.model = model;
        self.sets[model.index()] = set;
        self.held = Held::Nothing;
        self.held_range = None;
    }

    /// The data as the likelihood sees them, as they are.
    fn obs(&self) -> Obs {
        let s = self.set();
        let heads = s.tosses.iter().filter(|t| **t).count() as f64;
        Obs::new(self.model, if self.model == Model::Coin { Vec::new() } else { s.xs.clone() }, heads, s.tosses.len() as f64 - heads, self.sd)
    }

    /// The same, as they are drawn (gliding).
    fn obs_shown(&self) -> Obs {
        let r = &self.look_range;
        if self.model == Model::Coin {
            return Obs::new(Model::Coin, Vec::new(), r.get(6) as f64, r.get(7) as f64, r.get(8) as f64);
        }
        Obs::new(self.model, self.look_data.shown.iter().map(|x| *x as f64).collect(), 0.0, 0.0, r.get(8) as f64)
    }

    /// The fit of the data as they are (for the numbers and the captions).
    fn fit_now(&mut self) -> Fit {
        let o = self.obs();
        self.fit_true.of(&o)
    }

    /// The truth, if it is shown: the data are a draw from it.
    fn truth_shown(&self) -> Option<Truth> {
        let s = self.set();
        s.drawn.then_some(s.truth)
    }

    /// The data chart's x range and the top of its y range, from the data as they are.
    fn data_view(&self, fit: &Fit) -> ([f64; 2], f64) {
        let s = self.set();
        let model = self.model;
        if model == Model::Coin {
            return ([0.5, s.tosses.len().max(1) as f64 + 0.5], 1.0);
        }
        let o = self.obs();
        let mut span = (o.lo, o.hi);
        let take = |span: &mut (f64, f64), a: f64, b: f64| {
            if a.is_finite() && b.is_finite() {
                *span = (span.0.min(a), span.1.max(b));
            }
        };
        let th = s.theta;
        match fit {
            Fit::One(t) if model == Model::Mean => take(&mut span, t.at - 2.5 * self.sd, t.at + 2.5 * self.sd),
            Fit::One(t) if model == Model::Centre => {
                // the likelihood interval, and a little beyond, is on the same axis
                let [a, b] = t.span();
                take(&mut span, a - 0.8, b + 0.8);
            }
            Fit::Two(t) => take(&mut span, t.mu - 2.5 * t.sigma, t.mu + 2.5 * t.sigma),
            _ => {}
        }
        let truth = self.truth_shown().and_then(|t| t.dist(model, self.sd));
        if let Some(d) = truth {
            let q = if model == Model::Centre && !self.set().truth.tail { 0.1 } else { 0.01 };
            take(&mut span, d.quantile(q), d.quantile(1.0 - q));
        }
        if matches!(fit, Fit::Unbounded) {
            take(&mut span, th[0] - 2.0 * th[1], th[0] + 2.0 * th[1]);
        }
        if !(span.0.is_finite() && span.1.is_finite()) {
            // no data and no truth: round the value tried
            match model {
                Model::Mean => take(&mut span, th[0] - 3.0 * self.sd, th[0] + 3.0 * self.sd),
                Model::Both => take(&mut span, th[0] - 3.0 * th[1].max(0.1), th[0] + 3.0 * th[1].max(0.1)),
                Model::Rate => take(&mut span, 0.0, 3.0 / th[0].max(0.05)),
                _ => take(&mut span, th[0] - 5.0, th[0] + 5.0),
            }
        }
        let least = match model {
            Model::Mean => 2.0 * self.sd,
            Model::Rate => 1.0,
            _ => 2.0,
        };
        let (lo, hi) = span;
        let middle = 0.5 * (lo + hi);
        let half = (0.5 * (hi - lo)).max(0.5 * least) * 1.1;
        let (mut lo, hi) = (middle - half, middle + half);
        if model == Model::Rate {
            lo = 0.0;
        }
        // the height: room for the fitted curve and the truth
        let fitted = match fit {
            Fit::One(t) => dist_at(model, [t.at, 0.0], self.sd),
            Fit::Two(t) => dist_at(model, [t.mu, t.sigma], self.sd),
            _ => None,
        };
        let peak = |d: Dist| (0..=200).map(|i| d.pdf(lo + (hi - lo) * i as f64 / 200.0)).filter(|v| v.is_finite()).fold(0.0, f64::max);
        let mut top = fitted.map_or(0.0, peak).max(truth.map_or(0.0, peak));
        if top.is_nan() || top <= 0.0 {
            top = dist_at(model, th, self.sd).map_or(1.0, peak);
        }
        ([lo, hi], (top * 1.22).max(1e-6))
    }

    /// The likelihood chart's x range, and for the surface the top of its spread axis.
    fn param_view(&self, fit: &Fit, ax: [f64; 2]) -> ([f64; 2], f64) {
        let model = self.model;
        let target = self.truth_shown().and_then(|t| t.target(model));
        let ctop = match fit {
            Fit::Two(t) => {
                let mut top = (t.sigma_range[1] * 1.3).max(t.sigma * 1.6).max(t.wald()[1][1] * 1.1);
                if let Some([_, s]) = target {
                    top = top.max(s * 1.25);
                }
                top
            }
            _ => ((ax[1] - ax[0]) / 3.0).max(self.set().theta[1] * 1.2),
        };
        let cx = match model {
            Model::Coin => [0.0, 1.0],
            Model::Centre => match fit {
                Fit::One(t) => {
                    // the likelihood interval and every peak not far below the top
                    let [mut lo, mut hi] = t.span();
                    for &[x, v] in &t.tops {
                        if v > t.ll - DEPTH {
                            (lo, hi) = (lo.min(x), hi.max(x));
                        }
                    }
                    let w = hi - lo;
                    if ax[1] - ax[0] <= 4.0 * (w + 4.0) {
                        ax
                    } else {
                        let pad = (0.4 * w).max(2.0);
                        [lo - pad, hi + pad]
                    }
                }
                _ => ax,
            },
            Model::Rate => match fit {
                Fit::One(t) => {
                    let [mut lo, mut hi] = t.span();
                    if let Some([w0, w1]) = t.wald() {
                        (lo, hi) = (lo.min(w0), hi.max(w1));
                    }
                    if let Some([a, _]) = target {
                        (lo, hi) = (lo.min(a), hi.max(a));
                    }
                    let w = (hi - lo).max(1e-6);
                    [(lo - 0.5 * w).max(0.0), hi + 0.5 * w]
                }
                _ => [0.0, 3.0],
            },
            _ => ax,
        };
        (cx, ctop)
    }

    /// What the picture glides to: the value tried; the ranges of the charts, the
    /// coin's counts and the known spread (see [`Likelihood::charts`] for which is
    /// which); the data (the coin's tosses as 1 for heads and 0 for tails).
    fn goals(&mut self, fit: &Fit) -> [Vec<f32>; 3] {
        let model = self.model;
        let o = self.obs();
        let th = self.set().theta;
        let (ax, atop) = self.data_view(fit);
        let (ax, atop) = match self.held_range {
            Some([a, b, t]) => ([a as f64, b as f64], t as f64),
            None => (ax, atop),
        };
        let (cx, ctop) = self.param_view(fit, ax);
        let s = self.set();
        let heads = s.tosses.iter().filter(|t| **t).count() as f32;
        let data: Vec<f32> = if model == Model::Coin { s.tosses.iter().map(|t| *t as u8 as f32).collect() } else { s.xs.iter().map(|x| *x as f32).collect() };
        // the likelihood chart: its peak, and as deep as the curve goes across it if it
        // is shallow (a Cauchy curve's peaks are no more than a few apart), else `DEPTH`
        let (top, depth) = match fit {
            Fit::One(t) => {
                let lowest = self.curve_across(&o, cx).iter().copied().fold(f64::INFINITY, f64::min);
                (t.ll, if lowest.is_finite() { (1.2 * (t.ll - lowest)).max(4.0).min(DEPTH) } else { DEPTH })
            }
            _ => (0.0, DEPTH),
        };
        // the logarithms added: room for them at the value tried and at the peak
        let (mut lo, mut hi) = (0.0f64, 0.0f64);
        for at in [Some(th), peak_theta(fit)].into_iter().flatten() {
            for v in self.ladder(&data, at, o.sd) {
                if v.is_finite() {
                    (lo, hi) = (lo.min(v), hi.max(v));
                }
            }
        }
        let pad = ((hi - lo) * 0.08).max(0.5);
        let range = [ax[0], ax[1], atop, cx[0], cx[1], ctop, heads as f64, data.len() as f64 - heads as f64, self.sd, top, depth, lo - pad, hi + pad].map(|v| v as f32).to_vec();
        [vec![th[0] as f32, th[1] as f32], range, data]
    }

    /// What tells one of the labels that keep their places from the others: a model's
    /// labels are its own, and start afresh in their places when the model changes.
    fn tag(&self, k: u32) -> u32 {
        10 * self.model.index() as u32 + k
    }

    /// The value tried, as drawn.
    fn theta_shown(&self) -> [f64; 2] {
        [self.look_theta.get(0) as f64, self.look_theta.get(1) as f64]
    }

    /// The running sum of the logarithms of the heights, point by point, at `th`: 0
    /// before the first.
    fn ladder(&self, data: &[f32], th: [f64; 2], sd: f64) -> Vec<f64> {
        let mut sum = 0.0;
        let mut out = Vec::with_capacity(data.len() + 1);
        out.push(0.0);
        for &x in data {
            sum += ln_f(self.model, x as f64, th, sd);
            out.push(sum);
        }
        out
    }

    /// Clamp a value tried into the range where it means something.
    fn keep_theta(&self, th: [f64; 2], cx: [f64; 2], ctop: f64) -> [f64; 2] {
        let a = th[0].max(cx[0]).min(cx[1]);
        match self.model {
            Model::Coin => [a.max(0.001).min(0.999), 0.0],
            Model::Rate => [a.max(1e-3 * cx[1].max(1e-3)), 0.0],
            Model::Both => [a, th[1].max(ctop * 0.01).min(ctop)],
            _ => [a, 0.0],
        }
    }

    /// Make the data again from the truth (a setting of the draw changed).
    fn redraw(&mut self) {
        let (model, sd) = (self.model, self.sd);
        self.set_mut().redraw(model, sd);
    }

    /// The pointer has let go: a point let go below its chart is taken away.
    fn release(&mut self) {
        if let Held::Point(i) = self.held {
            if self.removing {
                let s = self.set_mut();
                if i < s.xs.len() {
                    s.xs.remove(i);
                    s.drawn = false;
                }
            }
            self.held_range = None;
        }
        self.held = Held::Nothing;
        self.removing = false;
    }
}

// ----- the layout --------------------------------------------------------------------------

/// Where everything goes on a canvas.
#[derive(Clone, Copy, Debug, PartialEq)]
struct Layout {
    view: Vec2,
    narrow: bool,
    /// Text sizes: titles, words, small print.
    big: f32,
    text: f32,
    small: f32,
    /// The data and the density; the logarithms added; the likelihood.
    a: Rect,
    b: Rect,
    c: Rect,
    /// Where the numbers are written.
    facts: Rect,
    /// The two buttons over the likelihood chart: its logarithm, or itself.
    switch: [Rect; 2],
    /// Room for the likelihood chart alone (a canvas squeezed small).
    compact: bool,
    /// Room for nothing.
    tiny: bool,
}

const SWITCH: [&str; 2] = ["log-likelihood", "likelihood"];

impl Layout {
    fn of(view: Vec2, inset: f32) -> Layout {
        let view = view.max(Vec2::splat(1.0));
        let narrow = view.x < 760.0;
        let (big, text, small) = if narrow { (13.0, 11.5, 10.5) } else { (15.0, 12.5, 11.5) };
        let tiny = view.x < 150.0 || view.y < 140.0;
        let compact = !tiny && (view.y < 470.0 || view.x < 250.0);
        let zero = (Vec2::ZERO, Vec2::ZERO);
        let r = |x0: f32, y0: f32, x1: f32, y1: f32| (Vec2::new(x0, y0), Vec2::new(x1.max(x0 + 1.0), y1.max(y0 + 1.0)));
        let top = inset.max(0.0) + if narrow { 40.0 } else { 52.0 };
        let (a, b, c, facts);
        if compact {
            let side = 8.0;
            c = r(side + 34.0, top, view.x - side - 6.0, view.y - 30.0);
            (a, b, facts) = (zero, zero, zero);
        } else if narrow {
            let side = 10.0;
            let (left, right) = (side + 32.0, view.x - side - 4.0);
            let facts_h = 5.0 * (small + 4.0) + 6.0;
            let chrome = 30.0 + 34.0 + 26.0 + 42.0 + 30.0;
            let rest = (view.y - top - chrome - facts_h - 8.0).max(60.0);
            let (ha, hb) = (rest * 0.37, rest * 0.17);
            a = r(left, top, right, top + ha);
            b = r(left, a.1.y + 30.0 + 34.0, right, a.1.y + 64.0 + hb);
            c = r(left, b.1.y + 26.0 + 42.0, right, view.y - facts_h - 8.0 - 30.0);
            facts = r(side, c.1.y + 32.0, view.x - side, view.y - 4.0);
        } else {
            let side = 22.0;
            let width = view.x - 2.0 * side;
            let split = side + width * 0.64;
            let left = side + 40.0;
            let rest = (view.y - top - 14.0 - 34.0 - 46.0 - 34.0).max(80.0);
            let ha = rest * 0.43;
            a = r(left, top, split, top + ha);
            c = r(left, a.1.y + 34.0 + 46.0, split, view.y - 14.0 - 34.0);
            let x0 = split + 56.0;
            b = r(x0 + 40.0, top, view.x - side - 6.0, top + ha * 0.72);
            facts = r(x0, b.1.y + 44.0, view.x - side, view.y - 12.0);
        }
        // the switch, at the right of the likelihood chart's title
        let row = if narrow { 20.0 } else { 22.0 };
        let words: Vec<f32> = SWITCH.iter().map(|w| text_width(w, small) + 16.0).collect();
        let x1 = c.1.x;
        let x0 = x1 - words[0] - words[1] - 6.0;
        let y0 = c.0.y - 44.0;
        let switch = [r(x0, y0, x0 + words[0], y0 + row), r(x0 + words[0] + 6.0, y0, x1, y0 + row)];
        Layout { view, narrow, big, text, small, a, b, c, facts, switch, compact, tiny }
    }
}

// ----- drawing helpers ---------------------------------------------------------------------

/// A curve given in a chart's units, drawn inside the chart with the same number of
/// points in every frame: each point is kept on the chart where it is across, and those
/// beyond the edge are ghosts, which draw nothing, but for the first one beyond, to
/// which the curve is drawn so that it reaches the edge. Every point then moves
/// smoothly as the curve does, also as it goes off the chart and comes back. (`Plot::path`
/// cuts the curve instead, and so changes its number of points as it moves, which
/// `eidos motion` cannot follow.)
fn kept_path(f: &mut Frame, kind: u8, p: &Plot, pts: &[(f64, f64)], width: f32, color: Color) {
    let (min, max) = p.rect;
    let at: Vec<Vec2> = pts
        .iter()
        .map(|&(x, y)| {
            let x = p.px_x(x);
            // (minus infinity is far below, plus infinity far above, and no number nowhere)
            let y = if y.is_nan() { f32::NAN } else { p.px_y(y) };
            Vec2::new(x, if y.is_nan() { max.y } else { y })
        })
        .collect();
    let inside = |q: Vec2| q.y >= min.y && q.y <= max.y && q.x >= min.x - 0.01 && q.x <= max.x + 0.01;
    let n = at.len();
    let out: Vec<(Vec3, f32)> = (0..n)
        .map(|i| {
            let q = at[i];
            let near = (i > 0 && inside(at[i - 1])) || (i + 1 < n && inside(at[i + 1]));
            let w = if inside(q) || near { width } else { 0.0 };
            let kept = Vec2::new(if q.x.is_finite() { q.x.max(min.x).min(max.x) } else { min.x }, q.y.max(min.y).min(max.y));
            (f.hud(kept), w)
        })
        .collect();
    let batch = plot::apart(f, kind, false);
    batch.brk();
    for (pos, w) in out {
        batch.push(pos, w, color);
    }
    batch.brk();
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
/// space), none of them below `bottom`. Returns the height taken.
fn wrap_px(f: &mut Frame, at: Vec2, width: f32, bottom: f32, text: &str, size: f32, color: Color) -> f32 {
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
    let step = size + 5.0;
    let mut used = 0.0;
    for line in &lines {
        if at.y + used + size > bottom + 1.0 {
            break;
        }
        f.text_px(at + Vec2::new(0.0, used), line.as_str(), size, color, Align2::LEFT_TOP);
        used += step;
    }
    used
}

/// The first of `options` that fits in `room` logical pixels at `size` (the last if none does).
fn pick(f: &Frame, options: &[String], room: f32, size: f32) -> String {
    options.iter().find(|t| f.text_size(t, size).x <= room).or(options.last()).cloned().unwrap_or_default()
}

/// A draggable dot `size` pixels across at a point of the screen: where the pointer has
/// taken it (not kept inside anything), while it is being dragged.
fn grip(f: &mut Frame, id: u32, at: Vec2, size: f32, color: Color) -> Option<Vec2> {
    let mut p = f.cam.pick_ground(at * f.px)?;
    if f.drag3_sized(id, &mut p, Some(Vec3::Z), color, size) {
        on_screen(f, p)
    } else {
        None
    }
}

/// Round values along an axis, chosen from where it is going (`goal`, over `pixels`:
/// one about every sixty) with three more beyond each end, so that they stay the same
/// while it glides there: each line and number keeps its place in its batch, sliding
/// with the axis, hidden while it is off the chart. (Chosen afresh from the axis as
/// drawn, a value would come in at one end as another went out at the other, and every
/// line would seem to `eidos motion` to have jumped to its neighbour's place.) The
/// step, and the values.
fn steady(goal: [f64; 2], pixels: f32) -> (f64, Vec<f64>) {
    let span = goal[1] - goal[0];
    if !(span.is_finite() && span > 0.0 && pixels > 0.0) {
        return (1.0, Vec::new());
    }
    let step = nice_step((span / (pixels as f64 / 60.0).max(1.0)) as f32) as f64;
    let first = (goal[0] / step).floor() as i64 - 3;
    let last = ((goal[1] / step).ceil() as i64 + 3).min(first + 70);
    (step, (first..=last).map(|k| k as f64 * step).collect())
}

/// A tick's number, with the digits its step needs.
fn tick_name(v: f64, step: f64) -> String {
    num(v + 0.0, (-step.log10().floor()).max(0.0).min(6.0) as usize)
}

/// Where a tick goes along the screen, and whether it is on the chart (`lo` to `hi`).
/// One off the chart waits, unseen, just beyond its edge.
fn tick_at(px: f32, lo: f32, hi: f32) -> (f32, bool) {
    let seen = px >= lo - 0.5 && px <= hi + 0.5;
    (if px.is_finite() { px.max(lo - 12.0).min(hi + 12.0) } else { lo }, seen)
}

/// The line along the bottom of a chart, its ticks and their numbers (see [`steady`]),
/// and the axis's name under the middle (`""` for none).
fn x_axis(f: &mut Frame, p: &Plot, goal: [f64; 2], name: &str, size: f32) {
    if p.tiny() {
        return;
    }
    let (min, max) = p.rect;
    plot::line_of(f, AXES, Vec2::new(min.x, max.y), max, 1.0, pal::DIM.a(0.8));
    let (step, ticks) = steady(goal, p.width());
    for v in ticks {
        let (x, seen) = tick_at(p.px_x(v), min.x, max.x);
        let (line, words) = if seen { (pal::DIM.a(0.8), pal::DIM) } else { (Color::CLEAR, Color::CLEAR) };
        plot::line_of(f, AXES, Vec2::new(x, max.y), Vec2::new(x, max.y + 4.0), 1.0, line);
        f.text_px(Vec2::new(x, max.y + 6.0), tick_name(v, step), size, words, Align2::CENTER_TOP);
    }
    if !name.is_empty() {
        f.text_px(Vec2::new((min.x + max.x) * 0.5, max.y + 8.0 + size), name, size, pal::DIM, Align2::CENTER_TOP);
    }
}

/// The line up the left of a chart, its ticks and their numbers, and the axis's name
/// above its top (`""` for none).
fn y_axis(f: &mut Frame, p: &Plot, goal: [f64; 2], name: &str, size: f32) {
    if p.tiny() {
        return;
    }
    let (min, max) = p.rect;
    plot::line_of(f, AXES, min, Vec2::new(min.x, max.y), 1.0, pal::DIM.a(0.8));
    let (step, ticks) = steady(goal, p.height() * 1.6);
    for v in ticks {
        let (y, seen) = tick_at(p.px_y(v), min.y, max.y);
        let (line, words) = if seen { (pal::DIM.a(0.8), pal::DIM) } else { (Color::CLEAR, Color::CLEAR) };
        plot::line_of(f, AXES, Vec2::new(min.x - 4.0, y), Vec2::new(min.x, y), 1.0, line);
        f.text_px(Vec2::new(min.x - 6.0, y), tick_name(v, step), size, words, Align2::RIGHT_CENTER);
    }
    if !name.is_empty() {
        f.text_px(Vec2::new(min.x - 30.0, min.y - 6.0), name, size, pal::DIM, Align2::LEFT_BOTTOM);
    }
}

/// Faint lines across a chart at the ticks of its y axis.
fn grid_y(f: &mut Frame, p: &Plot, goal: [f64; 2]) {
    if p.tiny() {
        return;
    }
    let (min, max) = p.rect;
    for v in steady(goal, p.height() * 1.6).1 {
        let (y, seen) = tick_at(p.px_y(v), min.y, max.y);
        plot::line_of(f, AXES, Vec2::new(min.x, y), Vec2::new(max.x, y), 1.0, if seen { pal::DIM.a(0.14) } else { Color::CLEAR });
    }
}

/// A short horizontal bar with an upright tick at each end (an interval), in screen space.
fn interval_bar(f: &mut Frame, kind: u8, a: f32, b: f32, y: f32, width: f32, color: Color) {
    plot::line_of(f, kind, Vec2::new(a, y), Vec2::new(b, y), width, color);
    for x in [a, b] {
        plot::line_of(f, kind, Vec2::new(x, y - 4.0), Vec2::new(x, y + 4.0), 1.5, color);
    }
}

/// The charts as drawn in this frame.
struct Charts {
    a: Plot,
    b: Plot,
    c: Plot,
    /// The likelihood itself is drawn in units of 10 to this power.
    k10: i32,
    /// The log-likelihood at `SAMPLES` points across the likelihood chart (one parameter).
    curve: Vec<f64>,
}

impl Scene for Likelihood {
    fn view(&self) -> View {
        View::Flat { center: Vec2::ZERO, half_height: 1.0 }
    }

    fn settings(&self, s: &mut ViewSettings) {
        s.grid = false;
        s.bloom = s.bloom.min(0.2);
    }

    fn animated(&self) -> bool {
        self.moving || self.held != Held::Nothing
    }

    fn pointer_hint(&self) -> Option<&'static str> {
        Some(if self.model == Model::Coin {
            "the purple mark: drag it along the curve    a toss: click to turn it over"
        } else {
            "the purple mark: drag it    a cyan point: drag it, or off the bottom to take it away    the top chart: click to add a point"
        })
    }

    fn ui(&mut self, ui: &mut egui::Ui, _: &Gfx) {
        ui::note(ui, r"A model says how probable data are, given its parameter $\theta$. Turn that round: keep the data actually seen, and ask of each value of $\theta$ how probable the model makes them. That is the likelihood of $\theta$. For values that come independently, it is the product of the model's density at each:");
        ui::equation(ui, r"L(\theta) = \prod_{i=1}^{n} f(x_i \mid \theta), \qquad \ell(\theta) = \ln L(\theta) = \sum_{i=1}^{n} \ln f(x_i \mid \theta)");
        ui::note(ui, r"It is not a probability of $\theta$: over the values of $\theta$ it need not add up to 1. The maximum likelihood estimate $\hat\theta$ is the value that makes the data most probable: the peak. The logarithm turns the product into a sum: where a product of many small numbers vanishes towards 0, the sum stays a number of a handy size, with its peak in the same place. How sharp the peak is says how sure the estimate is.");

        ui::section(ui, "Things to try");
        for (k, trial) in TRIALS.iter().enumerate() {
            if ui::experiment(ui, trial.0, trial.1) {
                self.set_up(k);
            }
        }

        ui::section(ui, "The model");
        let names: Vec<&str> = MODELS.iter().map(|m| m.name()).collect();
        let mut m = self.model.index();
        if ui::choice(ui, &names, &mut m) {
            self.model = MODELS[m];
            self.held = Held::Nothing;
            self.held_range = None;
        }
        ui::note(
            ui,
            match self.model {
                Model::Coin => r"Tosses of a coin, heads with chance $p$: a toss has the chance $p$ if it is heads and $1 - p$ if tails, so $L(p) = p^h (1 - p)^{n - h}$ for $h$ heads in $n$ tosses.",
                Model::Mean => r"Values from a bell (a normal distribution) of known spread $\sigma$, its centre $\mu$ unknown: $f(x \mid \mu) = e^{-(x - \mu)^2 / 2\sigma^2} / (\sigma\sqrt{2\pi})$.",
                Model::Both => r"Values from a bell, its centre $\mu$ and its spread $\sigma$ both unknown: the likelihood is a surface over the two.",
                Model::Rate => r"Waiting times from an exponential distribution of rate $\lambda$ (events per unit of time): $f(x \mid \lambda) = \lambda e^{-\lambda x}$ for $x \ge 0$.",
                Model::Centre => r"Values from a Cauchy curve of width 1, its centre $\theta$ unknown: $f(x \mid \theta) = 1 / (\pi(1 + (x - \theta)^2))$. Its tails are so heavy that it has no mean.",
            },
        );
        if self.model == Model::Mean {
            let mut sd = self.sd as f32;
            if ui::slider(ui, r"the known spread $\sigma$", &mut sd, 0.2..=3.0) {
                self.sd = sd as f64;
                if self.set().drawn {
                    self.redraw();
                }
            }
        }

        ui::section(ui, "The data");
        let model = self.model;
        if model == Model::Coin {
            let s = self.set_mut();
            let mut n = s.tosses.len();
            let mut heads = s.tosses.iter().filter(|t| **t).count();
            let mut again = ui::slider_int(ui, "tosses", &mut n, 0..=MOST);
            again |= ui::slider_int(ui, "heads", &mut heads, 0..=n);
            if again {
                (s.tosses, s.n, s.drawn) = (pattern(heads.min(n), n), n, false);
            }
            ui::note(ui, "Or click a toss on the picture to turn it over.");
            let mut p = s.truth.a as f32;
            let mut draw = ui::slider(ui, r"a coin's true $p$", &mut p, 0.01..=0.99);
            s.truth.a = p as f64;
            ui.horizontal(|ui| {
                draw |= ui.button("toss it").clicked();
                if ui.button("again").clicked() {
                    s.seed = s.seed.wrapping_add(1);
                    draw = true;
                }
            });
            if draw {
                s.n = s.tosses.len().max(1);
                self.redraw();
            }
            ui::note(ui, r"Tosses that coin as many times as there are tosses now; its $p$ is then shown in green.");
        } else {
            let sd = self.sd;
            let s = &mut self.sets[model.index()];
            let mut draw = ui::slider_int(ui, "values to draw", &mut s.n, 0..=MOST);
            let mut tail = s.truth.tail as usize;
            draw |= ui::choice(ui, &["from the model itself", "from a long right tail"], &mut tail);
            s.truth.tail = tail == 1;
            if !s.truth.tail {
                let (mut a, mut b) = (s.truth.a as f32, s.truth.b as f32);
                draw |= match model {
                    Model::Rate => ui::slider_log(ui, r"the true rate $\lambda$", &mut a, 0.2..=5.0),
                    _ => ui::slider(ui, r"the true centre", &mut a, -3.0..=3.0),
                };
                if model == Model::Both {
                    draw |= ui::slider_log(ui, r"the true spread", &mut b, 0.2..=3.0);
                }
                (s.truth.a, s.truth.b) = (a as f64, b as f64);
            }
            if ui.button("draw again").clicked() {
                s.seed = s.seed.wrapping_add(1);
                draw = true;
            }
            if draw {
                s.redraw(model, sd);
            }
            ui::note(ui, "The truth they come from is shown in green until a value is moved, added or taken away by hand. On the picture: drag a value, drag it off the bottom of its chart to take it away, or click the chart to add one.");
        }

        ui::section(ui, "The value tried");
        // (the ranges as they are, not as they are drawn: an experiment just set up
        // has not been drawn yet, and a slider clamps its value to its range)
        let fit = self.fit_now();
        let (cx, ctop) = self.param_view(&fit, self.data_view(&fit).0);
        let th = self.set().theta;
        // (a range that is no range, or not a number, would break the slider)
        let span = |lo: f64, hi: f64, at: f64| if lo.is_finite() && hi.is_finite() && hi > lo { (lo, hi) } else { (at - 1.0, at + 1.0) };
        let (lo, hi) = span(cx[0].min(th[0]), cx[1].max(th[0]), th[0]);
        let (mut a, mut b) = (th[0] as f32, th[1] as f32);
        let mut moved = ui::slider(ui, &format!("${}$", model.symbol()), &mut a, lo as f32..=hi as f32);
        if model == Model::Both {
            let (lo, hi) = span((ctop * 0.01).min(th[1]), ctop.max(th[1]).max(0.02), th[1]);
            moved |= ui::slider(ui, r"$\sigma$", &mut b, lo.max(1e-3) as f32..=hi.max(2e-3) as f32);
        }
        if moved {
            let kept = self.keep_theta([a as f64, b as f64], [lo, hi], ctop.max(th[1]).max(0.02));
            self.set_mut().theta = kept;
        }
        ui::note(ui, "The purple mark: the model is drawn at this value, and the data's heights under it. Drag it on the picture too.");
        let mut plain = self.plain as usize;
        if ui::choice(ui, &SWITCH, &mut plain) {
            self.plain = plain == 1;
        }
        ui::note(ui, "The likelihood itself, a product of many numbers that are mostly below 1, is tiny: it is drawn in units of a power of ten. Its logarithm has its peak in the same place.");

        ui::section(ui, "Reading the picture");
        ui::key(
            ui,
            &[
                (DATA, "cyan: the data (dots on the axis; for a coin, each toss, filled for heads)"),
                (ASSUMED, r"purple: the value tried, $\theta$, and everything at it: the model's density, the heights of the data under it, their logarithms added"),
                (pal::TEXT, "white: the log-likelihood (or likelihood) of every value"),
                (ESTIMATE, r"yellow: the peak $\hat\theta$, the density there, and the likelihood interval on the dashed line 1.92 below the peak (with two unknowns, the curve 1.92 below it)"),
                (OTHER, r"orange: the parabola with the peak's curvature, and $\hat\theta \pm 1.96\,\text{SE}$ where it crosses the same line"),
                (TRUTH, "green: the truth the data were drawn from"),
                (ALARM, "pink: a point that will be taken away when it is let go"),
            ],
        );
        ui::section(ui, "Words");
        ui::term(ui, "Density:", "for a quantity that can take any value, how probable values near a point are, per unit: the height of the model's curve. (For a coin's toss, the density is its chance.)");
        ui::term(ui, "Likelihood:", r"$L(\theta)$, the probability (or density) the model gives the data actually seen, read as a function of $\theta$. Not a distribution over $\theta$.");
        ui::term(ui, "Log-likelihood:", r"$\ell(\theta) = \ln L(\theta)$, a sum over the data. Only differences in it matter: adding the same number to every value changes nothing.");
        ui::term(ui, "Maximum likelihood estimate:", r"$\hat\theta$, the value with the highest likelihood.");
        ui::term(ui, "Information:", r"the curvature at the peak, $-\ell''(\hat\theta)$: a sharp peak is a sure estimate. It grows in proportion to the data: for the coin, the bell and the rate it is exactly $n$ times the Fisher information (the curvature one value gives on average) at $\hat\theta$; for the Cauchy curve it only comes near that, and varies from sample to sample.");
        ui::term(ui, "Standard error:", r"how far an estimate typically is from the truth. Here $\text{SE} = 1/\sqrt{\text{information}}$: a good approximation with much data, when the curve near its peak is close to its parabola.");
        ui::term(ui, "Likelihood interval:", r"the values whose log-likelihood is within 1.92 of the peak (likelihood at least 0.15 of it). If the model is right, in large samples it covers the true value about 95 times in 100, as $\hat\theta \pm 1.96\,\text{SE}$ does (Wilks's theorem); in small ones it keeps the curve's own shape.");

        ui::section(ui, "Numbers");
        let fit = self.fit_now();
        let s = self.set();
        let o = self.obs();
        ui::readout(ui, "values", format!("{}", o.n() as usize));
        let sym = model.symbol();
        let at = o.ll(s.theta);
        ui::readout_colored(ui, r"at the value tried: $\ell$", format!("${}$", ell(at)), ASSUMED);
        match &fit {
            Fit::Flat => ui::readout(ui, "peak", "none: flat"),
            Fit::Unbounded => ui::readout(ui, "peak", "none: no maximum"),
            Fit::One(t) => {
                ui::readout_colored(ui, &format!(r"peak $\hat{{{sym}}}$"), format!("${}$", val(t.at)), ESTIMATE);
                match (t.info, t.se()) {
                    (Some(info), Some(se)) => {
                        ui::readout(ui, r"curvature $-\ell''$", format!("${}$", val(info)));
                        ui::readout(ui, "standard error", format!("${}$", figures(se, 3)));
                    }
                    _ => ui::readout(ui, "curvature", "none: the peak is at the edge"),
                }
            }
            Fit::Two(t) => {
                ui::readout_colored(ui, r"peak $\hat\mu$, $\hat\sigma$", format!("${}, {}$", val(t.mu), val(t.sigma)), ESTIMATE);
                ui::readout(ui, r"standard errors", format!("${}, {}$", val(1.0 / t.info[0].sqrt()), val(1.0 / t.info[1].sqrt())));
            }
        }
        ui::see_also(
            ui,
            &[
                ("estimators", "judging an estimate by how it varies from sample to sample"),
                ("intervals", "how often an interval covers the truth"),
                ("bayes", "the likelihood times a prior"),
                ("entropy", "the average log-likelihood is a cross-entropy"),
            ],
        );
    }

    fn frame(&mut self, f: &mut Frame) {
        let lay = Layout::of(f.view_size(), f.top_inset);
        // a drag that has ended: a point let go below its chart is taken away
        if !f.handles.dragging() && self.held != Held::Nothing {
            self.release();
        }
        let fit = self.fit_now();
        let goals = self.goals(&fit);
        let key = (self.model, lay.view);
        if self.looked != Some(key) {
            self.look_theta.snap(&goals[0]);
            self.look_range.snap(&goals[1]);
            self.look_data.snap(&goals[2]);
            self.looked = Some(key);
        }
        let dt = f.real_dt;
        let mut moving = self.look_theta.follow(&goals[0], dt, false);
        moving |= self.look_range.follow(&goals[1], dt, false);
        moving |= self.look_data.follow(&goals[2], dt, true);
        self.moving = moving;
        if lay.tiny {
            return;
        }
        let charts = self.charts(&lay, false);
        // the visitor's hand: what it changes is drawn at once where the hand is
        let changed = self.handle(f, &lay, &charts);
        let charts = if changed {
            let fit = self.fit_now();
            let goals = self.goals(&fit);
            match self.held {
                Held::Theta => {
                    self.look_theta.snap(&goals[0]);
                    self.look_range.snap(&goals[1]);
                }
                Held::Point(_) => {
                    self.look_data.snap(&goals[2]);
                    self.look_range.snap(&goals[1]);
                }
                Held::Nothing => {}
            }
            self.charts(&lay, false)
        } else {
            charts
        };
        let goal = self.charts(&lay, true);
        let shown = self.obs_shown();
        let sfit = self.fit_shown.of(&shown);
        let fit = self.fit_now();
        if !lay.compact {
            self.draw_data(f, &lay, &charts, &goal, &shown, &sfit);
            self.draw_ladder(f, &lay, &charts, &goal, &shown, &sfit);
            self.draw_facts(f, &lay, &fit);
        }
        if self.model == Model::Both {
            self.draw_surface(f, &lay, &charts, &goal, &shown, &sfit);
        } else {
            self.draw_curve(f, &lay, &charts, &goal, &shown, &sfit);
        }
        // new round values on an axis: make sure the axes' batch and the labels are not
        // as many as in the last frame, so that nothing takes the new ones for the old
        // ones moved
        let mut ticks = Vec::new();
        for p in [&goal.a, &goal.b, &goal.c] {
            ticks.extend(steady(p.x, p.width()).1);
            ticks.push(f64::MAX);
            ticks.extend(steady(p.y, p.height() * 1.6).1);
            ticks.push(f64::MAX);
        }
        let count = |f: &mut Frame| (plot::apart(f, AXES, false).pts.iter().filter(|p| p.width >= 0.0).count(), f.dl.labels.len());
        if ticks != self.ticks {
            let (axes, labels) = count(f);
            if axes == self.drawn_last.0 {
                let far = f.hud(Vec2::ZERO);
                plot::apart(f, AXES, false).segment(far, far, 1.0, Color::CLEAR);
            }
            if labels == self.drawn_last.1 {
                f.text_px(Vec2::ZERO, "", 1.0, Color::CLEAR, Align2::LEFT_TOP);
            }
        }
        self.drawn_last = count(f);
        self.ticks = ticks;
    }
}

impl Likelihood {
    /// The charts as drawn in this frame, or (`goal`) as they will be once the picture
    /// has glided to where it is going.
    fn charts(&mut self, lay: &Layout, goal: bool) -> Charts {
        let (range, obs, data) = if goal {
            let fit = self.fit_now();
            let [_, range, data] = self.goals(&fit);
            (range, self.obs(), data)
        } else {
            (self.look_range.shown.clone(), self.obs_shown(), self.look_data.shown.clone())
        };
        // what the numbers of the range glide are
        let get = |i: usize| range.get(i).copied().unwrap_or(0.0) as f64;
        let (ax, atop, cx, ctop) = ([get(0), get(1)], get(2), [get(3), get(4)], get(5));
        let (top, depth, b_range) = (get(9), get(10), [get(11), get(12)]);
        let a = if self.model == Model::Coin { Plot::new(lay.a, ax, [0.0, 1.12]) } else { Plot::new(lay.a, ax, [0.0, atop]) };
        let k10 = (top / LN_10).floor().max(-300.0) as i32;
        let curve = if self.model == Model::Both || goal { Vec::new() } else { self.curve_across(&obs, cx) };
        let c = if self.model == Model::Both {
            Plot::new(lay.c, cx, [0.0, ctop])
        } else if self.plain {
            Plot::new(lay.c, cx, [0.0, 1.12 * (top - k10 as f64 * LN_10).exp()])
        } else {
            Plot::new(lay.c, cx, [top - depth, top + 0.1 * depth])
        };
        let b = Plot::new(lay.b, [0.0, data.len().max(1) as f64], b_range);
        Charts { a, b, c, k10, curve }
    }

    /// The log-likelihood of the data `o` at `SAMPLES` points from `x[0]` to `x[1]`
    /// (the last two kept: the picture as drawn and where it is going).
    fn curve_across(&mut self, o: &Obs, x: [f64; 2]) -> Vec<f64> {
        if let Some(k) = self.curves.iter().position(|(kept, r, _)| kept == o && *r == x) {
            let found = self.curves.remove(k);
            let v = found.2.clone();
            self.curves.insert(0, found);
            return v;
        }
        let v: Vec<f64> = (0..SAMPLES).map(|i| o.ll([x[0] + (x[1] - x[0]) * i as f64 / (SAMPLES - 1) as f64, 0.0])).collect();
        self.curves.insert(0, (o.clone(), x, v.clone()));
        self.curves.truncate(2);
        v
    }

    /// The visitor's hand on the picture: the value tried, the data, the switch. Says
    /// whether the data or the value tried changed.
    fn handle(&mut self, f: &mut Frame, lay: &Layout, ch: &Charts) -> bool {
        let mut changed = false;
        let pointer = plot::pointer(f);
        // the switch (where there is room for it beside a title)
        for (k, r) in lay.switch.iter().enumerate().filter(|_| lay.view.x >= 300.0) {
            if plot::button(f, *r, SWITCH[k], lay.small, self.plain == (k == 1)) {
                self.plain = k == 1;
            }
        }
        let (cx, ctop) = (ch.c.x, ch.c.y[1]);
        // the value tried
        let th = self.theta_shown();
        let at = if self.model == Model::Both {
            ch.c.at(th[0], th[1])
        } else {
            let shown = self.obs_shown();
            let y = self.y_of(shown.ll(th), ch.k10);
            ch.c.at(th[0], y)
        };
        let (min, max) = ch.c.rect;
        let at = if at.is_finite() { at.max(min).min(max) } else { Vec2::new(ch.c.px_x(th[0]).max(min.x).min(max.x), max.y) };
        // (a ring round the mark keeps it in sight when it sits on the peak's dot)
        hud_circle(f, at, 9.0, 1.6, ASSUMED);
        if let Some(to) = grip(f, 1, at, 13.0, ASSUMED) {
            let to = to.max(min).min(max);
            let kept = self.keep_theta([ch.c.x_at(to.x), ch.c.y_at(to.y)], cx, ctop);
            self.set_mut().theta = kept;
            self.held = Held::Theta;
            changed = true;
        } else if f.handles.dragging() && self.held == Held::Theta {
            // (held still)
        }
        // a click on the likelihood chart moves the value tried there
        if pointer.is_some_and(|p| plot::within(p, ch.c.rect, 0.0)) {
            f.handles.claim_clicks();
            if f.clicked_canvas() {
                if let Some(p) = pointer {
                    let kept = self.keep_theta([ch.c.x_at(p.x), ch.c.y_at(p.y)], cx, ctop);
                    self.set_mut().theta = kept;
                }
            }
        }
        if lay.compact {
            return changed;
        }
        let a = ch.a;
        let over_a = pointer.is_some_and(|p| plot::within(p, a.rect, 0.0));
        if self.model == Model::Coin {
            // a toss clicked turns over
            if over_a {
                f.handles.claim_clicks();
                if let (Some(p), true) = (pointer, f.ptr.clicked) {
                    let i = (a.x_at(p.x) - 0.5).floor();
                    let s = self.set_mut();
                    if i >= 0.0 && (i as usize) < s.tosses.len() {
                        let i = i as usize;
                        s.tosses[i] = !s.tosses[i];
                        s.drawn = false;
                    }
                }
            }
            return changed;
        }
        // the data points
        let shown = self.look_data.shown.clone();
        let n = shown.len();
        let size = if n <= 20 {
            12.0
        } else if n <= 60 {
            9.0
        } else {
            7.0
        };
        let mut moved = None;
        for (i, &x) in shown.iter().enumerate() {
            let px = a.px_x(x as f64);
            if !(px >= a.rect.0.x - 1.0 && px <= a.rect.1.x + 1.0) {
                continue;
            }
            let color = if self.held == Held::Point(i) && self.removing { ALARM } else { DATA };
            if let Some(to) = grip(f, 100 + i as u32, Vec2::new(px, a.rect.1.y), size, color) {
                moved = Some((i, to));
            }
        }
        if let Some((i, to)) = moved {
            if self.held_range.is_none() {
                let r = &self.look_range;
                self.held_range = Some([r.get(0), r.get(1), r.get(2)]);
            }
            self.held = Held::Point(i);
            self.removing = to.y > a.rect.1.y + 28.0;
            let x = a.x_at(to.x.max(a.rect.0.x).min(a.rect.1.x));
            let x = if self.model == Model::Rate { x.max(LEAST_WAIT) } else { x };
            let s = self.set_mut();
            if i < s.xs.len() {
                s.xs[i] = x;
                s.drawn = false;
            }
            changed = true;
        }
        // a click on the data chart adds a point there
        if over_a {
            f.handles.claim_clicks();
            if let (Some(p), true) = (pointer, f.clicked_canvas()) {
                let model = self.model;
                let s = self.set_mut();
                if s.xs.len() < MOST {
                    let x = a.x_at(p.x);
                    s.xs.push(if model == Model::Rate { x.max(LEAST_WAIT) } else { x });
                    s.drawn = false;
                }
            }
        }
        changed
    }

    /// A log-likelihood as the likelihood chart draws it: itself, or the likelihood in
    /// units of 10 to the power `k10`.
    fn y_of(&self, ll: f64, k10: i32) -> f64 {
        if self.plain && self.model != Model::Both {
            (ll - k10 as f64 * LN_10).exp()
        } else {
            ll
        }
    }

    /// The first chart: the data, the model's density at the value tried, at the peak
    /// and the truth, and a bar from each point up to the purple curve.
    fn draw_data(&self, f: &mut Frame, lay: &Layout, ch: &Charts, goal: &Charts, shown: &Obs, sfit: &Fit) {
        let p = &ch.a;
        if p.tiny() {
            return;
        }
        let model = self.model;
        let th = self.theta_shown();
        let data = &self.look_data.shown;
        let sym = model.symbol();
        let n = data.len();
        p.backing(f, 42.0, 32.0, 46.0, 10.0);
        let tried = if model == Model::Both { format!(r"$\mu = {}$, $\sigma = {}$", val(th[0]), val(th[1])) } else { format!("${sym} = {}$", val(th[0])) };
        let room = p.rect.1.x - p.rect.0.x + 30.0;
        let title = if model == Model::Coin {
            [format!("{n} tosses, and the chance of each if {tried}"), format!("the chance of each toss if {tried}"), format!("if {tried}")]
        } else {
            [format!("{n} values, and the model's density if {tried}"), format!("the density if {tried}"), format!("if {tried}")]
        };
        let title = pick(f, &title, room * if model == Model::Coin { 1.0 } else { 0.72 }, lay.big);
        p.title(f, &title, lay.big);
        grid_y(f, p, goal.a.y);
        if model == Model::Coin {
            y_axis(f, p, goal.a.y, "chance", lay.small);
            let (min, max) = p.rect;
            plot::line_of(f, AXES, Vec2::new(min.x, max.y), max, 1.0, pal::DIM.a(0.8));
            let pr = th[0];
            let w = (p.width() / n.max(1) as f32).max(0.5);
            let gap = if w > 6.0 { (w * 0.18).min(3.0) } else { 0.0 };
            for (i, &h) in data.iter().enumerate() {
                let h = h as f64;
                let height = h * pr + (1.0 - h) * (1.0 - pr);
                p.bar(f, i as f64 + 0.5, i as f64 + 1.5, 0.0, height, gap, ASSUMED.a(0.72));
            }
            // each toss under its bar: a filled dot for heads, a ring for tails
            let dot = (w * 0.6).max(2.5).min(11.0);
            for (i, &h) in data.iter().enumerate() {
                let at = Vec2::new(p.px_x(i as f64 + 1.0), max.y + dot * 0.5 + 4.0);
                plot::disc(f, at, dot, DATA.a(h));
                hud_circle(f, at, dot * 0.5, 1.3, DATA.a(0.9));
            }
            if n == 0 {
                f.text_px((min + max) * 0.5, "no tosses yet", lay.text, pal::DIM, Align2::CENTER_CENTER);
            } else if !lay.narrow {
                f.text_px(Vec2::new((min.x + max.x) * 0.5, max.y + dot + 8.0), "each toss: filled for heads, a ring for tails", lay.small, pal::DIM, Align2::CENTER_TOP);
            }
            // what the bars' heights are
            let words = if (pr - (1.0 - pr)).abs() < 0.12 {
                vec![(format!(r"heads: $p = {}$, tails: $1 - p = {}$", val(pr), val(1.0 - pr)), pr.max(1.0 - pr))]
            } else {
                vec![(format!(r"heads: $p = {}$", val(pr)), pr), (format!(r"tails: $1 - p = {}$", val(1.0 - pr)), 1.0 - pr)]
            };
            for (text, y) in words {
                let at = Vec2::new(max.x - 6.0, (p.px_y(y) - 3.0).max(min.y + 14.0));
                f.caption_px(at, text, lay.small, ASSUMED.mix(pal::WHITE, 0.35), Align2::RIGHT_BOTTOM);
            }
        } else {
            x_axis(f, p, goal.a.x, if model == Model::Rate { "waiting time" } else { "" }, lay.small);
            y_axis(f, p, goal.a.y, "density", lay.small);
            let xs: Vec<f64> = (0..SAMPLES).map(|i| p.x[0] + (p.x[1] - p.x[0]) * i as f64 / (SAMPLES - 1) as f64).collect();
            let curve = |f: &mut Frame, d: Dist, width: f32, color: Color| {
                let pts: Vec<(f64, f64)> = xs.iter().map(|&x| (x, d.pdf(x))).collect();
                kept_path(f, CURVES, p, &pts, width, color);
            };
            if let Some(d) = self.truth_shown().and_then(|t| t.dist(model, shown.sd)) {
                curve(f, d, 1.6, TRUTH.a(0.85));
            }
            if let Some(d) = peak_theta(sfit).and_then(|t| dist_at(model, t, shown.sd)) {
                curve(f, d, 1.6, ESTIMATE.a(0.8));
            }
            if let Some(d) = dist_at(model, th, shown.sd) {
                curve(f, d, 2.4, ASSUMED);
                // a bar from each point up to the purple curve
                for &x in data {
                    let x = x as f64;
                    let px = p.px_x(x);
                    if !(px >= p.rect.0.x - 0.5 && px <= p.rect.1.x + 0.5) {
                        continue;
                    }
                    let y = p.px_y(d.pdf(x)).max(p.rect.0.y);
                    let y = if y.is_finite() { y } else { p.rect.1.y };
                    plot::line_of(f, MARKS, Vec2::new(px, p.rect.1.y), Vec2::new(px, y), 2.0, ASSUMED.a(0.8));
                }
            }
            // what the other curves are
            let mut names = Vec::new();
            if peak_theta(sfit).is_some() {
                names.push((ESTIMATE, if model == Model::Both { "yellow: at the peak" } else { "yellow: the density at the peak" }));
            }
            if self.truth_shown().is_some() {
                names.push((TRUTH, if self.set().truth.tail { "green: the truth (a long tail)" } else { "green: the truth" }));
            }
            let mut x = p.rect.1.x;
            let y = p.rect.0.y - 24.0;
            for (color, text) in names.iter().rev() {
                let w = f.text_size(text, lay.small).x;
                if x - w < p.rect.0.x + f.text_size(&title, lay.big).x - 10.0 {
                    break;
                }
                f.text_px(Vec2::new(x, y), *text, lay.small, *color, Align2::RIGHT_BOTTOM);
                x -= w + 14.0;
            }
            if n == 0 {
                f.caption_px(Vec2::new((p.rect.0.x + p.rect.1.x) * 0.5, p.rect.1.y - 24.0), "no values yet: click here to add one", lay.text, pal::DIM, Align2::CENTER_BOTTOM);
            }
            if let Held::Point(_) = self.held {
                if self.removing {
                    f.caption_px(Vec2::new(p.rect.0.x + 8.0, p.rect.1.y - 8.0), "let go to take this value away", lay.small, ALARM, Align2::LEFT_BOTTOM);
                }
            }
        }
        // the product of the heights, as it is
        let o = self.obs();
        let at = o.ll(self.set().theta);
        let text = if model == Model::Coin {
            let p = self.set().theta[0];
            format!(r"multiplied: ${}^{{{}}} \times {}^{{{}}} = {}$", val(p), o.heads, val(1.0 - p), o.tails, exp_text(at))
        } else if lay.narrow {
            format!("multiplied: $L = {}$", exp_text(at))
        } else {
            format!("heights multiplied: $L = {}$", exp_text(at))
        };
        if model != Model::Coin {
            f.caption_px(Vec2::new(p.rect.1.x - 6.0, p.rect.0.y + 6.0), text, lay.small, ASSUMED.mix(pal::WHITE, 0.35), Align2::RIGHT_TOP);
        } else {
            f.caption_px(Vec2::new(p.rect.0.x + 6.0, p.rect.0.y + 6.0), text, lay.small, ASSUMED.mix(pal::WHITE, 0.35), Align2::LEFT_TOP);
        }
    }

    /// The second chart: the logarithms of the heights added up point by point, at the
    /// value tried and at the peak.
    fn draw_ladder(&self, f: &mut Frame, lay: &Layout, ch: &Charts, goal: &Charts, shown: &Obs, sfit: &Fit) {
        let p = &ch.b;
        if p.tiny() {
            return;
        }
        let data = &self.look_data.shown;
        let n = data.len();
        p.backing(f, 42.0, 30.0, 46.0, 10.0);
        let title = [
            "their logarithms, added one by one".to_string(),
            "their logarithms, added".to_string(),
            "logarithms added".to_string(),
        ];
        p.title(f, &pick(f, &title, p.width() + 30.0, lay.big), lay.big);
        grid_y(f, p, goal.b.y);
        x_axis(f, p, goal.b.x, if lay.narrow { "" } else if self.model == Model::Coin { "tosses so far" } else { "values so far" }, lay.small);
        y_axis(f, p, goal.b.y, "", lay.small);
        if n == 0 {
            f.text_px((p.rect.0 + p.rect.1) * 0.5, "nothing to add", lay.text, pal::DIM, Align2::CENTER_CENTER);
            return;
        }
        let th = self.theta_shown();
        let steps = |th: [f64; 2]| -> Vec<(f64, f64)> { self.ladder(data, th, shown.sd).into_iter().enumerate().map(|(k, v)| (k as f64, v)).collect() };
        let best = peak_theta(sfit).map(steps);
        if let Some(pts) = &best {
            kept_path(f, CURVES, p, pts, 1.4, ESTIMATE.a(0.75));
        }
        let pts = steps(th);
        kept_path(f, CURVES, p, &pts, 2.0, ASSUMED);
        if n <= 40 {
            for &(k, v) in pts.iter().skip(1) {
                p.dot(f, k, v, 4.5, ASSUMED);
            }
        }
        // where each ends: the log-likelihood (the peak's is the higher, and goes above)
        if let Some(&(k, v)) = pts.last() {
            let at = p.at(k, v).max(p.rect.0).min(p.rect.1);
            let text = format!(r"$\ell = {}$", ell(v));
            let w = f.text_size(&text, lay.small).x;
            f.text_placed("ladder end", self.tag(0), at, Vec2::new(-w * 0.5 - 6.0, 11.0), text, lay.small, ASSUMED.mix(pal::WHITE, 0.3));
        }
        if let Some(&(k, v)) = best.as_ref().and_then(|b| b.last()) {
            let at = p.at(k, v).max(p.rect.0).min(p.rect.1);
            let text = "at the peak".to_string();
            let w = f.text_size(&text, lay.small).x;
            f.text_placed("ladder end", self.tag(1), at, Vec2::new(-w * 0.5 - 6.0, -10.0), text, lay.small, ESTIMATE);
        }
    }

    /// The third chart, for one parameter: the (log-)likelihood of every value, its
    /// peak, the parabola there, the two intervals, the truth and the value tried.
    fn draw_curve(&self, f: &mut Frame, lay: &Layout, ch: &Charts, goal: &Charts, shown: &Obs, sfit: &Fit) {
        let p = &ch.c;
        if p.tiny() {
            return;
        }
        let model = self.model;
        let sym = model.symbol();
        let plain = self.plain;
        let k10 = ch.k10;
        p.backing(f, 42.0, 32.0, 46.0, 10.0);
        let room = lay.switch[0].0.x - p.rect.0.x + 22.0;
        let title = if plain {
            [format!(r"the likelihood $L({sym})$ of every value of ${sym}$"), format!(r"the likelihood $L({sym})$"), String::new()]
        } else {
            [format!(r"the log-likelihood $\ell({sym})$ of every value of ${sym}$"), format!(r"the log-likelihood $\ell({sym})$"), String::new()]
        };
        p.title(f, &pick(f, &title, room, lay.big), lay.big);
        grid_y(f, p, goal.c.y);
        x_axis(f, p, goal.c.x, &format!("${sym}$"), lay.small);
        let name = if !plain {
            r"$\ell$".to_string()
        } else if k10 == 0 {
            r"$L$".to_string()
        } else {
            format!(r"$L$, in units of $10^{{{k10}}}$")
        };
        y_axis(f, p, goal.c.y, &name, lay.small);
        let xs: Vec<f64> = (0..SAMPLES).map(|i| p.x[0] + (p.x[1] - p.x[0]) * i as f64 / (SAMPLES - 1) as f64).collect();
        let y_of = |ll: f64| self.y_of(ll, k10);
        let (min, max) = p.rect;
        let th = self.theta_shown();
        let mut wall = Vec::new();
        match sfit {
            Fit::One(t) => {
                let level = t.ll - drop95();
                let ly = p.px_y(y_of(level));
                p.hline(f, MARKS, y_of(level), 1.2, pal::DIM.a(0.9), true);
                let below = if plain { format!("{} of the peak", val((-drop95()).exp())) } else { format!("{} below the peak", val(drop95())) };
                // at an end of the line where the curve is below it
                let room = f.text_size(&below, lay.small).x + 16.0;
                let left_free = t.pieces.first().is_none_or(|q| p.px_x(q[0]) > min.x + room);
                let half = 0.5 * (room - 16.0);
                let along = if left_free { 6.0 + half } else { max.x - min.x - 6.0 - half };
                f.text_placed("level", self.tag(0), Vec2::new(min.x, ly), Vec2::new(along, -3.0 - 0.6 * lay.small), below, lay.small, pal::DIM);
                if let Some(info) = t.info {
                    let pts: Vec<(f64, f64)> = xs.iter().map(|&x| (x, y_of(t.ll - 0.5 * info * (x - t.at) * (x - t.at)))).collect();
                    kept_path(f, CURVES, p, &pts, 1.5, OTHER.a(0.65));
                }
                let pts: Vec<(f64, f64)> = xs.iter().zip(&ch.curve).map(|(&x, &v)| (x, y_of(v))).collect();
                kept_path(f, CURVES, p, &pts, 2.3, pal::TEXT);
                // the likelihood interval on the level line, and the parabola's under it
                let clamp_x = |x: f64| p.px_x(x).max(min.x).min(max.x);
                for piece in &t.pieces {
                    interval_bar(f, MARKS, clamp_x(piece[0]), clamp_x(piece[1]), ly, 4.0, ESTIMATE);
                }
                if let Some([w0, w1]) = t.wald() {
                    let y = ly + 11.0;
                    interval_bar(f, MARKS, clamp_x(w0), clamp_x(w1), y, 3.0, OTHER);
                    // beyond the range a value can take
                    for (end, at) in [(w0, min.x), (w1, max.x)] {
                        if (end < p.x[0] && p.px_x(end) < min.x) || (end > p.x[1] && p.px_x(end) > max.x) {
                            let d = if at == min.x { -1.0 } else { 1.0 };
                            hud_arrow(f, Vec2::new(at - d * 12.0, y), Vec2::new(at + d * 2.0, y), 2.0, OTHER);
                            let align = if at == min.x { Align2::LEFT_TOP } else { Align2::RIGHT_TOP };
                            f.text_px(Vec2::new(at + d * -4.0, y + 5.0), format!("to {}", val(end)), lay.small, OTHER, align);
                        }
                    }
                }
                // the peak
                let top = p.at(t.at, y_of(t.ll));
                plot::disc(f, top.max(min).min(max), 8.0, ESTIMATE);
                let label = format!(r"$\hat{{{sym}}} = {}$", val(t.at));
                f.text_placed("peak", self.tag(0), top.max(min).min(max), Vec2::new(0.0, -14.0), label, lay.text, ESTIMATE);
                wall.push(top);
            }
            _ => {
                let pts: Vec<(f64, f64)> = xs.iter().map(|&x| (x, y_of(0.0))).collect();
                kept_path(f, CURVES, p, &pts, 2.3, pal::TEXT);
                f.text_px(Vec2::new((min.x + max.x) * 0.5, min.y + 16.0), "no data: the same for every value, the data say nothing yet", lay.text, pal::DIM, Align2::CENTER_TOP);
            }
        }
        // the truth
        if let Some([a, _]) = self.truth_shown().and_then(|t| t.target(model)) {
            p.vline(f, MARKS, a, 1.5, TRUTH.a(0.8), true);
            let x = p.px_x(a);
            if x >= min.x && x <= max.x {
                let words = if self.set().truth.tail { "closest to the truth" } else { "the truth" };
                f.text_px(Vec2::new(x + 4.0, max.y - 26.0), words, lay.small, TRUTH, Align2::LEFT_BOTTOM);
            }
        }
        // the value tried: a line down to the axis, and its mark (the handle, drawn by
        // `handle`), and what it is
        let y = y_of(shown.ll(th));
        let at = p.at(th[0], y);
        let x = p.px_x(th[0]).max(min.x).min(max.x);
        let mark = if at.is_finite() { at.max(min).min(max) } else { Vec2::new(x, max.y) };
        plot::line_of(f, MARKS, Vec2::new(x, max.y), mark, 1.2, ASSUMED.a(0.7));
        let o = self.obs();
        let now = o.ll(self.set().theta);
        let text = if plain { format!("${sym} = {}$, $L = {}$", val(self.set().theta[0]), exp_text(now)) } else { format!("${sym} = {}$, $\\ell = {}$", val(self.set().theta[0]), ell(now)) };
        let w = f.text_size(&text, lay.small).x;
        // a tag at the foot of the line (kept on the chart, so it never has to change sides)
        let at = Vec2::new(x.max(min.x + 0.5 * w + 4.0).min(max.x - 0.5 * w - 4.0), max.y - 12.0);
        f.caption_px(at, text, lay.small, ASSUMED.mix(pal::WHITE, 0.3), Align2::CENTER_CENTER);
    }

    /// The third chart, for the bell with both unknown: the surface over its centre
    /// and spread, as level curves.
    fn draw_surface(&self, f: &mut Frame, lay: &Layout, ch: &Charts, goal: &Charts, shown: &Obs, sfit: &Fit) {
        let p = &ch.c;
        if p.tiny() {
            return;
        }
        let plain = self.plain;
        p.backing(f, 42.0, 32.0, 46.0, 10.0);
        let room = lay.switch[0].0.x - p.rect.0.x + 22.0;
        let (whole, short) = match (plain, sfit) {
            (_, Fit::Unbounded) => (r"the log-likelihood $\ell(\mu, \sigma)$: it has no peak", r"the log-likelihood $\ell(\mu, \sigma)$"),
            (true, _) => (r"the likelihood $L(\mu, \sigma)$, shaded, and curves so far below its peak", r"the likelihood $L(\mu, \sigma)$"),
            _ => (r"the log-likelihood $\ell(\mu, \sigma)$: curves so far below the peak", r"the log-likelihood $\ell(\mu, \sigma)$"),
        };
        let title = [whole.to_string(), short.to_string(), String::new()];
        p.title(f, &pick(f, &title, room, lay.big), lay.big);
        x_axis(f, p, goal.c.x, r"$\mu$", lay.small);
        y_axis(f, p, goal.c.y, r"$\sigma$", lay.small);
        let (min, max) = p.rect;
        let steps = [0.5, drop95(), 4.0, 8.0, 16.0, 32.0, 64.0];
        // the level the curves are measured from: the peak, or with no maximum the
        // highest the surface is just above the bottom of the chart
        let top = match sfit {
            Fit::Two(t) => Some(t.ll),
            Fit::Unbounded => Some(shown.ll([shown.mean, p.y[1] * 0.02])),
            _ => None,
        };
        if let Some(top) = top {
            if plain {
                // the likelihood itself as a shade, in proportion to the peak's
                let (nx, ny) = (((p.width() / 9.0) as usize).clamp(12, 90), ((p.height() / 9.0) as usize).clamp(10, 70));
                let (cw, chh) = (p.width() / nx as f32, p.height() / ny as f32);
                for j in 0..ny {
                    for i in 0..nx {
                        let (mu, sg) = (p.x_at(min.x + (i as f32 + 0.5) * cw), p.y_at(max.y - (j as f32 + 0.5) * chh));
                        let rel = (shown.ll([mu, sg]) - top).exp();
                        if rel > 0.004 {
                            let c = Vec2::new(min.x + (i as f32 + 0.5) * cw, max.y - (j as f32 + 0.5) * chh);
                            plot::fill(f, (c - Vec2::new(cw, chh) * 0.5, c + Vec2::new(cw, chh) * 0.5), pal::TEXT.a((0.55 * rel) as f32));
                        }
                    }
                }
            }
            let peaked = matches!(sfit, Fit::Two(_));
            for (l, step) in steps.iter().enumerate() {
                let (w, c) = if l == 1 && peaked { (2.0, ESTIMATE) } else { (1.0, pal::TEXT.a(0.4)) };
                kept_path(f, CURVES, p, &level_curve(shown, top - step, p.y[1] * 1.02), w, c);
            }
            // what the curves' levels are: the yellow one's above the peak, and the
            // others' below it, where the deeper ones are still on the chart
            if let Fit::Two(t) = sfit {
                let along = |v: f64| shown.ll([t.mu, v]);
                let mut last = f32::MIN;
                for (l, s) in steps.iter().enumerate().skip(1) {
                    let level = t.ll - s;
                    let sg = if l == 1 {
                        let mut hi = t.sigma * 2.0;
                        for _ in 0..60 {
                            if along(hi) < level {
                                break;
                            }
                            hi *= 2.0;
                        }
                        crossing(along, level, t.sigma, hi)
                    } else {
                        let mut lo = t.sigma * 0.5;
                        for _ in 0..200 {
                            if along(lo) < level {
                                break;
                            }
                            lo *= 0.5;
                        }
                        crossing(along, level, t.sigma, lo)
                    };
                    let at = p.at(t.mu, sg);
                    let below = l > 1;
                    if !(plot::within(at, p.rect, 0.0) && at.y < max.y - 24.0) || (below && at.y < last + 13.0) {
                        continue;
                    }
                    if below {
                        last = at.y;
                    }
                    let text = match (plain, l) {
                        (true, 1) => format!("{} of the peak", figures((-s).exp(), 2)),
                        (true, _) => format!("$e^{{-{s}}}$"),
                        (false, 1) => format!("{} below the peak", val(*s)),
                        _ => format!("{}", s),
                    };
                    let (offset, align) = if below { (Vec2::new(4.0, 1.0), Align2::LEFT_TOP) } else { (Vec2::new(4.0, -2.0), Align2::LEFT_BOTTOM) };
                    f.caption_px(at + offset, text, lay.small, if l == 1 { ESTIMATE } else { pal::DIM }, align);
                }
            }
        }
        match sfit {
            Fit::Two(t) => {
                // the parabola's own 1.92 curve: an ellipse
                let [[m0, m1], [s0, s1]] = t.wald();
                let ellipse: Vec<(f64, f64)> = (0..=96).map(|k| {
                    let a = k as f64 / 96.0 * std::f64::consts::TAU;
                    (t.mu + 0.5 * (m1 - m0) * a.cos(), t.sigma + 0.5 * (s1 - s0) * a.sin())
                }).collect();
                kept_path(f, CURVES, p, &ellipse, 1.5, OTHER.a(0.7));
                // the shadows on the two axes: each parameter's intervals
                let bar_x = |f: &mut Frame, r: [f64; 2], dy: f32, w: f32, c: Color| {
                    let (a, b) = (p.px_x(r[0]).max(min.x).min(max.x), p.px_x(r[1]).max(min.x).min(max.x));
                    interval_bar(f, MARKS, a, b, max.y - dy, w, c);
                };
                bar_x(f, t.mu_range, 7.0, 4.0, ESTIMATE);
                bar_x(f, [m0, m1], 17.0, 3.0, OTHER);
                let bar_y = |f: &mut Frame, r: [f64; 2], dx: f32, w: f32, c: Color| {
                    let (a, b) = (p.px_y(r[0]).max(min.y).min(max.y), p.px_y(r[1]).max(min.y).min(max.y));
                    let x = min.x + dx;
                    plot::line_of(f, MARKS, Vec2::new(x, a), Vec2::new(x, b), w, c);
                    for y in [a, b] {
                        plot::line_of(f, MARKS, Vec2::new(x - 4.0, y), Vec2::new(x + 4.0, y), 1.5, c);
                    }
                };
                bar_y(f, t.sigma_range, 7.0, 4.0, ESTIMATE);
                bar_y(f, [s0, s1], 17.0, 3.0, OTHER);
                // the yellow curve's furthest points, down and across to their shadows
                let n = shown.n();
                let best_sigma = |m: f64| (shown.ss / n + (m - shown.mean) * (m - shown.mean)).sqrt();
                for m in t.mu_range {
                    let a = p.at(m, best_sigma(m));
                    if plot::within(a, p.rect, 0.0) {
                        plot::dashed_of(f, MARKS, a, Vec2::new(a.x, max.y - 7.0), 1.0, ESTIMATE.a(0.45));
                    }
                }
                for sg in t.sigma_range {
                    let a = p.at(t.mu, sg);
                    if plot::within(a, p.rect, 0.0) {
                        plot::dashed_of(f, MARKS, a, Vec2::new(min.x + 7.0, a.y), 1.0, ESTIMATE.a(0.45));
                    }
                }
                // the spread that divides by n − 1, beside the peak's, which divides by n
                let s = (shown.ss / (n - 1.0)).sqrt();
                let sy = p.px_y(s);
                if sy > min.y && sy < max.y {
                    plot::line_of(f, MARKS, Vec2::new(min.x + 24.0, sy), Vec2::new(min.x + 36.0, sy), 1.5, pal::TEXT);
                    f.text_px(Vec2::new(min.x + 40.0, sy), r"$s$", lay.small, pal::TEXT, Align2::LEFT_CENTER);
                }
                let top = p.at(t.mu, t.sigma).max(min).min(max);
                plot::disc(f, top, 8.0, ESTIMATE);
                let label = format!(r"$\hat\mu = {}$, $\hat\sigma = {}$", val(t.mu), val(t.sigma));
                let w = f.text_size(&label, lay.text).x;
                let right = top.x + w + 16.0 < max.x;
                f.text_placed("peak", self.tag(0), top, Vec2::new(if right { w * 0.5 + 10.0 } else { -w * 0.5 - 10.0 }, 0.0), label, lay.text, ESTIMATE);
            }
            Fit::Unbounded => {
                let x = p.px_x(shown.mean).max(min.x).min(max.x);
                hud_arrow(f, Vec2::new(x, max.y - 60.0), Vec2::new(x, max.y - 4.0), 2.0, ESTIMATE);
                f.caption_px(Vec2::new((min.x + max.x) * 0.5, min.y + 10.0), r"no maximum: with no spread in the data, the likelihood grows without bound as $\sigma$ shrinks to 0", lay.small, ESTIMATE, Align2::CENTER_TOP);
            }
            _ => {
                f.text_px(Vec2::new((min.x + max.x) * 0.5, min.y + 16.0), "no data: the same for every value, the data say nothing yet", lay.text, pal::DIM, Align2::CENTER_TOP);
            }
        }
        if let Some([a, b]) = self.truth_shown().and_then(|t| t.target(Model::Both)) {
            let at = p.at(a, b);
            if plot::within(at, p.rect, 0.0) {
                hud_circle(f, at, 6.0, 1.8, TRUTH);
                let words = if self.set().truth.tail { "the closest bell to the truth" } else { "the truth" };
                f.text_px(at + Vec2::new(-9.0, -4.0), words, lay.small, TRUTH, Align2::RIGHT_BOTTOM);
            }
        }
    }

    /// What the numbers are, beside the charts (below them on a phone).
    fn facts(&self, fit: &Fit, short: bool) -> Vec<(String, Color)> {
        let model = self.model;
        let sym = model.symbol();
        let s = self.set();
        let o = self.obs();
        let th = s.theta;
        let at = o.ll(th);
        let tried = if model == Model::Both { format!(r"$\mu = {}$, $\sigma = {}$", val(th[0]), val(th[1])) } else { format!("${sym} = {}$", val(th[0])) };
        let mut lines = vec![(
            if short { format!(r"at {tried}: $L = {}$, $\ell = {}$", exp_text(at), ell(at)) } else { format!(r"at {tried} (purple): $L = {}$, $\ell = {}$", exp_text(at), ell(at)) },
            ASSUMED.mix(pal::WHITE, 0.3),
        )];
        let (c, z) = (val(drop95()), val(z95()));
        match fit {
            Fit::Flat => lines.push(("No data yet: the likelihood is 1 for every value, and the data say nothing yet.".to_string(), pal::DIM)),
            Fit::Unbounded => lines.push((r"No maximum: with one value (or all of them at one place), the likelihood grows without bound as $\sigma$ shrinks to 0 at that place.".to_string(), pal::DIM)),
            Fit::One(t) => {
                let hat = format!(r"\hat{{{sym}}}");
                lines.push((format!(r"peak: ${hat} = {}$, where $\ell = {}$", val(t.at), ell(t.ll)), ESTIMATE));
                match (t.info, t.se()) {
                    (Some(info), Some(se)) => {
                        lines.push((format!(r"curvature there {}, so $\text{{SE}} = 1/\sqrt{{{}}} = {}$", val(info), val(info), figures(se, 3)), pal::TEXT));
                        if !short {
                            // n times the information one value carries, at the estimate
                            let one = fisher(model, [t.at, 0.0], o.sd)[0];
                            let n = o.n();
                            let what = if model == Model::Coin { "toss" } else { "value" };
                            let text = if model == Model::Centre {
                                let values = if n == 1.0 { "1 value carries".to_string() } else { format!("{n} values carry") };
                                format!("One value carries {} of information on average, so {values} {} on average; the curvature here is {}.", val(one), val(n * one), val(info))
                            } else {
                                format!(r"That is {} times {}, the information one {what} carries at ${hat}$ (the curvature it gives on average).", n, val(one))
                            };
                            lines.push((text, pal::DIM));
                        }
                    }
                    _ => lines.push(("The peak is at the edge, where the curve is still rising: it has no level top, so no parabola, and no ± interval.".to_string(), pal::DIM)),
                }
                let word = if short { "likelihood interval" } else { "95 % likelihood interval" };
                lines.push((format!("{word}: {}", pieces_text(&t.pieces)), ESTIMATE));
                if let Some([w0, w1]) = t.wald() {
                    lines.push((format!(r"${hat} \pm {z}\,\text{{SE}}$, from the parabola: {} to {}", val(w0), val(w1)), OTHER));
                }
                if !short {
                    let [d0, d1] = domain(&o);
                    let says = if t.pieces.len() > 1 {
                        format!("{} peaks: the curvature at the highest says nothing of the others, and the values within {c} of the top come in {} pieces.", t.peaks, t.pieces.len())
                    } else if t.peaks > 1 {
                        format!("{} peaks: the curvature at the highest says nothing of the others.", t.peaks)
                    } else if model == Model::Mean {
                        "For a bell the log-likelihood is a parabola: the two intervals are the same.".to_string()
                    } else if t.wald().is_some_and(|[w0, w1]| (model == Model::Coin || model == Model::Rate) && (w0 < d0.max(0.0) || (model == Model::Coin && w1 > d1))) {
                        "The ± interval runs past the end of the values the parameter can take: the parabola is symmetric, the curve is not.".to_string()
                    } else {
                        String::new()
                    };
                    if !says.is_empty() {
                        lines.push((says, pal::DIM));
                    }
                }
            }
            Fit::Two(t) => {
                let n = o.n();
                let sn1 = (o.ss / (n - 1.0)).sqrt();
                lines.push((format!(r"peak: $\hat\mu = {}$, $\hat\sigma = {}$ (dividing by $n$; $s = {}$ divides by $n - 1$)", val(t.mu), val(t.sigma), val(sn1)), ESTIMATE));
                lines.push((format!(r"curvatures there {} and {}, so SEs {} and {}", val(t.info[0]), val(t.info[1]), figures(1.0 / t.info[0].sqrt(), 3), figures(1.0 / t.info[1].sqrt(), 3)), pal::TEXT));
                lines.push((format!(r"likelihood intervals: $\mu$ {} to {}, $\sigma$ {} to {}", val(t.mu_range[0]), val(t.mu_range[1]), val(t.sigma_range[0]), val(t.sigma_range[1])), ESTIMATE));
                let [[m0, m1], [s0, s1]] = t.wald();
                lines.push((format!(r"$\pm {z}\,\text{{SE}}$: $\mu$ {} to {}, $\sigma$ {} to {}", val(m0), val(m1), val(s0), val(s1)), OTHER));
                if !short {
                    lines.push(("The shadows of the yellow curve on the axes are the likelihood intervals; those of the orange ellipse, the ± ones. Each grey curve is as far below the peak as its number says.".to_string(), pal::DIM));
                }
            }
        }
        if let Some(t) = self.truth_shown() {
            let text = if t.tail {
                match t.target(model) {
                    Some([a, b]) if model == Model::Both => format!(r"truth: a long right tail, no bell; the closest bell has $\mu = {}$, $\sigma = {}$", val(a), val(b)),
                    Some([a, _]) => format!(r"truth: a long right tail; the closest of the model's curves has ${sym} = {}$", val(a)),
                    None => "truth: a long right tail".to_string(),
                }
            } else if model == Model::Both {
                format!(r"truth: $\mu = {}$, $\sigma = {}$", val(t.a), val(t.b))
            } else {
                format!("truth: ${sym} = {}$", val(t.a))
            };
            lines.push((text, TRUTH));
        }
        lines
    }

    fn draw_facts(&self, f: &mut Frame, lay: &Layout, fit: &Fit) {
        let (min, max) = lay.facts;
        if max.x - min.x < 60.0 || max.y - min.y < 20.0 {
            return;
        }
        let size = if lay.narrow { lay.small } else { lay.text };
        let mut y = min.y;
        for (text, color) in self.facts(fit, lay.narrow) {
            let used = wrap_px(f, Vec2::new(min.x, y), max.x - min.x, max.y, &text, size, color);
            y += used + if lay.narrow { 0.0 } else { 5.0 };
        }
        // the colours, at the foot of the column (a phone has the labels on the charts)
        if !lay.narrow {
            let both = self.model == Model::Both;
            let mut entries: Vec<(Color, &str)> = vec![
                (DATA, if self.model == Model::Coin { "the tosses" } else { "the data" }),
                (ASSUMED, "the value tried, and the model there"),
                (ESTIMATE, if both { "the peak, and the curve 1.92 below it" } else { "the peak, and the 95 % likelihood interval" }),
                (OTHER, if both { "the parabola's own curve 1.92 below" } else { r"the parabola at the peak, and $\pm 1.96\,\text{SE}$" }),
            ];
            if !both {
                entries.insert(2, (pal::TEXT, if self.plain { "the likelihood of every value" } else { "the log-likelihood of every value" }));
            }
            if self.truth_shown().is_some() {
                entries.push((TRUTH, "the truth the data were drawn from"));
            }
            let h = entries.len() as f32 * (lay.small + 6.0);
            if max.y - h > y + 12.0 {
                hud_key(f, Vec2::new(min.x, max.y - h), &entries, lay.small);
            }
        }
    }
}

/// The value at the peak, if there is one.
fn peak_theta(fit: &Fit) -> Option<[f64; 2]> {
    match fit {
        Fit::One(t) => Some([t.at, 0.0]),
        Fit::Two(t) => Some([t.mu, t.sigma]),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn trial(k: usize) -> (Likelihood, Fit) {
        let mut s = Likelihood::create();
        s.set_up(k);
        let fit = s.fit_now();
        (s, fit)
    }

    fn top(fit: &Fit) -> Top {
        match fit {
            Fit::One(t) => t.clone(),
            other => panic!("not one peak: {other:?}"),
        }
    }

    fn top2(fit: &Fit) -> Top2 {
        match fit {
            Fit::Two(t) => t.clone(),
            other => panic!("not a peak of the surface: {other:?}"),
        }
    }

    fn close(a: f64, b: f64, tol: f64) -> bool {
        (a - b).abs() <= tol
    }

    fn coin(heads: usize, n: usize) -> Obs {
        Obs::new(Model::Coin, Vec::new(), heads as f64, (n - heads) as f64, 1.0)
    }

    fn sample(model: Model, n: usize, seed: u64) -> Obs {
        let (xs, _) = draw(model, Truth::of(model), 1.0, n, seed);
        Obs::new(model, xs, 0.0, 0.0, 1.0)
    }

    /// The curvature of the log-likelihood at `th` along parameter `k`, by finite differences.
    fn bend(o: &Obs, th: [f64; 2], k: usize, h: f64) -> f64 {
        let (mut a, mut b) = (th, th);
        a[k] -= h;
        b[k] += h;
        -(o.ll(a) - 2.0 * o.ll(th) + o.ll(b)) / (h * h)
    }

    /// The peak found numerically is where the formula puts it: the share of heads, the
    /// mean, one over the mean; with the centre and the spread both unknown, the mean
    /// and the spread that divides by n.
    #[test]
    fn the_peak_is_where_the_formula_puts_it() {
        for (h, n) in [(7, 10), (70, 100), (1, 10), (3, 7), (199, 200), (5, 6)] {
            let t = top(&fit(&coin(h, n)));
            assert!(close(t.at, h as f64 / n as f64, 1e-7) && !t.edge && t.peaks == 1, "{h} in {n}: {}", t.at);
        }
        for seed in 1..6 {
            for n in [1, 2, 10, 60, 200] {
                let o = sample(Model::Mean, n, seed);
                let t = top(&fit(&o));
                assert!(close(t.at, o.mean, 1e-7) && t.peaks == 1, "the mean of {n}: {} against {}", t.at, o.mean);
                let o = sample(Model::Rate, n, seed);
                let t = top(&fit(&o));
                assert!(close(t.at * o.mean, 1.0, 1e-7), "the rate of {n}: {} against {}", t.at, 1.0 / o.mean);
                if n >= 2 {
                    let o = sample(Model::Both, n, seed);
                    let t = top2(&fit(&o));
                    let sigma = (o.ss / n as f64).sqrt();
                    assert!(close(t.mu, o.mean, 1e-7) && close(t.sigma / sigma, 1.0, 1e-7), "both of {n}: {t:?}, {} {sigma}", o.mean);
                }
            }
        }
    }

    /// The curvature at the peak is the curve's own (by finite differences), and for the
    /// coin, the bell and the rate it is exactly n times the Fisher information at the
    /// estimate. For the Cauchy curve it is not: there it is n/2 only on average.
    #[test]
    fn the_curvature_is_n_times_the_information() {
        let mut cases = vec![coin(7, 10), coin(70, 100), coin(1, 10), coin(13, 40)];
        for seed in 1..4 {
            cases.push(sample(Model::Mean, 10, seed));
            cases.push(sample(Model::Rate, 25, seed));
        }
        for o in &cases {
            let t = top(&fit(o));
            let info = t.info.expect("a level top");
            let fisher_n = o.n() * fisher(o.model, [t.at, 0.0], o.sd)[0];
            assert!(close(info / fisher_n, 1.0, 1e-6), "{:?}: {info} against {fisher_n}", o.model);
            let numeric = bend(o, [t.at, 0.0], 0, 1e-4 * t.at.abs().max(0.1));
            assert!(close(numeric / info, 1.0, 1e-4), "{:?}: {numeric} against {info}", o.model);
        }
        for seed in 1..4 {
            let o = sample(Model::Both, 12, seed);
            let t = top2(&fit(&o));
            let fisher_n = fisher(Model::Both, [t.mu, t.sigma], 1.0).map(|i| i * o.n());
            let h = 1e-4 * t.sigma;
            for k in 0..2 {
                assert!(close(t.info[k] / fisher_n[k], 1.0, 1e-6));
                assert!(close(bend(&o, [t.mu, t.sigma], k, h) / t.info[k], 1.0, 1e-4));
            }
            // none across both at the peak
            let at = |dm: f64, ds: f64| o.ll([t.mu + dm, t.sigma + ds]);
            let cross = (at(h, h) - at(h, -h) - at(-h, h) + at(-h, -h)) / (4.0 * h * h);
            assert!(cross.abs() < 1e-4 * t.info[0], "{cross}");
        }
        // the Cauchy curve: its own curvature, which is not n/2
        let o = Obs::new(Model::Centre, THREE.to_vec(), 0.0, 0.0, 1.0);
        let t = top(&fit(&o));
        let info = t.info.expect("a level top");
        assert!(close(bend(&o, [t.at, 0.0], 0, 1e-4) / info, 1.0, 1e-4));
        assert!(close(info, 1.87, 0.005) && (info - 0.5 * o.n()).abs() > 0.3, "{info}");
    }

    /// The opening and the first experiment: 7 heads in 10, the peak at 0.7, the 95 %
    /// likelihood interval from 0.39 to 0.92, with a fair coin inside it.
    #[test]
    fn seven_heads_in_ten() {
        let (s, fit) = trial(0);
        let t = top(&fit);
        assert!(close(t.at, 0.7, 1e-6) && t.pieces.len() == 1);
        let [a, b] = t.pieces[0];
        assert!(close(a, 0.3935, 5e-4) && close(b, 0.9154, 5e-4), "{a} {b}");
        assert!(a < 0.5 && 0.5 < b);
        assert!(TRIALS[0].1.contains(r"\hat p = 0.7") && TRIALS[0].1.contains(&format!("from {} to {}", val(a), val(b))));
        assert!(TRIALS[0].1.contains(&format!("within {} of", val(drop95()))));
        let opening = Likelihood::create();
        assert!(opening.model == Model::Coin && opening.set() == s.set() && s.set().theta[0] == 0.5);
        assert_eq!((s.set().tosses.len(), s.set().tosses.iter().filter(|t| **t).count()), (10, 7));
        // what the picture says
        let facts: Vec<String> = s.facts(&fit, false).into_iter().map(|l| l.0).collect();
        assert!(facts.iter().any(|l| l.contains("likelihood interval: 0.39 to 0.92")), "{facts:?}");
        assert!(facts.iter().any(|l| l.contains("from the parabola: 0.42 to 0.98")), "{facts:?}");
    }

    /// Ten times the tosses: the same peak, ten times the curvature, a standard error
    /// √10 times smaller, and a fair coin far outside.
    #[test]
    fn ten_times_the_tosses() {
        let (_, few) = trial(0);
        let (s, many) = trial(1);
        let (a, b) = (top(&few), top(&many));
        assert!(close(b.at, 0.7, 1e-6));
        let (ia, ib) = (a.info.unwrap_or(0.0), b.info.unwrap_or(0.0));
        assert!(close(ib / ia, 10.0, 1e-6) && ia.round() == 48.0 && ib.round() == 476.0, "{ia} {ib}");
        assert!(close(a.se().unwrap_or(0.0) / b.se().unwrap_or(1.0), 10f64.sqrt(), 1e-6) && val(10f64.sqrt()) == "3.16");
        let (wa, wb) = (a.wald().unwrap_or_default(), b.wald().unwrap_or_default());
        assert_eq!([val(wb[0]), val(wb[1]), val(wa[0]), val(wa[1])], ["0.61", "0.79", "0.42", "0.98"]);
        let drop = b.ll - s.obs().ll([0.5, 0.0]);
        assert!(close(drop, 8.22, 0.01) && drop > drop95(), "{drop}");
        // the likelihood interval narrows by less than √10: at 10 tosses it was lopsided
        let ratio = (a.span()[1] - a.span()[0]) / (b.span()[1] - b.span()[0]);
        assert!(close(ratio, 2.93, 0.01), "{ratio}");
        // lopsided at 10 tosses: further below the peak than above it
        assert!(a.at - a.span()[0] > 1.25 * (a.span()[1] - a.at), "{:?}", a.span());
        for words in ["(476 against 48)", r"$\sqrt{10} = 3.16$", "is 0.61 to 0.79, against 0.42 to 0.98", "8.2 below the peak", "narrows by 2.9"] {
            assert!(TRIALS[1].1.contains(words), "{words}");
        }
    }

    /// A bell's centre: the peak is the mean exactly, not the truth; a value moved moves
    /// it a tenth as far; and the two intervals are the same.
    #[test]
    fn a_bells_centre() {
        let (mut s, fit) = trial(2);
        let t = top(&fit);
        let o = s.obs();
        assert!(o.n() == 10.0 && close(t.at, o.mean, 1e-7) && s.truth_shown().is_some());
        assert!((t.at - s.set().truth.a).abs() > 0.05, "{} against the truth {}", t.at, s.set().truth.a);
        let w = t.wald().unwrap_or_default();
        assert!(t.pieces.len() == 1 && close(t.pieces[0][0], w[0], 1e-6) && close(t.pieces[0][1], w[1], 1e-6), "{:?} {w:?}", t.pieces);
        assert!(close(w[1] - t.at, z95() / 10f64.sqrt(), 1e-9));
        s.set_mut().xs[3] += 1.5;
        let moved = top(&s.fit_now()).at;
        assert!(close(moved - t.at, 0.15, 1e-6), "{}", moved - t.at);
        assert!(TRIALS[2].1.contains(r"$\pm 1.96/\sqrt{10}$") && TRIALS[2].1.contains("a tenth of the way"));
    }

    /// Centre and spread: the peak at the mean and the spread that divides by n, √(4/5)
    /// times the one that divides by n − 1; the curve 1.92 below the peak lopsided; its
    /// shadows the intervals.
    #[test]
    fn centre_and_spread() {
        let (s, fit) = trial(3);
        let t = top2(&fit);
        let o = s.obs();
        let n = o.n();
        let (sigma, s1) = ((o.ss / n).sqrt(), (o.ss / (n - 1.0)).sqrt());
        assert!(n == 5.0 && close(t.mu, o.mean, 1e-7) && close(t.sigma, sigma, 1e-7));
        assert!(close(t.sigma / s1, 0.8f64.sqrt(), 1e-6) && val(0.8f64.sqrt()) == "0.89");
        // here the truth's spread, 1, lies between the two
        assert!(t.sigma < 1.0 && s1 > 1.0 && s.truth_shown().is_some_and(|t| t.b == 1.0), "{} {s1}", t.sigma);
        // the curve reaches further up than down; the ellipse is even
        assert!(t.sigma_range[1] - t.sigma > 2.0 * (t.sigma - t.sigma_range[0]), "{:?}", t.sigma_range);
        // the region within 1.92 of the peak, on a fine grid: its shadows are the intervals
        let level = t.ll - drop95();
        let (mut mu, mut sg) = ([f64::MAX, f64::MIN], [f64::MAX, f64::MIN]);
        for i in 0..=1200 {
            for j in 1..=1200 {
                let (m, v) = (t.mu - 3.0 + 6.0 * i as f64 / 1200.0, 4.0 * j as f64 / 1200.0);
                if o.ll([m, v]) >= level {
                    (mu, sg) = ([mu[0].min(m), mu[1].max(m)], [sg[0].min(v), sg[1].max(v)]);
                }
            }
        }
        for k in 0..2 {
            assert!(close(mu[k], t.mu_range[k], 0.006) && close(sg[k], t.sigma_range[k], 0.004), "{mu:?} {sg:?} {t:?}");
        }
        // on average too low: the mean of the peak's spread over many samples of five
        let mut rng = Rng::new(9);
        let draws = 20000;
        let average = (0..draws)
            .map(|_| {
                let xs = Dist::Normal { mean: 0.0, sd: 1.0 }.draws(5, &mut rng);
                (stats::variance_n(&xs)).sqrt()
            })
            .sum::<f64>()
            / draws as f64;
        assert!(close(average, 0.84, 0.01), "{average}");
        for words in [r"$\sqrt{4/5} = 0.89$", "between them", "further up than down"] {
            assert!(TRIALS[3].1.contains(words), "{words}");
        }
    }

    /// The wrong model: the peak is the bell with the sample's mean and spread, which
    /// puts 11 % below 0; the surface is the one bell-shaped data with that mean and
    /// spread would give; and with ever more data the peak comes to the bell under
    /// which data from the truth are most probable on average.
    #[test]
    fn the_wrong_model() {
        let (s, fit) = trial(4);
        let t = top2(&fit);
        let o = s.obs();
        assert!(o.n() == 60.0 && s.set().truth.tail && s.truth_shown().is_some());
        let sigma = (o.ss / 60.0).sqrt();
        assert!(close(t.mu, o.mean, 1e-7) && close(t.sigma, sigma, 1e-7));
        assert!(close(s.set().theta[0], t.mu, 1e-9) && close(s.set().theta[1], t.sigma, 1e-9));
        let below = Dist::Normal { mean: t.mu, sd: t.sigma }.cdf(0.0);
        assert!(num(100.0 * below, 0) == "11" && o.lo > 0.0, "{below}");
        // the truth's own peak, where its values crowd
        assert_eq!(val((-0.75f64 * 0.75).exp()), "0.57");
        // the same surface as bell-shaped values with the same mean and spread
        let z: Vec<f64> = (0..60).map(|i| stats::normal_quantile((i as f64 + 0.5) / 60.0)).collect();
        let (zm, zs) = (stats::mean(&z), stats::variance_n(&z).sqrt());
        let bell = Obs::new(Model::Both, z.iter().map(|v| o.mean + sigma * (v - zm) / zs).collect(), 0.0, 0.0, 1.0);
        for th in [[0.0, 0.5], [1.3, 1.0], [2.0, 2.5]] {
            assert!(close(bell.ll(th) / o.ll(th), 1.0, 1e-9));
        }
        // the closest bell: data from the truth are most probable under it on average
        let target = s.set().truth.target(Model::Both).unwrap_or_default();
        let expected = |th: [f64; 2]| (0..4000).map(|i| ln_f(Model::Both, TAIL.quantile((i as f64 + 0.5) / 4000.0), th, 1.0)).sum::<f64>() / 4000.0;
        let best = expected(target);
        for (dm, ds) in [(0.05, 0.0), (-0.05, 0.0), (0.0, 0.05), (0.0, -0.05)] {
            assert!(best > expected([target[0] + dm, target[1] + ds]));
        }
        let (xs, _) = draw(Model::Both, s.set().truth, 1.0, 20000, 7);
        let big = top2(&fit2(&Obs::new(Model::Both, xs, 0.0, 0.0, 1.0)));
        assert!(close(big.mu, target[0], 0.03) && close(big.sigma, target[1], 0.04), "{big:?} {target:?}");
        assert_eq!(val(t.mu), "1.32");
        for words in ["11 %", "at 1.32", "near 0.57", "only through their mean and spread", "highest average log-likelihood"] {
            assert!(TRIALS[4].1.contains(words), "{words}");
        }
    }

    /// 1 head in 10: lopsided; the ± interval runs below 0, the likelihood interval
    /// stays inside and leans towards 1.
    #[test]
    fn one_head_in_ten() {
        let (s, fit) = trial(5);
        let t = top(&fit);
        let o = s.obs();
        assert!(close(t.at, 0.1, 1e-6) && t.pieces.len() == 1);
        let w = t.wald().unwrap_or_default();
        let [a, b] = t.pieces[0];
        assert_eq!([val(w[0]), val(w[1]), val(a), val(b)], ["−0.086", "0.29", "0.006", "0.37"]);
        assert!(w[0] < 0.0 && a > 0.0 && b - t.at > 2.5 * (t.at - a));
        // steeper towards 0 than towards 1
        assert!(o.ll([0.02, 0.0]) < o.ll([0.18, 0.0]) - 0.5);
        assert!(TRIALS[5].1.contains("from −0.086 to 0.29") && TRIALS[5].1.contains("0.006 to 0.37"));
    }

    /// Three values far apart and a Cauchy curve: three peaks, an interval in two pieces.
    #[test]
    fn three_peaks() {
        let (_, fit) = trial(6);
        let t = top(&fit);
        assert_eq!(t.peaks, 3);
        assert_eq!(val(t.at), "−1.03");
        assert_eq!(pieces_text(&t.pieces), "−6.67 to 2.04, and 3.35 to 5.42");
        let w = t.wald().unwrap_or_default();
        assert_eq!([val(w[0]), val(w[1])], ["−2.46", "0.40"]);
        // one peak near each value: the curve on a fine grid
        let o = Obs::new(Model::Centre, THREE.to_vec(), 0.0, 0.0, 1.0);
        let g: Vec<f64> = (0..=4000).map(|i| -9.0 + 17.0 * i as f64 / 4000.0).collect();
        let tops: Vec<f64> = (1..4000).filter(|&i| o.ll([g[i], 0.0]) > o.ll([g[i - 1], 0.0]) && o.ll([g[i], 0.0]) > o.ll([g[i + 1], 0.0])).map(|i| g[i]).collect();
        assert_eq!(tops.len(), 3);
        for (p, x) in tops.iter().zip(THREE) {
            assert!((p - x).abs() < 0.4, "{tops:?}");
        }
        for words in [r"$\hat\theta = -1.03$", "−6.67 to 2.04 and 3.35 to 5.42", "is −2.46 to 0.40"] {
            assert!(TRIALS[6].1.contains(words), "{words}");
        }
    }

    /// No data, all heads, one value with the spread unknown, values on top of one
    /// another: each says what it is, and every number stays finite.
    #[test]
    fn degenerate_data_say_what_they_are() {
        for m in MODELS {
            let o = Obs::new(m, Vec::new(), 0.0, 0.0, 1.0);
            assert_eq!(fit(&o), Fit::Flat);
            assert_eq!(o.ll([0.3, 1.0]), 0.0);
        }
        // all heads: the peak at the edge, still rising; no curvature, no ± interval
        let t = top(&fit(&coin(10, 10)));
        assert!(t.at == 1.0 && t.edge && t.info.is_none() && t.wald().is_none());
        assert_eq!(pieces_text(&t.pieces), "0.83 to 1.00");
        let t = top(&fit(&coin(0, 10)));
        assert!(t.at == 0.0 && t.edge && pieces_text(&t.pieces) == "0.00 to 0.17");
        // one value, or all of them at one place, with the spread unknown: no maximum
        for xs in [vec![2.0], vec![2.0, 2.0, 2.0]] {
            let o = Obs::new(Model::Both, xs, 0.0, 0.0, 1.0);
            assert_eq!(fit(&o), Fit::Unbounded);
            assert!(o.ll([2.0, 1e-3]) > o.ll([2.0, 1e-2]) + 2.0);
        }
        // values on top of each other are otherwise no trouble
        assert!(close(top(&fit(&Obs::new(Model::Mean, vec![1.0, 1.0, 1.0, 3.0], 0.0, 0.0, 1.0))).at, 1.5, 1e-7));
        assert!(close(top(&fit(&Obs::new(Model::Centre, vec![0.4, 0.4], 0.0, 0.0, 1.0))).at, 0.4, 1e-6));
        assert!(close(top(&fit(&Obs::new(Model::Rate, vec![0.5, 0.5], 0.0, 0.0, 1.0))).at, 2.0, 1e-6));
        let t = top2(&fit(&Obs::new(Model::Both, vec![1.0, 1.0, 3.0], 0.0, 0.0, 1.0)));
        assert!(t.mu_range.iter().chain(&t.sigma_range).all(|v| v.is_finite()));
        // and what the picture says of each is finite
        let mut s = Likelihood::create();
        for (model, xs, heads, n) in [(Model::Coin, vec![], 10, 10), (Model::Coin, vec![], 0, 0), (Model::Both, vec![2.0], 0, 0), (Model::Both, vec![], 0, 0), (Model::Rate, vec![], 0, 0), (Model::Centre, vec![1.0], 0, 0)] {
            s.model = model;
            let set = s.set_mut();
            (set.xs, set.tosses) = (xs, pattern(heads, n));
            let fit = s.fit_now();
            for (line, _) in s.facts(&fit, false).into_iter().chain(s.facts(&fit, true)) {
                let words: Vec<&str> = line.split(|c: char| !c.is_alphanumeric()).collect();
                assert!(!line.contains('?') && !words.contains(&"NaN") && !words.contains(&"inf"), "{model:?}: {line}");
            }
            let (ax, top) = s.data_view(&fit);
            let (cx, ctop) = s.param_view(&fit, ax);
            assert!(ax.iter().chain(&cx).chain([&top, &ctop]).all(|v| v.is_finite()) && ax[1] > ax[0] && cx[1] > cx[0] && top > 0.0, "{model:?}");
        }
    }

    /// The model's log-density is the logarithm of the density of `math::stats`, and the
    /// log-likelihood is their sum.
    #[test]
    fn the_log_density_is_the_densitys_logarithm() {
        for model in [Model::Mean, Model::Both, Model::Rate, Model::Centre] {
            for th in [[0.7, 1.3], [1.5, 0.4]] {
                for x in [0.05, 0.9, 2.5] {
                    let d = dist_at(model, th, 0.8).map_or(f64::NAN, |d| d.pdf(x));
                    let l = ln_f(model, x, th, 0.8);
                    assert!(close(l, d.ln(), 1e-12), "{model:?} {x} {th:?}: {l} {}", d.ln());
                }
            }
            let o = Obs::new(model, vec![0.3, 1.1, 2.4, 0.8], 0.0, 0.0, 0.8);
            let sum: f64 = o.xs.iter().map(|&x| ln_f(model, x, [0.9, 1.2], 0.8)).sum();
            assert!(close(o.ll([0.9, 1.2]), sum, 1e-12));
        }
        assert!(close(ln_f(Model::Coin, 1.0, [0.3, 0.0], 1.0), 0.3f64.ln(), 1e-15) && close(ln_f(Model::Coin, 0.0, [0.3, 0.0], 1.0), 0.7f64.ln(), 1e-15));
        assert!(close(coin(7, 10).ll([0.3, 0.0]), 7.0 * 0.3f64.ln() + 3.0 * 0.7f64.ln(), 1e-12));
    }

    /// The likelihood interval is where the curve crosses the line 1.92 below the peak,
    /// and the estimate ± 1.96 standard errors is where the parabola crosses it.
    #[test]
    fn the_intervals_are_where_the_line_is_crossed() {
        assert!(close(drop95(), 1.920729, 1e-6) && close(z95(), 1.959964, 1e-6) && close(z95() * z95() / 2.0, drop95(), 1e-9));
        for k in [0, 1, 2, 5, 6] {
            let (s, fit) = trial(k);
            let (t, o) = (top(&fit), s.obs());
            let [a, b] = domain(&o);
            for &end in t.pieces.iter().flatten() {
                if end > a && end < b {
                    assert!(close(o.ll([end, 0.0]), t.ll - drop95(), 1e-7), "{k}: {end}");
                }
            }
            let (info, [w0, _]) = (t.info.unwrap_or(0.0), t.wald().unwrap_or_default());
            assert!(close(0.5 * info * (w0 - t.at) * (w0 - t.at), drop95(), 1e-9));
        }
    }

    /// The level curves of the bell's surface are where the surface is at their level,
    /// and reach as far as its shadows say.
    #[test]
    fn the_level_curves_are_on_their_levels() {
        let (s, fit) = trial(3);
        let (t, o) = (top2(&fit), s.obs());
        for step in [0.5, drop95(), 8.0] {
            let curve = level_curve(&o, t.ll - step, f64::INFINITY);
            assert_eq!(curve.len(), 2 * (LEVEL_POINTS + 1));
            for &(m, sg) in &curve {
                assert!(close(o.ll([m, sg]), t.ll - step, 1e-6), "{step}: {m} {sg}");
            }
        }
        let curve = level_curve(&o, t.ll - drop95(), f64::INFINITY);
        let widest = curve.iter().fold([f64::MAX, f64::MIN], |r, p| [r[0].min(p.0), r[1].max(p.0)]);
        assert!(close(widest[0], t.mu_range[0], 0.01) && close(widest[1], t.mu_range[1], 0.01), "{widest:?} {:?}", t.mu_range);
        // with no spread in the data the curves close at σ = 0
        let one = Obs::new(Model::Both, vec![2.0], 0.0, 0.0, 1.0);
        let curve = level_curve(&one, one.ll([2.0, 0.5]), f64::INFINITY);
        assert!(curve.iter().all(|p| p.0.is_finite() && p.1.is_finite()) && curve[0] == (2.0, 0.0));
    }

    /// Numbers are written with the digits they need.
    #[test]
    fn numbers_are_written_plainly() {
        assert_eq!([val(0.0999999), val(0.00599), val(-0.0859), val(0.7), val(476.19)], ["0.10", "0.006", "−0.086", "0.70", "476.2"]);
        assert_eq!(exp_text(7.0 * 0.5f64.ln() + 3.0 * 0.5f64.ln()), r"9.8 \times 10^{-4}");
        assert_eq!(exp_text(0.0), "1.00");
        assert_eq!(exp_text(f64::NEG_INFINITY), "0");
    }

    /// The numbers behind the captions (run with --ignored --nocapture).
    #[test]
    #[ignore]
    fn explore() {
        for k in 0..TRIALS.len() {
            let (s, fit) = trial(k);
            println!("{}: {:?}", TRIALS[k].0, fit);
            for (line, _) in s.facts(&fit, false) {
                println!("    {line}");
            }
        }
    }
}

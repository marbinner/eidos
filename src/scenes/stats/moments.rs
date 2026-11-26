//! Mean, spread and shape: what each summary of a distribution is, and what it answers
//! to.
//!
//! A density rests on a beam over a pivot the visitor drags. Each side of the pivot
//! turns the beam by its turning moment (the area on that side times how far out it
//! lies, added up), and the beam tips towards the side whose moment is the greater: it
//! is level only at the mean. The median is where the area is halved, the mode where
//! the density peaks, and a bar of one standard deviation each side of the mean shows
//! the spread in the data's own units.
//!
//! Under it, a sample from the same distribution on the same axis, with the square on
//! each point's distance from the pivot (or that distance itself), and their average
//! against where the pivot is: a parabola with its bottom at the sample's mean, or a V
//! with its bottom at its median. A table gives the distribution's summaries beside the
//! sample's. Some shapes have no mean at all (the Cauchy): the beam says so, and a
//! sample drawn one point at a time shows its mean never settling.

use super::plot::{self, Plot, Rect, Ticks, AXES, CURVES, MARKS, OWN};
use super::*;
use crate::math::stats::{self, normal_cdf, normal_pdf, normal_quantile, Dist};

pub const DEF: SceneDef = SceneDef {
    id: "moments",
    title: "Mean, spread and shape",
    group: CHANCE,
    blurb: "The mean as a balance point, the variance as a spread, skewness and heavy tails: what each summary of a distribution sees, and what it misses.",
    make: |_| Box::new(Moments::create()),
};

// ----- the shapes ------------------------------------------------------------------------

/// The most points a sample has.
const MAX_N: usize = 200;
const SQRT3: f64 = 1.732_050_807_568_877_2;
/// The Laplace distribution's scale that gives it variance 1 (2b² = 1).
const LAPLACE_B: f64 = std::f64::consts::FRAC_1_SQRT_2;
/// The spread of each bell of the visitor's own mixture.
const BELL_SD: f64 = 0.6;

/// A summary of a distribution: a number, infinite, or not defined at all (a mean
/// where both tails pull infinitely hard).
#[derive(Clone, Copy, Debug, PartialEq)]
enum Value {
    Is(f64),
    Infinite,
    Undefined,
}

impl Value {
    fn get(self) -> Option<f64> {
        match self {
            Value::Is(v) if v.is_finite() => Some(v),
            _ => None,
        }
    }

    /// As the table writes it.
    fn show(self, decimals: usize) -> String {
        match self {
            Value::Is(v) => num(v, decimals),
            Value::Infinite => r"$\infty$".to_string(),
            Value::Undefined => "none".to_string(),
        }
    }
}

/// Where a density is highest.
#[derive(Clone, Copy, Debug, PartialEq)]
enum Peaks {
    One(f64),
    /// Two peaks of the same height.
    Two(f64, f64),
    /// The highest peak, and a lower one.
    Main(f64, f64),
    /// A flat top from one value to another: every value on it is a peak.
    Flat(f64, f64),
}

/// The shapes on offer, as the buttons name them.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Kind {
    Bell,
    LongTail,
    TwoHumps,
    Flat,
    Peaked,
    HeavyTails,
    Mixture,
}

const KINDS: [Kind; 7] = [Kind::Bell, Kind::LongTail, Kind::TwoHumps, Kind::Flat, Kind::Peaked, Kind::HeavyTails, Kind::Mixture];

impl Kind {
    fn name(self) -> &'static str {
        match self {
            Kind::Bell => "bell",
            Kind::LongTail => "long tail",
            Kind::TwoHumps => "two humps",
            Kind::Flat => "flat",
            Kind::Peaked => "peaked",
            Kind::HeavyTails => "heavy tails",
            Kind::Mixture => "your own",
        }
    }
}

/// The shapes' parameters, each kept while another shape is shown.
#[derive(Clone, Copy, Debug, PartialEq)]
struct Params {
    /// The long tail: the standard deviation of its logarithm.
    sigma: f64,
    /// Two humps: where the bells are (at −a and a).
    apart: f64,
    /// Heavy tails: Student's t's degrees of freedom.
    df: f64,
    /// The visitor's mixture: where the second bell is, and its share.
    at: f64,
    weight: f64,
}

impl Default for Params {
    fn default() -> Params {
        Params { sigma: 0.6, apart: 0.9, df: 3.0, at: 3.0, weight: 0.25 }
    }
}

/// One distribution, with everything the scene says about it worked out exactly (or,
/// for a mixture's peaks and median, by a search to many digits).
#[derive(Clone, Copy, Debug, PartialEq)]
enum Shape {
    /// The standard normal distribution.
    Bell,
    /// Log-normal: e to the power of a normal number with mean 0 and this spread.
    LongTail { sigma: f64 },
    /// Bells at −a and a, each with spread √(1 − a²): mean 0 and variance 1, as
    /// `math::causal::Noise::TwoHumps` has them at a = 0.9.
    TwoHumps { apart: f64 },
    /// Uniform from −√3 to √3: mean 0, variance 1.
    Flat,
    /// Laplace with variance 1: a sharp peak and tails falling like e^(−|x|√2).
    Peaked,
    /// Student's t; with 1 degree of freedom, the Cauchy distribution.
    HeavyTails { df: f64 },
    /// A bell at 0 with share 1 − w and one at `at` with share w, both of spread 0.6.
    Mixture { at: f64, weight: f64 },
}

impl Shape {
    fn of(kind: Kind, p: &Params) -> Shape {
        match kind {
            Kind::Bell => Shape::Bell,
            Kind::LongTail => Shape::LongTail { sigma: p.sigma },
            Kind::TwoHumps => Shape::TwoHumps { apart: p.apart },
            Kind::Flat => Shape::Flat,
            Kind::Peaked => Shape::Peaked,
            Kind::HeavyTails => Shape::HeavyTails { df: p.df },
            Kind::Mixture => Shape::Mixture { at: p.at, weight: p.weight },
        }
    }

    /// The shapes made of bells: (share, centre, spread) of each.
    fn bells(self) -> Option<[(f64, f64, f64); 2]> {
        match self {
            Shape::Bell => Some([(1.0, 0.0, 1.0), (0.0, 0.0, 1.0)]),
            Shape::TwoHumps { apart } => {
                let s = (1.0 - apart * apart).max(1e-6).sqrt();
                Some([(0.5, -apart, s), (0.5, apart, s)])
            }
            Shape::Mixture { at, weight } => Some([(1.0 - weight, 0.0, BELL_SD), (weight, at, BELL_SD)]),
            _ => None,
        }
    }

    fn pdf(self, x: f64) -> f64 {
        if let Some(b) = self.bells() {
            return b.iter().map(|&(w, m, s)| w * normal_pdf((x - m) / s) / s).sum();
        }
        match self {
            Shape::LongTail { sigma } => Dist::LogNormal { mu: 0.0, sigma }.pdf(x),
            Shape::Flat if x.abs() <= SQRT3 => 0.5 / SQRT3,
            Shape::Flat => 0.0,
            Shape::Peaked => (-x.abs() / LAPLACE_B).exp() / (2.0 * LAPLACE_B),
            Shape::HeavyTails { df } => Dist::StudentT { df }.pdf(x),
            _ => 0.0,
        }
    }

    fn cdf(self, x: f64) -> f64 {
        if let Some(b) = self.bells() {
            return b.iter().map(|&(w, m, s)| w * normal_cdf((x - m) / s)).sum();
        }
        match self {
            Shape::LongTail { sigma } => Dist::LogNormal { mu: 0.0, sigma }.cdf(x),
            Shape::Flat => ((x + SQRT3) / (2.0 * SQRT3)).max(0.0).min(1.0),
            Shape::Peaked => {
                if x < 0.0 {
                    0.5 * (x / LAPLACE_B).exp()
                } else {
                    1.0 - 0.5 * (-x / LAPLACE_B).exp()
                }
            }
            Shape::HeavyTails { df } => Dist::StudentT { df }.cdf(x),
            _ => 0.5,
        }
    }

    /// The value with a share u of the distribution below it. `guess`, a value near it
    /// (the same point before a parameter moved), saves the search steps.
    fn quantile(self, u: f64, guess: Option<f64>) -> f64 {
        let u = u.max(1e-12).min(1.0 - 1e-12);
        let z = normal_quantile(u);
        if let Some(b) = self.bells() {
            // (a share u of a mixture lies between the bells' own points for u)
            let (lo, hi) = b.iter().filter(|p| p.0 > 1e-12).fold((f64::INFINITY, f64::NEG_INFINITY), |(lo, hi), &(_, m, s)| (lo.min(m + s * z), hi.max(m + s * z)));
            if (hi - lo).is_nan() || hi - lo <= 1e-12 {
                return lo;
            }
            return solve(u, lo, hi, guess, |x| self.cdf(x), |x| self.pdf(x));
        }
        match self {
            Shape::LongTail { sigma } => (sigma * z).exp(),
            Shape::Flat => SQRT3 * (2.0 * u - 1.0),
            Shape::Peaked => {
                if u < 0.5 {
                    LAPLACE_B * (2.0 * u).ln()
                } else {
                    -LAPLACE_B * (2.0 * (1.0 - u)).ln()
                }
            }
            Shape::HeavyTails { df } if (df - 1.0).abs() < 1e-12 => (std::f64::consts::PI * (u - 0.5)).tan(),
            // (symmetric: and near 0 the cumulative probability is good to eight digits only)
            Shape::HeavyTails { .. } if u == 0.5 => 0.0,
            Shape::HeavyTails { df } => {
                let t = Dist::StudentT { df };
                // (t's points lie further out than the normal's)
                solve(u, z - 1.0, z + 1.0, guess, |x| t.cdf(x), |x| t.pdf(x))
            }
            _ => z,
        }
    }

    fn mean(self) -> Value {
        if let Some(b) = self.bells() {
            return Value::Is(b.iter().map(|&(w, m, _)| w * m).sum());
        }
        match self {
            Shape::LongTail { sigma } => Value::Is((0.5 * sigma * sigma).exp()),
            Shape::HeavyTails { df } if df <= 1.0 => Value::Undefined,
            _ => Value::Is(0.0),
        }
    }

    /// The variance, the skewness and the excess kurtosis.
    fn shape(self) -> [Value; 3] {
        use Value::*;
        if let Some(b) = self.bells() {
            // the moments about the mean of bells with centres m and spreads s
            let mu: f64 = b.iter().map(|&(w, m, _)| w * m).sum();
            let (mut m2, mut m3, mut m4) = (0.0, 0.0, 0.0);
            for &(w, m, s) in &b {
                let d = m - mu;
                m2 += w * (d * d + s * s);
                m3 += w * (d * d * d + 3.0 * d * s * s);
                m4 += w * (d.powi(4) + 6.0 * d * d * s * s + 3.0 * s.powi(4));
            }
            return [Is(m2), Is(m3 / m2.powf(1.5)), Is(m4 / (m2 * m2) - 3.0)];
        }
        match self {
            Shape::LongTail { sigma } => {
                let e = (sigma * sigma).exp();
                [Is((e - 1.0) * e), Is((e + 2.0) * (e - 1.0).sqrt()), Is(e.powi(4) + 2.0 * e.powi(3) + 3.0 * e * e - 6.0)]
            }
            Shape::Flat => [Is(1.0), Is(0.0), Is(-1.2)],
            Shape::Peaked => [Is(1.0), Is(0.0), Is(3.0)],
            Shape::HeavyTails { df } => {
                let variance = if df > 2.0 {
                    Is(df / (df - 2.0))
                } else if df > 1.0 {
                    Infinite
                } else {
                    Undefined
                };
                let skew = if df > 3.0 { Is(0.0) } else { Undefined };
                let kurtosis = if df > 4.0 {
                    Is(6.0 / (df - 4.0))
                } else if df > 2.0 {
                    Infinite
                } else {
                    Undefined
                };
                [variance, skew, kurtosis]
            }
            _ => [Undefined; 3],
        }
    }

    /// The standard deviation, where it is finite.
    fn sd(self) -> Option<f64> {
        self.shape()[0].get().map(f64::sqrt)
    }

    fn median(self) -> f64 {
        match self {
            Shape::LongTail { .. } => 1.0,
            Shape::Mixture { .. } => self.quantile(0.5, None),
            // (the rest are symmetric about 0)
            _ => 0.0,
        }
    }

    fn peaks(self) -> Peaks {
        match self {
            Shape::LongTail { sigma } => Peaks::One((-sigma * sigma).exp()),
            Shape::Flat => Peaks::Flat(-SQRT3, SQRT3),
            Shape::TwoHumps { apart } if apart * apart > 0.5 => {
                // two peaks once the bells are further apart than twice their spread:
                // where the slopes of the two bells cancel, between 0 and a
                let s2 = 1.0 - apart * apart;
                let slope = |x: f64| (x - apart) * (-(x - apart).powi(2) / (2.0 * s2)).exp() + (x + apart) * (-(x + apart).powi(2) / (2.0 * s2)).exp();
                let (mut lo, mut hi) = (1e-9, apart);
                for _ in 0..80 {
                    let mid = 0.5 * (lo + hi);
                    if slope(mid) < 0.0 {
                        lo = mid;
                    } else {
                        hi = mid;
                    }
                }
                let m = 0.5 * (lo + hi);
                Peaks::Two(-m, m)
            }
            Shape::Mixture { at, .. } => {
                let found = local_peaks(|x| self.pdf(x), at.min(0.0) - 4.0 * BELL_SD, at.max(0.0) + 4.0 * BELL_SD, 1200);
                match found[..] {
                    [(a, _), (b, _), ..] => Peaks::Main(a, b),
                    [(a, _)] => Peaks::One(a),
                    [] => Peaks::One(0.0),
                }
            }
            _ => Peaks::One(0.0),
        }
    }

    /// The turning moment of the part of the distribution above p, about p: the mean
    /// of how far beyond p it lies, counting what lies below as 0.
    fn beyond(self, p: f64) -> Value {
        if let Some(b) = self.bells() {
            return Value::Is(
                b.iter()
                    .map(|&(w, m, s)| {
                        let z = (p - m) / s;
                        w * (s * normal_pdf(z) + (m - p) * normal_cdf(-z))
                    })
                    .sum(),
            );
        }
        match self {
            Shape::LongTail { sigma } => {
                let mean = (0.5 * sigma * sigma).exp();
                if p <= 0.0 {
                    return Value::Is(mean - p);
                }
                // (the formula of Black and Scholes for a call option)
                let d2 = -p.ln() / sigma;
                Value::Is(mean * normal_cdf(d2 + sigma) - p * normal_cdf(d2))
            }
            Shape::Flat => Value::Is(if p >= SQRT3 {
                0.0
            } else if p <= -SQRT3 {
                -p
            } else {
                (SQRT3 - p).powi(2) / (4.0 * SQRT3)
            }),
            Shape::Peaked => Value::Is(if p >= 0.0 { 0.5 * LAPLACE_B * (-p / LAPLACE_B).exp() } else { -p + 0.5 * LAPLACE_B * (p / LAPLACE_B).exp() }),
            Shape::HeavyTails { df } if df > 1.0 => {
                // ∫ x f(x) dx from p on is (ν + p²) f(p) / (ν − 1) for Student's t
                let t = Dist::StudentT { df };
                Value::Is((df + p * p) / (df - 1.0) * t.pdf(p) - p * (1.0 - t.cdf(p)))
            }
            _ => Value::Infinite,
        }
    }

    /// The turning moment of the part below p, about p.
    fn below(self, p: f64) -> Value {
        match (self.beyond(p), self.mean()) {
            (Value::Is(r), Value::Is(m)) => Value::Is((r - (m - p)).max(0.0)),
            _ => Value::Infinite,
        }
    }

    /// The share within k standard deviations of the mean, where both are finite.
    fn within(self, k: f64) -> Option<f64> {
        let (m, s) = (self.mean().get()?, self.sd()?);
        Some(self.cdf(m + k * s) - self.cdf(m - k * s))
    }

    /// The values the charts show for this shape.
    fn range(self) -> [f64; 2] {
        match self {
            Shape::LongTail { sigma } => [-0.5, ((2.6 * sigma).exp() * 1.05).max(3.0)],
            Shape::HeavyTails { .. } => [-6.0, 6.0],
            Shape::Mixture { .. } => [-3.0, 5.0],
            _ => [-4.0, 4.0],
        }
    }

    /// How high the density chart reaches: above the density, with room for the names
    /// at the top. The four shapes with mean 0 and variance 1 share a scale, so that
    /// they can be compared.
    fn top(self) -> f64 {
        match self {
            Shape::LongTail { sigma } => 1.3 * self.pdf((-sigma * sigma).exp()),
            Shape::HeavyTails { .. } => 0.6,
            Shape::Mixture { .. } => 0.95,
            _ => 0.9,
        }
    }
}

/// The x with cdf(x) = u, by Newton's steps kept inside a bracket (halving it when a
/// step would leave it); the bracket starts at `lo` to `hi` and is widened until it
/// holds the answer.
fn solve(u: f64, mut lo: f64, mut hi: f64, guess: Option<f64>, cdf: impl Fn(f64) -> f64, pdf: impl Fn(f64) -> f64) -> f64 {
    let mut w = (hi - lo).max(1.0);
    for _ in 0..80 {
        if cdf(lo) <= u {
            break;
        }
        lo -= w;
        w *= 2.0;
    }
    let mut w = (hi - lo).max(1.0);
    for _ in 0..80 {
        if cdf(hi) >= u {
            break;
        }
        hi += w;
        w *= 2.0;
    }
    let mut x = guess.filter(|g| *g > lo && *g < hi).unwrap_or(0.5 * (lo + hi));
    for _ in 0..100 {
        let e = cdf(x) - u;
        if e > 0.0 {
            hi = x;
        } else {
            lo = x;
        }
        let d = pdf(x);
        let next = if d > 0.0 && d.is_finite() { x - e / d } else { f64::NAN };
        let next = if next > lo && next < hi { next } else { 0.5 * (lo + hi) };
        if (next - x).abs() <= 1e-13 * (1.0 + x.abs()) || hi - lo <= 1e-13 * (1.0 + x.abs()) {
            return next;
        }
        x = next;
    }
    x
}

/// The local peaks of a density from `lo` to `hi`, highest first: looked for at `n`
/// points and each then found to many digits (by golden sections).
fn local_peaks(pdf: impl Fn(f64) -> f64, lo: f64, hi: f64, n: usize) -> Vec<(f64, f64)> {
    let n = n.max(3);
    let h = (hi - lo) / n as f64;
    let ys: Vec<f64> = (0..=n).map(|i| pdf(lo + h * i as f64)).collect();
    let top = ys.iter().copied().fold(0.0, f64::max);
    let mut out = Vec::new();
    for i in 1..n {
        if ys[i] > ys[i - 1] && ys[i] >= ys[i + 1] && ys[i] > 1e-9 * top {
            let (mut a, mut b) = (lo + h * (i - 1) as f64, lo + h * (i + 1) as f64);
            let g = 0.5 * (5f64.sqrt() - 1.0);
            for _ in 0..60 {
                let (c, d) = (b - g * (b - a), a + g * (b - a));
                if pdf(c) > pdf(d) {
                    b = d;
                } else {
                    a = c;
                }
            }
            let x = 0.5 * (a + b);
            out.push((x, pdf(x)));
        }
    }
    out.sort_by(|a, b| b.1.total_cmp(&a.1));
    out
}

/// What is drawn while one shape turns into another: a mixture of the shapes, each with
/// its share. (A mixture of shapes that share a mean and a variance has that mean and
/// that variance too, so a change among the four shapes with mean 0 and spread 1 keeps
/// the beam level and the spread the same all the way.)
#[derive(Clone, Debug, PartialEq)]
struct Blend {
    parts: Vec<(Shape, f64)>,
}

/// A sum of values with weights: infinite if any is, undefined if any is.
fn weighted(parts: impl Iterator<Item = (Value, f64)>) -> Value {
    let (mut sum, mut infinite) = (0.0, false);
    for (v, w) in parts.filter(|p| p.1 > 0.0) {
        match v {
            Value::Is(x) => sum += w * x,
            Value::Infinite => infinite = true,
            Value::Undefined => return Value::Undefined,
        }
    }
    if infinite {
        Value::Infinite
    } else {
        Value::Is(sum)
    }
}

impl Blend {
    fn pdf(&self, x: f64) -> f64 {
        self.parts.iter().map(|(s, w)| w * s.pdf(x)).sum()
    }

    fn cdf(&self, x: f64) -> f64 {
        self.parts.iter().map(|(s, w)| w * s.cdf(x)).sum()
    }

    fn mean(&self) -> Value {
        weighted(self.parts.iter().map(|(s, w)| (s.mean(), *w)))
    }

    fn sd(&self) -> Option<f64> {
        let m = self.mean().get()?;
        // the mean of X² is each part's variance and squared mean, by their shares
        let mut second = 0.0;
        for (s, w) in self.parts.iter().filter(|p| p.1 > 0.0) {
            second += w * (s.shape()[0].get()? + s.mean().get()?.powi(2));
        }
        Some((second - m * m).max(0.0).sqrt())
    }

    fn beyond(&self, p: f64) -> Value {
        weighted(self.parts.iter().map(|(s, w)| (s.beyond(p), *w)))
    }

    fn below(&self, p: f64) -> Value {
        weighted(self.parts.iter().map(|(s, w)| (s.below(p), *w)))
    }

    /// The value with a share u below it, by halving (it lies between the parts' own).
    fn quantile(&self, u: f64) -> f64 {
        let (mut lo, mut hi) = self.parts.iter().filter(|p| p.1 > 0.0).fold((f64::INFINITY, f64::NEG_INFINITY), |(lo, hi), (s, _)| {
            let q = s.quantile(u, None);
            (lo.min(q), hi.max(q))
        });
        if hi.is_nan() || lo.is_nan() || hi <= lo {
            return if lo.is_finite() { lo } else { 0.0 };
        }
        for _ in 0..60 {
            let mid = 0.5 * (lo + hi);
            if self.cdf(mid) < u {
                lo = mid;
            } else {
                hi = mid;
            }
        }
        0.5 * (lo + hi)
    }

    fn median(&self) -> f64 {
        match self.parts[..] {
            [(s, _)] => s.median(),
            _ => self.quantile(0.5),
        }
    }
}

// ----- a sample's summaries -----------------------------------------------------------

/// What the table says of a sample.
#[derive(Clone, Debug, PartialEq)]
struct Summary {
    n: usize,
    mean: f64,
    median: f64,
    /// The root of the mean squared distance from the mean: dividing by n.
    sd: f64,
    skewness: Option<f64>,
    kurtosis: Option<f64>,
    /// The shares within one, two and three standard deviations of the mean, and
    /// beyond k of them.
    within: [Option<f64>; 3],
    beyond: Option<f64>,
}

fn summary(x: &[f64], k: f64) -> Summary {
    let (mean, sd) = (stats::mean(x), stats::variance_n(x).sqrt());
    let spread = x.len() >= 2 && sd > 0.0;
    let share = |inside: &dyn Fn(f64) -> bool| (spread).then(|| x.iter().filter(|v| inside(**v)).count() as f64 / x.len() as f64);
    Summary {
        n: x.len(),
        mean,
        median: stats::median(x),
        sd,
        skewness: spread.then(|| stats::skewness(x)),
        kurtosis: spread.then(|| stats::excess_kurtosis(x)),
        within: [1.0, 2.0, 3.0].map(|j| share(&|v| (v - mean).abs() <= j * sd)),
        beyond: share(&|v| (v - mean).abs() >= k * sd),
    }
}

/// The mean squared distance of the points from p.
fn msd(x: &[f64], p: f64) -> f64 {
    if x.is_empty() {
        return 0.0;
    }
    x.iter().map(|v| (v - p) * (v - p)).sum::<f64>() / x.len() as f64
}

/// The mean distance of the points from p.
fn mad(x: &[f64], p: f64) -> f64 {
    if x.is_empty() {
        return 0.0;
    }
    x.iter().map(|v| (v - p).abs()).sum::<f64>() / x.len() as f64
}

/// The first n uniform numbers of a seed: the sample is the shape's quantiles of them,
/// so that a sample keeps its points' order when the shape changes, and each point
/// slides to its new place.
fn uniforms(seed: u64) -> Vec<f64> {
    let mut rng = Rng::new(seed);
    (0..MAX_N).map(|_| stats::unit(&mut rng)).collect()
}

// ----- what glides ----------------------------------------------------------------------

/// Numbers drawn gliding to where they have gone, through two eases (as `plot::Glide`),
/// one of which can be put somewhere at once: a point that is being dragged follows the
/// hand, the rest glide.
#[derive(Clone, Debug, Default)]
struct Ease {
    between: Vec<f32>,
    shown: Vec<f32>,
}

impl Ease {
    /// Bring the numbers `dt` seconds nearer to `to`, `tau` seconds being one ease. A list
    /// that has grown or shrunk keeps what it had in common, and new numbers start where
    /// they are going. Says whether anything is still on its way.
    fn follow(&mut self, to: &[f32], dt: f32, tau: f32) -> bool {
        if self.shown.len() != to.len() {
            let n = self.shown.len().min(to.len());
            self.shown.truncate(n);
            self.between.truncate(n);
            self.shown.extend_from_slice(&to[n..]);
            self.between.extend_from_slice(&to[n..]);
        }
        if self.shown == to && self.between == to {
            return false;
        }
        let k = 1.0 - (-dt.max(0.0) / tau.max(1e-3)).exp();
        let mut away = false;
        for ((b, s), t) in self.between.iter_mut().zip(self.shown.iter_mut()).zip(to) {
            *b += (t - *b) * k;
            *s += (*b - *s) * k;
            let near = 2e-4 * (1.0 + t.abs());
            away |= !((t - *s).abs() <= near && (t - *b).abs() <= near);
        }
        // (exactly there in the end: the picture then stops changing)
        if !away {
            self.snap(to);
        }
        away
    }

    fn snap(&mut self, to: &[f32]) {
        self.shown = to.to_vec();
        self.between = to.to_vec();
    }

    fn set(&mut self, i: usize, v: f32) {
        if let (Some(s), Some(b)) = (self.shown.get_mut(i), self.between.get_mut(i)) {
            (*s, *b) = (v, v);
        }
    }

    fn get(&self, i: usize) -> f32 {
        self.shown.get(i).copied().unwrap_or(0.0)
    }
}

/// The numbers of `look`, by place.
const X0: usize = 0;
const X1: usize = 1;
const TOP: usize = 2;
const CURVE_TOP: usize = 3;
const TILT: usize = 4;
const PIVOT_AT: usize = 5;
const SAMPLE_MEAN: usize = 6;
const SAMPLE_MEDIAN: usize = 7;
const LOOKS: usize = 8;

/// How far the beam tips at most (radians: about eight degrees); less where its far end
/// would drop further than the room under the chart (see `Moments::draw_density`).
const TILT_MOST: f32 = 0.14;

/// One ease of the picture, and of a change of shape (seconds).
const GLIDE: f32 = 0.07;
const MORPH: f32 = 0.13;

// ----- colours, buttons and experiments --------------------------------------------------

/// The pivot and the bar of one standard deviation.
const PIVOT: Color = pal::WHITE;
/// Squares, their average and the sample mean.
const SQUARES: Color = ESTIMATE;
/// Distances, their average and the sample median.
const DISTANCES: Color = OTHER;
const TAILS: Color = ALARM;
/// The distribution's own marks (mean, median, mode), lighter than its fill.
fn truth_mark() -> Color {
    TRUTH.mix(pal::WHITE, 0.35)
}

const ID_PIVOT: u32 = 1;
const ID_HUMP: u32 = 2;
const ID_K: u32 = 3;
const ID_POINT: u32 = 100;

/// The shapes the third experiment passes through: mean 0 and standard deviation 1
/// each.
const TOUR: [Kind; 4] = [Kind::Bell, Kind::Flat, Kind::TwoHumps, Kind::Peaked];
/// How long the tour stays at each shape once it is there (seconds of scene time).
const HOLD: f32 = 1.6;

/// The draws a second when the sample is drawn one point at a time: faster as it grows,
/// so that the first few can be followed and two hundred take about seven seconds.
fn grow_rate(count: usize) -> f32 {
    8.0 + 0.3 * count as f32
}

/// The seeds the experiments draw their samples with.
const SEED_OPENING: u64 = 1;
const SEED_BALANCE: u64 = 3;
const SEED_TAIL: u64 = 11;
const SEED_FOUR: u64 = 2;
const SEED_BOUND: u64 = 5;
const SEED_CAUCHY: u64 = 21;
/// The long tail's experiment: its sample's size, and how far out its furthest point is
/// moved, and when (seconds after the button).
const TAIL_N: usize = 20;
const TAIL_FAR: f64 = 7.0;
const TAIL_WAIT: f32 = 1.2;

/// The experiments: (button, what to look for).
const TRIALS: [(&str, &str); 5] = [
    (
        "the balance point",
        "A large hump and a small one far to its right. The pivot is under the peak, and the beam tips right: each side turns the beam by its area times its distance from the pivot, added up, and the small hump is three units out (turning moments 0.18 on the left, 0.93 on the right). Drag the pivot right: the beam comes level only at the mean, 0.75, at neither peak. Then drag the small hump's top (the green dot): the mean follows it.",
    ),
    (
        "a long tail",
        "A log-normal distribution, the shape of incomes: most values small, a few far out. Its peak is at 0.57 and its median at 1.00, but it balances at its mean, 1.32: the far values pull the mean right. Then watch the sample of 20: a moment after it settles, its furthest point is moved from 3.63 to 7.00. The sample mean follows it, from 1.21 to 1.38; the median stays at 0.89. That point's square grows from 5.3 to 32.2, and the average square from 0.82 to 2.16.",
    ),
    (
        "same mean, same spread",
        r"Four shapes in turn, each with mean 0 and standard deviation 1: a bell, a flat block, two humps and a sharp peak. The beam stays level at 0 and the bar of one standard deviation keeps its length all the way, yet the shapes are nothing alike. The share within one standard deviation of the mean: 68.3 %, 57.7 %, 59.1 % and 75.7 %. The excess kurtosis tells them apart: 0, $-1.2$, $-1.31$ and 3.",
    ),
    (
        r"at most one in $k^2$",
        r"Pink: what lies $k = 2$ or more standard deviations from the mean. Chebyshev's inequality: for any distribution with a finite variance this is at most $1/k^2$, 25 % at $k = 2$ and 11.1 % at $k = 3$. Here (peaked) it is 5.9 % and 1.4 %; bell 4.6 % and 0.27 %; long tail 4.5 % and 1.7 %; heavy tails 4.1 % and 1.4 %; two humps 0.58 % and under 0.001 %; flat 0 and 0. Never above the bound, and well below it: only a shape with all its weight at the mean and at exactly $k$ standard deviations either side reaches it. Drag the pink marks to change $k$.",
    ),
    (
        "no balance point",
        r"Student's $t$ with one degree of freedom: the Cauchy distribution. It looks like a bell, but its tails are so heavy that each side's turning moment is infinite: no balance point, so no mean, and no variance. A sample of 200 is drawn one point at a time. Its mean (yellow) jumps at every draw far out: draw 35 is 94 and the mean jumps by 2.7, draw 47 is 132 and it jumps by 2.8, and after 200 draws it is at 1.29. It never settles: the mean of any number of Cauchy draws is spread exactly as one draw is. The median (orange) stays between $-0.04$ and 0.24 from the 50th draw on.",
    ),
];

// ----- the layout ------------------------------------------------------------------------

const NO_RECT: Rect = (Vec2::ZERO, Vec2::ZERO);

/// The room each part takes above and below its chart (pixels).
#[derive(Clone, Copy, Debug)]
struct Gaps {
    /// Above the density: its title.
    title_d: f32,
    /// Below the density: the pivot, the axis and its numbers.
    under_d: f32,
    /// Above the sample: its title and the two buttons.
    title_s: f32,
    /// Below the sample.
    gap: f32,
    /// Above the chart of the average square: its title and the name of its axis.
    title_c: f32,
    /// Below it: the axis's numbers and name.
    under_c: f32,
}

impl Gaps {
    fn of(narrow: bool) -> Gaps {
        if narrow {
            Gaps { title_d: 20.0, under_d: 48.0, title_s: 24.0, gap: 8.0, title_c: 34.0, under_c: 34.0 }
        } else {
            Gaps { title_d: 24.0, under_d: 64.0, title_s: 30.0, gap: 10.0, title_c: 42.0, under_c: 40.0 }
        }
    }
}

/// Where everything goes on a canvas.
#[derive(Clone, Copy, Debug)]
struct Layout {
    view: Vec2,
    narrow: bool,
    /// Text sizes: titles, words, small print.
    title: f32,
    text: f32,
    small: f32,
    gaps: Gaps,
    shapes: [Rect; 7],
    modes: [Rect; 2],
    density: Rect,
    strip: Rect,
    curve: Rect,
    table: Rect,
    /// The backing of the column of charts.
    column: Rect,
    /// Too small a canvas for more than the beam.
    compact: bool,
}

impl Layout {
    fn of(view: Vec2, inset: f32) -> Layout {
        let view = Vec2::new(view.x.max(1.0), view.y.max(1.0));
        let narrow = view.x < 760.0;
        let (title, text, small) = if narrow { (11.5, 11.0, 10.0) } else { (13.0, 12.0, 11.0) };
        let gaps = Gaps::of(narrow);
        let side = if narrow { 8.0 } else { 18.0 };
        let top = inset.max(0.0) + if narrow { 6.0 } else { 10.0 };
        let table_w = if narrow { 0.0 } else { (view.x * 0.27).max(230.0).min(300.0) };
        // (room at the left for the numbers on the last chart's axis)
        let l = side + if narrow { 30.0 } else { 38.0 };
        let r = if narrow { view.x - side - 6.0 } else { view.x - side - table_w - 26.0 };
        let row = if narrow { 22.0 } else { 24.0 };
        let mut shapes = [NO_RECT; 7];
        if narrow {
            let rows = [plot::even_row(4, side, view.x - side, top, row, 5.0), plot::even_row(3, side, view.x - side, top + row + 5.0, row, 5.0)];
            for (k, r) in rows.concat().into_iter().enumerate().take(7) {
                shapes[k] = r;
            }
        } else {
            for (k, r) in plot::even_row(7, side, r, top, row, 6.0).into_iter().enumerate().take(7) {
                shapes[k] = r;
            }
        }
        let y0 = shapes[6].1.y + if narrow { 10.0 } else { 14.0 };
        let table_h = if narrow { 11.0 * (small + 4.0) + 14.0 } else { 0.0 };
        let fixed = gaps.title_d + gaps.under_d + gaps.title_s + gaps.gap + gaps.title_c + gaps.under_c + table_h + 6.0;
        let free = (view.y - y0 - fixed).max(0.0);
        let (dh, sh, ch) = (free * 0.38, free * 0.39, free * 0.23);
        let compact = dh < 70.0 || r - l < 140.0;
        if compact {
            let density = (Vec2::new(l, y0 + gaps.title_d), Vec2::new(r.max(l + 1.0), (view.y - gaps.under_d).max(y0 + gaps.title_d + 1.0)));
            let column = (Vec2::new(side, y0), Vec2::new(r + 10.0, view.y - 4.0));
            return Layout { view, narrow, title, text, small, gaps, shapes, modes: [NO_RECT; 2], density, strip: NO_RECT, curve: NO_RECT, table: NO_RECT, column, compact };
        }
        let mut y = y0 + gaps.title_d;
        let density = (Vec2::new(l, y), Vec2::new(r, y + dh));
        y += dh + gaps.under_d + gaps.title_s;
        let strip = (Vec2::new(l, y), Vec2::new(r, y + sh));
        y += sh + gaps.gap + gaps.title_c;
        let curve = (Vec2::new(l, y), Vec2::new(r, y + ch));
        y += ch + gaps.under_c;
        let column = (Vec2::new(side, y0), Vec2::new(r + 10.0, y));
        let table = if narrow {
            (Vec2::new(side, y + 6.0), Vec2::new(view.x - side, y + 6.0 + table_h))
        } else {
            (Vec2::new(r + 26.0, y0), Vec2::new(view.x - side, y0 + 13.0 * (text + 8.0)))
        };
        // the two buttons at the right of the sample's title
        let (bw, bh) = (if narrow { 62.0 } else { 74.0 }, if narrow { 18.0 } else { 20.0 });
        let by = strip.0.y - gaps.title_s + 2.0;
        let modes = [(Vec2::new(r - 2.0 * bw - 4.0, by), Vec2::new(r - bw - 4.0, by + bh)), (Vec2::new(r - bw, by), Vec2::new(r, by + bh))];
        Layout { view, narrow, title, text, small, gaps, shapes, modes, density, strip, curve, table, column, compact }
    }
}

// ----- the scene -------------------------------------------------------------------------

pub struct Moments {
    kind: Kind,
    params: Params,
    /// The shapes on their way out, with their shares, while the newest comes in (its
    /// share is `morph`'s one number).
    before: Vec<(Shape, f64)>,
    morph: Ease,
    pivot: f64,
    /// Squares on the distances, or the distances.
    squares: bool,
    /// The tails beyond k standard deviations shaded.
    tails: bool,
    k: f64,
    n: usize,
    seed: u64,
    /// The sample's uniform numbers, in the order drawn, and the sample itself (the
    /// shape's quantiles of them, or where a point has been dragged).
    u: Vec<f64>,
    values: Vec<f64>,
    /// How many of the sample's points are in it: all but while it is drawn one at a time.
    count: usize,
    growing: bool,
    /// The chart under the sample shows its mean and median as the draws came in.
    history: bool,
    /// The mean and the median of the first m points, for each m.
    means: Vec<f64>,
    medians: Vec<f64>,
    pace: crate::clock::Pace,
    phase: Phase,
    /// How far the newest point is from the one before to the next (see `Phase`).
    share: f32,
    /// The third experiment's tour: the stop it is at, and how long it has been there.
    tour: Option<(usize, f32)>,
    /// A point to move once this much scene time has passed: (seconds, point, to).
    later: Option<(f32, usize, f64)>,
    /// What the picture shows, gliding: see `X0` and the rest.
    look: Ease,
    /// The sample as drawn.
    points: Ease,
    /// The peaks as drawn (see `peak_names`), and whether they go at once to where they
    /// are going (a parameter moved, by the hand: no glide behind it).
    peaks: Ease,
    peaks_at_once: bool,
    moving: bool,
    /// The round values on the axes last frame, and how many labels there were: see
    /// `Moments::new_ticks`.
    ticks: Vec<f64>,
    /// (the points of the axes' batch, and the labels)
    labels_last: (usize, usize),
}

impl Moments {
    /// (Nothing here needs the graphics device, so the tests can make one too.)
    fn create() -> Moments {
        let mut s = Moments {
            kind: Kind::LongTail,
            params: Params::default(),
            before: Vec::new(),
            morph: Ease::default(),
            pivot: 1.0,
            squares: true,
            tails: false,
            k: 2.0,
            n: 40,
            seed: SEED_OPENING,
            u: uniforms(SEED_OPENING),
            values: Vec::new(),
            count: 40,
            growing: false,
            history: false,
            means: Vec::new(),
            medians: Vec::new(),
            pace: crate::clock::Pace::default(),
            phase: Phase::default(),
            share: 1.0,
            tour: None,
            later: None,
            look: Ease::default(),
            points: Ease::default(),
            peaks: Ease::default(),
            peaks_at_once: true,
            moving: false,
            ticks: Vec::new(),
            labels_last: (0, 0),
        };
        s.morph.snap(&[1.0]);
        s.resample();
        // the opening: the pivot at the median, where half the area is on each side,
        // and the beam still tips
        s.pivot = s.target().median();
        let goals = s.goals();
        s.look.snap(&goals);
        s.points.snap(&s.drawn_values());
        s.glide(0.0);
        s
    }

    /// The shape chosen (the one coming in, while the picture changes).
    fn target(&self) -> Shape {
        Shape::of(self.kind, &self.params)
    }

    /// What is drawn: the shapes on their way out and the one coming in, with their shares.
    fn blend(&self) -> Blend {
        let m = (self.morph.get(0) as f64).max(0.0).min(1.0);
        let mut parts: Vec<(Shape, f64)> = self.before.iter().map(|(s, w)| (*s, w * (1.0 - m))).filter(|p| p.1 > 0.0).collect();
        parts.push((self.target(), if parts.is_empty() { 1.0 } else { m }));
        Blend { parts }
    }

    /// Change to another kind of shape, the picture changing gradually from the one drawn.
    fn choose(&mut self, kind: Kind) {
        let was = self.blend();
        self.kind = kind;
        self.peaks_at_once = false;
        self.before = was.parts;
        self.morph.snap(&[0.0]);
        self.after_change();
    }

    /// The shape's parameters have changed: at once, they change by small steps.
    fn reshape(&mut self) {
        self.peaks_at_once = true;
        self.before.clear();
        self.morph.snap(&[1.0]);
        self.after_change();
    }

    fn after_change(&mut self) {
        let [lo, hi] = self.target().range();
        self.pivot = self.pivot.max(lo).min(hi);
        self.resample();
    }

    /// The sample, from its uniform numbers, for the shape chosen.
    fn resample(&mut self) {
        let shape = self.target();
        let old = std::mem::take(&mut self.values);
        self.values = (0..self.n.min(MAX_N)).map(|i| shape.quantile(self.u[i], old.get(i).copied())).collect();
        self.count = if self.growing { self.count.min(self.n) } else { self.n };
        self.later = None;
        self.refresh();
    }

    /// The running means and medians, after the sample changed.
    fn refresh(&mut self) {
        let (mut sorted, mut sum) = (Vec::with_capacity(self.values.len()), 0.0);
        self.means.clear();
        self.medians.clear();
        for &v in &self.values {
            sum += v;
            let at = sorted.partition_point(|x: &f64| *x < v);
            sorted.insert(at, v);
            self.means.push(sum / sorted.len() as f64);
            self.medians.push(stats::quantile_sorted(&sorted, 0.5));
        }
    }

    /// A new sample (a new seed), all at once or one point at a time.
    fn draw_again(&mut self, one_at_a_time: bool) {
        self.seed = self.seed.wrapping_add(1);
        self.new_sample(one_at_a_time);
    }

    fn new_sample(&mut self, one_at_a_time: bool) {
        self.u = uniforms(self.seed);
        self.growing = one_at_a_time;
        self.history = one_at_a_time;
        self.count = if one_at_a_time { 0 } else { self.n };
        self.pace.reset();
        self.phase.reset();
        self.resample();
    }

    fn set_n(&mut self, n: usize) {
        self.n = n.max(1).min(MAX_N);
        if !self.growing {
            self.count = self.n;
        }
        self.resample();
    }

    /// The sample as it is now (the points drawn so far).
    fn sample(&self) -> &[f64] {
        &self.values[..self.count.min(self.values.len())]
    }

    fn drawn_values(&self) -> Vec<f32> {
        self.values.iter().map(|v| *v as f32).collect()
    }

    /// Set up one of the experiments in `TRIALS`: every setting it depends on.
    fn set_up(&mut self, k: usize) {
        self.tour = None;
        self.growing = false;
        self.history = false;
        self.params = Params::default();
        self.squares = true;
        self.tails = false;
        self.k = 2.0;
        self.n = 40;
        let (kind, seed, pivot) = match k {
            0 => (Kind::Mixture, SEED_BALANCE, 0.0),
            1 => {
                self.params.sigma = 0.75;
                self.n = TAIL_N;
                (Kind::LongTail, SEED_TAIL, (0.5f64 * 0.75 * 0.75).exp())
            }
            2 => (Kind::Bell, SEED_FOUR, 0.0),
            3 => {
                self.tails = true;
                (Kind::Peaked, SEED_BOUND, 0.0)
            }
            _ => {
                // (two hundred squares, most of them cut off, would hide the points coming in)
                self.squares = false;
                self.params.df = 1.0;
                self.n = MAX_N;
                (Kind::HeavyTails, SEED_CAUCHY, 0.0)
            }
        };
        self.seed = seed;
        self.u = uniforms(seed);
        self.count = self.n;
        self.choose(kind);
        self.pivot = pivot;
        match k {
            1 => {
                // the furthest point, moved out a moment after the sample has settled
                let far = (0..self.values.len()).max_by(|&a, &b| self.values[a].total_cmp(&self.values[b])).unwrap_or(0);
                self.later = Some((TAIL_WAIT, far, TAIL_FAR));
            }
            2 => self.tour = Some((0, 0.0)),
            4 => self.new_sample(true),
            _ => {}
        }
    }

    /// The tour, the sample drawn one at a time, and the point moved later: what goes
    /// on by itself in the scene's time.
    fn advance(&mut self, dt: f32, real_dt: f32) {
        if let Some((stop, held)) = self.tour {
            let arrived = self.before.is_empty() || self.morph.get(0) >= 1.0;
            if arrived {
                let held = held + dt;
                self.tour = if held < HOLD {
                    Some((stop, held))
                } else if stop + 1 < TOUR.len() {
                    self.choose(TOUR[stop + 1]);
                    Some((stop + 1, 0.0))
                } else {
                    None
                };
            }
        }
        if let Some((wait, i, to)) = self.later {
            let wait = wait - dt;
            if wait <= 0.0 {
                if let Some(v) = self.values.get_mut(i) {
                    *v = to;
                }
                self.later = None;
                self.refresh();
            } else {
                self.later = Some((wait, i, to));
            }
        }
        if self.growing && dt > 0.0 {
            let rate = grow_rate(self.count);
            if self.count < self.values.len() {
                let mut turn = scene_turn(&mut self.pace, rate, dt, real_dt);
                while self.count < self.values.len() && turn.next() {
                    self.count += 1;
                }
                let taken = self.pace.end(turn);
                self.share = self.phase.after(rate, dt, taken);
            } else {
                // the last point comes all the way in, at the same pace, and then all is still
                self.share = (self.share + rate * dt).min(1.0);
                self.growing = self.share < 1.0;
            }
        }
    }

    /// How far the newest point has come in, while the sample is drawn one point at a
    /// time (1 otherwise): what is worked out from the sample is drawn this share of the
    /// way from its value without that point to its value with it, so that it moves in
    /// every frame and not in steps.
    fn coming(&self) -> f64 {
        if self.growing {
            self.share.max(0.0).min(1.0) as f64
        } else {
            1.0
        }
    }

    /// Something worked out from the sample, as drawn (see `coming`).
    fn drawn_stat(&self, of: impl Fn(&[f64]) -> f64) -> f64 {
        let x = self.sample();
        let now = of(x);
        match (self.coming(), x.len()) {
            (t, n) if n >= 2 && t < 1.0 => {
                let before = of(&x[..n - 1]);
                before + (now - before) * t
            }
            _ => now,
        }
    }

    /// Where the picture's numbers are going: see `X0` and the rest.
    fn goals(&self) -> Vec<f32> {
        let target = self.target();
        let [x0, x1] = target.range();
        let blend = self.blend();
        // tip by the difference of the two sides' turning moments, which is the mean
        // less the pivot: towards the greater, further the greater it is
        let tilt = match blend.mean() {
            Value::Is(m) => ((m - self.pivot) / (0.06 * (x1 - x0))).tanh() as f32,
            _ => 0.0,
        };
        let x = self.sample();
        let s = summary(x, self.k);
        // high enough to show the bottom of the sample's curve well, and the curve up to
        // a few times its least value (or all of it, if it rises less than that)
        let (edges, least) = if self.squares { (msd(x, x0).max(msd(x, x1)), msd(x, s.mean)) } else { (mad(x, x0).max(mad(x, x1)), mad(x, s.median)) };
        let curve_top = edges.min((if self.squares { 4.0 } else { 3.0 } * least).max(0.25 * edges));
        let mut g = vec![0.0f32; LOOKS];
        g[X0] = x0 as f32;
        g[X1] = x1 as f32;
        g[TOP] = target.top() as f32;
        g[CURVE_TOP] = (1.05 * curve_top).max(1e-3) as f32;
        g[TILT] = tilt;
        g[PIVOT_AT] = self.pivot as f32;
        if !x.is_empty() {
            g[SAMPLE_MEAN] = self.drawn_stat(stats::mean) as f32;
            g[SAMPLE_MEDIAN] = self.drawn_stat(stats::median) as f32;
        }
        g
    }

    /// Bring what is drawn nearer to what is so, by the clock (it answers the hand).
    fn glide(&mut self, real_dt: f32) {
        let mut moving = false;
        if !self.before.is_empty() {
            moving |= self.morph.follow(&[1.0], real_dt, MORPH);
            if self.morph.get(0) >= 1.0 {
                self.before.clear();
            }
        }
        let goals = self.goals();
        moving |= self.look.follow(&goals, real_dt, GLIDE);
        // (while the points come in one at a time, the mean and the median move with them)
        if self.growing {
            for i in [SAMPLE_MEAN, SAMPLE_MEDIAN] {
                self.look.set(i, goals[i]);
            }
        }
        moving |= self.points.follow(&self.drawn_values(), real_dt, GLIDE);
        let peaks: Vec<f32> = peak_names(self.target()).iter().map(|p| p.0 as f32).collect();
        if self.peaks_at_once {
            self.peaks.snap(&peaks);
        } else {
            moving |= self.peaks.follow(&peaks, real_dt, GLIDE);
        }
        self.moving = moving;
    }
}

// ----- drawing ---------------------------------------------------------------------------

/// The beam's turn about the pivot: a point of the chart as the tipped beam carries it.
#[derive(Clone, Copy)]
struct Turn {
    o: Vec2,
    c: f32,
    s: f32,
}

impl Turn {
    fn new(o: Vec2, angle: f32) -> Turn {
        Turn { o, c: angle.cos(), s: angle.sin() }
    }

    fn on(&self, q: Vec2) -> Vec2 {
        let d = q - self.o;
        self.o + Vec2::new(d.x * self.c - d.y * self.s, d.x * self.s + d.y * self.c)
    }

    fn off(&self, q: Vec2) -> Vec2 {
        let d = q - self.o;
        self.o + Vec2::new(d.x * self.c + d.y * self.s, -d.x * self.s + d.y * self.c)
    }
}

fn quad(f: &mut Frame, level: usize, c: [Vec2; 4], color: Color) {
    if c.iter().all(|p| p.is_finite()) {
        let h = c.map(|p| f.hud_at(level, p));
        f.dl.tris_alpha.quad(h[0], h[1], h[2], h[3], color);
    }
}

fn tri(f: &mut Frame, level: usize, c: [Vec2; 3], color: Color) {
    if c.iter().all(|p| p.is_finite()) {
        let h = c.map(|p| f.hud_at(level, p));
        f.dl.tris_alpha.tri(h[0], h[1], h[2], color);
    }
}

/// A dot that can be dragged, `size` pixels across: where the pointer has taken it,
/// while it is being dragged.
fn grip(f: &mut Frame, id: u32, at: Vec2, color: Color, size: f32) -> Option<Vec2> {
    if !at.is_finite() {
        return None;
    }
    let mut p = f.cam.pick_ground(at * f.px)?;
    if f.drag3_sized(id, &mut p, Some(Vec3::Z), color, size) {
        on_screen(f, p)
    } else {
        None
    }
}

/// A name on a plate in one of the rows at the top of a chart, centred over `x` as far
/// as the chart allows. Returns where its foot is, over `x`, for a line down to what it
/// names.
#[allow(clippy::too_many_arguments)]
fn row_label(f: &mut Frame, rect: Rect, row: usize, x: f32, text: &str, size: f32, color: Color) -> Vec2 {
    let w = f.text_size(text, size).x;
    let cx = x.max(rect.0.x + 0.5 * w + 2.0).min(rect.1.x - 0.5 * w - 2.0);
    let y = rect.0.y + 3.0 + row as f32 * (size + 6.0);
    f.caption_px(Vec2::new(cx, y), text, size, color, Align2::CENTER_TOP);
    Vec2::new(x, y + size + 4.0)
}

/// A text broken into lines no wider than `width` pixels (a formula kept whole).
fn lines_px(f: &Frame, text: &str, size: f32, width: f32) -> Vec<String> {
    let mut words: Vec<String> = Vec::new();
    let mut inside = false;
    for piece in text.split(' ') {
        match words.last_mut() {
            Some(last) if inside => {
                last.push(' ');
                last.push_str(piece);
            }
            _ => words.push(piece.to_string()),
        }
        if piece.matches('$').count() % 2 == 1 {
            inside = !inside;
        }
    }
    let mut lines: Vec<String> = Vec::new();
    for word in words {
        match lines.last_mut() {
            Some(line) if f.text_size(&format!("{line} {word}"), size).x <= width => {
                line.push(' ');
                line.push_str(&word);
            }
            _ => lines.push(word),
        }
    }
    lines
}

/// A curve with one bottom (its value falls to the least at `least` and rises after),
/// drawn where it is inside its chart, between the places where it crosses the top, and
/// at the same number of points in every frame. (`Plot::curve` cuts a curve at the
/// edge: as the chart's height glides, a point cut off on one side and one gained on
/// the other would be taken for the same points moved.)
fn convex_curve(plot: &Plot, f: &mut Frame, width: f32, color: Color, g: impl Fn(f64) -> f64, least: f64) {
    let ([x0, x1], top) = (plot.x, plot.y[1]);
    let bottom = if least.is_finite() { least.max(x0).min(x1) } else { 0.5 * (x0 + x1) };
    let lowest = g(bottom);
    if lowest.is_nan() || lowest > top {
        return;
    }
    // (from the bottom out to where the curve leaves the chart, by halving)
    let cross = |mut inside: f64, mut outside: f64| {
        if g(outside) <= top {
            return outside;
        }
        for _ in 0..50 {
            let mid = 0.5 * (inside + outside);
            if g(mid) <= top {
                inside = mid;
            } else {
                outside = mid;
            }
        }
        inside
    };
    let (a, b) = (cross(bottom, x0), cross(bottom, x1));
    let n = 200;
    plot.path(f, CURVES, (0..n).map(|i| a + (b - a) * i as f64 / (n - 1) as f64).map(|x| (x, g(x).min(top))), width, color);
}

/// A shape's peaks in order across, each with what the picture calls it.
fn peak_names(s: Shape) -> Vec<(f64, String)> {
    match s.peaks() {
        Peaks::One(a) => vec![(a, format!("mode {}: the peak", num(a, 2)))],
        Peaks::Two(a, b) => vec![(a, format!("two peaks {}", num(a, 2))), (b, format!("and {}", num(b, 2)))],
        Peaks::Main(a, b) => {
            let (main, lower) = ((a, format!("mode {}", num(a, 2))), (b, format!("a lower peak {}", num(b, 2))));
            if a < b {
                vec![main, lower]
            } else {
                vec![lower, main]
            }
        }
        Peaks::Flat(..) => Vec::new(),
    }
}

/// A percentage, with as many digits as a small one needs.
fn percent(v: f64) -> String {
    let p = 100.0 * v;
    if p == 0.0 {
        "0 %".to_string()
    } else if p < 0.01 {
        "< 0.01 %".to_string()
    } else if p < 1.0 {
        format!("{} %", num(p, 2))
    } else {
        format!("{} %", num(p, 1))
    }
}

impl Moments {
    fn draw_buttons(&mut self, f: &mut Frame, lay: &Layout) {
        for (k, r) in lay.shapes.iter().enumerate() {
            let kind = KINDS[k];
            if plot::button(f, *r, kind.name(), lay.small, self.kind == kind) && self.kind != kind {
                self.tour = None;
                self.choose(kind);
            }
        }
        if !lay.compact {
            for (k, word) in ["squares", "distances"].iter().enumerate() {
                if plot::button(f, lay.modes[k], word, lay.small, self.squares == (k == 0)) {
                    self.squares = k == 0;
                    self.history &= self.growing;
                }
            }
        }
    }

    fn draw_density(&mut self, f: &mut Frame, lay: &Layout, blend: &Blend) {
        let (x0, x1, top) = (self.look.get(X0) as f64, self.look.get(X1) as f64, self.look.get(TOP) as f64);
        let plot = Plot::new(lay.density, [x0, x1], [0.0, top]);
        let (min, max) = plot.rect;
        let (small, text) = (lay.small, lay.text);
        let title = if lay.narrow { "the distribution, on a beam" } else { "the distribution, resting on a beam: drag the pivot (white)" };
        f.text_px(Vec2::new(min.x - 30.0, min.y - 6.0), title, lay.title, pal::TEXT, Align2::LEFT_BOTTOM);
        let pivot = self.look.get(PIVOT_AT) as f64;
        let o = plot.at(pivot, 0.0);
        // the beam's far end drops no further than the room under the chart allows
        let far = (o.x - min.x).max(max.x - o.x).max(1.0);
        let drop = lay.gaps.under_d - 20.0;
        let turn = Turn::new(o, (drop / far).min(1.0).asin().min(TILT_MOST) * self.look.get(TILT));
        let at = |x: f64, y: f64| turn.on(plot.at(x, y));

        let mean = blend.mean().get();
        let sd = blend.sd();
        let median = blend.median();
        let bound = match (mean, sd) {
            (Some(m), Some(s)) if self.tails => Some((m - self.k * s, m + self.k * s)),
            _ => None,
        };
        // the density at evenly spaced places, and at those where a colour changes
        let mut xs: Vec<f64> = (0..240).map(|i| x0 + (x1 - x0) * i as f64 / 239.0).collect();
        let (b0, b1) = bound.unwrap_or((x0, x1));
        xs.extend([median, b0, b1].map(|v| v.max(x0).min(x1)));
        xs.sort_by(f64::total_cmp);
        let ys: Vec<f64> = xs.iter().map(|x| blend.pdf(*x)).map(|y| if y.is_finite() { y.max(0.0).min(top) } else { 0.0 }).collect();
        for i in 0..xs.len() - 1 {
            let mid = 0.5 * (xs[i] + xs[i + 1]);
            let c = [at(xs[i], 0.0), at(xs[i], ys[i]), at(xs[i + 1], ys[i + 1]), at(xs[i + 1], 0.0)];
            quad(f, 1, c, TRUTH.a(if mid < median { 0.36 } else { 0.2 }));
            if bound.is_some_and(|(a, b)| mid < a || mid > b) {
                quad(f, 1, c, TAILS.a(0.55));
            }
        }
        let outline: Vec<Vec2> = xs.iter().zip(&ys).map(|(x, y)| at(*x, *y)).collect();
        plot::path_of(f, CURVES, &outline, 2.2, TRUTH);

        // the visitor's mixture: its two bells, and the second's top to drag
        if let Shape::Mixture { at: c, weight } = self.target() {
            let peak = 1.0 / (BELL_SD * std::f64::consts::TAU.sqrt());
            for (m, w) in [(0.0, 1.0 - weight), (c, weight)] {
                let bell: Vec<Vec2> = (0..80).map(|i| m - 3.5 * BELL_SD + 7.0 * BELL_SD * i as f64 / 79.0).map(|x| at(x, w * normal_pdf((x - m) / BELL_SD) / BELL_SD)).collect();
                plot::path_of(f, CURVES, &bell, 1.0, PIVOT.a(0.35));
            }
            if let Some(to) = grip(f, ID_HUMP, at(c, weight * peak), TRUTH, 15.0) {
                let q = plot.y_at(turn.off(to).y);
                let p = plot.x_at(turn.off(to).x);
                self.params.at = p.max(x0 + 0.5).min(x1 - 0.5);
                self.params.weight = (q / peak).max(0.0).min(1.0);
                self.tour = None;
                self.reshape();
            }
        }

        // the beam, and the pivot under it
        let no_mean = mean.is_none();
        let beam = if no_mean { TAILS.a(0.8) } else { pal::TEXT.a(0.75) };
        plot::line_of(f, MARKS, at(x0, 0.0), at(x1, 0.0), 4.0, beam);
        tri(f, 2, [o + Vec2::new(0.0, 2.0), o + Vec2::new(-10.0, 21.0), o + Vec2::new(10.0, 21.0)], PIVOT.a(0.8));
        if let Some(to) = grip(f, ID_PIVOT, o + Vec2::new(0.0, 21.0), PIVOT, 12.0) {
            self.pivot = plot.x_at(to.x).max(x0).min(x1);
            // (within a few pixels of the mean the pivot settles on it: a hand cannot
            // find a point exactly, and there the beam is level exactly)
            if let Some(m) = self.target().mean().get() {
                if (plot.px_x(m) - to.x).abs() < 5.0 {
                    self.pivot = m;
                }
            }
            self.look.set(PIVOT_AT, self.pivot as f32);
            self.tour = None;
        }
        f.text_px(o + Vec2::new(0.0, 30.0), "pivot", small, PIVOT.a(0.8), Align2::CENTER_TOP);

        // the beam is the axis: its round values tip with it (but for those the pivot covers)
        let ticks = Ticks::of(x0, x1, max.x - min.x);
        for v in ticks.values() {
            let p = plot.at(v, 0.0);
            plot::line_of(f, AXES, turn.on(p + Vec2::new(0.0, 2.0)), turn.on(p + Vec2::new(0.0, 6.0)), 1.0, pal::DIM.a(0.8));
            if (p.x - o.x).abs() > 16.0 {
                f.text_px(turn.on(p + Vec2::new(0.0, 7.0)), ticks.name(v), small, pal::DIM, Align2::CENTER_TOP);
            }
        }
        f.text_px(at(x1, 0.0) + Vec2::new(7.0, 0.0), "$x$", text, pal::DIM, Align2::LEFT_CENTER);

        // each side's turning moment about the pivot, over the chart's right-hand corner
        let moment = |v: Value| match v {
            Value::Is(m) => num(m, 2),
            _ => r"$\infty$".to_string(),
        };
        let verdict = match mean {
            Some(m) if (m - pivot).abs() <= 1e-6 * (x1 - x0) => ": level",
            Some(m) if m > pivot => ": it tips right",
            Some(_) => ": it tips left",
            None => "",
        };
        let words = format!("turning moments: left {}, right {}{verdict}", moment(blend.below(pivot)), moment(blend.beyond(pivot)));
        f.text_px(Vec2::new(max.x, min.y - 6.0), words, if lay.narrow { small } else { text }, pal::TEXT, Align2::RIGHT_BOTTOM);
        if no_mean {
            f.caption_px(o + Vec2::new(16.0, 21.0), "no balance point: each side turns the beam infinitely hard", small, TAILS, Align2::LEFT_CENTER);
        }

        // the median, the two halves, the mean and the bar of a standard deviation, the peak
        let mark = truth_mark();
        let foot = |f: &mut Frame, row: usize, x: f64, text: &str| row_label(f, (min, max), row, plot.at(x, 0.0).x, text, small, mark);
        // (each half named beside the median line, above the bar of a standard deviation)
        let up = (0.5 * (plot.px_y(0.0) - plot.px_y(blend.pdf(median).min(top)))).max(46.0);
        for (side, align) in [(-1.0, Align2::RIGHT_CENTER), (1.0, Align2::LEFT_CENTER)] {
            f.text_px(turn.on(plot.at(median, 0.0) + Vec2::new(9.0 * side, -up)), "half", small, mark, align);
        }
        let label = foot(f, 1, median, &format!("median {}", num(median, 2)));
        let tip = at(median, blend.pdf(median).min(top));
        plot::dashed_of(f, MARKS, at(median, 0.0), tip, 1.4, mark);
        plot::dashed_of(f, MARKS, tip, Vec2::new(tip.x, label.y.min(tip.y)), 1.0, mark.a(0.5));
        if let Some(m) = mean {
            let label = foot(f, 2, m, &format!("mean {}: the balance point", num(m, 2)));
            let base = at(m, 0.0);
            let up = turn.on(plot.at(m, 0.0) + Vec2::new(0.0, -11.0));
            let side = (turn.on(plot.at(m, 0.0) + Vec2::new(7.0, 0.0)) - base).normalize_or_zero() * 7.0;
            tri(f, 2, [base, up - side, up + side], mark);
            plot::dashed_of(f, MARKS, Vec2::new(label.x, label.y), up, 1.0, mark.a(0.5));
            if let Some(s) = sd {
                let (a, b) = (turn.on(plot.at(m - s, 0.0) + Vec2::new(0.0, -17.0)), turn.on(plot.at(m + s, 0.0) + Vec2::new(0.0, -17.0)));
                plot::line_of(f, MARKS, a, b, 1.6, PIVOT.a(0.75));
                for e in [a, b] {
                    let d = (up - base).normalize_or_zero() * 5.0;
                    plot::line_of(f, MARKS, e - d, e + d, 1.6, PIVOT.a(0.75));
                }
                for e in [a, b] {
                    let mid = (e + turn.on(plot.at(m, 0.0) + Vec2::new(0.0, -17.0))) * 0.5;
                    f.text_px(mid + Vec2::new(0.0, -3.0), r"$\sigma$", small, PIVOT, Align2::CENTER_BOTTOM);
                }
            }
        }
        // the peaks, gliding from the last shape's to the new one's while one turns into
        // the other (the mode of a shape changing by degrees can jump from one hump to
        // another), and named by the new one's
        let names = peak_names(self.target());
        for (j, (x, words)) in names.iter().enumerate() {
            let x = if j < self.peaks.shown.len() { self.peaks.get(j) as f64 } else { *x };
            let p = at(x, blend.pdf(x).min(top));
            plot::disc(f, p, 8.0, mark);
            let label = foot(f, 0, x, words);
            plot::dashed_of(f, MARKS, label, p, 1.0, mark.a(0.5));
        }
        if let Peaks::Flat(a, b) = self.target().peaks() {
            let h = blend.pdf(0.5 * (a + b)).min(top);
            plot::line_of(f, MARKS, at(a, h), at(b, h), 3.0, mark);
            foot(f, 0, 0.5 * (a + b), "mode: every value on the flat top");
        }

        // the tails beyond k standard deviations, and Chebyshev's bound on them
        if self.tails {
            match (mean, sd) {
                (Some(m), Some(s)) => {
                    for (j, x) in [m - self.k * s, m + self.k * s].into_iter().enumerate() {
                        let foot = at(x.max(x0).min(x1), 0.0);
                        plot::dashed_of(f, MARKS, foot, turn.on(plot.at(x.max(x0).min(x1), 0.0) + Vec2::new(0.0, -0.75 * (max.y - min.y))), 1.2, TAILS);
                        if x >= x0 && x <= x1 {
                            if let Some(to) = grip(f, ID_K + j as u32, foot, TAILS, 10.0) {
                                let v = plot.x_at(turn.off(to).x);
                                self.k = ((v - m).abs() / s).max(1.0).min(4.0);
                            }
                        }
                    }
                    let share = blend.parts.iter().map(|(sh, w)| w * (1.0 - sh.cdf(m + self.k * s) + sh.cdf(m - self.k * s))).sum::<f64>();
                    let lines = [format!(r"beyond $k = {}$ standard deviations: {}", num(self.k, 1), percent(share)), format!(r"Chebyshev: at most $1/k^2$, which is {}", percent(1.0 / (self.k * self.k)))];
                    for (j, line) in lines.iter().enumerate() {
                        f.caption_px(Vec2::new(max.x - 4.0, min.y + 3.0 + (3 + j) as f32 * (small + 6.0)), line.as_str(), small, TAILS, Align2::RIGHT_TOP);
                    }
                }
                _ => {
                    f.caption_px(Vec2::new(max.x - 4.0, min.y + 3.0 + 3.0 * (small + 6.0)), "no finite standard deviation: no tails to measure by it", small, TAILS, Align2::RIGHT_TOP);
                }
            }
        }
    }

    fn draw_strip(&mut self, f: &mut Frame, lay: &Layout) {
        let (x0, x1) = (self.look.get(X0) as f64, self.look.get(X1) as f64);
        let (min, max) = lay.strip;
        // as many units up as across for each pixel, so that a square is square
        let per = (max.x - min.x).max(1.0) as f64 / (x1 - x0);
        let plot = Plot::new(lay.strip, [x0, x1], [0.0, (max.y - min.y) as f64 / per]);
        let small = lay.small;
        let n = self.count.min(self.values.len());
        let title = match (lay.narrow, self.squares) {
            (true, true) => format!("a sample of {n}, and squares"),
            (true, false) => format!("a sample of {n}, and distances"),
            (false, true) => format!("a sample of {n} from it, and the square on each point's distance from the pivot"),
            (false, false) => format!("a sample of {n} from it, and each point's distance from the pivot"),
        };
        f.text_px(Vec2::new(min.x - 30.0, min.y - lay.gaps.title_s + 3.0), title, lay.title, pal::TEXT, Align2::LEFT_TOP);
        let pivot = self.look.get(PIVOT_AT) as f64;
        plot.vline(f, MARKS, pivot, 1.2, PIVOT.a(0.55), true);
        let shown: Vec<f64> = (0..n).map(|i| self.points.get(i) as f64).collect();
        let base = max.y;
        let clip_x = |x: f32| x.max(min.x).min(max.x);
        if self.squares {
            // (fainter when there are many, so that the pile of them can still be read)
            let ink = 0.45 * (40.0 / n.max(1) as f32).min(1.0).powf(0.7);
            let mut cut_any = false;
            for &x in &shown {
                let d = (x - pivot).abs();
                let (a, b) = (clip_x(plot.px_x(x.min(pivot))), clip_x(plot.px_x(x.max(pivot))));
                let up = base - d as f32 * per as f32;
                let cut = up < min.y;
                cut_any |= cut;
                let up = up.max(min.y);
                // (three sides, each a segment of its own, the top drawn clear where it is
                // cut off: the same points in every frame, however the squares grow)
                plot::line_of(f, OWN, Vec2::new(a, base), Vec2::new(a, up), 1.0, SQUARES.a(ink));
                plot::line_of(f, OWN, Vec2::new(a, up), Vec2::new(b, up), 1.0, if cut { Color::CLEAR } else { SQUARES.a(ink) });
                plot::line_of(f, OWN, Vec2::new(b, up), Vec2::new(b, base), 1.0, SQUARES.a(ink));
            }
            if cut_any {
                let words = if lay.narrow { "tall squares cut off" } else { "squares too tall for the strip are cut off" };
                f.caption_px(Vec2::new(max.x - 4.0, min.y + 3.0 + 2.0 * (small + 6.0)), words, small, SQUARES.a(0.8), Align2::RIGHT_TOP);
            }
            // the average square, on the pivot
            let area = self.drawn_stat(|x| msd(&shown[..x.len()], pivot));
            let side = area.sqrt();
            let (a, b) = (clip_x(plot.px_x(pivot - 0.5 * side)), clip_x(plot.px_x(pivot + 0.5 * side)));
            let up = (base - side as f32 * per as f32).max(min.y);
            plot::fill(f, (Vec2::new(a, up), Vec2::new(b, base)), SQUARES.a(0.28));
            plot::path_of(f, MARKS, &[Vec2::new(a, base), Vec2::new(a, up), Vec2::new(b, up), Vec2::new(b, base)], 2.4, SQUARES);
            let words = format!("average square {}", num(area, 2));
            let y = (up - 4.0).max(min.y + 2.0 * (small + 6.0) + small + 6.0);
            f.caption_px(Vec2::new(0.5 * (a + b), y), words, small, SQUARES, Align2::CENTER_BOTTOM);
        } else {
            // each distance on a row of its own, the points in order from left to right
            let mut order: Vec<usize> = (0..n).collect();
            order.sort_by(|&i, &j| shown[i].total_cmp(&shown[j]));
            let room = (max.y - min.y - 3.0 * (small + 6.0) - 6.0).max(4.0);
            let step = (room / (n as f32 + 1.0)).min(6.0);
            for (r, &i) in order.iter().enumerate() {
                let y = base - (r as f32 + 1.0) * step;
                plot::line_of(f, OWN, Vec2::new(clip_x(plot.px_x(pivot)), y), Vec2::new(clip_x(plot.px_x(shown[i])), y), 1.4, DISTANCES.a(0.7));
            }
            let mean = self.drawn_stat(|x| mad(&shown[..x.len()], pivot));
            let y = base - (n as f32 + 1.0) * step - 8.0;
            let (a, b) = (clip_x(plot.px_x(pivot - 0.5 * mean)), clip_x(plot.px_x(pivot + 0.5 * mean)));
            plot::line_of(f, MARKS, Vec2::new(a, y), Vec2::new(b, y), 3.0, DISTANCES);
            f.caption_px(Vec2::new(0.5 * (a + b), y - 4.0), format!("average distance {}", num(mean, 2)), small, DISTANCES, Align2::CENTER_BOTTOM);
        }
        // the sample's mean and median
        let rect = (min, max);
        let (sm, sd) = (self.look.get(SAMPLE_MEAN) as f64, self.look.get(SAMPLE_MEDIAN) as f64);
        let s = summary(self.sample(), self.k);
        let in_range = |v: f64| v >= x0 && v <= x1;
        if n > 0 {
            plot.vline(f, MARKS, sm, 1.6, SQUARES, false);
            plot.vline(f, MARKS, sd, 1.6, DISTANCES, true);
            let mean_words = if in_range(sm) { format!("sample mean {}", num(s.mean, 2)) } else { format!("sample mean {}, off the chart", num(s.mean, 2)) };
            row_label(f, rect, 0, plot.px_x(sm).max(min.x).min(max.x), &mean_words, small, SQUARES);
            row_label(f, rect, 1, plot.px_x(sd).max(min.x).min(max.x), &format!("sample median {}", num(s.median, 2)), small, DISTANCES);
        }
        if n == 1 {
            f.caption_px(Vec2::new(max.x - 4.0, base - 4.0), "one point: no spread", small, pal::TEXT, Align2::RIGHT_BOTTOM);
        }
        // the points themselves, to drag; those beyond the chart at its edges
        let (mut left, mut right) = (0, 0);
        for i in 0..n {
            let x = shown[i];
            left += (x < x0) as usize;
            right += (x > x1) as usize;
            let at = Vec2::new(plot.px_x(x.max(x0).min(x1)), base);
            // (the newest point, while they come one at a time, fades in where it lands)
            let color = if i + 1 == n { DATA.a(0.15 + 0.85 * self.coming() as f32) } else { DATA };
            if let Some(to) = grip(f, ID_POINT + i as u32, at, color, 8.0) {
                let v = plot.x_at(to.x).max(x0).min(x1);
                if let Some(slot) = self.values.get_mut(i) {
                    *slot = v;
                }
                self.points.set(i, v as f32);
                self.later = None;
                self.tour = None;
                self.refresh();
            }
        }
        if left > 0 {
            f.text_px(Vec2::new(min.x + 2.0, base - 14.0), format!("← {left} further out"), small, DATA, Align2::LEFT_BOTTOM);
        }
        if right > 0 {
            f.text_px(Vec2::new(max.x - 2.0, base - 14.0), format!("{right} further out →"), small, DATA, Align2::RIGHT_BOTTOM);
        }
    }

    fn draw_curve(&self, f: &mut Frame, lay: &Layout) {
        let (x0, x1) = (self.look.get(X0) as f64, self.look.get(X1) as f64);
        let top = self.look.get(CURVE_TOP) as f64;
        let plot = Plot::new(lay.curve, [x0, x1], [0.0, top]);
        let (min, max) = plot.rect;
        let small = lay.small;
        let title = match (lay.narrow, self.squares) {
            (true, true) => "the average square, against the pivot",
            (true, false) => "the average distance, against the pivot",
            (false, true) => "the average square (the mean squared distance), against where the pivot is",
            (false, false) => "the average distance (the mean absolute distance), against where the pivot is",
        };
        f.text_px(Vec2::new(min.x - 30.0, min.y - lay.gaps.title_c + 3.0), title, lay.title, pal::TEXT, Align2::LEFT_TOP);
        plot.y_grid(f);
        plot.y_axis(f, "", small);
        plot.x_axis(f, "where the pivot is", small);
        let n = self.count.min(self.values.len());
        let shown: Vec<f64> = (0..n).map(|i| self.points.get(i) as f64).collect();
        let pivot = self.look.get(PIVOT_AT) as f64;
        let (color, here) = if self.squares { (SQUARES, msd(&shown, pivot)) } else { (DISTANCES, mad(&shown, pivot)) };
        // the distribution's own curve, thin (of the shape as drawn, while one turns into another)
        let blend = self.blend();
        let truth_drawn = if self.squares {
            if let (Some(m), Some(sd)) = (blend.mean().get(), blend.sd()) {
                convex_curve(&plot, f, 1.2, TRUTH.a(0.8), |p| sd * sd + (m - p) * (m - p), m);
            }
            blend.sd().is_some()
        } else if blend.beyond(0.0).get().is_some() {
            convex_curve(&plot, f, 1.2, TRUTH.a(0.8), |p| blend.beyond(p).get().unwrap_or(f64::NAN) + blend.below(p).get().unwrap_or(f64::NAN), blend.median());
            true
        } else {
            false
        };
        // the sample's
        if self.squares {
            convex_curve(&plot, f, 2.2, color, |p| msd(&shown, p), stats::mean(&shown));
        } else {
            convex_curve(&plot, f, 2.2, color, |p| mad(&shown, p), stats::median(&shown));
        }
        // which curve is which
        let names = [("the sample's", color), ("the distribution's", TRUTH)];
        for (j, (name, c)) in names.iter().take(if truth_drawn { 2 } else { 1 }).enumerate() {
            f.text_px(Vec2::new(max.x - 4.0, min.y + 3.0 + (2 + j) as f32 * (small + 6.0)), *name, small, *c, Align2::RIGHT_TOP);
        }
        // the bottom, named at the top of the chart
        let s = summary(self.sample(), self.k);
        let (bottom, words) = if self.squares { (s.mean, "least at the sample mean") } else { (s.median, "least at the sample median") };
        let drawn_bottom = self.look.get(if self.squares { SAMPLE_MEAN } else { SAMPLE_MEDIAN }) as f64;
        let least = if self.squares { msd(&shown, drawn_bottom) } else { mad(&shown, drawn_bottom) };
        let b = plot.at(drawn_bottom.max(x0).min(x1), least.min(top));
        let label = row_label(f, (min, max), 0, b.x, &format!("{words}, {}", num(bottom, 2)), small, color);
        plot::dashed_of(f, MARKS, label, b, 1.2, color.a(0.8));
        plot::disc(f, b, 6.0, color);
        // the pivot, and the curve's height there
        plot.vline(f, MARKS, pivot, 1.2, PIVOT.a(0.55), true);
        let p = plot.at(pivot, here.min(top));
        plot::disc(f, p, 9.0, PIVOT);
        let words = if here > top { format!("{} at the pivot ↑", num(here, 2)) } else { format!("{} at the pivot", num(here, 2)) };
        // (on the side away from the bottom, and below the curve where it is high)
        let right = p.x >= b.x;
        let offset = Vec2::new(if right { 9.0 } else { -9.0 }, if p.y < min.y + 3.0 * (small + 6.0) { 14.0 } else { -5.0 });
        let align = match (right, offset.y > 0.0) {
            (true, false) => Align2::LEFT_BOTTOM,
            (false, false) => Align2::RIGHT_BOTTOM,
            (true, true) => Align2::LEFT_TOP,
            (false, true) => Align2::RIGHT_TOP,
        };
        f.caption_px(p + offset, words, small, PIVOT, align);
    }

    fn draw_history(&self, f: &mut Frame, lay: &Layout) {
        let (x0, x1) = (self.look.get(X0) as f64, self.look.get(X1) as f64);
        let n = self.values.len().max(1);
        let plot = Plot::new(lay.curve, [x0, x1], [0.0, n as f64]);
        let min = plot.rect.0;
        let small = lay.small;
        let title = if lay.narrow { "mean and median after each draw" } else { "the sample's mean and median after each draw (the draws counted upwards)" };
        f.text_px(Vec2::new(min.x - 30.0, min.y - lay.gaps.title_c + 3.0), title, lay.title, pal::TEXT, Align2::LEFT_TOP);
        plot.y_grid(f);
        plot.y_axis(f, "draws", small);
        plot.x_axis(f, "", small);
        let count = self.count.min(self.means.len());
        // the newest step drawn part of the way, so that the paths grow evenly
        let share = if self.growing { self.share.max(0.0).min(1.0) as f64 } else { 1.0 };
        for (values, color, kind, name, side) in [(&self.means, SQUARES, OWN + 2, "mean", 1.0), (&self.medians, DISTANCES, OWN + 3, "median", -1.0)] {
            let mut pts: Vec<(f64, f64)> = (0..count.saturating_sub(1)).map(|m| (values[m], (m + 1) as f64)).collect();
            if count >= 1 {
                let last = values[count - 1];
                let before = if count >= 2 { values[count - 2] } else { last };
                pts.push((before + (last - before) * share, count as f64 - 1.0 + share));
            }
            // named where it has got to (at the chart's edge if it is off it)
            if let Some(&(v, m)) = pts.last() {
                let at = plot.at(v.max(x0).min(x1), m);
                let align = if side > 0.0 { Align2::LEFT_BOTTOM } else { Align2::RIGHT_BOTTOM };
                f.text_px(at + Vec2::new(6.0 * side, -2.0), name, small, color, align);
            }
            plot.path(f, kind, pts, 1.8, color);
        }
    }

    fn draw_table(&self, f: &mut Frame, lay: &Layout) {
        let (min, max) = lay.table;
        if !(max.x - min.x > 60.0 && max.y - min.y > 40.0) {
            return;
        }
        hud_panel(f, min - Vec2::new(4.0, 4.0), max + Vec2::new(4.0, 4.0));
        let size = if lay.narrow { lay.small } else { lay.text };
        let line = size + if lay.narrow { 4.0 } else { 8.0 };
        let w = max.x - min.x;
        let (c1, c2) = (min.x + w * if lay.narrow { 0.72 } else { 0.7 }, max.x - 6.0);
        let x = self.sample();
        let s = summary(x, self.k);
        let target = self.target();
        let [variance, skew, kurtosis] = target.shape();
        let mode = match target.peaks() {
            Peaks::One(a) => num(a, 2),
            Peaks::Two(a, b) => format!("{} and {}", num(a, 2), num(b, 2)),
            Peaks::Main(a, b) => format!("{} (and {})", num(a, 2), num(b, 2)),
            Peaks::Flat(..) => "the flat top".to_string(),
        };
        let spread = match variance {
            Value::Is(v) => num(v.sqrt(), 2),
            other => other.show(2),
        };
        let opt = |v: Option<f64>, d: usize| v.map_or("—".to_string(), |v| num(v, d));
        let share = |v: Option<f64>| v.map_or("—".to_string(), percent);
        let mut rows: Vec<(String, String, String)> = vec![
            ("mean".into(), target.mean().show(2), num(s.mean, 2)),
            ("median".into(), num(target.median(), 2), num(s.median, 2)),
            ("mode".into(), mode, "—".into()),
            ("standard deviation".into(), spread, num(s.sd, 2)),
            ("skewness".into(), skew.show(2), opt(s.skewness, 2)),
            ("excess kurtosis".into(), kurtosis.show(2), opt(s.kurtosis, 2)),
        ];
        for (j, k) in [1.0, 2.0, 3.0].iter().enumerate() {
            rows.push((format!(r"within ${}\sigma$ of the mean", *k as u32), share(target.within(*k)), share(s.within[j])));
        }
        if self.tails {
            let beyond = target.within(self.k).map(|v| 1.0 - v);
            rows.push((format!(r"beyond $k\sigma$, $k = {}$", num(self.k, 1)), share(beyond), share(s.beyond)));
            rows.push((r"Chebyshev: at most $1/k^2$".into(), percent(1.0 / (self.k * self.k)), String::new()));
        }
        let mut y = min.y + 4.0;
        let heads = ("distribution", format!("sample of {}", s.n));
        f.text_px(Vec2::new(c1, y), heads.0, size, TRUTH, Align2::RIGHT_TOP);
        f.text_px(Vec2::new(c2, y), heads.1, size, DATA, Align2::RIGHT_TOP);
        y += line + 2.0;
        for (name, a, b) in rows {
            if y + size > max.y + 2.0 {
                break;
            }
            f.text_px(Vec2::new(min.x + 6.0, y), name, size, pal::DIM, Align2::LEFT_TOP);
            f.text_px(Vec2::new(c1, y), a, size, pal::TEXT, Align2::RIGHT_TOP);
            f.text_px(Vec2::new(c2, y), b, size, pal::TEXT, Align2::RIGHT_TOP);
            y += line;
        }
        if lay.narrow {
            let note = if s.n == 1 { "one point: no spread" } else { r"the sample's $\sigma$ divides by $n$" };
            f.text_px(Vec2::new(c2, y), note, size, pal::DIM, Align2::RIGHT_TOP);
        } else {
            let note = if s.n == 1 {
                "One point: no spread, and so no skewness or kurtosis."
            } else {
                "The sample's column is the sample's own: its standard deviation divides by $n$ (it is the side of the average square with the pivot at the sample mean), and its shares are counted about its own mean."
            };
            let lines = lines_px(f, note, lay.small, max.x - min.x - 12.0);
            for (j, text) in lines.iter().enumerate() {
                f.text_px(Vec2::new(min.x + 6.0, y + 6.0 + j as f32 * (lay.small + 4.0)), text.as_str(), lay.small, pal::DIM, Align2::LEFT_TOP);
            }
            let y = y + 6.0 + lines.len() as f32 * (lay.small + 4.0) + 22.0;
            if y + 6.0 * (lay.small + 6.0) < lay.view.y - 8.0 {
                let key: [(Color, &str); 6] = [
                    (TRUTH, "the distribution"),
                    (PIVOT, r"the pivot; $\pm 1$ standard deviation"),
                    (DATA, "a sample from it"),
                    (SQUARES, "squares, and the sample mean"),
                    (DISTANCES, "distances, and the sample median"),
                    (TAILS, r"beyond $k$ standard deviations"),
                ];
                let shown = if self.tails { &key[..] } else { &key[..5] };
                hud_panel(f, Vec2::new(min.x - 4.0, y - 8.0), Vec2::new(max.x + 4.0, y + shown.len() as f32 * (lay.small + 6.0) + 4.0));
                hud_key(f, Vec2::new(min.x + 6.0, y), shown, lay.small);
            }
        }
    }

    /// New round values on an axis: make sure the labels are not as many as in the last
    /// frame, so that nothing takes the new ones for the old ones moved.
    fn new_ticks(&mut self, f: &mut Frame, lay: &Layout) {
        let (x0, x1) = (self.look.get(X0) as f64, self.look.get(X1) as f64);
        let width = lay.density.1.x - lay.density.0.x;
        let mut ticks: Vec<f64> = Ticks::of(x0, x1, width).values().collect();
        ticks.push(f64::NAN);
        ticks.extend(Ticks::of(0.0, self.look.get(CURVE_TOP) as f64, (lay.curve.1.y - lay.curve.0.y) * 1.6).values());
        let changed = ticks.len() != self.ticks.len() || ticks.iter().zip(&self.ticks).any(|(a, b)| !(a == b || (a.is_nan() && b.is_nan())));
        let count = |f: &mut Frame| (plot::apart(f, AXES, false).pts.len(), f.dl.labels.len());
        let (lines, labels) = count(f);
        if changed && lines == self.labels_last.0 {
            let far = f.hud(Vec2::ZERO);
            plot::apart(f, AXES, false).segment(far, far, 1.0, Color::CLEAR);
        }
        if changed && labels == self.labels_last.1 {
            f.text_px(Vec2::ZERO, "", 1.0, Color::CLEAR, Align2::LEFT_TOP);
        }
        self.ticks = ticks;
        self.labels_last = count(f);
    }
}

impl Scene for Moments {
    fn view(&self) -> View {
        View::Flat { center: Vec2::ZERO, half_height: 1.0 }
    }

    fn settings(&self, s: &mut ViewSettings) {
        s.grid = false;
        s.bloom = s.bloom.min(0.2);
    }

    fn animated(&self) -> bool {
        self.moving || self.growing || self.tour.is_some() || self.later.is_some() || !self.before.is_empty()
    }

    fn pointer_hint(&self) -> Option<&'static str> {
        Some("drag: the pivot (white), the sample's points (cyan) and the other dots    click: the buttons")
    }

    fn ui(&mut self, ui: &mut egui::Ui, _: &Gfx) {
        ui::note(
            ui,
            "Three numbers say where a distribution is centred, and each answers a different question. The mean is the balance point: rest the shape on a beam, and the beam is level on a pivot there and nowhere else (if it has a mean at all). The median has half the area on each side. The mode is where the shape peaks. The standard deviation says how widely it spreads, in the data's own units; skewness and kurtosis say more about its shape. Under it, a sample from the same distribution: the average of the squares on its points' distances from the pivot is least at the sample's mean, the average of the distances least at its median.",
        );

        ui::section(ui, "Things to try");
        for (k, trial) in TRIALS.iter().enumerate() {
            if ui::experiment(ui, trial.0, trial.1) {
                self.set_up(k);
            }
        }

        ui::section(ui, "The shape");
        let mut kind = KINDS.iter().position(|k| *k == self.kind).unwrap_or(0);
        let names: Vec<&str> = KINDS.iter().map(|k| k.name()).collect();
        if ui::choice(ui, &names, &mut kind) && KINDS[kind] != self.kind {
            self.tour = None;
            self.choose(KINDS[kind]);
        }
        let mut changed = false;
        match self.kind {
            Kind::Bell => ui::note(ui, "The normal distribution with mean 0 and standard deviation 1."),
            Kind::LongTail => {
                let mut v = self.params.sigma as f32;
                changed |= ui::slider(ui, r"spread of its logarithm, $\sigma$", &mut v, 0.1..=1.0);
                self.params.sigma = v as f64;
                ui::note(ui, r"Log-normal: $e^Z$ for a normal number $Z$ with mean 0 and standard deviation $\sigma$. The larger $\sigma$, the longer the tail.");
            }
            Kind::TwoHumps => {
                let mut v = self.params.apart as f32;
                changed |= ui::slider(ui, r"how far apart, $a$", &mut v, 0.0..=0.95);
                self.params.apart = v as f64;
                ui::note(ui, r"Bells at $-a$ and $a$, each with standard deviation $\sqrt{1 - a^2}$: the mean stays 0 and the standard deviation 1. Two peaks once $a$ passes 0.71.");
            }
            Kind::Flat => ui::note(ui, r"Every value from $-\sqrt{3}$ to $\sqrt{3}$ as likely as every other: mean 0, standard deviation 1."),
            Kind::Peaked => ui::note(ui, r"The Laplace distribution, with mean 0 and standard deviation 1: a sharp peak, and tails that fall off more slowly than a bell's."),
            Kind::HeavyTails => {
                let mut v = self.params.df as f32;
                changed |= ui::slider_log(ui, r"degrees of freedom, $\nu$", &mut v, 1.0..=30.0);
                self.params.df = v as f64;
                ui::note(ui, r"Student's $t$. The fewer the degrees of freedom $\nu$, the heavier the tails, and the fewer the moments: the skewness is defined only for $\nu > 3$, the excess kurtosis is infinite for $2 < \nu \le 4$, the variance for $1 < \nu \le 2$, and at $\nu = 1$ (the Cauchy distribution) there is no mean, and so nothing measured from it.");
            }
            Kind::Mixture => {
                let (mut at, mut w) = (self.params.at as f32, self.params.weight as f32);
                changed |= ui::slider(ui, "where the second bell is", &mut at, -2.5..=4.5);
                changed |= ui::slider(ui, "its share of the area", &mut w, 0.0..=1.0);
                (self.params.at, self.params.weight) = (at as f64, w as f64);
                ui::note(ui, "Two bells of standard deviation 0.6, one at 0. Or drag the second bell's top (the green dot) on the picture: across moves it, up and down changes its share.");
            }
        }
        if changed {
            self.tour = None;
            self.reshape();
        }

        ui::section(ui, "The pivot");
        let target = self.target();
        let places = [target.mean().get(), Some(target.median()), match target.peaks() {
            Peaks::One(a) | Peaks::Two(a, _) | Peaks::Main(a, _) => Some(a),
            Peaks::Flat(..) => None,
        }];
        if let Some(i) = ui::buttons(ui, &["to the mean", "to the median", "to the peak"]) {
            if let Some(p) = places[i] {
                let [lo, hi] = target.range();
                self.pivot = p.max(lo).min(hi);
                self.tour = None;
            }
        }
        if places[0].is_none() {
            ui::note(ui, "This distribution has no mean: there is no balance point to move the pivot to.");
        }
        if places[2].is_none() {
            ui::note(ui, "A flat top has no one peak.");
        }
        ui::note(ui, "Or drag it along the beam. The beam tips towards the side with the larger turning moment, the further the larger the difference.");

        ui::section(ui, "The sample");
        let mut n = self.n;
        if ui::slider_int_log(ui, "points, $n$", &mut n, 1..=MAX_N) {
            self.set_n(n);
        }
        ui.horizontal(|ui| {
            if ui.button("draw again").clicked() {
                self.draw_again(false);
            }
            if ui.button("draw one at a time").clicked() {
                self.draw_again(true);
            }
        });
        let mut mode = (!self.squares) as usize;
        if ui::choice(ui, &["squares", "distances"], &mut mode) {
            self.squares = mode == 0;
            self.history &= self.growing;
        }
        ui::note(ui, "The average of the squares on the points' distances from the pivot is least with the pivot at the sample mean; the average of the distances themselves is least at the sample median (for an even number of points, anywhere between the two middle ones). Drag a point far out to see what each answers to. Drawn one at a time, the chart under the sample shows its mean and median after each draw.");

        ui::section(ui, "Tails");
        ui::checkbox(ui, &mut self.tails, r"shade what lies beyond $k$ standard deviations");
        let mut k = self.k as f32;
        if ui::slider(ui, "$k$", &mut k, 1.0..=4.0) {
            self.k = k as f64;
        }
        ui::note(ui, r"Chebyshev's inequality: whatever the shape, if its variance is finite, at most $1/k^2$ of it lies $k$ or more standard deviations from its mean.");

        ui::section(ui, "Reading the picture");
        ui::key(
            ui,
            &[
                (TRUTH, "green: the distribution: its shape (the density), resting on the beam; its mean (the triangle), median (dashed, the halves shaded apart) and mode (the dot)"),
                (PIVOT, r"white: the pivot, its line through the charts, and the bar of one standard deviation $\sigma$ each side of the mean"),
                (DATA, "cyan dots: a sample from the distribution (drag them)"),
                (SQUARES, "yellow: the squares on the distances from the pivot, their average, and the sample mean"),
                (DISTANCES, "orange: the distances from the pivot, their average, and the sample median"),
                (TAILS, r"pink: what lies beyond $k$ standard deviations; a beam with no balance point"),
            ],
        );

        ui::section(ui, "Words");
        ui::term(ui, "Mean (expected value):", r"the balance point. For a distribution with density $f$, $\mu = \int x f(x)\,dx$, where that integral has a value (the Cauchy's has none); for a sample, the sum divided by $n$.");
        ui::term(ui, "Median:", "the value with half the distribution (or half the sample) on each side.");
        ui::term(ui, "Mode:", "where the density is highest. A shape can have two peaks, or a flat top.");
        ui::term(ui, "Turning moment:", r"how hard one side turns the beam about the pivot $p$: its area times its distance, added up, $\int_p^\infty (x - p) f(x)\,dx$ on the right. The right's less the left's is $\mu - p$.");
        ui::term(ui, "Variance:", r"the mean squared distance from the mean, $\sigma^2 = \int (x - \mu)^2 f(x)\,dx$. For a sample here, the average square at the sample mean (dividing by $n$; why it is often divided by $n - 1$ is in Estimators).");
        ui::term(ui, "Standard deviation:", r"$\sigma$, the square root of the variance: a spread in the data's own units.");
        ui::term(ui, "Moment:", r"the $k$th moment about the mean is the mean of $(x - \mu)^k$: the second is the variance; the third and fourth, divided by $\sigma^3$ and $\sigma^4$, are the skewness and the kurtosis.");
        ui::term(ui, "Skewness:", r"the third moment divided by $\sigma^3$: 0 for a symmetric shape, and as a rule positive for a long right tail (the log-normal's) and negative for a long left one. (Only as a rule: it weighs cubed distances, and shapes can be made that break it, or that have skewness 0 without being symmetric.)");
        ui::term(ui, "Excess kurtosis:", "the fourth moment divided by the fourth power of the standard deviation, less 3, its value for a bell: positive when more of the variance comes from far out (heavy tails), negative for light tails (a flat block, two humps).");
        ui::term(ui, "Heavy tail:", "a tail that thins out slowly. Heavy enough, and a moment is infinite, or not defined at all.");
        ui::term(ui, "Chebyshev's inequality:", r"at most $1/k^2$ of any distribution with a finite variance lies $k$ or more standard deviations from its mean.");

        ui::section(ui, "Numbers");
        let blend = Blend { parts: vec![(target, 1.0)] };
        let shown = |v: Value| match v {
            Value::Is(v) => num(v, 3),
            _ => "infinite".to_string(),
        };
        ui::readout(ui, "the pivot", num(self.pivot, 2));
        ui::readout(ui, "turning moment of the left side", shown(blend.below(self.pivot)));
        ui::readout(ui, "of the right side", shown(blend.beyond(self.pivot)));
        let x = self.sample();
        ui::readout(ui, "average square at the pivot", num(msd(x, self.pivot), 3));
        ui::readout(ui, "average distance at the pivot", num(mad(x, self.pivot), 3));
        ui::see_also(
            ui,
            &[
                ("distributions", "the shapes themselves"),
                ("lln", "the sample mean settling, or not"),
                ("describe", "the same summaries of data, and robust ones"),
                (r"estimators", r"dividing by $n$ or $n - 1$"),
            ],
        );
    }

    fn frame(&mut self, f: &mut Frame) {
        let lay = Layout::of(f.view_size(), f.top_inset);
        // (a canvas squeezed to a sliver is one the controls have opened over, on a phone)
        let tiny = lay.view.x < 120.0 || lay.view.y < 120.0;
        if !tiny {
            self.draw_buttons(f, &lay);
        }
        self.advance(f.dt, f.real_dt);
        self.glide(f.real_dt);
        if tiny {
            return;
        }
        let (min, max) = lay.column;
        hud_panel(f, min, max);
        let blend = self.blend();
        self.draw_density(f, &lay, &blend);
        if !lay.compact {
            self.draw_strip(f, &lay);
            if self.history {
                self.draw_history(f, &lay);
            } else {
                self.draw_curve(f, &lay);
            }
            self.draw_table(f, &lay);
        }
        self.new_ticks(f, &lay);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::math::causal::Noise;

    /// The integral of g(x) f(x) over the shape, by the midpoint rule on a fine grid wide
    /// enough for light tails, broken where the flat block's density jumps (and for the
    /// log-normal, in the logarithm, where its tail is light).
    fn integral(s: Shape, g: impl Fn(f64) -> f64) -> f64 {
        let midpoints = |lo: f64, hi: f64, steps: usize, h: &dyn Fn(f64) -> f64| {
            let w = (hi - lo) / steps as f64;
            (0..steps).map(|i| h(lo + (i as f64 + 0.5) * w) * w).sum::<f64>()
        };
        if let Shape::LongTail { sigma } = s {
            return midpoints(-14.0 * sigma, 14.0 * sigma, 200_000, &|y: f64| g(y.exp()) * normal_pdf(y / sigma) / sigma);
        }
        let cuts = [-40.0, -SQRT3, SQRT3, 60.0];
        cuts.windows(2).map(|c| midpoints(c[0], c[1], 150_000, &|x: f64| g(x) * s.pdf(x))).sum()
    }

    /// Let the picture come to rest after a change (and what goes on by itself end).
    fn settle(s: &mut Moments) {
        for _ in 0..60 * 30 {
            s.advance(1.0 / 60.0, 1.0 / 60.0);
            s.glide(1.0 / 60.0);
            if !s.animated() {
                return;
            }
        }
        panic!("still moving after thirty seconds");
    }

    fn every() -> Vec<Shape> {
        vec![
            Shape::Bell,
            Shape::LongTail { sigma: 0.1 },
            Shape::LongTail { sigma: 0.6 },
            Shape::LongTail { sigma: 1.0 },
            Shape::TwoHumps { apart: 0.0 },
            Shape::TwoHumps { apart: 0.5 },
            Shape::TwoHumps { apart: 0.9 },
            Shape::TwoHumps { apart: 0.95 },
            Shape::Flat,
            Shape::Peaked,
            Shape::HeavyTails { df: 1.0 },
            Shape::HeavyTails { df: 1.5 },
            Shape::HeavyTails { df: 3.0 },
            Shape::HeavyTails { df: 4.5 },
            Shape::HeavyTails { df: 30.0 },
            Shape::Mixture { at: 3.0, weight: 0.25 },
            Shape::Mixture { at: -2.0, weight: 0.6 },
            Shape::Mixture { at: 0.0, weight: 0.5 },
            Shape::Mixture { at: 4.0, weight: 0.0 },
            Shape::Mixture { at: 4.0, weight: 1.0 },
        ]
    }

    /// The shapes whose moments a grid can integrate (tails thinner than any power).
    fn light(s: &Shape) -> bool {
        !matches!(s, Shape::HeavyTails { df } if *df < 10.0)
    }

    /// Each shape's density, cumulative probability, quantiles, mean, median, peaks and
    /// moments: against `math::stats` where it has the shape, and against integrals of
    /// the density for every shape with light tails.
    #[test]
    fn each_shape_has_the_summaries_it_says() {
        let pairs = [
            (Shape::Bell, Dist::STANDARD_NORMAL),
            (Shape::Flat, Dist::Uniform { lo: -SQRT3, hi: SQRT3 }),
            (Shape::LongTail { sigma: 0.6 }, Dist::LogNormal { mu: 0.0, sigma: 0.6 }),
            (Shape::LongTail { sigma: 1.0 }, Dist::LogNormal { mu: 0.0, sigma: 1.0 }),
            (Shape::HeavyTails { df: 1.0 }, Dist::Cauchy { centre: 0.0, scale: 1.0 }),
            (Shape::HeavyTails { df: 1.5 }, Dist::StudentT { df: 1.5 }),
            (Shape::HeavyTails { df: 3.0 }, Dist::StudentT { df: 3.0 }),
            (Shape::HeavyTails { df: 30.0 }, Dist::StudentT { df: 30.0 }),
        ];
        for (s, d) in pairs {
            assert_eq!(s.mean().get().is_some(), d.mean().is_some(), "{s:?}");
            if let (Some(a), Some(b)) = (s.mean().get(), d.mean()) {
                assert!((a - b).abs() < 1e-12, "{s:?}: mean {a} against {b}");
            }
            match (s.shape()[0], d.variance()) {
                (Value::Is(a), Some(b)) => assert!((a - b).abs() < 1e-12 * (1.0 + b), "{s:?}: variance {a} against {b}"),
                (v, None) => assert!(v != Value::Is(0.0) && v.get().is_none(), "{s:?}: {v:?}"),
                (v, b) => panic!("{s:?}: variance {v:?} against {b:?}"),
            }
            assert!((s.median() - d.quantile(0.5)).abs() < 1e-7, "{s:?}");
            for x in [-3.0, -0.4, 0.0, 0.3, 1.0, 2.2, 7.0] {
                assert!((s.pdf(x) - d.pdf(x)).abs() < 1e-12 && (s.cdf(x) - d.cdf(x)).abs() < 1e-12, "{s:?} at {x}");
            }
        }
        for s in every() {
            for &u in &[0.001, 0.02, 0.25, 0.5, 0.77, 0.98, 0.999] {
                let x = s.quantile(u, None);
                assert!(x.is_finite() && (s.cdf(x) - u).abs() < 1e-9, "{s:?} at {u}: {x}");
            }
            assert!((s.cdf(s.median()) - 0.5).abs() < 1e-9, "{s:?}");
            // the peaks are the highest places (or as high as each other, or a lower local one)
            let [lo, hi] = s.range();
            let grid: Vec<f64> = (0..=4000).map(|i| lo + (hi - lo) * i as f64 / 4000.0).collect();
            let top = grid.iter().map(|x| s.pdf(*x)).fold(0.0, f64::max);
            match s.peaks() {
                Peaks::One(a) => assert!(s.pdf(a) >= top - 1e-12, "{s:?}"),
                Peaks::Two(a, b) => assert!(s.pdf(a) >= top - 1e-12 && (s.pdf(a) - s.pdf(b)).abs() < 1e-12 && a < b, "{s:?}"),
                Peaks::Main(a, b) => {
                    assert!(s.pdf(a) >= top - 1e-12 && s.pdf(b) < s.pdf(a), "{s:?}");
                    assert!(s.pdf(b) >= s.pdf(b - 1e-3) && s.pdf(b) >= s.pdf(b + 1e-3), "{s:?}: no peak at {b}");
                }
                Peaks::Flat(a, b) => assert!(grid.iter().filter(|x| **x > a && **x < b).all(|x| (s.pdf(*x) - top).abs() < 1e-12), "{s:?}"),
            }
            if !light(&s) {
                continue;
            }
            let area = integral(s, |_| 1.0);
            let m = integral(s, |x| x);
            let central = |k: i32| integral(s, |x| (x - m).powi(k));
            let (v, m3, m4) = (central(2), central(3), central(4));
            let [var, skew, kurt] = s.shape().map(|v| v.get().expect("a light tail has every moment"));
            assert!((area - 1.0).abs() < 1e-6, "{s:?}: area {area}");
            assert!((m - s.mean().get().unwrap_or(f64::NAN)).abs() < 1e-6, "{s:?}: mean {m}");
            assert!((v - var).abs() < 1e-5 * (1.0 + var), "{s:?}: variance {v} against {var}");
            assert!((m3 / v.powf(1.5) - skew).abs() < 1e-4 * (1.0 + skew.abs()), "{s:?}: skewness {} against {skew}", m3 / v.powf(1.5));
            assert!((m4 / (v * v) - 3.0 - kurt).abs() < 1e-4 * (1.0 + kurt.abs()), "{s:?}: kurtosis {} against {kurt}", m4 / (v * v) - 3.0);
        }
        // Student's t: the moments that are infinite, and those that are not defined
        let t = |df: f64| Shape::HeavyTails { df }.shape();
        assert_eq!(t(1.0), [Value::Undefined; 3]);
        assert_eq!(t(1.5), [Value::Infinite, Value::Undefined, Value::Undefined]);
        assert_eq!(t(3.0), [Value::Is(3.0), Value::Undefined, Value::Infinite]);
        assert_eq!(t(4.5), [Value::Is(4.5 / 2.5), Value::Is(0.0), Value::Is(12.0)]);
        assert_eq!(Shape::HeavyTails { df: 1.0 }.mean(), Value::Undefined);
        assert_eq!((Value::Undefined.show(2), Value::Infinite.show(2)), ("none".to_string(), r"$\infty$".to_string()));
    }

    /// The four shapes of the third experiment are `math::causal::Noise`'s, with mean 0
    /// and variance 1; every shape drawn on the way from one to the next has them too,
    /// so the beam stays level and the bar of one standard deviation keeps its length;
    /// and the shares and kurtoses in the caption are theirs.
    #[test]
    fn the_four_shapes_share_mean_and_spread() {
        let four = TOUR.map(|k| Shape::of(k, &Params::default()));
        for (s, noise, within, kurtosis) in [(four[0], Noise::Normal, "68.3 %", "0.00"), (four[1], Noise::Uniform, "57.7 %", "−1.20"), (four[2], Noise::TwoHumps, "59.1 %", "−1.31"), (four[3], Noise::Laplace, "75.7 %", "3.00")] {
            for i in -60..=60 {
                let x = i as f64 / 10.0;
                assert!((s.pdf(x) - noise.density(x)).abs() < 1e-12, "{s:?} at {x}");
            }
            let (m, v) = (integral(s, |x| x), integral(s, |x| x * x));
            assert!(m.abs() < 1e-7 && (v - 1.0).abs() < 1e-6, "{s:?}: mean {m}, variance {v}");
            assert!(s.mean() == Value::Is(0.0) && s.sd().is_some_and(|v| (v - 1.0).abs() < 1e-15), "{s:?}");
            assert_eq!(percent(s.within(1.0).unwrap_or(0.0)), within, "{s:?}");
            assert_eq!(s.shape()[2].show(2), kurtosis, "{s:?}");
        }
        for a in 0..4 {
            for b in 0..4 {
                for w in 0..=10 {
                    let w = w as f64 / 10.0;
                    let blend = Blend { parts: vec![(four[a], 1.0 - w), (four[b], w)] };
                    assert!(blend.mean().get().is_some_and(|m| m.abs() < 1e-12) && blend.sd().is_some_and(|s| (s - 1.0).abs() < 1e-12), "{a} {b} {w}");
                }
            }
        }
        // the scene's own tour, at 60 frames a second: each shape in turn, the beam level
        let mut scene = Moments::create();
        scene.set_up(2);
        let mut seen = vec![scene.kind];
        // (from the shape shown before, the bell comes in first, and the beam levels)
        while !scene.before.is_empty() || scene.moving {
            scene.advance(1.0 / 60.0, 1.0 / 60.0);
            scene.glide(1.0 / 60.0);
        }
        let mut frames = 0;
        while scene.tour.is_some() || scene.animated() {
            scene.advance(1.0 / 60.0, 1.0 / 60.0);
            scene.glide(1.0 / 60.0);
            let blend = scene.blend();
            assert!(blend.mean().get().is_some_and(|m| m.abs() < 1e-9) && blend.sd().is_some_and(|s| (s - 1.0).abs() < 1e-9), "frame {frames}: {blend:?}");
            assert!(scene.goals()[TILT].abs() < 1e-6 && scene.look.get(TILT).abs() < 1e-6, "frame {frames}: the beam tips");
            if seen.last() != Some(&scene.kind) {
                seen.push(scene.kind);
            }
            frames += 1;
            assert!(frames < 60 * 20, "the tour has not ended");
        }
        assert_eq!(seen, TOUR.to_vec());
        assert!(frames > 60 * 4 * HOLD as usize, "{frames} frames");
    }

    /// Chebyshev's inequality holds for every shape on offer and every k the slider
    /// allows, and the shares in the caption are each shape's.
    #[test]
    fn chebyshev_holds_for_every_shape_and_k() {
        let mut shapes = vec![Shape::Bell, Shape::Flat, Shape::Peaked];
        shapes.extend((2..=20).map(|i| Shape::LongTail { sigma: i as f64 * 0.05 }));
        shapes.extend((0..=19).map(|i| Shape::TwoHumps { apart: i as f64 * 0.05 }));
        shapes.extend([2.05, 2.2, 2.5, 3.0, 4.0, 6.0, 10.0, 30.0].map(|df| Shape::HeavyTails { df }));
        for i in 0..=14 {
            for w in 0..=10 {
                shapes.push(Shape::Mixture { at: -2.5 + 0.5 * i as f64, weight: w as f64 / 10.0 });
            }
        }
        let mut checked = 0;
        for s in &shapes {
            for k in 0..=60 {
                let k = 1.0 + k as f64 * 0.05;
                let beyond = 1.0 - s.within(k).expect("a finite standard deviation");
                assert!(beyond <= 1.0 / (k * k) + 1e-12, "{s:?} at k = {k}: {beyond}");
                checked += 1;
            }
        }
        assert!(checked > 5000);
        // (no standard deviation, no bound to draw)
        assert!(Shape::HeavyTails { df: 2.0 }.within(2.0).is_none() && Shape::HeavyTails { df: 1.0 }.within(2.0).is_none());
        let beyond = |s: Shape, k: f64| 1.0 - s.within(k).unwrap_or(f64::NAN);
        let defaults = |kind| Shape::of(kind, &Params::default());
        for (kind, two, three) in [(Kind::Peaked, "5.9 %", "1.4 %"), (Kind::Bell, "4.6 %", "0.27 %"), (Kind::LongTail, "4.5 %", "1.7 %"), (Kind::HeavyTails, "4.1 %", "1.4 %"), (Kind::TwoHumps, "0.58 %", ""), (Kind::Flat, "0 %", "0 %")] {
            let s = defaults(kind);
            assert_eq!(percent(beyond(s, 2.0)), two, "{kind:?}");
            if three.is_empty() {
                assert!(beyond(s, 3.0) < 1e-5, "{kind:?}");
            } else {
                assert_eq!(percent(beyond(s, 3.0)), three, "{kind:?}");
            }
        }
        assert_eq!((percent(1.0 / 4.0), percent(1.0 / 9.0)), ("25.0 %".to_string(), "11.1 %".to_string()));
        // the experiment shades the peaked shape's tails beyond two standard deviations
        let mut scene = Moments::create();
        scene.set_up(3);
        assert!(scene.tails && scene.k == 2.0 && scene.kind == Kind::Peaked);
    }

    /// Each side's turning moment about p is its pull: the right's less the left's is
    /// the mean less p, and each is the integral it says; the Cauchy's are infinite.
    /// The beam is level exactly when the pivot is at the mean.
    #[test]
    fn the_turning_moments_are_what_they_say() {
        for s in every() {
            let Some(m) = s.mean().get() else {
                assert_eq!((s.beyond(0.5), s.below(0.5)), (Value::Infinite, Value::Infinite), "{s:?}");
                continue;
            };
            for p in [-3.0, -1.0, 0.0, 0.5, 1.0, 2.5, 6.0] {
                let (r, l) = (s.beyond(p).get().unwrap_or(f64::NAN), s.below(p).get().unwrap_or(f64::NAN));
                assert!(r >= 0.0 && l >= 0.0 && ((r - l) - (m - p)).abs() < 1e-9, "{s:?} at {p}: {r} and {l}");
                if light(&s) {
                    let right = integral(s, |x| (x - p).max(0.0));
                    assert!((r - right).abs() < 1e-6, "{s:?} at {p}: {r} against {right}");
                }
            }
        }
        // the first experiment: the pivot under the peak, the beam tipping right
        let mut scene = Moments::create();
        scene.set_up(0);
        settle(&mut scene);
        let t = scene.target();
        assert_eq!(t, Shape::Mixture { at: 3.0, weight: 0.25 });
        assert!(matches!(t.peaks(), Peaks::Main(a, b) if num(a, 2) == "0.00" && num(b, 2) == "3.00"));
        assert_eq!(scene.pivot, 0.0);
        let shown = |v: Value| v.get().map(|v| num(v, 2)).unwrap_or_default();
        assert_eq!((shown(t.below(0.0)), shown(t.beyond(0.0)), shown(t.mean())), ("0.18".into(), "0.93".into(), "0.75".into()));
        assert!(scene.goals()[TILT] > 0.9, "tips right: {}", scene.goals()[TILT]);
        scene.pivot = 0.75;
        assert_eq!(scene.goals()[TILT], 0.0);
        scene.pivot = 0.6;
        assert!(scene.goals()[TILT] > 0.0);
        scene.pivot = 0.9;
        assert!(scene.goals()[TILT] < 0.0);
        // the opening: the pivot at the median, half the area each side, and still it tips
        let scene = Moments::create();
        let t = scene.target();
        assert!((t.cdf(scene.pivot) - 0.5).abs() < 1e-12 && scene.goals()[TILT] > 0.0);
        assert_eq!((shown(t.below(scene.pivot)), shown(t.beyond(scene.pivot))), ("0.17".into(), "0.37".into()));
    }

    /// The average square against the pivot is a parabola whose bottom is the sample
    /// mean (its height there the variance dividing by n); the average distance is least
    /// at the sample median, and flat between the two middle points of an even sample.
    #[test]
    fn the_curves_bottom_out_at_the_sample_mean_and_median() {
        for (seed, n) in [(1, 1), (2, 2), (3, 7), (4, 40), (5, 200)] {
            let u = uniforms(seed);
            let x: Vec<f64> = u[..n].iter().map(|u| Shape::LongTail { sigma: 0.6 }.quantile(*u, None)).collect();
            let (m, v, med) = (stats::mean(&x), stats::variance_n(&x), stats::median(&x));
            for i in 0..=2000 {
                let p = -1.0 + 6.0 * i as f64 / 2000.0;
                assert!((msd(&x, p) - (v + (p - m).powi(2))).abs() < 1e-9, "n = {n} at {p}");
                assert!(msd(&x, p) >= msd(&x, m) && mad(&x, p) >= mad(&x, med) - 1e-12, "n = {n} at {p}");
            }
            assert!((msd(&x, m) - v).abs() < 1e-12 && (summary(&x, 2.0).sd - v.sqrt()).abs() < 1e-12);
            if n % 2 == 0 {
                let s = stats::sorted(&x);
                let (a, b) = (s[n / 2 - 1], s[n / 2]);
                assert!((mad(&x, a) - mad(&x, b)).abs() < 1e-12 && (mad(&x, 0.5 * (a + b)) - mad(&x, a)).abs() < 1e-12, "n = {n}");
            }
        }
        // the table's sample column is `math::stats`'s
        let x: Vec<f64> = uniforms(9)[..40].iter().map(|u| Shape::Peaked.quantile(*u, None)).collect();
        let s = summary(&x, 2.0);
        assert_eq!((s.mean, s.median, s.skewness, s.kurtosis), (stats::mean(&x), stats::median(&x), Some(stats::skewness(&x)), Some(stats::excess_kurtosis(&x))));
        // the chart under the sample reaches above its curve's bottom, whatever the setting
        let mut scene = Moments::create();
        for k in 0..TRIALS.len() {
            scene.set_up(k);
            for squares in [true, false] {
                scene.squares = squares;
                let x = scene.sample();
                let least = if squares { msd(x, summary(x, 2.0).mean) } else { mad(x, summary(x, 2.0).median) };
                assert!(scene.goals()[CURVE_TOP] as f64 > 2.5 * least || x.is_empty(), "experiment {k}, squares {squares}");
            }
        }
    }

    /// The second and the fifth experiments, whose captions give a sample's numbers.
    #[test]
    fn the_samples_do_what_the_captions_say() {
        let two = |v: f64| num(v, 2);
        let mut s = Moments::create();
        // a long tail: its peak, median and mean, and the furthest point moved out
        s.set_up(1);
        let t = s.target();
        assert_eq!((t.peaks(), two(t.median()), t.mean().get().map(two)), (Peaks::One((-0.5625f64).exp()), "1.00".to_string(), Some("1.32".to_string())));
        assert_eq!(two((-0.5625f64).exp()), "0.57");
        assert_eq!((s.n, s.sample().len(), s.goals()[TILT]), (TAIL_N, TAIL_N, 0.0));
        let (wait, far, to) = s.later.expect("a point to move");
        let p = s.pivot;
        assert_eq!((wait, to, two(s.values[far])), (TAIL_WAIT, TAIL_FAR, "3.63".to_string()));
        assert!(s.values.iter().all(|v| *v <= s.values[far]), "the furthest point");
        let before = summary(s.sample(), 2.0);
        let (square, average) = ((s.values[far] - p).powi(2), msd(s.sample(), p));
        assert_eq!((two(before.mean), two(before.median), num(square, 1), two(average)), ("1.21".into(), "0.89".into(), "5.3".into(), "0.82".into()));
        let mut time = 0.0;
        while s.later.is_some() {
            s.advance(0.05, 0.05);
            time += 0.05;
        }
        assert!((time - TAIL_WAIT).abs() < 0.06 && s.values[far] == TAIL_FAR);
        let after = summary(s.sample(), 2.0);
        assert_eq!((two(after.mean), two(after.median), num((TAIL_FAR - p).powi(2), 1), two(msd(s.sample(), p))), ("1.38".into(), "0.89".into(), "32.2".into(), "2.16".into()));
        // no balance point: a sample of 200 from the Cauchy distribution, one at a time
        s.set_up(4);
        assert!(s.growing && s.history && s.count == 0 && s.values.len() == 200);
        assert_eq!((s.target().mean(), s.target().beyond(0.0), s.target().shape()[0]), (Value::Undefined, Value::Infinite, Value::Undefined));
        let (x, means, medians) = (s.values.clone(), s.means.clone(), s.medians.clone());
        assert_eq!((num(x[34], 0), num(means[34] - means[33], 1)), ("94".to_string(), "2.7".to_string()));
        assert_eq!((num(x[46], 0), num(means[46] - means[45], 1)), ("132".to_string(), "2.8".to_string()));
        assert!(x.iter().all(|v| v.abs() <= x[46]), "draw 47 is the furthest");
        assert_eq!(two(means[199]), "1.29");
        let (lo, hi) = medians[49..].iter().fold((f64::INFINITY, f64::NEG_INFINITY), |(a, b), v| (a.min(*v), b.max(*v)));
        assert_eq!((two(lo), two(hi)), ("−0.04".to_string(), "0.24".to_string()));
        // the running mean and median are those of the first m draws
        for m in [1, 2, 35, 200] {
            assert!((means[m - 1] - stats::mean(&x[..m])).abs() < 1e-12 && medians[m - 1] == stats::median(&x[..m]));
        }
        // the draws come in at the rate asked, and all of them come
        let mut time = 0.0f32;
        while s.growing {
            s.advance(1.0 / 60.0, 1.0 / 60.0);
            time += 1.0 / 60.0;
            assert!(time < 10.0, "{} drawn after ten seconds", s.count);
        }
        assert!(s.count == 200 && time > 6.5 && time < 7.5, "{time} seconds");
    }

    /// One point, a mixture collapsed to one bell, a pivot pushed off the chart, a sample
    /// being drawn and still empty, the ends of every slider: every number the picture is
    /// drawn from is finite, and what has no value says so.
    #[test]
    fn degenerate_cases_are_told_honestly() {
        let one = summary(&[2.0], 2.0);
        assert_eq!((one.n, one.mean, one.median, one.sd, one.skewness, one.kurtosis, one.within), (1, 2.0, 2.0, 0.0, None, None, [None; 3]));
        let two = summary(&[1.0, 1.0], 2.0);
        assert_eq!((two.sd, two.skewness, two.within[0]), (0.0, None, None));
        assert!((msd(&[2.0], 0.5) - 2.25).abs() < 1e-15 && msd(&[], 1.0) == 0.0 && mad(&[], 1.0) == 0.0);
        // a mixture with its bells on top of each other, or one of them gone
        let same = Shape::Mixture { at: 0.0, weight: 0.4 };
        assert!(matches!(same.peaks(), Peaks::One(a) if a.abs() < 1e-6) && same.mean() == Value::Is(0.0));
        let [v, skew, kurt] = same.shape().map(|v| v.get().unwrap_or(f64::NAN));
        assert!((v - 0.36).abs() < 1e-12 && skew.abs() < 1e-12 && kurt.abs() < 1e-12);
        for (w, centre) in [(0.0, 0.0), (1.0, 4.0)] {
            let s = Shape::Mixture { at: 4.0, weight: w };
            assert!(matches!(s.peaks(), Peaks::One(a) if (a - centre).abs() < 1e-6) && (s.median() - centre).abs() < 1e-9, "{s:?}");
        }
        let mut scene = Moments::create();
        // the pivot pushed far off: kept on the beam
        scene.pivot = 100.0;
        scene.reshape();
        assert_eq!(scene.pivot, scene.target().range()[1]);
        // every kind at the ends of its slider, with one point and with two hundred
        for kind in KINDS {
            for params in [Params { sigma: 0.1, apart: 0.0, df: 1.0, at: -2.5, weight: 0.0 }, Params { sigma: 1.0, apart: 0.95, df: 30.0, at: 4.5, weight: 1.0 }, Params::default()] {
                for n in [1, 2, MAX_N] {
                    scene.params = params;
                    scene.kind = kind;
                    scene.set_n(n);
                    scene.reshape();
                    let goals = scene.goals();
                    assert!(goals.iter().all(|g| g.is_finite()), "{kind:?} {params:?} {n}: {goals:?}");
                    assert!(scene.values.iter().all(|v| v.is_finite()), "{kind:?} {params:?} {n}");
                    let blend = scene.blend();
                    let [lo, hi] = scene.target().range();
                    assert!((0..=200).all(|i| blend.pdf(lo + (hi - lo) * i as f64 / 200.0).is_finite()), "{kind:?} {params:?}");
                    assert!(blend.median().is_finite() && blend.quantile(0.25).is_finite());
                    // the beam tips only by a finite amount, and not at all with no mean
                    assert!(goals[TILT].abs() <= 1.0);
                    if blend.mean().get().is_none() {
                        assert_eq!(goals[TILT], 0.0);
                    }
                }
            }
        }
        // a sample drawn one at a time, before its first point
        scene.set_up(4);
        assert!(scene.sample().is_empty() && scene.goals().iter().all(|g| g.is_finite()));
    }

    /// Every part has its place in every window, inside it and in order down the screen.
    #[test]
    fn the_layout_fits_every_window() {
        for (w, h, inset) in [(1078.0, 1000.0, 0.0), (1100.0, 700.0, 0.0), (1400.0, 760.0, 0.0), (390.0, 756.0, 44.0), (390.0, 800.0, 44.0), (600.0, 700.0, 0.0), (700.0, 500.0, 0.0), (240.0, 180.0, 0.0), (1.0, 1.0, 0.0), (3000.0, 2000.0, 0.0)] {
            let lay = Layout::of(Vec2::new(w, h), inset);
            let rects = [lay.density, lay.strip, lay.curve, lay.table, lay.column];
            let all = rects.iter().chain(&lay.shapes).chain(&lay.modes);
            assert!(all.clone().all(|(a, b)| a.is_finite() && b.is_finite()), "{w}x{h}");
            if w >= 390.0 && h >= 700.0 {
                assert!(!lay.compact, "{w}x{h}");
                assert!(all.clone().all(|(a, b)| a.x >= 0.0 && a.y >= inset && b.x <= w + 0.5 && b.y <= h + 0.5), "{w}x{h}");
                assert!(lay.shapes.iter().all(|r| r.1.y < lay.density.0.y - lay.gaps.title_d + 1.0), "{w}x{h}");
                assert!(lay.density.1.y + lay.gaps.under_d <= lay.strip.0.y - lay.gaps.title_s + 0.5, "{w}x{h}");
                assert!(lay.strip.1.y < lay.curve.0.y - lay.gaps.title_c && lay.curve.1.y + lay.gaps.under_c <= h, "{w}x{h}");
                assert!(lay.density.1.y - lay.density.0.y > 110.0 && lay.strip.1.y - lay.strip.0.y > 100.0 && lay.curve.1.y - lay.curve.0.y > 65.0, "{w}x{h}");
                if lay.narrow {
                    assert!(lay.curve.1.y + lay.gaps.under_c <= lay.table.0.y && lay.table.1.y <= h, "{w}x{h}");
                } else {
                    assert!(lay.table.0.x > lay.density.1.x + 20.0, "{w}x{h}");
                }
            }
        }
    }

    /// After a change, the picture glides to the new one and comes to rest exactly there,
    /// at any frame rate.
    #[test]
    fn glides_arrive_and_stop() {
        for hz in [30.0f32, 60.0, 144.0] {
            for k in 0..TRIALS.len() {
                let mut s = Moments::create();
                s.set_up(k);
                let mut frames = 0;
                loop {
                    s.advance(1.0 / hz, 1.0 / hz);
                    s.glide(1.0 / hz);
                    frames += 1;
                    if !s.animated() {
                        break;
                    }
                    assert!((frames as f32) < 30.0 * hz, "experiment {k} still moving after thirty seconds at {hz} Hz");
                }
                assert_eq!(s.look.shown, s.goals(), "experiment {k} at {hz} Hz");
                assert_eq!(s.points.shown, s.drawn_values());
                assert!(s.before.is_empty() && s.morph.get(0) == 1.0);
            }
        }
    }

    #[test]
    #[ignore]
    fn explore() {
        let mut s = Moments::create();
        let t = s.target();
        println!("opening: {:?} mean {:?} median {} peaks {:?} left {:?} right {:?}", t, t.mean(), t.median(), t.peaks(), t.below(s.pivot), t.beyond(s.pivot));
        // the balance point
        s.set_up(0);
        let t = s.target();
        println!("balance: mean {:?} median {} peaks {:?} left {:?} right {:?} sd {:?}", t.mean(), t.median(), t.peaks(), t.below(0.0), t.beyond(0.0), t.sd());
        // a long tail
        s.set_up(1);
        let t = s.target();
        let (i, far, to) = s.later.unwrap();
        let before = summary(s.sample(), 2.0);
        let p = s.pivot;
        let (sq0, avg0, mad0) = ((s.values[far] - p).powi(2), msd(s.sample(), p), mad(s.sample(), p));
        let old = s.values[far];
        s.advance(i + 0.01, 0.016);
        let after = summary(s.sample(), 2.0);
        println!("tail: mean {:?} median {} peaks {:?} sd {:?}; point {old} -> {to}", t.mean(), t.median(), t.peaks(), t.sd());
        println!("   sample mean {} -> {}, median {} -> {}, sd {} -> {}", before.mean, after.mean, before.median, after.median, before.sd, after.sd);
        println!("   its square {} -> {}, average square at pivot {} -> {}, mean distance {} -> {}", sq0, (to - p).powi(2), avg0, msd(s.sample(), p), mad0, mad(s.sample(), p));
        let sorted = stats::sorted(s.sample());
        println!("   sorted {:?}", sorted);
        // four shapes
        for (kind, apart) in [(Kind::Bell, 0.9), (Kind::Flat, 0.9), (Kind::TwoHumps, 0.9), (Kind::Peaked, 0.9)] {
            let sh = Shape::of(kind, &Params { apart, ..Params::default() });
            println!("{kind:?}: mean {:?} shape {:?} within {:?} {:?} {:?} median {} peaks {:?}", sh.mean(), sh.shape(), sh.within(1.0), sh.within(2.0), sh.within(3.0), sh.median(), sh.peaks());
        }
        // the bound
        for sh in [Shape::Peaked, Shape::Bell, Shape::LongTail { sigma: 0.6 }, Shape::TwoHumps { apart: 0.9 }, Shape::Flat, Shape::HeavyTails { df: 3.0 }, Shape::Mixture { at: 3.0, weight: 0.25 }] {
            let beyond = |k: f64| sh.within(k).map(|w| 1.0 - w);
            println!("{sh:?}: beyond 2: {:?} beyond 3: {:?}", beyond(2.0), beyond(3.0));
        }
        // no balance point
        s.set_up(4);
        s.count = s.values.len();
        let x = s.sample();
        let biggest = x.iter().enumerate().max_by(|a, b| a.1.abs().total_cmp(&b.1.abs())).unwrap();
        println!("cauchy: final mean {} median {}; largest draw {:?}", s.means[199], s.medians[199], biggest);
        let jumps: Vec<(usize, f64, f64)> = (1..200).map(|m| (m + 1, s.means[m] - s.means[m - 1], x[m])).filter(|j| j.1.abs() > 0.3).collect();
        println!("   jumps of the mean: {jumps:?}");
        let tail = |v: &[f64], from: usize| v[from..].iter().fold((f64::INFINITY, f64::NEG_INFINITY), |(a, b), x| (a.min(*x), b.max(*x)));
        println!("   mean from 50 on {:?}, from 100 on {:?}; median from 50 on {:?}, from 100 on {:?}", tail(&s.means, 49), tail(&s.means, 99), tail(&s.medians, 49), tail(&s.medians, 99));
    }
}

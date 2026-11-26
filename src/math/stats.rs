//! Probability distributions and summaries of samples, for the statistics scenes.
//!
//! Everything is in f64. A distribution is a value of [`Dist`]: its density (or the
//! probability of each whole number, for one that counts), its cumulative
//! probability, its quantiles, its moments where they exist, and draws from it, all
//! from one place so that a curve on a picture, a number in a caption and a sample in
//! a simulation agree. The special functions the distributions are made of are here
//! too and are accurate to about twelve digits; the tests hold them to known values.

use super::rng::Rng;
use std::f64::consts::PI;

// ----- special functions ---------------------------------------------------------------

/// The natural logarithm of the gamma function, for x > 0 (Lanczos).
pub fn ln_gamma(x: f64) -> f64 {
    const G: f64 = 7.0;
    const C: [f64; 9] = [
        0.999_999_999_999_809_9,
        676.520_368_121_885_1,
        -1_259.139_216_722_402_8,
        771.323_428_777_653_1,
        -176.615_029_162_140_6,
        12.507_343_278_686_905,
        -0.138_571_095_265_720_12,
        9.984_369_578_019_572e-6,
        1.505_632_735_149_311_6e-7,
    ];
    if x < 0.5 {
        // the reflection formula
        return (PI / (PI * x).sin().abs()).ln() - ln_gamma(1.0 - x);
    }
    let x = x - 1.0;
    let mut a = C[0];
    let t = x + G + 0.5;
    for (i, c) in C.iter().enumerate().skip(1) {
        a += c / (x + i as f64);
    }
    0.5 * (2.0 * PI).ln() + (x + 0.5) * t.ln() - t + a.ln()
}

/// ln of the beta function B(a, b).
pub fn ln_beta(a: f64, b: f64) -> f64 {
    ln_gamma(a) + ln_gamma(b) - ln_gamma(a + b)
}

/// ln of the number of ways to choose k of n.
pub fn ln_choose(n: f64, k: f64) -> f64 {
    ln_gamma(n + 1.0) - ln_gamma(k + 1.0) - ln_gamma(n - k + 1.0)
}

/// The regularised lower incomplete gamma function P(a, x), a > 0: 0 at x = 0, rising
/// to 1. (The share of a gamma distribution of shape a below x.)
pub fn gamma_p(a: f64, x: f64) -> f64 {
    if x.is_nan() || a.is_nan() || a <= 0.0 {
        return f64::NAN;
    }
    if x <= 0.0 {
        return 0.0;
    }
    if x == f64::INFINITY {
        return 1.0;
    }
    if x < a + 1.0 {
        gamma_series(a, x)
    } else {
        1.0 - gamma_fraction(a, x)
    }
}

/// 1 − P(a, x), kept accurate where it is small.
pub fn gamma_q(a: f64, x: f64) -> f64 {
    if x.is_nan() || a.is_nan() || a <= 0.0 {
        return f64::NAN;
    }
    if x <= 0.0 {
        return 1.0;
    }
    if x == f64::INFINITY {
        return 0.0;
    }
    if x < a + 1.0 {
        1.0 - gamma_series(a, x)
    } else {
        gamma_fraction(a, x)
    }
}

/// How many terms a series or continued fraction here may take: they need a few times
/// the square root of the larger parameter (a thousand was too few beyond a shape of
/// about 20 000, and the sums then stopped short without saying so).
fn terms(size: f64) -> usize {
    1000 + (40.0 * size.abs().sqrt()).min(4e6) as usize
}

fn gamma_series(a: f64, x: f64) -> f64 {
    let (mut sum, mut term, mut n) = (1.0 / a, 1.0 / a, a);
    for _ in 0..terms(a.max(x)) {
        n += 1.0;
        term *= x / n;
        sum += term;
        if term.abs() < sum.abs() * 1e-16 {
            break;
        }
    }
    (sum.ln() - x + a * x.ln() - ln_gamma(a)).exp()
}

fn gamma_fraction(a: f64, x: f64) -> f64 {
    // Lentz's method for the continued fraction of Q(a, x)
    const TINY: f64 = 1e-300;
    let mut b = x + 1.0 - a;
    let mut c = 1.0 / TINY;
    let mut d = 1.0 / b.max(TINY);
    let mut h = d;
    for i in 1..terms(a.max(x)) {
        let an = -(i as f64) * (i as f64 - a);
        b += 2.0;
        d = an * d + b;
        if d.abs() < TINY {
            d = TINY;
        }
        c = b + an / c;
        if c.abs() < TINY {
            c = TINY;
        }
        d = 1.0 / d;
        let delta = d * c;
        h *= delta;
        if (delta - 1.0).abs() < 1e-16 {
            break;
        }
    }
    (-x + a * x.ln() - ln_gamma(a)).exp() * h
}

/// The error function.
pub fn erf(x: f64) -> f64 {
    if x >= 0.0 {
        gamma_p(0.5, x * x)
    } else {
        -gamma_p(0.5, x * x)
    }
}

/// 1 − erf(x), kept accurate where it is small.
pub fn erfc(x: f64) -> f64 {
    if x >= 0.0 {
        gamma_q(0.5, x * x)
    } else {
        1.0 + gamma_p(0.5, x * x)
    }
}

/// The regularised incomplete beta function I_x(a, b), a, b > 0: 0 at x = 0, 1 at
/// x = 1. (The share of a beta distribution below x.)
pub fn beta_i(a: f64, b: f64, x: f64) -> f64 {
    if x.is_nan() || a.is_nan() || b.is_nan() || a <= 0.0 || b <= 0.0 {
        return f64::NAN;
    }
    if x <= 0.0 {
        return 0.0;
    }
    if x >= 1.0 {
        return 1.0;
    }
    let front = (a * x.ln() + b * (-x).ln_1p() - ln_beta(a, b)).exp();
    // (the continued fraction converges fast on one side of the mean, so the other
    // side is taken through the symmetry I_x(a, b) = 1 − I_{1−x}(b, a))
    if x < (a + 1.0) / (a + b + 2.0) {
        front * beta_fraction(a, b, x) / a
    } else {
        1.0 - front * beta_fraction(b, a, 1.0 - x) / b
    }
}

fn beta_fraction(a: f64, b: f64, x: f64) -> f64 {
    const TINY: f64 = 1e-300;
    let (qab, qap, qam) = (a + b, a + 1.0, a - 1.0);
    let mut c = 1.0;
    let mut d = 1.0 - qab * x / qap;
    if d.abs() < TINY {
        d = TINY;
    }
    d = 1.0 / d;
    let mut h = d;
    for m in 1..terms(a.max(b)) {
        let m = m as f64;
        let m2 = 2.0 * m;
        let aa = m * (b - m) * x / ((qam + m2) * (a + m2));
        d = 1.0 + aa * d;
        if d.abs() < TINY {
            d = TINY;
        }
        c = 1.0 + aa / c;
        if c.abs() < TINY {
            c = TINY;
        }
        d = 1.0 / d;
        h *= d * c;
        let aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2));
        d = 1.0 + aa * d;
        if d.abs() < TINY {
            d = TINY;
        }
        c = 1.0 + aa / c;
        if c.abs() < TINY {
            c = TINY;
        }
        d = 1.0 / d;
        let delta = d * c;
        h *= delta;
        if (delta - 1.0).abs() < 1e-16 {
            break;
        }
    }
    h
}

/// The share of a standard normal distribution below z.
pub fn normal_cdf(z: f64) -> f64 {
    0.5 * erfc(-z / std::f64::consts::SQRT_2)
}

/// The density of the standard normal distribution at z.
pub fn normal_pdf(z: f64) -> f64 {
    (-0.5 * z * z).exp() / (2.0 * PI).sqrt()
}

/// The z below which a share p of a standard normal distribution lies (Acklam's
/// approximation, then one step of Halley's method: good to about fifteen digits).
/// −∞ at p ≤ 0 and +∞ at p ≥ 1.
pub fn normal_quantile(p: f64) -> f64 {
    if p.is_nan() {
        return f64::NAN;
    }
    if p <= 0.0 {
        return f64::NEG_INFINITY;
    }
    if p >= 1.0 {
        return f64::INFINITY;
    }
    const A: [f64; 6] = [-3.969683028665376e1, 2.209460984245205e2, -2.759285104469687e2, 1.38357751867269e2, -3.066479806614716e1, 2.506628277459239];
    const B: [f64; 5] = [-5.447609879822406e1, 1.615858368580409e2, -1.556989798598866e2, 6.680131188771972e1, -1.328068155288572e1];
    const C: [f64; 6] = [-7.784894002430293e-3, -3.223964580411365e-1, -2.400758277161838, -2.549732539343734, 4.374664141464968, 2.938163982698783];
    const D: [f64; 4] = [7.784695709041462e-3, 3.224671290700398e-1, 2.445134137142996, 3.754408661907416];
    let low = 0.02425;
    let x = if p < low {
        let q = (-2.0 * p.ln()).sqrt();
        (((((C[0] * q + C[1]) * q + C[2]) * q + C[3]) * q + C[4]) * q + C[5]) / ((((D[0] * q + D[1]) * q + D[2]) * q + D[3]) * q + 1.0)
    } else if p <= 1.0 - low {
        let q = p - 0.5;
        let r = q * q;
        (((((A[0] * r + A[1]) * r + A[2]) * r + A[3]) * r + A[4]) * r + A[5]) * q / (((((B[0] * r + B[1]) * r + B[2]) * r + B[3]) * r + B[4]) * r + 1.0)
    } else {
        let q = (-2.0 * (1.0 - p).ln()).sqrt();
        -(((((C[0] * q + C[1]) * q + C[2]) * q + C[3]) * q + C[4]) * q + C[5]) / ((((D[0] * q + D[1]) * q + D[2]) * q + D[3]) * q + 1.0)
    };
    let e = normal_cdf(x) - p;
    let u = e / normal_pdf(x).max(1e-300);
    x - u / (1.0 + x * u / 2.0)
}

// ----- distributions -------------------------------------------------------------------

/// A probability distribution of one number.
///
/// The ones that count (`Bernoulli`, `Binomial`, `Poisson`, `Geometric`) put their
/// probability on whole numbers: for those, [`Dist::pdf`] is the probability of exactly
/// that number (0 between whole numbers). Parameters outside their ranges (a spread of
/// 0 or less, a probability outside 0 to 1) give NaN, never a panic.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Dist {
    /// The bell curve with this mean and standard deviation.
    Normal { mean: f64, sd: f64 },
    /// Every value from `lo` to `hi` as likely as every other.
    Uniform { lo: f64, hi: f64 },
    /// The waiting time for an event that happens `rate` times per unit of time.
    Exponential { rate: f64 },
    /// The waiting time for `shape` such events (any shape above 0).
    Gamma { shape: f64, rate: f64 },
    /// On 0 to 1: `a − 1` successes and `b − 1` failures seen, as a belief about a share.
    Beta { a: f64, b: f64 },
    /// The sum of the squares of `df` standard normal numbers.
    ChiSquare { df: f64 },
    /// Student's t: a standard normal number divided by an estimate of its spread
    /// made from `df` independent pieces.
    StudentT { df: f64 },
    /// The ratio of two such estimates of a variance, from `d1` and `d2` pieces.
    F { d1: f64, d2: f64 },
    /// A bell with tails so heavy that it has no mean.
    Cauchy { centre: f64, scale: f64 },
    /// A number whose logarithm is normal with mean `mu` and standard deviation `sigma`.
    LogNormal { mu: f64, sigma: f64 },
    /// 1 with probability `p`, else 0.
    Bernoulli { p: f64 },
    /// The number of successes in `n` independent tries, each with probability `p`.
    Binomial { n: u32, p: f64 },
    /// The number of events in a time in which `mean` are expected.
    Poisson { mean: f64 },
    /// The number of the try on which the first success comes (1, 2, ...).
    Geometric { p: f64 },
}

fn share(p: f64) -> bool {
    (0.0..=1.0).contains(&p)
}

impl Dist {
    pub const STANDARD_NORMAL: Dist = Dist::Normal { mean: 0.0, sd: 1.0 };

    /// Beyond this many degrees of freedom Student's t is worked out through the normal
    /// distribution: the exact formulas subtract logarithms of gamma functions of the
    /// size of the degrees of freedom themselves, and lose their digits there.
    const MANY: f64 = 1e6;

    /// For Student's t with very many degrees of freedom: the standard normal number
    /// with the same cumulative probability as `x`, and its slope in `x` (the error is
    /// of the order of one over the square of the degrees of freedom).
    fn as_normal(x: f64, df: f64) -> (f64, f64) {
        let root = (1.0 + x * x / (2.0 * df)).sqrt();
        let scale = 1.0 - 1.0 / (4.0 * df);
        (x * scale / root, scale / (root * root * root))
    }

    /// Whether its parameters are in their ranges.
    pub fn valid(&self) -> bool {
        match *self {
            Dist::Normal { mean, sd } => mean.is_finite() && sd > 0.0 && sd.is_finite(),
            Dist::Uniform { lo, hi } => lo.is_finite() && hi.is_finite() && hi > lo,
            Dist::Exponential { rate } => rate > 0.0 && rate.is_finite(),
            Dist::Gamma { shape, rate } => shape > 0.0 && shape.is_finite() && rate > 0.0 && rate.is_finite(),
            Dist::Beta { a, b } => a > 0.0 && a.is_finite() && b > 0.0 && b.is_finite(),
            Dist::ChiSquare { df } | Dist::StudentT { df } => df > 0.0 && df.is_finite(),
            Dist::F { d1, d2 } => d1 > 0.0 && d1.is_finite() && d2 > 0.0 && d2.is_finite(),
            Dist::Cauchy { centre, scale } => centre.is_finite() && scale > 0.0 && scale.is_finite(),
            Dist::LogNormal { mu, sigma } => mu.is_finite() && sigma > 0.0 && sigma.is_finite(),
            Dist::Bernoulli { p } => share(p),
            Dist::Binomial { p, .. } => share(p),
            Dist::Poisson { mean } => mean > 0.0 && mean.is_finite(),
            Dist::Geometric { p } => p > 0.0 && p <= 1.0,
        }
    }

    /// Whether it puts its probability on whole numbers.
    pub fn discrete(&self) -> bool {
        matches!(self, Dist::Bernoulli { .. } | Dist::Binomial { .. } | Dist::Poisson { .. } | Dist::Geometric { .. })
    }

    /// The least and greatest values it can take (infinite where there is no end).
    pub fn support(&self) -> (f64, f64) {
        match *self {
            Dist::Normal { .. } | Dist::StudentT { .. } | Dist::Cauchy { .. } => (f64::NEG_INFINITY, f64::INFINITY),
            Dist::Uniform { lo, hi } => (lo, hi),
            Dist::Exponential { .. } | Dist::Gamma { .. } | Dist::ChiSquare { .. } | Dist::F { .. } | Dist::LogNormal { .. } | Dist::Poisson { .. } => {
                (0.0, f64::INFINITY)
            }
            Dist::Beta { .. } | Dist::Bernoulli { .. } => (0.0, 1.0),
            Dist::Binomial { n, .. } => (0.0, n as f64),
            Dist::Geometric { .. } => (1.0, f64::INFINITY),
        }
    }

    /// The density at x; for a distribution that counts, the probability of exactly x.
    pub fn pdf(&self, x: f64) -> f64 {
        if !self.valid() || x.is_nan() {
            return f64::NAN;
        }
        let (lo, hi) = self.support();
        if x < lo || x > hi || x.is_infinite() {
            return 0.0;
        }
        if self.discrete() {
            if x != x.round() {
                return 0.0;
            }
            let k = x.round();
            return match *self {
                Dist::Bernoulli { p } => {
                    if k == 1.0 {
                        p
                    } else {
                        1.0 - p
                    }
                }
                Dist::Binomial { n, p } => {
                    let n = n as f64;
                    // (0 to the power 0 is 1 here: with p = 0 every try fails)
                    if p == 0.0 {
                        return if k == 0.0 { 1.0 } else { 0.0 };
                    }
                    if p == 1.0 {
                        return if k == n { 1.0 } else { 0.0 };
                    }
                    (ln_choose(n, k) + k * p.ln() + (n - k) * (1.0 - p).ln()).exp()
                }
                Dist::Poisson { mean } => (k * mean.ln() - mean - ln_gamma(k + 1.0)).exp(),
                Dist::Geometric { p } => {
                    if p == 1.0 {
                        return if k == 1.0 { 1.0 } else { 0.0 };
                    }
                    ((k - 1.0) * (1.0 - p).ln()).exp() * p
                }
                _ => f64::NAN,
            };
        }
        match *self {
            Dist::Normal { mean, sd } => normal_pdf((x - mean) / sd) / sd,
            Dist::Uniform { lo, hi } => 1.0 / (hi - lo),
            Dist::Exponential { rate } => rate * (-rate * x).exp(),
            Dist::Gamma { shape, rate } => gamma_pdf(shape, rate, x),
            Dist::Beta { a, b } => {
                if (x == 0.0 && a < 1.0) || (x == 1.0 && b < 1.0) {
                    return f64::INFINITY;
                }
                if x == 0.0 {
                    return if a == 1.0 { b } else { 0.0 };
                }
                if x == 1.0 {
                    return if b == 1.0 { a } else { 0.0 };
                }
                ((a - 1.0) * x.ln() + (b - 1.0) * (1.0 - x).ln() - ln_beta(a, b)).exp()
            }
            Dist::ChiSquare { df } => gamma_pdf(df / 2.0, 0.5, x),
            Dist::StudentT { df } if df > Self::MANY => {
                let (z, slope) = Self::as_normal(x, df);
                normal_pdf(z) * slope
            }
            Dist::StudentT { df } => {
                (ln_gamma((df + 1.0) / 2.0) - ln_gamma(df / 2.0) - 0.5 * (df * PI).ln() - (df + 1.0) / 2.0 * (1.0 + x * x / df).ln()).exp()
            }
            Dist::F { d1, d2 } => {
                if x == 0.0 {
                    return if d1 < 2.0 {
                        f64::INFINITY
                    } else if d1 == 2.0 {
                        1.0
                    } else {
                        0.0
                    };
                }
                (0.5 * (d1 * (d1 * x).ln() + d2 * d2.ln() - (d1 + d2) * (d1 * x + d2).ln()) - x.ln() - ln_beta(d1 / 2.0, d2 / 2.0)).exp()
            }
            Dist::Cauchy { centre, scale } => {
                let z = (x - centre) / scale;
                1.0 / (PI * scale * (1.0 + z * z))
            }
            Dist::LogNormal { mu, sigma } => {
                if x <= 0.0 {
                    0.0
                } else {
                    normal_pdf((x.ln() - mu) / sigma) / (sigma * x)
                }
            }
            _ => f64::NAN,
        }
    }

    /// The probability of a value of x or less.
    pub fn cdf(&self, x: f64) -> f64 {
        if !self.valid() || x.is_nan() {
            return f64::NAN;
        }
        let (lo, hi) = self.support();
        if x < lo || x == f64::NEG_INFINITY {
            return 0.0;
        }
        if x >= hi {
            return 1.0;
        }
        match *self {
            Dist::Normal { mean, sd } => normal_cdf((x - mean) / sd),
            Dist::Uniform { lo, hi } => (x - lo) / (hi - lo),
            Dist::Exponential { rate } => -(-rate * x).exp_m1(),
            Dist::Gamma { shape, rate } => gamma_p(shape, rate * x),
            Dist::Beta { a, b } => beta_i(a, b, x),
            Dist::ChiSquare { df } => gamma_p(df / 2.0, x / 2.0),
            Dist::StudentT { df } if df > Self::MANY => normal_cdf(Self::as_normal(x, df).0),
            Dist::StudentT { df } => {
                let x2 = x * x;
                if x2 < df {
                    // near the middle: the half, and the share between 0 and x (the
                    // tail's form loses its digits where df / (df + x²) rounds to 1)
                    let middle = 0.5 * beta_i(0.5, df / 2.0, x2 / (df + x2));
                    if x >= 0.0 {
                        0.5 + middle
                    } else {
                        0.5 - middle
                    }
                } else {
                    let tail = 0.5 * beta_i(df / 2.0, 0.5, df / (df + x2));
                    if x >= 0.0 {
                        1.0 - tail
                    } else {
                        tail
                    }
                }
            }
            Dist::F { d1, d2 } => beta_i(d1 / 2.0, d2 / 2.0, d1 * x / (d1 * x + d2)),
            Dist::Cauchy { centre, scale } => 0.5 + ((x - centre) / scale).atan() / PI,
            Dist::LogNormal { mu, sigma } => {
                if x <= 0.0 {
                    0.0
                } else {
                    normal_cdf((x.ln() - mu) / sigma)
                }
            }
            Dist::Bernoulli { p } => 1.0 - p,
            Dist::Binomial { n, p } => {
                let k = x.floor();
                if p == 0.0 {
                    return 1.0;
                }
                if p == 1.0 {
                    return 0.0;
                }
                // P(X ≤ k) is the share of a beta distribution above p
                1.0 - beta_i(k + 1.0, n as f64 - k, p)
            }
            Dist::Poisson { mean } => gamma_q(x.floor() + 1.0, mean),
            Dist::Geometric { p } => 1.0 - (x.floor() * (1.0 - p).ln()).exp(),
        }
    }

    /// The value below which (or at which, for a distribution that counts) a share p
    /// lies: the least x with `cdf(x) ≥ p`. The ends of the support at p = 0 and 1.
    pub fn quantile(&self, p: f64) -> f64 {
        if !self.valid() || p.is_nan() {
            return f64::NAN;
        }
        let (lo, hi) = self.support();
        if p <= 0.0 {
            return lo;
        }
        if p >= 1.0 {
            return hi;
        }
        match *self {
            Dist::Normal { mean, sd } => mean + sd * normal_quantile(p),
            Dist::Uniform { lo, hi } => lo + (hi - lo) * p,
            Dist::Exponential { rate } => -(-p).ln_1p() / rate,
            Dist::Cauchy { centre, scale } => centre + scale * (PI * (p - 0.5)).tan(),
            Dist::LogNormal { mu, sigma } => (mu + sigma * normal_quantile(p)).exp(),
            Dist::Bernoulli { p: q } => {
                if p <= 1.0 - q {
                    0.0
                } else {
                    1.0
                }
            }
            Dist::Geometric { p: q } => {
                if q == 1.0 {
                    return 1.0;
                }
                // (the formula, then a step either way: rounding can put it one off
                // exactly where p is the cumulative probability of a whole number)
                let mut k = ((-p).ln_1p() / (1.0 - q).ln()).ceil().max(1.0);
                while k > 1.0 && self.cdf(k - 1.0) >= p {
                    k -= 1.0;
                }
                while self.cdf(k) < p {
                    k += 1.0;
                }
                k
            }
            Dist::Binomial { .. } | Dist::Poisson { .. } => self.count_quantile(p),
            _ => self.search_quantile(p),
        }
    }

    /// The least whole number with `cdf ≥ p`, by doubling and then halving.
    fn count_quantile(&self, p: f64) -> f64 {
        let (lo, hi) = self.support();
        let centre = self.mean().unwrap_or(lo).max(lo);
        let spread = self.variance().unwrap_or(1.0).sqrt().max(1.0);
        let (mut a, mut b) = (lo, (centre + spread).ceil().min(hi));
        while self.cdf(b) < p && b < hi {
            a = b;
            b = (b + (b - lo).max(spread) + 1.0).ceil().min(hi);
        }
        // (the cdf at `a − 1` is below p, or a is the least value; at b it is at least p)
        while b - a > 0.5 {
            let m = ((a + b) / 2.0).floor();
            if self.cdf(m) >= p {
                b = m;
            } else {
                a = m + 1.0;
            }
        }
        if self.cdf(a) >= p {
            a
        } else {
            b
        }
    }

    /// The x with `cdf(x) = p` for a distribution with a density, by Newton's steps
    /// kept inside a bracket (halving when a step would leave it).
    fn search_quantile(&self, p: f64) -> f64 {
        let (lo, hi) = self.support();
        // a bracket: from the support if it ends, else grown outwards from a guess
        let guess = match *self {
            Dist::StudentT { .. } => normal_quantile(p),
            _ => self.mean().unwrap_or(1.0),
        };
        let step = self.variance().filter(|v| v.is_finite()).unwrap_or(1.0).sqrt().max(1e-6);
        let (mut a, mut b) = (guess - step, guess + step);
        if lo.is_finite() {
            a = lo;
        } else {
            let mut w = step;
            while self.cdf(a) > p && w < 1e300 {
                w *= 2.0;
                a = guess - w;
            }
        }
        if hi.is_finite() {
            b = hi;
        } else {
            let mut w = step;
            while self.cdf(b) < p && w < 1e300 {
                w *= 2.0;
                b = guess + w;
            }
        }
        let mut x = guess.max(a).min(b);
        if !(x > a && x < b) {
            x = 0.5 * (a + b);
        }
        for _ in 0..400 {
            let e = self.cdf(x) - p;
            if e > 0.0 {
                b = x;
            } else {
                a = x;
            }
            let d = self.pdf(x);
            let next = if d.is_finite() && d > 0.0 { x - e / d } else { f64::NAN };
            let next = if next > a && next < b { next } else { 0.5 * (a + b) };
            // (relative to where it is: a quantile of 1e-20 is as much a number as one of 1)
            if (next - x).abs() <= 1e-14 * x.abs().max(next.abs()) || b - a <= 1e-15 * a.abs().max(b.abs()) || next == x {
                return next;
            }
            x = next;
        }
        x
    }

    /// The mean, where there is one.
    pub fn mean(&self) -> Option<f64> {
        if !self.valid() {
            return None;
        }
        match *self {
            Dist::Normal { mean, .. } => Some(mean),
            Dist::Uniform { lo, hi } => Some(0.5 * (lo + hi)),
            Dist::Exponential { rate } => Some(1.0 / rate),
            Dist::Gamma { shape, rate } => Some(shape / rate),
            Dist::Beta { a, b } => Some(a / (a + b)),
            Dist::ChiSquare { df } => Some(df),
            Dist::StudentT { df } => (df > 1.0).then_some(0.0),
            Dist::F { d2, .. } => (d2 > 2.0).then(|| d2 / (d2 - 2.0)),
            Dist::Cauchy { .. } => None,
            Dist::LogNormal { mu, sigma } => Some((mu + 0.5 * sigma * sigma).exp()),
            Dist::Bernoulli { p } => Some(p),
            Dist::Binomial { n, p } => Some(n as f64 * p),
            Dist::Poisson { mean } => Some(mean),
            Dist::Geometric { p } => Some(1.0 / p),
        }
    }

    /// The variance, where it is finite.
    pub fn variance(&self) -> Option<f64> {
        if !self.valid() {
            return None;
        }
        match *self {
            Dist::Normal { sd, .. } => Some(sd * sd),
            Dist::Uniform { lo, hi } => Some((hi - lo) * (hi - lo) / 12.0),
            Dist::Exponential { rate } => Some(1.0 / (rate * rate)),
            Dist::Gamma { shape, rate } => Some(shape / (rate * rate)),
            Dist::Beta { a, b } => Some(a * b / ((a + b) * (a + b) * (a + b + 1.0))),
            Dist::ChiSquare { df } => Some(2.0 * df),
            Dist::StudentT { df } => (df > 2.0).then(|| df / (df - 2.0)),
            Dist::F { d1, d2 } => (d2 > 4.0).then(|| 2.0 * d2 * d2 * (d1 + d2 - 2.0) / (d1 * (d2 - 2.0) * (d2 - 2.0) * (d2 - 4.0))),
            Dist::Cauchy { .. } => None,
            Dist::LogNormal { mu, sigma } => Some((sigma * sigma).exp_m1() * (2.0 * mu + sigma * sigma).exp()),
            Dist::Bernoulli { p } => Some(p * (1.0 - p)),
            Dist::Binomial { n, p } => Some(n as f64 * p * (1.0 - p)),
            Dist::Poisson { mean } => Some(mean),
            Dist::Geometric { p } => Some((1.0 - p) / (p * p)),
        }
    }

    /// The standard deviation, where it is finite.
    pub fn sd(&self) -> Option<f64> {
        self.variance().map(f64::sqrt)
    }

    /// One draw.
    pub fn sample(&self, rng: &mut Rng) -> f64 {
        if !self.valid() {
            return f64::NAN;
        }
        match *self {
            Dist::Normal { mean, sd } => mean + sd * normal(rng),
            Dist::Uniform { lo, hi } => lo + (hi - lo) * unit(rng),
            Dist::Exponential { rate } => -unit(rng).ln() / rate,
            Dist::Gamma { shape, rate } => gamma_draw(shape, rng) / rate,
            Dist::Beta { a, b } => {
                let (x, y) = (gamma_draw(a, rng), gamma_draw(b, rng));
                if x + y > 0.0 {
                    x / (x + y)
                } else {
                    0.5
                }
            }
            Dist::ChiSquare { df } => 2.0 * gamma_draw(df / 2.0, rng),
            Dist::StudentT { df } => {
                let z = normal(rng);
                z / (2.0 * gamma_draw(df / 2.0, rng) / df).sqrt().max(1e-300)
            }
            Dist::F { d1, d2 } => (gamma_draw(d1 / 2.0, rng) / d1) / (gamma_draw(d2 / 2.0, rng) / d2).max(1e-300),
            Dist::Cauchy { centre, scale } => centre + scale * (PI * (unit(rng) - 0.5)).tan(),
            Dist::LogNormal { mu, sigma } => (mu + sigma * normal(rng)).exp(),
            Dist::Bernoulli { p } => (unit(rng) < p) as u8 as f64,
            Dist::Binomial { n, p } => {
                if n <= 256 {
                    (0..n).filter(|_| unit(rng) < p).count() as f64
                } else {
                    self.quantile(unit(rng))
                }
            }
            Dist::Poisson { mean } => {
                if mean < 30.0 {
                    // events counted until their waiting times pass one unit of time
                    let limit = (-mean).exp();
                    let (mut k, mut product) = (0.0, unit(rng));
                    while product > limit {
                        k += 1.0;
                        product *= unit(rng);
                    }
                    k
                } else {
                    self.quantile(unit(rng))
                }
            }
            Dist::Geometric { .. } => self.quantile(unit(rng)),
        }
    }

    /// `n` draws.
    pub fn draws(&self, n: usize, rng: &mut Rng) -> Vec<f64> {
        (0..n).map(|_| self.sample(rng)).collect()
    }
}

fn gamma_pdf(shape: f64, rate: f64, x: f64) -> f64 {
    if x == 0.0 {
        return if shape < 1.0 {
            f64::INFINITY
        } else if shape == 1.0 {
            rate
        } else {
            0.0
        };
    }
    (shape * rate.ln() + (shape - 1.0) * x.ln() - rate * x - ln_gamma(shape)).exp()
}

/// A number strictly between 0 and 1, every value as likely as every other.
pub fn unit(rng: &mut Rng) -> f64 {
    // (52 bits: with 53 the largest value plus a half rounds up to exactly 1)
    ((rng.next_u64() >> 12) as f64 + 0.5) / (1u64 << 52) as f64
}

/// A standard normal number.
pub fn normal(rng: &mut Rng) -> f64 {
    (-2.0 * unit(rng).ln()).sqrt() * (2.0 * PI * unit(rng)).cos()
}

/// A draw from the gamma distribution of this shape and rate 1 (Marsaglia and Tsang).
fn gamma_draw(shape: f64, rng: &mut Rng) -> f64 {
    if shape < 1.0 {
        // a draw of shape + 1, scaled down
        return gamma_draw(shape + 1.0, rng) * unit(rng).powf(1.0 / shape);
    }
    let d = shape - 1.0 / 3.0;
    let c = 1.0 / (9.0 * d).sqrt();
    loop {
        let z = normal(rng);
        let v = 1.0 + c * z;
        if v <= 0.0 {
            continue;
        }
        let v = v * v * v;
        let u = unit(rng);
        if u.ln() < 0.5 * z * z + d - d * v + d * v.ln() {
            return d * v;
        }
    }
}

// ----- summaries of a sample -----------------------------------------------------------

/// The mean (0 of no numbers).
pub fn mean(x: &[f64]) -> f64 {
    if x.is_empty() {
        return 0.0;
    }
    x.iter().sum::<f64>() / x.len() as f64
}

/// The mean squared distance from the mean: dividing by n, as for a whole population.
pub fn variance_n(x: &[f64]) -> f64 {
    if x.is_empty() {
        return 0.0;
    }
    let m = mean(x);
    x.iter().map(|v| (v - m) * (v - m)).sum::<f64>() / x.len() as f64
}

/// The sample variance: dividing by n − 1, which makes it right on average for the
/// population the sample was drawn from (0 of fewer than two numbers).
pub fn variance(x: &[f64]) -> f64 {
    if x.len() < 2 {
        return 0.0;
    }
    variance_n(x) * x.len() as f64 / (x.len() - 1) as f64
}

/// The sample standard deviation (the square root of [`variance`]).
pub fn sd(x: &[f64]) -> f64 {
    variance(x).sqrt()
}

/// The numbers in rising order (those that are not numbers left out).
pub fn sorted(x: &[f64]) -> Vec<f64> {
    let mut s: Vec<f64> = x.iter().copied().filter(|v| !v.is_nan()).collect();
    s.sort_by(f64::total_cmp);
    s
}

/// The value a share p of the way up a sorted sample, between two of its numbers in
/// proportion where p falls between them (NaN of no numbers).
pub fn quantile_sorted(sorted: &[f64], p: f64) -> f64 {
    if sorted.is_empty() || p.is_nan() {
        return f64::NAN;
    }
    let at = p.max(0.0).min(1.0) * (sorted.len() - 1) as f64;
    let i = at.floor() as usize;
    let j = (i + 1).min(sorted.len() - 1);
    sorted[i] + (sorted[j] - sorted[i]) * (at - i as f64)
}

/// The middle value of a sample.
pub fn median(x: &[f64]) -> f64 {
    quantile_sorted(&sorted(x), 0.5)
}

/// The third standardised moment: 0 for a symmetric sample, positive for a long tail
/// to the right (0 if there is no spread).
pub fn skewness(x: &[f64]) -> f64 {
    let (m, v) = (mean(x), variance_n(x));
    if x.is_empty() || v <= 0.0 {
        return 0.0;
    }
    x.iter().map(|a| (a - m).powi(3)).sum::<f64>() / x.len() as f64 / v.powf(1.5)
}

/// The fourth standardised moment less 3: 0 for a normal sample, positive for heavy
/// tails (0 if there is no spread).
pub fn excess_kurtosis(x: &[f64]) -> f64 {
    let (m, v) = (mean(x), variance_n(x));
    if x.is_empty() || v <= 0.0 {
        return 0.0;
    }
    x.iter().map(|a| (a - m).powi(4)).sum::<f64>() / x.len() as f64 / (v * v) - 3.0
}

/// The covariance of two samples, dividing by n − 1.
pub fn covariance(x: &[f64], y: &[f64]) -> f64 {
    let n = x.len().min(y.len());
    if n < 2 {
        return 0.0;
    }
    let (mx, my) = (mean(&x[..n]), mean(&y[..n]));
    x.iter().zip(y).map(|(a, b)| (a - mx) * (b - my)).sum::<f64>() / (n - 1) as f64
}

/// The correlation coefficient, from −1 to 1 (0 if either sample has no spread).
pub fn correlation(x: &[f64], y: &[f64]) -> f64 {
    let n = x.len().min(y.len());
    let spread = (variance(&x[..n]) * variance(&y[..n])).sqrt();
    if spread > 0.0 && spread.is_finite() {
        (covariance(x, y) / spread).max(-1.0).min(1.0)
    } else {
        0.0
    }
}

/// The place of each number in the sorted order, from 1; numbers that are equal share
/// the mean of their places.
pub fn ranks(x: &[f64]) -> Vec<f64> {
    let mut order: Vec<usize> = (0..x.len()).collect();
    order.sort_by(|&a, &b| x[a].total_cmp(&x[b]));
    let mut ranks = vec![0.0; x.len()];
    let mut i = 0;
    while i < order.len() {
        let mut j = i;
        while j + 1 < order.len() && x[order[j + 1]] == x[order[i]] {
            j += 1;
        }
        let shared = (i + j) as f64 / 2.0 + 1.0;
        for &k in &order[i..=j] {
            ranks[k] = shared;
        }
        i = j + 1;
    }
    ranks
}

/// Counts of a sample in equal bins from `lo` to `hi`.
#[derive(Clone, Debug, PartialEq)]
pub struct Histogram {
    pub lo: f64,
    pub hi: f64,
    pub counts: Vec<u32>,
    /// How many numbers have been added, those outside `lo` to `hi` included.
    pub total: u64,
}

impl Histogram {
    pub fn new(lo: f64, hi: f64, bins: usize) -> Self {
        Histogram { lo, hi, counts: vec![0; bins.max(1)], total: 0 }
    }

    pub fn of(values: &[f64], lo: f64, hi: f64, bins: usize) -> Self {
        let mut h = Histogram::new(lo, hi, bins);
        for &v in values {
            h.add(v);
        }
        h
    }

    /// The width of a bin.
    pub fn width(&self) -> f64 {
        (self.hi - self.lo) / self.counts.len() as f64
    }

    /// The bin a value falls in, if it is inside the range.
    pub fn bin(&self, v: f64) -> Option<usize> {
        let w = self.width();
        if !(v >= self.lo && v <= self.hi) || w.is_nan() || w <= 0.0 {
            return None;
        }
        Some((((v - self.lo) / w) as usize).min(self.counts.len() - 1))
    }

    pub fn add(&mut self, v: f64) {
        if v.is_nan() {
            return;
        }
        self.total += 1;
        if let Some(i) = self.bin(v) {
            self.counts[i] += 1;
        }
    }

    pub fn clear(&mut self) {
        self.counts.iter_mut().for_each(|c| *c = 0);
        self.total = 0;
    }

    /// The height of bin `i` as a density: its share of all the numbers added,
    /// divided by its width, so that the bars' areas add up to the share inside the
    /// range and a density curve can be drawn over them.
    pub fn density(&self, i: usize) -> f64 {
        let w = self.width();
        if self.total == 0 || w.is_nan() || w <= 0.0 {
            return 0.0;
        }
        self.counts.get(i).map_or(0.0, |&c| c as f64 / self.total as f64 / w)
    }

    /// The middle of bin `i`.
    pub fn centre(&self, i: usize) -> f64 {
        self.lo + (i as f64 + 0.5) * self.width()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn close(a: f64, b: f64, tol: f64) -> bool {
        (a - b).abs() <= tol * (1.0 + b.abs())
    }

    #[test]
    fn the_special_functions_have_their_known_values() {
        assert!(close(ln_gamma(0.5), 0.572_364_942_924_700_1, 1e-12));
        assert!(close(ln_gamma(1.0), 0.0, 1e-12));
        assert!(close(ln_gamma(10.0), 12.801_827_480_081_469, 1e-12));
        assert!(close(ln_gamma(0.1), 2.252_712_651_734_206, 1e-12));
        assert!(close(erf(0.5), 0.520_499_877_813_046_5, 1e-12));
        assert!(close(erf(-0.5), -0.520_499_877_813_046_5, 1e-12));
        assert!(close(erfc(3.0), 2.209_049_699_858_544e-5, 1e-10));
        assert!(close(normal_cdf(1.96), 0.975_002_104_851_779_5, 1e-12));
        assert!(close(normal_cdf(-1.0), 0.158_655_253_931_457_07, 1e-12));
        assert!(close(normal_cdf(-6.0), 9.865_876_450_376_946e-10, 1e-9));
        assert!(close(normal_quantile(0.975), 1.959_963_984_540_054, 1e-12));
        assert!(close(normal_quantile(0.001), -3.090_232_306_167_813, 1e-12));
        assert!(close(gamma_p(1.0, 2.0), 1.0 - (-2.0f64).exp(), 1e-12));
        assert!(close(gamma_p(3.0, 2.5) + gamma_q(3.0, 2.5), 1.0, 1e-14));
        // I_x(2, 3) = x² (6 − 8x + 3x²)
        assert!(close(beta_i(2.0, 3.0, 0.5), 0.6875, 1e-12));
        assert!(close(beta_i(2.0, 3.0, 0.9), 0.81 * (6.0 - 7.2 + 2.43), 1e-12));
        assert!(close(ln_choose(10.0, 3.0), 120f64.ln(), 1e-12));
    }

    #[test]
    fn the_tables_in_the_back_of_the_book() {
        // the 95 % points every statistics book prints
        assert!(close(Dist::ChiSquare { df: 1.0 }.cdf(3.841_458_820_694_124), 0.95, 1e-10));
        assert!(close(Dist::ChiSquare { df: 2.0 }.quantile(0.95), 5.991_464_547_107_979, 1e-10));
        assert!(close(Dist::ChiSquare { df: 10.0 }.quantile(0.95), 18.307_038_053_275_146, 1e-10));
        assert!(close(Dist::StudentT { df: 10.0 }.cdf(2.228_138_851_986_274), 0.975, 1e-10));
        assert!(close(Dist::StudentT { df: 10.0 }.quantile(0.975), 2.228_138_851_986_274, 1e-10));
        assert!(close(Dist::StudentT { df: 1.0 }.cdf(1.0), 0.75, 1e-12));
        assert!(close(Dist::StudentT { df: 3.0 }.cdf(-2.0), 0.069_662_984_279_421_6, 1e-10));
        assert!(close(Dist::F { d1: 2.0, d2: 10.0 }.quantile(0.95), 4.102_821_015_130_399, 1e-9));
        assert!(close(Dist::F { d1: 3.0, d2: 20.0 }.cdf(3.098_391_212_141_683), 0.95, 1e-9));
        assert!(close(Dist::Binomial { n: 10, p: 0.5 }.cdf(3.0), 176.0 / 1024.0, 1e-12));
        assert!(close(Dist::Binomial { n: 10, p: 0.5 }.pdf(5.0), 252.0 / 1024.0, 1e-12));
        assert!(close(Dist::Poisson { mean: 2.0 }.cdf(2.0), 5.0 * (-2.0f64).exp(), 1e-12));
        assert!(close(Dist::Geometric { p: 0.25 }.cdf(3.0), 1.0 - 0.75f64.powi(3), 1e-12));
        assert!(close(Dist::Exponential { rate: 2.0 }.quantile(0.5), 2f64.ln() / 2.0, 1e-12));
        assert!(close(Dist::Gamma { shape: 2.0, rate: 1.0 }.cdf(1.0), 1.0 - 2.0 * (-1.0f64).exp(), 1e-12));
        assert!(close(Dist::Beta { a: 2.0, b: 3.0 }.quantile(0.6875), 0.5, 1e-10));
        assert!(close(Dist::LogNormal { mu: 0.0, sigma: 1.0 }.cdf(1.0), 0.5, 1e-12));
        assert!(close(Dist::Cauchy { centre: 0.0, scale: 1.0 }.quantile(0.75), 1.0, 1e-12));
    }

    fn every() -> Vec<Dist> {
        vec![
            Dist::Normal { mean: 1.0, sd: 2.0 },
            Dist::Uniform { lo: -1.0, hi: 3.0 },
            Dist::Exponential { rate: 0.7 },
            Dist::Gamma { shape: 0.6, rate: 2.0 },
            Dist::Gamma { shape: 4.5, rate: 0.5 },
            Dist::Beta { a: 0.5, b: 0.5 },
            Dist::Beta { a: 3.0, b: 7.0 },
            Dist::ChiSquare { df: 3.0 },
            Dist::StudentT { df: 2.5 },
            Dist::StudentT { df: 30.0 },
            Dist::F { d1: 4.0, d2: 12.0 },
            Dist::Cauchy { centre: 0.5, scale: 2.0 },
            Dist::LogNormal { mu: 0.2, sigma: 0.6 },
            Dist::Bernoulli { p: 0.3 },
            Dist::Binomial { n: 12, p: 0.35 },
            Dist::Binomial { n: 400, p: 0.02 },
            Dist::Poisson { mean: 3.4 },
            Dist::Poisson { mean: 60.0 },
            Dist::Geometric { p: 0.2 },
        ]
    }

    #[test]
    fn a_quantile_undoes_the_cumulative_probability() {
        for d in every() {
            for &p in &[0.001, 0.02, 0.25, 0.5, 0.8, 0.975, 0.999] {
                let x = d.quantile(p);
                assert!(x.is_finite(), "{d:?} at {p}: {x}");
                if d.discrete() {
                    // the least whole number whose cumulative probability reaches p
                    assert!(d.cdf(x) >= p - 1e-12, "{d:?} at {p}: cdf({x}) = {}", d.cdf(x));
                    assert!(x == d.support().0 || d.cdf(x - 1.0) < p + 1e-12, "{d:?} at {p}: cdf({}) = {}", x - 1.0, d.cdf(x - 1.0));
                } else {
                    assert!(close(d.cdf(x), p, 1e-9), "{d:?} at {p}: cdf({x}) = {}", d.cdf(x));
                }
            }
            assert_eq!((d.quantile(0.0), d.quantile(1.0)), d.support());
        }
    }

    #[test]
    fn the_density_is_the_slope_of_the_cumulative_probability() {
        for d in every().into_iter().filter(|d| !d.discrete()) {
            for &p in &[0.1, 0.4, 0.7, 0.95] {
                let x = d.quantile(p);
                let h = 1e-5 * (1.0 + x.abs());
                let slope = (d.cdf(x + h) - d.cdf(x - h)) / (2.0 * h);
                assert!(close(slope, d.pdf(x), 1e-5), "{d:?} at {x}: slope {slope}, density {}", d.pdf(x));
            }
        }
        // and for the ones that count, the probabilities add up to the cumulative one
        for d in every().into_iter().filter(|d| d.discrete()) {
            let top = d.quantile(0.999);
            let sum: f64 = (0..=top as u32).map(|k| d.pdf(k as f64)).sum();
            assert!(close(sum, d.cdf(top), 1e-10), "{d:?}: {sum} against {}", d.cdf(top));
        }
    }

    #[test]
    fn draws_have_the_mean_and_spread_the_formulas_give() {
        let mut rng = Rng::new(7);
        for d in every() {
            let x = d.draws(40_000, &mut rng);
            assert!(x.iter().all(|v| v.is_finite()), "{d:?}");
            let (lo, hi) = d.support();
            assert!(x.iter().all(|v| *v >= lo && *v <= hi), "{d:?}");
            if let (Some(m), Some(v)) = (d.mean(), d.variance()) {
                let se = (v / x.len() as f64).sqrt();
                assert!((mean(&x) - m).abs() < 5.0 * se, "{d:?}: mean {} for {m}", mean(&x));
                // (a heavy tail makes the sample variance itself very variable)
                if !matches!(d, Dist::StudentT { df } if df < 5.0) {
                    assert!(close(variance(&x), v, 0.12), "{d:?}: variance {} for {v}", variance(&x));
                }
            }
            // half the draws below the median, a tenth below the first decile
            for &p in &[0.1, 0.5, 0.9] {
                let q = d.quantile(p);
                let below = x.iter().filter(|v| **v <= q).count() as f64 / x.len() as f64;
                let expect = d.cdf(q);
                assert!((below - expect).abs() < 0.012, "{d:?} at {p}: {below} for {expect}");
            }
        }
    }

    #[test]
    fn what_has_no_mean_says_so_and_bad_parameters_give_no_number() {
        assert_eq!(Dist::Cauchy { centre: 0.0, scale: 1.0 }.mean(), None);
        assert_eq!(Dist::StudentT { df: 1.0 }.mean(), None);
        assert_eq!(Dist::StudentT { df: 2.0 }.variance(), None);
        assert_eq!(Dist::StudentT { df: 3.0 }.variance(), Some(3.0));
        assert_eq!(Dist::F { d1: 3.0, d2: 2.0 }.mean(), None);
        for d in [
            Dist::Normal { mean: 0.0, sd: 0.0 },
            Dist::Uniform { lo: 1.0, hi: 1.0 },
            Dist::Bernoulli { p: 1.5 },
            Dist::Gamma { shape: -1.0, rate: 1.0 },
            Dist::Poisson { mean: 0.0 },
        ] {
            assert!(!d.valid());
            assert!(d.pdf(0.5).is_nan() && d.cdf(0.5).is_nan() && d.quantile(0.5).is_nan() && d.mean().is_none());
            assert!(d.sample(&mut Rng::new(1)).is_nan());
        }
        // the ends of the ranges of p are distributions like any other
        let sure = Dist::Binomial { n: 5, p: 1.0 };
        assert_eq!((sure.pdf(5.0), sure.pdf(4.0), sure.cdf(4.0), sure.cdf(5.0), sure.quantile(0.5)), (1.0, 0.0, 0.0, 1.0, 5.0));
        let never = Dist::Binomial { n: 5, p: 0.0 };
        assert_eq!((never.pdf(0.0), never.cdf(0.0), never.quantile(0.5)), (1.0, 1.0, 0.0));
    }

    #[test]
    fn the_ends_of_the_line_and_large_and_small_parameters() {
        // limits at infinity, not NaN
        assert_eq!(Dist::STANDARD_NORMAL.cdf(f64::NEG_INFINITY), 0.0);
        assert_eq!(Dist::STANDARD_NORMAL.cdf(f64::INFINITY), 1.0);
        assert_eq!(normal_cdf(1e200), 1.0);
        assert_eq!(normal_cdf(-1e200), 0.0);
        assert_eq!(Dist::Gamma { shape: 3.0, rate: 1.0 }.pdf(f64::INFINITY), 0.0);
        assert_eq!(Dist::Poisson { mean: 3.0 }.pdf(f64::INFINITY), 0.0);
        for d in every() {
            let (lo, hi) = d.support();
            assert_eq!((d.cdf(lo - 1.0).max(0.0), d.cdf(hi)), (0.0, 1.0), "{d:?}");
            assert!(d.cdf(d.quantile(0.0)).is_finite() && d.pdf(f64::INFINITY) == 0.0 && d.pdf(f64::NEG_INFINITY) == 0.0, "{d:?}");
        }
        // large shapes: the sums run on until they have converged
        assert!(close(gamma_p(1e5, 1e5), 0.500_420_5, 1e-6), "{}", gamma_p(1e5, 1e5));
        assert!(close(gamma_p(1e6, 1e6), 0.500_132_98, 1e-6), "{}", gamma_p(1e6, 1e6));
        assert_eq!(Dist::Poisson { mean: 1e5 }.quantile(0.5), 100_000.0);
        // small quantiles are found, not lost in an absolute tolerance
        for d in [Dist::Gamma { shape: 0.1, rate: 1.0 }, Dist::Beta { a: 0.05, b: 0.05 }, Dist::ChiSquare { df: 0.05 }] {
            for &p in &[0.025, 0.05] {
                let x = d.quantile(p);
                assert!(x > 0.0 && close(d.cdf(x), p, 1e-8), "{d:?} at {p}: cdf({x:e}) = {}", d.cdf(x));
            }
        }
        // Student's t through its middle
        let t5 = Dist::StudentT { df: 5.0 };
        assert!(close(t5.cdf(1e-9) - 0.5, 1e-9 * t5.pdf(0.0), 1e-6));
        assert!(Dist::StudentT { df: 3.0 }.quantile(0.5).abs() < 1e-12);
        assert!(close(Dist::StudentT { df: 1e12 }.quantile(0.975), 1.959_963_984_540_054, 1e-9));
        // the geometric's quantile at its own steps
        let g = Dist::Geometric { p: 0.3 };
        for k in 1..40 {
            assert_eq!(g.quantile(g.cdf(k as f64)), k as f64);
        }
        // a draw is strictly inside 0 to 1 even for the largest number the generator gives
        assert!(((u64::MAX >> 12) as f64 + 0.5) / ((1u64 << 52) as f64) < 1.0);
    }

    #[test]
    fn summaries_of_a_sample() {
        let x = [2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0];
        assert_eq!(mean(&x), 5.0);
        assert_eq!(variance_n(&x), 4.0);
        assert!(close(variance(&x), 32.0 / 7.0, 1e-12));
        assert_eq!(median(&x), 4.5);
        assert_eq!(quantile_sorted(&sorted(&x), 0.0), 2.0);
        assert_eq!(quantile_sorted(&sorted(&x), 1.0), 9.0);
        assert_eq!(quantile_sorted(&sorted(&x), 0.25), 4.0);
        assert!(median(&[]).is_nan());
        assert_eq!(ranks(&[10.0, 30.0, 20.0, 30.0]), vec![1.0, 3.5, 2.0, 3.5]);
        assert!(close(correlation(&[1.0, 2.0, 3.0, 4.0], &[2.0, 4.0, 6.0, 8.5]), 0.998_7, 1e-3));
        assert_eq!(correlation(&[1.0, 1.0, 1.0], &[1.0, 2.0, 3.0]), 0.0);
        // a long tail to the right
        assert!(skewness(&[1.0, 1.0, 1.0, 2.0, 9.0]) > 1.0);
        assert!(skewness(&[1.0, 2.0, 3.0]).abs() < 1e-12);
        let mut rng = Rng::new(3);
        let z = Dist::STANDARD_NORMAL.draws(50_000, &mut rng);
        assert!(excess_kurtosis(&z).abs() < 0.08 && skewness(&z).abs() < 0.04);
        let h = Histogram::of(&[0.1, 0.2, 0.6, 0.95, 1.0, 1.7], 0.0, 1.0, 2);
        assert_eq!((h.counts.clone(), h.total), (vec![2, 3], 6));
        // areas: (2 + 3) of 6 numbers are inside
        assert!(close((h.density(0) + h.density(1)) * h.width(), 5.0 / 6.0, 1e-12));
    }
}

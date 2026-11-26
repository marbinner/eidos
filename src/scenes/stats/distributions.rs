//! A zoo of distributions: each one seen three ways, as its density (for a count, the
//! probability of each whole number), as its cumulative probability and as its quantile
//! function, one mark read off all three; what each is the distribution of, what its
//! parameters do, and the roads from one to another, measured by the greatest gap
//! between two cumulative curves.

use super::plot::{self, Glide, Plot, Rect};
use super::*;
use crate::clock::Pace;
use crate::math::stats::{self, Dist};

pub const DEF: SceneDef = SceneDef {
    id: "distributions",
    title: "A zoo of distributions",
    group: CHANCE,
    blurb: "The distributions statistics is built from: what their parameters do, density against cumulative probability against quantiles, and how one turns into another.",
    make: |_| Box::new(Zoo::create()),
};

// ----- the distributions -----------------------------------------------------------------

/// The fourteen distributions of `math::stats::Dist`, in the order of the buttons.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Kind {
    Normal,
    Uniform,
    Exponential,
    Gamma,
    Beta,
    ChiSquare,
    StudentT,
    F,
    Cauchy,
    LogNormal,
    Bernoulli,
    Binomial,
    Poisson,
    Geometric,
}

const KINDS: [Kind; 14] = [
    Kind::Normal,
    Kind::Uniform,
    Kind::Exponential,
    Kind::Gamma,
    Kind::Beta,
    Kind::ChiSquare,
    Kind::StudentT,
    Kind::F,
    Kind::Cauchy,
    Kind::LogNormal,
    Kind::Bernoulli,
    Kind::Binomial,
    Kind::Poisson,
    Kind::Geometric,
];

/// Where the ones that count begin in that order (their buttons are set apart).
const FIRST_COUNT: usize = 10;

/// A parameter: its name in the panel, its range, and where it starts.
struct Param {
    label: &'static str,
    lo: f64,
    hi: f64,
    log: bool,
    whole: bool,
    start: f64,
}

const fn param(label: &'static str, lo: f64, hi: f64, log: bool, whole: bool, start: f64) -> Param {
    Param { label, lo, hi, log, whole, start }
}

const NORMAL: [Param; 2] = [param(r"mean $\mu$", -5.0, 5.0, false, false, 0.0), param(r"standard deviation $\sigma$", 0.1, 5.0, true, false, 1.0)];
const UNIFORM: [Param; 2] = [param("from $a$", -5.0, 5.0, false, false, 0.0), param("to $b$", -5.0, 5.0, false, false, 1.0)];
const EXPONENTIAL: [Param; 1] = [param(r"rate $\lambda$", 0.1, 10.0, true, false, 1.0)];
const GAMMA: [Param; 2] = [param("shape $k$", 0.2, 30.0, true, false, 2.0), param(r"rate $\lambda$", 0.1, 10.0, true, false, 1.0)];
const BETA: [Param; 2] = [param("$a$", 0.2, 30.0, true, false, 2.0), param("$b$", 0.2, 30.0, true, false, 5.0)];
const CHI_SQUARE: [Param; 1] = [param("degrees of freedom $k$", 1.0, 80.0, true, true, 3.0)];
const STUDENT_T: [Param; 1] = [param("degrees of freedom $k$", 1.0, 100.0, true, true, 3.0)];
const F_RATIO: [Param; 2] = [param("degrees of freedom $d_1$", 1.0, 100.0, true, true, 5.0), param("degrees of freedom $d_2$", 1.0, 100.0, true, true, 10.0)];
const CAUCHY: [Param; 2] = [param("centre $x_0$", -5.0, 5.0, false, false, 0.0), param(r"scale $\gamma$", 0.1, 5.0, true, false, 1.0)];
const LOG_NORMAL: [Param; 2] =
    [param(r"$\mu$, the logarithm's mean", -2.0, 2.0, false, false, 0.0), param(r"$\sigma$, the logarithm's standard deviation", 0.1, 1.5, true, false, 0.6)];
const BERNOULLI: [Param; 1] = [param("chance of success $p$", 0.0, 1.0, false, false, 0.3)];
const BINOMIAL: [Param; 2] = [param("tries $n$", 1.0, 400.0, true, true, 10.0), param("chance of success $p$", 0.0, 1.0, false, false, 0.5)];
const POISSON: [Param; 1] = [param(r"mean $\lambda$", 0.1, 60.0, true, false, 4.0)];
const GEOMETRIC: [Param; 1] = [param("chance of success $p$", 0.02, 1.0, false, false, 0.3)];

impl Kind {
    fn index(self) -> usize {
        KINDS.iter().position(|k| *k == self).unwrap_or(0)
    }

    /// Does it count (put its probability on whole numbers)?
    fn counts(self) -> bool {
        self.index() >= FIRST_COUNT
    }

    /// Its name on its button, and in a sentence.
    fn name(self) -> &'static str {
        match self {
            Kind::Normal => "normal",
            Kind::Uniform => "uniform",
            Kind::Exponential => "exponential",
            Kind::Gamma => "gamma",
            Kind::Beta => "beta",
            Kind::ChiSquare => "chi-square",
            Kind::StudentT => r"Student's $t$",
            Kind::F => "$F$",
            Kind::Cauchy => "Cauchy",
            Kind::LogNormal => "log-normal",
            Kind::Bernoulli => "Bernoulli",
            Kind::Binomial => "binomial",
            Kind::Poisson => "Poisson",
            Kind::Geometric => "geometric",
        }
    }

    fn params(self) -> &'static [Param] {
        match self {
            Kind::Normal => &NORMAL,
            Kind::Uniform => &UNIFORM,
            Kind::Exponential => &EXPONENTIAL,
            Kind::Gamma => &GAMMA,
            Kind::Beta => &BETA,
            Kind::ChiSquare => &CHI_SQUARE,
            Kind::StudentT => &STUDENT_T,
            Kind::F => &F_RATIO,
            Kind::Cauchy => &CAUCHY,
            Kind::LogNormal => &LOG_NORMAL,
            Kind::Bernoulli => &BERNOULLI,
            Kind::Binomial => &BINOMIAL,
            Kind::Poisson => &POISSON,
            Kind::Geometric => &GEOMETRIC,
        }
    }

    /// The distribution with these parameters (a whole number where one is wanted).
    fn dist(self, p: [f64; 2]) -> Dist {
        // (NaN gives 1 here rather than a panic: `max` takes the number)
        let whole = |v: f64| v.round().max(1.0);
        let share = |v: f64| v.max(0.0).min(1.0);
        match self {
            Kind::Normal => Dist::Normal { mean: p[0], sd: p[1] },
            Kind::Uniform => Dist::Uniform { lo: p[0], hi: p[1] },
            Kind::Exponential => Dist::Exponential { rate: p[0] },
            Kind::Gamma => Dist::Gamma { shape: p[0], rate: p[1] },
            Kind::Beta => Dist::Beta { a: p[0], b: p[1] },
            Kind::ChiSquare => Dist::ChiSquare { df: whole(p[0]) },
            Kind::StudentT => Dist::StudentT { df: whole(p[0]) },
            Kind::F => Dist::F { d1: whole(p[0]), d2: whole(p[1]) },
            Kind::Cauchy => Dist::Cauchy { centre: p[0], scale: p[1] },
            Kind::LogNormal => Dist::LogNormal { mu: p[0], sigma: p[1] },
            Kind::Bernoulli => Dist::Bernoulli { p: share(p[0]) },
            Kind::Binomial => Dist::Binomial { n: whole(p[0]).min(400.0) as u32, p: share(p[1]) },
            Kind::Poisson => Dist::Poisson { mean: p[0] },
            Kind::Geometric => Dist::Geometric { p: p[0] },
        }
    }

    /// What it is the distribution of.
    fn story(self) -> &'static str {
        match self {
            Kind::Normal => "The bell curve. A sum of many small independent pieces, none of them dominant, is close to normal.",
            Kind::Uniform => "A number as likely to fall anywhere from $a$ to $b$ as anywhere else. From 0 to 1 it is the raw material the draws here are made from.",
            Kind::Exponential => r"The wait for the first event, when events come at random and independently, $\lambda$ of them per unit of time on average.",
            Kind::Gamma => r"The wait for the $k$-th such event, at rate $\lambda$: for a whole number $k$, the sum of $k$ independent exponential waits.",
            Kind::Beta => "A share between 0 and 1. For whole numbers $a$ and $b$: the $a$-th smallest of $a + b - 1$ independent uniform numbers from 0 to 1.",
            Kind::ChiSquare => "The sum of $k$ independent standard normal numbers, each squared.",
            Kind::StudentT => r"A standard normal number divided by $\sqrt{V/k}$, with $V$ an independent chi-square of $k$ degrees of freedom. For normal data, the distance of a sample's mean from the true mean, counted in its estimated standard errors, has it with $k = n - 1$.",
            Kind::F => r"The ratio $(U/d_1)/(V/d_2)$ of two independent chi-squares, each divided by its degrees of freedom: for normal data, the ratio of two independent estimates of the same variance.",
            Kind::Cauchy => r"The ratio of two independent standard normal numbers, stretched by $\gamma$ and moved to $x_0$: a bell with tails so heavy that it has no mean.",
            Kind::LogNormal => r"A number whose logarithm is normal. A product of many small independent positive factors is close to it, since its logarithm is a sum.",
            Kind::Bernoulli => "One try: 1 (a success) with probability $p$, else 0.",
            Kind::Binomial => "The number of successes in $n$ independent tries, each a success with probability $p$.",
            Kind::Poisson => r"The number of events in a stretch of time in which $\lambda$ are expected, events coming at random and independently.",
            Kind::Geometric => "The number of the try on which the first success comes, each try a success with probability $p$, independently.",
        }
    }

    /// The same, short enough for a phone's picture.
    fn short(self) -> &'static str {
        match self {
            Kind::Normal => "the bell curve",
            Kind::Uniform => "equally likely anywhere from $a$ to $b$",
            Kind::Exponential => "the wait for one random event",
            Kind::Gamma => "the wait for $k$ random events",
            Kind::Beta => "a share between 0 and 1",
            Kind::ChiSquare => "a sum of $k$ squared standard normals",
            Kind::StudentT => r"a normal over $\sqrt{\chi^2_k / k}$",
            Kind::F => "a ratio of two chi-squares",
            Kind::Cauchy => "a ratio of two normals: no mean",
            Kind::LogNormal => "its logarithm is normal",
            Kind::Bernoulli => "one try: 1 or 0",
            Kind::Binomial => "successes in $n$ tries",
            Kind::Poisson => "random events in a stretch of time",
            Kind::Geometric => "the try of the first success",
        }
    }

    /// Its parameters as a formula: `$n = 10$, $p = 0.5$`.
    fn settings(self, p: [f64; 2]) -> String {
        // (a small value with the digits it needs: 0.003 is not 0)
        let d = |v: f64| {
            if (v - v.round()).abs() < 1e-9 {
                num(v, 0)
            } else if v.abs() >= 10.0 {
                num(v, 1)
            } else if v.abs() >= 0.1 {
                trim(num(v, 2))
            } else {
                trim(num(v, (1 + (-v.abs().log10()).ceil() as usize).min(8)))
            }
        };
        let names: &[&str] = match self {
            Kind::Normal => &[r"\mu", r"\sigma"],
            Kind::Uniform => &["a", "b"],
            Kind::Exponential => &[r"\lambda"],
            Kind::Gamma => &["k", r"\lambda"],
            Kind::Beta => &["a", "b"],
            Kind::ChiSquare | Kind::StudentT => &["k"],
            Kind::F => &["d_1", "d_2"],
            Kind::Cauchy => &["x_0", r"\gamma"],
            Kind::LogNormal => &[r"\mu", r"\sigma"],
            Kind::Bernoulli | Kind::Geometric => &["p"],
            Kind::Binomial => &["n", "p"],
            Kind::Poisson => &[r"\lambda"],
        };
        let dist = self.dist(p);
        // (the numbers the distribution is made with: whole numbers where they are wanted)
        let value = |i: usize| match (dist, i) {
            (Dist::ChiSquare { df } | Dist::StudentT { df }, _) => df,
            (Dist::F { d1, .. }, 0) => d1,
            (Dist::F { d2, .. }, _) => d2,
            (Dist::Binomial { n, .. }, 0) => n as f64,
            (Dist::Binomial { p, .. }, _) => p,
            _ => p[i],
        };
        names.iter().enumerate().map(|(i, n)| format!("${n} = {}$", d(value(i)))).collect::<Vec<_>>().join(", ")
    }
}

/// A number without the zeros at its end: 0.50 is 0.5.
fn trim(s: String) -> String {
    if s.contains('.') {
        s.trim_end_matches('0').trim_end_matches('.').to_string()
    } else {
        s
    }
}

// ----- the roads between them -------------------------------------------------------------

/// A second distribution to lay over the first: where it turns into another, or is
/// another under a second name.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Road {
    NormalSame,
    PoissonSame,
    StandardNormal,
    GammaSame,
    CauchyQuartiles,
    NormalQuartiles,
    CauchyStandard,
    GammaShapeOne,
    UniformUnit,
}

impl Road {
    fn label(self) -> &'static str {
        match self {
            Road::NormalSame => "normal, same mean and sd",
            Road::PoissonSame => "Poisson, same mean",
            Road::StandardNormal => "standard normal",
            Road::GammaSame => r"gamma, shape $k/2$, rate $1/2$",
            Road::CauchyQuartiles => "Cauchy, same quartiles",
            Road::NormalQuartiles => "normal, same quartiles",
            Road::CauchyStandard => "Cauchy, centre 0, scale 1",
            Road::GammaShapeOne => "gamma, shape 1, same rate",
            Road::UniformUnit => "uniform from 0 to 1",
        }
    }

    /// The same in a sentence.
    fn noun(self) -> &'static str {
        match self {
            Road::NormalSame => "normal with the same mean and standard deviation",
            Road::PoissonSame => "Poisson with the same mean",
            Road::StandardNormal => "standard normal",
            Road::GammaSame => r"gamma with shape $k/2$ and rate $1/2$",
            Road::CauchyQuartiles => "Cauchy with the same quartiles",
            Road::NormalQuartiles => "normal with the same quartiles",
            Road::CauchyStandard => "Cauchy with centre 0 and scale 1",
            Road::GammaShapeOne => "gamma with shape 1 and the same rate",
            Road::UniformUnit => "uniform from 0 to 1",
        }
    }

    /// The second distribution for the first, or why there is none.
    fn of(self, d: &Dist) -> Result<Dist, &'static str> {
        if !d.valid() {
            return Err("no distribution to compare");
        }
        match (self, *d) {
            (Road::NormalSame, _) => match (d.mean(), d.sd()) {
                (Some(mean), Some(sd)) if sd > 0.0 => Ok(Dist::Normal { mean, sd }),
                (Some(_), Some(_)) => Err("no spread to match: all its probability is on one value"),
                _ => Err("no finite standard deviation to match"),
            },
            (Road::PoissonSame, _) => match d.mean() {
                Some(mean) if mean > 0.0 => Ok(Dist::Poisson { mean }),
                _ => Err("mean 0: no Poisson distribution has it"),
            },
            (Road::StandardNormal, _) => Ok(Dist::STANDARD_NORMAL),
            (Road::GammaSame, Dist::ChiSquare { df }) => Ok(Dist::Gamma { shape: df / 2.0, rate: 0.5 }),
            (Road::CauchyQuartiles, Dist::Normal { mean, sd }) => Ok(Dist::Cauchy { centre: mean, scale: sd * stats::normal_quantile(0.75) }),
            (Road::NormalQuartiles, Dist::Cauchy { centre, scale }) => Ok(Dist::Normal { mean: centre, sd: scale / stats::normal_quantile(0.75) }),
            (Road::CauchyStandard, _) => Ok(Dist::Cauchy { centre: 0.0, scale: 1.0 }),
            (Road::GammaShapeOne, Dist::Exponential { rate }) => Ok(Dist::Gamma { shape: 1.0, rate }),
            (Road::UniformUnit, _) => Ok(Dist::Uniform { lo: 0.0, hi: 1.0 }),
            _ => Err("no such road from here"),
        }
    }
}

/// The roads from each kind.
fn roads(kind: Kind) -> &'static [Road] {
    match kind {
        Kind::Normal => &[Road::CauchyQuartiles],
        Kind::Uniform | Kind::Gamma | Kind::F | Kind::LogNormal | Kind::Bernoulli | Kind::Poisson | Kind::Geometric => &[Road::NormalSame],
        Kind::Exponential => &[Road::NormalSame, Road::GammaShapeOne],
        Kind::Beta => &[Road::NormalSame, Road::UniformUnit],
        Kind::ChiSquare => &[Road::NormalSame, Road::GammaSame],
        Kind::StudentT => &[Road::StandardNormal, Road::CauchyStandard],
        Kind::Cauchy => &[Road::NormalQuartiles],
        Kind::Binomial => &[Road::PoissonSame, Road::NormalSame],
    }
}

/// The greatest distance between two cumulative curves over every x, where it is, and
/// the two curves' heights there (at a count's step, the height just before it where
/// that is where the distance is greatest).
#[derive(Clone, Copy, Debug, PartialEq)]
struct Gap {
    size: f64,
    at: f64,
    a: f64,
    b: f64,
}

/// Less than this is no gap: the same distribution under two names.
const NO_GAP: f64 = 1e-9;

fn greatest_gap(a: &Dist, b: &Dist) -> Option<Gap> {
    if !(a.valid() && b.valid()) {
        return None;
    }
    let mut best: Option<Gap> = None;
    let mut consider = |at: f64, fa: f64, fb: f64| {
        let size = (fa - fb).abs();
        if size.is_finite() && best.is_none_or(|g| size > g.size) {
            best = Some(Gap { size, at, a: fa, b: fb });
        }
    };
    match (a.discrete(), b.discrete()) {
        // both step at whole numbers and are flat between: the whole numbers are enough
        (true, true) => {
            let lo = a.quantile(1e-12).min(b.quantile(1e-12)).floor();
            let hi = a.quantile(1.0 - 1e-12).max(b.quantile(1.0 - 1e-12)).ceil().min(lo + 20_000.0);
            let mut k = lo;
            while k <= hi {
                consider(k, a.cdf(k), b.cdf(k));
                k += 1.0;
            }
        }
        // from one whole number to the next the staircase is flat and the curve rises,
        // so the distance is greatest at one end or the other (the far end approached
        // from the left)
        (true, false) | (false, true) => {
            let first = a.discrete();
            let (d, c) = if first { (a, b) } else { (b, a) };
            let lo = d.quantile(1e-12).min(c.quantile(1e-9).floor()) - 1.0;
            let hi = d.quantile(1.0 - 1e-12).max(c.quantile(1.0 - 1e-9).ceil()).min(lo + 20_000.0);
            let mut k = lo;
            while k <= hi {
                let fd = d.cdf(k);
                for x in [k, k + 1.0] {
                    let fc = c.cdf(x);
                    if first {
                        consider(x, fd, fc);
                    } else {
                        consider(x, fc, fd);
                    }
                }
                k += 1.0;
            }
        }
        // two smooth curves: a fine search over where either has its middle, then the
        // greatest point near the best found
        (false, false) => {
            let spread = |d: &Dist| {
                let (q1, m, q3) = (d.quantile(0.25), d.quantile(0.5), d.quantile(0.75));
                (m, (q3 - q1).max(1e-9))
            };
            let ((ma, sa), (mb, sb)) = (spread(a), spread(b));
            let lo = (ma - 8.0 * sa).min(mb - 8.0 * sb);
            let hi = (ma + 8.0 * sa).max(mb + 8.0 * sb);
            let n = 1200;
            let diff = |x: f64| (a.cdf(x) - b.cdf(x)).abs();
            let (mut at, mut most) = (lo, -1.0);
            for i in 0..=n {
                let x = lo + (hi - lo) * i as f64 / n as f64;
                if diff(x) > most {
                    (at, most) = (x, diff(x));
                }
            }
            // (and out in the tails, where a heavy tail may part from a light one)
            for p in [1e-4, 1e-3, 1e-2, 0.99, 0.999, 0.9999] {
                for x in [a.quantile(p), b.quantile(p)] {
                    if x.is_finite() && diff(x) > most {
                        (at, most) = (x, diff(x));
                    }
                }
            }
            let step = (hi - lo) / n as f64;
            let (mut l, mut r) = (at - step, at + step);
            let golden = 0.618_033_988_749_895;
            for _ in 0..60 {
                let (x1, x2) = (r - golden * (r - l), l + golden * (r - l));
                if diff(x1) >= diff(x2) {
                    r = x2;
                } else {
                    l = x1;
                }
            }
            for x in [at, 0.5 * (l + r)] {
                consider(x, a.cdf(x), b.cdf(x));
            }
        }
    }
    best
}

/// The quantile, `Dist::quantile` made exact for a count: the least whole number whose
/// cumulative probability reaches p. (For the geometric, `Dist::quantile` works it out
/// with logarithms, and at p = F(k) exactly it can round up to k + 1: with p = 0.3, the
/// quantile of F(11) came out as 12. A step or two along the staircase puts it right.)
fn quantile(d: &Dist, p: f64) -> f64 {
    let q = d.quantile(p);
    if !(d.discrete() && q.is_finite() && p > 0.0 && p < 1.0) {
        return q;
    }
    let (lo, hi) = d.support();
    let mut q = q;
    for _ in 0..4 {
        if q > lo && d.cdf(q - 1.0) >= p {
            q -= 1.0;
        } else if q < hi && d.cdf(q) < p {
            q += 1.0;
        } else {
            break;
        }
    }
    q
}

// ----- the charts' ranges -----------------------------------------------------------------

/// The values worth showing of a distribution: its middle 99 % (99.9 % of a count), no
/// further than four times the distance between its quartiles from its median (a heavy
/// tail would leave the middle a sliver), and out to an end of its support that is near.
/// A share (the beta) and a binomial of a few tries are shown whole.
fn core(d: &Dist) -> Option<(f64, f64)> {
    if !d.valid() {
        return None;
    }
    match *d {
        Dist::Binomial { n, .. } if n <= 40 => return Some((0.0, n as f64)),
        Dist::Beta { .. } => return Some((0.0, 1.0)),
        _ => {}
    }
    let (lo, hi) = d.support();
    let (a, b) = if d.discrete() {
        (d.quantile(0.0005), d.quantile(0.9995))
    } else {
        let (q1, m, q3) = (d.quantile(0.25), d.quantile(0.5), d.quantile(0.75));
        let iqr = q3 - q1;
        (d.quantile(0.005).max(m - 4.0 * iqr), d.quantile(0.995).min(m + 4.0 * iqr))
    };
    let w = (b - a).max(1e-9);
    let a = if lo.is_finite() && a - lo <= 0.3 * w { lo } else { a };
    let b = if hi.is_finite() && hi - b <= 0.3 * w { hi } else { b };
    (a.is_finite() && b.is_finite() && b >= a).then_some((a, b))
}

/// The x range of the charts for these cores: a little room either side, half a unit
/// round the bars of a count, and never narrower than five whole numbers for a count.
fn x_range((a, b): (f64, f64), counts: bool) -> (f64, f64) {
    if counts {
        let (a, b) = (a - 0.5, b + 0.5);
        let pad = (0.04 * (b - a)).max(0.5);
        let (a, b) = (a - pad, b + pad);
        let short = (5.0 - (b - a)).max(0.0) * 0.5;
        (a - short, b + short)
    } else {
        let pad = 0.07 * (b - a).max(1e-6);
        (a - pad, b + pad)
    }
}

/// The highest a density (or the probability of a whole number) goes on [x0, x1]. Where
/// it rises without bound at an end of its support, no more than four times its least
/// height at the quartiles and the median: the middle stays in view, and the spike runs
/// off the top of the chart.
fn peak(d: &Dist, x0: f64, x1: f64) -> f64 {
    if !d.valid() {
        return 0.0;
    }
    if d.discrete() {
        let (lo, hi) = d.support();
        let (k0, k1) = (x0.ceil().max(lo), x1.floor().min(hi).min(x0.ceil() + 4000.0));
        let mut most = 0.0f64;
        let mut k = k0;
        while k <= k1 {
            most = most.max(d.pdf(k));
            k += 1.0;
        }
        return most;
    }
    let (lo, hi) = d.support();
    let unbounded = [lo, hi].iter().any(|e| e.is_finite() && d.pdf(*e).is_infinite());
    let cap = if unbounded { 4.0 * [0.25, 0.5, 0.75].iter().map(|p| d.pdf(d.quantile(*p))).fold(f64::INFINITY, f64::min) } else { f64::INFINITY };
    let most = (0..=300).map(|i| d.pdf(x0 + (x1 - x0) * i as f64 / 300.0)).filter(|v| v.is_finite()).fold(0.0, f64::max);
    if cap.is_finite() && cap > 0.0 {
        most.min(cap)
    } else {
        most
    }
}

// ----- what is drawn of a distribution ----------------------------------------------------

/// Columns across a chart at which a smooth curve is drawn.
const COLUMNS: usize = 241;

/// A distribution as drawn over a range of x.
#[derive(Clone, Debug, PartialEq)]
enum Shape {
    /// The density and the cumulative probability at columns evenly across the range.
    Smooth { xs: Vec<f64>, pdf: Vec<f64>, cdf: Vec<f64> },
    /// The probabilities of the whole numbers from `first` on and their cumulative ones,
    /// with the cumulative probability below `first`.
    Counts { first: f64, pmf: Vec<f64>, cdf: Vec<f64>, below: f64 },
}

fn shape(d: &Dist, x0: f64, x1: f64) -> Shape {
    if d.discrete() {
        let (lo, hi) = d.support();
        let first = x0.ceil().max(lo);
        let last = x1.floor().min(hi).min(first + 4000.0);
        let below = if first.is_finite() { d.cdf(first - 1.0) } else { f64::NAN };
        let (mut pmf, mut cdf) = (Vec::new(), Vec::new());
        let mut k = first;
        let mut total = below;
        while k <= last {
            let p = d.pdf(k);
            total += p;
            pmf.push(p);
            cdf.push(total.min(1.0));
            k += 1.0;
        }
        Shape::Counts { first, pmf, cdf, below }
    } else {
        let xs: Vec<f64> = (0..COLUMNS).map(|i| x0 + (x1 - x0) * i as f64 / (COLUMNS - 1) as f64).collect();
        // (where the density has no bound, a height far above the chart, which the chart
        // cuts at its top edge)
        let pdf = xs.iter().map(|x| d.pdf(*x)).map(|v| if v.is_infinite() { 1e9 } else { v }).collect();
        let cdf = xs.iter().map(|x| d.cdf(*x)).collect();
        Shape::Smooth { xs, pdf, cdf }
    }
}

impl Shape {
    /// The cumulative curve as points (x, F(x)) from x0 to x1: a staircase for a count.
    fn cumulative(&self, x0: f64, x1: f64) -> Vec<(f64, f64)> {
        match self {
            Shape::Smooth { xs, cdf, .. } => xs.iter().copied().zip(cdf.iter().copied()).collect(),
            Shape::Counts { first, cdf, below, .. } => {
                let mut pts = Vec::with_capacity(2 * cdf.len() + 2);
                pts.push((x0, *below));
                let mut height = *below;
                for (i, c) in cdf.iter().enumerate() {
                    let k = first + i as f64;
                    pts.push((k, height));
                    pts.push((k, *c));
                    height = *c;
                }
                pts.push((x1, height));
                pts
            }
        }
    }
}

// ----- draws ------------------------------------------------------------------------------

/// How the draws are made.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Way {
    /// Uniform random numbers from 0 to 1 pushed through the quantile curve.
    Random,
    /// Evenly spread numbers from 0 to 1, each new one in the middle of one of the widest
    /// gaps left, pushed through the quantile curve.
    Even,
    /// As the distribution's story says: sums of squares, waits, tries.
    Story,
}

const WAYS: [Way; 3] = [Way::Random, Way::Even, Way::Story];
const HOW_MANY: [usize; 4] = [0, 100, 1000, 10_000];

/// The `i`th evenly spread number from 0 to 1 (counting from 0): 1/2, 1/4, 3/4, 1/8,
/// 5/8, ... (the binary digits of i + 1 read backwards after the point).
fn spread_number(i: usize) -> f64 {
    let (mut k, mut place, mut v) = (i as u64 + 1, 0.5, 0.0);
    while k > 0 {
        if k & 1 == 1 {
            v += place;
        }
        place *= 0.5;
        k >>= 1;
    }
    v
}

/// Is there a story to make draws of this distribution from, exactly? (The ones whose
/// story is the quantile curve itself have none, nor does a gamma or a beta whose story
/// needs a whole number where its parameter is not one.)
fn story_fits(d: &Dist) -> bool {
    let whole = |v: f64| (v - v.round()).abs() < 1e-9 && (1.0..=60.0).contains(&v);
    d.valid()
        && match *d {
            Dist::Normal { .. } | Dist::Uniform { .. } | Dist::Exponential { .. } => false,
            Dist::Gamma { shape, .. } => whole(shape),
            Dist::Beta { a, b } => whole(a) && whole(b),
            _ => true,
        }
}

fn has_story(kind: Kind, p: [f64; 2]) -> bool {
    story_fits(&kind.dist(p))
}

/// What each draw is, made by the story: for the line on the picture.
fn story_formula(kind: Kind, d: &Dist) -> String {
    let squares = |k: f64| match k as u32 {
        1 => "Z_1^2".to_string(),
        2 => "Z_1^2 + Z_2^2".to_string(),
        3 => "Z_1^2 + Z_2^2 + Z_3^2".to_string(),
        k => format!(r"Z_1^2 + \cdots + Z_{{{k}}}^2"),
    };
    match (kind, *d) {
        (_, Dist::Gamma { shape, .. }) => format!(r"each draw: the sum of {} exponential waits", num(shape, 0)),
        (_, Dist::Beta { a, b }) => format!("each draw: the {} smallest of {} uniform numbers", ordinal(a as u32), num(a + b - 1.0, 0)),
        (_, Dist::ChiSquare { df }) => format!("each draw: ${}$, with $Z_i$ standard normal", squares(df)),
        (_, Dist::StudentT { df }) => format!(r"each draw: $Z/\sqrt{{V/{}}}$, $V$ a sum of {} squared normals", num(df, 0), num(df, 0)),
        (_, Dist::F { .. }) => "each draw: a ratio of two sums of squared normals, each over its count".to_string(),
        (_, Dist::Cauchy { .. }) => r"each draw: $x_0 + \gamma\,Z_1/Z_2$".to_string(),
        (_, Dist::LogNormal { .. }) => r"each draw: $e^{\mu + \sigma Z}$".to_string(),
        (_, Dist::Bernoulli { .. }) => "each draw: one try".to_string(),
        (_, Dist::Binomial { n, .. }) => format!("each draw: the successes in {n} tries"),
        (_, Dist::Poisson { .. }) => "each draw: the events in one unit of time".to_string(),
        (_, Dist::Geometric { .. }) => "each draw: the try of the first success".to_string(),
        _ => String::new(),
    }
}

fn ordinal(n: u32) -> String {
    let tail = match (n % 10, n % 100) {
        (1, x) if x != 11 => "st",
        (2, x) if x != 12 => "nd",
        (3, x) if x != 13 => "rd",
        _ => "th",
    };
    format!("{n}{tail}")
}

/// A sum of `k` squared standard normal numbers.
fn squares(k: u32, rng: &mut Rng) -> f64 {
    (0..k).map(|_| stats::normal(rng).powi(2)).sum()
}

/// One draw made as the distribution's story says (None where it has no story).
fn story_draw(d: &Dist, rng: &mut Rng) -> Option<f64> {
    let wait = |rate: f64, rng: &mut Rng| -stats::unit(rng).ln() / rate;
    Some(match *d {
        Dist::Gamma { shape, rate } => (0..shape.round() as u32).map(|_| wait(rate, rng)).sum(),
        Dist::Beta { a, b } => {
            let mut u: Vec<f64> = (0..(a + b - 1.0).round() as usize).map(|_| stats::unit(rng)).collect();
            u.sort_by(f64::total_cmp);
            *u.get((a.round() as usize).saturating_sub(1))?
        }
        Dist::ChiSquare { df } => squares(df as u32, rng),
        Dist::StudentT { df } => stats::normal(rng) / (squares(df as u32, rng) / df).sqrt().max(1e-300),
        Dist::F { d1, d2 } => (squares(d1 as u32, rng) / d1) / (squares(d2 as u32, rng) / d2).max(1e-300),
        Dist::Cauchy { centre, scale } => {
            let (z1, z2) = (stats::normal(rng), stats::normal(rng));
            centre + scale * z1 / if z2 == 0.0 { 1e-300 } else { z2 }
        }
        Dist::LogNormal { mu, sigma } => (mu + sigma * stats::normal(rng)).exp(),
        Dist::Bernoulli { p } => (stats::unit(rng) < p) as u8 as f64,
        Dist::Binomial { n, p } => (0..n).filter(|_| stats::unit(rng) < p).count() as f64,
        Dist::Poisson { mean } => {
            // events at rate `mean` per unit of time, counted until their waits pass one unit
            let (mut t, mut k) = (wait(mean, rng), 0.0);
            while t <= 1.0 && k < 10_000.0 {
                k += 1.0;
                t += wait(mean, rng);
            }
            k
        }
        Dist::Geometric { p } => {
            let mut k = 1.0;
            while stats::unit(rng) >= p && k < 1e6 {
                k += 1.0;
            }
            k
        }
        _ => return None,
    })
}

/// The bins of the draws' histogram over a range: one per whole number for a count.
#[derive(Clone, Debug, Default, PartialEq)]
struct Bins {
    lo: f64,
    width: f64,
    counts: Vec<u32>,
}

impl Bins {
    fn over(x0: f64, x1: f64, counts: bool) -> Bins {
        if counts {
            let (k0, k1) = (x0.ceil(), x1.floor());
            let n = (k1 - k0 + 1.0).max(1.0).min(4000.0) as usize;
            Bins { lo: k0 - 0.5, width: 1.0, counts: vec![0; n] }
        } else {
            Bins { lo: x0, width: ((x1 - x0) / 40.0).max(1e-9), counts: vec![0; 40] }
        }
    }

    fn add(&mut self, x: f64) {
        let i = ((x - self.lo) / self.width).floor();
        if i >= 0.0 && (i as usize) < self.counts.len() {
            self.counts[i as usize] += 1;
        }
    }
}

/// How long a draw takes from being made to landing in the histogram (seconds).
const FLIGHT: f32 = 0.6;

/// The draws: made at a rate, each in flight for a moment, then counted in the histogram.
struct Draws {
    want: usize,
    way: Way,
    seed: u64,
    rng: Rng,
    /// Each draw and the uniform number it was made from (NaN for a story's).
    made: Vec<(f64, f64)>,
    /// Scene seconds since the draws began; how many have landed.
    clock: f32,
    landed: usize,
    pace: Pace,
    bins: Bins,
    /// What they were made for (kind, parameters, way, how many, seed).
    key: Option<(Kind, [u64; 2], Way, usize, u64)>,
    /// Their median once all have landed, and how many that was of.
    median: Option<(usize, f64)>,
    heights: Glide,
}

impl Draws {
    fn new() -> Draws {
        Draws {
            want: 0,
            way: Way::Random,
            seed: 1,
            rng: Rng::new(1),
            made: Vec::new(),
            clock: 0.0,
            landed: 0,
            pace: Pace::default(),
            bins: Bins::default(),
            key: None,
            median: None,
            heights: Glide::default(),
        }
    }

    /// Draws a second (they all come in about four seconds).
    fn rate(&self) -> f32 {
        (self.want as f32 / 4.0).max(25.0).min(1500.0)
    }

    fn born(&self, i: usize) -> f32 {
        i as f32 / self.rate()
    }

    /// Still making them, or some still in flight.
    fn busy(&self) -> bool {
        self.made.len() < self.want || self.landed < self.made.len()
    }

    fn restart(&mut self) {
        self.made.clear();
        self.clock = 0.0;
        self.landed = 0;
        self.pace.reset();
        self.rng = Rng::new(self.seed);
        self.bins.counts.iter_mut().for_each(|c| *c = 0);
    }

    /// How many draws have landed, counting the one landing now in part: each lands
    /// over its own share of time, 1/rate, so that the histogram changes in every frame
    /// rather than by whole draws as they happen to fall into frames.
    fn landing(&self) -> f64 {
        ((self.clock - FLIGHT) as f64 * self.rate() as f64).max(0.0).min(self.made.len() as f64)
    }

    /// The histogram's heights as densities: each bin's share of the draws landed so far
    /// (the one landing now in part), over its width.
    fn densities(&self) -> Vec<f32> {
        let mut counts: Vec<f64> = self.bins.counts.iter().map(|c| *c as f64).collect();
        let part = self.landing() - self.landed as f64;
        let mut total = self.landed as f64;
        if part > 0.0 && part < 1.0 {
            if let Some(&(x, _)) = self.made.get(self.landed) {
                let i = ((x - self.bins.lo) / self.bins.width).floor();
                if i >= 0.0 && (i as usize) < counts.len() {
                    counts[i as usize] += part;
                }
                total += part;
            }
        }
        if total <= 0.0 {
            return vec![0.0; counts.len()];
        }
        counts.iter().map(|c| (c / total / self.bins.width) as f32).collect()
    }

    /// Make one draw of `d`. (By its story only where the story is exactly right; else,
    /// as the picture then says, through the quantile curve.)
    fn make(&mut self, d: &Dist) {
        let i = self.made.len();
        let story = if self.way == Way::Story && story_fits(d) { story_draw(d, &mut self.rng) } else { None };
        let (x, u) = match self.way {
            Way::Story => match story {
                Some(x) => (x, f64::NAN),
                None => {
                    let u = stats::unit(&mut self.rng);
                    (quantile(d, u), u)
                }
            },
            Way::Even => {
                let u = spread_number(i);
                (quantile(d, u), u)
            }
            Way::Random => {
                let u = stats::unit(&mut self.rng);
                (quantile(d, u), u)
            }
        };
        self.made.push((x, u));
    }
}

// ----- experiments ------------------------------------------------------------------------

/// A slow change of one parameter, for an experiment: held at `from` for `wait` seconds,
/// then eased to `to` over `length` (scene time).
#[derive(Clone, Copy, Debug, PartialEq)]
struct Sweep {
    param: usize,
    from: f64,
    to: f64,
    wait: f32,
    length: f32,
    t: f32,
}

impl Sweep {
    fn value(&self) -> f64 {
        if self.done() {
            return self.to;
        }
        let s = ((self.t - self.wait) / self.length).max(0.0).min(1.0) as f64;
        let eased = s * s * (3.0 - 2.0 * s);
        self.from + (self.to - self.from) * eased
    }

    fn done(&self) -> bool {
        self.t >= self.wait + self.length
    }
}

struct Trial {
    button: &'static str,
    caption: &'static str,
    kind: Kind,
    params: [f64; 2],
    road: Option<Road>,
    marks: [f64; 2],
    two: bool,
    draws: usize,
    way: Way,
    sweep: Option<(usize, f64)>,
}

const TRIALS: [Trial; 6] = [
    Trial {
        button: "counting successes",
        caption: r"Ten tries, each a success with probability $p$. At $p = 0.5$ the bars stand evenly round 5; as $p$ falls to 0.1 they crowd against 0 and lean, with a tail to the right, and the mean $np$ falls from 5 to 1. The share up to 2 (shaded) grows from 0.055 to 0.930.",
        kind: Kind::Binomial,
        params: [10.0, 0.5],
        road: None,
        marks: [2.5, 6.5],
        two: false,
        draws: 0,
        way: Way::Random,
        sweep: Some((1, 0.1)),
    },
    Trial {
        button: "rare events",
        caption: r"200 tries, each with chance 0.02, against the Poisson with the same mean, 4 (orange): their cumulative curves are never more than 0.0030 apart. Set $n = 10$ and $p = 0.4$, the same mean, and the gap grows to 0.071: a count is close to Poisson when the tries are many and each is unlikely.",
        kind: Kind::Binomial,
        params: [200.0, 0.02],
        road: Some(Road::PoissonSame),
        marks: [4.5, 6.5],
        two: false,
        draws: 0,
        way: Way::Random,
        sweep: None,
    },
    Trial {
        button: "many tries make a bell",
        caption: r"100 tries at 0.5 against the normal with the same mean and standard deviation, 50 and 5: the greatest gap (pink) is 0.040, exactly half the tallest bar, and no smooth curve can come closer than that to a staircase. Set $p = 0.02$: the bars lean against 0, the normal puts 0.077 of its probability below 0, and the gap is 0.18.",
        kind: Kind::Binomial,
        params: [100.0, 0.5],
        road: Some(Road::NormalSame),
        marks: [54.5, 60.5],
        two: false,
        draws: 0,
        way: Way::Random,
        sweep: None,
    },
    Trial {
        button: "heavy tails",
        caption: r"Student's $t$ with 3 degrees of freedom against the standard normal: alike in the middle, far apart in the tails. Beyond the marks at $\pm 3$ lies 0.058 of the $t$ and 0.0027 of the normal, 21 times less. Set 30 degrees of freedom and the greatest gap falls from 0.049 to 0.0052.",
        kind: Kind::StudentT,
        params: [3.0, 0.0],
        road: Some(Road::StandardNormal),
        marks: [-3.0, 3.0],
        two: true,
        draws: 0,
        way: Way::Random,
        sweep: None,
    },
    Trial {
        button: "sums of squares",
        caption: r"Each draw is $Z_1^2 + Z_2^2 + Z_3^2$, three standard normal numbers squared and added, and they pile up under the chi-square curve with 3 degrees of freedom. The same curve is the gamma with shape $3/2$ and rate $1/2$ (orange, no gap at all). The exponential is the wait for one random event and the gamma with whole shape $k$ the wait for $k$ of them; the chi-square with 2 degrees of freedom is the exponential with rate $1/2$.",
        kind: Kind::ChiSquare,
        params: [3.0, 0.0],
        road: Some(Road::GammaSame),
        marks: [7.81, 9.0],
        two: false,
        draws: 1000,
        way: Way::Story,
        sweep: None,
    },
    Trial {
        button: "any distribution from a uniform one",
        caption: r"1000 evenly spread numbers from 0 to 1, each new one in the middle of one of the widest gaps left, pushed through the quantile curve: they land crowded where that curve is flat and sparse where it is steep, which is the shape of the density (the density at $Q(p)$ is 1 over the curve's slope there). Random numbers from 0 to 1 give the same shape, more raggedly.",
        kind: Kind::LogNormal,
        params: [0.0, 0.6],
        road: None,
        marks: [1.0, 2.0],
        two: false,
        draws: 1000,
        way: Way::Even,
        sweep: None,
    },
];

/// The opening: the standard normal with its mark one standard deviation above the mean.
const OPENING_MARKS: [f64; 2] = [1.0, -1.0];

// ----- the scene --------------------------------------------------------------------------

/// The colours of what is not in the course's own key: the mark and the probability it
/// reads, the same number in all three charts.
const MARK: Color = pal::WHITE;
const AREA: Color = pal::BLUE;

/// Line kinds of this scene's own (see `plot`): staircases, whose number of points
/// follows the whole numbers in range; the histogram's outline; a count's outlines.
const STAIRS: u8 = plot::OWN;
const HISTOGRAM: u8 = plot::OWN + 1;
const COMBS: u8 = plot::OWN + 2;

/// The top of the quantile chart's $p$ axis: a hair over 1, so that the tick at 1 is
/// counted in however the tick spacing rounds.
const P_TOP: f64 = 1.000_001;

/// Handles.
const MARK_HANDLE: u32 = 1;
const P_HANDLE: u32 = 3;
const PARAM_HANDLE: u32 = 10;

/// What the charts' range was aimed for: the kind, its parameters and the road.
type Aim = (Kind, [u64; 2], usize);

/// The curves as drawn, and what for: the kind, its parameters and the range as drawn
/// (the bits of each), and the road.
type Drawn = (Kind, [u32; 2], [u32; 2], usize, Shape, Option<Shape>);

pub struct Zoo {
    kind: Kind,
    /// The parameters of every kind, kept when another is chosen.
    params: [[f64; 2]; 14],
    /// The road taken (0: none; else one more than its place in `roads(kind)`).
    road: usize,
    /// The marks' places, and whether the second is in use.
    marks: [f64; 2],
    two: bool,
    /// On a phone, the chart drawn largest.
    tab: usize,
    sweep: Option<Sweep>,
    /// The charts' x range and the density's height, as aimed for, and what for.
    range: [f64; 3],
    aimed: Option<Aim>,
    /// What is drawn, gliding to what is set: the parameters, the range, the marks and
    /// their cumulative probabilities.
    shown_params: Glide,
    shown_range: Glide,
    shown_marks: Glide,
    /// A handle is held: what it moves follows the hand exactly, and the range stays.
    held: bool,
    /// Another kind has just been chosen: the picture changes at once (no road leads
    /// from one kind to another, so there is nothing to glide through).
    cut: bool,
    moving: bool,
    gap: Option<(Aim, Option<Gap>)>,
    /// The curves as drawn last, and what for.
    drawn: Option<Drawn>,
    draws: Draws,
    /// Lines of wrapped text, kept while the text is the same; and the widths of words
    /// (by word and size), so that a sentence whose numbers change while a mark is
    /// dragged costs only its new words.
    kept_lines: Vec<(String, u32, u32, Vec<String>)>,
    widths: std::collections::HashMap<(String, u32), f32>,
    /// The round values on the axes last frame, and how many axis points and labels
    /// were drawn (see `Zoo::part_ticks`).
    ticks: Vec<f64>,
    drawn_last: (usize, usize),
}

impl Zoo {
    fn create() -> Zoo {
        let mut params = [[0.0; 2]; 14];
        for (k, kind) in KINDS.iter().enumerate() {
            for (i, p) in kind.params().iter().enumerate() {
                params[k][i] = p.start;
            }
        }
        let mut z = Zoo {
            kind: Kind::Normal,
            params,
            road: 0,
            marks: OPENING_MARKS,
            two: false,
            tab: 0,
            sweep: None,
            range: [-3.0, 3.0, 0.45],
            aimed: None,
            shown_params: Glide::default(),
            shown_range: Glide::default(),
            shown_marks: Glide::default(),
            held: false,
            cut: true,
            moving: false,
            gap: None,
            drawn: None,
            draws: Draws::new(),
            kept_lines: Vec::new(),
            widths: std::collections::HashMap::new(),
            ticks: Vec::new(),
            drawn_last: (0, 0),
        };
        z.step(0.0, 0.0);
        z
    }

    fn p(&self) -> [f64; 2] {
        self.params[self.kind.index()]
    }

    /// The distribution as set.
    fn dist(&self) -> Dist {
        self.kind.dist(self.p())
    }

    /// The road taken, if any.
    fn road(&self) -> Option<Road> {
        self.road.checked_sub(1).and_then(|i| roads(self.kind).get(i).copied())
    }

    /// The second distribution as set: none, or one, or why there is none.
    fn other(&self) -> Option<Result<Dist, &'static str>> {
        self.road().map(|r| r.of(&self.dist()))
    }

    /// The distribution as drawn now (its parameters gliding to the ones set).
    fn shown_dist(&self) -> Dist {
        let p = &self.shown_params.shown;
        if p.len() == 2 {
            self.kind.dist([p[0] as f64, p[1] as f64])
        } else {
            self.dist()
        }
    }

    fn aim(&self) -> Aim {
        let p = self.p();
        (self.kind, [p[0].to_bits(), p[1].to_bits()], self.road)
    }

    /// Choose another kind: its parameters as they were left, the marks at the same
    /// cumulative probabilities as before, no road, and a range of its own.
    fn set_kind(&mut self, kind: Kind) {
        if kind == self.kind {
            return;
        }
        let before = self.dist();
        let shares = self.marks.map(|m| before.cdf(m).max(0.05).min(0.95));
        self.kind = kind;
        self.road = 0;
        self.sweep = None;
        self.aimed = None;
        let after = self.dist();
        self.marks = shares.map(|p| self.snap_mark(quantile(&after, p)));
        if !self.two {
            self.marks[1] = self.snap_mark(quantile(&after, 0.25));
        }
        let p = self.p();
        self.shown_params.snap(&[p[0] as f32, p[1] as f32]);
        self.cut = true;
    }

    /// A mark of a count sits between two bars: the shaded bars are then exactly those
    /// left of it.
    fn snap_mark(&self, x: f64) -> f64 {
        if self.kind.counts() {
            x.floor() + 0.5
        } else {
            x
        }
    }

    /// Set parameter `i` from the panel or a handle (within its range; for the uniform,
    /// its ends kept apart).
    fn set_param(&mut self, i: usize, v: f64) {
        let Some(spec) = self.kind.params().get(i) else { return };
        let mut v = if v.is_nan() { spec.start } else { v.max(spec.lo).min(spec.hi) };
        if spec.whole {
            v = v.round();
        }
        let k = self.kind.index();
        self.params[k][i] = v;
        if self.kind == Kind::Uniform {
            let gap = 0.05;
            if i == 0 && self.params[k][1] < v + gap {
                self.params[k][1] = (v + gap).min(spec.hi);
                self.params[k][0] = self.params[k][1] - gap;
            } else if i == 1 && self.params[k][0] > v - gap {
                self.params[k][0] = (v - gap).max(spec.lo);
                self.params[k][1] = self.params[k][0] + gap;
            }
        }
        self.sweep = None;
    }

    /// Set the scene up for experiment `k`.
    fn set_up(&mut self, k: usize) {
        let Some(t) = TRIALS.get(k) else { return };
        let same = t.kind == self.kind;
        self.set_kind(t.kind);
        let i = t.kind.index();
        self.params[i] = t.params;
        if !same {
            self.shown_params.snap(&[t.params[0] as f32, t.params[1] as f32]);
        }
        self.road = t.road.and_then(|r| roads(t.kind).iter().position(|x| *x == r)).map_or(0, |p| p + 1);
        self.marks = t.marks;
        self.two = t.two;
        self.draws.want = t.draws;
        self.draws.way = t.way;
        self.draws.seed = 7 + k as u64;
        self.draws.key = None;
        self.sweep = t.sweep.map(|(param, to)| Sweep { param, from: t.params[param], to, wait: 0.8, length: 2.5, t: 0.0 });
        self.aimed = None;
    }

    /// Aim the charts' range at what is set: kept while what is set fits it well, else
    /// made afresh. (While a handle is held it stays, so that nothing slides under the
    /// hand; during a sweep it is aimed once at both ends of it.)
    fn aim_range(&mut self) {
        let aim = self.aim();
        if self.held || self.aimed == Some(aim) {
            return;
        }
        let fresh = self.aimed.is_none_or(|a| a.0 != aim.0);
        let mut dists = vec![self.dist()];
        if let Some(sweep) = &self.sweep {
            if self.aimed.is_some() {
                self.aimed = Some(aim);
                return;
            }
            let mut p = self.p();
            p[sweep.param] = sweep.to;
            dists.push(self.kind.dist(p));
        }
        if let Some(Ok(other)) = self.other() {
            dists.push(other);
        }
        let cores: Vec<(f64, f64)> = dists.iter().filter_map(core).collect();
        let Some(first) = cores.first().copied() else {
            self.aimed = Some(aim);
            return;
        };
        let union = cores.iter().fold(first, |(a, b), (c, d)| (a.min(*c), b.max(*d)));
        let (x0, x1) = x_range(union, self.kind.counts());
        let [old0, old1, old_y] = self.range;
        let core_w = (union.1 - union.0).max(1e-9) + if self.kind.counts() { 1.0 } else { 0.0 };
        let keep = !fresh && old0 <= union.0 - if self.kind.counts() { 0.5 } else { 0.0 } && old1 >= union.1 + if self.kind.counts() { 0.5 } else { 0.0 } && core_w >= 0.5 * (old1 - old0);
        let (x0, x1) = if keep { (old0, old1) } else { (x0, x1) };
        let top = dists.iter().map(|d| peak(d, x0, x1)).fold(0.0, f64::max);
        let y = if !fresh && keep && old_y >= 1.12 * top && old_y <= 2.0 * top { old_y } else { 1.25 * top };
        let y = if y.is_finite() && y > 0.0 { y } else { 1.0 };
        self.range = [x0, x1, y];
        self.aimed = Some(aim);
    }

    /// A mark that a new range has left outside the charts (the parameters changed under
    /// it) comes back in, to the place with the same cumulative probability as near as the
    /// middle 90 % allows, and inside the range. (Not while a handle is held: then the
    /// range stays as it was.)
    fn bring_marks_in(&mut self, d: &Dist) {
        let [x0, x1, _] = self.range;
        let margin = 0.02 * (x1 - x0);
        if self.held || !d.valid() || x0.is_nan() || x1.is_nan() || x1 <= x0 {
            return;
        }
        for m in 0..2 {
            let x = self.marks[m];
            if x >= x0 && x <= x1 {
                continue;
            }
            let p = d.cdf(x).max(0.05).min(0.95);
            let to = quantile(d, p).max(x0 + margin).min(x1 - margin);
            self.marks[m] = self.snap_mark(to);
        }
    }

    /// Everything that changes with time, without drawing: the sweep, the draws, the
    /// glides. `dt` is scene time, `real_dt` real time.
    fn step(&mut self, dt: f32, real_dt: f32) {
        if let Some(mut sweep) = self.sweep {
            sweep.t += dt.max(0.0);
            let k = self.kind.index();
            self.params[k][sweep.param] = sweep.value();
            self.sweep = if sweep.done() { None } else { Some(sweep) };
        }
        self.aim_range();
        let d = self.dist();
        self.bring_marks_in(&d);
        // the gap, worked out once for what is set
        let aim = self.aim();
        if self.gap.as_ref().is_none_or(|(a, _)| *a != aim) {
            let gap = match self.other() {
                Some(Ok(other)) => greatest_gap(&d, &other),
                _ => None,
            };
            self.gap = Some((aim, gap));
        }
        self.work_draws(&d, dt, real_dt);
        // the glides (or, for a new kind, the new picture at once)
        let p = self.p();
        let marks = [self.marks[0], self.marks[1], d.cdf(self.marks[0]), d.cdf(self.marks[1])].map(|v| if v.is_finite() { v as f32 } else { 0.0 });
        if std::mem::take(&mut self.cut) {
            self.shown_range.snap(&self.range.map(|v| v as f32));
            self.shown_marks.snap(&marks);
        }
        let mut moving = false;
        if self.held {
            self.shown_params.snap(&[p[0] as f32, p[1] as f32]);
        } else {
            moving |= self.shown_params.follow(&[p[0] as f32, p[1] as f32], real_dt, false);
        }
        moving |= self.shown_range.follow(&self.range.map(|v| v as f32), real_dt, false);
        moving |= self.shown_marks.follow(&marks, real_dt, false);
        let heights = self.draws.densities();
        moving |= if dt > 0.0 { self.draws.heights.follow(&heights, dt, false) } else { self.draws.heights.follow(&heights, real_dt, false) };
        self.moving = moving;
    }

    /// Make the draws that are due and land the ones whose flight is over.
    fn work_draws(&mut self, d: &Dist, dt: f32, real_dt: f32) {
        let p = self.p();
        let key = (self.kind, [p[0].to_bits(), p[1].to_bits()], self.draws.way, self.draws.want, self.draws.seed);
        let [x0, x1, _] = self.range;
        let bins_key = Bins::over(x0, x1, self.kind.counts());
        if self.draws.key != Some(key) {
            self.draws.key = Some(key);
            self.draws.bins = bins_key;
            self.draws.restart();
        } else if self.draws.bins.lo != bins_key.lo || self.draws.bins.width != bins_key.width || self.draws.bins.counts.len() != bins_key.counts.len() {
            // a new range: the same draws counted in its bins
            self.draws.bins = bins_key;
            let landed: Vec<f64> = self.draws.made[..self.draws.landed].iter().map(|m| m.0).collect();
            for x in landed {
                self.draws.bins.add(x);
            }
        }
        if !self.draws.busy() || dt <= 0.0 {
            return;
        }
        self.draws.clock += dt;
        if self.draws.made.len() < self.draws.want {
            let rate = self.draws.rate();
            let mut turn = scene_turn(&mut self.draws.pace, rate, dt, real_dt);
            while self.draws.made.len() < self.draws.want && turn.next() {
                self.draws.make(d);
            }
            self.draws.pace.end(turn);
        }
        while self.draws.landed < self.draws.made.len() && (self.draws.landed + 1) as f64 <= self.draws.landing() {
            let x = self.draws.made[self.draws.landed].0;
            self.draws.bins.add(x);
            self.draws.landed += 1;
        }
    }

    /// All the draws made and landed at once (for the tests).
    #[cfg(test)]
    fn settle(&mut self) {
        let d = self.dist();
        self.step(0.0, 1.0);
        while self.draws.made.len() < self.draws.want {
            self.draws.make(&d);
        }
        while self.draws.landed < self.draws.made.len() {
            let x = self.draws.made[self.draws.landed].0;
            self.draws.bins.add(x);
            self.draws.landed += 1;
        }
        for _ in 0..200 {
            self.step(0.05, 0.05);
        }
    }
}

// ----- numbers in words -------------------------------------------------------------------

/// A cumulative probability further out in a tail than this is not read back through
/// the quantile curve (it is too steep there to say anything).
const FAR: f64 = 0.9999;

/// A probability, with as many decimals as it needs to say something: one near 1 with
/// enough to show how far from 1 it is (up to eight; then it is 1 to the digits shown).
fn prob(p: f64) -> String {
    if !p.is_finite() {
        "?".to_string()
    } else if p <= 0.0 {
        "0".to_string()
    } else if p >= 1.0 {
        "1".to_string()
    } else if p < 0.001 {
        format!("{p:.1e}")
    } else if p < 0.01 {
        num(p, 4)
    } else if p <= 0.999 {
        num(p, 3)
    } else {
        let decimals = ((-(1.0 - p).log10()).ceil() as usize + 1).min(8);
        let s = num(p, decimals);
        if s.starts_with('1') {
            "1".to_string()
        } else {
            s
        }
    }
}

/// A count's cumulative probability `f`, written where the quantile reads it back:
/// $Q$ of what is written must give the same whole number as $Q(f)$. The staircase
/// steps up just after each of its heights, so the height may not be rounded up: as
/// [`prob`] writes it where that gives the same number back (its rounding went down),
/// else exactly where a few more decimals end it (Q(0.0546875) = 2 for ten fair
/// tries, where Q(0.055) would be 3), else cut short after one more decimal than
/// `prob` gives, with dots: "0.0546…" is the start of the number itself.
fn quantile_arg(d: &Dist, f: f64) -> String {
    let k = quantile(d, f);
    let gives_k = |s: &str| s.parse::<f64>().is_ok_and(|v| quantile(d, v) == k);
    let shown = prob(f);
    if gives_k(&shown) {
        return shown;
    }
    // (exactly: to the last bits of the arithmetic, which may leave F(2) a hair below
    // 0.0546875, so here the decimals are trusted rather than the numerical quantile)
    for decimals in 4..=7 {
        let s = num(f, decimals);
        if s.parse::<f64>().is_ok_and(|v| (v - f).abs() <= 1e-12) {
            return s;
        }
    }
    let from = if f < 0.01 { 5 } else { 4 };
    for decimals in from..=12 {
        let scale = 10f64.powi(decimals as i32);
        let cut = (f * scale).floor() / scale;
        if cut > 0.0 && quantile(d, cut) == k {
            return format!(r"{}\ldots", num(cut, decimals));
        }
    }
    shown
}

/// A gap between two cumulative curves, to two significant digits.
fn gap_text(g: f64) -> String {
    if g < NO_GAP {
        "0".to_string()
    } else if g >= 0.1 {
        num(g, 2)
    } else {
        let decimals = (-g.log10().floor() as usize + 1).min(8);
        num(g, decimals)
    }
}

/// A value of x, with the decimals a chart of this width needs.
fn x_text(x: f64, width: f64) -> String {
    let decimals = if width <= 2.5 {
        3
    } else if width <= 30.0 {
        2
    } else if width <= 300.0 {
        1
    } else {
        0
    };
    num(x, decimals)
}

impl Zoo {
    /// The marks as set, in order.
    fn interval(&self) -> (f64, f64) {
        let (a, b) = (self.marks[0], self.marks[1]);
        (a.min(b), a.max(b))
    }

    /// What the mark reads, in words and numbers: the sentences under the picture.
    fn reading(&self) -> Vec<String> {
        let d = self.dist();
        if !d.valid() {
            return vec!["These parameters make no distribution.".to_string()];
        }
        let w = self.range[1] - self.range[0];
        let x = |v: f64| x_text(v, w);
        let (lo, hi) = d.support();
        let mut out = Vec::new();
        if self.kind.counts() {
            let k = |m: f64| (m - 0.5).round();
            if self.two {
                let (a, b) = self.interval();
                let (j, k) = (k(a) + 1.0, k(b));
                let p = d.cdf(b) - d.cdf(a);
                out.push(if j > k {
                    "The marks are together: no whole number lies between them, and the probability between them is 0.".to_string()
                } else if j == k {
                    format!(r"Shaded: the bar at {} alone, $P(X = {}) = {}$, which is $F({}) - F({})$, the rise of the staircase at {}.", num(j, 0), num(j, 0), prob(p), num(k, 0), num(j - 1.0, 0), num(j, 0))
                } else {
                    format!(
                        r"Shaded: the bars from {} to {}, $P({} \le X \le {}) = {}$. It is $F({}) - F({}) = {} - {}$; outside them lies {}.",
                        num(j, 0),
                        num(k, 0),
                        num(j, 0),
                        num(k, 0),
                        prob(p),
                        num(k, 0),
                        num(j - 1.0, 0),
                        prob(d.cdf(b)),
                        prob(d.cdf(a)),
                        prob(1.0 - p)
                    )
                });
            } else {
                let m = self.marks[0];
                let kk = k(m);
                let f = d.cdf(m);
                // (no probability at all beyond the mark: the support ends there, or p is 1)
                let none_beyond = kk >= hi || (f >= 1.0 && d.pdf(kk + 1.0) == 0.0 && d.pdf(kk + 2.0) == 0.0);
                out.push(if kk < lo {
                    format!("No bar lies left of the mark: every value is at least {}, so $F = 0$ there. The quantile curve starts at $Q(0) = {}$.", num(lo, 0), num(lo, 0))
                } else if f <= 0.0 {
                    format!(r"No bar with any probability lies left of the mark: $P(X \le {}) = 0$.", num(kk, 0))
                } else if none_beyond {
                    format!(r"Every bar with any probability lies left of the mark: $P(X \le {}) = 1$, and $F = 1$ from there on.", num(kk, 0))
                } else if f > FAR {
                    format!(r"Shaded: all but a sliver, $P(X \le {}) > {}$. The mark is far out in the right tail, where the staircase has all but reached 1.", num(kk, 0), num(FAR, 4))
                } else {
                    format!(
                        r"Shaded: the bars up to {}, $P(X \le {}) = {}$. It is the height of the cumulative staircase at {}, and the quantile reads it back: $Q({}) = {}$, the least whole number whose cumulative probability reaches that height.",
                        num(kk, 0),
                        num(kk, 0),
                        prob(f),
                        num(kk, 0),
                        quantile_arg(&d, f),
                        num(quantile(&d, f), 0)
                    )
                });
            }
        } else if self.two {
            let (a, b) = self.interval();
            let p = d.cdf(b) - d.cdf(a);
            out.push(if b - a < 1e-12 {
                "The marks are together, and an interval of no width has probability 0: a smooth distribution gives no single value any probability.".to_string()
            } else {
                format!(
                    r"Shaded: the area between the marks, $P({} < X \le {}) = {}$. It is $F({}) - F({}) = {} - {}$, the rise of the cumulative curve between them; outside them lies {}.",
                    x(a),
                    x(b),
                    prob(p),
                    x(b),
                    x(a),
                    prob(d.cdf(b)),
                    prob(d.cdf(a)),
                    prob(1.0 - p)
                )
            });
        } else {
            let m = self.marks[0];
            let f = d.cdf(m);
            out.push(if m <= lo {
                format!("No probability lies left of the mark: it is below every value the distribution takes, where $F(x) = 0$. The quantile curve starts at $Q(0) = {}$.", x(lo))
            } else if m >= hi {
                format!("All the probability lies left of the mark: no value is above {}, so $F(x) = 1$ from there on.", x(hi))
            } else if f > FAR {
                format!(r"Shaded: all but a sliver, $P(X \le {}) > {}$. The mark is far out in the right tail, where the cumulative curve has all but reached 1 and the quantile curve is nearly a wall.", x(m), num(FAR, 4))
            } else if f < 1.0 - FAR {
                format!(r"Shaded: next to nothing, $P(X \le {}) = {}$. The mark is far out in the left tail, where the cumulative curve has barely left 0 and the quantile curve is nearly a wall.", x(m), prob(f))
            } else {
                format!(
                    r"Shaded: the area under the density left of the mark, $P(X \le {}) = {}$. It is the height of the cumulative curve at {}, and the quantile curve reads it back: $Q({}) = {}$.",
                    x(m),
                    prob(f),
                    x(m),
                    prob(f),
                    x(quantile(&d, f))
                )
            });
        }
        if let Some(other) = self.other() {
            let name = self.road().map_or("", |r| r.noun());
            out.push(match (other, self.gap.as_ref().and_then(|g| g.1)) {
                (Err(why), _) => format!("Compared with the {name}: none, {why}."),
                (Ok(_), Some(g)) if g.size < NO_GAP => format!("Compared with the {name} (orange): no gap at all, the same distribution under another name."),
                (Ok(o), gap) => {
                    let share = if self.two {
                        let (a, b) = self.interval();
                        let p = o.cdf(b) - o.cdf(a);
                        format!("{} between the marks and {} outside them", prob(p), prob(1.0 - p))
                    } else {
                        format!("{} up to the mark", prob(o.cdf(self.marks[0])))
                    };
                    let apart = gap.map_or(String::new(), |g| format!(" The cumulative curves are at most {} apart (pink).", gap_text(g.size)));
                    format!("The {name} (orange) puts {share}.{apart}")
                }
            });
        }
        out
    }

    /// The same, short enough for a phone: the reading in a formula or two, the gap in a
    /// line. (Where the mark is beyond an end or far in a tail, the sentence as it is.)
    fn reading_short(&self) -> Vec<String> {
        let full = self.reading();
        let d = self.dist();
        let w = self.range[1] - self.range[0];
        let x = |v: f64| x_text(v, w);
        let mut out = Vec::new();
        let first = full.first().cloned().unwrap_or_default();
        out.push(if !first.starts_with("Shaded: the") {
            first
        } else if self.two {
            let (a, b) = self.interval();
            let p = d.cdf(b) - d.cdf(a);
            if self.kind.counts() {
                let (j, k) = ((a - 0.5).round() + 1.0, (b - 0.5).round());
                if j == k {
                    format!(r"Shaded: $P(X = {}) = {}$, the rise of the staircase at {}.", num(j, 0), prob(p), num(j, 0))
                } else {
                    format!(r"Shaded: $P({} \le X \le {}) = {}$; outside: {}.", num(j, 0), num(k, 0), prob(p), prob(1.0 - p))
                }
            } else {
                format!(r"Shaded: $P({} < X \le {}) = {}$, the rise of $F$ between the marks; outside: {}.", x(a), x(b), prob(p), prob(1.0 - p))
            }
        } else {
            let m = self.marks[0];
            let f = d.cdf(m);
            if self.kind.counts() {
                let k = (m - 0.5).round();
                format!(r"Shaded: $P(X \le {}) = {}$, the staircase's height at {}; $Q({}) = {}$.", num(k, 0), prob(f), num(k, 0), quantile_arg(&d, f), num(quantile(&d, f), 0))
            } else {
                format!(r"Shaded: $P(X \le {}) = {}$, the curve's height at {}; $Q({}) = {}$.", x(m), prob(f), x(m), prob(f), x(quantile(&d, f)))
            }
        });
        if let (Some(road), Some(second)) = (self.road(), full.get(1)) {
            out.push(match (self.other(), self.gap.as_ref().and_then(|g| g.1)) {
                (Some(Ok(_)), Some(g)) if g.size >= NO_GAP => format!("Orange: {}. Greatest gap (pink): {}.", road.label(), gap_text(g.size)),
                _ => second.clone(),
            });
        }
        out
    }

    /// The mean, the standard deviation and the median, in a line.
    fn moments(&self) -> String {
        let d = self.dist();
        if !d.valid() {
            return String::new();
        }
        let w = self.range[1] - self.range[0];
        let x = |v: f64| x_text(v, w);
        let median = quantile(&d, 0.5);
        let median = if self.kind.counts() { num(median, 0) } else { x(median) };
        match (d.mean(), d.sd()) {
            (Some(m), Some(s)) => format!("mean {}, standard deviation {}, median {}", x(m), x(s), median),
            (Some(m), None) => format!("mean {}, no standard deviation (the variance is infinite), median {}", x(m), median),
            _ => format!("no mean and no standard deviation, median {median}"),
        }
    }
}

// ----- where everything goes --------------------------------------------------------------

const NO_RECT: Rect = (Vec2::ZERO, Vec2::ZERO);

/// Where everything goes in a view.
#[derive(Clone, Debug, PartialEq)]
struct Layout {
    view: Vec2,
    narrow: bool,
    /// Too small for the charts: nothing is drawn.
    tiny: bool,
    big: f32,
    text: f32,
    small: f32,
    /// The distributions' buttons, in the order of `KINDS`.
    kinds: Vec<Rect>,
    /// The line naming the distribution: its top-left and its width.
    name: (Vec2, f32),
    /// On a phone, the tabs that choose the chart drawn largest.
    tabs: Option<[Rect; 3]>,
    density: Rect,
    cdf: Rect,
    quantile: Rect,
    /// The box with the key and the reading (on a desktop), or the caption under the
    /// charts (on a phone).
    words: Rect,
}

const TABS: [&str; 3] = ["density", "cumulative", "quantile"];

impl Layout {
    /// `width` measures a button's words at a size.
    fn of(view: Vec2, inset: f32, tab: usize, width: &dyn Fn(&str, f32) -> f32) -> Layout {
        let view = Vec2::new(view.x.max(1.0), view.y.max(1.0));
        let narrow = view.x < 760.0;
        let tiny = view.x < 200.0 || view.y < 260.0;
        let (big, text, small) = if narrow { (12.5, 11.5, 10.5) } else { (14.0, 12.5, 11.5) };
        let side = if narrow { 8.0 } else { 18.0 };
        let top = inset.max(0.0) + if narrow { 6.0 } else { 10.0 };
        // the buttons, wrapped into rows
        let (h, gap, apart) = if narrow { (22.0, 4.0, 10.0) } else { (24.0, 5.0, 16.0) };
        let size = if narrow { small } else { text };
        let mut kinds = Vec::with_capacity(KINDS.len());
        let (mut x, mut y) = (side, top);
        for (i, kind) in KINDS.iter().enumerate() {
            let w = width(kind.name(), size) + if narrow { 12.0 } else { 18.0 };
            if i == FIRST_COUNT {
                x += apart - gap;
            }
            if x + w > view.x - side && x > side {
                x = side;
                y += h + gap;
            }
            kinds.push((Vec2::new(x, y), Vec2::new(x + w, y + h)));
            x += w + gap;
        }
        let below_buttons = y + h + if narrow { 6.0 } else { 10.0 };
        let name_lines = 2.0;
        let name = (Vec2::new(side, below_buttons), view.x - 2.0 * side);
        let mut y = below_buttons + name_lines * (big + 5.0) + 4.0;
        let (density, cdf, quantile, words, tabs);
        if narrow {
            let row = 22.0;
            tabs = Some({
                let r = plot::even_row(3, side, view.x - side, y, row, 6.0);
                [r[0], r[1], r[2]]
            });
            y += row + 8.0;
            let caption = 3.0 * (small + 5.0) + 8.0;
            let bottom = view.y - caption - 4.0;
            // each chart: its title above (18), its numbers below (16)
            let (above, below, between) = (20.0, 16.0, 4.0);
            let room = (bottom - y - 3.0 * (above + below) - 2.0 * between).max(30.0);
            let share = |k: usize| if k == tab.min(2) { 0.46 } else { 0.27 };
            let (l, r) = (side + 30.0, view.x - side - 4.0);
            let mut rects = [NO_RECT; 3];
            for (k, rect) in rects.iter_mut().enumerate() {
                let hk = room * share(k);
                *rect = (Vec2::new(l, y + above), Vec2::new(r, y + above + hk));
                y += above + hk + below + between;
            }
            [density, cdf, quantile] = rects;
            words = (Vec2::new(side, view.y - caption), Vec2::new(view.x - side, view.y - 4.0));
        } else {
            tabs = None;
            let inner = view.x - 2.0 * side;
            let (l, r) = (side + 44.0, side + 0.6 * inner);
            let top_d = y + 44.0;
            let bottom_c = view.y - 40.0;
            let room = (bottom_c - top_d - 18.0 - 44.0).max(40.0);
            let hd = 0.55 * room;
            density = (Vec2::new(l, top_d), Vec2::new(r, top_d + hd));
            let top_c = top_d + hd + 18.0 + 44.0;
            cdf = (Vec2::new(l, top_c), Vec2::new(r, bottom_c));
            let ql = r + 100.0;
            quantile = (Vec2::new(ql, top_c), Vec2::new(view.x - side - 6.0, bottom_c));
            words = (Vec2::new(r + 56.0, y + 4.0), Vec2::new(view.x - side, top_c - 50.0));
        }
        Layout { view, narrow, tiny, big, text, small, kinds, name, tabs, density, cdf, quantile, words }
    }
}

// ----- drawing ----------------------------------------------------------------------------

impl Zoo {
    /// Text broken into lines no wider than `width` pixels from `at` down, no lower than
    /// `bottom`; the lines are kept while the text is the same. Returns the height used.
    #[allow(clippy::too_many_arguments)]
    fn words(&mut self, f: &mut Frame, at: Vec2, width: f32, bottom: f32, text: &str, size: f32, color: Color) -> f32 {
        let key = (width.to_bits(), size.to_bits());
        let lines = match self.kept_lines.iter().find(|(t, w, s, _)| t == text && (*w, *s) == key) {
            Some((.., lines)) => lines.clone(),
            None => {
                if self.widths.len() > 4000 {
                    self.widths.clear();
                }
                let mut measure = |w: &str| *self.widths.entry((w.to_string(), size.to_bits())).or_insert_with(|| f.text_size(w, size).x);
                let space = measure("x x") - measure("xx");
                // (a line is as wide as its words and the spaces between them, near enough:
                // a pixel or two of room is left for what that misses)
                let mut lines: Vec<(String, f32)> = Vec::new();
                for word in words_of(text) {
                    let w = measure(word);
                    match lines.last_mut() {
                        Some((last, used)) if *used + space + w <= width - 2.0 => {
                            last.push(' ');
                            last.push_str(word);
                            *used += space + w;
                        }
                        _ => lines.push((word.to_string(), w)),
                    }
                }
                let lines: Vec<String> = lines.into_iter().map(|(l, _)| l).collect();
                if self.kept_lines.len() > 24 {
                    self.kept_lines.remove(0);
                }
                self.kept_lines.push((text.to_string(), key.0, key.1, lines.clone()));
                lines
            }
        };
        let step = size + 5.0;
        let mut used = 0.0;
        for line in lines {
            if at.y + used + step > bottom + 2.0 {
                break;
            }
            f.text_px(at + Vec2::new(0.0, used), line, size, color, Align2::LEFT_TOP);
            used += step;
        }
        used
    }

    /// The curves of what is drawn now, worked out again only when it changes.
    fn curves(&mut self, x0: f64, x1: f64) -> (Shape, Option<Shape>) {
        let p = &self.shown_params.shown;
        let pb = [p.first().copied().unwrap_or(0.0).to_bits(), p.get(1).copied().unwrap_or(0.0).to_bits()];
        let rb = [(x0 as f32).to_bits(), (x1 as f32).to_bits()];
        if let Some((kind, pp, rr, road, main, other)) = &self.drawn {
            if *kind == self.kind && *pp == pb && *rr == rb && *road == self.road {
                return (main.clone(), other.clone());
            }
        }
        let d = self.shown_dist();
        let main = shape(&d, x0, x1);
        let other = self.road().and_then(|r| r.of(&d).ok()).map(|o| shape(&o, x0, x1));
        self.drawn = Some((self.kind, pb, rb, self.road, main.clone(), other.clone()));
        (main, other)
    }
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
                if i > start {
                    words.push(&text[start..i]);
                }
                start = i + 1;
            }
            _ => {}
        }
        i += 1;
    }
    if start < text.len() {
        words.push(&text[start..]);
    }
    words
}

impl Scene for Zoo {
    fn view(&self) -> View {
        View::Flat { center: Vec2::ZERO, half_height: 1.0 }
    }

    fn settings(&self, s: &mut ViewSettings) {
        s.grid = false;
        s.bloom = s.bloom.min(0.2);
    }

    fn animated(&self) -> bool {
        self.sweep.is_some() || self.moving || self.draws.busy()
    }

    fn pointer_hint(&self) -> Option<&'static str> {
        Some("white dots under the density: drag the marks    blue dots under the quantile curve: drag $p$    green dots on some curves: drag to reshape    buttons over the picture: click")
    }

    fn ui(&mut self, ui: &mut egui::Ui, _: &Gfx) {
        ui::note(ui, r"A distribution says how likely each value of a random number $X$ is, and it can be drawn three ways, each worked out from the others. The density $f$ (for a count, the probability of each whole number): its area up to $x$ (for a count, the sum of its bars) is the cumulative probability $F(x) = P(X \le x)$. And the quantile function $Q$, the cumulative curve read the other way: $Q(p)$ is the $x$ with $F(x) = p$ (for a count, the least whole number with $F \ge p$). Drag the white mark and watch the one number it reads, the shaded area, in all three.");

        ui::section(ui, "Things to try");
        for (k, trial) in TRIALS.iter().enumerate() {
            if ui::experiment(ui, trial.button, trial.caption) {
                self.set_up(k);
            }
        }

        ui::section(ui, "The distribution");
        ui::note(ui, self.kind.story());
        let kind = self.kind;
        for (i, spec) in kind.params().iter().enumerate() {
            let v = self.p()[i];
            if spec.whole {
                let mut n = v.round().max(spec.lo) as usize;
                let range = spec.lo as usize..=spec.hi as usize;
                let changed = if spec.log { ui::slider_int_log(ui, spec.label, &mut n, range) } else { ui::slider_int(ui, spec.label, &mut n, range) };
                if changed {
                    self.set_param(i, n as f64);
                }
            } else {
                let mut x = v as f32;
                let range = spec.lo as f32..=spec.hi as f32;
                let changed = if spec.log { ui::slider_log(ui, spec.label, &mut x, range) } else { ui::slider(ui, spec.label, &mut x, range) };
                if changed {
                    self.set_param(i, x as f64);
                }
            }
        }

        ui::section(ui, "Marks");
        let mut two = self.two as usize;
        if ui::choice(ui, &["one mark", "two marks"], &mut two) {
            self.two = two == 1;
        }
        ui::note(ui, r"Drag the white dots under the density, or the blue ones along the $p$ axis of the quantile chart. A count's marks sit between its bars.");

        ui::section(ui, "Compare with");
        let mut labels = vec!["nothing"];
        labels.extend(roads(self.kind).iter().map(|r| r.label()));
        let mut road = self.road.min(labels.len() - 1);
        if ui::choice(ui, &labels, &mut road) {
            self.road = road;
        }
        ui::note(ui, "A second distribution in orange, and the greatest gap between the two cumulative curves: no interval's probability differs between them by more than twice it.");

        ui::section(ui, "Draws");
        let mut many = HOW_MANY.iter().position(|n| *n == self.draws.want).unwrap_or(0);
        if ui::choice(ui, &["none", "100", "1000", "10 000"], &mut many) {
            self.draws.want = HOW_MANY[many];
        }
        let story = has_story(self.kind, self.p());
        let ways: &[&str] = if story { &["random", "evenly spread", "the story"] } else { &["random", "evenly spread"] };
        // (a story asked for where there is none is drawn through the quantile curve at random)
        let mut way = if self.draws.way == Way::Story && !story { 0 } else { WAYS.iter().position(|w| *w == self.draws.way).unwrap_or(0) };
        if ui::choice(ui, ways, &mut way) {
            self.draws.way = WAYS[way];
        }
        ui::note(ui, "Random or evenly spread numbers from 0 to 1 pushed through the quantile curve; or made as the distribution's story says (sums of squares, waits, tries), where it has one.");
        if ui.button("new draws").clicked() {
            self.draws.seed = self.draws.seed.wrapping_add(1);
        }

        ui::section(ui, "Reading the picture");
        ui::key(
            ui,
            &[
                (TRUTH, "green: the distribution (its density or bars, its cumulative curve, its quantile curve), its mean and one standard deviation either side"),
                (MARK, "white: the mark, and the value of $x$ it stands at"),
                (AREA, "blue: the probability up to the mark (or between the marks), the same number in all three charts"),
                (OTHER, "orange: the distribution compared with it"),
                (ALARM, "pink: the greatest gap between the two cumulative curves"),
                (DATA, "cyan: draws, and their histogram"),
            ],
        );
        ui::section(ui, "Words");
        ui::term(ui, "Density:", "for a number that can take any value, probability per unit of $x$. The probability of an interval is the area under the density over it; any single value has probability 0.");
        ui::term(ui, "Probability of a value:", "for a count, $P(X = k)$: the height of the bar at $k$. The bars' heights add up to 1.");
        ui::term(ui, "Cumulative probability:", r"$F(x) = P(X \le x)$, the area (or the bars) up to $x$. It rises from 0 to 1, smoothly or in steps.");
        ui::term(ui, "Quantile:", r"$Q(p)$, the least $x$ with $F(x) \ge p$: the $x$ below which a share $p$ lies. $Q(0.5)$ is the median, $Q(0.25)$ and $Q(0.75)$ the quartiles.");
        ui::term(ui, "Parameter:", "a number that picks one distribution out of a family, as $n$ and $p$ pick one binomial.");
        ui::term(ui, "Support:", "the values a distribution can take: every number, the positive ones, those from 0 to 1, or whole numbers.");
        ui::term(ui, "Mean, standard deviation:", "the balance point of the distribution, and the square root of the mean squared distance from it. Some distributions have neither.");
        ui::term(ui, "Standard normal:", "the normal distribution with mean 0 and standard deviation 1; $Z$ stands for a number drawn from it.");
        ui::term(ui, "Degrees of freedom:", "the $k$ of the chi-square, $t$ and $F$: how many independent squared standard normal numbers are summed in them.");
        ui::term(ui, "Greatest gap:", "the largest vertical distance between two cumulative curves (the Kolmogorov distance). Between a staircase and a smooth curve it is at least half the tallest step.");

        ui::section(ui, "Numbers");
        let d = self.dist();
        let w = self.range[1] - self.range[0];
        ui::readout(ui, "distribution", format!("{}, {}", self.kind.name(), self.kind.settings(self.p())));
        let value = |v: Option<f64>, none: &str| v.map_or(none.to_string(), |v| x_text(v, w));
        ui::readout(ui, "mean", value(d.mean(), "none"));
        ui::readout(ui, "standard deviation", value(d.sd(), if d.mean().is_some() { "infinite" } else { "none" }));
        ui::readout(ui, "median", if self.kind.counts() { num(quantile(&d, 0.5), 0) } else { x_text(quantile(&d, 0.5), w) });
        if self.two {
            let (a, b) = self.interval();
            ui::readout_colored(ui, "between the marks", prob(d.cdf(b) - d.cdf(a)), AREA);
        } else {
            ui::readout_colored(ui, "up to the mark", prob(d.cdf(self.marks[0])), AREA);
        }
        if let Some((_, Some(g))) = &self.gap {
            ui::readout_colored(ui, "greatest gap", gap_text(g.size), ALARM);
        }
        if self.draws.landed > 0 {
            let landed = &self.draws.made[..self.draws.landed];
            let mean = landed.iter().map(|m| m.0).sum::<f64>() / landed.len() as f64;
            ui::readout_colored(ui, "draws landed", count(self.draws.landed), DATA);
            ui::readout_colored(ui, "their mean", x_text(mean, w), DATA);
            // (sorted once, when the last has landed: not in every frame while they come)
            if !self.draws.busy() {
                let median = match self.draws.median {
                    Some((n, m)) if n == self.draws.landed => m,
                    _ => stats::median(&landed.iter().map(|m| m.0).collect::<Vec<_>>()),
                };
                self.draws.median = Some((self.draws.landed, median));
                ui::readout_colored(ui, "their median", x_text(median, w), DATA);
            }
        }
        ui::see_also(
            ui,
            &[
                ("chance", "probability as a share, and conditional probability"),
                ("moments", "what the mean and standard deviation say, and miss"),
                ("clt", "why the normal turns up everywhere"),
                ("lln", "why a histogram of many draws settles on the density"),
                ("likelihood", "choosing a parameter from data"),
            ],
        );
    }

    fn frame(&mut self, f: &mut Frame) {
        let measure = |t: &str, s: f32| f.text_size(t, s).x;
        let lay = Layout::of(f.view_size(), f.top_inset, self.tab, &measure);
        self.held = false;
        if lay.tiny {
            self.step(f.dt, f.real_dt);
            return;
        }
        // the buttons over the picture
        for (k, rect) in lay.kinds.iter().enumerate() {
            let size = if lay.narrow { lay.small } else { lay.text };
            if plot::button(f, *rect, KINDS[k].name(), size, KINDS[k] == self.kind) {
                self.set_kind(KINDS[k]);
            }
        }
        if let Some(tabs) = lay.tabs {
            for (k, rect) in tabs.iter().enumerate() {
                if plot::button(f, *rect, TABS[k], lay.small, self.tab == k) {
                    self.tab = k;
                }
            }
        }
        // the handles first, from the picture as it was drawn last, then the time step
        let (x0, x1, ymax) = (self.shown_range.get(0) as f64, self.shown_range.get(1) as f64, self.shown_range.get(2) as f64);
        let pd = Plot::new(lay.density, [x0, x1], [0.0, ymax]);
        let pq = Plot::new(lay.quantile, [0.0, P_TOP], [x0, x1]);
        self.handles(f, &pd, &pq);
        self.step(f.dt, f.real_dt);
        let (x0, x1, ymax) = (self.shown_range.get(0) as f64, self.shown_range.get(1) as f64, self.shown_range.get(2) as f64);
        let pd = Plot::new(lay.density, [x0, x1], [0.0, ymax]);
        let pc = Plot::new(lay.cdf, [x0, x1], [0.0, 1.04]);
        let pq = Plot::new(lay.quantile, [0.0, P_TOP], [x0, x1]);
        let (main, other) = self.curves(x0, x1);
        self.draw_name(f, &lay);
        self.draw_density(f, &lay, &pd, &main, other.as_ref());
        self.draw_cdf(f, &lay, &pc, &main, other.as_ref());
        self.draw_quantile(f, &lay, &pq, &main, other.as_ref());
        self.draw_flights(f, &pd, &pq);
        self.draw_words(f, &lay);
        self.part_ticks(f, [&pd, &pc, &pq]);
    }
}

impl Zoo {
    /// New round values on an axis: make sure the axes' batch and the labels are not as
    /// many as in the last frame, so that nothing takes the new ones for the old ones
    /// moved (the ranges glide, and a tick that goes off one end as another comes on at
    /// the other would look like every tick jumping one place along).
    fn part_ticks(&mut self, f: &mut Frame, plots: [&Plot; 3]) {
        let mut ticks = Vec::new();
        for p in plots {
            ticks.extend(plot::Ticks::of(p.x[0], p.x[1], p.width()).values());
            ticks.push(f64::NAN);
            ticks.extend(plot::Ticks::of(p.y[0], p.y[1], p.height() * 1.6).values());
            ticks.push(f64::NAN);
        }
        let same = ticks.len() == self.ticks.len() && ticks.iter().zip(&self.ticks).all(|(a, b)| a == b || (a.is_nan() && b.is_nan()));
        let count = |f: &mut Frame| (plot::apart(f, plot::AXES, false).pts.iter().filter(|p| p.width >= 0.0).count(), f.dl.labels.len());
        if !same {
            let (axes, labels) = count(f);
            if axes == self.drawn_last.0 {
                let far = f.hud(Vec2::ZERO);
                plot::apart(f, plot::AXES, false).segment(far, far, 1.0, Color::CLEAR);
            }
            if labels == self.drawn_last.1 {
                f.text_px(Vec2::ZERO, "", 1.0, Color::CLEAR, Align2::LEFT_TOP);
            }
        }
        self.ticks = ticks;
        self.drawn_last = count(f);
    }

    /// The marks' handles, the quantile chart's handles on its $p$ axis, and the
    /// parameters' handles on the density, where the parameters have a natural place.
    fn handles(&mut self, f: &mut Frame, pd: &Plot, pq: &Plot) {
        let marks = self.shown_marks.shown.clone();
        if marks.len() < 4 {
            return;
        }
        let d = self.dist();
        let count = if self.two { 2 } else { 1 };
        for m in 0..count {
            if let Some((x, _)) = pd.handle(f, MARK_HANDLE + m as u32, marks[m] as f64, pd.y[0], MARK) {
                self.marks[m] = self.snap_mark(x);
                self.held = true;
            }
            if let Some((p, _)) = pq.handle(f, P_HANDLE + m as u32, marks[2 + m] as f64, pq.y[0], AREA) {
                let p = p.max(0.0005).min(0.9995);
                let x = quantile(&d, p);
                if x.is_finite() {
                    self.marks[m] = if self.kind.counts() { x + 0.5 } else { x };
                }
                self.held = true;
            }
        }
        // the parameters' handles, on the curve as drawn
        let shown = self.shown_dist();
        let p = self.p();
        let at = |x: f64| (x, shown.pdf(x));
        let spots: Vec<((f64, f64), usize, &str)> = match shown {
            Dist::Normal { mean, sd } => vec![(at(mean), 0, r"$\mu$"), (at(mean + sd), 1, r"$\mu + \sigma$")],
            Dist::Uniform { lo, hi } => vec![((lo, 1.0 / (hi - lo)), 0, "$a$"), ((hi, 1.0 / (hi - lo)), 1, "$b$")],
            Dist::Exponential { rate } => vec![(at(1.0 / rate), 0, r"mean $1/\lambda$")],
            Dist::Gamma { shape, rate } => vec![(at(shape / rate), 1, r"mean $k/\lambda$")],
            Dist::Cauchy { centre, scale } => vec![(at(centre), 0, "$x_0$"), (at(centre + scale), 1, r"$x_0 + \gamma$")],
            Dist::LogNormal { mu, .. } => vec![(at(mu.exp()), 0, r"median $e^\mu$")],
            _ => Vec::new(),
        };
        for (n, ((x, y), param, label)) in spots.into_iter().enumerate() {
            if !(x >= pd.x[0] && x <= pd.x[1] && y.is_finite()) {
                continue;
            }
            let y = y.min(pd.y[1]);
            pd.text(f, x, y, Vec2::new(8.0, -8.0), label, 11.0, TRUTH, Align2::LEFT_BOTTOM);
            if let Some((to, _)) = pd.handle(f, PARAM_HANDLE + n as u32, x, y, TRUTH) {
                self.held = true;
                let v = match (self.kind, param) {
                    (Kind::Normal, 0) | (Kind::Uniform, _) | (Kind::Cauchy, 0) => to,
                    (Kind::Normal, _) => to - p[0],
                    (Kind::Exponential, _) => 1.0 / to.max(1e-9),
                    (Kind::Gamma, _) => p[0] / to.max(1e-9),
                    (Kind::Cauchy, _) => to - p[0],
                    (Kind::LogNormal, _) => to.max(1e-9).ln(),
                    _ => continue,
                };
                self.set_param(param, v);
            }
        }
    }

    fn draw_name(&mut self, f: &mut Frame, lay: &Layout) {
        let (at, width) = lay.name;
        let text = if lay.narrow {
            format!("{}, {}: {}", capital(self.kind.name()), self.kind.settings(self.p()), self.kind.short())
        } else {
            format!("{}, {}. {}", capital(self.kind.name()), self.kind.settings(self.p()), self.kind.story())
        };
        self.words(f, at, width, at.y + 2.0 * (lay.big + 5.0), &text, lay.big, pal::TEXT);
    }

    fn draw_density(&mut self, f: &mut Frame, lay: &Layout, plot: &Plot, main: &Shape, other: Option<&Shape>) {
        if plot.tiny() {
            return;
        }
        plot.backing(f, if lay.narrow { 30.0 } else { 44.0 }, 18.0, if lay.narrow { 20.0 } else { 44.0 }, 8.0);
        plot.y_grid(f);
        let counts = self.kind.counts();
        let title = if counts { r"probability of each value, $P(X = k)$" } else { "density $f(x)$" };
        if lay.narrow {
            f.text_px(Vec2::new(plot.rect.0.x - 26.0, plot.rect.0.y - 4.0), title, lay.small, pal::TEXT, Align2::LEFT_BOTTOM);
        } else {
            plot.title(f, title, lay.text);
        }
        plot.x_axis(f, "", lay.small);
        plot.y_axis(f, "", lay.small);
        let marks = &self.shown_marks.shown;
        let (m0, m1) = (marks.first().copied().unwrap_or(0.0) as f64, marks.get(1).copied().unwrap_or(0.0) as f64);
        let (from, to) = if self.two { (m0.min(m1), m0.max(m1)) } else { (plot.x[0], m0) };
        let d = self.shown_dist();
        // the draws' histogram, under the distribution
        let heights = self.draws.heights.shown.clone();
        let bins = &self.draws.bins;
        if self.draws.want > 0 && heights.len() == bins.counts.len() {
            let mut outline = Vec::with_capacity(2 * heights.len() + 2);
            for (i, h) in heights.iter().enumerate() {
                let (a, b) = (bins.lo + i as f64 * bins.width, bins.lo + (i + 1) as f64 * bins.width);
                let (a, b) = if counts { (a + 0.28, b - 0.28) } else { (a, b) };
                plot.bar(f, a, b, 0.0, *h as f64, 0.0, DATA.a(if counts { 0.75 } else { 0.1 }));
                if !counts {
                    // (a bar above the chart is cut at its top edge, as its fill is: the
                    // outline keeps its number of points)
                    let h = (*h as f64).min(plot.y[1]);
                    outline.push((a.max(plot.x[0]), h));
                    outline.push((b.min(plot.x[1]), h));
                }
            }
            if !counts {
                plot.path(f, HISTOGRAM, outline, 1.5, DATA);
            }
        }
        match main {
            Shape::Smooth { xs, pdf, .. } => {
                plot.area(f, from, to, 0.0, 120, AREA.a(0.26), |x| d.pdf(x).min(1e9));
                if let Some(Shape::Smooth { xs: ox, pdf: op, .. }) = other {
                    plot.path(f, plot::CURVES, ox.iter().copied().zip(op.iter().copied()), 3.0, OTHER.a(0.9));
                }
                plot.path(f, plot::CURVES, xs.iter().copied().zip(pdf.iter().copied()), 2.0, TRUTH);
            }
            Shape::Counts { first, pmf, .. } => {
                let unit = (plot.width() as f64 / (plot.x[1] - plot.x[0]).max(1e-9)) as f32;
                let gap = (0.12 * unit).min(4.0);
                for (i, p) in pmf.iter().enumerate() {
                    let k = first + i as f64;
                    let shaded = k > from && k < to;
                    let color = if shaded { AREA.a(0.55) } else { TRUTH.a(0.45) };
                    plot.bar(f, k - 0.5, k + 0.5, 0.0, *p, gap, color);
                }
                if let Some(Shape::Smooth { xs: ox, pdf: op, .. }) = other {
                    plot.path(f, plot::CURVES, ox.iter().copied().zip(op.iter().copied()), 2.2, OTHER.a(0.95));
                }
            }
        }
        if let Some(Shape::Counts { first, pmf, .. }) = other {
            // a count compared: the outlines of its bars
            let mut comb = Vec::with_capacity(4 * pmf.len());
            for (i, p) in pmf.iter().enumerate() {
                let k = first + i as f64;
                comb.extend([(k - 0.36, 0.0), (k - 0.36, *p), (k + 0.36, *p), (k + 0.36, 0.0)]);
            }
            plot.path(f, COMBS, comb, 1.6, OTHER.a(0.95));
        }
        // the mean, and one standard deviation either side, along the top
        let top = plot.y[1];
        let band = plot.rect.0.y + 9.0;
        match (d.mean(), d.sd()) {
            (Some(m), sd) if m.is_finite() => {
                plot.vline(f, plot::MARKS, m, 1.2, TRUTH.a(0.75), true);
                let s = sd.filter(|s| *s > 0.0 && s.is_finite());
                let (a, b) = match s {
                    Some(s) => (plot.px_x(m - s).max(plot.rect.0.x), plot.px_x(m + s).min(plot.rect.1.x)),
                    None => (plot.px_x(m), plot.px_x(m)),
                };
                if let Some(s) = s {
                    if b > a {
                        plot::line_of(f, plot::MARKS, Vec2::new(a, band), Vec2::new(b, band), 1.4, TRUTH.a(0.85));
                    }
                    for x in [plot.px_x(m - s), plot.px_x(m + s)] {
                        if x >= plot.rect.0.x && x <= plot.rect.1.x {
                            plot::line_of(f, plot::MARKS, Vec2::new(x, band - 4.0), Vec2::new(x, band + 4.0), 1.4, TRUTH.a(0.85));
                        }
                    }
                }
                let label = match sd {
                    Some(s) if s > 0.0 => r"mean $\pm 1$ sd",
                    Some(_) => "mean (no spread)",
                    None => "mean (no standard deviation)",
                };
                // right of the bar, sliding along with it and stopping at the chart's edge
                // (on a plate, so that it reads where it then lies over the bar or a mark):
                // never jumping to the other side
                let width = f.text_size(label, lay.small).x;
                let x = (b + 6.0).min(plot.rect.1.x - width - 4.0).max(plot.rect.0.x + 4.0);
                f.caption_px(Vec2::new(x, band), label, lay.small, TRUTH, Align2::LEFT_CENTER);
            }
            _ => {
                f.text_px(Vec2::new(plot.rect.0.x + 6.0, band), "no mean", lay.small, TRUTH, Align2::LEFT_CENTER);
            }
        }
        // where the density has no bound
        let (lo, hi) = d.support();
        for (end, place) in [(lo, Align2::LEFT_TOP), (hi, Align2::RIGHT_TOP)] {
            if end.is_finite() && d.pdf(end).is_infinite() && end >= plot.x[0] && end <= plot.x[1] {
                let x = plot.px_x(end) + if place == Align2::LEFT_TOP { 6.0 } else { -6.0 };
                f.text_px(Vec2::new(x, plot.rect.0.y + 2.0 * lay.small + 14.0), format!("↑ no bound at {}", num(end, 0)), lay.small, TRUTH, place);
            }
        }
        // the marks
        let count = if self.two { 2 } else { 1 };
        let w = plot.x[1] - plot.x[0];
        for m in [m0, m1].into_iter().take(count) {
            plot.vline(f, plot::MARKS, m, 1.3, MARK.a(0.85), false);
            if !counts && m >= plot.x[0] && m <= plot.x[1] {
                // right of the line's foot, stopping at the chart's edge
                let text = format!("$x = {}$", x_text(m, w));
                let width = f.text_size(&text, lay.small).x;
                let x = (plot.px_x(m) + 7.0).min(plot.rect.1.x - width - 2.0);
                f.text_px(Vec2::new(x, plot.rect.1.y - 3.0), text, lay.small, MARK, Align2::LEFT_BOTTOM);
            }
        }
        // the probability, written where it is shaded
        let truth = self.dist();
        let (value, x) = if self.two {
            let (a, b) = self.interval();
            (truth.cdf(b) - truth.cdf(a), 0.5 * (from + to))
        } else {
            (truth.cdf(self.marks[0]), m0)
        };
        let at = Vec2::new(plot.px_x(x).max(plot.rect.0.x + 30.0).min(plot.rect.1.x - 30.0), plot.px_y(0.45 * top));
        let align = if self.two { Align2::CENTER_CENTER } else { Align2::RIGHT_CENTER };
        let at = if self.two { at } else { at - Vec2::new(6.0, 0.0) };
        f.caption_px(at, format!("${}$", prob(value)), lay.text, AREA, align);
        // what lies beyond the left edge, counted in the number but not shaded
        let beyond = truth.cdf(plot.x[0]);
        if !self.two && beyond >= 0.0005 && m0 > plot.x[0] {
            f.text_px(Vec2::new(plot.rect.0.x + 6.0, plot.px_y(0.62 * top)), format!("← {} more beyond the edge", prob(beyond)), lay.small, AREA, Align2::LEFT_CENTER);
        }
        // how the draws are made
        if self.draws.want > 0 {
            let how = if self.draws.way == Way::Story && has_story(self.kind, self.p()) {
                story_formula(self.kind, &truth)
            } else if self.draws.way == Way::Even {
                r"each draw: $Q(p)$, $p$ evenly spread from 0 to 1".to_string()
            } else {
                r"each draw: $Q(U)$, $U$ uniform from 0 to 1".to_string()
            };
            f.text_px(Vec2::new(plot.rect.1.x - 6.0, plot.rect.0.y + 2.0 * lay.small + 14.0), how, lay.small, DATA, Align2::RIGHT_TOP);
        }
    }

    fn draw_cdf(&mut self, f: &mut Frame, lay: &Layout, plot: &Plot, main: &Shape, other: Option<&Shape>) {
        if plot.tiny() {
            return;
        }
        plot.backing(f, if lay.narrow { 30.0 } else { 44.0 }, if lay.narrow { 16.0 } else { 34.0 }, if lay.narrow { 20.0 } else { 44.0 }, 8.0);
        plot.y_grid(f);
        let title = r"cumulative probability $F(x) = P(X \le x)$";
        if lay.narrow {
            f.text_px(Vec2::new(plot.rect.0.x - 26.0, plot.rect.0.y - 4.0), title, lay.small, pal::TEXT, Align2::LEFT_BOTTOM);
            plot.x_axis(f, "", lay.small);
        } else {
            plot.title(f, title, lay.text);
            plot.x_axis(f, "$x$", lay.small);
        }
        plot.y_axis(f, "", lay.small);
        let (x0, x1) = (plot.x[0], plot.x[1]);
        let kind = |s: &Shape| if matches!(s, Shape::Counts { .. }) { STAIRS } else { plot::CURVES };
        if let Some(o) = other {
            plot.path(f, kind(o), o.cumulative(x0, x1), 3.0, OTHER.a(0.9));
        }
        plot.path(f, kind(main), main.cumulative(x0, x1), 2.0, TRUTH);
        // the greatest gap
        if let Some((_, Some(g))) = &self.gap {
            if g.size >= NO_GAP && g.at >= x0 && g.at <= x1 {
                let (a, b) = (plot.at(g.at, g.a), plot.at(g.at, g.b));
                plot::line_of(f, plot::MARKS, a, b, 2.6, ALARM);
                // right of the gap, stopping at the chart's edge
                let mid = (a + b) * 0.5;
                let text = format!("greatest gap {}", gap_text(g.size));
                let width = f.text_size(&text, lay.small).x;
                let x = (mid.x + 8.0).min(plot.rect.1.x - width - 4.0);
                f.caption_px(Vec2::new(x, mid.y), text, lay.small, ALARM, Align2::LEFT_CENTER);
            }
        }
        // the marks: up from the axis to the curve, across to the probability
        let marks = &self.shown_marks.shown;
        if marks.len() < 4 {
            return;
        }
        let count = if self.two { 2 } else { 1 };
        for m in 0..count {
            let (x, p) = (marks[m] as f64, marks[2 + m] as f64);
            // (held to the chart: a line never runs out of it into the next)
            let at = plot.at(x, p).max(plot.rect.0).min(plot.rect.1);
            plot::dashed_of(f, plot::MARKS, Vec2::new(at.x, plot.rect.1.y), at, 1.2, MARK.a(0.8));
            plot::line_of(f, plot::MARKS, Vec2::new(plot.rect.0.x, at.y), at, 1.6, AREA);
            plot.dot(f, x, p, 7.0, AREA);
            if !self.two {
                f.caption_px(Vec2::new(plot.rect.0.x + 6.0, at.y - 3.0), format!("${}$", prob(self.dist().cdf(self.marks[0]))), lay.small, AREA, Align2::LEFT_BOTTOM);
            }
        }
        if self.two {
            // the rise between the two heights, along the axis
            let (pa, pb) = (marks[2] as f64, marks[3] as f64);
            let (ya, yb) = (plot.px_y(pa), plot.px_y(pb));
            let x = plot.rect.0.x + 3.0;
            plot::line_of(f, plot::MARKS, Vec2::new(x, ya), Vec2::new(x, yb), 4.0, AREA);
            let (a, b) = self.interval();
            let d = self.dist();
            f.caption_px(Vec2::new(x + 8.0, 0.5 * (ya + yb)), format!("${}$", prob(d.cdf(b) - d.cdf(a))), lay.small, AREA, Align2::LEFT_CENTER);
        }
    }

    fn draw_quantile(&mut self, f: &mut Frame, lay: &Layout, plot: &Plot, main: &Shape, other: Option<&Shape>) {
        if plot.tiny() {
            return;
        }
        plot.backing(f, if lay.narrow { 30.0 } else { 44.0 }, if lay.narrow { 16.0 } else { 34.0 }, if lay.narrow { 20.0 } else { 44.0 }, 8.0);
        plot.y_grid(f);
        let counts = self.kind.counts();
        let title = if counts { r"quantile $Q(p)$: the least $k$ with $F(k) \ge p$" } else { r"quantile $Q(p)$: the $x$ with $F(x) = p$" };
        if lay.narrow {
            f.text_px(Vec2::new(plot.rect.0.x - 26.0, plot.rect.0.y - 4.0), title, lay.small, pal::TEXT, Align2::LEFT_BOTTOM);
            plot.x_axis(f, "", lay.small);
        } else {
            plot.title(f, title, lay.text);
            plot.x_axis(f, "$p$", lay.small);
        }
        plot.y_axis(f, "", lay.small);
        let (x0, x1) = (plot.y[0], plot.y[1]);
        let swap = |pts: Vec<(f64, f64)>| pts.into_iter().map(|(x, p)| (p, x)).collect::<Vec<_>>();
        let kind = |s: &Shape| if matches!(s, Shape::Counts { .. }) { STAIRS } else { plot::CURVES };
        if let Some(o) = other {
            plot.path(f, kind(o), swap(o.cumulative(x0, x1)), 3.0, OTHER.a(0.9));
        }
        plot.path(f, kind(main), swap(main.cumulative(x0, x1)), 2.0, TRUTH);
        let marks = &self.shown_marks.shown;
        if marks.len() < 4 {
            return;
        }
        let count = if self.two { 2 } else { 1 };
        let d = self.dist();
        let w = x1 - x0;
        for m in 0..count {
            let (x, p) = (marks[m] as f64, marks[2 + m] as f64);
            // the quantile of the mark's probability: for a count, the whole number below the mark
            let q = if counts { x.floor() } else { x };
            let at = plot.at(p, q.max(x0).min(x1)).max(plot.rect.0).min(plot.rect.1);
            plot::line_of(f, plot::MARKS, Vec2::new(at.x, plot.rect.1.y), at, 1.6, AREA);
            plot::dashed_of(f, plot::MARKS, Vec2::new(plot.rect.0.x, at.y), at, 1.2, MARK.a(0.8));
            plot.dot(f, p, q, 7.0, AREA);
            let f_m = d.cdf(self.marks[m]);
            // (far out in a tail, and where all or none of the probability is left of the
            // mark, there is nothing to read back: a count's Q(1) is not the least k with
            // F(k) = 1 but the end of its support)
            if !(f_m > 1.0 - FAR && f_m < FAR) {
                continue;
            }
            let truth = if counts { quantile(&d, f_m) } else { self.marks[m] };
            let label = if counts { format!("$Q({}) = {}$", quantile_arg(&d, f_m), num(truth, 0)) } else { format!("$Q({}) = {}$", prob(f_m), x_text(truth, w)) };
            // above and left of the point, where a rising curve leaves room, kept inside
            // the chart (it slides; it never jumps to another side)
            let size = f.text_size(&label, lay.small);
            let corner = Vec2::new((at.x - 8.0).max(plot.rect.0.x + size.x + 4.0), (at.y - 4.0).max(plot.rect.0.y + size.y + 2.0));
            f.caption_px(corner, label, lay.small, AREA, Align2::RIGHT_BOTTOM);
        }
    }

    /// The draws on their way: up from the $p$ axis of the quantile chart to its curve
    /// and across to the value, then down onto the density's axis (a story's draws fall
    /// straight onto it). Each is where its age puts it, in scene time, so they move
    /// evenly at any frame rate; they are drawn filled (as many come and go in a frame as
    /// the rate makes them, and a line or a point batch would take each for another).
    fn draw_flights(&mut self, f: &mut Frame, pd: &Plot, pq: &Plot) {
        let dr = &self.draws;
        if dr.landed >= dr.made.len() {
            return;
        }
        let stride = ((dr.rate() / 60.0).ceil() as usize).max(1);
        let mut shown = 0;
        let mut i = dr.made.len();
        while i > dr.landed && shown < 40 {
            i -= 1;
            if !i.is_multiple_of(stride) {
                continue;
            }
            shown += 1;
            let (x, u) = dr.made[i];
            let age = ((dr.clock - dr.born(i)) / FLIGHT).max(0.0).min(1.0) as f64;
            let fall = if u.is_nan() {
                age
            } else {
                // up the quantile chart, then across, then down the density chart
                let (y0, xq) = (pq.y[0], x.max(pq.y[0]).min(pq.y[1]));
                if age < 0.5 {
                    let up = (age / 0.25).min(1.0);
                    let across = ((age - 0.25) / 0.25).max(0.0);
                    let (foot, turn) = (pq.at(u, y0), pq.at(u, y0 + (xq - y0) * up));
                    let at = pq.at(u * (1.0 - across), y0 + (xq - y0) * up);
                    let faint = DATA.a(0.35);
                    plot::fill(f, (Vec2::new(foot.x - 0.5, turn.y), Vec2::new(foot.x + 0.5, foot.y)), faint);
                    if across > 0.0 {
                        plot::fill(f, (Vec2::new(at.x, turn.y - 0.5), Vec2::new(turn.x, turn.y + 0.5)), faint);
                    }
                    blob(f, at, 3.0, DATA);
                    continue;
                }
                (age - 0.5) / 0.5
            };
            if x >= pd.x[0] && x <= pd.x[1] {
                let y = pd.y[1] * (1.0 - fall * fall);
                blob(f, pd.at(x, y), 3.0, DATA);
            }
        }
    }

    fn draw_words(&mut self, f: &mut Frame, lay: &Layout) {
        let (min, max) = lay.words;
        if max.x - min.x < 40.0 || max.y - min.y < 20.0 {
            return;
        }
        let reading = if lay.narrow { self.reading_short() } else { self.reading() };
        if lay.narrow {
            hud_panel(f, min - Vec2::new(4.0, 4.0), max + Vec2::new(4.0, 2.0));
            let mut y = min.y;
            for line in reading.iter().take(2) {
                y += self.words(f, Vec2::new(min.x, y), max.x - min.x, max.y, line, lay.small, pal::TEXT) + 2.0;
            }
            return;
        }
        hud_panel(f, min - Vec2::new(10.0, 8.0), max + Vec2::new(8.0, 8.0));
        let mut entries: Vec<(Color, String)> = vec![
            (TRUTH, format!("the {} distribution", self.kind.name())),
            (MARK, "the mark: drag the white dot".to_string()),
            (AREA, "probability up to the mark: one number, three places".to_string()),
        ];
        if self.two {
            entries[1].1 = "the marks: drag the white dots".to_string();
            entries[2].1 = "probability between the marks: one number, three places".to_string();
        }
        if let Some(road) = self.road() {
            entries.push((OTHER, format!("compared: {}", road.label())));
            entries.push((ALARM, "greatest gap between the cumulative curves".to_string()));
        }
        if self.draws.want > 0 {
            entries.push((DATA, format!("{} of {} draws landed, and their histogram", count(self.draws.landed), count(self.draws.want))));
        }
        let refs: Vec<(Color, &str)> = entries.iter().map(|(c, s)| (*c, s.as_str())).collect();
        let used = hud_key(f, min, &refs, lay.small);
        let mut y = min.y + used + 6.0;
        let moments = self.moments();
        y += self.words(f, Vec2::new(min.x, y), max.x - min.x, max.y, &moments, lay.small, TRUTH) + 8.0;
        for line in reading {
            y += self.words(f, Vec2::new(min.x, y), max.x - min.x, max.y, &line, lay.text, pal::TEXT) + 6.0;
        }
    }
}

/// A small filled disc on the screen, over the charts.
fn blob(f: &mut Frame, at: Vec2, radius: f32, color: Color) {
    if !at.is_finite() {
        return;
    }
    let c = f.hud_at(2, at);
    let rim: Vec<Vec3> = (0..8).map(|k| f.hud_at(2, at + Vec2::from_angle(k as f32 * TAU / 8.0) * radius)).collect();
    for k in 0..8 {
        f.dl.tris_alpha.tri(c, rim[k], rim[(k + 1) % 8], color);
    }
}

/// A name with a capital letter (a formula is left as it is).
fn capital(name: &str) -> String {
    let mut c = name.chars();
    match c.next() {
        Some(first) if first.is_ascii_lowercase() => first.to_ascii_uppercase().to_string() + c.as_str(),
        _ => name.to_string(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A scene showing `kind` with these parameters, its range aimed and its glides
    /// arrived.
    fn showing(kind: Kind, p: [f64; 2]) -> Zoo {
        let mut z = Zoo::create();
        z.set_kind(kind);
        z.params[kind.index()] = p;
        for _ in 0..100 {
            z.step(0.05, 0.05);
        }
        z
    }

    /// Every kind with its parameters as it opens, at the ends of their ranges, and in
    /// the corners that matter (a shape below 1, p of 0 and 1, one degree of freedom).
    fn cases() -> Vec<(Kind, [f64; 2])> {
        let mut out = Vec::new();
        for kind in KINDS {
            let specs = kind.params();
            let start = [specs[0].start, specs.get(1).map_or(0.0, |p| p.start)];
            out.push((kind, start));
            for (i, spec) in specs.iter().enumerate() {
                for v in [spec.lo, spec.hi] {
                    let mut p = start;
                    p[i] = v;
                    out.push((kind, p));
                }
            }
            if specs.len() == 2 {
                out.push((kind, [specs[0].lo, specs[1].lo]));
                out.push((kind, [specs[0].hi, specs[1].hi]));
                out.push((kind, [specs[0].lo, specs[1].hi]));
                out.push((kind, [specs[0].hi, specs[1].lo]));
            }
        }
        out.retain(|(k, p)| !(*k == Kind::Uniform && p[1] <= p[0]));
        out.push((Kind::Uniform, [-5.0, -4.95]));
        out
    }

    /// The area under a density from `a` to `b`, by Simpson's rule on points crowded
    /// towards both ends (where a density may have no bound): x = a + (b − a) u(t), with
    /// u(t) = t⁸ / (t⁸ + (1 − t)⁸), whose slope vanishes at both ends fast enough that
    /// the density times it stays finite.
    fn area(d: &Dist, a: f64, b: f64) -> f64 {
        let (lo, hi) = d.support();
        let (a, b) = (a.max(lo), b.min(hi));
        if a.is_nan() || b.is_nan() || b <= a {
            return 0.0;
        }
        let g = |t: f64| {
            let (p, q) = (t.powi(8), (1.0 - t).powi(8));
            let u = p / (p + q);
            let slope = 8.0 * (t.powi(7) * q + p * (1.0 - t).powi(7)) / (p + q).powi(2);
            let v = d.pdf(a + (b - a) * u) * slope * (b - a);
            if v.is_finite() {
                v
            } else {
                0.0
            }
        };
        let n = 10_000;
        let h = 1.0 / n as f64;
        let mut sum = g(0.0) + g(1.0);
        for i in 1..n {
            sum += g(i as f64 * h) * if i % 2 == 1 { 4.0 } else { 2.0 };
        }
        sum * h / 3.0
    }

    /// The shaded area is the cumulative curve's height at the mark, and the quantile
    /// curve reads that height back to the mark: for every distribution, at marks across
    /// it and beyond its ends. (The shading starts at the chart's left edge: what lies
    /// beyond is the amount the picture writes there.)
    #[test]
    fn the_three_readings_agree_for_every_distribution() {
        let mut checked = 0;
        for (kind, p) in cases() {
            let z = showing(kind, p);
            let d = z.dist();
            assert!(d.valid(), "{kind:?} {p:?}");
            let [x0, x1, y] = z.range;
            assert!(x0.is_finite() && x1.is_finite() && x1 > x0 && y.is_finite() && y > 0.0, "{kind:?} {p:?}: {:?}", z.range);
            let (lo, hi) = d.support();
            let mut marks: Vec<f64> = [0.02, 0.1, 0.5, 0.9, 0.98].iter().map(|q| z.snap_mark(d.quantile(*q))).collect();
            marks.extend([x0 + 0.01 * (x1 - x0), x1 - 0.01 * (x1 - x0)].map(|m| z.snap_mark(m)));
            for m in marks {
                let f = d.cdf(m);
                if kind.counts() {
                    let mut sum = 0.0;
                    let mut k = lo;
                    while k <= m && k <= hi {
                        sum += d.pdf(k);
                        k += 1.0;
                    }
                    assert!((sum - f).abs() < 1e-9, "{kind:?} {p:?} at {m}: bars {sum}, F {f}");
                    // (read back to the bar left of the mark, where that bar has any probability)
                    if m >= lo && f < 1.0 - 1e-12 && d.pdf(m.floor()) > 0.0 {
                        assert_eq!(quantile(&d, f), m.floor(), "{kind:?} {p:?}: Q(F({m}))");
                    }
                } else if m >= hi || m <= lo {
                    // (beyond an end there is all or nothing; a density without bound at that
                    // end has mass closer to it than a number can be, so it is not integrated)
                    assert_eq!(f, if m >= hi { 1.0 } else { 0.0 }, "{kind:?} {p:?} at {m}");
                } else {
                    let shaded = area(&d, x0, m) + d.cdf(x0.min(m));
                    assert!((shaded - f).abs() < 2e-6, "{kind:?} {p:?} at {m}: area {shaded}, F {f}");
                    if m > lo && m < hi && f > 1e-9 && f < 1.0 - 1e-9 {
                        let back = d.quantile(f);
                        assert!((back - m).abs() <= 1e-6 * (1.0 + m.abs()), "{kind:?} {p:?}: Q(F({m})) = {back}");
                    }
                }
                checked += 1;
            }
            // the staircase drawn is the cumulative probability at the whole numbers
            if let Shape::Counts { first, cdf, below, .. } = shape(&d, x0, x1) {
                assert!((below - d.cdf(first - 1.0)).abs() < 1e-12);
                for (i, c) in cdf.iter().enumerate() {
                    assert!((c - d.cdf(first + i as f64)).abs() < 1e-9, "{kind:?} {p:?} at {}", first + i as f64);
                }
            }
        }
        assert!(checked > 400, "{checked}");
    }

    /// The quantile of a count reads its staircase back exactly, where the library's
    /// rounds past it.
    #[test]
    fn a_counts_quantile_reads_its_staircase_back() {
        let g = Dist::Geometric { p: 0.3 };
        assert_eq!(quantile(&g, g.cdf(11.0)), 11.0);
        for d in [g, Dist::Geometric { p: 0.02 }, Dist::Poisson { mean: 4.0 }, Dist::Binomial { n: 30, p: 0.3 }, Dist::Bernoulli { p: 0.5 }] {
            for k in 0..60 {
                let k = k as f64;
                if d.pdf(k) > 0.0 && d.cdf(k) < 1.0 - 1e-12 {
                    assert_eq!(quantile(&d, d.cdf(k)), k, "{d:?} at {k}");
                    // (halfway up the next step)
                    assert_eq!(quantile(&d, d.cdf(k) + 0.5 * d.pdf(k + 1.0)), k + 1.0, "{d:?} just past {k}");
                }
            }
        }
    }

    /// A mark that a change of parameters leaves beyond the charts comes back in.
    #[test]
    fn a_mark_left_outside_comes_back_in() {
        let mut z = showing(Kind::Gamma, [2.0, 1.0]);
        z.marks[0] = 3.3;
        z.set_param(0, 0.2);
        for _ in 0..50 {
            z.step(0.05, 0.05);
        }
        let [x0, x1, _] = z.range;
        assert!(z.marks[0] >= x0 && z.marks[0] <= x1, "{} outside {x0}..{x1}", z.marks[0]);
        let f = z.dist().cdf(z.marks[0]);
        assert!(f > 0.5 && f <= 0.95 + 1e-9, "{f}");
        let mut z = showing(Kind::Binomial, [400.0, 0.5]);
        z.marks = [210.5, 190.5];
        z.set_param(0, 10.0);
        for _ in 0..50 {
            z.step(0.05, 0.05);
        }
        assert!(z.marks.iter().all(|m| *m >= z.range[0] && *m <= z.range[1] && m.fract() == 0.5), "{:?} in {:?}", z.marks, z.range);
    }

    /// Outside the support, the readings say so; a mark of a count snaps between bars.
    #[test]
    fn marks_beyond_the_support_are_read_honestly() {
        let mut z = showing(Kind::Exponential, [1.0, 0.0]);
        z.marks[0] = -0.2;
        let r = z.reading().join(" ");
        assert!(r.contains("No probability lies left of the mark") && r.contains("$Q(0) = 0.00$"), "{r}");
        let mut z = showing(Kind::Uniform, [0.0, 1.0]);
        z.marks[0] = 1.05;
        assert!(z.reading()[0].starts_with("All the probability lies left of the mark"));
        let mut z = showing(Kind::Binomial, [10.0, 0.5]);
        assert_eq!(z.snap_mark(3.9), 3.5);
        z.marks[0] = -0.5;
        assert!(z.reading()[0].starts_with("No bar lies left of the mark"));
        z.marks[0] = 10.5;
        assert!(z.reading()[0].starts_with("Every bar with any probability lies left of the mark"));
        let mut z = showing(Kind::Binomial, [400.0, 1.0]);
        z.marks[0] = 397.5;
        assert!(z.reading()[0].starts_with("No bar with any probability"), "{}", z.reading()[0]);
        let mut z = showing(Kind::Geometric, [1.0, 0.0]);
        z.marks[0] = 6.5;
        assert!(z.reading()[0].contains(r"$P(X \le 6) = 1$"), "{}", z.reading()[0]);
        let mut z = showing(Kind::Poisson, [4.0, 0.0]);
        z.marks[0] = 30.5;
        assert!(z.reading()[0].starts_with("Shaded: all but a sliver"), "{}", z.reading()[0]);
        let mut z = showing(Kind::Normal, [0.0, 0.1]);
        z.marks[0] = 1.0;
        assert!(z.reading()[0].starts_with("Shaded: all but a sliver") && !z.reading()[0].contains("$Q("));
        z.marks[0] = -1.0;
        assert!(z.reading()[0].starts_with("Shaded: next to nothing"), "{}", z.reading()[0]);
        // two marks together: an interval of no width
        let mut z = showing(Kind::Normal, [0.0, 1.0]);
        z.two = true;
        z.marks = [0.5, 0.5];
        assert!(z.reading()[0].contains("probability 0"));
        let mut z = showing(Kind::Poisson, [4.0, 0.0]);
        z.two = true;
        z.marks = [2.5, 2.5];
        assert!(z.reading()[0].contains("no whole number lies between them"));
        z.marks = [2.5, 3.5];
        assert!(z.reading()[0].contains("$P(X = 3) = 0.195$"), "{}", z.reading()[0]);
    }

    /// Whatever the parameters, nothing drawn or written is not a number, and what has no
    /// mean or no variance says so.
    #[test]
    fn degenerate_settings_stay_finite_and_say_what_they_are() {
        for (kind, p) in cases() {
            let mut z = showing(kind, p);
            for road in 0..=roads(kind).len() {
                z.road = road;
                z.step(0.05, 0.05);
                let words = [z.reading().join(" "), z.moments(), kind.settings(p)].join(" ");
                for bad in ["NaN", "inf", "?"] {
                    assert!(!words.contains(bad), "{kind:?} {p:?} road {road}: {words}");
                }
                if let Some((_, Some(g))) = &z.gap {
                    assert!(g.size.is_finite() && g.size >= 0.0 && g.size <= 1.0 && g.at.is_finite(), "{kind:?} {p:?}: {g:?}");
                }
                let [x0, x1, _] = z.range;
                let (main, other) = (shape(&z.dist(), x0, x1), z.other().and_then(|o| o.ok()).map(|o| shape(&o, x0, x1)));
                for s in std::iter::once(&main).chain(other.as_ref()) {
                    for (x, y) in s.cumulative(x0, x1) {
                        assert!(x.is_finite() && (0.0..=1.0 + 1e-9).contains(&y), "{kind:?} {p:?}: ({x}, {y})");
                    }
                    if let Shape::Smooth { pdf, .. } = s {
                        assert!(pdf.iter().all(|v| v.is_finite() && *v >= 0.0), "{kind:?} {p:?}");
                    }
                }
            }
            // draws of every way are numbers
            for way in WAYS {
                let mut rng = Rng::new(3);
                let d = z.dist();
                let x = if way == Way::Story { story_draw(&d, &mut rng).unwrap_or(0.0) } else { d.quantile(stats::unit(&mut rng)) };
                assert!(x.is_finite(), "{kind:?} {p:?} {way:?}");
            }
        }
        let cauchy = showing(Kind::Cauchy, [0.0, 1.0]);
        assert!(cauchy.moments().starts_with("no mean and no standard deviation"));
        let t2 = showing(Kind::StudentT, [2.0, 0.0]);
        assert!(t2.moments().contains("no standard deviation (the variance is infinite)"));
        // the ends of p: one value with all the probability, no road to a spread
        let sure = showing(Kind::Binomial, [10.0, 1.0]);
        assert_eq!(sure.dist().cdf(9.5), 0.0);
        assert_eq!(Road::NormalSame.of(&sure.dist()), Err("no spread to match: all its probability is on one value"));
        assert_eq!(Road::PoissonSame.of(&Dist::Binomial { n: 10, p: 0.0 }), Err("mean 0: no Poisson distribution has it"));
        // a density without bound at 0: the chart's height is set by the middle, not the spike
        for (kind, p) in [(Kind::Gamma, [0.2, 1.0]), (Kind::Beta, [2.0, 0.2]), (Kind::Beta, [0.2, 0.2]), (Kind::ChiSquare, [1.0, 0.0]), (Kind::F, [1.0, 10.0])] {
            let spike = showing(kind, p);
            let d = spike.dist();
            let middle = d.pdf(d.quantile(0.5));
            assert!(spike.range[2].is_finite() && spike.range[2] < 6.0 * middle.max(d.pdf(d.quantile(0.25))).max(d.pdf(d.quantile(0.75))), "{kind:?} {p:?}: {:?}", spike.range);
        }
        // a finite peak is shown whole
        let tall = showing(Kind::LogNormal, [0.0, 1.5]);
        let mode = (-2.25f64).exp();
        assert!(tall.range[2] > tall.dist().pdf(mode));
    }

    /// An experiment shows the same thing whatever was done before it.
    #[test]
    fn each_experiment_sets_everything_it_depends_on() {
        let state = |z: &Zoo| (z.kind, z.p(), z.road, z.marks, z.two, z.draws.want, z.draws.way, z.draws.seed);
        for k in 0..TRIALS.len() {
            let mut a = Zoo::create();
            a.set_up(k);
            let mut b = showing(Kind::Poisson, [20.0, 0.0]);
            b.two = true;
            b.road = 1;
            b.draws.want = 10_000;
            b.draws.way = Way::Story;
            b.set_up((k + 1) % TRIALS.len());
            b.set_up(k);
            assert_eq!(state(&a), state(&b), "experiment {k}");
        }
    }

    /// Counting successes: ten tries, p sweeping from 0.5 to 0.1.
    #[test]
    fn counting_successes_leans() {
        let mut z = Zoo::create();
        z.set_up(0);
        let d = z.dist();
        assert_eq!((d, prob(d.cdf(z.marks[0]))), (Dist::Binomial { n: 10, p: 0.5 }, "0.055".to_string()));
        assert_eq!(d.mean(), Some(5.0));
        z.step(0.0, 0.0);
        let range = z.range;
        let mut frames = 0;
        while z.sweep.is_some() {
            z.step(1.0 / 60.0, 1.0 / 60.0);
            frames += 1;
            assert!(frames < 400);
            // the whole support stays in view all the way
            assert_eq!(z.range, range);
        }
        let d = z.dist();
        assert_eq!(d, Dist::Binomial { n: 10, p: 0.1 });
        assert_eq!(prob(d.cdf(z.marks[0])), "0.930");
        assert!((d.mean().unwrap_or(0.0) - 1.0).abs() < 1e-12);
        // leaning, with a tail to the right: the third standardised moment
        let (m, s) = (1.0, d.sd().unwrap_or(1.0));
        let skew: f64 = (0..=10).map(|k| d.pdf(k as f64) * ((k as f64 - m) / s).powi(3)).sum();
        assert!(skew > 0.8, "{skew}");
        assert!(TRIALS[0].caption.contains("from 0.055 to 0.930") && TRIALS[0].caption.contains("from 5 to 1"));
    }

    /// Where the quantile reads back a count's cumulative probability, the number it is
    /// given gives that count back: never rounded up past the staircase's height (ten
    /// fair tries: F(2) = 56/1024 = 0.0546875, shown as 0.055, but Q(0.055) is 3), and
    /// never further below it than the rounding of the probability shown beside it.
    #[test]
    fn the_quantile_of_a_count_reads_back_its_own_number() {
        let ten = Dist::Binomial { n: 10, p: 0.5 };
        assert_eq!((prob(ten.cdf(2.0)), quantile(&ten, 0.055)), ("0.055".to_string(), 3.0));
        assert_eq!(quantile_arg(&ten, ten.cdf(2.0)), "0.0546875");
        let mut z = Zoo::create();
        z.set_up(0);
        assert!(z.reading()[0].contains("$Q(0.0546875) = 2$") && z.reading_short()[0].contains("$Q(0.0546875) = 2$"), "{:?}", z.reading());
        let mut cut = 0;
        for d in [ten, Dist::Binomial { n: 10, p: 0.1 }, Dist::Binomial { n: 40, p: 0.3 }, Dist::Poisson { mean: 3.0 }, Dist::Poisson { mean: 0.2 }, Dist::Geometric { p: 0.3 }, Dist::Bernoulli { p: 0.3 }] {
            for k in 0..60 {
                let f = d.cdf(k as f64);
                if !(f > 1.0 - FAR && f < FAR) || d.pdf(k as f64) <= 0.0 {
                    continue;
                }
                let shown = quantile_arg(&d, f);
                let v: f64 = shown.trim_end_matches(r"\ldots").parse().unwrap_or(f64::NAN);
                cut += shown.ends_with(r"\ldots") as usize;
                // (a number written exactly may be a hair above F as computed)
                assert!(quantile(&d, v.min(f)) == k as f64 && v <= f + 1e-12 && f - v < 0.0006, "{d:?} at {k}: F = {f}, written {shown}");
            }
        }
        // (some heights are cut short: those of the Poisson do not end)
        assert!(cut > 0);
    }

    /// Rare events: the binomial against the Poisson with its mean.
    #[test]
    fn rare_events_are_nearly_poisson() {
        let mut z = Zoo::create();
        z.set_up(1);
        z.step(0.0, 0.0);
        let g = z.gap.as_ref().and_then(|g| g.1).expect("a gap");
        assert_eq!(gap_text(g.size), "0.0030");
        assert!(z.reading()[1].contains("at most 0.0030 apart"));
        let few = greatest_gap(&Dist::Binomial { n: 10, p: 0.4 }, &Dist::Poisson { mean: 4.0 }).expect("a gap");
        assert_eq!(gap_text(few.size), "0.071");
        let c = TRIALS[1].caption;
        assert!(c.contains("never more than 0.0030 apart") && c.contains("grows to 0.071") && c.contains("the same mean, 4"));
    }

    /// Many tries make a bell, and half the tallest bar is as close as a curve can come.
    #[test]
    fn many_tries_make_a_bell() {
        let mut z = Zoo::create();
        z.set_up(2);
        z.step(0.0, 0.0);
        let d = z.dist();
        let g = z.gap.as_ref().and_then(|g| g.1).expect("a gap");
        let tallest = (0..=100).map(|k| d.pdf(k as f64)).fold(0.0, f64::max);
        assert_eq!(gap_text(g.size), "0.040");
        assert!((g.size - 0.5 * tallest).abs() < 1e-9, "{} against {}", g.size, 0.5 * tallest);
        // no other bell does better: the gap is at least half the tallest step
        for (mean, sd) in [(50.0, 5.0), (49.6, 5.0), (50.4, 4.6), (50.0, 5.5), (50.5, 5.0)] {
            let other = greatest_gap(&d, &Dist::Normal { mean, sd }).expect("a gap");
            assert!(other.size >= 0.5 * tallest - 1e-12, "{mean} {sd}: {}", other.size);
        }
        let lean = Dist::Binomial { n: 100, p: 0.02 };
        let bell = Road::NormalSame.of(&lean).expect("a normal");
        assert_eq!(prob(bell.cdf(0.0)), "0.077");
        assert_eq!(gap_text(greatest_gap(&lean, &bell).expect("a gap").size), "0.18");
        let c = TRIALS[2].caption;
        assert!(c.contains("50 and 5") && c.contains("is 0.040, exactly half the tallest bar") && c.contains("0.077 of its probability below 0") && c.contains("the gap is 0.18"));
    }

    /// Heavy tails: Student's t with 3 degrees of freedom against the standard normal.
    #[test]
    fn the_t_has_heavier_tails() {
        let mut z = Zoo::create();
        z.set_up(3);
        z.step(0.0, 0.0);
        let (t, n) = (z.dist(), Dist::STANDARD_NORMAL);
        let (a, b) = z.interval();
        assert_eq!((a, b), (-3.0, 3.0));
        let outside = |d: &Dist| 1.0 - (d.cdf(b) - d.cdf(a));
        assert_eq!((prob(outside(&t)), prob(outside(&n))), ("0.058".to_string(), "0.0027".to_string()));
        assert_eq!((outside(&t) / outside(&n)).round(), 21.0);
        // alike in the middle: the quartiles within a seventh of each other
        assert!((t.quantile(0.75) / n.quantile(0.75) - 1.0).abs() < 0.14);
        assert_eq!(gap_text(greatest_gap(&t, &n).expect("a gap").size), "0.049");
        assert_eq!(gap_text(greatest_gap(&Dist::StudentT { df: 30.0 }, &n).expect("a gap").size), "0.0052");
        let c = TRIALS[3].caption;
        assert!(c.contains("lies 0.058 of the $t$ and 0.0027 of the normal, 21 times less") && c.contains("from 0.049 to 0.0052"));
        assert!(z.reading()[1].contains("0.0027 outside them"));
    }

    /// The quartiles of a sample, and how far they are from the distribution's in shares.
    fn quartile_misses(d: &Dist, x: &[f64]) -> f64 {
        let sorted = stats::sorted(x);
        [0.25, 0.5, 0.75].iter().map(|p| (d.cdf(stats::quantile_sorted(&sorted, *p)) - p).abs()).fold(0.0, f64::max)
    }

    /// The greatest distance between a sample's cumulative shares and the distribution's.
    fn sample_gap(d: &Dist, x: &[f64]) -> f64 {
        let sorted = stats::sorted(x);
        let n = sorted.len() as f64;
        sorted.iter().enumerate().map(|(i, v)| (d.cdf(*v) - i as f64 / n).abs().max((d.cdf(*v) - (i + 1) as f64 / n).abs())).fold(0.0, f64::max)
    }

    /// Sums of three squared normal numbers have the chi-square's quartiles, and the
    /// chi-square with 3 degrees of freedom is the gamma with shape 3/2 and rate 1/2.
    #[test]
    fn sums_of_squares_are_chi_square() {
        let mut z = Zoo::create();
        z.set_up(4);
        z.settle();
        let d = z.dist();
        assert_eq!(d, Dist::ChiSquare { df: 3.0 });
        let x: Vec<f64> = z.draws.made.iter().map(|m| m.0).collect();
        assert_eq!(x.len(), 1000);
        assert!(z.draws.made.iter().all(|m| m.1.is_nan()), "made by the story, not the quantile curve");
        // a quartile of 1000 draws misses by about 0.014 in share at most, as a rule
        assert!(quartile_misses(&d, &x) < 0.03, "{}", quartile_misses(&d, &x));
        let g = z.gap.as_ref().and_then(|g| g.1).expect("a gap");
        assert!(g.size < NO_GAP);
        assert!(z.reading()[1].contains("no gap at all"));
        assert_eq!(Road::GammaSame.of(&Dist::ChiSquare { df: 2.0 }), Ok(Dist::Gamma { shape: 1.0, rate: 0.5 }));
        assert!(greatest_gap(&Dist::ChiSquare { df: 2.0 }, &Dist::Exponential { rate: 0.5 }).expect("a gap").size < NO_GAP);
        assert!(story_formula(Kind::ChiSquare, &d).contains("Z_1^2 + Z_2^2 + Z_3^2"));
    }

    /// Evenly spread numbers pushed through the quantile curve land in the density's
    /// shape, more closely than random ones.
    #[test]
    fn evenly_spread_numbers_take_the_shape() {
        let first: Vec<f64> = (0..7).map(spread_number).collect();
        assert_eq!(first, vec![0.5, 0.25, 0.75, 0.125, 0.625, 0.375, 0.875]);
        let mut z = Zoo::create();
        z.set_up(5);
        z.settle();
        let d = z.dist();
        let even: Vec<f64> = z.draws.made.iter().map(|m| m.0).collect();
        assert_eq!(even.len(), 1000);
        assert!(quartile_misses(&d, &even) < 0.002, "{}", quartile_misses(&d, &even));
        z.draws.way = Way::Random;
        z.settle();
        let random: Vec<f64> = z.draws.made.iter().map(|m| m.0).collect();
        let (ge, gr) = (sample_gap(&d, &even), sample_gap(&d, &random));
        assert!(gr > 5.0 * ge, "random {gr}, even {ge}");
        // the density at Q(p) is 1 over the quantile curve's slope there
        for p in [0.1, 0.5, 0.9] {
            let h = 1e-6;
            let slope = (d.quantile(p + h) - d.quantile(p - h)) / (2.0 * h);
            assert!((d.pdf(d.quantile(p)) * slope - 1.0).abs() < 1e-5);
        }
        assert!(TRIALS[5].caption.starts_with("1000 evenly spread"));
    }

    /// Draws pushed through the quantile function have the distribution's quartiles, for
    /// every distribution; so do draws made by each story.
    #[test]
    fn draws_have_the_quartiles() {
        for kind in KINDS {
            let specs = kind.params();
            let p = [specs[0].start, specs.get(1).map_or(0.0, |s| s.start)];
            let d = kind.dist(p);
            for way in [Way::Random, Way::Story] {
                if way == Way::Story && !has_story(kind, p) {
                    continue;
                }
                let mut draws = Draws::new();
                draws.way = way;
                draws.rng = Rng::new(11);
                for _ in 0..4000 {
                    draws.make(&d);
                }
                let x: Vec<f64> = draws.made.iter().map(|m| m.0).collect();
                if kind.counts() {
                    // a count's quartile is a whole number: the share at or below it reaches p
                    let sorted = stats::sorted(&x);
                    for q in [0.25, 0.5, 0.75] {
                        let v = stats::quantile_sorted(&sorted, q);
                        assert!(d.cdf(v) >= q - 0.03 && d.cdf(v - 1.0) <= q + 0.03, "{kind:?} {way:?} at {q}: {v}");
                    }
                } else {
                    assert!(quartile_misses(&d, &x) < 0.025, "{kind:?} {way:?}: {}", quartile_misses(&d, &x));
                }
            }
        }
    }

    /// A story is used only where it is exactly right: a gamma whose shape is not a whole
    /// number is drawn through its quantile curve, not as a sum of waits.
    #[test]
    fn a_story_is_told_only_where_it_is_true() {
        let mut draws = Draws::new();
        draws.way = Way::Story;
        let d = Dist::Gamma { shape: 2.5, rate: 1.0 };
        assert!(!story_fits(&d) && story_fits(&Dist::Gamma { shape: 3.0, rate: 1.0 }));
        for _ in 0..4000 {
            draws.make(&d);
        }
        assert!(draws.made.iter().all(|m| !m.1.is_nan()), "through the quantile curve");
        let x: Vec<f64> = draws.made.iter().map(|m| m.0).collect();
        assert!(quartile_misses(&d, &x) < 0.025);
        assert!(!has_story(Kind::Beta, [2.0, 5.5]) && has_story(Kind::Beta, [2.0, 5.0]) && !has_story(Kind::Normal, [0.0, 1.0]));
    }

    /// The phone's short reading says what the long one says, in fewer words.
    #[test]
    fn the_short_reading_agrees() {
        for k in 0..TRIALS.len() {
            let mut z = Zoo::create();
            z.set_up(k);
            z.step(0.0, 0.0);
            let (long, short) = (z.reading().join(" "), z.reading_short().join(" "));
            for bad in ["NaN", "inf", "?"] {
                assert!(!short.contains(bad), "{short}");
            }
            // every probability in the short one is in the long one
            for word in short.split(|c: char| !(c.is_ascii_digit() || c == '.')).filter(|w| w.contains('.') && w.len() > 2) {
                let word = word.trim_end_matches('.');
                assert!(long.contains(word), "experiment {k}: {word} of {short} not in {long}");
            }
        }
        let z = Zoo::create();
        assert_eq!(z.reading_short()[0], r"Shaded: $P(X \le 1.00) = 0.841$, the curve's height at 1.00; $Q(0.841) = 1.00$.");
    }

    /// The opening reads the standard normal at one standard deviation above its mean.
    #[test]
    fn the_opening_reads_one_standard_deviation() {
        let z = Zoo::create();
        assert_eq!(z.dist(), Dist::STANDARD_NORMAL);
        let r = z.reading()[0].clone();
        assert!(r.contains(r"$P(X \le 1.00) = 0.841$") && r.contains("$Q(0.841) = 1.00$"), "{r}");
        assert!(!z.animated());
    }

    /// After a change the picture glides to the new one and comes to rest, at any rate.
    #[test]
    fn glides_arrive_and_stop() {
        for hz in [30.0f32, 60.0, 144.0] {
            for k in 0..TRIALS.len() {
                let mut z = Zoo::create();
                z.set_up(k);
                let mut frames = 0;
                while z.animated() {
                    z.step(1.0 / hz, 1.0 / hz);
                    frames += 1;
                    assert!(frames as f32 <= 20.0 * hz, "experiment {k} still moving after 20 s at {hz} Hz");
                }
                assert_eq!(z.shown_params.shown, z.p().map(|v| v as f32).to_vec());
                assert_eq!(z.shown_range.shown, z.range.map(|v| v as f32).to_vec());
                assert_eq!(z.draws.landed, z.draws.want);
            }
        }
    }

    /// The charts fit every window, one under another on a phone.
    #[test]
    fn the_layout_fits_every_window() {
        let width = |t: &str, s: f32| text_width(t, s);
        for (w, h, inset) in [(1078.0, 1000.0, 0.0), (1600.0, 1000.0, 0.0), (800.0, 600.0, 0.0), (390.0, 800.0, 44.0), (390.0, 756.0, 44.0), (600.0, 900.0, 0.0), (240.0, 180.0, 0.0), (1.0, 1.0, 0.0), (3000.0, 2000.0, 0.0)] {
            for tab in 0..3 {
                let lay = Layout::of(Vec2::new(w, h), inset, tab, &width);
                let rects = [lay.density, lay.cdf, lay.quantile, lay.words];
                assert!(rects.iter().chain(&lay.kinds).all(|(a, b)| a.is_finite() && b.is_finite()), "{w}x{h}");
                if lay.tiny {
                    continue;
                }
                for (a, b) in rects.iter().chain(&lay.kinds) {
                    assert!(a.x >= 0.0 && a.y >= inset && b.x <= w + 0.5 && b.y <= h + 0.5 && b.x > a.x && b.y > a.y, "{w}x{h} tab {tab}: {a} {b}");
                }
                let below_buttons = lay.kinds.iter().map(|r| r.1.y).fold(0.0, f32::max);
                assert!(lay.name.0.y >= below_buttons, "{w}x{h}");
                assert!(lay.density.0.y > lay.name.0.y + 2.0 * (lay.big + 5.0), "{w}x{h}");
                // one chart above the next, with room for numbers between
                assert!(lay.density.1.y + 16.0 < lay.cdf.0.y, "{w}x{h}");
                if lay.narrow {
                    assert!(lay.cdf.1.y + 16.0 < lay.quantile.0.y && lay.quantile.1.y < lay.words.0.y, "{w}x{h}");
                    assert!(lay.density.1.y - lay.density.0.y > 50.0, "{w}x{h}");
                } else {
                    assert!(lay.cdf.1.x + 60.0 < lay.quantile.0.x && lay.words.1.y < lay.quantile.0.y - 40.0, "{w}x{h}");
                    assert!(lay.density.1.y - lay.density.0.y > 120.0, "{w}x{h}");
                }
            }
        }
    }

    /// The numbers are written as the captions write them.
    #[test]
    fn numbers_in_words() {
        assert_eq!((prob(0.0), prob(1.0), prob(0.0027), prob(0.000_27), prob(0.841_34), prob(0.999_95)), ("0".into(), "1".into(), "0.0027".into(), "2.7e-4".into(), "0.841".into(), "0.999950".into()));
        assert_eq!((prob(0.9995), prob(1.0 - 1e-12)), ("0.99950".into(), "1".into()));
        assert_eq!((gap_text(0.039_795), gap_text(0.002_955), gap_text(0.176_7), gap_text(1e-12)), ("0.040".into(), "0.0030".into(), "0.18".into(), "0".into()));
        assert_eq!(Kind::Binomial.settings([10.0, 0.5]), "$n = 10$, $p = 0.5$");
        assert_eq!(Kind::StudentT.settings([2.7, 0.0]), "$k = 3$");
        assert_eq!(Kind::Binomial.settings([10.0, 0.003]), "$n = 10$, $p = 0.003$");
        assert_eq!(Kind::Geometric.settings([0.02, 0.0]), "$p = 0.02$");
        assert_eq!(Kind::Normal.settings([-1.25, 0.35]), r"$\mu = −1.25$, $\sigma = 0.35$");
        assert_eq!(ordinal(2), "2nd");
        assert_eq!(words_of("a $b c$  d"), vec!["a", "$b c$", "d"]);
    }
}

//! Hypothesis tests: assume nothing is going on, see where a statistic would fall if
//! that were so, and ask how surprising the one observed is. The p-value, the two kinds
//! of error, and what the power of a test depends on.
//!
//! The data are a sample of n points from a bell with mean μ and spread σ; the null
//! hypothesis says μ = μ₀ (0 here). The statistic is the distance of the sample's mean
//! from μ₀ in standard errors: with σ known, z = (x̄ − μ₀)/(σ/√n), standard normal under
//! the null; with σ estimated by the sample's own s, t = (x̄ − μ₀)/(s/√n), Student's t
//! with n − 1 degrees of freedom. When the effect μ − μ₀ is real, z is normal about
//! δ = (μ − μ₀)√n/σ, and t follows the noncentral t distribution of noncentrality δ,
//! worked out here by a sum over the distribution of s/σ ([`Spread`]).

use super::plot::{self, Glide, Plot, Rect, AXES, CURVES, MARKS, OWN};
use super::*;
use crate::clock::Pace;
use crate::math::stats::{self as st, normal_cdf, Dist};

pub const DEF: SceneDef = SceneDef {
    id: "testing",
    title: "Hypothesis tests",
    group: TESTING,
    blurb: "A null hypothesis, a test statistic and its $p$-value: the two kinds of error, and what the power of a test depends on.",
    make: |_| Box::new(Testing::create()),
};

// ----- the test ------------------------------------------------------------------------

/// The mean the null hypothesis says the data have.
const MU0: f64 = 0.0;

/// Where the null is rejected.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Tails {
    /// Far out on either side: the alternative is that μ is not μ₀.
    Two,
    /// Far out above only: the alternative is that μ is above μ₀.
    Upper,
}

/// Everything the test and its picture depend on, but the sample.
#[derive(Clone, Copy, Debug, PartialEq)]
struct Setup {
    /// The true mean less μ₀, in the data's units.
    effect: f64,
    /// The true spread of the data.
    sigma: f64,
    n: usize,
    alpha: f64,
    tails: Tails,
    /// The spread estimated from the sample (Student's t) rather than known (z).
    estimated: bool,
    /// In a run of many studies, the share of them whose effect is real; the rest have
    /// none (their μ is μ₀).
    real: f64,
}

/// Why there is no test.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum NoTest {
    /// σ = 0: every point is μ exactly.
    NoNoise,
    /// One point and the spread to be estimated from it.
    OnePoint,
}

/// The sizes of sample the controls reach.
const MOST_N: usize = 40_000;

impl Setup {
    /// Kept to the controls' ranges.
    fn kept(self) -> Setup {
        let fix = |v: f64, lo: f64, hi: f64, or: f64| if v.is_finite() { v.max(lo).min(hi) } else { or };
        Setup {
            effect: fix(self.effect, -2.0, 2.0, 0.0),
            sigma: fix(self.sigma, 0.0, 3.0, 1.0),
            n: self.n.clamp(1, MOST_N),
            alpha: fix(self.alpha, 0.0, 1.0, 0.05),
            real: fix(self.real, 0.0, 1.0, 1.0),
            ..self
        }
    }

    fn no_test(&self) -> Option<NoTest> {
        if self.sigma.is_nan() || self.sigma <= 0.0 {
            Some(NoTest::NoNoise)
        } else if self.estimated && self.n < 2 {
            Some(NoTest::OnePoint)
        } else {
            None
        }
    }

    /// The degrees of freedom of Student's t (with the spread estimated).
    fn df(&self) -> Option<f64> {
        (self.estimated && self.n >= 2).then_some(self.n as f64 - 1.0)
    }

    /// The statistic's distribution if the null is true.
    fn null(&self) -> Dist {
        match self.df() {
            Some(df) => Dist::StudentT { df },
            None => Dist::STANDARD_NORMAL,
        }
    }

    /// How far the effect moves the statistic: δ = (μ − μ₀)√n/σ, the centre of z when
    /// the effect is real (and the noncentrality of t).
    fn delta(&self) -> f64 {
        if self.sigma > 0.0 {
            self.effect * (self.n as f64).sqrt() / self.sigma
        } else {
            0.0
        }
    }

    /// The critical values: at or below the first, or at or above the second, the null
    /// is rejected (−∞ and +∞ where there is no such side).
    fn critical(&self) -> (f64, f64) {
        let null = self.null();
        match self.tails {
            Tails::Two => {
                let c = -null.quantile(self.alpha / 2.0);
                (-c, c)
            }
            Tails::Upper => (f64::NEG_INFINITY, -null.quantile(self.alpha)),
        }
    }

    /// The chance, if the null is true, of a statistic at least as far out as `stat`
    /// (on the side or sides the test looks at). Both nulls are symmetric about 0, so
    /// each tail is worked out from below, where it is accurate even when tiny.
    fn p_value(&self, stat: f64) -> f64 {
        let null = self.null();
        match self.tails {
            Tails::Two => (2.0 * null.cdf(-stat.abs())).min(1.0),
            Tails::Upper => null.cdf(-stat),
        }
    }

    fn rejects(&self, stat: f64) -> bool {
        let (lo, hi) = self.critical();
        stat <= lo || stat >= hi
    }

    /// The spread estimate's distribution, for sums over it.
    fn spread(&self, intervals: usize) -> Spread {
        match self.df() {
            Some(df) => Spread::estimated(df, intervals),
            None => Spread::known(),
        }
    }

    /// The chance of rejecting the null when the effect is as set: the power. (With
    /// no effect, it is α.)
    fn power(&self) -> f64 {
        if self.no_test().is_some() {
            return f64::NAN;
        }
        let (lo, hi) = self.critical();
        // (with few degrees of freedom the critical value is large and W spreads wide:
        // what is summed changes over a short stretch of it, and needs more points;
        // with many, W is narrow and a few dozen do)
        let intervals = match self.n {
            0..12 => 800,
            12..40 => 200,
            _ => 60,
        };
        let spread = self.spread(intervals);
        (spread.above(hi, self.delta()) + spread.below(lo, self.delta())).min(1.0)
    }

    /// The least n whose power reaches `target`, as far as ten million (None if not
    /// even that: no effect, or an effect on the side a one-tailed test does not look).
    fn n_for(&self, target: f64) -> Option<usize> {
        if self.sigma.is_nan() || self.sigma <= 0.0 {
            return None;
        }
        let at = |n: usize| Setup { n, ..*self }.power();
        let least = if self.estimated { 2 } else { 1 };
        // (with no effect the power is α at every n)
        if self.effect == 0.0 {
            return (self.alpha >= target).then_some(least);
        }
        // Start from a guess: for t, the answer for z, which is cheap to find and
        // close (the power of t is a little less at the same n).
        let guess = if self.estimated { Setup { estimated: false, ..*self }.n_for(target).unwrap_or(least).max(least) } else { least };
        let (mut lo, mut hi);
        if at(guess) >= target {
            if guess == least || at(least) >= target {
                return Some(least);
            }
            (lo, hi) = (least, guess);
        } else {
            // (the power grows with n: step up, in steps that double, until it is reached)
            let mut step = 4;
            lo = guess;
            loop {
                hi = lo + step;
                if hi > 10_000_000 {
                    return None;
                }
                if at(hi) >= target {
                    break;
                }
                (lo, step) = (hi, step * 2);
            }
        }
        // then halve the gap: the power falls short at `lo` and reaches it at `hi`
        while hi - lo > 1 {
            let mid = (lo + hi) / 2;
            if at(mid) >= target {
                hi = mid;
            } else {
                lo = mid;
            }
        }
        Some(hi)
    }
}

/// The spread estimate as a share of the true spread, W = s/σ, as points and weights
/// for sums over its distribution. With s from k degrees of freedom, W² is a chi-square
/// variable divided by k: its density is proportional to w^(k − 1) exp(−k w²/2).
/// Simpson's rule over a range that holds all of it that matters, the weights scaled
/// to add up to exactly 1. With σ known, W is 1.
#[derive(Clone, Debug)]
struct Spread {
    nodes: Vec<(f64, f64)>,
}

impl Spread {
    fn known() -> Spread {
        Spread { nodes: vec![(1.0, 1.0)] }
    }

    fn estimated(df: f64, intervals: usize) -> Spread {
        if !(1.0..1e9).contains(&df) {
            return Spread::known();
        }
        // (W is near 1 give or take 1/√(2k); its right tail is the longer)
        let s = (0.5 / df).sqrt();
        let (a, b) = ((1.0 - 10.0 * s).max(0.0), 1.0 + 14.0 * s);
        let m = (intervals.max(2) / 2) * 2;
        let h = (b - a) / m as f64;
        // the logarithm of the density, less its value at the mode (no constants
        // needed: the weights are scaled to add up to 1)
        let k = df - 1.0;
        let mode = (k / df).sqrt();
        let half = 0.5 * df;
        let log_density = |w: f64| {
            let rise = if k > 0.0 {
                if w > 0.0 {
                    k * (w / mode).ln()
                } else {
                    f64::NEG_INFINITY
                }
            } else {
                0.0
            };
            rise - half * (w * w - mode * mode)
        };
        let mut nodes: Vec<(f64, f64)> = (0..=m)
            .map(|i| {
                let w = a + h * i as f64;
                let rule = if i == 0 || i == m {
                    1.0
                } else if i % 2 == 1 {
                    4.0
                } else {
                    2.0
                };
                (w, rule * log_density(w).exp())
            })
            .collect();
        let total: f64 = nodes.iter().map(|n| n.1).sum();
        if !(total > 0.0 && total.is_finite()) {
            return Spread::known();
        }
        for n in &mut nodes {
            n.1 /= total;
        }
        Spread { nodes }
    }

    /// The chance that the statistic (Z + δ)/W is at least `c`: that Z ≥ cW − δ.
    fn above(&self, c: f64, delta: f64) -> f64 {
        if c == f64::NEG_INFINITY {
            return 1.0;
        }
        if c.is_nan() || c == f64::INFINITY {
            return 0.0;
        }
        self.nodes.iter().map(|&(w, p)| p * normal_cdf(delta - c * w)).sum()
    }

    /// The chance that it is at most `c`.
    fn below(&self, c: f64, delta: f64) -> f64 {
        if c == f64::INFINITY {
            return 1.0;
        }
        if c.is_nan() || c == f64::NEG_INFINITY {
            return 0.0;
        }
        self.nodes.iter().map(|&(w, p)| p * normal_cdf(c * w - delta)).sum()
    }

    /// Its density at x: given W = w, the statistic is (Z + δ)/w, of density w φ(xw − δ).
    fn density(&self, x: f64, delta: f64) -> f64 {
        self.nodes.iter().map(|&(w, p)| p * w * st::normal_pdf(x * w - delta)).sum()
    }
}

// ----- numbers as they are written -----------------------------------------------------

/// A probability as the scene writes it: two significant digits for a small one, never
/// 0 nor 1 for one that is not exactly that ("< 0.0001", "> 0.9999").
fn chance(p: f64) -> String {
    if p.is_nan() {
        return "?".to_string();
    }
    if p <= 0.0 {
        return "0".to_string();
    }
    if p >= 1.0 {
        return "1".to_string();
    }
    if p < 1e-4 {
        return r"< 0.0001".to_string();
    }
    if p > 0.9999 {
        return r"> 0.9999".to_string();
    }
    let digits = ((-p.log10()).ceil() as usize + 1).clamp(2, 5);
    let text = num(p, digits);
    // (a share just short of 1 is not written as 1)
    if text.starts_with('1') {
        return num(p, 4);
    }
    text
}

/// A p-value as the scene writes it: as [`chance`], and "< 0.0001" also where it is
/// too small for the arithmetic to tell from 0 (it never is 0, for a statistic that is
/// a number).
fn p_value_text(p: f64) -> String {
    if (0.0..1e-4).contains(&p) {
        return "< 0.0001".to_string();
    }
    chance(p)
}

/// A number for a read-out: plain, but a bound ("< 0.0001") is a formula.
fn readable(text: String) -> String {
    if text.starts_with('<') || text.starts_with('>') {
        format!("${text}$")
    } else {
        text
    }
}

/// "= 0.0057" or "< 0.0001": what follows a name in a formula.
fn is(p: f64) -> String {
    let p = if p == 0.0 { f64::MIN_POSITIVE } else { p };
    let text = chance(p);
    if text.starts_with('<') || text.starts_with('>') {
        text
    } else {
        format!("= {text}")
    }
}

/// α as the controls set it: as few digits as it needs.
fn alpha_text(a: f64) -> String {
    if a <= 0.0 {
        return "0".to_string();
    }
    let digits = ((-a.log10()).floor() as usize + 2).clamp(1, 5);
    let text = format!("{a:.digits$}");
    let text = text.trim_end_matches('0').trim_end_matches('.');
    if text.is_empty() {
        "0".to_string()
    } else {
        text.to_string()
    }
}

// ----- the sample ----------------------------------------------------------------------

/// The standard normal numbers a sample is made of, drawn from a seed: point i of a
/// sample is μ + σ zᵢ, so a larger sample keeps the points of a smaller one, and a
/// change of μ or σ moves the same points.
#[derive(Clone, Debug)]
struct Draws {
    seed: u64,
    rng: Rng,
    z: Vec<f64>,
    /// The mean and the spread (dividing by n − 1) of the first `of` of them.
    of: usize,
    mean: f64,
    sd: f64,
}

impl Draws {
    fn new(seed: u64) -> Draws {
        Draws { seed, rng: Rng::new(seed), z: Vec::new(), of: 0, mean: 0.0, sd: 0.0 }
    }

    /// Make sure there are `n`, and their mean and spread are worked out.
    fn ensure(&mut self, n: usize) {
        while self.z.len() < n {
            self.z.push(st::normal(&mut self.rng));
        }
        if self.of != n {
            let first = &self.z[..n.min(self.z.len())];
            (self.of, self.mean, self.sd) = (n, st::mean(first), st::sd(first));
        }
    }

    fn first(&self, n: usize) -> &[f64] {
        &self.z[..n.min(self.z.len())]
    }
}

/// What this sample says: its mean and spread, the statistic and its p-value.
#[derive(Clone, Copy, Debug, PartialEq)]
struct Observed {
    /// The statistic (the sample's own, or set by hand) and its p-value.
    stat: f64,
    p: f64,
    rejects: bool,
    /// The sample's mean and spread estimate (with the shift), and the standard error.
    xbar: f64,
    s: f64,
    se: f64,
    /// How far every point has been moved to make the statistic the one set by hand.
    shift: f64,
}

// ----- many studies --------------------------------------------------------------------

/// The most studies a run makes.
const MOST_STUDIES: usize = 10_000;
/// Samples up to this size are drawn point by point; larger ones go by their mean and
/// spread estimate, which for bell-shaped data have exactly known distributions
/// (normal, and σ² times a chi-square over its degrees of freedom, independent of each
/// other), so that a run of samples of 40 000 is as quick as one of 20.
const DIRECT: usize = 1000;
/// The histogram of the studies' statistics: bins of this width from −16 to 16.
const BIN: f64 = 0.2;
const BINS: usize = 160;
const BINS_FROM: f64 = -16.0;

/// What a run of studies depends on: the effect, σ, n, the share of real effects, the
/// seed (their bits).
type StudyKey = [u64; 5];

fn study_key(s: &Setup, seed: u64) -> StudyKey {
    [s.effect.to_bits(), s.sigma.to_bits(), s.n as u64, s.real.to_bits(), seed]
}

/// One study: whether its effect is real, and its z and t (t is not a number for a
/// sample of one).
fn study(s: &Setup, seed: u64, k: usize) -> (bool, f64, f64) {
    let mut rng = Rng::new(seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) ^ (k as u64).wrapping_mul(0xD1B5_4A32_D192_ED03));
    let real = st::unit(&mut rng) < s.real && s.effect != 0.0;
    let mu = MU0 + if real { s.effect } else { 0.0 };
    let n = s.n.max(1);
    let (mean, sd) = if n <= DIRECT {
        // (by Welford's running sums: the points need not be kept)
        let (mut m, mut q) = (0.0, 0.0);
        for i in 0..n {
            let x = mu + s.sigma * st::normal(&mut rng);
            let d = x - m;
            m += d / (i + 1) as f64;
            q += d * (x - m);
        }
        (m, if n > 1 { (q / (n - 1) as f64).sqrt() } else { f64::NAN })
    } else {
        let m = mu + s.sigma / (n as f64).sqrt() * st::normal(&mut rng);
        let df = (n - 1) as f64;
        (m, s.sigma * (Dist::ChiSquare { df }.sample(&mut rng) / df).sqrt())
    };
    let root = (n as f64).sqrt();
    (real, (mean - MU0) / (s.sigma / root), (mean - MU0) / (sd / root))
}

/// Studies of no effect and of a real one, rejected or not: the classic table.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
struct Tally {
    /// [no effect, real effect] × [rejected, not rejected].
    counts: [[usize; 2]; 2],
}

impl Tally {
    fn of(&self, real: bool) -> usize {
        self.counts[real as usize][0] + self.counts[real as usize][1]
    }

    fn rejected(&self, real: bool) -> usize {
        self.counts[real as usize][0]
    }

    /// The share of the row rejected (None with no studies in it).
    fn share(&self, real: bool) -> Option<f64> {
        let all = self.of(real);
        (all > 0).then(|| self.rejected(real) as f64 / all as f64)
    }

    /// Of all the rejections, the share whose null was true.
    fn false_share(&self) -> Option<f64> {
        let all = self.rejected(false) + self.rejected(true);
        (all > 0).then(|| self.rejected(false) as f64 / all as f64)
    }
}

/// A run of many studies, each a new sample from the truth, kept so that a change of α,
/// of the tails or of z and t counts the same studies again.
#[derive(Clone, Debug)]
struct Studies {
    key: StudyKey,
    real: Vec<bool>,
    z: Vec<f64>,
    t: Vec<f64>,
    /// What they were last counted by (α, tails, t), and the count.
    rule: Option<(u64, Tails, bool)>,
    critical: (f64, f64),
    tally: Tally,
    /// Their statistics in bins, [no effect, real effect].
    bins: [Vec<u32>; 2],
}

impl Studies {
    fn new() -> Studies {
        Studies { key: [0; 5], real: Vec::new(), z: Vec::new(), t: Vec::new(), rule: None, critical: (0.0, 0.0), tally: Tally::default(), bins: [vec![0; BINS], vec![0; BINS]] }
    }

    fn len(&self) -> usize {
        self.z.len()
    }

    fn done(&self) -> bool {
        self.len() >= MOST_STUDIES
    }

    /// Start afresh if what the studies depend on has changed; count again if only
    /// what they are counted by has.
    fn follow(&mut self, s: &Setup, seed: u64) {
        let key = study_key(s, seed);
        if key != self.key {
            self.key = key;
            self.real.clear();
            self.z.clear();
            self.t.clear();
            self.rule = None;
        }
        let rule = (s.alpha.to_bits(), s.tails, s.estimated);
        if self.rule != Some(rule) {
            self.rule = Some(rule);
            self.critical = s.critical();
            self.tally = Tally::default();
            self.bins.iter_mut().for_each(|b| b.iter_mut().for_each(|c| *c = 0));
            for k in 0..self.len() {
                self.count(k, s.estimated);
            }
        }
    }

    fn count(&mut self, k: usize, estimated: bool) {
        let stat = if estimated { self.t[k] } else { self.z[k] };
        let real = self.real[k] as usize;
        let (lo, hi) = self.critical;
        let rejected = stat <= lo || stat >= hi;
        self.tally.counts[real][if rejected { 0 } else { 1 }] += 1;
        let bin = (stat - BINS_FROM) / BIN;
        if bin >= 0.0 && bin < BINS as f64 {
            self.bins[real][bin as usize] += 1;
        }
    }

    /// One more study.
    fn add(&mut self, s: &Setup, seed: u64) {
        let (real, z, t) = study(s, seed, self.len());
        self.real.push(real);
        self.z.push(z);
        self.t.push(t);
        self.count(self.len() - 1, s.estimated);
    }

    /// The histogram's heights as densities (a share of all studies per unit), [no effect…, real effect…].
    fn heights(&self) -> Vec<f32> {
        let total = self.len().max(1) as f64 * BIN;
        self.bins.iter().flatten().map(|&c| (c as f64 / total) as f32).collect()
    }
}

// ----- the experiments -----------------------------------------------------------------

struct Trial {
    button: &'static str,
    caption: &'static str,
    effect: f64,
    n: usize,
    alpha: f64,
    tails: Tails,
    real: f64,
    /// The seed of the sample shown.
    seed: u64,
}

/// Every experiment has σ = 1 known (z), runs many studies at the seed `RUN_SEED`, and
/// shows the sample of its own seed.
const TRIALS: [Trial; 7] = [
    Trial {
        button: "nothing going on",
        caption: r"The true effect is zero: $\mu = \mu_0$. Yet this sample's $p$ is 0.015, below $\alpha = 0.05$: the null is rejected wrongly, a false alarm, which looks exactly like a discovery. Of many studies, the share that reject settles near 0.05 (the table's top row: 0.054 after 10 000 studies; the rest is chance). When nothing is going on, $\alpha$ is the rate of false alarms.",
        effect: 0.0,
        n: 20,
        alpha: 0.05,
        tails: Tails::Two,
        real: 1.0,
        seed: 159,
    },
    Trial {
        button: "a real effect, too little data",
        caption: r"An effect of half a standard deviation, $n = 20$, two tails, $\alpha = 0.05$: the power is 0.61, so about four studies in ten miss this real effect, and this sample is one of them ($p = 0.25$). With the spread estimated (switch to $t$) the power is 0.56: nearly half miss.",
        effect: 0.5,
        n: 20,
        alpha: 0.05,
        tails: Tails::Two,
        real: 1.0,
        seed: 117,
    },
    Trial {
        button: "enough data",
        caption: r"The same effect with $n = 32$, the least $n$ that gives power 0.8 with $\sigma$ known (it is 0.81). The green curve has moved out from $\delta = 2.24$ to 2.83, and most of it now lies beyond the critical value. With the spread estimated it takes $n = 34$.",
        effect: 0.5,
        n: 32,
        alpha: 0.05,
        tails: Tails::Two,
        real: 1.0,
        seed: 23,
    },
    Trial {
        button: "significant is not large",
        caption: r"An effect of 0.02 standard deviations and $n = 40\,000$. On the strip of data the mean can hardly be told from $\mu_0$, yet $p < 0.0001$, and 98 % of such studies reject. A small $p$ says that data like these would be rare if the effect were exactly zero; it does not say that the effect is large. The effect size, 0.02, says how large.",
        effect: 0.02,
        n: 40_000,
        alpha: 0.05,
        tails: Tails::Two,
        real: 1.0,
        seed: 9,
    },
    Trial {
        button: "one tail or two",
        caption: r"The second experiment's effect, another sample, and one tail: the alternative, chosen before looking, is that $\mu$ is above $\mu_0$. The critical value moves from 1.96 to 1.64 and the $p$-value halves, from 0.068 with two tails (click “two tails”) to 0.034. Choosing the tail after seeing which way the data went is a two-tailed test at $\alpha = 0.10$: twice the false alarms.",
        effect: 0.5,
        n: 20,
        alpha: 0.05,
        tails: Tails::Upper,
        real: 1.0,
        seed: 50,
    },
    Trial {
        button: "a stricter threshold",
        caption: r"$\alpha = 0.005$ instead of 0.05, with the second experiment's effect: the critical value moves out to 2.81, and this sample's $p = 0.017$, which would have rejected at 0.05, no longer does. When nothing is going on, one study in 200 now rejects instead of one in 20; but the power falls from 0.61 to 0.28. Power 0.8 now takes $n = 54$.",
        effect: 0.5,
        n: 20,
        alpha: 0.005,
        tails: Tails::Two,
        real: 1.0,
        seed: 10,
    },
    Trial {
        button: "what a $p$-value is not",
        caption: r"Now only one study in ten has a real effect (half a standard deviation, $n = 32$, power 0.81); the other nine have none. Of the studies that reject, about 36 % are false alarms: $0.9 \times 0.05 = 0.045$ of all studies against $0.1 \times 0.81 = 0.081$. So a result with $p < 0.05$ comes here from a true null more than a third of the time. That depends on how many of the hypotheses tested are true, which $p$ cannot know.",
        effect: 0.5,
        n: 32,
        alpha: 0.05,
        tails: Tails::Two,
        real: 0.1,
        seed: 23,
    },
];

/// The scene opens as the second experiment sets it up.
const OPENING: usize = 1;
/// The seed of the runs of studies the experiments make.
const RUN_SEED: u64 = 1;

// ----- the scene -----------------------------------------------------------------------

pub struct Testing {
    setup: Setup,
    seed: u64,
    draws: Draws,
    /// The statistic set by hand (dragged), if it is not the sample's own.
    held: Option<f64>,
    /// Was the statistic being dragged in the last frame?
    dragging: bool,
    running: bool,
    rate: f32,
    run_seed: u64,
    studies: Studies,
    pace: Pace,
    /// The power and the n for power 0.8, for the setup they were worked out for.
    worked: Option<(Setup, f64, Option<usize>)>,
    look: Look,
    curves: Curves,
    /// A large sample in bins (see `sample_bins`), and what it was binned for: the
    /// seed, n, the effect, σ and the strip's range.
    binned: ([u64; 6], Vec<f32>),
    /// The window last drawn in (a new one is drawn at once, without gliding).
    canvas: Option<Vec2>,
}

impl Testing {
    /// (Nothing here needs the graphics device, so the tests can make one too.)
    fn create() -> Testing {
        let mut s = Testing {
            setup: Setup { effect: 0.5, sigma: 1.0, n: 20, alpha: 0.05, tails: Tails::Two, estimated: false, real: 1.0 },
            seed: 0,
            draws: Draws::new(0),
            held: None,
            dragging: false,
            running: true,
            rate: 250.0,
            run_seed: RUN_SEED,
            studies: Studies::new(),
            pace: Pace::default(),
            worked: None,
            look: Look::default(),
            curves: Curves::default(),
            binned: ([0; 6], Vec::new()),
            canvas: None,
        };
        s.set_up(OPENING);
        s
    }

    /// Set up one of the experiments in `TRIALS`, and run many studies of it.
    fn set_up(&mut self, k: usize) {
        let t = &TRIALS[k];
        self.setup = Setup { effect: t.effect, sigma: 1.0, n: t.n, alpha: t.alpha, tails: t.tails, estimated: false, real: t.real };
        self.seed = t.seed;
        self.held = None;
        self.run_seed = RUN_SEED;
        self.running = true;
        self.refresh();
    }

    /// Bring what depends on the settings up to date with them.
    fn refresh(&mut self) {
        self.setup = self.setup.kept();
        if self.draws.seed != self.seed {
            self.draws = Draws::new(self.seed);
        }
        self.draws.ensure(self.setup.n);
        self.studies.follow(&self.setup, self.run_seed);
        if self.worked.is_none_or(|(s, _, _)| s != self.setup) {
            self.worked = Some((self.setup, self.setup.power(), self.setup.n_for(0.8)));
        }
    }

    fn power(&self) -> f64 {
        self.worked.map_or(f64::NAN, |w| w.1)
    }

    fn n_for(&self) -> Option<usize> {
        self.worked.and_then(|w| w.2)
    }

    /// The sample shown, as it stands.
    fn observed(&self) -> Option<Observed> {
        let s = &self.setup;
        if s.no_test().is_some() {
            return None;
        }
        let root = (s.n as f64).sqrt();
        let (mean, spread) = (self.draws.mean, self.draws.sd);
        let own_mean = MU0 + s.effect + s.sigma * mean;
        let sd = s.sigma * spread;
        let se = if s.estimated { sd / root } else { s.sigma / root };
        if !(se > 0.0 && se.is_finite()) {
            return None;
        }
        let own = (own_mean - MU0) / se;
        let stat = self.held.filter(|h| h.is_finite()).unwrap_or(own);
        let shift = (stat - own) * se;
        Some(Observed { stat, p: s.p_value(stat), rejects: s.rejects(stat), xbar: own_mean + shift, s: sd, se, shift })
    }

    /// Is a run of studies going on (and not finished)?
    fn studying(&self) -> bool {
        self.running && !self.studies.done() && self.setup.no_test().is_none()
    }

    /// The studies of this frame: so many a second of the scene's time, within the
    /// frame's allowance (none while the scene is paused).
    fn work(&mut self, dt: f32, real_dt: f32) {
        if !self.studying() || dt <= 0.0 {
            return;
        }
        let mut turn = scene_turn(&mut self.pace, self.rate, dt, real_dt);
        while !self.studies.done() && turn.next() {
            self.studies.add(&self.setup, self.run_seed);
        }
        self.pace.end(turn);
    }

    /// The run to its end at once (the tests of this file).
    #[cfg(test)]
    fn finish(&mut self, studies: usize) {
        self.refresh();
        while self.studies.len() < studies.min(MOST_STUDIES) {
            self.studies.add(&self.setup, self.run_seed);
        }
    }
}

// ----- the layout ----------------------------------------------------------------------

/// Where everything goes in a window.
#[derive(Clone, Copy, Debug, PartialEq)]
struct Layout {
    view: Vec2,
    narrow: bool,
    /// Text sizes: headings, words, small print.
    big: f32,
    text: f32,
    small: f32,
    /// The strip of data (its plot's rectangle) and its heading's place.
    strip: Rect,
    strip_head: Vec2,
    /// Where the arithmetic of the statistic is written (top-left) and how wide it may be.
    sum: Vec2,
    sum_width: f32,
    /// The chart of the statistic (its plot's rectangle).
    chart: Rect,
    /// The verdict on this sample, the key to the shadings, the buttons, and the table
    /// of many studies.
    verdict: Rect,
    key: Rect,
    tails: [Rect; 2],
    test: [Rect; 2],
    sample: [Rect; 2],
    run: [Rect; 2],
    table: Rect,
    /// How to read the picture, in a few lines (where there is room for them).
    note: Rect,
    /// Only the chart: the window is too small for the rest.
    compact: bool,
}

impl Layout {
    fn of(view: Vec2, inset: f32) -> Layout {
        let view = view.max(Vec2::splat(1.0));
        let narrow = view.x < 700.0;
        let (big, text, small) = if narrow { (13.0, 11.5, 10.5) } else { (15.0, 12.5, 11.5) };
        let side = if narrow { 10.0 } else { 22.0 };
        let top = inset.max(0.0) + if narrow { 8.0 } else { 14.0 };
        let narrow_now = view.x < 700.0;
        let compact = view.y < if narrow_now { 640.0 } else { 540.0 } || view.x < 300.0;
        let r = |x0: f32, y0: f32, x1: f32, y1: f32| (Vec2::new(x0, y0), Vec2::new(x1.max(x0 + 1.0), y1.max(y0 + 1.0)));
        let none = r(0.0, 0.0, 0.0, 0.0);
        let mut lay = Layout {
            view,
            narrow,
            big,
            text,
            small,
            strip: none,
            strip_head: Vec2::ZERO,
            sum: Vec2::ZERO,
            sum_width: 0.0,
            chart: none,
            verdict: none,
            key: none,
            tails: [none; 2],
            test: [none; 2],
            sample: [none; 2],
            run: [none; 2],
            table: none,
            note: none,
            compact,
        };
        if compact {
            // the verdict in a line, and the chart under it
            lay.narrow = true;
            lay.verdict = r(side, top, view.x - side, top + 18.0);
            lay.chart = r(side + 30.0, top + 30.0 + 44.0, view.x - side - 14.0, view.y - 40.0);
            return lay;
        }
        let button_h = if narrow { 26.0 } else { 28.0 };
        let pair = |row: Vec<Rect>| [row.first().copied().unwrap_or(none), row.get(1).copied().unwrap_or(none)];
        if narrow {
            let x1 = view.x - side;
            // the strip: a heading, two rows of labels, the points, the axis, the sum
            lay.strip_head = Vec2::new(side, top);
            // (a short phone has less room for the strip and the table)
            let short = view.y < 720.0;
            lay.strip = r(side + 6.0, top + 50.0, x1 - 6.0, top + if short { 82.0 } else { 96.0 });
            lay.sum = Vec2::new(side, lay.strip.1.y + 24.0);
            lay.sum_width = x1 - side;
            // below the chart: the key, the verdict, two rows of buttons, the table
            let bottom = view.y - 8.0;
            lay.table = r(side, bottom - if short { 80.0 } else { 100.0 }, x1, bottom);
            let buttons = lay.table.0.y - 8.0 - 2.0 * (button_h + 6.0);
            lay.verdict = r(side, buttons - 38.0, x1, buttons - 6.0);
            lay.key = r(side, lay.verdict.0.y - 22.0, x1, lay.verdict.0.y - 6.0);
            lay.chart = r(side + 30.0, lay.sum.y + 30.0 + 44.0, x1 - 14.0, lay.key.0.y - 40.0);
            let one = plot::even_row(4, side, x1, buttons, button_h, 6.0);
            let two = plot::even_row(4, side, x1, buttons + button_h + 6.0, button_h, 6.0);
            lay.tails = pair(one[..2.min(one.len())].to_vec());
            lay.test = pair(one.get(2..).map(<[Rect]>::to_vec).unwrap_or_default());
            lay.sample = pair(two[..2.min(two.len())].to_vec());
            lay.run = pair(two.get(2..).map(<[Rect]>::to_vec).unwrap_or_default());
        } else {
            // a column at the right for the verdict, the key, the buttons and the table
            let column = 300.0f32.min(view.x * 0.32);
            let (c0, c1) = (view.x - side - column, view.x - side);
            let x1 = c0 - 40.0;
            lay.strip_head = Vec2::new(side, top);
            lay.strip = r(side + 8.0, top + 62.0, x1, top + 124.0);
            lay.sum = Vec2::new(side, lay.strip.1.y + 30.0);
            lay.sum_width = x1 - side;
            lay.chart = r(side + 40.0, lay.sum.y + 58.0 + 44.0, x1, view.y - 52.0);
            lay.verdict = r(c0, top, c1, top + 70.0);
            lay.key = r(c0, lay.verdict.1.y + 6.0, c1, lay.verdict.1.y + 6.0 + 4.0 * (small + 7.0));
            let y = lay.key.1.y + 16.0;
            let row = |k: f32, gap: f32| pair(plot::even_row(2, c0, c1, y + k * (button_h + 8.0) + gap, button_h, 8.0));
            (lay.tails, lay.test, lay.sample, lay.run) = (row(0.0, 0.0), row(1.0, 0.0), row(2.0, 0.0), row(3.0, 14.0));
            let y = lay.run[0].1.y + 14.0;
            lay.table = r(c0, y, c1, (y + 152.0).min(view.y - 20.0));
            if view.y - 20.0 - lay.table.1.y > 120.0 {
                lay.note = r(c0, lay.table.1.y + 10.0, c1, view.y - 20.0);
            }
        }
        lay
    }
}

// ----- what is drawn, gliding ----------------------------------------------------------

// The picture's numbers, in the order the glide keeps them.
const X_LO: usize = 0;
const X_HI: usize = 1;
/// 1/df of the null's t (0 for z): the bell grows its heavy tails smoothly.
const INV_DF: usize = 2;
const DELTA: usize = 3;
const C_LO: usize = 4;
const C_HI: usize = 5;
/// 1 when the truth differs from the null (its curve is drawn), else 0.
const TRUTH_ON: usize = 6;
const D_LO: usize = 7;
const D_HI: usize = 8;
const MU: usize = 9;
const FRAME_LEN: usize = 10;

// The observed statistic and what hangs on it.
const STAT: usize = 0;
/// The p-value's tails: at or below the first, at or above the second.
const TAIL_LO: usize = 1;
const TAIL_HI: usize = 2;
const XBAR: usize = 3;
const SHIFT: usize = 4;
const SE: usize = 5;
const OBS_LEN: usize = 6;

/// Samples larger than this are drawn as a histogram rather than as points.
const DOTS: usize = 300;
/// The bins of that histogram, across the strip.
const SAMPLE_BINS: usize = 48;

#[derive(Clone, Debug, Default)]
struct Look {
    frame: Glide,
    obs: Glide,
    points: Glide,
    spread: Glide,
    studies: Glide,
    table: Glide,
    moving: bool,
}

/// The two curves of the chart, worked out at `CURVE_N` points across it for the
/// numbers they are drawn with (again only when those change).
#[derive(Clone, Debug, Default)]
struct Curves {
    key: [f32; 4],
    null: Vec<f64>,
    truth: Vec<f64>,
}

const CURVE_N: usize = 241;

impl Curves {
    fn update(&mut self, lo: f64, hi: f64, inv_df: f64, delta: f64) {
        let key = [lo as f32, hi as f32, inv_df as f32, delta as f32];
        if key == self.key && self.null.len() == CURVE_N {
            return;
        }
        self.key = key;
        let df = (inv_df > 1e-7).then(|| 1.0 / inv_df);
        let null = df.map_or(Dist::STANDARD_NORMAL, |df| Dist::StudentT { df });
        let spread = df.map_or_else(Spread::known, |df| Spread::estimated(df, 120));
        let x = |i: usize| lo + (hi - lo) * i as f64 / (CURVE_N - 1) as f64;
        self.null = (0..CURVE_N).map(|i| null.pdf(x(i))).collect();
        self.truth = (0..CURVE_N).map(|i| spread.density(x(i), delta)).collect();
    }

    /// A curve's height at x, between its points (0 off its ends).
    fn at(v: &[f64], lo: f64, hi: f64, x: f64) -> f64 {
        if v.len() < 2 || hi.is_nan() || lo.is_nan() || hi <= lo || !(lo..=hi).contains(&x) {
            return 0.0;
        }
        let at = (x - lo) / (hi - lo) * (v.len() - 1) as f64;
        let i = (at.floor() as usize).min(v.len() - 2);
        let t = at - i as f64;
        v[i] + (v[i + 1] - v[i]) * t
    }
}

/// How the chart's x range, the data strip's range and the curves' numbers are to be.
fn frame_goal(s: &Setup) -> Vec<f32> {
    let testable = s.no_test().is_none();
    let (lo, hi) = if testable { s.critical() } else { (f64::NEG_INFINITY, f64::INFINITY) };
    let delta = if testable { s.delta() } else { 0.0 };
    let truth = testable && s.effect != 0.0;
    // the range shows both curves' middles and the critical values, within ±14
    let mut x_lo = -4.2f64;
    let mut x_hi = 4.2f64;
    if lo.is_finite() {
        x_lo = x_lo.min(lo - 0.8);
    }
    if hi.is_finite() {
        x_hi = x_hi.max(hi + 0.8);
    }
    if truth {
        x_lo = x_lo.min(delta - 3.6);
        x_hi = x_hi.max(delta + 3.6);
    }
    let (x_lo, x_hi) = (x_lo.max(-14.0), x_hi.min(14.0));
    // what is off the chart waits just beyond its edge
    let off = |v: f64| v.max(x_lo - 1.5).min(x_hi + 1.5);
    let inv_df = s.df().map_or(0.0, |df| 1.0 / df);
    let mu = MU0 + s.effect;
    let reach = 3.4 * s.sigma.max(0.15);
    let (d_lo, d_hi) = (MU0.min(mu) - reach, MU0.max(mu) + reach);
    let mut g = vec![0.0f32; FRAME_LEN];
    g[X_LO] = x_lo as f32;
    g[X_HI] = x_hi as f32;
    g[INV_DF] = inv_df as f32;
    g[DELTA] = off(delta) as f32;
    g[C_LO] = off(lo) as f32;
    g[C_HI] = off(hi) as f32;
    g[TRUTH_ON] = if truth { 1.0 } else { 0.0 };
    g[D_LO] = d_lo as f32;
    g[D_HI] = d_hi as f32;
    g[MU] = mu as f32;
    g
}

// ----- colours and kinds of line ---------------------------------------------------------

/// What a test does right: the null kept when it is true.
const KEPT: Color = pal::DIM;
/// The rows of the table and the histogram's bars: by whether the effect was real.
fn origin_color(real: bool) -> Color {
    if real {
        TRUTH
    } else {
        ASSUMED
    }
}

/// Lines whose number changes: the rule of standard errors, the critical values and
/// the region of rejection, the marks in the table.
const RULER: u8 = OWN;
const CRITICAL: u8 = OWN + 1;
const REGION: u8 = OWN + 2;
/// (the table's two marks, `TABLE` and `TABLE + 1`)
const TABLE: u8 = OWN + 3;
const ARROWS: u8 = OWN + 5;
/// (and the outlines of the two histograms of many studies, `HIST` and `HIST + 1`)
const HIST: u8 = OWN + 6;

impl Scene for Testing {
    fn view(&self) -> View {
        View::Flat { center: Vec2::ZERO, half_height: 1.0 }
    }

    fn settings(&self, s: &mut ViewSettings) {
        s.grid = false;
        s.bloom = s.bloom.min(0.2);
    }

    fn animated(&self) -> bool {
        self.look.moving || self.studying() || self.dragging
    }

    fn pointer_hint(&self) -> Option<&'static str> {
        Some("yellow dot under the chart: drag the observed statistic    buttons over the picture: click")
    }

    fn ui(&mut self, ui: &mut egui::Ui, _: &Gfx) {
        ui::note(
            ui,
            r"A test asks whether data could easily have come about if nothing were going on. Assume nothing is: that is the null hypothesis, here that the data's mean $\mu$ is $\mu_0 = 0$. Measure how far the sample's mean $\bar x$ lies from $\mu_0$ in standard errors: the test statistic. If the null is true, and the data are bell-shaped as they are here, its distribution is known: the purple curve.",
        );
        ui::equation(ui, r"z = \frac{\bar x - \mu_0}{\sigma / \sqrt{n}}");
        ui::note(
            ui,
            r"The $p$-value is the chance, if the null is true, of a statistic at least as far out as the one observed (the pink area). If it is at most a threshold $\alpha$ chosen beforehand, the null is rejected; then, when the null is true, it is rejected wrongly with chance $\alpha$. When the effect is real, the statistic falls as the green curve says, and the chance of rejecting is the power.",
        );

        ui::section(ui, "Things to try");
        for (k, trial) in TRIALS.iter().enumerate() {
            if ui::experiment(ui, trial.button, trial.caption) {
                self.set_up(k);
            }
        }

        ui::section(ui, "The truth and the sample");
        let s = &mut self.setup;
        let mut effect = s.effect as f32;
        if ui::slider(ui, r"true effect $\mu - \mu_0$", &mut effect, -2.0..=2.0) {
            s.effect = effect as f64;
        }
        let mut sigma = s.sigma as f32;
        if ui::slider(ui, r"true spread $\sigma$", &mut sigma, 0.0..=3.0) {
            s.sigma = sigma as f64;
        }
        ui::note(ui, r"The data are drawn from a bell with mean $\mu = \mu_0 + \text{effect}$ and spread $\sigma$. The effect size is the effect divided by $\sigma$.");
        ui::slider_int_log(ui, "sample size $n$", &mut s.n, 1..=MOST_N);
        ui.horizontal(|ui| {
            if ui.button("new sample").clicked() {
                self.seed = self.seed.wrapping_add(1);
                self.held = None;
            }
            if self.held.is_some() && ui.button("use the sample").clicked() {
                self.held = None;
            }
        });
        ui::note(ui, "Dragging the yellow dot under the chart sets the statistic by hand: the sample slides along with it, so that its mean gives that statistic.");

        ui::section(ui, "The test");
        let s = &mut self.setup;
        let mut alpha = s.alpha as f32;
        if ui::slider_log_or_zero(ui, r"significance level $\alpha$", &mut alpha, 0.0001, 1.0) {
            s.alpha = alpha as f64;
        }
        let mut tails = (s.tails == Tails::Upper) as usize;
        if ui::choice(ui, &["two tails", "one tail (above)"], &mut tails) {
            s.tails = if tails == 1 { Tails::Upper } else { Tails::Two };
        }
        let mut test = s.estimated as usize;
        if ui::choice(ui, &[r"$\sigma$ known: $z$", r"$\sigma$ estimated: $t$"], &mut test) {
            s.estimated = test == 1;
        }
        ui::note(ui, r"With the spread estimated by the sample's own $s$, the statistic is $t = (\bar x - \mu_0)/(s/\sqrt n)$, and its null distribution is Student's $t$ with $n - 1$ degrees of freedom: heavier tails than the bell when $n$ is small, since $s$ may come out too small by chance.");

        ui::section(ui, "Many studies");
        ui.horizontal(|ui| {
            if ui.button(if self.running { "pause" } else { "run many studies" }).clicked() {
                self.running = !self.running;
            }
            if ui.button("start again").clicked() {
                self.run_seed = self.run_seed.wrapping_add(1);
                self.running = true;
            }
        });
        ui::slider_log(ui, "studies per second", &mut self.rate, 1.0..=5000.0);
        let mut real = self.setup.real as f32;
        if ui::slider(ui, "share of studies with the effect", &mut real, 0.0..=1.0) {
            self.setup.real = real as f64;
        }
        ui::note(ui, "Each study is a new sample from the truth, tested in the same way. Below 1, only that share of them has the effect, the rest none: as when many hypotheses are tested and only some are true.");
        self.refresh();

        ui::section(ui, "Reading the picture");
        ui::key(
            ui,
            &[
                (DATA, "cyan dots: the sample, $n$ points"),
                (ESTIMATE, r"yellow: what is worked out from it: its mean $\bar x$, and the statistic (the dot under the chart can be dragged)"),
                (ASSUMED, r"purple: what the null assumes: $\mu_0$, and the statistic's distribution if the null is true; shaded, the share $\alpha$ beyond the critical values, where the null is rejected"),
                (ALARM, r"pink: the $p$-value, the null's share at least as far out as the statistic; in the table, the errors"),
                (TRUTH, r"green: the truth: the true mean $\mu$, and the statistic's distribution as it really is; shaded, the power"),
                (pal::DIM, "outlined bars on the chart: the statistics of many studies, purple where the effect was zero, green where it was real"),
                (pal::WHITE, r"the table: the share of those studies that rejected the null, a white mark where it should settle ($\alpha$, the power)"),
            ],
        );
        ui::section(ui, "Words");
        ui::term(ui, "Null hypothesis:", r"what is assumed for the sake of argument, usually that nothing is going on: here $\mu = \mu_0$.");
        ui::term(ui, "Alternative:", r"what the test looks for instead: $\mu \ne \mu_0$ (two tails), or $\mu > \mu_0$ (one tail).");
        ui::term(ui, "Test statistic:", "a number worked out from the data whose distribution is known if the null is true.");
        ui::term(ui, "Standard error:", r"the spread of the sample's mean from sample to sample: $\sigma/\sqrt n$, estimated by $s/\sqrt n$ when $\sigma$ is not known.");
        ui::term(ui, "$p$-value:", "the chance, if the null is true, of a statistic at least as far out as the one observed. Not the chance that the null is true.");
        ui::term(ui, r"Significance level $\alpha$:", "the threshold, chosen before looking: the null is rejected when $p \\le \\alpha$.");
        ui::term(ui, "Critical value:", r"where the statistic's region of rejection starts: beyond it, $p \le \alpha$.");
        ui::term(ui, "Error of the first kind:", r"a false alarm: the null rejected when it is true. Its chance is $\alpha$.");
        ui::term(ui, "Error of the second kind:", r"a miss: the null kept when the effect is real. Its chance is $\beta$.");
        ui::term(ui, "Power:", r"$1 - \beta$, the chance of rejecting when the effect is real. It grows with the size of the effect (on the side the test looks at), with $n$ and with $\alpha$, and falls as $\sigma$ grows.");
        ui::term(ui, "Effect size:", r"the effect in units of the spread, $(\mu - \mu_0)/\sigma$.");
        ui::term(ui, "Degrees of freedom:", r"how many independent pieces the spread estimate is made of: $n - 1$ here.");

        ui::section(ui, "Numbers");
        let s = self.setup;
        match (s.no_test(), self.observed()) {
            (Some(NoTest::NoNoise), _) => ui::readout(ui, "test", r"none: $\sigma = 0$"),
            (Some(NoTest::OnePoint), _) => ui::readout(ui, "test", r"none: one point, no spread to estimate"),
            (None, Some(o)) => {
                ui::readout(ui, r"sample mean $\bar x$", num(o.xbar, 3));
                if s.estimated {
                    ui::readout(ui, r"sample spread $s$", num(o.s, 3));
                }
                ui::readout(ui, "standard error", num(o.se, 4));
                ui::readout(ui, if s.estimated { r"statistic $t$" } else { r"statistic $z$" }, num(o.stat, 2));
                ui::readout(ui, "$p$-value", readable(p_value_text(o.p)));
                ui::readout(ui, "verdict", if o.rejects { "reject the null" } else { "do not reject" });
            }
            (None, None) => ui::readout(ui, "test", "none"),
        }
        if s.no_test().is_none() {
            let (lo, hi) = s.critical();
            let c = match s.tails {
                Tails::Two => format!("{} and {}", num(lo, 2), num(hi, 2)),
                Tails::Upper => num(hi, 2),
            };
            ui::readout(ui, "critical value", if hi.is_finite() || lo.is_finite() { c } else { "none: never reject".to_string() });
            ui::readout(ui, r"$\delta = (\mu - \mu_0)\sqrt n/\sigma$", num(s.delta(), 2));
            ui::readout(ui, if s.effect == 0.0 { r"chance of rejecting (with no effect, $\alpha$)" } else { "power" }, readable(chance(self.power())));
            ui::readout(
                ui,
                "$n$ for power 0.8",
                match self.n_for() {
                    Some(n) => count(n),
                    None => "none reaches it".to_string(),
                },
            );
        }
        let t = &self.studies.tally;
        if self.studies.len() > 0 {
            ui::readout(ui, "studies", count(self.studies.len()));
            if let Some(share) = t.share(false) {
                ui::readout(ui, "  no effect: share rejected", num(share, 3));
            }
            if let Some(share) = t.share(true) {
                ui::readout(ui, "  real effect: share rejected", num(share, 3));
            }
        }
        ui::see_also(
            ui,
            &[
                ("pvalues", "many tests, and looking again and again"),
                ("intervals", "the values a test would not reject"),
                ("twogroups", "comparing two groups"),
                ("chance", "base rates: why a positive result may be false"),
            ],
        );
    }

    fn frame(&mut self, f: &mut Frame) {
        hold_batches(f);
        let lay = Layout::of(f.view_size(), f.top_inset);
        // (a canvas squeezed to a sliver is one the controls have opened over, on a phone)
        let tiny = lay.view.x < 160.0 || lay.view.y < 160.0;
        let was_dragging = self.dragging;
        self.dragging = false;
        if !tiny {
            if !lay.compact && self.buttons(f, &lay) {
                self.refresh();
            }
            self.drag(f, &lay);
        }
        self.refresh();
        self.work(f.dt, f.real_dt);
        let snap = self.canvas != Some(lay.view);
        if snap && !tiny {
            self.canvas = Some(lay.view);
        }
        self.follow(f.real_dt, snap, was_dragging || self.dragging);
        if tiny {
            return;
        }
        self.draw_chart(f, &lay);
        if !lay.compact {
            self.draw_strip(f, &lay);
            self.draw_table(f, &lay);
        }
        self.draw_verdict(f, &lay);
        if !lay.compact {
            self.draw_key(f, &lay);
            self.draw_note(f, &lay);
        }
    }
}

impl Testing {
    /// The buttons over the picture; true if a setting changed.
    fn buttons(&mut self, f: &mut Frame, lay: &Layout) -> bool {
        let size = lay.small;
        let mut changed = false;
        let s = &mut self.setup;
        for (k, (name, tails)) in [("two tails", Tails::Two), ("one tail", Tails::Upper)].into_iter().enumerate() {
            if plot::button(f, lay.tails[k], name, size, s.tails == tails) && s.tails != tails {
                s.tails = tails;
                changed = true;
            }
        }
        let names = if lay.narrow { [r"$\sigma$ known", r"$\sigma$ estimated"] } else { [r"$\sigma$ known: $z$", r"$\sigma$ estimated: $t$"] };
        for (k, name) in names.into_iter().enumerate() {
            if plot::button(f, lay.test[k], name, size, s.estimated == (k == 1)) && s.estimated != (k == 1) {
                s.estimated = k == 1;
                changed = true;
            }
        }
        if plot::button(f, lay.sample[0], "new sample", size, false) {
            self.seed = self.seed.wrapping_add(1);
            self.held = None;
            changed = true;
        }
        if self.held.is_some() && plot::button(f, lay.sample[1], "use the sample", size, false) {
            self.held = None;
            changed = true;
        }
        let run = if self.running && !self.studies.done() {
            "pause"
        } else if lay.narrow {
            "run studies"
        } else {
            "run many studies"
        };
        if plot::button(f, lay.run[0], run, size, false) {
            self.running = !self.running || self.studies.done();
            if self.studies.done() {
                self.run_seed = self.run_seed.wrapping_add(1);
            }
            changed = true;
        }
        if plot::button(f, lay.run[1], "start again", size, false) {
            self.run_seed = self.run_seed.wrapping_add(1);
            self.running = true;
            changed = true;
        }
        changed
    }

    /// The observed statistic, dragged along the chart's axis.
    fn drag(&mut self, f: &mut Frame, lay: &Layout) {
        let Some(o) = self.observed() else { return };
        let Some(chart) = self.chart_plot(lay) else { return };
        let shown = self.look.obs.get(STAT) as f64;
        let at = if shown.is_finite() { shown } else { o.stat };
        if let Some((x, _)) = chart.handle(f, 1, at, 0.0, ESTIMATE) {
            if x.is_finite() {
                self.held = Some(x);
                self.dragging = true;
            }
        }
    }

    /// Bring the picture `dt` seconds nearer to what it shows. (`snap`: there at once.)
    fn follow(&mut self, dt: f32, snap: bool, held: bool) {
        let s = self.setup;
        let frame = frame_goal(&s);
        let (x_lo, x_hi) = (frame[X_LO] as f64, frame[X_HI] as f64);
        let off = |v: f64| v.max(x_lo - 1.5).min(x_hi + 1.5);
        let mut obs = vec![0.0f32; OBS_LEN];
        if let Some(o) = self.observed() {
            obs[STAT] = off(o.stat) as f32;
            let (lo, hi) = match s.tails {
                Tails::Two => (-o.stat.abs(), o.stat.abs()),
                Tails::Upper => (f64::NEG_INFINITY, o.stat),
            };
            obs[TAIL_LO] = off(lo) as f32;
            obs[TAIL_HI] = off(hi) as f32;
            obs[XBAR] = o.xbar as f32;
            obs[SHIFT] = o.shift as f32;
            obs[SE] = o.se as f32;
        } else {
            obs[STAT] = (x_lo - 1.5) as f32;
            obs[TAIL_LO] = (x_lo - 1.5) as f32;
            obs[TAIL_HI] = (x_hi + 1.5) as f32;
            obs[XBAR] = (MU0 + s.effect) as f32;
        }
        let n = s.n;
        let points: Vec<f32> = if n <= DOTS { self.draws.first(n).iter().map(|z| (MU0 + s.effect + s.sigma * z) as f32).collect() } else { Vec::new() };
        let spread = if n > DOTS { self.sample_bins(&frame) } else { Vec::new() };
        let spread = spread.as_slice();
        let studies = self.studies.heights();
        let t = &self.studies.tally;
        let power = self.power();
        let table = [
            t.share(false).unwrap_or(0.0) as f32,
            t.share(true).unwrap_or(0.0) as f32,
            s.alpha as f32,
            if power.is_finite() { power as f32 } else { 0.0 },
        ];
        let look = &mut self.look;
        if snap {
            look.frame.snap(&frame);
            look.obs.snap(&obs);
            look.points.snap(&points);
            look.spread.snap(spread);
            look.studies.snap(&studies);
            look.table.snap(&table);
            look.moving = false;
            return;
        }
        let mut moving = look.frame.follow(&frame, dt, false);
        // (the statistic answers the hand at once)
        if held {
            look.obs.snap(&obs);
        } else {
            moving |= look.obs.follow(&obs, dt, false);
        }
        moving |= look.points.follow(&points, dt, true);
        moving |= look.spread.follow(spread, dt, false);
        moving |= look.studies.follow(&studies, dt, false);
        moving |= look.table.follow(&table, dt, false);
        look.moving = moving;
    }

    /// The sample (without the hand's shift) in bins across the strip's range, as
    /// heights that put a bell's peak near the top.
    fn sample_bins(&mut self, frame: &[f32]) -> Vec<f32> {
        let s = self.setup;
        let (lo, hi) = (frame[D_LO] as f64, frame[D_HI] as f64);
        let key = [self.seed, s.n as u64, s.effect.to_bits(), s.sigma.to_bits(), lo.to_bits(), hi.to_bits()];
        if key == self.binned.0 && self.binned.1.len() == SAMPLE_BINS {
            return self.binned.1.clone();
        }
        let width = (hi - lo) / SAMPLE_BINS as f64;
        let mut bins = [0u32; SAMPLE_BINS];
        for z in self.draws.first(s.n) {
            let x = MU0 + s.effect + s.sigma * z;
            let b = (x - lo) / width;
            if b >= 0.0 && b < SAMPLE_BINS as f64 {
                bins[b as usize] += 1;
            }
        }
        // a density, scaled so that a bell of spread σ peaks at 0.85 of the strip
        let scale = 0.85 / (st::normal_pdf(0.0) / s.sigma.max(0.15));
        let bins: Vec<f32> = bins.iter().map(|&c| (c as f64 / (s.n.max(1) as f64 * width) * scale) as f32).collect();
        self.binned = (key, bins.clone());
        bins
    }

    /// The chart of the statistic, with the ranges it is drawn with now.
    fn chart_plot(&self, lay: &Layout) -> Option<Plot> {
        let g = &self.look.frame;
        if g.shown.len() < FRAME_LEN {
            return None;
        }
        let p = Plot::new(lay.chart, [g.get(X_LO) as f64, g.get(X_HI) as f64], [0.0, 0.45]);
        (!p.tiny()).then_some(p)
    }


    /// The x axis of a chart: its round values chosen from where its range is going, so
    /// that they keep their places while it glides there; its name at its right end.
    fn axis_of(f: &mut Frame, p: &Plot, goal: (f64, f64), name: &str, size: f32) {
        let (min, max) = p.rect;
        plot::line_of(f, AXES, Vec2::new(min.x, max.y), max, 1.0, pal::DIM.a(0.8));
        let ticks = plot::Ticks::of(goal.0, goal.1, p.width());
        for v in ticks.values() {
            if v < p.x[0] - 1e-9 || v > p.x[1] + 1e-9 {
                continue;
            }
            let x = p.px_x(v);
            plot::line_of(f, AXES, Vec2::new(x, max.y), Vec2::new(x, max.y + 4.0), 1.0, pal::DIM.a(0.8));
            f.text_px(Vec2::new(x, max.y + 6.0), ticks.name(v), size, pal::DIM, Align2::CENTER_TOP);
        }
        if !name.is_empty() {
            f.text_px(Vec2::new(max.x + 8.0, max.y), name, size + 1.0, pal::TEXT, Align2::LEFT_CENTER);
        }
    }

    fn draw_chart(&mut self, f: &mut Frame, lay: &Layout) {
        let Some(chart) = self.chart_plot(lay) else { return };
        let s = self.setup;
        let (lo, hi) = (chart.x[0], chart.x[1]);
        let g = |i: usize| self.look.frame.get(i) as f64;
        let (inv_df, delta) = (g(INV_DF), g(DELTA));
        self.curves.update(lo, hi, inv_df, delta);
        let g = |i: usize| self.look.frame.get(i) as f64;
        let curves = &self.curves;
        let null = |x: f64| Curves::at(&curves.null, lo, hi, x);
        let truth = |x: f64| Curves::at(&curves.truth, lo, hi, x);
        let (c_lo, c_hi) = (g(C_LO), g(C_HI));
        let truth_on = (g(TRUTH_ON) as f32).clamp(0.0, 1.0);
        let o = |i: usize| self.look.obs.get(i) as f64;
        let (tail_lo, tail_hi) = (o(TAIL_LO), o(TAIL_HI));
        let testable = s.no_test().is_none();
        let (min, max) = chart.rect;

        chart.backing(f, 40.0, 40.0, 46.0, 26.0);
        let name = if s.estimated { "$t$" } else { "$z$" };
        let title = if lay.narrow {
            format!("where the statistic {name} falls, sample after sample")
        } else {
            format!("where the test statistic {name} falls, sample after sample: if the null is true, and as the truth is")
        };
        chart.title(f, &title, lay.text);
        chart.y_axis(f, "density", lay.small);

        // the statistics of many studies, behind everything: a faint body, and the
        // outline of each kind (those of no effect below, those of a real one on them).
        // Every bin is in the outline in every frame, so that it keeps its number of
        // points: one outside the chart is squeezed to nothing at its edge, and one
        // coming in grows from there.
        let heights = &self.look.studies.shown;
        if heights.len() == 2 * BINS {
            let top = chart.y[1];
            let mut outline: [Vec<Vec2>; 2] = [Vec::with_capacity(2 * BINS), Vec::with_capacity(2 * BINS)];
            let mut any = [false; 2];
            for b in 0..BINS {
                let x0 = BINS_FROM + b as f64 * BIN;
                let (a, z) = (x0.max(lo).min(hi), (x0 + BIN).max(lo).min(hi));
                let seen = ((z - a) / BIN).clamp(0.0, 1.0);
                let (h0, h1) = (heights[b] as f64 * seen, heights[BINS + b] as f64 * seen);
                if h0 > 1e-6 {
                    chart.bar(f, a, z, 0.0, h0, 0.0, origin_color(false).a(0.035));
                    any[0] = true;
                }
                if h1 > 1e-6 {
                    chart.bar(f, a, z, h0, h0 + h1, 0.0, origin_color(true).a(0.035));
                    any[1] = true;
                }
                for (k, height) in [h0, h0 + h1].into_iter().enumerate() {
                    let y = chart.px_y(height.max(0.0).min(top));
                    outline[k].push(Vec2::new(chart.px_x(a), y));
                    outline[k].push(Vec2::new(chart.px_x(z), y));
                }
            }
            for (k, line) in outline.iter().enumerate() {
                if any[k] {
                    plot::path_of(f, HIST + k as u8, line, 1.2, origin_color(k == 1).mix(pal::WHITE, 0.25).a(0.8));
                }
            }
        }

        if testable {
            // α: the null's share beyond the critical values
            chart.area(f, lo, c_lo, 0.0, 60, ASSUMED.a(0.30), null);
            chart.area(f, c_hi, hi, 0.0, 60, ASSUMED.a(0.30), null);
            // the power: the truth's share beyond them; β, the rest
            if truth_on > 0.0 {
                chart.area(f, c_hi, hi, 0.0, 90, TRUTH.a(0.24 * truth_on), truth);
                chart.area(f, lo, c_lo, 0.0, 90, TRUTH.a(0.24 * truth_on), truth);
                chart.area(f, c_lo, c_hi, 0.0, 120, TRUTH.a(0.025 * truth_on), truth);
            }
            // the p-value: the null's share at least as far out as the statistic
            chart.area(f, lo, tail_lo, 0.0, 60, ALARM.a(0.6), null);
            chart.area(f, tail_hi, hi, 0.0, 60, ALARM.a(0.6), null);
        }

        // the curves
        chart.curve(f, CURVES, CURVE_N, 2.4, TRUTH.a(truth_on), truth);
        chart.curve(f, CURVES, CURVE_N, 2.4, ASSUMED, null);

        // the critical values, and the region of rejection along the axis
        if testable {
            for c in [c_lo, c_hi] {
                chart.vline(f, CRITICAL, c, 1.3, ASSUMED.a(0.9), true);
            }
            for (a, b) in [(lo, c_lo), (c_hi, hi)] {
                let (a, b) = (a.max(lo), b.min(hi));
                if b > a {
                    plot::line_of(f, REGION, Vec2::new(chart.px_x(a), max.y), Vec2::new(chart.px_x(b), max.y), 5.0, ASSUMED);
                }
            }
        }
        let goal = frame_goal(&s);
        Self::axis_of(f, &chart, (goal[X_LO] as f64, goal[X_HI] as f64), name, lay.small);

        if !testable {
            let why = match s.no_test() {
                Some(NoTest::NoNoise) => format!(
                    r"No test: with $\sigma = 0$ every point is exactly $\mu = {}$. There is no chance in these data, and nothing for a test to weigh: the mean is known.",
                    num(MU0 + s.effect, 2)
                ),
                _ => r"No test: from one point the spread cannot be estimated ($n - 1 = 0$ degrees of freedom). Take $\sigma$ as known, or more points.".to_string(),
            };
            wrapped_px(f, Vec2::new(min.x + 16.0, min.y + 16.0), (max.x - min.x - 32.0).max(60.0), &why, lay.text, pal::TEXT);
            return;
        }

        let taken = self.stat_labels(f, &chart, lay);
        self.chart_labels(f, &chart, lay, &null, &taken);

        // the observed statistic: a line up the chart and its dot (`drag`), or an arrow
        // at the edge when it is beyond it
        let stat = o(STAT);
        if stat >= lo && stat <= hi {
            chart.vline(f, MARKS, stat, 2.0, ESTIMATE, false);
        } else if stat.is_finite() {
            let right = stat > hi;
            let (x, y) = (if right { max.x - 3.0 } else { min.x + 3.0 }, max.y - 14.0);
            let d = if right { 1.0 } else { -1.0 };
            for (a, b) in [((x - 22.0 * d, y), (x, y)), ((x - 7.0 * d, y - 5.0), (x, y)), ((x - 7.0 * d, y + 5.0), (x, y))] {
                plot::line_of(f, ARROWS, Vec2::new(a.0, a.1), Vec2::new(b.0, b.1), 2.0, ESTIMATE);
            }
        }
    }

    /// The names of the two curves, where the region of rejection starts, and the power.
    fn chart_labels(&self, f: &mut Frame, chart: &Plot, lay: &Layout, null: &impl Fn(f64) -> f64, taken: &[Rect]) {
        let s = &self.setup;
        let g = |i: usize| self.look.frame.get(i) as f64;
        let (min, max) = chart.rect;
        let size = lay.small;
        let (lo, hi) = (chart.x[0], chart.x[1]);
        let truth_on = g(TRUTH_ON) > 0.5;
        let right = s.delta() >= 0.0;
        // the null's name at its peak, the truth's at its own, each on its far side
        let peak = chart.at(0.0, null(0.0));
        let named = if truth_on { "if the null is true".to_string() } else { "if the null is true: here, the truth".to_string() };
        let w = f.text_size(&named, size);
        let side = if right || !truth_on { -1.0 } else { 1.0 };
        let at = Self::clear_of(f, "null", peak, w, side, taken, chart.rect);
        f.text_placed("null", 0, peak, at, named, size, ASSUMED);
        if truth_on {
            // (the truth's peak: the highest of its points)
            let (i, top) = self.curves.truth.iter().enumerate().fold((0, 0.0f64), |m, (i, &v)| if v > m.1 { (i, v) } else { m });
            let x = lo + (hi - lo) * i as f64 / (CURVE_N - 1) as f64;
            let peak = chart.at(x, top);
            let delta = s.delta();
            let named = if delta > hi + 1.0 || delta < lo - 1.0 {
                format!(r"the truth: $\mu = {}$, its curve centred at {}, off the chart", num(MU0 + s.effect, 2), num(delta, 1))
            } else {
                format!(r"the truth: $\mu = {}$", num(MU0 + s.effect, 2))
            };
            let w = f.text_size(&named, size);
            let at = Self::clear_of(f, "truth", peak, w, -side, taken, chart.rect);
            f.text_placed("truth", 0, peak, at, named, size, TRUTH);
            // the power, beside the truth's outer flank half way down (its shading is
            // the part beyond the critical value)
            let power = self.power();
            if power.is_finite() {
                let n = self.curves.truth.len().max(1);
                let flank = if right { (i..n).find(|&j| self.curves.truth[j] < 0.5 * top) } else { (0..=i).rev().find(|&j| self.curves.truth[j] < 0.5 * top) };
                if let Some(j) = flank {
                    let x = lo + (hi - lo) * j as f64 / (CURVE_N - 1) as f64;
                    let from = chart.at(x, 0.5 * top);
                    let text = format!("power {}", chance(power));
                    let w = f.text_size(&text, size);
                    let offset = Vec2::new(if right { w.x * 0.5 + 8.0 } else { -w.x * 0.5 - 8.0 }, 0.0);
                    let at = Self::kept_in(from + offset, w, chart.rect) - from;
                    f.text_placed("power", 0, from, at, text, size, TRUTH.mix(pal::WHITE, 0.3));
                }
            }
        }
        // "reject" under the axis, outside each critical value
        let name = if s.estimated { "t" } else { "z" };
        let y = max.y + 22.0;
        let (c_lo, c_hi) = (g(C_LO), g(C_HI));
        let (lo_c, hi_c) = s.critical();
        if c_hi < hi && hi_c.is_finite() {
            let text = format!(r"${name} \ge {}$: reject", num(hi_c, 2));
            let w = f.text_size(&text, size).x;
            let x = (chart.px_x(c_hi) + 4.0).max(min.x).min(max.x + 20.0 - w);
            f.text_px(Vec2::new(x, y), text, size, ASSUMED.mix(pal::WHITE, 0.3), Align2::LEFT_TOP);
        }
        if c_lo > lo && lo_c.is_finite() {
            let text = format!(r"reject: ${name} \le {}$", num(lo_c, 2));
            let w = f.text_size(&text, size).x;
            let x = (chart.px_x(c_lo) - 4.0).min(max.x).max(min.x - 34.0 + w);
            f.text_px(Vec2::new(x, y), text, size, ASSUMED.mix(pal::WHITE, 0.3), Align2::RIGHT_TOP);
        }
    }

    /// Where to put the name of a curve (its offset from the curve's peak): above the
    /// peak on the side `side` (−1 left, 1 right), or else on the other, or else below
    /// what is in the way; kept where it was while that is still clear.
    fn clear_of(f: &Frame, name: &'static str, peak: Vec2, w: Vec2, side: f32, taken: &[Rect], within: Rect) -> Vec2 {
        let box_at = |c: Vec2| (c - w * 0.5 - Vec2::splat(2.0), c + w * 0.5 + Vec2::splat(2.0));
        let free = |c: Vec2| {
            let (a, b) = box_at(c);
            taken.iter().all(|(c0, c1)| b.x <= c0.x || c1.x <= a.x || b.y <= c0.y || c1.y <= a.y)
        };
        let above = |side: f32| Self::kept_in(peak + Vec2::new(side * (w.x * 0.5 + 6.0), -(w.y * 0.5 + 4.0)), w, within);
        let mut candidates = vec![above(side), above(-side)];
        // below everything in the way on the first side
        let low = taken.iter().map(|r| r.1.y).fold(peak.y, f32::max);
        candidates.push(Self::kept_in(Vec2::new(above(side).x, low + w.y * 0.5 + 6.0), w, within));
        let kept = f.places.chosen(name, 0).map(|o| peak + o);
        let pick = kept
            .filter(|k| free(*k) && candidates.iter().any(|c| (*c - *k).length() < 1.0))
            .or_else(|| candidates.iter().copied().find(|c| free(*c)))
            .unwrap_or(candidates[0]);
        pick - peak
    }

    /// A label's centre moved so that the label (of size `w`) stays inside the rectangle.
    fn kept_in(centre: Vec2, w: Vec2, (min, max): Rect) -> Vec2 {
        let half = w * 0.5 + Vec2::splat(2.0);
        let x = centre.x.max(min.x + half.x).min(max.x - half.x);
        let y = centre.y.max(min.y + half.y).min(max.y - half.y);
        Vec2::new(if x.is_finite() { x } else { centre.x }, if y.is_finite() { y } else { centre.y })
    }

    /// The statistic and its p-value at the top of its line, on whichever side has room.
    fn stat_labels(&self, f: &mut Frame, chart: &Plot, lay: &Layout) -> Vec<Rect> {
        let mut taken = Vec::new();
        let Some(obs) = self.observed() else { return taken };
        let s = &self.setup;
        let (min, max) = chart.rect;
        let size = lay.text;
        let stat = self.look.obs.get(STAT) as f64;
        let name = if s.estimated { "t" } else { "z" };
        let lines = [
            (format!("${name} = {}$", num(obs.stat, 2)), ESTIMATE),
            (format!(r"$p {}$", is(obs.p)), ALARM.mix(pal::WHITE, 0.25)),
        ];
        let x = chart.px_x(stat).max(min.x).min(max.x);
        for (k, (text, color)) in lines.into_iter().enumerate() {
            let w = f.text_size(&text, size);
            let from = Vec2::new(x, min.y + 10.0 + k as f32 * (size + 6.0));
            // right of the line where there is room, else left; kept where it was while it fits
            let right_ok = x + 8.0 + w.x <= max.x;
            let left_ok = x - 8.0 - w.x >= min.x;
            let before = f.places.chosen("stat", k as u32).map(|o| o.x > 0.0);
            let right = match before {
                Some(true) if right_ok => true,
                Some(false) if left_ok => false,
                _ => right_ok || !left_ok,
            };
            let offset = Vec2::new(if right { w.x * 0.5 + 7.0 } else { -w.x * 0.5 - 7.0 }, w.y * 0.5);
            let centre = from + offset;
            taken.push((centre - w * 0.5, centre + w * 0.5));
            f.text_placed("stat", k as u32, from, offset, text, size, color);
        }
        taken
    }

    /// The strip of data: the sample's points, μ₀, μ and the mean x̄, and the arithmetic
    /// of the statistic under it.
    fn draw_strip(&self, f: &mut Frame, lay: &Layout) {
        let s = &self.setup;
        let g = |i: usize| self.look.frame.get(i) as f64;
        let o = |i: usize| self.look.obs.get(i) as f64;
        let strip = Plot::new(lay.strip, [g(D_LO), g(D_HI)], [0.0, 1.0]);
        if strip.tiny() {
            return;
        }
        let (min, max) = strip.rect;
        strip.backing(f, 8.0, 26.0, 52.0, 8.0);
        let head = if lay.narrow {
            format!(r"this sample: $n = {}$ {} drawn from the truth", count(s.n), if s.n == 1 { "point" } else { "points" })
        } else {
            format!(
                r"this sample: $n = {}$ {} drawn from the truth, a bell with mean $\mu = {}$ and spread $\sigma = {}$",
                count(s.n),
                if s.n == 1 { "point" } else { "points" },
                num(MU0 + s.effect, 2),
                num(s.sigma, 2)
            )
        };
        f.text_px(lay.strip_head, head, lay.text, pal::TEXT, Align2::LEFT_TOP);
        let shift = o(SHIFT);
        // the points, at heights of their own (by their place in the sample), or for a
        // large sample its histogram
        let points = &self.look.points.shown;
        if s.n <= DOTS {
            let size = if s.n > 100 { 4.0 } else { 6.0 };
            // (one outside the strip is drawn unseen at its edge: the batch keeps as many
            // points from frame to frame, each in its place)
            for (i, x) in points.iter().enumerate() {
                let at = strip.at(*x as f64 + shift, 0.12 + 0.76 * jitter(i));
                let inside = at.x >= min.x - 0.5 && at.x <= max.x + 0.5;
                plot::disc(f, Vec2::new(at.x.max(min.x).min(max.x), at.y), size, if inside { DATA } else { DATA.a(0.0) });
            }
        } else {
            let goal = frame_goal(s);
            let (a, b) = (goal[D_LO] as f64, goal[D_HI] as f64);
            let width = (b - a) / SAMPLE_BINS as f64;
            for (k, h) in self.look.spread.shown.iter().enumerate() {
                let x0 = a + k as f64 * width + shift;
                strip.bar(f, x0, x0 + width, 0.0, (*h as f64).min(1.0), 0.5, DATA.a(0.55));
            }
        }
        // μ₀, μ and x̄
        let mu = g(MU);
        let testable = s.no_test().is_none();
        strip.vline(f, MARKS, MU0, 1.6, ASSUMED, true);
        strip.vline(f, MARKS, mu, 1.6, TRUTH, false);
        let xbar = o(XBAR);
        if testable {
            strip.vline(f, MARKS, xbar, 2.0, ESTIMATE, false);
        }
        // their names: μ₀ and μ on the row above, on their far sides; x̄ on the row above that
        let size = lay.small;
        let right = mu >= MU0;
        let row = |k: f32| min.y - 9.0 - k * (size + 5.0);
        let label = |f: &mut Frame, name: &'static str, x: f64, k: f32, text: String, color: Color, right: bool| {
            let w = f.text_size(&text, size);
            let from = Vec2::new(strip.px_x(x).max(min.x).min(max.x), row(k));
            let off = Vec2::new(if right { w.x * 0.5 + 5.0 } else { -w.x * 0.5 - 5.0 }, 0.0);
            let at = Self::kept_in(from + off, w, (Vec2::new(min.x - 6.0, -1e9), Vec2::new(max.x + 6.0, 1e9))) - from;
            f.text_placed(name, 0, from, at, text, size, color);
        };
        let null = if s.effect == 0.0 { r"$\mu = \mu_0 = 0$: the null is true" } else { r"$\mu_0 = 0$ (the null)" };
        label(f, "mu0", MU0, 0.0, null.to_string(), ASSUMED.mix(pal::WHITE, 0.2), !right);
        if s.effect != 0.0 {
            label(f, "mu", mu, 0.0, format!(r"$\mu = {}$ (the truth)", num(MU0 + s.effect, 2)), TRUTH, right);
        }
        if let (true, Some(obs)) = (testable, self.observed()) {
            let text = format!(r"$\bar x = {}$", num(obs.xbar, if s.n > 1000 { 3 } else { 2 }));
            label(f, "xbar", xbar, 1.0, text, ESTIMATE, obs.xbar >= MU0);
            // the distance from μ₀ to x̄ in steps of the standard error, along the foot of the strip
            let se = o(SE);
            let (a, b) = (strip.px_x(MU0), strip.px_x(xbar));
            let step = (strip.px_x(MU0 + se) - a).abs();
            let y = max.y - 4.0;
            let (a, b) = (a.max(min.x).min(max.x), b.max(min.x).min(max.x));
            plot::line_of(f, RULER, Vec2::new(a, y), Vec2::new(b, y), 2.0, ESTIMATE.a(0.85));
            if step >= 3.0 && step.is_finite() {
                let steps = (((b - a).abs() / step) as usize).min(80);
                let d = if b >= a { step } else { -step };
                for k in 0..=steps {
                    let x = a + d * k as f32;
                    plot::line_of(f, RULER, Vec2::new(x, y - 4.0), Vec2::new(x, y + 1.0), 1.4, ESTIMATE.a(0.85));
                }
            }
        }
        let goal = frame_goal(s);
        Self::axis_of(f, &strip, (goal[D_LO] as f64, goal[D_HI] as f64), "", lay.small);
        // the arithmetic
        if let (true, Some(obs)) = (testable, self.observed()) {
            let sum = self.sum(&obs, lay, f);
            f.text_px(lay.sum, sum, lay.text, pal::TEXT, Align2::LEFT_TOP);
        }
    }

    /// The statistic worked out, as long a version as there is room for.
    fn sum(&self, o: &Observed, lay: &Layout, f: &Frame) -> String {
        let s = &self.setup;
        let (x, m0) = (tint(ESTIMATE, r"\bar x"), tint(ASSUMED, r"\mu_0"));
        let digits = if s.n > 1000 { 3 } else { 2 };
        let xbar = tint(ESTIMATE, &num(o.xbar, digits));
        let mu0 = tint(ASSUMED, &num(MU0, 0));
        let (name, spread, spread_value) = if s.estimated {
            ("t", tint(ESTIMATE, "s"), tint(ESTIMATE, &num(o.s, 2)))
        } else {
            ("z", tint(TRUTH, r"\sigma"), tint(TRUTH, &num(s.sigma, 2)))
        };
        let result = num(o.stat, 2);
        let n = count(s.n).replace('\u{2009}', r"\,");
        let after = if s.estimated { format!(r", with $n - 1 = {}$ degrees of freedom", count(s.n - 1)) } else { String::new() };
        let errors = if s.estimated { "estimated standard errors" } else { "standard errors" };
        let options = [
            format!(r"${name} = \dfrac{{{x} - {m0}}}{{{spread}/\sqrt{{n}}}} = \dfrac{{{xbar} - {mu0}}}{{{spread_value}/\sqrt{{{n}}}}} = {result}$: the mean is {result} {errors} from $\mu_0${after}"),
            format!(r"${name} = \dfrac{{{x} - {m0}}}{{{spread}/\sqrt{{n}}}} = \dfrac{{{xbar} - {mu0}}}{{{spread_value}/\sqrt{{{n}}}}} = {result}${after}"),
            format!(r"${name} = ({x} - {m0})/({spread}/\sqrt{{n}}) = ({xbar} - {mu0})/({spread_value}/\sqrt{{{n}}}) = {result}$"),
            format!(r"${name} = ({xbar} - {mu0})/({spread_value}/\sqrt{{{n}}}) = {result}$"),
        ];
        let first = if lay.narrow { 2 } else { 0 };
        options[first..].iter().find(|t| f.text_size(t, lay.text).x <= lay.sum_width).or(options.last()).cloned().unwrap_or_default()
    }

    /// What this sample says, and what that is when the truth is known.
    fn draw_verdict(&self, f: &mut Frame, lay: &Layout) {
        let s = &self.setup;
        let (min, max) = lay.verdict;
        let width = max.x - min.x;
        if width < 40.0 {
            return;
        }
        let Some(o) = self.observed() else {
            let why = match s.no_test() {
                Some(NoTest::NoNoise) => r"no test: with $\sigma = 0$ there is no chance in the data",
                _ => "no test: one point, and no spread to estimate",
            };
            f.text_px(min, why, if lay.narrow { lay.text } else { lay.big }, pal::TEXT, Align2::LEFT_TOP);
            return;
        };
        let a = alpha_text(s.alpha);
        let (relation, verdict) = if o.rejects { (r"\le", "reject the null") } else { (">", "do not reject the null") };
        let first = format!(r"this sample: $p {} {relation} \alpha = {a}$: {verdict}", is(o.p));
        let real = s.effect != 0.0;
        let (second, color) = match (real, o.rejects) {
            (false, true) => ("but there is no effect: a false alarm, an error of the first kind", ALARM),
            (false, false) => ("and there is no effect: right", pal::DIM),
            (true, true) => ("and the effect is real: right", TRUTH),
            (true, false) => ("but the effect is real: a miss, an error of the second kind", ALARM),
        };
        let held = if self.held.is_some() { " (the statistic set by hand)" } else { "" };
        let second = format!("{second}{held}");
        if lay.compact {
            let short = format!(r"$p {}$: {verdict}", is(o.p));
            f.text_px(min, short, lay.text, pal::TEXT, Align2::LEFT_TOP);
            return;
        }
        if lay.narrow {
            f.text_px(min, first, lay.text, pal::TEXT, Align2::LEFT_TOP);
            f.text_px(min + Vec2::new(0.0, lay.text + 5.0), second, lay.small, color.mix(pal::WHITE, 0.2), Align2::LEFT_TOP);
        } else {
            let h = wrapped_px(f, min, width, &first, lay.big, pal::TEXT);
            wrapped_px(f, min + Vec2::new(0.0, h + 2.0), width, &second, lay.text, color.mix(pal::WHITE, 0.2));
        }
    }

    /// What the shadings are, with their numbers.
    fn draw_key(&self, f: &mut Frame, lay: &Layout) {
        let s = &self.setup;
        let (min, max) = lay.key;
        if max.x - min.x < 60.0 {
            return;
        }
        let Some(o) = self.observed() else { return };
        let power = self.power();
        let truth = s.effect != 0.0;
        let a = alpha_text(s.alpha);
        let mut entries: Vec<(Color, String, String)> = vec![
            (ASSUMED.a(0.7), format!(r"$\alpha = {a}$"), "the null's share where it is rejected".to_string()),
            (ALARM, format!(r"$p {}$", is(o.p)), "the null's share beyond the statistic".to_string()),
        ];
        if truth && power.is_finite() {
            entries.push((TRUTH.a(0.8), format!("power {}", chance(power)), "the truth's share where the null is rejected".to_string()));
            entries.push((TRUTH.a(0.12), format!(r"$\beta {}$", is(1.0 - power)), "the truth's share where it is not: misses".to_string()));
        }
        let size = lay.small;
        let swatch = |f: &mut Frame, at: Vec2, color: Color| plot::fill(f, (at + Vec2::new(0.0, 2.0), at + Vec2::new(size - 1.0, size + 1.0)), color);
        if lay.narrow {
            // one row of names and numbers alone
            let mut x = min.x;
            for (color, number, _) in &entries {
                let w = f.text_size(number, size).x;
                if x + size + 5.0 + w > max.x {
                    break;
                }
                swatch(f, Vec2::new(x, min.y), *color);
                f.text_px(Vec2::new(x + size + 5.0, min.y), number.as_str(), size, pal::TEXT, Align2::LEFT_TOP);
                x += size + 5.0 + w + 14.0;
            }
        } else {
            for (k, (color, number, meaning)) in entries.iter().enumerate() {
                let at = Vec2::new(min.x, min.y + k as f32 * (size + 7.0));
                swatch(f, at, *color);
                let text = format!("{number}: {meaning}");
                f.text_px(at + Vec2::new(size + 7.0, 0.0), text, size, pal::TEXT, Align2::LEFT_TOP);
            }
        }
    }

    /// How to read the picture, for a visitor who has not opened the panel.
    fn draw_note(&self, f: &mut Frame, lay: &Layout) {
        let (min, max) = lay.note;
        if max.x - min.x < 100.0 || max.y - min.y < 100.0 {
            return;
        }
        let name = if self.setup.estimated { "$t$" } else { "$z$" };
        let truth = if self.setup.effect == 0.0 {
            format!("Here there is no effect: the truth is the null, and the purple curve is also where {name} really falls.")
        } else {
            format!("The green curve is where {name} really falls; its share beyond the critical values is the power.")
        };
        let note = format!(
            r"How to read it. The purple curve is where the statistic {name} would fall, sample after sample, if the null were true. The null is rejected when {name} lands beyond the critical values (dashed), in the purple tails: if the null is true, that happens with chance $\alpha$. The pink area beyond this sample's {name} is its $p$-value. {truth} Many studies, each a new sample from the truth tested the same way, show it: their {name}s fill the outlined bars, and the table counts how many rejected."
        );
        wrapped_px(f, min, max.x - min.x, &note, lay.small, pal::DIM);
    }

    /// The table of many studies: of those with no effect and of those with a real
    /// one, how many were rejected, as a bar each, with a mark where its share should
    /// settle (α, and the power).
    fn draw_table(&self, f: &mut Frame, lay: &Layout) {
        let (min, max) = lay.table;
        if max.x - min.x < 80.0 || max.y - min.y < 60.0 {
            return;
        }
        let s = &self.setup;
        let st = &self.studies;
        let t = &st.tally;
        let size = lay.small;
        let done = st.len();
        if s.no_test().is_some() {
            f.text_px(min, "many studies: no test to make", lay.text, pal::DIM, Align2::LEFT_TOP);
            return;
        }
        let head = if done == 0 {
            "many studies: none yet".to_string()
        } else {
            format!("many studies: {} of {}", count(done), count(MOST_STUDIES))
        };
        f.text_px(min, head, lay.text, pal::TEXT, Align2::LEFT_TOP);
        // (two lines a row where there is room for them; on a phone, one)
        let tall = max.y - min.y >= 150.0;
        let mut y = min.y + lay.text + 8.0;
        let h = if tall { 10.0 } else { 7.0 };
        let row_h = if tall { 2.0 * size + h + 20.0 } else { size + h + 12.0 };
        let shares = &self.look.table.shown;
        let power = self.power();
        let a = alpha_text(s.alpha);
        for real in [false, true] {
            let all = t.of(real);
            let rejected = t.rejected(real);
            let name = if real { "real effect" } else { "no effect" };
            let color = origin_color(real).mix(pal::WHITE, 0.25);
            let measured = num(rejected as f64 / all.max(1) as f64, 3);
            let expect = if real { format!("power {}", chance(power)) } else { format!(r"$\alpha = {a}$") };
            let (first, second) = match (all > 0, tall) {
                (false, _) => (format!("{name}: no such studies in this run"), None),
                (true, true) => (
                    format!("{name}: {} studies", count(all)),
                    Some(if real {
                        format!("rejected {} ({measured}), {expect}; missed {}", count(rejected), count(all - rejected))
                    } else {
                        format!("rejected {}: false alarms ({measured}), {expect}", count(rejected))
                    }),
                ),
                (true, false) => (format!("{name}: rejected {} of {} ({measured}), {expect}", count(rejected), count(all)), None),
            };
            f.text_px(Vec2::new(min.x, y), first, size, color, Align2::LEFT_TOP);
            let bar_top = y + size + 4.0;
            let (x0, x1) = (min.x, max.x);
            let (yes, no) = if real { (TRUTH, ALARM) } else { (ALARM, KEPT) };
            if all > 0 {
                let share = shares.get(real as usize).copied().unwrap_or(0.0).clamp(0.0, 1.0);
                let mid = x0 + (x1 - x0) * share;
                plot::fill(f, (Vec2::new(x0, bar_top), Vec2::new(mid, bar_top + h)), yes.a(0.85));
                plot::fill(f, (Vec2::new(mid, bar_top), Vec2::new(x1, bar_top + h)), no.a(0.45));
                // where the share should settle: α with no effect, the power with one
                // (each row's mark a kind of its own: they come and go apart)
                let settle = shares.get(2 + real as usize).copied().unwrap_or(0.0).clamp(0.0, 1.0);
                let x = x0 + (x1 - x0) * settle;
                plot::line_of(f, TABLE + real as u8, Vec2::new(x, bar_top - 3.0), Vec2::new(x, bar_top + h + 3.0), 2.0, pal::WHITE);
            } else {
                plot::fill(f, (Vec2::new(x0, bar_top), Vec2::new(x1, bar_top + h)), pal::DIM.a(0.12));
            }
            if let Some(second) = second {
                f.text_px(Vec2::new(min.x, bar_top + h + 4.0), second, size, pal::DIM, Align2::LEFT_TOP);
            }
            y += row_h;
        }
        if let (Some(share), true) = (t.false_share(), t.of(false) > 0 && t.of(true) > 0) {
            let all = t.rejected(false) + t.rejected(true);
            let text = format!("of the {} rejected, {} % had no effect", count(all), num(100.0 * share, 0));
            f.text_px(Vec2::new(min.x, y), text, size, ALARM.mix(pal::WHITE, 0.3), Align2::LEFT_TOP);
        }
    }
}

/// Every kind of line and point this scene draws, asked for in the same order at the
/// start of every frame, each with one unseen point: a batch not drawn in a frame
/// would drop out and move every later one up a place, and `eidos motion` would then
/// take one batch for another (the table's two marks, say).
fn hold_batches(f: &mut Frame) {
    let corner = f.hud(Vec2::ZERO);
    for kind in [AXES, CURVES, MARKS, plot::BUTTONS, RULER, CRITICAL, REGION, TABLE, TABLE + 1, ARROWS, HIST, HIST + 1] {
        plot::apart(f, kind, false).segment(corner, corner, 1.0, Color::CLEAR);
    }
    for kind in [MARKS, CRITICAL] {
        plot::apart(f, kind, true).segment(corner, corner, 1.0, Color::CLEAR);
    }
    plot::disc(f, Vec2::ZERO, 1.0, Color::CLEAR);
}

/// A height for point `i` of the strip, from 0 to 1, the same for every sample.
fn jitter(i: usize) -> f64 {
    let mut x = (i as u64).wrapping_add(0x9E37_79B9_7F4A_7C15);
    x = (x ^ (x >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    x = (x ^ (x >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    x ^= x >> 31;
    (x >> 11) as f64 / (1u64 << 53) as f64
}

/// Text broken into lines no wider than `width`, from `at` down, in logical pixels of
/// the screen (`kit::wrapped` takes the plane's own units). Returns the height taken.
fn wrapped_px(f: &mut Frame, at: Vec2, width: f32, text: &str, size: f32, color: Color) -> f32 {
    let ppu = pixels_per_unit(f).max(1e-6);
    let Some(world) = f.cam.pick_ground(at * f.px) else { return 0.0 };
    let w = width / ppu;
    wrapped(f, world.truncate(), w, text, size, color);
    wrapped_height(f, w, text, size)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn close(a: f64, b: f64, tol: f64) -> bool {
        (a - b).abs() <= tol
    }

    fn setup(effect: f64, n: usize, alpha: f64, tails: Tails, estimated: bool) -> Setup {
        Setup { effect, sigma: 1.0, n, alpha, tails, estimated, real: 1.0 }
    }

    /// An experiment as its button sets it up, with its run of studies made to the end.
    fn run(k: usize) -> Testing {
        let mut s = Testing::create();
        s.set_up(k);
        s.finish(MOST_STUDIES);
        s
    }

    /// The critical values and p-values every textbook prints, from `Dist`.
    #[test]
    fn critical_values_and_p_values_are_the_textbooks() {
        let z = setup(0.5, 20, 0.05, Tails::Two, false);
        assert!(close(z.critical().1, 1.959_964, 1e-6) && close(z.critical().0, -1.959_964, 1e-6));
        assert!(close(Setup { tails: Tails::Upper, ..z }.critical().1, 1.644_854, 1e-6));
        assert_eq!(Setup { tails: Tails::Upper, ..z }.critical().0, f64::NEG_INFINITY);
        assert!(close(Setup { alpha: 0.005, ..z }.critical().1, 2.807_034, 1e-6));
        // Student's t with 19 degrees of freedom
        let t = Setup { estimated: true, ..z };
        assert!(close(t.critical().1, 2.093_024, 1e-6));
        // p at the critical value is α, and halves with one tail on the side of the statistic
        assert!(close(z.p_value(1.959_964), 0.05, 1e-6) && close(t.p_value(2.093_024), 0.05, 1e-6));
        assert!(close(z.p_value(1.8), 2.0 * Setup { tails: Tails::Upper, ..z }.p_value(1.8), 1e-15));
        assert!(close(z.p_value(-1.0), 0.317_310_5, 1e-7));
        // a one-tailed test the wrong way: nearly 1
        assert!(Setup { tails: Tails::Upper, ..z }.p_value(-3.0) > 0.998);
        // far out the p-value is tiny but not zero
        let far = z.p_value(14.0);
        assert!(far > 0.0 && far < 1e-40, "{far}");
        assert_eq!(chance(far), "< 0.0001");
    }

    /// The power of the z test is the formula, exactly; of the t test, the noncentral t
    /// distribution's (values from SciPy's `nct`).
    #[test]
    fn the_power_is_the_formulas() {
        for (d, n, a, tails) in [(0.5, 20usize, 0.05, Tails::Two), (0.5, 32, 0.05, Tails::Two), (0.5, 20, 0.005, Tails::Two), (0.5, 20, 0.05, Tails::Upper), (-0.3, 50, 0.01, Tails::Two)] {
            let s = setup(d, n, a, tails, false);
            let delta = d * (n as f64).sqrt();
            let exact = match tails {
                Tails::Two => {
                    let c = st::normal_quantile(1.0 - a / 2.0);
                    normal_cdf(delta - c) + normal_cdf(-delta - c)
                }
                Tails::Upper => 1.0 - normal_cdf(st::normal_quantile(1.0 - a) - delta),
            };
            assert!(close(s.power(), exact, 1e-12), "{s:?}: {} against {exact}", s.power());
        }
        assert!(close(setup(0.5, 20, 0.05, Tails::Two, false).power(), 0.608_779, 1e-6));
        let scipy = [
            (0.5, 20usize, 0.05, Tails::Two, 0.564_504_410_5),
            (0.5, 34, 0.05, Tails::Two, 0.807_777_497_1),
            (0.5, 20, 0.005, Tails::Two, 0.212_742_604_3),
            (0.5, 2, 0.05, Tails::Two, 0.061_948_606_6),
            (0.5, 3, 0.05, Tails::Two, 0.084_107_055_8),
            (1.0, 5, 0.05, Tails::Two, 0.401_389_916_2),
            (0.5, 20, 0.05, Tails::Upper, 0.695_149_330_8),
            (0.2, 200, 0.01, Tails::Two, 0.590_571_010_2),
            (0.02, 40_000, 0.05, Tails::Two, 0.979_321_842_4),
        ];
        for (d, n, a, tails, want) in scipy {
            let s = setup(d, n, a, tails, true);
            assert!(close(s.power(), want, 1e-5), "{s:?}: {} against {want}", s.power());
        }
        // with no effect the power is α, with z and with t
        for estimated in [false, true] {
            let s = setup(0.0, 12, 0.05, Tails::Two, estimated);
            assert!(close(s.power(), 0.05, 1e-6), "{s:?}: {}", s.power());
        }
        // the noncentral t's density integrates to 1 and has SciPy's values
        let spread = Spread::estimated(19.0, 120);
        let delta = 0.5 * 20f64.sqrt();
        let total: f64 = (0..=4000).map(|i| -10.0 + 20.0 * i as f64 / 4000.0).map(|x| spread.density(x, delta) * 0.005).sum();
        assert!(close(total, 1.0, 1e-3), "{total}");
        assert!(close(Spread::estimated(19.0, 120).density(2.0, 2.236), 0.367_191_4, 2e-4));
        assert!(close(Spread::estimated(1.0, 120).density(0.5, 1.0), 0.283_160_99, 2e-3));
        assert!(close(Spread::estimated(4.0, 120).density(-3.0, 1.5), 0.000_626_24, 1e-4));
    }

    /// The least n for power 0.8: 32 with σ known, 34 with it estimated (half a standard
    /// deviation, two tails, α = 0.05), 54 at α = 0.005; none with no effect, or with
    /// one tail on the wrong side.
    #[test]
    fn the_n_for_power_eight_tenths() {
        let z = setup(0.5, 20, 0.05, Tails::Two, false);
        assert_eq!(z.n_for(0.8), Some(32));
        assert!(Setup { n: 31, ..z }.power() < 0.8 && Setup { n: 32, ..z }.power() >= 0.8);
        assert_eq!(Setup { estimated: true, ..z }.n_for(0.8), Some(34));
        assert!(Setup { n: 33, estimated: true, ..z }.power() < 0.8);
        assert_eq!(Setup { alpha: 0.005, ..z }.n_for(0.8), Some(54));
        assert_eq!(Setup { effect: 0.0, ..z }.n_for(0.8), None);
        assert_eq!(Setup { effect: -0.5, tails: Tails::Upper, ..z }.n_for(0.8), None);
        assert_eq!(Setup { effect: 2.0, ..z }.n_for(0.8), Some(2));
    }

    /// Student's t by simulation: samples drawn point by point and tested reject as
    /// often as the power written says, and as often as α when there is no effect.
    #[test]
    fn simulated_studies_agree_with_the_numbers_written() {
        for (d, n, estimated) in [(0.5, 20usize, true), (0.5, 20, false), (0.0, 20, true), (0.0, 3, true), (1.0, 5, true), (0.0, 1500, true), (0.1, 1500, true)] {
            let s = setup(d, n, 0.05, Tails::Two, estimated);
            let k = if n > 100 { 20_000 } else { 40_000 };
            let hits = (0..k)
                .filter(|&i| {
                    let (_, z, t) = study(&s, 7, i);
                    s.rejects(if estimated { t } else { z })
                })
                .count();
            let share = hits as f64 / k as f64;
            let want = s.power();
            let se = (want * (1.0 - want) / k as f64).sqrt();
            assert!((share - want).abs() < 3.5 * se, "{s:?}: {share} of {k} against {want}");
        }
    }

    /// Samples of more than `DIRECT` points go by their mean and spread alone: the same
    /// distributions as drawing the points.
    #[test]
    fn large_samples_are_drawn_as_their_points_would_be() {
        let s = setup(0.0, DIRECT + 1, 0.05, Tails::Two, true);
        let k = 20_000;
        let ts: Vec<f64> = (0..k).map(|i| study(&s, 3, i).2).collect();
        let zs: Vec<f64> = (0..k).map(|i| study(&s, 3, i).1).collect();
        // standard normal near enough (t with 1000 degrees of freedom has variance 1.002)
        assert!(st::mean(&ts).abs() < 0.03 && close(st::variance(&ts), 1.002, 0.035), "{} {}", st::mean(&ts), st::variance(&ts));
        assert!(st::mean(&zs).abs() < 0.03 && close(st::variance(&zs), 1.0, 0.035));
        // and t differs from z only by the spread estimate: s/σ near 1, give or take 1/√(2k)
        let ratios: Vec<f64> = (0..k).filter(|&i| zs[i].abs() > 0.5).map(|i| zs[i] / ts[i]).collect();
        let sd = st::sd(&ratios);
        assert!(close(st::mean(&ratios), 1.0, 0.003) && close(sd, (0.5 / DIRECT as f64).sqrt(), 0.003), "{} {sd}", st::mean(&ratios));
    }

    /// The numbers each experiment's caption says, from the scene as the button sets it up.
    #[test]
    fn the_experiments_show_what_their_captions_say() {
        let says = |k: usize, what: &str| assert!(TRIALS[k].caption.contains(what), "experiment {k} should say \"{what}\": {}", TRIALS[k].caption);
        let p = |s: &Testing| s.observed().map_or(f64::NAN, |o| o.p);
        // nothing going on: this sample is a false alarm; the false alarms settle near α
        let s = run(0);
        assert!(s.setup.effect == 0.0 && s.observed().is_some_and(|o| o.rejects));
        says(0, &format!("$p$ is {}", chance(p(&s))));
        let share = s.studies.tally.share(false).unwrap_or(f64::NAN);
        assert!(s.studies.tally.of(true) == 0 && (share - 0.05).abs() < 3.0 * (0.05f64 * 0.95 / 1e4).sqrt(), "{share}");
        says(0, &format!("{} after 10 000 studies", num(share, 3)));
        // a real effect, too little data: power 0.61 (0.56 with t), this sample a miss
        let s = run(1);
        assert!(s.observed().is_some_and(|o| !o.rejects));
        says(1, &format!("the power is {}", num(s.power(), 2)));
        says(1, &format!("($p = {}$)", chance(p(&s))));
        says(1, &format!("power is {}: nearly half", num(Setup { estimated: true, ..s.setup }.power(), 2)));
        let share = s.studies.tally.share(true).unwrap_or(f64::NAN);
        assert!((share - s.power()).abs() < 0.015 && share > 0.55 && share < 0.65, "{share}");
        // enough data: n = 32 is the least for 0.8, 34 with t
        let s = run(2);
        assert_eq!(s.n_for(), Some(32));
        assert_eq!(s.setup.n, 32);
        says(2, &format!("(it is {})", num(s.power(), 2)));
        says(2, &format!("$\\delta = {}$ to {}", num(Setup { n: 20, ..s.setup }.delta(), 2), num(s.setup.delta(), 2)));
        says(2, &format!("takes $n = {}$", Setup { estimated: true, ..s.setup }.n_for(0.8).unwrap_or(0)));
        assert!(s.observed().is_some_and(|o| o.rejects));
        // significant is not large: p < 0.0001, and 98 % reject
        let s = run(3);
        assert!(p(&s) < 1e-4);
        says(3, "$p < 0.0001$");
        says(3, &format!("{} % of such studies", num(100.0 * s.power(), 0)));
        let share = s.studies.tally.share(true).unwrap_or(f64::NAN);
        assert!((share - 0.98).abs() < 0.005, "{share}");
        // the mean is a fiftieth of the spread from μ₀
        assert!(s.observed().is_some_and(|o| o.xbar.abs() < 0.03));
        // one tail or two: the p-value halves, from 0.068 to 0.034, and rejects with one tail
        let s = run(4);
        let one = s.observed().unwrap();
        let two = Setup { tails: Tails::Two, ..s.setup }.p_value(one.stat);
        assert!(one.rejects && !Setup { tails: Tails::Two, ..s.setup }.rejects(one.stat));
        assert!(close(two, 2.0 * one.p, 1e-12));
        says(4, &format!("from {} with two tails", chance(two)));
        says(4, &format!("to {}.", chance(one.p)));
        says(4, &format!("from {} to {}", num(Setup { tails: Tails::Two, ..s.setup }.critical().1, 2), num(s.setup.critical().1, 2)));
        // choosing the tail afterwards rejects at |z| ≥ 1.645: a two-tailed test at 0.10
        assert!(close(Setup { tails: Tails::Two, alpha: 0.10, ..s.setup }.critical().1, s.setup.critical().1, 1e-9));
        // a stricter threshold: 2.81, the power from 0.61 to 0.28, n = 54; this sample
        // rejected at 0.05 but not at 0.005
        let s = run(5);
        let o = s.observed().unwrap();
        assert!(!o.rejects && Setup { alpha: 0.05, ..s.setup }.rejects(o.stat));
        says(5, &format!("out to {}", num(s.setup.critical().1, 2)));
        says(5, &format!("$p = {}$", chance(o.p)));
        says(5, &format!("from {} to {}", num(Setup { alpha: 0.05, ..s.setup }.power(), 2), num(s.power(), 2)));
        says(5, &format!("takes $n = {}$", s.n_for().unwrap_or(0)));
        let share = s.studies.tally.share(true).unwrap_or(f64::NAN);
        assert!((share - s.power()).abs() < 0.015, "{share}");
        // the p-value is not the chance the null is true: about 36 % of rejections are false alarms
        let s = run(6);
        says(6, &format!("power {}", num(s.power(), 2)));
        let expected = 0.9 * 0.05 / (0.9 * 0.05 + 0.1 * s.power());
        says(6, &format!("about {} %", num(100.0 * expected, 0)));
        says(6, &format!("$0.1 \\times {} = {}$", num(s.power(), 2), num(0.1 * s.power(), 3)));
        let t = &s.studies.tally;
        let real_share = t.of(true) as f64 / s.studies.len() as f64;
        let measured = t.false_share().unwrap_or(f64::NAN);
        assert!((real_share - 0.1).abs() < 0.01 && (measured - expected).abs() < 0.03, "{real_share} real, {measured} of rejections false against {expected}");
        assert!(measured > 1.0 / 3.0);
    }

    /// The scene opens as the second experiment sets it up.
    #[test]
    fn the_scene_opens_on_the_second_experiment() {
        let opened = Testing::create();
        let mut pressed = Testing::create();
        pressed.set_up(1);
        assert_eq!((opened.setup, opened.seed), (pressed.setup, pressed.seed));
    }

    /// Edges: σ = 0, one point with the spread estimated, α at 0 and 1, an effect so
    /// large that the power is 1, a statistic set far beyond the chart.
    #[test]
    fn degenerate_settings_are_told_honestly() {
        let mut s = Testing::create();
        s.setup.sigma = 0.0;
        s.refresh();
        assert_eq!(s.setup.no_test(), Some(NoTest::NoNoise));
        assert!(s.observed().is_none() && !s.studying());
        assert!(frame_goal(&s.setup).iter().all(|v| v.is_finite()));
        s.setup.sigma = 1.0;
        s.setup.n = 1;
        s.setup.estimated = true;
        s.refresh();
        assert_eq!(s.setup.no_test(), Some(NoTest::OnePoint));
        assert!(s.observed().is_none() && s.n_for().is_some());
        assert!(frame_goal(&s.setup).iter().all(|v| v.is_finite()));
        // one point with σ known is a test like any other
        s.setup.estimated = false;
        s.refresh();
        assert!(s.observed().is_some_and(|o| o.p > 0.0 && o.p <= 1.0));
        // α = 0: never reject, power 0; α = 1: always, power 1
        s.setup = setup(0.5, 20, 0.0, Tails::Two, false);
        s.refresh();
        assert_eq!(s.setup.critical(), (f64::NEG_INFINITY, f64::INFINITY));
        assert!(s.power() == 0.0 && !s.observed().unwrap().rejects);
        assert!(frame_goal(&s.setup).iter().all(|v| v.is_finite()));
        for tails in [Tails::Two, Tails::Upper] {
            s.setup = setup(0.5, 20, 1.0, tails, true);
            s.refresh();
            assert!(close(s.power(), 1.0, 1e-12) && s.observed().unwrap().rejects, "{tails:?}: {}", s.power());
            assert!(frame_goal(&s.setup).iter().all(|v| v.is_finite()));
        }
        // power 1, written as more than 0.9999
        s.setup = setup(2.0, 200, 0.05, Tails::Two, true);
        s.refresh();
        assert!(s.power() > 0.9999 && chance(s.power()) != "1.0000", "{}", chance(s.power()));
        // the statistic far beyond the chart: a number, written as < 0.0001
        s.held = Some(1e6);
        let o = s.observed().unwrap();
        assert!(o.p >= 0.0 && o.p.is_finite() && is(o.p) == "< 0.0001" && o.xbar.is_finite() && o.shift.is_finite());
        // (t's tails are heavy: at −30 with 199 degrees of freedom, near 10⁻⁷⁴)
        s.held = Some(-30.0);
        let p = s.observed().unwrap().p;
        assert!(p > 0.0 && p < 1e-60, "{p}");
        // a run under the null and every shape of setting keeps its numbers finite
        for (effect, sigma, n, alpha, estimated) in [(0.0, 3.0, 2, 0.5, true), (2.0, 0.01, 40_000, 1e-4, true), (-2.0, 3.0, 1, 0.9, false)] {
            s.setup = Setup { effect, sigma, n, alpha, tails: Tails::Two, estimated, real: 0.5 };
            s.held = None;
            s.finish(200);
            assert!(s.studies.z.iter().all(|z| z.is_finite()), "{:?}", s.setup);
            assert!(s.studies.heights().iter().all(|h| h.is_finite()));
            assert!(s.power().is_finite() && s.observed().is_some_and(|o| o.stat.is_finite() && o.p.is_finite()), "{:?}", s.setup);
        }
    }

    /// The probabilities as they are written.
    #[test]
    fn chances_are_written_without_lying() {
        assert_eq!(chance(0.25), "0.25");
        assert_eq!(chance(0.0151), "0.015");
        assert_eq!(chance(0.0047), "0.0047");
        assert_eq!(chance(0.000_34), "0.00034");
        assert_eq!(chance(3e-9), "< 0.0001");
        assert_eq!(chance(0.999_99), "> 0.9999");
        assert_eq!(chance(0.9991), "0.9991");
        assert_eq!(chance(1.0), "1");
        assert_eq!(is(0.25), "= 0.25");
        assert_eq!(is(1e-7), "< 0.0001");
        assert_eq!((readable(chance(1e-7)), readable(chance(0.25))), ("$< 0.0001$".to_string(), "0.25".to_string()));
        assert_eq!((alpha_text(0.05), alpha_text(0.005), alpha_text(0.1), alpha_text(1.0), alpha_text(0.0), alpha_text(0.0001)), ("0.05".into(), "0.005".into(), "0.1".into(), "1".into(), "0".into(), "0.0001".into()));
    }

    /// Changing α, the tails or the test counts the same studies again; changing what
    /// the studies are made from starts them afresh.
    #[test]
    fn a_change_of_rule_recounts_the_same_studies() {
        let mut s = Testing::create();
        s.finish(2000);
        let z: Vec<f64> = s.studies.z.clone();
        let before = s.studies.tally;
        s.setup.alpha = 0.005;
        s.refresh();
        assert_eq!(s.studies.z, z);
        assert!(s.studies.tally.rejected(true) < before.rejected(true));
        assert_eq!(s.studies.tally.of(true), 2000);
        s.setup.n = 21;
        s.refresh();
        assert_eq!(s.studies.len(), 0);
    }

    /// The layout keeps everything in the window and apart, from a phone to a desktop.
    #[test]
    fn the_layout_fits_every_window() {
        for (w, h) in [(1075.0, 1000.0), (1100.0, 700.0), (1600.0, 1000.0), (390.0, 800.0), (360.0, 640.0), (700.0, 500.0), (300.0, 600.0), (300.0, 520.0), (500.0, 400.0), (240.0, 180.0)] {
            let lay = Layout::of(Vec2::new(w, h), 40.0);
            let inside = |(a, b): Rect| a.x >= 0.0 && a.y >= 0.0 && b.x <= w + 0.5 && b.y <= h + 0.5 && a.is_finite() && b.is_finite();
            let apart = |(a, b): Rect, (c, d): Rect| b.x <= c.x || d.x <= a.x || b.y <= c.y || d.y <= a.y;
            assert!(inside(lay.chart), "{w}x{h}: {:?}", lay.chart);
            if lay.compact {
                continue;
            }
            let buttons: Vec<Rect> = [lay.tails, lay.test, lay.sample, lay.run].concat();
            let boxes: Vec<Rect> = [vec![lay.strip, lay.chart, lay.verdict, lay.key, lay.table], buttons.clone()].concat();
            let boxes: Vec<Rect> = if lay.note.1.y > lay.note.0.y + 2.0 { [boxes, vec![lay.note]].concat() } else { boxes };
            for (i, a) in boxes.iter().enumerate() {
                assert!(inside(*a), "{w}x{h}: box {i} {a:?}");
                for b in &boxes[i + 1..] {
                    assert!(apart(*a, *b), "{w}x{h}: {a:?} and {b:?} overlap");
                }
            }
            assert!(lay.chart.1.y - lay.chart.0.y > 150.0 && lay.chart.1.x - lay.chart.0.x > 200.0, "{w}x{h}: {:?}", lay.chart);
        }
    }

    /// The picture's glides come to rest after a change.
    #[test]
    fn glides_arrive_and_stop() {
        let mut s = Testing::create();
        s.running = false;
        s.follow(0.016, true, false);
        assert!(!s.look.moving);
        for k in [3, 0, 5, 4] {
            s.set_up(k);
            s.running = false;
            let mut frames = 0;
            s.follow(1.0 / 60.0, false, false);
            while s.look.moving {
                s.follow(1.0 / 60.0, false, false);
                frames += 1;
                assert!(frames < 300, "experiment {k} still moving");
            }
            assert!(frames > 3, "experiment {k}: {frames}");
        }
    }

    /// The statistics of many studies fall as the green curve says: a chi-square check
    /// of their histogram against it, for z, and for t with five points (where the
    /// noncentral t is far from a bell).
    #[test]
    fn the_studies_fall_as_the_truths_curve_says() {
        for (effect, n, estimated) in [(0.5, 32usize, false), (0.5, 20, false), (1.0, 5, true), (0.5, 20, true)] {
            let mut s = Testing::create();
            s.setup = setup(effect, n, 0.05, Tails::Two, estimated);
            s.finish(MOST_STUDIES);
            let (d, spread) = (s.setup.delta(), s.setup.spread(400));
            let total = s.studies.len() as f64;
            let (mut chi, mut cells) = (0.0, 0usize);
            for b in 0..BINS {
                let x0 = BINS_FROM + b as f64 * BIN;
                let expect = total * (spread.below(x0 + BIN, d) - spread.below(x0, d));
                if expect >= 5.0 {
                    chi += (s.studies.bins[1][b] as f64 - expect).powi(2) / expect;
                    cells += 1;
                }
            }
            let limit = Dist::ChiSquare { df: (cells - 1) as f64 }.quantile(0.995);
            assert!(chi < limit, "{:?}: chi-square {chi} over {cells} cells, limit {limit}", s.setup);
        }
    }

    /// The samples behind the captions: for each experiment, the seeds whose sample
    /// shows what it should (run with --ignored --nocapture).
    #[test]
    #[ignore]
    fn explore_seeds() {
        for (k, trial) in TRIALS.iter().enumerate() {
            let mut line = format!("{k} {:<48}", trial.button);
            let mut s = Testing::create();
            s.set_up(k);
            for seed in 0..400u64 {
                s.seed = seed;
                s.refresh();
                if let Some(o) = s.observed() {
                    let two = Setup { tails: Tails::Two, ..s.setup }.p_value(o.stat);
                    let keep = match k {
                        0 => o.p > 0.012 && o.p < 0.03,
                        1 => o.p > 0.18 && o.p < 0.32,
                        2 => o.stat > 2.6 && o.stat < 3.1,
                        3 => o.stat > 3.95 && o.stat < 4.3,
                        4 => o.stat > 1.75 && o.stat < 1.85,
                        5 => o.p > 0.012 && o.p < 0.03,
                        _ => o.stat > 2.6 && o.stat < 3.1,
                    };
                    if keep && line.len() < 300 {
                        line += &format!("  {seed}:{:.3}/{:.4}/{:.4}", o.stat, o.p, two);
                    }
                }
            }
            println!("{line}");
        }
    }

    /// The counts of each experiment's run (run with --ignored --nocapture).
    #[test]
    #[ignore]
    fn explore_runs() {
        for k in 0..TRIALS.len() {
            let s = run(k);
            let t = &s.studies.tally;
            println!(
                "{k} {:<48} power {:.4}  no effect {:?} {:?}  real {:?} {:?}  false share {:?}  p {:?}",
                TRIALS[k].button,
                s.power(),
                t.share(false),
                t.of(false),
                t.share(true),
                t.of(true),
                t.false_share(),
                s.observed().map(|o| o.p)
            );
        }
    }
}

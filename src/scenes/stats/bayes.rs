//! Bayesian updating: belief about an unknown, held as a distribution and sharpened by
//! data.
//!
//! The unknown is a coin's chance of heads `p` (or, in the second mode, the mean of a
//! bell whose spread is known). What is believed about it before any data is the prior.
//! Each observation multiplies it by how probable that observation is at each value
//! (the likelihood), and the product, rescaled to have area 1, is the posterior. The
//! priors here are conjugate (a Beta for the coin, a bell for the bell's mean), so every
//! posterior has a formula: Beta(a + heads, b + tails), or a bell whose precision is the
//! prior's plus the data's. One prior is cut off: it rules out every rate above one
//! half, and its posterior can pile up against that wall but never pass it. Its share
//! below the wall can be far too small to hold as a number, so it is worked with through
//! its logarithm ([`ln_beta_below`]).

use super::plot::{self, button, even_row, fill, line_of, path_of, Glide, Plot, Rect, Ticks, AXES, CURVES, MARKS, OWN};
use super::*;
use crate::math::stats::{beta_i, ln_beta, normal_cdf, normal_quantile, Dist};

pub const DEF: SceneDef = SceneDef {
    id: "bayes",
    title: "Bayesian updating",
    group: ESTIMATION,
    blurb: "A prior belief, the likelihood of the data, and the posterior they make together: belief about an unknown rate sharpening as data arrive.",
    make: |_| Box::new(Bayes::create()),
};

/// The most observations kept: the run stops there.
const MOST: usize = 1000;
/// How much of a belief its intervals hold: the middle 95 %, 2.5 % left out on each side.
const LEVEL: f64 = 0.95;
/// The bell's spread, which is known (only its mean is not).
const SIGMA: f64 = 1.0;
/// Points along each curve (an odd number, so that one half is one of them).
const SAMPLES: usize = 401;
/// The second mode's axis: the bell's mean from −4 to 4.
const BELL_REACH: f64 = 4.0;
/// The least and greatest strength (a + b) a coin's prior can be given by its handles.
const LEAST_SURE: f64 = 2.0;
const MOST_SURE: f64 = 2000.0;
/// The least and greatest spread of a prior for the bell's mean.
const BELL_SD: [f64; 2] = [0.05, 2.0];

// ----- the mathematics -------------------------------------------------------------------

/// What is observed: a coin's tosses (1 for heads, 0 for tails) or values from a bell.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Model {
    Coin,
    Bell,
}

impl Model {
    /// The axis of the unknown.
    fn range(self) -> [f64; 2] {
        match self {
            Model::Coin => [0.0, 1.0],
            Model::Bell => [-BELL_REACH, BELL_REACH],
        }
    }

    /// One observation when the unknown's true value is `truth`.
    fn draw(self, truth: f64, rng: &mut Rng) -> f64 {
        match self {
            Model::Coin => (crate::math::stats::unit(rng) < truth) as u8 as f64,
            Model::Bell => truth + SIGMA * crate::math::stats::normal(rng),
        }
    }
}

/// What the data come to, all a posterior needs of them: how many observations, and
/// their sum (for the coin, the number of heads). Fractions of an observation are
/// allowed: the picture shows the newest one coming in.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
struct Counts {
    n: f64,
    sum: f64,
}

impl Counts {
    fn of(values: &[f64]) -> Counts {
        Counts { n: values.len() as f64, sum: values.iter().sum() }
    }

    /// The coin's heads and tails.
    fn heads(&self) -> f64 {
        self.sum.max(0.0)
    }

    fn tails(&self) -> f64 {
        (self.n - self.sum).max(0.0)
    }

    /// The estimate from the data alone: the share of heads, or the values' mean.
    fn estimate(&self) -> Option<f64> {
        (self.n > 0.0).then(|| self.sum / self.n)
    }
}

/// A belief about the unknown.
#[derive(Clone, Copy, Debug, PartialEq)]
enum Belief {
    /// Density in proportion to p^(a − 1) (1 − p)^(b − 1) from 0 to `cut`, and 0 above
    /// it (with `cut` = 1, the Beta distribution itself). a and b are at least 1 here, so
    /// that the density is nowhere infinite.
    Beta { a: f64, b: f64, cut: f64 },
    /// A bell with this mean and spread.
    Normal { mean: f64, sd: f64 },
}

/// k ln y, as 0 when k is 0 (whatever y is: 0⁰ = 1 here).
fn xlogy(k: f64, y: f64) -> f64 {
    if k == 0.0 {
        0.0
    } else {
        k * y.ln()
    }
}

impl Belief {
    /// Does it rule out the values above a wall?
    fn is_cut(&self) -> bool {
        matches!(*self, Belief::Beta { cut, .. } if cut < 1.0)
    }

    /// ln of the share of the uncut Beta(a, b) below the cut: 0 with no cut.
    fn ln_kept(a: f64, b: f64, cut: f64) -> f64 {
        if cut >= 1.0 {
            0.0
        } else {
            ln_beta_below(a, b, cut)
        }
    }

    /// The density, as a function (what does not depend on the value worked out once).
    fn density(&self) -> impl Fn(f64) -> f64 {
        let (a, b, cut, mean, sd, coin) = match *self {
            Belief::Beta { a, b, cut } => (a, b, cut, 0.0, 1.0, true),
            Belief::Normal { mean, sd } => (1.0, 1.0, 1.0, mean, sd, false),
        };
        let k = if coin { ln_beta(a, b) + Belief::ln_kept(a, b, cut) } else { 0.0 };
        move |x: f64| {
            if !coin {
                return crate::math::stats::normal_pdf((x - mean) / sd) / sd;
            }
            if !(x >= 0.0 && x <= cut.min(1.0)) {
                return 0.0;
            }
            (xlogy(a - 1.0, x) + xlogy(b - 1.0, 1.0 - x) - k).exp()
        }
    }

    fn pdf(&self, x: f64) -> f64 {
        self.density()(x)
    }

    /// The probability it gives to the unknown being above t.
    fn above(&self, t: f64) -> f64 {
        match *self {
            Belief::Beta { a, b, cut } => {
                if t >= cut {
                    0.0
                } else if t <= 0.0 {
                    1.0
                } else if cut >= 1.0 {
                    // (the upper tail directly: accurate where it is small)
                    beta_i(b, a, 1.0 - t)
                } else {
                    -(ln_beta_below(a, b, t) - Belief::ln_kept(a, b, cut)).exp_m1()
                }
            }
            Belief::Normal { mean, sd } => normal_cdf((mean - t) / sd),
        }
    }

    /// The value with a share q of the belief below it.
    fn quantile(&self, q: f64) -> f64 {
        match *self {
            Belief::Beta { a, b, cut } if cut >= 1.0 => Dist::Beta { a, b }.quantile(q),
            Belief::Beta { a, b, cut } => cut_quantile(a, b, cut, q),
            Belief::Normal { mean, sd } => Dist::Normal { mean, sd }.quantile(q),
        }
    }

    /// The middle `LEVEL` of it: as much left out below as above.
    fn interval(&self) -> (f64, f64) {
        let out = (1.0 - LEVEL) / 2.0;
        (self.quantile(out), self.quantile(1.0 - out))
    }

    fn mean(&self) -> f64 {
        match *self {
            Belief::Beta { a, b, cut } => {
                let uncut = a / (a + b);
                if cut >= 1.0 {
                    uncut
                } else {
                    // E[p | p ≤ cut] = a/(a + b) · I_cut(a + 1, b) / I_cut(a, b)
                    uncut * (ln_beta_below(a + 1.0, b, cut) - ln_beta_below(a, b, cut)).exp()
                }
            }
            Belief::Normal { mean, .. } => mean,
        }
    }

    /// Where the density is highest, and how high.
    fn peak(&self) -> (f64, f64) {
        let x = match *self {
            Belief::Beta { a, b, cut } => {
                let mode = if a + b > 2.0 { (a - 1.0) / (a + b - 2.0) } else { 0.5 };
                if a == 1.0 && b == 1.0 {
                    0.5 * cut
                } else {
                    mode.min(cut)
                }
            }
            Belief::Normal { mean, .. } => mean,
        };
        (x, self.pdf(x))
    }

    /// The belief after the data: the conjugate update.
    fn after(&self, c: Counts) -> Belief {
        match *self {
            Belief::Beta { a, b, cut } => Belief::Beta { a: a + c.heads(), b: b + c.tails(), cut },
            Belief::Normal { mean, sd } => {
                // precisions add, and the means are weighed by them
                let (before, data) = (1.0 / (sd * sd), c.n / (SIGMA * SIGMA));
                let precision = before + data;
                Belief::Normal { mean: (mean * before + c.sum / (SIGMA * SIGMA)) / precision, sd: 1.0 / precision.sqrt() }
            }
        }
    }

    /// The numbers it is glided by.
    fn params(&self) -> [f32; 3] {
        match *self {
            Belief::Beta { a, b, cut } => [a as f32, b as f32, cut as f32],
            Belief::Normal { mean, sd } => [mean as f32, sd.ln() as f32, 0.0],
        }
    }

    fn from_params(model: Model, p: &[f32]) -> Belief {
        let at = |i: usize| p.get(i).copied().unwrap_or(1.0) as f64;
        match model {
            Model::Coin => Belief::Beta { a: at(0).max(1.0), b: at(1).max(1.0), cut: at(2).max(0.05).min(1.0) },
            Model::Bell => Belief::Normal { mean: at(0), sd: at(1).exp().max(1e-6) },
        }
    }

    /// What it is worth against the data: for the coin, as many tosses as it counts for
    /// in the posterior's mean (a + b); for the bell, its precision.
    fn strength(&self) -> f64 {
        match *self {
            Belief::Beta { a, b, .. } => a + b,
            Belief::Normal { sd, .. } => 1.0 / (sd * sd),
        }
    }
}

/// ln I_x(a, b): the logarithm of the share of Beta(a, b) below x, accurate also where
/// that share is far too small to hold as a number (a posterior piled up against a wall
/// it may not pass). The continued fraction is the one `stats::beta_i` uses.
fn ln_beta_below(a: f64, b: f64, x: f64) -> f64 {
    if !(a > 0.0 && b > 0.0) || x.is_nan() {
        return f64::NAN;
    }
    if x <= 0.0 {
        return f64::NEG_INFINITY;
    }
    if x >= 1.0 {
        return 0.0;
    }
    let front = a * x.ln() + b * (-x).ln_1p() - ln_beta(a, b);
    if x < (a + 1.0) / (a + b + 2.0) {
        front + (beta_fraction(a, b, x) / a).ln()
    } else {
        // through the other tail, which is the small one here: I_x(a, b) = 1 − I_{1−x}(b, a)
        let other = (front + (beta_fraction(b, a, 1.0 - x) / b).ln()).exp();
        (-other).ln_1p()
    }
}

/// The continued fraction of the incomplete beta function (Lentz's method).
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
    for m in 1..1000 {
        let m = m as f64;
        let m2 = 2.0 * m;
        for aa in [m * (b - m) * x / ((qam + m2) * (a + m2)), -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))] {
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
        }
        if (d * c - 1.0).abs() < 1e-16 {
            break;
        }
    }
    h
}

/// The value with a share q of Beta(a, b) cut off above `cut` below it: Newton's steps on
/// the logarithm of the share, kept inside a bracket.
fn cut_quantile(a: f64, b: f64, cut: f64, q: f64) -> f64 {
    if q.is_nan() {
        return f64::NAN;
    }
    if q <= 0.0 {
        return 0.0;
    }
    if q >= 1.0 {
        return cut;
    }
    let want = q.ln() + ln_beta_below(a, b, cut);
    let lb = ln_beta(a, b);
    let (mut lo, mut hi) = (0.0f64, cut);
    let mut x = if a + b > 2.0 { ((a - 1.0) / (a + b - 2.0)).min(cut * 0.999) } else { 0.5 * cut };
    if x.is_nan() || x <= 0.0 {
        x = 0.5 * cut;
    }
    for _ in 0..200 {
        let below = ln_beta_below(a, b, x);
        let miss = below - want;
        if miss > 0.0 {
            hi = x;
        } else {
            lo = x;
        }
        // d/dx ln I_x = density / I_x
        let slope = (xlogy(a - 1.0, x) + xlogy(b - 1.0, 1.0 - x) - lb - below).exp();
        let next = x - miss / slope;
        let next = if next > lo && next < hi { next } else { 0.5 * (lo + hi) };
        if (next - x).abs() <= 1e-14 * x.max(1e-300) || hi - lo <= 1e-15 {
            return next;
        }
        x = next;
    }
    x
}

/// The likelihood of the data at each value of the unknown, as a share of its greatest
/// value (1 everywhere before there are data: they say nothing yet).
fn likelihood(model: Model, c: Counts) -> impl Fn(f64) -> f64 {
    let (h, t, n) = (c.heads(), c.tails(), c.n.max(0.0));
    let est = c.estimate().unwrap_or(0.5);
    // k ln(y / y0), 0 when k is 0
    let term = |k: f64, y: f64, y0: f64| if k <= 0.0 { 0.0 } else { k * (y.ln() - y0.ln()) };
    move |x: f64| {
        if n <= 0.0 {
            return 1.0;
        }
        match model {
            Model::Coin => {
                if !(0.0..=1.0).contains(&x) {
                    return 0.0;
                }
                (term(h, x, est) + term(t, 1.0 - x, 1.0 - est)).exp()
            }
            Model::Bell => (-n * (x - est) * (x - est) / (2.0 * SIGMA * SIGMA)).exp(),
        }
    }
}

/// The 95 % confidence interval from the data alone: Wilson's for the coin, the mean
/// plus and minus its critical value times σ/√n for the bell (σ known). None without
/// data.
fn confidence(model: Model, c: Counts) -> Option<(f64, f64)> {
    let est = c.estimate()?;
    let z = normal_quantile(0.5 + LEVEL / 2.0);
    Some(match model {
        Model::Coin => {
            let (n, z2) = (c.n, z * z);
            let centre = (est + z2 / (2.0 * n)) / (1.0 + z2 / n);
            let half = z / (1.0 + z2 / n) * (est * (1.0 - est) / n + z2 / (4.0 * n * n)).sqrt();
            (centre - half, centre + half)
        }
        Model::Bell => {
            let half = z * SIGMA / c.n.sqrt();
            (est - half, est + half)
        }
    })
}

/// The coin's prior with mean `m` whose interval's end on one side (`high`) is at `x`:
/// the strength found by halving (the end comes nearer the mean as the strength grows).
/// As near as the least and greatest strengths allow.
fn strength_for_end(m: f64, x: f64, high: bool) -> f64 {
    let least = LEAST_SURE.max(1.0 / m).max(1.0 / (1.0 - m));
    let out = (1.0 - LEVEL) / 2.0;
    let target = if high { 1.0 - out } else { out };
    // the share below x: rises with the strength when x is above the mean, falls when below
    let below = |s: f64| Dist::Beta { a: m * s, b: (1.0 - m) * s }.cdf(x);
    let too_wide = |s: f64| if high { below(s) < target } else { below(s) > target };
    let (mut lo, mut hi) = (least.ln(), MOST_SURE.ln());
    if !too_wide(least) {
        return least;
    }
    if too_wide(MOST_SURE) {
        return MOST_SURE;
    }
    for _ in 0..48 {
        let mid = 0.5 * (lo + hi);
        if too_wide(mid.exp()) {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    (0.5 * (lo + hi)).exp()
}

// ----- what can be chosen ------------------------------------------------------------------

/// The coin's priors to choose from: (name, prior).
const COIN_PRIORS: [(&str, Belief); 6] = [
    ("flat", Belief::Beta { a: 1.0, b: 1.0, cut: 1.0 }),
    ("fair, mildly sure", Belief::Beta { a: 5.0, b: 5.0, cut: 1.0 }),
    ("fair, very sure", Belief::Beta { a: 50.0, b: 50.0, cut: 1.0 }),
    ("favours tails", Belief::Beta { a: 4.0, b: 16.0, cut: 1.0 }),
    ("favours heads", Belief::Beta { a: 16.0, b: 4.0, cut: 1.0 }),
    (r"never above $\tfrac12$", Belief::Beta { a: 1.0, b: 1.0, cut: 0.5 }),
];

/// The same for the bell's mean.
const BELL_PRIORS: [(&str, Belief); 5] = [
    ("vague", Belief::Normal { mean: 0.0, sd: 2.0 }),
    ("near 0, mildly sure", Belief::Normal { mean: 0.0, sd: 1.0 }),
    ("near 0, very sure", Belief::Normal { mean: 0.0, sd: 0.2 }),
    ("below 0, sure", Belief::Normal { mean: -1.5, sd: 0.5 }),
    ("above 0, sure", Belief::Normal { mean: 1.5, sd: 0.5 }),
];

fn priors(model: Model) -> &'static [(&'static str, Belief)] {
    match model {
        Model::Coin => &COIN_PRIORS,
        Model::Bell => &BELL_PRIORS,
    }
}

/// How the observations so far are replayed.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Order {
    /// The tails first, then the heads, each in the order they came.
    TailsFirst,
    /// Shuffled.
    Shuffled,
    /// All of them at once.
    AtOnce,
}

/// One of the "things to try": every setting its lesson depends on.
struct Trial {
    button: &'static str,
    caption: &'static str,
    model: Model,
    /// The first observer's prior, and the second's (as an index into the model's
    /// priors), if there is a second.
    prior: usize,
    second: Option<usize>,
    truth: f64,
    seed: u64,
    /// The run stops after this many observations.
    stop: usize,
    rate: f32,
    faster: bool,
    threshold: f64,
    confidence: bool,
    /// Replay the first `stop` observations in this order (the seed's, made at once).
    order: Option<Order>,
}

/// The seed whose first ten tosses at a rate of 0.7 are seven heads, and whose first
/// hundred are seventy (see the tests).
const SEED: u64 = 23;

const TRIALS: [Trial; 7] = [
    Trial {
        button: "seven heads in ten",
        caption: r"A flat prior: every rate equally believable. After 7 heads in 10 tosses the posterior is $\mathrm{Beta}(8, 4)$. With a flat prior it is the likelihood rescaled: the thin cyan curve runs inside the yellow one. Its peak is at 0.70, the share of heads, the estimate that makes the tosses most probable; its mean, $8/12 = 0.67$, is pulled a little towards one half.",
        model: Model::Coin,
        prior: 0,
        second: None,
        truth: 0.7,
        seed: SEED,
        stop: 10,
        rate: 3.0,
        faster: false,
        threshold: 0.5,
        confidence: false,
        order: None,
    },
    Trial {
        button: "a stubborn prior",
        caption: r"Very sure of a fair coin, as if 100 earlier tosses had come out 50 and 50, against a flat prior (orange). The same 7 heads in 10 take the flat prior's mean to 0.67, but this one's only from 0.50 to 0.52: in the posterior's mean the prior counts as 100 tosses and the data as 10, $(100 \times 0.5 + 7) / 110 = 0.52$.",
        model: Model::Coin,
        prior: 2,
        second: Some(0),
        truth: 0.7,
        seed: SEED,
        stop: 10,
        rate: 3.0,
        faster: false,
        threshold: 0.5,
        confidence: false,
        order: None,
    },
    Trial {
        button: "agreement in the end",
        caption: r"Two observers, one sure the coin favours tails (prior mean 0.2), the other sure it favours heads (0.8), each as if from 20 earlier tosses; the true rate is 0.7. Whatever the tosses, their posterior means differ by $12/(20 + n)$ after $n$ of them: 0.40 after 10, 0.10 after 100, 0.012 after 1000. Their posteriors close in on each other and on 0.7.",
        model: Model::Coin,
        prior: 3,
        second: Some(4),
        truth: 0.7,
        seed: SEED,
        stop: 1000,
        rate: 10.0,
        faster: true,
        threshold: 0.5,
        confidence: false,
        order: None,
    },
    Trial {
        button: "a closed mind",
        caption: r"This prior gives every rate above one half no chance at all, and the true rate is 0.7. The likelihood (cyan) says so ever more plainly, but the posterior can only pile up against 0.5, ever more tightly: after 1000 tosses its mean is 0.499, its 95 % credible interval 0.496 to 0.500, and the probability it gives to $p > 0.5$ is still exactly 0. No data can revive what a prior has ruled out.",
        model: Model::Coin,
        prior: 5,
        second: None,
        truth: 0.7,
        seed: SEED,
        stop: 1000,
        rate: 10.0,
        faster: true,
        threshold: 0.5,
        confidence: false,
        order: None,
    },
    Trial {
        button: "order does not matter",
        caption: r"The same 7 heads and 3 tails as in the first experiment, replayed with the three tails first. The posterior slides towards 0 and is then pulled back, and it ends exactly on the white curve, where they took it in their first order: $\mathrm{Beta}(8, 4)$. The tosses are taken to be independent with one fixed rate, so only the counts matter, not the order.",
        model: Model::Coin,
        prior: 0,
        second: None,
        truth: 0.7,
        seed: SEED,
        stop: 10,
        rate: 2.0,
        faster: false,
        threshold: 0.5,
        confidence: false,
        order: Some(Order::TailsFirst),
    },
    Trial {
        button: "credible and confidence",
        caption: r"A flat prior and 100 tosses, 70 of them heads. The 95 % credible interval (yellow) and Wilson's 95 % confidence interval (cyan) both run from 0.60 to 0.78: they differ by less than 0.001. They answer different questions: the first is where the posterior puts 95 % of its probability, given the prior; the second comes from a method made to cover the true rate in 95 % of samples (with 100 tosses of this coin, it does in 94 %). With a flat prior and this many tosses they happen to agree.",
        model: Model::Coin,
        prior: 0,
        second: None,
        truth: 0.7,
        seed: SEED,
        stop: 100,
        rate: 10.0,
        faster: true,
        threshold: 0.5,
        confidence: true,
        order: None,
    },
    Trial {
        button: "the mean of a bell",
        caption: r"Values from a bell of spread $\sigma = 1$ whose mean (green) is unknown, and a prior for it with mean 0 and spread 1: precision 1, one over its variance. Four values bring precision $4/\sigma^2 = 4$, and the posterior's precision is $1 + 4 = 5$. Its mean weighs the two means by their precisions: $(1 \times 0 + 4 \times 1.918)/5 = 1.53$, four fifths of the way from the prior's mean to the values' mean.",
        model: Model::Bell,
        prior: 1,
        second: None,
        truth: 1.5,
        seed: SEED,
        stop: 4,
        rate: 2.0,
        faster: false,
        threshold: 0.0,
        confidence: false,
        order: None,
    },
];

/// The scene opens as the first experiment sets it up.
const OPENING: usize = 0;

// ----- the scene ---------------------------------------------------------------------------

pub struct Bayes {
    model: Model,
    /// Each observer's prior (the second's is drawn only while `second` is on, and is
    /// one of the model's priors: `second_prior`).
    prior: [Belief; 2],
    second: bool,
    second_prior: usize,
    truth: f64,
    show_truth: bool,
    /// The probability above this value is shaded and given.
    threshold: f64,
    show_confidence: bool,
    /// The observations made, in the order they come, and how many of them have come.
    values: Vec<f64>,
    arrived: usize,
    seed: u64,
    rng: Rng,
    /// Replays so far (for the shuffles' seeds).
    replays: u64,
    running: bool,
    /// Observations per second, and whether that rate grows with the count.
    rate: f32,
    faster: bool,
    /// The run stops after this many.
    stop_at: Option<usize>,
    pace: Pace,
    phase: Phase,
    /// How far the newest observation has come in: 0 as it arrives, 1 once it is in.
    /// Numbers say the observations as they are; the picture takes this into account.
    share: f32,
    /// While the observations so far are replayed in another order: how many there are
    /// (the posterior they all make is drawn too, in white).
    reference: Option<usize>,
    look: Look,
    /// The canvas and the model last drawn (a new one is drawn at once, without gliding).
    canvas: Option<(Vec2, Model)>,
    /// Which handle of the prior is held (0 its mean, 1 and 2 its ends).
    held: Option<usize>,
    /// The axis's round values last drawn, and how many labels there were.
    ticks: [f64; 3],
    labels_last: usize,
    /// The top of the chart in this frame (see `Look::top`).
    top: f64,
    /// Beliefs' intervals and means worked out lately (see `Bayes::summary`).
    summaries: std::cell::RefCell<Vec<(Belief, [f64; 3])>>,
}

impl Bayes {
    /// (Nothing here needs the graphics device, so the tests can make one too.)
    fn create() -> Bayes {
        let mut s = Bayes {
            model: Model::Coin,
            prior: [COIN_PRIORS[0].1, COIN_PRIORS[4].1],
            second: false,
            second_prior: 4,
            truth: 0.7,
            show_truth: true,
            threshold: 0.5,
            show_confidence: false,
            values: Vec::new(),
            arrived: 0,
            seed: SEED,
            rng: Rng::new(SEED),
            replays: 0,
            running: false,
            rate: 3.0,
            faster: false,
            stop_at: None,
            pace: Pace::default(),
            phase: Phase::default(),
            share: 1.0,
            reference: None,
            look: Look::default(),
            canvas: None,
            held: None,
            ticks: [0.0; 3],
            labels_last: 0,
            top: 1.0,
            summaries: Default::default(),
        };
        s.set_up(OPENING);
        s
    }

    /// Set up one of the experiments in `TRIALS`: every setting it depends on.
    fn set_up(&mut self, k: usize) {
        let t = &TRIALS[k];
        let list = priors(t.model);
        self.model = t.model;
        self.prior[0] = list[t.prior.min(list.len() - 1)].1;
        self.second = t.second.is_some();
        self.second_prior = t.second.unwrap_or(4).min(list.len() - 1);
        self.prior[1] = list[self.second_prior].1;
        self.truth = t.truth;
        self.show_truth = true;
        self.threshold = t.threshold;
        self.show_confidence = t.confidence;
        self.rate = t.rate;
        self.faster = t.faster;
        self.seed = t.seed;
        self.start_over_with(t.seed);
        self.stop_at = Some(t.stop);
        self.running = true;
        if let Some(order) = t.order {
            self.make(t.stop);
            self.arrived = t.stop.min(self.values.len());
            self.replay(order);
        }
    }

    /// Forget the observations and start again from this seed.
    fn start_over_with(&mut self, seed: u64) {
        self.seed = seed;
        self.rng = Rng::new(seed);
        self.values.clear();
        self.arrived = 0;
        self.replays = 0;
        self.reference = None;
        self.stop_at = None;
        self.share = 1.0;
        self.pace.reset();
        self.phase.reset();
    }

    /// Make observations, without their coming in yet, until there are `n`.
    fn make(&mut self, n: usize) {
        while self.values.len() < n.min(MOST) {
            let v = self.model.draw(self.truth, &mut self.rng);
            self.values.push(v);
        }
    }

    /// The next observation comes in (made now if it is not made yet).
    fn take(&mut self) {
        if self.arrived >= MOST {
            return;
        }
        self.make(self.arrived + 1);
        self.arrived += 1;
        if self.reference.is_some_and(|r| self.arrived > r) {
            self.reference = None;
        }
    }

    /// Bring the observations so far in again, in another order (all of those being
    /// replayed, if a replay is on its way); the posterior they all make stays drawn,
    /// in white.
    fn replay(&mut self, order: Order) {
        let n = self.reference.unwrap_or(self.arrived).min(self.values.len());
        self.values.truncate(n);
        if n == 0 {
            return;
        }
        match order {
            Order::TailsFirst => self.values.sort_by(|a, b| a.total_cmp(b)),
            Order::Shuffled => {
                self.replays += 1;
                let mut rng = Rng::new(self.seed.wrapping_mul(7919).wrapping_add(self.replays));
                rng.shuffle(&mut self.values);
            }
            Order::AtOnce => {}
        }
        self.reference = Some(n);
        self.stop_at = Some(n);
        self.pace.reset();
        self.phase.reset();
        if order == Order::AtOnce {
            self.arrived = n;
            self.share = 1.0;
            self.running = false;
        } else {
            self.arrived = 0;
            self.share = 1.0;
            self.running = true;
        }
    }

    /// The observations that have come in.
    fn counts(&self) -> Counts {
        Counts::of(&self.values[..self.arrived.min(self.values.len())])
    }

    /// The same, with the newest only `share` of the way in: what the picture shows.
    fn counts_shown(&self) -> Counts {
        let c = self.counts();
        match self.arrived.checked_sub(1).and_then(|i| self.values.get(i)) {
            Some(&last) => {
                let out = (1.0 - self.share.max(0.0).min(1.0)) as f64;
                Counts { n: c.n - out, sum: c.sum - out * last }
            }
            None => c,
        }
    }

    fn posterior(&self, who: usize) -> Belief {
        self.prior[who].after(self.counts())
    }

    /// The posterior of all the observations being replayed.
    fn reference_belief(&self) -> Option<Belief> {
        let n = self.reference?;
        Some(self.prior[0].after(Counts::of(&self.values[..n.min(self.values.len())])))
    }

    /// Observations per second now.
    fn rate_now(&self) -> f32 {
        let rate = self.rate.max(0.1);
        let rate = if self.faster { rate * (self.arrived as f32 / 10.0).max(1.0) } else { rate };
        rate.min(FASTEST)
    }

    /// The observations owed for `dt` seconds of the scene's time.
    fn work(&mut self, dt: f32, real_dt: f32) {
        if !self.running {
            // (the newest observation settles in by the clock: it answers a click)
            self.share += (1.0 - self.share) * (1.0 - (-real_dt.max(0.0) / 0.09).exp());
            if 1.0 - self.share < 1e-3 {
                self.share = 1.0;
            }
            return;
        }
        if dt <= 0.0 {
            return;
        }
        let rate = self.rate_now();
        let stop = self.stop_at.unwrap_or(MOST).min(MOST);
        let mut turn = scene_turn(&mut self.pace, rate, dt, real_dt);
        let mut taken = 0;
        while self.arrived < stop && turn.next() {
            self.take();
            taken += 1;
        }
        self.pace.end(turn);
        self.share = self.phase.after(rate, dt, taken);
        if self.arrived >= stop {
            self.running = false;
            self.pace.reset();
            self.phase.reset();
        }
    }

    /// The run to its end at once (the tests of this file).
    #[cfg(test)]
    fn settle(&mut self) {
        let stop = self.stop_at.unwrap_or(MOST).min(MOST);
        while self.running && self.arrived < stop {
            self.take();
        }
        self.running = false;
        self.share = 1.0;
    }

    /// Switch between the coin and the bell, starting afresh.
    fn switch(&mut self, model: Model) {
        if model == self.model {
            return;
        }
        self.model = model;
        let list = priors(model);
        self.prior[0] = list[if model == Model::Coin { 0 } else { 1 }].1;
        self.second_prior = 4.min(list.len() - 1);
        self.prior[1] = list[self.second_prior].1;
        (self.truth, self.threshold) = match model {
            Model::Coin => (0.7, 0.5),
            Model::Bell => (1.5, 0.0),
        };
        self.running = false;
        let seed = self.seed;
        self.start_over_with(seed);
    }

    /// A belief's interval's ends and mean, kept for the beliefs of the last few frames
    /// (each takes a search for two quantiles, and most do not change from one frame to
    /// the next).
    fn summary(&self, b: &Belief) -> [f64; 3] {
        let mut kept = self.summaries.borrow_mut();
        if let Some((_, v)) = kept.iter().find(|(k, _)| k == b) {
            return *v;
        }
        let (lo, hi) = b.interval();
        let v = [lo, hi, b.mean()];
        if kept.len() >= 12 {
            kept.remove(0);
        }
        kept.push((*b, v));
        v
    }

    /// What the picture glides to.
    fn goals(&self) -> Goals {
        let shown = self.counts_shown();
        let post = [self.prior[0].after(shown), self.prior[1].after(shown)];
        let reference = self.reference_belief().unwrap_or(post[0]);
        let beliefs = [self.prior[0], post[0], self.prior[1], post[1], reference];
        let like = match self.model {
            Model::Coin => [shown.heads() as f32, shown.tails() as f32],
            Model::Bell => [shown.n as f32, shown.estimate().unwrap_or(0.0) as f32],
        };
        let mut marks = Vec::with_capacity(15);
        for b in &beliefs[..4] {
            marks.extend(self.summary(b).map(|v| v as f32));
        }
        // (the data's own marks go by the observations that have come in: an interval
        // from a fraction of one would be as wide as anything)
        let whole = self.counts();
        let est = whole.estimate();
        let fallback = self.prior[0].mean();
        let (clo, chi) = confidence(self.model, whole).unwrap_or((fallback, fallback));
        marks.extend([est.unwrap_or(fallback), clo, chi].map(|v| v as f32));
        let on = |b: bool| if b { 1.0 } else { 0.0 };
        let alpha = [on(self.second), on(self.reference.is_some()), on(self.show_truth), on(self.show_confidence && est.is_some()), on(est.is_some())];
        Goals { beliefs: beliefs.map(|b| b.params()), like, marks, alpha }
    }
}

// ----- what the picture shows ------------------------------------------------------------

/// What the picture glides to: each belief's parameters (the first observer's prior
/// and posterior, the second's, and the posterior of the observations replayed), the
/// likelihood's counts, the marks under the chart (each belief's
/// interval and mean, the data's estimate and confidence interval) and how much of
/// what may be hidden is shown (the second observer, the replay's posterior, the
/// truth, the confidence interval, the data's estimate).
struct Goals {
    beliefs: [[f32; 3]; 5],
    like: [f32; 2],
    marks: Vec<f32>,
    alpha: [f32; 5],
}

/// The picture as drawn, gliding to the goals.
#[derive(Default)]
struct Look {
    beliefs: [Glide; 5],
    like: Glide,
    marks: Glide,
    alpha: Glide,
    moving: bool,
}

impl Look {
    fn follow(&mut self, g: &Goals, dt: f32, snap: bool) {
        let dt = if snap { 1e6 } else { dt };
        let mut moving = false;
        for (glide, goal) in self.beliefs.iter_mut().zip(&g.beliefs) {
            moving |= glide.follow(goal, dt, false);
        }
        moving |= self.like.follow(&g.like, dt, false);
        moving |= self.marks.follow(&g.marks, dt, false);
        moving |= self.alpha.follow(&g.alpha, dt, false);
        if snap {
            for glide in self.beliefs.iter_mut().chain([&mut self.like, &mut self.marks, &mut self.alpha]) {
                let to = glide.shown.clone();
                glide.snap(&to);
            }
            moving = false;
        }
        self.moving = moving;
    }

    fn belief(&self, model: Model, k: usize) -> Belief {
        Belief::from_params(model, &self.beliefs[k].shown)
    }

    /// The top of the chart: a little above the tallest belief drawn (those fading in
    /// or out count as much as they are shown), so that no curve is ever cut off.
    fn top(&self, model: Model) -> f64 {
        let peak = |k: usize| self.belief(model, k).peak().1;
        let (second, replayed) = (self.alpha(0) as f64, self.alpha(1) as f64);
        let tallest = peak(0).max(peak(1)).max(second * peak(2).max(peak(3))).max(replayed * peak(4));
        if tallest.is_finite() && tallest > 0.0 {
            tallest * 1.15
        } else {
            1.0
        }
    }

    fn mark(&self, i: usize) -> f64 {
        self.marks.get(i) as f64
    }

    fn alpha(&self, i: usize) -> f32 {
        self.alpha.get(i).max(0.0).min(1.0)
    }

    /// The likelihood as drawn.
    fn counts(&self, model: Model) -> Counts {
        let (x, y) = (self.like.get(0) as f64, self.like.get(1) as f64);
        match model {
            Model::Coin => Counts { n: x + y, sum: x },
            Model::Bell => Counts { n: x, sum: x * y },
        }
    }
}

/// Kinds of line of this scene's own, whose numbers of points change (see `plot::apart`).
const RINGS: u8 = OWN;
const TICKS: u8 = OWN + 1;
const STRIPES: u8 = OWN + 2;

/// Put an invisible point in every batch the scene draws in, in the same order in
/// every frame. (A batch left empty in a frame is let go, and made again at the end of
/// the list when next drawn in: every batch after it would then change places, and
/// `eidos motion` compare each with another.)
fn anchor_batches(f: &mut Frame) {
    let corner = f.hud(Vec2::ZERO);
    for kind in [AXES, CURVES, MARKS, plot::BUTTONS, RINGS, TICKS, STRIPES] {
        plot::apart(f, kind, false).segment(corner, corner, 1.0, Color::CLEAR);
    }
    plot::apart(f, MARKS, true).segment(corner, corner, 1.0, Color::CLEAR);
    for style in [PointStyle { outline_px: 0.0, ..Default::default() }, PointStyle { outline_px: 0.0, size_scale: 1.0001, ..Default::default() }, PointStyle { outline_px: 0.0, size_scale: 1.0002, ..Default::default() }] {
        f.dl.points_with(PointKind::Overlay, style).push(corner, 1.0, Color::CLEAR);
    }
}

/// Slots of the strip of tosses beyond those seen: more than come in a frame at the
/// highest rate, so that a slot is out of sight for a while before it is used again.
const RING_SPARE: usize = 96;
/// The most observations a second, however fast the rate is set to grow.
const FASTEST: f32 = 1200.0;

/// Rows of marks under the chart: the first observer's prior, the data, the first
/// observer's posterior, the second's prior and posterior.
const ROWS: usize = 5;

/// Where everything goes on a canvas (logical pixels).
#[derive(Clone, Copy, Debug, PartialEq)]
struct Layout {
    view: Vec2,
    narrow: bool,
    /// Text sizes: words, small print.
    text: f32,
    small: f32,
    /// The line saying how many observations there are, and where its tokens go
    /// (the coin's tosses, newest at the right).
    counts: Vec2,
    tokens: Rect,
    buttons: [Rect; 3],
    /// The chart, and the rows of marks under it (the same across).
    chart: Rect,
    strip: Rect,
    /// The key, the numbers and the formula (beside the chart on a wide canvas, under it
    /// otherwise; on a phone, the numbers alone).
    key: Rect,
    numbers: Rect,
    formula: Rect,
    /// Too small a canvas for the chart.
    compact: bool,
}

impl Layout {
    fn of(view: Vec2, inset: f32, model: Model) -> Layout {
        let view = view.max(Vec2::splat(1.0));
        let narrow = view.x < 640.0;
        let (text, small) = if narrow { (11.5, 10.5) } else { (13.0, 12.0) };
        let side = if narrow { 8.0 } else { 20.0 };
        // room at the left of the chart for the rows' names and the y axis's numbers
        let left = if narrow { 66.0 } else { 92.0 };
        let top = inset.max(0.0) + if narrow { 6.0 } else { 12.0 };
        let row = if narrow { 15.0 } else { 19.0 };
        let tokens_h = if model == Model::Coin { if narrow { 16.0 } else { 18.0 } } else { 0.0 };
        // (a column beside the chart only where the chart keeps room beside it)
        let wide = !narrow && view.x >= 1000.0 && view.x > 1.3 * view.y;
        // (a short canvas: the numbers under the chart, and no key, which the panel has)
        let short = !narrow && !wide && view.y < 640.0;
        let column = if wide { (view.x * 0.27).max(300.0).min(380.0) } else { 0.0 };
        let right = view.x - side - if wide { column + 28.0 } else { 0.0 };
        let button_h = 24.0;
        // the head: the buttons, the counts and the tokens
        let (buttons, counts, tokens, head_end) = if narrow {
            let b = even_row(3, side, view.x - side, top, button_h, 6.0);
            let counts = Vec2::new(side, top + button_h + 6.0);
            let t = (Vec2::new(side, counts.y + text + 5.0), Vec2::new(view.x - side, counts.y + text + 5.0 + tokens_h));
            ([b[0], b[1], b[2]], counts, t, t.1.y + if tokens_h > 0.0 { 6.0 } else { 0.0 })
        } else {
            let width = 96.0;
            let x0 = if wide { view.x - side - column } else { right - 3.0 * width - 12.0 };
            let b = even_row(3, x0, x0 + 3.0 * width + 12.0, top - 3.0, button_h, 6.0);
            let counts = Vec2::new(side, top);
            let t = (Vec2::new(side + left, top + text + 9.0), Vec2::new(right, top + text + 9.0 + tokens_h));
            ([b[0], b[1], b[2]], counts, t, t.1.y + 6.0)
        };
        // the chart's title and the name of its y axis above it
        let chart_top = head_end + if narrow { 38.0 } else { 46.0 };
        let strip_h = ROWS as f32 * row;
        let under = if narrow { 6.0 + strip_h + 2.0 * small + 14.0 } else { 8.0 + strip_h + 2.0 * small + 20.0 };
        // what goes under the strip: the formula, then the key and the numbers side by side
        let block_h = if short { 4.0 * (small + 6.0) } else { 7.0 * (small + 6.0) };
        let formula_h = if narrow { 0.0 } else { text + 16.0 };
        let below = if narrow {
            7.0 * (small + 5.0) + 6.0
        } else if wide {
            10.0
        } else {
            formula_h + block_h + 10.0
        };
        let chart_bottom = view.y - under - below;
        let chart = (Vec2::new(side + left, chart_top), Vec2::new(right, chart_bottom.max(chart_top + 1.0)));
        let strip_top = chart.1.y + if narrow { 6.0 } else { 8.0 };
        let strip = (Vec2::new(chart.0.x, strip_top), Vec2::new(chart.1.x, strip_top + strip_h));
        let after = strip.1.y + 2.0 * small + if narrow { 14.0 } else { 20.0 };
        let (key, numbers, formula) = if narrow {
            let n = (Vec2::new(side, after), Vec2::new(view.x - side, view.y - 4.0));
            (n, n, (Vec2::ZERO, Vec2::ZERO))
        } else if wide {
            let x0 = view.x - side - column;
            let key = (Vec2::new(x0, top + button_h + 18.0), Vec2::new(view.x - side, top + button_h + 18.0 + block_h));
            let formula = (Vec2::new(x0, key.1.y + 10.0), Vec2::new(view.x - side, key.1.y + 10.0 + 2.0 * formula_h + 20.0));
            let numbers = (Vec2::new(x0, formula.1.y + 10.0), Vec2::new(view.x - side, view.y - 10.0));
            (key, numbers, formula)
        } else {
            let formula = (Vec2::new(side, after), Vec2::new(right, after + formula_h));
            let mid = side + (right - side) * 0.5;
            if short {
                let numbers = (Vec2::new(side, formula.1.y + 4.0), Vec2::new(right, view.y - 4.0));
                ((numbers.0, numbers.0), numbers, formula)
            } else {
                let key = (Vec2::new(side, formula.1.y + 4.0), Vec2::new(mid - 16.0, view.y - 8.0));
                let numbers = (Vec2::new(mid + 16.0, formula.1.y + 4.0), Vec2::new(right, view.y - 8.0));
                (key, numbers, formula)
            }
        };
        let compact = chart.1.y - chart.0.y < 70.0 || chart.1.x - chart.0.x < 120.0;
        // (too small for the chart: the numbers alone, at the top)
        let numbers = if compact { (Vec2::new(side, top), Vec2::new(view.x - side, view.y - 4.0)) } else { numbers };
        Layout { view, narrow, text, small, counts, tokens, buttons, chart, strip, key, numbers, formula, compact }
    }

    fn row_y(&self, k: usize) -> f32 {
        let h = (self.strip.1.y - self.strip.0.y) / ROWS as f32;
        self.strip.0.y + (k as f32 + 0.5) * h
    }
}

/// Points of a chart in its own units on the screen, held to its rectangle (not cut
/// at its edge: a curve keeps its number of points in every frame).
fn held_in(chart: &Plot, pts: Vec<(f64, f64)>) -> Vec<Vec2> {
    let (min, max) = chart.rect;
    pts.into_iter()
        .map(|(x, y)| {
            let p = chart.at(x, y);
            if p.is_finite() {
                p.max(min).min(max)
            } else {
                Vec2::new(chart.px_x(x), max.y).max(min).min(max)
            }
        })
        .collect()
}

/// `kit::wrapped` for a scene drawn in pixels: the text's top-left corner and its width
/// in logical pixels.
fn wrapped_px(f: &mut Frame, at: Vec2, width: f32, text: &str, size: f32, color: Color) -> usize {
    let Some(world) = f.cam.pick_ground(at * f.px) else { return 0 };
    let ppu = pixels_per_unit(f).max(1e-6);
    wrapped(f, world.truncate(), width / ppu, text, size, color)
}

/// A number as a count: whole if it is one, else to two places.
fn count_num(v: f64) -> String {
    if (v - v.round()).abs() < 1e-9 {
        num(v.round(), 0)
    } else {
        num(v, 2)
    }
}

/// The probability a belief gives to the unknown being above t, to two places: exactly
/// 0 or 1 only where the belief makes it so (above its wall, or below the coin's 0),
/// never rounded to them.
fn prob_above(b: &Belief, t: f64) -> String {
    if let Belief::Beta { cut, .. } = *b {
        if t >= cut {
            return "0".to_string();
        }
        if t <= 0.0 {
            return "1".to_string();
        }
    }
    let p = b.above(t);
    if p < 0.005 {
        "< 0.01".to_string()
    } else if p > 0.995 {
        "> 0.99".to_string()
    } else {
        num(p, 2)
    }
}

/// The belief's name: Beta(a, b), with its wall if it has one; or a bell's mean and spread.
fn belief_name(b: &Belief) -> String {
    match *b {
        Belief::Beta { a, b, cut } if cut < 1.0 => format!(r"$\mathrm{{Beta}}({}, {})$ below ${}$", count_num(a), count_num(b), num(cut, 2)),
        Belief::Beta { a, b, .. } => format!(r"$\mathrm{{Beta}}({}, {})$", count_num(a), count_num(b)),
        Belief::Normal { mean, sd } => format!("mean ${}$, spread ${}$", num(mean, 2), num(sd, 2)),
    }
}

/// A number to two places, or whole if it is whole.
fn short(v: f64) -> String {
    if (v - v.round()).abs() < 1e-9 {
        num(v.round(), 0)
    } else {
        num(v, 2)
    }
}

/// The coin's line of counts.
fn line_words(c: Counts, full: &str) -> String {
    if c.n == 0.0 {
        "no tosses yet".to_string()
    } else {
        format!("{} tosses: {} heads, {} tails{full}", count_num(c.n), count_num(c.heads()), count_num(c.tails()))
    }
}

/// p^(a−1) (1−p)^(b−1) as markup.
fn kernel(a: f64, b: f64) -> String {
    format!("p^{{{}}}(1-p)^{{{}}}", count_num(a - 1.0), count_num(b - 1.0))
}

impl Scene for Bayes {
    fn view(&self) -> View {
        View::Flat { center: Vec2::ZERO, half_height: 1.0 }
    }

    fn settings(&self, s: &mut ViewSettings) {
        s.grid = false;
        s.bloom = s.bloom.min(0.2);
    }

    fn animated(&self) -> bool {
        self.running || self.look.moving || self.share < 1.0 || self.held.is_some()
    }

    fn pointer_hint(&self) -> Option<&'static str> {
        Some("purple dots under the chart: drag to shape the prior    blue dot: drag the threshold    buttons over the picture: click")
    }

    fn ui(&mut self, ui: &mut egui::Ui, _: &Gfx) {
        ui::note(ui, r"A coin's chance of heads, $p$, is not known. Bayesian statistics treats it as uncertain and describes what is believed about it by a distribution over 0 to 1. The prior (purple) is the belief before the tosses; the likelihood (cyan) says how probable the tosses seen are at each $p$; their product, rescaled to have area 1, is the posterior (yellow), the belief after:");
        ui::equation(ui, r"\text{posterior}(p) \propto \text{prior}(p) \times \text{likelihood}(p)");
        ui::note(ui, r"A Beta prior and coin tosses give a Beta posterior: $\mathrm{Beta}(a, b)$ and $h$ heads and $t$ tails make $\mathrm{Beta}(a + h, b + t)$. In the posterior's mean the prior counts as $a + b$ tosses made earlier.");
        ui::note(ui, r"The posterior answers the question people want to ask, such as how probable it is that $p$ is above one half (the blue line). The price is having to state a prior.");

        ui::section(ui, "Things to try");
        for (k, trial) in TRIALS.iter().enumerate() {
            if ui::experiment(ui, trial.button, trial.caption) {
                self.set_up(k);
            }
        }

        ui::section(ui, "The data");
        let mut which = (self.model == Model::Bell) as usize;
        if ui::choice(ui, &["a coin", r"a bell's mean $\mu$"], &mut which) {
            self.switch(if which == 1 { Model::Bell } else { Model::Coin });
        }
        if self.model == Model::Bell {
            ui::note(ui, r"Values from a bell (a normal distribution) whose spread $\sigma = 1$ is known and whose mean $\mu$ is not. A bell-shaped prior for $\mu$ gives a bell-shaped posterior.");
        }
        let (name, range) = match self.model {
            Model::Coin => ("true rate", 0.0..=1.0),
            Model::Bell => (r"true mean $\mu$", -3.0..=3.0),
        };
        let mut truth = self.truth as f32;
        if ui::slider(ui, name, &mut truth, range) {
            self.truth = truth as f64;
        }
        ui::checkbox(ui, &mut self.show_truth, "show the truth (green)");
        ui::note(ui, "The truth only decides what comes next; the observer never sees it.");
        let mut rate = self.rate;
        let per = if self.model == Model::Coin { "tosses per second" } else { "values per second" };
        if ui::slider_log(ui, per, &mut rate, 1.0..=200.0) {
            self.rate = rate;
        }
        ui::checkbox(ui, &mut self.faster, "faster as they come");
        ui::note(ui, "Faster as they come: from the tenth on, the rate grows in proportion to the count (up to 1200 a second), so that 10 to 100 take as long as 100 to 1000.");
        if let Some(k) = ui::buttons(ui, &["the same in another order", "all at once"]) {
            self.replay(if k == 0 { Order::Shuffled } else { Order::AtOnce });
        }
        ui::note(ui, "Replays the observations so far; the white curve is where they all lead.");

        ui::section(ui, "The prior");
        let names: Vec<&str> = priors(self.model).iter().map(|p| p.0).collect();
        if let Some(k) = ui::buttons(ui, &names) {
            self.prior[0] = priors(self.model)[k].1;
        }
        ui::note(ui, "Or drag the purple dots under the chart: the middle one moves the prior's mean, the ends make it wider or narrower.");
        ui::checkbox(ui, &mut self.second, "a second observer (orange), with the prior:");
        if self.second {
            let mut k = self.second_prior.min(names.len() - 1);
            if ui::choice(ui, &names, &mut k) {
                self.second_prior = k;
                self.prior[1] = priors(self.model)[k].1;
            }
        }
        let [lo, hi] = self.model.range();
        let mut t = self.threshold as f32;
        if ui::slider(ui, "threshold", &mut t, lo as f32..=hi as f32) {
            self.threshold = t as f64;
        }
        ui::note(ui, "The blue line: the posterior's probability above it is striped and written beside it. Or drag its dot.");
        ui::checkbox(ui, &mut self.show_confidence, "the 95 % confidence interval too");
        ui::note(ui, r"From the data alone, on their row under the chart: Wilson's for the coin, the mean $\pm 1.96\,\sigma/\sqrt{n}$ for the bell.");

        ui::section(ui, "Reading the picture");
        ui::key(
            ui,
            &[
                (ASSUMED, "purple: the prior, belief before the data; under the chart its middle 95 % and its mean (drag the dots)"),
                (DATA, "cyan: the likelihood of the data so far, scaled to the posterior's height (only its shape matters); under the chart the data's own estimate"),
                (ESTIMATE, "yellow: the posterior, belief after the data; shaded and under the chart, its middle 95 %, the credible interval"),
                (pal::BLUE, "blue: the threshold (drag its dot); striped, the posterior's probability above it"),
                (TRUTH, "green: the true value, which decides what the data will be"),
                (OTHER, "orange: a second observer's prior (thin) and posterior"),
                (pal::WHITE, "white: while observations are replayed, the posterior of them all"),
            ],
        );
        ui::section(ui, "Words");
        ui::term(ui, "Prior:", "what is believed about the unknown before the data, as a probability distribution over its possible values.");
        ui::term(ui, "Likelihood:", "for each possible value, the probability of the data actually seen if that were the true value (for measured values like the bell's, their density). It is not a distribution over the values: its area need not be 1.");
        ui::term(ui, "Posterior:", "the belief after the data: prior times likelihood, rescaled to have area 1. This is Bayes' rule.");
        ui::term(ui, "Density:", "the height of a curve whose area between two values is the probability of lying between them. A narrow curve must be tall.");
        ui::term(ui, "Beta distribution:", r"$\mathrm{Beta}(a, b)$ has density in proportion to $p^{a-1}(1-p)^{b-1}$ on 0 to 1, and mean $a/(a + b)$.");
        ui::term(ui, "Conjugate prior:", "a family of priors that data of a given kind keep within it: a Beta prior and coin tosses give a Beta posterior.");
        ui::term(ui, "Credible interval:", "an interval to which the posterior gives probability 95 %; here the middle one, with 2.5 % on each side.");
        ui::term(ui, "Posterior mean:", r"the posterior's average: for the coin $(a + h)/(a + b + n)$.");
        ui::term(ui, "Precision:", "one over a variance: how sharply a belief, or a set of measurements, pins a value down.");
        ui::term(ui, "Cromwell's rule:", "give nothing that is possible a prior probability of exactly 0, for no data can raise it again.");

        ui::section(ui, "Numbers");
        let c = self.counts();
        match self.model {
            Model::Coin => ui::readout(ui, "tosses", format!("{} heads, {} tails", count_num(c.heads()), count_num(c.tails()))),
            Model::Bell => ui::readout(ui, "values", count_num(c.n)),
        }
        if let Some(e) = c.estimate() {
            let name = if self.model == Model::Coin { "share of heads" } else { r"their mean $\bar x$" };
            ui::readout(ui, name, num(e, 3));
        }
        for who in 0..1 + self.second as usize {
            let (prior, post) = (self.prior[who], self.posterior(who));
            let tag = if who == 0 { "" } else { "2nd " };
            ui::readout(ui, &format!("{tag}prior"), belief_name(&prior));
            ui::readout(ui, &format!("{tag}prior's mean"), num(prior.mean(), 3));
            match prior {
                Belief::Beta { .. } if !prior.is_cut() => ui::readout(ui, &format!("{tag}prior counts as"), format!("{} tosses", count_num(prior.strength()))),
                Belief::Normal { .. } => ui::readout(ui, &format!("{tag}prior's precision"), short(prior.strength())),
                _ => {}
            }
            ui::readout(ui, &format!("{tag}posterior"), belief_name(&post));
            ui::readout(ui, &format!("{tag}posterior's mean"), num(post.mean(), 3));
            let (lo, hi) = post.interval();
            ui::readout(ui, &format!("{tag}95 % credible interval"), format!("{} to {}", num(lo, 3), num(hi, 3)));
            ui::readout(ui, &format!("{tag}probability above {}", num(self.threshold, 2)), format!("${}$", prob_above(&post, self.threshold)));
        }
        if self.second {
            ui::readout(ui, "gap between the means", num((self.posterior(0).mean() - self.posterior(1).mean()).abs(), 3));
        }
        if let Some((lo, hi)) = confidence(self.model, c) {
            let name = if self.model == Model::Coin { "95 % confidence (Wilson)" } else { "95 % confidence interval" };
            ui::readout(ui, name, format!("{} to {}", num(lo, 3), num(hi, 3)));
        }
        ui::see_also(
            ui,
            &[
                ("chance", "Bayes' rule for two events"),
                ("likelihood", "the likelihood on its own: maximum likelihood"),
                ("intervals", "confidence intervals, and what 95 % means there"),
                ("mcmc", "sampling a posterior that has no formula"),
            ],
        );
    }

    fn frame(&mut self, f: &mut Frame) {
        let lay = Layout::of(f.view_size(), f.top_inset, self.model);
        // (a canvas squeezed to a sliver is one the controls have opened over, on a phone)
        let tiny = lay.view.x < 160.0 || lay.view.y < 160.0;
        if !tiny && !lay.compact {
            self.buttons(f, &lay);
        }
        self.work(f.dt, f.real_dt);
        if !tiny && !lay.compact {
            self.handles(f, &lay);
        } else {
            self.held = None;
        }
        let goals = self.goals();
        let snap = self.canvas != Some((lay.view, self.model)) || self.look.beliefs[0].shown.is_empty();
        if snap && !tiny {
            self.canvas = Some((lay.view, self.model));
        }
        self.look.follow(&goals, f.real_dt, snap);
        self.top = self.look.top(self.model);
        // (the prior being shaped is drawn where the hand puts it)
        if self.held.is_some() {
            self.look.beliefs[0].snap(&goals.beliefs[0]);
            let mut marks = self.look.marks.shown.clone();
            if marks.len() == goals.marks.len() {
                marks[..3].copy_from_slice(&goals.marks[..3]);
                self.look.marks.snap(&marks);
            }
        }
        if tiny {
            return;
        }
        if lay.compact {
            self.draw_numbers(f, &lay);
            return;
        }
        anchor_batches(f);
        let labels_before = f.dl.labels.len();
        self.draw_head(f, &lay);
        let ticks = self.draw_chart(f, &lay);
        self.draw_strip(f, &lay);
        self.draw_side(f, &lay);
        // new round values on the y axis: make sure the labels are not as many as in the
        // last frame, so that nothing takes the new ones for the old ones moved
        let drawn = f.dl.labels.len() - labels_before;
        if ticks != self.ticks && drawn == self.labels_last {
            f.text_px(Vec2::ZERO, "", 1.0, Color::CLEAR, Align2::LEFT_TOP);
        }
        self.ticks = ticks;
        self.labels_last = f.dl.labels.len() - labels_before;
    }
}

impl Bayes {
    /// The buttons over the picture.
    fn buttons(&mut self, f: &mut Frame, lay: &Layout) {
        let words = match (self.model, self.running) {
            (Model::Coin, false) => ["toss once", "run", "start over"],
            (Model::Coin, true) => ["toss once", "pause", "start over"],
            (Model::Bell, false) => ["draw one", "run", "start over"],
            (Model::Bell, true) => ["draw one", "pause", "start over"],
        };
        let size = lay.small;
        if button(f, lay.buttons[0], words[0], size, false) {
            self.running = false;
            self.stop_at = None;
            if self.arrived < MOST {
                self.take();
                self.share = 0.0;
            }
        }
        if button(f, lay.buttons[1], words[1], size, self.running) {
            if self.running {
                self.running = false;
            } else if self.arrived < MOST {
                if self.stop_at.is_some_and(|s| self.arrived >= s) {
                    self.stop_at = None;
                }
                self.running = true;
                self.pace.reset();
                self.phase.reset();
            }
        }
        if button(f, lay.buttons[2], words[2], size, false) {
            let seed = self.seed.wrapping_add(1);
            self.start_over_with(seed);
        }
    }

    /// The chart over the unknown and the strip under it share their x axis.
    fn plots(&self, lay: &Layout) -> (Plot, Plot) {
        let x = self.model.range();
        (Plot::new(lay.chart, x, [0.0, self.top.max(1e-6)]), Plot::new(lay.strip, x, [0.0, ROWS as f64]))
    }

    /// The prior's handles (its mean and its interval's ends, on its row under the
    /// chart) and the threshold's (at the top of its line).
    fn handles(&mut self, f: &mut Frame, lay: &Layout) {
        let (chart, strip) = self.plots(lay);
        let row = ROWS as f64 - 0.5;
        let goals_prior = self.prior[0];
        let (lo, hi) = goals_prior.interval();
        let mean = goals_prior.mean();
        let mut held = None;
        for (k, x) in [(1usize, lo), (2, hi), (0, mean)] {
            if let Some((to, _)) = strip.handle(f, 10 + k as u32, x, row, ASSUMED) {
                held = Some(k);
                self.shape_prior(k, to);
            }
        }
        if f.handles.dragging() && held.is_none() && self.held.is_some() {
            held = self.held;
        }
        if !f.handles.dragging() {
            held = None;
        }
        self.held = held;
        let top = chart.y[1];
        if let Some((to, _)) = chart.handle(f, 20, self.threshold, top, pal::BLUE) {
            let [a, b] = self.model.range();
            self.threshold = to.max(a).min(b);
        }
    }

    /// Move the prior's mean (`k` = 0) or one of its interval's ends (1 the low one, 2 the
    /// high one) to `x`. A prior that rules values out becomes an ordinary one.
    fn shape_prior(&mut self, k: usize, x: f64) {
        if !x.is_finite() {
            return;
        }
        match self.prior[0] {
            Belief::Beta { a, b, .. } => {
                let m0 = if self.prior[0].is_cut() { self.prior[0].mean() } else { a / (a + b) };
                let s0 = a + b;
                let (m, s) = match k {
                    0 => {
                        let m = x.max(0.02).min(0.98);
                        (m, s0.max(1.0 / m).max(1.0 / (1.0 - m)).min(MOST_SURE))
                    }
                    1 => (m0, strength_for_end(m0, x.max(0.0005).min(m0 - 1e-4), false)),
                    _ => (m0, strength_for_end(m0, x.max(m0 + 1e-4).min(0.9995), true)),
                };
                if m.is_finite() && s.is_finite() {
                    self.prior[0] = Belief::Beta { a: (m * s).max(1.0), b: ((1.0 - m) * s).max(1.0), cut: 1.0 };
                }
            }
            Belief::Normal { mean, sd } => {
                let z = normal_quantile(0.5 + LEVEL / 2.0);
                let (m, s) = match k {
                    0 => (x.max(-3.5).min(3.5), sd),
                    _ => (mean, ((x - mean).abs() / z).max(BELL_SD[0]).min(BELL_SD[1])),
                };
                self.prior[0] = Belief::Normal { mean: m, sd: s };
            }
        }
    }

    /// The line of counts and the strip of tosses, newest at the right.
    fn draw_head(&self, f: &mut Frame, lay: &Layout) {
        let c = self.counts();
        let full = if self.arrived >= MOST { " (the most kept here)" } else { "" };
        let line = match self.model {
            Model::Coin => line_words(c, full),
            Model::Bell => match c.estimate() {
                None => r"no values yet from the bell (spread $\sigma = 1$)".to_string(),
                Some(m) => format!(r"{} values from the bell (spread $\sigma = 1$): their mean $\bar x = {}${full}", count_num(c.n), num(m, 2)),
            },
        };
        f.text_px(lay.counts, line, lay.text, pal::TEXT, Align2::LEFT_TOP);
        if self.model != Model::Coin {
            return;
        }
        let (min, max) = lay.tokens;
        let h = max.y - min.y;
        let w = h + 3.0;
        let mid = (min.y + max.y) * 0.5;
        let r = h * 0.36;
        // the tosses fill the row from the left; once it is full it slides left as each
        // new one comes in (`v` counts them with the newest only `share` of the way in)
        let v = self.arrived as f32 - 1.0 + self.share.max(0.0).min(1.0);
        let slots = ((max.x - min.x) / w).floor().max(2.0);
        let slide = (v + 1.0 - slots).max(0.0);
        let first = slide.floor().max(0.0) as usize;
        let last = self.arrived.min(self.values.len());
        // Each toss is drawn in the slot its number gives it, in a ring of slots a few
        // more than are seen, and every slot is drawn in every frame (one not in use is
        // put out of sight): so that a toss is the same point of the picture from one
        // frame to the next, as it slides along (see `eidos motion`).
        let ring = slots as usize + RING_SPARE;
        let away = Vec2::splat(-10_000.0);
        let (mut discs, mut rings) = (vec![(away, Color::CLEAR); ring], vec![(away, 0.0f32); ring]);
        for k in first..last {
            let x = min.x + r + (k as f32 - slide) * w;
            // the newest fades in as it comes, the oldest out as it goes
            let coming = if k + 1 == self.arrived { self.share.max(0.0).min(1.0) } else { 1.0 };
            let alpha = coming * ((x - min.x + r) / w).max(0.0).min(1.0);
            let at = Vec2::new(x, mid);
            let slot = k % ring;
            discs[slot] = (at, if self.values[k] > 0.5 { DATA.a(alpha) } else { Color::CLEAR });
            rings[slot] = (at, alpha);
        }
        let style = PointStyle { outline_px: 0.0, size_scale: 1.0001, ..Default::default() };
        for (at, color) in discs {
            let p = f.hud_at(2, at);
            f.dl.points_with(PointKind::Overlay, style).push(p, 2.0 * r, color);
        }
        let circle: Vec<Vec2> = (0..=16).map(|i| Vec2::from_angle(i as f32 / 16.0 * TAU) * (r - 0.8)).collect();
        for (at, alpha) in rings {
            let pts: Vec<Vec3> = if alpha > 0.0 { circle.iter().map(|c| f.hud(at + *c)).collect() } else { vec![f.hud(away); circle.len()] };
            plot::apart(f, RINGS, false).polyline(pts, 1.6, DATA.a(0.9 * alpha));
        }
        // what the tokens are, at the right of the line of counts (at a place of its
        // own: the counts change width as they grow)
        let end = if lay.narrow { lay.view.x - 8.0 } else { lay.buttons[0].0.x - 16.0 };
        let (heads_w, tails_w) = (text_width("heads", lay.small), text_width("tails", lay.small));
        let tails_at = Vec2::new(end - tails_w - 5.0 - r, lay.counts.y + lay.text * 0.55);
        let at = tails_at - Vec2::new(r + 14.0 + heads_w + 5.0 + r, 0.0);
        let room = at.x - r > lay.counts.x + text_width(&line_words(Counts { n: 1000.0, sum: 1000.0 }, ""), lay.text) + 12.0;
        let p = f.hud_at(2, at);
        f.dl.points_with(PointKind::Overlay, PointStyle { outline_px: 0.0, size_scale: 1.0002, ..Default::default() }).push(p, 2.0 * r, DATA.a(if room { 1.0 } else { 0.0 }));
        let legend: Vec<Vec2> = (0..=16).map(|i| tails_at + Vec2::from_angle(i as f32 / 16.0 * TAU) * (r - 0.8)).collect();
        path_of(f, MARKS, &legend, 1.6, DATA.a(if room { 0.9 } else { 0.0 }));
        if room {
            f.text_px(at + Vec2::new(r + 5.0, 0.0), "heads", lay.small, pal::DIM, Align2::LEFT_CENTER);
            f.text_px(tails_at + Vec2::new(r + 5.0, 0.0), "tails", lay.small, pal::DIM, Align2::LEFT_CENTER);
        }
    }

    /// The chart: the beliefs and the likelihood over the unknown. Returns its y axis's
    /// round values (step, first, count), to tell when they change.
    fn draw_chart(&self, f: &mut Frame, lay: &Layout) -> [f64; 3] {
        let (chart, _) = self.plots(lay);
        let model = self.model;
        let look = &self.look;
        let (min, max) = lay.chart;
        hud_panel(f, min - Vec2::new(10.0, 8.0), Vec2::new(max.x + 10.0, lay.strip.1.y + 8.0));
        let unknown = if model == Model::Coin { "p" } else { r"\mu" };
        let title = if lay.narrow {
            format!("how believable each value of ${unknown}$ is")
        } else if model == Model::Coin {
            "how believable each rate of heads $p$ is, before and after the tosses".to_string()
        } else {
            r"how believable each mean $\mu$ is, before and after the values".to_string()
        };
        f.text_px(Vec2::new(min.x - 30.0, min.y - 26.0), title, lay.text, pal::TEXT, Align2::LEFT_BOTTOM);
        chart.y_axis(f, "probability density", lay.small);
        chart.y_grid(f);
        let ticks = Ticks::of(chart.y[0], chart.y[1], chart.height() * 1.6);
        // the axis across, under the strip, with its grid through both
        let (values, decimals) = round_values(chart.x[0], chart.x[1], chart.width());
        for v in values {
            let x = chart.px_x(v);
            line_of(f, AXES, Vec2::new(x, min.y), Vec2::new(x, lay.strip.1.y), 1.0, pal::DIM.a(0.12));
            f.text_px(Vec2::new(x, lay.strip.1.y + 5.0), num(v, decimals), lay.small, pal::DIM, Align2::CENTER_TOP);
        }
        line_of(f, AXES, Vec2::new(min.x, max.y), max, 1.0, pal::DIM.a(0.8));
        let name = match (model, lay.narrow) {
            (Model::Coin, false) => "$p$: the coin's chance of heads on each toss",
            (Model::Coin, true) => "$p$: the chance of heads",
            (Model::Bell, false) => r"$\mu$: the mean of the bell the values come from",
            (Model::Bell, true) => r"$\mu$: the bell's mean",
        };
        f.text_px(Vec2::new((min.x + max.x) * 0.5, lay.strip.1.y + 7.0 + lay.small), name, lay.small, pal::DIM, Align2::CENTER_TOP);

        let prior = look.belief(model, 0);
        let post = look.belief(model, 1);
        let (second, replayed, truth_a) = (look.alpha(0), look.alpha(1), look.alpha(2));
        let xs = samples(model);
        // the values a cut-off prior rules out
        if let Belief::Beta { cut, .. } = prior {
            if cut < 0.999 {
                fill(f, (Vec2::new(chart.px_x(cut), min.y), max), ASSUMED.a(0.07));
                let words = if lay.narrow { "ruled out" } else { "ruled out by the prior" };
                let room = max.x - chart.px_x(cut) > text_width(words, lay.small) + 12.0;
                let color = ASSUMED.mix(pal::TEXT, 0.3).a(if room { 1.0 } else { 0.0 });
                f.text_px(Vec2::new((chart.px_x(cut) + max.x) * 0.5, min.y + 8.0), words, lay.small, color, Align2::CENTER_TOP);
            }
        }
        // the posterior's middle 95 % and its probability above the threshold, shaded
        let g = post.density();
        chart.area(f, look.mark(3), look.mark(4), 0.0, 120, ESTIMATE.a(0.14), &g);
        // (the probability above the threshold in stripes, so that it reads through the
        // interval's shading rather than mixing with it)
        let mut stripes: Vec<Vec2> = Vec::new();
        let (from, to) = (chart.px_x(self.threshold.max(chart.x[0])), max.x);
        let step = 7.0;
        let mut x = (from / step).ceil() * step;
        // (a stripe beyond a wall is drawn at the wall, as the curve's points are, while
        // the threshold is below the wall; above it, there is nothing to shade)
        let wall = wall_of(&post);
        let shaded = self.threshold < wall;
        while x < to {
            let at = chart.x_at(x).min(wall);
            let v = if shaded { g(at) } else { 0.0 };
            let top = if v.is_finite() { chart.px_y(v).max(min.y).min(max.y) } else { max.y };
            let x_at = chart.px_x(at);
            stripes.extend([Vec2::new(x_at, max.y), Vec2::new(x_at, top), Vec2::NAN]);
            x += step;
        }
        path_of(f, STRIPES, &stripes, 1.0, pal::BLUE.a(0.6));
        // the curves: the second observer's under the first's
        let curve = |f: &mut Frame, b: Belief, width: f32, color: Color| {
            // (one not seen is not worked out: it lies along the axis, with as many points)
            let (line, wall, cut) = if color.0[3] > 0 { curve_points(&xs, &b, &b.density()) } else { curve_points(&xs, &b, &|_| 0.0) };
            path_of(f, CURVES, &held_in(&chart, line), width, color);
            path_of(f, CURVES, &held_in(&chart, wall), width, color.a(if cut { color.0[3] as f32 / 255.0 } else { 0.0 }));
        };
        curve(f, look.belief(model, 2), 1.4, OTHER.a(0.75 * second));
        curve(f, look.belief(model, 3), 2.2, OTHER.a(second));
        curve(f, prior, 2.2, ASSUMED);
        // (the likelihood is drawn as tall as the posterior: only its shape matters)
        let height = post.peak().1;
        let like = likelihood(model, look.counts(model));
        // the posterior of the observations being replayed, under the posterior: where
        // they will all lead
        curve(f, look.belief(model, 4), 1.6, pal::WHITE.a(0.8 * replayed));
        curve(f, post, 3.4, ESTIMATE);
        // the likelihood, thin, on top: where it has the posterior's shape (a flat prior)
        // it is a cyan line inside the yellow one
        let pts = held_in(&chart, xs.iter().map(|&x| (x, like(x) * height)).collect());
        path_of(f, CURVES, &pts, 1.3, DATA);
        // the truth, and the threshold
        let truth_x = self.truth.max(chart.x[0]).min(chart.x[1]);
        let tx = chart.px_x(truth_x);
        plot::dashed_of(f, MARKS, Vec2::new(tx, min.y), Vec2::new(tx, lay.strip.1.y), 1.6, TRUTH.a(0.9 * truth_a));
        let truth_words = format!("true ${unknown} = {}$", num(self.truth, 2));
        let right_room = max.x - tx > f.text_size(&truth_words, lay.small).x + 10.0;
        let (off, align) = if right_room { (Vec2::new(5.0, -4.0), Align2::LEFT_BOTTOM) } else { (Vec2::new(-5.0, -4.0), Align2::RIGHT_BOTTOM) };
        f.caption_px(Vec2::new(tx, max.y) + off, truth_words, lay.small, TRUTH.a(truth_a), align);
        let t = self.threshold.max(chart.x[0]).min(chart.x[1]);
        let x = chart.px_x(t);
        line_of(f, MARKS, Vec2::new(x, min.y), Vec2::new(x, max.y), 1.4, pal::BLUE.a(0.9));
        let above = prob_above(&self.posterior(0), self.threshold);
        let words = format!(r"$P({unknown} > {}) {} {above}$", num(self.threshold, 2), if above.starts_with(['<', '>']) { "" } else { "=" });
        let wide = f.text_size(&words, lay.small).x;
        let (off, align) = if max.x - x > wide + 14.0 { (Vec2::new(10.0, 4.0), Align2::LEFT_TOP) } else { (Vec2::new(-10.0, 4.0), Align2::RIGHT_TOP) };
        f.caption_px(Vec2::new(x, min.y) + off, words, lay.small, pal::BLUE.mix(pal::TEXT, 0.35), align);
        let first = ticks.values().next().unwrap_or(0.0);
        let count = ticks.values().count() as f64;
        [ticks.step, first, count]
    }

    /// The rows under the chart: each belief's middle 95 % and mean, and the data's
    /// estimate.
    fn draw_strip(&self, f: &mut Frame, lay: &Layout) {
        let (_, strip) = self.plots(lay);
        let look = &self.look;
        let (second, conf, data) = (look.alpha(0), look.alpha(3), look.alpha(4));
        let names: [(&str, Color, f32); ROWS] = [
            ("prior", ASSUMED, 1.0),
            ("data", DATA, 1.0),
            ("posterior", ESTIMATE, 1.0),
            ("2nd prior", OTHER, second),
            ("2nd posterior", OTHER, second),
        ];
        let (min, _) = lay.strip;
        for (k, (name, color, alpha)) in names.iter().enumerate() {
            f.text_px(Vec2::new(min.x - 10.0, lay.row_y(k)), *name, lay.small, color.mix(pal::TEXT, 0.15).a(*alpha), Align2::RIGHT_CENTER);
        }
        // (row k is drawn at y = ROWS − k − 0.5 in the strip's own numbers)
        let y = |k: usize| ROWS as f64 - k as f64 - 0.5;
        let bar = |f: &mut Frame, k: usize, lo: f64, hi: f64, mean: f64, width: f32, color: Color| {
            let (a, b) = (strip.at(lo.max(strip.x[0]), y(k)), strip.at(hi.min(strip.x[1]), y(k)));
            line_of(f, MARKS, a, b, width, color);
            let m = strip.at(mean.max(strip.x[0]).min(strip.x[1]), y(k));
            line_of(f, MARKS, m - Vec2::new(0.0, 5.0), m + Vec2::new(0.0, 5.0), 2.0, color);
        };
        // the prior's bar (its handles are drawn by `handles`)
        bar(f, 0, look.mark(0), look.mark(1), look.mark(2), 3.0, ASSUMED.a(0.8));
        bar(f, 2, look.mark(3), look.mark(4), look.mark(5), 4.0, ESTIMATE);
        bar(f, 3, look.mark(6), look.mark(7), look.mark(8), 2.0, OTHER.a(0.7 * second));
        bar(f, 4, look.mark(9), look.mark(10), look.mark(11), 4.0, OTHER.a(second));
        // the data: the confidence interval, the estimate, and for the bell each value
        bar(f, 1, look.mark(13), look.mark(14), look.mark(12), 4.0, DATA.a(0.75 * conf));
        strip.dot(f, look.mark(12), y(1), 9.0, DATA.a(data));
        if self.model == Model::Bell {
            let mut ticks: Vec<Vec2> = Vec::new();
            let half = (lay.strip.1.y - lay.strip.0.y) / ROWS as f32 * 0.32;
            let n = self.arrived.min(self.values.len());
            for (i, v) in self.values[..n].iter().enumerate() {
                if !(*v >= strip.x[0] && *v <= strip.x[1]) {
                    continue;
                }
                let at = strip.at(*v, y(1));
                let grow = if i + 1 == n { self.share.max(0.0).min(1.0) } else { 1.0 };
                ticks.extend([at - Vec2::new(0.0, half * grow), at + Vec2::new(0.0, half * grow), Vec2::NAN]);
            }
            path_of(f, TICKS, &ticks, 1.2, DATA.a(0.55));
        }
    }

    /// The key, the numbers and the formula: beside the chart or under it.
    fn draw_side(&self, f: &mut Frame, lay: &Layout) {
        if lay.narrow {
            self.draw_numbers(f, lay);
            return;
        }
        let s = lay.small;
        let unknown = if self.model == Model::Coin { "p" } else { r"\mu" };
        let data = if self.model == Model::Coin { "the share of heads" } else { "the values and their mean" };
        let mut entries: Vec<(Color, String)> = vec![
            (ASSUMED, "prior: belief before the data (drag its dots under the chart)".to_string()),
            (DATA, format!("likelihood (thin), scaled to the posterior; under the chart, {data}")),
            (ESTIMATE, "posterior: belief after; shaded, its middle 95 %".to_string()),
            (pal::BLUE, format!("threshold $t$: the striped tail is $P({unknown} > t)$")),
            (TRUTH, "the true value (dashed)".to_string()),
        ];
        if self.second {
            entries.push((OTHER, "a second observer: prior (thin) and posterior".to_string()));
        }
        if self.reference.is_some() {
            entries.push((pal::WHITE, "white: the posterior of all the replayed data".to_string()));
        }
        let (kmin, kmax) = lay.key;
        let row = s + 6.0;
        for (i, (color, words)) in entries.iter().enumerate() {
            let at = kmin + Vec2::new(0.0, i as f32 * row);
            if at.y + row > kmax.y + 2.0 {
                break;
            }
            fill(f, (at + Vec2::new(0.0, 2.0), at + Vec2::new(s - 2.0, s)), *color);
            f.text_px(at + Vec2::new(s + 6.0, s * 0.5 + 1.0), words.as_str(), s, pal::DIM, Align2::LEFT_CENTER);
        }
        self.draw_formula(f, lay);
        self.draw_numbers(f, lay);
    }

    /// prior × likelihood ∝ posterior, with the numbers of now.
    fn draw_formula(&self, f: &mut Frame, lay: &Layout) {
        let (min, max) = lay.formula;
        if max.x - min.x < 50.0 {
            return;
        }
        let c = self.counts();
        let (prior, post) = (self.prior[0], self.posterior(0));
        let line = match (prior, post) {
            (Belief::Beta { a, b, cut }, Belief::Beta { a: pa, b: pb, .. }) => {
                let wall = if cut < 1.0 { format!(r" \text{{ (0 above }} {})", num(cut, 1)) } else { String::new() };
                format!(
                    r"prior $\times$ likelihood $\propto$ posterior:   ${}{} \times {} \propto {}{}$,  so ${} = {}$",
                    tint(ASSUMED, &kernel(a, b)),
                    if cut < 1.0 { tint(ASSUMED, &wall) } else { String::new() },
                    tint(DATA, &format!("p^{{{}}}(1-p)^{{{}}}", count_num(c.heads()), count_num(c.tails()))),
                    tint(ESTIMATE, &kernel(pa, pb)),
                    if cut < 1.0 { tint(ESTIMATE, &wall) } else { String::new() },
                    tint(ESTIMATE, &format!(r"\mathrm{{Beta}}({} + {}, {} + {})", count_num(a), count_num(c.heads()), count_num(b), count_num(c.tails()))),
                    tint(ESTIMATE, &format!(r"\mathrm{{Beta}}({}, {})", count_num(pa), count_num(pb))),
                )
            }
            (Belief::Normal { mean, sd }, Belief::Normal { mean: pm, .. }) => {
                let (p0, pd) = (1.0 / (sd * sd), c.n / (SIGMA * SIGMA));
                let xbar = c.estimate().unwrap_or(0.0);
                format!(
                    r"precisions add: ${} + {} = {}$;   the posterior's mean weighs the means by them: $({} \times {} + {} \times {}) / {} = {}$",
                    tint(ASSUMED, &short(p0)),
                    tint(DATA, &short(pd)),
                    tint(ESTIMATE, &short(p0 + pd)),
                    tint(ASSUMED, &short(p0)),
                    tint(ASSUMED, &short(mean)),
                    tint(DATA, &short(pd)),
                    tint(DATA, &num(xbar, 3)),
                    short(p0 + pd),
                    tint(ESTIMATE, &num(pm, 3)),
                )
            }
            _ => String::new(),
        };
        wrapped_px(f, min, max.x - min.x, &line, lay.small + 0.5, pal::DIM);
    }

    /// The numbers on the picture.
    fn draw_numbers(&self, f: &mut Frame, lay: &Layout) {
        let (min, max) = lay.numbers;
        let s = lay.small;
        let c = self.counts();
        let unknown = if self.model == Model::Coin { "p" } else { r"\mu" };
        let mut lines: Vec<(String, Color)> = Vec::new();
        let post = self.posterior(0);
        let (lo, hi) = post.interval();
        let named = |b: &Belief| match b {
            Belief::Beta { .. } => format!("{}, mean {}", belief_name(b), num(b.mean(), 2)),
            Belief::Normal { .. } => belief_name(b),
        };
        lines.push((format!("prior: {}", named(&self.prior[0])), ASSUMED.mix(pal::TEXT, 0.3)));
        lines.push((format!("posterior: {}", named(&post)), ESTIMATE.mix(pal::TEXT, 0.2)));
        lines.push((format!("95 % credible interval: {} to {}", num(lo, 3), num(hi, 3)), ESTIMATE.mix(pal::TEXT, 0.2)));
        let above = prob_above(&post, self.threshold);
        let relation = if above.starts_with(['<', '>']) { "" } else { "=" };
        lines.push((format!(r"$P({unknown} > {}) {relation} {above}$", num(self.threshold, 2)), pal::BLUE.mix(pal::TEXT, 0.35)));
        if self.second {
            let other = self.posterior(1);
            lines.push((format!("2nd posterior: {}", named(&other)), OTHER.mix(pal::TEXT, 0.2)));
            lines.push((format!("gap between the means: {}", num((post.mean() - other.mean()).abs(), 3)), OTHER.mix(pal::TEXT, 0.2)));
        }
        if self.show_confidence {
            let words = match confidence(self.model, c) {
                Some((a, b)) if self.model == Model::Coin => format!("95 % confidence interval (Wilson's): {} to {}", num(a, 3), num(b, 3)),
                Some((a, b)) => format!("95 % confidence interval: {} to {}", num(a, 3), num(b, 3)),
                None => "95 % confidence interval: no data yet".to_string(),
            };
            lines.push((words, DATA.mix(pal::TEXT, 0.2)));
        }
        if lay.narrow && !lay.compact {
            // (a phone has no key beside the chart: the two things the rows under it do not name)
            lines.push(("thin cyan: the likelihood, scaled to the posterior".to_string(), pal::DIM));
            lines.push(("shaded: the posterior's middle 95 %; striped: above the threshold".to_string(), pal::DIM));
        }
        let row = s + if lay.narrow { 5.0 } else { 6.0 };
        for (i, (words, color)) in lines.iter().enumerate() {
            let at = min + Vec2::new(0.0, i as f32 * row);
            if at.y + row > max.y + 2.0 {
                break;
            }
            f.text_px(at, words.as_str(), s, *color, Align2::LEFT_TOP);
        }
    }
}

/// Round values from `lo` to `hi` drawn over `pixels`, about one every sixty pixels,
/// and how many decimals they need. (`plot::Ticks` works its step out in single
/// precision: from 0 to 1 by 0.2 it stops at 0.8, the step having come out a hair
/// above 0.2. Here the step is rounded to the round number it stands for.)
fn round_values(lo: f64, hi: f64, pixels: f32) -> (Vec<f64>, usize) {
    let step = Ticks::of(lo, hi, pixels).step;
    let digits = 10f64.powi(6 - step.log10().floor() as i32);
    let step = (step * digits).round() / digits;
    if !(step > 0.0 && step.is_finite() && hi > lo) {
        return (Vec::new(), 0);
    }
    let (first, last) = ((lo / step - 1e-9).ceil() as i64, ((hi / step + 1e-9).floor() as i64).min((lo / step).ceil() as i64 + 60));
    let decimals = (-(step.log10() + 1e-9).floor()).max(0.0).min(6.0) as usize;
    ((first..=last).map(|k| k as f64 * step).collect(), decimals)
}

/// Points in a chart's own units.
type Points = Vec<(f64, f64)>;

/// Where a belief's wall is, if it has one (else beyond every value).
fn wall_of(b: &Belief) -> f64 {
    match *b {
        Belief::Beta { cut, .. } if cut < 1.0 => cut,
        _ => f64::INFINITY,
    }
}

/// Where the curves are worked out: as many points in every frame, across the axis
/// but a hair inside its ends. (At p = 0 itself the density of Beta(a, b) jumps from b
/// to 0 as a passes 1; a hair inside, it changes smoothly with a.)
fn samples(model: Model) -> Vec<f64> {
    let [lo, hi] = model.range();
    let inside = (hi - lo) * 1e-4;
    (0..SAMPLES).map(|i| lo + inside + (hi - lo - 2.0 * inside) * i as f64 / (SAMPLES - 1) as f64).collect()
}

/// A belief's curve at the points `xs`, and its drop to 0 at a wall: the points beyond
/// the wall are drawn at the wall, so that each point of the curve stays the same
/// point as the wall moves; the drop is three points of its own, at the right end when
/// there is no wall (and then not seen: the third value says whether there is one).
fn curve_points(xs: &[f64], b: &Belief, d: &impl Fn(f64) -> f64) -> (Points, Points, bool) {
    let (cut, walled) = match *b {
        Belief::Beta { cut, .. } if cut < 1.0 => (cut, true),
        _ => (f64::INFINITY, false),
    };
    let line: Vec<(f64, f64)> = xs.iter().map(|&x| if x <= cut { (x, d(x)) } else { (cut, d(cut)) }).collect();
    let last = xs.last().copied().unwrap_or(0.0);
    let wall = if walled { vec![(cut, d(cut)), (cut, 0.0), (last, 0.0)] } else { vec![(last, d(last)); 3] };
    (line, wall, walled)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// An experiment as its button sets it up, run to its end.
    fn run(k: usize) -> Bayes {
        let mut s = Bayes::create();
        s.set_up(k);
        s.settle();
        s
    }

    /// The share of Beta(a, b) cut at `cut` between `lo` and `hi`, by Simpson's rule on
    /// a fine grid: the density as drawn, added up with no formula.
    fn area(b: &Belief, lo: f64, hi: f64) -> f64 {
        let d = b.density();
        let n = 200_000;
        let h = (hi - lo) / n as f64;
        let mut sum = d(lo) + d(hi);
        for i in 1..n {
            sum += d(lo + i as f64 * h) * if i % 2 == 1 { 4.0 } else { 2.0 };
        }
        sum * h / 3.0
    }

    /// The captions of the "things to try", held to account.
    #[test]
    fn the_experiments_show_what_their_captions_say() {
        // seven heads in ten, flat prior: Beta(8, 4), its peak at 0.70 (the share of
        // heads), its mean 8/12 = 0.67; the scaled likelihood is the posterior
        let s = run(0);
        let c = s.counts();
        assert_eq!((c.n, c.heads()), (10.0, 7.0));
        let post = s.posterior(0);
        assert_eq!(post, Belief::Beta { a: 8.0, b: 4.0, cut: 1.0 });
        assert!((post.peak().0 - 0.7).abs() < 1e-12 && c.estimate() == Some(0.7));
        assert_eq!((num(post.mean(), 2), num(8.0 / 12.0, 2)), ("0.67".to_string(), "0.67".to_string()));
        assert!(post.mean() > 0.5 && post.mean() < 0.7);
        let (like, top) = (likelihood(Model::Coin, c), post.peak().1);
        for x in samples(Model::Coin) {
            assert!((like(x) * top - post.pdf(x)).abs() < 1e-9 * top, "at {x}");
        }

        // a stubborn prior: Beta(50, 50) to Beta(57, 53), mean 0.50 to 0.52; the flat
        // prior's to 0.67; (100 × 0.5 + 7)/110
        let s = run(1);
        assert_eq!(s.prior[0], Belief::Beta { a: 50.0, b: 50.0, cut: 1.0 });
        assert_eq!(s.posterior(0), Belief::Beta { a: 57.0, b: 53.0, cut: 1.0 });
        assert_eq!(s.counts().heads(), 7.0);
        assert!(s.second && s.prior[1] == COIN_PRIORS[0].1);
        assert_eq!([s.prior[0].mean(), s.posterior(0).mean(), (100.0 * 0.5 + 7.0) / 110.0, s.posterior(1).mean()].map(|v| num(v, 2)), ["0.50", "0.52", "0.52", "0.67"]);

        // agreement in the end: prior means 0.2 and 0.8, each worth 20 tosses; the gap
        // is 12/(20 + n) whatever the tosses: 0.40, 0.10, 0.012; the two close in on
        // each other and on 0.7
        let mut s = Bayes::create();
        s.set_up(2);
        assert_eq!([s.prior[0].mean(), s.prior[1].mean(), s.prior[0].strength(), s.prior[1].strength()], [0.2, 0.8, 20.0, 20.0]);
        let mut gaps = Vec::new();
        for n in [10usize, 100, 1000] {
            while s.arrived < n {
                s.take();
            }
            let gap = s.posterior(1).mean() - s.posterior(0).mean();
            assert!((gap - 12.0 / (20.0 + n as f64)).abs() < 1e-12, "{n}: {gap}");
            gaps.push(gap);
        }
        assert_eq!([num(gaps[0], 2), num(gaps[1], 2), num(gaps[2], 3)], ["0.40", "0.10", "0.012"]);
        for who in 0..2 {
            let (lo, hi) = s.posterior(who).interval();
            assert!((s.posterior(who).mean() - 0.7).abs() < 0.02 && hi - lo < 0.06, "observer {who}");
        }

        // a closed mind: after 1000 tosses (713 heads) the mean is 0.499, the interval
        // 0.496 to 0.500, and the probability above one half exactly 0
        let s = run(3);
        let post = s.posterior(0);
        assert!(post.is_cut() && s.counts().n == 1000.0 && !s.second);
        assert_eq!((num(post.mean(), 3), post.above(0.5), prob_above(&post, 0.5)), ("0.499".to_string(), 0.0, "0".to_string()));
        let (lo, hi) = post.interval();
        assert_eq!((num(lo, 3), num(hi, 3)), ("0.496".to_string(), "0.500".to_string()));
        // the likelihood says the rate is near 0.7
        assert!((s.counts().estimate().unwrap_or(0.0) - 0.7).abs() < 0.02);

        // order does not matter: the three tails first, the posterior slides towards 0
        // (Beta(1, 4), mean 0.2), and ends exactly on Beta(8, 4), as in the first order
        let mut s = Bayes::create();
        s.set_up(4);
        assert_eq!(s.values, [0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]);
        assert_eq!(s.reference_belief(), Some(Belief::Beta { a: 8.0, b: 4.0, cut: 1.0 }));
        for _ in 0..3 {
            s.take();
        }
        assert_eq!(s.posterior(0), Belief::Beta { a: 1.0, b: 4.0, cut: 1.0 });
        assert!((s.posterior(0).mean() - 0.2).abs() < 1e-12);
        s.settle();
        assert_eq!(Some(s.posterior(0)), s.reference_belief());
        assert_eq!(s.posterior(0), run(0).posterior(0));
        // (the first experiment's tosses, in another order)
        let mut first = run(0).values;
        first.sort_by(f64::total_cmp);
        assert_eq!(first, s.values);

        // credible and confidence: 70 heads in 100; both from 0.60 to 0.78, apart by
        // less than 0.001
        let s = run(5);
        let c = s.counts();
        assert_eq!((c.n, c.heads()), (100.0, 70.0));
        let (lo, hi) = s.posterior(0).interval();
        let (wlo, whi) = confidence(Model::Coin, c).expect("an interval");
        assert_eq!([lo, hi, wlo, whi].map(|v| num(v, 2)), ["0.60", "0.78", "0.60", "0.78"]);
        assert!((lo - wlo).abs() < 0.001 && (hi - whi).abs() < 0.001, "{lo} {hi} {wlo} {whi}");
        assert!(s.show_confidence);

        // the mean of a bell: precision 1 + 4 = 5; (1 × 0 + 4 × 1.918)/5 = 1.53
        let s = run(6);
        let c = s.counts();
        assert_eq!(s.prior[0], Belief::Normal { mean: 0.0, sd: 1.0 });
        let xbar = c.estimate().unwrap_or(f64::NAN);
        let Belief::Normal { mean, sd } = s.posterior(0) else { panic!("a bell") };
        assert_eq!(c.n, 4.0);
        assert!((1.0 / (sd * sd) - 5.0).abs() < 1e-12 && (mean - 0.8 * xbar).abs() < 1e-12);
        assert_eq!((num(xbar, 3), num(mean, 2), num((4.0 * 1.918) / 5.0, 2)), ("1.918".to_string(), "1.53".to_string(), "1.53".to_string()));
        assert!(s.truth == 1.5 && s.show_truth);
    }

    /// Wilson's interval covers the true rate 0.7 in 94 % of samples of 100 tosses (what
    /// the "credible and confidence" caption says of it), worked out exactly over every
    /// number of heads.
    #[test]
    fn wilson_covers_ninety_four_in_a_hundred_here() {
        let n = 100u32;
        let coverage: f64 = (0..=n)
            .filter(|&h| confidence(Model::Coin, Counts { n: n as f64, sum: h as f64 }).is_some_and(|(lo, hi)| lo <= 0.7 && 0.7 <= hi))
            .map(|h| Dist::Binomial { n, p: 0.7 }.pdf(h as f64))
            .sum();
        assert_eq!(num(100.0 * coverage, 0), "94", "{coverage}");
        // and the interval for 7 heads in 10 is the one the books give
        let (lo, hi) = confidence(Model::Coin, Counts { n: 10.0, sum: 7.0 }).expect("an interval");
        assert_eq!((num(lo, 4), num(hi, 4)), ("0.3968".to_string(), "0.8922".to_string()));
        // the bell's: its mean ± 1.96 σ/√n, with σ = 1
        let (lo, hi) = confidence(Model::Bell, Counts { n: 4.0, sum: 4.0 }).expect("an interval");
        assert!((lo - (1.0 - normal_quantile(0.975) / 2.0)).abs() < 1e-12 && (hi + lo - 2.0).abs() < 1e-12);
        assert_eq!(confidence(Model::Coin, Counts::default()), None);
    }

    /// The posterior drawn is the prior times the likelihood, rescaled: worked out
    /// on a fine grid with no formula, for every prior here (the one with a wall too)
    /// and data of every kind, and the Beta parameters are a + heads, b + tails.
    #[test]
    fn the_posterior_is_the_prior_times_the_likelihood_rescaled() {
        let shaped = Belief::Beta { a: 2.35, b: 7.1, cut: 1.0 };
        let spike = Belief::Beta { a: 1000.0, b: 1000.0, cut: 1.0 };
        let walled = Belief::Beta { a: 3.0, b: 2.0, cut: 0.5 };
        let coins = COIN_PRIORS.iter().map(|p| p.1).chain([shaped, spike, walled]);
        for prior in coins {
            for (h, t) in [(0.0, 0.0), (7.0, 3.0), (10.0, 0.0), (0.0, 10.0), (70.0, 30.0), (713.0, 287.0)] {
                let c = Counts { n: h + t, sum: h };
                let post = prior.after(c);
                let Belief::Beta { a, b, cut } = prior else { panic!() };
                assert_eq!(post, Belief::Beta { a: a + h, b: b + t, cut });
                // prior × likelihood on a grid, rescaled to area 1 (Simpson's rule)
                let like = likelihood(Model::Coin, c);
                // (Simpson's rule from 0 to the wall: across it the product jumps)
                let (m, end) = (40_000, cut.min(1.0));
                let (before, after) = (prior.density(), post.density());
                let product: Vec<f64> = (0..=m).map(|i| end * i as f64 / m as f64).map(|x| before(x) * like(x)).collect();
                let total: f64 = product.iter().enumerate().map(|(i, v)| v * if i == 0 || i == m { 1.0 } else if i % 2 == 1 { 4.0 } else { 2.0 }).sum::<f64>() * end / (3.0 * m as f64);
                let peak = post.peak().1;
                for i in (0..=m).step_by(97) {
                    let x = end * i as f64 / m as f64;
                    let want = product[i] / total;
                    assert!((after(x) - want).abs() < 1e-7 * peak.max(1.0), "{prior:?} with {h} and {t} at {x}: {} against {want}", after(x));
                }
                // and nothing beyond the wall
                assert!(cut >= 1.0 || (post.pdf(cut + 1e-9) == 0.0 && post.pdf(0.9) == 0.0));
                // its mean: the prior counts as a + b earlier tosses (with no wall)
                if cut >= 1.0 {
                    let mean = ((a + b) * prior.mean() + h) / (a + b + h + t);
                    assert!((post.mean() - mean).abs() < 1e-12);
                }
            }
        }
        for prior in BELL_PRIORS.iter().map(|p| p.1) {
            for (n, sum) in [(0.0, 0.0), (4.0, 7.6), (100.0, 151.0), (1.0, -3.9)] {
                let c = Counts { n, sum };
                let post = prior.after(c);
                let like = likelihood(Model::Bell, c);
                let (lo, hi, m) = (-12.0, 12.0, 24_000);
                let h = (hi - lo) / m as f64;
                let product: Vec<f64> = (0..=m).map(|i| lo + i as f64 * h).map(|x| prior.pdf(x) * like(x)).collect();
                let total: f64 = product.iter().enumerate().map(|(i, v)| v * if i == 0 || i == m { 1.0 } else if i % 2 == 1 { 4.0 } else { 2.0 }).sum::<f64>() * h / 3.0;
                let peak = post.peak().1;
                for i in (0..=m).step_by(101) {
                    let x = lo + i as f64 * h;
                    assert!((post.pdf(x) - product[i] / total).abs() < 1e-6 * peak, "{prior:?} with {n}, {sum} at {x}");
                }
            }
        }
    }

    /// The intervals' ends are the quantiles: by `Dist::Beta` itself without a wall; with
    /// one, where the share below is 2.5 % and 97.5 % when the density is added up.
    /// The probability above a value, likewise.
    #[test]
    fn intervals_and_tails_are_what_the_density_says() {
        for (a, b) in [(1.0, 1.0), (8.0, 4.0), (57.0, 53.0), (717.0, 303.0), (1.0, 11.0)] {
            let post = Belief::Beta { a, b, cut: 1.0 };
            let beta = Dist::Beta { a, b };
            let out = (1.0 - LEVEL) / 2.0;
            assert_eq!(post.interval(), (beta.quantile(out), beta.quantile(1.0 - out)));
            assert!((out - 0.025).abs() < 1e-15);
            for t in [0.1, 0.5, 0.69] {
                assert!((post.above(t) - (1.0 - beta.cdf(t))).abs() < 1e-12);
            }
        }
        for (a, b) in [(1.0, 1.0), (3.0, 2.0), (8.0, 4.0), (714.0, 288.0), (60.0, 60.0)] {
            let post = Belief::Beta { a, b, cut: 0.5 };
            let (lo, hi) = post.interval();
            let whole = area(&post, 0.0, 0.5);
            assert!((whole - 1.0).abs() < 1e-6, "{a}, {b}: {whole}");
            assert!((area(&post, 0.0, lo) - 0.025).abs() < 1e-6 && (area(&post, 0.0, hi) - 0.975).abs() < 1e-6, "{a}, {b}: {lo} {hi}");
            assert!((area(&post, 0.3, 0.5) - post.above(0.3)).abs() < 1e-6);
            assert_eq!((post.above(0.5), post.above(0.8)), (0.0, 0.0));
            // its mean, added up
            let d = post.density();
            let n = 200_000;
            let mean: f64 = (0..n).map(|i| (i as f64 + 0.5) * 0.5 / n as f64).map(|x| x * d(x) * 0.5 / n as f64).sum();
            assert!((post.mean() - mean).abs() < 1e-6, "{a}, {b}: {} against {mean}", post.mean());
        }
        let bell = Belief::Normal { mean: 1.2, sd: 0.4 };
        let out = (1.0 - LEVEL) / 2.0;
        assert_eq!(bell.interval(), (Dist::Normal { mean: 1.2, sd: 0.4 }.quantile(out), Dist::Normal { mean: 1.2, sd: 0.4 }.quantile(1.0 - out)));
        assert!((bell.above(1.2) - 0.5).abs() < 1e-12);
    }

    /// The logarithm of the incomplete beta function agrees with `beta_i` where that is
    /// a number, and stays one where `beta_i` would be 0.
    #[test]
    fn the_log_share_agrees_where_it_can_and_goes_on_where_it_cannot() {
        for (a, b) in [(1.0, 1.0), (2.0, 3.0), (8.0, 4.0), (50.0, 50.0), (0.5, 3.0)] {
            for x in [0.01, 0.2, 0.5, 0.77, 0.99] {
                let (mine, theirs) = (ln_beta_below(a, b, x).exp(), beta_i(a, b, x));
                assert!((mine - theirs).abs() < 1e-12 * (1.0 + theirs), "{a}, {b}, {x}: {mine} against {theirs}");
            }
        }
        // far beyond what a number holds (e^-1200 or so), and still in order
        let (low, lower) = (ln_beta_below(3000.0, 300.0, 0.5), ln_beta_below(3000.0, 300.0, 0.45));
        assert!(low.is_finite() && low < -1000.0 && lower < low, "{low} {lower}");
        assert_eq!(beta_i(3000.0, 300.0, 0.5), 0.0);
        // where it is still a number, the two agree on it in their logarithms
        assert!((ln_beta_below(1600.0, 400.0, 0.5) - beta_i(1600.0, 400.0, 0.5).ln()).abs() < 1e-9);
        let walled = Belief::Beta { a: 3000.0, b: 300.0, cut: 0.5 };
        let (lo, hi) = walled.interval();
        assert!(lo.is_finite() && hi.is_finite() && lo < hi && hi <= 0.5 && walled.mean() < 0.5 && walled.mean() > lo);
        assert!(walled.pdf(0.4999).is_finite() && walled.peak().1.is_finite());
        assert!(ln_beta_below(2.0, 2.0, 0.0) == f64::NEG_INFINITY && ln_beta_below(2.0, 2.0, 1.0) == 0.0 && ln_beta_below(-1.0, 2.0, 0.5).is_nan());
    }

    /// The order of the observations does not matter: updating one at a time gives
    /// exactly what all at once does, in any order, and the replays end there.
    #[test]
    fn the_order_does_not_matter() {
        let s = run(5);
        let values = s.values.clone();
        let batch = s.prior[0].after(Counts::of(&values));
        let mut one_by_one = s.prior[0];
        for v in &values {
            one_by_one = one_by_one.after(Counts { n: 1.0, sum: *v });
        }
        assert_eq!(one_by_one, batch);
        let mut reversed = s.prior[0];
        for v in values.iter().rev() {
            reversed = reversed.after(Counts { n: 1.0, sum: *v });
        }
        assert_eq!(reversed, batch);
        // the replays: shuffled, and all at once
        for order in [Order::Shuffled, Order::AtOnce, Order::TailsFirst] {
            let mut s = run(5);
            s.replay(order);
            assert_eq!(s.reference_belief(), Some(batch));
            s.settle();
            assert_eq!(s.posterior(0), batch, "{order:?}");
            assert_eq!(Counts::of(&s.values).sum, 70.0);
        }
        // a replay asked for while one is on its way replays them all
        let mut s = run(5);
        s.replay(Order::Shuffled);
        for _ in 0..30 {
            s.take();
        }
        s.replay(Order::Shuffled);
        assert!(s.values.len() == 100 && s.reference == Some(100) && s.arrived == 0);
        s.settle();
        assert_eq!(s.posterior(0), batch);
        // the bell: one value at a time, to the last digits of rounding
        let s = run(6);
        let mut one = s.prior[0];
        for v in &s.values {
            one = one.after(Counts { n: 1.0, sum: *v });
        }
        let (Belief::Normal { mean: m1, sd: s1 }, Belief::Normal { mean: m2, sd: s2 }) = (one, s.posterior(0)) else { panic!() };
        assert!((m1 - m2).abs() < 1e-12 && (s1 - s2).abs() < 1e-12);
    }

    /// No data, a prior that is a spike, all heads, the threshold at either end, a
    /// wall the data contradict, the bell far off: every number drawn is finite, and
    /// the numbers say what has happened.
    #[test]
    fn degenerate_cases_are_told_honestly() {
        // no data: the posterior is the prior, the likelihood flat
        let mut s = Bayes::create();
        s.set_up(0);
        assert_eq!(s.posterior(0), s.prior[0]);
        assert!(samples(Model::Coin).into_iter().all(|x| likelihood(Model::Coin, s.counts())(x) == 1.0));
        // a spike: the strongest prior the handles make
        s.prior[0] = Belief::Beta { a: 0.5 * MOST_SURE, b: 0.5 * MOST_SURE, cut: 1.0 };
        s.settle();
        // all heads with a flat prior: the peak is at 1
        let mut heads = Bayes::create();
        heads.set_up(0);
        heads.values = vec![1.0; 10];
        heads.settle();
        let post = heads.posterior(0);
        assert_eq!(post, Belief::Beta { a: 11.0, b: 1.0, cut: 1.0 });
        assert_eq!(post.peak().0, 1.0);
        // the threshold at the ends
        assert_eq!((prob_above(&post, 0.0), prob_above(&post, 1.0)), ("1".to_string(), "0".to_string()));
        assert_eq!(prob_above(&Belief::Beta { a: 2.0, b: 900.0, cut: 1.0 }, 0.5), "< 0.01");
        assert_eq!(prob_above(&Belief::Beta { a: 900.0, b: 2.0, cut: 1.0 }, 0.5), "> 0.99");
        // a wall the data contradict: a thousand heads
        let mut wall = Bayes::create();
        wall.set_up(3);
        wall.values = vec![1.0; MOST];
        wall.settle();
        let post = wall.posterior(0);
        let (lo, hi) = post.interval();
        assert!(post.mean() < 0.5 && post.mean() > 0.49 && lo.is_finite() && hi <= 0.5 && post.above(0.5) == 0.0);
        // the goals and the layouts, for each of these and the experiments
        let lays = [(1075.0, 1000.0, 0.0), (390.0, 756.0, 44.0), (240.0, 180.0, 44.0), (1.0, 1.0, 0.0), (1400.0, 760.0, 0.0)];
        let mut scenes = vec![s, heads, wall];
        scenes.extend((0..TRIALS.len()).map(run));
        let mut far = run(6);
        far.truth = 3.0;
        far.values = vec![60.0, -60.0, 3.0];
        far.settle();
        scenes.push(far);
        for mut s in scenes {
            for share in [0.0f32, 0.37, 1.0] {
                s.share = share;
                let g = s.goals();
                let numbers = g.beliefs.iter().flatten().chain(&g.like).chain(&g.marks).chain(&g.alpha);
                assert!(numbers.clone().all(|v| v.is_finite()), "{:?}", s.prior[0]);
                s.look.follow(&g, 1.0 / 60.0, true);
                assert!(s.look.top(s.model).is_finite() && s.look.top(s.model) > 0.0);
                for k in 0..5 {
                    let b = s.look.belief(s.model, k);
                    let d = b.density();
                    let (line, wall, _) = curve_points(&samples(s.model), &b, &d);
                    assert!(line.iter().chain(&wall).all(|(x, y)| x.is_finite() && y.is_finite()), "{b:?}");
                }
            }
            for (w, h, inset) in lays {
                let lay = Layout::of(Vec2::new(w, h), inset, s.model);
                assert!([lay.chart, lay.strip, lay.key, lay.numbers, lay.formula, lay.tokens].iter().all(|(a, b)| a.is_finite() && b.is_finite()));
            }
        }
    }

    /// The layout of every window: finite, inside the canvas, the head above the chart
    /// above the rows above the numbers, and room for each.
    #[test]
    fn the_layout_fits_every_window() {
        for model in [Model::Coin, Model::Bell] {
            for (w, h, inset) in [(1075.0, 1000.0, 0.0), (1400.0, 760.0, 0.0), (390.0, 756.0, 44.0), (578.0, 700.0, 0.0), (700.0, 500.0, 0.0), (3000.0, 2000.0, 0.0)] {
                let lay = Layout::of(Vec2::new(w, h), inset, model);
                let inside = |(a, b): Rect| a.x >= 0.0 && a.y >= inset && b.x <= w + 0.5 && b.y <= h + 0.5 && b.x >= a.x && b.y >= a.y;
                assert!(!lay.compact, "{w}x{h}");
                for r in [lay.chart, lay.strip, lay.numbers, lay.tokens] {
                    assert!(inside(r), "{w}x{h}: {r:?}");
                }
                assert!(lay.buttons.iter().all(|b| inside(*b)), "{w}x{h}");
                let below = lay.strip.1.y < lay.numbers.0.y;
                let beside = lay.chart.1.x < lay.numbers.0.x && lay.chart.1.x < lay.key.0.x;
                assert!(lay.tokens.1.y <= lay.chart.0.y && lay.chart.1.y < lay.strip.0.y && (below || beside), "{w}x{h}");
                let tall = if h >= 640.0 { 250.0 } else { 120.0 };
                assert!(lay.chart.1.y - lay.chart.0.y > tall && lay.chart.1.x - lay.chart.0.x > 250.0, "{w}x{h}: {:?}", lay.chart);
                assert!(lay.numbers.1.y - lay.numbers.0.y > 60.0, "{w}x{h}: room for the numbers");
            }
            // too small for the chart: the numbers alone, from the top
            let lay = Layout::of(Vec2::new(240.0, 180.0), 44.0, model);
            assert!(lay.compact && lay.numbers.0.y >= 44.0);
        }
    }

    /// After a change, the picture glides to the new one and comes to rest exactly
    /// there, at any frame rate.
    #[test]
    fn glides_arrive_and_stop() {
        for hz in [30.0f32, 60.0, 144.0] {
            let mut s = run(0);
            s.look.follow(&s.goals(), 0.0, true);
            assert!(!s.look.moving);
            s.set_up(1);
            s.settle();
            let mut frames = 0;
            loop {
                let g = s.goals();
                s.look.follow(&g, 1.0 / hz, false);
                frames += 1;
                if !s.look.moving {
                    assert_eq!(s.look.beliefs[1].shown, g.beliefs[1].to_vec());
                    break;
                }
                assert!((frames as f32) < 2.0 * hz, "still gliding after two seconds at {hz} Hz");
            }
            assert!(frames as f32 > 0.2 * hz, "arrived in {frames} frames at {hz} Hz: no glide");
        }
    }

    /// Observations come at their rate in the scene's time, stop where the run should,
    /// and "faster as they come" takes as long from 10 to 100 as from 100 to 1000.
    #[test]
    fn a_run_comes_in_at_its_rate() {
        let mut s = Bayes::create();
        s.set_up(0);
        // (the steps cost nothing here, so the frame's allowance never runs out)
        let dt = 1.0 / 60.0;
        for _ in 0..60 {
            s.work(dt, dt);
            assert!((0.0..=1.0).contains(&s.share));
        }
        assert_eq!(s.arrived, 3);
        for _ in 0..600 {
            s.work(dt, dt);
        }
        assert!(s.arrived == 10 && !s.running && s.share == 1.0);
        // nothing comes while the scene's time stands still
        s.running = true;
        s.stop_at = None;
        s.work(0.0, dt);
        assert_eq!(s.arrived, 10);
        // faster as they come
        let mut s = Bayes::create();
        s.set_up(2);
        let (mut t, mut at) = (0.0f32, [0.0f32; 3]);
        while s.running {
            s.work(dt, dt);
            t += dt;
            for (k, n) in [10usize, 100, 1000].iter().enumerate() {
                if s.arrived >= *n && at[k] == 0.0 {
                    at[k] = t;
                }
            }
            assert!(t < 60.0);
        }
        let (first, second) = (at[1] - at[0], at[2] - at[1]);
        assert!(s.arrived == MOST && (first / second - 1.0).abs() < 0.05, "{first} s and {second} s");
        // and never past the most kept
        s.running = true;
        s.stop_at = None;
        s.work(1.0, dt);
        assert_eq!(s.arrived, MOST);
    }

    /// The handles shape the prior: its interval's end goes where it is dragged, its
    /// mean too (and it grows as sure as it must to keep its density finite), and a
    /// prior with a wall becomes an ordinary one.
    #[test]
    fn the_handles_shape_the_prior() {
        let mut s = Bayes::create();
        s.set_up(0);
        s.shape_prior(2, 0.7);
        let (lo, hi) = s.prior[0].interval();
        assert!((hi - 0.7).abs() < 1e-6 && (s.prior[0].mean() - 0.5).abs() < 1e-12 && (lo - 0.3).abs() < 1e-6, "{lo} {hi}");
        s.shape_prior(1, 0.45);
        assert!((s.prior[0].interval().0 - 0.45).abs() < 1e-6);
        s.shape_prior(0, 0.9);
        let Belief::Beta { a, b, cut } = s.prior[0] else { panic!() };
        assert!((a / (a + b) - 0.9).abs() < 1e-12 && a >= 1.0 && b >= 1.0 && cut == 1.0);
        // as wide as it can be, and as narrow
        s.shape_prior(0, 0.5);
        s.shape_prior(2, 1.0);
        assert_eq!(s.prior[0], Belief::Beta { a: 1.0, b: 1.0, cut: 1.0 });
        s.shape_prior(2, 0.5);
        assert!((s.prior[0].strength() - MOST_SURE).abs() < 1e-9);
        // a wall goes with the first drag
        s.prior[0] = COIN_PRIORS[5].1;
        s.shape_prior(0, 0.3);
        assert!(!s.prior[0].is_cut() && (s.prior[0].mean() - 0.3).abs() < 1e-12);
        // the bell: an end sets the spread
        s.switch(Model::Bell);
        s.shape_prior(2, 1.0);
        let Belief::Normal { mean, sd } = s.prior[0] else { panic!() };
        assert!((mean + normal_quantile(0.975) * sd - 1.0).abs() < 1e-12);
        s.shape_prior(0, f64::NAN);
        assert!(s.prior[0].mean().is_finite());
    }

    /// The truth decides what comes next, and nothing else: the posterior of the
    /// observations made does not depend on it.
    #[test]
    fn the_truth_is_not_seen() {
        let mut s = run(0);
        let before = s.posterior(0);
        s.truth = 0.1;
        assert_eq!(s.posterior(0), before);
        assert_eq!(s.goals().beliefs, { s.truth = 0.9; s.goals().beliefs });
    }

    /// Print the numbers behind the captions (run with --ignored --nocapture).
    #[test]
    #[ignore]
    fn explore() {
        for k in 0..TRIALS.len() {
            let s = run(k);
            let c = s.counts();
            let post = s.posterior(0);
            let (lo, hi) = post.interval();
            println!("{}: n {} sum {} | post {:?} mean {:.4} peak {:.4} | 95% {:.4} {:.4} | above {:.6}", TRIALS[k].button, c.n, c.sum, post, post.mean(), post.peak().0, lo, hi, post.above(s.threshold));
            if s.second {
                let other = s.posterior(1);
                println!("    second {:?} mean {:.4}, gap {:.5}", other, other.mean(), (post.mean() - other.mean()).abs());
            }
            if let Some((a, b)) = confidence(s.model, c) {
                println!("    confidence {a:.4} {b:.4}");
            }
        }
    }
}

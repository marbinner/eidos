//! Estimators and their errors: an estimate is worked out from a sample, so it changes
//! from sample to sample, and an estimator is judged by how its estimates are spread.

use super::plot::{self, Glide, Plot, Rect};
use super::*;
use crate::math::stats::{self as st, Dist};

pub const DEF: SceneDef = SceneDef {
    id: "estimators",
    title: "Estimators and their errors",
    group: ESTIMATION,
    blurb: "An estimate changes from sample to sample: bias, variance and mean squared error, the standard error, and why the variance divides by one less than the sample's size.",
    make: |_| Box::new(Estimators::create()),
};

// ----- what is estimated, from what -----------------------------------------------------

/// Three ways of estimating one thing, set against each other.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Contest {
    /// The population's variance: the squared distances from the sample's mean, added
    /// up and divided by n, by n − 1 and by n + 1.
    Spread,
    /// The centre of a symmetric population: the mean, the median, the trimmed mean.
    Centre,
    /// The top θ of a flat population from 0 to θ: twice the mean, the largest value,
    /// and the largest value times (n + 1)/n.
    Top,
}

impl Contest {
    const ALL: [Contest; 3] = [Contest::Spread, Contest::Centre, Contest::Top];

    fn index(self) -> usize {
        match self {
            Contest::Spread => 0,
            Contest::Centre => 1,
            Contest::Top => 2,
        }
    }

    /// Its name on the buttons.
    fn button(self) -> &'static str {
        ["the spread", "the centre", "the top of a range"][self.index()]
    }

    /// The estimators' names over their histograms.
    fn names(self) -> [&'static str; 3] {
        match self {
            Contest::Spread => [r"$S/n$, dividing by $n$", r"$S/(n - 1)$, dividing by $n - 1$", r"$S/(n + 1)$, dividing by $n + 1$"],
            Contest::Centre => ["the mean", "the median", "the trimmed mean (a tenth cut off each end)"],
            Contest::Top => ["twice the mean", "the largest value", r"the largest value $\times (n + 1)/n$"],
        }
    }

    /// Their short names, beside their bars of error.
    fn short(self) -> [&'static str; 3] {
        match self {
            Contest::Spread => [r"$S/n$", r"$S/(n - 1)$", r"$S/(n + 1)$"],
            Contest::Centre => ["mean", "median", "trimmed"],
            Contest::Top => ["twice the mean", "largest", "corrected"],
        }
    }

    /// What the estimates are estimates of (the common axis's name).
    fn of(self) -> &'static str {
        [r"estimate of $\sigma^2$", "estimate of the centre", r"estimate of $\theta$"][self.index()]
    }

    /// The same, over the histograms.
    fn title(self) -> &'static str {
        [r"estimates of $\sigma^2$", "estimates of the centre", r"estimates of $\theta$"][self.index()]
    }

    /// The shapes the population can have in this contest.
    fn shapes(self) -> &'static [Shape] {
        match self {
            Contest::Spread => &[Shape::Normal, Shape::Flat, Shape::Skewed],
            Contest::Centre => &[Shape::Normal, Shape::Flat, Shape::Heavy],
            Contest::Top => &[Shape::Flat],
        }
    }
}

/// The shapes a population can have.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Shape {
    /// A bell, variance 1.
    Normal,
    /// Flat, variance 1 (from −√3 to √3); for the top of a range, from 0 to θ.
    Flat,
    /// A long right tail (exponential), variance 1.
    Skewed,
    /// Heavy tails: Student's t with 3 degrees of freedom, scaled to variance 1.
    Heavy,
}

impl Shape {
    fn name(self) -> &'static str {
        match self {
            Shape::Normal => "a bell",
            Shape::Flat => "flat",
            Shape::Skewed => "a long right tail",
            Shape::Heavy => "heavy tails",
        }
    }
}

/// The top of the range in the third contest.
const THETA: f64 = 10.0;

const SQRT3: f64 = 1.732_050_807_568_877_2;

/// The population a contest draws from, in one of its shapes.
#[derive(Clone, Copy, Debug, PartialEq)]
struct Population {
    contest: Contest,
    shape: Shape,
}

impl Population {
    /// The distribution before scaling, and the factor its draws are multiplied by.
    fn dist(&self) -> (Dist, f64) {
        match (self.contest, self.shape) {
            (Contest::Top, _) => (Dist::Uniform { lo: 0.0, hi: THETA }, 1.0),
            (_, Shape::Normal) => (Dist::STANDARD_NORMAL, 1.0),
            (_, Shape::Flat) => (Dist::Uniform { lo: -SQRT3, hi: SQRT3 }, 1.0),
            (_, Shape::Skewed) => (Dist::Exponential { rate: 1.0 }, 1.0),
            (_, Shape::Heavy) => (Dist::StudentT { df: 3.0 }, 1.0 / SQRT3),
        }
    }

    fn draw(&self, rng: &mut Rng) -> f64 {
        let (d, scale) = self.dist();
        d.sample(rng) * scale
    }

    fn pdf(&self, x: f64) -> f64 {
        let (d, scale) = self.dist();
        d.pdf(x / scale) / scale
    }

    fn mean(&self) -> f64 {
        let (d, scale) = self.dist();
        d.mean().unwrap_or(0.0) * scale
    }

    fn variance(&self) -> f64 {
        let (d, scale) = self.dist();
        d.variance().unwrap_or(f64::NAN) * scale * scale
    }

    /// The true value of what the contest estimates.
    fn truth(&self) -> f64 {
        match self.contest {
            Contest::Spread => self.variance(),
            Contest::Centre => self.mean(),
            Contest::Top => THETA,
        }
    }

    /// Where the population's chart reaches, and whether it is drawn as flat (a
    /// rectangle) rather than as a curve.
    fn shown_range(&self) -> (f64, f64) {
        match (self.contest, self.shape) {
            (Contest::Top, _) => (-0.5, 21.0),
            (_, Shape::Skewed) => (-0.6, 6.4),
            _ => (-3.5, 3.5),
        }
    }

    fn flat(&self) -> bool {
        self.shape == Shape::Flat || self.contest == Contest::Top
    }

    /// The flat population's ends.
    fn ends(&self) -> (f64, f64) {
        if self.contest == Contest::Top {
            (0.0, THETA)
        } else {
            (-SQRT3, SQRT3)
        }
    }
}

/// The three estimates from one sample, sorted (None where an estimator has nothing to
/// go on: the spread of a single value divided by n − 1 is 0/0).
fn estimates(contest: Contest, sorted: &[f64]) -> [Option<f64>; 3] {
    let n = sorted.len();
    if n == 0 {
        return [None; 3];
    }
    let mean = st::mean(sorted);
    match contest {
        Contest::Spread => {
            let s = squares(sorted, mean);
            [Some(s / n as f64), (n >= 2).then(|| s / (n - 1) as f64), Some(s / (n + 1) as f64)]
        }
        Contest::Centre => {
            let cut = trimmed(n);
            [Some(mean), Some(st::quantile_sorted(sorted, 0.5)), Some(st::mean(&sorted[cut..n - cut]))]
        }
        Contest::Top => {
            let top = sorted[n - 1];
            [Some(2.0 * mean), Some(top), Some(top * (n + 1) as f64 / n as f64)]
        }
    }
}

/// The squared distances of the values from `from`, added up.
fn squares(values: &[f64], from: f64) -> f64 {
    values.iter().map(|x| (x - from) * (x - from)).sum()
}

/// How many values the trimmed mean leaves out at each end: a tenth of them, rounded
/// down (none below ten values).
fn trimmed(n: usize) -> usize {
    (n / 10).min(n.saturating_sub(1) / 2)
}

// ----- what the samples come to ----------------------------------------------------------

/// One estimator's record over the samples so far: the mean and spread of its estimates
/// (Welford's way, which stays accurate over many samples), and their mean squared
/// distance from the truth.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
struct Record {
    count: u64,
    mean: f64,
    m2: f64,
    squared_error: f64,
}

impl Record {
    fn add(&mut self, e: f64, truth: f64) {
        self.count += 1;
        let d = e - self.mean;
        self.mean += d / self.count as f64;
        self.m2 += d * (e - self.mean);
        self.squared_error += (e - truth) * (e - truth);
    }

    /// The average of the estimates less the truth.
    fn bias(&self, truth: f64) -> f64 {
        self.mean - truth
    }

    /// Their variance: the mean squared distance from their own average, dividing by
    /// the number of samples (so that bias² + variance is the mean squared error exactly).
    fn variance(&self) -> f64 {
        if self.count == 0 {
            0.0
        } else {
            self.m2 / self.count as f64
        }
    }

    /// The standard error: the estimates' standard deviation.
    fn se(&self) -> f64 {
        self.variance().sqrt()
    }

    /// The mean squared distance of the estimates from the truth, added up directly.
    fn mse(&self) -> f64 {
        if self.count == 0 {
            0.0
        } else {
            self.squared_error / self.count as f64
        }
    }
}

/// The samples of a run: their population, size and seed, and what each estimator has
/// made of them.
#[derive(Clone, Debug)]
struct Run {
    pop: Population,
    n: usize,
    rng: Rng,
    records: [Record; 3],
    /// Bins of the estimates, across the range the run's axis shows.
    hist: [st::Histogram; 3],
    /// The newest sample, sorted, and its estimates.
    sample: Vec<f64>,
    latest: [Option<f64>; 3],
    /// In the contest of the spread: the squared distances from the true mean, added
    /// up within each sample and then over the samples.
    around_mean: f64,
}

impl Run {
    fn new(pop: Population, n: usize, seed: u64, range: (f64, f64), bins: usize) -> Run {
        let hist = std::array::from_fn(|_| st::Histogram::new(range.0, range.1, bins));
        Run { pop, n: n.max(1), rng: Rng::new(seed), records: [Record::default(); 3], hist, sample: Vec::new(), latest: [None; 3], around_mean: 0.0 }
    }

    /// How many samples have been drawn.
    fn count(&self) -> u64 {
        self.records.iter().map(|r| r.count).max().unwrap_or(0)
    }

    /// Draw one more sample and record what each estimator makes of it.
    fn step(&mut self) {
        let (pop, n) = (self.pop, self.n);
        self.sample.clear();
        for _ in 0..n {
            let x = pop.draw(&mut self.rng);
            self.sample.push(x);
        }
        self.sample.sort_by(f64::total_cmp);
        self.latest = estimates(pop.contest, &self.sample);
        if pop.contest == Contest::Spread {
            self.around_mean += squares(&self.sample, pop.mean());
        }
        let truth = pop.truth();
        for k in 0..3 {
            if let Some(e) = self.latest[k] {
                self.records[k].add(e, truth);
                self.hist[k].add(e);
            }
        }
    }
}

/// A run kept to compare the next with, when only n has changed.
#[derive(Clone, Debug)]
struct Ghost {
    pop: Population,
    n: usize,
    hist: [st::Histogram; 3],
    records: [Record; 3],
    /// The range its axis showed.
    range: (f64, f64),
}

impl Ghost {
    fn of(run: &Run) -> Ghost {
        let range = (run.hist[0].lo, run.hist[0].hi);
        Ghost { pop: run.pop, n: run.n, hist: run.hist.clone(), records: run.records, range }
    }
}

// ----- the scales of a run -----------------------------------------------------------------

/// Samples in a run: it stops there, and its numbers are those the captions give.
const MOST: u64 = 20_000;
/// Samples in the run drawn beforehand for an experiment to compare with.
const BEFORE: u64 = 5000;
/// Samples drawn beforehand (with a seed of their own) to size the charts.
const PILOT: usize = 600;
/// While there are few samples, each is a block this share of a row's height.
const STACK: f64 = 6.0;
/// How much room the tallest bar is expected to leave above it.
const HEAD: f64 = 1.25;
/// Bins across the estimates' axis, about.
const BINS: f64 = 46.0;

/// What a run's charts show, sized from a pilot run beforehand so that nothing on them
/// has to change scale while the samples come.
#[derive(Clone, Debug, PartialEq)]
struct Scales {
    /// The estimates' common axis, and its bins.
    lo: f64,
    hi: f64,
    bins: usize,
    /// The largest share of the samples one bin is expected to hold.
    peak: f64,
    /// The end of the axis of the mean squared errors.
    mse_top: f64,
    /// Decimals for the estimates, and for the mean squared errors.
    digits: usize,
    mse_digits: usize,
}

/// A round step about `x` long (1, 2 or 5 times a power of ten).
fn round_step(x: f64) -> f64 {
    let x = if x.is_finite() && x > 0.0 { x } else { 1.0 };
    let p = 10f64.powf(x.log10().floor());
    let f = x / p;
    p * if f <= 1.0 {
        1.0
    } else if f <= 2.0 {
        2.0
    } else if f <= 5.0 {
        5.0
    } else {
        10.0
    }
}

/// Decimals that write numbers on an axis of this span to about a fiftieth of it.
fn digits_for(span: f64) -> usize {
    if !(span.is_finite() && span > 0.0) {
        return 2;
    }
    (-(span / 50.0).log10()).ceil().max(0.0).min(4.0) as usize
}

impl Scales {
    fn of(pop: Population, n: usize, seed: u64, ghost: Option<&Ghost>) -> Scales {
        let truth = pop.truth();
        let mut pilot = Run::new(pop, n, seed ^ 0x9E37_79B9_7F4A_7C15, (0.0, 1.0), 1);
        let mut values: [Vec<f64>; 3] = Default::default();
        for _ in 0..PILOT {
            pilot.step();
            for k in 0..3 {
                if let Some(e) = pilot.latest[k] {
                    values[k].push(e);
                }
            }
        }
        // the estimates' range: nearly all of every estimator's, and the truth
        let (mut lo, mut hi) = (truth, truth);
        for v in &mut values {
            v.sort_by(f64::total_cmp);
            if !v.is_empty() {
                lo = lo.min(st::quantile_sorted(v, 0.003));
                hi = hi.max(st::quantile_sorted(v, 0.997));
            }
        }
        if let Some(g) = ghost {
            lo = lo.min(g.range.0);
            hi = hi.max(g.range.1);
        }
        let pad = (hi - lo).max(1e-6) * 0.06;
        (lo, hi) = (lo - pad, hi + pad);
        match pop.contest {
            // (a variance is never below 0, and 0 belongs on its axis; one value has none)
            Contest::Spread => (lo, hi) = (0.0, hi.max(1.4 * truth)),
            Contest::Centre => {
                let reach = (truth - lo).max(hi - truth);
                (lo, hi) = (truth - reach, truth + reach);
            }
            Contest::Top => lo = lo.max(0.0),
        }
        // bins on round values, so that the truth falls on an edge between two
        let width = round_step((hi - lo) / BINS);
        let (lo, hi) = ((lo / width).floor() * width, (hi / width).ceil() * width);
        let bins = (((hi - lo) / width).round() as usize).max(1);
        let mut peak = 1.0 / bins as f64;
        let mut mse_top = 0.0f64;
        for (k, v) in values.iter().enumerate() {
            let h = st::Histogram::of(v, lo, hi, bins);
            let most = h.counts.iter().copied().max().unwrap_or(0) as f64;
            peak = peak.max(most / v.len().max(1) as f64);
            if !v.is_empty() {
                mse_top = mse_top.max(pilot.records[k].mse());
            }
        }
        if let Some(g) = ghost {
            for k in 0..3 {
                let h = &g.hist[k];
                let densest = (0..h.counts.len()).map(|i| h.density(i)).fold(0.0, f64::max);
                peak = peak.max(densest * width);
                mse_top = mse_top.max(g.records[k].mse());
            }
        }
        let mse_top = if mse_top.is_finite() && mse_top > 0.0 { mse_top * 1.3 } else { 1.0 };
        Scales { lo, hi, bins, peak, mse_top, digits: digits_for(hi - lo), mse_digits: digits_for(mse_top) }
    }

    fn width(&self) -> f64 {
        (self.hi - self.lo) / self.bins as f64
    }

    /// A value's place along the estimates' axis, 0 at its left end and 1 at its right.
    fn along(&self, v: f64) -> f32 {
        ((v - self.lo) / (self.hi - self.lo)) as f32
    }

    /// How many samples fill a row: a bar as tall as the tallest is expected to be,
    /// with room above it, once there are many; a block of a sixth of a row each
    /// while there are few.
    fn row_holds(&self, samples: u64) -> f64 {
        (samples as f64 * self.peak * HEAD).max(STACK)
    }
}

// ----- the experiments --------------------------------------------------------------------

/// An experiment: what it sets, and the run before it to compare with, if any.
struct Trial {
    button: &'static str,
    caption: &'static str,
    contest: Contest,
    shape: Shape,
    n: usize,
    /// A run with this n first, kept as an outline.
    before: Option<usize>,
}

const TRIALS: [Trial; 5] = [
    Trial {
        button: r"dividing by $n$ comes out low",
        caption: r"A bell of variance $\sigma^2 = 1$, samples of $n = 5$. Dividing by $n$, the estimates average 0.80: too low, by the factor $(n - 1)/n$, because the distances are measured from the sample's own mean, which lies closer to the sample than the true mean does (compare the two sums under the population). Dividing by $n - 1$, they average 1.00: unbiased. Yet the most biased, dividing by $n + 1$, has the least mean squared error: 0.33, against 0.36 and 0.50. Right on average is not the same as best.",
        contest: Contest::Spread,
        shape: Shape::Normal,
        n: 5,
        before: None,
    },
    Trial {
        button: "the mean against the median",
        caption: "A bell, samples of 25. Both are unbiased, but the median wobbles more: its standard error is 0.25, the mean's 0.20, a quarter as much again. For a bell the mean is the better (the more efficient) estimator of the centre.",
        contest: Contest::Centre,
        shape: Shape::Normal,
        n: 25,
        before: None,
    },
    Trial {
        button: "heavy tails",
        caption: "The same with heavy tails (Student's $t$ with 3 degrees of freedom, scaled to variance 1). The mean's standard error is 0.20 again: for the mean only the variance matters. But the median's is now 0.16: a few wild values throw the mean about, and the median does not care how far out they are. The trimmed mean, at 0.15, does better still. Which estimator is best depends on the population.",
        contest: Contest::Centre,
        shape: Shape::Heavy,
        n: 25,
        before: None,
    },
    Trial {
        button: "the top of a range",
        caption: r"Values spread evenly from 0 to $\theta = 10$, samples of 10. Twice the mean is unbiased but wobbles: mean squared error 3.33. The largest value is always low, by 0.9 on average, yet wobbles so much less that this run measures its error at only 1.55 (1.52 in the long run). Stretched by $(n + 1)/n$ it is unbiased too, and its error, 0.86 here, is in the long run exactly a quarter of twice the mean's: 0.83 against 3.33.",
        contest: Contest::Top,
        shape: Shape::Flat,
        n: 10,
        before: None,
    },
    Trial {
        button: "four times the data",
        caption: r"Back to the bell, now with samples of 100; the grey outlines are samples of 25. Four times the data halves every standard error (the mean's from 0.20 to 0.10, which is $\sigma/\sqrt{n}$) and quarters every mean squared error.",
        contest: Contest::Centre,
        shape: Shape::Normal,
        n: 100,
        before: Some(25),
    },
];

/// The seed every experiment, and the opening, draws its samples from.
const FIRST_SEED: u64 = 20;

// ----- the scene -------------------------------------------------------------------------

/// The three estimators of a contest, in the order they are listed: their colours.
const COLORS: [Color; 3] = [ESTIMATE, OTHER, pal::BLUE];
/// The sample's own mean, in the contest of the spread.
const XBAR: Color = pal::WHITE;

/// A run is kept as an outline to compare with, when n is changed, once it has this
/// many samples.
const GHOST_MIN: u64 = 300;
/// The rate the samples start at (a second), and how much faster each sample makes the
/// next come: slow enough at first to see each one land, then up to the rate asked for.
const RAMP_START: f32 = 2.5;
const RAMP_GROWTH: f32 = 1.5;
/// Samples a second, to begin with and in every experiment.
const RATE: f32 = 2000.0;
/// At most this many samples a second are shown on the left.
const SHOW_RATE: f32 = 5.0;
/// Seconds a sample shown takes to come in, or to go.
const FADE: f32 = 0.15;

/// A sample shown on the left: its values, sorted, and what they give; how long it has
/// been shown, and how long ago it was replaced.
#[derive(Clone, Debug)]
struct Shown {
    values: Vec<f64>,
    estimates: [Option<f64>; 3],
    mean: f64,
    age: f32,
    leaving: Option<f32>,
}

impl Shown {
    fn alpha(&self) -> f32 {
        let fade_in = (self.age / FADE).min(1.0);
        let fade_out = self.leaving.map_or(1.0, |l| (1.0 - l / FADE).max(0.0));
        fade_in * fade_out
    }
}

/// Where the marks drawn for each estimator are in `Estimators::marks`: its average,
/// standard error, bias², variance, and the estimate of the sample shown (all as
/// shares of their axes, so that a glide comes to rest within a hair of a pixel).
const PER: usize = 5;
/// After them, the sample shown's estimates on the population's axis, and its mean.
const TIERS: usize = 3 * PER;
const MEAN_MARK: usize = TIERS + 3;
const MARKS: usize = MEAN_MARK + 1;

pub struct Estimators {
    contest: Contest,
    /// The shape chosen in each contest.
    shapes: [Shape; 3],
    n: usize,
    /// Samples a second, once the first few have come.
    rate: f32,
    seed: u64,
    run: Run,
    scales: Scales,
    ghost: Option<Ghost>,
    pace: Pace,
    /// Scene time since a sample was last shown on the left.
    since_shown: f32,
    shown: Vec<Shown>,
    /// The histograms' bars (in rows' heights) and the marks, as drawn.
    bars: Glide,
    marks: Glide,
    /// Was anything still on its way in the last frame?
    moving: bool,
}

impl Estimators {
    /// (Nothing here needs the graphics device, so the tests can make one too.)
    fn create() -> Estimators {
        let pop = Population { contest: Contest::Spread, shape: Shape::Normal };
        let scales = Scales::of(pop, 5, FIRST_SEED, None);
        let run = Run::new(pop, 5, FIRST_SEED, (scales.lo, scales.hi), scales.bins);
        let mut s = Estimators {
            contest: Contest::Spread,
            shapes: [Shape::Normal; 3],
            n: 5,
            rate: RATE,
            seed: FIRST_SEED,
            run,
            scales,
            ghost: None,
            pace: Pace::default(),
            since_shown: 0.0,
            shown: Vec::new(),
            bars: Glide::default(),
            marks: Glide::default(),
            moving: true,
        };
        s.set_up(0);
        s
    }

    fn pop(&self) -> Population {
        Population { contest: self.contest, shape: self.shapes[self.contest.index()] }
    }

    /// Set up one of the experiments in `TRIALS`.
    fn set_up(&mut self, k: usize) {
        let t = &TRIALS[k];
        self.contest = t.contest;
        self.shapes[t.contest.index()] = t.shape;
        self.rate = RATE;
        self.seed = FIRST_SEED;
        self.ghost = None;
        if let Some(n) = t.before {
            // the run to compare with, whole
            let pop = self.pop();
            let scales = Scales::of(pop, n, FIRST_SEED, None);
            let mut before = Run::new(pop, n, FIRST_SEED, (scales.lo, scales.hi), scales.bins);
            while before.count() < BEFORE {
                before.step();
            }
            self.ghost = Some(Ghost::of(&before));
        }
        self.n = t.n;
        self.restart();
    }

    /// The settings have changed (or new samples are wanted): start a new run. A run
    /// that differs from the last in n alone keeps the last as an outline to compare
    /// with (or the outline it had, while the last has too few samples to show).
    fn change(&mut self) {
        let pop = self.pop();
        let kept = match self.ghost.take() {
            _ if self.run.pop == pop && self.run.n != self.n && self.run.count() >= GHOST_MIN => Some(Ghost::of(&self.run)),
            Some(g) if g.pop == pop && g.n != self.n => Some(g),
            _ => None,
        };
        self.ghost = kept;
        self.restart();
    }

    /// A new run, with the settings as they are, and its first sample at once.
    fn restart(&mut self) {
        self.n = self.n.clamp(1, 200);
        self.rate = self.rate.max(1.0).min(5000.0);
        let pop = self.pop();
        self.scales = Scales::of(pop, self.n, self.seed, self.ghost.as_ref());
        self.run = Run::new(pop, self.n, self.seed, (self.scales.lo, self.scales.hi), self.scales.bins);
        self.pace.reset();
        self.shown.clear();
        self.run.step();
        self.show_newest();
        self.since_shown = 0.0;
        // (the bars grow from nothing; the marks start where the first sample puts them)
        self.bars.snap(&vec![0.0; 3 * self.scales.bins]);
        self.marks.snap(&self.mark_goals());
        self.moving = true;
    }

    /// Samples a second now: slowly at first, then faster with each sample, up to the
    /// rate asked for.
    fn rate_now(&self) -> f32 {
        (RAMP_START + RAMP_GROWTH * self.run.count() as f32).min(self.rate)
    }

    /// The samples `dt` seconds of the scene's time ask for, within the frame's
    /// allowance; one of them shown on the left every so often.
    fn advance(&mut self, dt: f32, real_dt: f32) {
        if self.run.count() >= MOST {
            return;
        }
        let rate = self.rate_now();
        let mut turn = scene_turn(&mut self.pace, rate, dt, real_dt);
        self.since_shown += dt;
        while self.run.count() < MOST && turn.next() {
            self.run.step();
            if rate <= SHOW_RATE || self.since_shown >= 1.0 / SHOW_RATE {
                self.show_newest();
                self.since_shown = 0.0;
            }
        }
        self.pace.end(turn);
    }

    /// Show the newest sample on the left, the one shown before fading out.
    fn show_newest(&mut self) {
        for s in &mut self.shown {
            s.leaving.get_or_insert(0.0);
        }
        // (no more than a few at once, however fast they come)
        if self.shown.len() > 3 {
            self.shown.remove(0);
        }
        let values = self.run.sample.clone();
        let mean = st::mean(&values);
        self.shown.push(Shown { values, estimates: self.run.latest, mean, age: 0.0, leaving: None });
    }

    /// The sample shown now (the newest).
    fn current(&self) -> Option<&Shown> {
        self.shown.iter().rev().find(|s| s.leaving.is_none())
    }

    /// What the bars are going to: each bin's count as a share of a row's height.
    fn bar_goals(&self) -> Vec<f32> {
        let holds = self.scales.row_holds(self.run.count());
        self.run.hist.iter().flat_map(|h| h.counts.iter().map(move |&c| (c as f64 / holds) as f32)).collect()
    }

    /// What the marks are going to (see `PER`).
    fn mark_goals(&self) -> Vec<f32> {
        let (sc, truth) = (&self.scales, self.run.pop.truth());
        let mut m = vec![0.0f32; MARKS];
        let shown = self.current();
        for k in 0..3 {
            let r = &self.run.records[k];
            if r.count > 0 {
                let span = sc.hi - sc.lo;
                m[k * PER] = sc.along(r.mean);
                m[k * PER + 1] = (r.se() / span) as f32;
                m[k * PER + 2] = (r.bias(truth).powi(2) / sc.mse_top) as f32;
                m[k * PER + 3] = (r.variance() / sc.mse_top) as f32;
            }
            if let Some(e) = shown.and_then(|s| s.estimates[k]) {
                m[k * PER + 4] = sc.along(e);
                m[TIERS + k] = self.on_population(e);
            }
        }
        if let Some(s) = shown {
            m[MEAN_MARK] = self.on_population(s.mean);
        }
        // (nothing that is not a number goes to a glide: it would never arrive)
        for v in &mut m {
            if !v.is_finite() {
                *v = 0.0;
            }
        }
        m
    }

    /// A value's place along the population's axis, 0 at its left end and 1 at its right.
    fn on_population(&self, v: f64) -> f32 {
        let (lo, hi) = self.run.pop.shown_range();
        ((v - lo) / (hi - lo)) as f32
    }

    /// The mark `i` as drawn, back in the units of its axis.
    fn mark(&self, i: usize) -> f64 {
        self.marks.get(i) as f64
    }

    /// Is the estimator `k` defined at this n? (Dividing by n − 1 needs two values.)
    fn defined(&self, k: usize) -> bool {
        !(self.contest == Contest::Spread && k == 1 && self.n < 2)
    }

    /// The estimators with the least mean squared error so far (more than one where
    /// they are the same estimator at this n: the mean and the trimmed mean below ten
    /// values, say).
    fn best(&self) -> [bool; 3] {
        let r = &self.run.records;
        let least = (0..3).filter(|&k| r[k].count > 0).map(|k| r[k].mse()).fold(f64::INFINITY, f64::min);
        [0, 1, 2].map(|k| r[k].count > 0 && r[k].mse() <= least + 1e-12 * (1.0 + least))
    }
}

// ----- the layout ------------------------------------------------------------------------

/// Where everything goes on a canvas (logical pixels from its top-left corner).
#[derive(Clone, Copy, Debug)]
struct Layout {
    narrow: bool,
    /// Text sizes: titles, words, small print.
    big: f32,
    text: f32,
    small: f32,
    buttons: [Rect; 3],
    /// The population's chart, and the lines about the sample under it.
    pop: Rect,
    words: Rect,
    /// The sampling distributions: their title (its bottom-left corner), the three
    /// rows, and the line of the common axis under them.
    title: Vec2,
    rows: [Rect; 3],
    axis_y: f32,
    /// The bars of error: their title, where the bars themselves go, and the edges of
    /// the column they are in (their names go left of the bars).
    mse_title: Vec2,
    mse: Rect,
    mse_column: (f32, f32),
    /// Too small a canvas to draw the charts on.
    tiny: bool,
}

const NO_RECT: Rect = (Vec2::ZERO, Vec2::ZERO);

impl Layout {
    fn of(view: Vec2, inset: f32) -> Layout {
        let view = view.max(Vec2::splat(1.0));
        let narrow = view.x < 760.0;
        let (big, text, small) = if narrow { (13.0, 11.5, 10.5) } else { (14.5, 12.5, 11.5) };
        let side = if narrow { 10.0 } else { 20.0 };
        let top = inset.max(0.0) + if narrow { 8.0 } else { 12.0 };
        let button_h = if narrow { 26.0 } else { 24.0 };
        let buttons: Vec<Rect> = if narrow {
            plot::even_row(3, side, view.x - side, top, button_h, 6.0)
        } else {
            let mut x = side;
            Contest::ALL
                .iter()
                .map(|c| {
                    let w = text_width(c.button(), small) + 26.0;
                    let r = (Vec2::new(x, top), Vec2::new(x + w, top + button_h));
                    x += w + 8.0;
                    r
                })
                .collect()
        };
        let buttons = [0, 1, 2].map(|i| buttons.get(i).copied().unwrap_or(NO_RECT));
        let y0 = top + button_h + if narrow { 10.0 } else { 18.0 };
        // what each part needs beyond its chart: the population's two lines of title and
        // its numbers; the lines on the sample; a row's name above it and its standard
        // error under it; the common axis's numbers and name; the bars' title and axis
        let pop_title = 2.0 * big + 10.0;
        let numbers = small + 12.0;
        let axis = 2.0 * small + 18.0;
        let words_h = 4.0 * (small + 5.0) + 4.0;
        let label = text + 9.0;
        let under = if narrow { 13.0 } else { 17.0 };
        let mse_title = big + 12.0;
        let title_h = big + 12.0;
        let (pop, words, title, mut rows, axis_y, mse_title_at, mse, mse_column);
        if narrow {
            let (x0, x1) = (side, view.x - side);
            let fixed = y0 + pop_title + numbers + words_h + 8.0 + title_h + 3.0 * (label + under) + axis + mse_title + numbers + 6.0;
            let rest = (view.y - fixed).max(0.0);
            let ph = (rest * 0.3).clamp(30.0, 170.0);
            let mh = (rest * 0.15).clamp(30.0, 78.0);
            let rh = ((rest - ph - mh) / 3.0).max(0.0);
            let mut y = y0 + pop_title;
            pop = (Vec2::new(x0 + 4.0, y), Vec2::new(x1, y + ph));
            y += ph + numbers;
            words = (Vec2::new(x0, y), Vec2::new(x1, y + words_h));
            y += words_h + 8.0 + title_h;
            title = Vec2::new(x0, y - 8.0);
            rows = [NO_RECT; 3];
            for row in &mut rows {
                y += label;
                *row = (Vec2::new(x0, y), Vec2::new(x1, y + rh));
                y += rh + under;
            }
            axis_y = y;
            y += axis + mse_title;
            mse_title_at = Vec2::new(x0, y - 8.0);
            let names = Contest::ALL.iter().flat_map(|c| c.short()).map(|s| text_width(s, small)).fold(0.0, f32::max) + 10.0;
            mse = (Vec2::new(x0 + names, y), Vec2::new(x1 - 64.0, y + mh));
            mse_column = (x0, x1);
        } else {
            // on the left the population, the sample and the bars of error; on the
            // right the sampling distributions
            let lw = ((view.x - 2.0 * side) * 0.37).max(220.0);
            let (x0, x1) = (side, side + lw);
            // (the bars of error at the foot of the column, the lines on the sample
            // above them, at least seven lines of them, and the population given the rest)
            let mh = ((view.y - y0) * 0.2).clamp(60.0, 200.0);
            let mse_top = view.y - 10.0 - numbers - mh;
            let words_min = 7.0 * (small + 5.0) + 4.0;
            let ph = (mse_top - mse_title - 20.0 - words_min - numbers - 6.0 - pop_title - y0).clamp(40.0, 460.0);
            let mut y = y0 + pop_title;
            pop = (Vec2::new(x0 + 6.0, y), Vec2::new(x1, y + ph));
            y += ph + numbers + 6.0;
            words = (Vec2::new(x0, y), Vec2::new(x1, (mse_top - mse_title - 20.0).max(y)));
            mse_title_at = Vec2::new(x0, mse_top - 8.0);
            let names = Contest::ALL.iter().flat_map(|c| c.short()).map(|s| text_width(s, small)).fold(0.0, f32::max) + 14.0;
            mse = (Vec2::new(x0 + names, mse_top), Vec2::new(x1 - 76.0, mse_top + mh));
            mse_column = (x0, x1);
            let (r0, r1) = (x1 + 48.0, view.x - side);
            let rest = (view.y - y0 - title_h - 3.0 * (label + under) - axis - 6.0).max(0.0);
            let rh = rest / 3.0;
            let mut y = y0 + title_h;
            title = Vec2::new(r0, y - 8.0);
            rows = [NO_RECT; 3];
            for row in &mut rows {
                y += label;
                *row = (Vec2::new(r0, y), Vec2::new(r1, y + rh));
                y += rh + under;
            }
            axis_y = y;
        }
        let rh = rows[0].1.y - rows[0].0.y;
        let tiny = view.x < 220.0 || rh < 22.0 || pop.1.y - pop.0.y < 30.0 || mse.1.x - mse.0.x < 40.0 || mse.1.y - mse.0.y < 24.0;
        Layout { narrow, big, text, small, buttons, pop, words, title, rows, axis_y, mse_title: mse_title_at, mse, mse_column, tiny }
    }
}

/// Text broken into lines no wider than `width`, from `at` down, no lower than
/// `bottom`: returns the height taken. (Widths by `text_width`, which measures
/// formulas exactly and keeps what it measured.)
fn wrap_px(f: &mut Frame, at: Vec2, width: f32, bottom: f32, text: &str, size: f32, color: Color) -> f32 {
    let space = size * 0.3;
    let mut lines: Vec<(String, f32)> = Vec::new();
    for word in words_of(text) {
        let w = text_width(word, size);
        match lines.last_mut() {
            Some((line, used)) if *used + space + w <= width => {
                line.push(' ');
                line.push_str(word);
                *used += space + w;
            }
            _ => lines.push((word.to_string(), w)),
        }
    }
    let step = size + 5.0;
    let mut used = 0.0;
    for (line, _) in &lines {
        if at.y + used + step > bottom + 1.0 {
            break;
        }
        f.text_px(at + Vec2::new(0.0, used), line.as_str(), size, color, Align2::LEFT_TOP);
        used += step;
    }
    used
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

/// The kinds of line drawn in batches of their own (see `plot::apart`): the
/// histograms' outlines, the distances of the sample shown from its mean (as many as
/// the samples on the way in and out), and the outlines of the run before.
const OUTLINES: u8 = plot::OWN;
const DISTANCES: u8 = plot::OWN + 1;
const GHOSTS: u8 = plot::OWN + 2;

/// The dots of the samples shown, in a batch of their own: their number changes as a
/// sample comes and goes.
fn sample_dot(f: &mut Frame, at: Vec2, size: f32, color: Color) {
    if at.is_finite() {
        let p = f.hud(at);
        f.dl.points_with(PointKind::Overlay, PointStyle { size_scale: 1.0001, outline_px: 0.0, ..Default::default() }).push(p, size, color);
    }
}

/// Where the i-th of n dots goes up the strip of the sample: spread evenly, in an
/// order that has nothing to do with their values.
fn level(i: usize) -> f32 {
    ((i as f32 * 0.618_034 + 0.31) % 1.0) * 0.9 + 0.05
}

impl Scene for Estimators {
    fn view(&self) -> View {
        View::Flat { center: Vec2::ZERO, half_height: 1.0 }
    }

    fn settings(&self, s: &mut ViewSettings) {
        s.grid = false;
        s.bloom = s.bloom.min(0.2);
    }

    fn animated(&self) -> bool {
        self.run.count() < MOST || self.moving
    }

    fn pointer_hint(&self) -> Option<&'static str> {
        Some("buttons over the picture: click")
    }

    fn ui(&mut self, ui: &mut egui::Ui, _: &Gfx) {
        ui::note(ui, "An estimate is a number worked out from a sample, meant to come close to something about the population the sample came from. Draw another sample and it comes out different: an estimate has a distribution of its own, its sampling distribution. Here the population is known (in practice it never is), so every estimate can be set against the truth. Three estimators of the same thing are tried on thousands of samples and judged by how their estimates fall.");
        ui::equation(ui, r"\text{mean squared error} = \text{bias}^2 + \text{variance}");

        ui::section(ui, "Things to try");
        for (k, t) in TRIALS.iter().enumerate() {
            if ui::experiment(ui, t.button, t.caption) {
                self.set_up(k);
            }
        }

        ui::section(ui, "The contest");
        let mut c = self.contest.index();
        if ui::choice(ui, &Contest::ALL.map(|c| c.button()), &mut c) {
            self.contest = Contest::ALL[c];
            self.change();
        }
        ui::note(
            ui,
            match self.contest {
                Contest::Spread => r"Estimating the population's variance $\sigma^2$ from $S = \sum (x_i - \bar x)^2$, the squared distances of the values from their own mean $\bar x$, added up: divided by $n$, by $n - 1$, and by $n + 1$.",
                Contest::Centre => "Estimating the centre of a symmetric population: by the mean; by the median, the middle value; and by the trimmed mean, the mean of what is left when a tenth of the values is cut off each end (rounded down: below ten values nothing is cut, and it is the mean).",
                Contest::Top => r"Values spread evenly from 0 to a top $\theta$ (here 10), to be estimated: by twice the mean; by the largest value; and by the largest value stretched by $(n + 1)/n$.",
            },
        );
        if self.contest == Contest::Spread {
            ui::note(ui, r"Why $n - 1$: the sample's own mean is the point closest to the sample (in squared distance), so $S$ is never more than the same sum measured from the true mean $\mu$: it is less by exactly $n(\bar x - \mu)^2$.");
            ui::equation(ui, r"\sum (x_i - \mu)^2 = \sum (x_i - \bar x)^2 + n(\bar x - \mu)^2");
            ui::note(ui, r"On average the left side is $n\sigma^2$ and the last term $\sigma^2$ (the variance of $\bar x$ is $\sigma^2/n$), so $S$ averages $(n - 1)\sigma^2$: for independent values from any population with a finite variance.");
        }

        let shapes = self.contest.shapes();
        if shapes.len() > 1 {
            ui::section(ui, "The population");
            let names: Vec<&str> = shapes.iter().map(|s| s.name()).collect();
            let i = self.contest.index();
            let mut s = shapes.iter().position(|s| *s == self.shapes[i]).unwrap_or(0);
            if ui::choice(ui, &names, &mut s) {
                self.shapes[i] = shapes[s.min(shapes.len() - 1)];
                self.change();
            }
            ui::note(
                ui,
                match self.contest {
                    Contest::Spread => r"Each has variance $\sigma^2 = 1$. Dividing by $n - 1$ is unbiased for all three, but which divisor has the least error depends on the tails. In the long run at $n = 5$: $n + 1$ for the bell (0.33) and for the long tail (0.87), but $n$ for the flat shape (0.21, against 0.23 for $n + 1$).",
                    _ => r"Each has variance 1, so the mean's standard error is $1/\sqrt{n}$ for all three. Only symmetric shapes: with a lopsided one the mean and the median would be estimating two different things. Heavy tails: Student's $t$ with 3 degrees of freedom, scaled.",
                },
            );
        }

        ui::section(ui, "The samples");
        if ui::slider_int_log(ui, r"values in each sample ($n$)", &mut self.n, 1..=200) {
            self.change();
        }
        ui::note(ui, "A new $n$ starts again; the run before (once it has a few hundred samples) stays behind as a grey outline to compare with.");
        ui::slider_log(ui, "samples per second", &mut self.rate, 1.0..=5000.0);
        ui::note(ui, "The first few samples come slowly, so that each can be seen landing, then faster up to this rate. A run stops at 20 000 samples.");
        if ui.button("start over with new samples").clicked() {
            self.seed = self.seed.wrapping_add(1);
            self.ghost = None;
            self.restart();
        }

        ui::section(ui, "Reading the picture");
        let names = self.contest.names();
        let mut key: Vec<(Color, &str)> = vec![(TRUTH, "green: the population, and the true value of what is estimated"), (DATA, "cyan dots: the sample just drawn")];
        if self.contest == Contest::Spread {
            key.push((XBAR, r"white: the sample's own mean $\bar x$, and each value's distance from it (for samples of up to 20)"));
        }
        for k in 0..3 {
            key.push((COLORS[k], names[k]));
        }
        key.push((pal::MUTED, "each estimator, in its colour: its estimate from the sample shown (left); a histogram of its estimates from all the samples so far; their average (the line) and one standard error either side of it (the bar under it)"));
        key.push((pal::WHITE, "white dot on each histogram's axis: the estimate from the sample shown"));
        key.push((ALARM, r"pink: the bias, from the truth to the average of the estimates; and $\text{bias}^2$, the first part of each bar of error"));
        key.push((pal::DIM, "grey outlines: the run before, with another $n$"));
        ui::key(ui, &key);

        ui::section(ui, "Words");
        ui::term(ui, "Estimator:", r"a rule for working a number out from a sample, such as “add the values up and divide by $n$”.");
        ui::term(ui, "Estimate:", "what the rule gives for one sample.");
        ui::term(ui, "Sampling distribution:", "how an estimator's estimates are spread over all the samples that could be drawn. The histograms come closer to it as the samples pile up.");
        ui::term(ui, "Bias:", "the average of the estimates less the true value: how far off the estimator is on average.");
        ui::term(ui, "Unbiased:", "bias 0: right on average, which is not the same as right each time.");
        ui::term(ui, "Standard error:", r"the standard deviation of the sampling distribution: how far one estimate typically lies from the average of all of them. For the mean of $n$ independent values it is $\sigma/\sqrt{n}$, where $\sigma$ is the population's standard deviation.");
        ui::term(ui, "Mean squared error:", r"the average of $(\text{estimate} - \text{truth})^2$: how far off the estimates are, from both causes together (being off on average, and wobbling). It is exactly $\text{bias}^2 + \text{variance}$ (the variance of the estimates, the square of the standard error).");
        ui::term(ui, "Efficiency:", "of two unbiased estimators, the one with the smaller variance is the more efficient; the ratio of the variances is the relative efficiency.");
        ui::term(ui, "Consistent:", r"as $n$ grows, the chance of an estimate further from the truth than any given distance shrinks to 0. $S/n$ is biased but consistent: its bias, $-\sigma^2/n$, dies away as $n$ grows, and (for the populations here) so does its wobble.");
        ui::term(ui, "Degrees of freedom:", r"the distances $x_i - \bar x$ add up to 0, so once $n - 1$ of them are known the last is too: $S$ has $n - 1$ degrees of freedom, and it averages $(n - 1)\sigma^2$.");
        ui::note(ui, r"Here the variance of the estimates is their mean squared distance from their own average, divided by the number of samples $N$, so that the three numbers add up exactly. Dividing by $N - 1$ instead would make it larger by the factor $N/(N - 1)$: by 0.1 % at a thousand samples, by 0.005 % at 20 000.");

        ui::section(ui, "Numbers");
        let (sc, truth) = (&self.scales, self.run.pop.truth());
        ui::readout(ui, "samples so far", format!("{} of {}", count(self.run.count() as usize), count(MOST as usize)));
        ui::readout(ui, "true value", num(truth, sc.digits));
        let short = self.contest.short();
        for k in 0..3 {
            let r = &self.run.records[k];
            if !self.defined(k) {
                ui::readout_series(ui, short[k], "needs two values", COLORS[k]);
                continue;
            }
            ui::readout_series(ui, short[k], format!("average {}", num(r.mean, sc.digits)), COLORS[k]);
            ui::readout(ui, "  bias, standard error", format!("{}, {}", num(r.bias(truth), sc.digits), num(r.se(), sc.digits)));
            let d = sc.mse_digits;
            // (plain numbers: a formula in a value that changes in every frame is set again in every frame)
            ui::readout(ui, r"  $\text{bias}^2$, variance, their sum", format!("{}, {}, {}", num(r.bias(truth).powi(2), d), num(r.variance(), d), num(r.mse(), d)));
        }
        if self.contest == Contest::Centre {
            ui::readout(ui, r"the mean's standard error in theory, $\sigma/\sqrt{n}$", num(1.0 / (self.n as f64).sqrt(), sc.digits + 1));
        }
        ui::note(ui, "These are measured over the samples so far, so they wobble too, less and less as the samples pile up; another run (start over) gives slightly different ones.");
        ui::see_also(
            ui,
            &[
                ("clt", "the shape of the mean's sampling distribution"),
                ("likelihood", "a recipe for making estimators"),
                ("intervals", "the standard error turned into a range"),
                ("bootstrap", "the sampling distribution when the population is not known"),
            ],
        );
    }

    fn frame(&mut self, f: &mut Frame) {
        let lay = Layout::of(f.view_size(), f.top_inset);
        if !lay.tiny {
            for (i, c) in Contest::ALL.iter().enumerate() {
                if plot::button(f, lay.buttons[i], c.button(), lay.small, *c == self.contest) && *c != self.contest {
                    self.contest = *c;
                    self.change();
                }
            }
        }
        self.tick(f.dt, f.real_dt);
        if lay.tiny {
            // (a canvas squeezed to a sliver is one the controls have opened over, on a
            // phone: nothing there)
            let view = f.view_size();
            if view.x >= 200.0 && view.y >= 60.0 {
                f.text_px(view * 0.5, "a larger window shows the charts", 12.0, pal::DIM, Align2::CENTER_CENTER);
            }
            return;
        }
        self.draw_population(f, &lay);
        self.draw_words(f, &lay);
        self.draw_distributions(f, &lay);
        self.draw_errors(f, &lay);
    }
}

impl Estimators {
    /// A frame's worth of samples, `scene_dt` seconds of the scene's time, and what is
    /// drawn brought that much nearer to them.
    fn tick(&mut self, scene_dt: f32, real_dt: f32) {
        // (the picture follows the samples in the scene's time; while that stands
        // still, a change made by hand still comes to rest)
        let dt = if scene_dt > 0.0 { scene_dt } else { real_dt };
        if scene_dt > 0.0 {
            self.advance(scene_dt, real_dt);
        }
        for s in &mut self.shown {
            s.age += dt;
            if let Some(l) = &mut s.leaving {
                *l += dt;
            }
        }
        self.shown.retain(|s| s.leaving.is_none_or(|l| l < FADE));
        let fading = self.shown.iter().any(|s| s.age < FADE || s.leaving.is_some());
        let bars = self.bar_goals();
        let marks = self.mark_goals();
        let moving = self.bars.follow(&bars, dt, false) | self.marks.follow(&marks, dt, false);
        self.moving = moving || fading;
    }

    /// The population, the sample shown, and its estimates on the population's own axis.
    fn draw_population(&self, f: &mut Frame, lay: &Layout) {
        let pop = self.run.pop;
        let (lo, hi) = pop.shown_range();
        let (min, max) = lay.pop;
        let height = max.y - min.y;
        let peak = if pop.flat() { 1.0 / (pop.ends().1 - pop.ends().0) } else { (0..=200).map(|i| pop.pdf(lo + (hi - lo) * i as f64 / 200.0)).fold(0.0, f64::max) };
        // below the population's zero: the sample's strip, and a track for each estimate
        let tiers = if self.contest == Contest::Spread { 1 } else { 3 };
        let tier_h = if lay.narrow { 12.0 } else { 15.0 };
        let strip = (height * 0.26).clamp(16.0, 64.0);
        let below = 4.0 + strip + 4.0 + tiers as f32 * tier_h + 2.0;
        let top = peak * 1.1;
        let y_lo = -top * below as f64 / (height - below).max(1.0) as f64;
        let p = Plot::new(lay.pop, [lo, hi], [y_lo, top]);
        p.backing(f, 10.0, lay.small + 12.0, 2.0 * lay.big + 10.0, 10.0);
        let zero = p.px_y(0.0);
        let strip_top = zero + 4.0;
        let tier_y = |k: usize| strip_top + strip + 4.0 + (k as f32 + 0.5) * tier_h;

        // titles
        let titles = match self.contest {
            Contest::Spread => {
                let mu = num(pop.mean(), 0);
                [
                    format!(r"the population: {}, mean $\mu = {mu}$, variance $\sigma^2 = 1$", pop.shape.name()),
                    format!(r"the population: {}, $\mu = {mu}$, $\sigma^2 = 1$", pop.shape.name()),
                    format!(r"{}, $\mu = {mu}$, $\sigma^2 = 1$", pop.shape.name()),
                ]
            }
            Contest::Centre => [format!("the population: {}, centre 0", pop.shape.name()), format!("{}, centre 0", pop.shape.name()), pop.shape.name().to_string()],
            Contest::Top => [r"the population: flat from 0 to $\theta = 10$".to_string(), r"flat from 0 to $\theta = 10$".to_string(), r"$0$ to $\theta = 10$".to_string()],
        };
        let room = max.x - min.x + 8.0;
        let title = titles.iter().find(|t| text_width(t, lay.big) <= room).unwrap_or(&titles[2]);
        f.text_px(Vec2::new(min.x - 4.0, min.y - lay.big - 12.0), title.as_str(), lay.big, pal::TEXT, Align2::LEFT_BOTTOM);
        f.text_px(Vec2::new(min.x - 4.0, min.y - 6.0), format!("and the sample just drawn from it ($n = {}$)", self.n), lay.small, DATA, Align2::LEFT_BOTTOM);

        // the population
        if pop.flat() {
            let (a, b) = pop.ends();
            p.bar(f, a, b, 0.0, peak, 0.0, TRUTH.a(0.12));
            p.path(f, plot::CURVES, [(lo, 0.0), (a, 0.0), (a, peak), (b, peak), (b, 0.0), (hi, 0.0)], 2.0, TRUTH);
        } else {
            p.area(f, lo, hi, 0.0, 160, TRUTH.a(0.12), |x| pop.pdf(x));
            p.curve(f, plot::CURVES, 160, 2.0, TRUTH, |x| pop.pdf(x));
        }
        p.x_axis(f, "", lay.small);
        // the true value
        let (truth_at, truth_name) = match self.contest {
            Contest::Spread => (pop.mean(), r"$\mu$"),
            Contest::Centre => (pop.truth(), "true centre"),
            Contest::Top => (THETA, r"$\theta$"),
        };
        let tx = p.px_x(truth_at);
        plot::dashed_of(f, plot::MARKS, Vec2::new(tx, min.y), Vec2::new(tx, max.y), 1.5, TRUTH.a(0.9));
        f.text_px(Vec2::new(tx + 4.0, min.y + 2.0), truth_name, lay.small, TRUTH, Align2::LEFT_TOP);

        // the tracks of the estimates, each with its name at the left
        let short = self.contest.short();
        let clamp_x = |x: f32| x.max(min.x).min(max.x);
        let along = |v: f32| min.x + (max.x - min.x) * v;
        if self.contest == Contest::Spread {
            let y = tier_y(0);
            plot::line_of(f, plot::MARKS, Vec2::new(min.x, y), Vec2::new(max.x, y), 1.0, pal::DIM.a(0.25));
            f.text_px(Vec2::new(min.x + 2.0, y), r"$\bar x$", lay.small, XBAR, Align2::LEFT_CENTER);
            let x = clamp_x(along(self.marks.get(MEAN_MARK)));
            plot::line_of(f, plot::MARKS, Vec2::new(x, strip_top), Vec2::new(x, y + 4.0), 1.5, XBAR.a(0.8));
            plot::disc(f, Vec2::new(x, y), 7.0, XBAR);
        } else {
            for k in 0..3 {
                let y = tier_y(k);
                plot::line_of(f, plot::MARKS, Vec2::new(min.x, y), Vec2::new(max.x, y), 1.0, COLORS[k].a(0.22));
                f.text_px(Vec2::new(min.x + 2.0, y), short[k], lay.small, COLORS[k], Align2::LEFT_CENTER);
                let x = clamp_x(along(self.marks.get(TIERS + k)));
                plot::line_of(f, plot::MARKS, Vec2::new(x, y - 5.0), Vec2::new(x, y + 5.0), 2.0, COLORS[k]);
                plot::disc(f, Vec2::new(x, y), 7.0, COLORS[k]);
            }
        }

        // the samples shown: dots up the strip, and in the spread's contest, for a
        // sample of up to twenty, each one's distance from the sample's mean
        let mut beyond = 0;
        for s in &self.shown {
            let a = s.alpha();
            let n = s.values.len();
            let mean_x = p.px_x(s.mean);
            for (i, &v) in s.values.iter().enumerate() {
                let x = p.px_x(v);
                let y = strip_top + strip * level(i);
                if s.leaving.is_none() && !(min.x..=max.x).contains(&x) {
                    beyond += 1;
                }
                if self.contest == Contest::Spread && n <= 20 {
                    plot::line_of(f, DISTANCES, Vec2::new(clamp_x(x), y), Vec2::new(clamp_x(mean_x), y), 1.2, XBAR.a(0.45 * a));
                }
                sample_dot(f, Vec2::new(clamp_x(x), y), if n > 60 { 5.0 } else { 7.0 }, DATA.a(a));
            }
        }
        if beyond > 0 {
            f.caption_px(Vec2::new(max.x - 2.0, strip_top + 1.0), format!("{beyond} beyond the edge"), lay.small, DATA, Align2::RIGHT_TOP);
        }
    }

    /// The lines on the sample shown: what each estimator makes of it.
    fn draw_words(&self, f: &mut Frame, lay: &Layout) {
        let Some(s) = self.current() else { return };
        let (min, max) = lay.words;
        let width = max.x - min.x;
        let d = self.scales.digits.max(1);
        let e = |k: usize| s.estimates[k].map_or("needs two values".to_string(), |v| num(v, d));
        let short = self.contest.short();
        let mut y = min.y;
        let size = lay.small;
        match self.contest {
            Contest::Spread => {
                let (here, from_mu) = (num(squares(&s.values, s.mean), 2), num(squares(&s.values, self.run.pop.mean()), 2));
                let first = if lay.narrow {
                    format!(r"This sample: $S = \sum (x_i - \bar x)^2 = {here}$; from $\mu$ instead, ${from_mu}$: never less.")
                } else {
                    format!(r"This sample: $\bar x = {}$, and $S = \sum (x_i - \bar x)^2 = {here}$. Measured from the true mean instead, $\sum (x_i - \mu)^2 = {from_mu}$: never less.", num(s.mean, 2))
                };
                y += wrap_px(f, Vec2::new(min.x, y), width, max.y, &first, size, pal::DIM);
                let mut x = min.x;
                for k in 0..3 {
                    let text = match s.estimates[k] {
                        Some(v) => format!("{} $= {}$", short[k], num(v, d)),
                        None => format!("{}: needs two values", short[k]),
                    };
                    f.text_px(Vec2::new(x, y), text.as_str(), size, COLORS[k], Align2::LEFT_TOP);
                    x += text_width(&text, size) + 16.0;
                }
                y += size + 9.0;
                if self.n < 2 {
                    wrap_px(f, Vec2::new(min.x, y), width, max.y, r"With one value $S$ is always 0: there is no spread to see.", size, pal::DIM);
                    return;
                }
                // (the averages change in every frame: written apart from the formulas,
                // which are then set once)
                let samples = self.run.count().max(1) as f64;
                let (s_avg, mu_avg) = (num(self.run.records[0].mean * self.n as f64, 2), num(self.run.around_mean / samples, 2));
                let lead = if lay.narrow { "Averaged so far:" } else { "Averaged over the samples so far:" };
                let parts = [lead.to_string(), r"$S$".to_string(), format!("{s_avg},"), r"$\sum (x_i - \mu)^2$".to_string(), mu_avg];
                let mut x = min.x;
                for part in &parts {
                    f.text_px(Vec2::new(x, y), part.as_str(), size, pal::DIM, Align2::LEFT_TOP);
                    x += text_width(part, size) + size * 0.35;
                }
                y += size + 5.0;
                let last = if lay.narrow {
                    r"In the long run $S$ falls short by exactly $\sigma^2 = 1$."
                } else {
                    r"In the long run $S$ falls short by exactly $\sigma^2 = 1$, so $S/(n - 1)$ is right on average."
                };
                wrap_px(f, Vec2::new(min.x, y), width, max.y, last, size, pal::DIM);
            }
            Contest::Centre => {
                let cut = trimmed(s.values.len());
                let first = if cut == 0 { "This sample's estimates (below ten values nothing is trimmed):".to_string() } else { format!("This sample's estimates (trimmed: {cut} cut off each end):") };
                y += wrap_px(f, Vec2::new(min.x, y), width, max.y, &first, size, pal::DIM);
                let mut x = min.x;
                for k in 0..3 {
                    let text = format!("{} {}", short[k], e(k));
                    f.text_px(Vec2::new(x, y), text.as_str(), size, COLORS[k], Align2::LEFT_TOP);
                    x += text_width(&text, size) + 16.0;
                }
            }
            Contest::Top => {
                y += wrap_px(f, Vec2::new(min.x, y), width, max.y, r"This sample's estimates of $\theta$:", size, pal::DIM);
                let mut x = min.x;
                for k in 0..3 {
                    let text = format!("{} {}", short[k], e(k));
                    f.text_px(Vec2::new(x, y), text.as_str(), size, COLORS[k], Align2::LEFT_TOP);
                    x += text_width(&text, size) + 16.0;
                }
                y += size + 9.0;
                wrap_px(f, Vec2::new(min.x, y), width, max.y, r"The largest value is never above $\theta$, so it always falls short: on average by $\theta/(n + 1)$, which stretching it by $(n + 1)/n$ makes up.", size, pal::DIM);
            }
        }
    }

    /// The sampling distributions: a histogram for each estimator, on a common axis.
    fn draw_distributions(&self, f: &mut Frame, lay: &Layout) {
        let sc = &self.scales;
        let truth = self.run.pop.truth();
        let (left, right) = (lay.rows[0].0.x, lay.rows[0].1.x);
        let top = lay.rows[0].0.y;
        let axis = Plot::new((Vec2::new(left, top), Vec2::new(right, lay.axis_y)), [sc.lo, sc.hi], [0.0, 1.0]);
        axis.backing(f, 10.0, 2.0 * lay.small + 16.0, lay.big + lay.text + 22.0, 10.0);
        let samples = self.run.count();
        // (the count apart from the formulas: a label with mathematics in it that changes
        // in every frame would be set again in every frame)
        let title = format!("{} from samples of $n = {}$", self.contest.title(), self.n);
        f.text_px(lay.title, title.as_str(), lay.big, pal::TEXT, Align2::LEFT_BOTTOM);
        let so_far = if samples >= MOST { format!("(all {})", count(samples as usize)) } else { format!("({} so far)", count(samples as usize)) };
        f.text_px(lay.title + Vec2::new(text_width(&title, lay.big) + 8.0, 0.0), so_far, lay.small, pal::DIM, Align2::LEFT_BOTTOM);
        // faint lines up from the axis's round values, through every row
        for v in plot::Ticks::of(sc.lo, sc.hi, right - left).values() {
            let x = axis.px_x(v);
            plot::line_of(f, plot::AXES, Vec2::new(x, top), Vec2::new(x, lay.axis_y), 1.0, pal::DIM.a(0.1));
        }
        axis.x_axis(f, self.contest.of(), lay.small);
        // the truth, through every row (under each row's average: an unbiased
        // estimator's average lies on it, and shows)
        let tx = axis.px_x(truth);
        line_down(f, tx, top - 2.0, lay.axis_y);

        let names = self.contest.names();
        let span = sc.hi - sc.lo;
        let at = |v: f64| axis.px_x(sc.lo + v * span);
        let bins = sc.bins;
        let width = sc.width();
        for k in 0..3 {
            let rect = lay.rows[k];
            let (min, max) = rect;
            let p = Plot::new(rect, [sc.lo, sc.hi], [0.0, 1.0]);
            plot::line_of(f, plot::AXES, Vec2::new(min.x, max.y), max, 1.0, pal::DIM.a(0.6));
            f.text_px(Vec2::new(min.x, min.y - 5.0), names[k], lay.text, COLORS[k], Align2::LEFT_BOTTOM);
            if !self.defined(k) {
                f.text_px((min + max) * 0.5, r"needs two values: with one there is no spread to measure, and $S/(n - 1)$ is $0/0$", lay.small, pal::DIM, Align2::CENTER_CENTER);
                continue;
            }
            // the histogram, filled and outlined
            let heights: Vec<f64> = (0..bins).map(|i| self.bars.get(k * bins + i) as f64).collect();
            p.bars(f, sc.lo, width, &heights, 0.5, COLORS[k].a(0.3));
            let mut outline = Vec::with_capacity(2 * bins + 2);
            outline.push((sc.lo, 0.0));
            for (i, h) in heights.iter().enumerate() {
                let h = h.min(1.0);
                outline.push((sc.lo + i as f64 * width, h));
                outline.push((sc.lo + (i + 1) as f64 * width, h));
            }
            outline.push((sc.hi, 0.0));
            p.path(f, OUTLINES, outline, 1.2, COLORS[k].a(0.9));

            // the run before, as an outline
            if let Some(g) = &self.ghost {
                let h = &g.hist[k];
                let scale = width / (sc.peak * HEAD);
                let mut pts = Vec::with_capacity(2 * h.counts.len() + 2);
                pts.push((h.lo, 0.0));
                for i in 0..h.counts.len() {
                    let y = (h.density(i) * scale).min(1.0);
                    pts.push((h.lo + i as f64 * h.width(), y));
                    pts.push((h.lo + (i + 1) as f64 * h.width(), y));
                }
                pts.push((h.hi, 0.0));
                p.path(f, GHOSTS, pts, 1.2, pal::DIM.a(0.85));
                let r = &g.records[k];
                let y = max.y + 13.0;
                let (a, b) = (axis.px_x(r.mean - r.se()).max(min.x), axis.px_x(r.mean + r.se()).min(max.x));
                plot::line_of(f, GHOSTS, Vec2::new(a, y), Vec2::new(b, y), 1.5, pal::DIM.a(0.85));
                if k == 0 {
                    f.text_px(Vec2::new(b + 4.0, y), format!("$n = {}$", g.n), lay.small, pal::DIM, Align2::LEFT_CENTER);
                }
            }

            // the average, the bias from the truth to it, and one standard error either side
            let avg = at(self.mark(k * PER)).max(min.x).min(max.x);
            let se = self.mark(k * PER + 1) as f32 * (max.x - min.x);
            let bias_y = min.y + (max.y - min.y) * 0.3;
            plot::line_of(f, plot::MARKS, Vec2::new(tx, bias_y), Vec2::new(avg, bias_y), 2.5, ALARM);
            plot::line_of(f, plot::MARKS, Vec2::new(avg, min.y + 2.0), Vec2::new(avg, max.y), 2.0, COLORS[k].mix(pal::WHITE, 0.45));
            let y = max.y + 6.0;
            let (a, b) = ((avg - se).max(min.x), (avg + se).min(max.x));
            plot::line_of(f, plot::MARKS, Vec2::new(a, y), Vec2::new(b, y), 2.5, COLORS[k]);
            plot::line_of(f, plot::MARKS, Vec2::new(a, y - 3.5), Vec2::new(a, y + 3.5), 1.5, COLORS[k]);
            plot::line_of(f, plot::MARKS, Vec2::new(b, y - 3.5), Vec2::new(b, y + 3.5), 1.5, COLORS[k]);
            // the sample shown's estimate
            let this = at(self.mark(k * PER + 4)).max(min.x).min(max.x);
            plot::disc(f, Vec2::new(this, max.y), 6.0, pal::WHITE);

            // its numbers, over the row at the right
            let r = &self.run.records[k];
            let d = sc.digits;
            let parts = if lay.narrow {
                vec![(format!("bias {}", num(r.bias(truth), d)), ALARM), (format!("SE {}", num(r.se(), d)), pal::DIM)]
            } else {
                vec![(format!("average {}", num(r.mean, d)), pal::DIM), (format!("bias {}", num(r.bias(truth), d)), ALARM), (format!("standard error {}", num(r.se(), d)), pal::DIM)]
            };
            let mut x = max.x;
            for (text, color) in parts.iter().rev() {
                f.text_px(Vec2::new(x, min.y - 5.0), text.as_str(), lay.small, *color, Align2::RIGHT_BOTTOM);
                x -= text_width(text, lay.small) + 14.0;
            }
        }
        f.caption_px(Vec2::new(tx + 5.0, top + 3.0), format!("true {}", self.truth_words()), lay.small, TRUTH, Align2::LEFT_TOP);
    }

    /// How the true value is written.
    fn truth_words(&self) -> String {
        match self.contest {
            Contest::Spread => r"$\sigma^2 = 1$".to_string(),
            Contest::Centre => "centre 0".to_string(),
            Contest::Top => r"$\theta = 10$".to_string(),
        }
    }

    /// The mean squared errors, each split into bias² and variance.
    fn draw_errors(&self, f: &mut Frame, lay: &Layout) {
        let sc = &self.scales;
        let (min, max) = lay.mse;
        let p = Plot::new(lay.mse, [0.0, sc.mse_top], [0.0, 1.0]);
        p.backing(f, min.x - lay.mse_column.0 + 10.0, lay.small + 14.0, lay.big + 14.0, lay.mse_column.1 - max.x + 10.0);
        f.text_px(lay.mse_title, format!(r"mean squared error $= {} + \text{{variance}}$", tint(ALARM, r"\text{bias}^2")), lay.big, pal::TEXT, Align2::LEFT_BOTTOM);
        for v in plot::Ticks::of(0.0, sc.mse_top, max.x - min.x).values() {
            let x = p.px_x(v);
            plot::line_of(f, plot::AXES, Vec2::new(x, min.y), Vec2::new(x, max.y), 1.0, pal::DIM.a(0.1));
        }
        p.x_axis(f, "", lay.small);
        let short = self.contest.short();
        let row = (max.y - min.y) / 3.0;
        let bar = (row * 0.62).max(4.0);
        let best = self.best();
        let d = sc.mse_digits;
        for k in 0..3 {
            let y = min.y + row * (k as f32 + 0.5);
            f.text_px(Vec2::new(min.x - 8.0, y), short[k], lay.small, COLORS[k], Align2::RIGHT_CENTER);
            if !self.defined(k) {
                f.text_px(Vec2::new(min.x + 6.0, y), "needs two values", lay.small, pal::DIM, Align2::LEFT_CENTER);
                continue;
            }
            let along = |v: f32| (min.x + (max.x - min.x) * v).max(min.x).min(max.x);
            let b2 = self.marks.get(k * PER + 2);
            let var = self.marks.get(k * PER + 3);
            let (x0, x1, x2) = (min.x, along(b2), along(b2 + var));
            let (top, bottom) = (y - bar * 0.5, y + bar * 0.5);
            if x1 > x0 {
                plot::fill(f, (Vec2::new(x0, top), Vec2::new(x1, bottom)), ALARM.a(0.85));
            }
            if x2 > x1 {
                plot::fill(f, (Vec2::new(x1, top), Vec2::new(x2, bottom)), COLORS[k].a(0.7));
            }
            if let Some(g) = &self.ghost {
                let gx = p.px_x(g.records[k].mse()).min(max.x);
                let c = pal::DIM.a(0.85);
                plot::path_of(f, GHOSTS, &[Vec2::new(x0, top - 2.0), Vec2::new(gx, top - 2.0), Vec2::new(gx, bottom + 2.0), Vec2::new(x0, bottom + 2.0)], 1.2, c);
            }
            let r = &self.run.records[k];
            let mut text = num(r.mse(), d);
            if best[k] && self.run.count() >= 2 {
                text += "  least";
            }
            let off = if x2 >= max.x - 1.0 { "  (off the chart)" } else { "" };
            f.text_px(Vec2::new(x2 + 6.0, y), format!("{text}{off}"), lay.small, pal::TEXT, Align2::LEFT_CENTER);
        }
    }
}

/// The truth's line down through the rows.
fn line_down(f: &mut Frame, x: f32, from: f32, to: f32) {
    plot::line_of(f, plot::MARKS, Vec2::new(x, from), Vec2::new(x, to), 1.8, TRUTH);
}

#[cfg(test)]
mod tests {
    use super::*;

    fn pop(contest: Contest, shape: Shape) -> Population {
        Population { contest, shape }
    }

    /// Every population the panel offers.
    fn every() -> Vec<Population> {
        Contest::ALL.iter().flat_map(|&c| c.shapes().iter().map(move |&s| pop(c, s))).collect()
    }

    fn settled(p: Population, n: usize, seed: u64, samples: usize) -> Run {
        let mut run = Run::new(p, n, seed, (0.0, 1.0), 10);
        for _ in 0..samples {
            run.step();
        }
        run
    }

    /// An experiment's run, to its end, as the scene draws it.
    fn finished(k: usize) -> Estimators {
        let mut s = Estimators::create();
        s.set_up(k);
        while s.run.count() < MOST {
            s.run.step();
        }
        s
    }

    /// A number as the picture writes it, to the decimals its axis is written in.
    fn shown(v: f64, digits: usize) -> String {
        num(v, digits)
    }

    /// The mean squared error of S/divisor, exactly, for n independent values from a
    /// population of variance 1 and kurtosis `kappa` (its fourth moment over σ⁴): S
    /// averages n − 1, and S/(n − 1) has variance κ/n − (n − 3)/(n(n − 1)).
    fn exact_spread_mse(kappa: f64, n: f64, divisor: f64) -> f64 {
        let var_s = (n - 1.0) * (n - 1.0) * (kappa / n - (n - 3.0) / (n * (n - 1.0)));
        let bias = (n - 1.0) / divisor - 1.0;
        bias * bias + var_s / (divisor * divisor)
    }

    /// The captions of the "things to try", held to account: each experiment run to its
    /// end, as the scene runs it, and read as the picture writes its numbers.
    #[test]
    fn the_experiments_show_what_their_captions_say() {
        // dividing by n comes out low: 0.80; by n − 1, 1.00; by n + 1 the least error
        let s = finished(0);
        let (r, sc) = (&s.run.records, &s.scales);
        assert_eq!((sc.digits, sc.mse_digits), (2, 2));
        assert_eq!([shown(r[0].mean, 2), shown(r[1].mean, 2)], ["0.80", "1.00"]);
        assert_eq!([shown(r[2].mse(), 2), shown(r[0].mse(), 2), shown(r[1].mse(), 2)], ["0.33", "0.36", "0.50"]);
        assert!(r[2].bias(1.0) < r[0].bias(1.0) && r[0].bias(1.0) < -0.15 && s.best() == [false, false, true]);

        // a bell: the median's standard error 0.25, the mean's 0.20, both unbiased
        let s = finished(1);
        let r = &s.run.records;
        assert_eq!(s.scales.digits, 2);
        assert_eq!([shown(r[0].se(), 2), shown(r[1].se(), 2), shown(r[0].bias(0.0), 2), shown(r[1].bias(0.0), 2)], ["0.20", "0.25", "0.00", "0.00"]);
        let ratio = r[1].se() / r[0].se();
        assert!((1.2..1.3).contains(&ratio), "median against mean: {ratio}");

        // heavy tails: the mean's 0.20 again, the median's 0.16, the trimmed mean's 0.15
        let s = finished(2);
        let r = &s.run.records;
        assert_eq!([shown(r[0].se(), 2), shown(r[1].se(), 2), shown(r[2].se(), 2)], ["0.20", "0.16", "0.15"]);

        // the top of a range: this run measures errors 3.33, 1.55 and 0.86, the largest
        // value 0.9 low; in the long run (exactly, for n values spread evenly from 0 to
        // θ) the errors are θ²/(3n), 2θ²/((n + 1)(n + 2)) and θ²/(n(n + 2)): 3.33, 1.52
        // and 0.83, the last exactly a quarter of the first at n = 10 (and the largest
        // value θ/(n + 1) low)
        let s = finished(3);
        let (r, sc) = (&s.run.records, &s.scales);
        assert_eq!((sc.digits, sc.mse_digits), (1, 2));
        assert_eq!([shown(r[0].mse(), 2), shown(r[1].mse(), 2), shown(r[2].mse(), 2), shown(r[1].bias(THETA), 1)], ["3.33", "1.55", "0.86", "−0.9"]);
        let n = 10.0;
        let exact = [THETA * THETA / (3.0 * n), 2.0 * THETA * THETA / ((n + 1.0) * (n + 2.0)), THETA * THETA / (n * (n + 2.0))];
        assert_eq!(exact.map(|e| shown(e, 2)), ["3.33", "1.52", "0.83"]);
        assert!((exact[2] / exact[0] - 0.25).abs() < 1e-12 && shown(-THETA / (n + 1.0), 1) == "−0.9");
        for (k, e) in exact.iter().enumerate() {
            assert!((r[k].mse() / e - 1.0).abs() < 0.04, "estimator {k}: {} measured against {e}", r[k].mse());
        }
        let quarter = r[2].mse() / r[0].mse();
        assert!((0.23..0.28).contains(&quarter), "corrected against twice the mean: {quarter}");
        assert!(r[0].bias(THETA).abs() < 0.05 && r[2].bias(THETA).abs() < 0.05);
        // the largest value is never above θ (no estimate in a bin above it)
        let h = &s.run.hist[1];
        assert!(h.counts.iter().enumerate().all(|(i, &c)| c == 0 || h.centre(i) < THETA), "{:?}", h.counts);

        // four times the data: every standard error halves, every error quarters
        let s = finished(4);
        let (r, g) = (&s.run.records, &s.ghost.as_ref().expect("the run of 25 to compare with").records);
        assert_eq!(s.ghost.as_ref().map(|g| g.n), Some(25));
        assert_eq!([shown(r[0].se(), 2), shown(g[0].se(), 2)], ["0.10", "0.20"]);
        for k in 0..3 {
            let (half, quarter) = (r[k].se() / g[k].se(), r[k].mse() / g[k].mse());
            assert!((0.47..0.53).contains(&half) && (0.22..0.28).contains(&quarter), "{k}: {half}, {quarter}");
        }
    }

    /// The mean squared error is the bias squared and the variance, to rounding, for
    /// every estimator, population and size of sample, the degenerate ones included.
    #[test]
    fn the_error_is_bias_squared_and_variance() {
        for p in every() {
            for n in [1usize, 2, 5, 30] {
                let run = settled(p, n, 5, 2000);
                for (k, r) in run.records.iter().enumerate() {
                    if r.count == 0 {
                        // (only dividing by n − 1, with one value)
                        assert!(p.contest == Contest::Spread && k == 1 && n == 1);
                        continue;
                    }
                    let (b, v, m) = (r.bias(p.truth()), r.variance(), r.mse());
                    assert!(m.is_finite() && (m - (b * b + v)).abs() <= 1e-9 * (1.0 + m), "{p:?} n {n} {k}: {m} against {b}² + {v}");
                }
            }
        }
    }

    /// The mean's standard error is σ/√n, whatever the shape (each has σ = 1); twice
    /// the mean's, for the flat range from 0 to θ, is 2σ/√n = θ/√(3n).
    #[test]
    fn the_standard_error_of_the_mean_is_sigma_over_root_n() {
        for shape in [Shape::Normal, Shape::Flat, Shape::Heavy] {
            for (n, seed) in [(1usize, 1u64), (4, 2), (25, 3), (100, 4)] {
                let run = settled(pop(Contest::Centre, shape), n, seed, 20_000);
                let (se, want) = (run.records[0].se(), 1.0 / (n as f64).sqrt());
                // (heavy tails make the measured spread itself settle slowly)
                let tolerance = if shape == Shape::Heavy { 0.05 } else { 0.02 };
                assert!((se / want - 1.0).abs() < tolerance, "{shape:?} n {n}: {se} against {want}");
            }
        }
        let run = settled(pop(Contest::Top, Shape::Flat), 10, 6, 20_000);
        let want = THETA / 30f64.sqrt();
        assert!((run.records[0].se() / want - 1.0).abs() < 0.02, "{} against {want}", run.records[0].se());
    }

    /// Dividing by n − 1 is right on average for every shape, dividing by n is low by
    /// the factor (n − 1)/n; and the reason, sample by sample: the squared distances
    /// from the true mean are those from the sample's mean plus n(x̄ − μ)².
    #[test]
    fn dividing_by_n_minus_one_is_unbiased_for_every_shape() {
        for shape in [Shape::Normal, Shape::Flat, Shape::Skewed] {
            let p = pop(Contest::Spread, shape);
            for n in [2usize, 5, 20] {
                let mut run = settled(p, n, 8, 100_000);
                let r = &run.records;
                assert!((r[1].mean - 1.0).abs() < 0.02, "{shape:?} n {n}: S/(n − 1) averages {}", r[1].mean);
                let low = (n - 1) as f64 / n as f64;
                assert!((r[0].mean / low - 1.0).abs() < 0.02, "{shape:?} n {n}: S/n averages {} against {low}", r[0].mean);
                // S falls short of the sum around μ by σ² = 1 on average
                let short = run.around_mean / run.count() as f64 - r[0].mean * n as f64;
                assert!((short - 1.0).abs() < 0.05, "{shape:?} n {n}: short by {short}");
                for _ in 0..20 {
                    run.step();
                    let (m, mu) = (st::mean(&run.sample), p.mean());
                    let (around, here) = (squares(&run.sample, mu), squares(&run.sample, m));
                    assert!((around - here - n as f64 * (m - mu) * (m - mu)).abs() < 1e-9 * (1.0 + around));
                    assert!(here <= around + 1e-12);
                }
            }
        }
    }

    /// Which divisor has the least error depends on the tails (the panel's note on the
    /// populations): worked out exactly, and measured over many samples.
    #[test]
    fn the_least_error_depends_on_the_tails() {
        // the kurtosis of each shape: 3, 1.8 and 9
        for (shape, kappa) in [(Shape::Normal, 3.0), (Shape::Flat, 1.8), (Shape::Skewed, 9.0)] {
            let exact = [5.0, 4.0, 6.0].map(|d| exact_spread_mse(kappa, 5.0, d));
            let run = settled(pop(Contest::Spread, shape), 5, 9, 200_000);
            for k in 0..3 {
                let m = run.records[k].mse();
                assert!((m / exact[k] - 1.0).abs() < 0.03, "{shape:?} {k}: measured {m}, exactly {}", exact[k]);
            }
            let best = (0..3).min_by(|&a, &b| exact[a].total_cmp(&exact[b])).unwrap_or(0);
            match shape {
                Shape::Normal => assert!(best == 2 && shown(exact[2], 2) == "0.33"),
                Shape::Skewed => assert!(best == 2 && shown(exact[2], 2) == "0.87"),
                _ => assert!(best == 0 && shown(exact[0], 2) == "0.21" && shown(exact[2], 2) == "0.23"),
            }
        }
    }

    /// The largest of n values spread evenly from 0 to θ falls short of θ by θ/(n + 1)
    /// on average, and stretched by (n + 1)/n it does not (the note under the sample).
    #[test]
    fn the_largest_value_falls_short_by_theta_over_n_plus_one() {
        for (n, seed) in [(1usize, 1u64), (5, 2), (20, 3)] {
            let run = settled(pop(Contest::Top, Shape::Flat), n, seed, 40_000);
            let (short, want) = (-run.records[1].bias(THETA), THETA / (n + 1) as f64);
            assert!((short / want - 1.0).abs() < 0.02, "n {n}: short by {short}, against {want}");
            // (within four of its own standard errors over these samples: θ/√(n(n + 2)) each)
            let noise = THETA / ((n * (n + 2)) as f64).sqrt() / 200.0;
            assert!(run.records[2].bias(THETA).abs() < 4.0 * noise, "n {n}: {} against {noise}", run.records[2].bias(THETA));
        }
    }

    /// With one value there is no spread: dividing by n − 1 gives nothing (0/0), and
    /// the others always give 0, off by the whole variance.
    #[test]
    fn one_value_has_no_spread_to_divide() {
        assert_eq!(estimates(Contest::Spread, &[0.7]), [Some(0.0), None, Some(0.0)]);
        let mut s = Estimators::create();
        s.n = 1;
        s.change();
        for _ in 0..600 {
            s.tick(1.0 / 60.0, 1.0);
        }
        let r = &s.run.records;
        assert!(r[1].count == 0 && !s.defined(1) && s.run.count() > 100);
        assert!(r[0].mse() == 1.0 && r[2].mse() == 1.0 && r[0].variance() == 0.0);
        assert!(s.mark_goals().iter().chain(&s.bar_goals()).all(|v| v.is_finite()));
        // and one value is its own mean, median and trimmed mean
        assert_eq!(estimates(Contest::Centre, &[0.7]), [Some(0.7); 3]);
        assert_eq!(estimates(Contest::Top, &[4.0]), [Some(8.0), Some(4.0), Some(8.0)]);
    }

    /// The trimmed mean cuts a tenth of the values off each end, rounded down.
    #[test]
    fn trimming_cuts_a_tenth_rounded_down() {
        assert_eq!([1, 2, 5, 9, 10, 25, 99, 100, 200].map(trimmed), [0, 0, 0, 0, 1, 2, 9, 10, 20]);
        let v: Vec<f64> = (1..=10).map(|i| i as f64).collect();
        let mut wild = v.clone();
        wild[9] = 1000.0;
        // one wild value of ten moves the mean a long way, the trimmed mean not at all
        assert_eq!(estimates(Contest::Centre, &v)[2], estimates(Contest::Centre, &wild)[2]);
        assert!(estimates(Contest::Centre, &wild)[0].unwrap_or(0.0) > 100.0);
    }

    /// Every chart's scale is a drawable one that holds the truth and nearly all of the
    /// estimates, for every population and size of sample.
    #[test]
    fn the_charts_hold_what_they_show() {
        for p in every() {
            for n in [1usize, 2, 3, 5, 10, 50, 200] {
                let sc = Scales::of(p, n, 11, None);
                let t = p.truth();
                assert!(sc.lo.is_finite() && sc.hi.is_finite() && sc.lo < t && t < sc.hi, "{p:?} n {n}: {sc:?}");
                assert!((8..=80).contains(&sc.bins) && sc.peak > 0.0 && sc.peak <= 1.0 && sc.mse_top > 0.0 && sc.mse_top.is_finite(), "{p:?} n {n}: {sc:?}");
                let mut run = Run::new(p, n, 12, (sc.lo, sc.hi), sc.bins);
                for _ in 0..2000 {
                    run.step();
                }
                for (k, h) in run.hist.iter().enumerate() {
                    let inside: u32 = h.counts.iter().sum();
                    let out = h.total - inside as u64;
                    assert!(out as f64 <= 0.012 * h.total as f64, "{p:?} n {n} {k}: {out} of {} off the axis", h.total);
                    // the tallest bar fits its row, with a little to spare
                    let tallest = h.counts.iter().copied().max().unwrap_or(0) as f64 / sc.row_holds(run.count());
                    assert!(tallest < 1.05, "{p:?} n {n} {k}: tallest bar {tallest} of a row");
                }
            }
        }
    }

    /// The samples come one at a time at first, then fast, and stop at the end of the
    /// run; the sample shown changes no more than five times a second however fast
    /// they come; and once all is drawn the scene comes to rest.
    #[test]
    fn samples_come_slowly_then_fast_and_the_scene_comes_to_rest() {
        for hz in [60.0f32, 144.0] {
            let mut s = Estimators::create();
            let dt = 1.0 / hz;
            let frames = |s: &mut Estimators, seconds: f32| {
                let mut changes = 0;
                for _ in 0..(seconds * hz) as usize {
                    let before = s.current().map(|c| c.values.clone());
                    s.tick(dt, 1.0);
                    changes += (s.current().map(|c| c.values.clone()) != before) as usize;
                }
                changes
            };
            frames(&mut s, 1.0);
            let first = s.run.count();
            assert!((3..=10).contains(&first), "{hz} Hz: {first} samples in the first second");
            let shown = frames(&mut s, 4.0);
            assert!(s.run.count() > 2000 && shown <= 4 * SHOW_RATE as usize + 2, "{hz} Hz: {} samples, {shown} shown", s.run.count());
            frames(&mut s, 12.0);
            assert_eq!(s.run.count(), MOST);
            frames(&mut s, 1.0);
            assert!(!s.animated(), "{hz} Hz: still moving after the run");
        }
    }

    /// A change of n alone keeps the run before as an outline; a change of anything
    /// else, or new samples, does not.
    #[test]
    fn a_new_n_keeps_the_last_run_to_compare_with() {
        let mut s = Estimators::create();
        while s.run.count() < GHOST_MIN {
            s.run.step();
        }
        s.n = 20;
        s.change();
        assert_eq!(s.ghost.as_ref().map(|g| g.n), Some(5));
        // a run of too few samples leaves the outline as it was
        s.n = 40;
        s.change();
        assert_eq!(s.ghost.as_ref().map(|g| g.n), Some(5));
        // back to n = 5: nothing to compare
        s.n = 5;
        s.change();
        assert!(s.ghost.is_none());
        s.set_up(4);
        assert!(s.ghost.is_some());
        s.contest = Contest::Top;
        s.change();
        assert!(s.ghost.is_none());
    }

    /// Every window gets a layout of finite numbers; one with room for the charts gets
    /// them inside it, one under another in order, and the columns apart.
    #[test]
    fn the_layout_fits_every_window() {
        let sizes = [(1075.0, 1000.0, 0.0), (1600.0, 1000.0, 0.0), (578.0, 700.0, 0.0), (800.0, 600.0, 46.0), (390.0, 800.0, 46.0), (390.0, 600.0, 46.0), (240.0, 180.0, 46.0), (200.0, 150.0, 46.0), (3000.0, 400.0, 0.0), (1.0, 1.0, 0.0)];
        for (x, y, inset) in sizes {
            let view = Vec2::new(x, y);
            let lay = Layout::of(view, inset);
            let rects = [lay.pop, lay.words, lay.rows[0], lay.rows[1], lay.rows[2], lay.mse];
            let mut numbers = vec![lay.big, lay.text, lay.small, lay.title.x, lay.title.y, lay.axis_y, lay.mse_title.x, lay.mse_title.y, lay.mse_column.0, lay.mse_column.1];
            for (a, b) in rects.iter().chain(&lay.buttons) {
                numbers.extend([a.x, a.y, b.x, b.y]);
            }
            assert!(numbers.iter().all(|v| v.is_finite()), "{x} × {y}");
            if lay.tiny {
                continue;
            }
            for (a, b) in rects.iter().chain(&lay.buttons) {
                assert!(a.x >= 0.0 && a.y >= 0.0 && b.x <= x && b.y <= y && a.x <= b.x && a.y <= b.y, "{x} × {y}: {a} to {b}");
            }
            assert!(lay.rows[0].1.y < lay.rows[1].0.y && lay.rows[1].1.y < lay.rows[2].0.y && lay.rows[2].1.y < lay.axis_y, "{x} × {y}");
            assert!(lay.pop.1.y < lay.words.0.y && lay.words.1.y <= lay.mse_title.y, "{x} × {y}");
            if lay.narrow {
                assert!(lay.words.1.y < lay.title.y && lay.axis_y < lay.mse_title.y, "{x} × {y}");
            } else {
                assert!(lay.pop.1.x + 30.0 < lay.rows[0].0.x && lay.mse_column.1 < lay.rows[0].0.x, "{x} × {y}");
            }
        }
        // a desktop and a phone have room for the charts
        assert!(!Layout::of(Vec2::new(1075.0, 1000.0), 0.0).tiny && !Layout::of(Vec2::new(390.0, 800.0), 46.0).tiny);
    }

    /// The captions, names and notes the scene shows are markup that sets.
    #[test]
    fn the_words_are_markup_that_sets() {
        let mut texts: Vec<&str> = TRIALS.iter().flat_map(|t| [t.button, t.caption]).collect();
        for c in Contest::ALL {
            texts.extend(c.names());
            texts.extend(c.short());
            texts.extend([c.button(), c.of(), c.title()]);
        }
        for text in texts {
            let found = crate::typeset::inspect(text);
            assert!(found.error.is_none() && found.words.is_empty(), "{text}: {found:?}");
        }
    }

    /// How typical each seed's experiments come out: the largest distance of what they
    /// show from the theory, in units of what the picture's rounding hides (run with
    /// --ignored --nocapture).
    #[test]
    #[ignore]
    fn seeds() {
        for seed in 1..=40u64 {
            let mut worst = 0.0f64;
            let mut line = format!("seed {seed:<3}");
            for k in 0..TRIALS.len() {
                let mut s = Estimators::create();
                s.set_up(k);
                s.seed = seed;
                s.ghost = None;
                if let Some(n) = TRIALS[k].before {
                    let pop = s.pop();
                    let sc = Scales::of(pop, n, seed, None);
                    let mut before = Run::new(pop, n, seed, (sc.lo, sc.hi), sc.bins);
                    while before.count() < BEFORE {
                        before.step();
                    }
                    s.ghost = Some(Ghost::of(&before));
                }
                s.restart();
                while s.run.count() < MOST {
                    s.run.step();
                }
                let t = s.run.pop.truth();
                let r = &s.run.records;
                // (what theory says each shows, and how much rounding hides)
                let checks: Vec<(f64, f64, f64)> = match k {
                    0 => vec![(r[0].mean, 0.8, 0.005), (r[1].mean, 1.0, 0.005), (r[2].mean, 2.0 / 3.0, 0.005), (r[0].mse(), 0.36, 0.005), (r[1].mse(), 0.5, 0.005), (r[2].mse(), 1.0 / 3.0, 0.005)],
                    1 => vec![(r[0].bias(t), 0.0, 0.005), (r[1].bias(t), 0.0, 0.005), (r[0].se(), 0.2, 0.005), (r[1].se(), 0.2486, 0.005)],
                    2 => vec![(r[0].bias(t), 0.0, 0.005), (r[1].bias(t), 0.0, 0.005), (r[0].se(), 0.2, 0.005), (r[1].se(), 0.1588, 0.005)],
                    3 => vec![(r[0].bias(t), 0.0, 0.05), (r[2].bias(t), 0.0, 0.05), (r[0].mse(), 10.0 / 3.0, 0.05), (r[1].mse(), 1.515, 0.05), (r[2].mse(), 0.833, 0.05)],
                    _ => vec![(r[0].bias(t), 0.0, 0.005), (r[0].se(), 0.1, 0.005), (r[1].se(), 0.1245, 0.005), s.ghost.as_ref().map_or((0.0, 0.0, 1.0), |g| (g.records[0].se(), 0.2, 0.005))],
                };
                let w = checks.iter().map(|(v, want, unit)| (v - want).abs() / unit).fold(0.0, f64::max);
                line += &format!("  {w:.2}");
                worst = worst.max(w);
            }
            println!("{line}   worst {worst:.2}");
        }
    }

    /// The numbers behind the captions (run with --ignored --nocapture).
    #[test]
    #[ignore]
    fn explore() {
        for (k, t) in TRIALS.iter().enumerate() {
            let s = finished(k);
            let truth = s.run.pop.truth();
            println!("{}", t.button);
            for (j, r) in s.run.records.iter().enumerate() {
                println!("   {j}: avg {:.4} bias {:+.4} se {:.4} var {:.4} mse {:.4}", r.mean, r.bias(truth), r.se(), r.variance(), r.mse());
            }
            if let Some(g) = &s.ghost {
                for (j, r) in g.records.iter().enumerate() {
                    println!("   before {j}: avg {:.4} se {:.4} mse {:.4}", r.mean, r.se(), r.mse());
                }
            }
            println!("   scales {:?}", s.scales);
        }
        let line = |name: &str, p: Population, n: usize, seed: u64, samples: usize| {
            let run = settled(p, n, seed, samples);
            let t = p.truth();
            let mut s = format!("{name:<20} n {n:<4} seed {seed:<3}");
            for r in &run.records {
                s += &format!(" | avg {:.4} bias {:+.4} se {:.4} mse {:.4}", r.mean, r.bias(t), r.se(), r.mse());
            }
            println!("{s}");
        };
        for seed in [FIRST_SEED, 7] {
            line("spread flat", pop(Contest::Spread, Shape::Flat), 5, seed, 5000);
            line("spread skewed", pop(Contest::Spread, Shape::Skewed), 5, seed, 5000);
        }
    }
}

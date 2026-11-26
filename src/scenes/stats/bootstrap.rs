//! The bootstrap: how far a statistic would wander from sample to sample, found from
//! one sample alone by drawing from it with replacement.
//!
//! Three charts, one above another. The population (known here, so that the method can
//! be checked) and one sample from it; one resample being made, each draw hopping down
//! from the point it copies; and the statistic of every resample so far, with its
//! middle 95 %, over the statistic of fresh samples from the population.

use super::plot::{self, Glide, Plot, Rect, Ticks, AXES, CURVES, MARKS, OWN};
use super::*;
use crate::math::stats::{self as st, Dist, Histogram};

pub const DEF: SceneDef = SceneDef {
    id: "bootstrap",
    title: "The bootstrap",
    group: ESTIMATION,
    blurb: "Resample the sample: the spread of any statistic, and an interval for it, with no formula.",
    make: |_| Box::new(Bootstrap::create()),
};

// ----- the populations and the statistics ----------------------------------------------

#[derive(Clone, Copy, Debug, PartialEq)]
enum Pop {
    Bell,
    Tail,
    Flat,
}

impl Pop {
    const ALL: [Pop; 3] = [Pop::Bell, Pop::Tail, Pop::Flat];

    fn name(self) -> &'static str {
        match self {
            Pop::Bell => "bell",
            Pop::Tail => "long right tail",
            Pop::Flat => "flat",
        }
    }

    fn dist(self) -> Dist {
        match self {
            Pop::Bell => Dist::STANDARD_NORMAL,
            Pop::Tail => Dist::Exponential { rate: 1.0 },
            Pop::Flat => Dist::Uniform { lo: 0.0, hi: 1.0 },
        }
    }

    /// The values the chart of the data shows, before the sample widens it.
    fn reach(self) -> [f64; 2] {
        match self {
            Pop::Bell => [-3.2, 3.2],
            Pop::Tail => [-0.15, 5.0],
            Pop::Flat => [-0.08, 1.08],
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
enum Stat {
    Mean,
    Median,
    Sd,
    P90,
    Max,
    Corr,
}

impl Stat {
    const ALL: [Stat; 6] = [Stat::Mean, Stat::Median, Stat::Sd, Stat::P90, Stat::Max, Stat::Corr];

    /// Its name in a sentence ("the sample's ...").
    fn name(self) -> &'static str {
        match self {
            Stat::Mean => "mean",
            Stat::Median => "median",
            Stat::Sd => "standard deviation",
            Stat::P90 => "90th percentile",
            Stat::Max => "largest value",
            Stat::Corr => "correlation",
        }
    }

    /// Its name on a button.
    fn button(self) -> &'static str {
        match self {
            Stat::Mean => "mean",
            Stat::Median => "median",
            Stat::Sd => "standard deviation",
            Stat::P90 => "90th percentile",
            Stat::Max => "largest value",
            Stat::Corr => "correlation $r$",
        }
    }

    /// The sample's value of it, and the population's, written beside their marks.
    fn own_text(self, v: f64, digits: usize, long: bool) -> String {
        let name = if long { self.name() } else { self.short() };
        match self {
            Stat::Sd => format!(r"the sample's $s = {}$", num(v, digits)),
            Stat::Corr => format!(r"the sample's $r = {}$", num(v, digits)),
            _ => format!("the sample's {name} {}", num(v, digits)),
        }
    }

    fn true_text(self, v: f64, digits: usize) -> String {
        match self {
            Stat::Sd => format!(r"true $\sigma = {}$", num(v, digits)),
            Stat::Corr => format!(r"true $\rho = {}$", num(v, digits)),
            _ => format!("true {} {}", self.short(), num(v, digits)),
        }
    }

    /// Its name where there is little room.
    fn short(self) -> &'static str {
        match self {
            Stat::Mean => "mean",
            Stat::Median => "median",
            Stat::Sd => "sd",
            Stat::P90 => "90th percentile",
            Stat::Max => "largest",
            Stat::Corr => "$r$",
        }
    }
}

/// Everything the sample and its bootstrap depend on, but the seed.
#[derive(Clone, Copy, Debug, PartialEq)]
struct Setup {
    pop: Pop,
    stat: Stat,
    n: usize,
    /// The true correlation of the pairs (for the correlation only).
    rho: f64,
}

impl Setup {
    fn pairs(&self) -> bool {
        self.stat == Stat::Corr
    }

    /// The same with what does not matter to it set to one value: the pairs are always
    /// from the bell, and the correlation matters only to them.
    fn plain(self) -> Setup {
        let rho = if self.pairs() { self.rho.max(-0.99).min(0.99) } else { 0.0 };
        let pop = if self.pairs() { Pop::Bell } else { self.pop };
        Setup { pop, rho, n: self.n.clamp(1, N_MOST), ..self }
    }

    /// The statistic's value in the population, where it has one.
    fn true_value(&self) -> Option<f64> {
        let d = self.pop.dist();
        match self.stat {
            Stat::Mean => d.mean(),
            Stat::Median => Some(d.quantile(0.5)),
            Stat::Sd => d.sd(),
            Stat::P90 => Some(d.quantile(0.9)),
            Stat::Max => Some(d.support().1).filter(|v| v.is_finite()),
            Stat::Corr => Some(self.rho),
        }
    }

    /// Is every resample's value one of the sample's own values? (The largest value
    /// always; a median or percentile when it falls on one of the sorted values rather
    /// than between two.)
    fn on_values(&self) -> bool {
        let at = |p: f64| p * (self.n.max(1) - 1) as f64;
        match self.stat {
            Stat::Max => true,
            Stat::Median => at(0.5).fract() == 0.0,
            Stat::P90 => at(0.9).fract().abs() < 1e-9,
            _ => false,
        }
    }
}

/// The largest sample there may be.
const N_MOST: usize = 200;

/// One draw from the population: a value, or for the correlation a pair from the bell
/// in two dimensions with correlation `rho`.
fn draw_one(s: &Setup, rng: &mut Rng) -> (f64, f64) {
    if s.pairs() {
        let (a, b) = (st::normal(rng), st::normal(rng));
        (a, s.rho * a + (1.0 - s.rho * s.rho).max(0.0).sqrt() * b)
    } else {
        (s.pop.dist().sample(rng), 0.0)
    }
}

/// A sample of `s.n` from the population (`y` is empty but for pairs).
fn draw_sample(s: &Setup, rng: &mut Rng, x: &mut Vec<f64>, y: &mut Vec<f64>) {
    x.clear();
    y.clear();
    for _ in 0..s.n {
        let (a, b) = draw_one(s, rng);
        x.push(a);
        if s.pairs() {
            y.push(b);
        }
    }
}

/// The value a share p of the way up the numbers, as `st::quantile_sorted` gives it of
/// them sorted (the numbers are reordered, not sorted: this is the cost of one pass).
fn quantile_of(v: &mut [f64], p: f64) -> f64 {
    if v.is_empty() || p.is_nan() {
        return f64::NAN;
    }
    let at = p.max(0.0).min(1.0) * (v.len() - 1) as f64;
    let i = (at.floor() as usize).min(v.len() - 1);
    let (_, lo, rest) = v.select_nth_unstable_by(i, f64::total_cmp);
    let lo = *lo;
    let share = at - i as f64;
    if share == 0.0 || rest.is_empty() {
        return lo;
    }
    let hi = rest.iter().copied().fold(f64::INFINITY, f64::min);
    lo + (hi - lo) * share
}

/// The statistic of a sample (`x`, and `y` for pairs), or None where it has none: a
/// standard deviation of one value, a correlation with a column that does not vary.
/// (`x` is reordered.)
fn statistic(stat: Stat, x: &mut [f64], y: &[f64]) -> Option<f64> {
    if x.is_empty() {
        return None;
    }
    let v = match stat {
        Stat::Mean => st::mean(x),
        Stat::Median => quantile_of(x, 0.5),
        Stat::Sd => {
            if x.len() < 2 {
                return None;
            }
            st::sd(x)
        }
        Stat::P90 => quantile_of(x, 0.9),
        Stat::Max => x.iter().copied().fold(f64::NEG_INFINITY, f64::max),
        Stat::Corr => {
            let n = x.len().min(y.len());
            // (a column that does not vary has no correlation with anything)
            if n < 2 || st::variance(&x[..n]) <= 0.0 || st::variance(&y[..n]) <= 0.0 {
                return None;
            }
            st::correlation(&x[..n], &y[..n])
        }
    };
    v.is_finite().then_some(v)
}

/// A resample: `n` positions in the sample, each drawn from all of them with the same
/// chance, again and again (with replacement).
fn draw_positions(n: usize, rng: &mut Rng, out: &mut Vec<usize>) {
    out.clear();
    out.extend((0..n).map(|_| rng.below(n)));
}

/// Scratch room for working out a resample's statistic.
#[derive(Clone, Debug, Default)]
struct Scratch {
    at: Vec<usize>,
    x: Vec<f64>,
    y: Vec<f64>,
    seen: Vec<bool>,
}

impl Scratch {
    /// The statistic of the resample at the positions `self.at` of the sample.
    fn statistic(&mut self, stat: Stat, x: &[f64], y: &[f64]) -> Option<f64> {
        self.x.clear();
        self.y.clear();
        self.x.extend(self.at.iter().filter_map(|&i| x.get(i)));
        if stat == Stat::Corr {
            self.y.extend(self.at.iter().filter_map(|&i| y.get(i)));
        }
        statistic(stat, &mut self.x, &self.y)
    }

    /// How many of the sample's `n` points the resample has at least once.
    fn distinct(&mut self, n: usize) -> usize {
        self.seen.clear();
        self.seen.resize(n, false);
        let mut k = 0;
        for &i in &self.at {
            if let Some(s) = self.seen.get_mut(i) {
                if !*s {
                    *s = true;
                    k += 1;
                }
            }
        }
        k
    }
}

/// A number into a sorted list, keeping it sorted.
fn insert_sorted(v: &mut Vec<f64>, x: f64) {
    let at = v.partition_point(|a| *a < x);
    v.insert(at, x);
}

/// The percentile interval: from the 2.5th to the 97.5th percentile of the resampled
/// values (sorted).
fn percentile_interval(sorted: &[f64]) -> Option<(f64, f64)> {
    (!sorted.is_empty()).then(|| (st::quantile_sorted(sorted, 0.025), st::quantile_sorted(sorted, 0.975)))
}

// ----- the sample ----------------------------------------------------------------------

/// How many resamples a bootstrap makes, and how many fresh samples show the truth.
const MOST: usize = 10_000;
const FRESH: usize = 10_000;
/// Fresh samples and resamples made at once when the settings change, to choose the
/// range the resamples' values are counted over.
const PILOT: usize = 200;
/// Bins of that range.
const BINS: usize = 60;
/// How far above the fresh samples' peak the chart of the resamples reaches.
const CAP: f32 = 4.0;

/// The seeds of what is drawn from a sample's seed.
const RESAMPLES: u64 = 0xB007_5742;
const PILOT_RESAMPLES: u64 = 0x9170_7E57;
const FRESH_SEED: u64 = 0x7E57;
const TALLY_SEED: u64 = 0x7A11_0000;

/// The sample, sorted by its first column, and what it shows.
#[derive(Clone, Debug)]
struct Sample {
    key: (Setup, u64),
    x: Vec<f64>,
    y: Vec<f64>,
    /// Its statistic.
    own: Option<f64>,
    /// The range over which the resamples' values are counted, and the axes of the
    /// data's charts (across, and up for pairs).
    range: [f64; 2],
    across: [f64; 2],
    up: [f64; 2],
}

impl Sample {
    fn new(s: Setup, seed: u64, fresh: &[f64]) -> Sample {
        let (mut x, mut y) = (Vec::new(), Vec::new());
        draw_sample(&s, &mut Rng::new(seed), &mut x, &mut y);
        // (sorted, so that a new sample's points glide to theirs in order)
        let mut order: Vec<usize> = (0..x.len()).collect();
        order.sort_by(|&a, &b| x[a].total_cmp(&x[b]));
        let x: Vec<f64> = order.iter().map(|&i| x[i]).collect();
        let y: Vec<f64> = if s.pairs() { order.iter().map(|&i| y[i]).collect() } else { Vec::new() };
        let own = statistic(s.stat, &mut x.clone(), &y);
        // the range: what a few hundred resamples and fresh samples reach, and a margin
        let mut pilot = Scratch::default();
        let mut rng = Rng::new(seed ^ PILOT_RESAMPLES);
        let mut values: Vec<f64> = Vec::with_capacity(PILOT + fresh.len() + 2);
        for _ in 0..PILOT {
            draw_positions(x.len(), &mut rng, &mut pilot.at);
            values.extend(pilot.statistic(s.stat, &x, &y));
        }
        values.extend(fresh.iter().copied().take(PILOT));
        values.extend(own);
        values.extend(s.true_value());
        let (lo, hi) = values.iter().fold((f64::INFINITY, f64::NEG_INFINITY), |(a, b), v| (a.min(*v), b.max(*v)));
        let range = if lo.is_finite() && hi > lo + 1e-9 {
            let pad = (hi - lo) * 0.18;
            [lo - pad, hi + pad]
        } else if lo.is_finite() {
            // every value the same: a range round it the size of the data's
            let half = 0.1 * (s.pop.reach()[1] - s.pop.reach()[0]);
            [lo - half, lo + half]
        } else {
            [0.0, 1.0]
        };
        let widen = |reach: [f64; 2], v: &[f64]| {
            let (lo, hi) = v.iter().fold((reach[0], reach[1]), |(a, b), v| (a.min(*v), b.max(*v)));
            let pad = if lo < reach[0] || hi > reach[1] { 0.04 * (hi - lo) } else { 0.0 };
            [lo - if lo < reach[0] { pad } else { 0.0 }, hi + if hi > reach[1] { pad } else { 0.0 }]
        };
        let reach = if s.pairs() { [-3.2, 3.2] } else { s.pop.reach() };
        let across = widen(reach, &x);
        let up = if s.pairs() { widen(reach, &y) } else { [0.0, 1.0] };
        Sample { key: (s, seed), x, y, own, range, across, up }
    }

    fn n(&self) -> usize {
        self.x.len()
    }
}

// ----- the bootstrap -------------------------------------------------------------------

/// The resamples made so far, and what they come to.
#[derive(Clone, Debug)]
struct Boot {
    rng: Rng,
    /// Their values, sorted (those that have one).
    sorted: Vec<f64>,
    /// The same, counted in the bins of the sample's range.
    hist: Histogram,
    made: usize,
    /// Resamples whose statistic is not defined.
    undefined: usize,
    /// The sum over the resamples of the share of the sample's points each has.
    appear: f64,
    /// Resamples that have the sample's largest value (its last point, as sorted).
    with_top: usize,
    /// The value of the resample being shown made, kept out of the rest until it is
    /// seen to land among them (it is counted in `made` already).
    held: Option<f64>,
    scratch: Scratch,
}

impl Boot {
    fn new(seed: u64, range: [f64; 2]) -> Boot {
        Boot {
            rng: Rng::new(seed ^ RESAMPLES),
            sorted: Vec::with_capacity(MOST),
            hist: Histogram::new(range[0], range[1], BINS),
            made: 0,
            undefined: 0,
            appear: 0.0,
            with_top: 0,
            held: None,
            scratch: Scratch::default(),
        }
    }

    fn done(&self) -> bool {
        self.made >= MOST
    }

    /// One resample: its statistic and how many of the sample's points it has (its
    /// positions stay in `self.scratch.at`). With `hold`, its value is held back from
    /// the others until `release`.
    fn step(&mut self, s: &Setup, sample: &Sample, hold: bool) -> (Option<f64>, usize) {
        let n = sample.n();
        draw_positions(n, &mut self.rng, &mut self.scratch.at);
        let k = self.scratch.distinct(n);
        let value = self.scratch.statistic(s.stat, &sample.x, &sample.y);
        self.made += 1;
        self.appear += k as f64 / n.max(1) as f64;
        if n > 0 && self.scratch.seen.get(n - 1) == Some(&true) {
            self.with_top += 1;
        }
        match value {
            Some(v) if hold => self.held = Some(v),
            Some(v) => {
                insert_sorted(&mut self.sorted, v);
                self.hist.add(v);
            }
            None => self.undefined += 1,
        }
        (value, k)
    }

    /// The value held back joins the others.
    fn release(&mut self) {
        if let Some(v) = self.held.take() {
            insert_sorted(&mut self.sorted, v);
            self.hist.add(v);
        }
    }

    /// The resamples whose values are among the others.
    fn counted(&self) -> usize {
        self.made - self.held.is_some() as usize
    }

    fn interval(&self) -> Option<(f64, f64)> {
        percentile_interval(&self.sorted)
    }

    /// The standard deviation of the resamples' values: the bootstrap's standard error.
    fn se(&self) -> Option<f64> {
        (self.sorted.len() >= 2).then(|| st::sd(&self.sorted))
    }
}

/// The statistic of fresh samples from the population: its true sampling distribution,
/// which only a known population allows.
#[derive(Clone, Debug)]
struct Fresh {
    key: Option<Setup>,
    rng: Rng,
    values: Vec<f64>,
    undefined: usize,
    made: usize,
    /// Its values sorted, once all are made, and counted in the bins of a range.
    sorted: Vec<f64>,
    hist: Option<([f64; 2], Histogram)>,
    x: Vec<f64>,
    y: Vec<f64>,
}

impl Fresh {
    fn new() -> Fresh {
        Fresh { key: None, rng: Rng::new(FRESH_SEED), values: Vec::new(), undefined: 0, made: 0, sorted: Vec::new(), hist: None, x: Vec::new(), y: Vec::new() }
    }

    /// Start again for a new setup, and make the first few at once.
    fn restart(&mut self, s: Setup) {
        *self = Fresh { key: Some(s), values: Vec::with_capacity(FRESH), ..Fresh::new() };
        for _ in 0..PILOT {
            self.step(&s);
        }
    }

    fn done(&self) -> bool {
        self.made >= FRESH
    }

    fn step(&mut self, s: &Setup) {
        if self.done() {
            return;
        }
        draw_sample(s, &mut self.rng, &mut self.x, &mut self.y);
        match statistic(s.stat, &mut self.x, &self.y) {
            Some(v) => self.values.push(v),
            None => self.undefined += 1,
        }
        self.made += 1;
        if self.done() {
            self.sorted = st::sorted(&self.values);
        }
    }

    /// Its standard deviation, once all are made.
    fn se(&self) -> Option<f64> {
        (self.done() && self.sorted.len() >= 2).then(|| st::sd(&self.sorted))
    }

    /// Its counts in the bins of `range`, once all are made.
    fn hist(&mut self, range: [f64; 2]) -> Option<&Histogram> {
        if !self.done() {
            return None;
        }
        if self.hist.as_ref().is_none_or(|(r, _)| *r != range) {
            self.hist = Some((range, Histogram::of(&self.values, range[0], range[1], BINS)));
        }
        self.hist.as_ref().map(|(_, h)| h)
    }

    /// The share of fresh samples whose statistic is above `v`.
    fn above(&self, v: f64) -> Option<f64> {
        (self.done() && !self.sorted.is_empty()).then(|| (self.sorted.len() - self.sorted.partition_point(|a| *a <= v)) as f64 / self.sorted.len() as f64)
    }
}

/// How often the percentile interval covers the truth, counted over new samples.
#[derive(Clone, Debug)]
struct Tally {
    key: Option<Setup>,
    done: usize,
    covered: usize,
    /// Intervals wholly below the truth, and wholly above it.
    low: usize,
    high: usize,
    /// Samples whose statistic had no value, or whose resamples had none.
    undefined: usize,
    scratch: Scratch,
    values: Vec<f64>,
}

/// New samples counted, and resamples made of each.
const TALLY: usize = 1000;
const TALLY_RESAMPLES: usize = 1000;
/// New samples a second of the scene's time.
const TALLY_RATE: f32 = 100.0;

impl Tally {
    fn new() -> Tally {
        Tally { key: None, done: 0, covered: 0, low: 0, high: 0, undefined: 0, scratch: Scratch::default(), values: Vec::new() }
    }

    fn restart(&mut self, s: Setup) {
        *self = Tally { key: Some(s), ..Tally::new() };
    }

    fn pending(&self) -> bool {
        self.key.is_some() && self.done < TALLY
    }

    /// One new sample, its bootstrap, and whether its interval covers the truth.
    fn step(&mut self) {
        let Some(s) = self.key else { return };
        let Some(truth) = s.true_value() else { return };
        let mut rng = Rng::new(TALLY_SEED.wrapping_add(self.done as u64));
        let (mut x, mut y) = (Vec::new(), Vec::new());
        draw_sample(&s, &mut rng, &mut x, &mut y);
        self.values.clear();
        for _ in 0..TALLY_RESAMPLES {
            draw_positions(x.len(), &mut rng, &mut self.scratch.at);
            if let Some(v) = self.scratch.statistic(s.stat, &x, &y) {
                self.values.push(v);
            }
        }
        self.values.sort_by(f64::total_cmp);
        self.done += 1;
        match percentile_interval(&self.values) {
            Some((lo, _)) if truth < lo => self.low += 1,
            Some((_, hi)) if truth > hi => self.high += 1,
            Some(_) => self.covered += 1,
            None => self.undefined += 1,
        }
    }
}

// ----- the resample shown being made ---------------------------------------------------

/// The resample the middle chart shows being made: one in every `every` made (one every
/// `SHOW` seconds or more), so that its draws can be followed whatever the rate. (Its
/// value is held back from the chart below until it is seen to land there: see
/// `Boot::held`.)
#[derive(Clone, Debug)]
struct Shown {
    /// The sample's points it copies, in the order drawn, and its statistic.
    at: Vec<usize>,
    value: Option<f64>,
    /// The bar it falls into.
    bin: Option<usize>,
    every: usize,
    /// Resamples made since it.
    since: usize,
    /// Which of two batches its dots are drawn in.
    parity: bool,
}

/// The least time over which a resample is shown being made (seconds).
const SHOW: f32 = 2.0;

// When things happen while a resample is shown, as shares of its time: its draws start
// over the first `SPREAD`, each takes `HOP` to land, its statistic is marked at `MARKED`
// and falls into the chart below from `FALL` to `LAND`, and the last one's dots fade
// over the first `FADE` of the next.
const SPREAD: f32 = 0.40;
const HOP: f32 = 0.16;
const MARKED: f32 = 0.58;
const FALL: f32 = 0.62;
const LAND: f32 = 0.82;
const FADE: f32 = 0.08;

fn smooth(t: f32) -> f32 {
    let t = t.max(0.0).min(1.0);
    t * t * (3.0 - 2.0 * t)
}

/// How far draw `j` of `n` has got at share `p` of its resample's time (0 before it
/// starts, 1 once it has landed).
fn hop(j: usize, n: usize, p: f32) -> f32 {
    let start = SPREAD * j as f32 / n.saturating_sub(1).max(1) as f32;
    ((p - start) / HOP).max(0.0).min(1.0)
}

/// Resamples in one shown at this rate: so that one is shown at least every `SHOW` seconds.
fn every_at(rate: f32) -> usize {
    ((rate.max(1e-3) * SHOW).ceil() as usize).max(1)
}

// ----- the experiments -----------------------------------------------------------------

struct Trial {
    name: &'static str,
    look_for: &'static str,
    pop: Pop,
    stat: Stat,
    n: usize,
    rho: f64,
    seed: u64,
    rate: f32,
    count: bool,
}

const TRIALS: [Trial; 6] = [
    Trial {
        name: "the mean",
        look_for: r"The mean of 30 values from a bell: a statistic with a formula to check against. The resamples' means spread by 0.177; the formula $s/\sqrt{n}$ gives 0.180 from the same sample, and fresh samples' means (green) spread by 0.180. The bootstrap has found the standard error without the formula (a shade low, by $\sqrt{29/30}$: it treats the sample as the whole population, whose variance divides by $n$, not $n - 1$). Its bars are centred on the sample's mean, $-0.13$, not on the true 0: they show how far the mean wanders, not where the truth is.",
        pop: Pop::Bell,
        stat: Stat::Mean,
        n: 30,
        rho: 0.0,
        seed: 3,
        rate: 500.0,
        count: false,
    },
    Trial {
        name: "the median",
        look_for: r#"The median of 31 values from a long right tail. No simple formula gives its standard error from a sample. The resamples' medians spread by 0.217, fresh samples' (green) by 0.180: a fifth too much. The bars are spikes: with an odd number of values, a resample's median is always one of the sample's own values (the cyan ticks). For a median the bootstrap is rough: over 2,000 samples like this one it came out too high about as often as too low, but off by more than a third in 3 of 10, against fewer than 1 in 100 for the mean of 30. Press "new sample" a few times and watch it jump."#,
        pop: Pop::Tail,
        stat: Stat::Median,
        n: 31,
        rho: 0.0,
        seed: 3,
        rate: 500.0,
        count: false,
    },
    Trial {
        name: "who is in a resample",
        look_for: r"One resample at a time, slowly. Each of its 12 draws copies a point of the sample picked at random from all 12 (the dot it hops down from), so some points are copied twice or more (stacked) and some not at all (left dim above). On average a resample holds $1 - (1 - 1/12)^{12}$, or 64.8 %, of the points (for large $n$, $1 - 1/e$, or 63.2 %): raise the rate, and the average among the numbers settles there.",
        pop: Pop::Bell,
        stat: Stat::Mean,
        n: 12,
        rho: 0.0,
        seed: 3,
        rate: 0.3,
        count: false,
    },
    Trial {
        name: "the largest value",
        look_for: r"The bootstrap fails. A resample's largest value is the sample's largest, 1.88, whenever that point is drawn, which happens in 63.6 % of resamples, and it can never be larger: a spike at 1.88, smaller ones at the next values down, and the interval ending at 1.88. Yet fresh samples' largest values (green) spread to both sides of it, larger than 1.88 in 60.2 % of them, and nearly twice as widely (0.487 against 0.254). A resample can only reuse the sample's values, and the largest value hangs on the one point at the edge.",
        pop: Pop::Bell,
        stat: Stat::Max,
        n: 30,
        rho: 0.0,
        seed: 2,
        rate: 500.0,
        count: false,
    },
    Trial {
        name: "too little data",
        look_for: r"Eight values. Counted over 1,000 new samples (each interval made from 1,000 resamples), the 95 % interval holds the true mean in only 881. It is too narrow: the resamples' spread divides by $n$ where $s$ divides by $n - 1$, and nothing allows for $s$ being itself a guess, as Student's $t$ does; on average it is 0.77 times as wide as the $t$ interval, which holds the truth in 95 %. Even this sample, whose $s = 0.98$ is typical, gives 0.327 where fresh samples' means (green) spread by 0.352. With 30 values the count is 929: closer, still short.",
        pop: Pop::Bell,
        stat: Stat::Mean,
        n: 8,
        rho: 0.0,
        seed: 13,
        rate: 500.0,
        count: true,
    },
    Trial {
        name: "a correlation",
        look_for: r"30 pairs whose true correlation is 0.9; this sample's $r$ is 0.910. The resamples' $r$ pile up against 1, which $r$ can never pass, and trail off below: the interval runs from 0.812 to 0.954, 0.098 below $r$ and only 0.044 above. Fresh samples' $r$ (green) are lopsided the same way: their middle 95 % runs from 0.805 to 0.952. The percentile interval follows the shape; $r$ plus or minus two standard errors would not.",
        pop: Pop::Bell,
        stat: Stat::Corr,
        n: 30,
        rho: 0.9,
        seed: 1,
        rate: 500.0,
        count: false,
    },
];

/// The scene opens as the first experiment sets it up.
const OPENING: usize = 0;

// ----- the scene -----------------------------------------------------------------------

pub struct Bootstrap {
    setup: Setup,
    seed: u64,
    /// Resamples a second.
    rate: f32,
    show_truth: bool,
    /// Count how often the interval covers the truth over new samples.
    counting: bool,
    sample: Sample,
    boot: Boot,
    fresh: Fresh,
    tally: Tally,
    pace: Pace,
    phase: Phase,
    tally_pace: Pace,
    shown: Option<Shown>,
    before: Option<Shown>,
    /// How far through its time the shown resample is (see `Shown`).
    progress: f32,
    look: Look,
    /// The canvas last drawn on (a new one is drawn at once, without gliding).
    canvas: Option<Vec2>,
}

/// What the picture glides through, following what was worked out.
#[derive(Clone, Debug, Default)]
struct Look {
    /// The data's axes across and up, the resamples' axis, and the top of their chart.
    ranges: Glide,
    /// Each point of the sample: across, and up (pairs) or its level in its pile.
    points: Glide,
    /// The heights of the bars, as densities, and the same of the fresh samples.
    bars: Glide,
    fresh: Glide,
    /// The sample's statistic, the true value, the interval's ends; and how much the
    /// truth is shown (0 to 1).
    marks: Glide,
    moving: bool,
}

/// What the look is going to.
struct Goals {
    ranges: Vec<f32>,
    points: Vec<f32>,
    bars: Vec<f32>,
    fresh: Vec<f32>,
    marks: Vec<f32>,
}

impl Look {
    fn follow(&mut self, g: &Goals, dt: f32, snap: bool) {
        if snap {
            self.ranges.snap(&g.ranges);
            self.points.snap(&g.points);
            self.bars.snap(&g.bars);
            self.fresh.snap(&g.fresh);
            self.marks.snap(&g.marks);
            self.moving = false;
            return;
        }
        let mut moving = self.ranges.follow(&g.ranges, dt, false);
        moving |= self.points.follow(&g.points, dt, false);
        moving |= self.bars.follow(&g.bars, dt, false);
        moving |= self.fresh.follow(&g.fresh, dt, false);
        moving |= self.marks.follow(&g.marks, dt, false);
        self.moving = moving;
    }
}

impl Bootstrap {
    /// (Nothing here needs the graphics device, so the tests can make one too.)
    fn create() -> Bootstrap {
        let t = &TRIALS[OPENING];
        let setup = Setup { pop: t.pop, stat: t.stat, n: t.n, rho: t.rho };
        let mut fresh = Fresh::new();
        fresh.restart(setup.plain());
        let sample = Sample::new(setup.plain(), t.seed, &fresh.values);
        let boot = Boot::new(t.seed, sample.range);
        let mut s = Bootstrap {
            setup,
            seed: t.seed,
            rate: t.rate,
            show_truth: true,
            counting: t.count,
            sample,
            boot,
            fresh,
            tally: Tally::new(),
            pace: Pace::default(),
            phase: Phase::default(),
            tally_pace: Pace::default(),
            shown: None,
            before: None,
            progress: 0.0,
            look: Look::default(),
            canvas: None,
        };
        s.restart();
        s
    }

    /// Set up one of the experiments in `TRIALS`.
    fn set_up(&mut self, k: usize) {
        let Some(t) = TRIALS.get(k) else { return };
        self.setup = Setup { pop: t.pop, stat: t.stat, n: t.n, rho: t.rho };
        self.seed = t.seed;
        self.rate = t.rate;
        self.show_truth = true;
        self.counting = t.count;
        self.refresh();
    }

    /// Make the sample and the fresh samples again if what they depend on has changed,
    /// and start the resamples (and the count, if it is kept) afresh.
    fn refresh(&mut self) {
        self.setup.n = self.setup.n.clamp(1, N_MOST);
        self.rate = self.rate.max(0.2).min(2000.0);
        let s = self.setup.plain();
        if self.fresh.key != Some(s) {
            self.fresh.restart(s);
        }
        if self.sample.key != (s, self.seed) {
            self.sample = Sample::new(s, self.seed, &self.fresh.values);
            self.restart();
        }
        if !self.counting || s.true_value().is_none() {
            self.counting = false;
            self.tally.key = None;
        } else if self.tally.key != Some(s) {
            self.tally.restart(s);
            self.tally_pace.reset();
        }
        // a slower rate shortens the resample being shown to what it now allows
        if let Some(shown) = &mut self.shown {
            shown.every = shown.every.min(every_at(self.rate).max(shown.since + 1));
        }
    }

    /// Start the resamples afresh, the first made at once (at a slow rate the first
    /// would otherwise keep the middle chart empty for a whole turn).
    fn restart(&mut self) {
        self.boot = Boot::new(self.seed, self.sample.range);
        self.pace.reset();
        self.phase.reset();
        (self.shown, self.before, self.progress) = (None, None, 0.0);
        self.step();
    }

    /// One resample, and whether it is the next to be shown.
    fn step(&mut self) {
        let s = self.setup.plain();
        let starts = self.shown.as_ref().is_none_or(|sh| sh.since + 1 >= sh.every);
        if starts {
            self.boot.release();
        }
        let (value, _) = self.boot.step(&s, &self.sample, starts);
        if starts {
            let parity = self.shown.as_ref().is_some_and(|sh| !sh.parity);
            let bin = value.and_then(|v| self.boot.hist.bin(v));
            let shown = Shown { at: self.boot.scratch.at.clone(), value, bin, every: every_at(self.rate), since: 0, parity };
            self.before = self.shown.replace(shown);
        } else if let Some(sh) = &mut self.shown {
            sh.since += 1;
        }
    }

    /// The work a frame owes: resamples at the rate asked for, in the scene's time; the
    /// fresh samples and the count within an allowance of the frame's time (by the
    /// clock, so that they go on while the scene's time stands still: they answer a
    /// change of setting).
    fn work(&mut self, dt: f32, real_dt: f32) {
        let s = self.setup.plain();
        let budget = crate::clock::Budget::new(0.0015);
        while !self.fresh.done() && !budget.spent() {
            self.fresh.step(&s);
        }
        if dt > 0.0 {
            if !self.boot.done() {
                let mut turn = scene_turn(&mut self.pace, self.rate, dt, real_dt);
                while !self.boot.done() && turn.next() {
                    self.step();
                }
                let taken = self.pace.end(turn);
                let share = self.phase.after(self.rate, dt, taken);
                if let Some(sh) = &self.shown {
                    self.progress = ((sh.since as f32 + share) / sh.every as f32).min(1.0);
                }
            } else if let Some(sh) = &self.shown {
                // (the last one shown goes on to its end by the clock)
                self.progress = (self.progress + dt * self.rate / sh.every as f32).min(1.0);
            }
            if self.progress >= LAND {
                self.boot.release();
            }
        }
        if self.tally.pending() {
            let rate_dt = if dt > 0.0 { dt } else { real_dt };
            let mut turn = self.tally_pace.begin(TALLY_RATE, rate_dt, real_dt, 0.003);
            while self.tally.pending() && turn.next() {
                self.tally.step();
            }
            self.tally_pace.end(turn);
        }
    }

    /// Everything to the end at once (the tests of this file).
    #[cfg(test)]
    fn settle(&mut self) {
        self.refresh();
        let s = self.setup.plain();
        while !self.fresh.done() {
            self.fresh.step(&s);
        }
        while !self.boot.done() {
            self.step();
        }
        self.boot.release();
        while self.tally.pending() {
            self.tally.step();
        }
    }

    /// The heights of the bars as densities.
    fn bar_heights(&self) -> Vec<f32> {
        (0..BINS).map(|i| self.boot.hist.density(i) as f32).collect()
    }

    /// What the picture glides to, on this layout.
    fn goals(&mut self, lay: &Layout) -> Goals {
        let s = self.setup.plain();
        let sample = &self.sample;
        let pairs = s.pairs();
        let bars = self.bar_heights();
        let range = sample.range;
        let fresh: Vec<f32> = match self.fresh.hist(range) {
            Some(h) => (0..BINS).map(|i| h.density(i) as f32).collect(),
            None => vec![0.0; BINS],
        };
        let shown = if self.show_truth && self.fresh.done() { 1.0 } else { 0.0 };
        let tallest = bars.iter().chain(fresh.iter().filter(|_| shown > 0.0)).fold(0.0f32, |m, v| m.max(*v));
        // (a spike far above the fresh samples' peak runs off the top, its share
        // written beside it: drawn whole, it would flatten everything else)
        let peak = fresh.iter().fold(0.0f32, |m, v| m.max(*v));
        let tallest = if peak > 0.0 { tallest.min(CAP * peak) } else { tallest };
        let top = if tallest > 0.0 { tallest / 0.74 } else { 1.0 };
        // the data's axis up: the population's density (with room above its peak), or for pairs the second column
        let up = if pairs {
            sample.up
        } else {
            let d = s.pop.dist();
            let peak = (0..=120).map(|i| d.pdf(sample.across[0] + (sample.across[1] - sample.across[0]) * i as f64 / 120.0)).fold(0.0f64, f64::max);
            [0.0, if peak > 0.0 { peak / 0.8 } else { 1.0 }]
        };
        let ranges = vec![sample.across[0] as f32, sample.across[1] as f32, up[0] as f32, up[1] as f32, range[0] as f32, range[1] as f32, top];
        // the points: across, and up or their level in a pile
        let mut points = Vec::with_capacity(2 * sample.n());
        if pairs {
            for (a, b) in sample.x.iter().zip(&sample.y) {
                points.push(*a as f32);
                points.push(*b as f32);
            }
        } else {
            let px: Vec<f32> = sample.x.iter().map(|v| lay.across(0, sample.across, *v)).collect();
            let levels = stack(&px, lay.dot + 1.0);
            for (a, l) in sample.x.iter().zip(levels) {
                points.push(*a as f32);
                points.push(l as f32);
            }
        }
        let mid = 0.5 * (range[0] + range[1]);
        let own = sample.own.unwrap_or(mid);
        let truth = s.true_value().filter(|v| v.is_finite());
        let (lo, hi) = self.boot.interval().unwrap_or((own, own));
        let marks = vec![own as f32, truth.unwrap_or(mid) as f32, lo as f32, hi as f32, if self.show_truth { 1.0 } else { 0.0 }, shown];
        Goals { ranges, points, bars, fresh, marks }
    }
}

/// Levels for dots at these places across (pixels) so that none covers another: each
/// on the lowest level where it is at least `d` from every dot placed before it.
fn stack(xs: &[f32], d: f32) -> Vec<u16> {
    let d = d.max(0.5);
    let lo = xs.iter().copied().filter(|x| x.is_finite()).fold(f32::INFINITY, f32::min);
    let cell = |x: f32| if x.is_finite() && lo.is_finite() { ((x - lo) / d).max(0.0).min(1e5) as usize } else { 0 };
    let mut cells: Vec<Vec<(f32, u16)>> = Vec::new();
    let mut out = Vec::with_capacity(xs.len());
    for &x in xs {
        let c = cell(x);
        if cells.len() < c + 2 {
            cells.resize(c + 2, Vec::new());
        }
        let near = |level: u16, cells: &Vec<Vec<(f32, u16)>>| {
            (c.saturating_sub(1)..=c + 1).any(|k| cells.get(k).is_some_and(|v| v.iter().any(|&(p, l)| l == level && (p - x).abs() < d)))
        };
        let mut level = 0u16;
        while level < 1000 && near(level, &cells) {
            level += 1;
        }
        cells[c].push((x, level));
        out.push(level);
    }
    out
}

// ----- the layout ----------------------------------------------------------------------

/// Where everything goes on a canvas.
#[derive(Clone, Copy, Debug, PartialEq)]
struct Layout {
    view: Vec2,
    narrow: bool,
    /// Text sizes: titles, words, small print.
    big: f32,
    text: f32,
    small: f32,
    /// The size of a dot.
    dot: f32,
    /// The three charts: the population and its sample, one resample, all the resamples.
    rows: [Rect; 3],
    /// For pairs, the two scatters (in the first two rows).
    scatter: [Rect; 2],
    /// The numbers: beside the third chart, or under it.
    numbers: Rect,
    beside: bool,
    /// The right-hand end of the buttons over the picture, and their top.
    buttons: Vec2,
    /// Too small a canvas for the charts.
    compact: bool,
}

impl Layout {
    fn of(view: Vec2, inset: f32, pairs: bool, n: usize) -> Layout {
        let view = view.max(Vec2::splat(1.0));
        let narrow = view.x < 640.0;
        let (big, text, small) = if narrow { (13.0, 11.5, 10.5) } else { (14.5, 12.5, 11.5) };
        let side = if narrow { 10.0 } else { 22.0 };
        let top = inset.max(0.0) + if narrow { 8.0 } else { 14.0 };
        // above each chart its title, below it its numbers
        let title = big + 14.0;
        let axis = small + 14.0;
        let gap = if narrow { 6.0 } else { 12.0 };
        let beside = view.x >= 900.0;
        let numbers_h = if beside { 0.0 } else { 6.0 * (small + 5.0) + 8.0 };
        let left = side + 12.0;
        let right = view.x - side;
        let room = (view.y - top - 3.0 * (title + axis) - 2.0 * gap - numbers_h - small - 12.0).max(0.0);
        let shares = if pairs { [0.32, 0.32, 0.36] } else { [0.30, 0.22, 0.48] };
        let mut rows = [(Vec2::ZERO, Vec2::ZERO); 3];
        let mut y = top + title;
        for (k, share) in shares.iter().enumerate() {
            let h = room * share;
            let r = if k == 2 && beside { right - 290.0 } else { right };
            rows[k] = (Vec2::new(left, y), Vec2::new(r.max(left + 1.0), y + h));
            y += h + axis + gap + title;
        }
        let numbers = if beside {
            (Vec2::new(rows[2].1.x + 34.0, rows[2].0.y - title + 4.0), Vec2::new(right, rows[2].1.y + axis))
        } else {
            (Vec2::new(side, rows[2].1.y + axis + small + 8.0), Vec2::new(right, view.y - 6.0))
        };
        // a scatter is at most half as wide again as it is high, and the two are the
        // same size, one above the other, so that a resample's draws fall straight down
        let h = (rows[0].1.y - rows[0].0.y).min(rows[1].1.y - rows[1].0.y);
        let w = (rows[0].1.x - rows[0].0.x).min(1.5 * h).max(1.0);
        let scatter = [0, 1].map(|k| (rows[k].0, rows[k].0 + Vec2::new(w, h)));
        // dots shrink a little for a large sample
        let base = if narrow { 6.5 } else { 8.0 };
        let dot = base * (30.0 / n.max(1) as f32).sqrt().max(0.62).min(1.0);
        let compact = rows.iter().any(|(a, b)| b.y - a.y < 26.0 || b.x - a.x < 120.0);
        Layout { view, narrow, big, text, small, dot, rows, scatter, numbers, beside, buttons: Vec2::new(right, top - 2.0), compact }
    }

    /// Where the value `v` of the data is across the first two charts, for the axis `across`.
    fn across(&self, row: usize, across: [f64; 2], v: f64) -> f32 {
        let (min, max) = self.rows[row.min(2)];
        min.x + (max.x - min.x) * ((v - across[0]) / (across[1] - across[0]).max(1e-12)) as f32
    }
}

// ----- drawing -------------------------------------------------------------------------

// Point batches of their own (see `dot_of`), and line batches beyond the kit's.
const D_SAMPLE: u8 = 1;
const D_HOPS: u8 = 2;
const D_FALL: u8 = 4;
const L_FRESH: u8 = OWN;
const L_VALUES: u8 = OWN + 1;
const L_ELLIPSE: u8 = OWN + 2;
/// (and `OWN + 4`: a resample's mark is drawn in the batch of its parity, as its dots are)
const L_SHOWN: u8 = OWN + 3;

/// An invisible mark in every batch the scene draws in, first thing in every frame.
/// The draw list lets go of a batch that was empty in the last frame, and the batches
/// after it then take its place in the list: `eidos motion`, which follows a batch's
/// points by its place, would compare one batch with another.
fn anchors(f: &mut Frame) {
    for kind in [AXES, CURVES, MARKS, plot::BUTTONS, L_FRESH, L_VALUES, L_ELLIPSE, L_SHOWN, L_SHOWN + 1] {
        plot::line_of(f, kind, Vec2::ZERO, Vec2::new(0.0, 0.01), 1.0, Color::CLEAR);
    }
    plot::dashed_of(f, MARKS, Vec2::ZERO, Vec2::new(0.0, 0.01), 1.0, Color::CLEAR);
    for kind in [D_SAMPLE, D_HOPS, D_HOPS + 1, D_FALL] {
        dot_of(f, kind, Vec2::ZERO, 1.0, Color::CLEAR);
    }
}

/// A dot in a point batch of its own kind, on top of everything: a list of dots whose
/// number changes from frame to frame goes in one of these (they are told apart by a
/// hair of their size, as line batches are by `plot::apart`).
fn dot_of(f: &mut Frame, kind: u8, at: Vec2, size: f32, color: Color) {
    if at.is_finite() && size.is_finite() {
        let p = f.hud(at);
        f.dl.points_with(PointKind::Overlay, PointStyle { outline_px: 0.0, size_scale: 1.0 + kind as f32 * 1e-4, ..Default::default() }).push(p, size, color);
    }
}

/// A title in two parts of different colours, side by side: the colours name what the
/// chart under it shows. (A part may be empty.)
type Title = [(String, Color); 2];

/// How wide a title is drawn.
fn title_width(f: &Frame, t: &Title, size: f32) -> f32 {
    let gap = if t[1].0.is_empty() { 0.0 } else { 0.3 * size };
    t.iter().filter(|p| !p.0.is_empty()).map(|p| f.text_size(&p.0, size).x).sum::<f32>() + gap
}

/// The first of `options` that fits in `room` pixels at `size` (the last if none does),
/// written from `at` (its left end, at the bottom of the line). With `widest`, the
/// choice is made by those, the same options at their widest: a title whose count
/// grows then keeps its form, rather than changing it as the count gains a digit.
fn title(f: &mut Frame, at: Vec2, options: &[Title], widest: Option<&[Title]>, room: f32, size: f32) {
    let measured = widest.unwrap_or(options);
    let k = measured.iter().position(|t| title_width(f, t, size) <= room).unwrap_or(measured.len().saturating_sub(1));
    let Some(t) = options.get(k).or(options.last()) else { return };
    let first = f.text_size(&t[0].0, size).x;
    f.text_px(at, t[0].0.as_str(), size, t[0].1, Align2::LEFT_BOTTOM);
    f.text_px(at + Vec2::new(first + 0.3 * size, 0.0), t[1].0.as_str(), size, t[1].1, Align2::LEFT_BOTTOM);
}

fn part(text: impl Into<String>, color: Color) -> (String, Color) {
    (text.into(), color)
}

/// Decimals enough to tell values apart that differ by a tenth of `scale`.
fn decimals_for(scale: f64) -> usize {
    if !(scale.is_finite() && scale > 0.0) {
        return 2;
    }
    (1.0 - scale.log10().floor()).max(0.0).min(4.0) as usize
}

fn percent(share: f64) -> String {
    format!("{}\u{a0}%", num(100.0 * share, 1))
}

impl Bootstrap {
    /// The drawn charts: the data's first two (across, up) and the resamples'.
    fn plots(&self, lay: &Layout) -> [Plot; 3] {
        let r = &self.look.ranges;
        let g = |i: usize| r.get(i) as f64;
        let pairs = self.setup.pairs();
        let data = |k: usize| if pairs { lay.scatter[k] } else { lay.rows[k] };
        [Plot::new(data(0), [g(0), g(1)], [g(2), g(3)]), Plot::new(data(1), [g(0), g(1)], [g(2), g(3)]), Plot::new(lay.rows[2], [g(4), g(5)], [0.0, g(6).max(1e-9)])]
    }

    /// Where the sample's point `i` is drawn in the first chart.
    fn point_at(&self, p: &Plot, lay: &Layout, i: usize) -> Vec2 {
        let (a, b) = (self.look.points.get(2 * i) as f64, self.look.points.get(2 * i + 1));
        if self.setup.pairs() {
            p.at(a, b as f64)
        } else {
            Vec2::new(p.px_x(a), p.rect.1.y - 0.5 * lay.dot - 2.0 - b * self.pile_step(lay, p, 0))
        }
    }

    /// How far apart the levels of a pile are in chart `row`: a dot's size, or less if
    /// the tallest pile would not fit.
    fn pile_step(&self, lay: &Layout, p: &Plot, row: usize) -> f32 {
        let tallest = if row == 0 { (0..self.sample.n()).map(|i| self.look.points.get(2 * i + 1)).fold(0.0f32, f32::max) } else { 0.0 };
        let room = if row == 0 { 0.55 } else { 0.9 } * p.height();
        let step = lay.dot + 1.0;
        if tallest > 0.0 && (tallest + 1.0) * step > room {
            (room / (tallest + 1.0)).max(1.0)
        } else {
            step
        }
    }

    /// Where each draw of a resample lands in the second chart (in the order drawn).
    fn landing(&self, lay: &Layout, p: &Plot, at: &[usize]) -> Vec<Vec2> {
        let sample = &self.sample;
        if self.setup.pairs() {
            // the same place as in the sample's chart, a copy drawn twice set a little aside
            let mut copies = vec![0u16; sample.n()];
            at.iter()
                .map(|&i| {
                    let k = copies.get(i).copied().unwrap_or(0);
                    if let Some(c) = copies.get_mut(i) {
                        *c += 1;
                    }
                    let a = self.look.points.get(2 * i) as f64;
                    let b = self.look.points.get(2 * i + 1) as f64;
                    p.at(a, b) + Vec2::new(3.0, -3.0) * k as f32
                })
                .collect()
        } else {
            let xs: Vec<f32> = at.iter().map(|&i| p.px_x(self.look.points.get(2 * i) as f64)).collect();
            let levels = stack(&xs, lay.dot + 1.0);
            let tallest = levels.iter().copied().max().unwrap_or(0) as f32;
            let room = 0.92 * p.height();
            let step = if (tallest + 1.0) * (lay.dot + 1.0) > room { (room / (tallest + 1.0)).max(1.0) } else { lay.dot + 1.0 };
            xs.iter().zip(levels).map(|(x, l)| Vec2::new(*x, p.rect.1.y - 0.5 * lay.dot - 2.0 - l as f32 * step)).collect()
        }
    }
}

impl Scene for Bootstrap {
    fn view(&self) -> View {
        View::Flat { center: Vec2::ZERO, half_height: 1.0 }
    }

    fn settings(&self, s: &mut ViewSettings) {
        s.grid = false;
        s.bloom = s.bloom.min(0.2);
    }

    fn animated(&self) -> bool {
        self.look.moving || !self.fresh.done() || !self.boot.done() || (self.shown.is_some() && self.progress < 1.0) || self.tally.pending()
    }

    fn pointer_hint(&self) -> Option<&'static str> {
        Some("buttons over the picture: click")
    }

    fn ui(&mut self, ui: &mut egui::Ui, _: &Gfx) {
        ui::note(
            ui,
            r"How much would a statistic change if the study were done again, with a new sample? With one sample and no formula, the bootstrap answers by treating the sample as if it were the population. Draw $n$ values from it at random, each time from all of them, so that some points come twice and some not at all: a resample. Work the statistic out for it. Do that thousands of times. The spread of the resamples' values estimates the statistic's standard error, and their middle 95 % is an interval for its true value. It works when the sample resembles the population in what the statistic depends on, and fails when it cannot. Here the population is known, so the answer can be checked against fresh samples from it (green).",
        );

        ui::section(ui, "Things to try");
        for (k, trial) in TRIALS.iter().enumerate() {
            if ui::experiment(ui, trial.name, trial.look_for) {
                self.set_up(k);
            }
        }

        ui::section(ui, "The statistic");
        let mut stat = Stat::ALL.iter().position(|s| *s == self.setup.stat).unwrap_or(0);
        let names: Vec<&str> = Stat::ALL.iter().map(|s| s.button()).collect();
        if ui::choice(ui, &names, &mut stat) {
            self.setup.stat = Stat::ALL[stat];
        }
        ui::section(ui, "The population");
        if self.setup.pairs() {
            let mut rho = self.setup.rho as f32;
            if ui::slider(ui, r"true correlation $\rho$", &mut rho, -0.95..=0.95) {
                self.setup.rho = rho as f64;
            }
            ui::note(ui, r"For the correlation, the sample is of pairs from a bell in two dimensions whose correlation is $\rho$.");
        } else {
            let mut pop = Pop::ALL.iter().position(|p| *p == self.setup.pop).unwrap_or(0);
            let names: Vec<&str> = Pop::ALL.iter().map(|p| p.name()).collect();
            if ui::choice(ui, &names, &mut pop) {
                self.setup.pop = Pop::ALL[pop];
            }
        }
        ui::slider_int_log(ui, "values in the sample ($n$)", &mut self.setup.n, 1..=N_MOST);
        ui::slider_log(ui, "resamples per second", &mut self.rate, 0.2..=2000.0);
        ui::note(ui, "Up to 10,000 are made. One resample at a time is shown being made, taking two seconds or more; the rest are only counted.");
        ui::checkbox(ui, &mut self.show_truth, "show the truth (the population, and fresh samples from it)");
        ui.horizontal(|ui| {
            if ui.button("new sample").clicked() {
                self.seed = self.seed.wrapping_add(1);
            }
        });
        if self.setup.plain().true_value().is_some() {
            if !self.counting && ui.button("count coverage over 1,000 new samples").clicked() {
                self.counting = true;
            }
            ui::note(ui, "Coverage: draw new samples from the population, make each one's interval from 1,000 resamples, and count how often it holds the true value.");
        } else {
            ui::note(ui, "This population has no largest value, so there is no true value for an interval to hold, and no coverage to count.");
        }
        self.refresh();

        ui::section(ui, "Reading the picture");
        ui::key(
            ui,
            &[
                (TRUTH, "green: the population, the true value, and the statistic of 10,000 fresh samples from the population (the truth the bootstrap tries to find)"),
                (DATA, "cyan: the sample's points, and their copies in a resample; dim, the points a resample left out"),
                (ESTIMATE, "yellow: a resample's statistic, the bars of all the resamples' values, and their middle 95 % (the percentile interval)"),
                (pal::WHITE, "white: the sample's own statistic"),
                (ALARM, "pink, among the numbers: resamples whose statistic has no value (a standard deviation of one value, a correlation with a column that does not vary)"),
            ],
        );
        ui::section(ui, "Words");
        ui::term(ui, "Resample:", r"$n$ values drawn at random from the sample, each from all $n$ of its points.");
        ui::term(ui, "With replacement:", "each point drawn is put back before the next draw, so it can be drawn again.");
        ui::term(ui, "Sampling distribution:", "how a statistic would vary over fresh samples from the population. Only known here because the population is.");
        ui::term(ui, "Bootstrap distribution:", "how the statistic varies over resamples of the one sample: the bootstrap's stand-in for the sampling distribution.");
        ui::term(ui, "Standard error:", "the standard deviation of a statistic's sampling distribution: how far it typically wanders from sample to sample.");
        ui::term(ui, "Percentile interval:", "from the 2.5th to the 97.5th percentile of the resamples' values.");
        ui::term(ui, "Coverage:", "the share of samples whose interval holds the true value. A 95 % interval should have 95 %.");

        ui::section(ui, "Numbers");
        for (name, value, color) in self.numbers() {
            if color == pal::TEXT {
                ui::readout(ui, &name, value);
            } else {
                ui::readout_colored(ui, &name, value, color);
            }
        }
        ui::see_also(
            ui,
            &[
                ("estimators", "the sampling distribution, from a known population"),
                ("intervals", "confidence intervals from a formula, and their coverage"),
                ("twogroups", "shuffling labels: resampling to test, not to estimate"),
                ("describe", "looking at one sample"),
            ],
        );
    }

    fn frame(&mut self, f: &mut Frame) {
        let lay = Layout::of(f.view_size(), f.top_inset, self.setup.pairs(), self.sample.n());
        let tiny = lay.view.x < 160.0 || lay.view.y < 160.0;
        let buttons_left = if !tiny && !lay.compact { self.buttons(f, &lay) } else { lay.view.x };
        self.work(f.dt, f.real_dt);
        let goals = self.goals(&lay);
        let snap = self.canvas != Some(lay.view) || self.look.points.shown.len() != goals.points.len();
        if !tiny {
            self.canvas = Some(lay.view);
        }
        self.look.follow(&goals, f.real_dt, snap);
        if tiny {
            return;
        }
        if lay.compact {
            let at = Vec2::new(10.0, f.top_inset + 10.0);
            wrapped_px(f, at, lay.view.x - 20.0, "The bootstrap: make the window larger to see its charts.", lay.text, pal::DIM);
            return;
        }
        let plots = self.plots(&lay);
        anchors(f);
        self.draw_population(f, &lay, &plots[0], buttons_left);
        self.draw_resample(f, &lay, &plots);
        self.draw_resamples(f, &lay, &plots);
        self.draw_numbers(f, &lay);
    }
}

/// Words broken into lines no wider than `width` pixels, from `at` down.
fn wrapped_px(f: &mut Frame, at: Vec2, width: f32, text: &str, size: f32, color: Color) -> f32 {
    let mut lines: Vec<String> = vec![String::new()];
    for word in text.split(' ') {
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
    for (k, line) in lines.iter().enumerate() {
        f.text_px(at + Vec2::new(0.0, k as f32 * step), line.as_str(), size, color, Align2::LEFT_TOP);
    }
    lines.len() as f32 * step
}

impl Bootstrap {
    /// The buttons over the picture: a new sample, and the truth shown or hidden.
    /// Returns where they start across.
    fn buttons(&mut self, f: &mut Frame, lay: &Layout) -> f32 {
        let words = match (lay.narrow, self.show_truth) {
            (false, true) => ["new sample", "hide the truth"],
            (false, false) => ["new sample", "show the truth"],
            (true, true) => ["new sample", "hide truth"],
            (true, false) => ["new sample", "show truth"],
        };
        let h = lay.small + 9.0;
        let rects = plot::button_row(f, &words, 0.0, lay.buttons.y, h, lay.small, 6.0);
        let width = rects.last().map_or(0.0, |r| r.1.x);
        let shift = Vec2::new(lay.buttons.x - width, 0.0);
        let rects: Vec<Rect> = rects.into_iter().map(|(a, b)| (a + shift, b + shift)).collect();
        if plot::button(f, rects[0], words[0], lay.small, false) {
            self.seed = self.seed.wrapping_add(1);
            self.refresh();
        }
        if plot::button(f, rects[1], words[1], lay.small, false) {
            self.show_truth = !self.show_truth;
        }
        rects[0].0.x
    }

    /// Ticks along the bottom of a chart: the round values of `goal` (the range the
    /// chart is going to), those beyond the chart as drawn unseen at its edge, so that
    /// their number stays the same while it glides.
    fn x_axis(&self, f: &mut Frame, lay: &Layout, p: &Plot, goal: [f64; 2], numbers: bool) {
        let (min, max) = p.rect;
        line_of_axes(f, Vec2::new(min.x, max.y), max);
        let ticks = Ticks::of(goal[0], goal[1], max.x - min.x);
        for v in ticks.values() {
            let x = p.px_x(v);
            let seen = x >= min.x - 0.5 && x <= max.x + 0.5;
            let x = x.max(min.x).min(max.x);
            plot::line_of(f, AXES, Vec2::new(x, max.y), Vec2::new(x, max.y + 4.0), 1.0, pal::DIM.a(if seen { 0.8 } else { 0.0 }));
            if numbers {
                let text = if seen { ticks.name(v + 0.0) } else { String::new() };
                f.text_px(Vec2::new(x, max.y + 6.0), text, lay.small, pal::DIM, Align2::CENTER_TOP);
            }
        }
    }

    /// The top chart: the population (when the truth is shown) and the sample.
    fn draw_population(&self, f: &mut Frame, lay: &Layout, p: &Plot, buttons_left: f32) {
        let s = self.setup.plain();
        let n = self.sample.n();
        let (min, max) = p.rect;
        let alpha = self.look.marks.get(4);
        let what = if s.pairs() { " pairs" } else { "" };
        let options: Vec<Title> = if self.show_truth {
            let whole = if s.pairs() { "The population (95 % of its pairs fall in the ellipse)" } else { "The population" };
            vec![
                [part(whole, TRUTH), part(format!("and one sample of $n = {n}${what} from it"), DATA)],
                [part("The population", TRUTH), part(format!("and a sample of $n = {n}$"), DATA)],
                [part("Population", TRUTH), part(format!("and sample, $n = {n}$"), DATA)],
                [part(format!("A sample of $n = {n}$"), DATA), part("", DATA)],
            ]
        } else {
            vec![
                [part(format!("One sample of $n = {n}${what}"), DATA), part("(the population is hidden, as it is in practice)", pal::DIM)],
                [part(format!("A sample of $n = {n}$"), DATA), part("", DATA)],
            ]
        };
        let at = Vec2::new(lay.rows[0].0.x - 12.0, min.y - 8.0);
        title(f, at, &options, None, buttons_left - at.x - 12.0, lay.big);
        plot::fill(f, (min - Vec2::new(4.0, 4.0), max + Vec2::new(4.0, 0.0)), Color::hex(0x0E1322).a(0.7));
        if s.pairs() {
            // the ellipse inside which 95 % of the population's pairs lie
            let c = Dist::ChiSquare { df: 2.0 }.quantile(0.95).sqrt();
            let (rho, side) = (s.rho, (1.0 - s.rho * s.rho).max(0.0).sqrt());
            let pts: Vec<(f64, f64)> = (0..=96)
                .map(|k| {
                    let t = k as f64 / 96.0 * std::f64::consts::TAU;
                    let (a, b) = (c * t.cos(), c * t.sin());
                    (a, rho * a + side * b)
                })
                .collect();
            p.path(f, L_ELLIPSE, pts, 1.6, TRUTH.a(0.75 * alpha));
            self.y_axis_pairs(f, lay, p);
        } else {
            let d = s.pop.dist();
            let g = |x: f64| d.pdf(x);
            p.area(f, p.x[0], p.x[1], 0.0, 160, TRUTH.a(0.07 * alpha), g);
            if s.pop == Pop::Flat {
                // (straight up at its ends, not sloping)
                let pts = [(p.x[0], 0.0), (0.0, 0.0), (0.0, 1.0), (1.0, 1.0), (1.0, 0.0), (p.x[1], 0.0)];
                p.path(f, CURVES, pts, 2.0, TRUTH.a(alpha));
            } else {
                p.curve(f, CURVES, 160, 2.0, TRUTH.a(alpha), g);
            }
        }
        // the sample's statistic, and the true value
        if !s.pairs() {
            self.value_marks(f, lay, p, 0, ("the sample's", pal::WHITE));
        }
        // the sample's points, the ones the resample shown so far has copied bright
        let picked = self.picked();
        for i in 0..n {
            let at = self.point_at(p, lay, i);
            let lit = picked.get(i).copied().unwrap_or(1.0);
            dot_of(f, D_SAMPLE, at, lay.dot, DATA.a(0.3 + 0.7 * lit));
        }
        self.x_axis(f, lay, p, self.sample.across, true);
        if s.pairs() {
            let text = match self.sample.own {
                Some(r) => format!(r"the sample's $r = {}$", num(r, 2)),
                None => "the sample has no correlation".to_string(),
            };
            f.text_px(Vec2::new(max.x + 16.0, min.y + 4.0), text, lay.text, pal::WHITE, Align2::LEFT_TOP);
            let truth = if self.show_truth { format!(r"true $\rho = {}$", num(s.rho, 2)) } else { String::new() };
            f.text_px(Vec2::new(max.x + 16.0, min.y + 8.0 + lay.text + 4.0), truth, lay.text, TRUTH, Align2::LEFT_TOP);
        }
    }

    /// The numbers up the side of a scatter.
    fn y_axis_pairs(&self, f: &mut Frame, lay: &Layout, p: &Plot) {
        let (min, max) = p.rect;
        line_of_axes(f, min, Vec2::new(min.x, max.y));
        let ticks = Ticks::of(self.sample.up[0], self.sample.up[1], (max.y - min.y) * 1.6);
        for v in ticks.values() {
            let y = p.px_y(v);
            let seen = y >= min.y - 0.5 && y <= max.y + 0.5;
            let y = y.max(min.y).min(max.y);
            plot::line_of(f, AXES, Vec2::new(min.x - 4.0, y), Vec2::new(min.x, y), 1.0, pal::DIM.a(if seen { 0.8 } else { 0.0 }));
            f.text_px(Vec2::new(min.x - 6.0, y), if seen { ticks.name(v + 0.0) } else { String::new() }, lay.small, pal::DIM, Align2::RIGHT_CENTER);
        }
    }

    /// How bright each of the sample's points is drawn: 1 once the resample shown has
    /// copied it, less before (and the last one's fading).
    fn picked(&self) -> Vec<f32> {
        let n = self.sample.n();
        let Some(sh) = &self.shown else { return vec![1.0; n] };
        let mut lit = vec![0.0f32; n];
        for (j, &i) in sh.at.iter().enumerate() {
            if let Some(l) = lit.get_mut(i) {
                *l = l.max(smooth(hop(j, n, self.progress) * 4.0));
            }
        }
        if let Some(b) = &self.before {
            let fade = 1.0 - smooth(self.progress / FADE);
            for (j, &i) in b.at.iter().enumerate() {
                if let (Some(l), true) = (lit.get_mut(i), j < n) {
                    *l = l.max(fade);
                }
            }
        }
        lit
    }

    /// Lines up a chart at the sample's statistic (in `own.1`), and at the true value
    /// when the truth is shown, each named at its top on the side away from the other.
    /// (A standard deviation is drawn as a bar either side of the mean instead.)
    fn value_marks(&self, f: &mut Frame, lay: &Layout, p: &Plot, row: usize, own: (&str, Color)) {
        let s = self.setup.plain();
        let alpha = self.look.marks.get(4);
        let (min, max) = p.rect;
        let digits = decimals_for(self.sample.across[1] - self.sample.across[0]);
        let truth = s.true_value();
        match s.stat {
            Stat::Sd if row == 0 => {
                // the mean, and a bar a standard deviation either side of it
                let m = st::mean(&self.sample.x);
                let sd = self.sample.own.unwrap_or(0.0);
                let y = min.y + 0.18 * p.height();
                let (a, b) = (p.px_x(m - sd), p.px_x(m + sd));
                plot::line_of(f, MARKS, Vec2::new(a, y), Vec2::new(b, y), 2.0, own.1);
                for x in [a, b] {
                    plot::line_of(f, MARKS, Vec2::new(x, y - 5.0), Vec2::new(x, y + 5.0), 2.0, own.1);
                }
                let label = match self.sample.own {
                    Some(v) => s.stat.own_text(v, digits + 1, true),
                    None => "one value: no standard deviation".to_string(),
                };
                f.text_px(Vec2::new(0.5 * (a + b), y - 7.0), label, lay.small, own.1, Align2::CENTER_BOTTOM);
                let d = s.pop.dist();
                let (mu, sigma) = (d.mean().unwrap_or(0.0), d.sd().unwrap_or(0.0));
                let y = min.y + 0.36 * p.height();
                let (a, b) = (p.px_x(mu - sigma), p.px_x(mu + sigma));
                plot::dashed_of(f, MARKS, Vec2::new(a, y), Vec2::new(b, y), 1.6, TRUTH.a(alpha));
                for x in [a, b] {
                    plot::line_of(f, MARKS, Vec2::new(x, y - 4.0), Vec2::new(x, y + 4.0), 1.6, TRUTH.a(alpha));
                }
                let text = if alpha > 0.01 { s.stat.true_text(sigma, digits + 1) } else { String::new() };
                f.text_px(Vec2::new(0.5 * (a + b), y + 6.0), text, lay.small, TRUTH.a(alpha), Align2::CENTER_TOP);
            }
            _ => {
                let own_v = self.look.marks.get(0) as f64;
                let true_v = self.look.marks.get(1) as f64;
                let has_truth = truth.is_some();
                let x_own = p.px_x(if row == 0 { self.sample.own.unwrap_or(own_v) } else { own_v });
                let x_true = p.px_x(true_v);
                let inside = |x: f32| x >= min.x - 0.5 && x <= max.x + 0.5;
                if inside(x_own) {
                    plot::line_of(f, MARKS, Vec2::new(x_own, min.y + 14.0), Vec2::new(x_own, max.y), 1.8, own.1);
                }
                let true_alpha = if has_truth { alpha } else { 0.0 };
                if inside(x_true) {
                    plot::dashed_of(f, MARKS, Vec2::new(x_true, min.y + 14.0), Vec2::new(x_true, max.y), 1.6, TRUTH.a(true_alpha));
                }
                // the names, each on the side away from the other
                let left = x_own <= x_true;
                let own_text = match (row, self.sample.own) {
                    (0, Some(v)) => s.stat.own_text(v, digits + 1, true),
                    (_, Some(v)) => s.stat.own_text(v, decimals_for(p.x[1] - p.x[0]) + 1, false),
                    (_, None) => String::new(),
                };
                let true_text = match truth {
                    Some(v) if true_alpha > 0.01 => s.stat.true_text(v, if row == 0 { digits + 1 } else { decimals_for(p.x[1] - p.x[0]) + 1 }),
                    _ => String::new(),
                };
                let y = min.y + 6.0;
                let side = if left { -1.0 } else { 1.0 };
                // each beside its line, on the side away from the other's, but inside the chart
                let beside = |f: &mut Frame, tag: &'static str, x: f32, side: f32, text: String, color: Color| {
                    let w = f.text_size(&text, lay.small).x;
                    let from = Vec2::new(x.max(min.x).min(max.x), y);
                    let centre = (from.x + side * (0.5 * w + 5.0)).max(min.x + 0.5 * w + 2.0).min(max.x - 0.5 * w - 2.0);
                    f.text_placed(tag, 0, from, Vec2::new(centre - from.x, 0.0), text, lay.small, color);
                };
                let tags: [&'static str; 2] = if row == 0 { ["own above", "true above"] } else { ["own below", "true below"] };
                beside(f, tags[0], x_own, side, own_text, own.1);
                beside(f, tags[1], x_true, -side, true_text, TRUTH.a(true_alpha.max(0.0)));
            }
        }
    }

    /// The middle chart: one resample being made.
    fn draw_resample(&self, f: &mut Frame, lay: &Layout, plots: &[Plot; 3]) {
        let s = self.setup.plain();
        let n = self.sample.n();
        let (p0, p) = (&plots[0], &plots[1]);
        let (min, max) = p.rect;
        let every = self.shown.as_ref().map_or(1, |sh| sh.every);
        let draws = if n == 1 { "draw" } else { "draws" };
        let note = |long: bool| match (every > 1, long) {
            (false, _) => String::new(),
            (true, true) => format!("(one in every {} made is shown)", count_text(every)),
            (true, false) => format!("(1 in {})", count_text(every)),
        };
        let options: Vec<Title> = vec![
            [part(format!("One resample: {n} {draws} from the sample, with replacement"), pal::TEXT), part(note(true), pal::DIM)],
            [part(format!("A resample: {n} {draws}, with replacement"), pal::TEXT), part(note(true), pal::DIM)],
            [part(format!("A resample: {n} {draws}, with replacement"), pal::TEXT), part(note(false), pal::DIM)],
            [part("A resample", pal::TEXT), part(note(false), pal::DIM)],
        ];
        let at = Vec2::new(lay.rows[1].0.x - 12.0, min.y - 8.0);
        let k_room = f.text_size(&format!("{n} of the {n} points appear (100.0 %)"), lay.text).x;
        title(f, at, &options, None, lay.rows[1].1.x - at.x - k_room - 16.0, lay.big);
        plot::fill(f, (min - Vec2::new(4.0, 4.0), max + Vec2::new(4.0, 0.0)), Color::hex(0x0E1322).a(0.7));
        // how many of the sample's points it has so far
        let count = match &self.shown {
            Some(sh) => {
                let mut seen = vec![false; n];
                for (j, &i) in sh.at.iter().enumerate() {
                    if hop(j, n, self.progress) > 0.0 {
                        if let Some(v) = seen.get_mut(i) {
                            *v = true;
                        }
                    }
                }
                seen.iter().filter(|v| **v).count()
            }
            None => 0,
        };
        let done = self.shown.as_ref().is_some_and(|_| hop(n.saturating_sub(1), n, self.progress) >= 1.0);
        let k_text = if self.shown.is_none() {
            String::new()
        } else if n == 1 {
            "its one point appears".to_string()
        } else if done {
            format!("{count} of the {n} points appear ({})", percent(count as f64 / n.max(1) as f64))
        } else {
            format!("{count} of the {n} points so far")
        };
        let right = lay.rows[1].1.x;
        f.text_px(Vec2::new(right, min.y - 8.0), k_text, lay.text, DATA, Align2::RIGHT_BOTTOM);

        // the dots: the last resample's fading where they landed, this one's hopping down
        let starts: Vec<Vec2> = (0..n).map(|i| self.point_at(p0, lay, i)).collect();
        if let Some(b) = &self.before {
            let fade = 1.0 - smooth(self.progress / FADE);
            if fade > 0.0 {
                let kind = D_HOPS + b.parity as u8;
                for at in self.landing(lay, p, &b.at) {
                    dot_of(f, kind, at, lay.dot, DATA.a(fade));
                }
                self.shown_mark(f, lay, p, b, fade, false);
            }
        }
        if let Some(sh) = &self.shown {
            let kind = D_HOPS + sh.parity as u8;
            let ends = self.landing(lay, p, &sh.at);
            for (j, (&i, end)) in sh.at.iter().zip(&ends).enumerate() {
                let u = hop(j, n, self.progress);
                let from = starts.get(i).copied().unwrap_or(*end);
                let e = smooth(u);
                let arc = 4.0 * u * (1.0 - u) * 10.0;
                let at = from + (*end - from) * e - Vec2::new(0.0, arc);
                dot_of(f, kind, at, lay.dot, DATA.a(smooth(u * 6.0)));
            }
            // its statistic, once it is made
            let mark = smooth((self.progress - MARKED) / 0.04);
            self.shown_mark(f, lay, p, sh, mark, true);
        }
        if s.pairs() {
            self.y_axis_pairs(f, lay, p);
        }
        self.x_axis(f, lay, p, self.sample.across, !s.pairs() || lay.view.y > 700.0);
    }

    /// The statistic of a resample marked on the middle chart (with the strength
    /// `alpha`: nothing is drawn at 0), and with `falls` falling into the chart below.
    fn shown_mark(&self, f: &mut Frame, lay: &Layout, p: &Plot, sh: &Shown, alpha: f32, falls: bool) {
        let s = self.setup.plain();
        let (min, max) = p.rect;
        let digits = decimals_for(self.sample.range[1] - self.sample.range[0]) + 1;
        let text = match sh.value {
            Some(v) => match s.stat {
                Stat::Corr => format!(r"its $r = {}$", num(v, digits)),
                Stat::Sd => format!(r"its $s = {}$", num(v, digits)),
                _ => format!("its {} {}", s.stat.short(), num(v, digits)),
            },
            None => match s.stat {
                Stat::Corr => "a column has no spread: no r".to_string(),
                _ => format!("no {}", s.stat.name()),
            },
        };
        let at = self.mark_place(p, sh);
        let place = if s.stat == Stat::Corr { Vec2::new(max.x + 16.0, min.y + 4.0) } else { Vec2::new(at.x.max(min.x + 40.0).min(max.x - 40.0), min.y + 2.0) };
        if alpha > 0.01 {
            let kind = L_SHOWN + sh.parity as u8;
            match (s.stat, sh.value) {
                (Stat::Corr, _) | (_, None) => {}
                (Stat::Sd, Some(v)) => {
                    let m = sh.at.iter().filter_map(|&i| self.sample.x.get(i)).sum::<f64>() / sh.at.len().max(1) as f64;
                    let (a, b) = (p.px_x(m - v), p.px_x(m + v));
                    plot::line_of(f, kind, Vec2::new(a, at.y), Vec2::new(b, at.y), 2.0, ESTIMATE.a(alpha));
                }
                (_, Some(_)) => {
                    plot::line_of(f, kind, Vec2::new(at.x, min.y + 16.0), Vec2::new(at.x, max.y), 2.0, ESTIMATE.a(alpha));
                }
            }
            let align = if s.stat == Stat::Corr { Align2::LEFT_TOP } else { Align2::CENTER_TOP };
            f.text_px(place, text, lay.small, ESTIMATE.a(alpha), align);
        }
        // and falling into the chart below, where it settles into its bar
        let fall = (self.progress - FALL) / (LAND - FALL);
        if let (true, true, Some(v)) = (falls, (0.0..1.3).contains(&fall), sh.value) {
            let p3 = self.plots(lay)[2];
            let bin = sh.bin.and_then(|b| self.look.bars.shown.get(b).copied()).unwrap_or(0.0) as f64;
            let to = p3.at(v, bin.min(p3.y[1]));
            let from = if s.stat == Stat::Corr { place + Vec2::new(10.0, lay.small) } else { Vec2::new(at.x, min.y + 16.0) };
            let t = fall.min(1.0);
            let pos = Vec2::new(from.x + (to.x - from.x) * smooth(t), from.y + (to.y - from.y) * t * t);
            let size = (lay.dot + 3.0) * (1.0 - (fall - 1.0).max(0.0) / 0.3);
            dot_of(f, D_FALL, pos, size.max(0.5), ESTIMATE);
        }
    }

    /// Where a resample's statistic is marked on the middle chart.
    fn mark_place(&self, p: &Plot, sh: &Shown) -> Vec2 {
        let (min, _) = p.rect;
        match (self.setup.stat, sh.value) {
            (Stat::Sd, Some(_)) => {
                let m = sh.at.iter().filter_map(|&i| self.sample.x.get(i)).sum::<f64>() / sh.at.len().max(1) as f64;
                Vec2::new(p.px_x(m), min.y + 0.3 * p.height())
            }
            (Stat::Corr, _) | (_, None) => Vec2::new(p.rect.1.x, min.y),
            (_, Some(v)) => Vec2::new(p.px_x(v), min.y),
        }
    }

    /// The bottom chart: the statistic of every resample so far, against fresh samples'.
    fn draw_resamples(&self, f: &mut Frame, lay: &Layout, plots: &[Plot; 3]) {
        let s = self.setup.plain();
        let p = &plots[2];
        let (min, max) = p.rect;
        let made = self.boot.counted();
        let name = s.stat.name();
        let titles = |made: usize| {
            let made_text = if made == 0 { "no".to_string() } else { count_text(made) };
            let (resamples, owner, each) = match made {
                0 => ("resamples yet", "resamples'", ""),
                1 => ("resample", "resample's", ""),
                _ => ("resamples", "resamples'", "each of "),
            };
            let fresh = |long: bool| match (self.show_truth, self.fresh.done(), long) {
                (false, _, _) => String::new(),
                (true, false, _) => "(fresh samples: being drawn)".to_string(),
                (true, true, true) => "and of 10,000 fresh samples".to_string(),
                (true, true, false) => "and of fresh samples".to_string(),
            };
            let plural = if s.stat == Stat::Corr { "correlations" } else { "values" };
            vec![
                [part(format!("The bootstrap distribution: the {name} of {each}{made_text} {resamples}"), ESTIMATE), part(fresh(true), TRUTH)],
                [part(format!("The {name} of {each}{made_text} {resamples}"), ESTIMATE), part(fresh(true), TRUTH)],
                [part(format!("The {name} of {made_text} {resamples}"), ESTIMATE), part(fresh(false), TRUTH)],
                [part(format!("{made_text} {owner} {plural}"), ESTIMATE), part(fresh(false), TRUTH)],
            ]
        };
        let options: Vec<Title> = titles(made);
        let widest: Vec<Title> = titles(MOST);
        let at = Vec2::new(lay.rows[2].0.x - 12.0, min.y - 8.0);
        title(f, at, &options, Some(&widest), lay.rows[2].1.x - at.x + 6.0, lay.big);
        plot::fill(f, (min - Vec2::new(4.0, 4.0), max + Vec2::new(4.0, 0.0)), Color::hex(0x0E1322).a(0.7));
        let w = self.boot.hist.width();
        let lo = self.boot.hist.lo;
        let alpha = self.look.marks.get(5);
        // the bars
        for i in 0..BINS {
            let h = self.look.bars.get(i) as f64;
            p.bar(f, lo + i as f64 * w, lo + (i + 1) as f64 * w, 0.0, h.min(p.y[1]), 0.5, ESTIMATE.a(0.72));
        }
        // the tallest bar, if it runs off the top: its share of the resamples beside it
        let tallest = (0..BINS).max_by(|a, b| self.look.bars.get(*a).total_cmp(&self.look.bars.get(*b))).unwrap_or(0);
        let off_top = self.look.bars.get(tallest) as f64 > p.y[1] * 1.01;
        let label = if off_top {
            let share = self.boot.hist.counts.get(tallest).copied().unwrap_or(0) as f64 / self.boot.hist.total.max(1) as f64;
            format!("\u{2191} {} of the resamples", percent(share))
        } else {
            String::new()
        };
        if off_top {
            // (placed through the frame's memory of it: if the tallest bar changes, the
            // label glides to it)
            let x = p.px_x(lo + (tallest as f64 + 1.0) * w) + 6.0 + 0.5 * f.text_size(&label, lay.small).x;
            let at = Vec2::new(x, min.y + 0.36 * p.height());
            f.text_placed("off the top", 0, min, at - min, label, lay.small, ESTIMATE);
        }
        // the fresh samples' outline
        if alpha > 0.005 && !self.fresh.values.is_empty() {
            let mut pts = Vec::with_capacity(2 * BINS + 2);
            pts.push((lo, 0.0));
            for i in 0..BINS {
                let h = (self.look.fresh.get(i) as f64).min(p.y[1]);
                pts.push((lo + i as f64 * w, h));
                pts.push((lo + (i + 1) as f64 * w, h));
            }
            pts.push((lo + BINS as f64 * w, 0.0));
            p.path(f, L_FRESH, pts, 1.6, TRUTH.a(alpha));
        }
        // the sample's own values, where every resample's value is one of them
        if s.on_values() {
            for v in &self.sample.x {
                let x = p.px_x(*v);
                if x >= min.x && x <= max.x {
                    plot::line_of(f, L_VALUES, Vec2::new(x, max.y + 1.0), Vec2::new(x, max.y + 7.0), 1.6, DATA);
                }
            }
        }
        // the sample's statistic, the true value, and the middle 95 %
        if self.sample.own.is_some() {
            self.value_marks(f, lay, p, 2, ("the sample's", pal::WHITE));
        }
        let (a, b) = (self.look.marks.get(2) as f64, self.look.marks.get(3) as f64);
        let y = min.y + 0.15 * p.height() + 10.0;
        let (xa, xb) = (p.px_x(a).max(min.x).min(max.x), p.px_x(b).max(min.x).min(max.x));
        // (from two values on: the middle 95 % of one value says nothing)
        let has = self.boot.sorted.len() >= 2;
        let c = ESTIMATE.a(if has { 1.0 } else { 0.0 });
        plot::line_of(f, MARKS, Vec2::new(xa, y), Vec2::new(xb, y), 2.0, c);
        for x in [xa, xb] {
            plot::line_of(f, MARKS, Vec2::new(x, y - 5.0), Vec2::new(x, y + 5.0), 2.0, c);
        }
        let digits = self.se_digits() - 1;
        let label = if has {
            if b - a <= 0.0 && self.boot.interval().is_some_and(|(a, b)| b <= a) {
                if self.boot.undefined > 0 {
                    format!("every resample with a {name} has the same one: an interval of no width")
                } else {
                    format!("every resample's {name} is the same: an interval of no width")
                }
            } else {
                format!("middle 95 %: {} to {}", num(a, digits), num(b, digits))
            }
        } else {
            String::new()
        };
        let half = 0.5 * f.text_size(&label, lay.small).x + 4.0;
        let mid = (0.5 * (xa + xb)).max(min.x + half).min(max.x - half);
        f.caption_px(Vec2::new(mid, y - 7.0), label, lay.small, ESTIMATE, Align2::CENTER_BOTTOM);
        let none = if made > 0 && self.boot.sorted.is_empty() {
            match s.stat {
                Stat::Sd => "A resample of one value has no standard deviation.".to_string(),
                Stat::Corr => "No resample has a correlation: in each, a column has no spread.".to_string(),
                _ => String::new(),
            }
        } else {
            String::new()
        };
        f.text_px((min + max) * 0.5, none, lay.text, pal::DIM, Align2::CENTER_CENTER);
        self.x_axis(f, lay, p, self.sample.range, true);
        let axis = format!("{} of a resample", if s.stat == Stat::Corr { "the correlation $r$" } else { name });
        f.text_px(Vec2::new(0.5 * (min.x + max.x), max.y + 8.0 + lay.small), axis, lay.small, pal::DIM, Align2::CENTER_TOP);
    }

    /// Decimals for a standard error: about three figures, judged by the range of the
    /// chart (which stays put while the resamples come, so the digits do too). An
    /// interval's ends get one fewer.
    fn se_digits(&self) -> usize {
        (2.0 - (0.25 * (self.sample.range[1] - self.sample.range[0])).log10().floor()).max(2.0).min(5.0) as usize
    }

    /// The numbers the picture rests on: (name, value, colour).
    fn numbers(&self) -> Vec<(String, String, Color)> {
        let s = self.setup.plain();
        let boot = &self.boot;
        let digits = self.se_digits();
        let mut out = Vec::new();
        let se = boot.se().map_or("–".to_string(), |v| num(v, digits));
        out.push(("standard error, from the resamples".to_string(), se, ESTIMATE));
        if s.stat == Stat::Mean {
            let n = self.sample.n();
            let formula = if n >= 2 { num(st::sd(&self.sample.x) / (n as f64).sqrt(), digits) } else { "needs two values".to_string() };
            out.push((r"by the formula $s/\sqrt{n}$".to_string(), formula, pal::WHITE));
        }
        if self.show_truth {
            let fresh = match (self.fresh.done(), self.fresh.se()) {
                (false, _) => "being worked out".to_string(),
                (true, Some(v)) => num(v, digits),
                (true, None) => "none: no fresh sample has a value".to_string(),
            };
            out.push(("true, from 10,000 fresh samples".to_string(), fresh, TRUTH));
        }
        let interval = match boot.interval() {
            Some((a, b)) => format!("{} to {}", num(a, digits - 1), num(b, digits - 1)),
            None => "–".to_string(),
        };
        out.push(("95 % percentile interval".to_string(), interval, ESTIMATE));
        if s.stat == Stat::Max && boot.made > 0 {
            out.push(("resamples with the sample's largest".to_string(), percent(boot.with_top as f64 / boot.made as f64), ESTIMATE));
            if self.show_truth {
                if let Some(above) = self.sample.own.and_then(|v| self.fresh.above(v)) {
                    out.push(("fresh samples whose largest is larger".to_string(), percent(above), TRUTH));
                }
            }
        }
        if boot.undefined > 0 {
            out.push(("resamples with no value".to_string(), count_text(boot.undefined), ALARM));
        }
        if self.counting {
            let t = &self.tally;
            let text = if t.done > 0 { format!("{} of {} ({})", count_text(t.covered), count_text(t.done), percent(t.covered as f64 / t.done as f64)) } else { "–".to_string() };
            out.push(("intervals of new samples that hold the truth".to_string(), text, TRUTH));
        }
        if boot.made > 0 {
            out.push(("points in a resample, on average".to_string(), percent(boot.appear / boot.made as f64), DATA));
        }
        // (the one being shown is made already; the bars count it once it has landed)
        out.push(("resamples made".to_string(), format!("{} of {}", count_text(boot.made), count_text(MOST)), pal::TEXT));
        out
    }

    /// The numbers, on the picture: beside the bottom chart or under it.
    fn draw_numbers(&self, f: &mut Frame, lay: &Layout) {
        let (min, max) = lay.numbers;
        let rows = self.numbers();
        let line = lay.small + 6.0;
        if lay.beside {
            // a name on one line and its value under it, in two columns of room
            let mut y = min.y + 2.0;
            for (name, value, color) in &rows {
                f.text_px(Vec2::new(min.x, y), name.as_str(), lay.small, pal::DIM, Align2::LEFT_TOP);
                f.text_px(Vec2::new(min.x + 12.0, y + line - 2.0), value.as_str(), lay.text, *color, Align2::LEFT_TOP);
                y += 2.0 * line + 2.0;
                if y > max.y {
                    break;
                }
            }
        } else {
            let mut y = min.y + 2.0;
            for (name, value, color) in &rows {
                if y + line > max.y {
                    break;
                }
                f.text_px(Vec2::new(min.x, y), name.as_str(), lay.small, pal::DIM, Align2::LEFT_TOP);
                f.text_px(Vec2::new(max.x, y), value.as_str(), lay.small, *color, Align2::RIGHT_TOP);
                y += line;
            }
        }
    }
}

/// A count with commas between its thousands, as the captions write them.
fn count_text(n: usize) -> String {
    let digits = n.to_string();
    let mut out = String::new();
    for (i, c) in digits.chars().enumerate() {
        if i > 0 && (digits.len() - i).is_multiple_of(3) {
            out.push(',');
        }
        out.push(c);
    }
    out
}

/// An axis line.
fn line_of_axes(f: &mut Frame, a: Vec2, b: Vec2) {
    plot::line_of(f, AXES, a, b, 1.0, pal::DIM.a(0.8));
}

#[cfg(test)]
mod tests {
    use super::*;

    /// An experiment as its button sets it up, with every resample made, every fresh
    /// sample drawn and the count (if it keeps one) finished.
    fn run(k: usize) -> Bootstrap {
        let mut s = Bootstrap::create();
        s.set_up(k);
        s.settle();
        s
    }

    /// A number as the picture and the panel write it, by its name.
    fn number(s: &Bootstrap, name: &str) -> String {
        s.numbers().into_iter().find(|r| r.0 == name).map(|r| r.1).unwrap_or_else(|| panic!("no number {name:?}"))
    }

    const SE: &str = "standard error, from the resamples";
    const FORMULA: &str = r"by the formula $s/\sqrt{n}$";
    const TRUE_SE: &str = "true, from 10,000 fresh samples";
    const INTERVAL: &str = "95 % percentile interval";

    /// The captions of the "things to try", held to what the scene shows.
    #[test]
    fn the_experiments_show_what_their_captions_say() {
        // the mean: 0.177 from the resamples, 0.180 by the formula and from fresh
        // samples; a shade low by sqrt(29/30); centred on the sample's mean, -0.13
        let mean = run(0);
        assert!(mean.boot.made == MOST && mean.sample.n() == 30 && mean.setup.pop == Pop::Bell);
        assert_eq!([number(&mean, SE), number(&mean, FORMULA), number(&mean, TRUE_SE)], ["0.177", "0.180", "0.180"]);
        let formula = st::sd(&mean.sample.x) / 30f64.sqrt();
        let ratio = mean.boot.se().unwrap_or(0.0) / formula;
        assert!((ratio - (29.0f64 / 30.0).sqrt()).abs() < 0.006, "{ratio}");
        assert_eq!((num(mean.sample.own.unwrap_or(9.0), 2), mean.setup.true_value()), ("−0.13".to_string(), Some(0.0)));
        let middle = st::median(&mean.boot.sorted);
        assert!((middle - mean.sample.own.unwrap_or(9.0)).abs() < 0.01, "the bars' middle {middle}");

        // the median: 31 values from the long tail, 0.217 against 0.180 (a fifth too
        // much), and every resample's median one of the sample's own values
        let median = run(1);
        assert!(median.sample.n() == 31 && median.setup.pop == Pop::Tail && median.setup.on_values());
        assert_eq!([number(&median, SE), number(&median, TRUE_SE)], ["0.217", "0.180"]);
        let much = median.boot.se().unwrap_or(0.0) / median.fresh.se().unwrap_or(1.0);
        assert!((much - 1.2).abs() < 0.02, "{much}");
        assert!(median.boot.sorted.iter().all(|v| median.sample.x.contains(v)));
        // spikes: most of the bins the resamples reach between the interval's ends are
        // empty or far below their neighbours
        let distinct = {
            let mut d = median.boot.sorted.clone();
            d.dedup();
            d.len()
        };
        assert!(distinct <= 31, "{distinct} different medians");

        // who is in a resample: 12 points, one resample in every 3.3 seconds, and on
        // average 1 - (11/12)^12 = 64.8 % of the points
        let who = run(2);
        assert!(who.sample.n() == 12 && every_at(who.rate) == 1 && SHOW <= 1.0 / who.rate);
        let expected = 1.0 - (11.0f64 / 12.0).powi(12);
        assert_eq!(percent(expected), "64.8\u{a0}%");
        assert_eq!(number(&who, "points in a resample, on average"), "64.8\u{a0}%");
        assert_eq!(percent(1.0 - (-1.0f64).exp()), "63.2\u{a0}%");

        // the largest value: 1.88, in 63.6 % of resamples, never exceeded; fresh
        // samples' largest beyond it in 60.2 %, spread 0.487 against 0.254
        let top = run(3);
        let own = top.sample.own.unwrap_or(0.0);
        assert_eq!(num(own, 2), "1.88");
        assert_eq!(number(&top, "resamples with the sample's largest"), "63.6\u{a0}%");
        assert_eq!(number(&top, "fresh samples whose largest is larger"), "60.2\u{a0}%");
        assert_eq!([number(&top, SE), number(&top, TRUE_SE)], ["0.254", "0.487"]);
        assert!(top.boot.sorted.iter().all(|v| *v <= own));
        let at_top = top.boot.sorted.iter().filter(|v| **v == own).count();
        assert_eq!(at_top, top.boot.with_top);
        assert!(top.boot.interval().is_some_and(|(_, hi)| hi == own));
        assert!(top.fresh.se().unwrap_or(0.0) > 1.9 * top.boot.se().unwrap_or(1.0));

        // too little data: the interval holds the true mean in 881 of 1,000 new samples;
        // this sample's s = 0.98 is typical (the median of eight values' is 0.96), and
        // gives 0.327 against 0.352
        let few = run(4);
        assert!(few.sample.n() == 8 && few.counting);
        assert_eq!(num(st::sd(&few.sample.x), 2), "0.98");
        let mut spreads: Vec<f64> = (1..=4000u64).map(|seed| st::sd(&Sample::new(few.setup.plain(), seed, &[]).x)).collect();
        spreads.sort_by(f64::total_cmp);
        assert_eq!(num(st::quantile_sorted(&spreads, 0.5), 2), "0.96");
        assert_eq!([number(&few, SE), number(&few, TRUE_SE)], ["0.327", "0.352"]);
        assert_eq!(number(&few, "intervals of new samples that hold the truth"), "881 of 1,000 (88.1\u{a0}%)");
        assert_eq!((few.tally.covered, few.tally.done, few.tally.undefined), (881, TALLY, 0));
        // (and it misses on both sides alike)
        assert!(few.tally.low.abs_diff(few.tally.high) < 25, "{} below, {} above", few.tally.low, few.tally.high);

        // a correlation: r = 0.910, interval 0.812 to 0.954, 0.098 below and 0.044
        // above; fresh samples' middle 95 % from 0.805 to 0.952
        let corr = run(5);
        let r = corr.sample.own.unwrap_or(0.0);
        assert!(corr.sample.n() == 30 && corr.setup.rho == 0.9);
        assert_eq!(num(r, 3), "0.910");
        let (lo, hi) = corr.boot.interval().unwrap_or((0.0, 0.0));
        assert_eq!([num(lo, 3), num(hi, 3), num(r - lo, 3), num(hi - r, 3)], ["0.812", "0.954", "0.098", "0.044"]);
        assert_eq!(number(&corr, INTERVAL), "0.812 to 0.954");
        let f = &corr.fresh.sorted;
        assert_eq!([num(st::quantile_sorted(f, 0.025), 3), num(st::quantile_sorted(f, 0.975), 3)], ["0.805", "0.952"]);
        assert!(corr.boot.sorted.iter().all(|v| *v <= 1.0) && corr.boot.undefined == 0);
    }

    /// What the captions say of many samples: the count at 30 values, the width of the
    /// interval at 8 against Student's t, and how rough the bootstrap's standard error
    /// is for a median against a mean. (A few seconds: run with --ignored.)
    #[test]
    #[ignore]
    fn what_the_captions_say_of_many_samples_holds() {
        // 30 values: 929 of the same 1,000 new samples
        let mut s = Bootstrap::create();
        s.set_up(4);
        s.setup.n = 30;
        s.settle();
        assert_eq!((s.tally.covered, s.tally.done), (929, TALLY));
        // 8 values: 0.77 times as wide as the t interval, on average
        let setup = Setup { n: 8, ..s.setup };
        let t = Dist::StudentT { df: 7.0 }.quantile(0.975);
        let mut ratio = 0.0;
        for k in 0..2000u64 {
            let sample = Sample::new(setup, 7000 + k, &[]);
            let mut b = Boot::new(7000 + k, sample.range);
            for _ in 0..1000 {
                b.step(&setup, &sample, false);
            }
            let (lo, hi) = b.interval().unwrap_or((0.0, 0.0));
            ratio += (hi - lo) / (2.0 * t * st::sd(&sample.x) / 8f64.sqrt());
        }
        let ratio = ratio / 2000.0;
        assert_eq!(num(ratio, 2), "0.77");
        // the median's standard error is too high about as often as too low, but off by
        // more than a third in 3 samples of 10; the mean's in fewer than 1 in 100
        for (k, n, least, most) in [(1usize, 31usize, 0.27, 0.33), (0, 30, 0.0, 0.01)] {
            let t = &TRIALS[k];
            let setup = Setup { pop: t.pop, stat: t.stat, n, rho: 0.0 };
            let truth = run(k).fresh.se().unwrap_or(1.0);
            let mut off = 0;
            let mut ratios = Vec::new();
            for seed in 0..2000u64 {
                let sample = Sample::new(setup, 5000 + seed, &[]);
                let mut b = Boot::new(5000 + seed, sample.range);
                for _ in 0..1000 {
                    b.step(&setup, &sample, false);
                }
                let r = b.se().unwrap_or(0.0) / truth;
                off += ((r - 1.0).abs() > 1.0 / 3.0) as usize;
                ratios.push(r);
            }
            let share = off as f64 / 2000.0;
            assert!(share >= least && share <= most, "{}: off by a third in {share}", t.name);
            // too high as often as too low
            let middle = st::median(&ratios);
            let high = ratios.iter().filter(|r| **r > 1.0).count() as f64 / ratios.len() as f64;
            if k == 1 {
                assert!((middle - 1.0).abs() < 0.03 && (high - 0.5).abs() < 0.03, "{}: {middle}, too high in {high}", t.name);
            }
        }
    }

    /// The percentile interval's ends are the 2.5th and 97.5th percentiles of the
    /// resampled values: as many of them below the one as above the other, 2.5 % each.
    #[test]
    fn the_interval_is_the_middle_95_percent_of_the_resamples() {
        for k in [0, 1, 5] {
            let s = run(k);
            let v = &s.boot.sorted;
            assert!(v.windows(2).all(|w| w[0] <= w[1]) && v.len() == MOST);
            let (lo, hi) = s.boot.interval().unwrap_or((0.0, 0.0));
            assert_eq!((lo, hi), (st::quantile_sorted(&st::sorted(v), 0.025), st::quantile_sorted(&st::sorted(v), 0.975)));
            // (where values are tied at an end, those strictly beyond it are fewer)
            let share = |keep: &dyn Fn(f64) -> bool| v.iter().filter(|x| keep(**x)).count() as f64 / v.len() as f64;
            let (below, upto) = (share(&|x| x < lo), share(&|x| x <= lo));
            let (above, from) = (share(&|x| x > hi), share(&|x| x >= hi));
            let tied = k == 1;
            assert!(below <= 0.0251 && upto >= 0.0249 && above <= 0.0251 && from >= 0.0249, "{}: {below} below, {above} above", TRIALS[k].name);
            assert!(tied || ((below - 0.025).abs() < 0.0002 && (above - 0.025).abs() < 0.0002), "{}: {below} below, {above} above", TRIALS[k].name);
            // and the values are those of resamples made afresh from the same seed
            let setup = s.setup.plain();
            let mut again = Boot::new(s.seed, s.sample.range);
            let mut values: Vec<f64> = (0..MOST).filter_map(|_| again.step(&setup, &s.sample, false).0).collect();
            values.sort_by(f64::total_cmp);
            assert!(values == *v);
        }
    }

    /// A quantile found in one pass is the one `quantile_sorted` finds of the numbers
    /// sorted.
    #[test]
    fn a_quantile_in_one_pass_is_the_sorted_one() {
        let mut rng = Rng::new(5);
        for n in [1usize, 2, 3, 7, 8, 30, 31, 200] {
            let v: Vec<f64> = (0..n).map(|_| (st::unit(&mut rng) * 10.0).round() / 2.0).collect();
            for p in [0.0, 0.1, 0.25, 0.5, 0.9, 0.975, 1.0] {
                assert_eq!(quantile_of(&mut v.clone(), p), st::quantile_sorted(&st::sorted(&v), p), "{n} values at {p}");
            }
        }
        assert!(quantile_of(&mut [], 0.5).is_nan());
    }

    /// On average a resample holds 1 - (1 - 1/n)^n of the sample's points, for any n.
    #[test]
    fn a_resample_holds_its_share_of_the_points() {
        for n in [1usize, 2, 5, 12, 30, 200] {
            let setup = Setup { pop: Pop::Bell, stat: Stat::Mean, n, rho: 0.0 };
            let sample = Sample::new(setup, 1, &[]);
            let mut b = Boot::new(1, sample.range);
            for _ in 0..20_000 {
                b.step(&setup, &sample, false);
            }
            let share = b.appear / b.made as f64;
            let expected = 1.0 - (1.0 - 1.0 / n as f64).powi(n as i32);
            assert!((share - expected).abs() < 0.004, "n = {n}: {share} against {expected}");
            // and the sample's largest point is among them just as often
            let top = b.with_top as f64 / b.made as f64;
            assert!((top - expected).abs() < 0.012, "n = {n}: the largest in {top}");
        }
    }

    /// One value, two, values all equal, a column that does not vary: every number
    /// drawn is finite, and the picture says what has happened.
    #[test]
    fn degenerate_samples_are_told_honestly() {
        let lays = [Layout::of(Vec2::new(1075.0, 1000.0), 0.0, false, 1), Layout::of(Vec2::new(390.0, 756.0), 44.0, true, 2), Layout::of(Vec2::new(240.0, 180.0), 0.0, false, 200)];
        for stat in Stat::ALL {
            for pop in Pop::ALL {
                for n in [1usize, 2, 200] {
                    let mut s = Bootstrap::create();
                    s.setup = Setup { pop, stat, n, rho: 0.95 };
                    s.settle();
                    for lay in &lays {
                        let g = s.goals(lay);
                        let all = g.ranges.iter().chain(&g.points).chain(&g.bars).chain(&g.fresh).chain(&g.marks);
                        assert!(all.clone().all(|v| v.is_finite()), "{stat:?} {pop:?} {n}");
                        assert!(g.ranges[1] > g.ranges[0] && g.ranges[5] > g.ranges[4] && g.ranges[6] > 0.0);
                    }
                    for (name, value, _) in s.numbers() {
                        assert!(!value.contains('?') && !value.contains("NaN"), "{stat:?} {pop:?} {n}: {name} = {value}");
                    }
                    if n == 1 {
                        // every resample is the sample itself
                        match stat {
                            Stat::Sd | Stat::Corr => assert!(s.sample.own.is_none() && s.boot.sorted.is_empty() && s.boot.undefined == MOST),
                            _ => {
                                let (lo, hi) = s.boot.interval().unwrap_or((1.0, 0.0));
                                assert!(lo == hi && Some(lo) == s.sample.own, "{stat:?}");
                            }
                        }
                    }
                    if n == 2 && stat == Stat::Corr {
                        // a resample of two pairs has two different ones half the time
                        assert!(s.boot.undefined > 4500 && s.boot.undefined < 5500 && s.boot.sorted.iter().all(|r| r.abs() > 0.999));
                    }
                }
            }
        }
        // values all equal: an interval of no width, said plainly
        let mut s = Bootstrap::create();
        s.settle();
        for v in s.sample.x.iter_mut() {
            *v = 0.25;
        }
        s.boot = Boot::new(1, s.sample.range);
        while !s.boot.done() {
            s.step();
        }
        assert_eq!(s.boot.interval(), Some((0.25, 0.25)));
        // a column that does not vary: no correlation, for the sample or any resample
        let mut s = Bootstrap::create();
        s.setup.stat = Stat::Corr;
        s.settle();
        for v in s.sample.y.iter_mut() {
            *v = 1.0;
        }
        let x = s.sample.x.clone();
        assert!(statistic(Stat::Corr, &mut x.clone(), &s.sample.y).is_none());
        let mut b = Boot::new(1, s.sample.range);
        for _ in 0..100 {
            assert!(b.step(&s.setup.plain(), &s.sample, false).0.is_none());
        }
    }

    /// The layout of every window: finite, inside the canvas, the charts one under
    /// another, also on a phone.
    #[test]
    fn the_layout_fits_every_window() {
        for pairs in [false, true] {
            for (w, h, inset) in [(1075.0, 1000.0, 0.0), (1600.0, 1000.0, 0.0), (1400.0, 760.0, 0.0), (390.0, 756.0, 44.0), (700.0, 500.0, 0.0), (240.0, 180.0, 0.0), (1.0, 1.0, 0.0), (3000.0, 2000.0, 0.0)] {
                let lay = Layout::of(Vec2::new(w, h), inset, pairs, 30);
                let rects = lay.rows.iter().chain(&lay.scatter).chain([&lay.numbers]);
                assert!(rects.clone().all(|(a, b)| a.is_finite() && b.is_finite()), "{w}x{h}");
                if w >= 390.0 && h >= 700.0 {
                    assert!(!lay.compact, "{w}x{h}");
                    assert!(lay.rows[0].1.y < lay.rows[1].0.y && lay.rows[1].1.y < lay.rows[2].0.y, "{w}x{h}");
                    assert!(lay.rows[0].0.y >= inset + 20.0 && lay.rows.iter().all(|(a, b)| b.y - a.y > 75.0) && lay.rows[2].1.y - lay.rows[2].0.y > 140.0, "{w}x{h}");
                    assert!(rects.clone().all(|(a, b)| a.x >= 0.0 && a.y >= 0.0 && b.x <= w + 0.5 && b.y <= h + 0.5), "{w}x{h}");
                    assert!(lay.numbers.1.y - lay.numbers.0.y > 60.0, "{w}x{h}: room for the numbers");
                    // the scatters are no wider than half as much again as they are tall
                    for (a, b) in lay.scatter {
                        assert!(b.x - a.x <= 1.5 * (b.y - a.y) + 0.5);
                    }
                }
            }
        }
    }

    /// After a change, the picture glides to the new one and comes to rest exactly
    /// there, at any frame rate.
    #[test]
    fn glides_arrive_and_stop() {
        let lay = Layout::of(Vec2::new(1075.0, 1000.0), 0.0, false, 30);
        for hz in [30.0f32, 60.0, 144.0] {
            let mut s = Bootstrap::create();
            s.settle();
            let g = s.goals(&lay);
            s.look.follow(&g, 0.0, true);
            assert!(!s.look.moving);
            s.seed += 1;
            s.settle();
            let mut frames = 0;
            loop {
                let g = s.goals(&lay);
                s.look.follow(&g, 1.0 / hz, false);
                frames += 1;
                if !s.look.moving {
                    assert!(s.look.points.shown == g.points && s.look.bars.shown == g.bars && s.look.marks.shown == g.marks);
                    break;
                }
                assert!((frames as f32) < 2.0 * hz, "still gliding after two seconds at {hz} Hz");
            }
            assert!(frames as f32 > 0.2 * hz, "arrived in {frames} frames at {hz} Hz: no glide");
        }
    }

    /// The resample shown moves on evenly at every display rate: how far through its
    /// time it is grows by the same amount in every frame (but where the next begins),
    /// however the resamples fall into frames, and its statistic lands in the chart
    /// below before the next is shown.
    #[test]
    fn the_resample_shown_moves_evenly_at_every_display_rate() {
        for rate in [0.3f32, 0.5, 3.0, 90.0, 500.0] {
            for hz in [60.0f32, 75.0, 120.0, 144.0] {
                let mut s = Bootstrap::create();
                s.rate = rate;
                s.refresh();
                let dt = 1.0 / hz;
                let (mut last, mut wraps, mut steps) = (0.0f32, 0, Vec::new());
                let seconds = 2.5 * every_at(rate) as f32 / rate + 1.0;
                for _ in 0..(seconds * hz) as usize {
                    let before = s.shown.as_ref().map(|sh| sh.at.clone());
                    // (frames with time to spare: what a busy machine leaves is not what is tested here)
                    s.work(dt, 1.0);
                    let now = s.progress;
                    if s.shown.as_ref().map(|sh| sh.at.clone()) != before && before.is_some() {
                        wraps += 1;
                        assert!(last >= LAND, "rate {rate} at {hz} Hz: a new one shown at {last}, before the last landed");
                    } else if before.is_some() {
                        steps.push(now - last);
                    }
                    last = now;
                }
                let every = every_at(rate) as f32;
                let expected = dt * rate / every;
                assert!(wraps >= 2, "rate {rate} at {hz} Hz: {wraps} shown");
                for d in &steps {
                    assert!((d - expected).abs() < 1e-3 * (1.0 + 1.0 / expected) * expected.max(1e-3), "rate {rate} at {hz} Hz: a step of {d} for {expected}");
                }
            }
        }
    }

    /// A change of setting starts the resamples afresh; a change of rate does not.
    #[test]
    fn a_change_starts_the_resamples_afresh() {
        let mut s = Bootstrap::create();
        for _ in 0..50 {
            s.step();
        }
        assert!(s.boot.made == 51 && s.shown.is_some());
        s.rate = 10.0;
        s.refresh();
        assert_eq!(s.boot.made, 51);
        // (a restart makes its first resample at once, and shows it from its start)
        s.setup.n = 40;
        s.refresh();
        assert!(s.boot.made == 1 && s.shown.as_ref().is_some_and(|sh| sh.since == 0) && s.progress == 0.0 && s.sample.n() == 40);
        s.seed += 1;
        s.refresh();
        assert!(s.sample.key.1 == s.seed && s.boot.made == 1);
        // the fresh samples depend on the population, not on the sample
        let fresh = s.fresh.key;
        s.seed += 1;
        s.refresh();
        assert_eq!(s.fresh.key, fresh);
    }

    /// Every caption, note and name the panel shows is markup that sets.
    #[test]
    fn the_texts_are_markup_that_sets() {
        for t in &TRIALS {
            for text in [t.name, t.look_for] {
                let found = crate::typeset::inspect(text);
                assert!(found.error.is_none(), "{text}: {found:?}");
            }
        }
        for stat in Stat::ALL {
            assert!(crate::typeset::inspect(stat.button()).error.is_none());
        }
    }

    /// Samples of eight to choose from (run with --ignored --nocapture).
    #[test]
    #[ignore]
    fn candidates() {
        let setup = Setup { pop: Pop::Bell, stat: Stat::Mean, n: 8, rho: 0.0 };
        let mut all: Vec<f64> = Vec::new();
        for seed in 1..=4000u64 {
            let sample = Sample::new(setup, seed, &[]);
            all.push(st::sd(&sample.x));
            if seed <= 40 {
                println!("seed {seed}: mean {:.3} s {:.3}", st::mean(&sample.x), st::sd(&sample.x));
            }
        }
        all.sort_by(f64::total_cmp);
        println!("median s {:.3}, mean s {:.3}", st::quantile_sorted(&all, 0.5), st::mean(&all));
    }

    /// The numbers behind the captions (run with --ignored --nocapture).
    #[test]
    #[ignore]
    fn explore() {
        for k in 0..TRIALS.len() {
            let s = run(k);
            println!("{}:", TRIALS[k].name);
            for (name, value, _) in s.numbers() {
                println!("    {name}: {value}");
            }
            let f = &s.fresh.sorted;
            println!("    own {:?}, sample sd {:.4}, fresh 95% {:.4} to {:.4}", s.sample.own, st::sd(&s.sample.x), st::quantile_sorted(f, 0.025), st::quantile_sorted(f, 0.975));
            if let Some((lo, hi)) = s.boot.interval() {
                let own = s.sample.own.unwrap_or(0.0);
                println!("    interval {lo:.4} to {hi:.4}, arms {:.4} {:.4}, se {:.4}", own - lo, hi - own, s.boot.se().unwrap_or(0.0));
            }
        }
    }
}

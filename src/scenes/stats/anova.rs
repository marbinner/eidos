//! Several groups: analysis of variance.
//!
//! A few groups of points side by side, each with its mean (a bar to drag) and the grand
//! mean across all of them. The spread of all the points about the grand mean splits
//! exactly into a part between the groups (how far the group means lie from the grand
//! mean) and a part within them (how far the points lie from their own group's mean);
//! each part over its degrees of freedom is a mean square, and their ratio is F. Beside
//! it, the distribution F has when the true means are all equal: by formula, by
//! shuffling the points among the groups, or by drawing new samples from the truth and
//! counting how often the test (and, for comparison, every pair of groups tested on
//! its own) cries "different".

use super::plot::{self, Glide, Plot, Rect, AXES, CURVES, MARKS};
use super::*;
use crate::math::stats::{self as st, Dist, Histogram};

pub const DEF: SceneDef = SceneDef {
    id: "anova",
    title: "Several groups: analysis of variance",
    group: TESTING,
    blurb: "The variation between groups against the variation within them: the $F$ ratio and what it is compared with.",
    make: |_| Box::new(Anova::create()),
};

// ----- the data ------------------------------------------------------------------------

/// The most groups, and the fewest and most points in a group. (A group needs two
/// points to say anything about the scatter within it.)
const GROUPS: usize = 6;
const FEWEST: usize = 2;
const MOST: usize = 60;
/// The largest spread a group can be given.
const WIDEST: f64 = 3.0;
/// How far from 0 a group's true mean can be put.
const REACH: f64 = 8.0;
/// The test's threshold.
const LEVEL: f64 = 0.05;

/// What the data are drawn from: `k` groups, group `i` with `n[i]` points from a bell
/// of mean `mu[i]` and spread `sd[i]`, and the seed of the noise. (Only the first `k`
/// entries count.)
#[derive(Clone, Copy, Debug, PartialEq)]
struct Design {
    k: usize,
    n: [usize; GROUPS],
    mu: [f64; GROUPS],
    sd: [f64; GROUPS],
    seed: u64,
}

impl Design {
    /// `k` groups of `n` points, spread `sd`, with these true means.
    fn even(k: usize, n: usize, sd: f64, mu: &[f64], seed: u64) -> Design {
        let mut means = [0.0; GROUPS];
        for (m, v) in means.iter_mut().zip(mu) {
            *m = *v;
        }
        Design { k, n: [n; GROUPS], mu: means, sd: [sd; GROUPS], seed }.kept()
    }

    /// Everything inside its range.
    fn kept(mut self) -> Design {
        self.k = self.k.clamp(2, GROUPS);
        for i in 0..GROUPS {
            self.n[i] = self.n[i].clamp(FEWEST, MOST);
            self.sd[i] = if self.sd[i].is_finite() { self.sd[i].max(0.0).min(WIDEST) } else { 1.0 };
            self.mu[i] = if self.mu[i].is_finite() { self.mu[i].max(-REACH).min(REACH) } else { 0.0 };
        }
        self
    }

    fn total(&self) -> usize {
        self.n[..self.k].iter().sum()
    }

    /// Are the true means of the groups all the same?
    fn equal_means(&self) -> bool {
        self.mu[..self.k].iter().all(|m| (m - self.mu[0]).abs() < 1e-9)
    }

    /// Have all the groups the same spread?
    fn equal_spreads(&self) -> bool {
        self.sd[..self.k].iter().all(|s| (s - self.sd[0]).abs() < 1e-9)
    }

    /// Have all the groups the same size?
    fn equal_sizes(&self) -> bool {
        self.n[..self.k].iter().all(|n| *n == self.n[0])
    }
}

/// Standard normal numbers for every place a point can have, from one seed: a row of
/// `MOST` for each group. A group's first points stay the same when it grows, so that
/// more data adds points and moves none.
type Noise = [[f64; MOST]; GROUPS];

fn noise(seed: u64) -> Box<Noise> {
    let mut z = Box::new([[0.0; MOST]; GROUPS]);
    for (g, row) in z.iter_mut().enumerate() {
        let mut rng = Rng::new(seed.wrapping_mul(0x9E37_79B9_7F4A_7C15).wrapping_add(1000 * (g as u64 + 1)));
        for v in row.iter_mut() {
            *v = st::normal(&mut rng);
        }
    }
    z
}

/// The points of each group.
fn points(d: &Design, z: &Noise) -> Vec<Vec<f64>> {
    (0..d.k).map(|i| (0..d.n[i]).map(|j| d.mu[i] + d.sd[i] * z[i][j]).collect()).collect()
}

/// The mean of group `i`'s noise: how far its sample mean lies from its true mean, in
/// units of its spread.
fn noise_mean(d: &Design, z: &Noise, i: usize) -> f64 {
    let n = d.n[i].min(MOST);
    z[i][..n].iter().sum::<f64>() / n.max(1) as f64
}

/// Move group `i` (all its points together) so that its sample mean is `at`: what
/// dragging its bar does. (It moves the group's true mean by as much.)
fn put_mean(d: &mut Design, z: &Noise, i: usize, at: f64) {
    if i < GROUPS && at.is_finite() {
        d.mu[i] = (at - d.sd[i] * noise_mean(d, z, i)).max(-REACH).min(REACH);
    }
}

// ----- the analysis --------------------------------------------------------------------

/// The analysis of variance of some groups: the split of the total sum of squares into
/// a part between the groups and a part within them, the mean squares, F and its
/// p-value.
#[derive(Clone, Debug, PartialEq)]
struct Split {
    k: usize,
    total: usize,
    means: Vec<f64>,
    grand: f64,
    /// The sums of squares: between the groups, within them, and of all the points
    /// about the grand mean (worked out on its own: it is the other two's sum).
    between: f64,
    within: f64,
    all: f64,
    /// Degrees of freedom: k − 1 and N − k.
    df_b: f64,
    df_w: f64,
    ms_b: f64,
    ms_w: f64,
    /// F and its p-value; none when there is no scatter within the groups.
    f: Option<f64>,
    p: Option<f64>,
}

/// Is a sum of squares nothing at all, against the size of the numbers? (A mean of
/// equal numbers is not always exactly that number in floating point.)
fn nothing(ss: f64, scale: f64) -> bool {
    // (not a number counts as nothing too)
    ss.is_nan() || ss <= 1e-24 * (1.0 + scale)
}

impl Split {
    fn of(groups: &[Vec<f64>]) -> Split {
        let k = groups.len();
        let total: usize = groups.iter().map(Vec::len).sum();
        let means: Vec<f64> = groups.iter().map(|g| st::mean(g)).collect();
        let grand = groups.iter().flatten().sum::<f64>() / total.max(1) as f64;
        let between: f64 = groups.iter().zip(&means).map(|(g, m)| g.len() as f64 * (m - grand) * (m - grand)).sum();
        let within: f64 = groups.iter().zip(&means).map(|(g, m)| g.iter().map(|y| (y - m) * (y - m)).sum::<f64>()).sum();
        let all: f64 = groups.iter().flatten().map(|y| (y - grand) * (y - grand)).sum();
        let df_b = k.saturating_sub(1) as f64;
        let df_w = total.saturating_sub(k) as f64;
        let ms_b = if df_b > 0.0 { between / df_b } else { 0.0 };
        let ms_w = if df_w > 0.0 { within / df_w } else { 0.0 };
        let scale = groups.iter().flatten().map(|y| y * y).sum::<f64>();
        let f = (df_b > 0.0 && df_w > 0.0 && !nothing(within, scale)).then(|| ms_b / ms_w);
        let p = f.map(|f| (1.0 - Dist::F { d1: df_b, d2: df_w }.cdf(f)).max(0.0).min(1.0));
        Split { k, total, means, grand, between, within, all, df_b, df_w, ms_b, ms_w, f, p }
    }

    /// Does nothing vary at all (every point the same)?
    fn still(&self) -> bool {
        nothing(self.all, self.grand * self.grand * self.total as f64)
    }

    /// The share of the variation that lies between the groups (none if nothing varies).
    fn share(&self) -> Option<f64> {
        (!self.still()).then(|| (self.between / self.all).max(0.0).min(1.0))
    }

    /// The null distribution F is compared with (a stand-in when there are too few
    /// points for one: never, with two points a group).
    fn null(&self) -> Dist {
        Dist::F { d1: self.df_b.max(1.0), d2: self.df_w.max(1.0) }
    }
}

/// F alone, fast, for values laid end to end in groups of these sizes (0 when there is
/// no scatter within the groups and none between them; infinite when there is only
/// scatter between).
fn ratio(values: &[f64], sizes: &[usize]) -> f64 {
    let total = values.len();
    let k = sizes.len();
    if k < 2 || total <= k {
        return f64::NAN;
    }
    let grand = values.iter().sum::<f64>() / total as f64;
    let (mut between, mut within, mut at) = (0.0, 0.0, 0);
    for &n in sizes {
        let g = &values[at.min(total)..(at + n).min(total)];
        at += n;
        let m = g.iter().sum::<f64>() / g.len().max(1) as f64;
        between += g.len() as f64 * (m - grand) * (m - grand);
        within += g.iter().map(|y| (y - m) * (y - m)).sum::<f64>();
    }
    let f = (between / (k - 1) as f64) / (within / (total - k) as f64);
    if f.is_nan() {
        0.0
    } else {
        f
    }
}

/// Student's t for two groups with a pooled spread (group `a` less group `b`), and its
/// degrees of freedom.
fn pooled_t(a: &[f64], b: &[f64]) -> (f64, f64) {
    let (na, nb) = (a.len() as f64, b.len() as f64);
    let df = na + nb - 2.0;
    let (ma, mb) = (st::mean(a), st::mean(b));
    let ss = a.iter().map(|y| (y - ma) * (y - ma)).sum::<f64>() + b.iter().map(|y| (y - mb) * (y - mb)).sum::<f64>();
    let se = (ss / df * (1.0 / na + 1.0 / nb)).sqrt();
    ((ma - mb) / se, df)
}

/// The two-sided p-value of t.
fn t_p(t: f64, df: f64) -> f64 {
    (2.0 * (1.0 - Dist::StudentT { df }.cdf(t.abs()))).max(0.0).min(1.0)
}

// ----- the null distribution, built up -------------------------------------------------

/// What F is compared with: the formula alone, or with it a histogram built from the
/// data shuffled among the groups, or from new samples drawn from the truth.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Null {
    Formula,
    Shuffles,
    Samples,
}

/// Each with its name on a wide picture's button, a narrow one's, and the panel's.
const NULLS: [(Null, &str, &str); 3] =
    [(Null::Formula, "the formula alone", "formula"), (Null::Shuffles, "shuffle the groups", "shuffles"), (Null::Samples, "new samples from the truth", "new samples")];

/// How many shuffles or new samples a run takes, and how many a second.
const RUNS: usize = 2000;
const RATE: f32 = 400.0;
/// Points the F curve is drawn through.
const CURVE_POINTS: usize = 200;
/// Bars of the histogram, and the batch of lines its outline is drawn in.
const BINS: usize = 40;
const OUTLINE: u8 = plot::OWN;

/// The critical values of Student's t (two-sided, at `LEVEL`) for every number of
/// degrees of freedom two groups can have.
fn t_critical() -> Vec<f64> {
    (0..=2 * MOST).map(|df| if df == 0 { f64::INFINITY } else { Dist::StudentT { df: df as f64 }.quantile(1.0 - LEVEL / 2.0) }).collect()
}

/// A run of shuffles or new samples, and what it has found so far.
#[derive(Clone, Debug)]
struct Run {
    kind: Null,
    /// What it was started for: the design (and with it the data).
    key: Option<(Null, Design)>,
    rng: Rng,
    done: usize,
    /// The F of each, counted in bars from 0 to the chart's right end.
    hist: Histogram,
    /// Shuffles: how many reached the observed F. New samples: how many the test
    /// rejected (F beyond its critical value).
    reached: usize,
    /// New samples: how many had at least one pair of groups that a t test of the
    /// pair alone called different.
    pairs: usize,
    /// The sum of the new samples' F, for their mean.
    sum: f64,
    /// The counts as the words on the picture and in the panel say them: brought up to
    /// date every `SAID` steps and at the end. (Words with formulas in them that change
    /// in every frame would have to be set afresh in every frame.)
    said: Tally,
}

/// How often the counts in words are brought up to date (steps of a run).
const SAID: usize = 100;

/// A run's counts at some moment: what is written.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
struct Tally {
    done: usize,
    reached: usize,
    pairs: usize,
    sum: f64,
}

impl Tally {
    fn share(&self, count: usize) -> f64 {
        count as f64 / self.done.max(1) as f64
    }

    fn mean_f(&self) -> f64 {
        self.sum / self.done.max(1) as f64
    }

    /// The p-value by shuffling: the share of the shuffles, the data as they are
    /// counted among them, with an F at least as large.
    fn shuffled_p(&self) -> f64 {
        (1 + self.reached) as f64 / (1 + self.done) as f64
    }
}

impl Run {
    fn new() -> Run {
        Run { kind: Null::Formula, key: None, rng: Rng::new(1), done: 0, hist: Histogram::new(0.0, 1.0, BINS), reached: 0, pairs: 0, sum: 0.0, said: Tally::default() }
    }

    /// The counts as they are now.
    fn tally(&self) -> Tally {
        Tally { done: self.done, reached: self.reached, pairs: self.pairs, sum: self.sum }
    }

    /// Start afresh, for this kind and design, with a histogram reaching to `top`.
    fn restart(&mut self, kind: Null, design: &Design, top: f64) {
        self.kind = kind;
        self.key = Some((kind, *design));
        self.rng = Rng::new(design.seed.wrapping_mul(7919).wrapping_add(if kind == Null::Shuffles { 11 } else { 23 }));
        self.done = 0;
        self.hist = Histogram::new(0.0, top, BINS);
        (self.reached, self.pairs, self.sum) = (0, 0, 0.0);
        self.said = Tally::default();
    }

    fn pending(&self) -> bool {
        self.kind != Null::Formula && self.done < RUNS
    }

    /// One more shuffle or new sample.
    fn step(&mut self, s: &Sample, t_critical: &[f64]) {
        match self.kind {
            Null::Formula => {}
            Null::Shuffles => {
                let mut v: Vec<f64> = s.groups.iter().flatten().copied().collect();
                self.rng.shuffle(&mut v);
                let f = ratio(&v, &s.sizes);
                self.hist.add(f);
                if let Some(obs) = s.split.f {
                    // (a hair below: a shuffle that deals out the same groups has the same F)
                    if f >= obs * (1.0 - 1e-12) {
                        self.reached += 1;
                    }
                }
                self.done += 1;
                self.tell();
            }
            Null::Samples => {
                let d = &s.design;
                let mut v = Vec::with_capacity(d.total());
                for i in 0..d.k {
                    for _ in 0..d.n[i] {
                        v.push(d.mu[i] + d.sd[i] * st::normal(&mut self.rng));
                    }
                }
                let f = ratio(&v, &s.sizes);
                self.hist.add(f);
                if f > s.f_critical {
                    self.reached += 1;
                }
                self.sum += if f.is_finite() { f } else { 0.0 };
                if d.k >= 3 && some_pair(&v, &s.sizes, t_critical) {
                    self.pairs += 1;
                }
                self.done += 1;
                self.tell();
            }
        }
    }

    /// Bring the counts in words up to date, if it is time.
    fn tell(&mut self) {
        if self.done.is_multiple_of(SAID) || self.done == RUNS {
            self.said = self.tally();
        }
    }

    // (the counts as they are now, for the tests of this file)
    #[cfg(test)]
    fn shuffled_p(&self) -> f64 {
        self.tally().shuffled_p()
    }

    #[cfg(test)]
    fn share(&self, count: usize) -> f64 {
        self.tally().share(count)
    }

    #[cfg(test)]
    fn mean_f(&self) -> f64 {
        self.tally().mean_f()
    }
}

/// Did a t test of some pair of the groups alone (laid end to end in `values`) find
/// them different at `LEVEL`?
fn some_pair(values: &[f64], sizes: &[usize], critical: &[f64]) -> bool {
    let mut starts = Vec::with_capacity(sizes.len());
    let mut at = 0;
    for &n in sizes {
        starts.push(at);
        at += n;
    }
    let group = |i: usize| &values[starts[i].min(values.len())..(starts[i] + sizes[i]).min(values.len())];
    for a in 0..sizes.len() {
        for b in a + 1..sizes.len() {
            let (t, df) = pooled_t(group(a), group(b));
            if t.abs() > critical.get(df as usize).copied().unwrap_or(f64::INFINITY) {
                return true;
            }
        }
    }
    false
}

/// The sample as it is: its design, its points and their analysis.
#[derive(Clone, Debug)]
struct Sample {
    design: Design,
    groups: Vec<Vec<f64>>,
    sizes: Vec<usize>,
    split: Split,
    f_critical: f64,
    /// The F chart's right end, the height of the curve in it, and the curve: its
    /// height at 0 and at `CURVE_POINTS` places evenly across the chart (see [`curve`]).
    top: f64,
    peak: f64,
    curve: Vec<f64>,
}

impl Sample {
    fn new(design: Design, z: &Noise) -> Sample {
        let groups = points(&design, z);
        let sizes = groups.iter().map(Vec::len).collect();
        let split = Split::of(&groups);
        let null = split.null();
        let f_critical = null.quantile(1.0 - LEVEL);
        let top = chart_top(&null);
        // (its highest point past the very start, where it may be infinite)
        let peak = (1..=80).map(|i| null.pdf(top * i as f64 / 80.0)).filter(|v| v.is_finite()).fold(0.0, f64::max);
        let curve = curve(&null, top, chart_height(peak));
        Sample { design, groups, sizes, split, f_critical, top, peak, curve }
    }

    /// With two groups: the pooled t statistic, its degrees of freedom and p-value.
    fn t(&self) -> Option<(f64, f64, f64)> {
        if self.groups.len() != 2 || self.split.f.is_none() {
            return None;
        }
        let (t, df) = pooled_t(&self.groups[0], &self.groups[1]);
        t.is_finite().then(|| (t, df, t_p(t, df)))
    }
}

// ----- the experiments -----------------------------------------------------------------

/// Experiments: (button, what to look for). Their data are in [`trial`].
const TRIALS: [(&str, &str); 6] = [
    (
        "one group apart",
        r"Four groups of 10 points, three drawn from bells centred on 0 and one from a bell centred on 1.1, all of spread 1. Group 4's mean (yellow) lies far from the grand mean (dashed), so the part between the groups is large, 28.1 % of the total: $F = 4.69$, $p = 0.0072$. Dealing the same points out among the groups at random 2 000 times, as if the groups meant nothing, gives values of $F$ that follow the purple curve; only 17 of them reach 4.69. Drag group 4's bar further away: the part between grows, and $F$ with it.",
    ),
    (
        "nothing going on",
        r"Five groups of 10 from one bell. Their means still differ a little, by chance: $F = 1.02$, $p = 0.41$. Over 2 000 new samples from the same truth (green), $F$ averages 1.03 (the curve's mean is 1.05) and passes the 5 % line in 5.1 % of them: one test at 0.05, one 5 % risk of a false alarm. Testing each of the 10 pairs of groups on its own at 0.05 instead finds some pair different in 28.1 % of the same samples.",
    ),
    (
        "noisier groups",
        r"The first experiment's groups with every point twice as far from its group's mean, and the means kept where they were. The part between the groups is the same as before, 10.55; the part within is four times as large, 107.91. $F$ falls to a quarter, 1.17, and $p = 0.33$: the same differences between the means, against twice the scatter, are no evidence of anything.",
    ),
    (
        "four times the data",
        r"The first experiment's group means again, with 40 points in each group instead of 10. The part between counts each group's squared distance from the grand mean once for every point in it, so it is exactly four times as large, 42.20; the within mean square still estimates the spread squared, 1 (0.95, against 0.75). $F$ goes from 4.69 to 14.88. Over many new samples from the first experiment's truth, $F - 1$ averages 3.25 with 10 points a group and 12.25 with 40, 3.8 times as much: the same differences in the means count for more with more data.",
    ),
    (
        "two groups",
        r"With two groups $F$ is the square of Student's $t$ for the difference of their means with a pooled spread, the two-sample $t$ test: here $t = -2.60$ and $t^2 = 6.75 = F$, and the two tests give the same $p$, 0.018. Analysis of variance is that $t$ test, carried over to more groups.",
    ),
    (
        "unequal spreads",
        r"The test assumes one common spread. Here the true means are all 0, but the two small groups (6 points each) have spread 2 and the large one (24 points) spread 1. The within mean square is mostly the large, quiet group's, so it understates how far the small groups' means wander by chance: over 2 000 new samples $F$ passes the 5 % line in 19.9 % of them, about four times as often as it should. Give the large group the wide spread instead and it passes in 0.7 %, too seldom; with three groups of 12 and the same spreads, in 6.6 %.",
    ),
];

/// The scene opens as the first experiment sets it up.
const OPENING: usize = 0;
/// The true means of the first experiment, whose sample means the third and fourth keep.
const APART: [f64; 4] = [0.0, 0.0, 0.0, 1.1];

/// An experiment's data, what F is compared with in it, and for some the group means
/// the sample is moved to, as if dragged there (those of the opening sample).
fn trial(k: usize) -> (Design, Null, Option<[f64; GROUPS]>) {
    match k {
        0 => (Design::even(4, 10, 1.0, &APART, 29), Null::Shuffles, None),
        1 => (Design::even(5, 10, 1.0, &[0.0; 5], 11), Null::Samples, None),
        2 => (Design::even(4, 10, 2.0, &APART, 29), Null::Formula, Some(opening_means())),
        3 => (Design::even(4, 40, 1.0, &APART, 29), Null::Formula, Some(opening_means())),
        4 => (Design::even(2, 10, 1.0, &[0.0, 0.9], 9), Null::Formula, None),
        _ => {
            let mut d = Design::even(3, 6, 2.0, &[0.0; 3], 9);
            (d.n[2], d.sd[2]) = (24, 1.0);
            (d.kept(), Null::Samples, None)
        }
    }
}

/// The group means of the opening sample.
fn opening_means() -> [f64; GROUPS] {
    let (d, _, _) = trial(OPENING);
    let s = Sample::new(d, &noise(d.seed));
    let mut m = [0.0; GROUPS];
    for (a, b) in m.iter_mut().zip(&s.split.means) {
        *a = *b;
    }
    m
}

// ----- what is drawn, gliding ----------------------------------------------------------

/// The picture as drawn: it follows the numbers through [`Glide`]s, so that a new
/// sample, a dragged bar or a run that fills its histogram moves in every frame.
#[derive(Default)]
struct Look {
    /// The groups' chart's range up (low, high) and across (left, right); the F
    /// chart's right end and top.
    frame: Glide,
    /// Each group's points (their values).
    points: [Glide; GROUPS],
    /// The group means, the grand mean, the true means, and the highlighted point:
    /// where it is across (in groups), its value and its group's mean.
    marks: Glide,
    /// The sums and mean squares between and within, the observed F, the critical F,
    /// the true variance, and the share between.
    bars: Glide,
    /// The histogram's bars, as shares of the chart's height (like the curve's).
    hist: Glide,
    /// The F curve's heights (see [`curve`]).
    curve: Glide,
    moving: bool,
}

/// Places in [`Look::marks`].
const GRAND: usize = GROUPS;
const TRUE_MEANS: usize = GROUPS + 1;
const CHOSEN: usize = 2 * GROUPS + 1;

struct Goals {
    frame: Vec<f32>,
    points: Vec<Vec<f32>>,
    marks: Vec<f32>,
    bars: Vec<f32>,
    hist: Vec<f32>,
    curve: Vec<f32>,
}

impl Look {
    fn follow(&mut self, g: &Goals, dt: f32, snap: bool) {
        let mut moving = false;
        let mut go = |glide: &mut Glide, to: &[f32], keep: bool| {
            if snap {
                glide.snap(to);
            } else {
                moving |= glide.follow(to, dt, keep);
            }
        };
        go(&mut self.frame, &g.frame, false);
        for (glide, to) in self.points.iter_mut().zip(&g.points) {
            go(glide, to, true);
        }
        go(&mut self.marks, &g.marks, false);
        go(&mut self.bars, &g.bars, false);
        go(&mut self.hist, &g.hist, false);
        go(&mut self.curve, &g.curve, false);
        self.moving = moving;
    }
}

// ----- the layout ----------------------------------------------------------------------

/// How tall the sums of squares are on a phone, and at least elsewhere.
const SPLIT_NARROW: f32 = 166.0;
const SPLIT_LEAST: f32 = 190.0;

/// Where everything goes on a canvas (logical pixels).
#[derive(Clone, Copy, Debug, PartialEq)]
struct Layout {
    view: Vec2,
    narrow: bool,
    /// Text sizes: headings, words, small print.
    big: f32,
    text: f32,
    small: f32,
    /// The groups' chart (where the points are drawn), and its heading's corner.
    head: Vec2,
    strips: Rect,
    /// The sums of squares and mean squares.
    split: Rect,
    /// The chart of F, and its heading's corner.
    null_head: Vec2,
    null: Rect,
    /// The row of buttons that choose what F is compared with, and the words under it.
    buttons: Rect,
    counts: Rect,
    /// Too small a canvas for the pictures: the verdict alone.
    tiny: bool,
}

impl Layout {
    fn of(view: Vec2, inset: f32) -> Layout {
        let view = view.max(Vec2::splat(1.0));
        let narrow = view.x < 760.0;
        let (big, text, small) = if narrow { (13.0, 11.5, 10.5) } else { (15.0, 12.5, 11.5) };
        let side = if narrow { 10.0 } else { 20.0 };
        let numbers = if narrow { 28.0 } else { 36.0 };
        let top = inset.max(0.0) + if narrow { 6.0 } else { 12.0 };
        // a heading: its title and a line under it (and room for an axis's name)
        let heading = big + small + 22.0;
        // under the groups' chart: their names and sizes
        let names = 2.0 * small + 16.0;
        // under the F chart: its numbers and its name
        let axis = small + 14.0;
        let h = (view.y - top).max(1.0);
        let lay = |head: Vec2, strips: Rect, split: Rect, null_head: Vec2, null: Rect, buttons: Rect, counts: Rect| {
            let tiny = view.x < 230.0 || view.y < 300.0 || strips.1.y - strips.0.y < 60.0 || null.1.y - null.0.y < 40.0 || strips.1.x - strips.0.x < 120.0;
            Layout { view, narrow, big, text, small, head, strips, split, null_head, null, buttons, counts, tiny }
        };
        if narrow {
            // one under another: what has a fixed height first, and the rest shared
            // between the two charts
            let (x0, x1) = (side, view.x - side);
            let fixed = heading + names + 4.0 + SPLIT_NARROW + 16.0 + heading + axis + 6.0 + 24.0 + 8.0 + 66.0;
            let free = (h - fixed).max(20.0);
            let head = Vec2::new(x0, top);
            let s0 = top + heading;
            let strips = (Vec2::new(x0 + numbers, s0), Vec2::new(x1 - 10.0, s0 + free * 0.62));
            let split_top = strips.1.y + names + 4.0;
            let split = (Vec2::new(x0, split_top), Vec2::new(x1, split_top + SPLIT_NARROW));
            let null_head = Vec2::new(x0, split.1.y + 16.0);
            let n0 = null_head.y + heading;
            let null = (Vec2::new(x0 + 8.0, n0), Vec2::new(x1 - 4.0, n0 + free * 0.38));
            let b0 = null.1.y + axis + 6.0;
            let buttons = (Vec2::new(x0, b0), Vec2::new(x1, b0 + 24.0));
            let counts = (Vec2::new(x0, buttons.1.y + 8.0), Vec2::new(x1, view.y - 4.0));
            lay(head, strips, split, null_head, null, buttons, counts)
        } else {
            let mid = (view.x * 0.58).round();
            let (x0, x1) = (side, mid - 24.0);
            let head = Vec2::new(x0, top);
            let s0 = top + heading;
            // (half the height, less if the sums of squares under it would be squeezed)
            let tall = (h * 0.5).min(h - heading - names - 28.0 - SPLIT_LEAST).max(10.0);
            let strips = (Vec2::new(x0 + numbers, s0), Vec2::new(x1 - 14.0, s0 + tall));
            let split_top = strips.1.y + names + 14.0;
            let split = (Vec2::new(x0, split_top), Vec2::new(x1, view.y - 14.0));
            let (r0, r1) = (mid + 24.0, view.x - side);
            let null_head = Vec2::new(r0, top);
            let null = (Vec2::new(r0 + 10.0, s0), Vec2::new(r1 - 6.0, s0 + (h * 0.56).max(10.0)));
            let b0 = null.1.y + axis + 14.0;
            let buttons = (Vec2::new(r0, b0), Vec2::new(r1, b0 + 26.0));
            let counts = (Vec2::new(r0, buttons.1.y + 14.0), Vec2::new(r1, view.y - 14.0));
            lay(head, strips, split, null_head, null, buttons, counts)
        }
    }
}

/// The first of `options` that fits in `room` logical pixels at `size` (the last if none does).
fn pick(f: &Frame, options: &[String], room: f32, size: f32) -> String {
    options.iter().find(|t| f.text_size(t, size).x <= room).or(options.last()).cloned().unwrap_or_default()
}

/// The mark a legend shows for a kind of thing.
#[derive(Clone, Copy)]
enum Swatch {
    Dot(Color),
    Bar(Color),
    Dashed(Color),
}

/// A row of marks and their names from `at` (screen space), as many as fit before `right`.
fn legend(f: &mut Frame, at: Vec2, right: f32, entries: &[(Swatch, &str, &str)], size: f32) {
    // (the long names if they all fit, else the short ones)
    let width = |f: &Frame, long: bool| entries.iter().map(|e| 24.0 + 16.0 + f.text_size(if long { e.1 } else { e.2 }, size).x).sum::<f32>();
    let long = at.x + width(f, true) <= right;
    let mut x = at.x;
    let mid = at.y + size * 0.5 + 1.0;
    for (swatch, name, short) in entries {
        let name = if long { name } else { short };
        let w = f.text_size(name, size).x;
        if x + 24.0 + w > right {
            break;
        }
        match *swatch {
            Swatch::Dot(c) => plot::disc(f, Vec2::new(x + 9.0, mid), 6.0, c),
            Swatch::Bar(c) => plot::line_of(f, MARKS, Vec2::new(x, mid), Vec2::new(x + 18.0, mid), 3.0, c),
            Swatch::Dashed(c) => plot::dashed_of(f, MARKS, Vec2::new(x, mid), Vec2::new(x + 18.0, mid), 1.6, c),
        }
        f.text_px(Vec2::new(x + 24.0, at.y), *name, size, pal::DIM, Align2::LEFT_TOP);
        x += 24.0 + w + 16.0;
    }
}

/// A sum of squares, a mean square or F, with as many decimals as its size wants.
fn amount(v: f64) -> String {
    let a = v.abs();
    num(
        v,
        if a >= 1000.0 {
            0
        } else if a >= 100.0 {
            1
        } else {
            2
        },
    )
}

/// The sums of squares as they are written: between, within and total, to as many
/// decimals as the total's size wants, and the total written as the sum of the other
/// two as written, so that what is written adds up (the total is then within a unit of
/// its last digit, where rounding it on its own could put it a unit off the sum).
fn sums_text(between: f64, within: f64) -> [String; 3] {
    let total = (between + within).abs();
    let d = if total >= 10_000.0 {
        0
    } else if total >= 1000.0 {
        1
    } else {
        2
    };
    let k = 10f64.powi(d as i32);
    let (b, w) = ((between * k).round() / k, (within * k).round() / k);
    [num(b, d), num(w, d), num(b + w, d)]
}

/// A p-value as it is written: never 0.
fn p_text(p: f64) -> String {
    if !p.is_finite() {
        "?".to_string()
    } else if p < 1e-4 {
        r"< 0.0001".to_string()
    } else if p < 0.01 {
        num(p, 4)
    } else if p < 0.1 {
        num(p, 3)
    } else {
        num(p, 2)
    }
}

/// A p-value as a read-out writes it ("< 0.0001" as a formula).
fn p_value(p: f64) -> String {
    let t = p_text(p);
    if t.starts_with('<') {
        format!("${t}$")
    } else {
        t
    }
}

/// "p = 0.019" or "p < 0.0001", as a formula.
fn p_is(p: f64) -> String {
    let t = p_text(p);
    if t.starts_with('<') {
        format!("$p {t}$")
    } else {
        format!("$p = {t}$")
    }
}

/// A share in percent, to one decimal.
fn percent(share: f64) -> String {
    format!("{} %", num(100.0 * share, 1))
}

/// A whole number with a thin space between thousands (2 000).
fn whole(n: usize) -> String {
    let s = n.to_string();
    if s.len() <= 3 {
        return s;
    }
    let (head, tail) = s.split_at(s.len() - 3);
    format!("{head} {tail}")
}

/// Where point `j` of a group is across its column (in groups, from its centre): spread
/// evenly over the column by the golden ratio, the same place whatever the group's size.
fn jitter(j: usize) -> f64 {
    (((j as f64 + 1.0) * 0.618_033_988_75).fract() - 0.5) * 0.5
}

/// How far either side of its group's centre a mean's bar reaches (in groups).
const BAR: f64 = 0.36;

// ----- axes that glide -----------------------------------------------------------------

/// The round values along an axis that get a line and a number: every multiple of
/// `step` from `first` to `last` times it. They are chosen from where the axis is going
/// (its range once its glide has ended), a few beyond each end, and stay the same while
/// it glides there: each line and number keeps its place in its batch, sliding with the
/// axis, unseen while it is off the chart. (Chosen afresh from the axis as drawn, as
/// `Plot`'s own axes are, a value would come in at one end as another went out at the
/// other while the axis glides, and the lines would change partners or number from frame
/// to frame.)
#[derive(Clone, Copy, Debug, PartialEq)]
struct Steady {
    step: f64,
    first: i64,
    last: i64,
}

impl Steady {
    const NONE: Steady = Steady { step: 1.0, first: 0, last: -1 };

    /// For an axis going from `lo` to `hi` over `pixels`: a round step about sixty
    /// pixels long (as `plot::Ticks` chooses it).
    fn of(lo: f64, hi: f64, pixels: f32) -> Steady {
        let span = hi - lo;
        if !span.is_finite() || span <= 0.0 || pixels.is_nan() || pixels <= 0.0 {
            return Steady::NONE;
        }
        // (rounded to the decimal it stands for: in single precision 0.1 is a hair less)
        let step = (nice_step((span / (pixels as f64 / 60.0).max(1.0)) as f32) as f64 * 1e6).round() / 1e6;
        if step <= 0.0 {
            return Steady::NONE;
        }
        let first = (lo / step).floor() as i64 - 3;
        let last = ((hi / step).ceil() as i64 + 3).min(first + 60);
        Steady { step, first, last }
    }

    fn values(&self) -> impl Iterator<Item = f64> + '_ {
        (self.first..=self.last).map(|k| k as f64 * self.step)
    }

    /// A value written with just the digits the step needs.
    fn name(&self, v: f64) -> String {
        num(v, (-(self.step.log10() + 1e-9).floor()).max(0.0).min(6.0) as usize)
    }

    /// The axis up the left of a chart: its line, a tick, a number and a faint line
    /// across for each value (those off the chart drawn unseen at its edge).
    fn up(&self, f: &mut Frame, plot: &Plot, size: f32) {
        let (min, max) = plot.rect;
        plot::line_of(f, AXES, min, Vec2::new(min.x, max.y), 1.0, pal::DIM.a(0.8));
        for v in self.values() {
            let y = plot.px_y(v);
            let seen = y >= min.y - 0.5 && y <= max.y + 0.5;
            let y = y.max(min.y).min(max.y);
            plot::line_of(f, AXES, Vec2::new(min.x, y), Vec2::new(max.x, y), 1.0, pal::DIM.a(if seen { 0.14 } else { 0.0 }));
            plot::line_of(f, AXES, Vec2::new(min.x - 4.0, y), Vec2::new(min.x, y), 1.0, pal::DIM.a(if seen { 0.8 } else { 0.0 }));
            let name = if seen { self.name(v) } else { String::new() };
            f.text_px(Vec2::new(min.x - 6.0, y), name, size, pal::DIM, Align2::RIGHT_CENTER);
        }
    }

    /// The axis along the bottom of a chart: its line, a tick and a number for each
    /// value (those off the chart drawn unseen at its edge), and its name.
    fn across(&self, f: &mut Frame, plot: &Plot, name: &str, size: f32) {
        let (min, max) = plot.rect;
        plot::line_of(f, AXES, Vec2::new(min.x, max.y), max, 1.0, pal::DIM.a(0.8));
        for v in self.values() {
            let x = plot.px_x(v);
            let seen = x >= min.x - 0.5 && x <= max.x + 0.5;
            let x = x.max(min.x).min(max.x);
            plot::line_of(f, AXES, Vec2::new(x, max.y), Vec2::new(x, max.y + 4.0), 1.0, pal::DIM.a(if seen { 0.8 } else { 0.0 }));
            let text = if seen { self.name(v) } else { String::new() };
            f.text_px(Vec2::new(x, max.y + 6.0), text, size, pal::DIM, Align2::CENTER_TOP);
        }
        f.text_px(Vec2::new((min.x + max.x) * 0.5, max.y + 8.0 + size), name, size, pal::DIM, Align2::CENTER_TOP);
    }
}

// ----- the scene -----------------------------------------------------------------------

pub struct Anova {
    z: Box<Noise>,
    sample: Sample,
    null: Null,
    run: Run,
    pace: crate::clock::Pace,
    t_critical: Vec<f64>,
    /// Each group its own size and spread (in the panel).
    own: bool,
    show_truth: bool,
    /// The point whose distance from the grand mean is split (group, point), if one
    /// was clicked; otherwise one is chosen.
    chosen: Option<(usize, usize)>,
    /// The group whose bar is being dragged, and the groups' chart's range held still
    /// while it is.
    held: Option<usize>,
    held_range: Option<[f32; 2]>,
    look: Look,
    /// The round values on the two axes (up the groups' chart, across the chart of F),
    /// chosen from where the axes are going (see [`Steady`]); and how many lines of
    /// axes and labels the last frame drew (see the end of `frame`).
    ticks: [Steady; 2],
    drawn_last: (usize, usize),
}

impl Anova {
    /// (Nothing here needs the graphics device, so the tests can make one too.)
    fn create() -> Anova {
        let (d, null, _) = trial(OPENING);
        let z = noise(d.seed);
        let sample = Sample::new(d, &z);
        let mut s = Anova {
            z,
            sample,
            null,
            run: Run::new(),
            pace: crate::clock::Pace::default(),
            t_critical: t_critical(),
            own: false,
            show_truth: true,
            chosen: None,
            held: None,
            held_range: None,
            look: Look::default(),
            ticks: [Steady::NONE; 2],
            drawn_last: (0, 0),
        };
        s.set_up(OPENING);
        s
    }

    /// Set up one of the experiments.
    fn set_up(&mut self, k: usize) {
        let (mut d, null, means) = trial(k);
        if self.sample.design.seed != d.seed {
            self.z = noise(d.seed);
        }
        if let Some(means) = means {
            for i in 0..d.k {
                put_mean(&mut d, &self.z, i, means[i]);
            }
        }
        self.null = null;
        self.own = !(d.equal_sizes() && d.equal_spreads());
        self.show_truth = true;
        self.chosen = None;
        self.refresh(d);
    }

    /// The data for a new design, and the run started afresh if it depends on it.
    fn refresh(&mut self, d: Design) {
        let d = d.kept();
        if d.seed != self.sample.design.seed {
            self.z = noise(d.seed);
        }
        if d != self.sample.design || self.sample.groups.is_empty() {
            self.sample = Sample::new(d, &self.z);
        }
        if self.run.key != Some((self.null, d)) {
            self.run.restart(self.null, &d, self.sample.top);
            self.pace.reset();
        }
        if let Some((g, j)) = self.chosen {
            if g >= d.k || j >= d.n[g] {
                self.chosen = None;
            }
        }
    }

    /// The run to its end at once (the tests of this file).
    #[cfg(test)]
    fn settle(&mut self) {
        while self.run.pending() {
            self.run.step(&self.sample, &self.t_critical);
        }
    }

    /// The point whose distance is split: the one clicked, or else the furthest out
    /// of the group whose mean is furthest from the grand mean, on that group's side.
    fn highlighted(&self) -> Option<(usize, usize)> {
        if self.chosen.is_some() {
            return self.chosen;
        }
        let s = &self.sample.split;
        let g = (0..s.k).max_by(|&a, &b| (s.means[a] - s.grand).abs().total_cmp(&(s.means[b] - s.grand).abs()))?;
        let side = if s.means[g] >= s.grand { 1.0 } else { -1.0 };
        let pts = &self.sample.groups[g];
        let j = (0..pts.len()).max_by(|&a, &b| ((pts[a] - s.means[g]) * side).total_cmp(&((pts[b] - s.means[g]) * side)))?;
        Some((g, j))
    }

    /// The groups' chart's range up: a little beyond the points and the marks.
    fn range(&self) -> [f32; 2] {
        if let Some(r) = self.held_range {
            return r;
        }
        let d = &self.sample.design;
        let (mut lo, mut hi) = (f64::MAX, f64::MIN);
        for v in self.sample.groups.iter().flatten().chain(&self.sample.split.means) {
            lo = lo.min(*v);
            hi = hi.max(*v);
        }
        if self.show_truth {
            for m in &d.mu[..d.k] {
                lo = lo.min(*m);
                hi = hi.max(*m);
            }
        }
        if hi.is_nan() || lo.is_nan() || hi < lo {
            (lo, hi) = (-1.0, 1.0);
        }
        let pad = ((hi - lo) * 0.08).max(0.5);
        [(lo - pad) as f32, (hi + pad) as f32]
    }

    /// What the picture glides to.
    fn goals(&self) -> Goals {
        let s = &self.sample;
        let sp = &s.split;
        let d = &s.design;
        let k = d.k;
        let (top, peak) = (s.top, s.peak);
        let [lo, hi] = self.range();
        let frame = vec![lo, hi, -0.5, k as f32 - 0.5, top as f32, chart_height(peak) as f32];
        let points = (0..GROUPS).map(|i| s.groups.get(i).map(|g| g.iter().map(|v| *v as f32).collect()).unwrap_or_default()).collect();
        let mut marks = vec![sp.grand as f32; 2 * GROUPS + 4];
        for i in 0..k {
            marks[i] = sp.means[i] as f32;
            marks[TRUE_MEANS + i] = d.mu[i] as f32;
        }
        marks[GRAND] = sp.grand as f32;
        if let Some((g, j)) = self.highlighted() {
            if let (Some(y), Some(m)) = (s.groups.get(g).and_then(|p| p.get(j)), sp.means.get(g)) {
                marks[CHOSEN] = (g as f64 + jitter(j)) as f32;
                marks[CHOSEN + 1] = *y as f32;
                marks[CHOSEN + 2] = *m as f32;
            }
        }
        // (one true spread, if there is one and the truth is shown)
        let truth = if d.equal_spreads() && self.show_truth { d.sd[0] * d.sd[0] } else { f64::NAN };
        let f_obs = sp.f.unwrap_or(0.0).min(top * 1.5);
        let bars = vec![sp.between as f32, sp.within as f32, sp.ms_b as f32, sp.ms_w as f32, f_obs as f32, s.f_critical as f32, truth as f32, sp.share().unwrap_or(0.0) as f32];
        // (the histogram's bars too as shares of the chart's height, kept inside it)
        let high = chart_height(peak);
        let hist = (0..BINS).map(|i| if self.run.kind == Null::Formula { 0.0 } else { (self.run.hist.density(i) / high).min(1.0) as f32 }).collect();
        let curve = s.curve.iter().map(|v| *v as f32).collect();
        Goals { frame, points, marks, bars, hist, curve }
    }

    /// The groups' chart as it is drawn now.
    fn strips_plot(&self, lay: &Layout) -> Plot {
        let r = &self.look.frame.shown;
        let g = |i: usize| r.get(i).copied().unwrap_or(0.0) as f64;
        Plot::new(lay.strips, [g(2), g(3)], [g(0), g(1)])
    }

    /// The F chart as it is drawn now.
    fn null_plot(&self, lay: &Layout) -> Plot {
        let r = &self.look.frame.shown;
        let g = |i: usize| r.get(i).copied().unwrap_or(1.0) as f64;
        Plot::new(lay.null, [0.0, g(4)], [0.0, g(5)])
    }

    /// The yellow bars, dragged up and down; a click on a point highlights it.
    fn handles(&mut self, f: &mut Frame, lay: &Layout) {
        let plot = self.strips_plot(lay);
        let k = self.sample.design.k;
        let mut held = None;
        let mut design = self.sample.design;
        for i in 0..k {
            let at = self.look.marks.get(i) as f64;
            if let Some((_, y)) = plot.handle(f, 40 + i as u32, i as f64, at, ESTIMATE) {
                held = Some(i);
                put_mean(&mut design, &self.z, i, y);
            }
        }
        if f.handles.dragging() && held.is_none() {
            held = self.held;
        }
        if !f.handles.dragging() {
            held = None;
        }
        match (held, self.held) {
            (Some(_), None) => self.held_range = Some([self.look.frame.get(0), self.look.frame.get(1)]),
            (None, Some(_)) => self.held_range = None,
            _ => {}
        }
        self.held = held;
        if design != self.sample.design {
            self.refresh(design);
        }
        // a click near a point highlights it
        if let Some(p) = plot::pointer(f).filter(|p| plot::within(*p, lay.strips, 6.0)) {
            f.handles.claim_clicks();
            if f.ptr.clicked && !f.handles.on_handle() {
                let mut best = (14.0f32, None);
                for (g, glide) in self.look.points.iter().enumerate().take(k) {
                    for (j, v) in glide.shown.iter().enumerate() {
                        let at = plot.at(g as f64 + jitter(j), *v as f64);
                        let d = (at - p).length();
                        if d < best.0 {
                            best = (d, Some((g, j)));
                        }
                    }
                }
                if best.1.is_some() {
                    self.chosen = best.1;
                }
            }
        }
    }

    /// The work of the run: so many shuffles or samples a second of the scene's time
    /// (none while a bar is held), within the frame's allowance.
    fn work(&mut self, dt: f32, real_dt: f32) {
        if self.held.is_some() || dt <= 0.0 || !self.run.pending() {
            return;
        }
        let mut turn = scene_turn(&mut self.pace, RATE, dt, real_dt);
        while self.run.pending() && turn.next() {
            self.run.step(&self.sample, &self.t_critical);
        }
        self.pace.end(turn);
    }
}

/// How high the chart of F reaches, for a curve whose highest point is `peak`.
fn chart_height(peak: f64) -> f64 {
    (peak * 1.18).max(0.1)
}

/// Where the F curve is drawn through, across the chart: at 0, and at `CURVE_POINTS`
/// places evenly between (as shares of the chart's width).
fn curve_x(i: usize) -> f64 {
    if i == 0 {
        0.0
    } else {
        (i as f64 - 0.5) / CURVE_POINTS as f64
    }
}

/// The F curve's heights at those places, on a chart reaching to `top` and `high`, as
/// shares of the chart's height. A height above the chart is put at its top (with one
/// or two degrees of freedom the curve starts from as high as infinity): cut there
/// instead, the curve would gain and lose points as it changed, and could not be
/// followed from one frame to the next. These shares glide when the degrees of freedom
/// change, so that the curve moves evenly from one shape to the other (gliding the
/// degrees of freedom themselves, the curve near 0 leaps as the first of them passes 2),
/// and as shares they keep inside the chart while its height glides too.
fn curve(null: &Dist, top: f64, high: f64) -> Vec<f64> {
    (0..=CURVE_POINTS)
        .map(|i| {
            let y = null.pdf(top * curve_x(i)) / high;
            if y.is_nan() {
                0.0
            } else {
                y.max(0.0).min(1.0)
            }
        })
        .collect()
}

/// The right end of the chart of F: far enough to show the curve's tail.
fn chart_top(null: &Dist) -> f64 {
    let q = null.quantile(0.995);
    if q.is_finite() {
        (q * 1.15).max(3.0).min(40.0)
    } else {
        6.0
    }
}

impl Scene for Anova {
    fn view(&self) -> View {
        View::Flat { center: Vec2::ZERO, half_height: 1.0 }
    }

    fn settings(&self, s: &mut ViewSettings) {
        s.grid = false;
        s.bloom = s.bloom.min(0.2);
    }

    fn animated(&self) -> bool {
        self.look.moving || self.run.pending() || self.held.is_some()
    }

    fn pointer_hint(&self) -> Option<&'static str> {
        Some("yellow dots: drag a group's mean up or down    a point: click it to split its distance    buttons on the picture: click")
    }

    fn ui(&mut self, ui: &mut egui::Ui, _: &Gfx) {
        ui::note(ui, r"Several groups, and one question: are their means further apart than the scatter within the groups would put them by chance? Analysis of variance answers it by splitting the spread of all the points about the grand mean $\bar y$ in two, exactly:");
        ui::equation(ui, r"\sum_{i,j} (y_{ij} - \bar y)^2 \\ = \sum_i n_i (\bar y_i - \bar y)^2 + \sum_{i,j} (y_{ij} - \bar y_i)^2");
        ui::note(ui, r"that is, $\text{total} = \text{between} + \text{within}$, where $y_{ij}$ is point $j$ of group $i$, $\bar y_i$ its group's mean and $n_i$ the group's size. Each part divided by its degrees of freedom is a mean square, and their ratio is $F$:");
        ui::equation(ui, r"F = \frac{\text{between} / (k - 1)}{\text{within} / (N - k)}");
        ui::note(ui, r"for $k$ groups and $N$ points. If the points are independent, the groups share one spread $\sigma$ and their true means are all equal, both mean squares estimate $\sigma^2$, and $F$ tends to be near 1. If, moreover, each group is bell-shaped, $F$ follows the $F$ distribution with $k - 1$ and $N - k$ degrees of freedom, and its tail beyond the $F$ observed is the $p$-value.");

        ui::section(ui, "Things to try");
        for (k, trial) in TRIALS.iter().enumerate() {
            if ui::experiment(ui, trial.0, trial.1) {
                self.set_up(k);
            }
        }

        ui::section(ui, "The groups");
        let mut d = self.sample.design;
        let mut k = d.k;
        if ui::slider_int(ui, "groups", &mut k, 2..=GROUPS) {
            // a new group comes from the same bell as the others, on average
            let mean = d.mu[..d.k].iter().sum::<f64>() / d.k as f64;
            for i in d.k..k {
                d.mu[i] = mean;
                if !self.own {
                    (d.n[i], d.sd[i]) = (d.n[0], d.sd[0]);
                }
            }
            d.k = k;
        }
        if ui::checkbox(ui, &mut self.own, "each group its own size and spread").changed() && !self.own {
            d.n = [d.n[0]; GROUPS];
            d.sd = [d.sd[0]; GROUPS];
        }
        if self.own {
            egui::Grid::new("anova groups").num_columns(3).spacing([14.0, 4.0]).show(ui, |ui| {
                ui.label(ui::label_text(ui, "group", 12.0, Some(pal::DIM), false));
                ui.label(ui::label_text(ui, "points", 12.0, Some(pal::DIM), false));
                ui.label(ui::label_text(ui, r"spread $\sigma$", 12.0, Some(pal::DIM), false));
                ui.end_row();
                for i in 0..d.k {
                    ui.label(ui::label_text(ui, &format!("{}", i + 1), 12.5, Some(pal::TEXT), false));
                    ui.add(egui::DragValue::new(&mut d.n[i]).range(FEWEST..=MOST).speed(0.2));
                    ui.add(egui::DragValue::new(&mut d.sd[i]).range(0.0..=WIDEST).speed(0.01).max_decimals(2));
                    ui.end_row();
                }
            });
        } else {
            let mut n = d.n[0];
            if ui::slider_int(ui, "points in each group", &mut n, FEWEST..=MOST) {
                d.n = [n; GROUPS];
            }
            let mut sd = d.sd[0] as f32;
            if ui::slider(ui, r"spread $\sigma$ of each group", &mut sd, 0.0..=WIDEST as f32) {
                d.sd = [sd as f64; GROUPS];
            }
        }
        ui::note(ui, r"Each group is drawn from a bell of its own mean and spread, with two points at least: one alone says nothing about the scatter within its group. Drag a group's yellow bar to move the whole group (and its true mean) up or down.");
        ui.horizontal(|ui| {
            if ui.button(ui::label_text(ui, "draw again", 12.5, None, false)).clicked() {
                d.seed = d.seed.wrapping_add(1);
            }
            ui::checkbox(ui, &mut self.show_truth, "show the truth");
        });
        ui::section(ui, r"What $F$ is compared with");
        let mut which = NULLS.iter().position(|n| n.0 == self.null).unwrap_or(0);
        let names: Vec<&str> = NULLS.iter().map(|n| n.2).collect();
        if ui::choice(ui, &names, &mut which) {
            self.null = NULLS[which].0;
        }
        ui::note(ui, r"Shuffles deal the same points out among the groups at random, as if the groups meant nothing. New samples are drawn from the truth as it is set: with equal true means, every rejection among them is a false alarm.");
        self.refresh(d);

        ui::section(ui, "Reading the picture");
        ui::key(
            ui,
            &[
                (DATA, "cyan: the points, and the part of the spread within the groups (each point from its group's mean)"),
                (ESTIMATE, "yellow: the group means, the part of the spread between them (each mean from the grand mean), and the $F$ worked out from them"),
                (pal::WHITE, r"white: the grand mean $\bar y$ of all the points (dashed), and the ring round the point whose distance from it is split in two"),
                (TRUTH, "green: the true means and spread, and the $F$ of new samples drawn from them"),
                (ASSUMED, r"purple: the $F$ distribution, what $F$ does when the true means are equal, and the $F$ of shuffled groups"),
                (ALARM, r"pink: the tail beyond the observed $F$, whose area is $p$"),
            ],
        );
        ui::section(ui, "Words");
        ui::term(ui, "Grand mean:", r"the mean of all the points together, $\bar y$.");
        ui::term(
            ui,
            "Sums of squares:",
            r"sums of squared distances. Total: of every point from the grand mean. Between: of each group's mean from the grand mean, counted once for each point in the group. Within: of every point from its own group's mean. $\text{total} = \text{between} + \text{within}$, exactly, for any data.",
        );
        ui::term(
            ui,
            "Degrees of freedom:",
            r"how many of the distances are free to vary: $k - 1$ between (the group means' distances from the grand mean, weighted by size, add up to 0) and $N - k$ within (in each group the points' distances from its mean add up to 0).",
        );
        ui::term(ui, "Mean square:", "a sum of squares divided by its degrees of freedom: an estimate of a variance.");
        ui::term(ui, r"$F$ ratio:", "the mean square between the groups over the mean square within them.");
        ui::term(ui, r"$p$-value:", r"the chance, if the true means were all equal, of an $F$ at least as large as the one observed.");
        ui::term(
            ui,
            r"Share of the variation explained ($\eta^2$):",
            r"between over total. It is not 0 when the true means are equal: with independent, bell-shaped points of one spread it is then $(k - 1)/(N - 1)$ on average.",
        );

        ui::section(ui, "Numbers");
        let sp = &self.sample.split;
        ui::readout(ui, r"groups $k$, points $N$", format!("{}, {}", sp.k, sp.total));
        let [between, within, total] = sums_text(sp.between, sp.within);
        ui::readout(ui, "sum of squares between", between);
        ui::readout(ui, "sum of squares within", within);
        ui::readout(ui, "sum of squares, total", total);
        ui::readout(ui, r"degrees of freedom: $k - 1$, $N - k$", format!("{}, {}", sp.df_b, sp.df_w));
        ui::readout(ui, "mean squares: between, within", format!("{}, {}", amount(sp.ms_b), amount(sp.ms_w)));
        match (sp.f, sp.p) {
            (Some(fv), Some(p)) => {
                ui::readout(ui, r"$F$", amount(fv));
                ui::readout(ui, r"$p$, by the formula", p_value(p));
            }
            _ => ui::readout(ui, r"$F$", "none: no scatter within"),
        }
        ui::readout(ui, r"share between, $\eta^2$", sp.share().map_or("none".to_string(), percent));
        if let Some((t, df, p)) = self.sample.t() {
            ui::readout(ui, r"$t$ (pooled spread), $t^2$", format!("{}, {}", num(t, 2), amount(t * t)));
            ui::readout(ui, &format!(r"$p$ by $t$ with {df} degrees of freedom"), p_value(p));
        }
        let r = &self.run.said;
        match self.run.kind {
            Null::Shuffles if r.done > 0 => ui::readout(ui, "$p$ by shuffling", format!("{} ({} shuffles)", p_value(r.shuffled_p()), whole(r.done))),
            Null::Samples if r.done > 0 => {
                ui::readout(ui, r"new samples: $F$ past the 5 % line", format!("{} of {}", percent(r.share(r.reached)), whole(r.done)));
                ui::readout(ui, r"  their mean $F$", amount(r.mean_f()));
                if self.sample.design.k >= 3 {
                    ui::readout(ui, "  some pair of groups different alone", percent(r.share(r.pairs)));
                }
            }
            _ => {}
        }
        ui::see_also(
            ui,
            &[
                ("twogroups", r"two groups, where $F$ is $t^2$"),
                ("pvalues", "many tests at once, and peeking"),
                ("multireg", "regression: the same split into what is explained and what is left over"),
                ("regress", "fitting a line by least squares"),
            ],
        );
    }

    fn frame(&mut self, f: &mut Frame) {
        let lay = Layout::of(f.view_size(), f.top_inset);
        self.work(f.dt, f.real_dt);
        let goals = self.goals();
        // (what glides is in the data's own units, whatever the canvas: only the first
        // frame has nothing to glide from)
        let snap = self.look.frame.shown.is_empty();
        self.look.follow(&goals, f.real_dt, snap);
        // (up the groups' chart about one every forty pixels, as `Plot`'s own axes have them)
        let ticks = [Steady::of(goals.frame[0] as f64, goals.frame[1] as f64, (lay.strips.1.y - lay.strips.0.y) * 1.6), Steady::of(0.0, goals.frame[4] as f64, lay.null.1.x - lay.null.0.x)];
        let new_ticks = ticks != self.ticks;
        self.ticks = ticks;
        // the bars taken hold of where they are drawn in this frame, on the chart as it
        // is drawn in it (else, while the chart glides, a bar's dot would lag its bar)
        if lay.tiny {
            (self.held, self.held_range) = (None, None);
        } else {
            self.handles(f, &lay);
        }
        // (a group held in the hand is drawn where the hand has just put it)
        if let Some(i) = self.held {
            let goals = self.goals();
            if let (Some(g), Some(to)) = (self.look.points.get_mut(i), goals.points.get(i)) {
                g.snap(to);
            }
            if self.look.marks.shown.len() == goals.marks.len() {
                self.look.marks.shown[i] = goals.marks[i];
            }
        }
        if lay.tiny {
            // (a canvas squeezed to a sliver is one the controls have opened over, on a
            // phone: nothing is drawn there)
            if lay.view.x >= 160.0 && lay.view.y >= 120.0 {
                self.draw_tiny(f, &lay);
            }
            return;
        }
        self.draw_strips(f, &lay);
        self.draw_split(f, &lay);
        self.draw_null(f, &lay);
        // new round values on an axis: make sure the axes' batch and the labels are not
        // as many as in the last frame, so that nothing takes the new ones for the old
        // ones moved
        let count = |f: &mut Frame| (plot::apart(f, AXES, false).pts.len(), f.dl.labels.len());
        if new_ticks {
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
    }
}

// ----- drawing -------------------------------------------------------------------------

impl Anova {
    fn draw_tiny(&self, f: &mut Frame, lay: &Layout) {
        let mid = lay.view * 0.5;
        let sp = &self.sample.split;
        let line = match (sp.f, sp.p) {
            (Some(fv), Some(p)) => format!("$F = {}$, {}", amount(fv), p_is(p)),
            _ => "no scatter within the groups".to_string(),
        };
        let y = mid.y.max(f.top_inset + 20.0);
        f.text_px(Vec2::new(mid.x, y), line, lay.text, ESTIMATE, Align2::CENTER_CENTER);
        f.text_px(Vec2::new(mid.x, y + lay.text + 8.0), "(a larger window shows the groups)", lay.small, pal::DIM, Align2::CENTER_CENTER);
    }

    /// The groups side by side: their points, means and true means, the grand mean,
    /// and one point's distance from it split in two.
    fn draw_strips(&self, f: &mut Frame, lay: &Layout) {
        let plot = self.strips_plot(lay);
        let d = &self.sample.design;
        let k = d.k;
        let (min, max) = lay.strips;
        let width = max.x - lay.head.x;
        let names_h = 2.0 * lay.small + 12.0;
        plot.backing(f, min.x - lay.head.x + 6.0, names_h + 4.0, 8.0, 8.0);

        // the heading, and what the marks are
        let title = [
            format!("{k} groups of points, each with its mean, and the grand mean of them all"),
            format!("{k} groups, their means and the grand mean"),
            format!("{k} groups"),
        ];
        let title = pick(f, &title, width, lay.big);
        f.text_px(lay.head, title, lay.big, pal::TEXT, Align2::LEFT_TOP);
        let entries: [(Swatch, &str, &str); 4] = [
            (Swatch::Dot(DATA), "points", "points"),
            (Swatch::Bar(ESTIMATE), "a group's mean: drag it", "mean: drag it"),
            (Swatch::Dashed(pal::WHITE), r"grand mean $\bar y$", r"grand mean $\bar y$"),
            (Swatch::Dashed(TRUTH), "true mean", "true mean"),
        ];
        let shown = if self.show_truth { 4 } else { 3 };
        legend(f, lay.head + Vec2::new(0.0, lay.big + 7.0), max.x, &entries[..shown], lay.small);

        self.ticks[0].up(f, &plot, lay.small);
        // (the axis's name over its numbers: what each point is)
        f.text_px(Vec2::new(min.x - 8.0, min.y - 1.0), "$y$", lay.small, pal::DIM, Align2::RIGHT_BOTTOM);
        let column = plot.width() / (plot.x[1] - plot.x[0]).max(1.0) as f32;
        let dot = if lay.narrow { 4.0 } else { 5.0 }.min(column * 0.12).max(2.5);
        let marks = &self.look.marks;

        for i in 0..k {
            let x = i as f64;
            // the true mean
            if self.show_truth {
                let m = marks.get(TRUE_MEANS + i) as f64;
                let (a, b) = (plot.at(x - BAR, m), plot.at(x + BAR, m));
                plot::dashed_of(f, MARKS, a, b, 1.6, TRUTH.a(0.85));
            }
            // the points
            let pts = &self.look.points[i].shown;
            for (j, v) in pts.iter().enumerate() {
                let at = plot.at(x + jitter(j), *v as f64);
                plot::disc(f, at.max(min).min(max), dot, DATA.a(0.85));
            }
            // the mean
            let m = marks.get(i) as f64;
            plot::line_of(f, MARKS, plot.at(x - BAR, m), plot.at(x + BAR, m), 3.0, ESTIMATE);
            // its name and size under it
            let cx = plot.px_x(x);
            f.text_px(Vec2::new(cx, max.y + 6.0), format!("group {}", i + 1), lay.small, pal::TEXT, Align2::CENTER_TOP);
            let size = format!(r"$n = {}$", d.n[i]);
            let size = if self.own && !d.equal_spreads() { pick(f, &[format!(r"$n = {}$, $\sigma = {}$", d.n[i], num(d.sd[i], 1)), size], column - 6.0, lay.small) } else { size };
            f.text_px(Vec2::new(cx, max.y + 9.0 + lay.small), size, lay.small, pal::DIM, Align2::CENTER_TOP);
        }

        // the grand mean
        let grand = marks.get(GRAND) as f64;
        let gy = plot.px_y(grand);
        let keep = |y: f32| y.max(min.y).min(max.y);
        plot::dashed_of(f, MARKS, Vec2::new(min.x, keep(gy)), Vec2::new(max.x, keep(gy)), 1.4, pal::WHITE.a(if gy >= min.y && gy <= max.y { 0.8 } else { 0.0 }));
        // (its name just outside the chart's right edge, where nothing else is)
        f.text_px(Vec2::new(max.x + 3.0, keep(gy)), r"$\bar y$", lay.small, pal::WHITE, Align2::LEFT_CENTER);

        // the highlighted point, its distance from the grand mean split in two
        if let (Some(_), true) = (self.highlighted(), marks.shown.len() > CHOSEN + 2) {
            let (cx, cy, cm) = (marks.get(CHOSEN) as f64, marks.get(CHOSEN + 1) as f64, marks.get(CHOSEN + 2) as f64);
            let p = plot.at(cx, cy);
            let x1 = p.x + dot * 0.5 + 6.0;
            let x2 = x1 + 6.0;
            let (yg, ym) = (plot.px_y(grand), plot.px_y(cm));
            plot::line_of(f, MARKS, Vec2::new(x1, keep(yg)), Vec2::new(x1, keep(ym)), 3.0, ESTIMATE);
            plot::line_of(f, MARKS, Vec2::new(x1 - 3.0, keep(ym)), Vec2::new(x2 + 3.0, keep(ym)), 1.4, ESTIMATE);
            plot::line_of(f, MARKS, Vec2::new(x2, keep(ym)), Vec2::new(x2, keep(p.y)), 3.0, DATA);
            plot::line_of(f, MARKS, Vec2::new(x1 - 3.0, keep(yg)), Vec2::new(x1 + 3.0, keep(yg)), 1.4, pal::WHITE);
            plot::disc(f, p.max(min).min(max), dot + 5.0, pal::WHITE);
            plot::disc(f, p.max(min).min(max), dot + 1.0, DATA);
            // the two pieces named beside them: to the right if there is room before the
            // chart's edge, else to the left (gliding when the side changes)
            let short = lay.narrow || (yg - ym).abs() < 2.4 * lay.small || (ym - p.y).abs() < 2.4 * lay.small;
            let (between, within) = if short { ("between", "within") } else { ("between groups", "within its group") };
            let widest = f.text_size(within, lay.small).x.max(f.text_size(between, lay.small).x);
            let right = x2 + 10.0 + widest < max.x;
            for (n, (name, y, length, color)) in [(between, 0.5 * (yg + ym), yg - ym, ESTIMATE), (within, 0.5 * (ym + p.y), ym - p.y, DATA)].into_iter().enumerate() {
                // (a piece of no length is not named: there is nothing to name)
                let name = if length.abs() < 4.0 { "" } else { name };
                let half = f.text_size(name, lay.small).x * 0.5;
                let offset = if right { Vec2::new(x2 - p.x + 6.0 + half, 0.0) } else { Vec2::new(-(dot + 6.0 + half), 0.0) };
                f.text_placed("anova piece", n as u32, Vec2::new(p.x, keep(y)), offset, name, lay.small, color);
            }
        }
    }

    /// The sums of squares, split; the mean squares; F.
    fn draw_split(&self, f: &mut Frame, lay: &Layout) {
        let (min, max) = lay.split;
        hud_panel(f, min - Vec2::new(8.0, 6.0), max + Vec2::new(8.0, 6.0));
        let sp = &self.sample.split;
        let b = &self.look.bars;
        let w = max.x - min.x;
        let (text, small) = (lay.text, lay.small);
        let bar_h = if lay.narrow { 14.0 } else { 20.0 };
        let row_h = if lay.narrow { 12.0 } else { 18.0 };
        // (what the rows take, and the room left over shared out between them)
        let rows = 2.0 * (text + 8.0) + (small + 5.0) + (bar_h + 5.0) + small + (2.0 * row_h + 6.0) + (5.0 + small) + lay.big;
        let gap = ((max.y - min.y - rows) / 3.0).max(6.0).min(34.0);
        let mut y = min.y;
        // the sums of squares: one bar, split
        let head = [
            "sums of squares: each point's two pieces squared, and added up over all the points".to_string(),
            "sums of squares: the pieces squared and added up".to_string(),
            "sums of squares".to_string(),
        ];
        f.text_px(Vec2::new(min.x, y), pick(f, &head, w, text), text, pal::TEXT, Align2::LEFT_TOP);
        y += text + 8.0;
        let (between, within) = (b.get(0).max(0.0), b.get(1).max(0.0));
        let all = between + within;
        let split_x = if all > 0.0 { min.x + w * between / all } else { min.x };
        let [ss_b, ss_w, ss_t] = sums_text(sp.between, sp.within);
        f.text_px(Vec2::new(min.x, y), format!("between groups {ss_b}"), small, ESTIMATE, Align2::LEFT_TOP);
        f.text_px(Vec2::new(max.x, y), format!("within groups {ss_w}"), small, DATA, Align2::RIGHT_TOP);
        y += small + 5.0;
        hud_rect(f, Vec2::new(min.x, y), Vec2::new(max.x, y + bar_h), Color::hex(0x161C2B));
        if all > 0.0 {
            hud_rect_at(f, 2, Vec2::new(min.x, y), Vec2::new(split_x, y + bar_h), ESTIMATE.a(0.85));
            hud_rect_at(f, 2, Vec2::new(split_x, y), Vec2::new(max.x, y + bar_h), DATA.a(0.55));
        }
        y += bar_h + 5.0;
        let total = match sp.share() {
            Some(share) => [
                format!("total: ${ss_t} = {ss_b} + {ss_w}$; the share between groups is {}", percent(share)),
                format!("total {ss_t}; share between {}", percent(share)),
            ],
            None => ["total 0: every point is the same, nothing to split".to_string(), "total 0: nothing varies".to_string()],
        };
        f.text_px(Vec2::new(min.x, y), pick(f, &total, w, small), small, pal::TEXT, Align2::LEFT_TOP);
        y += small + gap;

        // the mean squares
        let head = [
            r"mean squares: each sum over its degrees of freedom".to_string(),
            r"mean squares: sum / degrees of freedom".to_string(),
            "mean squares".to_string(),
        ];
        f.text_px(Vec2::new(min.x, y), pick(f, &head, w, text), text, pal::TEXT, Align2::LEFT_TOP);
        y += text + 8.0;
        let names_w = if lay.narrow { 52.0 } else { 70.0 };
        let words_w = if lay.narrow { 112.0 } else { 160.0 };
        let x0 = min.x + names_w;
        let room = (w - names_w - words_w).max(10.0);
        let (ms_b, ms_w, truth) = (b.get(2).max(0.0), b.get(3).max(0.0), b.get(6));
        let most = ms_b.max(ms_w).max(if truth.is_finite() { truth } else { 0.0 }).max(1e-9);
        let scale = room / most;
        for (r, (name, ms, ss, df, color)) in [("between", ms_b, &ss_b, sp.df_b, ESTIMATE), ("within", ms_w, &ss_w, sp.df_w, DATA)].into_iter().enumerate() {
            let yy = y + r as f32 * (row_h + 6.0);
            f.text_px(Vec2::new(min.x, yy + row_h * 0.5), name, small, color, Align2::LEFT_CENTER);
            hud_rect(f, Vec2::new(x0, yy), Vec2::new(x0 + room, yy + row_h), Color::hex(0x161C2B));
            let end = x0 + (ms * scale).max(0.0).min(room);
            if end > x0 + 0.5 {
                hud_rect_at(f, 2, Vec2::new(x0, yy), Vec2::new(end, yy + row_h), color.a(if r == 0 { 0.85 } else { 0.55 }));
            }
            let words = format!("${ss} / {df} = {}$", amount(if r == 0 { sp.ms_b } else { sp.ms_w }));
            f.text_px(Vec2::new(x0 + room + 8.0, yy + row_h * 0.5), words, small, color, Align2::LEFT_CENTER);
        }
        // the true variance, which both estimate when the true means are equal
        let tick = x0 + (truth * scale).max(0.0).min(room);
        let bottom = y + 2.0 * row_h + 6.0;
        if truth.is_finite() {
            plot::line_of(f, MARKS, Vec2::new(tick, y - 4.0), Vec2::new(tick, bottom + 4.0), 2.0, TRUTH);
        }
        let truth_words = if self.sample.design.equal_spreads() && self.show_truth { format!(r"true $\sigma^2 = {}$", num(self.sample.design.sd[0].powi(2), 2)) } else { String::new() };
        f.text_px(Vec2::new(tick, bottom + 5.0), truth_words, small, TRUTH.a(if truth.is_finite() { 1.0 } else { 0.0 }), Align2::CENTER_TOP);
        y = bottom + 5.0 + small + gap;

        // F
        let line = match (sp.f, sp.p) {
            (Some(fv), Some(p)) => {
                let p = tint(ALARM, &p_is(p).replace('$', ""));
                let full = format!(r"$F = {} / {} = {}$, ${p}$", amount(sp.ms_b), amount(sp.ms_w), amount(fv));
                pick(f, &[full, format!(r"$F = {}$, ${p}$", amount(fv))], w, lay.big)
            }
            _ if sp.still() => "every point is the same: there is nothing to compare".to_string(),
            _ => r"no scatter within the groups: $F$ cannot be formed".to_string(),
        };
        f.text_px(Vec2::new(min.x, y), line, lay.big, ESTIMATE, Align2::LEFT_TOP);
    }

    /// The F distribution, the observed F and its tail; the shuffles or new samples,
    /// and what they have found.
    fn draw_null(&mut self, f: &mut Frame, lay: &Layout) {
        let plot = self.null_plot(lay);
        let sp = &self.sample.split;
        let (min, max) = lay.null;
        let width = lay.buttons.1.x - lay.null_head.x;
        plot.backing(f, min.x - lay.null_head.x + 6.0, lay.small + 20.0, 8.0, 8.0);
        let title = ["how large $F$ is when the true means are all equal".to_string(), "$F$ when the true means are equal".to_string(), "$F$".to_string()];
        f.text_px(lay.null_head, pick(f, &title, width, lay.big), lay.big, pal::TEXT, Align2::LEFT_TOP);
        let sub = [
            format!(r"the $F$ distribution with ${} = {}$ and ${} = {}$ degrees of freedom", r"k - 1", sp.df_b, r"N - k", sp.df_w),
            format!(r"$F$ distribution, ${}$ and ${}$ degrees of freedom", sp.df_b, sp.df_w),
            String::new(),
        ];
        f.text_px(lay.null_head + Vec2::new(0.0, lay.big + 6.0), pick(f, &sub, width, lay.small), lay.small, pal::DIM, Align2::LEFT_TOP);

        let top = plot.x[1];
        // the curve as drawn, and its height anywhere along it (between its points)
        let heights: Vec<f64> = self.look.curve.shown.iter().map(|v| (*v as f64).max(0.0).min(1.0) * plot.y[1]).collect();
        let points: Vec<(f64, f64)> = heights.iter().enumerate().map(|(i, h)| (top * curve_x(i), *h)).collect();
        let pdf = |x: f64| {
            let at = (x / top.max(1e-9) * CURVE_POINTS as f64 + 0.5).max(0.0);
            let i = (at.floor() as usize).min(CURVE_POINTS.saturating_sub(1));
            match (points.get(i), points.get(i + 1)) {
                (Some(a), Some(b)) if b.0 > a.0 => a.1 + (b.1 - a.1) * ((x - a.0) / (b.0 - a.0)).max(0.0).min(1.0),
                (Some(a), _) => a.1,
                _ => 0.0,
            }
        };
        // the tail beyond the observed F
        let b = &self.look.bars;
        let f_obs = b.get(4) as f64;
        if sp.f.is_some() {
            plot.area(f, f_obs, top, 0.0, 60, ALARM.a(0.35), pdf);
        }
        // the histogram, as its outline alone (a fill would hide the tail under it)
        // (its bars are shares of the chart's height, kept below its top: cut there, the
        // outline would gain and lose points from frame to frame, and could not be
        // followed from one to the next)
        let hist: Vec<f64> = self.look.hist.shown.iter().map(|v| (*v as f64).max(0.0).min(1.0) * plot.y[1]).collect();
        let hist_color = if self.run.kind == Null::Samples { TRUTH } else { ASSUMED };
        let width = top / BINS as f64;
        let outline = std::iter::once((0.0, 0.0))
            .chain(hist.iter().enumerate().flat_map(|(i, h)| [(i as f64 * width, *h), ((i + 1) as f64 * width, *h)]))
            .chain(std::iter::once((top, 0.0)));
        plot.path(f, OUTLINE, outline, 1.6, hist_color.a(if hist.iter().any(|h| *h > 0.0) { 0.9 } else { 0.0 }));
        plot.path(f, CURVES, points.iter().copied(), 2.4, ASSUMED);
        self.ticks[1].across(f, &plot, "$F$", lay.small);
        // the 5 % line
        let crit = b.get(5) as f64;
        plot.vline(f, MARKS, crit, 1.2, pal::DIM, true);
        plot.text(f, crit, plot.y[1], Vec2::new(4.0, 2.0), "5 % beyond", lay.small, pal::DIM, Align2::LEFT_TOP);
        // the observed F, and p
        let (label, p_label) = match (sp.f, sp.p) {
            (Some(fv), Some(p)) => (format!("$F = {}$", amount(fv)), p_is(p)),
            _ => (String::new(), String::new()),
        };
        if sp.f.is_some() {
            let x = f_obs.min(top);
            let px = plot.px_x(x);
            if f_obs <= top {
                plot.vline(f, MARKS, x, 2.4, ESTIMATE, false);
            } else {
                // beyond the chart: an arrow at its edge
                plot::line_of(f, MARKS, Vec2::new(max.x - 22.0, min.y + 14.0), Vec2::new(max.x - 2.0, min.y + 14.0), 2.4, ESTIMATE);
                plot::line_of(f, MARKS, Vec2::new(max.x - 8.0, min.y + 9.0), Vec2::new(max.x - 2.0, min.y + 14.0), 2.4, ESTIMATE);
            }
            let right = px + 8.0 + f.text_size(&label, lay.text).x < max.x;
            let (lx, align) = if right { (px + 6.0, Align2::LEFT_TOP) } else { (px - 6.0, Align2::RIGHT_TOP) };
            f.text_px(Vec2::new(lx, min.y + 22.0), label, lay.text, ESTIMATE, align);
            f.text_px(Vec2::new(lx, min.y + 26.0 + lay.text), p_label, lay.text, ALARM, align);
        } else {
            f.text_px(Vec2::new(max.x - 6.0, min.y + 22.0), "", lay.text, ESTIMATE, Align2::RIGHT_TOP);
            f.text_px(Vec2::new(max.x - 6.0, min.y + 26.0 + lay.text), "", lay.text, ALARM, Align2::RIGHT_TOP);
        }

        // what F is compared with: buttons
        let (bmin, bmax) = lay.buttons;
        let label_w = if lay.narrow { 0.0 } else { f.text_size("compare with:", lay.small).x + 10.0 };
        if !lay.narrow {
            f.text_px(Vec2::new(bmin.x, (bmin.y + bmax.y) * 0.5), "compare with:", lay.small, pal::DIM, Align2::LEFT_CENTER);
        }
        let rects = plot::even_row(NULLS.len(), bmin.x + label_w, bmax.x, bmin.y, bmax.y - bmin.y, 6.0);
        for (r, (kind, long, short)) in rects.iter().zip(NULLS) {
            let fits = f.text_size(long, lay.small).x + 12.0 < r.1.x - r.0.x;
            if plot::button(f, *r, if fits { long } else { short }, lay.small, self.null == kind) && self.null != kind {
                self.null = kind;
                let d = self.sample.design;
                self.refresh(d);
            }
        }

        // what the run has found
        self.draw_counts(f, lay);
    }

    /// The words under the F chart. What changes while a run goes on is said in short
    /// lines of their own, so that the longer paragraphs stay the same from frame to
    /// frame and are broken into lines once.
    fn draw_counts(&self, f: &mut Frame, lay: &Layout) {
        let (min, max) = lay.counts;
        let w = max.x - min.x;
        let s = lay.small;
        let step = s + 5.0;
        let sp = &self.sample.split;
        let r = &self.run.said;
        let d = &self.sample.design;
        // each: what it says in full, then shorter (the shortest that fits is used on a
        // phone, where an empty one is left out), its colour, and whether the full one is
        // a paragraph to wrap (or a line that changes as the run goes on)
        let mut lines: Vec<(Vec<String>, Color, bool)> = Vec::new();
        let observed = sp.f.map(amount);
        let done = whole(r.done);
        match self.run.kind {
            Null::Formula => lines.push((
                vec![
                    r"The purple curve is the $F$ distribution: what $F$ does when the true means are all equal. It is exact when the points are independent and every group is bell-shaped with one common spread.".to_string(),
                    "Exact for bell-shaped groups of one spread.".to_string(),
                    "The curve: if the means were equal.".to_string(),
                ],
                pal::DIM,
                true,
            )),
            Null::Shuffles => {
                lines.push((
                    vec![
                        "Purple histogram: the $F$ of the same points dealt out among the groups at random, as if the groups meant nothing.".to_string(),
                        if r.done > 0 { format!("Purple histogram: $F$ of the points shuffled, {done} times.") } else { "Purple histogram: $F$ of the points shuffled.".to_string() },
                        "Purple: shuffled.".to_string(),
                    ],
                    ASSUMED,
                    true,
                ));
                if r.done == 0 {
                    lines.push((vec!["Shuffling…".to_string(), "Shuffling…".to_string()], pal::DIM, false));
                } else if let (Some(fv), Some(p)) = (&observed, sp.p) {
                    let (reached, q) = (whole(r.reached), p_text(r.shuffled_p()));
                    lines.push((
                        vec![
                            format!(r"{reached} of {done} shuffles reach $F = {fv}$: $p = {q}$ by shuffling."),
                            format!(r"{reached} of {done} reach $F = {fv}$: $p = {q}$ by shuffling."),
                            format!(r"{reached} reach it: $p = {q}$."),
                        ],
                        pal::TEXT,
                        false,
                    ));
                    lines.push((vec![format!("(The data as they are count among the shuffles, so this $p$ is never 0. By the formula, {}.)", p_is(p)), String::new()], pal::DIM, true));
                }
            }
            Null::Samples => {
                let (share, mean) = (percent(r.share(r.reached)), amount(r.mean_f()));
                lines.push((
                    vec![
                        "Green histogram: the $F$ of new samples drawn from the truth as it is set.".to_string(),
                        if r.done > 0 { format!("Green histogram: $F$ of {done} new samples, mean {mean}.") } else { "Green histogram: $F$ of new samples.".to_string() },
                        "Green: new samples.".to_string(),
                    ],
                    TRUTH,
                    true,
                ));
                if r.done == 0 {
                    lines.push((vec!["Drawing new samples…".to_string(), "Drawing new samples…".to_string()], pal::DIM, false));
                } else {
                    lines.push((vec![format!("{done} new samples: $F$ passes the 5 % line in {share}; their mean $F$ is {mean}."), format!("Past the 5 % line: {share}.")], pal::TEXT, false));
                }
                let why = if d.equal_means() {
                    let curve = (sp.df_w > 2.0).then(|| format!(r" The curve's own mean is ${} / {} = {}$.", sp.df_w, sp.df_w - 2.0, amount(sp.df_w / (sp.df_w - 2.0))));
                    format!("The true means are equal, so each time it does is a false alarm.{}", curve.unwrap_or_default())
                } else {
                    "The true means differ, so the share past the line is the power: the chance of finding them different.".to_string()
                };
                lines.push((vec![why, String::new()], pal::DIM, true));
                if d.k >= 3 && r.done > 0 {
                    let pairs = d.k * (d.k - 1) / 2;
                    let share = percent(r.share(r.pairs));
                    lines.push((
                        vec![
                            format!("A $t$ test of each of the {pairs} pairs alone finds some pair different in {share}."),
                            format!("Some pair different alone: {share}."),
                        ],
                        pal::TEXT,
                        false,
                    ));
                }
            }
        }
        match (&observed, sp.p) {
            (Some(fv), Some(p)) => lines.push((
                vec![
                    format!(r"Pink: the share of the curve beyond $F = {fv}$, {}: the chance, if the true means were equal, of an $F$ at least this large.", p_is(p)),
                    format!(r"Pink: {}, the share beyond $F = {fv}$.", p_is(p)),
                    p_is(p),
                ],
                ALARM,
                true,
            )),
            _ if sp.still() => lines.push((vec!["Every point is the same: nothing varies, and there is nothing to compare.".to_string(), "Every point is the same.".to_string()], pal::TEXT, true)),
            _ => lines.push((
                vec![
                    r"No scatter within the groups: every point equals its group's mean, so $F$, a ratio with nothing below it, cannot be formed.".to_string(),
                    "No scatter within the groups: no $F$.".to_string(),
                ],
                pal::TEXT,
                true,
            )),
        }
        if let Some((t, df, p)) = self.sample.t() {
            lines.push((
                vec![
                    format!(r"Two groups: Student's $t = {}$ with {df} degrees of freedom, and $t^2 = {} = F$; the $t$ test gives the same {} as $F$.", num(t, 2), amount(t * t), p_is(p)),
                    format!(r"$t = {}$, $t^2 = {} = F$, by $t$: {}.", num(t, 2), amount(t * t), p_is(p)),
                ],
                pal::TEXT,
                true,
            ));
        }
        let mut y = min.y;
        for (options, color, wraps) in lines {
            if lay.narrow || !wraps {
                let text = pick(f, if lay.narrow { &options[1..] } else { &options }, w, s);
                if text.is_empty() {
                    continue;
                }
                if y + step > max.y + 2.0 {
                    break;
                }
                f.text_px(Vec2::new(min.x, y), text, s, color, Align2::LEFT_TOP);
                y += step;
            } else {
                let text = options.first().map(String::as_str).unwrap_or("");
                let h = paragraph_height(f, w, text, s);
                if y + h > max.y + 2.0 {
                    break;
                }
                paragraph(f, Vec2::new(min.x, y), w, text, s, color);
                y += h + 5.0;
            }
        }
    }
}

/// A paragraph broken into lines no wider than `width` logical pixels, its first
/// line's top-left corner at `at` (screen space): `kit::wrapped`, which works in the
/// picture's own units, given them.
fn paragraph(f: &mut Frame, at: Vec2, width: f32, text: &str, size: f32, color: Color) {
    let ppu = pixels_per_unit(f).max(1e-6);
    if let Some(world) = f.cam.pick_ground(at * f.px) {
        wrapped(f, world.truncate(), width / ppu, text, size, color);
    }
}

/// How tall [`paragraph`] draws a text.
fn paragraph_height(f: &Frame, width: f32, text: &str, size: f32) -> f32 {
    wrapped_height(f, width / pixels_per_unit(f).max(1e-6), text, size)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// An experiment as its button sets it up, its run (if it has one) finished.
    fn run(k: usize) -> Anova {
        let mut s = Anova::create();
        s.set_up(k);
        s.settle();
        s
    }

    /// The mean of F over `count` new samples from a design.
    fn mean_f(d: Design, count: usize) -> f64 {
        let s = Sample::new(d, &noise(d.seed));
        let mut r = Run::new();
        r.restart(Null::Samples, &d, 50.0);
        let critical = t_critical();
        for _ in 0..count {
            r.step(&s, &critical);
        }
        r.mean_f()
    }

    /// For any data, between + within = total, the pieces that make the degrees of
    /// freedom add up to 0, F and p are what `Dist::F` says, and the fast F used by the
    /// shuffles is the same F.
    #[test]
    fn between_and_within_add_up_to_the_total() {
        let mut rng = Rng::new(5);
        for trial in 0..300 {
            let k = 2 + trial % 5;
            let groups: Vec<Vec<f64>> = (0..k)
                .map(|i| {
                    let n = FEWEST + rng.below(MOST - FEWEST + 1);
                    let (centre, spread) = (10.0 * (i as f64 - 2.0) * rng.f32() as f64, 0.01 + 5.0 * rng.f32() as f64);
                    (0..n).map(|_| centre + spread * st::normal(&mut rng)).collect()
                })
                .collect();
            let s = Split::of(&groups);
            assert!((s.between + s.within - s.all).abs() <= 1e-10 * s.all, "trial {trial}: {} + {} against {}", s.between, s.within, s.all);
            let total: usize = groups.iter().map(Vec::len).sum();
            assert_eq!((s.df_b, s.df_w), ((k - 1) as f64, (total - k) as f64));
            let weighted: f64 = groups.iter().zip(&s.means).map(|(g, m)| g.len() as f64 * (m - s.grand)).sum();
            assert!(weighted.abs() < 1e-9 * (1.0 + s.all.sqrt()));
            for (g, m) in groups.iter().zip(&s.means) {
                assert!(g.iter().map(|y| y - m).sum::<f64>().abs() < 1e-9 * (1.0 + s.all.sqrt()));
            }
            let (fv, p) = (s.f.expect("an F"), s.p.expect("a p"));
            assert!((fv - (s.between / s.df_b) / (s.within / s.df_w)).abs() <= 1e-12 * fv.max(1.0));
            assert_eq!(p, (1.0 - Dist::F { d1: s.df_b, d2: s.df_w }.cdf(fv)).max(0.0).min(1.0));
            let sizes: Vec<usize> = groups.iter().map(Vec::len).collect();
            assert!((ratio(&groups.concat(), &sizes) - fv).abs() <= 1e-9 * fv.max(1.0));
            // what is written adds up, and the total written is within a unit of its
            // last digit
            let [b, w, t] = sums_text(s.between, s.within).map(|v| v.parse::<f64>().unwrap_or(f64::NAN));
            let unit: f64 = if s.all >= 10_000.0 {
                1.0
            } else if s.all >= 1000.0 {
                0.1
            } else {
                0.01
            };
            assert!((b + w - t).abs() < 1e-6 * unit.max(t) && (t - s.all).abs() <= 1.0001 * unit, "{b} + {w} = {t} for {}", s.all);
        }
    }

    /// The first experiment: one group apart, F and p, and the shuffles.
    #[test]
    fn one_group_apart() {
        let s = run(0);
        let sp = &s.sample.split;
        assert!(sp.k == 4 && sp.total == 40 && s.null == Null::Shuffles && s.sample.design.mu[..4] == APART);
        assert_eq!((amount(sp.f.unwrap_or(0.0)), p_text(sp.p.unwrap_or(1.0)), sp.share().map(percent)), ("4.69".into(), "0.0072".into(), Some("28.1 %".into())));
        // group 4 is the one furthest from the grand mean
        assert_eq!(s.highlighted().map(|h| h.0), Some(3));
        // 17 of 2 000 shuffles reach F, and they follow the curve: their histogram's
        // share below each bar's end is the curve's, near enough
        let r = &s.run;
        assert_eq!((r.done, r.reached, p_text(r.shuffled_p())), (RUNS, 17, "0.0090".into()));
        // (and the words say so at the end)
        assert_eq!(r.said, r.tally());
        let null = sp.null();
        let (mut below, mut most) = (0u32, 0.0f64);
        for (i, c) in r.hist.counts.iter().enumerate() {
            below += c;
            let edge = r.hist.lo + (i + 1) as f64 * r.hist.width();
            most = most.max((below as f64 / r.done as f64 - null.cdf(edge)).abs());
        }
        assert!(most < 0.025, "the shuffles' histogram is {most} from the curve");
        // dragging group 4 further away: the part between grows, and F with it
        let mut last = (sp.between, sp.f.unwrap_or(0.0));
        let mut s = s;
        for at in [1.6, 2.0, 2.4, 3.0] {
            let mut d = s.sample.design;
            put_mean(&mut d, &s.z, 3, at);
            s.refresh(d);
            let now = (s.sample.split.between, s.sample.split.f.unwrap_or(0.0));
            assert!((s.sample.split.means[3] - at).abs() < 1e-12 && now.0 > last.0 && now.1 > last.1, "at {at}: {now:?} after {last:?}");
            last = now;
        }
    }

    /// The second: nothing going on, over new samples; and the pairs tested alone.
    #[test]
    fn nothing_going_on() {
        let s = run(1);
        let sp = &s.sample.split;
        assert!(sp.k == 5 && sp.total == 50 && s.sample.design.equal_means() && s.null == Null::Samples);
        assert_eq!((amount(sp.f.unwrap_or(0.0)), p_text(sp.p.unwrap_or(0.0))), ("1.02".into(), "0.41".into()));
        let r = &s.run;
        assert_eq!(r.done, RUNS);
        assert_eq!((amount(r.mean_f()), amount(sp.df_w / (sp.df_w - 2.0))), ("1.03".into(), "1.05".into()));
        assert_eq!((percent(r.share(r.reached)), percent(r.share(r.pairs))), ("5.1 %".into(), "28.1 %".into()));
    }

    /// The third: the same group means, every point twice as far from its own.
    #[test]
    fn noisier_groups() {
        let (first, s) = (run(0), run(2));
        let (a, b) = (&first.sample.split, &s.sample.split);
        for i in 0..4 {
            assert!((a.means[i] - b.means[i]).abs() < 1e-12);
            for j in 0..10 {
                let (da, db) = (first.sample.groups[i][j] - a.means[i], s.sample.groups[i][j] - b.means[i]);
                assert!((db - 2.0 * da).abs() < 1e-12);
            }
        }
        assert!((a.between - b.between).abs() < 1e-9 && (b.within - 4.0 * a.within).abs() < 1e-9 * b.within);
        assert_eq!(sums_text(b.between, b.within), ["10.55".to_string(), "107.91".into(), "118.46".into()]);
        assert_eq!(sums_text(a.between, a.within), ["10.55".to_string(), "26.98".into(), "37.53".into()]);
        let (fa, fb) = (a.f.unwrap_or(0.0), b.f.unwrap_or(0.0));
        assert!((fb - fa / 4.0).abs() < 1e-9);
        assert_eq!((amount(fb), p_text(b.p.unwrap_or(0.0))), ("1.17".into(), "0.33".into()));
    }

    /// The fourth: the same group means, four times the data; and F − 1 on average
    /// over new samples from the first experiment's truth.
    #[test]
    fn four_times_the_data() {
        let (first, s) = (run(0), run(3));
        let (a, b) = (&first.sample.split, &s.sample.split);
        assert_eq!(s.sample.sizes, vec![40; 4]);
        for i in 0..4 {
            assert!((a.means[i] - b.means[i]).abs() < 1e-12);
        }
        assert!((b.between - 4.0 * a.between).abs() < 1e-9 * b.between);
        assert_eq!((sums_text(b.between, b.within)[0].as_str(), amount(b.ms_w), amount(a.ms_w)), ("42.20", "0.95".into(), "0.75".into()));
        assert_eq!((amount(a.f.unwrap_or(0.0)), amount(b.f.unwrap_or(0.0)), p_text(b.p.unwrap_or(1.0))), ("4.69".into(), "14.88".into(), "< 0.0001".into()));
        let ten = mean_f(Design::even(4, 10, 1.0, &APART, 29), 40_000) - 1.0;
        let forty = mean_f(Design::even(4, 40, 1.0, &APART, 29), 40_000) - 1.0;
        assert_eq!((amount(ten), amount(forty), num(forty / ten, 1)), ("3.25".into(), "12.25".into(), "3.8".into()));
        // (and what theory says: E F = (1 + n θ) d₂/(d₂ − 2), θ the spread of the true
        // means over σ² and k − 1)
        let theta = APART.iter().map(|m| (m - 0.275) * (m - 0.275)).sum::<f64>() / 3.0;
        let expect = |n: f64, d2: f64| (1.0 + n * theta) * d2 / (d2 - 2.0) - 1.0;
        assert!((ten - expect(10.0, 36.0)).abs() < 0.05 && (forty - expect(40.0, 156.0)).abs() < 0.15, "{ten} {forty}");
    }

    /// The fifth: with two groups F is t², and the p-values are the same; for any
    /// two groups, not only these.
    #[test]
    fn two_groups_f_is_t_squared() {
        let s = run(4);
        let sp = &s.sample.split;
        let (t, df, p) = s.sample.t().expect("two groups");
        assert_eq!((num(t, 2), amount(t * t), amount(sp.f.unwrap_or(0.0)), df), ("−2.60".into(), "6.75".into(), "6.75".into(), 18.0));
        assert_eq!((p_text(p), p_text(sp.p.unwrap_or(0.0))), ("0.018".into(), "0.018".into()));
        for seed in 1..60u64 {
            let mut d = Design::even(2, 2 + seed as usize % 40, 0.5 + seed as f64 * 0.03, &[0.0, (seed as f64 * 0.37).sin()], seed);
            d.n[1] = 2 + (seed as usize * 7) % 50;
            let s = Sample::new(d, &noise(seed));
            let (t, _, pt) = s.t().expect("two groups");
            let (fv, pf) = (s.split.f.unwrap_or(f64::NAN), s.split.p.unwrap_or(f64::NAN));
            assert!((t * t - fv).abs() <= 1e-10 * fv.max(1.0), "seed {seed}: {t}² against {fv}");
            assert!((pt - pf).abs() < 1e-12, "seed {seed}: {pt} against {pf}");
        }
    }

    /// The sixth: unequal spreads and sizes, equal means: how often F rejects, that
    /// way round, the other way round, and with equal sizes.
    #[test]
    fn unequal_spreads() {
        let s = run(5);
        let d = s.sample.design;
        assert!(d.k == 3 && d.n[..3] == [6, 6, 24] && d.sd[..3] == [2.0, 2.0, 1.0] && d.equal_means() && d.mu[0] == 0.0);
        assert!(s.own, "the panel shows each group's size and spread");
        assert_eq!(percent(s.run.share(s.run.reached)), "19.9 %");
        // the within mean square is mostly the large group's: 23 of its 33 degrees of freedom
        assert_eq!(s.sample.split.df_w, 33.0);
        let variant = |n: [usize; 3], sd: [f64; 3]| {
            let mut v = run(5);
            let mut d = v.sample.design;
            for i in 0..3 {
                (d.n[i], d.sd[i]) = (n[i], sd[i]);
            }
            v.refresh(d);
            v.settle();
            percent(v.run.share(v.run.reached))
        };
        assert_eq!(variant([6, 6, 24], [1.0, 1.0, 2.0]), "0.7 %");
        assert_eq!(variant([12, 12, 12], [2.0, 2.0, 1.0]), "6.6 %");
        // "about four times as often as it should", over many more samples
        let mut r = Run::new();
        r.restart(Null::Samples, &d, 50.0);
        let sample = Sample::new(d, &noise(d.seed));
        for _ in 0..40_000 {
            r.step(&sample, &s.t_critical);
        }
        let times = r.share(r.reached) / LEVEL;
        assert!(times > 3.5 && times < 4.2, "{times} times as often");
    }

    /// With bell-shaped groups of one spread and equal means, the share between is
    /// (k − 1)/(N − 1) on average (the panel's words); and F rejects at its level.
    #[test]
    fn under_the_null_the_numbers_are_what_the_words_say() {
        for (k, n) in [(3usize, 5usize), (4, 10), (6, 20)] {
            let d = Design::even(k, n, 1.3, &[0.7; GROUPS], 17);
            let s = Sample::new(d, &noise(17));
            let mut r = Run::new();
            r.restart(Null::Samples, &d, 50.0);
            let (mut sum, runs) = (0.0, 20_000);
            let (df_b, df_w) = ((k - 1) as f64, (k * n - k) as f64);
            let critical = t_critical();
            for _ in 0..runs {
                let before = r.sum;
                r.step(&s, &critical);
                let fv = r.sum - before;
                sum += fv * df_b / (fv * df_b + df_w);
            }
            let expect = df_b / (k * n - 1) as f64;
            assert!((sum / runs as f64 - expect).abs() < 0.004, "k {k}, n {n}: {} against {expect}", sum / runs as f64);
            assert!((r.share(r.reached) - LEVEL).abs() < 0.006, "k {k}, n {n}: rejects {}", r.share(r.reached));
        }
    }

    /// Dragging a bar moves its whole group, so that its mean is where the bar was
    /// put, and nothing else.
    #[test]
    fn a_dragged_bar_takes_its_group_with_it() {
        let mut s = Anova::create();
        let before = s.sample.clone();
        let mut d = s.sample.design;
        put_mean(&mut d, &s.z, 1, -2.5);
        s.refresh(d);
        assert!((s.sample.split.means[1] + 2.5).abs() < 1e-12);
        for (j, y) in s.sample.groups[1].iter().enumerate() {
            assert!((y - before.groups[1][j] - (-2.5 - before.split.means[1])).abs() < 1e-12);
        }
        assert_eq!(s.sample.groups[0], before.groups[0]);
        // the run starts again for the new data
        assert_eq!(s.run.done, 0);
    }

    /// No scatter within, nothing varying at all, equal means by construction, the
    /// fewest and most points and groups: every number drawn is finite, and the words
    /// say what has happened.
    #[test]
    fn degenerate_data_are_told_honestly() {
        let lays = [Layout::of(Vec2::new(1075.0, 1000.0), 0.0), Layout::of(Vec2::new(390.0, 756.0), 44.0), Layout::of(Vec2::new(240.0, 180.0), 0.0)];
        let mut cases = Vec::new();
        for k in [2, 3, 6] {
            for n in [FEWEST, 7, MOST] {
                for sd in [0.0, 1.0, WIDEST] {
                    for means in [[0.0; GROUPS], [0.0, 1.0, -2.0, 0.5, 3.0, -1.0], [REACH; GROUPS]] {
                        cases.push(Design::even(k, n, sd, &means, 3));
                    }
                }
            }
        }
        for d in cases {
            for null in [Null::Formula, Null::Shuffles, Null::Samples] {
                let mut s = Anova::create();
                s.null = null;
                s.refresh(d);
                for _ in 0..50 {
                    s.run.step(&s.sample, &s.t_critical);
                }
                let sp = &s.sample.split;
                if d.sd[0] == 0.0 {
                    assert!(sp.f.is_none() && sp.p.is_none(), "{d:?}");
                    assert_eq!(sp.still(), d.equal_means(), "{d:?}");
                } else {
                    assert!(sp.f.is_some_and(f64::is_finite) && sp.p.is_some_and(|p| (0.0..=1.0).contains(&p)), "{d:?}");
                }
                assert!(s.highlighted().is_some());
                let g = s.goals();
                let all = g.frame.iter().chain(&g.marks).chain(g.points.iter().flatten()).chain(&g.hist).chain(&g.curve).chain(&g.bars[..6]).chain(&g.bars[7..]);
                assert!(all.clone().all(|v| v.is_finite()), "{d:?} {null:?}");
                // (the curve and the histogram as shares of the chart's height, inside it)
                assert!(g.curve.iter().chain(&g.hist).all(|v| (0.0..=1.0).contains(v)), "{d:?} {null:?}");
                assert!(s.run.shuffled_p() > 0.0 && s.run.mean_f().is_finite());
                for lay in &lays {
                    assert!([lay.strips, lay.split, lay.null, lay.buttons, lay.counts].iter().all(|(a, b)| a.is_finite() && b.is_finite()));
                }
            }
        }
        // means made equal by dragging: nothing between, F = 0 and p = 1
        let mut s = Anova::create();
        let mut d = s.sample.design;
        for i in 0..4 {
            put_mean(&mut d, &s.z, i, 0.5);
        }
        s.refresh(d);
        assert!(s.sample.split.between < 1e-20 && s.sample.split.f.is_some_and(|f| f < 1e-20) && p_text(s.sample.split.p.unwrap_or(0.0)) == "1.00");
        // a p-value is never written as 0
        assert_eq!((p_text(0.0), p_text(3e-9), p_text(0.00031)), ("< 0.0001".into(), "< 0.0001".into(), "0.0003".into()));
        assert_eq!(whole(2000), "2 000");
    }

    /// The layout of every window: finite, inside the canvas, nothing on top of
    /// anything else.
    #[test]
    fn the_layout_fits_every_window() {
        for (w, h, inset) in [(1075.0, 1000.0, 0.0), (1400.0, 760.0, 0.0), (580.0, 700.0, 0.0), (390.0, 756.0, 44.0), (700.0, 500.0, 0.0), (240.0, 180.0, 0.0), (1.0, 1.0, 0.0), (3000.0, 2000.0, 0.0)] {
            let lay = Layout::of(Vec2::new(w, h), inset);
            let rects = [lay.strips, lay.split, lay.null, lay.buttons, lay.counts];
            assert!(rects.iter().all(|(a, b)| a.is_finite() && b.is_finite()), "{w}x{h}");
            if lay.tiny {
                assert!(w < 300.0 || h < 600.0, "{w}x{h} should have room");
                continue;
            }
            assert!(rects.iter().all(|(a, b)| a.x >= 0.0 && a.y >= inset && b.x <= w + 0.5 && b.y <= h + 0.5), "{w}x{h}");
            assert!(lay.strips.1.y < lay.split.0.y && lay.null.1.y < lay.buttons.0.y && lay.buttons.1.y < lay.counts.0.y, "{w}x{h}");
            if lay.narrow {
                assert!(lay.split.1.y < lay.null_head.y && lay.counts.1.y - lay.counts.0.y > 30.0, "{w}x{h}");
            } else {
                assert!(lay.strips.1.x < lay.null.0.x && lay.split.1.x < lay.buttons.0.x, "{w}x{h}");
            }
        }
    }

    /// The round values on an axis reach a few steps beyond where it is going, at about
    /// sixty pixels apart, and are written with the digits their step needs.
    #[test]
    fn the_axes_have_room_to_glide() {
        let t = Steady::of(-1.3, 2.6, 240.0);
        let v: Vec<f64> = t.values().collect();
        assert_eq!(t.step, 1.0);
        assert!(v[0] <= -1.3 - 2.0 * t.step && *v.last().unwrap_or(&0.0) >= 2.6 + 2.0 * t.step, "{v:?}");
        let fine = Steady::of(0.0, 0.4, 300.0);
        assert_eq!((t.name(2.0), fine.step, fine.name(0.1), Steady::of(0.0, 0.4, 100.0).name(0.2)), ("2".to_string(), 0.1, "0.1".to_string(), "0.2".to_string()));
        assert_eq!(Steady::of(1.0, 1.0, 200.0).values().count(), 0);
        assert_eq!(Steady::of(0.0, f64::NAN, 200.0).values().count(), 0);
    }

    /// After a change, the picture glides to the new one and comes to rest exactly
    /// there, at any frame rate.
    #[test]
    fn glides_arrive_and_stop() {
        for hz in [30.0f32, 60.0, 144.0] {
            let mut s = run(0);
            s.look.follow(&s.goals(), 0.0, true);
            assert!(!s.look.moving);
            s.set_up(2);
            let mut frames = 0;
            loop {
                let g = s.goals();
                s.look.follow(&g, 1.0 / hz, false);
                frames += 1;
                if !s.look.moving {
                    assert!(s.look.points[0].shown == g.points[0] && s.look.bars.shown == g.bars);
                    break;
                }
                assert!((frames as f32) < 2.0 * hz, "still gliding after two seconds at {hz} Hz");
            }
            assert!(frames as f32 > 0.2 * hz, "arrived in {frames} frames at {hz} Hz: no glide");
        }
    }

    /// The numbers behind the captions (run with --ignored --nocapture).
    #[test]
    #[ignore]
    fn explore() {
        for k in 0..TRIALS.len() {
            let s = run(k);
            let sp = &s.sample.split;
            let r = &s.run;
            println!(
                "{}: means {:?} between {:.3} within {:.3} ms {:.3} {:.3} F {:?} p {:?} share {:?} t {:?} | run {:?} {} reached {} pairs {} mean F {:.4}",
                TRIALS[k].0,
                sp.means.iter().map(|m| format!("{m:.3}")).collect::<Vec<_>>(),
                sp.between,
                sp.within,
                sp.ms_b,
                sp.ms_w,
                sp.f,
                sp.p,
                sp.share(),
                s.sample.t(),
                r.kind,
                r.done,
                r.reached,
                r.pairs,
                r.mean_f()
            );
        }
    }
}

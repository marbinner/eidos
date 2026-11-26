//! Comparing two groups: the t test, shuffling the labels, ranks, the size of a
//! difference, and pairs.

use super::*;
use crate::math::stats::{self as st, Dist};
use plot::{dashed_of, disc, line_of, Glide, Plot};

pub const DEF: SceneDef = SceneDef {
    id: "twogroups",
    title: "Comparing two groups",
    group: TESTING,
    blurb: "Is the difference between two groups more than chance would give? The $t$ test, shuffling the labels, and the size of an effect.",
    make: |_| Box::new(TwoGroups::create()),
};

// ----- the groups the data come from -------------------------------------------------------

/// The shape both groups are drawn with: a standard one (mean 0, spread 1), moved to
/// each group's mean and stretched to its spread.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Shape {
    Bell,
    /// An exponential less its mean: a long tail to the right.
    Skewed,
    /// Student's t with 3 degrees of freedom, scaled to spread 1: now and then a value
    /// very far out.
    Heavy,
}

impl Shape {
    const ALL: [Shape; 3] = [Shape::Bell, Shape::Skewed, Shape::Heavy];

    fn name(self) -> &'static str {
        match self {
            Shape::Bell => "bell (normal)",
            Shape::Skewed => "skewed",
            Shape::Heavy => "heavy tails",
        }
    }

    /// One draw, of mean 0 and standard deviation 1.
    fn draw(self, rng: &mut Rng) -> f64 {
        match self {
            Shape::Bell => st::normal(rng),
            Shape::Skewed => -st::unit(rng).ln() - 1.0,
            Shape::Heavy => Dist::StudentT { df: 3.0 }.sample(rng) / 3f64.sqrt(),
        }
    }
}

/// What the two samples are drawn from.
#[derive(Clone, Copy, Debug, PartialEq)]
struct Truth {
    /// Each unit measured twice (A and B), instead of two separate groups.
    paired: bool,
    /// In pairs, the correlation of a unit's two measurements.
    rho: f64,
    /// The sizes of A and B (in pairs, the first is the number of units).
    n: [usize; 2],
    mean: [f64; 2],
    sd: [f64; 2],
    shape: Shape,
}

impl Truth {
    fn sizes(&self) -> [usize; 2] {
        if self.paired {
            [self.n[0], self.n[0]]
        } else {
            self.n
        }
    }

    /// Kept in range, whatever a slider or an experiment asked for.
    fn tidy(mut self) -> Truth {
        self.n = self.n.map(|n| n.clamp(1, MOST));
        self.sd = self.sd.map(|s| if s.is_finite() { s.max(0.0).min(10.0) } else { 1.0 });
        self.mean = self.mean.map(|m| if m.is_finite() { m.max(-10.0).min(10.0) } else { 0.0 });
        self.rho = if self.rho.is_finite() { self.rho.max(-1.0).min(1.0) } else { 0.0 };
        self
    }
}

/// The most points in a group.
const MOST: usize = 1000;

/// A sample of each group, made from `seed`. The two groups are drawn from separate
/// streams, so that a change to one group's size leaves the other's points where they
/// were, and a change of mean or spread moves every point in proportion. In pairs, a
/// unit's B is made from its A's standard draw and one of its own: `rho` times the
/// first plus √(1 − rho²) times the second.
fn sample(t: &Truth, seed: u64) -> [Vec<f64>; 2] {
    let [na, nb] = t.sizes();
    let mut ra = Rng::new(seed.wrapping_mul(2).wrapping_add(1));
    let mut rb = Rng::new(seed.wrapping_mul(2).wrapping_add(2));
    let za: Vec<f64> = (0..na).map(|_| t.shape.draw(&mut ra)).collect();
    let zb: Vec<f64> = (0..nb).map(|_| t.shape.draw(&mut rb)).collect();
    let a = za.iter().map(|z| t.mean[0] + t.sd[0] * z).collect();
    let rest = (1.0 - t.rho * t.rho).max(0.0).sqrt();
    let b = (0..nb).map(|i| t.mean[1] + t.sd[1] * if t.paired { t.rho * za[i] + rest * zb[i] } else { zb[i] }).collect();
    [a, b]
}

// ----- the tests --------------------------------------------------------------------------------

/// The size below which a spread is taken for none: rounding leaves a few units in the
/// last place where the numbers are all equal.
fn tolerance(x: &[f64]) -> f64 {
    1e-9 * (1.0 + x.iter().fold(0.0f64, |m, v| m.max(v.abs())))
}

/// A t test: the difference, its standard error, t, the degrees of freedom and the
/// two-sided p-value.
#[derive(Clone, Copy, Debug, PartialEq)]
struct TTest {
    diff: f64,
    se: f64,
    t: f64,
    df: f64,
    p: f64,
}

/// The share of Student's t with `df` degrees of freedom at least `t` from 0, either way.
fn two_sided(t: f64, df: f64) -> f64 {
    (2.0 * Dist::StudentT { df }.cdf(-t.abs())).min(1.0)
}

fn t_test(diff: f64, se: f64, df: f64) -> TTest {
    let t = diff / se;
    TTest { diff, se, t, df, p: two_sided(t, df) }
}

/// Welch's test: each group's own spread, `se² = s_A²/n_A + s_B²/n_B`, and the degrees
/// of freedom of Welch and Satterthwaite. Needs two points in each group and some spread.
fn welch(a: &[f64], b: &[f64]) -> Option<TTest> {
    let (na, nb) = (a.len(), b.len());
    if na < 2 || nb < 2 {
        return None;
    }
    let tol = tolerance(a).max(tolerance(b));
    let (va, vb) = (st::variance(a) / na as f64, st::variance(b) / nb as f64);
    let se2 = va + vb;
    if !(se2.sqrt() > tol && se2.is_finite()) {
        return None;
    }
    let df = se2 * se2 / (va * va / (na - 1) as f64 + vb * vb / (nb - 1) as f64);
    Some(t_test(st::mean(b) - st::mean(a), se2.sqrt(), df))
}

/// The pooled spread of the two groups (each about its own mean), and its degrees of
/// freedom, if it has any.
fn pooled_sd(a: &[f64], b: &[f64]) -> Option<(f64, usize)> {
    let (na, nb) = (a.len(), b.len());
    if na == 0 || nb == 0 || na + nb < 3 {
        return None;
    }
    let df = na + nb - 2;
    let ss = st::variance(a) * (na.max(1) - 1) as f64 + st::variance(b) * (nb.max(1) - 1) as f64;
    let sp = (ss / df as f64).sqrt();
    (sp > tolerance(a).max(tolerance(b)) && sp.is_finite()).then_some((sp, df))
}

/// The classical test: one spread pooled from both groups, `n_A + n_B − 2` degrees of
/// freedom. (A group of one can borrow its spread from the other.)
fn pooled(a: &[f64], b: &[f64]) -> Option<TTest> {
    let (sp, df) = pooled_sd(a, b)?;
    let se = sp * (1.0 / a.len() as f64 + 1.0 / b.len() as f64).sqrt();
    Some(t_test(st::mean(b) - st::mean(a), se, df as f64))
}

/// Each unit's B less its A.
fn differences(a: &[f64], b: &[f64]) -> Vec<f64> {
    a.iter().zip(b).map(|(x, y)| y - x).collect()
}

/// The paired test: the differences' mean against 0, with `n − 1` degrees of freedom.
fn paired_t(a: &[f64], b: &[f64]) -> Option<TTest> {
    let d = differences(a, b);
    let n = d.len();
    if n < 2 {
        return None;
    }
    let se = st::sd(&d) / (n as f64).sqrt();
    (se > tolerance(&d) && se.is_finite()).then(|| t_test(st::mean(&d), se, (n - 1) as f64))
}

/// A test on ranks: its statistic, its two-sided p-value, and how that was found.
#[derive(Clone, Copy, Debug, PartialEq)]
struct RankTest {
    stat: f64,
    p: f64,
    /// From the exact distribution (no ties, few points); otherwise from the normal
    /// approximation, corrected for ties and for continuity.
    exact: bool,
    /// Every value tied: the ranks say nothing.
    all_tied: bool,
}

/// The most points for which a rank test's distribution is counted out exactly.
const EXACT_MOST: usize = 50;

/// An exact distribution of the rank-sum statistic, and the two sizes it is for.
type UTable = ((usize, usize), std::rc::Rc<Vec<f64>>);

thread_local! {
    /// The exact distributions of the rank-sum statistic counted so far.
    static U_TABLES: std::cell::RefCell<Vec<UTable>> = const { std::cell::RefCell::new(Vec::new()) };
}

/// How many of the ways to deal `k` of `n` ranks (1 to n) give each sum `s`, from
/// `s = k(k+1)/2` up: the number of subsets of each size and sum, built one rank at a time.
fn subset_sums(n: usize, k: usize) -> Vec<f64> {
    let least = k * (k + 1) / 2;
    let most = k * (2 * n + 1 - k) / 2;
    let width = most + 1;
    let mut ways = vec![0.0f64; (k + 1) * width];
    ways[0] = 1.0;
    for r in 1..=n {
        for j in (1..=k.min(r)).rev() {
            for s in (r..width).rev() {
                let add = ways[(j - 1) * width + s - r];
                ways[j * width + s] += add;
            }
        }
    }
    ways[k * width + least..k * width + width].to_vec()
}

/// The exact distribution of Mann and Whitney's U (the number of pairs in which B is
/// the larger) for groups of `m` and `n` with no ties: how many of the equally likely
/// deals of the labels give each value from 0 to `mn`.
fn u_counts(m: usize, n: usize) -> std::rc::Rc<Vec<f64>> {
    let key = (m.min(n), m.max(n));
    if let Some(found) = U_TABLES.with(|t| t.borrow().iter().find(|(k, _)| *k == key).map(|(_, v)| v.clone())) {
        return found;
    }
    let counts = std::rc::Rc::new(subset_sums(m + n, key.0));
    U_TABLES.with(|t| {
        let mut t = t.borrow_mut();
        if t.len() >= 8 {
            t.remove(0);
        }
        t.push((key, counts.clone()));
    });
    counts
}

/// Twice the smaller tail of a distribution counted out exactly, at `at`.
fn exact_two_sided(counts: &[f64], at: usize) -> f64 {
    let total: f64 = counts.iter().sum();
    let below: f64 = counts[..=at.min(counts.len() - 1)].iter().sum();
    let above: f64 = counts[at.min(counts.len() - 1)..].iter().sum();
    (2.0 * below.min(above) / total).min(1.0)
}

/// The two-sided p-value of a statistic by the normal approximation, with half a step
/// of continuity correction.
fn normal_two_sided(stat: f64, mean: f64, var: f64) -> f64 {
    let off = (stat - mean).abs();
    let z = (off - 0.5).max(0.0) / var.sqrt();
    (2.0 * st::normal_cdf(-z)).min(1.0)
}

/// The sizes of the groups of tied values (only those of two or more).
fn tie_groups(sorted: &[f64]) -> Vec<usize> {
    let mut groups = Vec::new();
    let mut i = 0;
    while i < sorted.len() {
        let mut j = i + 1;
        while j < sorted.len() && sorted[j] == sorted[i] {
            j += 1;
        }
        if j - i > 1 {
            groups.push(j - i);
        }
        i = j;
    }
    groups
}

/// Mann and Whitney's test (Wilcoxon's rank-sum test): U, the number of pairs (one
/// from each group) in which B's value is the larger, ties counting a half; under the
/// null each deal of the ranks to the groups is equally likely.
fn mann_whitney(a: &[f64], b: &[f64]) -> Option<RankTest> {
    let (m, n) = (a.len(), b.len());
    if m == 0 || n == 0 {
        return None;
    }
    let all: Vec<f64> = a.iter().chain(b).copied().collect();
    let ranks = st::ranks(&all);
    let rank_b: f64 = ranks[m..].iter().sum();
    let u = rank_b - (n * (n + 1)) as f64 / 2.0;
    let ties = tie_groups(&st::sorted(&all));
    let big = m + n;
    if ties.first() == Some(&big) {
        return Some(RankTest { stat: u, p: 1.0, exact: false, all_tied: true });
    }
    if ties.is_empty() && big <= EXACT_MOST {
        let counts = u_counts(m, n);
        return Some(RankTest { stat: u, p: exact_two_sided(&counts, u.round() as usize), exact: true, all_tied: false });
    }
    let (mf, nf, bf) = (m as f64, n as f64, big as f64);
    let tied: f64 = ties.iter().map(|&t| (t * t * t - t) as f64).sum();
    let var = mf * nf / 12.0 * ((bf + 1.0) - tied / (bf * (bf - 1.0)));
    Some(RankTest { stat: u, p: normal_two_sided(u, mf * nf / 2.0, var), exact: false, all_tied: false })
}

/// Wilcoxon's signed-rank test of differences: the sum of the ranks of the positive
/// ones (their sizes ranked, zeros left out); under the null each sign is a coin toss.
fn signed_rank(d: &[f64]) -> Option<RankTest> {
    if d.is_empty() {
        return None;
    }
    let tol = tolerance(d);
    let nonzero: Vec<f64> = d.iter().copied().filter(|v| v.abs() > tol).collect();
    let n = nonzero.len();
    if n == 0 {
        return Some(RankTest { stat: 0.0, p: 1.0, exact: false, all_tied: true });
    }
    let sizes: Vec<f64> = nonzero.iter().map(|v| v.abs()).collect();
    let ranks = st::ranks(&sizes);
    let v: f64 = nonzero.iter().zip(&ranks).filter(|(x, _)| **x > 0.0).map(|(_, r)| r).sum();
    let ties = tie_groups(&st::sorted(&sizes));
    if ties.is_empty() && n == d.len() && n <= EXACT_MOST {
        // every subset of the ranks 1..n is a set of positive signs, all equally likely
        let counts = subset_counts(n);
        return Some(RankTest { stat: v, p: exact_two_sided(&counts, v.round() as usize), exact: true, all_tied: false });
    }
    let nf = n as f64;
    let tied: f64 = ties.iter().map(|&t| (t * t * t - t) as f64).sum();
    let var = nf * (nf + 1.0) * (2.0 * nf + 1.0) / 24.0 - tied / 48.0;
    if var.is_nan() || var <= 0.0 {
        return Some(RankTest { stat: v, p: 1.0, exact: false, all_tied: true });
    }
    Some(RankTest { stat: v, p: normal_two_sided(v, nf * (nf + 1.0) / 4.0, var), exact: false, all_tied: false })
}

/// How many subsets of the ranks 1 to n have each sum from 0 to n(n+1)/2.
fn subset_counts(n: usize) -> Vec<f64> {
    let most = n * (n + 1) / 2;
    let mut ways = vec![0.0f64; most + 1];
    ways[0] = 1.0;
    for r in 1..=n {
        for s in (r..=most).rev() {
            ways[s] += ways[s - r];
        }
    }
    ways
}

/// Everything worked out from the two samples.
#[derive(Clone, Debug, PartialEq)]
struct Analysis {
    paired: bool,
    sizes: [usize; 2],
    means: [f64; 2],
    sds: [f64; 2],
    /// B's mean less A's (in pairs, also the mean of the differences).
    diff: f64,
    welch: Option<TTest>,
    pooled: Option<TTest>,
    /// In pairs: the test of the differences.
    paired_t: Option<TTest>,
    /// Mann and Whitney's test; in pairs, Wilcoxon's signed-rank test of the differences.
    ranks: Option<RankTest>,
    /// Cohen's d: the difference in units of the pooled spread.
    d: Option<f64>,
    /// The share of pairs (one from each group) in which B's value is the larger, ties
    /// counting a half.
    superiority: Option<f64>,
    /// In pairs: the correlation of a unit's two values.
    corr: Option<f64>,
}

fn analyse(data: &[Vec<f64>; 2], paired: bool) -> Analysis {
    let [a, b] = [&data[0][..], &data[1][..]];
    let paired = paired && a.len() == b.len();
    let means = [st::mean(a), st::mean(b)];
    let mw = mann_whitney(a, b);
    let superiority = mw.map(|r| r.stat / (a.len() * b.len()) as f64);
    Analysis {
        paired,
        sizes: [a.len(), b.len()],
        means,
        sds: [st::sd(a), st::sd(b)],
        diff: means[1] - means[0],
        welch: welch(a, b),
        pooled: pooled(a, b),
        paired_t: if paired { paired_t(a, b) } else { None },
        ranks: if paired { signed_rank(&differences(a, b)) } else { mw },
        d: pooled_sd(a, b).map(|(sp, _)| (means[1] - means[0]) / sp),
        superiority,
        corr: (paired && a.len() >= 2 && st::sd(a) > tolerance(a) && st::sd(b) > tolerance(b)).then(|| st::correlation(a, b)),
    }
}

impl Analysis {
    /// The t test the picture draws: the paired one in pairs, else Welch's.
    fn main(&self) -> Option<TTest> {
        if self.paired {
            self.paired_t
        } else {
            self.welch
        }
    }
}

// ----- shuffling the labels ---------------------------------------------------------------------

/// The shuffles asked for, and the bars of their histogram.
const SHUFFLES: usize = 2000;
const BINS: usize = 48;

/// Shuffles of the labels of one pair of samples: the difference of the means that each
/// gives, and how many are at least as far from 0 as the real one. Two groups: deal the
/// pooled points afresh into groups of the same sizes. Pairs: toss a coin for each unit,
/// to swap its two values or not (its difference changes sign).
#[derive(Clone, Debug)]
struct Shuffles {
    /// The version of the data these are shuffles of, and which round of shuffles of
    /// them ("shuffle again" starts another).
    version: u64,
    round: u64,
    paired: bool,
    /// The values dealt (the pooled points, A's first; in pairs, the differences).
    values: Vec<f64>,
    sizes: [usize; 2],
    observed: f64,
    total: f64,
    tol: f64,
    /// Every shuffle's difference, in the order they came.
    made: Vec<f64>,
    reached: usize,
    /// The histogram: `BINS` bars from −`range` to `range` (a shuffle beyond is counted
    /// all the same, but not drawn).
    range: f64,
    bins: Vec<u32>,
    /// The tallest bar (as a density) when last looked at: after the first few dozen
    /// shuffles, a few hundred, and all (looked at in every shuffle, the chart's height
    /// would move in jerks).
    peak_seen: f64,
    rng: Rng,
    /// A permutation of the pooled points, dealt from: after a shuffle its first
    /// entries are the points dealt to the smaller group (A, if the sizes are equal).
    /// In pairs, the units that kept their two values the right way round.
    order: Vec<usize>,
    signs: Vec<bool>,
}

impl Shuffles {
    fn new(data: &[Vec<f64>; 2], paired: bool, seed: u64) -> Shuffles {
        let paired = paired && data[0].len() == data[1].len();
        let values: Vec<f64> = if paired { differences(&data[0], &data[1]) } else { data[0].iter().chain(&data[1]).copied().collect() };
        let sizes = [data[0].len(), data[1].len()];
        let observed = st::mean(&data[1]) - st::mean(&data[0]);
        let total = values.iter().sum();
        let tol = tolerance(&values) * (1.0 + values.len() as f64).sqrt();
        Shuffles {
            version: 0,
            round: 0,
            paired,
            order: (0..values.len()).collect(),
            signs: vec![true; values.len()],
            values,
            sizes,
            observed,
            total,
            tol,
            made: Vec::new(),
            reached: 0,
            range: 1.0,
            bins: vec![0; BINS],
            peak_seen: 0.0,
            rng: Rng::new(seed ^ 0x5EED_5EED),
        }
    }

    /// Can shuffling say anything? (Both groups have a point; in pairs, a unit.)
    fn possible(&self) -> bool {
        self.sizes[0] > 0 && self.sizes[1] > 0
    }

    /// The group dealt each time is B (else A).
    fn deals_b(&self) -> bool {
        self.sizes[0] > self.sizes[1]
    }

    /// One more shuffle.
    fn step(&mut self) {
        if !self.possible() {
            return;
        }
        let v = if self.paired {
            let mut sum = 0.0;
            for (s, d) in self.signs.iter_mut().zip(&self.values) {
                *s = self.rng.next_u64() >> 63 == 1;
                sum += if *s { *d } else { -*d };
            }
            sum / self.values.len() as f64
        } else {
            let [na, nb] = self.sizes;
            let small = na.min(nb);
            let big = self.values.len();
            let mut sum = 0.0;
            for i in 0..small {
                let j = i + self.rng.below(big - i);
                self.order.swap(i, j);
                sum += self.values[self.order[i]];
            }
            // (the dealt group's sum, and the rest for the other)
            let (sa, sb) = if self.deals_b() { (self.total - sum, sum) } else { (sum, self.total - sum) };
            sb / nb as f64 - sa / na as f64
        };
        if v.abs() >= self.observed.abs() - self.tol {
            self.reached += 1;
        }
        let at = (v + self.range) / (2.0 * self.range) * BINS as f64;
        if at >= 0.0 && at < BINS as f64 {
            self.bins[at as usize] += 1;
        }
        self.made.push(v);
        if matches!(self.done(), 50 | 300 | SHUFFLES) {
            self.peak_seen = self.peak();
        }
    }

    fn pending(&self) -> bool {
        self.possible() && self.done() < SHUFFLES
    }

    /// The tallest bar as a density (each bar's share of the shuffles over its width).
    fn peak(&self) -> f64 {
        let most = self.bins.iter().copied().max().unwrap_or(0) as f64;
        most / self.done().max(1) as f64 / (2.0 * self.range / BINS as f64)
    }

    fn done(&self) -> usize {
        self.made.len()
    }

    /// The share of the shuffles, the real labelling counted among them, at least as far
    /// from 0 as the real difference: never 0.
    fn p(&self) -> f64 {
        (self.reached + 1) as f64 / (self.done() + 1) as f64
    }

    /// The number of different ways the labels can be dealt (in pairs, 2 to the number
    /// of units), as far as it is a number.
    fn ways(&self) -> f64 {
        if self.paired {
            2f64.powi(self.values.len().min(1000) as i32)
        } else {
            st::ln_choose(self.values.len() as f64, self.sizes[0] as f64).exp()
        }
    }

    /// Which group each pooled point was dealt to by the last shuffle (true: B); in pairs,
    /// which units kept their values the right way round.
    fn last_deal(&self) -> Vec<bool> {
        if self.paired {
            return self.signs.clone();
        }
        let small = self.sizes[0].min(self.sizes[1]);
        let dealt = self.deals_b();
        let mut to_b = vec![!dealt; self.values.len()];
        for &i in &self.order[..small.min(self.order.len())] {
            to_b[i] = dealt;
        }
        to_b
    }
}

/// The p-value of a sample by `count` shuffles (the studies' own test, 199 shuffles).
fn shuffled_p(data: &[Vec<f64>; 2], paired: bool, count: usize, seed: u64) -> Option<f64> {
    let mut s = Shuffles::new(data, paired, seed);
    if !s.possible() {
        return None;
    }
    for _ in 0..count {
        s.step();
    }
    Some(s.p())
}

// ----- many new studies --------------------------------------------------------------------------

/// The new studies counted, the shuffles each one's own test takes, and the level.
const STUDIES: usize = 1000;
const STUDY_SHUFFLES: usize = 199;
const LEVEL: f64 = 0.05;

/// The four tests counted over new studies. Two groups: Welch's, the pooled one,
/// shuffles and Mann–Whitney. Pairs: the paired t, Welch's ignoring the pairs, random
/// signs and the signed-rank test.
fn study(truth: &Truth, seed: u64) -> [Option<bool>; 4] {
    let data = sample(truth, seed);
    let [a, b] = [&data[0][..], &data[1][..]];
    let reject = |p: Option<f64>| p.map(|p| p <= LEVEL);
    if truth.paired {
        [
            reject(paired_t(a, b).map(|t| t.p)),
            reject(welch(a, b).map(|t| t.p)),
            reject(shuffled_p(&data, true, STUDY_SHUFFLES, seed)),
            reject(signed_rank(&differences(a, b)).map(|r| r.p)),
        ]
    } else {
        [
            reject(welch(a, b).map(|t| t.p)),
            reject(pooled(a, b).map(|t| t.p)),
            reject(shuffled_p(&data, false, STUDY_SHUFFLES, seed)),
            reject(mann_whitney(a, b).map(|r| r.p)),
        ]
    }
}

/// The count over new studies like the one shown: drawn afresh from the same truth.
#[derive(Clone, Debug)]
struct Studies {
    truth: Option<Truth>,
    done: usize,
    /// Per test: how many studies it could test, and how many it rejected.
    tested: [usize; 4],
    rejected: [usize; 4],
}

/// The seed of the `k`th new study.
fn study_seed(k: usize) -> u64 {
    1_000_003 + 7919 * k as u64
}

impl Studies {
    fn new() -> Studies {
        Studies { truth: None, done: 0, tested: [0; 4], rejected: [0; 4] }
    }

    fn restart(&mut self, truth: Truth) {
        *self = Studies { truth: Some(truth), ..Studies::new() };
    }

    fn pending(&self) -> bool {
        self.truth.is_some() && self.done < STUDIES
    }

    fn step(&mut self) {
        let Some(truth) = self.truth.filter(|_| self.done < STUDIES) else { return };
        for (k, r) in study(&truth, study_seed(self.done)).into_iter().enumerate() {
            if let Some(r) = r {
                self.tested[k] += 1;
                self.rejected[k] += r as usize;
            }
        }
        self.done += 1;
    }

    /// The share of the studies a test rejected, among those it could test.
    fn share(&self, k: usize) -> Option<f64> {
        (self.tested[k] > 0).then(|| self.rejected[k] as f64 / self.tested[k] as f64)
    }

    #[cfg(test)]
    fn finish(&mut self) {
        while self.pending() {
            self.step();
        }
    }
}

// ----- the experiments ---------------------------------------------------------------------------

/// One of the "things to try": the button, what to look for, and every setting it
/// depends on. `wild` moves B's largest value out to that value after the draw.
struct Trial {
    button: &'static str,
    caption: &'static str,
    truth: Truth,
    seed: u64,
    wild: Option<f64>,
    shuffle: bool,
    count: bool,
}

const CLEAR: Truth = Truth { paired: false, rho: 0.0, n: [20, 20], mean: [0.0, 1.0], sd: [1.0, 1.0], shape: Shape::Bell };

const TRIALS: [Trial; 5] = [
    Trial {
        button: "a clear difference",
        caption: r"Two groups of 20, drawn from bells one standard deviation apart. The difference of the means, 1.04, is 3.1 times its standard error: $t = 3.11$, $p = 0.0036$. Shuffling the labels 2000 times agrees: 6 shuffles give a difference as large ($p = 0.0035$). Yet the groups overlap a great deal: $d = 0.98$, and $B$'s value is the larger in only 75.5 % of the pairs (one point from each group).",
        truth: CLEAR,
        seed: 52,
        wild: None,
        shuffle: true,
        count: false,
    },
    Trial {
        button: "more noise",
        caption: r"The same true difference, three times the spread. The difference of the means is much the same (1.13), but its standard error has tripled: $t = 1.13$, $p = 0.27$, and the shuffles agree ($p = 0.26$). A difference this large is what chance alone often gives here: the test weighs the difference against the spread.",
        truth: Truth { sd: [3.0, 3.0], ..CLEAR },
        seed: 52,
        wild: None,
        shuffle: true,
        count: false,
    },
    Trial {
        button: "one wild value",
        caption: r"The clear difference again, with $B$'s largest value moved out to 20. The means move further apart (1.04 to 1.90), but $B$'s spread grows faster (1.07 to 4.40): $t$ falls to 1.88 and $p$ rises from 0.0036 to 0.075. The ranks have not changed (the point was $B$'s largest already), nor has their $p$ (0.0051). The shuffles split into two humps (the far point dealt to $A$ or to $B$), and their $p$ stays at 0.0035: the bell of the $t$ test is no longer the shape of what chance gives.",
        truth: CLEAR,
        seed: 52,
        wild: Some(20.0),
        shuffle: true,
        count: false,
    },
    Trial {
        button: "pairs",
        caption: r"Twenty units each measured twice; a unit's two values go together closely (correlation 0.90 here), and $B$ is 0.5 higher on average. As two separate groups, the spread from unit to unit swamps the difference: $p = 0.14$. As pairs, each unit is compared with itself: $p = 0.00017$. Over 1000 new studies the paired test finds the difference in 99.6 %; Welch's test, ignoring the pairs, in 18.4 %.",
        truth: Truth { paired: true, rho: 0.9, mean: [0.0, 0.5], ..CLEAR },
        seed: 4,
        wild: None,
        shuffle: true,
        count: true,
    },
    Trial {
        button: "unequal spreads and sizes",
        caption: r"The true means are equal. $A$ has 10 points of spread 3, $B$ 40 of spread 1. The pooled spread is mostly $B$'s (40 points against 10), too small for $A$, so the pooled test understates how far $A$'s mean wanders and sees differences that are not there: over 1000 new studies it says $p \le 0.05$ in 26.2 %, five times the 5 % it promises ($p = 0.0011$ for the sample shown). Shuffling (25.6 %) and ranks (16.5 %) test that the labels mean nothing at all, spreads included, and the spreads do differ. Welch's test, each group with its own spread, keeps to 5.7 %.",
        truth: Truth { n: [10, 40], mean: [0.0, 0.0], sd: [3.0, 1.0], ..CLEAR },
        seed: 3,
        wild: None,
        shuffle: true,
        count: true,
    },
];

/// The scene opens as the first experiment sets it up.
const OPENING: usize = 0;

/// Shuffles and new studies a second, unless the visitor asks for another rate.
const SHUFFLE_RATE: f32 = 500.0;
const STUDY_RATE: f32 = 250.0;

/// The most points in a group that can each be dragged.
const DRAGGABLE: usize = 60;

/// How long (seconds of the scene's time) a deal of the labels is shown before the
/// newest is shown instead: every deal while they come slowly, a few a second while
/// they come fast.
const DEAL_SHOWN: f32 = 0.22;

// ----- the scene ---------------------------------------------------------------------------------

pub struct TwoGroups {
    truth: Truth,
    seed: u64,
    /// The truth and seed the data were drawn from: they are drawn again when either
    /// changes; a dragged point keeps its place till then.
    drawn_from: (Truth, u64),
    data: [Vec<f64>; 2],
    /// Counts the changes to the data, and the version the analysis is of.
    version: u64,
    analysed: u64,
    analysis: Analysis,
    /// Shuffle the labels (asked for once, the shuffles start again after every change).
    shuffling: bool,
    shuffles: Shuffles,
    /// Turned by "shuffle again", for a new set of shuffles of the same data.
    round: u64,
    rate: f32,
    pace: Pace,
    /// Count over new studies (asked for once, it counts again after every change).
    counting: bool,
    studies: Studies,
    study_pace: Pace,
    /// The deal of the labels shown (see `advance_deal`): which, how many shuffles had
    /// been made when it was taken, and what its marks go from and to, and how far.
    deal: Option<Vec<bool>>,
    deal_taken: usize,
    deal_from: Vec<f32>,
    deal_to: Vec<f32>,
    deal_t: f32,
    look: Look,
    /// The canvas last drawn on (a new one is drawn at once, without gliding).
    canvas: Option<Vec2>,
    /// The point being dragged (group, index), and the data axis held still while it is.
    held: Option<(usize, usize)>,
    held_axis: Option<[f32; 2]>,
    /// The round values of the two axes as last drawn, and how many lines and labels the
    /// last frame drew (see the end of `frame`).
    ticks: [Ticks; 2],
    drawn_last: (usize, usize),
}

impl TwoGroups {
    /// (Nothing here needs the graphics device, so the tests can make one too.)
    fn create() -> TwoGroups {
        let empty: [Vec<f64>; 2] = [Vec::new(), Vec::new()];
        let mut s = TwoGroups {
            truth: CLEAR,
            seed: 0,
            drawn_from: (CLEAR, u64::MAX),
            analysis: analyse(&empty, false),
            shuffles: Shuffles::new(&empty, false, 0),
            data: empty,
            version: 0,
            analysed: u64::MAX,
            shuffling: false,
            round: 0,
            rate: SHUFFLE_RATE,
            pace: Pace::default(),
            counting: false,
            studies: Studies::new(),
            study_pace: Pace::default(),
            deal: None,
            deal_taken: 0,
            deal_from: Vec::new(),
            deal_to: Vec::new(),
            deal_t: 1.0,
            look: Look::default(),
            canvas: None,
            held: None,
            held_axis: None,
            ticks: [Ticks::NONE; 2],
            drawn_last: (0, 0),
        };
        s.set_up(OPENING);
        s
    }

    /// Set up one of the experiments in `TRIALS`.
    fn set_up(&mut self, k: usize) {
        let Some(trial) = TRIALS.get(k) else { return };
        self.truth = trial.truth;
        self.seed = trial.seed;
        self.rate = SHUFFLE_RATE;
        self.round = 0;
        (self.held, self.held_axis) = (None, None);
        self.redraw();
        if let Some(far) = trial.wild {
            let b = &mut self.data[1];
            if let Some(top) = (0..b.len()).max_by(|&i, &j| b[i].total_cmp(&b[j])) {
                b[top] = far;
                self.version += 1;
            }
        }
        self.shuffling = trial.shuffle;
        self.counting = trial.count;
        self.refresh();
    }

    /// Draw the data afresh from the truth and the seed.
    fn redraw(&mut self) {
        self.truth = self.truth.tidy();
        self.data = sample(&self.truth, self.seed);
        self.drawn_from = (self.truth, self.seed);
        self.version += 1;
    }

    /// Bring everything up to date with the settings and the data: the data drawn again
    /// if what they come from has changed, the analysis, and the shuffles and the count
    /// started afresh where they are kept and out of date.
    fn refresh(&mut self) {
        self.truth = self.truth.tidy();
        if self.drawn_from != (self.truth, self.seed) {
            self.redraw();
        }
        if self.analysed != self.version {
            self.analysis = analyse(&self.data, self.truth.paired);
            self.analysed = self.version;
        }
        let fresh = self.shuffles.version == self.version && self.shuffles.round == self.round;
        if self.shuffling && !fresh {
            self.restart_shuffles();
        }
        if !self.shuffling && self.shuffles.done() > 0 {
            self.shuffles = Shuffles::new(&[Vec::new(), Vec::new()], false, 0);
            self.forget_deal();
        }
        if self.counting && self.studies.truth != Some(self.truth) {
            self.studies.restart(self.truth);
            self.study_pace.reset();
        }
        if !self.counting && self.studies.truth.is_some() {
            self.studies = Studies::new();
        }
    }

    fn restart_shuffles(&mut self) {
        let mut s = Shuffles::new(&self.data, self.truth.paired, self.seed.wrapping_add(self.round.wrapping_mul(0x9E37_79B9)));
        s.version = self.version;
        s.round = self.round;
        s.range = null_aim(&self.analysis).0;
        self.shuffles = s;
        self.pace.reset();
        self.forget_deal();
    }

    fn forget_deal(&mut self) {
        self.deal = None;
        (self.deal_taken, self.deal_t) = (0, 1.0);
        self.deal_from.clear();
        self.deal_to.clear();
    }

    /// The deal shown. The newest deal of the labels is taken once the last has been
    /// shown for `DEAL_SHOWN` of the scene's time, and what it moves (the dealt groups'
    /// means; in pairs, the signed differences) goes there at an even pace over the next
    /// `DEAL_SHOWN`, from wherever the marks were drawn in the frame before: so they move
    /// a whole step in every frame and never stop between deals. (Glided there instead,
    /// they turned round through a standstill whenever a new deal lay the other way; and
    /// a leg that ended part of the way through a frame, the next starting from its end,
    /// could undo in that frame what the last did.)
    fn advance_deal(&mut self, dt: f32) {
        if !(self.shuffling && self.shuffles.done() > 0) {
            return;
        }
        if self.deal.is_none() {
            let d = self.shuffles.last_deal();
            self.deal_to = self.deal_values(Some(&d));
            self.deal_from = self.deal_to.clone();
            (self.deal, self.deal_taken, self.deal_t) = (Some(d), self.shuffles.done(), 0.0);
            return;
        }
        let here = self.shown_deal();
        self.deal_t += dt / DEAL_SHOWN;
        if self.deal_t < 1.0 {
            return;
        }
        if !(self.shuffles.pending() && self.shuffles.done() > self.deal_taken) {
            self.deal_t = 1.0;
            return;
        }
        let d = self.shuffles.last_deal();
        self.deal_to = self.deal_values(Some(&d));
        self.deal_from = if here.len() == self.deal_to.len() { here } else { self.deal_to.clone() };
        (self.deal, self.deal_taken) = (Some(d), self.shuffles.done());
        self.deal_t = (dt / DEAL_SHOWN).min(1.0);
    }

    /// The deal's marks where they are now: part of the way along their leg.
    fn shown_deal(&self) -> Vec<f32> {
        let t = self.deal_t.max(0.0).min(1.0);
        if self.deal_from.len() != self.deal_to.len() {
            return self.deal_to.clone();
        }
        self.deal_from.iter().zip(&self.deal_to).map(|(a, b)| a + (b - a) * t).collect()
    }

    /// The shuffles and the count, paced: so many a second of the scene's time (no
    /// shuffles while a point is held), within the frame's allowance.
    fn work(&mut self, dt: f32, real_dt: f32) {
        if dt <= 0.0 {
            return;
        }
        if self.shuffling && self.held.is_none() && self.shuffles.pending() {
            let mut turn = scene_turn(&mut self.pace, self.rate.max(1.0), dt, real_dt);
            while self.shuffles.pending() && turn.next() {
                self.shuffles.step();
            }
            self.pace.end(turn);
        }
        if self.counting && self.studies.pending() {
            let mut turn = scene_turn(&mut self.study_pace, STUDY_RATE, dt, real_dt);
            while self.studies.pending() && turn.next() {
                self.studies.step();
            }
            self.study_pace.end(turn);
        }
        self.advance_deal(dt);
    }

    /// The shuffles and the count to the end at once (the tests of this file).
    #[cfg(test)]
    fn settle(&mut self) {
        self.refresh();
        while self.shuffling && self.shuffles.pending() {
            self.shuffles.step();
        }
        self.studies.finish();
    }

    /// The data axis as it is aimed at: every value (in pairs, every difference and 0
    /// too), and a little room.
    fn axis_aim(&self) -> [f64; 2] {
        let mut all: Vec<f64> = self.data.iter().flatten().copied().collect();
        if self.analysis.paired {
            all.extend(differences(&self.data[0], &self.data[1]));
            all.push(0.0);
        }
        let (lo, hi) = all.iter().fold((f64::MAX, f64::MIN), |(lo, hi), v| (lo.min(*v), hi.max(*v)));
        if !(lo.is_finite() && hi.is_finite()) {
            return [-1.0, 1.0];
        }
        let span = hi - lo;
        if span < 1e-9 * (1.0 + lo.abs()) {
            return [lo - 1.0, hi + 1.0];
        }
        [lo - 0.06 * span, hi + 0.06 * span]
    }

    /// What the picture glides to.
    fn goals(&self) -> Goals {
        let a = &self.analysis;
        let points = [0, 1].map(|g| self.data[g].iter().map(|v| *v as f32).collect::<Vec<f32>>());
        let axis = match self.held_axis {
            Some(kept) => kept,
            None => self.axis_aim().map(|v| v as f32),
        };
        // the bells' height: the taller of the two, each a normal curve with its group's spread
        let peak = (0..2)
            .filter(|&g| a.sizes[g] >= 2 && a.sds[g] > tolerance(&self.data[g]))
            .map(|g| st::normal_pdf(0.0) / a.sds[g])
            .fold(0.0f64, f64::max);
        let bells = vec![if peak > 0.0 && peak.is_finite() { (1.12 * peak) as f32 } else { 1.0 }];
        let (range, top) = null_aim(a);
        let s = &self.shuffles;
        let showing = self.shuffling && s.done() > 0;
        // (the chart is as tall as the bell or the shuffles' tallest bar, as far as that is known)
        let top = if showing { top.max(1.08 * s.peak_seen) } else { top };
        let main = a.main().filter(|t| t.se.is_finite());
        let pooled = a.pooled.filter(|t| !a.paired && t.se.is_finite());
        let null = vec![
            range as f32,
            top as f32,
            main.map_or(0.0, |t| t.se as f32),
            main.map_or(1.0, |t| t.df as f32),
            pooled.map_or(0.0, |t| t.se as f32),
            pooled.map_or(1.0, |t| t.df as f32),
        ];
        let bars = if showing {
            let w = 2.0 * s.range / BINS as f64;
            s.bins.iter().map(|&c| (c as f64 / s.done() as f64 / w) as f32).collect()
        } else {
            vec![0.0; BINS]
        };
        let studies = (0..4).map(|k| self.studies.share(k).unwrap_or(0.0) as f32).collect();
        Goals { points, axis, bells, null, bars, studies }
    }

    /// What a deal moves: in two groups, the means of the points dealt to A and to B; in
    /// pairs, each difference with the sign the deal gave it, and their mean. Without a
    /// deal, the real ones.
    fn deal_values(&self, deal: Option<&Vec<bool>>) -> Vec<f32> {
        let [a, b] = [&self.data[0], &self.data[1]];
        if self.analysis.paired {
            let d = differences(a, b);
            let signed: Vec<f64> = match deal {
                Some(signs) if signs.len() == d.len() => d.iter().zip(signs).map(|(v, s)| if *s { *v } else { -*v }).collect(),
                _ => d,
            };
            let mean = st::mean(&signed);
            return signed.iter().map(|v| *v as f32).chain([mean as f32]).collect();
        }
        match deal {
            Some(to_b) if to_b.len() == a.len() + b.len() => {
                let (mut sums, mut counts) = ([0.0f64; 2], [0usize; 2]);
                for (v, g) in a.iter().chain(b).zip(to_b) {
                    sums[*g as usize] += v;
                    counts[*g as usize] += 1;
                }
                (0..2).map(|g| (sums[g] / counts[g].max(1) as f64) as f32).collect()
            }
            _ => vec![st::mean(a) as f32, st::mean(b) as f32],
        }
    }
}

/// The null chart's half-width and height, as aimed at: wide enough for nearly all of
/// the t curve and for the real difference, and as tall as the curve's peak.
fn null_aim(a: &Analysis) -> (f64, f64) {
    let main = a.main();
    let reach = |t: &TTest| t.se * Dist::StudentT { df: t.df.max(0.5) }.quantile(0.9995).min(12.0);
    let mut range = 1.25 * a.diff.abs();
    let mut top = 0.0f64;
    for t in main.iter().chain(a.pooled.iter().filter(|_| !a.paired)) {
        if t.se > 0.0 && t.se.is_finite() {
            range = range.max(reach(t));
            top = top.max(Dist::StudentT { df: t.df.max(0.5) }.pdf(0.0) / t.se);
        }
    }
    if !(range > 1e-9 && range.is_finite()) {
        range = 1.0;
    }
    if !(top > 0.0 && top.is_finite()) {
        top = 1.0 / range;
    }
    (range, 1.15 * top)
}

// ----- what glides ----------------------------------------------------------------------------------

/// What the picture glides to (see [`Look`]).
struct Goals {
    points: [Vec<f32>; 2],
    axis: [f32; 2],
    bells: Vec<f32>,
    /// The null chart: its half-width and height, and the standard error and degrees of
    /// freedom of the t curve drawn (then of the pooled one).
    null: Vec<f32>,
    bars: Vec<f32>,
    studies: Vec<f32>,
}

/// The picture as drawn: everything in it glides to where the numbers have gone, so
/// that a new sample, a setting changed, a deal of the labels and a histogram that fills
/// move in every frame, and come exactly to rest. Captions and read-outs say the true
/// numbers.
#[derive(Default)]
struct Look {
    points: [Glide; 2],
    axis: Glide,
    bells: Glide,
    null: Glide,
    bars: Glide,
    studies: Glide,
    moving: bool,
}

impl Look {
    fn follow(&mut self, g: &Goals, dt: f32, snap: bool) {
        let dt = if snap { 1e6 } else { dt };
        let mut moving = false;
        for k in 0..2 {
            moving |= self.points[k].follow(&g.points[k], dt, true);
        }
        moving |= self.axis.follow(&g.axis, dt, false);
        moving |= self.bells.follow(&g.bells, dt, false);
        moving |= self.null.follow(&g.null, dt, false);
        moving |= self.bars.follow(&g.bars, dt, false);
        moving |= self.studies.follow(&g.studies, dt, false);
        self.moving = moving;
    }
}

// ----- the layout ----------------------------------------------------------------------------------

/// Where everything goes on a canvas (logical pixels).
#[derive(Clone, Copy, Debug, PartialEq)]
struct Layout {
    view: Vec2,
    narrow: bool,
    /// Text sizes: headings, words, small print.
    big: f32,
    text: f32,
    small: f32,
    /// The buttons over the picture: the design (two groups, pairs) and "draw again".
    design: [Rect; 2],
    again: Rect,
    /// The data: the whole chart, the band of its bracket, the bells, the three rows,
    /// and where the rows' names go.
    data: Rect,
    bracket: f32,
    bells: Rect,
    rows: [Rect; 3],
    names: f32,
    /// The null distribution, and the line and button under it.
    null: Rect,
    note: f32,
    shuffle: Rect,
    /// The three tests (and, where there is room, what each assumes), and the new
    /// studies with their button: drawn if `blocks` (there is room for them).
    blocks: bool,
    tests: Rect,
    assumptions: bool,
    studies: Rect,
    count: Rect,
    /// Too small a canvas for the charts: the p-values alone.
    compact: bool,
}

/// A rectangle of the screen (top-left, bottom-right).
type Rect = (Vec2, Vec2);

/// The words on the buttons over the picture.
const DESIGNS: [&str; 2] = ["two groups", "pairs"];
const AGAIN: &str = "draw again";

/// A button as wide as its words, from `x` at `y` (measured without a frame).
fn button_at(text: &str, x: f32, y: f32, h: f32, size: f32) -> Rect {
    (Vec2::new(x, y), Vec2::new(x + text_width(text, size) + 18.0, y + h))
}

impl Layout {
    /// The layout of a canvas: everything if it fits; else the two charts alone (the
    /// p-values written under the second); else, on a canvas too small even for that,
    /// the numbers alone.
    fn of(view: Vec2, inset: f32) -> Layout {
        let full = Layout::build(view, inset, true);
        if full.fits() {
            return full;
        }
        let charts = Layout::build(view, inset, false);
        if charts.fits() {
            return charts;
        }
        Layout { compact: true, ..charts }
    }

    /// Is everything inside the canvas, with room to read it?
    fn fits(&self) -> bool {
        let tall_enough = self.data.1.y - self.data.0.y >= 110.0 && self.null.1.y - self.null.0.y >= 60.0 && self.null.1.x - self.null.0.x >= 200.0;
        let below = if self.blocks { self.studies.1.y - self.studies.0.y >= 70.0 && self.studies.1.y <= self.view.y } else { self.note + self.small <= self.view.y - 4.0 };
        tall_enough && below && self.view.x >= 300.0
    }

    fn build(view: Vec2, inset: f32, blocks: bool) -> Layout {
        let view = view.max(Vec2::splat(1.0));
        let narrow = view.x < 760.0;
        let (big, text, small) = if narrow { (13.0, 11.0, 10.0) } else { (15.0, 12.5, 11.5) };
        let side = if narrow { 10.0 } else { 22.0 };
        let top = inset.max(0.0) + if narrow { 6.0 } else { 12.0 };
        let row_h = if narrow { 22.0 } else { 24.0 };
        let names = if narrow { 50.0 } else { 84.0 };

        // the buttons at the top: at the right on a desktop, a row of their own on a phone
        let (design, again, data_top) = if narrow {
            let r = plot::even_row(3, side, view.x - side, top, row_h, 6.0);
            ([r[0], r[1]], r[2], top + row_h + 14.0)
        } else {
            let gap = 6.0;
            let w: Vec<f32> = DESIGNS.iter().chain([&AGAIN]).map(|t| text_width(t, text) + 18.0).collect();
            let x0 = view.x - side - w.iter().sum::<f32>() - 2.0 * gap - 18.0;
            let d0 = button_at(DESIGNS[0], x0, top, row_h, text);
            let d1 = button_at(DESIGNS[1], d0.1.x + gap, top, row_h, text);
            let ag = button_at(AGAIN, d1.1.x + gap + 18.0, top, row_h, text);
            ([d0, d1], ag, top + row_h + 14.0)
        };

        // the data: a band for the bracket, the bells, three rows of points (a short
        // window gives the data less, so that what is under it keeps its room)
        let data_h = match (narrow, blocks) {
            (true, true) => (view.y * 0.28).clamp(130.0, 250.0),
            (true, false) => (view.y * 0.36).clamp(110.0, 250.0),
            (false, _) if view.y < 760.0 => (view.y * 0.3).clamp(120.0, 230.0),
            _ => (view.y * 0.37).clamp(190.0, 370.0),
        };
        let data = (Vec2::new(side + names, data_top), Vec2::new(view.x - side, data_top + data_h));
        let band = if narrow { 20.0 } else { 26.0 };
        let bells_h = (data_h - band) * 0.34;
        let row = (data_h - band - bells_h) / 3.0;
        let bells = (Vec2::new(data.0.x, data.0.y + band), Vec2::new(data.1.x, data.0.y + band + bells_h));
        let rows = [0, 1, 2].map(|k| {
            let y = bells.1.y + k as f32 * row;
            (Vec2::new(data.0.x, y), Vec2::new(data.1.x, y + row))
        });
        // under it: the axis's numbers and name, and a line naming the marks
        let legend_y = data.1.y + small + 10.0 + small + 6.0;
        let lower = legend_y + small + if narrow { 16.0 } else { 40.0 };

        let (null, note, shuffle, tests, studies, count, assumptions);
        let button_h = if narrow { 20.0 } else { 22.0 };
        // (what goes under the null chart: its axis's numbers and name, then a line and a button)
        let under = 2.0 * small + 16.0;
        if narrow {
            let n0 = Vec2::new(side + 14.0, lower + text + 6.0);
            let null_h = if blocks { (view.y * 0.15).clamp(70.0, 170.0) } else { (view.y - n0.y - under - button_h - 10.0).min(260.0) };
            null = (n0, Vec2::new(view.x - side - 4.0, n0.y + null_h));
            note = null.1.y + under;
            let bw = text_width("shuffle the labels", small) + 16.0;
            shuffle = (Vec2::new(view.x - side - bw, note - 4.0), Vec2::new(view.x - side, note - 4.0 + button_h));
            let t0 = shuffle.1.y + 14.0;
            tests = (Vec2::new(side, t0), Vec2::new(view.x - side, t0 + text + 6.0 + 4.0 * (small + 5.0) + 4.0 + small + 2.0));
            studies = (Vec2::new(side, tests.1.y + 18.0), Vec2::new(view.x - side, view.y - 8.0));
            assumptions = false;
        } else {
            let split = if blocks { side + (view.x - 2.0 * side) * 0.57 } else { view.x - side };
            let bottom = view.y - 12.0;
            let n0 = Vec2::new(side + 20.0, lower + text + 14.0);
            let n1y = (bottom - (under + 2.0) - (button_h - 4.0)).max(n0.y + 40.0);
            null = (n0, Vec2::new(split - 10.0, n1y));
            note = null.1.y + under + 2.0;
            let bw = text_width("shuffle the labels 2000 times", small) + 18.0;
            shuffle = (Vec2::new(null.1.x - bw, note - 5.0), Vec2::new(null.1.x, note - 5.0 + button_h));
            let x0 = split + 36.0;
            let plain = text + 12.0 + 4.0 * (text + 10.0) + 8.0 + 2.0 * (small + 8.0);
            let fuller = plain + 4.0 * (small + 4.0);
            // (what each test assumes, under it, where that leaves room for the studies)
            assumptions = bottom - lower - fuller - 26.0 >= text + small + 4.0 * 20.0 + 40.0;
            let tests_h = if assumptions { fuller } else { plain };
            tests = (Vec2::new(x0, lower), Vec2::new(view.x - side, lower + tests_h));
            studies = (Vec2::new(x0, tests.1.y + 26.0), Vec2::new(view.x - side, bottom));
        }
        let cw = text_width("count 1000 new studies", small) + 18.0;
        count = (Vec2::new(studies.0.x, studies.0.y + text + 10.0), Vec2::new(studies.0.x + cw, studies.0.y + text + 10.0 + button_h));
        Layout {
            view,
            narrow,
            big,
            text,
            small,
            design,
            again,
            data,
            bracket: data.0.y + band * 0.55,
            bells,
            rows,
            names: side,
            null,
            note,
            shuffle,
            blocks,
            tests,
            assumptions,
            studies,
            count,
            compact: false,
        }
    }
}

/// Round values along an axis, chosen from where it is going and kept while it glides
/// there: each line and number keeps its place in its batch, sliding with the axis
/// (and unseen while it is off the chart). Chosen afresh from the axis as drawn, a
/// number would come in at one end as another went out at the other, and every
/// number would change places with its neighbour.
#[derive(Clone, Copy, Debug, PartialEq)]
struct Ticks {
    step: f64,
    first: i64,
    last: i64,
}

impl Ticks {
    const NONE: Ticks = Ticks { step: 1.0, first: 0, last: -1 };

    fn of(lo: f64, hi: f64, pixels: f32) -> Ticks {
        let span = hi - lo;
        if !span.is_finite() || span <= 0.0 || pixels.is_nan() || pixels <= 0.0 {
            return Ticks::NONE;
        }
        let step = nice_step((span / (pixels as f64 / 70.0).max(1.0)) as f32) as f64;
        let first = (lo / step).floor() as i64 - 2;
        let last = ((hi / step).ceil() as i64 + 2).min(first + 60);
        Ticks { step, first, last }
    }

    /// These round values while they still reach past both ends of an axis aimed at
    /// `lo` to `hi`, at the step it would be given: chosen afresh only when they must
    /// be, so that the numbers keep their places while the axis glides (in every frame
    /// of a drag, it would otherwise be a new set).
    fn keep_or(self, lo: f64, hi: f64, pixels: f32) -> Ticks {
        let new = Ticks::of(lo, hi, pixels);
        let covers = self.last >= self.first && self.first as f64 * self.step <= lo && self.last as f64 * self.step >= hi;
        if self.step == new.step && covers {
            self
        } else {
            new
        }
    }

    fn values(&self) -> impl Iterator<Item = f64> + '_ {
        (self.first..=self.last).map(|k| k as f64 * self.step)
    }

    fn name(&self, v: f64) -> String {
        let decimals = (-self.step.log10().floor()).max(0.0).min(6.0) as usize;
        num(v, decimals)
    }
}

/// The line kinds this scene draws in batches of their own (see `plot::apart`).
const PAIRS: u8 = plot::OWN;
const UPWARDS: u8 = plot::OWN + 1;
const DEALT: u8 = plot::OWN + 2;
const BRACKETS: u8 = plot::OWN + 3;
const OBSERVED: u8 = plot::OWN + 4;
const LEVEL_LINE: u8 = plot::OWN + 5;
const SHUFFLE_MARK: u8 = plot::OWN + 6;
const POOLED_CURVE: u8 = plot::OWN + 7;
const KEY: u8 = plot::OWN + 8;

/// Where a point sits up its row: the same place for the same point every time.
fn jitter(g: usize, i: usize) -> f32 {
    let k = (g * MOST + i) as f32 + 1.0;
    0.14 + 0.72 * (k * 0.618_034).fract()
}

/// The first of `options` that fits in `room` logical pixels at `size` (the last if none does).
fn pick(f: &Frame, options: &[String], room: f32, size: f32) -> String {
    options.iter().find(|t| f.text_size(t, size).x <= room).or(options.last()).cloned().unwrap_or_default()
}

/// A p-value as written: two significant digits, and "< 0.0001" below that.
fn p_text(p: f64) -> String {
    if !p.is_finite() {
        return "?".to_string();
    }
    if p < 1e-4 {
        return "< 0.0001".to_string();
    }
    let decimals = (1.0 - p.log10().floor()).max(2.0).min(5.0) as usize;
    num(p, decimals)
}

/// `$p = …$` or `$p < 0.0001$`.
fn p_formula(p: f64) -> String {
    let t = p_text(p);
    if t.starts_with('<') {
        format!("$p {t}$")
    } else {
        format!("$p = {t}$")
    }
}

/// A share as a percentage, to one decimal (a count of 1000 is exact to that).
fn percent(share: f64) -> String {
    format!("{} %", num(100.0 * share, 1))
}

impl Scene for TwoGroups {
    fn view(&self) -> View {
        View::Flat { center: Vec2::ZERO, half_height: 1.0 }
    }

    fn settings(&self, s: &mut ViewSettings) {
        s.grid = false;
        s.bloom = s.bloom.min(0.2);
    }

    fn animated(&self) -> bool {
        self.look.moving || (self.shuffling && self.shuffles.pending()) || (self.counting && self.studies.pending()) || self.held.is_some() || (self.deal.is_some() && self.deal_t < 1.0)
    }

    fn pointer_hint(&self) -> Option<&'static str> {
        Some("dots: drag one along its row    buttons over the picture: click")
    }

    fn ui(&mut self, ui: &mut egui::Ui, _: &Gfx) {
        ui::note(ui, r"Two groups of measurements, $A$ and $B$, and a difference between their means. Is it more than chance would give if the groups were really alike? Three ways to ask, each assuming something different. The $t$ test weighs the difference against its standard error, how far a difference of means wanders from sample to sample:");
        ui::equation(ui, r"t = \frac{\bar x_B - \bar x_A}{\sqrt{s_A^2 / n_A + s_B^2 / n_B}}");
        ui::note(ui, r"($\bar x$ a group's mean, $s$ its standard deviation, $n$ its size), and compares $t$ with Student's $t$ distribution: what $t$ follows, nearly, when nothing is going on and both groups are bell-shaped (exactly, if their spreads are equal and $t$ is worked out with one spread pooled from both groups: the pooled test). Shuffling the labels assumes no shape at all: if the labels meant nothing, dealing them afresh would give differences like the real one. A rank test uses only the order of the values. Then two more questions: how large the difference is, which is not how sure we are of it; and whether the same units were measured twice.");

        ui::section(ui, "Things to try");
        for (k, trial) in TRIALS.iter().enumerate() {
            if ui::experiment(ui, trial.button, trial.caption) {
                self.set_up(k);
            }
        }

        ui::section(ui, "The groups");
        let mut design = self.truth.paired as usize;
        if ui::choice(ui, &["two separate groups", "pairs: each unit measured twice"], &mut design) {
            self.truth.paired = design == 1;
        }
        if self.truth.paired {
            let mut rho = self.truth.rho as f32;
            if ui::slider(ui, r"how closely a unit's two values go together, $\rho$", &mut rho, -1.0..=1.0) {
                self.truth.rho = rho as f64;
            }
            ui::note(ui, r"The correlation of $A$ and $B$ within a unit. $B$ is made as $\rho$ times the unit's own standard draw for $A$ plus $\sqrt{1 - \rho^2}$ times one of its own, then moved and stretched to $B$'s mean and spread.");
        }
        for (g, name) in [(0, "$A$"), (1, "$B$")] {
            let mut mean = self.truth.mean[g] as f32;
            if ui::slider(ui, &format!("{name}: true mean"), &mut mean, -3.0..=3.0) {
                self.truth.mean[g] = mean as f64;
            }
            let mut sd = self.truth.sd[g] as f32;
            if ui::slider(ui, &format!("{name}: true spread (standard deviation)"), &mut sd, 0.0..=5.0) {
                self.truth.sd[g] = sd as f64;
            }
            if !self.truth.paired {
                ui::slider_int_log(ui, &format!("{name}: points"), &mut self.truth.n[g], 1..=MOST);
            }
        }
        if self.truth.paired {
            ui::slider_int_log(ui, "units (each with an $A$ and a $B$)", &mut self.truth.n[0], 1..=MOST);
        }
        let mut shape = Shape::ALL.iter().position(|s| *s == self.truth.shape).unwrap_or(0);
        let names: Vec<&str> = Shape::ALL.iter().map(|s| s.name()).collect();
        if ui::choice(ui, &names, &mut shape) {
            self.truth.shape = Shape::ALL[shape];
        }
        ui::note(ui, "The shape of both groups. Skewed: a long tail to the right. Heavy tails: now and then a value far out (Student's $t$ with 3 degrees of freedom).");
        if ui.button("draw again").clicked() {
            self.seed = self.seed.wrapping_add(1);
        }

        ui::section(ui, "The tests");
        ui::slider_log(ui, "shuffles per second", &mut self.rate, 1.0..=5000.0);
        ui::note(ui, "A few a second shows every deal of the labels; at hundreds a second the histogram fills in seconds.");
        ui::note(ui, "The shuffles' histogram is what chance gives if the labels mean nothing. Where the groups really differ, it comes out a little wider than the $t$ curve, because every deal mixes the real difference into both groups (in pairs with a large difference, much wider). It is still the right yardstick for the $p$-value, which asks what chance alone would give.");
        ui.horizontal_wrapped(|ui| {
            let text = if self.shuffling { "stop shuffling" } else { "shuffle the labels 2000 times" };
            if ui.button(text).clicked() {
                self.shuffling = !self.shuffling;
            }
            if !self.counting && ui.button("count 1000 new studies").clicked() {
                self.counting = true;
            }
        });
        ui::note(ui, "The new studies are fresh samples from the same groups (not the points shown, which may have been dragged), each put through all four tests; in each, the shuffling test makes 199 shuffles.");
        self.refresh();

        ui::section(ui, "Reading the picture");
        ui::key(
            ui,
            &[
                (DATA, "cyan: group $A$: its points, its mean (the tall tick) and its bell"),
                (OTHER, "orange: group $B$, the same"),
                (ESTIMATE, "yellow: the difference of the means (the bracket), and where it falls among the differences chance gives; in pairs, each unit's difference $B - A$"),
                (ASSUMED, "purple: what chance alone gives if the labels mean nothing: Student's $t$ (the curve; dashed, with a pooled spread) and the shuffles (the bars)"),
                (ALARM, "pink: as far from 0 as the real difference, or further: its share is the $p$-value; in the new studies, rejections that are false alarms"),
                (TRUTH, "green: in the new studies, rejections that are right"),
            ],
        );
        ui::section(ui, "Words");
        ui::term(ui, "Standard error:", r"how far a difference of two means typically wanders from sample to sample, estimated from the data as $\sqrt{s_A^2/n_A + s_B^2/n_B}$.");
        ui::term(ui, "Two-sample $t$ test:", r"$t$, the difference over its standard error, compared with Student's $t$ distribution. Exact for bell-shaped groups with equal spreads (the pooled test); with unequal spreads Welch's version is a close approximation; and close for other shapes once the groups are large.");
        ui::term(ui, "Welch's test:", "the $t$ test with each group's own spread, its degrees of freedom worked out from the spreads and sizes (Welch and Satterthwaite). The one drawn here.");
        ui::term(ui, "Pooled variance:", r"one spread from both groups together, $((n_A - 1) s_A^2 + (n_B - 1) s_B^2) / (n_A + n_B - 2)$: the classical $t$ test's, right only if the true spreads are equal.");
        ui::term(ui, "Degrees of freedom:", "the parameter of Student's $t$ distribution: the fewer, the heavier its tails.");
        ui::term(ui, "$p$-value:", "if the groups were alike, the probability of a difference at least as far from 0 as the one seen. Not the probability that they are alike.");
        ui::term(ui, "Permutation test:", "deal the labels afresh many times; the $p$-value is the share of the deals, the real one counted among them, that give a difference at least as far from 0. It assumes only that the groups were alike in every way if the labels mean nothing, so that every deal was as likely as the real one.");
        ui::term(ui, "Rank test (Mann–Whitney):", r"put all the values in order; $U$ counts the pairs (one value from each group) in which $B$'s is the larger, ties counting a half. How far out the extreme values lie does not matter, only their order. Its $p$-value is counted out exactly for up to 50 values without ties, and otherwise from a normal approximation.");
        ui::term(ui, "Paired test:", r"the same units measured twice: test the mean of each unit's difference $B - A$ against 0, with $t$ on $n - 1$ degrees of freedom. Shuffling then swaps a unit's two values or not; the rank test is Wilcoxon's signed-rank test.");
        ui::term(ui, "Effect size (Cohen's $d$):", "the difference of the means in units of the pooled standard deviation: how far apart the groups lie, whatever the sample size.");
        ui::term(ui, r"$P(B > A)$:", r"the share of the pairs (one point from each group) in which $B$'s is the larger: $U$ over $n_A n_B$. For two bells of the same spread it is $\Phi(d / \sqrt{2})$ ($\Phi$ the standard normal's cumulative probability): 76 % at $d = 1$.");

        ui::section(ui, "Numbers");
        self.readouts(ui);
        ui::see_also(
            ui,
            &[
                ("testing", "the logic of a test, its errors and its power"),
                ("bootstrap", "resampling to estimate, not to test"),
                ("anova", "more than two groups (with two, the pooled $t$ test squared)"),
                ("confound", "a difference between groups is not yet an effect"),
            ],
        );
    }

    fn frame(&mut self, f: &mut Frame) {
        let lay = Layout::of(f.view_size(), f.top_inset);
        // (a canvas squeezed to a sliver is one the controls have opened over, on a phone)
        let tiny = lay.view.x < 160.0 || lay.view.y < 160.0;
        if !tiny && !lay.compact {
            if self.draw_buttons(f, &lay) {
                self.refresh();
            }
            self.handles(f, &lay);
        } else {
            (self.held, self.held_axis) = (None, None);
        }
        self.refresh();
        self.work(f.dt, f.real_dt);
        let goals = self.goals();
        let snap = self.canvas != Some(lay.view) || self.look.points[0].shown.is_empty() && self.look.points[1].shown.is_empty();
        if !tiny {
            self.canvas = Some(lay.view);
        }
        self.look.follow(&goals, f.real_dt, snap);
        // (the point held is drawn where the hand puts it, on an axis held still)
        if let Some((g, i)) = self.held {
            if let (Some(v), Some(shown)) = (self.data[g].get(i), self.look.points[g].shown.get_mut(i)) {
                *shown = *v as f32;
            }
            self.look.axis.snap(&goals.axis);
        }
        if tiny {
            return;
        }
        if lay.compact {
            self.draw_compact(f, &lay);
            return;
        }
        let ticks = [
            self.ticks[0].keep_or(goals.axis[0] as f64, goals.axis[1] as f64, lay.data.1.x - lay.data.0.x),
            self.ticks[1].keep_or(-(goals.null[0] as f64), goals.null[0] as f64, lay.null.1.x - lay.null.0.x),
        ];
        let new_ticks = ticks != self.ticks;
        self.ticks = ticks;
        keep_batches(f);
        self.draw_data(f, &lay);
        self.draw_null(f, &lay);
        if lay.blocks {
            self.draw_tests(f, &lay);
            self.draw_studies(f, &lay);
        }
        // new round values on an axis: make sure the axes' batch and the labels are not
        // as many as in the last frame, so that nothing takes the new ones for the old
        // ones moved
        let count = |f: &mut Frame| (plot::apart(f, plot::AXES, false).pts.iter().filter(|p| p.width >= 0.0).count(), f.dl.labels.len());
        if new_ticks {
            let (lines, labels) = count(f);
            if lines == self.drawn_last.0 {
                let far = f.hud(Vec2::ZERO);
                plot::apart(f, plot::AXES, false).segment(far, far, 1.0, Color::CLEAR);
            }
            if labels == self.drawn_last.1 {
                f.text_px(Vec2::ZERO, "", 1.0, Color::CLEAR, Align2::LEFT_TOP);
            }
        }
        self.drawn_last = count(f);
    }
}

impl TwoGroups {
    /// The panel's numbers.
    fn readouts(&self, ui: &mut egui::Ui) {
        let a = &self.analysis;
        let p = |p: f64| p_text(p);
        let names = ["$A$", "$B$"];
        for g in 0..2 {
            let sd = if a.sizes[g] >= 2 { num(a.sds[g], 2) } else { "none".to_string() };
            ui::readout_colored(ui, &format!("{}: points, mean, spread", names[g]), format!("{}, {}, {}", a.sizes[g], num(a.means[g], 2), sd), if g == 0 { DATA } else { OTHER });
        }
        ui::readout(ui, r"difference of the means, $\bar x_B - \bar x_A$", num(a.diff, 3));
        let t_line = |t: &TTest| format!("{}, {}, {}, {}", num(t.se, 3), num(t.t, 2), num(t.df, 1), p(t.p));
        if a.paired {
            match a.paired_t {
                Some(t) => ui::readout(ui, "paired: SE, $t$, df, $p$", t_line(&t)),
                None => ui::readout(ui, "paired $t$ test", "none"),
            }
            match a.welch {
                Some(t) => ui::readout(ui, "ignoring the pairs (Welch): SE, $t$, df, $p$", t_line(&t)),
                None => ui::readout(ui, "ignoring the pairs (Welch)", "none"),
            }
            if let Some(r) = a.corr {
                ui::readout(ui, "correlation within a unit", num(r, 2));
            }
        } else {
            match a.welch {
                Some(t) => ui::readout(ui, "Welch: SE, $t$, df, $p$", t_line(&t)),
                None => ui::readout(ui, "Welch's $t$ test", "none"),
            }
            match a.pooled {
                Some(t) => ui::readout(ui, "pooled: SE, $t$, df, $p$", t_line(&t)),
                None => ui::readout(ui, "pooled $t$ test", "none"),
            }
        }
        let s = &self.shuffles;
        if self.shuffling && s.done() > 0 {
            ui::readout(ui, "shuffles reaching it, of all, $p$", format!("{}, {}, {}", s.reached, s.done(), p(s.p())));
        }
        match a.ranks {
            Some(r) => {
                let name = if a.paired { "signed ranks: $V$, $p$" } else { "ranks: $U$, $p$" };
                let how = if r.exact { "exact" } else { "approximate" };
                ui::readout(ui, name, format!("{}, {} ({how})", num(r.stat, 1), p(r.p)));
            }
            None => ui::readout(ui, "rank test", "none"),
        }
        ui::readout(ui, "Cohen's $d$", a.d.map_or("none".to_string(), |d| num(d, 2)));
        ui::readout(ui, r"$P(B > A)$", a.superiority.map_or("none".to_string(), |v| num(v, 3)));
        if self.counting {
            ui::readout(ui, "new studies counted", format!("{} of {STUDIES}", self.studies.done));
        }
    }

    /// The buttons over the picture; true if a setting changed.
    fn draw_buttons(&mut self, f: &mut Frame, lay: &Layout) -> bool {
        let mut changed = false;
        for (k, r) in lay.design.iter().enumerate() {
            let on = self.truth.paired == (k == 1);
            if plot::button(f, *r, DESIGNS[k], lay.text, on) && !on {
                self.truth.paired = k == 1;
                changed = true;
            }
        }
        if plot::button(f, lay.again, AGAIN, lay.text, false) {
            self.seed = self.seed.wrapping_add(1);
            changed = true;
        }
        let s = &self.shuffles;
        let text = match (self.shuffling, s.pending()) {
            (false, _) => if lay.narrow { "shuffle the labels" } else { "shuffle the labels 2000 times" },
            (true, true) => "stop",
            (true, false) => "shuffle again",
        };
        // (as wide as its longest words, whatever it says now: a button that changed its
        // size as its words changed would jump)
        if plot::button(f, lay.shuffle, text, lay.small, false) {
            match (self.shuffling, s.pending()) {
                (false, _) => self.shuffling = true,
                (true, true) => self.shuffling = false,
                (true, false) => self.round = self.round.wrapping_add(1),
            }
            changed = true;
        }
        if lay.blocks && !self.counting && plot::button(f, lay.count, "count 1000 new studies", lay.small, false) {
            self.counting = true;
            changed = true;
        }
        changed
    }

    /// The data chart as drawn: the values across, gliding.
    fn data_plot(&self, lay: &Layout) -> Plot {
        let x = [self.look.axis.get(0) as f64, self.look.axis.get(1) as f64];
        Plot::new(lay.data, x, [0.0, 1.0])
    }

    /// The points, each a handle that can be dragged along its row (in groups small
    /// enough to have room for that). Past either end of the chart, the axis widens.
    fn handles(&mut self, f: &mut Frame, lay: &Layout) {
        let plot = self.data_plot(lay);
        let mut held = None;
        let mut moved: Option<(usize, usize, f64)> = None;
        for g in 0..2 {
            if self.data[g].len() > DRAGGABLE {
                continue;
            }
            let color = if g == 0 { DATA } else { OTHER };
            let r = lay.rows[g];
            for i in 0..self.data[g].len() {
                let x = plot.px_x(self.look.points[g].get(i) as f64).max(r.0.x).min(r.1.x);
                let at = Vec2::new(x, r.0.y + jitter(g, i) * (r.1.y - r.0.y));
                if !at.is_finite() {
                    continue;
                }
                let Some(world) = f.cam.pick_ground(at * f.px) else { continue };
                let mut p = world;
                if f.drag3_sized(100 + (g * MOST + i) as u32, &mut p, None, color, dot_size(self.data[g].len(), lay.narrow) + 1.0) {
                    held = Some((g, i));
                    if let Some(s) = on_screen(f, p) {
                        moved = Some((g, i, plot.x_at(s.x.max(r.0.x).min(r.1.x))));
                    }
                }
            }
        }
        if f.handles.dragging() && held.is_none() {
            held = self.held;
        }
        if !f.handles.dragging() {
            held = None;
        }
        match (held, self.held) {
            (Some(_), None) => self.held_axis = Some([self.look.axis.get(0), self.look.axis.get(1)]),
            (None, Some(_)) => self.held_axis = None,
            _ => {}
        }
        self.held = held;
        // past an end of the chart the value goes on that way, the further out the
        // faster, its dot waiting at the edge; the axis stays still until it is let go,
        // and then glides to take it in
        if let (Some((g, i)), Some(kept), Some(ptr)) = (self.held, self.held_axis, plot::pointer(f)) {
            let span = (kept[1] - kept[0]).max(1e-6) as f64;
            let rate = span * 0.6 * f.real_dt.min(0.1) as f64;
            let (left, right) = (lay.data.0.x, lay.data.1.x);
            let now = self.data[g].get(i).copied().unwrap_or(0.0);
            if ptr.x > right {
                moved = Some((g, i, now.max(kept[1] as f64) + rate * ((ptr.x - right) / 40.0).min(3.0) as f64));
            } else if ptr.x < left {
                moved = Some((g, i, now.min(kept[0] as f64) - rate * ((left - ptr.x) / 40.0).min(3.0) as f64));
            }
        }
        if let Some((g, i, v)) = moved {
            if let Some(x) = self.data[g].get_mut(i) {
                if v.is_finite() && *x != v {
                    *x = v.max(-60.0).min(60.0);
                    self.version += 1;
                }
            }
        }
    }

    fn draw_data(&self, f: &mut Frame, lay: &Layout) {
        let plot = self.data_plot(lay);
        let (min, max) = lay.data;
        let paired = self.analysis.paired;
        hud_panel(f, Vec2::new(lay.names - 6.0, min.y - 8.0), Vec2::new(max.x + 8.0, max.y + 2.0 * lay.small + 16.0));
        let pts = [&self.look.points[0].shown, &self.look.points[1].shown];
        let mean_of = |v: &[f32]| if v.is_empty() { f64::NAN } else { v.iter().map(|x| *x as f64).sum::<f64>() / v.len() as f64 };
        let sd_of = |v: &[f32]| {
            let m = mean_of(v);
            if v.len() < 2 {
                return f64::NAN;
            }
            (v.iter().map(|x| (*x as f64 - m).powi(2)).sum::<f64>() / (v.len() - 1) as f64).sqrt()
        };
        let means = [mean_of(pts[0]), mean_of(pts[1])];
        let colors = [DATA, OTHER];

        // the bells: a normal curve with each group's mean and spread, faintly
        let bells = Plot::new(lay.bells, plot.x, [0.0, self.look.bells.get(0).max(1e-6) as f64]);
        for g in 0..2 {
            let (m, s) = (means[g], sd_of(pts[g]));
            if !(s > 1e-9 && m.is_finite()) {
                continue;
            }
            let bell = |x: f64| st::normal_pdf((x - m) / s) / s;
            bells.area(f, bells.x[0], bells.x[1], 0.0, 120, colors[g].a(0.04), bell);
            held_curve(f, &bells, plot::CURVES, 121, 1.6, colors[g].a(0.7), bell, false);
        }
        line_of(f, plot::AXES, Vec2::new(min.x, lay.bells.1.y), Vec2::new(max.x, lay.bells.1.y), 1.0, pal::DIM.a(0.35));
        for r in &lay.rows[1..] {
            line_of(f, plot::AXES, Vec2::new(min.x, r.0.y), Vec2::new(max.x, r.0.y), 1.0, pal::FAINT.a(0.5));
        }

        // the points (in pairs, each unit's two joined)
        let row_y = |g: usize, i: usize, k: usize| lay.rows[k].0.y + jitter(g, i) * (lay.rows[k].1.y - lay.rows[k].0.y);
        let keep_x = |x: f64| plot.px_x(x).max(min.x).min(max.x);
        let n = [pts[0].len(), pts[1].len()];
        if paired {
            for i in 0..n[0].min(n[1]) {
                let (a, b) = (Vec2::new(keep_x(pts[0][i] as f64), row_y(0, i, 0)), Vec2::new(keep_x(pts[1][i] as f64), row_y(1, i, 1)));
                line_of(f, PAIRS, a, b, 1.0, pal::DIM.a(0.45));
            }
        }
        for g in 0..2 {
            // (a group small enough to drag is drawn by its handles)
            if self.data[g].len() <= DRAGGABLE && !lay.compact {
                continue;
            }
            let size = dot_size(n[g], lay.narrow);
            let alpha = if n[g] > 300 { 0.5 } else { 0.85 };
            for i in 0..n[g] {
                disc(f, Vec2::new(keep_x(pts[g][i] as f64), row_y(g, i, g)), size, colors[g].a(alpha));
            }
        }

        // a point held beyond an end of the axis: its value, at that end
        if let Some((g, i)) = self.held {
            let v = self.data[g].get(i).copied().unwrap_or(0.0);
            let r = lay.rows[g];
            if v > plot.x[1] {
                f.caption_px(Vec2::new(max.x - 4.0, r.0.y + 3.0), format!("{} →", num(v, 1)), lay.small, colors[g], Align2::RIGHT_TOP);
            } else if v < plot.x[0] {
                f.caption_px(Vec2::new(min.x + 4.0, r.0.y + 3.0), format!("← {}", num(v, 1)), lay.small, colors[g], Align2::LEFT_TOP);
            }
        }

        // the means: a tall tick in each row, a dashed line up to the bracket
        for g in 0..2 {
            if !means[g].is_finite() {
                continue;
            }
            let x = keep_x(means[g]);
            let r = lay.rows[g];
            line_of(f, plot::MARKS, Vec2::new(x, r.0.y + 1.0), Vec2::new(x, r.1.y - 1.0), 2.8, colors[g]);
            dashed_of(f, UPWARDS, Vec2::new(x, r.0.y), Vec2::new(x, lay.bracket - 2.0), 1.2, colors[g].a(0.7));
        }
        // the bracket: the difference of the means
        if means[0].is_finite() && means[1].is_finite() {
            let (xa, xb, y) = (keep_x(means[0]), keep_x(means[1]), lay.bracket);
            line_of(f, BRACKETS, Vec2::new(xa, y), Vec2::new(xb, y), 2.0, ESTIMATE);
            for x in [xa, xb] {
                line_of(f, BRACKETS, Vec2::new(x, y - 4.0), Vec2::new(x, y + 6.0), 2.0, ESTIMATE);
            }
            let d = num(self.analysis.diff, 2);
            let words = [format!(r"difference of the means: $\bar x_B - \bar x_A = {d}$"), format!(r"$\bar x_B - \bar x_A = {d}$"), d.clone()];
            let room = (max.x - min.x) * 0.9;
            let text = pick(f, &words, room, lay.text);
            let w = f.text_size(&text, lay.text).x * 0.5 + 4.0;
            let cx = (0.5 * (xa + xb)).max(min.x + w).min(max.x - w);
            f.text_px(Vec2::new(cx, y - 5.0), text, lay.text, ESTIMATE, Align2::CENTER_BOTTOM);
        }

        // the third row: one deal of the labels, or in pairs each unit's difference
        let third = lay.rows[2];
        if paired {
            self.draw_differences(f, lay, &plot);
        } else {
            let deal = self.deal.as_ref().filter(|d| d.len() == n[0] + n[1] && self.shuffling);
            let size = dot_size(n[0].max(n[1]), lay.narrow) - 1.0;
            for g in 0..2 {
                for i in 0..n[g] {
                    let color = match deal {
                        Some(d) => colors[d[g * n[0] + i] as usize].a(0.85),
                        None => pal::DIM.a(0.5),
                    };
                    disc(f, Vec2::new(keep_x(pts[g][i] as f64), row_y(g, i, 2)), size, color);
                }
            }
            if deal.is_some() {
                let dm = self.shown_deal();
                if dm.len() == 2 {
                    for g in 0..2 {
                        let x = keep_x(dm[g] as f64);
                        line_of(f, DEALT, Vec2::new(x, third.0.y + 1.0), Vec2::new(x, third.1.y - 1.0), 2.4, colors[g]);
                    }
                    let label = format!(r"this deal: $\bar x_B - \bar x_A = {}$", num((dm[1] - dm[0]) as f64, 2));
                    f.caption_px(Vec2::new(max.x - 6.0, third.0.y + 3.0), label, lay.small, ASSUMED, Align2::RIGHT_TOP);
                }
            } else {
                let words = [format!("the same {} points, to be dealt afresh into groups of {} and {}", n[0] + n[1], n[0], n[1]), "to be dealt afresh".to_string()];
                let text = pick(f, &words, (max.x - min.x) - 12.0, lay.small);
                f.caption_px(Vec2::new(max.x - 6.0, third.0.y + 3.0), text, lay.small, pal::DIM, Align2::RIGHT_TOP);
            }
        }

        // the rows' names
        let named = |f: &mut Frame, k: usize, name: &str, color: Color, under: String| {
            let r = lay.rows[k];
            let mid = 0.5 * (r.0.y + r.1.y);
            f.text_px(Vec2::new(lay.names, mid - 1.0), name, lay.big, color, Align2::LEFT_BOTTOM);
            f.text_px(Vec2::new(lay.names, mid + 1.0), under, lay.small, pal::DIM, Align2::LEFT_TOP);
        };
        let count = |n: usize| match (n, paired) {
            (1, true) => "1 unit".to_string(),
            (1, false) => "1 point".to_string(),
            (n, true) => format!("{n} units"),
            (n, false) => format!("{n} points"),
        };
        named(f, 0, "$A$", DATA, count(n[0]));
        named(f, 1, "$B$", OTHER, count(n[1]));
        if paired {
            named(f, 2, "$B - A$", ESTIMATE, if lay.narrow { String::new() } else { "each unit".to_string() });
        } else {
            named(f, 2, if lay.narrow { "dealt" } else { "shuffled" }, pal::TEXT, if lay.narrow { String::new() } else { "one deal".to_string() });
        }

        // the axis, and a line naming the marks
        let ticks = self.ticks[0];
        axis(f, &plot, ticks, lay.small);
        f.text_px(Vec2::new(max.x, max.y + lay.small + 10.0), "value", lay.small, pal::DIM, Align2::RIGHT_TOP);
        let legend = if paired {
            [
                "dots: the measurements; grey lines join a unit's two; tall ticks: the means; faint bells: normal curves with each group's mean and spread; drag a dot".to_string(),
                "tall ticks: the means; grey lines: a unit's two values".to_string(),
                "ticks: means".to_string(),
            ]
        } else {
            [
                "dots: the measurements; tall ticks: each group's mean; faint bells: normal curves with each group's mean and spread; drag a dot".to_string(),
                "tall ticks: the means; bells: normal curves".to_string(),
                "ticks: means".to_string(),
            ]
        };
        let text = pick(f, &legend, max.x - lay.names - 60.0, lay.small);
        f.text_px(Vec2::new(lay.names, max.y + lay.small + 10.0), text, lay.small, pal::DIM, Align2::LEFT_TOP);
    }

    /// In pairs, the third row: each unit's difference, their mean against 0, and while
    /// shuffling, the differences with the signs of one deal.
    fn draw_differences(&self, f: &mut Frame, lay: &Layout, plot: &Plot) {
        let third = lay.rows[2];
        let (min, max) = lay.data;
        let keep_x = |x: f64| plot.px_x(x).max(min.x).min(max.x);
        let pts = [&self.look.points[0].shown, &self.look.points[1].shown];
        let n = pts[0].len().min(pts[1].len());
        let y = |i: usize| third.0.y + jitter(2, i) * (third.1.y - third.0.y);
        let size = dot_size(n, lay.narrow) - 0.5;
        // 0, which the null says the differences' mean is
        let zero = keep_x(0.0);
        dashed_of(f, LEVEL_LINE, Vec2::new(zero, third.0.y), Vec2::new(zero, third.1.y), 1.4, ASSUMED.a(0.8));
        let shuffling = self.shuffling && self.shuffles.pending();
        let signed = &self.shown_deal();
        for i in 0..n {
            let d = (pts[1][i] - pts[0][i]) as f64;
            disc(f, Vec2::new(keep_x(d), y(i)), size, ESTIMATE.a(if shuffling { 0.35 } else { 0.85 }));
        }
        if shuffling && signed.len() == n + 1 {
            for i in 0..n {
                disc(f, Vec2::new(keep_x(signed[i] as f64), y(i)), size, ASSUMED.a(0.9));
            }
            let x = keep_x(signed[n] as f64);
            line_of(f, DEALT, Vec2::new(x, third.0.y + 1.0), Vec2::new(x, third.1.y - 1.0), 2.4, ASSUMED);
            let label = format!("random signs: mean {}", num(signed[n] as f64, 2));
            f.caption_px(Vec2::new(max.x - 6.0, third.0.y + 3.0), label, lay.small, ASSUMED, Align2::RIGHT_TOP);
        }
        let mean = (0..n).map(|i| (pts[1][i] - pts[0][i]) as f64).sum::<f64>() / n.max(1) as f64;
        if n > 0 && mean.is_finite() {
            let x = keep_x(mean);
            line_of(f, plot::MARKS, Vec2::new(x, third.0.y + 1.0), Vec2::new(x, third.1.y - 1.0), 2.8, ESTIMATE);
        }
    }

    /// The null distribution: Student's t for the difference, the shuffles' histogram,
    /// the real difference and the tails beyond it.
    fn draw_null(&self, f: &mut Frame, lay: &Layout) {
        let a = &self.analysis;
        let nl = &self.look.null.shown;
        if nl.len() < 6 {
            return;
        }
        let (range, top) = (nl[0].max(1e-9) as f64, nl[1].max(1e-12) as f64);
        let plot = Plot::new(lay.null, [-range, range], [0.0, top]);
        let (min, max) = lay.null;
        plot.backing(f, if lay.narrow { 18.0 } else { 26.0 }, lay.small + 16.0, lay.text + 12.0, 10.0);
        let titles = [
            "if the labels meant nothing: the differences chance alone gives".to_string(),
            "what chance alone gives".to_string(),
            "chance alone".to_string(),
        ];
        let title = pick(f, &titles, max.x - min.x + 10.0, lay.text);
        f.text_px(Vec2::new(min.x - if lay.narrow { 12.0 } else { 18.0 }, min.y - 8.0), title, lay.text, pal::TEXT, Align2::LEFT_BOTTOM);

        // the shuffles
        let s = &self.shuffles;
        let diff = self.drawn_difference();
        let far = diff.abs();
        let bars = &self.look.bars.shown;
        if self.shuffling && s.done() > 0 && bars.len() == BINS {
            let w = 2.0 * s.range / BINS as f64;
            for (k, h) in bars.iter().enumerate() {
                let (x0, x1) = (-s.range + k as f64 * w, -s.range + (k + 1) as f64 * w);
                let beyond = x0.abs().min(x1.abs()) >= far - 1e-12 || (x0 + x1).abs() * 0.5 >= far;
                plot.bar(f, x0, x1, 0.0, *h as f64, 0.5, if beyond { ALARM.a(0.6) } else { ASSUMED.a(0.38) });
            }
        }
        // the t curves and the tails beyond the real difference
        let main = a.main();
        let (se, df) = (nl[2] as f64, nl[3].max(0.5) as f64);
        if main.is_some() && se > 0.0 {
            let curve = scaled_t(df, se);
            plot.area(f, far, range, 0.0, 60, ALARM.a(0.4), curve);
            plot.area(f, -range, -far, 0.0, 60, ALARM.a(0.4), curve);
            held_curve(f, &plot, plot::CURVES, 161, 2.2, ASSUMED, curve, false);
        }
        let (sp, dfp) = (nl[4] as f64, nl[5].max(0.5) as f64);
        if !a.paired && a.pooled.is_some() && sp > 0.0 && se > 0.0 && (sp / se - 1.0).abs() > 0.03 {
            held_curve(f, &plot, POOLED_CURVE, 161, 1.5, ASSUMED.a(0.8), scaled_t(dfp, sp), true);
        }
        // the real difference, and as far the other way
        if diff.is_finite() {
            plot.vline(f, OBSERVED, diff, 2.4, ESTIMATE, false);
            plot.vline(f, OBSERVED, -diff, 1.2, ESTIMATE.a(0.55), true);
            let x = plot.px_x(diff).max(min.x).min(max.x);
            let right = diff >= 0.0;
            let real = num(a.diff, 2);
            let words = [format!("the real difference, {real}"), format!("real: {real}"), real.clone()];
            let room = if right { max.x - x } else { x - min.x } + 40.0;
            let text = pick(f, &words, room.max(60.0), lay.small);
            let w = f.text_size(&text, lay.small).x;
            let fits_beside = if right { x + 6.0 + w < max.x } else { x - 6.0 - w > min.x };
            let (at, align) = match (right, fits_beside) {
                (true, true) => (Vec2::new(x + 6.0, min.y + 4.0), Align2::LEFT_TOP),
                (true, false) => (Vec2::new(x - 6.0, min.y + 4.0), Align2::RIGHT_TOP),
                (false, true) => (Vec2::new(x - 6.0, min.y + 4.0), Align2::RIGHT_TOP),
                (false, false) => (Vec2::new(x + 6.0, min.y + 4.0), Align2::LEFT_TOP),
            };
            f.text_px(at, text, lay.small, ESTIMATE, align);
            if let Some(t) = main {
                let line = [format!("{} standard errors: {}", num(t.t.abs(), 2), p_formula(t.p)), p_formula(t.p)];
                let text = pick(f, &line, room.max(60.0), lay.small);
                f.text_px(at + Vec2::new(0.0, lay.small + 4.0), text, lay.small, ALARM, align);
            }
        }
        // the newest shuffle shown, while they come
        if self.shuffling && s.pending() && s.done() > 0 && !a.paired {
            let dm = self.shown_deal();
            if dm.len() == 2 {
                let x = plot.px_x((dm[1] - dm[0]) as f64);
                if x >= min.x && x <= max.x {
                    line_of(f, SHUFFLE_MARK, Vec2::new(x, max.y), Vec2::new(x, max.y - 12.0), 2.6, ASSUMED);
                }
            }
        }
        if self.shuffling && s.pending() && s.done() > 0 && a.paired {
            let dm = self.shown_deal();
            if let Some(m) = dm.last() {
                let x = plot.px_x(*m as f64);
                if x >= min.x && x <= max.x {
                    line_of(f, SHUFFLE_MARK, Vec2::new(x, max.y), Vec2::new(x, max.y - 12.0), 2.6, ASSUMED);
                }
            }
        }
        if main.is_none() {
            let why = self.no_t_reason();
            let text = pick(f, &[why.to_string(), "no $t$ test".to_string()], max.x - min.x - 12.0, lay.small);
            f.text_px(Vec2::new(min.x + 6.0, min.y + lay.small * 3.0 + 12.0), text, lay.small, pal::DIM, Align2::LEFT_TOP);
        }
        self.null_key(f, lay, main.is_some());
        axis(f, &plot, self.ticks[1], lay.small);
        let name = if a.paired { "mean of the differences, if each unit\'s two values might be swapped" } else { r"difference of the means, $\bar x_B - \bar x_A$, if the labels are dealt at random" };
        let name = pick(f, &[name.to_string(), r"$\bar x_B - \bar x_A$".to_string()], max.x - min.x, lay.small);
        f.text_px(Vec2::new(0.5 * (min.x + max.x), max.y + lay.small + 10.0), name, lay.small, pal::DIM, Align2::CENTER_TOP);

        // under it: what the shuffles found (and, with no room for the tests' block, every p)
        let note = if !lay.blocks {
            let t = main.map_or("none".to_string(), |t| p_text(t.p));
            let sh = if self.shuffling && s.done() > 0 { format!(", shuffles {}", p_text(s.p())) } else { String::new() };
            let r = a.ranks.map_or("none".to_string(), |r| p_text(r.p));
            let long = format!("$p$ by $t$ {t}{sh}, by ranks {r}");
            pick(f, &[long, format!("$p$ by $t$ {t}{sh}")], lay.shuffle.0.x - min.x - 10.0, lay.small)
        } else if !self.shuffling {
            if lay.narrow { "or shuffle the labels and see" } else { "Or deal the labels afresh, again and again, and see." }.to_string()
        } else if !s.possible() {
            "nothing to shuffle: a group is empty".to_string()
        } else {
            let done = s.done();
            let reach = match s.reached {
                1 => format!("1 of {done} shuffles reaches it"),
                r => format!("{r} of {done} shuffles reach it"),
            };
            // (with few ways to deal the labels, the p-value has a floor worth saying)
            let ways = s.ways();
            let verb = if a.paired { "sign" } else { "deal" };
            let floor = if ways <= 20_000.0 { format!(" (only {} ways to {verb} them: $p$ cannot be below 1 in {})", num(ways, 0), num(ways, 0)) } else { String::new() };
            let long = format!("{reach}: $p = ({} + 1) / ({done} + 1) = {}${floor}", s.reached, p_text(s.p()));
            let short = format!("{reach}: {}", p_formula(s.p()));
            pick(f, &[long, short.clone(), p_formula(s.p())], lay.shuffle.0.x - min.x - 30.0, lay.small)
        };
        f.text_px(Vec2::new(min.x - if lay.narrow { 12.0 } else { 18.0 }, lay.note), note, lay.small, if self.shuffling { pal::TEXT } else { pal::DIM }, Align2::LEFT_TOP);
    }

    /// What the null chart's marks are, in its top-left corner.
    fn null_key(&self, f: &mut Frame, lay: &Layout, curve: bool) {
        let a = &self.analysis;
        let (min, _) = lay.null;
        let s = lay.small;
        let mut y = min.y + 6.0;
        let x = min.x + 8.0;
        let nl = &self.look.null.shown;
        let pooled_drawn = !a.paired && a.pooled.is_some() && nl.len() >= 6 && nl[2] > 0.0 && (nl[4] / nl[2] - 1.0).abs() > 0.03;
        let mut entries: Vec<(u8, &str)> = Vec::new();
        if curve {
            entries.push((0, if a.paired { "Student's $t$, for the mean difference" } else if lay.narrow { "Student's $t$" } else { "Student's $t$ (Welch's)" }));
        }
        if pooled_drawn {
            entries.push((1, if lay.narrow { "pooled" } else { "with a pooled spread" }));
        }
        if self.shuffling && self.shuffles.done() > 0 {
            entries.push((2, if a.paired { "random signs" } else { "shuffles" }));
        }
        for (mark, text) in entries {
            let mid = y + s * 0.5 + 1.0;
            match mark {
                0 => line_of(f, KEY, Vec2::new(x, mid), Vec2::new(x + 16.0, mid), 2.2, ASSUMED),
                1 => dashed_of(f, KEY, Vec2::new(x, mid), Vec2::new(x + 16.0, mid), 1.5, ASSUMED.a(0.8)),
                _ => plot::fill(f, (Vec2::new(x + 3.0, y + 2.0), Vec2::new(x + 13.0, y + s + 1.0)), ASSUMED.a(0.5)),
            }
            f.caption_px(Vec2::new(x + 22.0, y), text, s, pal::DIM, Align2::LEFT_TOP);
            y += s + 6.0;
        }
    }

    /// Why there is no t test.
    fn no_t_reason(&self) -> &'static str {
        let a = &self.analysis;
        if a.paired && a.sizes[0] < 2 {
            "no $t$ test: one unit, and no spread to weigh its difference against"
        } else if a.paired {
            "no $t$ test: every unit's difference is the same, no spread to weigh their mean against"
        } else if (a.sizes[0] < 2 || a.sizes[1] < 2) && a.pooled.is_some() {
            "no Welch's test: a group of one has no spread of its own (the pooled test borrows the other's)"
        } else if a.sizes[0] < 2 || a.sizes[1] < 2 {
            "no $t$ test: a group of one has no spread of its own"
        } else {
            "no $t$ test: no spread within the groups, nothing to weigh the difference against"
        }
    }

    /// The difference of the means as drawn (gliding with the points).
    fn drawn_difference(&self) -> f64 {
        let mean = |v: &[f32]| if v.is_empty() { f64::NAN } else { v.iter().map(|x| *x as f64).sum::<f64>() / v.len() as f64 };
        mean(&self.look.points[1].shown) - mean(&self.look.points[0].shown)
    }

    /// The three tests' answers, and the size of the difference.
    fn draw_tests(&self, f: &mut Frame, lay: &Layout) {
        let a = &self.analysis;
        let (min, max) = lay.tests;
        hud_panel(f, min - Vec2::splat(8.0), max + Vec2::splat(8.0));
        let width = max.x - min.x;
        let title = ["is it more than chance? the $p$-value by each test".to_string(), "the $p$-value by each test".to_string()];
        let title = pick(f, &title, width, lay.text);
        f.text_px(min, title, lay.text, pal::TEXT, Align2::LEFT_TOP);
        let s = &self.shuffles;
        let shuffled = || -> (String, String) {
            if !self.shuffling {
                ("not shuffled yet".to_string(), "?".to_string())
            } else if !s.possible() {
                ("nothing to shuffle".to_string(), "none".to_string())
            } else {
                (format!("{} of {} reach it", s.reached, s.done()), p_text(s.p()))
            }
        };
        let t_row = |t: Option<TTest>| t.map_or(("none".to_string(), "none".to_string()), |t| (format!("$t = {}$ on {} df", num(t.t, 2), num(t.df, 1)), p_text(t.p)));
        let rank_row = |r: Option<RankTest>, name: &str| {
            r.map_or(("none".to_string(), "none".to_string()), |r| {
                let how = if r.all_tied { "all tied".to_string() } else { format!("${name} = {}$", num(r.stat, if r.stat.fract() == 0.0 { 0 } else { 1 })) };
                (how, p_text(r.p))
            })
        };
        let rows: Vec<(&str, (String, String), &str)> = if a.paired {
            vec![
                ("$t$ test of the differences", t_row(a.paired_t), "assumes the differences bell-shaped, or many units"),
                ("random signs", shuffled(), "assumes only that a unit's two values could have come either way round"),
                ("signed ranks (Wilcoxon)", rank_row(a.ranks, "V"), "only the order of the differences' sizes counts, and their signs"),
                ("ignoring the pairs (Welch)", t_row(a.welch), "takes a unit's two values for unrelated: wrong for pairs"),
            ]
        } else {
            vec![
                ("Welch's $t$ test", t_row(a.welch), "assumes bell-shaped groups, or large ones; spreads may differ"),
                ("pooled $t$ test", t_row(a.pooled), "assumes the same, and that the two spreads are equal"),
                ("shuffling the labels", shuffled(), "assumes no shape: only that the labels may mean nothing at all"),
                ("ranks (Mann–Whitney)", rank_row(a.ranks, "U"), "only the order counts, not how far out a value lies"),
            ]
        };
        let step = if lay.narrow { lay.small + 5.0 } else { lay.text + 10.0 };
        let size = if lay.narrow { lay.small } else { lay.text };
        let mut y = min.y + lay.text + if lay.narrow { 6.0 } else { 12.0 };
        let detail_x = min.x + if lay.narrow { width * 0.48 } else { width * 0.42 };
        let p_x = max.x;
        let show_detail = width > 250.0;
        for (k, (name, (detail, p), assumes)) in rows.iter().enumerate() {
            let dim = a.paired && k == 3;
            let color = if dim { pal::DIM } else { pal::TEXT };
            f.text_px(Vec2::new(min.x, y), *name, size, color, Align2::LEFT_TOP);
            if show_detail {
                let room = p_x - detail_x - f.text_size("0.0000", size).x - 14.0;
                let d = pick(f, &[detail.clone(), String::new()], room, size);
                f.text_px(Vec2::new(detail_x, y), d, size, pal::DIM, Align2::LEFT_TOP);
            }
            let pv = if p.starts_with('<') || p.chars().next().is_some_and(|c| c.is_ascii_digit()) { format!("$p {}$", if p.starts_with('<') { p.clone() } else { format!("= {p}") }) } else { p.clone() };
            f.text_px(Vec2::new(p_x, y), pv, size, if dim { pal::DIM } else { ALARM.mix(pal::WHITE, 0.25) }, Align2::RIGHT_TOP);
            if lay.assumptions {
                f.text_px(Vec2::new(min.x + 12.0, y + size + 3.0), *assumes, lay.small, pal::DIM.a(0.8), Align2::LEFT_TOP);
                y += lay.small + 4.0;
            }
            y += step;
        }
        // how large the difference is
        y += if lay.narrow { 2.0 } else { 8.0 };
        // (with no spread to measure it in, there is no d: said in words, not as a formula)
        let (d, d_long) = match a.d {
            Some(d) => (format!("$d = {}$", num(d, 2)), format!("Cohen's $d = {}$, the means {} pooled standard deviations apart", num(d, 2), num(d, 2))),
            None => ("no $d$".to_string(), "Cohen's $d$: none, with no spread within the groups to measure in".to_string()),
        };
        let sup = a.superiority.map_or("none".to_string(), percent);
        if lay.narrow {
            let text = format!("how far apart: {d}; $B$'s the larger in {sup} of pairs");
            f.text_px(Vec2::new(min.x, y), text, lay.small, ESTIMATE, Align2::LEFT_TOP);
            return;
        }
        let lines = [format!("how far apart: {d_long}"), format!("Cohen's {d}")];
        let text = pick(f, &lines, width, lay.small);
        f.text_px(Vec2::new(min.x, y), text, lay.small, ESTIMATE, Align2::LEFT_TOP);
        y += lay.small + 6.0;
        let lines = [format!("of all pairs (one point from each group), $B$'s is the larger in {sup}"), format!("$B$'s the larger in {sup} of pairs")];
        let text = pick(f, &lines, width, lay.small);
        f.text_px(Vec2::new(min.x, y), text, lay.small, ESTIMATE, Align2::LEFT_TOP);
    }

    /// The count over new studies.
    fn draw_studies(&self, f: &mut Frame, lay: &Layout) {
        let (min, max) = lay.studies;
        // (before counting, only as tall as its title and button)
        let bottom = if self.counting { max.y } else { lay.count.1.y.min(max.y) };
        hud_panel(f, min - Vec2::splat(8.0), Vec2::new(max.x, bottom) + Vec2::splat(8.0));
        let width = max.x - min.x;
        let truth = &self.truth;
        let titles = [
            format!("{STUDIES} new studies like this one: how often each test says $p \\le 0.05$"),
            format!("{STUDIES} new studies: $p \\le 0.05$ how often?"),
        ];
        let title = pick(f, &titles, width, lay.text);
        f.text_px(min, title, lay.text, pal::TEXT, Align2::LEFT_TOP);
        if !self.counting {
            let words = ["New samples from the same groups, each tested as this one is: the share each test rejects is its power, or its false alarm rate.".to_string(), "new samples from the same groups".to_string()];
            let text = pick(f, &words, width - (lay.count.1.x - lay.count.0.x) - 16.0, lay.small);
            f.text_px(Vec2::new(lay.count.1.x + 12.0, lay.count.0.y + 4.0), text, lay.small, pal::DIM, Align2::LEFT_TOP);
            return;
        }
        let true_diff = truth.mean[1] - truth.mean[0];
        let null = true_diff == 0.0;
        let status = if null {
            "the true means are equal: every rejection is a false alarm".to_string()
        } else {
            format!("the true difference is {}: every rejection is right", num(true_diff, 2))
        };
        let status = if self.studies.done < STUDIES { format!("{status} ({} so far)", self.studies.done) } else { status };
        let status = pick(f, &[status.clone(), if null { "false alarms".to_string() } else { "right rejections".to_string() }], width, lay.small);
        let mut y = min.y + lay.text + 8.0;
        f.text_px(Vec2::new(min.x, y), status, lay.small, if null { ALARM } else { TRUTH }, Align2::LEFT_TOP);
        y += lay.small + 8.0;
        let names: [&str; 4] = if truth.paired { ["paired $t$", "ignoring pairs", "random signs", "signed ranks"] } else { ["Welch's $t$", "pooled $t$", "shuffles", "ranks"] };
        let label_w = names.iter().map(|n| f.text_size(n, lay.small).x).fold(0.0, f32::max) + 10.0;
        let value_w = f.text_size("100 %", lay.small).x + 8.0;
        let (x0, x1) = (min.x + label_w, max.x - value_w);
        let bar_h = ((max.y - y) / 4.0 - 5.0).max(6.0).min(18.0);
        let shown = &self.look.studies.shown;
        let color = if null { ALARM } else { TRUTH };
        for k in 0..4 {
            f.text_px(Vec2::new(min.x, y + bar_h * 0.5), names[k], lay.small, pal::DIM, Align2::LEFT_CENTER);
            hud_rect(f, Vec2::new(x0, y), Vec2::new(x1, y + bar_h), Color::hex(0x161C2B));
            let share = shown.get(k).copied().unwrap_or(0.0).max(0.0).min(1.0);
            if share * (x1 - x0) > 0.5 {
                hud_rect_at(f, 2, Vec2::new(x0, y), Vec2::new(x0 + share * (x1 - x0), y + bar_h), color.a(0.8));
            }
            let value = self.studies.share(k).map_or("none".to_string(), percent);
            f.text_px(Vec2::new(max.x, y + bar_h * 0.5), value, lay.small, pal::TEXT, Align2::RIGHT_CENTER);
            y += bar_h + 5.0;
        }
        // the 5 % a test promises when nothing is going on
        let x = x0 + 0.05 * (x1 - x0);
        let top = min.y + lay.text + 8.0 + lay.small + 6.0;
        dashed_of(f, LEVEL_LINE, Vec2::new(x, top), Vec2::new(x, y - 3.0), 1.2, pal::WHITE.a(0.7));
        if y + lay.small < max.y + 6.0 {
            f.text_px(Vec2::new(x, y - 2.0), "5 %", lay.small, pal::DIM, Align2::CENTER_TOP);
        }
    }

    /// On a canvas too small for the charts: the p-values alone.
    fn draw_compact(&self, f: &mut Frame, lay: &Layout) {
        let a = &self.analysis;
        let mut y = (f.top_inset + 10.0).max(10.0);
        let x = 10.0;
        let mut line = |f: &mut Frame, text: String, color: Color| {
            f.text_px(Vec2::new(x, y), text, lay.small, color, Align2::LEFT_TOP);
            y += lay.small + 6.0;
        };
        line(f, format!(r"$\bar x_B - \bar x_A = {}$", num(a.diff, 2)), ESTIMATE);
        if let Some(t) = a.main() {
            line(f, format!("$t$ test: {}", p_formula(t.p)), pal::TEXT);
        }
        if self.shuffling && self.shuffles.done() > 0 {
            line(f, format!("shuffles: {}", p_formula(self.shuffles.p())), pal::TEXT);
        }
        if let Some(r) = a.ranks {
            line(f, format!("ranks: {}", p_formula(r.p)), pal::TEXT);
        }
        line(f, "(a larger window shows the charts)".to_string(), pal::DIM);
    }
}

/// Every batch of lines this scene draws, kept in being and in the same order from frame
/// to frame: a batch not drawn in a frame is let go, and then those after it change
/// places in the list, which `eidos motion` would take for lines that moved. (Each gets
/// an invisible segment of no length at the corner.)
fn keep_batches(f: &mut Frame) {
    let corner = f.hud(Vec2::ZERO);
    for kind in [plot::AXES, plot::CURVES, plot::MARKS, plot::BUTTONS, PAIRS, UPWARDS, DEALT, BRACKETS, OBSERVED, LEVEL_LINE, SHUFFLE_MARK, POOLED_CURVE, KEY] {
        for dashed in [false, true] {
            plot::apart(f, kind, dashed).segment(corner, corner, 1.0, Color::CLEAR);
        }
    }
}

/// The curve y = g(x) across a chart, sampled at `n` points, with the parts above or
/// below the chart held at its edge rather than cut off: the same points in every
/// frame. (A curve cut where it leaves the chart has more points or fewer as its height
/// and the chart's glide past each other, and `eidos motion` would take each point for
/// its neighbour moved.)
#[allow(clippy::too_many_arguments)]
fn held_curve(f: &mut Frame, plot: &Plot, kind: u8, n: usize, width: f32, color: Color, g: impl Fn(f64) -> f64, dashed: bool) {
    if plot.tiny() {
        return;
    }
    let (min, max) = plot.rect;
    let [a, b] = plot.x;
    let pts: Vec<Vec3> = (0..n.max(2))
        .map(|i| a + (b - a) * i as f64 / (n.max(2) - 1) as f64)
        .map(|x| {
            let y = g(x);
            let y = if y.is_finite() { plot.px_y(y) } else { max.y };
            f.hud(Vec2::new(plot.px_x(x), y.max(min.y).min(max.y)))
        })
        .collect();
    plot::apart(f, kind, dashed).polyline(pts, width, color);
}

/// The density of `scale` times Student's t with `df` degrees of freedom: the null
/// distribution of a difference whose standard error is `scale`. (The same as
/// `Dist::StudentT`'s, with its constant worked out once for the whole curve.)
fn scaled_t(df: f64, scale: f64) -> impl Fn(f64) -> f64 + Copy {
    let c = (st::ln_gamma((df + 1.0) / 2.0) - st::ln_gamma(df / 2.0) - 0.5 * (df * std::f64::consts::PI).ln()).exp() / scale;
    move |x: f64| {
        let z = x / scale;
        c * (1.0 + z * z / df).powf(-(df + 1.0) / 2.0)
    }
}

/// The size of a group's dots.
fn dot_size(n: usize, narrow: bool) -> f32 {
    let base = match n {
        0..=30 => 8.0,
        31..=60 => 7.0,
        61..=200 => 5.0,
        _ => 3.5,
    };
    if narrow {
        base - 1.0
    } else {
        base
    }
}

/// An axis along the bottom of a chart: every round value of `ticks`, those off the
/// chart unseen at its edge (so that each keeps its place in its batch while the axis
/// glides).
fn axis(f: &mut Frame, plot: &Plot, ticks: Ticks, size: f32) {
    let (min, max) = plot.rect;
    line_of(f, plot::AXES, Vec2::new(min.x, max.y), max, 1.0, pal::DIM.a(0.8));
    for v in ticks.values() {
        let x = plot.px_x(v);
        let seen = x >= min.x - 0.5 && x <= max.x + 0.5;
        let x = x.max(min.x).min(max.x);
        line_of(f, plot::AXES, Vec2::new(x, max.y), Vec2::new(x, max.y + 4.0), 1.0, pal::DIM.a(if seen { 0.8 } else { 0.0 }));
        let text = if seen { ticks.name(v) } else { String::new() };
        f.text_px(Vec2::new(x, max.y + 6.0), text, size, pal::DIM, Align2::CENTER_TOP);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn close(a: f64, b: f64, tol: f64) -> bool {
        (a - b).abs() <= tol * (1.0 + b.abs())
    }

    /// The tests against values worked out by hand and by R.
    #[test]
    fn the_tests_give_the_textbook_answers() {
        // t.test(c(1,2,3,4,5), c(2,4,6,8,10)): Welch t = -1.8974, df = 5.882, p = 0.1075
        let (a, b) = ([1.0, 2.0, 3.0, 4.0, 5.0], [2.0, 4.0, 6.0, 8.0, 10.0]);
        let w = welch(&a, &b).expect("a test");
        assert!(close(w.t, 1.897_366_6, 1e-6) && close(w.df, 5.882_352_9, 1e-6) && close(w.p, 0.107_5, 2e-3), "{w:?}");
        // var.equal = TRUE: t = 1.8974 on 8 df, p = 0.09434
        let p = pooled(&a, &b).expect("a test");
        assert!(close(p.t, 1.897_366_6, 1e-6) && p.df == 8.0 && close(p.p, 0.094_34, 2e-3), "{p:?}");
        // paired: differences 1..5, mean 3, sd 1.5811, t = 4.2426 on 4 df, p = 0.01324
        let q = paired_t(&a, &b).expect("a test");
        assert!(close(q.t, 4.242_640_7, 1e-6) && q.df == 4.0 && close(q.p, 0.013_24, 2e-3), "{q:?}");
        // wilcox.test(c(1.1,2.2,3.3), c(4.4,5.5,6.6,7.7)): W (A's U) = 0, exact p = 0.05714 (= 2/35)
        let r = mann_whitney(&[1.1, 2.2, 3.3], &[4.4, 5.5, 6.6, 7.7]).expect("a test");
        assert!(r.exact && r.stat == 12.0 && close(r.p, 2.0 / 35.0, 1e-12), "{r:?}");
        // signed ranks of 1.5, −0.5, 2.5, 3.5, 4.5: V = 14, exact p = 2 × 2/32 = 0.125
        let s = signed_rank(&[1.5, -0.5, 2.5, 3.5, 4.5]).expect("a test");
        assert!(s.exact && s.stat == 14.0 && close(s.p, 0.125, 1e-12), "{s:?}");
    }

    /// One line of what a sample shows (for the explorations below).
    fn line(name: &str, data: &[Vec<f64>; 2], paired: bool, seed: u64) {
        let a = analyse(data, paired);
        let sp = shuffled_p(data, paired, SHUFFLES, seed).unwrap_or(f64::NAN);
        let main = a.main();
        println!(
            "{name}: diff {:.3} sds {:.3} {:.3} t {:.3} df {:.2} p {:.5}; welch p {:.5}; pooled p {:.5}; shuffles {:.5}; ranks {:.5} ({:?}); d {:.3}; sup {:.3}; corr {:?}",
            a.diff,
            a.sds[0],
            a.sds[1],
            main.map_or(f64::NAN, |t| t.t),
            main.map_or(f64::NAN, |t| t.df),
            main.map_or(f64::NAN, |t| t.p),
            a.welch.map_or(f64::NAN, |t| t.p),
            a.pooled.map_or(f64::NAN, |t| t.p),
            sp,
            a.ranks.map_or(f64::NAN, |r| r.p),
            a.ranks.map(|r| r.stat),
            a.d.unwrap_or(f64::NAN),
            a.superiority.unwrap_or(f64::NAN),
            a.corr
        );
    }

    /// The rejections over new studies of a truth (for the explorations below).
    fn count(truth: &Truth) -> String {
        let mut s = Studies::new();
        s.restart(*truth);
        s.finish();
        format!("{:?} of {:?}", s.rejected, s.tested)
    }

    /// An experiment as its button sets it up, with its shuffles and count run to the end.
    fn run(k: usize) -> TwoGroups {
        let mut s = TwoGroups::create();
        s.set_up(k);
        s.settle();
        s
    }

    /// The p-value by Welch's test, and by the shuffles, as the picture writes them.
    fn ps(s: &TwoGroups) -> (String, String) {
        (s.analysis.welch.map_or(String::new(), |t| p_text(t.p)), p_text(s.shuffles.p()))
    }

    /// The things to try, held to their captions.
    #[test]
    fn the_experiments_show_what_their_captions_say() {
        let all: Vec<TwoGroups> = (0..TRIALS.len()).map(run).collect();
        for (k, s) in all.iter().enumerate() {
            assert_eq!((s.truth, s.seed), (TRIALS[k].truth, TRIALS[k].seed), "{}", TRIALS[k].button);
            assert!(s.shuffling && s.shuffles.done() == SHUFFLES, "{}", TRIALS[k].button);
        }

        // a clear difference: groups of 20 a standard deviation apart; the difference
        // 1.04 is 3.1 standard errors, t = 3.11, p = 0.0036; 6 of 2000 shuffles reach
        // it, p = 0.0035; d = 0.98, B's the larger in 75.5 % of the pairs
        let clear = &all[0];
        let (a, w) = (&clear.analysis, clear.analysis.welch.expect("a t test"));
        assert!(clear.truth.n == [20, 20] && clear.truth.sd == [1.0, 1.0] && clear.truth.mean[1] - clear.truth.mean[0] == 1.0 && clear.truth.shape == Shape::Bell);
        assert_eq!((num(a.diff, 2), format!("{:.1}", w.t), num(w.t, 2)), ("1.04".into(), "3.1".into(), "3.11".into()));
        assert_eq!(ps(clear), ("0.0036".into(), "0.0035".into()));
        assert_eq!(clear.shuffles.reached, 6);
        assert_eq!((a.d.map(|d| num(d, 2)), a.superiority.map(percent)), (Some("0.98".into()), Some("75.5 %".into())));

        // more noise: three times the spread; the difference much the same (1.13), its
        // standard error tripled, t = 1.13, p = 0.27; shuffles p = 0.26
        let noisy = &all[1];
        let w1 = noisy.analysis.welch.expect("a t test");
        assert!(noisy.truth.sd == [3.0, 3.0] && noisy.truth.mean == clear.truth.mean);
        assert_eq!((num(noisy.analysis.diff, 2), num(w1.t, 2)), ("1.13".into(), "1.13".into()));
        assert!((w1.se / w.se - 3.0).abs() < 0.05, "standard errors {} and {}", w1.se, w.se);
        assert_eq!(ps(noisy), ("0.27".into(), "0.26".into()));

        // one wild value: B's largest moved out to 20; the means 1.04 to 1.90 apart, B's
        // spread 1.07 to 4.40, t = 1.88, p 0.0036 to 0.075; the ranks and their p (0.0051)
        // unchanged; the shuffles in two humps, p still 0.0035
        let wild = &all[2];
        let ww = wild.analysis.welch.expect("a t test");
        assert_eq!(wild.data[0], clear.data[0]);
        let changed: Vec<usize> = (0..20).filter(|&i| wild.data[1][i] != clear.data[1][i]).collect();
        assert_eq!(changed.len(), 1);
        let top = changed[0];
        assert!(wild.data[1][top] == 20.0 && clear.data[1].iter().all(|v| *v <= clear.data[1][top]), "the point moved was B's largest");
        assert_eq!((num(wild.analysis.diff, 2), num(clear.analysis.sds[1], 2), num(wild.analysis.sds[1], 2)), ("1.90".into(), "1.07".into(), "4.40".into()));
        assert_eq!((num(ww.t, 2), p_text(ww.p)), ("1.88".into(), "0.075".into()));
        assert_eq!(wild.analysis.ranks, clear.analysis.ranks);
        assert_eq!(wild.analysis.ranks.map(|r| p_text(r.p)), Some("0.0051".into()));
        assert_eq!(ps(wild).1, "0.0035");
        let made = &wild.shuffles.made;
        let near = made.iter().filter(|v| v.abs() < 0.3).count();
        let humps = made.iter().filter(|v| v.abs() > 0.6 && v.abs() < 1.3).count();
        assert!(near * 20 < humps && humps > made.len() / 2, "two humps: {near} near 0, {humps} in the humps");

        // pairs: correlation 0.90 here, B 0.5 higher; as two groups p = 0.14, as pairs
        // p = 0.00017; over 1000 new studies, 99.6 % against 18.4 %
        let pairs = &all[3];
        let pa = &pairs.analysis;
        assert!(pairs.truth.paired && pa.paired && pairs.truth.rho == 0.9 && pairs.truth.mean[1] - pairs.truth.mean[0] == 0.5 && pairs.truth.n[0] == 20);
        assert_eq!(pa.corr.map(|r| num(r, 2)), Some("0.90".into()));
        assert_eq!((pa.welch.map(|t| p_text(t.p)), pa.paired_t.map(|t| p_text(t.p))), (Some("0.14".into()), Some("0.00017".into())));
        assert_eq!(pairs.studies.done, STUDIES);
        assert_eq!((pairs.studies.share(0).map(percent), pairs.studies.share(1).map(percent)), (Some("99.6 %".into()), Some("18.4 %".into())));

        // unequal spreads and sizes: equal means, 10 of spread 3 against 40 of spread 1;
        // the pooled test 26.2 % (five times 5 %; 0.0011 on the sample shown), shuffles
        // 25.6 %, ranks 16.5 %, Welch's 5.7 %
        let unequal = &all[4];
        let t = unequal.truth;
        assert!(t.n == [10, 40] && t.sd == [3.0, 1.0] && t.mean[0] == t.mean[1] && !t.paired);
        let shares: Vec<String> = (0..4).map(|k| unequal.studies.share(k).map_or(String::new(), percent)).collect();
        assert_eq!(shares, ["5.7 %", "26.2 %", "25.6 %", "16.5 %"]);
        let times = unequal.studies.share(1).unwrap_or(0.0) / LEVEL;
        assert!(times > 4.5 && times < 5.5, "{times} times the level");
        assert_eq!(unequal.analysis.pooled.map(|t| p_text(t.p)), Some("0.0011".into()));
    }

    /// The share of shuffles counts the real labelling among them, so it is never 0, and
    /// it can never be below one in the number of ways to deal the labels.
    #[test]
    fn the_shuffles_count_the_real_labelling_too() {
        // a group of one against three: four ways to deal, and only the real one is as
        // far from 0 (−9, against 4.33, 3 and 1.67)
        let data = [vec![10.0], vec![0.0, 1.0, 2.0]];
        let mut s = Shuffles::new(&data, false, 1);
        assert_eq!(s.p(), 1.0);
        assert!((s.ways() - 4.0).abs() < 1e-9);
        for _ in 0..SHUFFLES {
            s.step();
        }
        assert!((s.p() - 0.25).abs() < 0.03 && s.p() > 0.0, "p = {}", s.p());
        // no t test with a group of one, but the pooled one borrows the other's spread
        assert!(welch(&data[0], &data[1]).is_none());
        assert_eq!(pooled(&data[0], &data[1]).map(|t| t.df), Some(2.0));
        // a clear difference: 252 ways to deal five and five, and only the real one and
        // its mirror image (A and B swapped) as far out: p near 2 in 252, never below
        let far = [vec![0.0, 0.1, 0.2, 0.3, 0.4], vec![10.0, 10.1, 10.2, 10.3, 10.4]];
        let p = shuffled_p(&far, false, 1999, 3).unwrap_or(0.0);
        assert!((p - 2.0 / 252.0).abs() < 0.004, "{p}");
        // far apart and many points: the real deal alone among the shuffles, p = 1 in (shuffles + 1)
        let far = [vec![0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7], vec![10.0, 10.1, 10.2, 10.3, 10.4, 10.5, 10.6, 10.7]];
        assert_eq!(shuffled_p(&far, false, 999, 3), Some(1.0 / 1000.0));
        // pairs: two units, four ways to sign them
        let mut s = Shuffles::new(&[vec![0.0, 0.0], vec![1.0, 2.0]], true, 2);
        assert!((s.ways() - 4.0).abs() < 1e-9);
        for _ in 0..SHUFFLES {
            s.step();
        }
        // (the real signs and all signs turned reach a mean as far from 0: half the deals)
        assert!((s.p() - 0.5).abs() < 0.03, "p = {}", s.p());
        // a deal puts as many points in each group as there were
        let mut s = Shuffles::new(&[vec![1.0; 3], vec![2.0; 7]], false, 4);
        s.step();
        let deal = s.last_deal();
        assert_eq!((deal.iter().filter(|b| !**b).count(), deal.iter().filter(|b| **b).count()), (3, 7));
        let mut s = Shuffles::new(&[vec![1.0; 7], vec![2.0; 3]], false, 4);
        s.step();
        assert_eq!(s.last_deal().iter().filter(|b| **b).count(), 3);
    }

    /// For bell-shaped groups the shuffles give the t test's answer, but for the noise of
    /// counting 2000 of them.
    #[test]
    fn t_and_shuffles_agree_for_bells() {
        let (mut worst, mut total) = (0.0f64, 0.0f64);
        for seed in 1..=40u64 {
            let data = sample(&Truth { mean: [0.0, 0.5], ..CLEAR }, seed);
            let t = welch(&data[0], &data[1]).map_or(f64::NAN, |t| t.p);
            let s = shuffled_p(&data, false, SHUFFLES, seed).unwrap_or(f64::NAN);
            worst = worst.max((t - s).abs());
            total += (t - s).abs();
        }
        assert!(worst < 0.025 && total / 40.0 < 0.0055, "worst {worst}, mean {}", total / 40.0);
    }

    /// For bell-shaped groups of the same spread, the share of pairs in which B is the
    /// larger is Φ(d/√2).
    #[test]
    fn d_and_the_share_of_pairs_agree_for_bells() {
        for (shift, seed) in [(0.0, 1u64), (0.5, 2), (1.0, 3), (1.5, 4)] {
            let a = analyse(&sample(&Truth { n: [1000, 1000], mean: [0.0, shift], ..CLEAR }, seed), false);
            let (d, sup) = (a.d.unwrap_or(f64::NAN), a.superiority.unwrap_or(f64::NAN));
            assert!((sup - st::normal_cdf(d / 2f64.sqrt())).abs() < 0.006, "shift {shift}: d {d}, share {sup}");
            assert!((d - shift).abs() < 0.12, "shift {shift}: d {d}");
        }
        // and as the panel says: 76 % at d = 1
        assert_eq!(percent(st::normal_cdf(1.0 / 2f64.sqrt())), "76.0 %");
    }

    /// The rank tests counted out exactly agree with their normal approximations where
    /// both apply; ties are allowed for, and all ties leave nothing to say.
    #[test]
    fn the_rank_tests_are_right_with_and_without_ties() {
        let data = sample(&Truth { n: [25, 25], mean: [0.0, 0.6], ..CLEAR }, 9);
        let (a, b) = (&data[0], &data[1]);
        let exact = mann_whitney(a, b).expect("a test");
        assert!(exact.exact);
        let all: Vec<f64> = a.iter().chain(b).copied().collect();
        let ranks = st::ranks(&all);
        let u = ranks[25..].iter().sum::<f64>() - 25.0 * 26.0 / 2.0;
        assert_eq!(u, exact.stat);
        let approx = normal_two_sided(u, 312.5, 25.0 * 25.0 * 51.0 / 12.0);
        assert!((approx - exact.p).abs() < 0.01, "{approx} against {}", exact.p);
        // U counts the pairs in which B is the larger
        let pairs = a.iter().map(|x| b.iter().filter(|y| *y > x).count()).sum::<usize>();
        assert_eq!(pairs as f64, u);
        // ties: the approximation, corrected, and a finite p
        let tied = mann_whitney(&[1.0, 2.0, 2.0, 3.0], &[2.0, 3.0, 3.0, 4.0]).expect("a test");
        assert!(!tied.exact && tied.p > 0.0 && tied.p <= 1.0 && tied.stat == 13.0, "{tied:?}");
        let flat = mann_whitney(&[1.0; 5], &[1.0; 4]).expect("a test");
        assert!(flat.all_tied && flat.p == 1.0);
        // the signed ranks: zeros left out, then the approximation
        let s = signed_rank(&[0.0, 1.0, 2.0, 3.0, -0.5]).expect("a test");
        assert!(!s.exact && s.stat == 9.0 && s.p > 0.0 && s.p < 1.0, "{s:?}");
        assert!(signed_rank(&[0.0, 0.0]).is_some_and(|r| r.all_tied && r.p == 1.0));
        let big: Vec<f64> = (1..=30).map(|i| i as f64 * if i % 3 == 0 { -1.0 } else { 1.0 }).collect();
        let e = signed_rank(&big).expect("a test");
        let n = 30.0;
        let approx = normal_two_sided(e.stat, n * (n + 1.0) / 4.0, n * (n + 1.0) * (2.0 * n + 1.0) / 24.0);
        assert!(e.exact && (approx - e.p).abs() < 0.01, "{approx} against {}", e.p);
    }

    /// A group of one, no spread, two identical groups, all values equal, one unit:
    /// every number stays finite and the picture says what has happened.
    #[test]
    fn degenerate_data_are_told_honestly() {
        let lays = [Layout::of(Vec2::new(1075.0, 1000.0), 0.0), Layout::of(Vec2::new(390.0, 756.0), 46.0), Layout::of(Vec2::new(240.0, 180.0), 0.0)];
        for paired in [false, true] {
            for n in [[1, 1], [1, 5], [2, 2], [5, 1], [300, 40]] {
                for sd in [[0.0, 0.0], [0.0, 1.0], [1.0, 1.0]] {
                    for shape in Shape::ALL {
                        let mut s = TwoGroups::create();
                        s.truth = Truth { paired, rho: 0.5, n, mean: [0.0, 0.5], sd, shape };
                        s.shuffling = true;
                        s.counting = true;
                        s.refresh();
                        for _ in 0..50 {
                            s.shuffles.step();
                            s.studies.step();
                        }
                        let a = &s.analysis;
                        for t in [a.welch, a.pooled, a.paired_t].into_iter().flatten() {
                            assert!(t.se > 0.0 && t.t.is_finite() && t.df > 0.0 && t.p >= 0.0 && t.p <= 1.0, "{paired} {n:?} {sd:?}: {t:?}");
                        }
                        assert!(s.shuffles.p() > 0.0 && s.shuffles.p() <= 1.0);
                        if let Some(r) = a.ranks {
                            assert!(r.p > 0.0 && r.p <= 1.0 && r.stat.is_finite(), "{r:?}");
                        }
                        if sd == [0.0, 0.0] {
                            assert!(a.main().is_none() && a.d.is_none(), "{paired} {n:?}: no spread, no t");
                        }
                        for lay in &lays {
                            let g = s.goals();
                            s.advance_deal(0.1);
                            let shown = s.shown_deal();
                            let all = g.points.iter().flatten().chain(&g.axis).chain(&g.bells).chain(&g.null).chain(&g.bars).chain(&shown).chain(&g.studies);
                            assert!(all.clone().all(|v| v.is_finite()), "{paired} {n:?} {sd:?} {shape:?}");
                            assert!(g.null[0] > 0.0 && g.null[1] > 0.0 && g.axis[1] > g.axis[0]);
                            let _ = lay;
                        }
                    }
                }
            }
        }
        // two identical groups: no difference, t = 0, p = 1
        let same = [vec![1.0, 2.0, 4.0], vec![1.0, 2.0, 4.0]];
        let a = analyse(&same, false);
        assert!(a.diff == 0.0 && a.welch.is_some_and(|t| t.t == 0.0 && (t.p - 1.0).abs() < 1e-12));
        assert!(a.ranks.is_some_and(|r| (r.p - 1.0).abs() < 1e-12) && a.superiority == Some(0.5));
        // every value equal: every shuffle as far out as the real difference (none): p = 1
        let flat = [vec![3.0; 6], vec![3.0; 6]];
        let p = shuffled_p(&flat, false, 200, 1).unwrap_or(0.0);
        assert_eq!(p, 1.0);
        let a = analyse(&flat, false);
        assert!(a.welch.is_none() && a.pooled.is_none() && a.ranks.is_some_and(|r| r.all_tied));
        // no spread within, a difference between: no t, the shuffles and the ranks still speak
        let apart = [vec![0.0; 10], vec![1.0; 10]];
        let a = analyse(&apart, false);
        assert!(a.welch.is_none() && a.ranks.is_some_and(|r| r.p < 0.001));
        assert!(shuffled_p(&apart, false, 999, 1).is_some_and(|p| p <= 2.0 / 1000.0));
    }

    /// The layout of every window: finite, inside the canvas, nothing on top of anything
    /// else.
    #[test]
    fn the_layout_fits_every_window() {
        for (w, h, inset) in [(1075.0, 1000.0, 0.0), (1600.0, 1000.0, 0.0), (1400.0, 760.0, 0.0), (846.0, 768.0, 0.0), (900.0, 600.0, 0.0), (390.0, 756.0, 46.0), (375.0, 667.0, 46.0), (390.0, 1500.0, 46.0), (800.0, 390.0, 46.0), (700.0, 500.0, 0.0), (580.0, 700.0, 0.0), (240.0, 180.0, 0.0), (1.0, 1.0, 0.0), (3000.0, 2000.0, 0.0)] {
            let lay = Layout::of(Vec2::new(w, h), inset);
            let mut rects = vec![lay.data, lay.bells, lay.rows[0], lay.rows[1], lay.rows[2], lay.null, lay.shuffle, lay.design[0], lay.design[1], lay.again];
            let blocks = [lay.tests, lay.studies, lay.count];
            assert!(rects.iter().chain(&blocks).all(|(a, b)| a.is_finite() && b.is_finite()), "{w}x{h}");
            if lay.compact {
                continue;
            }
            if lay.blocks {
                rects.extend(blocks);
            }
            assert!(lay.fits() && w >= 300.0, "{w}x{h}");
            assert!(rects.iter().all(|(a, b)| a.x >= 0.0 && a.y >= inset && b.x <= w + 0.5 && b.y <= h + 0.5 && b.x >= a.x && b.y >= a.y), "{w}x{h}: {rects:?}");
            let overlap = |p: Rect, q: Rect| p.0.x < q.1.x && q.0.x < p.1.x && p.0.y < q.1.y && q.0.y < p.1.y;
            // the blocks one below another or side by side, and the buttons clear of the charts
            let mut pairs = vec![(lay.data, lay.null), (lay.design[1], lay.again), (lay.again, lay.data), (lay.shuffle, lay.null)];
            if lay.blocks {
                pairs.extend([(lay.data, lay.tests), (lay.null, lay.tests), (lay.tests, lay.studies), (lay.null, lay.studies), (lay.shuffle, lay.tests)]);
                assert!(lay.studies.1.y - lay.studies.0.y >= 70.0, "{w}x{h}: room for the new studies");
            }
            for (p, q) in pairs {
                assert!(!overlap(p, q), "{w}x{h}: {p:?} and {q:?}");
            }
            assert!(lay.null.1.y - lay.null.0.y >= 60.0 && lay.data.1.y - lay.data.0.y >= 110.0, "{w}x{h}");
            // the usual windows have everything
            if (w, h) == (1075.0, 1000.0) || (w, h) == (390.0, 756.0) || (w, h) == (1400.0, 760.0) {
                assert!(lay.blocks, "{w}x{h}");
            }
        }
    }

    /// After a change the picture glides to the new one and comes exactly to rest, at any
    /// frame rate.
    #[test]
    fn glides_arrive_and_stop() {
        for hz in [30.0f32, 60.0, 144.0] {
            let mut s = TwoGroups::create();
            s.settle();
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
                    assert!(s.look.points[0].shown == g.points[0] && s.look.null.shown == g.null);
                    break;
                }
                assert!((frames as f32) < 2.0 * hz, "still gliding after two seconds at {hz} Hz");
            }
            assert!(frames as f32 > 0.2 * hz, "arrived in {frames} frames at {hz} Hz: no glide");
        }
    }

    /// While the shuffles come, the marks of the deal shown move in every frame: a whole
    /// step each, at any frame rate, also from one deal to the next (no frame in which
    /// they stand still between two in which they move).
    #[test]
    fn the_deal_shown_moves_in_every_frame() {
        for paired in [false, true] {
            for hz in [60.0f32, 75.0, 120.0, 144.0] {
                let mut s = TwoGroups::create();
                s.set_up(if paired { 3 } else { 0 });
                let mut moves = Vec::new();
                let mut last = s.shown_deal();
                for _ in 0..(2.5 * hz) as usize {
                    s.work(1.0 / hz, 1.0 / hz);
                    let now = s.shown_deal();
                    if now.len() == last.len() {
                        moves.push(now.iter().zip(&last).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max));
                    }
                    last = now;
                }
                assert!(s.shuffles.pending(), "still shuffling");
                // (after the first deal, which stands for one leg while the next is made)
                let first = (DEAL_SHOWN * hz) as usize + 2;
                for w in moves[first..].windows(3) {
                    let around = w[0].min(w[2]);
                    assert!(around <= 1e-6 || w[1] > 0.05 * around, "{paired} at {hz} Hz: {w:?}");
                }
                assert!(moves[first..].iter().filter(|m| **m > 0.0).count() > moves.len() / 2, "{paired} at {hz} Hz: the marks hardly moved");
            }
        }
    }

    /// The curve drawn is Student's t, scaled.
    #[test]
    fn the_curve_is_students_t() {
        for (df, scale) in [(1.0, 0.5), (4.5, 1.0), (37.6, 0.33), (1000.0, 2.0)] {
            let curve = scaled_t(df, scale);
            for x in [-3.0, -0.4, 0.0, 0.7, 5.0] {
                let want = Dist::StudentT { df }.pdf(x / scale) / scale;
                assert!((curve(x) - want).abs() <= 1e-10 * (1.0 + want), "{df} {scale} {x}");
            }
        }
    }

    /// A change to the data starts the shuffles afresh; a change to what they are drawn
    /// from draws them again and starts the count afresh; a dragged point leaves the count
    /// alone (it counts new studies from the truth, not this sample).
    #[test]
    fn a_change_starts_the_work_afresh() {
        let mut s = TwoGroups::create();
        s.counting = true;
        s.settle();
        assert!(s.shuffles.done() == SHUFFLES && s.studies.done == STUDIES);
        // a point dragged
        s.data[0][3] += 0.5;
        s.version += 1;
        s.refresh();
        assert!(s.shuffles.done() == 0 && s.shuffles.version == s.version && s.studies.done == STUDIES);
        assert!((s.analysis.means[0] - st::mean(&s.data[0])).abs() < 1e-12);
        // a setting changed: the data drawn again (the dragged point forgotten), the count anew
        let before = s.data.clone();
        s.truth.sd[1] = 2.0;
        s.refresh();
        assert!(s.data[0] != before[0] && s.studies.done == 0 && s.studies.truth == Some(s.truth));
        // "shuffle again": the same data, another round of shuffles
        s.settle();
        let (reached, round) = (s.shuffles.reached, s.round);
        s.round += 1;
        s.refresh();
        assert!(s.shuffles.done() == 0 && s.shuffles.round == round + 1);
        s.settle();
        assert!(s.shuffles.reached.abs_diff(reached) < 60, "{} and {reached}", s.shuffles.reached);
        // shuffling stopped: the histogram goes
        s.shuffling = false;
        s.refresh();
        assert!(s.shuffles.done() == 0 && s.deal.is_none());
    }

    /// The numbers behind the captions, as each experiment shows them (run with
    /// --ignored --nocapture).
    #[test]
    #[ignore]
    fn explore_trials() {
        for k in 0..TRIALS.len() {
            let s = run(k);
            let a = &s.analysis;
            println!("{}:", TRIALS[k].button);
            println!("  diff {} sds {} {} means {} {}", num(a.diff, 3), num(a.sds[0], 3), num(a.sds[1], 3), num(a.means[0], 3), num(a.means[1], 3));
            for (name, t) in [("welch", a.welch), ("pooled", a.pooled), ("paired", a.paired_t)] {
                if let Some(t) = t {
                    println!("  {name}: se {} t {} df {} p {} ({})", num(t.se, 3), num(t.t, 3), num(t.df, 2), p_text(t.p), t.p);
                }
            }
            println!("  shuffles: {} of {} p {}", s.shuffles.reached, s.shuffles.done(), p_text(s.shuffles.p()));
            if let Some(r) = a.ranks {
                println!("  ranks: {} p {} exact {}", r.stat, p_text(r.p), r.exact);
            }
            println!("  d {:?} sup {:?} corr {:?}", a.d.map(|d| num(d, 3)), a.superiority.map(percent), a.corr.map(|c| num(c, 3)));
            if s.counting {
                println!("  studies: {:?} of {:?}: {:?}", s.studies.rejected, s.studies.tested, (0..4).map(|k| s.studies.share(k).map(percent)).collect::<Vec<_>>());
            }
        }
    }

    /// How the first three experiments' sample was chosen: of the seeds whose clear
    /// difference is near the truth, what each shows with more noise and with a wild value.
    #[test]
    #[ignore]
    fn explore_seeds() {
        let base = Truth { paired: false, rho: 0.0, n: [20, 20], mean: [0.0, 1.0], sd: [1.0, 1.0], shape: Shape::Bell };
        for seed in 1..=60u64 {
            let d1 = sample(&base, seed);
            let a1 = analyse(&d1, false);
            let d3 = sample(&Truth { sd: [3.0, 3.0], ..base }, seed);
            let a3 = analyse(&d3, false);
            let (w1, w3) = (a1.welch.unwrap(), a3.welch.unwrap());
            if (a1.diff - 1.0).abs() < 0.12 && w1.t > 2.9 && w1.t < 3.6 {
                let s1 = shuffled_p(&d1, false, SHUFFLES, seed).unwrap();
                let s3 = shuffled_p(&d3, false, SHUFFLES, seed).unwrap();
                println!(
                    "seed {seed}: diff {:.3} t {:.2} p {:.4} shuf {:.4} ranks {:.4} d {:.2} sup {:.3} | sd3: diff {:.3} t {:.2} p {:.3} shuf {:.3} ranks {:.3}",
                    a1.diff,
                    w1.t,
                    w1.p,
                    s1,
                    a1.ranks.unwrap().p,
                    a1.d.unwrap(),
                    a1.superiority.unwrap(),
                    a3.diff,
                    w3.t,
                    w3.p,
                    s3,
                    a3.ranks.unwrap().p
                );
                let top = (0..20).max_by(|&i, &j| d1[1][i].total_cmp(&d1[1][j])).unwrap_or(0);
                for far in [9.0, 12.0, 15.0, 20.0] {
                    let mut d = d1.clone();
                    d[1][top] = far;
                    let a = analyse(&d, false);
                    print!("  wild {far}: p {:.4} shuf {:.4} ranks {:.4};", a.welch.unwrap().p, shuffled_p(&d, false, SHUFFLES, seed).unwrap(), a.ranks.unwrap().p);
                }
                println!();
            }
        }
    }

    /// Candidate settings for the experiments, and the counts over new studies behind them.
    #[test]
    #[ignore]
    fn explore_experiments() {
        let base = Truth { paired: false, rho: 0.0, n: [20, 20], mean: [0.0, 1.0], sd: [1.0, 1.0], shape: Shape::Bell };
        for seed in [3u64, 5, 6, 8] {
            for sd in [2.0, 2.5, 3.0] {
                let t = Truth { sd: [sd, sd], ..base };
                line(&format!("noise {sd} seed {seed}"), &sample(&t, seed), false, seed);
            }
            let data = sample(&base, seed);
            let top = (0..20).max_by(|&i, &j| data[1][i].total_cmp(&data[1][j])).unwrap_or(0);
            println!("seed {seed}: B's largest is point {top} at {:.3}", data[1][top]);
            for far in [6.0, 9.0, 12.0] {
                let mut d = data.clone();
                d[1][top] = far;
                line(&format!("wild {far} seed {seed}"), &d, false, seed);
            }
        }
        for seed in 1..=8u64 {
            for (shift, rho) in [(0.5, 0.9), (0.5, 0.8), (0.4, 0.9)] {
                let t = Truth { paired: true, rho, n: [20, 20], mean: [0.0, shift], ..base };
                let data = sample(&t, seed);
                line(&format!("pairs {shift} rho {rho} seed {seed}"), &data, true, seed);
            }
        }
        for (shift, rho) in [(0.5, 0.9), (0.5, 0.8), (0.4, 0.9)] {
            let t = Truth { paired: true, rho, n: [20, 20], mean: [0.0, shift], ..base };
            println!("pairs {shift} rho {rho}: studies {}", count(&t));
        }
        for (n, sd) in [([10, 40], [3.0, 1.0]), ([10, 40], [4.0, 1.0]), ([10, 40], [2.0, 1.0])] {
            let t = Truth { n, mean: [0.0, 0.0], sd, ..base };
            println!("unequal {n:?} {sd:?}: studies {}", count(&t));
            for seed in 1..=4u64 {
                line(&format!("unequal {n:?} {sd:?} seed {seed}"), &sample(&t, seed), false, seed);
            }
        }
        println!("clear: studies {}", count(&base));
        println!("noise 3: studies {}", count(&Truth { sd: [3.0, 3.0], ..base }));
    }
}

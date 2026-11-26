//! Looking at a sample: one sample of numbers and the standard pictures of it, all on
//! one axis: the points themselves, a histogram, a smoothed curve, a box plot, the
//! cumulative steps and a quantile plot against a bell. What a choice of width (or of
//! where the bins start) hides or invents, measured against the distribution the
//! points came from; what a box cannot show; and which summaries one wild point moves.

use super::plot::{self, Glide, Plot, Rect, AXES, CURVES, MARKS, OWN};
use super::*;
use crate::math::stats::{self as st, Dist};

pub const DEF: SceneDef = SceneDef {
    id: "describe",
    title: "Looking at a sample",
    group: ESTIMATION,
    blurb: "Five pictures of one sample: histogram, smoothed density, box plot, cumulative steps and quantile plot, and what each choice of width hides or invents.",
    make: |_| Box::new(Describe::create()),
};

// ----- where the sample comes from ------------------------------------------------------

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Source {
    Bell,
    Tail,
    Humps,
    Wild,
    /// Two sets of points made to share their five numbers (not drawn from anything).
    Pair,
}

impl Source {
    const ALL: [Source; 5] = [Source::Bell, Source::Tail, Source::Humps, Source::Wild, Source::Pair];

    /// Its name on a button, and on a phone's.
    fn name(self) -> &'static str {
        match self {
            Source::Bell => "a bell",
            Source::Tail => "a long right tail",
            Source::Humps => "two humps",
            Source::Wild => "wild values",
            Source::Pair => "same five numbers",
        }
    }

    fn short(self) -> &'static str {
        match self {
            Source::Bell => "bell",
            Source::Tail => "long tail",
            Source::Humps => "humps",
            Source::Wild => "wild",
            Source::Pair => "same box",
        }
    }

    fn about(self) -> &'static str {
        match self {
            Source::Bell => "A bell: the normal distribution with mean 0 and standard deviation 1.",
            Source::Tail => "A long right tail, like incomes or waiting times: log-normal (its logarithm is a bell of mean 0 and standard deviation 0.6), with median 1 and mean 1.20.",
            Source::Humps => r"Two humps: each point from a bell at $-1.2$ or one at $1.2$, as by a coin toss, each bell of standard deviation 0.5.",
            Source::Wild => "A bell with a few wild values: each point from a bell of standard deviation 1, or, one time in twenty, from one four times as wide.",
            Source::Pair => r"Two sets of points made to have the same five numbers (least $-3.2$, quartiles $-1$ and $1$, median $0$, greatest $3.2$): one with a single hump, one with two. They are not drawn from any distribution, so there is no truth to measure them against.",
        }
    }

    /// The parts it is a mixture of: each one's share and its distribution.
    fn parts(self) -> &'static [(f64, Dist)] {
        match self {
            Source::Bell => &[(1.0, Dist::Normal { mean: 0.0, sd: 1.0 })],
            Source::Tail => &[(1.0, Dist::LogNormal { mu: 0.0, sigma: 0.6 })],
            Source::Humps => &[(0.5, Dist::Normal { mean: -1.2, sd: 0.5 }), (0.5, Dist::Normal { mean: 1.2, sd: 0.5 })],
            Source::Wild => &[(0.95, Dist::Normal { mean: 0.0, sd: 1.0 }), (0.05, Dist::Normal { mean: 0.0, sd: 4.0 })],
            Source::Pair => &[],
        }
    }

    fn truth(self) -> bool {
        self != Source::Pair
    }

    fn pdf(self, x: f64) -> f64 {
        self.parts().iter().map(|(w, d)| w * d.pdf(x)).sum()
    }

    fn cdf(self, x: f64) -> f64 {
        self.parts().iter().map(|(w, d)| w * d.cdf(x)).sum()
    }

    /// Its middle: where the bins' edges start from at first.
    fn centre(self) -> f64 {
        if self == Source::Tail {
            1.0
        } else {
            0.0
        }
    }

    /// The stretch of the axis shown whatever the points are.
    fn core(self) -> (f64, f64) {
        match self {
            Source::Bell => (-3.5, 3.5),
            Source::Tail => (-0.3, 5.0),
            Source::Humps => (-3.2, 3.2),
            Source::Wild => (-4.0, 4.0),
            Source::Pair => (-3.6, 3.6),
        }
    }

    /// The area under the square of the density, ∫ f², in closed form (the errors of
    /// the pictures are measured against it).
    fn square_integral(self) -> f64 {
        let parts = self.parts();
        let mut sum = 0.0;
        for (wa, a) in parts {
            for (wb, b) in parts {
                sum += wa
                    * wb
                    * match (a, b) {
                        // two bells: the density of their difference at 0
                        (Dist::Normal { mean: ma, sd: sa }, Dist::Normal { mean: mb, sd: sb }) => {
                            let s = (sa * sa + sb * sb).sqrt();
                            st::normal_pdf((ma - mb) / s) / s
                        }
                        (Dist::LogNormal { mu, sigma }, Dist::LogNormal { .. }) => (-mu + sigma * sigma / 4.0).exp() / (2.0 * sigma * std::f64::consts::PI.sqrt()),
                        _ => f64::NAN,
                    };
            }
        }
        sum
    }

    fn draw(self, rng: &mut Rng) -> f64 {
        let parts = self.parts();
        let u = st::unit(rng);
        let mut below = 0.0;
        for (w, d) in parts {
            below += w;
            if u < below {
                return d.sample(rng);
            }
        }
        parts.last().map_or(0.0, |(_, d)| d.sample(rng))
    }
}

/// `n` points drawn from `source` (for the pair, the first of its two sets).
fn draw(source: Source, n: usize, seed: u64) -> Vec<f64> {
    if source == Source::Pair {
        let [one, _] = pair(n, seed);
        return one;
    }
    let mut rng = Rng::new(seed);
    (0..n).map(|_| source.draw(&mut rng)).collect()
}

/// The five numbers both sets of the pair are made to have: least, lower quartile,
/// median, upper quartile, greatest.
const FIVE: [f64; 5] = [-3.2, -1.0, 0.0, 1.0, 3.2];

/// How the pair's points lie within each quarter, for each set: in the inner quarters
/// (between the median and a quartile) and the outer ones (between a quartile and the
/// end), whether they crowd at the quartile (or else at the other end), and how much
/// (the spread of a falling exponential, as a share of the quarter: small crowds,
/// large is nearly even). The one-hump set crowds its inner points at the median; the
/// two-hump set crowds all of its points at the quartiles, and leaves the middle
/// nearly empty.
const CROWD: [[(bool, f64); 2]; 2] = [[(false, 1.2), (true, 1.4)], [(true, 0.5), (true, 0.2)]];

/// How many points the pair has, near `n`: one more than a multiple of four, so that
/// its quartiles fall on points of its own.
fn pair_size(n: usize) -> usize {
    4 * ((n.max(5) - 1 + 2) / 4).max(1) + 1
}

/// Two sets of points with the same five numbers (least, quartiles, median,
/// greatest), one with a single hump and one with two: made, not drawn from any
/// distribution. The five numbers are put in place, and the points between each two of
/// them scattered at random as `CROWD` says. Both come sorted, so that point `i` of one
/// is point `i` of the other: the same rank, and the five numbers the same points.
fn pair(n: usize, seed: u64) -> [Vec<f64>; 2] {
    let n = pair_size(n);
    let quarter = (n - 1) / 4;
    let mut rng = Rng::new(seed);
    [0, 1].map(|set| {
        let mut out = vec![0.0; n];
        for q in 0..4 {
            let (lo, hi) = (FIVE[q], FIVE[q + 1]);
            // inner or outer, and whether it crowds at its low end
            let inner = q == 1 || q == 2;
            let (at_quartile, spread) = CROWD[set][if inner { 0 } else { 1 }];
            let low_end = if inner { (q == 2) != at_quartile } else { (q == 3) == at_quartile };
            let mut u: Vec<f64> = (1..quarter).map(|_| st::unit(&mut rng)).collect();
            u.sort_by(f64::total_cmp);
            out[q * quarter] = lo;
            for (k, u) in u.iter().enumerate() {
                // a falling exponential on 0..1, cut off at 1 (its inverse share)
                let t = -spread * (u * (-1.0 / spread).exp_m1()).ln_1p();
                let t = if low_end { t } else { 1.0 - t };
                out[q * quarter + 1 + k] = lo + (hi - lo) * t.max(0.0).min(1.0);
            }
        }
        out[n - 1] = FIVE[4];
        out.sort_by(f64::total_cmp);
        out
    })
}

// ----- the histogram ------------------------------------------------------------------

/// A histogram's bars: the left edge of the first, the width of each, and how many of
/// the sample's points fall in each, from the bar of the least point to the bar of the
/// greatest. A bar holds the points from its left edge up to, not including, its
/// right edge.
#[derive(Clone, Debug, Default, PartialEq)]
struct Bars {
    first: f64,
    width: f64,
    counts: Vec<u32>,
    n: usize,
}

/// The most bars a histogram is given (far more than any setting here asks for).
const MOST_BARS: usize = 20_000;

/// The histogram of `x` with bars `width` wide, one of whose edges is at `mark`.
fn bars(x: &[f64], mark: f64, width: f64) -> Bars {
    let (lo, hi) = extent(x);
    if x.is_empty() || width.is_nan() || width <= 0.0 || !lo.is_finite() || !hi.is_finite() || !mark.is_finite() {
        return Bars { first: 0.0, width: width.max(1e-9), counts: Vec::new(), n: 0 };
    }
    let first = mark + ((lo - mark) / width).floor() * width;
    let count = (((hi - first) / width).floor().max(0.0) as usize + 1).min(MOST_BARS);
    let mut counts = vec![0u32; count];
    for v in x {
        let i = ((v - first) / width).floor().max(0.0) as usize;
        counts[i.min(count - 1)] += 1;
    }
    Bars { first, width, counts, n: x.len() }
}

impl Bars {
    /// The height of bar `i`: its share of the points per unit across, so that the
    /// bars' areas add up to 1 and the true density can be drawn over them.
    fn height(&self, i: usize) -> f64 {
        self.counts.get(i).map_or(0.0, |&c| c as f64 / (self.n.max(1) as f64 * self.width))
    }

    fn edge(&self, i: usize) -> f64 {
        self.first + i as f64 * self.width
    }

    fn tallest(&self) -> f64 {
        (0..self.counts.len()).map(|i| self.height(i)).fold(0.0, f64::max)
    }

    /// The bar a value is in.
    fn bar_of(&self, v: f64) -> Option<usize> {
        let i = ((v - self.first) / self.width).floor();
        (i >= 0.0 && (i as usize) < self.counts.len()).then_some(i as usize)
    }

    /// The squared gap between the bars and the true density, added up along the
    /// axis: ∫ (bars − f)², exactly (the bars are flat, so the cross term needs only
    /// the true share of each bar).
    fn error(&self, source: Source) -> f64 {
        let (mut square, mut cross) = (0.0, 0.0);
        // (an edge's share below it is worked out once for the two bars that meet there)
        let mut below: Option<(usize, f64)> = None;
        for (i, &c) in self.counts.iter().enumerate() {
            if c == 0 {
                continue;
            }
            let h = self.height(i);
            let left = match below {
                Some((j, f)) if j == i => f,
                _ => source.cdf(self.edge(i)),
            };
            let right = source.cdf(self.edge(i + 1));
            below = Some((i + 1, right));
            square += h * h * self.width;
            cross += h * (right - left);
        }
        square - 2.0 * cross + source.square_integral()
    }
}

fn extent(x: &[f64]) -> (f64, f64) {
    x.iter().fold((f64::INFINITY, f64::NEG_INFINITY), |(lo, hi), v| (lo.min(*v), hi.max(*v)))
}

// ----- the smoothed curve -------------------------------------------------------------

/// A kernel density estimate, the mean of a bell of spread `h` centred on each point,
/// worked out on a grid of evenly spaced places: its first place, the step between
/// places, and its value at each.
///
/// Worked out the quick way: each point's share is split between the two places
/// either side of it, in proportion to how near it is to each, and each place's share
/// is then spread out as a bell (cut off four spreads out, where 0.006 % of it is
/// left). With places a fifth of a spread apart or closer, that is within one percent
/// of the curve's highest value of the exact sum, between the places too (a test
/// holds it).
#[derive(Clone, Debug, Default, PartialEq)]
struct Smooth {
    lo: f64,
    step: f64,
    values: Vec<f64>,
}

/// The most places a smoothed curve is worked out at.
const MOST_PLACES: usize = 40_000;

/// How many places the smoothed curve is drawn at, across the axis shown.
const CURVE_POINTS: usize = 241;

impl Smooth {
    /// The curve of `x` with bandwidth `h`, over `lo` to `hi` and four bandwidths
    /// beyond (so that it has all of every bump of every point inside).
    fn of(x: &[f64], h: f64, lo: f64, hi: f64) -> Smooth {
        if x.is_empty() || h.is_nan() || h <= 0.0 || !(lo.is_finite() && hi.is_finite()) {
            return Smooth::default();
        }
        let (lo, hi) = (lo.min(hi) - 4.0 * h, lo.max(hi) + 4.0 * h);
        let span = hi - lo;
        let step = (h / 5.0).min(span / 480.0).max(span / MOST_PLACES as f64);
        let places = (span / step).ceil() as usize + 2;
        Smooth::on(x, h, lo, step, places)
    }

    /// The same, for drawing at `points` evenly spaced places from `lo` to `hi`: on a
    /// grid that has those places among its own, a whole number of steps apart, so that
    /// the values drawn are worked out there and not read between two places. (Read
    /// between places of a grid that moves with the bandwidth, a curve wobbles by a
    /// fraction of a pixel while the bandwidth changes.)
    fn drawn(x: &[f64], h: f64, lo: f64, hi: f64, points: usize) -> Smooth {
        if x.is_empty() || h.is_nan() || h <= 0.0 || !(lo.is_finite() && hi.is_finite() && hi > lo) || points < 2 {
            return Smooth::default();
        }
        let pitch = (hi - lo) / (points - 1) as f64;
        let most = (MOST_PLACES / points).max(1) as f64;
        let between = (5.0 * pitch / h).ceil().max(1.0).min(most);
        let step = pitch / between;
        let before = (4.0 * h / step).ceil().min(MOST_PLACES as f64 / 4.0);
        let places = ((points - 1) as f64 * between + 2.0 * before) as usize + 1;
        Smooth::on(x, h, lo - before * step, step, places)
    }

    /// The curve on `places` places `step` apart from `start`. (A point beyond the grid
    /// is left out: on the grid, its bump would be smaller than a millionth of its top.)
    fn on(x: &[f64], h: f64, start: f64, step: f64, places: usize) -> Smooth {
        let (lo, places) = (start, places.max(2));
        let mut share = vec![0.0f64; places];
        let each = 1.0 / x.len().max(1) as f64;
        for v in x {
            let t = (v - lo) / step;
            if !(t >= 0.0 && t <= (places - 1) as f64) {
                continue;
            }
            let i = (t.floor() as usize).min(places - 2);
            let frac = (t - i as f64).min(1.0);
            share[i] += each * (1.0 - frac);
            share[i + 1] += each * frac;
        }
        let reach = (4.0 * h / step).ceil() as usize;
        let bump: Vec<f64> = (0..=reach).map(|j| st::normal_pdf(j as f64 * step / h) / h).collect();
        let mut values = vec![0.0f64; places];
        for (i, &s) in share.iter().enumerate() {
            if s == 0.0 {
                continue;
            }
            let (from, to) = (i.saturating_sub(reach), (i + reach).min(places - 1));
            for (k, value) in values[from..=to].iter_mut().enumerate() {
                *value += s * bump[(from + k).abs_diff(i)];
            }
        }
        Smooth { lo, step, values }
    }

    /// The value at x, between the two places either side of it (0 off the grid).
    fn at(&self, x: f64) -> f64 {
        let t = (x - self.lo) / self.step;
        if t.is_nan() || t < 0.0 || self.values.is_empty() {
            return 0.0;
        }
        let i = t.floor() as usize;
        match (self.values.get(i), self.values.get(i + 1)) {
            (Some(a), Some(b)) => a + (b - a) * (t - i as f64),
            (Some(a), None) if t == i as f64 => *a,
            _ => 0.0,
        }
    }

    /// The area under it (the trapezoid rule over its places).
    #[cfg(test)]
    fn area(&self) -> f64 {
        let ends = self.values.first().copied().unwrap_or(0.0) + self.values.last().copied().unwrap_or(0.0);
        (self.values.iter().sum::<f64>() - 0.5 * ends) * self.step
    }

    fn tallest(&self) -> f64 {
        self.values.iter().copied().fold(0.0, f64::max)
    }

    /// The squared gap between it and the true density, added up along the axis:
    /// over its grid by the trapezoid rule, and beyond the grid (where it is 0) the
    /// true density's own ∫ f² there.
    fn error(&self, source: Source) -> f64 {
        let (mut gap, mut square) = (0.0, 0.0);
        let last = self.values.len().saturating_sub(1);
        for (i, v) in self.values.iter().enumerate() {
            let f = source.pdf(self.lo + i as f64 * self.step);
            let w = if i == 0 || i == last { 0.5 } else { 1.0 };
            gap += w * (v - f) * (v - f);
            square += w * f * f;
        }
        gap * self.step + (source.square_integral() - square * self.step).max(0.0)
    }
}

/// One point's bump at x: a bell of spread `h` with area 1/n.
fn bump(x: f64, centre: f64, h: f64, n: usize) -> f64 {
    st::normal_pdf((x - centre) / h) / (h * n.max(1) as f64)
}

// ----- the widths a picture can have ---------------------------------------------------

/// The range of the bins' width and of the bandwidth.
const WIDTHS: (f64, f64) = (0.02, 4.0);
const BANDS: (f64, f64) = (0.02, 2.0);
/// How many widths the error is worked out at, for its curve.
const TRIED: usize = 40;

/// The `k`-th of the widths tried, evenly spaced by ratio over `range`.
fn tried(range: (f64, f64), k: usize) -> f64 {
    range.0 * (range.1 / range.0).powf(k as f64 / (TRIED - 1) as f64)
}

/// A common rule of thumb for the bins (Freedman and Diaconis's): twice the
/// interquartile range divided by the cube root of n. (With no spread between the
/// quartiles, Scott's rule, from the standard deviation; with no spread at all, none.)
fn bin_rule(sorted: &[f64]) -> Option<f64> {
    let n = sorted.len() as f64;
    let iqr = st::quantile_sorted(sorted, 0.75) - st::quantile_sorted(sorted, 0.25);
    let w = if iqr > 0.0 { 2.0 * iqr / n.cbrt() } else { 3.49 * st::sd(sorted) / n.cbrt() };
    (w > 0.0 && w.is_finite()).then_some(w.max(WIDTHS.0).min(WIDTHS.1))
}

/// A common rule of thumb for the bandwidth (Silverman's, R's default): 0.9 times the smaller of
/// the standard deviation and the interquartile range over 1.34, divided by the fifth
/// root of n. Made for a sample from a bell.
fn band_rule(sorted: &[f64]) -> Option<f64> {
    let n = sorted.len() as f64;
    let sd = st::sd(sorted);
    let iqr = (st::quantile_sorted(sorted, 0.75) - st::quantile_sorted(sorted, 0.25)) / 1.34;
    let spread = if sd > 0.0 && iqr > 0.0 { sd.min(iqr) } else { sd.max(iqr) };
    let h = 0.9 * spread / n.powf(0.2);
    (h > 0.0 && h.is_finite()).then_some(h.max(BANDS.0).min(BANDS.1))
}

// ----- the box ------------------------------------------------------------------------

/// What a box plot draws: the quartiles and the median (by `quantile_sorted`), and the
/// whiskers' ends, the furthest points within one and a half box lengths of the box
/// (or the box's own edge, if there is none beyond it within that reach).
#[derive(Clone, Copy, Debug, PartialEq)]
struct Five {
    q1: f64,
    median: f64,
    q3: f64,
    low: f64,
    high: f64,
}

impl Five {
    fn of(sorted: &[f64]) -> Option<Five> {
        if sorted.is_empty() {
            return None;
        }
        let q = |p| st::quantile_sorted(sorted, p);
        let (q1, median, q3) = (q(0.25), q(0.5), q(0.75));
        let reach = 1.5 * (q3 - q1);
        // (with few points, the furthest within reach can be inside the box, between a
        // quartile and the point next to it: the whisker then has no length)
        let low = sorted.iter().copied().find(|v| *v >= q1 - reach).unwrap_or(q1).min(q1);
        let high = sorted.iter().rev().copied().find(|v| *v <= q3 + reach).unwrap_or(q3).max(q3);
        Some(Five { q1, median, q3, low, high })
    }

    fn iqr(&self) -> f64 {
        self.q3 - self.q1
    }

    /// Is a point beyond the whiskers (marked singly)?
    fn beyond(&self, v: f64) -> bool {
        v < self.low || v > self.high
    }

    /// The line through the quartiles in the quantile plot, at x (none without a box).
    fn line(&self, x: f64) -> f64 {
        if self.iqr() > 0.0 {
            quartile_score() * (2.0 * (x - self.q1) / self.iqr() - 1.0)
        } else {
            f64::NAN
        }
    }
}

/// Where point `i` (from 0) of `n` sorted ones goes up the quantile plot, its normal
/// score: the value a standard bell has a share (i + ½)/n below.
fn normal_score(i: usize, n: usize) -> f64 {
    st::normal_quantile((i as f64 + 0.5) / n.max(1) as f64)
}

/// The normal score of a quartile: where the quartiles go up the quantile plot.
fn quartile_score() -> f64 {
    st::normal_quantile(0.75)
}

/// How many peaks a row of values has: places higher than the nearest different value
/// on each side (a flat top counts once).
#[cfg(test)]
fn peaks(v: &[f64]) -> usize {
    let mut count = 0;
    let mut rising = false;
    let mut last = f64::NEG_INFINITY;
    for &x in v {
        if x > last {
            rising = true;
        } else if x < last {
            if rising {
                count += 1;
            }
            rising = false;
        }
        last = x;
    }
    count + rising as usize
}

/// 1st, 2nd, 3rd, 4th, ..., 11th, 12th, 13th, ..., 21st.
fn ordinal(k: usize) -> String {
    let suffix = match (k % 10, k % 100) {
        (_, 11..=13) => "th",
        (1, _) => "st",
        (2, _) => "nd",
        (3, _) => "rd",
        _ => "th",
    };
    format!("{k}{suffix}")
}

/// The least of a list and where it is (None if empty or not a number).
fn least(v: &[f64]) -> Option<(usize, f64)> {
    v.iter().copied().enumerate().filter(|(_, e)| e.is_finite()).min_by(|a, b| a.1.total_cmp(&b.1))
}

// ----- the experiments ----------------------------------------------------------------

/// Something an experiment sets moving: the bins' start sliding through one width, the
/// pair's points flowing from one set to the other, one point flying out.
#[derive(Clone, Copy, Debug, PartialEq)]
enum Act {
    Still,
    Sweep,
    Flow,
    Fly(f64),
}

struct Trial {
    button: &'static str,
    caption: &'static str,
    source: Source,
    n: usize,
    seed: u64,
    /// The bins' width and one of their edges (None: the rule of thumb).
    bins: Option<(f64, f64)>,
    /// The bandwidth (None: the rule of thumb).
    band: Option<f64>,
    /// Which pair of pictures a phone shows.
    tab: usize,
    act: Act,
}

/// The seeds the experiments draw with.
const HUMPS_SEED: u64 = 4;
const PAIR_SEED: u64 = 3;
const TAIL_SEED: u64 = 1;
const WILD_SEED: u64 = 5;

const TRIALS: [Trial; 6] = [
    Trial {
        button: "too wide hides",
        caption: "Two humps, 200 points. Bins 2.4 wide make one hump of the two. Their error (the squared gap to the green truth, added up along the axis) is 0.13; from this start the least, 0.016, is near a width of 0.35. The smoothed curve uses a common rule of thumb for its bandwidth (Silverman's, R's default), which is made for a bell: here its bandwidth, 0.41, is about twice the best, 0.21, and in the dip between the humps the curve stands at more than twice the true density (error 0.015, against 0.006 at the best).",
        source: Source::Humps,
        n: 200,
        seed: HUMPS_SEED,
        bins: Some((2.4, -1.2)),
        band: None,
        tab: 0,
        act: Act::Still,
    },
    Trial {
        button: "too narrow invents",
        caption: "The same points. Bins 0.06 wide: a comb of 26 peaks where the truth has two (error 0.086, against 0.015 near a width of 0.46). A bandwidth of 0.03: 32 peaks (error 0.043, against 0.006 near 0.21). Too narrow, and a picture shows this sample's accidents, not where its points came from.",
        source: Source::Humps,
        n: 200,
        seed: HUMPS_SEED,
        bins: Some((0.06, 0.0)),
        band: Some(0.03),
        tab: 0,
        act: Act::Still,
    },
    Trial {
        button: "move the bins",
        caption: "The same points in bins 1.2 wide, with the bins' start (the white handle at the bottom of the histogram) sliding one width to the right over six seconds. From 0, four bars of 43 to 57 points make one flat-topped heap; half a bin on, the middle bar holds 19 points and its neighbours 72 and 90: two humps. Then one heap again. Same points, same width: only where the bins start has changed.",
        source: Source::Humps,
        n: 200,
        seed: HUMPS_SEED,
        bins: Some((1.2, 0.0)),
        band: None,
        tab: 0,
        act: Act::Sweep,
    },
    Trial {
        button: "what a box cannot show",
        caption: r"Two sets of 201 points, made to have the same five numbers: least $-3.2$, quartiles $-1$ and $1$, median $0$, greatest $3.2$. The first has one hump. After a moment its points flow, each keeping its rank, into the second, which has two. The box does not move at all: it is drawn from those five numbers alone. The histogram, the curve and the steps show the change, and so does the standard deviation, from 1.49 to 1.19. “The other set”, over the picture, flows back.",
        source: Source::Pair,
        n: 201,
        seed: PAIR_SEED,
        bins: Some((0.4, 0.0)),
        band: Some(0.3),
        tab: 0,
        act: Act::Flow,
    },
    Trial {
        button: "straight means a bell",
        caption: "A long right tail, 300 points. In the quantile plot, points from a bell would lie along a straight line, like the orange one through the quartiles. These rise steeply on the left and flatten out to the right: the largest lie much further out, and the smallest closer in, than a bell's would. Choose the other sources over the picture: a bell's points follow the line, loosely at the very ends (where any sample strays most); two humps make a flat step in the middle; wild values flatten it at both ends. (Many programs draw it the other way round, the normal scores across: then a long right tail bends up at the right.)",
        source: Source::Tail,
        n: 300,
        seed: TAIL_SEED,
        bins: None,
        band: None,
        tab: 1,
        act: Act::Still,
    },
    Trial {
        button: "one wild value",
        caption: r"A bell, 40 points. After a moment one of the two points in the middle (the 20th from the bottom) flies out to 7. The mean goes from 0.14 to 0.31 and the standard deviation from 0.87 to 1.39; the median from $-0.01$ to 0.05, and the interquartile range from 1.21 to 1.31. Drag it further out: the mean follows it a fortieth of the way and the standard deviation keeps growing, while the median and the quartiles do not move at all.",
        source: Source::Bell,
        n: 40,
        seed: WILD_SEED,
        bins: None,
        band: None,
        tab: 0,
        act: Act::Fly(7.0),
    },
];

/// How the scene opens.
const OPENING: (Source, usize, u64) = (Source::Humps, 200, HUMPS_SEED);

/// An experiment's movement, under way: what moves, and how long it has been going
/// (seconds of the scene's time).
#[derive(Clone, Debug, PartialEq)]
struct Running {
    script: Script,
    t: f32,
    wait: f32,
}

#[derive(Clone, Debug, PartialEq)]
enum Script {
    /// The bins' edge at `from` slides one width to the right.
    Sweep { from: f64 },
    /// The pair's points flow to set `to`.
    Flow { to: usize },
    /// Point `index` flies from `from` to `to`.
    Fly { index: usize, from: f64, to: f64 },
}

impl Script {
    /// How long the movement takes (seconds).
    fn length(&self) -> f32 {
        match self {
            Script::Sweep { .. } => 6.0,
            Script::Flow { .. } => 2.4,
            Script::Fly { .. } => 1.8,
        }
    }
}

impl Running {
    /// How far through, from 0 to 1, eased in and out.
    fn share(&self) -> f64 {
        let s = ((self.t - self.wait) / self.script.length()).max(0.0).min(1.0) as f64;
        // (the sweep goes at an even pace: it is the bins' start that matters, not how it gets there)
        match self.script {
            Script::Sweep { .. } => s,
            _ => s * s * (3.0 - 2.0 * s),
        }
    }

    fn over(&self) -> bool {
        self.t >= self.wait + self.script.length()
    }
}

// ----- what is worked out from the sample ---------------------------------------------

/// What the pictures and the numbers need of the sample as it is (not as drawn).
#[derive(Clone, Debug, Default)]
struct Stats {
    five: Option<Five>,
    mean: f64,
    sd: f64,
    /// The errors of the histogram and the smoothed curve as they are (NaN without a truth).
    bars_error: f64,
    smooth_error: f64,
    /// The highest the bars, the curve and the truth reach.
    tallest: f64,
    /// The stretch of the axis the pictures show.
    range: [f64; 2],
    outliers: usize,
    /// A number that changes when the sample does (and not when only a width does).
    sample: u64,
}

/// The errors of the histogram (with the bins' edges where they are) and of the
/// smoothed curve at each of the widths tried, for the curves of error against width.
/// Worked out a few widths at a time, within a time allowance each frame.
#[derive(Clone, Debug, Default)]
struct Errors {
    /// What they are the errors of: the sample's version and the bins' edge.
    key: Option<(u64, u64)>,
    bars: Vec<f64>,
    smooth: Vec<f64>,
}

impl Errors {
    fn done(&self, key: (u64, u64), source: Source) -> bool {
        self.key == Some(key) && (!source.truth() || (self.bars.len() == TRIED && self.smooth.len() == TRIED))
    }

    /// Work on them until `spent` says to stop.
    fn work(&mut self, key: (u64, u64), x: &[f64], mark: f64, source: Source, spent: impl Fn() -> bool) {
        if self.key != Some(key) {
            *self = Errors { key: Some(key), ..Default::default() };
        }
        if !source.truth() || x.is_empty() {
            return;
        }
        let (lo, hi) = extent(x);
        while self.bars.len() < TRIED || self.smooth.len() < TRIED {
            if self.bars.len() < TRIED {
                let w = tried(WIDTHS, self.bars.len());
                self.bars.push(bars(x, mark, w).error(source));
            } else {
                let h = tried(BANDS, self.smooth.len());
                self.smooth.push(Smooth::of(x, h, lo, hi).error(source));
            }
            if spent() {
                break;
            }
        }
    }
}

/// What is drawn, glided to where it is going (see `plot::Glide`): the points, the
/// heights they are stacked to in the strip, the bars' heights, the bins' edge and
/// width and the bandwidth, the stretch of the axis, the tops of the density charts
/// and of the quantile plot, and the curves of error against width (their
/// logarithms). (A bar's height steps whenever a point crosses an edge: drawn as it
/// steps, the bars would twitch while the bins or the points move.)
#[derive(Default)]
struct Look {
    dots: Glide,
    levels: Glide,
    heights: Glide,
    /// The box's five numbers (a whisker's end jumps from one point to the next as
    /// the reach of the box passes points).
    five: Glide,
    /// Where the point under the pointer is marked: its bar's edges and height, its
    /// height in the steps and in the quantile plot (which step as it passes other
    /// points while it is dragged), and which point it is.
    marks: Glide,
    marked: Option<usize>,
    params: Glide,
    range: Glide,
    tops: Glide,
    errors: Glide,
    moving: bool,
}

/// What the pictures draw, worked out from the points as they are shown.
#[derive(Clone, Debug, Default)]
struct Drawn {
    /// The points in rising order, the place in the sample of each, and the place in
    /// that order of each point of the sample.
    sorted: Vec<f64>,
    order: Vec<usize>,
    rank: Vec<usize>,
    bars: Bars,
    smooth: Smooth,
    five: Option<Five>,
    mean: f64,
    sd: f64,
    /// The normal score of each place in the order.
    scores: Vec<f64>,
}

impl Drawn {
    fn of(dots: &[f32], mark: f64, width: f64, band: f64, range: [f64; 2], scores: Vec<f64>) -> Drawn {
        let x: Vec<f64> = dots.iter().map(|v| *v as f64).collect();
        let mut order: Vec<usize> = (0..x.len()).collect();
        order.sort_by(|&a, &b| x[a].total_cmp(&x[b]));
        let sorted: Vec<f64> = order.iter().map(|&i| x[i]).collect();
        let mut rank = vec![0; x.len()];
        for (r, &i) in order.iter().enumerate() {
            rank[i] = r;
        }
        Drawn {
            bars: bars(&x, mark, width),
            smooth: Smooth::drawn(&x, band, range[0], range[1], CURVE_POINTS),
            five: Five::of(&sorted),
            mean: st::mean(&x),
            sd: st::sd(&x),
            sorted,
            order,
            rank,
            scores,
        }
    }
}

/// The rank (place in rising order) each of `n` places of a new sample is given, so
/// that the old sample's places keep their order: the least of the old points' place
/// gets the least new value, and so on, spread evenly over the new ranks where there
/// are more of them; places beyond the old sample's get the ranks left over, in order.
/// A point's place in the list is only what it is known by while it is drawn: the
/// sample is the same set of values whatever their order. But the picture glides each
/// dot from its old place to its new one, and this way the shape changes in order
/// instead of in a scramble.
fn ranks_for(old: &[f64], n: usize) -> Vec<usize> {
    let kept = old.len().min(n);
    let mut places: Vec<usize> = (0..kept).collect();
    places.sort_by(|&a, &b| old[a].total_cmp(&old[b]));
    let mut rank = vec![0; n];
    let mut taken = vec![false; n];
    for (k, &place) in places.iter().enumerate() {
        let r = if kept > 1 { (k * (n - 1) + (kept - 1) / 2) / (kept - 1) } else { (n - 1) / 2 };
        rank[place] = r.min(n - 1);
        taken[r.min(n - 1)] = true;
    }
    let mut free = (0..n).filter(|r| !taken[*r]);
    for r in rank.iter_mut().skip(kept) {
        *r = free.next().unwrap_or(0);
    }
    rank
}

/// Heights to stack dots `gap` pixels wide at these places across (pixels, rising) so
/// that none covers another: each on the lowest level where the last dot is far
/// enough to its left.
fn stack(across: &[f32], gap: f32) -> Vec<usize> {
    let mut ends: Vec<f32> = Vec::new();
    across
        .iter()
        .map(|&x| {
            let level = ends.iter().position(|&e| x - e >= gap).unwrap_or(ends.len()).min(400);
            if level == ends.len() {
                ends.push(x);
            } else {
                ends[level] = x;
            }
            level
        })
        .collect()
}

// ----- the layout ---------------------------------------------------------------------

/// The pictures, in the order of `Layout::charts`.
const STRIP: usize = 0;
const HIST: usize = 1;
const SMOOTH: usize = 2;
const BOX: usize = 3;
const STEPS: usize = 4;
const QQ: usize = 5;

const NO_RECT: Rect = (Vec2::ZERO, Vec2::ZERO);

/// On a narrow canvas: the tabs, and the two pictures each shows under the points and
/// the box.
const TABS: [&str; 2] = ["histogram, smoothed", "cumulative, quantiles"];
const TAB_CHARTS: [[usize; 2]; 2] = [[HIST, SMOOTH], [STEPS, QQ]];

const AGAIN: &str = "draw again";
const OTHER_SET: &str = "the other set";

/// Where everything goes on a canvas.
#[derive(Clone, Copy, Debug, PartialEq)]
struct Layout {
    view: Vec2,
    narrow: bool,
    /// Text sizes: titles, words, small print.
    big: f32,
    text: f32,
    small: f32,
    tabs: Option<[Rect; 2]>,
    sources: [Rect; 5],
    again: Rect,
    other: Rect,
    /// The data rectangle of each picture (titles go above, numbers below and left).
    charts: [Rect; 6],
    shown: [bool; 6],
    buttons: bool,
    /// Room for the curves of error against width, inside the density charts, and how
    /// big they are; and where the brackets of the width and the bandwidth go up their
    /// charts (a share of the height: under the curves of error, where there are some).
    insets: bool,
    inset: Vec2,
    bracket: f64,
}

/// Rectangles for buttons side by side from (x, y), each as wide as its words (by the
/// estimate that needs no frame: the layout is worked out, and tested, without one).
fn words_row(words: &[&str], x: f32, y: f32, h: f32, size: f32, gap: f32) -> Vec<Rect> {
    let mut left = x;
    words
        .iter()
        .map(|w| {
            let width = text_width(w, size) + 18.0;
            let r = (Vec2::new(left, y), Vec2::new(left + width, y + h));
            left += width + gap;
            r
        })
        .collect()
}

/// A column of pictures: its left and right, and its pictures from the top with the
/// height of each.
type Column = (f32, f32, Vec<(usize, f32)>);

/// The width a picture has, and its error there.
type Best = Option<(f64, f64)>;

impl Layout {
    fn of(view: Vec2, inset: f32, tab: usize) -> Layout {
        let view = view.max(Vec2::splat(1.0));
        let narrow = view.x < 700.0;
        let (big, text, small) = if narrow { (12.5, 11.0, 10.0) } else { (13.5, 12.0, 10.5) };
        let side = if narrow { 8.0 } else { 16.0 };
        let numbers = if narrow { 30.0 } else { 38.0 };
        let top = inset.max(0.0) + if narrow { 6.0 } else { 10.0 };
        let row = if narrow { 22.0 } else { 24.0 };
        let (title, axis, gap) = (big + 12.0, small + 14.0, if narrow { 6.0 } else { 12.0 });
        let mut lay = Layout {
            view,
            narrow,
            big,
            text,
            small,
            tabs: None,
            sources: [NO_RECT; 5],
            again: NO_RECT,
            other: NO_RECT,
            charts: [NO_RECT; 6],
            shown: [false; 6],
            buttons: false,
            insets: false,
            inset: Vec2::ZERO,
            bracket: 0.9,
        };
        let tab = tab.min(1);
        // the buttons, where there is room for them and for pictures under them
        let buttons = view.y - top > 330.0 && view.x > 220.0;
        let mut charts_top = top;
        if buttons {
            lay.buttons = true;
            if narrow {
                let tabs = words_row(&TABS, side, top, row, text, 5.0);
                lay.tabs = Some([tabs[0], tabs[1]]);
                let w = text_width(AGAIN, small) + 18.0;
                lay.again = (Vec2::new(view.x - side - w, top), Vec2::new(view.x - side, top + row));
                let sources = plot::even_row(5, side, view.x - side, top + row + 6.0, row, 5.0);
                lay.sources = [0, 1, 2, 3, 4].map(|i| sources[i]);
                lay.other = (Vec2::new(view.x - side - 92.0, top + 2.0 * row + 12.0), Vec2::new(view.x - side, top + 3.0 * row + 12.0));
                charts_top = top + 2.0 * row + 14.0;
            } else {
                let names: Vec<&str> = Source::ALL.iter().map(|s| s.name()).collect();
                let sources = words_row(&names, side, top, row, small, 6.0);
                lay.sources = [0, 1, 2, 3, 4].map(|i| sources[i]);
                let end = sources[4].1.x;
                lay.again = words_row(&[AGAIN], end + 18.0, top, row, small, 0.0)[0];
                lay.other = words_row(&[OTHER_SET], lay.again.1.x + 6.0, top, row, small, 0.0)[0];
                charts_top = top + row + 12.0;
            }
        }
        let avail = view.y - charts_top - 4.0;
        let block = title + axis + gap;
        // (each column its left and right, and its pictures from the top with their heights)
        let mut columns: Vec<Column> = Vec::new();
        if narrow {
            let s1 = (avail * 0.12).max(40.0).min(80.0);
            let s2 = (avail * 0.09).max(34.0).min(56.0);
            let t = (avail - 4.0 * block - s1 - s2) / 2.0;
            let (x0, x1) = (side + numbers, view.x - side);
            if t >= 70.0 {
                let [a, b] = TAB_CHARTS[tab];
                columns.push((x0, x1, vec![(STRIP, s1), (BOX, s2), (a, t), (b, t)]));
            } else if avail >= 2.0 * block + 70.0 {
                // too short for more: the points and the box
                let s = (avail - 2.0 * block - 34.0).min(90.0);
                columns.push((x0, x1, vec![(STRIP, s), (BOX, 34.0)]));
            } else if avail >= block + 24.0 {
                columns.push((x0, x1, vec![(STRIP, (avail - block).min(90.0))]));
            }
        } else {
            let s = (avail * 0.15).max(46.0).min(120.0);
            let t = (avail - 3.0 * block - s) / 2.0;
            let col_gap = 36.0;
            let col_w = (view.x - 2.0 * side - col_gap) / 2.0;
            let lefts = [side + numbers, side + col_w + col_gap + numbers];
            let rights = [side + col_w, view.x - side];
            if t >= 60.0 {
                columns.push((lefts[0], rights[0], vec![(STRIP, s), (HIST, t), (SMOOTH, t)]));
                columns.push((lefts[1], rights[1], vec![(BOX, s), (STEPS, t), (QQ, t)]));
            } else if avail >= block + 24.0 {
                let s = (avail - block).min(120.0);
                columns.push((lefts[0], rights[0], vec![(STRIP, s)]));
                columns.push((lefts[1], rights[1], vec![(BOX, s)]));
            }
        }
        for (x0, x1, charts) in columns {
            let mut y = charts_top;
            for (c, h) in charts {
                lay.charts[c] = (Vec2::new(x0, y + title), Vec2::new(x1.max(x0 + 1.0), y + title + h));
                lay.shown[c] = true;
                y += h + block;
            }
        }
        let (hmin, hmax) = lay.charts[HIST];
        lay.insets = !narrow && lay.shown[HIST] && hmax.x - hmin.x >= 300.0 && hmax.y - hmin.y >= 150.0;
        if lay.insets {
            lay.inset = Vec2::new(((hmax.x - hmin.x) * 0.34).min(180.0), ((hmax.y - hmin.y) * 0.24).min(76.0));
            lay.bracket = (1.0 - (lay.inset.y + 8.0) / (hmax.y - hmin.y)) as f64;
        }
        lay
    }

    /// How much higher than the highest bar or curve the density charts reach: enough
    /// to keep everything under the brackets (and so under the curves of error).
    fn headroom(&self) -> f64 {
        1.08 / self.bracket
    }
}

// ----- the scene ----------------------------------------------------------------------

/// The handles' numbers.
const POINT: u32 = 1;
const START: u32 = 2;
const WIDTH: u32 = 3;
const BAND: u32 = 4;

/// The colour of the handles that set the widths, and of the point under the pointer.
const HAND: Color = pal::WHITE;

/// Kinds of line batch of this scene's own (see `plot`): the steps (as many corners as
/// points), the bumps of the smoothed curve (as many as there are points, up to a
/// dozen), and what marks the point under the pointer.
const STEP_LINES: u8 = OWN;
const BUMP_LINES: u8 = OWN + 1;
const HOVER_LINES: u8 = OWN + 2;

pub struct Describe {
    source: Source,
    n: usize,
    seed: u64,
    width: f64,
    mark: f64,
    auto_width: bool,
    band: f64,
    auto_band: bool,
    tab: usize,
    /// For the pair: which of its two sets is shown (or being flowed to).
    which: usize,
    /// The sample, in the order drawn: a point is known by its place here.
    x: Vec<f64>,
    pair: [Vec<f64>; 2],
    /// The point whose bump the bandwidth is measured on.
    bump: usize,
    /// Counts the changes to the sample and the settings.
    version: u64,
    stats: Stats,
    stats_for: Option<u64>,
    errors: Errors,
    running: Option<Running>,
    /// The handle held, and the point it is (for the points).
    held: Option<u32>,
    point: Option<usize>,
    /// The point offered to the hand in the last frame (a handle takes hold in the
    /// frame it is pressed, and reports a movement only in the frames after).
    offered: Option<usize>,
    /// The stretch of the axis, and the tops of the charts, while a handle is held:
    /// they stay put under the hand.
    frozen: Option<[f64; 2]>,
    frozen_tops: Option<[f32; 2]>,
    look: Look,
    drawn: Drawn,
    drawn_ok: bool,
    scores_for: usize,
    canvas: Option<Vec2>,
    /// Whether the last frame had room for the curves of error against width.
    insets: bool,
    /// The round values on the axes in the last frame, and how many grid lines and
    /// labels were drawn (see the end of `frame`).
    ticks: Vec<[i64; 3]>,
    counted: (usize, usize),
}

impl Describe {
    fn create() -> Describe {
        let (source, n, seed) = OPENING;
        let mut s = Describe {
            source,
            n,
            seed,
            width: 0.5,
            mark: source.centre(),
            auto_width: true,
            band: 0.3,
            auto_band: true,
            tab: 0,
            which: 0,
            x: Vec::new(),
            pair: [Vec::new(), Vec::new()],
            bump: 0,
            version: 0,
            stats: Stats::default(),
            stats_for: None,
            errors: Errors::default(),
            running: None,
            held: None,
            point: None,
            offered: None,
            frozen: None,
            frozen_tops: None,
            look: Look::default(),
            drawn: Drawn::default(),
            drawn_ok: false,
            scores_for: usize::MAX,
            canvas: None,
            insets: false,
            ticks: Vec::new(),
            counted: (0, 0),
        };
        s.redraw();
        s
    }

    /// Something about the sample or the settings changed.
    fn changed(&mut self) {
        self.version = self.version.wrapping_add(1);
    }

    /// Draw the sample afresh (from the source, the number of points and the seed),
    /// its points given out to the places of the last one's by rank (see `ranks_for`).
    fn redraw(&mut self) {
        let old = std::mem::take(&mut self.x);
        if self.source == Source::Pair {
            self.n = pair_size(self.n);
            let rank = ranks_for(&old, self.n);
            // (the pair's two sets come sorted: both given out alike, a point's
            // counterpart in the other set is still its own rank)
            self.pair = pair(self.n, self.seed).map(|set| rank.iter().map(|&r| set[r]).collect());
            self.which = self.which.min(1);
            self.x = self.pair[self.which].clone();
        } else {
            self.n = self.n.max(1).min(2000);
            let fresh = st::sorted(&draw(self.source, self.n, self.seed));
            self.x = ranks_for(&old, fresh.len()).iter().map(|&r| fresh[r]).collect();
        }
        // the bump the bandwidth is measured on: a point three tenths of the way up
        let s = st::sorted(&self.x);
        let at = st::quantile_sorted(&s, 0.3);
        self.bump = (0..self.x.len()).min_by(|&a, &b| (self.x[a] - at).abs().total_cmp(&(self.x[b] - at).abs())).unwrap_or(0);
        // an edge of the bins inside what is shown
        let (lo, hi) = self.source.core();
        if !(self.mark > lo && self.mark < hi) {
            self.mark = self.source.centre();
        }
        self.changed();
    }

    /// Stop whatever an experiment set moving, where it is.
    fn halt(&mut self) {
        if let Some(Running { script: Script::Flow { to }, .. }) = self.running {
            let s = self.running.as_ref().map_or(1.0, Running::share);
            self.which = if s >= 0.5 { to } else { 1 - to };
        }
        self.running = None;
    }

    /// Set up one of the experiments in `TRIALS`.
    fn set_up(&mut self, k: usize) {
        let Some(t) = TRIALS.get(k) else { return };
        self.halt();
        (self.source, self.n, self.seed, self.which, self.tab) = (t.source, t.n, t.seed, 0, t.tab);
        self.mark = t.source.centre();
        self.redraw();
        match t.bins {
            Some((w, mark)) => (self.width, self.mark, self.auto_width) = (w, mark, false),
            None => self.auto_width = true,
        }
        match t.band {
            Some(h) => (self.band, self.auto_band) = (h, false),
            None => self.auto_band = true,
        }
        self.running = match t.act {
            Act::Still => None,
            Act::Sweep => Some(Running { script: Script::Sweep { from: self.mark }, t: 0.0, wait: 0.8 }),
            Act::Flow => Some(Running { script: Script::Flow { to: 1 }, t: 0.0, wait: 1.2 }),
            Act::Fly(to) => {
                // the point in the middle flies out (of two in the middle, the lower)
                let s = st::sorted(&self.x);
                let m = s.get((s.len().max(1) - 1) / 2).copied().unwrap_or(0.0);
                let index = self.x.iter().position(|v| *v == m).unwrap_or(0);
                Some(Running { script: Script::Fly { index, from: self.x[index], to }, t: 0.0, wait: 1.0 })
            }
        };
        self.refresh();
    }

    /// Let an experiment's movement go on for `dt` seconds of the scene's time.
    fn advance(&mut self, dt: f32) {
        let Some(mut run) = self.running.take() else { return };
        run.t += dt.max(0.0);
        let s = run.share();
        match run.script {
            Script::Sweep { from } => self.mark = from + self.width * s,
            Script::Flow { to } => {
                let from = 1 - to;
                if let (Some(a), Some(b)) = (self.pair.get(from), self.pair.get(to)) {
                    if a.len() == self.x.len() && b.len() == self.x.len() {
                        for (v, (a, b)) in self.x.iter_mut().zip(a.iter().zip(b)) {
                            *v = a + (b - a) * s;
                        }
                    }
                }
                if run.over() {
                    self.which = to;
                }
            }
            Script::Fly { index, from, to } => {
                if let Some(v) = self.x.get_mut(index) {
                    *v = from + (to - from) * s;
                }
            }
        }
        self.changed();
        if !run.over() {
            self.running = Some(run);
        }
    }

    /// Run whatever an experiment set moving to its end (for the tests).
    #[cfg(test)]
    fn finish(&mut self) {
        while self.running.is_some() {
            self.advance(0.25);
        }
        self.refresh();
    }

    /// Work out again what follows from the sample and the settings, if they changed.
    fn refresh(&mut self) {
        if self.stats_for == Some(self.version) {
            return;
        }
        self.stats_for = Some(self.version);
        let sorted = st::sorted(&self.x);
        if self.auto_width {
            if let Some(w) = bin_rule(&sorted) {
                self.width = w;
            }
        }
        if self.auto_band {
            if let Some(h) = band_rule(&sorted) {
                self.band = h;
            }
        }
        self.width = self.width.max(WIDTHS.0).min(WIDTHS.1);
        self.band = self.band.max(BANDS.0).min(BANDS.1);
        let five = Five::of(&sorted);
        let (lo, hi) = extent(&sorted);
        let (core_lo, core_hi) = self.source.core();
        let pad = 0.05 * (hi - lo).max(core_hi - core_lo);
        let range = if lo.is_finite() { [core_lo.min(lo - pad), core_hi.max(hi + pad)] } else { [core_lo, core_hi] };
        let b = bars(&self.x, self.mark, self.width);
        let smooth = Smooth::of(&self.x, self.band, lo, hi);
        let truth = self.source.truth();
        let peak = if truth { (0..=400).map(|i| self.source.pdf(range[0] + (range[1] - range[0]) * i as f64 / 400.0)).fold(0.0, f64::max) } else { 0.0 };
        self.stats = Stats {
            five,
            mean: st::mean(&self.x),
            sd: st::sd(&self.x),
            bars_error: if truth { b.error(self.source) } else { f64::NAN },
            smooth_error: if truth { smooth.error(self.source) } else { f64::NAN },
            tallest: b.tallest().max(smooth.tallest()).max(peak),
            range,
            outliers: five.map_or(0, |f| sorted.iter().filter(|v| f.beyond(**v)).count()),
            sample: self.sample_hash(),
        };
    }

    /// The key of the errors' curves: the sample, and the bins' edge.
    fn error_key(&self) -> (u64, u64) {
        // (the sample's own, not the version: settings that leave the sample alone keep the curves)
        (self.stats.sample, self.mark.to_bits())
    }

    fn sample_hash(&self) -> u64 {
        use std::hash::{Hash, Hasher};
        let mut h = std::collections::hash_map::DefaultHasher::new();
        for v in &self.x {
            v.to_bits().hash(&mut h);
        }
        (self.source as u8).hash(&mut h);
        h.finish()
    }

    /// The widths with the least error, from the curves (when they are done).
    fn best(&self) -> (Best, Best) {
        if !self.errors.done(self.error_key(), self.source) || !self.source.truth() {
            return (None, None);
        }
        let b = least(&self.errors.bars).map(|(k, e)| (tried(WIDTHS, k), e));
        let s = least(&self.errors.smooth).map(|(k, e)| (tried(BANDS, k), e));
        (b, s)
    }

    /// Bring what is drawn `dt` seconds nearer to what it shows, and work out the
    /// pictures again where anything moved.
    fn follow(&mut self, lay: &Layout, dt: f32) {
        let snap = self.canvas != Some(lay.view);
        self.canvas = Some(lay.view);
        let dots: Vec<f32> = self.x.iter().map(|v| *v as f32).collect();
        let had = self.look.dots.shown.len();
        if had > 0 && dots.len() > had {
            // more points: each new one starts where the old one next to it in rank is
            // drawn, and glides out with the rest (started where they are going, they
            // would be passed by the old ones on their way)
            let mut order: Vec<usize> = (0..dots.len()).collect();
            order.sort_by(|&a, &b| dots[a].total_cmp(&dots[b]));
            let first_old = order.iter().copied().find(|&i| i < had).unwrap_or(0);
            let mut start = self.look.dots.shown.clone();
            start.resize(dots.len(), 0.0);
            let mut below = None;
            for &i in &order {
                if i < had {
                    below = Some(i);
                } else {
                    start[i] = self.look.dots.get(below.unwrap_or(first_old));
                }
            }
            self.look.dots.snap(&start);
        }
        let mut moving = self.look.dots.follow(&dots, dt, true);
        // (under the hand, a point is where the hand puts it)
        if self.point.is_some() {
            self.look.dots.snap(&dots);
        }
        let params = [self.mark as f32, self.width as f32, self.band as f32];
        moving |= self.look.params.follow(&params, dt, false);
        if matches!(self.held, Some(START | WIDTH | BAND)) || matches!(self.running, Some(Running { script: Script::Sweep { .. }, .. })) {
            self.look.params.snap(&params);
        }
        let range = self.frozen.unwrap_or(self.stats.range);
        moving |= self.look.range.follow(&[range[0] as f32, range[1] as f32], dt, false);
        // the top of the density charts, a round number above the highest
        let top = self.stats.tallest * lay.headroom();
        let step = nice_step((top / 10.0).max(1e-3) as f32) as f64;
        let top = ((top / step).ceil() * step).max(0.1);
        let z = normal_score(self.n.max(1) - 1, self.n.max(1)).max(1.0) + 0.35;
        let z = (z * 2.0).ceil() / 2.0;
        moving |= self.look.tops.follow(&self.frozen_tops.unwrap_or([top as f32, z as f32]), dt, false);
        if self.errors.done(self.error_key(), self.source) && self.source.truth() {
            let logs: Vec<f32> = self.errors.bars.iter().chain(&self.errors.smooth).map(|e| e.max(1e-6).log10() as f32).collect();
            moving |= self.look.errors.follow(&logs, dt, false);
        }
        // the pictures, from the points as shown
        if moving || !self.drawn_ok || snap {
            let p = &self.look.params.shown;
            let r = &self.look.range.shown;
            let n = self.look.dots.shown.len();
            let scores = if self.scores_for == n { std::mem::take(&mut self.drawn.scores) } else { (0..n).map(|i| normal_score(i, n)).collect() };
            self.scores_for = n;
            self.drawn = Drawn::of(&self.look.dots.shown, p[0] as f64, p[1] as f64, p[2] as f64, [r[0] as f64, r[1] as f64], scores);
            self.drawn_ok = true;
        }
        let b = &self.drawn.bars;
        let heights: Vec<f32> = (0..b.counts.len()).map(|i| b.height(i) as f32).collect();
        moving |= self.look.heights.follow(&heights, dt, true);
        if let Some(f) = self.drawn.five {
            moving |= self.look.five.follow(&[f.low, f.q1, f.median, f.q3, f.high].map(|v| v as f32), dt, false);
        }
        // the dots' heights in the strip, from where they are drawn across
        let (min, max) = lay.charts[STRIP];
        let levels: Vec<f32> = if lay.shown[STRIP] {
            let r = &self.look.range.shown;
            let plot = Plot::new((min, max), [r[0] as f64, r[1] as f64], [0.0, 1.0]);
            let across: Vec<f32> = self.drawn.sorted.iter().map(|v| plot.px_x(*v)).collect();
            let by_rank = stack(&across, 0.92 * dot_size(n_of(&self.drawn)));
            let mut out = vec![0.0f32; by_rank.len()];
            for (r, &i) in self.drawn.order.iter().enumerate() {
                out[i] = by_rank[r] as f32;
            }
            out
        } else {
            vec![0.0; self.drawn.sorted.len()]
        };
        moving |= self.look.levels.follow(&levels, dt, !snap);
        if snap {
            self.look.levels.snap(&levels);
        }
        self.look.moving = moving;
    }
}

fn n_of(d: &Drawn) -> usize {
    d.sorted.len()
}

/// A point of a chart, kept on it: what a glide takes outside for a moment (a curve
/// above the top while the top rises to it, a step beyond the edge while the axis
/// widens) is drawn at the edge, not cut off. Cut off, a curve would have fewer
/// points in some frames than in others, and nothing in its batch could be followed
/// from one frame to the next then.
fn onto(p: &Plot, (x, y): (f64, f64)) -> (f64, f64) {
    (x.max(p.x[0]).min(p.x[1]), y.max(p.y[0]).min(p.y[1]))
}

/// The curve y = g(x) across the chart at `n` points, each kept on it (see `onto`).
#[allow(clippy::too_many_arguments)]
fn curve_on(f: &mut Frame, p: &Plot, kind: u8, n: usize, width: f32, color: Color, g: impl Fn(f64) -> f64) {
    let (lo, hi, last) = (p.x[0], p.x[1], (n.max(2) - 1) as f64);
    p.path(f, kind, (0..n.max(2)).map(|i| lo + (hi - lo) * i as f64 / last).map(|x| onto(p, (x, g(x)))), width, color);
}

/// A dot in a batch of its own (see `plot`), for marks that come and go (the points
/// beyond the whiskers): in the batch of the sample's dots they would make it longer
/// or shorter from one frame to the next, and nothing in it could be followed then.
fn mark(f: &mut Frame, at: Vec2, size: f32, color: Color) {
    if at.is_finite() {
        let p = f.hud(at);
        f.dl.points_with(PointKind::Overlay, PointStyle { size_scale: 1.0001, ..Default::default() }).push(p, size, color);
    }
}

/// How wide a point's dot is in the strip (pixels): smaller the more there are.
fn dot_size(n: usize) -> f32 {
    (8.5 - 1.5 * (n.max(1) as f32).log10()).max(3.0).min(7.5)
}

/// How wide a point's dot is in the quantile plot.
fn qq_dot(n: usize) -> f32 {
    (6.0 - 0.9 * (n.max(1) as f32).log10()).max(2.6).min(5.0)
}

impl Scene for Describe {
    fn view(&self) -> View {
        View::Flat { center: Vec2::ZERO, half_height: 1.0 }
    }

    fn settings(&self, s: &mut ViewSettings) {
        s.grid = false;
        s.bloom = s.bloom.min(0.2);
    }

    fn animated(&self) -> bool {
        self.look.moving || self.running.is_some() || !self.errors.done(self.error_key(), self.source)
    }

    fn pointer_hint(&self) -> Option<&'static str> {
        Some("dots of the sample: drag sideways    white handles: drag    buttons over the picture: click    any picture: point at a value")
    }

    fn ui(&mut self, ui: &mut egui::Ui, _: &Gfx) {
        ui::note(ui, "One sample of numbers, and six ways of looking at it, all on one axis across: the points themselves; a histogram; a smoothed curve; a box plot; the cumulative steps; and a quantile plot against a bell. Point at a value in any of them and the point there lights up in all.");
        ui::note(ui, "The histogram and the smoothed curve each need a width, and the histogram a place for its bins to start: too wide hides what is there, too narrow invents what is not. Here the points are drawn from a known distribution (green), so each picture's error can be measured. The box, the steps and the quantile plot need no width, and each shows something else.");

        ui::section(ui, "Things to try");
        for (k, trial) in TRIALS.iter().enumerate() {
            if ui::experiment(ui, trial.button, trial.caption) {
                self.set_up(k);
            }
        }

        ui::section(ui, "The sample");
        let mut source = Source::ALL.iter().position(|s| *s == self.source).unwrap_or(0);
        let names: Vec<&str> = Source::ALL.iter().map(|s| s.name()).collect();
        if ui::choice(ui, &names, &mut source) {
            self.halt();
            self.source = Source::ALL[source];
            self.which = 0;
            self.mark = self.source.centre();
            self.redraw();
        }
        ui::note(ui, self.source.about());
        let mut n = self.n;
        if ui::slider_int_log(ui, "points", &mut n, 1..=2000) && n != self.n {
            self.halt();
            self.n = n;
            self.redraw();
        }
        ui.horizontal(|ui| {
            if ui.button(ui::label_text(ui, AGAIN, 12.5, None, false)).clicked() {
                self.halt();
                self.seed = self.seed.wrapping_add(1);
                self.redraw();
            }
            if self.source == Source::Pair && ui.button(ui::label_text(ui, OTHER_SET, 12.5, None, false)).clicked() {
                self.flow();
            }
        });

        ui::section(ui, "Histogram");
        let mut w = self.width as f32;
        if ui::slider_log(ui, "width of the bins", &mut w, WIDTHS.0 as f32..=WIDTHS.1 as f32) {
            self.halt();
            self.width = w as f64;
            self.auto_width = false;
            self.changed();
        }
        let mut share = ((self.mark / self.width).rem_euclid(1.0)) as f32;
        if ui::slider(ui, "where they start (share of a bin)", &mut share, 0.0..=1.0) {
            self.halt();
            self.mark = ((self.mark / self.width).floor() + share as f64) * self.width;
            self.changed();
        }
        if ui.add_enabled(!self.auto_width, egui::Button::new(ui::label_text(ui, "rule of thumb", 12.5, None, false))).clicked() {
            self.auto_width = true;
            self.changed();
        }
        ui::note(ui, r"A common rule of thumb (Freedman and Diaconis's): bins $2\,\mathrm{IQR} / \sqrt[3]{n}$ wide. Or drag the two white handles on the histogram: the lower one moves every edge, the upper one sets the width.");

        ui::section(ui, "Smoothed curve");
        let mut h = self.band as f32;
        if ui::slider_log(ui, r"bandwidth $h$", &mut h, BANDS.0 as f32..=BANDS.1 as f32) {
            self.halt();
            self.band = h as f64;
            self.auto_band = false;
            self.changed();
        }
        if ui.add_enabled(!self.auto_band, egui::Button::new(ui::label_text(ui, "rule of thumb", 12.5, None, false))).clicked() {
            self.auto_band = true;
            self.changed();
        }
        ui::note(ui, r"Silverman's rule of thumb (R's default): $h = 0.9 \min(s, \mathrm{IQR}/1.34)\, n^{-1/5}$, made for a sample from a bell. Or drag the white handle on the curve's chart.");
        self.refresh();

        ui::section(ui, "Reading the picture");
        ui::key(
            ui,
            &[
                (DATA, "cyan: the sample: its points, the bars of its histogram, its cumulative steps and its points in the quantile plot"),
                (ESTIMATE, r"yellow: the smoothed curve (and, faint, the bumps it is the mean of, at their true size), and the mean $\bar x$ with one standard deviation $s$ either side"),
                (OTHER, "orange: the median and the quartiles: the box, and the line through the quartiles in the quantile plot"),
                (
                    ALARM,
                    if self.insets {
                        "pink: points beyond the box's whiskers, marked singly; and each picture's error against its width (top right of the two density charts)"
                    } else {
                        "pink: points beyond the box's whiskers, marked singly"
                    },
                ),
                (TRUTH, "green: the distribution the points were drawn from: its density, its cumulative curve, and where its points would lie in the quantile plot"),
                (HAND, "white: the point under the pointer, in every picture; and the handles that set the widths"),
            ],
        );

        ui::section(ui, "Words");
        ui::term(ui, "Histogram:", "bars over bins of equal width, each as high as the share of the points in it divided by the width, so that the bars' areas add up to 1, like the area under a density.");
        ui::term(ui, "Bin:", "one bar's stretch of the axis. Where the bins start is a choice too: the same points and width with the edges moved can tell another story.");
        ui::term(ui, "Kernel density estimate:", r"the smoothed curve: the mean of one bump (here a bell) centred on each point, so each point adds area $1/n$.");
        ui::term(ui, "Bandwidth:", r"the spread $h$ of each bump: the smoothed curve's width.");
        ui::term(ui, "Error of a picture:", r"the squared gap between it and the true density, added up along the axis: $\int (\hat f - f)^2\,dx$. It can be measured only when the truth is known, as here.");
        ui::term(ui, "Quartiles:", "the values a quarter and three quarters of the way up the sorted points; the median is half way. (Between two points, here, in proportion; programs differ slightly in this.)");
        ui::term(ui, "Interquartile range (IQR):", "the upper quartile less the lower: the width of the middle half of the points.");
        ui::term(ui, "Box plot:", "a box from quartile to quartile with a line at the median, and whiskers out to the furthest points within one and a half box lengths; any point beyond is drawn singly.");
        ui::term(ui, "Outlier:", "a point far from the rest. The box plot's rule marks some as such; that is a rule of thumb, not proof that anything is wrong with them.");
        ui::term(ui, "Empirical cumulative distribution:", r"the cumulative steps: the share of the points at or below each value, rising by $1/n$ at each point.");
        ui::term(ui, "Normal score:", r"for the $i$-th smallest of $n$ points, the value a standard bell has a share $(i - \tfrac12)/n$ below.");
        ui::term(ui, "Quantile plot:", "each point at its value across and its normal score up. Points from a bell lie near a straight line; each kind of bend says how the sample differs from a bell.");
        ui::term(ui, r"Mean $\bar x$, standard deviation $s$:", r"the average; and the square root of the squared distances from it, added up and divided by $n - 1$ (about their average).");
        ui::term(ui, "Robust:", "hardly moved by a few wild values. The median and the quartiles are; the mean and the standard deviation are not.");

        ui::section(ui, "Numbers");
        let s = &self.stats;
        ui::readout(ui, "points", count(self.x.len()));
        ui::readout_colored(ui, r"mean $\bar x$", format!("${}$", num(s.mean, 3)), ESTIMATE);
        ui::readout_colored(ui, r"standard deviation $s$", if self.x.len() < 2 { "needs two points".to_string() } else { format!("${}$", num(s.sd, 3)) }, ESTIMATE);
        if let Some(f) = s.five {
            ui::readout_colored(ui, "median", format!("${}$", num(f.median, 3)), OTHER);
            ui::readout_colored(ui, "quartiles", format!("${}$ and ${}$", num(f.q1, 3), num(f.q3, 3)), OTHER);
            ui::readout_colored(ui, "interquartile range", format!("${}$", num(f.iqr(), 3)), OTHER);
            ui::readout_colored(ui, "beyond the whiskers", count(s.outliers), ALARM);
        }
        ui::readout(ui, "width of the bins", format!("${}$", num(self.width, 3)));
        ui::readout(ui, r"bandwidth $h$", format!("${}$", num(self.band, 3)));
        if self.source.truth() {
            let (b, h) = self.best();
            ui::readout_colored(ui, "histogram's error", format!("${}$", num(s.bars_error, 4)), ALARM);
            if let Some((w, e)) = b {
                ui::readout(ui, "  least, of the widths tried", format!("${}$ at ${}$", num(e, 4), num(w, 2)));
            }
            ui::readout_colored(ui, "smoothed curve's error", format!("${}$", num(s.smooth_error, 4)), ALARM);
            if let Some((w, e)) = h {
                ui::readout(ui, "  least, of the bandwidths tried", format!("${}$ at ${}$", num(e, 4), num(w, 2)));
            }
        }
        ui::see_also(
            ui,
            &[
                ("moments", "the same summaries of a whole distribution"),
                ("estimators", "how far a summary of a sample is from the truth"),
                ("clt", "the shape that averages take"),
                ("modelselect", "choosing a width is choosing how complex a model to fit"),
            ],
        );
    }

    fn frame(&mut self, f: &mut Frame) {
        let lay = Layout::of(f.view_size(), f.top_inset, self.tab);
        self.insets = lay.insets;
        let tiny = lay.view.x < 120.0 || lay.view.y < 100.0;
        if !tiny {
            self.buttons(f, &lay);
            self.handles(f, &lay);
        }
        if f.dt > 0.0 {
            self.advance(f.dt);
        }
        self.refresh();
        let key = self.error_key();
        if !self.errors.done(key, self.source) && self.held.is_none() && self.running.is_none() {
            let budget = crate::clock::Budget::new(0.0015);
            let (x, mark, source) = (&self.x, self.mark, self.source);
            self.errors.work(key, x, mark, source, || budget.spent());
        }
        self.follow(&lay, f.real_dt);
        if tiny {
            return;
        }
        self.draw(f, &lay);
    }
}

impl Describe {
    /// The pair's points flow to its other set.
    fn flow(&mut self) {
        if self.source != Source::Pair {
            return;
        }
        let from = match &self.running {
            Some(Running { script: Script::Flow { to }, .. }) => *to,
            _ => self.which,
        };
        self.halt();
        self.which = from;
        self.x = self.pair[from].clone();
        self.running = Some(Running { script: Script::Flow { to: 1 - from }, t: 0.0, wait: 0.0 });
        self.changed();
    }

    /// The buttons over the picture.
    fn buttons(&mut self, f: &mut Frame, lay: &Layout) {
        if !lay.buttons {
            return;
        }
        if let Some(tabs) = lay.tabs {
            for (k, r) in tabs.iter().enumerate() {
                if plot::button(f, *r, TABS[k], lay.text, self.tab == k) {
                    self.tab = k;
                }
            }
        }
        for (k, r) in lay.sources.iter().enumerate() {
            let source = Source::ALL[k];
            let name = if lay.narrow { source.short() } else { source.name() };
            if plot::button(f, *r, name, lay.small, self.source == source) && self.source != source {
                self.halt();
                self.source = source;
                self.which = 0;
                self.mark = source.centre();
                self.redraw();
            }
        }
        if plot::button(f, lay.again, AGAIN, lay.small, false) {
            self.halt();
            self.seed = self.seed.wrapping_add(1);
            self.redraw();
        }
        if self.source == Source::Pair && plot::button(f, lay.other, OTHER_SET, lay.small, false) {
            self.flow();
        }
    }

    /// The handles: the point nearest the pointer in the strip, the bins' edge and
    /// width, and the bandwidth.
    fn handles(&mut self, f: &mut Frame, lay: &Layout) {
        if !f.handles.dragging() {
            (self.held, self.point, self.frozen, self.frozen_tops) = (None, None, None, None);
        }
        let range = self.shown_range();
        // a point of the strip: the one held, or the one nearest the pointer
        if lay.shown[STRIP] {
            let plot = Plot::new(lay.charts[STRIP], range, [0.0, 1.0]);
            let dragging = f.handles.dragging();
            let target = self.point.or_else(|| {
                if dragging {
                    // (the one under the hand when it pressed, if it was a point)
                    return self.offered.filter(|_| self.held.is_none());
                }
                let p = plot::pointer(f)?;
                if !plot::within(p, plot.rect, 8.0) {
                    return None;
                }
                (0..self.look.dots.shown.len()).map(|i| (i, self.dot_at(&plot, i))).filter(|(_, at)| at.distance(p) < 12.0).min_by(|a, b| a.1.distance(p).total_cmp(&b.1.distance(p))).map(|(i, _)| i)
            });
            self.offered = target;
            if let Some(i) = target.filter(|i| *i < self.x.len()) {
                let at = self.dot_at(&plot, i);
                if let Some(to) = plot::screen_handle(f, POINT, at, DATA) {
                    self.grab(POINT, range);
                    self.point = Some(i);
                    let (lo, hi) = self.source.core();
                    let v = plot.x_at(to.x.max(plot.rect.0.x).min(plot.rect.1.x)).max(lo - 40.0).min(hi + 40.0);
                    if let Some(x) = self.x.get_mut(i) {
                        if *x != v {
                            *x = v;
                            self.changed();
                        }
                    }
                }
            }
        }
        let p = &self.look.params.shown;
        let (mark, width, band) = (p.first().copied().unwrap_or(0.0) as f64, p.get(1).copied().unwrap_or(1.0) as f64, p.get(2).copied().unwrap_or(1.0) as f64);
        let top = self.look.tops.get(0) as f64;
        if lay.shown[HIST] {
            let plot = Plot::new(lay.charts[HIST], range, [0.0, top.max(1e-3)]);
            if let Some((x, _)) = plot.handle(f, START, mark, 0.0, HAND) {
                self.grab(START, range);
                self.mark = x;
                self.changed();
            }
            if let Some((x, _)) = plot.handle(f, WIDTH, mark + width, top * lay.bracket, HAND) {
                self.grab(WIDTH, range);
                self.width = (x - mark).max(WIDTHS.0).min(WIDTHS.1);
                self.auto_width = false;
                self.changed();
            }
        }
        if lay.shown[SMOOTH] {
            let plot = Plot::new(lay.charts[SMOOTH], range, [0.0, top.max(1e-3)]);
            let centre = self.look.dots.shown.get(self.bump).copied().unwrap_or(0.0) as f64;
            if let Some((x, _)) = plot.handle(f, BAND, centre + band, top * lay.bracket, HAND) {
                self.grab(BAND, range);
                self.band = (x - centre).max(BANDS.0).min(BANDS.1);
                self.auto_band = false;
                self.changed();
            }
        }
    }

    /// A handle has been taken hold of: what an experiment set moving stops, and the
    /// axis stays put while it is held.
    fn grab(&mut self, id: u32, range: [f64; 2]) {
        if self.held != Some(id) {
            self.halt();
            self.held = Some(id);
            self.frozen = Some(range);
            self.frozen_tops = Some([self.look.tops.get(0), self.look.tops.get(1)]);
        }
    }

    fn shown_range(&self) -> [f64; 2] {
        let r = &self.look.range.shown;
        match (r.first(), r.get(1)) {
            (Some(a), Some(b)) => [*a as f64, *b as f64],
            _ => self.stats.range,
        }
    }

    /// Where point `i`'s dot is drawn in the strip.
    fn dot_at(&self, plot: &Plot, i: usize) -> Vec2 {
        let x = self.look.dots.get(i) as f64;
        let level = self.look.levels.get(i);
        let (pitch, d) = self.pitch(plot);
        Vec2::new(plot.px_x(x), plot.rect.1.y - d * 0.5 - 1.0 - level * pitch)
    }

    /// How far apart the strip's levels are, and how wide its dots: as far apart as
    /// the dots are wide, closer where the highest pile would not fit.
    fn pitch(&self, plot: &Plot) -> (f32, f32) {
        let d = dot_size(self.look.dots.shown.len());
        let highest = self.look.levels.shown.iter().copied().fold(0.0f32, f32::max);
        let room = (plot.height() - d - 2.0).max(1.0);
        (if highest * d * 0.92 > room { room / highest.max(1.0) } else { d * 0.92 }, d)
    }
}

// ----- drawing --------------------------------------------------------------------------

impl Describe {
    fn draw(&mut self, f: &mut Frame, lay: &Layout) {
        let range = self.shown_range();
        let top = (self.look.tops.get(0) as f64).max(1e-3);
        let ztop = (self.look.tops.get(1) as f64).max(0.5);
        let plots: [Plot; 6] = [
            Plot::new(lay.charts[STRIP], range, [0.0, 1.0]),
            Plot::new(lay.charts[HIST], range, [0.0, top]),
            Plot::new(lay.charts[SMOOTH], range, [0.0, top]),
            Plot::new(lay.charts[BOX], range, [0.0, 1.0]),
            Plot::new(lay.charts[STEPS], range, [0.0, 1.0]),
            Plot::new(lay.charts[QQ], range, [-ztop, ztop]),
        ];
        for c in 0..6 {
            if lay.shown[c] {
                let p = &plots[c];
                let left = if lay.narrow { 30.0 } else { 38.0 };
                p.backing(f, left, lay.small + 14.0, lay.big + 10.0, 6.0);
            }
        }
        let hover = self.hovered(f, lay, &plots);
        if let Some((i, _)) = hover {
            let d = &self.drawn;
            let v = self.look.dots.get(i) as f64;
            let (left, right, height) = d.bars.bar_of(v).map_or((v, v, 0.0), |k| (d.bars.edge(k), d.bars.edge(k + 1), self.look.heights.get(k) as f64));
            let share = d.sorted.partition_point(|s| *s <= v) as f64 / d.sorted.len().max(1) as f64;
            let z = d.rank.get(i).and_then(|r| d.scores.get(*r)).copied().unwrap_or(0.0);
            let goal = [left, right, height, share, z].map(|v| v as f32);
            if self.look.marked == Some(i) {
                self.look.moving |= self.look.marks.follow(&goal, f.real_dt, false);
            } else {
                self.look.marks.snap(&goal);
            }
        }
        self.look.marked = hover.map(|(i, _)| i);
        if lay.shown[STRIP] {
            self.draw_strip(f, lay, &plots[STRIP], hover);
        }
        if lay.shown[HIST] {
            self.draw_hist(f, lay, &plots[HIST], hover);
        }
        if lay.shown[SMOOTH] {
            self.draw_smooth(f, lay, &plots[SMOOTH], hover);
        }
        if lay.shown[BOX] {
            self.draw_box(f, lay, &plots[BOX], hover);
        }
        if lay.shown[STEPS] {
            self.draw_steps(f, lay, &plots[STEPS], hover);
        }
        if lay.shown[QQ] {
            self.draw_qq(f, lay, &plots[QQ], hover);
        }
        // the point under the pointer: a line through it in every picture
        if let Some((i, _)) = hover {
            let v = self.look.dots.get(i) as f64;
            for c in 0..6 {
                if lay.shown[c] {
                    plots[c].vline(f, HOVER_LINES, v, 1.0, HAND.a(0.35), true);
                }
            }
        }
        // new round values on an axis: make sure the axes' batch and the labels are not
        // as many as in the last frame, so that nothing takes the new ones for the old
        // ones moved
        let sign = |t: plot::Ticks| {
            let first = t.values().next().map_or(0, |v| (v / t.step).round() as i64);
            [t.step.to_bits() as i64, first, t.values().count() as i64]
        };
        let ticks: Vec<[i64; 3]> = (0..6)
            .filter(|c| lay.shown[*c])
            .flat_map(|c| {
                let p = &plots[c];
                [sign(plot::Ticks::of(p.x[0], p.x[1], p.width())), sign(plot::Ticks::of(p.y[0], p.y[1], p.height() * 1.6))]
            })
            .collect();
        let count = |f: &mut Frame| (plot::apart(f, AXES, false).pts.iter().filter(|p| p.width >= 0.0).count(), f.dl.labels.len());
        if ticks != self.ticks {
            let (axes, labels) = count(f);
            if axes == self.counted.0 {
                let far = f.hud(Vec2::ZERO);
                plot::apart(f, AXES, false).segment(far, far, 1.0, Color::CLEAR);
            }
            if labels == self.counted.1 {
                f.text_px(Vec2::ZERO, "", 1.0, Color::CLEAR, Align2::LEFT_TOP);
            }
        }
        self.ticks = ticks;
        self.counted = count(f);
    }

    /// The point under the pointer, if it is over a picture: its place in the sample,
    /// and the picture.
    fn hovered(&self, f: &Frame, lay: &Layout, plots: &[Plot; 6]) -> Option<(usize, usize)> {
        if let Some(i) = self.point {
            return Some((i, STRIP));
        }
        // (a hand on a handle of a width is not pointing at a value)
        if self.held.is_some() || f.handles.dragging() {
            return None;
        }
        let p = plot::pointer(f)?;
        let c = (0..6).find(|&c| lay.shown[c] && plot::within(p, plots[c].rect, 4.0))?;
        let d = &self.drawn;
        if d.sorted.is_empty() {
            return None;
        }
        if c == STRIP {
            // the nearest dot
            return (0..self.look.dots.shown.len()).map(|i| (i, self.dot_at(&plots[STRIP], i).distance(p))).filter(|(_, e)| *e < 14.0).min_by(|a, b| a.1.total_cmp(&b.1)).map(|(i, _)| (i, c));
        }
        let x = plots[c].x_at(p.x);
        let after = d.sorted.partition_point(|v| *v < x);
        let near = [after.saturating_sub(1), after.min(d.sorted.len() - 1)];
        let r = near.into_iter().min_by(|&a, &b| (d.sorted[a] - x).abs().total_cmp(&(d.sorted[b] - x).abs()))?;
        // (only a point close to the pointer across: within a dozen pixels)
        ((plots[c].px_x(d.sorted[r]) - p.x).abs() < 12.0).then(|| (d.order[r], c))
    }

    fn title(&self, f: &mut Frame, lay: &Layout, p: &Plot, text: &str) {
        let left = if lay.narrow { 30.0 } else { 38.0 };
        f.text_px(Vec2::new(p.rect.0.x - left + 4.0, p.rect.0.y - 7.0), text, lay.big, pal::TEXT, Align2::LEFT_BOTTOM);
    }

    /// A note about the point under the pointer, on a plate at the top left of the
    /// picture it is in.
    fn note(&self, f: &mut Frame, lay: &Layout, p: &Plot, text: &str) {
        f.caption_px(p.rect.0 + Vec2::new(6.0, 6.0), text, lay.small, pal::TEXT, Align2::LEFT_TOP);
    }

    fn draw_strip(&self, f: &mut Frame, lay: &Layout, p: &Plot, hover: Option<(usize, usize)>) {
        let n = self.x.len();
        let title = if lay.narrow { format!("the {} points: drag one", count(n)) } else { format!("the sample: {} points, a dot each (drag one sideways)", count(n)) };
        self.title(f, lay, p, &title);
        let (_, d) = self.pitch(p);
        for i in 0..self.look.dots.shown.len() {
            let at = self.dot_at(p, i);
            if hover.is_some_and(|(h, _)| h == i) {
                plot::disc(f, at, d + 5.0, HAND);
            }
            plot::disc(f, at, d, DATA);
        }
        p.x_axis(f, "", lay.small);
        if let Some((i, STRIP)) = hover {
            let r = self.drawn.rank.get(i).copied().unwrap_or(0);
            let v = self.look.dots.get(i) as f64;
            self.note(f, lay, p, &format!("${}$: the {} smallest of {}", num(v, 2), ordinal(r + 1), n));
        }
    }

    fn draw_hist(&self, f: &mut Frame, lay: &Layout, p: &Plot, hover: Option<(usize, usize)>) {
        let d = &self.drawn;
        let b = &d.bars;
        let truth = self.source.truth();
        let w = self.look.params.get(1) as f64;
        let mark = self.look.params.get(0) as f64;
        let (best, _) = self.best();
        let mut title = format!("histogram: bins ${}$ wide", num(self.width, 2));
        if truth && !lay.narrow {
            title += &format!(", error ${}$", num(self.stats.bars_error, 3));
            if let Some((bw, be)) = best {
                title += &format!(" (least ${}$, near ${}$)", num(be, 3), num(bw, 2));
            }
        } else if truth {
            title += &format!(", error ${}$", num(self.stats.bars_error, 3));
        }
        self.title(f, lay, p, &title);
        p.y_grid(f);
        // the bars, and their tops as a line
        let px = (p.px_x(b.first + b.width) - p.px_x(b.first)).abs();
        let gap = if px > 6.0 { 0.6 } else { 0.0 };
        let heights: Vec<f64> = (0..b.counts.len()).map(|i| self.look.heights.get(i) as f64).collect();
        p.bars(f, b.first, b.width, &heights, gap, DATA.a(0.42));
        // (their tops, bright: filled too, not lines, since there are as many as there are bars)
        for (i, h) in heights.iter().enumerate() {
            let (a, z) = ((p.px_x(b.edge(i)) + gap).max(p.rect.0.x), (p.px_x(b.edge(i + 1)) - gap).min(p.rect.1.x));
            let y = p.px_y(*h).max(p.rect.0.y);
            if z > a && *h > 0.0 {
                plot::fill(f, (Vec2::new(a, y), Vec2::new(z, y + 1.5)), DATA.a(0.95));
            }
        }
        if truth {
            let src = self.source;
            curve_on(f, p, CURVES, 241, 2.0, TRUTH, |x| src.pdf(x));
        }
        // the bar of the point under the pointer
        if let Some((i, c)) = hover {
            let v = self.look.dots.get(i) as f64;
            if let Some(k) = b.bar_of(v) {
                let m = |j: usize| self.look.marks.get(j) as f64;
                let (h, left, right, shown) = (b.height(k), m(0), m(1), m(2));
                p.path(f, HOVER_LINES, [(left, 0.0), (left, shown), (right, shown), (right, 0.0)].map(|q| onto(p, q)), 1.6, HAND);
                if c == HIST {
                    let note = format!("in the bin from ${}$ to ${}$: {} of {} points, so it is ${}$ high", num(b.edge(k), 2), num(b.edge(k + 1), 2), b.counts[k], b.n, num(h, 3));
                    self.note(f, lay, p, &note);
                }
            }
        }
        // the edge that moves them all, and the width
        let base = p.rect.1.y;
        let y = p.px_y(p.y[1] * lay.bracket);
        let (x0, x1) = (p.px_x(mark), p.px_x(mark + w));
        if x0 >= p.rect.0.x - 0.5 && x0 <= p.rect.1.x + 0.5 {
            plot::dashed_of(f, MARKS, Vec2::new(x0, base), Vec2::new(x0, y), 1.0, HAND.a(0.6));
            f.caption_px(Vec2::new(x0 - 10.0, base - 6.0), "start", lay.small, HAND, Align2::RIGHT_BOTTOM);
        }
        let (a, z) = (x0.max(p.rect.0.x), x1.min(p.rect.1.x));
        if z > a {
            plot::line_of(f, MARKS, Vec2::new(a, y), Vec2::new(z, y), 1.4, HAND);
            f.caption_px(Vec2::new(a - 10.0, y), format!("width ${}$", num(w, 2)), lay.small, HAND, Align2::RIGHT_CENTER);
        }
        if lay.insets && truth {
            self.draw_inset(f, lay, p, 0, w, WIDTHS, "error against width");
        }
        p.y_axis(f, "", lay.small);
        p.x_axis(f, "", lay.small);
    }

    fn draw_smooth(&self, f: &mut Frame, lay: &Layout, p: &Plot, hover: Option<(usize, usize)>) {
        let d = &self.drawn;
        let truth = self.source.truth();
        let h = self.look.params.get(2) as f64;
        let n = d.sorted.len();
        let (_, best) = self.best();
        let mut title = format!("smoothed: bumps of spread $h = {}$", num(self.band, 2));
        if truth && !lay.narrow {
            title += &format!(", error ${}$", num(self.stats.smooth_error, 3));
            if let Some((bw, be)) = best {
                title += &format!(" (least ${}$, near ${}$)", num(be, 3), num(bw, 2));
            }
        }
        self.title(f, lay, p, &title);
        p.y_grid(f);
        // a few of the bumps, faint, at their true size: up to a dozen, evenly through the points
        let centre = self.look.dots.get(self.bump) as f64;
        let picks = n.min(12);
        // (each bump over the part of the axis shown, always at as many points: a bump cut
        // off at the edge must not pass a point to the next one in the batch)
        let (lo, hi) = (p.x[0], p.x[1]);
        let around = move |c: f64| {
            let (a, b) = ((c - 4.0 * h).max(lo).min(hi), (c + 4.0 * h).max(lo).min(hi));
            (0..61).map(move |j| a + (b - a) * j as f64 / 60.0)
        };
        for k in 0..picks {
            let r = if picks > 1 { k * (n - 1) / (picks - 1) } else { 0 };
            let c = d.sorted[r];
            p.path(f, BUMP_LINES, around(c).map(|x| onto(p, (x, bump(x, c, h, n)))), 1.0, ESTIMATE.a(0.35));
        }
        p.path(f, MARKS, around(centre).map(|x| onto(p, (x, bump(x, centre, h, n)))), 1.4, ESTIMATE.a(0.8));
        let smooth = &d.smooth;
        curve_on(f, p, CURVES, CURVE_POINTS, 2.2, ESTIMATE, |x| smooth.at(x));
        if truth {
            let src = self.source;
            curve_on(f, p, CURVES, 241, 2.0, TRUTH, |x| src.pdf(x));
        }
        if let Some((i, c)) = hover {
            let v = self.look.dots.get(i) as f64;
            p.path(f, HOVER_LINES, around(v).map(|x| onto(p, (x, bump(x, v, h, n)))), 1.6, HAND);
            if c == SMOOTH {
                self.note(f, lay, p, &format!("its bump (white): a bell of spread ${}$ and area $1/{}$", num(h, 2), n));
            }
        }
        // the bandwidth, measured on one point's bump
        let y = p.px_y(p.y[1] * lay.bracket);
        let (x0, x1) = (p.px_x(centre), p.px_x(centre + h));
        let (a, z) = (x0.max(p.rect.0.x), x1.min(p.rect.1.x));
        if z > a {
            plot::dashed_of(f, MARKS, Vec2::new(x0.max(p.rect.0.x), p.px_y(bump(centre, centre, h, n))), Vec2::new(x0.max(p.rect.0.x), y), 1.0, HAND.a(0.6));
            plot::line_of(f, MARKS, Vec2::new(a, y), Vec2::new(z, y), 1.4, HAND);
            f.caption_px(Vec2::new(a - 10.0, y), format!("$h = {}$", num(h, 2)), lay.small, HAND, Align2::RIGHT_CENTER);
        }
        if lay.insets && truth {
            self.draw_inset(f, lay, p, 1, h, BANDS, "error against bandwidth");
        }
        p.y_axis(f, "", lay.small);
        p.x_axis(f, "", lay.small);
    }

    /// The curve of a picture's error against its width, in the top right corner of
    /// its chart, with where the width is now.
    #[allow(clippy::too_many_arguments)]
    fn draw_inset(&self, f: &mut Frame, lay: &Layout, p: &Plot, which: usize, now: f64, range: (f64, f64), name: &str) {
        let logs = &self.look.errors.shown;
        if logs.len() != 2 * TRIED {
            return;
        }
        let curve = &logs[which * TRIED..(which + 1) * TRIED];
        let (w, h) = (lay.inset.x, lay.inset.y);
        let max = Vec2::new(p.rect.1.x - 4.0, p.rect.0.y + 4.0 + h);
        let min = Vec2::new(max.x - w, p.rect.0.y + 4.0);
        hud_rect_at(f, 2, min, max, Color::hex(0x0B0F1A));
        let (lo, hi) = curve.iter().fold((f32::MAX, f32::MIN), |(a, b), v| (a.min(*v), b.max(*v)));
        let inner = (min + Vec2::new(8.0, lay.small + 8.0), max - Vec2::new(8.0, 6.0));
        let q = Plot::new(inner, [range.0.ln(), range.1.ln()], [lo as f64 - 0.1, hi as f64 + 0.1]);
        let stale = !self.errors.done(self.error_key(), self.source);
        q.path(f, CURVES, (0..TRIED).map(|k| (tried(range, k).ln(), curve[k] as f64)), 1.6, ALARM.a(if stale { 0.35 } else { 1.0 }));
        if let Some((k, _)) = least(&curve.iter().map(|v| *v as f64).collect::<Vec<_>>()) {
            q.dot(f, tried(range, k).ln(), curve[k] as f64, 6.0, ALARM);
        }
        q.vline(f, MARKS, now.max(range.0).min(range.1).ln(), 1.2, HAND, false);
        f.text_px(min + Vec2::new(6.0, 4.0), name, lay.small - 0.5, pal::DIM, Align2::LEFT_TOP);
        f.text_px(Vec2::new(max.x - 6.0, min.y + 4.0), "wider →", lay.small - 1.0, pal::DIM, Align2::RIGHT_TOP);
    }

    fn draw_box(&self, f: &mut Frame, lay: &Layout, p: &Plot, hover: Option<(usize, usize)>) {
        let d = &self.drawn;
        self.title(f, lay, p, if lay.narrow { r"box plot, and $\bar x \pm s$" } else { r"box plot (five numbers), and mean $\pm$ standard deviation" });
        let left = p.rect.0.x - 6.0;
        let glided = <[f32; 5]>::try_from(self.look.five.shown.as_slice()).ok().map(|a| a.map(|v| v as f64));
        if let (Some(five), Some([low, q1, median, q3, high])) = (d.five, glided) {
            // (the box drawn as glided; which points are beyond it, as they are)
            let (y0, y1) = (0.5, 0.92);
            let ym = 0.5 * (y0 + y1);
            let on = |q: (f64, f64)| {
                let (x, y) = onto(p, q);
                p.at(x, y)
            };
            let box_line = |f: &mut Frame, a: (f64, f64), b: (f64, f64), width: f32| plot::line_of(f, MARKS, on(a), on(b), width, OTHER);
            box_line(f, (q1, y0), (q3, y0), 1.6);
            box_line(f, (q3, y0), (q3, y1), 1.6);
            box_line(f, (q3, y1), (q1, y1), 1.6);
            box_line(f, (q1, y1), (q1, y0), 1.6);
            box_line(f, (median, y0), (median, y1), 2.6);
            box_line(f, (low, ym), (q1, ym), 1.4);
            box_line(f, (q3, ym), (high, ym), 1.4);
            box_line(f, (low, y0 + 0.1), (low, y1 - 0.1), 1.4);
            box_line(f, (high, y0 + 0.1), (high, y1 - 0.1), 1.4);
            for v in d.sorted.iter().filter(|v| five.beyond(**v)) {
                mark(f, p.at(*v, ym), 5.0, ALARM);
            }
            f.text_px(Vec2::new(left, p.px_y(ym)), "box", lay.small, OTHER, Align2::RIGHT_CENTER);
        }
        // the mean and one standard deviation either side
        let y = 0.22;
        let at = |x: f64, y: f64| {
            let (x, y) = onto(p, (x, y));
            p.at(x, y)
        };
        if d.sorted.len() >= 2 {
            plot::line_of(f, MARKS, at(d.mean - d.sd, y), at(d.mean + d.sd, y), 1.6, ESTIMATE);
            for e in [d.mean - d.sd, d.mean + d.sd] {
                plot::line_of(f, MARKS, at(e, y - 0.1), at(e, y + 0.1), 1.6, ESTIMATE);
            }
        }
        plot::disc(f, at(d.mean, y), 7.0, ESTIMATE);
        f.text_px(Vec2::new(left, p.px_y(y)), r"$\bar x \pm s$", lay.small, ESTIMATE, Align2::RIGHT_CENTER);
        if let Some((i, c)) = hover {
            let v = self.look.dots.get(i) as f64;
            p.dot(f, v, 0.71, 6.0, HAND);
            if c == BOX {
                if let Some(five) = d.five {
                    let where_ = if five.beyond(v) {
                        "beyond the whiskers: marked singly"
                    } else if v < five.q1 || v > five.q3 {
                        "outside the box, within a whisker"
                    } else {
                        "inside the box: the middle half"
                    };
                    self.note(f, lay, p, &format!("${}$: {}", num(v, 2), where_));
                }
            }
        }
        p.x_axis(f, "", lay.small);
    }

    fn draw_steps(&self, f: &mut Frame, lay: &Layout, p: &Plot, hover: Option<(usize, usize)>) {
        let d = &self.drawn;
        let n = d.sorted.len();
        self.title(f, lay, p, if lay.narrow { "cumulative: share at or below" } else { "cumulative steps: the share of the points at or below each value" });
        // a quarter, a half and three quarters, and where the box's numbers are
        for (k, share) in [0.25, 0.5, 0.75].into_iter().enumerate() {
            p.hline(f, MARKS, share, 1.0, OTHER.a(0.45), true);
            if let Some(five) = d.five {
                let x = [five.q1, five.median, five.q3][k];
                plot::dashed_of(f, MARKS, p.at(x, 0.0), p.at(x, share), 1.0, OTHER.a(0.7));
            }
        }
        if self.source.truth() {
            let src = self.source;
            curve_on(f, p, CURVES, 241, 2.0, TRUTH, |x| src.cdf(x));
        }
        let mut pts = Vec::with_capacity(2 * n + 2);
        pts.push((p.x[0], 0.0));
        for (i, v) in d.sorted.iter().enumerate() {
            pts.push((*v, i as f64 / n as f64));
            pts.push((*v, (i + 1) as f64 / n as f64));
        }
        pts.push((p.x[1], 1.0));
        p.path(f, STEP_LINES, pts.into_iter().map(|q| onto(p, q)), 1.8, DATA);
        if let Some((i, c)) = hover {
            let v = self.look.dots.get(i) as f64;
            let at_or_below = d.sorted.partition_point(|s| *s <= v);
            let share = at_or_below as f64 / n.max(1) as f64;
            p.dot(f, v, self.look.marks.get(3) as f64, 7.0, HAND);
            if c == STEPS {
                self.note(f, lay, p, &format!("{} of {} points at or below ${}$: the steps are at ${}$ there", at_or_below, n, num(v, 2), num(share, 3)));
            }
        }
        // the share up the side, in quarters
        for share in [0.0, 0.25, 0.5, 0.75, 1.0] {
            let y = p.px_y(share);
            plot::line_of(f, AXES, Vec2::new(p.rect.0.x - 4.0, y), Vec2::new(p.rect.0.x, y), 1.0, pal::DIM.a(0.8));
            f.text_px(Vec2::new(p.rect.0.x - 6.0, y), num(share, 2), lay.small, pal::DIM, Align2::RIGHT_CENTER);
        }
        plot::line_of(f, AXES, p.rect.0, Vec2::new(p.rect.0.x, p.rect.1.y), 1.0, pal::DIM.a(0.8));
        p.x_axis(f, "", lay.small);
    }

    /// Where the quantile plot's true curve is on the chart: from where it comes in to
    /// where it goes out (it is drawn at as many points whatever that is, so that a
    /// point cut off at one end is not taken for one moved from the other).
    fn qq_truth_span(&self, p: &Plot) -> Option<(f64, f64)> {
        if !self.source.truth() {
            return None;
        }
        let (src, top) = (self.source, p.y[1]);
        let (lo, hi) = (p.x[0], p.x[1]);
        // the least x in lo..hi with the curve at or above `z`, by halving
        let from = |z: f64| {
            let share = st::normal_cdf(z);
            if src.cdf(lo) >= share {
                return lo;
            }
            let (mut a, mut b) = (lo, hi);
            for _ in 0..60 {
                let m = 0.5 * (a + b);
                if src.cdf(m) >= share {
                    b = m;
                } else {
                    a = m;
                }
            }
            b
        };
        let (a, b) = (from(-top), from(top));
        (b > a && src.cdf(lo) < st::normal_cdf(top)).then_some((a, b))
    }

    fn draw_qq(&self, f: &mut Frame, lay: &Layout, p: &Plot, hover: Option<(usize, usize)>) {
        let d = &self.drawn;
        let n = d.sorted.len();
        self.title(f, lay, p, if lay.narrow { "quantile plot: normal score up" } else { "quantile plot: each point at its value across and its normal score up" });
        p.hline(f, MARKS, 0.0, 1.0, pal::DIM.a(0.3), false);
        if let Some(five) = d.five.filter(|f| f.iqr() > 0.0) {
            // (from where it comes onto the chart to where it leaves, at as many points
            // whatever that is: see `onto`)
            let top = p.y[1];
            let (a, b) = ((five.q1 + 0.5 * five.iqr() * (1.0 - top / quartile_score())).max(p.x[0]), (five.q1 + 0.5 * five.iqr() * (1.0 + top / quartile_score())).min(p.x[1]));
            if b > a {
                p.path(f, MARKS, (0..81).map(|k| a + (b - a) * k as f64 / 80.0).map(|x| onto(p, (x, five.line(x)))), 1.4, OTHER.a(0.85));
            }
        }
        if let Some((a, b)) = self.qq_truth_span(p) {
            let (src, top) = (self.source, p.y[1] * (1.0 - 1e-9));
            p.path(f, CURVES, (0..241).map(|k| a + (b - a) * k as f64 / 240.0).map(|x| (x, st::normal_quantile(src.cdf(x)).max(-top).min(top))), 2.0, TRUTH);
        }
        let size = qq_dot(n);
        for (r, v) in d.sorted.iter().enumerate() {
            let (x, z) = onto(p, (*v, d.scores.get(r).copied().unwrap_or(0.0)));
            plot::disc(f, p.at(x, z), size, DATA);
        }
        if let Some((i, c)) = hover {
            let r = d.rank.get(i).copied().unwrap_or(0);
            let v = self.look.dots.get(i) as f64;
            let z = d.scores.get(r).copied().unwrap_or(0.0);
            let shown = self.look.marks.get(4) as f64;
            p.dot(f, v, shown, size + 5.0, HAND);
            p.dot(f, v, shown, size, DATA);
            if c == QQ {
                let share = (r as f64 + 0.5) / n.max(1) as f64;
                self.note(f, lay, p, &format!("the {} smallest of {}: up at ${}$, where a bell has ${}$ % below", ordinal(r + 1), n, num(z, 2), num(100.0 * share, 1)));
            }
        }
        p.y_axis(f, "", lay.small);
        p.x_axis(f, "", lay.small);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A scene set up as experiment `k` and run to the end of what it sets moving,
    /// with its errors' curves worked out.
    fn after(k: usize) -> Describe {
        let mut s = Describe::create();
        s.set_up(k);
        s.finish();
        s.work_out_errors();
        s
    }

    impl Source {
        /// The density at 601 places from `lo` to `hi`.
        fn pdf_row(self, lo: f64, hi: f64) -> Vec<f64> {
            (0..=600).map(|i| self.pdf(lo + (hi - lo) * i as f64 / 600.0)).collect()
        }
    }

    impl Describe {
        fn work_out_errors(&mut self) {
            let key = self.error_key();
            let x = self.x.clone();
            self.errors.work(key, &x, self.mark, self.source, || false);
        }

        fn smooth_now(&self) -> Smooth {
            let (lo, hi) = extent(&self.x);
            Smooth::of(&self.x, self.band, lo, hi)
        }
    }

    /// A caption says this number, written to these places as the scene writes it.
    fn says(caption: &str, v: f64, decimals: usize) {
        let text = num(v, decimals);
        let written = if v < 0.0 && text != num(0.0, decimals) { format!("${}$", text.replace('−', "-")) } else { text.clone() };
        assert!(caption.contains(&written), "the caption does not say {written} ({v}):\n{caption}");
    }

    /// The slopes of the quantile plot's bottom fifth, middle fifth and top fifth of
    /// points, each as a share of the slope of the line through the quartiles.
    fn slopes(x: &[f64]) -> [f64; 3] {
        let s = st::sorted(x);
        let n = s.len();
        let five = Five::of(&s).expect("points");
        let line = 2.0 * quartile_score() / five.iqr();
        let fit = |a: usize, b: usize| {
            let zs: Vec<f64> = (a..b).map(|i| normal_score(i, n)).collect();
            st::covariance(&s[a..b], &zs) / st::variance(&s[a..b]) / line
        };
        [fit(0, n / 5), fit(2 * n / 5, 3 * n / 5), fit(4 * n / 5, n)]
    }

    /// Every way of drawing a histogram here puts each point in one bar, and the bars'
    /// areas add up to 1: with any width (wider than all the points too), any start,
    /// any number of points.
    #[test]
    fn the_bars_areas_add_up_to_one() {
        for source in Source::ALL {
            for n in [1, 2, 10, 200, 2000] {
                let x = draw(source, n, 7);
                for width in [0.02, 0.3, 1.2, 4.0, 50.0] {
                    for mark in [0.0, 0.37, -5.0, 11.0] {
                        let b = bars(&x, mark, width);
                        let area: f64 = (0..b.counts.len()).map(|i| b.height(i) * b.width).sum();
                        assert!((area - 1.0).abs() < 1e-9, "{source:?} {n} points, width {width}, mark {mark}: area {area}");
                        assert_eq!(b.counts.iter().sum::<u32>() as usize, x.len());
                        // every point in the bar it says
                        for v in &x {
                            let i = b.bar_of(*v).expect("in a bar");
                            assert!(*v >= b.edge(i) - 1e-9 && *v < b.edge(i + 1) + 1e-9);
                        }
                        // and an edge where it was put
                        let k = ((mark - b.first) / width).round();
                        assert!((b.first + k * width - mark).abs() < 1e-9 * (1.0 + mark.abs() + width * k.abs()));
                    }
                }
                // a width larger than the points' whole range: one bar of height 1/width
                let (lo, hi) = extent(&x);
                let b = bars(&x, lo, 2.0 * (hi - lo) + 1.0);
                assert_eq!(b.counts, vec![x.len() as u32]);
            }
        }
    }

    /// The smoothed curve is the mean of the points' bumps: within one percent of its
    /// peak of the exact sum everywhere, and its area is 1.
    #[test]
    fn the_smoothed_curve_is_the_mean_of_its_bumps_and_its_area_is_1() {
        for source in [Source::Bell, Source::Tail, Source::Humps, Source::Wild] {
            for n in [1, 5, 200] {
                let x = draw(source, n, 3);
                let (lo, hi) = extent(&x);
                for h in [0.02, 0.05, 0.3, 1.0, 2.0] {
                    let s = Smooth::of(&x, h, lo, hi);
                    assert!((s.area() - 1.0).abs() < 1e-3, "{source:?} {n} points, h {h}: area {}", s.area());
                    let peak = s.tallest();
                    for k in 0..2000 {
                        let at = lo - 3.0 * h + (hi - lo + 6.0 * h) * k as f64 / 1999.0;
                        let exact: f64 = x.iter().map(|v| bump(at, *v, h, n)).sum();
                        assert!((s.at(at) - exact).abs() < 1e-2 * peak, "{source:?} {n} points, h {h}, at {at}: {} against {exact}, peak {peak}", s.at(at));
                    }
                }
            }
        }
        assert_eq!(Smooth::of(&[], 0.3, 0.0, 1.0), Smooth::default());
        assert!(Smooth::of(&[1.0], 0.0, 0.0, 1.0).values.is_empty());
    }

    /// The errors are the squared gaps they say they are: the bars' exactly, the curve's
    /// against the gap worked out point by point from its bumps, within a percent.
    #[test]
    fn the_errors_are_the_squared_gaps() {
        for source in [Source::Bell, Source::Tail, Source::Humps, Source::Wild] {
            let x = draw(source, 200, 5);
            let (lo, hi) = extent(&x);
            let (a, b, m) = (lo - 30.0, hi + 30.0, 240_000);
            let dx = (b - a) / m as f64;
            for width in [0.05, 0.4, 2.0] {
                let bs = bars(&x, 0.1, width);
                let numeric: f64 = (0..m)
                    .map(|i| {
                        let at = a + (i as f64 + 0.5) * dx;
                        let h = bs.bar_of(at).map_or(0.0, |k| bs.height(k));
                        (h - source.pdf(at)).powi(2)
                    })
                    .sum::<f64>()
                    * dx;
                assert!((bs.error(source) - numeric).abs() < 1e-3 * numeric + 1e-6, "{source:?} bars {width}: {} against {numeric}", bs.error(source));
            }
            for h in [0.05, 0.3, 1.0] {
                let numeric: f64 = (0..m / 20)
                    .map(|i| {
                        let at = a + (i as f64 + 0.5) * dx * 20.0;
                        let f: f64 = x.iter().map(|v| bump(at, *v, h, x.len())).sum();
                        (f - source.pdf(at)).powi(2)
                    })
                    .sum::<f64>()
                    * dx
                    * 20.0;
                let e = Smooth::of(&x, h, lo, hi).error(source);
                assert!((e - numeric).abs() < 0.01 * numeric, "{source:?} smoothed {h}: {e} against {numeric}");
            }
            // the closed form of ∫ f², against the sum
            let numeric: f64 = (0..m).map(|i| source.pdf(a + (i as f64 + 0.5) * dx).powi(2)).sum::<f64>() * dx;
            assert!((source.square_integral() - numeric).abs() < 1e-6, "{source:?}: {} against {numeric}", source.square_integral());
        }
    }

    /// The box's numbers are the sample's quartiles and median by `quantile_sorted`;
    /// its whiskers reach the furthest points within one and a half box lengths, and
    /// what is marked singly is beyond them.
    #[test]
    fn the_box_is_the_quartiles_and_the_whiskers_reach_as_far_as_they_may() {
        for source in [Source::Bell, Source::Tail, Source::Humps, Source::Wild] {
            for n in [1, 2, 3, 4, 10, 40, 300] {
                let s = st::sorted(&draw(source, n, 11));
                let f = Five::of(&s).expect("points");
                assert_eq!((f.q1, f.median, f.q3), (st::quantile_sorted(&s, 0.25), st::quantile_sorted(&s, 0.5), st::quantile_sorted(&s, 0.75)));
                let (lo, hi) = (f.q1 - 1.5 * f.iqr(), f.q3 + 1.5 * f.iqr());
                assert!(f.low >= lo && f.high <= hi && f.low <= f.q1 && f.high >= f.q3, "{source:?} {n}: {f:?}");
                assert!((s.contains(&f.low) || f.low == f.q1) && (s.contains(&f.high) || f.high == f.q3));
                // nothing between the whiskers' ends and the reach
                assert!(s.iter().all(|v| !((*v > f.high && *v <= hi) || (*v < f.low && *v >= lo))));
                for v in &s {
                    assert_eq!(f.beyond(*v), *v < lo || *v > hi, "{source:?} {n}: {v} and {f:?}");
                }
            }
        }
        assert!(Five::of(&[]).is_none());
    }

    /// For a large sample from a bell, the quantile plot's points lie on the line
    /// through the quartiles: the middle 98 % of them within 0.2 of it (normal score),
    /// over six samples of 2000. (The furthest were 0.11 to 0.18 off; the very ends
    /// stray further, up to 0.7, as the extremes of any sample do.)
    #[test]
    fn a_large_bell_lies_on_the_line() {
        let mut worst: f64 = 0.0;
        for seed in 1..=6u64 {
            let s = st::sorted(&draw(Source::Bell, 2000, seed));
            let five = Five::of(&s).expect("points");
            let n = s.len();
            for i in n / 100..n - n / 100 {
                worst = worst.max((normal_score(i, n) - five.line(s[i])).abs());
            }
        }
        assert!(worst < 0.2 && worst > 0.1, "furthest {worst}");
    }

    /// The two sets of the pair have exactly the same five numbers, and so the same box,
    /// at every size and seed; the first has one hump and the second two (each smoothed
    /// with a bandwidth of 0.3); and while one flows into the other, every set between
    /// has those five numbers too.
    #[test]
    fn the_pair_has_one_box_and_two_shapes() {
        for seed in 1..=8u64 {
            for n in [5, 9, 101, 201, 2000] {
                let [a, b] = pair(n, seed);
                assert_eq!(a.len(), b.len());
                assert_eq!(a.len() % 4, 1);
                let (fa, fb) = (Five::of(&a).expect("points"), Five::of(&b).expect("points"));
                assert_eq!(fa, fb, "{n} points, seed {seed}");
                assert_eq!([fa.low, fa.q1, fa.median, fa.q3, fa.high], FIVE);
                assert!(a.windows(2).all(|w| w[0] <= w[1]) && b.windows(2).all(|w| w[0] <= w[1]));
            }
        }
        let [a, b] = pair(201, PAIR_SEED);
        let (sa, sb) = (Smooth::of(&a, 0.3, -4.0, 4.0), Smooth::of(&b, 0.3, -4.0, 4.0));
        assert_eq!((peaks(&sa.values), peaks(&sb.values)), (1, 2));
        assert!(sa.at(0.0) > 1.5 * sa.at(1.0) && sb.at(1.0) > 2.0 * sb.at(0.0), "{} {} {} {}", sa.at(0.0), sa.at(1.0), sb.at(0.0), sb.at(1.0));
        // the flow, as the experiment runs it
        let mut s = Describe::create();
        s.set_up(3);
        let lay = Layout::of(Vec2::new(1075.0, 1000.0), 0.0, 0);
        let mut boxes = 0;
        for _ in 0..400 {
            s.advance(1.0 / 60.0);
            s.refresh();
            s.follow(&lay, 1.0 / 60.0);
            let shown = s.drawn.five.expect("points");
            assert_eq!([shown.low, shown.q1, shown.median, shown.q3, shown.high], FIVE.map(|v| v as f32 as f64));
            assert_eq!(s.stats.five.map(|f| [f.low, f.q1, f.median, f.q3, f.high]), Some(FIVE));
            boxes += 1;
        }
        assert!(boxes == 400 && s.running.is_none() && s.which == 1);
        assert_eq!(st::sorted(&s.x), b);
    }

    /// The first two experiments: too wide and too narrow, with the numbers their
    /// captions give.
    #[test]
    fn too_wide_hides_and_too_narrow_invents() {
        let s = after(0);
        let c = TRIALS[0].caption;
        let b = bars(&s.x, s.mark, s.width);
        let heights: Vec<f64> = (0..b.counts.len()).map(|i| b.height(i)).collect();
        assert_eq!(peaks(&heights), 1, "{:?}", b.counts);
        assert_eq!(peaks(&Source::Humps.pdf_row(-3.0, 3.0)), 2);
        says(c, s.stats.bars_error, 2);
        let (best_bars, best_band) = s.best();
        let (bw, be) = best_bars.expect("done");
        says(c, be, 3);
        says(c, bw, 2);
        says(c, s.band, 2);
        let (hw, he) = best_band.expect("done");
        says(c, hw, 2);
        assert!((s.band / hw - 2.0).abs() < 0.1, "{} against {hw}", s.band);
        says(c, s.stats.smooth_error, 3);
        says(c, he, 3);
        let dip = s.smooth_now().at(0.0) / Source::Humps.pdf(0.0);
        assert!(dip > 2.0 && dip < 2.4, "{dip}");

        let s = after(1);
        let c = TRIALS[1].caption;
        let b = bars(&s.x, s.mark, s.width);
        let heights: Vec<f64> = (0..b.counts.len()).map(|i| b.height(i)).collect();
        assert!(c.contains(&format!("{} peaks where", peaks(&heights))), "{}", peaks(&heights));
        assert!(c.contains(&format!("0.03: {} peaks", peaks(&s.smooth_now().values))), "{}", peaks(&s.smooth_now().values));
        says(c, s.stats.bars_error, 3);
        says(c, s.stats.smooth_error, 3);
        let (best_bars, best_band) = s.best();
        let ((bw, be), (hw, he)) = (best_bars.expect("done"), best_band.expect("done"));
        for (v, d) in [(bw, 2), (be, 3), (hw, 2), (he, 3)] {
            says(c, v, d);
        }
    }

    /// The histogram with the bins started at 0 and half a bin on, and the sweep from
    /// the one to the other and on to the first again.
    #[test]
    fn moving_the_bins_makes_and_unmakes_two_humps() {
        let mut s = Describe::create();
        s.set_up(2);
        let c = TRIALS[2].caption;
        let flat = bars(&s.x, 0.0, 1.2);
        assert_eq!(flat.counts, vec![1, 44, 43, 54, 57, 1]);
        assert!(c.contains("four bars of 43 to 57 points"));
        let humps = bars(&s.x, 0.6, 1.2);
        assert_eq!(humps.counts, vec![9, 72, 19, 90, 10]);
        assert!(c.contains("19 points and its neighbours 72 and 90"));
        // half way through the sweep, half a bin on; at the end, one bin on
        assert_eq!(s.mark, 0.0);
        s.advance(0.8 + 3.0);
        assert!((s.mark - 0.6).abs() < 1e-9, "{}", s.mark);
        s.finish();
        assert!((s.mark - 1.2).abs() < 1e-9 && bars(&s.x, s.mark, s.width).counts == flat.counts);
        assert!(c.contains("six seconds") && (Script::Sweep { from: 0.0 }).length() == 6.0);
    }

    /// The pair's experiment: the same five numbers in the caption as in the sets.
    #[test]
    fn the_box_cannot_show_the_humps() {
        let s = after(3);
        let c = TRIALS[3].caption;
        assert!(c.contains(&format!("{} points", s.x.len())));
        for v in FIVE {
            assert!(c.contains(&format!("${}$", kit_num(v))), "{v}");
        }
        let [a, b] = pair(201, PAIR_SEED);
        says(c, st::sd(&a), 2);
        says(c, st::sd(&b), 2);
        assert!((s.stats.sd - st::sd(&b)).abs() < 1e-12);
    }

    fn kit_num(v: f64) -> String {
        let text = if v == v.round() { num(v, 0) } else { num(v, 1) };
        text.replace('−', "-")
    }

    /// The quantile plot's bends, for the long right tail and for each other source at
    /// the same size and seed (chosen with the buttons, as the caption says).
    #[test]
    fn straight_means_a_bell() {
        let mut s = after(4);
        let [bottom, middle, top] = slopes(&s.x);
        // steep on the left, flat on the right
        assert!(bottom > 2.5 && top < 0.45 && (middle - 1.0).abs() < 0.1, "tail: {bottom} {middle} {top}");
        let mut seen = Vec::new();
        for source in [Source::Bell, Source::Humps, Source::Wild] {
            s.source = source;
            s.redraw();
            s.refresh();
            assert_eq!(s.x.len(), 300);
            seen.push((source, slopes(&s.x)));
        }
        let [b, m, t] = seen[0].1;
        assert!([b, m, t].iter().all(|v| (v - 1.0).abs() < 0.15), "bell: {b} {m} {t}");
        // and the bell's points near the line, the middle nine tenths within 0.25 of it
        // (normal score), the ends less near (0.22 and 0.44 at this seed)
        let sorted = st::sorted(&draw(Source::Bell, 300, TAIL_SEED));
        let five = Five::of(&sorted).expect("points");
        let off = |i: usize| (normal_score(i, 300) - five.line(sorted[i])).abs();
        let (middle, all) = ((15..285).map(off).fold(0.0, f64::max), (0..300).map(off).fold(0.0, f64::max));
        assert!(middle < 0.25 && all > 1.5 * middle && all < 0.5, "{middle} {all}");
        let [b, m, t] = seen[1].1;
        assert!(m < 0.55 && b > 2.5 && t > 2.5, "humps: {b} {m} {t}");
        let [b, m, t] = seen[2].1;
        assert!(b < 0.4 && t < 0.5 && m > 0.85, "wild: {b} {m} {t}");
    }

    /// One wild value: the numbers before and after, and a drag further out.
    #[test]
    fn one_wild_value_moves_the_mean_and_not_the_median() {
        let mut s = Describe::create();
        s.set_up(5);
        let c = TRIALS[5].caption;
        let before = (s.stats.mean, s.stats.sd, s.stats.five.expect("points"));
        s.finish();
        let after = (s.stats.mean, s.stats.sd, s.stats.five.expect("points"));
        for (v, d) in [(before.0, 2), (after.0, 2), (before.1, 2), (after.1, 2), (before.2.median, 2), (after.2.median, 2), (before.2.iqr(), 2), (after.2.iqr(), 2)] {
            says(c, v, d);
        }
        assert_eq!(s.x.len(), 40);
        assert!(c.contains("40 points") && c.contains("a fortieth") && c.contains("the 20th from the bottom"));
        // the one that flew was the 20th from the bottom
        let sorted_before = st::sorted(&draw(Source::Bell, 40, WILD_SEED));
        assert!(st::sorted(&s.x).iter().filter(|v| **v != 7.0).zip(sorted_before.iter().enumerate().filter(|(i, _)| *i != 19).map(|(_, v)| v)).all(|(a, b)| a == b));
        let index = s.x.iter().position(|v| *v == 7.0).expect("it flew to 7");
        assert_eq!(TRIALS[5].act, Act::Fly(7.0));
        // further out: the mean a fortieth of the way, the median and quartiles not at all
        s.x[index] = 12.0;
        s.changed();
        s.refresh();
        assert!((s.stats.mean - after.0 - 5.0 / 40.0).abs() < 1e-12 && s.stats.sd > after.1 + 0.5);
        let further = s.stats.five.expect("points");
        assert_eq!((further.q1, further.median, further.q3), (after.2.q1, after.2.median, after.2.q3));
    }

    /// Degenerate samples: one point, two equal ones, all on one value, the widest
    /// bins and narrowest bandwidth. Everything worked out and drawn from them is a
    /// number, and says so honestly where there is nothing to say.
    #[test]
    fn degenerate_samples_stay_finite() {
        let lay = Layout::of(Vec2::new(1075.0, 1000.0), 0.0, 0);
        for (name, x) in [("one point", vec![0.5]), ("two equal points", vec![1.0, 1.0]), ("all on one value", vec![-2.0; 50])] {
            for (width, band) in [(WIDTHS.0, BANDS.0), (WIDTHS.1, BANDS.1)] {
                let mut s = Describe::create();
                s.x = x.clone();
                s.n = x.len();
                (s.width, s.band) = (width, band);
                s.changed();
                s.refresh();
                // no spread: the rules of thumb have nothing to go on, and the widths stay
                let sorted = st::sorted(&s.x);
                assert!(bin_rule(&sorted).is_none() && band_rule(&sorted).is_none(), "{name}");
                assert_eq!((s.width, s.band), (width, band), "{name}");
                assert!(s.stats.sd == 0.0 && s.stats.five.is_some_and(|f| f.iqr() == 0.0 && f.line(0.0).is_nan()), "{name}");
                for _ in 0..200 {
                    s.follow(&lay, 1.0 / 60.0);
                }
                let d = &s.drawn;
                assert!(d.sorted.iter().all(|v| v.is_finite()) && d.smooth.values.iter().all(|v| v.is_finite()), "{name}");
                assert!(s.stats.range.iter().all(|v| v.is_finite()) && s.stats.range[1] > s.stats.range[0]);
                assert!(s.stats.bars_error.is_finite() && s.stats.smooth_error.is_finite() && s.stats.tallest.is_finite());
                assert!(s.look.tops.shown.iter().all(|v| v.is_finite() && *v > 0.0));
                assert!(d.scores.iter().all(|v| v.is_finite()));
                let b = &d.bars;
                assert!((0..b.counts.len()).all(|i| b.height(i).is_finite()));
            }
        }
    }

    /// The layout fits every window, from a large desktop to a phone and below.
    #[test]
    fn the_layout_fits_every_window() {
        for (w, h, inset) in [(1075.0, 1000.0, 0.0), (1400.0, 760.0, 0.0), (575.0, 700.0, 0.0), (390.0, 756.0, 44.0), (700.0, 500.0, 0.0), (1075.0, 420.0, 0.0), (240.0, 180.0, 0.0), (1.0, 1.0, 0.0), (3000.0, 2000.0, 0.0)] {
            for tab in [0, 1] {
                let lay = Layout::of(Vec2::new(w, h), inset, tab);
                let all = lay.charts.iter().chain(&lay.sources).chain([&lay.again, &lay.other]).chain(lay.tabs.iter().flatten());
                assert!(all.clone().all(|(a, b)| a.is_finite() && b.is_finite()), "{w}x{h}");
                for c in 0..6 {
                    let (a, b) = lay.charts[c];
                    if lay.shown[c] {
                        assert!(a.x >= 0.0 && a.y >= inset && b.x <= w + 0.5 && b.y <= h + 0.5 && b.x > a.x && b.y > a.y, "{w}x{h}: chart {c} at {a} {b}");
                    }
                }
                // no two pictures on top of each other
                for c in 0..6 {
                    for d in c + 1..6 {
                        if lay.shown[c] && lay.shown[d] {
                            let (a, b) = (lay.charts[c], lay.charts[d]);
                            assert!(a.1.x < b.0.x || b.1.x < a.0.x || a.1.y < b.0.y || b.1.y < a.0.y, "{w}x{h}: {c} and {d}");
                        }
                    }
                }
                if lay.buttons {
                    let last = if lay.narrow { lay.sources[4].1.y } else { lay.again.1.y };
                    assert!(lay.charts.iter().zip(lay.shown).filter(|(_, s)| *s).all(|(r, _)| r.0.y > last + 20.0), "{w}x{h}");
                    assert!(lay.sources.iter().chain([&lay.again]).all(|(_, b)| b.x <= w), "{w}x{h}: the buttons run off");
                }
                if w >= 390.0 && h >= 700.0 {
                    // on a phone and on a desktop: the points, the box and two more at least
                    assert_eq!(lay.shown.iter().filter(|s| **s).count(), if lay.narrow { 4 } else { 6 }, "{w}x{h}");
                    assert!(lay.charts[STRIP].1.y - lay.charts[STRIP].0.y >= 40.0);
                    let tall = if lay.narrow { TAB_CHARTS[tab][0] } else { HIST };
                    assert!(lay.charts[tall].1.y - lay.charts[tall].0.y > 130.0, "{w}x{h}");
                }
            }
        }
    }

    /// After a change, the picture glides to the new one and comes to rest exactly
    /// there, at any frame rate.
    #[test]
    fn glides_arrive_and_stop() {
        let lay = Layout::of(Vec2::new(1075.0, 1000.0), 0.0, 0);
        for hz in [30.0f32, 60.0, 144.0] {
            let mut s = Describe::create();
            for _ in 0..10 {
                s.follow(&lay, 1.0 / hz);
            }
            assert!(!s.look.moving);
            s.set_up(0);
            let mut frames = 0;
            loop {
                s.follow(&lay, 1.0 / hz);
                frames += 1;
                if !s.look.moving {
                    break;
                }
                assert!((frames as f32) < 2.0 * hz, "still gliding after two seconds at {hz} Hz");
            }
            assert!(frames as f32 > 0.15 * hz, "arrived in {frames} frames at {hz} Hz: no glide");
            assert_eq!(s.look.params.shown, vec![s.mark as f32, s.width as f32, s.band as f32]);
            assert_eq!(s.look.dots.shown, s.x.iter().map(|v| *v as f32).collect::<Vec<_>>());
        }
    }

    /// The errors' curves are worked out a width at a time when asked to stop at once,
    /// and are kept while only a width changes.
    #[test]
    fn the_errors_come_in_small_pieces() {
        let mut s = Describe::create();
        s.n = 2000;
        s.redraw();
        s.refresh();
        let key = s.error_key();
        let x = s.x.clone();
        let mut calls = 0;
        while !s.errors.done(key, s.source) {
            s.errors.work(key, &x, s.mark, s.source, || true);
            calls += 1;
            assert!(calls <= 2 * TRIED);
        }
        assert_eq!(calls, 2 * TRIED);
        s.band = 0.9;
        s.auto_band = false;
        s.changed();
        s.refresh();
        assert!(s.errors.done(s.error_key(), s.source));
        s.mark += 0.1;
        s.changed();
        s.refresh();
        assert!(!s.errors.done(s.error_key(), s.source));
    }

    /// The dots in the strip never cover one another on a level.
    #[test]
    fn stacked_dots_do_not_cover_each_other() {
        let x: Vec<f32> = st::sorted(&draw(Source::Humps, 2000, 2)).iter().map(|v| (500.0 + 80.0 * v) as f32).collect();
        let levels = stack(&x, 4.0);
        for (i, a) in x.iter().enumerate() {
            for (j, b) in x.iter().enumerate().skip(i + 1) {
                if levels[i] == levels[j] {
                    assert!((b - a).abs() >= 4.0 - 1e-4, "{i} and {j}");
                }
            }
        }
        assert_eq!(ordinal(1), "1st");
        assert_eq!(ordinal(12), "12th");
        assert_eq!(ordinal(23), "23rd");
        assert_eq!(ordinal(111), "111th");
        assert_eq!(peaks(&[0.0, 1.0, 1.0, 0.0, 2.0, 0.0]), 2);
    }

    /// Numbers behind the captions (run with --ignored --nocapture).
    #[test]
    #[ignore]
    fn explore() {
        for k in 0..TRIALS.len() {
            let s = after(k);
            let f = s.stats.five.expect("points");
            let (b, h) = s.best();
            println!(
                "{}: width {:.3} mark {:.3} band {:.3}; errors {:.4} {:.4}; best {b:?} {h:?}; mean {:.3} sd {:.3} median {:.3} iqr {:.3}, {} beyond the whiskers",
                TRIALS[k].button,
                s.width,
                s.mark,
                s.band,
                s.stats.bars_error,
                s.stats.smooth_error,
                s.stats.mean,
                s.stats.sd,
                f.median,
                f.iqr(),
                s.stats.outliers
            );
        }
        for seed in 1..=6u64 {
            let sorted = st::sorted(&draw(Source::Bell, 300, seed));
            let five = Five::of(&sorted).expect("points");
            let off: Vec<f64> = (0..300).map(|i| (normal_score(i, 300) - five.line(sorted[i])).abs()).collect();
            let worst = |a: usize, b: usize| off[a..b].iter().copied().fold(0.0, f64::max);
            println!("bell 300 seed {seed}: middle 96 % {:.3}, middle 90 % {:.3}, all {:.3}, slopes {:?}", worst(6, 294), worst(15, 285), worst(0, 300), slopes(&sorted));
        }
        let [a, b] = pair(201, PAIR_SEED);
        println!("pair: mean {:.3} {:.3}, sd {:.3} {:.3}", st::mean(&a), st::mean(&b), st::sd(&a), st::sd(&b));
    }
}

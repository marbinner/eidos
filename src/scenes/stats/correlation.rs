//! Correlation and its traps: what the correlation coefficient measures (how closely a
//! scatter follows a straight line, and nothing else), and the ways it misleads.
//!
//! The picture is a scatter in standard units with its four quadrants tinted by the
//! sign of the product of a point's two standard scores, r being the average of those
//! products; the three lines through the centre (the SD line and the two regression
//! lines); and, beside it, the products laid end to end, the share r², and one of
//! three charts: the top tenth (regression to the mean), the ranks (Spearman's
//! correlation) and r over new samples (correlation from nothing). Everything is
//! drawn in screen space, laid out from the canvas.

use super::plot::{self, button, button_row, dashed_of, disc, even_row, fill, line_of, within, Glide, Plot, Rect, Ticks, AXES, CURVES, MARKS, OWN};
use super::*;
use crate::math::stats as st;

pub const DEF: SceneDef = SceneDef {
    id: "correlation",
    title: "Correlation and its traps",
    group: MODELS,
    blurb: "What the correlation coefficient measures: straight-line trends only, pulled about by one outlier, and the reason extremes are followed by less extreme values.",
    make: |_| Box::new(Correlation::create()),
};

// ----- colours ---------------------------------------------------------------------------

/// A product of standard scores that is positive (both above their means, or both
/// below), and one that is negative.
const PLUS: Color = pal::BLUE;
const MINUS: Color = pal::RED;
/// How strongly the quadrants are tinted, and each product's rectangle over them.
const QUADRANT: f32 = 0.007;
const PRODUCT: f32 = 0.022;
/// The line of standard deviations: the prediction if the correlation were perfect,
/// assumed for the sake of comparison (the course's colour for what is assumed).
const SD_LINE: Color = ASSUMED;
/// The points of the top tenth, and their averages.
const TOP: Color = pal::WHITE;
/// Points left out of the numbers (below the median of x).
const LEFT_OUT: Color = pal::FAINT;

// ----- the data sets ---------------------------------------------------------------------

/// Anscombe's quartet as he published it (F. J. Anscombe, "Graphs in statistical
/// analysis", The American Statistician 27, 1973): the x of the first three sets, the
/// x of the fourth, and the four sets' y. The points are in the same order in every
/// set, so that one set glides into the next point by point (in the first three only
/// up and down: they share their x).
const ANSCOMBE_X: [f64; 11] = [10.0, 8.0, 13.0, 9.0, 11.0, 14.0, 6.0, 4.0, 12.0, 7.0, 5.0];
const ANSCOMBE_X4: [f64; 11] = [8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 19.0, 8.0, 8.0, 8.0];
const ANSCOMBE_Y: [[f64; 11]; 4] = [
    [8.04, 6.95, 7.58, 8.81, 8.33, 9.96, 7.24, 4.26, 10.84, 4.82, 5.68],
    [9.14, 8.14, 8.74, 8.77, 9.26, 8.10, 6.13, 3.10, 9.13, 7.26, 4.74],
    [7.46, 6.77, 12.74, 7.11, 7.81, 8.84, 6.08, 5.39, 8.15, 6.42, 5.73],
    [6.58, 5.76, 7.71, 8.84, 8.47, 7.04, 5.25, 12.50, 5.56, 7.91, 6.89],
];
const QUARTET: [&str; 4] = ["I", "II", "III", "IV"];
const QUARTET_SHAPES: [&str; 4] = ["a line with noise", "a curve", "a line and one outlier", "one far point makes it all"];

/// Where the far point starts, in the cloud's own units (the cloud has spread 1 each way).
const FAR: [f64; 2] = [8.0, 8.0];

#[derive(Clone, Copy, Debug, PartialEq)]
enum Set {
    /// A sample from a normal cloud with correlation ρ.
    Cloud,
    /// The same, one point fewer, and one point far out.
    Far,
    /// One of Anscombe's four.
    Anscombe(usize),
    /// y = x² at evenly spaced x from −1 to 1.
    Square,
    /// y = eˣ at evenly spaced x from 0 to 4.
    Exp,
}

const SET_NAMES: [&str; 5] = ["a cloud", "one far point", "Anscombe's quartet", r"$y = x^2$", r"$y = e^x$"];

impl Set {
    fn index(self) -> usize {
        match self {
            Set::Cloud => 0,
            Set::Far => 1,
            Set::Anscombe(_) => 2,
            Set::Square => 3,
            Set::Exp => 4,
        }
    }

    fn of(index: usize) -> Set {
        [Set::Cloud, Set::Far, Set::Anscombe(0), Set::Square, Set::Exp][index.min(4)]
    }

    /// Is it drawn at random (from ρ, n and a seed)?
    fn random(self) -> bool {
        matches!(self, Set::Cloud | Set::Far)
    }

    /// The curve the points were made from, if there is one.
    fn curve(self) -> Option<fn(f64) -> f64> {
        match self {
            Set::Square => Some(|x| x * x),
            Set::Exp => Some(f64::exp),
            _ => None,
        }
    }
}

/// n points from a normal cloud: each variable of mean 0 and spread 1, with correlation ρ.
fn cloud_from(rho: f64, n: usize, rng: &mut Rng) -> Vec<[f64; 2]> {
    let rho = rho.max(-1.0).min(1.0);
    let c = (1.0 - rho * rho).max(0.0).sqrt();
    (0..n)
        .map(|_| {
            let a = st::normal(rng);
            let b = st::normal(rng);
            [a, rho * a + c * b]
        })
        .collect()
}

fn make(set: Set, rho: f64, n: usize, seed: u64) -> Vec<[f64; 2]> {
    match set {
        Set::Cloud => cloud_from(rho, n.max(1), &mut Rng::new(seed)),
        Set::Far => {
            let mut pts = cloud_from(rho, n.max(2) - 1, &mut Rng::new(seed));
            pts.push(FAR);
            pts
        }
        Set::Anscombe(k) => {
            let k = k.min(3);
            (0..11).map(|i| [if k == 3 { ANSCOMBE_X4[i] } else { ANSCOMBE_X[i] }, ANSCOMBE_Y[k][i]]).collect()
        }
        // (x = k/10 and −k/10 exactly, so that the cloud is exactly symmetric)
        Set::Square => (0..=20).map(|i| (i as f64 - 10.0) / 10.0).map(|x| [x, x * x]).collect(),
        Set::Exp => (0..=20).map(|i| i as f64 / 5.0).map(|x| [x, x.exp()]).collect(),
    }
}

// ----- what a scatter says ---------------------------------------------------------------

/// The numbers of a scatter, worked out from points in their own units.
#[derive(Clone, Debug, Default, PartialEq)]
struct Summary {
    /// The points the numbers are of: all of them, or those with x above its median.
    used: Vec<usize>,
    mean: [f64; 2],
    /// The standard deviations, dividing by n (so that r is the average product).
    sd: [f64; 2],
    /// Does each variable vary at all?
    varies: [bool; 2],
    /// The correlation coefficient, if both vary.
    r: Option<f64>,
    /// Each used point's product of standard scores, in the order of `used`.
    products: Vec<f64>,
    /// Spearman's: r of the ranks, if both sets of ranks vary.
    spearman: Option<f64>,
    median_x: f64,
}

impl Summary {
    /// A value of variable `k` in standard units (0 for a variable with no spread).
    fn z(&self, v: f64, k: usize) -> f64 {
        if self.varies[k] {
            (v - self.mean[k]) / self.sd[k]
        } else {
            0.0
        }
    }

    /// The spread used to draw standard units: the standard deviation, or 1 for a
    /// variable with none (its points then sit on its mean).
    fn unit(&self, k: usize) -> f64 {
        if self.varies[k] {
            self.sd[k]
        } else {
            1.0
        }
    }

    /// Both regression lines and the SD line, as a point and a direction each, in the
    /// data's own units: y on x (slope r s_y/s_x), x on y (slope s_y/(r s_x), upright
    /// when r = 0) and the SD line (slope ±s_y/s_x; none when r = 0, which has no sign).
    fn lines(&self) -> Option<[Option<[f64; 2]>; 3]> {
        let r = self.r?;
        let [sx, sy] = self.sd;
        let sign = if r > 1e-12 {
            Some(1.0)
        } else if r < -1e-12 {
            Some(-1.0)
        } else {
            None
        };
        Some([sign.map(|s| [sx, s * sy]), Some([sx, r * sy]), Some([r * sx, sy])])
    }

    /// The share of the variance of y that the regression line on x accounts for.
    fn r2(&self) -> Option<f64> {
        self.r.map(|r| r * r)
    }

    /// The sums of the positive and of the negative products, each divided by n: r is
    /// their sum.
    fn halves(&self) -> (f64, f64) {
        let n = self.products.len().max(1) as f64;
        let plus = self.products.iter().filter(|p| **p > 0.0).sum::<f64>() / n;
        let minus = self.products.iter().filter(|p| **p < 0.0).sum::<f64>() / n;
        (plus, minus)
    }
}

/// Does a sample of numbers vary? (Not if every number is the same, near enough for
/// rounding.)
fn has_spread(sd: f64, mean: f64) -> bool {
    sd.is_finite() && sd > 1e-9 * (1.0 + mean.abs())
}

fn summarise(pts: &[[f64; 2]], window: bool) -> Summary {
    let xs: Vec<f64> = pts.iter().map(|p| p[0]).collect();
    let median_x = st::median(&xs);
    let used: Vec<usize> = if window { (0..pts.len()).filter(|&i| pts[i][0] > median_x).collect() } else { (0..pts.len()).collect() };
    let column = |k: usize| used.iter().map(|&i| pts[i][k]).collect::<Vec<f64>>();
    let (x, y) = (column(0), column(1));
    let mean = [st::mean(&x), st::mean(&y)];
    let sd = [st::variance_n(&x).sqrt(), st::variance_n(&y).sqrt()];
    let varies = [has_spread(sd[0], mean[0]), has_spread(sd[1], mean[1])];
    let mut s = Summary { used, mean, sd, varies, median_x, ..Default::default() };
    if varies[0] && varies[1] {
        s.products = x.iter().zip(&y).map(|(a, b)| s.z(*a, 0) * s.z(*b, 1)).collect();
        s.r = Some(st::mean(&s.products).max(-1.0).min(1.0));
        let (rx, ry) = (st::ranks(&x), st::ranks(&y));
        let spread = |r: &[f64]| has_spread(st::variance_n(r).sqrt(), st::mean(r));
        if spread(&rx) && spread(&ry) {
            s.spearman = Some(st::correlation(&rx, &ry));
        }
    } else {
        s.products = vec![0.0; x.len()];
    }
    s
}

/// The top tenth of the used points on variable `by` (at least one point), and their
/// average standard scores on both variables.
fn top_tenth(pts: &[[f64; 2]], s: &Summary, by: usize) -> (Vec<usize>, Option<[f64; 2]>) {
    let by = by.min(1);
    let mut order = s.used.clone();
    order.sort_by(|&a, &b| pts[b][by].total_cmp(&pts[a][by]).then(a.cmp(&b)));
    let k = ((order.len() as f64 / 10.0).round() as usize).max(1).min(order.len());
    order.truncate(k);
    if order.is_empty() || !(s.varies[0] && s.varies[1]) {
        return (order, None);
    }
    let avg = |c: usize| order.iter().map(|&i| s.z(pts[i][c], c)).sum::<f64>() / k as f64;
    let avg = [avg(0), avg(1)];
    (order, Some(avg))
}

// ----- r over new samples ----------------------------------------------------------------

/// Bins of r from −1 to 1 for the chart of new samples, and the r beyond which a sample
/// counts as looking related.
const BINS: usize = 20;
const TAIL: f64 = 0.5;
/// New samples stop coming after this many.
const MOST_SAMPLES: u32 = 3000;

/// ₂F₁(a, b; c; z) by its series, for 0 ≤ z ≤ 1 and c > a + b (where it converges at
/// z = 1 too).
fn hypergeometric(a: f64, b: f64, c: f64, z: f64) -> f64 {
    let (mut term, mut sum) = (1.0, 1.0);
    for k in 0..20_000 {
        let k = k as f64;
        term *= (a + k) * (b + k) / ((c + k) * (k + 1.0)) * z;
        sum += term;
        if term.abs() < 1e-13 * sum.abs() {
            break;
        }
    }
    sum
}

/// The density of the correlation coefficient of n points drawn from a normal cloud
/// with correlation ρ: R. A. Fisher's exact distribution (1915), in the form
///
///   f(r) = (n − 2) Γ(n − 1) (1 − ρ²)^((n−1)/2) (1 − r²)^((n−4)/2)
///          / (√(2π) Γ(n − ½) (1 − ρr)^(n − 3/2)) · ₂F₁(½, ½; n − ½; (1 + ρr)/2),
///
/// for |ρ| < 1, |r| < 1 and n ≥ 3 (0 elsewhere).
fn r_density(r: f64, rho: f64, n: usize) -> f64 {
    if r.is_nan() || r.abs() >= 1.0 {
        return 0.0;
    }
    // (the density in θ, where r = sin θ, divided by dr/dθ = cos θ)
    r_density_in_angle(r.asin(), rho, n) / (1.0 - r * r).sqrt()
}

/// The same density as one in θ, where r = sin θ: f(sin θ) cos θ, which is finite at
/// θ = ±π/2 for every n ≥ 3 (there (1 − r²)^((n−4)/2) cos θ = cos^(n−3) θ).
fn r_density_in_angle(t: f64, rho: f64, n: usize) -> f64 {
    if n < 3 || rho.is_nan() || rho.abs() >= 1.0 || !t.is_finite() {
        return 0.0;
    }
    let (r, c) = (t.sin().max(-1.0).min(1.0), t.cos().max(0.0));
    let k = n as f64 - 3.0;
    let power = if k == 0.0 {
        0.0
    } else if c > 0.0 {
        k * c.ln()
    } else {
        return 0.0;
    };
    let n = n as f64;
    let ln = (n - 2.0).ln() + st::ln_gamma(n - 1.0) - st::ln_gamma(n - 0.5) - 0.5 * (2.0 * std::f64::consts::PI).ln() + 0.5 * (n - 1.0) * (1.0 - rho * rho).ln() + power
        - (n - 1.5) * (1.0 - rho * r).ln();
    ln.exp() * hypergeometric(0.5, 0.5, n - 0.5, 0.5 * (1.0 + rho * r))
}

/// The probability that r of n such points is between `lo` and `hi` (in −1 to 1): by
/// Simpson's rule in θ, with r = sin θ, which leaves nothing infinite at the ends.
fn r_probability(lo: f64, hi: f64, rho: f64, n: usize) -> f64 {
    let (a, b) = (lo.max(-1.0).min(1.0).asin(), hi.max(-1.0).min(1.0).asin());
    if a.is_nan() || b.is_nan() || b <= a {
        return 0.0;
    }
    let steps = 400;
    let h = (b - a) / steps as f64;
    let g = |t: f64| r_density_in_angle(t, rho, n);
    let mut sum = g(a) + g(b);
    for i in 1..steps {
        sum += g(a + i as f64 * h) * if i % 2 == 1 { 4.0 } else { 2.0 };
    }
    sum * h / 3.0
}

/// New samples of the cloud, one after another, and what their r come to.
struct Samples {
    /// Drawing them (while the chart beside the scatter shows them).
    running: bool,
    /// New samples a second (of the scene's time).
    rate: f32,
    pace: Pace,
    rng: Rng,
    /// How many r fell in each bin, how many there are, and how many are beyond ±TAIL.
    counts: Vec<u32>,
    total: u32,
    beyond: u32,
    /// The ρ and n they are of: new ones start the count afresh.
    key: Option<(u32, usize)>,
    /// The r of the latest.
    last: f64,
}

impl Samples {
    fn new() -> Samples {
        Samples { running: false, rate: 2.0, pace: Pace::default(), rng: Rng::new(SAMPLES_SEED), counts: vec![0; BINS], total: 0, beyond: 0, key: None, last: f64::NAN }
    }

    fn restart(&mut self, rho: f32, n: usize) {
        self.rng = Rng::new(SAMPLES_SEED);
        self.counts = vec![0; BINS];
        (self.total, self.beyond, self.last) = (0, 0, f64::NAN);
        self.key = Some((rho.to_bits(), n));
        self.pace.reset();
    }

    /// One more sample: its points, and its r counted.
    fn draw(&mut self, rho: f32, n: usize) -> Vec<[f64; 2]> {
        let pts = cloud_from(rho as f64, n, &mut self.rng);
        if let Some(r) = summarise(&pts, false).r {
            let bin = (((r + 1.0) / 2.0 * BINS as f64) as usize).min(BINS - 1);
            self.counts[bin] += 1;
            self.total += 1;
            self.beyond += (r.abs() > TAIL) as u32;
            self.last = r;
        }
        pts
    }

    fn done(&self) -> bool {
        self.total >= MOST_SAMPLES
    }
}

// ----- the experiments -------------------------------------------------------------------

/// The seeds of the samples shown: the opening cloud, the far point's cloud, the
/// thousand points of the top tenth, the window's two hundred, and the new samples.
const OPENING_SEED: u64 = 25;
const FAR_SEED: u64 = 7;
const TENTH_SEED: u64 = 326;
const WINDOW_SEED: u64 = 154;
const SAMPLES_SEED: u64 = 11;

/// Experiments: (button, what to look for). `Correlation::set_up` sets each one up.
const TRIALS: [(&str, &str); 7] = [
    (
        "four data sets, one $r$",
        r#"Anscombe's quartet (1973): four sets of 11 points with the same means ($\bar x = 9.00$, $\bar y = 7.50$), the same spreads, the same $r = 0.82$ and the same regression line $y = 3.00 + 0.500x$. Click I to IV over the picture: a line with noise, a curve, a line with one outlier, and a single far point that makes all of the correlation. The numbers cannot tell them apart; a picture can."#,
    ),
    (
        "a curve with $r = 0$",
        r"Every $y$ is exactly $x^2$, yet $r = 0$: each point's product is cancelled by its mirror image's, on the other side of $x = 0$. $r$ measures only how well a straight line does, and the best one here is level. No correlation does not mean no relation.",
    ),
    (
        "one far point",
        r"Twenty points with no relation ($r = -0.04$ among themselves) and one far point: $r = 0.80$, and that one point's product is most of the bar. Drag it into the cloud and $r$ falls back to about 0; drag it to the top left and $r = -0.81$. The rank correlation hardly notices it (0.12).",
    ),
    (
        "regression to the mean",
        r#"A thousand points with $r = 0.50$. The top tenth on $x$ (white) are on average 1.77 standard deviations above the mean of $x$, but only 0.87 above that of $y$: about $r$ times as far, where the yellow line puts them. Nothing pulls them back: it is what $|r| < 1$ means. It works both ways: the top tenth on $y$ ("by $y$") are only 0.87 above the mean of $x$."#,
    ),
    (
        "a narrow window",
        r"200 points with $r = 0.80$, looked at only where $x$ is above its median: $r = 0.63$. How $y$ goes with $x$ has not changed: the yellow line of these points lies on the dashed one of all of them (slope 0.80 either way), and they scatter about it as much as all do. Only $x$ varies less, so the line accounts for less of $y$'s variance: $r^2$ falls from 0.65 to 0.40.",
    ),
    (
        "correlation from nothing",
        r"New samples of 10 points from two unrelated variables ($\rho = 0$), two a second. Their $r$ scatters widely about 0, beyond $\pm 0.5$ (pink) in about one sample in seven: 14 % in the long run, for normal data. From 10 points, an $r$ of 0.5 is no evidence of a relation.",
    ),
    (
        "ranks",
        r"$y = e^x$: $y$ rises with every step in $x$, but not along a straight line, so $r = 0.89$. The rank correlation (Spearman's) is $r$ of the ranks instead of the values (right): they lie on a line, so it is exactly 1. It measures whether $y$ rises with $x$, not how straight.",
    ),
];

// ----- the layout ------------------------------------------------------------------------

/// What the chart beside the scatter shows.
#[derive(Clone, Copy, Debug, PartialEq)]
enum Side {
    Tenth,
    Ranks,
    Samples,
}

const SIDES: [(Side, &str); 3] = [(Side::Tenth, "top tenth"), (Side::Ranks, "ranks"), (Side::Samples, "new samples")];

/// Where everything goes on a canvas (logical pixels).
#[derive(Clone, Copy, Debug)]
struct Layout {
    view: Vec2,
    narrow: bool,
    /// Too small a canvas for more than the scatter.
    compact: bool,
    /// Text sizes: headings, words, small print.
    big: f32,
    text: f32,
    small: f32,
    head: Vec2,
    /// The row of buttons over the scatter, from its left and from its right.
    buttons: (Vec2, f32),
    button_h: f32,
    scatter: Rect,
    /// Room at the left of the scatter for its numbers.
    numbers: f32,
    /// Where the names of the lines start, under the scatter, and how wide they may run.
    legend: Vec2,
    legend_w: f32,
    /// The products, laid end to end; r²; the tabs; and the chart they choose.
    bar: Rect,
    share: Rect,
    tabs: [Rect; 3],
    side: Rect,
}

impl Layout {
    fn of(view: Vec2, inset: f32) -> Layout {
        let view = Vec2::new(view.x.max(1.0), view.y.max(1.0));
        // (small print below 760 pixels across; one column only on a canvas that is
        // also upright, a phone's)
        let small_print = view.x < 760.0;
        let narrow = view.x < 560.0 || (small_print && view.y > view.x);
        let (big, text, small) = if small_print { (13.0, 11.5, 10.5) } else { (15.0, 12.5, 11.5) };
        let margin = if small_print { 10.0 } else { 20.0 };
        let top = inset.max(0.0) + if small_print { 6.0 } else { 12.0 };
        let head = Vec2::new(margin, top);
        let head_h = big + 6.0 + text + 8.0;
        let button_h = if small_print { 22.0 } else { 24.0 };
        let numbers = if small_print { 30.0 } else { 38.0 };
        // under the scatter: its numbers and the name of x, then the names of the lines
        let below = small + 8.0 + small + 8.0;
        let buttons_y = top + head_h;
        let scatter_top = buttons_y + button_h + 12.0 + text;
        let compact = view.x < 300.0 || view.y < 360.0;
        let zero: Rect = (Vec2::ZERO, Vec2::ZERO);
        if compact {
            let top = inset.max(0.0) + 8.0 + text + 10.0;
            let s = (view.x - 2.0 * 8.0 - numbers).min(view.y - top - below).max(10.0);
            let scatter = (Vec2::new(8.0 + numbers, top), Vec2::new(8.0 + numbers + s, top + s));
            return Layout {
                view,
                narrow,
                compact,
                big,
                text,
                small,
                head: Vec2::new(8.0, inset.max(0.0) + 6.0),
                buttons: (Vec2::ZERO, 0.0),
                button_h,
                scatter,
                numbers,
                legend: Vec2::ZERO,
                legend_w: 0.0,
                bar: zero,
                share: zero,
                tabs: [zero; 3],
                side: zero,
            };
        }
        let (bar_h, share_h, gap) = (text + 10.0 + 20.0 + 6.0 + small, text + 8.0 + 14.0 + 6.0 + small, if small_print { 14.0 } else { 22.0 });
        if narrow {
            // one column: the scatter, its legend, the bars, the tabs and their chart
            let legend_h = 2.0 * (small + 6.0);
            let side_min = 160.0;
            // (as wide as the canvas allows, and as tall as it can be beside the rest,
            // up to square: one standard deviation is as long across as up either way)
            let width = view.x - 2.0 * margin - numbers - 6.0;
            let rest = view.y - scatter_top - below - legend_h - bar_h - share_h - button_h - side_min - 3.0 * gap - 16.0;
            let height = width.min(rest).max(60.0);
            let left = margin + numbers;
            let scatter = (Vec2::new(left, scatter_top), Vec2::new(left + width.max(60.0), scatter_top + height));
            let legend = Vec2::new(margin, scatter.1.y + below);
            let mut y = legend.y + legend_h + 6.0;
            let (x0, x1) = (margin + 4.0, view.x - margin - 4.0);
            let bar = (Vec2::new(x0, y), Vec2::new(x1, y + bar_h));
            y += bar_h + gap;
            let share = (Vec2::new(x0, y), Vec2::new(x1, y + share_h));
            y += share_h + gap;
            let tabs = even_row(3, x0, x1, y, button_h, 6.0);
            y += button_h + 12.0;
            let side = (Vec2::new(x0, y), Vec2::new(x1, (view.y - 10.0).max(y + 20.0)));
            Layout {
                view,
                narrow,
                compact,
                big,
                text,
                small,
                head,
                buttons: (Vec2::new(margin, buttons_y), view.x - margin),
                button_h,
                scatter,
                numbers,
                legend,
                legend_w: view.x - 2.0 * margin,
                bar,
                share,
                tabs: [tabs[0], tabs[1], tabs[2]],
                side,
            }
        } else {
            // the scatter on the left, the rest in a column beside it
            let legend_h = 2.0 * (small + 6.0);
            let column_gap = if small_print { 44.0 } else { 56.0 };
            let s = (view.y - scatter_top - below - legend_h - 10.0).min((view.x - 2.0 * margin - numbers - column_gap) * 0.6).max(60.0);
            let scatter = (Vec2::new(margin + numbers, scatter_top), Vec2::new(margin + numbers + s, scatter_top + s));
            let legend = Vec2::new(margin, scatter.1.y + below);
            let (x0, x1) = (scatter.1.x + column_gap, view.x - margin);
            let mut y = buttons_y + 4.0;
            let bar = (Vec2::new(x0, y), Vec2::new(x1, y + bar_h));
            y += bar_h + gap;
            let share = (Vec2::new(x0, y), Vec2::new(x1, y + share_h));
            y += share_h + gap;
            let tabs = even_row(3, x0, x1, y, button_h, 8.0);
            y += button_h + 14.0;
            let side = (Vec2::new(x0, y), Vec2::new(x1, (view.y - 16.0).min(y + 480.0).max(scatter.1.y + below).max(y + 20.0)));
            Layout {
                view,
                narrow,
                compact,
                big,
                text,
                small,
                head,
                buttons: (Vec2::new(margin, buttons_y), scatter.1.x),
                button_h,
                scatter,
                numbers,
                legend,
                legend_w: scatter.1.x - margin,
                bar,
                share,
                tabs: [tabs[0], tabs[1], tabs[2]],
                side,
            }
        }
    }
}

/// The first of `options` that fits in `room` logical pixels at `size` (the last if none does).
fn pick(f: &Frame, options: &[String], room: f32, size: f32) -> String {
    options.iter().find(|t| f.text_size(t, size).x <= room).or(options.last()).cloned().unwrap_or_default()
}

/// Buttons side by side, ending at `right`.
fn row_ending(f: &Frame, words: &[&str], right: f32, y: f32, h: f32, size: f32) -> Vec<Rect> {
    let rects = button_row(f, words, 0.0, y, h, size, 6.0);
    let shift = right - rects.last().map_or(0.0, |r| r.1.x);
    rects.into_iter().map(|(a, b)| (a + Vec2::new(shift, 0.0), b + Vec2::new(shift, 0.0))).collect()
}

/// Text broken into lines no wider than `width`, from `at` down (screen space): all of
/// it if it fits above `bottom`, and none of it if not (a sentence cut short would say
/// something else). Returns the height taken.
fn wrap(f: &mut Frame, at: Vec2, width: f32, bottom: f32, text: &str, size: f32, color: Color) -> f32 {
    let mut lines: Vec<String> = vec![String::new()];
    for word in words_of(text) {
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
    if at.y + lines.len() as f32 * step > bottom + 2.0 {
        return 0.0;
    }
    for (k, line) in lines.iter().enumerate() {
        f.text_px(at + Vec2::new(0.0, k as f32 * step), line.as_str(), size, color, Align2::LEFT_TOP);
    }
    lines.len() as f32 * step
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
                words.push(&text[start..i]);
                start = i + 1;
            }
            _ => {}
        }
        i += 1;
    }
    words.push(&text[start.min(text.len())..]);
    words
}

/// The kinds of line drawn in batches of their own (see `plot::apart`): those whose
/// number of points can change from one frame to the next, each apart.
const SD_KIND: u8 = OWN;
const YX_KIND: u8 = OWN + 1;
const XY_KIND: u8 = OWN + 2;
const MEDIAN_KIND: u8 = OWN + 3;
const TENTH_KIND: u8 = OWN + 4;
const MEANS_KIND: u8 = OWN + 5;
const BAR_KIND: u8 = OWN + 6;
const SIDE_KIND: u8 = OWN + 7;
const RANK_KIND: u8 = OWN + 8;
const NOW_KIND: u8 = OWN + 9;
const LEGEND_KIND: u8 = OWN + 10;
const DENSITY_KIND: u8 = OWN + 11;
const FOCUS_KIND: u8 = OWN + 12;
const ALL_KIND: u8 = OWN + 13;

/// Above this many points the points cannot be dragged (they would be a carpet of
/// handles), and their products are not drawn one by one (they would be a blur).
const DRAG_MOST: usize = 60;
const PRODUCTS_MOST: usize = 150;

// ----- the scene -------------------------------------------------------------------------

pub struct Correlation {
    set: Set,
    rho: f32,
    n: usize,
    seed: u64,
    /// The data, in their own units: what every number is of.
    pts: Vec<[f64; 2]>,
    /// Some point has been moved by hand since the data were made.
    edited: bool,
    standard: bool,
    /// Which lines: the SD line, the regression of y on x, that of x on y.
    lines: [bool; 3],
    products: bool,
    /// Only the points with x above its median.
    window: bool,
    side: Side,
    /// The top tenth on x (0) or on y (1).
    by: usize,
    samples: Samples,
    /// The points as drawn ([x0, y0, x1, y1, …], their own units), the ranges of the
    /// scatter's axes as drawn (x from, to, y from, to), and the chart of new samples
    /// (each bin's height, then the latest r).
    shown: Glide,
    range: Glide,
    bars: Glide,
    /// The sign of the SD line's slope as drawn: it turns from one sign to the other,
    /// rather than flipping in a frame, when a new sample's r has the other sign.
    sd_sign: Glide,
    /// The angles of the two regression lines as drawn, in standard units (y on x,
    /// then x on y): they turn evenly to the data's, where the r of the points on
    /// their way (which is no sample's) would swing them to and fro.
    angles: Glide,
    /// The top tenth as drawn (where it starts, on the variable it is chosen by, and
    /// its average standard scores on x and y), and each point's two ranks as drawn:
    /// both change in steps as a point is dragged past another, and glide there.
    tenth: Glide,
    ranks: Glide,
    /// The point being dragged.
    held: Option<usize>,
    /// The canvas last drawn on (a new one is drawn at once, without gliding).
    canvas: Option<Vec2>,
    /// The round values on the scatter's axes (chosen where the axes are going), and how
    /// many axis lines and labels the last frame drew.
    ticks: Option<[Ticks; 2]>,
    tick_key: [f64; 6],
    drawn_last: (usize, usize),
    /// Something is still on its way.
    moving: bool,
}

impl Correlation {
    /// (Nothing here needs the graphics device, so the tests can make one too.)
    fn create() -> Correlation {
        let mut s = Correlation {
            set: Set::Cloud,
            rho: 0.6,
            n: 40,
            seed: OPENING_SEED,
            pts: Vec::new(),
            edited: false,
            standard: true,
            lines: [true; 3],
            products: false,
            window: false,
            side: Side::Tenth,
            by: 0,
            samples: Samples::new(),
            shown: Glide::default(),
            range: Glide::default(),
            bars: Glide::default(),
            sd_sign: Glide::default(),
            angles: Glide::default(),
            tenth: Glide::default(),
            ranks: Glide::default(),
            held: None,
            canvas: None,
            ticks: None,
            tick_key: [f64::NAN; 6],
            drawn_last: (0, 0),
            moving: false,
        };
        s.remake();
        s
    }

    /// Make the data afresh from the settings.
    fn remake(&mut self) {
        self.n = self.n.clamp(2, 1000);
        self.rho = self.rho.max(-1.0).min(1.0);
        self.pts = make(self.set, self.rho as f64, self.n, self.seed);
        self.edited = false;
        self.held = None;
        if self.side == Side::Samples {
            self.start_samples();
        }
    }

    /// Show new samples beside the scatter: of the cloud, counted afresh if its ρ or n
    /// has changed.
    fn start_samples(&mut self) {
        if self.set != Set::Cloud {
            self.set = Set::Cloud;
            self.pts = make(self.set, self.rho as f64, self.n, self.seed);
            self.edited = false;
        }
        if self.samples.key != Some((self.rho.to_bits(), self.n)) {
            self.samples.restart(self.rho, self.n);
        }
        self.samples.running = !self.samples.done();
    }

    fn choose_side(&mut self, side: Side) {
        self.side = side;
        if side == Side::Samples {
            self.start_samples();
        } else {
            self.samples.running = false;
        }
    }

    fn choose_set(&mut self, set: Set) {
        self.set = set;
        if !set.random() && self.side == Side::Samples {
            self.choose_side(Side::Tenth);
        }
        self.remake();
    }

    /// Set up one of the experiments in `TRIALS`: every setting its lesson depends on.
    fn set_up(&mut self, k: usize) {
        (self.standard, self.lines, self.products, self.window, self.by) = (true, [true; 3], false, false, 0);
        self.samples.running = false;
        self.side = Side::Tenth;
        match k {
            0 => {
                self.set = Set::Anscombe(0);
                self.standard = false;
                self.side = Side::Ranks;
            }
            1 => {
                self.set = Set::Square;
                self.side = Side::Ranks;
            }
            2 => {
                (self.set, self.rho, self.n, self.seed) = (Set::Far, 0.0, 21, FAR_SEED);
                self.side = Side::Ranks;
            }
            3 => (self.set, self.rho, self.n, self.seed) = (Set::Cloud, 0.5, 1000, TENTH_SEED),
            4 => {
                (self.set, self.rho, self.n, self.seed) = (Set::Cloud, 0.8, 200, WINDOW_SEED);
                (self.window, self.standard, self.lines) = (true, false, [false, true, false]);
                self.side = Side::Ranks;
            }
            5 => {
                (self.set, self.rho, self.n, self.seed) = (Set::Cloud, 0.0, 10, OPENING_SEED);
                self.samples.rate = 2.0;
                self.samples.key = None;
                self.side = Side::Samples;
            }
            _ => {
                self.set = Set::Exp;
                self.side = Side::Ranks;
            }
        }
        self.remake();
    }

    /// The numbers of the data as they are.
    fn summary(&self) -> Summary {
        summarise(&self.pts, self.window)
    }

    /// The points as drawn.
    fn shown_points(&self) -> Vec<[f64; 2]> {
        self.shown.shown.chunks_exact(2).map(|p| [p[0] as f64, p[1] as f64]).collect()
    }

    fn flat(&self) -> Vec<f32> {
        self.pts.iter().flat_map(|p| [p[0] as f32, p[1] as f32]).collect()
    }

    /// Where the scatter's axes are going, for the data as they are: in standard
    /// units, the same number of standard deviations each way, as many as the points
    /// need (and one standard deviation as many pixels across as up); in the data's
    /// own, a little beyond the points.
    fn target_range(&self, lay: &Layout, s: &Summary) -> [f32; 4] {
        let (w, h) = ((lay.scatter.1.x - lay.scatter.0.x).max(1.0) as f64, (lay.scatter.1.y - lay.scatter.0.y).max(1.0) as f64);
        if self.standard && !s.used.is_empty() {
            let mut far = 3.2f64;
            for &i in &s.used {
                for k in 0..2 {
                    far = far.max(1.08 * s.z(self.pts[i][k], k).abs());
                }
            }
            let far = far.min(40.0);
            let side = w.min(h);
            let (hx, hy) = (far * w / side * s.unit(0), far * h / side * s.unit(1));
            return [(s.mean[0] - hx) as f32, (s.mean[0] + hx) as f32, (s.mean[1] - hy) as f32, (s.mean[1] + hy) as f32];
        }
        let reach = |k: usize| {
            let (lo, hi) = self.pts.iter().fold((f64::MAX, f64::MIN), |(a, b), p| (a.min(p[k]), b.max(p[k])));
            if lo.is_nan() || hi.is_nan() || hi < lo {
                return [-1.0, 1.0];
            }
            let pad = ((hi - lo) * 0.08).max(0.5 * (1e-3 + 1e-3 * lo.abs().max(hi.abs())));
            [(lo - pad) as f32, (hi + pad) as f32]
        };
        let (x, y) = (reach(0), reach(1));
        [x[0], x[1], y[0], y[1]]
    }

    /// The chart of the scatter as it is drawn now.
    fn plot(&self, lay: &Layout) -> Plot {
        let r = &self.range.shown;
        let at = |i: usize| r.get(i).copied().unwrap_or(0.0) as f64;
        Plot::new(lay.scatter, [at(0), at(1)], [at(2), at(3)])
    }

    /// Bring everything drawn a frame nearer to where it is going.
    fn glide(&mut self, lay: &Layout, dt: f32) {
        let to = self.flat();
        let mut moving = if self.held.is_some() {
            self.shown.snap(&to);
            false
        } else {
            self.shown.follow(&to, dt, false)
        };
        let snap = self.canvas != Some(lay.view) || self.range.shown.len() != 4;
        if self.held.is_none() {
            let target = self.target_range(lay, &self.summary());
            if snap {
                self.range.snap(&target);
            } else {
                moving |= self.range.follow(&target, dt, false);
            }
        }
        self.canvas = Some(lay.view);
        let width = 2.0 / BINS as f64;
        let total = self.samples.total.max(1) as f64;
        let mut bars: Vec<f32> = self.samples.counts.iter().map(|&c| (c as f64 / total / width) as f32).collect();
        bars.push(if self.samples.last.is_finite() { self.samples.last as f32 } else { 0.0 });
        moving |= self.bars.follow(&bars, dt, true);
        // (r = 0 has no sign: the line is not drawn then, and keeps the last one)
        let sign = match self.summary().r {
            Some(r) if r > 1e-12 => Some(1.0),
            Some(r) if r < -1e-12 => Some(-1.0),
            _ => None,
        };
        match (sign, self.sd_sign.shown.len()) {
            (Some(v), 1) if !snap => moving |= self.sd_sign.follow(&[v], dt, false),
            (Some(v), _) => self.sd_sign.snap(&[v]),
            _ => {}
        }
        if let Some(t) = self.tenth_target() {
            if snap || self.tenth.shown.len() != 3 {
                self.tenth.snap(&t);
            } else {
                moving |= self.tenth.follow(&t, dt, false);
            }
        }
        let ranks = self.ranks_target();
        if snap {
            self.ranks.snap(&ranks);
        } else {
            moving |= self.ranks.follow(&ranks, dt, false);
        }
        // (while a point is held, the lines answer the hand at once)
        if let Some(r) = self.summary().r {
            let to = [r.atan() as f32, 1.0f64.atan2(r) as f32];
            if snap || self.held.is_some() || self.angles.shown.len() != 2 {
                self.angles.snap(&to);
            } else {
                moving |= self.angles.follow(&to, dt, false);
            }
        }
        self.moving = moving;
    }

    /// Where the top tenth starts (between its least value and the next below, on the
    /// variable it is chosen by) and its average standard scores, for the data as they are.
    fn tenth_target(&self) -> Option<[f32; 3]> {
        let s = self.summary();
        let by = self.by.min(1);
        let (top, avg) = top_tenth(&self.pts, &s, by);
        let avg = avg?;
        let edge = top.iter().map(|&i| self.pts[i][by]).fold(f64::MAX, f64::min);
        let rest = s.used.iter().filter(|i| !top.contains(i)).map(|&i| self.pts[i][by]).fold(f64::MIN, f64::max);
        let cut = if rest > f64::MIN { 0.5 * (edge + rest) } else { edge };
        Some([cut as f32, avg[0] as f32, avg[1] as f32])
    }

    /// Each used point's rank on x and on y, for the data as they are.
    fn ranks_target(&self) -> Vec<f32> {
        let s = self.summary();
        let x: Vec<f64> = s.used.iter().map(|&i| self.pts[i][0]).collect();
        let y: Vec<f64> = s.used.iter().map(|&i| self.pts[i][1]).collect();
        st::ranks(&x).iter().zip(st::ranks(&y)).flat_map(|(a, b)| [*a as f32, b as f32]).collect()
    }

    /// New samples, so many a second of the scene's time, the latest one shown.
    fn sample(&mut self, dt: f32, real_dt: f32) {
        if self.side != Side::Samples || !self.samples.running || dt <= 0.0 || self.held.is_some() {
            return;
        }
        let mut turn = scene_turn(&mut self.samples.pace, self.samples.rate, dt, real_dt);
        let mut latest = None;
        while !self.samples.done() && turn.next() {
            latest = Some(self.samples.draw(self.rho, self.n));
        }
        self.samples.pace.end(turn);
        if let Some(pts) = latest {
            self.pts = pts;
            self.edited = false;
        }
        if self.samples.done() {
            self.samples.running = false;
        }
    }

    /// What each point is drawn as: in the top tenth, left out of the numbers, the far
    /// point, or one like the rest.
    fn colors(&self, pts: &[[f64; 2]], s: &Summary) -> Vec<Color> {
        let n = pts.len();
        let mut colors = vec![DATA; n];
        let used: std::collections::HashSet<usize> = s.used.iter().copied().collect();
        for (i, c) in colors.iter_mut().enumerate() {
            if !used.contains(&i) {
                *c = LEFT_OUT;
            }
        }
        if self.set == Set::Far && n > 0 && used.contains(&(n - 1)) {
            colors[n - 1] = ALARM;
        }
        if let (Side::Tenth, (top, Some(_))) = (self.side, top_tenth(pts, s, self.by)) {
            for i in top {
                colors[i] = TOP;
            }
        }
        colors
    }

    fn point_size(&self) -> f32 {
        match self.pts.len() {
            0..=60 => 9.0,
            61..=200 => 5.5,
            _ => 3.6,
        }
    }

    /// The points as handles, while there are few enough: a point dragged is put where
    /// the pointer puts it (the axes stand still meanwhile).
    fn handles(&mut self, f: &mut Frame, plot: &Plot) {
        let n = self.pts.len();
        if n > DRAG_MOST || plot.tiny() || self.shown.shown.len() != 2 * n {
            self.held = None;
            return;
        }
        let pts = self.shown_points();
        let colors = self.colors(&pts, &summarise(&pts, self.window));
        let size = self.point_size();
        let mut held = None;
        for i in 0..n {
            // (a point outside the axes while they glide waits at their edge: the same
            // points in every frame)
            let at = plot.at(pts[i][0], pts[i][1]);
            if !at.is_finite() {
                continue;
            }
            let at = at.max(plot.rect.0).min(plot.rect.1);
            let Some(world) = f.cam.pick_ground(at * f.px) else { continue };
            let mut p = world;
            let big = if colors[i] == ALARM { 3.0 } else { 0.0 };
            if f.drag3_sized(100 + i as u32, &mut p, Some(Vec3::Z), colors[i], size + big) {
                held = Some(i);
                if let Some(screen) = on_screen(f, p) {
                    let q = screen.max(plot.rect.0).min(plot.rect.1);
                    let (x, y) = (plot.x_at(q.x), plot.y_at(q.y));
                    if x.is_finite() && y.is_finite() {
                        self.pts[i] = [x, y];
                        self.edited = true;
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
        self.held = held;
    }
}

impl Scene for Correlation {
    fn view(&self) -> View {
        View::Flat { center: Vec2::ZERO, half_height: 1.0 }
    }

    fn settings(&self, s: &mut ViewSettings) {
        s.grid = false;
        s.bloom = s.bloom.min(0.2);
    }

    fn animated(&self) -> bool {
        self.moving || self.held.is_some() || (self.side == Side::Samples && self.samples.running)
    }

    fn pointer_hint(&self) -> Option<&'static str> {
        Some("drag a dot to move that point    buttons over the picture: click")
    }

    fn ui(&mut self, ui: &mut egui::Ui, _: &Gfx) {
        ui::note(ui, r"The correlation coefficient $r$ says how closely a scatter of points follows a straight line: 1 if they lie exactly on a rising line, $-1$ on a falling one, 0 if the best straight line through them is level. To work it out, put each variable in standard units (how many standard deviations from its mean), multiply each point's two numbers, and average the products. A point in the blue quadrants adds to $r$, one in the red ones takes away:");
        ui::equation(ui, r"r = \frac{1}{n} \sum_{i=1}^{n} \frac{x_i - \bar x}{s_x} \cdot \frac{y_i - \bar y}{s_y}");

        ui::section(ui, "Things to try");
        for (k, trial) in TRIALS.iter().enumerate() {
            if ui::experiment(ui, trial.0, trial.1) {
                self.set_up(k);
            }
        }

        ui::section(ui, "The data");
        let mut which = self.set.index();
        if ui::choice(ui, &SET_NAMES, &mut which) {
            self.choose_set(Set::of(which));
        }
        if self.set.random() {
            let mut again = ui::slider(ui, r"correlation $\rho$ of the cloud", &mut self.rho, -1.0..=1.0);
            again |= ui::slider_int_log(ui, "points", &mut self.n, 2..=1000);
            ui::note(ui, r"The cloud is normal, each variable of mean 0 and spread 1; $\rho$ is its correlation, the $r$ of all the points it could give. A sample's $r$ differs from it by chance.");
            if ui.button("draw again").clicked() {
                self.seed = self.seed.wrapping_add(1);
                again = true;
            }
            if again {
                self.remake();
            }
        }
        if let Set::Anscombe(k) = self.set {
            let mut k = k;
            if ui::choice(ui, &QUARTET, &mut k) {
                self.choose_set(Set::Anscombe(k));
            }
        }

        ui::section(ui, "Show");
        checkbox(ui, &mut self.standard, "standard units");
        ui::note(ui, "Otherwise the data's own units. In standard units a regression line's slope is $r$ itself.");
        checkbox(ui, &mut self.lines[0], "the SD line");
        checkbox(ui, &mut self.lines[1], r"the regression line of $y$ on $x$");
        checkbox(ui, &mut self.lines[2], r"the regression line of $x$ on $y$");
        checkbox(ui, &mut self.products, "every point's product, as a rectangle");
        ui::note(ui, &format!("Each from the centre to its point, faintly; one point's is always drawn (the one under the pointer, or else the largest). Its area is the product. (Not drawn above {PRODUCTS_MOST} points.)"));
        checkbox(ui, &mut self.window, r"only the points with $x$ above its median");

        ui::section(ui, "Beside the scatter");
        let mut side = SIDES.iter().position(|s| s.0 == self.side).unwrap_or(0);
        let names: Vec<&str> = SIDES.iter().map(|s| s.1).collect();
        if ui::choice(ui, &names, &mut side) {
            self.choose_side(SIDES[side].0);
        }
        match self.side {
            Side::Tenth => {
                ui::choice(ui, &[r"top tenth on $x$", r"top tenth on $y$"], &mut self.by);
            }
            Side::Samples => {
                ui::slider_log(ui, "new samples a second", &mut self.samples.rate, 1.0..=60.0);
                if ui.button("count again").clicked() {
                    self.samples.restart(self.rho, self.n);
                    self.samples.running = true;
                }
                ui::note(ui, &format!("New samples of the cloud, with its $\\rho$ and as many points, until there are {MOST_SAMPLES}."));
            }
            Side::Ranks => {}
        }

        ui::section(ui, "Reading the picture");
        ui::key(
            ui,
            &[
                (DATA, "cyan dots: the data; drag them (up to 60 points)"),
                (PLUS, "blue: the quadrants where a point's product is positive (both numbers above their means, or both below), and those products"),
                (MINUS, "red: where it is negative, and those products"),
                (SD_LINE, r"purple dashed: the SD line, slope $\pm 1$ in standard units: the prediction if the correlation were perfect, and where the eye tends to put the line"),
                (ESTIMATE, r"yellow: the regression line of $y$ on $x$, the best straight-line prediction of $y$ (slope $r$ in standard units); and $r$"),
                (OTHER, r"orange: the regression line of $x$ on $y$ (slope $1/r$ in standard units)"),
                (TOP, "white dots: the top tenth, and their averages"),
                (TRUTH, r"green: the curve the points were made from; the cloud's $\rho$; how $r$ is spread in the long run"),
                (ALARM, r"pink: the far point; samples whose $r$ is beyond $\pm 0.5$"),
                (LEFT_OUT, r"grey dots: points left out (below the median of $x$)"),
            ],
        );

        ui::section(ui, "Words");
        ui::term(ui, "Standard units:", r"how many standard deviations a value is from its variable's mean, $(x - \bar x)/s_x$. The standard deviation $s_x$ (the square root of the average squared distance from the mean) divides by $n$ here, so that $r$ is exactly the average of the products.");
        ui::term(ui, "Correlation coefficient $r$:", r"the average product of the two variables in standard units, from $-1$ to 1; $\pm 1$ only when the points lie exactly on a line. It is the same in any units, and the same with $x$ and $y$ swapped.");
        ui::term(ui, "$r^2$:", r"the share of the variance of $y$ (its standard deviation squared) that the regression line on $x$ accounts for: what is left over about the line is $1 - r^2$ of it.");
        ui::term(ui, "Regression line:", r"the straight line that predicts $y$ from $x$ with the least average squared miss. In standard units its slope is $r$. The line that predicts $x$ from $y$ is another one, unless $r = \pm 1$.");
        ui::term(ui, "SD line:", r"through the means, with slope $\pm 1$ in standard units ($\pm s_y/s_x$ in the data's own): where the eye tends to put the line. When $|r| < 1$ the regression line is less steep.");
        ui::term(ui, "Regression to the mean:", r"when the points scatter about a straight line and $|r| < 1$, those chosen for an extreme $x$ have, on average, a less extreme $y$: the regression line puts them $r$ times as many standard deviations from the mean of $y$ as they are from that of $x$. No force pulls them back; it is the scatter.");
        ui::term(ui, "Rank correlation (Spearman's):", r"$r$ of the points' ranks (1 for the smallest value, 2 for the next, …) instead of their values: 1 whenever $y$ rises with every rise in $x$, straight or not.");
        ui::term(ui, "Outlier:", "a point far from the rest. One can make a correlation, or hide one.");

        ui::section(ui, "Numbers");
        let s = self.summary();
        let n = s.used.len();
        match s.r {
            Some(r) => {
                ui::readout_colored(ui, "$r$", num(r, 3), ESTIMATE);
                ui::readout(ui, "$r^2$", num(r * r, 3));
            }
            None => ui::readout(ui, "$r$", "none (no spread)"),
        }
        ui::readout(ui, "rank correlation", s.spearman.map_or("none".to_string(), |v| num(v, 3)));
        ui::readout(ui, "$n$", if self.window { format!("{n} of {}", self.pts.len()) } else { n.to_string() });
        ui::readout(ui, r"means $\bar x$, $\bar y$", format!("${}$", tuple(&s.mean, 2)));
        ui::readout(ui, r"standard deviations $s_x$, $s_y$", format!("${}$", tuple(&s.sd, 3)));
        if let (Some(r), true) = (s.r, s.varies[0] && s.varies[1]) {
            let b = r * s.sd[1] / s.sd[0];
            ui::readout_colored(ui, r"$y$ on $x$", format!(r"$y = {} {} {}x$", num(s.mean[1] - b * s.mean[0], 2), if b < 0.0 { "-" } else { "+" }, num(b.abs(), 3)), ESTIMATE);
            if r.abs() > 1e-12 {
                let d = r * s.sd[0] / s.sd[1];
                ui::readout_colored(ui, r"$x$ on $y$", format!(r"$x = {} {} {}y$", num(s.mean[0] - d * s.mean[1], 2), if d < 0.0 { "-" } else { "+" }, num(d.abs(), 3)), OTHER);
            } else {
                ui::readout_colored(ui, r"$x$ on $y$", format!(r"$x = {}$ for every $y$", num(s.mean[0], 2)), OTHER);
            }
        }
        if self.side == Side::Tenth {
            if let (top, Some(avg)) = top_tenth(&self.pts, &s, self.by) {
                ui::readout_colored(ui, &format!("top tenth ({} points): their $x$, $y$", top.len()), format!("${}$", tuple(&avg, 2)), TOP);
            }
        }
        if self.side == Side::Samples {
            let t = &self.samples;
            ui::readout(ui, r"new samples, and with $|r| > 0.5$", format!("{}, {}", t.total, t.beyond));
        }
        ui::see_also(
            ui,
            &[
                ("gauss2", "the correlated bell: the middle of each of its slices is on the regression line"),
                ("regress", "fitting lines and curves by least squares"),
                ("multireg", "several inputs at once"),
                ("confound", "what a correlation does not tell: a hidden common cause"),
            ],
        );
    }

    fn frame(&mut self, f: &mut Frame) {
        let lay = Layout::of(f.view_size(), f.top_inset);
        // (a canvas squeezed to a sliver is one the controls have opened over, on a phone)
        let tiny = lay.view.x < 120.0 || lay.view.y < 120.0;
        self.sample(f.dt, f.real_dt);
        if tiny {
            self.held = None;
        } else {
            let plot = self.plot(&lay);
            self.handles(f, &plot);
        }
        self.glide(&lay, f.real_dt);
        if tiny {
            return;
        }
        let plot = self.plot(&lay);
        let pts = self.shown_points();
        let s = summarise(&pts, self.window);
        let pointer = plot::pointer(f);
        let focus = self.focus(pointer, &plot, &pts, &s);
        self.choose_ticks(&lay);
        let new_ticks = self.tick_key_now() != self.tick_key;
        self.tick_key = self.tick_key_now();

        self.draw_heading(f, &lay, &s);
        self.draw_scatter(f, &lay, &plot, &pts, &s, focus);
        if !lay.compact {
            self.draw_buttons(f, &lay, pointer);
            self.draw_legend(f, &lay, &s);
            self.draw_products(f, &lay, &s, focus);
            self.draw_share(f, &lay, &s);
            for (k, (side, name)) in SIDES.iter().enumerate() {
                if button(f, lay.tabs[k], name, lay.text, self.side == *side) {
                    self.choose_side(*side);
                }
            }
            match self.side {
                Side::Tenth => self.draw_tenth(f, &lay, &pts, &s),
                Side::Ranks => self.draw_ranks(f, &lay, &pts, &s),
                Side::Samples => self.draw_samples(f, &lay),
            }
        }
        // new round values on an axis: make sure the axes' batch and the labels are not
        // as many as in the last frame, so that nothing takes the new ones for the old
        // ones moved
        let count = |f: &mut Frame| (plot::apart(f, AXES, false).pts.iter().filter(|p| p.width >= 0.0).count(), f.dl.labels.len());
        if new_ticks {
            let (lines, labels) = count(f);
            if lines == self.drawn_last.0 {
                line_of(f, AXES, Vec2::ZERO, Vec2::ZERO, 1.0, Color::CLEAR);
            }
            if labels == self.drawn_last.1 {
                f.text_px(Vec2::ZERO, "", 1.0, Color::CLEAR, Align2::LEFT_TOP);
            }
        }
        self.drawn_last = count(f);
    }
}

// ----- drawing ---------------------------------------------------------------------------

impl Correlation {
    /// The round values of the scatter's axes, from where the axes are going: in
    /// standard units or the data's own.
    fn choose_ticks(&mut self, lay: &Layout) {
        let s = self.summary();
        let target = self.target_range(lay, &s);
        let (w, h) = (lay.scatter.1.x - lay.scatter.0.x, lay.scatter.1.y - lay.scatter.0.y);
        // (in standard units one standard deviation is as long across as up, so the two
        // axes get the same step)
        let axis = |k: usize| {
            let (lo, hi) = (target[2 * k] as f64, target[2 * k + 1] as f64);
            let (lo, hi) = if self.standard { ((lo - s.mean[k]) / s.unit(k), (hi - s.mean[k]) / s.unit(k)) } else { (lo, hi) };
            Ticks::of(lo, hi, if k == 0 { w } else if self.standard { h } else { h * 1.3 })
        };
        self.ticks = Some([axis(0), axis(1)]);
    }

    fn tick_key_now(&self) -> [f64; 6] {
        match &self.ticks {
            Some([a, b]) => {
                let first = |t: &Ticks| t.values().next().unwrap_or(f64::NAN);
                [a.step, first(a), a.values().count() as f64, b.step, first(b), b.values().count() as f64]
            }
            None => [f64::NAN; 6],
        }
    }

    /// What the data are, and their numbers (of the points as drawn, like everything
    /// on the picture: the panel's read-outs give the data's own).
    fn draw_heading(&self, f: &mut Frame, lay: &Layout, s: &Summary) {
        let n = self.pts.len();
        let rho = format!(r"${} = {}$", tint(TRUTH, r"\rho"), num(self.rho as f64, 2));
        let what = match self.set {
            Set::Cloud if self.side == Side::Samples && self.samples.total > 0 => vec![format!("new sample number {}: {n} points from a normal cloud with {rho}", self.samples.total), format!("sample {}: {n} points, {rho}", self.samples.total)],
            Set::Cloud => vec![format!("a sample of {n} points from a normal cloud with {rho}"), format!("{n} points from a cloud with {rho}")],
            Set::Far => vec![format!("{} points from a normal cloud with {rho}, and one far point (pink)", n - 1), format!("{} points, {rho}, and one far point", n - 1)],
            Set::Anscombe(k) => vec![format!("Anscombe's quartet, set {}: {}", QUARTET[k], QUARTET_SHAPES[k]), format!("Anscombe's set {}: {}", QUARTET[k], QUARTET_SHAPES[k]), format!("Anscombe's set {}", QUARTET[k])],
            Set::Square => vec![r"$y = x^2$ at 21 evenly spaced $x$ from $-1$ to 1".to_string(), r"$y = x^2$, 21 points".to_string()],
            Set::Exp => vec![r"$y = e^x$ at 21 evenly spaced $x$ from 0 to 4".to_string(), r"$y = e^x$, 21 points".to_string()],
        };
        let what: Vec<String> = if self.edited { what.iter().map(|w| format!("{w}, points moved by hand")).chain(what.iter().cloned()).collect() } else { what };
        let room = lay.view.x - 2.0 * lay.head.x;
        if !lay.compact {
            let line = pick(f, &what, room, lay.big);
            f.text_px(lay.head, line, lay.big, pal::TEXT, Align2::LEFT_TOP);
        }
        let at = if lay.compact { lay.head } else { lay.head + Vec2::new(0.0, lay.big + 6.0) };
        let r = tint(ESTIMATE, "r");
        let count = if self.window { format!("$n = {}$ of {}", s.used.len(), n) } else { format!("$n = {}$", s.used.len()) };
        let numbers = match (s.r, s.spearman) {
            (Some(v), sp) => {
                let rank = sp.map_or("none".to_string(), |v| num(v, 2));
                let two = if s.used.len() == 2 { ": two points always lie on one line" } else { "" };
                vec![
                    format!(r"${r} = {}$      $r^2 = {}$      rank correlation {rank}      {count}{two}", num(v, 2), num(v * v, 2)),
                    format!(r"${r} = {}$    $r^2 = {}$    rank corr. {rank}    {count}", num(v, 2), num(v * v, 2)),
                    format!(r"${r} = {}$   $r^2 = {}$   {count}", num(v, 2), num(v * v, 2)),
                    format!(r"${r} = {}$", num(v, 2)),
                ]
            }
            (None, _) => {
                let why = match (s.used.len(), s.varies) {
                    (0, _) => "no points".to_string(),
                    (1, _) => "one point".to_string(),
                    (_, [false, false]) => r"neither $x$ nor $y$ varies".to_string(),
                    (_, [false, _]) => r"$x$ does not vary".to_string(),
                    _ => r"$y$ does not vary".to_string(),
                };
                vec![format!(r"${r}$: no correlation to speak of ({why})      {count}"), format!(r"${r}$: none ({why})"), format!(r"${r}$: none")]
            }
        };
        let line = pick(f, &numbers, room, lay.text);
        f.text_px(at, line, lay.text, pal::TEXT, Align2::LEFT_TOP);
    }

    fn draw_buttons(&mut self, f: &mut Frame, lay: &Layout, _pointer: Option<Vec2>) {
        let (at, right) = lay.buttons;
        let size = lay.small;
        let units = ["standard units", "own units"];
        let rects = button_row(f, &units, at.x, at.y, lay.button_h, size, 6.0);
        for (k, r) in rects.iter().enumerate() {
            if button(f, *r, units[k], size, self.standard == (k == 0)) {
                self.standard = k == 0;
            }
        }
        let after = rects.last().map_or(at.x, |r| r.1.x) + 14.0;
        if let Set::Anscombe(k) = self.set {
            let rects = row_ending(f, &QUARTET, right, at.y, lay.button_h, size);
            if rects.first().is_some_and(|r| r.0.x > after) {
                for (j, r) in rects.iter().enumerate() {
                    if button(f, *r, QUARTET[j], size, j == k) {
                        self.choose_set(Set::Anscombe(j));
                    }
                }
            }
        } else if self.set.random() {
            let rects = row_ending(f, &["draw again"], right, at.y, lay.button_h, size);
            if rects.first().is_some_and(|r| r.0.x > after) && button(f, rects[0], "draw again", size, false) {
                self.seed = self.seed.wrapping_add(1);
                if self.side == Side::Samples {
                    // (new samples are coming anyway: this one is the next of them)
                    self.samples.running = !self.samples.done();
                } else {
                    self.remake();
                }
            }
        }
    }

    /// The point whose product is drawn out: the one being dragged, else the one under
    /// the pointer, else the one with the largest product in the data as they are (so
    /// that it changes only when they do). As its place in `s.used`.
    fn focus(&self, pointer: Option<Vec2>, plot: &Plot, pts: &[[f64; 2]], s: &Summary) -> Option<usize> {
        s.r?;
        let place = |i: usize| s.used.iter().position(|&u| u == i);
        if let Some(i) = self.held {
            return place(i);
        }
        if let Some(p) = pointer.filter(|p| within(*p, plot.rect, 0.0)) {
            let near = s.used.iter().enumerate().map(|(j, &i)| (j, plot.at(pts[i][0], pts[i][1]).distance(p))).filter(|(_, d)| d.is_finite() && *d < 14.0).min_by(|a, b| a.1.total_cmp(&b.1));
            if let Some((j, _)) = near {
                return Some(j);
            }
        }
        if self.side == Side::Samples && self.samples.running {
            return None;
        }
        let truth = self.summary();
        let largest = truth.products.iter().enumerate().max_by(|a, b| a.1.abs().total_cmp(&b.1.abs()).then(b.0.cmp(&a.0)))?;
        place(*truth.used.get(largest.0)?)
    }

    fn draw_scatter(&self, f: &mut Frame, lay: &Layout, plot: &Plot, pts: &[[f64; 2]], s: &Summary, focus: Option<usize>) {
        let (min, max) = plot.rect;
        plot.backing(f, lay.numbers + 4.0, lay.small + 10.0, 8.0, 8.0);
        if plot.tiny() {
            return;
        }
        let both = s.varies[0] && s.varies[1];
        let centre = plot.at(s.mean[0], s.mean[1]);
        let c = Vec2::new(centre.x.max(min.x).min(max.x), centre.y.max(min.y).min(max.y));
        // the quadrants about the means: the products' signs
        if both && centre.is_finite() {
            // (faint: blending is done in linear light, where a small share of a bright
            // colour over black looks far brighter than its share)
            fill(f, (Vec2::new(c.x, min.y), Vec2::new(max.x, c.y)), PLUS.a(QUADRANT));
            fill(f, (Vec2::new(min.x, c.y), Vec2::new(c.x, max.y)), PLUS.a(QUADRANT));
            fill(f, (min, c), MINUS.a(QUADRANT));
            fill(f, (c, max), MINUS.a(QUADRANT));
            line_of(f, MEANS_KIND, Vec2::new(c.x, min.y), Vec2::new(c.x, max.y), 1.0, pal::DIM.a(0.45));
            line_of(f, MEANS_KIND, Vec2::new(min.x, c.y), Vec2::new(max.x, c.y), 1.0, pal::DIM.a(0.45));
        }
        // each point's product, as the rectangle from the centre to the point
        if both && self.products && s.used.len() <= PRODUCTS_MOST && centre.is_finite() {
            for (j, &i) in s.used.iter().enumerate() {
                let p = plot.at(pts[i][0], pts[i][1]);
                if !p.is_finite() {
                    continue;
                }
                let (a, b) = (centre.min(p).max(min).min(max), centre.max(p).max(min).min(max));
                let sign = s.products.get(j).copied().unwrap_or(0.0);
                fill(f, (a, b), if sign >= 0.0 { PLUS.a(PRODUCT) } else { MINUS.a(PRODUCT) });
            }
        }
        // one point's product drawn out: the rectangle, and the sum that makes it
        let mut taken: Option<Rect> = None;
        if let (Some(j), true) = (focus, both && centre.is_finite()) {
            let i = s.used[j];
            let p = plot.at(pts[i][0], pts[i][1]);
            let product = s.products.get(j).copied().unwrap_or(0.0);
            let color = if product >= 0.0 { PLUS } else { MINUS };
            // (only while the point is on the chart: a label kept beside a point off it,
            // as the axes glide in, would chase it)
            if p.is_finite() && within(p, plot.rect, 0.5) {
                let (a, b) = (centre.min(p).max(min).min(max), centre.max(p).max(min).min(max));
                fill(f, (a, b), color.a(0.06));
                let corners = [a, Vec2::new(b.x, a.y), b, Vec2::new(a.x, b.y), a];
                plot::path_of(f, FOCUS_KIND, &corners, 1.4, color.mix(pal::WHITE, 0.45));
                let (zx, zy) = (s.z(pts[i][0], 0), s.z(pts[i][1], 1));
                let away = if p.y <= centre.y { -1.0 } else { 1.0 };
                let label = format!(r"${} \times {} = {}$", num(zx, 2), num(zy, 2), num(product, 2));
                let size = f.text_size(&label, lay.small);
                // over the point if it is above the mean of y, under it if below (only ever
                // a short way to the other side), kept on the chart; and where that would
                // cover the point (at the top or the bottom), on its other side
                let half = size * 0.5 + Vec2::splat(4.0);
                let place = |dir: f32| {
                    let at = p + Vec2::new(0.0, dir * (10.0 + half.y));
                    Vec2::new(at.x.max(min.x + half.x).min(max.x - half.x), at.y.max(min.y + half.y).min(max.y - half.y))
                };
                let covers = |at: Vec2| (p.y - at.y).abs() < half.y + 4.0 && (p.x - at.x).abs() < half.x + 4.0;
                let at = if covers(place(away)) { place(-away) } else { place(away) };
                // (gliding to a new side, as the point crosses a mean, rather than hopping
                // there; on a plate, like a caption)
                f.text_placed("product", i as u32, p, at - p, label, lay.small, color.mix(pal::WHITE, 0.55));
                if let Some(placed) = f.dl.labels.last_mut() {
                    placed.backed = true;
                }
                taken = Some((at - half, at + half));
            }
        }
        // the signs of the products, in the corners
        if both {
            let o = 10.0;
            for (corner, plus, align) in [
                (Vec2::new(max.x - o, min.y + o), true, Align2::RIGHT_TOP),
                (Vec2::new(min.x + o, min.y + o), false, Align2::LEFT_TOP),
                (Vec2::new(min.x + o, max.y - o), true, Align2::LEFT_BOTTOM),
                (Vec2::new(max.x - o, max.y - o), false, Align2::RIGHT_BOTTOM),
            ] {
                let (text, color) = if plus { ("product +", PLUS) } else { ("product −", MINUS) };
                let text = if lay.narrow { if plus { "+" } else { "−" } } else { text };
                // (not under the product drawn out)
                let size = f.text_size(text, lay.small);
                let lo = corner - Vec2::new(if align == Align2::RIGHT_TOP || align == Align2::RIGHT_BOTTOM { size.x } else { 0.0 }, if align == Align2::LEFT_BOTTOM || align == Align2::RIGHT_BOTTOM { size.y } else { 0.0 });
                // (an empty one in its place when covered: the labels keep their order)
                let covered = taken.is_some_and(|(a, b)| lo.x < b.x && lo.x + size.x > a.x && lo.y < b.y && lo.y + size.y > a.y);
                f.text_px(corner, if covered { "" } else { text }, lay.small, color.mix(pal::WHITE, 0.25), align);
            }
        }
        // the curve the points come from, across the x they were made at (at fixed x,
        // so that it moves evenly when the axes do)
        if let Some(g) = self.set.curve() {
            let (lo, hi) = self.pts.iter().fold((f64::MAX, f64::MIN), |(a, b), p| (a.min(p[0]), b.max(p[0])));
            if hi > lo {
                // (kept on the chart, as the points are, while the axes glide)
                let m = 160;
                let curve: Vec<Vec2> = (0..=m).map(|i| lo + (hi - lo) * i as f64 / m as f64).map(|x| plot.at(x, g(x)).max(min).min(max)).collect();
                plot::path_of(f, CURVES, &curve, 1.6, TRUTH.a(0.75));
            }
        }
        // the window: the median of x, and what is left of it
        if self.window {
            plot.vline(f, MEDIAN_KIND, s.median_x, 1.3, pal::DIM, true);
            let x = plot.px_x(s.median_x);
            if x >= min.x && x <= max.x {
                f.text_px(Vec2::new(x + 5.0, min.y + 22.0), "median of $x$", lay.small, pal::DIM, Align2::LEFT_TOP);
            }
        }
        // the top tenth: where it starts, and its average
        if self.side == Side::Tenth {
            if let (Some(_), 3) = (top_tenth(pts, s, self.by).1, self.tenth.shown.len()) {
                let by = self.by;
                let (cut, avg) = (self.tenth.get(0) as f64, [self.tenth.get(1) as f64, self.tenth.get(2) as f64]);
                if by == 0 {
                    plot.vline(f, TENTH_KIND, cut, 1.2, TOP.a(0.5), true);
                } else {
                    plot.hline(f, TENTH_KIND, cut, 1.2, TOP.a(0.5), true);
                }
                let at = plot.at(s.mean[0] + avg[0] * s.unit(0), s.mean[1] + avg[1] * s.unit(1));
                if at.is_finite() {
                    // (waiting at the edge while the axes glide, like the points)
                    let at = at.max(min).min(max);
                    let arm = 9.0;
                    line_of(f, TENTH_KIND, at - Vec2::new(arm, 0.0), at + Vec2::new(arm, 0.0), 2.4, TOP);
                    line_of(f, TENTH_KIND, at - Vec2::new(0.0, arm), at + Vec2::new(0.0, arm), 2.4, TOP);
                    f.caption_px(at + Vec2::new(12.0, 10.0), "their average", lay.small, TOP, Align2::LEFT_TOP);
                }
            }
        }
        // the three lines through the centre
        if let (Some(dirs), true) = (s.lines(), centre.is_finite()) {
            for (k, kind, color) in [(0usize, SD_KIND, SD_LINE.a(0.9)), (1, YX_KIND, ESTIMATE), (2, XY_KIND, OTHER)] {
                let Some(d) = dirs[k] else { continue };
                // (as drawn: the SD line turning from one sign to the other, and the
                // regression lines at their gliding angles)
                let d = match k {
                    0 => [d[0], self.sd_sign.get(0).max(-1.0).min(1.0) as f64 * s.sd[1]],
                    _ if self.angles.shown.len() == 2 => {
                        let a = self.angles.get(k - 1) as f64;
                        [a.cos() * s.sd[0], a.sin() * s.sd[1]]
                    }
                    _ => d,
                };
                if !self.lines[k] {
                    continue;
                }
                let q = plot.at(s.mean[0] + d[0], s.mean[1] + d[1]);
                let dir = (q - centre).normalize_or_zero();
                if dir == Vec2::ZERO {
                    continue;
                }
                let reach = (max - min).length() + (centre - c).length() + 10.0;
                let ends = plot::clip(&[centre - dir * reach, centre, centre + dir * reach], plot.rect);
                let ends: Vec<Vec2> = ends.into_iter().filter(|p| p.is_finite()).collect();
                if let (Some(a), Some(b)) = (ends.first(), ends.last()) {
                    if k == 0 {
                        dashed_of(f, kind, *a, *b, 1.6, color);
                    } else {
                        line_of(f, kind, *a, *b, if k == 1 { 2.6 } else { 2.0 }, color);
                    }
                }
            }
        }
        // with the window: the regression line of all the points, to compare
        if self.window && self.lines[1] {
            let all = summarise(pts, false);
            if let (Some([_, Some(d), _]), true) = (all.lines(), self.window) {
                let from = plot.at(all.mean[0], all.mean[1]);
                let dir = (plot.at(all.mean[0] + d[0], all.mean[1] + d[1]) - from).normalize_or_zero();
                if dir != Vec2::ZERO && from.is_finite() {
                    let reach = (max - min).length() * 2.0 + (from - c).length();
                    let ends: Vec<Vec2> = plot::clip(&[from - dir * reach, from, from + dir * reach], plot.rect).into_iter().filter(|p| p.is_finite()).collect();
                    if let (Some(a), Some(b)) = (ends.first(), ends.last()) {
                        dashed_of(f, ALL_KIND, *a, *b, 1.8, ESTIMATE.a(0.75));
                    }
                }
            }
        }
        // the points (as handles while there are few: see `handles`)
        if self.pts.len() > DRAG_MOST {
            let colors = self.colors(pts, s);
            let size = self.point_size();
            for (i, p) in pts.iter().enumerate() {
                let at = plot.at(p[0], p[1]);
                if at.is_finite() {
                    disc(f, at.max(min).min(max), size, colors[i].a(if colors[i] == DATA { 0.8 } else { 1.0 }));
                }
            }
        }
        self.draw_axes(f, lay, plot, s);
    }

    /// The scatter's axes: their round values (in standard units or the data's own),
    /// every one drawn where the axes put it now, those off the chart unseen at its
    /// edge; and their names.
    fn draw_axes(&self, f: &mut Frame, lay: &Layout, plot: &Plot, s: &Summary) {
        let (min, max) = plot.rect;
        line_of(f, AXES, Vec2::new(min.x, max.y), max, 1.0, pal::DIM.a(0.8));
        line_of(f, AXES, min, Vec2::new(min.x, max.y), 1.0, pal::DIM.a(0.8));
        let Some(ticks) = &self.ticks else { return };
        for (k, t) in ticks.iter().enumerate() {
            for v in t.values() {
                let data = if self.standard { s.mean[k] + v * s.unit(k) } else { v };
                if k == 0 {
                    let x = plot.px_x(data);
                    let seen = x >= min.x - 0.5 && x <= max.x + 0.5;
                    let x = if x.is_finite() { x.max(min.x).min(max.x) } else { min.x };
                    line_of(f, AXES, Vec2::new(x, max.y), Vec2::new(x, max.y + 4.0), 1.0, pal::DIM.a(if seen { 0.8 } else { 0.0 }));
                    f.text_px(Vec2::new(x, max.y + 6.0), if seen { t.name(v) } else { String::new() }, lay.small, pal::DIM, Align2::CENTER_TOP);
                } else {
                    let y = plot.px_y(data);
                    let seen = y >= min.y - 0.5 && y <= max.y + 0.5;
                    let y = if y.is_finite() { y.max(min.y).min(max.y) } else { max.y };
                    line_of(f, AXES, Vec2::new(min.x - 4.0, y), Vec2::new(min.x, y), 1.0, pal::DIM.a(if seen { 0.8 } else { 0.0 }));
                    f.text_px(Vec2::new(min.x - 6.0, y), if seen { t.name(v) } else { String::new() }, lay.small, pal::DIM, Align2::RIGHT_CENTER);
                }
            }
        }
        // (a variable that does not vary has no standard units: its distance from its mean is drawn)
        let name = |k: usize| match (self.standard, s.varies[k]) {
            (true, true) => format!("${}$ in standard units", ["x", "y"][k]),
            (true, false) => format!(r"${v} - ar {v}$ (${v}$ does not vary)", v = ["x", "y"][k]),
            _ => format!("${}$", ["x", "y"][k]),
        };
        let (xn, yn) = (name(0), name(1));
        f.text_px(Vec2::new((min.x + max.x) * 0.5, max.y + 8.0 + lay.small), xn, lay.small, pal::DIM, Align2::CENTER_TOP);
        f.text_px(Vec2::new(min.x - lay.numbers + 4.0, min.y - 10.0), yn, lay.small, pal::DIM, Align2::LEFT_BOTTOM);
    }

    /// The names of the lines, under the scatter.
    fn draw_legend(&self, f: &mut Frame, lay: &Layout, s: &Summary) {
        let Some(r) = s.r else { return };
        let size = lay.small;
        let b = r * s.sd[1] / s.sd[0];
        let names = if self.standard {
            [
                if r > 1e-12 {
                    "SD line: slope 1".to_string()
                } else if r < -1e-12 {
                    "SD line: slope −1".to_string()
                } else {
                    r"SD line: none ($r = 0$ has no sign)".to_string()
                },
                format!(r"$y$ on $x$: slope $r = {}$", num(r, 2)),
                if r.abs() > 1e-12 { format!(r"$x$ on $y$: slope $1/r = {}$", num(1.0 / r, 2)) } else { r"$x$ on $y$: upright ($r = 0$)".to_string() },
            ]
        } else {
            let d = r * s.sd[0] / s.sd[1];
            let sign = |v: f64| if v < 0.0 { "-" } else { "+" };
            [
                if r < 0.0 { format!(r"SD line: slope $-s_y/s_x = {}$", num(-s.sd[1] / s.sd[0], 2)) } else { format!(r"SD line: slope $s_y/s_x = {}$", num(s.sd[1] / s.sd[0], 2)) },
                format!(r"$y$ on $x$: $y = {} {} {}x$", num(s.mean[1] - b * s.mean[0], 2), sign(b), num(b.abs(), 3)),
                if r.abs() > 1e-12 { format!(r"$x$ on $y$: $x = {} {} {}y$", num(s.mean[0] - d * s.mean[1], 2), sign(d), num(d.abs(), 3)) } else { format!(r"$x$ on $y$: $x = {}$", num(s.mean[0], 2)) },
            ]
        };
        let colors = [SD_LINE.a(0.8), ESTIMATE, OTHER];
        let (mut x, mut y) = (lay.legend.x, lay.legend.y);
        let right = lay.legend.x + lay.legend_w;
        for k in [1, 2, 0] {
            if !self.lines[k] {
                continue;
            }
            let w = 22.0 + f.text_size(&names[k], size).x;
            if x > lay.legend.x && x + w > right {
                x = lay.legend.x;
                y += size + 6.0;
            }
            let mid = y + size * 0.5 + 1.0;
            if k == 0 {
                dashed_of(f, LEGEND_KIND, Vec2::new(x, mid), Vec2::new(x + 16.0, mid), 1.6, colors[k]);
            } else {
                line_of(f, LEGEND_KIND, Vec2::new(x, mid), Vec2::new(x + 16.0, mid), 2.4, colors[k]);
            }
            f.text_px(Vec2::new(x + 22.0, y), names[k].as_str(), size, pal::DIM, Align2::LEFT_TOP);
            x += w + 18.0;
        }
        if self.window && self.lines[1] {
            let text = r"dashed: $y$ on $x$ for all the points";
            let w = 22.0 + f.text_size(text, size).x;
            if x > lay.legend.x && x + w > right {
                x = lay.legend.x;
                y += size + 6.0;
            }
            let mid = y + size * 0.5 + 1.0;
            dashed_of(f, LEGEND_KIND, Vec2::new(x, mid), Vec2::new(x + 16.0, mid), 1.8, ESTIMATE.a(0.75));
            f.text_px(Vec2::new(x + 22.0, y), text, size, pal::DIM, Align2::LEFT_TOP);
        }
    }

    /// r as the average of the products: each point's product a piece of a bar, the
    /// positive ones to the right of 0 and the negative ones to the left, each 1/n of
    /// its size; r is where the two come to together.
    fn draw_products(&self, f: &mut Frame, lay: &Layout, s: &Summary, focus: Option<usize>) {
        let (min, max) = lay.bar;
        hud_panel(f, min - Vec2::splat(8.0), max + Vec2::splat(8.0));
        let width = max.x - min.x;
        let n = s.products.len();
        let Some(r) = s.r else {
            let why = if n < 2 { "too few points for a correlation" } else { "no correlation to speak of: a variable that does not vary has no standard units" };
            wrap(f, min, width, max.y, why, lay.text, pal::DIM);
            return;
        };
        let title = [
            format!(r"$r = {}$: the average of the {n} products, laid end to end", num(r, 2)),
            format!(r"$r = {}$: the average of the {n} products", num(r, 2)),
            format!(r"$r = {}$: the products, averaged", num(r, 2)),
        ];
        let title = pick(f, &title, width, lay.text);
        f.text_px(min, title, lay.text, pal::TEXT, Align2::LEFT_TOP);
        let (top, bottom) = (min.y + lay.text + 10.0, min.y + lay.text + 30.0);
        let x_of = |v: f64| min.x + width * ((v + 1.0) * 0.5) as f32;
        fill(f, (Vec2::new(min.x, top), Vec2::new(max.x, bottom)), Color::hex(0x161C2B).a(0.9));
        // (out from 0: the point drawn out in the scatter first, so that its piece grows
        // and shrinks in place as it is dragged, then the rest, largest first; the pieces
        // told apart by their shades)
        let mut order: Vec<usize> = (0..n).collect();
        order.sort_by(|&a, &b| (Some(b) == focus).cmp(&(Some(a) == focus)).then(s.products[b].abs().total_cmp(&s.products[a].abs())).then(a.cmp(&b)));
        let (mut right, mut left) = (0.0f64, 0.0f64);
        let (mut kr, mut kl) = (0usize, 0usize);
        let mut marked = None;
        for j in order {
            let p = s.products[j];
            let piece = p / n.max(1) as f64;
            let (a, b) = if p > 0.0 {
                let shade = if kr % 2 == 0 { 0.95 } else { 0.62 };
                fill(f, (Vec2::new(x_of(right), top), Vec2::new(x_of(right + piece), bottom)), PLUS.a(shade));
                right += piece;
                kr += 1;
                (right - piece, right)
            } else if p < 0.0 {
                let shade = if kl % 2 == 0 { 0.95 } else { 0.62 };
                fill(f, (Vec2::new(x_of(left + piece), top), Vec2::new(x_of(left), bottom)), MINUS.a(shade));
                left += piece;
                kl += 1;
                (left, left - piece)
            } else {
                (0.0, 0.0)
            };
            if focus == Some(j) {
                marked = Some((a, b));
            }
        }
        // the piece of the point drawn out in the scatter
        if let Some((a, b)) = marked {
            let (x0, x1) = (x_of(a), x_of(b).max(x_of(a) + 1.0));
            let c = [Vec2::new(x0, top - 2.0), Vec2::new(x1, top - 2.0), Vec2::new(x1, bottom + 2.0), Vec2::new(x0, bottom + 2.0), Vec2::new(x0, top - 2.0)];
            plot::path_of(f, FOCUS_KIND, &c, 1.4, pal::WHITE);
        }
        line_of(f, BAR_KIND, Vec2::new(x_of(0.0), top - 3.0), Vec2::new(x_of(0.0), bottom + 3.0), 1.0, pal::DIM);
        let at = x_of(r);
        line_of(f, BAR_KIND, Vec2::new(at, top - 6.0), Vec2::new(at, bottom + 6.0), 2.6, ESTIMATE);
        let below = bottom + 6.0;
        let (plus, minus) = s.halves();
        f.text_px(Vec2::new(min.x, below), "−1", lay.small, pal::DIM, Align2::LEFT_TOP);
        f.text_px(Vec2::new(max.x, below), "1", lay.small, pal::DIM, Align2::RIGHT_TOP);
        f.text_px(Vec2::new(x_of(-0.5), below), format!("negative: {}", num(minus, 2)), lay.small, MINUS.mix(pal::WHITE, 0.3), Align2::CENTER_TOP);
        f.text_px(Vec2::new(x_of(0.5), below), format!("positive: {}", num(plus, 2)), lay.small, PLUS.mix(pal::WHITE, 0.3), Align2::CENTER_TOP);
    }

    /// r² as a share: of the variance of y, what the regression line on x accounts for
    /// and what is left over about it.
    fn draw_share(&self, f: &mut Frame, lay: &Layout, s: &Summary) {
        let (min, max) = lay.share;
        hud_panel(f, min - Vec2::splat(8.0), max + Vec2::splat(8.0));
        let width = max.x - min.x;
        let Some(r2) = s.r2() else {
            f.text_px(min, "$r^2$: none", lay.text, pal::DIM, Align2::LEFT_TOP);
            return;
        };
        let title = [
            format!(r"$r^2 = {}$: the share of the variance of $y$ that the yellow line accounts for", num(r2, 2)),
            format!(r"$r^2 = {}$: the share of $y$'s variance the yellow line accounts for", num(r2, 2)),
            format!(r"$r^2 = {}$: the yellow line's share of $y$'s variance", num(r2, 2)),
            format!(r"$r^2 = {}$", num(r2, 2)),
        ];
        let title = pick(f, &title, width, lay.text);
        f.text_px(min, title, lay.text, pal::TEXT, Align2::LEFT_TOP);
        let (top, bottom) = (min.y + lay.text + 8.0, min.y + lay.text + 22.0);
        let split = min.x + width * r2 as f32;
        fill(f, (Vec2::new(min.x, top), Vec2::new(split, bottom)), ESTIMATE.a(0.8));
        fill(f, (Vec2::new(split, top), Vec2::new(max.x, bottom)), Color::hex(0x2A3142).a(0.95));
        let below = bottom + 6.0;
        f.text_px(Vec2::new(min.x, below), format!("accounted for: {} %", num(100.0 * r2, 0)), lay.small, ESTIMATE, Align2::LEFT_TOP);
        f.text_px(Vec2::new(max.x, below), format!("left over about the line: {} %", num(100.0 * (1.0 - r2), 0)), lay.small, pal::DIM, Align2::RIGHT_TOP);
    }

    /// The top tenth, in standard units: how far above the mean they are on the
    /// variable they were chosen by, and on the other.
    fn draw_tenth(&mut self, f: &mut Frame, lay: &Layout, pts: &[[f64; 2]], s: &Summary) {
        let (min, max) = lay.side;
        hud_panel(f, min - Vec2::splat(8.0), max + Vec2::splat(8.0));
        let names = ["x", "y"];
        let (a, b) = (self.by.min(1), 1 - self.by.min(1));
        let words = ["by $x$", "by $y$"];
        let rects = row_ending(f, &words, max.x, min.y - 3.0, lay.text + 8.0, lay.small);
        for (k, rect) in rects.iter().enumerate() {
            if button(f, *rect, words[k], lay.small, self.by == k) {
                self.by = k;
            }
        }
        let room = rects.first().map_or(max.x, |r| r.0.x) - min.x - 8.0;
        let title = [format!("the top tenth on ${}$ (white), in standard units", names[a]), format!("the top tenth on ${}$", names[a])];
        let title = pick(f, &title, room, lay.text);
        f.text_px(min, title, lay.text, pal::TEXT, Align2::LEFT_TOP);
        let (top, avg) = top_tenth(pts, s, self.by);
        let Some(avg) = avg else {
            wrap(f, min + Vec2::new(0.0, lay.text + 14.0), max.x - min.x, max.y, "No top tenth to speak of: a variable that does not vary has no standard units.", lay.small, pal::DIM);
            return;
        };
        let r = s.r.unwrap_or(0.0);
        // (placed where the glide has them; the numbers are the points' own)
        let glided = if self.tenth.shown.len() == 3 { [self.tenth.get(1) as f64, self.tenth.get(2) as f64] } else { avg };
        let label_w = if lay.narrow { 54.0 } else { 70.0 };
        let (x0, x1) = (min.x + label_w, max.x - 14.0);
        // (from a little below the mean to 3, wider where the numbers need it: below the
        // mean for a negative r)
        let values = [glided[a], glided[b], r * glided[a]];
        let lo = values.iter().fold(-0.25f64, |m, v| m.min(v - 0.3)).max(-40.0);
        let hi = values.iter().fold(3.0f64, |m, v| m.max(v + 0.3)).min(40.0);
        let x_of = |v: f64| x0 + (x1 - x0) * ((v.max(lo).min(hi) - lo) / (hi - lo)) as f32;
        let row = (lay.small + 26.0).max(34.0);
        let ya = min.y + lay.text + 30.0;
        let yb = ya + row + 10.0;
        let count = if top.len() == 1 { "1 point".to_string() } else { format!("{} points", top.len()) };
        f.text_px(Vec2::new(min.x, ya), format!("their ${}$", names[a]), lay.small, TOP, Align2::LEFT_CENTER);
        f.text_px(Vec2::new(min.x, yb), format!("their ${}$", names[b]), lay.small, TOP, Align2::LEFT_CENTER);
        for y in [ya, yb] {
            line_of(f, SIDE_KIND, Vec2::new(x0, y), Vec2::new(x1, y), 1.0, pal::DIM.a(0.7));
            line_of(f, SIDE_KIND, Vec2::new(x_of(0.0), y - 5.0), Vec2::new(x_of(0.0), y + 5.0), 1.0, pal::DIM);
        }
        // straight down from the first: as far from the mean on the other (the SD line's guess)
        let (pa, pb, guess) = (x_of(glided[a]), x_of(glided[b]), x_of(r * glided[a]));
        dashed_of(f, SIDE_KIND, Vec2::new(pa, ya + 6.0), Vec2::new(pa, yb), 1.4, SD_LINE.a(0.8));
        line_of(f, SIDE_KIND, Vec2::new(guess, yb - 9.0), Vec2::new(guess, yb + 9.0), 2.6, ESTIMATE);
        disc(f, Vec2::new(pa, ya), 10.0, TOP);
        disc(f, Vec2::new(pb, yb), 10.0, TOP);
        f.text_px(Vec2::new(pa, ya - 8.0), format!("{} ({count})", num(avg[a], 2)), lay.small, TOP, Align2::CENTER_BOTTOM);
        f.text_px(Vec2::new(pb, yb + 9.0), num(avg[b], 2), lay.small, TOP, Align2::CENTER_TOP);
        // (the yellow line's name on whichever side of its mark keeps it off the dashed line)
        let label = format!(r"yellow line: $r \times {} = {}$", num(avg[a], 2), num(r * avg[a], 2));
        let w = f.text_size(&label, lay.small).x;
        let (at, align) = if guess - 6.0 - w >= min.x && guess < pa { (guess - 6.0, Align2::RIGHT_BOTTOM) } else { (guess + 6.0, Align2::LEFT_BOTTOM) };
        f.text_px(Vec2::new(at, yb - 10.0), label, lay.small, ESTIMATE, align);
        // the scale, under both rows
        let axis_y = yb + 9.0 + lay.small + 8.0;
        for v in (lo.ceil() as i64)..=(hi.floor() as i64) {
            f.text_px(Vec2::new(x_of(v as f64), axis_y), num(v as f64, 0), lay.small, pal::DIM, Align2::CENTER_TOP);
        }
        f.text_px(Vec2::new(x_of(0.0) - 8.0, axis_y), "mean", lay.small, pal::DIM, Align2::RIGHT_TOP);
        let text = format!(
            "The yellow line, the best straight-line prediction of ${b}$ from ${a}$, puts them $r$ times as many standard deviations from the mean of ${b}$ as they are from that of ${a}$ (on its other side if $r < 0$): less extreme, whenever $|r| < 1$. The SD line (purple, straight down) would put them as far.",
            a = names[a],
            b = names[b]
        );
        wrap(f, Vec2::new(min.x, axis_y + lay.small + 14.0), max.x - min.x, max.y, &text, lay.small, pal::DIM);
    }

    /// The ranks of the points, against each other: Spearman's correlation is r of these.
    fn draw_ranks(&self, f: &mut Frame, lay: &Layout, pts: &[[f64; 2]], s: &Summary) {
        let (min, max) = lay.side;
        hud_panel(f, min - Vec2::splat(8.0), max + Vec2::splat(8.0));
        let width = max.x - min.x;
        let value = s.spearman.map_or("none".to_string(), |v| num(v, 2));
        let title = [format!("ranks: the rank correlation is $r$ of these, {value}"), format!("ranks: rank correlation {value}")];
        let title = pick(f, &title, width, lay.text);
        f.text_px(min, title, lay.text, pal::TEXT, Align2::LEFT_TOP);
        let n = s.used.len();
        let side = (width - 40.0).min(max.y - min.y - lay.text - 60.0).max(10.0);
        let left = min.x + 34.0 + ((width - 40.0 - side) * 0.5).max(0.0);
        let rect = (Vec2::new(left, min.y + lay.text + 30.0), Vec2::new(left + side, min.y + lay.text + 30.0 + side));
        let p = Plot::new(rect, [0.5, n as f64 + 0.5], [0.5, n as f64 + 0.5]);
        if p.tiny() || n == 0 {
            return;
        }
        let (min_r, max_r) = rect;
        line_of(f, SIDE_KIND, Vec2::new(min_r.x, max_r.y), max_r, 1.0, pal::DIM.a(0.8));
        line_of(f, SIDE_KIND, min_r, Vec2::new(min_r.x, max_r.y), 1.0, pal::DIM.a(0.8));
        f.text_px(Vec2::new(p.px_x(1.0), max_r.y + 5.0), "1", lay.small, pal::DIM, Align2::CENTER_TOP);
        f.text_px(Vec2::new(p.px_x(n as f64), max_r.y + 5.0), n.to_string(), lay.small, pal::DIM, Align2::CENTER_TOP);
        f.text_px(Vec2::new(min_r.x - 5.0, p.px_y(1.0)), "1", lay.small, pal::DIM, Align2::RIGHT_CENTER);
        f.text_px(Vec2::new(min_r.x - 5.0, p.px_y(n as f64)), n.to_string(), lay.small, pal::DIM, Align2::RIGHT_CENTER);
        f.text_px(Vec2::new((min_r.x + max_r.x) * 0.5, max_r.y + 5.0), "rank of $x$", lay.small, pal::DIM, Align2::CENTER_TOP);
        f.text_px(Vec2::new(min_r.x - 28.0, min_r.y - 6.0), "rank of $y$", lay.small, pal::DIM, Align2::LEFT_BOTTOM);
        let x: Vec<f64> = s.used.iter().map(|&i| pts[i][0]).collect();
        let y: Vec<f64> = s.used.iter().map(|&i| pts[i][1]).collect();
        let (mut rx, mut ry) = (st::ranks(&x), st::ranks(&y));
        // (each point where its ranks are gliding: a rank changes in a step as a point
        // passes another)
        if self.ranks.shown.len() == 2 * n {
            for j in 0..n {
                (rx[j], ry[j]) = (self.ranks.get(2 * j) as f64, self.ranks.get(2 * j + 1) as f64);
            }
        }
        if let Some(sp) = s.spearman {
            let (mx, my) = (st::mean(&rx), st::mean(&ry));
            let slope = sp * (st::variance_n(&ry) / st::variance_n(&rx)).sqrt();
            p.path(f, RANK_KIND, [(0.5, my + slope * (0.5 - mx)), (n as f64 + 0.5, my + slope * (n as f64 + 0.5 - mx))], 2.0, ESTIMATE.a(0.9));
        }
        let size = if n <= 60 { 6.0 } else if n <= 200 { 4.0 } else { 2.6 };
        let colors = self.colors(pts, s);
        for (j, &i) in s.used.iter().enumerate() {
            p.dot(f, rx[j], ry[j], size, colors[i]);
        }
    }

    /// r over new samples of the cloud: a histogram, how it is spread in the long run,
    /// and how many are beyond ±0.5.
    fn draw_samples(&self, f: &mut Frame, lay: &Layout) {
        let (min, max) = lay.side;
        hud_panel(f, min - Vec2::splat(8.0), max + Vec2::splat(8.0));
        let width = max.x - min.x;
        let t = &self.samples;
        let n = self.n;
        let rho = num(self.rho as f64, 2);
        let title = [format!(r"$r$ in {} new samples of {n} points, $\rho = {rho}$", t.total), format!(r"$r$ in {} new samples", t.total)];
        let title = pick(f, &title, width, lay.text);
        f.text_px(min, title, lay.text, pal::TEXT, Align2::LEFT_TOP);
        let foot = 2.0 * (lay.small + 5.0) + 26.0;
        let rect = (Vec2::new(min.x + 10.0, min.y + lay.text + 22.0), Vec2::new(max.x - 10.0, (max.y - foot).max(min.y + lay.text + 40.0)));
        let heights: Vec<f64> = (0..BINS).map(|i| self.bars.get(i) as f64).collect();
        let width_bin = 2.0 / BINS as f64;
        let density: Vec<f64> = (0..=80).map(|i| -1.0 + 2.0 * i as f64 / 80.0).map(|r| r_density(r, self.rho as f64, n)).collect();
        let tallest = heights.iter().copied().fold(0.0, f64::max).max(density[4..77].iter().copied().fold(0.0, f64::max)).max(0.5);
        let p = Plot::new(rect, [-1.0, 1.0], [0.0, tallest * 1.12]);
        if p.tiny() {
            return;
        }
        for (i, h) in heights.iter().enumerate() {
            let lo = -1.0 + i as f64 * width_bin;
            let mid = lo + 0.5 * width_bin;
            let color = if mid.abs() > TAIL { ALARM.a(0.75) } else { ESTIMATE.a(0.55) };
            p.bar(f, lo, lo + width_bin, 0.0, *h, 1.0, color);
        }
        if n >= 3 && self.rho.abs() < 1.0 {
            p.path(f, DENSITY_KIND, (0..=80).map(|i| (-1.0 + 2.0 * i as f64 / 80.0, density[i])).filter(|(_, d)| d.is_finite()), 1.8, TRUTH);
        }
        p.vline(f, MARKS, self.rho as f64, 1.4, TRUTH.a(0.8), true);
        if t.total > 0 {
            p.vline(f, NOW_KIND, self.bars.get(BINS) as f64, 2.4, ESTIMATE, false);
        }
        p.x_axis(f, "", lay.small);
        for v in [-TAIL, TAIL] {
            p.vline(f, MARKS, v, 1.0, ALARM.a(0.6), true);
        }
        let share = if t.total > 0 { 100.0 * t.beyond as f64 / t.total as f64 } else { 0.0 };
        let long = if n >= 3 && self.rho.abs() < 1.0 {
            let inside = r_probability(-TAIL, TAIL, self.rho as f64, n);
            format!("{} %", num(100.0 * (1.0 - inside), 0))
        } else if n == 2 {
            "all of them: two points are always on a line".to_string()
        } else {
            "all of them".to_string()
        };
        let y = rect.1.y + lay.small + 14.0;
        let lines = [
            format!(r"beyond $\pm 0.5$ (pink): {} of {} ({} %)", t.beyond, t.total, num(share, 0)),
            format!("in the long run, for normal data (green curve): {long}"),
        ];
        for (k, line) in lines.iter().enumerate() {
            let color = if k == 0 { ALARM.mix(pal::WHITE, 0.3) } else { TRUTH.mix(pal::WHITE, 0.2) };
            let text = pick(f, &[line.clone(), if k == 0 { format!("beyond ±0.5: {} of {}", t.beyond, t.total) } else { format!("in the long run: {long}") }], width, lay.small);
            f.text_px(Vec2::new(min.x, y + k as f32 * (lay.small + 5.0)), text, lay.small, color, Align2::LEFT_TOP);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn column(p: &[[f64; 2]], k: usize) -> Vec<f64> {
        p.iter().map(|q| q[k]).collect()
    }

    fn r_of(p: &[[f64; 2]]) -> f64 {
        st::correlation(&column(p, 0), &column(p, 1))
    }

    /// An experiment as its button sets it up.
    fn trial(k: usize) -> Correlation {
        let mut s = Correlation::create();
        s.set_up(k);
        s
    }

    /// r as the average product of standard scores is the usual formula's r, and the
    /// regression slopes in standard units are r and 1/r.
    #[test]
    fn r_is_the_average_product() {
        for (set, rho, n, seed) in [(Set::Cloud, 0.6, 40, OPENING_SEED), (Set::Cloud, -0.3, 7, 3), (Set::Far, 0.0, 21, FAR_SEED), (Set::Anscombe(2), 0.0, 11, 0), (Set::Exp, 0.0, 21, 0), (Set::Cloud, 0.9, 1000, 5)] {
            let p = make(set, rho, n, seed);
            let s = summarise(&p, false);
            let r = s.r.expect("an r");
            assert!((r - r_of(&p)).abs() < 1e-12, "{set:?}: {r} against {}", r_of(&p));
            let (plus, minus) = s.halves();
            assert!((plus + minus - r).abs() < 1e-12);
            // the lines' directions in standard units: (1, r) and (r, 1)
            let [_, yx, xy] = s.lines().expect("lines");
            let (yx, xy) = (yx.expect("y on x"), xy.expect("x on y"));
            assert!((yx[1] / s.sd[1] / (yx[0] / s.sd[0]) - r).abs() < 1e-12);
            assert!((xy[1] / s.sd[1] / (xy[0] / s.sd[0]) - 1.0 / r).abs() < 1e-9 * (1.0 + 1.0 / r.abs()));
            // the y-on-x line is least squares: its slope in own units is the fitted one
            let (b, _) = crate::math::causal::line(&column(&p, 0), &column(&p, 1));
            assert!((yx[1] / yx[0] - b).abs() < 1e-9 * (1.0 + b.abs()));
            // and r² is the share of y's variance it accounts for
            let a = s.mean[1] - b * s.mean[0];
            let left: f64 = p.iter().map(|q| (q[1] - a - b * q[0]).powi(2)).sum::<f64>() / p.len() as f64;
            assert!((1.0 - left / st::variance_n(&column(&p, 1)) - r * r).abs() < 1e-9);
            // the rank correlation is r of the ranks
            let sp = st::correlation(&st::ranks(&column(&p, 0)), &st::ranks(&column(&p, 1)));
            assert!((s.spearman.expect("ranks") - sp).abs() < 1e-12);
        }
    }

    /// Anscombe's quartet: the same means, spreads, r and line, to two decimals; and
    /// different rank correlations.
    #[test]
    fn anscombes_numbers() {
        let mut ranks = Vec::new();
        for k in 0..4 {
            let p = make(Set::Anscombe(k), 0.0, 11, 0);
            let (x, y) = (column(&p, 0), column(&p, 1));
            let (b, a) = crate::math::causal::line(&x, &y);
            assert_eq!((num(st::mean(&x), 2), num(st::mean(&y), 2)), ("9.00".to_string(), "7.50".to_string()), "set {k}");
            assert_eq!((num(st::variance(&x), 2), num(r_of(&p), 2)), ("11.00".to_string(), "0.82".to_string()), "set {k}");
            assert!((r_of(&p) - 0.8164).abs() < 2e-4, "set {k}: {}", r_of(&p));
            assert!(["4.12", "4.13"].contains(&num(st::variance(&y), 2).as_str()), "set {k}: {}", st::variance(&y));
            assert_eq!((num(a, 2), num(b, 3)), ("3.00".to_string(), "0.500".to_string()), "set {k}");
            ranks.push(num(summarise(&p, false).spearman.expect("ranks"), 2));
        }
        assert_eq!(ranks, ["0.82", "0.69", "0.99", "0.50"]);
    }

    /// The captions of the things to try, held to account on the data their buttons make.
    #[test]
    fn the_experiments_show_what_their_captions_say() {
        // four data sets, one r: shown in their own units, the ranks beside them
        let s = trial(0);
        assert!(s.set == Set::Anscombe(0) && !s.standard && s.side == Side::Ranks);
        assert_eq!(TRIALS.len(), 7);

        // a curve with r = 0: every y is x², r is 0 (to rounding), each product cancelled by its mirror's
        let s = trial(1);
        let sum = s.summary();
        assert!(s.pts.iter().all(|p| p[1] == p[0] * p[0]));
        assert!(sum.r.expect("an r").abs() < 1e-12 && num(sum.r.unwrap_or(1.0), 2) == "0.00");
        for (j, &i) in sum.used.iter().enumerate() {
            let mirror = sum.used.iter().position(|&m| s.pts[m][0] == -s.pts[i][0]).expect("a mirror image");
            assert!((sum.products[j] + sum.products[mirror]).abs() < 1e-12);
        }
        // (the line of x on y is upright: its direction has no x)
        assert!(sum.lines().expect("lines")[2].expect("x on y")[0].abs() < 1e-12);

        // one far point: −0.04 among the twenty, 0.80 with it, most of the bar its own
        let mut s = trial(2);
        assert_eq!((s.set, s.pts.len(), s.pts[20]), (Set::Far, 21, FAR));
        assert_eq!(num(r_of(&s.pts[..20]), 2), "−0.04");
        let sum = s.summary();
        assert_eq!(num(sum.r.unwrap_or(0.0), 2), "0.80");
        let own = sum.products[20] / sum.products.iter().filter(|p| **p > 0.0).sum::<f64>();
        assert!(own > 0.75, "the far point's share of the positive products: {own}");
        assert_eq!(num(sum.spearman.unwrap_or(0.0), 2), "0.12");
        // into the cloud: about 0; to the top left: −0.81
        s.pts[20] = [0.0, 0.0];
        assert!(r_of(&s.pts).abs() < 0.06, "{}", r_of(&s.pts));
        s.pts[20] = [-FAR[0], FAR[1]];
        assert_eq!(num(r_of(&s.pts), 2), "−0.81");

        // regression to the mean: r = 0.50; the top tenth on x at 1.77 and 0.87, about r
        // times as far; on y at 1.75 and 0.87
        let mut s = trial(3);
        assert!(s.side == Side::Tenth && s.by == 0 && s.standard && s.pts.len() == 1000);
        let sum = s.summary();
        let r = sum.r.unwrap_or(0.0);
        assert_eq!(num(r, 2), "0.50");
        let (top, avg) = top_tenth(&s.pts, &sum, 0);
        let avg = avg.expect("averages");
        assert_eq!((top.len(), num(avg[0], 2), num(avg[1], 2)), (100, "1.77".to_string(), "0.87".to_string()));
        assert!((avg[1] / (r * avg[0]) - 1.0).abs() < 0.03, "{} against r times {}", avg[1], avg[0]);
        s.by = 1;
        let (_, avg) = top_tenth(&s.pts, &sum, 1);
        let avg = avg.expect("averages");
        assert_eq!((num(avg[1], 2), num(avg[0], 2)), ("1.75".to_string(), "0.87".to_string()));

        // a narrow window: 0.80 to 0.63; the slope 0.80 either way, the scatter about it
        // as much; r² from 0.65 to 0.40
        let mut s = trial(4);
        assert!(s.window && !s.standard && s.lines == [false, true, false] && s.pts.len() == 200);
        let kept = s.summary();
        s.window = false;
        let all = s.summary();
        assert_eq!((num(all.r.unwrap_or(0.0), 2), num(kept.r.unwrap_or(0.0), 2)), ("0.80".to_string(), "0.63".to_string()));
        assert_eq!((num(all.r2().unwrap_or(0.0), 2), num(kept.r2().unwrap_or(0.0), 2)), ("0.65".to_string(), "0.40".to_string()));
        assert_eq!(kept.used.len(), 100);
        let fit = |sum: &Summary| {
            let p: Vec<[f64; 2]> = sum.used.iter().map(|&i| s.pts[i]).collect();
            let (b, a) = crate::math::causal::line(&column(&p, 0), &column(&p, 1));
            let left = (p.iter().map(|q| (q[1] - a - b * q[0]).powi(2)).sum::<f64>() / p.len() as f64).sqrt();
            (b, left)
        };
        let ((b_all, left_all), (b_kept, left_kept)) = (fit(&all), fit(&kept));
        assert_eq!((num(b_all, 2), num(b_kept, 2)), ("0.80".to_string(), "0.80".to_string()));
        assert!((left_kept / left_all - 1.0).abs() < 0.1, "scatter about the line {left_all} and {left_kept}");
        // (x varies less: its spread in the window is well under the whole's)
        assert!(kept.sd[0] < 0.7 * all.sd[0]);

        // correlation from nothing: samples of 10, ρ = 0, six a second; beyond ±0.5 in 14 %
        let s = trial(5);
        assert!(s.side == Side::Samples && s.samples.running && s.n == 10 && s.rho == 0.0 && s.samples.rate == 2.0);
        let theory = 1.0 - r_probability(-TAIL, TAIL, 0.0, 10);
        assert_eq!(num(100.0 * theory, 0), "14");
        assert!((6.5..7.5).contains(&(1.0 / theory)), "one in {}", 1.0 / theory);

        // ranks: r = 0.89, the rank correlation exactly 1
        let s = trial(6);
        let sum = s.summary();
        assert_eq!((s.set, s.side), (Set::Exp, Side::Ranks));
        assert_eq!(num(sum.r.unwrap_or(0.0), 2), "0.89");
        assert_eq!(sum.spearman, Some(1.0));
    }

    /// The share beyond ±0.5 in samples of 10 that the caption and the chart give, by the
    /// t distribution (the classical route) and by the scene's own samples.
    #[test]
    fn one_sample_in_seven_looks_related() {
        let t = st::Dist::StudentT { df: 8.0 };
        let cut = TAIL * 8f64.sqrt() / (1.0 - TAIL * TAIL).sqrt();
        let by_t = 2.0 * (1.0 - t.cdf(cut));
        let by_density = 1.0 - r_probability(-TAIL, TAIL, 0.0, 10);
        assert!((by_t - by_density).abs() < 1e-6 && (by_t - 0.1411).abs() < 1e-4, "{by_t} {by_density}");
        let mut s = trial(5);
        for _ in 0..20_000 {
            s.samples.draw(0.0, 10);
        }
        let share = s.samples.beyond as f64 / s.samples.total as f64;
        assert!((share - by_t).abs() < 0.008, "{share} against {by_t}");
    }

    /// Fisher's density of r: it is a density (area 1), it is the textbook formula for
    /// unrelated variables, and samples follow it.
    #[test]
    fn the_density_of_r_is_right() {
        for (rho, n) in [(0.0, 3), (0.0, 4), (0.0, 10), (0.5, 10), (-0.8, 25), (0.95, 6), (0.3, 300)] {
            let area = r_probability(-1.0, 1.0, rho, n);
            assert!((area - 1.0).abs() < 1e-4, "ρ {rho}, n {n}: area {area}");
        }
        // ρ = 0: (1 − r²)^((n−4)/2) / B(1/2, (n−2)/2)
        for n in [3usize, 5, 10, 40] {
            for r in [-0.7, 0.0, 0.3, 0.9] {
                let plain = (1.0f64 - r * r).powf((n as f64 - 4.0) / 2.0) / st::ln_beta(0.5, (n as f64 - 2.0) / 2.0).exp();
                assert!((r_density(r, 0.0, n) / plain - 1.0).abs() < 1e-8, "n {n}, r {r}");
            }
        }
        // samples of 12 from a cloud with ρ = 0.6: the share in each fifth of the range
        let mut rng = Rng::new(3);
        let mut counts = [0usize; 10];
        let runs = 40_000;
        for _ in 0..runs {
            let r = summarise(&cloud_from(0.6, 12, &mut rng), false).r.unwrap_or(0.0);
            counts[(((r + 1.0) / 0.2) as usize).min(9)] += 1;
        }
        for (k, c) in counts.iter().enumerate() {
            let lo = -1.0 + 0.2 * k as f64;
            let expect = r_probability(lo, lo + 0.2, 0.6, 12);
            let share = *c as f64 / runs as f64;
            assert!((share - expect).abs() < 0.006 + 0.05 * expect, "[{lo}, {}]: {share} against {expect}", lo + 0.2);
        }
    }

    /// The degenerate scatters say what has happened, and nothing drawn is not a number.
    #[test]
    fn degenerate_data_are_told_honestly() {
        // no spread in x: no r, no lines
        let upright: Vec<[f64; 2]> = (0..8).map(|i| [2.0, i as f64]).collect();
        let s = summarise(&upright, false);
        assert!(s.r.is_none() && s.lines().is_none() && !s.varies[0] && s.varies[1]);
        assert!(top_tenth(&upright, &s, 0).1.is_none());
        // all on one spot
        let spot = vec![[1.0, 1.0]; 5];
        let s = summarise(&spot, false);
        assert!(s.r.is_none() && s.spearman.is_none() && s.products.iter().all(|p| *p == 0.0));
        // two points: r is ±1
        for q in [[3.0, 5.0], [-1.0, 4.0], [0.5, -2.0]] {
            let r = summarise(&[[0.0, 0.0], q], false).r.expect("an r");
            assert!((r.abs() - 1.0).abs() < 1e-12, "{q:?}: {r}");
        }
        // one point, no points
        assert!(summarise(&[[1.0, 2.0]], false).r.is_none() && summarise(&[], false).r.is_none());
        // the window over identical x leaves nothing
        let s = summarise(&upright, true);
        assert!(s.used.is_empty() && s.r.is_none());
        // r exactly 0: the line of x on y is upright, and there is no SD line
        let s = summarise(&[[-1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.0, -1.0]], false);
        let lines = s.lines().expect("lines");
        assert!(s.r == Some(0.0) && lines[0].is_none() && lines[2].expect("x on y")[0] == 0.0);
        // every setting of the sliders' ends, and every set: the ranges are finite
        let lay = Layout::of(Vec2::new(1075.0, 1000.0), 0.0);
        for set in [Set::Cloud, Set::Far, Set::Anscombe(3), Set::Square, Set::Exp] {
            for (rho, n) in [(-1.0, 2), (1.0, 2), (1.0, 1000), (0.0, 3), (-1.0, 60)] {
                let mut c = Correlation::create();
                (c.set, c.rho, c.n) = (set, rho, n);
                c.remake();
                for standard in [true, false] {
                    for window in [true, false] {
                        (c.standard, c.window) = (standard, window);
                        let s = c.summary();
                        assert!(c.target_range(&lay, &s).iter().all(|v| v.is_finite()), "{set:?} {rho} {n}");
                        assert!(s.products.iter().all(|p| p.is_finite()) && s.r.is_none_or(|r| r.is_finite()));
                    }
                }
            }
        }
    }

    /// The layout of every window: finite, inside the canvas, nothing on top of anything.
    #[test]
    fn the_layout_fits_every_window() {
        for (w, h, inset) in [(1075.0, 1000.0, 0.0), (1100.0, 700.0, 0.0), (1600.0, 1000.0, 0.0), (390.0, 800.0, 44.0), (390.0, 756.0, 44.0), (700.0, 500.0, 0.0), (240.0, 180.0, 0.0), (1.0, 1.0, 0.0)] {
            let lay = Layout::of(Vec2::new(w, h), inset);
            let rects = [lay.scatter, lay.bar, lay.share, lay.side, lay.tabs[0], lay.tabs[1], lay.tabs[2]];
            assert!(rects.iter().all(|(a, b)| a.is_finite() && b.is_finite()), "{w}x{h}");
            if lay.compact {
                continue;
            }
            assert!(rects.iter().all(|(a, b)| a.x >= 0.0 && a.y >= inset && b.x <= w + 0.5 && b.y <= h + 0.5), "{w}x{h}: {rects:?}");
            let side = lay.scatter.1.x - lay.scatter.0.x;
            assert!(side > 150.0, "{w}x{h}: a scatter {side} across");
            if lay.narrow {
                assert!(lay.scatter.1.y < lay.bar.0.y && lay.bar.1.y < lay.share.0.y && lay.share.1.y < lay.tabs[0].0.y && lay.tabs[0].1.y < lay.side.0.y, "{w}x{h}");
                assert!(lay.side.1.y - lay.side.0.y > 140.0, "{w}x{h}: {:?}", lay.side);
            } else {
                assert!(lay.scatter.1.x < lay.bar.0.x && lay.bar.1.y < lay.share.0.y && lay.share.1.y < lay.side.0.y, "{w}x{h}");
            }
        }
    }

    /// A change of data glides and then comes exactly to rest, at any frame rate.
    #[test]
    fn glides_arrive_and_stop() {
        let lay = Layout::of(Vec2::new(1075.0, 1000.0), 0.0);
        for hz in [30.0f32, 60.0, 144.0] {
            let mut s = Correlation::create();
            s.glide(&lay, 1.0 / hz);
            assert!(!s.moving);
            // set I to set II: the same 11 points, moved
            s.set_up(0);
            s.glide(&lay, 1.0 / hz);
            s.choose_set(Set::Anscombe(1));
            let mut frames = 0;
            loop {
                s.glide(&lay, 1.0 / hz);
                frames += 1;
                if !s.moving {
                    break;
                }
                assert!((frames as f32) < 3.0 * hz, "still gliding after three seconds at {hz} Hz");
            }
            assert!(frames as f32 > 0.2 * hz, "arrived in {frames} frames at {hz} Hz: no glide");
            assert_eq!(s.shown_points(), s.pts.iter().map(|p| [p[0] as f32 as f64, p[1] as f32 as f64]).collect::<Vec<_>>());
        }
    }
}

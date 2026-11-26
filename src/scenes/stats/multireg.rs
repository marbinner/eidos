//! Regression with several inputs: what a coefficient means when there is more than one
//! input, what inputs that move together do to it, and what the residuals show.

use super::plot::{self, Plot, Rect, AXES, CURVES, MARKS};
use super::*;
use crate::math::stats as st;

pub const DEF: SceneDef = SceneDef {
    id: "multireg",
    title: "Regression with several inputs",
    group: MODELS,
    blurb: "Each coefficient is how the fit changes with one input, the others held fixed: what that means, what inputs that move together do to it, and what the leftovers show.",
    make: |_| Box::new(MultiReg::create()),
};

// ----- the data ------------------------------------------------------------------------

/// The most points a sample has.
const MOST: usize = 200;
/// How far the floor reaches either way from 0, in the inputs' own units.
const REACH: f64 = 3.0;
/// Where the point put far out lies (the inputs are standard normal draws: this is
/// about three standard deviations out along each).
const FAR: [f64; 2] = [3.0, -2.8];

/// What the sample is made from: the truth's two coefficients and its bend, the
/// correlation of the inputs, the noise, the number of points, the seed, and whether
/// the first point is put far out in the inputs.
#[derive(Clone, Copy, Debug, PartialEq)]
struct Setup {
    b1: f64,
    b2: f64,
    bend: f64,
    rho: f64,
    sigma: f64,
    n: usize,
    seed: u64,
    far: bool,
}

impl Setup {
    /// The truth at (x1, x2): the mean of y there.
    fn truth(&self, x1: f64, x2: f64) -> f64 {
        self.b1 * x1 + self.b2 * x2 + self.bend * (x1 * x1 - 1.0)
    }

    /// The settings kept within the controls' ranges.
    fn kept(mut self) -> Setup {
        let within = |v: f64, lo: f64, hi: f64| if v.is_finite() { v.max(lo).min(hi) } else { lo.max(0.0).min(hi) };
        self.b1 = within(self.b1, -2.0, 2.0);
        self.b2 = within(self.b2, -2.0, 2.0);
        self.bend = within(self.bend, 0.0, 1.5);
        self.rho = within(self.rho, -1.0, 1.0);
        self.sigma = within(self.sigma, 0.0, 2.0);
        self.n = self.n.clamp(3, MOST);
        self
    }
}

/// Three standard normal draws for each of the most points a sample can have: the
/// first two make the inputs, the third the noise. The same seed gives the same draws
/// whatever the settings, so that a setting moved moves the points, not reshuffles them.
fn draws(seed: u64) -> Vec<[f64; 3]> {
    let mut rng = Rng::new(seed);
    (0..MOST).map(|_| [st::normal(&mut rng), st::normal(&mut rng), st::normal(&mut rng)]).collect()
}

/// The inputs of a sample: x1 a standard normal draw, x2 one with correlation ρ to it.
fn inputs(s: &Setup, z: &[[f64; 3]]) -> (Vec<f64>, Vec<f64>) {
    let other = (1.0 - s.rho * s.rho).max(0.0).sqrt();
    let n = s.n.min(z.len());
    let mut x1: Vec<f64> = z[..n].iter().map(|d| d[0]).collect();
    let mut x2: Vec<f64> = z[..n].iter().map(|d| s.rho * d[0] + other * d[1]).collect();
    if s.far && n > 0 {
        (x1[0], x2[0]) = (FAR[0], FAR[1]);
    }
    (x1, x2)
}

/// Least squares with two inputs and a constant, and what it leaves.
#[derive(Clone, Debug, Default, PartialEq)]
struct Fit {
    n: usize,
    /// The means of x1, x2 and y.
    mean: [f64; 3],
    /// Sums of products of the deviations from the means.
    s11: f64,
    s22: f64,
    s12: f64,
    s1y: f64,
    s2y: f64,
    syy: f64,
    /// b0, b1, b2; none where the inputs cannot be told apart (one is a straight-line
    /// function of the other, so that many planes fit equally well).
    b: Option<[f64; 3]>,
    /// The fitted values (the plane's height over each point, or where there is no
    /// one plane, the height of the line that every equally good plane passes through)
    /// and the residuals.
    fitted: Vec<f64>,
    resid: Vec<f64>,
    rss: f64,
}

/// The inputs count as moving exactly together when what is left of one once the other
/// is known is less than this share of it (rounding leaves about 1e-16 of an exact copy).
const SAME: f64 = 1e-10;

fn fit(x1: &[f64], x2: &[f64], y: &[f64]) -> Fit {
    let n = x1.len().min(x2.len()).min(y.len());
    if n == 0 {
        return Fit::default();
    }
    let mean = [st::mean(&x1[..n]), st::mean(&x2[..n]), st::mean(&y[..n])];
    let (mut s11, mut s22, mut s12, mut s1y, mut s2y, mut syy) = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
    for i in 0..n {
        let (a, b, c) = (x1[i] - mean[0], x2[i] - mean[1], y[i] - mean[2]);
        s11 += a * a;
        s22 += b * b;
        s12 += a * b;
        s1y += a * c;
        s2y += b * c;
        syy += c * c;
    }
    let det = s11 * s22 - s12 * s12;
    let b = (s11 > 0.0 && s22 > 0.0 && det > SAME * s11 * s22).then(|| {
        let b1 = (s22 * s1y - s12 * s2y) / det;
        let b2 = (s11 * s2y - s12 * s1y) / det;
        [mean[2] - b1 * mean[0] - b2 * mean[1], b1, b2]
    });
    let fitted: Vec<f64> = (0..n)
        .map(|i| match b {
            Some([b0, b1, b2]) => b0 + b1 * x1[i] + b2 * x2[i],
            // one input settles everything the two can: the line through the points
            None if s11 > 0.0 => mean[2] + s1y / s11 * (x1[i] - mean[0]),
            None if s22 > 0.0 => mean[2] + s2y / s22 * (x2[i] - mean[1]),
            None => mean[2],
        })
        .collect();
    let resid: Vec<f64> = (0..n).map(|i| y[i] - fitted[i]).collect();
    let rss = resid.iter().map(|r| r * r).sum();
    Fit { n, mean, s11, s22, s12, s1y, s2y, syy, b, fitted, resid, rss }
}

impl Fit {
    /// The sums of squares and products of input `j` with itself, with the other
    /// input, and with y; and the other input's with itself and with y.
    fn sums(&self, j: usize) -> (f64, f64, f64, f64, f64) {
        if j == 0 {
            (self.s11, self.s12, self.s1y, self.s22, self.s2y)
        } else {
            (self.s22, self.s12, self.s2y, self.s11, self.s1y)
        }
    }

    /// The correlation of the two inputs in the sample (0 if either has no spread).
    fn r_inputs(&self) -> f64 {
        let spread = (self.s11 * self.s22).sqrt();
        if spread > 0.0 && spread.is_finite() {
            (self.s12 / spread).max(-1.0).min(1.0)
        } else {
            0.0
        }
    }

    /// Does y vary at all? (Otherwise there is nothing for R² to be a share of.)
    fn y_varies(&self) -> bool {
        self.syy > 1e-20 * (1.0 + self.mean[2] * self.mean[2]) * self.n as f64
    }

    /// The share of y's variation about its mean that the fit accounts for.
    fn r2(&self) -> Option<f64> {
        self.y_varies().then(|| (1.0 - self.rss / self.syy).max(0.0).min(1.0))
    }

    /// R² with its penalty for the inputs used (both, or one where they cannot be told
    /// apart): none where the fit leaves no points over.
    fn adj_r2(&self) -> Option<f64> {
        let p = if self.b.is_some() { 2 } else { 1 };
        adjusted(self.r2()?, self.n, p)
    }

    /// The spread of the noise as estimated from the residuals (dividing by n − 3, the
    /// points the plane leaves free): none for three points or fewer.
    fn sigma_hat(&self) -> Option<f64> {
        (self.n > 3).then(|| (self.rss / (self.n - 3) as f64).sqrt())
    }

    /// The standard errors of b1 and b2.
    fn se(&self) -> Option<[f64; 2]> {
        self.b?;
        let s = self.sigma_hat()?;
        let det = self.s11 * self.s22 - self.s12 * self.s12;
        Some([(s * s * self.s22 / det).sqrt(), (s * s * self.s11 / det).sqrt()])
    }

    /// The straight line through y against input `j` alone: (slope, intercept).
    fn alone(&self, j: usize) -> Option<(f64, f64)> {
        let (sjj, _, sjy, _, _) = self.sums(j);
        (sjj > 0.0).then(|| (sjy / sjj, self.mean[2] - sjy / sjj * self.mean[j]))
    }

    /// That line's R².
    fn alone_r2(&self, j: usize) -> Option<f64> {
        let (sjj, _, sjy, _, _) = self.sums(j);
        (sjj > 0.0 && self.y_varies()).then(|| (sjy * sjy / (sjj * self.syy)).max(0.0).min(1.0))
    }

    /// The slope of the other input on input `j`: how much it rises, on average, for
    /// one more unit of input `j`.
    fn other_on(&self, j: usize) -> Option<f64> {
        let (sjj, sjo, _, _, _) = self.sums(j);
        (sjj > 0.0).then(|| sjo / sjj)
    }

    /// How far a point at (x1, x2) can pull the fitted value at itself: lifting its y by
    /// 1 lifts the plane under it by this much. Between 1/n and 1.
    fn leverage(&self, x1: f64, x2: f64) -> Option<f64> {
        self.b?;
        let det = self.s11 * self.s22 - self.s12 * self.s12;
        let (a, b) = (x1 - self.mean[0], x2 - self.mean[1]);
        Some(1.0 / self.n as f64 + (self.s22 * a * a - 2.0 * self.s12 * a * b + self.s11 * b * b) / det)
    }

    /// The height of the plane at (x1, x2).
    fn at(&self, x1: f64, x2: f64) -> Option<f64> {
        self.b.map(|[b0, b1, b2]| b0 + b1 * x1 + b2 * x2)
    }

    /// Where the inputs cannot be told apart: input `j` (the one with spread) and the
    /// other as a straight-line function of it, other = c · input + d, and the one
    /// combination of the coefficients that is settled, b_j + c · b_other.
    fn tied(&self) -> Option<(usize, f64, f64, f64)> {
        if self.b.is_some() {
            return None;
        }
        let j = if self.s11 > 0.0 {
            0
        } else if self.s22 > 0.0 {
            1
        } else {
            return None;
        };
        let (sjj, sjo, sjy, _, _) = self.sums(j);
        let c = sjo / sjj;
        let d = self.mean[1 - j] - c * self.mean[j];
        Some((j, c, d, sjy / sjj))
    }
}

/// R² adjusted for `p` inputs fitted to `n` points: none where nothing is left over.
fn adjusted(r2: f64, n: usize, p: usize) -> Option<f64> {
    (n > p + 1).then(|| 1.0 - (1.0 - r2) * (n - 1) as f64 / (n - p - 1) as f64)
}

/// What is left of `of` once a straight line in `by` has been taken out: the
/// residuals of `of` regressed on `by` (with a constant).
fn left_of(of: &[f64], by: &[f64]) -> Vec<f64> {
    crate::math::causal::residuals(&[by], of)
}

/// One sample as the scene holds it: the inputs, the outputs before the marked points
/// are lifted, and which points are marked.
#[derive(Clone, Debug, Default)]
struct Sample {
    x1: Vec<f64>,
    x2: Vec<f64>,
    y: Vec<f64>,
    /// The point with the most leverage and the one with the least.
    marked: [usize; 2],
}

impl Sample {
    fn new(s: &Setup, z: &[[f64; 3]]) -> Sample {
        let (x1, x2) = inputs(s, z);
        let y: Vec<f64> = (0..x1.len()).map(|i| s.truth(x1[i], x2[i]) + s.sigma * z[i][2]).collect();
        let plain = fit(&x1, &x2, &y);
        // how far out each point is: its leverage, or (where there is none) its distance
        let far = |i: usize| plain.leverage(x1[i], x2[i]).unwrap_or_else(|| (x1[i] - plain.mean[0]).powi(2) + (x2[i] - plain.mean[1]).powi(2));
        let order = |a: &usize, b: &usize| far(*a).total_cmp(&far(*b)).then(b.cmp(a));
        let most = (0..x1.len()).max_by(order).unwrap_or(0);
        let least = (0..x1.len()).min_by(order).unwrap_or(0);
        Sample { x1, x2, y, marked: [most, least] }
    }

    /// The outputs with the marked points lifted.
    fn lifted(&self, lift: [f64; 2]) -> Vec<f64> {
        let mut y = self.y.clone();
        for (k, &i) in self.marked.iter().enumerate() {
            if let Some(v) = y.get_mut(i) {
                *v += lift[k];
            }
        }
        // (the two marked points are one where there is a single point to mark)
        if self.marked[0] == self.marked[1] {
            if let Some(v) = y.get_mut(self.marked[0]) {
                *v -= lift[1];
            }
        }
        y
    }
}

/// Coefficients from new samples with the same inputs and fresh noise: what the
/// standard errors describe.
#[derive(Clone, Debug)]
struct Cloud {
    /// Asked for (and then it fills up to `NEW_SAMPLES`).
    on: bool,
    /// (b1, b2) of each new sample, and the plane of each.
    coef: Vec<[f64; 2]>,
    planes: Vec<[f64; 3]>,
    rng: Rng,
    pace: Pace,
    phase: Phase,
}

/// How many new samples a cloud holds, and how many it gains a second.
const NEW_SAMPLES: usize = 400;
const SAMPLE_RATE: f32 = 160.0;

impl Cloud {
    fn new(seed: u64) -> Cloud {
        Cloud { on: false, coef: Vec::new(), planes: Vec::new(), rng: Rng::new(seed ^ 0x5EED_C10D), pace: Pace::default(), phase: Phase::default() }
    }

    fn running(&self) -> bool {
        self.on && self.coef.len() < NEW_SAMPLES
    }

    /// One new sample: the same inputs, the truth, and new noise; its fit kept.
    /// Returns false where the inputs cannot be told apart (no coefficients to keep).
    fn step(&mut self, s: &Setup, x1: &[f64], x2: &[f64]) -> bool {
        let y: Vec<f64> = x1.iter().zip(x2).map(|(a, b)| s.truth(*a, *b) + s.sigma * st::normal(&mut self.rng)).collect();
        match fit(x1, x2, &y).b {
            Some([b0, b1, b2]) => {
                self.coef.push([b1, b2]);
                self.planes.push([b0, b1, b2]);
                true
            }
            None => false,
        }
    }

    /// The spreads of b1, b2 and b1 + b2 over the new samples (standard deviations),
    /// and the share of them in which b1 has the other sign from the truth's.
    fn spreads(&self, s: &Setup) -> Option<([f64; 3], f64)> {
        if self.coef.len() < 2 {
            return None;
        }
        let col = |k: usize| -> Vec<f64> { self.coef.iter().map(|c| if k < 2 { c[k] } else { c[0] + c[1] }).collect() };
        let flipped = self.coef.iter().filter(|c| c[0] * s.b1 < 0.0).count() as f64 / self.coef.len() as f64;
        Some(([st::sd(&col(0)), st::sd(&col(1)), st::sd(&col(2))], flipped))
    }
}

// ----- the experiments -------------------------------------------------------------------

/// One thing to try: its button and caption, the settings, how far the point with the
/// most leverage is lifted, whether new samples are drawn and the truth shown, and the
/// page of charts it wants where they do not all fit.
struct Trial {
    name: &'static str,
    look: &'static str,
    setup: Setup,
    lift: f64,
    cloud: bool,
    truth: bool,
    page: usize,
}

#[allow(clippy::too_many_arguments)]
const fn setup(b1: f64, b2: f64, bend: f64, rho: f64, sigma: f64, n: usize, seed: u64, far: bool) -> Setup {
    Setup { b1, b2, bend, rho, sigma, n, seed, far }
}

const TRIALS: [Trial; 6] = [
    Trial {
        name: "unrelated inputs",
        look: r"The inputs are unrelated: their correlation in this sample is $-0.001$. Alone, $x_1$'s slope is $0.906$; with $x_2$ held fixed, $b_1 = 0.905$. For $x_2$: $-0.696$ alone, $b_2 = -0.695$. Each chart with the other input held fixed looks like the one beside it. The two slopes agree exactly only when the inputs' correlation in the sample is exactly 0 (or the other coefficient is).",
        setup: setup(1.0, -0.6, 0.0, 0.0, 1.0, 60, 322, false),
        lift: 0.0,
        cloud: false,
        truth: false,
        page: 0,
    },
    Trial {
        name: "inputs that move together",
        look: r"$x_2$ rises with $x_1$ (correlation $0.83$); $y$ rises with $x_2$ ($\beta_2 = 1.5$) and falls a little with $x_1$ ($\beta_1 = -0.5$). Alone, $x_1$ seems to raise $y$: slope $0.60$. With $x_2$ held fixed its coefficient is $b_1 = -0.47$, near the truth: to the last digit, the slope of what $x_2$ leaves of $y$ against what it leaves of $x_1$ (the chart at the top right). The slope alone is $b_1$ plus $b_2$ times how much $x_2$ rises with $x_1$: $-0.47 + 1.34 \times 0.80 = 0.60$, exactly. Leaving out an input that matters changes the answer: that is confounding.",
        setup: setup(-0.5, 1.5, 0.0, 0.8, 1.0, 60, 322, false),
        lift: 0.0,
        cloud: false,
        truth: false,
        page: 0,
    },
    Trial {
        name: "nearly the same input twice",
        look: r"$\rho = 0.98$: the points lie along a line on the floor, and the plane is pinned along it but free to swing about it. Over 400 new samples (the same inputs, new noise) $b_1$ ranges from $-1.2$ to $3.3$ and has the wrong sign in 10 % of them, while $b_1 + b_2$ stays between $1.6$ and $2.5$, and the plane's height over the middle of the points varies by $0.14$ (its spread, against $0.79$ for $b_1$). $b_1$ varies 5.9 times as much as with an unrelated $x_2$: $1/\sqrt{1 - r^2}$, with the sample's $r = 0.985$, is $5.8$.",
        setup: setup(1.0, 1.0, 0.0, 0.98, 1.0, 50, 322, false),
        lift: 0.0,
        cloud: true,
        truth: false,
        page: 1,
    },
    Trial {
        name: "a useless input",
        look: r"$x_2$ is noise that has nothing to do with $y$ ($\beta_2 = 0$, and unrelated to $x_1$). With $x_1$ alone, $R^2 = 0.268$ and adjusted $R^2 = 0.242$. With $x_2$ added, $R^2$ rises to $0.293$, as it must: adding an input can never lower it. Adjusted $R^2$ falls a little, to $0.240$. It fell in 66 % of 2000 new samples like this one.",
        setup: setup(1.0, 0.0, 0.0, 0.0, 1.5, 30, 322, false),
        lift: 0.0,
        cloud: false,
        truth: false,
        page: 0,
    },
    Trial {
        name: "a bend",
        look: r"The truth (green) bends up along $x_1$ ($\gamma = 0.8$). The plane fits as well as a plane can ($R^2 = 0.44$), and what it misses shows in the residuals against the fitted values: an arc, on average $+0.89$ for the fifth of the points with the lowest fitted values, $-0.50$ for the middle three fifths and $+0.62$ for the highest fifth. A model of the right shape leaves no such pattern.",
        setup: setup(1.0, 0.5, 0.8, 0.3, 0.5, 80, 322, false),
        lift: 0.0,
        cloud: false,
        truth: true,
        page: 1,
    },
    Trial {
        name: "one point with leverage",
        look: r"The white point far out, at $(3.0, -2.8)$, has leverage $h = 0.40$. Lifted by 4, it lifts the plane under it by $1.62$, which is $0.40 \times 4$, and tilts the whole plane: $b_2$ falls from $0.39$ to $0.04$. Drag the white point in the middle ($h = 0.04$) up by the same 4: the plane under it rises by only $0.14$, and the coefficients hardly change.",
        setup: setup(1.0, 0.5, 0.0, 0.0, 0.8, 30, 322, true),
        lift: 4.0,
        cloud: false,
        truth: false,
        page: 0,
    },
];

/// The scene opens as this experiment sets it up.
const OPENING: usize = 1;

// ----- where things go on the canvas -------------------------------------------------------

/// The charts, in the order of `Layout::charts`.
const ALONE_1: usize = 0;
const HELD_1: usize = 1;
const RESIDUALS: usize = 4;
const SAMPLES: usize = 5;

/// The pages of charts where they do not all fit: their tabs.
const PAGES: [&str; 2] = ["the slopes", "residuals, new samples"];

/// Room round a chart's data for what is written round it: left (the tick numbers),
/// above (its title and the y axis's name), right, below (the tick numbers and the x
/// axis's name).
const LEFT: f32 = 38.0;
const ABOVE: f32 = 46.0;
const RIGHT: f32 = 10.0;
const BELOW: f32 = 34.0;
const TAB_H: f32 = 24.0;

/// Where everything goes on a canvas of `view` logical pixels, `inset` of whose top lies
/// under the application's buttons.
#[derive(Clone, Copy, Debug, PartialEq)]
struct Layout {
    view: Vec2,
    /// The part of the canvas the picture in space has.
    pic: Rect,
    /// Whether all six charts fit (otherwise a page of them at a time, chosen by tabs).
    all: bool,
    /// The charts' data rectangles: x1 alone, x1 with x2 held fixed, x2 alone, x2 with
    /// x1 held fixed, the residuals, the new samples (None: not on the page shown).
    charts: [Option<Rect>; 6],
    /// The row of tabs (its top-left corner), where there are pages.
    tabs: Option<Vec2>,
    /// Text sizes: titles, and numbers and names.
    title: f32,
    small: f32,
}

impl Layout {
    fn of(view: Vec2, inset: f32, page: usize) -> Layout {
        let view = view.max(Vec2::splat(1.0));
        let top = inset.max(0.0).min(view.y * 0.5);
        let mut lay = Layout { view, pic: (Vec2::new(0.0, top), view), all: false, charts: [None; 6], tabs: None, title: 11.5, small: 10.5 };
        // a canvas too small for charts has the picture alone
        if view.x < 300.0 || view.y < 300.0 {
            return lay;
        }
        if view.x >= 700.0 && view.y >= 600.0 {
            (lay.title, lay.small) = (12.5, 11.0);
        }
        // a cell of the grid of charts, and the data rectangle in it
        let data = |min: Vec2, max: Vec2| -> Rect { (min + Vec2::new(LEFT, ABOVE), max - Vec2::new(RIGHT, BELOW)) };
        let page = page.min(PAGES.len() - 1);
        // two columns of charts in the rectangle from `min` to `max`, in `rows` rows:
        // all six, or the page shown
        let grid = |lay: &mut Layout, min: Vec2, max: Vec2, rows: usize| {
            let cell = Vec2::new((max.x - min.x) * 0.5, (max.y - min.y) / rows as f32);
            let at = |c: usize, r: usize| min + Vec2::new(c as f32 * cell.x, r as f32 * cell.y);
            let one = |c: usize, r: usize| data(at(c, r), at(c, r) + cell);
            let row = |r: usize| data(at(0, r), at(0, r) + Vec2::new(2.0 * cell.x, cell.y));
            if rows == 3 {
                lay.charts = [Some(one(0, 0)), Some(one(1, 0)), Some(one(0, 1)), Some(one(1, 1)), Some(one(0, 2)), Some(one(1, 2))];
            } else if page == 0 {
                lay.charts = [Some(one(0, 0)), Some(one(1, 0)), Some(one(0, 1)), Some(one(1, 1)), None, None];
            } else {
                lay.charts = [None, None, None, None, Some(row(0)), Some(row(1))];
            }
        };
        // beside the picture on a canvas wider than it is tall, below it otherwise
        if view.x >= 760.0 && view.x >= view.y * 0.8 {
            let column = (view.x * 0.47).max(380.0).min(620.0);
            let x0 = view.x - column - 6.0;
            lay.pic = (Vec2::new(0.0, top), Vec2::new(x0, view.y));
            lay.all = view.y - top >= 640.0;
            if lay.all {
                grid(&mut lay, Vec2::new(x0, top + 6.0), Vec2::new(view.x - 6.0, view.y - 4.0), 3);
            } else {
                lay.tabs = Some(Vec2::new(x0 + 10.0, top + 8.0));
                grid(&mut lay, Vec2::new(x0, top + TAB_H + 12.0), Vec2::new(view.x - 6.0, view.y - 4.0), 2);
            }
        } else {
            let below = top + ((view.y - top) * 0.46).max(150.0);
            lay.pic = (Vec2::new(0.0, top), Vec2::new(view.x, below));
            lay.tabs = Some(Vec2::new(10.0, below + 4.0));
            grid(&mut lay, Vec2::new(2.0, below + TAB_H + 8.0), Vec2::new(view.x - 2.0, view.y - 2.0), 2);
        }
        // charts too small to read are left out, and the picture has the canvas
        if lay.charts.iter().flatten().any(|(min, max)| max.x - min.x < 60.0 || max.y - min.y < 40.0) {
            (lay.pic, lay.charts, lay.tabs, lay.all) = ((Vec2::new(0.0, top), view), [None; 6], None, false);
        }
        lay
    }

    /// The middle of the picture's part of the canvas, and the radius in pixels the
    /// box in space is drawn at (its widest, as it turns).
    fn stage(&self) -> (Vec2, f32) {
        let (min, max) = self.pic;
        let size = (max - min).max(Vec2::splat(1.0));
        // (a little below the middle: the equation is written over the top)
        let room = (size.y - 70.0).max(size.y * 0.6);
        (Vec2::new((min.x + max.x) * 0.5, max.y - room * 0.5), (size.x * 0.5).min(room * 0.5) * 0.86)
    }
}

/// The box in space: world units per unit of an input, its half-height in world units,
/// and the radius of the sphere it turns within (in world units).
const S: f32 = 0.8;
const H: f32 = 2.4;
const BOX_RADIUS: f32 = 3.7;
/// The tangent of half the camera's field of view (`Camera::fov_y` is 0.62).
const TAN_HALF_FOV: f32 = 0.3203;

/// The camera's distance that draws the box `radius` pixels across on a canvas `height`
/// pixels high.
fn distance(radius: f32, height: f32) -> f32 {
    BOX_RADIUS / radius.max(1.0) * height.max(1.0) * 0.5 / TAN_HALF_FOV
}

/// Round heights for the top of the box, in units of y.
fn round_up(v: f64) -> f64 {
    let v = if v.is_finite() { v.max(0.5) } else { 1.0 };
    let p = 10f64.powf(v.log10().floor());
    [1.0, 1.25, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0].iter().map(|m| m * p).find(|m| *m >= v * 0.999_999).unwrap_or(10.0 * p)
}

// ----- what is drawn, gliding --------------------------------------------------------------

/// Numbers shown gliding to where they have gone, through two eases one after the
/// other (as `plot::Glide`), with a time constant chosen at each change: slow after an
/// experiment's button, so that the way from the old picture to the new one can be
/// seen, and quick after a slider.
#[derive(Clone, Debug, Default)]
struct Ease {
    between: Vec<f32>,
    shown: Vec<f32>,
}

impl Ease {
    /// Bring the numbers `dt` seconds nearer to `to`; a list that has grown or shrunk
    /// keeps what it had in common. Says whether anything is still on its way.
    fn follow(&mut self, to: &[f32], dt: f32, tau: f32) -> bool {
        if self.shown.len() != to.len() {
            let n = self.shown.len().min(to.len());
            self.shown.truncate(n);
            self.between.truncate(n);
            self.shown.extend_from_slice(&to[n..]);
            self.between.extend_from_slice(&to[n..]);
        }
        if self.shown == to {
            return false;
        }
        let k = if tau > 0.0 { 1.0 - (-dt.max(0.0) / tau).exp() } else { 1.0 };
        let mut apart = 0.0f32;
        for ((b, s), t) in self.between.iter_mut().zip(self.shown.iter_mut()).zip(to) {
            *b += (t - *b) * k;
            *s += (*b - *s) * k;
            apart = apart.max((t - *s).abs()).max((t - *b).abs());
        }
        // (exactly there in the end: the picture then stops changing)
        if apart.is_nan() || apart <= 2e-4 {
            self.snap(to);
            return false;
        }
        true
    }

    fn snap(&mut self, to: &[f32]) {
        self.shown = to.to_vec();
        self.between = to.to_vec();
    }
}

/// How long the glide after an experiment's button takes (seconds: the time constant of
/// each of its two eases), and after anything else.
const SLOW: f32 = 0.22;
const QUICK: f32 = 0.06;

/// The sample as drawn: inputs, outputs before the lifts, and the outputs as they are.
struct Drawn {
    x: [Vec<f64>; 2],
    y0: Vec<f64>,
    y: Vec<f64>,
    fit: Fit,
    /// The fit before the marked points were lifted.
    plain: Fit,
}

// ----- the scene -------------------------------------------------------------------------

pub struct MultiReg {
    setup: Setup,
    /// The draws of `setup.seed`.
    z: Vec<[f64; 3]>,
    z_seed: u64,
    sample: Sample,
    /// The setup the sample was made from.
    made: Setup,
    /// How far the two marked points have been lifted from where the sample put them.
    lift: [f64; 2],
    cloud: Cloud,
    show_truth: bool,
    turning: bool,
    /// The page of charts shown where they do not all fit.
    page: usize,
    /// The sample as drawn (the lifts first, then each point's x1, x2 and y), the top of
    /// the box as drawn and where it is going, and the time constant of the glide.
    look: Ease,
    reach: Ease,
    reach_goal: f64,
    tau: f32,
    /// A marked point is being dragged.
    holding: bool,
    /// The canvas last drawn on, how much of its top the application's buttons cover,
    /// and whether the camera is to be pointed again (the picture's place follows them).
    canvas: Vec2,
    inset: f32,
    new_view: bool,
    /// The charts drawn this frame; the round values on their axes and on the box's
    /// last frame (hashes), and how many points the charts' axes and the box's ticks
    /// had, and how many labels were written.
    plots: Vec<Plot>,
    ticks: (u64, u64),
    drawn_last: (usize, usize, usize),
}

impl MultiReg {
    /// (Nothing here needs the graphics device, so the tests can make one too.)
    fn create() -> MultiReg {
        let s = TRIALS[OPENING].setup;
        let z = draws(s.seed);
        let sample = Sample::new(&s, &z);
        let mut m = MultiReg {
            setup: s,
            z,
            z_seed: s.seed,
            sample,
            made: s,
            lift: [0.0; 2],
            cloud: Cloud::new(s.seed),
            show_truth: false,
            turning: true,
            page: 0,
            look: Ease::default(),
            reach: Ease::default(),
            reach_goal: 1.0,
            tau: QUICK,
            holding: false,
            canvas: Vec2::new(1080.0, 1000.0),
            inset: 0.0,
            new_view: false,
            plots: Vec::new(),
            ticks: (0, 0),
            drawn_last: (0, 0, 0),
        };
        m.set_up(OPENING);
        // (the picture starts where it is, not gliding from nothing)
        m.look.snap(&m.goal());
        m.reach_goal = m.reach_wanted();
        m.reach.snap(&[m.reach_goal as f32]);
        m
    }

    /// Set up one of the experiments in `TRIALS`.
    fn set_up(&mut self, k: usize) {
        let t = &TRIALS[k];
        self.setup = t.setup;
        self.refresh();
        self.lift = [t.lift, 0.0];
        self.show_truth = t.truth;
        self.page = t.page;
        self.cloud = Cloud::new(self.setup.seed);
        self.cloud.on = t.cloud;
        self.tau = SLOW;
    }

    /// What the picture glides to: the lifts, then each point's x1, x2 and y (before
    /// the lifts).
    fn goal(&self) -> Vec<f32> {
        let s = &self.sample;
        let mut to = Vec::with_capacity(2 + 3 * s.x1.len());
        to.extend(self.lift.map(|v| v as f32));
        for i in 0..s.x1.len() {
            to.extend([s.x1[i] as f32, s.x2[i] as f32, s.y[i] as f32]);
        }
        to
    }

    /// The top of the box: room for every point and the plane over the floor, rounded
    /// up. (The truth, where it is shown, is cut where it leaves the box.)
    fn reach_wanted(&self) -> f64 {
        let (y, f) = (self.y(), self.fit());
        let corners = [(-REACH, -REACH), (REACH, -REACH), (REACH, REACH), (-REACH, REACH)];
        let mut most = y.iter().fold(0.0f64, |m, v| m.max(v.abs()));
        for (a, b) in corners {
            most = most.max(f.at(a, b).unwrap_or(0.0).abs());
        }
        round_up(most * 1.04)
    }

    /// The sample as drawn now.
    fn drawn(&self) -> Drawn {
        let v = &self.look.shown;
        let lift = [v.first().copied().unwrap_or(0.0) as f64, v.get(1).copied().unwrap_or(0.0) as f64];
        let n = v.len().saturating_sub(2) / 3;
        let at = |i: usize, k: usize| v.get(2 + 3 * i + k).copied().unwrap_or(0.0) as f64;
        let x = [(0..n).map(|i| at(i, 0)).collect::<Vec<f64>>(), (0..n).map(|i| at(i, 1)).collect::<Vec<f64>>()];
        let y0: Vec<f64> = (0..n).map(|i| at(i, 2)).collect();
        // (the marked points as they are now: the drawn list may be shorter for a frame)
        let mut y = y0.clone();
        for (k, &i) in self.sample.marked.iter().enumerate() {
            if k == 1 && self.sample.marked[0] == i {
                continue;
            }
            if let Some(v) = y.get_mut(i) {
                *v += lift[k];
            }
        }
        Drawn::of(&x[0], &x[1], &y0, &y)
    }

    /// The sample as it is (where the drawing glides to).
    fn goal_drawn(&self) -> Drawn {
        Drawn::of(&self.sample.x1, &self.sample.x2, &self.sample.y, &self.y())
    }

    /// Where the camera looks from, for this canvas.
    fn the_view(&self) -> View {
        let lay = Layout::of(self.canvas, self.inset, self.page);
        let (_, radius) = lay.stage();
        View::Orbit { target: Vec3::ZERO, yaw: -0.95, pitch: 0.3, dist: distance(radius, self.canvas.y) }
    }

    /// Make the sample again if what it depends on has changed (and forget the lifts and
    /// the new samples, which belong to the sample that was).
    fn refresh(&mut self) {
        self.setup = self.setup.kept();
        if self.setup == self.made && self.sample.x1.len() == self.setup.n {
            return;
        }
        if self.z_seed != self.setup.seed || self.z.len() != MOST {
            self.z = draws(self.setup.seed);
            self.z_seed = self.setup.seed;
        }
        let before = self.sample.marked;
        self.sample = Sample::new(&self.setup, &self.z);
        if self.sample.marked != before || self.setup.n != self.made.n || self.setup.seed != self.made.seed {
            self.lift = [0.0; 2];
        }
        self.made = self.setup;
        let on = self.cloud.on;
        self.cloud = Cloud::new(self.setup.seed);
        self.cloud.on = on;
    }

    /// The outputs as they are, with the marked points lifted.
    fn y(&self) -> Vec<f64> {
        self.sample.lifted(self.lift)
    }

    /// The fit of the sample as it is.
    fn fit(&self) -> Fit {
        fit(&self.sample.x1, &self.sample.x2, &self.y())
    }

    /// New samples, so many a second of the scene's time (the tests take them at once).
    fn draw_new(&mut self, steps: usize) -> usize {
        let mut taken = 0;
        while taken < steps && self.cloud.running() {
            if !self.cloud.step(&self.setup, &self.sample.x1, &self.sample.x2) {
                self.cloud.on = false;
                break;
            }
            taken += 1;
        }
        taken
    }
}

impl Scene for MultiReg {
    fn view(&self) -> View {
        self.the_view()
    }

    fn take_view(&mut self) -> Option<View> {
        std::mem::take(&mut self.new_view).then(|| self.the_view())
    }

    fn settings(&self, s: &mut ViewSettings) {
        s.grid = false;
    }

    fn animated(&self) -> bool {
        self.turning || self.cloud.running() || self.holding || self.look.shown != self.goal() || self.reach.shown.first().is_none_or(|r| *r != self.reach_goal as f32)
    }

    /// The points stand in a box: turn it slowly so that the plane's tilt can be read
    /// (not while a point is being dragged: the plane it is dragged in would turn too).
    fn auto_rotate(&self) -> f32 {
        if self.turning && !self.holding {
            0.12
        } else {
            0.0
        }
    }

    fn pointer_hint(&self) -> Option<&'static str> {
        Some("white points: drag up or down    elsewhere: drag to turn the box")
    }

    fn ui(&mut self, ui: &mut egui::Ui, _: &Gfx) {
        ui::note(ui, r"Two inputs, $x_1$ and $x_2$, on the floor, and an output $y$ upwards: one point for each case. Least squares fits the plane that makes the sum of the squared residuals (the pink lines, from each point straight up or down to the plane) as small as it can be:");
        ui::equation(ui, r"\hat y = b_0 + b_1 x_1 + b_2 x_2");
        ui::note(ui, r"$b_1$ is how much $\hat y$ rises for one more unit of $x_1$ with $x_2$ held fixed: the slope of the plane along $x_1$. It equals the slope of $y$ against $x_1$ alone only when the inputs are uncorrelated in the sample, or $b_2 = 0$. Drag a white point up or down; drag elsewhere to turn the box.");

        ui::section(ui, "Things to try");
        for (k, t) in TRIALS.iter().enumerate() {
            if ui::experiment(ui, t.name, t.look) {
                self.set_up(k);
            }
        }

        let before = self.setup;
        ui::section(ui, "The truth");
        slide(ui, r"$\beta_1$", &mut self.setup.b1, -2.0..=2.0);
        slide(ui, r"$\beta_2$", &mut self.setup.b2, -2.0..=2.0);
        slide(ui, r"bend $\gamma$", &mut self.setup.bend, 0.0..=1.5);
        slide(ui, r"noise $\sigma$", &mut self.setup.sigma, 0.0..=2.0);
        ui::note(ui, r"Each point's $y$ is made from its inputs as $\beta_1 x_1 + \beta_2 x_2 + \gamma\,(x_1^2 - 1) + \sigma\varepsilon$, with $\varepsilon$ a new standard normal draw for each point. The fit does not know this; $b_1$ and $b_2$ are its estimates of $\beta_1$ and $\beta_2$ (for a plane, $\gamma = 0$).");
        ui::section(ui, "The inputs");
        slide(ui, r"their correlation $\rho$", &mut self.setup.rho, -1.0..=1.0);
        ui::note(ui, r"Each input is a standard normal draw; $\rho$ says how closely they move together. At $\rho = \pm 1$ they are one input twice.");
        ui::slider_int_log(ui, "points", &mut self.setup.n, 3..=MOST);
        checkbox(ui, &mut self.setup.far, "put one point far out");
        ui.horizontal(|ui| {
            if ui.button("draw again").clicked() {
                self.setup.seed = self.setup.seed.wrapping_add(1);
            }
            checkbox(ui, &mut self.show_truth, "show the truth");
        });
        checkbox(ui, &mut self.turning, "turn the box slowly");
        if self.setup != before {
            self.tau = QUICK;
            self.refresh();
        }
        let tied = self.fit().b.is_none();
        if !tied && ui.button(format!("draw {NEW_SAMPLES} new samples")).clicked() {
            self.cloud = Cloud::new(self.setup.seed.wrapping_add(self.cloud.coef.len() as u64));
            self.cloud.on = true;
        }
        ui::note(ui, r"New samples have the same inputs and new noise: the chart of $b_1$ and $b_2$ shows how they vary from one to the next, which is what their standard errors measure.");

        ui::section(ui, "Reading the picture");
        ui::key(
            ui,
            &[
                (DATA, "cyan: the data, one point for each case, and its shadow on the floor, where its inputs are"),
                (ESTIMATE, r"yellow: what is fitted: the plane, its slopes $b_1$ along $x_1$ and $b_2$ along $x_2$, the lines in the charts with the other input held fixed, and the coefficients of new samples"),
                (OTHER, "orange: the straight line through $y$ against one input alone"),
                (ALARM, "pink: the residuals"),
                (TREND, "white curve: the residuals' trend, the parabola fitted through them: flat, but for the noise, when the plane is the right shape"),
                (TRUTH, "green: the truth: the surface the points were made from, and its coefficients"),
                (pal::WHITE, "white: the point with the most leverage and the one with the least, to drag; dashed yellow: the plane before they were moved"),
                (pal::DIM, "grey ellipse: where 95 in 100 of the coefficient pairs of new samples are expected to fall"),
            ],
        );
        ui::section(ui, "Words");
        ui::term(ui, "Coefficient:", r"$b_1$, the change in the fitted $\hat y$ for one more unit of $x_1$ with $x_2$ held fixed.");
        ui::term(ui, "Holding fixed (adjusting for):", r"comparing cases with the same $x_2$. The fit does it for every $x_2$ at once: $b_1$ is the slope of what $x_2$ leaves of $y$ against what $x_2$ leaves of $x_1$, each less its straight-line fit on $x_2$.");
        ui::term(ui, "Added-variable chart:", "that chart: one for each input. Its line's slope is the coefficient exactly, and its residuals are the plane's.");
        ui::term(ui, "Fitted value:", r"$\hat y$, the height of the plane over a point's inputs.");
        ui::term(ui, "Residual:", r"$y - \hat y$: how far a point is above (or below) the plane.");
        ui::term(ui, "Standard error:", "how much a coefficient varies from sample to sample, estimated from one sample: from the residuals' spread and how much of its input is left once the other input is known.");
        ui::term(ui, "$R^2$:", r"the share of the spread of $y$ about its mean that the fit accounts for: 1 less (sum of squared residuals) / (sum of squared distances of $y$ from its mean).");
        ui::term(ui, "Adjusted $R^2$:", r"$1 - (1 - R^2)(n - 1)/(n - p - 1)$ for $n$ points and $p$ inputs: $R^2$ less a charge for each input, so that it can fall when a useless one is added.");
        ui::term(ui, "Collinearity:", "inputs that move (nearly) together. Each can stand in for the other, so the data say little about how to share the effect between them: their coefficients are uncertain, their combined effect is not.");
        ui::term(ui, "Leverage:", r"how far out a point's inputs are, as a number $h$ from $1/n$ to 1: lift the point by 1 and the plane under it rises by $h$.");
        ui::term(ui, "Confounding:", "a slope that mixes in the effect of an input left out, one that moves with the input kept and also affects the output.");

        self.numbers(ui);
        ui::see_also(ui, &[("regress", "one input, and a curve"), ("lsq", "the fit as a projection"), ("confound", "what leaving out an input does"), ("modelselect", "how many inputs to keep")]);
    }

    fn frame(&mut self, f: &mut Frame) {
        let view = f.view_size();
        let lay = Layout::of(view, f.top_inset, self.page);
        if (view - self.canvas).abs().max_element() > 24.0 || f.top_inset != self.inset {
            (self.canvas, self.inset) = (view, f.top_inset);
            self.new_view = true;
        }
        self.holding = f.handles.dragging();
        self.refresh();

        // new samples, so many a second of the scene's time
        if self.cloud.running() && f.dt > 0.0 {
            let mut turn = scene_turn(&mut self.cloud.pace, SAMPLE_RATE, f.dt, f.real_dt);
            let mut taken = 0;
            while self.cloud.running() && turn.next() {
                taken += self.draw_new(1);
            }
            self.cloud.pace.end(turn);
            self.cloud.phase.after(SAMPLE_RATE, f.dt, taken);
        }

        // the picture glides to the sample as it is (by the clock: it answers the hand)
        if !self.holding {
            self.reach_goal = self.reach_wanted();
        }
        let goal = self.goal();
        if !self.look.follow(&goal, f.real_dt, self.tau) {
            self.tau = QUICK;
        }
        self.reach.follow(&[self.reach_goal as f32], f.real_dt, 0.12);

        let space = self.space(f, &lay);
        self.handles(f, &space);
        let d = self.drawn();
        self.draw_space(f, &lay, &space, &d);
        self.draw_charts(f, &lay, &d);
        self.bookkeeping(f);
    }
}

/// A slider for a setting kept in f64: the setting is written only when the slider is
/// moved (written back every frame, a value such as 0.8 would come back as the nearest
/// f32, a setting changed, and the sample made again).
fn slide(ui: &mut egui::Ui, label: &str, value: &mut f64, range: std::ops::RangeInclusive<f32>) {
    let mut v = *value as f32;
    if ui::slider(ui, label, &mut v, range) {
        *value = v as f64;
    }
}

// ----- the picture in space ----------------------------------------------------------------

/// The box in space as drawn this frame: where its middle is, world units up for a unit
/// of y, and the y at its top (the floor is at minus that).
struct Space {
    at: Vec3,
    k: f32,
    reach: f64,
    /// Whether the picture has room for words (not the sliver of it beside the
    /// controls, open on a phone).
    words: bool,
}

impl Space {
    fn w(&self, x1: f64, x2: f64, y: f64) -> Vec3 {
        self.at + Vec3::new(x1 as f32 * S, x2 as f32 * S, y as f32 * self.k)
    }
}

/// A polygon over the floor, as (x1, x2, y), cut to the heights from `lo` to `hi` (a
/// plane steeper than the box is cut where it leaves it).
fn cut(poly: &[[f64; 3]], lo: f64, hi: f64) -> Vec<[f64; 3]> {
    let mut pts = poly.to_vec();
    for (level, keep_above) in [(lo, true), (hi, false)] {
        let inside = |p: &[f64; 3]| if keep_above { p[2] >= level } else { p[2] <= level };
        let mut out = Vec::with_capacity(pts.len() + 2);
        for i in 0..pts.len() {
            let (a, b) = (pts[i], pts[(i + 1) % pts.len()]);
            if inside(&a) {
                out.push(a);
            }
            if inside(&a) != inside(&b) && (b[2] - a[2]).abs() > 0.0 {
                let t = (level - a[2]) / (b[2] - a[2]);
                out.push([a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t, level]);
            }
        }
        pts = out;
    }
    pts
}

/// The plane of (b0, b1, b2) over the floor, cut to the box's heights (a little
/// beyond them).
fn plane_over_floor(b: [f64; 3], reach: f64) -> Vec<[f64; 3]> {
    let corners = [(-REACH, -REACH), (REACH, -REACH), (REACH, REACH), (-REACH, REACH)];
    let poly: Vec<[f64; 3]> = corners.iter().map(|&(a, c)| [a, c, b[0] + b[1] * a + b[2] * c]).collect();
    cut(&poly, -reach * 1.15, reach * 1.15)
}

/// The part of the straight segment from `a` to `b` (each (x1, x2, y)) whose height is
/// within `lim` of 0, if any.
fn clip(a: [f64; 3], b: [f64; 3], lim: f64) -> Option<([f64; 3], [f64; 3])> {
    let d = [b[0] - a[0], b[1] - a[1], b[2] - a[2]];
    let (mut t0, mut t1) = (0.0f64, 1.0f64);
    if d[2].abs() > 0.0 {
        let (u, v) = ((-lim - a[2]) / d[2], (lim - a[2]) / d[2]);
        t0 = t0.max(u.min(v));
        t1 = t1.min(u.max(v));
    } else if a[2].abs() > lim || a[2].is_nan() {
        return None;
    }
    let at = |t: f64| [a[0] + d[0] * t, a[1] + d[1] * t, a[2] + d[2] * t];
    (t0 <= t1).then(|| (at(t0), at(t1)))
}

/// That part as a line in the batch `kind` of `lines3`: always two points, drawn clear
/// where nothing of the segment is in the box (so that the batch keeps its points).
#[allow(clippy::too_many_arguments)]
fn edge(f: &mut Frame, kind: u8, sp: &Space, a: [f64; 3], b: [f64; 3], lim: f64, width: f32, color: Color) {
    let (p, q, c) = match clip(a, b, lim) {
        Some((p, q)) => (p, q, color),
        None => (a, a, Color::CLEAR),
    };
    let at = |p: [f64; 3]| sp.w(p[0], p[1], p[2].max(-3.0 * lim).min(3.0 * lim));
    lines3(f, kind).segment(at(p), at(q), width, c);
}

impl MultiReg {
    /// Where the box is this frame: its middle off the camera's target along the
    /// camera's own right and up, so that it stays in the middle of the picture's part
    /// of the canvas while the view turns about it.
    fn space(&self, f: &Frame, lay: &Layout) -> Space {
        let view = f.view_size().max(Vec2::splat(1.0));
        let (centre, _) = lay.stage();
        let per_px = 2.0 * f.cam.dist() * (f.cam.fov_y * 0.5).tan() / view.y;
        let off = (centre - view * 0.5) * per_px;
        let at = f.cam.target() + f.cam.right * off.x - f.cam.up * off.y;
        let reach = self.reach.shown.first().copied().unwrap_or(1.0).max(0.1) as f64;
        let (min, max) = lay.pic;
        let words = (max - min).min_element() >= 160.0;
        Space { at: if at.is_finite() { at } else { Vec3::ZERO }, k: H / reach as f32, reach, words }
    }

    /// The two marked points, dragged up and down in the upright plane facing the eye.
    fn handles(&mut self, f: &mut Frame, sp: &Space) {
        let d = self.drawn();
        let facing = Vec3::new(f.cam.fwd.x, f.cam.fwd.y, 0.0).try_normalize().unwrap_or(-f.cam.fwd);
        for k in 0..2 {
            let i = self.sample.marked[k];
            if (k == 1 && i == self.sample.marked[0]) || i >= d.y.len() {
                continue;
            }
            let mut p = sp.w(d.x[0][i], d.x[1][i], d.y[i]);
            if f.drag3_sized(40 + k as u32, &mut p, Some(facing), pal::WHITE, 11.0) {
                let to = ((p.z - sp.at.z) / sp.k) as f64;
                if let Some(y) = self.sample.y.get(i) {
                    self.lift[k] = (to - y).max(-12.0).min(12.0);
                }
                self.look.snap(&self.goal());
            }
        }
    }

    fn draw_space(&self, f: &mut Frame, lay: &Layout, sp: &Space, d: &Drawn) {
        let (r, top) = (REACH, sp.reach);
        let floor = -top;
        // the floor, its grid, and the three axes along the edges from one corner
        let corners = [(-r, -r), (r, -r), (r, r), (-r, r)];
        let ring: Vec<Vec3> = corners.iter().map(|&(a, b)| sp.w(a, b, floor)).collect();
        f.dl.lines.closed(&ring, 1.0, pal::FAINT);
        for v in [-2.0, -1.0, 0.0, 1.0, 2.0] {
            f.line(sp.w(v, -r, floor), sp.w(v, r, floor), 1.0, pal::FAINT.a(0.45));
            f.line(sp.w(-r, v, floor), sp.w(r, v, floor), 1.0, pal::FAINT.a(0.45));
        }
        f.arrow(sp.w(-r, -r, floor), sp.w(r + 0.5, -r, floor), 1.6, pal::DIM);
        f.arrow(sp.w(-r, -r, floor), sp.w(-r, r + 0.5, floor), 1.6, pal::DIM);
        f.arrow(sp.w(-r, -r, floor), sp.w(-r, -r, top * 1.18), 1.6, pal::DIM);
        if sp.words {
            f.text_off(sp.w(r + 0.75, -r, floor), Vec2::ZERO, "$x_1$", 15.0, pal::TEXT, Align2::CENTER_CENTER);
            f.text_off(sp.w(-r, r + 0.75, floor), Vec2::ZERO, "$x_2$", 15.0, pal::TEXT, Align2::CENTER_CENTER);
            f.text_off(sp.w(-r, -r, top * 1.18), Vec2::new(0.0, -10.0), "$y$", 15.0, pal::TEXT, Align2::CENTER_BOTTOM);
            for v in [-2.0, 0.0, 2.0] {
                f.text_off(sp.w(v, -r, floor), Vec2::new(0.0, 6.0), num(v, 0), lay.small, pal::DIM, Align2::CENTER_TOP);
                f.text_off(sp.w(-r, v, floor), Vec2::new(0.0, 6.0), num(v, 0), lay.small, pal::DIM, Align2::CENTER_TOP);
            }
        }
        // (chosen from where the box's height is going, and kept while it glides there:
        // see `axes`; not the lowest, which is the floor's corner)
        let (step, values) = round_values(-self.reach_goal, self.reach_goal, 260.0);
        for v in values {
            let seen = v > -top * 0.99 && v <= top * 1.0001;
            let at = v.max(-top).min(top);
            lines3(f, TICKS).segment(sp.w(-r, -r, at), sp.w(-r + 0.18, -r, at), 1.0, if seen { pal::DIM } else { Color::CLEAR });
            if sp.words {
                f.text_off(sp.w(-r, -r, at), Vec2::new(-6.0, 0.0), if seen { tick_name(v, step) } else { String::new() }, lay.small, pal::DIM, Align2::RIGHT_CENTER);
            }
        }

        // the truth, as a net of curves (along x1 it bends, along x2 it is straight); what
        // leaves the box is drawn clear, not left out, so that the net keeps its points
        // while the box's height glides (see `lines3`)
        if self.show_truth {
            let s = self.setup;
            let lim = top * 1.15;
            for v in [-3.0, -1.5, 0.0, 1.5, 3.0] {
                let along: Vec<(Vec3, Color)> = (0..=24)
                    .map(|i| -r + 2.0 * r * i as f64 / 24.0)
                    .map(|a| (a, s.truth(a, v)))
                    .map(|(a, y)| (sp.w(a, v, y.max(-3.0 * lim).min(3.0 * lim)), if y.abs() <= lim { TRUTH.a(0.55) } else { Color::CLEAR }))
                    .collect();
                let net = lines3(f, NET);
                net.brk();
                for (p, c) in along {
                    net.push(p, 1.2, c);
                }
                net.brk();
                edge(f, NET, sp, [v, -r, s.truth(v, -r)], [v, r, s.truth(v, r)], lim, 1.2, TRUTH.a(0.55));
            }
        }

        let fd = &d.fit;
        let n = d.y.len();
        // the plane, its grid, and its slopes along each input through the middle of the points
        if let Some(b) = fd.b {
            let poly = plane_over_floor(b, top);
            let pts: Vec<Vec3> = poly.iter().map(|p| sp.w(p[0], p[1], p[2])).collect();
            for i in 1..pts.len().saturating_sub(1) {
                f.dl.tris_alpha.tri(pts[0], pts[i], pts[i + 1], ESTIMATE.a(0.055));
            }
            let at = |x1: f64, x2: f64| [x1, x2, b[0] + b[1] * x1 + b[2] * x2];
            let segment = |f: &mut Frame, a: (f64, f64), c: (f64, f64), width: f32, color: Color| edge(f, PLANE, sp, at(a.0, a.1), at(c.0, c.1), top * 1.15, width, color);
            for (a, c) in [((-r, -r), (r, -r)), ((r, -r), (r, r)), ((r, r), (-r, r)), ((-r, r), (-r, -r))] {
                segment(f, a, c, 1.6, ESTIMATE.a(0.8));
            }
            for v in [-2.0, -1.0, 1.0, 2.0] {
                segment(f, (-r, v), (r, v), 1.0, ESTIMATE.a(0.16));
                segment(f, (v, -r), (v, r), 1.0, ESTIMATE.a(0.16));
            }
            let (m1, m2) = (fd.mean[0], fd.mean[1]);
            segment(f, (-r, m2), (r, m2), 2.6, ESTIMATE);
            segment(f, (m1, -r), (m1, r), 2.6, ESTIMATE);
            let label = |f: &mut Frame, x1: f64, x2: f64, text: &str| {
                let y = b[0] + b[1] * x1 + b[2] * x2;
                if y.abs() <= top * 1.1 && sp.words {
                    f.text_off(sp.w(x1, x2, y), Vec2::new(0.0, -6.0), text, lay.title, ESTIMATE, Align2::CENTER_BOTTOM);
                }
            };
            label(f, r + 0.3, m2, "$b_1$");
            label(f, m1, r + 0.3, "$b_2$");
        } else if let Some((j, c, dd, settled)) = fd.tied() {
            // no one plane: the line every equally good plane passes through, and some of them
            let along = |t: f64| -> [f64; 3] {
                let (xj, xo) = (t, c * t + dd);
                let y = fd.mean[2] + settled * (xj - fd.mean[j]);
                if j == 0 {
                    [xj, xo, y]
                } else {
                    [xo, xj, y]
                }
            };
            let line: Vec<Vec3> = (0..=40)
                .map(|i| along(-2.0 * r + 4.0 * r * i as f64 / 40.0))
                .map(|p| if p[0].abs() <= r && p[1].abs() <= r && p[2].abs() <= top * 1.15 { sp.w(p[0], p[1], p[2]) } else { Vec3::NAN })
                .collect();
            let mut on = false;
            for p in line {
                if p.is_finite() {
                    lines3(f, TIED).push(p, 2.8, ESTIMATE);
                    on = true;
                } else if on {
                    lines3(f, TIED).brk();
                    on = false;
                }
            }
            lines3(f, TIED).brk();
            for t in [-1.0, -0.35, 0.35, 1.0] {
                // b_other = t, b_j = settled − c t: the same heights all along the line
                let bo = t * top / r;
                let bj = settled - c * bo;
                let b0 = fd.mean[2] - bj * fd.mean[j] - bo * fd.mean[1 - j];
                let b = if j == 0 { [b0, bj, bo] } else { [b0, bo, bj] };
                let pts: Vec<Vec3> = plane_over_floor(b, top).iter().map(|p| sp.w(p[0], p[1], p[2])).collect();
                for i in 1..pts.len().saturating_sub(1) {
                    f.dl.tris_alpha.tri(pts[0], pts[i], pts[i + 1], ESTIMATE.a(0.045));
                }
            }
        }

        // the plane before the marked points were moved
        if d.y.iter().zip(&d.y0).any(|(a, b)| (a - b).abs() > 1e-3) {
            if let Some(b) = d.plain.b {
                let at = |x1: f64, x2: f64| [x1, x2, b[0] + b[1] * x1 + b[2] * x2];
                for ((a1, a2), (c1, c2)) in [((-r, -r), (r, -r)), ((r, -r), (r, r)), ((r, r), (-r, r)), ((-r, r), (-r, -r))] {
                    match clip(at(a1, a2), at(c1, c2), top * 1.15) {
                        Some((p, q)) => f.dashed(sp.w(p[0], p[1], p[2]), sp.w(q[0], q[1], q[2]), 1.4, ESTIMATE.a(0.7)),
                        None => f.dashed(sp.w(a1, a2, 0.0), sp.w(a1, a2, 0.0), 1.4, Color::CLEAR),
                    }
                }
            }
        }

        // the planes of the latest new samples, fading as they age
        let planes = &self.cloud.planes;
        let latest = planes.len().saturating_sub(FAN);
        let phase = self.cloud.phase.share();
        for (age, b) in planes[latest..].iter().rev().enumerate() {
            let fade = (1.0 - (age as f32 + if self.cloud.running() { phase } else { 1.0 }) / (FAN as f32 + 1.0)).max(0.0);
            let pts: Vec<Vec3> = plane_over_floor(*b, top).iter().map(|p| sp.w(p[0], p[1], p[2])).collect();
            for i in 1..pts.len().saturating_sub(1) {
                f.dl.tris_alpha.tri(pts[0], pts[i], pts[i + 1], ESTIMATE.a(0.032 * fade));
            }
        }

        // the points, their shadows and their residuals
        for i in 0..n {
            let (a, b, y) = (d.x[0][i], d.x[1][i], d.y[i]);
            f.dl.points_with(PointKind::Disc, PointStyle::default()).push(sp.w(a, b, floor), 4.5, DATA.a(0.4));
            f.sphere(sp.w(a, b, y), 0.11, DATA);
            if let Some(h) = fd.fitted.get(i) {
                lines3(f, STICKS).segment(sp.w(a, b, y), sp.w(a, b, *h), 1.5, ALARM.a(0.85));
            }
        }
        // the marked points' leverage
        for k in 0..2 {
            let i = self.sample.marked[k];
            if (k == 1 && i == self.sample.marked[0]) || i >= n {
                continue;
            }
            if let Some(h) = fd.leverage(d.x[0][i], d.x[1][i]).filter(|_| sp.words) {
                f.text_off(sp.w(d.x[0][i], d.x[1][i], d.y[i]), Vec2::new(10.0, -8.0), format!("leverage ${}$", num(h, 2)), lay.small, pal::WHITE, Align2::LEFT_BOTTOM);
            }
        }
        self.draw_equation(f, lay);
    }

    /// The fit's equation and its numbers, over the top of the picture.
    fn draw_equation(&self, f: &mut Frame, lay: &Layout) {
        let fit = self.fit();
        let (min, max) = lay.pic;
        if max.x - min.x < 200.0 {
            return;
        }
        let size = lay.title;
        let at = min + Vec2::new(14.0, 12.0);
        let room = max.x - min.x - 28.0;
        let mut lines: Vec<(String, Color)> = Vec::new();
        match fit.b {
            Some([b0, b1, b2]) => {
                lines.push((format!(r"$\hat y = {} {} {}$", num(b0, 2), term(b1, "x_1"), term(b2, "x_2")), ESTIMATE));
                match fit.se() {
                    Some([s1, s2]) => lines.push((format!(r"$b_1 = {} \pm {}$,  $b_2 = {} \pm {}$  ($\pm$ one standard error)", num(b1, 2), num(s1, 2), num(b2, 2), num(s2, 2)), pal::TEXT)),
                    None => lines.push(("three points fix a plane: nothing is left over to measure the noise by".to_string(), pal::TEXT)),
                }
            }
            None => match fit.tied() {
                Some((j, c, dd, settled)) => {
                    let (xj, xo, bj, bo) = if j == 0 { ("x_1", "x_2", "b_1", "b_2") } else { ("x_2", "x_1", "b_2", "b_1") };
                    lines.push((format!(r"these two inputs cannot be told apart: ${xo} = {}\,{xj} {}$", num(c, 2), signed_number(dd)), ALARM));
                    lines.push((format!(r"every plane through the yellow line fits as well: only ${bj} {}\,{bo} = {}$ is settled", signed_number(c), num(settled, 2)), pal::TEXT));
                }
                None => lines.push(("the inputs do not vary: no slope can be fitted".to_string(), ALARM)),
            },
        }
        let r2 = match (fit.r2(), fit.adj_r2()) {
            (Some(r), Some(a)) => format!("$R^2 = {}$, adjusted ${}$", num(r, 3), num(a, 3)),
            (Some(r), None) => format!("$R^2 = {}$", num(r, 3)),
            _ => "$y$ does not vary: nothing to explain".to_string(),
        };
        let r = fit.r_inputs();
        lines.push((format!("{r2};  inputs' correlation $r = {}$", num(r, if r.abs() > 0.95 { 3 } else { 2 })), pal::TEXT));
        let mut y = at.y;
        for (text, color) in lines {
            // (a line too long for the picture's part of the canvas is left out)
            if f.text_size(&text, size).x <= room {
                f.caption_px(Vec2::new(at.x, y), text.as_str(), size, color, Align2::LEFT_TOP);
                y += size + 9.0;
            }
        }
        // what the colours are, along the foot of the picture
        let mut key = vec![(DATA, "data"), (ESTIMATE, "fitted plane"), (ALARM, "residuals"), (pal::WHITE, "drag up or down")];
        if self.show_truth {
            key.push((TRUTH, "the truth"));
        }
        if max.y - min.y > 220.0 {
            key_row(f, Vec2::new(min.x + 14.0, max.y - 8.0), max.x - min.x - 28.0, &key, lay.small);
        }
    }
}

/// A colour key in a row (on as many rows as it needs, the last ending at `bottom_left`),
/// no wider than `room` pixels.
fn key_row(f: &mut Frame, bottom_left: Vec2, room: f32, entries: &[(Color, &str)], size: f32) {
    let widths: Vec<f32> = entries.iter().map(|(_, t)| size + 5.0 + f.text_size(t, size).x + 16.0).collect();
    // which row each entry goes in
    let mut rows = vec![0usize; entries.len()];
    let (mut row, mut used) = (0, 0.0);
    for (i, w) in widths.iter().enumerate() {
        if used > 0.0 && used + w > room {
            (row, used) = (row + 1, 0.0);
        }
        rows[i] = row;
        used += w;
    }
    let line = size + 7.0;
    let mut x = bottom_left.x;
    for (i, (color, text)) in entries.iter().enumerate() {
        if i > 0 && rows[i] != rows[i - 1] {
            x = bottom_left.x;
        }
        let y = bottom_left.y - (row - rows[i]) as f32 * line - size * 0.5;
        hud_rect(f, Vec2::new(x, y - size * 0.4), Vec2::new(x + size * 0.8, y + size * 0.4), *color);
        f.text_px(Vec2::new(x + size + 3.0, y), *text, size, pal::TEXT.a(0.85), Align2::LEFT_CENTER);
        x += widths[i];
    }
}

/// Lines in space whose number of points can change from one frame to the next (a
/// plane cut where it leaves the box, a net cut likewise), each kind in a batch of its
/// own: `eidos motion` follows a batch's points by their places in it, and pieces of one
/// batch that gain and lose points in the same frame would put every point after them
/// in another's place.
fn lines3<'b>(f: &'b mut Frame<'_>, kind: u8) -> &'b mut crate::gfx::draw::LineBatch {
    f.dl.lines_with(LineKind::Blend, LineStyle { fade_power: 90.0 + kind as f32, ..Default::default() })
}

/// The kinds of `lines3`: the truth's net, the fitted plane, the residuals, the line of
/// equally good planes.
const NET: u8 = 0;
const PLANE: u8 = 1;
const STICKS: u8 = 2;
const TIED: u8 = 3;
/// The round values up the box's axis.
const TICKS: u8 = 4;

/// How many planes of the latest new samples are drawn in space.
const FAN: usize = 6;

/// `+ 1.50\,x_1` or `- 0.48\,x_1`: a term of the fitted equation.
fn term(v: f64, x: &str) -> String {
    let sign = if v < 0.0 && num(v, 2).starts_with('−') { "-" } else { "+" };
    format!(r"{sign} {}\,{x}", num(v.abs(), 2))
}

/// `+ 0.50` or `- 0.50`, for a constant after a term.
fn signed_number(v: f64) -> String {
    let sign = if v < 0.0 && num(v, 2).starts_with('−') { "-" } else { "+" };
    format!("{sign} {}", num(v.abs(), 2))
}

// ----- the charts --------------------------------------------------------------------------

/// The smallest and largest of some values (0 to 1 if there are none, or they are not numbers).
fn extent(v: &[f64]) -> (f64, f64) {
    let (lo, hi) = v.iter().filter(|x| x.is_finite()).fold((f64::MAX, f64::MIN), |(a, b), &x| (a.min(x), b.max(x)));
    if lo <= hi {
        (lo, hi)
    } else {
        (0.0, 1.0)
    }
}

/// How far some values reach either way from `centre` (never less than `least`).
fn half_width(v: &[f64], centre: f64, least: f64) -> f64 {
    v.iter().filter(|x| x.is_finite()).fold(least, |m, x| m.max((x - centre).abs()))
}

/// The first of some texts that fits in `room` pixels at `size` (the last if none does).
fn fitting(f: &Frame, options: &[String], room: f32, size: f32) -> String {
    options.iter().find(|t| f.text_size(t, size).x <= room).or(options.last()).cloned().unwrap_or_default()
}

impl Drawn {
    fn of(x1: &[f64], x2: &[f64], y0: &[f64], y: &[f64]) -> Drawn {
        Drawn { x: [x1.to_vec(), x2.to_vec()], y0: y0.to_vec(), y: y.to_vec(), fit: fit(x1, x2, y), plain: fit(x1, x2, y0) }
    }

    /// The ranges of the two charts of input `j`'s row: across and up the chart of the
    /// input alone, and of the chart with the other held fixed. The two have the same
    /// widths, so that their slopes can be compared by eye.
    fn row_ranges(&self, j: usize) -> [[f64; 2]; 4] {
        let (x, other) = (&self.x[j], &self.x[1 - j]);
        let (ex, ey) = (left_of(x, other), left_of(&self.y, other));
        let (mx, my) = (self.fit.mean[j], self.fit.mean[2]);
        let hx = half_width(x, mx, 0.0).max(half_width(&ex, 0.0, 0.0)).max(0.5) * 1.08;
        let hy = half_width(&self.y, my, 0.0).max(half_width(&ey, 0.0, 0.0)).max(0.5) * 1.08;
        [[mx - hx, mx + hx], [my - hy, my + hy], [-hx, hx], [-hy, hy]]
    }

    /// The ranges of the chart of the residuals against the fitted values.
    fn residual_ranges(&self) -> [[f64; 2]; 2] {
        let (lo, hi) = extent(&self.fit.fitted);
        let pad = ((hi - lo) * 0.06).max(0.2);
        let h = half_width(&self.fit.resid, 0.0, 0.3) * 1.15;
        [[lo - pad, hi + pad], [-h, h]]
    }
}

/// The 95th percentile of the chi-square distribution with two degrees of freedom.
fn chi_95() -> f64 {
    st::Dist::ChiSquare { df: 2.0 }.quantile(0.95)
}

/// The covariance of b1 and b2 over new samples with these inputs (σ² times the inverse
/// of the inputs' sums of squares and products): (var b1, cov, var b2), where there is
/// one; and the ranges of the chart of them, round the truth's coefficients, wide
/// enough for the ellipse where 95 in 100 fall and for this sample's `own` pair, and the
/// same scale both ways on `rect`.
fn sample_ranges(s: &Setup, fd: &Fit, own: Option<[f64; 2]>, rect: Rect) -> (Option<[f64; 3]>, [[f64; 2]; 2]) {
    let det = fd.s11 * fd.s22 - fd.s12 * fd.s12;
    let v = s.sigma * s.sigma;
    let cov = (fd.b.is_some() && det > 0.0).then(|| [v * fd.s22 / det, -v * fd.s12 / det, v * fd.s11 / det]);
    let chi = chi_95();
    let mut hx = cov.map_or(0.3, |c| (chi * c[0]).sqrt() * 1.3).max(0.3);
    let mut hy = cov.map_or(0.3, |c| (chi * c[2]).sqrt() * 1.3).max(0.3);
    if let Some([a, b]) = own {
        hx = hx.max((a - s.b1).abs() * 1.15);
        hy = hy.max((b - s.b2).abs() * 1.15);
    }
    let (w, h) = ((rect.1.x - rect.0.x).max(1.0) as f64, (rect.1.y - rect.0.y).max(1.0) as f64);
    let per = (hx / w).max(hy / h);
    (cov, [[s.b1 - per * w, s.b1 + per * w], [s.b2 - per * h, s.b2 + per * h]])
}

/// Round values for an axis going to `lo`..`hi` over `pixels`: the step `Plot`'s axes
/// would choose, and every multiple of it from three steps below `lo` to three above `hi`
/// (sixty at most).
fn round_values(lo: f64, hi: f64, pixels: f32) -> (f64, Vec<f64>) {
    let step = plot::Ticks::of(lo, hi, pixels).step;
    if !(lo.is_finite() && hi.is_finite() && hi > lo && step > 0.0) {
        return (step, Vec::new());
    }
    let first = (lo / step).floor() as i64 - 3;
    let last = ((hi / step).ceil() as i64 + 3).min(first + 60);
    (step, (first..=last).map(|k| k as f64 * step).collect())
}

/// A tick's value written with the digits its step needs (as `Ticks::name` writes it).
fn tick_name(v: f64, step: f64) -> String {
    num(v, (-step.log10().floor()).max(0.0).min(6.0) as usize)
}

/// A chart's axes, grid, tick numbers and names, as `Plot::x_axis`, `y_axis` and `y_grid`
/// draw them, but with the round values chosen from where the chart's ranges are going
/// (`goal`), a few beyond each end, and kept while its ranges glide there: each tick
/// keeps its place in its batch and slides with the axis, unseen while it is off the
/// chart. (Chosen afresh in every frame, ticks would come and go at the ends of a gliding
/// axis, and the one that came would be taken for the one that went, moved.)
fn axes(f: &mut Frame, p: &Plot, goal: &Plot, across: &str, up: &str, size: f32) {
    if p.tiny() {
        return;
    }
    let (min, max) = p.rect;
    let (line, grid) = (pal::DIM.a(0.8), pal::DIM.a(0.14));
    plot::line_of(f, AXES, Vec2::new(min.x, max.y), max, 1.0, line);
    plot::line_of(f, AXES, min, Vec2::new(min.x, max.y), 1.0, line);
    // (one off the chart waits at its edge, unseen, rather than racing along beyond it)
    let (step, values) = round_values(goal.x[0], goal.x[1], goal.width());
    for v in values {
        let x = p.px_x(v);
        let seen = x >= min.x - 0.5 && x <= max.x + 0.5;
        let x = if x.is_finite() { x.max(min.x).min(max.x) } else { min.x };
        plot::line_of(f, AXES, Vec2::new(x, max.y), Vec2::new(x, max.y + 4.0), 1.0, if seen { line } else { Color::CLEAR });
        f.text_px(Vec2::new(x, max.y + 6.0), if seen { tick_name(v, step) } else { String::new() }, size, pal::DIM, Align2::CENTER_TOP);
    }
    let (step, values) = round_values(goal.y[0], goal.y[1], goal.height() * 1.6);
    for v in values {
        let y = p.px_y(v);
        let seen = y >= min.y - 0.5 && y <= max.y + 0.5;
        let y = if y.is_finite() { y.max(min.y).min(max.y) } else { max.y };
        plot::line_of(f, AXES, Vec2::new(min.x - 4.0, y), Vec2::new(min.x, y), 1.0, if seen { line } else { Color::CLEAR });
        plot::line_of(f, AXES, Vec2::new(min.x, y), Vec2::new(max.x, y), 1.0, if seen { grid } else { Color::CLEAR });
        f.text_px(Vec2::new(min.x - 6.0, y), if seen { tick_name(v, step) } else { String::new() }, size, pal::DIM, Align2::RIGHT_CENTER);
    }
    if !across.is_empty() {
        f.text_px(Vec2::new((min.x + max.x) * 0.5, max.y + 8.0 + size), across, size, pal::DIM, Align2::CENTER_TOP);
    }
    if !up.is_empty() {
        f.text_px(Vec2::new(min.x - 30.0, min.y - 6.0), up, size, pal::DIM, Align2::LEFT_BOTTOM);
    }
}

impl MultiReg {
    fn draw_charts(&mut self, f: &mut Frame, lay: &Layout, d: &Drawn) {
        self.plots.clear();
        let truth_fit = self.fit();
        // where the charts' ranges are going: their round values are chosen from these
        let goal = self.goal_drawn();
        // the tabs, where the charts come a page at a time
        if let Some(at) = lay.tabs {
            let rects = plot::button_row(f, &PAGES, at.x, at.y, TAB_H - 4.0, lay.small, 6.0);
            for (k, r) in rects.into_iter().enumerate() {
                if plot::button(f, r, PAGES[k], lay.small, self.page == k) {
                    self.page = k;
                }
            }
        }
        for j in 0..2 {
            self.draw_row(f, lay, d, &goal, &truth_fit, j);
        }
        if let Some(rect) = lay.charts[RESIDUALS] {
            self.draw_residuals(f, lay, d, &goal, rect);
        }
        if let Some(rect) = lay.charts[SAMPLES] {
            self.draw_samples(f, lay, d, &goal, &truth_fit, rect);
        }
    }

    /// Input `j` alone, and with the other held fixed: the two charts of one row, drawn
    /// to the same scale so that their slopes can be compared by eye.
    fn draw_row(&mut self, f: &mut Frame, lay: &Layout, d: &Drawn, goal: &Drawn, t: &Fit, j: usize) {
        let (alone_at, held_at) = (lay.charts[ALONE_1 + 2 * j], lay.charts[HELD_1 + 2 * j]);
        if alone_at.is_none() && held_at.is_none() {
            return;
        }
        let (x, other) = (&d.x[j], &d.x[1 - j]);
        let (ex, ey) = (left_of(x, other), left_of(&d.y, other));
        let fd = &d.fit;
        let [alone_x, alone_y, held_x, held_y] = d.row_ranges(j);
        let to = goal.row_ranges(j);
        let (xn, on) = if j == 0 { ("x_1", "x_2") } else { ("x_2", "x_1") };
        let bn = if j == 0 { "b_1" } else { "b_2" };
        let size = lay.small;
        if let Some(rect) = alone_at {
            let (p, g) = (Plot::new(rect, alone_x, alone_y), Plot::new(rect, to[0], to[1]));
            self.plots.push(g);
            p.backing(f, LEFT, BELOW, ABOVE, RIGHT);
            let slope = t.alone(j).map(|(s, _)| num(s, 2)).unwrap_or_else(|| "none".to_string());
            let r2 = t.alone_r2(j).map(|r| format!(", $R^2 = {}$", num(r, 3))).unwrap_or_default();
            let title = fitting(f, &[format!("${xn}$ alone: slope ${slope}${r2}"), format!("${xn}$ alone: slope ${slope}$"), format!("alone: ${slope}$")], p.width() + LEFT - 4.0, lay.title);
            p.title(f, &title, lay.title);
            axes(f, &p, &g, &format!("${xn}$"), "$y$", size);
            for i in 0..x.len().min(d.y.len()) {
                p.dot(f, x[i], d.y[i], 4.0, DATA);
            }
            if let Some((s, c)) = fd.alone(j) {
                let [a, b] = alone_x;
                p.path(f, CURVES, [(a, c + s * a), (b, c + s * b)], 2.2, OTHER);
            }
        }
        if let Some(rect) = held_at {
            let (p, g) = (Plot::new(rect, held_x, held_y), Plot::new(rect, to[2], to[3]));
            self.plots.push(g);
            p.backing(f, LEFT, BELOW, ABOVE, RIGHT);
            let coef = t.b.map(|b| num(b[1 + j], 2));
            let title = match &coef {
                Some(c) => fitting(f, &[format!("${on}$ held fixed: slope ${bn} = {c}$"), format!("${on}$ held: ${bn} = {c}$"), format!("${bn} = {c}$")], p.width() + LEFT - 4.0, lay.title),
                None => fitting(f, &[format!("${on}$ held fixed: no slope"), "no slope".to_string()], p.width() + LEFT - 4.0, lay.title),
            };
            p.title(f, &title, lay.title);
            let names = |f: &Frame, long: String, short: String, room: f32| fitting(f, &[long, short], room, size);
            let across = names(f, format!("what ${on}$ leaves of ${xn}$"), format!("${xn}$, ${on}$ out"), p.width());
            let up = names(f, format!("what ${on}$ leaves of $y$"), format!("$y$, ${on}$ out"), p.width() + LEFT - 4.0);
            axes(f, &p, &g, &across, &up, size);
            for i in 0..ex.len().min(ey.len()) {
                p.dot(f, ex[i], ey[i], 4.0, DATA);
            }
            match fd.b.map(|b| b[1 + j]) {
                Some(s) => {
                    let [a, b] = held_x;
                    p.path(f, CURVES, [(a, s * a), (b, s * b)], 2.2, ESTIMATE);
                }
                None => {
                    let text = format!("nothing is left of ${xn}$ once ${on}$ is known");
                    let mid = (rect.0 + rect.1) * 0.5;
                    f.caption_px(mid, fitting(f, &[text, "nothing left".to_string()], p.width() - 8.0, size), size, ALARM, Align2::CENTER_CENTER);
                }
            }
        }
    }

    /// The residuals against the fitted values, with their trend.
    fn draw_residuals(&mut self, f: &mut Frame, lay: &Layout, d: &Drawn, goal: &Drawn, rect: Rect) {
        let fd = &d.fit;
        let [across, up] = d.residual_ranges();
        let to = goal.residual_ranges();
        let (p, g) = (Plot::new(rect, across, up), Plot::new(rect, to[0], to[1]));
        self.plots.push(g);
        let size = lay.small;
        p.backing(f, LEFT, BELOW, ABOVE, RIGHT);
        let title = fitting(f, &["residuals against fitted values".to_string(), "residuals".to_string()], p.width() + LEFT - 4.0, lay.title);
        p.title(f, &title, lay.title);
        axes(f, &p, &g, r"fitted value $\hat y$", r"residual $y - \hat y$", size);
        p.hline(f, MARKS, 0.0, 1.6, ESTIMATE, false);
        for (v, r) in fd.fitted.iter().zip(&fd.resid) {
            p.path(f, MARKS, [(*v, 0.0), (*v, *r)], 1.2, ALARM.a(0.7));
        }
        for (v, r) in fd.fitted.iter().zip(&fd.resid) {
            p.dot(f, *v, *r, 4.0, DATA);
        }
        // the residuals' trend: the parabola through them (a smoother would wiggle with
        // the noise and show a pattern where there is none)
        let (lo, hi) = extent(&fd.fitted);
        if let Some(c) = parabola(&fd.fitted, &fd.resid) {
            p.curve(f, CURVES, 48, 1.6, TREND, |v| if v >= lo && v <= hi { c.at(v) } else { f64::NAN });
            p.text(f, hi, c.at(hi), Vec2::new(-4.0, -6.0), "trend", size, TREND, Align2::RIGHT_BOTTOM);
        }
    }

    /// The coefficients of new samples with the same inputs, round the truth's.
    fn draw_samples(&mut self, f: &mut Frame, lay: &Layout, d: &Drawn, goal: &Drawn, t: &Fit, rect: Rect) {
        let s = self.setup;
        let size = lay.small;
        let fd = &d.fit;
        let own = t.b.map(|b| [b[1], b[2]]);
        let (cov, [across, up]) = sample_ranges(&s, fd, own, rect);
        let (_, to) = sample_ranges(&s, &goal.fit, own, rect);
        let (p, g) = (Plot::new(rect, across, up), Plot::new(rect, to[0], to[1]));
        self.plots.push(g);
        p.backing(f, LEFT, BELOW, ABOVE, RIGHT);
        let count = self.cloud.coef.len();
        let title = if count > 0 { format!("$b_1$, $b_2$ of {count} new samples") } else { "$b_1$, $b_2$ of new samples".to_string() };
        let button = format!("draw {NEW_SAMPLES} new samples");
        let button_w = f.text_size(&button, size).x + 18.0;
        let title = fitting(f, &[title, "new samples".to_string()], p.width() + LEFT - button_w - 12.0, lay.title);
        p.title(f, &title, lay.title);
        axes(f, &p, &g, "$b_1$", "$b_2$", size);
        if fd.b.is_none() {
            let mid = (rect.0 + rect.1) * 0.5;
            let text = fitting(f, &["the inputs cannot be told apart: no coefficients".to_string(), "no coefficients".to_string()], p.width() - 8.0, size);
            f.caption_px(mid, text, size, ALARM, Align2::CENTER_CENTER);
            return;
        }
        if let Some(c) = cov {
            // the ellipse: the square root of the covariance (a triangular one) times a circle
            let l11 = c[0].sqrt();
            let l21 = if l11 > 0.0 { c[1] / l11 } else { 0.0 };
            let l22 = (c[2] - l21 * l21).max(0.0).sqrt();
            let k = chi_95().sqrt();
            let ring = (0..=72).map(|i| i as f64 / 72.0 * std::f64::consts::TAU).map(|a| (s.b1 + k * l11 * a.cos(), s.b2 + k * (l21 * a.cos() + l22 * a.sin())));
            p.path(f, CURVES, ring, 1.4, pal::DIM);
        }
        // the new samples (in a batch of their own: their number grows)
        let style = PointStyle { outline_px: 0.0, intensity: 0.99, ..Default::default() };
        for c in &self.cloud.coef {
            let at = p.at(c[0], c[1]);
            if at.is_finite() && plot::within(at, rect, 0.5) {
                let q = f.hud(at);
                f.dl.points_with(PointKind::Overlay, style).push(q, 3.5, ESTIMATE.a(0.55));
            }
        }
        // the truth, and this sample's own pair
        let (tx, ty) = (p.px_x(s.b1), p.px_y(s.b2));
        plot::line_of(f, MARKS, Vec2::new(tx - 6.0, ty), Vec2::new(tx + 6.0, ty), 2.0, TRUTH);
        plot::line_of(f, MARKS, Vec2::new(tx, ty - 6.0), Vec2::new(tx, ty + 6.0), 2.0, TRUTH);
        // (the two names on opposite sides, away from each other, unless one would leave
        // the chart that way: below the truth, above this sample)
        let right = own.is_none_or(|[a, _]| a <= s.b1);
        let side = |f: &Frame, x: f32, text: &str, right: bool| {
            let w = f.text_size(text, size).x + 10.0;
            let (fits_right, fits_left) = (x + w <= rect.1.x, x - w >= rect.0.x);
            fits_right && (right || !fits_left)
        };
        let towards = |right: bool, down: f32| if right { (Vec2::new(9.0, down), if down > 0.0 { Align2::LEFT_TOP } else { Align2::LEFT_BOTTOM }) } else { (Vec2::new(-9.0, down), if down > 0.0 { Align2::RIGHT_TOP } else { Align2::RIGHT_BOTTOM }) };
        let (to, align) = towards(side(f, p.px_x(s.b1), "truth", right), 7.0);
        p.text(f, s.b1, s.b2, to, "truth", size, TRUTH, align);
        if let Some([a, b]) = own {
            p.dot(f, a, b, 9.0, ESTIMATE);
            let (to, align) = towards(side(f, p.px_x(a), "this sample", !right), -6.0);
            p.text(f, a, b, to, "this sample", size, ESTIMATE, align);
        }
        // the button that draws them
        let label = if self.cloud.running() {
            "stop".to_string()
        } else if count >= NEW_SAMPLES {
            "again".to_string()
        } else {
            button
        };
        // (as wide whatever it says, so that nothing moves when it changes)
        let r = (Vec2::new(rect.1.x - button_w, rect.0.y - ABOVE + 4.0), Vec2::new(rect.1.x, rect.0.y - ABOVE + 24.0));
        if plot::button(f, r, &label, size, self.cloud.running()) {
            if self.cloud.running() {
                self.cloud.on = false;
            } else {
                self.cloud = Cloud::new(self.setup.seed.wrapping_add(count as u64 + 1));
                self.cloud.on = true;
            }
        }
    }

    /// New round values on an axis: make sure the batches they are drawn in and the
    /// labels are not as many as in the last frame, so that nothing takes the new ticks
    /// for the old ones moved.
    fn bookkeeping(&mut self, f: &mut Frame) {
        use std::hash::{Hash, Hasher};
        let hash = |values: &mut dyn Iterator<Item = f64>| {
            let mut h = std::collections::hash_map::DefaultHasher::new();
            values.for_each(|v| v.to_bits().hash(&mut h));
            h.finish()
        };
        // (as the charts and the box choose them: see `Plot::x_axis`, `Plot::y_axis`)
        let charts = hash(&mut self.plots.iter().flat_map(|p| {
            let (across, up) = (round_values(p.x[0], p.x[1], p.width()).1, round_values(p.y[0], p.y[1], p.height() * 1.6).1);
            across.into_iter().chain([f64::NAN]).chain(up).collect::<Vec<f64>>()
        }));
        let space = hash(&mut round_values(-self.reach_goal, self.reach_goal, 260.0).1.into_iter());
        let counts = |f: &mut Frame| (plot::apart(f, AXES, false).pts.len(), lines3(f, TICKS).pts.len(), f.dl.labels.len());
        let (axes, ticks, labels) = counts(f);
        let far = f.hud(Vec2::ZERO);
        if charts != self.ticks.0 && axes == self.drawn_last.0 {
            plot::apart(f, AXES, false).segment(far, far, 1.0, Color::CLEAR);
        }
        if space != self.ticks.1 && ticks == self.drawn_last.1 {
            lines3(f, TICKS).segment(far, far, 1.0, Color::CLEAR);
        }
        if (charts, space) != self.ticks && labels == self.drawn_last.2 {
            f.text_px(Vec2::ZERO, "", 1.0, Color::CLEAR, Align2::LEFT_TOP);
        }
        self.ticks = (charts, space);
        self.drawn_last = counts(f);
    }

    /// The numbers, in the panel.
    fn numbers(&self, ui: &mut egui::Ui) {
        let fit = self.fit();
        ui::section(ui, "Numbers");
        match (fit.b, fit.se()) {
            (Some([b0, b1, b2]), se) => {
                let with = |v: f64, k: usize| match se {
                    Some(s) => format!("${} \\pm {}$", num(v, 3), num(s[k], 3)),
                    None => format!("${}$", num(v, 3)),
                };
                ui::readout_colored(ui, r"$b_1$ ($\pm$ its standard error)", with(b1, 0), ESTIMATE);
                ui::readout_colored(ui, "$b_2$", with(b2, 1), ESTIMATE);
                ui::readout(ui, "$b_0$", format!("${}$", num(b0, 3)));
            }
            _ => {
                if let Some((j, c, _, settled)) = fit.tied() {
                    let (bj, bo) = if j == 0 { ("b_1", "b_2") } else { ("b_2", "b_1") };
                    ui::readout_colored(ui, &format!("${bj} {}\\,{bo}$", signed_number(c)), format!("${}$", num(settled, 3)), ALARM);
                    ui::note(ui, "The inputs cannot be told apart: only this combination of the coefficients is settled.");
                }
            }
        }
        for j in 0..2 {
            let (xn, other) = if j == 0 { ("x_1", "x_2") } else { ("x_2", "x_1") };
            if let (Some((slope, _)), Some(d)) = (fit.alone(j), fit.other_on(j)) {
                ui::readout_colored(ui, &format!("slope on ${xn}$ alone"), format!("${}$", num(slope, 3)), OTHER);
                if let Some(b) = fit.b {
                    let (bj, bo) = (b[1 + j], b[2 - j]);
                    ui::note(ui, &format!(r"$= b_{} + b_{} \times {}$, where ${}$ is how much ${other}$ rises with ${xn}$: ${} = {} + ({}) \times {}$", j + 1, 2 - j, num(d, 3), num(d, 3), num(slope, 3), num(bj, 3), num(bo, 3), num(d, 3)));
                }
            }
        }
        let r2 = |v: Option<f64>| v.map_or("none".to_string(), |r| format!("${}$", num(r, 3)));
        ui::readout(ui, "$R^2$", r2(fit.r2()));
        ui::readout(ui, "adjusted $R^2$", r2(fit.adj_r2()));
        ui::readout(ui, "$R^2$ with $x_1$ alone, adjusted", format!("{}, {}", r2(fit.alone_r2(0)), r2(fit.alone_r2(0).and_then(|r| adjusted(r, fit.n, 1)))));
        ui::readout(ui, r"residuals' spread $\hat\sigma$", fit.sigma_hat().map_or("none".to_string(), |s| format!("${}$", num(s, 3))));
        let r = fit.r_inputs();
        ui::readout(ui, "inputs' correlation $r$", format!("${}$", num(r, 3)));
        if r.abs() < 1.0 && fit.b.is_some() {
            ui::readout(ui, r"$1/\sqrt{1 - r^2}$", format!("${}$", num(1.0 / (1.0 - r * r).sqrt(), 2)));
            ui::note(ui, r"Inputs that move together make each coefficient's standard error larger by $1/\sqrt{1 - r^2}$ than unrelated inputs with the same spreads would.");
        }
        let marked = &self.sample.marked;
        let h = |k: usize| marked.get(k).and_then(|&i| fit.leverage(*self.sample.x1.get(i)?, *self.sample.x2.get(i)?));
        if let (Some(far), Some(near)) = (h(0), h(1)) {
            ui::readout(ui, "leverage of the white points", format!("${}$ and ${}$", num(far, 3), num(near, 3)));
        }
        if let Some(([s1, s2, sum], flipped)) = self.cloud.spreads(&self.setup) {
            ui::readout(ui, "new samples", self.cloud.coef.len().to_string());
            ui::readout(ui, "their spread: $b_1$, $b_2$, $b_1 + b_2$", format!("${}$, ${}$, ${}$", num(s1, 3), num(s2, 3), num(sum, 3)));
            ui::readout(ui, r"$b_1$ of the other sign from $\beta_1$", format!("{} %", num(100.0 * flipped, 0)));
        }
    }
}

/// A parabola fitted by least squares, in a variable centred and scaled (`mid`, `half`)
/// so that its equations are well conditioned.
struct Parabola {
    c: [f64; 3],
    mid: f64,
    half: f64,
}

impl Parabola {
    fn at(&self, v: f64) -> f64 {
        let u = (v - self.mid) / self.half;
        self.c[0] + self.c[1] * u + self.c[2] * u * u
    }
}

/// The parabola through the points (v, r): none for fewer than four points, or values
/// of v that do not spread.
fn parabola(v: &[f64], r: &[f64]) -> Option<Parabola> {
    let n = v.len().min(r.len());
    let (lo, hi) = extent(&v[..n]);
    if n < 4 || hi <= lo {
        return None;
    }
    let (mid, half) = ((lo + hi) * 0.5, (hi - lo) * 0.5);
    let (mut a, mut b) = (vec![0.0; 9], vec![0.0; 3]);
    for i in 0..n {
        let u = (v[i] - mid) / half;
        let p = [1.0, u, u * u];
        for row in 0..3 {
            b[row] += p[row] * r[i];
            for col in 0..3 {
                a[row * 3 + col] += p[row] * p[col];
            }
        }
    }
    let c = crate::math::linalg::solve(&mut a, &mut b, 3)?;
    (c.iter().all(|x| x.is_finite())).then(|| Parabola { c: [c[0], c[1], c[2]], mid, half })
}

/// The colour of the residuals' trend.
const TREND: Color = Color::hex(0xC9D1E0);

#[cfg(test)]
mod tests {
    use super::*;

    use crate::math::causal;

    /// The scene as experiment `k` sets it up, its new samples all drawn.
    fn trial(k: usize) -> MultiReg {
        let mut m = MultiReg::create();
        m.set_up(k);
        m.draw_new(NEW_SAMPLES);
        m
    }

    /// The scene as experiment `k` sets it up, with other settings.
    fn varied(k: usize, change: impl Fn(&mut Setup)) -> MultiReg {
        let mut m = MultiReg::create();
        m.set_up(k);
        change(&mut m.setup);
        m.refresh();
        m
    }

    /// The slope of the line through the origin that fits these points best.
    fn slope_through_origin(x: &[f64], y: &[f64]) -> Option<f64> {
        let (xx, xy) = x.iter().zip(y).fold((0.0, 0.0), |(xx, xy), (a, b)| (xx + a * a, xy + a * b));
        (xx > 0.0).then(|| xy / xx)
    }

    /// `v` as a caption writes it, to `places` decimals.
    fn says(v: f64, places: usize) -> String {
        num(v, places)
    }

    fn close(a: f64, b: f64, tolerance: f64) -> bool {
        (a - b).abs() <= tolerance * (1.0 + a.abs().max(b.abs()))
    }

    /// The fit is least squares: it agrees with the library's regression on several
    /// inputs, and what it leaves sums to zero and is at right angles to both inputs.
    #[test]
    fn the_fit_is_least_squares() {
        for k in 0..TRIALS.len() {
            for (seed, rho) in [(322, None), (5, Some(0.5)), (77, Some(-0.9)), (8, Some(0.999))] {
                let m = varied(k, |s| {
                    s.seed = seed;
                    if let Some(r) = rho {
                        s.rho = r;
                    }
                });
                let (y, f) = (m.y(), m.fit());
                let b = f.b.expect("a plane");
                let w = causal::regress(&[&m.sample.x1, &m.sample.x2], &y).expect("the library's fit");
                assert!(close(w[0], b[1], 1e-8) && close(w[1], b[2], 1e-8) && close(w[2], b[0], 1e-8), "trial {k}, seed {seed}: {w:?} against {b:?}");
                let dot = |a: &[f64]| a.iter().zip(&f.resid).map(|(x, r)| x * r).sum::<f64>();
                let scale = y.iter().map(|v| v.abs()).sum::<f64>() * 10.0;
                assert!(f.resid.iter().sum::<f64>().abs() < 1e-9 * scale && dot(&m.sample.x1).abs() < 1e-9 * scale && dot(&m.sample.x2).abs() < 1e-9 * scale);
            }
        }
    }

    /// The added-variable chart: the line through what x2 leaves of y against what it
    /// leaves of x1 has the coefficient b1 as its slope, to the last digit written and
    /// to ten significant digits beyond, and what it leaves is what the plane leaves.
    #[test]
    fn the_added_variable_chart_has_the_coefficient_as_its_slope() {
        for k in 0..TRIALS.len() {
            for seed in [322, 5, 77] {
                let m = varied(k, |s| s.seed = seed);
                let (y, f) = (m.y(), m.fit());
                let b = f.b.expect("a plane");
                for j in 0..2 {
                    let (x, o) = if j == 0 { (&m.sample.x1, &m.sample.x2) } else { (&m.sample.x2, &m.sample.x1) };
                    let (ex, ey) = (left_of(x, o), left_of(&y, o));
                    let slope = slope_through_origin(&ex, &ey).expect("a slope");
                    assert!(close(slope, b[1 + j], 1e-10) && says(slope, 3) == says(b[1 + j], 3), "trial {k}, seed {seed}, input {j}: {slope} against {}", b[1 + j]);
                    let worst = (0..f.n).map(|i| (ey[i] - slope * ex[i] - f.resid[i]).abs()).fold(0.0, f64::max);
                    assert!(worst < 1e-9, "trial {k}, seed {seed}: residuals differ by {worst}");
                }
            }
        }
    }

    /// The slope on one input alone is its coefficient plus the other coefficient times
    /// the slope of the other input on this one: exactly, in every sample.
    #[test]
    fn a_slope_alone_is_the_coefficient_and_what_the_other_input_brings() {
        for k in 0..TRIALS.len() {
            for (seed, rho) in [(322, 0.8), (6, 0.0), (9, -0.95), (40, 0.3)] {
                let f = varied(k, |s| (s.seed, s.rho) = (seed, rho)).fit();
                let b = f.b.expect("a plane");
                for j in 0..2 {
                    let (alone, carried) = (f.alone(j).expect("a slope").0, f.other_on(j).expect("a slope"));
                    let said = b[1 + j] + b[2 - j] * carried;
                    assert!(close(alone, said, 1e-12), "trial {k}, seed {seed}, input {j}: {alone} against {said}");
                }
            }
        }
    }

    /// The standard errors are σ̂ √((XᵀX)⁻¹) (worked out here by solving the normal
    /// equations afresh), and larger than with unrelated inputs of the same spreads by
    /// exactly 1/√(1 − r²); and over many samples the truth lies within t standard
    /// errors of the coefficient (t for 95 in 100) in 95 samples in 100.
    #[test]
    fn standard_errors_are_what_they_say() {
        for (k, seed) in [(0, 322), (1, 322), (2, 322), (2, 17), (5, 3)] {
            let m = varied(k, |s| s.seed = seed);
            let f = m.fit();
            let n = f.n;
            // XᵀX with a column of ones, and its inverse's diagonal, column by column
            let cols = [vec![1.0; n], m.sample.x1.clone(), m.sample.x2.clone()];
            let xtx: Vec<f64> = (0..9).map(|e| cols[e / 3].iter().zip(&cols[e % 3]).map(|(a, b)| a * b).sum()).collect();
            let s = f.sigma_hat().unwrap();
            let se = f.se().unwrap();
            for j in 1..3 {
                let mut unit = vec![0.0; 3];
                unit[j] = 1.0;
                let inv = crate::math::linalg::solve(&mut xtx.clone(), &mut unit, 3).unwrap();
                assert!(close(se[j - 1], s * inv[j].sqrt(), 1e-9), "trial {k}: {} against {}", se[j - 1], s * inv[j].sqrt());
            }
            let r = f.r_inputs();
            assert!(close(se[0] * (1.0 - r * r).sqrt(), s / f.s11.sqrt(), 1e-12) && close(se[1] * (1.0 - r * r).sqrt(), s / f.s22.sqrt(), 1e-12));
        }
        for k in [0, 1, 2] {
            let t = st::Dist::StudentT { df: (TRIALS[k].setup.n - 3) as f64 }.quantile(0.975);
            let (mut within, runs) = (0, 2000);
            for seed in 0..runs {
                let m = varied(k, |s| s.seed = 5000 + seed);
                let (b, se) = (m.fit().b.unwrap(), m.fit().se().unwrap());
                within += ((b[1] - m.setup.b1).abs() <= t * se[0]) as usize;
            }
            let share = within as f64 / runs as f64;
            assert!((share - 0.95).abs() < 0.012, "trial {k}: the truth within t standard errors in {share} of the samples");
        }
    }

    /// R² never falls when an input is added (here: x2 added to x1, and x1 to x2, in
    /// samples of every kind); when the input is noise, adjusted R² falls in about two
    /// samples of three: the caption's 66 % of 2000.
    #[test]
    fn r_squared_never_falls_when_an_input_is_added() {
        for k in 0..TRIALS.len() {
            for seed in 0..60 {
                let f = varied(k, |s| s.seed = 700 + seed).fit();
                let both = f.r2().unwrap();
                assert!(both >= f.alone_r2(0).unwrap() - 1e-12 && both >= f.alone_r2(1).unwrap() - 1e-12, "trial {k}, seed {seed}");
            }
        }
        let (mut falls, runs) = (0, 2000);
        for seed in 1000..1000 + runs as u64 {
            let f = varied(3, |s| s.seed = seed).fit();
            let (both, alone) = (f.adj_r2().unwrap(), adjusted(f.alone_r2(0).unwrap(), f.n, 1).unwrap());
            falls += (both < alone) as usize;
        }
        assert_eq!(says(100.0 * falls as f64 / runs as f64, 0), "66", "{falls} of {runs}");
    }

    /// Unrelated inputs: the slopes alone and the coefficients agree, as the caption
    /// says, on its sample.
    #[test]
    fn unrelated_inputs_give_the_same_slopes() {
        let f = trial(0).fit();
        let b = f.b.unwrap();
        assert_eq!(says(f.r_inputs(), 3), "−0.001");
        assert_eq!([says(f.alone(0).unwrap().0, 3), says(b[1], 3), says(f.alone(1).unwrap().0, 3), says(b[2], 3)], ["0.906", "0.905", "−0.696", "−0.695"]);
    }

    /// Inputs that move together: alone x1 seems to raise y, held fixed it lowers it;
    /// the caption's numbers on its sample, and the same signs in nearly every sample.
    #[test]
    fn inputs_that_move_together_turn_a_slope_round() {
        let f = trial(1).fit();
        let b = f.b.unwrap();
        assert_eq!([says(f.r_inputs(), 2), says(f.alone(0).unwrap().0, 2), says(b[1], 2), says(b[2], 2), says(f.other_on(0).unwrap(), 2)], ["0.83", "0.60", "−0.47", "1.34", "0.80"]);
        let (mut rising, mut falling, runs) = (0, 0, 1000);
        for seed in 0..runs {
            let f = varied(1, |s| s.seed = 9000 + seed).fit();
            rising += (f.alone(0).unwrap().0 > 0.0) as usize;
            falling += (f.b.unwrap()[1] < 0.0) as usize;
        }
        assert!(rising == runs as usize && falling > 980, "alone rising in {rising}, held fixed falling in {falling} of {runs}");
    }

    /// Nearly the same input twice: over the new samples each coefficient swings, the
    /// sum and the plane over the middle of the points hardly move, and the spread is
    /// what 1/√(1 − r²) says.
    #[test]
    fn nearly_the_same_input_twice_swings_the_coefficients() {
        let m = trial(2);
        let f = m.fit();
        assert_eq!(m.cloud.coef.len(), NEW_SAMPLES);
        let b1: Vec<f64> = m.cloud.coef.iter().map(|c| c[0]).collect();
        let sum: Vec<f64> = m.cloud.coef.iter().map(|c| c[0] + c[1]).collect();
        let (lo, hi) = extent(&b1);
        let (sum_lo, sum_hi) = extent(&sum);
        let ([s1, _, _], flipped) = m.cloud.spreads(&m.setup).unwrap();
        let middle: Vec<f64> = m.cloud.planes.iter().map(|b| b[0] + b[1] * f.mean[0] + b[2] * f.mean[1]).collect();
        assert_eq!([says(lo, 1), says(hi, 1), says(sum_lo, 1), says(sum_hi, 1)], ["−1.2", "3.3", "1.6", "2.5"]);
        assert_eq!([says(100.0 * flipped, 0), says(st::sd(&middle), 2), says(s1, 2), says(f.r_inputs(), 3)], ["10", "0.14", "0.79", "0.985"]);
        // the same inputs but for x2 unrelated to x1 (x1 is the same draws)
        let mut apart = trial(2);
        apart.setup.rho = 0.0;
        apart.refresh();
        apart.cloud.on = true;
        apart.draw_new(NEW_SAMPLES);
        let ratio = s1 / apart.cloud.spreads(&apart.setup).unwrap().0[0];
        let r = f.r_inputs();
        assert_eq!([says(ratio, 1), says(1.0 / (1.0 - r * r).sqrt(), 1)], ["5.9", "5.8"]);
    }

    /// A useless input: the caption's R² and adjusted R² (and the share of samples in
    /// which the adjusted one falls is held by `r_squared_never_falls_when_an_input_is_added`).
    #[test]
    fn a_useless_input_raises_r_squared_and_lowers_the_adjusted() {
        let f = trial(3).fit();
        let alone = f.alone_r2(0).unwrap();
        assert_eq!([says(alone, 3), says(adjusted(alone, f.n, 1).unwrap(), 3), says(f.r2().unwrap(), 3), says(f.adj_r2().unwrap(), 3)], ["0.268", "0.242", "0.293", "0.240"]);
    }

    /// The mean residual over the fifth of the points with the lowest fitted values,
    /// the middle three fifths, and the highest fifth.
    fn arc(f: &Fit) -> [f64; 3] {
        let mut order: Vec<usize> = (0..f.n).collect();
        order.sort_by(|a, b| f.fitted[*a].total_cmp(&f.fitted[*b]));
        let part = |from: f64, to: f64| st::mean(&order[(from * f.n as f64) as usize..(to * f.n as f64) as usize].iter().map(|&i| f.resid[i]).collect::<Vec<f64>>());
        [part(0.0, 0.2), part(0.2, 0.8), part(0.8, 1.0)]
    }

    /// A bend: the residuals against the fitted values make an arc, on the caption's
    /// sample and in nearly every other; with no bend they do not.
    #[test]
    fn a_bend_shows_in_the_residuals() {
        let f = trial(4).fit();
        let [low, middle, high] = arc(&f);
        assert_eq!([says(f.r2().unwrap(), 2), format!("+{}", says(low, 2)), says(middle, 2), format!("+{}", says(high, 2))], ["0.44", "+0.89", "−0.50", "+0.62"]);
        let (mut arcs, mut flat_arcs, runs) = (0, 0, 300);
        for seed in 0..runs {
            let [a, b, c] = arc(&varied(4, |s| s.seed = 400 + seed).fit());
            arcs += (a > b && c > b) as usize;
            let [a, b, c] = arc(&varied(4, |s| (s.seed, s.bend) = (400 + seed, 0.0)).fit());
            flat_arcs += (a > b + 0.3 && c > b + 0.3) as usize;
        }
        assert!(arcs == runs as usize && flat_arcs < runs as usize / 20, "an arc in {arcs} of {runs} samples with the bend; a deep one in {flat_arcs} without");
    }

    /// One point with leverage: lifting a point by d lifts the plane under it by h d,
    /// exactly; the caption's numbers for the far point and the middle one.
    #[test]
    fn a_point_far_out_pulls_the_plane() {
        let m = trial(5);
        let (far, mid) = (m.sample.marked[0], m.sample.marked[1]);
        let (x1, x2) = (&m.sample.x1, &m.sample.x2);
        assert_eq!((far, x1[far], x2[far]), (0, FAR[0], FAR[1]));
        let before = fit(x1, x2, &m.sample.y);
        let after = m.fit();
        let h = after.leverage(x1[far], x2[far]).unwrap();
        let risen = after.fitted[far] - before.fitted[far];
        assert!(close(risen, 4.0 * h, 1e-12), "{risen} against {}", 4.0 * h);
        let (b0, b1) = (before.b.unwrap(), after.b.unwrap());
        assert_eq!([says(h, 2), says(risen, 2), says(b0[2], 2), says(b1[2], 2)], ["0.40", "1.62", "0.39", "0.04"]);
        let mut y = m.sample.y.clone();
        y[mid] += 4.0;
        let middle = fit(x1, x2, &y);
        let h_mid = middle.leverage(x1[mid], x2[mid]).unwrap();
        let risen_mid = middle.fitted[mid] - before.fitted[mid];
        let moved = (middle.b.unwrap()[1] - b0[1]).abs().max((middle.b.unwrap()[2] - b0[2]).abs());
        assert!(close(risen_mid, 4.0 * h_mid, 1e-12) && says(h_mid, 2) == "0.04" && says(risen_mid, 2) == "0.14" && moved < 0.03, "{h_mid} {risen_mid} {moved}");
        // and every point: the plane under it rises by its leverage times the lift
        for i in 0..x1.len() {
            let mut y = m.sample.y.clone();
            y[i] -= 2.5;
            let g = fit(x1, x2, &y);
            assert!(close(g.fitted[i] - before.fitted[i], -2.5 * before.leverage(x1[i], x2[i]).unwrap(), 1e-10));
        }
    }

    /// The degenerate cases: inputs that are one input twice, three points, no noise,
    /// an output that does not vary. Nothing is a number that is not finite, and what
    /// cannot be worked out is none.
    #[test]
    fn degenerate_samples_say_what_has_happened() {
        for rho in [1.0, -1.0] {
            let mut m = varied(1, |s| s.rho = rho);
            m.look.snap(&m.goal());
            let f = m.fit();
            assert!(f.b.is_none() && f.se().is_none() && f.leverage(0.0, 0.0).is_none());
            let (j, c, _, settled) = f.tied().expect("only a combination is settled");
            assert!(j == 0 && (c - rho).abs() < 1e-12 && settled.is_finite());
            assert!(f.fitted.iter().chain(&f.resid).all(|v| v.is_finite()) && f.r2().unwrap().is_finite() && f.adj_r2().unwrap().is_finite());
            // the fitted values are those of the line through y against x1 alone
            assert!(close(f.r2().unwrap(), f.alone_r2(0).unwrap(), 1e-9));
            let d = m.drawn();
            assert!(d.fit.b.is_none() && d.fit.fitted.iter().all(|v| v.is_finite()));
            // the marked points are still two different points
            assert!(m.sample.marked[0] != m.sample.marked[1]);
        }
        // three points: a plane through all three, nothing left over to measure the noise by
        let f = varied(1, |s| s.n = 3).fit();
        assert!(f.b.is_some() && f.rss < 1e-20 && f.sigma_hat().is_none() && f.se().is_none() && f.adj_r2().is_none());
        assert!((f.r2().unwrap() - 1.0).abs() < 1e-12);
        // no noise: the plane is the truth, and its standard errors are 0
        let f = varied(1, |s| s.sigma = 0.0).fit();
        let b = f.b.unwrap();
        assert!((b[1] + 0.5).abs() < 1e-12 && (b[2] - 1.5).abs() < 1e-12 && f.se().unwrap().iter().all(|s| *s < 1e-12));
        assert_eq!(says(f.se().unwrap()[0], 3), "0.000");
        // nothing to explain
        let f = varied(1, |s| (s.b1, s.b2, s.sigma) = (0.0, 0.0, 0.0)).fit();
        assert!(f.r2().is_none() && f.adj_r2().is_none() && f.b.unwrap().iter().all(|v| *v == 0.0));
        // settings beyond the controls' ends are brought back to them
        let m = varied(1, |s| (s.rho, s.sigma, s.n, s.b1) = (f64::NAN, -1.0, 0, 7.0));
        assert!(m.setup.rho == 0.0 && m.setup.sigma == 0.0 && m.setup.n == 3 && m.setup.b1 == 2.0);
    }

    /// New samples come at a rate, stop at their number, and stop altogether where the
    /// inputs cannot be told apart.
    #[test]
    fn new_samples_come_and_stop() {
        let mut m = MultiReg::create();
        m.set_up(2);
        assert!(m.cloud.running());
        assert_eq!(m.draw_new(10), 10);
        assert_eq!(m.draw_new(NEW_SAMPLES), NEW_SAMPLES - 10);
        assert!(!m.cloud.running() && m.cloud.planes.len() == NEW_SAMPLES);
        let mut m = varied(2, |s| s.rho = 1.0);
        m.cloud.on = true;
        assert_eq!(m.draw_new(5), 0);
        assert!(!m.cloud.running());
    }

    /// The charts keep to the canvas and clear of one another, whatever its shape, with
    /// room round each for its title, numbers and names; the picture has a part of its
    /// own, and the box drawn in it fits it.
    #[test]
    fn the_charts_fit_every_canvas() {
        for (x, y) in [(1080.0f32, 1000.0f32), (1600.0, 1000.0), (580.0, 700.0), (1100.0, 700.0), (390.0, 800.0), (360.0, 640.0), (844.0, 390.0), (760.0, 500.0), (300.0, 300.0), (2400.0, 1300.0), (700.0, 1100.0)] {
            for page in 0..PAGES.len() {
                let inset = if x < 900.0 { 46.0 } else { 0.0 };
                let lay = Layout::of(Vec2::new(x, y), inset, page);
                let boxes: Vec<Rect> = lay.charts.iter().flatten().map(|(min, max)| (*min - Vec2::new(LEFT, ABOVE), *max + Vec2::new(RIGHT, BELOW))).collect();
                // (every canvas but the smallest has charts)
                assert!(!boxes.is_empty() || x * y < 200_000.0, "{x}x{y}: no charts");
                for (i, (min, max)) in boxes.iter().enumerate() {
                    assert!(min.x >= 0.0 && min.y >= inset && max.x <= x + 0.01 && max.y <= y + 0.01, "{x}x{y} page {page}: chart {i} leaves the canvas");
                    let data = (max.x - min.x - LEFT - RIGHT, max.y - min.y - ABOVE - BELOW);
                    assert!(data.0 >= 60.0 && data.1 >= 40.0, "{x}x{y} page {page}: chart {i} is too small: {data:?}");
                    for (k, (a, b)) in boxes.iter().enumerate().skip(i + 1) {
                        assert!(max.x <= a.x + 0.01 || b.x <= min.x + 0.01 || max.y <= a.y + 0.01 || b.y <= min.y + 0.01, "{x}x{y} page {page}: charts {i} and {k} overlap");
                    }
                    // and clear of the picture's part and the tabs
                    let (p0, p1) = lay.pic;
                    assert!(p1.x <= min.x + 0.01 || p1.y <= min.y + 0.01, "{x}x{y}: chart {i} is over the picture");
                    if let Some(t) = lay.tabs {
                        assert!(t.y + TAB_H - 4.0 <= min.y + 0.01 && t.y >= p1.y.min(p0.y + 1e4) - 1e4, "{x}x{y}: the tabs are over chart {i}");
                    }
                }
                let ((p0, p1), (centre, radius)) = (lay.pic, lay.stage());
                assert!(radius > 40.0 && centre.x - radius >= p0.x - 0.01 && centre.x + radius <= p1.x + 0.01 && centre.y + radius <= p1.y + 0.01, "{x}x{y}: the box leaves the picture");
                assert!(distance(radius, y).is_finite() && distance(radius, y) > 0.0);
            }
        }
        // too small for charts: the picture alone, and nothing that is not a number
        for (x, y) in [(240.0f32, 180.0f32), (1.0, 1.0), (0.0, 0.0)] {
            let lay = Layout::of(Vec2::new(x, y), 46.0, 0);
            assert!(lay.charts.iter().all(Option::is_none) && lay.tabs.is_none());
            let (centre, radius) = lay.stage();
            assert!(centre.is_finite() && radius.is_finite() && distance(radius, y).is_finite());
        }
    }

    /// A plane steeper than the box is cut at its top and bottom, and the cut keeps
    /// the plane's own heights where it is inside; round heights are never below what
    /// they round.
    #[test]
    fn planes_are_cut_to_the_box() {
        let b = [0.3, 2.0, -1.5];
        let poly = plane_over_floor(b, 4.0);
        assert!(poly.len() >= 3);
        for p in &poly {
            assert!(p[2].abs() <= 4.0 * 1.15 + 1e-9 && (p[2] - (b[0] + b[1] * p[0] + b[2] * p[1])).abs() < 1e-9 && p[0].abs() <= REACH + 1e-9);
        }
        assert_eq!(plane_over_floor([100.0, 0.0, 0.0], 4.0).len(), 0);
        // a straight line cut to the band of heights: the part inside, on the line
        let (a, b) = ([-3.0, 0.0, -9.0], [3.0, 1.0, 9.0]);
        let (p, q) = clip(a, b, 4.5).unwrap();
        assert!((p[2] + 4.5).abs() < 1e-12 && (q[2] - 4.5).abs() < 1e-12 && (p[0] + 1.5).abs() < 1e-12 && (q[1] - 0.75).abs() < 1e-12);
        assert_eq!(clip([0.0, 0.0, 1.0], [1.0, 0.0, 2.0], 4.5), Some(([0.0, 0.0, 1.0], [1.0, 0.0, 2.0])));
        assert!(clip([0.0, 0.0, 5.0], [1.0, 0.0, 9.0], 4.5).is_none() && clip([0.0, 0.0, 5.0], [1.0, 0.0, 5.0], 4.5).is_none());
        assert!(clip([0.0, 0.0, f64::NAN], [1.0, 0.0, 1.0], 4.5).is_none_or(|(p, q)| p[2].is_finite() && q[2].is_finite()));
        for v in [0.0, 0.3, 0.99, 1.0, 1.01, 2.6, 7.2, 13.0, 999.0, f64::NAN] {
            let r = round_up(v);
            assert!(r.is_finite() && r >= v.max(0.5) * 0.999_999 && r <= v.max(0.5) * 1.34 + 1e-9 || v.is_nan(), "{v} rounds to {r}");
        }
    }

    /// The glide arrives exactly, and keeps what a list had in common when it grows.
    #[test]
    fn the_glide_arrives() {
        let mut e = Ease::default();
        assert!(!e.follow(&[1.0, 2.0], 0.016, SLOW));
        let mut frames = 0;
        while e.follow(&[3.0, 2.0], 1.0 / 60.0, SLOW) {
            frames += 1;
            assert!(frames < 400);
        }
        assert!(frames > 20 && e.shown == vec![3.0, 2.0], "{frames}");
        e.follow(&[3.0, 2.0, 9.0], 0.016, QUICK);
        assert_eq!(e.shown, vec![3.0, 2.0, 9.0]);
    }

    /// Seeds whose samples show each experiment's lesson plainly (run with --ignored
    /// --nocapture).
    #[test]
    #[ignore]
    fn seeds() {
        for seed in 1..400u64 {
            let z = draws(seed);
            let at = |k: usize| {
                let mut s = TRIALS[k].setup;
                s.seed = seed;
                let sample = Sample::new(&s, &z);
                fit(&sample.x1, &sample.x2, &sample.y)
            };
            let (f0, f1, f2, f3) = (at(0), at(1), at(2), at(3));
            let b = |f: &Fit| f.b.unwrap();
            let c0 = f0.r_inputs().abs() < 0.04 && (b(&f0)[1] - 1.0).abs() < 0.12 && (b(&f0)[2] + 0.6).abs() < 0.12;
            let c1 = (b(&f1)[1] + 0.5).abs() < 0.12 && f1.alone(0).unwrap().0 > 0.5;
            let c2 = (b(&f2)[1] - 1.0).abs() < 0.5 && (b(&f2)[2] - 1.0).abs() < 0.5 && (f2.r_inputs() - 0.98).abs() < 0.006;
            let (r_alone, both, alone) = (f3.alone_r2(0).unwrap(), f3.adj_r2().unwrap(), adjusted(f3.alone_r2(0).unwrap(), f3.n, 1).unwrap());
            let c3 = (0.22..0.42).contains(&r_alone) && both < alone && f3.r2().unwrap() - r_alone > 0.002;
            let c5 = {
                let mut s = TRIALS[5].setup;
                s.seed = seed;
                let sample = Sample::new(&s, &z);
                let plain = fit(&sample.x1, &sample.x2, &sample.y);
                (b(&plain)[1] - 1.0).abs() < 0.15 && (b(&plain)[2] - 0.5).abs() < 0.15
            };
            let score = [c0, c1, c2, c3, c5].iter().filter(|c| **c).count();
            if score >= 4 {
                println!("seed {seed}: {score} {c0} {c1} {c2} {c3} {c5}  r0 {:.3} b1 {:.2} r2 {:.2}->{:.2}", f0.r_inputs(), b(&f1)[1], r_alone, f3.r2().unwrap());
            }
        }
    }

    /// What each experiment shows (run with --ignored --nocapture).
    #[test]
    #[ignore]
    fn explore() {
        for k in 0..TRIALS.len() {
            let m = trial(k);
            let f = m.fit();
            let plain = fit(&m.sample.x1, &m.sample.x2, &m.sample.y);
            println!("== {} ==", TRIALS[k].name);
            println!("  r inputs {:.3}  b {:?}  se {:?}", f.r_inputs(), f.b, f.se());
            println!("  alone {:?} {:?}  other_on {:?} {:?}", f.alone(0), f.alone(1), f.other_on(0), f.other_on(1));
            println!("  r2 {:?} adj {:?}  alone r2 {:?} {:?}  adj alone {:?}", f.r2(), f.adj_r2(), f.alone_r2(0), f.alone_r2(1), f.alone_r2(0).and_then(|r| adjusted(r, f.n, 1)));
            println!("  sigma hat {:?}", f.sigma_hat());
            let lev: Vec<_> = m.sample.marked.iter().map(|&i| f.leverage(m.sample.x1[i], m.sample.x2[i])).collect();
            println!("  marked {:?} leverage {:?}  plain b {:?}", m.sample.marked, lev, plain.b);
            if let Some(sp) = m.cloud.spreads(&m.setup) {
                println!("  cloud {} spreads {:?}", m.cloud.coef.len(), sp);
                let b1: Vec<f64> = m.cloud.coef.iter().map(|c| c[0]).collect();
                let sum: Vec<f64> = m.cloud.coef.iter().map(|c| c[0] + c[1]).collect();
                let range = |v: &[f64]| (v.iter().cloned().fold(f64::MAX, f64::min), v.iter().cloned().fold(f64::MIN, f64::max));
                println!("  b1 range {:?}  sum range {:?}", range(&b1), range(&sum));
            }
            // the same design with unrelated inputs: how much the spread of b1 grows
            if k == 2 {
                let mut apart = MultiReg::create();
                apart.set_up(k);
                apart.setup.rho = 0.0;
                apart.refresh();
                apart.cloud.on = true;
                apart.draw_new(NEW_SAMPLES);
                let (a, b) = (m.cloud.spreads(&m.setup).unwrap().0, apart.cloud.spreads(&apart.setup).unwrap().0);
                let fa = apart.fit();
                println!("  spread ratio b1 {:.3}, b2 {:.3}; 1/sqrt(1-r^2) = {:.3}; se ratio {:?}", a[0] / b[0], a[1] / b[1], 1.0 / (1.0 - f.r_inputs().powi(2)).sqrt(), f.se().zip(fa.se()).map(|(x, y)| (x[0] / y[0], x[1] / y[1])));
                // the fitted values over the new samples, at each point: their spread
                println!("  fitted at the centre: sd {:.3}", {
                    let v: Vec<f64> = m.cloud.planes.iter().map(|b| b[0] + b[1] * f.mean[0] + b[2] * f.mean[1]).collect();
                    st::sd(&v)
                });
                let worst = (0..f.n)
                    .map(|i| {
                        let v: Vec<f64> = m.cloud.planes.iter().map(|b| b[0] + b[1] * m.sample.x1[i] + b[2] * m.sample.x2[i]).collect();
                        st::sd(&v)
                    })
                    .fold(0.0, f64::max);
                println!("  fitted values at the points: largest sd {worst:.3}");
            }
            // a useless input: over many new samples, how often adjusted R² falls
            if k == 3 {
                let (mut falls, mut runs) = (0, 0);
                for seed in 1000..3000 {
                    let mut s = TRIALS[k].setup;
                    s.seed = seed;
                    let z = draws(seed);
                    let sample = Sample::new(&s, &z);
                    let g = fit(&sample.x1, &sample.x2, &sample.y);
                    let (both, alone) = (g.adj_r2().unwrap(), adjusted(g.alone_r2(0).unwrap(), g.n, 1).unwrap());
                    assert!(g.r2().unwrap() >= g.alone_r2(0).unwrap() - 1e-12);
                    falls += (both < alone) as usize;
                    runs += 1;
                }
                println!("  adjusted R2 falls in {falls} of {runs}");
            }
            if k == 4 {
                let mut order: Vec<usize> = (0..f.n).collect();
                order.sort_by(|a, b| f.fitted[*a].total_cmp(&f.fitted[*b]));
                let part = |from: f64, to: f64| {
                    let (a, b) = ((from * f.n as f64) as usize, (to * f.n as f64) as usize);
                    st::mean(&order[a..b].iter().map(|&i| f.resid[i]).collect::<Vec<f64>>())
                };
                println!("  residual means: lowest fifth {:.3}, middle {:.3}, top fifth {:.3}", part(0.0, 0.2), part(0.2, 0.8), part(0.8, 1.0));
                println!("  r2 {:?}", f.r2());
            }
            if k == 5 {
                let (far, mid) = (m.sample.marked[0], m.sample.marked[1]);
                let before = &plain;
                println!("  far {far} at ({:.2}, {:.2}) h {:?}; plane under it before {:.3} after {:.3}", m.sample.x1[far], m.sample.x2[far], f.leverage(m.sample.x1[far], m.sample.x2[far]), before.fitted[far], f.fitted[far]);
                println!("  b before {:?} after {:?}", before.b, f.b);
                let mut y = m.sample.y.clone();
                y[mid] += 4.0;
                let g = fit(&m.sample.x1, &m.sample.x2, &y);
                println!("  middle {mid} at ({:.2}, {:.2}) h {:?}; plane under it before {:.3} after {:.3}; b {:?}", m.sample.x1[mid], m.sample.x2[mid], g.leverage(m.sample.x1[mid], m.sample.x2[mid]), before.fitted[mid], g.fitted[mid], g.b);
            }
        }
    }
}


//! Cause or effect? Telling the direction of an arrow from a scatter, by fitting each
//! variable from the other and asking whether what is left over is independent of the
//! input.

use super::*;
use crate::clock::Pace;
use crate::math::causal::{bandwidth, correlation, mean, variance, Hsic, Noise};

pub const DEF: SceneDef = SceneDef {
    id: "direction",
    title: "Cause or effect?",
    group: GROUP,
    blurb: "Fit each variable from the other: in the causal direction, what is left over is independent of the input.",
    make: |_| Box::new(Direction::create()),
};

// ----- colours ------------------------------------------------------------------------------

/// The samples.
const DOT: Color = pal::TEXT;
/// The curve fitted to the points, and the same curve laid flat under them.
const FIT: Color = pal::YELLOW;
/// What is left over, slice by slice.
const SLICE: Color = pal::BLUE;
/// The dependence of shuffled pairings, and of the real one.
const SHUFFLED: Color = pal::DIM;
const PAIRED: Color = pal::WHITE;

// ----- making the data -----------------------------------------------------------------------

/// Which way the data are made.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Truth {
    XCausesY,
    YCausesX,
    /// A third variable Z, never shown, causes both; neither causes the other.
    Hidden,
}

const TRUTHS: [Truth; 3] = [Truth::XCausesY, Truth::YCausesX, Truth::Hidden];

/// The shape of the effect: f in effect = f(cause) + noise.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Curve {
    Line,
    Bend,
    S,
    Parabola,
    /// Through the five points the visitor drags.
    Drawn,
}

const CURVES: [(Curve, &str); 5] = [(Curve::Line, "straight"), (Curve::Bend, "gentle bend"), (Curve::S, "smooth step"), (Curve::Parabola, "parabola"), (Curve::Drawn, "drawn")];

/// Where the drawn curve's points are, along the cause (whose spread is 1 whatever
/// its shape: these are inside the cloud for every shape).
const KNOTS: [f64; 5] = [-1.5, -0.75, 0.0, 0.75, 1.5];
/// The drawn curve as it starts.
const DRAWN: [f64; 5] = [-1.4, 0.5, -0.4, 0.4, 1.4];
/// How far the drawn curve's points can be dragged, up or down.
const REACH: f64 = 3.5;

impl Curve {
    fn at(self, c: f64, heights: &[f64; 5]) -> f64 {
        match self {
            Curve::Line => c,
            // (a smooth hinge: level to the left, rising to the right)
            Curve::Bend => 0.9 * (1.6 * c).exp().ln_1p() - 0.9,
            Curve::S => 1.6 * (1.4 * c).tanh(),
            Curve::Parabola => c * c - 1.0,
            Curve::Drawn => through(heights, c),
        }
    }
}

/// A smooth curve through the drawn points (a cubic between each two, with the slope
/// at each point that of the line through its neighbours), and straight on beyond
/// the first and the last.
fn through(h: &[f64; 5], c: f64) -> f64 {
    let n = KNOTS.len();
    let slope = |k: usize| {
        let (a, b) = (k.saturating_sub(1), (k + 1).min(n - 1));
        (h[b] - h[a]) / (KNOTS[b] - KNOTS[a])
    };
    if !c.is_finite() {
        return 0.0;
    }
    if c <= KNOTS[0] {
        return h[0] + slope(0) * (c - KNOTS[0]);
    }
    if c >= KNOTS[n - 1] {
        return h[n - 1] + slope(n - 1) * (c - KNOTS[n - 1]);
    }
    let k = KNOTS.iter().rposition(|&x| x <= c).unwrap_or(0).min(n - 2);
    let w = KNOTS[k + 1] - KNOTS[k];
    let t = (c - KNOTS[k]) / w;
    let (t2, t3) = (t * t, t * t * t);
    (2.0 * t3 - 3.0 * t2 + 1.0) * h[k] + (t3 - 2.0 * t2 + t) * w * slope(k) + (-2.0 * t3 + 3.0 * t2) * h[k + 1] + (t3 - t2) * w * slope(k + 1)
}

/// Φ, the share of a bell (normal) below v.
fn bell_below(v: f64) -> f64 {
    // (the complementary error function to a few parts in ten million: Numerical Recipes' erfcc)
    let z = v.abs() / std::f64::consts::SQRT_2;
    let t = 1.0 / (1.0 + 0.5 * z);
    let tail = t
        * (-z * z - 1.265_512_23
            + t * (1.000_023_68
                + t * (0.374_091_96 + t * (0.096_784_18 + t * (-0.186_288_06 + t * (0.278_868_07 + t * (-1.135_203_98 + t * (1.488_515_87 + t * (-0.822_152_23 + t * 0.170_872_77)))))))))
            .exp();
    if v >= 0.0 {
        1.0 - 0.5 * tail
    } else {
        0.5 * tail
    }
}

/// The value below which a share `u` of a shape of noise lies. A sample drawn as the
/// quantile of a uniform number has that shape; and the same uniform number gives the
/// point of the same rank in every shape, so a change of shape moves each point a
/// little, instead of dealing them all out afresh.
fn quantile(noise: Noise, u: f64) -> f64 {
    let u = u.max(1e-12).min(1.0 - 1e-12);
    let invert = |below: &dyn Fn(f64) -> f64| {
        let (mut lo, mut hi) = (-12.0f64, 12.0f64);
        for _ in 0..52 {
            let mid = 0.5 * (lo + hi);
            if below(mid) < u {
                lo = mid;
            } else {
                hi = mid;
            }
        }
        0.5 * (lo + hi)
    };
    match noise {
        Noise::Normal => invert(&bell_below),
        Noise::Uniform => (u - 0.5) * 12f64.sqrt(),
        Noise::Laplace => {
            let b = 0.5f64.sqrt();
            if u < 0.5 {
                b * (2.0 * u).ln()
            } else {
                -b * (2.0 * (1.0 - u)).ln()
            }
        }
        Noise::TwoHumps => {
            let s = (1.0f64 - 0.81).sqrt();
            invert(&|v| 0.5 * (bell_below((v - 0.9) / s) + bell_below((v + 0.9) / s)))
        }
    }
}

/// A uniform number in (0, 1), to double precision.
fn uniform(rng: &mut Rng) -> f64 {
    ((rng.next_u64() >> 11) as f64 + 0.5) / (1u64 << 53) as f64
}

/// Everything the data depend on.
#[derive(Clone, Copy, Debug, PartialEq)]
struct Setup {
    truth: Truth,
    curve: Curve,
    noise: Noise,
    /// How much noise is added to the effect (the cause's own spread is 1).
    amount: f64,
    n: usize,
    /// The drawn curve's heights at `KNOTS`.
    heights: [f64; 5],
}

/// How closely X follows the hidden Z, when there is one: X = Z plus this share of the
/// noise added to Y.
const HIDDEN_SHARE: f64 = 0.35;

/// The random part of a sample: for each point, the cause's own value, the noise
/// added to the effect, and the noise in X's reading of a hidden Z (each with the
/// shape asked for, spread 1).
struct Draws {
    key: (u64, usize, Noise),
    parts: [Vec<f64>; 3],
}

impl Draws {
    fn new(seed: u64, n: usize, noise: Noise) -> Draws {
        let mut rng = Rng::new(seed);
        let mut parts = [Vec::with_capacity(n), Vec::with_capacity(n), Vec::with_capacity(n)];
        for _ in 0..n {
            for part in parts.iter_mut() {
                part.push(quantile(noise, uniform(&mut rng)));
            }
        }
        Draws { key: (seed, n, noise), parts }
    }

    /// The sample (X, Y) these draws make with `setup`.
    fn sample(&self, s: &Setup) -> (Vec<f64>, Vec<f64>) {
        let [cause, noise, other] = &self.parts;
        let n = cause.len();
        let (mut x, mut y) = (Vec::with_capacity(n), Vec::with_capacity(n));
        for i in 0..n {
            let effect = s.curve.at(cause[i], &s.heights) + s.amount * noise[i];
            let (a, b) = match s.truth {
                Truth::XCausesY => (cause[i], effect),
                Truth::YCausesX => (effect, cause[i]),
                Truth::Hidden => (cause[i] + HIDDEN_SHARE * s.amount * other[i], effect),
            };
            x.push(a);
            y.push(b);
        }
        (x, y)
    }
}

/// The sample of `setup` with this seed: (X, Y).
fn sample(setup: &Setup, seed: u64) -> (Vec<f64>, Vec<f64>) {
    Draws::new(seed, setup.n, setup.noise).sample(setup)
}

// ----- fitting ---------------------------------------------------------------------------------

/// A smooth curve through the scatter of `output` against `input`, with no shape
/// assumed: at each place, the straight line that best fits the points near it (each
/// weighted by a bell of width `h` round the place), taken where it is. (The plain
/// weighted average of the points near, `causal::smooth_at`, is pulled towards the
/// middle at the ends of a sloping cloud, and that pull would itself leave residuals
/// that depend on the input: with a parabola and 300 points, it made the causal
/// direction look dependent in 29 samples of 100.)
fn local_line(input: &[f64], output: &[f64], at: f64, h: f64) -> f64 {
    let (mut s0, mut s1, mut s2, mut t0, mut t1) = (0.0, 0.0, 0.0, 0.0, 0.0);
    for (&a, &b) in input.iter().zip(output) {
        let d = a - at;
        let w = (-0.5 * (d / h) * (d / h)).exp();
        s0 += w;
        s1 += w * d;
        s2 += w * d * d;
        t0 += w * b;
        t1 += w * d * b;
    }
    let det = s0 * s2 - s1 * s1;
    if det > 1e-9 * s0 * s0 * h * h && det.is_finite() {
        (s2 * t0 - s1 * t1) / det
    } else if s0 > 1e-300 {
        t0 / s0
    } else {
        mean(output)
    }
}

/// The inputs in order, and the width of the smoother's bell for them: the usual
/// width (`causal::bandwidth`) where the points are dense, and wide enough to reach a
/// share of them where they are sparse (at the ends of a bell-shaped cloud, a few
/// points alone would swing the curve about).
struct Width {
    sorted: Vec<f64>,
    least: f64,
    reach: usize,
}

impl Width {
    fn of(input: &[f64]) -> Width {
        let mut sorted = input.to_vec();
        sorted.sort_by(f64::total_cmp);
        Width { least: bandwidth(input), reach: (input.len() / 12).max(6), sorted }
    }

    /// The width at `at`: at least the distance to the `reach`-th nearest input.
    fn at(&self, at: f64) -> f64 {
        let v = &self.sorted;
        let n = v.len();
        if n == 0 {
            return self.least;
        }
        let k = self.reach.min(n);
        // walk out from where `at` would go, taking the nearer side each time
        let (mut lo, mut hi) = (v.partition_point(|&x| x < at), v.partition_point(|&x| x < at));
        let mut far = 0.0f64;
        for _ in 0..k {
            let left = lo.checked_sub(1).map(|i| at - v[i]);
            let right = v.get(hi).map(|x| x - at);
            match (left, right) {
                (Some(l), Some(r)) if l <= r => {
                    far = l;
                    lo -= 1;
                }
                (_, Some(r)) => {
                    far = r;
                    hi += 1;
                }
                (Some(l), None) => {
                    far = l;
                    lo -= 1;
                }
                (None, None) => break,
            }
        }
        // (a bell this wide gives the furthest of them about a third of the weight)
        self.least.max(far * 0.67)
    }
}

/// What a fit leaves over: each output less the curve at its input.
fn leftover(input: &[f64], output: &[f64]) -> Vec<f64> {
    let w = Width::of(input);
    input.iter().zip(output).map(|(&a, &b)| b - local_line(input, output, a, w.at(a))).collect()
}

/// Do the numbers vary at all? (Not if they are all the same, or not numbers.)
fn varies(v: &[f64]) -> bool {
    variance(v) > 0.0
}

/// Is nothing (to rounding) left over: the output an exact function of the input, or
/// with no spread to begin with?
fn exact(output: &[f64], residuals: &[f64]) -> bool {
    let spread = variance(output).sqrt();
    spread.is_nan() || spread <= 0.0 || variance(residuals).sqrt() <= 1e-7 * spread
}

/// How many points the fitted curve is drawn with.
const CURVE: usize = 64;

/// One direction's fit: `output` from `input`.
struct Fit {
    residuals: Vec<f64>,
    /// The fitted curve across the input's range, (input, value).
    curve: Vec<(f64, f64)>,
    /// Nothing left over (see [`exact`]).
    exact: bool,
    /// The correlation of what is left over with the input.
    corr: f64,
}

fn fit(input: &[f64], output: &[f64]) -> Fit {
    let residuals = leftover(input, output);
    let (lo, hi) = extent(input);
    let w = Width::of(input);
    let curve = (0..CURVE)
        .map(|k| {
            let at = lo + (hi - lo) * k as f64 / (CURVE - 1) as f64;
            (at, local_line(input, output, at, w.at(at)))
        })
        .collect();
    Fit { exact: exact(output, &residuals), corr: correlation(input, &residuals), residuals, curve }
}

/// The smallest and largest of some numbers ((0, 0) for none).
fn extent(v: &[f64]) -> (f64, f64) {
    let lo = v.iter().copied().fold(f64::INFINITY, f64::min);
    let hi = v.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    if lo.is_finite() && hi.is_finite() {
        (lo, hi)
    } else {
        (0.0, 0.0)
    }
}

// ----- the test ---------------------------------------------------------------------------------

/// Below this share of shuffles at least as dependent, the residuals are called dependent.
const LEVEL: f64 = 0.05;
/// How many shuffled pairings each dependence is judged against.
const SHUFFLES: usize = 99;
/// With this many shuffles reaching the real pairing, p is at least `LEVEL`.
const ENOUGH: usize = 4;

/// p, from how many of `shuffles` shuffled pairings were at least as dependent as the
/// real one (the real pairing counts as one of the pairings).
fn p_of(reached: usize, shuffles: usize) -> f64 {
    (reached + 1) as f64 / (shuffles + 1) as f64
}

/// The verdict on one sample.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Verdict {
    XCausesY,
    YCausesX,
    /// What is left over is independent of the input both ways: both fit.
    Both,
    /// Dependent both ways: neither fits.
    Neither,
}

impl Verdict {
    fn of(forwards_independent: bool, backwards_independent: bool) -> Verdict {
        match (forwards_independent, backwards_independent) {
            (true, false) => Verdict::XCausesY,
            (false, true) => Verdict::YCausesX,
            (true, true) => Verdict::Both,
            (false, false) => Verdict::Neither,
        }
    }

    /// Its place in the counts over many samples: X → Y, cannot tell, Y → X.
    fn slot(self) -> usize {
        match self {
            Verdict::XCausesY => 0,
            Verdict::Both | Verdict::Neither => 1,
            Verdict::YCausesX => 2,
        }
    }
}

/// One dependence being judged: the real pairing's, and shuffled ones as they come.
struct Side {
    hsic: Option<Hsic>,
    value: f64,
    shuffles: Vec<f64>,
    rng: Rng,
    /// Nothing is left over to test (an exact fit): independent of anything.
    exact: bool,
}

impl Side {
    fn new(seed: u64) -> Side {
        Side { hsic: None, value: 0.0, shuffles: Vec::with_capacity(SHUFFLES), rng: Rng::new(seed), exact: false }
    }

    fn done(&self) -> bool {
        self.exact || self.shuffles.len() >= SHUFFLES
    }

    fn reached(&self) -> usize {
        self.shuffles.iter().filter(|s| **s >= self.value).count()
    }

    fn p(&self) -> f64 {
        if self.exact {
            1.0
        } else {
            p_of(self.reached(), self.shuffles.len())
        }
    }

    fn independent(&self) -> bool {
        self.p() >= LEVEL
    }

    /// One piece of the work on `input` against `other`: the table, or one shuffle.
    fn step(&mut self, input: &[f64], other: &[f64]) {
        match &self.hsic {
            None => {
                let h = Hsic::new(input, other);
                self.value = h.value();
                self.hsic = Some(h);
            }
            Some(h) => {
                let s = h.shuffled(&mut self.rng);
                self.shuffles.push(s);
            }
        }
    }
}

/// The sample shown, and its tests: what is left over with X as the input (0), with Y
/// as the input (1), and, when both look independent, X against Y itself (2).
struct Data {
    key: (Setup, u64),
    x: Vec<f64>,
    y: Vec<f64>,
    fits: [Fit; 2],
    sides: [Side; 3],
}

impl Data {
    fn new(setup: Setup, seed: u64, draws: &Draws) -> Data {
        let (x, y) = draws.sample(&setup);
        let fits = [fit(&x, &y), fit(&y, &x)];
        let mut sides = [0u64, 1, 2].map(|k| Side::new(seed.wrapping_mul(0x9E37_79B9).wrapping_add(31 + k)));
        for k in 0..2 {
            sides[k].exact = fits[k].exact;
        }
        Data { key: (setup, seed), x, y, fits, sides }
    }

    /// The pairs side `k` judges: input, and what it is paired with.
    fn pairs(&self, k: usize) -> (&[f64], &[f64]) {
        match k {
            0 => (&self.x, &self.fits[0].residuals),
            1 => (&self.y, &self.fits[1].residuals),
            _ => (&self.x, &self.y),
        }
    }

    /// Is there work left?
    fn pending(&self) -> bool {
        !self.sides[0].done() || !self.sides[1].done() || (self.both_fit() && !self.sides[2].done())
    }

    fn both_fit(&self) -> bool {
        self.sides[0].done() && self.sides[1].done() && self.sides[0].independent() && self.sides[1].independent()
    }

    /// One piece of the work: the side with the least done goes next.
    fn step(&mut self) {
        let next = (0..2).filter(|&k| !self.sides[k].done()).min_by_key(|&k| self.sides[k].shuffles.len() + if self.sides[k].hsic.is_some() { 1 } else { 0 });
        let k = match next {
            Some(k) => k,
            None if self.both_fit() && !self.sides[2].done() => 2,
            None => return,
        };
        let (input, other): (&[f64], &[f64]) = match k {
            0 => (&self.x, &self.fits[0].residuals),
            1 => (&self.y, &self.fits[1].residuals),
            _ => (&self.x, &self.y),
        };
        self.sides[k].step(input, other);
    }

    /// The verdict, once both sides are judged.
    fn verdict(&self) -> Option<Verdict> {
        (self.sides[0].done() && self.sides[1].done()).then(|| Verdict::of(self.sides[0].independent(), self.sides[1].independent()))
    }

    /// Do X and Y look independent of each other (asked only when both fits do)?
    fn unrelated(&self) -> Option<bool> {
        (self.both_fit() && self.sides[2].done()).then(|| self.sides[2].independent())
    }
}

// ----- many samples -------------------------------------------------------------------------------

/// How many new samples the verdicts are counted over.
const RUNS: usize = 100;
/// Shuffles in one piece of the work on a sample.
const CHUNK: usize = 6;

/// One direction of the test on a sample of the count, done as briefly as the
/// verdict allows.
struct Quick {
    residuals: Vec<f64>,
    exact: bool,
    hsic: Option<Hsic>,
    value: f64,
    done: usize,
    reached: usize,
    sum: f64,
}

impl Quick {
    fn new(input: &[f64], output: &[f64]) -> Quick {
        let residuals = leftover(input, output);
        Quick { exact: exact(output, &residuals), residuals, hsic: None, value: 0.0, done: 0, reached: 0, sum: 0.0 }
    }

    /// Independent as soon as enough shuffles reach the real pairing (the answer all of
    /// them would give).
    fn decided(&self) -> bool {
        self.exact || self.reached >= ENOUGH || self.done >= SHUFFLES
    }

    fn independent(&self) -> bool {
        self.exact || self.reached >= ENOUGH
    }

    fn complete(&self) -> bool {
        self.exact || self.done >= SHUFFLES
    }

    fn p(&self) -> f64 {
        if self.exact {
            1.0
        } else {
            p_of(self.reached, self.done)
        }
    }

    /// The real pairing's dependence over the shuffles' average (to break a tie in p).
    fn ratio(&self) -> f64 {
        if self.exact || self.done == 0 || self.sum <= 0.0 {
            0.0
        } else {
            self.value / (self.sum / self.done as f64)
        }
    }

    fn shuffle(&mut self, rng: &mut Rng) {
        if let Some(h) = &self.hsic {
            let s = h.shuffled(rng);
            self.sum += s;
            self.reached += (s >= self.value) as usize;
            self.done += 1;
        }
    }
}

/// A sample of the count, part way through.
struct Run {
    x: Vec<f64>,
    y: Vec<f64>,
    sides: Vec<Quick>,
    rng: Rng,
}

/// What the method says on a sample: its verdict, and the side it would pick if it
/// had to pick one (0: X → Y, 1: Y → X).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Outcome {
    verdict: Verdict,
    forced: usize,
}

impl Run {
    fn new(setup: &Setup, seed: u64) -> Run {
        let (x, y) = sample(setup, seed);
        Run { x, y, sides: Vec::with_capacity(2), rng: Rng::new(seed ^ 0x5DEE_CE66) }
    }

    /// One piece of the work; the outcome when it is the last.
    fn step(&mut self) -> Option<Outcome> {
        // the fits, one at a time; then the tables
        match self.sides.len() {
            0 => {
                self.sides.push(Quick::new(&self.x, &self.y));
                return None;
            }
            1 => {
                self.sides.push(Quick::new(&self.y, &self.x));
                return None;
            }
            _ => {}
        }
        for k in 0..2 {
            let side = &mut self.sides[k];
            if !side.exact && side.hsic.is_none() {
                let input = if k == 0 { &self.x } else { &self.y };
                let h = Hsic::new(input, &side.residuals);
                side.value = h.value();
                side.hsic = Some(h);
                return None;
            }
        }
        // shuffles until both are decided; and if the verdict is that both fit or
        // neither does, on to all of them, to pick a side if one must be picked
        let tied = self.sides.iter().all(Quick::decided) && self.sides[0].independent() == self.sides[1].independent();
        let wanting = |q: &Quick| if tied { !q.complete() } else { !q.decided() };
        let Some(k) = (0..2).filter(|&k| wanting(&self.sides[k])).min_by_key(|&k| self.sides[k].done) else {
            return Some(self.outcome());
        };
        for _ in 0..CHUNK {
            if !wanting(&self.sides[k]) {
                break;
            }
            self.sides[k].shuffle(&mut self.rng);
        }
        None
    }

    fn outcome(&self) -> Outcome {
        let (a, b) = (&self.sides[0], &self.sides[1]);
        let verdict = Verdict::of(a.independent(), b.independent());
        let forced = if a.p() != b.p() {
            if a.p() > b.p() {
                0
            } else {
                1
            }
        } else if a.ratio() <= b.ratio() {
            0
        } else {
            1
        };
        Outcome { verdict, forced }
    }
}

/// The method's verdicts over `RUNS` new samples like the one shown.
struct Tally {
    key: Option<(Setup, u64)>,
    /// X → Y, cannot tell, Y → X.
    verdicts: [usize; 3],
    /// Forced to pick: X → Y, Y → X.
    forced: [usize; 2],
    done: usize,
    run: Option<Run>,
}

impl Tally {
    fn new() -> Tally {
        Tally { key: None, verdicts: [0; 3], forced: [0; 2], done: 0, run: None }
    }

    /// Start counting afresh for these settings.
    fn restart(&mut self, setup: Setup, seed: u64) {
        *self = Tally::new();
        self.key = Some((setup, seed));
    }

    fn pending(&self) -> bool {
        self.key.is_some() && self.done < RUNS
    }

    /// The seed of new sample `k` of the count (none of them the one shown).
    fn seed_of(seed: u64, k: usize) -> u64 {
        seed.wrapping_mul(1_000_003).wrapping_add(7_919 * (k as u64 + 1))
    }

    fn step(&mut self) {
        let Some((setup, seed)) = self.key else { return };
        if self.done >= RUNS {
            return;
        }
        let k = self.done;
        let run = self.run.get_or_insert_with(|| Run::new(&setup, Tally::seed_of(seed, k)));
        if let Some(out) = run.step() {
            self.verdicts[out.verdict.slot()] += 1;
            self.forced[out.forced] += 1;
            self.done += 1;
            self.run = None;
        }
    }

    /// The whole count at once (the tests).
    #[cfg(test)]
    fn finish(&mut self) {
        while self.pending() {
            self.step();
        }
    }
}

// ----- gliding ----------------------------------------------------------------------------

/// How long a glide takes: the time in which each of its two eases covers about two
/// thirds of what is left (seconds).
const FOLLOW: f32 = 0.075;
/// Closer than this (in the units of what glides), it is there.
const THERE: f32 = 2e-4;

/// Numbers drawn on their way to what they are: two eases one after the other (so
/// that a move sets off gently), and exactly there in the end.
#[derive(Default)]
struct Glide {
    between: Vec<f32>,
    shown: Vec<f32>,
}

impl Glide {
    /// Bring the numbers `dt` seconds nearer to `to`. With `keep`, a list that has
    /// grown or shrunk keeps what it had in common (the first points of a sample are
    /// the same points whatever its size); otherwise it starts where it is going.
    /// Says whether anything is still on its way.
    fn follow(&mut self, to: &[f32], dt: f32, keep: bool) -> bool {
        if self.shown.len() != to.len() {
            if keep && !self.shown.is_empty() {
                let n = self.shown.len().min(to.len());
                self.shown.truncate(n);
                self.between.truncate(n);
                self.shown.extend_from_slice(&to[n..]);
                self.between.extend_from_slice(&to[n..]);
            } else {
                self.shown = to.to_vec();
                self.between = to.to_vec();
            }
        }
        if self.shown == to {
            return false;
        }
        let k = 1.0 - (-dt.max(0.0) / FOLLOW).exp();
        let mut apart = 0.0f32;
        for ((b, s), t) in self.between.iter_mut().zip(self.shown.iter_mut()).zip(to) {
            *b += (t - *b) * k;
            *s += (*b - *s) * k;
            apart = apart.max((t - *s).abs()).max((t - *b).abs());
        }
        // (exactly there in the end, not nearly: the picture then stops changing)
        if apart.is_nan() || apart <= THERE {
            self.shown = to.to_vec();
            self.between = to.to_vec();
            return false;
        }
        true
    }

    fn snap(&mut self, to: &[f32]) {
        self.shown = to.to_vec();
        self.between = to.to_vec();
    }
}

/// Everything that is drawn gliding: the samples (X, Y pairs), what each fit leaves
/// (forwards, backwards pairs), the ranges of the axes, the fitted curves, the true
/// curve, the slices, the dependences on the strips and the counts.
#[derive(Default)]
struct Look {
    points: Glide,
    residuals: Glide,
    ranges: Glide,
    curves: [Glide; 2],
    truth: Glide,
    /// How much of the true curve shows in each column (0 to 1).
    truth_alpha: Glide,
    slices: [Glide; 2],
    strips: Glide,
    bars: Glide,
    moving: bool,
}

/// The numbers the look glides to, worked out from the data (see `Direction::goals`).
struct Goals {
    points: Vec<f32>,
    residuals: Vec<f32>,
    ranges: Vec<f32>,
    curves: [Vec<f32>; 2],
    truth: Vec<f32>,
    truth_alpha: Vec<f32>,
    slices: [Vec<f32>; 2],
    strips: Vec<f32>,
    bars: Vec<f32>,
}

impl Look {
    fn follow(&mut self, g: &Goals, dt: f32, snap: bool) {
        if snap {
            self.points.snap(&g.points);
            self.residuals.snap(&g.residuals);
            self.ranges.snap(&g.ranges);
            for c in 0..2 {
                self.curves[c].snap(&g.curves[c]);
                self.slices[c].snap(&g.slices[c]);
            }
            self.truth.snap(&g.truth);
            self.truth_alpha.snap(&g.truth_alpha);
            self.strips.snap(&g.strips);
            self.bars.snap(&g.bars);
            self.moving = false;
            return;
        }
        let mut moving = self.points.follow(&g.points, dt, true);
        moving |= self.residuals.follow(&g.residuals, dt, true);
        moving |= self.ranges.follow(&g.ranges, dt, false);
        for c in 0..2 {
            moving |= self.curves[c].follow(&g.curves[c], dt, false);
            moving |= self.slices[c].follow(&g.slices[c], dt, false);
        }
        moving |= self.truth.follow(&g.truth, dt, false);
        moving |= self.truth_alpha.follow(&g.truth_alpha, dt, false);
        moving |= self.strips.follow(&g.strips, dt, false);
        moving |= self.bars.follow(&g.bars, dt, false);
        self.moving = moving;
    }
}

// ----- slices ---------------------------------------------------------------------------------------

/// What is left over in one slice of the input: where the slice starts and ends, and
/// the residuals' 10th, 25th, 50th, 75th and 90th percentiles in it.
const PER_SLICE: usize = 7;

/// The k-th of the `q`-quantiles of sorted numbers (by straight lines between them).
fn percentile(sorted: &[f64], q: f64) -> f64 {
    if sorted.is_empty() {
        return 0.0;
    }
    let at = q.max(0.0).min(1.0) * (sorted.len() - 1) as f64;
    let (i, t) = (at.floor() as usize, at.fract());
    let j = (i + 1).min(sorted.len() - 1);
    sorted[i] + (sorted[j] - sorted[i]) * t
}

const MARKS: [f64; 5] = [0.1, 0.25, 0.5, 0.75, 0.9];

/// The input cut into `k` slices with as many points in each, and what is left over
/// in each: `PER_SLICE` numbers a slice, and then the same five percentiles for all of
/// the residuals together.
fn slices(input: &[f64], residuals: &[f64], k: usize) -> Vec<f32> {
    let n = input.len().min(residuals.len());
    let k = k.max(1);
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&a, &b| input[a].total_cmp(&input[b]));
    let mut out = Vec::with_capacity(k * PER_SLICE + MARKS.len());
    for s in 0..k {
        let (from, to) = (s * n / k, ((s + 1) * n / k).max(s * n / k + 1).min(n));
        let part = order.get(from..to).unwrap_or(&[]);
        let (lo, hi) = match (part.first(), part.last()) {
            (Some(&a), Some(&b)) => (input[a], input[b]),
            _ => (0.0, 0.0),
        };
        // (each slice reaches halfway to the next, so that together they cover the cloud)
        let start = if s == 0 { lo } else { 0.5 * (lo + order.get(from.wrapping_sub(1)).map_or(lo, |&i| input[i])) };
        let end = if s + 1 == k { hi } else { 0.5 * (hi + order.get(to).map_or(hi, |&i| input[i])) };
        let mut r: Vec<f64> = part.iter().map(|&i| residuals[i]).collect();
        r.sort_by(f64::total_cmp);
        out.push(start as f32);
        out.push(end as f32);
        out.extend(MARKS.iter().map(|&q| percentile(&r, q) as f32));
    }
    let mut all: Vec<f64> = residuals[..n].to_vec();
    all.sort_by(f64::total_cmp);
    out.extend(MARKS.iter().map(|&q| percentile(&all, q) as f32));
    out
}

/// How many slices: about thirty points in each, and none narrower on the screen than
/// a finger's width on average.
fn slice_count(n: usize, width: f32) -> usize {
    ((n as f32 / 30.0).round() as usize).clamp(2, 7).min((width / 34.0).max(2.0) as usize)
}

// ----- the experiments -------------------------------------------------------------------------------

/// Experiments: (button, what to look for, truth, curve, noise, how much noise, points).
/// Each shows the sample of seed `FIRST_SEED`, and counts the verdicts over `RUNS` new
/// samples.
const TRIALS: [(&str, &str, Truth, Curve, Noise, f64, usize); 6] = [
    (
        "a bent cause",
        r"$X$ causes $Y$ through a smooth step, with bell-shaped noise. On the left, what is left over is a level band. On the right it is narrow in the middle and wide at both ends: its widest slice is 3.5 times its narrowest, against 1.5 on the left. Only on the left is it independent of the input ($p = 0.94$, against $p = 0.01$): $X \to Y$, the verdict in 99 of 100 new samples too.",
        Truth::XCausesY,
        Curve::S,
        Noise::Normal,
        0.4,
        200,
    ),
    (
        "turned round",
        r"The same points with $X$ and $Y$ swapped: now $Y$ causes $X$. The two columns trade pictures, and the verdict turns round with them, here and in 97 of 100 new samples. The method follows the data, not the variable drawn across.",
        Truth::YCausesX,
        Curve::S,
        Noise::Normal,
        0.4,
        200,
    ),
    (
        "a straight line, flat noise",
        r"$Y = X + \text{noise}$, the cause and the noise both flat (uniform). A straight line fits both ways, and what is left over is uncorrelated with the input both ways (correlations 0.01 and 0.00). Yet on the right it is narrow at both ends and wide in the middle: dependent ($p = 0.01$). On the left it is a level band ($p = 0.94$). Right in all of 100 new samples.",
        Truth::XCausesY,
        Curve::Line,
        Noise::Uniform,
        1.0,
        200,
    ),
    (
        "a straight line, bell noise",
        r"The same with the cause and the noise both bell-shaped. Now both directions leave a level band ($p = 0.87$ and $p = 0.61$): $Y$ could be made from $X$, or $X$ from $Y$, each with independent bell-shaped noise, and nothing in the data says which. The verdict is that it cannot tell, in all of 100 new samples. Forced to pick, it is right in 51: a coin toss.",
        Truth::XCausesY,
        Curve::Line,
        Noise::Normal,
        1.0,
        200,
    ),
    (
        "twenty points",
        r"The bent cause with twenty points: too few to show the dependence on the right ($p = 0.99$ both ways), and the verdict is that it cannot tell. Over 100 new samples it cannot tell in 94, is right in 6 and wrong in none; forced to pick a side, it would be wrong in 22.",
        Truth::XCausesY,
        Curve::S,
        Noise::Normal,
        0.4,
        20,
    ),
    (
        "a hidden common cause",
        r"A hidden $Z$ causes both: $X$ is a close reading of $Z$, and $Y$ a smooth step of $Z$ plus noise. Setting $X$ by hand would do nothing to $Y$. Yet the data look just as if $X$ caused $Y$, and the verdict is $X \to Y$ ($p = 0.47$ against $p = 0.01$), here and in 94 of 100 new samples. The method assumes there is no hidden common cause, and cannot see one.",
        Truth::Hidden,
        Curve::S,
        Noise::Normal,
        0.4,
        200,
    ),
];

/// The scene opens as the first experiment sets it up.
const OPENING: usize = 0;
/// The seed of the sample every experiment shows.
const FIRST_SEED: u64 = 4;

// ----- the scene ---------------------------------------------------------------------------------------

pub struct Direction {
    setup: Setup,
    seed: u64,
    show_truth: bool,
    /// Count the verdicts over new samples (asked for once, it counts again after every change).
    counting: bool,
    draws: Draws,
    data: Data,
    tally: Tally,
    pace: Pace,
    tally_pace: Pace,
    look: Look,
    /// The canvas last drawn on (a new one is drawn at once, without gliding).
    canvas: Option<Vec2>,
    /// How far each strip reaches (its logarithm, so that it glides in proportion) and
    /// where its real pairing is: kept while a new sample's dependence is being worked
    /// out, so that nothing moves until there is something to move to.
    strip_goal: [[f32; 2]; 2],
    /// Was each strip drawn in the last frame, and has it been given its reach for this
    /// sample?
    marked: [bool; 2],
    aimed: [bool; 2],
    /// The round values on each column's axes: across, up the scatter, up what is left
    /// over (see [`Ticks`]); and how many grid points and labels the last frame drew.
    ticks: [[Ticks; 3]; 2],
    drawn_last: (usize, usize),
    /// The drawn curve's point being dragged, and the ranges of the axes held still
    /// while it is.
    held: Option<usize>,
    held_ranges: Option<Vec<f32>>,
}

impl Direction {
    /// (Nothing here needs the graphics device, so the tests can make one too.)
    fn create() -> Direction {
        let (_, _, truth, curve, noise, amount, n) = TRIALS[OPENING];
        let (setup, seed) = (Setup { truth, curve, noise, amount, n, heights: DRAWN }, FIRST_SEED);
        let draws = Draws::new(seed, setup.n, setup.noise);
        let data = Data::new(setup, seed, &draws);
        Direction {
            setup,
            seed,
            show_truth: true,
            counting: false,
            draws,
            data,
            tally: Tally::new(),
            pace: Pace::default(),
            tally_pace: Pace::default(),
            look: Look::default(),
            canvas: None,
            strip_goal: [[0.0, 0.0]; 2],
            ticks: [[Ticks::NONE; 3]; 2],
            drawn_last: (0, 0),
            marked: [false; 2],
            aimed: [false; 2],
            held: None,
            held_ranges: None,
        }
    }

    /// Set up one of the experiments in `TRIALS`, and count its verdicts over new samples.
    fn set_up(&mut self, k: usize) {
        let (_, _, truth, curve, noise, amount, n) = TRIALS[k];
        self.setup = Setup { truth, curve, noise, amount, n, heights: DRAWN };
        self.seed = FIRST_SEED;
        self.show_truth = true;
        self.counting = true;
        self.refresh();
    }

    /// Make the data again if what they depend on has changed, and start their tests
    /// (and the count, if it is kept) afresh.
    fn refresh(&mut self) {
        self.setup.n = self.setup.n.clamp(10, 300);
        self.setup.amount = self.setup.amount.max(0.0).min(1.5);
        if self.draws.key != (self.seed, self.setup.n, self.setup.noise) {
            self.draws = Draws::new(self.seed, self.setup.n, self.setup.noise);
        }
        if self.data.key != (self.setup, self.seed) {
            self.data = Data::new(self.setup, self.seed, &self.draws);
            self.pace.reset();
        }
        if self.counting && self.tally.key != Some((self.setup, self.seed)) {
            self.tally.restart(self.setup, self.seed);
            self.tally_pace.reset();
        }
    }

    /// The tests and the count, paced: so many pieces of work a second of the scene's
    /// time (none while the data are being dragged), within the frame's allowance. (When
    /// the scene's time stands still, by the clock: the work answers the visitor's hand.)
    fn work(&mut self, dt: f32, real_dt: f32) {
        let dt = if dt > 0.0 { dt } else { real_dt };
        if self.held.is_some() {
            return;
        }
        if self.data.pending() {
            let mut turn = self.pace.begin(TEST_RATE, dt, real_dt, 0.004);
            while self.data.pending() && turn.next() {
                self.data.step();
            }
            self.pace.end(turn);
        } else if self.tally.pending() {
            let mut turn = self.tally_pace.begin(TALLY_RATE, dt, real_dt, 0.004);
            while self.tally.pending() && turn.next() {
                self.tally.step();
            }
            self.tally_pace.end(turn);
        }
    }

    /// The tests to the end at once (the tests of this file).
    #[cfg(test)]
    fn settle(&mut self) {
        self.refresh();
        while self.data.pending() {
            self.data.step();
        }
    }

    /// Which column the cause is drawn across in, if one of them is the truth.
    fn true_column(&self) -> Option<usize> {
        match self.setup.truth {
            Truth::XCausesY => Some(0),
            Truth::YCausesX => Some(1),
            Truth::Hidden => None,
        }
    }

    /// Are the true curve's five points shown, to be dragged? (With the truth, when
    /// one of the variables causes the other.)
    fn knots_shown(&self) -> bool {
        self.show_truth && self.true_column().is_some()
    }

    /// What the picture glides to, on this layout.
    fn goals(&self, lay: &Layout) -> Goals {
        let d = &self.data;
        let n = d.x.len();
        let mut points = Vec::with_capacity(2 * n);
        let mut residuals = Vec::with_capacity(2 * n);
        for i in 0..n {
            points.push(d.x[i] as f32);
            points.push(d.y[i] as f32);
            residuals.push(d.fits[0].residuals[i] as f32);
            residuals.push(d.fits[1].residuals[i] as f32);
        }
        // the axes reach a little beyond the points; what is left over has 0 in the middle
        let reach = |v: &[f64]| {
            let (lo, hi) = extent(v);
            let pad = ((hi - lo) * 0.06).max(0.05);
            [(lo - pad) as f32, (hi + pad) as f32]
        };
        let deep = |r: &[f64]| {
            let most = r.iter().fold(0.0f64, |m, v| m.max(v.abs()));
            let most = if most > 1e-9 && most.is_finite() { most * 1.12 } else { 1.0 };
            [-most as f32, most as f32]
        };
        let ranges = match &self.held_ranges {
            Some(kept) => kept.clone(),
            None => [reach(&d.x), reach(&d.y), deep(&d.fits[0].residuals), deep(&d.fits[1].residuals)].concat(),
        };
        let curves = [0, 1].map(|c| d.fits[c].curve.iter().flat_map(|&(a, b)| [a as f32, b as f32]).collect::<Vec<f32>>());
        // the true curve, as (X, Y) pairs across the cause's range: drawn in the cause's column
        let cause: &[f64] = if self.setup.truth == Truth::YCausesX { &d.y } else { &d.x };
        let (lo, hi) = extent(cause);
        // (and then its five points, which glide with it)
        let truth = (0..CURVE)
            .map(|k| lo + (hi - lo) * k as f64 / (CURVE - 1) as f64)
            .chain(KNOTS)
            .flat_map(|c| {
                let e = self.setup.curve.at(c, &self.setup.heights);
                match self.setup.truth {
                    Truth::YCausesX => [e as f32, c as f32],
                    _ => [c as f32, e as f32],
                }
            })
            .collect();
        let truth_alpha = [0, 1].map(|c| if self.show_truth && self.true_column() == Some(c) { 1.0 } else { 0.0 }).to_vec();
        let slices = [0, 1].map(|c| {
            let (input, _) = d.pairs(c);
            let k = slice_count(n, lay.plot_width());
            slices(input, &d.fits[c].residuals, k)
        });
        let strips = self.strip_goal.concat();
        let planned = RUNS as f32;
        let bars = [self.tally.verdicts[0], self.tally.verdicts[1], self.tally.verdicts[2], self.tally.forced[0], self.tally.forced[1]].map(|c| c as f32 / planned).to_vec();
        Goals { points, residuals, ranges, curves, truth, truth_alpha, slices, strips, bars }
    }
}

/// How many shuffles a strip waits for before it is drawn: enough to judge how far it
/// should reach.
const FIRST_LOOK: usize = 12;

/// Pieces of work a second: the tests of the sample shown (two tables and 198 shuffles,
/// in about half a second), and the count over new samples.
const TEST_RATE: f32 = 420.0;
const TALLY_RATE: f32 = 1000.0;

// ----- the layout ------------------------------------------------------------------------------------

/// A rectangle on the screen (logical pixels): top-left, bottom-right.
type Rect = (Vec2, Vec2);

/// Where everything goes on a canvas.
#[derive(Clone, Copy, Debug, PartialEq)]
struct Layout {
    view: Vec2,
    narrow: bool,
    /// Text sizes: headings, words, small print.
    big: f32,
    text: f32,
    small: f32,
    /// Each column's heading (top-left corner) and its three pictures.
    head: [Vec2; 2],
    scatter: [Rect; 2],
    resid: [Rect; 2],
    strip: [Rect; 2],
    /// The verdict, the truth beside it, the verdict's reasons, and the counts over
    /// new samples.
    verdict: Rect,
    truth: Rect,
    words: Rect,
    counts: Rect,
    /// The radius of the variables' discs in the small graphs.
    node: f32,
    /// Too small a canvas for the pictures: the verdict alone.
    compact: bool,
}

impl Layout {
    fn of(view: Vec2, inset: f32) -> Layout {
        let view = view.max(Vec2::splat(1.0));
        let narrow = view.x < 700.0;
        let (big, text, small) = if narrow { (13.0, 11.5, 10.5) } else { (15.0, 12.5, 11.5) };
        let side = if narrow { 8.0 } else { 20.0 };
        let gap = if narrow { 12.0 } else { 46.0 };
        // room at the left of each picture for its numbers
        let numbers = if narrow { 24.0 } else { 34.0 };
        let top = inset.max(0.0) + if narrow { 6.0 } else { 12.0 };
        let node = if narrow { 13.0 } else { 16.0 };
        // the verdict, the truth and the counts (on a phone, one above the other)
        let graphs = text + 2.0 * node + 14.0;
        let band = if narrow { graphs + 3.0 * (small + 4.0) + 112.0 } else { 138.0 };
        let head = big + small + 14.0;
        let line = small + 9.0;
        let width = ((view.x - 2.0 * side - gap) * 0.5).max(10.0);
        let strip_h = if narrow { 38.0 } else { 52.0 };
        // what is left for the two pictures
        let rest = (view.y - top - head - band - strip_h - 5.0 * line - 3.0 * 8.0).max(20.0);
        let (scatter_h, resid_h) = (rest * 0.57, rest * 0.43);
        let compact = resid_h < 50.0 || width < 110.0;
        let mut heads = [Vec2::ZERO; 2];
        let (mut scatter, mut resid, mut strip) = ([(Vec2::ZERO, Vec2::ZERO); 2], [(Vec2::ZERO, Vec2::ZERO); 2], [(Vec2::ZERO, Vec2::ZERO); 2]);
        for c in 0..2 {
            let x0 = side + c as f32 * (width + gap);
            let (l, r) = (x0 + numbers, x0 + width);
            heads[c] = Vec2::new(x0, top);
            let y = top + head;
            scatter[c] = (Vec2::new(l, y), Vec2::new(r, y + scatter_h));
            let y = y + scatter_h + line + line;
            resid[c] = (Vec2::new(l, y), Vec2::new(r, y + resid_h));
            let y = y + resid_h + line + line + 8.0;
            strip[c] = (Vec2::new(l, y), Vec2::new(r, y + strip_h));
        }
        let band_top = strip[0].1.y + line + 14.0;
        let (verdict, truth, words, counts) = if narrow {
            let mid = view.x * 0.5;
            let graphs_end = band_top + graphs;
            let words_end = graphs_end + 4.0 + 3.0 * (small + 4.0);
            (
                (Vec2::new(side, band_top), Vec2::new(mid - 6.0, graphs_end)),
                (Vec2::new(mid + 6.0, band_top), Vec2::new(view.x - side, graphs_end)),
                (Vec2::new(side, graphs_end + 4.0), Vec2::new(view.x - side, words_end)),
                (Vec2::new(side, words_end + 12.0), Vec2::new(view.x - side, view.y - 6.0)),
            )
        } else {
            let w = view.x - 2.0 * side;
            let a = side + w * 0.40;
            let b = a + w * 0.17;
            (
                (Vec2::new(side, band_top), Vec2::new(a - 12.0, band_top + graphs)),
                (Vec2::new(a, band_top), Vec2::new(b - 12.0, view.y - 10.0)),
                (Vec2::new(side, band_top + graphs + 4.0), Vec2::new(a - 12.0, view.y - 10.0)),
                (Vec2::new(b + 12.0, band_top), Vec2::new(view.x - side, view.y - 10.0)),
            )
        };
        Layout { view, narrow, big, text, small, head: heads, scatter, resid, strip, verdict, truth, words, counts, node, compact }
    }

    fn plot_width(&self) -> f32 {
        self.scatter[0].1.x - self.scatter[0].0.x
    }
}

/// From a picture's numbers to the screen: `rect` shows `across` (lo, hi) along and
/// `up` (lo, hi) up.
#[derive(Clone, Copy)]
struct Map {
    rect: Rect,
    across: [f32; 2],
    up: [f32; 2],
}

impl Map {
    fn to(&self, a: f32, b: f32) -> Vec2 {
        let (min, max) = self.rect;
        let t = (a - self.across[0]) / (self.across[1] - self.across[0]).max(1e-9);
        let s = (b - self.up[0]) / (self.up[1] - self.up[0]).max(1e-9);
        Vec2::new(min.x + t * (max.x - min.x), max.y - s * (max.y - min.y))
    }

    /// The number up the picture at screen height `y`.
    fn up_at(&self, y: f32) -> f32 {
        let (min, max) = self.rect;
        self.up[0] + (max.y - y) / (max.y - min.y).max(1e-6) * (self.up[1] - self.up[0])
    }
}

/// Is the point inside the rectangle (with a margin)?
fn within(p: Vec2, (min, max): Rect, margin: f32) -> bool {
    p.x >= min.x - margin && p.x <= max.x + margin && p.y >= min.y - margin && p.y <= max.y + margin
}

/// The first of `options` that fits in `room` logical pixels at `size` (the last if none does).
fn pick(f: &Frame, options: &[String], room: f32, size: f32) -> String {
    options.iter().find(|t| f.text_size(t, size).x <= room).or(options.last()).cloned().unwrap_or_default()
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

/// Text broken into lines no wider than `width` logical pixels, from `at` down (screen
/// space). Returns the height taken.
fn wrap(f: &mut Frame, at: Vec2, width: f32, text: &str, size: f32, color: Color) -> f32 {
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
    for (k, line) in lines.iter().enumerate() {
        f.text_px(at + Vec2::new(0.0, k as f32 * step), line.as_str(), size, color, Align2::LEFT_TOP);
    }
    lines.len() as f32 * step
}

/// A button over the picture: true when it is clicked.
fn button(f: &mut Frame, (min, max): Rect, text: &str, size: f32, pointer: Option<Vec2>) -> bool {
    let over = pointer.is_some_and(|p| within(p, (min, max), 0.0));
    hud_rect(f, min, max, if over { Color::hex(0x33405E) } else { Color::hex(0x222B40) });
    let edge = [min, Vec2::new(max.x, min.y), max, Vec2::new(min.x, max.y), min];
    path_of(f, BUTTONS, &edge, 1.0, pal::LINK.a(if over { 0.9 } else { 0.45 }));
    f.text_px((min + max) * 0.5, text, size, pal::TEXT, Align2::CENTER_CENTER);
    if over {
        f.handles.claim_clicks();
        f.handles.capture();
    }
    over && f.ptr.clicked
}

/// The names of the two variables in a formula, each in the colour of its part in a
/// column: the cause across (`input`), the effect up.
fn names(c: usize) -> (String, String) {
    let (input, output) = if c == 0 { ("X", "Y") } else { ("Y", "X") };
    (tint(CAUSE, input), tint(EFFECT, output))
}

/// The round values along one axis that get a grid line and a number: every multiple
/// of `step` from `first` to `last` times it. They are chosen from where the axis is
/// going, a few beyond each end, and stay the same while it glides there: each line
/// and number keeps its place in its batch, sliding with the axis, unseen while it is
/// off the picture. (Chosen afresh from the axis as drawn, a line would come in at one
/// end as another went out at the other, and every line would change partners.)
#[derive(Clone, Copy, Debug, PartialEq)]
struct Ticks {
    step: f32,
    first: i64,
    last: i64,
}

impl Ticks {
    const NONE: Ticks = Ticks { step: 1.0, first: 0, last: -1 };

    /// For an axis going from `lo` to `hi` over `pixels`: a round step about sixty
    /// pixels long.
    fn of(lo: f32, hi: f32, pixels: f32) -> Ticks {
        let span = hi - lo;
        if !span.is_finite() || span <= 0.0 || pixels.is_nan() || pixels <= 0.0 {
            return Ticks::NONE;
        }
        let step = nice_step((hi - lo) / (pixels / 60.0).max(1.0));
        let first = (lo / step).floor() as i64 - 3;
        let last = ((hi / step).ceil() as i64 + 3).min(first + 60);
        Ticks { step, first, last }
    }

    fn values(&self) -> impl Iterator<Item = f32> + '_ {
        (self.first..=self.last).map(|k| k as f32 * self.step)
    }
}

impl Scene for Direction {
    fn view(&self) -> View {
        View::Flat { center: Vec2::ZERO, half_height: 1.0 }
    }

    fn settings(&self, s: &mut ViewSettings) {
        s.grid = false;
        s.bloom = s.bloom.min(0.2);
    }

    fn animated(&self) -> bool {
        self.look.moving || self.data.pending() || (self.counting && self.tally.pending()) || self.held.is_some()
    }

    fn pointer_hint(&self) -> Option<&'static str> {
        Some("green dots on the true curve: drag up or down to bend it    buttons over the picture: click")
    }

    fn ui(&mut self, ui: &mut egui::Ui, _: &Gfx) {
        ui::note(ui, r"Two variables and a scatter of them, and no experiment to settle it: did $X$ produce $Y$, or $Y$ produce $X$? Their correlation is the same both ways, so the scatter alone cannot say, unless something more is assumed. Assume that the effect is some function of its cause plus noise that has nothing to do with the cause:");
        ui::equation(ui, r"Y = f(X) + N, \quad N \text{ independent of } X");
        ui::note(ui, r"Then fit $Y$ from $X$ with a smooth curve and look at what is left over, the residuals: if $X$ is the cause, they are the noise $N$ itself (but for the fit's own small misses), alike at every $X$. Fit $X$ from $Y$ as well: what is left over is uncorrelated with $Y$, near enough, but in general not independent of it: its spread or shape changes along $Y$. The direction in which what is left over is independent of the input is the causal one.");
        ui::note(ui, r"When the effect bends, or the cause or the noise is not bell-shaped, only one direction fits an effect plus independent noise, but for a few special combinations of curve, cause and noise (Hoyer and others, 2009, for curves; Shimizu and others, 2006, for straight lines). With a straight line and the cause and the noise both bell-shaped, both directions fit exactly as well, and nothing can tell them apart. And the method rests on two things it cannot check: that nothing hidden causes both variables, and that the noise is added to the effect, not multiplied into it.");

        ui::section(ui, "Things to try");
        for (k, trial) in TRIALS.iter().enumerate() {
            if ui::experiment(ui, trial.0, trial.1) {
                self.set_up(k);
            }
        }

        ui::section(ui, "The data");
        let mut truth = TRUTHS.iter().position(|t| *t == self.setup.truth).unwrap_or(0);
        if ui::choice(ui, &[r"$X$ causes $Y$", r"$Y$ causes $X$", r"a hidden $Z$ causes both"], &mut truth) {
            self.setup.truth = TRUTHS[truth];
        }
        ui::note(ui, r"With a hidden $Z$: $X$ is a close reading of $Z$, and $Y$ is $f(Z)$ plus noise. Neither causes the other.");
        let mut curve = CURVES.iter().position(|c| c.0 == self.setup.curve).unwrap_or(0);
        let curve_names: Vec<&str> = CURVES.iter().map(|c| c.1).collect();
        if ui::choice(ui, &curve_names, &mut curve) {
            self.setup.curve = CURVES[curve].0;
        }
        ui::note(ui, "The shape of the effect. Or drag the green dots on the true curve (while the truth is shown) to draw one of your own through them.");
        let mut noise = Noise::ALL.iter().position(|n| *n == self.setup.noise).unwrap_or(0);
        let noise_names: Vec<&str> = Noise::ALL.iter().map(|n| n.name()).collect();
        if ui::choice(ui, &noise_names, &mut noise) {
            self.setup.noise = Noise::ALL[noise];
        }
        ui::note(ui, "The shape of the noise, and of the cause's own scatter.");
        let mut amount = self.setup.amount as f32;
        if ui::slider(ui, r"noise $\sigma$", &mut amount, 0.0..=1.5) {
            self.setup.amount = amount as f64;
        }
        ui::note(ui, r"The effect is $f(\text{cause}) + \sigma N$, with $N$ and the cause each of spread 1. At $\sigma = 0$ there is no noise, and nothing for the test to go on.");
        ui::slider_int_log(ui, "points", &mut self.setup.n, 10..=300);
        ui.horizontal(|ui| {
            if ui.button("new sample").clicked() {
                self.seed = self.seed.wrapping_add(1);
            }
            ui.checkbox(&mut self.show_truth, "show the truth");
        });
        if !self.counting && ui.button(format!("count the verdicts over {RUNS} new samples")).clicked() {
            self.counting = true;
        }
        self.refresh();

        ui::section(ui, "Reading the picture");
        ui::key(
            ui,
            &[
                (DOT, "dots: the samples; in each column the cause it supposes is across, the effect up"),
                (CAUSE, "cyan: the variable a column supposes is the cause"),
                (EFFECT, "orange: the variable it supposes is the effect"),
                (FIT, "yellow: the smooth curve fitted to the points, and under them the same curve laid flat, from which what is left over is measured"),
                (SLICE, "blue boxes: what is left over in each slice of the input, with as many points in each: the middle 80 % (light), the middle half (dark) and the median; dashed, the middle 80 % of all of it"),
                (SHUFFLED, "grey dots: how dependent what is left over looks when its pairing with the input is shuffled"),
                (PAIRED, "white line: how dependent it is as paired"),
                (TRUTH, "green: the truth (the true curve, the true arrow), and verdicts that are right"),
                (NAIVE, "pink: verdicts that are wrong"),
                (THIRD, r"purple: the hidden common cause $Z$"),
            ],
        );
        ui::section(ui, "Words");
        ui::term(ui, "Residuals:", "what a fit leaves over: each output less the fitted curve at its input.");
        ui::term(ui, "Independent:", "knowing one says nothing about the other: in every slice of one, the other has the same spread and shape. More than uncorrelated, which only says there is no straight-line trend.");
        ui::term(ui, "Additive noise model:", r"$\text{effect} = f(\text{cause}) + \text{noise}$, with the noise independent of the cause.");
        ui::term(ui, "HSIC:", "the Hilbert–Schmidt independence criterion, a measure of dependence of any kind, not only a trend: 0 for independent variables, but for sampling noise, and larger the more one says about the other.");
        ui::term(ui, "Shuffled pairing:", "the same values, with the partners dealt out again at random. That destroys any dependence, so it shows what independence looks like for these values.");
        ui::term(ui, r"$p$:", "the share of the pairings, the real one among them, that look at least as dependent as the real one. Below 0.05: called dependent.");

        ui::section(ui, "Numbers");
        let d = &self.data;
        ui::readout(ui, r"correlation of $X$ and $Y$", format!("${}$", num(correlation(&d.x, &d.y), 2)));
        for (c, name) in [(0, r"with $X$ as the input"), (1, r"with $Y$ as the input")] {
            let s = &d.sides[c];
            let p = if s.exact {
                "nothing left over".to_string()
            } else if s.done() {
                format!("$p = {}$", num(s.p(), 2))
            } else {
                "testing".to_string()
            };
            ui::readout(ui, name, p);
            ui::readout(ui, "  residuals' correlation with it", format!("${}$", num(d.fits[c].corr, 2)));
        }
        if self.counting {
            let t = &self.tally;
            ui::readout(ui, r"new samples: $X \to Y$, cannot tell, $Y \to X$", format!("{}, {}, {} of {}", t.verdicts[0], t.verdicts[1], t.verdicts[2], t.done));
        }
        ui::see_also(
            ui,
            &[
                ("lingam", "why it works for straight lines: the shape of the noise"),
                ("discover", "more than two variables"),
                ("confound", "what a hidden common cause does"),
                ("pc", "discovery from independences alone"),
            ],
        );
    }

    fn frame(&mut self, f: &mut Frame) {
        let lay = Layout::of(f.view_size(), f.top_inset);
        let pointer = f.ptr.pos.map(|p| p / f.px);
        // (a canvas squeezed to a sliver is one the controls have opened over, on a phone)
        let tiny = lay.view.x < 160.0 || lay.view.y < 160.0;

        if tiny || lay.compact {
            (self.held, self.held_ranges) = (None, None);
        } else {
            self.handles(f, &lay);
            self.refresh();
        }
        self.work(f.dt, f.real_dt);
        self.aim_strips();
        let goals = self.goals(&lay);
        let snap = self.canvas != Some(lay.view);
        if snap && !tiny {
            self.canvas = Some(lay.view);
        }
        self.look.follow(&goals, f.real_dt, snap);
        // (the curve whose point is held is drawn where the hand puts it)
        if self.held.is_some() {
            self.look.truth.snap(&goals.truth);
        }
        // a strip that has just appeared is drawn where it is, not glided to from where
        // the last sample's was
        for c in 0..2 {
            let now = self.strip_ready(c);
            if now && !self.marked[c] && self.look.strips.shown.len() == 4 {
                for i in 2 * c..2 * c + 2 {
                    (self.look.strips.shown[i], self.look.strips.between[i]) = (goals.strips[i], goals.strips[i]);
                }
            }
            self.marked[c] = now;
        }
        let ticks = [0, 1].map(|c| {
            let r = &goals.ranges;
            let (across, up) = if c == 0 { ((r[0], r[1]), (r[2], r[3])) } else { ((r[2], r[3]), (r[0], r[1])) };
            let (s, d) = (lay.scatter[c], lay.resid[c]);
            [Ticks::of(across.0, across.1, s.1.x - s.0.x), Ticks::of(up.0, up.1, (s.1.y - s.0.y) * 1.3), Ticks::of(r[4 + 2 * c], r[5 + 2 * c], (d.1.y - d.0.y) * 1.3)]
        });
        let new_ticks = ticks != self.ticks;
        self.ticks = ticks;
        if tiny {
            return;
        }
        if lay.compact {
            self.draw_compact(f, &lay);
            return;
        }
        for c in 0..2 {
            self.draw_column(f, &lay, c);
        }
        self.draw_legend(f, &lay);
        self.draw_verdict(f, &lay, pointer);
        self.draw_truth(f, &lay, pointer);
        self.draw_counts(f, &lay, pointer);
        // new round values on an axis: make sure the grid's batch and the labels are not
        // as many as in the last frame, so that nothing takes the new ones for the old
        // ones moved
        let count = |f: &mut Frame| (apart(f, GRID, false).pts.iter().filter(|p| p.width >= 0.0).count(), f.dl.labels.len());
        if new_ticks {
            let (grid, labels) = count(f);
            if grid == self.drawn_last.0 {
                let far = f.hud(Vec2::ZERO);
                apart(f, GRID, false).segment(far, far, 1.0, Color::CLEAR);
            }
            if labels == self.drawn_last.1 {
                f.text_px(Vec2::ZERO, "", 1.0, Color::CLEAR, Align2::LEFT_TOP);
            }
        }
        self.drawn_last = count(f);
    }
}

impl Direction {
    /// Is strip `c` drawn: has the real pairing been measured, and enough shuffles to
    /// say how far the strip should reach?
    fn strip_ready(&self, c: usize) -> bool {
        let s = &self.data.sides[c];
        s.hsic.is_some() && (s.done() || s.shuffles.len() >= FIRST_LOOK)
    }

    /// How far each strip reaches, and where its real pairing is: from the first dozen
    /// shuffles, half as far again as the furthest of them or the real pairing (later
    /// shuffles beyond that wait at the edge); and once all are in, a little beyond the
    /// furthest of all. (Following the furthest shuffle as each comes would move the
    /// strip in jerks.)
    fn aim_strips(&mut self) {
        for c in 0..2 {
            let s = &self.data.sides[c];
            if !self.strip_ready(c) {
                self.aimed[c] = false;
                continue;
            }
            let furthest = s.shuffles.iter().copied().fold(s.value, f64::max).max(1e-12);
            if s.done() {
                self.strip_goal[c] = [(furthest * 1.15).ln() as f32, s.value as f32];
            } else if !self.aimed[c] {
                self.strip_goal[c] = [(furthest * 1.5).ln() as f32, s.value as f32];
                self.aimed[c] = true;
            }
        }
    }

    /// The maps of column `c`'s two pictures, as drawn.
    fn maps(&self, lay: &Layout, c: usize) -> Option<(Map, Map)> {
        let r = &self.look.ranges.shown;
        if r.len() < 8 {
            return None;
        }
        let (x, y) = ([r[0], r[1]], [r[2], r[3]]);
        let (across, up) = if c == 0 { (x, y) } else { (y, x) };
        let deep = if c == 0 { [r[4], r[5]] } else { [r[6], r[7]] };
        Some((Map { rect: lay.scatter[c], across, up }, Map { rect: lay.resid[c], across, up: deep }))
    }

    /// The true curve's five points, dragged up and down in the cause's column. The
    /// first drag of a curve that has a formula makes it a drawn one through its own
    /// five points.
    fn handles(&mut self, f: &mut Frame, lay: &Layout) {
        if !self.knots_shown() {
            (self.held, self.held_ranges) = (None, None);
            return;
        }
        let Some(c) = self.true_column() else { return };
        let Some((map, _)) = self.maps(lay, c) else { return };
        let mut held = None;
        for k in 0..KNOTS.len() {
            // (where the point is drawn: gliding with the curve)
            let t = &self.look.truth.shown;
            let Some(&[a, b]) = t.get(2 * (CURVE + k)..2 * (CURVE + k) + 2).and_then(|v| <&[f32; 2]>::try_from(v).ok()) else { continue };
            let at = if c == 0 { map.to(a, b) } else { map.to(b, a) };
            if !within(at, map.rect, 0.0) {
                continue;
            }
            let Some(world) = f.cam.pick_ground(at * f.px) else { continue };
            let mut p = world.truncate();
            if f.drag2(40 + k as u32, &mut p, TRUTH) {
                held = Some(k);
                if self.setup.curve != Curve::Drawn {
                    let from = self.setup.curve;
                    self.setup.heights = KNOTS.map(|x| from.at(x, &self.setup.heights).max(-REACH).min(REACH));
                    self.setup.curve = Curve::Drawn;
                }
                if let Some(screen) = on_screen(f, p.extend(0.0)) {
                    let y = screen.y.max(map.rect.0.y).min(map.rect.1.y);
                    let h = map.up_at(y) as f64;
                    if h.is_finite() {
                        self.setup.heights[k] = h.max(-REACH).min(REACH);
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
            (Some(_), None) => self.held_ranges = Some(self.look.ranges.shown.clone()),
            (None, Some(_)) => self.held_ranges = None,
            _ => {}
        }
        self.held = held;
    }

    fn draw_column(&self, f: &mut Frame, lay: &Layout, c: usize) {
        let Some((scatter, resid)) = self.maps(lay, c) else { return };
        let (input, output) = names(c);
        let fitted = if c == 0 { r"\hat f" } else { r"\hat g" };
        let width = lay.scatter[c].1.x - lay.head[c].x;
        let true_here = self.show_truth && self.true_column() == Some(c);

        // the heading
        let head = lay.head[c];
        let title = format!("if ${input}$ causes ${output}$");
        f.text_px(head, title.as_str(), lay.big, pal::TEXT, Align2::LEFT_TOP);
        // (in both columns, seen in one: so that the labels keep their order when the truth turns round)
        let w = f.text_size(&title, lay.big).x;
        f.text_px(head + Vec2::new(w + 10.0, 2.0), "(true)", lay.small, TRUTH.a(if true_here { 1.0 } else { 0.0 }), Align2::LEFT_TOP);
        let model = [
            format!(r"fit ${output} = {fitted}({input}) + \text{{noise}}$: the yellow curve"),
            format!(r"fit ${output}$ from ${input}$: the yellow curve"),
            format!(r"fit ${output}$ from ${input}$"),
        ];
        let line = pick(f, &model, width, lay.small);
        f.text_px(head + Vec2::new(0.0, lay.big + 5.0), line, lay.small, pal::DIM, Align2::LEFT_TOP);

        // the scatter
        let (min, max) = lay.scatter[c];
        hud_panel(f, min - Vec2::splat(4.0), max + Vec2::splat(4.0));
        self.axes(f, lay, &scatter, self.ticks[c][0], self.ticks[c][1], true);
        let pts = &self.look.points.shown;
        let n = pts.len() / 2;
        let size = if lay.narrow { 3.0 } else { 3.8 };
        let alpha = if n > 150 { 0.62 } else { 0.8 };
        // (a point outside the axes while they glide waits at the edge: the same points every frame)
        let keep = |p: Vec2, (lo, hi): Rect| p.max(lo).min(hi);
        for i in 0..n {
            let (a, b) = if c == 0 { (pts[2 * i], pts[2 * i + 1]) } else { (pts[2 * i + 1], pts[2 * i]) };
            let p = keep(scatter.to(a, b), lay.scatter[c]);
            f.dot(f.hud_at(1, p), size, DOT.a(alpha));
        }
        // the true curve (in both columns always, seen only in the cause's)
        let t = &self.look.truth.shown;
        let seen = self.look.truth_alpha.shown.get(c).copied().unwrap_or(0.0);
        let true_pts: Vec<Vec2> = (0..(t.len() / 2).min(CURVE)).map(|k| if c == 0 { scatter.to(t[2 * k], t[2 * k + 1]) } else { scatter.to(t[2 * k + 1], t[2 * k]) }).collect();
        dashed_px(f, TRUE_CURVE + c as u8, &true_pts, 2.0, TRUTH.a(0.9 * seen), lay.scatter[c]);
        // the fitted curve
        let cv = &self.look.curves[c].shown;
        let fit_pts: Vec<Vec2> = (0..cv.len() / 2).map(|k| scatter.to(cv[2 * k], cv[2 * k + 1])).collect();
        path_of(f, FITTED, &clip(&fit_pts, lay.scatter[c]), 2.4, FIT);
        // names in the corner
        let mut key_at = min + Vec2::new(8.0, 6.0);
        f.text_px(key_at, format!("${fitted}$, fitted"), lay.small, FIT, Align2::LEFT_TOP);
        key_at.y += lay.small + 5.0;
        let name = if lay.narrow { "true" } else { "true curve: drag its dots" };
        f.text_px(key_at, name, lay.small, TRUTH.a(seen), Align2::LEFT_TOP);
        f.text_px(Vec2::new(max.x, max.y + 3.0), format!("${input}$"), lay.text, CAUSE, Align2::RIGHT_TOP);
        f.text_px(Vec2::new(min.x - 4.0, min.y - 2.0), format!("${output}$"), lay.text, EFFECT, Align2::RIGHT_TOP);

        // what is left over
        let (rmin, rmax) = lay.resid[c];
        let label = [
            format!(r"what is left over, ${output} - {fitted}({input})$, against ${input}$"),
            format!(r"left over: ${output} - {fitted}({input})$"),
            "left over".to_string(),
        ];
        let label = pick(f, &label, width, lay.text);
        f.text_px(Vec2::new(lay.head[c].x, rmin.y - 8.0), label, lay.text, pal::TEXT, Align2::LEFT_BOTTOM);
        hud_panel(f, rmin - Vec2::splat(4.0), rmax + Vec2::splat(4.0));
        self.axes(f, lay, &resid, self.ticks[c][0], self.ticks[c][2], false);
        self.draw_slices(f, &resid, c);
        let r = &self.look.residuals.shown;
        for i in 0..n.min(r.len() / 2) {
            let a = if c == 0 { pts[2 * i] } else { pts[2 * i + 1] };
            let p = keep(resid.to(a, r[2 * i + c]), lay.resid[c]);
            f.dot(f.hud_at(2, p), size, DOT.a(alpha));
        }
        // the fitted curve laid flat
        line_of(f, SLICES, resid.to(resid.across[0], 0.0), resid.to(resid.across[1], 0.0), 1.6, FIT.a(0.85));

        // how dependent it is
        self.draw_strip(f, lay, c);
    }

    /// One line under the pictures of what is left over, naming the marks in them.
    fn draw_legend(&self, f: &mut Frame, lay: &Layout) {
        let y = lay.resid[0].1.y + 8.0;
        let (x0, x1) = (lay.head[0].x, lay.scatter[1].1.x);
        let s = lay.small;
        let long = lay.view.x >= 1000.0;
        let parts: [(&str, u8); 3] = if long {
            [("box: the middle 80 % of what is left over in a slice (darker: the middle half)", 0), ("its median", 1), ("the middle 80 % of all slices together", 2)]
        } else {
            [("each slice's middle 80 %", 0), ("median", 1), ("all slices", 2)]
        };
        let mut x = x0;
        for (text, mark) in parts {
            let swatch = Vec2::new(x, y + 2.0);
            match mark {
                0 => {
                    fill(f, swatch, swatch + Vec2::new(14.0, s - 2.0), Color::hex(0x1B3566).a(0.9));
                    line_of(f, LEGEND, swatch, swatch + Vec2::new(14.0, 0.0), 1.6, SLICE.mix(pal::WHITE, 0.2));
                    line_of(f, LEGEND, swatch + Vec2::new(0.0, s - 2.0), swatch + Vec2::new(14.0, s - 2.0), 1.6, SLICE.mix(pal::WHITE, 0.2));
                }
                1 => line_of(f, LEGEND, swatch + Vec2::new(2.0, s * 0.5 - 1.0), swatch + Vec2::new(12.0, s * 0.5 - 1.0), 2.0, pal::WHITE),
                _ => dashed_px(f, LEGEND, &[swatch + Vec2::new(0.0, s * 0.5 - 1.0), swatch + Vec2::new(16.0, s * 0.5 - 1.0)], 1.3, pal::WHITE.a(0.55), (Vec2::ZERO, lay.view)),
            }
            let w = f.text_size(text, s).x;
            if x + 20.0 + w > x1 {
                break;
            }
            f.text_px(Vec2::new(x + 20.0, y), text, s, pal::DIM, Align2::LEFT_TOP);
            x += 20.0 + w + 22.0;
        }
    }

    /// A picture's grid, and its numbers: the input's under it (only under the scatter:
    /// the picture under that has the same), and the output's at its left. Every round
    /// value of `across` and `up` is drawn, those off the picture unseen at its edge.
    fn axes(&self, f: &mut Frame, lay: &Layout, map: &Map, across: Ticks, up: Ticks, under: bool) {
        let (min, max) = map.rect;
        for v in across.values() {
            let x = map.to(v, map.up[0]).x;
            let seen = x >= min.x - 0.5 && x <= max.x + 0.5;
            let x = x.max(min.x).min(max.x);
            line_of(f, GRID, Vec2::new(x, min.y), Vec2::new(x, max.y), 1.0, pal::FAINT.a(if seen { 0.35 } else { 0.0 }));
            if under {
                let text = if seen && x > min.x + 8.0 && x < max.x - 14.0 { fmt_tick(v, across.step) } else { String::new() };
                f.text_px(Vec2::new(x, max.y + 5.0), text, lay.small, pal::DIM, Align2::CENTER_TOP);
            }
        }
        for v in up.values() {
            let y = map.to(map.across[0], v).y;
            let seen = y >= min.y - 0.5 && y <= max.y + 0.5;
            let y = y.max(min.y).min(max.y);
            line_of(f, GRID, Vec2::new(min.x, y), Vec2::new(max.x, y), 1.0, pal::FAINT.a(if seen && v != 0.0 { 0.35 } else { 0.0 }));
            let text = if seen && y > min.y + 14.0 && y < max.y - 6.0 { fmt_tick(v, up.step) } else { String::new() };
            f.text_px(Vec2::new(min.x - 6.0, y), text, lay.small, pal::DIM, Align2::RIGHT_CENTER);
        }
    }

    /// Column `c`'s slices of what is left over, as a band cut into slices: in each, a
    /// box from the 10th to the 90th percentile of its residuals (darker over the
    /// middle half, with a mark at the median); dashed across, the 10th and 90th
    /// percentiles of all of them together.
    fn draw_slices(&self, f: &mut Frame, map: &Map, c: usize) {
        let s = &self.look.slices[c].shown;
        if s.len() < MARKS.len() {
            return;
        }
        let k = (s.len() - MARKS.len()) / PER_SLICE;
        let (min, max) = map.rect;
        let keep = |p: Vec2| Vec2::new(p.x.max(min.x).min(max.x), p.y.max(min.y).min(max.y));
        let all = &s[k * PER_SLICE..];
        for q in [all[0], all[4]] {
            let y = keep(map.to(0.0, q)).y;
            dashed_px(f, DASHED, &[Vec2::new(min.x, y), Vec2::new(max.x, y)], 1.3, pal::WHITE.a(0.55), map.rect);
        }
        let edge = SLICE.mix(pal::WHITE, 0.2);
        for j in 0..k {
            let v = &s[j * PER_SLICE..(j + 1) * PER_SLICE];
            let (x0, x1) = (keep(map.to(v[0], 0.0)).x + 1.0, keep(map.to(v[1], 0.0)).x - 1.0);
            let x1 = x1.max(x0 + 1.0);
            let y = |q: f32| keep(map.to(v[0], q)).y;
            // (dark colours: a translucent bright one over black looks far brighter than its share)
            fill(f, Vec2::new(x0, y(v[6])), Vec2::new(x1, y(v[2])), Color::hex(0x1B3566).a(0.55));
            fill(f, Vec2::new(x0, y(v[5])), Vec2::new(x1, y(v[3])), Color::hex(0x24457F).a(0.5));
            for q in [v[2], v[6]] {
                line_of(f, SLICES, Vec2::new(x0, y(q)), Vec2::new(x1, y(q)), 2.2, edge);
            }
            let mid = 0.5 * (x0 + x1);
            let tick = ((x1 - x0) * 0.25).clamp(3.0, 10.0);
            line_of(f, SLICES, Vec2::new(mid - tick, y(v[4])), Vec2::new(mid + tick, y(v[4])), 2.0, pal::WHITE.a(0.9));
        }
    }

    /// Column `c`'s strip: the dependence as paired, against shuffled pairings.
    fn draw_strip(&self, f: &mut Frame, lay: &Layout, c: usize) {
        let (min, max) = lay.strip[c];
        let width = max.x - lay.head[c].x;
        let (input, _) = names(c);
        let label = [format!(r"how much it depends on ${input}$: as paired (white), shuffled (grey)"), format!(r"dependence on ${input}$"), "dependence".to_string()];
        let label = pick(f, &label, width, lay.text);
        f.text_px(Vec2::new(lay.head[c].x, min.y - 6.0), label, lay.text, pal::TEXT, Align2::LEFT_BOTTOM);
        hud_panel(f, min - Vec2::splat(4.0), max + Vec2::splat(4.0));
        let side = &self.data.sides[c];
        let st = &self.look.strips.shown;
        let ready = self.strip_ready(c);
        let (top, value) = (st.get(2 * c).map_or(1.0, |v| v.exp()).max(1e-12), st.get(2 * c + 1).copied().unwrap_or(0.0));
        let x_of = |v: f32| min.x + (v / top).max(0.0).min(1.0) * (max.x - min.x);
        line_of(f, STRIPS, Vec2::new(min.x, max.y), Vec2::new(max.x, max.y), 1.0, pal::DIM.a(0.7));
        f.text_px(Vec2::new(min.x - 6.0, max.y), "0", lay.small, pal::DIM, Align2::RIGHT_CENTER);
        if !lay.narrow {
            f.text_px(Vec2::new(max.x, max.y + 5.0), "more dependent →", lay.small, pal::DIM, Align2::RIGHT_TOP);
        }
        // (the dots below a row for the real pairing's name)
        let (row, h) = (lay.small + 4.0, max.y - min.y);
        for (k, s) in side.shuffles.iter().enumerate().filter(|_| ready) {
            let jitter = ((k as f32 + 1.0) * 0.618_034).fract();
            let p = Vec2::new(x_of(*s as f32), min.y + row + 2.0 + jitter * (h - row - 6.0).max(1.0));
            let at = f.hud_at(1, p);
            let reached = *s >= side.value;
            f.dl.points_with(PointKind::Disc, PointStyle { outline_px: 0.0, ..Default::default() }).push(at, 4.0, if reached { pal::TEXT.a(0.85) } else { SHUFFLED.a(0.7) });
        }
        let below = Vec2::new(lay.head[c].x, max.y + 6.0);
        if side.exact {
            f.text_px(below, "nothing left over: the fit is exact", lay.small, pal::DIM, Align2::LEFT_TOP);
            return;
        }
        if ready {
            let x = x_of(value);
            line_of(f, STRIPS, Vec2::new(x, min.y - 2.0), Vec2::new(x, max.y + 2.0), 2.4, PAIRED);
            // (on the side with room, gliding when it changes sides)
            let half = f.text_size("as paired", lay.small) * 0.5;
            let right = x + 6.0 + 2.0 * half.x < max.x;
            let offset = Vec2::new(if right { 6.0 + half.x } else { -6.0 - half.x }, row * 0.5);
            f.text_placed("as paired", c as u32, Vec2::new(x, min.y), offset, "as paired", lay.small, PAIRED);
        }
        let done = side.shuffles.len();
        let options = if !side.done() {
            vec![format!("testing: {done} of {SHUFFLES} shuffles"), "testing".to_string()]
        } else {
            let reached = side.reached();
            let p = num(side.p(), 2);
            let verdict = if side.independent() { "independent" } else { "dependent" };
            let how = match reached {
                0 => "no shuffle reaches it".to_string(),
                1 => "1 shuffle reaches it".to_string(),
                r => format!("{r} of {SHUFFLES} shuffles reach it"),
            };
            vec![format!("{verdict}: {how} ($p = {p}$)"), format!("{verdict} ($p = {p}$)"), verdict.to_string()]
        };
        let text = pick(f, &options, width, lay.small);
        let color = if side.done() { pal::TEXT } else { pal::DIM };
        f.text_px(below, text, lay.small, color, Align2::LEFT_TOP);
    }

    /// On a canvas too small for the pictures: the verdict alone, and what would show more.
    fn draw_compact(&self, f: &mut Frame, lay: &Layout) {
        let mid = lay.view * 0.5;
        let r = 12.0;
        let verdict = self.data.verdict();
        let (cx, cy) = match verdict {
            Some(Verdict::XCausesY) => (CAUSE, EFFECT),
            Some(Verdict::YCausesX) => (EFFECT, CAUSE),
            _ => (pal::TEXT, pal::TEXT),
        };
        let at = Vec2::new(mid.x - 60.0, mid.y.max(f.top_inset + 30.0));
        let nodes = [Node { at, name: "$X$", color: cx }, Node { at: at + Vec2::new(64.0, 0.0), name: "$Y$", color: cy }];
        let links = match verdict {
            Some(Verdict::XCausesY) => vec![(0, 1, Link::Arrow, PAIRED, 2.0)],
            Some(Verdict::YCausesX) => vec![(1, 0, Link::Arrow, PAIRED, 2.0)],
            _ => vec![],
        };
        hud_graph(f, &nodes, &links, r);
        let head = match verdict {
            None => "weighing…",
            Some(Verdict::XCausesY) | Some(Verdict::YCausesX) => "verdict",
            Some(_) => "cannot tell",
        };
        f.text_px(at + Vec2::new(64.0 + r + 10.0, 0.0), head, lay.text, pal::TEXT, Align2::LEFT_CENTER);
        f.text_px(at + Vec2::new(-r, r + 12.0), "(a larger window shows the pictures)", lay.small, pal::DIM, Align2::LEFT_TOP);
    }

    fn draw_verdict(&self, f: &mut Frame, lay: &Layout, _pointer: Option<Vec2>) {
        let (min, max) = lay.verdict;
        let (wmin, wmax) = lay.words;
        hud_panel(f, min - Vec2::splat(6.0), Vec2::new(max.x, max.y.max(wmax.y)) + Vec2::splat(4.0));
        hud_panel(f, wmin - Vec2::splat(6.0), wmax + Vec2::splat(4.0));
        f.text_px(min, "verdict", lay.text, pal::DIM, Align2::LEFT_TOP);
        let d = &self.data;
        let r = lay.node;
        let graph_y = min.y + lay.text + 10.0 + r;
        let gx = min.x + r + 4.0;
        let span = if lay.narrow { 70.0 } else { 92.0 };
        let verdict = d.verdict();
        let unrelated = d.unrelated();
        let (cx, cy) = match verdict {
            Some(Verdict::XCausesY) => (CAUSE, EFFECT),
            Some(Verdict::YCausesX) => (EFFECT, CAUSE),
            _ => (pal::TEXT, pal::TEXT),
        };
        let nodes = [Node { at: Vec2::new(gx, graph_y), name: "$X$", color: cx }, Node { at: Vec2::new(gx + span, graph_y), name: "$Y$", color: cy }];
        let flat = !varies(&d.x) || !varies(&d.y);
        let links: Vec<(usize, usize, Link, Color, f32)> = match verdict {
            _ if flat => vec![],
            Some(Verdict::XCausesY) => vec![(0, 1, Link::Arrow, PAIRED, 2.2)],
            Some(Verdict::YCausesX) => vec![(1, 0, Link::Arrow, PAIRED, 2.2)],
            Some(Verdict::Neither) => vec![(0, 1, Link::Line, pal::DIM, 1.6)],
            Some(Verdict::Both) if unrelated == Some(false) => vec![(0, 1, Link::Line, pal::DIM, 1.6)],
            _ => vec![],
        };
        hud_graph(f, &nodes, &links, r);
        let words_x = gx + span + r + 14.0;
        let head = match verdict {
            None => "weighing…".to_string(),
            Some(Verdict::XCausesY) => r"$X \to Y$".to_string(),
            Some(Verdict::YCausesX) => r"$Y \to X$".to_string(),
            Some(_) => "cannot tell".to_string(),
        };
        f.text_px(Vec2::new(words_x, graph_y), head, lay.big + 2.0, if verdict.is_some() { pal::WHITE } else { pal::DIM }, Align2::LEFT_CENTER);
        let (pf, pb) = (num(d.sides[0].p(), 2), num(d.sides[1].p(), 2));
        let single = |v: &[f64]| !varies(v);
        let why = match verdict {
            Some(_) if single(&d.x) || single(&d.y) => {
                let which = match (single(&d.x), single(&d.y)) {
                    (true, true) => r"$X$ and $Y$ each take",
                    (true, false) => r"$X$ takes",
                    _ => r"$Y$ takes",
                };
                format!("{which} a single value: nothing varies with anything, and there is no direction to find.")
            }
            Some(_) if d.fits[0].exact && d.fits[1].exact => {
                r"Nothing is left over either way: each variable is an exact function of the other, and the test has nothing to go on.".to_string()
            }
            None => String::new(),
            Some(Verdict::XCausesY) => format!(r"Only with $X$ as the input is what is left over independent of it ($p = {pf}$, against $p = {pb}$ with $Y$ as the input)."),
            Some(Verdict::YCausesX) => format!(r"Only with $Y$ as the input is what is left over independent of it ($p = {pb}$, against $p = {pf}$ with $X$ as the input)."),
            Some(Verdict::Both) => match unrelated {
                Some(true) => r"What is left over looks independent of the input both ways, and $X$ and $Y$ look independent of each other: as far as these data go, neither causes the other.".to_string(),
                _ => format!(r"What is left over looks independent of the input both ways ($p = {pf}$ and $p = {pb}$): both directions fit."),
            },
            Some(Verdict::Neither) => format!(r"What is left over depends on the input both ways ($p = {pf}$ and $p = {pb}$): neither direction fits an effect plus independent noise."),
        };
        let quiet = d.fits[0].exact && d.fits[1].exact || single(&d.x) || single(&d.y);
        let why = if self.setup.amount == 0.0 && verdict.is_some() && !quiet { format!("{why} (There is no noise: what is left over is only the smoother's own small misses.)") } else { why };
        wrap(f, wmin, wmax.x - wmin.x, &why, lay.small, pal::TEXT);
    }

    fn draw_truth(&mut self, f: &mut Frame, lay: &Layout, pointer: Option<Vec2>) {
        let (min, max) = lay.truth;
        hud_panel(f, min - Vec2::splat(6.0), max + Vec2::splat(4.0));
        f.text_px(min, "truth", lay.text, pal::DIM, Align2::LEFT_TOP);
        let r = lay.node;
        if !self.show_truth {
            let size = Vec2::new((max.x - min.x).min(150.0), 26.0);
            let at = min + Vec2::new(0.0, lay.text + 8.0);
            if button(f, (at, at + size), "show the truth", lay.text, pointer) {
                self.show_truth = true;
            }
            if at.y + size.y + 6.0 + lay.small < max.y {
                f.text_px(at + Vec2::new(0.0, size.y + 6.0), "(guess first)", lay.small, pal::DIM, Align2::LEFT_TOP);
            }
            return;
        }
        let graph_y = min.y + lay.text + 10.0 + r;
        let gx = min.x + r + 4.0;
        let span = if lay.narrow { 70.0 } else { 92.0 };
        let (x, y) = (Vec2::new(gx, graph_y), Vec2::new(gx + span, graph_y));
        match self.setup.truth {
            Truth::XCausesY => hud_graph(f, &[Node { at: x, name: "$X$", color: CAUSE }, Node { at: y, name: "$Y$", color: EFFECT }], &[(0, 1, Link::Arrow, TRUTH, 2.2)], r),
            Truth::YCausesX => hud_graph(f, &[Node { at: x, name: "$X$", color: EFFECT }, Node { at: y, name: "$Y$", color: CAUSE }], &[(1, 0, Link::Arrow, TRUTH, 2.2)], r),
            Truth::Hidden => {
                // (under X and Y where the block has room; on a phone, in a row between them)
                let (x, y, z) = if lay.narrow {
                    (x, Vec2::new(gx + 7.2 * r, graph_y), Vec2::new(gx + 3.6 * r, graph_y))
                } else {
                    (x, y, Vec2::new(gx + span * 0.5, graph_y + 2.6 * r))
                };
                hud_graph(
                    f,
                    &[Node { at: x, name: "$X$", color: EFFECT }, Node { at: y, name: "$Y$", color: EFFECT }, Node { at: z, name: "$Z$", color: THIRD }],
                    &[(2, 0, Link::Arrow, TRUTH, 2.2), (2, 1, Link::Arrow, TRUTH, 2.2)],
                    r,
                );
            }
        }
        let w = f.text_size("hide", lay.small).x + 16.0;
        let at = Vec2::new(max.x - w, min.y - 3.0);
        if button(f, (at, at + Vec2::new(w, lay.small + 8.0)), "hide", lay.small, pointer) {
            self.show_truth = false;
        }
    }

    fn draw_counts(&mut self, f: &mut Frame, lay: &Layout, pointer: Option<Vec2>) {
        let (min, max) = lay.counts;
        hud_panel(f, min - Vec2::splat(6.0), max + Vec2::splat(4.0));
        let width = max.x - min.x;
        if !self.counting {
            f.text_px(min, "how often is the verdict right?", lay.text, pal::DIM, Align2::LEFT_TOP);
            let size = Vec2::new(width.min(300.0), 30.0);
            let at = min + Vec2::new(0.0, lay.text + 12.0);
            if button(f, (at, at + size), &format!("count over {RUNS} new samples"), lay.text, pointer) {
                self.counting = true;
                self.refresh();
            }
            return;
        }
        let t = &self.tally;
        let title = if t.done < RUNS {
            [format!("the verdict on {RUNS} new samples like this one ({} so far)", t.done), format!("{RUNS} new samples ({} so far)", t.done)]
        } else {
            [format!("the verdict on {RUNS} new samples like this one"), format!("{RUNS} new samples")]
        };
        let title = pick(f, &title, width, lay.text);
        f.text_px(min, title, lay.text, pal::DIM, Align2::LEFT_TOP);
        let bars = &self.look.bars.shown;
        if bars.len() < 5 {
            return;
        }
        let shown = self.show_truth;
        let (right, wrong) = (TRUTH, NAIVE);
        // the colour of a verdict: right or wrong when the truth is shown
        let arrow_color = |k: usize| -> Color {
            if !shown {
                return pal::TEXT.a(0.75);
            }
            match (self.setup.truth, k) {
                (Truth::XCausesY, 0) | (Truth::YCausesX, 2) => right,
                _ => wrong,
            }
        };
        let names = [r"$X \to Y$", "cannot tell", r"$Y \to X$"];
        let label_w = if lay.narrow { 0.0 } else { 120.0 };
        let bar_h = if lay.narrow { 14.0 } else { 18.0 };
        let mut y = min.y + lay.text + 10.0;
        for row in 0..2 {
            let (shares, colors): (Vec<f32>, Vec<Color>) = if row == 0 {
                (bars[0..3].to_vec(), vec![arrow_color(0), UNDECIDED, arrow_color(2)])
            } else {
                (bars[3..5].to_vec(), vec![arrow_color(0), arrow_color(2)])
            };
            let what = if row == 0 { "verdict" } else { "forced to pick" };
            let (x0, x1) = if lay.narrow { (min.x, max.x) } else { (min.x + label_w, max.x) };
            // (on a phone the row's name goes in front of its counts, under the bar)
            if !lay.narrow {
                f.text_px(Vec2::new(min.x, y + bar_h * 0.5), what, lay.small, pal::DIM, Align2::LEFT_CENTER);
            }
            hud_rect(f, Vec2::new(x0, y), Vec2::new(x1, y + bar_h), Color::hex(0x161C2B));
            let mut x = x0;
            for (share, color) in shares.iter().zip(&colors) {
                let w = share.max(0.0).min(1.0) * (x1 - x0);
                if w > 0.5 {
                    hud_rect_at(f, 2, Vec2::new(x, y), Vec2::new((x + w).min(x1), y + bar_h), *color);
                }
                x += w;
            }
            y += bar_h + 4.0;
            // what each part counts
            let counts: Vec<usize> = if row == 0 { t.verdicts.to_vec() } else { t.forced.to_vec() };
            let parts: Vec<(String, Color)> = if row == 0 {
                (0..3).map(|k| (format!("{} {}", names[k], counts[k]), colors[k])).collect()
            } else {
                vec![(format!("{} {}", names[0], counts[0]), colors[0]), (format!("{} {}", names[2], counts[1]), colors[1])]
            };
            let mut px = x0;
            if lay.narrow {
                let name = format!("{what}:");
                f.text_px(Vec2::new(px, y), name.as_str(), lay.small, pal::DIM, Align2::LEFT_TOP);
                px += f.text_size(&name, lay.small).x + 10.0;
            }
            for (text, color) in parts {
                let w = f.text_size(&text, lay.small).x;
                f.text_px(Vec2::new(px, y), text.as_str(), lay.small, if color == UNDECIDED { pal::DIM } else { color }, Align2::LEFT_TOP);
                px += w + 16.0;
            }
            y += lay.small + if lay.narrow { 7.0 } else { 10.0 };
        }
        if !lay.narrow && y + lay.small < max.y {
            let note = if self.setup.truth == Truth::Hidden { "Forced to pick, it can only be wrong here: neither arrow is there." } else { "Forced to pick: the side whose residuals look more independent (the larger p)." };
            f.text_px(Vec2::new(min.x, y), note, lay.small, pal::DIM, Align2::LEFT_TOP);
        }
    }
}

/// Grey: a verdict that does not pick a side.
const UNDECIDED: Color = pal::DIM;

/// The kinds of line drawn in batches of their own (see [`apart`]).
const GRID: u8 = 0;
const FITTED: u8 = 1;
const SLICES: u8 = 2;
const STRIPS: u8 = 3;
const BUTTONS: u8 = 4;
const LEGEND: u8 = 5;
/// (Dashed lines: the slices' all-together marks, then each column's true curve.)
const DASHED: u8 = 6;
const TRUE_CURVE: u8 = 7;

/// A batch of lines on top of everything for one kind of line alone. `eidos motion`
/// follows a batch's points by their places in it, so a kind whose number of points
/// changes from frame to frame (grid lines as an axis glides, a curve cut at an edge,
/// a mark that comes and goes) is kept apart from the others, whose points then keep
/// their places. (The fading is for trails: in a line batch it only tells one batch
/// from another.)
fn apart<'b>(f: &'b mut Frame<'_>, kind: u8, dash: bool) -> &'b mut crate::gfx::draw::LineBatch {
    let style = if dash { LineStyle { dash: 7.0, duty: 0.55, depth_bias: 0.0, fade_power: 30.0 + kind as f32, ..Default::default() } } else { LineStyle { depth_bias: 0.0, fade_power: 30.0 + kind as f32, ..Default::default() } };
    f.dl.lines_with(LineKind::Overlay, style)
}

/// A line in screen space, in the batch of its kind.
fn line_of(f: &mut Frame, kind: u8, a: Vec2, b: Vec2, width: f32, color: Color) {
    let (a, b) = (f.hud(a), f.hud(b));
    apart(f, kind, false).segment(a, b, width, color);
}

/// A polyline in screen space, in the batch of its kind.
fn path_of(f: &mut Frame, kind: u8, pts: &[Vec2], width: f32, color: Color) {
    let points: Vec<Vec3> = pts.iter().map(|p| f.hud(*p)).collect();
    apart(f, kind, false).polyline(points, width, color);
}

/// A translucent rectangle in screen space, over a panel's backing.
fn fill(f: &mut Frame, min: Vec2, max: Vec2, color: Color) {
    let c = [min, Vec2::new(max.x, min.y), max, Vec2::new(min.x, max.y)].map(|p| f.hud_at(1, p));
    f.dl.tris_alpha.quad(c[0], c[1], c[2], c[3], color);
}

/// A polyline kept inside a rectangle: a point outside it next to one inside is
/// brought back along the segment between them to the edge, and one with no
/// neighbour inside is left out (the line has a gap there).
fn clip(pts: &[Vec2], (min, max): Rect) -> Vec<Vec2> {
    let inside = |p: Vec2| p.is_finite() && p.x >= min.x - 0.5 && p.x <= max.x + 0.5 && p.y >= min.y - 0.5 && p.y <= max.y + 0.5;
    // where the segment from `a` (inside) towards `b` leaves the rectangle
    let edge = |a: Vec2, b: Vec2| {
        let d = b - a;
        let mut t = 1.0f32;
        if d.x != 0.0 {
            t = t.min(((if d.x > 0.0 { max.x } else { min.x }) - a.x) / d.x);
        }
        if d.y != 0.0 {
            t = t.min(((if d.y > 0.0 { max.y } else { min.y }) - a.y) / d.y);
        }
        a + d * t.max(0.0)
    };
    (0..pts.len())
        .map(|i| {
            let p = pts[i];
            if inside(p) || !p.is_finite() {
                return p;
            }
            let prev = i.checked_sub(1).map(|j| pts[j]).filter(|q| inside(*q));
            let next = pts.get(i + 1).copied().filter(|q| inside(*q));
            match (prev, next) {
                (Some(a), _) | (None, Some(a)) => edge(a, p),
                _ => Vec2::NAN,
            }
        })
        .collect()
}

/// A dashed polyline in screen space, kept inside a rectangle, in the batch of its kind.
fn dashed_px(f: &mut Frame, kind: u8, pts: &[Vec2], width: f32, color: Color, rect: Rect) {
    let pts = clip(pts, rect);
    let points: Vec<Vec3> = pts.iter().map(|p| f.hud(*p)).collect();
    apart(f, kind, true).polyline(points, width, color);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::math::causal::kurtosis;

    /// An experiment as its button sets it up, with the tests of the sample it shows
    /// run to the end.
    fn run(k: usize) -> Direction {
        let mut s = Direction::create();
        s.set_up(k);
        s.settle();
        s
    }

    /// p of side `c`, as the picture writes it.
    fn p(s: &Direction, c: usize) -> String {
        num(s.data.sides[c].p(), 2)
    }

    /// The spreads (10th to 90th percentile) of what is left over in each slice of
    /// column `c`, as a desktop picture draws them.
    fn spreads(s: &Direction, c: usize) -> Vec<f64> {
        let (input, _) = s.data.pairs(c);
        let k = slice_count(s.data.x.len(), 480.0);
        let v = slices(input, &s.data.fits[c].residuals, k);
        (0..k).map(|j| (v[j * PER_SLICE + 6] - v[j * PER_SLICE + 2]) as f64).collect()
    }

    /// The widest slice's spread over the narrowest's.
    fn ratio(v: &[f64]) -> f64 {
        v.iter().copied().fold(0.0f64, f64::max) / v.iter().copied().fold(f64::MAX, f64::min)
    }

    /// The samples the "things to try" show, held to their captions.
    #[test]
    fn the_samples_shown_say_what_the_captions_say() {
        let all: Vec<Direction> = (0..TRIALS.len()).map(run).collect();
        // what is left over is uncorrelated with its input, near enough, every time
        for (k, s) in all.iter().enumerate() {
            for c in 0..2 {
                assert!(s.data.fits[c].corr.abs() < 0.07, "{}, side {c}: correlation {}", TRIALS[k].0, s.data.fits[c].corr);
            }
            assert!(s.counting && s.show_truth && s.seed == FIRST_SEED);
        }

        // a bent cause: a level band on the left; on the right narrow in the middle and
        // wide at both ends, its widest slice 3.5 times its narrowest (1.5 on the left);
        // p = 0.94 against 0.01, and X -> Y
        let bent = &all[0];
        assert!((bent.setup.truth, bent.setup.curve, bent.setup.noise) == (Truth::XCausesY, Curve::S, Noise::Normal));
        assert_eq!(bent.data.verdict(), Some(Verdict::XCausesY));
        assert_eq!((p(bent, 0), p(bent, 1)), ("0.94".to_string(), "0.01".to_string()));
        let (left, right) = (spreads(bent, 0), spreads(bent, 1));
        assert!(format!("{:.1}", ratio(&left)) == "1.5" && format!("{:.1}", ratio(&right)) == "3.5", "{} and {}", ratio(&left), ratio(&right));
        let narrowest = (0..right.len()).min_by(|&a, &b| right[a].total_cmp(&right[b])).unwrap_or(0);
        assert!(narrowest > 0 && narrowest + 1 < right.len(), "{right:?}");
        assert!(right[0] > 2.5 * right[narrowest] && right[right.len() - 1] > 2.5 * right[narrowest], "{right:?}");

        // turned round: the same points with X and Y swapped; the columns trade
        // pictures, and the verdict turns round with them
        let turned = &all[1];
        assert!(turned.data.x == bent.data.y && turned.data.y == bent.data.x);
        assert!(turned.data.fits[0].residuals == bent.data.fits[1].residuals && turned.data.fits[1].residuals == bent.data.fits[0].residuals);
        assert_eq!(turned.data.verdict(), Some(Verdict::YCausesX));

        // a straight line, flat noise: uncorrelated both ways (0.01 and 0.00); on the
        // right narrow at both ends and wide in the middle, dependent; on the left a
        // level band; p = 0.94 and 0.01
        let flat = &all[2];
        assert!((flat.setup.curve, flat.setup.noise) == (Curve::Line, Noise::Uniform));
        assert_eq!((num(flat.data.fits[0].corr, 2), num(flat.data.fits[1].corr, 2)), ("0.01".to_string(), "0.00".to_string()));
        assert_eq!(flat.data.verdict(), Some(Verdict::XCausesY));
        assert_eq!((p(flat, 0), p(flat, 1)), ("0.94".to_string(), "0.01".to_string()));
        let (left, right) = (spreads(flat, 0), spreads(flat, 1));
        let widest = right.iter().copied().fold(0.0f64, f64::max);
        assert!(right[0] < 0.55 * widest && right[right.len() - 1] < 0.4 * widest && ratio(&left) < 1.4, "{left:?} {right:?}");

        // a straight line, bell noise: both ways a level band, p = 0.87 and 0.61, and
        // the verdict that it cannot tell
        let bell = &all[3];
        assert!((bell.setup.curve, bell.setup.noise) == (Curve::Line, Noise::Normal));
        assert_eq!(bell.data.verdict(), Some(Verdict::Both));
        assert_eq!((p(bell, 0), p(bell, 1)), ("0.87".to_string(), "0.61".to_string()));
        assert!(ratio(&spreads(bell, 0)) < 1.7 && ratio(&spreads(bell, 1)) < 1.7);

        // twenty points: p = 0.99 both ways, and it cannot tell
        let few = &all[4];
        assert!(few.data.x.len() == 20 && few.setup.curve == Curve::S);
        assert_eq!(few.data.verdict(), Some(Verdict::Both));
        assert_eq!((p(few, 0), p(few, 1)), ("0.99".to_string(), "0.99".to_string()));

        // a hidden common cause: X -> Y, p = 0.47 against 0.01
        let hidden = &all[5];
        assert!(hidden.setup.truth == Truth::Hidden && hidden.true_column().is_none());
        assert_eq!(hidden.data.verdict(), Some(Verdict::XCausesY));
        assert_eq!((p(hidden, 0), p(hidden, 1)), ("0.47".to_string(), "0.01".to_string()));
    }

    /// The counts over new samples that the captions quote (a few seconds: run with
    /// --ignored).
    #[test]
    #[ignore]
    fn the_counts_in_the_captions_hold() {
        let counts: Vec<([usize; 3], [usize; 2])> = std::thread::scope(|scope| {
            let runs: Vec<_> = (0..TRIALS.len())
                .map(|k| {
                    scope.spawn(move || {
                        let mut s = run(k);
                        s.tally.finish();
                        assert_eq!(s.tally.done, RUNS);
                        (s.tally.verdicts, s.tally.forced)
                    })
                })
                .collect();
            runs.into_iter().map(|r| r.join().expect("a count")).collect()
        });
        // (X -> Y, cannot tell, Y -> X; and forced to pick, X -> Y, Y -> X)
        assert_eq!(counts[0].0, [99, 1, 0], "a bent cause");
        assert_eq!(counts[1].0, [0, 3, 97], "turned round");
        assert_eq!(counts[2].0, [100, 0, 0], "a straight line, flat noise");
        assert_eq!(counts[3], ([0, 100, 0], [51, 49]), "a straight line, bell noise");
        assert_eq!(counts[4], ([6, 94, 0], [78, 22]), "twenty points");
        assert_eq!(counts[5].0, [94, 6, 0], "a hidden common cause");
    }

    /// Each shape of noise, drawn through its quantiles, is the shared one: mean 0,
    /// spread 1, the tails its name promises, and as much of it below each value as
    /// its density says.
    #[test]
    fn the_noise_drawn_has_the_shapes_it_is_said_to() {
        for (noise, tails) in [(Noise::Normal, 0.0), (Noise::Uniform, -1.2), (Noise::Laplace, 3.0), (Noise::TwoHumps, -1.34)] {
            // (as many as the shared test of these shapes draws: a heavy tail's kurtosis is slow to settle)
            let mut rng = Rng::new(9);
            let v: Vec<f64> = (0..200_000).map(|_| quantile(noise, uniform(&mut rng))).collect();
            assert!(mean(&v).abs() < 0.01 && (variance(&v) - 1.0).abs() < 0.02, "{noise:?}: mean {}, variance {}", mean(&v), variance(&v));
            assert!((kurtosis(&v) - tails).abs() < 0.15, "{noise:?}: kurtosis {}", kurtosis(&v));
            for at in [-1.5, -0.6, 0.0, 0.3, 1.1] {
                let share = v.iter().filter(|v| **v < at).count() as f64 / v.len() as f64;
                let steps = 6000;
                let h = (at + 12.0) / steps as f64;
                let said: f64 = (0..steps).map(|i| noise.density(-12.0 + (i as f64 + 0.5) * h) * h).sum();
                assert!((share - said).abs() < 0.004, "{noise:?}: {share} below {at}, the density says {said}");
            }
            // the same uniform numbers give points of the same rank in every shape
            let other = Draws::new(9, 1000, Noise::Uniform);
            let mine = Draws::new(9, 1000, noise);
            for i in 1..1000 {
                assert_eq!(other.parts[0][i] < other.parts[0][i - 1], mine.parts[0][i] < mine.parts[0][i - 1]);
            }
        }
        assert!((bell_below(0.0) - 0.5).abs() < 1e-7 && (bell_below(1.96) - 0.975).abs() < 1e-4 && bell_below(-40.0) >= 0.0);
        assert!(quantile(Noise::Normal, 0.0).is_finite() && quantile(Noise::Laplace, 1.0).is_finite() && quantile(Noise::TwoHumps, 0.5).abs() < 1e-6);
    }

    /// The drawn curve passes through its points, smoothly, and goes straight on
    /// beyond the first and the last.
    #[test]
    fn the_drawn_curve_passes_through_its_points() {
        let h = [1.2, -0.6, 0.0, 0.9, -1.0];
        for k in 0..KNOTS.len() {
            assert!((through(&h, KNOTS[k]) - h[k]).abs() < 1e-12);
            // no jump at a point
            assert!((through(&h, KNOTS[k] - 1e-7) - through(&h, KNOTS[k] + 1e-7)).abs() < 1e-5);
        }
        for side in [-1.0, 1.0] {
            let (a, b, c) = (through(&h, 3.0 * side), through(&h, 4.0 * side), through(&h, 5.0 * side));
            assert!((a - 2.0 * b + c).abs() < 1e-9, "not straight beyond the ends");
        }
        assert!(through(&h, f64::NAN).is_finite() && through(&h, 1e300).is_finite() || through(&h, 1e300).is_infinite());
        // the presets' own values at the points, which a first drag starts from
        for (curve, _) in CURVES {
            for x in KNOTS {
                assert!(curve.at(x, &DRAWN).is_finite());
            }
        }
    }

    /// The smoother follows a straight line exactly, to its ends; a curve closely,
    /// where the points are; and leaves nothing to test when nothing is left over.
    #[test]
    fn the_smoother_follows_what_it_is_given() {
        let draws = Draws::new(3, 200, Noise::Normal);
        let x = &draws.parts[0];
        let line: Vec<f64> = x.iter().map(|v| 2.0 * v + 1.0).collect();
        let r = leftover(x, &line);
        assert!(r.iter().all(|v| v.abs() < 1e-9) && exact(&line, &r));
        let s: Vec<f64> = x.iter().zip(&draws.parts[1]).map(|(c, n)| 1.6 * (1.4 * c).tanh() + 0.4 * n).collect();
        let f = fit(x, &s);
        assert!(!f.exact && f.curve.len() == CURVE);
        for &(at, v) in &f.curve {
            if at.abs() < 1.5 {
                assert!((v - 1.6 * (1.4 * at).tanh()).abs() < 0.2, "at {at}: {v}");
            }
        }
        // the width widens where the points thin out
        let w = Width::of(x);
        assert!(w.at(0.0) >= w.least && w.at(3.5) > 1.5 * w.at(0.0), "{} and {}", w.at(0.0), w.at(3.5));
        assert!(Width::of(&[]).at(0.0) > 0.0 && Width::of(&[1.0; 5]).at(1.0) > 0.0);
    }

    /// Stopping the shuffles as soon as the answer is known gives the answer that all
    /// of them would; and the count's verdicts are what the two tests say.
    #[test]
    fn stopping_early_gives_the_same_answer() {
        assert!(p_of(ENOUGH, SHUFFLES) >= LEVEL && p_of(ENOUGH - 1, SHUFFLES) < LEVEL);
        let (mut independent, mut dependent) = (0, 0);
        for (curve, noise, amount, seed) in [(Curve::S, Noise::Normal, 0.4, 1u64), (Curve::Line, Noise::Uniform, 1.0, 2), (Curve::Line, Noise::Normal, 1.0, 3), (Curve::S, Noise::Normal, 0.4, 4)] {
            let setup = Setup { truth: Truth::XCausesY, curve, noise, amount, n: 120, heights: DRAWN };
            let (x, y) = sample(&setup, seed);
            for (input, output) in [(&x, &y), (&y, &x)] {
                let mut q = Quick::new(input, output);
                let h = Hsic::new(input, &q.residuals);
                q.value = h.value();
                let all = (0..SHUFFLES).filter({
                    let mut rng = Rng::new(seed);
                    let value = q.value;
                    move |_| h.shuffled(&mut rng) >= value
                });
                let full = p_of(all.count(), SHUFFLES) >= LEVEL;
                q.hsic = Some(Hsic::new(input, &q.residuals));
                let mut rng = Rng::new(seed);
                while !q.decided() {
                    q.shuffle(&mut rng);
                }
                assert_eq!(q.independent(), full);
                if full {
                    independent += 1;
                } else {
                    dependent += 1;
                }
            }
        }
        assert!(independent >= 3 && dependent >= 2, "{independent} independent, {dependent} dependent");
        assert_eq!([Verdict::of(true, false), Verdict::of(false, true), Verdict::of(true, true), Verdict::of(false, false)], [Verdict::XCausesY, Verdict::YCausesX, Verdict::Both, Verdict::Neither]);
        assert_eq!([Verdict::XCausesY, Verdict::Both, Verdict::Neither, Verdict::YCausesX].map(Verdict::slot), [0, 1, 1, 2]);
    }

    /// No noise, a flat curve, a single value, the fewest points, the most: every
    /// number drawn is finite, the tests finish, and the verdict says what happened.
    #[test]
    fn degenerate_data_are_told_honestly() {
        let lays = [Layout::of(Vec2::new(1075.0, 1000.0), 0.0), Layout::of(Vec2::new(390.0, 756.0), 44.0), Layout::of(Vec2::new(240.0, 180.0), 0.0)];
        let mut flat = DRAWN;
        flat.iter_mut().for_each(|h| *h = 0.5);
        for truth in TRUTHS {
            for (curve, heights) in CURVES.map(|c| (c.0, DRAWN)).into_iter().chain([(Curve::Drawn, flat)]) {
                for amount in [0.0, 0.4] {
                    for n in [10, 300] {
                        let mut s = Direction::create();
                        s.setup = Setup { truth, curve, noise: Noise::Laplace, amount, n, heights };
                        s.settle();
                        let d = &s.data;
                        assert!(d.verdict().is_some() && !d.pending());
                        for c in 0..2 {
                            assert!(d.fits[c].residuals.iter().all(|v| v.is_finite()) && d.fits[c].curve.iter().all(|(a, b)| a.is_finite() && b.is_finite()));
                            assert!(d.sides[c].p() > 0.0 && d.sides[c].p() <= 1.0);
                        }
                        for lay in &lays {
                            let g = s.goals(lay);
                            let numbers = g.points.iter().chain(&g.residuals).chain(&g.ranges).chain(&g.curves[0]).chain(&g.curves[1]).chain(&g.truth).chain(&g.slices[0]).chain(&g.slices[1]).chain(&g.strips).chain(&g.bars);
                            assert!(numbers.clone().all(|v| v.is_finite()), "{truth:?} {curve:?} {amount} {n}");
                        }
                    }
                }
            }
        }
        // no noise on a straight line: nothing is left over either way
        let mut s = Direction::create();
        s.setup = Setup { truth: Truth::XCausesY, curve: Curve::Line, noise: Noise::Normal, amount: 0.0, n: 200, heights: DRAWN };
        s.settle();
        assert!(s.data.fits[0].exact && s.data.fits[1].exact && s.data.verdict() == Some(Verdict::Both));
        // a flat curve with noise: X and Y look independent of each other
        s.setup = Setup { truth: Truth::XCausesY, curve: Curve::Drawn, noise: Noise::Normal, amount: 0.5, n: 200, heights: [0.0; 5] };
        s.settle();
        assert_eq!((s.data.verdict(), s.data.unrelated()), (Some(Verdict::Both), Some(true)));
        // a flat curve and no noise: Y takes a single value
        s.setup = Setup { truth: Truth::XCausesY, curve: Curve::Drawn, noise: Noise::Normal, amount: 0.0, n: 200, heights: flat };
        s.settle();
        assert!(variance(&s.data.y) == 0.0 && s.data.fits[0].exact);
    }

    /// The layout of every window: finite, inside the canvas, the pictures of a column
    /// one under another, and the columns side by side, also on a phone.
    #[test]
    fn the_layout_fits_every_window() {
        for (w, h, inset) in [(1075.0, 1000.0, 0.0), (1400.0, 760.0, 0.0), (390.0, 756.0, 44.0), (700.0, 500.0, 0.0), (240.0, 180.0, 0.0), (1.0, 1.0, 0.0), (3000.0, 2000.0, 0.0)] {
            let lay = Layout::of(Vec2::new(w, h), inset);
            let rects = [lay.scatter, lay.resid, lay.strip].concat();
            let all = rects.iter().chain([&lay.verdict, &lay.truth, &lay.words, &lay.counts]);
            assert!(all.clone().all(|(a, b)| a.is_finite() && b.is_finite()), "{w}x{h}");
            if w >= 390.0 && h >= 700.0 {
                for c in 0..2 {
                    assert!(lay.scatter[c].1.y < lay.resid[c].0.y && lay.resid[c].1.y < lay.strip[c].0.y && lay.strip[c].1.y < lay.verdict.0.y, "{w}x{h}");
                    assert!(lay.scatter[c].0.y >= inset && lay.scatter[c].1.y - lay.scatter[c].0.y > 100.0, "{w}x{h}");
                }
                assert!(lay.scatter[0].1.x < lay.scatter[1].0.x && lay.scatter[1].1.x <= w);
                assert!(all.clone().all(|(a, b)| a.x >= 0.0 && a.y >= 0.0 && b.x <= w + 0.5 && b.y <= h + 0.5), "{w}x{h}");
                assert!(lay.counts.1.y - lay.counts.0.y > 60.0, "{w}x{h}: room for the counts");
            }
        }
    }

    /// After a change, the picture glides to the new one and comes to rest exactly
    /// there, at any frame rate.
    #[test]
    fn glides_arrive_and_stop() {
        let lay = Layout::of(Vec2::new(1075.0, 1000.0), 0.0);
        for hz in [30.0f32, 60.0, 144.0] {
            let mut s = Direction::create();
            s.settle();
            s.look.follow(&s.goals(&lay), 0.0, true);
            assert!(!s.look.moving);
            s.set_up(1);
            s.settle();
            let mut frames = 0;
            loop {
                let g = s.goals(&lay);
                s.look.follow(&g, 1.0 / hz, false);
                frames += 1;
                if !s.look.moving {
                    assert!(s.look.points.shown == g.points && s.look.curves[0].shown == g.curves[0]);
                    break;
                }
                assert!((frames as f32) < 2.0 * hz, "still gliding after two seconds at {hz} Hz");
            }
            assert!(frames as f32 > 0.2 * hz, "arrived in {frames} frames at {hz} Hz: no glide");
        }
    }

    /// A change of setting starts the tests afresh, and the count too once it is kept.
    #[test]
    fn a_change_starts_the_tests_afresh() {
        let mut s = Direction::create();
        s.settle();
        assert!(!s.data.pending() && !s.counting && !s.tally.pending());
        s.setup.noise = Noise::Uniform;
        s.refresh();
        assert!(s.data.pending() && s.data.sides[0].shuffles.is_empty() && !s.tally.pending());
        s.counting = true;
        s.refresh();
        assert!(s.tally.pending() && s.tally.done == 0);
        // the work is in small pieces: two tables, then one shuffle at a time
        s.data.step();
        assert!(s.data.sides[0].hsic.is_some() && s.data.sides[1].hsic.is_none());
        s.data.step();
        s.data.step();
        assert_eq!(s.data.sides[0].shuffles.len() + s.data.sides[1].shuffles.len(), 1);
        s.seed += 1;
        s.refresh();
        assert!(s.tally.key == Some((s.setup, s.seed)) && s.tally.done == 0);
    }



    /// Candidate pictures, compared (run with --ignored --nocapture).
    #[test]
    #[ignore]
    fn candidates() {
        for (curve, noise, amount) in [(Curve::S, Noise::Normal, 0.4), (Curve::Bend, Noise::Normal, 0.4), (Curve::Parabola, Noise::Normal, 0.4), (Curve::Drawn, Noise::Normal, 0.4), (Curve::Line, Noise::Uniform, 1.0), (Curve::Line, Noise::Normal, 1.0), (Curve::Line, Noise::Laplace, 1.0), (Curve::Line, Noise::TwoHumps, 1.0)] {
            for n in [100, 200, 300] {
                let setup = Setup { truth: Truth::XCausesY, curve, noise, amount, n, heights: DRAWN };
                let mut counts = [0usize; 3];
                for seed in 1..=40u64 {
                    let mut d = Data::new(setup, seed, &Draws::new(seed, n, noise));
                    while d.pending() {
                        d.step();
                    }
                    if let Some(v) = d.verdict() {
                        counts[v.slot()] += 1;
                    }
                }
                println!("{curve:?} {noise:?} {amount} n={n}: {counts:?}");
            }
        }
    }

    /// The numbers behind the captions (run with --ignored --nocapture).
    #[test]
    #[ignore]
    fn explore() {
        for k in 0..TRIALS.len() {
            let mut s = run(k);
            let d = &s.data;
            println!(
                "{}: verdict {:?}, p {:.2} {:.2}, corr {:.3} {:.3}, unrelated {:?}, spread ratio {:.2} {:.2}",
                TRIALS[k].0,
                d.verdict(),
                d.sides[0].p(),
                d.sides[1].p(),
                d.fits[0].corr,
                d.fits[1].corr,
                d.unrelated(),
                ratio(&spreads(&s, 0)),
                ratio(&spreads(&s, 1)),
            );
            s.tally.finish();
            println!("    count {:?}, forced {:?}", s.tally.verdicts, s.tally.forced);
        }
    }
}

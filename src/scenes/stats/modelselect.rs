//! Overfitting and cross-validation.
//!
//! Points scattered about a known curve, and polynomials of every degree from 0 to 15
//! fitted to them by least squares (or one degree with a ridge penalty of every
//! strength). On the left, the data and the curve of the size marked, or each fold held
//! back in turn with the curve fitted to the rest and its misses. On the right, three
//! errors against the size: on the points fitted to, on points held back
//! (cross-validation), and, since the truth is known here, the true error on new data to
//! judge both by. Under it, what cross-validation chooses on many new data sets from the
//! same truth. And a contest of twenty useless lines, whose winner looks good only on the
//! points that chose it.
//!
//! The polynomials are written in Legendre polynomials (in raw powers, degree 15 loses
//! its digits), and fitted by Givens rotations row by row into one triangle that holds
//! the fits of every degree at once. Cross-validation fits again without each fold;
//! leaving one out uses the exact shortcut through the leverages (both are tested
//! against fitting afresh by another method).

use super::plot::{self, Glide, Plot};
use super::*;
use crate::math::stats::{normal, unit};

pub const DEF: SceneDef = SceneDef {
    id: "modelselect",
    title: "Overfitting and cross-validation",
    group: MODELS,
    blurb: "A model can fit its own data too well: error on data held back, cross-validation, and a penalty on size.",
    make: |_| Box::new(ModelSelect::create()),
};

// ----- polynomials -----------------------------------------------------------------------

/// The highest degree fitted, and the number of coefficients that takes.
const MAX_DEG: usize = 15;
const P: usize = MAX_DEG + 1;

/// The Legendre polynomials at x, each scaled so that its mean square over -1 to 1 is 1:
/// p_k = √(2k + 1) P_k. Over that interval they are orthogonal, so a curve's coefficients
/// in them say how much of each it has, and the sum of the squares of all but the first
/// is the curve's mean squared distance from its own average. (Raw powers of x are
/// nearly parallel by degree 10, and a fit in them loses its digits.)
fn legendre(x: f64) -> [f64; P] {
    let mut p = [0.0; P];
    p[0] = 1.0;
    p[1] = x;
    for k in 1..MAX_DEG {
        p[k + 1] = ((2 * k + 1) as f64 * x * p[k] - k as f64 * p[k - 1]) / (k + 1) as f64;
    }
    for (k, v) in p.iter_mut().enumerate() {
        *v *= ((2 * k + 1) as f64).sqrt();
    }
    p
}

/// The curve with coefficients `c` at the point whose Legendre values are `row`.
fn value(c: &[f64; P], row: &[f64; P]) -> f64 {
    c.iter().zip(row).map(|(a, b)| a * b).sum()
}

// ----- the truth -------------------------------------------------------------------------

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Shape {
    Wave,
    Step,
    Corner,
    Line,
}

const SHAPES: [(Shape, &str); 4] = [(Shape::Wave, "wave"), (Shape::Step, "smooth step"), (Shape::Corner, "corner"), (Shape::Line, "straight line")];

/// Where the corner is.
const CORNER: f64 = 0.2;

impl Shape {
    fn at(self, x: f64) -> f64 {
        match self {
            Shape::Wave => (3.0 * x + 0.6).sin() + 0.3 * x,
            Shape::Step => (5.0 * (x - 0.1)).tanh(),
            Shape::Corner => 1.5 * (x - CORNER).abs() - 0.5,
            Shape::Line => 0.8 * x,
        }
    }
}

/// A shape in the Legendre polynomials: its coefficients up to degree 15 and its mean
/// square over -1 to 1. With them, how far any polynomial is from the shape on average
/// is a sum of squares (see [`Truth::distance`]).
#[derive(Clone, Copy, Debug)]
struct Truth {
    coef: [f64; P],
    square: f64,
}

impl Truth {
    /// By Simpson's rule on 4000 pieces, which has the corner at a joint between two of
    /// them (where the rule assumes nothing smooth).
    fn of(shape: Shape) -> Truth {
        const N: usize = 4000;
        let h = 2.0 / N as f64;
        let (mut coef, mut square) = ([0.0; P], 0.0);
        for i in 0..=N {
            let x = -1.0 + h * i as f64;
            let weight = if i == 0 || i == N {
                1.0
            } else if i % 2 == 1 {
                4.0
            } else {
                2.0
            } * h
                / 6.0;
            let (y, p) = (shape.at(x), legendre(x));
            for k in 0..P {
                coef[k] += weight * y * p[k];
            }
            square += weight * y * y;
        }
        Truth { coef, square }
    }

    /// The mean squared distance, over -1 to 1, between the polynomial of degree `d` with
    /// coefficients `c` and the shape: the coefficients' differences squared, and what of
    /// the shape no polynomial of that degree has.
    fn distance(&self, c: &[f64; P], d: usize) -> f64 {
        let d = d.min(MAX_DEG);
        let beyond = (self.square - self.coef[..=d].iter().map(|a| a * a).sum::<f64>()).max(0.0);
        beyond + (0..=d).map(|k| (c[k] - self.coef[k]).powi(2)).sum::<f64>()
    }
}

// ----- least squares, row by row ----------------------------------------------------------

/// A least-squares problem in the P Legendre columns brought, one row at a time, by
/// Givens rotations, to a triangle R and the right-hand side z = Qᵀy. Its leading corner
/// of d + 1 rows and columns is the same problem for the first d + 1 columns alone: one
/// triangle holds the fits of every degree.
#[derive(Clone, Debug)]
struct Tri {
    r: [[f64; P]; P],
    z: [f64; P],
    /// The squared length of each column of what was added, to judge whether a column
    /// is independent of those before it.
    size: [f64; P],
    /// How many data rows (penalty rows are not counted).
    rows: usize,
}

impl Tri {
    fn new() -> Tri {
        Tri { r: [[0.0; P]; P], z: [0.0; P], size: [0.0; P], rows: 0 }
    }

    /// Of the points of `data` that `keep` keeps.
    fn of(data: &Data, keep: impl Fn(usize) -> bool) -> Tri {
        let mut t = Tri::new();
        for i in (0..data.x.len()).filter(|&i| keep(i)) {
            t.add(&data.rows[i], data.y[i]);
        }
        t
    }

    fn add(&mut self, row: &[f64; P], y: f64) {
        for (s, v) in self.size.iter_mut().zip(row) {
            *s += v * v;
        }
        self.rows += 1;
        self.rotate_in(*row, y, 0);
    }

    /// A row that is `s` in column `k` and 0 elsewhere, with 0 to fit: a ridge penalty
    /// of s² on that coefficient.
    fn add_penalty(&mut self, k: usize, s: f64) {
        let mut row = [0.0; P];
        row[k] = s;
        self.size[k] += s * s;
        self.rotate_in(row, 0.0, k);
    }

    fn rotate_in(&mut self, mut row: [f64; P], mut y: f64, from: usize) {
        for k in from..P {
            if row[k] == 0.0 {
                continue;
            }
            let (a, b) = (self.r[k][k], row[k]);
            let rho = a.hypot(b);
            let (c, s) = (a / rho, b / rho);
            self.r[k][k] = rho;
            row[k] = 0.0;
            for j in k + 1..P {
                let (u, v) = (self.r[k][j], row[j]);
                self.r[k][j] = c * u + s * v;
                row[j] = c * v - s * u;
            }
            let (u, v) = (self.z[k], y);
            self.z[k] = c * u + s * v;
            y = c * v - s * u;
        }
    }

    /// The least-squares coefficients in the first `d` + 1 columns (zero beyond), if
    /// those columns are independent: otherwise many curves fit equally well, and there
    /// is no one answer.
    fn solve(&self, d: usize) -> Option<[f64; P]> {
        // (a diagonal that is not a number counts as nothing: hence the test this way round)
        let independent = |k: usize| self.r[k][k].abs().partial_cmp(&(1e-11 * self.size[k].sqrt())) == Some(std::cmp::Ordering::Greater);
        if d >= P || !(0..=d).all(independent) {
            return None;
        }
        let mut c = [0.0; P];
        for k in (0..=d).rev() {
            let s = self.z[k] - (k + 1..=d).map(|j| self.r[k][j] * c[j]).sum::<f64>();
            c[k] = s / self.r[k][k];
        }
        c.iter().all(|v| v.is_finite()).then_some(c)
    }

    /// The leverage of a point with these Legendre values in the fit of each degree
    /// (the d-th number for the first d + 1 columns): ‖R⁻ᵀ row‖² over the leading corner,
    /// which nests like the fits. A point's leverage is how much its own y moves its own
    /// fitted value.
    fn leverage(&self, row: &[f64; P]) -> [f64; P] {
        let (mut w, mut out, mut sum) = ([0.0; P], [0.0; P], 0.0);
        for k in 0..P {
            let s = row[k] - (0..k).map(|j| self.r[j][k] * w[j]).sum::<f64>();
            w[k] = s / self.r[k][k];
            sum += w[k] * w[k];
            out[k] = sum;
        }
        out
    }
}

// ----- the data ----------------------------------------------------------------------------

/// Points with x spread evenly at random over -1 to 1 and y the truth there plus noise.
/// The noise is kept as standard normal draws `e`, so that a change of its size or of
/// the truth moves the same points (and more points keep the first ones).
#[derive(Clone, Debug, Default)]
struct Data {
    x: Vec<f64>,
    e: Vec<f64>,
    y: Vec<f64>,
    /// The Legendre values at each x.
    rows: Vec<[f64; P]>,
}

impl Data {
    fn draw(seed: u64, n: usize, shape: Shape, noise: f64) -> Data {
        let mut rng = Rng::new(seed);
        let (mut x, mut e) = (Vec::with_capacity(n), Vec::with_capacity(n));
        for _ in 0..n {
            x.push(2.0 * unit(&mut rng) - 1.0);
            e.push(normal(&mut rng));
        }
        let rows = x.iter().map(|&v| legendre(v)).collect();
        let mut d = Data { x, e, y: Vec::new(), rows };
        d.set_y(shape, noise);
        d
    }

    fn set_y(&mut self, shape: Shape, noise: f64) {
        self.y = self.x.iter().zip(&self.e).map(|(&x, &e)| shape.at(x) + noise * e).collect();
    }

    fn len(&self) -> usize {
        self.x.len()
    }
}

/// The fold of each of `n` points, `k` folds dealt like cards from a shuffled pack: the
/// folds' sizes differ by one at most.
fn deal(n: usize, k: usize, seed: u64) -> Vec<usize> {
    let mut order: Vec<usize> = (0..n).collect();
    Rng::new(seed).shuffle(&mut order);
    let mut fold = vec![0; n];
    for (j, &i) in order.iter().enumerate() {
        fold[i] = j % k.max(1);
    }
    fold
}

// ----- what is compared ------------------------------------------------------------------------

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Mode {
    /// Least squares of every degree from 0 to 15.
    Degree,
    /// One degree with a ridge penalty of every strength in `PENALTIES`.
    Penalty,
    /// Twenty straight lines, each on an input of pure noise.
    Useless,
}

/// The strengths of the ridge penalty compared: 1, 2 and 5 times each power of ten.
const PENALTIES: [f64; 28] = [
    1e-7, 2e-7, 5e-7, 1e-6, 2e-6, 5e-6, 1e-5, 2e-5, 5e-5, 1e-4, 2e-4, 5e-4, 1e-3, 2e-3, 5e-3, 1e-2, 2e-2, 5e-2, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0,
];

/// The number of useless models in the contest.
const LINES: usize = 20;

/// Everything the numbers depend on (not which data set, nor how the folds are dealt).
#[derive(Clone, Copy, Debug, PartialEq)]
struct Setup {
    shape: Shape,
    /// The noise's standard deviation σ.
    noise: f64,
    /// Points (in the contest, the points each line is fitted to, and as many more held back).
    n: usize,
    folds: usize,
    /// Leave one out: as many folds as points.
    loo: bool,
    mode: Mode,
    /// The degree given a penalty (in that mode).
    degree: usize,
}

impl Setup {
    /// The folds used: as asked, but never more than the points (then each is a fold).
    fn folds_used(&self) -> usize {
        if self.loo {
            self.n
        } else {
            self.folds.min(self.n)
        }
    }

    /// How many sizes of model are compared.
    fn sizes(&self) -> usize {
        match self.mode {
            Mode::Degree => P,
            Mode::Penalty => PENALTIES.len(),
            Mode::Useless => LINES,
        }
    }

    /// The same, with what does not matter in its mode set aside: what a count over new
    /// data sets is kept for.
    fn key(&self) -> Setup {
        let mut s = *self;
        if s.mode != Mode::Penalty {
            s.degree = 0;
        }
        if s.mode == Mode::Useless {
            (s.shape, s.folds, s.loo) = (Shape::Wave, 0, false);
        } else {
            (s.folds, s.loo) = (s.folds_used(), s.folds_used() == s.n);
        }
        s
    }
}

/// The fit of model size `size` to the points in `tri`: least squares of that degree,
/// or of the setup's degree with that penalty. In the penalty, `tri.rows` makes it a
/// penalty on the mean squared miss, whatever the number of points:
/// (1/m) Σ (yᵢ − ŷᵢ)² + λ (c₁² + … + c_d²).
fn fit(tri: &Tri, s: &Setup, size: usize) -> Option<[f64; P]> {
    fit_penalised(tri, s, size, tri.rows)
}

/// The same with the penalty set for `m` points (leave one out: the fits of n − 1 points).
fn fit_penalised(tri: &Tri, s: &Setup, size: usize, m: usize) -> Option<[f64; P]> {
    match s.mode {
        Mode::Degree => (size < tri.rows).then(|| tri.solve(size)).flatten(),
        Mode::Penalty => {
            let t = penalised(tri, s, size, m)?;
            t.solve(s.degree.min(MAX_DEG))
        }
        Mode::Useless => None,
    }
}

/// The triangle with the ridge penalty's rows added.
fn penalised(tri: &Tri, s: &Setup, size: usize, m: usize) -> Option<Tri> {
    let lambda = *PENALTIES.get(size)?;
    let mut t = tri.clone();
    let strength = (m as f64 * lambda).sqrt();
    for k in 1..=s.degree.min(MAX_DEG) {
        t.add_penalty(k, strength);
    }
    Some(t)
}

/// The degree of the curves of a setup at this size.
fn degree_of(s: &Setup, size: usize) -> usize {
    match s.mode {
        Mode::Degree => size.min(MAX_DEG),
        _ => s.degree.min(MAX_DEG),
    }
}

/// Every size of model on one data set: the error on the points fitted to, the error
/// on points held back (cross-validation) with its standard error, and the true error
/// on new data, which is known here because the truth is.
#[derive(Clone, Debug, Default)]
struct Sweep {
    /// Mean squared miss on the points fitted to (NaN where no unique fit).
    train: Vec<f64>,
    /// Cross-validated error: every point's squared miss when its own fold was held
    /// back, averaged over all the points (NaN where some fold has no unique fit).
    cv: Vec<f64>,
    /// Its standard error: the spread of the folds' own means over √k.
    se: Vec<f64>,
    /// The true error of the fit to all the points: σ² plus its mean squared distance
    /// from the truth.
    truth: Vec<f64>,
    /// Each fold's mean squared miss, by size (leave one out: each point's).
    folds: Vec<Vec<f64>>,
    /// The fit to all the points, by size.
    fits: Vec<Option<[f64; P]>>,
    /// The fold of each point.
    fold_of: Vec<usize>,
    /// The number of folds.
    k: usize,
}

impl Sweep {
    fn new(data: &Data, s: &Setup, truth: &Truth, deal_seed: u64) -> Sweep {
        let n = data.len();
        let sizes = s.sizes();
        let k = s.folds_used().max(1);
        let fold_of = deal(n, k, deal_seed);
        let all = Tri::of(data, |_| true);
        let noise2 = s.noise * s.noise;
        let mut sw = Sweep {
            train: vec![f64::NAN; sizes],
            cv: vec![f64::NAN; sizes],
            se: vec![f64::NAN; sizes],
            truth: vec![f64::NAN; sizes],
            folds: vec![Vec::new(); sizes],
            fits: vec![None; sizes],
            fold_of,
            k,
        };
        if s.mode == Mode::Useless || n == 0 {
            return sw;
        }
        for size in 0..sizes {
            let Some(c) = fit(&all, s, size) else { continue };
            sw.fits[size] = Some(c);
            sw.train[size] = (0..n).map(|i| (data.y[i] - value(&c, &data.rows[i])).powi(2)).sum::<f64>() / n as f64;
            sw.truth[size] = noise2 + truth.distance(&c, degree_of(s, size));
        }
        if n < 2 {
            return sw;
        }
        if k == n {
            sw.leave_one_out(data, s, &all);
        } else {
            sw.k_fold(data, s);
        }
        sw.exact_zeros(data);
        sw
    }

    /// An error that is only rounding (a curve through every point, or a truth with no
    /// noise that a curve has exactly) is 0: left as 1e-31 it would choose among equal
    /// fits by the last bit, and draw a cliff of thirty powers of ten. (Below 10⁻¹³ of
    /// the data's mean square, an error is not worked out to a single digit anyway.)
    fn exact_zeros(&mut self, data: &Data) {
        let tiny = 1e-13 * (crate::math::stats::mean(&data.y.iter().map(|y| y * y).collect::<Vec<f64>>()) + 1e-300);
        for v in self.train.iter_mut().chain(&mut self.cv).chain(&mut self.se).chain(&mut self.truth).chain(self.folds.iter_mut().flatten()) {
            if v.abs() < tiny {
                *v = 0.0;
            }
        }
    }

    /// Each fold held back in turn, and the curve fitted to the rest afresh.
    fn k_fold(&mut self, data: &Data, s: &Setup) {
        let (n, k) = (data.len(), self.k);
        let mut misses = vec![vec![f64::NAN; k]; s.sizes()];
        for f in 0..k {
            let tri = Tri::of(data, |i| self.fold_of[i] != f);
            let held: Vec<usize> = (0..n).filter(|&i| self.fold_of[i] == f).collect();
            for (size, m) in misses.iter_mut().enumerate() {
                if let Some(c) = fit(&tri, s, size) {
                    m[f] = held.iter().map(|&i| (data.y[i] - value(&c, &data.rows[i])).powi(2)).sum::<f64>() / held.len().max(1) as f64;
                }
            }
        }
        let sizes: Vec<f64> = (0..k).map(|f| self.fold_of.iter().filter(|&&g| g == f).count() as f64).collect();
        for (size, m) in misses.into_iter().enumerate() {
            if m.iter().all(|v| v.is_finite()) {
                self.cv[size] = m.iter().zip(&sizes).map(|(v, w)| v * w).sum::<f64>() / n as f64;
                self.se[size] = crate::math::stats::sd(&m) / (k as f64).sqrt();
            }
            self.folds[size] = m;
        }
    }

    /// Each point held back alone. Its miss when left out is its miss in the fit to all
    /// the points divided by one less its leverage, exactly, for least squares and for a
    /// ridge penalty alike (the same penalty, set for n − 1 points). Where the leverage
    /// is within a hair of 1 that division loses its digits, and the point is left out
    /// and the curve fitted again instead.
    fn leave_one_out(&mut self, data: &Data, s: &Setup, all: &Tri) {
        let n = data.len();
        // (by degree, every point's leverage in every degree's fit at once: they nest)
        let nested: Vec<[f64; P]> = if s.mode == Mode::Degree { data.rows.iter().map(|r| all.leverage(r)).collect() } else { Vec::new() };
        for size in 0..s.sizes() {
            let d = degree_of(s, size);
            if s.mode == Mode::Degree && d + 2 > n {
                continue;
            }
            let tri = match s.mode {
                Mode::Penalty => match penalised(all, s, size, n - 1) {
                    Some(t) => t,
                    None => continue,
                },
                _ => all.clone(),
            };
            let Some(c) = tri.solve(d) else { continue };
            let mut miss = Vec::with_capacity(n);
            for i in 0..n {
                let h = match nested.get(i) {
                    Some(l) => l[d],
                    None => tri.leverage(&data.rows[i])[d],
                };
                let e = data.y[i] - value(&c, &data.rows[i]);
                let left = if 1.0 - h > 1e-7 {
                    Some(e / (1.0 - h))
                } else {
                    let without = Tri::of(data, |j| j != i);
                    fit_penalised(&without, s, size, n - 1).map(|c| data.y[i] - value(&c, &data.rows[i]))
                };
                match left {
                    Some(v) if v.is_finite() => miss.push(v * v),
                    _ => break,
                }
            }
            if miss.len() == n {
                self.cv[size] = crate::math::stats::mean(&miss);
                self.se[size] = crate::math::stats::sd(&miss) / (n as f64).sqrt();
                self.folds[size] = miss;
            }
        }
    }

    /// The size with the lowest cross-validated error (the smaller, of two alike).
    fn lowest(&self) -> Option<usize> {
        argmin(&self.cv)
    }

    /// The size whose true error is lowest.
    fn best(&self) -> Option<usize> {
        argmin(&self.truth)
    }
}

/// Where the smallest number is, of those that are numbers (the first, of two alike).
fn argmin(v: &[f64]) -> Option<usize> {
    v.iter().enumerate().filter(|(_, x)| x.is_finite()).fold(None, |best: Option<(usize, f64)>, (i, &x)| match best {
        Some((_, b)) if b <= x => best,
        _ => Some((i, x)),
    })
    .map(|(i, _)| i)
}

// ----- the contest of useless models --------------------------------------------------------

/// Twenty straight lines, each fitted to y on its own input z, where every z and y is
/// independent noise: no line can predict anything. Each is judged on the same points
/// held back, and the one that misses them least is the winner.
#[derive(Clone, Debug, Default)]
struct Contest {
    /// The points the lines are fitted to: each row's twenty inputs, and y.
    z_fit: Vec<[f64; LINES]>,
    y_fit: Vec<f64>,
    /// The points held back to judge them.
    z_held: Vec<[f64; LINES]>,
    y_held: Vec<f64>,
    /// Each line's intercept and slope.
    line: Vec<(f64, f64)>,
    train: Vec<f64>,
    held: Vec<f64>,
    /// Each line's true error on new points: σ² + a² + b² for the line a + b z, since
    /// y has mean 0 and variance σ², z has mean 0 and variance 1, and the two are
    /// independent.
    truth: Vec<f64>,
    winner: usize,
}

fn contest(seed: u64, n: usize, noise: f64) -> Contest {
    let mut rng = Rng::new(seed);
    let rows = |count: usize, rng: &mut Rng| -> (Vec<[f64; LINES]>, Vec<f64>) {
        let mut z = Vec::with_capacity(count);
        let mut y = Vec::with_capacity(count);
        for _ in 0..count {
            let mut r = [0.0; LINES];
            for v in r.iter_mut() {
                *v = normal(rng);
            }
            z.push(r);
            y.push(noise * normal(rng));
        }
        (z, y)
    };
    let (z_fit, y_fit) = rows(n, &mut rng);
    let (z_held, y_held) = rows(n, &mut rng);
    let mut c = Contest { z_fit, y_fit, z_held, y_held, ..Default::default() };
    let miss = |z: &[[f64; LINES]], y: &[f64], j: usize, (a, b): (f64, f64)| {
        z.iter().zip(y).map(|(r, &y)| (y - a - b * r[j]).powi(2)).sum::<f64>() / y.len().max(1) as f64
    };
    for j in 0..LINES {
        let zs: Vec<f64> = c.z_fit.iter().map(|r| r[j]).collect();
        let (mz, my) = (crate::math::stats::mean(&zs), crate::math::stats::mean(&c.y_fit));
        let szz: f64 = zs.iter().map(|z| (z - mz).powi(2)).sum();
        let szy: f64 = zs.iter().zip(&c.y_fit).map(|(z, y)| (z - mz) * (y - my)).sum();
        let b = if szz > 0.0 { szy / szz } else { 0.0 };
        let line = (my - b * mz, b);
        c.line.push(line);
        c.train.push(miss(&c.z_fit, &c.y_fit, j, line));
        c.held.push(miss(&c.z_held, &c.y_held, j, line));
        c.truth.push(noise * noise + line.0 * line.0 + line.1 * line.1);
    }
    c.winner = argmin(&c.held).unwrap_or(0);
    c
}

// ----- many new data sets ---------------------------------------------------------------------

/// How many new data sets (or contests) are counted.
const RUNS: usize = 200;

/// What cross-validation chooses on many new data sets from the same truth (or, in the
/// contest, how the winner's held-back error compares with its true error).
#[derive(Clone, Debug, Default)]
struct Tally {
    key: Option<Setup>,
    done: usize,
    /// How often each size was chosen.
    chosen: Vec<u32>,
    /// The same with five folds, on the same data sets (kept when the setup is leave one out).
    five: Vec<u32>,
    /// The true error of each size, summed over the data sets, and on how many it could be fitted.
    errors: Vec<f64>,
    fitted: Vec<u32>,
    /// In the contest: the winner's true error less its held-back error, and the same for
    /// a line picked at random.
    gaps: Vec<f64>,
    random: Vec<f64>,
}

/// The seed of the `k`-th new data set: the same for every setup, so that two setups are
/// compared on the same draws.
fn seed_of(k: usize) -> u64 {
    0x5EED_0000_0000 + 7919 * k as u64
}

impl Tally {
    fn restart(&mut self, s: &Setup) {
        let sizes = s.sizes();
        *self = Tally {
            key: Some(s.key()),
            chosen: vec![0; sizes],
            five: vec![0; sizes],
            errors: vec![0.0; sizes],
            fitted: vec![0; sizes],
            ..Default::default()
        };
    }

    fn pending(&self) -> bool {
        self.key.is_some() && self.done < RUNS
    }

    fn step(&mut self, s: &Setup, truth: &Truth) {
        let seed = seed_of(self.done);
        if s.mode == Mode::Useless {
            let c = contest(seed, s.n, s.noise);
            let w = c.winner;
            self.gaps.push(c.truth[w] - c.held[w]);
            let r = Rng::new(seed ^ 0xA5A5).below(LINES);
            self.random.push(c.truth[r] - c.held[r]);
        } else {
            let data = Data::draw(seed, s.n, s.shape, s.noise);
            let sweep = Sweep::new(&data, s, truth, seed.wrapping_add(1));
            if let Some(i) = sweep.lowest() {
                self.chosen[i] += 1;
            }
            for (i, &t) in sweep.truth.iter().enumerate() {
                if t.is_finite() {
                    self.errors[i] += t;
                    self.fitted[i] += 1;
                }
            }
            if s.folds_used() == s.n && s.n > 5 {
                let five = Setup { folds: 5, loo: false, ..*s };
                if let Some(i) = Sweep::new(&data, &five, truth, seed.wrapping_add(1)).lowest() {
                    self.five[i] += 1;
                }
            }
        }
        self.done += 1;
    }

    /// The size whose true error is lowest on average over the data sets counted (of
    /// those that could be fitted on all of them).
    fn best(&self) -> Option<usize> {
        let mean: Vec<f64> = self.errors.iter().zip(&self.fitted).map(|(e, &k)| if k as usize == self.done && k > 0 { e / k as f64 } else { f64::NAN }).collect();
        argmin(&mean)
    }
}

// ----- the experiments ---------------------------------------------------------------------------

/// One of the "things to try": everything it sets.
struct Trial {
    label: &'static str,
    caption: &'static str,
    setup: Setup,
    /// The size marked on the error chart (None: where the held-back error is lowest).
    at: Option<usize>,
    /// Each fold held back in turn on the left, rather than the fit to all the data.
    turns: bool,
}

/// Thirty points from the wave, five folds: where the scene starts.
const WAVE: Setup = Setup { shape: Shape::Wave, noise: 0.3, n: 30, folds: 5, loo: false, mode: Mode::Degree, degree: 12 };

const TRIALS: [Trial; 6] = [
    Trial {
        label: "too stiff, too wild",
        caption: "Degree 12 bends to follow the points and swings between them, off the chart near the left end. On the points it was fitted to, its error is 0.061, against 0.37 for a straight line (degree 1): six times better. On points held back (five folds) it is about 110 000, against 0.51: far worse. Drag the white dot under the error chart to degree 1 and back.",
        setup: WAVE,
        at: Some(12),
        turns: false,
    },
    Trial {
        label: "five folds",
        caption: "The 30 points are dealt into five folds of 6. Each fold in turn is held back (orange), the curve is fitted to the other 24, and its misses on the 6 are measured (pink). Together they make the orange curve on the error chart: lowest at degree 4, which is also truly best here (green). Press “deal again”: the orange curve moves a little, and over the first twenty deals its lowest point is at degree 4 in 14 and at degree 3 in 6. The choice is itself an estimate.",
        setup: WAVE,
        at: None,
        turns: true,
    },
    Trial {
        label: "a penalty instead",
        caption: r"Degree 12 again, now with a ridge penalty that pulls its coefficients towards zero, and the penalty $\lambda$ across the error chart. At the left it is too small to matter, and the curve is as wild as before. Drag the white dot to the right: the curve calms down, and at the far right it is nearly flat. The held-back error is high at both ends and lowest in between, at $\lambda = 0.02$; the true error is lowest at $\lambda = 0.1$.",
        setup: Setup { mode: Mode::Penalty, ..WAVE },
        at: Some(0),
        turns: false,
    },
    Trial {
        label: "more data, a bigger model",
        caption: "300 points from a smooth step. On 200 new data sets of this size, cross-validation chooses degree 8 most often, and degree 10 is best on average. Set the points to 30: then it chooses degree 3 most often, and degree 3 is best on average. More points pin down more coefficients, so a bigger model pays.",
        setup: Setup { shape: Shape::Step, n: 300, ..WAVE },
        at: None,
        turns: false,
    },
    Trial {
        label: "the winner's curse",
        caption: "Twenty straight lines, each fitted to $y$ on an input of pure noise: none can predict anything. All are judged on the same 30 points held back, and the one that misses them least wins (ringed). Its held-back error, 0.61, is the lowest of the twenty; on new data it is ordinary, its true error of 1.09 among the others' (1.07 to 1.30). Over 200 new contests the winner's true error is above its held-back error by 0.12 on average; for a line picked at random the two differ by nothing beyond chance (−0.03 here). An error used to choose flatters what it chose: a final test needs points that played no part in any choice.",
        setup: Setup { mode: Mode::Useless, noise: 1.0, ..WAVE },
        at: None,
        turns: false,
    },
    Trial {
        label: "leave one out",
        caption: "As many folds as points: each point is held back alone, and the curve fitted to the other 29, nearly all of them. So the estimate is nearly unbiased: on 200 new data sets, at degree 4, it averages 0.116, as the true error does; five folds, each fit made from 24 points, average 0.139. But its choices are more variable: they spread wider than five folds' on the same data sets (outlined), with a standard deviation of 2.1 degrees against 1.6, and degree 11 or more in 8 data sets against 2.",
        setup: Setup { loo: true, ..WAVE },
        at: None,
        turns: true,
    },
];

/// The scene opens as this experiment sets it up.
const OPENING: usize = 1;
/// The data set every experiment shows, and the first deal of its folds.
const FIRST_SEED: u64 = 2;
const FIRST_DEAL: u64 = 1;

/// New data sets counted a second.
const TALLY_RATE: f32 = 80.0;

// ----- the scene -----------------------------------------------------------------------------------

pub struct ModelSelect {
    setup: Setup,
    seed: u64,
    deal: u64,
    /// The size marked on the error chart: a degree (as wished: the one drawn is never
    /// more than the points allow) and a penalty's place in `PENALTIES`.
    at: [usize; 2],
    /// The line marked in the contest (None: the winner).
    line: Option<usize>,
    show_truth: bool,
    /// Each fold held back in turn on the left.
    turns: bool,
    truths: [Truth; 4],
    data: Data,
    sweep: Sweep,
    contest: Contest,
    /// What `data`, `sweep` and `contest` were worked out for.
    made: Option<(Setup, u64, u64)>,
    tally: Tally,
    pace: Pace,
    /// The fold held back now, and for how long it has been (seconds of the scene's time).
    fold: usize,
    fold_time: f32,
    /// The curve fitted without that fold, and what it was fitted for.
    without: Option<(WithoutKey, Option<[f64; P]>)>,
    /// The white dot while the hand holds it: where it is along the error chart.
    held: Option<f64>,
    look: Look,
    /// The canvas and the mode last drawn (a change is drawn at once, without gliding).
    canvas: Option<(Vec2, Mode)>,
    /// The Legendre values along the left chart, where its curves are drawn.
    grid: Vec<(f64, [f64; P])>,
}

type WithoutKey = (usize, usize, Setup, u64, u64);

/// Points along the left chart's curves.
const CURVE: usize = 161;

impl ModelSelect {
    /// (Nothing here needs the graphics device, so the tests can make one too.)
    fn create() -> ModelSelect {
        let mut s = ModelSelect {
            setup: WAVE,
            seed: FIRST_SEED,
            deal: FIRST_DEAL,
            at: [4, 0],
            line: None,
            show_truth: true,
            turns: true,
            truths: SHAPES.map(|(shape, _)| Truth::of(shape)),
            data: Data::default(),
            sweep: Sweep::default(),
            contest: Contest::default(),
            made: None,
            tally: Tally::default(),
            pace: Pace::default(),
            fold: 0,
            fold_time: 0.0,
            without: None,
            held: None,
            look: Look::default(),
            canvas: None,
            grid: (0..CURVE).map(|i| -1.0 + 2.0 * i as f64 / (CURVE - 1) as f64).map(|x| (x, legendre(x))).collect(),
        };
        s.set_up(OPENING);
        s
    }

    /// Set up one of the experiments in `TRIALS`.
    fn set_up(&mut self, k: usize) {
        let t = &TRIALS[k];
        self.setup = t.setup;
        (self.seed, self.deal, self.show_truth, self.turns, self.line, self.held) = (FIRST_SEED, FIRST_DEAL, true, t.turns, None, None);
        (self.fold, self.fold_time) = (0, 0.0);
        self.refresh();
        match (self.setup.mode, t.at) {
            (Mode::Degree, Some(d)) => self.at[0] = d,
            (Mode::Degree, None) => self.at[0] = self.sweep.lowest().unwrap_or(4),
            (Mode::Penalty, Some(i)) => self.at[1] = i,
            (Mode::Penalty, None) => self.at[1] = self.sweep.lowest().unwrap_or(0),
            (Mode::Useless, at) => self.line = at,
        }
    }

    fn truth(&self) -> &Truth {
        &self.truths[SHAPES.iter().position(|(s, _)| *s == self.setup.shape).unwrap_or(0)]
    }

    /// Work out again whatever the settings have changed, and start the count over new
    /// data sets afresh when what it counts has changed.
    fn refresh(&mut self) {
        let s = &mut self.setup;
        s.n = s.n.clamp(6, 300);
        s.folds = s.folds.clamp(2, 10);
        s.noise = if s.noise.is_finite() { s.noise.max(0.0).min(1.0) } else { 0.3 };
        s.degree = s.degree.clamp(1, MAX_DEG);
        self.at = [self.at[0].min(MAX_DEG), self.at[1].min(PENALTIES.len() - 1)];
        self.line = self.line.filter(|&j| j < LINES);
        let key = (self.setup, self.seed, self.deal);
        if self.made != Some(key) {
            let s = self.setup;
            if s.mode == Mode::Useless {
                self.contest = contest(self.seed, s.n, s.noise);
            } else {
                // (the same points, moved, when only the truth or the noise has changed)
                let same = self.made.is_some_and(|(o, seed, _)| seed == self.seed && o.n == s.n) && self.data.len() == s.n;
                if same {
                    self.data.set_y(s.shape, s.noise);
                } else {
                    self.data = Data::draw(self.seed, s.n, s.shape, s.noise);
                }
                self.sweep = Sweep::new(&self.data, &s, self.truth(), self.deal);
            }
            self.fold %= self.sweep.k.max(1);
            self.made = Some(key);
        }
        if self.tally.key != Some(self.setup.key()) {
            self.tally.restart(&self.setup);
            self.pace.reset();
        }
    }

    /// The highest degree with a unique fit to all the points.
    fn most_degree(&self) -> usize {
        (0..P).rev().find(|&d| self.sweep.fits.get(d).is_some_and(Option::is_some)).unwrap_or(0)
    }

    /// The size marked: the degree drawn, the penalty's place, the line.
    fn marked(&self) -> usize {
        match self.setup.mode {
            Mode::Degree => self.at[0].min(self.most_degree()),
            Mode::Penalty => self.at[1],
            Mode::Useless => self.line.unwrap_or(self.contest.winner),
        }
    }

    /// Is each fold shown held back in turn?
    fn turning(&self) -> bool {
        self.turns && self.setup.mode != Mode::Useless && self.sweep.k >= 2
    }

    /// How long each fold is held back for: long enough to look at, and a round of them
    /// in a few seconds when they are many.
    fn fold_period(&self) -> f32 {
        (6.0 / self.sweep.k.max(1) as f32).clamp(0.3, 1.2)
    }

    /// The curve drawn on the left: the fit to all the points at the size marked, or
    /// (folds in turn) the fit without the fold held back.
    fn shown_fit(&mut self) -> Option<[f64; P]> {
        let size = self.marked();
        if !self.turning() {
            return self.sweep.fits.get(size).copied().flatten();
        }
        let key = (self.fold, size, self.setup, self.seed, self.deal);
        if let Some((k, c)) = &self.without {
            if *k == key {
                return *c;
            }
        }
        let fold_of = &self.sweep.fold_of;
        let tri = Tri::of(&self.data, |i| fold_of.get(i) != Some(&key.0));
        let c = fit(&tri, &self.setup, size);
        self.without = Some((key, c));
        c
    }

    /// The x of a size on the error chart: the degree, the penalty's power of ten, the
    /// line's number.
    fn size_x(&self, i: usize) -> f64 {
        size_x(self.setup.mode, i)
    }

    /// The size nearest a place along the error chart, of those that can be marked.
    fn nearest(&self, x: f64) -> usize {
        match self.setup.mode {
            Mode::Degree => (x.round().max(0.0) as usize).min(self.most_degree()),
            Mode::Penalty => (0..PENALTIES.len()).min_by(|&a, &b| (size_x(Mode::Penalty, a) - x).abs().total_cmp(&(size_x(Mode::Penalty, b) - x).abs())).unwrap_or(0),
            Mode::Useless => ((x.round() - 1.0).max(0.0) as usize).min(LINES - 1),
        }
    }

    /// The error chart's numbers by size: on the points fitted to, held back (and one
    /// standard error below and above), and the truth.
    fn series(&self) -> [Vec<f64>; 5] {
        if self.setup.mode == Mode::Useless {
            let c = &self.contest;
            let none = vec![f64::NAN; c.held.len()];
            return [c.train.clone(), c.held.clone(), none.clone(), none, c.truth.clone()];
        }
        let sw = &self.sweep;
        let lower: Vec<f64> = sw.cv.iter().zip(&sw.se).map(|(c, s)| c - s).collect();
        let upper: Vec<f64> = sw.cv.iter().zip(&sw.se).map(|(c, s)| c + s).collect();
        [sw.train.clone(), sw.cv.clone(), lower, upper, sw.truth.clone()]
    }

    /// The range of the error chart (powers of ten): from a little below the lowest
    /// error to well above the errors of the stiffest model, so that the region where the
    /// choice is made fills it; errors far larger go off its top.
    fn error_range(&self) -> [f64; 2] {
        let [train, cv, _, _, truth] = self.series();
        let noise2 = self.setup.noise * self.setup.noise;
        let finite = |v: &[f64]| v.iter().copied().filter(|x| x.is_finite() && *x > 0.0).collect::<Vec<f64>>();
        let all: Vec<f64> = [finite(&train), finite(&cv), finite(&truth)].concat();
        let low = all.iter().copied().fold(f64::INFINITY, f64::min);
        let (lo, hi) = if self.setup.mode == Mode::Useless {
            let high = all.iter().copied().fold(noise2, f64::max);
            (lg(low.min(noise2)) - 0.06, lg(high) + 0.06)
        } else {
            let stiff = if self.setup.mode == Mode::Degree { 0 } else { PENALTIES.len() - 1 };
            let top = [train.get(stiff), cv.get(stiff), truth.get(stiff)].into_iter().flatten().copied().filter(|v| v.is_finite()).fold(noise2, f64::max);
            // (no more than four powers of ten below the stiffest model's errors)
            (lg(low.min(top)).max(lg(top) - 4.0) - 0.25, lg(top) + 0.5)
        };
        let (lo, hi) = if lo.is_finite() && hi.is_finite() { (lo, hi) } else { (-2.0, 1.0) };
        let least = if self.setup.mode == Mode::Useless { 0.3 } else { 1.2 };
        let short = (least - (hi - lo)).max(0.0) * 0.5;
        [lo - short, hi + short]
    }

    /// What the picture glides to.
    fn goals(&mut self) -> Goals {
        let mode = self.setup.mode;
        let size = self.marked();
        let mut g = Goals::default();
        if mode == Mode::Useless {
            let c = &self.contest;
            let j = size;
            for (r, y) in c.z_fit.iter().zip(&c.y_fit).chain(c.z_held.iter().zip(&c.y_held)) {
                g.points.extend([r[j] as f32, *y as f32]);
            }
            let (a, b) = c.line.get(j).copied().unwrap_or((0.0, 0.0));
            g.curve = vec![a as f32, b as f32];
            g.curve_on = true;
            let ys: Vec<f64> = c.y_fit.iter().chain(&c.y_held).copied().collect();
            g.left = padded(&ys, 0.12);
        } else {
            for (x, y) in self.data.x.iter().zip(&self.data.y) {
                g.points.extend([*x as f32, *y as f32]);
            }
            let fit = self.shown_fit();
            g.curve_on = fit.is_some();
            g.curve = fit.unwrap_or([0.0; P]).map(|v| v.max(-1e12).min(1e12) as f32).to_vec();
            let shape = self.setup.shape;
            let ys: Vec<f64> = self.data.y.iter().copied().chain(self.grid.iter().step_by(8).map(|(x, _)| shape.at(*x))).collect();
            g.left = padded(&ys, 0.08);
        }
        let series = self.series();
        for v in &series {
            // (a size with no number keeps a place in the glide, at its neighbour's)
            let mut last = f64::NAN;
            for &x in v {
                if x.is_finite() {
                    last = x;
                }
                g.shown.push(x.is_finite());
                g.series.push(lg(if x.is_finite() { x } else { last }) as f32);
            }
        }
        let fill = g.series.iter().copied().find(|v| v.is_finite()).unwrap_or(0.0);
        for v in g.series.iter_mut().filter(|v| !v.is_finite()) {
            *v = fill;
        }
        g.folds = self.sweep.folds.get(size).filter(|_| mode != Mode::Useless).map(|m| m.iter().map(|&v| lg(v) as f32).collect()).unwrap_or_default();
        g.right = self.error_range().map(|v| v as f32);
        // errors far off the chart glide from just beyond its edge, not from where they
        // are (10⁵, say): then everything on the chart moves together as it changes
        let span = g.right[1] - g.right[0];
        let (lo, hi) = (g.right[0] - 0.1 * span, g.right[1] + 0.1 * span);
        for v in g.series.iter_mut().chain(g.folds.iter_mut()) {
            *v = v.max(lo).min(hi);
        }
        g.mark = self.held.unwrap_or_else(|| self.size_x(size)) as f32;
        let t = &self.tally;
        if mode == Mode::Useless {
            let width = gap_range(&self.setup);
            g.bars = [histogram(&t.gaps, width), histogram(&t.random, width)].concat();
        } else {
            g.bars = t.chosen.iter().chain(&t.five).map(|&c| c as f32).collect();
        }
        let most = g.bars.iter().copied().fold(0.0f32, f32::max);
        g.top = TOPS.into_iter().find(|&t| t >= most * 1.08).unwrap_or(TOPS[TOPS.len() - 1]);
        let which = match mode {
            Mode::Useless => [(Some(self.contest.winner), 1), (None, 4)],
            _ => [(self.sweep.lowest(), 1), (self.sweep.best(), 4)],
        };
        let n_sizes = self.setup.sizes();
        for (r, (i, k)) in which.into_iter().enumerate() {
            let i = i.filter(|&i| i < n_sizes && g.shown.get(k * n_sizes + i) == Some(&true));
            g.rings_on[r] = i.is_some();
            g.rings.extend(i.map_or([g.mark, g.right[0]], |i| [self.size_x(i) as f32, g.series[k * n_sizes + i]]));
        }
        if mode == Mode::Useless {
            let w = gap_range(&self.setup);
            let mean = |v: &[f64]| if v.is_empty() { 0.0 } else { (v.iter().sum::<f64>() / v.len() as f64).max(-w).min(w) as f32 };
            g.lines = vec![mean(&t.gaps), mean(&t.random)];
            let fitted = self.contest.y_fit.len();
            g.held = (0..g.points.len() / 2).map(|i| if i < fitted { 0.0 } else { 1.0 }).collect();
        } else {
            g.lines = vec![t.best().map_or(g.mark, |i| self.size_x(i) as f32)];
            let turning = self.turning();
            g.held = self.sweep.fold_of.iter().map(|&q| if turning && q == self.fold { 1.0 } else { 0.0 }).collect();
        }
        g
    }
}

/// log₁₀ of an error, with a floor: an error of exactly 0 (a curve through every point)
/// is drawn at the chart's foot.
fn lg(v: f64) -> f64 {
    v.max(1e-12).log10()
}

fn size_x(mode: Mode, i: usize) -> f64 {
    match mode {
        Mode::Degree => i as f64,
        Mode::Penalty => PENALTIES.get(i).copied().unwrap_or(1.0).log10(),
        Mode::Useless => (i + 1) as f64,
    }
}

/// From a little below the least of the numbers to a little above the greatest.
fn padded(v: &[f64], share: f64) -> [f32; 2] {
    let (lo, hi) = v.iter().filter(|x| x.is_finite()).fold((f64::INFINITY, f64::NEG_INFINITY), |(a, b), &x| (a.min(x), b.max(x)));
    if !(lo.is_finite() && hi.is_finite()) {
        return [-1.0, 1.0];
    }
    let pad = ((hi - lo) * share).max(0.1);
    [(lo - pad) as f32, (hi + pad) as f32]
}

/// The bins of the contest's histograms: true error less held-back error, from −w to w
/// in `GAP_BINS` bins.
const GAP_BINS: usize = 24;

/// How far either way the contest's histograms reach: three times the spread of one
/// line's held-back error (σ² √(2/n), near enough), so that nearly every contest falls
/// inside; the few beyond go in the end bins.
fn gap_range(s: &Setup) -> f64 {
    (3.0 * s.noise * s.noise * (2.0 / s.n.max(1) as f64).sqrt()).max(1e-9)
}

fn histogram(v: &[f64], width: f64) -> Vec<f32> {
    let mut counts = vec![0.0f32; GAP_BINS];
    for &x in v.iter().filter(|x| x.is_finite()) {
        let at = ((x + width) / (2.0 * width) * GAP_BINS as f64).floor();
        counts[(at.max(0.0) as usize).min(GAP_BINS - 1)] += 1.0;
    }
    counts
}

/// A number of an error as a read-out shows it: three decimals, fewer for large ones,
/// and a power of ten for the very large and very small.
fn err_num(v: f64) -> String {
    let a = v.abs();
    if !v.is_finite() {
        "none".to_string()
    } else if v == 0.0 {
        "0".to_string()
    } else if a >= 1e4 || (a < 0.001 && a > 0.0) {
        format!("{v:.1e}")
    } else if a >= 100.0 {
        format!("{v:.0}")
    } else if a >= 10.0 {
        num(v, 1)
    } else {
        num(v, 3)
    }
}

/// A penalty as a formula's number: 0.02, 5, or 2 × 10⁻⁵.
fn penalty_num(v: f64) -> String {
    if v >= 0.01 {
        let s = format!("{v:.2}");
        s.trim_end_matches('0').trim_end_matches('.').to_string()
    } else {
        let s = format!("{v:e}");
        match s.strip_prefix("1e") {
            Some(power) => format!("10^{{{power}}}"),
            None => s,
        }
    }
}

// ----- what glides -------------------------------------------------------------------------------

/// What the picture is drawn from, as it should be now.
#[derive(Clone, Debug, Default)]
struct Goals {
    /// The left chart's points, x and y of each.
    points: Vec<f32>,
    /// The left chart's curve: its Legendre coefficients (in the contest, a line's
    /// intercept and slope), and whether there is one.
    curve: Vec<f32>,
    curve_on: bool,
    /// The left chart's range up.
    left: [f32; 2],
    /// The error chart's five series (on the points fitted to, held back, one standard
    /// error below and above, the truth), log₁₀, one after another, and which are numbers.
    series: Vec<f32>,
    shown: Vec<bool>,
    /// Each fold's mean squared miss at the size marked, log₁₀.
    folds: Vec<f32>,
    /// The error chart's range up (log₁₀) and where the white dot is.
    right: [f32; 2],
    mark: f32,
    /// The count's bars, and the top of its chart.
    bars: Vec<f32>,
    top: f32,
    /// How far each point is held back: 1 if it is (its miss is drawn), 0 if not.
    held: Vec<f32>,
    /// The lines on the count: where the best size on average is, or the two means of
    /// the contest (true error less held-back error, of the winner and of a line at random).
    lines: Vec<f32>,
    /// The rings on the error chart, x and y of each: the lowest held-back error (in the
    /// contest, the winner) and the truly best; and which are drawn.
    rings: Vec<f32>,
    rings_on: [bool; 2],
}

/// The picture as drawn: each part following its goal (see `plot::Glide`).
#[derive(Clone, Debug, Default)]
struct Look {
    points: Glide,
    curve: Glide,
    left: Glide,
    series: Glide,
    folds: Glide,
    right: Glide,
    mark: Glide,
    bars: Glide,
    top: Glide,
    held: Glide,
    lines: Glide,
    rings: Glide,
    moving: bool,
}

impl Look {
    fn follow(&mut self, g: &Goals, dt: f32, snap: bool) {
        let mut moving = false;
        let (mark, top) = ([g.mark], [g.top]);
        let parts: [(&mut Glide, &[f32], bool); 12] = [
            (&mut self.points, &g.points, true),
            (&mut self.curve, &g.curve, false),
            (&mut self.left, &g.left, false),
            (&mut self.series, &g.series, false),
            (&mut self.folds, &g.folds, true),
            (&mut self.right, &g.right, false),
            (&mut self.mark, &mark, false),
            (&mut self.bars, &g.bars, true),
            (&mut self.top, &top, false),
            (&mut self.held, &g.held, true),
            (&mut self.lines, &g.lines, false),
            (&mut self.rings, &g.rings, false),
        ];
        for (glide, to, keep) in parts {
            if snap {
                glide.snap(to);
            } else {
                moving |= glide.follow(to, dt, keep);
            }
        }
        self.moving = moving;
    }
}

// ----- the layout ----------------------------------------------------------------------------------

type Rect = plot::Rect;
const NO_RECT: Rect = (Vec2::ZERO, Vec2::ZERO);

/// Where everything goes on a canvas.
#[derive(Clone, Debug)]
struct Layout {
    narrow: bool,
    /// Text sizes: headings, words, small print.
    big: f32,
    text: f32,
    small: f32,
    modes: Vec<Rect>,
    again: Rect,
    deal: Rect,
    shows: Vec<Rect>,
    /// Each chart's heading (top-left), its key's top, and its rectangle.
    left_head: Vec2,
    left_key: Vec2,
    left: Rect,
    right_head: Vec2,
    right_key: Vec2,
    right: Rect,
    tally_head: Vec2,
    tally_key: Vec2,
    tally: Rect,
    /// The numbers under the left chart (none on a phone).
    words: Rect,
    /// Too small for the charts.
    tiny: bool,
}

/// Room for tick numbers left of a chart, and for them and an axis name under it.
const LEFT_ROOM: f32 = 40.0;
const UNDER: f32 = 34.0;

/// The kinds of line drawn (see `plot::apart`): each a batch whose number of points
/// changes only with what it shows.
const K_TRUTH: u8 = plot::OWN;
const K_FIT: u8 = plot::OWN + 1;
const K_MISS: u8 = plot::OWN + 2;
const K_SERIES: u8 = plot::OWN + 3;
const K_MARK: u8 = plot::OWN + 4;
const K_KEY: u8 = plot::OWN + 5;
const K_TALLY: u8 = plot::OWN + 6;
const K_GRID: u8 = plot::OWN + 7;

/// A swatch in a key.
#[derive(Clone, Copy, Debug, PartialEq)]
enum Swatch {
    Line,
    Dashed,
    Dot,
    Ring,
    Bar,
    Outline,
}

type Entry = (Swatch, Color, &'static str);

/// The entries of a key laid out in lines no wider than `room`: which entries go on each.
fn key_lines(f: &Frame, entries: &[Entry], room: f32, size: f32) -> Vec<Vec<usize>> {
    let mut lines: Vec<Vec<usize>> = Vec::new();
    let mut used = 0.0;
    for (i, e) in entries.iter().enumerate() {
        let w = 22.0 + f.text_size(e.2, size).x;
        match lines.last_mut() {
            Some(line) if used + 16.0 + w <= room => {
                line.push(i);
                used += 16.0 + w;
            }
            _ => {
                lines.push(vec![i]);
                used = w;
            }
        }
    }
    lines
}

fn key_height(f: &Frame, entries: &[Entry], room: f32, size: f32) -> f32 {
    key_lines(f, entries, room, size).len() as f32 * (size + 6.0)
}

fn draw_key(f: &mut Frame, at: Vec2, room: f32, entries: &[Entry], size: f32) {
    let lines = key_lines(f, entries, room, size);
    for (row, line) in lines.iter().enumerate() {
        let mut x = at.x;
        let y = at.y + row as f32 * (size + 6.0) + size * 0.5;
        for &i in line {
            let (swatch, color, text) = entries[i];
            let (a, b) = (Vec2::new(x, y), Vec2::new(x + 16.0, y));
            match swatch {
                Swatch::Line => plot::line_of(f, K_KEY, a, b, 2.2, color),
                Swatch::Dashed => plot::dashed_of(f, K_KEY, a, b, 1.6, color),
                Swatch::Dot => plot::disc(f, (a + b) * 0.5, 7.0, color),
                Swatch::Ring => ring(f, K_KEY, (a + b) * 0.5, 5.0, color),
                Swatch::Bar => plot::fill(f, (Vec2::new(x + 3.0, y - 5.0), Vec2::new(x + 13.0, y + 5.0)), color),
                Swatch::Outline => {
                    let r = [Vec2::new(x + 3.0, y + 5.0), Vec2::new(x + 3.0, y - 5.0), Vec2::new(x + 13.0, y - 5.0), Vec2::new(x + 13.0, y + 5.0)];
                    plot::path_of(f, K_KEY, &r, 1.2, color);
                }
            }
            f.text_px(Vec2::new(x + 22.0, y), text, size, pal::TEXT.a(0.85), Align2::LEFT_CENTER);
            x += 22.0 + f.text_size(text, size).x + 16.0;
        }
    }
}

/// A segment between two points in a chart's units, the part of it inside the chart.
/// Every call draws one segment, of nothing and unseen if none of it is inside, so that
/// a line made of them keeps its number of points in its batch while it goes in and out
/// of the chart (see `plot::apart`).
fn segment_in(f: &mut Frame, kind: u8, p: &Plot, a: (f64, f64), b: (f64, f64), width: f32, color: Color) {
    let (a, b) = (p.at(a.0, a.1), p.at(b.0, b.1));
    if !(a.is_finite() && b.is_finite()) {
        return;
    }
    let (min, max) = p.rect;
    // (Liang and Barsky: the share of the way from a to b at which it enters and leaves)
    let d = b - a;
    let (mut t0, mut t1) = (0.0f32, 1.0f32);
    for (q, r) in [(-d.x, a.x - min.x), (d.x, max.x - a.x), (-d.y, a.y - min.y), (d.y, max.y - a.y)] {
        if q == 0.0 {
            if r < 0.0 {
                t1 = -1.0;
            }
        } else if q < 0.0 {
            t0 = t0.max(r / q);
        } else {
            t1 = t1.min(r / q);
        }
    }
    if t0 <= t1 {
        plot::line_of(f, kind, a + d * t0, a + d * t1, width, color);
    } else {
        let at = a.max(min).min(max);
        plot::line_of(f, kind, at, at, width, Color::CLEAR);
    }
}

/// A dot at (x, y) in a chart's units, unseen at the chart's edge when outside it (so
/// that the dots keep their places in their batch).
fn dot_in(f: &mut Frame, p: &Plot, x: f64, y: f64, size: f32, color: Color) {
    let at = p.at(x, y);
    if at.is_finite() {
        let seen = if plot::within(at, p.rect, 0.5) { 1.0 } else { 0.0 };
        plot::disc(f, at.max(p.rect.0).min(p.rect.1), size, color.a(color.0[3] as f32 / 255.0 * seen));
    }
}

/// A circle on the screen, in the batch of its kind.
fn ring(f: &mut Frame, kind: u8, at: Vec2, r: f32, color: Color) {
    let pts: Vec<Vec2> = (0..=24).map(|i| at + Vec2::from_angle(i as f32 / 24.0 * TAU) * r).collect();
    plot::path_of(f, kind, &pts, 1.6, color);
}

impl ModelSelect {
    fn left_entries(&self) -> Vec<Entry> {
        if self.setup.mode == Mode::Useless {
            return vec![(Swatch::Dot, DATA, "points it was fitted to"), (Swatch::Dot, OTHER, "held back"), (Swatch::Line, ESTIMATE, "the line"), (Swatch::Line, ALARM, "misses")];
        }
        let mut e = vec![(Swatch::Dot, DATA, "data"), (Swatch::Line, TRUTH, "the truth"), (Swatch::Line, ESTIMATE, "fitted curve")];
        if self.turning() {
            e.extend([(Swatch::Dot, OTHER, "held back"), (Swatch::Line, ALARM, "its misses")]);
        }
        e
    }

    fn right_entries(&self) -> Vec<Entry> {
        if self.setup.mode == Mode::Useless {
            return vec![
                (Swatch::Dot, DATA, "on the points fitted to"),
                (Swatch::Dot, OTHER, "held back"),
                (Swatch::Dot, TRUTH, "on new data (true)"),
                (Swatch::Dashed, TRUTH.a(0.7), r"noise $\sigma^2$"),
                (Swatch::Ring, pal::WHITE, "the winner"),
            ];
        }
        vec![
            (Swatch::Line, DATA, "on the points fitted to"),
            (Swatch::Line, OTHER, r"held back, $\pm 1$ s.e."),
            (Swatch::Line, TRUTH, "on new data (true)"),
            (Swatch::Dashed, TRUTH.a(0.7), r"noise $\sigma^2$"),
            (Swatch::Dot, ALARM, "each fold"),
            (Swatch::Ring, OTHER, "lowest held back"),
            (Swatch::Ring, TRUTH, "truly best"),
        ]
    }

    fn tally_entries(&self) -> Vec<Entry> {
        if self.setup.mode == Mode::Useless {
            return vec![(Swatch::Bar, ALARM.a(0.8), "the winner"), (Swatch::Bar, pal::DIM.a(0.6), "a line picked at random")];
        }
        let mut e = vec![(Swatch::Bar, OTHER.a(0.75), "chosen"), (Swatch::Line, TRUTH, "best on average")];
        if self.sweep.k == self.setup.n {
            e.push((Swatch::Outline, pal::WHITE.a(0.8), "five folds, same data sets"));
        }
        e
    }

    fn layout(&self, f: &Frame) -> Layout {
        let view = f.view_size().max(Vec2::splat(1.0));
        let narrow = view.x < 700.0;
        let (big, text, small) = if narrow { (12.5, 11.5, 10.5) } else { (14.0, 12.5, 11.5) };
        let side = if narrow { 10.0 } else { 22.0 };
        let top = f.top_inset.max(0.0) + if narrow { 6.0 } else { 12.0 };
        let h = if narrow { 26.0 } else { 28.0 };
        let mut lay = Layout {
            narrow,
            big,
            text,
            small,
            modes: Vec::new(),
            again: NO_RECT,
            deal: NO_RECT,
            shows: Vec::new(),
            left_head: Vec2::ZERO,
            left_key: Vec2::ZERO,
            left: NO_RECT,
            right_head: Vec2::ZERO,
            right_key: Vec2::ZERO,
            right: NO_RECT,
            tally_head: Vec2::ZERO,
            tally_key: Vec2::ZERO,
            tally: NO_RECT,
            words: NO_RECT,
            tiny: view.x < 200.0 || view.y < 240.0,
        };
        let useless = self.setup.mode == Mode::Useless;
        let loo = self.sweep.k == self.setup.n;
        if narrow {
            lay.modes = plot::even_row(3, side, view.x - side, top, h, 6.0);
        } else {
            lay.modes = plot::button_row(f, &MODE_WORDS, side, top, h, text, 8.0);
            let words: Vec<&str> = if useless || loo { vec!["draw again"] } else { vec!["draw again", "deal again"] };
            let widths: f32 = words.iter().map(|w| f.text_size(w, text).x + 18.0).sum::<f32>() + 8.0 * (words.len() - 1) as f32;
            let row = plot::button_row(f, &words, view.x - side - widths, top, h, text, 8.0);
            lay.again = row[0];
            lay.deal = row.get(1).copied().unwrap_or(NO_RECT);
        }
        let y0 = top + h + if narrow { 10.0 } else { 18.0 };
        // the columns: side by side on a desktop, one under another on a phone
        let gap = 56.0;
        let colw = if narrow { view.x - 2.0 * side } else { ((view.x - 2.0 * side - gap) * 0.5).max(10.0) };
        let (lx, rx) = (side, if narrow { side } else { side + colw + gap });
        let room = colw - 4.0;
        let (le, re, te) = (self.left_entries(), self.right_entries(), self.tally_entries());
        let (lk, rk, tk) = (key_height(f, &le, room, small), key_height(f, &re, room, small), key_height(f, &te, room, small));
        // a heading, the buttons beside it (or under it), the key, the axis name
        let shows_h = if useless { 0.0 } else { 24.0 };
        let left_above = big + 8.0 + if narrow { 0.0 } else { shows_h + 8.0 } + lk + 18.0;
        let right_above = big + 8.0 + rk + 18.0;
        let tally_above = big + 8.0 + tk + 18.0;
        if narrow {
            let fixed = left_above + UNDER + 8.0 + right_above + UNDER + 8.0 + tally_above + UNDER;
            let rest = view.y - y0 - fixed - 6.0;
            let show_tally = rest > 360.0;
            let (hl, hr, ht) = if show_tally { (rest * 0.37, rest * 0.36, rest * 0.27) } else { let r = rest + tally_above + UNDER; (r * 0.5, r * 0.5, 0.0) };
            let mut y = y0;
            lay.left_head = Vec2::new(lx, y);
            if !useless {
                let words: [&str; 2] = ["all data", "folds in turn"];
                let w: f32 = words.iter().map(|w| f.text_size(w, small).x + 16.0).sum::<f32>() + 6.0;
                lay.shows = plot::button_row(f, &words, view.x - side - w, y - 3.0, 22.0, small, 6.0);
            }
            lay.left_key = Vec2::new(lx, y + big + 8.0);
            y += left_above;
            lay.left = (Vec2::new(lx + LEFT_ROOM, y), Vec2::new(view.x - side, y + hl.max(10.0)));
            y = lay.left.1.y + UNDER + 8.0;
            lay.right_head = Vec2::new(rx, y);
            lay.right_key = Vec2::new(rx, y + big + 8.0);
            y += right_above;
            lay.right = (Vec2::new(rx + LEFT_ROOM, y), Vec2::new(view.x - side, y + hr.max(10.0)));
            y = lay.right.1.y + UNDER + 8.0;
            if show_tally {
                lay.tally_head = Vec2::new(rx, y);
                lay.tally_key = Vec2::new(rx, y + big + 8.0);
                y += tally_above;
                lay.tally = (Vec2::new(rx + LEFT_ROOM, y), Vec2::new(view.x - side, y + ht.max(10.0)));
            }
        } else {
            let bottom = view.y - 12.0;
            // the right column: the error chart over the count
            let both = (bottom - y0 - right_above - UNDER - 14.0 - tally_above - UNDER).max(20.0);
            let (hr, ht) = (both * 0.64, both * 0.36);
            lay.right_head = Vec2::new(rx, y0);
            lay.right_key = Vec2::new(rx, y0 + big + 8.0);
            let ry = y0 + right_above;
            lay.right = (Vec2::new(rx + LEFT_ROOM, ry), Vec2::new(view.x - side, ry + hr));
            let ty = lay.right.1.y + UNDER + 14.0;
            lay.tally_head = Vec2::new(rx, ty);
            lay.tally_key = Vec2::new(rx, ty + big + 8.0);
            lay.tally = (Vec2::new(rx + LEFT_ROOM, ty + tally_above), Vec2::new(view.x - side, ty + tally_above + ht));
            // the left column: the data over the numbers, its chart as tall as the error chart and its key
            lay.left_head = Vec2::new(lx, y0);
            if !useless {
                lay.shows = plot::button_row(f, &SHOW_WORDS, lx, y0 + big + 8.0, 24.0, small, 6.0);
            }
            lay.left_key = Vec2::new(lx, y0 + big + 8.0 + shows_h + 8.0);
            let ly = y0 + left_above;
            let lh = (lay.right.1.y - ly).max((colw - LEFT_ROOM) * 0.6).min(colw);
            lay.left = (Vec2::new(lx + LEFT_ROOM, ly), Vec2::new(lx + colw, ly + lh));
            lay.words = (Vec2::new(lx, lay.left.1.y + UNDER + 18.0), Vec2::new(lx + colw, bottom));
        }
        lay
    }
}

const MODE_WORDS: [&str; 3] = ["polynomial degree", "ridge penalty", "twenty useless lines"];
const MODE_SHORT: [&str; 3] = ["degree", "penalty", "useless lines"];
const SHOW_WORDS: [&str; 2] = ["fitted to all the data", "each fold held back in turn"];
const MODES: [Mode; 3] = [Mode::Degree, Mode::Penalty, Mode::Useless];

// ----- the axes ------------------------------------------------------------------------------------

/// The ticks of an axis in powers of ten from 10^lo to 10^hi over `pixels`: each power
/// (or every second or third, where they would crowd) when it spans more than a decade
/// or so, round values otherwise; and `beyond` more of them past each end. Each tick is
/// its place (a power) and its number.
fn log_ticks(lo: f64, hi: f64, pixels: f32, beyond: i64) -> Vec<(f64, String)> {
    if !(lo.is_finite() && hi.is_finite() && hi > lo && pixels > 0.0) {
        return Vec::new();
    }
    if hi - lo >= 1.2 {
        let room = (pixels / 26.0).max(2.0) as f64;
        let (a, b) = (lo.ceil() as i64, hi.floor() as i64);
        let every = (((b - a + 1) as f64 / room).ceil() as i64).max(1);
        (a - every * beyond..=b + every * beyond).filter(|k| k.rem_euclid(every) == 0).map(|k| (k as f64, power_name(k))).collect()
    } else {
        lin_ticks(10f64.powf(lo), 10f64.powf(hi), pixels, beyond).into_iter().filter(|(v, _)| *v > 0.0).map(|(v, name)| (v.log10(), name)).collect()
    }
}

/// Round values from `lo` to `hi` over `pixels` (about one every 45 pixels), and `beyond`
/// more past each end.
fn lin_ticks(lo: f64, hi: f64, pixels: f32, beyond: i64) -> Vec<(f64, String)> {
    let t = plot::Ticks::of(lo, hi, pixels * 1.3);
    let values: Vec<f64> = t.values().collect();
    let (Some(&first), Some(&last)) = (values.first(), values.last()) else { return Vec::new() };
    let before = (1..=beyond).rev().map(|k| first - k as f64 * t.step);
    let after = (1..=beyond).map(|k| last + k as f64 * t.step);
    before.chain(values.iter().copied()).chain(after).map(|v| (v, t.name(v))).collect()
}

/// A value axis up the left of a chart, with a faint line across at each tick. The ticks
/// are chosen from where the axis is going (`goal`, its range once a glide is over), one
/// past each end, and drawn where it is now: each keeps its place in its batch while the
/// axis glides, unseen while it is outside the chart. (Chosen afresh from the axis as it
/// glides, ticks would come and go in the middle of a glide.)
/// The error chart's axis, in powers of ten.
fn draw_log_y_goal(f: &mut Frame, p: &Plot, g: &Goals, size: f32) {
    let ticks = log_ticks(g.right[0] as f64, g.right[1] as f64, p.height(), 1);
    value_axis(f, p, &ticks, "log scale", size);
}

fn value_axis(f: &mut Frame, p: &Plot, ticks: &[(f64, String)], name: &str, size: f32) {
    let (min, max) = p.rect;
    plot::line_of(f, plot::AXES, min, Vec2::new(min.x, max.y), 1.0, pal::DIM.a(0.8));
    for (v, label) in ticks {
        let y = p.px_y(*v);
        let seen = if y >= min.y - 0.5 && y <= max.y + 0.5 { 1.0 } else { 0.0 };
        plot::line_of(f, K_GRID, Vec2::new(min.x, y), Vec2::new(max.x, y), 1.0, pal::DIM.a(0.14 * seen));
        plot::line_of(f, K_GRID, Vec2::new(min.x - 4.0, y), Vec2::new(min.x, y), 1.0, pal::DIM.a(0.8 * seen));
        f.text_px(Vec2::new(min.x - 6.0, y), label.as_str(), size, pal::DIM.a(seen), Align2::RIGHT_CENTER);
    }
    f.text_px(Vec2::new(min.x - 34.0, min.y - 6.0), name, size, pal::DIM, Align2::LEFT_BOTTOM);
}

/// A power of ten as a tick's number: 0.01, 1, 1000, or 10⁵ as a formula.
fn power_name(k: i64) -> String {
    match k {
        -3..=-1 => num(10f64.powi(k as i32), (-k) as usize),
        0..=3 => format!("{}", 10i64.pow(k as u32)),
        _ => format!("$10^{{{k}}}$"),
    }
}

/// The ticks along the error chart's x axis: every size, and which are numbered.
fn size_ticks(mode: Mode, width: f32) -> Vec<(f64, Option<String>)> {
    match mode {
        Mode::Degree => {
            let every = if width / 16.0 >= 22.0 {
                1
            } else if width / 16.0 >= 12.0 {
                3
            } else {
                5
            };
            (0..P).map(|d| (d as f64, (d % every == 0).then(|| d.to_string()))).collect()
        }
        Mode::Penalty => {
            let every = if width / 10.0 >= 42.0 { 1 } else { 3 };
            (-7..=2).map(|k: i64| (k as f64, (k.rem_euclid(every) == if every == 1 { 0 } else { 2 }).then(|| format!("$10^{{{k}}}$")))).collect()
        }
        Mode::Useless => (1..=LINES).map(|j| (j as f64, (j == 1 || j % 5 == 0).then(|| j.to_string()))).collect(),
    }
}

/// The error chart's range across.
fn size_range(mode: Mode) -> [f64; 2] {
    match mode {
        Mode::Degree => [-0.5, MAX_DEG as f64 + 0.5],
        Mode::Penalty => [-7.3, 2.3],
        Mode::Useless => [0.3, LINES as f64 + 0.7],
    }
}

fn draw_x_ticks(f: &mut Frame, p: &Plot, ticks: &[(f64, Option<String>)], name: &str, size: f32) {
    let (min, max) = p.rect;
    plot::line_of(f, plot::AXES, Vec2::new(min.x, max.y), max, 1.0, pal::DIM.a(0.8));
    for (x, label) in ticks {
        let at = p.px_x(*x);
        plot::line_of(f, plot::AXES, Vec2::new(at, max.y), Vec2::new(at, max.y + if label.is_some() { 4.0 } else { 2.5 }), 1.0, pal::DIM.a(0.8));
        if let Some(text) = label {
            f.text_px(Vec2::new(at, max.y + 6.0), text.as_str(), size, pal::DIM, Align2::CENTER_TOP);
        }
    }
    f.text_px(Vec2::new((min.x + max.x) * 0.5, max.y + 9.0 + size), name, size, pal::DIM, Align2::CENTER_TOP);
}

/// The heights the count's chart can have: each a multiple of four, so that its four
/// steps up are whole numbers.
const TOPS: [f32; 11] = [8.0, 12.0, 20.0, 40.0, 60.0, 80.0, 100.0, 120.0, 160.0, 200.0, 240.0];

/// The count's axis: a tick at each quarter of its height, numbered with what is there.
/// The ticks stay where they are while the height glides from one of `TOPS` to another
/// (their numbers count up or down), and are round numbers once it is there.
fn count_axis(f: &mut Frame, p: &Plot, name: &str, size: f32) {
    let (min, max) = p.rect;
    plot::line_of(f, plot::AXES, min, Vec2::new(min.x, max.y), 1.0, pal::DIM.a(0.8));
    for q in 0..=4 {
        let y = max.y - (max.y - min.y) * q as f32 / 4.0;
        plot::line_of(f, plot::AXES, Vec2::new(min.x, y), Vec2::new(max.x, y), 1.0, pal::DIM.a(0.14));
        plot::line_of(f, plot::AXES, Vec2::new(min.x - 4.0, y), Vec2::new(min.x, y), 1.0, pal::DIM.a(0.8));
        let value = p.y[0] + (p.y[1] - p.y[0]) * q as f64 / 4.0;
        f.text_px(Vec2::new(min.x - 6.0, y), format!("{}", value.round()), size, pal::DIM, Align2::RIGHT_CENTER);
    }
    f.text_px(Vec2::new(min.x - 34.0, min.y - 6.0), name, size, pal::DIM, Align2::LEFT_BOTTOM);
}


impl Scene for ModelSelect {
    fn view(&self) -> View {
        View::Flat { center: Vec2::ZERO, half_height: 1.0 }
    }

    fn settings(&self, s: &mut ViewSettings) {
        s.grid = false;
        s.bloom = s.bloom.min(0.2);
    }

    fn animated(&self) -> bool {
        self.look.moving || self.tally.pending() || self.turning() || self.held.is_some()
    }

    fn pointer_hint(&self) -> Option<&'static str> {
        Some("white dot under the error chart: drag it along    buttons over the picture: click")
    }

    fn ui(&mut self, ui: &mut egui::Ui, _: &Gfx) {
        ui::note(ui, "A model flexible enough fits the data it was fitted to closely, and new data badly. Here polynomials of every degree are fitted by least squares to points scattered about a known curve. The error on the points a curve was fitted to can only fall (or stay level) as the degree grows, so it cannot choose the degree. The error on points held back can: cross-validation holds back each part of the data in turn, fits the rest, and measures the misses. And because the truth is known here, the true error on new data can be drawn beside both, to judge them by.");

        ui::section(ui, "Things to try");
        for (k, trial) in TRIALS.iter().enumerate() {
            if ui::experiment(ui, trial.label, trial.caption) {
                self.set_up(k);
            }
        }

        ui::section(ui, "What is compared");
        let mut mode = MODES.iter().position(|m| *m == self.setup.mode).unwrap_or(0);
        if ui::choice(ui, &["polynomials by degree", "one degree, a ridge penalty", "twenty useless lines"], &mut mode) {
            self.setup.mode = MODES[mode];
        }
        match self.setup.mode {
            Mode::Degree => {
                ui::slider_int(ui, "degree marked", &mut self.at[0], 0..=MAX_DEG);
                ui::note(ui, "Or drag the white dot under the error chart. A degree needs more points than it has coefficients (one more than the degree): with fewer, many curves pass through all of them and least squares cannot choose one, so the degree stops there.");
            }
            Mode::Penalty => {
                ui::slider_int(ui, "degree", &mut self.setup.degree, 1..=MAX_DEG);
                let mut lambda = PENALTIES[self.at[1]] as f32;
                if ui::slider_log(ui, r"penalty $\lambda$", &mut lambda, PENALTIES[0] as f32..=PENALTIES[PENALTIES.len() - 1] as f32) {
                    self.at[1] = self.nearest((lambda as f64).log10());
                }
                ui::note(ui, r"The curve of this degree that makes the mean squared miss plus $\lambda$ times the sum of its squared coefficients (all but the constant) as small as can be:");
                ui::equation(ui, r"\frac{1}{n} \sum_{i=1}^{n} \bigl(y_i - \hat y(x_i)\bigr)^2 + \lambda \left(c_1^2 + \cdots + c_d^2\right)");
                ui::note(ui, r"The curve is written as $\hat y(x) = c_0 + c_1 p_1(x) + \cdots + c_d p_d(x)$ in the Legendre polynomials $p_k$, scaled so that the penalised sum is the curve's mean squared distance from its own average over $-1$ to $1$. A large $\lambda$ flattens the curve towards the mean of $y$.");
            }
            Mode::Useless => {
                ui::note(ui, r"Twenty straight lines $y = a + b z_j$, each fitted to $y$ on its own input $z_j$. Every $z_j$ and $y$ is independent normal noise, so no line can predict $y$ at all: each one's true error is $\sigma^2 + a^2 + b^2$, at least $\sigma^2$. Each is judged on the same points held back; the winner misses them least.");
            }
        }

        ui::section(ui, "The data");
        if self.setup.mode != Mode::Useless {
            let mut shape = SHAPES.iter().position(|(s, _)| *s == self.setup.shape).unwrap_or(0);
            let names: Vec<&str> = SHAPES.iter().map(|(_, n)| *n).collect();
            if ui::choice(ui, &names, &mut shape) {
                self.setup.shape = SHAPES[shape].0;
            }
            ui::note(ui, r"The true curve; $x$ is spread evenly at random from $-1$ to $1$.");
        }
        let mut noise = self.setup.noise as f32;
        if ui::slider(ui, r"noise $\sigma$", &mut noise, 0.0..=1.0) {
            self.setup.noise = noise as f64;
        }
        ui::slider_int_log(ui, "points", &mut self.setup.n, 6..=300);
        if self.setup.mode == Mode::Useless {
            ui::note(ui, "Each line is fitted to this many points and judged on as many more.");
        }
        ui.horizontal(|ui| {
            if ui.button("draw again").clicked() {
                self.seed = self.seed.wrapping_add(1);
                self.line = None;
            }
            if self.setup.mode != Mode::Useless {
                ui::checkbox(ui, &mut self.show_truth, "show the truth");
            }
        });

        if self.setup.mode != Mode::Useless {
            ui::section(ui, "Cross-validation");
            ui::slider_int(ui, "folds", &mut self.setup.folds, 2..=10);
            ui::checkbox(ui, &mut self.setup.loo, "leave one out: as many folds as points");
            if !self.setup.loo && self.setup.folds > self.setup.n {
                ui::note(ui, "More folds than points: each point is a fold of its own, which is leaving one out.");
            }
            ui.horizontal(|ui| {
                if self.setup.folds_used() < self.setup.n && ui.button("deal again").clicked() {
                    self.deal = self.deal.wrapping_add(1);
                }
                ui::checkbox(ui, &mut self.turns, "each fold held back in turn");
            });
            ui::note(ui, "Dealing the points into folds again changes which points go together, and so the held-back error a little.");
        }
        self.refresh();

        ui::section(ui, "Reading the picture");
        let mut key: Vec<(Color, &str)> = vec![
            (DATA, "cyan: the points fitted to, and the error on them"),
            (OTHER, "orange: points held back, and the error on them, with a band of one standard error either way and its lowest point ringed"),
            (ALARM, "pink: the misses on held-back points, and each fold's mean squared miss"),
            (TRUTH, "green: the truth, the true error on new data, the noise's variance (dashed), and what is truly best"),
            (ESTIMATE, "yellow: the fitted curve, or line"),
            (pal::WHITE, "white: the size marked, which the left chart shows"),
        ];
        if self.setup.mode == Mode::Useless {
            key[4] = (ESTIMATE, "yellow: the line marked, fitted to its own input");
        }
        ui::key(ui, &key);

        ui::section(ui, "Words");
        ui::term(ui, "Overfitting:", "fitting the noise in the points along with the curve they came from: a small error on them, a large one on new data.");
        ui::term(ui, "Training error:", "the mean squared miss on the points the curve was fitted to.");
        ui::term(ui, "Held-back (validation) error:", "the mean squared miss on points that played no part in the fit.");
        ui::term(ui, "Cross-validation:", "deal the points into folds; hold back each fold in turn, fit the rest, measure the misses on it; average every point's squared miss.");
        ui::term(ui, "Fold:", "one of the parts the points are dealt into. Leave one out: every point a fold of its own.");
        ui::term(ui, "Standard error:", "here, how much the folds' own mean misses differ, divided by the square root of their number: a rough measure of how far the average could be off (rough, since the folds share points in their fits).");
        ui::term(ui, "Ridge penalty (regularisation):", "a cost on the size of the coefficients added to the misses: a dial from a wild curve to a flat one.");
        ui::term(ui, "Model selection:", "choosing among models (degrees, penalties, inputs) by an estimate of how each will do on new data.");
        ui::term(ui, "Test set:", "points kept apart from every fit and every choice, used once at the end to judge the model chosen.");

        self.numbers(ui);
        ui::see_also(
            ui,
            &[
                ("regress", "the same fits with their squares drawn, and their error against the true curve for every degree"),
                ("pvalues", "many tests: the same curse of choosing the best"),
                ("multireg", "which inputs to keep"),
                ("describe", "a histogram's bin width is a model size too"),
            ],
        );
    }

    fn frame(&mut self, f: &mut Frame) {
        self.refresh();
        let lay = self.layout(f);
        let snap = self.canvas.is_none_or(|(view, mode)| mode != self.setup.mode || (view - f.view_size()).length() > 0.5);
        if !lay.tiny {
            self.canvas = Some((f.view_size(), self.setup.mode));
        }
        // the folds in turn and the count over new data sets go by the scene's time
        if f.dt > 0.0 {
            if self.turning() {
                self.fold_time += f.dt;
                let period = self.fold_period();
                if self.fold_time >= period {
                    self.fold_time = (self.fold_time - period).min(period);
                    self.fold = (self.fold + 1) % self.sweep.k.max(1);
                }
            }
            if self.tally.pending() {
                let truth = *self.truth();
                let mut turn = scene_turn(&mut self.pace, TALLY_RATE, f.dt, f.real_dt);
                while self.tally.pending() && turn.next() {
                    self.tally.step(&self.setup, &truth);
                }
                self.pace.end(turn);
            }
        }
        if lay.tiny {
            let g = self.goals();
            self.look.follow(&g, f.real_dt, true);
            let view = f.view_size();
            if view.x > 160.0 && view.y > 90.0 {
                let at = Vec2::new(view.x * 0.5, (f.top_inset + view.y) * 0.5);
                f.text_px(at, "a window too small for the charts", 11.0, pal::DIM, Align2::CENTER_CENTER);
            }
            return;
        }
        self.buttons(f, &lay);
        // the white dot is dragged before the goals are made, so that they follow it
        let right = self.right_plot(&lay);
        self.drag_mark(f, &right);
        let g = self.goals();
        self.look.follow(&g, f.real_dt, snap);
        self.draw_left(f, &lay, &g);
        let right = self.right_plot(&lay);
        self.draw_right(f, &lay, &right, &g);
        if lay.tally.1.y > lay.tally.0.y + 20.0 {
            self.draw_tally(f, &lay, &right);
        }
        if lay.words.1.y - lay.words.0.y > 60.0 {
            self.draw_words(f, &lay);
        }
    }
}

impl ModelSelect {
    /// The buttons over the picture.
    fn buttons(&mut self, f: &mut Frame, lay: &Layout) {
        let words = if lay.narrow { MODE_SHORT } else { MODE_WORDS };
        for (i, r) in lay.modes.iter().enumerate() {
            if plot::button(f, *r, words[i], lay.text, self.setup.mode == MODES[i]) && self.setup.mode != MODES[i] {
                self.setup.mode = MODES[i];
                self.held = None;
            }
        }
        if plot::button(f, lay.again, "draw again", lay.text, false) {
            self.seed = self.seed.wrapping_add(1);
            self.line = None;
        }
        if plot::button(f, lay.deal, "deal again", lay.text, false) {
            self.deal = self.deal.wrapping_add(1);
        }
        let shows = if lay.narrow { ["all data", "folds in turn"] } else { SHOW_WORDS };
        for (i, r) in lay.shows.iter().enumerate() {
            if plot::button(f, *r, shows[i], lay.small, self.turns == (i == 1)) {
                self.turns = i == 1;
            }
        }
        self.refresh();
    }

    fn right_plot(&self, lay: &Layout) -> Plot {
        let r = &self.look.right.shown;
        let range = if r.len() == 2 { [r[0] as f64, r[1] as f64] } else { self.error_range() };
        Plot::new(lay.right, size_range(self.setup.mode), range)
    }

    /// The white dot under the error chart: dragged along, it marks the nearest size.
    fn drag_mark(&mut self, f: &mut Frame, p: &Plot) {
        if p.tiny() {
            self.held = None;
            return;
        }
        let x = self.held.unwrap_or_else(|| self.look.mark.shown.first().map_or(self.size_x(self.marked()), |&v| v as f64));
        if let Some((to, _)) = p.handle(f, 7, x, p.y[0], pal::WHITE) {
            let to = to.max(size_range(self.setup.mode)[0]).min(size_range(self.setup.mode)[1]);
            self.held = Some(to);
            let size = self.nearest(to);
            match self.setup.mode {
                Mode::Degree => self.at[0] = size,
                Mode::Penalty => self.at[1] = size,
                Mode::Useless => self.line = Some(size),
            }
        } else if !f.handles.dragging() {
            self.held = None;
        }
    }

    fn draw_left(&self, f: &mut Frame, lay: &Layout, g: &Goals) {
        let useless = self.setup.mode == Mode::Useless;
        let range = self.look.left.shown.clone();
        let yr = if range.len() == 2 { [range[0] as f64, range[1] as f64] } else { [-2.0, 2.0] };
        // (room at the top for the line that says what the curve is)
        let height = (lay.left.1.y - lay.left.0.y).max(40.0) as f64;
        let room = (lay.small as f64 + 14.0) / height;
        let with_room = |[lo, hi]: [f64; 2]| [lo, hi + (hi - lo) * room / (1.0 - room).max(0.5)];
        let yr = with_room(yr);
        let goal = with_room([g.left[0] as f64, g.left[1] as f64]);
        let xr = if useless { [-3.2, 3.2] } else { [-1.05, 1.05] };
        let p = Plot::new(lay.left, xr, yr);
        p.backing(f, LEFT_ROOM + 6.0, UNDER + 4.0, lay.left.0.y - lay.left_head.y + 6.0, 10.0);
        let size = self.marked();
        let heading = if useless {
            if lay.narrow {
                format!("line {} on its own input", size + 1)
            } else {
                format!("line {}, fitted to $y$ on its own input $z_{{{}}}$", size + 1, size + 1)
            }
        } else {
            "the data and the fitted curve".to_string()
        };
        f.text_px(lay.left_head, heading, lay.big, pal::TEXT, Align2::LEFT_TOP);
        let entries = self.left_entries();
        draw_key(f, lay.left_key, lay.left.1.x - lay.left_key.x, &entries, lay.small);
        p.x_axis(f, if useless { r"$z$, the line's input" } else { "$x$" }, lay.small);
        value_axis(f, &p, &lin_ticks(goal[0], goal[1], p.height(), 1), "$y$", lay.small);
        if p.tiny() {
            return;
        }

        let pts = &self.look.points.shown;
        let n = pts.len() / 2;
        let dot = if n <= 60 {
            6.5
        } else if n <= 150 {
            5.0
        } else {
            3.8
        };
        // the curve, as drawn: its coefficients glide
        let c: Vec<f64> = self.look.curve.shown.iter().map(|&v| v as f64).collect();
        let curve_at = |x: f64, row: &[f64; P]| -> f64 {
            if useless {
                c.first().copied().unwrap_or(0.0) + c.get(1).copied().unwrap_or(0.0) * x
            } else {
                c.iter().zip(row).map(|(a, b)| a * b).sum()
            }
        };
        if useless {
            let held_from = self.contest.y_fit.len();
            let j = size;
            // the misses on the points held back
            for i in held_from..n {
                let (x, y) = (pts[2 * i] as f64, pts[2 * i + 1] as f64);
                let at = curve_at(x, &[0.0; P]);
                let (a, b) = (p.at(x, y), p.at(x, at));
                plot::line_of(f, K_MISS, a, b.max(p.rect.0).min(p.rect.1), 1.4, ALARM.a(0.8));
            }
            p.curve(f, K_FIT, 2, 2.6, ESTIMATE, |x| curve_at(x, &[0.0; P]));
            for i in 0..n {
                let color = if i < held_from { DATA } else { OTHER };
                p.dot(f, pts[2 * i] as f64, pts[2 * i + 1] as f64, dot - 0.5, color);
            }
            let c = &self.contest;
            let text = format!("held back: mean squared miss {}", err_num(c.held.get(j).copied().unwrap_or(f64::NAN)));
            f.caption_px(p.rect.0 + Vec2::new(8.0, 8.0), text, lay.small, pal::TEXT, Align2::LEFT_TOP);
            return;
        }
        if self.show_truth {
            let shape = self.setup.shape;
            p.path(f, K_TRUTH, self.grid.iter().map(|(x, _)| (*x, shape.at(*x))), 2.0, TRUTH.a(0.85));
        }
        let turning = self.turning();
        let fold_of = &self.sweep.fold_of;
        // how far each point is held back, as drawn: a new fold's misses grow as the
        // last one's shrink
        let share = |i: usize| self.look.held.shown.get(i).copied().unwrap_or(0.0).max(0.0).min(1.0);
        if g.curve_on {
            for i in 0..n.min(self.data.len()) {
                let s = share(i);
                let (x, y) = (pts[2 * i] as f64, pts[2 * i + 1] as f64);
                let at = curve_at(x, &self.data.rows[i]);
                let (a, b) = (p.at(x, y), p.at(x, y + (at - y) * s as f64));
                plot::line_of(f, K_MISS, a, b.max(p.rect.0).min(p.rect.1), 1.5, ALARM.a(0.85 * s));
            }
            let pts: Vec<(f64, f64)> = self.grid.iter().map(|(x, row)| (*x, curve_at(*x, row))).collect();
            for w in pts.windows(2) {
                segment_in(f, K_FIT, &p, w[0], w[1], 2.6, ESTIMATE);
            }
        }
        for i in 0..n {
            let s = share(i);
            p.dot(f, pts[2 * i] as f64, pts[2 * i + 1] as f64, dot + s, DATA.mix(OTHER, s));
        }
        // what the curve is
        let (deg, n_all) = (degree_of(&self.setup, size), self.data.len());
        let what = match self.setup.mode {
            Mode::Penalty => format!(r"degree {deg}, $\lambda = {}$", penalty_num(PENALTIES[size])),
            _ => format!("degree {deg}"),
        };
        let text = if !g.curve_on {
            "no unique curve".to_string()
        } else if turning {
            let k = self.sweep.k;
            let out = fold_of.iter().filter(|&&q| q == self.fold).count();
            if k == n_all {
                format!("point {} held back: {what} fitted to the other {}", self.fold + 1, n_all - 1)
            } else {
                format!("fold {} of {k} held back: {what} fitted to the other {}", self.fold + 1, n_all - out)
            }
        } else {
            format!("{what}, fitted to all {n_all} points")
        };
        f.caption_px(p.rect.0 + Vec2::new(8.0, 8.0), text, lay.small, pal::TEXT, Align2::LEFT_TOP);
        if self.setup.mode == Mode::Degree && self.at[0] > self.most_degree() {
            let note = format!("degree {} would have {} coefficients for {} points: no unique fit", self.at[0], self.at[0] + 1, n_all);
            f.caption_px(Vec2::new(p.rect.0.x + 8.0, p.rect.1.y - 8.0), note, lay.small, ALARM, Align2::LEFT_BOTTOM);
        }
    }

    fn draw_right(&self, f: &mut Frame, lay: &Layout, p: &Plot, g: &Goals) {
        let mode = self.setup.mode;
        let sizes = self.setup.sizes();
        p.backing(f, LEFT_ROOM + 6.0, UNDER + 4.0, lay.right.0.y - lay.right_head.y + 6.0, 10.0);
        let heading = match (mode, lay.narrow) {
            (Mode::Degree, false) => "mean squared error, by the degree of the polynomial",
            (Mode::Degree, true) => "mean squared error by degree",
            (Mode::Penalty, false) => r"mean squared error, by the penalty $\lambda$ (degree fixed)",
            (Mode::Penalty, true) => r"mean squared error by penalty $\lambda$",
            (Mode::Useless, false) => "mean squared error of each line",
            (Mode::Useless, true) => "error of each line",
        };
        f.text_px(lay.right_head, heading, lay.big, pal::TEXT, Align2::LEFT_TOP);
        let entries = self.right_entries();
        draw_key(f, lay.right_key, lay.right.1.x - lay.right_key.x, &entries, lay.small);
        let x_name = match mode {
            Mode::Degree => "degree",
            Mode::Penalty => r"penalty $\lambda$: stronger to the right",
            Mode::Useless => "line",
        };
        draw_x_ticks(f, p, &size_ticks(mode, p.width()), x_name, lay.small);
        draw_log_y_goal(f, p, g, lay.small);
        if p.tiny() {
            return;
        }
        let s = &self.look.series.shown;
        if s.len() != 5 * sizes || g.shown.len() != 5 * sizes {
            return;
        }
        let at = |k: usize, i: usize| s[k * sizes + i] as f64;
        let real = |k: usize, i: usize| g.shown[k * sizes + i];
        // degrees with no unique fit to all the points, or to some fold's
        if mode == Mode::Degree {
            let all = (0..sizes).filter(|&i| real(0, i)).max().unwrap_or(0);
            let cv = (0..sizes).filter(|&i| real(1, i)).max();
            let shade = |f: &mut Frame, from: f64, color: Color, text: &str| {
                let x0 = p.px_x(from).max(p.rect.0.x);
                if x0 < p.rect.1.x - 2.0 {
                    plot::fill(f, (Vec2::new(x0, p.rect.0.y), p.rect.1), color);
                    if p.rect.1.x - x0 > f.text_size(text, lay.small).x + 10.0 {
                        f.text_px(Vec2::new(x0 + 5.0, p.rect.0.y + 6.0), text, lay.small, pal::DIM, Align2::LEFT_TOP);
                    }
                }
            };
            if all + 1 < sizes {
                shade(f, all as f64 + 0.5, pal::DIM.a(0.16), "no unique fit");
            }
            if let Some(cv) = cv.filter(|&c| c < all) {
                let x1 = p.px_x(all as f64 + 0.5);
                let x0 = p.px_x(cv as f64 + 0.5);
                plot::fill(f, (Vec2::new(x0, p.rect.0.y), Vec2::new(x1, p.rect.1.y)), pal::DIM.a(0.08));
            }
        }
        // the noise's variance: no model's true error is below it
        let noise2 = self.setup.noise * self.setup.noise;
        if noise2 > 0.0 {
            p.hline(f, plot::MARKS, noise2.log10(), 1.3, TRUTH.a(0.55), true);
        }
        let xs: Vec<f64> = (0..sizes).map(|i| self.size_x(i)).collect();
        let [train, cv, _, _, truth] = self.series();
        if train.iter().chain(&cv).chain(&truth).any(|&v| v == 0.0) {
            f.text_px(Vec2::new(p.rect.1.x - 6.0, p.rect.1.y - 5.0), "errors of exactly 0 lie below the chart", lay.small, pal::DIM, Align2::RIGHT_BOTTOM);
        }
        let pieces = |k: usize| -> Vec<(f64, f64)> { (0..sizes).map(|i| if real(k, i) { (xs[i], at(k, i)) } else { (f64::NAN, f64::NAN) }).collect() };
        if mode != Mode::Useless {
            // the band of one standard error either way about the held-back error
            for i in 0..sizes.saturating_sub(1) {
                if real(2, i) && real(2, i + 1) {
                    let (a, b) = (xs[i], xs[i + 1]);
                    let lerp = |k: usize| move |x: f64| at(k, i) + (at(k, i + 1) - at(k, i)) * (x - a) / (b - a);
                    p.band(f, a, b, 2, OTHER.a(0.08), lerp(2), lerp(3));
                }
            }
            for (k, color) in [(0, DATA), (4, TRUTH), (1, OTHER)] {
                let pts = pieces(k);
                for w in pts.windows(2) {
                    segment_in(f, K_SERIES, p, w[0], w[1], 2.0, color);
                }
            }
        } else {
            // each line's three errors joined, so that they read as one line's
            for i in 0..sizes {
                let (lo, hi) = (at(1, i).min(at(4, i)).min(at(0, i)), at(1, i).max(at(4, i)).max(at(0, i)));
                plot::line_of(f, K_SERIES, p.at(xs[i], lo).max(p.rect.0).min(p.rect.1), p.at(xs[i], hi).max(p.rect.0).min(p.rect.1), 1.0, pal::DIM.a(0.35));
            }
        }
        for (k, color, size) in [(0, DATA, 5.0), (4, TRUTH, 5.5), (1, OTHER, 6.0)] {
            for i in 0..sizes {
                if real(k, i) {
                    dot_in(f, p, xs[i], at(k, i), if mode == Mode::Useless { size + 1.5 } else { size }, color);
                }
            }
        }
        // the white mark, and each fold's own mean miss on it
        let mark = self.look.mark.get(0) as f64;
        p.vline(f, K_MARK, mark, 1.4, pal::WHITE.a(0.6), false);
        if mode != Mode::Useless {
            let folds = &self.look.folds.shown;
            let turning = self.turning();
            for (q, &v) in folds.iter().enumerate() {
                let now = turning && q == self.fold;
                dot_in(f, p, mark, v as f64, if now { 7.0 } else { 4.0 }, if now { ALARM } else { ALARM.a(0.75) });
            }
        }
        // the lowest held-back error (the winner) and the truly best, gliding when they move
        let looks = if mode == Mode::Useless { [(pal::WHITE, 9.0), (TRUTH, 11.0)] } else { [(OTHER, 8.0), (TRUTH, 11.0)] };
        for (r, (color, radius)) in looks.into_iter().enumerate() {
            let c = p.at(self.look.rings.get(2 * r) as f64, self.look.rings.get(2 * r + 1) as f64);
            if g.rings_on[r] {
                let seen = if plot::within(c, p.rect, 0.5) { 1.0 } else { 0.0 };
                ring(f, K_MARK, c.max(p.rect.0).min(p.rect.1), radius, color.a(color.0[3] as f32 / 255.0 * seen));
            }
        }
    }

    fn draw_tally(&self, f: &mut Frame, lay: &Layout, right: &Plot) {
        let mode = self.setup.mode;
        let t = &self.tally;
        let top = self.look.top.get(0).max(1.0) as f64;
        let bars = &self.look.bars.shown;
        let heading = match (mode, lay.narrow) {
            (Mode::Useless, false) => format!("true error less held-back error, over {} new contests", t.done),
            (Mode::Useless, true) => format!("true less held-back error, {} contests", t.done),
            (_, false) => format!("what cross-validation chose on {} new data sets like this one", t.done),
            (_, true) => format!("chosen on {} new data sets", t.done),
        };
        let x = if mode == Mode::Useless {
            let w = gap_range(&self.setup);
            [-w, w]
        } else {
            right.x
        };
        let p = Plot::new(lay.tally, x, [0.0, top]);
        p.backing(f, LEFT_ROOM + 6.0, UNDER + 4.0, lay.tally.0.y - lay.tally_head.y + 6.0, 10.0);
        f.text_px(lay.tally_head, heading, lay.big, pal::TEXT, Align2::LEFT_TOP);
        let entries = self.tally_entries();
        draw_key(f, lay.tally_key, lay.tally.1.x - lay.tally_key.x, &entries, lay.small);
        count_axis(f, &p, if mode == Mode::Useless { "contests" } else { "data sets" }, lay.small);
        if mode == Mode::Useless {
            p.x_axis(f, "true error less held-back error", lay.small);
        } else {
            draw_x_ticks(f, &p, &size_ticks(mode, p.width()), if mode == Mode::Degree { "degree chosen" } else { r"penalty $\lambda$ chosen" }, lay.small);
        }
        if p.tiny() {
            return;
        }
        if mode == Mode::Useless {
            let w = gap_range(&self.setup);
            let bin = 2.0 * w / GAP_BINS as f64;
            for (k, color) in [(1, pal::DIM.a(0.45)), (0, ALARM.a(0.7))] {
                for b in 0..GAP_BINS {
                    let h = bars.get(k * GAP_BINS + b).copied().unwrap_or(0.0) as f64;
                    p.bar(f, -w + b as f64 * bin, -w + (b + 1) as f64 * bin, 0.0, h, 1.0, color);
                }
            }
            p.vline(f, plot::MARKS, 0.0, 1.0, pal::DIM.a(0.8), true);
            let mean = |v: &[f64]| if v.is_empty() { f64::NAN } else { v.iter().sum::<f64>() / v.len() as f64 };
            // (the lines glide to the means, which move a little with each contest)
            for (k, v, color, y) in [(0, mean(&t.gaps), ALARM, 6.0), (1, mean(&t.random), pal::DIM, 22.0)] {
                if v.is_finite() {
                    let x = self.look.lines.get(k) as f64;
                    p.vline(f, K_TALLY, x, 1.8, color, false);
                    let at = Vec2::new(p.px_x(x) + 5.0, p.rect.0.y + y);
                    let decimals = (1.0 - w.log10().floor()).max(2.0) as usize;
                    f.text_px(at, format!("mean {}", num(v, decimals)), lay.small, color, Align2::LEFT_TOP);
                }
            }
            return;
        }
        let sizes = self.setup.sizes();
        for i in 0..sizes {
            let h = bars.get(i).copied().unwrap_or(0.0) as f64;
            let x = self.size_x(i);
            let half = if mode == Mode::Degree { 0.42 } else { 0.13 };
            p.bar(f, x - half, x + half, 0.0, h, 0.0, OTHER.a(0.75));
        }
        if self.sweep.k == self.setup.n {
            for i in 0..sizes {
                let h = bars.get(sizes + i).copied().unwrap_or(0.0) as f64;
                let x = self.size_x(i);
                let half = if mode == Mode::Degree { 0.42 } else { 0.13 };
                if h > 0.0 {
                    // (thin filled sides, like the bars they stand beside)
                    let (a, b) = (p.at(x - half, 0.0), p.at(x + half, h.min(top)));
                    let w = 1.2;
                    for side in [(Vec2::new(a.x, b.y), Vec2::new(a.x + w, a.y)), (Vec2::new(b.x - w, b.y), Vec2::new(b.x, a.y)), (Vec2::new(a.x, b.y), Vec2::new(b.x, b.y + w))] {
                        plot::fill(f, side, pal::WHITE.a(0.8));
                    }
                }
            }
        }
        // (once every data set is in: before, it would wander as they come)
        if t.best().is_some() && !t.pending() {
            p.vline(f, plot::MARKS, self.look.lines.get(0) as f64, 2.0, TRUTH, false);
        }
    }

    /// The numbers of the size marked, under the left chart.
    fn draw_words(&self, f: &mut Frame, lay: &Layout) {
        let (min, max) = lay.words;
        let size = lay.text;
        let line = size + 9.0;
        let mut y = min.y;
        let column = min.x + ((max.x - min.x) * 0.62).min(330.0);
        let i = self.marked();
        let row = |f: &mut Frame, y: f32, color: Color, name: &str, value: String| {
            plot::fill(f, (Vec2::new(min.x, y + 3.0), Vec2::new(min.x + 9.0, y + size)), color);
            f.text_px(Vec2::new(min.x + 16.0, y), name, size, pal::TEXT.a(0.9), Align2::LEFT_TOP);
            f.text_px(Vec2::new(column, y), value, size, pal::TEXT, Align2::LEFT_TOP);
        };
        if self.setup.mode == Mode::Useless {
            let c = &self.contest;
            let n = self.setup.n;
            let title = if i == c.winner { format!("line {}, the winner:", i + 1) } else { format!("line {}:", i + 1) };
            f.text_px(Vec2::new(min.x, y), title, size, pal::TEXT, Align2::LEFT_TOP);
            y += line;
            row(f, y, DATA, &format!("on the {n} points it was fitted to"), format!("${}$", err_num(c.train[i])));
            y += line;
            row(f, y, OTHER, &format!("on the {n} points held back"), format!("${}$", err_num(c.held[i])));
            y += line;
            row(f, y, TRUTH, "on new data (true)", format!("${}$", err_num(c.truth[i])));
            y += line + 4.0;
            let others = |v: &[f64]| {
                let o: Vec<f64> = v.iter().enumerate().filter(|(j, _)| *j != i).map(|(_, x)| *x).collect();
                (o.iter().copied().fold(f64::INFINITY, f64::min), o.iter().copied().fold(f64::NEG_INFINITY, f64::max))
            };
            let (h0, h1) = others(&c.held);
            let (t0, t1) = others(&c.truth);
            let text = format!("The other 19: held back {} to {}; on new data {} to {}.", num(h0, 2), num(h1, 2), num(t0, 2), num(t1, 2));
            y += wrapped_px(f, Vec2::new(min.x, y), max.x - min.x, &text, lay.small, pal::DIM) + 12.0;
            if max.y - y > 60.0 {
                wrapped_px(f, Vec2::new(min.x, y), max.x - min.x, USELESS_WORDS, lay.small, pal::DIM);
            }
            return;
        }
        let sw = &self.sweep;
        let s = &self.setup;
        let title = match s.mode {
            Mode::Penalty => format!(r"degree {} with penalty $\lambda = {}$:", s.degree, penalty_num(PENALTIES[i])),
            _ => format!("degree {i}:"),
        };
        f.text_px(Vec2::new(min.x, y), title, size, pal::TEXT, Align2::LEFT_TOP);
        y += line;
        let k = sw.k;
        let folds = if k == s.n { "held back one at a time".to_string() } else { format!("held back, {k} folds") };
        let get = |v: &[f64]| v.get(i).copied().unwrap_or(f64::NAN);
        row(f, y, DATA, &format!("on the {} points it was fitted to", s.n), format!("${}$", err_num(get(&sw.train))));
        y += line;
        let cv = get(&sw.cv);
        let held = if cv.is_finite() { format!(r"${} \pm {}$", err_num(cv), err_num(get(&sw.se))) } else { "no unique fit".to_string() };
        row(f, y, OTHER, &folds, held);
        y += line;
        row(f, y, TRUTH, "on new data (true)", format!("${}$", err_num(get(&sw.truth))));
        y += line + 4.0;
        let name = |j: Option<usize>| match (j, s.mode) {
            (None, _) => "none".to_string(),
            (Some(j), Mode::Penalty) => format!(r"$\lambda = {}$", penalty_num(PENALTIES[j])),
            (Some(j), _) => format!("degree {j}"),
        };
        let text = format!("Lowest held-back error: {}. Truly best: {}.", name(sw.lowest()), name(sw.best()));
        y += wrapped_px(f, Vec2::new(min.x, y), max.x - min.x, &text, lay.small, pal::DIM) + 12.0;
        let words = if s.mode == Mode::Penalty { PENALTY_WORDS } else { DEGREE_WORDS };
        if max.y - y > 60.0 {
            wrapped_px(f, Vec2::new(min.x, y), max.x - min.x, words, lay.small, pal::DIM);
        }
    }

    /// The numbers in the panel.
    fn numbers(&self, ui: &mut egui::Ui) {
        ui::section(ui, "Numbers");
        let i = self.marked();
        let s = &self.setup;
        let t = &self.tally;
        if s.mode == Mode::Useless {
            let c = &self.contest;
            if let (Some(train), Some(held), Some(truth)) = (c.train.get(i), c.held.get(i), c.truth.get(i)) {
                ui::readout(ui, "line marked", if i == c.winner { format!("{} (the winner)", i + 1) } else { (i + 1).to_string() });
                ui::readout_series(ui, "on the points fitted to", format!("${}$", err_num(*train)), DATA);
                ui::readout_series(ui, "held back", format!("${}$", err_num(*held)), OTHER);
                ui::readout_series(ui, "on new data (true)", format!("${}$", err_num(*truth)), TRUTH);
            }
            let mean = |v: &[f64]| if v.is_empty() { f64::NAN } else { v.iter().sum::<f64>() / v.len() as f64 };
            ui::readout(ui, "contests counted", format!("{} of {RUNS}", t.done));
            if t.done > 0 {
                ui::readout(ui, "true less held back, winner", format!("${}$", num(mean(&t.gaps), 3)));
                ui::readout(ui, "the same, a line at random", format!("${}$", num(mean(&t.random), 3)));
            }
            return;
        }
        let sw = &self.sweep;
        let get = |v: &[f64]| v.get(i).copied().unwrap_or(f64::NAN);
        match s.mode {
            Mode::Penalty => ui::readout(ui, r"penalty marked", format!("${}$", penalty_num(PENALTIES[i]))),
            _ => ui::readout(ui, "degree marked", i.to_string()),
        }
        ui::readout_series(ui, "on the points fitted to", format!("${}$", err_num(get(&sw.train))), DATA);
        let cv = get(&sw.cv);
        ui::readout_series(ui, "held back", if cv.is_finite() { format!(r"${} \pm {}$", err_num(cv), err_num(get(&sw.se))) } else { "no unique fit".to_string() }, OTHER);
        ui::readout_series(ui, "on new data (true)", format!("${}$", err_num(get(&sw.truth))), TRUTH);
        let name = |j: Option<usize>| match (j, s.mode) {
            (None, _) => "none".to_string(),
            (Some(j), Mode::Penalty) => format!("${}$", penalty_num(PENALTIES[j])),
            (Some(j), _) => j.to_string(),
        };
        ui::readout(ui, "lowest held-back error at", name(sw.lowest()));
        ui::readout(ui, "truly best at", name(sw.best()));
        ui::readout(ui, "folds", if sw.k == s.n { format!("{} (leave one out)", sw.k) } else { sw.k.to_string() });
        ui::readout(ui, "new data sets counted", format!("{} of {RUNS}", t.done));
        let most = (0..t.chosen.len()).max_by_key(|&j| (t.chosen[j], std::cmp::Reverse(j))).filter(|_| t.done > 0);
        ui::readout(ui, "chosen most often", name(most));
        ui::readout(ui, "best on average", name(t.best().filter(|_| t.done > 0)));
    }
}

/// What the error chart's curves are, under the numbers on a desktop.
const DEGREE_WORDS: &str = "On the points it was fitted to (cyan), the error can only fall or stay level as the degree grows: a curve of higher degree can do anything one of lower degree can, and more. Held back (orange), each point's miss is measured by a curve fitted without it, so the error falls while the curve learns the shape, and rises again once it starts to follow the noise. On new data (green) is what matters, and is known here only because the truth is. No curve does better there, on average, than the noise's own variance (dashed).";
const PENALTY_WORDS: &str = "On the points fitted to (cyan), a stronger penalty can only raise the error or leave it: it holds the curve back from them. Held back (orange), each point's miss is measured by a curve fitted without it, from the curve of this degree with next to no penalty at the left to a flat line at the right. On new data (green) is what matters, and is known here only because the truth is.";
const USELESS_WORDS: &str = "No line can predict y: every line's true error (green) is above the noise's variance (dashed). Held back (orange), the errors differ by chance, and the winner is the line that chance favoured most on these points.";

/// `kit::wrapped` in logical pixels from the top-left of the picture (it works in the
/// plane's own units).
/// Returns how tall it is.
fn wrapped_px(f: &mut Frame, at: Vec2, width: f32, text: &str, size: f32, color: Color) -> f32 {
    let Some(world) = f.cam.pick_ground(at * f.px) else { return 0.0 };
    let width = width / pixels_per_unit(f).max(1e-6);
    wrapped(f, world.truncate(), width, text, size, color);
    wrapped_height(f, width, text, size)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn opening() -> Setup {
        WAVE
    }

    /// An experiment as its button sets it up.
    fn trial(k: usize) -> ModelSelect {
        let mut s = ModelSelect::create();
        s.set_up(k);
        s
    }

    /// The count over new data sets, to the end.
    fn counted(s: &mut ModelSelect) -> &Tally {
        let truth = *s.truth();
        while s.tally.pending() {
            s.tally.step(&s.setup, &truth);
        }
        &s.tally
    }

    /// The most chosen size, and the chosen sizes' mean and standard deviation.
    fn spread(counts: &[u32]) -> (usize, f64, f64) {
        let total: f64 = counts.iter().map(|&c| c as f64).sum();
        let mean = counts.iter().enumerate().map(|(i, &c)| i as f64 * c as f64).sum::<f64>() / total;
        let var = counts.iter().enumerate().map(|(i, &c)| (i as f64 - mean).powi(2) * c as f64).sum::<f64>() / total;
        let most = (0..counts.len()).max_by_key(|&i| (counts[i], std::cmp::Reverse(i))).unwrap_or(0);
        (most, mean, var.sqrt())
    }

    fn mean(v: &[f64]) -> f64 {
        v.iter().sum::<f64>() / v.len() as f64
    }

    /// Gaussian elimination with partial pivoting: a solver that shares nothing with the scene's.
    fn solve_dense(mut a: Vec<Vec<f64>>, mut b: Vec<f64>) -> Vec<f64> {
        let n = b.len();
        for c in 0..n {
            let p = (c..n).max_by(|&i, &j| a[i][c].abs().total_cmp(&a[j][c].abs())).unwrap();
            a.swap(c, p);
            b.swap(c, p);
            for r in c + 1..n {
                let f = a[r][c] / a[c][c];
                for j in c..n {
                    a[r][j] -= f * a[c][j];
                }
                b[r] -= f * b[c];
            }
        }
        let mut x = vec![0.0; n];
        for c in (0..n).rev() {
            x[c] = (b[c] - (c + 1..n).map(|j| a[c][j] * x[j]).sum::<f64>()) / a[c][c];
        }
        x
    }

    /// Least squares of degree `d` in raw powers of x, by the normal equations: fine for
    /// the small degrees of these checks.
    fn raw_fit(x: &[f64], y: &[f64], d: usize) -> impl Fn(f64) -> f64 {
        let k = d + 1;
        let mut a = vec![vec![0.0; k]; k];
        let mut b = vec![0.0; k];
        for (&x, &y) in x.iter().zip(y) {
            for i in 0..k {
                b[i] += x.powi(i as i32) * y;
                for j in 0..k {
                    a[i][j] += x.powi((i + j) as i32);
                }
            }
        }
        let c = solve_dense(a, b);
        move |x: f64| c.iter().enumerate().map(|(i, c)| c * x.powi(i as i32)).sum()
    }

    /// The ridge fit of degree `d` by its normal equations: (XᵀX/m + λD) c = Xᵀy/m.
    fn ridge_fit(rows: &[[f64; P]], y: &[f64], d: usize, lambda: f64) -> Vec<f64> {
        let (k, m) = (d + 1, rows.len() as f64);
        let mut a = vec![vec![0.0; k]; k];
        let mut b = vec![0.0; k];
        for (r, &y) in rows.iter().zip(y) {
            for i in 0..k {
                b[i] += r[i] * y / m;
                for j in 0..k {
                    a[i][j] += r[i] * r[j] / m;
                }
            }
        }
        for (i, row) in a.iter_mut().enumerate().skip(1) {
            row[i] += lambda;
        }
        solve_dense(a, b)
    }

    fn close(a: f64, b: f64, tol: f64) -> bool {
        (a - b).abs() <= tol * (a.abs() + b.abs()).max(1e-300)
    }

    #[test]
    fn the_legendre_polynomials_are_orthonormal_and_the_truth_is_measured_right() {
        // the mean of p_j p_k over -1 to 1, by Simpson's rule on a fine grid
        let n = 8000;
        let simpson = |i: usize, n: usize| if i == 0 || i == n { 1.0 } else if i % 2 == 1 { 4.0 } else { 2.0 } * (2.0 / n as f64) / 6.0;
        let mut gram = [[0.0; P]; P];
        for i in 0..=n {
            let x = -1.0 + 2.0 * i as f64 / n as f64;
            let (w, p) = (simpson(i, n), legendre(x));
            for j in 0..P {
                for k in 0..P {
                    gram[j][k] += w * p[j] * p[k];
                }
            }
        }
        for j in 0..P {
            for k in 0..P {
                let want = if j == k { 1.0 } else { 0.0 };
                assert!((gram[j][k] - want).abs() < 1e-8, "({j}, {k}): {}", gram[j][k]);
            }
        }
        // a polynomial's mean squared distance from each shape, as a sum of squares and
        // directly (on a grid five times finer, with the corner at a joint again)
        let mut rng = Rng::new(3);
        for (shape, _) in SHAPES {
            let truth = Truth::of(shape);
            for d in [0, 3, 9, 15] {
                let mut c = [0.0; P];
                for v in c.iter_mut().take(d + 1) {
                    *v = normal(&mut rng);
                }
                let m = 20_000;
                let direct: f64 = (0..=m).map(|i| (i, -1.0 + 2.0 * i as f64 / m as f64)).map(|(i, x)| simpson(i, m) * (value(&c, &legendre(x)) - shape.at(x)).powi(2)).sum();
                assert!(close(truth.distance(&c, d), direct, 1e-7), "{shape:?} degree {d}: {} against {direct}", truth.distance(&c, d));
            }
        }
    }

    /// The fits of every degree are nested: each adds a column to the last, so the error
    /// on the points fitted to can never rise with the degree. Nor can it fall as the
    /// penalty grows. And no fit's true error is below the noise's variance.
    #[test]
    fn training_error_never_rises_with_the_degree_nor_falls_with_the_penalty() {
        for (n, shape) in [(12, Shape::Wave), (30, Shape::Step), (30, Shape::Corner), (300, Shape::Line), (300, Shape::Wave)] {
            for seed in 1..6 {
                let s = Setup { shape, n, ..WAVE };
                let truth = Truth::of(shape);
                let data = Data::draw(seed, n, shape, s.noise);
                let sw = Sweep::new(&data, &s, &truth, 1);
                let fitted: Vec<f64> = sw.train.iter().copied().filter(|v| v.is_finite()).collect();
                assert_eq!(fitted.len(), P.min(n), "n {n}: {:?}", sw.train);
                for w in fitted.windows(2) {
                    assert!(w[1] <= w[0] * (1.0 + 1e-9) + 1e-12, "n {n} seed {seed}: {:?}", sw.train);
                }
                assert!(sw.truth.iter().filter(|v| v.is_finite()).all(|&t| t >= s.noise * s.noise));
                let pen = Sweep::new(&data, &Setup { mode: Mode::Penalty, ..s }, &truth, 1);
                for w in pen.train.windows(2) {
                    assert!(w[1] >= w[0] * (1.0 - 1e-9), "n {n} seed {seed}: {:?}", pen.train);
                }
            }
        }
    }

    /// The cross-validated error is made by fitting the curve again without each fold:
    /// the same numbers come from fitting it again here, in raw powers of x by another
    /// solver, and for the penalty by its normal equations.
    #[test]
    fn cross_validation_refits_without_each_fold() {
        let truth = Truth::of(Shape::Wave);
        for (n, k) in [(12, 3), (13, 5), (20, 10)] {
            let s = Setup { n, folds: k, ..WAVE };
            let data = Data::draw(9, n, s.shape, s.noise);
            let sw = Sweep::new(&data, &s, &truth, 4);
            let pen = Sweep::new(&data, &Setup { mode: Mode::Penalty, degree: 7, ..s }, &truth, 4);
            assert_eq!(sw.fold_of, deal(n, k, 4));
            for f in 0..k {
                let keep: Vec<usize> = (0..n).filter(|&i| sw.fold_of[i] != f).collect();
                let held: Vec<usize> = (0..n).filter(|&i| sw.fold_of[i] == f).collect();
                let (x, y): (Vec<f64>, Vec<f64>) = keep.iter().map(|&i| (data.x[i], data.y[i])).unzip();
                for d in 0..=5 {
                    let g = raw_fit(&x, &y, d);
                    let miss = held.iter().map(|&i| (data.y[i] - g(data.x[i])).powi(2)).sum::<f64>() / held.len() as f64;
                    assert!(close(sw.folds[d][f], miss, 1e-7), "n {n}, fold {f}, degree {d}: {} against {miss}", sw.folds[d][f]);
                }
                let rows: Vec<[f64; P]> = keep.iter().map(|&i| data.rows[i]).collect();
                for size in [0, 9, 15, 21, 27] {
                    let c = ridge_fit(&rows, &y, 7, PENALTIES[size]);
                    let miss = held.iter().map(|&i| (data.y[i] - (0..8).map(|j| c[j] * data.rows[i][j]).sum::<f64>()).powi(2)).sum::<f64>() / held.len() as f64;
                    assert!(close(pen.folds[size][f], miss, 1e-6), "n {n}, fold {f}, penalty {}: {} against {miss}", PENALTIES[size], pen.folds[size][f]);
                }
            }
            // and the whole is every point's squared miss, averaged
            for d in 0..=5 {
                let pooled = (0..k).map(|f| sw.folds[d][f] * sw.fold_of.iter().filter(|&&g| g == f).count() as f64).sum::<f64>() / n as f64;
                assert!(close(sw.cv[d], pooled, 1e-12));
            }
        }
    }

    /// Leaving one point out: the shortcut (its miss in the fit to all, divided by one
    /// less its leverage) gives what fitting again without it gives.
    #[test]
    fn the_leave_one_out_shortcut_equals_fitting_again() {
        let truth = Truth::of(Shape::Step);
        for n in [8, 15, 40] {
            let s = Setup { shape: Shape::Step, n, loo: true, ..WAVE };
            let data = Data::draw(5, n, s.shape, s.noise);
            let sw = Sweep::new(&data, &s, &truth, 1);
            let pen = Sweep::new(&data, &Setup { mode: Mode::Penalty, degree: 9, ..s }, &truth, 1);
            assert_eq!(sw.k, n);
            for i in 0..n {
                let keep: Vec<usize> = (0..n).filter(|&j| j != i).collect();
                let (x, y): (Vec<f64>, Vec<f64>) = keep.iter().map(|&j| (data.x[j], data.y[j])).unzip();
                for d in 0..=(n - 2).min(5) {
                    let miss = (data.y[i] - raw_fit(&x, &y, d)(data.x[i])).powi(2);
                    assert!(close(sw.folds[d][i], miss, 1e-6), "n {n}, point {i}, degree {d}: {} against {miss}", sw.folds[d][i]);
                }
                // (and the degrees that the scene's own refit reaches)
                let without = Tri::of(&data, |j| j != i);
                for d in 6..=(n - 2).min(MAX_DEG) {
                    let c = without.solve(d).unwrap();
                    let miss = (data.y[i] - value(&c, &data.rows[i])).powi(2);
                    assert!(close(sw.folds[d][i], miss, 1e-5), "n {n}, point {i}, degree {d}: {} against {miss}", sw.folds[d][i]);
                }
                let rows: Vec<[f64; P]> = keep.iter().map(|&j| data.rows[j]).collect();
                for size in [0, 10, 20, 27] {
                    let c = ridge_fit(&rows, &y, 9, PENALTIES[size]);
                    let miss = (data.y[i] - (0..10).map(|j| c[j] * data.rows[i][j]).sum::<f64>()).powi(2);
                    assert!(close(pen.folds[size][i], miss, 1e-5), "n {n}, point {i}, penalty {}: {} against {miss}", PENALTIES[size], pen.folds[size][i]);
                }
            }
            // no degree beyond what n − 1 points can fit
            let last = (n - 1).min(P);
            assert!(sw.cv[..last].iter().all(|v| v.is_finite()) && sw.cv[last..].iter().all(|v| v.is_nan()), "n {n}: {:?}", sw.cv);
        }
    }

    /// The penalised fit is where the slope of the mean squared miss plus λ times the
    /// sum of the squared coefficients (all but the constant) is zero along every
    /// coefficient; with a vast penalty it is the flat line at the mean of y.
    #[test]
    fn the_penalty_is_what_the_panel_says() {
        let s = Setup { mode: Mode::Penalty, ..WAVE };
        let data = Data::draw(FIRST_SEED, s.n, s.shape, s.noise);
        let all = Tri::of(&data, |_| true);
        let n = data.len() as f64;
        for size in [0, 8, 16, 24, 27] {
            let c = fit(&all, &s, size).unwrap();
            let lambda = PENALTIES[size];
            for k in 0..=12 {
                let misses: f64 = (0..data.len()).map(|i| -2.0 * (data.y[i] - value(&c, &data.rows[i])) * data.rows[i][k]).sum::<f64>() / n;
                let penalty = if k == 0 { 0.0 } else { 2.0 * lambda * c[k] };
                assert!((misses + penalty).abs() < 1e-9 * (1.0 + misses.abs()), "λ {lambda}, coefficient {k}: slope {}", misses + penalty);
            }
        }
        let mut t = all.clone();
        for k in 1..=12 {
            t.add_penalty(k, 1e8);
        }
        let c = t.solve(12).unwrap();
        assert!((c[0] - mean(&data.y)).abs() < 1e-9 && c[1..].iter().all(|v| v.abs() < 1e-9), "{c:?}");
        // more coefficients than points: a penalty still makes the fit unique
        let few = Data::draw(3, 6, Shape::Wave, 0.3);
        let s = Setup { mode: Mode::Penalty, n: 6, degree: 15, ..WAVE };
        let sw = Sweep::new(&few, &s, &Truth::of(Shape::Wave), 1);
        assert!(sw.fits.iter().all(Option::is_some) && sw.cv.iter().chain(&sw.train).chain(&sw.truth).all(|v| v.is_finite()), "{:?}", sw.cv);
    }

    /// In the contest a line a + b z misses y by y − a − b z, with y of mean 0 and
    /// variance σ², z of mean 0 and variance 1, independent: σ² + a² + b² on average.
    #[test]
    fn a_useless_lines_true_error_is_what_the_panel_says() {
        let mut rng = Rng::new(11);
        let (a, b, sigma) = (0.3, -0.4, 0.8);
        let m = 400_000;
        let sum: f64 = (0..m).map(|_| (sigma * normal(&mut rng) - a - b * normal(&mut rng)).powi(2)).sum();
        let want = sigma * sigma + a * a + b * b;
        assert!((sum / m as f64 - want).abs() < 0.005 * want, "{} against {want}", sum / m as f64);
        let c = contest(4, 30, 1.0);
        for j in 0..LINES {
            let (a, b) = c.line[j];
            assert!(close(c.truth[j], 1.0 + a * a + b * b, 1e-12) && c.truth[j] > 1.0);
        }
        assert!(c.held.iter().all(|&h| h >= c.held[c.winner]));
    }

    #[test]
    fn too_stiff_too_wild() {
        let s = trial(0);
        assert_eq!((s.setup.mode, s.marked(), s.turning()), (Mode::Degree, 12, false));
        let sw = &s.sweep;
        let (train1, train12, cv1, cv12) = (sw.train[1], sw.train[12], sw.cv[1], sw.cv[12]);
        assert!((train12 - 0.061).abs() < 0.0005 && (train1 - 0.37).abs() < 0.005, "{train12} {train1}");
        assert!((train1 / train12 - 6.0).abs() < 0.25);
        assert!((cv12 / 110_000.0 - 1.0).abs() < 0.01 && (cv1 - 0.51).abs() < 0.005, "{cv12} {cv1}");
        // it bends to follow the points: closer to them than the noise is to the truth
        assert!(train12 < s.setup.noise * s.setup.noise);
        // and leaves the chart near the left end, though not the right
        let c = sw.fits[12].unwrap();
        let ys: Vec<f64> = s.data.y.iter().copied().chain(s.grid.iter().step_by(8).map(|(x, _)| s.setup.shape.at(*x))).collect();
        let [lo, hi] = padded(&ys, 0.08).map(|v| v as f64);
        let out = |x: f64| !(lo..=hi + 0.2 * (hi - lo)).contains(&value(&c, &legendre(x)));
        assert!((0..=20).any(|i| out(-1.0 + 0.01 * i as f64)), "it stays on the chart at the left");
        assert!(!(0..=20).any(|i| out(1.0 - 0.01 * i as f64)), "it leaves the chart at the right");
    }

    #[test]
    fn five_folds() {
        let s = trial(1);
        assert!(s.turning() && s.sweep.k == 5 && s.setup.n == 30);
        let sizes: Vec<usize> = (0..5).map(|f| s.sweep.fold_of.iter().filter(|&&g| g == f).count()).collect();
        assert_eq!(sizes, vec![6; 5]);
        assert_eq!((s.sweep.lowest(), s.sweep.best(), s.marked()), (Some(4), Some(4), 4));
        // over the first twenty deals of these points
        let mut at = [0; P];
        for deal in FIRST_DEAL..FIRST_DEAL + 20 {
            at[Sweep::new(&s.data, &s.setup, s.truth(), deal).lowest().unwrap()] += 1;
        }
        assert_eq!((at[4], at[3], at.iter().sum::<usize>()), (14, 6, 20), "{at:?}");
    }

    #[test]
    fn a_penalty_instead() {
        let s = trial(2);
        assert_eq!((s.setup.mode, s.setup.degree, s.marked()), (Mode::Penalty, 12, 0));
        let sw = &s.sweep;
        assert_eq!((PENALTIES[sw.lowest().unwrap()], PENALTIES[sw.best().unwrap()]), (0.02, 0.1));
        // too small to matter at the left: the same curve as without a penalty
        let plain = Sweep::new(&s.data, &Setup { mode: Mode::Degree, ..s.setup }, s.truth(), FIRST_DEAL);
        let (a, b) = (sw.fits[0].unwrap(), plain.fits[12].unwrap());
        let gap = s.grid.iter().map(|(_, r)| (value(&a, r) - value(&b, r)).abs()).fold(0.0, f64::max);
        let size = s.grid.iter().map(|(_, r)| value(&b, r).abs()).fold(0.0, f64::max);
        assert!(gap < 0.01 * size, "{gap} against {size}");
        // nearly flat at the right: its spread a hundredth of the truth's, or less
        let flat = sw.fits[PENALTIES.len() - 1].unwrap();
        let spread = flat[1..].iter().map(|c| c * c).sum::<f64>();
        let truth = s.truth();
        let truth_spread = truth.square - truth.coef[0] * truth.coef[0];
        assert!(spread < 0.01 * truth_spread, "{spread} against {truth_spread}");
        // high at both ends, lowest in between
        let last = PENALTIES.len() - 1;
        assert!(sw.cv[0] > 5.0 * sw.cv[16] && sw.cv[last] > 5.0 * sw.cv[16]);
        assert!(sw.truth[0] > 5.0 * sw.truth[18] && sw.truth[last] > 3.0 * sw.truth[18]);
    }

    #[test]
    fn more_data_a_bigger_model() {
        let mut s = trial(3);
        assert_eq!((s.setup.shape, s.setup.n), (Shape::Step, 300));
        let t = counted(&mut s);
        assert_eq!((spread(&t.chosen).0, t.best(), t.done), (8, Some(10), RUNS), "{:?}", t.chosen);
        s.setup.n = 30;
        s.refresh();
        let t = counted(&mut s);
        assert_eq!((spread(&t.chosen).0, t.best()), (3, Some(3)), "{:?}", t.chosen);
    }

    #[test]
    fn the_winners_curse() {
        let mut s = trial(4);
        assert_eq!((s.setup.mode, s.setup.noise, s.marked()), (Mode::Useless, 1.0, s.contest.winner));
        let c = &s.contest;
        let w = c.winner;
        let others = |v: &[f64]| v.iter().enumerate().filter(|(j, _)| *j != w).map(|(_, x)| *x).fold((f64::INFINITY, f64::NEG_INFINITY), |(a, b), x| (a.min(x), b.max(x)));
        let (t0, t1) = others(&c.truth);
        assert!((c.held[w] - 0.61).abs() < 0.005 && (c.truth[w] - 1.09).abs() < 0.005, "{} {}", c.held[w], c.truth[w]);
        assert!((t0 - 1.07).abs() < 0.005 && (t1 - 1.30).abs() < 0.005 && c.truth[w] > t0 && c.truth[w] < t1, "{t0} {t1}");
        let t = counted(&mut s);
        let (winner, random) = (mean(&t.gaps), mean(&t.random));
        assert!((winner - 0.12).abs() < 0.005 && (random + 0.03).abs() < 0.005, "{winner} {random}");
        // the line picked at random is out by nothing but chance: over many more contests, by nothing
        let many: Vec<f64> = (0..4000)
            .map(|k| {
                let c = contest(seed_of(k), 30, 1.0);
                let r = Rng::new(seed_of(k) ^ 0xA5A5).below(LINES);
                c.truth[r] - c.held[r]
            })
            .collect();
        let sd = crate::math::stats::sd(&many) / (many.len() as f64).sqrt();
        assert!(mean(&many).abs() < 2.0 * sd, "{} ± {sd}", mean(&many));
        // (and −0.03 is within what chance gives for 200)
        assert!(random.abs() < 2.5 * crate::math::stats::sd(&t.random) / (RUNS as f64).sqrt());
    }

    #[test]
    fn leave_one_out() {
        let mut s = trial(5);
        assert!(s.sweep.k == 30 && s.turning());
        let t = counted(&mut s);
        let ((_, _, loo), (_, _, five)) = (spread(&t.chosen), spread(&t.five));
        assert!((loo - 2.1).abs() < 0.05 && (five - 1.6).abs() < 0.05, "{loo} {five}");
        let high = |c: &[u32]| c[11..].iter().sum::<u32>();
        assert_eq!((high(&t.chosen), high(&t.five)), (8, 2));
        // nearly unbiased: on average over the same data sets its estimate is near the true
        // error of a fit to all 30 points; five folds, fitting 24, overestimate it
        let truth = Truth::of(Shape::Wave);
        let (mut l, mut f, mut e) = (0.0, 0.0, 0.0);
        for k in 0..RUNS {
            let data = Data::draw(seed_of(k), 30, Shape::Wave, 0.3);
            let deal = seed_of(k).wrapping_add(1);
            let loo = Sweep::new(&data, &s.setup, &truth, deal);
            let five = Sweep::new(&data, &WAVE, &truth, deal);
            (l, f, e) = (l + loo.cv[4], f + five.cv[4], e + loo.truth[4]);
        }
        let (l, f, e) = (l / RUNS as f64, f / RUNS as f64, e / RUNS as f64);
        println!("degree 4: leave one out {l}, five folds {f}, true {e}");
        assert!((l - 0.116).abs() < 0.0005 && (e - 0.116).abs() < 0.0005 && (f - 0.139).abs() < 0.0005, "{l} {f} {e}");
    }

    /// The scene in every mode and at the ends of its settings: every number drawn is a
    /// number, and the degree stops where the points run out.
    #[test]
    fn degenerate_settings_stay_finite() {
        let mut s = ModelSelect::create();
        for mode in MODES {
            for (n, folds, loo, noise) in [(6, 10, false, 0.3), (6, 5, false, 0.0), (7, 2, true, 1.0), (300, 10, false, 0.0), (30, 5, false, 0.0)] {
                s.setup = Setup { mode, n, folds, loo, noise, degree: 15, ..WAVE };
                s.at = [15, PENALTIES.len() - 1];
                s.turns = true;
                s.refresh();
                for fold in 0..s.sweep.k {
                    s.fold = fold;
                    let g = s.goals();
                    let numbers = g.points.iter().chain(&g.curve).chain(&g.left).chain(&g.series).chain(&g.folds).chain(&g.right).chain(&g.bars).chain([&g.mark, &g.top]);
                    assert!(numbers.into_iter().all(|v| v.is_finite()), "{mode:?} n {n} folds {folds} noise {noise}: {g:?}");
                    assert!(g.right[1] > g.right[0] && g.left[1] > g.left[0]);
                }
                if mode == Mode::Degree {
                    // more folds than points is leaving one out; the degree stops at n − 1
                    assert_eq!(s.sweep.k, if loo || folds >= n { n } else { folds });
                    assert_eq!(s.marked(), (n - 1).min(MAX_DEG));
                    let fold_fit = n - n.div_ceil(s.sweep.k);
                    let last_cv = (0..P).filter(|&d| s.sweep.cv[d].is_finite()).max();
                    assert_eq!(last_cv, Some((fold_fit - 1).min(MAX_DEG)), "n {n}: {:?}", s.sweep.cv);
                }
            }
        }
    }

    /// The numbers behind the captions (run with --ignored --nocapture).
    #[test]
    #[ignore]
    fn explore() {
        let s = opening();
        let truth = Truth::of(s.shape);
        let data = Data::draw(FIRST_SEED, s.n, s.shape, s.noise);
        let sw = Sweep::new(&data, &s, &truth, FIRST_DEAL);
        println!("lowest {:?} best {:?}\ntrain {:?}\ncv {:?}\ntruth {:?}", sw.lowest(), sw.best(), sw.train, sw.cv, sw.truth);
        for (name, s) in [("wave 30", opening()), ("step 30", Setup { shape: Shape::Step, ..opening() }), ("step 300", Setup { shape: Shape::Step, n: 300, ..opening() }), ("penalty", Setup { mode: Mode::Penalty, ..opening() })] {
            let truth = Truth::of(s.shape);
            let mut t = Tally::default();
            t.restart(&s);
            while t.pending() {
                t.step(&s, &truth);
            }
            println!("{name}: chosen {:?} best {:?} spread {:?}", t.chosen, t.best(), spread(&t.chosen));
        }
    }
}

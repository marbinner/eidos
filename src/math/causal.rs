//! Statistics for the causal inference scenes: fitting, what a fit leaves over, and
//! measures of dependence. Everything is in f64 and works on plain slices of samples
//! (one number per unit; slices passed together are as long as each other, and what
//! is beyond the shortest is ignored).

use super::linalg::solve;
use super::rng::Rng;

pub fn mean(x: &[f64]) -> f64 {
    if x.is_empty() {
        return 0.0;
    }
    x.iter().sum::<f64>() / x.len() as f64
}

/// The mean squared distance from the mean (dividing by n, not n − 1).
pub fn variance(x: &[f64]) -> f64 {
    covariance(x, x)
}

pub fn covariance(x: &[f64], y: &[f64]) -> f64 {
    let n = x.len().min(y.len());
    if n == 0 {
        return 0.0;
    }
    let (mx, my) = (mean(&x[..n]), mean(&y[..n]));
    x.iter().zip(y).map(|(a, b)| (a - mx) * (b - my)).sum::<f64>() / n as f64
}

/// The correlation coefficient, from −1 to 1; 0 if either has no spread at all.
pub fn correlation(x: &[f64], y: &[f64]) -> f64 {
    let spread = (variance(x) * variance(y)).sqrt();
    if spread > 0.0 && spread.is_finite() {
        (covariance(x, y) / spread).max(-1.0).min(1.0)
    } else {
        0.0
    }
}

/// The least-squares line through the points: (slope, intercept) of y ≈ slope x +
/// intercept. A level line through the mean if x has no spread.
pub fn line(x: &[f64], y: &[f64]) -> (f64, f64) {
    let vx = variance(x);
    let slope = if vx > 0.0 { covariance(x, y) / vx } else { 0.0 };
    (slope, mean(y) - slope * mean(x))
}

/// Least squares on several regressors at once, with an intercept: one coefficient
/// per column and then the intercept. None if the columns do not settle them (one is
/// a combination of the others, or there are too few samples).
pub fn regress(columns: &[&[f64]], y: &[f64]) -> Option<Vec<f64>> {
    let k = columns.len() + 1;
    let n = columns.iter().map(|c| c.len()).fold(y.len(), usize::min);
    if n < k {
        return None;
    }
    // the normal equations, with a column of ones last
    let at = |j: usize, i: usize| if j + 1 == k { 1.0 } else { columns[j][i] };
    let (mut a, mut b) = (vec![0.0; k * k], vec![0.0; k]);
    for i in 0..n {
        for r in 0..k {
            b[r] += at(r, i) * y[i];
            for c in 0..k {
                a[r * k + c] += at(r, i) * at(c, i);
            }
        }
    }
    solve(&mut a, &mut b, k).filter(|w| w.iter().all(|v| v.is_finite()))
}

/// What is left of y once the best combination of the columns (and a constant) has
/// been taken away. With no columns, or columns that settle nothing, y less its mean.
pub fn residuals(columns: &[&[f64]], y: &[f64]) -> Vec<f64> {
    match regress(columns, y) {
        Some(w) => {
            (0..y.len()).map(|i| y[i] - w[columns.len()] - columns.iter().zip(&w).map(|(c, w)| c.get(i).copied().unwrap_or(0.0) * w).sum::<f64>()).collect()
        }
        None => {
            let m = mean(y);
            y.iter().map(|v| v - m).collect()
        }
    }
}

/// The correlation of x and y once what the `given` variables explain of each (by a
/// straight-line fit) has been taken away. For normal variables, 0 exactly when x and
/// y are independent given the others.
pub fn partial_correlation(x: &[f64], y: &[f64], given: &[&[f64]]) -> f64 {
    correlation(&residuals(given, x), &residuals(given, y))
}

/// A smooth curve through a scatter, with no shape assumed: the estimate of the mean
/// of y at `at` is the average of the ys, each weighted by how near its x is (a bell
/// of width `h`). The mean of y if nothing is near enough to weigh anything.
pub fn smooth_at(x: &[f64], y: &[f64], at: f64, h: f64) -> f64 {
    let h = if h > 0.0 { h } else { 1.0 };
    let (mut top, mut bottom) = (0.0, 0.0);
    for (a, b) in x.iter().zip(y) {
        let w = (-0.5 * ((a - at) / h).powi(2)).exp();
        top += w * b;
        bottom += w;
    }
    if bottom > 1e-300 {
        top / bottom
    } else {
        mean(y)
    }
}

/// That curve's value at every sample's own x.
pub fn smooth(x: &[f64], y: &[f64], h: f64) -> Vec<f64> {
    x.iter().map(|at| smooth_at(x, y, *at, h)).collect()
}

/// A width for `smooth` that suits the samples: a share of their spread that shrinks
/// slowly as they grow in number (the rule of thumb n^(−1/5)).
pub fn bandwidth(x: &[f64]) -> f64 {
    let sd = variance(x).sqrt();
    if sd > 0.0 && sd.is_finite() {
        0.6 * sd * (x.len().max(2) as f64).powf(-0.2)
    } else {
        1.0
    }
}

/// How far a sample's shape is from a bell's in its tails: 0 for a normal sample,
/// negative for flatter ones (−1.2 for a uniform), positive for peaked ones with
/// heavy tails (3 for a Laplace).
pub fn kurtosis(x: &[f64]) -> f64 {
    let (m, v) = (mean(x), variance(x));
    if x.is_empty() || v <= 0.0 {
        return 0.0;
    }
    x.iter().map(|a| (a - m).powi(4)).sum::<f64>() / x.len() as f64 / (v * v) - 3.0
}

/// A typical distance between two of the samples (the median of the distances among
/// up to 200 of them, evenly picked): the width of the bells the dependence measure
/// compares them with.
fn typical_distance(x: &[f64]) -> f64 {
    let step = x.len().div_ceil(200).max(1);
    let picked: Vec<f64> = x.iter().step_by(step).copied().collect();
    let mut d: Vec<f64> = Vec::with_capacity(picked.len() * picked.len() / 2);
    for i in 0..picked.len() {
        for j in 0..i {
            d.push((picked[i] - picked[j]).abs());
        }
    }
    d.retain(|v| v.is_finite() && *v > 0.0);
    if d.is_empty() {
        return 1.0;
    }
    let mid = d.len() / 2;
    *d.select_nth_unstable_by(mid, f64::total_cmp).1
}

/// A measure of dependence that sees every kind, not only a straight-line trend: the
/// Hilbert–Schmidt independence criterion of Gretton and others (2005), with bell
/// kernels as wide as a typical distance between samples. It is 0 (but for sampling
/// noise) exactly when the two are independent, and larger the more one says about
/// the other. Its size has no natural unit: compare it with what the same samples
/// give when their pairing is shuffled (`shuffled`, `p_value`).
///
/// It keeps two n × n tables, so it is for hundreds of samples, not many thousands.
pub struct Hsic {
    n: usize,
    /// The first variable's table of likenesses, centred (row and column means removed).
    k: Vec<f64>,
    /// The second variable's, as it is.
    l: Vec<f64>,
}

impl Hsic {
    pub fn new(x: &[f64], y: &[f64]) -> Hsic {
        let n = x.len().min(y.len());
        let table = |v: &[f64]| {
            let s = typical_distance(&v[..n]);
            let mut t = vec![0.0; n * n];
            for i in 0..n {
                for j in 0..n {
                    t[i * n + j] = (-0.5 * ((v[i] - v[j]) / s).powi(2)).exp();
                }
            }
            t
        };
        let (mut k, l) = (table(x), table(y));
        // centre: take each row's mean and each column's mean away, and add the mean of all back
        if n > 0 {
            let rows: Vec<f64> = (0..n).map(|i| k[i * n..(i + 1) * n].iter().sum::<f64>() / n as f64).collect();
            let all = rows.iter().sum::<f64>() / n as f64;
            for i in 0..n {
                for j in 0..n {
                    k[i * n + j] += all - rows[i] - rows[j];
                }
            }
        }
        Hsic { n, k, l }
    }

    /// The measure for the samples as they are paired.
    pub fn value(&self) -> f64 {
        let n = self.n;
        if n == 0 {
            return 0.0;
        }
        self.k.iter().zip(&self.l).map(|(a, b)| a * b).sum::<f64>() / (n * n) as f64
    }

    /// The measure with the second variable's samples dealt out again at random among
    /// the first's: what independence looks like for samples shaped like these.
    pub fn shuffled(&self, rng: &mut Rng) -> f64 {
        let n = self.n;
        if n == 0 {
            return 0.0;
        }
        let mut order: Vec<usize> = (0..n).collect();
        rng.shuffle(&mut order);
        let mut sum = 0.0;
        for i in 0..n {
            let (row, to) = (&self.k[i * n..(i + 1) * n], order[i] * n);
            for j in 0..n {
                sum += row[j] * self.l[to + order[j]];
            }
        }
        sum / (n * n) as f64
    }

    /// The share of `shuffles` random re-pairings that look at least as dependent as
    /// the true pairing: small (below 0.05, say) is evidence of dependence, anything
    /// else is what independent samples would give. Never exactly 0: with s shuffles
    /// it cannot say less than 1 / (s + 1).
    pub fn p_value(&self, shuffles: usize, rng: &mut Rng) -> f64 {
        let seen = self.value();
        let as_large = (0..shuffles).filter(|_| self.shuffled(rng) >= seen).count();
        (as_large + 1) as f64 / (shuffles + 1) as f64
    }
}

/// The dependence measure alone (see [`Hsic`]).
pub fn hsic(x: &[f64], y: &[f64]) -> f64 {
    Hsic::new(x, y).value()
}

/// Shapes of noise, each with mean 0 and variance 1, so that swapping one for another
/// changes the shape and nothing else.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Noise {
    /// The bell curve (Gaussian).
    Normal,
    /// Every value between −√3 and √3 equally likely: flat, with hard edges.
    Uniform,
    /// A sharp peak with long tails.
    Laplace,
    /// Two bells side by side, at −0.9 and 0.9.
    TwoHumps,
}

impl Noise {
    pub const ALL: [Noise; 4] = [Noise::Normal, Noise::Uniform, Noise::Laplace, Noise::TwoHumps];

    pub fn name(self) -> &'static str {
        match self {
            Noise::Normal => "bell (normal)",
            Noise::Uniform => "flat (uniform)",
            Noise::Laplace => "peaked (Laplace)",
            Noise::TwoHumps => "two humps",
        }
    }

    pub fn sample(self, rng: &mut Rng) -> f64 {
        match self {
            Noise::Normal => rng.normal() as f64,
            Noise::Uniform => (rng.f32() as f64 - 0.5) * 12f64.sqrt(),
            // (the difference of two exponentials, scaled to variance 1)
            Noise::Laplace => (rng.exponential() as f64 - rng.exponential() as f64) * 0.5f64.sqrt(),
            Noise::TwoHumps => {
                let side = if rng.bool(0.5) { 0.9 } else { -0.9 };
                side + rng.normal() as f64 * (1.0f64 - 0.81).sqrt()
            }
        }
    }

    /// Its probability density at `v`.
    pub fn density(self, v: f64) -> f64 {
        let bell = |v: f64, sd: f64| (-0.5 * (v / sd).powi(2)).exp() / (sd * std::f64::consts::TAU.sqrt());
        match self {
            Noise::Normal => bell(v, 1.0),
            Noise::Uniform => {
                let half = 3f64.sqrt();
                if v.abs() <= half {
                    0.5 / half
                } else {
                    0.0
                }
            }
            Noise::Laplace => {
                let b = 0.5f64.sqrt();
                (-v.abs() / b).exp() / (2.0 * b)
            }
            Noise::TwoHumps => {
                let sd = (1.0f64 - 0.81).sqrt();
                0.5 * (bell(v - 0.9, sd) + bell(v + 0.9, sd))
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn samples(noise: Noise, n: usize, seed: u64) -> Vec<f64> {
        let mut rng = Rng::new(seed);
        (0..n).map(|_| noise.sample(&mut rng)).collect()
    }

    /// Every shape of noise has mean 0 and variance 1, the tails its name promises,
    /// and a density that adds up to 1 and agrees with what is sampled.
    #[test]
    fn the_noises_differ_in_shape_only() {
        for (noise, tails) in [(Noise::Normal, 0.0), (Noise::Uniform, -1.2), (Noise::Laplace, 3.0), (Noise::TwoHumps, -1.34)] {
            let x = samples(noise, 200_000, 7);
            assert!(mean(&x).abs() < 0.01, "{noise:?}: mean {}", mean(&x));
            assert!((variance(&x) - 1.0).abs() < 0.02, "{noise:?}: variance {}", variance(&x));
            assert!((kurtosis(&x) - tails).abs() < 0.15, "{noise:?}: kurtosis {}", kurtosis(&x));
            let (steps, width) = (4000, 16.0);
            let h = width / steps as f64;
            let at = |i: usize| -width / 2.0 + (i as f64 + 0.5) * h;
            let area: f64 = (0..steps).map(|i| noise.density(at(i)) * h).sum();
            let second: f64 = (0..steps).map(|i| at(i) * at(i) * noise.density(at(i)) * h).sum();
            assert!((area - 1.0).abs() < 2e-3 && (second - 1.0).abs() < 2e-3, "{noise:?}: density area {area}, variance {second}");
            // and the share of samples within half a unit of 0 is the density's area there
            let near = x.iter().filter(|v| v.abs() < 0.5).count() as f64 / x.len() as f64;
            let said: f64 = (0..steps).filter(|i| at(*i).abs() < 0.5).map(|i| noise.density(at(i)) * h).sum();
            assert!((near - said).abs() < 0.01, "{noise:?}: {near} of the samples near 0, the density says {said}");
        }
    }

    #[test]
    fn lines_and_planes_are_fitted_exactly() {
        let x: Vec<f64> = (0..50).map(|i| i as f64 * 0.1 - 2.0).collect();
        let z: Vec<f64> = (0..50).map(|i| ((i * 7) % 11) as f64 - 5.0).collect();
        let y: Vec<f64> = x.iter().map(|v| 2.5 * v - 1.0).collect();
        let (slope, intercept) = line(&x, &y);
        assert!((slope - 2.5).abs() < 1e-12 && (intercept + 1.0).abs() < 1e-12);
        assert!((correlation(&x, &y) - 1.0).abs() < 1e-12);
        let both: Vec<f64> = x.iter().zip(&z).map(|(a, b)| 2.0 * a - 3.0 * b + 1.0).collect();
        let w = regress(&[&x, &z], &both).unwrap();
        assert!((w[0] - 2.0).abs() < 1e-9 && (w[1] + 3.0).abs() < 1e-9 && (w[2] - 1.0).abs() < 1e-9, "{w:?}");
        assert!(residuals(&[&x, &z], &both).iter().all(|r| r.abs() < 1e-9));
        // nothing to go on: no fit, no panic, and what is left is y about its mean
        assert!(regress(&[&x, &x], &y).is_none());
        assert!(residuals(&[&x, &x], &y).iter().sum::<f64>().abs() < 1e-9);
        assert_eq!(line(&[1.0, 1.0, 1.0], &[2.0, 4.0, 6.0]), (0.0, 4.0));
        assert_eq!(correlation(&[1.0, 1.0], &[2.0, 3.0]), 0.0);
        assert_eq!((mean(&[]), variance(&[]), kurtosis(&[]), hsic(&[], &[])), (0.0, 0.0, 0.0, 0.0));
    }

    /// In a chain x → m → y, x and y go together, and not at all once m is held.
    #[test]
    fn holding_the_middle_of_a_chain_parts_its_ends() {
        let mut rng = Rng::new(3);
        let n = 20_000;
        let x: Vec<f64> = (0..n).map(|_| rng.normal() as f64).collect();
        let m: Vec<f64> = x.iter().map(|v| 0.8 * v + rng.normal() as f64).collect();
        let y: Vec<f64> = m.iter().map(|v| 0.8 * v + rng.normal() as f64).collect();
        assert!(correlation(&x, &y) > 0.35, "{}", correlation(&x, &y));
        assert!(partial_correlation(&x, &y, &[&m]).abs() < 0.02, "{}", partial_correlation(&x, &y, &[&m]));
        assert!((partial_correlation(&x, &y, &[]) - correlation(&x, &y)).abs() < 1e-12);
    }

    /// The smooth curve follows a bend that a line cannot.
    #[test]
    fn the_smoother_follows_a_curve() {
        let mut rng = Rng::new(5);
        let x: Vec<f64> = (0..600).map(|_| rng.range(-2.0, 2.0) as f64).collect();
        let y: Vec<f64> = x.iter().map(|v| v * v + 0.1 * rng.normal() as f64).collect();
        let h = bandwidth(&x);
        for at in [-1.5, -0.5, 0.0, 1.0] {
            assert!((smooth_at(&x, &y, at, h) - at * at).abs() < 0.12, "at {at}: {}", smooth_at(&x, &y, at, h));
        }
        assert!(line(&x, &y).0.abs() < 0.15);
        assert_eq!(smooth(&x, &y, h).len(), x.len());
        assert!(smooth_at(&x, &y, 1e9, h).is_finite() && smooth_at(&x, &y, 0.0, 0.0).is_finite());
    }

    /// The dependence measure tells a parabola (no straight-line trend at all) from
    /// independence, and does not cry wolf at independent samples.
    #[test]
    fn dependence_is_seen_where_correlation_sees_none() {
        let mut rng = Rng::new(11);
        let n = 300;
        let x: Vec<f64> = (0..n).map(|_| rng.normal() as f64).collect();
        let y: Vec<f64> = x.iter().map(|v| v * v + 0.3 * rng.normal() as f64).collect();
        let z: Vec<f64> = (0..n).map(|_| rng.normal() as f64).collect();
        assert!(correlation(&x, &y).abs() < 0.2);
        let (bent, apart) = (Hsic::new(&x, &y), Hsic::new(&x, &z));
        assert!(bent.value() > 10.0 * apart.value(), "{} against {}", bent.value(), apart.value());
        assert!(bent.p_value(200, &mut rng) < 0.01);
        assert!(apart.p_value(200, &mut rng) > 0.05);
        // shuffling keeps the measure in the range of independent samples
        let worst = (0..50).map(|_| bent.shuffled(&mut rng)).fold(0.0, f64::max);
        assert!(worst < bent.value() / 5.0, "{worst} against {}", bent.value());
        assert!((hsic(&x, &y) - bent.value()).abs() < 1e-15);
    }

    /// What the whole of causal discovery by residuals rests on: a straight-line fit
    /// in the causal direction leaves noise independent of the input; in the other
    /// direction it leaves something uncorrelated with its input but dependent on
    /// it, unless the noise is bell-shaped, when nothing tells the two apart.
    #[test]
    fn residuals_are_independent_one_way_only_unless_the_noise_is_normal() {
        let n = 400;
        for (noise, told) in [(Noise::Uniform, true), (Noise::Normal, false)] {
            let mut rng = Rng::new(17);
            let x: Vec<f64> = (0..n).map(|_| noise.sample(&mut rng)).collect();
            let y: Vec<f64> = x.iter().map(|v| v + noise.sample(&mut rng)).collect();
            let (forwards, backwards) = (residuals(&[&x], &y), residuals(&[&y], &x));
            assert!(correlation(&forwards, &x).abs() < 1e-9 && correlation(&backwards, &y).abs() < 1e-9);
            let (right, wrong) = (Hsic::new(&x, &forwards).p_value(200, &mut rng), Hsic::new(&y, &backwards).p_value(200, &mut rng));
            assert!(right > 0.05, "{noise:?}: the causal direction's residuals look dependent (p = {right})");
            assert_eq!(wrong < 0.01, told, "{noise:?}: against the arrow, p = {wrong}");
        }
    }
}

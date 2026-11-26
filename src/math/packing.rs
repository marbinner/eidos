//! Spreading directions apart: unit vectors placed so that the largest |dot product|
//! between any two is as small as it can be (for the packing scene).
//!
//! A direction and its opposite count as the same here (the dot product only changes
//! sign), so each vector stands for a line through the origin, and the problem is to
//! pack lines: to make the smallest angle between any two of them as large as possible.
//! The largest |dot product| is the cosine of that angle, and is called the coherence
//! of the set.

use crate::math::rng::Rng;
use glam::DVec3;

/// The best arrangements known of n lines through the origin of space, for n = 3 to 55:
/// sin² of the smallest angle between two of the lines. From N. J. A. Sloane's tables
/// of packings in Grassmannian spaces (neilsloane.com/grass, the rows for m = 3,
/// n = 1), the numbers behind Table 1 of Conway, Hardin and Sloane, "Packing lines,
/// planes, etc." (Experimental Mathematics, 1996). Those for n ≤ 6 were proved best
/// by Fejes Tóth in 1965, and n = 7 in that paper; the rest are the best found.
const BEST_SPACE: [f64; 53] = [
    1.000000000000000,
    0.888888888888889,
    0.799999999999999,
    0.800000000000000,
    0.666666666666667,
    0.580628514621758,
    0.551954085526548,
    0.529210992451761,
    0.489583348932409,
    0.445688721477877,
    0.409964971108470,
    0.390628992130893,
    0.381325614173148,
    0.368524269666694,
    0.332854698888097,
    0.319330521178142,
    0.300009387775088,
    0.291972234676310,
    0.284209946852422,
    0.279189472772918,
    0.257690403520805,
    0.252464036071819,
    0.238729837953039,
    0.230800083465611,
    0.224023578236251,
    0.218197811785817,
    0.213560845613763,
    0.206083829524788,
    0.199074255890140,
    0.191478238089024,
    0.186355763864167,
    0.182051475639965,
    0.176873541869869,
    0.172969977394539,
    0.169159004330758,
    0.166585408506718,
    0.163407377655453,
    0.156827559568762,
    0.152564470649171,
    0.149016165933882,
    0.146919917606886,
    0.142778750069872,
    0.140913249295026,
    0.138426779180331,
    0.136244207001001,
    0.133922710127452,
    0.130314072572911,
    0.127171795359092,
    0.124832568762863,
    0.122547134450475,
    0.120006865673398,
    0.118727643427711,
    0.118140805167447,
];

/// The most lines the table of best known arrangements covers.
pub const MOST: usize = 55;

/// The lowest coherence known for `n` lines in `dim` dimensions (2 or 3), and whether
/// it has been proved that none is lower. In the plane the answer is simple: the lines
/// a half turn divided into n equal angles, cos(π/n).
pub fn best_known(n: usize, dim: usize) -> Option<(f64, bool)> {
    if n <= dim {
        // as many lines as dimensions or fewer: all at right angles
        return Some((0.0, true));
    }
    if dim == 2 {
        return Some(((std::f64::consts::PI / n as f64).cos(), true));
    }
    let sin2 = *BEST_SPACE.get(n.checked_sub(3)?)?;
    Some(((1.0 - sin2).max(0.0).sqrt(), n <= 7))
}

/// The Welch bound: n unit vectors in `dim` dimensions cannot all have |dot products|
/// below √((n − dim) / (dim (n − 1))). (Their squared dot products add up to at least
/// n²/dim, the n of them with themselves included; that leaves at least this much, on
/// average, for each of the others.)
pub fn welch(n: usize, dim: usize) -> f64 {
    if n <= dim {
        return 0.0;
    }
    ((n - dim) as f64 / (dim * (n - 1)) as f64).sqrt()
}

/// A second floor, for crowds: more than dim (dim + 1) / 2 unit vectors in `dim`
/// dimensions always include two with |dot product| at least 1/√dim (the orthoplex
/// bound, from Rankin's bound for points on a sphere).
pub fn orthoplex(n: usize, dim: usize) -> f64 {
    if n > dim * (dim + 1) / 2 {
        1.0 / (dim as f64).sqrt()
    } else {
        0.0
    }
}

/// How sharp the soft maximum starts, how sharp it is when a run resumes after a
/// disturbance, and how sharp it ends: at the end it is within ln(pairs) / SHARPEST
/// of the true maximum of the squared dot products.
const SOFTEST: f64 = 2.0;
const RESUMED: f64 = 64.0;
const SHARPEST: f64 = 1.0e7;

/// By how much the soft maximum is sharpened each time the vectors have come to rest,
/// and the slope (of the soft maximum, per radian turned) below which they count as at
/// rest.
const SHARPEN: f64 = 2.0;
const AT_REST: f64 = 1e-9;

/// No run is longer than this many steps (none comes near it).
const LONGEST: u32 = 5000;

/// A curvature below minus this, where the vectors rest, means a saddle; then the
/// vector that moves most is turned this far (radians) to get off it; and a run does
/// that at most this often.
const SADDLE: f64 = 1e-6;
const OFF_A_SADDLE: f64 = 2e-3;
const MOST_SADDLES: u32 = 24;
const SADDLES_FROM: f64 = 1.0e4;

/// The furthest any one vector turns in a single step, in radians: small enough for
/// the steps to be watched as a motion.
pub const REACH: f64 = 0.012;

/// A set of unit vectors on its way to being spread apart.
///
/// The largest |dot product| is not a smooth function of the vectors (which pair is
/// the worst keeps changing), so what is minimised is a soft maximum of the squared
/// dot products, (1/β) ln Σ exp(β (vᵢ·vⱼ)²): smooth, and within ln(pairs)/β of the
/// largest. Each step is a Newton step on it, damped so that no vector turns by more
/// than `REACH`; whenever the vectors have come to rest for the current β it is
/// doubled, up to `SHARPEST`, where the soft maximum and the true one agree to seven
/// decimals. Where the run ends is a local best: an arrangement no small change
/// improves. It need not be the best of all.
pub struct Spread {
    /// The vectors, each of length 1 (in the plane: with z = 0).
    pub v: Vec<DVec3>,
    dim: usize,
    beta: f64,
    damp: f64,
    kicks: u32,
    steps: u32,
    settled: bool,
}

impl Spread {
    /// `n` directions drawn at random (evenly over the circle or the sphere) from `seed`.
    pub fn new(n: usize, dim: usize, seed: u64) -> Spread {
        Spread::start(n, dim, seed, SOFTEST)
    }

    /// The same, with the soft maximum as sharp as `sharpness` from the first step. A
    /// soft start lets every pair push on every other and leads most starts to the same
    /// few, very regular arrangements; a sharp start only ever moves the closest pairs,
    /// and ends in more varied places. For some numbers of lines the best arrangement
    /// known is only found the second way.
    pub fn start(n: usize, dim: usize, seed: u64, sharpness: f64) -> Spread {
        let dim = dim.clamp(2, 3);
        let mut rng = Rng::new(seed);
        let v = (0..n)
            .map(|_| loop {
                let p = DVec3::new(rng.normal() as f64, rng.normal() as f64, if dim == 3 { rng.normal() as f64 } else { 0.0 });
                if p.length_squared() > 1e-6 {
                    break p.normalize();
                }
            })
            .collect();
        Spread { v, dim, beta: sharpness.clamp(SOFTEST, SHARPEST), damp: 0.0, kicks: 0, steps: 0, settled: n < 2 }
    }

    pub fn dim(&self) -> usize {
        self.dim
    }

    /// Steps taken since the start (or since the last disturbance).
    pub fn steps(&self) -> u32 {
        self.steps
    }

    /// Has the run come to rest at a local best?
    pub fn settled(&self) -> bool {
        self.settled
    }

    /// The coherence: the largest |dot product| between two of the vectors.
    pub fn coherence(&self) -> f64 {
        coherence(&self.v)
    }

    /// Put vector `i` somewhere else (a visitor dragging it), and take the run up again
    /// from there with a fairly sharp maximum, so that the rest is not rearranged first.
    pub fn place(&mut self, i: usize, to: DVec3) {
        let mut to = if self.dim == 2 { DVec3::new(to.x, to.y, 0.0) } else { to };
        if !to.is_finite() || to.length_squared() < 1e-12 {
            return;
        }
        to = to.normalize();
        if let Some(v) = self.v.get_mut(i) {
            *v = to;
            self.wake();
        }
    }

    /// Take the run up again (after the vectors were changed from outside).
    pub fn wake(&mut self) {
        (self.beta, self.damp, self.kicks, self.steps) = (RESUMED, 0.0, 0, 0);
        self.settled = self.v.len() < 2;
    }

    /// The directions each vector can move in and stay on the sphere (or the circle):
    /// unit vectors at right angles to it and to each other.
    fn tangents(&self) -> Vec<[DVec3; 2]> {
        self.v
            .iter()
            .map(|v| {
                if self.dim == 2 {
                    [DVec3::new(-v.y, v.x, 0.0), DVec3::ZERO]
                } else {
                    let (a, b) = v.any_orthonormal_pair();
                    [a, b]
                }
            })
            .collect()
    }

    /// The soft maximum of the squared dot products of `v`, at sharpness `beta`.
    fn soft_max(v: &[DVec3], beta: f64) -> f64 {
        let mut squares = Vec::with_capacity(v.len() * v.len() / 2);
        for i in 0..v.len() {
            for j in i + 1..v.len() {
                let c = v[i].dot(v[j]);
                squares.push(c * c);
            }
        }
        let top = squares.iter().fold(0.0f64, |a, b| a.max(*b));
        top + squares.iter().map(|f| (beta * (f - top)).exp()).sum::<f64>().ln() / beta
    }

    /// The vectors after each has moved by its part of `step` along its tangents.
    fn moved(&self, tangents: &[[DVec3; 2]], step: &[f64]) -> Vec<DVec3> {
        let t = self.dim - 1;
        self.v.iter().zip(tangents).enumerate().map(|(i, (v, b))| (*v + (0..t).map(|a| b[a] * step[i * t + a]).sum::<DVec3>()).normalize()).collect()
    }

    /// The soft maximum at the present sharpness, with its slope and its curvature in
    /// the coordinates of `tangents` (a matrix by rows).
    fn shape(&self, tangents: &[[DVec3; 2]]) -> (f64, Vec<f64>, Vec<f64>) {
        let (n, t) = (self.v.len(), self.dim - 1);
        let m = n * t;
        let beta = self.beta;

        // every pair's weight in the soft maximum
        let mut pairs = Vec::with_capacity(n * (n - 1) / 2);
        for i in 0..n {
            for j in i + 1..n {
                pairs.push((i, j, self.v[i].dot(self.v[j])));
            }
        }
        let top = pairs.iter().fold(0.0f64, |a, p| a.max(p.2 * p.2));
        let total: f64 = pairs.iter().map(|p| (beta * (p.2 * p.2 - top)).exp()).sum();
        let now = top + total.ln() / beta;

        // its slope and curvature in the tangent coordinates. To second order a vector
        // moved by u along its tangents B is v + Bu − ½|u|²v, so for c = vᵢ·vⱼ the
        // slope is (Bᵢᵀvⱼ, Bⱼᵀvᵢ) and the curvature is −c on the two diagonal blocks
        // and BᵢᵀBⱼ off them.
        let mut slope = vec![0.0f64; m];
        let mut curve = vec![0.0f64; m * m];
        for &(i, j, c) in &pairs {
            let p = (beta * (c * c - top)).exp() / total;
            if p < 1e-30 {
                continue;
            }
            let mut dc = [0.0f64; 4];
            for a in 0..t {
                dc[a] = tangents[i][a].dot(self.v[j]);
                dc[t + a] = tangents[j][a].dot(self.v[i]);
            }
            let at = |k: usize| if k < t { i * t + k } else { j * t + k - t };
            for k in 0..2 * t {
                slope[at(k)] += p * 2.0 * c * dc[k];
                for l in 0..2 * t {
                    // of c²: 2 ∇c ∇cᵀ + 2c ∇²c; and the soft maximum's own β p ∇f ∇fᵀ
                    let second = if k == l {
                        -c
                    } else if (k < t) != (l < t) {
                        let (a, b) = if k < t { (k, l - t) } else { (l, k - t) };
                        tangents[i][a].dot(tangents[j][b])
                    } else {
                        0.0
                    };
                    curve[at(k) * m + at(l)] += p * (2.0 * dc[k] * dc[l] + 2.0 * c * second) + beta * p * 4.0 * c * c * dc[k] * dc[l];
                }
            }
        }
        for k in 0..m {
            for l in 0..m {
                curve[k * m + l] -= beta * slope[k] * slope[l];
            }
        }
        (now, slope, curve)
    }

    /// If the vectors rest on a saddle (the curvature `curve` falls away in some
    /// direction), that direction. Turning all the vectors together changes nothing, so
    /// some curvatures are zero but for rounding: only one clearly below zero counts.
    fn way_down(&self, curve: &[f64], m: usize) -> Option<Vec<f64>> {
        if self.kicks >= MOST_SADDLES || self.beta < SADDLES_FROM {
            return None;
        }
        let mut lifted = curve.to_vec();
        for k in 0..m {
            lifted[k * m + k] += SADDLE;
        }
        if cholesky(&mut lifted, m, 0.0) {
            return None;
        }
        let (values, vectors) = crate::math::linalg::eig_sym(curve, m);
        (values.last().is_some_and(|l| *l < -SADDLE)).then(|| vectors[(m - 1) * m..].to_vec())
    }

    /// One step. Returns false once the run has settled (and then does nothing).
    pub fn step(&mut self) -> bool {
        if self.settled {
            return false;
        }
        let (n, t) = (self.v.len(), self.dim - 1);
        let m = n * t;
        let tangents = self.tangents();
        self.steps += 1;
        if self.steps > LONGEST {
            self.settled = true;
            return false;
        }
        let beta = self.beta;
        let (now, slope, curve) = self.shape(&tangents);

        // at rest for this sharpness (no slope left): sharpen and go on, or, at the
        // sharpest, stop
        // (the weights are exponentials of β times a difference of rounded numbers, so
        // the slope is only known to about β × 10⁻¹⁵)
        let rest = AT_REST.max(1e-14 * beta);
        if slope.iter().all(|g| g.abs() < rest) {
            // at rest: at the bottom of a bowl, or balanced on a saddle? (Very regular
            // arrangements often are saddles: breaking the symmetry helps.) A step
            // along the way the curvature falls, to whichever side is lower, and on.
            if let Some(way) = self.way_down(&curve, m) {
                let size = (0..n).map(|i| (0..t).map(|a| way[i * t + a] * way[i * t + a]).sum::<f64>().sqrt()).fold(1e-300f64, f64::max);
                let moves = [1.0, -1.0].map(|sign| self.moved(&tangents, &way.iter().map(|w| w * sign * OFF_A_SADDLE / size).collect::<Vec<_>>()));
                let heights = [Self::soft_max(&moves[0], beta), Self::soft_max(&moves[1], beta)];
                let [first, second] = moves;
                let (lower, height) = if heights[0] <= heights[1] { (first, heights[0]) } else { (second, heights[1]) };
                if height < now {
                    self.v = lower;
                    self.kicks += 1;
                    return true;
                }
            }
            if self.beta >= SHARPEST {
                self.settled = true;
                return false;
            }
            self.beta = (self.beta * SHARPEN).min(SHARPEST);
            return true;
        }

        // a Newton step, damped until the curvature is that of a bowl, no vector turns
        // further than REACH, and the soft maximum really comes down
        let scale = (0..m).map(|k| curve[k * m + k].abs()).fold(1e-12f64, f64::max);
        let mut damp = self.damp.max(1e-14 * scale);
        for _ in 0..80 {
            let mut l = curve.clone();
            for k in 0..m {
                l[k * m + k] += damp;
            }
            if !cholesky(&mut l, m, 1e-15 * scale) {
                damp *= 10.0;
                continue;
            }
            let mut step: Vec<f64> = slope.iter().map(|g| -g).collect();
            cholesky_solve(&l, m, &mut step);
            let reach = (0..n).map(|i| (0..t).map(|a| step[i * t + a] * step[i * t + a]).sum::<f64>().sqrt()).fold(0.0f64, f64::max);
            if reach.is_nan() || reach > REACH {
                damp *= 4.0;
                continue;
            }
            let moved = self.moved(&tangents, &step);
            let gain: f64 = slope.iter().zip(&step).map(|(g, s)| -g * s).sum();
            if Self::soft_max(&moved, beta) <= now - 1e-4 * gain + 1e-15 {
                self.v = moved;
                self.damp = damp / 8.0;
                return true;
            }
            damp *= 4.0;
        }
        // no step that helps, however small: this is as far as it goes
        self.settled = true;
        false
    }

    /// Run to the end (for tests and searches; a scene takes its steps a few a frame).
    pub fn finish(&mut self) -> f64 {
        for _ in 0..20_000 {
            if !self.step() {
                break;
            }
        }
        self.coherence()
    }
}

/// The largest |dot product| between two of the vectors (0 for fewer than two).
pub fn coherence(v: &[DVec3]) -> f64 {
    let mut most = 0.0f64;
    for i in 0..v.len() {
        for j in i + 1..v.len() {
            most = most.max(v[i].dot(v[j]).abs());
        }
    }
    most
}

/// The Cholesky factor of the symmetric matrix `a` (n × n, by rows), written over its
/// lower triangle. False if it is not positive definite by a margin of `least`.
fn cholesky(a: &mut [f64], n: usize, least: f64) -> bool {
    for j in 0..n {
        let mut d = a[j * n + j];
        for k in 0..j {
            d -= a[j * n + k] * a[j * n + k];
        }
        // (written so that a NaN fails too)
        if d.is_nan() || d <= least {
            return false;
        }
        let d = d.sqrt();
        a[j * n + j] = d;
        for i in j + 1..n {
            let mut s = a[i * n + j];
            for k in 0..j {
                s -= a[i * n + k] * a[j * n + k];
            }
            a[i * n + j] = s / d;
        }
    }
    true
}

/// Solve L Lᵀ x = b for the factor `l` from [`cholesky`]; `b` becomes x.
fn cholesky_solve(l: &[f64], n: usize, b: &mut [f64]) {
    for i in 0..n {
        let mut s = b[i];
        for k in 0..i {
            s -= l[i * n + k] * b[k];
        }
        b[i] = s / l[i * n + i];
    }
    for i in (0..n).rev() {
        let mut s = b[i];
        for k in i + 1..n {
            s -= l[k * n + i] * b[k];
        }
        b[i] = s / l[i * n + i];
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The table against the arrangements whose value is known in closed form, and the
    /// two floors against the same.
    #[test]
    fn the_table_and_the_floors_agree_with_the_known_arrangements() {
        let third = 1.0f64 / 3.0;
        for (n, value, proved) in
            [(2, 0.0, true), (3, 0.0, true), (4, third, true), (5, 0.2f64.sqrt(), true), (6, 0.2f64.sqrt(), true), (7, third.sqrt(), true)]
        {
            let (best, is_proved) = best_known(n, 3).unwrap();
            assert!((best - value).abs() < 1e-7 && is_proved == proved, "{n}: {best}");
        }
        assert!(!best_known(8, 3).unwrap().1 && best_known(MOST + 1, 3).is_none() && best_known(MOST, 3).is_some());
        assert!((best_known(5, 2).unwrap().0 - 36f64.to_radians().cos()).abs() < 1e-12);
        // the table falls as it should: more lines are never further apart
        for n in 3..MOST {
            assert!(best_known(n + 1, 3).unwrap().0 >= best_known(n, 3).unwrap().0 - 1e-9, "{n}");
        }
        // no entry of the table is below a floor
        for n in 2..=MOST {
            let (best, _) = best_known(n, 3).unwrap();
            assert!(best >= welch(n, 3) - 1e-7 && best >= orthoplex(n, 3) - 1e-7, "{n}");
        }
        assert!((welch(4, 3) - third).abs() < 1e-12 && (welch(6, 3) - 0.2f64.sqrt()).abs() < 1e-12 && (welch(5, 3) - (1.0f64 / 6.0).sqrt()).abs() < 1e-12);
        assert!(
            orthoplex(6, 3) == 0.0 && (orthoplex(7, 3) - third.sqrt()).abs() < 1e-12 && (orthoplex(4, 2) - 0.5f64.sqrt()).abs() < 1e-12 && welch(2, 3) == 0.0
        );
    }

    /// Runs settle, in a number of steps a scene can afford, at arrangements the soft
    /// maximum no longer blurs: for six lines every pair at the same angle to nine
    /// decimals. No step turns a vector further than `REACH`, and the largest |dot
    /// product| at the end is never above where the run began.
    #[test]
    fn runs_settle_in_small_steps() {
        for (n, dim) in [(2, 2), (2, 3), (3, 3), (6, 3), (9, 3), (13, 3), (24, 3), (7, 2)] {
            for seed in 0..3 {
                let mut s = Spread::new(n, dim, seed);
                let began = s.coherence();
                let mut before = s.v.clone();
                while s.step() {
                    let turned = s.v.iter().zip(&before).map(|(a, b)| a.dot(*b).min(1.0).acos()).fold(0.0f64, f64::max);
                    assert!(turned <= REACH * 1.001, "{n} lines: a step of {turned}");
                    assert!(s.v.iter().all(|v| (v.length() - 1.0).abs() < 1e-12 && (dim == 3 || v.z == 0.0)));
                    before = s.v.clone();
                }
                assert!(s.settled() && s.steps() < 1500 && s.coherence() <= began + 1e-12, "{n} lines, seed {seed}: {} steps", s.steps());
                assert!(!s.step() && s.v == before, "a settled run moved");
            }
        }
        let mut six = Spread::new(6, 3, 1);
        six.finish();
        let dots: Vec<f64> = (0..6).flat_map(|i| (i + 1..6).map(move |j| (i, j))).map(|(i, j)| six.v[i].dot(six.v[j]).abs()).collect();
        assert!(dots.iter().all(|d| (d - 0.2f64.sqrt()).abs() < 1e-9), "{dots:?}");
    }

    /// Where a run rests is a local best: no small move of the vectors lowers the
    /// largest |dot product| by more than the soft maximum's blur.
    #[test]
    fn no_small_change_improves_a_settled_arrangement() {
        let mut rng = Rng::new(3);
        for (n, dim, seed) in [(5, 3, 0), (8, 3, 0), (14, 3, 0), (20, 3, 2), (6, 2, 1)] {
            let mut s = Spread::new(n, dim, seed);
            let rest = s.finish();
            for size in [1e-2, 1e-3, 1e-4] {
                for _ in 0..300 {
                    let moved: Vec<DVec3> =
                        s.v.iter()
                            .map(|v| {
                                (*v + DVec3::new(rng.normal() as f64, rng.normal() as f64, if dim == 3 { rng.normal() as f64 } else { 0.0 }) * size).normalize()
                            })
                            .collect();
                    assert!(coherence(&moved) > rest - 2e-6, "{n} lines: a move of size {size} lowers {rest} to {}", coherence(&moved));
                }
            }
        }
    }

    /// Not a test: the search behind `STARTS`. For each number of lines, the first
    /// start (softest first, then by seed) whose run ends at the table's arrangement.
    /// `TRIES=400 cargo test --profile fast --lib math::packing -- --ignored --nocapture find_starts`
    #[test]
    #[ignore]
    fn find_starts() {
        // (only when asked for by name, with TRIES set: not as one of the slow tests)
        let Some(tries) = std::env::var("TRIES").ok().and_then(|t| t.parse::<u64>().ok()) else { return };
        let lines: Vec<String> = std::thread::scope(|scope| {
            let handles: Vec<_> = (2..=40usize)
                .map(|n| {
                    scope.spawn(move || {
                        let (best, _) = best_known(n, 3).unwrap();
                        let (mut least, mut least_at) = (f64::MAX, (0.0, 0));
                        for sharpness in [2.0, 32.0, 512.0, 128.0, 2048.0, 8.0, 8192.0] {
                            for seed in 0..tries {
                                let mut s = Spread::start(n, 3, seed, sharpness);
                                let mu = s.finish();
                                if mu < least - 1e-9 {
                                    (least, least_at) = (mu, (sharpness, seed));
                                }
                                if mu < best + 1e-6 {
                                    return format!("    ({sharpness:.1}, {seed}), // {n}: {mu:.7} ({:+.1e}), {} steps", mu - best, s.steps());
                                }
                            }
                        }
                        format!("    // {n}: NOT FOUND; least {least:.7} ({:+.1e}) at {least_at:?}", least - best)
                    })
                })
                .collect();
            handles.into_iter().map(|h| h.join().unwrap()).collect()
        });
        for line in lines {
            println!("{line}");
        }
    }

    /// A survey, not a test: for each number of lines and each starting sharpness, how
    /// many of the first seeds end at the best known arrangement, the first seed that
    /// does, and how long a run is.
    /// `TRIES=200 MOST=32 cargo test --profile fast --lib math::packing -- --ignored --nocapture survey`
    #[test]
    #[ignore]
    fn survey() {
        // (only when asked for by name, with TRIES set: not as one of the slow tests)
        let Some(tries) = std::env::var("TRIES").ok().and_then(|t| t.parse::<u64>().ok()) else { return };
        let most: usize = std::env::var("MOST").ok().and_then(|t| t.parse().ok()).unwrap_or(32);
        let cases: Vec<(usize, usize)> = (2..=most).map(|n| (3, n)).chain((2..=12).map(|n| (2, n))).collect();
        let lines: Vec<String> = std::thread::scope(|scope| {
            let handles: Vec<_> = cases
                .iter()
                .map(|&(dim, n)| {
                    scope.spawn(move || {
                        let (best, _) = best_known(n, dim).unwrap();
                        let mut line = format!("dim {dim} n {n:2}: table {best:.7}");
                        let (mut least, mut least_at) = (f64::MAX, (0.0, 0));
                        for sharpness in [2.0, 32.0, 512.0, 8192.0] {
                            let (mut hits, mut first, mut steps, mut longest, mut kicks) = (0, None, 0u32, 0u32, 0);
                            let start = std::time::Instant::now();
                            for seed in 0..tries {
                                let mut s = Spread::start(n, dim, seed, sharpness);
                                let mu = s.finish();
                                steps += s.steps();
                                kicks += s.kicks;
                                longest = longest.max(s.steps());
                                if mu < least - 1e-9 {
                                    (least, least_at) = (mu, (sharpness, seed));
                                }
                                if mu < best + 2e-6 {
                                    hits += 1;
                                    first.get_or_insert(seed);
                                }
                            }
                            let per = start.elapsed().as_secs_f64() * 1e3 / tries as f64;
                            line += &format!(
                                " | {sharpness:4}: {hits:3} first {:>4} steps {:4}<{longest:4} kicks {kicks:3} {per:5.1}ms",
                                first.map_or("-".into(), |f| f.to_string()),
                                steps / tries as u32
                            );
                        }
                        line + &format!(" | least {least:.7} ({:+.1e}) at {least_at:?}", least - best)
                    })
                })
                .collect();
            handles.into_iter().map(|h| h.join().unwrap()).collect()
        });
        for line in lines {
            println!("{line}");
        }
    }
}

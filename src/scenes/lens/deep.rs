//! A deep network with random weights, and how nudges grow or die on their way
//! through it: the layers' Jacobians, the finite-time Lyapunov exponents (found by
//! carrying orthonormal directions through the layers), and what becomes of the
//! Jacobian when it is averaged over inputs. No drawing: the Lyapunov scene shows it.

use crate::math::linalg::{self, identity, matmul, matmul_transposed, matvec, norm};
use crate::math::rng::Rng;
use std::rc::Rc;

/// A deep network with random weights.
#[derive(Clone, Copy, PartialEq, Debug)]
pub struct Spec {
    pub width: usize,
    pub depth: usize,
    /// Weights have standard deviation `gain / sqrt(width)`.
    pub gain: f32,
    /// False: each layer replaces the state, h' = φ(W h). True: each layer adds to it,
    /// h' = h + beta φ(W h), as the blocks of a transformer do.
    pub residual: bool,
    pub beta: f32,
    /// True: no activation function (φ is the identity); otherwise φ = tanh.
    pub linear: bool,
    pub seed: u64,
}

#[derive(Clone)]
pub struct Net {
    pub spec: Spec,
    /// `depth` blocks of `width` by `width` standard normal numbers (row-major): the
    /// weights before they are scaled by the gain.
    z: Rc<Vec<f64>>,
}

impl Net {
    pub fn new(spec: Spec) -> Net {
        let mut rng = Rng::new(spec.seed);
        let z = (0..spec.depth * spec.width * spec.width).map(|_| rng.normal() as f64).collect();
        Net { spec, z: Rc::new(z) }
    }

    /// The network with other settings: the same random numbers if only the gain or
    /// the kind of layer changed (so that a slider changes one thing at a time).
    pub fn with(&self, spec: Spec) -> Net {
        let old = self.spec;
        if (old.width, old.depth, old.seed) == (spec.width, spec.depth, spec.seed) {
            Net { spec, z: self.z.clone() }
        } else {
            Net::new(spec)
        }
    }

    fn scale(&self) -> f64 {
        self.spec.gain as f64 / (self.spec.width as f64).sqrt()
    }

    /// Input number `k` (0 is the one the scene follows): standard normal numbers.
    pub fn input(&self, k: u64) -> Vec<f64> {
        let mut rng = Rng::new(self.spec.seed.wrapping_mul(31).wrapping_add(1000 + k));
        (0..self.spec.width).map(|_| rng.normal() as f64).collect()
    }

    /// Layer `l` applied to the state `h`: the next state, and the slope φ' at each neuron.
    fn layer(&self, l: usize, h: &[f64], next: &mut [f64], slope: &mut [f64]) {
        let (n, s, spec) = (self.spec.width, self.scale(), self.spec);
        let w = &self.z[l * n * n..(l + 1) * n * n];
        for i in 0..n {
            let pre = s * w[i * n..(i + 1) * n].iter().zip(h).map(|(w, h)| w * h).sum::<f64>();
            let (out, d) = if spec.linear {
                (pre, 1.0)
            } else {
                let t = pre.tanh();
                (t, 1.0 - t * t)
            };
            next[i] = if spec.residual { h[i] + spec.beta as f64 * out } else { out };
            slope[i] = d;
        }
    }

    /// The states before every layer and after the last (`depth + 1` rows), and the
    /// slopes at every layer (`depth` rows), for the input `x`.
    pub fn run(&self, x: &[f64]) -> (Vec<f64>, Vec<f64>) {
        let (n, depth) = (self.spec.width, self.spec.depth);
        let mut states = vec![0.0; (depth + 1) * n];
        let mut slopes = vec![0.0; depth * n];
        states[..n].copy_from_slice(x);
        for l in 0..depth {
            let (before, after) = states.split_at_mut((l + 1) * n);
            self.layer(l, &before[l * n..], &mut after[..n], &mut slopes[l * n..(l + 1) * n]);
        }
        (states, slopes)
    }

    /// The Jacobian of layer `l` where its neurons have the slopes `slope`: how a
    /// nudge to the state before the layer moves the state after it.
    pub fn jacobian(&self, l: usize, slope: &[f64], out: &mut [f64]) {
        let (n, s, spec) = (self.spec.width, self.scale(), self.spec);
        let w = &self.z[l * n * n..(l + 1) * n * n];
        let k = if spec.residual { spec.beta as f64 * s } else { s };
        for i in 0..n {
            for j in 0..n {
                out[i * n + j] = k * slope[i] * w[i * n + j];
            }
            if spec.residual {
                out[i * n + i] += 1.0;
            }
        }
    }
}

/// How many random nudges are followed.
pub const RANDOM: usize = 6;
/// The size of the nudge that is really made (not infinitely small).
pub const REAL: f64 = 0.1;

/// What is cheap enough to work out the moment a setting changes.
pub struct Quick {
    /// The Jacobian of every layer at the input the scene follows (`depth` blocks).
    pub jac: Vec<f64>,
    /// For each random nudge, the natural logarithm of how much it has grown by each
    /// layer (`depth + 1` numbers, the first 0).
    pub random: Vec<Vec<f64>>,
    /// The same for a real nudge of size `REAL` along the first of those directions:
    /// the two runs of the network really differ by it. Not a number once the
    /// difference is lost in rounding.
    pub real: Vec<f64>,
    /// The average of φ'² over every neuron of every layer.
    pub slope_square: f64,
}

pub fn quick(net: &Net) -> Quick {
    let (n, depth) = (net.spec.width, net.spec.depth);
    let x = net.input(0);
    let (states, slopes) = net.run(&x);
    let mut jac = vec![0.0; depth * n * n];
    for l in 0..depth {
        net.jacobian(l, &slopes[l * n..(l + 1) * n], &mut jac[l * n * n..(l + 1) * n * n]);
    }
    let mut rng = Rng::new(net.spec.seed.wrapping_mul(77).wrapping_add(5));
    let mut first = Vec::new();
    let mut random = Vec::with_capacity(RANDOM);
    let mut next = vec![0.0; n];
    for k in 0..RANDOM {
        let mut d: Vec<f64> = (0..n).map(|_| rng.normal() as f64).collect();
        let len = norm(&d);
        d.iter_mut().for_each(|v| *v /= len);
        if k == 0 {
            first = d.clone();
        }
        let mut trace = vec![0.0];
        for l in 0..depth {
            matvec(&jac[l * n * n..(l + 1) * n * n], &d, &mut next, n);
            std::mem::swap(&mut d, &mut next);
            trace.push(norm(&d).max(1e-300).ln());
        }
        random.push(trace);
    }
    // the same network run again from a slightly different input
    let nudged: Vec<f64> = x.iter().zip(&first).map(|(x, d)| x + REAL * d).collect();
    let (other, _) = net.run(&nudged);
    let real = (0..=depth)
        .map(|l| {
            let (a, b) = (&states[l * n..(l + 1) * n], &other[l * n..(l + 1) * n]);
            let gap = a.iter().zip(b).map(|(a, b)| (a - b) * (a - b)).sum::<f64>().sqrt();
            // (below this the difference of two rounded numbers means nothing)
            if gap > 1e-11 * norm(a).max(1e-30) {
                (gap / REAL).ln()
            } else {
                f64::NAN
            }
        })
        .collect();
    let slope_square = slopes.iter().map(|s| s * s).sum::<f64>() / slopes.len().max(1) as f64;
    Quick { jac, random, real, slope_square }
}

/// The finite-time Lyapunov exponents of the network at one input.
pub struct Exponents {
    /// The network they were worked out for.
    pub spec: Spec,
    pub width: usize,
    pub depth: usize,
    /// `depth + 1` rows of `width` numbers: the natural logarithm of how much the
    /// i-th singular direction has grown by each layer (once the directions before it
    /// have been taken out of it). The last row is ln of the singular values of the
    /// whole Jacobian.
    pub grown: Vec<f64>,
    /// `depth + 1` numbers: the natural logarithm of how much a nudge along vₙ, the
    /// direction that the whole network shrinks most, has grown by each layer: all of
    /// it, where the last column of `grown` counts only the part at right angles to
    /// where the other directions have gone. Both start at 0 and end at ln sₙ; in
    /// between this one lies higher, by as much as several units.
    pub smallest: Vec<f64>,
    /// For each layer, what the unit circle in the plane of the two leading
    /// directions has become: `[s, a, b, d]` for the matrix `e^s [[a, b], [0, d]]`
    /// (in an orthonormal frame of the plane the circle has been carried to).
    pub ellipse: Vec<[f64; 4]>,
}

impl Exponents {
    /// The i-th exponent (0 is the largest): growth rate per layer over the whole depth.
    pub fn chi(&self, i: usize) -> f64 {
        self.grown[self.depth * self.width + i] / self.depth as f64
    }

    /// The half-axes of the ellipse after `layer` layers, as natural logarithms.
    pub fn half_axes(&self, layer: usize) -> (f64, f64) {
        let [s, a, b, d] = self.ellipse[layer.min(self.depth)];
        // singular values of the triangular matrix [[a, b], [0, d]]
        let (sum, det) = (a * a + b * b + d * d, (a * d).abs());
        let root = (sum * sum - 4.0 * det * det).max(0.0).sqrt();
        let big = ((sum + root) * 0.5).sqrt();
        let small = if big > 0.0 { det / big } else { 0.0 };
        (s + big.max(1e-300).ln(), s + small.max(1e-300).ln())
    }
}

/// How many other inputs the Jacobian is compared at, and how many of its singular
/// values are kept.
pub const INPUTS: usize = 12;
pub const KEPT: usize = 12;

/// The Jacobian of the whole network at `INPUTS` other inputs, against their average.
///
/// In a chaotic network the Jacobians at different inputs differ in size by large
/// factors, and a plain average would be little more than the biggest of them. So
/// each is first scaled to have largest singular value 1: what is compared is what
/// the Jacobians do, not how big they are.
pub struct Averaging {
    /// For each input, ln(sᵢ / s₁) for the largest singular values of the Jacobian
    /// there (so every list starts at 0).
    pub each: Vec<Vec<f64>>,
    /// ln of the largest singular values of the average of the scaled Jacobians.
    pub mean: Vec<f64>,
    /// ln s₁ of the Jacobian at each input, before scaling.
    pub sizes: Vec<f64>,
    /// How alike two of the Jacobians are, on average over pairs: the cosine of the
    /// angle between them as lists of numbers (1: the same, 0: unrelated).
    pub overlap: f64,
    /// What `kept` would be for Jacobians with nothing in common: the same scaled
    /// Jacobians, each first turned at random on its input side and on its output
    /// side, then averaged (the mean of `SCRAMBLES` such trials).
    pub unrelated: f64,
}

impl Averaging {
    /// The share of a Jacobian's leading singular value that is left in the average:
    /// 1 if they all agree, and `unrelated` if they have nothing in common. (That is
    /// more than 1/INPUTS, which it would be only if their directions were all at
    /// right angles to one another; a dozen directions taken at random are not.)
    pub fn kept(&self) -> f64 {
        self.mean[0].exp()
    }

    /// In words: do the Jacobians agree? Judged by their overlap.
    pub fn verdict(&self) -> Agreement {
        if self.overlap > AGREE {
            Agreement::Whole
        } else if self.overlap > UNRELATED {
            Agreement::Part
        } else {
            Agreement::None
        }
    }
}

/// How far the Jacobians at different inputs agree (see [`Averaging::verdict`]).
#[derive(Clone, Copy, PartialEq, Debug)]
pub enum Agreement {
    Whole,
    Part,
    None,
}

/// The overlap above which the Jacobians are said to agree, and below which they are
/// said to have next to nothing in common.
const AGREE: f64 = 0.7;
const UNRELATED: f64 = 0.15;
/// How many times the Jacobians are scrambled for [`Averaging::unrelated`].
const SCRAMBLES: usize = 2;

/// A leading exponent within this of zero counts as the edge between order and
/// chaos (per layer: a nudge then changes by a factor of at most e^(±EDGE)).
pub const EDGE: f64 = 0.1;

/// Back-and-forth passes that sharpen the singular directions before they are used.
/// (Two are enough: see the test `two_sweeps_are_enough`.)
const SWEEPS: usize = 2;

enum Stage {
    /// Multiply the layers' Jacobians together, up to layer `l`.
    Product(usize),
    /// Take the product apart for a first estimate of the singular directions.
    Decompose,
    /// Sharpen the estimate: pass `p` (even: forward, odd: backward), step `i`.
    Sweep(usize, usize),
    /// Carry the directions forward once more, noting how each grows at every layer.
    Record(usize),
    /// The Jacobian at other input `k`, up to layer `l`.
    Inputs(usize, usize),
    Mean,
    /// Scramble `t`: the Jacobian at other input `k` again, turned at random.
    Scramble(usize, usize),
    Done,
}

/// The work that takes longer than a frame may: done a layer at a time.
///
/// The singular values of a product of many matrices differ by so many orders of
/// magnitude that the product itself, once multiplied out, has lost all but the
/// largest to rounding. The remedy (Benettin et al., 1980) is never to multiply out:
/// carry a set of orthonormal directions through the layers one at a time,
/// orthonormalising after each, and add up the logarithms of what the
/// orthonormalising had to divide by. Started from the right singular directions
/// of the whole product, those sums are exactly the logarithms of its singular
/// values. The directions come from the multiplied-out product (good for the large
/// ones) sharpened by passing them forward and back through the layers (which fixes
/// the small ones).
pub struct Job {
    net: Net,
    /// The layers' Jacobians at the input the scene follows.
    jac: Vec<f64>,
    sweeps: usize,
    stage: Stage,
    /// A product of layers' Jacobians being built up (rescaled as it goes: only its
    /// directions and the ratios of its singular values are wanted).
    p: Vec<f64>,
    /// The orthonormal directions being carried through the layers, as columns.
    q: Vec<f64>,
    /// While they are carried through for the record: for each layer, the upper
    /// triangular R of its orthonormalising (the layer's Jacobian, in the directions
    /// before and after it).
    r: Vec<f64>,
    /// Where a matrix product is written before it replaces one of its factors.
    t: Vec<f64>,
    /// One layer's Jacobian at another input.
    layer: Vec<f64>,
    grown: Vec<f64>,
    ellipse: Vec<[f64; 4]>,
    slopes: Vec<f64>,
    /// The logarithm of what `p` has been divided by.
    scale: f64,
    /// The Jacobians at the other inputs, each scaled to largest singular value 1,
    /// one after another; their sum over INPUTS; and the sum of them scaled to
    /// length 1 as lists of numbers (for the overlap).
    scaled: Vec<f64>,
    sum: Vec<f64>,
    unit_sum: Vec<f64>,
    each: Vec<Vec<f64>>,
    sizes: Vec<f64>,
    /// For the scrambles: their random numbers, the sum being built, and the leading
    /// singular values found so far.
    rng: Rng,
    scrambled: Vec<f64>,
    unrelated: Vec<f64>,
    pub exponents: Option<Exponents>,
    pub averaging: Option<Averaging>,
}

impl Job {
    pub fn new(net: &Net, jac: &[f64]) -> Job {
        let n = net.spec.width;
        Job {
            net: net.clone(),
            jac: jac.to_vec(),
            sweeps: SWEEPS,
            stage: Stage::Product(0),
            p: identity(n),
            q: Vec::new(),
            r: Vec::new(),
            t: vec![0.0; n * n],
            layer: vec![0.0; n * n],
            grown: Vec::new(),
            ellipse: Vec::new(),
            slopes: Vec::new(),
            scale: 0.0,
            scaled: Vec::new(),
            sum: vec![0.0; n * n],
            unit_sum: vec![0.0; n * n],
            each: Vec::new(),
            sizes: Vec::new(),
            rng: Rng::new(net.spec.seed.wrapping_mul(131).wrapping_add(7)),
            scrambled: vec![0.0; n * n],
            unrelated: Vec::new(),
            exponents: None,
            averaging: None,
        }
    }

    pub fn spec(&self) -> Spec {
        self.net.spec
    }

    pub fn done(&self) -> bool {
        matches!(self.stage, Stage::Done)
    }

    /// What the unit circle in the plane of the first two columns of `q` becomes,
    /// layer by layer.
    fn follow_circle(&mut self) {
        let (n, depth) = (self.net.spec.width, self.net.spec.depth);
        let mut a: Vec<f64> = (0..n).map(|i| self.q[i * n]).collect();
        let mut b: Vec<f64> = (0..n).map(|i| self.q[i * n + 1.min(n - 1)]).collect();
        let (mut ja, mut jb) = (vec![0.0; n], vec![0.0; n]);
        // the matrix so far, e^s [[c0, c1], [0, c2]]
        let (mut s, mut c) = (0.0f64, [1.0f64, 0.0, 1.0]);
        self.ellipse = vec![[s, c[0], c[1], c[2]]];
        for l in 0..depth {
            let j = &self.jac[l * n * n..(l + 1) * n * n];
            matvec(j, &a, &mut ja, n);
            matvec(j, &b, &mut jb, n);
            let r11 = norm(&ja).max(1e-300);
            ja.iter_mut().for_each(|v| *v /= r11);
            let r12: f64 = ja.iter().zip(&jb).map(|(a, b)| a * b).sum();
            jb.iter_mut().zip(&ja).for_each(|(b, a)| *b -= r12 * a);
            let r22 = norm(&jb).max(1e-300);
            jb.iter_mut().for_each(|v| *v /= r22);
            std::mem::swap(&mut a, &mut ja);
            std::mem::swap(&mut b, &mut jb);
            c = [r11 * c[0], r11 * c[1] + r12 * c[2], r22 * c[2]];
            let big = c.iter().fold(0.0f64, |m, v| m.max(v.abs())).max(1e-300);
            c.iter_mut().for_each(|v| *v /= big);
            s += big.ln();
            self.ellipse.push([s, c[0], c[1], c[2]]);
        }
    }

    /// [`Exponents::smallest`], from the triangular matrices of the record. Carried
    /// forward, a nudge along vₙ would soon be swamped by the little of it that rounding
    /// puts along the directions that grow. So it is carried backward instead, from
    /// where it ends up (the last direction after the last layer) through each layer
    /// undone, where it is the nudge that grows fastest and errors die away. (Should a
    /// layer be impossible to undo, the last column of `grown` is given instead.)
    fn smallest(&self) -> Vec<f64> {
        let (n, depth) = (self.net.spec.width, self.net.spec.depth);
        // the nudge in the directions after layer l, made length 1 again at every
        // layer; and the logarithm of its length, taken as 0 after the last layer
        let mut c = vec![0.0; n];
        c[n - 1] = 1.0;
        let mut log = vec![0.0; depth + 1];
        for l in (0..depth).rev() {
            let r = &self.r[l * n * n..(l + 1) * n * n];
            // c ← R⁻¹ c, by back substitution
            for i in (0..n).rev() {
                let rest: f64 = (i + 1..n).map(|k| r[i * n + k] * c[k]).sum();
                c[i] = (c[i] - rest) / r[i * n + i];
            }
            let size = norm(&c);
            if !(size.is_finite() && size > 0.0) {
                return (0..=depth).map(|l| self.grown[l * n + n - 1]).collect();
            }
            c.iter_mut().for_each(|v| *v /= size);
            log[l] = log[l + 1] + size.ln();
        }
        // (for a nudge of size 1 at the input)
        log.iter().map(|v| v - log[0]).collect()
    }

    /// Do one piece of the work. Returns roughly how many multiplications it took.
    pub fn step(&mut self) -> f64 {
        let (n, depth) = (self.net.spec.width, self.net.spec.depth);
        let cube = (n * n * n) as f64;
        let layer = |l: usize| l * n * n..(l + 1) * n * n;
        match self.stage {
            Stage::Product(l) => {
                matmul(&self.jac[layer(l)], &self.p, &mut self.t, n);
                std::mem::swap(&mut self.p, &mut self.t);
                // (only the directions are wanted from this product, so its size may go)
                let big = self.p.iter().fold(0.0f64, |m, v| m.max(v.abs())).max(1e-300);
                self.p.iter_mut().for_each(|v| *v /= big);
                self.stage = if l + 1 < depth { Stage::Product(l + 1) } else { Stage::Decompose };
                cube
            }
            Stage::Decompose => {
                let (_, _, v) = linalg::svd(&self.p, n, n);
                self.q = v;
                self.stage = Stage::Sweep(0, 0);
                12.0 * cube
            }
            Stage::Sweep(pass, i) => {
                if pass >= 2 * self.sweeps {
                    self.grown = vec![0.0; (depth + 1) * n];
                    self.r = vec![0.0; depth * n * n];
                    self.follow_circle();
                    self.stage = Stage::Record(0);
                    return 4.0 * (depth * n * n) as f64;
                }
                if pass % 2 == 0 {
                    matmul(&self.jac[layer(i)], &self.q, &mut self.t, n);
                } else {
                    matmul_transposed(&self.jac[layer(depth - 1 - i)], &self.q, &mut self.t, n);
                }
                std::mem::swap(&mut self.q, &mut self.t);
                linalg::orthonormalise(&mut self.q, n, n);
                self.stage = if i + 1 < depth { Stage::Sweep(pass, i + 1) } else { Stage::Sweep(pass + 1, 0) };
                3.0 * cube
            }
            Stage::Record(l) => {
                matmul(&self.jac[layer(l)], &self.q, &mut self.t, n);
                std::mem::swap(&mut self.q, &mut self.t);
                let r = &mut self.r[layer(l)];
                qr(&mut self.q, r, n);
                for i in 0..n {
                    self.grown[(l + 1) * n + i] = self.grown[l * n + i] + r[i * n + i].max(1e-300).ln();
                }
                if l + 1 < depth {
                    self.stage = Stage::Record(l + 1);
                } else {
                    let smallest = self.smallest();
                    self.r = Vec::new();
                    self.exponents = Some(Exponents {
                        spec: self.net.spec,
                        width: n,
                        depth,
                        grown: std::mem::take(&mut self.grown),
                        smallest,
                        ellipse: std::mem::take(&mut self.ellipse),
                    });
                    self.stage = Stage::Inputs(0, 0);
                    return 3.0 * cube + (depth * n * n) as f64;
                }
                3.0 * cube
            }
            Stage::Inputs(k, l) => {
                if l == 0 {
                    let x = self.net.input(k as u64 + 1);
                    self.slopes = self.net.run(&x).1;
                    self.p = identity(n);
                    self.scale = 0.0;
                }
                self.net.jacobian(l, &self.slopes[l * n..(l + 1) * n], &mut self.layer);
                matmul(&self.layer, &self.p, &mut self.t, n);
                std::mem::swap(&mut self.p, &mut self.t);
                let big = self.p.iter().fold(0.0f64, |m, v| m.max(v.abs())).max(1e-300);
                self.p.iter_mut().for_each(|v| *v /= big);
                self.scale += big.ln();
                if l + 1 < depth {
                    self.stage = Stage::Inputs(k, l + 1);
                    return cube;
                }
                let (_, s, _) = linalg::svd(&self.p, n, n);
                let top = s[0].max(1e-300);
                self.each.push(s.iter().take(KEPT).map(|v| (v / top).max(1e-300).ln()).collect());
                self.sizes.push(top.ln() + self.scale);
                let length = norm(&self.p).max(1e-300);
                for ((sum, unit), p) in self.sum.iter_mut().zip(&mut self.unit_sum).zip(&self.p) {
                    *sum += p / (top * INPUTS as f64);
                    *unit += p / length;
                }
                self.scaled.extend(self.p.iter().map(|p| p / top));
                self.stage = if k + 1 < INPUTS { Stage::Inputs(k + 1, 0) } else { Stage::Mean };
                13.0 * cube
            }
            Stage::Mean => {
                let (_, s, _) = linalg::svd(&self.sum, n, n);
                let mean = s.iter().take(KEPT).map(|v| v.max(1e-300).ln()).collect();
                let k = INPUTS as f64;
                let overlap = (self.unit_sum.iter().map(|v| v * v).sum::<f64>() - k) / (k * (k - 1.0));
                self.averaging =
                    Some(Averaging { each: std::mem::take(&mut self.each), mean, sizes: std::mem::take(&mut self.sizes), overlap, unrelated: 0.0 });
                self.stage = Stage::Scramble(0, 0);
                12.0 * cube
            }
            Stage::Scramble(trial, k) => {
                // the Jacobian at input k, turned at random on both sides: Q A R
                if k == 0 {
                    self.scrambled.fill(0.0);
                }
                let rotation = |rng: &mut Rng, out: &mut Vec<f64>| {
                    out.clear();
                    out.extend((0..n * n).map(|_| rng.normal() as f64));
                    linalg::orthonormalise(out, n, n);
                };
                rotation(&mut self.rng, &mut self.layer);
                matmul(&self.layer, &self.scaled[k * n * n..(k + 1) * n * n], &mut self.t, n);
                rotation(&mut self.rng, &mut self.layer);
                matmul(&self.t, &self.layer, &mut self.p, n);
                for (sum, v) in self.scrambled.iter_mut().zip(&self.p) {
                    *sum += v / INPUTS as f64;
                }
                if k + 1 < INPUTS {
                    self.stage = Stage::Scramble(trial, k + 1);
                    return 6.0 * cube;
                }
                self.unrelated.push(linalg::svd(&self.scrambled, n, n).1[0]);
                if trial + 1 < SCRAMBLES {
                    self.stage = Stage::Scramble(trial + 1, 0);
                } else {
                    if let Some(a) = &mut self.averaging {
                        a.unrelated = self.unrelated.iter().sum::<f64>() / SCRAMBLES as f64;
                    }
                    self.stage = Stage::Done;
                }
                18.0 * cube
            }
            Stage::Done => 0.0,
        }
    }
}

/// Orthonormalise the columns of the `n` by `n` matrix `a` (row-major) in place, and
/// write into `r` the upper triangular R with (`a` as it was) = Q R. On its diagonal:
/// how long each column was once the earlier ones had been taken out of it. Each
/// column is taken against the earlier ones twice (modified Gram-Schmidt with
/// reorthogonalisation), which leaves them orthonormal to rounding.
///
/// `linalg::orthonormalise` takes a column that is less than 1e-12 of its own length
/// away from the earlier ones for one that depends on them, and gives it length 0. In
/// a network that is a real length: a tanh neuron far into saturation has a slope of
/// 1e-13, and squashes the nudges it passes on by that much. Here a column is taken
/// for one that depends on the earlier ones only if what is left of it is rounding;
/// even then its length is the one measured, and a direction that completes the set
/// takes its place.
fn qr(a: &mut [f64], r: &mut [f64], n: usize) {
    r.fill(0.0);
    let length = |a: &[f64], j: usize| (0..n).map(|i| a[i * n + j] * a[i * n + j]).sum::<f64>().sqrt();
    for j in 0..n {
        let mut left = [0.0; 2];
        for pass in 0..2 {
            for k in 0..j {
                let d: f64 = (0..n).map(|i| a[i * n + j] * a[i * n + k]).sum();
                for i in 0..n {
                    a[i * n + j] -= d * a[i * n + k];
                }
                r[k * n + j] += d;
            }
            left[pass] = length(a, j);
        }
        r[j * n + j] = left[1];
        // the second pass takes away only the rounding of the first; if that was most
        // of what was left, nothing of the column's own was
        if left[1] > 0.5 * left[0] {
            for i in 0..n {
                a[i * n + j] /= left[1];
            }
        } else {
            complete(a, n, j);
        }
    }
}

/// Make column `j` of the `n` by `n` matrix `a` a unit vector at right angles to the
/// (orthonormal) columns before it: of the axes, the one with most left once they are
/// taken out.
fn complete(a: &mut [f64], n: usize, j: usize) {
    let mut best = (0.0, vec![0.0; n]);
    for axis in 0..n {
        let mut e = vec![0.0; n];
        e[axis] = 1.0;
        for _ in 0..2 {
            for k in 0..j {
                let d: f64 = (0..n).map(|i| e[i] * a[i * n + k]).sum();
                (0..n).for_each(|i| e[i] -= d * a[i * n + k]);
            }
        }
        let len = norm(&e);
        if len > best.0 {
            best = (len, e);
        }
    }
    for i in 0..n {
        a[i * n + j] = if best.0 > 0.0 { best.1[i] / best.0 } else { 0.0 };
    }
}

/// Everything, worked out in one go (the scene spreads the same work over frames).
#[cfg(test)]
pub fn solve(net: &Net) -> (Quick, Exponents, Averaging) {
    let quick = quick(net);
    let mut job = Job::new(net, &quick.jac);
    while !job.done() {
        job.step();
    }
    (quick, job.exponents.take().unwrap(), job.averaging.take().unwrap())
}

/// The digamma function ψ(x) for x > 0.
fn digamma(mut x: f64) -> f64 {
    let mut shift = 0.0;
    while x < 6.0 {
        shift -= 1.0 / x;
        x += 1.0;
    }
    let f = 1.0 / (x * x);
    shift + x.ln() - 0.5 / x - f * (1.0 / 12.0 - f * (1.0 / 120.0 - f / 252.0))
}

/// The i-th Lyapunov exponent (0 is the largest) of an endlessly long product of
/// `n` by `n` matrices of independent normal numbers of standard deviation
/// `gain / sqrt(n)`: Newman's formula (1986).
pub fn newman(n: usize, gain: f64, i: usize) -> f64 {
    gain.ln() + 0.5 * (digamma((n - i) as f64 / 2.0) - (n as f64 / 2.0).ln())
}

#[cfg(test)]
pub fn test_spec(width: usize, depth: usize, gain: f32) -> Spec {
    Spec { width, depth, gain, residual: false, beta: 0.3, linear: false, seed: 11 }
}

#[cfg(test)]
mod tests {
    use super::test_spec as spec;
    use super::*;

    /// The whole Jacobian multiplied out (only sensible for short networks).
    fn product(net: &Net, jac: &[f64]) -> Vec<f64> {
        let n = net.spec.width;
        let (mut p, mut t) = (identity(n), vec![0.0; n * n]);
        for l in 0..net.spec.depth {
            matmul(&jac[l * n * n..(l + 1) * n * n], &p, &mut t, n);
            std::mem::swap(&mut p, &mut t);
        }
        p
    }

    #[test]
    fn the_jacobian_is_the_derivative_of_the_network() {
        for residual in [false, true] {
            let net = Net::new(Spec { residual, ..spec(5, 4, 1.4) });
            let x = net.input(0);
            let (states, _) = net.run(&x);
            let j = product(&net, &quick(&net).jac);
            let h = 1e-6;
            for col in 0..5 {
                let mut y = x.clone();
                y[col] += h;
                let (other, _) = net.run(&y);
                for row in 0..5 {
                    let numeric = (other[4 * 5 + row] - states[4 * 5 + row]) / h;
                    assert!((numeric - j[row * 5 + col]).abs() < 1e-4, "{numeric} vs {}", j[row * 5 + col]);
                }
            }
        }
    }

    #[test]
    fn small_gain_is_order_and_large_gain_is_chaos() {
        for seed in [1, 2, 3] {
            let ordered = solve(&Net::new(Spec { seed, ..spec(24, 40, 0.6) })).1;
            let chaotic = solve(&Net::new(Spec { seed, ..spec(24, 40, 3.5) })).1;
            assert!(ordered.chi(0) < -0.2, "{}", ordered.chi(0));
            assert!(chaotic.chi(0) > 0.15, "{}", chaotic.chi(0));
        }
    }

    #[test]
    fn the_exponents_are_the_singular_values_of_the_whole_jacobian() {
        for (residual, linear) in [(false, false), (true, false), (false, true)] {
            let net = Net::new(Spec { residual, linear, ..spec(6, 5, 1.3) });
            let (quick, exps, _) = solve(&net);
            let (_, s, _) = linalg::svd(&product(&net, &quick.jac), 6, 6);
            for i in 0..6 {
                let want = s[i].ln() / 5.0;
                assert!((exps.chi(i) - want).abs() < 1e-7, "direction {i}: {} vs {want}", exps.chi(i));
            }
        }
    }

    /// The first exponent again, found another way: by nudging along a direction,
    /// carrying the nudge through the layers and back through their transposes, and
    /// repeating until the direction stops changing.
    #[test]
    fn the_first_exponent_is_the_most_any_nudge_can_grow() {
        for (width, depth, gain) in [(24, 30, 3.0), (40, 60, 1.5), (12, 60, 0.7)] {
            let net = Net::new(spec(width, depth, gain));
            let (quick, exps, _) = solve(&net);
            let n = width;
            let mut v: Vec<f64> = net.input(99);
            let mut t = vec![0.0; n];
            let mut growth = 0.0;
            for _ in 0..400 {
                let len = norm(&v);
                v.iter_mut().for_each(|x| *x /= len);
                growth = 0.0;
                for l in 0..depth {
                    matvec(&quick.jac[l * n * n..(l + 1) * n * n], &v, &mut t, n);
                    let len = norm(&t);
                    growth += len.ln();
                    v.iter_mut().zip(&t).for_each(|(v, t)| *v = t / len);
                }
                for l in (0..depth).rev() {
                    // the transpose of layer l applied to v
                    let j = &quick.jac[l * n * n..(l + 1) * n * n];
                    t.fill(0.0);
                    for i in 0..n {
                        for k in 0..n {
                            t[k] += j[i * n + k] * v[i];
                        }
                    }
                    let len = norm(&t);
                    v.iter_mut().zip(&t).for_each(|(v, t)| *v = t / len);
                }
            }
            assert!((exps.chi(0) - growth / depth as f64).abs() < 1e-6, "{width} wide, {depth} deep: {} vs {}", exps.chi(0), growth / depth as f64);
        }
    }

    #[test]
    fn two_sweeps_are_enough() {
        for (width, depth, gain, residual) in [(24, 30, 3.0, false), (40, 60, 1.5, false), (24, 12, 0.7, false), (24, 30, 1.5, true), (8, 60, 1.0, false)] {
            let net = Net::new(Spec { residual, ..spec(width, depth, gain) });
            let quick = quick(&net);
            let run = |sweeps: usize| {
                let mut job = Job::new(&net, &quick.jac);
                job.sweeps = sweeps;
                while job.exponents.is_none() {
                    job.step();
                }
                job.exponents.take().unwrap()
            };
            let (usual, careful) = (run(SWEEPS), run(12));
            let worst = (0..width).map(|i| (usual.chi(i) - careful.chi(i)).abs()).fold(0.0, f64::max);
            assert!(worst < 2e-3, "{width} wide, {depth} deep, gain {gain}: off by {worst}");
        }
    }

    /// The exponents only, without the averaging.
    fn exponents(net: &Net) -> (Quick, Exponents) {
        let quick = quick(net);
        let mut job = Job::new(net, &quick.jac);
        while job.exponents.is_none() {
            job.step();
        }
        let exps = job.exponents.take().unwrap();
        (quick, exps)
    }

    /// At gain 4 some neurons are so saturated that their slope is 1e-13 (in the 40 by
    /// 60 network of seed 6, one at layer 42 is). The direction such a layer squashes
    /// is squashed by that much, no more: counted as nothing, it would take the last
    /// exponent from -14.7 to -25.8.
    #[test]
    fn the_exponents_add_up_to_the_growth_of_volume() {
        for (width, depth, gain, seed) in [(16, 40, 1.6, 11), (40, 60, 4.0, 6), (40, 60, 4.0, 42), (40, 60, 4.0, 88)] {
            let net = Net::new(Spec { seed, ..spec(width, depth, gain) });
            let (_, exps) = exponents(&net);
            let n = width;
            // ln |det| of each layer's Jacobian, diag(φ′) times the scaled weights, from
            // those parts: the Jacobian's own smallest singular value is lost to
            // rounding when a neuron is that saturated
            let slopes = net.run(&net.input(0)).1;
            let volume: f64 = (0..depth)
                .map(|l| {
                    let weights = linalg::svd(&net.z[l * n * n..(l + 1) * n * n], n, n).1.iter().map(|s| s.ln()).sum::<f64>();
                    slopes[l * n..(l + 1) * n].iter().map(|s| s.ln()).sum::<f64>() + n as f64 * net.scale().ln() + weights
                })
                .sum();
            let sum: f64 = (0..n).map(|i| exps.chi(i)).sum();
            assert!((sum - volume / depth as f64).abs() < 1e-11, "seed {seed}: {sum} vs {}", volume / depth as f64);
            if gain == 4.0 {
                assert!(slopes.iter().fold(1.0f64, |m, s| m.min(*s)) < 1e-12, "seed {seed}: no neuron that saturated");
            }
        }
    }

    /// A direction a layer squashes to 1e-13 of its length is recorded at that length,
    /// and one squashed to nothing still leaves the directions orthonormal.
    #[test]
    fn the_record_keeps_a_nearly_flattened_direction() {
        let n = 12;
        let mut rng = Rng::new(5);
        for tiny in [1e-13, 1e-15, 0.0] {
            // a layer one of whose neurons is (nearly) flat, applied to orthonormal directions
            let base: Vec<f64> = (0..n * n).map(|_| rng.normal() as f64).collect();
            let mut layer = base.clone();
            layer[3 * n..4 * n].iter_mut().for_each(|v| *v *= tiny);
            let mut q: Vec<f64> = (0..n * n).map(|_| rng.normal() as f64).collect();
            linalg::orthonormalise(&mut q, n, n);
            let (mut a, mut r) = (vec![0.0; n * n], vec![0.0; n * n]);
            matmul(&layer, &q, &mut a, n);
            let was = a.clone();
            qr(&mut a, &mut r, n);
            let mut t = vec![0.0; n * n];
            matmul_transposed(&a, &a, &mut t, n);
            let off = t.iter().zip(identity(n)).map(|(t, i)| (t - i).abs()).fold(0.0, f64::max);
            assert!(off < 1e-14, "{tiny}: not orthonormal by {off:e}");
            matmul(&a, &r, &mut t, n);
            let back = t.iter().zip(&was).map(|(t, w)| (t - w).abs()).fold(0.0, f64::max);
            assert!(back < 1e-14, "{tiny}: Q R is off by {back:e}");
            let last = r[n * n - 1];
            if tiny > 0.0 {
                // the lengths multiply to |det|, tiny as it is
                let det = tiny.ln() + linalg::svd(&base, n, n).1.iter().map(|s| s.ln()).sum::<f64>();
                let lengths: f64 = (0..n).map(|i| r[i * n + i].ln()).sum();
                assert!((lengths - det).abs() < 1e-9, "{tiny}: {lengths} vs {det}");
            } else {
                assert!(last < 1e-25, "{last:e}");
            }
        }
    }

    /// The pink line is the size of the whole nudge along vₙ: as found by carrying vₙ
    /// forward, where that is still accurate (a short network); and elsewhere from 0 at
    /// the input to ln sₙ at the end, never below the part of it that is at right
    /// angles to the other directions, and in places well above.
    #[test]
    fn the_nudge_along_the_last_direction_is_followed_whole() {
        for (n, depth, gain, seed) in [(6, 6, 1.3, 1), (24, 6, 1.0, 1), (12, 8, 0.7, 3)] {
            let net = Net::new(Spec { seed, ..spec(n, depth, gain) });
            let (quick, exps) = exponents(&net);
            let (mut a, mut t) = (identity(n), vec![0.0; n * n]);
            let mut each = vec![a.clone()];
            for l in 0..depth {
                matmul(&quick.jac[l * n * n..(l + 1) * n * n], &a, &mut t, n);
                std::mem::swap(&mut a, &mut t);
                each.push(a.clone());
            }
            let v = linalg::svd(&a, n, n).2;
            let last: Vec<f64> = (0..n).map(|i| v[i * n + n - 1]).collect();
            let mut out = vec![0.0; n];
            for (l, a) in each.iter().enumerate() {
                matvec(a, &last, &mut out, n);
                assert!((norm(&out).ln() - exps.smallest[l]).abs() < 2e-7, "{n} wide, layer {l}: {} vs {}", norm(&out).ln(), exps.smallest[l]);
            }
        }
        let shapes = [
            (spec(24, 30, 3.0), false, false),
            (spec(24, 30, 0.7), false, false),
            (spec(24, 30, 1.5), true, false),
            (spec(24, 30, 1.3), false, true),
            (Spec { seed: 6, ..spec(40, 60, 4.0) }, false, false),
        ];
        for (s, residual, linear) in shapes {
            let exps = exponents(&Net::new(Spec { residual, linear, ..s })).1;
            let (n, depth) = (s.width, s.depth);
            let part = |l: usize| exps.grown[l * n + n - 1];
            assert!(exps.smallest.iter().all(|v| v.is_finite()) && exps.smallest[0] == 0.0);
            assert!((exps.smallest[depth] - part(depth)).abs() < 1e-9, "{s:?}: ends at {} against {}", exps.smallest[depth], part(depth));
            let gap = (0..=depth).map(|l| exps.smallest[l] - part(l)).fold(f64::MIN, f64::max);
            assert!((0..=depth).all(|l| exps.smallest[l] > part(l) - 1e-9) && gap > 0.25, "{s:?}: largest gap {gap}");
        }
    }

    #[test]
    fn no_nudge_grows_faster_than_the_first_exponent_or_dies_faster_than_the_last() {
        for gain in [0.7, 1.5, 3.0] {
            let net = Net::new(spec(24, 30, gain));
            let (quick, exps, _) = solve(&net);
            for trace in &quick.random {
                let rate = trace[30] / 30.0;
                assert!(rate <= exps.chi(0) + 1e-9 && rate >= exps.chi(23) - 1e-9, "{rate} outside {} .. {}", exps.chi(23), exps.chi(0));
            }
        }
    }

    /// "The lines settle to a common slope": once a random nudge has had a few layers
    /// to swing round to the direction that grows fastest, it grows as fast as any.
    #[test]
    fn random_nudges_end_up_growing_at_the_same_rate() {
        for gain in [0.7, 3.0] {
            let net = Net::new(spec(24, 60, gain));
            let (quick, exps, _) = solve(&net);
            let late = |trace: &dyn Fn(usize) -> f64| (trace(60) - trace(30)) / 30.0;
            let best = late(&|l| exps.grown[l * 24]);
            for trace in &quick.random {
                let rate = late(&|l| trace[l]);
                assert!((rate - best).abs() < 0.03, "gain {gain}: {rate} against {best}");
            }
        }
    }

    #[test]
    fn a_deep_linear_network_has_the_exponents_of_newmans_formula() {
        let n = 5;
        let net = Net::new(Spec { linear: true, ..spec(n, 3000, 1.1) });
        let exps = solve(&net).1;
        for i in 0..n {
            let want = newman(n, 1.1f32 as f64, i);
            assert!((exps.chi(i) - want).abs() < 0.03, "direction {i}: {} vs {want}", exps.chi(i));
        }
    }

    #[test]
    fn the_circle_of_nudges_ends_as_the_ellipse_of_the_two_leading_singular_values() {
        let net = Net::new(spec(12, 20, 1.8));
        let exps = solve(&net).1;
        assert_eq!(exps.half_axes(0), (0.0, 0.0));
        let (big, small) = exps.half_axes(20);
        assert!((big - exps.grown[20 * 12]).abs() < 1e-6 && (small - exps.grown[20 * 12 + 1]).abs() < 1e-6, "{big} {small}");
    }

    #[test]
    fn a_real_nudge_follows_the_infinitely_small_one_until_it_is_large() {
        let net = Net::new(spec(24, 30, 3.0));
        let quick = quick(&net);
        // at first the two agree
        assert!((quick.real[2] - quick.random[0][2]).abs() < 0.15, "{} vs {}", quick.real[2], quick.random[0][2]);
        // in the end the real one has levelled off, about as big as the activity
        let last = quick.real[30];
        assert!(quick.random[0][30] - last > 1.0, "{} vs {last}", quick.random[0][30]);
        let activity = norm(&net.run(&net.input(0)).0[30 * 24..]);
        let share = REAL * last.exp() / activity;
        assert!(share > 0.5 && share < 2.0, "{share}");
        // in an ordered network the two stay together all the way down
        let quick = super::quick(&Net::new(spec(24, 30, 0.7)));
        assert!((quick.real[30] - quick.random[0][30]).abs() < 0.05);
    }

    #[test]
    fn the_smallest_and_largest_networks_work() {
        for (width, depth) in [(4, 4), (40, 60), (4, 60), (40, 4)] {
            for (residual, linear) in [(false, false), (true, false), (false, true), (true, true)] {
                let (quick, exps, avg) = solve(&Net::new(Spec { residual, linear, ..spec(width, depth, 1.5) }));
                assert!(quick.random.iter().flatten().all(|v| v.is_finite()));
                assert!((0..width).all(|i| exps.chi(i).is_finite()) && (1..width).all(|i| exps.chi(i) <= exps.chi(i - 1) + 1e-9));
                assert_eq!(avg.mean.len(), width.min(KEPT));
                assert!(avg.kept() > 0.0 && avg.kept() <= 1.0 + 1e-9 && avg.overlap <= 1.0 + 1e-9);
                assert!(avg.unrelated > 0.0 && avg.unrelated <= 1.0 + 1e-9, "{width} wide: unrelated {}", avg.unrelated);
            }
        }
    }

    #[test]
    fn the_rule_of_thumb_is_close() {
        for gain in [0.8, 1.5, 2.5] {
            let net = Net::new(spec(40, 60, gain));
            let (quick, exps, _) = solve(&net);
            let estimate = 0.5 * ((gain as f64).powi(2) * quick.slope_square).ln();
            assert!((exps.chi(0) - estimate).abs() < 0.1, "gain {gain}: {} vs {estimate}", exps.chi(0));
        }
    }
}

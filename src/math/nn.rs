//! A small dense neural network with manual backpropagation.
//!
//! Storage is one column per neuron: the values a neuron takes over the whole batch are
//! contiguous. Every inner loop then runs along the batch (`y += w * x` over a column,
//! one activation function over a column), which the compiler turns into SIMD code, in
//! the browser build too. Nothing in the hot loops calls a maths library: `exp`, `sin`
//! and `tanh` are short polynomial forms. Activations of every layer are kept after a
//! forward pass, because the scenes draw them.

use super::rng::Rng;
use std::f32::consts::{FRAC_1_PI, FRAC_1_SQRT_2, FRAC_PI_2, LOG2_E, SQRT_2};

/// Hidden-layer nonlinearities. The last two are the normalised Hermite polynomials
/// of degree two and three: activations with exactly one Hermite component, which is
/// what makes them interesting to study.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Act {
    Tanh,
    Relu,
    Sigmoid,
    Silu,
    Gelu,
    ReluSquared,
    Sin,
    Hermite2,
    Hermite3,
}

/// Run `$body` with `$v` bound to the activation as a compile-time constant, so that
/// the loop in `$body` is compiled once per activation with the function inlined.
macro_rules! specialise {
    ($act:expr, $v:ident, $body:expr) => {
        match $act {
            Act::Tanh => {
                const $v: Act = Act::Tanh;
                $body
            }
            Act::Relu => {
                const $v: Act = Act::Relu;
                $body
            }
            Act::Sigmoid => {
                const $v: Act = Act::Sigmoid;
                $body
            }
            Act::Silu => {
                const $v: Act = Act::Silu;
                $body
            }
            Act::Gelu => {
                const $v: Act = Act::Gelu;
                $body
            }
            Act::ReluSquared => {
                const $v: Act = Act::ReluSquared;
                $body
            }
            Act::Sin => {
                const $v: Act = Act::Sin;
                $body
            }
            Act::Hermite2 => {
                const $v: Act = Act::Hermite2;
                $body
            }
            Act::Hermite3 => {
                const $v: Act = Act::Hermite3;
                $body
            }
        }
    };
}

impl Act {
    pub const ALL: [Act; 9] = [Act::Tanh, Act::Relu, Act::Sigmoid, Act::Silu, Act::Gelu, Act::ReluSquared, Act::Sin, Act::Hermite2, Act::Hermite3];

    pub fn name(self) -> &'static str {
        match self {
            Act::Tanh => "tanh",
            Act::Relu => "ReLU",
            Act::Sigmoid => "sigmoid",
            Act::Silu => "SiLU",
            Act::Gelu => "GELU",
            Act::ReluSquared => "ReLU²",
            Act::Sin => "sin",
            Act::Hermite2 => "(x²−1)/√2",
            Act::Hermite3 => "(x³−3x)/√6",
        }
    }

    /// The range of outputs worth showing when a neuron's response is drawn.
    pub fn display_range(self) -> [f32; 2] {
        match self {
            Act::Tanh | Act::Sin => [-1.0, 1.0],
            Act::Sigmoid => [0.0, 1.0],
            _ => [-2.0, 2.0],
        }
    }

    #[inline(always)]
    pub fn f(self, z: f32) -> f32 {
        match self {
            Act::Tanh => fast_tanh(z),
            Act::Relu => z.max(0.0),
            Act::Sigmoid => 0.5 + 0.5 * fast_tanh(0.5 * z),
            // the clamps keep far-negative inputs from producing numbers below the
            // normal floating-point range (see `TINY`)
            Act::Silu => z.max(-40.0) * logistic(z),
            Act::Gelu => z.max(-12.0) * normal_cdf(z),
            Act::ReluSquared => {
                let r = z.max(0.0);
                r * r
            }
            Act::Sin => fast_sin(z),
            Act::Hermite2 => (z * z - 1.0) * FRAC_1_SQRT_2,
            Act::Hermite3 => (z * z * z - 3.0 * z) * 0.408_248_3,
        }
    }

    /// Derivative with respect to the input `z`, given also the output `a = f(z)`.
    #[inline(always)]
    pub fn df(self, z: f32, a: f32) -> f32 {
        match self {
            Act::Tanh => 1.0 - a * a,
            Act::Relu => {
                if a > 0.0 {
                    1.0
                } else {
                    0.0
                }
            }
            Act::Sigmoid => a * (1.0 - a),
            Act::Silu => {
                let s = logistic(z);
                s * (1.0 + z.clamp(-40.0, 40.0) * (1.0 - s))
            }
            Act::Gelu => {
                let c = z.clamp(-12.0, 12.0);
                normal_cdf(z) + c * fast_exp(-0.5 * c * c) * 0.398_942_3
            }
            Act::ReluSquared => 2.0 * z.max(0.0),
            Act::Sin => fast_sin(z + FRAC_PI_2),
            Act::Hermite2 => z * SQRT_2,
            Act::Hermite3 => (3.0 * z * z - 3.0) * 0.408_248_3,
        }
    }

    /// The activation applied to a whole column.
    pub fn apply(self, z: &[f32], out: &mut [f32]) {
        specialise!(self, A, {
            for (o, &z) in out.iter_mut().zip(z) {
                *o = A.f(z);
            }
        })
    }

    /// Multiply `d` by the derivative, element by element (the backward pass).
    fn scale_by_derivative(self, z: &[f32], a: &[f32], d: &mut [f32]) {
        specialise!(self, A, {
            for ((d, &z), &a) in d.iter_mut().zip(z).zip(a) {
                *d *= A.df(z, a);
            }
        })
    }
}

/// Numbers smaller than this are treated as zero in the optimiser's state. Values
/// below the normal floating-point range (about 1e-38) are extremely slow to compute
/// with on many processors, WebAssembly cannot switch that behaviour off, and a
/// saturated network produces them constantly: gradients underflow, and decaying
/// averages linger there for thousands of steps.
const TINY: f32 = 1e-30;

#[inline(always)]
fn flush(x: f32) -> f32 {
    if x.abs() < TINY {
        0.0
    } else {
        x
    }
}

/// Round to the nearest integer without a library call (exact for |x| < 2^22).
#[inline(always)]
fn round(x: f32) -> f32 {
    const BIG: f32 = 12_582_912.0; // 1.5 * 2^23: floats this large are whole numbers
    (x + BIG) - BIG
}

/// `exp(x)` by reduction to `2^k * exp(r)` with `|r| <= ln(2)/2` and a degree-six
/// polynomial; relative error below 2e-6. The argument is clamped to [-87, 88], where
/// the result is still a normal number.
#[inline(always)]
pub fn fast_exp(x: f32) -> f32 {
    let x = x.clamp(-87.0, 88.0);
    let k = round(x * LOG2_E);
    // ln(2) split in two so the reduction keeps its low bits
    let r = (x - k * 0.693_359_4) + k * 2.121_944_4e-4;
    let p = 1.0 + r * (1.0 + r * (0.5 + r * (1.0 / 6.0 + r * (1.0 / 24.0 + r * (1.0 / 120.0 + r * (1.0 / 720.0))))));
    p * f32::from_bits((((k as i32) + 127) as u32) << 23)
}

/// `sin(x)` by reduction to `[-pi/2, pi/2]` and an odd polynomial; error below 2e-6
/// for arguments up to a few thousand.
#[inline(always)]
pub fn fast_sin(x: f32) -> f32 {
    // beyond this a float no longer resolves the phase; clamping keeps the result bounded
    let x = x.clamp(-4e6, 4e6);
    let k = round(x * FRAC_1_PI);
    // pi split in two; the first part times a small whole number is exact
    let r = (x - k * 3.140_625) - k * 9.676_536e-4;
    let r2 = r * r;
    let s = r * (1.0 + r2 * (-1.0 / 6.0 + r2 * (1.0 / 120.0 + r2 * (-1.0 / 5040.0 + r2 * (1.0 / 362_880.0 - r2 / 39_916_800.0)))));
    // (-1)^k, by setting the sign bit
    f32::from_bits(s.to_bits() ^ (((k as i32) as u32) << 31))
}

#[inline(always)]
fn logistic(z: f32) -> f32 {
    1.0 / (1.0 + fast_exp((-z).clamp(-40.0, 40.0)))
}

/// The standard normal distribution function, from a classical rational approximation
/// of erf (Abramowitz and Stegun 7.1.26; absolute error below 1e-6). Beyond twelve
/// standard deviations it stops changing.
#[inline(always)]
pub fn normal_cdf(z: f32) -> f32 {
    let x = (z.abs() * FRAC_1_SQRT_2).min(8.5);
    let t = 1.0 / (1.0 + 0.327_591_1 * x);
    let poly = t * (0.254_829_6 + t * (-0.284_496_74 + t * (1.421_413_7 + t * (-1.453_152 + t * 1.061_405_4))));
    let tail = 0.5 * poly * fast_exp(-x * x);
    if z >= 0.0 {
        1.0 - tail
    } else {
        tail
    }
}

/// How the weights are updated from the gradient.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Optim {
    Sgd,
    Momentum,
    Nesterov,
    AdaGrad,
    RmsProp,
    Adam,
    /// Momentum orthogonalised by Newton-Schulz (Jordan et al., 2024): each weight
    /// matrix moves along an approximately orthogonal matrix, so every direction of
    /// the layer is updated at the same rate. See [`Mlp::step`].
    Muon,
}

impl Optim {
    pub const ALL: [Optim; 7] = [Optim::Sgd, Optim::Momentum, Optim::Nesterov, Optim::AdaGrad, Optim::RmsProp, Optim::Adam, Optim::Muon];

    pub fn name(self) -> &'static str {
        match self {
            Optim::Sgd => "SGD",
            Optim::Momentum => "momentum",
            Optim::Nesterov => "Nesterov",
            Optim::AdaGrad => "AdaGrad",
            Optim::RmsProp => "RMSProp",
            Optim::Adam => "Adam",
            Optim::Muon => "Muon",
        }
    }

    /// Does the momentum setting matter to this optimiser?
    pub fn uses_momentum(self) -> bool {
        matches!(self, Optim::Momentum | Optim::Nesterov | Optim::Muon)
    }

    /// The momentum its authors (or long habit) recommend.
    pub fn default_momentum(self) -> f32 {
        if self == Optim::Muon {
            0.95
        } else {
            0.9
        }
    }

    /// A learning rate at which this optimiser trains small networks comfortably.
    pub fn default_lr(self) -> f32 {
        match self {
            Optim::Sgd => 0.3,
            Optim::Momentum | Optim::Nesterov => 0.05,
            Optim::AdaGrad => 0.1,
            Optim::RmsProp => 0.01,
            Optim::Adam | Optim::Muon => 0.02,
        }
    }
}

/// tanh as a ratio of two small polynomials (a truncated continued fraction) with a
/// clamped argument. Within 1e-3 of the true value everywhere and several times faster than
/// the library call, which otherwise dominates the cost of drawing a network's response
/// over thousands of points every frame. The network trains against this exact
/// function. (Its derivative is taken as 1 - a² from the same values: the slope the
/// true tanh has where it takes that value. That is this function's own slope to
/// within a percent or two up to |x| = 3; on the last stretch before the clamp at
/// 4.2, where both are below 0.002, it is up to four times too large, and on the
/// clamped part it gives 0.0015 for a flat function.)
#[inline(always)]
pub fn fast_tanh(x: f32) -> f32 {
    // beyond this the true function is within 1e-3 of +-1 and the ratio would turn over
    let x = x.clamp(-4.2, 4.2);
    let x2 = x * x;
    let num = x * (10395.0 + x2 * (1260.0 + x2 * 21.0));
    let den = 10395.0 + x2 * (4725.0 + x2 * (210.0 + x2));
    num / den
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Loss {
    /// Mean squared error on the raw outputs.
    Mse,
    /// Binary cross-entropy on a single logit; targets are 0 or 1.
    Logistic,
    /// Cross-entropy over one logit per class; each target is a class index.
    Softmax,
}

// ---- the four loops everything is built from -----------------------------------

#[inline(always)]
fn col(v: &[f32], j: usize, n: usize) -> &[f32] {
    &v[j * n..(j + 1) * n]
}

/// The inner loops everything here is built from (`y += a x`, the same for four
/// columns at once, dot product, sum), in the fastest form for the target. Other
/// numerical code should use these rather than write its own loops.
pub use kernels::{axpy, axpy4, dot, sum};

/// The plain versions: what the compiler vectorises by itself (well, natively).
/// The sums use sixteen running totals: one running total must be added in order,
/// which rules out SIMD, and sixteen keep four SIMD registers busy at once.
mod portable {
    #[inline]
    pub fn axpy(y: &mut [f32], a: f32, x: &[f32]) {
        for (y, &x) in y.iter_mut().zip(x) {
            *y += a * x;
        }
    }

    /// Four `axpy`s in one pass over `y`.
    #[inline]
    pub fn axpy4(y: &mut [f32], a: [f32; 4], x: [&[f32]; 4]) {
        let n = y.len();
        let (x0, x1, x2, x3) = (&x[0][..n], &x[1][..n], &x[2][..n], &x[3][..n]);
        for i in 0..n {
            y[i] += ((a[0] * x0[i] + a[1] * x1[i]) + a[2] * x2[i]) + a[3] * x3[i];
        }
    }

    /// The sixteen totals added up, in the one order every version uses.
    #[inline]
    pub fn fold(acc: [f32; 16]) -> f32 {
        let mut four = [0.0f32; 4];
        for i in 0..4 {
            four[i] = (acc[i] + acc[i + 4]) + (acc[i + 8] + acc[i + 12]);
        }
        (four[0] + four[1]) + (four[2] + four[3])
    }

    #[inline]
    pub fn dot(a: &[f32], b: &[f32]) -> f32 {
        // (over the common length, as the SIMD version does)
        let n = a.len().min(b.len());
        let (a, b) = (&a[..n], &b[..n]);
        let mut acc = [0.0f32; 16];
        let (mut ca, mut cb) = (a.chunks_exact(16), b.chunks_exact(16));
        for (x, y) in (&mut ca).zip(&mut cb) {
            for i in 0..16 {
                acc[i] += x[i] * y[i];
            }
        }
        let mut s = fold(acc);
        for (x, y) in ca.remainder().iter().zip(cb.remainder()) {
            s += x * y;
        }
        s
    }

    #[inline]
    pub fn sum(a: &[f32]) -> f32 {
        let mut acc = [0.0f32; 16];
        let mut ca = a.chunks_exact(16);
        for x in &mut ca {
            for i in 0..16 {
                acc[i] += x[i];
            }
        }
        let mut s = fold(acc);
        for x in ca.remainder() {
            s += x;
        }
        s
    }
}

#[cfg(not(all(target_arch = "wasm32", target_feature = "simd128")))]
use portable as kernels;

/// The same four loops written out with WebAssembly's SIMD instructions, four
/// registers at a time. The compiler vectorises the plain versions for this target
/// too, but without unrolling them, and the browser's compiler does not unroll
/// either; written out, the arithmetic-table job trains markedly faster (measure with
/// `tools/wasm-bench.mjs`). Every result is bit for bit that of the plain version:
/// the same operations in the same order.
#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
mod kernels {
    use core::arch::wasm32::*;

    #[inline(always)]
    unsafe fn load(p: *const f32) -> v128 {
        v128_load(p as *const v128)
    }

    #[inline(always)]
    unsafe fn store(p: *mut f32, v: v128) {
        v128_store(p as *mut v128, v)
    }

    #[inline]
    pub fn axpy(y: &mut [f32], a: f32, x: &[f32]) {
        let n = y.len().min(x.len());
        let (py, px, va) = (y.as_mut_ptr(), x.as_ptr(), f32x4_splat(a));
        let mut i = 0;
        // SAFETY: every load and store is within the first `n` elements of both slices.
        unsafe {
            while i + 16 <= n {
                for k in [0, 4, 8, 12] {
                    store(py.add(i + k), f32x4_add(load(py.add(i + k)), f32x4_mul(va, load(px.add(i + k)))));
                }
                i += 16;
            }
            while i + 4 <= n {
                store(py.add(i), f32x4_add(load(py.add(i)), f32x4_mul(va, load(px.add(i)))));
                i += 4;
            }
        }
        for i in i..n {
            y[i] += a * x[i];
        }
    }

    #[inline]
    pub fn axpy4(y: &mut [f32], a: [f32; 4], x: [&[f32]; 4]) {
        let n = y.len();
        let (x0, x1, x2, x3) = (&x[0][..n], &x[1][..n], &x[2][..n], &x[3][..n]);
        let py = y.as_mut_ptr();
        let (p0, p1, p2, p3) = (x0.as_ptr(), x1.as_ptr(), x2.as_ptr(), x3.as_ptr());
        let (a0, a1, a2, a3) = (f32x4_splat(a[0]), f32x4_splat(a[1]), f32x4_splat(a[2]), f32x4_splat(a[3]));
        let mut i = 0;
        // SAFETY: the four inputs were cut to `n` elements above, and `i + 4 <= n`.
        unsafe {
            while i + 8 <= n {
                for k in [0, 4] {
                    let j = i + k;
                    let t = f32x4_add(
                        f32x4_add(f32x4_add(f32x4_mul(a0, load(p0.add(j))), f32x4_mul(a1, load(p1.add(j)))), f32x4_mul(a2, load(p2.add(j)))),
                        f32x4_mul(a3, load(p3.add(j))),
                    );
                    store(py.add(j), f32x4_add(load(py.add(j)), t));
                }
                i += 8;
            }
            while i + 4 <= n {
                let t = f32x4_add(
                    f32x4_add(f32x4_add(f32x4_mul(a0, load(p0.add(i))), f32x4_mul(a1, load(p1.add(i)))), f32x4_mul(a2, load(p2.add(i)))),
                    f32x4_mul(a3, load(p3.add(i))),
                );
                store(py.add(i), f32x4_add(load(py.add(i)), t));
                i += 4;
            }
        }
        for i in i..n {
            y[i] += ((a[0] * x0[i] + a[1] * x1[i]) + a[2] * x2[i]) + a[3] * x3[i];
        }
    }

    /// Four registers of four totals each, added up in the order `portable::fold` uses.
    #[inline(always)]
    fn fold(acc: [v128; 4]) -> f32 {
        let four = f32x4_add(f32x4_add(acc[0], acc[1]), f32x4_add(acc[2], acc[3]));
        (f32x4_extract_lane::<0>(four) + f32x4_extract_lane::<1>(four)) + (f32x4_extract_lane::<2>(four) + f32x4_extract_lane::<3>(four))
    }

    #[inline]
    pub fn dot(a: &[f32], b: &[f32]) -> f32 {
        let n = a.len().min(b.len());
        let (pa, pb) = (a.as_ptr(), b.as_ptr());
        let mut acc = [f32x4_splat(0.0); 4];
        let mut i = 0;
        // SAFETY: every load is within the first `n` elements of both slices.
        unsafe {
            while i + 16 <= n {
                for k in 0..4 {
                    acc[k] = f32x4_add(acc[k], f32x4_mul(load(pa.add(i + 4 * k)), load(pb.add(i + 4 * k))));
                }
                i += 16;
            }
        }
        let mut s = fold(acc);
        for i in i..n {
            s += a[i] * b[i];
        }
        s
    }

    #[inline]
    pub fn sum(a: &[f32]) -> f32 {
        let (n, pa) = (a.len(), a.as_ptr());
        let mut acc = [f32x4_splat(0.0); 4];
        let mut i = 0;
        // SAFETY: every load is within the slice.
        unsafe {
            while i + 16 <= n {
                for k in 0..4 {
                    acc[k] = f32x4_add(acc[k], load(pa.add(i + 4 * k)));
                }
                i += 16;
            }
        }
        let mut s = fold(acc);
        for &x in &a[i..] {
            s += x;
        }
        s
    }
}

/// Do the written-out loops give exactly what the plain ones give? (They are only
/// compiled for the browser, where the unit tests do not run, so the benchmark
/// script calls this first.)
pub fn kernels_agree() -> bool {
    let mut rng = Rng::new(77);
    let data: Vec<f32> = (0..5 * 211).map(|_| rng.normal()).collect();
    let column = |k: usize, n: usize| &data[k * 211..k * 211 + n];
    (0..=211).all(|n| {
        let (mut y1, mut y2) = (column(4, n).to_vec(), column(4, n).to_vec());
        axpy(&mut y1, 0.37, column(0, n));
        portable::axpy(&mut y2, 0.37, column(0, n));
        let w = [0.9, -1.3, 0.2, 2.5];
        let xs = [column(0, n), column(1, n), column(2, n), column(3, n)];
        axpy4(&mut y1, w, xs);
        portable::axpy4(&mut y2, w, xs);
        y1.iter().zip(&y2).all(|(a, b)| a.to_bits() == b.to_bits())
            && dot(column(0, n), column(1, n)).to_bits() == portable::dot(column(0, n), column(1, n)).to_bits()
            && sum(column(2, n)).to_bits() == portable::sum(column(2, n)).to_bits()
    })
}

/// Make `v` at least `n` long. Buffers never shrink, so alternating between a small
/// training batch and a large display batch allocates nothing.
#[inline]
fn grow(v: &mut Vec<f32>, n: usize) {
    if v.len() < n {
        v.resize(n, 0.0);
    }
}

/// One parameter update rule, with its constants.
#[derive(Clone, Copy)]
enum Rule {
    Sgd { lr: f32 },
    Momentum { lr: f32, mu: f32 },
    Nesterov { lr: f32, mu: f32 },
    AdaGrad { lr: f32 },
    RmsProp { lr: f32 },
    Adam { step: f32 },
}

fn update(rule: Rule, p: &mut [f32], g: &[f32], m: &mut [f32], v: &mut [f32]) {
    let all = p.iter_mut().zip(g).zip(m.iter_mut()).zip(v.iter_mut());
    match rule {
        Rule::Sgd { lr } => {
            for (((p, &g), _), _) in all {
                *p -= lr * g;
            }
        }
        Rule::Momentum { lr, mu } => {
            for (((p, &g), m), _) in all {
                *m = flush(mu * *m + g);
                *p -= lr * *m;
            }
        }
        Rule::Nesterov { lr, mu } => {
            for (((p, &g), m), _) in all {
                *m = flush(mu * *m + g);
                *p -= lr * (g + mu * *m);
            }
        }
        Rule::AdaGrad { lr } => {
            for (((p, &g), _), v) in all {
                let g = if g.abs() < 1e-15 { 0.0 } else { g };
                *v += g * g;
                *p -= lr * g / (v.sqrt() + 1e-8);
            }
        }
        Rule::RmsProp { lr } => {
            for (((p, &g), _), v) in all {
                let g = if g.abs() < 1e-15 { 0.0 } else { g };
                *v = flush(0.9 * *v + 0.1 * g * g);
                *p -= lr * g / (v.sqrt() + 1e-8);
            }
        }
        Rule::Adam { step } => {
            for (((p, &g), m), v) in all {
                // squaring anything smaller would leave the normal range
                let g = if g.abs() < 1e-15 { 0.0 } else { g };
                *m = flush(0.9 * *m + 0.1 * g);
                *v = flush(0.999 * *v + 0.001 * g * g);
                *p -= step * *m / (v.sqrt() + 1e-8);
            }
        }
    }
}

/// Adam's step size after `steps` steps at learning rate `lr`: the two bias
/// corrections folded into one factor.
pub fn adam_step_size(steps: u32, lr: f32) -> f32 {
    // (the first step is step 1; asked about step 0, answer for that rather than 0/0)
    let steps = steps.max(1);
    let c1 = 1.0 - 0.9f32.powi(steps.min(10_000) as i32);
    let c2 = 1.0 - 0.999f32.powi(steps.min(100_000) as i32);
    lr * c2.sqrt() / c1
}

/// One Adam update of the parameters `p` from their gradient `g`, with the running
/// averages `m` and `v` (of the gradient and of its square); `step` comes from
/// [`adam_step_size`]. Other models' training loops use this rather than write the
/// rule out again: it carries the guards that keep the averages out of the denormal
/// numbers, which WebAssembly computes very slowly.
pub fn adam_update(step: f32, p: &mut [f32], g: &[f32], m: &mut [f32], v: &mut [f32]) {
    update(Rule::Adam { step }, p, g, m, v);
}

/// Scratch space for [`orthogonalise`].
#[derive(Clone, Default)]
struct Ortho {
    x: Vec<f32>,
    y: Vec<f32>,
    a: Vec<f32>,
    b: Vec<f32>,
}

/// Replace the `rows` by `cols` matrix `g` (row-major) by an approximately orthogonal
/// matrix with the same singular vectors: if `g = U S V'`, the result is close to
/// `U V'`, every singular value having been pushed to roughly one (between about 0.7
/// and 1.2, less for a matrix whose singular values differ a thousandfold). This is the Newton-Schulz iteration of the
/// Muon reference implementation, coefficients and step count included: five rounds
/// of `X <- a X + (b A + c A^2) X` with `A = X X'`, on the matrix scaled to norm one
/// and laid on its side if it is tall, so that `A` is the smaller of the two products.
fn orthogonalise(g: &mut [f32], rows: usize, cols: usize, ws: &mut Ortho) {
    const A: f32 = 3.4445;
    const B: f32 = -4.7750;
    const C: f32 = 2.0315;
    const STEPS: usize = 5;
    let tall = rows > cols;
    let (m, n) = if tall { (cols, rows) } else { (rows, cols) };
    if m == 0 {
        return;
    }
    ws.x.resize(m * n, 0.0);
    ws.y.resize(m * n, 0.0);
    ws.a.resize(m * m, 0.0);
    ws.b.resize(m * m, 0.0);
    if tall {
        for (i, row) in ws.x.chunks_exact_mut(n).enumerate() {
            for (j, v) in row.iter_mut().enumerate() {
                *v = g[j * cols + i];
            }
        }
    } else {
        ws.x.copy_from_slice(g);
    }
    let scale = 1.0 / (dot(&ws.x, &ws.x).sqrt() + 1e-7);
    ws.x.iter_mut().for_each(|v| *v *= scale);
    for _ in 0..STEPS {
        let (x, y, a, b) = (&ws.x, &mut ws.y, &mut ws.a, &mut ws.b);
        // A = X X' (symmetric)
        for i in 0..m {
            for j in i..m {
                let v = dot(col(x, i, n), col(x, j, n));
                a[i * m + j] = v;
                a[j * m + i] = v;
            }
        }
        // B = b A + c A A
        for (i, bi) in b.chunks_exact_mut(m).enumerate() {
            for (o, &v) in bi.iter_mut().zip(col(a, i, m)) {
                *o = B * v;
            }
            for k in 0..m {
                axpy(bi, C * a[i * m + k], col(a, k, m));
            }
        }
        // X <- a X + B X
        for (i, yi) in y.chunks_exact_mut(n).enumerate() {
            for (o, &v) in yi.iter_mut().zip(col(x, i, n)) {
                *o = A * v;
            }
            for k in 0..m {
                axpy(yi, b[i * m + k], col(x, k, n));
            }
        }
        std::mem::swap(&mut ws.x, &mut ws.y);
    }
    if tall {
        for (i, row) in ws.x.chunks_exact(n).enumerate() {
            for (j, &v) in row.iter().enumerate() {
                g[j * cols + i] = v;
            }
        }
    } else {
        g.copy_from_slice(&ws.x);
    }
}

#[derive(Clone)]
pub struct Mlp {
    /// Layer widths, input first.
    pub sizes: Vec<usize>,
    /// `w[l]` is `sizes[l + 1]` rows by `sizes[l]` columns, row-major: row `j` holds
    /// the weights into neuron `j`.
    pub w: Vec<Vec<f32>>,
    pub b: Vec<Vec<f32>>,
    pub act: Act,
    gw: Vec<Vec<f32>>,
    gb: Vec<Vec<f32>>,
    mw: Vec<Vec<f32>>,
    vw: Vec<Vec<f32>>,
    mb: Vec<Vec<f32>>,
    vb: Vec<Vec<f32>>,
    steps: u32,
    /// The optimiser of the last step: its running averages mean nothing to another.
    last: Option<Optim>,
    /// `a[0]` is the input batch, `a[l + 1]` the output of layer `l`; one column of
    /// `batch` values per neuron.
    a: Vec<Vec<f32>>,
    /// `z[l]` is the input to layer `l`'s nonlinearity, in the same layout.
    z: Vec<Vec<f32>>,
    delta: Vec<f32>,
    delta_prev: Vec<f32>,
    scratch: Vec<f32>,
    /// For inputs given as indices of the active units (see [`Mlp::forward_hot`]).
    hot: Vec<u32>,
    hot_k: usize,
    batch: usize,
    ortho: Ortho,
}

impl Mlp {
    pub fn new(sizes: &[usize], act: Act, rng: &mut Rng) -> Self {
        let layers = sizes.len() - 1;
        let mut w = Vec::with_capacity(layers);
        let mut b = Vec::with_capacity(layers);
        for l in 0..layers {
            let (fan_in, fan_out) = (sizes[l], sizes[l + 1]);
            // He for rectifiers; for saturating units Xavier times a gain of 1.6 (about
            // the 5/3 usually given for tanh)
            let scale = match act {
                Act::Relu | Act::Silu | Act::Gelu => (2.0 / fan_in as f32).sqrt(),
                // polynomial activations amplify large inputs, so start them small
                Act::ReluSquared | Act::Hermite2 | Act::Hermite3 => (1.0 / fan_in as f32).sqrt(),
                Act::Tanh | Act::Sigmoid | Act::Sin => (2.0 / (fan_in + fan_out) as f32).sqrt() * 1.6,
            };
            w.push((0..fan_in * fan_out).map(|_| rng.normal() * scale).collect::<Vec<f32>>());
            b.push((0..fan_out).map(|_| if act == Act::Relu { rng.range(-0.3, 0.3) } else { rng.normal() * 0.3 }).collect::<Vec<f32>>());
        }
        let zw: Vec<Vec<f32>> = w.iter().map(|x| vec![0.0; x.len()]).collect();
        let zb: Vec<Vec<f32>> = b.iter().map(|x| vec![0.0; x.len()]).collect();
        Mlp {
            sizes: sizes.to_vec(),
            w,
            b,
            act,
            gw: zw.clone(),
            gb: zb.clone(),
            mw: zw.clone(),
            vw: zw,
            mb: zb.clone(),
            vb: zb,
            steps: 0,
            last: None,
            a: vec![Vec::new(); sizes.len()],
            z: vec![Vec::new(); sizes.len() - 1],
            delta: Vec::new(),
            delta_prev: Vec::new(),
            scratch: Vec::new(),
            hot: Vec::new(),
            hot_k: 0,
            batch: 0,
            ortho: Ortho::default(),
        }
    }

    /// A network with the given parameters (in the layout of [`Mlp::params`]) and a
    /// fresh optimiser state: how a network is handed to another thread.
    pub fn from_params(sizes: &[usize], act: Act, params: &[f32]) -> Self {
        let mut net = Mlp::new(sizes, act, &mut Rng::new(0));
        net.set_params(params);
        net
    }

    pub fn layers(&self) -> usize {
        self.sizes.len() - 1
    }

    pub fn n_params(&self) -> usize {
        self.w.iter().map(Vec::len).sum::<usize>() + self.b.iter().map(Vec::len).sum::<usize>()
    }

    /// Multiply every weight (not the biases) by `gain`: a larger or smaller start.
    pub fn scale_weights(&mut self, gain: f32) {
        for p in self.w.iter_mut().flatten() {
            *p *= gain;
        }
    }

    /// All parameters as one vector (weights then biases, layer by layer).
    pub fn params(&self) -> Vec<f32> {
        let mut out = Vec::with_capacity(self.n_params());
        for l in 0..self.layers() {
            out.extend_from_slice(&self.w[l]);
            out.extend_from_slice(&self.b[l]);
        }
        out
    }

    pub fn set_params(&mut self, p: &[f32]) {
        let mut i = 0;
        for l in 0..self.layers() {
            let n = self.w[l].len();
            self.w[l].copy_from_slice(&p[i..i + n]);
            i += n;
            let n = self.b[l].len();
            self.b[l].copy_from_slice(&p[i..i + n]);
            i += n;
        }
    }

    /// The gradient left by the last [`Mlp::backward`], in the layout of [`Mlp::params`].
    pub fn grad(&self) -> Vec<f32> {
        let mut out = Vec::with_capacity(self.n_params());
        for l in 0..self.layers() {
            out.extend_from_slice(&self.gw[l]);
            out.extend_from_slice(&self.gb[l]);
        }
        out
    }

    /// Run `n` inputs through the network. `x` holds one sample after another,
    /// `sizes[0]` values each. Returns the outputs (the last layer is linear): `n`
    /// values per output neuron, one neuron after another.
    pub fn forward(&mut self, x: &[f32], n: usize) -> &[f32] {
        let n_in = self.sizes[0];
        grow(&mut self.a[0], n * n_in);
        if n_in == 1 {
            self.a[0][..n].copy_from_slice(&x[..n]);
        } else if n > 0 {
            for (i, column) in self.a[0][..n * n_in].chunks_exact_mut(n).enumerate() {
                for (v, sample) in column.iter_mut().zip(x.chunks_exact(n_in)) {
                    *v = sample[i];
                }
            }
        }
        self.hot_k = 0;
        self.run(n)
    }

    /// The same for inputs that are zero except for `k` units equal to one (one-hot
    /// codes, or several side by side): `active` lists those `k` indices for each
    /// sample. The first layer is then a few additions per neuron instead of a sum
    /// over every input.
    pub fn forward_hot(&mut self, active: &[u32], k: usize, n: usize) -> &[f32] {
        assert!(k >= 1, "forward_hot needs at least one active unit per sample");
        self.hot.clear();
        self.hot.extend_from_slice(&active[..n * k]);
        self.hot_k = k;
        self.run(n)
    }

    fn run(&mut self, n: usize) -> &[f32] {
        let layers = self.layers();
        self.batch = n;
        for l in 0..layers {
            let (n_in, n_out) = (self.sizes[l], self.sizes[l + 1]);
            grow(&mut self.z[l], n * n_out);
            grow(&mut self.a[l + 1], n * n_out);
            if n == 0 {
                continue;
            }
            let z = &mut self.z[l][..n * n_out];
            let (w, b) = (&self.w[l], &self.b[l]);
            if l == 0 && self.hot_k > 0 {
                for (j, zj) in z.chunks_exact_mut(n).enumerate() {
                    let row = &w[j * n_in..(j + 1) * n_in];
                    for (v, active) in zj.iter_mut().zip(self.hot.chunks_exact(self.hot_k)) {
                        *v = b[j] + active.iter().map(|&i| row[i as usize]).sum::<f32>();
                    }
                }
            } else {
                let input = &self.a[l][..n * n_in];
                for (j, zj) in z.chunks_exact_mut(n).enumerate() {
                    zj.fill(b[j]);
                    let row = &w[j * n_in..(j + 1) * n_in];
                    let mut i = 0;
                    while i + 4 <= n_in {
                        axpy4(
                            zj,
                            [row[i], row[i + 1], row[i + 2], row[i + 3]],
                            [col(input, i, n), col(input, i + 1, n), col(input, i + 2, n), col(input, i + 3, n)],
                        );
                        i += 4;
                    }
                    while i < n_in {
                        axpy(zj, row[i], col(input, i, n));
                        i += 1;
                    }
                }
            }
            let out = &mut self.a[l + 1][..n * n_out];
            if l + 1 == layers {
                out.copy_from_slice(z);
            } else {
                self.act.apply(z, out);
            }
        }
        &self.a[layers][..n * self.sizes[layers]]
    }

    /// Activations of layer `l` (0 = first hidden layer, `layers() - 1` = the outputs)
    /// from the last forward pass: one column of batch-size values per neuron.
    pub fn layer(&self, l: usize) -> &[f32] {
        &self.a[l + 1][..self.batch * self.sizes[l + 1]]
    }

    /// What neuron `j` of layer `l` did on every sample of the last forward pass.
    pub fn neuron(&self, l: usize, j: usize) -> &[f32] {
        col(&self.a[l + 1], j, self.batch)
    }

    /// Loss of the last forward pass against targets `y`: one value per sample for
    /// the two classification losses, the layout of the outputs for squared error.
    pub fn loss(&mut self, y: &[f32], loss: Loss) -> f32 {
        self.loss_and_delta(y, loss, false)
    }

    /// The loss and, if wanted, its derivative with respect to the outputs (left in
    /// `self.delta`).
    fn loss_and_delta(&mut self, y: &[f32], loss: Loss, want_delta: bool) -> f32 {
        let n = self.batch;
        let k = *self.sizes.last().unwrap();
        if n == 0 {
            return 0.0;
        }
        let out = &self.a[self.sizes.len() - 1][..k * n];
        grow(&mut self.delta, k * n);
        let delta = &mut self.delta[..k * n];
        match loss {
            Loss::Mse => {
                let scale = 1.0 / (k * n) as f32;
                let mut total = 0.0;
                for ((d, &o), &t) in delta.iter_mut().zip(out).zip(y) {
                    let e = o - t;
                    total += e * e;
                    *d = 2.0 * e * scale;
                }
                total * scale
            }
            Loss::Logistic => {
                let scale = 1.0 / n as f32;
                for ((d, &z), &t) in delta.iter_mut().zip(out).zip(y) {
                    *d = (logistic(z) - t) * scale;
                }
                // numerically stable log(1 + e^z) - t z
                out.iter().zip(y).map(|(&z, &t)| z.max(0.0) - z * t + (1.0 + fast_exp(-z.abs())).ln()).sum::<f32>() * scale
            }
            Loss::Softmax => {
                grow(&mut self.scratch, 2 * n);
                let (top, total) = self.scratch[..2 * n].split_at_mut(n);
                top.copy_from_slice(col(out, 0, n));
                for c in 1..k {
                    for (t, &v) in top.iter_mut().zip(col(out, c, n)) {
                        *t = t.max(v);
                    }
                }
                total.fill(0.0);
                for (c, dc) in delta.chunks_exact_mut(n).enumerate() {
                    for (((d, &v), &t), s) in dc.iter_mut().zip(col(out, c, n)).zip(top.iter()).zip(total.iter_mut()) {
                        *d = fast_exp((v - t).max(-60.0));
                        *s += *d;
                    }
                }
                let scale = 1.0 / n as f32;
                let class = |s: usize| (y[s] as usize).min(k - 1);
                let value: f32 = (0..n).map(|s| total[s].ln() + top[s] - out[class(s) * n + s]).sum();
                if want_delta {
                    for s in total.iter_mut() {
                        *s = scale / *s;
                    }
                    for dc in delta.chunks_exact_mut(n) {
                        for (d, &s) in dc.iter_mut().zip(total.iter()) {
                            *d *= s;
                        }
                    }
                    for s in 0..n {
                        delta[class(s) * n + s] -= scale;
                    }
                }
                value * scale
            }
        }
    }

    /// Backpropagate from the last forward pass; fills the gradient and returns the loss.
    pub fn backward(&mut self, y: &[f32], loss: Loss) -> f32 {
        let layers = self.layers();
        let n = self.batch;
        let value = self.loss_and_delta(y, loss, true);
        if n == 0 {
            // nothing to learn from: and not the last batch's gradient over again
            self.gw.iter_mut().chain(&mut self.gb).for_each(|g| g.fill(0.0));
            return value;
        }
        for l in (0..layers).rev() {
            let (n_in, n_out) = (self.sizes[l], self.sizes[l + 1]);
            let delta = &self.delta[..n_out * n];
            let (gw, gb) = (&mut self.gw[l], &mut self.gb[l]);
            if l == 0 && self.hot_k > 0 {
                gw.fill(0.0);
                for (j, dj) in delta.chunks_exact(n).enumerate() {
                    gb[j] = sum(dj);
                    let row = &mut gw[j * n_in..(j + 1) * n_in];
                    for (&d, active) in dj.iter().zip(self.hot.chunks_exact(self.hot_k)) {
                        for &i in active {
                            row[i as usize] += d;
                        }
                    }
                }
            } else {
                let input = &self.a[l][..n_in * n];
                for (j, dj) in delta.chunks_exact(n).enumerate() {
                    gb[j] = sum(dj);
                    for (i, g) in gw[j * n_in..(j + 1) * n_in].iter_mut().enumerate() {
                        *g = dot(dj, col(input, i, n));
                    }
                }
            }
            if l > 0 {
                grow(&mut self.delta_prev, n_in * n);
                let prev = &mut self.delta_prev[..n_in * n];
                let w = &self.w[l];
                for (i, pi) in prev.chunks_exact_mut(n).enumerate() {
                    pi.fill(0.0);
                    let at = |j: usize| w[j * n_in + i];
                    let mut j = 0;
                    while j + 4 <= n_out {
                        axpy4(
                            pi,
                            [at(j), at(j + 1), at(j + 2), at(j + 3)],
                            [col(delta, j, n), col(delta, j + 1, n), col(delta, j + 2, n), col(delta, j + 3, n)],
                        );
                        j += 4;
                    }
                    while j < n_out {
                        axpy(pi, at(j), col(delta, j, n));
                        j += 1;
                    }
                }
                self.act.scale_by_derivative(&self.z[l - 1][..n_in * n], &self.a[l][..n_in * n], prev);
                std::mem::swap(&mut self.delta, &mut self.delta_prev);
            }
        }
        value
    }

    fn update_all(&mut self, rule: Rule) {
        for l in 0..self.layers() {
            update(rule, &mut self.w[l], &self.gw[l], &mut self.mw[l], &mut self.vw[l]);
            update(rule, &mut self.b[l], &self.gb[l], &mut self.mb[l], &mut self.vb[l]);
        }
    }

    /// Count one more step. (Saturating: at full speed a small network takes 2^32
    /// steps in under an hour, and a count wrapped to zero would make Adam's step
    /// size 0/0.)
    fn count_step(&mut self) {
        self.steps = self.steps.saturating_add(1);
    }

    /// Adam's step size at the current step count.
    fn adam_step_size(&self, lr: f32) -> f32 {
        adam_step_size(self.steps, lr)
    }

    /// One Adam update using the current gradient.
    pub fn adam(&mut self, lr: f32) {
        self.count_step();
        self.update_all(Rule::Adam { step: self.adam_step_size(lr) });
    }

    /// One Muon update using the current gradient (which it overwrites). For each
    /// weight matrix: a running average of the gradient with decay `beta`, the
    /// Nesterov blend of that average with the gradient, the blend made orthogonal,
    /// and a step of size `lr` along it (times `sqrt(rows / cols)` for tall matrices),
    /// exactly as in the reference implementation. Biases are vectors, which cannot be
    /// made orthogonal; they take an Adam step at the same learning rate.
    ///
    /// The reference recommends Muon for hidden layers only, with Adam for the first
    /// and last. Here it is applied to every weight matrix, so that it has something
    /// to act on in a network with one hidden layer.
    fn muon(&mut self, lr: f32, beta: f32) {
        self.count_step();
        for l in 0..self.layers() {
            let (rows, cols) = (self.sizes[l + 1], self.sizes[l]);
            let (g, m) = (&mut self.gw[l], &mut self.mw[l]);
            for (g, m) in g.iter_mut().zip(m.iter_mut()) {
                *m = flush(beta * *m + (1.0 - beta) * *g);
                *g = (1.0 - beta) * *g + beta * *m;
            }
            orthogonalise(g, rows, cols, &mut self.ortho);
            let step = lr * (rows as f32 / cols as f32).max(1.0).sqrt();
            for (p, &u) in self.w[l].iter_mut().zip(g.iter()) {
                *p -= step * u;
            }
            let rule = Rule::Adam { step: self.adam_step_size(lr) };
            update(rule, &mut self.b[l], &self.gb[l], &mut self.mb[l], &mut self.vb[l]);
        }
    }

    /// Note which optimiser is about to step, starting it from a clean state if the
    /// last step was another's. (Callers used to have to ask for this themselves, in
    /// a message that could arrive some hundred steps before the new setting did.)
    fn using(&mut self, opt: Optim) {
        if self.last.is_some_and(|last| last != opt) {
            self.reset_optimizer();
        }
        self.last = Some(opt);
    }

    /// Forget the optimiser's accumulated state. Switching optimiser does this by
    /// itself; this is for starting the same one afresh.
    pub fn reset_optimizer(&mut self) {
        for v in self.mw.iter_mut().chain(&mut self.vw).chain(&mut self.mb).chain(&mut self.vb) {
            v.fill(0.0);
        }
        self.steps = 0;
    }

    /// One update with the chosen optimiser, using the current gradient. `momentum`
    /// matters only to the two momentum methods and to Muon.
    ///
    /// `decay` is decoupled weight decay, the same for every optimiser: each step the
    /// weights (not the biases) shrink by the factor `1 - lr * decay`, separately
    /// from the gradient step. For plain gradient descent that is the usual L2
    /// penalty; for Adam it is AdamW.
    ///
    /// A gradient of absurd size is scaled back, which keeps a diverging run from
    /// turning into NaN before anyone has seen it diverge.
    pub fn step(&mut self, opt: Optim, lr: f32, momentum: f32, decay: f32) {
        self.using(opt);
        // (summed in f64: the square of a gradient this is meant for overflows f32)
        let norm2: f64 = self.gw.iter().chain(&self.gb).flatten().map(|&g| g as f64 * g as f64).sum();
        if norm2 > 1e6 {
            let k = (1e3 / norm2.sqrt()) as f32;
            self.gw.iter_mut().chain(&mut self.gb).flatten().for_each(|g| *g *= k);
        }
        if decay > 0.0 {
            // (never below zero: a decay larger than the weight would otherwise
            // flip its sign and grow it, step after step)
            let keep = (1.0 - lr * decay).max(0.0);
            for p in self.w.iter_mut().flatten() {
                *p = flush(*p * keep);
            }
        }
        match opt {
            Optim::Adam => self.adam(lr),
            Optim::Muon => self.muon(lr, momentum),
            other => {
                self.count_step();
                self.update_all(match other {
                    Optim::Sgd => Rule::Sgd { lr },
                    Optim::Momentum => Rule::Momentum { lr, mu: momentum },
                    Optim::Nesterov => Rule::Nesterov { lr, mu: momentum },
                    Optim::AdaGrad => Rule::AdaGrad { lr },
                    _ => Rule::RmsProp { lr },
                });
            }
        }
    }

    /// Forward, backward and an Adam step on one batch. Returns the loss before the step.
    pub fn train(&mut self, x: &[f32], y: &[f32], n: usize, lr: f32, loss: Loss) -> f32 {
        self.forward(x, n);
        let value = self.backward(y, loss);
        self.using(Optim::Adam);
        self.adam(lr);
        value
    }

    /// Loss on a batch without changing anything.
    pub fn eval(&mut self, x: &[f32], y: &[f32], n: usize, loss: Loss) -> f32 {
        self.forward(x, n);
        self.loss(y, loss)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gradient_matches_finite_differences() {
        let cases = Act::ALL.into_iter().flat_map(|a| [(a, Loss::Mse), (a, Loss::Logistic), (a, Loss::Softmax)]);
        for (act, loss) in cases {
            let mut rng = Rng::new(1);
            let outputs = if loss == Loss::Softmax { 3 } else { 1 };
            let mut net = Mlp::new(&[2, 5, 4, outputs], act, &mut rng);
            let n = 7;
            let x: Vec<f32> = (0..n * 2).map(|_| rng.range(-1.0, 1.0)).collect();
            let y: Vec<f32> = (0..n)
                .map(|i| match loss {
                    Loss::Logistic => (i % 2) as f32,
                    Loss::Softmax => (i % 3) as f32,
                    Loss::Mse => rng.range(-1.0, 1.0),
                })
                .collect();
            net.forward(&x, n);
            net.backward(&y, loss);
            let grad = net.grad();
            let p0 = net.params();
            let h = 1e-2;
            // (every third parameter, and the last one: the output layer's bias)
            for i in (0..p0.len()).step_by(3).chain([p0.len() - 1]) {
                let mut p = p0.clone();
                p[i] += h;
                net.set_params(&p);
                let up = net.eval(&x, &y, n, loss);
                p[i] -= 2.0 * h;
                net.set_params(&p);
                let down = net.eval(&x, &y, n, loss);
                let fd = (up - down) / (2.0 * h);
                assert!((fd - grad[i]).abs() < 2e-3 + 0.03 * fd.abs(), "{act:?} {loss:?} param {i}: {fd} vs {}", grad[i]);
            }
        }
    }

    #[test]
    fn fast_tanh_is_close_and_monotone() {
        let mut worst = 0.0f32;
        let mut prev = -1.0f32;
        for i in -8000..=8000 {
            let x = i as f32 * 0.001;
            let y = fast_tanh(x);
            worst = worst.max((y - x.tanh()).abs());
            assert!(y >= prev - 1e-7, "not monotone at {x}");
            prev = y;
        }
        assert!(worst < 1e-3, "worst error {worst}");
        assert_eq!(fast_tanh(0.0), 0.0);
        assert!((fast_tanh(30.0) - 1.0).abs() < 1e-3 && (fast_tanh(-30.0) + 1.0).abs() < 1e-3);
        assert!(fast_tanh(30.0) <= 1.0 && fast_tanh(-30.0) >= -1.0);
    }

    #[test]
    fn activations_are_what_they_say() {
        // GELU and SiLU against direct formulas; the Hermite pair against the polynomials
        for i in -60..=60 {
            let z = i as f32 * 0.1;
            let phi = 0.5 * (1.0 + libm_erf(z as f64 / std::f64::consts::SQRT_2)) as f32;
            assert!((normal_cdf(z) - phi).abs() < 2e-6, "cdf at {z}");
            assert!((Act::Gelu.f(z) - z * phi).abs() < 2e-5);
            assert!((Act::Silu.f(z) - z / (1.0 + (-z).exp())).abs() < 1e-5);
            assert!((Act::Sin.f(z * 7.0) - (z * 7.0).sin()).abs() < 2e-6, "sin at {}", z * 7.0);
            assert!((Act::Hermite2.f(z) - crate::math::hermite::h(2, z as f64) as f32).abs() < 1e-4);
            assert!((Act::Hermite3.f(z) - crate::math::hermite::h(3, z as f64) as f32).abs() < 1e-3);
            assert!((Act::ReluSquared.f(z) - z.max(0.0).powi(2)).abs() < 1e-6);
        }
        for i in -8600..=8700 {
            let x = i as f32 * 0.01;
            let (fast, exact) = (fast_exp(x), x.exp());
            assert!((fast - exact).abs() <= 2e-6 * exact, "exp({x}): {fast} vs {exact}");
        }
        assert!(fast_exp(-200.0) >= 0.0 && fast_exp(200.0).is_finite());
        // every derivative agrees with a finite difference of its own function
        for act in Act::ALL {
            for i in -40..=40 {
                let z = i as f32 * 0.1 + 0.013;
                let fd = (act.f(z + 1e-3) - act.f(z - 1e-3)) / 2e-3;
                assert!((act.df(z, act.f(z)) - fd).abs() < 3e-3 + 0.01 * fd.abs(), "{act:?} at {z}");
            }
        }
    }

    /// erf by its power series, as an independent reference for the test above.
    fn libm_erf(x: f64) -> f64 {
        let (mut term, mut sum) = (x, x);
        for n in 1..200 {
            term *= -x * x / n as f64;
            sum += term / (2 * n + 1) as f64;
        }
        2.0 / std::f64::consts::PI.sqrt() * sum
    }

    /// Each rule against the formula it was published with, starting from a state
    /// with history in it, so that no rule can pass by being one of the others.
    #[test]
    fn each_rule_takes_its_published_step() {
        let (p0, g, m0, v0, lr, mu) = (1.0f64, 0.5f64, 0.3f64, 0.2f64, 0.1f64, 0.9f64);
        let one = |rule: Rule| {
            let (mut p, mut m, mut v) = ([p0 as f32], [m0 as f32], [v0 as f32]);
            update(rule, &mut p, &[g as f32], &mut m, &mut v);
            [p[0] as f64, m[0] as f64, v[0] as f64]
        };
        let near = |got: [f64; 3], want: [f64; 3]| got.iter().zip(want).all(|(a, b)| (a - b).abs() < 1e-6);
        let (lr32, mu32) = (lr as f32, mu as f32);
        assert!(near(one(Rule::Sgd { lr: lr32 }), [p0 - lr * g, m0, v0]));
        // momentum as PyTorch writes it: m <- mu m + g, then a step along m
        let m1 = mu * m0 + g;
        assert!(near(one(Rule::Momentum { lr: lr32, mu: mu32 }), [p0 - lr * m1, m1, v0]));
        // Nesterov: the step looks ahead along the new momentum
        assert!(near(one(Rule::Nesterov { lr: lr32, mu: mu32 }), [p0 - lr * (g + mu * m1), m1, v0]));
        // AdaGrad sums the squares; RMSProp averages them (decay 0.9); epsilon outside the root
        let v1 = v0 + g * g;
        assert!(near(one(Rule::AdaGrad { lr: lr32 }), [p0 - lr * g / (v1.sqrt() + 1e-8), m0, v1]));
        let v1 = 0.9 * v0 + 0.1 * g * g;
        assert!(near(one(Rule::RmsProp { lr: lr32 }), [p0 - lr * g / (v1.sqrt() + 1e-8), m0, v1]));
        // Adam, at its seventh step, against the paper's form with both corrections
        // written out (the folded step size moves epsilon, by a part in a million here)
        let (m1, v1, t) = (0.9 * m0 + 0.1 * g, 0.999 * v0 + 0.001 * g * g, 7);
        let (m_hat, v_hat) = (m1 / (1.0 - 0.9f64.powi(t)), v1 / (1.0 - 0.999f64.powi(t)));
        let got = one(Rule::Adam { step: adam_step_size(t as u32, lr32) });
        assert!(near(got, [p0 - lr * m_hat / v_hat.sqrt(), m1, v1]), "{got:?}");
        // its very first step is the learning rate itself, whatever the gradient's size
        for g in [3.0f32, -0.02] {
            let (mut p, mut m, mut v) = ([0.0f32], [0.0f32], [0.0f32]);
            update(Rule::Adam { step: adam_step_size(1, 0.01) }, &mut p, &[g], &mut m, &mut v);
            assert!((p[0] + 0.01 * g.signum()).abs() < 1e-6, "{}", p[0]);
        }
        // (and asking for the step size before any step is counted is not 0/0)
        assert_eq!(adam_step_size(0, 0.01), adam_step_size(1, 0.01));
    }

    #[test]
    fn a_change_of_optimiser_starts_the_new_one_afresh() {
        let x = [0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0];
        let y = [0.0, 1.0, 1.0, 0.0];
        let mut net = Mlp::new(&[2, 8, 1], Act::Tanh, &mut Rng::new(3));
        for _ in 0..50 {
            net.forward(&x, 4);
            net.backward(&y, Loss::Logistic);
            net.step(Optim::Momentum, 0.1, 0.9, 0.0);
        }
        // one Adam step with the momentum run's leftovers still in place must equal
        // one from a network that was told to forget them
        let mut fresh = net.clone();
        fresh.reset_optimizer();
        for n in [&mut net, &mut fresh] {
            n.forward(&x, 4);
            n.backward(&y, Loss::Logistic);
            n.step(Optim::Adam, 0.01, 0.9, 0.0);
        }
        assert_eq!(net.params(), fresh.params());
        assert_eq!(net.steps, 1);
    }

    #[test]
    fn every_optimiser_learns_xor() {
        let x = [0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0];
        let y = [0.0, 1.0, 1.0, 0.0];
        for opt in Optim::ALL {
            let mut rng = Rng::new(3);
            let mut net = Mlp::new(&[2, 8, 1], Act::Tanh, &mut rng);
            let mut loss = 1.0;
            for _ in 0..3000 {
                net.forward(&x, 4);
                loss = net.backward(&y, Loss::Logistic);
                net.step(opt, opt.default_lr(), 0.9, 0.0);
            }
            assert!(loss < 0.1, "{opt:?} ended at loss {loss}");
        }
    }

    /// Prints what the passes cost per activation (run with --ignored --nocapture).
    #[test]
    #[ignore]
    fn cost() {
        let mut rng = Rng::new(5);
        let grid: Vec<f32> = (0..7744 * 2).map(|_| rng.range(-1.2, 1.2)).collect();
        let y: Vec<f32> = (0..32).map(|i| (i % 2) as f32).collect();
        for act in Act::ALL {
            let mut net = Mlp::new(&[2, 8, 8, 8, 8, 1], act, &mut rng);
            let t = std::time::Instant::now();
            for _ in 0..200 {
                net.forward(&grid, 7744);
            }
            let display = t.elapsed().as_secs_f64() * 1e3 / 200.0;
            let t = std::time::Instant::now();
            for _ in 0..20_000 {
                net.forward(&grid, 32);
                net.backward(&y, Loss::Logistic);
                net.step(Optim::Adam, 1e-4, 0.9, 0.0);
            }
            let train = t.elapsed().as_secs_f64() * 1e6 / 20_000.0;
            println!("{:<12} {display:>6.3} ms per 7744-point forward pass   {train:>6.2} us per training step of 32   (4 x 8 network)", act.name());
        }
    }

    #[test]
    fn orthogonalise_evens_out_the_singular_values() {
        let mut rng = Rng::new(6);
        let mut ws = Ortho::default();
        for (rows, cols) in [(4usize, 9usize), (9, 4), (5, 5), (1, 7), (7, 1)] {
            // rows of very different sizes: singular values spread over a factor of thirty
            let mut g: Vec<f32> = (0..rows * cols).map(|_| rng.normal()).collect();
            let m = rows.min(cols);
            for i in 0..rows {
                for j in 0..cols {
                    let k = if rows <= cols { i } else { j };
                    g[i * cols + j] *= [1.0, 0.3, 0.1, 0.03, 0.5][k % 5];
                }
            }
            let before = g.clone();
            orthogonalise(&mut g, rows, cols, &mut ws);
            // the short side's vectors (rows of a wide matrix, columns of a tall one)
            // are now close to orthonormal
            let short = |k: usize| -> Vec<f32> {
                if rows <= cols {
                    g[k * cols..(k + 1) * cols].to_vec()
                } else {
                    (0..rows).map(|i| g[i * cols + k]).collect()
                }
            };
            for a in 0..m {
                for b in a..m {
                    let d = dot(&short(a), &short(b));
                    if a == b {
                        assert!((0.3..1.5).contains(&d), "{rows}x{cols}: |v{a}|^2 = {d}");
                    } else {
                        assert!(d.abs() < 0.35, "{rows}x{cols}: v{a}.v{b} = {d}");
                    }
                }
            }
            // and it still points the way the gradient did
            assert!(dot(&g, &before) > 0.0);
            // a tall matrix is handled as the transpose of the wide one
            let mut t: Vec<f32> = (0..rows * cols).map(|k| before[(k % rows) * cols + k / rows]).collect();
            orthogonalise(&mut t, cols, rows, &mut ws);
            for i in 0..rows {
                for j in 0..cols {
                    assert!((t[j * rows + i] - g[i * cols + j]).abs() < 1e-5);
                }
            }
        }
        // nothing to orthogonalise: stays zero, no NaN
        let mut zero = vec![0.0f32; 12];
        orthogonalise(&mut zero, 3, 4, &mut ws);
        assert!(zero.iter().all(|&v| v == 0.0));
    }

    #[test]
    fn muon_moves_every_direction_at_the_same_rate() {
        // one linear layer, squared error: the gradient is far larger along some
        // directions than others, yet after a Muon step the weights have moved by a
        // matrix whose singular values are all near the learning rate
        let mut rng = Rng::new(8);
        let mut net = Mlp::new(&[6, 3], Act::Tanh, &mut rng);
        let n = 40;
        let x: Vec<f32> = (0..n * 6).map(|i| rng.normal() * [3.0, 1.0, 0.3, 0.1, 1.0, 0.03][i % 6]).collect();
        let y: Vec<f32> = (0..n * 3).map(|_| rng.normal()).collect();
        let before = net.w[0].clone();
        net.forward(&x, n);
        net.backward(&y, Loss::Mse);
        net.step(Optim::Muon, 0.02, 0.95, 0.0);
        let moved: Vec<f32> = net.w[0].iter().zip(&before).map(|(a, b)| (a - b) / 0.02).collect();
        for a in 0..3 {
            for b in a..3 {
                let d = dot(&moved[a * 6..(a + 1) * 6], &moved[b * 6..(b + 1) * 6]);
                assert!(if a == b { (0.3..1.5).contains(&d) } else { d.abs() < 0.35 }, "rows {a},{b}: {d}");
            }
        }
    }

    #[test]
    fn a_network_survives_being_rebuilt_from_its_parameters() {
        let mut rng = Rng::new(12);
        let mut net = Mlp::new(&[3, 5, 2], Act::Gelu, &mut rng);
        let x: Vec<f32> = (0..4 * 3).map(|_| rng.normal()).collect();
        let mut copy = Mlp::from_params(&net.sizes, net.act, &net.params());
        assert_eq!(net.forward(&x, 4), copy.forward(&x, 4));
    }

    #[test]
    fn kernels_agree_with_their_plain_versions() {
        assert!(kernels_agree());
        // and the plain versions with the obvious ones
        let mut rng = Rng::new(3);
        let (a, b): (Vec<f32>, Vec<f32>) = (0..100).map(|_| (rng.normal(), rng.normal())).unzip();
        let naive: f64 = a.iter().zip(&b).map(|(x, y)| *x as f64 * *y as f64).sum();
        assert!((dot(&a, &b) as f64 - naive).abs() < 1e-4);
        assert!((sum(&a) as f64 - a.iter().map(|x| *x as f64).sum::<f64>()).abs() < 1e-4);
    }

    #[test]
    fn layout_is_one_column_per_neuron() {
        let mut rng = Rng::new(2);
        let mut net = Mlp::new(&[3, 4, 2], Act::Tanh, &mut rng);
        let x: Vec<f32> = (0..5 * 3).map(|_| rng.range(-1.0, 1.0)).collect();
        let out = net.forward(&x, 5).to_vec();
        assert_eq!(out.len(), 10);
        for s in 0..5 {
            // by hand, for sample s
            let h: Vec<f32> = (0..4).map(|j| Act::Tanh.f(net.b[0][j] + (0..3).map(|i| net.w[0][j * 3 + i] * x[s * 3 + i]).sum::<f32>())).collect();
            for j in 0..4 {
                assert!((net.neuron(0, j)[s] - h[j]).abs() < 1e-5);
                assert_eq!(net.layer(0)[j * 5 + s], net.neuron(0, j)[s]);
            }
            for c in 0..2 {
                let o = net.b[1][c] + (0..4).map(|j| net.w[1][c * 4 + j] * h[j]).sum::<f32>();
                assert!((out[c * 5 + s] - o).abs() < 1e-5);
            }
        }
    }

    #[test]
    fn index_inputs_match_dense_inputs() {
        let mut rng = Rng::new(4);
        let (p, n) = (5usize, 9usize);
        let mut net = Mlp::new(&[2 * p, 6, 4, p], Act::Gelu, &mut rng);
        let pairs: Vec<(usize, usize)> = (0..n).map(|_| (rng.below(p), rng.below(p))).collect();
        let y: Vec<f32> = pairs.iter().map(|&(a, b)| ((a + b) % p) as f32).collect();
        let mut dense = vec![0.0f32; n * 2 * p];
        let mut active = Vec::new();
        for (s, &(a, b)) in pairs.iter().enumerate() {
            dense[s * 2 * p + a] = 1.0;
            dense[s * 2 * p + p + b] = 1.0;
            active.extend([a as u32, (p + b) as u32]);
        }
        let out_dense = net.forward(&dense, n).to_vec();
        let loss_dense = net.backward(&y, Loss::Softmax);
        let grad_dense = net.grad();
        let out_hot = net.forward_hot(&active, 2, n).to_vec();
        let loss_hot = net.backward(&y, Loss::Softmax);
        let grad_hot = net.grad();
        assert!((loss_dense - loss_hot).abs() < 1e-6);
        for (a, b) in out_dense.iter().zip(&out_hot) {
            assert!((a - b).abs() < 1e-5);
        }
        for (a, b) in grad_dense.iter().zip(&grad_hot) {
            assert!((a - b).abs() < 1e-6, "{a} vs {b}");
        }
    }

    #[test]
    fn nothing_leaves_the_normal_range() {
        // far outside anything sensible, values must stay normal numbers (or zero)
        let normal = |v: f32| v == 0.0 || v.is_normal();
        for act in [Act::Tanh, Act::Sigmoid, Act::Silu, Act::Gelu, Act::Relu, Act::Sin] {
            for z in [-1e30f32, -1e6, -700.0, -90.0, -45.0, -30.0, -13.0, 13.0, 30.0, 45.0, 90.0, 700.0, 1e6] {
                let a = act.f(z);
                assert!(normal(a) && normal(act.df(z, a)), "{act:?} at {z}: {a} {}", act.df(z, a));
            }
        }
        // a network trained far past saturation keeps a clean optimiser state
        let mut rng = Rng::new(9);
        let x = [0.0f32, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 0.0];
        let y = [0.0f32, 0.0, 1.0, 1.0];
        for opt in Optim::ALL {
            let mut net = Mlp::new(&[2, 6, 1], Act::Tanh, &mut rng);
            for _ in 0..60_000 {
                net.forward(&x, 4);
                net.backward(&y, Loss::Logistic);
                net.step(opt, 0.05, 0.9, 1e-3);
            }
            let state = net.mw.iter().chain(&net.vw).chain(&net.mb).chain(&net.vb).chain(&net.w).chain(&net.b).chain(&net.gw).flatten();
            for &v in state {
                assert!(normal(v), "{opt:?}: {v:e}");
            }
        }
    }

    #[test]
    fn learns_xor() {
        let mut rng = Rng::new(3);
        let mut net = Mlp::new(&[2, 6, 1], Act::Tanh, &mut rng);
        let x = [0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0];
        let y = [0.0, 1.0, 1.0, 0.0];
        let mut loss = 1.0;
        for _ in 0..800 {
            loss = net.train(&x, &y, 4, 0.05, Loss::Logistic);
        }
        assert!(loss < 0.05, "xor loss {loss}");
        let out = net.forward(&x, 4).to_vec();
        assert!(out[0] < 0.0 && out[1] > 0.0 && out[2] > 0.0 && out[3] < 0.0);
    }

    /// Settings the sliders allow that used to end in nonsense: a decay larger than
    /// the weight, a gradient too large to square, a step count past what its counter
    /// holds, a batch of nothing.
    #[test]
    fn extreme_settings_stay_sane() {
        let mut rng = Rng::new(3);
        let fresh = |rng: &mut Rng| Mlp::new(&[2, 3, 1], Act::Tanh, rng);
        // lr 1, decay 3, no gradient: the weights go to zero and stay there
        let mut net = fresh(&mut rng);
        net.gw.iter_mut().chain(&mut net.gb).for_each(|g| g.fill(0.0));
        for _ in 0..6 {
            net.step(Optim::Sgd, 1.0, 0.0, 3.0);
        }
        assert!(net.w.iter().flatten().all(|&w| w == 0.0));
        // a gradient of size 1e19 is scaled back, not zeroed: the step has length lr * 1000
        let mut net = fresh(&mut rng);
        let before = net.params();
        net.gw.iter_mut().chain(&mut net.gb).for_each(|g| g.fill(1e19));
        net.step(Optim::Sgd, 1e-3, 0.0, 0.0);
        let moved: f32 = net.params().iter().zip(&before).map(|(a, b)| (a - b) * (a - b)).sum::<f32>().sqrt();
        assert!((moved - 1.0).abs() < 1e-3, "moved {moved}");
        // the step counter stops at its largest value rather than starting again
        let mut net = fresh(&mut rng);
        net.steps = u32::MAX - 1;
        for _ in 0..3 {
            net.gw.iter_mut().for_each(|g| g.fill(0.1));
            net.step(Optim::Adam, 0.01, 0.9, 0.0);
        }
        assert!(net.steps == u32::MAX && net.params().iter().all(|p| p.is_finite()));
        // an empty batch leaves no gradient to apply
        let mut net = fresh(&mut rng);
        net.train(&[0.3, -0.2, 0.8, 0.1], &[1.0, 0.0], 2, 0.05, Loss::Logistic);
        let before = net.params();
        net.forward(&[], 0);
        net.backward(&[], Loss::Logistic);
        net.step(Optim::Sgd, 0.5, 0.0, 0.0);
        assert_eq!(net.params(), before);
        // the two dot products agree on slices of unequal length
        let (a, b): (Vec<f32>, Vec<f32>) = ((0..33).map(|i| 1.0 + i as f32).collect(), vec![1.0; 17]);
        assert_eq!(dot(&a, &b), (1..=17).sum::<i32>() as f32);
    }
}

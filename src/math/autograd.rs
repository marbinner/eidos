//! Reverse-mode automatic differentiation on small dense matrices: the engine under
//! the backpropagation scenes (the networks of `arch` are built on it).
//!
//! A [`Tape`] records a computation as it is carried out: each operation is a node,
//! whose value is worked out the moment it is recorded. [`Tape::backward`] then sweeps
//! the record in reverse. It starts the loss's gradient at 1 and hands each node the
//! derivative of the loss with respect to its value (its gradient), and each node
//! passes that on to its inputs by the chain rule. Afterwards the gradient of every
//! node can be read, not just the parameters': the inputs, every intermediate value,
//! every attention pattern. That is what the scenes draw.
//!
//! Conventions, kept throughout:
//!
//! - Everything is `f64`, dense and row-major: entry (i, j) of a `rows` by `cols`
//!   matrix is `value[i * cols + j]`.
//! - A batch is a matrix with one ROW per example (in a transformer, one row per
//!   position of the sequence). The weights of a layer are stored with one row per
//!   output unit (outputs by inputs), so that for one example written as a column `x`
//!   the layer is `W x`, and for a batch `X` it is `X Wᵀ`, which is [`Tape::matmul_t`].
//! - A gradient has the shape of the thing it is the gradient of.
//! - A [`Var`] is a handle to a node, valid until the tape is cleared.
//!
//! What happens when something does not fit: a shape that does not fit an operation
//! is a mistake of the caller, caught by a `debug_assert`; in a release build the
//! operation instead makes a node of the shape it would have had, filled with NaN, that
//! passes no gradient back, so that the mistake shows on the screen instead of ending
//! the program. Numbers that are not finite are carried through by the arithmetic
//! (a NaN in gives NaN out) and never cause a panic. Sizes of zero are allowed
//! everywhere, and a mean over nothing (the loss of an empty batch) is 0.
//!
//! Cost: values, gradients and scratch numbers live in a few long buffers that
//! [`Tape::clear`] empties without freeing, so a scene that clears the tape and runs a
//! forward and a backward pass every frame allocates nothing after the first. A
//! backward sweep costs about twice the forward pass (a matrix product hands back two
//! products), and works out the gradient of every node on the way.
//!
//! An example: one layer of three tanh units on a batch of four points, and its
//! gradient.
//!
//! ```text
//!     let mut tape = Tape::new();
//!     let x = tape.constant(4, 2, &points);        // 4 examples, 2 coordinates each
//!     let w = tape.param(3, 2, &weights);          // 3 units, 2 inputs each
//!     let b = tape.param(1, 3, &biases);
//!     let z = tape.matmul_t(x, w);                 // z = x Wᵀ, 4 by 3
//!     let z = tape.add_row(z, b);                  // + b on every row
//!     let a = tape.act(z, Act::Tanh);
//!     let y = tape.constant(4, 3, &wanted);
//!     let loss = tape.mse(a, y);
//!     tape.backward(loss);
//!     tape.grad(w)                                 // 3 by 2, like w
//!     tape.grad(z)                                 // 4 by 3: what came back to each unit
//! ```

// (Nothing calls this yet: the backpropagation scenes are its users.)
#![allow(dead_code)]

use std::f64::consts::{FRAC_1_SQRT_2, FRAC_2_SQRT_PI};

/// The target of a row that has none: [`Tape::cross_entropy`] leaves a row out when its
/// target is this, or anything else that is not a column of the logits. (In a
/// sequence the last position has no next token to predict.)
pub const NO_TARGET: usize = usize::MAX;

/// A handle to one node of a [`Tape`]: a matrix the tape has recorded. It is a number
/// (cheap to copy), valid until the tape is cleared; nodes are numbered in the order
/// they were recorded, so every node's inputs have smaller numbers than it.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct Var(usize);

impl Var {
    /// Where the node is on its tape: 0 for the first one recorded.
    pub fn index(self) -> usize {
        self.0
    }
}

// ----- activation functions ----------------------------------------------------------

/// The nonlinearities a unit can apply, each with its slope. On the way back a unit
/// multiplies the gradient arriving at its output by its slope at the input it had on
/// the way forward: the slope is the unit's gate.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Act {
    /// x itself; slope 1 everywhere (a network of these is only a linear map).
    Identity,
    /// max(0, x); slope 1 above 0 and 0 at and below it: a gate open or shut.
    Relu,
    /// x above 0, 0.01 x below; slope 1 or 0.01 (at 0 itself, 0.01): never quite shut.
    LeakyRelu,
    /// tanh x; slope 1 − tanh² x, at most 1 (at 0), near 0 once |x| is large.
    Tanh,
    /// The logistic function 1 / (1 + e^−x); slope σ(x) σ(−x), at most 1/4 (at 0).
    Sigmoid,
    /// x Φ(x), with Φ the standard normal distribution function: the exact GELU, not
    /// its tanh approximation. Slope Φ(x) + x φ(x) (φ the normal density): a little
    /// above 1 near x = 1.4, a little below 0 for x under about −0.75, never exactly 0
    /// except at x ≈ −0.7518.
    Gelu,
    /// x σ(x), also called swish; slope σ(x) (1 + x σ(−x)), a little above 1 near
    /// x = 2.4, a little below 0 for x under about −1.28.
    Silu,
    /// x above 0, e^x − 1 below (going to −1); slope 1 above 0 and e^x below.
    Elu,
    /// ln(1 + e^x), a smoothed ReLU; slope σ(x).
    Softplus,
}

impl Act {
    /// All nine, in the order a control would list them.
    pub const ALL: [Act; 9] = [Act::Identity, Act::Relu, Act::LeakyRelu, Act::Tanh, Act::Sigmoid, Act::Gelu, Act::Silu, Act::Elu, Act::Softplus];

    /// The usual name, for a label.
    pub fn name(self) -> &'static str {
        match self {
            Act::Identity => "identity",
            Act::Relu => "ReLU",
            Act::LeakyRelu => "leaky ReLU",
            Act::Tanh => "tanh",
            Act::Sigmoid => "sigmoid",
            Act::Gelu => "GELU",
            Act::Silu => "SiLU",
            Act::Elu => "ELU",
            Act::Softplus => "softplus",
        }
    }

    /// The function itself (see each variant for its formula). At ±∞ it gives its
    /// limits; at NaN, NaN.
    pub fn f(self, x: f64) -> f64 {
        self.f_df(x).0
    }

    /// Its slope at the input `x` (the derivative of [`Act::f`] there; at a kink, the
    /// slope of the piece below it). At ±∞ the limits; at NaN, NaN.
    pub fn df(self, x: f64) -> f64 {
        self.f_df(x).1
    }

    /// The value and the slope together: what a forward pass keeps for the backward
    /// one. (GELU's two share the normal distribution function, its costly part.)
    fn f_df(self, x: f64) -> (f64, f64) {
        match self {
            Act::Identity => (x, if x.is_nan() { x } else { 1.0 }),
            Act::Relu => {
                if x > 0.0 {
                    (x, 1.0)
                } else if x.is_nan() {
                    (x, x)
                } else {
                    (0.0, 0.0)
                }
            }
            Act::LeakyRelu => {
                if x > 0.0 {
                    (x, 1.0)
                } else if x.is_nan() {
                    (x, x)
                } else {
                    (0.01 * x, 0.01)
                }
            }
            Act::Tanh => {
                let t = x.tanh();
                (t, 1.0 - t * t)
            }
            Act::Sigmoid => {
                let (s, r) = sigmoids(x);
                (s, s * r)
            }
            Act::Gelu => {
                if x.is_infinite() {
                    // (x Φ(x) and x φ(x) are ∞ times 0 there)
                    return if x > 0.0 { (x, 1.0) } else { (0.0, 0.0) };
                }
                let (cdf, pdf) = normal_cdf_and_density(x);
                (x * cdf, cdf + x * pdf)
            }
            Act::Silu => {
                if x.is_infinite() {
                    return if x > 0.0 { (x, 1.0) } else { (0.0, 0.0) };
                }
                let (s, r) = sigmoids(x);
                (x * s, s * (1.0 + x * r))
            }
            Act::Elu => {
                if x > 0.0 {
                    (x, 1.0)
                } else {
                    (x.exp_m1(), x.exp())
                }
            }
            Act::Softplus => {
                // ln(1 + e^x) = max(x, 0) + ln(1 + e^−|x|), which cannot overflow
                let e = (-x.abs()).exp();
                let f = if x > 0.0 { x + e.ln_1p() } else { e.ln_1p() };
                (f, if x >= 0.0 { 1.0 / (1.0 + e) } else { e / (1.0 + e) })
            }
        }
    }
}

/// σ(x) = 1 / (1 + e^−x) and σ(−x) = 1 − σ(x), each in a form that loses no digits
/// and cannot overflow, from one exponential.
fn sigmoids(x: f64) -> (f64, f64) {
    let e = (-x.abs()).exp();
    let (big, small) = (1.0 / (1.0 + e), e / (1.0 + e));
    if x >= 0.0 {
        (big, small)
    } else if x.is_nan() {
        (x, x)
    } else {
        (small, big)
    }
}

/// e^(−x²/2). Far out, where only its relative accuracy is left to lose, x² is split
/// into a part that squares exactly and the rest (as W. J. Cody does), so that the
/// rounding of x² costs nothing: Φ(−20) keeps its fifteen digits.
fn gauss(x: f64) -> f64 {
    let a = x.abs();
    if a <= 4.0 {
        (-0.5 * x * x).exp()
    } else if a > 40.0 {
        // (below the smallest double)
        0.0
    } else {
        let h = (a * 16.0).trunc() / 16.0;
        (-0.5 * h * h).exp() * (-0.5 * (a - h) * (a + h)).exp()
    }
}

/// The standard normal distribution function Φ(x) = erfc(−x / √2) / 2 and density
/// φ(x) = e^(−x²/2) / √(2π), together (GELU needs both, and they share the
/// exponential). Φ is good to about fifteen digits, relative, everywhere, also far out
/// in the lower tail where it is tiny. `stats::normal_cdf` is as accurate but goes
/// through the incomplete gamma function, which is far too slow for every unit of a
/// network in every frame; this costs a polynomial and an exponential.
fn normal_cdf_and_density(x: f64) -> (f64, f64) {
    let e = gauss(x);
    let y = x.abs() * FRAC_1_SQRT_2;
    // the smaller of Φ(x) and 1 − Φ(x): erfc(y) / 2
    let tail = if y < 1.0 { 0.5 * (1.0 - erf_taylor(y)) } else { 0.5 * e * erfcx(y) };
    let cdf = if x < 0.0 { tail } else { 1.0 - tail };
    (cdf, e * (0.5 * FRAC_2_SQRT_PI * FRAC_1_SQRT_2))
}

fn normal_cdf(x: f64) -> f64 {
    normal_cdf_and_density(x).0
}

/// The Taylor coefficients of erf about 0: erf(y) = y Σ c_n y^(2n), c_n = (2/√π) (−1)^n
/// / (n! (2n + 1)). Eighteen terms reach the last digit for |y| < 1.
const ERF_TAYLOR: [f64; 18] = {
    let mut c = [0.0; 18];
    let mut factorial = 1.0;
    let mut n = 0;
    while n < 18 {
        if n > 0 {
            factorial *= n as f64;
        }
        let sign = if n % 2 == 0 { 1.0 } else { -1.0 };
        c[n] = sign * FRAC_2_SQRT_PI / (factorial * (2 * n + 1) as f64);
        n += 1;
    }
    c
};

/// Chebyshev coefficients of the scaled complementary error function e^(y²) erfc(y) on
/// [1, 2], [2, 4] and [4, 8]: interpolation at the Chebyshev points of a reference
/// worked out to 160 digits (the positive series of erf, in Python's `decimal`). Each
/// piece is good to 5e-15, relative (the tests hold the whole to `stats::erfc` and to
/// values printed to seventeen digits).
const ERFCX_PIECES: [(f64, f64, &[f64]); 3] = [
    (
        1.0,
        2.0,
        &[
            0.331427281639452,
            -0.08500213770674317,
            0.009951127341466118,
            -0.0010810888102208056,
            0.00011023243631196334,
            -1.0636830518073736e-05,
            9.775012679883824e-07,
            -8.597922881807089e-08,
            7.267708457270316e-09,
            -5.923544617777073e-10,
            4.66839344959169e-11,
            -3.5659364350237864e-12,
            2.6454024156426686e-13,
            -1.942890293094024e-14,
            7.22570151860206e-16,
        ],
    ),
    (
        2.0,
        4.0,
        &[
            0.18742956902622407,
            -0.05794652883669096,
            0.008595247387396553,
            -0.0012284613640242475,
            0.00016974366686180648,
            -2.273832933698107e-05,
            2.959844378859951e-06,
            -3.7513575580165137e-07,
            4.637282876523368e-08,
            -5.5994555257942045e-09,
            6.613158076707037e-10,
            -7.648263812399926e-11,
            8.670983684153396e-12,
            -9.645679317000283e-13,
            1.0516125007834867e-13,
            -1.1379786002407855e-14,
            1.261336714088025e-15,
            -3.2805548401250286e-16,
        ],
    ),
    (
        4.0,
        8.0,
        &[
            0.09799557556862197,
            -0.03261716242283878,
            0.0053544445130489235,
            -0.0008675085617618249,
            0.00013878345773327721,
            -2.1933248302171516e-05,
            3.4257168244710085e-06,
            -5.289955530139357e-07,
            8.079076530583057e-08,
            -1.220751169741051e-08,
            1.825515117725411e-09,
            -2.7024964430042895e-10,
            3.961744382844532e-11,
            -5.752667348327864e-12,
            8.275514776369761e-13,
            -1.1807075784911615e-13,
            1.676728931137835e-14,
            -2.325625072635854e-15,
            3.1389364791622435e-16,
        ],
    ),
];

/// erf(y) for |y| < 1, from its Taylor series.
fn erf_taylor(y: f64) -> f64 {
    let y2 = y * y;
    let mut s = 0.0;
    for c in ERF_TAYLOR.iter().rev() {
        s = s * y2 + c;
    }
    y * s
}

/// The scaled complementary error function e^(y²) erfc(y), for y ≥ 1 (and NaN): a
/// Chebyshev piece up to 8, and beyond that twelve levels of the continued fraction
/// 1 / (y + (1/2) / (y + 1 / (y + (3/2) / ...))) over √π, which there is exact to
/// rounding.
fn erfcx(y: f64) -> f64 {
    match ERFCX_PIECES.iter().find(|p| y <= p.1) {
        Some(&(a, b, c)) => {
            // Clenshaw's recurrence for Σ c_k T_k(t), t the position in [a, b] as [−1, 1]
            let t = (2.0 * y - a - b) / (b - a);
            let (mut b1, mut b2) = (0.0, 0.0);
            for &ck in c[1..].iter().rev() {
                (b1, b2) = (2.0 * t * b1 - b2 + ck, b1);
            }
            t * b1 - b2 + c[0]
        }
        None => {
            let mut f = y;
            for k in (1..=12).rev() {
                f = y + 0.5 * k as f64 / f;
            }
            0.5 * FRAC_2_SQRT_PI / f
        }
    }
}

// ----- the inner loops ---------------------------------------------------------------

/// y += Σ over t < `count` of c_t x_t, where c_t = `coef[c0 + t * cstride]` and x_t is
/// the row of `y.len()` numbers starting at `x[t * xstride]`: every matrix product here,
/// forward and back, is a loop of these. The sum for a stretch of y is kept in
/// registers while it runs over t, and added to y once: a loop of `y += c x` instead
/// would load and store y at every term, which on these sizes takes most of the time.
fn accumulate(y: &mut [f64], coef: &[f64], c0: usize, cstride: usize, x: &[f64], xstride: usize, count: usize) {
    let n = y.len();
    let mut at = 0;
    while at + 16 <= n {
        stretch::<16>(y, coef, c0, cstride, x, xstride, count, at);
        at += 16;
    }
    if at + 8 <= n {
        stretch::<8>(y, coef, c0, cstride, x, xstride, count, at);
        at += 8;
    }
    if at + 4 <= n {
        stretch::<4>(y, coef, c0, cstride, x, xstride, count, at);
        at += 4;
    }
    while at < n {
        stretch::<1>(y, coef, c0, cstride, x, xstride, count, at);
        at += 1;
    }
}

/// [`accumulate`] for the W numbers of y from `at`.
#[inline(always)]
#[allow(clippy::too_many_arguments)]
fn stretch<const W: usize>(y: &mut [f64], coef: &[f64], c0: usize, cstride: usize, x: &[f64], xstride: usize, count: usize, at: usize) {
    let mut acc = [0.0; W];
    for t in 0..count {
        let c = coef[c0 + t * cstride];
        let row = &x[t * xstride + at..t * xstride + at + W];
        for q in 0..W {
            acc[q] += c * row[q];
        }
    }
    for q in 0..W {
        y[at + q] += acc[q];
    }
}

/// The transpose of the `rows` by `cols` matrix `a` into `out` (resized to fit).
fn transpose_into(a: &[f64], rows: usize, cols: usize, out: &mut Vec<f64>) {
    out.clear();
    out.resize(rows * cols, 0.0);
    for i in 0..rows {
        for j in 0..cols {
            out[j * rows + i] = a[i * cols + j];
        }
    }
}

/// y += a x, over the shorter of the two.
#[inline]
fn axpy(y: &mut [f64], a: f64, x: &[f64]) {
    for (y, &x) in y.iter_mut().zip(x) {
        *y += a * x;
    }
}

/// The dot product over the shorter of the two, with four running totals (one would
/// have to be added in order, which keeps the compiler from using SIMD).
#[inline]
fn dot(a: &[f64], b: &[f64]) -> f64 {
    let n = a.len().min(b.len());
    let (a, b) = (&a[..n], &b[..n]);
    let (ca, cb) = (a.chunks_exact(4), b.chunks_exact(4));
    let (ra, rb) = (ca.remainder(), cb.remainder());
    let mut acc = [0.0; 4];
    for (x, y) in ca.zip(cb) {
        for i in 0..4 {
            acc[i] += x[i] * y[i];
        }
    }
    let mut s = (acc[0] + acc[1]) + (acc[2] + acc[3]);
    for (x, y) in ra.iter().zip(rb) {
        s += x * y;
    }
    s
}

/// The mean of a slice, 0 for an empty one.
fn mean_of(v: &[f64]) -> f64 {
    if v.is_empty() {
        0.0
    } else {
        v.iter().sum::<f64>() / v.len() as f64
    }
}

// ----- the tape ----------------------------------------------------------------------

/// What made a node, and from what.
#[derive(Clone, Copy, Debug)]
enum Op {
    Constant,
    Param,
    MatMul(Var, Var),
    MatMulT(Var, Var),
    Add(Var, Var),
    Sub(Var, Var),
    Mul(Var, Var),
    AddRow(Var, Var),
    MulRow(Var, Var),
    Scale(Var, f64),
    Transpose(Var),
    /// (keeps the slope at every entry)
    Act(Var),
    /// (keeps 1 / the standard deviation of every row)
    LayerNorm(Var),
    /// (keeps 1 / the root mean square of every row)
    RmsNorm(Var),
    /// (keeps 1 / the standard deviation of every column)
    BatchNorm(Var),
    Softmax(Var),
    SoftmaxCausal(Var),
    /// (keeps the row numbers)
    Gather(Var),
    /// (keeps the targets and the probabilities; the number of rows that had a target)
    CrossEntropy(Var, usize),
    Mse(Var, Var),
    Sum(Var),
    Mean(Var),
    /// A shape that did not fit: NaN, and nothing passed back.
    Invalid,
}

#[derive(Clone, Copy, Debug)]
struct Node {
    op: Op,
    rows: usize,
    cols: usize,
    /// Where its value (and its gradient) starts in the tape's buffers.
    at: usize,
    /// Where its kept numbers start, and its kept indices.
    aux: usize,
    idx: usize,
}

impl Node {
    fn len(&self) -> usize {
        self.rows * self.cols
    }

    fn span(&self) -> std::ops::Range<usize> {
        self.at..self.at + self.len()
    }
}

/// A record of a computation on matrices, which [`Tape::backward`] differentiates. See
/// the module's documentation for the conventions (row-major, one row per example,
/// weights one row per output unit).
///
/// Every operation records a node, works out its value at once, and returns its
/// [`Var`]; [`Tape::value`] and [`Tape::grad`] read any node back.
#[derive(Clone, Debug, Default)]
pub struct Tape {
    nodes: Vec<Node>,
    vals: Vec<f64>,
    grads: Vec<f64>,
    aux: Vec<f64>,
    idx: Vec<usize>,
    live: Vec<bool>,
    /// Room for a transposed matrix in a product.
    scratch: Vec<f64>,
}

impl Tape {
    /// An empty tape.
    pub fn new() -> Self {
        Tape::default()
    }

    /// Forget every node, keeping the memory for the next pass. Every `Var` handed out
    /// so far is then meaningless.
    pub fn clear(&mut self) {
        self.nodes.clear();
        self.vals.clear();
        self.grads.clear();
        self.aux.clear();
        self.idx.clear();
        self.live.clear();
    }

    /// How many nodes are recorded.
    pub fn len(&self) -> usize {
        self.nodes.len()
    }

    /// Is nothing recorded?
    pub fn is_empty(&self) -> bool {
        self.nodes.is_empty()
    }

    fn node(&self, v: Var) -> Option<Node> {
        self.nodes.get(v.0).copied()
    }

    /// The number of rows of `v` (0 for a handle this tape does not have).
    pub fn rows(&self, v: Var) -> usize {
        self.node(v).map_or(0, |n| n.rows)
    }

    /// The number of columns of `v`.
    pub fn cols(&self, v: Var) -> usize {
        self.node(v).map_or(0, |n| n.cols)
    }

    /// Rows and columns together.
    pub fn shape(&self, v: Var) -> (usize, usize) {
        (self.rows(v), self.cols(v))
    }

    /// The value of `v`, row-major (empty for a handle this tape does not have).
    pub fn value(&self, v: Var) -> &[f64] {
        self.node(v).map_or(&[], |n| &self.vals[n.span()])
    }

    /// The derivative of whatever [`Tape::backward`] (or [`Tape::backward_seeded`]) was
    /// last called on with respect to `v`, entry by entry, in the shape of `v`: zeros if
    /// that does not depend on `v`, or if nothing has been differentiated yet. For a
    /// node in the middle of the computation it is the derivative with everything that
    /// made the node held fixed: how the result would change if this value alone were
    /// nudged.
    pub fn grad(&self, v: Var) -> &[f64] {
        self.node(v).map_or(&[], |n| &self.grads[n.span()])
    }

    /// Is `v` a parameter (made by [`Tape::param`])?
    pub fn is_param(&self, v: Var) -> bool {
        self.node(v).is_some_and(|n| matches!(n.op, Op::Param))
    }

    /// Every parameter on the tape, in the order they were recorded.
    pub fn params(&self) -> impl Iterator<Item = Var> + '_ {
        self.nodes.iter().enumerate().filter(|(_, n)| matches!(n.op, Op::Param)).map(|(i, _)| Var(i))
    }

    // ----- recording ---------------------------------------------------------------

    /// Record a node of the given shape, with `aux` kept numbers, its value zero.
    fn push(&mut self, op: Op, rows: usize, cols: usize, aux: usize) -> Var {
        let n = rows * cols;
        let at = self.vals.len();
        self.vals.resize(at + n, 0.0);
        self.grads.resize(at + n, 0.0);
        let aux_at = self.aux.len();
        self.aux.resize(aux_at + aux, 0.0);
        self.nodes.push(Node { op, rows, cols, at, aux: aux_at, idx: self.idx.len() });
        Var(self.nodes.len() - 1)
    }

    /// The node for a shape that did not fit: what the operation would have made,
    /// filled with NaN.
    fn invalid(&mut self, rows: usize, cols: usize) -> Var {
        let v = self.push(Op::Invalid, rows, cols, 0);
        let span = self.nodes[v.0].span();
        self.vals[span].fill(f64::NAN);
        v
    }

    /// The values recorded before node `v` (its inputs' among them), its own value to
    /// fill in, and its kept numbers.
    fn parts(&mut self, v: Var) -> (&[f64], &mut [f64], &mut [f64]) {
        let node = self.nodes[v.0];
        let (before, rest) = self.vals.split_at_mut(node.at);
        (before, &mut rest[..node.len()], &mut self.aux[node.aux..])
    }

    /// The inputs of an operation, if they are all this tape's.
    fn inputs<const N: usize>(&self, vs: [Var; N]) -> Option<[Node; N]> {
        let mut out = [Node { op: Op::Invalid, rows: 0, cols: 0, at: 0, aux: 0, idx: 0 }; N];
        for (o, v) in out.iter_mut().zip(vs) {
            *o = self.node(v)?;
        }
        Some(out)
    }

    fn leaf(&mut self, op: Op, rows: usize, cols: usize, data: &[f64]) -> Var {
        let fits = rows.checked_mul(cols) == Some(data.len());
        debug_assert!(fits, "a {rows} by {cols} matrix needs {} numbers, not {}", rows.saturating_mul(cols), data.len());
        let v = self.push(op, rows, cols, 0);
        let (_, out, _) = self.parts(v);
        if fits {
            out.copy_from_slice(data);
        } else {
            out.fill(f64::NAN);
        }
        v
    }

    /// A matrix that is data, not something to learn: `rows` by `cols` numbers,
    /// row-major. (Its gradient is still worked out: the gradient with respect to a
    /// network's input is a constant's.) `data` must hold exactly `rows * cols` numbers.
    pub fn constant(&mut self, rows: usize, cols: usize, data: &[f64]) -> Var {
        self.leaf(Op::Constant, rows, cols, data)
    }

    /// The same, marked as a parameter: a weight, a bias, a gain, an embedding. The
    /// mark changes nothing in the arithmetic; it lets [`Tape::params`] and
    /// [`Tape::is_param`] tell the learnt numbers from the data.
    pub fn param(&mut self, rows: usize, cols: usize, data: &[f64]) -> Var {
        self.leaf(Op::Param, rows, cols, data)
    }

    /// A 1 by 1 constant.
    pub fn scalar(&mut self, x: f64) -> Var {
        self.leaf(Op::Constant, 1, 1, &[x])
    }

    /// The matrix product `a b` of an m by k and a k by n matrix: m by n, entry (i, j)
    /// the sum over p of a(i, p) b(p, j). On the way back, a receives g bᵀ and b
    /// receives aᵀ g (g the gradient of the product).
    pub fn matmul(&mut self, a: Var, b: Var) -> Var {
        let Some([na, nb]) = self.inputs([a, b]) else { return self.unknown(0, 0) };
        let (m, k, n) = (na.rows, na.cols, nb.cols);
        if !fits(k == nb.rows, "matmul: the columns of a must match the rows of b") {
            return self.invalid(m, n);
        }
        let v = self.push(Op::MatMul(a, b), m, n, 0);
        let (vals, out, _) = self.parts(v);
        let (av, bv) = (&vals[na.span()], &vals[nb.span()]);
        // row i of the product is the sum over p of a(i, p) times row p of b
        for i in 0..m {
            accumulate(&mut out[i * n..(i + 1) * n], av, i * k, 1, bv, n, k);
        }
        v
    }

    /// The product `a bᵀ` of an m by k and an n by k matrix, without forming the
    /// transpose: m by n, entry (i, j) the dot product of row i of a with row j of b.
    /// This is how a layer whose weights W are stored one row per output unit is
    /// applied to a batch X stored one row per example (`X Wᵀ`), and how attention
    /// scores are made from queries and keys (`Q Kᵀ`). On the way back, a receives g b
    /// and b receives gᵀ a.
    pub fn matmul_t(&mut self, a: Var, b: Var) -> Var {
        let Some([na, nb]) = self.inputs([a, b]) else { return self.unknown(0, 0) };
        let (m, k, n) = (na.rows, na.cols, nb.rows);
        if !fits(k == nb.cols, "matmul_t: a and b must have the same number of columns") {
            return self.invalid(m, n);
        }
        let v = self.push(Op::MatMulT(a, b), m, n, 0);
        let node = self.nodes[v.0];
        let (vals, rest) = self.vals.split_at_mut(node.at);
        let (av, out) = (&vals[na.span()], &mut rest[..node.len()]);
        // with bᵀ at hand, row i is the sum over p of a(i, p) times row p of bᵀ
        transpose_into(&vals[nb.span()], n, k, &mut self.scratch);
        for i in 0..m {
            accumulate(&mut out[i * n..(i + 1) * n], av, i * k, 1, &self.scratch, n, k);
        }
        v
    }

    /// An operation on two matrices of the same shape, entry by entry.
    fn entrywise(&mut self, a: Var, b: Var, op: Op, f: fn(f64, f64) -> f64) -> Var {
        let Some([na, nb]) = self.inputs([a, b]) else { return self.unknown(0, 0) };
        if !fits(na.rows == nb.rows && na.cols == nb.cols, "add, sub and mul need two matrices of the same shape") {
            return self.invalid(na.rows, na.cols);
        }
        let v = self.push(op, na.rows, na.cols, 0);
        let (vals, out, _) = self.parts(v);
        for ((o, &x), &y) in out.iter_mut().zip(&vals[na.span()]).zip(&vals[nb.span()]) {
            *o = f(x, y);
        }
        v
    }

    /// a + b, entry by entry (the shapes equal). Both receive the gradient unchanged:
    /// this is how a skip connection hands the gradient straight back.
    pub fn add(&mut self, a: Var, b: Var) -> Var {
        self.entrywise(a, b, Op::Add(a, b), |x, y| x + y)
    }

    /// a − b, entry by entry. a receives the gradient, b its negative.
    pub fn sub(&mut self, a: Var, b: Var) -> Var {
        self.entrywise(a, b, Op::Sub(a, b), |x, y| x - y)
    }

    /// a times b, entry by entry (not the matrix product). a receives g times b, and b
    /// receives g times a.
    pub fn mul(&mut self, a: Var, b: Var) -> Var {
        self.entrywise(a, b, Op::Mul(a, b), |x, y| x * y)
    }

    /// An operation of a matrix with a row vector (1 by n) applied to every row.
    fn with_row(&mut self, a: Var, r: Var, op: Op, f: fn(f64, f64) -> f64) -> Var {
        let Some([na, nr]) = self.inputs([a, r]) else { return self.unknown(0, 0) };
        if !fits(nr.rows == 1 && nr.cols == na.cols, "add_row and mul_row need a 1 by n row for an m by n matrix") {
            return self.invalid(na.rows, na.cols);
        }
        let v = self.push(op, na.rows, na.cols, 0);
        let (vals, out, _) = self.parts(v);
        let row = &vals[nr.span()];
        let n = na.cols;
        for (o, x) in out.chunks_exact_mut(n.max(1)).zip(vals[na.span()].chunks_exact(n.max(1))) {
            for ((o, &x), &r) in o.iter_mut().zip(x).zip(row) {
                *o = f(x, r);
            }
        }
        v
    }

    /// The row vector `r` (1 by n) added to every row of the m by n matrix `a`: a bias,
    /// the same for every example. On the way back `a` receives the gradient unchanged
    /// and `r` the sum of its rows (the gradient of a bias is the sum over the batch).
    pub fn add_row(&mut self, a: Var, r: Var) -> Var {
        self.with_row(a, r, Op::AddRow(a, r), |x, r| x + r)
    }

    /// Every row of `a` multiplied entry by entry by the row vector `r`: a gain, one
    /// factor per column. On the way back `a` receives g times r, and entry j of `r`
    /// the sum over the rows of g(i, j) a(i, j).
    pub fn mul_row(&mut self, a: Var, r: Var) -> Var {
        self.with_row(a, r, Op::MulRow(a, r), |x, r| x * r)
    }

    /// s times every entry of `a`; `a` receives s times the gradient.
    pub fn scale(&mut self, a: Var, s: f64) -> Var {
        let Some([na]) = self.inputs([a]) else { return self.unknown(0, 0) };
        let v = self.push(Op::Scale(a, s), na.rows, na.cols, 0);
        let (vals, out, _) = self.parts(v);
        for (o, &x) in out.iter_mut().zip(&vals[na.span()]) {
            *o = s * x;
        }
        v
    }

    /// The transpose: an m by n matrix becomes n by m, entry (j, i) being a(i, j). The
    /// gradient goes back transposed.
    pub fn transpose(&mut self, a: Var) -> Var {
        let Some([na]) = self.inputs([a]) else { return self.unknown(0, 0) };
        let (m, n) = (na.rows, na.cols);
        let v = self.push(Op::Transpose(a), n, m, 0);
        let (vals, out, _) = self.parts(v);
        let av = &vals[na.span()];
        for i in 0..m {
            for j in 0..n {
                out[j * m + i] = av[i * n + j];
            }
        }
        v
    }

    /// The activation `f` applied to every entry. The slope at every entry is worked
    /// out on the way and kept: on the way back each entry's gradient is multiplied by
    /// it (the gate).
    pub fn act(&mut self, a: Var, f: Act) -> Var {
        let Some([na]) = self.inputs([a]) else { return self.unknown(0, 0) };
        let v = self.push(Op::Act(a), na.rows, na.cols, na.len());
        let (vals, out, slopes) = self.parts(v);
        for ((o, s), &x) in out.iter_mut().zip(slopes.iter_mut()).zip(&vals[na.span()]) {
            (*o, *s) = f.f_df(x);
        }
        v
    }

    /// Layer normalisation without gain or bias (those are [`Tape::mul_row`] and
    /// [`Tape::add_row`] after it): each row x of n entries has its mean m taken off
    /// and is divided by its standard deviation, y = (x − m) / s with s = √(v + eps)
    /// and v the mean of (x − m)² over the row. Every row comes out with mean 0 and
    /// mean square v / (v + eps), which is 1 but for the guard `eps` (1e-5 is usual;
    /// it keeps a constant row from being divided by zero).
    ///
    /// On the way back, the gradient g arriving at a row becomes (g − mean(g) − y
    /// mean(g y)) / s: its part along the all-ones direction and its part along y are
    /// taken out, and the rest divided by the row's spread. So what comes back is at
    /// right angles to the all-ones vector and to the input itself (a change of the
    /// input's level or size changes nothing, but for `eps`), and is half as large for
    /// an input twice as spread out.
    pub fn layer_norm(&mut self, a: Var, eps: f64) -> Var {
        self.standardise(a, eps, Op::LayerNorm(a), true)
    }

    /// RMS normalisation without gain: each row x of n entries is divided by its root
    /// mean square, y = x / r with r = √(mean of x² + eps); nothing is taken off. On
    /// the way back the gradient g becomes (g − y mean(g y)) / r: only its part along
    /// the input itself is taken out (not the all-ones part), and the rest divided by
    /// the input's size. So the output does not change when the input is scaled (but
    /// for `eps`), and the gradient that comes back is 1/s times as large when the
    /// input is s times as large.
    pub fn rms_norm(&mut self, a: Var, eps: f64) -> Var {
        self.standardise(a, eps, Op::RmsNorm(a), false)
    }

    fn standardise(&mut self, a: Var, eps: f64, op: Op, centre: bool) -> Var {
        let Some([na]) = self.inputs([a]) else { return self.unknown(0, 0) };
        let (m, n) = (na.rows, na.cols);
        let v = self.push(op, m, n, m);
        let (vals, out, inv) = self.parts(v);
        if n == 0 {
            return v;
        }
        for ((o, x), inv) in out.chunks_exact_mut(n).zip(vals[na.span()].chunks_exact(n)).zip(inv.iter_mut()) {
            let mean = if centre { mean_of(x) } else { 0.0 };
            let spread = x.iter().map(|&x| (x - mean) * (x - mean)).sum::<f64>() / n as f64;
            *inv = 1.0 / (spread + eps).sqrt();
            for (o, &x) in o.iter_mut().zip(x) {
                *o = (x - mean) * *inv;
            }
        }
        v
    }

    /// Batch normalisation without gain or bias: each COLUMN (one unit, over the
    /// examples of the batch: the rows) has its mean taken off and is divided by its
    /// standard deviation, y = (x − m) / √(v + eps), with the statistics of this batch.
    /// So one example's output, and its gradient, depend on every other example in the
    /// batch. (After training a network is usually run with fixed statistics instead:
    /// that is `sub` and `mul_row` with constants, not this.) On the way back, column
    /// by column, the gradient becomes (g − mean(g) − y mean(g y)) / √(v + eps), the
    /// means over the rows.
    pub fn batch_norm(&mut self, a: Var, eps: f64) -> Var {
        let Some([na]) = self.inputs([a]) else { return self.unknown(0, 0) };
        let (m, n) = (na.rows, na.cols);
        let v = self.push(Op::BatchNorm(a), m, n, n);
        let (vals, out, inv) = self.parts(v);
        let x = &vals[na.span()];
        for j in 0..n {
            let column = (0..m).map(|i| x[i * n + j]);
            let mean = if m == 0 { 0.0 } else { column.clone().sum::<f64>() / m as f64 };
            let spread = if m == 0 { 0.0 } else { column.map(|x| (x - mean) * (x - mean)).sum::<f64>() / m as f64 };
            inv[j] = 1.0 / (spread + eps).sqrt();
            for i in 0..m {
                out[i * n + j] = (x[i * n + j] - mean) * inv[j];
            }
        }
        v
    }

    /// The softmax of every row: entry j becomes e^(x_j) / (the sum over the row of
    /// e^(x_k)), so each row is a set of positive weights adding up to 1. (Worked out
    /// with the row's largest entry taken off first, which changes nothing but keeps
    /// the exponentials from overflowing.) On the way back a row's gradient g becomes
    /// p (g − the sum over the row of g p): the Jacobian of a softmax is diag(p) − p pᵀ,
    /// which is near 0 when one entry has almost all the weight.
    pub fn softmax_rows(&mut self, a: Var) -> Var {
        let Some([na]) = self.inputs([a]) else { return self.unknown(0, 0) };
        let v = self.push(Op::Softmax(a), na.rows, na.cols, 0);
        let n = na.cols;
        let (vals, out, _) = self.parts(v);
        if n > 0 {
            for (o, x) in out.chunks_exact_mut(n).zip(vals[na.span()].chunks_exact(n)) {
                softmax(x, o);
            }
        }
        v
    }

    /// The softmax with a causal mask, for a square matrix of attention scores (rows
    /// the positions that look, columns the positions looked at): row i is a softmax
    /// over columns 0 to i alone, and columns after i are 0. Attention that cannot look
    /// ahead; and no gradient goes back to the scores of the future, or through them to
    /// the positions they belong to.
    pub fn softmax_rows_causal(&mut self, a: Var) -> Var {
        let Some([na]) = self.inputs([a]) else { return self.unknown(0, 0) };
        let n = na.cols;
        if !fits(na.rows == n, "softmax_rows_causal needs a square matrix") {
            return self.invalid(na.rows, n);
        }
        let v = self.push(Op::SoftmaxCausal(a), n, n, 0);
        let (vals, out, _) = self.parts(v);
        let x = &vals[na.span()];
        for i in 0..n {
            softmax(&x[i * n..i * n + i + 1], &mut out[i * n..i * n + i + 1]);
        }
        v
    }

    /// An embedding lookup: row r of the result is row `rows[r]` of `table`, so the
    /// result has `rows.len()` rows and as many columns as the table. A row number the
    /// table does not have gives a row of NaN. On the way back each table row receives
    /// the sum of the gradients of the result's rows that copied it (a token that
    /// appears twice gets both).
    pub fn gather_rows(&mut self, table: Var, rows: &[usize]) -> Var {
        let Some([nt]) = self.inputs([table]) else { return self.unknown(rows.len(), 0) };
        let n = nt.cols;
        let v = self.push(Op::Gather(table), rows.len(), n, 0);
        self.idx.extend_from_slice(rows);
        let (vals, out, _) = self.parts(v);
        let t = &vals[nt.span()];
        for (o, &r) in out.chunks_exact_mut(n.max(1)).zip(rows) {
            if r < nt.rows {
                o.copy_from_slice(&t[r * n..(r + 1) * n]);
            } else {
                o.fill(f64::NAN);
            }
        }
        v
    }

    /// Softmax with cross-entropy, a 1 by 1 loss: the mean, over the rows that have a
    /// target, of minus the log of the softmax probability of that row's target class.
    /// `targets[r]` is the class of row r (a column of `logits`); a row whose target is
    /// [`NO_TARGET`], or any other number that is not a column, is left out of the mean
    /// (and gets no gradient). With no row scored the loss is 0.
    ///
    /// On the way back row r of the logits receives (p_r − e_t) / m: its softmax
    /// probabilities minus the one-hot vector of its target, over the number m of
    /// scored rows. The probabilities are kept from the forward pass.
    pub fn cross_entropy(&mut self, logits: Var, targets: &[usize]) -> Var {
        self.cross_entropy_with(logits, targets.len(), targets.iter().copied())
    }

    /// [`Tape::cross_entropy`] with the targets given one by one (`count` of them), so
    /// that callers holding them in another form need not collect them first.
    pub(crate) fn cross_entropy_with(&mut self, logits: Var, count: usize, targets: impl Iterator<Item = usize>) -> Var {
        let Some([nl]) = self.inputs([logits]) else { return self.unknown(1, 1) };
        let (m, n) = (nl.rows, nl.cols);
        if !fits(count == m, "cross_entropy needs one target per row") {
            return self.invalid(1, 1);
        }
        let v = self.push(Op::CrossEntropy(logits, 0), 1, 1, m * n);
        let first = self.idx.len();
        self.idx.extend(targets.take(m));
        let node = self.nodes[v.0];
        let (before, rest) = self.vals.split_at_mut(node.at);
        let (x, probs, targets) = (&before[nl.span()], &mut self.aux[node.aux..node.aux + m * n], &self.idx[first..]);
        let (mut total, mut scored) = (0.0, 0);
        if n > 0 {
            for ((x, p), &t) in x.chunks_exact(n).zip(probs.chunks_exact_mut(n)).zip(targets) {
                let top = x.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
                let sum: f64 = x.iter().map(|&x| (x - top).exp()).sum();
                for (p, &x) in p.iter_mut().zip(x) {
                    *p = (x - top).exp() / sum;
                }
                if t < n {
                    total += sum.ln() + top - x[t];
                    scored += 1;
                }
            }
        }
        rest[0] = if scored == 0 { 0.0 } else { total / scored as f64 };
        self.nodes[v.0].op = Op::CrossEntropy(logits, scored);
        v
    }

    /// The mean squared error, a 1 by 1 loss: the mean over every entry of (a − b)²
    /// (the shapes equal; 0 for empty matrices). a receives 2 (a − b) / N times the
    /// gradient, N the number of entries, and b the negative of that.
    pub fn mse(&mut self, a: Var, b: Var) -> Var {
        let Some([na, nb]) = self.inputs([a, b]) else { return self.unknown(1, 1) };
        if !fits(na.rows == nb.rows && na.cols == nb.cols, "mse needs two matrices of the same shape") {
            return self.invalid(1, 1);
        }
        let v = self.push(Op::Mse(a, b), 1, 1, 0);
        let (vals, out, _) = self.parts(v);
        let total: f64 = vals[na.span()].iter().zip(&vals[nb.span()]).map(|(x, y)| (x - y) * (x - y)).sum();
        out[0] = if na.len() == 0 { 0.0 } else { total / na.len() as f64 };
        v
    }

    /// The sum of every entry, 1 by 1; each entry receives the gradient unchanged.
    pub fn sum(&mut self, a: Var) -> Var {
        let Some([na]) = self.inputs([a]) else { return self.unknown(1, 1) };
        let v = self.push(Op::Sum(a), 1, 1, 0);
        let (vals, out, _) = self.parts(v);
        out[0] = vals[na.span()].iter().sum();
        v
    }

    /// The mean of every entry, 1 by 1 (0 for an empty matrix); each entry receives
    /// the gradient over the number of entries.
    pub fn mean(&mut self, a: Var) -> Var {
        let Some([na]) = self.inputs([a]) else { return self.unknown(1, 1) };
        let v = self.push(Op::Mean(a), 1, 1, 0);
        let (vals, out, _) = self.parts(v);
        out[0] = mean_of(&vals[na.span()]);
        v
    }

    /// A handle this tape does not have was passed in: a mistake like a shape that does
    /// not fit.
    fn unknown(&mut self, rows: usize, cols: usize) -> Var {
        fits(false, "a Var from another tape, or from before the tape was cleared");
        self.invalid(rows, cols)
    }

    // ----- differentiating ---------------------------------------------------------

    /// Differentiate the 1 by 1 node `v` (a loss): its gradient is started at 1 and
    /// swept back through everything it was made from, after the gradients of an
    /// earlier sweep are cleared. Afterwards [`Tape::grad`] gives the derivative of `v`
    /// with respect to any node. (A `v` that is not 1 by 1 is a mistake: in a release
    /// build every gradient that reaches anything is then NaN.)
    pub fn backward(&mut self, v: Var) {
        let fits = fits(self.node(v).is_some_and(|n| n.rows == 1 && n.cols == 1), "backward needs a 1 by 1 node: use backward_seeded");
        self.sweep(v, |g| g.fill(if fits { 1.0 } else { f64::NAN }));
    }

    /// Differentiate a node of any shape along a direction: its gradient is started at
    /// `seed` (as many numbers as `v` has entries) and swept back. Afterwards the
    /// gradient of a node u is seedᵀ times the Jacobian of v with respect to u, the
    /// vector-Jacobian product: what backpropagation computes, without ever forming the
    /// Jacobian. Earlier gradients are cleared first.
    pub fn backward_seeded(&mut self, v: Var, seed: &[f64]) {
        let fits = fits(self.node(v).is_some_and(|n| n.len() == seed.len()), "backward_seeded needs one seed number per entry");
        self.sweep(v, |g| if fits { g.copy_from_slice(seed) } else { g.fill(f64::NAN) });
    }

    /// The Jacobian of `output` with respect to `input`: one row for each entry of
    /// `output` and one column for each entry of `input` (both counted row-major), so
    /// entry (k, l) is the derivative of output entry k with respect to input entry l,
    /// everything else that `output` depends on held fixed. One backward sweep per
    /// entry of `output`: for small sizes. Afterwards the tape's gradients are those of
    /// the last sweep (run `backward` again for a loss's). A Jacobian-vector product is
    /// this matrix times the vector.
    pub fn jacobian(&mut self, output: Var, input: Var) -> Vec<f64> {
        let (rows, cols) = (self.value(output).len(), self.value(input).len());
        let mut jac = vec![0.0; rows * cols];
        for k in 0..rows {
            self.sweep(output, |g| g[k] = 1.0);
            jac[k * cols..(k + 1) * cols].copy_from_slice(self.grad(input));
        }
        jac
    }

    /// Clear every gradient, seed `v`'s, and sweep back over everything `v` was made
    /// from (and only that: a node is visited when something downstream of it, on the
    /// way to `v`, has passed it a gradient).
    fn sweep(&mut self, v: Var, seed: impl FnOnce(&mut [f64])) {
        self.grads.fill(0.0);
        let Some(node) = self.node(v) else { return };
        seed(&mut self.grads[node.span()]);
        self.live.clear();
        self.live.resize(self.nodes.len(), false);
        self.live[v.0] = true;
        for i in (0..=v.0).rev() {
            if self.live[i] {
                self.back(i);
            }
        }
    }

    /// Pass node `i`'s gradient on to its inputs.
    fn back(&mut self, i: usize) {
        let Tape { nodes, vals, grads, aux, idx, live, scratch } = self;
        let node = nodes[i];
        let (before, rest) = grads.split_at_mut(node.at);
        let g = &rest[..node.len()];
        let value = |v: Var| &vals[nodes[v.0].span()];
        let shape = |v: Var| (nodes[v.0].rows, nodes[v.0].cols);
        let kept = &aux[node.aux..];
        let own = &vals[node.span()];
        // the gradient buffer of an input, to add into
        macro_rules! into {
            ($v:expr) => {{
                live[$v.0] = true;
                &mut before[nodes[$v.0].span()]
            }};
        }
        match node.op {
            Op::Constant | Op::Param | Op::Invalid => {}
            Op::MatMul(a, b) => {
                // a (m by k) receives g bᵀ: row i, the sum over j of g(i, j) times row j
                // of bᵀ; b (k by n) receives aᵀ g: row p, the sum over i of a(i, p) times
                // row i of g
                let ((m, k), n) = (shape(a), node.cols);
                let (av, bv) = (value(a), value(b));
                transpose_into(bv, k, n, scratch);
                let ga = into!(a);
                for i in 0..m {
                    accumulate(&mut ga[i * k..(i + 1) * k], g, i * n, 1, scratch, k, n);
                }
                let gb = into!(b);
                for p in 0..k {
                    accumulate(&mut gb[p * n..(p + 1) * n], av, p, k, g, n, m);
                }
            }
            Op::MatMulT(a, b) => {
                // a (m by k) receives g b: row i, the sum over j of g(i, j) times row j of
                // b; b (n by k) receives gᵀ a: row j, the sum over i of g(i, j) times row i
                // of a
                let ((m, k), n) = (shape(a), node.cols);
                let (av, bv) = (value(a), value(b));
                let ga = into!(a);
                for i in 0..m {
                    accumulate(&mut ga[i * k..(i + 1) * k], g, i * n, 1, bv, k, n);
                }
                let gb = into!(b);
                for j in 0..n {
                    accumulate(&mut gb[j * k..(j + 1) * k], g, j, n, av, k, m);
                }
            }
            Op::Add(a, b) | Op::Sub(a, b) => {
                axpy(into!(a), 1.0, g);
                axpy(into!(b), if matches!(node.op, Op::Add(..)) { 1.0 } else { -1.0 }, g);
            }
            Op::Mul(a, b) => {
                let (av, bv) = (value(a), value(b));
                for ((d, &g), &y) in into!(a).iter_mut().zip(g).zip(bv) {
                    *d += g * y;
                }
                for ((d, &g), &x) in into!(b).iter_mut().zip(g).zip(av) {
                    *d += g * x;
                }
            }
            Op::AddRow(a, r) => {
                axpy(into!(a), 1.0, g);
                let gr = into!(r);
                for row in g.chunks_exact(node.cols.max(1)) {
                    axpy(gr, 1.0, row);
                }
            }
            Op::MulRow(a, r) => {
                let n = node.cols.max(1);
                let (av, rv) = (value(a), value(r));
                for (d, g) in into!(a).chunks_exact_mut(n).zip(g.chunks_exact(n)) {
                    for ((d, &g), &r) in d.iter_mut().zip(g).zip(rv) {
                        *d += g * r;
                    }
                }
                let gr = into!(r);
                for (g, x) in g.chunks_exact(n).zip(av.chunks_exact(n)) {
                    for ((d, &g), &x) in gr.iter_mut().zip(g).zip(x) {
                        *d += g * x;
                    }
                }
            }
            Op::Scale(a, s) => axpy(into!(a), s, g),
            Op::Transpose(a) => {
                let (m, n) = shape(a);
                let ga = into!(a);
                for i in 0..m {
                    for j in 0..n {
                        ga[i * n + j] += g[j * m + i];
                    }
                }
            }
            Op::Act(a) => {
                for ((d, &g), &s) in into!(a).iter_mut().zip(g).zip(kept) {
                    *d += g * s;
                }
            }
            Op::LayerNorm(a) | Op::RmsNorm(a) => {
                let n = node.cols;
                if n == 0 {
                    return;
                }
                let centre = matches!(node.op, Op::LayerNorm(_));
                let ga = into!(a);
                for (((d, g), y), &inv) in ga.chunks_exact_mut(n).zip(g.chunks_exact(n)).zip(own.chunks_exact(n)).zip(kept) {
                    let mg = if centre { mean_of(g) } else { 0.0 };
                    let mgy = dot(g, y) / n as f64;
                    for ((d, &g), &y) in d.iter_mut().zip(g).zip(y) {
                        *d += inv * (g - mg - y * mgy);
                    }
                }
            }
            Op::BatchNorm(a) => {
                let (m, n) = (node.rows, node.cols);
                let ga = into!(a);
                for j in 0..n {
                    let (mut mg, mut mgy) = (0.0, 0.0);
                    for i in 0..m {
                        mg += g[i * n + j];
                        mgy += g[i * n + j] * own[i * n + j];
                    }
                    (mg, mgy) = (mg / m as f64, mgy / m as f64);
                    for i in 0..m {
                        ga[i * n + j] += kept[j] * (g[i * n + j] - mg - own[i * n + j] * mgy);
                    }
                }
            }
            Op::Softmax(a) => {
                let n = node.cols.max(1);
                for ((d, g), p) in into!(a).chunks_exact_mut(n).zip(g.chunks_exact(n)).zip(own.chunks_exact(n)) {
                    softmax_back(g, p, d);
                }
            }
            Op::SoftmaxCausal(a) => {
                let n = node.cols;
                let ga = into!(a);
                for i in 0..n {
                    let row = i * n..i * n + i + 1;
                    softmax_back(&g[row.clone()], &own[row.clone()], &mut ga[row]);
                }
            }
            Op::Gather(t) => {
                let (rows, n) = (shape(t).0, node.cols);
                let gt = into!(t);
                for (r, &from) in idx[node.idx..node.idx + node.rows].iter().enumerate() {
                    if from < rows {
                        axpy(&mut gt[from * n..(from + 1) * n], 1.0, &g[r * n..(r + 1) * n]);
                    }
                }
            }
            Op::CrossEntropy(logits, scored) => {
                let (m, n) = shape(logits);
                if scored == 0 || n == 0 {
                    return;
                }
                let k = g[0] / scored as f64;
                let gl = into!(logits);
                for (r, &t) in idx[node.idx..node.idx + m].iter().enumerate() {
                    if t < n {
                        axpy(&mut gl[r * n..(r + 1) * n], k, &kept[r * n..(r + 1) * n]);
                        gl[r * n + t] -= k;
                    }
                }
            }
            Op::Mse(a, b) => {
                let count = value(a).len();
                if count == 0 {
                    return;
                }
                let k = 2.0 * g[0] / count as f64;
                let (av, bv) = (value(a), value(b));
                for ((d, &x), &y) in into!(a).iter_mut().zip(av).zip(bv) {
                    *d += k * (x - y);
                }
                for ((d, &x), &y) in into!(b).iter_mut().zip(av).zip(bv) {
                    *d -= k * (x - y);
                }
            }
            Op::Sum(a) => into!(a).iter_mut().for_each(|d| *d += g[0]),
            Op::Mean(a) => {
                let count = value(a).len().max(1) as f64;
                into!(a).iter_mut().for_each(|d| *d += g[0] / count);
            }
        }
    }
}

/// Is the shape right? (A `debug_assert` if not: the caller's mistake.)
fn fits(ok: bool, what: &str) -> bool {
    debug_assert!(ok, "{what}");
    ok
}

/// The softmax of `x` into `out`.
fn softmax(x: &[f64], out: &mut [f64]) {
    let top = x.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
    let mut sum = 0.0;
    for (o, &x) in out.iter_mut().zip(x) {
        *o = (x - top).exp();
        sum += *o;
    }
    for o in out.iter_mut() {
        *o /= sum;
    }
}

/// d += p (g − Σ g p): a gradient back through a softmax whose output was p.
fn softmax_back(g: &[f64], p: &[f64], d: &mut [f64]) {
    let s = dot(g, p);
    for ((d, &g), &p) in d.iter_mut().zip(g).zip(p) {
        *d += p * (g - s);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::math::rng::Rng;

    fn uniform(rng: &mut Rng, n: usize, lo: f64, hi: f64) -> Vec<f64> {
        (0..n).map(|_| lo + (hi - lo) * rng.f32() as f64).collect()
    }

    /// Inputs of a computation: shapes and values.
    type Inputs = Vec<(usize, usize, Vec<f64>)>;

    /// Random inputs of the given shapes, entries uniform in [−1.5, 1.5].
    fn random_inputs(rng: &mut Rng, shapes: &[(usize, usize)]) -> Inputs {
        shapes.iter().map(|&(r, c)| (r, c, uniform(rng, r * c, -1.5, 1.5))).collect()
    }

    /// Record `inputs` as parameters on a cleared tape and run `build` on them.
    fn run<F: Fn(&mut Tape, &[Var]) -> Var + ?Sized>(tape: &mut Tape, inputs: &Inputs, build: &F) -> (Vec<Var>, Var) {
        tape.clear();
        let vars: Vec<Var> = inputs.iter().map(|(r, c, v)| tape.param(*r, *c, v)).collect();
        let out = build(tape, &vars);
        (vars, out)
    }

    /// The heart of the tests: the gradient the tape gives for Σ R ⊙ build(inputs), R a
    /// fixed random matrix of the output's shape (so every entry of the output, and
    /// every road back, counts), against a central difference of the same sum, for
    /// every entry of every input. They must agree to a relative 1e-6 (with a floor of
    /// 1e-9 of the largest gradient, for entries whose gradient is zero or nearly so).
    fn check<F: Fn(&mut Tape, &[Var]) -> Var + ?Sized>(name: &str, inputs: &Inputs, build: &F, rng: &mut Rng) {
        let mut tape = Tape::new();
        let (vars, out) = run(&mut tape, inputs, build);
        let seed = uniform(rng, tape.value(out).len(), -1.0, 1.0);
        tape.backward_seeded(out, &seed);
        let grads: Vec<Vec<f64>> = vars.iter().map(|&v| tape.grad(v).to_vec()).collect();
        let largest = grads.iter().flatten().fold(0.0f64, |a, &b| a.max(b.abs()));
        let objective = |inputs: &Inputs| {
            let mut tape = Tape::new();
            let (_, out) = run(&mut tape, inputs, build);
            tape.value(out).iter().zip(&seed).map(|(a, b)| a * b).sum::<f64>()
        };
        for (i, input) in inputs.iter().enumerate() {
            for e in 0..input.2.len() {
                let x = input.2[e];
                let h = 1e-5 * (1.0 + x.abs());
                let mut moved = inputs.clone();
                moved[i].2[e] = x + h;
                let up = objective(&moved);
                moved[i].2[e] = x - h;
                let down = objective(&moved);
                let fd = (up - down) / (2.0 * h);
                let g = grads[i][e];
                assert!((fd - g).abs() <= 1e-6 * g.abs().max(fd.abs()) + 1e-9 * largest.max(1.0), "{name}: input {i} entry {e}: tape {g}, differences {fd}");
            }
        }
    }

    /// Shapes of one matrix to try: a single number, a single row, a single column,
    /// and two larger ones.
    const SHAPES: [(usize, usize); 5] = [(1, 1), (1, 5), (4, 1), (3, 4), (5, 3)];

    #[test]
    fn every_operation_has_the_gradient_of_its_finite_differences() {
        let mut rng = Rng::new(1);
        type Build = Box<dyn Fn(&mut Tape, &[Var]) -> Var>;
        fn op(f: impl Fn(&mut Tape, &[Var]) -> Var + 'static) -> Build {
            Box::new(f)
        }
        // operations of one matrix
        let unary: Vec<(&str, Build)> = vec![
            ("scale", op(|t, v| t.scale(v[0], -1.7))),
            ("transpose", op(|t, v| t.transpose(v[0]))),
            ("layer_norm", op(|t, v| t.layer_norm(v[0], 1e-5))),
            ("rms_norm", op(|t, v| t.rms_norm(v[0], 1e-5))),
            ("batch_norm", op(|t, v| t.batch_norm(v[0], 1e-5))),
            ("softmax_rows", op(|t, v| t.softmax_rows(v[0]))),
            ("sum", op(|t, v| t.sum(v[0]))),
            ("mean", op(|t, v| t.mean(v[0]))),
        ];
        for (name, build) in &unary {
            for &(r, c) in &SHAPES {
                check(&format!("{name} {r}x{c}"), &random_inputs(&mut rng, &[(r, c)]), build.as_ref(), &mut rng);
            }
        }
        // operations of two matrices of one shape
        let binary: Vec<(&str, Build)> = vec![
            ("add", op(|t, v| t.add(v[0], v[1]))),
            ("sub", op(|t, v| t.sub(v[0], v[1]))),
            ("mul", op(|t, v| t.mul(v[0], v[1]))),
            ("mse", op(|t, v| t.mse(v[0], v[1]))),
            ("add_row", op(|t, v| t.add_row(v[0], v[1]))),
            ("mul_row", op(|t, v| t.mul_row(v[0], v[1]))),
        ];
        for (name, build) in &binary {
            for &(r, c) in &SHAPES {
                let second = if name.ends_with("_row") { (1, c) } else { (r, c) };
                check(&format!("{name} {r}x{c}"), &random_inputs(&mut rng, &[(r, c), second]), build.as_ref(), &mut rng);
            }
        }
        // the same matrix twice: both roads back must add up
        check("mul of a with itself", &random_inputs(&mut rng, &[(3, 3)]), &|t, v| t.mul(v[0], v[0]), &mut rng);
        check("matmul of a with itself", &random_inputs(&mut rng, &[(3, 3)]), &|t, v| t.matmul(v[0], v[0]), &mut rng);
        // products
        for (m, k, n) in [(1, 1, 1), (1, 4, 1), (1, 3, 5), (3, 4, 2), (4, 1, 3), (2, 5, 6)] {
            check(&format!("matmul {m}x{k} {k}x{n}"), &random_inputs(&mut rng, &[(m, k), (k, n)]), &|t, v| t.matmul(v[0], v[1]), &mut rng);
            check(&format!("matmul_t {m}x{k} {n}x{k}"), &random_inputs(&mut rng, &[(m, k), (n, k)]), &|t, v| t.matmul_t(v[0], v[1]), &mut rng);
        }
        // the causal softmax is for square matrices
        for n in [1, 2, 4, 6] {
            check(&format!("softmax_rows_causal {n}x{n}"), &random_inputs(&mut rng, &[(n, n)]), &|t, v| t.softmax_rows_causal(v[0]), &mut rng);
        }
        // a lookup that takes some rows twice and one not at all
        for (rows, cols) in [(1, 1), (5, 3), (4, 1)] {
            let picks: Vec<usize> = (0..7).map(|_| rng.below(rows)).chain([0, rows - 1]).collect();
            check(&format!("gather_rows {rows}x{cols}"), &random_inputs(&mut rng, &[(rows, cols)]), &|t, v| t.gather_rows(v[0], &picks), &mut rng);
        }
        // cross-entropy, with every row scored and with one row left out
        for &(r, c) in &SHAPES {
            let mut targets: Vec<usize> = (0..r).map(|_| rng.below(c)).collect();
            check(&format!("cross_entropy {r}x{c}"), &random_inputs(&mut rng, &[(r, c)]), &|t, v| t.cross_entropy(v[0], &targets), &mut rng);
            if r > 1 {
                targets[1] = NO_TARGET;
                check(
                    &format!("cross_entropy {r}x{c}, a row unscored"),
                    &random_inputs(&mut rng, &[(r, c)]),
                    &|t, v| t.cross_entropy(v[0], &targets),
                    &mut rng,
                );
            }
        }
    }

    #[test]
    fn every_activation_has_the_gradient_of_its_finite_differences() {
        let mut rng = Rng::new(2);
        for act in Act::ALL {
            for &(r, c) in &SHAPES {
                // (inputs kept away from 0, where three of them have a kink, and spread
                // wide enough to reach the saturated parts)
                let values = (0..r * c).map(|_| (0.05 + 3.0 * rng.f32() as f64) * if rng.bool(0.5) { 1.0 } else { -1.0 }).collect();
                check(&format!("{} {r}x{c}", act.name()), &vec![(r, c, values)], &|t, v| t.act(v[0], act), &mut rng);
            }
        }
    }

    /// Each activation's slope is the derivative of its function (against central
    /// differences at many points), the value and slope worked out together are the
    /// two worked out alone, and a few values are known exactly.
    #[test]
    fn activations_are_what_they_say() {
        for act in Act::ALL {
            for i in -800..=800 {
                let x = i as f64 * 0.01 + 0.003;
                let h = 1e-5;
                let fd = (act.f(x + h) - act.f(x - h)) / (2.0 * h);
                assert!((fd - act.df(x)).abs() < 1e-8 * (1.0 + fd.abs()), "{} at {x}: slope {} against {fd}", act.name(), act.df(x));
            }
        }
        let close = |a: f64, b: f64| (a - b).abs() <= 1e-15 * (1.0 + b.abs());
        assert!(close(Act::Gelu.f(1.0), 0.841_344_746_068_542_9));
        assert!(close(Act::Gelu.f(-0.75), -1.699_705_142_826_511_4e-1));
        assert!(close(Act::Silu.f(1.0), 1.0 / (1.0 + (-1.0f64).exp())));
        assert!(close(Act::Softplus.f(0.0), std::f64::consts::LN_2));
        assert!(close(Act::Elu.f(-1.0), (-1.0f64).exp() - 1.0));
        assert!(close(Act::LeakyRelu.f(-2.0), -0.02));
        assert_eq!((Act::Sigmoid.f(0.0), Act::Sigmoid.df(0.0)), (0.5, 0.25));
        assert_eq!((Act::Relu.df(0.0), Act::LeakyRelu.df(0.0), Act::Relu.df(1e-300)), (0.0, 0.01, 1.0));
        // GELU's slope crosses zero near −0.7518 (its lowest point), SiLU's near −1.2785
        assert!(Act::Gelu.df(-0.752) < 0.0 && Act::Gelu.df(-0.751) > 0.0, "{} {}", Act::Gelu.df(-0.752), Act::Gelu.df(-0.751));
        assert!(Act::Silu.df(-1.279) < 0.0 && Act::Silu.df(-1.278) > 0.0);
        // the largest slopes the docs name
        assert!(Act::Sigmoid.df(0.0) == 0.25 && Act::Tanh.df(0.0) == 1.0);
        assert!(Act::Gelu.df(1.4) > 1.12 && Act::Silu.df(2.4) > 1.09);
        // the names differ
        for (i, a) in Act::ALL.iter().enumerate() {
            assert!(Act::ALL[..i].iter().all(|b| b.name() != a.name()));
        }
    }

    /// Far out, the functions give their limits, not NaN; and NaN stays NaN.
    #[test]
    fn activations_at_the_ends_of_the_line() {
        let (inf, nan) = (f64::INFINITY, f64::NAN);
        for act in Act::ALL {
            for x in [inf, -inf, 1e300, -1e300, 800.0, -800.0, 40.0, -40.0] {
                let (f, df) = (act.f(x), act.df(x));
                assert!(!f.is_nan() && !df.is_nan(), "{} at {x}: {f}, {df}", act.name());
                assert!(df.is_finite(), "{} at {x}: slope {df}", act.name());
            }
            assert!(act.f(nan).is_nan() && act.df(nan).is_nan(), "{}", act.name());
        }
        assert_eq!(Act::Gelu.f(-inf), 0.0);
        assert_eq!(Act::Silu.f(-inf), 0.0);
        assert_eq!(Act::Elu.f(-inf), -1.0);
        assert_eq!(Act::Softplus.f(-inf), 0.0);
        assert_eq!(Act::Softplus.f(inf), inf);
        assert_eq!(Act::Sigmoid.f(-inf), 0.0);
        assert_eq!(Act::Tanh.f(-inf), -1.0);
        assert_eq!(Act::Gelu.df(inf), 1.0);
    }

    /// The normal distribution function that GELU is made of, against `stats` (twelve
    /// digits) over the whole range where it is not 0 or 1, and against values worked
    /// out to twenty digits (Python's `decimal`) to fifteen.
    #[test]
    fn the_normal_distribution_function_is_accurate() {
        for i in -3700..=900 {
            let x = i as f64 * 0.01;
            let (ours, theirs) = (normal_cdf(x), crate::math::stats::normal_cdf(x));
            assert!((ours - theirs).abs() <= 1e-11 * theirs, "at {x}: {ours} against {theirs}");
        }
        for (x, want) in [
            (-20.0, 2.7536241186062337e-89),
            (-10.0, 7.619853024160525e-24),
            (-6.0, 9.86587645037698e-10),
            (-3.0, 0.0013498980316300946),
            (-1.5, 0.06680720126885807),
            (-0.75, 0.2266273523768682),
            (-0.3, 0.3820885778110474),
            (0.7, 0.758036347776927),
            (2.0, 0.9772498680518208),
            (5.0, 0.9999997133484281),
        ] {
            let got = normal_cdf(x);
            assert!((got - want).abs() <= 1e-14 * want, "Φ({x}) = {got}, not {want}");
        }
        // and the pieces join: no step where one gives way to the next (the
        // arguments 1, 2, 4 and 8 of erfc), beyond what the slope makes over the gap
        for y in [1.0f64, 2.0, 4.0, 8.0] {
            let x = -y * std::f64::consts::SQRT_2;
            let (below, above) = (normal_cdf(x - 1e-12), normal_cdf(x + 1e-12));
            assert!((above - below).abs() <= 1e-14 * above + 2e-12 * above * x.abs() * 1.5, "step at {x}: {below} {above}");
        }
        assert_eq!((normal_cdf(f64::NEG_INFINITY), normal_cdf(f64::INFINITY), normal_cdf(-40.0)), (0.0, 1.0, 0.0));
        assert!(normal_cdf(f64::NAN).is_nan());
    }

    /// The loss's gradient with respect to the logits is (softmax − one-hot) divided by
    /// the number of rows scored, exactly as the docs say; unscored rows get nothing.
    #[test]
    fn cross_entropy_hands_back_softmax_minus_one_hot() {
        let mut rng = Rng::new(3);
        let (m, n) = (5, 4);
        let logits = uniform(&mut rng, m * n, -3.0, 3.0);
        let mut targets: Vec<usize> = (0..m).map(|_| rng.below(n)).collect();
        for unscored in [None, Some(2)] {
            if let Some(r) = unscored {
                targets[r] = NO_TARGET;
            }
            let scored = targets.iter().filter(|&&t| t < n).count() as f64;
            let mut tape = Tape::new();
            let x = tape.param(m, n, &logits);
            let p = tape.softmax_rows(x);
            let loss = tape.cross_entropy(x, &targets);
            tape.backward(loss);
            let mut want_loss = 0.0;
            for r in 0..m {
                for j in 0..n {
                    let (pj, g) = (tape.value(p)[r * n + j], tape.grad(x)[r * n + j]);
                    let want = if targets[r] < n { (pj - if j == targets[r] { 1.0 } else { 0.0 }) / scored } else { 0.0 };
                    assert!((g - want).abs() < 1e-15, "row {r} column {j}: {g} against {want}");
                }
                if targets[r] < n {
                    want_loss -= tape.value(p)[r * n + targets[r]].ln() / scored;
                }
            }
            assert!((tape.value(loss)[0] - want_loss).abs() < 1e-14);
        }
        // nothing scored: no loss and no gradient
        let mut tape = Tape::new();
        let x = tape.param(2, 3, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let loss = tape.cross_entropy(x, &[NO_TARGET, 7]);
        tape.backward(loss);
        assert_eq!(tape.value(loss), &[0.0]);
        assert!(tape.grad(x).iter().all(|&g| g == 0.0));
    }

    /// Whole computations, not single steps: a three-layer network with each
    /// normalisation (and its gain and bias), on both losses, and one attention head
    /// with the causal mask.
    #[test]
    fn compositions_have_the_gradient_of_their_finite_differences() {
        let mut rng = Rng::new(4);
        type Norm = fn(&mut Tape, Var) -> Var;
        let norms: [(&str, Norm); 3] = [("layer", |t, x| t.layer_norm(x, 1e-5)), ("rms", |t, x| t.rms_norm(x, 1e-5)), ("batch", |t, x| t.batch_norm(x, 1e-5))];
        for (name, norm) in norms {
            for classes in [false, true] {
                let targets = [0usize, 2, 1, 1, 0];
                let wanted = uniform(&mut rng, 10, -1.0, 1.0);
                let net = move |t: &mut Tape, v: &[Var]| {
                    // v: x, then for each of three layers its weights and bias, then a
                    // gain and a bias for the two normalisations
                    let mut h = v[0];
                    for l in 0..3 {
                        h = t.matmul_t(h, v[1 + 2 * l]);
                        h = t.add_row(h, v[2 + 2 * l]);
                        if l < 2 {
                            h = norm(t, h);
                            h = t.mul_row(h, v[7 + 2 * l]);
                            h = t.add_row(h, v[8 + 2 * l]);
                            h = t.act(h, if l == 0 { Act::Tanh } else { Act::Gelu });
                        }
                    }
                    if classes {
                        t.cross_entropy(h, &targets)
                    } else {
                        let y = t.constant(5, 2, &wanted);
                        t.mse(h, y)
                    }
                };
                let outputs = if classes { 3 } else { 2 };
                let shapes = [(5, 3), (4, 3), (1, 4), (4, 4), (1, 4), (outputs, 4), (1, outputs), (1, 4), (1, 4), (1, 4), (1, 4)];
                check(&format!("three layers, {name} normalisation"), &random_inputs(&mut rng, &shapes), &net, &mut rng);
            }
        }
        let head = |t: &mut Tape, v: &[Var]| {
            let (q, k, val) = (t.matmul_t(v[0], v[1]), t.matmul_t(v[0], v[2]), t.matmul_t(v[0], v[3]));
            let s = t.matmul_t(q, k);
            let s = t.scale(s, 1.0 / 3f64.sqrt());
            let p = t.softmax_rows_causal(s);
            let mix = t.matmul(p, val);
            t.matmul_t(mix, v[4])
        };
        let shapes = [(6, 4), (3, 4), (3, 4), (3, 4), (4, 3)];
        check("an attention head", &random_inputs(&mut rng, &shapes), &head, &mut rng);
    }

    /// A causal head hands nothing to the future: the gradient of position i's output
    /// reaches no position after i.
    #[test]
    fn a_causal_head_sends_no_gradient_ahead() {
        let mut rng = Rng::new(5);
        let inputs = random_inputs(&mut rng, &[(6, 4), (3, 4), (3, 4), (3, 4)]);
        let mut tape = Tape::new();
        let v: Vec<Var> = inputs.iter().map(|(r, c, x)| tape.param(*r, *c, x)).collect();
        let (q, k, val) = (tape.matmul_t(v[0], v[1]), tape.matmul_t(v[0], v[2]), tape.matmul_t(v[0], v[3]));
        let s = tape.matmul_t(q, k);
        let p = tape.softmax_rows_causal(s);
        let out = tape.matmul(p, val);
        for i in 0..6 {
            let mut seed = vec![0.0; 18];
            seed[i * 3..i * 3 + 3].copy_from_slice(&[0.3, -1.0, 0.7]);
            tape.backward_seeded(out, &seed);
            for j in 0..6 {
                let reached = tape.grad(v[0])[j * 4..j * 4 + 4].iter().any(|&g| g != 0.0);
                assert_eq!(reached, j <= i, "position {i}'s gradient at position {j}");
            }
            // and the masked scores get exactly nothing
            assert!(tape.grad(s).iter().enumerate().all(|(e, &g)| e % 6 <= e / 6 || g == 0.0));
        }
    }

    /// The softmax's Jacobian is diag(p) − p pᵀ (what attngrad shows for a row).
    #[test]
    fn the_softmax_jacobian_is_diag_p_minus_p_p_transposed() {
        let mut tape = Tape::new();
        let x = tape.param(1, 4, &[0.3, -1.2, 2.0, 0.1]);
        let p = tape.softmax_rows(x);
        let pv = tape.value(p).to_vec();
        let jac = tape.jacobian(p, x);
        for i in 0..4 {
            for j in 0..4 {
                let want = if i == j { pv[i] } else { 0.0 } - pv[i] * pv[j];
                assert!((jac[i * 4 + j] - want).abs() < 1e-15);
            }
        }
    }

    /// `jacobian` against finite differences of every output entry, and each of its
    /// rows against a seeded sweep (a vector-Jacobian product with a unit vector).
    #[test]
    fn the_jacobian_agrees_with_finite_differences() {
        let mut rng = Rng::new(6);
        let (x0, w0) = (uniform(&mut rng, 6, -1.0, 1.0), uniform(&mut rng, 12, -1.0, 1.0));
        let build = |t: &mut Tape, x: &[f64]| {
            let x = t.param(2, 3, x);
            let w = t.constant(4, 3, &w0);
            let z = t.matmul_t(x, w);
            let z = t.layer_norm(z, 1e-5);
            (x, t.act(z, Act::Tanh))
        };
        let mut tape = Tape::new();
        let (x, y) = build(&mut tape, &x0);
        let jac = tape.jacobian(y, x);
        assert_eq!(jac.len(), 8 * 6);
        for l in 0..6 {
            let h = 1e-6;
            let mut moved = x0.clone();
            moved[l] += h;
            let mut t = Tape::new();
            let (_, up) = build(&mut t, &moved);
            let up = t.value(up).to_vec();
            moved[l] -= 2.0 * h;
            let mut t = Tape::new();
            let (_, down) = build(&mut t, &moved);
            for k in 0..8 {
                let fd = (up[k] - t.value(down)[k]) / (2.0 * h);
                assert!((fd - jac[k * 6 + l]).abs() < 1e-8, "entry ({k}, {l}): {} against {fd}", jac[k * 6 + l]);
            }
        }
        for k in 0..8 {
            let mut seed = vec![0.0; 8];
            seed[k] = 1.0;
            tape.backward_seeded(y, &seed);
            assert_eq!(tape.grad(x), &jac[k * 6..(k + 1) * 6]);
        }
    }

    /// What the normalisation scene says, held to the operations themselves.
    #[test]
    fn the_normalisations_have_the_properties_the_scenes_state() {
        let mut rng = Rng::new(7);
        let row = uniform(&mut rng, 6, -2.0, 3.0);
        let eps = 1e-14;
        // layer normalisation: each row comes out with mean 0 and mean square 1
        let mut tape = Tape::new();
        let x = tape.param(3, 6, &[row.clone(), uniform(&mut rng, 6, -1.0, 1.0), uniform(&mut rng, 6, 5.0, 9.0)].concat());
        let y = tape.layer_norm(x, eps);
        for r in tape.value(y).chunks(6) {
            let mean = r.iter().sum::<f64>() / 6.0;
            let square = r.iter().map(|v| v * v).sum::<f64>() / 6.0;
            assert!(mean.abs() < 1e-15 && (square - 1.0).abs() < 1e-12, "{mean} {square}");
        }
        // its Jacobian sends the row itself and the all-ones vector to 0, so the
        // gradient that comes back is at right angles to both
        let jacobian_of = |norm: fn(&mut Tape, Var, f64) -> Var, row: &[f64]| {
            let mut tape = Tape::new();
            let x = tape.param(1, row.len(), row);
            let y = norm(&mut tape, x, eps);
            tape.jacobian(y, x)
        };
        let apply = |jac: &[f64], v: &[f64]| -> Vec<f64> { jac.chunks(v.len()).map(|r| dot(r, v)).collect() };
        let biggest = |v: Vec<f64>| v.iter().fold(0.0f64, |a, b| a.max(b.abs()));
        let jac = jacobian_of(Tape::layer_norm, &row);
        assert!(biggest(apply(&jac, &row)) < 1e-12);
        assert!(biggest(apply(&jac, &[1.0; 6])) < 1e-12);
        let g = uniform(&mut rng, 6, -1.0, 1.0);
        let back: Vec<f64> = (0..6).map(|j| (0..6).map(|i| g[i] * jac[i * 6 + j]).sum()).collect();
        assert!(dot(&back, &row).abs() < 1e-12 && back.iter().sum::<f64>().abs() < 1e-12);
        // RMS normalisation sends the row itself to 0 but not the all-ones vector
        let jac = jacobian_of(Tape::rms_norm, &row);
        assert!(biggest(apply(&jac, &row)) < 1e-12);
        assert!(biggest(apply(&jac, &[1.0; 6])) > 0.1);
        // its output does not change when its input is scaled, and the gradient that
        // comes back is 1/s times as large for an input s times as large
        let seed = uniform(&mut rng, 6, -1.0, 1.0);
        let run_rms = |s: f64| {
            let mut tape = Tape::new();
            let scaled: Vec<f64> = row.iter().map(|v| s * v).collect();
            let x = tape.param(1, 6, &scaled);
            let y = tape.rms_norm(x, eps);
            tape.backward_seeded(y, &seed);
            (tape.value(y).to_vec(), tape.grad(x).to_vec())
        };
        let (y1, g1) = run_rms(1.0);
        for s in [0.1, 3.0, 250.0] {
            let (ys, gs) = run_rms(s);
            for k in 0..6 {
                assert!((ys[k] - y1[k]).abs() < 1e-12);
                assert!((gs[k] * s - g1[k]).abs() < 1e-12 * (1.0 + g1[k].abs()));
            }
        }
        // the same for layer normalisation, which also ignores a shift
        let run_layer = |s: f64, shift: f64| {
            let mut tape = Tape::new();
            let moved: Vec<f64> = row.iter().map(|v| s * v + shift).collect();
            let x = tape.param(1, 6, &moved);
            let y = tape.layer_norm(x, eps);
            tape.backward_seeded(y, &seed);
            (tape.value(y).to_vec(), tape.grad(x).to_vec())
        };
        let (y1, g1) = run_layer(1.0, 0.0);
        let (ys, gs) = run_layer(4.0, -2.5);
        for k in 0..6 {
            assert!((ys[k] - y1[k]).abs() < 1e-12 && (gs[k] * 4.0 - g1[k]).abs() < 1e-12);
        }
        // batch normalisation standardises columns, and ties the examples together:
        // change one example and another's gradient changes
        let batch = uniform(&mut rng, 12, -1.0, 2.0);
        let run_batch = |data: &[f64]| {
            let mut tape = Tape::new();
            let x = tape.param(4, 3, data);
            let y = tape.batch_norm(x, eps);
            let w = tape.constant(4, 3, &[0.3, -0.2, 0.9, 1.1, 0.4, -0.7, 0.2, 0.8, -0.5, 0.6, -0.1, 0.3]);
            let l = tape.mul(y, w);
            let l = tape.sum(l);
            tape.backward(l);
            (tape.value(y).to_vec(), tape.grad(x).to_vec())
        };
        let (y, g) = run_batch(&batch);
        for j in 0..3 {
            let column: Vec<f64> = (0..4).map(|i| y[i * 3 + j]).collect();
            assert!(column.iter().sum::<f64>().abs() < 1e-14);
            assert!((column.iter().map(|v| v * v).sum::<f64>() / 4.0 - 1.0).abs() < 1e-12);
        }
        let mut changed = batch.clone();
        changed[3 * 3] += 0.5;
        let (_, g2) = run_batch(&changed);
        assert!((g2[0] - g[0]).abs() > 1e-3, "example 0's gradient did not depend on example 3");
    }

    /// Empty matrices, 1 by 1 matrices and numbers that are not finite, through every
    /// operation forward and back: nothing panics, and a NaN that goes in comes out.
    #[test]
    fn nothing_panics_on_empty_tiny_or_non_finite_input() {
        let ops = |t: &mut Tape, a: Var, r: Var, sq: Var, picks: &[usize], targets: &[usize]| {
            let mut out = vec![
                t.matmul_t(a, a),
                t.add(a, a),
                t.sub(a, a),
                t.mul(a, a),
                t.add_row(a, r),
                t.mul_row(a, r),
                t.scale(a, 2.0),
                t.transpose(a),
                t.layer_norm(a, 1e-5),
                t.rms_norm(a, 1e-5),
                t.batch_norm(a, 1e-5),
                t.softmax_rows(a),
                t.softmax_rows_causal(sq),
                t.gather_rows(a, picks),
                t.cross_entropy(a, targets),
                t.mse(a, a),
                t.sum(a),
                t.mean(a),
            ];
            for act in Act::ALL {
                out.push(t.act(a, act));
            }
            let at = t.transpose(a);
            out.push(t.matmul(a, at));
            out
        };
        for (rows, cols) in [(0, 0), (0, 3), (3, 0), (1, 1), (2, 3)] {
            for fill in [0.5, f64::NAN, f64::INFINITY, f64::NEG_INFINITY, 0.0] {
                let mut tape = Tape::new();
                let a = tape.param(rows, cols, &vec![fill; rows * cols]);
                let r = tape.param(1, cols, &vec![fill; cols]);
                let sq = tape.param(rows, rows, &vec![fill; rows * rows]);
                let picks: Vec<usize> = (0..rows).chain([rows + 5]).collect();
                let targets: Vec<usize> = (0..rows).map(|i| i % cols.max(1)).collect();
                let outs = ops(&mut tape, a, r, sq, &picks, &targets);
                for &o in &outs {
                    let n = tape.value(o).len();
                    tape.backward_seeded(o, &vec![1.0; n]);
                    if fill.is_nan() && n > 0 && rows * cols > 0 {
                        // (the row-lookup's rows are copies, and the causal softmax of
                        // one NaN is NaN: every output that holds numbers holds a NaN)
                        assert!(tape.value(o).iter().any(|v| v.is_nan()), "{rows}x{cols}: NaN lost by node {}", o.index());
                    }
                }
                tape.backward(outs[16]);
                let _ = tape.jacobian(outs[8], a);
                // a row the table does not have is NaN, not a panic
                let g = outs[13];
                assert!(tape.value(g)[rows * cols..].iter().all(|v| v.is_nan()));
            }
        }
        // a handle from a cleared tape reads as empty
        let mut tape = Tape::new();
        let x = tape.scalar(1.0);
        tape.clear();
        assert!(tape.value(x).is_empty() && tape.grad(x).is_empty() && tape.shape(x) == (0, 0));
    }

    /// Shapes that do not fit: in a release build, a node of the expected shape full of
    /// NaN that passes nothing back. (A debug build stops at the `debug_assert`, so
    /// this only runs without one, as the tests are normally run.)
    #[test]
    fn shapes_that_do_not_fit_give_nan_not_a_panic() {
        if cfg!(debug_assertions) {
            return;
        }
        let mut tape = Tape::new();
        let a = tape.param(2, 3, &[1.0; 6]);
        let b = tape.param(4, 2, &[1.0; 8]);
        let bad = [
            (tape.matmul(a, b), (2, 2)),
            (tape.matmul_t(a, b), (2, 4)),
            (tape.add(a, b), (2, 3)),
            (tape.mul_row(a, b), (2, 3)),
            (tape.softmax_rows_causal(a), (2, 3)),
            (tape.mse(a, b), (1, 1)),
            (tape.cross_entropy(a, &[0]), (1, 1)),
            (tape.constant(3, 3, &[1.0; 4]), (3, 3)),
        ];
        for (v, shape) in bad {
            assert_eq!(tape.shape(v), shape);
            assert!(tape.value(v).iter().all(|x| x.is_nan()));
            let n = tape.value(v).len();
            tape.backward_seeded(v, &vec![1.0; n]);
            assert!(tape.grad(a).iter().chain(tape.grad(b)).all(|&g| g == 0.0));
        }
        let loss = tape.sum(a);
        tape.backward_seeded(loss, &[1.0, 2.0]);
        assert!(tape.grad(a).iter().all(|g| g.is_nan()));
        tape.backward(a);
        assert!(tape.grad(a).iter().all(|g| g.is_nan()));
        // and a handle from before a clear is no handle at all
        tape.clear();
        tape.backward(a);
        assert!(tape.matmul(a, b).index() == 0 && tape.value(Var(0)).iter().all(|x| x.is_nan()));
    }

    /// Gradients are cleared before every sweep (not added to), a node the loss does
    /// not depend on gets zeros, and a cleared tape keeps its memory: a second pass of
    /// the same size allocates nothing and gives the same numbers.
    #[test]
    fn sweeps_start_afresh_and_a_cleared_tape_keeps_its_memory() {
        let mut tape = Tape::new();
        let pass = |tape: &mut Tape| {
            tape.clear();
            let x = tape.param(2, 2, &[1.0, 2.0, 3.0, 4.0]);
            let unused = tape.param(1, 2, &[5.0, 6.0]);
            let y = tape.act(x, Act::Silu);
            let after = tape.scale(x, 3.0);
            let loss = tape.sum(y);
            let _ = tape.mean(after);
            tape.backward(loss);
            (x, unused, after, loss, tape.grad(x).to_vec())
        };
        let (x, unused, after, loss, first) = pass(&mut tape);
        tape.backward(loss);
        tape.backward(loss);
        assert_eq!(tape.grad(x), first.as_slice());
        assert!(tape.grad(unused).iter().chain(tape.grad(after)).all(|&g| g == 0.0));
        let capacity = (tape.vals.capacity(), tape.grads.capacity(), tape.aux.capacity(), tape.nodes.capacity());
        let (_, _, _, _, second) = pass(&mut tape);
        assert_eq!(first, second);
        assert_eq!(capacity, (tape.vals.capacity(), tape.grads.capacity(), tape.aux.capacity(), tape.nodes.capacity()));
        assert_eq!(tape.params().count(), 2);
        assert!(tape.is_param(x) && !tape.is_param(after));
    }

    /// The example in the module's documentation, as written there.
    #[test]
    fn the_example_in_the_docs_runs() {
        let (points, weights, biases) = ([0.1, 0.2, -0.3, 0.5, 0.9, -1.0, 0.0, 0.4], [0.5, -0.2, 0.3, 0.8, -0.6, 0.1], [0.0, 0.1, -0.1]);
        let wanted = [0.0; 12];
        let mut tape = Tape::new();
        let x = tape.constant(4, 2, &points);
        let w = tape.param(3, 2, &weights);
        let b = tape.param(1, 3, &biases);
        let z = tape.matmul_t(x, w);
        let z = tape.add_row(z, b);
        let a = tape.act(z, Act::Tanh);
        let y = tape.constant(4, 3, &wanted);
        let loss = tape.mse(a, y);
        tape.backward(loss);
        assert_eq!((tape.grad(w).len(), tape.grad(z).len()), (6, 12));
        // by hand: dL/dw(j, i) = Σ over examples of (2 a / 12) (1 − a²) x(i)
        let av = tape.value(a);
        for j in 0..3 {
            for i in 0..2 {
                let want: f64 = (0..4).map(|s| 2.0 * av[s * 3 + j] / 12.0 * (1.0 - av[s * 3 + j].powi(2)) * points[s * 2 + i]).sum();
                assert!((tape.grad(w)[j * 2 + i] - want).abs() < 1e-15);
            }
        }
    }
}

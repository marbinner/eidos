//! The optimisers of the Optimisation course: every rule its scenes train with, as a
//! step on an [`arch::Model`](super::arch::Model) that can be looked inside, and the
//! matrix functions that Muon and its relatives are made of. Every scene that says
//! "Adam", "Muon" or "Shampoo" means what this file computes, so that two scenes never
//! disagree about a rule.
//!
//! How a scene uses it, each training step:
//!
//! ```text
//!     let mut model = Model::mlp(&spec, &mut Rng::new(seed));
//!     let hyper = Hyper::default_for(Rule::Muon);        // or Hyper::muon_blog(), ...
//!     let mut opt = Optimizer::new(&model, Rule::Muon, hyper);
//!     let schedule = Schedule::cosine(hyper.lr, 0.0, 2000).with_warmup(100, 0.0);
//!
//!     let trace = model.loss_mlp(&mut tape, &x, &y, Loss::CrossEntropy);  // a mean loss
//!     tape.backward(trace.loss);
//!     model.zero_grads();
//!     model.accumulate(&tape, &trace);              // the batch's MEAN gradient
//!     opt.step(&mut model, schedule.at(opt.steps()));
//!     opt.update(i); opt.decay(i); opt.ortho_out(i); // what the step did to group i
//! ```
//!
//! Where the rules come from. Each rule is written as its source gives it, with the
//! source's constants and defaults, and its documentation names the source, the form
//! (a paper's or a library's) and the convention. The sources are the ones the course's
//! fact-checked notes cite; the short names used below are:
//!
//! - PyTorch: the PyTorch 2.15 documentation of `torch.optim.SGD`, `Adagrad`, `RMSprop`,
//!   `Adam` and `AdamW` (their algorithm boxes and defaults).
//! - Kingma and Ba: "Adam: A Method for Stochastic Optimization" (2015), Algorithm 1.
//! - Loshchilov and Hutter: "Decoupled Weight Decay Regularization" (2019), Algorithm 2.
//! - Cohen et al. 2022: "Adaptive Gradient Methods at the Edge of Stability" (its Table 1
//!   of momentum conventions, and its definition of Adam's preconditioner).
//! - The blog (S1): Jordan et al., "Muon: An optimizer for hidden layers in neural
//!   networks" (Dec 2024), its algorithm box and code.
//! - The reference (S2): the KellerJordan/Muon repository, `muon.py` as of its last change
//!   (Dec 2025), and its history.
//! - Moonshot (S7): "Muon is Scalable for LLM Training" (Feb 2025), §2.2.
//! - torch Muon (S21): `torch.optim.Muon` (PyTorch's `_muon.py`, main branch).
//! - Shampoo (S8): Gupta, Koren and Singer (2018), Algorithm 1. SOAP (S9): Vyas et al.
//!   (2024) and its official code (`soap.py`, Dec 2024). And so on for Signum (S12), Lion
//!   (S13), Scion (S14), AdaMuon (S16), PolarGrad (S17), Polar Express (S18), NorMuon
//!   (S19), stochastic spectral descent (S10), Orthogonal-SGDM (S11).
//!
//! Conventions, kept throughout:
//!
//! - Everything is f64, dense and row-major (as in `autograd` and `arch`): a group of
//!   `rows` by `cols` holds entry (i, j) at `i * cols + j`. A weight matrix has one row
//!   per output unit, so rows are fan-out and columns fan-in, as in PyTorch, which is the
//!   convention every shape factor below is written in.
//! - The gradient a step uses is what the model's groups hold in `grad`: the scene has
//!   put the MEAN gradient of its batch there. The learning rate is the `lr` handed to
//!   [`Optimizer::step`], so that a schedule ([`Schedule`]) is the scene's business.
//! - Nothing here panics on any number a slider can produce: a zero gradient, a matrix
//!   of rank 1, a 1 by 1 matrix, a learning rate of 0. Numbers that are not finite are
//!   carried (a NaN gradient makes NaN values), never unwrapped. A step allocates
//!   nothing: every buffer is made when the optimiser is.
//! - Running averages are flushed to 0 when they fall below 1e-300 in size (as
//!   `arch::Model::step_adam` does): below the normal range of doubles arithmetic becomes
//!   very slow, in WebAssembly most of all. No result above that size changes.
//! - The reference Muon runs its polynomial iteration in bfloat16; here everything is in
//!   f64 (as are the course notes' own tables), so the numbers are, to rounding, those of
//!   exact arithmetic.
//!
//! What a step costs, to budget a frame by: the figures are in [`Optimizer`]'s
//! documentation. In short, a few nanoseconds per parameter for the rules that treat
//! parameters alike, and for the matrix rules mostly the orthogonalisation or the
//! eigen-decompositions: about half a millisecond on a 6-layer MLP 32 wide, two to five
//! milliseconds on a 4-block transformer 32 wide (Shampoo with MLPs of 128: eighteen).

// (Nothing uses this yet: the Optimisation scenes are its users.)
#![allow(dead_code)]

use super::arch::{Group, Kind, Model, Part};

// =====================================================================================
// Matrix functions
// =====================================================================================

/// The rounding unit of f64, about 2.2e-16.
const ULP: f64 = f64::EPSILON;

/// The most sweeps of rotations an eigen-decomposition makes (it needs about six to
/// ten for the matrices here; this only bounds the work on a matrix that will not
/// settle).
const SWEEPS: usize = 50;

/// Numbers this small are flushed to 0 in the running averages (see the module's
/// documentation).
#[inline]
fn flush(x: f64) -> f64 {
    if x.abs() < 1e-300 {
        0.0
    } else {
        x
    }
}

/// The sign of one number: 1, −1, or 0 for 0 (either zero); NaN for NaN. (Without
/// branches on the sign: gradients with many exact zeros, a transformer's unused
/// embedding rows, would make them unpredictable.)
#[inline]
fn sgn(x: f64) -> f64 {
    let s = (x > 0.0) as u8 as f64 - (x < 0.0) as u8 as f64;
    if x.is_nan() {
        f64::NAN
    } else {
        s
    }
}

/// The square root of a number that should not be negative but may be, by rounding: 0
/// for a negative number, NaN kept.
#[inline]
fn root0(x: f64) -> f64 {
    if x < 0.0 {
        0.0
    } else {
        x.sqrt()
    }
}

fn dot(a: &[f64], b: &[f64]) -> f64 {
    let n = a.len().min(b.len());
    let (a, b) = (&a[..n], &b[..n]);
    let mut acc = [0.0f64; 4];
    let (ca, cb) = (a.chunks_exact(4), b.chunks_exact(4));
    let (ra, rb) = (ca.remainder(), cb.remainder());
    for (x, y) in ca.zip(cb) {
        acc[0] += x[0] * y[0];
        acc[1] += x[1] * y[1];
        acc[2] += x[2] * y[2];
        acc[3] += x[3] * y[3];
    }
    let mut s = (acc[0] + acc[1]) + (acc[2] + acc[3]);
    for (x, y) in ra.iter().zip(rb) {
        s += x * y;
    }
    s
}

/// `out` (rows by rows) = X Xᵀ for X `rows` by `cols`.
fn gram_rows(x: &[f64], rows: usize, cols: usize, out: &mut [f64]) {
    for i in 0..rows {
        let xi = &x[i * cols..(i + 1) * cols];
        for j in i..rows {
            let s = dot(xi, &x[j * cols..(j + 1) * cols]);
            out[i * rows + j] = s;
            out[j * rows + i] = s;
        }
    }
}

/// `out` (cols by cols) = Xᵀ X for X `rows` by `cols`.
fn gram_cols(x: &[f64], rows: usize, cols: usize, out: &mut [f64]) {
    let out = &mut out[..cols * cols];
    out.fill(0.0);
    for k in 0..rows {
        let xk = &x[k * cols..(k + 1) * cols];
        for i in 0..cols {
            let a = xk[i];
            for (o, &b) in out[i * cols + i..(i + 1) * cols].iter_mut().zip(&xk[i..]) {
                *o += a * b;
            }
        }
    }
    for i in 0..cols {
        for j in i + 1..cols {
            out[j * cols + i] = out[i * cols + j];
        }
    }
}

/// `out` (m by n) = A B, for A m by k and B k by n. Blocks of 4 rows by 8 columns of
/// the result are added up in registers (most of the work of the polynomial iteration
/// and of SOAP's rotations is here).
fn mul(a: &[f64], b: &[f64], out: &mut [f64], m: usize, k: usize, n: usize) {
    let (a, b, out) = (&a[..m * k], &b[..k * n], &mut out[..m * n]);
    let (m4, n8) = (m - m % 4, n - n % 8);
    for i in (0..m4).step_by(4) {
        for j in (0..n8).step_by(8) {
            let mut acc = [[0.0f64; 8]; 4];
            for p in 0..k {
                let bp: &[f64; 8] = b[p * n + j..p * n + j + 8].try_into().unwrap_or(&[0.0; 8]);
                for (r, row) in acc.iter_mut().enumerate() {
                    let x = a[(i + r) * k + p];
                    for q in 0..8 {
                        row[q] += x * bp[q];
                    }
                }
            }
            for (r, row) in acc.iter().enumerate() {
                out[(i + r) * n + j..(i + r) * n + j + 8].copy_from_slice(row);
            }
        }
        // the columns left over
        for r in i..i + 4 {
            for j in n8..n {
                out[r * n + j] = (0..k).map(|p| a[r * k + p] * b[p * n + j]).sum();
            }
        }
    }
    // the rows left over
    for i in m4..m {
        let o = &mut out[i * n..(i + 1) * n];
        o.fill(0.0);
        for p in 0..k {
            let x = a[i * k + p];
            for (o, &y) in o.iter_mut().zip(&b[p * n..(p + 1) * n]) {
                *o += x * y;
            }
        }
    }
}

/// `out` (m by n) = A Bᵀ, for A m by k and B n by k.
fn mul_nt(a: &[f64], b: &[f64], out: &mut [f64], m: usize, k: usize, n: usize) {
    for i in 0..m {
        let ai = &a[i * k..(i + 1) * k];
        for j in 0..n {
            out[i * n + j] = dot(ai, &b[j * k..(j + 1) * k]);
        }
    }
}

/// `out` (m by n) = Aᵀ B, for A k by m and B k by n.
fn mul_tn(a: &[f64], b: &[f64], out: &mut [f64], m: usize, k: usize, n: usize) {
    let out = &mut out[..m * n];
    out.fill(0.0);
    for p in 0..k {
        let bp = &b[p * n..(p + 1) * n];
        for i in 0..m {
            let x = a[p * m + i];
            for (o, &y) in out[i * n..(i + 1) * n].iter_mut().zip(bp) {
                *o += x * y;
            }
        }
    }
}

/// Room for the matrix functions to work in, kept from call to call so that a step
/// allocates nothing: the buffers grow to the largest size asked for and stay. One
/// can be shared by any number of calls (not at the same time).
#[derive(Clone, Debug, Default)]
pub struct Scratch {
    /// k by k: a Gram matrix, a factor matrix, the matrix being rotated.
    g: Vec<f64>,
    /// k by k: eigenvectors (one per row).
    vt: Vec<f64>,
    /// k by k: products.
    w: Vec<f64>,
    /// k: eigenvalues.
    vals: Vec<f64>,
    /// Three matrices the size of the largest group.
    x: Vec<f64>,
    y: Vec<f64>,
    z: Vec<f64>,
}

fn grow(v: &mut Vec<f64>, n: usize) {
    if v.len() < n {
        v.resize(n, 0.0);
    }
}

impl Scratch {
    /// Empty; it grows on first use.
    pub fn new() -> Scratch {
        Scratch::default()
    }

    /// Make room for square matrices of side `k` and matrices of `n` entries.
    fn fit(&mut self, k: usize, n: usize) {
        grow(&mut self.g, k * k);
        grow(&mut self.vt, k * k);
        grow(&mut self.w, k * k);
        grow(&mut self.vals, k);
        grow(&mut self.x, n);
        grow(&mut self.y, n);
        grow(&mut self.z, n);
    }
}

/// The eigen-decomposition of the symmetric `n` by `n` matrix `a` (which is destroyed),
/// by cyclic Jacobi rotations (the method of `linalg::eig_sym`, written here without
/// allocating): eigenvalues largest first into `vals`, and eigenvector k, of unit
/// length, as row k of `vt`. With `warm`, `vt` already holds an orthonormal basis (rows)
/// near the eigenvectors and `a` is the matrix written in that basis (vt A vtᵀ): the
/// rotations then continue from it, which takes far fewer sweeps when the matrix has
/// changed little since that basis was found. Sweeps stop when the off-diagonal part
/// is below n·ε times the matrix's size (ε the rounding unit): the eigenvalues are then
/// right to about that absolute accuracy. A matrix with a number that is not finite
/// gives NaN eigenvalues and eigenvectors at once.
fn jacobi(a: &mut [f64], n: usize, vt: &mut [f64], vals: &mut [f64], warm: bool) {
    let (a, vt, vals) = (&mut a[..n * n], &mut vt[..n * n], &mut vals[..n]);
    if !warm {
        vt.fill(0.0);
        for i in 0..n {
            vt[i * n + i] = 1.0;
        }
    }
    if a.iter().any(|x| !x.is_finite()) {
        vals.fill(f64::NAN);
        vt.fill(f64::NAN);
        return;
    }
    let norm = a.iter().map(|x| x * x).sum::<f64>().sqrt();
    let stop = (n as f64 * ULP * norm).powi(2);
    let skip = ULP * norm / n.max(1) as f64;
    for _ in 0..SWEEPS {
        let mut off = 0.0;
        for p in 0..n {
            for q in p + 1..n {
                off += 2.0 * a[p * n + q] * a[p * n + q];
            }
        }
        if off <= stop {
            break;
        }
        for p in 0..n {
            for q in p + 1..n {
                let apq = a[p * n + q];
                if apq.abs() <= skip {
                    continue;
                }
                let (app, aqq) = (a[p * n + p], a[q * n + q]);
                // the rotation in the (p, q) plane that zeroes a[p][q]
                let theta = (aqq - app) / (2.0 * apq);
                let t = theta.signum() / (theta.abs() + (theta * theta + 1.0).sqrt());
                let c = 1.0 / (t * t + 1.0).sqrt();
                let s = t * c;
                for k in 0..n {
                    let (x, y) = (a[p * n + k], a[q * n + k]);
                    a[p * n + k] = c * x - s * y;
                    a[q * n + k] = s * x + c * y;
                }
                a[p * n + p] = app - t * apq;
                a[q * n + q] = aqq + t * apq;
                a[p * n + q] = 0.0;
                a[q * n + p] = 0.0;
                for k in 0..n {
                    if k != p && k != q {
                        a[k * n + p] = a[p * n + k];
                        a[k * n + q] = a[q * n + k];
                    }
                }
                for k in 0..n {
                    let (x, y) = (vt[p * n + k], vt[q * n + k]);
                    vt[p * n + k] = c * x - s * y;
                    vt[q * n + k] = s * x + c * y;
                }
            }
        }
    }
    for i in 0..n {
        vals[i] = a[i * n + i];
    }
    // largest first
    for i in 0..n {
        let mut best = i;
        for j in i + 1..n {
            if vals[j] > vals[best] {
                best = j;
            }
        }
        if best != i {
            vals.swap(i, best);
            for k in 0..n {
                vt.swap(i * n + k, best * n + k);
            }
        }
    }
}

/// The eigen-decomposition of the symmetric `m` (n by n, kept) into `vt` (rows) and
/// `vals`, through `work` and `work2` (n by n each). With `warm`, starting from the
/// basis already in `vt` (see [`jacobi`]).
fn eig_into(m: &[f64], n: usize, vt: &mut [f64], vals: &mut [f64], work: &mut [f64], work2: &mut [f64], warm: bool) {
    let warm = warm && vt[..n * n].iter().all(|x| x.is_finite());
    if warm {
        mul(&vt[..n * n], m, work2, n, n, n);
        mul_nt(&work2[..n * n], &vt[..n * n], work, n, n, n);
        for i in 0..n {
            for j in i + 1..n {
                let s = 0.5 * (work[i * n + j] + work[j * n + i]);
                work[i * n + j] = s;
                work[j * n + i] = s;
            }
        }
    } else {
        work[..n * n].copy_from_slice(&m[..n * n]);
    }
    jacobi(work, n, vt, vals, warm);
}

/// The eigenvalues (largest first) and unit eigenvectors of the symmetric `n` by `n`
/// matrix `a` (row-major; only its upper triangle and diagonal are read, and the lower
/// triangle is taken to mirror them): eigenvector k is `vectors[k * n..(k + 1) * n]`,
/// the layout of `linalg::eig_sym`, whose method (cyclic Jacobi rotations) this is,
/// written to work in a [`Scratch`] instead of allocating. This is what Shampoo and
/// SOAP find their matrix roots and eigenvectors with. Where an eigenvalue repeats, its
/// eigenvectors are any orthonormal basis of its eigenspace. A matrix with a number that
/// is not finite gives NaN throughout.
pub fn sym_eigen(a: &[f64], n: usize) -> (Vec<f64>, Vec<f64>) {
    if a.len() < n * n {
        return (vec![f64::NAN; n], vec![f64::NAN; n * n]);
    }
    let mut m = vec![0.0; n * n];
    for i in 0..n {
        for j in i..n {
            m[i * n + j] = a[i * n + j];
            m[j * n + i] = a[i * n + j];
        }
    }
    let (mut vt, mut vals) = (vec![0.0; n * n], vec![0.0; n]);
    jacobi(&mut m, n, &mut vt, &mut vals, false);
    (vals, vt)
}

/// The Gram matrix of the shorter side of `a` (A Aᵀ for a wide or square matrix, Aᵀ A
/// for a tall one, k by k with k = min(rows, cols)) and its eigen-decomposition, left in
/// `s.vals` and `s.vt`. Returns k.
fn gram_eigen(a: &[f64], rows: usize, cols: usize, s: &mut Scratch) -> usize {
    let k = rows.min(cols);
    s.fit(k, rows * cols);
    if rows <= cols {
        gram_rows(a, rows, cols, &mut s.g);
    } else {
        gram_cols(a, rows, cols, &mut s.g);
    }
    jacobi(&mut s.g, k, &mut s.vt, &mut s.vals, false);
    k
}

/// Whether `a` holds a whole `rows` by `cols` matrix (a shorter slice is a mistake of
/// the caller: the functions below then answer NaN instead of panicking).
fn holds(a: &[f64], rows: usize, cols: usize) -> bool {
    let ok = a.len() >= rows * cols;
    debug_assert!(ok, "{} numbers are not a {rows} by {cols} matrix", a.len());
    ok
}

/// The singular values of the `rows` by `cols` matrix `a`, largest first (min(rows,
/// cols) of them), into `out`, working in `s`. They are the square roots of the
/// eigenvalues of the smaller Gram matrix (A Aᵀ or Aᵀ A), found by Jacobi rotations:
/// fast and sure on matrices of any rank (unlike `linalg::svd`, slow on matrices that
/// have lost rank), at the price that a singular value below about 1e-8 of the largest
/// is not resolved (its square is lost in the rounding of the largest one's), and comes
/// out as rounding noise of that size or 0.
pub fn singular_values_into(a: &[f64], rows: usize, cols: usize, out: &mut Vec<f64>, s: &mut Scratch) {
    out.clear();
    if !holds(a, rows, cols) {
        out.resize(rows.min(cols), f64::NAN);
        return;
    }
    let k = gram_eigen(a, rows, cols, s);
    out.extend(s.vals[..k].iter().map(|&l| root0(l)));
}

/// The singular values of the `rows` by `cols` matrix `a`, largest first; see
/// [`singular_values_into`] (which allocates nothing).
pub fn singular_values(a: &[f64], rows: usize, cols: usize) -> Vec<f64> {
    let mut out = Vec::new();
    singular_values_into(a, rows, cols, &mut out, &mut Scratch::new());
    out
}

/// The Frobenius norm: the square root of the sum of the squares of the entries (and of
/// the squares of the singular values).
pub fn frobenius(a: &[f64]) -> f64 {
    dot(a, a).sqrt()
}

/// The largest absolute value of an entry (NaN if any entry is NaN; 0 for no entries).
pub fn max_abs(a: &[f64]) -> f64 {
    let mut m = 0.0f64;
    for &x in a {
        if x.is_nan() {
            return f64::NAN;
        }
        m = m.max(x.abs());
    }
    m
}

/// The spectral norm: the largest singular value (how much the matrix can stretch a
/// vector). See [`singular_values`].
pub fn spectral_norm(a: &[f64], rows: usize, cols: usize) -> f64 {
    singular_values(a, rows, cols).first().copied().unwrap_or(0.0)
}

/// The nuclear (trace) norm: the sum of the singular values. See [`singular_values`].
pub fn nuclear_norm(a: &[f64], rows: usize, cols: usize) -> f64 {
    singular_values(a, rows, cols).iter().sum()
}

/// The stable rank: the squared Frobenius norm over the squared spectral norm, the sum
/// of the squared singular values over the largest of them. It lies between 1 and the
/// rank, and is near the number of singular values close to the largest. 0 for the
/// zero matrix.
pub fn stable_rank(a: &[f64], rows: usize, cols: usize) -> f64 {
    let s = singular_values(a, rows, cols);
    let top = s.first().copied().unwrap_or(0.0);
    if top == 0.0 {
        return 0.0;
    }
    s.iter().map(|x| x * x).sum::<f64>() / (top * top)
}

/// One-sided Jacobi (Hestenes) on the `k` rows of `b` (k by m): plane rotations of pairs
/// of rows until every pair is orthogonal to the rounding unit, the same rotations
/// applied to the rows of `qt` (k by k, set to the identity first), so that afterwards
/// the original b = qtᵀ b_now with the rows of b_now orthogonal: their lengths are the
/// singular values, qtᵀ holds the left singular vectors and the normalised rows the right
/// ones. A row whose squared length is at or below (ε ‖b‖_F)² (ε the rounding unit) is
/// rounding noise and is left alone, which is what keeps a matrix that has lost rank from
/// rotating for ever (the trouble with `linalg::svd`). Unlike singular values taken from
/// a Gram matrix, these are right to about ε σ_max each, and the vectors to about ε over
/// the relative gap. Returns the squared lengths in `len2`.
fn hestenes(b: &mut [f64], k: usize, m: usize, qt: &mut [f64], len2: &mut [f64]) {
    let (b, qt, len2) = (&mut b[..k * m], &mut qt[..k * k], &mut len2[..k]);
    qt.fill(0.0);
    for i in 0..k {
        qt[i * k + i] = 1.0;
    }
    let tiny = (ULP * frobenius(b)).powi(2);
    for _ in 0..SWEEPS {
        // the lengths, afresh each sweep, then kept up to date through its rotations
        for i in 0..k {
            len2[i] = dot(&b[i * m..(i + 1) * m], &b[i * m..(i + 1) * m]);
        }
        let mut rotated = false;
        for p in 0..k {
            for q in p + 1..k {
                let (alpha, beta) = (len2[p], len2[q]);
                if alpha <= tiny || beta <= tiny {
                    continue;
                }
                let (bp, bq) = b[p * m..(q + 1) * m].split_at_mut((q - p) * m);
                let (bp, bq) = (&mut bp[..m], &mut bq[..m]);
                let gamma = dot(bp, bq);
                if gamma.abs() <= ULP * (alpha * beta).sqrt() {
                    continue;
                }
                rotated = true;
                let zeta = (beta - alpha) / (2.0 * gamma);
                let t = zeta.signum() / (zeta.abs() + (zeta * zeta + 1.0).sqrt());
                let c = 1.0 / (t * t + 1.0).sqrt();
                let s = c * t;
                for j in 0..m {
                    let (x, y) = (bp[j], bq[j]);
                    bp[j] = c * x - s * y;
                    bq[j] = s * x + c * y;
                }
                len2[p] = alpha - t * gamma;
                len2[q] = beta + t * gamma;
                let (qp, qq) = qt[p * k..(q + 1) * k].split_at_mut((q - p) * k);
                for (x, y) in qp[..k].iter_mut().zip(&mut qq[..k]) {
                    let (a, b) = (*x, *y);
                    *x = c * a - s * b;
                    *y = s * a + c * b;
                }
            }
        }
        if !rotated {
            break;
        }
    }
    for i in 0..k {
        len2[i] = dot(&b[i * m..(i + 1) * m], &b[i * m..(i + 1) * m]);
    }
}

/// The polar factor of `a` into `out` (both `rows` by `cols`), working in `s`, and the
/// nuclear norm (the sum of the singular values that count). For the reduced singular
/// value decomposition a = U S Vᵀ, this is U Vᵀ taken over the singular values that are
/// not zero: the "orthogonalised" matrix Muon's iteration aims at. It is computed by
/// one-sided Jacobi rotations on the shorter side's vectors (the rows of a wide matrix,
/// the columns of a tall one), which orthogonalise them without forming a Gram matrix, so
/// the result is right to about ε κ (ε = 2.2e-16 the rounding unit, κ the ratio of the
/// largest singular value to the smallest that counts): 2e-10 for κ = 1e6.
///
/// "Zero" means exactly this: a singular value σ counts as zero when σ ≤ 4 max(rows,
/// cols) ε σ_max (2.8e-14 σ_max for a 32 by 32 matrix), the size of the rounding in a
/// matrix that has lost rank; a direction whose vector is no longer than ε ‖a‖_F is
/// rounding and is never rotated. Such directions are left out, so a matrix of rank r
/// gives the partial isometry U_r V_rᵀ, which has r singular values of 1 and the rest 0 and
/// is not semi-orthogonal. That is also what the reference iteration does to a
/// rank-deficient input (the course notes, muon.md §8.1: any odd polynomial keeps zero
/// singular values at zero, so the iteration aims at U_r V_rᵀ, the smallest-Frobenius-norm
/// maximiser of ⟨A, T⟩ over spectral norm ≤ 1; the Polar Express paper found directions
/// below 1e-3 σ_max "barely" matter to training). The zero matrix gives the zero matrix; a
/// 1 by 1 matrix its sign; a matrix with a number that is not finite, NaN throughout.
pub fn polar_into(a: &[f64], rows: usize, cols: usize, out: &mut [f64], s: &mut Scratch) -> f64 {
    let n = rows * cols;
    if !holds(a, rows, cols) || out.len() < n {
        out.iter_mut().for_each(|x| *x = f64::NAN);
        return f64::NAN;
    }
    if n == 0 {
        return 0.0;
    }
    if a[..n].iter().any(|x| !x.is_finite()) {
        out[..n].fill(f64::NAN);
        return f64::NAN;
    }
    let (k, m) = (rows.min(cols), rows.max(cols));
    s.fit(k, n);
    let wide = rows <= cols;
    // the shorter side's vectors, one per row
    if wide {
        s.x[..n].copy_from_slice(&a[..n]);
    } else {
        for i in 0..rows {
            for j in 0..cols {
                s.x[j * rows + i] = a[i * cols + j];
            }
        }
    }
    hestenes(&mut s.x, k, m, &mut s.vt, &mut s.vals);
    let largest = s.vals[..k].iter().fold(0.0f64, |t, &l| t.max(l)).sqrt();
    let zero = 4.0 * m as f64 * ULP * largest;
    let mut nuclear = 0.0;
    for i in 0..k {
        let sigma = s.vals[i].sqrt();
        let f = if sigma > zero {
            nuclear += sigma;
            1.0 / sigma
        } else {
            0.0
        };
        s.x[i * m..(i + 1) * m].iter_mut().for_each(|x| *x *= f);
    }
    // U Vᵀ = Σᵢ qtᵢ (bᵢ / σᵢ)ᵀ: k by m, the matrix itself for a wide one, its transpose
    // for a tall one
    if wide {
        mul_tn(&s.vt, &s.x, out, k, k, m);
    } else {
        mul_tn(&s.vt, &s.x, &mut s.y, k, k, m);
        for i in 0..rows {
            for j in 0..cols {
                out[i * cols + j] = s.y[j * rows + i];
            }
        }
    }
    nuclear
}

/// The polar factor U Vᵀ of the `rows` by `cols` matrix `a`, over the singular values
/// that are not zero; see [`polar_into`] for exactly what that means and how it is
/// computed (that one allocates nothing).
pub fn polar_factor(a: &[f64], rows: usize, cols: usize) -> Vec<f64> {
    let mut out = vec![0.0; rows * cols];
    polar_into(a, rows, cols, &mut out, &mut Scratch::new());
    out
}

/// The sign of every entry: 1 for a positive one, −1 for a negative one, 0 for 0, NaN
/// for NaN. (Sign descent's direction, and the steepest-descent direction under the
/// max-norm ℓ∞: Bernstein and Newhouse, "Old Optimizer, New Norm", Prop. 2.)
pub fn sign(a: &[f64]) -> Vec<f64> {
    a.iter().map(|&x| sgn(x)).collect()
}

/// The inverse `p`-th root of the symmetric positive semi-definite `n` by `n` matrix
/// `a` with damping, (A + damping · I)^(−p), into `out`, working in `s`: with the
/// eigen-decomposition A + damping I = Q diag(λ) Qᵀ ([`sym_eigen`]'s method), the
/// matrix Q diag(λ^(−p)) Qᵀ. An eigenvalue at or below 4 n ε λ_max (rounding: ε =
/// 2.2e-16), or below zero, counts as zero and contributes nothing, so that without
/// damping the root is a pseudo-inverse root, as Shampoo needs for a gradient that is
/// not square or not of full rank (the course notes, muon.md §8.2, and DistributedShampoo's
/// pseudo-inverse option, §6.3). Shampoo uses p = 1/4 on both sides (its Algorithm 1).
/// A matrix with a number that is not finite gives NaN throughout.
pub fn inverse_root_into(a: &[f64], n: usize, p: f64, damping: f64, out: &mut [f64], s: &mut Scratch) {
    if !holds(a, n, n) || out.len() < n * n {
        out.iter_mut().for_each(|x| *x = f64::NAN);
        return;
    }
    s.fit(n, n * n);
    for i in 0..n {
        for j in 0..n {
            s.x[i * n + j] = a[i * n + j] + if i == j { damping } else { 0.0 };
        }
    }
    s.y[..n * n].copy_from_slice(&s.x[..n * n]);
    jacobi(&mut s.y, n, &mut s.vt, &mut s.vals, false);
    root_from_eigen(&s.vt, &s.vals, n, p, &mut s.w, out);
}

/// Q diag(f(λ)) Qᵀ with f(λ) = λ^(−p) for the eigenvalues that count, from eigenvectors
/// (rows of `vt`) and eigenvalues, into `out`, using `w` (n by n).
fn root_from_eigen(vt: &[f64], vals: &[f64], n: usize, p: f64, w: &mut [f64], out: &mut [f64]) {
    if n == 0 {
        return;
    }
    let largest = vals[0];
    if !largest.is_finite() {
        out[..n * n].fill(f64::NAN);
        return;
    }
    let zero = 4.0 * n as f64 * ULP * largest;
    for i in 0..n {
        let l = vals[i];
        let f = if l > zero && l > 0.0 {
            if p == 0.25 {
                1.0 / l.sqrt().sqrt()
            } else if p == 0.5 {
                1.0 / l.sqrt()
            } else {
                l.powf(-p)
            }
        } else {
            0.0
        };
        for j in 0..n {
            w[i * n + j] = f * vt[i * n + j];
        }
    }
    mul_tn(vt, w, out, n, n, n);
}

/// The inverse `p`-th root of a symmetric positive semi-definite matrix with damping;
/// see [`inverse_root_into`] (which allocates nothing).
pub fn inverse_root(a: &[f64], n: usize, p: f64, damping: f64) -> Vec<f64> {
    let mut out = vec![0.0; n * n];
    inverse_root_into(a, n, p, damping, &mut out, &mut Scratch::new());
    out
}

/// Orthonormalise the rows of the `n` by `n` matrix `a` in place, by Gram–Schmidt run
/// twice: the Q of a QR factorisation (of the transpose), up to the signs of its
/// vectors. A row that is a combination of the earlier ones (to rounding) is replaced
/// by the unit vector that completes the set best, so the result is orthonormal in every
/// case. (SOAP's refresh, `torch.linalg.qr` in its code: its Householder QR may choose
/// other signs, and another basis where the matrix has lost rank; signs do not change
/// SOAP's step.)
fn orthonormalise_rows(a: &mut [f64], n: usize) {
    for pass in 0..2 {
        for j in 0..n {
            let before = dot(&a[j * n..(j + 1) * n], &a[j * n..(j + 1) * n]).sqrt();
            for k in 0..j {
                let d = dot(&a[j * n..(j + 1) * n], &a[k * n..(k + 1) * n]);
                for i in 0..n {
                    a[j * n + i] -= d * a[k * n + i];
                }
            }
            let len = dot(&a[j * n..(j + 1) * n], &a[j * n..(j + 1) * n]).sqrt();
            if pass == 0 && (len.is_nan() || len <= 1e-12 * before) {
                // the coordinate direction least covered by the earlier rows
                let mut best = (f64::NEG_INFINITY, 0);
                for i in 0..n {
                    let covered: f64 = (0..j).map(|k| a[k * n + i] * a[k * n + i]).sum();
                    if 1.0 - covered > best.0 {
                        best = (1.0 - covered, i);
                    }
                }
                let e = best.1;
                for i in 0..n {
                    a[j * n + i] = if i == e { 1.0 } else { 0.0 };
                }
                for k in 0..j {
                    let d = a[k * n + e];
                    for i in 0..n {
                        a[j * n + i] -= d * a[k * n + i];
                    }
                }
                let len = dot(&a[j * n..(j + 1) * n], &a[j * n..(j + 1) * n]).sqrt();
                for i in 0..n {
                    a[j * n + i] /= len;
                }
                continue;
            }
            for i in 0..n {
                a[j * n + i] /= len;
            }
        }
    }
}

// ----- the polynomial iteration -------------------------------------------------------

/// An odd polynomial p(x) = a x + b x³ + c x⁵: one round of the iteration Muon uses in
/// place of the polar factor. Applied to a matrix X as a X + b (X Xᵀ) X + c (X Xᵀ)² X, it
/// keeps the singular vectors and applies p to every singular value (the blog, S1,
/// "Proving that NS iteration orthogonalizes the update"), so `rounds` rounds apply p
/// composed with itself that many times. The constants are the published ones, each
/// named after its source; the lists of different polynomials per round are in
/// [`Iteration`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Odd5 {
    /// The coefficient of x: the slope at 0, which sets how fast small singular values
    /// grow.
    pub a: f64,
    /// The coefficient of x³.
    pub b: f64,
    /// The coefficient of x⁵ (0 for a cubic).
    pub c: f64,
}

impl Odd5 {
    /// The polynomial a x + b x³ + c x⁵.
    pub const fn new(a: f64, b: f64, c: f64) -> Odd5 {
        Odd5 { a, b, c }
    }

    /// The same polynomial applied to x / s: (a/s, b/s³, c/s⁵). This is the Polar
    /// Express paper's "safety factor" (S18: each p(x) replaced by p(x / 1.01)).
    pub const fn safe(self, s: f64) -> Odd5 {
        Odd5 { a: self.a / s, b: self.b / (s * s * s), c: self.c / (s * s * s * s * s) }
    }

    /// The classical Newton–Schulz cubic 1.5 x − 0.5 x³ (Bernstein and Newhouse,
    /// "Old Optimizer, New Norm", App. A; Kovarik 1970, Björck and Bowie 1971): it
    /// converges to 1 from any start in (0, √3), slowly for small values (0.01 is still
    /// 0.076 after 5 rounds).
    pub const CUBIC: Odd5 = Odd5::new(1.5, -0.5, 0.0);

    /// The reference's quintic (3.4445, −4.7750, 2.0315), unchanged in the reference
    /// repository since its first version (S1, S2): tuned for the largest slope at 0
    /// (3.4445) while its iterates stay in about [0.7, 1.3]. It does not converge: its
    /// fixed points 0.868 and 1.264 are both unstable, and after enough rounds every
    /// value in (0, 1] lands in [0.6818, 1.1344] (the course notes, muon.md §8.3).
    pub const JORDAN: Odd5 = Odd5::new(3.4445, -4.7750, 2.0315);

    /// The blog's "simple baseline" quintic (2, −1.5, 0.5) (S1), which converges to 1;
    /// the track-3 benchmark of the NanoGPT speedrun uses it for 12 rounds (S3).
    pub const BLOG_BASELINE: Odd5 = Odd5::new(2.0, -1.5, 0.5);

    /// The classical degree-5 polynomial (15 x − 10 x³ + 3 x⁵) / 8 (the Polar Express
    /// paper, S18 §1.2), which converges to 1.
    pub const CLASSICAL_QUINTIC: Odd5 = Odd5::new(15.0 / 8.0, -10.0 / 8.0, 3.0 / 8.0);

    /// p(x), evaluated in the order the matrix round uses: a x + (b x² + c x⁴) x.
    pub fn at(self, x: f64) -> f64 {
        let x2 = x * x;
        self.a * x + (self.b * x2 + self.c * x2 * x2) * x
    }
}

/// "Tuned Muon Newton-Schulz coefficients", the NanoGPT speedrun's medium track, 14 Feb
/// 2025 (S3 records; attributed to You): five triples, one per round.
pub const YOU: [Odd5; 5] = [
    Odd5::new(4.0848, -6.8946, 2.9270),
    Odd5::new(3.9505, -6.3029, 2.6377),
    Odd5::new(3.7418, -5.5913, 2.3037),
    Odd5::new(2.8769, -3.1427, 1.2046),
    Odd5::new(2.8366, -3.0525, 1.2012),
];

/// The Polar Express coefficients of the NanoGPT speedrun's record 38 (29 Sep 2025, S3:
/// "Computed for num_iters=5, safety_factor=2e-2, cushion=2"), used with the input divided
/// by 1.02 ‖X‖_F + 1e-6; full numbers from the course notes' `calc/ns_more.py`.
pub const POLAR_EXPRESS_RECORD: [Odd5; 5] = [
    Odd5::new(8.156554524902461, -22.48329292557795, 15.878769915207462),
    Odd5::new(4.042929935166739, -2.808917465908714, 0.5000178451051316),
    Odd5::new(3.8916678022926607, -2.772484153217685, 0.5060648178503393),
    Odd5::new(3.285753657755655, -2.3681294933425376, 0.46449024233003106),
    Odd5::new(2.3465413258596377, -1.7097828382687081, 0.42323551169305323),
];

/// The first five Polar Express polynomials of the paper's code (S18 v5, App. A), each
/// already divided by the paper's safety factor 1.01 (p(x / 1.01)), used with the input
/// divided by 1.01 ‖X‖_F + 1e-7. The paper's list has eight; the course notes give the
/// first five (those a five-round run uses) and the last, (1.875, −1.25, 0.375), but not
/// the sixth and seventh, so only the five are here. These differ from the speedrun's
/// ([`POLAR_EXPRESS_RECORD`]): there is no single set of "Polar Express coefficients".
pub const POLAR_EXPRESS_PAPER: [Odd5; 5] = [
    Odd5::new(8.28721201814563, -23.595886519098837, 17.300387312530933).safe(1.01),
    Odd5::new(4.107059111542203, -2.9478499167379106, 0.5448431082926601).safe(1.01),
    Odd5::new(3.9486908534822946, -2.908902115962949, 0.5518191394370137).safe(1.01),
    Odd5::new(3.3184196573706015, -2.488488024314874, 0.51004894012372).safe(1.01),
    Odd5::new(2.300652019954817, -1.6689039845747493, 0.4188073119525673).safe(1.01),
];

/// How the iteration scales its input first, so that every singular value starts in
/// [0, 1]: X / (scale ‖X‖_F + eps), or with `clamp`, X / max(‖X‖_F, eps). (The Frobenius
/// norm is at least the largest singular value, so this bounds it by 1; "benign because
/// Ortho(cG) = Ortho(G)", S1.)
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Normalise {
    /// The factor on the Frobenius norm (1; 1.01 or 1.02 for the Polar Express lists).
    pub scale: f64,
    /// Added to the norm (or, with `clamp`, its floor): what keeps a zero matrix from
    /// being 0/0.
    pub eps: f64,
    /// Divide by max(‖X‖_F, eps) instead (torch's Muon, S21).
    pub clamp: bool,
}

impl Normalise {
    /// The reference's: X / (‖X‖_F + 1e-7) (S2, and the blog's code, S1).
    pub const REFERENCE: Normalise = Normalise { scale: 1.0, eps: 1e-7, clamp: false };
    /// torch's Muon: X / max(‖X‖_F, 1e-7) (S21, `norm().clamp(min=eps)`).
    pub const TORCH: Normalise = Normalise { scale: 1.0, eps: 1e-7, clamp: true };

    fn divisor(self, norm: f64) -> f64 {
        if self.clamp {
            norm.max(self.eps)
        } else {
            self.scale * norm + self.eps
        }
    }
}

/// The polynomial iteration Muon uses in place of the polar factor, as a named list of
/// polynomials, one per round: normalise the input as `normalise` says, then apply, in
/// round r (from 0), `polys[r]`, or the last polynomial of the list once the list has run
/// out (so a list of one polynomial is that polynomial repeated `rounds` times). Each
/// round works on the smaller Gram matrix as the reference does: for X wide or square,
/// A = X Xᵀ, B = b A + c A², X ← a X + B X; for X tall the same with Xᵀ X (S1, S2: "a
/// tall matrix is transposed so that the Gram matrix is the smaller one"). In exact
/// arithmetic the result is U p_R(...p_1(S / d)...) Vᵀ for X = U S Vᵀ and d the
/// normaliser: [`Iteration::scalar`] says what it does to one singular value.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Iteration {
    /// A name for a label, after the source.
    pub name: &'static str,
    /// The polynomials, one per round (the last repeated if `rounds` is longer).
    pub polys: &'static [Odd5],
    /// How many rounds.
    pub rounds: usize,
    /// How the input is scaled first.
    pub normalise: Normalise,
}

impl Iteration {
    /// The reference's: [`Odd5::JORDAN`] for 5 rounds after X / (‖X‖_F + 1e-7) (S2 since 2
    /// Feb 2025, and the blog; from Nov 2024 to Jan 2025 the reference took 6 rounds).
    pub const JORDAN: Iteration = Iteration { name: "Jordan", polys: &[Odd5::JORDAN], rounds: 5, normalise: Normalise::REFERENCE };
    /// torch's Muon: the same quintic for 5 rounds, after X / max(‖X‖_F, 1e-7) (S21).
    pub const TORCH: Iteration = Iteration { name: "torch", polys: &[Odd5::JORDAN], rounds: 5, normalise: Normalise::TORCH };
    /// The classical cubic [`Odd5::CUBIC`], 5 rounds, after the reference's normalisation.
    pub const CUBIC: Iteration = Iteration { name: "cubic", polys: &[Odd5::CUBIC], rounds: 5, normalise: Normalise::REFERENCE };
    /// The blog's convergent baseline [`Odd5::BLOG_BASELINE`] for 12 rounds, as the
    /// speedrun's track-3 benchmark runs it ("not optimizing for wallclock speed", S3).
    pub const BLOG_BASELINE: Iteration = Iteration { name: "(2, -1.5, 0.5)", polys: &[Odd5::BLOG_BASELINE], rounds: 12, normalise: Normalise::REFERENCE };
    /// The classical quintic [`Odd5::CLASSICAL_QUINTIC`], 5 rounds.
    pub const CLASSICAL_QUINTIC: Iteration =
        Iteration { name: "classical quintic", polys: &[Odd5::CLASSICAL_QUINTIC], rounds: 5, normalise: Normalise::REFERENCE };
    /// [`YOU`]'s five polynomials after the reference's normalisation (the course notes
    /// do not say how that record normalised; their computation, `ns_more.py`, divides by
    /// the Frobenius norm alone, as here).
    pub const YOU: Iteration = Iteration { name: "You", polys: &YOU, rounds: 5, normalise: Normalise::REFERENCE };
    /// The speedrun's Polar Express ([`POLAR_EXPRESS_RECORD`]) after X / (1.02 ‖X‖_F + 1e-6).
    pub const POLAR_EXPRESS_RECORD: Iteration =
        Iteration { name: "Polar Express (speedrun)", polys: &POLAR_EXPRESS_RECORD, rounds: 5, normalise: Normalise { scale: 1.02, eps: 1e-6, clamp: false } };
    /// The paper's Polar Express ([`POLAR_EXPRESS_PAPER`]) after X / (1.01 ‖X‖_F + 1e-7).
    pub const POLAR_EXPRESS_PAPER: Iteration =
        Iteration { name: "Polar Express (paper)", polys: &POLAR_EXPRESS_PAPER, rounds: 5, normalise: Normalise { scale: 1.01, eps: 1e-7, clamp: false } };

    /// Every named iteration, in the order a control would list them.
    pub const ALL: [Iteration; 8] = [
        Iteration::JORDAN,
        Iteration::TORCH,
        Iteration::CUBIC,
        Iteration::BLOG_BASELINE,
        Iteration::CLASSICAL_QUINTIC,
        Iteration::YOU,
        Iteration::POLAR_EXPRESS_RECORD,
        Iteration::POLAR_EXPRESS_PAPER,
    ];

    /// The same with another number of rounds.
    pub const fn with_rounds(self, rounds: usize) -> Iteration {
        Iteration { rounds, ..self }
    }

    /// The polynomial of round `r` (from 0).
    pub fn poly(&self, r: usize) -> Odd5 {
        match self.polys.len() {
            0 => Odd5::new(1.0, 0.0, 0.0),
            n => self.polys[r.min(n - 1)],
        }
    }

    /// What the whole iteration does to one singular value σ of a matrix X, given as
    /// s = σ / ‖X‖_F (so s is in [0, 1]): it divides by the normalisation's `scale`
    /// (leaving out its tiny `eps`) and applies the rounds' polynomials in turn.
    pub fn scalar(&self, s: f64) -> f64 {
        let mut x = if self.normalise.clamp { s } else { s / self.normalise.scale };
        for r in 0..self.rounds {
            x = self.poly(r).at(x);
        }
        x
    }

    /// Run the iteration on the `rows` by `cols` matrix `a` in place, working in `s`
    /// (nothing is allocated once `s` has grown to the size). A zero matrix stays zero;
    /// numbers that are not finite are carried.
    pub fn run(&self, a: &mut [f64], rows: usize, cols: usize, s: &mut Scratch) {
        if !holds(a, rows, cols) {
            a.iter_mut().for_each(|x| *x = f64::NAN);
            return;
        }
        let n = rows * cols;
        let d = self.normalise.divisor(frobenius(&a[..n]));
        a[..n].iter_mut().for_each(|x| *x /= d);
        rounds(self.polys, self.rounds, a, rows, cols, s);
    }
}

/// What `rounds` rounds of the polynomial `poly` do to one number s: p(p(...p(s))). For
/// a singular value already divided by the Frobenius norm, this is what
/// [`iterate`] does to it.
pub fn iterate_scalar(poly: Odd5, s: f64, rounds: usize) -> f64 {
    let mut x = s;
    for _ in 0..rounds {
        x = poly.at(x);
    }
    x
}

/// The matrix version, in place on the `rows` by `cols` matrix `a`: normalise as the
/// reference does (X / (‖X‖_F + 1e-7)), then `count` rounds of `poly` on the smaller
/// Gram matrix (see [`Iteration`]; this is `Iteration` with a list of one polynomial).
pub fn iterate(poly: Odd5, count: usize, a: &mut [f64], rows: usize, cols: usize, s: &mut Scratch) {
    if !holds(a, rows, cols) {
        a.iter_mut().for_each(|x| *x = f64::NAN);
        return;
    }
    let n = rows * cols;
    let d = Normalise::REFERENCE.divisor(frobenius(&a[..n]));
    a[..n].iter_mut().for_each(|x| *x /= d);
    rounds(&[poly], count, a, rows, cols, s);
}

/// `count` rounds of the polynomials `polys` (round r takes `polys[r]`, or the last one)
/// on the already normalised `rows` by `cols` matrix `x`.
fn rounds(polys: &[Odd5], count: usize, x: &mut [f64], rows: usize, cols: usize, s: &mut Scratch) {
    let (n, k) = (rows * cols, rows.min(cols));
    if polys.is_empty() || n == 0 {
        return;
    }
    s.fit(k, n);
    let x = &mut x[..n];
    for r in 0..count {
        let p = polys[r.min(polys.len() - 1)];
        // A = X Xᵀ (or Xᵀ X, for a tall X: the smaller Gram matrix)
        if rows <= cols {
            gram_rows(x, rows, cols, &mut s.g);
        } else {
            gram_cols(x, rows, cols, &mut s.g);
        }
        // B = b A + c A A (A is symmetric, so A A is the Gram matrix of its rows: half
        // the products)
        if p.c != 0.0 {
            gram_rows(&s.g, k, k, &mut s.w);
            for i in 0..k * k {
                s.w[i] = p.b * s.g[i] + p.c * s.w[i];
            }
        } else {
            for i in 0..k * k {
                s.w[i] = p.b * s.g[i];
            }
        }
        // X ← a X + B X (or X B for a tall X)
        if rows <= cols {
            mul(&s.w, x, &mut s.x, k, k, cols);
        } else {
            mul(x, &s.w, &mut s.x, rows, k, k);
        }
        for i in 0..n {
            x[i] = p.a * x[i] + s.x[i];
        }
    }
}

// =====================================================================================
// The rules
// =====================================================================================

/// Which rule an [`Optimizer`] follows. Each is written as its source gives it; the
/// knobs are in [`Hyper`] and [`Hyper::default_for`] gives each rule its source's
/// defaults. Below, θ is a parameter, g its gradient (after clipping and any coupled
/// weight decay, see [`Hyper`]), lr the learning rate handed to the step, and t the
/// number of steps taken, this one included. Each says what it does with a zero
/// gradient.
///
/// The first nine treat every parameter alike. The other seven are matrix rules: they
/// act on the groups [`Hyper::groups`] chooses, each group as one matrix, and every
/// other group takes the fall-back rule [`Hyper::aux`] (as the reference Muon gives its
/// embeddings, head and biases to Adam).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Rule {
    /// Plain gradient descent: θ ← θ − lr g (PyTorch `SGD` with momentum 0). A zero
    /// gradient moves nothing.
    Sgd,
    /// Heavy-ball (Polyak) momentum, in the convention [`Hyper::convention`] names (the
    /// course notes, curvature.md A2 and A4, and Cohen et al. 2022 Table 1):
    ///
    /// - [`Convention::Sum`] (PyTorch `SGD`, the default): b ← μ b + (1 − τ) g, θ ← θ − lr
    ///   b, with b = g at the first step (PyTorch starts the buffer at the first
    ///   gradient; with the dampening τ at its default 0 that is the same as starting at
    ///   0). μ is [`Hyper::momentum`], τ [`Hyper::dampening`]. Stable on a quadratic for
    ///   lr × curvature < 2 + 2μ.
    /// - [`Convention::Ema`]: m ← μ m + (1 − μ) g, θ ← θ − lr m: the same steps as Sum at
    ///   a learning rate (1 − μ) times smaller; stable for lr × curvature < (2 + 2μ) /
    ///   (1 − μ).
    /// - [`Convention::LrInside`] (Sutskever et al.; Goyal et al. Eq. 10): v ← μ v + lr g,
    ///   θ ← θ − v: the same as Sum while lr is constant, different when it changes (and
    ///   it keeps coasting at lr = 0).
    ///
    /// With a zero gradient the buffer decays by μ per step and the parameters coast on.
    Momentum,
    /// Nesterov's momentum, in the form that needs only the gradient at the present
    /// point: with [`Convention::Sum`] (PyTorch `SGD(nesterov=True)`, TensorFlow; the
    /// course notes, curvature.md A3), b ← μ b + (1 − τ) g (b = g at the first step) and θ
    /// ← θ − lr (g + μ b); stable on a quadratic for lr × curvature < (2 + 2μ) / (1 + 2μ).
    /// With [`Convention::Ema`] (Cohen et al. 2022, EmaNesterov), m ← μ m + (1 − μ) g and
    /// θ ← θ − lr (μ m + (1 − μ) g). The sources give no learning-rate-inside form that
    /// uses only the present gradient (Sutskever's evaluates it at a point ahead), so
    /// [`Convention::LrInside`] is taken as Sum here. A zero gradient: as momentum.
    Nesterov,
    /// AdaGrad as PyTorch `Adagrad` gives it (Duchi, Hazan, Singer): s ← s + g² (entry
    /// by entry, never forgotten, starting at [`Hyper::initial_accumulator`]), θ ← θ −
    /// lr / (1 + (t − 1) η) × g / (√s + ε), η = [`Hyper::lr_decay`], ε = [`Hyper::eps`].
    /// At the first step every coordinate moves by about lr; with a steady gradient the
    /// t-th step is lr / √t. A zero gradient moves nothing and leaves s as it is (where s
    /// is 0 and ε is 0 the step is taken as 0, not 0/0).
    AdaGrad,
    /// RMSProp as PyTorch `RMSprop` gives it: v ← α v + (1 − α) g² (α = [`Hyper::alpha`]),
    /// and with [`Hyper::centered`] also ḡ ← α ḡ + (1 − α) g and v − ḡ² in place of v
    /// (negative by rounding taken as 0); then with [`Hyper::momentum`] μ > 0, b ← μ b + g
    /// / (√v + ε) and θ ← θ − lr b, else θ ← θ − lr g / (√v + ε). The root is taken
    /// before ε is added. No bias correction: the first step is lr g / (√(1 − α) |g| + ε),
    /// 10 lr in size at α = 0.99. A zero gradient moves nothing (without momentum) and
    /// lets v decay.
    RmsProp,
    /// Adam as Kingma and Ba's Algorithm 1 gives it (and PyTorch `Adam`): m ← β1 m + (1 −
    /// β1) g, v ← β2 v + (1 − β2) g², m̂ = m / (1 − β1^t), v̂ = v / (1 − β2^t), θ ← θ − lr
    /// m̂ / (√v̂ + ε), ε after the root. [`Hyper::bias_correction`] off leaves out the two
    /// divisions (the form of Cohen et al.'s Eq. 1); [`Hyper::amsgrad`] uses the running
    /// maximum of v (then corrected), as PyTorch's `amsgrad`. AdamW is Adam with
    /// [`WeightDecay::Decoupled`] ([`Hyper::adamw`]); "Adam with L2" is Adam with
    /// [`WeightDecay::Coupled`]. The first step moves every parameter by lr |g| / (|g| +
    /// ε), lr for any gradient well above ε. With a zero gradient m and v decay and the
    /// parameters coast on m̂ / √v̂ (where the denominator is exactly 0, ε being 0 and no
    /// gradient yet, the step is 0).
    Adam,
    /// Sign descent, signSGD (Bernstein et al. 2018, Alg. 1): θ ← θ − lr sign(g), the
    /// sign of 0 being 0. Steepest descent under the max-norm. A zero gradient (entry)
    /// moves nothing.
    SignSgd,
    /// Signum, sign descent with momentum (Bernstein et al. 2018, Alg. 2): m ← β m + (1 −
    /// β) g, θ ← θ − lr sign(m), β = [`Hyper::momentum`]. With a zero gradient m decays
    /// but keeps its sign: every parameter keeps moving by the full lr until m is
    /// exactly 0.
    Signum,
    /// Lion (Chen et al. 2023, Alg. 2): c ← β1 m + (1 − β1) g, θ ← θ − lr (sign(c) + λ θ),
    /// then m ← β2 m + (1 − β2) g. The decay λ θ is Lion's own, decoupled and multiplied
    /// by lr ([`WeightDecay::Decoupled`]). With a zero gradient c = β1 m: full-size steps
    /// go on while m is not 0.
    Lion,
    /// Muon and its published forms, all settings of one step on each chosen matrix
    /// (S1, S2, S7, S11, S21 and the course notes' muon.md §2, §4, §5.4, §5.13, §8.4,
    /// §8.5):
    ///
    /// 1. the matrix to orthogonalise, U: with [`OrthoInput::Momentum`] (Muon), the
    ///    momentum B ← μ B + G ([`Convention::Sum`]) or M ← μ M + (1 − μ) G
    ///    ([`Convention::Ema`], the reference's present code), and with Nesterov
    ///    ([`Hyper::nesterov`]) U = G + μ B or (1 − μ) G + μ M, else U = B or M. The two
    ///    conventions give the same step, since the orthogonalisation does not see the
    ///    scale (up to its 1e-7). With [`OrthoInput::Gradient`] (Orthogonal-SGDM, S11),
    ///    U = G and the momentum comes after;
    /// 2. its sign first, with [`Hyper::sign_first`] (AdaMuon);
    /// 3. O = the polar factor of U, by the polynomial iteration or exactly ([`Ortho`]);
    /// 4. with [`Hyper::variance`], O divided by the root of a running mean of its squares
    ///    (NorMuon per row, AdaMuon per entry);
    /// 5. times the shape factor [`Hyper::scale`];
    /// 6. θ ← θ − lr × that (Momentum input), or, with Gradient input, that fed to the
    ///    momentum of [`Hyper::convention`] and θ ← θ − lr × the momentum (or − v, lr
    ///    inside, as S11 writes it).
    ///
    /// Named forms: [`Hyper::muon_reference`] (the default), [`Hyper::muon_blog`],
    /// [`Hyper::muon_moonshot`], [`Hyper::muon_torch`], [`Hyper::orthogonal_sgdm`]. With a
    /// zero gradient the momentum decays and, since the orthogonalisation does not see
    /// the scale, full-size steps go on until it is exactly 0 (then O is 0: no step).
    Muon,
    /// NorMuon (S19, Alg. 1), Muon with a per-neuron second moment: M ← β1 M + (1 − β1) G,
    /// O = NS5(M), v ← β2 v + (1 − β2) × (the mean over each row of O ⊙ O), Ô = O with row
    /// i divided by √vᵢ + ε, and θ ← θ − lr λ θ − 0.2 lr √(mn) / ‖Ô‖_F × Ô, so that the
    /// step's RMS is 0.2 lr. [`Hyper::normuon`]. A zero gradient: as Muon (a zero Ô gives
    /// no step).
    NorMuon,
    /// AdaMuon (S16 v3, Alg. 1): M ← β M + G, O = NS(sign(M)), V ← β V + (1 − β) O ⊙ O, Ô =
    /// O ⊘ (√V + ε), θ ← θ − lr (0.2 √(mn) / ‖Ô‖_F × Ô + λ θ). [`Hyper::adamuon`]. A zero
    /// gradient: as Muon.
    AdaMuon,
    /// The step scaled by the nuclear norm: steepest descent under the spectral norm with
    /// a squared-norm penalty (Bernstein and Newhouse, Prop. 5), Δθ = −lr ‖G‖_nuc U Vᵀ. The
    /// default is Carlson, Cevher and Carin's stochastic spectral descent (S10, Alg. 1):
    /// on every matrix, θ ← θ − lr (‖G‖_nuc / (rows × cols)) U Vᵀ with an exact
    /// decomposition and no momentum; on every vector, θ ← θ − lr (‖g‖₁ / its length)
    /// sign(g) ([`AuxRule::ScaledSign`]). [`Hyper::polar_grad`] gives PolarGrad (S17 eq.
    /// 4), θ ← θ − lr ‖G‖_nuc U Vᵀ, and [`Hyper::polar_muon`] its momentum-first form.
    /// A zero gradient moves nothing: the step shrinks with the gradient ("null-gradient
    /// consistency", S17), unlike Muon's.
    SpectralDescent,
    /// Scion's constrained form (S14 §3): d ← (1 − α) d + α g (an EMA, α = 1 −
    /// [`Hyper::momentum`]), θ ← (1 − lr) θ + lr lmo(d), lmo(d) = −ρ' U Vᵀ the linear
    /// minimisation oracle over a spectral-norm ball of radius ρ' = ρ √(fan-out / fan-in)
    /// on hidden layers and ρ max(1, √(fan-out / fan-in)) on an MLP's input layer
    /// ([`Scale::Scion`]). The (1 − lr) θ is weight decay of λ = 1 ([`Hyper::scion`]); with
    /// none, it is the unconstrained uSCG ([`Hyper::scion_unconstrained`]). Scion also
    /// gives its last layer a sign oracle and a one-hot input layer a column-normalising
    /// one, but the course notes do not give their radii, so those groups (and vectors)
    /// take the fall-back. A zero gradient: as Muon (and the constrained form keeps
    /// shrinking θ).
    Scion,
    /// Shampoo exactly as Gupta, Koren and Singer's Algorithm 1 for a matrix (S8): L₀ = ε
    /// I, R₀ = ε I; L ← L + G Gᵀ, R ← R + Gᵀ G; θ ← θ − lr L^(−p) G R^(−p), p =
    /// [`Hyper::exponent`] (1/4 in the paper) and ε = [`Hyper::damping`]. The roots are
    /// taken from eigen-decompositions every step (every [`Hyper::root_every`] steps, if
    /// a scene asks for fewer), an eigenvalue at rounding level counting as zero (a
    /// pseudo-inverse root, as without damping a gradient that is not square needs). With
    /// p = 1/4, ε = 0 and one step, the step is −lr U Vᵀ, Muon's with no momentum and exact
    /// orthogonalisation (muon.md §8.2); with p = 1/2 it is −lr U S⁻¹ Vᵀ instead. A zero
    /// gradient moves nothing and leaves L and R as they are.
    Shampoo,
    /// SOAP as its authors' official code gives it (S9, `soap.py` of Dec 2024; muon.md
    /// §5.2): Adam run in the eigenbasis of Shampoo's factors. The first step only sets
    /// up L = (1 − β_s) G Gᵀ, R = (1 − β_s) Gᵀ G and their eigenvectors Q_L, Q_R, and
    /// moves nothing. Then each step: G' = Q_Lᵀ G Q_R; M ← β1 M + (1 − β1) G', V ← β2 V +
    /// (1 − β2) G'²; N = Q_L (M ⊘ (√V + ε)) Q_Rᵀ; θ ← θ − lr √(1 − β2^t) / (1 − β1^t) N
    /// (t counting from the second call), then θ ← θ − lr λ θ (after the step, with lr,
    /// not the corrected rate); L ← β_s L + (1 − β_s) G Gᵀ and R likewise; and every
    /// [`Hyper::precondition_every`] steps the eigenvectors are re-sorted by their
    /// estimated eigenvalues (V permuted to match, never rotated) and refreshed by one
    /// power-iteration step and a QR factorisation. The momentum is kept in the rotated
    /// basis and rotated exactly when the basis changes (equal in exact arithmetic to
    /// the code's rotating it out and back every step). A side longer than
    /// [`Hyper::max_precond_dim`] is not rotated; groups that are not matrices take the
    /// code's rotation-free form (its AdamW, first step skipped) at the fall-back's rate,
    /// betas, ε and decay. With a zero gradient M and V decay and the parameters coast.
    Soap,
}

impl Rule {
    /// Every rule, in the order a control would list them.
    pub const ALL: [Rule; 16] = [
        Rule::Sgd,
        Rule::Momentum,
        Rule::Nesterov,
        Rule::AdaGrad,
        Rule::RmsProp,
        Rule::Adam,
        Rule::SignSgd,
        Rule::Signum,
        Rule::Lion,
        Rule::Muon,
        Rule::NorMuon,
        Rule::AdaMuon,
        Rule::SpectralDescent,
        Rule::Scion,
        Rule::Shampoo,
        Rule::Soap,
    ];

    /// A name for a label.
    pub fn name(self) -> &'static str {
        match self {
            Rule::Sgd => "gradient descent",
            Rule::Momentum => "momentum",
            Rule::Nesterov => "Nesterov",
            Rule::AdaGrad => "AdaGrad",
            Rule::RmsProp => "RMSProp",
            Rule::Adam => "Adam",
            Rule::SignSgd => "signSGD",
            Rule::Signum => "Signum",
            Rule::Lion => "Lion",
            Rule::Muon => "Muon",
            Rule::NorMuon => "NorMuon",
            Rule::AdaMuon => "AdaMuon",
            Rule::SpectralDescent => "spectral descent",
            Rule::Scion => "Scion",
            Rule::Shampoo => "Shampoo",
            Rule::Soap => "SOAP",
        }
    }

    /// Whether it acts on matrices (and gives the other groups to a fall-back).
    pub fn is_matrix_rule(self) -> bool {
        matches!(self, Rule::Muon | Rule::NorMuon | Rule::AdaMuon | Rule::SpectralDescent | Rule::Scion | Rule::Shampoo | Rule::Soap)
    }

    /// Whether it is one of the Muon family (one step, set by [`Hyper`]: see
    /// [`Rule::Muon`]).
    fn is_muon_family(self) -> bool {
        matches!(self, Rule::Muon | Rule::NorMuon | Rule::AdaMuon | Rule::SpectralDescent | Rule::Scion)
    }
}

/// How a momentum buffer adds up the gradients (the course notes, noise.md §0 and
/// curvature.md A4; Cohen et al. 2022, Table 1). Both of the first two are in common use,
/// and a scene on momentum says which it shows.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Convention {
    /// "Standard" or "sum", learning rate outside: v ← β v + g, θ ← θ − lr v (PyTorch
    /// `SGD`, TensorFlow, Goyal et al. Eq. 9, Shallue et al.). Its effective learning rate
    /// is lr / (1 − β).
    Sum,
    /// An exponential moving average: m ← β m + (1 − β) g, θ ← θ − lr m (Adam's m, Signum,
    /// the reference Muon's present code). Effective learning rate lr. The same steps as
    /// Sum at a learning rate (1 − β) times this one.
    Ema,
    /// Learning rate inside the buffer: v ← β v + lr g, θ ← θ − v (Sutskever et al.,
    /// Goyal et al. Eq. 10, Loshchilov and Hutter's SGDW, Orthogonal-SGDM). The same as
    /// Sum for a constant lr; when lr changes, past steps keep their old rate.
    LrInside,
}

impl Convention {
    /// A name for a label.
    pub fn name(self) -> &'static str {
        match self {
            Convention::Sum => "v = βv + g",
            Convention::Ema => "v = βv + (1−β)g",
            Convention::LrInside => "v = βv + lr g",
        }
    }
}

/// Weight decay, and how it is applied (the course notes, noise.md §8 and curvature.md
/// F8). The coefficient is λ. [`Optimizer::decay`] reports what the decoupled forms took
/// off; the coupled form goes through the rule and is part of [`Optimizer::update`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum WeightDecay {
    /// None.
    None,
    /// Added to the gradient before the rule sees it, g ← g + λ θ: an L2 penalty λ/2
    /// ‖θ‖² on the loss (PyTorch's `weight_decay` for SGD, Adagrad, RMSprop and Adam;
    /// "Adam with L2 regularization" in Loshchilov and Hutter). Under Adam it is divided
    /// by √v̂ like the gradient, so it does not shrink every weight alike.
    Coupled(f64),
    /// Decoupled and multiplied by the learning rate: θ ← θ − lr λ θ whatever the
    /// gradient (PyTorch `AdamW`, the reference Muon, Moonshot, torch's Muon (with the
    /// unadjusted lr), Lion's λ, NorMuon, AdaMuon). With a schedule the decay follows it.
    Decoupled(f64),
    /// Decoupled and multiplied by the schedule's multiplier only: θ ← θ − (lr /
    /// hyper.lr) λ θ, Loshchilov and Hutter's Algorithm 2, whose decay η_t λ θ is not
    /// multiplied by the base rate α (here `Hyper::lr`). The same as `Decoupled(λ /
    /// hyper.lr)`. (With `Hyper::lr` 0 or not finite the multiplier is taken as 1.)
    Scheduled(f64),
}

/// Which groups a matrix rule acts on (the others take the fall-back, [`Hyper::aux`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Groups {
    /// The reference's choice (S1, S2): "Muon is an optimizer for 2D parameters of neural
    /// network hidden layers"; "scalar and vector parameters of the network, as well as
    /// the input and output layers, should be optimized by a standard method such as
    /// AdamW"; for transformers "AdamW should be used for the embedding and final
    /// classifier head layers". (The authors say the embedding's case follows from the
    /// modular norm and the output layer's "is instead driven by empirics".) Exactly:
    ///
    /// - an MLP: the weights ([`Kind::Weight`]) of the hidden layers ([`Part::Layer`]),
    ///   except, in a plain MLP, layer 1's, which reads the data (the input layer; a
    ///   residual MLP's input layer is its input map, [`Part::Input`]). Not the read-out,
    ///   biases or normalisation gains and biases;
    /// - a transformer: every head's query, key, value and output matrices
    ///   ([`Kind::Query`], [`Kind::Key`], [`Kind::Value`], [`Kind::HeadOut`]) and the MLP
    ///   weights ([`Kind::Weight`] in [`Part::Mlp`]). Not the token and position
    ///   embeddings, the unembedding, biases or gains. (`arch` keeps each head's matrices
    ///   apart, so each head's query is orthogonalised on its own; the reference
    ///   orthogonalises a layer's Q, K and V each as one matrix of all heads, and the
    ///   blog reports that taking Q, K and V apart helps.)
    Hidden,
    /// As [`Groups::Hidden`], and an MLP's input layer too (Scion's spectral layers for
    /// an input that is not one-hot, S14: "Spectral → Spectral → Sign").
    HiddenAndInput,
    /// Every group that is a matrix of more than one row and more than one column,
    /// whatever it is (Shampoo's Algorithm 1 is for any matrix; SOAP's code rotates every
    /// 2-D parameter).
    Matrices,
}

/// How a Muon-family rule orthogonalises.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Ortho {
    /// The polynomial iteration, as a named list of polynomials ([`Iteration`]).
    Polynomial(Iteration),
    /// The exact polar factor U Vᵀ ([`polar_into`]: over the singular values that are
    /// not zero).
    Exact,
}

/// What a Muon-family rule makes orthogonal.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum OrthoInput {
    /// The momentum (Muon: "moves the momentum to before the orthogonalization, which we
    /// find performs better empirically", S1).
    Momentum,
    /// The gradient, before it is averaged (Orthogonal-SGDM, S11 §2.2: O = U Vᵀ of the
    /// gradient, then v ← γ v + lr O, θ ← θ − v).
    Gradient,
}

/// The running second moment some Muon-family rules divide the orthogonalised matrix
/// by, with [`Hyper::beta2`] and [`Hyper::eps`]: no bias correction (their algorithms
/// have none).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Variance {
    /// None (Muon).
    None,
    /// Per row, one number per neuron (NorMuon): vᵢ ← β2 vᵢ + (1 − β2) mean_j Oᵢⱼ², row i
    /// divided by √vᵢ + ε.
    PerRow,
    /// Per entry (AdaMuon): V ← β2 V + (1 − β2) O ⊙ O, O ⊘ (√V + ε).
    PerEntry,
}

/// The factor a Muon-family rule multiplies the orthogonalised matrix by, which depends
/// on the matrix's shape: A rows (fan-out) by B columns (fan-in). The course notes,
/// muon.md §8.5, compare them: with exact orthogonalisation of a full-rank matrix, Keller's
/// gives every shape an update RMS of lr / √B, Moonshot's 0.2 lr.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Scale {
    /// 1 (the blog's algorithm box).
    One,
    /// √max(1, A / B): the reference code (S2) and torch's "original" (S21). The step's
    /// spectral norm is lr when fan-out ≤ fan-in, lr √(A / B) otherwise.
    Keller,
    /// √(A / B): Bernstein's derivation (S5, S6: the RMS→RMS dual), Dion, torch's
    /// "spectral_unclamped".
    Bernstein,
    /// k √max(A, B), k = 0.2 in the source: Moonshot (S7 eq. 4), torch's
    /// "match_rms_adamw", Kimi K2: an update RMS of k lr (S7: "AdamW's update RMS is
    /// usually around 0.2 to 0.4", so that AdamW's tuned rates can be reused).
    MatchAdamW(f64),
    /// k √(A B) / ‖D‖_F for the matrix D it multiplies, k = 0.2 in the sources (NorMuon,
    /// AdaMuon): exactly an update RMS of k lr, whatever D. (0 for a zero D.)
    Rms(f64),
    /// The nuclear norm of the matrix handed to the orthogonalisation, ‖U‖_nuc (PolarGrad,
    /// S17: tr(H) for U H = polar(G)); with the exact polar factor, the steepest-descent
    /// step of S4 Prop. 5.
    Nuclear,
    /// ‖U‖_nuc / (A B): Carlson et al.'s stochastic spectral descent (S10, Alg. 1, "‖Σ‖₁ /
    /// MJ" for an M by J weight matrix).
    NuclearPerEntry,
    /// Scion's radius (S14 §3.1): ρ √(A / B) on a hidden layer, ρ max(1, √(A / B)) on an
    /// MLP's input layer, ρ the number given.
    Scion(f64),
}

/// The fall-back rule a matrix rule's other groups take.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum AuxRule {
    /// Adam with bias correction ([`Rule::Adam`]'s formulas, Kingma and Ba's form), with
    /// the [`Aux`]'s betas, ε and decay.
    Adam,
    /// Carlson et al.'s rule for vectors (S10, Alg. 1): θ ← θ − lr (‖g‖₁ / length) sign(g),
    /// the sign scaled by the mean absolute gradient (the sharp operator of ℓ∞).
    ScaledSign,
}

/// The fall-back of a matrix rule: the rule its other groups take, and its settings.
/// Its learning rate is `lr`, scaled with the learning rate handed to the step: the
/// groups step at `lr × (step's lr) / hyper.lr`, so that at the matrix rule's own
/// [`Hyper::lr`] each group gets exactly its reference rate, and a schedule moves both
/// (with `hyper.lr` 0 or not finite, at `lr` itself). Under [`Rule::Soap`] the other
/// groups always take SOAP's own rotation-free form, with these betas, ε and decay.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Aux {
    /// Which rule.
    pub rule: AuxRule,
    /// Its learning rate (see above).
    pub lr: f64,
    /// Adam's β1.
    pub beta1: f64,
    /// Adam's β2.
    pub beta2: f64,
    /// Adam's ε.
    pub eps: f64,
    /// Its weight decay.
    pub weight_decay: WeightDecay,
}

impl Aux {
    /// The reference's auxiliary Adam (S2, `MuonWithAuxAdam`): lr 3e-4, betas (0.9, 0.95),
    /// ε 1e-10, no weight decay. (The course notes do not say whether its moments are
    /// bias-corrected; they are here, as in Kingma and Ba.)
    pub const REFERENCE: Aux = Aux { rule: AuxRule::Adam, lr: 3e-4, beta1: 0.9, beta2: 0.95, eps: 1e-10, weight_decay: WeightDecay::None };

    /// Adam with Kingma and Ba's defaults (betas (0.9, 0.999), ε 1e-8) at learning rate
    /// `lr`, no decay.
    pub const fn adam(lr: f64) -> Aux {
        Aux { rule: AuxRule::Adam, lr, beta1: 0.9, beta2: 0.999, eps: 1e-8, weight_decay: WeightDecay::None }
    }

    /// AdamW with PyTorch's defaults (betas (0.9, 0.999), ε 1e-8, decoupled decay 0.01)
    /// at learning rate `lr`.
    pub const fn adamw(lr: f64) -> Aux {
        Aux { rule: AuxRule::Adam, lr, beta1: 0.9, beta2: 0.999, eps: 1e-8, weight_decay: WeightDecay::Decoupled(0.01) }
    }
}

/// Every knob of every rule. A rule reads the knobs its documentation names and leaves
/// the others alone; [`Hyper::default_for`] gives each rule its source's defaults (and
/// says where the source gives none), and the named constructors give the published
/// forms of the matrix rules. Change any field freely before making an [`Optimizer`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Hyper {
    /// The source's default learning rate. The optimiser steps at the rate handed to
    /// [`Optimizer::step`]; this one is what a scene passes when it has no schedule, the
    /// base the fall-back's rate is scaled from ([`Aux`]), and α in
    /// [`WeightDecay::Scheduled`].
    pub lr: f64,
    /// Weight decay, coupled or decoupled (every rule).
    pub weight_decay: WeightDecay,
    /// Gradient clipping by global norm (every rule), Zhang et al.'s g ← min(1, c / ‖g‖)
    /// g with ‖g‖ the length of the whole gradient, every group's entries together,
    /// before anything else (coupled decay comes after). Infinity: none. A cap of 0 or
    /// below zeroes the gradient; a NaN cap clips nothing. (As `arch::Model::clip`.)
    pub clip: f64,
    /// The momentum coefficient (μ or β): heavy ball and Nesterov, RMSProp's momentum,
    /// Signum's β, the Muon family's momentum (for Scion, 1 − α).
    pub momentum: f64,
    /// How the momentum adds up ([`Convention`]): heavy ball, Nesterov and the Muon
    /// family.
    pub convention: Convention,
    /// PyTorch `SGD`'s dampening τ (heavy ball and Nesterov, [`Convention::Sum`]).
    pub dampening: f64,
    /// Nesterov's momentum, for the Muon family ([`Rule::Nesterov`] is always Nesterov,
    /// [`Rule::Momentum`] never).
    pub nesterov: bool,
    /// β1: Adam, Lion, SOAP.
    pub beta1: f64,
    /// β2: Adam, Lion, SOAP, and the second moment of NorMuon and AdaMuon.
    pub beta2: f64,
    /// ε: AdaGrad, RMSProp, Adam, SOAP, NorMuon, AdaMuon.
    pub eps: f64,
    /// Adam's and SOAP's bias correction.
    pub bias_correction: bool,
    /// Adam's AMSGrad (the running maximum of v).
    pub amsgrad: bool,
    /// RMSProp's α.
    pub alpha: f64,
    /// RMSProp's centred form.
    pub centered: bool,
    /// AdaGrad's learning-rate decay η.
    pub lr_decay: f64,
    /// AdaGrad's starting accumulator.
    pub initial_accumulator: f64,
    /// Which groups a matrix rule acts on.
    pub groups: Groups,
    /// What the other groups of a matrix rule take.
    pub aux: Aux,
    /// How the Muon family orthogonalises.
    pub ortho: Ortho,
    /// What the Muon family makes orthogonal.
    pub ortho_input: OrthoInput,
    /// AdaMuon's sign before the orthogonalisation.
    pub sign_first: bool,
    /// NorMuon's and AdaMuon's second moment.
    pub variance: Variance,
    /// The Muon family's shape factor.
    pub scale: Scale,
    /// Shampoo's exponent p in L^(−p) G R^(−p) (1/4 in its Algorithm 1).
    pub exponent: f64,
    /// Shampoo's damping ε (L₀ = R₀ = ε I).
    pub damping: f64,
    /// Shampoo: take the roots afresh every this many steps (1, every step, is the
    /// paper's; DistributedShampoo, which the speedrun's benchmark used, refreshes
    /// every 5 or every step; 0 is taken as 1). Between refreshes the last roots are used.
    pub root_every: u32,
    /// SOAP's `shampoo_beta` β_s for the factors (negative: use β2, the code's default).
    pub shampoo_beta: f64,
    /// SOAP's `precondition_frequency`: refresh the eigenvectors every this many steps
    /// (0 is taken as 1).
    pub precondition_every: u32,
    /// SOAP's `max_precond_dim`: a side longer than this is not rotated.
    pub max_precond_dim: usize,
    /// SOAP's `normalize_grads`: N divided by 1e-30 + its RMS.
    pub normalize_grads: bool,
}

impl Hyper {
    /// What the knobs a rule does not name are set to.
    const BASE: Hyper = Hyper {
        lr: 1e-3,
        weight_decay: WeightDecay::None,
        clip: f64::INFINITY,
        momentum: 0.0,
        convention: Convention::Sum,
        dampening: 0.0,
        nesterov: false,
        beta1: 0.9,
        beta2: 0.999,
        eps: 1e-8,
        bias_correction: true,
        amsgrad: false,
        alpha: 0.99,
        centered: false,
        lr_decay: 0.0,
        initial_accumulator: 0.0,
        groups: Groups::Hidden,
        aux: Aux::REFERENCE,
        ortho: Ortho::Polynomial(Iteration::JORDAN),
        ortho_input: OrthoInput::Momentum,
        sign_first: false,
        variance: Variance::None,
        scale: Scale::One,
        exponent: 0.25,
        damping: 1e-12,
        root_every: 1,
        shampoo_beta: -1.0,
        precondition_every: 10,
        max_precond_dim: 10_000,
        normalize_grads: false,
    };

    /// Each rule's defaults, from its source where it gives them:
    ///
    /// - gradient descent: PyTorch `SGD`, lr 0.001;
    /// - momentum and Nesterov: PyTorch `SGD`'s lr 0.001, dampening 0 and the Sum
    ///   convention; momentum 0.9 (PyTorch's default is 0, which is not momentum; 0.9 is
    ///   Goyal et al.'s and the value the course notes' stability figures use);
    /// - AdaGrad: PyTorch, lr 0.01, ε 1e-10, no decay, accumulator 0;
    /// - RMSProp: PyTorch, lr 0.01, α 0.99, ε 1e-8, no momentum, not centred;
    /// - Adam: Kingma and Ba (and PyTorch), lr 0.001, betas (0.9, 0.999), ε 1e-8;
    /// - signSGD and Signum: their paper gives no defaults; lr 0.001 is chosen, and β 0.9
    ///   for Signum (the value its Theorem 3 is illustrated with);
    /// - Lion: betas (0.9, 0.99) (S13); its paper gives lr only as "3-10x smaller than
    ///   that for AdamW": 3e-4 is chosen (AdamW's PyTorch default 1e-3 over 3), no decay;
    /// - Muon: [`Hyper::muon_reference`]; NorMuon [`Hyper::normuon`]; AdaMuon
    ///   [`Hyper::adamuon`]; spectral descent [`Hyper::spectral_descent`]; Scion
    ///   [`Hyper::scion`]; Shampoo [`Hyper::shampoo`]; SOAP [`Hyper::soap`].
    pub fn default_for(rule: Rule) -> Hyper {
        let b = Hyper::BASE;
        match rule {
            Rule::Sgd => Hyper { lr: 1e-3, ..b },
            Rule::Momentum | Rule::Nesterov => Hyper { lr: 1e-3, momentum: 0.9, convention: Convention::Sum, ..b },
            Rule::AdaGrad => Hyper { lr: 1e-2, eps: 1e-10, ..b },
            Rule::RmsProp => Hyper { lr: 1e-2, alpha: 0.99, eps: 1e-8, ..b },
            Rule::Adam => Hyper { lr: 1e-3, beta1: 0.9, beta2: 0.999, eps: 1e-8, ..b },
            Rule::SignSgd => Hyper { lr: 1e-3, ..b },
            Rule::Signum => Hyper { lr: 1e-3, momentum: 0.9, ..b },
            Rule::Lion => Hyper { lr: 3e-4, beta1: 0.9, beta2: 0.99, ..b },
            Rule::Muon => Hyper::muon_reference(),
            Rule::NorMuon => Hyper::normuon(),
            Rule::AdaMuon => Hyper::adamuon(),
            Rule::SpectralDescent => Hyper::spectral_descent(),
            Rule::Scion => Hyper::scion(),
            Rule::Shampoo => Hyper::shampoo(),
            Rule::Soap => Hyper::soap(),
        }
    }

    /// AdamW as PyTorch gives it: Adam's defaults with decoupled decay 0.01, θ ← θ − lr
    /// λ θ. (Loshchilov and Hutter's own form multiplies the decay by the schedule only:
    /// set `weight_decay` to [`WeightDecay::Scheduled`]; their λ is PyTorch's times lr.)
    pub fn adamw() -> Hyper {
        Hyper { weight_decay: WeightDecay::Decoupled(0.01), ..Hyper::default_for(Rule::Adam) }
    }

    /// Muon as the reference repository's `muon.py` has it now (S2, last changed Dec
    /// 2025): lr 0.02, momentum 0.95 in the EMA convention (`lerp`), Nesterov,
    /// [`Iteration::JORDAN`] (5 rounds of (3.4445, −4.7750, 2.0315) after X / (‖X‖_F +
    /// 1e-7)), the factor √max(1, A / B), decoupled decay 0 (θ ← (1 − lr λ) θ when set),
    /// on the hidden matrices, and the auxiliary Adam of `MuonWithAuxAdam` for the rest
    /// ([`Aux::REFERENCE`]).
    pub fn muon_reference() -> Hyper {
        Hyper {
            lr: 0.02,
            momentum: 0.95,
            convention: Convention::Ema,
            nesterov: true,
            ortho: Ortho::Polynomial(Iteration::JORDAN),
            ortho_input: OrthoInput::Momentum,
            scale: Scale::Keller,
            weight_decay: WeightDecay::None,
            groups: Groups::Hidden,
            aux: Aux::REFERENCE,
            ..Hyper::BASE
        }
    }

    /// Muon as the blog's algorithm box writes it (S1, "Algorithm 2 Muon"): B ← μ B + G
    /// (the Sum convention, B₀ = 0), O = NewtonSchulz5(B), θ ← θ − lr O: no Nesterov, no
    /// shape factor, no weight decay. The box names no values: lr 0.02 and μ 0.95 are the
    /// reference code's; the other groups as the reference.
    pub fn muon_blog() -> Hyper {
        Hyper { convention: Convention::Sum, nesterov: false, scale: Scale::One, ..Hyper::muon_reference() }
    }

    /// Muon at scale as Moonshot AI give it (S7 §2.2, eq. 4): M ← μ M + G with μ 0.95,
    /// Nesterov ("putting μ M_t + ∇L to the Newton-Schulz iteration", their footnote 1),
    /// the reference's iteration with N = 5, and θ ← θ − lr (0.2 O √max(A, B) + λ θ), "to
    /// directly reuse the learning rate and weight decay tuned for AdamW". Their tuned
    /// values are not in the course notes, so lr 0.001 and λ 0.01 are PyTorch AdamW's
    /// defaults; the other groups (RMSNorm, LM head, embeddings) AdamW with the same.
    pub fn muon_moonshot() -> Hyper {
        Hyper {
            lr: 1e-3,
            convention: Convention::Sum,
            nesterov: true,
            scale: Scale::MatchAdamW(0.2),
            weight_decay: WeightDecay::Decoupled(0.01),
            aux: Aux::adamw(1e-3),
            ..Hyper::muon_reference()
        }
    }

    /// `torch.optim.Muon` (S21): lr 0.001, decoupled decay 0.1 (applied with the
    /// unadjusted lr, as here), momentum 0.95 (`lerp`, the EMA convention), Nesterov, the
    /// reference coefficients for 5 rounds after X / max(‖X‖_F, 1e-7)
    /// ([`Iteration::TORCH`]), the "original" factor √max(1, A / B). It takes only 2-D
    /// parameters and has no fall-back of its own: the hidden matrices take it here, the
    /// rest AdamW with PyTorch's defaults.
    pub fn muon_torch() -> Hyper {
        Hyper {
            lr: 1e-3,
            ortho: Ortho::Polynomial(Iteration::TORCH),
            weight_decay: WeightDecay::Decoupled(0.1),
            aux: Aux::adamw(1e-3),
            ..Hyper::muon_reference()
        }
    }

    /// Orthogonal-SGDM (S11 §2.2), the earlier method that orthogonalises the gradient
    /// itself, exactly (by SVD), and then applies momentum with the learning rate inside:
    /// v ← γ v + lr U Vᵀ, θ ← θ − v; with γ 0.9 and lr 0.1, the SGD values its ResNet
    /// experiment was tuned with. Every matrix takes it; the rest Adam (Kingma and Ba's
    /// defaults, lr 0.001 at the base rate): the source does not say what its vectors got.
    pub fn orthogonal_sgdm() -> Hyper {
        Hyper {
            lr: 0.1,
            momentum: 0.9,
            convention: Convention::LrInside,
            nesterov: false,
            ortho: Ortho::Exact,
            ortho_input: OrthoInput::Gradient,
            scale: Scale::One,
            weight_decay: WeightDecay::None,
            groups: Groups::Matrices,
            aux: Aux::adam(1e-3),
            ..Hyper::BASE
        }
    }

    /// Stochastic spectral descent (S10, Alg. 1): on every matrix θ ← θ − lr (‖G‖_nuc /
    /// (rows × cols)) U Vᵀ, exact, no momentum; on every vector θ ← θ − lr (‖g‖₁ / length)
    /// sign(g). Its RBM experiments' step sizes are not in the course notes: lr 0.1 is
    /// chosen (the per-entry division makes its matrix steps far smaller than
    /// gradient descent's at the same lr).
    pub fn spectral_descent() -> Hyper {
        Hyper {
            lr: 0.1,
            momentum: 0.0,
            convention: Convention::Sum,
            nesterov: false,
            ortho: Ortho::Exact,
            ortho_input: OrthoInput::Momentum,
            scale: Scale::NuclearPerEntry,
            groups: Groups::Matrices,
            aux: Aux { rule: AuxRule::ScaledSign, lr: 0.1, ..Aux::adam(0.1) },
            ..Hyper::BASE
        }
    }

    /// PolarGrad (S17 eq. 4): U H = polar(G), θ ← θ − lr tr(H) U = θ − lr ‖G‖_nuc U Vᵀ, exact,
    /// no momentum, on the hidden matrices. No defaults are in the course notes: lr 0.001,
    /// and the reference's auxiliary Adam for the rest.
    pub fn polar_grad() -> Hyper {
        Hyper {
            lr: 1e-3,
            momentum: 0.0,
            convention: Convention::Sum,
            nesterov: false,
            ortho: Ortho::Exact,
            scale: Scale::Nuclear,
            groups: Groups::Hidden,
            ..Hyper::muon_reference()
        }
    }

    /// PolarMuon (S17 §3.3), PolarGrad with momentum first: M ← β M + (1 − β) G, U H =
    /// polar(M), θ ← (1 − λ lr) θ − lr tr(H) U. β 0.95 (Muon's) and λ 0 are chosen: the
    /// course notes give no values.
    pub fn polar_muon() -> Hyper {
        Hyper { momentum: 0.95, convention: Convention::Ema, ..Hyper::polar_grad() }
    }

    /// NorMuon (S19, Alg. 1): β1 0.95 (the Muon β of its experiments), the reference
    /// iteration, a per-row second moment and an update RMS of 0.2 lr, decoupled decay
    /// 0.1 on the hidden matrices (its setup); lr 4e-4 (its base rate, without the
    /// depth-μP scaling it applied). Not in the course notes, so chosen: β2 0.95, ε 1e-8,
    /// and Adam with Kingma and Ba's defaults at the same rate for the other groups.
    pub fn normuon() -> Hyper {
        Hyper {
            lr: 4e-4,
            momentum: 0.95,
            convention: Convention::Ema,
            nesterov: false,
            ortho: Ortho::Polynomial(Iteration::JORDAN),
            variance: Variance::PerRow,
            beta2: 0.95,
            eps: 1e-8,
            scale: Scale::Rms(0.2),
            weight_decay: WeightDecay::Decoupled(0.1),
            groups: Groups::Hidden,
            aux: Aux::adam(4e-4),
            ..Hyper::BASE
        }
    }

    /// AdaMuon (S16, Alg. 1): M ← β M + G, the sign before the iteration, a per-entry
    /// second moment with the same β, an update RMS of 0.2 lr. Not in the course notes,
    /// so chosen: β 0.95 (Muon's), the reference iteration (5 rounds), ε 1e-8, λ 0, and lr
    /// 0.001 (AdamW's PyTorch default, which an update RMS of 0.2 lr is meant to reuse),
    /// Adam with Kingma and Ba's defaults for the other groups.
    pub fn adamuon() -> Hyper {
        Hyper {
            lr: 1e-3,
            momentum: 0.95,
            convention: Convention::Sum,
            nesterov: false,
            sign_first: true,
            ortho: Ortho::Polynomial(Iteration::JORDAN),
            variance: Variance::PerEntry,
            beta2: 0.95,
            eps: 1e-8,
            scale: Scale::Rms(0.2),
            weight_decay: WeightDecay::None,
            groups: Groups::Hidden,
            aux: Aux::adam(1e-3),
            ..Hyper::BASE
        }
    }

    /// Scion's constrained form (S14 §3): an EMA of the gradient with α = 0.05 (momentum
    /// 0.95, the value at which S14 says Muon without Nesterov is uSCG), the exact polar
    /// factor (the oracle is an argmin), radius ρ = 1 times Scion's √(fan-out / fan-in)
    /// (max(1, ·) on an MLP's input layer), and θ ← (1 − lr) θ + lr lmo(d) (decoupled
    /// decay 1), on the hidden layers and an MLP's input layer; the reference's auxiliary
    /// Adam for the rest. Its radius and step size are not in the course notes: ρ 1 and lr
    /// 0.02 are chosen.
    pub fn scion() -> Hyper {
        Hyper {
            lr: 0.02,
            momentum: 0.95,
            convention: Convention::Ema,
            nesterov: false,
            ortho: Ortho::Exact,
            ortho_input: OrthoInput::Momentum,
            scale: Scale::Scion(1.0),
            weight_decay: WeightDecay::Decoupled(1.0),
            groups: Groups::HiddenAndInput,
            aux: Aux::REFERENCE,
            ..Hyper::BASE
        }
    }

    /// Scion's unconstrained form, uSCG (S14 §3): as [`Hyper::scion`] with no decay, θ ← θ
    /// + lr lmo(d).
    pub fn scion_unconstrained() -> Hyper {
        Hyper { weight_decay: WeightDecay::None, ..Hyper::scion() }
    }

    /// Shampoo's Algorithm 1 (S8): exponent 1/4, roots every step, on every matrix; the
    /// rest Adam with Kingma and Ba's defaults (Algorithm 1 is for matrices; vectors would
    /// need its Algorithm 2, which the course notes do not give in full). Not in the course
    /// notes, so chosen: damping 1e-12 (DistributedShampoo's default `epsilon`, §6.3) and
    /// lr 0.01 (its step is AdaGrad-like, so AdaGrad's PyTorch default).
    pub fn shampoo() -> Hyper {
        Hyper { lr: 0.01, exponent: 0.25, damping: 1e-12, root_every: 1, groups: Groups::Matrices, aux: Aux::adam(1e-3), ..Hyper::BASE }
    }

    /// SOAP's official code's defaults (S9): lr 0.003, betas (0.95, 0.95), `shampoo_beta`
    /// −1 (use β2), ε 1e-8, decoupled decay 0.01 (after the step), refresh every 10 steps,
    /// `max_precond_dim` 10000, no gradient normalisation, bias correction; every 2-D
    /// parameter rotated, the rest its rotation-free form with the same settings.
    pub fn soap() -> Hyper {
        Hyper {
            lr: 3e-3,
            beta1: 0.95,
            beta2: 0.95,
            shampoo_beta: -1.0,
            eps: 1e-8,
            weight_decay: WeightDecay::Decoupled(0.01),
            precondition_every: 10,
            max_precond_dim: 10_000,
            normalize_grads: false,
            bias_correction: true,
            groups: Groups::Matrices,
            aux: Aux { rule: AuxRule::Adam, lr: 3e-3, beta1: 0.95, beta2: 0.95, eps: 1e-8, weight_decay: WeightDecay::Decoupled(0.01) },
            ..Hyper::BASE
        }
    }
}

// ----- schedules ----------------------------------------------------------------------

/// What a learning rate does after its warm-up.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Anneal {
    /// It stays at the peak.
    Constant,
    /// Linear decay to `floor` over `steps` steps, then the floor: Shallue et al.'s η_t =
    /// η₀ − (1 − α) η₀ t / T for t ≤ T, α η₀ after (here floor = α η₀).
    Linear { floor: f64, steps: u64 },
    /// Cosine decay to `floor` over `steps` steps, then the floor: SGDR's eq. 5 within one
    /// run (Loshchilov and Hutter 2017), η_t = floor + ½ (peak − floor)(1 + cos(π t / T)),
    /// without the restarts.
    Cosine { floor: f64, steps: u64 },
}

/// A learning rate as a function of the step: an optional linear warm-up, then constant,
/// a linear decay or a cosine decay, to a floor. Step t counts from 0 (the first step's
/// rate is `at(0)`; with an optimiser, `at(opt.steps())`).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Schedule {
    /// The rate the warm-up rises to and the decay starts from.
    pub peak: f64,
    /// Steps of warm-up (0: none).
    pub warmup: u64,
    /// The rate the warm-up starts from (at step 0).
    pub warmup_from: f64,
    /// What comes after.
    pub anneal: Anneal,
}

impl Schedule {
    /// The same rate at every step.
    pub fn constant(lr: f64) -> Schedule {
        Schedule { peak: lr, warmup: 0, warmup_from: 0.0, anneal: Anneal::Constant }
    }

    /// A linear warm-up from 0 to `lr` over `steps` steps, then constant.
    pub fn warmup(lr: f64, steps: u64) -> Schedule {
        Schedule { peak: lr, warmup: steps, warmup_from: 0.0, anneal: Anneal::Constant }
    }

    /// Linear decay from `lr` to `floor` over `steps` steps, then `floor`.
    pub fn linear(lr: f64, floor: f64, steps: u64) -> Schedule {
        Schedule { peak: lr, warmup: 0, warmup_from: 0.0, anneal: Anneal::Linear { floor, steps } }
    }

    /// Cosine decay from `lr` to `floor` over `steps` steps, then `floor`.
    pub fn cosine(lr: f64, floor: f64, steps: u64) -> Schedule {
        Schedule { peak: lr, warmup: 0, warmup_from: 0.0, anneal: Anneal::Cosine { floor, steps } }
    }

    /// The same with a linear warm-up from `from` over the first `steps` steps first: Kalra
    /// and Barkeshli's η_t = η_init + (η_target − η_init) t / T_wrm (Goyal et al.'s "increment
    /// it by a constant amount at each iteration"). The decay then starts when the
    /// warm-up ends, its steps counted from there (a choice: the sources do not combine
    /// the two).
    pub fn with_warmup(self, steps: u64, from: f64) -> Schedule {
        Schedule { warmup: steps, warmup_from: from, ..self }
    }

    /// The learning rate at step `t` (from 0). A decay over 0 steps is at its floor from
    /// the start; numbers that are not finite are carried.
    pub fn at(&self, t: u64) -> f64 {
        if t < self.warmup {
            return self.warmup_from + (self.peak - self.warmup_from) * t as f64 / self.warmup as f64;
        }
        let t = t - self.warmup;
        match self.anneal {
            Anneal::Constant => self.peak,
            Anneal::Linear { floor, steps } => {
                if t >= steps {
                    floor
                } else {
                    self.peak - (self.peak - floor) * t as f64 / steps as f64
                }
            }
            Anneal::Cosine { floor, steps } => {
                if t >= steps {
                    floor
                } else {
                    floor + 0.5 * (self.peak - floor) * (1.0 + (std::f64::consts::PI * t as f64 / steps as f64).cos())
                }
            }
        }
    }
}

// ----- the optimiser ------------------------------------------------------------------

/// One of Shampoo's or SOAP's factor matrices (L or R) and what was found from it.
#[derive(Clone, Debug)]
struct Side {
    n: usize,
    /// The factor: Shampoo's sum, SOAP's moving average.
    f: Vec<f64>,
    /// Its eigenvectors, one per row (SOAP's Q, transposed).
    vt: Vec<f64>,
    /// Its eigenvalues, largest first (SOAP: the estimates used to sort, after a
    /// refresh).
    vals: Vec<f64>,
    /// Shampoo's inverse root.
    root: Vec<f64>,
    /// Whether the eigenvectors have been found once (later ones start from them).
    ready: bool,
}

impl Side {
    fn new(n: usize) -> Side {
        Side { n, f: vec![0.0; n * n], vt: vec![0.0; n * n], vals: vec![0.0; n], root: vec![0.0; n * n], ready: false }
    }

    fn reset(&mut self, damping: f64) {
        let n = self.n;
        for v in [&mut self.f, &mut self.vt, &mut self.root] {
            v.fill(0.0);
        }
        for i in 0..n {
            self.f[i * n + i] = damping;
            self.vt[i * n + i] = 1.0;
        }
        self.vals.fill(0.0);
        self.ready = false;
    }
}

/// What a scene can read of one of Shampoo's or SOAP's factors ([`Optimizer::factors`]).
#[derive(Clone, Copy, Debug)]
pub struct Factor<'a> {
    /// Its side: rows of the group for L, columns for R.
    pub n: usize,
    /// The factor matrix, n by n: Shampoo's ε I + Σ G Gᵀ (or Gᵀ G), SOAP's moving average.
    pub matrix: &'a [f64],
    /// Its eigenvectors, one per row (n by n): for Shampoo from the last time the roots
    /// were taken, for SOAP the rotation in use (SOAP's Q, transposed).
    pub eigenvectors: &'a [f64],
    /// Its eigenvalues, largest first, as last found (SOAP: the estimates it sorted by).
    pub eigenvalues: &'a [f64],
    /// Shampoo's inverse root L^(−p) (or R^(−p)); empty for SOAP.
    pub root: &'a [f64],
}

/// The state of one group.
#[derive(Clone, Debug)]
struct Slot {
    /// Whether it takes the optimiser's rule (or, for a matrix rule, its fall-back).
    main: bool,
    rows: usize,
    cols: usize,
    /// An MLP's input layer (Scion's radius).
    input_layer: bool,
    update: Vec<f64>,
    decay: Vec<f64>,
    /// The momentum, first moment or velocity.
    m: Vec<f64>,
    /// The second moment (AdaGrad's sum, RMSProp's and Adam's v, NorMuon's per-row v).
    v: Vec<f64>,
    /// A third: RMSProp's mean gradient, AMSGrad's running maximum.
    w: Vec<f64>,
    mhat: Vec<f64>,
    vhat: Vec<f64>,
    ortho_in: Vec<f64>,
    ortho_out: Vec<f64>,
    left: Option<Side>,
    right: Option<Side>,
    /// SOAP: whether the first call (which only sets up) has been made, and its own step
    /// count after it.
    started: bool,
    t: u64,
}

impl Slot {
    fn new(rule: Rule, h: &Hyper, rows: usize, cols: usize, main: bool, input_layer: bool) -> Slot {
        let n = rows * cols;
        let z = |len: usize| vec![0.0; len];
        let mut s = Slot {
            main,
            rows,
            cols,
            input_layer,
            update: z(n),
            decay: z(n),
            m: z(n),
            v: z(n),
            w: Vec::new(),
            mhat: Vec::new(),
            vhat: Vec::new(),
            ortho_in: Vec::new(),
            ortho_out: Vec::new(),
            left: None,
            right: None,
            started: false,
            t: 0,
        };
        let adam = if main { rule == Rule::Adam } else { rule != Rule::Soap && h.aux.rule == AuxRule::Adam };
        if adam {
            s.mhat = z(n);
            s.vhat = z(n);
        }
        if main && matches!(rule, Rule::RmsProp | Rule::Adam) {
            s.w = z(n);
        }
        if rule == Rule::Soap || (main && rule.is_matrix_rule()) {
            s.ortho_in = z(n);
            s.ortho_out = z(n);
        }
        if main {
            match rule {
                Rule::Shampoo => {
                    s.left = Some(Side::new(rows));
                    s.right = Some(Side::new(cols));
                }
                Rule::Soap => {
                    s.left = (rows <= h.max_precond_dim).then(|| Side::new(rows));
                    s.right = (cols <= h.max_precond_dim).then(|| Side::new(cols));
                }
                _ if rule.is_muon_family() && h.variance == Variance::PerRow => s.v = z(rows),
                _ => {}
            }
        }
        s
    }

    fn n(&self) -> usize {
        self.rows * self.cols
    }
}

/// Whether a group is an MLP's input layer: a residual MLP's input map, or a plain
/// MLP's first hidden layer (which reads the data). A transformer has none (its
/// embedding reads one-hot tokens).
fn is_input_layer(g: &Group, residual: Option<bool>) -> bool {
    match residual {
        Some(res) => g.kind == Kind::Weight && (g.part == Part::Input || (!res && g.part == Part::Layer && g.layer == 1)),
        None => false,
    }
}

/// Whether a group is a hidden matrix in the reference's sense ([`Groups::Hidden`]).
fn is_hidden(g: &Group, residual: Option<bool>) -> bool {
    match residual {
        Some(res) => g.kind == Kind::Weight && g.part == Part::Layer && (res || g.layer != 1),
        None => matches!(g.kind, Kind::Query | Kind::Key | Kind::Value | Kind::HeadOut) || (g.kind == Kind::Weight && g.part == Part::Mlp),
    }
}

/// Whether a matrix rule acts on a group.
fn takes(groups: Groups, g: &Group, residual: Option<bool>) -> bool {
    match groups {
        Groups::Hidden => is_hidden(g, residual),
        Groups::HiddenAndInput => is_hidden(g, residual) || is_input_layer(g, residual),
        Groups::Matrices => g.rows > 1 && g.cols > 1,
    }
}

/// The state of one rule on one model: every running average and factor, and what the
/// last step did, group by group (in the order of [`Model::groups`]). Made with
/// [`Optimizer::new`]; [`Optimizer::step`] takes one step. A clone is an independent
/// copy.
///
/// After a step a scene can read, per group: [`Optimizer::update`] (what the rule added
/// to the values) and [`Optimizer::decay`] (what decoupled weight decay added), whose sum
/// is the change of the values; [`Optimizer::rule_of`] (which rule the group took); the
/// momentum, Adam's m̂ and v̂, and for the matrix rules the matrix handed to the
/// orthogonalisation and what came out, and Shampoo's and SOAP's factors. For the whole
/// model, [`Optimizer::preconditioner`].
///
/// What a step costs, at each rule's defaults, measured natively in a `--profile fast`
/// build on a shared and busy server (best of five runs of 100 steps; the ignored test
/// `the_cost_of_a_step` prints them; the browser has not been measured), in
/// microseconds, for an MLP of 6 ReLU layers 32 wide (5.4 thousand parameters), a
/// transformer of 4 blocks 32 wide with 2 heads and MLPs of 64 (36 thousand), and the
/// same with 4 heads and MLPs of 128:
///
/// ```text
///                        MLP 6 x 32   transformer, MLP 64   transformer, MLP 128
///     gradient descent        6             44                    61
///     momentum, Nesterov     13             96                   134
///     AdaGrad                16            112                   162
///     RMSProp                27            197                   283
///     Adam                   25            177                   254
///     signSGD, Signum, Lion  11-17          80-125               116-185
///     Muon                  490           2170                  3080
///     NorMuon, AdaMuon      500-520       2130-2250             3310-3430
///     spectral descent,    1020           4600                  5000-5400
///       Scion (exact polar)
///     Shampoo               540           4080                 18200
///     SOAP                  390           2530                  6270
/// ```
///
/// The orthogonalisation is nearly all of a Muon-family step: 5 rounds of the reference
/// iteration on the hidden matrices alone cost 560, 2130 and 2760 microseconds (the 5
/// hidden matrices of the MLP; the 40 and 72 of the transformers), and the exact polar
/// factor 1040, 4800 and 5270. Shampoo's cost is its eigen-decompositions (a 128 by 128
/// factor for the MLPs of 128): [`Hyper::root_every`] divides it. The rules that treat
/// parameters alike cost a few nanoseconds per parameter.
#[derive(Clone, Debug)]
pub struct Optimizer {
    rule: Rule,
    hyper: Hyper,
    slots: Vec<Slot>,
    t: u64,
    grad_norm: f64,
    clip_scale: f64,
    /// The gradient a group's rule sees (clipped, with coupled decay).
    g: Vec<f64>,
    scratch: Scratch,
}

impl Optimizer {
    /// The state of `rule` with the settings `hyper` on `model` (its groups' number and
    /// shapes are fixed from now on; their values may change). A matrix rule decides here
    /// which groups it takes ([`Hyper::groups`]). Every buffer a step needs is made now.
    pub fn new(model: &Model, rule: Rule, hyper: Hyper) -> Optimizer {
        let residual = model.mlp_spec().map(|s| s.residual);
        let (mut side, mut most) = (0, 0);
        let slots: Vec<Slot> = model
            .groups()
            .iter()
            .map(|g| {
                let main = !rule.is_matrix_rule() || takes(hyper.groups, g, residual);
                most = most.max(g.rows * g.cols);
                if main && rule.is_matrix_rule() {
                    side = side.max(if matches!(rule, Rule::Shampoo | Rule::Soap) { g.rows.max(g.cols) } else { g.rows.min(g.cols) });
                }
                Slot::new(rule, &hyper, g.rows, g.cols, main, is_input_layer(g, residual))
            })
            .collect();
        let mut scratch = Scratch::new();
        scratch.fit(side, most);
        let mut opt = Optimizer { rule, hyper, slots, t: 0, grad_norm: 0.0, clip_scale: 1.0, g: vec![0.0; most], scratch };
        opt.reset();
        opt
    }

    /// Forget everything: running averages, factors and the step count, as if new.
    pub fn reset(&mut self) {
        self.t = 0;
        self.grad_norm = 0.0;
        self.clip_scale = 1.0;
        let (rule, h) = (self.rule, self.hyper);
        for s in &mut self.slots {
            for v in [&mut s.update, &mut s.decay, &mut s.m, &mut s.v, &mut s.w, &mut s.mhat, &mut s.vhat, &mut s.ortho_in, &mut s.ortho_out] {
                v.fill(0.0);
            }
            if s.main && rule == Rule::AdaGrad {
                s.v.fill(h.initial_accumulator);
            }
            let damping = if rule == Rule::Shampoo { h.damping } else { 0.0 };
            for side in [&mut s.left, &mut s.right].into_iter().flatten() {
                side.reset(damping);
            }
            s.started = false;
            s.t = 0;
        }
    }

    /// One step from the gradient the model's groups hold (the scene has put the MEAN
    /// gradient of its batch there) at learning rate `lr`: clip (if [`Hyper::clip`] says
    /// so), add coupled decay, take the rule's step on every group, and apply decoupled
    /// decay. Every value changes by `update + decay`. Nothing is allocated. A model with
    /// other groups than the optimiser was made for is a mistake: nothing happens (in a
    /// debug build, a panic).
    pub fn step(&mut self, model: &mut Model, lr: f64) {
        let Optimizer { rule, hyper, slots, t, grad_norm, clip_scale, g: gbuf, scratch } = self;
        let (rule, h) = (*rule, *hyper);
        let groups = model.groups_mut();
        let fits = groups.len() == slots.len() && groups.iter().zip(slots.iter()).all(|(g, s)| g.value.len() == s.n() && g.grad.len() == s.n());
        if !fits {
            debug_assert!(fits, "this optimiser was made for another model");
            return;
        }
        *t = t.saturating_add(1);
        let t = *t;
        let norm = groups.iter().map(|g| dot(&g.grad, &g.grad)).sum::<f64>().sqrt();
        let k = if norm > h.clip {
            if h.clip > 0.0 {
                h.clip / norm
            } else {
                0.0
            }
        } else {
            1.0
        };
        (*grad_norm, *clip_scale) = (norm, k);
        let based = h.lr.is_finite() && h.lr != 0.0;
        let sched = if based { lr / h.lr } else { 1.0 };
        let aux_lr = if based { h.aux.lr * sched } else { h.aux.lr };
        for (s, group) in slots.iter_mut().zip(groups.iter_mut()) {
            let Group { value, grad, .. } = group;
            let n = value.len();
            let (lr, wd) = if s.main { (lr, h.weight_decay) } else { (aux_lr, h.aux.weight_decay) };
            let (value, grad) = (&mut value[..n], &grad[..n]);
            let g = &mut gbuf[..n];
            let coupled = if let WeightDecay::Coupled(l) = wd { l } else { 0.0 };
            if coupled != 0.0 {
                for i in 0..n {
                    g[i] = k * grad[i] + coupled * value[i];
                }
            } else if k != 1.0 {
                for i in 0..n {
                    g[i] = k * grad[i];
                }
            } else {
                g.copy_from_slice(grad);
            }
            let g = &gbuf[..n];
            if rule == Rule::Soap {
                let (b1, b2, eps) = if s.main { (h.beta1, h.beta2, h.eps) } else { (h.aux.beta1, h.aux.beta2, h.aux.eps) };
                let run = SoapRun {
                    lr,
                    b1,
                    b2,
                    bs: if h.shampoo_beta >= 0.0 { h.shampoo_beta } else { b2 },
                    eps,
                    bias: h.bias_correction,
                    normalise: h.normalize_grads,
                    every: h.precondition_every.max(1) as u64,
                };
                let stepped = soap(&run, s, g, scratch);
                // SOAP's decay comes after its step, on the stepped values
                let lambda = match wd {
                    WeightDecay::Decoupled(l) if stepped => lr * l,
                    WeightDecay::Scheduled(l) if stepped => sched * l,
                    _ => 0.0,
                };
                let (decay, update) = (&mut s.decay[..n], &s.update[..n]);
                for i in 0..n {
                    decay[i] = if lambda == 0.0 { 0.0 } else { -lambda * (value[i] + update[i]) };
                }
            } else {
                if !s.main {
                    match h.aux.rule {
                        AuxRule::Adam => adam(s, g, lr, h.aux.beta1, h.aux.beta2, h.aux.eps, true, false, t),
                        AuxRule::ScaledSign => {
                            let mean = g.iter().map(|x| x.abs()).sum::<f64>() / n.max(1) as f64;
                            for i in 0..n {
                                s.update[i] = -lr * (mean * sgn(g[i]));
                            }
                        }
                    }
                } else if rule == Rule::Shampoo {
                    shampoo(&h, s, g, lr, t, scratch);
                } else if rule.is_muon_family() {
                    muon(&h, s, g, lr, scratch);
                } else {
                    elementwise(rule, &h, s, g, lr, t);
                }
                let lambda = match wd {
                    WeightDecay::Decoupled(l) => lr * l,
                    WeightDecay::Scheduled(l) => sched * l,
                    _ => 0.0,
                };
                let decay = &mut s.decay[..n];
                if lambda == 0.0 {
                    decay.fill(0.0);
                } else {
                    for i in 0..n {
                        decay[i] = -lambda * value[i];
                    }
                }
            }
            let (update, decay) = (&s.update[..n], &s.decay[..n]);
            for i in 0..n {
                value[i] += update[i] + decay[i];
            }
        }
    }

    /// The rule.
    pub fn rule(&self) -> Rule {
        self.rule
    }

    /// The settings.
    pub fn hyper(&self) -> &Hyper {
        &self.hyper
    }

    /// Steps taken since the start or [`Optimizer::reset`].
    pub fn steps(&self) -> u64 {
        self.t
    }

    /// The length of the whole gradient at the last step, every group's entries together,
    /// before clipping.
    pub fn grad_norm(&self) -> f64 {
        self.grad_norm
    }

    /// The factor clipping multiplied the gradient by at the last step (1: not clipped).
    pub fn clip_scale(&self) -> f64 {
        self.clip_scale
    }

    /// What the last step's rule added to group `group`'s values, weight decay apart
    /// (coupled decay went through the rule and is in here). Empty for a group that does
    /// not exist.
    pub fn update(&self, group: usize) -> &[f64] {
        self.slots.get(group).map_or(&[], |s| &s.update)
    }

    /// What decoupled weight decay added to group `group`'s values at the last step: −lr λ
    /// θ (negative where θ is positive), or under SOAP −lr λ (θ + update). So the change of
    /// the values is `update + decay`. All 0 without decoupled decay.
    pub fn decay(&self, group: usize) -> &[f64] {
        self.slots.get(group).map_or(&[], |s| &s.decay)
    }

    /// Which rule group `group` takes: the optimiser's, or for a matrix rule's other
    /// groups the fall-back's ([`Rule::Adam`]; Carlson's scaled sign is reported as
    /// [`Rule::SpectralDescent`], SOAP's rotation-free form as [`Rule::Soap`]: see
    /// [`Optimizer::as_matrix`]).
    pub fn rule_of(&self, group: usize) -> Rule {
        match self.slots.get(group) {
            Some(s) if !s.main => match (self.rule, self.hyper.aux.rule) {
                (Rule::Soap, _) => Rule::Soap,
                (_, AuxRule::Adam) => Rule::Adam,
                (_, AuxRule::ScaledSign) => Rule::SpectralDescent,
            },
            _ => self.rule,
        }
    }

    /// Whether a matrix rule treats group `group` as a matrix (false for a rule that is
    /// not a matrix rule, and for a matrix rule's other groups).
    pub fn as_matrix(&self, group: usize) -> bool {
        self.rule.is_matrix_rule() && self.slots.get(group).is_some_and(|s| s.main)
    }

    /// The momentum of group `group`: heavy ball's and Nesterov's buffer (in its
    /// convention; with the learning rate inside, the velocity), RMSProp's momentum
    /// buffer, Signum's and Lion's m, Adam's raw first moment m, the Muon family's buffer
    /// (with Gradient input, the momentum of the orthogonalised gradients), SOAP's M (in
    /// the rotated basis). Zeros for a rule without one.
    pub fn momentum(&self, group: usize) -> &[f64] {
        self.slots.get(group).map_or(&[], |s| &s.m)
    }

    /// The second moment of group `group`: AdaGrad's sum of squares, RMSProp's and Adam's
    /// raw v, SOAP's V (rotated basis), NorMuon's per-row v (one per row) and AdaMuon's V.
    pub fn second_moment(&self, group: usize) -> &[f64] {
        self.slots.get(group).map_or(&[], |s| &s.v)
    }

    /// Adam's m̂ = m / (1 − β1^t) at the last step (Adam and the Adam fall-back; empty
    /// otherwise). Without bias correction, m.
    pub fn m_hat(&self, group: usize) -> &[f64] {
        self.slots.get(group).map_or(&[], |s| &s.mhat)
    }

    /// Adam's v̂ = v / (1 − β2^t) at the last step (with AMSGrad, of the running maximum).
    pub fn v_hat(&self, group: usize) -> &[f64] {
        self.slots.get(group).map_or(&[], |s| &s.vhat)
    }

    /// The matrix the last step handed to the orthogonalisation (Muon family: U, after
    /// the sign for AdaMuon; Shampoo: G, handed to its preconditioners; SOAP: G' = Q_Lᵀ G
    /// Q_R, the gradient in the rotated basis). Empty for groups a matrix rule does not
    /// take as a matrix (SOAP's rotation-free groups: G).
    pub fn ortho_in(&self, group: usize) -> &[f64] {
        self.slots.get(group).map_or(&[], |s| &s.ortho_in)
    }

    /// What came out (Muon family: the orthogonalised O, before the second moment and the
    /// shape factor; Shampoo: L^(−p) G R^(−p); SOAP: N = Q_L (M ⊘ (√V + ε)) Q_Rᵀ, before
    /// the step size).
    pub fn ortho_out(&self, group: usize) -> &[f64] {
        self.slots.get(group).map_or(&[], |s| &s.ortho_out)
    }

    /// Shampoo's or SOAP's left and right factors of group `group` (L, rows by rows, and
    /// R, columns by columns), where it has them.
    pub fn factors(&self, group: usize) -> (Option<Factor<'_>>, Option<Factor<'_>>) {
        fn view(side: &Option<Side>, shampoo: bool) -> Option<Factor<'_>> {
            side.as_ref().map(|s| Factor { n: s.n, matrix: &s.f, eigenvectors: &s.vt, eigenvalues: &s.vals, root: if shampoo { &s.root } else { &[] } })
        }
        let shampoo = self.rule == Rule::Shampoo;
        match self.slots.get(group) {
            Some(s) => (view(&s.left, shampoo), view(&s.right, shampoo)),
            None => (None, None),
        }
    }

    /// The number each parameter's step was divided by at the last step, for a rule that
    /// divides each parameter's step by a number, in the flat order of the groups (group
    /// 0's entries, then group 1's, ...), into `out`: P in θ ← θ − lr P⁻¹ m, so that the
    /// "preconditioned sharpness" is the largest eigenvalue of P⁻¹ H (Cohen et al. 2022 §2;
    /// the course notes, curvature.md B2), and full-batch Adam with its preconditioner held
    /// fixed is stable below (2 + 2 β1) / ((1 − β1) lr) of it, 38 / lr at β1 = 0.9:
    ///
    /// - Adam: Cohen et al.'s eq. 2, P = (1 − β1^t) (√(v / (1 − β2^t)) + ε), with v the raw
    ///   second moment (AMSGrad's maximum, with that option) and the momentum m the raw
    ///   first moment (so P⁻¹ m is m̂ / (√v̂ + ε)); without bias correction, eq. 1, P = √v +
    ///   ε. (Note the factor (1 − β1^t) the corrected form carries.)
    /// - RMSProp: √v + ε (√(v − ḡ²) + ε centred), the number it divides g by.
    /// - AdaGrad: (1 + (t − 1) η)(√s + ε), its learning-rate decay included.
    ///
    /// Any other rule, any group a matrix rule's fall-back takes, and every parameter
    /// before the first step, gets NaN. Returns whether every number is a real one.
    /// Nothing is allocated once `out` has had the room.
    pub fn preconditioner(&self, out: &mut Vec<f64>) -> bool {
        out.clear();
        let h = &self.hyper;
        let t = self.t;
        let ti = t.min(i32::MAX as u64) as i32;
        let mut all = t > 0;
        for s in &self.slots {
            let n = s.n();
            match (s.main && t > 0, self.rule) {
                (true, Rule::Adam) => {
                    let v = if h.amsgrad { &s.w } else { &s.v };
                    if h.bias_correction {
                        let (c1, c2) = (1.0 - h.beta1.powi(ti), 1.0 - h.beta2.powi(ti));
                        out.extend(v.iter().map(|&v| c1 * ((v / c2).sqrt() + h.eps)));
                    } else {
                        out.extend(v.iter().map(|&v| v.sqrt() + h.eps));
                    }
                }
                (true, Rule::RmsProp) => {
                    out.extend((0..n).map(|i| root0(if h.centered { s.v[i] - s.w[i] * s.w[i] } else { s.v[i] }) + h.eps));
                }
                (true, Rule::AdaGrad) => {
                    let f = 1.0 + (t - 1) as f64 * h.lr_decay;
                    out.extend(s.v.iter().map(|&v| f * (v.sqrt() + h.eps)));
                }
                _ => {
                    all = false;
                    out.extend(std::iter::repeat_n(f64::NAN, n));
                }
            }
        }
        all
    }
}

/// The rules that treat every parameter alike, on one group: `s.update` ← the step.
fn elementwise(rule: Rule, h: &Hyper, s: &mut Slot, g: &[f64], lr: f64, t: u64) {
    let n = g.len();
    let (u, m, v) = (&mut s.update[..n], &mut s.m[..n], &mut s.v[..n]);
    match rule {
        Rule::Sgd => {
            for i in 0..n {
                u[i] = -lr * g[i];
            }
        }
        Rule::Momentum | Rule::Nesterov => {
            let nesterov = rule == Rule::Nesterov;
            let mu = h.momentum;
            let convention = if nesterov && h.convention == Convention::LrInside { Convention::Sum } else { h.convention };
            match convention {
                Convention::Sum => {
                    let keep = 1.0 - h.dampening;
                    for i in 0..n {
                        let b = if t == 1 { g[i] } else { flush(mu * m[i] + keep * g[i]) };
                        m[i] = b;
                        u[i] = -lr * if nesterov { g[i] + mu * b } else { b };
                    }
                }
                Convention::Ema => {
                    for i in 0..n {
                        let b = flush(mu * m[i] + (1.0 - mu) * g[i]);
                        m[i] = b;
                        u[i] = -lr * if nesterov { mu * b + (1.0 - mu) * g[i] } else { b };
                    }
                }
                Convention::LrInside => {
                    for i in 0..n {
                        let b = flush(mu * m[i] + lr * g[i]);
                        m[i] = b;
                        u[i] = -b;
                    }
                }
            }
        }
        Rule::AdaGrad => {
            let rate = lr / (1.0 + (t - 1) as f64 * h.lr_decay);
            for i in 0..n {
                let sum = v[i] + g[i] * g[i];
                v[i] = sum;
                let d = sum.sqrt() + h.eps;
                u[i] = if d == 0.0 { 0.0 } else { -rate * g[i] / d };
            }
        }
        Rule::RmsProp => {
            let a = h.alpha;
            let w = &mut s.w[..n];
            for i in 0..n {
                let sq = flush(a * v[i] + (1.0 - a) * g[i] * g[i]);
                v[i] = sq;
                let mut vt = sq;
                if h.centered {
                    let mean = flush(a * w[i] + (1.0 - a) * g[i]);
                    w[i] = mean;
                    vt = sq - mean * mean;
                }
                let d = root0(vt) + h.eps;
                let q = if d == 0.0 { 0.0 } else { g[i] / d };
                u[i] = if h.momentum > 0.0 {
                    let b = flush(h.momentum * m[i] + q);
                    m[i] = b;
                    -lr * b
                } else {
                    -lr * q
                };
            }
        }
        Rule::Adam => adam(s, g, lr, h.beta1, h.beta2, h.eps, h.bias_correction, h.amsgrad, t),
        Rule::SignSgd => {
            for i in 0..n {
                u[i] = -lr * sgn(g[i]);
            }
        }
        Rule::Signum => {
            let b = h.momentum;
            for i in 0..n {
                let x = flush(b * m[i] + (1.0 - b) * g[i]);
                m[i] = x;
                u[i] = -lr * sgn(x);
            }
        }
        Rule::Lion => {
            for i in 0..n {
                let c = h.beta1 * m[i] + (1.0 - h.beta1) * g[i];
                u[i] = -lr * sgn(c);
                m[i] = flush(h.beta2 * m[i] + (1.0 - h.beta2) * g[i]);
            }
        }
        _ => u.fill(0.0),
    }
}

/// Adam on one group (Kingma and Ba's Algorithm 1), with or without bias correction and
/// AMSGrad: `s.update`, `s.mhat`, `s.vhat`.
#[allow(clippy::too_many_arguments)]
fn adam(s: &mut Slot, g: &[f64], lr: f64, b1: f64, b2: f64, eps: f64, bias: bool, amsgrad: bool, t: u64) {
    let n = g.len();
    let ti = t.min(i32::MAX as u64) as i32;
    let (c1, c2) = if bias { (1.0 - b1.powi(ti), 1.0 - b2.powi(ti)) } else { (1.0, 1.0) };
    let (u, m, v, mhat, vhat) = (&mut s.update[..n], &mut s.m[..n], &mut s.v[..n], &mut s.mhat[..n], &mut s.vhat[..n]);
    if amsgrad {
        let top = &mut s.w[..n];
        for i in 0..n {
            m[i] = flush(b1 * m[i] + (1.0 - b1) * g[i]);
            v[i] = flush(b2 * v[i] + (1.0 - b2) * g[i] * g[i]);
            top[i] = if top[i] > v[i] { top[i] } else { v[i] };
            let (mh, vh) = (m[i] / c1, top[i] / c2);
            mhat[i] = mh;
            vhat[i] = vh;
            let d = vh.sqrt() + eps;
            u[i] = if d == 0.0 { 0.0 } else { -(lr * mh / d) };
        }
        return;
    }
    for i in 0..n {
        m[i] = flush(b1 * m[i] + (1.0 - b1) * g[i]);
        v[i] = flush(b2 * v[i] + (1.0 - b2) * g[i] * g[i]);
        let (mh, vh) = (m[i] / c1, v[i] / c2);
        mhat[i] = mh;
        vhat[i] = vh;
        let d = vh.sqrt() + eps;
        u[i] = if d == 0.0 { 0.0 } else { -(lr * mh / d) };
    }
}

/// The shape factor of a Muon-family step (see [`Scale`]).
fn shape_factor(scale: Scale, rows: usize, cols: usize, nuclear: f64, frob: f64, input_layer: bool) -> f64 {
    let (a, b) = (rows as f64, cols as f64);
    match scale {
        Scale::One => 1.0,
        Scale::Keller => (a / b).max(1.0).sqrt(),
        Scale::Bernstein => (a / b).sqrt(),
        Scale::MatchAdamW(k) => k * a.max(b).sqrt(),
        Scale::Rms(k) => {
            if frob == 0.0 {
                0.0
            } else {
                k * (a * b).sqrt() / frob
            }
        }
        Scale::Nuclear => nuclear,
        Scale::NuclearPerEntry => nuclear / (a * b),
        Scale::Scion(rho) => {
            if input_layer {
                rho * (a / b).sqrt().max(1.0)
            } else {
                rho * (a / b).sqrt()
            }
        }
    }
}

/// The nuclear norm of `a`, in `s`.
fn nuclear_in(a: &[f64], rows: usize, cols: usize, s: &mut Scratch) -> f64 {
    let k = gram_eigen(a, rows, cols, s);
    s.vals[..k].iter().map(|&l| root0(l)).sum()
}

/// One step of the Muon family on one matrix (see [`Rule::Muon`]).
fn muon(h: &Hyper, s: &mut Slot, g: &[f64], lr: f64, sc: &mut Scratch) {
    let (r, c) = (s.rows, s.cols);
    let n = r * c;
    let mu = h.momentum;
    let first = h.ortho_input == OrthoInput::Momentum;
    // 1. the matrix to orthogonalise
    if first {
        let ema = h.convention == Convention::Ema;
        for i in 0..n {
            let (m, u) = if ema {
                let m = flush(mu * s.m[i] + (1.0 - mu) * g[i]);
                (m, if h.nesterov { (1.0 - mu) * g[i] + mu * m } else { m })
            } else {
                let m = flush(mu * s.m[i] + g[i]);
                (m, if h.nesterov { g[i] + mu * m } else { m })
            };
            s.m[i] = m;
            s.ortho_in[i] = u;
        }
    } else {
        s.ortho_in[..n].copy_from_slice(&g[..n]);
    }
    if h.sign_first {
        s.ortho_in[..n].iter_mut().for_each(|x| *x = sgn(*x));
    }
    // 2. orthogonalise
    let needs_nuclear = matches!(h.scale, Scale::Nuclear | Scale::NuclearPerEntry);
    let nuclear = match h.ortho {
        Ortho::Exact => polar_into(&s.ortho_in, r, c, &mut s.ortho_out, sc),
        Ortho::Polynomial(it) => {
            s.ortho_out[..n].copy_from_slice(&s.ortho_in[..n]);
            it.run(&mut s.ortho_out, r, c, sc);
            if needs_nuclear {
                nuclear_in(&s.ortho_in, r, c, sc)
            } else {
                0.0
            }
        }
    };
    // 3. the second moment
    let d = &mut s.update[..n];
    d.copy_from_slice(&s.ortho_out[..n]);
    let b2 = h.beta2;
    match h.variance {
        Variance::None => {}
        Variance::PerRow => {
            for i in 0..r {
                let row = &mut d[i * c..(i + 1) * c];
                let v = flush(b2 * s.v[i] + (1.0 - b2) * dot(row, row) / c as f64);
                s.v[i] = v;
                let den = v.sqrt() + h.eps;
                row.iter_mut().for_each(|x| *x = if den == 0.0 { 0.0 } else { *x / den });
            }
        }
        Variance::PerEntry => {
            for i in 0..n {
                let v = flush(b2 * s.v[i] + (1.0 - b2) * d[i] * d[i]);
                s.v[i] = v;
                let den = v.sqrt() + h.eps;
                d[i] = if den == 0.0 { 0.0 } else { d[i] / den };
            }
        }
    }
    // 4. the shape factor, and the step
    let f = shape_factor(h.scale, r, c, nuclear, frobenius(d), s.input_layer);
    if first {
        d.iter_mut().for_each(|x| *x = -lr * (f * *x));
        return;
    }
    // momentum after the orthogonalisation (Orthogonal-SGDM)
    for i in 0..n {
        let o = f * d[i];
        d[i] = match h.convention {
            Convention::Sum => {
                let m = flush(mu * s.m[i] + o);
                s.m[i] = m;
                -lr * if h.nesterov { o + mu * m } else { m }
            }
            Convention::Ema => {
                let m = flush(mu * s.m[i] + (1.0 - mu) * o);
                s.m[i] = m;
                -lr * if h.nesterov { (1.0 - mu) * o + mu * m } else { m }
            }
            Convention::LrInside => {
                let v = flush(mu * s.m[i] + lr * o);
                s.m[i] = v;
                -v
            }
        };
    }
}

/// Take a Shampoo factor's inverse root afresh.
fn side_root(side: &mut Side, p: f64, sc: &mut Scratch) {
    let n = side.n;
    eig_into(&side.f, n, &mut side.vt, &mut side.vals, &mut sc.g, &mut sc.w, side.ready);
    root_from_eigen(&side.vt, &side.vals, n, p, &mut sc.w, &mut side.root);
    side.ready = true;
}

/// One step of Shampoo (Algorithm 1) on one matrix.
fn shampoo(h: &Hyper, s: &mut Slot, g: &[f64], lr: f64, t: u64, sc: &mut Scratch) {
    let (r, c) = (s.rows, s.cols);
    let n = r * c;
    let (Some(left), Some(right)) = (s.left.as_mut(), s.right.as_mut()) else {
        s.update.fill(0.0);
        return;
    };
    gram_rows(g, r, c, &mut sc.g);
    for i in 0..r * r {
        left.f[i] += sc.g[i];
    }
    gram_cols(g, r, c, &mut sc.g);
    for i in 0..c * c {
        right.f[i] += sc.g[i];
    }
    let every = h.root_every.max(1) as u64;
    if !left.ready || !right.ready || (t - 1).is_multiple_of(every) {
        side_root(left, h.exponent, sc);
        side_root(right, h.exponent, sc);
    }
    s.ortho_in[..n].copy_from_slice(&g[..n]);
    mul(&left.root, g, &mut sc.x, r, r, c);
    mul(&sc.x, &right.root, &mut s.ortho_out, r, c, c);
    for i in 0..n {
        s.update[i] = -lr * s.ortho_out[i];
    }
}

/// The settings of one SOAP step on one group.
struct SoapRun {
    lr: f64,
    b1: f64,
    b2: f64,
    bs: f64,
    eps: f64,
    bias: bool,
    normalise: bool,
    every: u64,
}

/// Q_Lᵀ X Q_R (the eigenbasis), for sides that are rotated, through `tmp`.
fn rotate_in(left: Option<&Side>, right: Option<&Side>, x: &[f64], out: &mut [f64], r: usize, c: usize, tmp: &mut [f64]) {
    match (left, right) {
        (Some(l), Some(q)) => {
            mul(&l.vt, x, tmp, r, r, c);
            mul_nt(tmp, &q.vt, out, r, c, c);
        }
        (Some(l), None) => mul(&l.vt, x, out, r, r, c),
        (None, Some(q)) => mul_nt(x, &q.vt, out, r, c, c),
        (None, None) => out[..r * c].copy_from_slice(&x[..r * c]),
    }
}

/// Q_L X Q_Rᵀ (back to the original basis), through `tmp`.
fn rotate_out(left: Option<&Side>, right: Option<&Side>, x: &[f64], out: &mut [f64], r: usize, c: usize, tmp: &mut [f64]) {
    match (left, right) {
        (Some(l), Some(q)) => {
            mul_tn(&l.vt, x, tmp, r, r, c);
            mul(tmp, &q.vt, out, r, c, c);
        }
        (Some(l), None) => mul_tn(&l.vt, x, out, r, r, c),
        (None, Some(q)) => mul(x, &q.vt, out, r, c, c),
        (None, None) => out[..r * c].copy_from_slice(&x[..r * c]),
    }
}

/// SOAP's first eigenvectors: those of F + 1e-30 I, largest eigenvalue first
/// (`torch.linalg.eigh`, its columns flipped).
fn soap_first_basis(side: &mut Side, sc: &mut Scratch) {
    let n = side.n;
    sc.g[..n * n].copy_from_slice(&side.f);
    for i in 0..n {
        sc.g[i * n + i] += 1e-30;
    }
    jacobi(&mut sc.g, n, &mut side.vt, &mut side.vals, false);
    side.ready = true;
}

/// SOAP's refresh of one side: sort the eigenvectors by their estimated eigenvalues
/// diag(Qᵀ F Q), largest first, permuting V along that axis to match; then Q ← QR(F Q).
fn soap_refresh(side: &mut Side, v: &mut [f64], r: usize, c: usize, is_left: bool, sc: &mut Scratch) {
    let n = side.n;
    // row k of T = vt F is (F q_k)ᵀ (F is symmetric)
    mul(&side.vt, &side.f, &mut sc.g, n, n, n);
    for k in 0..n {
        side.vals[k] = dot(&sc.g[k * n..(k + 1) * n], &side.vt[k * n..(k + 1) * n]);
    }
    for i in 0..n {
        let mut best = i;
        for j in i + 1..n {
            if side.vals[j] > side.vals[best] {
                best = j;
            }
        }
        if best != i {
            side.vals.swap(i, best);
            for k in 0..n {
                side.vt.swap(i * n + k, best * n + k);
                sc.g.swap(i * n + k, best * n + k);
            }
            if is_left {
                for k in 0..c {
                    v.swap(i * c + k, best * c + k);
                }
            } else {
                for k in 0..r {
                    v.swap(k * c + i, k * c + best);
                }
            }
        }
    }
    side.vt.copy_from_slice(&sc.g[..n * n]);
    if side.vt.iter().all(|x| x.is_finite()) {
        orthonormalise_rows(&mut side.vt, n);
    } else {
        side.vt.fill(f64::NAN);
    }
}

/// One SOAP step on one group (rotated on the sides it has). Returns whether it moved
/// the parameters (the first call only sets up).
fn soap(p: &SoapRun, s: &mut Slot, g: &[f64], sc: &mut Scratch) -> bool {
    let (r, c) = (s.rows, s.cols);
    let n = r * c;
    if !s.started {
        if let Some(side) = s.left.as_mut() {
            gram_rows(g, r, c, &mut side.f);
            side.f.iter_mut().for_each(|x| *x *= 1.0 - p.bs);
            soap_first_basis(side, sc);
        }
        if let Some(side) = s.right.as_mut() {
            gram_cols(g, r, c, &mut side.f);
            side.f.iter_mut().for_each(|x| *x *= 1.0 - p.bs);
            soap_first_basis(side, sc);
        }
        s.started = true;
        s.t = 0;
        s.update.fill(0.0);
        return false;
    }
    rotate_in(s.left.as_ref(), s.right.as_ref(), g, &mut s.ortho_in, r, c, &mut sc.x);
    s.t += 1;
    let t = s.t;
    for i in 0..n {
        let gi = s.ortho_in[i];
        let m = flush(p.b1 * s.m[i] + (1.0 - p.b1) * gi);
        let v = flush(p.b2 * s.v[i] + (1.0 - p.b2) * gi * gi);
        s.m[i] = m;
        s.v[i] = v;
        sc.y[i] = m / (v.sqrt() + p.eps);
    }
    rotate_out(s.left.as_ref(), s.right.as_ref(), &sc.y, &mut s.ortho_out, r, c, &mut sc.x);
    if p.normalise {
        let rms = (dot(&s.ortho_out[..n], &s.ortho_out[..n]) / n.max(1) as f64).sqrt();
        s.ortho_out[..n].iter_mut().for_each(|x| *x /= 1e-30 + rms);
    }
    let ti = t.min(i32::MAX as u64) as i32;
    let alpha = if p.bias { p.lr * (1.0 - p.b2.powi(ti)).sqrt() / (1.0 - p.b1.powi(ti)) } else { p.lr };
    for i in 0..n {
        s.update[i] = -alpha * s.ortho_out[i];
    }
    // the factors, from the raw gradient
    if let Some(side) = s.left.as_mut() {
        gram_rows(g, r, c, &mut sc.g);
        for i in 0..r * r {
            side.f[i] = p.bs * side.f[i] + (1.0 - p.bs) * sc.g[i];
        }
    }
    if let Some(side) = s.right.as_mut() {
        gram_cols(g, r, c, &mut sc.g);
        for i in 0..c * c {
            side.f[i] = p.bs * side.f[i] + (1.0 - p.bs) * sc.g[i];
        }
    }
    if t.is_multiple_of(p.every) && (s.left.is_some() || s.right.is_some()) {
        // the momentum to the original basis, a new basis, and back
        rotate_out(s.left.as_ref(), s.right.as_ref(), &s.m, &mut sc.z, r, c, &mut sc.x);
        if let Some(side) = s.left.as_mut() {
            soap_refresh(side, &mut s.v, r, c, true, sc);
        }
        if let Some(side) = s.right.as_mut() {
            soap_refresh(side, &mut s.v, r, c, false, sc);
        }
        rotate_in(s.left.as_ref(), s.right.as_ref(), &sc.z, &mut s.m, r, c, &mut sc.x);
    }
    true
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::math::arch::{repeats, spirals, Init, Loss, MlpSpec, Norm, TransformerSpec};
    use crate::math::autograd::{Act, Tape};
    use crate::math::linalg;
    use crate::math::rng::Rng;

    fn randn(n: usize, rng: &mut Rng) -> Vec<f64> {
        (0..n).map(|_| rng.normal() as f64).collect()
    }

    /// A B, written out.
    fn naive(a: &[f64], b: &[f64], m: usize, k: usize, n: usize) -> Vec<f64> {
        let mut out = vec![0.0; m * n];
        for i in 0..m {
            for j in 0..n {
                out[i * n + j] = (0..k).map(|p| a[i * k + p] * b[p * n + j]).sum();
            }
        }
        out
    }

    fn tr(a: &[f64], m: usize, n: usize) -> Vec<f64> {
        (0..n * m).map(|k| a[(k % m) * n + k / m]).collect()
    }

    fn assert_close(a: &[f64], b: &[f64], tol: f64, what: &str) {
        assert_eq!(a.len(), b.len(), "{what}: lengths");
        for (i, (x, y)) in a.iter().zip(b).enumerate() {
            assert!((x - y).abs() <= tol * y.abs().max(1.0), "{what}: entry {i}: {x} against {y}");
        }
    }

    /// A rows by cols matrix with the given singular values (min(rows, cols) of them) and
    /// random singular vectors: (A, U rows by k, V cols by k).
    fn with_singular_values(rows: usize, cols: usize, s: &[f64], rng: &mut Rng) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
        let k = rows.min(cols);
        let mut u = randn(rows * k, rng);
        linalg::orthonormalise(&mut u, rows, k);
        let mut v = randn(cols * k, rng);
        linalg::orthonormalise(&mut v, cols, k);
        let mut a = vec![0.0; rows * cols];
        for i in 0..rows {
            for j in 0..cols {
                a[i * cols + j] = (0..k).map(|p| u[i * k + p] * s[p] * v[j * k + p]).sum();
            }
        }
        (a, u, v)
    }

    /// U Vᵀ over the nonzero singular values, from `linalg::svd`.
    fn svd_polar(a: &[f64], rows: usize, cols: usize) -> Vec<f64> {
        let (u, s, v) = linalg::svd(a, rows, cols);
        let k = rows.min(cols);
        let mut out = vec![0.0; rows * cols];
        for p in 0..k {
            if s[p] > 1e-12 * s[0] {
                for i in 0..rows {
                    for j in 0..cols {
                        out[i * cols + j] += u[i * k + p] * v[j * k + p];
                    }
                }
            }
        }
        out
    }

    /// (A + damping I)^(−p) from `linalg::eig_sym`, over eigenvalues above 1e-12 of the
    /// largest.
    fn eig_root(a: &[f64], n: usize, p: f64, damping: f64) -> Vec<f64> {
        let mut m = a.to_vec();
        (0..n).for_each(|i| m[i * n + i] += damping);
        let (vals, vecs) = linalg::eig_sym(&m, n);
        let mut out = vec![0.0; n * n];
        for k in 0..n {
            if vals[k] > 1e-12 * vals[0] {
                let f = vals[k].powf(-p);
                for i in 0..n {
                    for j in 0..n {
                        out[i * n + j] += f * vecs[k * n + i] * vecs[k * n + j];
                    }
                }
            }
        }
        out
    }

    /// The reference's Newton–Schulz code, written out as S2 has it: transpose a tall
    /// matrix, divide by scale ‖X‖_F + eps (or by max(‖X‖_F, eps)), then A = X Xᵀ, B = b A
    /// + c A A, X = a X + B X for each round's triple, and transpose back.
    #[allow(clippy::too_many_arguments)]
    fn reference_ns(g: &[f64], rows: usize, cols: usize, polys: &[Odd5], rounds: usize, scale: f64, eps: f64, clamp: bool) -> Vec<f64> {
        let tall = rows > cols;
        let (mut x, m, n) = if tall { (tr(g, rows, cols), cols, rows) } else { (g.to_vec(), rows, cols) };
        let norm = x.iter().map(|v| v * v).sum::<f64>().sqrt();
        let d = if clamp { norm.max(eps) } else { scale * norm + eps };
        x.iter_mut().for_each(|v| *v /= d);
        for r in 0..rounds {
            let p = polys[r.min(polys.len() - 1)];
            let a = naive(&x, &tr(&x, m, n), m, n, m);
            let aa = naive(&a, &a, m, m, m);
            let b: Vec<f64> = (0..m * m).map(|i| p.b * a[i] + p.c * aa[i]).collect();
            let bx = naive(&b, &x, m, m, n);
            x = (0..m * n).map(|i| p.a * x[i] + bx[i]).collect();
        }
        if tall {
            tr(&x, m, n)
        } else {
            x
        }
    }

    // ----- the matrix functions --------------------------------------------------------

    /// The eigen-solver agrees with `linalg::eig_sym` on symmetric matrices of every size
    /// here, on positive semi-definite ones that have lost rank, on zero and on NaN; and a
    /// warm start gives the same decomposition.
    #[test]
    fn the_eigen_solver_agrees_with_linalg() {
        let mut rng = Rng::new(1);
        for n in [1, 2, 3, 5, 8, 17, 32] {
            for case in 0..3 {
                let a: Vec<f64> = match case {
                    0 => {
                        let b = randn(n * n, &mut rng);
                        (0..n * n).map(|k| 0.5 * (b[k] + b[(k % n) * n + k / n])).collect()
                    }
                    1 => {
                        let g = randn(n * 2, &mut rng);
                        naive(&g, &tr(&g, n, 2), n, 2, n)
                    }
                    _ => {
                        let g = randn(n * n, &mut rng);
                        naive(&g, &tr(&g, n, n), n, n, n)
                    }
                };
                let (vals, vecs) = sym_eigen(&a, n);
                let (lvals, _) = linalg::eig_sym(&a, n);
                let size = max_abs(&vals).max(1e-300);
                assert_close(&vals, &lvals, 1e-13 * size, &format!("eigenvalues, n {n}, case {case}"));
                for k in 0..n {
                    let v = &vecs[k * n..(k + 1) * n];
                    let av = naive(&a, v, n, n, 1);
                    let resid: f64 = (0..n).map(|i| (av[i] - vals[k] * v[i]).powi(2)).sum::<f64>().sqrt();
                    assert!(resid < 1e-13 * size * n as f64, "n {n} case {case}: A v - λ v = {resid}");
                    for j in 0..n {
                        let d = dot(v, &vecs[j * n..(j + 1) * n]);
                        assert!((d - if j == k { 1.0 } else { 0.0 }).abs() < 1e-13, "n {n}: not orthonormal ({d})");
                    }
                }
                // a warm start from a nearby basis gives the same eigenvalues
                let mut b = a.clone();
                (0..n * n).for_each(|k| b[k] += 1e-3 * a[(k % n) * n + k / n]);
                let (mut vt, mut vals2, mut w1, mut w2) = (vecs.clone(), vec![0.0; n], vec![0.0; n * n], vec![0.0; n * n]);
                eig_into(&b, n, &mut vt, &mut vals2, &mut w1, &mut w2, true);
                let (cold, _) = sym_eigen(&b, n);
                assert_close(&vals2, &cold, 1e-13 * size, "warm start");
            }
        }
        let (vals, vecs) = sym_eigen(&[0.0; 9], 3);
        assert_eq!(vals, vec![0.0; 3]);
        assert!(vecs.iter().all(|x| x.is_finite()));
        let (vals, _) = sym_eigen(&[1.0, f64::NAN, f64::NAN, 1.0], 2);
        assert!(vals.iter().all(|x| x.is_nan()));
        assert_eq!(sym_eigen(&[1.0], 4).0.len(), 4);
    }

    /// Singular values agree with `linalg::svd`; the norms are what they say.
    #[test]
    fn singular_values_and_norms() {
        let mut rng = Rng::new(2);
        for (rows, cols) in [(1, 1), (1, 5), (5, 1), (3, 7), (7, 3), (16, 16), (32, 12)] {
            let a = randn(rows * cols, &mut rng);
            let s = singular_values(&a, rows, cols);
            let (_, ls, _) = linalg::svd(&a, rows, cols);
            assert_close(&s, &ls[..rows.min(cols)], 1e-12 * ls[0], &format!("{rows} by {cols}"));
            let frob = frobenius(&a);
            assert!((frob - s.iter().map(|x| x * x).sum::<f64>().sqrt()).abs() < 1e-12 * frob);
            assert!((spectral_norm(&a, rows, cols) - s[0]).abs() < 1e-12 * s[0]);
            assert!((nuclear_norm(&a, rows, cols) - s.iter().sum::<f64>()).abs() < 1e-12 * s[0] * s.len() as f64);
            assert!((stable_rank(&a, rows, cols) - frob * frob / (s[0] * s[0])).abs() < 1e-10);
            assert_eq!(max_abs(&a), a.iter().fold(0.0f64, |m, x| m.max(x.abs())));
        }
        // a matrix that has lost rank, which linalg::svd is slow on: rank 2 of 24 by 24
        let (a, _, _) = with_singular_values(24, 24, &[&[3.0, 0.5][..], &[0.0; 22][..]].concat(), &mut rng);
        let s = singular_values(&a, 24, 24);
        assert!((s[0] - 3.0).abs() < 1e-12 && (s[1] - 0.5).abs() < 1e-12 && s[2] < 1e-6, "{s:?}");
        assert!((stable_rank(&a, 24, 24) - (9.0 + 0.25) / 9.0).abs() < 1e-12);
        assert_eq!(singular_values(&[0.0; 6], 2, 3), vec![0.0, 0.0]);
        assert_eq!(stable_rank(&[0.0; 6], 2, 3), 0.0);
        assert!(max_abs(&[1.0, f64::NAN]).is_nan());
        assert!(singular_values(&[1.0, 2.0], 2, 3).iter().all(|x| x.is_nan()));
    }

    /// The polar factor has orthonormal columns (tall) or rows (wide); A is the polar
    /// factor times a symmetric positive semi-definite matrix; ⟨A, polar⟩ is the nuclear
    /// norm; it agrees with U Vᵀ from `linalg::svd`; a rank-deficient matrix gives U_r V_rᵀ;
    /// zero gives zero, a 1 by 1 its sign, NaN NaN.
    #[test]
    fn the_polar_factor_is_orthonormal_and_a_is_polar_times_psd() {
        let mut rng = Rng::new(3);
        for (rows, cols) in [(1, 1), (1, 6), (6, 1), (4, 9), (9, 4), (12, 12), (40, 7)] {
            let a = randn(rows * cols, &mut rng);
            let mut p = vec![0.0; rows * cols];
            let nuclear = polar_into(&a, rows, cols, &mut p, &mut Scratch::new());
            assert_close(&p, &svd_polar(&a, rows, cols), 1e-11, &format!("polar of {rows} by {cols}"));
            // orthonormal columns (tall) or rows (wide)
            let k = rows.min(cols);
            let gram = if rows >= cols { naive(&tr(&p, rows, cols), &p, cols, rows, cols) } else { naive(&p, &tr(&p, rows, cols), rows, cols, rows) };
            let eye: Vec<f64> = (0..k * k).map(|i| if i % (k + 1) == 0 { 1.0 } else { 0.0 }).collect();
            assert_close(&gram, &eye, 1e-12, "orthonormal");
            // A = P H, H = Pᵀ A symmetric positive semi-definite
            let h = naive(&tr(&p, rows, cols), &a, cols, rows, cols);
            assert_close(&h, &tr(&h, cols, cols), 1e-12, "H symmetric");
            let (hv, _) = linalg::eig_sym(&h, cols);
            assert!(hv.iter().all(|&l| l > -1e-12), "H positive semi-definite: {hv:?}");
            assert_close(&naive(&p, &h, rows, cols, cols), &a, 1e-12, "A = P H");
            // ⟨A, P⟩ is the nuclear norm
            let inner = dot(&a, &p);
            let nuc = nuclear_norm(&a, rows, cols);
            assert!((inner - nuc).abs() < 1e-12 * nuc && (nuclear - nuc).abs() < 1e-12 * nuc, "{inner} {nuclear} {nuc}");
        }
        // rank 3 of 10 by 8: U_3 V_3ᵀ, three singular values of 1, the rest 0
        let (a, u, v) = with_singular_values(10, 8, &[2.0, 1.0, 0.25, 0.0, 0.0, 0.0, 0.0, 0.0], &mut rng);
        let p = polar_factor(&a, 10, 8);
        let mut want = vec![0.0; 80];
        for i in 0..10 {
            for j in 0..8 {
                want[i * 8 + j] = (0..3).map(|q| u[i * 8 + q] * v[j * 8 + q]).sum();
            }
        }
        assert_close(&p, &want, 1e-10, "U_r V_rᵀ");
        let s = singular_values(&p, 10, 8);
        assert!(s[..3].iter().all(|x| (x - 1.0).abs() < 1e-10) && s[3..].iter().all(|&x| x < 1e-6), "{s:?}");
        assert_eq!(polar_factor(&[0.0; 6], 2, 3), vec![0.0; 6]);
        assert_eq!(polar_factor(&[-0.3], 1, 1), vec![-1.0]);
        assert!(polar_factor(&[1.0, f64::NAN, 0.0, 1.0], 2, 2).iter().all(|x| x.is_nan()));
        assert!(polar_factor(&[1.0, f64::INFINITY, 0.0, 1.0], 2, 2).iter().all(|x| x.is_nan()));
        // rank 1
        let p = polar_factor(&naive(&[1.0, 2.0, 2.0], &[3.0, 0.0, 4.0, 0.0], 3, 1, 4), 3, 4);
        let want: Vec<f64> = naive(&[1.0 / 3.0, 2.0 / 3.0, 2.0 / 3.0], &[0.6, 0.0, 0.8, 0.0], 3, 1, 4);
        assert_close(&p, &want, 1e-12, "rank 1");
        assert_eq!(sign(&[2.0, -0.5, 0.0, -0.0]), vec![1.0, -1.0, 0.0, 0.0]);
        assert!(sign(&[f64::NAN])[0].is_nan());
    }

    /// The matrix iteration keeps the singular vectors and applies to each singular value
    /// what `iterate_scalar` (or `Iteration::scalar`) predicts, for every named iteration
    /// and both orientations.
    #[test]
    fn the_matrix_iteration_acts_on_singular_values_as_the_scalar_one() {
        let mut rng = Rng::new(4);
        for (rows, cols) in [(6, 15), (15, 6), (9, 9)] {
            let k = rows.min(cols);
            let s: Vec<f64> = (0..k).map(|i| 2.0 * 0.6f64.powi(i as i32)).collect();
            let (a, u, v) = with_singular_values(rows, cols, &s, &mut rng);
            let frob = frobenius(&a);
            // iterate(): the reference normalisation, then rounds of one polynomial
            for poly in [Odd5::JORDAN, Odd5::CUBIC, Odd5::BLOG_BASELINE, Odd5::CLASSICAL_QUINTIC] {
                for count in [1, 3, 5, 8] {
                    let mut x = a.clone();
                    iterate(poly, count, &mut x, rows, cols, &mut Scratch::new());
                    let want: Vec<f64> = s.iter().map(|&si| iterate_scalar(poly, si / (frob + 1e-7), count)).collect();
                    let mut expect = vec![0.0; rows * cols];
                    for i in 0..rows {
                        for j in 0..cols {
                            expect[i * cols + j] = (0..k).map(|p| u[i * k + p] * want[p] * v[j * k + p]).sum();
                        }
                    }
                    assert_close(&x, &expect, 1e-10, &format!("{poly:?} x {count} on {rows} by {cols}"));
                }
            }
            for it in Iteration::ALL {
                let mut x = a.clone();
                it.run(&mut x, rows, cols, &mut Scratch::new());
                let got = singular_values(&x, rows, cols);
                // (scalar() leaves out the normaliser's tiny eps; here it is put in)
                let d = if it.normalise.clamp { frob.max(it.normalise.eps) } else { it.normalise.scale * frob + it.normalise.eps };
                let mut want: Vec<f64> = s.iter().map(|&si| (0..it.rounds).fold(si / d, |x, r| it.poly(r).at(x))).collect();
                want.sort_by(|a, b| b.total_cmp(a));
                assert_close(&got, &want, 1e-10, it.name);
                let approx: Vec<f64> = s.iter().map(|&si| it.scalar(si / frob)).collect();
                assert!(approx.iter().zip(&s).all(|(a, &si)| (a - (0..it.rounds).fold(si / d, |x, r| it.poly(r).at(x))).abs() < 1e-4));
                // and the singular vectors are kept: Uᵀ X V is diagonal
                let d = naive(&naive(&tr(&u, rows, k), &x, k, rows, cols), &v, k, cols, k);
                for i in 0..k {
                    for j in 0..k {
                        if i != j {
                            assert!(d[i * k + j].abs() < 1e-10, "{}: off-diagonal {}", it.name, d[i * k + j]);
                        }
                    }
                }
                // the reference's code written out gives the same matrix
                let n = it.normalise;
                let want = reference_ns(&a, rows, cols, it.polys, it.rounds, n.scale, n.eps, n.clamp);
                assert_close(&x, &want, 1e-11, &format!("{} against the reference code", it.name));
            }
        }
        // zero stays zero; NaN is carried; a 1 by 1 matrix goes where the scalar goes
        let mut z = vec![0.0; 6];
        Iteration::JORDAN.run(&mut z, 2, 3, &mut Scratch::new());
        assert_eq!(z, vec![0.0; 6]);
        let mut n = vec![1.0, f64::NAN, 0.0, 1.0];
        Iteration::JORDAN.run(&mut n, 2, 2, &mut Scratch::new());
        assert!(n.iter().all(|x| x.is_nan()));
        let mut one = vec![-2.0];
        Iteration::JORDAN.run(&mut one, 1, 1, &mut Scratch::new());
        assert!((one[0] + Iteration::JORDAN.scalar(2.0 / (2.0 + 1e-7))).abs() < 1e-14);
    }

    /// Where the reference polynomial sends [0, 1] after 1 to 8 rounds (what the scenes
    /// quote), against the course notes' computed table (muon.md §8.3, `ns_table.out`,
    /// `ns_more.out`); and its fixed points and largest value.
    #[test]
    // (the notes' table has a 0.7071 in it that is not 1/√2)
    #[allow(clippy::approx_constant)]
    fn where_the_reference_polynomial_sends_the_unit_interval() {
        let p = Odd5::JORDAN;
        // the notes' grid: 50001 points log-spaced over [1e-6, 0.1], 90001 even over [0.1, 1]
        let mut grid: Vec<f64> = (0..50001).map(|i| 10f64.powf(-6.0 + 5.0 * i as f64 / 50000.0)).collect();
        grid.extend((0..90001).map(|i| 0.1 + 0.9 * i as f64 / 90000.0));
        grid.push(0.0);
        let starts = [0.0, 1e-4, 1e-3, 3e-3, 1e-2, 3e-2, 0.1];
        let mut x = grid.clone();
        println!("Jordan quintic (3.4445, -4.7750, 2.0315): the image of [s0, 1] after k rounds");
        println!("k  | {}", starts.iter().map(|s| format!("s0 = {s:<6}     ")).collect::<String>());
        let mut table = Vec::new();
        for k in 1..=8 {
            x.iter_mut().for_each(|v| *v = p.at(*v));
            let row: Vec<(f64, f64)> = starts
                .iter()
                .map(|&s0| {
                    let pick = x.iter().zip(&grid).filter(|(_, &s)| s >= s0).map(|(v, _)| *v);
                    pick.fold((f64::INFINITY, f64::NEG_INFINITY), |(lo, hi), v| (lo.min(v), hi.max(v)))
                })
                .collect();
            println!("{k}  | {}", row.iter().map(|(lo, hi)| format!("[{lo:.4}, {hi:.4}]  ")).collect::<String>());
            table.push(row);
        }
        let near = |got: f64, want: f64, tol: f64, what: &str| assert!((got - want).abs() <= tol, "{what}: {got} against {want}");
        // the notes: max over (0, 1] 1.2024 for k = 1 to 8; over [0.1, 1] min 0.3397 (k=1),
        // then 0.6818; max 1.2024 (k = 1, 2), then 1.1344
        for k in 0..8 {
            near(table[k][0].1, 1.2024, 1e-4, "largest");
            near(table[k][6].0, if k == 0 { 0.3397 } else { 0.6818 }, 1e-4, "min over [0.1, 1]");
            near(table[k][6].1, if k < 2 { 1.2024 } else { 1.1344 }, 1e-4, "max over [0.1, 1]");
            assert_eq!(table[k][0].0, 0.0);
        }
        // min over [1e-3, 1]: the value at 1e-3 (the notes' grid steps past it by 5e-6)
        for (k, want) in [(0, 0.0034), (1, 0.0119), (2, 0.0409), (3, 0.1404), (4, 0.4705), (5, 0.6818), (7, 0.6818)] {
            near(table[k][2].0, want, 1e-4, &format!("min over [1e-3, 1] after {}", k + 1));
        }
        // after 5 rounds (ns_more.out): [0.048, 1.202], [0.471, 1.202], [0.682, 1.202],
        // [0.682, 1.134], [0.682, 1.134], [0.682, 1.134] from s0 = 1e-4, ..., 0.1
        for (j, (lo, hi)) in [(1, (0.048, 1.202)), (2, (0.471, 1.202)), (3, (0.682, 1.202)), (4, (0.682, 1.134)), (5, (0.682, 1.134)), (6, (0.682, 1.134))] {
            near(table[4][j].0, lo, 6e-4, "5 rounds, low");
            near(table[4][j].1, hi, 6e-4, "5 rounds, high");
        }
        // the notes' table of iterates (4 decimals)
        let rows: [(f64, [f64; 5]); 6] = [
            (0.001, [0.0034, 0.0119, 0.0409, 0.1404, 0.4705]),
            (0.01, [0.0344, 0.1184, 0.4001, 1.0931, 0.6989]),
            (0.1, [0.3397, 0.9921, 0.7071, 1.1066, 0.7121]),
            (0.5, [1.1889, 0.8962, 0.8244, 0.9379, 0.7654]),
            (0.9, [0.8187, 0.9470, 0.7539, 1.0455, 0.6820]),
            (1.0, [0.7010, 1.1136, 0.7207, 1.0900, 0.6964]),
        ];
        for (s, want) in rows {
            for (k, w) in want.iter().enumerate() {
                near(iterate_scalar(p, s, k + 1), *w, 5.1e-5, &format!("phi^{}({s})", k + 1));
            }
        }
        // the cubic: 0.01 is still 0.0758 after 5 rounds; the blog's baseline sends 0.01 to 0.3120
        near(iterate_scalar(Odd5::CUBIC, 0.01, 5), 0.0758, 5.1e-5, "cubic");
        near(iterate_scalar(Odd5::BLOG_BASELINE, 0.01, 5), 0.3120, 5.1e-5, "baseline");
        // fixed points 0.8680 (slope -1.58) and 1.2637 (slope 6.47); phi(1) = 0.701
        for (x0, slope) in [(0.8680258, -1.5824), (1.26372907, 6.4735)] {
            near(p.at(x0), x0, 1e-6, "fixed point");
            let d = (p.at(x0 + 1e-7) - p.at(x0 - 1e-7)) / 2e-7;
            near(d, slope, 1e-3, "slope at the fixed point");
        }
        near(p.at(1.0), 0.701, 1e-12, "phi(1)");
    }

    /// Shampoo after one step with no damping is the polar factor of a full-rank G:
    /// L^(−1/4) G R^(−1/4) with L = G Gᵀ, R = Gᵀ G; with the exponent 1/2 it is U S⁻¹ Vᵀ
    /// instead. The same through the optimiser.
    #[test]
    fn shampoo_without_accumulation_is_the_polar_factor() {
        let mut rng = Rng::new(5);
        for (rows, cols) in [(5, 5), (4, 9), (9, 4), (1, 3)] {
            let g = randn(rows * cols, &mut rng);
            let l = naive(&g, &tr(&g, rows, cols), rows, cols, rows);
            let r = naive(&tr(&g, rows, cols), &g, cols, rows, cols);
            let quarter = |p: f64| {
                let left = inverse_root(&l, rows, p, 0.0);
                let right = inverse_root(&r, cols, p, 0.0);
                naive(&naive(&left, &g, rows, rows, cols), &right, rows, cols, cols)
            };
            assert_close(&quarter(0.25), &polar_factor(&g, rows, cols), 1e-9, &format!("{rows} by {cols}"));
            // the exponent 1/2: U S⁻¹ Vᵀ, singular values the reciprocals of G's
            let half = quarter(0.5);
            let mut want: Vec<f64> = singular_values(&g, rows, cols).iter().map(|s| 1.0 / s).collect();
            want.sort_by(|a, b| b.total_cmp(a));
            assert_close(&singular_values(&half, rows, cols), &want, 1e-8, "exponent 1/2");
            // with damping, against linalg's eigen-decomposition
            assert_close(&inverse_root(&l, rows, 0.25, 1e-3), &eig_root(&l, rows, 0.25, 1e-3), 1e-11, "damped root");
        }
        // through the optimiser: one step of Shampoo with no damping moves by −lr U Vᵀ
        let (mut model, _, _) = bench(9);
        let mut opt = Optimizer::new(&model, Rule::Shampoo, Hyper { damping: 0.0, ..Hyper::shampoo() });
        model.groups_mut()[2].grad = randn(25, &mut rng);
        let g = model.groups()[2].grad.clone();
        opt.step(&mut model, 0.1);
        let want: Vec<f64> = polar_factor(&g, 5, 5).iter().map(|x| -0.1 * x).collect();
        assert_close(opt.update(2), &want, 1e-9, "Shampoo's first step");
        assert!(opt.as_matrix(2) && opt.rule_of(2) == Rule::Shampoo && opt.rule_of(3) == Rule::Adam);
        let (l, r) = opt.factors(2);
        assert_close(l.unwrap().matrix, &naive(&g, &tr(&g, 5, 5), 5, 5, 5), 1e-14, "L");
        assert_eq!(r.unwrap().n, 5);
    }

    /// Muon with the exact polar factor gives an update whose singular values are all lr
    /// times the shape factor, for a full-rank momentum; with the reference iteration its
    /// singular values are where the scalar iteration sends them, and for every condition
    /// number from 1 to 1e6 those whose normalised input is at least 0.003 lie in [0.68,
    /// 1.2025] (the docstring claims "something like Uniform(0.5, 1.5)"; the notes compute
    /// [0.682, 1.202]). The ranges measured are printed.
    #[test]
    fn muon_singular_values_over_condition_numbers() {
        let mut rng = Rng::new(6);
        let spec = MlpSpec { inputs: 3, width: 24, depth: 3, outputs: 2, act: Act::Tanh, ..MlpSpec::default() };
        let base = Model::mlp(&spec, &mut Rng::new(1));
        let lr = 0.02;
        println!("condition | all output singular values | those from inputs >= 0.003 | smallest normalised input");
        for kappa in [1.0f64, 10.0, 1e2, 1e3, 1e4, 1e5, 1e6] {
            let k = 24;
            let s: Vec<f64> = (0..k).map(|i| kappa.powf(-(i as f64) / (k - 1) as f64)).collect();
            let (g, _, _) = with_singular_values(24, 24, &s, &mut rng);
            for exact in [true, false] {
                let mut model = base.clone();
                let ortho = if exact { Ortho::Exact } else { Ortho::Polynomial(Iteration::JORDAN) };
                let mut opt = Optimizer::new(&model, Rule::Muon, Hyper { ortho, ..Hyper::muon_reference() });
                model.groups_mut()[2].grad = g.clone();
                opt.step(&mut model, lr);
                let out: Vec<f64> = singular_values(opt.update(2), 24, 24).iter().map(|x| x / lr).collect();
                if exact {
                    assert!(out.iter().all(|x| (x - 1.0).abs() < 1e-9), "κ {kappa}: {out:?}");
                    continue;
                }
                // the matrix handed in is (1 − β)(1 + β) g (an EMA buffer and Nesterov's
                // mix at the first step): its singular values after the normalisation
                let c = (1.0 - 0.95) * (1.0 + 0.95);
                let frob = s.iter().map(|x| x * x).sum::<f64>().sqrt();
                let inputs: Vec<f64> = s.iter().map(|x| c * x / (c * frob + 1e-7)).collect();
                let mut want: Vec<f64> = inputs.iter().map(|&x| iterate_scalar(Odd5::JORDAN, x, 5)).collect();
                want.sort_by(|a, b| b.total_cmp(a));
                assert_close(&out, &want, 1e-9, &format!("κ {kappa}"));
                let big: Vec<f64> = inputs.iter().filter(|&&x| x >= 0.003).map(|&x| iterate_scalar(Odd5::JORDAN, x, 5)).collect();
                let range = |v: &[f64]| v.iter().fold((f64::INFINITY, 0.0f64), |(lo, hi), &x| (lo.min(x), hi.max(x)));
                let (all, kept) = (range(&out), range(&big));
                println!("{kappa:>9.0e} | [{:.4}, {:.4}]           | [{:.4}, {:.4}]          | {:.2e}", all.0, all.1, kept.0, kept.1, inputs[k - 1]);
                assert!(kept.0 >= 0.68 && kept.1 <= 1.2025, "κ {kappa}: {kept:?}");
            }
        }
        // a wide and a tall matrix, with each shape factor
        for (rows, cols) in [(4, 10), (10, 4)] {
            let spec = MlpSpec { inputs: cols, width: rows, depth: 1, outputs: 1, ..MlpSpec::default() };
            let mut model = Model::mlp(&spec, &mut Rng::new(2));
            for (scale, factor) in [
                (Scale::One, 1.0),
                (Scale::Keller, (rows as f64 / cols as f64).max(1.0).sqrt()),
                (Scale::Bernstein, (rows as f64 / cols as f64).sqrt()),
                (Scale::MatchAdamW(0.2), 0.2 * (rows.max(cols) as f64).sqrt()),
            ] {
                let mut opt = Optimizer::new(&model, Rule::Muon, Hyper { ortho: Ortho::Exact, scale, groups: Groups::Matrices, ..Hyper::muon_reference() });
                model.groups_mut()[0].grad = randn(rows * cols, &mut rng);
                opt.step(&mut model, lr);
                let out = singular_values(opt.update(0), rows, cols);
                assert!(out.iter().all(|x| (x / (lr * factor) - 1.0).abs() < 1e-9), "{scale:?} on {rows} by {cols}: {out:?}");
            }
        }
    }

    // ----- the rules against their formulas ------------------------------------------------

    /// A small plain tanh MLP and a batch: groups 0 layer 1 weights (5 by 3, the input
    /// layer), 1 its bias, 2 and 4 the hidden weights (5 by 5), 3 and 5 their biases, 6 the
    /// read-out (2 by 5), 7 its bias.
    fn bench(seed: u64) -> (Model, Vec<f64>, Vec<f64>) {
        let spec = MlpSpec { inputs: 3, width: 5, depth: 3, outputs: 2, act: Act::Tanh, init: Init::Xavier, ..MlpSpec::default() };
        let mut rng = Rng::new(seed);
        let model = Model::mlp(&spec, &mut rng);
        let x = randn(8 * 3, &mut rng);
        let y = (0..8).map(|i| (i % 2) as f64).collect();
        (model, x, y)
    }

    /// Put the mean gradient of the batch in the groups.
    fn fill_grads(model: &mut Model, tape: &mut Tape, x: &[f64], y: &[f64]) {
        let trace = model.loss_mlp(tape, x, y, Loss::CrossEntropy);
        tape.backward(trace.loss);
        model.zero_grads();
        model.accumulate(tape, &trace);
    }

    /// What a reference implementation is handed each step: the step (from 1), its rate,
    /// the groups (their gradients and shapes), and the values it keeps, to update.
    type Reference<'a> = Box<dyn FnMut(usize, f64, &[Group], &mut [Vec<f64>]) + 'a>;

    /// Steps of `rule` with `hyper` at the rates `lrs` agree with the reference to 1e-12.
    fn agree(name: &str, rule: Rule, hyper: Hyper, lrs: &[f64], reference: Reference) {
        agree_with(name, rule, hyper, lrs, 0.0, reference);
    }

    /// The same with normal noise of size `noise` added to every gradient (which makes
    /// every gradient matrix well conditioned).
    fn agree_with(name: &str, rule: Rule, hyper: Hyper, lrs: &[f64], noise: f64, mut reference: Reference) {
        let (mut model, x, y) = bench(11);
        let mut tape = Tape::new();
        let mut rng = Rng::new(111);
        let mut opt = Optimizer::new(&model, rule, hyper);
        let mut values: Vec<Vec<f64>> = model.groups().iter().map(|g| g.value.clone()).collect();
        for (t, &lr) in lrs.iter().enumerate() {
            fill_grads(&mut model, &mut tape, &x, &y);
            if noise > 0.0 {
                model.groups_mut().iter_mut().for_each(|g| g.grad.iter_mut().for_each(|d| *d += noise * rng.normal() as f64));
            }
            reference(t + 1, lr, model.groups(), &mut values);
            opt.step(&mut model, lr);
            for (i, g) in model.groups().iter().enumerate() {
                assert_close(&g.value, &values[i], 1e-12, &format!("{name}, step {}, group {i}", t + 1));
            }
        }
    }

    /// Adam on one group, as Kingma and Ba's Algorithm 1 writes it.
    #[derive(Default)]
    struct RefAdam {
        m: Vec<Vec<f64>>,
        v: Vec<Vec<f64>>,
    }

    impl RefAdam {
        #[allow(clippy::too_many_arguments)]
        fn step(&mut self, i: usize, t: usize, g: &[f64], theta: &mut [f64], lr: f64, b1: f64, b2: f64, eps: f64) {
            if self.m.len() <= i {
                self.m.resize(i + 1, Vec::new());
                self.v.resize(i + 1, Vec::new());
            }
            if self.m[i].is_empty() {
                self.m[i] = vec![0.0; g.len()];
                self.v[i] = vec![0.0; g.len()];
            }
            for j in 0..g.len() {
                self.m[i][j] = b1 * self.m[i][j] + (1.0 - b1) * g[j];
                self.v[i][j] = b2 * self.v[i][j] + (1.0 - b2) * g[j] * g[j];
                let mhat = self.m[i][j] / (1.0 - b1.powi(t as i32));
                let vhat = self.v[i][j] / (1.0 - b2.powi(t as i32));
                theta[j] -= lr * mhat / (vhat.sqrt() + eps);
            }
        }
    }

    fn zeros(groups: &[Group]) -> Vec<Vec<f64>> {
        groups.iter().map(|g| vec![0.0; g.value.len()]).collect()
    }

    /// Every rule that treats parameters alike, three steps against its source's
    /// pseudo-code written out.
    #[test]
    fn the_elementwise_rules_agree_with_their_formulas() {
        let lrs = [0.05, 0.02, 0.08];
        agree(
            "SGD",
            Rule::Sgd,
            Hyper::default_for(Rule::Sgd),
            &lrs,
            Box::new(|_, lr, gs, th| {
                for (g, th) in gs.iter().zip(th) {
                    for j in 0..th.len() {
                        th[j] -= lr * g.grad[j];
                    }
                }
            }),
        );
        // PyTorch SGD with momentum and dampening: the buffer starts at the first gradient
        let mut b: Vec<Vec<f64>> = Vec::new();
        agree(
            "heavy ball (PyTorch, dampening 0.3)",
            Rule::Momentum,
            Hyper { momentum: 0.8, dampening: 0.3, ..Hyper::default_for(Rule::Momentum) },
            &lrs,
            Box::new(move |t, lr, gs, th| {
                if b.is_empty() {
                    b = zeros(gs);
                }
                for (i, g) in gs.iter().enumerate() {
                    for j in 0..g.grad.len() {
                        b[i][j] = if t > 1 { 0.8 * b[i][j] + (1.0 - 0.3) * g.grad[j] } else { g.grad[j] };
                        th[i][j] -= lr * b[i][j];
                    }
                }
            }),
        );
        for (conv, nesterov) in [(Convention::Ema, false), (Convention::LrInside, false), (Convention::Sum, true), (Convention::Ema, true)] {
            let mut b: Vec<Vec<f64>> = Vec::new();
            agree(
                &format!("momentum {conv:?}, Nesterov {nesterov}"),
                if nesterov { Rule::Nesterov } else { Rule::Momentum },
                Hyper { momentum: 0.9, convention: conv, ..Hyper::default_for(Rule::Momentum) },
                &lrs,
                Box::new(move |_, lr, gs, th| {
                    if b.is_empty() {
                        b = zeros(gs);
                    }
                    let mu = 0.9;
                    for (i, g) in gs.iter().enumerate() {
                        for j in 0..g.grad.len() {
                            let gj = g.grad[j];
                            match (conv, nesterov) {
                                (Convention::Ema, false) => {
                                    b[i][j] = mu * b[i][j] + (1.0 - mu) * gj;
                                    th[i][j] -= lr * b[i][j];
                                }
                                (Convention::LrInside, _) => {
                                    b[i][j] = mu * b[i][j] + lr * gj;
                                    th[i][j] -= b[i][j];
                                }
                                (Convention::Sum, true) => {
                                    b[i][j] = mu * b[i][j] + gj;
                                    th[i][j] -= lr * (gj + mu * b[i][j]);
                                }
                                _ => {
                                    b[i][j] = mu * b[i][j] + (1.0 - mu) * gj;
                                    th[i][j] -= lr * (mu * b[i][j] + (1.0 - mu) * gj);
                                }
                            }
                        }
                    }
                }),
            );
        }
        // AdaGrad with its learning-rate decay, a starting accumulator and coupled decay
        let mut s: Vec<Vec<f64>> = Vec::new();
        agree(
            "AdaGrad",
            Rule::AdaGrad,
            Hyper { lr_decay: 0.1, initial_accumulator: 0.2, weight_decay: WeightDecay::Coupled(0.01), ..Hyper::default_for(Rule::AdaGrad) },
            &lrs,
            Box::new(move |t, lr, gs, th| {
                if s.is_empty() {
                    s = gs.iter().map(|g| vec![0.2; g.value.len()]).collect();
                }
                let clr = lr / (1.0 + (t - 1) as f64 * 0.1);
                for (i, g) in gs.iter().enumerate() {
                    for j in 0..g.grad.len() {
                        let gj = g.grad[j] + 0.01 * th[i][j];
                        s[i][j] += gj * gj;
                        th[i][j] -= clr * gj / (s[i][j].sqrt() + 1e-10);
                    }
                }
            }),
        );
        for (centered, mu) in [(false, 0.0), (true, 0.5)] {
            let (mut v, mut ga, mut bb) = (zeros(&[]), zeros(&[]), zeros(&[]));
            agree(
                &format!("RMSProp centred {centered}"),
                Rule::RmsProp,
                Hyper { centered, momentum: mu, ..Hyper::default_for(Rule::RmsProp) },
                &lrs,
                Box::new(move |_, lr, gs, th| {
                    if v.is_empty() {
                        (v, ga, bb) = (zeros(gs), zeros(gs), zeros(gs));
                    }
                    for (i, g) in gs.iter().enumerate() {
                        for j in 0..g.grad.len() {
                            let gj = g.grad[j];
                            v[i][j] = 0.99 * v[i][j] + (1.0 - 0.99) * gj * gj;
                            let mut vt = v[i][j];
                            if centered {
                                ga[i][j] = 0.99 * ga[i][j] + (1.0 - 0.99) * gj;
                                vt -= ga[i][j] * ga[i][j];
                            }
                            if mu > 0.0 {
                                bb[i][j] = mu * bb[i][j] + gj / (vt.sqrt() + 1e-8);
                                th[i][j] -= lr * bb[i][j];
                            } else {
                                th[i][j] -= lr * gj / (vt.sqrt() + 1e-8);
                            }
                        }
                    }
                }),
            );
        }
        // Adam: AdamW (PyTorch), Adam with L2 and AMSGrad, Adam without bias correction and
        // with the paper's scheduled decay
        for case in 0..3 {
            let hyper = match case {
                0 => Hyper { weight_decay: WeightDecay::Decoupled(0.1), ..Hyper::default_for(Rule::Adam) },
                1 => Hyper { weight_decay: WeightDecay::Coupled(0.05), amsgrad: true, ..Hyper::default_for(Rule::Adam) },
                _ => Hyper { weight_decay: WeightDecay::Scheduled(0.3), bias_correction: false, beta1: 0.8, beta2: 0.9, ..Hyper::default_for(Rule::Adam) },
            };
            let (mut m, mut v, mut vmax) = (zeros(&[]), zeros(&[]), zeros(&[]));
            agree(
                &format!("Adam case {case}"),
                Rule::Adam,
                hyper,
                &lrs,
                Box::new(move |t, lr, gs, th| {
                    if m.is_empty() {
                        (m, v, vmax) = (zeros(gs), zeros(gs), zeros(gs));
                    }
                    let (b1, b2) = (hyper.beta1, hyper.beta2);
                    for (i, g) in gs.iter().enumerate() {
                        for j in 0..g.grad.len() {
                            let gj = g.grad[j] + if case == 1 { 0.05 * th[i][j] } else { 0.0 };
                            m[i][j] = b1 * m[i][j] + (1.0 - b1) * gj;
                            v[i][j] = b2 * v[i][j] + (1.0 - b2) * gj * gj;
                            vmax[i][j] = vmax[i][j].max(v[i][j]);
                            let vv = if case == 1 { vmax[i][j] } else { v[i][j] };
                            let (mh, vh) = if case == 2 { (m[i][j], vv) } else { (m[i][j] / (1.0 - b1.powi(t as i32)), vv / (1.0 - b2.powi(t as i32))) };
                            let step = lr * mh / (vh.sqrt() + 1e-8);
                            let decay = match case {
                                0 => lr * 0.1 * th[i][j],
                                2 => (lr / 1e-3) * 0.3 * th[i][j],
                                _ => 0.0,
                            };
                            th[i][j] -= step + decay;
                        }
                    }
                }),
            );
        }
        agree(
            "signSGD",
            Rule::SignSgd,
            Hyper::default_for(Rule::SignSgd),
            &lrs,
            Box::new(|_, lr, gs, th| {
                for (g, th) in gs.iter().zip(th) {
                    for j in 0..th.len() {
                        th[j] -= lr * g.grad[j].signum();
                    }
                }
            }),
        );
        let mut m: Vec<Vec<f64>> = Vec::new();
        agree(
            "Signum",
            Rule::Signum,
            Hyper::default_for(Rule::Signum),
            &lrs,
            Box::new(move |_, lr, gs, th| {
                if m.is_empty() {
                    m = zeros(gs);
                }
                for (i, g) in gs.iter().enumerate() {
                    for j in 0..g.grad.len() {
                        m[i][j] = 0.9 * m[i][j] + (1.0 - 0.9) * g.grad[j];
                        th[i][j] -= lr * m[i][j].signum();
                    }
                }
            }),
        );
        let mut m: Vec<Vec<f64>> = Vec::new();
        agree(
            "Lion",
            Rule::Lion,
            Hyper { weight_decay: WeightDecay::Decoupled(0.5), ..Hyper::default_for(Rule::Lion) },
            &lrs,
            Box::new(move |_, lr, gs, th| {
                if m.is_empty() {
                    m = zeros(gs);
                }
                for (i, g) in gs.iter().enumerate() {
                    for j in 0..g.grad.len() {
                        let c = 0.9 * m[i][j] + (1.0 - 0.9) * g.grad[j];
                        th[i][j] -= lr * (c.signum() + 0.5 * th[i][j]);
                        m[i][j] = 0.99 * m[i][j] + (1.0 - 0.99) * g.grad[j];
                    }
                }
            }),
        );
    }

    /// The polar factor is right to about ε κ, measured against `linalg::svd` for
    /// condition numbers up to 1e6 (the errors are printed).
    #[test]
    fn the_polar_factor_is_accurate_on_ill_conditioned_matrices() {
        let mut rng = Rng::new(31);
        for kappa in [1.0f64, 1e2, 1e4, 1e6] {
            for (rows, cols) in [(20, 12), (12, 20)] {
                let k = 12;
                let s: Vec<f64> = (0..k).map(|i| kappa.powf(-(i as f64) / (k - 1) as f64)).collect();
                let (a, u, v) = with_singular_values(rows, cols, &s, &mut rng);
                let mut want = vec![0.0; rows * cols];
                for i in 0..rows {
                    for j in 0..cols {
                        want[i * cols + j] = (0..k).map(|p| u[i * k + p] * v[j * k + p]).sum();
                    }
                }
                let p = polar_factor(&a, rows, cols);
                let err = p.iter().zip(&want).map(|(x, y)| (x - y).abs()).fold(0.0, f64::max);
                println!("condition {kappa:.0e}, {rows} by {cols}: largest error {err:.1e}");
                assert!(err < 1e-14 * kappa.max(100.0), "κ {kappa}: {err}");
            }
        }
    }

    /// The matrix rules, three or four steps against their sources' pseudo-code written
    /// out (the reference code's Newton–Schulz, `linalg::svd` for exact polar factors,
    /// `linalg::eig_sym` for Shampoo's roots and SOAP's first basis, `linalg::orthonormalise`
    /// for SOAP's QR), the other groups against Adam's formulas.
    #[test]
    fn the_matrix_rules_agree_with_their_formulas() {
        let lrs = [0.01, 0.02, 0.015];
        let hidden = [2usize, 4];
        let matrices = [0usize, 2, 4, 6];
        let jordan = [Odd5::JORDAN];
        // (name, hyper, which groups, the form)
        #[derive(Clone, Copy, PartialEq)]
        enum Form {
            Reference,
            Blog,
            Moonshot,
            Torch,
            OrthogonalSgdm,
            NorMuon,
            AdaMuon,
            Carlson,
            PolarMuon,
            Scion,
            PolarExpress,
        }
        let cases: [(&str, Rule, Hyper, &[usize], Form); 13] = [
            ("Muon (reference), hidden", Rule::Muon, Hyper { weight_decay: WeightDecay::Decoupled(0.1), ..Hyper::muon_reference() }, &hidden, Form::Reference),
            ("Muon (reference), matrices", Rule::Muon, Hyper { groups: Groups::Matrices, ..Hyper::muon_reference() }, &matrices, Form::Reference),
            ("Muon (blog)", Rule::Muon, Hyper { groups: Groups::Matrices, ..Hyper::muon_blog() }, &matrices, Form::Blog),
            ("Muon (Moonshot)", Rule::Muon, Hyper { groups: Groups::Matrices, ..Hyper::muon_moonshot() }, &matrices, Form::Moonshot),
            ("Muon (torch)", Rule::Muon, Hyper { groups: Groups::Matrices, ..Hyper::muon_torch() }, &matrices, Form::Torch),
            ("Orthogonal-SGDM", Rule::Muon, Hyper::orthogonal_sgdm(), &matrices, Form::OrthogonalSgdm),
            ("NorMuon", Rule::NorMuon, Hyper { groups: Groups::Matrices, ..Hyper::normuon() }, &matrices, Form::NorMuon),
            (
                "AdaMuon",
                Rule::AdaMuon,
                Hyper { groups: Groups::Matrices, weight_decay: WeightDecay::Decoupled(0.2), ..Hyper::adamuon() },
                &matrices,
                Form::AdaMuon,
            ),
            ("spectral descent", Rule::SpectralDescent, Hyper::spectral_descent(), &matrices, Form::Carlson),
            ("PolarMuon", Rule::SpectralDescent, Hyper { weight_decay: WeightDecay::Decoupled(0.3), ..Hyper::polar_muon() }, &hidden, Form::PolarMuon),
            ("Scion", Rule::Scion, Hyper::scion(), &[0, 2, 4], Form::Scion),
            (
                "Muon, Polar Express (speedrun)",
                Rule::Muon,
                Hyper { ortho: Ortho::Polynomial(Iteration::POLAR_EXPRESS_RECORD), groups: Groups::Matrices, ..Hyper::muon_reference() },
                &matrices,
                Form::PolarExpress,
            ),
            (
                "Muon, Polar Express (paper), exact rounds",
                Rule::Muon,
                Hyper { ortho: Ortho::Polynomial(Iteration::POLAR_EXPRESS_PAPER), groups: Groups::Matrices, ..Hyper::muon_reference() },
                &matrices,
                Form::PolarExpress,
            ),
        ];
        for (name, rule, hyper, chosen, form) in cases {
            let chosen = chosen.to_vec();
            let (mut m, mut v, mut adam) = (Vec::<Vec<f64>>::new(), Vec::<Vec<f64>>::new(), RefAdam::default());
            agree(
                name,
                rule,
                hyper,
                &lrs,
                Box::new(move |t, lr, gs, th| {
                    if m.is_empty() {
                        (m, v) = (zeros(gs), zeros(gs));
                    }
                    let aux_lr = hyper.aux.lr * lr / hyper.lr;
                    for (i, g) in gs.iter().enumerate() {
                        let (r, c, n) = (g.rows, g.cols, g.value.len());
                        let gr = &g.grad;
                        if !chosen.contains(&i) {
                            if form == Form::Carlson {
                                let mean = gr.iter().map(|x| x.abs()).sum::<f64>() / n as f64;
                                for j in 0..n {
                                    th[i][j] -= lr * mean * gr[j].signum();
                                }
                            } else {
                                let a = hyper.aux;
                                let decay = if let WeightDecay::Decoupled(l) = a.weight_decay { l } else { 0.0 };
                                let before = th[i].clone();
                                adam.step(i, t, gr, &mut th[i], aux_lr, a.beta1, a.beta2, a.eps);
                                for j in 0..n {
                                    th[i][j] -= aux_lr * decay * before[j];
                                }
                            }
                            continue;
                        }
                        let mu = hyper.momentum;
                        let ns = |x: &[f64]| match hyper.ortho {
                            Ortho::Polynomial(it) => reference_ns(x, r, c, it.polys, it.rounds, it.normalise.scale, it.normalise.eps, it.normalise.clamp),
                            Ortho::Exact => svd_polar(x, r, c),
                        };
                        let (rf, cf) = (r as f64, c as f64);
                        match form {
                            Form::Reference | Form::Torch | Form::PolarExpress => {
                                // momentum.lerp_(grad, 1 - beta); update = grad.lerp_(momentum, beta)
                                for j in 0..n {
                                    m[i][j] = mu * m[i][j] + (1.0 - mu) * gr[j];
                                }
                                let u: Vec<f64> = (0..n).map(|j| (1.0 - mu) * gr[j] + mu * m[i][j]).collect();
                                let o = ns(&u);
                                let f = (rf / cf).max(1.0).sqrt();
                                let wd = if let WeightDecay::Decoupled(l) = hyper.weight_decay { l } else { 0.0 };
                                for j in 0..n {
                                    th[i][j] = th[i][j] * (1.0 - lr * wd) - lr * f * o[j];
                                }
                            }
                            Form::Blog => {
                                for j in 0..n {
                                    m[i][j] = mu * m[i][j] + gr[j];
                                }
                                let o = reference_ns(&m[i], r, c, &jordan, 5, 1.0, 1e-7, false);
                                for j in 0..n {
                                    th[i][j] -= lr * o[j];
                                }
                            }
                            Form::Moonshot => {
                                for j in 0..n {
                                    m[i][j] = mu * m[i][j] + gr[j];
                                }
                                let u: Vec<f64> = (0..n).map(|j| mu * m[i][j] + gr[j]).collect();
                                let o = ns(&u);
                                for j in 0..n {
                                    th[i][j] -= lr * (0.2 * o[j] * rf.max(cf).sqrt() + 0.01 * th[i][j]);
                                }
                            }
                            Form::OrthogonalSgdm => {
                                let o = svd_polar(gr, r, c);
                                for j in 0..n {
                                    m[i][j] = mu * m[i][j] + lr * o[j];
                                    th[i][j] -= m[i][j];
                                }
                            }
                            Form::NorMuon => {
                                for j in 0..n {
                                    m[i][j] = 0.95 * m[i][j] + (1.0 - 0.95) * gr[j];
                                }
                                let o = reference_ns(&m[i], r, c, &jordan, 5, 1.0, 1e-7, false);
                                let mut oh = o.clone();
                                for row in 0..r {
                                    let ms = (0..c).map(|q| o[row * c + q] * o[row * c + q]).sum::<f64>() / cf;
                                    v[i][row] = 0.95 * v[i][row] + (1.0 - 0.95) * ms;
                                    (0..c).for_each(|q| oh[row * c + q] = o[row * c + q] / (v[i][row].sqrt() + 1e-8));
                                }
                                let eta = 0.2 * lr * (rf * cf).sqrt() / oh.iter().map(|x| x * x).sum::<f64>().sqrt();
                                for j in 0..n {
                                    th[i][j] = th[i][j] - lr * 0.1 * th[i][j] - eta * oh[j];
                                }
                            }
                            Form::AdaMuon => {
                                for j in 0..n {
                                    m[i][j] = 0.95 * m[i][j] + gr[j];
                                }
                                let signs: Vec<f64> = m[i].iter().map(|x| x.signum()).collect();
                                let o = reference_ns(&signs, r, c, &jordan, 5, 1.0, 1e-7, false);
                                let mut oh = vec![0.0; n];
                                for j in 0..n {
                                    v[i][j] = 0.95 * v[i][j] + (1.0 - 0.95) * o[j] * o[j];
                                    oh[j] = o[j] / (v[i][j].sqrt() + 1e-8);
                                }
                                let gamma = 0.2 * (rf * cf).sqrt() / oh.iter().map(|x| x * x).sum::<f64>().sqrt();
                                for j in 0..n {
                                    th[i][j] -= lr * (gamma * oh[j] + 0.2 * th[i][j]);
                                }
                            }
                            Form::Carlson => {
                                // [A, Σ, B] = svd(dW); W ← W − ε (‖Σ‖₁ / MJ) A Bᵀ
                                let (_, sv, _) = linalg::svd(gr, r, c);
                                let o = svd_polar(gr, r, c);
                                let nuc: f64 = sv.iter().sum();
                                for j in 0..n {
                                    th[i][j] -= lr * nuc / (rf * cf) * o[j];
                                }
                            }
                            Form::PolarMuon => {
                                for j in 0..n {
                                    m[i][j] = mu * m[i][j] + (1.0 - mu) * gr[j];
                                }
                                let (_, sv, _) = linalg::svd(&m[i], r, c);
                                let o = svd_polar(&m[i], r, c);
                                let nuc: f64 = sv.iter().sum();
                                for j in 0..n {
                                    th[i][j] = (1.0 - 0.3 * lr) * th[i][j] - lr * nuc * o[j];
                                }
                            }
                            Form::Scion => {
                                // d = α g + (1 − α) d; x = (1 − γ) x + γ lmo(d), lmo = −ρ' U Vᵀ
                                for j in 0..n {
                                    m[i][j] = 0.05 * gr[j] + (1.0 - 0.05) * m[i][j];
                                }
                                let o = svd_polar(&m[i], r, c);
                                let radius = if i == 0 { (rf / cf).sqrt().max(1.0) } else { (rf / cf).sqrt() };
                                for j in 0..n {
                                    th[i][j] = (1.0 - lr) * th[i][j] - lr * radius * o[j];
                                }
                            }
                        }
                    }
                }),
            );
        }
        // Shampoo: L = εI + Σ G Gᵀ, R = εI + Σ Gᵀ G, W ← W − lr L^(−p) G R^(−p)
        for (p, every) in [(0.25, 1u32), (0.5, 2)] {
            let hyper = Hyper { damping: 1e-3, exponent: p, root_every: every, ..Hyper::shampoo() };
            let (mut ls, mut rs, mut roots, mut adam) =
                (Vec::<Vec<f64>>::new(), Vec::<Vec<f64>>::new(), Vec::<(Vec<f64>, Vec<f64>)>::new(), RefAdam::default());
            agree(
                &format!("Shampoo p {p} every {every}"),
                Rule::Shampoo,
                hyper,
                &lrs,
                Box::new(move |t, lr, gs, th| {
                    if ls.is_empty() {
                        ls = gs.iter().map(|g| (0..g.rows * g.rows).map(|k| if k % (g.rows + 1) == 0 { 1e-3 } else { 0.0 }).collect()).collect();
                        rs = gs.iter().map(|g| (0..g.cols * g.cols).map(|k| if k % (g.cols + 1) == 0 { 1e-3 } else { 0.0 }).collect()).collect();
                        roots = vec![(Vec::new(), Vec::new()); gs.len()];
                    }
                    for (i, g) in gs.iter().enumerate() {
                        let (r, c) = (g.rows, g.cols);
                        if !matrices.contains(&i) {
                            adam.step(i, t, &g.grad, &mut th[i], hyper.aux.lr * lr / hyper.lr, 0.9, 0.999, 1e-8);
                            continue;
                        }
                        let ggt = naive(&g.grad, &tr(&g.grad, r, c), r, c, r);
                        let gtg = naive(&tr(&g.grad, r, c), &g.grad, c, r, c);
                        (0..r * r).for_each(|k| ls[i][k] += ggt[k]);
                        (0..c * c).for_each(|k| rs[i][k] += gtg[k]);
                        if (t - 1) % every as usize == 0 {
                            roots[i] = (eig_root(&ls[i], r, p, 0.0), eig_root(&rs[i], c, p, 0.0));
                        }
                        let d = naive(&naive(&roots[i].0, &g.grad, r, r, c), &roots[i].1, r, c, c);
                        for j in 0..r * c {
                            th[i][j] -= lr * d[j];
                        }
                    }
                }),
            );
        }
        // SOAP, the official code's steps (with the momentum rotated out and back every
        // step, as the code does), refreshing every 2 steps
        let hyper = Hyper { precondition_every: 2, groups: Groups::Hidden, ..Hyper::soap() };
        #[derive(Clone, Default)]
        struct SoapRef {
            l: Vec<f64>,
            r: Vec<f64>,
            ql: Vec<f64>,
            qr: Vec<f64>,
            m: Vec<f64>,
            v: Vec<f64>,
            t: i32,
            started: bool,
        }
        let mut st: Vec<SoapRef> = Vec::new();
        // eigenvectors as the columns of an n by n matrix, largest eigenvalue first
        let eig_cols = |f: &[f64], n: usize| {
            let mut a = f.to_vec();
            (0..n).for_each(|i| a[i * n + i] += 1e-30);
            let (_, vecs) = linalg::eig_sym(&a, n);
            tr(&vecs, n, n)
        };
        let refresh = |f: &[f64], q: &mut Vec<f64>, v: &mut Vec<f64>, n: usize, rows: usize, cols: usize, left: bool| {
            let fq = naive(f, q, n, n, n);
            let est: Vec<f64> = (0..n).map(|k| (0..n).map(|i| q[i * n + k] * fq[i * n + k]).sum()).collect();
            let mut order: Vec<usize> = (0..n).collect();
            order.sort_by(|&a, &b| est[b].total_cmp(&est[a]));
            let q2: Vec<f64> = (0..n * n).map(|k| q[(k / n) * n + order[k % n]]).collect();
            let v2: Vec<f64> =
                (0..rows * cols).map(|k| if left { v[order[k / cols] * cols + k % cols] } else { v[(k / cols) * cols + order[k % cols]] }).collect();
            let mut s = naive(f, &q2, n, n, n);
            linalg::orthonormalise(&mut s, n, n);
            *q = s;
            *v = v2;
        };
        // (with noise in the gradients: SOAP's step divides each coordinate of the rotated
        // momentum by its own root second moment, so the eigenvectors of the factors'
        // tiny, nearly equal eigenvalues, which two eigen-solvers agree on only to rounding
        // over the gap, would matter at 1e-7 with the network's own badly conditioned
        // gradients)
        agree_with(
            "SOAP",
            Rule::Soap,
            hyper,
            &[0.01, 0.02, 0.015, 0.01, 0.02],
            0.1,
            Box::new(move |_, lr, gs, th| {
                if st.is_empty() {
                    st = vec![SoapRef::default(); gs.len()];
                }
                let (b1, b2, eps, wd) = (0.95, 0.95, 1e-8, 0.01);
                for (i, g) in gs.iter().enumerate() {
                    let (r, c, n) = (g.rows, g.cols, g.value.len());
                    let rot = hidden.contains(&i);
                    let lr = if rot { lr } else { hyper.aux.lr * lr / hyper.lr };
                    let s = &mut st[i];
                    let gg = &g.grad;
                    if !s.started {
                        s.started = true;
                        s.m = vec![0.0; n];
                        s.v = vec![0.0; n];
                        if rot {
                            s.l = naive(gg, &tr(gg, r, c), r, c, r).iter().map(|x| (1.0 - b2) * x).collect();
                            s.r = naive(&tr(gg, r, c), gg, c, r, c).iter().map(|x| (1.0 - b2) * x).collect();
                            s.ql = eig_cols(&s.l, r);
                            s.qr = eig_cols(&s.r, c);
                        }
                        continue;
                    }
                    let gp = if rot { naive(&naive(&tr(&s.ql, r, r), gg, r, r, c), &s.qr, r, c, c) } else { gg.clone() };
                    s.t += 1;
                    for j in 0..n {
                        s.m[j] = b1 * s.m[j] + (1.0 - b1) * gp[j];
                        s.v[j] = b2 * s.v[j] + (1.0 - b2) * gp[j] * gp[j];
                    }
                    let x: Vec<f64> = (0..n).map(|j| s.m[j] / (s.v[j].sqrt() + eps)).collect();
                    let nn = if rot { naive(&naive(&s.ql, &x, r, r, c), &tr(&s.qr, c, c), r, c, c) } else { x };
                    let alpha = lr * (1.0 - b2.powi(s.t)).sqrt() / (1.0 - b1.powi(s.t));
                    for j in 0..n {
                        th[i][j] -= alpha * nn[j];
                    }
                    for j in 0..n {
                        th[i][j] -= lr * wd * th[i][j];
                    }
                    if rot {
                        let morig = naive(&naive(&s.ql, &s.m, r, r, c), &tr(&s.qr, c, c), r, c, c);
                        let ggt = naive(gg, &tr(gg, r, c), r, c, r);
                        let gtg = naive(&tr(gg, r, c), gg, c, r, c);
                        (0..r * r).for_each(|k| s.l[k] = b2 * s.l[k] + (1.0 - b2) * ggt[k]);
                        (0..c * c).for_each(|k| s.r[k] = b2 * s.r[k] + (1.0 - b2) * gtg[k]);
                        if s.t % 2 == 0 {
                            refresh(&s.l, &mut s.ql, &mut s.v, r, r, c, true);
                            refresh(&s.r, &mut s.qr, &mut s.v, c, r, c, false);
                        }
                        s.m = naive(&naive(&tr(&s.ql, r, r), &morig, r, r, c), &s.qr, r, c, c);
                    }
                }
            }),
        );
    }

    /// Adam at its default settings agrees with `arch::Model::step_adam`.
    #[test]
    fn adam_agrees_with_the_model_s_own_step() {
        let (mut a, x, y) = bench(12);
        let mut b = a.clone();
        let mut tape = Tape::new();
        let mut opt = Optimizer::new(&a, Rule::Adam, Hyper::default_for(Rule::Adam));
        for step in 0..6 {
            fill_grads(&mut a, &mut tape, &x, &y);
            fill_grads(&mut b, &mut tape, &x, &y);
            opt.step(&mut a, 0.01);
            b.step_adam(0.01);
            for (ga, gb) in a.groups().iter().zip(b.groups()) {
                assert_close(&ga.value, &gb.value, 1e-12, &format!("step {step}"));
            }
        }
    }

    /// Adam's first step moves every parameter by the learning rate against its
    /// gradient's sign (for gradients well above ε); with both betas 0 and ε 0 every step
    /// is the sign of the gradient.
    #[test]
    fn adam_s_first_step_and_its_sign_limit() {
        let (mut model, x, y) = bench(13);
        let mut tape = Tape::new();
        fill_grads(&mut model, &mut tape, &x, &y);
        let mut opt = Optimizer::new(&model, Rule::Adam, Hyper::default_for(Rule::Adam));
        let grads: Vec<Vec<f64>> = model.groups().iter().map(|g| g.grad.clone()).collect();
        opt.step(&mut model, 0.01);
        for (i, g) in grads.iter().enumerate() {
            for (u, d) in opt.update(i).iter().zip(g) {
                if d.abs() > 1e-4 {
                    assert!((u + 0.01 * d.signum()).abs() < 1e-6, "first step {u} for gradient {d}");
                }
            }
        }
        let mut opt = Optimizer::new(&model, Rule::Adam, Hyper { beta1: 0.0, beta2: 0.0, eps: 0.0, ..Hyper::default_for(Rule::Adam) });
        for _ in 0..3 {
            fill_grads(&mut model, &mut tape, &x, &y);
            let grads: Vec<Vec<f64>> = model.groups().iter().map(|g| g.grad.clone()).collect();
            opt.step(&mut model, 0.01);
            for (i, g) in grads.iter().enumerate() {
                for (u, d) in opt.update(i).iter().zip(g) {
                    assert!((u + 0.01 * sgn(*d)).abs() < 1e-17, "{u} for gradient {d}");
                }
            }
        }
    }

    /// Muon's two momentum conventions give the same step (the orthogonalisation does not
    /// see the scale), to rounding, with and without Nesterov: with the iteration once its
    /// normaliser's 1e-7 is taken out (with it, the two differ by about 1e-7 / ‖U‖_F
    /// relative, U being the EMA's smaller matrix), and with the exact polar factor.
    #[test]
    fn muon_s_two_conventions_give_the_same_step() {
        let no_eps = Iteration { normalise: Normalise { eps: 0.0, ..Normalise::REFERENCE }, ..Iteration::JORDAN };
        for (ortho, tol) in [(Ortho::Polynomial(no_eps), 1e-10), (Ortho::Exact, 1e-11)] {
            for nesterov in [true, false] {
                let (mut a, x, y) = bench(14);
                let mut b = a.clone();
                let mut tape = Tape::new();
                let base = Hyper { ortho, nesterov, groups: Groups::Matrices, ..Hyper::muon_reference() };
                let mut ema = Optimizer::new(&a, Rule::Muon, Hyper { convention: Convention::Ema, ..base });
                let mut sum = Optimizer::new(&b, Rule::Muon, Hyper { convention: Convention::Sum, ..base });
                for step in 0..5 {
                    fill_grads(&mut a, &mut tape, &x, &y);
                    fill_grads(&mut b, &mut tape, &x, &y);
                    ema.step(&mut a, 0.02);
                    sum.step(&mut b, 0.02);
                    for i in [0, 2, 4, 6] {
                        let size = max_abs(sum.update(i));
                        assert_close(ema.update(i), sum.update(i), tol * size, &format!("{ortho:?} Nesterov {nesterov} step {step} group {i}"));
                    }
                }
            }
        }
    }

    /// Decoupled decay shrinks a weight by exactly lr × λ × weight whatever the gradient;
    /// coupled decay under Adam does not (it goes through the division by √v̂).
    #[test]
    fn weight_decay_decoupled_and_coupled() {
        let (model, _, _) = bench(15);
        let mut rng = Rng::new(15);
        for rule in Rule::ALL {
            let hyper = Hyper { weight_decay: WeightDecay::Decoupled(0.3), ..Hyper::default_for(rule) };
            for scale in [0.0, 1.0, 1e3] {
                let mut m = model.clone();
                let mut opt = Optimizer::new(&m, rule, hyper);
                for g in m.groups_mut() {
                    g.grad = randn(g.grad.len(), &mut rng).iter().map(|x| scale * x).collect();
                }
                let before: Vec<Vec<f64>> = m.groups().iter().map(|g| g.value.clone()).collect();
                opt.step(&mut m, 0.05);
                for (i, b) in before.iter().enumerate() {
                    let lam = if opt.as_matrix(i) || !rule.is_matrix_rule() { 0.3 } else { 0.0 };
                    if rule == Rule::Soap {
                        // SOAP: after its step, on the stepped values (and its first call only sets up)
                        let want: Vec<f64> = b.iter().zip(opt.update(i)).map(|(x, u)| if opt.steps() > 1 { -0.05 * 0.3 * (x + u) } else { 0.0 }).collect();
                        assert_close(opt.decay(i), &want, 1e-15, "SOAP's decay");
                        continue;
                    }
                    let want: Vec<f64> = b.iter().map(|x| -(0.05 * lam) * x).collect();
                    assert_eq!(opt.decay(i), &want[..], "{} decay, gradient scale {scale}", rule.name());
                }
            }
        }
        // coupled decay under Adam: the change is not −lr λ θ
        let mut m = model.clone();
        m.groups_mut().iter_mut().for_each(|g| g.value.iter_mut().for_each(|x| *x = 10.0 * *x + 1.0));
        let mut plain = m.clone();
        let mut opt = Optimizer::new(&m, Rule::Adam, Hyper { weight_decay: WeightDecay::Coupled(0.3), ..Hyper::default_for(Rule::Adam) });
        let mut base = Optimizer::new(&plain, Rule::Adam, Hyper::default_for(Rule::Adam));
        let before: Vec<f64> = m.groups()[2].value.clone();
        opt.step(&mut m, 0.05);
        base.step(&mut plain, 0.05);
        assert!(opt.decay(2).iter().all(|&d| d == 0.0));
        let extra: Vec<f64> = opt.update(2).iter().zip(base.update(2)).map(|(a, b)| a - b).collect();
        assert!(extra.iter().zip(&before).any(|(e, x)| (e + 0.05 * 0.3 * x).abs() > 0.1 * (0.05 * 0.3 * x).abs()), "{extra:?}");
        // with a zero gradient, Adam with coupled decay moves every weight by about lr, not lr λ θ
        assert!(opt.update(2).iter().all(|u| u.abs() <= 0.05 * 1.0001), "{:?}", opt.update(2));
    }

    /// Clipping by global norm: the step is the step on the gradient scaled by c / ‖g‖;
    /// it reports the norm and the factor; a cap above the norm, or NaN, changes nothing; a
    /// cap of 0 zeroes the gradient.
    #[test]
    fn clipping_by_global_norm() {
        let (model, x, y) = bench(16);
        let mut tape = Tape::new();
        let mut m = model.clone();
        fill_grads(&mut m, &mut tape, &x, &y);
        // (summed as the optimiser sums it, so that the factor is exactly 1/4: Shampoo's
        // damped null directions would magnify a difference in the last bit)
        let norm = m.groups().iter().map(|g| dot(&g.grad, &g.grad)).sum::<f64>().sqrt();
        assert!((norm - m.grad_norm()).abs() < 1e-15 * norm);
        for rule in [Rule::Sgd, Rule::Adam, Rule::Muon, Rule::Shampoo] {
            let mut clipped = m.clone();
            let mut scaled = m.clone();
            scaled.groups_mut().iter_mut().for_each(|g| g.grad.iter_mut().for_each(|d| *d *= 0.25));
            let mut a = Optimizer::new(&clipped, rule, Hyper { clip: norm / 4.0, ..Hyper::default_for(rule) });
            let mut b = Optimizer::new(&scaled, rule, Hyper::default_for(rule));
            a.step(&mut clipped, 0.01);
            b.step(&mut scaled, 0.01);
            assert!((a.grad_norm() - norm).abs() < 1e-15 * norm && (a.clip_scale() - 0.25).abs() < 1e-15);
            for i in 0..clipped.groups().len() {
                assert_close(a.update(i), b.update(i), 1e-12, rule.name());
            }
            for cap in [2.0 * norm, f64::NAN, f64::INFINITY] {
                let mut c = m.clone();
                let mut opt = Optimizer::new(&c, rule, Hyper { clip: cap, ..Hyper::default_for(rule) });
                opt.step(&mut c, 0.01);
                assert_eq!(opt.clip_scale(), 1.0);
            }
        }
        let mut c = m.clone();
        let mut opt = Optimizer::new(&c, Rule::Sgd, Hyper { clip: 0.0, ..Hyper::default_for(Rule::Sgd) });
        opt.step(&mut c, 0.01);
        assert_eq!(opt.clip_scale(), 0.0);
        assert!(c.groups().iter().zip(m.groups()).all(|(a, b)| a.value == b.value));
    }

    /// Every rule and every named form: update + decay is the change of the values.
    fn all_forms() -> Vec<(String, Rule, Hyper)> {
        let mut forms: Vec<(String, Rule, Hyper)> = Rule::ALL.iter().map(|&r| (r.name().to_string(), r, Hyper::default_for(r))).collect();
        for (name, rule, h) in [
            ("AdamW", Rule::Adam, Hyper::adamw()),
            ("Muon blog", Rule::Muon, Hyper::muon_blog()),
            ("Muon Moonshot", Rule::Muon, Hyper::muon_moonshot()),
            ("Muon torch", Rule::Muon, Hyper::muon_torch()),
            ("Orthogonal-SGDM", Rule::Muon, Hyper::orthogonal_sgdm()),
            ("PolarGrad", Rule::SpectralDescent, Hyper::polar_grad()),
            ("PolarMuon", Rule::SpectralDescent, Hyper::polar_muon()),
            ("uSCG", Rule::Scion, Hyper::scion_unconstrained()),
            ("Muon exact", Rule::Muon, Hyper { ortho: Ortho::Exact, ..Hyper::muon_reference() }),
            ("SOAP refresh 1", Rule::Soap, Hyper { precondition_every: 1, ..Hyper::soap() }),
            ("Shampoo root 3", Rule::Shampoo, Hyper { root_every: 3, ..Hyper::shampoo() }),
            ("momentum lr inside", Rule::Momentum, Hyper { convention: Convention::LrInside, ..Hyper::default_for(Rule::Momentum) }),
        ] {
            forms.push((name.to_string(), rule, h));
        }
        for it in Iteration::ALL {
            forms.push((format!("Muon {}", it.name), Rule::Muon, Hyper { ortho: Ortho::Polynomial(it), ..Hyper::muon_reference() }));
        }
        forms
    }

    #[test]
    fn update_and_decay_add_up_to_the_change() {
        let (model, x, y) = bench(17);
        let mut tape = Tape::new();
        for (name, rule, h) in all_forms() {
            let mut m = model.clone();
            let mut opt = Optimizer::new(&m, rule, Hyper { weight_decay: WeightDecay::Decoupled(0.1), ..h });
            for _ in 0..4 {
                fill_grads(&mut m, &mut tape, &x, &y);
                let before: Vec<Vec<f64>> = m.groups().iter().map(|g| g.value.clone()).collect();
                opt.step(&mut m, h.lr);
                for (i, g) in m.groups().iter().enumerate() {
                    let want: Vec<f64> = (0..g.value.len()).map(|j| before[i][j] + (opt.update(i)[j] + opt.decay(i)[j])).collect();
                    assert_eq!(g.value, want, "{name}");
                }
            }
        }
    }

    /// Nothing panics, for every rule and form, on a zero gradient, a rank-1 gradient (a
    /// batch of one), 1 by 1 matrices, a learning rate of 0, NaN and infinite gradients,
    /// and on a transformer; a zero gradient leaves gradient descent, sign descent,
    /// AdaGrad and Shampoo where they are, and NaN is carried.
    #[test]
    fn nothing_panics_on_any_setting() {
        let mut rng = Rng::new(18);
        let specs = [
            MlpSpec { inputs: 3, width: 5, depth: 3, outputs: 2, act: Act::Tanh, ..MlpSpec::default() },
            MlpSpec { inputs: 1, width: 1, depth: 3, outputs: 1, ..MlpSpec::default() },
            MlpSpec { inputs: 2, width: 6, depth: 2, outputs: 3, residual: true, norm: Norm::Layer, ..MlpSpec::default() },
        ];
        let mut tape = Tape::new();
        for (name, rule, h) in all_forms() {
            for spec in specs {
                let model = Model::mlp(&spec, &mut rng);
                // a batch of one: every weight gradient has rank 1
                let x = randn(spec.inputs, &mut rng);
                for case in 0..6 {
                    let mut m = model.clone();
                    let mut opt = Optimizer::new(&m, rule, h);
                    let lr = if case == 2 { 0.0 } else { h.lr };
                    for _ in 0..3 {
                        fill_grads(&mut m, &mut tape, &x, &[0.0]);
                        match case {
                            0 => m.zero_grads(),
                            3 => m.groups_mut()[0].grad[0] = f64::NAN,
                            4 => m.groups_mut().iter_mut().for_each(|g| g.grad.fill(f64::INFINITY)),
                            5 => m.groups_mut().iter_mut().for_each(|g| g.grad.iter_mut().for_each(|d| *d *= 1e-200)),
                            _ => {}
                        }
                        let before: Vec<Vec<f64>> = m.groups().iter().map(|g| g.value.clone()).collect();
                        opt.step(&mut m, lr);
                        let mut buf = Vec::new();
                        opt.preconditioner(&mut buf);
                        assert_eq!(buf.len(), m.n_params());
                        match case {
                            0 if matches!(rule, Rule::Sgd | Rule::SignSgd | Rule::AdaGrad | Rule::Shampoo) && h.weight_decay == WeightDecay::None => {
                                assert!(m.groups().iter().zip(&before).all(|(g, b)| g.value == *b), "{name}: a zero gradient moved it")
                            }
                            3 => assert!(m.groups()[0].value.iter().any(|v| v.is_nan()) || (rule == Rule::Soap && opt.steps() == 1), "{name}: NaN not carried"),
                            1 | 2 | 5 => assert!(m.groups().iter().all(|g| g.value.iter().all(|v| v.is_finite())), "{name} case {case}: not finite"),
                            _ => {}
                        }
                    }
                }
            }
            // a transformer
            let spec = TransformerSpec { vocab: 7, context: 6, d_model: 8, heads: 2, layers: 2, d_mlp: 12, ..TransformerSpec::default() };
            let mut model = Model::transformer(&spec, &mut rng);
            let mut opt = Optimizer::new(&model, rule, h);
            for _ in 0..3 {
                let (tokens, targets) = repeats(6, 7, &mut rng);
                let trace = model.loss_transformer(&mut tape, &tokens, &targets);
                tape.backward(trace.loss);
                model.zero_grads();
                model.accumulate(&tape, &trace);
                opt.step(&mut model, h.lr);
            }
            assert!(model.groups().iter().all(|g| g.value.iter().all(|v| v.is_finite())), "{name} on a transformer");
        }
    }

    /// Every rule at its default settings drives the loss of the spirals down from its
    /// start in a few hundred full-batch steps (a smoke test, not a comparison).
    #[test]
    fn every_rule_trains_the_spirals() {
        let (x, y) = spirals(150, 3, &mut Rng::new(19));
        let spec = MlpSpec { inputs: 2, width: 32, depth: 3, outputs: 3, act: Act::Tanh, init: Init::Xavier, ..MlpSpec::default() };
        let mut tape = Tape::new();
        for rule in Rule::ALL {
            let mut model = Model::mlp(&spec, &mut Rng::new(1));
            let h = Hyper::default_for(rule);
            let mut opt = Optimizer::new(&model, rule, h);
            let loss = |model: &Model, tape: &mut Tape| {
                let trace = model.loss_mlp(tape, &x, &y, Loss::CrossEntropy);
                tape.value(trace.loss)[0]
            };
            let first = loss(&model, &mut tape);
            for _ in 0..300 {
                fill_grads(&mut model, &mut tape, &x, &y);
                opt.step(&mut model, h.lr);
            }
            let last = loss(&model, &mut tape);
            println!("{:<17} loss {first:.4} to {last:.4}", rule.name());
            assert!(last < 0.99 * first, "{}: loss {first} to {last}", rule.name());
        }
    }

    /// The groups a matrix rule takes are the documented ones.
    #[test]
    fn groups_are_chosen_as_documented() {
        let mut rng = Rng::new(20);
        let names = |model: &Model, groups: Groups| {
            let opt = Optimizer::new(model, Rule::Muon, Hyper { groups, ..Hyper::muon_reference() });
            model.groups().iter().enumerate().filter(|(i, _)| opt.as_matrix(*i)).map(|(_, g)| g.name.clone()).collect::<Vec<_>>()
        };
        let plain = Model::mlp(&MlpSpec { depth: 3, ..MlpSpec::default() }, &mut rng);
        assert_eq!(names(&plain, Groups::Hidden), ["layer 2 weights", "layer 3 weights"]);
        assert_eq!(names(&plain, Groups::HiddenAndInput), ["layer 1 weights", "layer 2 weights", "layer 3 weights"]);
        assert_eq!(names(&plain, Groups::Matrices), ["layer 1 weights", "layer 2 weights", "layer 3 weights", "read-out weights"]);
        let residual = Model::mlp(&MlpSpec { depth: 2, residual: true, norm: Norm::Layer, ..MlpSpec::default() }, &mut rng);
        assert_eq!(names(&residual, Groups::Hidden), ["layer 1 weights", "layer 2 weights"]);
        assert_eq!(names(&residual, Groups::HiddenAndInput), ["input map weights", "layer 1 weights", "layer 2 weights"]);
        let t = Model::transformer(&TransformerSpec { layers: 1, heads: 1, ..TransformerSpec::default() }, &mut rng);
        assert_eq!(
            names(&t, Groups::Hidden),
            [
                "block 1 head 1 query",
                "block 1 head 1 key",
                "block 1 head 1 value",
                "block 1 head 1 output",
                "block 1 MLP in weights",
                "block 1 MLP out weights"
            ]
        );
        let all = names(&t, Groups::Matrices);
        assert!(
            all.contains(&"token embedding".to_string())
                && all.contains(&"unembedding".to_string())
                && !all.iter().any(|n| n.contains("bias") || n.contains("gain"))
        );
        // what the other groups take
        let opt = Optimizer::new(&plain, Rule::Muon, Hyper::muon_reference());
        let rules: Vec<Rule> = (0..plain.groups().len()).map(|i| opt.rule_of(i)).collect();
        assert_eq!(rules, [Rule::Adam, Rule::Adam, Rule::Muon, Rule::Adam, Rule::Muon, Rule::Adam, Rule::Adam, Rule::Adam]);
        let opt = Optimizer::new(&plain, Rule::Adam, Hyper::default_for(Rule::Adam));
        assert!((0..8).all(|i| opt.rule_of(i) == Rule::Adam && !opt.as_matrix(i)));
        assert_eq!(opt.update(99), &[] as &[f64]);
    }

    /// The schedules give the sources' formulas.
    #[test]
    fn schedules() {
        let c = Schedule::cosine(1.0, 0.1, 100);
        assert_eq!(c.at(0), 1.0);
        assert!((c.at(50) - 0.55).abs() < 1e-15 && (c.at(25) - (0.1 + 0.45 * (1.0 + (std::f64::consts::PI / 4.0).cos()))).abs() < 1e-15);
        assert_eq!(c.at(100), 0.1);
        assert_eq!(c.at(1000), 0.1);
        let l = Schedule::linear(0.4, 0.04, 10);
        assert!((l.at(5) - 0.22).abs() < 1e-15 && l.at(10) == 0.04 && l.at(11) == 0.04);
        let w = Schedule::warmup(1.0, 4);
        assert_eq!([w.at(0), w.at(1), w.at(2), w.at(4), w.at(9)], [0.0, 0.25, 0.5, 1.0, 1.0]);
        let both = Schedule::cosine(1.0, 0.0, 10).with_warmup(5, 0.5);
        assert_eq!([both.at(0), both.at(5), both.at(15)], [0.5, 1.0, 0.0]);
        assert!((both.at(10) - 0.5).abs() < 1e-15);
        assert_eq!(Schedule::constant(0.3).at(12345), 0.3);
        assert_eq!(Schedule::linear(1.0, 0.2, 0).at(0), 0.2);
        assert!(Schedule::constant(f64::NAN).at(3).is_nan());
    }

    /// The preconditioner is the number the step divided by (Cohen et al.'s P, with its
    /// factor 1 − β1^t for Adam with bias correction): update = −lr m / P for Adam,
    /// −lr g / P for RMSProp and AdaGrad; NaN for other rules.
    #[test]
    fn the_preconditioner_is_what_the_step_divided_by() {
        let (model, x, y) = bench(21);
        let mut tape = Tape::new();
        for (rule, h) in [
            (Rule::Adam, Hyper::default_for(Rule::Adam)),
            (Rule::Adam, Hyper { bias_correction: false, ..Hyper::default_for(Rule::Adam) }),
            (Rule::Adam, Hyper { amsgrad: true, ..Hyper::default_for(Rule::Adam) }),
            (Rule::RmsProp, Hyper { centered: true, ..Hyper::default_for(Rule::RmsProp) }),
            (Rule::AdaGrad, Hyper { lr_decay: 0.2, ..Hyper::default_for(Rule::AdaGrad) }),
            (Rule::Sgd, Hyper::default_for(Rule::Sgd)),
            (Rule::Muon, Hyper::muon_reference()),
        ] {
            let mut m = model.clone();
            let mut opt = Optimizer::new(&m, rule, h);
            let mut p = Vec::new();
            assert!(!opt.preconditioner(&mut p) && p.iter().all(|x| x.is_nan()));
            for _ in 0..3 {
                fill_grads(&mut m, &mut tape, &x, &y);
                opt.step(&mut m, 0.01);
            }
            let real = opt.preconditioner(&mut p);
            assert_eq!(real, matches!(rule, Rule::Adam | Rule::RmsProp | Rule::AdaGrad));
            if !real {
                continue;
            }
            let mut k = 0;
            for (i, g) in m.groups().iter().enumerate() {
                for j in 0..g.value.len() {
                    let top = if rule == Rule::Adam { opt.momentum(i)[j] } else { g.grad[j] };
                    let want = -0.01 * top / p[k];
                    assert!((opt.update(i)[j] - want).abs() < 1e-13 * want.abs().max(1e-12), "{} {h:?}", rule.name());
                    k += 1;
                }
            }
        }
    }

    /// The cost of a step of each rule on a 6-layer 32-wide MLP and a 4-block 32-wide
    /// transformer, and of the orthogonalisation alone. Run with `cargo test --profile
    /// fast --lib the_cost_of_a_step -- --ignored --nocapture`. Each figure is the best of
    /// five runs of a few hundred steps, and the median of the five beside it.
    #[test]
    #[ignore]
    fn the_cost_of_a_step() {
        let time = |mut f: Box<dyn FnMut() + '_>| {
            for _ in 0..30 {
                f();
            }
            let mut runs: Vec<f64> = (0..5)
                .map(|_| {
                    let start = std::time::Instant::now();
                    for _ in 0..100 {
                        f();
                    }
                    start.elapsed().as_secs_f64() * 1e6 / 100.0
                })
                .collect();
            runs.sort_by(f64::total_cmp);
            (runs[0], runs[2])
        };
        let mut rng = Rng::new(22);
        let mut tape = Tape::new();
        let mlp = MlpSpec { inputs: 2, width: 32, depth: 6, outputs: 3, ..MlpSpec::default() };
        let mut models: Vec<(String, Model)> = vec![("MLP 6 x 32".into(), Model::mlp(&mlp, &mut rng))];
        for (heads, d_mlp) in [(2, 64), (4, 128)] {
            let spec = TransformerSpec { layers: 4, d_model: 32, heads, d_mlp, vocab: 16, context: 12, ..TransformerSpec::default() };
            models.push((format!("transformer 4 x 32, {heads} heads, MLP {d_mlp}"), Model::transformer(&spec, &mut rng)));
        }
        let (x, y) = spirals(16, 3, &mut rng);
        for (name, model) in &mut models {
            if model.mlp_spec().is_some() {
                fill_grads(model, &mut tape, &x, &y);
            } else {
                let (tokens, targets) = repeats(12, 16, &mut rng);
                let trace = model.loss_transformer(&mut tape, &tokens, &targets);
                tape.backward(trace.loss);
                model.zero_grads();
                model.accumulate(&tape, &trace);
            }
            println!("{name}:");
            for rule in Rule::ALL {
                let h = Hyper::default_for(rule);
                let mut m = model.clone();
                let mut opt = Optimizer::new(&m, rule, h);
                let (best, median) = time(Box::new(|| opt.step(&mut m, 1e-6)));
                println!("  {:<17} {best:>7.1} us (median {median:.1})", rule.name());
            }
            // the orthogonalisation alone, on the hidden matrices' gradients
            let opt = Optimizer::new(model, Rule::Muon, Hyper::muon_reference());
            let mats: Vec<(Vec<f64>, usize, usize)> =
                model.groups().iter().enumerate().filter(|(i, _)| opt.as_matrix(*i)).map(|(_, g)| (g.grad.clone(), g.rows, g.cols)).collect();
            let mut s = Scratch::new();
            let mut work: Vec<Vec<f64>> = mats.iter().map(|(g, _, _)| g.clone()).collect();
            for (label, ortho) in [("5 rounds of Jordan", Ortho::Polynomial(Iteration::JORDAN)), ("exact polar factor", Ortho::Exact)] {
                let (best, median) = time(Box::new(|| {
                    for ((g, r, c), w) in mats.iter().zip(work.iter_mut()) {
                        match ortho {
                            Ortho::Polynomial(it) => {
                                w.copy_from_slice(g);
                                it.run(w, *r, *c, &mut s);
                            }
                            Ortho::Exact => {
                                polar_into(g, *r, *c, w, &mut s);
                            }
                        }
                    }
                }));
                println!("  ({} on the {} hidden matrices: {best:.1} us, median {median:.1})", label, mats.len());
            }
        }
    }

    impl Optimizer {
        /// Where every buffer is and how big it is (a step that allocated would move or
        /// grow one).
        fn buffers(&self) -> Vec<(usize, usize)> {
            let sc = &self.scratch;
            let mut out: Vec<(usize, usize)> =
                [&self.g, &sc.g, &sc.vt, &sc.w, &sc.vals, &sc.x, &sc.y, &sc.z].iter().map(|v| (v.as_ptr() as usize, v.capacity())).collect();
            for s in &self.slots {
                for v in [&s.update, &s.decay, &s.m, &s.v, &s.w, &s.mhat, &s.vhat, &s.ortho_in, &s.ortho_out] {
                    out.push((v.as_ptr() as usize, v.capacity()));
                }
                for side in [&s.left, &s.right].into_iter().flatten() {
                    for v in [&side.f, &side.vt, &side.vals, &side.root] {
                        out.push((v.as_ptr() as usize, v.capacity()));
                    }
                }
            }
            out
        }
    }

    /// A step allocates nothing: every buffer the optimiser and its scratch hold stays
    /// where it is, at its size, from the first step on (every rule and form, on an MLP
    /// and a transformer).
    #[test]
    fn a_step_allocates_nothing() {
        let mut rng = Rng::new(23);
        let mut tape = Tape::new();
        let (x, y) = spirals(12, 3, &mut rng);
        let spec = MlpSpec { inputs: 2, width: 9, depth: 3, outputs: 3, ..MlpSpec::default() };
        let tspec = TransformerSpec { vocab: 7, context: 6, d_model: 8, heads: 2, layers: 1, d_mlp: 12, ..TransformerSpec::default() };
        for (name, rule, h) in all_forms() {
            for transformer in [false, true] {
                let mut model = if transformer { Model::transformer(&tspec, &mut rng) } else { Model::mlp(&spec, &mut rng) };
                let mut opt = Optimizer::new(&model, rule, Hyper { precondition_every: 2, root_every: 1, ..h });
                let before = opt.buffers();
                for _ in 0..5 {
                    if transformer {
                        let (tokens, targets) = repeats(6, 7, &mut rng);
                        let trace = model.loss_transformer(&mut tape, &tokens, &targets);
                        tape.backward(trace.loss);
                        model.zero_grads();
                        model.accumulate(&tape, &trace);
                    } else {
                        fill_grads(&mut model, &mut tape, &x, &y);
                    }
                    opt.step(&mut model, h.lr);
                    assert_eq!(opt.buffers(), before, "{name}: a buffer moved or grew");
                }
            }
        }
    }
}

//! Sparse recovery: finding the answer with the fewest non-zero entries (for the
//! compressed-sensing scene).
//!
//! Every matrix here is m × n and row-major, usually with fewer rows than columns: m
//! equations y = A x in n unknowns, which infinitely many x satisfy. Two of them are
//! of interest. The shortest (least 2-norm) has a formula, x = Aᵀ(AAᵀ)⁻¹y. The one
//! with the least 1-norm Σ|xᵢ| ("basis pursuit") is the answer to a linear
//! programme: it is found exactly by following the path of the lasso down to zero
//! penalty ([`Homotopy`]), which also hands back the proof that nothing has a
//! smaller 1-norm (a dual certificate, [`Solution::nu`]). The path it follows can be
//! drawn as a road from the shortest solution to that one (see [`Homotopy::path`]).

use crate::math::rng::Rng;

/// The dot product, in four running sums so that the compiler can use vector
/// instructions (one sum must add in order, which it may not reorder).
#[inline]
fn dot(a: &[f64], b: &[f64]) -> f64 {
    let n = a.len().min(b.len());
    let (a, b) = (&a[..n], &b[..n]);
    let mut s = [0.0f64; 4];
    for (p, q) in a.chunks_exact(4).zip(b.chunks_exact(4)) {
        for l in 0..4 {
            s[l] += p[l] * q[l];
        }
    }
    let tail = n - n % 4;
    let mut t = (s[0] + s[1]) + (s[2] + s[3]);
    for i in tail..n {
        t += a[i] * b[i];
    }
    t
}

/// `out += k v`.
#[inline]
fn add_scaled(out: &mut [f64], k: f64, v: &[f64]) {
    out.iter_mut().zip(v).for_each(|(o, v)| *o += k * v);
}

/// Factor a symmetric positive definite `n` × `n` matrix (row-major, rows `stride`
/// long) as L Lᵀ, in place: L is left in the lower triangle. False if it is not
/// positive definite (to working precision).
fn cholesky(g: &mut [f64], n: usize, stride: usize) -> bool {
    for j in 0..n {
        let row_j = j * stride;
        let d = g[row_j + j] - dot(&g[row_j..row_j + j], &g[row_j..row_j + j]);
        if !(d > 0.0 && d.is_finite()) {
            return false;
        }
        let d = d.sqrt();
        g[row_j + j] = d;
        for i in j + 1..n {
            let row_i = i * stride;
            let s = g[row_i + j] - dot(&g[row_i..row_i + j], &g[row_j..row_j + j]);
            g[row_i + j] = s / d;
        }
    }
    true
}

/// Solve L Lᵀ x = b in place, with L from [`cholesky`].
fn cholesky_solve(l: &[f64], n: usize, stride: usize, b: &mut [f64]) {
    for i in 0..n {
        let row = &l[i * stride..i * stride + i];
        b[i] = (b[i] - dot(row, &b[..i])) / l[i * stride + i];
    }
    for i in (0..n).rev() {
        let mut s = b[i];
        for k in i + 1..n {
            s -= l[k * stride + i] * b[k];
        }
        b[i] = s / l[i * stride + i];
    }
}

/// The shortest solution of A x = r for any r, with A Aᵀ factored once (for a matrix
/// whose rows are independent).
pub struct Shortest {
    m: usize,
    n: usize,
    l: Vec<f64>,
}

impl Shortest {
    pub fn of(a: &[f64], m: usize, n: usize) -> Option<Shortest> {
        let mut l = vec![0.0; m * m];
        for i in 0..m {
            for k in 0..=i {
                let s = dot(&a[i * n..(i + 1) * n], &a[k * n..(k + 1) * n]);
                l[i * m + k] = s;
                l[k * m + i] = s;
            }
        }
        cholesky(&mut l, m, m).then_some(Shortest { m, n, l })
    }

    /// x = Aᵀ (A Aᵀ)⁻¹ r.
    pub fn solve(&self, a: &[f64], r: &[f64]) -> Vec<f64> {
        let mut lambda = r[..self.m].to_vec();
        cholesky_solve(&self.l, self.m, self.m, &mut lambda);
        let mut x = vec![0.0; self.n];
        apply_t(a, self.m, self.n, &lambda, &mut x);
        x
    }
}

/// The 1-norm Σ|xᵢ|.
pub fn norm1(x: &[f64]) -> f64 {
    x.iter().map(|v| v.abs()).sum()
}

/// The 2-norm (length).
pub fn norm2(x: &[f64]) -> f64 {
    dot(x, x).sqrt()
}

/// `out = A x`.
pub fn apply(a: &[f64], m: usize, n: usize, x: &[f64], out: &mut [f64]) {
    for (i, o) in out[..m].iter_mut().enumerate() {
        *o = dot(&a[i * n..(i + 1) * n], x);
    }
}

/// `out = Aᵀ v`.
pub fn apply_t(a: &[f64], m: usize, n: usize, v: &[f64], out: &mut [f64]) {
    out[..n].fill(0.0);
    for i in 0..m {
        let vi = v[i];
        if vi != 0.0 {
            out[..n].iter_mut().zip(&a[i * n..(i + 1) * n]).for_each(|(o, a)| *o += a * vi);
        }
    }
}

/// Basis pursuit's answer, with the proof that it is one.
#[derive(Clone, Debug)]
pub struct Solution {
    pub x: Vec<f64>,
    /// A dual certificate: |(Aᵀν)ᵢ| ≤ 1 for every i, and (Aᵀν)ᵢ = sign(xᵢ) wherever
    /// xᵢ ≠ 0. Then for any z with A z = y, ‖z‖₁ ≥ Σ zᵢ(Aᵀν)ᵢ = yᵀν = ‖x‖₁, so no
    /// solution of the equations has a smaller 1-norm than x. (See [`Check`].)
    pub nu: Vec<f64>,
    /// Steps the solver took.
    pub steps: usize,
}

/// How well a [`Solution`] proves itself, all of it measured.
#[derive(Clone, Copy, Debug)]
pub struct Check {
    /// ‖Ax − y‖ / ‖y‖: does it agree with the measurements?
    pub residual: f64,
    /// max |(Aᵀν)ᵢ|: at most 1 for a certificate.
    pub dual: f64,
    /// The largest |(Aᵀν)ᵢ − sign(xᵢ)| over the entries that are not zero.
    pub sign_gap: f64,
    /// ‖x‖₁ minus the lower bound the certificate proves for every solution
    /// (yᵀν / max(1, max |(Aᵀν)ᵢ|)), as a share of ‖x‖₁: 0 for a proven minimiser.
    pub gap: f64,
}

impl Check {
    pub fn of(a: &[f64], m: usize, n: usize, y: &[f64], s: &Solution) -> Check {
        let mut ax = vec![0.0; m];
        apply(a, m, n, &s.x, &mut ax);
        let miss: Vec<f64> = ax.iter().zip(y).map(|(p, q)| p - q).collect();
        let residual = norm2(&miss) / norm2(&y[..m]).max(1e-300);
        let mut atnu = vec![0.0; n];
        apply_t(a, m, n, &s.nu, &mut atnu);
        let dual = atnu.iter().fold(0.0f64, |d, v| d.max(v.abs()));
        let big = s.x.iter().fold(0.0f64, |d, v| d.max(v.abs()));
        let sign_gap = s.x.iter().zip(&atnu).filter(|(x, _)| x.abs() > 1e-9 * big).fold(0.0f64, |g, (x, t)| g.max((t - x.signum()).abs()));
        let bound = dot(&y[..m], &s.nu) / dual.max(1.0);
        let l1 = norm1(&s.x);
        Check { residual, dual, sign_gap, gap: if l1 > 0.0 { (l1 - bound) / l1 } else { 0.0 } }
    }

    /// Is the solution proven to be a least-1-norm solution, to within rounding?
    pub fn proven(&self, tol: f64) -> bool {
        self.residual < tol && self.dual < 1.0 + tol && self.sign_gap < tol && self.gap.abs() < tol
    }
}

/// Where a [`Homotopy`] is.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Status {
    Running,
    Done,
    /// Something only a degenerate matrix does (two columns that are the same, say):
    /// the path could not be followed.
    Failed,
}

/// Basis pursuit, solved exactly by homotopy: the answer to "minimise
/// ½‖y − Ax‖² + λ‖x‖₁" is followed as λ comes down from the largest value at which
/// it is still zero to 0, where it becomes the least-1-norm solution of A x = y.
/// Along the way the answer changes linearly between events at which one entry
/// becomes non-zero (it "joins") or returns to zero; each step goes to the next event.
/// When basis pursuit recovers a k-sparse signal this takes k steps or a few more
/// (entries that join on the way can leave again: 8 to 38 steps, typically 12, for
/// 8 non-zero entries of 100 from 40 measurements); when it does not, about as many as
/// there are measurements, or more.
///
/// The work is a step at a time (see [`Homotopy::step`]), so that it can be spread
/// over frames; [`basis_pursuit`] runs it to the end. The problem is copied in, with
/// A stored a column at a time, which is how every step reads it.
pub struct Homotopy {
    m: usize,
    n: usize,
    /// The columns of A, one after the other, and y.
    cols: Vec<f64>,
    y: Vec<f64>,
    x: Vec<f64>,
    lambda: f64,
    /// λ at the start: the problem's scale, against which steps count as zero.
    start: f64,
    active: Vec<usize>,
    /// Which entries are active, by entry.
    on: Vec<bool>,
    signs: Vec<f64>,
    /// The Gram matrix of the active columns and its Cholesky factor, rows `cap` long.
    gram: Vec<f64>,
    chol: Vec<f64>,
    cap: usize,
    /// Squared length of each column, and whether it may join (a column of zeros
    /// never does).
    lengths: Vec<f64>,
    usable: Vec<bool>,
    /// The entry that has just returned to zero: it cannot join again at once.
    dropped: Option<usize>,
    steps: usize,
    status: Status,
    nu: Vec<f64>,
    /// The answer at λ = λ₀ (zero) and after every step, one after the other, and λ
    /// at each, if asked for.
    path: Option<Vec<f64>>,
    lambdas: Vec<f64>,
    // scratch
    r: Vec<f64>,
    c: Vec<f64>,
    d: Vec<f64>,
    v: Vec<f64>,
    av: Vec<f64>,
}

/// What ends a step of the homotopy.
enum Event {
    End,
    Join(usize, f64),
    Drop(usize),
}

impl Homotopy {
    pub fn new(a: &[f64], m: usize, n: usize, y: &[f64]) -> Homotopy {
        let mut cols = vec![0.0; n * m];
        for i in 0..m {
            for j in 0..n {
                cols[j * m + i] = a[i * n + j];
            }
        }
        let col = |j: usize| &cols[j * m..(j + 1) * m];
        let lengths: Vec<f64> = (0..n).map(|j| dot(col(j), col(j))).collect();
        let c: Vec<f64> = (0..n).map(|j| dot(col(j), &y[..m])).collect();
        let longest = lengths.iter().fold(0.0f64, |l, v| l.max(*v));
        let usable: Vec<bool> = lengths.iter().map(|&l| l > 1e-24 * longest && l > 0.0).collect();
        let first = (0..n).filter(|&j| usable[j]).max_by(|&i, &j| c[i].abs().total_cmp(&c[j].abs()));
        let cap = m.min(n).max(1);
        let mut h = Homotopy {
            m,
            n,
            cols,
            y: y[..m].to_vec(),
            x: vec![0.0; n],
            lambda: 0.0,
            start: 0.0,
            active: Vec::with_capacity(cap),
            on: vec![false; n],
            signs: Vec::with_capacity(cap),
            gram: vec![0.0; cap * cap],
            chol: vec![0.0; cap * cap],
            cap,
            lengths,
            usable,
            dropped: None,
            steps: 0,
            status: Status::Running,
            nu: vec![0.0; m],
            path: None,
            lambdas: Vec::new(),
            r: vec![0.0; m],
            c,
            d: vec![0.0; cap],
            v: vec![0.0; m],
            av: vec![0.0; n],
        };
        match first {
            Some(j) if h.c[j].abs() > 1e-300 && h.c[j].is_finite() => {
                h.lambda = h.c[j].abs();
                h.start = h.lambda;
                h.active.push(j);
                h.on[j] = true;
                h.signs.push(h.c[j].signum());
                h.gram[0] = h.lengths[j];
                h.chol[0] = h.lengths[j].sqrt();
            }
            // nothing measured (y = 0, or A = 0): the answer is zero
            _ => h.status = Status::Done,
        }
        h
    }

    /// Keep the answer after every step (see [`Homotopy::path`]).
    pub fn recording(mut self) -> Homotopy {
        self.path = Some(self.x.clone());
        self.lambdas = vec![self.lambda];
        self
    }

    /// The answers at the events so far, n numbers each: zero at the start, then one
    /// after every step, the last (when done) the least-1-norm solution. Between two
    /// events the answer moves in a straight line. Empty unless recording.
    pub fn path(&self) -> &[f64] {
        self.path.as_deref().unwrap_or(&[])
    }

    /// λ at each point of [`Homotopy::path`]: falling, the last 0.
    pub fn lambdas(&self) -> &[f64] {
        &self.lambdas
    }

    pub fn status(&self) -> Status {
        self.status
    }

    pub fn steps(&self) -> usize {
        self.steps
    }

    /// The answer, once [`Status::Done`].
    pub fn solution(&self) -> Option<Solution> {
        (self.status == Status::Done).then(|| Solution { x: self.x.clone(), nu: self.nu.clone(), steps: self.steps })
    }

    fn col(&self, j: usize) -> &[f64] {
        &self.cols[j * self.m..(j + 1) * self.m]
    }

    /// Go on to the next event; returns where things stand.
    pub fn step(&mut self) -> Status {
        if self.status != Status::Running {
            return self.status;
        }
        let status = self.advance();
        if let Some(path) = &mut self.path {
            if status != Status::Failed {
                path.extend_from_slice(&self.x);
                self.lambdas.push(self.lambda);
            }
        }
        status
    }

    fn advance(&mut self) -> Status {
        let (n, s) = (self.n, self.active.len());
        self.steps += 1;
        // (the path has at most a few times n events; more means it is going round)
        if self.steps > 8 * n + 64 {
            self.status = Status::Failed;
            return self.status;
        }
        // the correlations of the columns with what is not yet explained: carried from
        // step to step (they change in a straight line too), and worked out afresh every
        // few steps so that rounding does not build up
        if self.steps % 8 == 1 {
            let mut r = std::mem::take(&mut self.r);
            r.copy_from_slice(&self.y);
            for &j in &self.active {
                add_scaled(&mut r, -self.x[j], self.col(j));
            }
            for j in 0..n {
                self.c[j] = dot(self.col(j), &r);
            }
            self.r = r;
        }
        // the direction: as λ falls by γ, the active entries change by γ d, keeping their
        // correlations at ±λ
        self.d[..s].copy_from_slice(&self.signs);
        cholesky_solve(&self.chol, s, self.cap, &mut self.d[..s]);
        let mut v = std::mem::take(&mut self.v);
        v.fill(0.0);
        for (&j, &dj) in self.active.iter().zip(&self.d[..s]) {
            add_scaled(&mut v, dj, self.col(j));
        }
        for j in 0..n {
            self.av[j] = dot(self.col(j), &v);
        }
        self.v = v;

        let floor = 1e-12 * self.start;
        let lambda = self.lambda;
        let (mut best, mut event) = (lambda, Event::End);
        for j in 0..n {
            if !self.usable[j] || self.on[j] || Some(j) == self.dropped {
                continue;
            }
            let (cj, aj) = (self.c[j], self.av[j]);
            // it joins when its correlation reaches +λ or −λ, both falling as λ does
            if 1.0 - aj > 1e-12 {
                let gamma = (lambda - cj) / (1.0 - aj);
                if gamma > floor && gamma < best {
                    (best, event) = (gamma, Event::Join(j, 1.0));
                }
            }
            if 1.0 + aj > 1e-12 {
                let gamma = (lambda + cj) / (1.0 + aj);
                if gamma > floor && gamma < best {
                    (best, event) = (gamma, Event::Join(j, -1.0));
                }
            }
        }
        for (idx, &j) in self.active.iter().enumerate() {
            let dj = self.d[idx];
            if dj != 0.0 {
                let gamma = -self.x[j] / dj;
                if gamma > floor && gamma < best {
                    (best, event) = (gamma, Event::Drop(idx));
                }
            }
        }
        // (an event that comes at the very end is the end)
        if best > lambda - 1e-11 * self.start {
            (best, event) = (lambda, Event::End);
        }
        for (&j, &dj) in self.active.iter().zip(&self.d[..s]) {
            self.x[j] += best * dj;
        }
        add_scaled(&mut self.c, -best, &self.av);
        self.lambda -= best;
        let cap = self.cap;
        match event {
            Event::End => self.finish(),
            Event::Join(j, sign) => {
                if s >= cap {
                    self.status = Status::Failed;
                    return self.status;
                }
                // the new column's row of the Gram matrix, and the factor grown by one row
                for idx in 0..s {
                    let g = dot(self.col(self.active[idx]), self.col(j));
                    self.gram[s * cap + idx] = g;
                    self.gram[idx * cap + s] = g;
                }
                self.gram[s * cap + s] = self.lengths[j];
                let mut row: Vec<f64> = self.gram[s * cap..s * cap + s].to_vec();
                for i in 0..s {
                    row[i] = (row[i] - dot(&self.chol[i * cap..i * cap + i], &row[..i])) / self.chol[i * cap + i];
                }
                let last = self.lengths[j] - dot(&row, &row);
                if last.is_nan() || last <= 1e-12 * self.lengths[j] {
                    // the new column is (to working precision) a combination of the others
                    self.status = Status::Failed;
                    return self.status;
                }
                self.chol[s * cap..s * cap + s].copy_from_slice(&row);
                self.chol[s * cap + s] = last.sqrt();
                self.active.push(j);
                self.on[j] = true;
                self.signs.push(sign);
                self.dropped = None;
            }
            Event::Drop(idx) => {
                let j = self.active.remove(idx);
                self.signs.remove(idx);
                self.on[j] = false;
                self.x[j] = 0.0;
                // the Gram matrix without that row and column, and its factor afresh
                for p in 0..s {
                    let from_p = p + (p >= idx) as usize;
                    if from_p >= s {
                        break;
                    }
                    for q in 0..s - 1 {
                        let from_q = q + (q >= idx) as usize;
                        self.gram[p * cap + q] = self.gram[from_p * cap + from_q];
                    }
                }
                self.chol[..(s - 1) * cap].copy_from_slice(&self.gram[..(s - 1) * cap]);
                if !cholesky(&mut self.chol, s - 1, cap) {
                    self.status = Status::Failed;
                    return self.status;
                }
                self.dropped = Some(j);
                if self.active.is_empty() {
                    self.status = Status::Failed;
                }
            }
        }
        self.status
    }

    /// At λ = 0: the active entries exactly (least squares on them, which the
    /// measurements now fit exactly), and the certificate ν = A_S (A_Sᵀ A_S)⁻¹ sign.
    fn finish(&mut self) {
        let s = self.active.len();
        let mut rhs: Vec<f64> = self.active.iter().map(|&j| dot(self.col(j), &self.y)).collect();
        cholesky_solve(&self.chol, s, self.cap, &mut rhs);
        for (&j, &v) in self.active.iter().zip(&rhs) {
            self.x[j] = v;
        }
        self.nu.copy_from_slice(&self.v);
        self.lambda = 0.0;
        // An entry can reach zero exactly as λ does (one that joined on the way down
        // and is not in the answer, when the signal is recovered): it is zero, and the
        // certificate still holds for it (|(Aᵀν)ᵢ| = 1 is allowed where xᵢ = 0). An
        // entry of any size with the wrong sign would mean an event was missed: then
        // this is no answer.
        let big = self.x.iter().fold(0.0f64, |b, v| b.max(v.abs()));
        let mut ok = self.x.iter().all(|v| v.is_finite());
        for (&j, &sign) in self.active.iter().zip(&self.signs) {
            if self.x[j].abs() <= 1e-10 * big {
                self.x[j] = 0.0;
            } else if self.x[j] * sign < 0.0 {
                ok = false;
            }
        }
        self.status = if ok { Status::Done } else { Status::Failed };
    }
}

/// Basis pursuit, run to the end: the least-1-norm solution of A x = y and its
/// certificate. (None only for a degenerate matrix, two columns the same, say: never
/// one with random entries, nor one that reads entries.)
pub fn basis_pursuit(a: &[f64], m: usize, n: usize, y: &[f64]) -> Option<Solution> {
    let mut h = Homotopy::new(a, m, n, y);
    while h.step() == Status::Running {}
    h.solution()
}

/// The mutual coherence of A: the largest |cos| of the angle between two of its
/// columns, the most alike pair. None if a column is all zeros (that entry of x is
/// never measured at all), or there are fewer than two columns.
pub fn coherence(a: &[f64], m: usize, n: usize) -> Option<f64> {
    if n < 2 {
        return None;
    }
    // the columns, scaled to length 1, side by side in rows
    let mut cols = vec![0.0; n * m];
    for i in 0..m {
        for j in 0..n {
            cols[j * m + i] = a[i * n + j];
        }
    }
    for j in 0..n {
        let col = &mut cols[j * m..(j + 1) * m];
        let len = norm2(col);
        if len.is_nan() || len <= 1e-150 {
            return None;
        }
        col.iter_mut().for_each(|v| *v /= len);
    }
    let mut worst = 0.0f64;
    for i in 0..n {
        for j in 0..i {
            worst = worst.max(dot(&cols[i * m..(i + 1) * m], &cols[j * m..(j + 1) * m]).abs());
        }
    }
    Some(worst.min(1.0))
}

/// The Welch bound: no m × n matrix (m ≤ n) has columns less alike than this,
/// √((n − m) / (m (n − 1))).
pub fn welch_bound(m: usize, n: usize) -> f64 {
    if m == 0 || n < 2 || m >= n {
        return 0.0;
    }
    (((n - m) as f64) / ((m * (n - 1)) as f64)).sqrt()
}

/// The classical guarantee from coherence μ: every signal with fewer than
/// (1 + 1/μ) / 2 non-zero entries is recovered by basis pursuit, whatever they are.
/// Returns the largest such number of non-zeros (capped at `n`).
pub fn guaranteed(mu: f64, n: usize) -> usize {
    if mu.is_nan() || mu <= 1e-12 {
        return n;
    }
    let bound = (1.0 + 1.0 / mu) / 2.0;
    // the largest whole number strictly below the bound
    let k = (bound - 1e-12).ceil() - 1.0;
    (k.max(0.0) as usize).min(n)
}

/// Fill `row` with independent standard normal numbers drawn by `rng`.
pub fn normals(row: &mut [f64], rng: &mut Rng) {
    row.iter_mut().for_each(|v| *v = rng.normal() as f64);
}

/// A k-sparse signal of length n: the non-zero entries at `k` positions picked at
/// random, each of random sign and size between 0.4 and 1. Drawn so that the same
/// seed with a larger k keeps the same entries and adds more.
pub fn sparse_signal(n: usize, k: usize, rng: &mut Rng) -> Vec<f64> {
    let mut order: Vec<usize> = (0..n).collect();
    rng.shuffle(&mut order);
    let mut x = vec![0.0; n];
    for &i in order.iter().take(k.min(n)) {
        let size = 0.4 + 0.6 * rng.f32() as f64;
        x[i] = if rng.bool(0.5) { size } else { -size };
    }
    x
}

/// An m × n matrix of independent normal numbers with variance 1/m (so that each
/// column has length about 1).
pub fn gaussian_matrix(m: usize, n: usize, rng: &mut Rng) -> Vec<f64> {
    let mut a = vec![0.0; m * n];
    normals(&mut a, rng);
    let scale = 1.0 / (m.max(1) as f64).sqrt();
    a.iter_mut().for_each(|v| *v *= scale);
    a
}

/// The first m rows of the n × n identity: reading the first m entries of x directly.
pub fn first_rows(m: usize, n: usize) -> Vec<f64> {
    let mut a = vec![0.0; m * n];
    for i in 0..m.min(n) {
        a[i * n + i] = 1.0;
    }
    a
}

/// How far `x` is from `truth`, as a share of the truth's length (the distance
/// itself if the truth is zero).
pub fn relative_error(x: &[f64], truth: &[f64]) -> f64 {
    let diff: f64 = x.iter().zip(truth).map(|(p, q)| (p - q) * (p - q)).sum::<f64>().sqrt();
    let size = norm2(truth);
    if size > 0.0 {
        diff / size
    } else {
        diff
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A random problem: A (m × n Gaussian), a k-sparse x and y = A x.
    fn problem(m: usize, n: usize, k: usize, seed: u64) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
        let mut rng = Rng::new(seed);
        let a = gaussian_matrix(m, n, &mut rng);
        let x = sparse_signal(n, k, &mut rng);
        let mut y = vec![0.0; m];
        apply(&a, m, n, &x, &mut y);
        (a, x, y)
    }

    /// The least 1-norm over all solutions, by brute force: the minimum of a linear
    /// programme is at a vertex, and the vertices of {x : A x = y} written as
    /// x = u − v with u, v ≥ 0 have their non-zero entries on m independent columns.
    /// So try every m columns, solve exactly on them, and keep the smallest 1-norm.
    fn brute_force(a: &[f64], m: usize, n: usize, y: &[f64]) -> f64 {
        let mut best = f64::INFINITY;
        let mut pick: Vec<usize> = (0..m).collect();
        loop {
            let mut sq: Vec<f64> = (0..m).flat_map(|i| pick.iter().map(move |&j| (i, j))).map(|(i, j)| a[i * n + j]).collect();
            if let Some(z) = crate::math::linalg::solve(&mut sq, &mut y.to_vec(), m) {
                best = best.min(norm1(&z));
            }
            // the next m-subset in order
            let mut i = m;
            while i > 0 && pick[i - 1] == n - m + i - 1 {
                i -= 1;
            }
            if i == 0 {
                return best;
            }
            pick[i - 1] += 1;
            for t in i..m {
                pick[t] = pick[t - 1] + 1;
            }
        }
    }

    /// The homotopy's answer agrees with the measurements and carries a certificate
    /// that holds: |Aᵀν| ≤ 1 everywhere, = sign(x) on the answer's support, and
    /// ‖x‖₁ = yᵀν. That alone proves it is a least-1-norm solution, in cases that
    /// recover the signal and in cases that do not.
    #[test]
    fn the_answer_comes_with_a_proof() {
        let (mut recovered, mut missed) = (0, 0);
        for (m, n, k) in [
            (20, 64, 3),
            (20, 64, 8),
            (20, 64, 15),
            (40, 100, 8),
            (25, 100, 8),
            (40, 100, 25),
            (64, 128, 20),
            (100, 128, 60),
            (8, 32, 6),
            (1, 16, 2),
            (32, 32, 16),
        ] {
            for seed in 0..8u64 {
                let (a, truth, y) = problem(m, n, k, 1000 * seed + (m * n + k) as u64);
                let mut h = Homotopy::new(&a, m, n, &y);
                while h.step() == Status::Running {}
                assert_eq!(h.status(), Status::Done, "m {m}, n {n}, k {k}, seed {seed}");
                let s = h.solution().unwrap();
                let c = Check::of(&a, m, n, &y, &s);
                assert!(c.proven(1e-8), "m {m}, n {n}, k {k}, seed {seed}: {c:?}");
                // at most m non-zero entries: a vertex
                assert!(s.x.iter().filter(|v| v.abs() > 1e-12).count() <= m);
                if relative_error(&s.x, &truth) < 1e-4 {
                    recovered += 1;
                } else {
                    missed += 1;
                    // a failure has a strictly smaller 1-norm than the signal
                    assert!(norm1(&s.x) < norm1(&truth) * (1.0 - 1e-9), "m {m}, n {n}, k {k}, seed {seed}");
                }
            }
        }
        // both kinds of case were tried
        assert!(recovered > 20 && missed > 20, "recovered {recovered}, missed {missed}");
    }

    /// Against brute force on problems small enough to try every vertex, recovered
    /// or not, the homotopy's 1-norm is the least there is.
    #[test]
    fn nothing_has_a_smaller_one_norm() {
        for (m, n, k) in [(3, 8, 1), (3, 8, 2), (4, 10, 2), (4, 10, 3), (5, 12, 4), (2, 9, 2), (6, 12, 5)] {
            for seed in 0..6u64 {
                let (a, _, y) = problem(m, n, k, 77 + seed * 31 + n as u64);
                let s = basis_pursuit(&a, m, n, &y).unwrap();
                let best = brute_force(&a, m, n, &y);
                assert!((norm1(&s.x) - best).abs() < 1e-9 * best, "m {m}, n {n}, k {k}, seed {seed}: {} against {best}", norm1(&s.x));
            }
        }
    }

    /// Signals sparse enough for their number of measurements come back exactly:
    /// the right positions and the right values.
    #[test]
    fn a_sparse_signal_comes_back_exactly() {
        for seed in 0..10u64 {
            let (a, truth, y) = problem(40, 100, 5, seed);
            let s = basis_pursuit(&a, 40, 100, &y).unwrap();
            assert!(relative_error(&s.x, &truth) < 1e-12, "seed {seed}: {}", relative_error(&s.x, &truth));
            // about k steps
            assert!(s.steps <= 8, "seed {seed}: {} steps", s.steps);
        }
        // one measurement of one spike: found if its column is the most correlated
        let a = vec![0.1, 0.9, -0.3];
        let s = basis_pursuit(&a, 1, 3, &[0.45]).unwrap();
        assert!((s.x[1] - 0.5).abs() < 1e-15 && s.x[0] == 0.0 && s.x[2] == 0.0, "{:?}", s.x);
    }

    /// Reading entries directly: the zero columns are never used, the entries read
    /// come back and the rest stay zero.
    #[test]
    fn reading_entries_gives_back_what_was_read() {
        let (m, n) = (10, 30);
        let a = first_rows(m, n);
        let mut x = vec![0.0; n];
        (x[2], x[7], x[20]) = (0.5, -0.8, 0.9);
        let mut y = vec![0.0; m];
        apply(&a, m, n, &x, &mut y);
        let s = basis_pursuit(&a, m, n, &y).unwrap();
        assert!(Check::of(&a, m, n, &y, &s).proven(1e-12));
        assert_eq!((s.x[2], s.x[7], s.x[20]), (0.5, -0.8, 0.0));
        assert_eq!(coherence(&a, m, n), None, "columns of zeros have no direction");
        // and nothing at all measured
        let s = basis_pursuit(&a, m, n, &vec![0.0; m]).unwrap();
        assert!(s.x.iter().all(|v| *v == 0.0));
    }

    /// The road the scene shows: each point of the homotopy's path, plus the shortest
    /// correction for what it leaves unexplained, agrees with the measurements; the
    /// road starts at the shortest solution and ends at the least-1-norm one.
    #[test]
    fn the_road_from_the_shortest_to_the_least_one_norm() {
        for (m, n, k) in [(40, 100, 8), (25, 100, 8), (40, 100, 20), (12, 32, 3)] {
            for seed in 0..4u64 {
                let (a, _, y) = problem(m, n, k, 300 + seed);
                let mut h = Homotopy::new(&a, m, n, &y).recording();
                while h.step() == Status::Running {}
                let end = h.solution().unwrap();
                let short = Shortest::of(&a, m, n).unwrap();
                let shortest = short.solve(&a, &y);
                let path = h.path();
                let points = path.len() / n;
                assert_eq!(points, h.steps() + 1);
                assert_eq!(h.lambdas().len(), points);
                assert!(h.lambdas().windows(2).all(|w| w[1] <= w[0]) && h.lambdas()[points - 1] == 0.0);
                let mut road = Vec::new();
                for t in 0..points {
                    let x = &path[t * n..(t + 1) * n];
                    let mut ax = vec![0.0; m];
                    apply(&a, m, n, x, &mut ax);
                    let rest: Vec<f64> = y.iter().zip(&ax).map(|(p, q)| p - q).collect();
                    let fix = short.solve(&a, &rest);
                    let p: Vec<f64> = x.iter().zip(&fix).map(|(x, f)| x + f).collect();
                    apply(&a, m, n, &p, &mut ax);
                    assert!(relative_error(&ax, &y) < 1e-12, "m {m} k {k} seed {seed}, point {t}");
                    road.push(p);
                }
                assert!(relative_error(&road[0], &shortest) < 1e-14);
                assert!(relative_error(&road[points - 1], &end.x) < 1e-12);
            }
        }
    }

    /// The shortest solution is the shortest: it agrees with the measurements, and
    /// anything else that does (the signal itself, say) is longer.
    #[test]
    fn the_shortest_solution_is_shortest() {
        let (a, truth, y) = problem(30, 80, 6, 9);
        let x = Shortest::of(&a, 30, 80).unwrap().solve(&a, &y);
        let mut ax = vec![0.0; 30];
        apply(&a, 30, 80, &x, &mut ax);
        assert!(relative_error(&ax, &y) < 1e-12);
        assert!(norm2(&x) < norm2(&truth));
        // and it is the signal's part in the row space: what is left is at right
        // angles to every row
        let rest: Vec<f64> = truth.iter().zip(&x).map(|(t, s)| t - s).collect();
        for i in 0..30 {
            assert!(dot(&a[i * 80..(i + 1) * 80], &rest).abs() < 1e-12);
        }
    }

    #[test]
    fn coherence_and_its_bounds() {
        // two columns at 60 degrees, one at right angles to both
        let a = vec![1.0, 0.5, 0.0, 0.0, 0.75f64.sqrt(), 0.0, 0.0, 0.0, 2.0];
        assert!((coherence(&a, 3, 3).unwrap() - 0.5).abs() < 1e-12);
        assert_eq!(guaranteed(0.5, 10), 1);
        assert_eq!(guaranteed(0.2, 10), 2);
        assert_eq!(guaranteed(1.0 / 3.0, 10), 1, "k < 2 exactly: one");
        assert_eq!(guaranteed(0.0, 10), 10);
        // no matrix beats the Welch bound
        for seed in 0..5 {
            let mut rng = Rng::new(seed);
            let (m, n) = (8, 20);
            let a = gaussian_matrix(m, n, &mut rng);
            assert!(coherence(&a, m, n).unwrap() >= welch_bound(m, n));
        }
        assert!((welch_bound(2, 3) - 0.5).abs() < 1e-12, "three lines at 60 degrees in the plane");
        assert_eq!(welch_bound(5, 5), 0.0);
    }

    /// The solver at the scene's largest size, timed (run with --ignored --nocapture).
    #[test]
    #[ignore]
    fn timing() {
        let n = 128;
        for (m, k) in [(40, 8), (64, 12), (64, 30), (100, 50), (128, 64)] {
            let problems: Vec<_> = (0..20).map(|s| problem(m, n, k, s)).collect();
            let start = std::time::Instant::now();
            let mut steps = 0;
            for (a, _, y) in &problems {
                steps += basis_pursuit(a, m, n, y).unwrap().steps;
            }
            let each = start.elapsed().as_secs_f64() / 20.0;
            let start = std::time::Instant::now();
            for (a, _, y) in &problems {
                let _ = Shortest::of(a, m, n).map(|s| s.solve(a, y));
            }
            let shortest = start.elapsed().as_secs_f64() / 20.0;
            let start = std::time::Instant::now();
            for (a, _, _) in &problems {
                let _ = coherence(a, m, n);
            }
            let mu = start.elapsed().as_secs_f64() / 20.0;
            let mut rng = Rng::new(1);
            let start = std::time::Instant::now();
            for _ in 0..20 {
                let _ = gaussian_matrix(m, n, &mut rng);
            }
            let draw = start.elapsed().as_secs_f64() / 20.0;
            println!(
                "n {n} m {m} k {k}: basis pursuit {:.3} ms ({} steps), shortest {:.3} ms, coherence {:.3} ms, drawing A {:.3} ms",
                each * 1e3,
                steps / 20,
                shortest * 1e3,
                mu * 1e3,
                draw * 1e3
            );
        }
    }
}

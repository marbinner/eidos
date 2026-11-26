//! Schur form and the QR algorithm. In a perpendicular basis of its own every square
//! matrix is upper triangular, but for a 2 × 2 block on the diagonal for each complex
//! pair of eigenvalues: A = Q T Qᵀ. Factoring A = QR and multiplying the factors back
//! the other way round, again and again, finds that basis; it is how eigenvalues are
//! computed in practice.

use super::*;
use crate::scenes::stats::plot::{self, Plot, Rect};
use glam::DVec2;

pub const DEF: SceneDef = SceneDef {
    id: "schur",
    title: "Schur form and the QR algorithm",
    group: FACTOR,
    blurb: "Every matrix with real eigenvalues is triangular in some perpendicular basis, and repeating one factorisation finds it: how eigenvalues are computed.",
    make: |_| Box::new(Schur::new()),
};

// ----- the arithmetic, in double precision -------------------------------------------

/// The matrices here are 4 × 4, by rows: `a[i][j]` is in row i and column j.
const N: usize = 4;
type M = [[f64; N]; N];

const ID: M = [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]];

/// The most steps a run takes.
const LAST: usize = 120;

/// An entry just below the diagonal counts as nothing, for the shifted algorithm, once
/// it is this small beside the two diagonal entries either side of it: that row is
/// then done, and the work goes on above it.
const TOL: f64 = 1e-12;

fn mul(a: &M, b: &M) -> M {
    let mut c = [[0.0; N]; N];
    for (i, row) in c.iter_mut().enumerate() {
        for (j, v) in row.iter_mut().enumerate() {
            *v = (0..N).map(|l| a[i][l] * b[l][j]).sum();
        }
    }
    c
}

fn transpose(a: &M) -> M {
    let mut t = [[0.0; N]; N];
    for (i, row) in a.iter().enumerate() {
        for (j, v) in row.iter().enumerate() {
            t[j][i] = *v;
        }
    }
    t
}

fn trace(a: &M) -> f64 {
    (0..N).map(|i| a[i][i]).sum()
}

/// The determinant, by elimination with the largest pivot of each column.
fn det(a: &M) -> f64 {
    let mut m = *a;
    let mut d = 1.0;
    for c in 0..N {
        let p = (c..N).max_by(|&i, &j| m[i][c].abs().total_cmp(&m[j][c].abs())).unwrap_or(c);
        if m[p][c] == 0.0 {
            return 0.0;
        }
        if p != c {
            m.swap(p, c);
            d = -d;
        }
        d *= m[c][c];
        for r in c + 1..N {
            let f = m[r][c] / m[c][c];
            for k in c..N {
                m[r][k] -= f * m[c][k];
            }
        }
    }
    d
}

/// The square root of the sum of the squares of the entries (the Frobenius norm).
fn size(a: &M) -> f64 {
    a.iter().flatten().map(|v| v * v).sum::<f64>().sqrt()
}

fn minus(a: &M, b: &M) -> M {
    let mut c = *a;
    for (row, brow) in c.iter_mut().zip(b) {
        for (v, w) in row.iter_mut().zip(brow) {
            *v -= w;
        }
    }
    c
}

/// B = QR for the leading m × m part of `b`: Q orthogonal, R upper triangular with no
/// negative number on its diagonal (which makes the pair unique when B is invertible),
/// by Householder reflections. Outside that part Q is the identity and R is 0.
fn qr(b: &M, m: usize) -> (M, M) {
    let m = m.min(N);
    let mut r = [[0.0; N]; N];
    for i in 0..m {
        r[i][..m].copy_from_slice(&b[i][..m]);
    }
    let mut q = ID;
    for j in 0..m.saturating_sub(1) {
        let alpha = (j..m).map(|i| r[i][j] * r[i][j]).sum::<f64>().sqrt();
        // (a column of zeros, or of numbers that are not numbers, needs no reflection)
        if alpha.is_nan() || alpha <= 0.0 {
            continue;
        }
        // the reflection that takes column j, from the diagonal down, onto the axis
        let mut v = [0.0; N];
        for i in j..m {
            v[i] = r[i][j];
        }
        v[j] += if r[j][j] >= 0.0 { alpha } else { -alpha };
        let vv: f64 = (j..m).map(|i| v[i] * v[i]).sum();
        if vv.is_nan() || vv <= 0.0 {
            continue;
        }
        for c in 0..m {
            let d = 2.0 * (j..m).map(|i| v[i] * r[i][c]).sum::<f64>() / vv;
            for i in j..m {
                r[i][c] -= d * v[i];
            }
        }
        for row in q.iter_mut().take(m) {
            let d = 2.0 * (j..m).map(|i| row[i] * v[i]).sum::<f64>() / vv;
            for i in j..m {
                row[i] -= d * v[i];
            }
        }
    }
    // no negative number on R's diagonal: a row of R and the column of Q with it change sign
    for j in 0..m {
        if r[j][j] < 0.0 {
            for v in r[j].iter_mut() {
                *v = -*v;
            }
            for row in q.iter_mut() {
                row[j] = -row[j];
            }
        }
    }
    // (what the reflections leave below R's diagonal is rounding)
    for (i, row) in r.iter_mut().enumerate() {
        for v in row.iter_mut().take(i) {
            *v = 0.0;
        }
    }
    (q, r)
}

/// One step of the QR algorithm: A = QR, then RQ. Returns RQ and Q (RQ = QᵀAQ).
fn plain_step(a: &M) -> (M, M) {
    let (q, r) = qr(a, N);
    (mul(&r, &q), q)
}

/// Whether the entry below the diagonal in row `i` counts as nothing beside its
/// diagonal neighbours.
fn small(a: &M, i: usize) -> bool {
    a[i][i - 1].abs() <= TOL * (a[i][i].abs() + a[i - 1][i - 1].abs())
}

/// The eigenvalues of the 2 × 2 block of `a` at rows and columns i and i + 1, as
/// complex numbers (x, y) = x + iy: a complex pair, the one above the real line first,
/// or two real numbers.
fn block_eig(a: &M, i: usize) -> [DVec2; 2] {
    let mean = 0.5 * (a[i][i] + a[i + 1][i + 1]);
    let half = 0.5 * (a[i][i] - a[i + 1][i + 1]);
    let q = half * half + a[i][i + 1] * a[i + 1][i];
    if q < 0.0 {
        let im = (-q).sqrt();
        [DVec2::new(mean, im), DVec2::new(mean, -im)]
    } else {
        let r = q.sqrt();
        [DVec2::new(mean + r, 0.0), DVec2::new(mean - r, 0.0)]
    }
}

/// Whether that block has a complex pair of eigenvalues.
fn complex_block(a: &M, i: usize) -> bool {
    let half = 0.5 * (a[i][i] - a[i + 1][i + 1]);
    half * half + a[i][i + 1] * a[i + 1][i] < 0.0
}

/// The part of `a` the next shifted step works on, rows and columns `lo..=hi`; none once
/// every eigenvalue is found. From the bottom up, a row whose entry below the diagonal
/// counts as nothing is done, and so is a 2 × 2 block of a complex pair that is cut off
/// from the rows above it (no real shift can find a complex eigenvalue); the part is
/// what is left at the bottom, up to the next entry that counts as nothing.
fn active(a: &M) -> Option<(usize, usize)> {
    let mut hi = N - 1;
    loop {
        if hi == 0 {
            return None;
        }
        if small(a, hi) {
            hi -= 1;
            continue;
        }
        if complex_block(a, hi - 1) && (hi == 1 || small(a, hi - 1)) {
            if hi < 2 {
                return None;
            }
            hi -= 2;
            continue;
        }
        break;
    }
    let mut lo = hi;
    while lo > 0 && !small(a, lo) {
        lo -= 1;
    }
    Some((lo, hi))
}

/// One shifted step on the part still worked on: A − sI = QR there, with the guess s
/// its bottom-right entry, then RQ + sI; the rest of the matrix is carried along by
/// the same Q, so that the whole is QᵀAQ. None once every eigenvalue is found.
fn shifted_step(a: &M) -> Option<(M, M, usize, usize, f64)> {
    let (lo, hi) = active(a)?;
    let s = a[hi][hi];
    let m = hi - lo + 1;
    let mut b = [[0.0; N]; N];
    for i in 0..m {
        for j in 0..m {
            b[i][j] = a[lo + i][lo + j] - if i == j { s } else { 0.0 };
        }
    }
    let (qb, _) = qr(&b, m);
    let mut q = ID;
    for i in 0..m {
        for j in 0..m {
            q[lo + i][lo + j] = qb[i][j];
        }
    }
    Some((mul(&mul(&transpose(&q), a), &q), q, lo, hi, s))
}

/// What the picture of the eigenvalues shows for a matrix: its diagonal entries, but
/// for a 2 × 2 block on the diagonal with a complex pair of eigenvalues, which shows
/// that pair (looking for such blocks from the top down).
fn estimates(a: &M) -> [DVec2; N] {
    let mut out = [DVec2::ZERO; N];
    let mut i = 0;
    while i < N {
        if i + 1 < N && complex_block(a, i) {
            let pair = block_eig(a, i);
            (out[i], out[i + 1]) = (pair[0], pair[1]);
            i += 2;
        } else {
            out[i] = DVec2::new(a[i][i], 0.0);
            i += 1;
        }
    }
    out
}

// ----- the matrices ------------------------------------------------------------------

/// The turns the matrices are made with: each a turn in the plane of two axes, by the
/// angle in radians.
const TURNS: [(usize, usize, f64); 6] = [(0, 1, 0.7), (1, 2, 1.1), (2, 3, 0.5), (0, 2, 0.9), (1, 3, 0.4), (0, 3, 0.6)];

/// The orthogonal matrix every preset is seen through: the product of the turns.
fn q0() -> M {
    let mut q = ID;
    for (i, j, angle) in TURNS {
        let mut g = ID;
        let (c, s) = (angle.cos(), angle.sin());
        (g[i][i], g[j][j], g[i][j], g[j][i]) = (c, c, -s, s);
        q = mul(&q, &g);
    }
    q
}

/// A matrix to start from: A = Q₀ T Q₀ᵀ, with T upper triangular but for a 2 × 2 block
/// for a complex pair, so that its eigenvalues are known exactly, and its Schur form.
struct Preset {
    /// Its name on a button over the picture and in the panel, and on a phone.
    name: &'static str,
    short: &'static str,
    t: M,
    /// Its eigenvalues as complex numbers (x, y) = x + iy, largest in size first.
    eig: [(f64, f64); N],
    note: &'static str,
}

/// The upper part T has above its diagonal (the same for all but the symmetric one).
const fn triangle(d: [f64; N]) -> M {
    [[d[0], 1.0, 0.5, -1.0], [0.0, d[1], 1.0, 0.5], [0.0, 0.0, d[2], 1.0], [0.0, 0.0, 0.0, d[3]]]
}

const PRESETS: [Preset; 5] = [
    Preset {
        name: "well apart",
        short: "apart",
        t: triangle([4.0, 2.0, 0.7, 0.14]),
        eig: [(4.0, 0.0), (2.0, 0.0), (0.7, 0.0), (0.14, 0.0)],
        note: r"Eigenvalues 4, 2, 0.7 and 0.14: no two of the same size. Without shifts the entries just below the diagonal shrink by $2/4 = 0.5$, $0.7/2 = 0.35$ and $0.14/0.7 = 0.2$ a step.",
    },
    Preset {
        name: "symmetric",
        short: "symm.",
        t: [[4.0, 0.0, 0.0, 0.0], [0.0, -2.0, 0.0, 0.0], [0.0, 0.0, 0.8, 0.0], [0.0, 0.0, 0.0, 0.2]],
        eig: [(4.0, 0.0), (-2.0, 0.0), (0.8, 0.0), (0.2, 0.0)],
        note: r"$A^\top = A$, eigenvalues 4, $-2$, 0.8 and 0.2. Every $A_k$ is symmetric too, so it ends diagonal.",
    },
    Preset {
        name: "close in size",
        short: "close",
        t: triangle([4.0, 2.0, 1.8, 0.5]),
        eig: [(4.0, 0.0), (2.0, 0.0), (1.8, 0.0), (0.5, 0.0)],
        note: r"Eigenvalues 4, 2, 1.8 and 0.5: 2 and 1.8 are close in size, so without shifts $a_{32}$ shrinks by only $1.8/2 = 0.9$ a step.",
    },
    Preset {
        name: "a complex pair",
        short: "complex",
        t: [[4.0, 1.0, 0.5, -1.0], [0.0, 1.2, 1.6, 0.5], [0.0, -1.6, 1.2, 1.0], [0.0, 0.0, 0.0, 0.5]],
        eig: [(4.0, 0.0), (1.2, 1.6), (1.2, -1.6), (0.5, 0.0)],
        note: r"Eigenvalues 4, $1.2 \pm 1.6i$ and 0.5. The pair have the same size, 2: nothing can put them in order of size, and a $2 \times 2$ block stays. With shifts it stays too (no real guess $s$ is a complex eigenvalue), and everything else is found in 11 steps.",
    },
    Preset {
        name: r"$2$ and $-2$",
        short: r"$\pm 2$",
        t: triangle([3.0, 2.0, -2.0, 0.5]),
        eig: [(3.0, 0.0), (2.0, 0.0), (-2.0, 0.0), (0.5, 0.0)],
        note: r"Eigenvalues 3, 2, $-2$ and 0.5: two real ones of the same size. Without shifts the $2 \times 2$ block between them never settles: it swings back and forth from one step to the next, its diagonal entries $\pm 0.12$ and $\pm 0.19$ by turns, neither of them an eigenvalue, though the block's own eigenvalues are 2 and $-2$. With shifts all four are found in 13 steps.",
    },
];

/// The matrix of preset `p`.
fn matrix(p: usize) -> M {
    let q = q0();
    mul(&mul(&q, &PRESETS[p.min(PRESETS.len() - 1)].t), &transpose(&q))
}

/// For each entry just below the diagonal, the rate at which the plain algorithm is
/// expected to shrink it, |λᵢ₊₁| / |λᵢ| with the eigenvalues in order of size; none
/// where two real eigenvalues have the same size (no rate is claimed for such a matrix).
fn rates(p: usize) -> [Option<f64>; N - 1] {
    let eig = PRESETS[p.min(PRESETS.len() - 1)].eig;
    let len = |(x, y): (f64, f64)| x.hypot(y);
    let tie = (0..N - 1).any(|i| eig[i].1 == 0.0 && eig[i + 1].1 == 0.0 && (len(eig[i]) - len(eig[i + 1])).abs() < 1e-9);
    let mut out = [None; N - 1];
    if !tie {
        for (i, r) in out.iter_mut().enumerate() {
            *r = Some(len(eig[i + 1]) / len(eig[i]));
        }
    }
    out
}

// ----- a run of steps ----------------------------------------------------------------

/// What is done after a step.
#[derive(Clone, Copy, Debug, PartialEq)]
enum Next {
    /// A plain step.
    Plain,
    /// A shifted step on rows and columns lo..=hi, with the guess s.
    Shifted { lo: usize, hi: usize, s: f64 },
    /// Nothing: with shifts, every eigenvalue is found.
    Found,
    /// Nothing: the run has gone as far as it goes.
    End,
}

#[derive(Clone, Debug)]
struct Step {
    /// Aₖ, and Q = Q₀Q₁⋯Qₖ₋₁ so far: Aₖ = QᵀAQ.
    a: M,
    q: M,
    next: Next,
}

/// The steps from A₀ = A, as far as they go.
#[derive(Clone, Debug)]
struct Run {
    preset: usize,
    /// The kind of step it was made with, and whether every step is of that kind
    /// (shifts may be switched on or off part of the way).
    shifted: bool,
    pure: bool,
    steps: Vec<Step>,
    /// For each entry just below the diagonal, its predicted size: log₁₀ of it at
    /// step 0 and log₁₀ of the rate, the line through the run's own value where the
    /// rate has long taken over (none where no rate is claimed, or the run is not all
    /// plain steps).
    predicted: [Option<(f64, f64)>; N - 1],
}

impl Run {
    fn new(preset: usize, shifted: bool) -> Run {
        let a = matrix(preset);
        let mut run = Run { preset, shifted, pure: true, steps: vec![Step { a, q: ID, next: Next::End }], predicted: [None; N - 1] };
        run.carry_on(shifted);
        if !shifted {
            run.predicted = run.predict();
        }
        run
    }

    /// Plain steps all the way.
    fn plain(&self) -> bool {
        self.pure && !self.shifted
    }

    /// The last step there is.
    fn last(&self) -> usize {
        self.steps.len() - 1
    }

    fn a(&self, k: usize) -> &M {
        &self.steps[k.min(self.last())].a
    }

    fn next(&self, k: usize) -> Next {
        self.steps[k.min(self.last())].next
    }

    /// Carry the run on from its last step, plain or shifted, as far as it goes.
    fn carry_on(&mut self, shifted: bool) {
        while self.last() < LAST {
            let Some(now) = self.steps.last_mut() else { return };
            let (a, q) = (now.a, now.q);
            let (next_a, qk) = if shifted {
                match shifted_step(&a) {
                    Some((b, qk, lo, hi, s)) => {
                        now.next = Next::Shifted { lo, hi, s };
                        (b, qk)
                    }
                    None => {
                        now.next = Next::Found;
                        return;
                    }
                }
            } else {
                now.next = Next::Plain;
                plain_step(&a)
            };
            self.steps.push(Step { a: next_a, q: mul(&q, &qk), next: Next::End });
        }
    }

    /// From step `k` on, the other way (with shifts or without).
    fn switch(&mut self, k: usize, shifted: bool) {
        self.steps.truncate(k.min(self.last()) + 1);
        if let Some(now) = self.steps.last_mut() {
            now.next = Next::End;
        }
        if shifted != self.shifted {
            self.pure = false;
            self.predicted = [None; N - 1];
        }
        self.carry_on(shifted);
    }

    /// The predicted lines (see `predicted`): each through its entry's value at the
    /// last step, where the rate has long taken over.
    fn predict(&self) -> [Option<(f64, f64)>; N - 1] {
        let rates = rates(self.preset);
        let mut out = [None; N - 1];
        for (i, r) in rates.iter().enumerate() {
            let Some(r) = *r else { continue };
            let size = |k: usize| self.steps[k].a[i + 1][i].abs();
            let anchor = (0..=self.last()).rev().find(|&k| size(k) > 0.0 && size(k).is_finite()).unwrap_or(0);
            let v = size(anchor);
            if v > 0.0 && r > 0.0 {
                out[i] = Some((v.log10() - anchor as f64 * r.log10(), r.log10()));
            }
        }
        out
    }
}

/// The largest entry below the diagonal, leaving out the one inside each 2 × 2 block of
/// a complex pair on it (which is to stay): what is left to die away.
fn below(a: &M) -> f64 {
    let mut most = 0.0f64;
    let mut i = 0;
    while i < N {
        // the block from row and column i, one wide or two
        let width = if i + 1 < N && complex_block(a, i) { 2 } else { 1 };
        for row in a.iter().skip(i + width) {
            for v in &row[i..i + width] {
                most = most.max(v.abs());
            }
        }
        i += width;
    }
    most
}

/// What `below` is called, for a matrix: with a block of a complex pair on its
/// diagonal, the entry inside the block is left out, and the name says so.
fn below_name(a: &M) -> &'static str {
    if (0..N - 1).any(|i| complex_block(a, i)) {
        "largest entry below the diagonal, the block's aside"
    } else {
        "largest entry below the diagonal"
    }
}

// ----- numbers -----------------------------------------------------------------------

/// The lowest power of ten the colours and the chart show, and the highest.
const FLOOR: f64 = -16.0;
const CEILING: f64 = 1.0;

/// An entry's size on the colour scale: log₁₀ |v|, kept within the scale.
fn level(v: f64) -> f32 {
    if v.is_finite() && v != 0.0 {
        v.abs().log10().max(FLOOR).min(CEILING) as f32
    } else {
        FLOOR as f32
    }
}

/// An entry as a cell shows it: two decimals, and one figure in powers of ten once
/// that would read 0.00.
fn cell(v: f64) -> String {
    if v.is_finite() {
        format!("${}$", short(v))
    } else {
        "?".into()
    }
}

/// A small number in one figure and a power of ten, as markup: 3 × 10⁻⁵.
fn tiny(v: f64) -> String {
    let mut e = v.abs().log10().floor() as i32;
    let mut m = (v.abs() / 10f64.powi(e)).round();
    if m >= 10.0 {
        m = 1.0;
        e += 1;
    }
    let sign = if v < 0.0 { "-" } else { "" };
    format!(r"{sign}{} \times 10^{{{e}}}", m as i64)
}

/// A number as markup: two decimals while that says something, then one figure and a
/// power of ten (0 only for 0).
fn short(v: f64) -> String {
    if !v.is_finite() {
        "?".into()
    } else if v == 0.0 {
        "0".into()
    } else if v.abs() >= 0.005 {
        num(v, 2)
    } else {
        tiny(v)
    }
}

/// A complex number as markup: 1.20 + 1.60i, or a real one.
fn complex(z: DVec2, decimals: usize) -> String {
    if z.y.abs() < 0.5 * 10f64.powi(-(decimals as i32)) {
        num(z.x, decimals)
    } else {
        let sign = if z.y < 0.0 { "-" } else { "+" };
        format!("{} {sign} {}i", num(z.x, decimals), num(z.y.abs(), decimals))
    }
}

/// An entry's name, as markup: a₂₁ for row 2 and column 1.
fn entry(i: usize, j: usize) -> String {
    format!("a_{{{}{}}}", i + 1, j + 1)
}

// ----- what is drawn -----------------------------------------------------------------

/// The matrix part of the picture at a place along a run (a step, or between two).
#[derive(Clone, Copy, Debug)]
struct Look {
    /// Each entry's place on the colour scale, and its value.
    level: [[f32; N]; N],
    value: [[f64; N]; N],
    /// The dots in the picture of the eigenvalues.
    dots: [DVec2; N],
    /// The part the next shifted step works on: its first and last rows (as numbers,
    /// so that a new part glides from the old), how strongly it is drawn, and the
    /// guess s.
    part: [f32; 2],
    part_alpha: f32,
    shift: f64,
}

impl Look {
    fn at(run: &Run, s: f64) -> Look {
        let last = run.last();
        let s = if s.is_finite() { s.max(0.0).min(last as f64) } else { 0.0 };
        let k0 = (s.floor() as usize).min(last);
        let k1 = (k0 + 1).min(last);
        let u = (s - k0 as f64).max(0.0).min(1.0);
        let (a0, a1) = (run.a(k0), run.a(k1));
        let mut level = [[0.0; N]; N];
        for (i, row) in level.iter_mut().enumerate() {
            for (j, l) in row.iter_mut().enumerate() {
                let (from, to) = (self::level(a0[i][j]), self::level(a1[i][j]));
                *l = from + (to - from) * u as f32;
            }
        }
        let value = if u < 0.5 { *a0 } else { *a1 };
        let (d0, d1) = (estimates(a0), estimates(a1));
        let dots = std::array::from_fn(|i| d0[i].lerp(d1[i], u));
        let part = |k: usize| match run.next(k) {
            Next::Shifted { lo, hi, s } => Some(([lo as f32, hi as f32], s)),
            _ => None,
        };
        let uf = u as f32;
        let (part, part_alpha, shift) = match (part(k0), part(k1)) {
            (Some((p, s)), Some((q, t))) => ([p[0] + (q[0] - p[0]) * uf, p[1] + (q[1] - p[1]) * uf], 1.0, s + (t - s) * u),
            (Some((p, s)), None) if k1 > k0 => (p, 1.0 - uf, s),
            (Some((p, s)), None) => (p, 1.0, s),
            (None, Some((q, t))) => (q, uf, t),
            (None, None) => ([0.0, N as f32 - 1.0], 0.0, 0.0),
        };
        Look { level, value, dots, part, part_alpha, shift }
    }

    /// A share `t` of the way from `self` to `other`.
    fn towards(&self, other: &Look, t: f32) -> Look {
        let mut out = *other;
        for i in 0..N {
            for j in 0..N {
                out.level[i][j] = self.level[i][j] + (other.level[i][j] - self.level[i][j]) * t;
            }
            out.dots[i] = self.dots[i].lerp(other.dots[i], t as f64);
        }
        if t < 0.5 {
            out.value = self.value;
        }
        match (self.part_alpha > 0.0, other.part_alpha > 0.0) {
            (true, true) => {
                out.part = [self.part[0] + (other.part[0] - self.part[0]) * t, self.part[1] + (other.part[1] - self.part[1]) * t];
                out.shift = self.shift + (other.shift - self.shift) * t as f64;
            }
            (true, false) => (out.part, out.shift) = (self.part, self.shift),
            _ => {}
        }
        out.part_alpha = self.part_alpha + (other.part_alpha - self.part_alpha) * t;
        out
    }
}

/// The colour of a cell: its entry's size on the scale, from dark (10⁻¹⁶ and less)
/// through blue to a pale sea green (10 and more; 1 is nearly as pale).
fn heat(level: f32) -> Color {
    let t = ((level as f64 - FLOOR) / (CEILING - FLOOR)).max(0.0).min(1.0) as f32;
    const STOPS: [(f32, u32); 4] = [(0.0, 0x0B0F1C), (0.45, 0x1C2A57), (0.75, 0x2A5C8A), (1.0, 0x3D9C93)];
    for w in STOPS.windows(2) {
        let ((t0, c0), (t1, c1)) = (w[0], w[1]);
        if t <= t1 {
            return Color::hex(c0).mix(Color::hex(c1), ((t - t0) / (t1 - t0)).max(0.0).min(1.0));
        }
    }
    Color::hex(STOPS[3].1)
}

/// The colours of the three entries just below the diagonal, a₂₁, a₃₂ and a₄₃, in the
/// matrix and on the chart: what is to die away.
const SUBS: [Color; N - 1] = [pal::RED, pal::ORANGE, pal::YELLOW];
/// The true eigenvalues, the diagonal and the estimates it makes, and the shift.
const TRUE: Color = pal::GREEN;
const DIAG: Color = pal::WHITE;
const SHIFT: Color = pal::BLUE;
/// The other three entries below the diagonal, on the chart.
const OTHERS: Color = Color::hex(0x8792A8);

/// Kinds of line on the screen with a batch each (see `plot::apart`), so that one
/// whose number of points changes leaves the others to be followed: the chart's six
/// series, the trails of the dots, the outline of the part worked on, the predicted
/// lines and the faint lines of the chart.
const SERIES: u8 = plot::OWN;
const TRAILS: u8 = plot::OWN + 6;
const PART: u8 = plot::OWN + 10;
const PREDICTED: u8 = plot::OWN + 11;
const RULES: u8 = plot::OWN + 12;

/// Every kind of line in the order its batch is made in every frame.
fn in_order(f: &mut Frame) {
    for kind in [plot::AXES, RULES] {
        plot::apart(f, kind, false);
    }
    for kind in SERIES..SERIES + 6 {
        plot::apart(f, kind, false);
    }
    for kind in TRAILS..TRAILS + N as u8 {
        plot::apart(f, kind, false);
    }
    plot::apart(f, PART, true);
    plot::apart(f, PREDICTED, true);
}

// ----- where things go ---------------------------------------------------------------

#[derive(Clone, Debug)]
struct Layout {
    narrow: bool,
    /// The size of the words on the buttons, and of the small print.
    text: f32,
    small: f32,
    /// One step, run, start again, shifts.
    controls: Vec<Rect>,
    /// One for each preset, and where "matrix:" is written before them.
    presets: Vec<Rect>,
    lead: Option<Vec2>,
    /// The matrix: its heading, its top-left corner and the side of a cell.
    heading: Vec2,
    grid: Option<(Vec2, f32)>,
    /// The colour scale under it, and the lines of words under that (and how wide).
    scale: Option<Rect>,
    words: Option<(Vec2, f32)>,
    /// The step's two factors, under the words (a wide canvas only).
    factors: Option<Vec2>,
    chart: Option<Rect>,
    plane: Option<Rect>,
}

/// Words on the controls: the longest each can say, for their widths.
const CONTROL_WORDS: [&str; 4] = ["▶ one step", "❚❚ pause", "↺ start again", "shifts: off"];

/// The room the chart's title and key take above it, and its numbers below it; the
/// room the plane's title and key take above it.
const CHART_ABOVE: f32 = 58.0;
const CHART_BELOW: f32 = 30.0;
const PLANE_ABOVE: f32 = 40.0;

/// How tall the lines of words under the matrix are, each.
fn line_h(text: f32) -> f32 {
    text + 9.0
}

impl Layout {
    fn of(f: &Frame, view: Vec2, inset: f32) -> Layout {
        let view = view.max(Vec2::ONE);
        let narrow = view.x < 620.0;
        let (text, small) = if narrow { (11.5, 10.5) } else { (12.5, 11.5) };
        let side = if narrow { 10.0 } else { 16.0 };
        let top = inset.max(0.0) + if narrow { 6.0 } else { 12.0 };
        let (h, gap) = if narrow { (28.0, 5.0) } else { (28.0, 8.0) };
        let none = Layout {
            narrow,
            text,
            small,
            controls: Vec::new(),
            presets: Vec::new(),
            lead: None,
            heading: Vec2::new(side, top),
            grid: None,
            scale: None,
            words: None,
            factors: None,
            chart: None,
            plane: None,
        };
        if view.x < 240.0 || view.y < 180.0 {
            return none;
        }
        // the controls, and under them the matrices
        let controls = if narrow {
            plot::even_row(4, side, view.x - side, top, h, gap)
        } else {
            let mut x = side;
            CONTROL_WORDS
                .iter()
                .map(|w| {
                    let width = f.text_size(w, text).x + 22.0;
                    let r = (Vec2::new(x, top), Vec2::new(x + width, top + h));
                    x += width + gap;
                    r
                })
                .collect()
        };
        let row2 = top + h + gap;
        let (presets, lead) = if narrow {
            (plot::even_row(PRESETS.len(), side, view.x - side, row2, h, gap), None)
        } else {
            let lead_w = f.text_size("matrix:", text).x + 10.0;
            let names: Vec<&str> = PRESETS.iter().map(|p| p.name).collect();
            (plot::button_row(f, &names, side + lead_w, row2, h, text, gap), Some(Vec2::new(side, row2 + h * 0.5)))
        };
        let body = row2 + h + if narrow { 10.0 } else { 18.0 };
        let mut lay = Layout { controls, presets, lead, heading: Vec2::new(side, body), ..none };
        let head_h = if narrow { 22.0 } else { 28.0 };
        // under the matrix: the scale's name, the scale, its numbers; then the words
        let scale_h = if narrow { 54.0 } else { 64.0 };
        let words_h = if narrow { 2.0 } else { 3.0 } * line_h(small);
        if narrow {
            // a phone: everything one under another, the chart and the plane side by
            // side at the foot, the matrix as large as what is left lets it be
            let fixed = body + head_h + scale_h + words_h + 6.0 + CHART_ABOVE + CHART_BELOW;
            let charts_h = (view.y * 0.27).max(110.0);
            let cell = ((view.x - 2.0 * side) / N as f32).min((view.y - fixed - charts_h) / N as f32).min(72.0).floor();
            if cell < 26.0 {
                // (no room for the charts: the matrix alone)
                let cell = ((view.y - body - head_h - 8.0) / N as f32).min((view.x - 2.0 * side) / N as f32).floor();
                if cell >= 12.0 {
                    lay.grid = Some((Vec2::new(side, body + head_h), cell));
                }
                return lay;
            }
            let g = cell * N as f32;
            let gx = ((view.x - g) * 0.5).floor();
            lay.heading = Vec2::new(gx, body);
            lay.grid = Some((Vec2::new(gx, body + head_h), cell));
            let scale_y = body + head_h + g + 22.0;
            lay.scale = Some((Vec2::new(gx + 4.0, scale_y), Vec2::new(gx + g - 4.0, scale_y + 8.0)));
            lay.words = Some((Vec2::new(side, body + head_h + g + scale_h), view.x - 2.0 * side));
            let charts_top = body + head_h + g + scale_h + words_h + 6.0 + CHART_ABOVE;
            let bottom = view.y - CHART_BELOW;
            if bottom - charts_top >= 50.0 {
                let half = ((view.x - 2.0 * side - 12.0) * 0.5).floor();
                lay.chart = Some((Vec2::new(side + 30.0, charts_top), Vec2::new(side + half, bottom)));
                lay.plane = Some((Vec2::new(side + half + 12.0, charts_top), Vec2::new(view.x - side, bottom + CHART_BELOW - 6.0)));
            }
            return lay;
        }
        // wider: the matrix on the left, the chart and the plane on the right
        let room_y = view.y - body - head_h - scale_h - words_h - 16.0;
        let cell = (view.x * 0.44 / N as f32).min(room_y / N as f32).min(120.0).floor();
        if cell < 24.0 {
            return lay;
        }
        let g = cell * N as f32;
        lay.grid = Some((Vec2::new(side, body + head_h), cell));
        let scale_y = body + head_h + g + 28.0;
        lay.scale = Some((Vec2::new(side + 4.0, scale_y), Vec2::new(side + g - 4.0, scale_y + 9.0)));
        let words_at = Vec2::new(side, body + head_h + g + scale_h + 4.0);
        lay.words = Some((words_at, g.max(280.0)));
        // the factors under the words, if they fit
        let factors_at = words_at + Vec2::new(0.0, words_h + 12.0);
        if factors_at.y + FACTORS_H <= view.y - 8.0 && g >= 330.0 {
            lay.factors = Some(factors_at);
        }
        let x = side + g + 66.0;
        let right = view.x - side;
        if right - x < 150.0 {
            return lay;
        }
        let chart_top = body + CHART_ABOVE;
        let room = view.y - 12.0 - chart_top - CHART_BELOW - PLANE_ABOVE;
        let chart_h = (room * 0.6).max(0.0).floor();
        if chart_h < 60.0 {
            return lay;
        }
        lay.chart = Some((Vec2::new(x, chart_top), Vec2::new(right, chart_top + chart_h)));
        let plane_top = chart_top + chart_h + CHART_BELOW + PLANE_ABOVE;
        lay.plane = Some((Vec2::new(x - 30.0, plane_top), Vec2::new(right, view.y - 12.0)));
        lay
    }
}

/// How tall the factors are: a line of words, two matrices and two lines under them.
const FACTORS_H: f32 = 150.0;

// ----- the experiments ---------------------------------------------------------------

struct Trial {
    label: &'static str,
    look_for: &'static str,
    preset: usize,
    shifts: bool,
    /// Steps a second, and the step it plays to.
    rate: f32,
    to: usize,
}

const TRIALS: [Trial; 5] = [
    Trial {
        label: "eigenvalues well apart",
        look_for: r"$\lambda = 4$, 2, 0.7 and 0.14. No step changes them ($\tr A_k$ and $\det A_k$ in the numbers stay put), and the entries below the diagonal die away, each at its own pace: on the chart $|a_{21}|$ shrinks by $2/4 = 0.5$ a step, $|a_{32}|$ by $0.7/2 = 0.35$ and $|a_{43}|$ by $0.14/0.7 = 0.2$, along the dashed lines. After 40 steps the largest is $4 \times 10^{-12}$ and the diagonal reads 4, 2, 0.7, 0.14 in order of size: $A_{40}$ is the Schur form, the $T$ this matrix was made from, but for the signs of some entries.",
        preset: 0,
        shifts: false,
        rate: 4.0,
        to: 40,
    },
    Trial {
        label: "a symmetric matrix",
        look_for: r"$A^\top = A$, and then every $A_{k+1} = Q_k^\top A_k Q_k$ is symmetric too: the entries above the diagonal die with those below, and $A_k$ becomes diagonal. That is the spectral theorem, $A = QDQ^\top$ with perpendicular eigenvectors in $Q$. The diagonal ends at 4, $-2$, 0.8, 0.2: in order of size, so the negative one is second. After 40 steps the largest entry off the diagonal is $3 \times 10^{-12}$.",
        preset: 1,
        shifts: false,
        rate: 4.0,
        to: 40,
    },
    Trial {
        label: "two eigenvalues close in size",
        look_for: r"$\lambda = 4$, 2, 1.8 and 0.5. Between 2 and 1.8, $|a_{32}|$ shrinks by only $1.8/2 = 0.9$ a step: after 100 steps it is still $1 \times 10^{-6}$, and the diagonal there reads 2.000006 and 1.799994. Carried on, it would reach $10^{-16}$ at step 320; $|a_{21}|$ and $|a_{43}|$ were there by steps 58 and 28.",
        preset: 2,
        shifts: false,
        rate: 10.0,
        to: 100,
    },
    Trial {
        label: "the same with shifts",
        look_for: r"The same matrix. Each step now factors $A_k - sI$, with the guess $s$ the bottom corner of the part outlined in blue, and adds $s$ back. Near an eigenvalue $A_k - sI$ is nearly singular, and the bottom row falls fast: $|a_{43}|$ goes 1.26, 0.09, 0.003, $1 \times 10^{-5}$, $5 \times 10^{-10}$, $1 \times 10^{-19}$, the zeros after the point roughly doubling with each of the last steps. That row is then done, with 4 in its corner; the outline shrinks and the work moves up a row. All four are found in 13 steps, against 220 without shifts, and not in order of size.",
        preset: 2,
        shifts: true,
        rate: 1.5,
        to: LAST,
    },
    Trial {
        label: "a complex pair",
        look_for: r"$\lambda = 4$, $1.2 \pm 1.6i$ and 0.5. The pair have the same size, $|1.2 \pm 1.6i| = 2$, so nothing tells them apart by size: $|a_{32}|$, between them, does not die (it settles at 1.60), and a $2 \times 2$ block stays on the diagonal. Its diagonal entries (here both 1.2) are not eigenvalues, but its eigenvalues are the pair: two white dots land on the green rings off the real line. Round the block everything dies, at 0.5 and 0.25 a step.",
        preset: 3,
        shifts: false,
        rate: 4.0,
        to: 40,
    },
];

/// The step the scene opens at, of the first preset without shifts.
const OPENING: usize = 10;

// ----- the scene -----------------------------------------------------------------------

/// How fast the drawn step follows the step reached: each of two eases one after the
/// other covers two thirds of what is left in this many seconds.
const FOLLOW: f64 = 0.07;
/// How long a change of matrix takes to fade through (seconds).
const BLEND: f32 = 0.45;

pub struct Schur {
    preset: usize,
    shifts: bool,
    run: Run,
    /// The step reached, and the step drawn, which follows it through `between`.
    k: usize,
    between: f64,
    shown: f64,
    playing: bool,
    /// Where playing stops of itself.
    stop_at: usize,
    /// Steps a second, and the part of a step owed.
    rate: f32,
    owed: f32,
    /// After a change of matrix: the picture as it was, which fades into the new one,
    /// and how far that has gone (1: there).
    from: Option<(Look, Run, f64)>,
    blend: f32,
    /// The chart's steps across, and the plane's ranges (real from, to; imaginary
    /// half-height), as drawn: they glide to new ones.
    span: plot::Glide,
    plane: plot::Glide,
}

impl Schur {
    fn new() -> Schur {
        let mut scene = Schur {
            preset: 0,
            shifts: false,
            run: Run::new(0, false),
            k: OPENING,
            between: OPENING as f64,
            shown: OPENING as f64,
            playing: false,
            stop_at: LAST,
            rate: 4.0,
            owed: 0.0,
            from: None,
            blend: 1.0,
            span: plot::Glide::default(),
            plane: plot::Glide::default(),
        };
        scene.span.snap(&[scene.span_wanted()]);
        scene.plane.snap(&plane_range(0));
        scene
    }

    /// The picture as drawn now (between the old matrix and the new while they fade).
    fn look(&self) -> Look {
        let now = Look::at(&self.run, self.shown);
        match &self.from {
            Some((old, _, _)) => old.towards(&now, smoothstep(self.blend)),
            None => now,
        }
    }

    /// Start afresh from preset `p` (with or without shifts): the old picture fades
    /// into the new one at step 0.
    fn start(&mut self, p: usize, shifts: bool) {
        let p = p.min(PRESETS.len() - 1);
        let old = self.look();
        let old_run = std::mem::replace(&mut self.run, Run::new(p, shifts));
        self.from = Some((old, old_run, self.shown));
        self.blend = 0.0;
        (self.preset, self.shifts) = (p, shifts);
        (self.k, self.between, self.shown) = (0, 0.0, 0.0);
        (self.playing, self.owed, self.stop_at) = (false, 0.0, LAST);
    }

    /// Set experiment `i` up: its matrix from the start, then play it.
    fn set_up(&mut self, i: usize) {
        let t = &TRIALS[i.min(TRIALS.len() - 1)];
        self.start(t.preset, t.shifts);
        self.rate = t.rate;
        (self.stop_at, self.playing) = (t.to, true);
    }

    /// Another matrix, from its start.
    fn choose(&mut self, p: usize) {
        if p != self.preset || self.k != 0 || !self.run.pure || self.run.shifted != self.shifts {
            self.start(p, self.shifts);
        }
    }

    /// Shifts on or off: from the step reached on. (At the start the run is made
    /// afresh, all of the one kind.)
    fn set_shifts(&mut self, on: bool) {
        if on == self.shifts {
            return;
        }
        self.shifts = on;
        if self.k == 0 {
            self.start(self.preset, on);
        } else {
            self.run.switch(self.k, on);
            self.stop_at = LAST;
        }
    }

    fn can_step(&self) -> bool {
        self.k < self.run.last()
    }

    fn one_step(&mut self) {
        self.playing = false;
        if self.can_step() {
            self.k += 1;
        }
    }

    fn toggle_play(&mut self) {
        if self.playing {
            self.playing = false;
        } else if self.can_step() {
            (self.playing, self.owed, self.stop_at) = (true, 0.0, LAST);
        }
    }

    /// Back to A₀: the steps glide back, or the run is made afresh if it was not all
    /// of the kind now chosen.
    fn again(&mut self) {
        self.playing = false;
        if self.run.pure && self.run.shifted == self.shifts {
            self.k = 0;
        } else {
            self.start(self.preset, self.shifts);
        }
    }

    /// The chart's steps across: a round number past the step reached and the step the
    /// run plays to.
    fn span_wanted(&self) -> f32 {
        let end = if self.playing { self.stop_at.min(self.run.last()) } else { 0 };
        let most = self.k.max(end).max(if self.run.plain() { 20 } else { 10 });
        let round = if most <= 20 { 5 } else { 20 };
        (most.div_ceil(round) * round) as f32
    }

    /// Move everything on: `dt` seconds of the scene's time (playing) and `real_dt` of
    /// the clock's (the glides, so that a step asked for while the scene's time stands
    /// still is still seen through).
    fn advance(&mut self, dt: f32, real_dt: f32) {
        if self.playing {
            self.owed += self.rate.max(0.0) * dt.max(0.0);
            // (a very long frame falls behind rather than taking many steps at once)
            self.owed = self.owed.min(3.0);
            let end = self.stop_at.min(self.run.last());
            while self.owed >= 1.0 && self.k < end {
                self.k += 1;
                self.owed -= 1.0;
            }
            if self.k >= end {
                (self.playing, self.owed) = (false, 0.0);
            }
        }
        let to = self.k as f64;
        let e = 1.0 - (-(real_dt.max(0.0) as f64) / FOLLOW).exp();
        self.between += (to - self.between) * e;
        self.shown += (self.between - self.shown) * e;
        // (exactly there in the end: the picture then stops changing)
        if ((to - self.between).abs() < 1e-3 && (to - self.shown).abs() < 1e-3) || !self.shown.is_finite() || !self.between.is_finite() {
            (self.between, self.shown) = (to, to);
        }
        if self.from.is_some() {
            self.blend += real_dt.max(0.0) / BLEND;
            if self.blend >= 1.0 || !self.blend.is_finite() {
                (self.blend, self.from) = (1.0, None);
            }
        }
        self.span.follow(&[self.span_wanted()], real_dt, false);
        self.plane.follow(&plane_range(self.preset), real_dt, false);
    }

    fn moving(&self) -> bool {
        self.playing
            || self.shown != self.k as f64
            || self.between != self.k as f64
            || self.from.is_some()
            || self.span.shown != [self.span_wanted()]
            || self.plane.shown != plane_range(self.preset)
    }

    /// The step whose numbers are shown (the step drawn, to the nearest).
    fn named(&self) -> usize {
        (self.shown.round().max(0.0) as usize).min(self.run.last())
    }
}

/// The part of the complex plane shown for preset `p`: the real parts from and to, and
/// the imaginary half-height, with room round the eigenvalues.
fn plane_range(p: usize) -> [f32; 3] {
    let eig = PRESETS[p.min(PRESETS.len() - 1)].eig;
    let (mut lo, mut hi, mut im) = (0.0f64, 0.0f64, 0.0f64);
    for (x, y) in eig {
        (lo, hi, im) = (lo.min(x), hi.max(x), im.max(y.abs()));
    }
    let pad = 0.5 + 0.06 * (hi - lo);
    [(lo - pad) as f32, (hi + pad) as f32, (im + pad).max(1.0) as f32]
}

impl Scene for Schur {
    fn view(&self) -> View {
        View::Flat { center: Vec2::ZERO, half_height: 1.0 }
    }

    fn settings(&self, s: &mut ViewSettings) {
        s.grid = false;
        s.bloom = s.bloom.min(0.2);
    }

    fn animated(&self) -> bool {
        self.moving()
    }

    fn pointer_hint(&self) -> Option<&'static str> {
        Some("buttons over the picture: click")
    }

    fn ui(&mut self, ui: &mut egui::Ui, _: &Gfx) {
        ui::note(ui, r"Not every square matrix can be made diagonal by a change of basis, but every one with real eigenvalues can be made upper triangular by a change to a perpendicular basis: $A = QTQ^\top$ with $Q$ orthogonal and $T$ upper triangular, the eigenvalues on its diagonal. That is the Schur form. (A complex pair of eigenvalues keeps a $2 \times 2$ block on the diagonal of $T$.) The QR algorithm finds it: factor $A_k = Q_k R_k$, $Q_k$ orthogonal and $R_k$ upper triangular, multiply the factors the other way round, and repeat:");
        ui::equation(ui, r"A_{k+1} = R_k Q_k = Q_k^\top A_k Q_k");
        ui::note(ui, r"So every $A_k$ has the eigenvalues of $A$; and for a typical matrix whose eigenvalues all differ in size, the entries below the diagonal die away, which leaves the eigenvalues on the diagonal. With a refinement (shifts, below) this is how computers find eigenvalues. Step it with the buttons over the picture.");

        ui::section(ui, "Things to try");
        for (i, t) in TRIALS.iter().enumerate() {
            if ui::experiment(ui, t.label, t.look_for) {
                self.set_up(i);
            }
        }

        ui::section(ui, "Steps");
        ui.horizontal_wrapped(|ui| {
            if ui.add_enabled(self.can_step(), egui::Button::new("▶ one step")).clicked() {
                self.one_step();
            }
            if ui.button(if self.playing { "❚❚ pause" } else { "▶▶ run" }).clicked() {
                self.toggle_play();
            }
            if ui.button("↺ start again").clicked() {
                self.again();
            }
        });
        ui::slider_log(ui, "steps a second", &mut self.rate, 0.5..=20.0);
        let mut shifts = self.shifts;
        if ui::checkbox(ui, &mut shifts, "shifts").changed() {
            self.set_shifts(shifts);
        }
        ui::note(ui, r"With shifts, each step works on the part outlined in blue: it factors that part less $sI$ as $Q_k R_k$ and puts back $R_k Q_k + sI$, the rest carried along by the same $Q_k$, so that the whole is still $Q_k^\top A_k Q_k$. The guess $s$ is the part's bottom-right entry. Once the entry left of that corner is below $10^{-12}$ times the two diagonal entries beside it, that row is done and the part shrinks; so does a $2 \times 2$ block of a complex pair, once cut off from the rows above it. Switched on or off, shifts take effect from the step reached; start again to have them from the start.");

        ui::section(ui, "Matrix");
        let names: Vec<&str> = PRESETS.iter().map(|p| p.name).collect();
        let mut chosen = self.preset;
        if ui::choice(ui, &names, &mut chosen) {
            self.choose(chosen);
        }
        ui::note(ui, PRESETS[self.preset].note);
        ui::note(ui, r"Each was made as $Q_0 T Q_0^\top$ from a triangular $T$ (with a $2 \times 2$ block for a complex pair) and a fixed orthogonal $Q_0$, so its eigenvalues, the green rings, are known exactly.");

        ui::section(ui, "Reading the picture");
        ui::key(
            ui,
            &[
                (heat(-2.0), r"cells: the entries of $A_k$, each coloured by its size on a logarithmic scale (the strip under the matrix): the larger, the paler; $10^{-16}$ and less, darkest"),
                (DIAG, r"white: the diagonal of $A_k$, outlined in the matrix; in the plane, a dot for each diagonal entry, or for each eigenvalue of a $2 \times 2$ block on the diagonal with a complex pair"),
                (SUBS[0], r"red, orange, yellow: $a_{21}$, $a_{32}$ and $a_{43}$, the entries just below the diagonal; on the chart their sizes step by step, and dashed, the rates predicted for them"),
                (OTHERS, "grey lines: the other three entries below the diagonal"),
                (TRUE, "green rings: the eigenvalues of $A$, in the complex plane"),
                (SHIFT, r"blue, with shifts: the part still worked on, and its bottom-right entry, the guess $s$ (also on the real line of the plane)"),
            ],
        );

        ui::section(ui, "Words");
        ui::term(ui, "Orthogonal matrix:", r"$Q^\top Q = I$: its columns are perpendicular and of length 1, so it turns or mirrors and keeps lengths, and $Q^{-1} = Q^\top$.");
        ui::term(ui, "QR factorisation:", r"$A = QR$ with $Q$ orthogonal and $R$ upper triangular. Here $R$'s diagonal has no negative numbers, which makes the two unique when $A$ is invertible.");
        ui::term(ui, "Similar matrices:", r"$B = P^{-1} A P$: the same map written in another basis, with the same eigenvalues. Each step is one, with $P = Q_k$.");
        ui::term(ui, "Schur form:", r"$A = QTQ^\top$, $Q$ orthogonal and $T$ upper triangular with the eigenvalues on its diagonal, in any order one likes. For a real $A$ with complex eigenvalues, $T$ keeps a $2 \times 2$ block for each conjugate pair; allowing complex numbers in $Q$ (and the conjugate transpose in place of $Q^\top$) makes it triangular.");
        ui::term(ui, "Shift:", r"a guess $s$ at an eigenvalue. $A - sI$ has the eigenvalues $\lambda - s$, one of them tiny when the guess is good; the bottom row then dies in a few steps, as in the fourth thing to try.");
        ui::term(ui, "Log scale:", r"each mark up is $10^4$ times the one below, so an entry multiplied by the same factor every step falls along a straight line.");
        ui::term(ui, "The power method, on every direction:", r"without shifts, $A^k = (Q_0 Q_1 \cdots Q_{k-1})(R_{k-1} \cdots R_1 R_0)$. So the first column of $Q_0 \cdots Q_{k-1}$ is $A^k$ times $(1, 0, 0, 0)$ scaled to length 1: the power method. Its first two columns are a perpendicular basis of the plane $A^k$ makes of the first two axes, its first three of the space it makes of the first three, and so on: the power method on planes and spaces.");

        ui::section(ui, "Numbers");
        let k = self.named();
        let a = self.run.a(k);
        let a0 = self.run.a(0);
        ui::readout(ui, "step $k$", format!("${k}$"));
        if let Next::Shifted { s, .. } = self.run.next(k) {
            ui::readout_colored(ui, "the guess $s$", format!("${}$", num(s, 4)), SHIFT);
        }
        ui::readout(ui, r"$\tr A_k$, $\tr A$", format!("${}, {}$", num(trace(a), 5), num(trace(a0), 5)));
        ui::readout(ui, r"$\det A_k$, $\det A$", format!("${}, {}$", num(det(a), 5), num(det(a0), 5)));
        let q = &self.run.steps[k].q;
        ui::readout(ui, r"$\lVert A - Q A_k Q^\top \rVert$", format!("${}$", short(size(&minus(a0, &mul(&mul(q, a), &transpose(q)))))));
        ui::readout(ui, below_name(a), format!("${}$", short(below(a))));
        // the dots' values beside the rings', each a column (the dots by their places
        // on the diagonal, the eigenvalues by size)
        let dots: Vec<String> = estimates(a).iter().map(|z| complex(*z, 4)).collect();
        let rings: Vec<String> = PRESETS[self.preset].eig.iter().map(|(x, y)| complex(DVec2::new(*x, *y), 2)).collect();
        let column = |items: &[String], color: &str| format!(r"\begin{{bmatrix}} {} \end{{bmatrix}}", items.iter().map(|t| format!(r"\textcolor{{{color}}}{{{t}}}")).collect::<Vec<_>>().join(r" \\ "));
        ui::note(ui, "What the diagonal says (the white dots), and the eigenvalues (the green rings):");
        ui::equation(ui, &format!(r"{}\qquad{}", column(&dots, "white"), column(&rings, "green")));
        ui::see_also(ui, &[("gram", "Gram–Schmidt: the QR factorisation"), ("powers", "the power method: $A^k x$, one direction at a time"), ("spectral", "symmetric matrices: $T$ is diagonal"), ("jordan", "when eigenvectors run out")]);
    }

    fn frame(&mut self, f: &mut Frame) {
        let view = f.view_size();
        // (a canvas squeezed to a sliver is one the controls have opened over, on a phone)
        let covered = view.x < 160.0;
        let lay = Layout::of(f, view, f.top_inset);
        if !covered {
            // the picture is for its buttons: no moving the view, and two quick clicks
            // are two steps
            f.handles.capture();
            f.handles.claim_clicks();
            self.buttons(f, &lay);
        }
        self.advance(f.dt, f.real_dt);
        if covered {
            return;
        }
        in_order(f);
        let look = self.look();
        self.grid(f, &lay, &look);
        self.words(f, &lay);
        self.factors(f, &lay);
        self.chart(f, &lay);
        self.plane(f, &lay, &look);
    }
}

impl Schur {
    /// The buttons over the picture: the steps, the shifts and the matrices.
    fn buttons(&mut self, f: &mut Frame, lay: &Layout) {
        let size = lay.text;
        if let [step, run, again, shifts] = lay.controls[..] {
            let words = if lay.narrow {
                ["step", if self.playing { "pause" } else { "run" }, "again", if self.shifts { "shifts on" } else { "shifts off" }]
            } else {
                ["▶ one step", if self.playing { "❚❚ pause" } else { "▶▶ run" }, "↺ start again", if self.shifts { "shifts: on" } else { "shifts: off" }]
            };
            if plot::button(f, step, words[0], size, false) {
                self.one_step();
            }
            if plot::button(f, run, words[1], size, self.playing) {
                self.toggle_play();
            }
            if plot::button(f, again, words[2], size, false) {
                self.again();
            }
            if plot::button(f, shifts, words[3], size, self.shifts) {
                self.set_shifts(!self.shifts);
            }
        }
        if let Some(at) = lay.lead {
            f.text_px(at, "matrix:", size, pal::DIM, Align2::LEFT_CENTER);
        }
        for (p, r) in lay.presets.iter().enumerate() {
            let name = if lay.narrow { PRESETS[p].short } else { PRESETS[p].name };
            if plot::button(f, *r, name, size, p == self.preset) {
                self.choose(p);
            }
        }
    }

    /// The matrix as a grid of cells, each coloured by its entry's size, and the colour
    /// scale under it.
    fn grid(&self, f: &mut Frame, lay: &Layout, look: &Look) {
        let Some((at, c)) = lay.grid else { return };
        let k = self.named();
        let heading = if lay.narrow { format!("$A_{{{k}}}$, after {k} {}", steps(k)) } else { format!("$A_{{{k}}}$, the matrix after {k} {} of the QR algorithm", steps(k)) };
        f.text_px(lay.heading, heading, if lay.narrow { 13.0 } else { 14.0 }, pal::TEXT, Align2::LEFT_TOP);
        let size = (c * 0.16).max(8.0).min(16.0);
        for i in 0..N {
            for j in 0..N {
                let min = at + Vec2::new(j as f32, i as f32) * c;
                hud_rect(f, min + Vec2::splat(1.0), min + Vec2::splat(c - 1.0), heat(look.level[i][j]));
                if c >= 30.0 {
                    f.text_px(min + Vec2::splat(c * 0.5), cell(look.value[i][j]), size, pal::TEXT, Align2::CENTER_CENTER);
                }
            }
        }
        // the diagonal, outlined
        for i in 0..N {
            let min = at + Vec2::splat(i as f32 * c);
            hud_outline(f, min + Vec2::splat(2.0), min + Vec2::splat(c - 2.0), 1.6, DIAG.a(0.9));
        }
        // the entries just below it, named in their colours
        for (i, color) in SUBS.iter().enumerate() {
            let min = at + Vec2::new(i as f32, i as f32 + 1.0) * c;
            if c >= 40.0 {
                f.text_px(min + Vec2::new(5.0, 4.0), format!("${}$", entry(i + 1, i)), if c >= 70.0 { 11.5 } else { 10.0 }, *color, Align2::LEFT_TOP);
            }
            plot::line_of(f, RULES, min + Vec2::new(2.0, c - 3.0), min + Vec2::new(c - 2.0, c - 3.0), 2.0, color.a(0.9));
        }
        // the part a shifted step works on, and its corner, the guess s
        let [lo, hi] = look.part;
        let (p0, p1) = (at + Vec2::splat(lo * c - 3.0), at + Vec2::splat((hi + 1.0) * c + 3.0));
        let corners = [p0, Vec2::new(p1.x, p0.y), p1, Vec2::new(p0.x, p1.y), p0];
        let alpha = look.part_alpha.max(0.0).min(1.0);
        for w in corners.windows(2) {
            plot::dashed_of(f, PART, w[0], w[1], 2.0, SHIFT.a(alpha));
        }
        let corner = at + Vec2::splat(hi * c);
        hud_outline(f, corner + Vec2::splat(5.0), corner + Vec2::splat(c - 5.0), 2.0, SHIFT.a(alpha));
        if alpha > 0.0 && c >= 40.0 {
            f.caption_px(corner + Vec2::splat(c - 8.0), "$s$", 12.0, SHIFT.mix(pal::WHITE, 0.3).a(alpha), Align2::RIGHT_BOTTOM);
        }
        // the colour scale
        if let Some((min, max)) = lay.scale {
            let strips = 34;
            for s in 0..strips {
                let (a, b) = (s as f32 / strips as f32, (s + 1) as f32 / strips as f32);
                let x = |t: f32| min.x + (max.x - min.x) * t;
                let level = FLOOR as f32 + (CEILING - FLOOR) as f32 * (a + b) * 0.5;
                hud_rect(f, Vec2::new(x(a), min.y), Vec2::new(x(b) + 0.5, max.y), heat(level));
            }
            let small = lay.small;
            for (e, name) in [(-16, "10^{-16}"), (-12, "10^{-12}"), (-8, "10^{-8}"), (-4, "10^{-4}"), (0, "1")] {
                let t = ((e as f64 - FLOOR) / (CEILING - FLOOR)) as f32;
                let x = min.x + (max.x - min.x) * t;
                plot::line_of(f, plot::AXES, Vec2::new(x, max.y), Vec2::new(x, max.y + 3.0), 1.0, pal::DIM);
                // (the end ones inside the strip's ends, so that neither runs off the picture)
                let align = match e {
                    -16 => Align2::LEFT_TOP,
                    0 => Align2::RIGHT_TOP,
                    _ => Align2::CENTER_TOP,
                };
                let nudge = match e {
                    -16 => -3.0,
                    0 => 3.0,
                    _ => 0.0,
                };
                f.text_px(Vec2::new(x + nudge, max.y + 4.0), format!("${name}$"), small - 0.5, pal::DIM, align);
            }
            f.text_px(Vec2::new(min.x - 4.0, min.y - 5.0), r"the colours: the size of an entry, $|a_{ij}|$", small - 0.5, pal::DIM, Align2::LEFT_BOTTOM);
        }
    }

    /// The lines of words under the matrix: the step, and what it keeps.
    fn words(&self, f: &mut Frame, lay: &Layout) {
        let Some((at, width)) = lay.words else { return };
        let small = lay.small;
        let k = self.named();
        let next = self.run.next(k);
        let step = match next {
            Next::Shifted { lo, hi, .. } if hi - lo + 1 < N => r"a step, on the outlined part: $A_k - sI = Q_k R_k$, then $R_k Q_k + sI$",
            Next::Shifted { .. } => r"a step: $A_k - sI = Q_k R_k$, then $A_{k+1} = R_k Q_k + sI$",
            _ if self.shifts => r"a step: $A_k - sI = Q_k R_k$, then $A_{k+1} = R_k Q_k + sI$",
            _ => r"a step: $A_k = Q_k R_k$, then $A_{k+1} = R_k Q_k$",
        };
        let kept = if k == 0 {
            r"$A_0 = A$; each step is $Q_k^\top A_k Q_k$, so every $A_k$ has the eigenvalues of $A$".to_string()
        } else if k == 1 {
            r"$A_1 = Q_0^\top A Q_0$, $Q_0$ orthogonal: the eigenvalues of $A$".to_string()
        } else {
            format!(r"$A_{{{k}}} = Q^\top A Q$ with $Q = Q_0 Q_1 \cdots Q_{{{}}}$ orthogonal: the eigenvalues of $A$", k - 1)
        };
        let status = match next {
            Next::Shifted { s, .. } => format!(r"the guess $s = {}$ (blue): the bottom corner of the outlined part", num(s, 4)),
            Next::Found => format!(r"every eigenvalue found after {k} {}: the Schur form", steps(k)),
            _ if k >= self.run.last() => format!("the run ends here, at step {k}"),
            _ => format!(r"{}: ${}$", below_name(self.run.a(k)), short(below(self.run.a(k)))),
        };
        let lines = if lay.narrow { [step.to_string(), status, String::new()] } else { [step.to_string(), kept, status] };
        let h = line_h(small);
        for (n, line) in lines.iter().enumerate() {
            let color = if n == 0 { pal::TEXT } else { pal::DIM };
            // (a line too long for the room is drawn smaller, to fit)
            let w = f.text_size(line, small).x;
            let s = if w > width && w > 0.0 { (small * width / w).max(8.0) } else { small };
            f.text_px(at + Vec2::new(0.0, n as f32 * h), line.as_str(), s, color, Align2::LEFT_TOP);
        }
    }

    /// The step's two factors, written out: Q orthogonal, R upper triangular.
    fn factors(&self, f: &mut Frame, lay: &Layout) {
        let Some(at) = lay.factors else { return };
        let k = self.named();
        let a = *self.run.a(k);
        let (lo, hi, s, shifted) = match self.run.next(k) {
            Next::Shifted { lo, hi, s } => (lo, hi, s, true),
            Next::Plain => (0, N - 1, 0.0, false),
            Next::Found => {
                f.text_px(at, "no more steps: every eigenvalue is found", lay.small, pal::DIM, Align2::LEFT_TOP);
                return;
            }
            Next::End => {
                f.text_px(at, "no more steps: the run ends here", lay.small, pal::DIM, Align2::LEFT_TOP);
                return;
            }
        };
        let m = hi - lo + 1;
        let mut b = [[0.0; N]; N];
        for i in 0..m {
            for j in 0..m {
                b[i][j] = a[lo + i][lo + j] - if i == j { s } else { 0.0 };
            }
        }
        let (q, r) = qr(&b, m);
        let title = if shifted {
            if m == N {
                format!(r"this step: $A_{{{k}}} - sI = Q_{{{k}}} R_{{{k}}}$")
            } else {
                format!(r"this step, on the outlined part: $A_{{{k}}} - sI = Q_{{{k}}} R_{{{k}}}$")
            }
        } else {
            format!(r"this step: $A_{{{k}}} = Q_{{{k}}} R_{{{k}}}$")
        };
        f.text_px(at, title, lay.small, pal::TEXT, Align2::LEFT_TOP);
        let size = 11.0;
        let widest = 5;
        let rows = |m_: &M, kind: u8| -> Vec<Vec<(String, Color)>> {
            (0..m)
                .map(|i| {
                    (0..m)
                        .map(|j| {
                            let v = m_[i][j];
                            let color = if kind == 1 && i > j {
                                pal::FAINT
                            } else if kind == 1 && shifted && i == m - 1 && j == m - 1 {
                                SHIFT
                            } else {
                                pal::TEXT
                            };
                            (format!("${}$", num(v, 2)), color)
                        })
                        .collect()
                })
                .collect()
        };
        let top = at.y + 22.0;
        let mid = top + matrix_size(m, m, widest, size).y * 0.5;
        let mut x = at.x;
        f.text_px(Vec2::new(x, mid), format!("$Q_{{{k}}} =$"), 12.0, pal::TEXT, Align2::LEFT_CENTER);
        x += 36.0;
        x += hud_matrix(f, Vec2::new(x, top), &rows(&q, 0), widest, size, pal::DIM).x + 16.0;
        f.text_px(Vec2::new(x, mid), format!("$R_{{{k}}} =$"), 12.0, pal::TEXT, Align2::LEFT_CENTER);
        x += 36.0;
        hud_matrix(f, Vec2::new(x, top), &rows(&r, 1), widest, size, pal::DIM);
        if shifted {
            let lines = [
                format!(r"$R_{{{k}}}$'s last diagonal entry: ${}$", short(r[m - 1][m - 1])),
                format!(r"the product of its diagonal is $|\det(A_{{{k}}} - sI)|$, small when $s$ is near an eigenvalue"),
            ];
            let y = top + matrix_size(m, m, widest, size).y + 6.0;
            let room = lay.words.map_or(300.0, |(_, width)| width);
            for (n, line) in lines.iter().enumerate() {
                let small = lay.small - 0.5;
                let w = f.text_size(line, small).x;
                let s = if w > room && w > 0.0 { (small * room / w).max(8.0) } else { small };
                f.text_px(Vec2::new(at.x, y + n as f32 * line_h(small)), line.as_str(), s, pal::DIM, Align2::LEFT_TOP);
            }
        }
    }

    /// The chart of the entries below the diagonal against the step.
    fn chart(&self, f: &mut Frame, lay: &Layout) {
        let Some(rect) = lay.chart else { return };
        let span = self.span.get(0).max(1.0) as f64;
        let p = Plot::new(rect, [0.0, span], [FLOOR, CEILING]);
        if p.tiny() {
            return;
        }
        let (min, max) = rect;
        let small = lay.small;
        // the title and the key
        let title = if lay.narrow { "below the diagonal" } else { "the entries below the diagonal, step by step" };
        f.text_px(Vec2::new(min.x - 30.0, min.y - CHART_ABOVE + 2.0), title, small + 0.5, pal::TEXT, Align2::LEFT_TOP);
        let mut x = min.x - 30.0;
        let key_y = min.y - CHART_ABOVE + 24.0;
        for (i, color) in SUBS.iter().enumerate() {
            let name = format!("$|{}|$", entry(i + 1, i));
            hud_rect(f, Vec2::new(x, key_y - 4.5), Vec2::new(x + 9.0, key_y + 4.5), *color);
            f.text_px(Vec2::new(x + 13.0, key_y), name.as_str(), small - 0.5, pal::DIM, Align2::LEFT_CENTER);
            x += 13.0 + f.text_size(&name, small - 0.5).x + 10.0;
        }
        if !lay.narrow {
            hud_rect(f, Vec2::new(x, key_y - 1.0), Vec2::new(x + 9.0, key_y + 1.0), OTHERS);
            f.text_px(Vec2::new(x + 13.0, key_y), "the other three", small - 0.5, pal::DIM, Align2::LEFT_CENTER);
            let dashed = match (self.run.plain(), self.run.predicted.iter().any(Option::is_some)) {
                (true, true) => r"dashed: the predicted rates, $|\lambda_{i+1}/\lambda_i|$ a step",
                (true, false) => "no rates are drawn: two real eigenvalues have the same size",
                _ => "",
            };
            f.text_px(Vec2::new(min.x - 30.0, key_y + 15.0), dashed, small - 0.5, pal::DIM, Align2::LEFT_CENTER);
        }
        // the powers of ten up the side
        for e in (FLOOR as i32)..=0 {
            let y = p.px_y(e as f64);
            let named = e % 4 == 0;
            if e % 2 == 0 {
                plot::line_of(f, plot::AXES, Vec2::new(min.x, y), Vec2::new(max.x, y), 1.0, pal::FAINT.a(if named { 0.55 } else { 0.25 }));
            }
            if named {
                let text = if e == 0 { "$1$".to_string() } else { format!("$10^{{{e}}}$") };
                f.text_px(Vec2::new(min.x - 5.0, y), text, small - 1.0, pal::DIM, Align2::RIGHT_CENTER);
            }
        }
        p.x_axis(f, "", small - 1.0);
        f.text_px(Vec2::new(max.x, max.y + 8.0 + small), "step $k$", small - 1.0, pal::DIM, Align2::RIGHT_TOP);
        // the old run fades out, the new one in
        let fade = smoothstep(self.blend);
        if let Some((_, run, shown)) = &self.from {
            self.series(f, &p, run, *shown, 1.0 - fade, lay.narrow);
        } else {
            for kind in [SERIES, SERIES + 1, SERIES + 2, SERIES + 3, SERIES + 4, SERIES + 5, PREDICTED] {
                plot::apart(f, kind, kind == PREDICTED);
            }
        }
        self.series(f, &p, &self.run, self.shown, if self.from.is_some() { fade } else { 1.0 }, lay.narrow);
    }

    /// One run's lines on the chart, up to the step `now`, at the strength `alpha`.
    fn series(&self, f: &mut Frame, p: &Plot, run: &Run, now: f64, alpha: f32, narrow: bool) {
        let last = run.last();
        let now = now.max(0.0).min(last as f64);
        let k0 = (now.floor() as usize).min(last);
        let k1 = (k0 + 1).min(last);
        let u = now - k0 as f64;
        let (x0, x1) = (p.x[0], p.x[1]);
        // the predicted lines, faint and dashed
        for (i, line) in run.predicted.iter().enumerate() {
            let Some((c, r)) = *line else { continue };
            let pts = [p.at(x0, c + x0 * r), p.at(x1, c + x1 * r)];
            let cut = plot::clip(&pts, p.rect);
            for w in cut.windows(2) {
                plot::dashed_of(f, PREDICTED, w[0], w[1], 1.3, SUBS[i].a(0.55 * alpha));
            }
        }
        // each entry below the diagonal: bright up to the step drawn, and gathered at
        // its head beyond it (the same number of points in every frame)
        for (e, (i, j)) in [(1, 0), (2, 1), (3, 2), (2, 0), (3, 1), (3, 0)].into_iter().enumerate() {
            let color = if e < 3 { SUBS[e] } else { OTHERS.a(0.55) };
            let y = |k: usize| run.a(k)[i][j].abs().max(1e-300).log10();
            let head = (now, y(k0) + (y(k1) - y(k0)) * u);
            let pts: Vec<(f64, f64)> = (0..=LAST).map(|k| if (k as f64) < now && k <= last { (k as f64, y(k)) } else { head }).collect();
            p.path(f, SERIES + e as u8, pts, if e < 3 { 2.0 } else { 1.2 }, color.a(color.0[3] as f32 / 255.0 * alpha));
            if e < 3 {
                p.dot(f, head.0, head.1, if narrow { 5.0 } else { 6.5 }, color.a(alpha));
            }
        }
    }

    /// The eigenvalues in the complex plane: the true ones as rings, the diagonal's
    /// estimates as dots, with the paths they took.
    fn plane(&self, f: &mut Frame, lay: &Layout, look: &Look) {
        let Some((min, max)) = lay.plane else { return };
        if max.x - min.x < 40.0 || max.y - min.y < 30.0 {
            return;
        }
        let small = lay.small;
        let title = if lay.narrow { "eigenvalues" } else { "the eigenvalues in the complex plane" };
        let above = if lay.narrow { CHART_ABOVE } else { PLANE_ABOVE };
        f.text_px(Vec2::new(min.x, min.y - above + 2.0), title, small + 0.5, pal::TEXT, Align2::LEFT_TOP);
        if !lay.narrow {
            let key = r"green rings: the eigenvalues of $A$      white dots: what the diagonal of $A_k$ says";
            f.text_px(Vec2::new(min.x, min.y - PLANE_ABOVE + 22.0), key, small - 0.5, pal::DIM, Align2::LEFT_TOP);
        }
        let [lo, hi, half] = [self.plane.get(0), self.plane.get(1), self.plane.get(2)];
        let scale = ((max.x - min.x) / (hi - lo).max(1e-3)).min((max.y - min.y) / (2.0 * half).max(1e-3));
        let mid = (min + max) * 0.5;
        let centre = 0.5 * (lo + hi);
        let at = |z: DVec2| mid + Vec2::new(z.x as f32 - centre, -(z.y as f32)) * scale;
        let inside = |p: Vec2| p.max(min).min(max);
        // the axes, and the whole numbers along the real one
        let (left, right) = (at(DVec2::new(lo as f64, 0.0)).x, at(DVec2::new(hi as f64, 0.0)).x);
        let zero = at(DVec2::ZERO);
        plot::line_of(f, plot::AXES, Vec2::new(left, zero.y), Vec2::new(right, zero.y), 1.0, pal::DIM.a(0.7));
        if zero.x >= min.x && zero.x <= max.x {
            plot::line_of(f, plot::AXES, Vec2::new(zero.x, at(DVec2::new(0.0, half as f64)).y), Vec2::new(zero.x, at(DVec2::new(0.0, -half as f64)).y), 1.0, pal::DIM.a(0.5));
        }
        let eig = PRESETS[self.preset].eig;
        for n in (lo.ceil() as i32)..=(hi.floor() as i32) {
            let x = at(DVec2::new(n as f64, 0.0)).x;
            plot::line_of(f, plot::AXES, Vec2::new(x, zero.y), Vec2::new(x, zero.y + 4.0), 1.0, pal::DIM);
            // (the number is left out where a ring on the line is, which is named above it)
            let ring = eig.iter().any(|(ex, ey)| *ey == 0.0 && (at(DVec2::new(*ex, 0.0)).x - x).abs() < 12.0);
            f.text_px(Vec2::new(x, zero.y + 5.0), if ring { String::new() } else { format!("${n}$") }, small - 1.5, pal::DIM, Align2::CENTER_TOP);
        }
        if half >= 1.0 {
            for n in [-1i32, 1] {
                let y = at(DVec2::new(0.0, n as f64)).y;
                if zero.x >= min.x && zero.x <= max.x {
                    plot::line_of(f, plot::AXES, Vec2::new(zero.x - 3.0, y), Vec2::new(zero.x + 3.0, y), 1.0, pal::DIM);
                    f.text_px(Vec2::new(zero.x - 5.0, y), if n > 0 { "$i$" } else { "$-i$" }, small - 1.5, pal::DIM, Align2::RIGHT_CENTER);
                }
            }
        }
        // the true eigenvalues (the old matrix's fading out, after a change)
        let fade = if self.from.is_some() { smoothstep(self.blend) } else { 1.0 };
        if let Some((_, old, _)) = &self.from {
            rings(f, old.preset, 1.0 - fade, &at, small);
        }
        rings(f, self.preset, fade, &at, small);
        // the paths the dots took, and the dots
        let run = &self.run;
        let now = self.shown.max(0.0).min(run.last() as f64);
        // (each trail ends where its run's own dot is: while an old matrix fades into a
        // new one, the dots drawn are between the two, and the old trails fade out as
        // the new ones fade in)
        if let Some((_, old, shown)) = &self.from {
            trails(f, old, *shown, 1.0 - fade, &at, min, max);
        }
        trails(f, run, now, fade, &at, min, max);
        // the guess s, on the real line
        if look.part_alpha > 0.0 {
            let s = inside(at(DVec2::new(look.shift, 0.0)));
            plot::line_of(f, plot::AXES, s - Vec2::new(0.0, 9.0), s + Vec2::new(0.0, 9.0), 2.0, SHIFT.a(look.part_alpha));
            f.text_px(s + Vec2::new(0.0, 11.0), "$s$", small, SHIFT.mix(pal::WHITE, 0.3).a(look.part_alpha), Align2::CENTER_TOP);
        } else {
            plot::line_of(f, plot::AXES, Vec2::ZERO, Vec2::ZERO, 2.0, SHIFT.a(0.0));
        }
        for dot in look.dots {
            plot::disc(f, inside(at(dot)), 9.0, DIAG);
        }
    }
}

/// The rings of preset `p`'s eigenvalues in the plane (`at` takes a complex number
/// to the screen), with their values beside them: above the line, and a row higher
/// for one too close to the one before to share the row.
fn rings(f: &mut Frame, p: usize, alpha: f32, at: &dyn Fn(DVec2) -> Vec2, small: f32) {
    let eig = PRESETS[p.min(PRESETS.len() - 1)].eig;
    let mut order: Vec<usize> = (0..N).collect();
    order.sort_by(|a, b| eig[*a].0.total_cmp(&eig[*b].0).then(eig[*b].1.total_cmp(&eig[*a].1)));
    let (mut last_x, mut raised) = (f32::MIN, false);
    for &n in &order {
        let (x, y) = eig[n];
        let c = at(DVec2::new(x, y));
        hud_circle(f, c, 7.0, 1.6, TRUE.a(alpha));
        if y == 0.0 {
            raised = c.x - last_x < 30.0 && !raised;
            last_x = c.x;
            let up = if raised { 23.0 } else { 10.0 };
            f.text_px(c - Vec2::new(0.0, up), format!("${}$", short_value(x)), small - 0.5, TRUE.a(alpha), Align2::CENTER_BOTTOM);
        } else if y > 0.0 {
            f.text_px(c + Vec2::new(11.0, -6.0), format!(r"${} \pm {}i$", short_value(x), short_value(y)), small - 0.5, TRUE.a(alpha), Align2::LEFT_BOTTOM);
        }
    }
}

/// The paths the dots of a run took up to the step `now`, each ending where the run's
/// own dot is then (the same number of points in every frame: those not yet reached
/// are gathered there), kept inside the plane's rectangle.
fn trails(f: &mut Frame, run: &Run, now: f64, alpha: f32, at: &dyn Fn(DVec2) -> Vec2, min: Vec2, max: Vec2) {
    let now = now.max(0.0).min(run.last() as f64);
    let k0 = (now.floor() as usize).min(run.last());
    let inside = |p: Vec2| p.max(min).min(max);
    let own = Look::at(run, now);
    for (i, dot) in own.dots.iter().enumerate() {
        let head = inside(at(*dot));
        let pts: Vec<Vec2> = (0..=LAST).map(|k| if k <= k0 && (k as f64) < now { inside(at(estimates(run.a(k))[i])) } else { head }).collect();
        plot::path_of(f, TRAILS + i as u8, &pts, 1.0, DIAG.a(0.28 * alpha));
    }
}

/// "step" or "steps".
fn steps(k: usize) -> &'static str {
    if k == 1 {
        "step"
    } else {
        "steps"
    }
}

/// A value for a label: as few decimals as it needs, up to two.
fn short_value(v: f64) -> String {
    let two = num(v, 2);
    let trimmed = two.trim_end_matches('0').trim_end_matches('.');
    if trimmed.is_empty() || trimmed == "−" {
        "0".into()
    } else {
        trimmed.to_string()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn close(a: &M, b: &M, tol: f64) -> bool {
        size(&minus(a, b)) <= tol
    }

    /// A = QR with Q orthogonal and R upper triangular with a diagonal of no negative
    /// numbers, for many matrices and for singular ones, and every block size.
    #[test]
    fn qr_factors_a_matrix() {
        let mut rng = Rng::new(7);
        let mut cases: Vec<M> = (0..200).map(|_| std::array::from_fn(|_| std::array::from_fn(|_| rng.range(-3.0, 3.0) as f64))).collect();
        cases.push([[0.0; N]; N]);
        cases.push(ID);
        cases.push([[1.0, 2.0, 3.0, 4.0], [2.0, 4.0, 6.0, 8.0], [0.0, 1.0, 0.0, 1.0], [1.0, 1.0, 1.0, 1.0]]);
        cases.extend((0..PRESETS.len()).map(matrix));
        for a in cases {
            for m in 1..=N {
                let (q, r) = qr(&a, m);
                let mut part = [[0.0; N]; N];
                for i in 0..m {
                    part[i][..m].copy_from_slice(&a[i][..m]);
                }
                assert!(close(&mul(&q, &r), &part, 1e-12 * size(&a).max(1.0)), "{a:?} {m}");
                assert!(close(&mul(&transpose(&q), &q), &ID, 1e-13), "{q:?}");
                for i in 0..N {
                    assert!(r[i][i] >= 0.0);
                    for j in 0..i {
                        assert_eq!(r[i][j], 0.0);
                    }
                }
            }
        }
        // the determinant by elimination against a product of known factors
        let a = matrix(0);
        assert!((det(&a) - 4.0 * 2.0 * 0.7 * 0.14).abs() < 1e-12);
    }

    /// Every step, plain or shifted, is QᵀAQ with Q orthogonal: the trace, the
    /// determinant and A = Q Aₖ Qᵀ all hold to rounding, all the way along every run.
    #[test]
    fn every_step_keeps_the_eigenvalues() {
        for p in 0..PRESETS.len() {
            let a = matrix(p);
            let scale = size(&a);
            for shifted in [false, true] {
                let run = Run::new(p, shifted);
                for (k, step) in run.steps.iter().enumerate() {
                    let ak = &step.a;
                    assert!((trace(ak) - trace(&a)).abs() < 1e-12 * scale, "{} {shifted} step {k}: trace", PRESETS[p].name);
                    assert!((det(ak) - det(&a)).abs() < 1e-12 * scale.powi(4), "{} {shifted} step {k}: det {} {}", PRESETS[p].name, det(ak), det(&a));
                    assert!(close(&mul(&transpose(&step.q), &step.q), &ID, 1e-12));
                    assert!(close(&mul(&mul(&step.q, ak), &transpose(&step.q)), &a, 1e-12 * scale), "{} {shifted} step {k}", PRESETS[p].name);
                    // the step from here is QᵀAQ with the Q it was made with
                    if k < run.last() {
                        let (_, q) = match step.next {
                            Next::Plain => plain_step(ak),
                            Next::Shifted { .. } => shifted_step(ak).map(|(b, q, ..)| (b, q)).expect("a shifted step"),
                            other => panic!("{other:?} before the end"),
                        };
                        assert!(close(&mul(&mul(&transpose(&q), ak), &q), &run.steps[k + 1].a, 1e-12 * scale));
                    }
                }
            }
        }
    }

    /// The presets have the eigenvalues they are said to have: each is Q₀TQ₀ᵀ with Q₀
    /// orthogonal and T triangular but for a block whose eigenvalues are the pair; the
    /// trace and determinant agree; and the shifted algorithm, run to the end (a
    /// computation of its own), finds them.
    #[test]
    fn the_presets_have_the_eigenvalues_they_say() {
        let q = q0();
        assert!(close(&mul(&transpose(&q), &q), &ID, 1e-14));
        let near = |a: &[DVec2], b: &[DVec2], tol: f64| {
            let mut left: Vec<DVec2> = b.to_vec();
            a.iter().all(|z| match left.iter().position(|w| (*w - *z).length() < tol) {
                Some(i) => {
                    left.remove(i);
                    true
                }
                None => false,
            })
        };
        for (p, preset) in PRESETS.iter().enumerate() {
            let eig: Vec<DVec2> = preset.eig.iter().map(|(x, y)| DVec2::new(*x, *y)).collect();
            assert!(near(&estimates(&preset.t), &eig, 1e-12), "{}: T", preset.name);
            let a = matrix(p);
            assert!((trace(&a) - eig.iter().map(|z| z.x).sum::<f64>()).abs() < 1e-12);
            let product = eig.iter().fold(DVec2::new(1.0, 0.0), |acc, z| DVec2::new(acc.x * z.x - acc.y * z.y, acc.x * z.y + acc.y * z.x));
            assert!((det(&a) - product.x).abs() < 1e-12 && product.y.abs() < 1e-12, "{}", preset.name);
            // in order of size, largest first
            assert!(eig.windows(2).all(|w| w[0].length() >= w[1].length() - 1e-12), "{}", preset.name);
            let run = Run::new(p, true);
            assert!(run.next(run.last()) == Next::Found, "{}: found", preset.name);
            assert!(near(&estimates(run.a(run.last())), &eig, 1e-9), "{}: {:?}", preset.name, estimates(run.a(run.last())));
        }
    }

    /// Without shifts, each entry just below the diagonal shrinks by the predicted
    /// rate in the end (to a part in a thousand, steps 60 to 70; for the matrix whose
    /// eigenvalues are well apart, already by steps 30 to 40), and an entry between
    /// a complex pair stays as it is; no rate is claimed for 2 and −2.
    #[test]
    fn the_rates_are_the_predicted_ones() {
        for p in 0..PRESETS.len() {
            let run = Run::new(p, false);
            let rates = rates(p);
            if p == 4 {
                assert!(rates.iter().all(Option::is_none) && run.predicted.iter().all(Option::is_none));
                continue;
            }
            for (i, r) in rates.iter().enumerate() {
                let r = r.expect("a rate");
                let v = |k: usize| run.a(k)[i + 1][i].abs();
                let measured = (v(70) / v(60)).powf(0.1);
                assert!((measured - r).abs() < 1e-3 * r, "{} {i}: {measured} against {r}", PRESETS[p].name);
                // the dashed line goes through the measured one there
                let (c, lr) = run.predicted[i].expect("a line");
                assert!((c + 65.0 * lr - v(65).log10()).abs() < 0.01, "{} {i}", PRESETS[p].name);
            }
        }
        // the numbers the notes give
        let apart = Run::new(0, false);
        for (i, r) in rates(0).iter().enumerate() {
            let v = |k: usize| apart.a(k)[i + 1][i].abs();
            assert!(((v(40) / v(30)).powf(0.1) - r.unwrap_or(0.0)).abs() < 1e-4);
        }
        let rate = |p: usize, i: usize| num(rates(p)[i].unwrap_or(f64::NAN), 2);
        assert_eq!([rate(0, 0), rate(0, 1), rate(0, 2)], ["0.50", "0.35", "0.20"]);
        assert_eq!(rate(2, 1), "0.90");
        assert_eq!([rate(3, 0), rate(3, 1), rate(3, 2)], ["0.50", "1.00", "0.25"]);
    }

    /// The scene as an experiment's button leaves it, run at 60 frames a second until
    /// it comes to rest.
    fn trial(i: usize) -> Schur {
        let mut scene = Schur::new();
        scene.set_up(i);
        for _ in 0..60 * 120 {
            scene.advance(1.0 / 60.0, 1.0 / 60.0);
            if !scene.moving() {
                break;
            }
        }
        assert!(!scene.moving() && !scene.playing);
        scene
    }

    fn diagonal(a: &M) -> [f64; N] {
        std::array::from_fn(|i| a[i][i])
    }

    /// The captions of the "things to try", held to account.
    #[test]
    fn well_apart_does_what_its_caption_says() {
        let scene = trial(0);
        assert!(scene.k == 40 && scene.preset == 0 && !scene.shifts);
        let a = scene.run.a(40);
        let a0 = matrix(0);
        assert!((trace(a) - trace(&a0)).abs() < 1e-12 && (det(a) - det(&a0)).abs() < 1e-12);
        assert_eq!(tiny(below(a)), r"4 \times 10^{-12}");
        assert_eq!(diagonal(a).map(short_value), ["4", "2", "0.7", "0.14"]);
        // A₄₀ is T but for the signs of some entries: D T D with D = diag(±1)
        let t = &PRESETS[0].t;
        let d: [f64; N] = std::array::from_fn(|j| if j == 0 { 1.0 } else { (a[0][j] / t[0][j]).signum() });
        let mut dtd = *t;
        for (i, row) in dtd.iter_mut().enumerate() {
            for (j, v) in row.iter_mut().enumerate() {
                *v *= d[i] * d[j];
            }
        }
        assert!(close(a, &dtd, 1e-10), "{a:?}");
        assert!(d.iter().any(|v| *v < 0.0), "the signs do differ");
    }

    #[test]
    fn the_symmetric_matrix_does_what_its_caption_says() {
        let scene = trial(1);
        assert!(scene.k == 40 && scene.preset == 1);
        let a0 = matrix(1);
        assert!(close(&a0, &transpose(&a0), 1e-14));
        for k in 0..=40 {
            let a = scene.run.a(k);
            assert!(close(a, &transpose(a), 1e-12), "step {k}");
        }
        let a = scene.run.a(40);
        assert_eq!(diagonal(a).map(short_value), ["4", "−2", "0.8", "0.2"]);
        let mut off = 0.0f64;
        for i in 0..N {
            for j in 0..N {
                if i != j {
                    off = off.max(a[i][j].abs());
                }
            }
        }
        assert_eq!(tiny(off), r"3 \times 10^{-12}");
    }

    #[test]
    fn close_in_size_does_what_its_caption_says() {
        let scene = trial(2);
        assert!(scene.k == 100 && scene.preset == 2 && !scene.shifts);
        let run = &scene.run;
        let a = run.a(100);
        assert_eq!(tiny(a[2][1]).trim_start_matches('-'), r"1 \times 10^{-6}");
        assert!(num(a[1][1], 6) == "2.000006" && num(a[2][2], 6) == "1.799994", "{} {}", a[1][1], a[2][2]);
        let first = |i: usize| (0..=run.last()).find(|&k| run.a(k)[i + 1][i].abs() < 1e-16);
        assert_eq!((first(0), first(2)), (Some(58), Some(28)));
        // carried on past the run
        let (mut b, mut k) = (*run.a(LAST), LAST);
        while b[2][1].abs() >= 1e-16 {
            b = plain_step(&b).0;
            k += 1;
            assert!(k < 1000);
        }
        assert_eq!(k, 320);
    }

    #[test]
    fn shifts_do_what_their_caption_says() {
        let scene = trial(3);
        let run = &scene.run;
        assert!(scene.shifts && scene.preset == 2 && scene.k == 13 && run.last() == 13 && run.next(13) == Next::Found);
        let v: Vec<f64> = (0..=5).map(|k| run.a(k)[3][2].abs()).collect();
        let said = [num(v[0], 2), num(v[1], 2), num(v[2], 3), tiny(v[3]), tiny(v[4]), tiny(v[5])];
        assert_eq!(said, ["1.26", "0.09", "0.003", r"1 \times 10^{-5}", r"5 \times 10^{-10}", r"1 \times 10^{-19}"]);
        // the zeros after the point about doubling, step by step
        for k in 2..5 {
            let ratio = v[k + 1].log10() / v[k].log10();
            assert!((1.7..2.3).contains(&ratio), "step {k}: {ratio}");
        }
        // then that row is done, with 4 in its corner, and the work moves up a row
        assert!(matches!(run.next(4), Next::Shifted { lo: 0, hi: 3, .. }));
        assert!(matches!(run.next(5), Next::Shifted { lo: 0, hi: 2, .. }));
        assert!((run.a(5)[3][3] - 4.0).abs() < 1e-12);
        // against 220 without shifts (by the same measure of done)
        let (mut a, mut k) = (matrix(2), 0);
        while !(1..N).all(|i| small(&a, i)) {
            a = plain_step(&a).0;
            k += 1;
        }
        assert_eq!(k, 220);
        assert!((1..N).all(|i| small(run.a(13), i)));
        // not in order of size
        let d = diagonal(run.a(13));
        assert!(d.windows(2).any(|w| w[0].abs() < w[1].abs()), "{d:?}");
    }

    #[test]
    fn a_complex_pair_does_what_its_caption_says() {
        let scene = trial(4);
        assert!(scene.k == 40 && scene.preset == 3 && !scene.shifts);
        let run = &scene.run;
        // the entry between the pair settles at 1.60
        for k in 30..=40 {
            assert_eq!(num(run.a(k)[2][1].abs(), 2), "1.60", "step {k}");
        }
        let a = run.a(40);
        assert!(num(a[1][1], 2) == "1.20" && num(a[2][2], 2) == "1.20");
        let pair = block_eig(a, 1);
        assert!((pair[0] - DVec2::new(1.2, 1.6)).length() < 1e-9 && (pair[1] - DVec2::new(1.2, -1.6)).length() < 1e-9);
        // the dots are the pair, off the real line, and the rest on it
        let dots = estimates(a);
        assert!((dots[1] - pair[0]).length() < 1e-12 && (dots[2] - pair[1]).length() < 1e-12);
        assert!((dots[0] - DVec2::new(4.0, 0.0)).length() < 1e-9 && (dots[3] - DVec2::new(0.5, 0.0)).length() < 1e-9);
        assert!(below(a) < 1e-11, "{}", below(a));
        assert!((DVec2::new(1.2, 1.6).length() - 2.0).abs() < 1e-12);
    }

    /// The notes on the matrices: 2 and −2 never settle without shifts, the complex
    /// pair keeps its block with them, and both are found.
    #[test]
    fn the_notes_on_the_matrices_hold() {
        let run = Run::new(4, false);
        for k in 40..LAST - 1 {
            let (a, b) = (run.a(k), run.a(k + 1));
            // a swing from one step to the next, the same every other step
            assert!(size(&minus(a, b)) > 0.5 && close(a, run.a(k + 2), 1e-6), "step {k}");
            let d = num(a[1][1].abs(), 2);
            assert!(d == "0.12" || d == "0.19", "{d}");
            assert!((a[1][1] + a[2][2]).abs() < 1e-6);
            assert!(num(a[1][1].abs(), 2) != num(b[1][1].abs(), 2));
            let pair = block_eig(a, 1);
            assert!((pair[0].x - 2.0).abs() < 1e-6 && (pair[1].x + 2.0).abs() < 1e-6 && pair[0].y == 0.0, "{pair:?}");
        }
        assert_eq!(Run::new(4, true).last(), 13);
        let complex = Run::new(3, true);
        assert_eq!(complex.last(), 11);
        let a = complex.a(11);
        assert!(complex_block(a, 0) && (block_eig(a, 0)[0] - DVec2::new(1.2, 1.6)).length() < 1e-6);
    }

    /// Without shifts, the first column of Q₀Q₁⋯Qₖ₋₁ is Aᵏ(1, 0, 0, 0) scaled to
    /// length 1: the power method.
    #[test]
    fn the_first_column_is_the_power_method() {
        for p in 0..PRESETS.len() {
            let run = Run::new(p, false);
            let a = matrix(p);
            let mut x = [1.0, 0.0, 0.0, 0.0];
            for k in 1..=30 {
                x = std::array::from_fn(|i| (0..N).map(|j| a[i][j] * x[j]).sum());
                let len = x.iter().map(|v| v * v).sum::<f64>().sqrt();
                x = x.map(|v| v / len);
                let q = &run.steps[k].q;
                assert!((0..N).all(|i| (q[i][0] - x[i]).abs() < 1e-9), "{} step {k}", PRESETS[p].name);
            }
        }
    }

    /// And its first j columns span what Aᵏ makes of the first j axes.
    #[test]
    fn the_first_columns_span_what_a_power_makes_of_the_first_axes() {
        for p in 0..PRESETS.len() {
            let run = Run::new(p, false);
            let mut power = ID;
            for k in 1..=20 {
                power = mul(&matrix(p), &power);
                let q = &run.steps[k].q;
                for j in 1..=N {
                    for c in 0..j {
                        // column c of Aᵏ, scaled to length 1, less its shadow on q's first j columns
                        let len = (0..N).map(|i| power[i][c] * power[i][c]).sum::<f64>().sqrt();
                        let x: [f64; N] = std::array::from_fn(|i| power[i][c] / len);
                        let mut rest = x;
                        for col in 0..j {
                            let d: f64 = (0..N).map(|i| q[i][col] * x[i]).sum();
                            for (i, r) in rest.iter_mut().enumerate() {
                                *r -= d * q[i][col];
                            }
                        }
                        assert!(rest.iter().map(|v| v * v).sum::<f64>().sqrt() < 1e-9, "{} step {k}: {j} columns, axis {c}", PRESETS[p].name);
                    }
                }
            }
        }
    }

    /// The scene opens at step 10 of the first matrix, at rest; a step asked for glides
    /// there and comes exactly to rest, at any display's rate.
    #[test]
    fn it_opens_at_rest_and_a_step_glides() {
        let scene = Schur::new();
        assert!(scene.k == OPENING && scene.preset == 0 && !scene.shifts && !scene.moving());
        for hz in [30.0f32, 60.0, 144.0, 240.0] {
            let mut scene = Schur::new();
            scene.one_step();
            let mut last = scene.shown;
            let mut frames = 0;
            while scene.moving() {
                scene.advance(1.0 / hz, 1.0 / hz);
                assert!(scene.shown >= last && scene.shown <= (OPENING + 1) as f64);
                last = scene.shown;
                frames += 1;
                assert!(frames as f32 / hz < 1.5, "still moving at {hz} Hz");
            }
            assert_eq!(scene.shown, (OPENING + 1) as f64);
        }
    }

    /// Shifts switched on part of the way carry on from there; start again then makes
    /// the run afresh, all of one kind.
    #[test]
    fn shifts_switched_on_late_carry_on() {
        let mut scene = Schur::new();
        scene.choose(2);
        scene.k = 30;
        scene.set_shifts(true);
        assert!(!scene.run.pure && scene.run.next(29) == Next::Plain && matches!(scene.run.next(30), Next::Shifted { .. }));
        assert!(scene.run.next(scene.run.last()) == Next::Found && scene.run.last() < 40, "{}", scene.run.last());
        assert!(scene.run.predicted.iter().all(Option::is_none));
        scene.again();
        assert!(scene.run.pure && scene.run.shifted && scene.k == 0 && scene.run.last() == 13);
        // and back off from the start: plain again, with its predictions
        scene.set_shifts(false);
        assert!(scene.run.plain() && scene.run.predicted.iter().all(Option::is_some));
    }

    /// Whatever is pressed, in whatever order, with frames of any length, everything
    /// drawn from stays finite and nothing is out of range.
    #[test]
    fn any_input_leaves_everything_finite() {
        let mut rng = Rng::new(3);
        let mut scene = Schur::new();
        for n in 0..4000 {
            match rng.below(9) {
                0 => scene.one_step(),
                1 => scene.toggle_play(),
                2 => scene.again(),
                3 => scene.set_shifts(!scene.shifts),
                4 => scene.choose(rng.below(PRESETS.len() + 2)),
                5 => scene.set_up(rng.below(TRIALS.len())),
                6 => scene.rate = rng.range(0.5, 20.0),
                _ => {}
            }
            let dt = match rng.below(5) {
                0 => 0.0,
                1 => 1.0 / 240.0,
                2 => 0.5,
                3 => 5.0,
                _ => 1.0 / 60.0,
            };
            scene.advance(dt, dt);
            assert!(scene.k <= scene.run.last() && scene.shown.is_finite() && scene.between.is_finite(), "{n}");
            let look = scene.look();
            assert!(look.level.iter().flatten().all(|v| v.is_finite()) && look.dots.iter().all(|d| d.is_finite()) && look.shift.is_finite(), "{n}");
            assert!(look.part.iter().all(|v| v.is_finite()) && look.part_alpha.is_finite());
            assert!(scene.span.shown.iter().chain(&scene.plane.shown).all(|v| v.is_finite()));
        }
    }

    /// A cell's number: two decimals, then one figure and a power of ten.
    #[test]
    fn numbers_are_written_plainly() {
        assert_eq!(cell(1.234), "$1.23$");
        assert_eq!(cell(-0.004), r"$-4 \times 10^{-3}$");
        assert_eq!(cell(9.6e-7), r"$1 \times 10^{-6}$");
        assert_eq!(cell(0.0), "$0$");
        assert_eq!(cell(f64::NAN), "?");
        assert_eq!(short_value(0.14), "0.14");
        assert_eq!(short_value(-2.0), "−2");
        assert_eq!(complex(DVec2::new(1.2, -1.6), 2), "1.20 - 1.60i");
        assert_eq!(entry(2, 1), "a_{32}");
    }
}

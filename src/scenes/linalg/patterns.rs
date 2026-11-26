//! A zoo of larger matrices: n × n matrices (n from 3 to 8) known by their pattern,
//! where their zeros are, how their entries repeat, what their rows or columns add up
//! to, and what each pattern buys: what the matrix does to a list of n numbers, how
//! many numbers it takes to store and how much work to solve with, where its
//! eigenvalues can lie, and how near it comes to having no inverse.

use super::*;
use crate::scenes::stats::plot::{self, Rect};
use glam::DVec2;

pub const DEF: SceneDef = SceneDef {
    id: "patterns",
    title: "A zoo of larger matrices",
    group: SPECIAL,
    blurb: "Banded, circulant, companion, stochastic and more: matrices known by their pattern, what each pattern buys, and where its eigenvalues lie.",
    make: |_| Box::new(Patterns::new()),
};

// ----- matrices up to 8 × 8, in double precision ---------------------------------------

/// The largest and the smallest n.
const MAX: usize = 8;
const LEAST: usize = 3;

/// An n × n matrix, n up to 8, by rows: entry (i, j) is `a[i * MAX + j]`. (Room for the
/// largest, so that a matrix is a plain value to copy and keep.)
#[derive(Clone, Copy, Debug, PartialEq)]
struct Mat {
    n: usize,
    a: [f64; MAX * MAX],
}

impl Mat {
    fn zero(n: usize) -> Mat {
        Mat { n: n.clamp(1, MAX), a: [0.0; MAX * MAX] }
    }

    fn from_fn(n: usize, mut f: impl FnMut(usize, usize) -> f64) -> Mat {
        let mut m = Mat::zero(n);
        for i in 0..m.n {
            for j in 0..m.n {
                m.a[i * MAX + j] = f(i, j);
            }
        }
        m
    }

    #[cfg(test)]
    fn identity(n: usize) -> Mat {
        Mat::from_fn(n, |i, j| if i == j { 1.0 } else { 0.0 })
    }

    fn get(&self, i: usize, j: usize) -> f64 {
        self.a[(i * MAX + j).min(MAX * MAX - 1)]
    }

    fn set(&mut self, i: usize, j: usize, v: f64) {
        if i < MAX && j < MAX {
            self.a[i * MAX + j] = v;
        }
    }

    fn mul(&self, b: &Mat) -> Mat {
        let n = self.n.min(b.n);
        Mat::from_fn(n, |i, j| (0..n).map(|k| self.get(i, k) * b.get(k, j)).sum())
    }

    /// A x, for a list x of n numbers (missing ones count as 0).
    fn apply(&self, x: &[f64]) -> Vec<f64> {
        (0..self.n).map(|i| (0..self.n).map(|j| self.get(i, j) * x.get(j).copied().unwrap_or(0.0)).sum()).collect()
    }

    #[cfg(test)]
    fn transpose(&self) -> Mat {
        Mat::from_fn(self.n, |i, j| self.get(j, i))
    }

    fn trace(&self) -> f64 {
        (0..self.n).map(|i| self.get(i, i)).sum()
    }

    /// The size of the largest entry.
    fn largest(&self) -> f64 {
        (0..self.n).flat_map(|i| (0..self.n).map(move |j| (i, j))).fold(0.0f64, |s, (i, j)| s.max(self.get(i, j).abs()))
    }

    fn finite(&self) -> bool {
        (0..self.n).all(|i| (0..self.n).all(|j| self.get(i, j).is_finite()))
    }

    /// The n × n entries alone, by rows.
    fn dense(&self) -> Vec<f64> {
        (0..self.n).flat_map(|i| (0..self.n).map(move |j| (i, j))).map(|(i, j)| self.get(i, j)).collect()
    }

    /// The determinant, by elimination with the largest pivot of each column.
    fn det(&self) -> f64 {
        let n = self.n;
        let mut m = self.dense();
        let mut d = 1.0;
        for c in 0..n {
            let p = (c..n).max_by(|&i, &j| m[i * n + c].abs().total_cmp(&m[j * n + c].abs())).unwrap_or(c);
            if m[p * n + c] == 0.0 {
                return 0.0;
            }
            if p != c {
                for k in 0..n {
                    m.swap(p * n + k, c * n + k);
                }
                d = -d;
            }
            d *= m[c * n + c];
            for r in c + 1..n {
                let f = m[r * n + c] / m[c * n + c];
                for k in c..n {
                    m[r * n + k] -= f * m[c * n + k];
                }
            }
        }
        d
    }
}

// ----- eigenvalues of any real matrix ---------------------------------------------------

/// A matrix counted from 1, as the classic eigenvalue routines are written (row and
/// column 0 unused): it keeps the translation of them exact.
type Grid1 = [[f64; MAX + 1]; MAX + 1];

/// The eigenvalues of a real matrix, as complex numbers (x, y) = x + iy, complex ones in
/// conjugate pairs: the matrix is balanced, brought to upper Hessenberg form by
/// elimination (the same eigenvalues), and then Francis's double-shift QR iteration
/// splits off one eigenvalue or a pair at a time from the bottom (as EISPACK's `hqr`
/// does). None if an entry is not a number or the iteration does not settle.
fn eigenvalues(m: &Mat) -> Option<Vec<DVec2>> {
    let n = m.n;
    if !m.finite() {
        return None;
    }
    // A triangular matrix's eigenvalues are its diagonal entries, exactly. (The iteration
    // would find them too, but only to about the n-th root of the rounding when one is
    // repeated with a single eigenvector, as for the matrices of differences and sums:
    // a scatter round the true value, which is not what the matrix has.)
    let upper = (0..n).all(|i| (0..i).all(|j| m.get(i, j) == 0.0));
    let lower = (0..n).all(|i| (i + 1..n).all(|j| m.get(i, j) == 0.0));
    if upper || lower {
        return Some((0..n).map(|i| DVec2::new(m.get(i, i), 0.0)).collect());
    }
    let mut a: Grid1 = [[0.0; MAX + 1]; MAX + 1];
    for i in 0..n {
        for j in 0..n {
            a[i + 1][j + 1] = m.get(i, j);
        }
    }
    balance(&mut a, n);
    hessenberg(&mut a, n);
    hqr(&mut a, n)
}

/// Scale rows and columns by powers of 2 (a diagonal change of basis, which keeps the
/// eigenvalues and adds no rounding) until each row and its column are of a size: the
/// iteration is then far more accurate for a matrix whose entries differ widely in size
/// (a companion matrix, a Vandermonde one).
fn balance(a: &mut Grid1, n: usize) {
    const RADIX: f64 = 2.0;
    for _ in 0..64 {
        let mut done = true;
        for i in 1..=n {
            let (mut c, mut r) = (0.0f64, 0.0f64);
            for j in 1..=n {
                if j != i {
                    c += a[j][i].abs();
                    r += a[i][j].abs();
                }
            }
            if c == 0.0 || r == 0.0 {
                continue;
            }
            let s = c + r;
            let mut f = 1.0;
            let mut steps = 0;
            while c < r / RADIX && steps < 200 {
                f *= RADIX;
                c *= RADIX * RADIX;
                steps += 1;
            }
            while c > r * RADIX && steps < 400 {
                f /= RADIX;
                c /= RADIX * RADIX;
                steps += 1;
            }
            if (c + r) / f < 0.95 * s {
                done = false;
                for j in 1..=n {
                    a[i][j] /= f;
                    a[j][i] *= f;
                }
            }
        }
        if done {
            return;
        }
    }
}

/// Bring the matrix to upper Hessenberg form (zeros below the first diagonal under the
/// main one) by elimination with row exchanges, each step undone on the other side so
/// that the eigenvalues stay.
fn hessenberg(a: &mut Grid1, n: usize) {
    for m in 2..n {
        let (mut x, mut i) = (0.0f64, m);
        for j in m..=n {
            if a[j][m - 1].abs() > x.abs() {
                x = a[j][m - 1];
                i = j;
            }
        }
        if i != m {
            for j in (m - 1)..=n {
                let t = a[i][j];
                a[i][j] = a[m][j];
                a[m][j] = t;
            }
            for row in a.iter_mut().take(n + 1).skip(1) {
                row.swap(i, m);
            }
        }
        if x != 0.0 {
            for i in (m + 1)..=n {
                let y = a[i][m - 1];
                if y != 0.0 {
                    let y = y / x;
                    a[i][m - 1] = y;
                    for j in m..=n {
                        a[i][j] -= y * a[m][j];
                    }
                    for j in 1..=n {
                        a[j][m] += y * a[j][i];
                    }
                }
            }
        }
    }
    // (what elimination leaves below the first diagonal are its multipliers)
    for (i, row) in a.iter_mut().enumerate().take(n + 1).skip(3) {
        for v in row.iter_mut().take(i - 1).skip(1) {
            *v = 0.0;
        }
    }
}

/// |a| with the sign of b (+ for b = 0).
fn sign(a: f64, b: f64) -> f64 {
    if b >= 0.0 {
        a.abs()
    } else {
        -a.abs()
    }
}

/// The eigenvalues of an upper Hessenberg matrix by the double-shift QR iteration.
fn hqr(a: &mut Grid1, n: usize) -> Option<Vec<DVec2>> {
    const ITERATIONS: usize = 60;
    let (mut wr, mut wi) = ([0.0f64; MAX + 1], [0.0f64; MAX + 1]);
    let mut anorm = 0.0;
    for i in 1..=n {
        for j in (i - 1).max(1)..=n {
            anorm += a[i][j].abs();
        }
    }
    let mut nn = n as isize;
    let mut t = 0.0;
    while nn >= 1 {
        let mut its = 0;
        loop {
            // the lowest entry below the diagonal that counts as nothing
            let mut l = nn;
            while l >= 2 {
                let lu = l as usize;
                let mut s = a[lu - 1][lu - 1].abs() + a[lu][lu].abs();
                if s == 0.0 {
                    s = anorm;
                }
                if a[lu][lu - 1].abs() + s == s {
                    a[lu][lu - 1] = 0.0;
                    break;
                }
                l -= 1;
            }
            let nu = nn as usize;
            let mut x = a[nu][nu];
            if l == nn {
                // one eigenvalue split off
                (wr[nu], wi[nu]) = (x + t, 0.0);
                nn -= 1;
            } else {
                let mut y = a[nu - 1][nu - 1];
                let mut w = a[nu][nu - 1] * a[nu - 1][nu];
                if l == nn - 1 {
                    // two: a real pair or a complex one
                    let p = 0.5 * (y - x);
                    let q = p * p + w;
                    let z = q.abs().sqrt();
                    x += t;
                    if q >= 0.0 {
                        let z = p + sign(z, p);
                        (wr[nu - 1], wr[nu]) = (x + z, x + z);
                        if z != 0.0 {
                            wr[nu] = x - w / z;
                        }
                        (wi[nu - 1], wi[nu]) = (0.0, 0.0);
                    } else {
                        (wr[nu - 1], wr[nu]) = (x + p, x + p);
                        (wi[nu - 1], wi[nu]) = (-z, z);
                    }
                    nn -= 2;
                } else {
                    if its >= ITERATIONS {
                        return None;
                    }
                    // now and then a shift of another kind, to shake loose a cycle
                    if its > 0 && its % 10 == 0 {
                        t += x;
                        for i in 1..=nu {
                            a[i][i] -= x;
                        }
                        let s = a[nu][nu - 1].abs() + a[nu - 1][nu - 2].abs();
                        x = 0.75 * s;
                        y = x;
                        w = -0.4375 * s * s;
                    }
                    its += 1;
                    // the shifts, and where to start: two small entries in a row below the diagonal
                    let mut m = nn - 2;
                    let (mut p, mut q, mut r);
                    let mut z;
                    loop {
                        let mu = m as usize;
                        z = a[mu][mu];
                        let rr = x - z;
                        let ss = y - z;
                        p = (rr * ss - w) / a[mu + 1][mu] + a[mu][mu + 1];
                        q = a[mu + 1][mu + 1] - z - rr - ss;
                        r = a[mu + 2][mu + 1];
                        let s = (p.abs() + q.abs() + r.abs()).max(f64::MIN_POSITIVE);
                        p /= s;
                        q /= s;
                        r /= s;
                        if m == l {
                            break;
                        }
                        let u = a[mu][mu - 1].abs() * (q.abs() + r.abs());
                        let v = p.abs() * (a[mu - 1][mu - 1].abs() + z.abs() + a[mu + 1][mu + 1].abs());
                        if u + v == v {
                            break;
                        }
                        m -= 1;
                    }
                    let mu = m as usize;
                    for i in (mu + 2)..=nu {
                        a[i][i - 2] = 0.0;
                        if i != mu + 2 {
                            a[i][i - 3] = 0.0;
                        }
                    }
                    // the double step, on rows l to nn and columns m to nn
                    let lu = l as usize;
                    for k in mu..nu {
                        if k != mu {
                            p = a[k][k - 1];
                            q = a[k + 1][k - 1];
                            r = if k != nu - 1 { a[k + 2][k - 1] } else { 0.0 };
                            x = p.abs() + q.abs() + r.abs();
                            if x != 0.0 {
                                p /= x;
                                q /= x;
                                r /= x;
                            }
                        }
                        let s = sign((p * p + q * q + r * r).sqrt(), p);
                        if s == 0.0 {
                            continue;
                        }
                        if k == mu {
                            if l != m {
                                a[k][k - 1] = -a[k][k - 1];
                            }
                        } else {
                            a[k][k - 1] = -s * x;
                        }
                        p += s;
                        x = p / s;
                        y = q / s;
                        z = r / s;
                        q /= p;
                        r /= p;
                        for j in k..=nu {
                            p = a[k][j] + q * a[k + 1][j];
                            if k != nu - 1 {
                                p += r * a[k + 2][j];
                                a[k + 2][j] -= p * z;
                            }
                            a[k + 1][j] -= p * y;
                            a[k][j] -= p * x;
                        }
                        let last = nu.min(k + 3);
                        for i in lu..=last {
                            p = x * a[i][k] + y * a[i][k + 1];
                            if k != nu - 1 {
                                p += z * a[i][k + 2];
                                a[i][k + 2] -= p * r;
                            }
                            a[i][k + 1] -= p * q;
                            a[i][k] -= p;
                        }
                    }
                }
            }
            if l >= nn - 1 {
                break;
            }
        }
    }
    let out: Vec<DVec2> = (1..=n).map(|i| DVec2::new(wr[i], wi[i])).collect();
    out.iter().all(|z| z.is_finite()).then_some(out)
}

/// The singular values, largest first, and the direction b is most sensitive in: the
/// unit vector A sends the least (the left singular vector of the smallest).
fn singular(m: &Mat) -> (Vec<f64>, Vec<f64>) {
    let n = m.n;
    if !m.finite() {
        return (vec![0.0; n], vec![0.0; n]);
    }
    let (u, s, _) = crate::math::linalg::svd(&m.dense(), n, n);
    let mut last: Vec<f64> = (0..n).map(|i| u.get(i * n + n - 1).copied().unwrap_or(0.0)).collect();
    // (a singular vector's sign is anyone's choice: its largest entry is made positive,
    // so that the direction does not hang on how the decomposition was found)
    let big = (0..n).max_by(|&a, &b| last[a].abs().total_cmp(&last[b].abs())).unwrap_or(0);
    if last.get(big).is_some_and(|v| *v < 0.0) {
        last.iter_mut().for_each(|v| *v = -*v);
    }
    (s, last)
}

/// x with A x = b, if A has an inverse.
fn solve(m: &Mat, b: &[f64]) -> Option<Vec<f64>> {
    let n = m.n;
    let mut a = m.dense();
    let mut b: Vec<f64> = (0..n).map(|i| b.get(i).copied().unwrap_or(0.0)).collect();
    crate::math::linalg::solve(&mut a, &mut b, n).filter(|x| x.iter().all(|v| v.is_finite()))
}

/// The multiplications and divisions that solving A x = b by elimination takes, for a
/// matrix with `p` diagonals below the main one and `q` above it (all others zero), when
/// no rows are exchanged or the exchanges keep that shape: for each column, a multiplier
/// for each row below and the rest of that row times it, then the same rows in the
/// forward substitution, and the back substitution.
fn work(n: usize, p: usize, q: usize) -> usize {
    (0..n)
        .map(|k| {
            let below = p.min(n - 1 - k);
            let right = q.min(n - 1 - k);
            below * (1 + right) + below + right + 1
        })
        .sum()
}

/// That for a full matrix.
fn full_work(n: usize) -> usize {
    work(n, n.saturating_sub(1), n.saturating_sub(1))
}

// ----- the kinds ------------------------------------------------------------------------

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Kind {
    Diagonal,
    Tridiagonal,
    Banded,
    Triangular,
    Nilpotent,
    Hessenberg,
    Blocks,
    Toeplitz,
    Circulant,
    LowRank,
    Markov,
    Laplacian,
    Dominant,
    Differences,
    Sums,
    Permutation,
    Companion,
    Vandermonde,
    Hilbert,
}

use Kind::*;

/// The kinds in families: their name over the picture, a shorter one for a phone's tab.
struct Family {
    name: &'static str,
    short: &'static str,
    kinds: &'static [Kind],
}

const FAMILIES: [Family; 4] = [
    Family { name: "by their zeros", short: "zeros", kinds: &[Diagonal, Tridiagonal, Banded, Triangular, Nilpotent, Hessenberg, Blocks] },
    Family { name: "by repeats", short: "repeats", kinds: &[Toeplitz, Circulant, LowRank] },
    Family { name: "by sums and signs", short: "sums, signs", kinds: &[Markov, Laplacian, Dominant] },
    Family { name: "with names of their own", short: "named", kinds: &[Differences, Sums, Permutation, Companion, Vandermonde, Hilbert] },
];

/// Every kind, family by family.
#[cfg(test)]
fn all_kinds() -> impl Iterator<Item = Kind> {
    FAMILIES.iter().flat_map(|f| f.kinds.iter().copied())
}

/// A whole number the reader sets for some kinds: its name, its range for an n, and
/// where it starts.
struct Param {
    name: &'static str,
    lo: usize,
    hi: usize,
    start: usize,
}

/// Whether a cell is set by the pattern or free.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Mark {
    /// Any number may stand here.
    Free,
    /// The pattern makes it 0.
    Zero,
    /// The pattern (or n alone) sets its value.
    Set,
}

impl Kind {
    fn name(self) -> &'static str {
        match self {
            Diagonal => "diagonal",
            Tridiagonal => "tridiagonal",
            Banded => "banded",
            Triangular => "triangular",
            Nilpotent => "nilpotent",
            Hessenberg => "Hessenberg",
            Blocks => "block diagonal",
            Toeplitz => "Toeplitz",
            Circulant => "circulant",
            LowRank => "low rank",
            Markov => "Markov",
            Laplacian => "graph Laplacian",
            Dominant => "diagonally dominant",
            Differences => "differences",
            Sums => "running sums",
            Permutation => "permutation",
            Companion => "companion",
            Vandermonde => "Vandermonde",
            Hilbert => "Hilbert",
        }
    }

    /// The letter it goes by on the picture.
    fn symbol(self) -> &'static str {
        match self {
            Tridiagonal => "K",
            Nilpotent => "N",
            Laplacian => "L",
            Markov => "M",
            Differences => "D",
            Sums => "S",
            Permutation => "P",
            Companion => "C",
            Vandermonde => "V",
            Hilbert => "H",
            _ => "A",
        }
    }

    fn family(self) -> usize {
        FAMILIES.iter().position(|f| f.kinds.contains(&self)).unwrap_or(0)
    }

    fn param(self, n: usize) -> Option<Param> {
        let top = n.saturating_sub(1).max(1);
        match self {
            Banded => Some(Param { name: "width", lo: 1, hi: top, start: 2.min(top) }),
            Blocks => Some(Param { name: "first block", lo: 1, hi: top, start: n.div_ceil(2).min(top) }),
            LowRank => Some(Param { name: "rank", lo: 1, hi: top, start: 1 }),
            _ => None,
        }
    }

    /// Whether n alone sets the matrix (there is no other one of the kind to show).
    fn set_by_n(self) -> bool {
        matches!(self, Tridiagonal | Differences | Sums | Hilbert)
    }

    /// What the pattern says of the cell in row i and column j.
    fn mark(self, s: &Setup, i: usize, j: usize) -> Mark {
        let (n, param) = (s.n, s.param);
        let zero_unless = |free: bool, mark: Mark| if free { mark } else { Mark::Zero };
        match self {
            Diagonal => zero_unless(i == j, Mark::Free),
            Tridiagonal => zero_unless(i.abs_diff(j) <= 1, Mark::Set),
            Banded => zero_unless(i.abs_diff(j) <= param, Mark::Free),
            Triangular => zero_unless(j >= i, Mark::Free),
            Nilpotent => zero_unless(j > i, Mark::Free),
            Hessenberg => zero_unless(i <= j + 1, Mark::Free),
            Blocks => zero_unless((i < param) == (j < param), Mark::Free),
            Toeplitz | Circulant | LowRank | Markov | Dominant => Mark::Free,
            Hilbert => Mark::Set,
            // (the graph sets the diagonal and a -1 for each edge; where there is none, 0)
            Laplacian => zero_unless(i == j || s.edges[i.min(MAX - 1)][j.min(MAX - 1)], Mark::Set),
            Differences => zero_unless(i == j || i == j + 1, Mark::Set),
            Sums => zero_unless(j <= i, Mark::Set),
            // (the shuffle sets where the 1s are, and every other cell is then 0)
            Permutation => zero_unless(shuffle(n, s.seed)[i.min(MAX - 1)] == j, Mark::Set),
            Companion => {
                if j == n - 1 {
                    Mark::Free
                } else {
                    zero_unless(i == j + 1, Mark::Set)
                }
            }
            // (the points are the second column; every other entry is a power of one)
            Vandermonde => {
                if j == 1 {
                    Mark::Free
                } else {
                    Mark::Set
                }
            }
        }
    }

    /// How it is named over the matrix.
    fn heading(self, n: usize, param: usize) -> String {
        match self {
            Diagonal => format!("a diagonal matrix $A$, $n = {n}$"),
            Tridiagonal => format!("$K$, the second-difference matrix, $n = {n}$"),
            Banded => format!("a banded matrix $A$, width $w = {param}$"),
            Triangular => format!("an upper triangular matrix $A$, $n = {n}$"),
            Nilpotent => format!("$N$, strictly upper triangular, $n = {n}$"),
            Hessenberg => format!("an upper Hessenberg matrix $A$, $n = {n}$"),
            Blocks => format!("a block diagonal matrix $A$: blocks of {param} and {}", n.saturating_sub(param)),
            Toeplitz => format!("a Toeplitz matrix $A$, $n = {n}$"),
            Circulant => format!("a circulant matrix $A$, $n = {n}$"),
            LowRank if param == 1 => format!(r"$A = uv^\top$, rank 1, $n = {n}$"),
            LowRank => format!(r"$A = u_1 v_1^\top + \dots + u_{param} v_{param}^\top$, rank {param}"),
            Markov => format!("$M$, a Markov matrix on {n} states"),
            Laplacian => format!("$L$, the Laplacian of a graph on {n} points"),
            Dominant => format!("a diagonally dominant matrix $A$, $n = {n}$"),
            Differences => format!("$D$, the difference matrix, $n = {n}$"),
            Sums => format!("$S$, the running-sum matrix, $n = {n}$"),
            Permutation => format!("$P$, a permutation matrix, $n = {n}$"),
            Companion => format!("$C$, the companion matrix of $p$, $n = {n}$"),
            Vandermonde => format!("$V$, a Vandermonde matrix, $n = {n}$ points"),
            Hilbert => format!("$H$, the Hilbert matrix, $n = {n}$"),
        }
    }
}

// ----- making a matrix of a kind -------------------------------------------------------

/// The roots of the companion matrix's polynomial: a complex pair for each of the first
/// n / 2 points (each stands for itself and its mirror image below the real line), and a
/// real root if n is odd.
#[derive(Clone, Copy, Debug, PartialEq)]
struct Roots {
    pairs: [DVec2; MAX / 2],
    real: f64,
}

impl Roots {
    /// Pairs spread round a circle of radius 1.1, and a real root at 0.5.
    fn spread(n: usize) -> Roots {
        let p = n / 2;
        let pairs = std::array::from_fn(|k| DVec2::from_angle(std::f64::consts::PI * (2 * k + 1) as f64 / (2 * p + 1).max(1) as f64) * 1.1);
        Roots { pairs, real: 0.5 }
    }

    /// Every root, the pairs as two.
    fn all(&self, n: usize) -> Vec<DVec2> {
        let mut out = Vec::with_capacity(n);
        for z in self.pairs.iter().take(n / 2) {
            out.push(*z);
            out.push(DVec2::new(z.x, -z.y));
        }
        if n % 2 == 1 {
            out.push(DVec2::new(self.real, 0.0));
        }
        out
    }

    /// The coefficients c₀, …, cₙ₋₁ of p(λ) = λⁿ + cₙ₋₁λⁿ⁻¹ + … + c₀ with these roots.
    fn coefficients(&self, n: usize) -> Vec<f64> {
        // the polynomial, lowest power first, multiplied out one factor at a time
        let mut poly = vec![1.0];
        let mut times = |factor: &[f64]| {
            let mut out = vec![0.0; poly.len() + factor.len() - 1];
            for (i, a) in poly.iter().enumerate() {
                for (j, b) in factor.iter().enumerate() {
                    out[i + j] += a * b;
                }
            }
            poly = out;
        };
        for z in self.pairs.iter().take(n / 2) {
            times(&[z.length_squared(), -2.0 * z.x, 1.0]);
        }
        if n % 2 == 1 {
            times(&[-self.real, 1.0]);
        }
        poly.truncate(n);
        poly
    }
}

/// Which points of a graph are joined (both ways).
type Edges = [[bool; MAX]; MAX];

/// Two rings joined by one edge: the first ⌈n/2⌉ points in one (a path, if there are
/// only two), the rest in the other, and an edge from the last of the first to the
/// first of the second.
fn two_rings(n: usize) -> Edges {
    let mut e = [[false; MAX]; MAX];
    let mut join = |i: usize, j: usize| {
        if i != j && i < MAX && j < MAX {
            e[i][j] = true;
            e[j][i] = true;
        }
    };
    let a = n.div_ceil(2);
    for (from, to) in [(0, a), (a, n)] {
        for k in from..to.saturating_sub(1) {
            join(k, k + 1);
        }
        if to - from >= 3 {
            join(from, to - 1);
        }
    }
    if a < n {
        join(a - 1, a);
    }
    e
}

/// The bridge of `two_rings`: the edge whose cutting splits the graph in two.
fn bridge(n: usize) -> (usize, usize) {
    let a = n.div_ceil(2);
    (a - 1, a.min(n - 1))
}

/// Which piece each point of the graph is in, the pieces numbered from 0 in the order
/// their first points come, and how many there are.
fn pieces(edges: &Edges, n: usize) -> ([usize; MAX], usize) {
    let mut piece = [usize::MAX; MAX];
    let mut count = 0;
    for start in 0..n {
        if piece[start] != usize::MAX {
            continue;
        }
        let mut stack = vec![start];
        piece[start] = count;
        while let Some(i) = stack.pop() {
            for j in 0..n {
                if edges[i][j] && piece[j] == usize::MAX {
                    piece[j] = count;
                    stack.push(j);
                }
            }
        }
        count += 1;
    }
    (piece, count)
}

/// The cycle lengths of the shuffle the permutation kind starts with.
fn first_cycles(n: usize) -> &'static [usize] {
    match n {
        0..=3 => &[3],
        4 => &[3, 1],
        5 => &[3, 2],
        6 => &[3, 2, 1],
        7 => &[4, 3],
        _ => &[4, 3, 1],
    }
}

/// The shuffle σ of the permutation kind: P has a 1 in row i and column σ(i), so that
/// (Px)ᵢ = x_σ(i). Seed 0 is made of the cycles of `first_cycles`, each on points next
/// to one another; any other seed is a random shuffle (not the identity).
fn shuffle(n: usize, seed: u64) -> [usize; MAX] {
    let mut sigma: [usize; MAX] = std::array::from_fn(|i| i);
    if seed == 0 {
        let mut start = 0;
        for &len in first_cycles(n) {
            for k in 0..len {
                if start + k < n {
                    sigma[start + k] = start + (k + 1) % len;
                }
            }
            start += len;
        }
        return sigma;
    }
    let mut rng = Rng::new(seed.wrapping_mul(7919).wrapping_add(n as u64));
    for _ in 0..20 {
        rng.shuffle(&mut sigma[..n]);
        if (0..n).any(|i| sigma[i] != i) {
            break;
        }
    }
    sigma
}

/// The cycles of a shuffle, each from its smallest point: i, σ(i), σ(σ(i)), ...
fn cycles(sigma: &[usize; MAX], n: usize) -> Vec<Vec<usize>> {
    let mut seen = [false; MAX];
    let mut out = Vec::new();
    for start in 0..n {
        if seen[start] {
            continue;
        }
        let mut cycle = Vec::new();
        let mut i = start;
        while !seen[i] && cycle.len() <= n {
            seen[i] = true;
            cycle.push(i);
            i = sigma[i].min(n - 1);
        }
        out.push(cycle);
    }
    out
}

/// The cycles written out: (1 2 3)(4 5)(6).
fn cycle_text(cs: &[Vec<usize>]) -> String {
    let parts: Vec<String> = cs.iter().map(|c| format!("({})", c.iter().map(|i| (i + 1).to_string()).collect::<Vec<_>>().join(r"\,"))).collect();
    parts.join("")
}

/// Everything that decides the matrix.
#[derive(Clone, Copy, Debug, PartialEq)]
struct Setup {
    kind: Kind,
    n: usize,
    /// The width, the first block or the rank (see `Kind::param`; 0 for other kinds).
    param: usize,
    /// 0 for the member each kind starts with, anything else for another one.
    seed: u64,
    edges: Edges,
    roots: Roots,
}

impl Setup {
    fn new(kind: Kind, n: usize) -> Setup {
        let n = n.clamp(LEAST, MAX);
        Setup { kind, n, param: kind.param(n).map_or(0, |p| p.start), seed: 0, edges: two_rings(n), roots: Roots::spread(n) }
    }
}

/// A number to one decimal: random entries read easily.
fn tenth(v: f32) -> f64 {
    (v as f64 * 10.0).round() / 10.0
}

/// A random number from lo to hi, to one decimal, at least 0.2 from 0.
fn entry(rng: &mut Rng, lo: f32, hi: f32) -> f64 {
    for _ in 0..20 {
        let v = tenth(rng.range(lo, hi));
        if v.abs() >= 0.2 {
            return v;
        }
    }
    0.5
}

/// The diagonal the diagonal kind starts with.
const FIRST_DIAGONAL: [f64; MAX] = [2.0, -1.0, 0.5, 1.5, -0.5, 1.0, -1.5, 0.8];

fn build(s: &Setup) -> Mat {
    let n = s.n.clamp(LEAST, MAX);
    let mut rng = Rng::new(s.seed.wrapping_mul(1_000_003).wrapping_add(s.kind as u64 * 101 + n as u64 * 7 + 11));
    let p = s.param;
    let random = |rng: &mut Rng, keep: &dyn Fn(usize, usize) -> bool| {
        let mut m = Mat::zero(n);
        for i in 0..n {
            for j in 0..n {
                if keep(i, j) {
                    m.set(i, j, entry(rng, -1.5, 1.5));
                }
            }
        }
        m
    };
    // each diagonal entry made larger than the rest of its row, by at least 0.3
    let dominant = |rng: &mut Rng, m: &mut Mat| {
        for i in 0..n {
            let rest: f64 = (0..n).filter(|&j| j != i).map(|j| m.get(i, j).abs()).sum();
            let side = if rng.bool(0.75) { 1.0 } else { -1.0 };
            m.set(i, i, side * (rest + tenth(rng.range(0.3, 1.2))).max(0.3));
        }
    };
    match s.kind {
        Diagonal => {
            let mut m = Mat::zero(n);
            for i in 0..n {
                m.set(i, i, if s.seed == 0 { FIRST_DIAGONAL[i] } else { entry(&mut rng, -2.0, 2.0) });
            }
            m
        }
        Tridiagonal => Mat::from_fn(n, |i, j| match i.abs_diff(j) {
            0 => 2.0,
            1 => -1.0,
            _ => 0.0,
        }),
        Banded => {
            let mut m = random(&mut rng, &|i, j| i != j && i.abs_diff(j) <= p);
            dominant(&mut rng, &mut m);
            m
        }
        Triangular => random(&mut rng, &|i, j| j >= i),
        Nilpotent => random(&mut rng, &|i, j| j > i),
        Hessenberg => random(&mut rng, &|i, j| i <= j + 1),
        Blocks => random(&mut rng, &|i, j| (i < p) == (j < p)),
        Toeplitz => {
            let t: Vec<f64> = (0..2 * n - 1).map(|_| entry(&mut rng, -1.5, 1.5)).collect();
            Mat::from_fn(n, |i, j| t[j + n - 1 - i])
        }
        Circulant => {
            let c: Vec<f64> = if s.seed == 0 {
                (0..n)
                    .map(|j| match j {
                        0 => 0.5,
                        1 => 0.25,
                        _ if j == n - 1 => 0.25,
                        _ => 0.0,
                    })
                    .collect()
            } else {
                (0..n).map(|_| entry(&mut rng, -1.0, 1.0)).collect()
            };
            Mat::from_fn(n, |i, j| c[(j + n - i) % n])
        }
        LowRank => {
            let mut m = Mat::zero(n);
            for _ in 0..p.max(1) {
                let u: Vec<f64> = (0..n).map(|_| entry(&mut rng, -1.0, 1.0)).collect();
                let v: Vec<f64> = (0..n).map(|_| entry(&mut rng, -1.0, 1.0)).collect();
                for i in 0..n {
                    for j in 0..n {
                        m.set(i, j, m.get(i, j) + u[i] * v[j]);
                    }
                }
            }
            m
        }
        Markov => {
            // each column: shares to two decimals, the largest made up to a sum of 1
            let mut m = Mat::zero(n);
            for j in 0..n {
                let w: Vec<f64> = (0..n).map(|_| (rng.f32() as f64).powi(2) + 0.02).collect();
                let total: f64 = w.iter().sum();
                // (none below 0.01: every entry above 0, so the chain settles)
                let mut col: Vec<f64> = w.iter().map(|v| ((v / total * 100.0).round() / 100.0).max(0.01)).collect();
                let big = (0..n).max_by(|&a, &b| col[a].total_cmp(&col[b])).unwrap_or(0);
                col[big] = 0.0;
                let rest: f64 = col.iter().sum();
                col[big] = ((1.0 - rest) * 100.0).round() / 100.0;
                for i in 0..n {
                    m.set(i, j, col[i]);
                }
            }
            m
        }
        Laplacian => {
            let mut m = Mat::zero(n);
            for i in 0..n {
                for j in 0..n {
                    if i != j && s.edges[i][j] {
                        m.set(i, j, -1.0);
                        m.set(i, i, m.get(i, i) + 1.0);
                    }
                }
            }
            m
        }
        Dominant => {
            let mut m = random(&mut rng, &|i, j| i != j);
            for i in 0..n {
                for j in 0..n {
                    if i != j {
                        m.set(i, j, tenth(m.get(i, j) as f32 * 0.6));
                    }
                }
            }
            dominant(&mut rng, &mut m);
            m
        }
        Differences => Mat::from_fn(n, |i, j| {
            if i == j {
                1.0
            } else if i == j + 1 {
                -1.0
            } else {
                0.0
            }
        }),
        Sums => Mat::from_fn(n, |i, j| if j <= i { 1.0 } else { 0.0 }),
        Permutation => {
            let sigma = shuffle(n, s.seed);
            Mat::from_fn(n, |i, j| if sigma[i] == j { 1.0 } else { 0.0 })
        }
        Companion => {
            let c = s.roots.coefficients(n);
            let mut m = Mat::zero(n);
            for i in 1..n {
                m.set(i, i - 1, 1.0);
            }
            for i in 0..n {
                m.set(i, n - 1, -c.get(i).copied().unwrap_or(0.0));
            }
            m
        }
        Vandermonde => {
            let t = points(n, s.seed);
            Mat::from_fn(n, |i, j| t[i].powi(j as i32))
        }
        Hilbert => Mat::from_fn(n, |i, j| 1.0 / (i + j + 1) as f64),
    }
}

/// The points of the Vandermonde kind: evenly from 0 to 1 for seed 0, else n different
/// ones among 0, 0.05, ..., 1, in order.
fn points(n: usize, seed: u64) -> Vec<f64> {
    if seed == 0 {
        return (0..n).map(|i| i as f64 / (n - 1).max(1) as f64).collect();
    }
    let mut grid: Vec<usize> = (0..=20).collect();
    let mut rng = Rng::new(seed.wrapping_mul(31).wrapping_add(n as u64));
    rng.shuffle(&mut grid);
    let mut chosen: Vec<usize> = grid[..n].to_vec();
    chosen.sort_unstable();
    chosen.into_iter().map(|k| k as f64 * 0.05).collect()
}

/// The list x each kind starts with: one that shows what the kind does.
fn first_x(kind: Kind, n: usize) -> [f64; MAX] {
    let mut x = [0.0; MAX];
    let general = [1.0, -0.6, 0.4, 1.2, -1.0, 0.7, -0.3, 0.9];
    for (i, v) in x.iter_mut().enumerate().take(n) {
        *v = match kind {
            Diagonal | Hilbert => 1.0,
            // a tent: straight but at its top
            Tridiagonal => {
                let top = n.div_ceil(2) as f64;
                ((i + 1).min(n - i) as f64) * 1.6 / top
            }
            Circulant => {
                if i == 2.min(n - 1) {
                    1.5
                } else {
                    0.0
                }
            }
            // everything in the first state
            Markov => {
                if i == 0 {
                    1.0
                } else {
                    0.0
                }
            }
            // 1 on the first ring, 0 on the second
            Laplacian => {
                if i < n.div_ceil(2) {
                    1.0
                } else {
                    0.0
                }
            }
            // a rise and a fall
            Differences => [0.5, 1.0, 1.5, 1.2, 0.9, 0.6, 0.3, 0.0][i],
            Sums => [1.0, 0.5, -1.0, 0.0, 0.5, -0.5, 1.0, -0.5][i],
            // a ramp, to see where each entry goes
            Permutation => 0.25 * (i + 1) as f64,
            // the polynomial 1 - 2t + t²
            Vandermonde => [1.0, -2.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0][i],
            _ => general[i],
        };
    }
    x
}

// ----- what is worked out from the setup -----------------------------------------------

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Mode {
    /// x is dragged, and Ax is drawn.
    Multiply,
    /// b is dragged, and x = A⁻¹b is drawn.
    Solve,
}

/// The change made to b by the nudge: this share of its size.
const NUDGE: f64 = 1e-6;

/// A singular value this much smaller than the largest counts as 0.
const RANK_TOL: f64 = 1e-12;

/// Everything drawn, worked out from the setup and the bars: kept until one of them changes.
#[derive(Clone, Debug)]
struct Derived {
    m: Mat,
    /// The eigenvalues, in an order that keeps each near where it was before a change.
    eig: Vec<DVec2>,
    /// The singular values, largest first, and how many count as more than 0.
    sv: Vec<f64>,
    rank: usize,
    /// The two rows of bars (top: x; bottom: Ax or b), and in solve mode, after a nudge,
    /// the x there was before it.
    top: Vec<f64>,
    bottom: Vec<f64>,
    ghost: Option<Vec<f64>>,
    /// In solve mode: whether there is a solution to draw.
    solved: bool,
    /// The ranges of the bars, and of the plane (the middle of the real parts shown, and
    /// the half-width and half-height), and of the singular values (powers of ten).
    top_range: f64,
    bottom_range: f64,
    plane: [f64; 3],
    sv_range: [f64; 2],
    /// Which row the handles are on: 0 the top (x), 1 the bottom (b).
    handle_row: f64,
    /// What the pattern buys, in words, for this matrix (see `words`).
    words: [String; 3],
}

/// The round number at or above v: 1, 1.5, 2, 3, 4, 5, 6 or 8 times a power of ten
/// (steps fine enough that bars fill most of their chart).
fn nice(v: f64) -> f64 {
    let v = if v.is_finite() { v.max(1e-12) } else { 1.0 };
    let power = 10f64.powf(v.log10().floor());
    let lead = v / power;
    let step = [1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0].into_iter().find(|s| lead <= s + 1e-9).unwrap_or(10.0);
    step * power
}

/// The other list's order kept: `new` reordered so that each is as near as can be to
/// the one in its place in `old` (closest pairs first).
fn matched(old: &[DVec2], new: Vec<DVec2>) -> Vec<DVec2> {
    if old.len() != new.len() {
        let mut new = new;
        new.sort_by(|a, b| a.x.total_cmp(&b.x).then(b.y.total_cmp(&a.y)));
        return new;
    }
    let n = new.len();
    let mut pairs: Vec<(f64, usize, usize)> = Vec::with_capacity(n * n);
    for (i, o) in old.iter().enumerate() {
        for (j, z) in new.iter().enumerate() {
            pairs.push(((*o - *z).length_squared(), i, j));
        }
    }
    pairs.sort_by(|a, b| a.0.total_cmp(&b.0));
    let mut out = vec![DVec2::NAN; n];
    let (mut used_old, mut used_new) = ([false; MAX], [false; MAX]);
    for (_, i, j) in pairs {
        if i < MAX && j < MAX && !used_old[i] && !used_new[j] {
            out[i] = new[j];
            used_old[i] = true;
            used_new[j] = true;
        }
    }
    out
}

/// The 2n eigenvalue-like points to keep in sight in the plane besides the eigenvalues:
/// what the kind promises them (see `promise`), at the target matrix.
fn plane_points(s: &Setup, m: &Mat, discs: bool) -> Vec<DVec2> {
    let n = s.n;
    let mut pts = vec![DVec2::ZERO];
    match s.kind {
        Tridiagonal => pts.extend([DVec2::new(4.0, 0.0)]),
        Markov | Permutation => pts.extend([DVec2::new(-1.0, 0.0), DVec2::new(1.0, 0.0), DVec2::new(0.0, 1.0), DVec2::new(0.0, -1.0)]),
        Companion => pts.extend(s.roots.all(n)),
        Circulant => {
            let c: Vec<f64> = (0..n).map(|j| m.get(0, j)).collect();
            pts.extend((0..48).map(|k| symbol(&c, std::f64::consts::TAU * k as f64 / 48.0)));
        }
        _ => {}
    }
    if discs || s.kind == Dominant {
        for (c, r) in gershgorin(m) {
            pts.extend([c - DVec2::new(r, 0.0), c + DVec2::new(r, 0.0), c + DVec2::new(0.0, r), c - DVec2::new(0.0, r)]);
        }
    }
    pts
}

/// The first row's Fourier transform at the angle θ: Σ cⱼ e^{imθ}, where m is j for the
/// first half of the row and j − n for the second (the entries past the middle are the
/// ones just before the diagonal, round the ring). At θ = 2πk/n it is the circulant's
/// k-th eigenvalue, as ω^{(j−n)k} = ω^{jk}; between those angles it is the curve that
/// joins them most smoothly (for a symmetric row, a stretch of the real line).
fn symbol(c: &[f64], theta: f64) -> DVec2 {
    let n = c.len() as i64;
    c.iter().enumerate().fold(DVec2::ZERO, |acc, (j, v)| {
        let m = if 2 * j as i64 <= n { j as i64 } else { j as i64 - n };
        acc + DVec2::from_angle(m as f64 * theta) * *v
    })
}

/// The Gershgorin discs of a matrix: about each diagonal entry, with the sum of the
/// sizes of the rest of its row as radius.
fn gershgorin(m: &Mat) -> Vec<(DVec2, f64)> {
    (0..m.n).map(|i| (DVec2::new(m.get(i, i), 0.0), (0..m.n).filter(|&j| j != i).map(|j| m.get(i, j).abs()).sum())).collect()
}

/// The power of ten at or below v, and at or above it.
fn decades(lo: f64, hi: f64) -> [f64; 2] {
    let lo = lo.max(1e-300).log10().floor();
    let hi = hi.max(1e-300).log10().ceil().max(lo + 1.0);
    [lo, hi]
}

// ----- the scene ------------------------------------------------------------------------

/// What can be pressed on the picture.
#[derive(Clone, Copy, Debug, PartialEq)]
enum Hit {
    Kind(Kind),
    Tab(usize),
    Smaller,
    Larger,
    Another,
    Multiply,
    Solve,
    Nudge,
    Discs,
    Less,
    More,
}

/// Something that happens a moment after an experiment's button, so that the reader
/// sees the picture before and after it.
#[derive(Clone, Copy, Debug, PartialEq)]
enum Later {
    /// Cut the graph's edge between two points.
    Cut(usize, usize),
    /// Change b by one part in a million.
    Nudge,
}

/// How long before it (seconds of the scene's time).
const LATER: f32 = 1.0;

/// How long a change of n takes to fade out and in (seconds), and how long the old
/// promise in the plane takes to fade into the new one.
const FADE: f32 = 0.36;
const SWAP: f32 = 0.3;

pub struct Patterns {
    setup: Setup,
    mode: Mode,
    nudge: bool,
    discs: bool,
    /// The bars: x, dragged in multiply mode, and b, dragged in solve mode.
    x: [f64; MAX],
    b: [f64; MAX],
    /// The family whose kinds are shown, on a narrow picture.
    tab: usize,
    now: Derived,
    /// The matrix the last change started from (its numbers are written until the
    /// colours are half-way to the new ones).
    before: Mat,
    /// The numbers drawn, which glide to `now`'s.
    shown: Ease,
    /// After a change of n: the picture as it was, fading out, and how far the change has gone.
    leaving: Option<(Pic, f32)>,
    /// After any other change: the promise in the plane as it was, fading out.
    old_promise: Option<(Vec<Item>, f32)>,
    later: Option<(Later, f32)>,
    /// Whether a root of the companion's polynomial is held (the plane then keeps its range).
    holding_root: bool,
    /// Lines of words broken to a width, kept.
    wraps: std::collections::HashMap<(String, u32, u32), Vec<String>>,
}

impl Patterns {
    fn new() -> Patterns {
        let setup = Setup::new(Tridiagonal, 6);
        let x = first_x(Tridiagonal, 6);
        let mut scene = Patterns {
            setup,
            mode: Mode::Multiply,
            nudge: false,
            discs: false,
            x,
            b: [0.0; MAX],
            tab: Tridiagonal.family(),
            now: blank(6),
            before: Mat::zero(6),
            shown: Ease::default(),
            leaving: None,
            old_promise: None,
            later: None,
            holding_root: false,
            wraps: Default::default(),
        };
        scene.refresh();
        scene.shown.snap(&numbers(&scene.now));
        scene.before = scene.now.m;
        scene
    }

    /// Work everything out again from the setup and the bars.
    fn refresh(&mut self) {
        let s = &self.setup;
        let n = s.n;
        let m = build(s);
        let eig = matched(if self.now.eig.len() == n { &self.now.eig } else { &[] }, eigenvalues(&m).unwrap_or_else(|| vec![DVec2::ZERO; n]));
        let (sv, worst) = singular(&m);
        let top_sv = sv.first().copied().unwrap_or(0.0);
        let rank = sv.iter().filter(|v| **v > RANK_TOL * top_sv).count();
        let invertible = rank == n;
        let (mut top, mut ghost, mut solved) = (self.x[..n].to_vec(), None, true);
        let bottom = match self.mode {
            Mode::Multiply => m.apply(&self.x[..n]),
            Mode::Solve => self.b[..n].to_vec(),
        };
        match self.mode {
            Mode::Multiply => {}
            Mode::Solve => {
                let found = if invertible { solve(&m, &bottom) } else { None };
                match found {
                    Some(x) => {
                        top = x;
                        if self.nudge {
                            let size = crate::math::linalg::norm(&bottom);
                            let moved: Vec<f64> = bottom.iter().zip(&worst).map(|(b, w)| b + NUDGE * size * w).collect();
                            if let Some(x2) = solve(&m, &moved) {
                                ghost = Some(std::mem::replace(&mut top, x2));
                            }
                        }
                    }
                    None => {
                        solved = false;
                        top = vec![0.0; n];
                    }
                }
            }
        }
        let held = |v: &[f64]| v.iter().fold(0.0f64, |s, x| s.max(x.abs()));
        // (the row being dragged keeps its range while it is dragged: a value can then be
        // taken to the top of its chart and no further)
        let dragged_range = |v: &[f64]| nice(held(v).max(2.0));
        let made_range = |v: &[f64]| nice(held(v).max(0.05));
        let (top_range, bottom_range) = match self.mode {
            Mode::Multiply => (dragged_range(&top), made_range(&bottom)),
            Mode::Solve => (made_range(&top).max(ghost.as_deref().map_or(0.0, made_range)), dragged_range(&bottom)),
        };
        let plane = if self.holding_root && self.now.eig.len() == n {
            self.now.plane
        } else {
            let mut pts = plane_points(s, &m, self.discs);
            pts.extend(eig.iter().copied());
            let (lo, hi) = pts.iter().fold((f64::MAX, f64::MIN), |(lo, hi), z| (lo.min(z.x), hi.max(z.x)));
            let tall = pts.iter().fold(0.0f64, |t, z| t.max(z.y.abs()));
            let pad = 0.12 * (hi - lo).max(2.0 * tall).max(1.0);
            [0.5 * (lo + hi), (0.5 * (hi - lo) + pad).max(0.8), (tall + pad).max(0.8)]
        };
        let positive: Vec<f64> = sv.iter().copied().filter(|v| *v > RANK_TOL * top_sv).collect();
        let sv_range = decades(positive.last().copied().unwrap_or(1.0), top_sv.max(1e-300));
        let handle_row = if self.mode == Mode::Solve { 1.0 } else { 0.0 };
        self.now = Derived { m, eig, sv, rank, top, bottom, ghost, solved, top_range, bottom_range, plane, sv_range, handle_row, words: Default::default() };
        self.now.words = words(&self.setup, &self.now);
    }

    /// Take a new setup (and with it, maybe, a new x and b): what changes glides, or for
    /// a new n fades out and in.
    fn take(&mut self, setup: Setup, x: Option<[f64; MAX]>) {
        let new_n = setup.n != self.setup.n;
        let before = self.pic(1.0);
        if new_n {
            self.leaving = Some((before, 0.0));
        } else if setup != self.setup || self.leaving.is_none() {
            self.old_promise = Some((promise(&before), 0.0));
        }
        self.before = if new_n { build(&setup) } else { self.pic(1.0).drawn };
        self.setup = setup;
        if let Some(x) = x {
            self.x = x;
        }
        self.tab = setup.kind.family();
        if self.mode == Mode::Solve {
            // the b that the x shown would give
            let m = build(&self.setup);
            let ax = m.apply(&self.x[..setup.n]);
            self.b = [0.0; MAX];
            self.b[..setup.n].copy_from_slice(&ax);
        }
        self.refresh();
        if new_n {
            self.shown.snap(&numbers(&self.now));
        }
    }

    /// A kind chosen: its first member, at the same n, with the list that shows it.
    fn choose(&mut self, kind: Kind) {
        let mut s = Setup::new(kind, self.setup.n);
        if kind == self.setup.kind {
            s = Setup { seed: self.setup.seed, ..self.setup };
        }
        self.later = None;
        self.nudge = false;
        self.take(s, Some(first_x(kind, s.n)));
    }

    /// Another n: the same kind, its first member.
    fn resize(&mut self, n: usize) {
        let n = n.clamp(LEAST, MAX);
        if n == self.setup.n {
            return;
        }
        let kind = self.setup.kind;
        let mut s = Setup::new(kind, n);
        if let Some(p) = kind.param(n) {
            s.param = self.setup.param.max(p.lo).min(p.hi);
        }
        self.later = None;
        self.take(s, Some(first_x(kind, n)));
    }

    /// Another matrix of the kind.
    fn another(&mut self) {
        if self.setup.kind.set_by_n() {
            return;
        }
        let mut s = self.setup;
        s.seed = s.seed.wrapping_add(1);
        let n = s.n;
        let mut rng = Rng::new(s.seed.wrapping_mul(977).wrapping_add(n as u64));
        match s.kind {
            Laplacian => {
                // a random graph: each pair joined with chance 2/5
                s.edges = [[false; MAX]; MAX];
                for i in 0..n {
                    for j in i + 1..n {
                        if rng.bool(0.4) {
                            s.edges[i][j] = true;
                            s.edges[j][i] = true;
                        }
                    }
                }
            }
            Companion => {
                for z in s.roots.pairs.iter_mut() {
                    *z = DVec2::new(tenth(rng.range(-1.4, 1.4)), tenth(rng.range(0.2, 1.4)));
                }
                s.roots.real = tenth(rng.range(-1.4, 1.4));
            }
            _ => {}
        }
        self.later = None;
        self.take(s, None);
    }

    fn set_param(&mut self, v: usize) {
        let Some(p) = self.setup.kind.param(self.setup.n) else { return };
        let v = v.max(p.lo).min(p.hi);
        if v != self.setup.param {
            self.later = None;
            self.take(Setup { param: v, ..self.setup }, None);
        }
    }

    fn set_mode(&mut self, mode: Mode) {
        if mode == self.mode {
            return;
        }
        let n = self.setup.n;
        self.later = None;
        match mode {
            // b is what x gave
            Mode::Solve => {
                self.b = [0.0; MAX];
                self.b[..n].copy_from_slice(&self.now.bottom[..n]);
            }
            // x is the solution, where there was one
            Mode::Multiply => {
                self.nudge = false;
                if self.now.solved {
                    self.x = [0.0; MAX];
                    self.x[..n].copy_from_slice(&self.now.top[..n]);
                    let range = nice(self.x.iter().fold(0.0f64, |s, v| s.max(v.abs())).max(2.0));
                    for v in self.x.iter_mut() {
                        *v = v.max(-range).min(range);
                    }
                }
            }
        }
        self.mode = mode;
        self.refresh();
    }

    fn set_nudge(&mut self, on: bool) {
        if self.mode == Mode::Solve && on != self.nudge {
            self.nudge = on;
            self.refresh();
        }
    }

    fn set_discs(&mut self, on: bool) {
        if on != self.discs {
            self.old_promise = Some((promise(&self.pic(1.0)), 0.0));
            self.discs = on;
            self.refresh();
        }
    }

    /// Join two points of the graph, or cut the edge between them.
    fn toggle_edge(&mut self, i: usize, j: usize) {
        let n = self.setup.n;
        if self.setup.kind != Laplacian || i == j || i >= n || j >= n {
            return;
        }
        let mut s = self.setup;
        s.edges[i][j] = !s.edges[i][j];
        s.edges[j][i] = s.edges[i][j];
        self.take(s, None);
    }

    /// The things to try.
    fn set_up(&mut self, k: usize) {
        self.nudge = false;
        self.later = None;
        match k {
            // differences undo sums
            0 => {
                self.mode = Mode::Solve;
                self.take(Setup::new(Differences, 6), Some(first_x(Differences, 6)));
                self.b = [0.0; MAX];
                self.b[..6].copy_from_slice(&TRIAL_B);
                self.refresh();
            }
            // a circulant blur
            1 => {
                self.mode = Mode::Multiply;
                self.take(Setup::new(Circulant, 8), Some(first_x(Circulant, 8)));
            }
            // cut an edge
            2 => {
                self.mode = Mode::Multiply;
                self.take(Setup::new(Laplacian, 6), Some(first_x(Laplacian, 6)));
                let (i, j) = bridge(6);
                self.later = Some((Later::Cut(i, j), LATER));
            }
            // one part in a million
            3 => {
                self.mode = Mode::Solve;
                self.take(Setup::new(Hilbert, 6), Some(first_x(Hilbert, 6)));
                self.later = Some((Later::Nudge, LATER));
            }
            // drag a root
            _ => {
                self.mode = Mode::Multiply;
                self.take(Setup::new(Companion, 5), Some(first_x(Companion, 5)));
            }
        }
        self.discs = false;
        self.refresh();
    }

    /// Move everything on: the moment after an experiment by the scene's time, the
    /// glides and fades by the clock's (they answer the reader's hand).
    fn advance(&mut self, dt: f32, real_dt: f32) {
        if let Some((what, left)) = self.later {
            let left = left - dt.max(0.0);
            if left <= 0.0 || !left.is_finite() {
                self.later = None;
                match what {
                    Later::Cut(i, j) => {
                        if self.setup.edges[i.min(MAX - 1)][j.min(MAX - 1)] {
                            self.toggle_edge(i, j);
                        }
                    }
                    Later::Nudge => self.set_nudge(true),
                }
            } else {
                self.later = Some((what, left));
            }
        }
        let dt = real_dt.max(0.0);
        if let Some((_, t)) = &mut self.leaving {
            *t += dt / FADE;
            if *t >= 1.0 || !t.is_finite() {
                self.leaving = None;
            }
        }
        if let Some((_, t)) = &mut self.old_promise {
            *t += dt / SWAP;
            if *t >= 1.0 || !t.is_finite() {
                self.old_promise = None;
            }
        }
        // (while the new n fades in, its numbers are already where they go)
        if self.leaving.is_none() {
            self.shown.follow(&numbers(&self.now), dt);
        }
    }

    fn moving(&self) -> bool {
        self.later.is_some() || self.leaving.is_some() || self.old_promise.is_some() || self.shown.shown != numbers(&self.now)
    }

    /// What is drawn now, at the strength `alpha`.
    fn pic(&self, alpha: f32) -> Pic {
        Pic::of(&self.setup, self.mode, self.nudge, self.discs, &self.now, &self.before, &self.shown.shown, alpha)
    }
}

/// A derived state with nothing in it (before the first `refresh`).
fn blank(n: usize) -> Derived {
    Derived {
        m: Mat::zero(n),
        eig: Vec::new(),
        sv: vec![0.0; n],
        rank: 0,
        top: vec![0.0; n],
        bottom: vec![0.0; n],
        ghost: None,
        solved: true,
        top_range: 1.0,
        bottom_range: 1.0,
        plane: [0.0, 1.0, 1.0],
        sv_range: [0.0, 1.0],
        handle_row: 0.0,
        words: Default::default(),
    }
}

/// The b of the first thing to try.
const TRIAL_B: [f64; 6] = [1.0, 0.5, -1.0, 0.0, 1.5, -0.5];

// ----- numbers that glide ---------------------------------------------------------------

/// Numbers drawn on their way to new ones, through two short eases one after the other
/// (as `plot::Glide`), exactly there in the end; a part of them can be put in place at
/// once (what the reader's hand is moving).
#[derive(Clone, Debug, Default)]
struct Ease {
    between: Vec<f32>,
    shown: Vec<f32>,
}

impl Ease {
    const EASE: f32 = 0.07;
    const THERE: f32 = 1e-4;

    fn follow(&mut self, to: &[f32], dt: f32) -> bool {
        if self.shown.len() != to.len() {
            self.snap(to);
            return false;
        }
        if self.shown == to {
            return false;
        }
        let k = 1.0 - (-dt.max(0.0) / Self::EASE).exp();
        let (mut far, mut moved) = (false, false);
        for ((b, s), t) in self.between.iter_mut().zip(self.shown.iter_mut()).zip(to) {
            let (b0, s0) = (*b, *s);
            *b += (t - *b) * k;
            *s += (*b - *s) * k;
            moved |= *b != b0 || *s != s0;
            let near = Self::THERE * t.abs().max(1.0);
            far |= !((t - *s).abs() <= near && (t - *b).abs() <= near);
        }
        if !far || !moved {
            self.snap(to);
            return false;
        }
        true
    }

    fn snap(&mut self, to: &[f32]) {
        self.shown = to.to_vec();
        self.between = to.to_vec();
    }

    /// Put the numbers from `at` on in place at once.
    fn put(&mut self, at: usize, values: &[f32]) {
        for (k, v) in values.iter().enumerate() {
            if let (Some(s), Some(b)) = (self.shown.get_mut(at + k), self.between.get_mut(at + k)) {
                (*s, *b) = (*v, *v);
            }
        }
    }
}

/// Where each part of the numbers is, for an n: the matrix (n²), the eigenvalues (2n),
/// the singular values as powers of ten (n), the bars (top, bottom, ghost: n each), and
/// eight more (the bars' ranges, the plane's three, the singular values' two, and which
/// row the handles are on).
struct Slots {
    n: usize,
}

impl Slots {
    fn eig(&self) -> usize {
        self.n * self.n
    }
    fn sv(&self) -> usize {
        self.n * self.n + 2 * self.n
    }
    fn top(&self) -> usize {
        self.n * self.n + 3 * self.n
    }
    fn bottom(&self) -> usize {
        self.n * self.n + 4 * self.n
    }
    fn ghost(&self) -> usize {
        self.n * self.n + 5 * self.n
    }
    fn ranges(&self) -> usize {
        self.n * self.n + 6 * self.n
    }
    fn len(&self) -> usize {
        self.ranges() + 8
    }
}

/// A singular value of 0 is kept this far below the singular values' range (powers of
/// ten), which puts it at the left end of the strip, where 0 is written.
const SV_ZERO: f64 = 2.0;

/// The numbers of a derived state, in the order of `Slots`.
fn numbers(d: &Derived) -> Vec<f32> {
    let n = d.m.n;
    let mut out = Vec::with_capacity(Slots { n }.len());
    out.extend(d.m.dense().iter().map(|v| *v as f32));
    for k in 0..n {
        let z = d.eig.get(k).copied().unwrap_or(DVec2::ZERO);
        out.extend([z.x as f32, z.y as f32]);
    }
    let top = d.sv.first().copied().unwrap_or(0.0);
    out.extend((0..n).map(|k| {
        let v = d.sv.get(k).copied().unwrap_or(0.0);
        if v > RANK_TOL * top {
            v.log10() as f32
        } else {
            (d.sv_range[0] - SV_ZERO) as f32
        }
    }));
    // (the bars as shares of their charts' heights: what glides is what is seen)
    let row = |v: &[f64], range: f64| (0..n).map(|k| (v.get(k).copied().unwrap_or(0.0) / range).max(-1.0).min(1.0) as f32).collect::<Vec<f32>>();
    out.extend(row(&d.top, d.top_range));
    out.extend(row(&d.bottom, d.bottom_range));
    out.extend(row(d.ghost.as_deref().unwrap_or(&d.top), d.top_range));
    out.extend([d.top_range, d.bottom_range, d.plane[0], d.plane[1], d.plane[2], d.sv_range[0], d.sv_range[1], d.handle_row].map(|v| v as f32));
    // (nothing that is not a number is ever drawn from)
    for v in out.iter_mut() {
        if !v.is_finite() {
            *v = 0.0;
        }
    }
    out
}

// ----- what is drawn --------------------------------------------------------------------

/// The picture at one moment: the setup and the numbers as drawn.
#[derive(Clone, Debug)]
struct Pic {
    setup: Setup,
    mode: Mode,
    nudge: bool,
    discs: bool,
    /// The matrix as drawn (its colours), and as written (its numbers: the new ones from
    /// half-way through a glide).
    drawn: Mat,
    written: Mat,
    eig: Vec<DVec2>,
    /// The singular values as powers of ten, as drawn, and as they are.
    sv_log: Vec<f64>,
    sv: Vec<f64>,
    rank: usize,
    /// The bars, each as a share of its chart's half-height (−1 to 1), and the ranges
    /// the charts stand for.
    top: Vec<f64>,
    bottom: Vec<f64>,
    ghost: Option<Vec<f64>>,
    solved: bool,
    top_range: f64,
    bottom_range: f64,
    plane: [f64; 3],
    sv_range: [f64; 2],
    /// Which row the handles are on (between the two while they move to the other).
    handle_row: f32,
    words: [String; 3],
    alpha: f32,
}

impl Pic {
    /// `before` is the matrix the last change started from: its numbers are written
    /// until the colours are half-way to the new ones.
    #[allow(clippy::too_many_arguments)]
    fn of(s: &Setup, mode: Mode, nudge: bool, discs: bool, d: &Derived, before: &Mat, shown: &[f32], alpha: f32) -> Pic {
        let n = s.n;
        let slots = Slots { n };
        let fresh = numbers(d);
        let shown = if shown.len() == slots.len() { shown } else { &fresh };
        let at = |k: usize| shown.get(k).copied().unwrap_or(0.0) as f64;
        let drawn = Mat::from_fn(n, |i, j| at(i * n + j));
        let apart = |a: &Mat, b: &Mat| (0..n).flat_map(|i| (0..n).map(move |j| (i, j))).map(|(i, j)| (a.get(i, j) - b.get(i, j)).powi(2)).sum::<f64>().sqrt();
        let written = if before.n == n && apart(&drawn, &d.m) > 0.5 * apart(before, &d.m) { *before } else { d.m };
        Pic {
            setup: *s,
            mode,
            nudge,
            discs,
            drawn,
            written,
            eig: (0..n).map(|k| DVec2::new(at(slots.eig() + 2 * k), at(slots.eig() + 2 * k + 1))).collect(),
            sv_log: (0..n).map(|k| at(slots.sv() + k)).collect(),
            sv: d.sv.clone(),
            rank: d.rank,
            top: (0..n).map(|k| at(slots.top() + k)).collect(),
            bottom: (0..n).map(|k| at(slots.bottom() + k)).collect(),
            ghost: d.ghost.as_ref().map(|_| (0..n).map(|k| at(slots.ghost() + k)).collect()),
            solved: d.solved,
            top_range: d.top_range,
            bottom_range: d.bottom_range,
            plane: [at(slots.ranges() + 2), at(slots.ranges() + 3).max(0.1), at(slots.ranges() + 4).max(0.1)],
            sv_range: [at(slots.ranges() + 5), at(slots.ranges() + 6).max(at(slots.ranges() + 5) + 1.0)],
            handle_row: (at(slots.ranges() + 7) as f32).max(0.0).min(1.0),
            words: d.words.clone(),
            alpha,
        }
    }
}

/// Something drawn in the plane to show where the kind's eigenvalues must lie.
#[derive(Clone, Debug)]
enum Item {
    /// A ring of this many pixels round a point, and how many eigenvalues it stands for.
    Ring(DVec2, f32, Color, usize),
    /// A circle about a point (plane units), dashed or not.
    Circle(DVec2, f64, Color, bool),
    /// A curve through these points.
    Curve(Vec<DVec2>, Color),
    /// A stretch of the real line, from and to (to may be past the edge).
    Band(f64, f64, Color),
    /// A shaded disc with its edge.
    Disc(DVec2, f64, Color),
}

/// Colours of the cycles of a shuffle, the pieces of a graph and the blocks.
const SET_COLORS: [Color; 8] = [pal::PINK, pal::CYAN, pal::PURPLE, pal::YELLOW, pal::BLUE, pal::GREEN, pal::LINK, pal::MUTED];
const PROMISE: Color = pal::GREEN;
const EIGEN: Color = pal::ORANGE;
const DISCS: Color = pal::BLUE;
const SINGULAR: Color = pal::CYAN;
const DRAGGED: Color = pal::WHITE;
const MADE: Color = pal::YELLOW;
const CHANGE: Color = pal::RED;

/// What the kind promises the eigenvalues, for a picture.
fn promise(p: &Pic) -> Vec<Item> {
    let s = &p.setup;
    let n = s.n;
    let m = &p.drawn;
    let ring = |z: DVec2, count: usize| Item::Ring(z, 8.0, PROMISE, count);
    let mut out = Vec::new();
    match s.kind {
        Diagonal | Triangular => out.extend(gather((0..n).map(|i| DVec2::new(m.get(i, i), 0.0)).collect()).into_iter().map(|(z, k)| ring(z, k))),
        Tridiagonal => {
            out.push(Item::Band(0.0, 4.0, PROMISE));
            out.extend((1..=n).map(|k| ring(DVec2::new(2.0 - 2.0 * (k as f64 * std::f64::consts::PI / (n + 1) as f64).cos(), 0.0), 1)));
        }
        Nilpotent => out.push(ring(DVec2::ZERO, n)),
        Differences | Sums => out.push(ring(DVec2::new(1.0, 0.0), n)),
        Blocks => {
            let k = s.param.max(1).min(n - 1);
            for (b, (from, to)) in [(0, k), (k, n)].into_iter().enumerate() {
                let block = Mat::from_fn(to - from, |i, j| m.get(from + i, from + j));
                for z in eigenvalues(&block).unwrap_or_default() {
                    out.push(Item::Ring(z, 8.0 + 3.0 * b as f32, SET_COLORS[1 + b], 1));
                }
            }
        }
        Circulant => {
            let c: Vec<f64> = (0..n).map(|j| m.get(0, j)).collect();
            out.push(Item::Curve((0..=120).map(|k| symbol(&c, std::f64::consts::TAU * k as f64 / 120.0)).collect(), PROMISE));
            out.extend(gather((0..n).map(|k| symbol(&c, std::f64::consts::TAU * k as f64 / n as f64)).collect()).into_iter().map(|(z, k)| ring(z, k)));
        }
        LowRank => out.push(ring(DVec2::ZERO, n.saturating_sub(s.param))),
        Markov => {
            out.push(Item::Circle(DVec2::ZERO, 1.0, PROMISE, false));
            out.push(ring(DVec2::new(1.0, 0.0), 1));
            if let Some(second) = second_size(&p.eig) {
                out.push(Item::Circle(DVec2::ZERO, second, PROMISE, true));
            }
        }
        Laplacian => {
            let (_, count) = pieces(&s.edges, n);
            out.push(Item::Band(0.0, f64::INFINITY, PROMISE));
            out.push(ring(DVec2::ZERO, count));
        }
        Permutation => {
            out.push(Item::Circle(DVec2::ZERO, 1.0, PROMISE.a(0.5), false));
            let sigma = shuffle(n, s.seed);
            for (c, cycle) in cycles(&sigma, n).iter().enumerate() {
                let len = cycle.len();
                for k in 0..len {
                    let z = DVec2::from_angle(std::f64::consts::TAU * k as f64 / len as f64);
                    out.push(Item::Ring(z, 7.0 + 3.5 * c as f32, SET_COLORS[c % SET_COLORS.len()], 1));
                }
            }
        }
        Hilbert => out.push(Item::Band(0.0, f64::INFINITY, PROMISE)),
        Banded | Hessenberg | Toeplitz | Vandermonde | Companion | Dominant => {}
    }
    if p.discs || s.kind == Dominant {
        for (c, r) in gershgorin(m) {
            out.push(Item::Disc(c, r, DISCS));
        }
    }
    out
}

/// Points that fall together, gathered with how many there are of each.
fn gather(points: Vec<DVec2>) -> Vec<(DVec2, usize)> {
    let mut out: Vec<(DVec2, usize)> = Vec::new();
    for z in points {
        match out.iter_mut().find(|(w, _)| (*w - z).length() < 1e-6) {
            Some((_, k)) => *k += 1,
            None => out.push((z, 1)),
        }
    }
    out
}

/// The second largest size among eigenvalues (the largest being 1, for a Markov matrix).
fn second_size(eig: &[DVec2]) -> Option<f64> {
    let mut sizes: Vec<f64> = eig.iter().map(|z| z.length()).collect();
    sizes.sort_by(|a, b| b.total_cmp(a));
    sizes.get(1).copied()
}

// ----- words ----------------------------------------------------------------------------

/// What the kind does to x, what it costs, and what it promises the eigenvalues: three
/// sentences, with this matrix's numbers in them.
fn words(s: &Setup, d: &Derived) -> [String; 3] {
    let n = s.n;
    let sym = s.kind.symbol();
    let full = full_work(n);
    let p = s.param;
    match s.kind {
        Diagonal => [
            r"$(Ax)_i = d_i x_i$: each entry of $x$ is scaled by its own number, and no entry touches another.".into(),
            format!("{n} numbers to store, not $n^2 = {}$; $Ax = b$ is solved by {n} divisions, $x_i = b_i / d_i$, if no $d_i$ is 0.", n * n),
            "The eigenvalues are the diagonal entries (green), each with an axis for its eigenvector.".into(),
        ],
        Tridiagonal => [
            r"$(Kx)_i = 2x_i - x_{i-1} - x_{i+1}$ (with $x_0 = x_{n+1} = 0$): the second difference, 0 where $x$ runs straight and above 0 where it bends down.".into(),
            format!("$3n - 2 = {}$ numbers in its band (or none: they are 2 and $-1$); elimination stays in the band: {} multiplications and divisions, about $5n$, against {full} for a full matrix.", 3 * n - 2, work(n, 1, 1)),
            r"Symmetric and positive definite: its eigenvalues are $2 - 2\cos(k\pi/(n+1))$ for $k = 1, \dots, n$ (green), all between 0 and 4.".into(),
        ],
        Banded => [
            format!("$(Ax)_i$ uses only the entries of $x$ within $w = {p}$ {} of $i$.", if p == 1 { "place" } else { "places" }),
            format!(
                "{} numbers in the band; elimination fills nothing outside it (no row exchanges are needed: each diagonal entry outweighs the rest of its row): {} multiplications and divisions, at most $n(w^2 + 3w + 1)$, against {full}.",
                band_count(n, p),
                work(n, p, p)
            ),
            "A band promises nothing about the eigenvalues (the Gershgorin discs hold them for any matrix).".into(),
        ],
        Triangular => [
            r"$(Ax)_i$ uses only $x_i, \dots, x_n$: the entries at $i$ and after it.".into(),
            format!("$n(n+1)/2 = {}$ numbers; $Ax = b$ is solved from the bottom up ($x_n = b_n / a_{{nn}}$ first): {} multiplications and divisions, against {full}.", n * (n + 1) / 2, work(n, 0, n - 1)),
            r"The eigenvalues are the diagonal entries (green): $\det(A - \lambda I)$ is the product of the $a_{ii} - \lambda$.".into(),
        ],
        Nilpotent => [
            r"$(Nx)_i$ uses only $x_{i+1}, \dots, x_n$: each time $N$ is applied the last entry that is not 0 moves up a place at least, so $N^n x = 0$ for every $x$.".into(),
            format!(r"$n(n-1)/2 = {}$ numbers. No inverse: its last row is 0, so $(Nx)_n = 0$ whatever $x$ is.", n * (n - 1) / 2),
            r"Every eigenvalue is 0 ($\det(N - \lambda I) = (-\lambda)^n$), and $N^n = 0$: nilpotent.".into(),
        ],
        Hessenberg => [
            r"Zeros below the first diagonal under the main one: $(Ax)_i$ uses $x_{i-1}, \dots, x_n$.".into(),
            format!(
                "{} numbers; elimination has one row to clear in each column: $n^2 + 2n - 2 = {}$ multiplications and divisions, against {full}. A QR step keeps this shape and is cheap on it, so eigenvalues are computed from it.",
                n * (n + 1) / 2 + n - 1,
                work(n, 1, n - 1)
            ),
            "No promise: every square matrix has the eigenvalues of one of this shape.".into(),
        ],
        Blocks => [
            format!(
                "Two problems side by side: the first {p} {} of $x$ {} through the first block, the other {} through the second, and the two never mix.",
                if p == 1 { "entry" } else { "entries" },
                if p == 1 { "goes" } else { "go" },
                n - p
            ),
            format!("${p}^2 + {}^2 = {}$ numbers; $Ax = b$ is two smaller problems: {} multiplications and divisions, against {full}.", n - p, p * p + (n - p) * (n - p), full_work(p) + full_work(n - p)),
            "Its eigenvalues are those of the two blocks together (rings in the blocks' colours).".into(),
        ],
        Toeplitz => [
            r"Constant along each diagonal: $(Ax)_i = \sum_j t_{j-i}\, x_j$, the same weights slid along, cut off at the ends.".into(),
            format!(r"$2n - 1 = {}$ numbers. Elimination does not use the repeats ({full} multiplications and divisions); methods made for Toeplitz matrices (Levinson's) need work proportional to $n^2$, when every top-left block is invertible.", 2 * n - 1),
            "No simple promise for its eigenvalues (unlike a circulant's).".into(),
        ],
        Circulant => [
            r"Each row is the row above moved one place along, wrapping round: $Ax$ is a moving weighted sum round a ring of $n$.".into(),
            format!(r"{n} numbers, the first row; solved through the Fourier transform in work proportional to $n \log n$ (if no eigenvalue is 0). Any two $n \times n$ circulants commute."),
            r"Its eigenvalues are the first row's Fourier transform, $\lambda_k = \sum_j c_j \omega^{jk}$ with $\omega = e^{2\pi i/n}$: on the green curve, which the first row draws.".into(),
        ],
        LowRank => [
            if p == 1 { r"$A = uv^\top$: $Ax = (v \cdot x)\,u$, always a multiple of $u$.".to_string() } else { format!(r"$A = u_1 v_1^\top + \dots + u_{p} v_{p}^\top$: every $Ax$ is a combination of $u_1, \dots, u_{p}$.") },
            format!(r"$2rn = {}$ numbers, and $Ax$ in {} multiplications rather than $n^2 = {}$. No inverse: $Ax = b$ has a solution only for $b$ in the span of the $u$s.", 2 * p * n, 2 * p * n, n * n),
            format!(
                "At least $n - r = {}$ {} 0, and {} of the singular values {} above 0: the rank.",
                n - p,
                if n - p == 1 { "eigenvalue is" } else { "eigenvalues are" },
                d.rank,
                if d.rank == 1 { "is" } else { "are" }
            ),
        ],
        Markov => [
            r"Columns of shares adding to 1: $x$ is how much is in each of $n$ states, and $Mx$ where it is a step later. $Mx$ adds up to what $x$ adds up to.".into(),
            match second_size(&d.eig) {
                Some(l2) => format!(r"$n(n-1) = {}$ numbers are free (each column's last share is what the others leave). Every entry here is above 0, so step after step $M^k x$ settles at the steady state (the eigenvector of 1, scaled to what $x$ adds up to); the rest shrinks roughly like $|\lambda_2|^k = {}^k$.", n * (n - 1), num(l2, 2)),
                None => String::new(),
            },
            r"An eigenvalue 1, and none larger in size: all lie on or inside the unit circle (green); the dashed circle has the second largest size, $|\lambda_2|$.".into(),
        ],
        Laplacian => {
            let (_, count) = pieces(&s.edges, n);
            let edges = (0..n).flat_map(|i| (i + 1..n).map(move |j| (i, j))).filter(|(i, j)| s.edges[*i][*j]).count();
            let zeros = d.eig.iter().filter(|z| z.length() < 1e-9).count();
            [
                r"$(Lx)_i$ is the sum of $x_i - x_j$ over the points $j$ joined to $i$: how far $x_i$ stands above its neighbours. Rows add to 0, so $L$ times all ones is 0.".into(),
                format!("Set by the graph's {edges} {}: $-1$ for each, and on the diagonal how many each point has. Click a $-1$ to cut that edge, an empty cell to join two points.", if edges == 1 { "edge" } else { "edges" }),
                format!(
                    r"Real and never negative ($x^\top L x$ is the sum of $(x_i - x_j)^2$ over the edges), and as many zero eigenvalues as the graph has pieces: here {count} {} and {zeros} {}.",
                    if count == 1 { "piece" } else { "pieces" },
                    if zeros == 1 { "zero" } else { "zeros" }
                ),
            ]
        }
        Dominant => [
            r"Each diagonal entry outweighs the rest of its row: $|a_{ii}| > \sum_{j \ne i} |a_{ij}|$.".into(),
            format!("Always invertible, and elimination never meets a pivot of 0, so no row exchanges are needed ({full} multiplications and divisions); simple repeated methods (Jacobi's) converge for it."),
            "Every eigenvalue lies in a Gershgorin disc (blue): about $a_{ii}$, with the rest of row $i$'s sizes added up as radius. No disc reaches 0.".into(),
        ],
        Differences => [
            r"$(Dx)_i = x_i - x_{i-1}$: each entry less the one before it.".into(),
            r"Set by $n$. Solving $Dx = b$ takes running sums, $x_i = b_1 + \dots + b_i$: $n - 1$ additions and no multiplication. So $D^{-1} = S$, the lower triangle of ones.".into(),
            "Every eigenvalue is 1 (the diagonal), yet there is only one line of eigenvectors, along the last axis.".into(),
        ],
        Sums => [
            r"$(Sx)_i = x_1 + \dots + x_i$: the running sum.".into(),
            r"Set by $n$. Solving $Sx = b$ takes differences, $x_i = b_i - b_{i-1}$: $n - 1$ subtractions. So $S^{-1} = D$.".into(),
            "Every eigenvalue is 1 (the diagonal), with one line of eigenvectors, along the last axis.".into(),
        ],
        Permutation => {
            let cs = cycles(&shuffle(n, s.seed), n);
            [
                format!(r"One 1 in each row and column: $Px$ is $x$ shuffled, $(Px)_i = x_{{\sigma(i)}}$. The cycles of the shuffle are ${}$, in colours.", cycle_text(&cs)),
                r"$n$ numbers (where the ones are); $P^{-1} = P^\top$, so $Px = b$ is solved by shuffling back, with no arithmetic. Orthogonal: every singular value is 1.".into(),
                r"A cycle of length $\ell$ gives the $\ell$th roots of 1, evenly round the unit circle, in its colour.".into(),
            ]
        }
        Companion => [
            r"$Cx$ moves each entry of $x$ down a place, and adds $x_n$ times the last column.".into(),
            r"$n$ numbers, the last column: $-c_0, \dots, -c_{n-1}$ for $p(\lambda) = \lambda^n + c_{n-1}\lambda^{n-1} + \dots + c_0$. $Cx = b$ takes about $2n$ steps (the first row gives $x_n$, each other row one more $x_i$), if 0 is not a root.".into(),
            r"Its eigenvalues are the roots of $p$, since $\det(\lambda I - C) = p(\lambda)$. Drag the white roots: the orange dots follow.".into(),
        ],
        Vandermonde => {
            let t = points(n, s.seed);
            let det: f64 = (0..n).flat_map(|i| (i + 1..n).map(move |j| (i, j))).map(|(i, j)| t[j] - t[i]).product();
            [
                r"Row $i$ is $1, t_i, t_i^2, \dots$ at the point $t_i$: with $x$ a polynomial's coefficients, $Vx$ is its values at the $n$ points.".into(),
                format!(r"$n$ numbers, the points; solving $Vx = b$ finds the polynomial through $n$ points (about $n^2$ work by Newton's divided differences). $\det V = \prod_{{i<j}} (t_j - t_i) = {}$: 0 only if two points meet.", short(det)),
                if s.seed == 0 {
                    format!(r"No promise for the eigenvalues. Its condition number, $\kappa = {}$, grows fast with $n$: for evenly spaced points, about 7 times for each point added.", kappa_text(&d.sv))
                } else {
                    format!(r"No promise for the eigenvalues. Its condition number is $\kappa = {}$; it grows fast with $n$.", kappa_text(&d.sv))
                },
            ]
        }
        Hilbert => [
            r"$h_{ij} = 1/(i + j - 1)$: symmetric and positive definite.".into(),
            format!(r"Set by $n$. $\kappa = {}$: a change in $b$ can grow up to $\kappa$ times in $x$, each as a share of its own size.", kappa_text(&d.sv)),
            format!(
                "Its eigenvalues are real and positive (green), from {} down to ${}$: the small ones make it nearly singular.",
                short(d.eig.iter().fold(0.0f64, |m, z| m.max(z.x))),
                short(d.eig.iter().fold(f64::MAX, |m, z| m.min(z.x)))
            ),
        ],
    }
    .map(|line| line.replace("$A", &format!("${sym}")).replace("(A", &format!("({sym}")))
}

/// The entries within w of the diagonal.
fn band_count(n: usize, w: usize) -> usize {
    (0..n).map(|i| (0..n).filter(|&j| i.abs_diff(j) <= w).count()).sum()
}

/// The condition number as markup: σ₁/σₙ, or ∞ for no inverse.
fn kappa_text(sv: &[f64]) -> String {
    let top = sv.first().copied().unwrap_or(0.0);
    let last = sv.last().copied().unwrap_or(0.0);
    if last.is_nan() || last <= RANK_TOL * top {
        r"\infty".into()
    } else {
        short(top / last)
    }
}

/// A number as markup: whole numbers as they are; two decimals, or three for a small
/// one, without trailing zeros; then two figures and a power of ten.
fn short(v: f64) -> String {
    if !v.is_finite() {
        return "?".into();
    }
    let a = v.abs();
    if a == 0.0 {
        return "0".into();
    }
    if (v - v.round()).abs() < 1e-9 && a < 1e4 {
        return num(v.round(), 0);
    }
    if (0.1..1e4).contains(&a) {
        return trim(num(v, if a >= 100.0 { 0 } else { 2 }));
    }
    if (0.001..0.1).contains(&a) {
        return trim(num(v, 3));
    }
    let mut e = a.log10().floor() as i32;
    let mut m = (a / 10f64.powi(e) * 10.0).round() / 10.0;
    if m >= 10.0 {
        m /= 10.0;
        e += 1;
    }
    let sign = if v < 0.0 { "-" } else { "" };
    let lead = trim(format!("{m:.1}"));
    format!(r"{sign}{lead} \times 10^{{{e}}}")
}

/// A decimal without the zeros at its end.
fn trim(s: String) -> String {
    if s.contains('.') {
        s.trim_end_matches('0').trim_end_matches('.').to_string()
    } else {
        s
    }
}

/// A cell's number, as markup.
fn cell_text(kind: Kind, v: f64, i: usize, j: usize) -> String {
    if kind == Hilbert {
        return if i + j == 0 { "$1$".into() } else { format!(r"$\dfrac{{1}}{{{}}}$", i + j + 1) };
    }
    format!("${}$", short(v))
}

// ----- where things go ------------------------------------------------------------------

/// A button over the picture: where, what it says, what it does, whether it is lit.
struct Button {
    rect: Rect,
    text: String,
    hit: Hit,
    on: bool,
}

struct Layout {
    narrow: bool,
    text: f32,
    small: f32,
    buttons: Vec<Button>,
    /// Words written among the buttons (the families' names, n, the width).
    labels: Vec<(Vec2, String, Align2)>,
    heading: Vec2,
    /// The matrix's top-left corner and the side of the whole grid.
    grid: Option<(Vec2, f32)>,
    /// The colour scale under it.
    scale: Option<Rect>,
    top_bars: Option<Rect>,
    bottom_bars: Option<Rect>,
    plane: Option<Rect>,
    sv: Option<Rect>,
    words: Option<(Vec2, f32, f32)>,
}

/// The room above a row of bars for its title, and below the lower one for its numbers.
const BARS_ABOVE: f32 = 20.0;
const BARS_BELOW: f32 = 16.0;
/// The room above the plane for its title and key, and the singular values' strip.
const PLANE_ABOVE: f32 = 38.0;
const SV_ABOVE: f32 = 20.0;
const SV_H: f32 = 18.0;
const SV_BELOW: f32 = 16.0;

/// Buttons in rows from (x0, y), as wide as their words, wrapping at x1. Returns the
/// rectangles and where the next row would start.
#[allow(clippy::too_many_arguments)]
fn flow(f: &Frame, words: &[String], x0: f32, x1: f32, y: f32, h: f32, gap: f32, size: f32) -> (Vec<Rect>, f32) {
    let (mut x, mut y) = (x0, y);
    let mut out = Vec::with_capacity(words.len());
    for w in words {
        let width = f.text_size(w, size).x + 16.0;
        if x > x0 && x + width > x1 {
            x = x0;
            y += h + gap;
        }
        out.push((Vec2::new(x, y), Vec2::new(x + width, y + h)));
        x += width + gap;
    }
    (out, if words.is_empty() { y } else { y + h + gap })
}

impl Layout {
    fn of(f: &Frame, view: Vec2, inset: f32, sc: &Patterns) -> Layout {
        let view = view.max(Vec2::ONE);
        let narrow = view.x < 760.0;
        let (text, small) = if narrow { (11.5, 10.5) } else { (12.5, 11.5) };
        let side = if narrow { 10.0 } else { 16.0 };
        let top = inset.max(0.0) + if narrow { 6.0 } else { 12.0 };
        let (h, gap) = if narrow { (25.0, 4.0) } else { (24.0, 5.0) };
        let mut lay = Layout {
            narrow,
            text,
            small,
            buttons: Vec::new(),
            labels: Vec::new(),
            heading: Vec2::new(side, top),
            grid: None,
            scale: None,
            top_bars: None,
            bottom_bars: None,
            plane: None,
            sv: None,
            words: None,
        };
        if view.x < 240.0 || view.y < 180.0 {
            return lay;
        }
        let s = &sc.setup;
        let right = view.x - side;
        // the kinds: four labelled rows, or (narrow) a row of families and the chosen one's kinds
        let mut y = top;
        let kind_button = |k: Kind| Button { rect: (Vec2::ZERO, Vec2::ZERO), text: k.name().to_string(), hit: Hit::Kind(k), on: k == s.kind };
        if narrow {
            let tabs = plot::even_row(FAMILIES.len(), side, right, y, h, gap);
            for (i, r) in tabs.into_iter().enumerate() {
                lay.buttons.push(Button { rect: r, text: FAMILIES[i].short.to_string(), hit: Hit::Tab(i), on: i == sc.tab });
            }
            y += h + gap;
            let kinds = FAMILIES[sc.tab.min(FAMILIES.len() - 1)].kinds;
            let names: Vec<String> = kinds.iter().map(|k| k.name().to_string()).collect();
            let (rects, next) = flow(f, &names, side, right, y, h, gap, text);
            for (k, r) in kinds.iter().zip(rects) {
                lay.buttons.push(Button { rect: r, ..kind_button(*k) });
            }
            y = next;
        } else {
            let lead = FAMILIES.iter().map(|fam| f.text_size(fam.name, small).x).fold(0.0f32, f32::max) + 14.0;
            for fam in &FAMILIES {
                lay.labels.push((Vec2::new(side, y + h * 0.5), fam.name.to_string(), Align2::LEFT_CENTER));
                let names: Vec<String> = fam.kinds.iter().map(|k| k.name().to_string()).collect();
                let (rects, next) = flow(f, &names, side + lead, right, y, h, gap, text);
                for (k, r) in fam.kinds.iter().zip(rects) {
                    lay.buttons.push(Button { rect: r, ..kind_button(*k) });
                }
                y = next;
            }
        }
        // the controls: n, another, the mode, the nudge, the discs, the kind's own number
        y += if narrow { 2.0 } else { 4.0 };
        let mut controls: Vec<(Hit, String, bool)> = vec![(Hit::Smaller, "−".into(), false), (Hit::Larger, "+".into(), false)];
        if !s.kind.set_by_n() {
            controls.push((Hit::Another, "another".into(), false));
        }
        controls.push((Hit::Multiply, r"$x \to Ax$".replace('A', s.kind.symbol()), sc.mode == Mode::Multiply));
        controls.push((Hit::Solve, r"solve $Ax = b$".replace('A', s.kind.symbol()), sc.mode == Mode::Solve));
        if sc.mode == Mode::Solve {
            controls.push((Hit::Nudge, "nudge $b$".into(), sc.nudge));
        }
        if s.kind != Dominant {
            controls.push((Hit::Discs, if narrow { "discs".into() } else { "Gershgorin discs".into() }, sc.discs));
        }
        if s.kind.param(s.n).is_some() {
            controls.push((Hit::Less, "−".into(), false));
            controls.push((Hit::More, "+".into(), false));
        }
        // (n is written between its two buttons, and the kind's number between its own)
        let n_text = format!("$n = {}$", s.n);
        let param_text = s.kind.param(s.n).map(|p| format!("{} {}", p.name, s.param.max(p.lo).min(p.hi)));
        let mut x = side;
        let mut row_y = y;
        let gap_x = if narrow { 4.0 } else { 6.0 };
        for (hit, word, on) in controls {
            let mut width = f.text_size(&word, text).x + 16.0;
            if matches!(hit, Hit::Smaller | Hit::Larger | Hit::Less | Hit::More) {
                width = h;
            }
            let label = match hit {
                Hit::Larger => Some(n_text.clone()),
                Hit::More => param_text.clone(),
                _ => None,
            };
            let label_w = label.as_ref().map_or(0.0, |t| f.text_size(t, text).x + 10.0);
            if x > side && x + width + label_w > right {
                x = side;
                row_y += h + gap;
            }
            if let Some(t) = label {
                lay.labels.push((Vec2::new(x + label_w * 0.5 - 1.0, row_y + h * 0.5), t, Align2::CENTER_CENTER));
                x += label_w;
            }
            lay.buttons.push(Button { rect: (Vec2::new(x, row_y), Vec2::new(x + width, row_y + h)), text: word, hit, on });
            x += width + gap_x;
            if matches!(hit, Hit::Smaller | Hit::Less) {
                x -= gap_x - 2.0;
            }
        }
        let body = row_y + h + if narrow { 10.0 } else { 18.0 };
        let head_h = if narrow { 20.0 } else { 26.0 };
        let scale_h = if narrow { 30.0 } else { 36.0 };
        lay.heading = Vec2::new(side, body);
        if narrow {
            // the matrix and the plane side by side, the bars, the strip and the words under them
            let bars_h = if view.y - body > 520.0 { 50.0 } else { 40.0 };
            let below = 2.0 * (BARS_ABOVE + bars_h) + BARS_BELOW + SV_ABOVE + SV_H + SV_BELOW + 70.0;
            let g = ((view.x - 2.0 * side) * 0.55).min(view.y - body - head_h - scale_h - below).min(360.0).floor();
            if g < 90.0 {
                let g = (view.y - body - head_h - 8.0).min(view.x - 2.0 * side).floor();
                if g >= 60.0 {
                    lay.grid = Some((Vec2::new(side, body + head_h), g));
                }
                return lay;
            }
            let at = Vec2::new(side, body + head_h);
            lay.grid = Some((at, g));
            lay.scale = Some((Vec2::new(side + 2.0, at.y + g + 16.0), Vec2::new(side + g - 2.0, at.y + g + 22.0)));
            let px = side + g + 40.0;
            // (its title under the matrix's heading, beside the matrix)
            if right - px >= 80.0 {
                lay.plane = Some((Vec2::new(px, at.y + 22.0), Vec2::new(right, at.y + g)));
            }
            let mut y = at.y + g + scale_h + BARS_ABOVE;
            lay.top_bars = Some((Vec2::new(side, y), Vec2::new(right, y + bars_h)));
            y += bars_h + BARS_ABOVE;
            lay.bottom_bars = Some((Vec2::new(side, y), Vec2::new(right, y + bars_h)));
            y += bars_h + BARS_BELOW + SV_ABOVE;
            lay.sv = Some((Vec2::new(side + 24.0, y), Vec2::new(right - 24.0, y + SV_H)));
            y += SV_H + SV_BELOW + 4.0;
            let room = view.y - y - 6.0;
            lay.words = Some((Vec2::new(side, y), right - side, room));
            return lay;
        }
        // wide: the matrix and the bars on the left, the plane, the strip and the words on the right
        let bars_h = if view.y - body > 760.0 { 84.0 } else { 64.0 };
        let left_below = scale_h + 2.0 * (BARS_ABOVE + bars_h) + BARS_BELOW + 18.0;
        let g = (view.x * 0.44).min(view.y - body - head_h - left_below - 8.0).min(520.0).floor();
        if g < 120.0 {
            return lay;
        }
        let at = Vec2::new(side, body + head_h);
        lay.grid = Some((at, g));
        lay.scale = Some((Vec2::new(side + 2.0, at.y + g + 18.0), Vec2::new(side + g - 2.0, at.y + g + 25.0)));
        let mut by = at.y + g + scale_h + BARS_ABOVE + 10.0;
        lay.top_bars = Some((Vec2::new(side, by), Vec2::new(side + g, by + bars_h)));
        by += bars_h + BARS_ABOVE + 6.0;
        lay.bottom_bars = Some((Vec2::new(side, by), Vec2::new(side + g, by + bars_h)));
        let x = side + g + 64.0;
        if right - x < 200.0 {
            return lay;
        }
        let plane_top = body + PLANE_ABOVE;
        let plane_h = ((view.y - plane_top) * 0.46).min(right - x).max(120.0).floor();
        lay.plane = Some((Vec2::new(x, plane_top), Vec2::new(right, plane_top + plane_h)));
        let sv_y = plane_top + plane_h + 26.0 + SV_ABOVE;
        lay.sv = Some((Vec2::new(x, sv_y), Vec2::new(right - 20.0, sv_y + SV_H)));
        let words_y = sv_y + SV_H + SV_BELOW + 10.0;
        let room = view.y - words_y - 10.0;
        lay.words = Some((Vec2::new(x - 30.0, words_y), right - x + 30.0, room));
        lay
    }
}

// ----- kinds of line, each its own batch --------------------------------------------------

/// Kinds of line with a batch each (see `plot::apart`): the outlines of cells, the
/// promise and the old promise fading out, the discs, the graph, the bars' marks, the
/// guides along tied diagonals, the nudge's dashed lines, and the marks along the
/// plane's axes and the singular values' strip (whose number never changes).
const CELLS: u8 = plot::OWN;
const PROMISED: u8 = plot::OWN + 1;
const PROMISED_OLD: u8 = plot::OWN + 2;
const DISC_LINES: u8 = plot::OWN + 3;
const GRAPH: u8 = plot::OWN + 4;
const BAR_MARKS: u8 = plot::OWN + 5;
const TIES: u8 = plot::OWN + 6;
const GHOST: u8 = plot::OWN + 7;
const PLANE_MARKS: u8 = plot::OWN + 8;
const SV_MARKS: u8 = plot::OWN + 9;

/// The powers of ten the singular values' strip has marks for.
const DECADES: (i32, i32) = (-16, 6);

/// Every kind of line, in the order its batch is made in every frame.
fn in_order(f: &mut Frame) {
    for kind in [plot::AXES, plot::CURVES, plot::MARKS, plot::BUTTONS, CELLS, PROMISED, PROMISED_OLD, DISC_LINES, GRAPH, BAR_MARKS, TIES, PLANE_MARKS, SV_MARKS] {
        plot::apart(f, kind, false);
    }
    plot::apart(f, GHOST, true);
}

impl Scene for Patterns {
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
        Some("buttons over the picture: click    white dots: drag")
    }

    fn ui(&mut self, ui: &mut egui::Ui, _: &Gfx) {
        ui::note(ui, r"Many of the matrices that matter are large, and are known by their pattern: where their zeros are, how their entries repeat, what their rows or columns add up to. A pattern is worth knowing because it pays, in the numbers it takes to store the matrix, in the work it takes to solve $Ax = b$, and in where its eigenvalues can lie. Choose a kind with the buttons over the picture, and drag the bars of $x$: the matrix acts on a list of $n$ numbers.");

        ui::section(ui, "Things to try");
        for (k, t) in TRIALS.iter().enumerate() {
            if ui::experiment(ui, t.label, t.look_for) {
                self.set_up(k);
            }
        }

        ui::section(ui, "Kind");
        for (i, fam) in FAMILIES.iter().enumerate() {
            ui::note(ui, fam.name);
            let names: Vec<&str> = fam.kinds.iter().map(|k| k.name()).collect();
            let mut chosen = fam.kinds.iter().position(|k| *k == self.setup.kind).unwrap_or(usize::MAX);
            if ui::choice(ui, &names, &mut chosen) {
                if let Some(k) = fam.kinds.get(chosen) {
                    self.choose(*k);
                    self.tab = i;
                }
            }
        }
        for line in self.now.words.clone().iter().filter(|l| !l.is_empty()) {
            ui::note(ui, line);
        }
        if !self.setup.kind.set_by_n() && ui.button(ui::label_text(ui, "another of this kind", 12.5, None, false)).clicked() {
            self.another();
        }

        ui::section(ui, "Size and settings");
        let mut n = self.setup.n;
        if ui::slider_int(ui, "$n$", &mut n, LEAST..=MAX) {
            self.resize(n);
        }
        if let Some(p) = self.setup.kind.param(self.setup.n) {
            let mut v = self.setup.param;
            if ui::slider_int(ui, p.name, &mut v, p.lo..=p.hi) {
                self.set_param(v);
            }
        }
        let mut mode = if self.mode == Mode::Multiply { 0 } else { 1 };
        if ui::choice(ui, &[r"$x \to Ax$", r"solve $Ax = b$"], &mut mode) {
            self.set_mode(if mode == 0 { Mode::Multiply } else { Mode::Solve });
        }
        ui::note(ui, r"$x \to Ax$: drag the bars of $x$ and see $Ax$. Solve: drag the bars of $b$ and see the $x$ with $Ax = b$, if there is one.");
        if self.mode == Mode::Solve {
            let mut nudge = self.nudge;
            if ui::checkbox(ui, &mut nudge, "nudge $b$").changed() {
                self.set_nudge(nudge);
            }
            ui::note(ui, r"Changes $b$ by one part in a million of its size, in the direction that moves $x$ the most (the left singular vector of the smallest singular value). The change in $x$ is drawn in red.");
        }
        let mut discs = self.discs;
        if ui::checkbox(ui, &mut discs, "the Gershgorin discs").changed() {
            self.set_discs(discs);
        }
        ui::note(ui, "A disc about each diagonal entry, as wide as the rest of its row's sizes added up: every eigenvalue of every matrix lies in one of them.");

        ui::section(ui, "Reading the picture");
        ui::key(
            ui,
            &[
                (TEAL, "the cells: the entries, teal above 0 and rose below, brighter the larger beside the largest; dark for 0"),
                (pal::FAINT, "an empty cell: one the pattern makes 0; an inner frame: an entry the pattern sets; a plain cell: a free one"),
                (DRAGGED, "white: what you drag (the bars of $x$, or of $b$ when solving; the roots of $p$)"),
                (MADE, "yellow: what the matrix makes of it ($Ax$, or the $x$ that solves $Ax = b$)"),
                (EIGEN, "orange dots: the eigenvalues, in the complex plane"),
                (PROMISE, "green: where the kind promises its eigenvalues lie"),
                (DISCS, "blue: the Gershgorin discs"),
                (SINGULAR, "cyan: the singular values, on a scale of powers of ten"),
                (CHANGE, r"red: the change in $x$ after a nudge, or what is lost (a singular value of 0)"),
            ],
        );

        ui::section(ui, "Words");
        ui::term(ui, "Pattern:", "which entries are 0, which repeat, what the rows or columns add up to: a property every matrix of the kind shares.");
        ui::term(ui, &format!("{}:", capital(self.setup.kind.name())), kind_meaning(self.setup.kind));
        ui::term(ui, "Eigenvalue:", r"a number $\lambda$ with $Ax = \lambda x$ for some $x \ne 0$; for a real matrix it may be complex, and complex ones come in pairs $a \pm bi$.");
        ui::term(ui, "Singular values:", r"$\sigma_1 \ge \dots \ge \sigma_n \ge 0$: how much $A$ stretches each of $n$ perpendicular directions chosen for it, the most stretched first ('SVD, step by step'). As many are above 0 as the rank.");
        ui::term(ui, "Condition number:", r"$\kappa = \sigma_1 / \sigma_n$: the most that a change in $b$ can be magnified in $x$, each measured as a share of its own size; $\infty$ when there is no inverse.");
        ui::term(ui, "Work:", &format!("the multiplications and divisions that solving $Ax = b$ by elimination takes: $n^3/3 + n^2 - n/3$ for a full matrix ({} at $n = {}$), far fewer where the pattern's zeros let it skip whole rows.", full_work(self.setup.n), self.setup.n));
        ui::term(ui, "Gershgorin disc:", r"the disc about $a_{ii}$ of radius $\sum_{j \ne i} |a_{ij}|$: every eigenvalue lies in one of them.");

        ui::section(ui, "Numbers");
        let d = &self.now;
        let sym = self.setup.kind.symbol();
        ui::readout(ui, "$n$", format!("${}$", self.setup.n));
        ui::readout(ui, &format!(r"$\det {sym}$"), format!("${}$", short(d.m.det())));
        ui::readout(ui, &format!(r"$\tr {sym}$"), format!("${}$", short(d.m.trace())));
        ui::readout(ui, "rank", format!("${}$", d.rank));
        ui::readout_colored(ui, r"$\kappa = \sigma_1 / \sigma_n$", format!("${}$", kappa_text(&d.sv)), SINGULAR);
        ui::readout_colored(ui, r"largest $|\lambda|$", format!("${}$", short(d.eig.iter().fold(0.0f64, |m, z| m.max(z.length())))), EIGEN);
        if self.mode == Mode::Solve {
            if let Some(ghost) = &d.ghost {
                let (dx, x0) = (crate::math::linalg::norm(&d.top.iter().zip(ghost).map(|(a, b)| a - b).collect::<Vec<_>>()), crate::math::linalg::norm(ghost));
                ui::readout_colored(ui, r"$|\Delta x| / |x|$ after the nudge", format!("${}$", short(dx / x0.max(1e-300))), CHANGE);
            }
        }
        ui::see_also(ui, &[("special", "the zoo of $2 \\times 2$ matrices"), ("schur", "Hessenberg form and the QR algorithm"), ("dft", "circulants and the Fourier basis"), ("condition", "condition numbers")]);
    }

    fn frame(&mut self, f: &mut Frame) {
        let view = f.view_size();
        // (a canvas squeezed to a sliver is one the controls have opened over, on a phone)
        let covered = view.x < 160.0;
        let lay = Layout::of(f, view, f.top_inset, self);
        if !covered {
            f.handles.capture();
            f.handles.claim_clicks();
            self.press(f, &lay);
        }
        self.advance(f.dt, f.real_dt);
        if covered {
            return;
        }
        in_order(f);
        let leaving = self.leaving.as_ref().map(|(p, t)| (p.clone(), *t));
        match leaving {
            Some((old, t)) => {
                // the old picture fades out as the new one fades in, where it stands
                let t = smoothstep(t);
                self.draw(f, &lay, &Pic { alpha: 1.0 - t, ..old }, false);
                let now = self.pic(t);
                self.draw(f, &lay, &now, false);
            }
            None => {
                let now = self.pic(1.0);
                self.draw(f, &lay, &now, true);
            }
        }
    }
}

/// A word with its first letter a capital.
fn capital(s: &str) -> String {
    let mut c = s.chars();
    match c.next() {
        Some(first) => first.to_uppercase().collect::<String>() + c.as_str(),
        None => String::new(),
    }
}

/// What the chosen kind is, for the panel's words.
fn kind_meaning(kind: Kind) -> &'static str {
    match kind {
        Diagonal => "zeros everywhere off the main diagonal.",
        Tridiagonal => "zeros but on the main diagonal and the two beside it; here the one with 2 and $-1$ there, the stiffness matrix of a chain of springs ('Coupled springs: normal modes').",
        Banded => r"zeros farther than $w$ places from the main diagonal ($w$, the width; 1 is tridiagonal).",
        Triangular => "upper triangular: zeros below the main diagonal.",
        Nilpotent => r"some power of it is 0; a strictly upper triangular matrix (zeros on and below the diagonal) is, with $N^n = 0$.",
        Hessenberg => "upper Hessenberg: zeros below the first diagonal under the main one, the shape the QR algorithm works on ('Schur form and the QR algorithm').",
        Blocks => "square blocks down the diagonal and zeros everywhere else.",
        Toeplitz => r"constant along each diagonal: $a_{ij}$ depends only on $j - i$.",
        Circulant => "each row is the one above it moved one place to the right, the last entry wrapping round to the front; its eigenvectors are the waves of 'The Fourier basis'.",
        LowRank => r"of rank $r$ less than $n$: a sum of $r$ outer products $u v^\top$ (column times row), as in 'Low-rank approximation'.",
        Markov => "no negative entries, and each column adds up to 1: the chances of going from one state to each other in a step ('Powers of a matrix').",
        Laplacian => "of a graph: on the diagonal, how many edges meet at each point; $-1$ for each edge; 0 elsewhere.",
        Dominant => r"diagonally dominant: $|a_{ii}| > \sum_{j \ne i} |a_{ij}|$ in every row.",
        Differences => r"1 on the diagonal and $-1$ just below it: $(Dx)_i = x_i - x_{i-1}$.",
        Sums => r"1 on and below the diagonal: $(Sx)_i = x_1 + \dots + x_i$.",
        Permutation => "the identity with its rows shuffled: one 1 in each row and each column, 0 elsewhere.",
        Companion => r"ones just below the diagonal and a polynomial's coefficients, signs changed, in the last column: its eigenvalues are the polynomial's roots.",
        Vandermonde => r"row $i$ is $1, t_i, t_i^2, \dots, t_i^{n-1}$ for a point $t_i$.",
        Hilbert => r"$h_{ij} = 1/(i + j - 1)$: the classic matrix that is hard to solve accurately ('Sensitive equations').",
    }
}

// ----- the experiments --------------------------------------------------------------------

struct Trial {
    label: &'static str,
    look_for: &'static str,
}

const TRIALS: [Trial; 5] = [
    Trial {
        label: "differences undo sums",
        look_for: r"Solving $Dx = b$ with the difference matrix $D$, from the top down: $x_1 = b_1$, $x_2 = b_1 + b_2$, and so on: the yellow $x$ is the running sums of the white $b$. So $D^{-1} = S$, the lower triangle of ones, and $DS = I$: row $i$ of $DS$ is row $i$ of $S$ less row $i - 1$, which leaves a single 1, on the diagonal. Choose 'running sums' to see solving with $S$ take differences.",
    },
    Trial {
        label: "a circulant blur",
        look_for: r"Each row is the row above moved one place along, wrapping round: $Ax$ keeps half of each entry and gives a quarter to each neighbour, round a ring. The spike of 1.5 comes out as 0.375, 0.75, 0.375. The eigenvalues, $0.5 + 0.5\cos(2\pi k/8)$, are the first row's Fourier transform, on the green curve (here the stretch from 0 to 1). One is 0, for $(1, -1, 1, -1, \dots)$, which the blur averages away: this blur has no inverse ($\kappa = \infty$).",
    },
    Trial {
        label: "cut an edge of the graph",
        look_for: r"Two triangles of points joined by one edge, from point 3 to point 4: one piece, one eigenvalue 0 ($L$ times all ones is 0). A moment later the edge is cut: the two entries $-1$ go, two diagonal entries drop from 3 to 2, and the eigenvalue 0.44 slides to 0. Two pieces, two zero eigenvalues. The $x$ that is 1 on one piece and 0 on the other now has $Lx = 0$; before, the edge gave it $1$ and $-1$ at its ends. Click an empty cell to join two points.",
    },
    Trial {
        label: "the Hilbert matrix: one part in a million",
        look_for: r"$x = (1, 1, 1, 1, 1, 1)$ solves $Hx = b$ (the dashed lines). A moment later $b$ is changed by one part in a million, in the direction that does the most harm: too little to see. The answer becomes about $(1.04, -0.15, 8.79, -19.2, 23.3, -7.8)$: it moved (red) by 13 times its own size, $1.3 \times 10^7$ times the change in $b$, near the most it can be, $\kappa = 1.5 \times 10^7$.",
    },
    Trial {
        label: "drag a root of the polynomial",
        look_for: r"The white dots in the plane are roots of $p(\lambda) = (\lambda - r_1) \cdots (\lambda - r_5)$; drag one (its mirror image below the real line follows, so that $p$ has real coefficients). Those coefficients, signs changed, are the last column of $C$, and the eigenvalues of $C$, the orange dots, stay on the roots wherever they go: $\det(\lambda I - C) = p(\lambda)$. This is one way roots of polynomials are computed in practice.",
    },
];

// ----- drawing -----------------------------------------------------------------------------

/// The colours of the cells: a dark one for 0, teal for entries above 0 and rose for
/// those below.
const BASE: Color = Color::hex(0x161C2D);
const TEAL: Color = Color::hex(0x3AAE97);
const ROSE: Color = Color::hex(0xD0628F);

/// An entry's colour, beside the largest entry `scale`.
fn heat(v: f64, scale: f64) -> Color {
    if !(v.is_finite() && scale > 0.0) || v == 0.0 {
        return BASE;
    }
    let t = (v.abs() / scale).sqrt().min(1.0) as f32;
    BASE.mix(if v > 0.0 { TEAL } else { ROSE }, 0.18 + 0.82 * t)
}

/// "step" words for a number: 1 → "", 2 → "²".
fn power(k: usize) -> String {
    if k == 1 {
        String::new()
    } else {
        format!("^{{{k}}}")
    }
}

impl Patterns {
    /// The buttons over the picture: pressed before anything else is drawn (they take
    /// their clicks).
    fn press(&mut self, f: &mut Frame, lay: &Layout) {
        let mut hit = None;
        for b in &lay.buttons {
            if plot::button(f, b.rect, &b.text, lay.text, b.on) {
                hit = Some(b.hit);
            }
        }
        for (at, text, align) in &lay.labels {
            f.text_px(*at, text.as_str(), lay.small, pal::DIM, *align);
        }
        match hit {
            Some(Hit::Kind(k)) => self.choose(k),
            Some(Hit::Tab(i)) => self.tab = i.min(FAMILIES.len() - 1),
            Some(Hit::Smaller) => self.resize(self.setup.n.saturating_sub(1)),
            Some(Hit::Larger) => self.resize(self.setup.n + 1),
            Some(Hit::Another) => self.another(),
            Some(Hit::Multiply) => self.set_mode(Mode::Multiply),
            Some(Hit::Solve) => self.set_mode(Mode::Solve),
            Some(Hit::Nudge) => self.set_nudge(!self.nudge),
            Some(Hit::Discs) => self.set_discs(!self.discs),
            Some(Hit::Less) => self.set_param(self.setup.param.saturating_sub(1)),
            Some(Hit::More) => self.set_param(self.setup.param + 1),
            None => {}
        }
    }

    /// The whole picture at its strength; `live` for the one the reader can touch.
    fn draw(&mut self, f: &mut Frame, lay: &Layout, p: &Pic, live: bool) {
        self.grid(f, lay, p, live);
        self.bars(f, lay, p, live);
        self.plane(f, lay, p, live);
        self.strip(f, lay, p);
        self.say(f, lay, p);
    }

    /// The matrix as a grid of cells.
    fn grid(&mut self, f: &mut Frame, lay: &Layout, p: &Pic, live: bool) {
        let Some((at, g)) = lay.grid else { return };
        let s = &p.setup;
        let n = s.n;
        let alpha = p.alpha;
        let c = g / n as f32;
        f.text_px(lay.heading, s.kind.heading(n, s.param), if lay.narrow { 12.5 } else { 14.0 }, pal::TEXT.a(alpha), Align2::LEFT_TOP);
        let scale = p.drawn.largest().max(1e-12);
        let size = (c * 0.27).max(8.5).min(14.0);
        let sigma = shuffle(n, s.seed);
        let cs = cycles(&sigma, n);
        let cycle_of = |i: usize| cs.iter().position(|c| c.contains(&i)).unwrap_or(0);
        let gap = if c > 14.0 { 1.5 } else { 0.5 };
        for i in 0..n {
            for j in 0..n {
                let min = at + Vec2::new(j as f32, i as f32) * c;
                let (lo, hi) = (min + Vec2::splat(gap), min + Vec2::splat(c - gap));
                let mark = s.kind.mark(s, i, j);
                let v = p.drawn.get(i, j);
                if mark == Mark::Zero {
                    // an empty cell: the pattern's 0
                    let e = [lo, Vec2::new(hi.x, lo.y), hi, Vec2::new(lo.x, hi.y), lo];
                    plot::path_of(f, CELLS, &e, 1.0, pal::FAINT.a(0.55 * alpha));
                    continue;
                }
                let color = if s.kind == Permutation && v > 0.5 { SET_COLORS[cycle_of(i) % SET_COLORS.len()].mix(BASE, 0.25) } else { heat(v, scale) };
                plot::fill(f, (lo, hi), color.a(alpha));
                if mark == Mark::Set && c >= 14.0 {
                    let inset = (c * 0.08).max(2.5);
                    let (a, b) = (lo + Vec2::splat(inset), hi - Vec2::splat(inset));
                    plot::path_of(f, CELLS, &[a, Vec2::new(b.x, a.y), b, Vec2::new(a.x, b.y), a], 1.0, pal::DIM.a(0.5 * alpha));
                }
                if c >= 30.0 {
                    // (a free entry that happens to be 0 is written faintly: the eye goes to the others)
                    let w = p.written.get(i, j);
                    let ink = if w == 0.0 { pal::DIM.a(0.7 * alpha) } else { pal::TEXT.a(alpha) };
                    f.text_px((lo + hi) * 0.5, cell_text(s.kind, w, i, j), size, ink, Align2::CENTER_CENTER);
                }
            }
        }
        // what the pattern ties: the diagonals of a Toeplitz or circulant matrix, the blocks
        match s.kind {
            Toeplitz | Circulant => {
                for d in 1..n {
                    let (a, b) = (at + Vec2::new(d as f32 + 0.5, 0.5) * c, at + Vec2::new(n as f32 - 0.5, (n - d) as f32 - 0.5) * c);
                    plot::line_of(f, TIES, a, b, 1.0, pal::DIM.a(0.35 * alpha));
                    let (a, b) = (at + Vec2::new(0.5, d as f32 + 0.5) * c, at + Vec2::new((n - d) as f32 - 0.5, n as f32 - 0.5) * c);
                    plot::line_of(f, TIES, a, b, 1.0, pal::DIM.a(0.35 * alpha));
                }
            }
            Blocks => {
                let k = s.param.min(n);
                for (b, (from, to)) in [(0, k), (k, n)].into_iter().enumerate() {
                    let (lo, hi) = (at + Vec2::splat(from as f32 * c), at + Vec2::splat(to as f32 * c));
                    plot::path_of(f, TIES, &[lo, Vec2::new(hi.x, lo.y), hi, Vec2::new(lo.x, hi.y), lo], 2.0, SET_COLORS[1 + b].a(0.9 * alpha));
                }
            }
            _ => {}
        }
        // the sums the pattern fixes: a Markov matrix's columns, a Laplacian's rows
        let tiny = lay.small - 1.0;
        if s.kind == Markov && c >= 22.0 {
            for j in 0..n {
                let sum: f64 = (0..n).map(|i| p.written.get(i, j)).sum();
                f.text_px(at + Vec2::new((j as f32 + 0.5) * c, g + 3.0), format!("${}$", short(sum)), tiny, PROMISE.a(alpha), Align2::CENTER_TOP);
            }
        }
        if s.kind == Laplacian && c >= 22.0 {
            for i in 0..n {
                let sum: f64 = (0..n).map(|j| p.written.get(i, j)).sum();
                f.text_px(at + Vec2::new(g + 5.0, (i as f32 + 0.5) * c), format!("${}$", short(sum)), tiny, PROMISE.a(alpha), Align2::LEFT_CENTER);
            }
        }
        // a click on a cell of the Laplacian cuts or joins
        if live && s.kind == Laplacian {
            if let Some(ptr) = plot::pointer(f) {
                let cell = (ptr - at) / c;
                if cell.x >= 0.0 && cell.y >= 0.0 && cell.x < n as f32 && cell.y < n as f32 {
                    let (i, j) = (cell.y as usize, cell.x as usize);
                    if i != j {
                        let min = at + Vec2::new(j as f32, i as f32) * c;
                        hud_outline(f, min + Vec2::splat(1.0), min + Vec2::splat(c - 1.0), 1.6, pal::WHITE.a(0.8));
                        let mirror = at + Vec2::new(i as f32, j as f32) * c;
                        hud_outline(f, mirror + Vec2::splat(1.0), mirror + Vec2::splat(c - 1.0), 1.0, pal::WHITE.a(0.4));
                        if f.ptr.clicked {
                            f.ptr.clicked = false;
                            self.toggle_edge(i, j);
                        }
                    }
                }
            }
        }
        // the colour scale
        if let Some((min, max)) = lay.scale {
            let strips = 30;
            for k in 0..strips {
                let (a, b) = (k as f32 / strips as f32, (k + 1) as f32 / strips as f32);
                let v = (((a + b) as f64 - 1.0) * scale).max(-scale).min(scale);
                let x = |t: f32| min.x + (max.x - min.x) * t;
                plot::fill(f, (Vec2::new(x(a), min.y), Vec2::new(x(b) + 0.5, max.y)), heat(v, scale).a(alpha));
            }
            let tick = lay.small - 1.5;
            f.text_px(Vec2::new(min.x, max.y + 2.0), format!("$-{}$", short(scale)), tick, pal::DIM.a(alpha), Align2::LEFT_TOP);
            f.text_px(Vec2::new((min.x + max.x) * 0.5, max.y + 2.0), "$0$", tick, pal::DIM.a(alpha), Align2::CENTER_TOP);
            f.text_px(Vec2::new(max.x, max.y + 2.0), format!("${}$", short(scale)), tick, pal::DIM.a(alpha), Align2::RIGHT_TOP);
            // (not over a Markov matrix's column sums)
            if !lay.narrow && s.kind != Markov {
                f.text_px(Vec2::new(min.x, min.y - 3.0), "the colours: each entry's sign and size beside the largest", tick, pal::DIM.a(alpha), Align2::LEFT_BOTTOM);
            }
        }
    }

    /// The two rows of bars: x and Ax, or x and b.
    fn bars(&mut self, f: &mut Frame, lay: &Layout, p: &Pic, live: bool) {
        let (Some(top), Some(bottom)) = (lay.top_bars, lay.bottom_bars) else { return };
        let s = &p.setup;
        let n = s.n;
        let sym = s.kind.symbol();
        let alpha = p.alpha;
        let cs = cycles(&shuffle(n, s.seed), n);
        let cycle_of = |i: usize| cs.iter().position(|c| c.contains(&i)).unwrap_or(0);
        let solve = p.mode == Mode::Solve;
        let (top_name, bottom_name) = if solve {
            (
                if !p.solved {
                    format!("$x$: none, ${sym}$ has no inverse")
                } else if p.nudge && p.ghost.is_some() {
                    format!(r"$x = {sym}^{{-1}}(b + \Delta b)$; dashed: ${sym}^{{-1}} b$; red: the change")
                } else {
                    format!("$x = {sym}^{{-1}} b$, the solution")
                },
                "$b$, the right-hand side: drag the white dots".to_string(),
            )
        } else {
            ("$x$, the list going in: drag the white dots".to_string(), format!("${sym}x$, what comes out"))
        };
        let rows = [(top, &p.top, p.top_range, top_name, !solve), (bottom, &p.bottom, p.bottom_range, bottom_name, solve)];
        let mut moved: Option<(bool, usize, f64)> = None;
        // each row's middle, half-height and range, and its bars' tips
        let mut scales = [(0.0f32, 1.0f32, 1.0f64); 2];
        let mut tips = [[Vec2::ZERO; MAX]; 2];
        for (r, (rect, values, range, name, dragged)) in rows.into_iter().enumerate() {
            let (min, max) = rect;
            let w = (max.x - min.x) / n as f32;
            let mid = (min.y + max.y) * 0.5;
            let half = (max.y - min.y) * 0.5;
            f.text_px(Vec2::new(min.x, min.y - 5.0), name.as_str(), lay.small, pal::TEXT.a(0.9 * alpha), Align2::LEFT_BOTTOM);
            f.text_px(Vec2::new(max.x, min.y - 5.0), format!(r"$\pm {}$", short(range)), lay.small - 1.0, pal::DIM.a(alpha), Align2::RIGHT_BOTTOM);
            plot::line_of(f, plot::AXES, Vec2::new(min.x, mid), Vec2::new(max.x, mid), 1.0, pal::DIM.a(0.7 * alpha));
            for (e, edge) in [min.y, max.y].into_iter().enumerate() {
                let _ = e;
                plot::line_of(f, plot::AXES, Vec2::new(min.x, edge), Vec2::new(max.x, edge), 1.0, pal::FAINT.a(0.4 * alpha));
            }
            let missing = solve && r == 0 && !p.solved;
            let color = if dragged { DRAGGED } else { MADE };
            scales[r] = (mid, half, range);
            for k in 0..n {
                let v = values.get(k).copied().unwrap_or(0.0);
                let x0 = min.x + (k as f32 + 0.2) * w;
                let x1 = min.x + (k as f32 + 0.8) * w;
                let y = mid - v.max(-1.0).min(1.0) as f32 * half;
                let fill_color = if s.kind == Permutation { SET_COLORS[cycle_of(k) % SET_COLORS.len()] } else { color };
                if !missing {
                    plot::fill(f, (Vec2::new(x0, y.min(mid)), Vec2::new(x1, y.max(mid))), fill_color.a(0.55 * alpha));
                    plot::line_of(f, BAR_MARKS, Vec2::new(x0, y), Vec2::new(x1, y), 2.0, fill_color.a(alpha));
                }
                // the nudge: the x before it, dashed, and the change in red (without a
                // nudge, the dashes stand unseen where they will appear, at the bar's top)
                if r == 0 && solve {
                    let shown = p.ghost.is_some() && !missing;
                    let g = p.ghost.as_ref().and_then(|g| g.get(k).copied()).unwrap_or(v);
                    let gy = mid - g.max(-1.0).min(1.0) as f32 * half;
                    plot::dashed_of(f, GHOST, Vec2::new(x0 - 2.0, gy), Vec2::new(x1 + 2.0, gy), 1.4, pal::WHITE.a(if shown { 0.8 * alpha } else { 0.0 }));
                    if shown {
                        plot::fill(f, (Vec2::new(x0 + 0.25 * (x1 - x0), y.min(gy)), Vec2::new(x1 - 0.25 * (x1 - x0), y.max(gy))), CHANGE.a(0.75 * alpha));
                    }
                }
                if r == 1 {
                    f.text_px(Vec2::new((x0 + x1) * 0.5, max.y + 2.0), format!("${}$", k + 1), lay.small - 1.5, pal::DIM.a(alpha), Align2::CENTER_TOP);
                }
                tips[r][k.min(MAX - 1)] = Vec2::new((x0 + x1) * 0.5, y);
            }
            if missing {
                f.text_px(Vec2::new((min.x + max.x) * 0.5, mid), format!(r"no inverse: ${sym}x = b$ has no solution for most $b$"), lay.small, CHANGE.a(alpha), Align2::CENTER_CENTER);
            }
        }
        // the handles, on the row that is dragged (gliding to the other when the mode changes)
        if live {
            let r = if solve { 1 } else { 0 };
            let (mid, half, range) = scales[r];
            for k in 0..n {
                let at = tips[0][k].lerp(tips[1][k], p.handle_row);
                if let Some(to) = plot::screen_handle(f, k as u32, at, DRAGGED) {
                    let v = ((mid - to.y) / half.max(1.0)) as f64 * range;
                    moved = Some((r == 0, k, ((v * 100.0).round() / 100.0).max(-range).min(range)));
                }
            }
        }
        if let Some((is_x, k, v)) = moved {
            let n = self.setup.n;
            if is_x {
                self.x[k.min(MAX - 1)] = v;
            } else {
                self.b[k.min(MAX - 1)] = v;
            }
            let slots = Slots { n };
            self.refresh();
            // (what the hand moves is drawn where the hand is)
            let at = if is_x { slots.top() + k } else { slots.bottom() + k };
            let range = if is_x { self.now.top_range } else { self.now.bottom_range };
            self.shown.put(at, &[(v / range) as f32]);
        }
    }

    /// The eigenvalues in the complex plane, over what the kind promises them.
    fn plane(&mut self, f: &mut Frame, lay: &Layout, p: &Pic, live: bool) {
        let Some((min, max)) = lay.plane else { return };
        if max.x - min.x < 60.0 || max.y - min.y < 50.0 {
            return;
        }
        let s = &p.setup;
        let alpha = p.alpha;
        let small = lay.small;
        let title = if lay.narrow { "eigenvalues".to_string() } else { "the eigenvalues (orange) in the complex plane".to_string() };
        let key = promise_words(s, p.discs);
        if lay.narrow {
            f.text_px(Vec2::new(min.x - 30.0, min.y - 6.0), title, small + 0.5, pal::TEXT.a(alpha), Align2::LEFT_BOTTOM);
        } else {
            f.text_px(Vec2::new(min.x - 30.0, min.y - PLANE_ABOVE + 2.0), title, small + 0.5, pal::TEXT.a(alpha), Align2::LEFT_TOP);
            f.text_px(Vec2::new(min.x - 30.0, min.y - PLANE_ABOVE + 21.0), key, small - 0.5, pal::DIM.a(alpha), Align2::LEFT_TOP);
        }
        let [cx, hx, hy] = p.plane;
        let scale = ((max.x - min.x) as f64 / (2.0 * hx)).min((max.y - min.y) as f64 / (2.0 * hy));
        let mid = (min + max) * 0.5;
        let at = move |z: DVec2| mid + Vec2::new(((z.x - cx) * scale) as f32, -(z.y * scale) as f32);
        let inside = |q: Vec2| q.max(min).min(max);
        // the axes, and marks at every half along them, each mark always drawn (unseen off
        // the plane) so that the marks glide with the plane and none takes another's place
        let zero = at(DVec2::ZERO);
        let seen_x = |x: f32| x >= min.x - 0.5 && x <= max.x + 0.5;
        let seen_y = |y: f32| y >= min.y - 0.5 && y <= max.y + 0.5;
        plot::line_of(f, plot::AXES, Vec2::new(min.x, zero.y), Vec2::new(max.x, zero.y), 1.0, pal::DIM.a(0.7 * alpha));
        let axis_x = zero.x.max(min.x).min(max.x);
        let upright = seen_x(zero.x);
        plot::line_of(f, plot::AXES, Vec2::new(axis_x, min.y), Vec2::new(axis_x, max.y), 1.0, pal::DIM.a(if upright { 0.5 * alpha } else { 0.0 }));
        // (numbers on the whole ones, or every other, or on the halves too, by the room)
        let per = scale as f32;
        let every = if per >= 90.0 {
            1
        } else if per >= 34.0 {
            2
        } else if per >= 17.0 {
            4
        } else {
            10
        };
        for k in -REAL_MARKS..=REAL_MARKS {
            let v = k as f64 * 0.5;
            let q = at(DVec2::new(v, 0.0));
            let seen = seen_x(q.x);
            let x = q.x.max(min.x).min(max.x);
            let long = if k % 2 == 0 { 4.0 } else { 2.5 };
            plot::line_of(f, PLANE_MARKS, Vec2::new(x, zero.y), Vec2::new(x, zero.y + long), 1.0, pal::DIM.a(if seen { alpha } else { 0.0 }));
            let named = seen && k != 0 && k % every == 0;
            f.text_px(Vec2::new(x, zero.y + 5.0), format!("${}$", half_text(k)), small - 1.5, pal::DIM.a(if named { alpha } else { 0.0 }), Align2::CENTER_TOP);
        }
        for k in -IMAG_MARKS..=IMAG_MARKS {
            let q = at(DVec2::new(0.0, k as f64 * 0.5));
            let seen = upright && seen_y(q.y) && k != 0;
            let y = q.y.max(min.y).min(max.y);
            plot::line_of(f, PLANE_MARKS, Vec2::new(axis_x - 3.0, y), Vec2::new(axis_x + 3.0, y), 1.0, pal::DIM.a(if seen { alpha } else { 0.0 }));
            let named = seen && k % every == 0;
            let text = match k {
                2 => "$i$".to_string(),
                -2 => "$-i$".to_string(),
                0 => String::new(),
                _ => format!("${}i$", half_text(k)),
            };
            f.text_px(Vec2::new(axis_x - 5.0, y), text, small - 1.5, pal::DIM.a(if named { alpha } else { 0.0 }), Align2::RIGHT_CENTER);
        }
        // the promise, and the old one fading out
        let items = promise(p);
        draw_items(f, &items, &at, (min, max), alpha * self.old_promise.as_ref().map_or(1.0, |(_, t)| smoothstep(*t)), PROMISED);
        if let Some((old, t)) = &self.old_promise {
            draw_items(f, old, &at, (min, max), alpha * (1.0 - smoothstep(*t)), PROMISED_OLD);
        } else {
            plot::apart(f, PROMISED_OLD, false);
        }
        // the eigenvalues, and how many fall together where several do
        for z in &p.eig {
            plot::disc(f, inside(at(*z)), 9.0, EIGEN.a(alpha));
        }
        for (z, k) in gather(self.now.eig.iter().map(|z| DVec2::new((z.x * 1e6).round() / 1e6, (z.y * 1e6).round() / 1e6)).collect()) {
            if k > 1 {
                let q = inside(at(z));
                f.text_px(q + Vec2::new(7.0, -7.0), format!(r"$\times {k}$"), small - 0.5, EIGEN.a(alpha), Align2::LEFT_BOTTOM);
            }
        }
        // the companion's roots, to drag
        if s.kind == Companion {
            let n = s.n;
            let mut moved: Option<(usize, DVec2)> = None;
            let mut held = false;
            let pairs = n / 2;
            for k in 0..pairs + n % 2 {
                let z = if k < pairs { s.roots.pairs[k] } else { DVec2::new(s.roots.real, 0.0) };
                if k < pairs {
                    hud_circle(f, inside(at(DVec2::new(z.x, -z.y))), 8.0, 1.5, DRAGGED.a(0.8 * alpha));
                }
                hud_circle(f, inside(at(z)), 10.0, 1.8, DRAGGED.a(alpha));
                if live {
                    if let Some(to) = plot::screen_handle(f, 40 + k as u32, inside(at(z)), DRAGGED) {
                        held = true;
                        let to = to.max(min).min(max);
                        let w = DVec2::new(cx + (to.x - mid.x) as f64 / scale, -((to.y - mid.y) as f64) / scale);
                        let w = DVec2::new((w.x * 100.0).round() / 100.0, ((w.y * 100.0).round() / 100.0).max(0.0));
                        moved = Some((k, if k < pairs { w } else { DVec2::new(w.x, 0.0) }));
                    }
                } else {
                    plot::disc(f, inside(at(z)), 11.0, DRAGGED.a(alpha));
                }
            }
            if held != self.holding_root {
                self.holding_root = held;
                if !held {
                    self.refresh();
                }
            }
            if let Some((k, w)) = moved {
                let mut roots = self.setup.roots;
                if k < pairs {
                    roots.pairs[k] = w;
                } else {
                    roots.real = w.x;
                }
                if roots != self.setup.roots {
                    self.setup.roots = roots;
                    self.refresh();
                }
            }
        }
    }

    /// The singular values on a scale of powers of ten, and the condition number.
    fn strip(&self, f: &mut Frame, lay: &Layout, p: &Pic) {
        let Some((min, max)) = lay.sv else { return };
        if max.x - min.x < 60.0 {
            return;
        }
        let alpha = p.alpha;
        let small = lay.small;
        let [lo, hi] = p.sv_range;
        let span = (hi - lo).max(1e-6);
        // (a singular value of 0 has no place on the scale: it is drawn to the left of it)
        let nought = min.x - 24.0;
        let x_of = |l: f64| (min.x + (max.x - min.x) * ((l - lo) / span) as f32).max(nought).min(max.x);
        let mid = (min.y + max.y) * 0.5;
        let title = if lay.narrow { "singular values" } else { r"singular values $\sigma_1 \ge \dots \ge \sigma_n$, on a scale of powers of ten" };
        f.text_px(Vec2::new(min.x - if lay.narrow { 14.0 } else { 30.0 }, min.y - 5.0), title, small, pal::TEXT.a(0.9 * alpha), Align2::LEFT_BOTTOM);
        let kappa = if p.rank == p.setup.n {
            format!(r"$\kappa = \sigma_1/\sigma_n = {}$", kappa_text(&p.sv))
        } else {
            format!(r"$\kappa = \infty$: {} of them 0", p.setup.n - p.rank)
        };
        f.text_px(Vec2::new(max.x + 20.0, min.y - 5.0), kappa, small, if p.rank == p.setup.n { SINGULAR } else { CHANGE }.a(alpha), Align2::RIGHT_BOTTOM);
        plot::line_of(f, plot::AXES, Vec2::new(min.x, mid), Vec2::new(max.x, mid), 1.0, pal::DIM.a(0.7 * alpha));
        // the powers of ten, each always drawn (unseen off the strip), so that they glide
        // with it; about one in every 50 pixels named
        let every = (((hi - lo) as f32 * 50.0 / (max.x - min.x).max(1.0)).ceil() as i32).max(1);
        for e in DECADES.0..=DECADES.1 {
            let at = min.x + (max.x - min.x) * ((e as f64 - lo) / span) as f32;
            let seen = at >= min.x - 0.5 && at <= max.x + 0.5;
            let x = at.max(min.x).min(max.x);
            plot::line_of(f, SV_MARKS, Vec2::new(x, mid - 4.0), Vec2::new(x, mid + 4.0), 1.0, pal::DIM.a(if seen { alpha } else { 0.0 }));
            let named = seen && e.rem_euclid(every) == 0;
            let text = match e {
                0 => "$1$".to_string(),
                1 => "$10$".to_string(),
                _ => format!("$10^{{{e}}}$"),
            };
            f.text_px(Vec2::new(x, max.y + 1.0), text, small - 1.5, pal::DIM.a(if named { alpha } else { 0.0 }), Align2::CENTER_TOP);
        }
        for (k, l) in p.sv_log.iter().enumerate() {
            let zero = k >= p.rank;
            let x = x_of(*l);
            plot::disc(f, Vec2::new(x, mid), 8.0, if zero { CHANGE } else { SINGULAR }.a(alpha));
        }
        if p.rank < p.setup.n {
            f.text_px(Vec2::new(nought, mid + 6.0), "$0$", small - 1.5, CHANGE.a(alpha), Align2::CENTER_TOP);
        }
    }

    /// The lines on what the pattern buys, and below them the kind's own picture.
    fn say(&mut self, f: &mut Frame, lay: &Layout, p: &Pic) {
        let Some((at, width, room)) = lay.words else { return };
        let size = if lay.narrow { 11.0 } else { 12.0 };
        let mut y = at.y;
        let lines = p.words.clone();
        let line_h = size + 4.0;
        for (k, text) in lines.iter().enumerate() {
            let wrapped = self.wrap(f, text, size, width);
            // (a sentence is written whole or not at all: the panel has every one)
            if y + line_h * wrapped.len() as f32 > at.y + room {
                return;
            }
            for line in wrapped {
                f.text_px(Vec2::new(at.x, y), line, size, if k == 0 { pal::TEXT } else { pal::DIM }.a(p.alpha), Align2::LEFT_TOP);
                y += line_h;
            }
            y += 4.0;
        }
        let left = at.y + room - y;
        if left >= 110.0 && matches!(p.setup.kind, Laplacian | Nilpotent) {
            let r = (Vec2::new(at.x, y + 8.0), Vec2::new(at.x + width, at.y + room));
            match p.setup.kind {
                Laplacian => graph(f, r, p, lay.small),
                _ => powers(f, r, p, lay.small),
            }
        }
    }

    /// Words broken to a width (logical pixels), formulas kept whole; kept from frame to frame.
    fn wrap(&mut self, f: &Frame, text: &str, size: f32, width: f32) -> Vec<String> {
        let key = (text.to_string(), size.to_bits(), width.to_bits());
        if let Some(lines) = self.wraps.get(&key) {
            return lines.clone();
        }
        let mut words: Vec<&str> = Vec::new();
        let (mut start, mut inside) = (0, false);
        for (i, c) in text.char_indices() {
            match c {
                '$' => inside = !inside,
                ' ' if !inside => {
                    words.push(&text[start..i]);
                    start = i + 1;
                }
                _ => {}
            }
        }
        words.push(&text[start.min(text.len())..]);
        let mut lines: Vec<String> = Vec::new();
        for word in words.into_iter().filter(|w| !w.is_empty()) {
            match lines.last_mut() {
                Some(last) if f.text_size(&format!("{last} {word}"), size).x <= width => {
                    last.push(' ');
                    last.push_str(word);
                }
                _ => lines.push(word.to_string()),
            }
        }
        if self.wraps.len() > 400 {
            self.wraps.clear();
        }
        self.wraps.insert(key, lines.clone());
        lines
    }
}

/// The plane's marks: every half from −8 to 8 along the real line, and from −4i to 4i
/// up the imaginary one.
const REAL_MARKS: i32 = 16;
const IMAG_MARKS: i32 = 8;

/// k halves, as markup: 1.5, −2, 0.5.
fn half_text(k: i32) -> String {
    if k % 2 == 0 {
        num((k / 2) as f64, 0)
    } else {
        num(k as f64 * 0.5, 1)
    }
}

/// What is drawn of the promise, in words, over the plane.
fn promise_words(s: &Setup, discs: bool) -> String {
    let base = match s.kind {
        Diagonal | Triangular => "green rings: the diagonal entries",
        Tridiagonal => r"green: $2 - 2\cos(k\pi/(n+1))$, between 0 and 4",
        Nilpotent => "green: all of them at 0",
        Differences | Sums => "green: all of them at 1",
        Blocks => "rings: each block's own eigenvalues, in its colour",
        Circulant => "green curve: the Fourier transform of the first row",
        LowRank => r"green: $n - r$ of them at 0",
        Markov => r"green: the unit circle, and 1; dashed: $|\lambda_2|$",
        Laplacian => "green: no negative ones; a 0 for each piece",
        Dominant => "blue: the Gershgorin discs, none of them reaching 0",
        Permutation => "rings: the roots of 1 for each cycle, in its colour",
        Companion => "white: the roots of $p$; drag them",
        Hilbert => "green: the positive real numbers",
        Banded | Hessenberg | Toeplitz | Vandermonde => "the pattern promises nothing here",
    };
    if discs && s.kind != Dominant {
        format!("{base}; blue: the Gershgorin discs")
    } else {
        base.to_string()
    }
}

/// The promise drawn: `at` takes a complex number to the screen, everything kept in `rect`.
fn draw_items(f: &mut Frame, items: &[Item], at: &dyn Fn(DVec2) -> Vec2, rect: Rect, alpha: f32, kind: u8) {
    let (min, max) = rect;
    let inside = |q: Vec2| q.max(min).min(max);
    plot::apart(f, kind, false);
    for item in items {
        match item {
            Item::Ring(z, r, color, count) => {
                let q = at(*z);
                if plot::within(q, rect, 0.0) && *count > 0 {
                    hud_circle(f, q, *r, 1.6, color.a(0.9 * alpha));
                }
            }
            Item::Circle(c, r, color, dashed) => {
                let pts: Vec<Vec2> = (0..=96).map(|k| at(*c + DVec2::from_angle(std::f64::consts::TAU * k as f64 / 96.0) * *r)).collect();
                let cut = plot::clip(&pts, rect);
                if *dashed {
                    for w in cut.windows(2).step_by(2) {
                        plot::dashed_of(f, GHOST, w[0], w[1], 1.3, color.a(0.75 * alpha));
                    }
                } else {
                    plot::path_of(f, kind, &cut, 1.6, color.a(0.8 * alpha));
                }
            }
            Item::Curve(pts, color) => {
                let pts: Vec<Vec2> = pts.iter().map(|z| at(*z)).collect();
                plot::path_of(f, kind, &plot::clip(&pts, rect), 2.0, color.a(0.8 * alpha));
            }
            Item::Band(from, to, color) => {
                let (a, b) = (inside(at(DVec2::new(*from, 0.0))), inside(at(DVec2::new(to.min(1e12), 0.0))));
                plot::fill(f, (Vec2::new(a.x, a.y - 4.0), Vec2::new(b.x, b.y + 4.0)), color.a(0.22 * alpha));
            }
            Item::Disc(c, r, color) => {
                let centre = at(*c);
                let radius = (at(*c + DVec2::new(*r, 0.0)).x - centre.x).max(0.0);
                let pts: Vec<Vec2> = (0..=72).map(|k| centre + Vec2::from_angle(TAU * k as f32 / 72.0) * radius).collect();
                plot::path_of(f, DISC_LINES, &plot::clip(&pts, rect), 1.2, color.a(0.7 * alpha));
                let _ = inside;
            }
        }
    }
}

/// The graph of a Laplacian: its points round a circle, coloured by piece, and its edges.
fn graph(f: &mut Frame, (min, max): Rect, p: &Pic, small: f32) {
    let n = p.setup.n;
    let alpha = p.alpha;
    let (piece, count) = pieces(&p.setup.edges, n);
    let centre = (min + max) * 0.5 + Vec2::new(0.0, 6.0);
    let radius = ((max.y - min.y) * 0.5 - 16.0).min((max.x - min.x) * 0.25).max(10.0);
    let spot = |i: usize| centre + Vec2::from_angle(PI - TAU * i as f32 / n as f32) * radius;
    f.text_px(Vec2::new(min.x, min.y), format!("the graph: {count} {}", if count == 1 { "piece" } else { "pieces" }), small, pal::TEXT.a(alpha), Align2::LEFT_TOP);
    for i in 0..n {
        for j in i + 1..n {
            if p.setup.edges[i][j] {
                plot::line_of(f, GRAPH, spot(i), spot(j), 1.6, pal::DIM.a(alpha));
            }
        }
    }
    for i in 0..n {
        plot::disc(f, spot(i), 15.0, SET_COLORS[piece[i] % SET_COLORS.len()].a(alpha));
        f.text_px(spot(i), format!("${}$", i + 1), small - 1.0, Color::hex(0x0B0F1C).a(alpha), Align2::CENTER_CENTER);
    }
}

/// The powers of a strictly upper triangular matrix, each a small grid: the zeros climb
/// until the n-th power is 0.
fn powers(f: &mut Frame, (min, max): Rect, p: &Pic, small: f32) {
    let n = p.setup.n;
    let alpha = p.alpha;
    let m = p.written;
    f.text_px(min, "its powers: each climbs a diagonal higher", small, pal::TEXT.a(alpha), Align2::LEFT_TOP);
    let gap = 10.0;
    let side = (((max.x - min.x) - gap * (n - 1) as f32) / n as f32).min(max.y - min.y - 40.0).max(0.0);
    if side < 20.0 {
        return;
    }
    let c = side / n as f32;
    let mut pw = m;
    for k in 1..=n {
        let at = min + Vec2::new((k - 1) as f32 * (side + gap), 22.0);
        let scale = pw.largest();
        for i in 0..n {
            for j in 0..n {
                let lo = at + Vec2::new(j as f32, i as f32) * c;
                let v = pw.get(i, j);
                let color = if v.abs() <= 1e-12 * scale.max(1e-300) || scale == 0.0 { BASE } else { heat(v, scale) };
                plot::fill(f, (lo + Vec2::splat(0.5), lo + Vec2::splat(c - 0.5)), color.a(alpha));
            }
        }
        let name = if k == n { format!("$N^{{{k}}} = 0$") } else { format!("$N{}$", power(k)) };
        f.text_px(at + Vec2::new(side * 0.5, side + 3.0), name, small - 1.0, pal::DIM.a(alpha), Align2::CENTER_TOP);
        pw = pw.mul(&m);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Whether two lists of complex numbers are the same up to order, each within `tol`.
    fn same_set(a: &[DVec2], b: &[DVec2], tol: f64) -> bool {
        if a.len() != b.len() {
            return false;
        }
        let mut left: Vec<DVec2> = b.to_vec();
        a.iter().all(|z| match left.iter().enumerate().min_by(|x, y| (*x.1 - *z).length().total_cmp(&(*y.1 - *z).length())) {
            Some((i, w)) if (*w - *z).length() <= tol => {
                left.remove(i);
                true
            }
            _ => false,
        })
    }

    fn real(v: &[f64]) -> Vec<DVec2> {
        v.iter().map(|x| DVec2::new(*x, 0.0)).collect()
    }

    /// A random orthogonal n × n matrix.
    fn orthogonal(rng: &mut Rng, n: usize) -> Mat {
        let mut a: Vec<f64> = (0..n * n).map(|_| rng.normal() as f64).collect();
        crate::math::linalg::orthonormalise(&mut a, n, n);
        Mat::from_fn(n, |i, j| a[i * n + j])
    }

    /// Eigenvalues of Q T Qᵀ, with T upper triangular but for 2 × 2 blocks for complex
    /// pairs, are the ones T was made with: for every n up to 8, real ones and pairs.
    #[test]
    fn eigenvalues_of_matrices_made_from_them() {
        let mut rng = Rng::new(5);
        for n in 1..=MAX {
            for case in 0..150 {
                // eigenvalues at least 0.3 apart, some of them in complex pairs
                let mut t = Mat::zero(n);
                let mut want = Vec::new();
                let mut i = 0;
                while i < n {
                    let pair = i + 1 < n && rng.bool(0.4);
                    let a = loop {
                        let a = rng.range(-3.0, 3.0) as f64;
                        if want.iter().all(|z: &DVec2| (z.x - a).abs() > 0.3) {
                            break a;
                        }
                    };
                    if pair {
                        let (b, c) = (rng.range(0.4, 2.0) as f64, rng.range(0.4, 2.0) as f64);
                        t.set(i, i, a);
                        t.set(i + 1, i + 1, a);
                        t.set(i, i + 1, b);
                        t.set(i + 1, i, -c);
                        let im = (b * c).sqrt();
                        want.extend([DVec2::new(a, im), DVec2::new(a, -im)]);
                        i += 2;
                    } else {
                        t.set(i, i, a);
                        want.push(DVec2::new(a, 0.0));
                        i += 1;
                    }
                }
                for r in 0..n {
                    for c in r + 1..n {
                        if t.get(r, c) == 0.0 && !(c == r + 1 && t.get(c, r) != 0.0) {
                            t.set(r, c, rng.range(-1.0, 1.0) as f64);
                        }
                    }
                }
                let q = orthogonal(&mut rng, n);
                let a = q.mul(&t).mul(&q.transpose());
                let got = eigenvalues(&a).expect("it settles");
                assert!(same_set(&got, &want, 1e-7), "n = {n}, case {case}: {got:?} against {want:?}");
            }
        }
    }

    /// For any matrix the eigenvalues add up to the trace and multiply to the
    /// determinant, and come in conjugate pairs.
    #[test]
    fn eigenvalues_add_to_the_trace_and_multiply_to_the_determinant() {
        let mut rng = Rng::new(9);
        for n in LEAST..=MAX {
            for _ in 0..300 {
                let a = Mat::from_fn(n, |_, _| rng.range(-2.0, 2.0) as f64);
                let eig = eigenvalues(&a).expect("it settles");
                let sum = eig.iter().fold(DVec2::ZERO, |s, z| s + *z);
                let product = eig.iter().fold(DVec2::new(1.0, 0.0), |p, z| DVec2::new(p.x * z.x - p.y * z.y, p.x * z.y + p.y * z.x));
                assert!((sum.x - a.trace()).abs() < 1e-9 && sum.y.abs() < 1e-9, "{sum:?} {}", a.trace());
                assert!((product.x - a.det()).abs() < 1e-8 * (1.0 + a.det().abs()) && product.y.abs() < 1e-8 * (1.0 + a.det().abs()), "{product:?} {}", a.det());
                let mirrored: Vec<DVec2> = eig.iter().map(|z| DVec2::new(z.x, -z.y)).collect();
                assert!(same_set(&eig, &mirrored, 1e-9));
            }
        }
        // and nothing that is not a number gets an answer
        let mut bad = Mat::identity(4);
        bad.set(1, 2, f64::NAN);
        assert!(eigenvalues(&bad).is_none());
        assert_eq!(eigenvalues(&Mat::zero(5)), Some(vec![DVec2::ZERO; 5]));
    }

    /// Members of a kind for the tests: the first, and many others, at every n.
    fn members(kind: Kind) -> Vec<Setup> {
        let mut out = Vec::new();
        for n in LEAST..=MAX {
            let first = Setup::new(kind, n);
            out.push(first);
            if kind.set_by_n() {
                continue;
            }
            let mut scene = Patterns::new();
            scene.take(first, None);
            for _ in 0..25 {
                scene.another();
                out.push(scene.setup);
            }
            if let Some(p) = kind.param(n) {
                for v in p.lo..=p.hi {
                    for seed in 0..4 {
                        out.push(Setup { param: v, seed, ..first });
                    }
                }
            }
        }
        out
    }

    fn eig_of(s: &Setup) -> Vec<DVec2> {
        eigenvalues(&build(s)).expect("it settles")
    }

    #[test]
    fn the_second_difference_matrix_has_its_eigenvalues() {
        for n in LEAST..=MAX {
            let s = Setup::new(Tridiagonal, n);
            let want: Vec<f64> = (1..=n).map(|k| 2.0 - 2.0 * (k as f64 * std::f64::consts::PI / (n + 1) as f64).cos()).collect();
            let got = eig_of(&s);
            assert!(same_set(&got, &real(&want), 1e-10), "{n}: {got:?}");
            assert!(want.iter().all(|l| *l > 0.0 && *l < 4.0));
            let m = build(&s);
            assert_eq!(m, m.transpose());
        }
        // the numbers the picture opens with
        let scene = Patterns::new();
        assert_eq!(kappa_text(&scene.now.sv), "19.2");
        let x = first_x(Tridiagonal, 6);
        let kx = build(&Setup::new(Tridiagonal, 6)).apply(&x[..6]);
        // a tent: 0 where it runs straight, above 0 where it bends down
        assert!(kx.iter().enumerate().all(|(i, v)| if i == 2 || i == 3 { *v > 0.5 } else { v.abs() < 1e-12 }), "{kx:?}");
    }

    #[test]
    fn diagonal_and_triangular_matrices_have_their_diagonals_for_eigenvalues() {
        for kind in [Diagonal, Triangular, Differences, Sums, Nilpotent] {
            for s in members(kind) {
                let m = build(&s);
                let diag: Vec<f64> = (0..s.n).map(|i| m.get(i, i)).collect();
                assert!(same_set(&eig_of(&s), &real(&diag), 1e-12), "{kind:?} {s:?}");
                for i in 0..s.n {
                    for j in 0..s.n {
                        if kind.mark(&s, i, j) == Mark::Zero {
                            assert_eq!(m.get(i, j), 0.0);
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn a_strictly_upper_triangular_matrix_is_nilpotent() {
        for s in members(Nilpotent) {
            let n = s.n;
            let m = build(&s);
            let mut power = m;
            for k in 1..n {
                // zero on and below its k-th diagonal above the main one
                for i in 0..n {
                    for j in 0..(i + k).min(n) {
                        assert!(power.get(i, j).abs() < 1e-12, "{k} {i} {j}");
                    }
                }
                power = power.mul(&m);
            }
            assert!(power.largest() < 1e-12, "N^n = 0");
            assert!(eig_of(&s).iter().all(|z| z.length() == 0.0));
            assert_eq!(build(&s).det(), 0.0);
        }
    }

    #[test]
    fn differences_and_running_sums_undo_each_other() {
        for n in LEAST..=MAX {
            let d = build(&Setup::new(Differences, n));
            let s = build(&Setup::new(Sums, n));
            assert_eq!(d.mul(&s), Mat::identity(n));
            assert_eq!(s.mul(&d), Mat::identity(n));
            // only one line of eigenvectors, the last axis, for each: D - I and S - I have rank n - 1
            for m in [d, s] {
                let mut less = m;
                for i in 0..n {
                    less.set(i, i, 0.0);
                }
                let (sv, _) = singular(&less);
                assert_eq!(sv.iter().filter(|v| **v > 1e-9).count(), n - 1);
                let last: Vec<f64> = (0..n).map(|i| if i == n - 1 { 1.0 } else { 0.0 }).collect();
                assert_eq!(less.apply(&last), vec![0.0; n]);
            }
        }
    }

    #[test]
    fn banded_hessenberg_and_block_matrices_keep_their_shapes() {
        for kind in [Banded, Hessenberg, Blocks, Toeplitz, Circulant, Companion, Laplacian, Permutation] {
            for s in members(kind) {
                let m = build(&s);
                for i in 0..s.n {
                    for j in 0..s.n {
                        if kind.mark(&s, i, j) == Mark::Zero {
                            assert_eq!(m.get(i, j), 0.0, "{kind:?} {s:?} {i} {j}");
                        }
                    }
                }
                match kind {
                    // each diagonal entry outweighs the rest of its row: no row exchanges
                    Banded => assert!(gershgorin(&m).iter().all(|(c, r)| c.x.abs() > *r)),
                    Toeplitz => assert!((1..s.n).all(|i| (1..s.n).all(|j| m.get(i, j) == m.get(i - 1, j - 1)))),
                    Circulant => assert!((1..s.n).all(|i| (0..s.n).all(|j| m.get(i, j) == m.get(i - 1, (j + s.n - 1) % s.n)))),
                    Blocks => {
                        let k = s.param;
                        let first = Mat::from_fn(k, |i, j| m.get(i, j));
                        let second = Mat::from_fn(s.n - k, |i, j| m.get(k + i, k + j));
                        let mut both = eigenvalues(&first).unwrap();
                        both.extend(eigenvalues(&second).unwrap());
                        assert!(same_set(&eig_of(&s), &both, 1e-8), "{s:?}");
                    }
                    _ => {}
                }
            }
        }
    }

    /// A QR step keeps the Hessenberg shape: A = QR, then RQ.
    #[test]
    fn a_qr_step_keeps_the_hessenberg_shape() {
        for s in members(Hessenberg) {
            let n = s.n;
            let a = build(&s);
            let mut q = a.dense();
            crate::math::linalg::orthonormalise(&mut q, n, n);
            let q = Mat::from_fn(n, |i, j| q[i * n + j]);
            let r = q.transpose().mul(&a);
            let next = r.mul(&q);
            for i in 0..n {
                for j in 0..i.saturating_sub(1) {
                    assert!(next.get(i, j).abs() < 1e-12 * a.largest(), "{i} {j}: {}", next.get(i, j));
                }
            }
        }
    }

    #[test]
    fn circulants_have_the_fourier_transform_of_their_first_row_for_eigenvalues() {
        for s in members(Circulant) {
            let n = s.n;
            let m = build(&s);
            let c: Vec<f64> = (0..n).map(|j| m.get(0, j)).collect();
            let want: Vec<DVec2> = (0..n).map(|k| symbol(&c, std::f64::consts::TAU * k as f64 / n as f64)).collect();
            assert!(same_set(&eig_of(&s), &want, 1e-9), "{s:?}");
            // and any two commute
            let other = build(&Setup { seed: s.seed + 7, ..s });
            let (ab, ba) = (m.mul(&other), other.mul(&m));
            assert!((0..n).all(|i| (0..n).all(|j| (ab.get(i, j) - ba.get(i, j)).abs() < 1e-12)));
        }
        // the blur: a stretch of the real line, from 0 to 1
        let blur: Vec<f64> = (0..8).map(|j| build(&Setup::new(Circulant, 8)).get(0, j)).collect();
        for k in 0..100 {
            let z = symbol(&blur, std::f64::consts::TAU * k as f64 / 100.0);
            assert!(z.y.abs() < 1e-12 && z.x > -1e-12 && z.x < 1.0 + 1e-12);
        }
    }

    #[test]
    fn a_permutation_has_the_roots_of_1_of_its_cycles() {
        for s in members(Permutation) {
            let n = s.n;
            let sigma = shuffle(n, s.seed);
            let mut want = Vec::new();
            for c in cycles(&sigma, n) {
                want.extend((0..c.len()).map(|k| DVec2::from_angle(std::f64::consts::TAU * k as f64 / c.len() as f64)));
            }
            assert!(same_set(&eig_of(&s), &want, 1e-7), "{s:?}: {:?}", eig_of(&s));
            let m = build(&s);
            assert_eq!(m.transpose().mul(&m), Mat::identity(n));
            assert!(singular(&m).0.iter().all(|v| (v - 1.0).abs() < 1e-12));
            // (Px)_i = x_sigma(i)
            let x: Vec<f64> = (0..n).map(|i| i as f64 + 1.0).collect();
            assert!(m.apply(&x).iter().enumerate().all(|(i, v)| *v == x[sigma[i]]));
        }
        assert_eq!(cycle_text(&cycles(&shuffle(6, 0), 6)), r"(1\,2\,3)(4\,5)(6)");
    }

    #[test]
    fn a_companion_matrix_has_the_roots_of_its_polynomial() {
        let mut rng = Rng::new(3);
        for n in LEAST..=MAX {
            for case in 0..200 {
                let mut roots = Roots::spread(n);
                if case > 0 {
                    for z in roots.pairs.iter_mut() {
                        *z = DVec2::new(rng.range(-1.5, 1.5) as f64, rng.range(0.1, 1.5) as f64);
                    }
                    roots.real = rng.range(-1.5, 1.5) as f64;
                }
                let s = Setup { roots, ..Setup::new(Companion, n) };
                let all = roots.all(n);
                // (roots close together are found less closely: about the square root of their distance's share)
                let nearest = all.iter().enumerate().flat_map(|(i, a)| all.iter().skip(i + 1).map(move |b| (*a - *b).length())).fold(f64::MAX, f64::min);
                let tol = if nearest > 0.05 { 1e-8 } else { 1e-5 };
                assert!(same_set(&eig_of(&s), &all, tol), "n = {n}, case {case}");
                // the last column is the coefficients with their signs changed
                let c = roots.coefficients(n);
                let m = build(&s);
                for (i, ci) in c.iter().enumerate() {
                    assert_eq!(m.get(i, n - 1), -ci);
                }
            }
        }
        // the roots "another" gives too
        for s in members(Companion) {
            assert!(same_set(&eig_of(&s), &s.roots.all(s.n), 1e-5), "{s:?}");
        }
    }

    #[test]
    fn low_rank_matrices_show_their_rank() {
        for s in members(LowRank) {
            let (sv, _) = singular(&build(&s));
            let rank = sv.iter().filter(|v| **v > RANK_TOL * sv[0]).count();
            assert_eq!(rank, s.param, "{s:?} {sv:?}");
            let zeros = eig_of(&s).iter().filter(|z| z.length() < 1e-7 * sv[0]).count();
            assert!(zeros >= s.n - s.param, "{s:?}");
        }
        // rank one: the other eigenvalue is the trace, v . u
        for s in members(LowRank).into_iter().filter(|s| s.param == 1) {
            let m = build(&s);
            let biggest = eig_of(&s).into_iter().max_by(|a, b| a.length().total_cmp(&b.length())).unwrap();
            // (when v . u is 0 the matrix is nilpotent, and its eigenvalues are found only to
            // about the square root of the rounding)
            assert!((biggest - DVec2::new(m.trace(), 0.0)).length() < 1e-7 * m.largest(), "{s:?}: {biggest:?} {}", m.trace());
        }
    }

    #[test]
    fn markov_matrices_keep_their_sums_and_settle() {
        for s in members(Markov) {
            let m = build(&s);
            let n = s.n;
            for j in 0..n {
                assert!(((0..n).map(|i| m.get(i, j)).sum::<f64>() - 1.0).abs() < 1e-12);
                assert!((0..n).all(|i| m.get(i, j) > 0.0));
            }
            let eig = eig_of(&s);
            assert!(eig.iter().any(|z| (*z - DVec2::new(1.0, 0.0)).length() < 1e-10), "{eig:?}");
            assert!(eig.iter().all(|z| z.length() <= 1.0 + 1e-10));
            // every entry above 0: all the others smaller in size than 1
            assert!(second_size(&eig).unwrap() < 1.0 - 1e-6);
            let x: Vec<f64> = (0..n).map(|i| (i as f64 * 0.7).sin()).collect();
            assert!((m.apply(&x).iter().sum::<f64>() - x.iter().sum::<f64>()).abs() < 1e-12);
            // step after step, the steady state scaled to what x adds up to
            let mut steady: Vec<f64> = vec![1.0 / n as f64; n];
            let mut y = x.clone();
            for _ in 0..2000 {
                steady = m.apply(&steady);
                y = m.apply(&y);
            }
            let total: f64 = x.iter().sum();
            assert!(y.iter().zip(&steady).all(|(a, b)| (a - total * b).abs() < 1e-9), "{y:?}");
            assert!(m.apply(&steady).iter().zip(&steady).all(|(a, b)| (a - b).abs() < 1e-12));
        }
    }

    #[test]
    fn a_laplacian_has_a_zero_eigenvalue_for_each_piece() {
        let mut seen = [false; MAX + 1];
        for s in members(Laplacian) {
            let n = s.n;
            let m = build(&s);
            assert_eq!(m, m.transpose());
            assert!((0..n).all(|i| (0..n).map(|j| m.get(i, j)).sum::<f64>() == 0.0));
            let eig = eig_of(&s);
            assert!(eig.iter().all(|z| z.y.abs() < 1e-9 && z.x > -1e-9), "{eig:?}");
            let (_, count) = pieces(&s.edges, n);
            assert_eq!(eig.iter().filter(|z| z.length() < 1e-8).count(), count, "{s:?}");
            seen[count] = true;
        }
        // random graphs in one piece and in several
        assert!(seen[1] && seen[2] && seen[3]);
        // the first graph is one piece, and cutting its bridge makes two
        for n in LEAST..=MAX {
            let mut s = Setup::new(Laplacian, n);
            assert_eq!(pieces(&s.edges, n).1, 1);
            let (i, j) = bridge(n);
            assert!(s.edges[i][j]);
            (s.edges[i][j], s.edges[j][i]) = (false, false);
            assert_eq!(pieces(&s.edges, n).1, 2);
        }
    }

    #[test]
    fn the_gershgorin_discs_hold_every_eigenvalue() {
        let mut rng = Rng::new(11);
        for n in LEAST..=MAX {
            for _ in 0..200 {
                let a = Mat::from_fn(n, |_, _| rng.range(-2.0, 2.0) as f64);
                let discs = gershgorin(&a);
                for z in eigenvalues(&a).unwrap() {
                    assert!(discs.iter().any(|(c, r)| (z - *c).length() <= r + 1e-9));
                }
            }
        }
        for s in members(Dominant) {
            let m = build(&s);
            let discs = gershgorin(&m);
            assert!(discs.iter().all(|(c, r)| c.x.abs() > *r), "no disc reaches 0");
            for z in eig_of(&s) {
                assert!(discs.iter().any(|(c, r)| (z - *c).length() <= r + 1e-9));
            }
        }
    }

    #[test]
    fn the_hilbert_matrix_is_positive_definite_and_hard_to_solve() {
        let kappas: Vec<String> = (LEAST..=MAX).map(|n| kappa_text(&singular(&build(&Setup::new(Hilbert, n))).0)).collect();
        assert_eq!(kappas, [r"524", r"1.6 \times 10^{4}", r"4.8 \times 10^{5}", r"1.5 \times 10^{7}", r"4.8 \times 10^{8}", r"1.5 \times 10^{10}"]);
        for n in LEAST..=MAX {
            let s = Setup::new(Hilbert, n);
            assert!(eig_of(&s).iter().all(|z| z.y == 0.0 && z.x > 0.0));
        }
        let eig = eig_of(&Setup::new(Hilbert, 6));
        let (lo, hi) = eig.iter().fold((f64::MAX, 0.0f64), |(lo, hi), z| (lo.min(z.x), hi.max(z.x)));
        assert_eq!((short(hi), short(lo)), ("1.62".to_string(), r"1.1 \times 10^{-7}".to_string()));
    }

    #[test]
    fn the_vandermonde_determinant_and_condition() {
        for s in members(Vandermonde) {
            let t = points(s.n, s.seed);
            let product: f64 = (0..s.n).flat_map(|i| (i + 1..s.n).map(move |j| (i, j))).map(|(i, j)| t[j] - t[i]).product();
            let det = build(&s).det();
            assert!((det - product).abs() <= 1e-9 * product.abs().max(1e-12), "{det} {product}");
            assert!(t.windows(2).all(|w| w[0] < w[1]));
        }
        // evenly spaced: about 7 times worse for each point added
        let kappa = |n: usize| {
            let sv = singular(&build(&Setup::new(Vandermonde, n))).0;
            sv[0] / sv[n - 1]
        };
        for n in 4..=MAX {
            let ratio = kappa(n) / kappa(n - 1);
            assert!((6.4..7.6).contains(&ratio), "{n}: {ratio}");
        }
        assert_eq!(kappa_text(&singular(&build(&Setup::new(Vandermonde, 6))).0), "4924");
    }

    /// Elimination within a band, counting its multiplications and divisions: the
    /// counts `work` says, and the right answer.
    #[test]
    fn the_work_counts_are_what_elimination_does() {
        let mut rng = Rng::new(21);
        for n in 1..=MAX {
            for p in 0..n {
                for q in 0..n {
                    // a matrix with that band, its diagonal large enough to need no exchanges
                    let mut a = Mat::from_fn(n, |i, j| if i > j + p || j > i + q { 0.0 } else { rng.range(-1.0, 1.0) as f64 });
                    for i in 0..n {
                        a.set(i, i, 2.0 * n as f64);
                    }
                    let b: Vec<f64> = (0..n).map(|_| rng.range(-1.0, 1.0) as f64).collect();
                    let (mut m, mut y, mut count) = (a, b.clone(), 0usize);
                    for k in 0..n {
                        for i in k + 1..(k + p + 1).min(n) {
                            let f = m.get(i, k) / m.get(k, k);
                            count += 1;
                            for j in k + 1..(k + q + 1).min(n) {
                                m.set(i, j, m.get(i, j) - f * m.get(k, j));
                                count += 1;
                            }
                            m.set(i, k, 0.0);
                            y[i] -= f * y[k];
                            count += 1;
                        }
                    }
                    let mut x = vec![0.0; n];
                    for i in (0..n).rev() {
                        let mut v = y[i];
                        for j in i + 1..(i + q + 1).min(n) {
                            v -= m.get(i, j) * x[j];
                            count += 1;
                        }
                        x[i] = v / m.get(i, i);
                        count += 1;
                    }
                    // nothing outside the band was ever touched, and the answer is right
                    assert_eq!(count, work(n, p, q), "{n} {p} {q}");
                    let back = a.apply(&x);
                    assert!(back.iter().zip(&b).all(|(u, v)| (u - v).abs() < 1e-12));
                }
            }
        }
        for n in LEAST..=MAX {
            assert_eq!(work(n, 0, 0), n);
            assert_eq!(work(n, 1, 1), 5 * n - 4);
            assert_eq!(work(n, 0, n - 1), n * (n + 1) / 2);
            assert_eq!(work(n, 1, n - 1), n * n + 2 * n - 2);
            assert_eq!(3 * full_work(n), n * n * n + 3 * n * n - n);
            for w in 1..n {
                assert!(work(n, w, w) <= n * (w * w + 3 * w + 1));
            }
        }
        assert_eq!(full_work(6), 106);
    }

    /// The scene as an experiment's button leaves it, run at 60 frames a second until
    /// it comes to rest.
    fn trial(k: usize) -> Patterns {
        let mut scene = Patterns::new();
        scene.set_up(k);
        for _ in 0..60 * 20 {
            scene.advance(1.0 / 60.0, 1.0 / 60.0);
            if !scene.moving() {
                break;
            }
        }
        assert!(!scene.moving());
        scene
    }

    #[test]
    fn differences_undo_sums_as_the_caption_says() {
        let scene = trial(0);
        assert!(scene.setup.kind == Differences && scene.setup.n == 6 && scene.mode == Mode::Solve);
        let mut sum = 0.0;
        for (i, b) in TRIAL_B.iter().enumerate() {
            sum += b;
            assert!((scene.now.top[i] - sum).abs() < 1e-12, "x is the running sums of b");
            assert_eq!(scene.now.bottom[i], *b);
        }
        assert!(scene.now.solved);
    }

    #[test]
    fn the_circulant_blur_does_what_its_caption_says() {
        let scene = trial(1);
        assert!(scene.setup.kind == Circulant && scene.setup.n == 8 && scene.mode == Mode::Multiply);
        let ax = &scene.now.bottom;
        assert_eq!([ax[1], ax[2], ax[3]], [0.375, 0.75, 0.375]);
        assert!(ax.iter().enumerate().all(|(i, v)| (1..=3).contains(&i) || *v == 0.0));
        let want: Vec<f64> = (0..8).map(|k| 0.5 + 0.5 * (std::f64::consts::TAU * k as f64 / 8.0).cos()).collect();
        assert!(same_set(&scene.now.eig, &real(&want), 1e-12));
        assert!(want.iter().all(|v| (-1e-12..=1.0).contains(v)));
        // one of them 0, for the alternating list: no inverse
        let alternating: Vec<f64> = (0..8).map(|i| if i % 2 == 0 { 1.0 } else { -1.0 }).collect();
        assert!(scene.now.m.apply(&alternating).iter().all(|v| v.abs() < 1e-15));
        assert_eq!(scene.now.rank, 7);
        assert_eq!(kappa_text(&scene.now.sv), r"\infty");
    }

    #[test]
    fn cutting_an_edge_does_what_its_caption_says() {
        let mut scene = Patterns::new();
        scene.set_up(2);
        // before the cut: one piece, one zero, the second eigenvalue 0.44, and Lx = ±1 at the edge
        for _ in 0..30 {
            scene.advance(1.0 / 60.0, 1.0 / 60.0);
        }
        assert!(scene.setup.kind == Laplacian && scene.setup.n == 6 && scene.later.is_some());
        let (i, j) = bridge(6);
        assert_eq!((i + 1, j + 1), (3, 4), "the edge from point 3 to point 4");
        let mut sizes: Vec<f64> = scene.now.eig.iter().map(|z| z.x).collect();
        sizes.sort_by(f64::total_cmp);
        assert!(sizes[0].abs() < 1e-12 && num(sizes[1], 2) == "0.44", "{sizes:?}");
        assert_eq!(scene.now.bottom, vec![0.0, 0.0, 1.0, -1.0, 0.0, 0.0]);
        assert_eq!((scene.now.m.get(2, 2), scene.now.m.get(3, 3)), (3.0, 3.0));
        let m = &scene.now.m;
        assert!(m.get(0, 1) == -1.0 && m.get(0, 2) == -1.0 && m.get(1, 2) == -1.0 && m.get(3, 4) == -1.0 && m.get(3, 5) == -1.0 && m.get(4, 5) == -1.0, "two triangles");
        // after it
        let scene = trial(2);
        let m = &scene.now.m;
        assert!(m.get(2, 3) == 0.0 && m.get(3, 2) == 0.0 && m.get(2, 2) == 2.0 && m.get(3, 3) == 2.0);
        assert_eq!(scene.now.eig.iter().filter(|z| z.length() < 1e-12).count(), 2);
        assert_eq!(pieces(&scene.setup.edges, 6).1, 2);
        assert!(scene.now.bottom.iter().all(|v| *v == 0.0), "Lx = 0");
    }

    #[test]
    fn the_hilbert_nudge_does_what_its_caption_says() {
        let scene = trial(3);
        assert!(scene.setup.kind == Hilbert && scene.setup.n == 6 && scene.mode == Mode::Solve && scene.nudge);
        let before = scene.now.ghost.clone().expect("the x before the nudge");
        assert!(before.iter().all(|v| (v - 1.0).abs() < 1e-8), "{before:?}");
        let x = &scene.now.top;
        let said: Vec<String> = x.iter().enumerate().map(|(i, v)| num(*v, if i < 3 { 2 } else { 1 })).collect();
        assert_eq!(said, ["1.04", "−0.15", "8.79", "−19.2", "23.3", "−7.8"]);
        let norm = crate::math::linalg::norm;
        let change: Vec<f64> = x.iter().zip(&before).map(|(a, b)| a - b).collect();
        let grew = norm(&change) / norm(&before);
        assert_eq!(num(grew, 0), "13");
        assert_eq!(short(grew / NUDGE), r"1.3 \times 10^{7}");
        assert_eq!(kappa_text(&scene.now.sv), r"1.5 \times 10^{7}");
        assert!(grew / NUDGE <= singular(&scene.now.m).0[0] / singular(&scene.now.m).0[5]);
        // b changed by one part in a million of its size
        let b = &scene.now.bottom;
        let size = norm(b);
        let worst = singular(&scene.now.m).1;
        let moved: Vec<f64> = b.iter().zip(&worst).map(|(v, w)| v + NUDGE * size * w).collect();
        let diff: Vec<f64> = moved.iter().zip(b).map(|(a, b)| a - b).collect();
        assert!((norm(&diff) / norm(b) - NUDGE).abs() < 1e-12);
    }

    #[test]
    fn the_companion_roots_are_followed_as_the_caption_says() {
        let mut scene = trial(4);
        assert!(scene.setup.kind == Companion && scene.setup.n == 5);
        assert!(same_set(&scene.now.eig, &scene.setup.roots.all(5), 1e-9));
        // a root dragged somewhere else: the eigenvalues go with it
        for (k, to) in [(0, DVec2::new(0.3, 0.2)), (1, DVec2::new(-1.2, 0.05)), (0, DVec2::new(1.4, 1.3))] {
            scene.setup.roots.pairs[k] = to;
            scene.refresh();
            assert!(same_set(&scene.now.eig, &scene.setup.roots.all(5), 1e-8));
        }
    }

    /// The scene opens at rest, every glide arrives and stops at any display's rate, and
    /// a change of n fades out and in.
    #[test]
    fn it_opens_at_rest_and_changes_glide_to_rest() {
        let scene = Patterns::new();
        assert!(!scene.moving());
        assert_eq!(scene.shown.shown, numbers(&scene.now));
        for hz in [30.0f32, 60.0, 144.0, 240.0] {
            for k in 0..TRIALS.len() {
                let mut scene = Patterns::new();
                scene.set_up(k);
                let mut frames = 0;
                while scene.moving() {
                    scene.advance(1.0 / hz, 1.0 / hz);
                    frames += 1;
                    assert!(frames as f32 / hz < 4.0, "experiment {k} still moving at {hz} Hz");
                }
                assert_eq!(scene.shown.shown, numbers(&scene.now));
            }
        }
        let mut scene = Patterns::new();
        scene.resize(7);
        assert!(scene.leaving.is_some() && scene.setup.n == 7);
        scene.choose(Markov);
        assert!(scene.old_promise.is_some());
    }

    /// Whatever is pressed, in whatever order, with frames of any length, everything
    /// drawn from stays finite and every word is made.
    #[test]
    fn any_input_leaves_everything_finite() {
        let mut rng = Rng::new(4);
        let mut scene = Patterns::new();
        let kinds: Vec<Kind> = all_kinds().collect();
        assert_eq!(kinds.len(), 19);
        for step in 0..3000 {
            match rng.below(12) {
                0 => scene.choose(kinds[rng.below(kinds.len())]),
                1 => scene.resize(rng.below(12)),
                2 => scene.another(),
                3 => scene.set_mode(if rng.bool(0.5) { Mode::Multiply } else { Mode::Solve }),
                4 => scene.set_nudge(rng.bool(0.5)),
                5 => scene.set_discs(rng.bool(0.5)),
                6 => scene.set_param(rng.below(10)),
                7 => scene.toggle_edge(rng.below(9), rng.below(9)),
                8 => scene.set_up(rng.below(TRIALS.len())),
                9 => {
                    let k = rng.below(MAX);
                    scene.x[k] = rng.range(-3.0, 3.0) as f64;
                    scene.b[k] = rng.range(-3.0, 3.0) as f64;
                    scene.refresh();
                }
                _ => {}
            }
            let dt = [0.0, 1.0 / 240.0, 1.0 / 60.0, 0.5, 5.0][rng.below(5)];
            scene.advance(dt, dt);
            assert!(scene.shown.shown.iter().all(|v| v.is_finite()), "{step}");
            let p = scene.pic(1.0);
            assert!(p.drawn.finite() && p.eig.iter().all(|z| z.is_finite()) && p.sv_log.iter().all(|v| v.is_finite()), "{step}");
            assert!(p.top.iter().chain(&p.bottom).all(|v| v.is_finite()) && p.top_range > 0.0 && p.bottom_range > 0.0);
            for line in words(&scene.setup, &scene.now) {
                assert!(!line.contains('?') && !line.contains("NaN") && !line.contains("inf"), "{line}");
            }
        }
    }

    /// Every kind at every n has words with no number missing.
    #[test]
    fn every_kind_has_its_words() {
        for kind in all_kinds() {
            for s in members(kind).into_iter().take(40) {
                let mut scene = Patterns::new();
                scene.take(s, Some(first_x(kind, s.n)));
                for line in words(&scene.setup, &scene.now) {
                    assert!(!line.is_empty() && !line.contains('?'), "{kind:?}: {line}");
                }
            }
        }
    }

    #[test]
    fn numbers_are_written_plainly() {
        assert_eq!(short(2.0), "2");
        assert_eq!(short(-1.0), "−1");
        assert_eq!(short(0.25), "0.25");
        assert_eq!(short(0.5), "0.5");
        assert_eq!(short(0.008), "0.008");
        assert_eq!(short(19.1957), "19.2");
        assert_eq!(short(4924.2), "4924");
        assert_eq!(short(1.5e7), r"1.5 \times 10^{7}");
        assert_eq!(short(-3.2e-4), r"-3.2 \times 10^{-4}");
        assert_eq!(short(f64::NAN), "?");
        assert_eq!(cell_text(Hilbert, 0.2, 1, 3), r"$\dfrac{1}{5}$");
        assert_eq!(nice(2.0), 2.0);
        assert_eq!(nice(2.45), 3.0);
        assert_eq!(nice(21.3), 30.0);
        assert_eq!(nice(0.75), 0.8);
    }
}

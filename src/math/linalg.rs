//! Small-matrix linear algebra used by the scenes: eigen-decompositions, SVD,
//! and a dense solver for least squares.

use glam::{Mat2, Mat3, Vec2, Vec3};

/// Eigenvalues of a 2x2 matrix.
#[derive(Clone, Copy, Debug)]
pub enum Eig2 {
    /// Two real eigenvalues (largest first) with unit eigenvectors.
    Real { l1: f32, l2: f32, v1: Vec2, v2: Vec2 },
    /// A complex-conjugate pair `re +- i im`.
    Complex { re: f32, im: f32 },
}

/// `size` is the size of the matrix's entries: what counts as zero is judged against it.
fn eigvec2(m: Mat2, l: f32, size: f32) -> Vec2 {
    // rows of (M - l I) are both orthogonal to the eigenvector; use the larger one
    let (a, b, c, d) = (m.x_axis.x - l, m.y_axis.x, m.x_axis.y, m.y_axis.y - l);
    let r1 = Vec2::new(a, b);
    let r2 = Vec2::new(c, d);
    let r = if r1.length_squared() >= r2.length_squared() { r1 } else { r2 };
    if r.length_squared() <= 1e-12 * size * size {
        Vec2::X
    } else {
        Vec2::new(-r.y, r.x).normalize()
    }
}

/// (Every threshold is relative to the size of the entries, so a matrix a thousand
/// times smaller has the same eigenvectors.)
pub fn eig2(m: Mat2) -> Eig2 {
    let size = m.to_cols_array().iter().fold(0.0f32, |s, v| s.max(v.abs()));
    let tr = m.x_axis.x + m.y_axis.y;
    let det = m.determinant();
    let disc = tr * tr * 0.25 - det;
    // tr²/4 and det can each be as large as size², and are rounded to the f32 steps at
    // that size: a repeated eigenvalue (disc = 0) comes out a few steps either side of
    // zero, so only a disc further below counts as a complex pair
    if disc < -4.0 * f32::EPSILON * size * size {
        return Eig2::Complex { re: tr * 0.5, im: (-disc).sqrt() };
    }
    let s = disc.max(0.0).sqrt();
    let (l1, l2) = (tr * 0.5 + s, tr * 0.5 - s);
    let v1 = eigvec2(m, l1, size);
    // a repeated eigenvalue of a scaled identity has every direction as eigenvector
    let v2 = if s <= 1e-6 * size && (m.y_axis.x.abs() + m.x_axis.y.abs()) <= 1e-6 * size { Vec2::Y } else { eigvec2(m, l2, size) };
    Eig2::Real { l1, l2, v1, v2 }
}

/// Singular value decomposition of a 2x2 matrix: `m = u * diag(s) * v^T`, with
/// `s.x >= s.y >= 0` and `u`, `v` orthogonal.
#[derive(Clone, Copy, Debug)]
pub struct Svd2 {
    pub u: Mat2,
    pub s: Vec2,
    pub v: Mat2,
}

pub fn svd2(m: Mat2) -> Svd2 {
    // eigen-decomposition of the symmetric matrix M^T M gives V and the squared singular values
    let mtm = m.transpose() * m;
    let (a, b, d) = (mtm.x_axis.x, mtm.y_axis.x, mtm.y_axis.y);
    let theta = 0.5 * (2.0 * b).atan2(a - d);
    let (sn, cs) = theta.sin_cos();
    let v = Mat2::from_cols(Vec2::new(cs, sn), Vec2::new(-sn, cs));
    let mv = m * v;
    let s1 = mv.x_axis.length();
    let s2 = mv.y_axis.length();
    let u1 = if s1 > 1e-9 { mv.x_axis / s1 } else { Vec2::X };
    let mut u2 = if s2 > 1e-9 { mv.y_axis / s2 } else { Vec2::new(-u1.y, u1.x) };
    if u1.dot(u2).abs() > 1e-3 {
        u2 = Vec2::new(-u1.y, u1.x);
    }
    Svd2 { u: Mat2::from_cols(u1, u2), s: Vec2::new(s1, s2), v }
}

/// Singular value decomposition of a 3x3 matrix: `m = u * diag(s) * v^T`, the
/// singular values largest first, `u` and `v` orthogonal whatever the rank of `m`.
pub fn svd3(m: Mat3) -> (Mat3, Vec3, Mat3) {
    let c = m.to_cols_array_2d();
    let a: Vec<f64> = (0..9).map(|k| c[k % 3][k / 3] as f64).collect();
    let (u, s, v) = svd(&a, 3, 3);
    let matrix = |q: &[f64]| Mat3::from_cols_array_2d(&[0, 1, 2].map(|k| [q[k] as f32, q[3 + k] as f32, q[6 + k] as f32]));
    // the entries are f32, good to about 1e-7 of the largest: a singular value below a
    // millionth of the largest is what rounding left of a zero
    let value = |k: usize| if s[k] < 1e-6 * s[0] { 0.0 } else { s[k] as f32 };
    (matrix(&u), Vec3::new(value(0), value(1), value(2)), matrix(&v))
}

/// Make column `j` of the `m` by `n` matrix `a` (row-major) a unit vector at right
/// angles to the columns listed in `others`, which must be orthonormal: the
/// coordinate direction that has most left once they are taken out of it.
fn complete_column(a: &mut [f64], m: usize, n: usize, j: usize, others: &[usize]) {
    let mut best = (0.0, vec![0.0; m]);
    for axis in 0..m {
        let mut e = vec![0.0; m];
        e[axis] = 1.0;
        for _ in 0..2 {
            for &k in others {
                let d: f64 = (0..m).map(|i| e[i] * a[i * n + k]).sum();
                (0..m).for_each(|i| e[i] -= d * a[i * n + k]);
            }
        }
        let len = e.iter().map(|x| x * x).sum::<f64>().sqrt();
        if len > best.0 {
            best = (len, e);
        }
    }
    for i in 0..m {
        a[i * n + j] = if best.0 > 0.0 { best.1[i] / best.0 } else { 0.0 };
    }
}

/// The singular value decomposition `A = U diag(s) Vᵀ` of an `m` by `n` matrix
/// (row-major, `m >= 1`, `n >= 1`), by one-sided Jacobi rotations: slow for big
/// matrices but accurate, and the matrices here are small. Returns `(u, s, v)` with
/// the singular values largest first; column `k` of U is `u[i * r + k]` for row `i`
/// (U is `m` by `r`, V is `n` by `r`, where `r = min(m, n)`). The columns of U and of
/// V are orthonormal whatever the rank: where a singular value is zero (to rounding),
/// its column is any direction that completes the set.
pub fn svd(a: &[f64], m: usize, n: usize) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    // work on the transpose if the matrix is wide, so that columns are the short side
    if m < n {
        let mut t = vec![0.0; m * n];
        for i in 0..m {
            for j in 0..n {
                t[j * m + i] = a[i * n + j];
            }
        }
        let (u, s, v) = svd(&t, n, m);
        return (v, s, u);
    }
    // columns of `w` are rotated until they are orthogonal: then w = U diag(s)
    let mut w = a[..m * n].to_vec();
    let mut v = vec![0.0; n * n];
    for i in 0..n {
        v[i * n + i] = 1.0;
    }
    for _sweep in 0..60 {
        let mut off = 0.0f64;
        for p in 0..n {
            for q in p + 1..n {
                let (mut alpha, mut beta, mut gamma) = (0.0, 0.0, 0.0);
                for i in 0..m {
                    let (x, y) = (w[i * n + p], w[i * n + q]);
                    alpha += x * x;
                    beta += y * y;
                    gamma += x * y;
                }
                if gamma.abs() <= 1e-15 * (alpha * beta).sqrt() || gamma == 0.0 {
                    continue;
                }
                off = off.max(gamma.abs() / (alpha * beta).sqrt().max(1e-300));
                let zeta = (beta - alpha) / (2.0 * gamma);
                let t = zeta.signum() / (zeta.abs() + (1.0 + zeta * zeta).sqrt());
                let c = 1.0 / (1.0 + t * t).sqrt();
                let sn = c * t;
                for i in 0..m {
                    let (x, y) = (w[i * n + p], w[i * n + q]);
                    w[i * n + p] = c * x - sn * y;
                    w[i * n + q] = sn * x + c * y;
                }
                for i in 0..n {
                    let (x, y) = (v[i * n + p], v[i * n + q]);
                    v[i * n + p] = c * x - sn * y;
                    v[i * n + q] = sn * x + c * y;
                }
            }
        }
        if off < 1e-13 {
            break;
        }
    }
    let norms: Vec<f64> = (0..n).map(|j| (0..m).map(|i| w[i * n + j] * w[i * n + j]).sum::<f64>().sqrt()).collect();
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&a, &b| norms[b].total_cmp(&norms[a]));
    let (mut u, mut s, mut vs) = (vec![0.0; m * n], vec![0.0; n], vec![0.0; n * n]);
    // a column this much shorter than the longest is rounding noise, not a direction
    let nothing = 1e-13 * norms[order[0]];
    let mut real = Vec::with_capacity(n);
    for (k, &j) in order.iter().enumerate() {
        for i in 0..n {
            vs[i * n + k] = v[i * n + j];
        }
        if norms[j] > nothing {
            s[k] = norms[j];
            (0..m).for_each(|i| u[i * n + k] = w[i * n + j] / norms[j]);
            real.push(k);
        }
    }
    for k in real.len()..n {
        complete_column(&mut u, m, n, k, &real);
        real.push(k);
    }
    (u, s, vs)
}

/// Orthonormalise the columns of the `m` by `n` matrix `a` (row-major, `m >= n`) in
/// place, by modified Gram-Schmidt run twice, and return the diagonal of R: how long
/// each column was once the earlier ones had been taken out of it. (This is the step
/// that Lyapunov exponents are computed with: the logarithms of these lengths, summed
/// over many steps.) A column that is a combination of the earlier ones has length
/// zero; it is replaced by a direction that completes the set, so that the result
/// has orthonormal columns in every case.
pub fn orthonormalise(a: &mut [f64], m: usize, n: usize) -> Vec<f64> {
    let mut r = vec![1.0; n];
    let length = |a: &[f64], j: usize| (0..m).map(|i| a[i * n + j] * a[i * n + j]).sum::<f64>().sqrt();
    let before: Vec<f64> = (0..n).map(|j| length(a, j)).collect();
    for pass in 0..2 {
        for j in 0..n {
            if r[j] == 0.0 {
                continue;
            }
            for k in 0..j {
                let d: f64 = (0..m).map(|i| a[i * n + j] * a[i * n + k]).sum();
                for i in 0..m {
                    a[i * n + j] -= d * a[i * n + k];
                }
            }
            let len = length(a, j);
            // what is left of a dependent column is rounding noise: no direction
            if pass == 0 && len <= 1e-12 * before[j] {
                r[j] = 0.0;
                let earlier: Vec<usize> = (0..j).collect();
                complete_column(a, m, n, j, &earlier);
                continue;
            }
            r[j] *= len;
            for i in 0..m {
                a[i * n + j] /= len;
            }
        }
    }
    r
}

/// Eigenvalues (largest first) and unit eigenvectors of the symmetric `n` by `n`
/// matrix `a` (row-major), by Jacobi rotations: plenty for the small matrices drawn
/// here. Eigenvector `k` is `vectors[k * n..(k + 1) * n]`.
pub fn eig_sym(a: &[f64], n: usize) -> (Vec<f64>, Vec<f64>) {
    let mut a = a[..n * n].to_vec();
    let mut v = vec![0.0; n * n];
    for i in 0..n {
        v[i * n + i] = 1.0;
    }
    for _sweep in 0..64 {
        let off: f64 = (0..n).flat_map(|p| (0..n).filter(move |&q| q != p).map(move |q| (p, q))).map(|(p, q)| a[p * n + q] * a[p * n + q]).sum();
        let diagonal: f64 = (0..n).map(|i| a[i * n + i] * a[i * n + i]).sum();
        if off <= 1e-26 * (diagonal + f64::MIN_POSITIVE) {
            break;
        }
        for p in 0..n {
            for q in p + 1..n {
                let apq = a[p * n + q];
                if apq.abs() < 1e-300 {
                    continue;
                }
                // the rotation in the (p, q) plane that zeroes a[p][q]
                let theta = (a[q * n + q] - a[p * n + p]) / (2.0 * apq);
                let t = theta.signum() / (theta.abs() + (theta * theta + 1.0).sqrt());
                let c = 1.0 / (t * t + 1.0).sqrt();
                let s = t * c;
                for k in 0..n {
                    let (x, y) = (a[k * n + p], a[k * n + q]);
                    a[k * n + p] = c * x - s * y;
                    a[k * n + q] = s * x + c * y;
                }
                for k in 0..n {
                    let (x, y) = (a[p * n + k], a[q * n + k]);
                    a[p * n + k] = c * x - s * y;
                    a[q * n + k] = s * x + c * y;
                }
                for k in 0..n {
                    let (x, y) = (v[k * n + p], v[k * n + q]);
                    v[k * n + p] = c * x - s * y;
                    v[k * n + q] = s * x + c * y;
                }
            }
        }
    }
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&i, &j| a[j * n + j].total_cmp(&a[i * n + i]));
    let values = order.iter().map(|&i| a[i * n + i]).collect();
    let vectors = order.iter().flat_map(|&i| (0..n).map(move |k| (k, i))).map(|(k, i)| v[k * n + i]).collect();
    (values, vectors)
}

/// The real eigenvalues of a general 3x3 matrix with a unit eigenvector for each.
pub fn eig3_real(m: Mat3) -> Vec<(f32, Vec3)> {
    // characteristic polynomial: l^3 - tr l^2 + c1 l - det = 0, its coefficients
    // worked out in f64 (in f32 their rounding errors decide the number of roots)
    let a = m.to_cols_array_2d().map(|col| col.map(|v| v as f64));
    let tr = a[0][0] + a[1][1] + a[2][2];
    let c1 = a[0][0] * a[1][1] - a[1][0] * a[0][1] + a[0][0] * a[2][2] - a[2][0] * a[0][2] + a[1][1] * a[2][2] - a[2][1] * a[1][2];
    let det = a[0][0] * (a[1][1] * a[2][2] - a[2][1] * a[1][2]) - a[1][0] * (a[0][1] * a[2][2] - a[2][1] * a[0][2])
        + a[2][0] * (a[0][1] * a[1][2] - a[1][1] * a[0][2]);
    // depressed cubic t^3 + p t + q with l = t + tr/3
    let p = c1 - tr * tr / 3.0;
    let q = -2.0 * tr * tr * tr / 27.0 + tr * c1 / 3.0 - det;
    let disc = q * q / 4.0 + p * p * p / 27.0;
    let mut roots: Vec<f64> = Vec::new();
    // the discriminant is a difference of these two terms: it counts as positive
    // (one real root) only if it is not lost in their rounding
    let size = q * q / 4.0 + (p * p * p).abs() / 27.0;
    let scale = tr.abs() + c1.abs().sqrt() + det.abs().cbrt();
    if disc > 1e-9 * size && size > 1e-30 * scale.powi(6) {
        let s = disc.sqrt();
        roots.push((-q / 2.0 + s).cbrt() + (-q / 2.0 - s).cbrt());
    } else if p.abs() <= 1e-10 * scale * scale {
        roots.push(0.0);
    } else {
        let r = (-p / 3.0).max(0.0).sqrt();
        let arg = (3.0 * q / (2.0 * p * r.max(1e-300))).clamp(-1.0, 1.0);
        let phi = arg.acos() / 3.0;
        for k in 0..3 {
            roots.push(2.0 * r * (phi - 2.0 * std::f64::consts::PI * k as f64 / 3.0).cos());
        }
    }
    // From here every threshold is relative to the size of the entries, as in eig2, so
    // that a matrix a thousand times smaller has the same eigenvectors. (Not to the size
    // of the eigenvalues: a matrix can have large entries and only small eigenvalues,
    // and the rows below would then have rounding noise taken for a direction.)
    let entries = m.to_cols_array().iter().fold(0.0f32, |s, v| s.max(v.abs()));
    let mut out: Vec<(f32, Vec3)> = Vec::new();
    for t in roots {
        // the entries are f32, good to about 1e-7 of the largest: an eigenvalue below a
        // millionth of it is what rounding has left of a zero
        let l = (t + tr / 3.0) as f32;
        let l = if l.abs() < 1e-6 * entries { 0.0 } else { l };
        if out.iter().any(|(o, _)| (o - l).abs() < 1e-4 * entries) {
            continue;
        }
        // null vector of (M - l I): the largest cross product of two of its rows
        let b = m - Mat3::from_diagonal(Vec3::splat(l));
        let r = [b.row(0), b.row(1), b.row(2)];
        let cands = [r[0].cross(r[1]), r[0].cross(r[2]), r[1].cross(r[2])];
        let best = cands.iter().copied().max_by(|x, y| x.length_squared().total_cmp(&y.length_squared())).unwrap();
        if best.length_squared() > 1e-12 * entries.powi(4) {
            out.push((l, best.normalize()));
        } else {
            // a repeated eigenvalue with a whole plane of eigenvectors: pick one in it
            let row = r.iter().copied().max_by(|x, y| x.length_squared().total_cmp(&y.length_squared())).unwrap();
            let v = if row.length_squared() > 1e-12 * entries * entries { row.normalize().any_orthonormal_vector() } else { Vec3::X };
            out.push((l, v));
        }
    }
    out.sort_by(|a, b| b.0.total_cmp(&a.0));
    out
}

/// Solve `A x = b` for a dense `n x n` system (row-major `a`) by Gaussian
/// elimination with partial pivoting. Returns None if the matrix is singular.
pub fn solve(a: &mut [f64], b: &mut [f64], n: usize) -> Option<Vec<f64>> {
    // (singular means a pivot that is nothing beside the entries, whatever their size)
    let size = a[..n * n].iter().fold(0.0f64, |s, v| s.max(v.abs()));
    for col in 0..n {
        let mut piv = col;
        for r in col + 1..n {
            if a[r * n + col].abs() > a[piv * n + col].abs() {
                piv = r;
            }
        }
        if a[piv * n + col].abs() <= 1e-13 * size {
            return None;
        }
        if piv != col {
            for k in 0..n {
                a.swap(col * n + k, piv * n + k);
            }
            b.swap(col, piv);
        }
        let d = a[col * n + col];
        for r in col + 1..n {
            let f = a[r * n + col] / d;
            if f != 0.0 {
                for k in col..n {
                    a[r * n + k] -= f * a[col * n + k];
                }
                b[r] -= f * b[col];
            }
        }
    }
    let mut x = vec![0.0; n];
    for r in (0..n).rev() {
        let mut acc = b[r];
        for k in r + 1..n {
            acc -= a[r * n + k] * x[k];
        }
        x[r] = acc / a[r * n + r];
    }
    Some(x)
}

/// Least-squares polynomial fit of degree `deg` with ridge penalty `lambda`.
/// Coefficients are returned lowest power first, in the variable `(x - cx) / sx`
/// (centred and scaled for conditioning); use [`poly_eval`] with the same pair.
pub fn poly_fit(pts: &[Vec2], deg: usize, lambda: f64) -> Option<(Vec<f64>, f64, f64)> {
    if pts.is_empty() {
        return None;
    }
    let n = deg + 1;
    let (lo, hi) = pts.iter().fold((f32::MAX, f32::MIN), |(lo, hi), p| (lo.min(p.x), hi.max(p.x)));
    let cx = ((lo + hi) * 0.5) as f64;
    let sx = (((hi - lo) * 0.5) as f64).max(1e-6);
    let mut ata = vec![0.0f64; n * n];
    let mut atb = vec![0.0f64; n];
    let mut pow = vec![0.0f64; 2 * n];
    for p in pts {
        let x = (p.x as f64 - cx) / sx;
        pow[0] = 1.0;
        for k in 1..2 * n {
            pow[k] = pow[k - 1] * x;
        }
        for i in 0..n {
            atb[i] += pow[i] * p.y as f64;
            for j in 0..n {
                ata[i * n + j] += pow[i + j];
            }
        }
    }
    for i in 0..n {
        // do not penalise the intercept
        ata[i * n + i] += if i == 0 { 1e-12 } else { lambda + 1e-12 };
    }
    solve(&mut ata, &mut atb, n).map(|c| (c, cx, sx))
}

pub fn poly_eval(c: &[f64], cx: f64, sx: f64, x: f32) -> f32 {
    let t = (x as f64 - cx) / sx;
    c.iter().rev().fold(0.0, |acc, &k| acc * t + k) as f32
}

// ----- square matrices of any size, as flat slices -----------------------------------

/// The `n` by `n` identity (row-major, like every matrix in the functions below).
pub fn identity(n: usize) -> Vec<f64> {
    let mut m = vec![0.0; n * n];
    for i in 0..n {
        m[i * n + i] = 1.0;
    }
    m
}

/// `out = a b` for `n` by `n` matrices.
pub fn matmul(a: &[f64], b: &[f64], out: &mut [f64], n: usize) {
    out.fill(0.0);
    for i in 0..n {
        let row = &mut out[i * n..(i + 1) * n];
        for j in 0..n {
            let k = a[i * n + j];
            for (o, &x) in row.iter_mut().zip(&b[j * n..(j + 1) * n]) {
                *o += k * x;
            }
        }
    }
}

/// `out = aᵀ b`.
pub fn matmul_transposed(a: &[f64], b: &[f64], out: &mut [f64], n: usize) {
    out.fill(0.0);
    for i in 0..n {
        let from = &b[i * n..(i + 1) * n];
        for j in 0..n {
            let k = a[i * n + j];
            for (o, &x) in out[j * n..(j + 1) * n].iter_mut().zip(from) {
                *o += k * x;
            }
        }
    }
}

/// `out = a x`.
pub fn matvec(a: &[f64], x: &[f64], out: &mut [f64], n: usize) {
    for i in 0..n {
        out[i] = a[i * n..(i + 1) * n].iter().zip(x).map(|(a, x)| a * x).sum();
    }
}

/// The length of a vector.
pub fn norm(v: &[f64]) -> f64 {
    v.iter().map(|x| x * x).sum::<f64>().sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn eig_sym_decomposes_a_symmetric_matrix() {
        let mut rng = crate::math::rng::Rng::new(5);
        for n in [1usize, 2, 5, 9, 16] {
            let b: Vec<f64> = (0..n * n).map(|_| rng.normal() as f64).collect();
            // b + b' is symmetric
            let a: Vec<f64> = (0..n * n).map(|k| b[k] + b[(k % n) * n + k / n]).collect();
            let (values, vectors) = eig_sym(&a, n);
            for k in 0..n {
                let v = &vectors[k * n..(k + 1) * n];
                if k > 0 {
                    assert!(values[k] <= values[k - 1]);
                }
                for i in 0..n {
                    let av: f64 = (0..n).map(|j| a[i * n + j] * v[j]).sum();
                    assert!((av - values[k] * v[i]).abs() < 1e-9, "n={n}: A v != lambda v");
                }
                for m in 0..=k {
                    let dot: f64 = v.iter().zip(&vectors[m * n..(m + 1) * n]).map(|(x, y)| x * y).sum();
                    assert!((dot - if m == k { 1.0 } else { 0.0 }).abs() < 1e-9);
                }
            }
        }
    }

    #[test]
    fn eig2_real_and_complex() {
        let m = Mat2::from_cols(Vec2::new(2.0, 1.0), Vec2::new(1.0, 2.0));
        match eig2(m) {
            Eig2::Real { l1, l2, v1, v2 } => {
                assert!((l1 - 3.0).abs() < 1e-5 && (l2 - 1.0).abs() < 1e-5);
                assert!((m * v1 - v1 * l1).length() < 1e-4);
                assert!((m * v2 - v2 * l2).length() < 1e-4);
            }
            _ => panic!("expected real eigenvalues"),
        }
        let rot = Mat2::from_angle(0.7);
        match eig2(rot) {
            Eig2::Complex { re, im } => assert!((re - 0.7f32.cos()).abs() < 1e-5 && (im - 0.7f32.sin()).abs() < 1e-5),
            _ => panic!("expected complex eigenvalues"),
        }
    }

    #[test]
    fn svd2_reconstructs() {
        for m in [
            Mat2::from_cols(Vec2::new(1.5, 0.3), Vec2::new(-0.8, 0.9)),
            Mat2::from_cols(Vec2::new(1.0, 2.0), Vec2::new(2.0, 4.0)),
            Mat2::from_cols(Vec2::new(0.0, 1.0), Vec2::new(-1.0, 0.0)),
        ] {
            let d = svd2(m);
            let r = d.u * Mat2::from_diagonal(d.s) * d.v.transpose();
            assert!((r.x_axis - m.x_axis).length() + (r.y_axis - m.y_axis).length() < 1e-4, "{m:?}");
            assert!(d.s.x >= d.s.y && d.s.y >= 0.0);
            assert!((d.u.transpose() * d.u - Mat2::IDENTITY).x_axis.length() < 1e-3);
        }
    }

    #[test]
    fn svd3_rebuilds_the_matrix() {
        let a = Mat3::from_cols(Vec3::new(1.0, 0.2, -0.4), Vec3::new(0.7, 1.4, 0.1), Vec3::new(-0.3, 0.6, 0.8));
        let (u, s, v) = svd3(a);
        let r = u * Mat3::from_diagonal(s) * v.transpose();
        let err: f32 = (0..3).map(|k| (r.col(k) - a.col(k)).length()).sum();
        assert!(err < 1e-5, "svd3 error {err}");
    }

    #[test]
    fn eig3_real_finds_eigenpairs() {
        let a = Mat3::from_cols(Vec3::new(2.0, 0.0, 0.0), Vec3::new(1.0, 3.0, 0.0), Vec3::new(0.5, -1.0, -1.0));
        let e = eig3_real(a);
        assert_eq!(e.len(), 3);
        for (l, v) in e {
            assert!((a * v - v * l).length() < 1e-3, "l = {l}");
        }
        // a rotation about z has one real eigenvalue, 1, with eigenvector z
        let r = Mat3::from_rotation_z(0.9);
        let e = eig3_real(r);
        assert_eq!(e.len(), 1);
        assert!((e[0].0 - 1.0).abs() < 1e-4 && e[0].1.z.abs() > 0.999);
    }

    #[test]
    fn poly_fit_recovers_a_cubic() {
        let f = |x: f32| 0.5 * x * x * x - 2.0 * x + 1.0;
        let pts: Vec<Vec2> = (0..20).map(|i| i as f32 * 0.3 - 3.0).map(|x| Vec2::new(x, f(x))).collect();
        let (c, cx, sx) = poly_fit(&pts, 3, 0.0).unwrap();
        for x in [-2.5f32, 0.0, 1.7] {
            assert!((poly_eval(&c, cx, sx, x) - f(x)).abs() < 1e-3);
        }
    }

    #[test]
    fn svd_rebuilds_the_matrix_and_orders_its_values() {
        let mut rng = crate::math::rng::Rng::new(5);
        for (m, n) in [(5usize, 3usize), (3, 5), (6, 6), (1, 4), (4, 1)] {
            let a: Vec<f64> = (0..m * n).map(|_| rng.normal() as f64).collect();
            let (u, s, v) = svd(&a, m, n);
            let r = m.min(n);
            assert!(s.windows(2).all(|w| w[0] >= w[1]) && s.iter().all(|&x| x >= 0.0));
            for i in 0..m {
                for j in 0..n {
                    let back: f64 = (0..r).map(|k| u[i * r + k] * s[k] * v[j * r + k]).sum();
                    assert!((back - a[i * n + j]).abs() < 1e-9, "{m}x{n} entry ({i},{j}): {back} vs {}", a[i * n + j]);
                }
            }
            for k in 0..r {
                for l in 0..r {
                    let uu: f64 = (0..m).map(|i| u[i * r + k] * u[i * r + l]).sum();
                    let vv: f64 = (0..n).map(|i| v[i * r + k] * v[i * r + l]).sum();
                    let want = if k == l { 1.0 } else { 0.0 };
                    assert!((uu - want).abs() < 1e-9 && (vv - want).abs() < 1e-9);
                }
            }
        }
    }

    #[test]
    fn orthonormalise_returns_the_lengths_whose_product_is_the_volume() {
        // for a square matrix the product of R's diagonal is |det|
        let mut a = vec![2.0, 1.0, 0.0, 0.0, 3.0, 1.0, 1.0, 0.0, 1.0];
        let r = orthonormalise(&mut a, 3, 3);
        let det = 2.0 * (3.0 - 0.0) - 1.0 * (0.0 - 1.0);
        assert!((r.iter().product::<f64>() - det).abs() < 1e-9, "{r:?}");
        for k in 0..3 {
            for l in 0..3 {
                let d: f64 = (0..3).map(|i| a[i * 3 + k] * a[i * 3 + l]).sum();
                assert!((d - if k == l { 1.0 } else { 0.0 }).abs() < 1e-12);
            }
        }
    }

    /// A rotation has exactly one real eigenvalue, 1, along its axis; the test for
    /// "one real root" used to be loose enough to return 0.96 for a turn of 20 degrees.
    #[test]
    fn a_rotation_has_one_real_eigenvalue() {
        for degrees in [1.0f32, 10.0, 20.0, 90.0, 179.0] {
            for axis in [Vec3::Z, Vec3::new(1.0, 1.0, 1.0).normalize(), Vec3::new(0.3, -0.5, 0.8).normalize()] {
                let found = eig3_real(Mat3::from_axis_angle(axis, degrees.to_radians()));
                assert_eq!(found.len(), 1, "{degrees} degrees about {axis}: {found:?}");
                assert!((found[0].0 - 1.0).abs() < 1e-5 && found[0].1.dot(axis).abs() > 0.9999, "{degrees} degrees about {axis}: {found:?}");
            }
        }
        // eigenvalues 1 and 0.5 +- 0.05 i: still only the 1 is real
        let block = Mat3::from_cols(Vec3::new(1.0, 0.0, 0.0), Vec3::new(0.0, 0.5, 0.05), Vec3::new(0.0, -0.05, 0.5));
        let found = eig3_real(block);
        assert!(found.len() == 1 && (found[0].0 - 1.0).abs() < 1e-5, "{found:?}");
        // and three real ones are all found
        let found = eig3_real(Mat3::from_cols(Vec3::new(2.0, 0.0, 0.0), Vec3::new(1.0, -1.0, 0.0), Vec3::new(0.0, 3.0, 0.5)));
        let mut values: Vec<f32> = found.iter().map(|e| e.0).collect();
        values.sort_by(f32::total_cmp);
        assert!(values.len() == 3 && (values[0] + 1.0).abs() < 1e-5 && (values[1] - 0.5).abs() < 1e-5 && (values[2] - 2.0).abs() < 1e-5, "{values:?}");
    }

    #[test]
    fn svd3_is_orthogonal_whatever_the_rank() {
        let orthogonal = |q: Mat3| (q.transpose() * q - Mat3::IDENTITY).to_cols_array().iter().fold(0.0f32, |m, v| m.max(v.abs()));
        let cases = [
            Mat3::from_cols(Vec3::new(1.0, 4.0, 7.0), Vec3::new(2.0, 5.0, 8.0), Vec3::new(3.0, 6.0, 9.0)), // rank 2
            Mat3::from_cols(Vec3::new(1.0, 2.0, 3.0), Vec3::new(2.0, 4.0, 6.0), Vec3::new(-1.0, -2.0, -3.0)), // rank 1
            Mat3::ZERO,
            Mat3::from_cols(Vec3::new(1.2, 0.3, 0.0), Vec3::new(-0.4, 1.0, 0.3), Vec3::new(0.2, 0.2, 1.4)),
        ];
        for m in cases {
            let (u, s, v) = svd3(m);
            assert!(orthogonal(u) < 1e-5 && orthogonal(v) < 1e-5, "{m:?}: U off by {}, V by {}", orthogonal(u), orthogonal(v));
            assert!(s.x >= s.y && s.y >= s.z && s.z >= 0.0);
            let back = u * Mat3::from_diagonal(s) * v.transpose();
            let error = (back - m).to_cols_array().iter().fold(0.0f32, |e, x| e.max(x.abs()));
            assert!(error < 1e-5 * (1.0 + s.x), "{m:?} rebuilt with error {error}");
        }
        // the zero singular value of a rank-2 matrix is zero, not a thousandth of the largest
        assert!(svd3(cases[0]).1.z < 1e-5);
    }

    #[test]
    fn dependent_columns_and_scale() {
        // the second column is twice the first: its length in R is zero, and the
        // result still has orthonormal columns
        let mut a = vec![1.0, 2.0, 0.0, 2.0, 4.0, 0.0, 3.0, 6.0, 1.0];
        let r = orthonormalise(&mut a, 3, 3);
        assert!(r[0] > 3.0 && r[1] == 0.0 && r[2] > 0.0, "{r:?}");
        for k in 0..3 {
            for l in 0..3 {
                let d: f64 = (0..3).map(|i| a[i * 3 + k] * a[i * 3 + l]).sum();
                assert!((d - if k == l { 1.0 } else { 0.0 }).abs() < 1e-12, "columns {k}, {l}: {d}");
            }
        }
        // solving does not depend on the units the system is written in
        for scale in [1e-14, 1.0, 1e9] {
            let (mut m, mut b) = (vec![2.0 * scale, 1.0 * scale, 1.0 * scale, 3.0 * scale], vec![3.0 * scale, 4.0 * scale]);
            let x = solve(&mut m, &mut b, 2).expect("a regular system");
            assert!((x[0] - 1.0).abs() < 1e-9 && (x[1] - 1.0).abs() < 1e-9, "scale {scale}: {x:?}");
            let (mut m, mut b) = ([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0].map(|v| v * scale).to_vec(), vec![scale; 3]);
            assert!(solve(&mut m, &mut b, 3).is_none(), "scale {scale}: a singular system");
        }
        // nor do eigenvectors: a small rotation is still a rotation, a small shear still has its direction
        assert!(matches!(eig2(Mat2::from_angle(0.3) * 1e-4), Eig2::Complex { .. }));
        match eig2(Mat2::from_cols(Vec2::new(2e-4, 0.0), Vec2::new(1e-4, 1e-4))) {
            Eig2::Real { l1, l2, v1, v2 } => assert!(
                (l1 - 2e-4).abs() < 1e-9 && (l2 - 1e-4).abs() < 1e-9 && v1.x.abs() > 0.999 && (v2.x.abs() - v2.y.abs()).abs() < 1e-3,
                "{l1} {l2} {v1} {v2}"
            ),
            other => panic!("{other:?}"),
        }
    }

    /// A matrix with a repeated eigenvalue and one eigenvector direction (as from
    /// typing entries to two decimals) was often called complex: tr²/4 - det came out a
    /// rounding error below zero. Every such matrix with entries in steps of 0.1 in
    /// [-3, 3], and the same matrices nudged to a complex pair.
    #[test]
    fn a_repeated_eigenvalue_with_one_direction_is_real() {
        let (mut count, mut worst_value, mut worst_dot) = (0, 0.0f32, 1.0f32);
        for a in -30i32..=30 {
            for d in -30i32..=30 {
                let half = (a - d) / 2;
                if (a - d) % 2 != 0 || half == 0 {
                    continue;
                }
                for b in (-30i32..=30).filter(|&b| b != 0 && (half * half) % b == 0) {
                    // ((a - d)/2)² + b c = 0: the eigenvalue (a + d)/2 twice
                    let c = -half * half / b;
                    if c.abs() > 30 {
                        continue;
                    }
                    let [a, b, c, d] = [a, b, c, d].map(|v| v as f32 / 10.0);
                    let size = a.abs().max(b.abs()).max(c.abs()).max(d.abs());
                    let m = Mat2::from_cols(Vec2::new(a, c), Vec2::new(b, d));
                    let (l, along) = ((a + d) * 0.5, Vec2::new(b, -(a - d) * 0.5).normalize());
                    match eig2(m) {
                        Eig2::Real { l1, l2, v1, v2 } => {
                            worst_value = worst_value.max((l1 - l).abs().max((l2 - l).abs()) / size);
                            worst_dot = worst_dot.min(v1.dot(along).abs().min(v2.dot(along).abs()));
                        }
                        other => panic!("[[{a}, {b}], [{c}, {d}]]: {other:?}"),
                    }
                    // a hundredth less of b c makes the pair complex, and it is still seen as one
                    let nudged = Mat2::from_cols(Vec2::new(a, c - 0.01 * b.signum()), Vec2::new(b, d));
                    assert!(matches!(eig2(nudged), Eig2::Complex { .. }), "[[{a}, {b}], [{c}, {d}]] nudged: {:?}", eig2(nudged));
                    count += 1;
                }
            }
        }
        assert_eq!(count, 10056);
        // (the eigenvalues are split by at most the square root of the rounding error)
        assert!(worst_value < 5e-4 && worst_dot > 0.9999, "eigenvalues off by {worst_value} of the size, eigenvectors by a cosine of {worst_dot}");
        // the one a visitor found: -0.1 twice, along (1, -1)
        match eig2(Mat2::from_cols(Vec2::new(-0.5, 0.4), Vec2::new(-0.4, 0.3))) {
            Eig2::Real { l1, l2, v1, .. } => assert!((l1 + 0.1).abs() < 1e-6 && (l2 + 0.1).abs() < 1e-6 && (v1.x + v1.y).abs() < 1e-6, "{l1} {l2} {v1}"),
            other => panic!("{other:?}"),
        }
    }

    /// The input is f32: what rounding leaves of a zero eigenvalue or singular value
    /// (1e-16, 4e-8) is shown as 0, but a value that is merely small is kept.
    #[test]
    fn rounding_noise_is_zero() {
        let rank2 = Mat3::from_cols(Vec3::new(1.0, 0.0, 0.5), Vec3::new(0.0, 1.0, 0.5), Vec3::new(1.0, 1.0, 1.0));
        let rank1 = Mat3::from_cols(Vec3::new(1.0, 0.5, 0.8), Vec3::new(0.6, 0.3, 0.48), Vec3::new(-0.8, -0.4, -0.64));
        let values = |m: Mat3| eig3_real(m).iter().map(|e| e.0).collect::<Vec<f32>>();
        let v = values(rank2);
        assert!(v.len() == 3 && (v[0] - 2.0).abs() < 1e-6 && (v[1] - 1.0).abs() < 1e-6 && v[2] == 0.0, "{v:?}");
        let v = values(rank1);
        assert!(v.len() == 2 && (v[0] - 0.66).abs() < 1e-6 && v[1] == 0.0, "{v:?}");
        for m in [rank2, rank1] {
            // and the eigenvector of the zero is one that the matrix sends to nothing
            let null = eig3_real(m).last().map_or(Vec3::ZERO, |e| e.1);
            assert!((null.length() - 1.0).abs() < 1e-6 && (m * null).length() < 1e-6, "{m:?}: {null}");
        }
        let s = svd3(rank1).1;
        assert!((s.x - 1.944_222).abs() < 1e-5 && s.y == 0.0 && s.z == 0.0, "{s}");
        assert_eq!(svd3(rank2).1.z, 0.0);
        let small = Mat3::from_diagonal(Vec3::new(1.0, 0.5, 2e-6));
        let (v, s) = (values(small), svd3(small).1);
        assert!(v.len() == 3 && (v[2] - 2e-6).abs() < 1e-11 && (s.z - 2e-6).abs() < 1e-11, "{v:?} {s}");
    }

    /// Eigenpairs do not depend on the units: the same matrix at a ten-thousandth, a
    /// thousandth, once and a thousand times. (With fixed thresholds the smallest lost
    /// an eigenvalue as a "duplicate" and paired another with the wrong vector.)
    #[test]
    fn eig3_real_does_not_depend_on_the_units() {
        let triangular = Mat3::from_cols(Vec3::new(2.0, 0.0, 0.0), Vec3::new(1.0, 3.0, 0.0), Vec3::new(0.5, -1.0, -1.0));
        let pairs = [(3.0, Vec3::new(1.0, 1.0, 0.0)), (2.0, Vec3::X), (-1.0, Vec3::new(-0.25, 0.25, 1.0))];
        let diagonal = Mat3::from_diagonal(Vec3::new(1.0, 2.0, 3.0));
        let diagonal_pairs = [(3.0, Vec3::Z), (2.0, Vec3::Y), (1.0, Vec3::X)];
        for (m, want) in [(triangular, pairs), (diagonal, diagonal_pairs)] {
            for scale in [1e-4f32, 1e-3, 1.0, 1e3] {
                let found = eig3_real(m * scale);
                assert_eq!(found.len(), 3, "{scale}: {found:?}");
                for ((l, v), (wl, wv)) in found.iter().zip(want) {
                    assert!((l / scale - wl).abs() < 1e-5 && v.dot(wv.normalize()).abs() > 0.99999, "{scale}: {found:?}");
                }
            }
        }
        // Entries of 1 and eigenvalues of nearly 0 (u vᵀ with u ⟂ v, whose entries f32
        // cannot hold exactly): judged against the eigenvalues, rounding noise in the
        // rows was taken for a direction, and (1, 0, 0) given as an eigenvector of 0.
        let m = Mat3::from_cols(Vec3::new(0.3, 0.09, 0.0), Vec3::new(-1.0, -0.3, 0.0), Vec3::new(0.2, 0.06, 0.0));
        for scale in [1e-3f32, 1.0] {
            let m = m * scale;
            for (l, v) in eig3_real(m) {
                assert!(l.abs() < 1e-4 * scale && (m * v - v * l).length() < 1e-6 * scale, "{scale}: {l} {v}");
            }
        }
    }
}

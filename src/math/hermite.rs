//! Probabilists' Hermite polynomials and integration against the standard normal.
//!
//! `He_n` is the monic family with `He_{n+1} = x He_n - n He_{n-1}`; `h_n = He_n / sqrt(n!)`
//! is the orthonormal one: `E[h_m(X) h_n(X)] = [m = n]` for `X ~ N(0, 1)`. Everything
//! here is `f64`: the polynomials grow fast and the Gaussian tails are tiny, and their
//! products are what the scenes integrate.

use std::f64::consts::PI;

/// Standard normal density.
#[inline]
pub fn phi(x: f64) -> f64 {
    (-0.5 * x * x).exp() / (2.0 * PI).sqrt()
}

/// `He_n(x)`.
pub fn he(n: usize, x: f64) -> f64 {
    let (mut a, mut b) = (1.0, x);
    if n == 0 {
        return a;
    }
    for k in 1..n {
        let c = x * b - k as f64 * a;
        a = b;
        b = c;
    }
    b
}

/// Orthonormal `h_0(x) ..= h_nmax(x)` into `out` (which sets `nmax = out.len() - 1`).
/// Uses the normalised recurrence `sqrt(n+1) h_{n+1} = x h_n - sqrt(n) h_{n-1}`,
/// which stays well scaled where the raw polynomials would overflow.
#[inline]
pub fn h_all(x: f64, out: &mut [f64]) {
    if out.is_empty() {
        return;
    }
    out[0] = 1.0;
    if out.len() > 1 {
        out[1] = x;
    }
    for n in 1..out.len() - 1 {
        out[n + 1] = (x * out[n] - (n as f64).sqrt() * out[n - 1]) / ((n + 1) as f64).sqrt();
    }
}

/// Orthonormal `h_n(x)`.
pub fn h(n: usize, x: f64) -> f64 {
    let (mut a, mut b) = (1.0, x);
    if n == 0 {
        return a;
    }
    for k in 1..n {
        let c = (x * b - (k as f64).sqrt() * a) / ((k + 1) as f64).sqrt();
        a = b;
        b = c;
    }
    b
}

/// Integer coefficients of `He_n`, lowest power first. Exact up to n = 30 or so.
pub fn he_coeffs(n: usize) -> Vec<i64> {
    let mut a = vec![1i64];
    if n == 0 {
        return a;
    }
    let mut b = vec![0i64, 1];
    for k in 1..n {
        // c = x b - k a
        let mut c = vec![0i64; b.len() + 1];
        for (i, &v) in b.iter().enumerate() {
            c[i + 1] += v;
        }
        for (i, &v) in a.iter().enumerate() {
            c[i] -= k as i64 * v;
        }
        a = b;
        b = c;
    }
    b
}

/// `He_n(x)` written out, e.g. `x⁴ − 6x² + 3`.
pub fn he_formula(n: usize) -> String {
    const SUP: [&str; 10] = ["⁰", "¹", "²", "³", "⁴", "⁵", "⁶", "⁷", "⁸", "⁹"];
    let sup = |k: usize| -> String { k.to_string().chars().map(|c| SUP[c as usize - '0' as usize]).collect() };
    let c = he_coeffs(n);
    let mut s = String::new();
    for k in (0..=n).rev() {
        let v = c[k];
        if v == 0 {
            continue;
        }
        if s.is_empty() {
            if v < 0 {
                s.push('−');
            }
        } else {
            s.push_str(if v < 0 { " − " } else { " + " });
        }
        let a = v.unsigned_abs();
        if a != 1 || k == 0 {
            s.push_str(&a.to_string());
        }
        match k {
            0 => {}
            1 => s.push('x'),
            _ => {
                s.push('x');
                s.push_str(&sup(k));
            }
        }
    }
    s
}

const GRID_L: f64 = 12.0;
const GRID_M: usize = 12_000;

/// `E[f(X)]` for `X ~ N(0, 1)`, by the trapezoid rule on a fine grid. This handles
/// kinks and jumps (ReLU, sign) gracefully, which a Gauss-Hermite rule does not.
pub fn gauss_mean(mut f: impl FnMut(f64) -> f64) -> f64 {
    let step = 2.0 * GRID_L / GRID_M as f64;
    let mut acc = 0.0;
    for i in 0..=GRID_M {
        let x = -GRID_L + step * i as f64;
        let v = f(x) * phi(x);
        if v.is_finite() {
            acc += if i == 0 || i == GRID_M { 0.5 * v } else { v };
        }
    }
    acc * step
}

/// Hermite coefficients `c_n = E[f(X) h_n(X)]` for `n = 0 ..= nmax`. Then
/// `f = sum c_n h_n` in the Gaussian mean-square sense, and `E[f²] = sum c_n²`.
pub fn coefficients(mut f: impl FnMut(f64) -> f64, nmax: usize) -> Vec<f64> {
    let step = 2.0 * GRID_L / GRID_M as f64;
    let mut c = vec![0.0; nmax + 1];
    let mut hs = vec![0.0; nmax + 1];
    for i in 0..=GRID_M {
        let x = -GRID_L + step * i as f64;
        let v = f(x) * phi(x) * if i == 0 || i == GRID_M { 0.5 } else { 1.0 };
        if !v.is_finite() || v == 0.0 {
            continue;
        }
        h_all(x, &mut hs);
        for (cn, hn) in c.iter_mut().zip(&hs) {
            *cn += v * hn;
        }
    }
    for cn in &mut c {
        *cn *= step;
    }
    c
}

/// The `n` real zeros of `He_n`, ascending.
pub fn zeros(n: usize) -> Vec<f64> {
    if n == 0 {
        return Vec::new();
    }
    // all zeros lie inside (-sqrt(4n+2), sqrt(4n+2)); scan for sign changes, then bisect
    let r = (4.0 * n as f64 + 2.0).sqrt() + 0.5;
    let steps = 60 * n + 200;
    let mut out = Vec::with_capacity(n);
    let mut x0 = -r;
    let mut f0 = h(n, x0);
    for i in 1..=steps {
        let x1 = -r + 2.0 * r * i as f64 / steps as f64;
        let f1 = h(n, x1);
        if f0 == 0.0 {
            out.push(x0);
        } else if f0 * f1 < 0.0 {
            let (mut a, mut b, mut fa) = (x0, x1, f0);
            for _ in 0..80 {
                let m = 0.5 * (a + b);
                let fm = h(n, m);
                if fa * fm <= 0.0 {
                    b = m;
                } else {
                    a = m;
                    fa = fm;
                }
            }
            out.push(0.5 * (a + b));
        }
        x0 = x1;
        f0 = f1;
    }
    // odd degrees have an exact zero at the origin
    if n % 2 == 1 {
        if let Some(z) = out.iter_mut().min_by(|a, b| a.abs().total_cmp(&b.abs())) {
            *z = 0.0;
        }
    }
    out
}

/// Gauss-Hermite quadrature for the standard normal: `n` nodes (the zeros of `He_n`)
/// and weights summing to 1, exact for every polynomial of degree below `2n`.
pub fn gauss_hermite(n: usize) -> (Vec<f64>, Vec<f64>) {
    let nodes = zeros(n);
    let weights = nodes
        .iter()
        .map(|&x| {
            let p = h(n - 1, x);
            1.0 / (n as f64 * p * p)
        })
        .collect();
    (nodes, weights)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn explicit_values_and_formulas() {
        assert_eq!(he(0, 1.7), 1.0);
        assert!((he(2, 3.0) - 8.0).abs() < 1e-12);
        assert!((he(3, 2.0) - 2.0).abs() < 1e-12);
        assert!((he(4, 1.0) + 2.0).abs() < 1e-12);
        assert_eq!(he_coeffs(4), vec![3, 0, -6, 0, 1]);
        assert_eq!(he_coeffs(5), vec![0, 15, 0, -10, 0, 1]);
        assert_eq!(he_formula(0), "1");
        assert_eq!(he_formula(1), "x");
        assert_eq!(he_formula(2), "x² − 1");
        assert_eq!(he_formula(5), "x⁵ − 10x³ + 15x");
        assert_eq!(he_formula(10), "x¹⁰ − 45x⁸ + 630x⁶ − 3150x⁴ + 4725x² − 945");
        // normalised and raw versions agree
        let mut fact = 1.0f64;
        for n in 0..12 {
            if n > 0 {
                fact *= n as f64;
            }
            assert!((h(n, 1.3) - he(n, 1.3) / fact.sqrt()).abs() < 1e-10);
        }
        let mut all = [0.0; 9];
        h_all(-0.8, &mut all);
        for (n, v) in all.iter().enumerate() {
            assert!((v - h(n, -0.8)).abs() < 1e-12);
        }
    }

    #[test]
    fn orthonormal_under_the_gaussian() {
        for m in 0..=16 {
            for n in 0..=16 {
                let g = gauss_mean(|x| h(m, x) * h(n, x));
                let want = if m == n { 1.0 } else { 0.0 };
                assert!((g - want).abs() < 1e-8, "<h{m}, h{n}> = {g}");
            }
        }
        // and the monomials are not: E[x² x⁴] = 15
        assert!((gauss_mean(|x| x.powi(6)) - 15.0).abs() < 1e-7);
    }

    #[test]
    fn ladder_and_eigenfunction_identities() {
        let d = |f: &dyn Fn(f64) -> f64, x: f64| (f(x + 1e-4) - f(x - 1e-4)) / 2e-4;
        let d2 = |f: &dyn Fn(f64) -> f64, x: f64| (f(x + 1e-3) - 2.0 * f(x) + f(x - 1e-3)) / 1e-6;
        for n in 1..9 {
            for &x in &[-1.7, 0.3, 2.2] {
                let f = move |t: f64| he(n, t);
                // lowering: He_n' = n He_{n-1}
                assert!((d(&f, x) - n as f64 * he(n - 1, x)).abs() < 1e-4 * (1.0 + he(n, x).abs()));
                // raising: x He_n - He_n' = He_{n+1}, with the slope measured, not taken
                // from the recurrence that `he` itself is computed by (that would be
                // the same arithmetic on both sides, and could not fail)
                let raised = x * he(n, x) - d(&f, x);
                assert!((raised - he(n + 1, x)).abs() < 1e-7 * (1.0 + he(n + 1, x).abs()), "n = {n}, x = {x}");
                // Ornstein-Uhlenbeck: He_n'' - x He_n' = -n He_n
                let lhs = d2(&f, x) - x * d(&f, x);
                assert!((lhs + n as f64 * he(n, x)).abs() < 2e-2 * (1.0 + he(n, x).abs()), "n = {n}, x = {x}");
            }
        }
    }

    #[test]
    fn mehler_noise_operator_scales_by_rho_to_the_n() {
        // E[h_n(rho x + sqrt(1 - rho²) Z)] = rho^n h_n(x)
        for &rho in &[0.3f64, -0.6, 0.9] {
            for n in 0..8 {
                for &x in &[-1.2, 0.4, 2.0] {
                    let s = (1.0 - rho * rho).sqrt();
                    let lhs = gauss_mean(|z| h(n, rho * x + s * z));
                    assert!((lhs - rho.powi(n as i32) * h(n, x)).abs() < 1e-7, "n = {n}");
                }
            }
        }
    }

    #[test]
    fn coefficients_of_known_functions() {
        // ReLU: 1/sqrt(2 pi), 1/2, 1/(2 sqrt(pi)), 0, ...
        let c = coefficients(|x| x.max(0.0), 6);
        assert!((c[0] - 0.398_942_28).abs() < 1e-6);
        assert!((c[1] - 0.5).abs() < 1e-6);
        assert!((c[2] - 0.282_094_79).abs() < 1e-6);
        assert!(c[3].abs() < 1e-6 && c[5].abs() < 1e-6);
        // Parseval: E[relu²] = 1/2
        let c = coefficients(|x| x.max(0.0), 60);
        let total: f64 = c.iter().map(|v| v * v).sum();
        assert!((total - 0.5).abs() < 2e-3, "{total}");
        // a polynomial has a finite expansion: x³ = He_3 + 3 He_1 = sqrt(6) h_3 + 3 h_1
        let c = coefficients(|x| x * x * x, 6);
        assert!((c[1] - 3.0).abs() < 1e-7 && (c[3] - 6f64.sqrt()).abs() < 1e-7);
        assert!(c[0].abs() < 1e-9 && c[2].abs() < 1e-9 && c[4].abs() < 1e-7 && c[5].abs() < 1e-7);
        // exp(t x) = exp(t²/2) sum t^n/sqrt(n!) h_n   (the generating function)
        let t = 0.7f64;
        let c = coefficients(|x| (t * x).exp(), 6);
        let mut fact = 1.0f64;
        for (n, cn) in c.iter().enumerate() {
            if n > 0 {
                fact *= n as f64;
            }
            assert!((cn - (t * t / 2.0).exp() * t.powi(n as i32) / fact.sqrt()).abs() < 1e-7);
        }
    }

    #[test]
    fn gauss_hermite_rule() {
        for n in [1usize, 2, 5, 12, 30] {
            let (x, w) = gauss_hermite(n);
            assert_eq!(x.len(), n);
            assert!((w.iter().sum::<f64>() - 1.0).abs() < 1e-9, "weights of {n}");
            assert!(x.windows(2).all(|p| p[0] < p[1]));
            // exact for polynomials below degree 2n: check the even moments (2k - 1)!!
            let mut want = 1.0;
            for k in 1..n {
                want *= (2 * k - 1) as f64;
                let got: f64 = x.iter().zip(&w).map(|(x, w)| w * x.powi(2 * k as i32)).sum();
                assert!((got - want).abs() < 1e-7 * want.max(1.0), "n = {n}, moment {}", 2 * k);
            }
        }
        // zeros of consecutive degrees interlace
        let (a, b) = (zeros(7), zeros(8));
        for i in 0..7 {
            assert!(b[i] < a[i] && a[i] < b[i + 1]);
        }
        // the rule reproduces orthonormality exactly up to combined degree 2n - 1
        let (x, w) = gauss_hermite(6);
        for m in 0..6 {
            for k in 0..(12 - m).min(6) {
                let s: f64 = x.iter().zip(&w).map(|(x, w)| w * h(m, *x) * h(k, *x)).sum();
                assert!((s - if m == k { 1.0 } else { 0.0 }).abs() < 1e-9);
            }
        }
    }
}

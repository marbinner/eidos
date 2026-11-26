//! Powers, roots and logarithms of a matrix: A to the power t for every t, as the
//! path of maps e^(tL) through a logarithm L of A; the square roots B with B² = A;
//! and the matrices for which neither exists.

use super::*;
use glam::{DMat2, DVec2};

pub const DEF: SceneDef = SceneDef {
    id: "matfun",
    title: "Powers, roots and logarithms of a matrix",
    group: EXPONENTIAL,
    blurb: "A matrix to the power one half, or any power: functions of a matrix through its eigenvalues, and where that fails.",
    make: |_| Box::new(Matfun::new()),
};

// ----- the arithmetic, in double precision -------------------------------------------

/// The time knob runs from I (t = 0) through A (t = 1) to A² (t = 2).
const T_MAX: f64 = 2.0;
/// Units of t a second, played.
const PLAY_RATE: f64 = 0.5;
/// The largest entry of A either way.
const ENTRY_MAX: f64 = 4.0;
/// What counts as exactly zero, for a number of the size of the entries (squared, for
/// a determinant).
const EXACT: f64 = 1e-9;

/// The quarter turn: e^(θJ) is the turn by θ.
const J: DMat2 = DMat2::from_cols(DVec2::new(0.0, 1.0), DVec2::new(-1.0, 0.0));
const I: DMat2 = DMat2::IDENTITY;

/// A matrix by its rows, [[a, b], [c, d]], the way matrices are written.
fn rows([a, b, c, d]: [f64; 4]) -> DMat2 {
    DMat2::from_cols(DVec2::new(a, c), DVec2::new(b, d))
}

fn trace(m: DMat2) -> f64 {
    m.x_axis.x + m.y_axis.y
}

/// The size of the largest entry.
fn size(m: DMat2) -> f64 {
    m.to_cols_array().iter().fold(0.0f64, |s, v| s.max(v.abs()))
}

/// (tr A / 2)² − det A: positive for two real eigenvalues, negative for a complex pair,
/// zero for a double one. Worked out as ((a − d)/2)² + bc, so that a double eigenvalue
/// gives 0 exactly rather than the difference of two nearly equal numbers.
fn quarter_disc(m: DMat2) -> f64 {
    let half = 0.5 * (m.x_axis.x - m.y_axis.y);
    half * half + m.y_axis.x * m.x_axis.y
}

/// e^(Lt), in closed form (as in the scene "expm"). L = sI + B with s = tr L / 2, and B
/// squares to qI. So e^(Lt) = e^(st) (c I + g B), with c = cosh(√q t) and g = sinh(√q t)/√q
/// if q > 0, the same with cos and sin of √−q t if q < 0; near q t² = 0 the three are one
/// power series, summed as one, so that nothing is divided by a number near zero.
fn expm(l: DMat2, t: f64) -> DMat2 {
    let s = 0.5 * trace(l);
    let b = l - I * s;
    let q = quarter_disc(l);
    let z = q * t * t;
    let (c, g) = if z.abs() < 1.0 {
        let (mut c, mut g, mut tc, mut tg) = (1.0, 1.0, 1.0, 1.0);
        for k in 1..=12 {
            let k = k as f64;
            tc *= z / ((2.0 * k - 1.0) * (2.0 * k));
            tg *= z / ((2.0 * k) * (2.0 * k + 1.0));
            c += tc;
            g += tg;
        }
        (c, g * t)
    } else if q > 0.0 {
        let r = q.sqrt();
        ((r * t).cosh(), (r * t).sinh() / r)
    } else {
        let r = (-q).sqrt();
        ((r * t).cos(), (r * t).sin() / r)
    };
    (I * c + b * g) * (s * t).exp()
}

/// The unit direction a matrix that is (taken to be) flat sends to 0: across its
/// longer row.
fn null_direction(m: DMat2) -> DVec2 {
    let (r1, r2) = (m.row(0), m.row(1));
    let r = if r1.length_squared() >= r2.length_squared() { r1 } else { r2 };
    if r.length_squared() > 0.0 {
        r.perp().normalize()
    } else {
        DVec2::X
    }
}

/// What a real 2×2 matrix is, up to a change of basis.
#[derive(Clone, Copy, Debug, PartialEq)]
enum Kind {
    /// Two different real eigenvalues, the larger first, with unit eigenvectors.
    Real { l1: f64, l2: f64, v1: DVec2, v2: DVec2 },
    /// A complex pair r(cos θ ± i sin θ), 0 < θ < π: A = r(cos θ I + sin θ K), where
    /// K = (A − (tr A/2) I)/(r sin θ) squares to −I (K = J for a turn and a stretch).
    Turning { r: f64, theta: f64, k: DMat2 },
    /// One eigenvalue twice, with a single eigenline v: A = lI + N, N ≠ 0, N² = 0.
    Defective { l: f64, n: DMat2, v: DVec2 },
    /// A = lI: every line is an eigenline.
    Scalar { l: f64 },
}

fn kind(a: DMat2) -> Kind {
    let scale = size(a);
    let s = 0.5 * trace(a);
    let b = a - I * s;
    if size(b) <= EXACT * scale {
        return Kind::Scalar { l: s };
    }
    let q = quarter_disc(a);
    let flat = EXACT * scale * scale;
    if q < -flat {
        let w = (-q).sqrt();
        return Kind::Turning { r: s.hypot(w), theta: w.atan2(s), k: b * (1.0 / w) };
    }
    if q <= flat {
        return Kind::Defective { l: s, n: b, v: null_direction(b) };
    }
    let rho = q.sqrt();
    // (the one further from 0 as s ± ρ, the other from the determinant: no difference
    // of two nearly equal numbers)
    let det = a.determinant();
    let (l1, l2) = if s >= 0.0 {
        let l1 = s + rho;
        (l1, det / l1)
    } else {
        let l2 = s - rho;
        (det / l2, l2)
    };
    Kind::Real { l1, l2, v1: null_direction(b - I * rho), v2: null_direction(b + I * rho) }
}

/// Whether A flattens the plane (an eigenvalue 0), for a matrix of its size.
fn singular(a: DMat2) -> bool {
    let scale = size(a);
    a.determinant().abs() <= EXACT * scale * scale
}

/// How many real square roots, or real logarithms, a matrix has.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Count {
    None,
    One,
    Two,
    Four,
    /// One logarithm for each angle θ + 360° k (a matrix with complex eigenvalues).
    PerTurn,
    Many,
}

/// The real square roots and the real logarithms of A: how many there are.
fn counts(a: DMat2) -> (Count, Count) {
    let flat = singular(a);
    match kind(a) {
        // (flat, its eigenvalues are 0 and tr A)
        Kind::Real { .. } if flat => (if trace(a) > 0.0 { Count::Two } else { Count::None }, Count::None),
        Kind::Real { l2, .. } if l2 > 0.0 => (Count::Four, Count::One),
        Kind::Real { .. } => (Count::None, Count::None),
        Kind::Turning { .. } => (Count::Two, Count::PerTurn),
        Kind::Defective { l, .. } if l > 0.0 && !flat => (Count::Two, Count::One),
        Kind::Defective { .. } => (Count::None, Count::None),
        Kind::Scalar { l: 0.0 } => (Count::Many, Count::None),
        Kind::Scalar { .. } => (Count::Many, Count::Many),
    }
}

/// Whether A has more than one real logarithm to choose among (by its angle).
fn windings(a: DMat2) -> bool {
    matches!(counts(a).1, Count::PerTurn | Count::Many)
}

/// ln(l1)/... : (ln l1 − ln l2)/(l1 − l2) for two positive numbers l1 > l2, without
/// dividing a small difference by a small difference when they are close: there it is
/// (1/s)(1 + u/3 + u²/5 + u³/7 + …) with s their mean and u = (half their gap / s)².
fn log_slope(l1: f64, l2: f64) -> f64 {
    let (s, rho) = (0.5 * (l1 + l2), 0.5 * (l1 - l2));
    let x = rho / s;
    if x < 1e-3 {
        let u = x * x;
        (1.0 + u / 3.0 + u * u / 5.0 + u * u * u / 7.0) / s
    } else {
        (l1.ln() - l2.ln()) / (l1 - l2)
    }
}

/// A real logarithm L of A (e^L = A), if A has one. For a matrix with complex
/// eigenvalues the angle of L is θ + 360° × `winding` (0: the principal logarithm); for
/// a multiple of I it is 360° × `winding` (or 180° more, if the multiple is negative),
/// turning as J does. Every other matrix has one real logarithm at most.
fn log_of(a: DMat2, winding: i32) -> Option<DMat2> {
    let b = a - I * (0.5 * trace(a));
    let flat = singular(a);
    match kind(a) {
        Kind::Real { l1, l2, .. } if l2 > 0.0 && !flat => Some(I * (0.5 * (l1.ln() + l2.ln())) + b * log_slope(l1, l2)),
        Kind::Turning { r, theta, .. } => {
            let w = (-quarter_disc(a)).sqrt();
            Some(I * r.ln() + b * ((theta + std::f64::consts::TAU * winding as f64) / w))
        }
        Kind::Defective { l, n, .. } if l > 0.0 && !flat => Some(I * l.ln() + n * (1.0 / l)),
        Kind::Scalar { l } if l > 0.0 => Some(I * l.ln() + J * (std::f64::consts::TAU * winding as f64)),
        Kind::Scalar { l } if l < 0.0 => Some(I * (-l).ln() + J * (std::f64::consts::PI + std::f64::consts::TAU * winding as f64)),
        _ => None,
    }
}

/// The angle of that logarithm, in radians: how far A^t turns from t = 0 to 1 (none
/// for a matrix that has no choice of angle).
fn log_angle(a: DMat2, winding: i32) -> Option<f64> {
    let k = std::f64::consts::TAU * winding as f64;
    match kind(a) {
        Kind::Turning { theta, .. } => Some(theta + k),
        Kind::Scalar { l } if l > 0.0 => Some(k),
        Kind::Scalar { l } if l < 0.0 => Some(std::f64::consts::PI + k),
        _ => None,
    }
}

/// A^t = e^(tL), with L the logarithm of that winding (none without a real logarithm).
fn power(a: DMat2, winding: i32, t: f64) -> Option<DMat2> {
    log_of(a, winding).map(|l| expm(l, t))
}

/// The straight blend from I to A.
fn blend(a: DMat2, t: f64) -> DMat2 {
    I * (1.0 - t) + a * t
}

/// The real square roots the scene offers: all of them where there are finitely many
/// (the principal one first, where there is one), and four where there are infinitely
/// many. None where there is none.
fn roots(a: DMat2) -> Vec<DMat2> {
    let b = a - I * (0.5 * trace(a));
    match (kind(a), counts(a).0) {
        (Kind::Real { l1, l2, .. }, Count::Four) => {
            // B = ±√l1 P1 ± √l2 P2, P1 and P2 the projections onto the eigenlines
            // along each other
            let p1 = (a - I * l2) * (1.0 / (l1 - l2));
            let p2 = I - p1;
            let (r1, r2) = (l1.sqrt(), l2.sqrt());
            vec![p1 * r1 + p2 * r2, p1 * r1 - p2 * r2, p2 * r2 - p1 * r1, -(p1 * r1 + p2 * r2)]
        }
        (Kind::Real { .. }, Count::Two) => {
            // rank 1, λ = tr A and 0: A² = (tr A) A, so (A/√tr A)² = A
            let r = a * (1.0 / trace(a).sqrt());
            vec![r, -r]
        }
        (Kind::Turning { r, theta, .. }, _) => {
            // half the angle and the root of the size
            let w = (-quarter_disc(a)).sqrt();
            let half = (I * (0.5 * theta).cos() + b * ((0.5 * theta).sin() / w)) * r.sqrt();
            vec![half, -half]
        }
        (Kind::Defective { l, n, .. }, Count::Two) => {
            let root = I * l.sqrt() + n * (0.5 / l.sqrt());
            vec![root, -root]
        }
        (Kind::Scalar { l }, _) => {
            let k = l.abs().sqrt();
            let mats = if l > 0.0 {
                // ±I, and every mirror (any S ≠ ±I with S² = I)
                [I, -I, rows([1.0, 0.0, 0.0, -1.0]), rows([0.0, 1.0, 1.0, 0.0])]
            } else if l < 0.0 {
                // every K with K² = −I: J, −J, and J seen in other coordinates
                [J, -J, rows([1.0, -2.0, 1.0, -1.0]), rows([-1.0, 2.0, -1.0, 1.0])]
            } else {
                // B² = 0: B = 0, and every B that flattens the plane onto its own null line
                [DMat2::ZERO, rows([0.0, 1.0, 0.0, 0.0]), rows([0.0, 0.0, 1.0, 0.0]), rows([1.0, 1.0, -1.0, -1.0])]
            };
            mats.iter().map(|m| *m * k.max(if l == 0.0 { 1.0 } else { 0.0 })).collect()
        }
        _ => Vec::new(),
    }
}

/// Whether A has a principal square root and logarithm: no eigenvalue at 0 or on the
/// negative real axis. (It is then `roots(a)[0]` and `log_of(a, 0)`.)
fn principal(a: DMat2) -> bool {
    let flat = singular(a);
    match kind(a) {
        Kind::Real { l2, .. } => l2 > 0.0 && !flat,
        Kind::Turning { .. } => true,
        Kind::Defective { l, .. } | Kind::Scalar { l } => l > 0.0 && !flat,
    }
}

/// The winding whose path A^t passes through the root `b` at t = 1/2, if one does.
fn path_through(a: DMat2, b: DMat2) -> Option<i32> {
    let windings: &[i32] = if windings(a) { &[0, -1, 1] } else { &[0] };
    windings.iter().copied().find(|&k| power(a, k, 0.5).is_some_and(|h| size(h - b) <= 1e-9 * size(b).max(1.0)))
}

// ----- complex numbers: the eigenvalues ----------------------------------------------

/// A complex number as (real part, imaginary part).
type Cx = DVec2;

fn c_exp(z: Cx) -> Cx {
    DVec2::from_angle(z.y) * z.x.exp()
}

fn c_square(z: Cx) -> Cx {
    DVec2::new(z.x * z.x - z.y * z.y, 2.0 * z.x * z.y)
}

/// The eigenvalues of m: the larger real one first, or the one above the real axis.
fn eigenvalues(m: DMat2) -> [Cx; 2] {
    let s = 0.5 * trace(m);
    let q = quarter_disc(m);
    if q >= 0.0 {
        let r = q.sqrt();
        [DVec2::new(s + r, 0.0), DVec2::new(s - r, 0.0)]
    } else {
        let w = (-q).sqrt();
        [DVec2::new(s, w), DVec2::new(s, -w)]
    }
}

/// `mu` in the order of `lam`: each `mu[i]`, put through `f`, the nearer to `lam[i]`.
fn paired(lam: [Cx; 2], mu: [Cx; 2], f: impl Fn(Cx) -> Cx) -> [Cx; 2] {
    let straight = f(mu[0]).distance(lam[0]) + f(mu[1]).distance(lam[1]);
    let crossed = f(mu[1]).distance(lam[0]) + f(mu[0]).distance(lam[1]);
    if crossed < straight {
        [mu[1], mu[0]]
    } else {
        mu
    }
}

/// The eigenvalues of the logarithm of that winding, in the order of A's (e^μ = λ).
fn log_eigenvalues(a: DMat2, winding: i32) -> Option<[Cx; 2]> {
    log_of(a, winding).map(|l| paired(eigenvalues(a), eigenvalues(l), c_exp))
}

// ----- words and numbers -------------------------------------------------------------

/// A number for a matrix on the screen, as a formula, in five characters at most (so
/// that the brackets never move as the numbers change): two decimals, then one, then
/// none, and a power of ten beyond 9999.
fn cell(v: f64) -> String {
    let a = v.abs();
    let number = if !v.is_finite() {
        "?".into()
    } else if a < 9.995 {
        num(v, 2)
    } else if a < 99.95 {
        num(v, 1)
    } else if a < 9999.5 {
        num(v, 0)
    } else {
        format!("{v:.0e}").replace('-', "−")
    };
    format!("${number}$")
}

/// The widest `cell` can be, in characters of its number, a minus sign counted as two.
const CELL: usize = 6;

/// A number for the screen: two decimals and a true minus sign, two significant
/// figures below 0.1 (an area of 0.0076 is not 0.01), and powers of ten when too long.
fn short(v: f64) -> String {
    let a = v.abs();
    if !v.is_finite() {
        "?".into()
    } else if a >= 9999.5 || (a > 0.0 && a < 1e-4) {
        format!("{v:.1e}").replace('-', "−")
    } else if a > 0.0 && a < 0.0995 {
        let digits = (1 - a.log10().floor() as i32).clamp(2, 6) as usize;
        num(v, digits)
    } else {
        num(v, 2)
    }
}

/// An angle in degrees, whole if it is whole, else to a tenth, with a true minus sign.
fn degrees(radians: f64) -> String {
    let d = radians.to_degrees();
    let d = if d.abs() < 0.05 { 0.0 } else { d };
    let text = if (d - d.round()).abs() < 0.05 { format!("{:.0}°", d.round()) } else { format!("{d:.1}°") };
    text.replace('-', "−")
}

/// The picture's heading: what A is, and what it has of roots and logarithms.
fn describe(a: DMat2) -> (String, String) {
    let flat = singular(a);
    let (roots, logs) = counts(a);
    let first = match kind(a) {
        Kind::Real { .. } if flat => format!(r"$\det A = 0$, it flattens the plane: $\lambda = {}$ and $0$", num(trace(a), 2)),
        Kind::Real { l1, l2, .. } if l2 > 0.0 => format!(r"two positive eigenvalues: $\lambda = {}$ and ${}$", num(l1, 2), num(l2, 2)),
        Kind::Real { l1, l2, .. } if l1 < 0.0 => format!(r"two negative eigenvalues: $\lambda = {}$ and ${}$", num(l1, 2), num(l2, 2)),
        Kind::Real { l1, l2, .. } => format!(r"$\det A < 0$, it flips the plane: $\lambda = {}$ and ${}$", num(l1, 2), num(l2, 2)),
        Kind::Turning { r, theta, k } => {
            let turn = size(k - J) < 1e-9;
            match (turn, (r - 1.0).abs() < 1e-9) {
                (true, true) => format!(r"a turn by {}: $\lambda = \cos\theta \pm i \sin\theta$, $\theta = {}$", degrees(theta), degrees(theta).replace('°', r"^\circ")),
                (true, false) => format!(r"a turn by {} and a stretch by {}: $\lambda = {}(\cos\theta \pm i \sin\theta)$, $\theta = {}$", degrees(theta), num(r, 2), num(r, 2), degrees(theta).replace('°', r"^\circ")),
                _ => format!(r"complex eigenvalues: $\lambda = r(\cos\theta \pm i \sin\theta)$, $r = {}$, $\theta = {}$", num(r, 2), degrees(theta).replace('°', r"^\circ")),
            }
        }
        Kind::Defective { l, .. } => format!(r"one eigenvalue twice, $\lambda = {}$, and a single eigenline", num(l, 2)),
        Kind::Scalar { l: 0.0 } => "$A = 0$".to_string(),
        Kind::Scalar { l: 1.0 } => r"$A = I$: every line is an eigenline, $\lambda = 1$ twice".to_string(),
        Kind::Scalar { l: -1.0 } => r"$A = -I$, the half turn: $\lambda = -1$ twice".to_string(),
        Kind::Scalar { l } => format!(r"$A = {}\,I$: every line is an eigenline", num(l, 2)),
    };
    let second = match (roots, logs) {
        (Count::Four, _) => "one real logarithm, and four real square roots".to_string(),
        (Count::Two, Count::PerTurn) => "a real logarithm for each whole number of extra turns; two real square roots".to_string(),
        (Count::Two, Count::One) => "one real logarithm, and two real square roots".to_string(),
        (Count::Two, _) => r"two real square roots; no logarithm, as $e^L$ never flattens the plane".to_string(),
        (Count::Many, Count::None) => r"no logarithm; square roots without end: $0$ and every $B$ with $B^2 = 0$".to_string(),
        (Count::Many, _) if matches!(kind(a), Kind::Scalar { l } if l < 0.0) => r"real square roots and real logarithms without end ($J$ and $\pi J$ among them)".to_string(),
        (Count::Many, _) => "real square roots and real logarithms without end".to_string(),
        (Count::None, _) if flat => "no logarithm, and no real square root".to_string(),
        (Count::None, _) if a.determinant() < 0.0 => r"no real square root, no real logarithm: $(\det B)^2$ and $e^{\tr L}$ are never negative".to_string(),
        (Count::None, _) => r"no real square root and no real logarithm, though $\det A > 0$".to_string(),
        _ => String::new(),
    };
    (first, second)
}

// ----- experiments and quick choices -------------------------------------------------

/// The turn by 170°, by its rows.
fn turn_170() -> [f64; 4] {
    let (s, c) = 170f64.to_radians().sin_cos();
    [c, -s, s, c]
}

const STRETCH: [f64; 4] = [1.04, 0.4, 0.4, 1.04];
const TURN_STRETCH: [f64; 4] = [0.75, -1.0, 1.0, 0.75];
const MIRROR: [f64; 4] = [1.0, 0.0, 0.0, -1.0];

#[derive(Clone, Copy, Debug, PartialEq)]
enum Show {
    /// A^t, for t from 0 to 2.
    Power,
    /// A square root B, shown acting twice.
    Root,
}

struct Trial {
    label: &'static str,
    look_for: &'static str,
    a: fn() -> [f64; 4],
    show: Show,
    winding: i32,
    blend: bool,
    /// Where t plays to, from 0.
    play_to: f64,
    /// Step through the square roots.
    tour: bool,
}

const TRIALS: [Trial; 5] = [
    Trial {
        label: "halfway to $A$: a square root",
        look_for: r"$A$ turns by 53.13° and stretches by 1.25. Its principal square root $B = A^{1/2}$ turns by half the angle, 26.57°, and stretches by $\sqrt{1.25} = 1.118$. The blue letter is $B$ applied once; $B$ applied again (the second arrow) lands it exactly on the yellow one, where $A$ takes it: $B^2 = A$. The white curve is the path of the letter's corner under $A^t$ from $t = 0$ to 1: $B$ takes the corner to its middle, $t = 1/2$.",
        a: || TURN_STRETCH,
        show: Show::Root,
        winding: 0,
        blend: false,
        play_to: 1.0,
        tour: false,
    },
    Trial {
        label: "the straight blend collapses",
        look_for: r"$A$ is the turn by 170°. At $t = 0.5$, $A^t$ is the turn by 85° (yellow), and its area is 1, as at every $t$: $\det A^t = (\det A)^t = 1$. The straight blend $(1 - t)I + tA$ (blue) is that same turn shrunk to 0.087 of its size: its area is 0.0076. In the complex plane its eigenvalues cut straight across the circle and pass 0.087 from 0, where those of $A^t$ go round it.",
        a: turn_170,
        show: Show::Power,
        winding: 0,
        blend: true,
        play_to: 0.5,
        tour: false,
    },
    Trial {
        label: "four square roots of a stretch",
        look_for: r"$A$ stretches by $\lambda = 1.44$ along $(1, 1)$ and by $0.64$ along $(1, -1)$. A square root $B$ keeps those two lines and stretches them by $\pm 1.2$ and $\pm 0.8$: four choices of sign, four roots, and the picture steps through them. Each, applied twice, lands the letter on the yellow one. The two with one minus sign flip the letter over ($\det B = -0.96$); the one with two is the first turned a half turn. Only $B_1$, both signs positive, is $A^{1/2}$.",
        a: || STRETCH,
        show: Show::Root,
        winding: 0,
        blend: false,
        play_to: 1.0,
        tour: true,
    },
    Trial {
        label: "a mirror has no real square root",
        look_for: r"$A$ mirrors the plane in the $x$ axis: $\det A = -1$. A real $B$ with $B^2 = A$ would have $(\det B)^2 = -1$: there is none. Nor is there a real logarithm, as $\det e^L = e^{\tr L} > 0$; so there is no real $A^t$, and nothing yellow. The blue blend is one path from $I$ to $A$, and at $t = 0.5$ it is flat: its determinant is 0. Every path of maps from $I$ to $A$ without jumps is flat somewhere, since its determinant goes from 1 to −1.",
        a: || MIRROR,
        show: Show::Power,
        winding: 0,
        blend: true,
        play_to: 1.0,
        tour: false,
    },
    Trial {
        label: "the logarithm winds",
        look_for: r"The turn by 170° again, now through the logarithm $L = -190^\circ \times J$ ($-3.32J$ in radians) instead of $170^\circ \times J$: $e^L$ is the same turn. $A^t$ now turns the other way, by $-190^\circ \times t$: the white path of the letter's corner goes round clockwise, and meets the principal path (dashed, anticlockwise) at $t = 1$. Halfway, at $t = 0.5$, it is the turn by −95°: the other square root of $A$, as $85^\circ + 180^\circ = 265^\circ$ is the same turn.",
        a: turn_170,
        show: Show::Power,
        winding: -1,
        blend: false,
        play_to: 1.0,
        tour: false,
    },
];

/// A quick choice of A: its button, the matrix by rows, and what it shows.
type Preset = (&'static str, fn() -> [f64; 4], &'static str);

const PRESETS: [Preset; 7] = [
    (
        "stretch",
        || STRETCH,
        r"$\lambda = 1.44$ along $(1, 1)$ and $0.64$ along $(1, -1)$. $A^t$ stretches the same two lines, by $1.44^t$ and $0.64^t$: written $A = VDV^{-1}$ in its eigenvectors, $A^t = VD^tV^{-1}$.",
    ),
    ("turn and stretch", || TURN_STRETCH, r"$A$ is 1.25 times the turn by 53.13°. $A^t$ is $1.25^t$ times the turn by $53.13^\circ \times t$: its eigenvalues $\lambda^t$ spiral out from 1 in the complex plane."),
    ("turn by 170°", turn_170, r"$A^t$ is the turn by $170^\circ \times t$ (with the principal logarithm): halfway, the turn by 85°, whose square is $A$."),
    (
        "shear",
        || [1.0, 1.0, 0.0, 1.0],
        r"$\lambda = 1$ twice, with a single eigenline: no basis of eigenvectors, so the eigenvalue recipe has no $V$ to use. The series does it: $N = A - I$ has $N^2 = 0$, so $\log A = N - N^2/2 + \cdots = N$, and $A^t = e^{tN} = I + tN$, the shear by $t$. The shear by ½ is a square root. Here the straight blend is $A^t$ itself: $(1 - t)I + t(I + N) = I + tN$.",
    ),
    ("mirror", || MIRROR, r"$\det A = -1$: no real square root and no real logarithm, so no real $A^t$ but for whole $t$."),
    (
        "half turn",
        || [-1.0, 0.0, 0.0, -1.0],
        r"$A = -I$: its eigenvalue $-1$ is on the negative real axis, yet $\pi J$ is a real logarithm ($e^{\pi J}$ is the half turn) and $J$ a real square root. No eigenvalue at 0 or below is enough for a real logarithm, not needed.",
    ),
    (
        "two negative $\\lambda$",
        || [-1.0, 0.0, 0.0, -2.0],
        r"$\lambda = -1$ and $-2$: $\det A = 2 > 0$, and still no real square root. A root $B$ would keep $A$'s two eigenlines, and stretch them by numbers whose squares are $-1$ and $-2$.",
    ),
];

// ----- following -----------------------------------------------------------------------

/// How fast a change is glided to: each of the two eases covers two thirds of what is
/// left in this many seconds.
const FOLLOW: f64 = 0.09;

/// Numbers as asked for and as drawn: the drawn ones follow through `mid`, two eases
/// one after the other, so that they set off gently and arrive exactly.
#[derive(Clone, Copy, Debug, PartialEq)]
struct Follow<const N: usize> {
    to: [f64; N],
    mid: [f64; N],
    at: [f64; N],
}

impl<const N: usize> Follow<N> {
    fn new(v: [f64; N]) -> Self {
        Follow { to: v, mid: v, at: v }
    }

    fn arrived(&self) -> bool {
        self.at == self.to && self.mid == self.to
    }

    fn snap(&mut self) {
        (self.mid, self.at) = (self.to, self.to);
    }

    fn step(&mut self, dt: f64) {
        if self.arrived() || dt <= 0.0 || !dt.is_finite() {
            return;
        }
        let k = 1.0 - (-dt / FOLLOW).exp();
        let mut apart = 0.0f64;
        for i in 0..N {
            self.mid[i] += (self.to[i] - self.mid[i]) * k;
            self.at[i] += (self.mid[i] - self.at[i]) * k;
            apart = apart.max((self.to[i] - self.mid[i]).abs()).max((self.to[i] - self.at[i]).abs());
        }
        // (exactly there in the end: the picture then stops changing)
        if apart < 2e-4 || !apart.is_finite() {
            self.snap();
        }
    }
}

/// A map as drawn (its matrix, and how much it shows), following the one asked for:
/// at once while what is asked for only moves along its path (as t does), gliding when
/// it changes to another (`key`).
#[derive(Clone, Copy, Debug)]
struct Drawn {
    follow: Follow<5>,
    key: Option<[u64; 8]>,
}

impl Drawn {
    fn new() -> Drawn {
        Drawn { follow: Follow::new([1.0, 0.0, 0.0, 1.0, 1.0]), key: None }
    }

    fn aim(&mut self, m: DMat2, alpha: f64, key: [u64; 8]) {
        let fresh = self.key.is_none() || (self.key == Some(key) && self.follow.arrived());
        self.follow.to = [m.x_axis.x, m.x_axis.y, m.y_axis.x, m.y_axis.y, alpha];
        if fresh {
            self.follow.snap();
        }
        self.key = Some(key);
    }

    fn matrix(&self) -> DMat2 {
        let v = self.follow.at;
        DMat2::from_cols(DVec2::new(v[0], v[1]), DVec2::new(v[2], v[3]))
    }

    fn alpha(&self) -> f32 {
        self.follow.at[4].clamp(0.0, 1.0) as f32
    }
}

fn bits(m: DMat2) -> [u64; 4] {
    m.to_cols_array().map(f64::to_bits)
}

// ----- the scene -----------------------------------------------------------------------

/// What is drawn: the matrix, as a power or a square root, through which logarithm,
/// and which root.
#[derive(Clone, Copy, Debug, PartialEq)]
struct Setting {
    a: DMat2,
    show: Show,
    winding: i32,
    root: usize,
}

/// Where t goes once the setting asked for is reached.
#[derive(Clone, Copy, Debug, PartialEq)]
enum Goal {
    Rest,
    Glide(f64),
    Play(f64),
}

/// Seconds the axes go without numbers after the view is moved to a new place.
const SETTLE: f32 = 0.6;

/// How faded the paths and eigenlines are when they are not drawn at all.
const FADED: f64 = 0.3;

/// Seconds the tour of the square roots spends on each.
const TOUR_STOP: f32 = 2.6;

pub struct Matfun {
    now: Setting,
    /// A setting asked for and not reached yet: a change of path waits for t to come
    /// back to 0, where every path is at I, and a square root for t to reach 1, where
    /// A^t is A.
    next: Option<Setting>,
    /// Go back to t = 0 on the way, even for the same path (an experiment plays from
    /// the start).
    afresh: bool,
    goal: Goal,
    /// The time, and the middle of its two eases when it glides.
    t: f64,
    t_mid: f64,
    blend: bool,
    /// The tour of the square roots: seconds into it.
    tour: Option<f32>,
    /// The yellow map (A^t, or A = B²) and the blue one (the blend, or the root B).
    yellow: Drawn,
    blue: Drawn,
    /// How much the paths and eigenlines show: they fade out while t goes back to 0
    /// for a new path, and in again for the new one.
    fade: f64,
    /// How much the images of the unit circle show: not with the square roots.
    round: f64,
    /// The part of the complex plane shown: its centre and half its size each way.
    plane: Follow<4>,
    caption: Option<Caption>,
    heading: Option<Heading>,
    canvas: Vec2,
    inset: f32,
    measured: bool,
    new_view: bool,
    /// Seconds more the axes go without their numbers: while the view glides to a new
    /// place, numbers would slide past and swap with one another.
    settle: f32,
}

impl Matfun {
    fn new() -> Self {
        let a = rows(turn_170());
        let mut scene = Matfun {
            now: Setting { a, show: Show::Power, winding: 0, root: 0 },
            next: None,
            afresh: false,
            goal: Goal::Rest,
            t: 0.5,
            t_mid: 0.5,
            blend: true,
            tour: None,
            yellow: Drawn::new(),
            blue: Drawn::new(),
            fade: 1.0,
            round: 1.0,
            plane: Follow::new(plane_fit(a)),
            caption: None,
            heading: None,
            canvas: Vec2::new(1080.0, 1000.0),
            inset: 0.0,
            measured: false,
            new_view: false,
            settle: 0.0,
        };
        scene.aim();
        scene
    }

    /// Set up experiment `k` of `TRIALS`.
    fn set_up(&mut self, k: usize) {
        let trial = &TRIALS[k];
        self.blend = trial.blend;
        let s = Setting { a: rows((trial.a)()), show: trial.show, winding: trial.winding, root: 0 };
        self.ask(s, true, Goal::Play(trial.play_to));
        self.tour = trial.tour.then_some(0.0);
    }

    /// Ask for a setting: the picture makes its way there (see `next`), and t then goes
    /// to the goal. `afresh` plays the path from t = 0 even if it is the same one.
    fn ask(&mut self, s: Setting, afresh: bool, goal: Goal) {
        let s = Setting { a: within_limits(s.a), winding: if windings(s.a) { s.winding.clamp(-1, 1) } else { 0 }, ..s };
        self.next = Some(s);
        self.afresh = afresh;
        self.goal = goal;
        self.tour = None;
    }

    /// A new matrix, by a button or a typed number: shown as it was asked for before
    /// (power or roots), the same t, the principal logarithm and the first root.
    fn ask_matrix(&mut self, a: DMat2) {
        let target = self.next.unwrap_or(self.now);
        let t = self.t;
        self.ask(Setting { a, winding: 0, root: 0, ..target }, false, Goal::Play(t.max(0.0)));
        if target.show == Show::Power && t <= 0.0 {
            self.goal = Goal::Rest;
        }
    }

    /// Move t by hand to `t`: whatever was on its way is there at once.
    fn set_t(&mut self, t: f64) {
        if let Some(s) = self.next.take() {
            self.now = s;
        }
        let t = t.clamp(0.0, T_MAX);
        (self.t, self.t_mid, self.goal, self.afresh, self.tour) = (t, t, Goal::Rest, false, None);
        self.now.show = Show::Power;
    }

    fn playing(&self) -> bool {
        matches!(self.goal, Goal::Play(_)) || self.next.is_some()
    }

    /// Play from where t is (from 0, if it is at the end), or pause.
    fn play_or_pause(&mut self) {
        if self.playing() {
            if let Some(s) = self.next.take() {
                self.now = s;
            }
            self.goal = Goal::Rest;
            self.afresh = false;
            return;
        }
        let s = Setting { show: Show::Power, ..self.now };
        if self.t >= T_MAX - 1e-9 {
            self.ask(s, true, Goal::Play(T_MAX));
        } else {
            self.ask(s, false, Goal::Play(T_MAX));
        }
    }

    /// The next step from `now` towards `s`, and the time at which it can be taken
    /// (none: at once).
    fn step_towards(&self, s: Setting) -> (Setting, Option<f64>) {
        let n = self.now;
        let new_path = n.a != s.a || n.winding != s.winding || self.afresh;
        if n.show == Show::Root && (new_path || s.show == Show::Power) {
            // (a root is drawn at t = 1, where A^t is A)
            return (Setting { show: Show::Power, ..n }, None);
        }
        if new_path {
            // every path starts at I: change the path there
            return (Setting { show: Show::Power, ..s }, Some(0.0));
        }
        if n.show == Show::Power && s.show == Show::Root {
            return (s, Some(1.0));
        }
        (s, None)
    }

    /// Move t towards `to` by two eases over the clock's `dt` (going back), or at the
    /// playing rate over the scene's `scene_dt` (going on).
    fn move_t(&mut self, to: f64, scene_dt: f64, dt: f64, play: bool) {
        if play && to > self.t {
            self.t = (self.t + scene_dt.max(0.0) * PLAY_RATE).min(to);
            self.t_mid = self.t;
            return;
        }
        if dt <= 0.0 || !dt.is_finite() {
            return;
        }
        let k = 1.0 - (-dt / FOLLOW).exp();
        self.t_mid += (to - self.t_mid) * k;
        self.t += (self.t_mid - self.t) * k;
        if (to - self.t).abs().max((to - self.t_mid).abs()) < 1e-3 {
            (self.t, self.t_mid) = (to, to);
        }
    }

    /// Move everything on: `dt` seconds of the scene's time (playing, the tour) and
    /// `real_dt` of the clock's (going back, and the glides, seen through even while
    /// the scene's time stands still).
    fn advance(&mut self, dt: f32, real_dt: f32) {
        let (dt, real) = (dt as f64, real_dt as f64);
        for _ in 0..6 {
            let Some(s) = self.next else { break };
            if self.now == s && !self.afresh {
                self.next = None;
                break;
            }
            let (step, meet) = self.step_towards(s);
            if let Some(m) = meet {
                if self.t != m {
                    // (forwards at the playing rate, so that the path is seen; back quickly)
                    self.move_t(m, dt, real, true);
                    if self.t != m {
                        break;
                    }
                }
                // (a new path waits for the old one's lines to have faded out: the two
                // are never drawn one in place of the other)
                if m == 0.0 && self.fade > FADED {
                    break;
                }
            }
            if meet == Some(0.0) {
                self.afresh = false;
            }
            self.now = step;
        }
        if self.next.is_none() {
            match self.goal {
                Goal::Rest => {}
                Goal::Glide(to) => {
                    self.move_t(to, dt, real, false);
                    if self.t == to {
                        self.goal = Goal::Rest;
                    }
                }
                Goal::Play(stop) => {
                    if self.now.show == Show::Root || stop < self.t {
                        self.goal = Goal::Rest;
                    } else {
                        self.move_t(stop, dt, real, true);
                        if self.t >= stop {
                            self.goal = Goal::Rest;
                        }
                    }
                }
            }
        }
        // the tour of the roots: each in turn, then back to the first
        if let (Some(s), None, Show::Root) = (self.tour, self.next, self.now.show) {
            let n = roots(self.now.a).len().max(1);
            let s = s + dt as f32;
            let stop = (s / TOUR_STOP) as usize;
            self.now.root = if stop >= n { 0 } else { stop };
            self.tour = (stop < n).then_some(s);
        }
        self.aim();
        self.yellow.follow.step(real);
        self.blue.follow.step(real);
        let fade_to = if self.next.is_some_and(|s| s.a != self.now.a || s.winding != self.now.winding || self.afresh) { 0.0 } else { 1.0 };
        if self.fade != fade_to && real > 0.0 {
            self.fade += (fade_to - self.fade) * (1.0 - (-real / 0.1).exp());
            if (self.fade - fade_to).abs() < 2e-3 {
                self.fade = fade_to;
            }
        }
        let round_to = if self.now.show == Show::Power { 1.0 } else { 0.0 };
        if self.round != round_to && real > 0.0 {
            self.round += (round_to - self.round) * (1.0 - (-real / 0.12).exp());
            if (self.round - round_to).abs() < 2e-3 {
                self.round = round_to;
            }
        }
        self.plane.to = plane_fit(self.now.a);
        self.plane.step(real);
    }

    /// Point the drawn maps at what the setting asks for now.
    fn aim(&mut self) {
        let Setting { a, show, winding, root } = self.now;
        let [p, q, r, s] = bits(a);
        let key = [show as u64, root as u64, winding as i64 as u64, p, q, r, s, self.blend as u64];
        let (yellow, ya) = match show {
            Show::Power => power(a, winding, self.t).map_or((I, 0.0), |m| (m, 1.0)),
            Show::Root => (a, 1.0),
        };
        let (blue, ba) = match show {
            Show::Power => (blend(a, self.t), if self.blend { 1.0 } else { 0.0 }),
            Show::Root => roots(a).get(root).map_or((a, 0.0), |b| (*b, 1.0)),
        };
        let mut yellow_key = key;
        yellow_key[7] = 0;
        self.yellow.aim(within_limits_far(yellow), ya, yellow_key);
        self.blue.aim(within_limits_far(blue), ba, key);
    }

    fn animated_now(&self) -> bool {
        self.next.is_some() || self.goal != Goal::Rest || self.tour.is_some() || self.settle > 0.0 || !self.yellow.follow.arrived() || !self.blue.follow.arrived() || self.fade != 1.0 || self.round != if self.now.show == Show::Power { 1.0 } else { 0.0 } || !self.plane.arrived()
    }

    /// The canonical view: the picture's frame in the middle of the part of the
    /// canvas it has to itself.
    fn the_view(&self) -> View {
        let lay = Layout::of(self.canvas, self.inset);
        let (min, max) = lay.frame_room();
        let room = (max - min).max(Vec2::splat(60.0));
        let per_unit = (room.x / (2.0 * FRAME_HALF.x)).min(room.y / (2.0 * FRAME_HALF.y)).max(1e-3);
        // (screen y runs down, the scene's up)
        let off = ((min + max) * 0.5 - self.canvas * 0.5) / per_unit;
        View::Fit { center: Vec2::new(FRAME_CENTRE.x - off.x, FRAME_CENTRE.y + off.y), half: self.canvas.max(Vec2::ONE) * (0.5 / per_unit) }
    }
}

/// The matrix with every entry within ±`ENTRY_MAX` (and 0 for one that is not a number).
fn within_limits(m: DMat2) -> DMat2 {
    DMat2::from_cols_array(&m.to_cols_array().map(|v| if v.is_finite() { v.max(-ENTRY_MAX).min(ENTRY_MAX) } else { 0.0 }))
}

/// A matrix to draw: finite, and not so large that the drawing loses its precision (a
/// root of a matrix with two nearly equal eigenvalues can be very large, and is cut
/// off at the edge of the picture anyway).
fn within_limits_far(m: DMat2) -> DMat2 {
    DMat2::from_cols_array(&m.to_cols_array().map(|v| if v.is_finite() { v.max(-1e6).min(1e6) } else { 0.0 }))
}

/// The part of the complex plane that holds everything drawn there for A: 0 and the
/// unit circle, the eigenvalues λ, λ² and the blend's at t = 2 (2λ − 1), and for a
/// matrix whose logarithms turn, the circle of radius |λ|² that their paths may sweep.
/// Its centre and half its width and height.
fn plane_fit(a: DMat2) -> [f64; 4] {
    let lam = eigenvalues(a);
    let mut pts = vec![DVec2::new(-1.0, -1.0), DVec2::new(1.0, 1.0)];
    for l in lam {
        pts.extend([l, c_square(l), l * 2.0 - DVec2::X]);
    }
    if windings(a) {
        let r = lam[0].length_squared().max(1.0);
        pts.extend([DVec2::splat(-r), DVec2::splat(r)]);
    }
    let (lo, hi) = pts.iter().filter(|p| p.is_finite()).fold((DVec2::splat(f64::MAX), DVec2::splat(f64::MIN)), |(lo, hi), p| (lo.min(*p), hi.max(*p)));
    let mid = (lo + hi) * 0.5;
    let half = ((hi - lo) * 0.5 * 1.12).max(DVec2::splat(0.5)).min(DVec2::splat(80.0));
    [mid.x, mid.y, half.x, half.y]
}

// ----- where things go -----------------------------------------------------------------

/// The part of the scene the view is fitted to (its centre, and half its size).
const FRAME_CENTRE: Vec2 = Vec2::new(0.3, 0.3);
const FRAME_HALF: Vec2 = Vec2::new(2.5, 2.3);

/// The screen panel's parts, in logical pixels.
#[derive(Clone, Debug)]
struct Panel {
    min: Vec2,
    max: Vec2,
    /// The size of the numbers in the matrices.
    text: f32,
    /// The two tabs: the power and the square roots.
    tabs: [(Vec2, Vec2); 2],
    /// Top-left corner of the first row of matrices, and of the second (not on a phone).
    mats: Vec2,
    second: Option<Vec2>,
    /// Top-left of the two lines about areas.
    lines: Vec2,
    /// The row of the time track (or of the roots' buttons), and the row under it.
    controls: (Vec2, Vec2),
    choice: (Vec2, Vec2),
    /// The complex plane's title (not on a phone) and its box.
    plane_title: Option<Vec2>,
    plane: (Vec2, Vec2),
}

impl Panel {
    /// The two ends of the time track, at its middle, and the play button.
    fn bar(&self) -> ((Vec2, Vec2), (Vec2, Vec2)) {
        let (lo, hi) = self.controls;
        let play = (Vec2::new(hi.x - 30.0, lo.y), Vec2::new(hi.x, hi.y));
        let y = (lo.y + hi.y) * 0.5;
        ((Vec2::new(lo.x + 14.0, y), Vec2::new(play.0.x - 70.0, y)), play)
    }
}

/// `n` buttons side by side across the box `(lo, hi)`, at most `widest` wide each.
fn button_row((lo, hi): (Vec2, Vec2), n: usize, widest: f32) -> Vec<(Vec2, Vec2)> {
    let gap = 6.0;
    let w = ((hi.x - lo.x - gap * (n as f32 - 1.0)) / n.max(1) as f32).min(widest).max(10.0);
    (0..n).map(|i| (Vec2::new(lo.x + i as f32 * (w + gap), lo.y), Vec2::new(lo.x + i as f32 * (w + gap) + w, hi.y))).collect()
}

const LINE_H: f32 = 17.0;
const ROW_H: f32 = 26.0;

#[derive(Clone, Debug)]
struct Layout {
    /// A phone held upright: the panel goes below the picture.
    narrow: bool,
    /// The part of the canvas the picture has to itself.
    picture: (Vec2, Vec2),
    /// None on a canvas too small to spare the room (the numbers are in the side panel).
    panel: Option<Panel>,
    /// Where the caption goes: its left and right, and its foot.
    caption: Option<(f32, f32, f32)>,
}

impl Layout {
    fn of(view: Vec2, inset: f32) -> Layout {
        let narrow = view.x < 620.0 && view.y >= 560.0;
        let pad = if narrow { 10.0 } else { 12.0 };
        let width = if narrow { view.x } else { 340.0 };
        // the numbers' size: as large as lets "A = [ ]  Aᵗ = [ ]" fit across, up to 13.5
        let across = |s: f32| 2.0 * matrix_size(2, 2, CELL, s).x + 4.0 * s + 60.0;
        let mut text = 13.5f32;
        while across(text) > width - 2.0 * pad && text > 9.5 {
            text -= 0.5;
        }
        let mat_h = matrix_size(2, 2, CELL, text).y;
        // the rows of words and buttons under the matrices
        let block = 2.0 * LINE_H + 6.0 + ROW_H + 6.0 + ROW_H - 2.0;
        let whole = |caption| Layout { narrow, picture: (Vec2::new(0.0, inset), view), panel: None, caption };
        // the panel from its top-left corner `min`, `width` wide, with the complex plane
        // `plane` tall: under the rows on a wide canvas, beside them on a phone
        let place = |min: Vec2, width: f32, plane: f32| {
            let x = min.x + pad;
            let right = min.x + width - pad;
            let mut y = min.y + pad;
            let tab_w = ((right - x - 8.0) * 0.5).min(150.0);
            let tabs = [(Vec2::new(x, y), Vec2::new(x + tab_w, y + ROW_H - 4.0)), (Vec2::new(x + tab_w + 8.0, y), Vec2::new(x + 2.0 * tab_w + 8.0, y + ROW_H - 4.0))];
            y += ROW_H + 4.0;
            let mats = Vec2::new(x, y);
            y += mat_h + 8.0;
            let second = (!narrow).then(|| {
                let at = Vec2::new(x, y);
                y += mat_h + 8.0;
                at
            });
            let rows_right = if narrow { right - plane - 10.0 } else { right };
            let top = y;
            let lines = Vec2::new(x, y);
            y += 2.0 * LINE_H + 6.0;
            let controls = (Vec2::new(x, y), Vec2::new(rows_right, y + ROW_H));
            y += ROW_H + 6.0;
            let choice = (Vec2::new(x, y), Vec2::new(rows_right, y + ROW_H - 2.0));
            y += ROW_H + 6.0;
            let (plane_title, plane) = if narrow {
                (None, (Vec2::new(right - plane, top), Vec2::new(right, top + plane)))
            } else {
                y += 4.0;
                let title = Vec2::new(x, y);
                y += LINE_H + 2.0;
                (Some(title), (Vec2::new(x, y), Vec2::new(right, y + plane)))
            };
            let foot = plane.1.y.max(y);
            Panel { min, max: Vec2::new(min.x + width, foot + pad), text, tabs, mats, second, lines, controls, choice, plane_title, plane }
        };
        if narrow {
            // the panel below the picture, the plane beside the rows, the caption at its foot
            let plane = (view.x * 0.34).clamp(100.0, 150.0).max(block);
            let caption_h = CAPTION_LINES as f32 * 19.0 + 8.0;
            let rest = pad + ROW_H + 4.0 + mat_h + 8.0 + plane + pad + caption_h;
            // (the picture keeps at least the top 44 %)
            let below = (view.y - rest).max(view.y * 0.44);
            if view.y - below < rest - 1.0 {
                return whole(Some((0.0, view.x, view.y - 10.0)));
            }
            let mut panel = place(Vec2::new(0.0, below), view.x, plane);
            panel.max = view;
            return Layout { narrow, picture: (Vec2::new(0.0, inset), Vec2::new(view.x, below)), panel: Some(panel), caption: Some((0.0, view.x, view.y - 10.0)) };
        }
        let top = inset + 12.0;
        let above = pad + ROW_H + 4.0 + 2.0 * (mat_h + 8.0) + block + 6.0 + 4.0 + LINE_H + 2.0;
        let plane = (width - 2.0 * pad).min(view.y - 12.0 - top - above - pad).min(260.0);
        if view.x < 600.0 || plane < 100.0 {
            return whole(Some((0.0, view.x, view.y - 16.0)));
        }
        let panel = place(Vec2::new(view.x - width - 12.0, top), width, plane);
        let picture = (Vec2::new(0.0, inset), Vec2::new(panel.min.x - 8.0, view.y));
        Layout { narrow, picture, panel: Some(panel), caption: Some((0.0, picture.1.x, view.y - 16.0)) }
    }

    /// The part of the canvas the scene's frame is fitted into: the picture's, less the
    /// heading at its top and the caption at its foot.
    fn frame_room(&self) -> (Vec2, Vec2) {
        let (min, max) = self.picture;
        let foot = if self.narrow { 6.0 } else { 56.0 };
        (min + Vec2::new(12.0, 58.0), max - Vec2::new(12.0, foot))
    }
}

/// The most lines the caption takes on a phone.
const CAPTION_LINES: usize = 2;

/// How wide a text is on the screen: one with mathematics in it exactly (and kept from
/// frame to frame), words alone as `text_width` judges. None for one with mathematics
/// in it before the fonts are in place (the first frame).
fn measured(f: &Frame, text: &str, size: f32) -> Option<f32> {
    if crate::typeset::has_math(text) {
        crate::typeset::measure(f.ctx, text, size).map(|s| s.x)
    } else {
        Some(text_width(text, size))
    }
}

/// The same, and before the fonts are in place as `text_width` judges it.
fn width(f: &Frame, text: &str, size: f32) -> f32 {
    measured(f, text, size).unwrap_or_else(|| text_width(text, size))
}

/// A matrix of numbers on the panel, as `hud_matrix` draws it, with a number too wide
/// for its cell drawn smaller, to fit: the brackets stay where they are.
fn matrix(f: &mut Frame, top_left: Vec2, rows: &[Vec<(String, Color)>], size: f32) -> Vec2 {
    let room = matrix_cell(CELL, size).x - 2.0;
    let mut fitted = rows.to_vec();
    let mut smaller = Vec::new();
    for (i, row) in fitted.iter_mut().enumerate() {
        for (j, (text, color)) in row.iter_mut().enumerate() {
            if !text.contains('e') {
                continue;
            }
            let wide = width(f, text, size);
            if wide > room {
                smaller.push((i, j, std::mem::take(text), *color, size * room / wide));
            }
        }
    }
    let total = hud_matrix(f, top_left, &fitted, CELL, size, pal::DIM);
    for (i, j, text, color, s) in smaller {
        let (lo, hi) = matrix_entry(top_left, CELL, size, i, j);
        f.text_px((lo + hi) * 0.5, text, s, color, Align2::CENTER_CENTER);
    }
    total
}

/// A matrix's entries for `matrix`, its columns in the colours given.
fn entries(m: DMat2, cols: [Color; 2]) -> Vec<Vec<(String, Color)>> {
    (0..2).map(|i| (0..2).map(|j| (cell(m.col(j)[i]), cols[j])).collect()).collect()
}

fn inside_rect(p: Vec2, (min, max): (Vec2, Vec2)) -> bool {
    p.cmpge(min).all() && p.cmple(max).all()
}

// ----- drawing what may run far off ----------------------------------------------------

/// The part of the segment from `a` to `b` inside the box `lo..hi`, if any.
fn clip(a: DVec2, b: DVec2, lo: DVec2, hi: DVec2) -> Option<(DVec2, DVec2)> {
    if !(a.is_finite() && b.is_finite()) {
        return None;
    }
    let d = b - a;
    let (mut t0, mut t1) = (0.0f64, 1.0f64);
    for (p, q) in [(-d.x, a.x - lo.x), (d.x, hi.x - a.x), (-d.y, a.y - lo.y), (d.y, hi.y - a.y)] {
        if p == 0.0 {
            if q < 0.0 {
                return None;
            }
        } else {
            let r = q / p;
            if p < 0.0 {
                t0 = t0.max(r);
            } else {
                t1 = t1.min(r);
            }
        }
    }
    (t0 <= t1).then(|| (if t0 <= 0.0 { a } else { a + d * t0 }, if t1 >= 1.0 { b } else { a + d * t1 }))
}

/// The convex polygon cut down to the box `lo..hi`.
fn clip_polygon(polygon: &[DVec2], lo: DVec2, hi: DVec2) -> Vec<DVec2> {
    if !polygon.iter().all(|p| p.is_finite()) {
        return Vec::new();
    }
    let mut out = polygon.to_vec();
    for (axis, bound, above) in [(0, lo.x, true), (0, hi.x, false), (1, lo.y, true), (1, hi.y, false)] {
        let input = std::mem::take(&mut out);
        let n = input.len();
        let keep = |p: DVec2| if above { p[axis] >= bound } else { p[axis] <= bound };
        let cross = |p: DVec2, q: DVec2| {
            let mut c = p.lerp(q, (bound - p[axis]) / (q[axis] - p[axis]));
            c[axis] = bound;
            c
        };
        for i in 0..n {
            let (prev, cur) = (input[(i + n - 1) % n], input[i]);
            if keep(cur) {
                if !keep(prev) {
                    out.push(cross(prev, cur));
                }
                out.push(cur);
            } else if keep(prev) {
                out.push(cross(prev, cur));
            }
        }
    }
    out
}

/// A path's points kept to the box `lo..hi` (the box reaches a view's width beyond the
/// view on every side, so what is moved is out of sight; and unlike cutting the path,
/// this keeps the number of points the same as the path swings in and out of it).
fn within(points: impl IntoIterator<Item = DVec2>, lo: DVec2, hi: DVec2) -> Vec<Vec3> {
    points.into_iter().map(|p| if p.is_finite() { p.clamp(lo, hi) } else { lo }.as_vec2().extend(0.0)).collect()
}

/// The parts of the picture drawn as lines whose number of points changes as the
/// picture moves: each has a batch of its own (different from the usual one by a hair
/// of its width), so that from one frame to the next the parts that stay the same can
/// still be compared point for point (that is how `eidos motion` tells smooth motion
/// from jumps).
#[derive(Clone, Copy, PartialEq)]
enum Part {
    /// The unit circle where it starts, dashed: its dashes are as many as its size on
    /// the screen makes them.
    Start = 1,
    /// The paths of (1, 0) and (0, 1) so far, which grow with t.
    Travelled,
    /// The same paths still ahead.
    Ahead,
    /// The square roots' arrows and the path of the corner.
    Roots,
    /// The marks that show which way a path goes.
    Marks,
    /// The principal path of the corner, dashed, when another logarithm is chosen.
    Principal,
    /// Marks in the complex plane that come and go with the matrix: rings, the
    /// blend's lines, the roots' lines.
    PlaneMarks,
    /// In the complex plane: the eigenvalues' paths so far, and ahead.
    PlaneTravelled,
    PlaneAhead,
    /// The tabs and the buttons on the panel, which come and go.
    Buttons,
}

/// Every part, in the order their batches are made at the start of each frame: so that
/// a batch is the same batch from one frame to the next (in the list of batches)
/// whichever parts are drawn.
const PARTS: [Part; 10] = [Part::Start, Part::Travelled, Part::Ahead, Part::Roots, Part::Marks, Part::Principal, Part::PlaneMarks, Part::PlaneTravelled, Part::PlaneAhead, Part::Buttons];

/// The points marking t = 1/2 on the corner's path, a batch of their own.
fn half_marks<'a>(f: &'a mut Frame) -> &'a mut crate::gfx::draw::PointBatch {
    f.dl.points_with(PointKind::Disc, PointStyle { outline_px: 1.25 + 1e-4, ..Default::default() })
}

fn lines_for<'a>(f: &'a mut Frame, part: Part) -> &'a mut crate::gfx::draw::LineBatch {
    let kind = if matches!(part, Part::PlaneTravelled | Part::PlaneAhead | Part::Buttons | Part::PlaneMarks) { LineKind::Overlay } else { LineKind::Blend };
    f.dl.lines_with(kind, LineStyle { width_scale: 1.0 + 1e-4 * part as u8 as f32, depth_bias: if kind == LineKind::Overlay { 0.0 } else { LineStyle::default().depth_bias }, ..Default::default() })
}

/// An arrow from `a` to `b` (its shaft in the batch of `part`, or the usual one), the
/// part of it in the box `lo..hi`: without its head if the head is outside.
#[allow(clippy::too_many_arguments)]
fn arrow_in(f: &mut Frame, part: Option<Part>, a: DVec2, b: DVec2, lo: DVec2, hi: DVec2, width: f32, c: Color) {
    let Some((p, q)) = clip(a, b, lo, hi) else { return };
    let (p, q) = (p.as_vec2().extend(0.0), q.as_vec2().extend(0.0));
    let d = q - p;
    let len = d.length();
    if len < 1e-9 {
        return;
    }
    let dir = d / len;
    let mut end = q;
    if clip(a, b, lo, hi).is_some_and(|(_, q)| q == b) {
        let wpp = f.cam.world_per_px(q) * f.px;
        let head = ((width * 3.2 + 5.0) * wpp).min(len * 0.6);
        let side = dir.cross(Vec3::Z).normalize_or_zero() * (head * 0.42);
        let base = q - dir * head;
        f.dl.tris_alpha.tri(q, base + side, base - side, c);
        end = q - dir * (head * 0.8);
    }
    match part {
        Some(part) => lines_for(f, part).segment(p, end, width, c),
        None => f.dl.lines.segment(p, end, width, c),
    }
}

// ----- the letter F --------------------------------------------------------------------

/// The letter F's three bars (the stem, the top arm, the middle arm), each a rectangle
/// [x0, y0, x1, y1].
const BARS: [[f64; 4]; 3] = [[0.35, 0.15, 0.6, 1.4], [0.6, 1.18, 1.15, 1.4], [0.6, 0.68, 0.97, 0.88]];
/// Its outline, corner by corner.
const OUTLINE: [[f64; 2]; 10] = [[0.35, 0.15], [0.6, 0.15], [0.6, 0.68], [0.97, 0.68], [0.97, 0.88], [0.6, 0.88], [0.6, 1.18], [1.15, 1.18], [1.15, 1.4], [0.35, 1.4]];
/// The corner whose journey the square roots show: the tip of the top arm.
const CORNER: DVec2 = DVec2::new(1.15, 1.29);
/// Points round the unit circle.
const ROUND: usize = 72;
/// Points on each path of (1, 0) and (0, 1), from t = 0 to 2.
const TRACK: usize = 400;
/// And on each path of an eigenvalue, in the complex plane.
const PLANE_TRACK: usize = 240;

fn unit_circle() -> impl Iterator<Item = DVec2> {
    (0..=ROUND).map(|i| DVec2::from_angle(i as f64 / ROUND as f64 * std::f64::consts::TAU))
}

/// Points of x under A^t = e^(tL) at t = 0, h, 2h, …, T_MAX, h = T_MAX / n.
fn track(l: DMat2, x: DVec2, n: usize) -> Vec<DVec2> {
    let step = expm(l, T_MAX / n as f64);
    let mut p = x;
    let mut out = Vec::with_capacity(n + 1);
    out.push(p);
    for _ in 0..n {
        p = step * p;
        out.push(p);
    }
    out
}

/// The last point of a track of `n` steps at or before the time t.
fn before(t: f64, n: usize) -> usize {
    ((t / T_MAX * n as f64).floor().max(0.0) as usize).min(n)
}

// ----- the scene's frame ---------------------------------------------------------------

/// The names written before the panel's matrices.
fn leads(show: Show) -> [&'static str; 4] {
    match show {
        Show::Power => ["$A =$", "$A^t =$", "$L =$", ""],
        Show::Root => ["$A =$", "$B =$", "$B^2 =$", ""],
    }
}

/// The matrix editor: four numbers to drag or type, within ±`ENTRY_MAX`. Returns
/// (changed, typed).
fn matrix_edit(ui: &mut egui::Ui, m: &mut DMat2) -> (bool, bool) {
    let (mut changed, mut typed) = (false, false);
    let mut a = [[m.x_axis.x, m.y_axis.x], [m.x_axis.y, m.y_axis.y]];
    egui::Grid::new("matfun_a").spacing([6.0, 4.0]).show(ui, |ui| {
        for row in &mut a {
            for v in row.iter_mut() {
                let r = ui.add(egui::DragValue::new(v).speed(0.01).fixed_decimals(2).range(-ENTRY_MAX..=ENTRY_MAX));
                changed |= r.changed();
                typed |= r.changed() && !r.dragged();
            }
            ui.end_row();
        }
    });
    if changed {
        *m = rows([a[0][0], a[0][1], a[1][0], a[1][1]]);
    }
    (changed, typed)
}

impl Scene for Matfun {
    fn view(&self) -> View {
        self.the_view()
    }

    fn take_view(&mut self) -> Option<View> {
        let new = std::mem::take(&mut self.new_view);
        if new {
            self.settle = SETTLE;
        }
        new.then(|| self.the_view())
    }

    fn settings(&self, s: &mut ViewSettings) {
        s.grid_alpha = 0.10;
        s.axis_alpha = 0.35;
    }

    fn animated(&self) -> bool {
        self.animated_now()
    }

    fn ui(&mut self, ui: &mut egui::Ui, _: &Gfx) {
        ui::note(ui, r"What could $A^{1/2}$ mean, or $A^t$? If $A = VDV^{-1}$, with its eigenvectors in $V$ and its eigenvalues in $D$, a function of $A$ is the function applied to each eigenvalue: $f(A) = Vf(D)V^{-1}$. For every $t$ at once, the way is a logarithm $L$ of $A$, with $e^L = A$:");
        ui::equation(ui, r"A^t = e^{tL}");
        ui::note(ui, r"a path of maps from $I$ at $t = 0$ to $A$ at $t = 1$ (the yellow letter). Halfway along it is a square root $B$: $B^2 = A$. Some matrices have many of each, and some none.");

        ui::section(ui, "Things to try");
        for (k, trial) in TRIALS.iter().enumerate() {
            if ui::experiment(ui, trial.label, trial.look_for) {
                self.set_up(k);
            }
        }

        ui::section(ui, "Show");
        let asked = self.next.unwrap_or(self.now);
        let mut show = asked.show as usize;
        if ui::choice(ui, &["the power $A^t$", "square roots"], &mut show) {
            let s = Setting { show: if show == 0 { Show::Power } else { Show::Root }, ..asked };
            self.ask(s, false, Goal::Rest);
        }
        if asked.show == Show::Power {
            ui.horizontal(|ui| {
                if ui.button(if self.playing() { "❚❚ pause" } else { "▶ play" }).clicked() {
                    self.play_or_pause();
                }
            });
            let mut t = self.t as f32;
            if ui::slider(ui, "time $t$", &mut t, 0.0..=T_MAX as f32) {
                self.set_t(t as f64);
            }
            ui::note(ui, r"$t = 0$ is $I$, $t = 1$ is $A$, $t = 2$ is $A^2$; the knob on the picture's panel does the same.");
            if windings(asked.a) {
                let names: Vec<String> = [-1, 0, 1].iter().map(|&k| log_angle(asked.a, k).map_or(String::new(), degrees)).collect();
                let labels: Vec<&str> = names.iter().map(String::as_str).collect();
                let mut pick = (asked.winding + 1).clamp(0, 2) as usize;
                ui::note(ui, "The angle of the logarithm $L$: how far $A^t$ turns from $t = 0$ to 1.");
                if ui::choice(ui, &labels, &mut pick) {
                    self.ask(Setting { winding: pick as i32 - 1, ..asked }, false, Goal::Play(self.t.max(1e-9)));
                }
            }
            ui::checkbox(ui, &mut self.blend, r"the straight blend $(1 - t)I + tA$");
        } else {
            let found = roots(asked.a);
            if found.is_empty() {
                ui::note(ui, "This matrix has no real square root.");
            } else {
                let names: Vec<String> = (1..=found.len()).map(|k| format!("$B_{k}$")).collect();
                let labels: Vec<&str> = names.iter().map(String::as_str).collect();
                let mut pick = asked.root.min(found.len() - 1);
                if ui::choice(ui, &labels, &mut pick) {
                    self.ask(Setting { root: pick, ..asked }, false, Goal::Rest);
                }
                if counts(asked.a).0 == Count::Many {
                    ui::note(ui, "Four of the square roots, of infinitely many.");
                }
            }
        }

        ui::section(ui, "Matrix $A$");
        let names: Vec<&str> = PRESETS.iter().map(|p| p.0).collect();
        if let Some(i) = ui::buttons(ui, &names) {
            self.ask_matrix(rows((PRESETS[i].1)()));
        }
        let mut a = asked.a;
        let (changed, typed) = matrix_edit(ui, &mut a);
        if changed {
            if typed {
                self.ask_matrix(a);
            } else {
                // (a number dragged changes the path as it goes)
                if let Some(s) = self.next.take() {
                    self.now = s;
                }
                self.now = Setting { a: within_limits(a), winding: if windings(a) { self.now.winding } else { 0 }, ..self.now };
                self.now.root = self.now.root.min(roots(self.now.a).len().saturating_sub(1));
                (self.afresh, self.tour) = (false, None);
            }
        }
        if let Some(p) = PRESETS.iter().find(|p| rows((p.1)()) == asked.a) {
            ui::note(ui, p.2);
        }

        ui::section(ui, "Reading the picture");
        ui::key(
            ui,
            &[
                (pal::YELLOW, r"yellow: a letter and the unit circle carried by $A^t$ (with the square roots, by $B^2 = A$); dashed, where they start"),
                (pal::BLUE, r"blue: the same carried by the straight blend $(1 - t)I + tA$, and its eigenvalues; with the square roots, the root $B$ once, and its eigenvalues"),
                (pal::WHITE, r"white: the path of the letter's corner under $A^t$; dashed, the principal path, when another logarithm is chosen"),
                (pal::PINK, r"pink: $(1, 0)$ carried along, and its path as $t$ goes from 0 to 2: the first column of $A^t$ (or of $B$)"),
                (pal::GREEN, "green: $(0, 1)$, and the second column"),
                (pal::ORANGE, r"orange and purple: the eigenlines of $A$, dashed (when its eigenvalues are real); in the complex plane, the eigenvalues $\lambda^t$ of $A^t$, and rings at $\lambda$"),
            ],
        );

        ui::section(ui, "Words");
        ui::term(ui, "$A^t$:", r"for $A$ with a real logarithm $L$, the map $e^{tL}$: $I$ at $t = 0$, $A$ at $t = 1$, $A^2$ at $t = 2$, and $A^s A^t = A^{s + t}$ all along. It is the flow whose time-1 map is $A$; its eigenvalues are $\lambda^t = e^{t\mu}$, $\mu$ those of $L$.");
        ui::term(ui, "Logarithm:", r"a matrix $L$ with $e^L = A$. When $A$ has complex eigenvalues $\lambda = r(\cos\theta \pm i \sin\theta)$, a real $L$ has the eigenvalues $\ln r \pm i\theta$, and $\theta$ can be changed by any whole number of turns: the same $A$, another path. Two different positive eigenvalues allow one real $L$ only.");
        ui::term(ui, "Square root:", r"a matrix $B$ with $B^2 = A$. It shares $A$'s eigenvectors when $A$ has two different eigenvalues, and has $\pm\sqrt{\lambda}$ for eigenvalues.");
        ui::term(ui, "Principal:", r"the square root whose eigenvalues lie to the right of 0, and the logarithm whose eigenvalues have imaginary parts between $-\pi$ and $\pi$. Both exist, and are real, when no eigenvalue of $A$ is 0 or negative; the root is then $e^{L/2} = A^{1/2}$.");
        ui::term(ui, "$f(A) = Vf(D)V^{-1}$:", r"$f$ applied to each eigenvalue, in a basis of eigenvectors. For an $f$ given by a power series it agrees with the series (the scene on $e^{At}$); a matrix without such a basis (a shear) needs the series, or a limit.");
        ui::term(ui, "Straight blend:", r"$(1 - t)I + tA$, the straight line from $I$ to $A$ among matrices. Its eigenvalues $(1 - t) + t\lambda$ move on straight lines, and where one passes 0 the blend is flat.");
        ui::term(ui, "$J$:", r"the quarter turn, $(x, y) \mapsto (-y, x)$; $e^{\theta J}$ is the turn by $\theta$ (in radians).");
        ui::term(ui, r"$\log e^L$:", r"$e^{\log A} = A$ for every logarithm, but the principal $\log e^L$ is $L$ only while the imaginary parts of $L$'s eigenvalues stay between $-\pi$ and $\pi$: $e^{-190^\circ J}$ is the turn by 170°, whose principal logarithm is $170^\circ \times J$.");

        ui::section(ui, "Numbers");
        let s = self.now;
        let lam = eigenvalues(s.a);
        let eig = |z: Cx| if z.y.abs() < 1e-12 { num(z.x, 2) } else { format!(r"{} \pm {}i", num(z.x, 2), num(z.y.abs(), 2)) };
        let lam_text = if lam[0].y.abs() > 1e-12 { format!("${}$", eig(lam[0])) } else { format!("${}, {}$", num(lam[0].x, 2), num(lam[1].x, 2)) };
        ui::readout(ui, "eigenvalues of $A$", lam_text);
        ui::readout(ui, r"$\det A$", format!("${}$", short(s.a.determinant())));
        match s.show {
            Show::Power => {
                ui::readout(ui, "$t$", format!("${}$", num(self.t, 2)));
                match (power(s.a, s.winding, self.t), log_of(s.a, s.winding)) {
                    (Some(m), Some(l)) => {
                        ui::readout(ui, r"$\det A^t$", format!("${}$", short(m.determinant())));
                        ui::readout(ui, r"$(\det A)^t$", format!("${}$", short(s.a.determinant().powf(self.t))));
                        ui::readout(ui, r"$e^L$ against $A$, off by", format!("${}$", short(size(expm(l, 1.0) - s.a))));
                    }
                    _ => ui::readout(ui, r"$A^t$", "none (no real logarithm)"),
                }
                ui::readout(ui, r"$\det$ of the blend", format!("${}$", short(blend(s.a, self.t).determinant())));
            }
            Show::Root => {
                if let Some(b) = roots(s.a).get(s.root) {
                    ui::readout(ui, r"$\det B$", format!("${}$", short(b.determinant())));
                    ui::readout(ui, r"$B^2$ against $A$, off by", format!("${}$", short(size(*b * *b - s.a))));
                }
            }
        }
        ui::see_also(ui, &[("expm", "the matrix exponential, and flows"), ("basis", "a matrix in the basis of its eigenvectors"), ("powers", "whole powers, step by step"), ("eigen", "finding eigenvalues"), ("jordan", "when eigenvectors run out")]);
    }

    fn frame(&mut self, f: &mut Frame) {
        let view = f.view_size();
        let lay = Layout::of(view, f.top_inset);
        let before = Layout::of(self.canvas, self.inset);
        // (a canvas squeezed to a sliver is one the controls have opened over, on a
        // phone: the picture is not seen then, and is left as it was for when they close)
        let covered = view.x < 160.0;
        let changed = !covered && (lay.narrow != before.narrow || lay.panel.is_some() != before.panel.is_some() || (view - self.canvas).abs().max_element() > 24.0);
        if !covered && (changed || !self.measured) {
            self.new_view |= changed;
            (self.canvas, self.measured) = (view, true);
        }
        if !covered {
            self.new_view |= (f.top_inset - self.inset).abs() > 1.0;
            self.inset = f.top_inset;
        }

        let pointer = f.ptr.pos.map(|p| p / f.px);
        if let Some(panel) = lay.panel.as_ref().filter(|_| !covered) {
            if pointer.is_some_and(|p| inside_rect(p, (panel.min, panel.max))) {
                // the panel is for its own handles and buttons: no moving the view from there
                f.handles.capture();
                f.handles.claim_clicks();
            }
            self.panel_input(f, panel);
        }
        self.advance(f.dt, f.real_dt);

        let labels_before = f.dl.labels.len();
        let plates = self.picture(f, &lay);
        let numbers_from = f.dl.labels.len();
        if self.settle <= 0.0 {
            f.axes2();
        }
        self.settle = (self.settle - f.real_dt.max(0.0)).max(0.0);
        // no number of the axes is written over the heading or the caption
        let mut k = 0;
        f.dl.labels.retain(|l| {
            k += 1;
            k <= numbers_from || !plates.iter().flatten().any(|plate| overlap(label_box(l), *plate))
        });
        if covered {
            f.dl.labels.truncate(labels_before);
            return;
        }
        // nothing written into the picture may show through the panel
        if let Some(p) = &lay.panel {
            let (lo, hi) = (p.min - Vec2::splat(2.0), p.max + Vec2::splat(2.0));
            let mut k = 0;
            f.dl.labels.retain(|l| {
                k += 1;
                k <= labels_before || l.backed || !inside_rect(l.pos, (lo, hi))
            });
            self.panel(f, p);
        }
    }
}

impl Matfun {
    /// The panel's handles and buttons: the tabs, the time knob, the track, the play
    /// button, the logarithms' and the roots' buttons.
    fn panel_input(&mut self, f: &mut Frame, panel: &Panel) {
        let pointer = f.ptr.pos.map(|p| p / f.px);
        let clicked = f.clicked_canvas();
        let asked = self.next.unwrap_or(self.now);
        if asked.show == Show::Power {
            let ((b0, b1), play) = panel.bar();
            let knob = b0 + (b1 - b0) * (self.t / T_MAX).clamp(0.0, 1.0) as f32;
            if let Some(mut w) = f.cam.pick_ground(knob * f.px).map(|p| p.truncate()) {
                if f.drag2(1, &mut w, pal::WHITE) {
                    if let Some(p) = on_screen(f, w.extend(0.0)) {
                        self.set_t(((p.x - b0.x) / (b1.x - b0.x).max(1.0)) as f64 * T_MAX);
                    }
                }
            }
            if let Some(p) = pointer.filter(|_| clicked) {
                if inside_rect(p, play) {
                    self.play_or_pause();
                } else if inside_rect(p, (b0 - Vec2::new(8.0, 12.0), b1 + Vec2::new(8.0, 12.0))) {
                    // a click on the track: t glides there
                    let t = (((p.x - b0.x) / (b1.x - b0.x).max(1.0)) as f64 * T_MAX).clamp(0.0, T_MAX);
                    if let Some(s) = self.next.take() {
                        self.now = s;
                    }
                    (self.goal, self.afresh, self.tour) = (Goal::Glide(t), false, None);
                } else if windings(asked.a) {
                    for (k, rect) in button_row(panel.choice, 3, 70.0).into_iter().enumerate() {
                        if inside_rect(p, rect) {
                            let winding = [-1, 0, 1][k];
                            if winding != asked.winding {
                                self.ask(Setting { winding, ..asked }, false, Goal::Play(self.t.max(1e-9)));
                            }
                        }
                    }
                }
            }
        } else if let Some(p) = pointer.filter(|_| clicked) {
            let n = roots(asked.a).len();
            for (k, rect) in button_row(panel.controls, n, 58.0).into_iter().enumerate() {
                if inside_rect(p, rect) && k != asked.root {
                    self.ask(Setting { root: k, ..asked }, false, Goal::Rest);
                }
            }
        }
        if let Some(p) = pointer.filter(|_| clicked) {
            for (k, rect) in panel.tabs.iter().enumerate() {
                let show = if k == 0 { Show::Power } else { Show::Root };
                if inside_rect(p, *rect) && show != asked.show {
                    self.ask(Setting { show, ..asked }, false, Goal::Rest);
                }
            }
        }
    }

    /// Everything drawn in the scene's own coordinates, and the names written beside it.
    /// Returns the boxes of the heading and of the caption, which are written over it.
    fn picture(&mut self, f: &mut Frame, lay: &Layout) -> [Option<(Vec2, Vec2)>; 2] {
        let (lo, hi) = f.cam.flat_bounds();
        let span = (hi - lo).as_dvec2();
        let (box_lo, box_hi) = (lo.as_dvec2() - span, hi.as_dvec2() + span);
        let Setting { a, show, winding, root } = self.now;
        let fade = self.fade as f32;
        // (the lines of a path are drawn only while they have not faded out: an old
        // path is gone before a new one comes)
        let paths = self.fade > FADED;
        let (ym, ya) = (self.yellow.matrix(), self.yellow.alpha());
        let (bm, ba) = (self.blue.matrix(), self.blue.alpha());

        for part in PARTS {
            lines_for(f, part);
        }
        half_marks(f);
        let mut room = Room::new(f);
        keep_ticks_clear(f, &mut room);
        if let Some(p) = &lay.panel {
            room.wall(p.min - Vec2::splat(4.0), p.max + Vec2::splat(4.0));
        }
        let heading = self.heading(f, lay);
        room.wall(heading.0, heading.1);
        let mut plates = [Some(heading), None];
        if let Some((left, right, bottom)) = lay.caption {
            let parts: &[&str] = match (show, lay.narrow) {
                (Show::Power, true) => &["dashed: the letter and the circle at $t = 0$", "pink, green, white: paths of $(1, 0)$, $(0, 1)$, the corner"],
                (Show::Power, false) => &["dashed: the letter and the unit circle at $t = 0$", "pink, green and white: the paths of $(1, 0)$, $(0, 1)$ and the letter's corner, $t$ from 0 to 2"],
                (Show::Root, true) => &["dashed: the letter at the start", "blue: $B$ once; yellow: $B$ twice"],
                (Show::Root, false) => &["dashed: the letter and the unit circle at the start", "blue: $B$ applied once; yellow: twice, $B^2 = A$", "white: the path of the corner under $A^t$, $t$ from 0 to 1"],
            };
            if let Some(plate) = caption(f, parts, (show, lay.narrow), (left, right), bottom, &mut self.caption) {
                room.wall(plate.0, plate.1);
                plates[1] = Some(plate);
            }
        }

        // the eigenlines of A, which every A^t and every square root keeps
        let reach = (hi - lo).length() as f64;
        let lines: Vec<(f64, DVec2, Color, u32)> = match kind(a) {
            Kind::Real { l1, l2, v1, v2 } => vec![(l1, v1, pal::ORANGE, 0), (l2, v2, pal::PURPLE, 1)],
            Kind::Defective { l, v, .. } => vec![(l, v, pal::ORANGE, 0)],
            _ => Vec::new(),
        };
        for &(_, v, c, _) in lines.iter().filter(|_| paths) {
            if let Some((p, q)) = clip(-v * reach, v * reach, box_lo, box_hi) {
                f.dashed(p.as_vec2(), q.as_vec2(), 1.4, c.a(0.75 * fade));
                room.line(f, p.as_vec2().extend(0.0), q.as_vec2().extend(0.0));
            }
        }

        // the F and the circle where they start
        let outline = |m: DMat2| -> Vec<DVec2> { OUTLINE.iter().map(|p| m * DVec2::from(*p)).collect() };
        let start = outline(I);
        for k in 0..start.len() {
            f.dashed(start[k].as_vec2(), start[(k + 1) % start.len()].as_vec2(), 1.1, pal::YELLOW.a(0.4));
        }
        let circle: Vec<DVec2> = unit_circle().collect();
        dashed(f, Part::Start, &circle, 48, 1.0, pal::YELLOW.a(0.3));

        // the paths of (1, 0), (0, 1) and the F's corner under A^t, so far and ahead; and
        // the corner's principal path from I to A, dashed, when another logarithm is chosen
        let log = log_of(a, winding);
        if let (Show::Power, Some(l), true) = (show, log, paths) {
            let k = before(self.t, TRACK);
            for (x, c, w) in [(DVec2::X, pal::PINK, 1.4), (DVec2::Y, pal::GREEN, 1.4), (CORNER, pal::WHITE, 1.8)] {
                let points = track(l, x, TRACK);
                let end = ym * x;
                lines_for(f, Part::Travelled).polyline(within(points[..=k].iter().copied().chain([end]), box_lo, box_hi), w, c.a(0.75 * fade));
                lines_for(f, Part::Ahead).polyline(within([end].into_iter().chain(points[k + 1..].iter().copied()), box_lo, box_hi), w * 0.7, c.a(0.28 * fade));
            }
            // which way the corner goes: a mark halfway along what it has travelled
            let wpp = f.cam.world_per_px(Vec3::ZERO) * f.px;
            if self.t > 0.05 {
                // (where the corner was at half the time, and its velocity there, L x)
                let half = expm(l, 0.5 * self.t) * CORNER;
                chevron(f, half, l * half, wpp, pal::WHITE.a(0.9 * fade));
            }
            if winding != 0 {
                if let Some(l0) = log_of(a, 0) {
                    // (e^(uL/2), u from 0 to 2, is the path to t = 1)
                    let principal = track(l0 * 0.5, CORNER, TRACK / 2);
                    let half = expm(l0, 0.5) * CORNER;
                    chevron(f, half, l0 * half, wpp, pal::WHITE.a(0.7 * fade));
                    let principal: Vec<DVec2> = principal.into_iter().map(|p| p.clamp(box_lo, box_hi)).collect();
                    dashed(f, Part::Principal, &principal, 36, 1.4, pal::WHITE.a(0.55 * fade));
                }
            }
            let at = ym * CORNER;
            if inside_box(at, box_lo, box_hi) && ya > 0.002 {
                f.dot(at.as_vec2(), 8.0, pal::WHITE.a(ya));
            }
            let path: Vec<Vec3> = track(l, CORNER, 40).iter().take(1 + (self.t / T_MAX * 40.0) as usize).map(|p| p.as_vec2().extend(0.0)).collect();
            for w in path.windows(2) {
                room.line(f, w[0], w[1]);
            }
        }

        // the shapes: blue under yellow
        // (both always, if unseen: what is drawn then keeps its number of points as the
        // picture changes from the power to the roots and back)
        for (m, alpha, c) in [(bm, ba, pal::BLUE), (ym, ya, pal::YELLOW)] {
            draw_shape(f, m, c, alpha, self.round as f32, box_lo, box_hi);
            if alpha > 0.3 {
                let corners: Vec<Vec3> = OUTLINE.iter().map(|p| (m * DVec2::from(*p)).as_vec2().extend(0.0)).collect();
                for k in 0..corners.len() {
                    room.line(f, corners[k], corners[(k + 1) % corners.len()]);
                }
                if show == Show::Power {
                    let ring: Vec<Vec3> = unit_circle().step_by(6).map(|p| (m * p).as_vec2().extend(0.0)).collect();
                    for w in ring.windows(2) {
                        room.line(f, w[0], w[1]);
                    }
                }
            }
        }

        // the columns of the matrix the pink and green are: A^t, or the root B
        let cols = if show == Show::Root { bm } else { ym };
        let cols_alpha = if show == Show::Root { ba } else { ya };
        {
            for (x, c) in [(DVec2::X, pal::PINK), (DVec2::Y, pal::GREEN)] {
                let tip = cols * x;
                arrow_in(f, None, DVec2::ZERO, tip, box_lo, box_hi, 2.6, c.a(cols_alpha));
                room.line(f, Vec3::ZERO, tip.as_vec2().extend(0.0));
            }
        }

        // a square root acting twice: the corner p to Bp, and Bp to B²p = Ap; and the
        // path of p under A^t to t = 1, if the root lies on one, halfway along it
        let mut arrows = Vec::new();
        if show == Show::Root && ba > 0.002 {
            let (p, bp, ap) = (CORNER, bm * CORNER, ym * CORNER);
            if let Some(k) = roots(a).get(root).and_then(|b| path_through(a, *b)).filter(|_| paths) {
                if let Some(l) = log_of(a, k) {
                    let n = TRACK / 2;
                    // (e^(uL/2) for u from 0 to 2 is A^t for t from 0 to 1)
                    let points = track(l * 0.5, p, n);
                    lines_for(f, Part::Roots).polyline(within(points.iter().copied(), box_lo, box_hi), 1.4, pal::WHITE.a(0.55 * ba * fade));
                    // the middle of the path, t = 1/2: where B takes p
                    let mid = points[n / 2];
                    if inside_box(mid, box_lo, box_hi) {
                        half_marks(f).push(mid.as_vec2().extend(0.0), 7.0, pal::WHITE.a(0.8 * ba * fade));
                    }
                }
            }
            for (from, to) in [(p, bp), (bp, ap)] {
                arrow_in(f, Some(Part::Roots), from, to, box_lo, box_hi, 2.2, pal::BLUE.mix(pal::TEXT, 0.2).a(ba));
                room.line(f, from.as_vec2().extend(0.0), to.as_vec2().extend(0.0));
                arrows.push(((from + to) * 0.5, (to - from).perp()));
            }
            for q in [p, bp, ap] {
                if inside_box(q, box_lo, box_hi) {
                    f.dot(q.as_vec2(), 7.0, pal::WHITE);
                }
            }
        }

        // the names
        let seen = |f: &Frame, p: DVec2| p.is_finite() && on_screen(f, p.as_vec2().extend(0.0)).is_some_and(|q| inside_rect(q, (Vec2::ZERO, f.view_size())));
        // (a shape's name over its top arm, on whichever side of that point is clear)
        let beside = |m: DMat2| (m * DVec2::new(0.75, 1.6)).as_vec2().extend(0.0);
        let away = |p: DVec2| Vec2::new(p.x as f32, -(p.y as f32));
        let outward = |m: DMat2| away(m * DVec2::new(0.0, 1.0));
        if ya > 0.5 && seen(f, ym * DVec2::new(0.62, 0.65)) {
            let text = if show == Show::Root { "$B^2 = A$" } else { "$A^t$" };
            room.label(f, Tag("yellow", 0), beside(ym), text, 15.0, pal::YELLOW, outward(ym), 4.0, false);
        }
        // (with the roots, the arrows name B)
        if ba > 0.5 && show == Show::Power && seen(f, bm * DVec2::new(0.62, 0.65)) {
            room.label(f, Tag("blue", 0), beside(bm), "blend", 14.0, pal::BLUE.mix(pal::TEXT, 0.25), outward(bm), 8.0, false);
        }
        for (i, (at, bow)) in arrows.iter().enumerate() {
            if seen(f, *at) {
                room.label(f, Tag("apply", i as u32), at.as_vec2().extend(0.0), "$B$", 13.0, pal::BLUE.mix(pal::TEXT, 0.35), away(*bow), 4.0, false);
            }
        }
        if paths {
            for &(l, v, c, i) in &lines {
                let at = v * 2.4;
                if seen(f, at) {
                    let text = if lines.len() == 2 { format!(r"$\lambda_{} = {}$", i + 1, num(l, 2)) } else { format!(r"$\lambda = {}$", num(l, 2)) };
                    room.label(f, Tag("eigen", i), at.as_vec2().extend(0.0), &text, 13.0, c, away(v.perp()), 8.0, true);
                }
            }
        }
        plates
    }

    /// The heading at the top of the picture: what A is, and in a line under it what it
    /// has of roots and logarithms. Returns the box it takes.
    fn heading(&mut self, f: &mut Frame, lay: &Layout) -> (Vec2, Vec2) {
        let at = lay.picture.0 + Vec2::new(16.0, 14.0);
        let (first, second) = describe(self.now.a);
        let room = (lay.picture.1.x - lay.picture.0.x - 32.0).max(40.0);
        let kept = self.heading.as_ref().filter(|h| (h.of.0.as_str(), h.of.1.as_str(), h.of.2) == (first.as_str(), second.as_str(), room)).map(|h| h.set.clone());
        let set = match kept {
            Some(set) => set,
            None => match heading_set(&first, &second, room, |t, s| measured(f, t, s)) {
                Some(set) => {
                    self.heading = Some(Heading { of: (first.clone(), second.clone(), room), set: set.clone() });
                    set
                }
                None => heading_set(&first, &second, room, |t, s| Some(width(f, t, s))).unwrap_or((first.clone(), 13.0, second.clone(), 11.0, room)),
            },
        };
        let (first, size, second, small, wide) = set;
        f.caption_px(at, first.as_str(), size, pal::TEXT, Align2::LEFT_TOP);
        f.caption_px(at + Vec2::new(0.0, size + 8.0), second.as_str(), small, pal::DIM, Align2::LEFT_TOP);
        (at - Vec2::splat(6.0), at + Vec2::new(wide + 12.0, size + small + 16.0))
    }

    /// The screen panel: the tabs, the matrices, areas, the time track or the roots'
    /// buttons, the logarithm's angle, and the eigenvalues in the complex plane.
    fn panel(&self, f: &mut Frame, p: &Panel) {
        hud_panel(f, p.min, p.max);
        let Setting { a, show, winding, root } = self.now;
        let s = p.text;
        let small = 12.0;
        let found = roots(a);

        // the tabs
        for (k, (rect, name)) in p.tabs.iter().zip(["the power $A^t$", "square roots"]).enumerate() {
            let on = (k == 0) == (show == Show::Power);
            hud_rect(f, rect.0, rect.1, if on { Color::hex(0x24304A) } else { Color::hex(0x121A2C) });
            button_outline(f, *rect, if on { pal::TEXT.a(0.7) } else { pal::DIM.a(0.5) });
            f.text_px((rect.0 + rect.1) * 0.5, name, 12.5, if on { pal::TEXT } else { pal::DIM }, Align2::CENTER_CENTER);
        }

        // the matrices
        let names = leads(show);
        let mid_of = |top: Vec2| top.y + matrix_size(2, 2, CELL, s).y * 0.5;
        let lead_w = |f: &Frame, t: &str| width(f, t, s);
        let first_w = lead_w(f, names[0]).max(lead_w(f, names[2]));
        let mut x = p.mats.x;
        f.text_px(Vec2::new(x, mid_of(p.mats)), names[0], s, pal::TEXT, Align2::LEFT_CENTER);
        x += first_w + 6.0;
        x += matrix(f, Vec2::new(x, p.mats.y), &entries(a, [pal::TEXT, pal::TEXT]), s).x + 12.0;
        f.text_px(Vec2::new(x, mid_of(p.mats)), names[1], s, pal::TEXT, Align2::LEFT_CENTER);
        x += lead_w(f, names[1]) + 6.0;
        let second_matrix = match show {
            Show::Power => power(a, winding, self.t),
            Show::Root => found.get(root).copied(),
        };
        match second_matrix {
            Some(m) => {
                matrix(f, Vec2::new(x, p.mats.y), &entries(m, [pal::PINK, pal::GREEN]), s);
            }
            None => {
                let why = if show == Show::Power { "none: no real logarithm" } else { "none" };
                f.text_px(Vec2::new(x, mid_of(p.mats)), why, small, pal::RED, Align2::LEFT_CENTER);
            }
        }
        if let Some(second) = p.second {
            f.text_px(Vec2::new(second.x, mid_of(second)), names[2], s, pal::TEXT, Align2::LEFT_CENTER);
            let x = second.x + first_w + 6.0;
            let third = match show {
                Show::Power => log_of(a, winding),
                Show::Root => found.get(root).map(|b| *b * *b),
            };
            match third {
                Some(m) => {
                    let size = matrix(f, Vec2::new(x, second.y), &entries(m, [pal::TEXT, pal::TEXT]), s);
                    let note = match (show, log_angle(a, winding)) {
                        (Show::Power, Some(angle)) => Some(format!("angle {}", degrees(angle))),
                        (Show::Power, None) => Some(if principal(a) { "the only real one".to_string() } else { String::new() }),
                        (Show::Root, _) => Some("$= A$".to_string()),
                    };
                    if let Some(note) = note {
                        f.text_px(Vec2::new(x + size.x + 10.0, mid_of(second)), note, small, pal::DIM, Align2::LEFT_CENTER);
                    }
                }
                None => {
                    f.text_px(Vec2::new(x, mid_of(second)), "none: no real logarithm", small, pal::RED, Align2::LEFT_CENTER);
                }
            }
        }

        // areas
        let y2 = p.lines.y + LINE_H;
        match show {
            Show::Power => {
                match power(a, winding, self.t) {
                    Some(m) => f.text_px(p.lines, format!(r"area: $\det A^t = (\det A)^t = {}$", short(m.determinant())), small, pal::YELLOW, Align2::LEFT_TOP),
                    None => f.text_px(p.lines, r"no real $A^t$: $A$ has no real logarithm", small, pal::YELLOW, Align2::LEFT_TOP),
                }
                let blend_det = blend(a, self.t).determinant();
                let c = if self.blend { pal::BLUE.mix(pal::TEXT, 0.3) } else { pal::DIM };
                f.text_px(Vec2::new(p.lines.x, y2), format!(r"blend: $\det\,((1 - t)I + tA) = {}$", short(blend_det)), small, c, Align2::LEFT_TOP);
            }
            Show::Root => match found.get(root) {
                Some(b) => {
                    let mu = paired(eigenvalues(a), eigenvalues(*b), c_square);
                    let eig = if mu[0].y.abs() > 1e-12 { format!(r"${} \pm {}i$", num(mu[0].x, 2), num(mu[0].y.abs(), 2)) } else { format!("${}$ and ${}$", num(mu[0].x, 2), num(mu[1].x, 2)) };
                    f.text_px(p.lines, format!("$B$: eigenvalues {eig}"), small, pal::BLUE.mix(pal::TEXT, 0.3), Align2::LEFT_TOP);
                    f.text_px(Vec2::new(p.lines.x, y2), format!(r"$\det B = {}$, and $(\det B)^2 = \det A$", short(b.determinant())), small, pal::DIM, Align2::LEFT_TOP);
                }
                None => {
                    let why = if a.determinant() < -EXACT * size(a).powi(2) { r"$(\det B)^2$ cannot be $\det A < 0$" } else { "no real $B$ with $B^2 = A$" };
                    f.text_px(p.lines, "no real square root", small, pal::RED, Align2::LEFT_TOP);
                    f.text_px(Vec2::new(p.lines.x, y2), why, small, pal::DIM, Align2::LEFT_TOP);
                }
            },
        }

        // the time track, or the roots' buttons
        match show {
            Show::Power => {
                let ((b0, b1), play) = p.bar();
                f.text_px(b0 - Vec2::new(12.0, 0.0), "$t$", 14.0, pal::TEXT, Align2::CENTER_CENTER);
                hud_line(f, b0, b1, 2.0, pal::DIM);
                for (v, name) in [(0.0, "$I$"), (0.5, "$A^{1/2}$"), (1.0, "$A$"), (2.0, "$A^2$")] {
                    let x = b0.x + (b1.x - b0.x) * (v / T_MAX) as f32;
                    hud_line(f, Vec2::new(x, b0.y - 4.0), Vec2::new(x, b0.y + 4.0), 1.2, pal::DIM);
                    f.text_px(Vec2::new(x, b0.y + 6.0), name, 10.5, pal::DIM, Align2::CENTER_TOP);
                }
                let knob = b0 + (b1 - b0) * (self.t / T_MAX).clamp(0.0, 1.0) as f32;
                hud_line(f, b0, knob, 3.0, pal::WHITE.a(0.8));
                f.text_px(Vec2::new(b1.x + 12.0, b0.y), format!("$t = {}$", num(self.t, 2)), 13.0, pal::TEXT, Align2::LEFT_CENTER);
                hud_rect(f, play.0, play.1, Color::hex(0x1A2236));
                hud_outline(f, play.0, play.1, 1.0, pal::DIM);
                f.text_px((play.0 + play.1) * 0.5, if self.playing() { "❚❚" } else { "▶" }, 14.0, pal::TEXT, Align2::CENTER_CENTER);
                // the logarithm's angle, where there is a choice
                if windings(a) {
                    for (k, rect) in button_row(p.choice, 3, 70.0).into_iter().enumerate() {
                        let w = [-1, 0, 1][k];
                        let on = w == winding;
                        hud_rect(f, rect.0, rect.1, if on { Color::hex(0x24304A) } else { Color::hex(0x121A2C) });
                        button_outline(f, rect, if on { pal::TEXT.a(0.7) } else { pal::DIM.a(0.5) });
                        let text = log_angle(a, w).map_or(String::new(), degrees);
                        f.text_px((rect.0 + rect.1) * 0.5, text, 12.0, if on { pal::TEXT } else { pal::DIM }, Align2::CENTER_CENTER);
                    }
                    let last = button_row(p.choice, 3, 70.0)[2].1.x;
                    if p.choice.1.x - last > 80.0 {
                        f.text_px(Vec2::new(last + 10.0, (p.choice.0.y + p.choice.1.y) * 0.5), "the angle of $L$", small, pal::DIM, Align2::LEFT_CENTER);
                    }
                } else {
                    let why = match counts(a).1 {
                        Count::One => "one real logarithm, so one path $A^t$",
                        _ => "no real logarithm, so no path $A^t$",
                    };
                    f.text_px(Vec2::new(p.choice.0.x, (p.choice.0.y + p.choice.1.y) * 0.5), why, small, pal::DIM, Align2::LEFT_CENTER);
                }
            }
            Show::Root => {
                if found.is_empty() {
                    f.text_px(Vec2::new(p.controls.0.x, (p.controls.0.y + p.controls.1.y) * 0.5), "no real square root to show", small, pal::DIM, Align2::LEFT_CENTER);
                } else {
                    let rects = button_row(p.controls, found.len(), 58.0);
                    for (k, rect) in rects.iter().enumerate() {
                        let on = k == root;
                        hud_rect(f, rect.0, rect.1, if on { Color::hex(0x24304A) } else { Color::hex(0x121A2C) });
                        button_outline(f, *rect, if on { pal::BLUE.a(0.9) } else { pal::DIM.a(0.5) });
                        f.text_px((rect.0 + rect.1) * 0.5, format!("$B_{}$", k + 1), 13.0, if on { pal::TEXT } else { pal::DIM }, Align2::CENTER_CENTER);
                    }
                }
                let what = found.get(root).map_or(String::new(), |b| root_note(a, *b, root));
                f.text_px(Vec2::new(p.choice.0.x, (p.choice.0.y + p.choice.1.y) * 0.5), what, small, pal::DIM, Align2::LEFT_CENTER);
            }
        }

        self.plane(f, p);
    }

    /// The complex plane: the eigenvalues of A^t on their paths from 1 (t = 0) to λ²
    /// (t = 2), rings at λ, and the blend's on their straight lines; with the square
    /// roots, the eigenvalues of B beside those of A.
    fn plane(&self, f: &mut Frame, p: &Panel) {
        let Setting { a, show, winding, .. } = self.now;
        let (fade, paths) = (self.fade as f32, self.fade > FADED);
        let blend_alpha = if show == Show::Power { self.blue.alpha() } else { 0.0 };
        let title = match show {
            Show::Power => r"eigenvalues $\lambda^t$ of $A^t$, and of the blend (blue)",
            Show::Root => r"eigenvalues $\lambda$ of $A$, and $\mu$ of $B$ (blue): $\mu^2 = \lambda$",
        };
        if let Some(at) = p.plane_title {
            f.text_px(at, title, 11.5, pal::TEXT, Align2::LEFT_TOP);
        }
        let (min, max) = p.plane;
        hud_rect(f, min, max, Color::hex(0x10172A));
        let [cx, cy, hx, hy] = self.plane.at;
        let room = max - min;
        let scale = (room.x as f64 / (2.0 * hx)).min(room.y as f64 / (2.0 * hy)).max(1e-6);
        let mid = (min + max) * 0.5;
        let to_px = |z: Cx| (mid + Vec2::new(((z.x - cx) * scale) as f32, (-(z.y - cy) * scale) as f32)).clamp(min, max);
        // the axes, the unit circle, and the marks 0 and 1
        let zero = to_px(DVec2::ZERO);
        hud_line(f, Vec2::new(min.x, zero.y), Vec2::new(max.x, zero.y), 1.0, pal::DIM.a(0.6));
        hud_line(f, Vec2::new(zero.x, min.y), Vec2::new(zero.x, max.y), 1.0, pal::DIM.a(0.6));
        hud_path(f, unit_circle().map(to_px), 1.0, pal::DIM.a(0.5));
        f.text_px(zero + Vec2::new(-4.0, 3.0), "$0$", 10.5, pal::DIM, Align2::RIGHT_TOP);
        f.text_px(to_px(DVec2::X) + Vec2::new(3.0, 3.0), "$1$", 10.5, pal::DIM, Align2::LEFT_TOP);
        let lam = eigenvalues(a);
        let colours = [pal::ORANGE, pal::PURPLE];
        let dot = |f: &mut Frame, z: Cx, size: f32, c: Color| {
            let at = f.hud_at(2, to_px(z));
            f.dl.overlay_discs.push(at, size, c);
        };
        match show {
            Show::Power => {
                // the blend's: straight from 1 through λ (t = 1)
                if blend_alpha > 0.002 && paths {
                    for l in lam {
                        let end = l * 2.0 - DVec2::X;
                        let (p, q) = (f.hud(to_px(DVec2::X)), f.hud(to_px(end)));
                        lines_for(f, Part::PlaneMarks).segment(p, q, 1.0, pal::BLUE.a(0.45 * fade * blend_alpha));
                    }
                }
                // λ^t = e^(tμ), from 1 to λ²
                if let Some(mu) = log_eigenvalues(a, winding).filter(|_| paths) {
                    let k = before(self.t, PLANE_TRACK);
                    for (i, m) in mu.iter().enumerate() {
                        let path: Vec<Vec2> = (0..=PLANE_TRACK).map(|j| to_px(c_exp(*m * (j as f64 / PLANE_TRACK as f64 * T_MAX)))).collect();
                        let now = to_px(c_exp(*m * self.t));
                        let c = colours[i];
                        let travelled: Vec<Vec3> = path[..=k].iter().copied().chain([now]).map(|q| f.hud(q)).collect();
                        lines_for(f, Part::PlaneTravelled).polyline(travelled, 1.8, c.a(0.85 * fade));
                        let ahead: Vec<Vec3> = [now].into_iter().chain(path[k + 1..].iter().copied()).map(|q| f.hud(q)).collect();
                        lines_for(f, Part::PlaneAhead).polyline(ahead, 1.0, c.a(0.35 * fade));
                    }
                }
                if paths {
                    for (i, l) in lam.iter().enumerate() {
                        let ring: Vec<Vec3> = (0..=16).map(|k| f.hud(to_px(*l) + Vec2::from_angle(k as f32 * TAU / 16.0) * 5.0)).collect();
                        lines_for(f, Part::PlaneMarks).polyline(ring, 1.4, colours[i].a(0.9 * fade));
                    }
                    name_eigenvalues(f, lam, to_px, fade);
                }
                if blend_alpha > 0.002 {
                    for l in lam {
                        dot(f, DVec2::X * (1.0 - self.t) + l * self.t, 8.0, pal::BLUE.a(blend_alpha));
                    }
                }
                if let Some(mu) = log_eigenvalues(a, winding) {
                    for (i, m) in mu.iter().enumerate() {
                        dot(f, c_exp(*m * self.t), 9.0, colours[i]);
                    }
                }
            }
            Show::Root => {
                for (i, l) in lam.iter().enumerate() {
                    dot(f, *l, 9.0, colours[i]);
                }
                name_eigenvalues(f, lam, to_px, 1.0);
                // (the root as drawn: it glides from one root to the next)
                if self.blue.alpha() > 0.002 {
                    let mu = paired(lam, eigenvalues(self.blue.matrix()), c_square);
                    for (i, m) in mu.iter().enumerate() {
                        let (p, q) = (f.hud(to_px(*m)), f.hud(to_px(lam[i])));
                        lines_for(f, Part::PlaneMarks).segment(p, q, 1.0, colours[i].a(0.4));
                        dot(f, *m, 8.0, pal::BLUE);
                    }
                }
            }
        }
    }
}

/// The names of A's eigenvalues in the complex plane: λ and its conjugate above and
/// below their places, or λ₁ above and λ₂ below for two real ones (on the real axis).
fn name_eigenvalues(f: &mut Frame, lam: [Cx; 2], to_px: impl Fn(Cx) -> Vec2, fade: f32) {
    let complex = lam[0].y.abs() > 1e-12;
    let names = if complex { [r"$\lambda$", r"$\bar\lambda$"] } else { [r"$\lambda_1$", r"$\lambda_2$"] };
    let same = lam[0].distance(lam[1]) < 1e-9;
    for (i, (l, name)) in lam.iter().zip(names).enumerate() {
        if same && i == 1 {
            break;
        }
        let (off, align) = if i == 0 { (Vec2::new(0.0, -8.0), Align2::CENTER_BOTTOM) } else { (Vec2::new(0.0, 8.0), Align2::CENTER_TOP) };
        let name = if same { r"$\lambda$ twice" } else { name };
        f.text_px(to_px(*l) + off, name, 11.5, [pal::ORANGE, pal::PURPLE][i].a(fade), align);
    }
}

/// What a square root is, in a few words.
fn root_note(a: DMat2, b: DMat2, k: usize) -> String {
    if k == 0 && principal(a) {
        return r"the principal root, $A^{1/2}$".to_string();
    }
    if let Some(w) = path_through(a, b) {
        if let Some(angle) = log_angle(a, w) {
            return format!("halfway along the path of angle {}", degrees(angle));
        }
    }
    if b.determinant() < 0.0 {
        return r"it flips the plane: $\det B < 0$".to_string();
    }
    match (kind(a), counts(a).1) {
        // (B = √|λ| K with K² = −I: halfway along e^(t(ln|λ| I + πK)), a path that
        // turns as K does rather than as J does)
        (Kind::Scalar { l }, _) if l < 0.0 => "halfway along a path not drawn here".to_string(),
        (Kind::Scalar { .. }, _) if size(b) == 0.0 => "$B = 0$".to_string(),
        (Kind::Scalar { .. }, _) => r"$B^2 = 0$, though $B \ne 0$".to_string(),
        (_, Count::None) => "no logarithm, so no path $A^t$".to_string(),
        _ => "on no path $A^t$".to_string(),
    }
}

/// A path drawn in `n` dashes, each the first part of an `n`th of its points: as many
/// points in every frame, whatever the zoom (`Frame::dashed_path` measures its dashes on
/// the screen, so that a zoom changes how many there are).
fn dashed(f: &mut Frame, part: Part, points: &[DVec2], n: usize, width: f32, c: Color) {
    let batch = lines_for(f, part);
    let per = (points.len().saturating_sub(1) as f64 / n.max(1) as f64).max(1.0);
    for k in 0..n {
        let (from, to) = ((k as f64 * per) as usize, ((k as f64 + 0.55) * per).ceil() as usize);
        if to >= points.len() {
            break;
        }
        batch.brk();
        for p in &points[from..=to] {
            batch.push(p.as_vec2().extend(0.0), width, c);
        }
    }
    batch.brk();
}

/// A small chevron at `p` pointing along `dir`, its size fixed on the screen (`wpp`:
/// world units a pixel).
fn chevron(f: &mut Frame, p: DVec2, dir: DVec2, wpp: f32, c: Color) {
    let d = dir.normalize_or_zero().as_vec2() * (6.0 * wpp);
    if d == Vec2::ZERO || !p.is_finite() {
        return;
    }
    let (p, n) = (p.as_vec2(), d.perp());
    lines_for(f, Part::Marks).polyline([p - d + n, p, p - d - n].map(|q| q.extend(0.0)), 2.0, c);
}

/// The letter (filled, and its outline) and the unit circle (`round` as much as it
/// shows) carried by `m`, in `c`.
#[allow(clippy::too_many_arguments)]
fn draw_shape(f: &mut Frame, m: DMat2, c: Color, alpha: f32, round: f32, lo: DVec2, hi: DVec2) {
    for bar in BARS.iter().filter(|_| alpha > 0.002) {
        let quad = [DVec2::new(bar[0], bar[1]), DVec2::new(bar[2], bar[1]), DVec2::new(bar[2], bar[3]), DVec2::new(bar[0], bar[3])].map(|p| m * p);
        let cut = clip_polygon(&quad, lo, hi);
        for i in 1..cut.len().saturating_sub(1) {
            f.dl.tris_alpha.tri(cut[0].as_vec2().extend(0.0), cut[i].as_vec2().extend(0.0), cut[i + 1].as_vec2().extend(0.0), c.a(0.22 * alpha));
        }
    }
    let outline: Vec<Vec3> = within(OUTLINE.iter().chain([&OUTLINE[0]]).map(|p| m * DVec2::from(*p)), lo, hi);
    f.dl.lines.polyline(outline, 2.0, c.a(0.95 * alpha));
    let ring = within(unit_circle().map(|p| m * p), lo, hi);
    f.dl.lines.polyline(ring, 1.6, c.a(0.8 * alpha * round));
}

fn button_outline(f: &mut Frame, (lo, hi): (Vec2, Vec2), c: Color) {
    let corners = [lo, Vec2::new(hi.x, lo.y), hi, Vec2::new(lo.x, hi.y), lo].map(|q| f.hud(q));
    lines_for(f, Part::Buttons).polyline(corners, 1.0, c);
}

fn inside_box(p: DVec2, lo: DVec2, hi: DVec2) -> bool {
    p.is_finite() && p.cmpge(lo).all() && p.cmple(hi).all()
}

/// The caption's lines as last broken, and how wide the widest is, kept with what they
/// were broken for: which parts, and the room across.
struct Caption {
    of: ((Show, bool), f32),
    lines: Vec<String>,
    widest: f32,
}

/// `parts` side by side as long as they fit across `room`, on further lines when they
/// do not, and the widest line's width (none if `measure` cannot say).
fn caption_lines(parts: &[&str], room: f32, size: f32, measure: impl Fn(&str) -> Option<f32>) -> Option<(Vec<String>, f32)> {
    let gap = "      ";
    let mut lines: Vec<String> = Vec::new();
    for part in parts {
        let fits = match lines.last() {
            Some(last) => measure(last)? + text_width(gap, size) + measure(part)? <= room,
            None => false,
        };
        match lines.last_mut() {
            Some(last) if fits => {
                last.push_str(gap);
                last.push_str(part);
            }
            _ => lines.push(part.to_string()),
        }
    }
    let mut widest = 0.0f32;
    for line in &lines {
        widest = widest.max(measure(line)?);
    }
    Some((lines, widest))
}

/// A caption along the bottom of the picture, on a dark plate, naming what has no name
/// of its own there. Returns the box it takes (none on a canvas too small to spare it).
fn caption(f: &mut Frame, parts: &[&str], of: (Show, bool), (left, right): (f32, f32), bottom: f32, kept: &mut Option<Caption>) -> Option<(Vec2, Vec2)> {
    let size = 12.0;
    if f.view_size().y < 300.0 {
        return None;
    }
    let room = right - left - 36.0;
    if kept.as_ref().is_none_or(|c| c.of != (of, room)) {
        *kept = caption_lines(parts, room, size, |t| measured(f, t, size)).map(|(lines, widest)| Caption { of: (of, room), lines, widest });
    }
    let (lines, widest) = match kept {
        Some(c) => (c.lines.clone(), c.widest),
        None => caption_lines(parts, room, size, |t| Some(width(f, t, size))).unwrap_or_default(),
    };
    let count = lines.len();
    for (k, line) in lines.into_iter().enumerate() {
        let up = (count - 1 - k) as f32 * (size + 7.0);
        f.caption_px(Vec2::new(left + 18.0, bottom - up), line, size, pal::DIM, Align2::LEFT_BOTTOM);
    }
    Some((Vec2::new(left + 10.0, bottom - count as f32 * (size + 7.0) - 4.0), Vec2::new(left + 26.0 + widest, bottom + 6.0)))
}

/// The heading's sizes as last chosen, kept with what they were chosen for.
struct Heading {
    of: (String, String, f32),
    set: (String, f32, String, f32, f32),
}

/// The heading's two lines as they are set, each with its size, and how wide the wider
/// is: each as large as fits across `room`, and cut short of its explanation (after
/// the first colon, or the first ", as" or " (") where even the smaller size does not
/// fit. None if `measure` cannot say.
fn heading_set(first: &str, second: &str, room: f32, measure: impl Fn(&str, f32) -> Option<f32>) -> Option<(String, f32, String, f32, f32)> {
    let fit = |text: &str, sizes: [f32; 2]| -> Option<(String, f32)> {
        let short = [": ", ", as ", " ("].iter().filter_map(|cut| text.find(cut)).min().map_or(text, |at| &text[..at]);
        for (text, size) in [(text, sizes[0]), (text, sizes[1]), (short, sizes[0]), (short, sizes[1])] {
            if measure(text, size)? <= room {
                return Some((text.to_string(), size));
            }
        }
        Some((short.to_string(), sizes[1]))
    };
    let (first, size) = fit(first, [15.0, 12.5])?;
    let (second, small) = fit(second, [12.5, 10.5])?;
    let wide = measure(&first, size)?.max(measure(&second, small)?).min(room);
    Some((first, size, second, small, wide))
}

/// Keep names off the numbers the axes are marked with (where `Frame::axes2` puts them).
fn keep_ticks_clear(f: &Frame, room: &mut Room) {
    let (lo, hi) = f.cam.flat_bounds();
    let view = f.view_size();
    let wpp = (hi.y - lo.y) / view.y.max(1.0);
    if !(wpp.is_finite() && wpp > 0.0) {
        return;
    }
    let step = nice_step(wpp * 95.0);
    let margin = 16.0 * wpp;
    let y_anchor = 0.0f32.max(lo.y + margin * 1.6).min(hi.y - margin * 0.4);
    let x_anchor = 0.0f32.max(lo.x + margin * 0.4).min(hi.x - margin * 2.4);
    let count = |lo: f32, hi: f32| ((hi - lo) / step).floor().clamp(0.0, 200.0) as i32;
    let first = |lo: f32| (lo / step).ceil();
    for i in 0..=count(lo.x, hi.x) {
        let x = (first(lo.x) + i as f32) * step;
        if let Some(p) = on_screen(f, Vec3::new(x, y_anchor, 0.0)) {
            room.boxed(p + Vec2::new(-12.0, 4.0), p + Vec2::new(12.0, 18.0));
        }
    }
    for i in 0..=count(lo.y, hi.y) {
        let y = (first(lo.y) + i as f32) * step;
        if let Some(p) = on_screen(f, Vec3::new(x_anchor, y, 0.0)) {
            room.boxed(p + Vec2::new(5.0, -7.0), p + Vec2::new(30.0, 7.0));
        }
    }
}

/// The box a label takes on the screen, nearly.
fn label_box(l: &crate::gfx::draw::Label) -> (Vec2, Vec2) {
    let size = Vec2::new(text_width(&l.text, l.size), l.size + 2.0);
    let share = |a: egui::Align| match a {
        egui::Align::Min => 0.0,
        egui::Align::Center => 0.5,
        egui::Align::Max => 1.0,
    };
    let min = l.pos - size * Vec2::new(share(l.align.x()), share(l.align.y()));
    (min, min + size)
}

/// Whether two boxes on the screen meet.
fn overlap(a: (Vec2, Vec2), b: (Vec2, Vec2)) -> bool {
    a.0.x < b.1.x && b.0.x < a.1.x && a.0.y < b.1.y && b.0.y < a.1.y
}

#[cfg(test)]
mod tests {
    use super::*;

    fn close(m: DMat2, n: DMat2, tol: f64) -> bool {
        (m - n).to_cols_array().iter().all(|v| v.abs() <= tol)
    }

    /// e^L by its series, summed until the terms are nothing.
    fn by_series(l: DMat2) -> DMat2 {
        let (mut sum, mut term) = (I, I);
        for k in 1..80 {
            term = term * l * (1.0 / k as f64);
            sum += term;
        }
        sum
    }

    /// The scene as an experiment's button leaves it, run at 60 frames a second until
    /// nothing moves (or for `seconds` at most).
    fn run(k: usize, seconds: f32) -> Matfun {
        let mut scene = Matfun::new();
        scene.set_up(k);
        settle(&mut scene, seconds);
        scene
    }

    fn settle(scene: &mut Matfun, seconds: f32) {
        for _ in 0..(seconds * 60.0) as usize {
            scene.advance(1.0 / 60.0, 1.0 / 60.0);
            if !scene.animated_now() {
                break;
            }
        }
    }

    /// Matrices of every kind, and random ones.
    fn cases() -> Vec<DMat2> {
        let mut cases: Vec<DMat2> = PRESETS.iter().map(|p| rows((p.1)())).collect();
        cases.extend([
            rows([2.0, 1.0, 1.0, 2.0]),
            rows([0.3, 1.7, -0.4, 1.1]),
            rows([2.0, 0.0, 0.0, 2.0]),
            DMat2::IDENTITY,
            rows([-0.5, 0.0, 0.0, -0.5]),
            rows([2.0, 3.0, 0.0, 2.0]),
            rows([-1.0, 1.0, 0.0, -1.0]),
            rows([0.0, 1.0, 0.0, 0.0]),
            rows([1.0, 2.0, 0.5, 1.0]),
            rows([-1.0, 2.0, 0.5, -1.0]),
            rows([0.0, -4.0, 0.01, 0.0]),
            rows([1.0, 1.0, 1.0, 1.0]),
            rows([-1.0, -1.0, -1.0, -1.0]),
            rows([1.0, 0.0, 0.0, 1.0 + 1e-6]),
            DMat2::ZERO,
        ]);
        let mut rng = Rng::new(7);
        cases.extend((0..400).map(|_| rows([0; 4].map(|_| (rng.range(-ENTRY_MAX as f32, ENTRY_MAX as f32) as f64 * 100.0).round() / 100.0))));
        cases
    }

    /// The exponential agrees with its series; every logarithm the scene offers
    /// exponentiates back to A; A^t is I at 0, A at 1 and A² at 2, with determinant
    /// (det A)^t all the way; and every square root squares back to A.
    #[test]
    fn logarithms_and_roots_give_a_back() {
        for a in cases() {
            let scale = size(a).max(1.0);
            let (roots_n, logs_n) = counts(a);
            for w in [0, -1, 1] {
                if w != 0 && !windings(a) {
                    continue;
                }
                let Some(l) = log_of(a, w) else {
                    assert!(logs_n == Count::None, "{a:?}: no logarithm, but {logs_n:?}");
                    continue;
                };
                assert!(logs_n != Count::None, "{a:?}");
                let s = by_series(l);
                assert!(close(expm(l, 1.0), s, 1e-9 * size(s).max(1.0)), "{a:?}: {:?} against {s:?}", expm(l, 1.0));
                assert!(close(expm(l, 1.0), a, 1e-9 * scale), "{a:?}, winding {w}: e^L = {:?}", expm(l, 1.0));
                for t in [0.0, 0.25, 0.5, 1.0, 1.5, 2.0] {
                    let m = power(a, w, t).unwrap();
                    assert!((m.determinant() - a.determinant().powf(t)).abs() <= 1e-8 * scale.powf(2.0 * t.max(1.0)), "{a:?} at {t}");
                    assert!(m.determinant() > 0.0, "{a:?} at {t}: never flat");
                }
                assert!(close(power(a, w, 0.0).unwrap(), I, 1e-12));
                assert!(close(power(a, w, 2.0).unwrap(), a * a, 1e-8 * scale * scale), "{a:?}");
                // half of it done twice is A
                let h = power(a, w, 0.5).unwrap();
                assert!(close(h * h, a, 1e-9 * scale), "{a:?}");
            }
            let found = roots(a);
            let expect = match roots_n {
                Count::None => 0,
                Count::Two => 2,
                Count::Four | Count::Many => 4,
                _ => panic!("{a:?}: {roots_n:?}"),
            };
            assert_eq!(found.len(), expect, "{a:?}");
            for b in &found {
                assert!(close(*b * *b, a, 1e-9 * scale * size(*b).max(1.0)), "{a:?}: {b:?} squared is {:?}", *b * *b);
            }
            // different roots are different
            for i in 0..found.len() {
                for j in 0..i {
                    assert!(size(found[i] - found[j]) > 1e-6, "{a:?}");
                }
            }
            // the principal root is the one on the principal path, with eigenvalues to
            // the right of 0
            if principal(a) {
                let h = power(a, 0, 0.5).unwrap();
                assert!(close(found[0], h, 1e-9 * scale), "{a:?}: {:?} against {h:?}", found[0]);
                assert!(eigenvalues(found[0]).iter().all(|z| z.x > 0.0), "{a:?}");
                assert_eq!(path_through(a, found[0]), Some(0));
            }
        }
    }

    /// The counts said are the counts there are: for two different positive
    /// eigenvalues, four square roots, two of them flipping the plane over; with
    /// complex ones two, and one real logarithm for each whole number of turns.
    #[test]
    fn the_counts_follow_the_eigenvalues() {
        let expect = [
            ([1.16, 0.8, 0.8, 1.16], Count::Four, Count::One),
            ([0.75, -1.0, 1.0, 0.75], Count::Two, Count::PerTurn),
            ([1.0, 1.0, 0.0, 1.0], Count::Two, Count::One),
            ([1.0, 0.0, 0.0, -1.0], Count::None, Count::None),
            ([-1.0, 0.0, 0.0, -1.0], Count::Many, Count::Many),
            ([-1.0, 0.0, 0.0, -2.0], Count::None, Count::None),
            ([-1.0, 1.0, 0.0, -1.0], Count::None, Count::None),
            ([0.0, 1.0, 0.0, 0.0], Count::None, Count::None),
            ([2.0, 1.0, 0.0, 0.0], Count::Two, Count::None),
            ([-2.0, 1.0, 0.0, 0.0], Count::None, Count::None),
            ([0.0, 0.0, 0.0, 0.0], Count::Many, Count::None),
            ([3.0, 0.0, 0.0, 3.0], Count::Many, Count::Many),
        ];
        for (m, r, l) in expect {
            assert_eq!(counts(rows(m)), (r, l), "{m:?}");
        }
        // the stretch: B = V diag(±1.2, ±0.8) V⁻¹, keeping (1, 1) and (1, −1)
        let a = rows(STRETCH);
        let found = roots(a);
        let want = [[1.0, 0.2, 0.2, 1.0], [0.2, 1.0, 1.0, 0.2], [-0.2, -1.0, -1.0, -0.2], [-1.0, -0.2, -0.2, -1.0]];
        for (b, w) in found.iter().zip(want) {
            assert!(close(*b, rows(w), 1e-12), "{b:?} against {w:?}");
            for v in [DVec2::new(1.0, 1.0), DVec2::new(1.0, -1.0)] {
                assert!((*b * v).perp_dot(v).abs() < 1e-12);
            }
        }
        let dets: Vec<String> = found.iter().map(|b| num(b.determinant(), 2)).collect();
        assert_eq!(dets, ["0.96", "−0.96", "−0.96", "0.96"]);
    }

    /// Experiment 1: the principal root of the turn and stretch turns by half the
    /// angle and stretches by the root of 1.25; applied twice it is A; the corner's
    /// path under A^t passes through Bp at t = 1/2.
    #[test]
    fn the_square_root_does_what_its_caption_says() {
        let scene = run(0, 30.0);
        let Setting { a, show, root, .. } = scene.now;
        assert!(show == Show::Root && root == 0 && a == rows(TURN_STRETCH));
        let Kind::Turning { r, theta, k } = kind(a) else { panic!() };
        assert!(close(k, J, 1e-15) && (r - 1.25).abs() < 1e-15 && degrees(theta) == "53.1°" && format!("{:.2}", theta.to_degrees()) == "53.13");
        let b = roots(a)[0];
        assert!(close(b, rows([1.0, -0.5, 0.5, 1.0]), 1e-15), "{b:?}");
        let half = b.x_axis.y.atan2(b.x_axis.x);
        assert!(format!("{:.2}", half.to_degrees()) == "26.57" && num(b.x_axis.length(), 3) == "1.118" && num(1.25f64.sqrt(), 3) == "1.118");
        assert!(close(b * b, a, 1e-15));
        // drawn: blue is B, yellow is A
        assert!(close(scene.blue.matrix(), b, 1e-12) && close(scene.yellow.matrix(), a, 1e-12) && scene.blue.alpha() == 1.0);
        assert_eq!(path_through(a, b), Some(0));
        let mid = track(log_of(a, 0).unwrap() * 0.5, CORNER, TRACK / 2)[TRACK / 4];
        assert!((mid - b * CORNER).length() < 1e-12);
    }

    /// Experiment 2: at t = 0.5 A^t is the turn by 85°, area 1; the blend is that turn
    /// shrunk to 0.087, area 0.0076, with eigenvalues 0.087 from 0.
    #[test]
    fn the_blend_does_what_its_caption_says() {
        let scene = run(1, 30.0);
        let Setting { a, show, winding, .. } = scene.now;
        assert!(show == Show::Power && winding == 0 && scene.blend && scene.t == 0.5);
        let m = power(a, 0, 0.5).unwrap();
        let (s, c) = 85f64.to_radians().sin_cos();
        assert!(close(m, rows([c, -s, s, c]), 1e-14));
        for t in [0.1, 0.5, 1.3, 2.0] {
            assert!((power(a, 0, t).unwrap().determinant() - 1.0).abs() < 1e-13);
        }
        let b = blend(a, 0.5);
        assert!(close(b, rows([c, -s, s, c]) * c, 1e-15) && num(c, 3) == "0.087");
        assert!(short(b.determinant()) == "0.0076" && eigenvalues(b).iter().all(|z| num(z.length(), 3) == "0.087"));
        // and t = 0.5 is the blend's flattest
        let least = (0..=200).map(|i| blend(a, i as f64 / 100.0).determinant()).fold(f64::MAX, f64::min);
        assert!((least - b.determinant()).abs() < 1e-15);
        assert!(close(scene.yellow.matrix(), m, 1e-12) && close(scene.blue.matrix(), b, 1e-12));
    }

    /// Experiment 3: four roots, stepped through, each squaring to A; the two with one
    /// minus sign flip; the last is the first turned a half turn; the first is A^½.
    #[test]
    fn the_four_roots_do_what_their_caption_says() {
        let mut scene = Matfun::new();
        scene.set_up(2);
        let mut shown = Vec::new();
        for _ in 0..(60.0 * 20.0) as usize {
            scene.advance(1.0 / 60.0, 1.0 / 60.0);
            if scene.now.show == Show::Root && shown.last() != Some(&scene.now.root) {
                shown.push(scene.now.root);
            }
        }
        assert_eq!(shown, [0, 1, 2, 3, 0]);
        assert!(!scene.animated_now());
        let a = scene.now.a;
        assert!(matches!(kind(a), Kind::Real { l1, l2, .. } if (l1 - 1.44).abs() < 1e-15 && (l2 - 0.64).abs() < 1e-15));
        let found = roots(a);
        assert!(found[1].determinant() < 0.0 && found[2].determinant() < 0.0 && num(found[1].determinant(), 2) == "−0.96");
        assert!(close(found[3], -found[0], 0.0) && close(found[3], rows([-1.0, 0.0, 0.0, -1.0]) * found[0], 1e-15));
        assert!(close(found[0], power(a, 0, 0.5).unwrap(), 1e-12));
        assert!(eigenvalues(found[0]).iter().all(|z| z.x > 0.0) && found[1..].iter().all(|b| eigenvalues(*b).iter().any(|z| z.x < 0.0)));
    }

    /// Experiment 4: the mirror has no real root and no real logarithm; the blend is
    /// flat at t = 0.5 and nowhere else from 0 to 1; nothing yellow is drawn.
    #[test]
    fn the_mirror_does_what_its_caption_says() {
        let scene = run(3, 30.0);
        let a = scene.now.a;
        assert!(a == rows(MIRROR) && a.determinant() == -1.0 && scene.t == 1.0);
        assert_eq!(counts(a), (Count::None, Count::None));
        assert!(roots(a).is_empty() && log_of(a, 0).is_none() && scene.yellow.alpha() == 0.0 && scene.blue.alpha() == 1.0);
        assert!(blend(a, 0.5).determinant() == 0.0);
        let dets: Vec<f64> = (0..=100).map(|i| blend(a, i as f64 / 100.0).determinant()).collect();
        assert!(dets[..50].iter().all(|d| *d > 0.0) && dets[51..].iter().all(|d| *d < 0.0));
    }

    /// Experiment 5: the logarithm −190° J gives the same turn; A^t turns clockwise; at
    /// t = 0.5 it is the turn by −95°, the other square root; at t = 1 the paths meet.
    #[test]
    fn the_winding_does_what_its_caption_says() {
        let scene = run(4, 30.0);
        let Setting { a, winding, .. } = scene.now;
        assert!(winding == -1 && scene.t == 1.0);
        let l = log_of(a, -1).unwrap();
        assert!(close(l, J * (-190f64).to_radians(), 1e-14) && num((-190f64).to_radians(), 2) == "−3.32");
        assert!(degrees(log_angle(a, -1).unwrap()) == "−190°" && degrees(log_angle(a, 0).unwrap()) == "170°");
        assert!(close(expm(l, 1.0), a, 1e-14) && close(log_of(a, 0).unwrap(), J * 170f64.to_radians(), 1e-14));
        // clockwise: the angle of (1, 0) goes down from 0
        let x: Vec<f64> = (0..=10).map(|i| (power(a, -1, i as f64 / 10.0).unwrap() * DVec2::X).to_angle()).collect();
        assert!(x[1] < 0.0 && x[5] < 0.0);
        let h = power(a, -1, 0.5).unwrap();
        let (s, c) = (-95f64).to_radians().sin_cos();
        assert!(close(h, rows([c, -s, s, c]), 1e-14) && close(h, -roots(a)[0], 1e-14) && close(h, roots(a)[1], 1e-14));
        assert!(close(power(a, -1, 1.0).unwrap(), power(a, 0, 1.0).unwrap(), 1e-14));
        assert!(close(scene.yellow.matrix(), a, 1e-12));
    }

    /// What the quick choices' notes say.
    #[test]
    fn the_quick_choices_do_what_their_notes_say() {
        // the stretch: lambda 1.44 along (1, 1), 0.64 along (1, -1)
        let a = rows(STRETCH);
        assert!((a * DVec2::ONE - DVec2::ONE * 1.44).length() < 1e-15 && (a * DVec2::new(1.0, -1.0) - DVec2::new(1.0, -1.0) * 0.64).length() < 1e-15);
        for t in [0.3, 1.7] {
            let m = power(a, 0, t).unwrap();
            assert!((m * DVec2::ONE - DVec2::ONE * 1.44f64.powf(t)).length() < 1e-12 && (m * DVec2::new(1.0, -1.0) - DVec2::new(1.0, -1.0) * 0.64f64.powf(t)).length() < 1e-12);
        }
        // the turn and stretch: 1.25 times the turn by 53.13 degrees
        let b = rows(TURN_STRETCH);
        let (s, c) = (b.x_axis.y / 1.25, b.x_axis.x / 1.25);
        assert!((s.hypot(c) - 1.0).abs() < 1e-15 && format!("{:.2}", s.atan2(c).to_degrees()) == "53.13");
        // the shear: log A = N, A^t = I + tN, the shear by 1/2 a root
        let shear = rows([1.0, 1.0, 0.0, 1.0]);
        let n = shear - I;
        assert!(close(n * n, DMat2::ZERO, 0.0) && close(log_of(shear, 0).unwrap(), n, 1e-15));
        assert!(close(power(shear, 0, 0.37).unwrap(), I + n * 0.37, 1e-15) && close(roots(shear)[0], rows([1.0, 0.5, 0.0, 1.0]), 1e-15));
        assert!(matches!(kind(shear), Kind::Defective { l, .. } if l == 1.0));
        // and the straight blend is A^t itself
        for t in [0.3, 1.0, 1.7] {
            assert!(close(blend(shear, t), power(shear, 0, t).unwrap(), 1e-15));
        }
        // the half turn: pi J a logarithm, J a root
        let h = rows([-1.0, 0.0, 0.0, -1.0]);
        assert!(close(log_of(h, 0).unwrap(), J * std::f64::consts::PI, 0.0) && close(expm(J * std::f64::consts::PI, 1.0), h, 1e-15) && close(roots(h)[0], J, 0.0));
        // two negative: no root
        assert!(roots(rows([-1.0, 0.0, 0.0, -2.0])).is_empty() && rows([-1.0, 0.0, 0.0, -2.0]).determinant() == 2.0);
        // the turn by 170: halfway the turn by 85, whose square is A
        let t = rows(turn_170());
        let half = power(t, 0, 0.5).unwrap();
        assert!(close(half * half, t, 1e-15) && (half.x_axis.to_angle().to_degrees() - 85.0).abs() < 1e-12);
    }

    /// The roots' notes on the panel, and the picture's heading, say what is so.
    #[test]
    fn the_notes_and_headings_hold() {
        for a in cases() {
            let scale = size(a).max(1.0);
            for (k, b) in roots(a).iter().enumerate() {
                let note = root_note(a, *b, k);
                let tol = 1e-9 * scale * size(*b).max(1.0);
                if note.starts_with("the principal") {
                    assert!(principal(a) && close(*b, power(a, 0, 0.5).unwrap(), tol), "{a:?}");
                } else if note.starts_with("halfway along the path of angle") {
                    let w = path_through(a, *b).unwrap();
                    assert!(close(power(a, w, 0.5).unwrap(), *b, tol) && note.ends_with(&degrees(log_angle(a, w).unwrap())), "{a:?}: {note}");
                } else if note.starts_with("it flips") {
                    assert!(b.determinant() < 0.0);
                } else if note == "halfway along a path not drawn here" {
                    // B = √|λ| K, K² = −I, and e^(t(ln|λ| I + πK)) goes from I to A through B
                    let Kind::Scalar { l } = kind(a) else { panic!("{a:?}") };
                    let kk = *b * (1.0 / (-l).sqrt());
                    let log = I * (-l).ln() + kk * std::f64::consts::PI;
                    assert!(l < 0.0 && close(kk * kk, -I, 1e-12) && close(expm(log, 0.5), *b, tol) && close(expm(log, 1.0), a, tol), "{a:?}");
                    assert!(path_through(a, *b).is_none());
                } else if note == "$B = 0$" {
                    assert!(size(*b) == 0.0 && size(a) == 0.0);
                } else if note.starts_with("$B^2 = 0$") {
                    assert!(size(*b * *b) == 0.0 && size(*b) > 0.0 && size(a) == 0.0);
                } else if note.starts_with("no logarithm") {
                    assert!(log_of(a, 0).is_none() && counts(a).1 == Count::None);
                } else {
                    // (A has one real logarithm, and its path is elsewhere halfway)
                    assert!(note == "on no path $A^t$" && counts(a).1 == Count::One && !close(power(a, 0, 0.5).unwrap(), *b, 1e-6), "{a:?}: {note}");
                }
            }
        }
        let says = |m: [f64; 4]| describe(rows(m));
        assert_eq!(says(STRETCH).0, r"two positive eigenvalues: $\lambda = 1.44$ and $0.64$");
        assert_eq!(says(STRETCH).1, "one real logarithm, and four real square roots");
        assert_eq!(says(TURN_STRETCH).0, r"a turn by 53.1° and a stretch by 1.25: $\lambda = 1.25(\cos\theta \pm i \sin\theta)$, $\theta = 53.1^\circ$");
        assert_eq!(says(turn_170()).0, r"a turn by 170°: $\lambda = \cos\theta \pm i \sin\theta$, $\theta = 170^\circ$");
        assert_eq!(says(turn_170()).1, "a real logarithm for each whole number of extra turns; two real square roots");
        assert_eq!(says(MIRROR).1, r"no real square root, no real logarithm: $(\det B)^2$ and $e^{\tr L}$ are never negative");
        assert_eq!(says([-1.0, 0.0, 0.0, -2.0]).1, r"no real square root and no real logarithm, though $\det A > 0$");
        assert_eq!(says([2.0, 1.0, 0.0, 0.0]).0, r"$\det A = 0$, it flattens the plane: $\lambda = 2.00$ and $0$");
        assert!(says([-2.0, 1.0, 0.0, 0.0]).1 == "no logarithm, and no real square root" && says([0.0, 1.0, 0.0, 0.0]).1 == "no logarithm, and no real square root");
        assert_eq!(says([1.0, 1.0, 0.0, 1.0]).1, "one real logarithm, and two real square roots");
        assert!(says([-1.0, 0.0, 0.0, -1.0]).1.contains("without end") && says([0.0; 4]).1.starts_with("no logarithm; square roots without end"));
        // (a matrix that flattens the plane is never e^L: det e^L = e^(tr L) > 0)
        for l in [rows([0.3, -2.0, 1.0, 0.1]), rows([-4.0, 0.0, 0.0, -4.0]), J * 3.0] {
            assert!(expm(l, 1.0).determinant() > 0.0 && (expm(l, 1.0).determinant() - trace(l).exp()).abs() < 1e-12 * trace(l).exp().max(1.0));
        }
    }

    /// The panel's words: the principal root's and logarithm's angles are within a
    /// quarter and half a turn; log(e^L) is L only for angles within half a turn.
    #[test]
    fn the_words_hold() {
        // e^(-190 J) is the turn by 170, whose principal logarithm is 170 J, not -190 J
        let l = J * (-190f64).to_radians();
        let back = log_of(expm(l, 1.0), 0).unwrap();
        assert!(close(back, J * 170f64.to_radians(), 1e-13) && !close(back, l, 0.1));
        let l = J * 1.2 + I * 0.3;
        assert!(close(log_of(expm(l, 1.0), 0).unwrap(), l, 1e-13));
        // a root shares A's eigenvectors when A has two different eigenvalues
        for a in cases() {
            if let Kind::Real { l1, l2, v1, v2 } = kind(a) {
                if (l1 - l2).abs() > 1e-3 {
                    for b in roots(a) {
                        for v in [v1, v2] {
                            assert!((b * v).perp_dot(v).abs() < 1e-7 * size(b).max(1.0), "{a:?}");
                        }
                    }
                }
            }
            if principal(a) {
                let root = roots(a)[0];
                for z in eigenvalues(root) {
                    assert!(z.y.atan2(z.x).abs() < std::f64::consts::FRAC_PI_2, "{a:?}");
                }
                for z in log_eigenvalues(a, 0).unwrap() {
                    assert!(z.y.abs() < std::f64::consts::PI + 1e-12, "{a:?}");
                }
            }
        }
    }

    /// The setting is changed only where nothing jumps: a new path at t = 0, a square
    /// root at t = 1; and the drawn maps then glide or follow the path.
    #[test]
    fn changes_wait_for_their_meeting_point() {
        let mut scene = Matfun::new();
        scene.set_up(3);
        let mut last_t = scene.t;
        let mut last_a = scene.now.a;
        let mut steps = 0.0f64;
        for _ in 0..600 {
            scene.advance(1.0 / 60.0, 1.0 / 60.0);
            if scene.now.a != last_a {
                // (at 0, and played on from there in the same frame)
                assert!(scene.t <= PLAY_RATE / 60.0 + 1e-6 && last_t < 2e-3, "changed at {} after {last_t}", scene.t);
                last_a = scene.now.a;
            }
            steps = steps.max((scene.t - last_t).abs());
            last_t = scene.t;
        }
        assert!(steps < 0.1, "{steps}");
        // a root asked for while t is at 0.3: shown once t is at 1
        let mut scene = Matfun::new();
        scene.ask(Setting { show: Show::Root, ..scene.now }, false, Goal::Rest);
        for _ in 0..600 {
            scene.advance(1.0 / 60.0, 1.0 / 60.0);
            if scene.now.show == Show::Root {
                assert!(scene.t == 1.0);
            }
        }
        assert!(scene.now.show == Show::Root && !scene.animated_now());
    }

    /// Degenerate input: every number drawn or read out is finite, whatever A and t.
    #[test]
    fn degenerate_input_leaves_everything_finite() {
        let mut mats = cases();
        mats.extend([rows([ENTRY_MAX; 4]), rows([ENTRY_MAX, -ENTRY_MAX, ENTRY_MAX, ENTRY_MAX]), rows([0.0, ENTRY_MAX, -0.01, 0.0]), rows([1.0, ENTRY_MAX, 1e-7, 1.0]), rows([1.0, 4.0, 1e-12, 1.0])]);
        for a in mats {
            for w in [-1, 0, 1] {
                for t in [0.0, 1e-9, 0.5, 1.0, 2.0] {
                    if let Some(m) = power(a, w, t) {
                        assert!(m.as_mat2().is_finite(), "{a:?} {w} {t}");
                    }
                    assert!(blend(a, t).as_mat2().is_finite());
                }
                if let Some(mu) = log_eigenvalues(a, w) {
                    assert!(mu.iter().all(|z| c_exp(*z * 2.0).is_finite()), "{a:?}");
                }
            }
            for b in roots(a) {
                assert!(b.as_mat2().is_finite() && within_limits_far(b).as_mat2().is_finite(), "{a:?}");
            }
            assert!(plane_fit(a).iter().all(|v| v.is_finite() && *v != 0.0 || v.is_finite()), "{a:?}");
            let (one, two) = describe(a);
            assert!(!one.contains("NaN") && !two.contains("NaN") && !one.contains("inf"), "{a:?}");
            let mut scene = Matfun::new();
            scene.ask_matrix(a);
            settle(&mut scene, 10.0);
            assert!(scene.yellow.matrix().as_mat2().is_finite() && scene.blue.matrix().as_mat2().is_finite() && scene.plane.at.iter().all(|v| v.is_finite()));
        }
    }

    #[test]
    fn the_layout_fits_every_canvas() {
        for (w, h) in [(1080.0, 1000.0), (580.0, 700.0), (390.0, 800.0), (390.0, 560.0), (800.0, 600.0), (1600.0, 1000.0), (240.0, 180.0), (620.0, 480.0), (1100.0, 700.0)] {
            for inset in [0.0, 46.0] {
                let view = Vec2::new(w, h);
                let lay = Layout::of(view, inset);
                let (min, max) = lay.picture;
                assert!(min.cmpge(Vec2::ZERO).all() && max.cmple(view).all() && (max - min).min_element() > 50.0, "{w}x{h}: {lay:?}");
                if let Some(p) = &lay.panel {
                    let within = |q: Vec2| q.cmpge(p.min - Vec2::splat(0.5)).all() && q.cmple(p.max + Vec2::splat(0.5)).all();
                    assert!(p.min.cmpge(Vec2::ZERO).all() && p.max.cmple(view + Vec2::splat(0.5)).all(), "{w}x{h}: {p:?}");
                    assert!(within(p.plane.0) && within(p.plane.1) && within(p.controls.1) && p.bar().0 .1.x > p.bar().0 .0.x + 60.0, "{w}x{h}: {p:?}");
                    assert!(p.min.y >= inset || lay.narrow, "{w}x{h}: under the buttons");
                }
            }
        }
    }
}

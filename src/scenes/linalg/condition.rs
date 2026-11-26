//! Sensitive equations: how a small error in b can become a large error in x, and the
//! condition number that says how large.

use super::*;
use glam::{DMat2, DVec2};

pub const DEF: SceneDef = SceneDef {
    id: "condition",
    title: "Sensitive equations",
    group: "Linear algebra: equations",
    blurb: r"When a small error in $b$ becomes a large error in $x$: the condition number of a matrix.",
    make: |_| Box::new(Condition::create()),
};

// ----- colours: one each, in both pictures and in the key ----------------------------

/// The equations: their lines, and the bands where they can be when b is off.
const LINE: Color = pal::BLUE;
/// Every reading within ε of b, and every answer they give.
const DOUBT: Color = pal::YELLOW;
/// One reading's error δb, and the error δx it makes in the answer.
const ERROR: Color = pal::PINK;
/// u₁ and v₁, the directions of the largest stretch σ₁.
const MOST: Color = pal::ORANGE;
/// u₂ and v₂, the directions of the smallest stretch σ₂.
const LEAST: Color = pal::PURPLE;
/// The ridge answers.
const RIDGE: Color = pal::GREEN;
/// b and x themselves.
const TRUTH: Color = pal::WHITE;

/// Seconds a change of setting (an experiment, a preset, a typed number) takes to play,
/// and a change of framing to glide.
const CHANGE_TIME: f32 = 0.9;
const GLIDE_TIME: f32 = 0.6;

/// Room kept round what each picture shows, for the names of things (logical pixels).
const PAD: f32 = 34.0;

/// A handle is kept at least this far from the one it belongs to (logical pixels), so
/// that the two can still be told apart and picked when the circle of errors is tiny.
const GRIP: f32 = 18.0;

// ----- the arithmetic, in double precision --------------------------------------------

/// Row `i` of `a`: the left-hand side of equation `i`, and the normal of its line.
fn row(a: DMat2, i: usize) -> DVec2 {
    DVec2::new(a.x_axis[i], a.y_axis[i])
}

fn dmat(m: Mat2) -> DMat2 {
    DMat2::from_cols(m.x_axis.as_dvec2(), m.y_axis.as_dvec2())
}

/// A matrix as the scene knows it: its singular values and directions, and its inverse.
#[derive(Clone, Copy, Debug)]
struct System {
    a: DMat2,
    /// σ₁ ≥ σ₂ ≥ 0: the most and the least A stretches any direction.
    s: [f64; 2],
    /// u₁, u₂ (where A sends v₁ and v₂, divided by σ₁ and σ₂) and v₁, v₂, at right
    /// angles in each pair: A vₖ = σₖ uₖ.
    u: [DVec2; 2],
    v: [DVec2; 2],
    /// None when A is flat (det A = 0 as far as its entries can tell).
    inv: Option<DMat2>,
}

impl System {
    /// Worked out in closed form rather than by `svd2`, which is in single precision:
    /// σ₂ = |det A| / σ₁ keeps its digits however nearly flat A is, and the scene's
    /// numbers go to a condition number of a million.
    fn of(a: DMat2) -> System {
        let (r1, r2) = (row(a, 0), row(a, 1));
        let (p, q, r) = (r1.length_squared(), r1.dot(r2), r2.length_squared());
        let det = a.determinant();
        let size = [r1.x, r1.y, r2.x, r2.y].iter().fold(0.0f64, |m, v| m.max(v.abs()));
        // the entries come from single precision, good to about one part in ten
        // million each: a determinant no larger than that may be exactly 0 as typed
        let flat = det.abs() <= 1e-6 * size * size || !det.is_finite();
        let s1 = ((p + r) * 0.5 + (((p - r) * 0.5).powi(2) + q * q).sqrt()).sqrt();
        if !(s1 > 0.0 && s1.is_finite()) {
            return System { a, s: [0.0, 0.0], u: [DVec2::X, DVec2::Y], v: [DVec2::X, DVec2::Y], inv: None };
        }
        // u₁ is the eigenvector of A Aᵀ for σ₁²; its sign is fixed so that it points
        // to the right (or up), and u₂ is a quarter turn on from it
        let half = 0.5 * (2.0 * q).atan2(p - r);
        let mut u1 = DVec2::new(half.cos(), half.sin());
        if u1.x < -1e-12 || (u1.x.abs() <= 1e-12 && u1.y < 0.0) {
            u1 = -u1;
        }
        let u2 = u1.perp();
        let v1 = (a.transpose() * u1 / s1).normalize_or(DVec2::X);
        // A v₂ = σ₂ u₂ with σ₂ ≥ 0: v₂ is a quarter turn from v₁, the same way round as
        // u₂ from u₁ unless A flips the plane over
        let v2 = if det < 0.0 && !flat { -v1.perp() } else { v1.perp() };
        let s2 = if flat { 0.0 } else { det.abs() / s1 };
        System { a, s: [s1, s2], u: [u1, u2], v: [v1, v2], inv: (!flat).then(|| a.inverse()) }
    }

    /// κ = σ₁/σ₂: infinite for a flat matrix (and for the zero matrix, which has no
    /// stretch at all).
    fn kappa(&self) -> f64 {
        if self.inv.is_some() {
            self.s[0] / self.s[1]
        } else {
            f64::INFINITY
        }
    }

    /// How many dimensions A keeps: 2, 1 if it squashes the plane onto a line, 0 if A = 0.
    fn rank(&self) -> usize {
        if self.inv.is_some() {
            2
        } else if self.s[0] > 0.0 {
            1
        } else {
            0
        }
    }
}

/// What a scene's settings make: b, the error δb, and what they come to in x.
#[derive(Clone, Copy, Debug, PartialEq)]
struct Shown {
    a: DMat2,
    /// The right-hand side: what the measurement should read.
    b: DVec2,
    /// The error δb, of length ε (but on its way from one error to another, which it
    /// takes straight across the circle, it can be shorter).
    db: DVec2,
    eps: f64,
    lambda: f64,
}

impl Shown {
    /// With the error of length ε pointing `turn` radians round from the x axis.
    fn new(a: DMat2, b: DVec2, turn: f64, eps: f64, lambda: f64) -> Shown {
        Shown { a, b, db: DVec2::from_angle(turn) * eps, eps, lambda }
    }
}

/// The answers to a setting.
#[derive(Clone, Copy, Debug)]
struct Picture {
    sys: System,
    b: DVec2,
    db: DVec2,
    eps: f64,
    lambda: f64,
    /// x = A⁻¹b and δx = A⁻¹δb, when A has an inverse.
    x: Option<DVec2>,
    dx: Option<DVec2>,
    /// Ridge's answer for a reading r is R r, with R = (AᵀA + λI)⁻¹Aᵀ: R, when λ > 0.
    ridge: Option<DMat2>,
}

impl Picture {
    fn of(s: &Shown) -> Picture {
        let sys = System::of(s.a);
        let db = s.db;
        let (x, dx) = match sys.inv {
            Some(inv) => (Some(inv * s.b), Some(inv * db)),
            None => (None, None),
        };
        let ridge = (s.lambda > 0.0).then(|| {
            let m = s.a.transpose() * s.a + DMat2::from_diagonal(DVec2::splat(s.lambda));
            // (positive definite: its determinant is at least λ²)
            m.inverse() * s.a.transpose()
        });
        Picture { sys, b: s.b, db, eps: s.eps, lambda: s.lambda, x, dx, ridge }
    }

    /// |δb| / |b|: none when b = 0.
    fn rel_b(&self) -> Option<f64> {
        let size = self.b.length();
        (size > 0.0).then(|| self.db.length() / size)
    }

    /// |δx| / |x|: none when there is no single x, or x = 0.
    fn rel_x(&self) -> Option<f64> {
        let (x, dx) = (self.x?, self.dx?);
        let size = x.length();
        (size > 0.0 && size.is_finite()).then(|| dx.length() / size)
    }

    /// How many times b's relative error x's is: none when there is no error, or
    /// either relative error is not there to compare.
    fn ratio(&self) -> Option<f64> {
        let (rx, rb) = (self.rel_x()?, self.rel_b()?);
        (rb > 0.0).then(|| rx / rb).filter(|r| r.is_finite())
    }

    /// The answers for every reading within ε: x + A⁻¹(ε (cos t, sin t)).
    fn ellipse(&self, n: usize) -> Option<Vec<DVec2>> {
        let (x, inv) = (self.x?, self.sys.inv?);
        Some((0..n).map(|i| x + inv * (DVec2::from_angle(i as f64 / n as f64 * std::f64::consts::TAU) * self.eps)).collect())
    }

    /// Where ridge's answer goes with no error at all, R b, and the answers it gives
    /// for every reading within ε.
    fn ridge_centre(&self) -> Option<DVec2> {
        self.ridge.map(|r| r * self.b)
    }

    fn ridge_ellipse(&self, n: usize) -> Option<Vec<DVec2>> {
        let r = self.ridge?;
        let centre = r * self.b;
        Some((0..n).map(|i| centre + r * (DVec2::from_angle(i as f64 / n as f64 * std::f64::consts::TAU) * self.eps)).collect())
    }

    /// Ridge's answer for the reading b + δb.
    fn ridge_answer(&self) -> Option<DVec2> {
        self.ridge.map(|r| r * (self.b + self.db))
    }

    /// How far ridge's answer is from x, as a share of |x|: for this reading, and with
    /// no error at all.
    fn ridge_errors(&self) -> Option<(f64, f64)> {
        let (x, at, centre) = (self.x?, self.ridge_answer()?, self.ridge_centre()?);
        let size = x.length();
        (size > 0.0).then(|| ((at - x).length() / size, (centre - x).length() / size))
    }

    /// When A squashes the plane onto a line: the x that fit some reading within ε of b
    /// fill a strip without end, v₁·x from `lo` to `hi` (None if no reading within ε
    /// lies on the line A squashes onto, so that no x fits any of them).
    fn strip(&self) -> Option<(DVec2, f64, f64)> {
        if self.sys.rank() != 1 {
            return None;
        }
        // A x = σ₁ (v₁·x) u₁: the readings that have answers lie on the line along u₁
        let (s1, u1, v1) = (self.sys.s[0], self.sys.u[0], self.sys.v[0]);
        let along = u1.dot(self.b);
        let off = (self.b - u1 * along).length();
        // (the readings on that line within ε of b reach this far either way along it)
        let half = self.eps * self.eps - off * off;
        (half >= 0.0).then(|| (v1, (along - half.sqrt()) / s1, (along + half.sqrt()) / s1))
    }

    /// The box a pane is framed on, by its corners. To scale, a pane shows 0 as well as
    /// b or x; close up, only the circle of readings round b, or the answers round x.
    fn content(&self, pane: Pane) -> (DVec2, DVec2) {
        let mut pts = if pane.close { Vec::new() } else { vec![DVec2::ZERO] };
        // (with no error, close up on a stretch the size of a tenth of b or of x)
        let tenth = |v: DVec2| DVec2::splat(0.1 * v.length().max(1e-9));
        if pane.space == 0 {
            let reach = if pane.close { if self.eps > 0.0 { DVec2::splat(1.7 * self.eps) } else { tenth(self.b) } } else { DVec2::splat(self.eps) };
            pts.extend([self.b - reach, self.b + reach]);
        } else if let (Some(x), Some(inv)) = (self.x, self.sys.inv) {
            // the ellipse reaches ε × |row j of A⁻¹| either way along axis j
            let mut reach = DVec2::new(row(inv, 0).length(), row(inv, 1).length()) * self.eps;
            let ridge = self.ridge.zip(self.ridge_centre()).map(|(r, c)| (c, DVec2::new(row(r, 0).length(), row(r, 1).length()) * self.eps));
            if pane.close {
                // (and a little more: the lines are seen crossing, the bands overlapping;
                // with ridge, close up on its answers and x, the plain ones being in
                // the picture to scale)
                reach = if ridge.is_some() { DVec2::ZERO } else if self.eps > 0.0 { reach * 1.35 } else { tenth(x) };
            }
            pts.extend([x - reach, x + reach]);
            if let Some((c, reach)) = ridge {
                let reach = if pane.close { reach * 1.35 } else { reach };
                pts.extend([c - reach, c + reach]);
            }
        } else {
            pts.push(DVec2::ZERO);
            // a strip or lines without end: a stretch of them either side of where they
            // pass nearest 0
            let mut near = Vec::new();
            if let Some((n, lo, hi)) = self.strip() {
                near.push(n * ((lo + hi) * 0.5));
                pts.extend([n * lo, n * hi]);
            }
            for i in 0..2 {
                let a = row(self.sys.a, i);
                if a.length_squared() > 0.0 {
                    near.push(a * (self.b[i] / a.length_squared()));
                }
            }
            let reach = near.iter().fold(1e-9f64, |m, p| m.max(p.length())).max(self.eps / self.sys.s[0].max(1e-300)).min(1e12);
            let along = self.sys.v[1] * reach;
            for p in near {
                pts.extend([p - along, p + along]);
            }
        }
        let lo = pts.iter().fold(DVec2::INFINITY, |m, p| m.min(*p));
        let hi = pts.iter().fold(DVec2::NEG_INFINITY, |m, p| m.max(*p));
        if lo.is_finite() && hi.is_finite() {
            (lo, hi)
        } else {
            (DVec2::splat(-1.0), DVec2::ONE)
        }
    }
}

// ----- changes of setting, played --------------------------------------------------------

/// `a` as a turn and a stretch along two directions at right angles: a = R(θ) P, or
/// a = R(θ) P M with M the mirror (x, y) → (x, −y) if `mirror`, P symmetric with no
/// negative stretch (the polar decomposition). None for the angle if `a` is 0, which
/// has every turn in it.
fn polar(a: DMat2, mirror: bool) -> (Option<f64>, DMat2) {
    let m = if mirror { a * DMat2::from_diagonal(DVec2::new(1.0, -1.0)) } else { a };
    let (c, s) = (m.x_axis.x + m.y_axis.y, m.x_axis.y - m.y_axis.x);
    if c == 0.0 && s == 0.0 {
        return (None, m);
    }
    let theta = s.atan2(c);
    (Some(theta), DMat2::from_angle(-theta) * m)
}

/// A share `s` of the way from the matrix `from` to `to`: turning by the turn between
/// them while the stretch slides from one to the other. Two stretches with no negative
/// stretch in them have none in any mixture of them either, so the plane is never
/// squashed flat on the way, unless one end is flat itself or one end flips the plane
/// over and the other does not (then nothing can help it, and the entries slide
/// straight).
fn between_matrices(from: DMat2, to: DMat2, s: f64) -> DMat2 {
    if s >= 1.0 {
        return to;
    }
    if s <= 0.0 {
        return from;
    }
    let (d0, d1) = (from.determinant(), to.determinant());
    let (m0, m1) = (d0 < 0.0, d1 < 0.0);
    // (a flat end takes the other's side)
    let mirror = if d0 == 0.0 { m1 } else { m0 };
    if !(d0 == 0.0 || d1 == 0.0 || m0 == m1) {
        return from * (1.0 - s) + to * s;
    }
    let (t0, p0) = polar(from, mirror);
    let (t1, p1) = polar(to, mirror);
    let (t0, t1) = match (t0, t1) {
        (Some(a), Some(b)) => (a, b),
        (Some(a), None) => (a, a),
        (None, Some(b)) => (b, b),
        (None, None) => (0.0, 0.0),
    };
    let turn = (t1 - t0 + std::f64::consts::PI).rem_euclid(std::f64::consts::TAU) - std::f64::consts::PI;
    let m = DMat2::from_angle(t0 + turn * s) * (p0 * (1.0 - s) + p1 * s);
    if mirror {
        m * DMat2::from_diagonal(DVec2::new(1.0, -1.0))
    } else {
        m
    }
}

/// The setting a share `s` of the way from `from` to `to`. The error goes straight from
/// one to the other (not round the circle: where the circle's direction u₁ is passed,
/// A⁻¹ would swing δx round in a moment). Where both matrices have
/// inverses, it is A⁻¹ that goes the way between them: the picture of the answers is
/// A⁻¹ of the picture of the readings, so each point of it then moves smoothly from
/// where it was to where it will be (sliding A itself to 0.01 I, say, would leave x
/// nearly still and then fling it off at the very end).
fn between(from: &Shown, to: &Shown, s: f64) -> Shown {
    if s >= 1.0 {
        return *to;
    }
    let a = match (System::of(from.a).inv, System::of(to.a).inv) {
        (Some(p), Some(q)) => {
            let inv = between_matrices(p, q, s);
            let det = inv.determinant();
            if det != 0.0 && det.is_finite() {
                inv.inverse()
            } else {
                between_matrices(from.a, to.a, s)
            }
        }
        _ => between_matrices(from.a, to.a, s),
    };
    Shown {
        a,
        b: from.b.lerp(to.b, s),
        db: from.db.lerp(to.db, s),
        eps: from.eps + (to.eps - from.eps) * s,
        lambda: from.lambda + (to.lambda - from.lambda) * s,
    }
}

// ----- the experiments ---------------------------------------------------------------

/// What a setting is made of, as an experiment or the opening asks for it.
#[derive(Clone, Copy, Debug)]
struct Setup {
    /// A by its rows.
    rows: [f32; 4],
    b: Aim,
    err: Aim,
    eps: f32,
    lambda: f32,
}

/// A direction asked for: fixed, or along a singular direction of A (u₁ or u₂ of the
/// right-hand side's picture), at the given length (for b) or sign (for the error).
#[derive(Clone, Copy, Debug)]
enum Aim {
    At(f64, f64),
    Along(usize, f64),
}

/// Two nearly parallel lines: x + 0.6y = 1.6 and 0.8x + 0.6y = 1.4, which cross at
/// (1, 1). κ = 19.6.
const STEEP: [f32; 4] = [1.0, 0.6, 0.8, 0.6];

/// The length of b in the opening, kept when b is turned.
const B_SIZE: f64 = 2.126_029_162_546_93;

const OPENING: Setup = Setup { rows: STEEP, b: Aim::At(1.6, 1.4), err: Aim::At(0.0, 1.0), eps: 0.1, lambda: 0.0 };

const TRIALS: [(&str, &str, Setup); 5] = [
    (
        "well conditioned",
        r"Two lines that cross at 70°, not at 6°. $x$ is $(1, 1)$ as before, and the reading is off the same way, by $\varepsilon = 0.1$ in its second number. The ellipse is small, and only 1.47 times as long as it is wide: $b$ is off by 6.9 % and $x$ by 7.3 %, 1.06 times as much. With this matrix $\kappa = 1.47$: no $b$ and no error can make $x$'s relative error more than 1.47 times $b$'s.",
        Setup { rows: [1.2, 0.2, -0.5, 0.9], b: Aim::At(1.4, 0.4), err: Aim::At(0.0, 1.0), eps: 0.1, lambda: 0.0 },
    ),
    (
        "the worst case",
        r"$b$ along $u_1$ and the error along $u_2$. $b$ is off by 4.7 % and $x$ by 92 %: exactly $\kappa = 19.6$ times as much, the most any $b$ and any error can do with this matrix. A $b$ along $u_1$ comes from an $x$ along $v_1$, the way $A$ stretches most, so $x$ is as short as it can be for a $b$ this long; and of all errors this long, one along $u_2$ comes from the longest $\delta x$.",
        Setup { rows: STEEP, b: Aim::Along(0, B_SIZE), err: Aim::Along(1, 1.0), eps: 0.1, lambda: 0.0 },
    ),
    (
        r"a lucky $b$ and a lucky error",
        r"The same matrix, $b$ as long and the error as large, but $b$ along $u_2$ and the error along $u_1$. $x$ is now 27 long, and the error moves it only across the ellipse's narrow way, by 0.065: $x$ is off by 0.24 %, $\kappa = 19.6$ times less than $b$'s 4.7 %. Every other $b$ and error lands between this and the worst case: from $1/\kappa$ to $\kappa$ times $b$'s relative error.",
        Setup { rows: STEEP, b: Aim::Along(1, B_SIZE), err: Aim::Along(0, 1.0), eps: 0.1, lambda: 0.0 },
    ),
    (
        "a small determinant is not the trouble",
        r"$A = 0.01 I$. Its determinant is 0.0001, twelve hundred times smaller than the nearly parallel pair's 0.12, yet $\kappa = 1$: $A^{-1}$ makes everything a hundred times larger, $x$ and its error alike, so $x$ is off by exactly as much as $b$, 4.7 %, whatever $b$ and whatever the error. The lines cross at right angles and the ellipse is a circle. What makes equations sensitive is a matrix that stretches some directions far more than others, not one that shrinks them all alike.",
        Setup { rows: [0.01, 0.0, 0.0, 0.01], b: Aim::At(1.6, 1.4), err: Aim::At(0.0, 1.0), eps: 0.1, lambda: 0.0 },
    ),
    (
        "ridge: steadier, slightly wrong",
        r"The nearly parallel pair, solved by ridge with $\lambda = 0.02$ (green). Its ellipse of answers is 4.3 times shorter than the plain one, but its middle has moved off $x$: with no error at all, ridge is 15 % off. For this reading ridge is 32 % off where the plain answer is 69 % off, and over the whole circle of errors ridge is never more than 36 % off, where the plain answer can be 90 % off. Turn the error to $u_1$ and ridge does worse than the plain answer: 16 % against 4.6 %.",
        Setup { rows: STEEP, b: Aim::At(1.6, 1.4), err: Aim::At(0.0, -1.0), eps: 0.1, lambda: 0.02 },
    ),
];

/// Quick choices for A, by condition number: (button, A by its rows). Each is
/// [[1, c], [c, 1]], whose stretches are 1 + c and 1 − c.
const PRESETS: [(&str, [f32; 4]); 6] = [
    (r"$\kappa = 1$", [1.0, 0.0, 0.0, 1.0]),
    ("$3$", [1.0, 0.5, 0.5, 1.0]),
    ("$9$", [1.0, 0.8, 0.8, 1.0]),
    ("$19$", [1.0, 0.9, 0.9, 1.0]),
    ("$99$", [1.0, 0.98, 0.98, 1.0]),
    (r"$\infty$", [1.0, 1.0, 1.0, 1.0]),
];

fn matrix_of(r: [f32; 4]) -> Mat2 {
    Mat2::from_cols(Vec2::new(r[0], r[2]), Vec2::new(r[1], r[3]))
}

// ----- numbers for the screen ---------------------------------------------------------

/// A number to `digits` significant digits (but no fewer decimals than none), with a
/// true minus sign: 2.13, 0.0782, 19.6, 160. Very large or small ones as 1.2e7.
fn sig(v: f64, digits: i32) -> String {
    if !v.is_finite() {
        return if v == f64::INFINITY { "∞".into() } else { "?".into() };
    }
    if v == 0.0 {
        return "0".into();
    }
    let a = v.abs();
    if !(1e-4..1e6).contains(&a) {
        return format!("{v:.1e}").replace('-', "−");
    }
    let decimals = |a: f64| (digits - 1 - a.log10().floor() as i32).clamp(0, 8) as usize;
    let d = decimals(a);
    // (a number that rounds up to the next power of ten has one decimal fewer: 0.0100, not 0.01000)
    let rounded: f64 = format!("{a:.d$}").parse().unwrap_or(a);
    num(v, decimals(rounded).min(d))
}

/// A share as a percentage, to two significant digits, as the captions write it in
/// words: 4.7 %, 49 %, 0.24 %.
#[cfg(test)]
fn pct(v: f64) -> String {
    format!("{} %", sig(v * 100.0, 2))
}

/// The same as markup, to go between dollars, as the picture and the numbers write it
/// (where a very small one, 2.0e−6, is set as 2.0 × 10⁻⁶).
fn pct_tex(v: f64) -> String {
    format!(r"{}\,\%", sig(v * 100.0, 2))
}

/// An equation as written on the screen (markup, to go between dollars): 1.00x₁ +
/// 0.60x₂ = 1.60.
fn equation(a: DVec2, b: f64) -> String {
    let second = num(a.y, 2);
    let (op, second) = match second.strip_prefix('−') {
        Some(rest) => ("-", rest.to_string()),
        None => ("+", second),
    };
    format!("{}x_1 {op} {second}x_2 = {}", num(a.x, 2), num(b, 2))
}

// ----- the layout ------------------------------------------------------------------------

/// Which plane a view shows (0: the readings b, 1: the answers x), and whether to scale
/// from 0 or close up on the doubt round b or x.
#[derive(Clone, Copy, Debug, PartialEq)]
struct Pane {
    space: usize,
    close: bool,
    /// Top-left and bottom-right corners, logical pixels.
    rect: (Vec2, Vec2),
}

/// Where things go on a canvas, in logical pixels from its top-left corner.
#[derive(Clone, Copy, Debug, PartialEq)]
struct Layout {
    /// One picture above the other (a phone held upright), not side by side.
    narrow: bool,
    /// Close-ups under the pictures to scale (a canvas tall enough for both).
    split: bool,
    /// b to scale, x to scale, and (if `split`) b close up, x close up.
    panes: [Pane; 4],
    /// The numbers under them (none on a canvas too small to spare the room).
    numbers: Option<(Vec2, Vec2)>,
    /// Logical pixels per unit of the stage, and the canvas.
    ppu: f32,
    view: Vec2,
}

impl Layout {
    fn of(view: Vec2, inset: f32) -> Layout {
        let view = view.max(Vec2::splat(1.0));
        let ppu = (view.y / STAGE_H).min(view.x / 2.0).max(1e-3);
        let narrow = view.x < 620.0 && view.y >= 560.0;
        let numbers_h = if narrow { 92.0 } else { 74.0 };
        let numbers = (view.y >= 330.0 && view.x >= 300.0).then(|| (Vec2::new(10.0, view.y - 8.0 - numbers_h), Vec2::new(view.x - 10.0, view.y - 8.0)));
        let bottom = numbers.map_or(view.y - 6.0, |(lo, _)| lo.y - 10.0);
        // (room above each picture for its title)
        let title = 24.0;
        let top = inset.max(0.0) + 4.0 + title;
        let pane = |space: usize, close: bool, lo: Vec2, hi: Vec2| Pane { space, close, rect: (lo, hi.max(lo + Vec2::ONE)) };
        let (split, panes) = if narrow {
            // b's picture above, a little smaller; between them the arrow and x's title
            let gap = 50.0;
            let h = ((bottom - top - gap) * 0.5).max(10.0);
            let first = pane(0, false, Vec2::new(10.0, top), Vec2::new(view.x - 10.0, top + h * 0.9));
            let second = pane(1, false, Vec2::new(10.0, first.rect.1.y + gap), Vec2::new(view.x - 10.0, bottom));
            (false, [first, second, first, second])
        } else {
            let gap = (view.x * 0.06).clamp(30.0, 74.0);
            let mid = view.x * 0.5;
            let (l0, l1, r0, r1) = (12.0, mid - gap * 0.5, mid + gap * 0.5, view.x - 12.0);
            let split = bottom - top >= 560.0;
            if split {
                // the pictures to scale need less room than the close-ups
                let h = bottom - top - title - 8.0;
                let y1 = top + h * 0.42;
                let y2 = y1 + title + 8.0;
                (
                    true,
                    [
                        pane(0, false, Vec2::new(l0, top), Vec2::new(l1, y1)),
                        pane(1, false, Vec2::new(r0, top), Vec2::new(r1, y1)),
                        pane(0, true, Vec2::new(l0, y2), Vec2::new(l1, bottom)),
                        pane(1, true, Vec2::new(r0, y2), Vec2::new(r1, bottom)),
                    ],
                )
            } else {
                let first = pane(0, false, Vec2::new(l0, top), Vec2::new(l1, bottom));
                let second = pane(1, false, Vec2::new(r0, top), Vec2::new(r1, bottom));
                (false, [first, second, first, second])
            }
        };
        Layout { narrow, split, panes, numbers, ppu, view }
    }

    /// How many panes there are.
    fn count(&self) -> usize {
        if self.split {
            4
        } else {
            2
        }
    }

    /// A point of the canvas (logical pixels) on the stage.
    fn stage(&self, px: Vec2) -> Vec2 {
        Vec2::new(px.x - self.view.x * 0.5, self.view.y * 0.5 - px.y) / self.ppu
    }

    /// The middle of pane `k` on the stage, and how far it reaches either way (stage
    /// units), less the room kept for names.
    fn room(&self, k: usize) -> (DVec2, DVec2) {
        let (lo, hi) = self.panes[k].rect;
        let mid = self.stage((lo + hi) * 0.5).as_dvec2();
        let half = (((hi - lo) * 0.5 - Vec2::splat(PAD)) / self.ppu).max(Vec2::splat(4.0 / self.ppu)).as_dvec2();
        (mid, half)
    }

    /// Pane `k`'s corners on the stage (lower left, upper right).
    fn bounds(&self, k: usize) -> (Vec2, Vec2) {
        let (lo, hi) = self.panes[k].rect;
        let (a, b) = (self.stage(lo), self.stage(hi));
        (a.min(b), a.max(b))
    }

    /// Are the details (the directions u and v, the error's handle and names) drawn in
    /// pane `k`? In the close-ups when there are some, else in the pictures to scale.
    fn details(&self, k: usize) -> bool {
        self.panes[k].close || !self.split
    }
}

/// How a pane shows its plane: a point p of it is drawn at the middle of the pane plus
/// (p − centre) × `scale` stage units.
#[derive(Clone, Copy, Debug, PartialEq)]
struct Framing {
    centre: DVec2,
    scale: f64,
}

impl Framing {
    /// The framing that shows the box `lo`–`hi` as large as it fits in `half` either
    /// way of the middle.
    fn fit(half: DVec2, lo: DVec2, hi: DVec2) -> Framing {
        let size = ((hi - lo) * 0.5).max(DVec2::ZERO);
        let biggest = size.max_element();
        // (a point, or a box with no height, still has a scale to be drawn at)
        let size = if biggest > 0.0 && biggest.is_finite() { size.max(DVec2::splat(biggest * 0.02)) } else { DVec2::ONE };
        let scale = (half.x / size.x).min(half.y / size.y);
        let centre = (lo + hi) * 0.5;
        Framing { centre: if centre.is_finite() { centre } else { DVec2::ZERO }, scale: if scale > 0.0 && scale.is_finite() { scale } else { 1.0 } }
    }

    /// Does it still show the box `lo`–`hi` well: all of it, and not at less than
    /// two fifths of the size it could be?
    fn shows(&self, half: DVec2, lo: DVec2, hi: DVec2) -> bool {
        let best = Framing::fit(half, lo, hi);
        let slack = half * 0.08;
        let (a, b) = ((lo - self.centre) * self.scale, (hi - self.centre) * self.scale);
        a.cmpge(-half - slack).all() && b.cmple(half + slack).all() && self.scale >= 0.4 * best.scale
    }

    /// This framing, moved by a share `share` of the way `was` differs from `fit` (in
    /// scale, by factors; in where 0 is drawn, straight), in a pane whose middle is at
    /// `mid` on the stage.
    fn shifted(&self, was: &Framing, fit: &Framing, mid: DVec2, share: f64) -> Framing {
        let scale = self.scale * (was.scale / fit.scale).powf(share);
        let zero = |f: &Framing| mid - f.centre * f.scale;
        let at = zero(self) + (zero(was) - zero(fit)) * share;
        let out = Framing { centre: (mid - at) / scale, scale };
        if out.centre.is_finite() && out.scale.is_finite() && out.scale > 0.0 {
            out
        } else {
            *self
        }
    }

    /// A share `s` of the way to `to`, in a pane whose middle is at `mid` on the stage:
    /// a zoom about the one place of the pane that both framings draw the same point
    /// of the plane at, the scale changing by equal factors in equal times (every point
    /// then goes straight towards or away from that place; slid straight across while
    /// it zooms, a picture zoomed far in would first fly off to one side). Without a
    /// change of scale, a slide.
    fn towards(&self, to: &Framing, mid: DVec2, s: f64) -> Framing {
        if s >= 1.0 {
            return *to;
        }
        let (kf, kt) = (self.scale, to.scale);
        // where each framing draws the plane's 0, on the stage
        let (zf, zt) = (mid - self.centre * kf, mid - to.centre * kt);
        let scale = kf * (kt / kf).powf(s);
        let zero = if (kt / kf - 1.0).abs() < 1e-9 {
            zf.lerp(zt, s)
        } else {
            let fixed = (zt * kf - zf * kt) / (kf - kt);
            fixed + (zf - fixed) * (scale / kf)
        };
        let out = Framing { centre: (mid - zero) / scale, scale };
        if out.centre.is_finite() && out.scale.is_finite() && out.scale > 0.0 {
            out
        } else {
            *to
        }
    }
}

/// From a pane's plane to the stage and back.
#[derive(Clone, Copy, Debug)]
struct Map {
    mid: DVec2,
    half: DVec2,
    f: Framing,
    /// Logical pixels per unit of the stage.
    ppu: f64,
}

impl Map {
    fn of(lay: &Layout, k: usize, f: Framing) -> Map {
        let (lo, hi) = lay.bounds(k);
        Map { mid: ((lo + hi) * 0.5).as_dvec2(), half: ((hi - lo) * 0.5).as_dvec2(), f, ppu: lay.ppu as f64 }
    }

    fn to(&self, p: DVec2) -> Vec2 {
        (self.mid + (p - self.f.centre) * self.f.scale).as_vec2()
    }

    fn back(&self, s: Vec2) -> DVec2 {
        self.f.centre + (s.as_dvec2() - self.mid) / self.f.scale
    }

    /// The part of the plane the pane shows.
    fn seen(&self) -> (DVec2, DVec2) {
        (self.f.centre - self.half / self.f.scale, self.f.centre + self.half / self.f.scale)
    }

    /// Plane units per logical pixel.
    fn per_px(&self) -> f64 {
        1.0 / (self.f.scale * self.ppu)
    }
}

// ----- clipping ------------------------------------------------------------------------

/// The part of the convex polygon `poly` where n·p ≤ c.
fn cut(poly: &[DVec2], n: DVec2, c: f64) -> Vec<DVec2> {
    let mut out = Vec::with_capacity(poly.len() + 2);
    for i in 0..poly.len() {
        let (p, q) = (poly[i], poly[(i + 1) % poly.len()]);
        let (dp, dq) = (n.dot(p) - c, n.dot(q) - c);
        if dp <= 0.0 {
            out.push(p);
        }
        if (dp < 0.0 && dq > 0.0) || (dp > 0.0 && dq < 0.0) {
            out.push(p + (q - p) * (dp / (dp - dq)));
        }
    }
    out
}

fn clip_to(poly: &[DVec2], (lo, hi): (DVec2, DVec2)) -> Vec<DVec2> {
    let mut p = poly.to_vec();
    for (n, c) in [(DVec2::X, hi.x), (-DVec2::X, -lo.x), (DVec2::Y, hi.y), (-DVec2::Y, -lo.y)] {
        p = cut(&p, n, c);
    }
    p
}

/// The part of the segment from `a` to `b` inside the box.
fn segment_in(a: DVec2, b: DVec2, (lo, hi): (DVec2, DVec2)) -> Option<(DVec2, DVec2)> {
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
    (t0 <= t1 && t0.is_finite() && t1.is_finite()).then(|| (a + d * t0, a + d * t1))
}

/// The part of the line n·p = c inside the box.
fn line_in(n: DVec2, c: f64, bx: (DVec2, DVec2)) -> Option<(DVec2, DVec2)> {
    let len = n.length();
    if !(len > 0.0 && c.is_finite()) {
        return None;
    }
    let (n, c) = (n / len, c / len);
    let (lo, hi) = bx;
    let mid = (lo + hi) * 0.5;
    // the point of the line nearest the box's middle, and far enough either way
    let foot = mid + n * (c - n.dot(mid));
    let along = n.perp() * ((hi - lo).length() + 1.0);
    segment_in(foot - along, foot + along, bx)
}

/// The band c − w ≤ n·p ≤ c + w inside the box, as a polygon (n of length 1).
fn band_in(n: DVec2, c: f64, w: f64, (lo, hi): (DVec2, DVec2)) -> Vec<DVec2> {
    let corners = [lo, DVec2::new(hi.x, lo.y), hi, DVec2::new(lo.x, hi.y)];
    cut(&cut(&corners, n, c + w), -n, -(c - w))
}

// ----- the scene ------------------------------------------------------------------------

/// A change of framing under way: from where, how far (0 to 1), and whether it goes
/// in step with a change of setting (else by the clock).
#[derive(Clone, Copy, Debug)]
struct Glide {
    from: [Framing; 4],
    t: f32,
    in_step: bool,
}

pub struct Condition {
    /// A as edited.
    a: Mat2,
    b: DVec2,
    /// The direction of the error δb (radians).
    turn: f64,
    eps: f32,
    lambda: f32,
    /// A change of setting under way: where it started, and how far it has got (0 to 1).
    change: Option<(Shown, f32)>,
    /// How each pane is framed as drawn, the framing it is heading for, and a glide
    /// between them under way (from where, how far).
    framing: [Framing; 4],
    aim: [Framing; 4],
    glide: Option<Glide>,
    /// The framing must be fitted afresh (a new setting); at once if `snap` (a new canvas).
    refit: bool,
    snap: bool,
    /// The layout last drawn in.
    layout: Option<Layout>,
    /// The directions u₁, u₂ and along each line as last drawn: each new one is taken
    /// the same way round (a direction and its opposite are the same line), so that
    /// what is named at one end of it stays at that end.
    kept: [DVec2; 4],
}

impl Condition {
    /// (Nothing here needs the graphics device, so the tests can make one too.)
    fn create() -> Self {
        let none = Framing { centre: DVec2::ZERO, scale: 1.0 };
        let mut s = Condition {
            a: Mat2::IDENTITY,
            b: DVec2::ZERO,
            turn: 0.0,
            eps: 0.1,
            lambda: 0.0,
            change: None,
            framing: [none; 4],
            aim: [none; 4],
            glide: None,
            refit: true,
            snap: true,
            layout: None,
            kept: [DVec2::X, DVec2::Y, DVec2::X, DVec2::X],
        };
        s.apply(&OPENING);
        s.change = None;
        s.snap = true;
        s
    }

    /// The setting as asked for.
    fn goal(&self) -> Shown {
        Shown::new(dmat(self.a), self.b, self.turn, self.eps as f64, self.lambda as f64)
    }

    /// The setting as drawn: on its way to the one asked for while a change plays.
    fn shown(&self) -> Shown {
        match &self.change {
            Some((from, t)) => between(from, &self.goal(), smoothstep(*t) as f64),
            None => self.goal(),
        }
    }

    /// Take a setting, and play the way to it from what is drawn now.
    fn apply(&mut self, setup: &Setup) {
        let from = self.shown();
        self.a = matrix_of(setup.rows);
        let sys = System::of(dmat(self.a));
        self.b = match setup.b {
            Aim::At(x, y) => DVec2::new(x, y),
            Aim::Along(k, length) => sys.u[k.min(1)] * length,
        };
        self.turn = match setup.err {
            Aim::At(x, y) => y.atan2(x),
            Aim::Along(k, sign) => {
                let d = sys.u[k.min(1)] * sign;
                d.y.atan2(d.x)
            }
        };
        (self.eps, self.lambda) = (setup.eps, setup.lambda);
        self.begin(from);
    }

    fn set_up(&mut self, k: usize) {
        self.apply(&TRIALS[k].2);
    }

    /// Play the way from `from` to the setting now asked for, and frame the panes on
    /// where it ends.
    fn begin(&mut self, from: Shown) {
        self.change = Some((from, 0.0));
        self.refit = true;
    }

    /// Move a change of setting on by `dt` seconds of the scene's time (all the way
    /// if its time stands still: a click must still be answered).
    fn advance(&mut self, dt: f32) {
        if let Some((_, t)) = self.change.as_mut() {
            *t = if dt > 0.0 { *t + dt / CHANGE_TIME } else { 1.0 };
            if *t >= 1.0 {
                self.change = None;
            }
        }
    }

    /// Head the framing for what the setting asked for will show: when it is new, or
    /// when what is now aimed at does not show it well (but not while a handle is held:
    /// the panes keep their scale while something is dragged in them).
    ///
    /// The close-ups are the exception: they follow b and x wherever they are dragged,
    /// keeping their scale (the circle and the ellipse do not change their size as b
    /// moves), so that what is round b and x stays in view as they go.
    fn frame_on(&mut self, lay: &Layout, holding: bool) {
        let p = Picture::of(&self.goal());
        let mut moved = false;
        for k in 0..lay.count() {
            let (_, half) = lay.room(k);
            let (lo, hi) = p.content(lay.panes[k]);
            let follow = holding && lay.panes[k].close;
            if self.refit || follow || (!holding && !self.aim[k].shows(half, lo, hi)) {
                let to = Framing::fit(half, lo, hi);
                if follow && !self.refit {
                    (self.aim[k], self.framing[k]) = (to, to);
                } else {
                    moved |= to != self.aim[k];
                    self.aim[k] = to;
                }
            }
        }
        if self.snap {
            self.framing = self.aim;
            self.glide = None;
        } else if moved {
            // (in step with a change of setting that has just begun, so that the
            // picture and its framing arrive together)
            let in_step = self.change.is_some_and(|(_, t)| t < 0.25);
            self.glide = Some(Glide { from: self.framing, t: self.change.filter(|_| in_step).map_or(0.0, |(_, t)| t), in_step });
        }
        (self.refit, self.snap) = (false, false);
    }

    /// Move the framing's glide on: in step with the change of setting it began with,
    /// or by `dt` seconds of the clock (it answers the hand). (Not the close-ups while a
    /// handle is held: they follow b and x.)
    fn glide_on(&mut self, lay: &Layout, dt: f32, holding: bool) {
        let change = self.change;
        let shown = self.shown();
        let goal_has_inverse = System::of(dmat(self.a)).inv.is_some();
        let Some(g) = self.glide.as_mut() else {
            self.framing = self.aim;
            return;
        };
        g.t = match (g.in_step, change) {
            (true, Some((_, t))) => t,
            (true, None) => 1.0,
            (false, _) => g.t + dt.max(0.0) / GLIDE_TIME,
        };
        let s = smoothstep(g.t) as f64;
        // in step with a change between two matrices with inverses, the framing heads for
        // what is drawn at each moment, not for where it will end: on the way, x can go
        // far beyond where it starts and ends (and out of view), the answers' picture
        // being A⁻¹ of the readings' all the way
        let follow = match (g.in_step, change) {
            (true, Some((from, _))) => System::of(from.a).inv.is_some() && goal_has_inverse,
            _ => false,
        };
        let now = follow.then(|| (Picture::of(&shown), Picture::of(&change.map_or(shown, |c| c.0))));
        for k in 0..4 {
            if holding && lay.panes[k].close {
                continue;
            }
            let (mid, half) = lay.room(k);
            self.framing[k] = match (&now, k < lay.count()) {
                (Some((now, start)), true) => {
                    // the fit of what is drawn now, less what the framing at the start
                    // differed from the fit of what was drawn then, by less and less
                    let fit = |p: &Picture| {
                        let (lo, hi) = p.content(lay.panes[k]);
                        Framing::fit(half, lo, hi)
                    };
                    fit(now).shifted(&g.from[k], &fit(start), mid, 1.0 - s)
                }
                _ => g.from[k].towards(&self.aim[k], mid, s),
            };
        }
        if g.t >= 1.0 {
            self.framing = self.aim;
            self.glide = None;
        }
    }

    fn maps(&self, lay: &Layout) -> [Map; 4] {
        [0, 1, 2, 3].map(|k| Map::of(lay, k, self.framing[k]))
    }

    /// The matrix's numbers and its quick choices, in the panel. A number typed or a
    /// choice taken is played from what is drawn; one dragged is followed as it goes.
    fn matrix_controls(&mut self, ui: &mut egui::Ui) {
        let from = self.shown();
        let mut chosen = false;
        ui.horizontal_wrapped(|ui| {
            for (name, r) in PRESETS {
                crate::typeset::seen(name);
                if ui.button(ui::label_text(ui, name, 12.5, None, false)).clicked() {
                    self.a = matrix_of(r);
                    chosen = true;
                }
            }
        });
        let mut a = self.a;
        let typed = mat2_edit(ui, &mut a);
        if a.is_finite() {
            // (within ±100: what the editor holds)
            self.a = Mat2::from_cols(a.x_axis.max(Vec2::splat(-100.0)).min(Vec2::splat(100.0)), a.y_axis.max(Vec2::splat(-100.0)).min(Vec2::splat(100.0)));
        }
        if chosen || typed {
            self.begin(from);
        }
    }

    /// The handles: b and x (white) in the pictures to scale, the error (pink) on the
    /// circle and on the ellipse where the details are. Each is kept within its pane.
    fn handles(&mut self, f: &mut Frame, lay: &Layout, maps: &[Map; 4]) {
        let p = Picture::of(&self.goal());
        let keep = |k: usize, s: Vec2| {
            let (lo, hi) = lay.bounds(k);
            s.max(lo).min(hi)
        };
        for k in 0..lay.count() {
            let (pane, map) = (lay.panes[k], &maps[k]);
            let id = 1 + 2 * k as u32;
            let (from, d) = match (pane.space, p.x, p.dx) {
                (0, ..) => (p.b, p.db),
                (_, Some(x), Some(dx)) => (x, dx),
                _ => continue,
            };
            if !pane.close {
                let mut at = map.to(from);
                if f.drag2(id, &mut at, TRUTH) {
                    let q = map.back(keep(k, at));
                    let b = if pane.space == 0 { q } else { p.sys.a * q };
                    // (each drag can take b to the edge of a picture that then widens to
                    // show it: no further than a million, however often that is done)
                    if b.is_finite() {
                        self.b = b.clamp_length_max(1e6);
                    }
                }
            }
            if lay.details(k) && p.eps > 0.0 {
                let mut at = grip(map, from, d);
                if f.drag2(id + 1, &mut at, ERROR) {
                    let q = map.back(keep(k, at)) - from;
                    let e = if pane.space == 0 { q } else { p.sys.a * q };
                    if e.length() > 0.0 && e.is_finite() {
                        self.turn = e.y.atan2(e.x);
                    }
                }
            }
        }
    }

    /// Turn u₁, u₂ (with v₁, v₂) and the lines' directions the way they were last drawn,
    /// and remember them. Returns the lines' directions.
    fn steady(&mut self, p: &mut Picture) -> [DVec2; 2] {
        for k in 0..2 {
            if p.sys.u[k].dot(self.kept[k]) < 0.0 {
                (p.sys.u[k], p.sys.v[k]) = (-p.sys.u[k], -p.sys.v[k]);
            }
            self.kept[k] = p.sys.u[k];
        }
        let mut along = [DVec2::X; 2];
        for (i, d) in along.iter_mut().enumerate() {
            let mut a = row(p.sys.a, i).perp().normalize_or(self.kept[2 + i]);
            if a.dot(self.kept[2 + i]) < 0.0 {
                a = -a;
            }
            (*d, self.kept[2 + i]) = (a, a);
        }
        along
    }

    /// What the numbers come to, in words.
    fn verdict(&self, p: &Picture) -> String {
        let sys = p.sys;
        match sys.rank() {
            0 => return "$A = 0$: every $x$ gives $Ax = 0$, so no reading but $0$ has an answer, and $0$ has every $x$.".into(),
            1 => {
                return match p.strip() {
                    Some(_) => r"$A$ squashes the plane onto a line ($\det A = 0$): the two lines are parallel, and $\kappa = ∞$. A reading on the line $A$ squashes onto (the orange dashed line on the left) has a whole line of answers; any other has none. The readings within $\varepsilon$ that do have answers give a strip without end, not an ellipse.".into(),
                    None => r"$A$ squashes the plane onto a line ($\det A = 0$): the two lines are parallel, and $\kappa = ∞$. No reading within $\varepsilon$ of $b$ lies on the line $A$ squashes onto (the orange dashed line on the left), so none of them has an answer.".into(),
                };
            }
            _ => {}
        }
        let k = sys.kappa();
        // (the numbers set as formulas, so that a very large or small one, which `sig`
        // writes 1.2e7, is set as 1.2 × 10⁷)
        let mut text = format!(r"$\kappa = {}$: whatever $b$ and the error, $x$'s relative error is between ${}$ and ${}$ times $b$'s.", sig(k, 3), sig(1.0 / k, 3), sig(k, 3));
        match (p.rel_b(), p.ratio()) {
            (None, _) => text += r" Here $b = 0$, so $x = 0$ too: there is no relative error to speak of (it would be $\varepsilon/0$).",
            (_, None) if p.eps == 0.0 => text += r" Here $\varepsilon = 0$: no error, so nothing to amplify.",
            (_, Some(r)) => text += &format!(" Here it is ${}$ times.", sig(r, 3)),
            _ => {}
        }
        text
    }
}

impl Scene for Condition {
    fn view(&self) -> View {
        STAGE_VIEW
    }

    fn settings(&self, s: &mut ViewSettings) {
        s.grid = false;
        s.bloom = s.bloom.min(0.2);
    }

    /// Moving only while a change of setting plays or the framing glides.
    fn animated(&self) -> bool {
        self.change.is_some() || self.glide.is_some()
    }

    fn pointer_hint(&self) -> Option<&'static str> {
        Some("white dots: move $b$, or $x$    pink dots: turn the error    drag elsewhere: move    scroll: zoom")
    }

    fn ui(&mut self, ui: &mut egui::Ui, _: &Gfx) {
        ui::note(ui, r"Two equations in two unknowns, $Ax = b$, are two lines, and the answer $x$ is where they cross. But $b$ is measured, and every measurement is a little off. Here the reading of $b$ is off by $\delta b$, an arrow of length $\varepsilon$ that can point any way. The pictures of $b$ show every reading $b + \delta b$ could be; the pictures of $x$, the answer each one gives. $A^{-1}$ turns the circle of readings into an ellipse of answers. When the lines are nearly parallel the ellipse is long and thin: a small error in $b$ is a large error in $x$.");
        ui::note(ui, "Drag $b$ and $x$ (white dots), and the error (pink dot) round its circle or its ellipse.");

        ui::section(ui, "Things to try");
        for (k, trial) in TRIALS.iter().enumerate() {
            if ui::experiment(ui, trial.0, trial.1) {
                self.set_up(k);
            }
        }

        ui::section(ui, "Matrix $A$");
        self.matrix_controls(ui);
        ui::note(ui, "The buttons choose");
        ui::equation(ui, r"A = \begin{bmatrix} 1 & c \\ c & 1 \end{bmatrix}");
        ui::note(ui, r"with $\kappa = 1, 3, 9, 19, 99$ or $\infty$: two lines ever nearer parallel. Or drag a number sideways, or click it and type. Row $i$ is the left-hand side of equation $i$.");
        ui::section(ui, "The error");
        ui::slider_log_or_zero(ui, r"$\varepsilon$, the length of $\delta b$", &mut self.eps, 0.001, 1.0);
        ui::note(ui, "The yellow circle holds every error up to this length.");
        ui::section(ui, "Ridge");
        ui::slider_log_or_zero(ui, r"$\lambda$", &mut self.lambda, 0.0001, 10.0);
        ui::note(ui, r"0: the plain answer $A^{-1}b$ alone. Above 0, ridge's answer $(A^\top A + \lambda I)^{-1} A^\top b$ is drawn as well, in green, with its own ellipse.");

        ui::section(ui, "Reading the picture");
        ui::key(
            ui,
            &[
                (TRUTH, r"white: $b$, the right-hand side, and $x = A^{-1}b$, its answer (each from $0$)"),
                (ERROR, r"pink: the error $\delta b$ in a reading of $b$, and the error $\delta x$ it makes in $x$ (side by side, a faint line joins the two); dashed, the lines that reading gives, which cross at $x + \delta x$"),
                (DOUBT, r"yellow: every reading within $\varepsilon$ of $b$, and the answer each one gives: $A^{-1}$ of the circle, an ellipse"),
                (LINE, r"blue: the equations as lines, and the band each can be in when its number of $b$ is off by up to $\varepsilon$ (in the pictures of $b$: the strips where each number is off by up to $\varepsilon$)"),
                (MOST, r"orange: $u_1$, the direction of error that $A^{-1}$ leaves shortest, and $v_1$, the way it then points: the ellipse's short half-axis, $\varepsilon/\sigma_1$. Dashed, when $A$ squashes the plane flat: the line every $Ax$ is on"),
                (LEAST, r"purple: $u_2$, the direction of error that $A^{-1}$ makes longest, and $v_2$: the ellipse's long half-axis, $\varepsilon/\sigma_2$"),
                (RIDGE, r"green (when $\lambda > 0$): ridge's answer, the ellipse of its answers, and how far its middle is from $x$"),
            ],
        );
        ui::note(ui, "On a wide screen the pictures above show $b$ and $x$ to scale, from $0$, and those below are close-ups round them; on a narrow one, only the pictures to scale. Each picture has its own scale: read the numbers along its edges.");
        ui::section(ui, "Words");
        ui::term(ui, "Relative error:", r"the size of an error as a share of the size of what it is in: $|\delta b| / |b|$ for $b$, $|\delta x| / |x|$ for $x$.");
        ui::term(ui, r"Singular values $\sigma_1, \sigma_2$:", r"the most and the least $A$ stretches any arrow. $A$ sends $v_1$ to $\sigma_1 u_1$ and $v_2$ to $\sigma_2 u_2$, so $A^{-1}$ divides an error along $u_1$ by $\sigma_1$ and an error along $u_2$ by $\sigma_2$: of all errors of one length, the first comes out shortest and the second longest.");
        ui::term(ui, r"Condition number $\kappa$:", r"$\sigma_1/\sigma_2$. However $b$ and the error are chosen, $x$'s relative error is between $1/\kappa$ and $\kappa$ times $b$'s. $\kappa = 1$ for turns, mirror images and scalings of them, the best there is; $\kappa = \infty$ when $A$ squashes the plane flat.");
        ui::term(ui, "Ill-conditioned:", r"with a large $\kappa$: a small error in $b$ can make a large error in $x$.");
        ui::term(ui, "Ridge (regularisation):", r"solving $(A^\top A + \lambda I)x = A^\top b$ instead of $Ax = b$. It shrinks the part of $x$ along each $v_k$ by $\sigma_k^2/(\sigma_k^2 + \lambda)$: hardly at all where $\sigma_k$ is large, much where it is small, which is where errors are blown up.");

        ui::section(ui, "Numbers");
        let p = Picture::of(&self.goal());
        let sys = p.sys;
        // (every value a formula, as the points must be and as a very large or small
        // number must be, which `sig` writes 1.2e7: so that the list is set alike)
        let value = |v: f64| format!("${}$", sig(v, 3));
        let point = |v: DVec2| format!("$({}, {})$", sig(v.x, 3), sig(v.y, 3));
        let share = |v: f64| format!("${}$", pct_tex(v));
        ui::readout_colored(ui, r"$\sigma_1$", value(sys.s[0]), MOST);
        ui::readout_colored(ui, r"$\sigma_2$", value(sys.s[1]), LEAST);
        ui::readout(ui, r"$\kappa = \sigma_1/\sigma_2$", value(sys.kappa()));
        ui::readout(ui, r"$\det A$", value(if sys.inv.is_some() { sys.a.determinant() } else { 0.0 }));
        ui::readout_colored(ui, "$b$", point(p.b), TRUTH);
        match p.x {
            Some(x) => ui::readout_colored(ui, "$x = A^{-1}b$", point(x), TRUTH),
            None => ui::readout_colored(ui, "$x = A^{-1}b$", "none: $A$ has no inverse", TRUTH),
        }
        ui::readout_colored(ui, "relative error in $b$", p.rel_b().map_or("none: $b = 0$".to_string(), share), ERROR);
        ui::readout_colored(ui, "relative error in $x$", p.rel_x().map_or("none".to_string(), share), ERROR);
        ui::readout(ui, "$x$'s relative error ÷ $b$'s", p.ratio().map_or("none".to_string(), value));
        if let Some((now, bias)) = p.ridge_errors() {
            ui::readout_colored(ui, "ridge: off by, this reading", share(now), RIDGE);
            ui::readout_colored(ui, "ridge: off by, with no error", share(bias), RIDGE);
        }
        ui::note(ui, &self.verdict(&p));
        ui::see_also(
            ui,
            &[
                ("svd", r"where $\sigma_1$, $\sigma_2$, the $u$'s and the $v$'s come from"),
                ("overdet", "more equations than unknowns, with a patch of doubt of its own"),
                ("elim", "solving the equations by elimination"),
                ("quadform", "an ellipse as a level curve of a quadratic form"),
            ],
        );
    }

    fn frame(&mut self, f: &mut Frame) {
        let lay = Layout::of(f.view_size(), f.top_inset);
        // a new canvas (or the first) is framed afresh, at once (but not one squeezed
        // to a sliver: the panes are as they were when it opens out again)
        if self.layout != Some(lay) && lay.view.x >= 200.0 && lay.view.y >= 150.0 {
            self.layout = Some(lay);
            (self.refit, self.snap) = (true, true);
        }
        self.advance(f.dt);
        self.glide_on(&lay, f.real_dt, f.handles.dragging());
        let maps = self.maps(&lay);
        if self.change.is_none() {
            self.handles(f, &lay, &maps);
        }
        self.frame_on(&lay, f.handles.dragging());
        let mut p = Picture::of(&self.shown());
        let along = self.steady(&mut p);
        let look = Look { live: self.change.is_none(), moving: self.change.is_some() || self.glide.is_some(), holding: f.handles.dragging(), along };
        draw(f, &lay, &self.maps(&lay), &p, look);
    }
}

/// The stalk from the end of the error `d` (from `from`) to its handle, when the handle
/// is kept further out than the end; while a change plays, the dot that stands for the
/// handle is at the end itself (the error can pass through nothing on its way, and has
/// no direction to keep the handle out along then).
fn handle_stalk(f: &mut Frame, map: &Map, from: DVec2, d: DVec2, live: bool) {
    let (end, g) = (map.to(from + d), grip(map, from, d));
    if live && g.distance(end) > 1.0 && in_view(map, from + d) {
        f.line(end, g, 1.2, ERROR.a(0.6));
    } else {
        f.line(NOWHERE, NOWHERE, 1.2, ERROR.a(0.0));
    }
    if !live {
        dot(f, map, from + d, 12.0, ERROR);
    }
}

/// Where the handle for the error `d` from `from` goes: at the end of `d`, or at least
/// `GRIP` pixels from `from` along it where `d` is shorter than that on the screen.
fn grip(map: &Map, from: DVec2, d: DVec2) -> Vec2 {
    let short = GRIP as f64 * map.per_px();
    let len = d.length();
    let reach = if len > 0.0 && len < short { d * (short / len) } else { d };
    map.to(from + reach)
}

// ----- drawing --------------------------------------------------------------------------

fn v3(p: Vec2) -> Vec3 {
    p.extend(0.0)
}

/// A filled convex polygon of a pane's plane.
fn fill(f: &mut Frame, map: &Map, poly: &[DVec2], color: Color) {
    if poly.len() < 3 {
        return;
    }
    let first = v3(map.to(poly[0]));
    for w in poly[1..].windows(2) {
        f.dl.tris_alpha.tri(first, v3(map.to(w[0])), v3(map.to(w[1])), color);
    }
}

/// Where a piece of the picture goes when it is out of its pane's view: far off any
/// canvas (stage units), drawn see-through. Each piece is drawn in every frame, in view
/// or not, so that the list of what is drawn keeps its order from frame to frame; the
/// check of how evenly the picture moves (`eidos motion`) compares frames point by point.
const NOWHERE: Vec2 = Vec2::new(-1e4, -1e4);

/// A closed curve of a pane's plane, cut where it leaves what the pane shows: as one
/// short segment for each step round it.
fn outline(f: &mut Frame, room: &mut Room, map: &Map, pts: &[DVec2], width: f32, color: Color) {
    let seen = map.seen();
    for i in 0..pts.len() {
        match segment_in(pts[i], pts[(i + 1) % pts.len()], seen) {
            Some((p, q)) => f.line(map.to(p), map.to(q), width, color),
            None => f.line(NOWHERE, NOWHERE, width, color.a(0.0)),
        }
    }
    for i in (0..pts.len()).step_by(8) {
        if let Some((p, q)) = segment_in(pts[i], pts[(i + 8) % pts.len()], seen) {
            room.line(f, v3(map.to(p)), v3(map.to(q)));
        }
    }
}

/// A segment of a pane's plane, cut to what it shows.
fn seg(f: &mut Frame, room: &mut Room, map: &Map, a: DVec2, b: DVec2, width: f32, color: Color) {
    match segment_in(a, b, map.seen()) {
        Some((p, q)) => {
            let (p, q) = (map.to(p), map.to(q));
            f.line(p, q, width, color);
            room.line(f, v3(p), v3(q));
        }
        None => f.line(NOWHERE, NOWHERE, width, color.a(0.0)),
    }
}

/// An arrow of a pane's plane: with its head if its tip is in view.
fn arrow(f: &mut Frame, room: &mut Room, map: &Map, a: DVec2, b: DVec2, width: f32, color: Color) {
    let Some((p, q)) = segment_in(a, b, map.seen()) else {
        f.line(NOWHERE, NOWHERE, width, color.a(0.0));
        return;
    };
    let (sp, sq) = (map.to(p), map.to(q));
    // (the head is a fill; the shaft's two ends are what is kept in order)
    if q == b && (sq - sp).length() > 1e-6 {
        f.arrow(sp, sq, width, color);
    } else {
        f.line(sp, sq, width, color);
    }
    room.line(f, v3(sp), v3(sq));
}

/// Is a point of the plane in what the pane shows?
fn in_view(map: &Map, p: DVec2) -> bool {
    let (lo, hi) = map.seen();
    p.cmpge(lo).all() && p.cmple(hi).all()
}

fn dot(f: &mut Frame, map: &Map, p: DVec2, size: f32, color: Color) {
    if in_view(map, p) {
        f.dot(map.to(p), size, color);
    } else {
        f.dot(NOWHERE, size, color.a(0.0));
    }
}

/// A thin line of the grid, drawn as a sliver of a fill: while a pane zooms, its grid
/// is spaced afresh (a step of 5 becomes one of 10), which is the backdrop changing,
/// not anything in the picture moving, and is drawn as the backdrop is.
fn grid_line(f: &mut Frame, a: Vec2, b: Vec2, width: f32, color: Color) {
    let across = (b - a).normalize_or_zero().perp() * (width * 0.5);
    f.dl.tris_alpha.quad(v3(a - across), v3(b - across), v3(b + across), v3(a + across), color);
}

/// The line n·p = c of a pane's plane (a line without end), as far as the pane shows
/// it, `width` pixels wide; names keep off it. It is drawn as a sliver of fill, like
/// the grid: seen through the pane's window, the ends of a turning line race along its
/// edges, the faster the nearer the line is to lying along one. They are where the
/// window cuts the line, not points of anything in the picture.
fn endless(f: &mut Frame, room: &mut Room, map: &Map, n: DVec2, c: f64, width: f32, color: Color) -> Option<(DVec2, DVec2)> {
    let (s, e) = line_in(n, c, map.seen())?;
    let (a, b) = (map.to(s), map.to(e));
    grid_line(f, a, b, width / map.ppu as f32, color);
    room.line(f, v3(a), v3(b));
    Some((s, e))
}

/// The same dashed, the dashes kept in step with the point `anchor` of the plane (its
/// foot on the line), so that they move with the line rather than crawl along it.
fn endless_dashed(f: &mut Frame, map: &Map, n: DVec2, c: f64, anchor: DVec2, width: f32, color: Color) {
    let Some((s, e)) = line_in(n, c, map.seen()) else { return };
    let len = n.length();
    let (n, c) = (n / len, c / len);
    let foot = anchor - n * (n.dot(anchor) - c);
    let along = n.perp();
    // (in pixels along the line, from the foot)
    let px = 1.0 / map.per_px();
    let (t0, t1) = ((s - foot).dot(along) * px, (e - foot).dot(along) * px);
    let (t0, t1) = (t0.min(t1), t0.max(t1));
    let period = 9.0;
    if !(t0.is_finite() && t1.is_finite()) || (t1 - t0) / period > 2000.0 {
        return;
    }
    let mut t = (t0 / period).floor() * period;
    while t < t1 {
        let (a, b) = (t.max(t0), (t + 0.55 * period).min(t1));
        if b > a {
            let at = |u: f64| map.to(foot + along * (u / px));
            grid_line(f, at(a), at(b), width / map.ppu as f32, color);
        }
        t += period;
    }
}

/// The backing of a pane, its grid, and (unless the pane is being framed afresh) the
/// numbers along its edges.
fn backdrop(f: &mut Frame, room: &mut Room, lay: &Layout, map: &Map, k: usize, moving: bool) {
    let (lo, hi) = lay.bounds(k);
    f.dl.tris_alpha.rect(lo, hi, 0.0, Color::hex(0x0B0F1A).a(0.9));
    frame_rect(f, lo, hi, 1.0, pal::FAINT.a(0.7));
    let (slo, shi) = map.seen();
    let px = map.per_px();
    let step = nice_step((90.0 * px) as f32) as f64;
    if !(step > 0.0 && step.is_finite()) {
        return;
    }
    let ticks = |a: f64, b: f64| {
        let first = (a / step).ceil();
        let n = (((b - a) / step).floor().max(0.0) as usize).min(60);
        (0..=n).map(move |i| (first + i as f64) * step).filter(move |v| *v <= b)
    };
    // (a pixel, in stage units)
    let w = 1.0 / lay.ppu;
    for x in ticks(slo.x, shi.x) {
        let zero = x.abs() < step * 1e-6;
        grid_line(f, map.to(DVec2::new(x, slo.y)), map.to(DVec2::new(x, shi.y)), if zero { 1.4 * w } else { w }, pal::FAINT.a(if zero { 0.9 } else { 0.32 }));
        if !moving && !zero && x > slo.x + 30.0 * px && x < shi.x - 16.0 * px {
            f.text_off(map.to(DVec2::new(x, slo.y)), Vec2::new(0.0, -3.0), fmt_tick(x as f32, step as f32), 10.5, pal::DIM, Align2::CENTER_BOTTOM);
        }
    }
    for y in ticks(slo.y, shi.y) {
        let zero = y.abs() < step * 1e-6;
        grid_line(f, map.to(DVec2::new(slo.x, y)), map.to(DVec2::new(shi.x, y)), if zero { 1.4 * w } else { w }, pal::FAINT.a(if zero { 0.9 } else { 0.32 }));
        if !moving && !zero && y > slo.y + 20.0 * px && y < shi.y - 12.0 * px {
            f.text_off(map.to(DVec2::new(slo.x, y)), Vec2::new(4.0, 0.0), fmt_tick(y as f32, step as f32), 10.5, pal::DIM, Align2::LEFT_CENTER);
        }
    }
    // names keep off the numbers along the edges
    let (top_left, bottom_right) = lay.panes[k].rect;
    room.wall(Vec2::new(top_left.x, bottom_right.y - 16.0), bottom_right);
    room.wall(top_left, Vec2::new(top_left.x + 36.0, bottom_right.y));
    // the axes' names, at their far ends
    let names = if lay.panes[k].space == 0 { ["$b_1$", "$b_2$"] } else { ["$x_1$", "$x_2$"] };
    name(f, room, !moving && slo.y < 0.0 && shi.y > 0.0, |f, room| room.label(f, ("axis", 2 * k), v3(map.to(DVec2::new(shi.x, 0.0))), names[0], 12.0, pal::DIM, Vec2::new(-1.0, 1.0), 4.0, true));
    name(f, room, !moving && slo.x < 0.0 && shi.x > 0.0, |f, room| room.label(f, ("axis", 2 * k + 1), v3(map.to(DVec2::new(0.0, shi.y))), names[1], 12.0, pal::DIM, Vec2::new(1.0, 1.0), 4.0, true));
}

/// A name, written by `write` if `when`.
fn name(f: &mut Frame, room: &mut Room, when: bool, write: impl FnOnce(&mut Frame, &mut Room) -> bool) {
    if when {
        write(f, room);
    }
}

/// The direction on the screen of a step `d` from `from` in a pane.
fn screen_dir(f: &Frame, map: &Map, from: DVec2, d: DVec2) -> Vec2 {
    match (on_screen(f, v3(map.to(from))), on_screen(f, v3(map.to(from + d)))) {
        (Some(a), Some(b)) => (b - a).normalize_or_zero(),
        _ => Vec2::ZERO,
    }
}

/// How wide a label is: words by their number of characters, as `text_width` counts
/// them, and a text with formulas in it exactly (measured once and kept by the frame,
/// where `text_width` would work its formulas out again at every call).
fn width(f: &Frame, text: &str, size: f32) -> f32 {
    if crate::typeset::has_math(text) {
        f.text_size(text, size).x
    } else {
        text_width(text, size)
    }
}

/// A few lines of words on a plate in a pane's top right-hand corner, each in its own
/// colour; names keep off them.
fn corner_note(f: &mut Frame, room: &mut Room, lay: &Layout, k: usize, lines: &[(String, Color)]) {
    let (lo, hi) = lay.panes[k].rect;
    let size = 12.5;
    let width = lines.iter().map(|(t, _)| width(f, t, size)).fold(0.0, f32::max);
    if hi.x - lo.x < width + 80.0 || hi.y - lo.y < 90.0 {
        return;
    }
    let right = hi.x - 10.0;
    for (i, (text, c)) in lines.iter().enumerate() {
        f.caption_px(Vec2::new(right, lo.y + 10.0 + i as f32 * (size + 8.0)), text.as_str(), size, *c, Align2::RIGHT_TOP);
    }
    room.wall(Vec2::new(right - width - 12.0, lo.y + 4.0), Vec2::new(right + 6.0, lo.y + 16.0 + lines.len() as f32 * (size + 8.0)));
}

/// How the picture is to be drawn this frame.
#[derive(Clone, Copy, Debug)]
struct Look {
    /// The handles are offered (no change of setting is playing): else they are drawn
    /// as plain dots where they will be.
    live: bool,
    /// A change of setting plays, or a pane is being framed afresh: the numbers along
    /// the edges and the names of things, but for b and x, wait until it is still
    /// (looking for room as the picture moves, they would leap from place to place, and
    /// come and go).
    moving: bool,
    /// A handle is held: the close-ups follow b and x, and their numbers wait too.
    holding: bool,
    /// The direction along each equation's line, kept the same way round from frame to
    /// frame, so that its name stays at the same end.
    along: [DVec2; 2],
}

/// The whole picture: the panes of b and of x, the arrows between them, the numbers.
fn draw(f: &mut Frame, lay: &Layout, maps: &[Map; 4], p: &Picture, look: Look) {
    let mut room = Room::new(f);
    let view = lay.view;
    // names stay in their own pane: everything between and round the panes is walled off
    let panes = &lay.panes[..lay.count()];
    let left = panes.iter().map(|q| q.rect.0.x).fold(f32::MAX, f32::min);
    let right = panes.iter().map(|q| q.rect.1.x).fold(f32::MIN, f32::max);
    let top = panes.iter().map(|q| q.rect.0.y).fold(f32::MAX, f32::min);
    let bottom = panes.iter().map(|q| q.rect.1.y).fold(f32::MIN, f32::max);
    room.wall(Vec2::ZERO, Vec2::new(view.x, top));
    room.wall(Vec2::new(0.0, bottom), view);
    room.wall(Vec2::ZERO, Vec2::new(left, view.y));
    room.wall(Vec2::new(right, 0.0), view);
    let [first, second, third, _] = lay.panes;
    if lay.narrow {
        room.wall(Vec2::new(0.0, first.rect.1.y), Vec2::new(view.x, second.rect.0.y));
    } else {
        room.wall(Vec2::new(first.rect.1.x, 0.0), Vec2::new(second.rect.0.x, view.y));
        if lay.split {
            room.wall(Vec2::new(0.0, first.rect.1.y), Vec2::new(view.x, third.rect.0.y));
        }
    }

    for k in 0..lay.count() {
        if lay.panes[k].space == 0 {
            draw_b(f, &mut room, lay, k, &maps[k], p, look);
        } else {
            draw_x(f, &mut room, lay, k, &maps[k], p, look);
        }
    }
    // the error and what it does, joined across the gap: the same reading, in both
    // (beside each other only: across a phone's pictures it would cross them both)
    if let (Some(dx), Some(x), false) = (p.dx, p.x, p.eps == 0.0 || lay.narrow) {
        let (i, j) = if lay.split { (2, 3) } else { (0, 1) };
        if in_view(&maps[i], p.b + p.db) && in_view(&maps[j], x + dx) {
            f.dashed(maps[i].to(p.b + p.db), maps[j].to(x + dx), 1.2, ERROR.a(0.4));
        }
    }
    between_panes(f, lay, p);
    titles(f, lay, p.sys.inv.is_some());
    if let Some(at) = lay.numbers {
        numbers(f, at, p);
    }
}

/// How strongly the singular directions are drawn: not at all when κ = 1, when every
/// direction is stretched alike and none of them is u₁ more than another, and fully
/// from κ = 1.05 on (so that they fade as a matrix comes to κ = 1, rather than spin).
fn distinct(sys: &System) -> f32 {
    ((sys.kappa() - 1.0) / 0.05).clamp(0.0, 1.0) as f32
}

/// The pane of the readings: b, the circle of every reading within ε, the strips where
/// each number of b is off by up to ε, u₁ and u₂, and the error δb.
fn draw_b(f: &mut Frame, room: &mut Room, lay: &Layout, k: usize, map: &Map, p: &Picture, look: Look) {
    backdrop(f, room, lay, map, k, look.moving || (look.holding && lay.panes[k].close));
    let live = look.live;
    let seen = map.seen();
    let (b, db, eps) = (p.b, p.db, p.eps);
    let details = lay.details(k);
    let n = k as u32;
    // each number of b off by up to ε: a strip for each, crossing in a square round b
    if details && eps > 0.0 {
        for axis in [DVec2::X, DVec2::Y] {
            fill(f, map, &band_in(axis, axis.dot(b), eps, seen), LINE.a(if lay.panes[k].close { 0.09 } else { 0.05 }));
        }
    }
    // when A squashes the plane onto a line, every Ax lies on it
    if p.sys.rank() == 1 {
        if let Some((s, e)) = line_in(p.sys.u[1], 0.0, seen) {
            endless_dashed(f, map, p.sys.u[1], 0.0, DVec2::ZERO, 1.6, MOST.a(0.85));
            room.line(f, v3(map.to(s)), v3(map.to(e)));
            // (towards the end u₁ points to, which stays the same end as the line turns)
            let (s, e) = if (e - s).dot(p.sys.u[0]) > 0.0 { (s, e) } else { (e, s) };
            let spots: Vec<Vec3> = [0.85, 0.65, 0.45, 0.25].iter().map(|t| v3(map.to(s.lerp(e, *t)))).collect();
            name(f, room, details && !look.moving, |f, room| room.label_any(f, ("image", n), &spots, "every $Ax$ is on this line", 12.0, MOST, p.sys.u[0].perp().as_vec2() * Vec2::new(1.0, -1.0), 8.0, true));
        }
    }
    // the circle of readings
    let circle: Vec<DVec2> = (0..128).map(|i| b + DVec2::from_angle(i as f64 / 128.0 * std::f64::consts::TAU) * eps).collect();
    if eps > 0.0 {
        fill(f, map, &clip_to(&circle, seen), DOUBT.a(0.14));
        outline(f, room, map, &circle, 2.0, DOUBT);
    }
    // u₁ and u₂ through b, at least long enough to be seen
    let reach = (eps * 1.35).max(30.0 * map.per_px());
    let strength = distinct(&p.sys);
    if details {
        for (i, c) in [(0, MOST), (1, LEAST)] {
            let d = p.sys.u[i] * reach;
            // (with κ = 1 no direction is u₁ more than another, and the one worked
            // out is arbitrary: drawn nowhere)
            if strength > 0.0 {
                seg(f, room, map, b - d, b + d, 1.8, c.a(0.9 * strength));
            } else {
                f.line(NOWHERE, NOWHERE, 1.8, c.a(0.0));
            }
        }
    }
    // b, and the error
    arrow(f, room, map, DVec2::ZERO, b, 2.6, TRUTH);
    if eps > 0.0 {
        arrow(f, room, map, b, b + db, 2.2, ERROR);
        dot(f, map, b + db, 7.0, ERROR);
        if details {
            handle_stalk(f, map, b, db, live);
        }
    }
    if !live && !lay.panes[k].close {
        dot(f, map, b, 12.0, TRUTH);
    }
    // the names, the most important first
    let out = |f: &Frame, d: DVec2| screen_dir(f, map, DVec2::ZERO, d);
    let along = if b.length() > 0.0 { out(f, b) } else { Vec2::new(1.0, -1.0) };
    // beside the arrow, a little back from its tip
    name(f, room, in_view(map, b), |f, room| room.label(f, ("b", n), v3(map.to(b)), "$b$", 16.0, TRUTH, along.perp() - along * 0.6, 9.0, false));
    // (not while the error passes across the circle on its way from one direction to
    // another: its name would leap from one side of b to the other)
    let named_error = eps > 0.0 && in_view(map, b + db) && db.length() >= 0.5 * eps && !look.moving;
    if details {
        name(f, room, named_error, |f, room| room.label(f, ("δb", n), v3(grip(map, b, db)), r"$\delta b$", 15.0, ERROR, out(f, db), 9.0, false));
        for (i, (text, c)) in [("$u_1$", MOST), ("$u_2$", LEAST)].into_iter().enumerate() {
            let d = p.sys.u[i] * reach;
            // (at the end it points to, or nowhere: moved to the other end, the name
            // would leap across the circle)
            name(f, room, strength > 0.0 && in_view(map, b + d) && !look.moving, |f, room| room.label(f, ("u", 2 * n + i as u32), v3(map.to(b + d)), text, 14.0, c.a(strength), out(f, p.sys.u[i]), 6.0, true));
        }
        let text = if eps > 0.0 { format!(r"radius $\varepsilon = {}$", sig(eps, 2)) } else { r"$\varepsilon = 0$: no error".to_string() };
        if lay.split {
            corner_note(f, room, lay, k, &[(text, DOUBT)]);
        } else {
            // (beside the circle where there is room, once the picture is still: looking
            // for room as it moves, it would leap from side to side)
            let top = b + DVec2::new(0.0, eps);
            name(f, room, !look.moving, |f, room| room.label_any(f, ("eps", n), &[v3(map.to(top)), v3(map.to(b - DVec2::new(0.0, eps)))], &text, 12.0, DOUBT, Vec2::new(0.0, -1.0), 6.0, true));
        }
    } else {
        name(f, room, named_error, |f, room| room.label(f, ("δb", n), v3(map.to(b + db)), r"$\delta b$", 13.0, ERROR, out(f, db), 8.0, true));
    }
}

/// The pane of the answers: the lines and their bands, the ellipse (or strip) of
/// answers, its half-axes, x and δx, the lines of the reading, and ridge.
fn draw_x(f: &mut Frame, room: &mut Room, lay: &Layout, k: usize, map: &Map, p: &Picture, look: Look) {
    backdrop(f, room, lay, map, k, look.moving || (look.holding && lay.panes[k].close));
    let live = look.live;
    let seen = map.seen();
    let (a, b, db, eps) = (p.sys.a, p.b, p.db, p.eps);
    let details = lay.details(k);
    let n = k as u32;
    // the bands, then the answers, then the lines on top
    for i in 0..2 {
        let r = row(a, i);
        let len = r.length();
        if len > 0.0 && eps > 0.0 {
            // |r·x − bᵢ| ≤ ε is a band ε/|r| either side of the line
            fill(f, map, &band_in(r / len, b[i] / len, eps / len, seen), LINE.a(0.11));
        }
    }
    if let Some(e) = p.ellipse(160) {
        if eps > 0.0 {
            fill(f, map, &clip_to(&e, seen), DOUBT.a(0.16));
            outline(f, room, map, &e, 2.0, DOUBT);
        }
    } else if let Some((v, lo, hi)) = p.strip() {
        fill(f, map, &band_in(v, (lo + hi) * 0.5, (hi - lo) * 0.5, seen), DOUBT.a(0.16));
        for c in [lo, hi] {
            endless(f, room, map, v, c, 2.0, DOUBT);
        }
    }
    // ridge's ellipse, and how far its middle is from x
    if let (Some(re), Some(rc)) = (p.ridge_ellipse(160), p.ridge_centre()) {
        if eps > 0.0 {
            fill(f, map, &clip_to(&re, seen), RIDGE.a(0.12));
            outline(f, room, map, &re, 1.8, RIDGE);
        }
        if let Some(x) = p.x {
            if let Some((s, e)) = segment_in(x, rc, seen) {
                f.dashed(map.to(s), map.to(e), 2.0, RIDGE);
                room.line(f, v3(map.to(s)), v3(map.to(e)));
            }
        }
        dot(f, map, rc, 7.0, RIDGE);
    }
    // the lines of the reading b + δb (dashed, under the equations' own lines, which
    // they lie on where that number of b is not off at all), then the equations' lines
    if eps > 0.0 {
        // (the dashes kept in step with where the reading's answer is, or would be)
        let anchor = p.x.zip(p.dx).map_or(DVec2::ZERO, |(x, dx)| x + dx);
        for i in 0..2 {
            endless_dashed(f, map, row(a, i), b[i] + db[i], anchor, 1.3, ERROR.a(0.85));
        }
    }
    let mut drawn = Vec::new();
    for i in 0..2 {
        if endless(f, room, map, row(a, i), b[i], 2.0, LINE).is_some() {
            drawn.push(i);
        }
    }
    // the half-axes of the ellipse
    let strength = distinct(&p.sys);
    if let (Some(x), true) = (p.x, details && eps > 0.0) {
        for (i, c) in [(0, MOST), (1, LEAST)] {
            if strength > 0.0 {
                seg(f, room, map, x, x + p.sys.v[i] * (eps / p.sys.s[i]), 2.6, c.a(strength));
            } else {
                f.line(NOWHERE, NOWHERE, 2.6, c.a(0.0));
            }
        }
    }
    // x, and the error in it
    if let (Some(x), Some(dx)) = (p.x, p.dx) {
        arrow(f, room, map, DVec2::ZERO, x, 2.6, TRUTH);
        if eps > 0.0 {
            arrow(f, room, map, x, x + dx, 2.2, ERROR);
            dot(f, map, x + dx, 7.0, ERROR);
            if details {
                handle_stalk(f, map, x, dx, live);
            }
        }
        if !live && !lay.panes[k].close {
            dot(f, map, x, 12.0, TRUTH);
        }
    }
    // ridge's answer for this reading, and how far it is from x
    if let (Some(at), Some(x)) = (p.ridge_answer(), p.x) {
        arrow(f, room, map, x, at, 2.2, RIDGE);
        dot(f, map, at, 8.0, RIDGE);
    }

    // the names, the most important first
    let out = |f: &Frame, d: DVec2| screen_dir(f, map, DVec2::ZERO, d);
    if let (Some(x), Some(dx)) = (p.x, p.dx) {
        let along = if x.length() > 0.0 { out(f, x) } else { Vec2::new(1.0, -1.0) };
        name(f, room, in_view(map, x), |f, room| room.label(f, ("x", n), v3(map.to(x)), "$x$", 16.0, TRUTH, along.perp() - along * 0.6, 9.0, false));
        let named_error = eps > 0.0 && details && in_view(map, x + dx) && db.length() >= 0.5 * eps && !look.moving;
        name(f, room, named_error, |f, room| room.label(f, ("δx", n), v3(grip(map, x, dx)), r"$\delta x$", 15.0, ERROR, out(f, dx), 9.0, false));
    }
    // each line's number, beside it towards one end of what the pane shows: the point
    // of the line nearest a point kept the same way along it from frame to frame, so
    // that the number moves with the line and never jumps to its other end
    let (slo, shi) = seen;
    let (middle, r) = ((slo + shi) * 0.5, ((shi - slo) * 0.5).min_element());
    for i in drawn {
        let normal = row(a, i).normalize_or_zero();
        let goal = middle + look.along[i] * (0.72 * r);
        let at = goal - normal * (normal.dot(goal) - b[i] / row(a, i).length());
        name(f, room, in_view(map, at) && !look.moving, |f, room| room.label(f, ("line", 2 * n + i as u32), v3(map.to(at)), &(i + 1).to_string(), 14.0, LINE.mix(pal::WHITE, 0.35), out(f, normal), 7.0, true));
    }
    if let (Some(at), Some(rc)) = (p.ridge_answer(), p.ridge_centre()) {
        let dir = p.x.map_or(Vec2::new(1.0, 0.0), |x| out(f, rc - x));
        name(f, room, in_view(map, at) && !look.moving, |f, room| room.label(f, ("ridge", n), v3(map.to(at)), "ridge", 13.0, RIDGE, dir, 8.0, false));
    }
    if let (Some(x), true) = (p.x, details && eps > 0.0) {
        let mut lines = vec![(format!(r"$\varepsilon/\sigma_1 = {}$", sig(eps / p.sys.s[0], 2)), MOST), (format!(r"$\varepsilon/\sigma_2 = {}$", sig(eps / p.sys.s[1], 2)), LEAST)];
        if let Some((now, bias)) = p.ridge_errors() {
            lines.push((format!(r"ridge ($\lambda = {}$): ${}$ off", sig(p.lambda, 2), pct_tex(now)), RIDGE));
            lines.push((format!("${}$ off with no error at all", pct_tex(bias)), RIDGE));
        }
        if lay.split {
            corner_note(f, room, lay, k, &lines);
        } else {
            // beside the middle of each half-axis, on either side of it (once the picture
            // is still: looking for room as it moves, they would leap from side to side)
            for (i, (text, c)) in lines.iter().take(2).enumerate() {
                let half = p.sys.v[i] * (eps / p.sys.s[i]);
                let across = out(f, half).perp();
                let mid = x + half * 0.5;
                name(f, room, !look.moving, |f, room| room.label_any(f, ("half", 2 * n + i as u32), &[v3(map.to(mid)), v3(map.to(x + half))], text, 12.5, *c, across, 10.0, true));
            }
        }
    }
    // what is the matter, when there is no single answer
    let says = match p.sys.rank() {
        0 => Some("$A = 0$: no lines; every $x$ gives $0$"),
        1 => Some(if p.strip().is_some() { "parallel lines: a strip of answers without end" } else { r"parallel lines: no reading within $\varepsilon$ has an answer" }),
        _ => None,
    };
    if let Some(text) = says {
        let (lo, hi) = lay.panes[k].rect;
        f.caption_px(Vec2::new((lo.x + hi.x) * 0.5, lo.y + 10.0), text, 13.0, DOUBT, Align2::CENTER_TOP);
    }
}

/// The arrows from the panes of b to those of x: A⁻¹ takes each reading to its answer.
fn between_panes(f: &mut Frame, lay: &Layout, p: &Picture) {
    let invertible = p.sys.inv.is_some();
    let c = if invertible { pal::TEXT } else { pal::DIM };
    let text = if invertible { "$A^{-1}$" } else { "no $A^{-1}$" };
    let pairs: &[(usize, usize)] = if lay.split { &[(0, 1), (2, 3)] } else { &[(0, 1)] };
    for &(i, j) in pairs {
        let ((a0, a1), (b0, _)) = (lay.panes[i].rect, lay.panes[j].rect);
        if lay.narrow {
            // down the right-hand side of the gap (x's title is on the left)
            let x = a1.x - 30.0;
            let (from, to) = (Vec2::new(x, a1.y + 5.0), Vec2::new(x, b0.y - 5.0));
            if to.y - from.y > 16.0 {
                hud_arrow(f, from, to, 1.8, c);
                f.text_px(Vec2::new(x - 9.0, (from.y + to.y) * 0.5), text, 14.0, c, Align2::RIGHT_CENTER);
            }
        } else {
            let y = (a0.y + a1.y) * 0.5;
            let (from, to) = (Vec2::new(a1.x + 5.0, y), Vec2::new(b0.x - 5.0, y));
            if to.x - from.x > 16.0 {
                hud_arrow(f, from, to, 1.8, c);
                f.text_px(Vec2::new((from.x + to.x) * 0.5, y - 8.0), text, 14.0, c, Align2::CENTER_BOTTOM);
            }
        }
    }
}

/// What each pane shows, above it.
fn titles(f: &mut Frame, lay: &Layout, invertible: bool) {
    for k in 0..lay.count() {
        let pane = lay.panes[k];
        let (lo, hi) = pane.rect;
        let room = hi.x - lo.x - (if lay.narrow { 70.0 } else { 4.0 });
        let choices: &[&str] = match (pane.space, pane.close, lay.split) {
            (1, false, _) if !invertible => &["the answers: $A$ has no inverse", "no inverse", "$x$"],
            (0, false, true) => &["$b$, the right-hand side, to scale from $0$", "$b$, to scale from $0$", "$b$"],
            (1, false, true) => &["$x = A^{-1}b$, the answer, to scale from $0$", "$x$, to scale from $0$", "$x$"],
            (0, true, _) => &[r"close up: every reading within $\varepsilon$ of $b$", "close up: the readings", "close up"],
            (1, true, _) => &["close up: the answer to each reading", "close up: the answers", "close up"],
            (0, ..) => &[r"the readings: $b$, off by $\delta b$, with $|\delta b| = \varepsilon$", r"the readings: $b + \delta b$", "$b$"],
            _ => &["the answers: $x = A^{-1}b$, and $A^{-1}$ of each reading", "the answers: $x = A^{-1}b$", "$x$"],
        };
        if let Some(t) = choices.iter().find(|t| width(f, t, 13.0) <= room) {
            f.text_px(Vec2::new(lo.x + 2.0, lo.y - 7.0), *t, 13.0, pal::TEXT, Align2::LEFT_BOTTOM);
        }
    }
}

/// The relative errors, their ratio and κ, under the panes.
fn numbers(f: &mut Frame, (lo, hi): (Vec2, Vec2), p: &Picture) {
    hud_panel(f, lo, hi);
    let sys = p.sys;
    let rb = match p.rel_b() {
        Some(r) => format!(r"$|\delta b| / |b| = {} / {} = {}$", sig(p.db.length(), 3), sig(p.b.length(), 3), pct_tex(r)),
        None => r"$b = 0$: none (it would be $\varepsilon/0$)".into(),
    };
    let rx = match (p.x, p.dx, p.rel_x()) {
        (Some(x), Some(dx), Some(r)) => format!(r"$|\delta x| / |x| = {} / {} = {}$", sig(dx.length(), 3), sig(x.length(), 3), pct_tex(r)),
        (Some(_), _, None) => "$x = 0$: none".into(),
        _ => "no single $x$".into(),
    };
    let k = sys.kappa();
    let bound = if sys.inv.is_some() { format!(r"$\kappa = \sigma_1/\sigma_2 = {} / {} = {}$", sig(sys.s[0], 3), sig(sys.s[1], 3), sig(k, 3)) } else { r"$\kappa = \infty$: $A$ squashes the plane flat".into() };
    let times = match p.ratio() {
        Some(r) => format!(r"$x$'s relative error is ${}$ times $b$'s: between $1/\kappa$ and $\kappa$", sig(r, 3)),
        None if p.eps == 0.0 => r"$\varepsilon = 0$: no error to amplify".into(),
        None => "no ratio to take".into(),
    };
    let (x, y) = (lo.x + 12.0, lo.y + 10.0);
    if hi.x - lo.x > 760.0 {
        let (size, line) = (13.0, 20.0);
        let col = x + 92.0;
        f.text_px(Vec2::new(x, y), "error in $b$", size, pal::DIM, Align2::LEFT_TOP);
        f.text_px(Vec2::new(col, y), rb, size, ERROR, Align2::LEFT_TOP);
        f.text_px(Vec2::new(x, y + line), "error in $x$", size, pal::DIM, Align2::LEFT_TOP);
        f.text_px(Vec2::new(col, y + line), rx, size, ERROR, Align2::LEFT_TOP);
        f.text_px(Vec2::new(x, y + 2.0 * line), times, size, pal::TEXT, Align2::LEFT_TOP);
        let right = hi.x - 16.0;
        f.text_px(Vec2::new(right, y), bound, size, pal::TEXT, Align2::RIGHT_TOP);
        gauge(f, Vec2::new(right - 280.0, y + line + 6.0), 280.0, p);
        // the equations themselves, between the two, where there is room for them
        let middle = x + 360.0;
        if right - 300.0 - middle > 190.0 {
            f.text_px(Vec2::new(middle, y), "the equations", size, pal::DIM, Align2::LEFT_TOP);
            for i in 0..2 {
                f.text_px(Vec2::new(middle, y + line * (i + 1) as f32), format!("{}:  ${}$", i + 1, equation(row(sys.a, i), p.b[i])), size, LINE.mix(pal::WHITE, 0.35), Align2::LEFT_TOP);
            }
        }
    } else {
        for (i, (text, c)) in [(format!("$b$ off by {rb}"), ERROR), (format!("$x$ off by {rx}"), ERROR), (times, pal::TEXT), (bound, pal::TEXT)].into_iter().enumerate() {
            f.text_px(Vec2::new(x, y + i as f32 * 19.0), text, 12.0, c, Align2::LEFT_TOP);
        }
    }
}

/// From 1/κ to κ on a scale of factors, with where this b and this error land.
fn gauge(f: &mut Frame, at: Vec2, width: f32, p: &Picture) {
    let k = p.sys.kappa();
    if !k.is_finite() {
        return;
    }
    let span = k.ln();
    let x_of = |r: f64| at.x + width * if span > 1e-9 { (0.5 + 0.5 * (r.ln() / span).clamp(-1.0, 1.0)) as f32 } else { 0.5 };
    let y = at.y + 6.0;
    hud_line(f, Vec2::new(at.x, y), Vec2::new(at.x + width, y), 2.0, pal::DIM);
    let marks: Vec<(f64, String)> = if span > 1e-9 {
        vec![(1.0 / k, format!(r"$1/\kappa = {}$", sig(1.0 / k, 2))), (1.0, "$1$".into()), (k, format!(r"$\kappa = {}$", sig(k, 3)))]
    } else {
        vec![(1.0, r"$1 = \kappa$".into())]
    };
    for (i, (r, name)) in marks.iter().enumerate() {
        let x = x_of(*r);
        hud_line(f, Vec2::new(x, y - 5.0), Vec2::new(x, y + 5.0), 1.4, pal::DIM);
        // (the ends' names inward, so that they stay under the gauge)
        let align = if marks.len() == 1 || i == 1 { Align2::CENTER_TOP } else if i == 0 { Align2::LEFT_TOP } else { Align2::RIGHT_TOP };
        f.text_px(Vec2::new(x, y + 8.0), name.as_str(), 11.0, pal::DIM, align);
    }
    if let Some(r) = p.ratio() {
        let x = x_of(r);
        hud_rect_at(f, 2, Vec2::new(x - 3.0, y - 7.0), Vec2::new(x + 3.0, y + 7.0), ERROR);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::TAU;

    /// The scene after an experiment's button, played to the end.
    fn scene(k: usize) -> Condition {
        let mut s = Condition::create();
        s.set_up(k);
        assert!(s.animated(), "a new setting is played");
        for _ in 0..200 {
            s.advance(1.0 / 60.0);
        }
        assert!(s.change.is_none() && s.shown() == s.goal());
        s
    }

    fn trial(k: usize) -> Picture {
        Picture::of(&scene(k).shown())
    }

    fn opening() -> Picture {
        Picture::of(&Condition::create().shown())
    }

    /// The same setting with the error turned to `turn` (radians).
    fn turned(p: &Picture, turn: f64) -> Picture {
        Picture::of(&Shown::new(p.sys.a, p.b, turn, p.eps, p.lambda))
    }

    fn angle(d: DVec2) -> f64 {
        d.y.atan2(d.x)
    }

    /// The angle between the two lines, in degrees.
    fn crossing(p: &Picture) -> f64 {
        let (r1, r2) = (row(p.sys.a, 0), row(p.sys.a, 1));
        (r1.dot(r2).abs() / (r1.length() * r2.length())).min(1.0).acos().to_degrees()
    }

    /// Are two directions the same line (either way along it)?
    fn along(a: DVec2, b: DVec2) -> bool {
        a.normalize().perp_dot(b.normalize()).abs() < 1e-9
    }

    /// The greatest of `f` over errors in 3600 directions.
    fn worst(p: &Picture, f: impl Fn(&Picture) -> f64) -> f64 {
        (0..3600).map(|i| f(&turned(p, i as f64 / 3600.0 * TAU))).fold(f64::MIN, f64::max)
    }

    fn random(rng: &mut Rng) -> DMat2 {
        DMat2::from_cols(DVec2::new(rng.normal() as f64, rng.normal() as f64), DVec2::new(rng.normal() as f64, rng.normal() as f64))
    }

    fn caption(k: usize) -> &'static str {
        TRIALS[k].1
    }

    /// The opening: two lines 6° apart crossing at (1, 1), κ = 19.6; the reading is off
    /// by ε = 0.1 in its second number, 4.7 % of b, and x by 69 %, 14.6 times as much.
    #[test]
    fn the_opening_picture() {
        let p = opening();
        assert_eq!(format!("{:.0}°", crossing(&p)), "6°");
        assert!((p.x.unwrap() - DVec2::ONE).length() < 1e-6 && (p.db - DVec2::new(0.0, 0.1)).length() < 1e-8);
        assert_eq!([sig(p.sys.kappa(), 3), pct(p.rel_b().unwrap()), pct(p.rel_x().unwrap()), sig(p.ratio().unwrap(), 3)], ["19.6", "4.7 %", "69 %", "14.6"]);
        // long and thin: the ellipse is κ times as long as it is wide
        let e = p.ellipse(3600).unwrap();
        let x = p.x.unwrap();
        let (long, short) = e.iter().fold((0.0f64, f64::MAX), |(l, s), q| (l.max(q.distance(x)), s.min(q.distance(x))));
        assert!((long / short - p.sys.kappa()).abs() < 1e-3 * p.sys.kappa(), "{long} / {short}");
        let scene = Condition::create();
        assert!(scene.verdict(&p).contains(r"$\kappa = 19.6$: whatever $b$ and the error, $x$'s relative error is between $0.0510$ and $19.6$ times $b$'s. Here it is $14.6$ times."));
    }

    /// The captions of the "things to try", held to account.
    #[test]
    fn the_experiments_show_what_their_captions_say() {
        let says = |k: usize, text: &str| assert!(caption(k).contains(text), "caption {k} does not say \"{text}\"");
        let open = opening();

        // well conditioned: lines 70° apart (the opening's 6°), x at (1, 1), the same
        // error; a small ellipse 1.47 times as long as wide; 6.9 % becomes 7.3 %
        let p = trial(0);
        says(0, &format!("cross at {:.0}°, not at {:.0}°", crossing(&p), crossing(&open)));
        assert!((p.x.unwrap() - DVec2::ONE).length() < 1e-6 && (p.db - open.db).length() < 1e-12);
        let x = p.x.unwrap();
        let e = p.ellipse(3600).unwrap();
        let (long, short) = e.iter().fold((0.0f64, f64::MAX), |(l, s), q| (l.max(q.distance(x)), s.min(q.distance(x))));
        assert!(long < 0.1 * x.length(), "small: {long}");
        says(0, &format!("only {} times as long as it is wide", sig(long / short, 3)));
        says(0, &format!("$b$ is off by {} and $x$ by {}, {} times as much", pct(p.rel_b().unwrap()), pct(p.rel_x().unwrap()), sig(p.ratio().unwrap(), 3)));
        says(0, &format!(r"$\kappa = {}$: no $b$ and no error can make $x$'s relative error more than {} times $b$'s", sig(p.sys.kappa(), 3), sig(p.sys.kappa(), 3)));

        // the worst case: b along u₁, the error along u₂, and the ratio is exactly κ;
        // x as short as it can be for a b this long, δx as long as it can be
        let p = trial(1);
        assert!(along(p.b, p.sys.u[0]) && along(p.db, p.sys.u[1]));
        let k = p.sys.kappa();
        assert!((p.ratio().unwrap() - k).abs() < 1e-9 * k, "{} against {k}", p.ratio().unwrap());
        says(1, &format!(r"$b$ is off by {} and $x$ by {}: exactly $\kappa = {}$ times as much", pct(p.rel_b().unwrap()), pct(p.rel_x().unwrap()), sig(k, 3)));
        let inv = p.sys.inv.unwrap();
        let (x, dx) = (p.x.unwrap(), p.dx.unwrap());
        for i in 0..360 {
            let d = DVec2::from_angle(i as f64 / 360.0 * TAU);
            assert!((inv * (d * p.b.length())).length() >= x.length() * (1.0 - 1e-12));
            assert!((inv * (d * p.eps)).length() <= dx.length() * (1.0 + 1e-12));
        }
        // and nothing beats it: the greatest ratio over every error is this one
        assert!(worst(&p, |q| q.ratio().unwrap()) <= k * (1.0 + 1e-9));

        // a lucky b and a lucky error: b along u₂ (as long), the error along u₁ (as
        // large); x is 27 long, the error moves it 0.065 across the narrow way, and
        // the ratio is exactly 1/κ
        let q = trial(2);
        assert!(along(q.b, q.sys.u[1]) && along(q.db, q.sys.u[0]));
        assert!((q.b.length() - p.b.length()).abs() < 1e-12 && q.eps == p.eps);
        assert!(along(q.dx.unwrap(), q.sys.v[0]), "across the ellipse's narrow way");
        says(2, &format!("$x$ is now {:.0} long", q.x.unwrap().length()));
        says(2, &format!(r"narrow way, by {}: $x$ is off by {}, $\kappa = {}$ times less than $b$'s {}", sig(q.dx.unwrap().length(), 2), pct(q.rel_x().unwrap()), sig(k, 3), pct(q.rel_b().unwrap())));
        assert!((q.ratio().unwrap() - 1.0 / k).abs() < 1e-9 / k);

        // a small determinant: 0.0001, twelve hundred times smaller than 0.12, yet
        // κ = 1: x and every error a hundred times larger, the same 4.7 % whatever b
        // and whatever the error; lines at right angles, a circle of answers
        let p = trial(3);
        let det = p.sys.a.determinant();
        assert!((det - 1e-4).abs() < 1e-10 && (open.sys.a.determinant() / det).round() == 1200.0, "{det}");
        says(3, r"Its determinant is 0.0001, twelve hundred times smaller than the nearly parallel pair's 0.12, yet $\kappa = 1$");
        assert!((p.sys.kappa() - 1.0).abs() < 1e-9);
        assert!((p.x.unwrap() - p.b * 100.0).length() < 1e-4 * p.x.unwrap().length());
        for b in [p.b, DVec2::new(-0.3, 2.0), DVec2::new(5.0, 0.01)] {
            for i in 0..36 {
                let q = Picture::of(&Shown::new(p.sys.a, b, i as f64 * 10f64.to_radians(), p.eps, 0.0));
                assert!((q.ratio().unwrap() - 1.0).abs() < 1e-9);
            }
        }
        says(3, &format!("$x$ is off by exactly as much as $b$, {}", pct(p.rel_x().unwrap())));
        assert!(row(p.sys.a, 0).dot(row(p.sys.a, 1)) == 0.0);
        let x = p.x.unwrap();
        assert!(p.ellipse(360).unwrap().iter().all(|q| (q.distance(x) - p.eps * 100.0).abs() < 1e-3), "a circle");

        // ridge: λ = 0.02; an ellipse 4.3 times shorter, its middle 15 % off x; for
        // this reading 32 % off against 69 %; never more than 36 % off, where the plain
        // answer can be 90 % off; with the error along u₁, 16 % against 4.6 %
        let p = trial(4);
        says(4, &format!(r"$\lambda = {}$", sig(p.lambda, 1)));
        let r = p.ridge.unwrap();
        let ridge_long = (0..3600).map(|i| (r * DVec2::from_angle(i as f64 / 3600.0 * TAU)).length()).fold(0.0, f64::max) * p.eps;
        let plain_long = p.eps / p.sys.s[1];
        says(4, &format!("{} times shorter than the plain one", sig(plain_long / ridge_long, 2)));
        let (now, bias) = p.ridge_errors().unwrap();
        assert!(bias > 0.1, "its middle has moved off x");
        says(4, &format!("with no error at all, ridge is {} off", pct(bias)));
        says(4, &format!("ridge is {} off where the plain answer is {} off", pct(now), pct(p.rel_x().unwrap())));
        let ridge_worst = worst(&p, |q| q.ridge_errors().unwrap().0);
        let plain_worst = worst(&p, |q| q.rel_x().unwrap());
        says(4, &format!("never more than {} off, where the plain answer can be {} off", pct(ridge_worst), pct(plain_worst)));
        for sign in [1.0, -1.0] {
            let q = turned(&p, angle(p.sys.u[0] * sign));
            let (ridge, plain) = (q.ridge_errors().unwrap().0, q.rel_x().unwrap());
            assert!(ridge > plain);
            says(4, &format!("ridge does worse than the plain answer: {} against {}", pct(ridge), pct(plain)));
        }
    }

    /// However A, b and the error are chosen, x's relative error is between 1/κ and κ
    /// times b's, and both ends are met: b along u₁ with the error along u₂, b along u₂
    /// with the error along u₁.
    #[test]
    fn kappa_bounds_the_ratio_and_is_met() {
        let mut rng = Rng::new(5);
        for _ in 0..400 {
            let a = random(&mut rng);
            let sys = System::of(a);
            let Some(_) = sys.inv else { continue };
            let k = sys.kappa();
            for _ in 0..10 {
                let b = DVec2::new(rng.normal() as f64, rng.normal() as f64) * 3.0;
                let p = Picture::of(&Shown::new(a, b, rng.range(0.0, 7.0) as f64, rng.range(0.001, 1.0) as f64, 0.0));
                let r = p.ratio().unwrap();
                assert!(r <= k * (1.0 + 1e-9) && r >= (1.0 - 1e-9) / k, "{a}: ratio {r}, κ {k}");
            }
            let met = |b: DVec2, e: DVec2| Picture::of(&Shown::new(a, b, angle(e), 0.3, 0.0)).ratio().unwrap();
            assert!((met(sys.u[0] * 2.0, sys.u[1]) - k).abs() < 1e-7 * k, "{a}");
            assert!((met(sys.u[1] * 2.0, -sys.u[0]) * k - 1.0).abs() < 1e-7, "{a}");
        }
        // turns, mirror images and scalings of them: κ = 1
        for a in [DMat2::from_angle(0.7), DMat2::from_diagonal(DVec2::new(1.0, -1.0)) * DMat2::from_angle(-2.0) * 3.0, DMat2::from_diagonal(DVec2::splat(0.01))] {
            assert!((System::of(a).kappa() - 1.0).abs() < 1e-9, "{a}");
        }
    }

    /// The answers to every reading within ε are A⁻¹ of the circle: an ellipse whose
    /// half-axes are ε/σ₂ along v₂ and ε/σ₁ along v₁, which fits inside the bands where
    /// they cross and touches each band's edges; the reading's own lines cross at
    /// x + δx; ridge shrinks x's part along vₖ by σₖ²/(σₖ² + λ).
    #[test]
    fn the_ellipse_is_the_circle_sent_back() {
        let mut rng = Rng::new(9);
        let mut tried = 0;
        while tried < 200 {
            let a = random(&mut rng);
            let sys = System::of(a);
            let Some(inv) = sys.inv else { continue };
            // (beyond that, 3600 points round the circle are too few to find the
            // ellipse's narrowest point to a thousandth)
            if sys.kappa() > 30.0 {
                continue;
            }
            tried += 1;
            let b = DVec2::new(rng.normal() as f64, rng.normal() as f64);
            let p = Picture::of(&Shown::new(a, b, rng.range(0.0, 7.0) as f64, 0.2, 0.05));
            let x = p.x.unwrap();
            // A v = σ u, and the directions at right angles
            for k in 0..2 {
                assert!((a * sys.v[k] - sys.u[k] * sys.s[k]).length() < 1e-9 * sys.s[0], "{a}");
                assert!((inv * sys.u[k] * sys.s[k] - sys.v[k]).length() < 1e-7, "{a}");
            }
            assert!(sys.u[0].dot(sys.u[1]).abs() < 1e-12 && sys.v[0].dot(sys.v[1]).abs() < 1e-12);
            let e = p.ellipse(3600).unwrap();
            let size = |q: &DVec2| q.distance(x);
            let long = e.iter().map(size).fold(0.0, f64::max);
            let short = e.iter().map(size).fold(f64::MAX, f64::min);
            assert!((long - p.eps / sys.s[1]).abs() < 1e-6 * long && (short - p.eps / sys.s[0]).abs() < 1e-3 * short, "{a}: {long} {short}");
            for q in &e {
                assert!(((a * (*q - x)).length() - p.eps).abs() < 1e-9 * p.eps);
            }
            for i in 0..2 {
                let r = row(a, i);
                let reach = e.iter().map(|q| r.dot(*q - x).abs()).fold(0.0, f64::max);
                assert!(reach <= p.eps * (1.0 + 1e-9) && reach >= p.eps * (1.0 - 1e-5), "{a}: line {i} reach {reach}");
                // the lines of the reading b + δb pass through x + δx
                assert!((r.dot(x + p.dx.unwrap()) - (b[i] + p.db[i])).abs() < 1e-9 * (1.0 + x.length() * r.length()));
            }
            // ridge's shrink along each v
            let rb = p.ridge_centre().unwrap();
            for k in 0..2 {
                let shrink = sys.s[k].powi(2) / (sys.s[k].powi(2) + p.lambda);
                assert!((sys.v[k].dot(rb) - shrink * sys.v[k].dot(x)).abs() < 1e-7 * (1.0 + x.length()), "{a}");
            }
        }
    }

    /// The closed form agrees with the general SVD (and keeps its digits nearly flat).
    #[test]
    fn the_singular_values_agree_with_the_general_svd() {
        let mut rng = Rng::new(2);
        for _ in 0..300 {
            let a = random(&mut rng);
            let sys = System::of(a);
            let (_, s, _) = crate::math::linalg::svd(&[a.x_axis.x, a.y_axis.x, a.x_axis.y, a.y_axis.y], 2, 2);
            assert!((sys.s[0] - s[0]).abs() < 1e-9 * s[0] && (sys.s[1] - s[1]).abs() < 1e-7 * s[0], "{a}: {:?} against {s:?}", sys.s);
        }
        // nearly flat: σ₂ = |det| / σ₁ to full precision
        let a = DMat2::from_cols(DVec2::new(1.0, 0.99), DVec2::new(0.99, 0.98));
        let sys = System::of(a);
        assert!((sys.s[0] * sys.s[1] - 1e-4).abs() < 1e-15 && sys.kappa() > 3e4, "{:?}", sys.s);
    }

    /// Flat matrices say so: κ is ∞ (written so), there is no x and no ratio; the
    /// readings that have answers fill a strip (each x of it fits a reading within ε,
    /// none beyond it does), or there are none; A = 0 and a zero row too.
    #[test]
    fn a_flat_matrix_says_what_has_become_of_the_answers() {
        let flat = DMat2::from_cols(DVec2::ONE, DVec2::ONE);
        let scene = Condition::create();
        let p = Picture::of(&Shown::new(flat, DVec2::new(1.6, 1.4), 1.0, 0.1, 0.0));
        assert!(p.sys.rank() == 1 && p.sys.kappa() == f64::INFINITY && sig(p.sys.kappa(), 3) == "∞");
        assert!(p.x.is_none() && p.ratio().is_none() && p.strip().is_none());
        assert!(scene.verdict(&p).contains("none of them has an answer"));
        let p = Picture::of(&Shown::new(flat, DVec2::new(1.5, 1.55), 1.0, 0.1, 0.0));
        let (v, lo, hi) = p.strip().expect("a strip");
        assert!(hi > lo && scene.verdict(&p).contains("strip without end"));
        for i in 0..=20 {
            let t = lo + (hi - lo) * i as f64 / 20.0;
            for along in [-50.0, 0.0, 3.0] {
                let x = v * t + v.perp() * along;
                assert!((flat * x - p.b).length() <= p.eps * (1.0 + 1e-9), "{t}");
            }
        }
        for t in [lo - 1e-3, hi + 1e-3] {
            assert!((flat * (v * t) - p.b).length() > p.eps);
        }
        let zero = Picture::of(&Shown::new(DMat2::ZERO, DVec2::new(1.0, 1.0), 0.0, 0.1, 0.0));
        assert!(zero.sys.rank() == 0 && zero.strip().is_none() && scene.verdict(&zero).starts_with("$A = 0$"));
        let one_row = Picture::of(&Shown::new(DMat2::from_cols(DVec2::new(1.0, 0.0), DVec2::new(2.0, 0.0)), DVec2::new(1.0, 0.05), 0.0, 0.1, 0.0));
        assert!(one_row.sys.rank() == 1 && one_row.strip().is_some());
    }

    /// b = 0, ε = 0, flat, nearly flat, huge and tiny matrices, with and without ridge:
    /// everything that is drawn or read out is finite, or not there, and the words say
    /// what is missing.
    #[test]
    fn degenerate_settings_leave_everything_finite() {
        let scene = Condition::create();
        let matrices = [
            DMat2::ZERO,
            DMat2::from_cols(DVec2::ONE, DVec2::ONE),
            DMat2::from_cols(DVec2::new(1.0, 0.0), DVec2::ZERO),
            DMat2::from_cols(DVec2::new(1.0, 0.99), DVec2::new(0.99, 0.98)),
            DMat2::from_cols(DVec2::new(100.0, -100.0), DVec2::new(100.0, 100.0)),
            DMat2::from_cols(DVec2::new(0.01, 0.0), DVec2::new(0.0, -0.01)),
            DMat2::from_cols(DVec2::new(1e-7, 0.0), DVec2::new(0.0, 1e-7)),
        ];
        let panes = [false, true].map(|close| [0, 1].map(|space| Pane { space, close, rect: (Vec2::ZERO, Vec2::splat(300.0)) }));
        for a in matrices {
            for b in [DVec2::ZERO, DVec2::new(1.6, 1.4), DVec2::new(-1e3, 4e2)] {
                for eps in [0.0, 0.1, 1.0] {
                    for lambda in [0.0, 0.02, 10.0] {
                        let p = Picture::of(&Shown::new(a, b, 2.0, eps, lambda));
                        let finite = |v: Option<DVec2>| v.is_none_or(|v| v.is_finite());
                        assert!(finite(p.x) && finite(p.dx) && finite(p.ridge_centre()) && finite(p.ridge_answer()), "{a} {b} {eps} {lambda}");
                        assert!(p.ellipse(16).is_none_or(|e| e.iter().all(|q| q.is_finite())));
                        assert!(p.ridge_ellipse(16).is_none_or(|e| e.iter().all(|q| q.is_finite())));
                        assert!(p.strip().is_none_or(|(v, lo, hi)| v.is_finite() && lo.is_finite() && hi.is_finite()));
                        for pane in panes.iter().flatten() {
                            let (lo, hi) = p.content(*pane);
                            let f = Framing::fit(DVec2::splat(2.0), lo, hi);
                            assert!(lo.is_finite() && hi.is_finite() && f.centre.is_finite() && f.scale.is_finite() && f.scale > 0.0, "{a} {b} {eps} {lambda}: {lo} {hi}");
                        }
                        let says = scene.verdict(&p);
                        assert!(!says.contains("NaN") && !says.contains("inf") && !says.contains('?'), "{says}");
                        if b == DVec2::ZERO {
                            assert!(p.rel_b().is_none() && p.ratio().is_none());
                            if p.sys.rank() == 2 {
                                assert!(says.contains("b = 0"));
                            }
                        }
                        if eps == 0.0 {
                            assert!(p.ratio().is_none());
                            if p.sys.rank() == 2 && b != DVec2::ZERO {
                                assert!(says.contains(r"$\varepsilon = 0$"));
                            }
                        }
                    }
                }
            }
        }
    }

    /// On the way from one matrix to another (both turning the plane the same way
    /// round), the plane is never squashed flat, and the way ends exactly where it
    /// should.
    #[test]
    fn the_way_between_settings_never_squashes_the_plane() {
        let mut ends: Vec<DMat2> = TRIALS.iter().map(|t| dmat(matrix_of(t.2.rows))).collect();
        ends.extend(PRESETS[..5].iter().map(|(_, r)| dmat(matrix_of(*r))));
        ends.push(DMat2::from_angle(3.0));
        let mut rng = Rng::new(4);
        ends.extend((0..30).map(|_| random(&mut rng)).filter(|m| m.determinant() > 0.05));
        for &from in &ends {
            for &to in &ends {
                let least = from.determinant().min(to.determinant());
                for i in 0..=50 {
                    let m = between_matrices(from, to, i as f64 / 50.0);
                    assert!(m.determinant() >= least * 0.999 || m.determinant() > 0.0 && m.determinant() > least * 0.2, "{from} to {to} at {i}: {}", m.determinant());
                    assert!(m.is_finite());
                }
                assert_eq!(between_matrices(from, to, 1.0), to);
                assert!((between_matrices(from, to, 0.0) - from).abs().to_cols_array().iter().all(|v| *v < 1e-12));
                // (and from the start, the polar parts give the matrix back)
                let m = between_matrices(from, to, 1e-9);
                assert!((m - from).abs().to_cols_array().iter().all(|v| *v < 1e-6), "{from} to {to}: {m}");
            }
        }
        // a flat end: the plane goes flat only at the very end
        let flat = dmat(matrix_of(PRESETS[5].1));
        for i in 0..50 {
            assert!(between_matrices(ends[0], flat, i as f64 / 50.0).determinant() > 0.0);
        }
    }

    /// A change of setting plays, then the scene rests; with the scene's time standing
    /// still it is answered at once.
    #[test]
    fn a_change_plays_and_then_rests() {
        let mut s = Condition::create();
        assert!(!s.animated());
        s.set_up(1);
        let start = s.shown();
        s.advance(0.3);
        let mid = s.shown();
        assert!(s.animated() && mid != start && mid != s.goal());
        s.advance(0.0);
        assert!(!s.animated() && s.shown() == s.goal());
    }

    /// In every window, each pane's framing shows what it must, all of it, and keeps
    /// showing it (so that it is not fitted afresh every frame).
    #[test]
    fn every_pane_shows_what_it_must() {
        for (view, inset) in [(Vec2::new(1078.0, 1000.0), 0.0), (Vec2::new(390.0, 800.0), 48.0), (Vec2::new(800.0, 390.0), 40.0), (Vec2::new(1600.0, 1000.0), 0.0), (Vec2::new(240.0, 180.0), 30.0), (Vec2::new(620.0, 560.0), 0.0)] {
            let lay = Layout::of(view, inset);
            assert_eq!(lay.split, view == Vec2::new(1078.0, 1000.0) || view == Vec2::new(1600.0, 1000.0), "{view}");
            for p in (0..TRIALS.len()).map(trial).chain([opening()]) {
                for k in 0..lay.count() {
                    let (_, half) = lay.room(k);
                    let (lo, hi) = p.content(lay.panes[k]);
                    let f = Framing::fit(half, lo, hi);
                    assert!(f.shows(half, lo, hi), "{view}: pane {k}");
                    let map = Map::of(&lay, k, f);
                    let (a, b) = lay.bounds(k);
                    for q in [lo, hi] {
                        let s = map.to(q);
                        assert!(s.cmpge(a - Vec2::splat(1e-4)).all() && s.cmple(b + Vec2::splat(1e-4)).all(), "{view}: pane {k}");
                    }
                }
            }
            // the panes lie on the canvas, below the buttons, and apart
            for k in 0..lay.count() {
                let (lo, hi) = lay.panes[k].rect;
                assert!(lo.cmpge(Vec2::ZERO).all() && hi.cmple(view).all() && lo.y >= inset && hi.cmpgt(lo).all(), "{view}: {lo} {hi}");
            }
        }
    }

    #[test]
    fn numbers_are_written_plainly() {
        assert_eq!(sig(0.009_999_999_776, 3), "0.0100");
        assert_eq!(sig(19.615_688, 3), "19.6");
        assert_eq!(sig(-0.5, 2), "−0.50");
        assert_eq!(sig(f64::INFINITY, 3), "∞");
        assert_eq!(sig(0.0, 3), "0");
        assert_eq!(sig(2.5e7, 3), "2.5e7");
        assert_eq!([pct(0.047_036), pct(0.687_18), pct(0.002_397_9)], ["4.7 %", "69 %", "0.24 %"]);
    }

    /// The quick choices have the condition numbers their buttons say.
    #[test]
    fn the_presets_have_the_condition_numbers_they_say() {
        let want = [1.0, 3.0, 9.0, 19.0, 99.0, f64::INFINITY];
        for ((name, r), k) in PRESETS.iter().zip(want) {
            let got = System::of(dmat(matrix_of(*r))).kappa();
            assert!(got == k || (got - k).abs() < 1e-4 * k, "{name}: κ = {got}");
            let says = if k.is_finite() { format!("{}", k as i64) } else { r"\infty".to_string() };
            assert!(name.trim_end_matches('$').ends_with(&says), "{name}");
        }
    }

    /// A change of framing zooms about the one place both framings draw the same point
    /// of the plane at: that point stays put all the way, and the way ends where it should.
    #[test]
    fn a_change_of_framing_zooms_about_the_point_it_keeps() {
        let mid = DVec2::new(0.4, -1.0);
        let from = Framing { centre: DVec2::new(1.6, 1.4), scale: 2e3 };
        for to in [Framing { centre: DVec2::new(1.6, 1.4), scale: 3.0 }, Framing { centre: DVec2::new(40.0, -7.0), scale: 0.05 }, Framing { centre: DVec2::new(0.3, 0.2), scale: 2e3 }] {
            let place = |f: &Framing, p: DVec2| mid + (p - f.centre) * f.scale;
            // the point drawn at the same place by both (none when the scales are equal)
            let kept = (to.scale != from.scale).then(|| (place(&to, DVec2::ZERO) - place(&from, DVec2::ZERO)) / (from.scale - to.scale));
            for i in 0..=20 {
                let now = from.towards(&to, mid, i as f64 / 20.0);
                assert!(now.scale > 0.0 && now.centre.is_finite());
                if let Some(p) = kept {
                    assert!((place(&now, p) - place(&from, p)).length() < 1e-6 * (1.0 + place(&from, p).length()), "{to:?} at {i}");
                }
            }
            assert_eq!(from.towards(&to, mid, 1.0), to);
        }
    }
}

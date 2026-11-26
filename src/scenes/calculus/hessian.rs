//! Curvature: the Hessian and Newton's method.

use super::*;
use crate::math::expr::Func;
use glam::DVec2;

pub const DEF: SceneDef = SceneDef {
    id: "hessian",
    title: "Curvature: the Hessian and Newton's method",
    group: "Calculus",
    blurb: "The second derivatives of a surface as a matrix: bowls, domes and saddles, and the step that jumps to the bottom of the local bowl.",
    make: |_| Box::new(Hessian::new()),
};

// ----- second derivatives, exactly ------------------------------------------------------

/// A number with its first and second derivatives in the two variables $x$ and $y$
/// (second-order forward-mode differentiation): its value, its gradient, and the three
/// different entries of its Hessian, in double precision. Running a formula's bytecode
/// on these gives the Hessian exactly (up to rounding), with no differences taken.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
struct Hyper {
    v: f64,
    g: [f64; 2],
    /// ∂²/∂x², ∂²/∂x∂y, ∂²/∂y².
    h: [f64; 3],
}

/// `k d`, but nothing where `d` is nothing (as `Dual` does it): a slope that is infinite
/// where the argument does not vary at all (the square root at 0, held fixed) adds
/// nothing, rather than infinity times zero.
fn along(k: f64, d: f64) -> f64 {
    if d == 0.0 {
        0.0
    } else {
        k * d
    }
}

impl Hyper {
    /// The variable `i` (0 for $x$, 1 for $y$) at the value `v`.
    fn var(v: f64, i: usize) -> Hyper {
        let mut g = [0.0; 2];
        g[i.min(1)] = 1.0;
        Hyper { v, g, h: [0.0; 3] }
    }

    fn is_constant(&self) -> bool {
        self.g == [0.0; 2] && self.h == [0.0; 3]
    }

    /// φ(self), given φ, φ′ and φ″ at self's value: the chain rule to second order,
    /// ∂ᵢ∂ⱼ φ(u) = φ′(u) ∂ᵢ∂ⱼu + φ″(u) ∂ᵢu ∂ⱼu.
    fn chain(self, f0: f64, f1: f64, f2: f64) -> Hyper {
        let [gx, gy] = self.g;
        let [hxx, hxy, hyy] = self.h;
        Hyper {
            v: f0,
            g: [along(f1, gx), along(f1, gy)],
            h: [along(f1, hxx) + along(f2, gx * gx), along(f1, hxy) + along(f2, gx * gy), along(f1, hyy) + along(f2, gy * gy)],
        }
    }

    /// φ(self, o), given φ, its two first partials (in its first argument and its
    /// second) and its three second partials (aa, ab, bb) there.
    fn chain2(self, o: Hyper, f0: f64, [fa, fb]: [f64; 2], [faa, fab, fbb]: [f64; 3]) -> Hyper {
        let (a, b) = (self, o);
        let g = [along(fa, a.g[0]) + along(fb, b.g[0]), along(fa, a.g[1]) + along(fb, b.g[1])];
        let second = |i: usize, j: usize, k: usize| {
            along(fa, a.h[k]) + along(fb, b.h[k]) + along(faa, a.g[i] * a.g[j]) + along(fab, a.g[i] * b.g[j] + a.g[j] * b.g[i]) + along(fbb, b.g[i] * b.g[j])
        };
        Hyper { v: f0, g, h: [second(0, 0, 0), second(0, 1, 1), second(1, 1, 2)] }
    }

    fn recip(self) -> Hyper {
        let r = 1.0 / self.v;
        self.chain(r, -r * r, 2.0 * r * r * r)
    }
}

/// A function of one number, its first and its second derivative, at `x`.
fn derivatives(f: Func, x: f64) -> (f64, f64, f64) {
    use std::f64::consts::{LN_10, LN_2};
    match f {
        Func::Sin => (x.sin(), x.cos(), -x.sin()),
        Func::Cos => (x.cos(), -x.sin(), -x.cos()),
        Func::Tan => {
            let t = x.tan();
            (t, 1.0 + t * t, 2.0 * t * (1.0 + t * t))
        }
        Func::Asin => {
            let r = 1.0 - x * x;
            (x.asin(), 1.0 / r.sqrt(), x / (r * r.sqrt()))
        }
        Func::Acos => {
            let r = 1.0 - x * x;
            (x.acos(), -1.0 / r.sqrt(), -x / (r * r.sqrt()))
        }
        Func::Atan => {
            let r = 1.0 + x * x;
            (x.atan(), 1.0 / r, -2.0 * x / (r * r))
        }
        Func::Sinh => (x.sinh(), x.cosh(), x.sinh()),
        Func::Cosh => (x.cosh(), x.sinh(), x.cosh()),
        Func::Tanh => {
            let t = x.tanh();
            (t, 1.0 - t * t, -2.0 * t * (1.0 - t * t))
        }
        Func::Exp => {
            let e = x.exp();
            (e, e, e)
        }
        Func::Ln => (x.ln(), 1.0 / x, -1.0 / (x * x)),
        Func::Log10 => (x.log10(), 1.0 / (x * LN_10), -1.0 / (x * x * LN_10)),
        Func::Log2 => (x.log2(), 1.0 / (x * LN_2), -1.0 / (x * x * LN_2)),
        Func::Sqrt => {
            let s = x.sqrt();
            (s, 0.5 / s, -0.25 / (s * x))
        }
        Func::Abs => (x.abs(), if x >= 0.0 { 1.0 } else { -1.0 }, 0.0),
        Func::Sign => (
            if x > 0.0 {
                1.0
            } else if x < 0.0 {
                -1.0
            } else {
                0.0
            },
            0.0,
            0.0,
        ),
        Func::Floor => (x.floor(), 0.0, 0.0),
        Func::Ceil => (x.ceil(), 0.0, 0.0),
        Func::Relu => (x.max(0.0), if x > 0.0 { 1.0 } else { 0.0 }, 0.0),
        Func::Sigmoid => {
            let s = 1.0 / (1.0 + (-x).exp());
            let d = s * (1.0 - s);
            (s, d, d * (1.0 - 2.0 * s))
        }
        Func::Gauss => {
            let e = (-x * x).exp();
            (e, -2.0 * x * e, (4.0 * x * x - 2.0) * e)
        }
    }
}

impl Num for Hyper {
    fn cst(v: f32) -> Self {
        Hyper { v: v as f64, ..Default::default() }
    }
    fn value(self) -> f32 {
        self.v as f32
    }
    fn add(self, o: Self) -> Self {
        Hyper { v: self.v + o.v, g: [self.g[0] + o.g[0], self.g[1] + o.g[1]], h: [self.h[0] + o.h[0], self.h[1] + o.h[1], self.h[2] + o.h[2]] }
    }
    fn sub(self, o: Self) -> Self {
        self.add(o.neg())
    }
    fn mul(self, o: Self) -> Self {
        let (a, b) = (self, o);
        Hyper {
            v: a.v * b.v,
            g: [a.g[0] * b.v + a.v * b.g[0], a.g[1] * b.v + a.v * b.g[1]],
            h: [
                a.h[0] * b.v + a.v * b.h[0] + 2.0 * a.g[0] * b.g[0],
                a.h[1] * b.v + a.v * b.h[1] + a.g[0] * b.g[1] + a.g[1] * b.g[0],
                a.h[2] * b.v + a.v * b.h[2] + 2.0 * a.g[1] * b.g[1],
            ],
        }
    }
    fn div(self, o: Self) -> Self {
        if o.is_constant() {
            // (dividing by a number: no rounding from a reciprocal's derivatives)
            let r = 1.0 / o.v;
            return Hyper { v: self.v * r, g: self.g.map(|d| d * r), h: self.h.map(|d| d * r) };
        }
        self.mul(o.recip())
    }
    fn neg(self) -> Self {
        Hyper { v: -self.v, g: self.g.map(|d| -d), h: self.h.map(|d| -d) }
    }
    fn call(self, f: Func) -> Self {
        let (f0, f1, f2) = derivatives(f, self.v);
        self.chain(f0, f1, f2)
    }
    /// a^b. A constant exponent by the power rule, which also covers a negative base
    /// with a whole exponent, and zero; a constant base as an exponential; otherwise
    /// the general rule for a positive base.
    fn pow(self, o: Self) -> Self {
        let (a, n) = (self.v, o.v);
        let v = a.powf(n);
        if o.is_constant() {
            let f1 = if n == 0.0 { 0.0 } else { n * a.powf(n - 1.0) };
            let f2 = if n == 0.0 || n == 1.0 { 0.0 } else { n * (n - 1.0) * a.powf(n - 2.0) };
            return self.chain(v, f1, f2);
        }
        let ln = a.ln();
        if self.is_constant() {
            return o.chain(v, v * ln, v * ln * ln);
        }
        let below = a.powf(n - 1.0);
        self.chain2(o, v, [n * below, v * ln], [n * (n - 1.0) * a.powf(n - 2.0), below * (1.0 + n * ln), v * ln * ln])
    }
    /// atan2(a, b), whose partials are b/r², −a/r² and, a level down, −2ab/r⁴, (a² −
    /// b²)/r⁴, 2ab/r⁴ (r² = a² + b²): finite everywhere but the origin, where the angle
    /// has no slope to speak of (zero is given, as `Dual` gives it).
    fn atan2(self, o: Self) -> Self {
        let (a, b) = (self.v, o.v);
        let r2 = a * a + b * b;
        let v = a.atan2(b);
        if r2 <= 0.0 {
            return Hyper { v, ..Default::default() };
        }
        let r4 = r2 * r2;
        self.chain2(o, v, [b / r2, -a / r2], [-2.0 * a * b / r4, (a * a - b * b) / r4, 2.0 * a * b / r4])
    }
}

// ----- the function near a point -----------------------------------------------------------

/// A symmetric 2×2 matrix [[a, b], [b, c]]: a Hessian.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
struct Sym {
    a: f64,
    b: f64,
    c: f64,
}

/// A symmetric matrix's eigenvalues, the larger first, and the unit eigenvector of the
/// larger (the other's is at right angles to it). With equal eigenvalues every
/// direction is an eigenvector, and `v1` is the $x$ axis.
#[derive(Clone, Copy, Debug, PartialEq)]
struct Eig {
    l1: f64,
    l2: f64,
    v1: DVec2,
}

impl Eig {
    fn v2(&self) -> DVec2 {
        self.v1.perp()
    }

    /// The larger of the two in size.
    fn big(&self) -> f64 {
        self.l1.abs().max(self.l2.abs())
    }

    /// Is the eigenvalue `l` zero, as far as these numbers can tell? (A part in 10⁹ of
    /// the other one, or nothing at all.)
    fn is_zero(&self, l: f64) -> bool {
        l.abs() <= FLAT * self.big() || self.big() < 1e-300
    }
}

/// An eigenvalue smaller than this share of the other is taken for 0: what is left of a
/// zero after the rounding of the arithmetic that found it is some ten thousand times
/// smaller.
const FLAT: f64 = 1e-12;

impl Sym {
    fn of(h: [f64; 3]) -> Sym {
        Sym { a: h[0], b: h[1], c: h[2] }
    }

    fn det(&self) -> f64 {
        self.a * self.c - self.b * self.b
    }

    fn apply(&self, v: DVec2) -> DVec2 {
        DVec2::new(self.a * v.x + self.b * v.y, self.b * v.x + self.c * v.y)
    }

    fn is_finite(&self) -> bool {
        self.a.is_finite() && self.b.is_finite() && self.c.is_finite()
    }

    /// m ± d, with m the mean of the diagonal and d = √(((a − c)/2)² + b²), and the
    /// larger's direction at half the angle of (a − c, 2b). The one of the two that is
    /// smaller in size is found as det H divided by the other: as a difference m − d of
    /// nearly equal numbers it would lose its digits.
    fn eigen(&self) -> Eig {
        let m = 0.5 * (self.a + self.c);
        let d = (0.5 * (self.a - self.c)).hypot(self.b);
        let theta = 0.5 * (2.0 * self.b).atan2(self.a - self.c);
        let (l1, l2) = if m >= 0.0 {
            let l1 = m + d;
            (l1, if l1 != 0.0 { self.det() / l1 } else { 0.0 })
        } else {
            let l2 = m - d;
            (self.det() / l2, l2)
        };
        // (and never out of order by a rounding)
        Eig { l1: l1.max(l2), l2: l2.min(l1), v1: DVec2::new(theta.cos(), theta.sin()) }
    }
}

/// What a function is near a point: its value, gradient and Hessian there.
#[derive(Clone, Copy, Debug, PartialEq)]
struct Local {
    f: f64,
    g: DVec2,
    h: Sym,
}

impl Local {
    fn at(expr: &Expr, p: DVec2) -> Local {
        let r = expr.eval_num::<Hyper>(&[Hyper::var(p.x, 0), Hyper::var(p.y, 1)]);
        Local { f: r.v, g: DVec2::new(r.g[0], r.g[1]), h: Sym::of(r.h) }
    }

    fn is_finite(&self) -> bool {
        self.f.is_finite() && self.g.is_finite() && self.h.is_finite()
    }

    /// The fitted paraboloid's height a step `d` from the point:
    /// f + ∇f·d + ½ dᵀHd.
    fn model(&self, d: DVec2) -> f64 {
        self.f + self.g.dot(d) + 0.5 * d.dot(self.h.apply(d))
    }

    /// Newton's step: to the point where the fitted paraboloid is flat, −H⁻¹∇f. Where an
    /// eigenvalue is 0 there is none, and it is said so rather than divided by.
    fn newton(&self, flat_slope: f64) -> Step {
        if !self.is_finite() {
            return Step::Undefined;
        }
        if self.g.length() <= flat_slope {
            return Step::Here;
        }
        let e = self.h.eigen();
        if e.is_zero(e.l2) || e.is_zero(e.l1) {
            return Step::Singular;
        }
        let Sym { a, b, c } = self.h;
        let d = -DVec2::new(c * self.g.x - b * self.g.y, a * self.g.y - b * self.g.x) / self.h.det();
        if d.is_finite() {
            Step::Go(d)
        } else {
            Step::Undefined
        }
    }
}

/// Newton's step from a point.
#[derive(Clone, Copy, Debug, PartialEq)]
enum Step {
    /// The gradient is 0 already: the point is where the paraboloid is flat.
    Here,
    /// H has an eigenvalue 0: the paraboloid is a trough with no single flat point.
    Singular,
    /// The function has no value or no derivatives here.
    Undefined,
    Go(DVec2),
}

// ----- kinds of point --------------------------------------------------------------------

/// What the second derivatives say: the shape of the fitted paraboloid, and what a
/// stationary point is.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Kind {
    /// Both eigenvalues positive.
    Bowl,
    /// Both negative.
    Dome,
    /// One of each.
    Saddle,
    /// One of them 0 (the other not).
    Trough,
    /// Both 0.
    Flat,
}

impl Kind {
    fn of(e: &Eig) -> Kind {
        match (e.is_zero(e.l1), e.is_zero(e.l2)) {
            (true, true) => Kind::Flat,
            (true, false) | (false, true) => Kind::Trough,
            _ if e.l2 > 0.0 => Kind::Bowl,
            _ if e.l1 < 0.0 => Kind::Dome,
            _ => Kind::Saddle,
        }
    }
}

// ----- runs of steps -----------------------------------------------------------------------

/// Where a run within this distance of the point it ends at counts as arrived.
const ARRIVE: f64 = 1e-6;
/// The most steps a run of Newton's method, and of gradient descent, is followed for.
const NEWTON_MOST: usize = 100;
const DESCENT_MOST: usize = 4000;

/// How a run ended.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum End {
    /// At a stationary point: its steps had all but vanished.
    Arrived,
    /// A step took it off the map.
    Left,
    /// Newton's method met a singular H: there is no step to take.
    Singular,
    /// Still going after the most steps it is followed for.
    Going,
    /// The function had no value, or no derivatives, where it went.
    Undefined,
}

#[derive(Clone, Debug, PartialEq)]
struct Run {
    /// The start and every point a step went to (the last one perhaps off the map).
    pts: Vec<DVec2>,
    end: End,
    /// The stationary point it ends at, worked out to the precision of the arithmetic
    /// (an arrived run's end, polished by a few of Newton's steps).
    limit: Option<DVec2>,
    /// How far that may be from the true stationary point: nothing to speak of for a
    /// run that came to rest, the length of its last step for one that stopped where H
    /// is singular and the gradient has all but vanished (Newton's method closes in
    /// slowly on such a point, and its last step says how slowly).
    blur: f64,
}

impl Run {
    /// The number of steps after which the run stays within `ARRIVE` of its limit.
    fn steps_to_arrive(&self) -> Option<usize> {
        let limit = self.limit?;
        let mut k = None;
        for (i, p) in self.pts.iter().enumerate().rev() {
            if (*p - limit).length() < ARRIVE {
                k = Some(i);
            } else {
                break;
            }
        }
        k
    }

    /// The steps taken.
    fn steps(&self) -> usize {
        self.pts.len().saturating_sub(1)
    }

    /// How far each point of the run is from its limit (to within `blur`).
    fn errors(&self) -> Vec<f64> {
        self.limit.map(|q| self.pts.iter().map(|p| (*p - q).length()).collect()).unwrap_or_default()
    }
}

/// The map: the square of half-width `e` round `centre`.
#[derive(Clone, Copy, Debug, PartialEq)]
struct Map {
    centre: DVec2,
    e: f64,
}

impl Map {
    fn contains(&self, p: DVec2) -> bool {
        let d = (p - self.centre).abs();
        p.is_finite() && d.x <= self.e * 1.000_001 && d.y <= self.e * 1.000_001
    }

    fn lo(&self) -> DVec2 {
        self.centre - DVec2::splat(self.e)
    }

    fn hi(&self) -> DVec2 {
        self.centre + DVec2::splat(self.e)
    }

    /// The point, kept on the map.
    fn keep(&self, p: DVec2) -> DVec2 {
        if p.is_finite() {
            p.max(self.lo()).min(self.hi())
        } else {
            self.centre
        }
    }
}

/// A step so short (against the size of the point) that the run has come to rest:
/// near the rounding of the arithmetic. (A run that closes in by a fixed share each
/// step, as gradient descent does, is then still far nearer its limit than `ARRIVE`,
/// unless that share is a part in a million or less.)
fn resting(step: DVec2, p: DVec2) -> bool {
    step.length() <= 1e-13 * (1.0 + p.length())
}

/// Newton's method from `p`: one step after another to the flat point of the
/// paraboloid fitted where it stands, until the steps vanish, a step leaves the map, or
/// H is singular. A singular H where the gradient is no more than `flat_slope` is a
/// stationary point arrived at.
fn newton_run(expr: &Expr, p: DVec2, map: Map, flat_slope: f64) -> Run {
    let mut pts = vec![p];
    let mut end = End::Going;
    let mut blur = 0.0;
    for _ in 0..NEWTON_MOST {
        let here = *pts.last().unwrap_or(&p);
        let local = Local::at(expr, here);
        match local.newton(0.0) {
            Step::Here => {
                end = End::Arrived;
                break;
            }
            Step::Singular if local.g.length() <= flat_slope => {
                end = End::Arrived;
                blur = match pts.as_slice() {
                    [.., a, b] => (*b - *a).length(),
                    _ => 0.0,
                };
                break;
            }
            Step::Singular => {
                end = End::Singular;
                break;
            }
            Step::Undefined => {
                end = End::Undefined;
                break;
            }
            Step::Go(d) => {
                let next = here + d;
                pts.push(next);
                if !map.contains(next) {
                    end = End::Left;
                    break;
                }
                if resting(d, here) {
                    end = End::Arrived;
                    break;
                }
            }
        }
    }
    let limit = (end == End::Arrived).then(|| *pts.last().unwrap_or(&p));
    Run { pts, end, limit, blur }
}

/// Gradient descent from `p` with step size `h`: x ← x − h∇f(x), until the steps
/// vanish or it leaves the map.
fn descent_run(expr: &Expr, p: DVec2, h: f64, map: Map, flat_slope: f64) -> Run {
    let mut pts = vec![p];
    let mut end = End::Going;
    for _ in 0..DESCENT_MOST {
        let here = *pts.last().unwrap_or(&p);
        let l = Local::at(expr, here);
        if !l.is_finite() {
            end = End::Undefined;
            break;
        }
        let d = -h * l.g;
        if resting(d, here) {
            end = End::Arrived;
            break;
        }
        let next = here + d;
        pts.push(next);
        if !map.contains(next) {
            end = End::Left;
            break;
        }
    }
    // (where it rests, polished by Newton's method: gradient descent stops a little
    // short of its limit, by as much as its last steps were long)
    let last = *pts.last().unwrap_or(&p);
    let polished = (end == End::Arrived).then(|| newton_run(expr, last, map, flat_slope)).filter(|n| n.limit.is_some_and(|q| (q - last).length() < 1e-4));
    let limit = (end == End::Arrived).then(|| polished.as_ref().and_then(|n| n.limit).unwrap_or(last));
    let blur = polished.map_or(0.0, |n| n.blur);
    Run { pts, end, limit, blur }
}

/// What a stationary point is, from its Hessian.
fn kind_at(expr: &Expr, p: DVec2) -> Kind {
    Kind::of(&Local::at(expr, p).h.eigen())
}

// ----- the functions -----------------------------------------------------------------------

/// A function the scene knows, with the part of the plane to show it on, where to put
/// the point, and gradient descent's step size.
struct Preset {
    name: &'static str,
    formula: &'static str,
    centre: [f64; 2],
    e: f64,
    p: [f64; 2],
    h: f64,
    /// How strongly tall walls are pressed down in the picture (0: heights as they are).
    squash: f32,
}

const LANDSCAPE: usize = 0;
const QUADRATIC: usize = 1;
const VALLEY: usize = 2;
const WELLS: usize = 3;
const FLAT_BOTTOM: usize = 4;
/// The things to try that show the last four.
#[cfg(test)]
const QUADRATIC_TRY: usize = 1;
#[cfg(test)]
const VALLEY_TRY: usize = 2;
#[cfg(test)]
const WELLS_TRY: usize = 3;
#[cfg(test)]
const FLAT_TRY: usize = 4;

const PRESETS: [Preset; 6] = [
    Preset { name: "landscape", formula: "sin(x) cos(0.8 y) + 0.25 sin(x + y)", centre: [0.0, 0.0], e: 2.8, p: [-1.2, -0.85], h: 0.8, squash: 0.0 },
    Preset { name: "quadratic bowl", formula: "1.5 x^2 + 2 x y + y^2", centre: [0.0, 0.0], e: 2.5, p: [-1.4, 1.7], h: 0.4, squash: 0.0 },
    Preset { name: "curved valley", formula: "(1 - x)^2 + 3 (y - x^2)^2", centre: [0.0, 1.0], e: 2.0, p: [-1.2, 1.0], h: 0.06, squash: 12.0 },
    Preset { name: "two wells", formula: "(x^2 - 1)^2 + y^2", centre: [0.0, 0.0], e: 1.8, p: [0.35, 0.9], h: 0.15, squash: 0.0 },
    Preset { name: "flat bottom", formula: "x^2 + y^4", centre: [0.0, 0.0], e: 1.2, p: [0.0, 0.0], h: 0.2, squash: 0.0 },
    Preset { name: "monkey saddle", formula: "x^3 - 3 x y^2", centre: [0.0, 0.0], e: 1.5, p: [0.9, 0.5], h: 0.1, squash: 0.0 },
];

/// Where the tour of the landscape stops: near its minimum, one of its saddle points and
/// its maximum (each found exactly by Newton's method from there).
const TOUR_NEAR: [[f64; 2]; 3] = [[-1.57, 0.0], [-0.075, 1.89], [1.57, 0.0]];

/// One of the "things to try": a function as a preset sets it up, and whether the point
/// then tours the landscape's stationary points.
struct Trial {
    label: &'static str,
    look_for: &'static str,
    preset: usize,
    tour: bool,
}

const TRIALS: [Trial; 5] = [
    Trial {
        label: "a bowl, a dome and a saddle",
        look_for: r"The point visits a minimum, a saddle point and a maximum of the landscape, and stops at each. At all three $\nabla f = 0$, and the two eigenvalues of $H$ tell them apart: $1.38$ and $0.76$ at the minimum (both positive: a bowl), $0.80$ and $-1.28$ at the saddle (one of each), $-0.76$ and $-1.38$ at the maximum (both negative: a dome). The white paraboloid takes each shape in turn.",
        preset: LANDSCAPE,
        tour: true,
    },
    Trial {
        label: "Newton on a quadratic",
        look_for: r"$f = 1.5x^2 + 2xy + y^2$ is its own fitted paraboloid: the white sheet lies on the surface everywhere, and Newton's first step lands exactly on the minimum. Gradient descent needs 76 steps to come within $10^{-6}$ of it, with $h = 0.4$, the step size that makes the slower of its two directions shrink fastest. Even so each step only multiplies the distance along each eigenvector by $0.82$, because $\lambda_1 = 4.56$ is about ten times $\lambda_2 = 0.44$.",
        preset: QUADRATIC,
        tour: false,
    },
    Trial {
        label: "a narrow curved valley",
        look_for: r"At its minimum $(1, 1)$ this valley curves 83 times as much across as along ($\lambda_1 = 31.6$, $\lambda_2 = 0.38$). With a step size above $2/\lambda_1 = 0.063$ gradient descent never settles, and with $h = 0.06$ it needs 595 steps to come within $10^{-6}$ of the minimum; Newton's method needs 9. Newton's last distances from the minimum, $0.074$, $0.0042$, $1.9 \times 10^{-5}$, $2.9 \times 10^{-10}$, are each about the square of the one before.",
        preset: VALLEY,
        tour: false,
    },
    Trial {
        label: "Newton runs to a saddle",
        look_for: r"Two wells with a pass between them. Where the point is, the fitted paraboloid is a saddle, and Newton's step goes to its saddle point: after 3 steps the run is on the pass at $(0, 0)$, where $\nabla f = 0$ too. Gradient descent goes downhill, into the well at $(1, 0)$. Drag the point close to the line $x = 0.58$, where $f_{xx} = 0$: $H$ is singular there, and beside it Newton's step is longer than the map.",
        preset: WELLS,
        tour: false,
    },
    Trial {
        label: "a flat direction",
        look_for: r"At the origin $\nabla f = 0$ and the eigenvalues are $2$ and $0$: the test says nothing. This point is a minimum, because along $y$ the surface rises as $y^4$, too slowly for a second derivative to see; $x^2 - y^4$ has the same $H$ here and is a saddle. Drag the point away: Newton's run back is slow, its distance multiplied by only $2/3$ at each step after the first, because $H$ is singular where it is going.",
        preset: FLAT_BOTTOM,
        tour: false,
    },
];

// ----- colours, each with one meaning ------------------------------------------------------

/// The point p, and the paraboloid fitted there.
const POINT: Color = pal::WHITE;
const FIT: Color = pal::WHITE;
/// The gradient (as in the scene "surface").
const GRAD: Color = pal::YELLOW;
/// λ₁, the larger eigenvalue, and its eigenvector; λ₂ likewise (as in "quadform").
const V1: Color = pal::ORANGE;
const V2: Color = pal::PURPLE;
/// Newton's step and its run.
const NEWTON: Color = pal::GREEN;
/// Gradient descent's run (as in "descent" and "quadform").
const DESCENT: Color = pal::CYAN;
/// The heights.
const CMAP: Cmap = Cmap::Slate;

// ----- the picture's measurements -------------------------------------------------------------

/// The surface is this wide either way from its middle, in the scene's units.
const HALF: f32 = 3.0;
/// The tallest the surface is drawn.
const TALLEST: f32 = 2.6;
/// Samples of the surface (and the map) each way, and of the fitted paraboloid.
const SURF_N: u32 = 120;
const FIT_N: u32 = 33;
/// The view: from where (the camera's own field of view, as `tan` of half of it).
const YAW: f32 = -0.95;
const PITCH: f32 = 0.72;
const TAN_HALF_FOV: f32 = 0.3203;
/// The heights the point's handle on the surface is held within (where f has no value,
/// or a value beyond those drawn).
const BOX_Z: [f32; 2] = [-0.5 * TALLEST, 0.5 * TALLEST];
/// The colours' range over the heights' shares (the darker part of the scale: the
/// coloured lines have to stand out), and the contour lines' spacing.
const SHADES: [f32; 2] = [-0.3, 1.75];
const CONTOUR: f32 = 1.0 / 14.0;

/// The heights of a function as drawn.
#[derive(Clone, Copy, Debug, PartialEq)]
struct Heights {
    lo: f64,
    hi: f64,
    squash: f32,
    /// How tall, in the scene's units, the range from `lo` to `hi` is drawn.
    span: f32,
}

impl Heights {
    /// A height as a share of the range, pressed down if `squash` asks (and below the
    /// range carried on as it arrives there, for the paraboloid's sake).
    fn share(&self, v: f64) -> f32 {
        let u = ((v - self.lo) / (self.hi - self.lo)) as f32;
        let s = self.squash;
        if s <= 0.0 {
            u
        } else if u >= 0.0 {
            (1.0 + s * u).ln() / (1.0 + s).ln()
        } else {
            s * u / (1.0 + s).ln()
        }
    }

    /// The height of the scene a value is drawn at.
    fn z(&self, v: f64) -> f32 {
        (self.share(v) - 0.5) * self.span
    }
}

/// The heights to draw, from the values sampled (holes left out). Where a few samples
/// stand far beyond the rest (a pole), the range is that of the rest, and the surface is
/// cut open beyond it.
fn heights_of(zs: &[f32], e: f64, squash: f32) -> Heights {
    let mut finite: Vec<f32> = zs.iter().copied().filter(|z| z.is_finite()).collect();
    let (mut lo, mut hi) = finite.iter().fold((f32::INFINITY, f32::NEG_INFINITY), |(a, b), &z| (a.min(z), b.max(z)));
    if finite.len() >= 100 {
        let n = finite.len();
        let p_lo = *finite.select_nth_unstable_by(n / 50, f32::total_cmp).1;
        let p_hi = *finite.select_nth_unstable_by(n - 1 - n / 50, f32::total_cmp).1;
        let reach = 4.0 * (p_hi - p_lo).max(1e-9);
        (lo, hi) = (lo.max(p_lo - reach), hi.min(p_hi + reach));
    }
    let (lo, hi) = if lo.is_finite() && hi.is_finite() { (lo as f64, hi as f64) } else { (0.0, 1.0) };
    let hi = hi.max(lo + 1e-9 * lo.abs().max(1.0));
    let span = if squash > 0.0 { TALLEST } else { ((hi - lo) * HALF as f64 / e).min(TALLEST as f64) as f32 };
    Heights { lo, hi, squash, span: span.max(0.05) }
}

fn smoothstep(t: f32) -> f32 {
    let t = t.clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

/// The stretch of the parameter round 0, within `lo..hi`, over which `seen` holds
/// throughout: found in 64 steps each way, its ends then narrowed down by halving. None
/// if 0 itself is not seen.
fn stretch(lo: f64, hi: f64, seen: impl Fn(f64) -> bool) -> Option<(f64, f64)> {
    if !(seen(0.0) && lo <= 0.0 && hi >= 0.0) {
        return None;
    }
    let end = |far: f64| {
        let mut inside = 0.0;
        for i in 1..=64 {
            let t = far * i as f64 / 64.0;
            if !seen(t) {
                let (mut a, mut b) = (inside, t);
                for _ in 0..24 {
                    let m = 0.5 * (a + b);
                    if seen(m) {
                        a = m;
                    } else {
                        b = m;
                    }
                }
                return a;
            }
            inside = t;
        }
        inside
    };
    Some((end(lo), end(hi)))
}

/// The part of the segment from `a` to `b` inside the box `lo..hi`.
fn clip(a: DVec2, b: DVec2, lo: DVec2, hi: DVec2) -> Option<(DVec2, DVec2)> {
    let (mut t0, mut t1) = (0.0f64, 1.0f64);
    let d = b - a;
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
    (t0 <= t1 && d.is_finite() && a.is_finite()).then(|| (a + d * t0, a + d * t1))
}

// ----- words ---------------------------------------------------------------------------------------

/// A number for the screen, to go inside a formula: two decimals, or two figures as a
/// power of ten when it is smaller than that or very large; 0 for less than 10⁻¹², the
/// rounding left of a 0 (an entry of H that is 0 is worked out as 10⁻¹⁷, say).
fn short(v: f64) -> String {
    let a = v.abs();
    if !v.is_finite() {
        "?".into()
    } else if a < 1e-12 {
        "0".into()
    } else if (0.01..10_000.0).contains(&a) {
        num(v, 2)
    } else {
        format!("{v:.1e}")
    }
}

/// A distance in the list of a run's: two figures, as a power of ten below a hundredth.
fn distance(d: f64) -> String {
    if d >= 0.1 {
        format!("{d:.2}")
    } else if d >= 0.01 {
        format!("{d:.3}")
    } else {
        format!("{d:.1e}")
    }
}

/// What a point is (at a stationary point), or what the paraboloid fitted there is
/// (elsewhere): a short headline and a sentence.
fn verdict(kind: Kind, stationary: bool) -> (&'static str, &'static str) {
    match (stationary, kind) {
        (true, Kind::Bowl) => ("A minimum.", r"$\nabla f = 0$, and both eigenvalues are positive: the surface rises every way from here."),
        (true, Kind::Dome) => ("A maximum.", r"$\nabla f = 0$, and both eigenvalues are negative: the surface falls every way from here."),
        (true, Kind::Saddle) => ("A saddle point.", r"$\nabla f = 0$, and the eigenvalues have opposite signs: up along one eigenvector, down along the other."),
        (true, Kind::Trough) => ("The test says nothing.", r"$\nabla f = 0$, but an eigenvalue is $0$: along its eigenvector only higher derivatives can tell up from down."),
        (true, Kind::Flat) => ("The test says nothing.", r"$\nabla f = 0$ and $H = 0$: only higher derivatives can tell what this point is."),
        (false, Kind::Bowl) => ("The fitted paraboloid is a bowl.", r"Both eigenvalues are positive. $\nabla f \ne 0$, so this is not a stationary point; Newton's step goes to the bowl's lowest point."),
        (false, Kind::Dome) => ("The fitted paraboloid is a dome.", r"Both eigenvalues are negative. $\nabla f \ne 0$, so this is not a stationary point; Newton's step goes to the dome's top."),
        (false, Kind::Saddle) => ("The fitted paraboloid is a saddle.", r"The eigenvalues have opposite signs. $\nabla f \ne 0$, so this is not a stationary point; Newton's step goes to the saddle's flat point."),
        (false, Kind::Trough) => ("The fitted paraboloid is a trough.", r"An eigenvalue is $0$: $H$ is singular, the trough has no single flat point, and Newton's method has no step to take."),
        (false, Kind::Flat) => ("The fitted paraboloid is a plane.", r"$H = 0$: there is no curvature to go by, and Newton's method has no step to take."),
    }
}

/// The stationary point a run ends at, in a few words.
fn kind_words(kind: Kind) -> &'static str {
    match kind {
        Kind::Bowl => "a minimum",
        Kind::Dome => "a maximum",
        Kind::Saddle => "a saddle point",
        Kind::Trough | Kind::Flat => "a point the test cannot judge",
    }
}

/// How a run went, in a few words: "3 steps, to a minimum".
fn run_words(run: &Run, kind: Option<Kind>) -> String {
    let steps = |n: usize| if n == 1 { "1 step".to_string() } else { format!("{} steps", count(n)) };
    match run.end {
        End::Arrived => match (run.steps_to_arrive(), kind) {
            (Some(0), Some(_)) => "none needed: $p$ is a stationary point".to_string(),
            (Some(n), Some(k)) => format!("{}, to {}", steps(n), kind_words(k)),
            _ => format!("{}, then at rest", steps(run.steps())),
        },
        End::Left => format!("left the map at step {}", run.steps()),
        End::Singular => format!("stopped after {}: $H$ singular", steps(run.steps())),
        End::Going => format!("still going after {}", steps(run.steps())),
        End::Undefined => format!("no value after {}", steps(run.steps())),
    }
}

/// Newton's distances from where its run ends, the last few, as a formula ("…, 0.21,
/// 4.9 × 10⁻³, 6.2 × 10⁻⁸"); empty if the run does not arrive.
fn distances_text(run: &Run, most: usize) -> String {
    // (not the last few, which differ from the end only by the rounding of the arithmetic,
    // or by less than the end is known to)
    let rounding = (1e-14 * (1.0 + run.limit.map_or(0.0, |q| q.length()))).max(20.0 * run.blur);
    let shown: Vec<String> = run.errors().iter().filter(|e| **e > rounding).map(|e| distance(*e)).collect();
    if shown.is_empty() {
        return String::new();
    }
    let from = shown.len().saturating_sub(most);
    let dots = if from > 0 { r"\ldots,\ " } else { "" };
    format!("${dots}{}$", shown[from..].join(r",\ "))
}

/// The fitted paraboloid's level curves, as points along them in the plane (a fixed
/// number for each, evenly spread in its parameter): the one through the point, and the one a quarter as high above (or below) the flat point; for a saddle,
/// also the two lines through the flat point at its own height. None where there is no
/// flat point (H singular). `e` is the map's half-width, which the curves need not
/// reach far beyond.
fn model_levels(at: &At, e: f64) -> Vec<Vec<DVec2>> {
    if at.kind == Kind::Trough && at.local.is_finite() {
        return trough_level(at, e);
    }
    let Step::Go(d) = at.step else { return Vec::new() };
    let (l1, l2, v1, v2) = (at.eig.l1, at.eig.l2, at.eig.v1, at.eig.v2());
    let c = at.p + d;
    let point = |u1: f64, u2: f64| c + v1 * u1 + v2 * u2;
    // how high p stands above the flat point, in the paraboloid: ½ (λ₁ u₁² + λ₂ u₂²)
    let (a, b) = (-d.dot(v1), -d.dot(v2));
    let kappa = 0.5 * (l1 * a * a + l2 * b * b);
    let mut curves = Vec::new();
    for k in [kappa, 0.25 * kappa] {
        let (r1, r2) = (2.0 * k / l1, 2.0 * k / l2);
        if !(r1.is_finite() && r2.is_finite()) || k == 0.0 {
            continue;
        }
        if r1 > 0.0 && r2 > 0.0 {
            // a bowl's or a dome's: an ellipse
            let (a, b) = (r1.sqrt(), r2.sqrt());
            curves.push((0..=120).map(|i| (i as f64 / 120.0 * std::f64::consts::TAU).sin_cos()).map(|(sn, cs)| point(a * cs, b * sn)).collect());
        } else if r1 > 0.0 || r2 > 0.0 {
            // a saddle's: a hyperbola, in two branches
            let (a, b) = (r1.abs().sqrt(), r2.abs().sqrt());
            let reach = (6.0 * e / a.min(b)).max(1.0).acosh().min(8.0);
            for side in [-1.0, 1.0] {
                let branch = (0..=80).map(|i| -reach + 2.0 * reach * i as f64 / 80.0);
                let pts = if r1 > 0.0 { branch.map(|t| point(side * a * t.cosh(), b * t.sinh())).collect() } else { branch.map(|t| point(a * t.sinh(), side * b * t.cosh())).collect() };
                curves.push(pts);
            }
        }
    }
    if l1 > 0.0 && l2 < 0.0 {
        // the saddle's own level: two lines crossing at the flat point
        let slope = (l1 / -l2).sqrt();
        let far = 6.0 * e;
        for side in [-1.0, 1.0] {
            let dir = (v1 + v2 * (side * slope)).normalize();
            curves.push((0..=60).map(|i| c + dir * (far * (i as f64 / 30.0 - 1.0))).collect());
        }
    }
    curves
}

/// A trough's level curve through the point (H with one eigenvalue 0): with the
/// curved direction `a` and the flat one `b`, the paraboloid is f + gₐs + g_b t + ½λs²
/// a step s along `a` and t along `b`, so its level through p is t = −(gₐs + ½λs²)/g_b,
/// a parabola; or, where the gradient has no part along `b`, the two lines s = 0 and
/// s = −2gₐ/λ (one line, the trough's floor, at a stationary point).
fn trough_level(at: &At, e: f64) -> Vec<Vec<DVec2>> {
    let (l1, l2, v1, v2) = (at.eig.l1, at.eig.l2, at.eig.v1, at.eig.v2());
    let (lambda, a, b) = if at.eig.is_zero(l2) { (l1, v1, v2) } else { (l2, v2, v1) };
    let (ga, gb) = (at.local.g.dot(a), at.local.g.dot(b));
    let far = 6.0 * e;
    let line = |s: f64| (0..=60).map(|i| at.p + a * s + b * (far * (i as f64 / 30.0 - 1.0))).collect::<Vec<DVec2>>();
    if gb.abs() <= 1e-9 * at.local.g.length().max(1e-300) || at.local.g.length() == 0.0 {
        let other = -2.0 * ga / lambda;
        return if other.abs() > 1e-12 && other.is_finite() { vec![line(0.0), line(other)] } else { vec![line(0.0)] };
    }
    vec![(0..=120)
        .map(|i| {
            let s = -far + 2.0 * far * i as f64 / 120.0;
            at.p + a * s + b * (-(ga * s + 0.5 * lambda * s * s) / gb)
        })
        .collect()]
}

// ----- the layout --------------------------------------------------------------------------------

const CARD_W: f32 = 390.0;
/// The size of the card's text.
const CARD_TEXT: f32 = 13.0;
/// The map's panel: the room round the map, above it (its title) and the rows under it.
const PAD: f32 = 10.0;
const TITLE_H: f32 = 20.0;
const ROW_H: f32 = 19.0;
const BUTTON_H: f32 = 30.0;

/// How many lines a text takes at `size` in `width` pixels (an estimate a little on
/// the generous side, for room set aside before anything is drawn).
fn lines(text: &str, size: f32, width: f32) -> usize {
    (text_width(text, size) / (0.9 * width)).ceil().max(1.0) as usize
}

/// The card's height at `width`: room for the longest of its verdicts (none for their
/// sentences, if `compact`).
fn card_height(width: f32, compact: bool) -> f32 {
    let kinds = [Kind::Bowl, Kind::Dome, Kind::Saddle, Kind::Trough, Kind::Flat];
    let longest = kinds.iter().flat_map(|k| [verdict(*k, true), verdict(*k, false)]).map(|(_, detail)| lines(detail, 12.0, width - 24.0)).max().unwrap_or(2);
    let detail = if compact { 0.0 } else { 15.0 * longest as f32 + 4.0 };
    10.0 + 20.0 + 22.0 + matrix_size(2, 2, 6, CARD_TEXT).y + 8.0 + 19.0 + detail + 6.0 + 18.0 + 8.0
}

/// How the runs are told: in two rows (how many steps each took), or in full (with
/// Newton's distances from its end, and the button that moves the point there).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Rows {
    None,
    Short,
    Full,
}

impl Rows {
    fn height(self) -> f32 {
        match self {
            Rows::None => 0.0,
            Rows::Short => 6.0 + 2.0 * ROW_H,
            Rows::Full => 6.0 + 4.0 * ROW_H + BUTTON_H,
        }
    }
}

/// The height of a box of the runs' rows alone.
const RUNS_H: f32 = 2.0 * PAD + 4.0 * ROW_H + BUTTON_H;

/// The map's panel: its backing, the map's square in it, and the runs under the map
/// if they have no box of their own.
#[derive(Clone, Copy, Debug, PartialEq)]
struct MapPanel {
    min: Vec2,
    max: Vec2,
    /// The map's top-left corner, and its side.
    at: Vec2,
    side: f32,
    rows: Rows,
}

impl MapPanel {
    fn height(side: f32, rows: Rows) -> f32 {
        PAD + TITLE_H + side + rows.height() + PAD
    }

    fn new(min: Vec2, width: f32, side: f32, rows: Rows) -> MapPanel {
        let at = Vec2::new(min.x + 0.5 * (width - side), min.y + PAD + TITLE_H);
        MapPanel { min, max: min + Vec2::new(width, MapPanel::height(side, rows)), at, side, rows }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
struct Layout {
    /// A phone held upright: the panels go below the picture.
    narrow: bool,
    card: Option<(Vec2, Vec2)>,
    panel: Option<MapPanel>,
    /// The runs in a box of their own (under the card).
    runs: Option<(Vec2, Vec2)>,
    /// The part of the canvas the picture has to itself.
    picture: (Vec2, Vec2),
}

impl Layout {
    /// For a canvas of `view` logical pixels, `inset` of whose top lies under the
    /// application's buttons.
    fn of(view: Vec2, inset: f32) -> Layout {
        let whole = (Vec2::new(0.0, inset.min(view.y * 0.5)), view.max(Vec2::ONE));
        let none = Layout { narrow: false, card: None, panel: None, runs: None, picture: whole };
        if view.x < 620.0 && view.y >= 560.0 {
            // a phone held upright: the map and the card one above the other, below the picture
            let width = view.x - 16.0;
            let card_h = card_height(width, true);
            let side = (width - 2.0 * PAD).min(0.3 * view.y);
            let card_top = view.y - 8.0 - card_h;
            let card = Some((Vec2::new(8.0, card_top), Vec2::new(8.0 + width, view.y - 8.0)));
            let panel_top = card_top - 6.0 - MapPanel::height(side, Rows::Short);
            if panel_top - whole.0.y < 0.18 * view.y {
                // (too short for both: the card alone, the rest in the side panel)
                return Layout { narrow: true, card, picture: (whole.0, Vec2::new(view.x, card_top - 4.0)), ..none };
            }
            let panel = Some(MapPanel::new(Vec2::new(8.0, panel_top), width, side, Rows::Short));
            return Layout { narrow: true, card, panel, runs: None, picture: (whole.0, Vec2::new(view.x, panel_top - 4.0)) };
        }
        if view.x >= 2.0 * CARD_W + 120.0 && view.y >= 600.0 {
            // a band along the bottom, the picture above it: the card and the runs at the
            // left, the map at the right
            let card_h = card_height(CARD_W, false);
            let side = (0.34 * view.y).min(view.x - CARD_W - 2.0 * PAD - 60.0).clamp(220.0, 380.0);
            let band = MapPanel::height(side, Rows::None).max(card_h + 6.0 + RUNS_H);
            let low = view.y - 14.0;
            let width = side + 2.0 * PAD;
            let panel = MapPanel::new(Vec2::new(view.x - 14.0 - width, low - band), width, side, Rows::None);
            let card = (Vec2::new(14.0, low - band), Vec2::new(14.0 + CARD_W, low - band + card_h));
            let runs = (Vec2::new(14.0, card.1.y + 6.0), Vec2::new(14.0 + CARD_W, card.1.y + 6.0 + RUNS_H));
            return Layout { narrow: false, card: Some(card), panel: Some(panel), runs: Some(runs), picture: (whole.0, Vec2::new(view.x, low - band - 6.0)) };
        }
        if view.x >= 560.0 && view.y >= 420.0 {
            // the map at the right, the picture to its left (the card in the side panel)
            let top = 14.0f32.max(inset + 8.0);
            let side = (0.42 * view.x).min(view.y - top - 14.0 - MapPanel::height(0.0, Rows::Full)).min(360.0);
            if side >= 160.0 {
                let width = side + 2.0 * PAD;
                let panel = MapPanel::new(Vec2::new(view.x - 10.0 - width, top), width, side, Rows::Full);
                return Layout { panel: Some(panel), picture: (whole.0, Vec2::new(panel.min.x - 6.0, view.y)), ..none };
            }
        }
        // too small to spare the room: the numbers are in the side panel
        none
    }

    fn covers(&self, p: Vec2) -> bool {
        self.card.iter().chain(self.runs.iter()).chain(self.panel.map(|m| (m.min, m.max)).iter()).any(|(min, max)| p.cmpge(*min).all() && p.cmple(*max).all())
    }
}

/// The camera's right and up directions, seen from `yaw` at `PITCH`.
fn screen_axes(yaw: f32) -> (Vec3, Vec3) {
    let right = Vec3::new(-yaw.sin(), yaw.cos(), 0.0);
    let up = Vec3::new(-PITCH.sin() * yaw.cos(), -PITCH.sin() * yaw.sin(), PITCH.cos());
    (right, up)
}

/// A button drawn over the picture (logical pixels): true when it is clicked.
fn hud_button(f: &mut Frame, min: Vec2, max: Vec2, text: &str, pointer: Option<Vec2>) -> bool {
    let over = pointer.is_some_and(|p| p.cmpge(min).all() && p.cmple(max).all());
    hud_rect(f, min, max, if over { Color::hex(0x33405E) } else { Color::hex(0x222B40) });
    f.text_px((min + max) * 0.5, text, 12.0, pal::TEXT, Align2::CENTER_CENTER);
    over && f.ptr.clicked
}

/// Words broken into lines no wider than `width` pixels, from `at` down, 15 pixels
/// apart; formulas are kept whole. (The lines are kept from frame to frame for a text
/// drawn again at the same size and width: measuring is the costly part.)
fn wrapped_px(f: &mut Frame, at: Vec2, width: f32, text: &str, size: f32, color: Color) {
    type Kept = Vec<(String, u32, u32, Vec<String>)>;
    thread_local! {
        static KEPT: std::cell::RefCell<Kept> = const { std::cell::RefCell::new(Vec::new()) };
    }
    let key = (size.to_bits(), width.to_bits());
    let found = KEPT.with(|k| k.borrow().iter().find(|(t, s, w, _)| t == text && (*s, *w) == key).map(|(_, _, _, lines)| lines.clone()));
    let lines = found.unwrap_or_else(|| {
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
        words.push(&text[start..]);
        let mut lines: Vec<String> = Vec::new();
        for word in words {
            match lines.last_mut() {
                Some(line) if f.text_size(&format!("{line} {word}"), size).x <= width => {
                    line.push(' ');
                    line.push_str(word);
                }
                _ => lines.push(word.to_string()),
            }
        }
        KEPT.with(|k| {
            let mut k = k.borrow_mut();
            if k.len() > 40 {
                k.clear();
            }
            k.push((text.to_string(), key.0, key.1, lines.clone()));
        });
        lines
    });
    for (i, line) in lines.into_iter().enumerate() {
        f.text_px(at + Vec2::new(0.0, 15.0 * i as f32), line, size, color, Align2::LEFT_TOP);
    }
}

/// The map's square on the screen: where a point of the plane is drawn, and back.
#[derive(Clone, Copy, Debug)]
struct OnMap {
    at: Vec2,
    side: f32,
    map: Map,
}

impl OnMap {
    fn px(&self, p: DVec2) -> Vec2 {
        let u = (p - self.map.lo()) / (2.0 * self.map.e);
        Vec2::new(self.at.x + u.x as f32 * self.side, self.at.y + (1.0 - u.y as f32) * self.side)
    }

    fn point(&self, s: Vec2) -> DVec2 {
        let u = DVec2::new(((s.x - self.at.x) / self.side) as f64, (1.0 - (s.y - self.at.y) / self.side) as f64);
        self.map.lo() + u * (2.0 * self.map.e)
    }
}

/// The parts of the picture drawn with lines whose number of points changes as the
/// point moves (a curve cut where it leaves the map, an arrow that loses its head off
/// it). Each has a line batch of its own, different from the others by a hair of its
/// width, so that from one frame to the next the parts that keep their number of
/// points can still be compared point for point: that is how `eidos motion` tells
/// smooth motion from jumps.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Part {
    SliceOne = 1,
    SliceTwo,
    AxisOne,
    AxisTwo,
    Path,
    GhostGrid,
    GhostAxes,
    GhostPath,
    CrossOne,
    CrossTwo,
    Gradient,
    Step,
    Descent,
    Newton,
}

/// Lines in the scene (`Blend`, hidden behind nearer surfaces) or on the screen
/// (`Overlay`) for one part. The ghost parts are seen through the surface where it
/// covers them: drawn faintly, and pulled towards the camera far enough to show
/// through the layers of surface near the paraboloid (in depth, a part in five
/// hundred: about a fifth of the camera's distance), not through the panels on the
/// screen.
fn lines_of<'a>(f: &'a mut Frame, part: Part) -> &'a mut crate::gfx::draw::LineBatch {
    let ghost = matches!(part, Part::GhostGrid | Part::GhostAxes | Part::GhostPath);
    let screen = matches!(part, Part::CrossOne | Part::CrossTwo | Part::Gradient | Part::Step | Part::Descent | Part::Newton);
    let kind = if screen { LineKind::Overlay } else { LineKind::Blend };
    let depth_bias = if ghost { 0.002 } else if screen { 0.0 } else { 2e-4 };
    f.dl.lines_with(kind, LineStyle { width_scale: 1.0 + 1e-4 * part as u8 as f32, depth_bias, ..Default::default() })
}

/// The dots on the map that change in number as the point moves, each kind in a batch
/// of its own (as `Part` does for lines): every point of a run, and the paraboloid's
/// level curves.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Dots {
    Descent = 1,
    Newton,
    Levels,
}

fn dots_of<'a>(f: &'a mut Frame, dots: Dots) -> &'a mut crate::gfx::draw::PointBatch {
    f.dl.points_with(PointKind::Overlay, PointStyle { size_scale: 1.0 + 1e-4 * dots as u8 as f32, ..Default::default() })
}

/// Every part's batch, made in the same order in every frame, each with a point that is
/// not seen. A batch left empty in one frame is let go, and a later one would then take
/// its place in the list: the parts must stay where they were to be compared from one
/// frame to the next.
fn every_batch(f: &mut Frame) {
    use Part::*;
    for part in [SliceOne, SliceTwo, AxisOne, AxisTwo, Path, GhostGrid, GhostAxes, GhostPath, CrossOne, CrossTwo, Gradient, Step, Descent, Newton] {
        let at = f.hud(Vec2::ZERO);
        let batch = lines_of(f, part);
        batch.push(at, 0.0, Color::CLEAR);
        batch.brk();
    }
    for dots in [Dots::Descent, Dots::Newton, Dots::Levels] {
        let at = f.hud(Vec2::ZERO);
        dots_of(f, dots).push(at, 0.0, Color::CLEAR);
    }
}

/// A name to be written on the map: its tag, the places (logical pixels) it may be
/// written beside, the first tried first, its text, size and colour, and the side of
/// the first place to try first.
type Name = (&'static str, Vec<Vec2>, String, f32, Color, Vec2);

/// Write `text` beside one of the screen points `places` (logical pixels), on a side
/// clear of what `room` holds, trying first the side towards `towards` of the first. A
/// place it had in the last frame is kept for as long as it is still clear, even when
/// a place tried earlier has come clear again: the name changes places only when it
/// must, and then glides there (it is kept as an offset from the first point, whichever
/// it is beside). Unlike `Room::label_any`, which goes back to its first choice as soon
/// as it can, and so, beside a line's two ends, leapt from one end to the other.
#[allow(clippy::too_many_arguments)]
fn name_beside(f: &mut Frame, room: &mut Room, name: &'static str, places: &[Vec2], text: &str, size: f32, color: Color, towards: Vec2) {
    let Some(&from) = places.first() else { return };
    let half = f.text_size(text, size) * 0.5 + Vec2::new(4.0, 1.0);
    let want = towards.normalize_or_zero();
    let mut sides: Vec<Vec2> = (0..8).map(|k| Vec2::from_angle(k as f32 * TAU / 8.0)).collect();
    sides.sort_by(|p, q| q.dot(want).total_cmp(&p.dot(want)));
    sides.insert(0, want);
    let kept = f.places.chosen(name, 0).map(|offset| from + offset).filter(|c| room.room_for(from, *c, half) >= 3.0);
    let centre = kept.unwrap_or_else(|| {
        let mut best: Option<(f32, Vec2)> = None;
        'places: for at in places {
            for side in &sides {
                let c = *at + *side * 5.0 + Vec2::new(side.x * half.x, side.y * half.y);
                let r = room.room_for(*at, c, half);
                if best.is_none_or(|b| r > b.0) {
                    best = Some((r, c));
                }
                if r >= 5.0 {
                    break 'places;
                }
            }
        }
        best.map_or(from, |b| b.1)
    });
    room.boxed(centre - half, centre + half);
    f.text_placed(name, 0, from, centre - from, text, size, color);
}

/// An arrow on the screen (logical pixels), its shaft and head in the batch of `part`,
/// as `hud_arrow` draws one.
fn arrow_px(f: &mut Frame, part: Part, a: Vec2, b: Vec2, width: f32, color: Color) {
    let d = (b - a).normalize_or_zero();
    let n = Vec2::new(-d.y, d.x);
    let head = (4.0 + 2.5 * width).min((b - a).length() * 0.5);
    let pts = [(a, b), (b, b - d * head + n * head * 0.5), (b, b - d * head - n * head * 0.5)].map(|(p, q)| (f.hud(p), f.hud(q)));
    let batch = lines_of(f, part);
    for (p, q) in pts {
        batch.segment(p, q, width, color);
    }
}

// ----- the scene ---------------------------------------------------------------------------------------

/// The tour of the landscape's stationary points: from where the point was, to each
/// stop in turn, staying a while at each.
#[derive(Clone, Copy, Debug)]
struct Tour {
    from: DVec2,
    stops: [DVec2; 3],
    /// Seconds of the scene's time since it set off.
    t: f32,
}

/// Seconds on the way from one stop to the next, and at each.
const LEG: f32 = 1.8;
const STAY: f32 = 3.0;

impl Tour {
    /// Where the point is, and whether the tour is over.
    fn at(&self) -> (DVec2, bool) {
        let mut t = self.t;
        let mut from = self.from;
        for (i, stop) in self.stops.iter().enumerate() {
            let leg = if (from - *stop).length() < 1e-9 { 0.0 } else { LEG };
            if t < leg {
                return (from.lerp(*stop, smoothstep(t / leg) as f64), false);
            }
            t -= leg;
            if i == self.stops.len() - 1 {
                return (*stop, true);
            }
            if t < STAY {
                return (*stop, false);
            }
            t -= STAY;
            from = *stop;
        }
        (from, true)
    }
}

/// The landscape's minimum, saddle point and maximum, exactly: Newton's method from
/// near each.
fn tour_stops(expr: &Expr, map: Map) -> [DVec2; 3] {
    TOUR_NEAR.map(|[x, y]| {
        let near = DVec2::new(x, y);
        newton_run(expr, near, map, 0.0).limit.unwrap_or(near)
    })
}

/// Everything worked out at the point as drawn.
#[derive(Clone, Debug)]
struct At {
    p: DVec2,
    local: Local,
    eig: Eig,
    kind: Kind,
    stationary: bool,
    step: Step,
    newton: Run,
    descent: Run,
    /// What the stationary points the runs end at are.
    newton_kind: Option<Kind>,
    descent_kind: Option<Kind>,
}

impl At {
    fn work_out(expr: &Expr, p: DVec2, map: Map, h: f64, slope: f64) -> At {
        let local = Local::at(expr, p);
        let eig = local.h.eigen();
        // (a gradient this small, against the slopes of the map, is the rounding of a 0)
        let flat_slope = 1e-9 * slope;
        let newton = newton_run(expr, p, map, flat_slope);
        let descent = descent_run(expr, p, h, map, flat_slope);
        At {
            p,
            local,
            eig,
            kind: Kind::of(&eig),
            stationary: local.g.length() <= flat_slope,
            step: local.newton(flat_slope),
            newton_kind: newton.limit.map(|q| kind_at(expr, q)),
            descent_kind: descent.limit.map(|q| kind_at(expr, q)),
            newton,
            descent,
        }
    }
}

/// The surfaces kept on the GPU: made in the first frame (the scene is set up, and
/// tested, without them).
struct Gpu {
    surf: Surface,
    /// The map: the same heights flat, drawn on the screen.
    flat: Surface,
    /// The fitted paraboloid.
    fit: Surface,
}

impl Gpu {
    fn new(gfx: &Gfx) -> Gpu {
        let mut g = Gpu { surf: Surface::new(gfx, SURF_N, SURF_N), flat: Surface::new(gfx, SURF_N, SURF_N), fit: Surface::new(gfx, FIT_N, FIT_N) };
        for (s, lit) in [(&mut g.surf, 1.0), (&mut g.flat, 0.0)] {
            let st = &mut s.style;
            st.cmap = Some(CMAP);
            st.range = SHADES;
            st.contour = CONTOUR;
            st.contour_alpha = if lit > 0.5 { 0.35 } else { 0.5 };
            st.line_color = Color::hex(0xE4F1FF);
            st.lit = lit;
            st.specular = 0.06;
            st.rim = 0.35;
        }
        let st = &mut g.fit.style;
        st.cmap = None;
        st.color = FIT;
        st.opacity = 0.3;
        st.lit = 0.55;
        st.specular = 0.5;
        st.grid = [8.0, 8.0];
        st.grid_alpha = 0.45;
        st.line_color = FIT;
        g
    }
}

pub struct Hessian {
    f: ExprField,
    map: Map,
    squash: f32,
    /// Gradient descent's step size.
    h: f64,
    /// The point as asked for, and as drawn: the drawn one follows through `between`,
    /// two eases one after the other, after a jump (a button); a hand moves all three.
    goal: DVec2,
    between: DVec2,
    p: DVec2,
    tour: Option<Tour>,
    show_fit: bool,
    show_runs: bool,
    // the function, sampled
    dirty: bool,
    xs: Vec<f32>,
    ys: Vec<f32>,
    zs: Vec<f32>,
    batch: Batch,
    heights: Heights,
    /// The typical slope on the map (the spread of heights over its width).
    slope: f64,
    /// The surfaces, made in the first frame.
    gpu: Option<Gpu>,
    /// What is worked out at the drawn point (None: to be worked out again).
    at: Option<At>,
    /// Whether the paraboloid on the GPU is the one fitted there (it is filled again
    /// only when the point or the function changes: filling it is a new picture).
    fit_filled: bool,
    /// The eigenvector last drawn: the next one is drawn pointing its way (an
    /// eigenvector has no sign of its own, and the formula's turns half a turn
    /// whenever a − c changes sign with b < 0, which would draw the same lines the
    /// other way round, every point of them leaping to the far end).
    last_v1: DVec2,
    /// Points of the surface as drawn, every so many samples: what the view is fitted to.
    outline: Vec<Vec3>,
    // the canvas the view was fitted to
    canvas: Vec2,
    inset: f32,
    measured: bool,
    new_view: bool,
}

/// How quickly the drawn point follows a jump (seconds for each of its two eases to
/// cover most of the way).
const FOLLOW: f64 = 0.12;

impl Hessian {
    fn new() -> Self {
        let mut s = Hessian {
            f: ExprField::new(PRESETS[LANDSCAPE].formula, &["x", "y"]),
            map: Map { centre: DVec2::ZERO, e: 2.8 },
            squash: 0.0,
            h: 0.8,
            goal: DVec2::ZERO,
            between: DVec2::ZERO,
            p: DVec2::ZERO,
            tour: None,
            show_fit: true,
            show_runs: true,
            dirty: true,
            xs: Vec::new(),
            ys: Vec::new(),
            zs: Vec::new(),
            batch: Batch::default(),
            heights: Heights { lo: 0.0, hi: 1.0, squash: 0.0, span: 1.0 },
            slope: 1.0,
            gpu: None,
            at: None,
            fit_filled: false,
            last_v1: DVec2::X,
            outline: Vec::new(),
            canvas: Vec2::new(1080.0, 1000.0),
            inset: 0.0,
            measured: false,
            new_view: false,
        };
        s.load(LANDSCAPE);
        s.rebuild();
        s
    }

    /// Set a preset up: its function, its map, its step size and its point (gliding
    /// there if the function stays the same).
    fn load(&mut self, k: usize) {
        let pr = &PRESETS[k];
        let same = self.f.text == pr.formula && self.f.error.is_none();
        if !same {
            self.f.set(pr.formula);
        }
        let map = Map { centre: DVec2::from(pr.centre), e: pr.e };
        if map != self.map || pr.squash != self.squash || !same {
            (self.map, self.squash, self.dirty) = (map, pr.squash, true);
        }
        self.h = pr.h;
        self.tour = None;
        self.place(DVec2::from(pr.p), same && !self.dirty);
        self.at = None;
    }

    /// Set one of the things to try up.
    fn set_up(&mut self, k: usize) {
        let t = &TRIALS[k];
        self.load(t.preset);
        (self.show_fit, self.show_runs) = (true, true);
        if t.tour {
            // (the point goes where the tour takes it, from where it is)
            let stops = tour_stops(&self.f.expr, self.map);
            self.goal = self.p;
            self.between = self.p;
            self.tour = Some(Tour { from: self.p, stops, t: 0.0 });
        }
    }

    /// Put the point at `q`: gliding there if `glide`, at once otherwise.
    fn place(&mut self, q: DVec2, glide: bool) {
        self.goal = self.map.keep(q);
        if !glide {
            (self.between, self.p) = (self.goal, self.goal);
        }
    }

    /// Move the drawn point `dt` seconds nearer to the one asked for.
    fn follow(&mut self, dt: f64) {
        if self.p == self.goal && self.between == self.goal {
            return;
        }
        let k = 1.0 - (-dt.max(0.0) / FOLLOW).exp();
        self.between += (self.goal - self.between) * k;
        self.p += (self.between - self.p) * k;
        let apart = (self.goal - self.between).length().max((self.goal - self.p).length());
        // (exactly there in the end: the picture then stops changing)
        if apart < 1e-5 * self.map.e || !apart.is_finite() {
            (self.between, self.p) = (self.goal, self.goal);
        }
    }

    /// Sample the function over the map, and fill the surface and the map from it.
    fn rebuild(&mut self) {
        let n = SURF_N as usize;
        if self.xs.len() != n * n {
            self.xs.resize(n * n, 0.0);
            self.ys.resize(n * n, 0.0);
            self.zs.resize(n * n, 0.0);
        }
        let (c, e) = (self.map.centre, self.map.e);
        for j in 0..n {
            for i in 0..n {
                self.xs[j * n + i] = (c.x - e + 2.0 * e * i as f64 / (n - 1) as f64) as f32;
                self.ys[j * n + i] = (c.y - e + 2.0 * e * j as f64 / (n - 1) as f64) as f32;
            }
        }
        self.f.expr.eval_batch(&[&self.xs, &self.ys], &mut self.zs, &mut self.batch);
        let before = self.heights.span;
        self.heights = heights_of(&self.zs, e, self.squash);
        let hs = self.heights;
        // (the view is fitted to the surface: to a new one afresh)
        self.new_view |= (hs.span - before).abs() > 0.02 * before || self.measured;
        self.slope = ((hs.hi - hs.lo) / (2.0 * e)).max(1e-300);
        let k = HALF / e as f32;
        let (cx, cy) = (c.x as f32, c.y as f32);
        let (xs, ys, zs) = (&self.xs, &self.ys, &self.zs);
        let surface: Vec<[f32; 4]> = (0..n * n)
            .map(|i| {
                let w = hs.share(zs[i] as f64);
                // (cut open beyond the range: a pole)
                let z = if (-1e-3..=1.0 + 1e-3).contains(&w) { (w - 0.5) * hs.span } else { f32::NAN };
                [(xs[i] - cx) * k, (ys[i] - cy) * k, z, w]
            })
            .collect();
        let every = (n - 1) / 15;
        self.outline = surface.iter().enumerate().filter(|(i, v)| (i % n).is_multiple_of(every) && (i / n).is_multiple_of(every) && v[2].is_finite()).map(|(_, v)| Vec3::new(v[0], v[1], v[2])).collect();
        if let Some(g) = self.gpu.as_mut() {
            g.surf.data_mut().copy_from_slice(&surface);
            for (i, v) in g.flat.data_mut().iter_mut().enumerate() {
                let w = hs.share(zs[i] as f64);
                let (u, t) = ((i % n) as f32 / (n - 1) as f32, (i / n) as f32 / (n - 1) as f32);
                *v = [u, t, 0.0, if w.is_finite() { w.clamp(0.0, 1.0) } else { f32::NAN }];
            }
        }
        self.dirty = false;
        self.at = None;
    }

    /// What is worked out at the drawn point, up to date.
    fn at(&mut self) -> &At {
        if self.dirty {
            self.rebuild();
        }
        if self.at.as_ref().is_none_or(|a| a.p != self.p) {
            let mut a = At::work_out(&self.f.expr, self.p, self.map, self.h, self.slope);
            if a.eig.v1.dot(self.last_v1) < 0.0 {
                a.eig.v1 = -a.eig.v1;
            }
            self.last_v1 = a.eig.v1;
            self.at = Some(a);
            self.fit_filled = false;
        }
        self.at.as_ref().expect("worked out just now")
    }

    /// A point of the plane in the scene, at height `z`.
    fn world(&self, p: DVec2, z: f32) -> Vec3 {
        let k = HALF as f64 / self.map.e;
        let q = (p - self.map.centre) * k;
        Vec3::new(q.x as f32, q.y as f32, z)
    }

    /// The part of the paraboloid fitted at the point that is drawn, in coordinates
    /// along the eigenvectors (from the point, along `v1` and along `v2`): round the
    /// point, and on as far as its flat point, where Newton's step lands.
    ///
    /// A flat point off the map is reached for less the further off it is, and not at
    /// all from half the map's width beyond its edge: so the drawn part changes
    /// smoothly as the point moves, also where the step grows without limit and turns
    /// round (where H is singular), instead of leaping from one side to the other.
    fn fit_box(&self, at: &At) -> [[f64; 2]; 2] {
        let margin = 0.25 * self.map.e;
        let d = match at.step {
            Step::Go(d) if self.newton_shown() => d * self.reached(at.p + d),
            _ => DVec2::ZERO,
        };
        let most = 2.0 * self.map.e;
        [d.dot(at.eig.v1), d.dot(at.eig.v2())].map(|c| [(c.min(0.0) - margin).max(-most), (c.max(0.0) + margin).min(most)])
    }

    /// How much of the way to a point off the map is drawn: all of it while it is on the
    /// map, less and less the further it is off, and none from half the map's width
    /// beyond the edge.
    fn reached(&self, q: DVec2) -> f64 {
        let off = ((q - self.map.centre).abs() - DVec2::splat(self.map.e)).max_element().max(0.0);
        if q.is_finite() {
            (1.0 - off / (0.5 * self.map.e)).max(0.0)
        } else {
            0.0
        }
    }

    /// The height a value of the fitted paraboloid is drawn at, a hair above the
    /// surface it touches; none beyond the heights drawn.
    fn fit_z(&self, v: f64) -> f32 {
        let w = self.heights.share(v);
        if (-0.08..=1.08).contains(&w) {
            (w - 0.5) * self.heights.span + 0.012
        } else {
            f32::NAN
        }
    }

    /// Fill the paraboloid fitted at the point: a rectangle round it with its sides
    /// along the eigenvectors (so its grid lines run along them), within the map.
    fn fill_fit(&mut self, at: &At) {
        let [[s0, s1], [t0, t1]] = self.fit_box(at);
        let (v1, v2) = (at.eig.v1, at.eig.v2());
        let n = FIT_N as usize;
        let data: Vec<[f32; 4]> = (0..n * n)
            .map(|idx| {
                let (i, j) = (idx % n, idx / n);
                let s = s0 + (s1 - s0) * i as f64 / (n - 1) as f64;
                let t = t0 + (t1 - t0) * j as f64 / (n - 1) as f64;
                let d = v1 * s + v2 * t;
                let z = if self.map.contains(at.p + d) { self.fit_z(at.local.model(d)) } else { f32::NAN };
                let q = self.world(at.p + d, z);
                [q.x, q.y, q.z, 0.0]
            })
            .collect();
        if let Some(g) = self.gpu.as_mut() {
            g.fit.data_mut().copy_from_slice(&data);
        }
    }

    /// Whether Newton's step and the runs are drawn: not while the tour moves the point
    /// (between its stops, where H changes sign, the step sweeps across the map from
    /// one edge to the other and back, which is not what the tour is about; at the stops
    /// there is no step).
    fn newton_shown(&self) -> bool {
        self.tour.is_none()
    }

    /// The view that fits the picture into the part of the canvas the panels leave free.
    fn the_view(&self) -> View {
        let lay = Layout::of(self.canvas, self.inset);
        let (right, up) = screen_axes(YAW);
        let (mut lo, mut hi) = (Vec2::splat(f32::MAX), Vec2::splat(f32::MIN));
        let corners = [-HALF, HALF].into_iter().flat_map(|x| [-HALF, HALF].map(|y| Vec3::new(x, y, 0.0)));
        for p in self.outline.iter().copied().chain(corners) {
            let s = Vec2::new(p.dot(right), p.dot(up));
            (lo, hi) = (lo.min(s), hi.max(s));
        }
        let middle = (lo + hi) * 0.5;
        // (room round it for the nearer side, which is seen larger)
        let size = (hi - lo) * 1.06;
        let (min, max) = lay.picture;
        let caption = if self.canvas.y >= 300.0 && !lay.narrow { 30.0 } else { 0.0 };
        let (min, max) = (min + Vec2::new(0.0, 6.0), max - Vec2::new(0.0, 6.0 + caption));
        let region = (max - min).max(Vec2::splat(60.0));
        let per_unit = (region.x / size.x).min(region.y / size.y).max(1.0);
        let dist = (self.canvas.y * 0.5 / per_unit / TAN_HALF_FOV).clamp(4.0, 80.0);
        let per_unit = self.canvas.y * 0.5 / (dist * TAN_HALF_FOV);
        let shift = ((min + max) * 0.5 - self.canvas * 0.5) / per_unit;
        let target = right * middle.x + up * middle.y - right * shift.x + up * shift.y;
        View::Orbit { target, yaw: YAW, pitch: PITCH, dist }
    }
}

impl Scene for Hessian {
    fn view(&self) -> View {
        self.the_view()
    }

    fn take_view(&mut self) -> Option<View> {
        std::mem::take(&mut self.new_view).then(|| self.the_view())
    }

    fn settings(&self, s: &mut ViewSettings) {
        s.grid = false;
    }

    fn animated(&self) -> bool {
        self.tour.is_some() || self.p != self.goal || self.between != self.goal
    }

    fn ui(&mut self, ui: &mut egui::Ui, _: &Gfx) {
        ui::note(ui, r"The gradient says which way a surface $z = f(x, y)$ slopes; the second derivatives say how that slope changes, which is how the surface curves. At the white point $p$ they make the Hessian $H$, and with the gradient they give the paraboloid that fits the surface best there (the white sheet):");
        ui::equation(ui, r"f(p + d) \approx f(p) + \nabla f \cdot d + \tfrac{1}{2}\, d^\top H d");
        ui::note(ui, "Newton's method steps to where that paraboloid is flat, fits a new one there, and steps again.");

        ui::section(ui, "Things to try");
        for (k, t) in TRIALS.iter().enumerate() {
            if ui::experiment(ui, t.label, t.look_for) {
                self.set_up(k);
            }
        }

        ui::section(ui, "The function");
        if self.f.ui(ui, "z =") {
            self.tour = None;
            (self.dirty, self.at) = (true, None);
        }
        let names: Vec<&str> = PRESETS.iter().map(|p| p.name).collect();
        if let Some(k) = ui::buttons(ui, &names) {
            self.load(k);
        }
        ui::note(ui, "Any formula in $x$ and $y$. Drag the white point, on the surface or on the map.");
        let mut e = self.map.e as f32;
        if ui::slider(ui, r"map $\pm$", &mut e, 0.5..=8.0) {
            self.map.e = e as f64;
            self.place(self.goal, false);
            (self.dirty, self.at, self.tour) = (true, None, None);
        }
        ui::note(ui, "How far the map reaches each way from its middle.");

        ui::section(ui, "Gradient descent");
        let mut h = self.h as f32;
        if ui::slider_log(ui, "step size $h$", &mut h, 0.001..=3.0) {
            self.h = h as f64;
            self.at = None;
        }
        ui::note(ui, r"Each of its steps is $x \leftarrow x - h\,\nabla f(x)$. Too large a step makes it bounce from side to side of a valley, and never settle; too small a one crawls.");
        if ui.button(ui::label_text(ui, "move the point to where Newton's run ends", 12.5, None, false)).clicked() {
            if let Some(q) = self.at().newton.limit {
                self.tour = None;
                self.place(q, true);
            }
        }
        ui::checkbox(ui, &mut self.show_fit, "the fitted paraboloid");
        ui::checkbox(ui, &mut self.show_runs, "the runs of steps");

        ui::section(ui, "Reading the picture");
        ui::key(
            ui,
            &[
                (CMAP.color(0.55), "the surface $z = f(x, y)$; on the map, the same heights seen from above, with contour lines"),
                (POINT, "the point $p$: drag it, on the surface or on the map"),
                (FIT, "the paraboloid fitted at $p$, which touches the surface there and curves as it does; dotted on the map, its level curves"),
                (GRAD, r"the gradient $\nabla f$ at $p$, on the map"),
                (V1, r"the direction of greatest curvature, the eigenvector of $H$ with the larger eigenvalue $\lambda_1$; through $p$, the paraboloid (thick) and the surface (thin) along it"),
                (V2, r"the direction of least curvature, at right angles to it, with the smaller eigenvalue $\lambda_2$"),
                (NEWTON, r"Newton's step $-H^{-1}\nabla f$, to the paraboloid's flat point; on the map, the run of such steps, a dot for each and a ring where it ends"),
                (DESCENT, "the run of gradient descent from the same point, likewise"),
            ],
        );
        ui::section(ui, "Words");
        ui::term(ui, "Hessian $H$:", r"the matrix of second derivatives, $\begin{bmatrix} f_{xx} & f_{xy} \\ f_{xy} & f_{yy} \end{bmatrix}$. It is symmetric ($f_{xy} = f_{yx}$) wherever the second derivatives are continuous.");
        ui::term(ui, "Curvature (here):", r"the second derivative along a direction: how fast the slope that way changes. Along a unit direction $u$ it is $u^\top H u$.");
        ui::term(ui, "Eigenvector, eigenvalue:", r"a direction $v$ that $H$ only stretches, $Hv = \lambda v$. The curvature is greatest, $\lambda_1$, along one eigenvector, and least, $\lambda_2$, along the other.");
        ui::term(ui, "Stationary point:", r"where $\nabla f = 0$, so the tangent plane is level. The eigenvalues of $H$ there tell a minimum (both positive), a maximum (both negative) and a saddle point (one of each); if one of them is $0$, they cannot tell.");
        ui::term(ui, "Saddle point:", "a stationary point with the surface rising one way and falling another, like a mountain pass.");
        ui::term(ui, "Singular:", r"a matrix with an eigenvalue $0$. It has no inverse, so $-H^{-1}\nabla f$ does not exist.");
        ui::term(ui, "Newton's method:", r"$x \leftarrow x - H^{-1}\nabla f(x)$, repeated: each step goes to the flat point of the paraboloid fitted where it stands.");

        ui::section(ui, "Numbers");
        let a = self.at().clone();
        let l = a.local;
        let value = |v: f64| format!("${}$", short(v));
        ui::readout(ui, "$p$", format!("${}$", tuple(&[a.p.x, a.p.y], 3)));
        ui::readout(ui, "$f(p)$", value(l.f));
        ui::readout_colored(ui, r"$\nabla f$", if a.stationary { "$0$".to_string() } else { format!("$({}, {})$", short(l.g.x), short(l.g.y)) }, GRAD);
        ui::readout(ui, r"$f_{xx},\ f_{xy},\ f_{yy}$", format!(r"${},\ {},\ {}$", short(l.h.a), short(l.h.b), short(l.h.c)));
        ui::readout_colored(ui, r"$\lambda_1$", value(a.eig.l1), V1);
        ui::readout_colored(ui, r"$\lambda_2$", value(a.eig.l2), V2);
        ui::readout(ui, r"$\det H = \lambda_1 \lambda_2$", value(l.h.det()));
        let step = match a.step {
            Step::Go(d) => format!("$({}, {})$", short(d.x), short(d.y)),
            Step::Here => r"none: $\nabla f = 0$".into(),
            Step::Singular => "none: $H$ is singular".into(),
            Step::Undefined => "none".into(),
        };
        ui::readout_colored(ui, "Newton's step", step, NEWTON);
        ui::readout_series(ui, "Newton's method", run_words(&a.newton, a.newton_kind), NEWTON);
        ui::readout_series(ui, "gradient descent", run_words(&a.descent, a.descent_kind), DESCENT);
        ui::note(ui, r"Steps counted until the run stays within $10^{-6}$ of the point it ends at.");
        if !l.is_finite() {
            ui::note(ui, "$f$ has no value, or no second derivatives, at this point.");
        }
        ui::see_also(
            ui,
            &[
                ("surface", "the gradient and the tangent plane"),
                ("quadform", "the paraboloid of a symmetric matrix, and its eigenvectors"),
                ("descent", "Newton's method among other optimisers"),
                ("spectral", "a symmetric matrix as its eigenvalues and eigenvectors"),
                ("lagrange", "stationary points on a curve"),
            ],
        );
    }

    fn frame(&mut self, f: &mut Frame) {
        every_batch(f);
        // the tour, or the glide after a jump
        if let Some(tour) = self.tour.as_mut() {
            tour.t += f.dt;
            let (q, done) = tour.at();
            (self.goal, self.between, self.p) = (q, q, q);
            if done {
                self.tour = None;
                self.fit_filled = false;
            }
        } else {
            self.follow(f.dt as f64);
        }
        if self.gpu.is_none() {
            self.gpu = Some(Gpu::new(f.gfx));
            self.dirty = true;
        }
        if self.dirty {
            self.rebuild();
        }

        // the panels' places follow the canvas, and the picture's place follows theirs
        let view = f.view_size();
        let lay = Layout::of(view, f.top_inset);
        let before = if (self.canvas, self.inset) == (view, f.top_inset) { lay } else { Layout::of(self.canvas, self.inset) };
        // (a canvas squeezed to a sliver is one the controls have opened over, on a
        // phone: the picture is not seen then, and is left as it was for when they close)
        let covered = view.x < 160.0;
        let changed = !covered && (lay.narrow != before.narrow || lay.panel.is_some() != before.panel.is_some() || (view - self.canvas).abs().max_element() > 24.0);
        let first = !covered && !self.measured && (view - self.canvas).abs().max_element() > 24.0;
        self.new_view |= changed || first;
        if !covered && (changed || !self.measured) {
            self.canvas = view;
            self.measured = true;
        }
        self.inset = f.top_inset;
        let pointer = f.ptr.pos.map(|p| p / f.px);
        if pointer.is_some_and(|p| lay.covers(p)) {
            // the panels are for reading, their button and the point: no turning the
            // view from there
            f.handles.capture();
            f.handles.claim_clicks();
        }

        // the point: dragged on the map, or on the surface
        let on_map = lay.panel.map(|m| OnMap { at: m.at, side: m.side, map: self.map });
        let mut moved = None;
        if let Some(m) = on_map {
            let mut handle = f.hud_at(2, m.px(self.p));
            if f.drag3(1, &mut handle, None, POINT) {
                moved = on_screen(f, handle).map(|s| m.point(s));
            }
        }
        let hs = self.heights;
        let value = Local::at(&self.f.expr, self.p).f;
        let lifted = if value.is_finite() { hs.z(value).max(BOX_Z[0]).min(BOX_Z[1]) } else { 0.0 };
        let mut top = self.world(self.p, lifted);
        if f.drag3(2, &mut top, Some(Vec3::Z), POINT) {
            let k = self.map.e / HALF as f64;
            moved = Some(self.map.centre + DVec2::new(top.x as f64, top.y as f64) * k);
        }
        if let Some(q) = moved.filter(|q| q.is_finite()) {
            let q = self.map.keep(q);
            self.tour = None;
            (self.goal, self.between, self.p) = (q, q, q);
        }

        let at = self.at().clone();
        let fitted = self.show_fit && at.local.is_finite();
        if fitted && !self.fit_filled {
            self.fill_fit(&at);
            self.fit_filled = true;
        }
        if let Some(g) = self.gpu.as_mut() {
            f.surface(&mut g.surf);
            if fitted {
                f.surface(&mut g.fit);
            }
        }
        let mut room = Room::new(f);
        for (min, max) in lay.card.into_iter().chain(lay.runs).chain(lay.panel.map(|m| (m.min, m.max))) {
            room.wall(min - Vec2::splat(4.0), max + Vec2::splat(4.0));
        }
        self.picture(f, &at, &mut room, lay.narrow);
        if let Some(rect) = lay.card {
            self.card(f, rect, &at, lay.narrow);
        }
        if let Some(panel) = lay.panel {
            self.map_panel(f, panel, &at, pointer);
        }
        if let Some((min, max)) = lay.runs {
            hud_panel(f, min, max);
            self.runs(f, min + Vec2::splat(PAD), max.x - min.x - 2.0 * PAD, Rows::Full, &at, pointer);
        }
        if !lay.narrow && lay.picture.1.y - lay.picture.0.y >= 260.0 {
            // what the picture shows that has no label of its own, at its foot
            let mut parts = vec!["white sheet: the paraboloid fitted at $p$", "green: Newton's step, down the paraboloid to its flat point"];
            if self.squash > 0.0 {
                parts.push("tall walls are drawn lower than they are");
            }
            for (i, part) in parts.iter().rev().enumerate() {
                let at = Vec2::new(lay.picture.0.x + 18.0, lay.picture.1.y - 10.0 - 21.0 * i as f32);
                f.caption_px(at, *part, 12.0, pal::DIM, Align2::LEFT_BOTTOM);
            }
        }
    }
}

impl Hessian {
    /// The surface's part of the picture: the point on it, the paraboloid's grid and
    /// axes, the surface along the eigenvectors, and Newton's step down the paraboloid
    /// (with the paraboloid named, if `named`).
    fn picture(&mut self, f: &mut Frame, at: &At, room: &mut Room, named: bool) {
        let hs = self.heights;
        let l = at.local;
        if !l.is_finite() {
            return;
        }
        let top = self.world(at.p, hs.z(l.f));
        let on_surface = (-1e-3..=1.0 + 1e-3).contains(&hs.share(l.f));
        if on_surface {
            f.sphere(top, 0.13, POINT);
            room.label(f, "p", top, "$p$", 15.0, POINT, Vec2::new(-1.0, -1.0), 10.0, true);
        }
        let (v1, v2) = (at.eig.v1, at.eig.v2());
        let [along1, along2] = self.fit_box(at);
        if !self.show_fit {
            return;
        }
        // (every curve is drawn with as many points in every frame, spread over the part
        // of it that is seen: see `Part`)
        let (lo, hi) = (self.map.lo(), self.map.hi());
        let paraboloid = |d: DVec2| self.world(at.p + d, self.fit_z(l.model(d)) + 0.008);
        let seen = |d: DVec2| self.map.contains(at.p + d) && paraboloid(d).is_finite();
        // where the paraboloid passes under the surface, its grid is seen through it,
        // faintly (at heights held to those drawn)
        let [[s0, s1], [t0, t1]] = [along1, along2];
        let held = |d: DVec2| self.world(at.p + d, (hs.share(l.model(d)).clamp(-0.08, 1.08) - 0.5) * hs.span + 0.012);
        for k in 0..=6 {
            let c = k as f64 / 6.0;
            let (s, t) = (s0 + (s1 - s0) * c, t0 + (t1 - t0) * c);
            for (from, to) in [(v1 * s + v2 * t0, v1 * s + v2 * t1), (v1 * s0 + v2 * t, v1 * s1 + v2 * t)] {
                if let Some((a, b)) = clip(at.p + from, at.p + to, lo, hi) {
                    let pts: Vec<Vec3> = (0..=24).map(|i| held(a.lerp(b, i as f64 / 24.0) - at.p)).collect();
                    lines_of(f, Part::GhostGrid).polyline(pts, 1.0, FIT.a(0.3));
                }
            }
        }
        // the surface itself through p along the eigenvectors (thin), and the
        // paraboloid's two axes there (thick): parabolas curved by λ₁ and λ₂, which the
        // surface follows near p and leaves further out
        let reach = 3.0 * self.map.e;
        for (v, color, part) in [(v1, V1, Part::SliceOne), (v2, V2, Part::SliceTwo)] {
            let Some((a, b)) = clip(at.p - v * reach, at.p + v * reach, lo, hi) else { continue };
            let pts: Vec<Vec3> = (0..=160)
                .map(|i| {
                    let q = a.lerp(b, i as f64 / 160.0);
                    let w = hs.share(self.f.expr.eval(&[q.x as f32, q.y as f32]) as f64);
                    // (none beyond the heights drawn: a pole)
                    if (-1e-3..=1.0 + 1e-3).contains(&w) {
                        self.world(q, (w - 0.5) * hs.span + 0.012)
                    } else {
                        Vec3::NAN
                    }
                })
                .collect();
            lines_of(f, part).polyline(pts, 1.5, color.a(0.75));
        }
        // (on a phone, which has no caption under the picture, the sheet is named on it)
        if named {
            let corners: Vec<Vec3> = [(s1, t1), (s0, t1), (s1, t0), (s0, t0)].iter().map(|(s, t)| v1 * *s + v2 * *t).filter(|d| seen(*d)).map(paraboloid).collect();
            room.label_any(f, "sheet", &corners, "paraboloid fitted at $p$", 12.0, FIT, Vec2::new(1.0, -1.0), 6.0, true);
        }
        for (v, [t0, t1], color, part) in [(v1, along1, V1, Part::AxisOne), (v2, along2, V2, Part::AxisTwo)] {
            let Some((a, b)) = stretch(t0, t1, |t| seen(v * t)) else { continue };
            let pts: Vec<Vec3> = (0..=48).map(|i| paraboloid(v * (a + (b - a) * i as f64 / 48.0))).collect();
            lines_of(f, Part::GhostAxes).polyline(pts.iter().copied(), 1.6, color.a(0.4));
            lines_of(f, part).polyline(pts, 2.4, color);
        }
        // Newton's step: down (or up) the paraboloid to its flat point
        if let (Step::Go(d), true) = (at.step, self.newton_shown()) {
            if let Some((_, b)) = stretch(0.0, 1.0, |t| seen(d * t)) {
                let pts: Vec<Vec3> = (0..=40).map(|i| paraboloid(d * (b * i as f64 / 40.0)) + Vec3::Z * 0.008).collect();
                lines_of(f, Part::GhostPath).polyline(pts.iter().copied(), 2.0, NEWTON.a(0.45));
                lines_of(f, Part::Path).polyline(pts, 3.0, NEWTON);
            }
            let flat = paraboloid(d) + Vec3::Z * 0.008;
            if seen(d) {
                f.sphere(flat, 0.11, NEWTON);
                let ring: Vec<Vec3> = (0..=24).map(|i| flat + Vec3::from((Vec2::from_angle(i as f32 / 24.0 * TAU) * 0.09, 0.0))).collect();
                lines_of(f, Part::GhostPath).polyline(ring, 2.0, NEWTON.a(0.6));
                let towards = on_screen(f, flat).zip(on_screen(f, top)).map_or(Vec2::X, |(a, b)| a - b);
                room.label(f, "flat", flat, "flat point", 12.5, NEWTON, towards, 8.0, true);
            }
        }
    }

    /// The card of numbers at the point: p, the gradient, H and its eigenvalues, and
    /// what they say (on a phone, without the sentence).
    fn card(&self, f: &mut Frame, (min, max): (Vec2, Vec2), at: &At, compact: bool) {
        hud_panel(f, min, max);
        let l = at.local;
        let x0 = min.x + 12.0;
        let width = max.x - min.x - 24.0;
        let mut y = min.y + 10.0;
        f.text_px(Vec2::new(x0, y), format!("$p = {}$", tuple(&[at.p.x, at.p.y], 2)), CARD_TEXT, POINT, Align2::LEFT_TOP);
        f.text_px(Vec2::new(max.x - 12.0, y), format!("$f(p) = {}$", short(l.f)), CARD_TEXT, pal::DIM, Align2::RIGHT_TOP);
        y += 20.0;
        let grad = if at.stationary { r"$\nabla f = 0$".to_string() } else { format!(r"$\nabla f = ({}, {})$", short(l.g.x), short(l.g.y)) };
        f.text_px(Vec2::new(x0, y), grad, CARD_TEXT, GRAD, Align2::LEFT_TOP);
        y += 22.0;
        let msize = matrix_size(2, 2, 6, CARD_TEXT);
        f.text_px(Vec2::new(x0, y + msize.y * 0.5), "$H =$", 14.0, pal::TEXT, Align2::LEFT_CENTER);
        let mx = x0 + f.text_size("$H =$", 14.0).x + 6.0;
        let cell = |v: f64| (format!("${}$", short(v)), pal::TEXT);
        hud_matrix(f, Vec2::new(mx, y), &[vec![cell(l.h.a), cell(l.h.b)], vec![cell(l.h.b), cell(l.h.c)]], 6, CARD_TEXT, pal::DIM);
        let lx = mx + msize.x + 16.0;
        f.text_px(Vec2::new(lx, y + msize.y * 0.25), format!(r"$\lambda_1 = {}$", short(at.eig.l1)), CARD_TEXT, V1, Align2::LEFT_CENTER);
        f.text_px(Vec2::new(lx, y + msize.y * 0.75), format!(r"$\lambda_2 = {}$", short(at.eig.l2)), CARD_TEXT, V2, Align2::LEFT_CENTER);
        y += msize.y + 8.0;
        let (head, detail) = verdict(at.kind, at.stationary);
        f.text_px(Vec2::new(x0, y), head, 13.5, pal::TEXT, Align2::LEFT_TOP);
        y += 19.0;
        if !compact {
            wrapped_px(f, Vec2::new(x0, y), width, detail, 12.0, pal::DIM);
        }
        let step = match at.step {
            Step::Go(d) => format!(r"Newton's step $-H^{{-1}}\nabla f = ({}, {})$", short(d.x), short(d.y)),
            Step::Here => r"Newton's step: none, as $\nabla f = 0$ here".into(),
            Step::Singular => r"Newton's step: none, as $H$ is singular".into(),
            Step::Undefined => "Newton's step: none".into(),
        };
        f.text_px(Vec2::new(x0, max.y - 9.0), step, 12.5, NEWTON, Align2::LEFT_BOTTOM);
    }

    /// The map: the function seen from above, with the point, the gradient, the
    /// eigenvectors, Newton's step and both runs; and under it, how the runs went.
    fn map_panel(&mut self, f: &mut Frame, panel: MapPanel, at: &At, pointer: Option<Vec2>) {
        hud_panel(f, panel.min, panel.max);
        let m = OnMap { at: panel.at, side: panel.side, map: self.map };
        let (lo, hi) = (self.map.lo(), self.map.hi());
        let s = panel.side;
        // the heights, flat: the map's surface laid on the screen
        let o = f.hud_at(1, m.at + Vec2::new(0.0, s));
        let across = f.hud_at(1, m.at + Vec2::new(s, s)) - o;
        let upward = f.hud_at(1, m.at) - o;
        let normal = across.cross(upward).normalize_or_zero() * 1e-3;
        if let Some(g) = self.gpu.as_mut() {
            g.flat.style.model = Mat4::from_cols(across.extend(0.0), upward.extend(0.0), normal.extend(0.0), o.extend(1.0));
            f.surface(&mut g.flat);
        }
        hud_outline(f, m.at, m.at + Vec2::splat(s), 1.0, pal::DIM.a(0.8));
        // (what the map spans, said as briefly as the room asks)
        let n = |v: f64| num(v, if (v * 10.0).fract().abs() < 1e-9 { 1 } else { 2 });
        let span = if lo.x == lo.y && hi.x == hi.y {
            format!(r"${} \le x, y \le {}$", n(lo.x), n(hi.x))
        } else {
            format!(r"${} \le x \le {}$, ${} \le y \le {}$", n(lo.x), n(hi.x), n(lo.y), n(hi.y))
        };
        let long = format!("from above: {span}");
        let title = if f.text_size(&long, 12.0).x <= panel.max.x - panel.min.x - 2.0 * PAD { long } else { span };
        f.text_px(Vec2::new(panel.min.x + PAD, panel.min.y + PAD - 1.0), title, 12.0, pal::DIM, Align2::LEFT_TOP);

        // labels stay on the map: everything round it is walled off
        let mut room = Room::new(f);
        let view = f.view_size();
        let (a, b) = (m.at, m.at + Vec2::splat(s));
        room.wall(Vec2::ZERO, Vec2::new(view.x, a.y));
        room.wall(Vec2::new(0.0, b.y), view);
        room.wall(Vec2::ZERO, Vec2::new(a.x, view.y));
        room.wall(Vec2::new(b.x, 0.0), view);
        let l = at.local;
        let here = m.px(at.p);
        let line = |f: &mut Frame, room: &mut Room, part: Part, p: DVec2, q: DVec2, width: f32, color: Color, head: bool| {
            if let Some((p, q)) = clip(p, q, lo, hi) {
                let (sp, sq) = (m.px(p), m.px(q));
                if head {
                    arrow_px(f, part, sp, sq, width, color);
                } else {
                    let (a, b) = (f.hud(sp), f.hud(sq));
                    lines_of(f, part).segment(a, b, width, color);
                }
                room.segment_px(sp, sq);
                Some(sq)
            } else {
                None
            }
        };

        // the runs: every point a step reached
        if self.show_runs && self.newton_shown() {
            for (run, color, part) in [(&at.descent, DESCENT, Part::Descent), (&at.newton, NEWTON, Part::Newton)] {
                let mut segments: Vec<(Vec2, Vec2)> = Vec::new();
                for w in run.pts.windows(2) {
                    if let Some((p, q)) = clip(w[0], w[1], lo, hi) {
                        segments.push((m.px(p), m.px(q)));
                    }
                }
                for (k, (p, q)) in segments.iter().enumerate() {
                    let (p3, q3) = (f.hud(*p), f.hud(*q));
                    lines_of(f, part).segment(p3, q3, if part == Part::Newton { 1.6 } else { 1.3 }, color.a(0.8));
                    if k < 60 {
                        room.segment_px(*p, *q);
                    }
                }
                let spots: Vec<Vec2> = run.pts.iter().skip(1).take(300).filter(|q| self.map.contains(**q)).map(|q| m.px(*q)).collect();
                for spot in spots.iter().take(40) {
                    room.boxed(*spot - Vec2::splat(3.0), *spot + Vec2::splat(3.0));
                }
                let dots: Vec<Vec3> = spots.iter().map(|q| f.hud(*q)).collect();
                let batch = dots_of(f, if part == Part::Newton { Dots::Newton } else { Dots::Descent });
                for d in dots {
                    batch.push(d, if part == Part::Newton { 5.5 } else { 4.0 }, color);
                }
                if let Some(q) = run.limit.filter(|q| self.map.contains(*q)) {
                    let ring: Vec<Vec3> = (0..=24).map(|i| f.hud(m.px(q) + Vec2::from_angle(i as f32 / 24.0 * TAU) * if part == Part::Newton { 8.0 } else { 11.0 })).collect();
                    lines_of(f, part).polyline(ring, 1.6, color);
                }
            }
        }

        // the paraboloid's level curves, dotted: through p, and a quarter as high above
        // its flat point (for a saddle, also the two lines through it at its own height)
        let reached = match at.step {
            Step::Go(d) => self.reached(at.p + d) > 0.0,
            _ => true,
        };
        if self.show_fit && self.newton_shown() && reached {
            let mut spots: Vec<Vec2> = Vec::new();
            // (dots at evenly spread values of each curve's parameter, about seven pixels
            // apart: as many as its length on the map asks, counted in eights, so that
            // while the curve changes with the point the dots keep their number, and
            // each moves with it)
            for curve in model_levels(at, self.map.e).into_iter().filter(|c| c.len() >= 2) {
                let length: f32 = curve.windows(2).filter_map(|w| clip(w[0], w[1], lo, hi)).map(|(a, b)| (m.px(b) - m.px(a)).length()).sum();
                let dots = (8.0 * (length / 56.0).ceil()).max(8.0).min(240.0) as usize;
                let last = (curve.len() - 1) as f64;
                for k in 0..=dots {
                    let at = last * k as f64 / dots as f64;
                    let (i, t) = ((at.floor() as usize).min(curve.len() - 2), at - at.floor());
                    let q = curve[i].lerp(curve[i + 1], t.min(1.0));
                    if self.map.contains(q) {
                        spots.push(m.px(q));
                    }
                }
            }
            let spots: Vec<Vec3> = spots.into_iter().map(|q| f.hud(q)).collect();
            let batch = dots_of(f, Dots::Levels);
            for spot in spots {
                batch.push(spot, 2.4, FIT.a(0.9));
            }
        }
        // the eigenvectors, a cross at the point; the gradient (its length shows the
        // slope, up to a limit); Newton's step. Their names are placed once all of them
        // are drawn, so that none is written over another's line.
        let mut names: Vec<Name> = Vec::new();
        let arm = 0.075 * 2.0 * self.map.e;
        if l.is_finite() {
            for (v, color, lambda, name, part) in [(at.eig.v1, V1, at.eig.l1, "l1", Part::CrossOne), (at.eig.v2(), V2, at.eig.l2, "l2", Part::CrossTwo)] {
                if let Some(tip) = line(f, &mut room, part, at.p - v * arm, at.p + v * arm, 2.4, color, false) {
                    let text = format!(r"$\lambda_{} = {}$", if name == "l1" { 1 } else { 2 }, short(lambda));
                    // (beside either end of its line)
                    let tail = clip(at.p + v * arm, at.p - v * arm, lo, hi).map_or(tip, |(_, q)| m.px(q));
                    names.push((name, vec![tip, tail], text, 13.0, color, tip - here));
                }
            }
        }
        if !at.stationary && l.is_finite() {
            let len = (l.g.length() / (3.0 * self.slope)).sqrt().clamp(0.35, 1.0) * 0.15 * 2.0 * self.map.e;
            if let Some(tip) = line(f, &mut room, Part::Gradient, at.p, at.p + l.g.normalize_or_zero() * len, 2.6, GRAD, true) {
                names.push(("grad", vec![tip], r"$\nabla f$".to_string(), 14.0, GRAD, tip - here));
            }
        }
        match at.step {
            _ if !self.newton_shown() => {}
            Step::Go(d) => {
                let to = at.p + d;
                let inside = self.map.contains(to);
                if self.reached(to) > 0.0 {
                    if let Some(tip) = line(f, &mut room, Part::Step, at.p, to, 2.8, NEWTON, inside) {
                        let text = if inside { "Newton's step".to_string() } else { format!("Newton's step: {} long, off the map", short(d.length())) };
                        names.push(("step", vec![tip], text, 12.5, NEWTON, tip - here));
                    }
                } else {
                    // (far off the map: only its length is told, beside the point; the
                    // arrow, clipped at the edge, would sweep round it as the step
                    // grows without limit and turns round)
                    names.push(("far", vec![here], format!("Newton's step: {} long, far off the map", short(d.length())), 12.5, NEWTON, Vec2::new(0.0, 1.0)));
                }
            }
            Step::Singular => names.push(("far", vec![here], r"no Newton step: $H$ is singular".to_string(), 12.5, NEWTON, Vec2::new(0.0, 1.0))),
            Step::Here | Step::Undefined => {}
        }
        room.boxed(here - Vec2::splat(8.0), here + Vec2::splat(8.0));
        for (name, places, text, size, color, towards) in names {
            if matches!(name, "l1" | "l2") {
                // (the eigenvalues are always named)
                name_beside(f, &mut room, name, &places, &text, size, color, towards);
            } else {
                // (the arrows only where there is room)
                let points: Vec<Vec3> = places.iter().map(|q| f.hud(*q)).collect();
                room.label_any(f, name, &points, &text, size, color, towards, if towards == Vec2::new(0.0, 1.0) { 14.0 } else { 5.0 }, true);
            }
        }

        // how the runs went
        if panel.rows != Rows::None {
            self.runs(f, Vec2::new(panel.min.x + PAD, m.at.y + s + 6.0), panel.max.x - panel.min.x - 2.0 * PAD, panel.rows, at, pointer);
        }
    }

    /// How the runs went, from `at` down: a row for each, Newton's distances from its
    /// end, and the button that moves the point there (the last two in full only).
    fn runs(&mut self, f: &mut Frame, at_px: Vec2, width: f32, rows: Rows, at: &At, pointer: Option<Vec2>) {
        let x0 = at_px.x;
        let mut y = at_px.y;
        let full = rows == Rows::Full;
        if full {
            f.text_px(Vec2::new(x0, y), r"Runs from $p$, each until within $10^{-6}$ of its end:", 12.0, pal::DIM, Align2::LEFT_TOP);
            y += ROW_H;
        }
        let both = [("Newton", &at.newton, at.newton_kind, NEWTON), ("gradient descent", &at.descent, at.descent_kind, DESCENT)];
        for (i, (name, run, kind, color)) in both.into_iter().enumerate() {
            f.dot(f.hud_at(2, Vec2::new(x0 + 4.0, y + 8.0)), 7.0, color);
            let words = if i == 1 { format!("{name}, $h = {}$: {}", short(self.h), run_words(run, kind)) } else { format!("{name}: {}", run_words(run, kind)) };
            f.text_px(Vec2::new(x0 + 13.0, y), words, 12.5, color, Align2::LEFT_TOP);
            y += ROW_H;
            if i == 0 && full {
                // (as many of the last distances as fit)
                for most in (1..=5).rev() {
                    let text = distances_text(run, most);
                    if text.is_empty() {
                        break;
                    }
                    let text = format!("to its end: {text}");
                    if most == 1 || f.text_size(&text, 12.0).x <= width - 13.0 {
                        f.text_px(Vec2::new(x0 + 13.0, y), text, 12.0, pal::DIM, Align2::LEFT_TOP);
                        break;
                    }
                }
                y += ROW_H;
            }
        }
        // the button that puts the point where Newton's run ends
        if full {
            if let Some(q) = at.newton.limit.filter(|q| (*q - at.p).length() > 1e-9) {
                let (bmin, bmax) = (Vec2::new(x0, y + 6.0), Vec2::new(x0 + width.min(250.0), y + 6.0 + BUTTON_H - 6.0));
                if hud_button(f, bmin, bmax, "move the point to where Newton ends", pointer) {
                    self.tour = None;
                    self.place(q, true);
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn parse(formula: &str) -> Expr {
        Expr::parse(formula, &["x", "y"]).unwrap()
    }

    /// The function itself, in double precision, from the same bytecode.
    fn value(expr: &Expr, x: f64, y: f64) -> f64 {
        expr.eval_num::<Hyper>(&[Hyper { v: x, ..Default::default() }, Hyper { v: y, ..Default::default() }]).v
    }

    /// The gradient and Hessian against differences of the function's values, for
    /// formulas that use every kind of operation and function.
    #[test]
    fn the_hessian_is_the_second_derivatives() {
        let formulas = [
            "sin(x) cos(0.8 y) + 0.25 sin(x + y)",
            "(1 - x)^2 + 2 (y - x^2)^2",
            "x^3 - 3 x y^2",
            "exp(-x^2 - y^2) / (1 + x^2)",
            "tan(0.3 x) + asin(0.2 y) + acos(0.1 x y) + atan(x - y)",
            "sinh(0.5 x) cosh(0.3 y) + tanh(x y)",
            "ln(2 + x^2) + log10(3 + y^2) + log2(1 + x^2 y^2)",
            "sqrt(5 + x + y) + abs(x + 4) + relu(y + 3)",
            "sigmoid(x y) + gauss(x - y)",
            "(2 + sin(x))^(1 + 0.3 y)",
            "2^(x y) + x^0.5 + y^-2",
            "atan2(y + 0.3, x - 0.2) + min(x, 3) * max(y, -3)",
        ];
        let points = [(0.7, -0.4), (1.3, 0.9), (0.2, 1.7), (2.1, -1.1)];
        for formula in formulas.into_iter().chain(PRESETS.iter().map(|p| p.formula)) {
            let e = parse(formula);
            for (x, y) in points {
                let l = Local::at(&e, DVec2::new(x, y));
                let f = |x: f64, y: f64| value(&e, x, y);
                // central differences at two spacings, combined so that their errors of
                // order s² cancel (Richardson)
                let both = |d: &dyn Fn(f64) -> f64| (4.0 * d(5e-4) - d(1e-3)) / 3.0;
                let fx = both(&|s| (f(x + s, y) - f(x - s, y)) / (2.0 * s));
                let fy = both(&|s| (f(x, y + s) - f(x, y - s)) / (2.0 * s));
                let fxx = both(&|s| (f(x + s, y) - 2.0 * f(x, y) + f(x - s, y)) / (s * s));
                let fyy = both(&|s| (f(x, y + s) - 2.0 * f(x, y) + f(x, y - s)) / (s * s));
                let fxy = both(&|s| (f(x + s, y + s) - f(x + s, y - s) - f(x - s, y + s) + f(x - s, y - s)) / (4.0 * s * s));
                let close = |a: f64, b: f64, tol: f64| (a - b).abs() <= tol * (1.0 + b.abs());
                assert!(l.is_finite(), "{formula} at ({x}, {y}): {l:?}");
                assert!(close(l.f, f(x, y), 1e-15), "{formula}");
                assert!(close(l.g.x, fx, 1e-7) && close(l.g.y, fy, 1e-7), "{formula} at ({x}, {y}): gradient {:?} against ({fx}, {fy})", l.g);
                assert!(
                    close(l.h.a, fxx, 1e-5) && close(l.h.b, fxy, 1e-5) && close(l.h.c, fyy, 1e-5),
                    "{formula} at ({x}, {y}): Hessian {:?} against ({fxx}, {fxy}, {fyy})",
                    l.h
                );
            }
        }
    }

    /// Second derivatives where the function is awkward: a square root at 0 held
    /// fixed, negative bases with whole exponents, a zero exponent, a quotient by a
    /// number, the angle at the origin. Each is the exact answer, and finite.
    #[test]
    fn awkward_places_give_exact_finite_derivatives() {
        let at = |formula: &str, x: f64, y: f64| Local::at(&parse(formula), DVec2::new(x, y));
        // x³ at −2: −8, 12, −12
        let l = at("x^3", -2.0, 0.0);
        assert_eq!((l.f, l.g.x, l.h.a), (-8.0, 12.0, -12.0));
        // y⁻² at −0.5: 4, −2y⁻³ = 16, 6y⁻⁴ = 96
        let l = at("y^-2", 0.0, -0.5);
        assert!((l.f - 4.0).abs() < 1e-12 && (l.g.y - 16.0).abs() < 1e-12 && (l.h.c - 96.0).abs() < 1e-10, "{l:?}");
        // pow with a computed constant exponent and a negative base
        let l = at("pow(x, 3)", -2.0, 1.0);
        assert!((l.f + 8.0).abs() < 1e-12 && (l.g.x - 12.0).abs() < 1e-12 && (l.h.a + 12.0).abs() < 1e-12, "{l:?}");
        // x⁰ at 0: 1, and no slope
        let l = at("x^0 + pow(x, 0)", 0.0, 0.0);
        assert!(l.f == 2.0 && l.g == DVec2::ZERO && l.h == Sym::default(), "{l:?}");
        // √(x) + y at x = 0: the infinite slope along x does not reach y
        let l = at("sqrt(x) + y", 0.0, 1.0);
        assert!(l.g.y == 1.0 && l.h.c == 0.0 && l.h.b == 0.0, "{l:?}");
        // a quotient by a number
        let l = at("(x^2 + y^2) / 4", 1.0, 2.0);
        assert!(l.g == DVec2::new(0.5, 1.0) && l.h == Sym { a: 0.5, b: 0.0, c: 0.5 }, "{l:?}");
        // the angle at the origin has no slope to speak of: zeros, not NaN
        let l = at("atan2(y, x)", 0.0, 0.0);
        assert!(l.is_finite(), "{l:?}");
        // and where the function has no value, nothing is finite, and nothing panics
        let l = at("sqrt(-1 - x^2)", 0.3, 0.0);
        assert!(!l.is_finite() && l.newton(1e-9) == Step::Undefined);
    }

    /// The eigenvalues come larger first, with eigenvectors at right angles that H only
    /// stretches; equal ones take the x axis; and the kinds follow the signs, with a 0
    /// that the rounding has left a hair off still a 0.
    #[test]
    fn eigenvalues_and_what_they_say() {
        let mut rng = Rng::new(7);
        for _ in 0..500 {
            let h = Sym { a: rng.range(-5.0, 5.0) as f64, b: rng.range(-5.0, 5.0) as f64, c: rng.range(-5.0, 5.0) as f64 };
            let e = h.eigen();
            assert!(e.l1 >= e.l2);
            for (l, v) in [(e.l1, e.v1), (e.l2, e.v2())] {
                assert!((h.apply(v) - v * l).length() < 1e-12 * (1.0 + e.big()), "{h:?}: {e:?}");
                assert!((v.length() - 1.0).abs() < 1e-14);
            }
            assert!((e.l1 * e.l2 - h.det()).abs() < 1e-11 * (1.0 + e.big() * e.big()));
        }
        assert_eq!(Sym { a: 3.0, b: 0.0, c: 3.0 }.eigen().v1, DVec2::X);
        let kind = |a: f64, b: f64, c: f64| Kind::of(&Sym { a, b, c }.eigen());
        assert_eq!([kind(2.0, 0.0, 1.0), kind(-1.0, 0.0, -2.0), kind(1.0, 0.0, -1.0), kind(2.0, 0.0, 0.0), kind(0.0, 0.0, 0.0)], [Kind::Bowl, Kind::Dome, Kind::Saddle, Kind::Trough, Kind::Flat]);
        assert_eq!([kind(1.0, 1.0, 1.0), kind(2.0, 0.0, 1e-14), kind(2.0, 0.0, 1e-9)], [Kind::Trough, Kind::Trough, Kind::Bowl]);
        // a singular H gives no step, and says so; a gradient of 0 needs none
        let flat = Local { f: 0.0, g: DVec2::new(1.0, 0.5), h: Sym { a: 1.0, b: 1.0, c: 1.0 } };
        assert_eq!(flat.newton(1e-9), Step::Singular);
        assert_eq!(Local { g: DVec2::ZERO, ..flat }.newton(1e-9), Step::Here);
    }

    /// Newton's step goes to the point where the fitted paraboloid is flat: there the
    /// paraboloid's gradient, ∇f + Hd, is 0, and its height is f − ½ ∇f·H⁻¹∇f.
    #[test]
    fn newtons_step_lands_where_the_paraboloid_is_flat() {
        let mut rng = Rng::new(11);
        for _ in 0..500 {
            let l = Local {
                f: rng.range(-2.0, 2.0) as f64,
                g: DVec2::new(rng.range(-3.0, 3.0) as f64, rng.range(-3.0, 3.0) as f64),
                h: Sym { a: rng.range(-4.0, 4.0) as f64, b: rng.range(-4.0, 4.0) as f64, c: rng.range(-4.0, 4.0) as f64 },
            };
            let Step::Go(d) = l.newton(1e-12) else { continue };
            let scale = 1.0 + d.length() * l.h.eigen().big();
            assert!((l.g + l.h.apply(d)).length() < 1e-9 * scale, "{l:?}: {d}");
            // and every other point of the paraboloid is higher (bowl), lower (dome) or
            // either (saddle): the flat point is its only stationary point
            let e = 1e-3;
            for k in 0..8 {
                let u = DVec2::from_angle(k as f64 * 0.785) * e;
                let change = l.model(d + u) - l.model(d);
                assert!((change - 0.5 * u.dot(l.h.apply(u))).abs() < 1e-9, "{change}");
            }
        }
    }

    /// The scene as a thing to try leaves it, once the point has glided there.
    fn tried(k: usize) -> Hessian {
        let mut s = Hessian::new();
        s.set_up(k);
        s.follow(10.0);
        s
    }

    /// The landscape's tour: it stops at a minimum, a saddle point and a maximum, where
    /// the gradient is 0 and the eigenvalues are those the caption gives; the card's
    /// verdict at each says so; and the tour ends at the maximum, after which nothing
    /// moves.
    #[test]
    fn the_tour_tells_a_bowl_a_dome_and_a_saddle_apart() {
        let mut s = tried(0);
        let tour = s.tour.expect("a tour");
        let expected = [(Kind::Bowl, 1.38, 0.76, "A minimum."), (Kind::Saddle, 0.80, -1.28, "A saddle point."), (Kind::Dome, -0.76, -1.38, "A maximum.")];
        for (stop, (kind, l1, l2, head)) in tour.stops.iter().zip(expected) {
            let a = At::work_out(&s.f.expr, *stop, s.map, s.h, s.slope);
            assert!(a.stationary && a.kind == kind, "{stop}: {a:?}");
            assert!((a.eig.l1 - l1).abs() < 0.005 && (a.eig.l2 - l2).abs() < 0.005, "{stop}: {:?}", a.eig);
            assert_eq!(verdict(a.kind, a.stationary).0, head);
        }
        // the point stays at each stop for a while, and between them moves smoothly
        let mut t = 0.0;
        let mut at_stop = [0.0f32; 3];
        let mut last = tour.from;
        let step = 1.0 / 60.0;
        let mut moving = s.tour;
        while let Some(mut tour) = moving {
            tour.t = t;
            let (q, done) = tour.at();
            assert!((q - last).length() < 0.05, "a jump at {t} s: {last} to {q}");
            for (k, stop) in tour.stops.iter().enumerate() {
                if (q - *stop).length() < 1e-12 {
                    at_stop[k] += step;
                }
            }
            last = q;
            t += step;
            moving = (!done).then_some(tour);
        }
        assert!(at_stop[0] > 2.9 && at_stop[1] > 2.9, "{at_stop:?}");
        assert!((last - tour.stops[2]).length() < 1e-12);
        // as the scene plays it: the tour comes to an end, at the maximum, and at rest
        s.tour = Some(Tour { t: 100.0, ..tour });
        let (q, done) = s.tour.unwrap().at();
        assert!(done && q == tour.stops[2]);
    }

    /// On a quadratic the paraboloid is the function itself, and Newton's first step
    /// lands on the minimum; gradient descent at its best step size for this bowl
    /// needs the number of steps the caption gives.
    #[test]
    fn newton_solves_a_quadratic_in_one_step() {
        let mut s = tried(QUADRATIC_TRY);
        let a = s.at().clone();
        // the paraboloid fitted anywhere is the function
        for q in [DVec2::new(2.0, -1.0), DVec2::new(-2.4, 0.3), DVec2::ZERO] {
            let d = q - a.p;
            assert!((a.local.model(d) - value(&s.f.expr, q.x, q.y)).abs() < 1e-12);
        }
        assert!(a.kind == Kind::Bowl && !a.stationary);
        assert!(a.newton.steps_to_arrive() == Some(1) && a.newton.limit.unwrap().length() < 1e-14 && (a.newton.pts[1]).length() < 1e-14, "{:?}", a.newton);
        assert!((a.eig.l1 - 4.56).abs() < 0.005 && (a.eig.l2 - 0.44).abs() < 0.005 && (s.h - 2.0 / (a.eig.l1 + a.eig.l2)).abs() < 1e-12);
        assert_eq!(a.descent.steps_to_arrive(), Some(76));
        assert!(a.descent.limit.unwrap().length() < 1e-12);
        // along each eigenvector gradient descent multiplies the distance by 1 − hλ: at
        // h = 0.4 both by 0.82 in size, and no other h makes the larger of the two smaller
        let worst = |h: f64| (1.0 - h * a.eig.l1).abs().max((1.0 - h * a.eig.l2).abs());
        assert!(((1.0 - s.h * a.eig.l1).abs() - 0.82).abs() < 0.005 && ((1.0 - s.h * a.eig.l2).abs() - 0.82).abs() < 0.005);
        for k in 0..=200 {
            let h = 0.3 + 0.001 * k as f64;
            assert!(worst(h) >= worst(s.h) - 1e-12, "{h}");
        }
        assert!((a.eig.l1 / a.eig.l2 - 10.0).abs() < 0.5);
    }

    /// The narrow curved valley: its curvatures at the minimum; gradient descent never
    /// settles with a step above 2/λ₁, and needs 595 steps at 0.06; Newton needs 9, its
    /// last distances being each about the square of the one before.
    #[test]
    fn in_a_narrow_valley_newton_needs_9_steps_and_gradient_descent_595() {
        let mut s = tried(VALLEY_TRY);
        let e = Local::at(&s.f.expr, DVec2::ONE).h.eigen();
        assert!((e.l1 - 31.6).abs() < 0.05 && (e.l2 - 0.38).abs() < 0.005 && (e.l1 / e.l2 - 83.0).abs() < 0.5, "{e:?}");
        assert!((2.0 / e.l1 - 0.063).abs() < 0.0005);
        let a = s.at().clone();
        assert!((s.h - 0.06).abs() < 1e-12);
        assert!(a.descent.steps_to_arrive() == Some(595) && (a.descent.limit.unwrap() - DVec2::ONE).length() < 1e-12);
        for h in [0.064, 0.066, 0.07, 0.08, 0.1] {
            let run = descent_run(&s.f.expr, a.p, h, s.map, 1e-9);
            assert!(run.limit.is_none(), "h = {h}: {:?}", run.end);
        }
        assert!(a.newton.steps_to_arrive() == Some(9) && (a.newton.limit.unwrap() - DVec2::ONE).length() < 1e-14);
        let errs: Vec<f64> = a.newton.pts.iter().map(|q| (*q - DVec2::ONE).length()).collect();
        let tail = &errs[6..10];
        for (got, want) in tail.iter().zip([0.074, 0.0042, 1.9e-5, 2.9e-10]) {
            assert!((got / want - 1.0).abs() < 0.03, "{tail:?}");
        }
        for w in tail.windows(2) {
            let c = w[1] / (w[0] * w[0]);
            assert!((0.7..1.1).contains(&c), "{tail:?}: {c}");
        }
        // the card writes the same
        assert_eq!(distances_text(&a.newton, 4), r"$\ldots,\ 0.074,\ 4.2e-3,\ 1.9e-5,\ 2.9e-10$");
    }

    /// Two wells: the paraboloid at the point is a saddle; Newton's run goes to the pass
    /// in 3 steps, gradient descent into the well at (1, 0); and where f_xx = 0, H is
    /// singular and, beside it, the step is longer than the map.
    #[test]
    fn newton_runs_to_the_saddle_and_gradient_descent_to_a_well() {
        let mut s = tried(WELLS_TRY);
        let a = s.at().clone();
        assert!(a.kind == Kind::Saddle && !a.stationary);
        assert!(a.newton.steps_to_arrive() == Some(3) && a.newton.limit.unwrap().length() < 1e-14 && a.newton_kind == Some(Kind::Saddle), "{:?}", a.newton);
        assert!((a.descent.limit.unwrap() - DVec2::X).length() < 1e-12 && a.descent_kind == Some(Kind::Bowl));
        let x0 = 1.0 / 3f64.sqrt();
        assert!((x0 - 0.58).abs() < 0.005);
        let l = Local::at(&s.f.expr, DVec2::new(x0, a.p.y));
        assert!(Kind::of(&l.h.eigen()) == Kind::Trough && l.newton(1e-9) == Step::Singular, "{l:?}");
        for x in [0.55, 0.56, 0.57, 0.575, 0.58, 0.59, 0.6] {
            let Step::Go(d) = Local::at(&s.f.expr, DVec2::new(x, a.p.y)).newton(1e-9) else { panic!("{x}") };
            assert!(d.length() > 2.0 * s.map.e, "{x}: {d}");
        }
    }

    /// A flat direction: at the origin the gradient is 0 and the eigenvalues are 2 and 0,
    /// so the test says nothing; the point is a minimum here, and a saddle for x² − y⁴,
    /// whose H there is the same; and Newton's run back from anywhere has its distance
    /// multiplied by 2/3 at each step after the first.
    #[test]
    fn a_flat_direction_leaves_the_test_silent() {
        let mut s = tried(FLAT_TRY);
        let a = s.at().clone();
        assert!(a.stationary && a.p == DVec2::ZERO && a.kind == Kind::Trough && a.eig.l1 == 2.0 && a.eig.l2 == 0.0, "{a:?}");
        assert_eq!(verdict(a.kind, a.stationary).0, "The test says nothing.");
        let other = Local::at(&parse("x^2 - y^4"), DVec2::ZERO);
        assert_eq!(other.h, a.local.h);
        for y in [-1.0, -0.1, 0.01, 0.5] {
            assert!(value(&s.f.expr, 0.0, y) > 0.0 && value(&parse("x^2 - y^4"), 0.0, y) < 0.0);
        }
        for start in [DVec2::new(0.9, -1.0), DVec2::new(-1.1, 0.7), DVec2::new(0.2, 1.2)] {
            let run = newton_run(&s.f.expr, start, s.map, 1e-9 * s.slope);
            assert!(run.end == End::Arrived && run.limit.unwrap().length() < ARRIVE, "{start}: {:?}", run.end);
            let d: Vec<f64> = run.pts.iter().map(|q| q.length()).collect();
            assert!(run.pts[1].x.abs() < 1e-15);
            for w in d[1..].windows(2) {
                assert!((w[1] / w[0] - 2.0 / 3.0).abs() < 1e-9, "{start}: {d:?}");
            }
            // (the distances the card lists are those to the origin, near enough)
            assert!(run.blur < 1e-6);
        }
    }

    /// Every thing to try sets up the same scene whatever was done before.
    #[test]
    fn every_thing_to_try_sets_everything_it_needs() {
        for k in 0..TRIALS.len() {
            let fresh = tried(k);
            for before in 0..TRIALS.len() {
                let mut s = tried(before);
                s.place(DVec2::new(0.3, -0.2), false);
                s.h = 2.5;
                s.f.set("x y");
                (s.dirty, s.at, s.show_fit, s.show_runs) = (true, None, false, false);
                s.set_up(k);
                s.follow(10.0);
                assert!(s.f.text == fresh.f.text && s.map == fresh.map && s.h == fresh.h && s.squash == fresh.squash && s.show_fit && s.show_runs);
                assert_eq!((s.p, s.tour.map(|t| t.stops)), (fresh.p, fresh.tour.map(|t| t.stops)));
            }
        }
    }

    /// Degenerate functions: a plane (H = 0), a constant, no value anywhere, a pole;
    /// nothing worked out is a panic, and what is drawn from it stays finite or says
    /// none.
    #[test]
    fn degenerate_functions_are_told_honestly() {
        for (formula, kind, step) in [("x + 2 y", Kind::Flat, Step::Singular), ("3", Kind::Flat, Step::Here), ("x^2", Kind::Trough, Step::Here)] {
            let mut s = Hessian::new();
            s.f.set(formula);
            (s.dirty, s.at) = (true, None);
            s.place(DVec2::ZERO, false);
            let a = s.at().clone();
            assert!(a.kind == kind && a.step == step, "{formula}: {a:?}");
            assert!(model_levels(&a, s.map.e).iter().flatten().all(|q| q.is_finite()));
        }
        for formula in ["sqrt(-1 - x^2)", "1/(x^2 + y^2)", "1/x", "tan(3 x)", "exp(40 x)", "x^3 - 3 x y^2"] {
            let mut s = Hessian::new();
            s.f.set(formula);
            (s.dirty, s.at) = (true, None);
            for q in [DVec2::ZERO, DVec2::new(0.4, -0.7), DVec2::new(2.8, 2.8)] {
                s.place(q, false);
                let a = s.at().clone();
                assert!(s.heights.lo.is_finite() && s.heights.hi > s.heights.lo && s.heights.span.is_finite());
                assert!(model_levels(&a, s.map.e).iter().flatten().all(|q| q.is_finite()), "{formula} at {q}");
                let _ = (run_words(&a.newton, a.newton_kind), run_words(&a.descent, a.descent_kind), distances_text(&a.newton, 5));
            }
        }
    }

    /// The layouts of every window: the panels lie on the canvas and off one another,
    /// and the picture keeps a part of it.
    #[test]
    fn the_panels_fit_every_window() {
        for (w, h) in [(1080.0, 1000.0), (1400.0, 1080.0), (580.0, 700.0), (390.0, 800.0), (360.0, 640.0), (800.0, 390.0), (700.0, 500.0), (240.0, 180.0), (1.0, 1.0)] {
            for inset in [0.0, 52.0] {
                let view = Vec2::new(w, h);
                let lay = Layout::of(view, inset);
                let rects: Vec<(Vec2, Vec2)> = lay.card.into_iter().chain(lay.runs).chain(lay.panel.map(|m| (m.min, m.max))).collect();
                for (a, b) in &rects {
                    assert!(a.cmpge(Vec2::ZERO).all() && b.cmple(view + Vec2::splat(0.5)).all() && a.cmplt(*b).all(), "{view}: {rects:?}");
                }
                for (i, r) in rects.iter().enumerate() {
                    for q in &rects[i + 1..] {
                        assert!(r.1.x <= q.0.x || q.1.x <= r.0.x || r.1.y <= q.0.y || q.1.y <= r.0.y, "{view}: {r:?} and {q:?} overlap");
                    }
                }
                let (a, b) = lay.picture;
                assert!(b.y - a.y >= 0.0 && b.x - a.x >= 0.0 && a.is_finite() && b.is_finite(), "{view}: {:?}", lay.picture);
                if let Some(m) = lay.panel {
                    assert!(m.at.y + m.side + m.rows.height() <= m.max.y && m.at.x >= m.min.x && m.at.x + m.side <= m.max.x + 0.5);
                }
            }
        }
    }

    /// Every text the scene can put up is set as written: the verdicts, the run words,
    /// the captions.
    #[test]
    fn every_text_is_typeset_as_written() {
        let kinds = [Kind::Bowl, Kind::Dome, Kind::Saddle, Kind::Trough, Kind::Flat];
        let mut texts: Vec<String> = kinds.iter().flat_map(|k| [verdict(*k, true), verdict(*k, false)]).flat_map(|(a, b)| [a.to_string(), b.to_string()]).collect();
        texts.extend(TRIALS.iter().flat_map(|t| [t.label.to_string(), t.look_for.to_string()]));
        let run = Run { pts: vec![DVec2::ZERO; 3], end: End::Singular, limit: None, blur: 0.0 };
        texts.push(run_words(&run, None));
        for text in &texts {
            let found = crate::typeset::inspect(text);
            assert!(found.error.is_none() && found.words.is_empty(), "{found:?} in {text}");
        }
        assert_eq!(short(-0.004), "−4.0e-3".replace('−', "-"));
        assert_eq!(short(1234.5), "1234.50");
        assert_eq!(short(-1.5), "−1.50");
    }
}

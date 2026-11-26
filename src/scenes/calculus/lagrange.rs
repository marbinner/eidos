//! Lagrange multipliers: the highest point of a path across a hillside. Where the path
//! crosses a contour line of the hill, one way along it is uphill; so at its highest point
//! it touches a contour line instead of crossing one, and the gradient of the height is at
//! right angles to the path there, like the gradient of the function the path is drawn by.

use super::*;
use crate::math::expr::Func;
use crate::scenes::stats::plot::{self, Plot};
use glam::{DMat2, DVec2};
use std::f64::consts::PI as PI64;

pub const DEF: SceneDef = SceneDef {
    id: "lagrange",
    title: "Lagrange multipliers",
    group: "Calculus",
    blurb: "The highest point of a path on a hillside: where the path runs along a contour, and what the multiplier is the price of.",
    make: |_| Box::new(Lagrange::new()),
};

// ----- the height f, in double precision, with its gradient ---------------------------

/// A number with its two partial derivatives, in double precision: a formula worked out
/// on it gives the formula's value and its exact gradient (as `Dual` does in single
/// precision). The tests hold it to finite differences.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
struct D64 {
    v: f64,
    dx: f64,
    dy: f64,
}

fn call64(f: Func, x: f64) -> f64 {
    match f {
        Func::Sin => x.sin(),
        Func::Cos => x.cos(),
        Func::Tan => x.tan(),
        Func::Asin => x.asin(),
        Func::Acos => x.acos(),
        Func::Atan => x.atan(),
        Func::Sinh => x.sinh(),
        Func::Cosh => x.cosh(),
        Func::Tanh => x.tanh(),
        Func::Exp => x.exp(),
        Func::Ln => x.ln(),
        Func::Log10 => x.log10(),
        Func::Log2 => x.log2(),
        Func::Sqrt => x.sqrt(),
        Func::Abs => x.abs(),
        Func::Sign => {
            if x > 0.0 {
                1.0
            } else if x < 0.0 {
                -1.0
            } else {
                0.0
            }
        }
        Func::Floor => x.floor(),
        Func::Ceil => x.ceil(),
        Func::Relu => x.max(0.0),
        Func::Sigmoid => 1.0 / (1.0 + (-x).exp()),
        Func::Gauss => (-x * x).exp(),
    }
}

/// The derivative of `f` at `x`, given `y = f(x)`.
fn deriv64(f: Func, x: f64, y: f64) -> f64 {
    match f {
        Func::Sin => x.cos(),
        Func::Cos => -x.sin(),
        Func::Tan => 1.0 + y * y,
        Func::Asin => 1.0 / (1.0 - x * x).sqrt(),
        Func::Acos => -1.0 / (1.0 - x * x).sqrt(),
        Func::Atan => 1.0 / (1.0 + x * x),
        Func::Sinh => x.cosh(),
        Func::Cosh => x.sinh(),
        Func::Tanh => 1.0 - y * y,
        Func::Exp => y,
        Func::Ln => 1.0 / x,
        Func::Log10 => 1.0 / (x * std::f64::consts::LN_10),
        Func::Log2 => 1.0 / (x * std::f64::consts::LN_2),
        Func::Sqrt => 0.5 / y,
        Func::Abs => {
            if x >= 0.0 {
                1.0
            } else {
                -1.0
            }
        }
        Func::Sign | Func::Floor | Func::Ceil => 0.0,
        Func::Relu => {
            if x > 0.0 {
                1.0
            } else {
                0.0
            }
        }
        Func::Sigmoid => y * (1.0 - y),
        Func::Gauss => -2.0 * x * y,
    }
}

impl D64 {
    /// The slope `k` along a direction in which the argument changes at rate `d`: nothing
    /// where it does not change, however steep `k` is (as `Dual` does it).
    fn along(k: f64, d: f64) -> f64 {
        if d == 0.0 {
            0.0
        } else {
            k * d
        }
    }

    fn grad(self) -> DVec2 {
        DVec2::new(self.dx, self.dy)
    }
}

impl Num for D64 {
    fn cst(v: f32) -> Self {
        D64 { v: v as f64, dx: 0.0, dy: 0.0 }
    }
    fn value(self) -> f32 {
        self.v as f32
    }
    fn add(self, o: Self) -> Self {
        D64 { v: self.v + o.v, dx: self.dx + o.dx, dy: self.dy + o.dy }
    }
    fn sub(self, o: Self) -> Self {
        D64 { v: self.v - o.v, dx: self.dx - o.dx, dy: self.dy - o.dy }
    }
    fn mul(self, o: Self) -> Self {
        D64 { v: self.v * o.v, dx: self.dx * o.v + self.v * o.dx, dy: self.dy * o.v + self.v * o.dy }
    }
    fn div(self, o: Self) -> Self {
        let inv = 1.0 / o.v;
        let q = self.v * inv;
        D64 { v: q, dx: (self.dx - q * o.dx) * inv, dy: (self.dy - q * o.dy) * inv }
    }
    fn neg(self) -> Self {
        D64 { v: -self.v, dx: -self.dx, dy: -self.dy }
    }
    fn call(self, f: Func) -> Self {
        let y = call64(f, self.v);
        let d = deriv64(f, self.v, y);
        D64 { v: y, dx: D64::along(d, self.dx), dy: D64::along(d, self.dy) }
    }
    /// d(a^b) = b a^(b-1) da + a^b ln(a) db, each term only where its factor varies.
    fn pow(self, o: Self) -> Self {
        let v = self.v.powf(o.v);
        let by_base = if (self.dx != 0.0 || self.dy != 0.0) && o.v != 0.0 { o.v * self.v.powf(o.v - 1.0) } else { 0.0 };
        let by_exponent = if (o.dx != 0.0 || o.dy != 0.0) && v != 0.0 { v * self.v.ln() } else { 0.0 };
        D64 { v, dx: D64::along(by_base, self.dx) + D64::along(by_exponent, o.dx), dy: D64::along(by_base, self.dy) + D64::along(by_exponent, o.dy) }
    }
    fn atan2(self, o: Self) -> Self {
        let r2 = self.v * self.v + o.v * o.v;
        let (ka, kb) = if r2 > 0.0 { (o.v / r2, -self.v / r2) } else { (0.0, 0.0) };
        D64 { v: self.v.atan2(o.v), dx: ka * self.dx + kb * o.dx, dy: ka * self.dy + kb * o.dy }
    }
    fn min(self, o: Self) -> Self {
        if self.v <= o.v || o.v.is_nan() {
            self
        } else {
            o
        }
    }
    fn max(self, o: Self) -> Self {
        if self.v >= o.v || o.v.is_nan() {
            self
        } else {
            o
        }
    }
}

/// The variables a formula of the scene may use.
const VARS: [&str; 2] = ["x", "y"];

/// f and its gradient at p.
fn height(expr: &Expr, p: DVec2) -> D64 {
    expr.eval_num(&[D64 { v: p.x, dx: 1.0, dy: 0.0 }, D64 { v: p.y, dx: 0.0, dy: 1.0 }])
}

/// An angle in (−π, π].
fn wrap(t: f64) -> f64 {
    let w = (t + PI64).rem_euclid(2.0 * PI64) - PI64;
    if w <= -PI64 {
        w + 2.0 * PI64
    } else {
        w
    }
}

fn smooth(t: f64) -> f64 {
    let t = t.clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

// ----- the path: g(x, y) = c ------------------------------------------------------------

/// Which curve the point must stay on.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Kind {
    /// g = x² + y²: a circle of radius √c.
    Circle,
    /// g = x²/4 + y²: an ellipse twice as wide as it is tall.
    Ellipse,
    /// g = x + 2y: a straight line.
    Line,
}

const KINDS: [Kind; 3] = [Kind::Circle, Kind::Ellipse, Kind::Line];

/// The line's g is NORMAL · (x, y), so NORMAL is its gradient.
const NORMAL: DVec2 = DVec2::new(1.0, 2.0);

/// How far the line is drawn either way from its point nearest 0.
const REACH: f64 = 4.5;

/// Where round a closed curve its c handle sits (the direction from 0).
const HANDLE_ANGLE: f64 = 160.0 * PI64 / 180.0;

/// Where along the line its c handle sits.
const HANDLE_S: f64 = 2.2;

impl Kind {
    fn g(self, p: DVec2) -> f64 {
        match self {
            Kind::Circle => p.length_squared(),
            Kind::Ellipse => 0.25 * p.x * p.x + p.y * p.y,
            Kind::Line => NORMAL.dot(p),
        }
    }

    fn grad(self, p: DVec2) -> DVec2 {
        match self {
            Kind::Circle => 2.0 * p,
            Kind::Ellipse => DVec2::new(0.5 * p.x, 2.0 * p.y),
            Kind::Line => NORMAL,
        }
    }

    fn closed(self) -> bool {
        self != Kind::Line
    }

    /// The values c may take. (For the circle and the ellipse c stays above 0: at 0 the
    /// curve is a single point, where ∇g = 0 and the method says nothing.)
    fn c_range(self) -> [f64; 2] {
        match self {
            Kind::Circle => [0.04, 5.0],
            Kind::Ellipse => [0.04, 1.5],
            Kind::Line => [-5.0, 5.0],
        }
    }

    fn clamp_c(self, c: f64) -> f64 {
        let [lo, hi] = self.c_range();
        if c.is_finite() {
            c.max(lo).min(hi)
        } else {
            lo.max(0.0).min(hi)
        }
    }

    /// The place the curve's parameter runs over: the direction from 0 for a closed
    /// curve, the distance along the line from its point nearest 0 for the line.
    fn t_range(self) -> [f64; 2] {
        if self.closed() {
            [-PI64, PI64]
        } else {
            [-REACH, REACH]
        }
    }

    /// The unit arrow along the line, the way its parameter grows: ∇g turned a quarter
    /// turn anticlockwise, as for the closed curves.
    fn line_tangent() -> DVec2 {
        NORMAL.perp() / NORMAL.length()
    }

    /// The point of the curve g = c at the place t.
    fn at(self, c: f64, t: f64) -> DVec2 {
        match self {
            Kind::Circle => DVec2::from_angle(t) * c.max(0.0).sqrt(),
            Kind::Ellipse => {
                let u = DVec2::from_angle(t);
                u * (c.max(0.0) / (0.25 * u.x * u.x + u.y * u.y)).sqrt()
            }
            Kind::Line => NORMAL * (c / NORMAL.length_squared()) + Kind::line_tangent() * t,
        }
    }

    /// The place on the curve g = c that the point q stands for: the same direction from
    /// 0, or the foot of the perpendicular from q onto the line.
    fn place_of(self, c: f64, q: DVec2) -> f64 {
        if self.closed() {
            if q.length_squared() > 0.0 {
                q.y.atan2(q.x)
            } else {
                0.0
            }
        } else {
            let foot = NORMAL * (c / NORMAL.length_squared());
            (q - foot).dot(Kind::line_tangent()).max(-REACH).min(REACH)
        }
    }

    /// The place of sample `i` of `n` along the curve (the last one of a closed curve
    /// stops short of going all the way round).
    fn sample_t(self, i: usize, n: usize) -> f64 {
        let [a, b] = self.t_range();
        let steps = if self.closed() { n } else { n.max(2) - 1 };
        a + (b - a) * i as f64 / steps as f64
    }

    /// Where the c handle sits.
    fn handle_t(self) -> f64 {
        if self.closed() {
            HANDLE_ANGLE
        } else {
            HANDLE_S
        }
    }

    /// The c that puts the curve through the handle dragged to q: along its own direction
    /// from 0 for a closed curve, anywhere for the line.
    fn c_through(self, q: DVec2) -> f64 {
        let c = if self.closed() {
            let u = DVec2::from_angle(HANDLE_ANGLE);
            self.g(u * q.dot(u).max(0.05))
        } else {
            self.g(q)
        };
        self.clamp_c(c)
    }

    fn name(self) -> &'static str {
        match self {
            Kind::Circle => "circle",
            Kind::Ellipse => "ellipse",
            Kind::Line => "line",
        }
    }

    /// g, as a formula.
    fn g_text(self) -> &'static str {
        match self {
            Kind::Circle => "x^2 + y^2",
            Kind::Ellipse => "x^2/4 + y^2",
            Kind::Line => "x + 2y",
        }
    }

    /// The c the scene gives a curve of this kind when there is no other to go by.
    fn usual_c(self) -> f64 {
        match self {
            Kind::Circle | Kind::Ellipse => 1.0,
            Kind::Line => 2.0,
        }
    }
}

// ----- what f does along the path -------------------------------------------------------

/// What f does at one place of the curve g = c.
#[derive(Clone, Copy, Debug, PartialEq)]
struct Here {
    t: f64,
    p: DVec2,
    f: f64,
    grad_f: DVec2,
    grad_g: DVec2,
    /// The multiple of ∇g in ∇f: ∇f·∇g/|∇g|². At a stationary point all of ∇f is that.
    lambda: f64,
    /// The part of ∇f along the curve, ∇f·T, for T the unit arrow along it (the way t
    /// grows): how fast f changes as the point moves along the curve.
    slope: f64,
}

impl Here {
    /// The unit arrow along the curve.
    fn tangent(&self) -> DVec2 {
        self.grad_g.perp() / self.grad_g.length()
    }

    /// The same place, one turn further round (or back) a closed curve.
    fn turned(self, turns: f64) -> Here {
        Here { t: self.t + 2.0 * PI64 * turns, ..self }
    }

    /// Is ∇f in line with ∇g here (no part of it along the curve, near enough)?
    fn stationary(&self) -> bool {
        self.slope.abs() <= 5e-4 * (1.0 + self.grad_f.length())
    }
}

fn here(expr: &Expr, kind: Kind, c: f64, t: f64) -> Here {
    let p = kind.at(c, t);
    let d = height(expr, p);
    let gf = d.grad();
    let gg = kind.grad(p);
    let n2 = gg.length_squared();
    let (lambda, slope) = if n2 > 0.0 { (gf.dot(gg) / n2, gf.dot(gg.perp()) / n2.sqrt()) } else { (f64::NAN, f64::NAN) };
    Here { t, p, f: d.v, grad_f: gf, grad_g: gg, lambda, slope }
}

/// The place between `lo` and `hi` (whose slopes have opposite signs) where the slope
/// along the curve is 0, by halving.
fn refine(expr: &Expr, kind: Kind, c: f64, lo: Here, hi: Here) -> Here {
    let (mut a, mut b) = (lo, hi);
    for _ in 0..90 {
        let t = 0.5 * (a.t + b.t);
        if !(t > a.t && t < b.t) {
            break;
        }
        let m = here(expr, kind, c, t);
        if m.slope == 0.0 || !m.slope.is_finite() {
            return if m.slope == 0.0 { m } else { a };
        }
        if (m.slope > 0.0) == (a.slope > 0.0) {
            a = m;
        } else {
            b = m;
        }
    }
    if a.slope.abs() <= b.slope.abs() {
        a
    } else {
        b
    }
}

/// A place where f stops changing along the curve.
#[derive(Clone, Copy, Debug)]
struct Stationary {
    at: Here,
    /// A top (f rises to it from before and falls after); else a bottom.
    top: bool,
}

/// How many places along the curve f is looked at.
const SAMPLES: usize = 480;

/// What f does all along the curve g = c.
#[derive(Clone, Debug, Default)]
struct Along {
    samples: Vec<Here>,
    stationary: Vec<Stationary>,
    /// f is the same all along the curve (every point is stationary).
    flat: bool,
    /// The highest point of the curve; none if f has no value on it, or on a line if f
    /// is still rising at an end of the stretch drawn.
    highest: Option<Here>,
    /// How many tops are as high as the highest.
    ties: usize,
}

/// Is f the same all along these samples?
fn is_flat(samples: &[Here]) -> bool {
    let finite = samples.iter().filter(|h| h.f.is_finite() && h.slope.is_finite());
    let (mut most, mut steep, mut count) = (0.0f64, 0.0f64, 0);
    for h in finite {
        most = most.max(h.slope.abs());
        steep = steep.max(h.grad_f.length());
        count += 1;
    }
    count == samples.len() && count > 0 && most <= 1e-9 * (1.0 + steep)
}

/// The highest of `samples` (of the curve g = c), refined to where the slope along the
/// curve is 0 beside it.
fn highest_of(expr: &Expr, kind: Kind, c: f64, samples: &[Here], flat: bool) -> Option<Here> {
    let n = samples.len();
    let (i, best) = samples.iter().enumerate().filter(|(_, h)| h.f.is_finite()).max_by(|a, b| a.1.f.total_cmp(&b.1.f))?;
    if flat {
        return Some(*best);
    }
    // (on a line, a highest sample at an end of the stretch drawn is not a highest point:
    // f may go on rising beyond it)
    if !kind.closed() && (i == 0 || i == n - 1) {
        return None;
    }
    let prev = if i > 0 { samples[i - 1] } else { samples[n - 1].turned(-1.0) };
    let next = if i + 1 < n { samples[i + 1] } else { samples[0].turned(1.0) };
    let found = if best.slope > 0.0 && next.slope < 0.0 {
        Some(refine(expr, kind, c, *best, next))
    } else if best.slope < 0.0 && prev.slope > 0.0 {
        Some(refine(expr, kind, c, prev, *best))
    } else {
        None
    };
    let at = found.filter(|h| h.f >= best.f).unwrap_or(*best);
    Some(if kind.closed() { Here { t: wrap(at.t), ..at } } else { at })
}

fn along(expr: &Expr, kind: Kind, c: f64, n: usize) -> Along {
    let samples: Vec<Here> = (0..n).map(|i| here(expr, kind, c, kind.sample_t(i, n))).collect();
    let flat = is_flat(&samples);
    let mut stationary = Vec::new();
    if !flat {
        // every change of sign of the slope along the curve (for a closed curve, round
        // past the first sample again), each refined; a corner of f, where the slope
        // jumps across 0 without passing through it, is left out
        let count = if kind.closed() { n + 1 } else { n };
        let sample = |i: usize| if i < n { samples[i] } else { samples[i - n].turned(1.0) };
        let mut last: Option<Here> = None;
        for i in 0..count {
            let s = sample(i);
            if !s.slope.is_finite() {
                last = None;
                continue;
            }
            if s.slope == 0.0 {
                continue;
            }
            if let Some(before) = last.filter(|b| (b.slope > 0.0) != (s.slope > 0.0)) {
                let root = refine(expr, kind, c, before, s);
                if root.slope.abs() <= 1e-7 * (1.0 + root.grad_f.length()) && stationary.len() < 64 {
                    let at = if kind.closed() { Here { t: wrap(root.t), ..root } } else { root };
                    stationary.push(Stationary { at, top: before.slope > 0.0 });
                }
            }
            last = Some(s);
        }
    }
    let highest = highest_of(expr, kind, c, &samples, flat);
    let ties = match highest {
        Some(h) => stationary.iter().filter(|s| s.top && (s.at.f - h.f).abs() <= 1e-9 * (1.0 + h.f.abs())).count().max(1),
        None => 0,
    };
    Along { samples, stationary, flat, highest, ties }
}

// ----- the best point, on the curve or inside it ----------------------------------------

/// The tops of f's hills that lie in the region a closed curve may enclose (strict local
/// highest points, where ∇f = 0), with their heights.
fn hilltops(expr: &Expr) -> Vec<(DVec2, f64)> {
    const N: usize = 105;
    const BOX: f64 = 2.6;
    let at = |i: usize| -BOX + 2.0 * BOX * i as f64 / (N - 1) as f64;
    // (the grid in single precision, all at once: it only says where to start climbing)
    let (mut xs, mut ys) = (Vec::with_capacity(N * N), Vec::with_capacity(N * N));
    for j in 0..N {
        for i in 0..N {
            xs.push(at(i) as f32);
            ys.push(at(j) as f32);
        }
    }
    let mut single = vec![0.0f32; N * N];
    expr.eval_batch(&[&xs, &ys], &mut single, &mut Batch::default());
    let values: Vec<f64> = single.iter().map(|v| *v as f64).collect();
    let mut tops: Vec<(DVec2, f64)> = Vec::new();
    for j in 1..N - 1 {
        for i in 1..N - 1 {
            let v = values[j * N + i];
            let higher = v.is_finite() && (-1i32..=1).all(|dj| (-1i32..=1).all(|di| (di == 0 && dj == 0) || v > values[(j as i32 + dj) as usize * N + (i as i32 + di) as usize]));
            if !higher {
                continue;
            }
            if let Some(p) = climb(expr, DVec2::new(at(i), at(j))) {
                if p.abs().max_element() <= BOX && tops.iter().all(|(q, _)| q.distance(p) > 1e-6) && tops.len() < 32 {
                    tops.push((p, height(expr, p).v));
                }
            }
        }
    }
    tops
}

/// The top of the hill near `p0`: where ∇f = 0 and f curves down every way, found by
/// Newton's steps on the gradient (the second derivatives by differences of the exact
/// gradient). None if the steps do not settle on such a point.
fn climb(expr: &Expr, p0: DVec2) -> Option<DVec2> {
    let hessian = |p: DVec2| {
        let h = 1e-5;
        let gx = (height(expr, p + DVec2::X * h).grad() - height(expr, p - DVec2::X * h).grad()) / (2.0 * h);
        let gy = (height(expr, p + DVec2::Y * h).grad() - height(expr, p - DVec2::Y * h).grad()) / (2.0 * h);
        let off = 0.5 * (gx.y + gy.x);
        DMat2::from_cols(DVec2::new(gx.x, off), DVec2::new(off, gy.y))
    };
    let mut p = p0;
    for _ in 0..60 {
        let d = height(expr, p);
        let g = d.grad();
        if !(d.v.is_finite() && g.is_finite()) {
            return None;
        }
        let hess = hessian(p);
        let down = hess.determinant() > 0.0 && hess.x_axis.x < 0.0;
        if g.length() <= 1e-11 * (1.0 + d.v.abs()) {
            return down.then_some(p);
        }
        let step = if down { -(hess.inverse() * g) } else { g * 0.1 };
        let step = if step.length() > 0.25 { step * (0.25 / step.length()) } else { step };
        if !step.is_finite() {
            return None;
        }
        p += step;
    }
    let d = height(expr, p);
    let hess = hessian(p);
    (d.grad().length() <= 1e-8 * (1.0 + d.v.abs()) && hess.determinant() > 0.0 && hess.x_axis.x < 0.0).then_some(p)
}

/// The best point: the highest f on the curve g = c, or in the region g ≤ c.
#[derive(Clone, Copy, Debug, PartialEq)]
enum Best {
    /// On the curve, at a top; `ties` tops are as high.
    Edge { at: Here, ties: usize },
    /// Inside the region, at the top of a hill, where ∇f = 0: λ = 0 there.
    Inside { p: DVec2, f: f64 },
    /// f is the same all along the curve: every point of it is a highest point.
    Everywhere { f: f64 },
}

impl Best {
    fn f(&self) -> f64 {
        match self {
            Best::Edge { at, .. } => at.f,
            Best::Inside { f, .. } | Best::Everywhere { f } => *f,
        }
    }
}

/// The best point at c, from what f does along the curve and the hilltops.
fn best_of(along: &Along, tops: &[(DVec2, f64)], kind: Kind, c: f64, inside: bool) -> Option<Best> {
    let edge = if along.flat {
        along.highest.map(|h| Best::Everywhere { f: h.f })
    } else {
        along.highest.map(|at| Best::Edge { at, ties: along.ties })
    };
    if inside && kind.closed() {
        let hill = tops.iter().filter(|(p, v)| kind.g(*p) < c && v.is_finite()).max_by(|a, b| a.1.total_cmp(&b.1));
        if let Some(&(p, f)) = hill {
            if edge.is_none_or(|e| f > e.f() + 1e-12 * (1.0 + f.abs())) {
                return Some(Best::Inside { p, f });
            }
        }
    }
    edge
}

/// The highest value at c alone (for the chart of it against c, and its slope).
fn best_value(expr: &Expr, tops: &[(DVec2, f64)], kind: Kind, c: f64, inside: bool, n: usize) -> Option<f64> {
    let samples: Vec<Here> = (0..n).map(|i| here(expr, kind, c, kind.sample_t(i, n))).collect();
    let flat = is_flat(&samples);
    let on = highest_of(expr, kind, c, &samples, flat).map(|h| h.f);
    let hill = if inside && kind.closed() { tops.iter().filter(|(p, v)| kind.g(*p) < c && v.is_finite()).map(|t| t.1).fold(None, |m: Option<f64>, v| Some(m.map_or(v, |m| m.max(v)))) } else { None };
    match (on, hill) {
        (Some(a), Some(b)) => Some(a.max(b)),
        (a, b) => a.or(b),
    }
}

/// The highest value f*(c) against c, over the whole range of c: for the chart.
#[derive(Clone, Debug, Default)]
struct Profile {
    cs: Vec<f64>,
    best: Vec<f64>,
    /// On the curve alone (the same as `best` unless the region counts).
    on_curve: Vec<f64>,
}

const PROFILE_N: usize = 61;

/// The c of the chart's `i`th of `n` points: evenly in √c for a closed curve, so that its
/// size grows evenly, and evenly in c for the line.
fn profile_c(kind: Kind, i: usize, n: usize) -> f64 {
    let [lo, hi] = kind.c_range();
    let s = i as f64 / (n - 1) as f64;
    if kind.closed() {
        let (a, b) = (lo.sqrt(), hi.sqrt());
        (a + (b - a) * s).powi(2)
    } else {
        lo + (hi - lo) * s
    }
}

fn profile(expr: &Expr, tops: &[(DVec2, f64)], kind: Kind, inside: bool) -> Profile {
    let mut out = Profile::default();
    for i in 0..PROFILE_N {
        let c = profile_c(kind, i, PROFILE_N);
        let on = best_value(expr, tops, kind, c, false, 180).unwrap_or(f64::NAN);
        let hill = if inside && kind.closed() { tops.iter().filter(|(p, v)| kind.g(*p) < c && v.is_finite()).map(|t| t.1).fold(f64::NAN, f64::max) } else { f64::NAN };
        let best = if on.is_finite() || hill.is_finite() { on.max(hill) } else { f64::NAN };
        out.cs.push(c);
        out.on_curve.push(on);
        out.best.push(best);
    }
    out
}

/// f* at `c2`, a hair from the c that `along` is for: each top followed from where it
/// is at c, by halving a small bracket round it (a top moves little for a small change of
/// c), the highest of them taken, and the hilltops inside if the region counts. None if a
/// top cannot be followed so (it has gone, or the highest point is not at a top).
fn best_near(expr: &Expr, tops: &[(DVec2, f64)], kind: Kind, along: &Along, c2: f64, inside: bool) -> Option<f64> {
    let highest = along.highest?;
    if along.flat {
        return None;
    }
    let [a, b] = kind.t_range();
    let width = 4.0 * (b - a) / SAMPLES as f64;
    let mut best = f64::NEG_INFINITY;
    let mut found_highest = false;
    for s in along.stationary.iter().filter(|s| s.top) {
        let (lo, hi) = (here(expr, kind, c2, s.at.t - width), here(expr, kind, c2, s.at.t + width));
        if !(lo.slope > 0.0 && hi.slope < 0.0) {
            return None;
        }
        best = best.max(refine(expr, kind, c2, lo, hi).f);
        found_highest |= (s.at.f - highest.f).abs() <= 1e-12 * (1.0 + highest.f.abs());
    }
    if !found_highest {
        return None;
    }
    if inside && kind.closed() {
        for (_, v) in tops.iter().filter(|(p, v)| kind.g(*p) < c2 && v.is_finite()) {
            best = best.max(*v);
        }
    }
    best.is_finite().then_some(best)
}

/// The slope of f*(c) at c, by a difference across a small step of c either side: from
/// the tops followed a step each way where that can be done, else from f* found afresh.
fn slope_of_best(expr: &Expr, tops: &[(DVec2, f64)], kind: Kind, c: f64, inside: bool, along: &Along) -> Option<f64> {
    let [lo, hi] = kind.c_range();
    let step = 1e-4 * (hi - lo);
    let (a, b) = ((c - step).max(lo), (c + step).min(hi));
    let near = best_near(expr, tops, kind, along, a, inside).zip(best_near(expr, tops, kind, along, b, inside));
    let (fa, fb) = match near {
        Some(pair) => pair,
        None => (best_value(expr, tops, kind, a, inside, SAMPLES)?, best_value(expr, tops, kind, b, inside, SAMPLES)?),
    };
    let s = (fb - fa) / (b - a);
    s.is_finite().then_some(s)
}

// ----- the contour map ------------------------------------------------------------------

/// The heights the map is coloured over, the spacing of its contour lines, and the places
/// where some of them are labelled with their heights: worked out once for each formula
/// from the part of the plane the view is fitted to, so that they stay put as the view
/// moves.
#[derive(Clone, Debug, Default)]
struct Levels {
    lo: f32,
    hi: f32,
    /// 0: no contour lines (f the same everywhere, or no value anywhere).
    step: f32,
    labels: Vec<(DVec2, f64)>,
}

/// The square the levels are worked out over, and the part of it labels go in.
const LEVEL_BOX: f32 = 2.4;
const LABEL_BOX: [f64; 2] = [2.15, 2.0];

fn levels(expr: &Expr) -> Levels {
    const N: usize = 97;
    let at = |i: usize| -LEVEL_BOX + 2.0 * LEVEL_BOX * i as f32 / (N - 1) as f32;
    let mut xs = Vec::with_capacity(N * N);
    let mut ys = Vec::with_capacity(N * N);
    for j in 0..N {
        for i in 0..N {
            xs.push(at(i));
            ys.push(at(j));
        }
    }
    let mut zs = vec![0.0f32; N * N];
    expr.eval_batch(&[&xs, &ys], &mut zs, &mut Batch::default());
    let mut finite: Vec<f32> = zs.iter().copied().filter(|z| z.is_finite()).collect();
    if finite.len() < 16 {
        return Levels::default();
    }
    finite.sort_by(f32::total_cmp);
    let (lo, hi) = (finite[finite.len() * 3 / 100], finite[finite.len() * 97 / 100]);
    // (no spread to speak of: f the same everywhere, or nearly)
    if hi - lo <= 1e-6 * lo.abs().max(1.0) || (hi - lo).is_nan() {
        return Levels { lo: lo - 0.5, hi: lo + 0.5, step: 0.0, labels: Vec::new() };
    }
    let step = plot::round_step(((hi - lo) / 12.0) as f64) as f32;
    // the levels, every other one if there are many; where each crosses the edges of the
    // grid's cells inside the label box, all found in one pass over the edges
    let (first, last) = ((lo / step).ceil() as i64, (hi / step).floor() as i64);
    let every = if last - first > 9 { 2 } else { 1 };
    let count = (last - first + 1).clamp(0, 64) as usize;
    let mut candidates: Vec<Vec<(f64, DVec2)>> = vec![Vec::new(); count];
    for j in 0..N {
        for i in 0..N {
            for (di, dj) in [(1, 0), (0, 1)] {
                if i + di >= N || j + dj >= N {
                    continue;
                }
                let (a, b) = (zs[j * N + i], zs[(j + dj) * N + i + di]);
                if !(a.is_finite() && b.is_finite()) {
                    continue;
                }
                let from = ((a.min(b) / step).ceil() as i64).max(first);
                let to = ((a.max(b) / step).floor() as i64).min(first + count as i64 - 1);
                for k in (from..=to).filter(|k| k.rem_euclid(every) == 0) {
                    let level = k as f32 * step;
                    if (a > level) == (b > level) {
                        continue;
                    }
                    let s = ((level - a) / (b - a)) as f64;
                    let p = DVec2::new(at(i) as f64 + s * (at(i + di) - at(i)) as f64, at(j) as f64 + s * (at(j + dj) - at(j)) as f64);
                    if p.x.abs() <= LABEL_BOX[0] && p.y.abs() <= LABEL_BOX[1] {
                        // (as far out as it goes: the middle is where the path and its arrows are)
                        candidates[(k - first) as usize].push((p.x.abs().max(p.y.abs()) + 0.2 * p.length(), p));
                    }
                }
            }
        }
    }
    let mut labels: Vec<(DVec2, f64)> = Vec::new();
    for (k, mut places) in candidates.into_iter().enumerate() {
        places.sort_by(|x, y| y.0.total_cmp(&x.0));
        if let Some(&(_, p)) = places.iter().find(|(_, p)| labels.iter().all(|(q, _)| q.distance(*p) >= 0.75)) {
            labels.push((p, (first + k as i64) as f64 * step as f64));
        }
    }
    Levels { lo, hi, step, labels }
}

/// A contour line's height as written on the map: with the decimals its spacing needs.
fn level_text(v: f64, step: f32) -> String {
    let decimals = (-((step as f64) * (1.0 + 1e-9)).log10().floor()).max(0.0).min(6.0) as usize;
    num(v, decimals)
}

/// The vertices across and down of the map, which covers the view.
const MAP_N: usize = 150;

/// The coloured map with its contour lines: a picture that covers the view, worked out
/// afresh whenever the view or the formula changes.
#[derive(Default)]
struct Map {
    surface: Option<Surface>,
    key: Option<([u32; 4], u64)>,
    xs: Vec<f32>,
    ys: Vec<f32>,
    zs: Vec<f32>,
    batch: Batch,
}

/// The contour line of f at `level` over the map's grid, as polylines: each cell's piece
/// of it found from its four corners (where a cell's corners are high and low in turn, by
/// the value at its middle), and the pieces joined end to end.
fn contour(xs: &[f32], ys: &[f32], zs: &[f32], n: usize, level: f32) -> Vec<Vec<Vec2>> {
    if n < 2 || xs.len() < n * n || ys.len() < n * n || zs.len() < n * n || !level.is_finite() {
        return Vec::new();
    }
    // an edge of the grid by number: across from (i, j) is 2(jn + i), up from it 2(jn + i) + 1
    let across = |i: usize, j: usize| 2 * (j * n + i);
    let up = |i: usize, j: usize| 2 * (j * n + i) + 1;
    let point = |a: usize, b: usize| {
        let (za, zb) = (zs[a], zs[b]);
        let s = ((level - za) / (zb - za)).clamp(0.0, 1.0);
        Vec2::new(xs[a] + (xs[b] - xs[a]) * s, ys[a] + (ys[b] - ys[a]) * s)
    };
    // each vertex once: above the level (1), not (0), or no value (16). A cell whose
    // corners add up to 0 or 4 is all on one side, as most are, and is passed over at a
    // glance; 16 or more, and a corner has no value.
    let side: Vec<u8> = zs[..n * n].iter().map(|z| if z.is_finite() { (*z > level) as u8 } else { 16 }).collect();
    let mut pieces: Vec<[(usize, Vec2); 2]> = Vec::new();
    for j in 0..n - 1 {
        let (low, high) = (&side[j * n..(j + 1) * n], &side[(j + 1) * n..(j + 2) * n]);
        for i in 0..n - 1 {
            let sum = low[i] + low[i + 1] + high[i] + high[i + 1];
            if sum == 0 || sum >= 4 {
                continue;
            }
            // corners anticlockwise from the lower left, and the edges after each
            let k = [j * n + i, j * n + i + 1, (j + 1) * n + i + 1, (j + 1) * n + i];
            let z = k.map(|k| zs[k]);
            let above = k.map(|k| side[k] == 1);
            let edges = [across(i, j), up(i + 1, j), across(i, j + 1), up(i, j)];
            let cross = |e: usize| (edges[e], point(k[e], k[(e + 1) % 4]));
            let changes: Vec<usize> = (0..4).filter(|&e| above[e] != above[(e + 1) % 4]).collect();
            if changes.len() == 2 {
                pieces.push([cross(changes[0]), cross(changes[1])]);
            } else {
                // a saddle: the corners on the other side from the middle are cut off,
                // each by the piece across the two edges beside it
                let middle = (z.iter().sum::<f32>() * 0.25) > level;
                for v in 0..4 {
                    if above[v] != middle {
                        pieces.push([cross((v + 3) % 4), cross(v)]);
                    }
                }
            }
        }
    }
    // join the pieces through the edges they share
    let mut at_edge: std::collections::HashMap<usize, Vec<usize>> = std::collections::HashMap::with_capacity(pieces.len() * 2);
    for (k, piece) in pieces.iter().enumerate() {
        for end in piece {
            at_edge.entry(end.0).or_default().push(k);
        }
    }
    let mut used = vec![false; pieces.len()];
    let mut lines = Vec::new();
    for start in 0..pieces.len() {
        if used[start] {
            continue;
        }
        used[start] = true;
        let mut line: std::collections::VecDeque<(usize, Vec2)> = pieces[start].iter().copied().collect();
        // grow it from each end in turn, for as long as an unused piece goes on from there
        for forwards in [true, false] {
            loop {
                let end = if forwards { line.back() } else { line.front() }.map(|e| e.0);
                let Some(edge) = end else { break };
                let next = at_edge.get(&edge).and_then(|ks| ks.iter().copied().find(|k| !used[*k]));
                let Some(k) = next else { break };
                used[k] = true;
                let other = if pieces[k][0].0 == edge { pieces[k][1] } else { pieces[k][0] };
                if forwards {
                    line.push_back(other);
                } else {
                    line.push_front(other);
                }
            }
        }
        lines.push(line.into_iter().map(|e| e.1).collect());
    }
    lines
}

// ----- the arrows' scale ----------------------------------------------------------------

/// The longest arrow is drawn about this long (in the plane's units).
const ARROW_REACH: f64 = 1.0;

/// The scale the arrows are drawn at: the largest of 1, 2, 2.5, 4 and 5 times a power of
/// ten that keeps the longest gradient on the curve within `ARROW_REACH`. Both gradients
/// share it, so that their lengths compare as they are.
fn arrow_scale(longest: f64) -> f64 {
    if !(longest.is_finite() && longest > 1e-12) {
        return 1.0;
    }
    let want = (ARROW_REACH / longest).clamp(1e-6, 1e6);
    let power = 10f64.powf(want.log10().floor());
    let lead = want / power;
    let nice = [5.0, 4.0, 2.5, 2.0, 1.0].into_iter().find(|m| *m <= lead * (1.0 + 1e-9)).unwrap_or(1.0);
    nice * power
}

/// The scale in words, for the picture's caption.
fn scale_text(k: f64) -> String {
    let digits = if k >= 1.0 { 0 } else { ((-k.log10()).ceil() as usize + 1).min(6) };
    let text = format!("{:.*}", digits, k);
    let text = if text.contains('.') { text.trim_end_matches('0').trim_end_matches('.').to_string() } else { text };
    format!(r"arrows drawn at ${text} \times$ their length")
}

// ----- the presets and experiments ------------------------------------------------------

/// The heights to choose from: the button, the formula, and what the picture's heading
/// calls it.
const PRESETS: [(&str, &str, &str); 6] = [
    ("$x + 2y$", "x + 2y", "$f = x + 2y$"),
    ("two hills", "2 exp(-(x-1.5)^2 - (y-0.5)^2) + 1.4 exp(-(x+1)^2 - (y+1.3)^2)", "two hills"),
    ("a dome", "1 - (x-0.5)^2 - (y-0.3)^2", "a dome"),
    ("$xy$", "xy", "$f = xy$"),
    ("nearest to $0$", "-(x^2 + y^2)", r"$f = -(x^2 + y^2)$"),
    ("ripples", "sin(2x) + cos(3y)", "ripples"),
];

/// The worked example: the formula of `PRESETS[0]`.
const LINEAR: usize = 0;
const HILLS: usize = 1;
const DOME: usize = 2;
const SADDLE: usize = 3;

/// One of the "things to try".
struct Trial {
    label: &'static str,
    look_for: &'static str,
    preset: usize,
    kind: Kind,
    inside: bool,
    /// c as the film starts and as it ends.
    c: [f64; 2],
    seconds: f64,
}

const TRIALS: [Trial; 5] = [
    Trial {
        label: "slide to the top",
        look_for: r"The point starts where the circle $x^2 + y^2 = 1$ runs straight uphill: all of $\nabla f$ (yellow) points along it, and the pink arrow, the part along the circle, is the whole of it. As the point climbs, less and less of $\nabla f$ lies along the circle. At the top, $(0.447, 0.894)$, none does: the pink arrow is gone, $\nabla f$ lies along $\nabla g$ (cyan), and the white contour line through the point touches the circle instead of crossing it. There $f = \sqrt{5} = 2.236$ and $\nabla f = \lambda \nabla g$ with $\lambda = \sqrt{5}/2 = 1.118$: the yellow arrow is 1.118 times as long as the cyan one.",
        preset: LINEAR,
        kind: Kind::Circle,
        inside: false,
        c: [1.0, 1.0],
        seconds: 2.5,
    },
    Trial {
        label: "the multiplier is a price",
        look_for: r"$c$ rises from 1 to 1.1: the circle grows, and its highest value of $f$ rises from 2.236 to 2.345, by 0.109. The multiplier at $c = 1$ was $\lambda = 1.118$, and $1.118 \times 0.1 = 0.112$: nearly the same. $\lambda$ is the rate at which the highest value $f^*$ rises as $c$ rises, the slope of the green curve in the lower chart (the purple line touches it). The rate itself changes a little as $c$ does ($\lambda = 1.066$ at $c = 1.1$), which is why the two numbers are not quite equal.",
        preset: LINEAR,
        kind: Kind::Circle,
        inside: false,
        c: [1.0, 1.1],
        seconds: 2.0,
    },
    Trial {
        label: "every top and bottom satisfies it",
        look_for: r"Two hills, and the circle $x^2 + y^2 = 1$ between them. At four of its points $\nabla f = \lambda \nabla g$, marked here and on the chart: two tops (green) and two bottoms (orange). The point slides from the highest, where $f = 1.429$ and $\lambda = 0.824$, through a bottom, where the two arrows point opposite ways ($\lambda = -0.124$), to the other top: the arrows are in line again ($\lambda = 0.585$), yet $f$ is only 0.934 there. The equations find every candidate; only comparing their heights says which is the highest.",
        preset: HILLS,
        kind: Kind::Circle,
        inside: false,
        c: [1.0, 1.0],
        seconds: 3.0,
    },
    Trial {
        label: "an inequality that does not bind",
        look_for: r"Now every point of the shaded disc $x^2 + y^2 \le c$ is allowed (the white point stays on its edge), and the disc grows from $c = 0.15$ to $c = 1$. At first the dome's top, $(0.5, 0.3)$, is outside it: the best point is on the edge, with $\lambda > 0$ (0.506 at the start), and $f^*$ rises with $c$. Once $c$ passes 0.34 the top is inside: it is the best point, $\nabla f = 0$ there, and $\lambda = 0$. The constraint no longer binds: a larger disc gains nothing, and the green curve is flat. The highest point on the circle itself is then lower (0.826 at $c = 1$), with $\lambda = -0.417$: on the edge a negative $\lambda$ says the best point lies inside.",
        preset: DOME,
        kind: Kind::Circle,
        inside: true,
        c: [0.15, 1.0],
        seconds: 3.0,
    },
    Trial {
        label: "touching twice at the same height",
        look_for: r"$f = xy$ is the same at $(x, y)$ and at $(-x, -y)$. On the circle $x^2 + y^2 = 2$ its highest value, 1, is reached twice, at $(1, 1)$ and at $(-1, -1)$, where the circle touches the two branches of the contour line $xy = 1$, with $\lambda = 0.5$ at both. The point slides from one to the other, through the lowest point, $(1, -1)$, where $f = -1$ and $\lambda = -0.5$. A best point need not be the only one, and the condition finds all of them.",
        preset: SADDLE,
        kind: Kind::Circle,
        inside: false,
        c: [2.0, 2.0],
        seconds: 3.0,
    },
];

// ----- what glides ----------------------------------------------------------------------

/// What the picture is drawn for: the curve's c, and the point's place on it.
#[derive(Clone, Copy, Debug, PartialEq)]
struct Setting {
    c: f64,
    place: f64,
}

/// How fast a change is glided to: each of the two eases one after the other covers two
/// thirds of what is left in this many seconds.
const FOLLOW: f64 = 0.09;

/// A setting as asked for, and as drawn: the drawn one follows it through `between`, two
/// eases one after the other, so that it sets off gently and arrives exactly.
#[derive(Clone, Copy, Debug)]
struct Glide {
    to: Setting,
    between: Setting,
    shown: Setting,
}

impl Glide {
    fn at(s: Setting) -> Glide {
        Glide { to: s, between: s, shown: s }
    }

    fn arrived(&self) -> bool {
        self.shown == self.to && self.between == self.to
    }

    /// Change part of the setting at once, in all three: what a hand moves.
    fn now(&mut self, change: impl Fn(&mut Setting)) {
        for s in [&mut self.to, &mut self.between, &mut self.shown] {
            change(s);
        }
    }

    fn step(&mut self, dt: f64) {
        if self.arrived() || dt <= 0.0 {
            return;
        }
        let k = 1.0 - (-dt / FOLLOW).exp();
        let ease = |to: f64, between: &mut f64, shown: &mut f64| {
            *between += (to - *between) * k;
            *shown += (*between - *shown) * k;
            (to - *between).abs().max((to - *shown).abs())
        };
        let (to, b, s) = (self.to, &mut self.between, &mut self.shown);
        let apart = ease(to.c, &mut b.c, &mut s.c).max(ease(to.place, &mut b.place, &mut s.place));
        // (exactly there in the end: the picture then stops changing)
        if apart < 1e-5 || !apart.is_finite() {
            *self = Glide::at(self.to);
        }
    }
}

/// A film: the setting carried from one to another over some seconds of the scene's time.
#[derive(Clone, Copy, Debug)]
struct Film {
    from: Setting,
    to: Setting,
    seconds: f64,
    t: f64,
}

impl Film {
    fn now(&self) -> Setting {
        if self.t >= self.seconds {
            return self.to;
        }
        let s = smooth(self.t / self.seconds.max(1e-6));
        Setting { c: self.from.c + (self.to.c - self.from.c) * s, place: self.from.place + (self.to.place - self.from.place) * s }
    }
}

/// A change that waits for the picture to fade out: a new formula, curve or region, and
/// what to show with it.
#[derive(Clone, Debug)]
struct Change {
    formula: Option<&'static str>,
    kind: Kind,
    inside: bool,
    /// The setting to start from (None: the point stays where it is, on the new curve).
    setting: Option<Setting>,
    film: Option<Film>,
    since: Option<Since>,
}

/// The comparison of the change in the highest value with λ times the change in c, from
/// where c was when it began: c, f*(c) and λ there.
#[derive(Clone, Copy, Debug, PartialEq)]
struct Since {
    c: f64,
    best: f64,
    lambda: f64,
}

/// How long the picture takes to fade out before a change of formula or curve, and to
/// fade in after it (seconds).
const FADE_OUT: f32 = 0.12;
const FADE_IN: f32 = 0.3;

// ----- where things go ------------------------------------------------------------------

/// The part of the plane the view is fitted to.
const FRAME_HALF: Vec2 = Vec2::new(2.2, 2.2);

const PAD: f32 = 12.0;
const LINE_H: f32 = 17.0;
/// The width of the panel on a wide canvas.
const BOARD_W: f32 = 372.0;
/// Room under a chart for its numbers and the name of its axis, and left of it for its
/// numbers.
const BELOW: f32 = 32.0;
const LEFT: f32 = 36.0;

/// The panel drawn on the picture: the numbers at the point, the chart of f along the
/// curve, the highest point, and the chart of the highest value against c. In logical
/// pixels.
#[derive(Clone, Debug)]
struct Board {
    min: Vec2,
    max: Vec2,
    /// The top-left of the two lines about the point.
    point: Vec2,
    along_title: Vec2,
    along: plot::Rect,
    highest: Vec2,
    /// The button that takes the point to the highest point.
    go: plot::Rect,
    best_title: Vec2,
    best: plot::Rect,
    slope: Vec2,
    since: Vec2,
}

#[derive(Clone, Debug)]
struct Layout {
    /// A phone held upright: the panel goes below the picture.
    narrow: bool,
    /// The part of the canvas the picture has to itself.
    picture: (Vec2, Vec2),
    /// None on a canvas too small to spare the room (the numbers are in the side panel).
    board: Option<Board>,
}

/// Everything in the board but its two charts, from top to bottom.
fn board_fixed(pad: f32) -> f32 {
    pad + 2.0 * LINE_H + 10.0 + LINE_H + 6.0 + BELOW + LINE_H + 10.0 + LINE_H + 6.0 + BELOW + 2.0 * LINE_H + pad - 2.0
}

fn place_board(min: Vec2, width: f32, chart_h: f32, pad: f32) -> Board {
    let x = min.x + pad;
    let right = min.x + width - pad;
    let mut y = min.y + pad;
    let point = Vec2::new(x, y);
    y += 2.0 * LINE_H + 10.0;
    let along_title = Vec2::new(x, y);
    y += LINE_H + 6.0;
    let along = (Vec2::new(x + LEFT, y), Vec2::new(right - 8.0, y + chart_h));
    y += chart_h + BELOW;
    let highest = Vec2::new(x, y);
    let go = (Vec2::new(right - 78.0, y - 4.0), Vec2::new(right, y + LINE_H + 1.0));
    y += LINE_H + 10.0;
    let best_title = Vec2::new(x, y);
    y += LINE_H + 6.0;
    let best = (Vec2::new(x + LEFT, y), Vec2::new(right - 8.0, y + chart_h));
    y += chart_h + BELOW;
    let slope = Vec2::new(x, y);
    y += LINE_H;
    let since = Vec2::new(x, y);
    y += LINE_H;
    Board { min, max: Vec2::new(min.x + width, y + pad - 2.0), point, along_title, along, highest, go, best_title, best, slope, since }
}

impl Layout {
    fn of(view: Vec2, inset: f32) -> Layout {
        let narrow = view.x < 620.0 && view.y >= 560.0;
        let whole = Layout { narrow, picture: (Vec2::new(0.0, inset), view), board: None };
        if narrow {
            let pad = 10.0;
            let fixed = board_fixed(pad);
            let chart_h = ((view.x - 60.0) * 0.24).clamp(56.0, 86.0);
            // (the picture keeps at least the top 45 %)
            let top = (view.y - fixed - 2.0 * chart_h).max(view.y * 0.45);
            let chart_h = ((view.y - top - fixed) * 0.5).min(chart_h);
            if chart_h < 40.0 {
                return whole;
            }
            let mut board = place_board(Vec2::new(0.0, top), view.x, chart_h, pad);
            board.max = view;
            return Layout { narrow, picture: (Vec2::new(0.0, inset), Vec2::new(view.x, top)), board: Some(board) };
        }
        let top = inset + 12.0;
        let chart_h = (BOARD_W * 0.3).min((view.y - 12.0 - top - board_fixed(PAD)) * 0.5);
        if view.x < 640.0 || chart_h < 56.0 {
            return whole;
        }
        let board = place_board(Vec2::new(view.x - BOARD_W - 12.0, top), BOARD_W, chart_h, PAD);
        let picture = (Vec2::new(0.0, inset), Vec2::new(board.min.x - 8.0, view.y));
        Layout { narrow, picture, board: Some(board) }
    }

    /// The part of the canvas the scene's frame is fitted into: the picture's, less the
    /// heading at its top and the caption at its foot.
    fn frame_room(&self) -> (Vec2, Vec2) {
        let (min, max) = self.picture;
        let foot = if self.narrow { 8.0 } else { 40.0 };
        (min + Vec2::new(12.0, 44.0), max - Vec2::new(12.0, foot))
    }
}

fn inside_rect(p: Vec2, (min, max): (Vec2, Vec2)) -> bool {
    p.cmpge(min).all() && p.cmple(max).all()
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

fn overlap(a: (Vec2, Vec2), b: (Vec2, Vec2)) -> bool {
    a.0.x < b.1.x && b.0.x < a.1.x && a.0.y < b.1.y && b.0.y < a.1.y
}

/// The parts of the picture drawn as lines whose number of points changes from frame to
/// frame, each in a batch of its own (different from the usual one by a hair of its
/// width), so that what stays the same can still be compared point for point from one
/// frame to the next (`eidos motion`).
#[derive(Clone, Copy, PartialEq)]
enum Part {
    /// The contour line through the point.
    Level = 1,
    /// The curve g = c (as many points as its kind has: a change of kind changes them).
    Curve = 2,
}

fn lines_for<'a>(f: &'a mut Frame, part: Part) -> &'a mut crate::gfx::draw::LineBatch {
    f.dl.lines_with(LineKind::Blend, LineStyle { width_scale: 1.0 + 1e-4 * part as u8 as f32, ..Default::default() })
}

/// A batch of lines with nothing in it this frame is given points that cannot be seen: a
/// batch left empty is dropped, and every batch after it would then be compared with the
/// one before it in the frame before. (Three of them, not the two of a segment: a line of
/// two points that appears where there were none must not be compared with them.)
fn keep_batch(batch: &mut crate::gfx::draw::LineBatch) {
    if batch.pts.len() <= 1 {
        batch.brk();
        for _ in 0..3 {
            batch.push(Vec3::ZERO, 0.0, Color::CLEAR);
        }
    }
}

/// A label on one side of a point of the picture, at a fixed distance, written where it
/// is clear of what is drawn and fading out where something comes over it: it never goes
/// round to another side. (For labels too wide to glide round a point in a frame or two
/// without being taken for one that jumps.) `alpha` is how strongly it is written, which
/// eases towards 1 or 0; `moving` is set while it does.
#[allow(clippy::too_many_arguments)]
fn fading_label(room: &mut Room, f: &mut Frame, at: Vec2, text: &str, size: f32, color: Color, towards: Vec2, gap: f32, alpha: &mut f32, moving: &mut bool) {
    let Some(from) = on_screen(f, at.extend(0.0)).filter(|p| p.is_finite() && at.is_finite()) else { return };
    let half = f.text_size(text, size) * 0.5 + Vec2::new(4.0, 1.0);
    let side = towards.normalize_or_zero();
    // (the box's nearest point `gap` from the point, whichever way the side is)
    let centre = from + side * (gap + half.x * side.x.abs() + half.y * side.y.abs());
    let clear = room.clearance(centre - half, centre + half) >= 3.0;
    if clear {
        room.boxed(centre - half, centre + half);
    }
    let to = if clear { 1.0 } else { 0.0 };
    *alpha += (to - *alpha) * (1.0 - (-f.real_dt.max(0.0) / 0.12).exp());
    if (to - *alpha).abs() < 0.01 {
        *alpha = to;
    } else {
        *moving = true;
    }
    f.text_px(centre, text, size, color.a(color.0[3] as f32 / 255.0 * *alpha), Align2::CENTER_CENTER);
}

/// A label beside a point of the picture that keeps its place in the list of labels
/// whether or not it is written: where there is no room for it, or it is not wanted, it
/// is written in no colour where it last was. (A label that comes and goes moves every
/// label after it up or down the list, and `eidos motion` would take them all for labels
/// that jump.)
#[allow(clippy::too_many_arguments)]
fn steady_label(room: &mut Room, f: &mut Frame, name: &'static str, at: Vec2, text: &str, size: f32, color: Color, towards: Vec2, gap: f32, wanted: bool) {
    if !at.is_finite() {
        return;
    }
    if wanted && room.label(f, name, at.extend(0.0), text, size, color, towards, gap, true) {
        return;
    }
    if let Some(from) = on_screen(f, at.extend(0.0)).filter(|p| p.is_finite()) {
        let offset = f.places.chosen(name, 0).unwrap_or(Vec2::ZERO);
        f.text_placed(name, 0, from, offset, text, size, color.a(0.0));
    }
}

/// The chart's kinds of line (see `plot::apart`).
const ALONG_CURVE: u8 = plot::OWN;
const BEST_CURVE: u8 = plot::OWN + 1;
const BEST_EDGE: u8 = plot::OWN + 2;
const TANGENT: u8 = plot::OWN + 3;
const SINCE: u8 = plot::OWN + 4;

// ----- colours --------------------------------------------------------------------------

/// The curve g = c and its gradient.
const CURVE: Color = pal::CYAN;
/// The gradient of f.
const GRAD_F: Color = pal::YELLOW;
/// The part of ∇f along the curve.
const ALONG: Color = pal::PINK;
/// The multiplier λ, and the slope it is.
const LAMBDA: Color = pal::PURPLE;
/// Tops, the highest point, and the highest value f*.
const TOP: Color = pal::GREEN;
const BOTTOM: Color = pal::ORANGE;

// ----- the scene ------------------------------------------------------------------------

pub struct Lagrange {
    /// The formula as typed or chosen.
    f: ExprField,
    /// The formula the picture is drawn for (it waits for the picture to fade out when a
    /// button changes it), and a count of its changes, by which what is worked out from
    /// it is kept.
    expr: Expr,
    drawn: String,
    version: u64,
    kind: Kind,
    inside: bool,
    glide: Glide,
    film: Option<Film>,
    /// Play the film once the glide to its start has arrived (and the picture is in).
    waiting: bool,
    change: Option<Change>,
    /// How far the picture has faded in (0 to 1); the frame in which a change is made
    /// draws nothing that depends on it.
    fade: f32,
    blank: bool,
    since: Option<Since>,
    /// The c handle was being dragged in the frame before.
    holding_c: bool,
    // what is worked out from the formula, the curve and c, kept while they stay the same
    tops: Vec<(DVec2, f64)>,
    levels: Levels,
    along: Along,
    along_key: Option<(u64, Kind, u64)>,
    best: Option<Best>,
    profile: Profile,
    profile_key: Option<(u64, Kind, bool)>,
    slope: Option<f64>,
    slope_key: Option<(u64, Kind, bool, u64)>,
    // the picture
    map: Map,
    level_lines: Vec<Vec<Vec2>>,
    level_key: Option<([u32; 4], u64, u32)>,
    /// What the level line drawn last was found for, and how many points it had.
    level_drawn: Option<([u32; 4], u64, u32)>,
    level_count: usize,
    /// How strongly each contour height is written (it fades where something is drawn
    /// over its place), and whether any is still fading.
    label_alpha: Vec<f32>,
    /// The same for the name of the curve and of the highest point.
    name_alpha: [f32; 2],
    labels_moving: bool,
    /// The arrows' scale as asked for and as drawn (it glides).
    scale: f64,
    scale_shown: f64,
    /// The ranges of the two charts' heights as drawn (they glide).
    ranges: plot::Glide,
    ranges_moving: bool,
    /// The canvas last drawn on and how much of its top the application's buttons cover.
    canvas: Vec2,
    inset: f32,
    measured: bool,
    new_view: bool,
}

impl Lagrange {
    fn new() -> Self {
        let f = ExprField::new(PRESETS[LINEAR].1, &VARS);
        let mut scene = Lagrange {
            expr: f.expr.clone(),
            drawn: f.text.clone(),
            f,
            version: 0,
            kind: Kind::Circle,
            inside: false,
            glide: Glide::at(Setting { c: 1.0, place: 0.0 }),
            film: None,
            waiting: false,
            change: None,
            fade: 1.0,
            blank: false,
            since: None,
            holding_c: false,
            tops: Vec::new(),
            levels: Levels::default(),
            along: Along::default(),
            along_key: None,
            best: None,
            profile: Profile::default(),
            profile_key: None,
            slope: None,
            slope_key: None,
            map: Map::default(),
            level_lines: Vec::new(),
            level_key: None,
            level_drawn: None,
            level_count: 0,
            label_alpha: Vec::new(),
            name_alpha: [1.0; 2],
            labels_moving: false,
            scale: 1.0,
            scale_shown: 1.0,
            ranges: plot::Glide::default(),
            ranges_moving: false,
            canvas: Vec2::new(1080.0, 1000.0),
            inset: 0.0,
            measured: false,
            new_view: false,
        };
        scene.formula_changed();
        // (opening, nothing glides or fades: the first experiment's film plays from its start)
        scene.set_up(0);
        if let Some(change) = scene.change.take() {
            scene.apply(change);
        }
        scene.glide = Glide::at(scene.glide.to);
        scene.fade = 1.0;
        scene.blank = false;
        scene.refresh();
        scene.scale_shown = scene.scale;
        scene
    }

    /// The formula drawn has changed: what depends on it alone is worked out again.
    fn formula_changed(&mut self) {
        self.version += 1;
        self.tops = hilltops(&self.expr);
        self.levels = levels(&self.expr);
        self.map.key = None;
        self.level_key = None;
        self.label_alpha.clear();
    }

    /// Set up experiment `k` of `TRIALS`: fade to its formula and curve, put the point at
    /// the film's start, fade in, and play the film.
    fn set_up(&mut self, k: usize) {
        let trial = &TRIALS[k];
        let formula = PRESETS[trial.preset].1;
        let expr = Expr::parse(formula, &VARS).unwrap_or_else(|_| self.expr.clone());
        let [c0, c1] = trial.c;
        // the film's places, from where f's stationary points are on its curve
        let marks = along(&expr, trial.kind, c0, SAMPLES);
        let tops: Vec<f64> = marks.stationary.iter().filter(|s| s.top).map(|s| s.at.t).collect();
        let highest = marks.highest.map_or(0.0, |h| h.t);
        let (from, to) = match k {
            // from where the circle runs straight uphill (along ∇f = (1, 2)) to the top
            0 => ((-1.0f64).atan2(2.0), highest),
            1 => (highest, highest),
            // from the highest top, clockwise, to the other one
            2 => {
                let other = tops.iter().copied().filter(|t| (t - highest).abs() > 1e-6).fold(highest, |o, t| if (t - highest).abs() > (o - highest).abs() { t } else { o });
                (highest, if other < highest { other } else { other - 2.0 * PI64 })
            }
            // towards the dome's top: the highest point of every circle
            3 => (0.3f64.atan2(0.5), 0.3f64.atan2(0.5)),
            // from (1, 1) clockwise to (−1, −1)
            _ => (0.25 * PI64, -0.75 * PI64),
        };
        let from = Setting { c: c0, place: from };
        let to = Setting { c: c1, place: to };
        let since = (k == 1).then(|| {
            let at = marks.highest.map_or((f64::NAN, f64::NAN), |h| (h.f, h.lambda));
            Since { c: c0, best: at.0, lambda: at.1 }
        });
        self.ask(Change { formula: Some(formula), kind: trial.kind, inside: trial.inside, setting: Some(from), film: Some(Film { from, to, seconds: trial.seconds, t: 0.0 }), since });
    }

    /// Ask for a change: made at once if it changes nothing but the setting (which glides
    /// there), after the picture has faded out if it changes the formula, the curve or
    /// the region.
    fn ask(&mut self, change: Change) {
        let same = change.formula.is_none_or(|t| t == self.drawn) && change.kind == self.kind && change.inside == self.inside;
        self.film = None;
        self.waiting = false;
        if same {
            // (a change waiting to be made is not made after all: the box shows the
            // formula drawn again)
            if self.change.take().is_some() && self.f.text != self.drawn {
                let drawn = self.drawn.clone();
                self.f.set(&drawn);
            }
            if let Some(s) = change.setting {
                self.glide.to = Setting { c: self.kind.clamp_c(s.c), place: self.near_place(s.place) };
                // (a film's places are meant as they are, the way round included)
                if let Some(film) = change.film {
                    self.glide.to.place = film.from.place;
                }
            }
            self.film = change.film;
            self.waiting = self.film.is_some();
            self.since = change.since;
            return;
        }
        if let Some(t) = change.formula {
            self.f.set(t);
        }
        self.change = Some(change);
    }

    /// Make a change that has waited for the picture to fade out: the picture is then
    /// blank for a frame, and fades in.
    fn apply(&mut self, change: Change) {
        let point = self.kind.at(self.glide.shown.c, self.glide.shown.place);
        if let Some(text) = change.formula.filter(|t| *t != self.drawn) {
            if let Ok(expr) = Expr::parse(text, &VARS) {
                self.expr = expr;
                self.drawn = text.to_string();
                self.formula_changed();
            }
        }
        let kind_changed = change.kind != self.kind;
        self.kind = change.kind;
        self.inside = change.inside && change.kind.closed();
        let setting = match change.setting {
            Some(s) => Setting { c: self.kind.clamp_c(s.c), place: s.place },
            // the new curve through where the point was, the point staying there
            None if kind_changed => {
                let g = self.kind.g(point);
                let c = if g.is_finite() && (!self.kind.closed() || g > 1e-9) { self.kind.clamp_c(g) } else { self.kind.usual_c() };
                Setting { c, place: self.kind.place_of(c, point) }
            }
            None => self.glide.shown,
        };
        self.glide = Glide::at(setting);
        self.film = change.film;
        self.waiting = self.film.is_some();
        self.since = change.since;
        self.fade = 0.0;
        self.blank = true;
        self.along_key = None;
        self.profile_key = None;
        self.slope_key = None;
        self.level_key = None;
        self.refresh();
        // (the arrows start at their new scale: there is nothing on show to glide from)
        self.scale_shown = self.scale;
        self.ranges = plot::Glide::default();
    }

    /// A place for the point near where it is drawn: for a closed curve, the same
    /// direction the nearest way round.
    fn near_place(&self, t: f64) -> f64 {
        if self.kind.closed() {
            let shown = self.glide.shown.place;
            shown + wrap(t - shown)
        } else {
            t.max(-REACH).min(REACH)
        }
    }

    /// Work out again what depends on what has changed: f along the curve at c, the best
    /// point, the chart of f* against c, and the slope of f* at c.
    fn refresh(&mut self) {
        let c = self.glide.shown.c;
        let key = (self.version, self.kind, c.to_bits());
        if self.along_key != Some(key) {
            self.along = along(&self.expr, self.kind, c, SAMPLES);
            self.along_key = Some(key);
            self.best = best_of(&self.along, &self.tops, self.kind, c, self.inside);
            let longest = self.along.samples.iter().map(|h| h.grad_f.length().max(h.grad_g.length())).filter(|v| v.is_finite()).fold(0.0, f64::max);
            self.scale = arrow_scale(longest);
        }
        let key = (self.version, self.kind, self.inside);
        if self.profile_key != Some(key) {
            self.profile = profile(&self.expr, &self.tops, self.kind, self.inside);
            self.profile_key = Some(key);
            self.best = best_of(&self.along, &self.tops, self.kind, c, self.inside);
        }
        let key = (self.version, self.kind, self.inside, c.to_bits());
        if self.slope_key != Some(key) {
            self.slope = slope_of_best(&self.expr, &self.tops, self.kind, c, self.inside, &self.along);
            self.slope_key = Some(key);
        }
    }

    /// What f does at the point as drawn.
    fn at_point(&self) -> Here {
        here(&self.expr, self.kind, self.glide.shown.c, self.glide.shown.place)
    }

    /// λ at the best point (0 inside, where ∇f = 0; at the point if f is the same all
    /// along the curve).
    fn best_lambda(&self) -> Option<f64> {
        match self.best? {
            Best::Edge { at, .. } => Some(at.lambda),
            Best::Inside { .. } => Some(0.0),
            Best::Everywhere { .. } => Some(self.at_point().lambda),
        }
    }

    /// The tops that are the highest point of the curve (more than one if several are
    /// as high), when the best point is on the curve.
    fn highest_stationary(&self) -> Vec<Here> {
        match self.best {
            Some(Best::Edge { at, ties }) if ties > 1 => self.along.stationary.iter().filter(|s| s.top && (s.at.f - at.f).abs() <= 1e-9 * (1.0 + at.f.abs())).map(|s| s.at).collect(),
            Some(Best::Edge { at, .. }) => vec![at],
            _ => Vec::new(),
        }
    }

    fn highest_points(&self) -> Vec<DVec2> {
        self.highest_stationary().into_iter().map(|h| h.p).filter(|p| p.is_finite()).collect()
    }

    /// Glide the point to the highest point of the curve (the nearest, if there are
    /// several as high).
    fn glide_to_highest(&mut self) {
        let h = self.along.highest;
        let ties: Vec<f64> = match h {
            Some(h) if self.along.ties > 1 => self.along.stationary.iter().filter(|s| s.top && (s.at.f - h.f).abs() <= 1e-9 * (1.0 + h.f.abs())).map(|s| s.at.t).collect(),
            Some(h) => vec![h.t],
            None => Vec::new(),
        };
        let shown = self.glide.shown.place;
        let nearest = ties.into_iter().map(|t| self.near_place(t)).min_by(|a, b| (a - shown).abs().total_cmp(&(b - shown).abs()));
        if let Some(t) = nearest {
            self.stop_film();
            self.glide.to.place = t;
        }
    }

    fn glide_to_lowest(&mut self) {
        let low = self.along.samples.iter().filter(|h| h.f.is_finite()).min_by(|a, b| a.f.total_cmp(&b.f)).copied();
        // (refined to the bottom beside the lowest sample)
        let low = low.and_then(|l| self.along.stationary.iter().filter(|s| !s.top).map(|s| s.at).min_by(|a, b| a.f.total_cmp(&b.f)).filter(|s| s.f <= l.f).or(Some(l)));
        if let Some(l) = low {
            self.stop_film();
            self.glide.to.place = self.near_place(l.t);
        }
    }

    fn stop_film(&mut self) {
        self.film = None;
        self.waiting = false;
    }

    /// Move everything on: `dt` seconds of the scene's time (the film) and `real_dt` of
    /// the clock's (glides and fades go by the clock, so that they are seen through even
    /// while the scene's time stands still).
    fn advance(&mut self, dt: f32, real_dt: f32) {
        self.blank = false;
        if self.change.is_some() {
            self.fade = (self.fade - real_dt.max(0.0) / FADE_OUT).max(0.0);
            if self.fade <= 0.0 {
                if let Some(change) = self.change.take() {
                    self.apply(change);
                }
            }
        } else if self.fade < 1.0 {
            self.fade = (self.fade + real_dt.max(0.0) / FADE_IN).min(1.0);
        }
        self.glide.step(real_dt as f64);
        if self.waiting && self.change.is_none() && self.fade >= 1.0 && self.glide.arrived() {
            self.waiting = false;
        }
        if let (Some(film), false) = (self.film.as_mut(), self.waiting || self.change.is_some()) {
            film.t = (film.t + dt.max(0.0) as f64).min(film.seconds);
            let s = film.now();
            let done = film.t >= film.seconds;
            self.glide.now(|g| *g = s);
            if done {
                self.film = None;
            }
        }
        // (a direction that has gone round and round is the same direction)
        if self.kind.closed() && self.glide.arrived() && self.glide.to.place.abs() > 4.0 * PI64 {
            self.glide = Glide::at(Setting { place: wrap(self.glide.to.place), ..self.glide.to });
        }
        self.refresh();
        if self.scale_shown != self.scale {
            let k = 1.0 - (-(real_dt as f64) / 0.15).exp();
            self.scale_shown += (self.scale - self.scale_shown) * k;
            if (self.scale_shown / self.scale - 1.0).abs() < 1e-3 || !self.scale_shown.is_finite() {
                self.scale_shown = self.scale;
            }
        }
        let ranges = self.chart_ranges();
        if self.ranges.shown.len() != ranges.len() {
            self.ranges.snap(&ranges);
            self.ranges_moving = false;
        } else {
            self.ranges_moving = self.ranges.follow(&ranges, real_dt, false);
        }
    }

    fn at_rest(&self) -> bool {
        self.glide.arrived() && self.film.is_none() && !self.waiting && self.change.is_none() && self.fade >= 1.0 && self.scale_shown == self.scale && !self.ranges_moving && !self.labels_moving
    }

    /// The heights the two charts show: of f along the curve, and of f* against c, each a
    /// little wider than the values, to round numbers.
    fn chart_ranges(&self) -> [f32; 4] {
        let span = |values: &mut dyn Iterator<Item = f64>| {
            // (heights beyond any a chart can show are left out: a formula may give 1e300)
            const MOST: f64 = 1e30;
            let (mut lo, mut hi) = (f64::INFINITY, f64::NEG_INFINITY);
            for v in values.filter(|v| v.abs() <= MOST) {
                lo = lo.min(v);
                hi = hi.max(v);
            }
            // (nothing at all to show)
            if lo > hi {
                return [0.0f32, 1.0];
            }
            let pad = if hi - lo > 1e-9 * (1.0 + hi.abs()) { 0.08 * (hi - lo) } else { 0.5 * (1.0 + hi.abs()) };
            let step = plot::round_step((hi - lo + 2.0 * pad) / 8.0);
            let (a, b) = (((lo - pad) / step).floor() * step, ((hi + pad) / step).ceil() * step);
            let (a, b) = (a.max(-2.0 * MOST), b.min(2.0 * MOST));
            if a.is_finite() && b.is_finite() && b > a {
                [a as f32, b as f32]
            } else {
                [lo as f32, (lo + 1.0) as f32]
            }
        };
        let [a, b] = span(&mut self.along.samples.iter().map(|h| h.f));
        let [c, d] = span(&mut self.profile.best.iter().chain(&self.profile.on_curve).copied());
        [a, b, c, d]
    }

    /// The canonical view: the picture's frame in the middle of the part of the canvas
    /// it has to itself.
    fn the_view(&self) -> View {
        let lay = Layout::of(self.canvas, self.inset);
        let (min, max) = lay.frame_room();
        let room = (max - min).max(Vec2::splat(60.0));
        let per_unit = (room.x / (2.0 * FRAME_HALF.x)).min(room.y / (2.0 * FRAME_HALF.y)).max(1e-3);
        let off = ((min + max) * 0.5 - self.canvas * 0.5) / per_unit;
        View::Fit { center: Vec2::new(-off.x, off.y), half: self.canvas.max(Vec2::ONE) * (0.5 / per_unit) }
    }

    /// The formula's name, for the picture's heading.
    fn f_name(&self) -> &'static str {
        PRESETS.iter().find(|p| p.1 == self.drawn).map_or("your $f$", |p| p.2)
    }

    /// The picture's heading: what is being looked for.
    fn heading(&self) -> String {
        let c = num(self.glide.shown.c, 2);
        let g = self.kind.g_text();
        if self.inside {
            let region = if self.kind == Kind::Circle { "disc" } else { "ellipse" };
            format!(r"{} in the {region} ${g} \le {c}$", self.f_name())
        } else {
            format!("{} on the {} ${g} = {c}$", self.f_name(), self.kind.name())
        }
    }
}

impl Scene for Lagrange {
    fn view(&self) -> View {
        self.the_view()
    }

    fn take_view(&mut self) -> Option<View> {
        std::mem::take(&mut self.new_view).then(|| self.the_view())
    }

    fn settings(&self, s: &mut ViewSettings) {
        s.grid_alpha = 0.07;
        s.axis_alpha = 0.3;
    }

    fn animated(&self) -> bool {
        !self.at_rest()
    }

    fn pointer_hint(&self) -> Option<&'static str> {
        Some("drag the white dot along the path, and the cyan dot to change $c$")
    }

    fn ui(&mut self, ui: &mut egui::Ui, _: &Gfx) {
        self.refresh();
        ui::note(ui, r"Where on a path across a hillside is the path highest? The hill is $f(x, y)$, drawn as contour lines (lines of equal height); the path is the curve $g(x, y) = c$. Where the path crosses a contour line and $\nabla f \ne 0$, one way along it leads uphill; so at its highest point, if $\nabla f \ne 0$ there, it cannot cross the contour line, only touch it. There the gradient $\nabla f$, which is at right angles to the contour line, is also at right angles to the path, as $\nabla g$ is:");
        ui::equation(ui, r"\nabla f = \lambda \nabla g, \quad g(x, y) = c");
        ui::note(ui, r"The number $\lambda$, the Lagrange multiplier, says what the constraint costs: raise $c$ a little, and the highest value changes by about $\lambda$ times as much.");

        ui::section(ui, "Things to try");
        for (k, trial) in TRIALS.iter().enumerate() {
            if ui::experiment(ui, trial.label, trial.look_for) {
                self.set_up(k);
            }
        }

        ui::section(ui, "The hill $f$");
        if self.f.ui(ui, "f(x, y) =") {
            // (typed: drawn at once, letter by letter, with nothing to fade)
            self.expr = self.f.expr.clone();
            self.drawn = self.f.text.clone();
            self.change = None;
            self.formula_changed();
            self.stop_film();
        }
        let names: Vec<&str> = PRESETS.iter().map(|p| p.0).collect();
        if let Some(i) = ui::buttons(ui, &names) {
            self.ask(Change { formula: Some(PRESETS[i].1), kind: self.kind, inside: self.inside, setting: None, film: None, since: None });
        }
        ui::note(ui, "Any formula in $x$ and $y$.");

        ui::section(ui, "The path $g = c$");
        let labels = [r"circle $x^2 + y^2 = c$", r"ellipse $x^2/4 + y^2 = c$", r"line $x + 2y = c$"];
        let mut k = KINDS.iter().position(|k| *k == self.change.as_ref().map_or(self.kind, |c| c.kind)).unwrap_or(0);
        if ui::choice(ui, &labels, &mut k) {
            let inside = self.inside && KINDS[k].closed();
            self.ask(Change { formula: None, kind: KINDS[k], inside, setting: None, film: None, since: None });
        }
        let [lo, hi] = self.kind.c_range();
        let mut c = self.glide.to.c as f32;
        if ui::slider(ui, "$c$", &mut c, lo as f32..=hi as f32) {
            self.stop_film();
            let c = self.kind.clamp_c(c as f64);
            self.glide.now(|s| s.c = c);
        }
        if self.kind.closed() {
            let mut region = self.change.as_ref().map_or(self.inside, |c| c.inside) as usize;
            if ui::choice(ui, &["on the curve, $g = c$", r"inside it too, $g \le c$"], &mut region) {
                self.ask(Change { formula: None, kind: self.kind, inside: region == 1, setting: None, film: None, since: None });
            }
            ui::note(ui, r"With $g \le c$ every point of the region the curve encloses is allowed: the best point is then on the curve, or at the top of a hill inside it. (The white point stays on the curve.)");
        }
        let (mut place, range, name) = if self.kind.closed() {
            (wrap(self.glide.to.place).to_degrees() as f32, -180.0f32..=180.0, r"the point's direction $\theta$ (degrees)")
        } else {
            (self.glide.to.place as f32, -REACH as f32..=REACH as f32, "the point's place $s$")
        };
        if ui::slider(ui, name, &mut place, range) {
            self.stop_film();
            let t = if self.kind.closed() { self.near_place((place as f64).to_radians()) } else { place as f64 };
            self.glide.now(|s| s.place = t);
        }
        ui::note(ui, r"$\theta$: the direction of the point seen from $0$. $s$: on the line, the distance from its point nearest $0$. Or drag the white dot.");
        if let Some(i) = ui::buttons(ui, &["to the highest point", "to the lowest point"]) {
            if i == 0 {
                self.glide_to_highest();
            } else {
                self.glide_to_lowest();
            }
        }

        ui::section(ui, "Reading the picture");
        ui::key(
            ui,
            &[
                (Cmap::Slate.color(0.7), "the map: the height $f$, lighter where it is higher, with contour lines and their heights"),
                (CURVE, r"cyan: the path $g = c$ (the dot on it changes $c$), and the arrow $\nabla g$ at the point"),
                (pal::WHITE, "white: the point (drag it along the path), and the contour line through it"),
                (GRAD_F, r"yellow: $\nabla f$ at the point; dashed: its part across the path, $\lambda \nabla g$, drawn from the end of the pink arrow"),
                (ALONG, r"pink: the part of $\nabla f$ along the path, which is 0 at a stationary point"),
                (TOP, "green: the tops of $f$ along the path, and the highest point; the chart of the highest value $f^*$ against $c$"),
                (BOTTOM, "orange: the bottoms"),
                (LAMBDA, r"purple: $\lambda$, and on the lower chart a line of slope $\lambda$"),
            ],
        );

        ui::section(ui, "Words");
        ui::term(ui, "Contour line:", "a line joining points where $f$ has the same value, as a line of equal height joins them on a map.");
        ui::term(ui, r"Gradient $\nabla f$:", r"the arrow $(\partial f/\partial x, \partial f/\partial y)$. It points straight uphill, at right angles to the contour line, and its length is the steepest slope.");
        ui::term(ui, "Constraint:", r"the condition the point must meet: on the curve $g(x, y) = c$, or in the region $g(x, y) \le c$.");
        ui::term(ui, "Stationary point:", r"here, a point of the path where $f$ stops changing as the point moves along it: the part of $\nabla f$ along the path is 0. Tops and bottoms are such points.");
        ui::term(ui, r"Lagrange multiplier $\lambda$:", r"the number with $\nabla f = \lambda \nabla g$ at a stationary point. Elsewhere on the path the scene splits $\nabla f$ into $\lambda \nabla g$, with $\lambda = \nabla f \cdot \nabla g / |\nabla g|^2$, and a part along the path. With the constraint written $g = c$, $\lambda$ at the highest point is the slope of the highest value $f^*(c)$, wherever the highest point moves smoothly as $c$ changes; written $-g = -c$, $\lambda$ would change sign.");
        ui::term(ui, "Binding:", r"a constraint $g \le c$ binds if the best point lies on the curve $g = c$; then $\lambda \ge 0$. If it lies inside, $\nabla f = 0$ there, the constraint does nothing, and $\lambda = 0$.");
        ui::term(ui, r"Where $\nabla g = 0$:", r"the method says nothing (the curve may have a corner there, or shrink to a point). On these three curves $\nabla g$ is never 0, which is why $c$ stays above 0 for the circle and the ellipse.");

        ui::section(ui, "Numbers");
        let h = self.at_point();
        let fixed = |v: f64| if v.is_finite() { format!("${}$", num(v, 3)) } else { "none".to_string() };
        ui::readout(ui, "the point", if h.p.is_finite() { format!("${}$", tuple(&[h.p.x, h.p.y], 3)) } else { "none".into() });
        ui::readout(ui, "$f$ there", fixed(h.f));
        ui::readout_colored(ui, r"$\lambda$ there", fixed(h.lambda), LAMBDA);
        ui::readout_colored(ui, r"part of $\nabla f$ along the path", fixed(h.slope), ALONG);
        let best = self.best;
        ui::readout_colored(ui, "highest value $f^*$", best.map_or("none".into(), |b| fixed(b.f())), TOP);
        ui::readout_colored(ui, r"$\lambda$ at the highest point", self.best_lambda().map_or("none".into(), fixed), LAMBDA);
        ui::readout(ui, r"slope of $f^*$ against $c$", self.slope.map_or("none".into(), fixed));
        ui::see_also(ui, &[("surface", "gradients and contour lines"), ("hessian", "telling tops from bottoms by curvature"), ("underdet", "the smallest answer on a line, where a growing circle touches it"), ("descent", "climbing down without a constraint")]);
    }

    fn frame(&mut self, f: &mut Frame) {
        // the panel's place follows the canvas, and the picture's place follows the panel's
        let view = f.view_size();
        let lay = Layout::of(view, f.top_inset);
        let before = Layout::of(self.canvas, self.inset);
        // (a canvas squeezed to a sliver is one the controls have opened over, on a phone)
        let covered = view.x < 160.0;
        let changed = !covered && (lay.narrow != before.narrow || lay.board.is_some() != before.board.is_some() || (view - self.canvas).abs().max_element() > 24.0);
        if !covered && (changed || !self.measured) {
            self.new_view |= changed;
            (self.canvas, self.measured) = (view, true);
        }
        if !covered {
            self.new_view |= (f.top_inset - self.inset).abs() > 1.0;
            self.inset = f.top_inset;
        }

        let pointer = plot::pointer(f);
        if let Some(b) = lay.board.as_ref().filter(|_| !covered) {
            if pointer.is_some_and(|p| inside_rect(p, (b.min, b.max))) {
                // the panel is for its own buttons and handles: no moving the view from there
                f.handles.capture();
                f.handles.claim_clicks();
            }
        }
        self.advance(f.dt, f.real_dt);
        // the board's button (drawn with the board) and its chart's handle
        if let Some(b) = lay.board.as_ref().filter(|_| !covered && !self.blank && self.fade > 0.05) {
            if pointer.is_some_and(|p| f.ptr.clicked && inside_rect(p, b.go)) {
                self.glide_to_highest();
            }
            let [lo, hi] = self.kind.t_range();
            let (x0, x1) = if self.kind.closed() { (-180.0, 180.0) } else { (lo, hi) };
            let chart = Plot::new(b.along, [x0, x1], [self.ranges.get(0) as f64, self.ranges.get(1) as f64]);
            let h = self.at_point();
            let x = if self.kind.closed() { wrap(h.t).to_degrees() } else { h.t };
            if h.f.is_finite() {
                if let Some((x, _)) = chart.handle(f, 3, x, h.f.max(chart.y[0]).min(chart.y[1]), pal::WHITE) {
                    self.stop_film();
                    let t = if self.kind.closed() { self.near_place(x.to_radians()) } else { x.max(lo).min(hi) };
                    self.glide.now(|s| s.place = t);
                }
            }
        }
        if !self.blank {
            self.handles(f, &lay);
        }
        // (what a hand has moved is drawn in this frame)
        self.refresh();

        self.map(f, covered);
        let labels_before = f.dl.labels.len();
        let plates = self.picture(f, &lay, covered);
        let numbers_from = f.dl.labels.len();
        f.axes2();
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
        if let Some(b) = &lay.board {
            let (lo, hi) = (b.min - Vec2::splat(2.0), b.max + Vec2::splat(2.0));
            let mut k = 0;
            f.dl.labels.retain(|l| {
                k += 1;
                k <= labels_before || l.backed || !inside_rect(l.pos, (lo, hi))
            });
            self.board(f, b);
        }
    }
}

impl Lagrange {
    /// The two handles on the picture: the point on the curve, and the c handle.
    fn handles(&mut self, f: &mut Frame, lay: &Layout) {
        let under_board = |f: &Frame, p: DVec2| lay.board.as_ref().is_some_and(|b| on_screen(f, p.as_vec2().extend(0.0)).is_some_and(|q| inside_rect(q, (b.min, b.max))));
        let Setting { c, place } = self.glide.shown;
        let point = self.kind.at(c, place);
        let mut q = point.as_vec2();
        if q.is_finite() && (!under_board(f, point) || f.handles.dragging()) && f.drag2(1, &mut q, pal::WHITE) {
            self.stop_film();
            let t = self.kind.place_of(c, q.as_dvec2());
            let t = if self.kind.closed() { place + wrap(t - place) } else { t };
            self.glide.now(|s| s.place = t);
        }
        let handle = self.kind.at(c, self.kind.handle_t());
        let mut q = handle.as_vec2();
        let mut held = false;
        if q.is_finite() && (!under_board(f, handle) || f.handles.dragging()) && f.drag2(2, &mut q, CURVE) {
            held = true;
            if !self.holding_c {
                // the comparison with λ starts from where c was taken hold of
                self.since = self.best.map(|b| Since { c, best: b.f(), lambda: self.best_lambda().unwrap_or(f64::NAN) });
            }
            self.stop_film();
            let c = self.kind.c_through(q.as_dvec2());
            self.glide.now(|s| s.c = c);
        }
        self.holding_c = held || (self.holding_c && f.handles.dragging());
    }

    /// The coloured map of f with its contour lines, covering the view.
    fn map(&mut self, f: &mut Frame, covered: bool) {
        if self.map.surface.is_none() {
            self.map.surface = Some(Surface::new(f.gfx, MAP_N as u32, MAP_N as u32));
        }
        let (lo, hi) = f.cam.flat_bounds();
        let key = ([lo.x.to_bits(), lo.y.to_bits(), hi.x.to_bits(), hi.y.to_bits()], self.version);
        if !covered && self.map.key != Some(key) && (hi - lo).is_finite() {
            self.map.key = Some(key);
            let n = MAP_N;
            let m = &mut self.map;
            m.xs.resize(n * n, 0.0);
            m.ys.resize(n * n, 0.0);
            m.zs.resize(n * n, 0.0);
            for j in 0..n {
                for i in 0..n {
                    m.xs[j * n + i] = lo.x + (hi.x - lo.x) * i as f32 / (n - 1) as f32;
                    m.ys[j * n + i] = lo.y + (hi.y - lo.y) * j as f32 / (n - 1) as f32;
                }
            }
            self.expr.eval_batch(&[&m.xs, &m.ys], &mut m.zs, &mut m.batch);
            if let Some(s) = &mut m.surface {
                for (k, v) in s.data_mut().iter_mut().enumerate() {
                    *v = [m.xs[k], m.ys[k], -0.5, m.zs[k]];
                }
            }
        }
        let levels = &self.levels;
        if let Some(s) = &mut self.map.surface {
            let st = &mut s.style;
            st.lit = 0.0;
            st.cmap = Some(Cmap::Slate);
            // (the highest heights short of the map's palest end, which would wash out
            // what is drawn over them)
            let span = levels.hi - levels.lo;
            st.range = [levels.lo - 0.15 * span, levels.hi + span];
            st.contour = levels.step;
            st.contour_alpha = 0.55;
            st.line_color = Color::hex(0xD6E4F0);
            st.opacity = 0.55 * self.fade;
            if self.fade > 0.0 {
                f.surface(s);
            }
        }
    }

    /// Everything drawn in the plane, and the names written beside it. Returns the boxes
    /// of the heading and of the caption, which are written over it.
    fn picture(&mut self, f: &mut Frame, lay: &Layout, covered: bool) -> [Option<(Vec2, Vec2)>; 2] {
        // (the batches of the parts whose number of points changes, in the same order
        // every frame)
        for part in [Part::Level, Part::Curve] {
            lines_for(f, part);
        }
        let mut plates = [None, None];
        let shown = !self.blank && self.fade > 0.0;
        let alpha = self.fade;
        let Setting { c, .. } = self.glide.shown;
        let h = self.at_point();
        let k = self.scale_shown;
        let (lo, hi) = f.cam.flat_bounds();
        let mut room = Room::new(f);
        if let Some(b) = &lay.board {
            room.wall(b.min - Vec2::splat(4.0), b.max + Vec2::splat(4.0));
        }

        // the heading, at the top of the picture, and the caption at its foot
        let heading_at = lay.picture.0 + Vec2::new(16.0, 14.0);
        let mut size = 15.0;
        let text = self.heading();
        let room_across = lay.picture.1.x - lay.picture.0.x - 32.0;
        while f.text_size(&text, size).x > room_across && size > 11.0 {
            size -= 1.0;
        }
        let wide = f.text_size(&text, size).x;
        f.caption_px(heading_at, text, size, pal::TEXT, Align2::LEFT_TOP);
        plates[0] = Some((heading_at - Vec2::splat(6.0), heading_at + Vec2::new(wide + 12.0, size + 10.0)));
        if !lay.narrow && f.view_size().y >= 300.0 {
            let parts = ["white: the contour line through the point", "pink: the part of $\\nabla f$ along the path"];
            let line = format!("{}      {}      {}", parts[0], parts[1], scale_text(self.scale));
            let (left, bottom) = (lay.picture.0.x + 18.0, lay.picture.1.y - 14.0);
            let w = f.text_size(&line, 12.0).x;
            let line = if w <= lay.picture.1.x - lay.picture.0.x - 36.0 { line } else { scale_text(self.scale) };
            let w = f.text_size(&line, 12.0).x;
            if w <= lay.picture.1.x - lay.picture.0.x - 36.0 {
                f.caption_px(Vec2::new(left, bottom), line, 12.0, pal::DIM, Align2::LEFT_BOTTOM);
                plates[1] = Some((Vec2::new(left - 8.0, bottom - 20.0), Vec2::new(left + w + 8.0, bottom + 6.0)));
            }
        }
        for plate in plates.iter().flatten() {
            room.wall(plate.0, plate.1);
        }

        // the contour line through the point
        if shown && h.f.is_finite() {
            let bounds = [lo.x.to_bits(), lo.y.to_bits(), hi.x.to_bits(), hi.y.to_bits()];
            let key = (bounds, self.version, (h.f as f32).to_bits());
            if self.level_key != Some(key) && self.map.key.is_some_and(|m| m.0 == bounds) {
                self.level_key = Some(key);
                let m = &self.map;
                self.level_lines = contour(&m.xs, &m.ys, &m.zs, MAP_N, h.f as f32);
            }
            let rebuilt = self.level_key != self.level_drawn;
            self.level_drawn = self.level_key;
            let batch = lines_for(f, Part::Level);
            for line in &self.level_lines {
                batch.polyline(line.iter().map(|p| p.extend(0.0)), 1.6, pal::WHITE.a(0.85 * alpha));
            }
            // (found afresh, the line's points are in another order, and as many of them
            // only by chance: then one more, out of sight, so that it is never taken for
            // the same points moved)
            if rebuilt && batch.pts.len() == self.level_count {
                batch.push(Vec3::ZERO, 0.0, Color::CLEAR);
            }
            self.level_count = batch.pts.len();
            for line in &self.level_lines {
                for pair in line.windows(2).step_by(3) {
                    room.line(f, pair[0].extend(0.0), pair[1].extend(0.0));
                }
            }
        }
        keep_batch(lines_for(f, Part::Level));

        // the region g ≤ c, shaded
        let ring: Vec<DVec2> = if self.kind.closed() { (0..240).map(|i| self.kind.at(c, self.kind.sample_t(i, 240))).collect() } else { Vec::new() };
        if shown && self.inside && ring.iter().all(|p| p.is_finite()) {
            for i in 0..ring.len() {
                let (a, b) = (ring[i].as_vec2(), ring[(i + 1) % ring.len()].as_vec2());
                f.dl.tris_alpha.tri(Vec3::ZERO, a.extend(0.0), b.extend(0.0), CURVE.a(0.10 * alpha));
            }
        }

        // the curve g = c
        if shown {
            let batch = lines_for(f, Part::Curve);
            if self.kind.closed() {
                let pts: Vec<Vec3> = ring.iter().map(|p| p.as_vec2().extend(0.0)).collect();
                batch.closed(&pts, 2.8, CURVE.a(alpha));
            } else {
                let pts = (0..=200).map(|i| self.kind.at(c, -REACH + 2.0 * REACH * i as f64 / 200.0).as_vec2().extend(0.0));
                batch.polyline(pts, 2.8, CURVE.a(alpha));
            }
            for i in 0..ring.len() / 6 {
                room.line(f, ring[6 * i].as_vec2().extend(0.0), ring[(6 * i + 6) % ring.len()].as_vec2().extend(0.0));
            }
            if !self.kind.closed() {
                room.line(f, self.kind.at(c, -REACH).as_vec2().extend(0.0), self.kind.at(c, REACH).as_vec2().extend(0.0));
            }
        }
        keep_batch(lines_for(f, Part::Curve));

        if !shown {
            return plates;
        }

        // the tops and bottoms along the curve, and the best point
        for s in &self.along.stationary {
            let p = s.at.p.as_vec2();
            if p.is_finite() {
                f.dot(p, 9.0, if s.top { TOP } else { BOTTOM }.a(alpha));
                if let Some(q) = on_screen(f, p.extend(0.0)) {
                    room.boxed(q - Vec2::splat(5.0), q + Vec2::splat(5.0));
                }
            }
        }
        let best_at = match self.best {
            Some(Best::Edge { at, .. }) => Some((at.p, false)),
            Some(Best::Inside { p, .. }) => Some((p, true)),
            _ => None,
        };
        for p in self.highest_points() {
            f.dot(p.as_vec2(), 14.0, TOP.a(alpha));
        }
        if let Some((p, true)) = best_at.filter(|b| b.0.is_finite()) {
            f.dot(p.as_vec2(), 14.0, TOP.a(alpha));
            f.dot(p.as_vec2(), 6.0, Color::hex(0x0A0D16).a(0.8 * alpha));
        }
        // the hills' tops inside the region (one of them may be the best point)
        if self.inside {
            for (p, _) in self.tops.iter().filter(|(p, _)| self.kind.g(*p) < c) {
                if Some(*p) != best_at.map(|b| b.0) && p.is_finite() {
                    f.dot(p.as_vec2(), 8.0, TOP.a(0.6 * alpha));
                }
            }
        }

        // the arrows at the point: ∇g, the part of ∇f along the curve, and ∇f
        let p = h.p.as_vec2();
        let arrow = |v: DVec2| (v * k).as_vec2();
        let tips = [p + arrow(h.grad_g), p + arrow(h.tangent() * h.slope), p + arrow(h.grad_f)];
        let ok = p.is_finite() && tips.iter().all(|t| t.is_finite());
        if ok {
            let [g_tip, along_tip, f_tip] = tips;
            let visible = |a: Vec2, b: Vec2| a.distance(b) > 1e-4 * (hi - lo).length();
            // (the yellow arrow widest and underneath, so that one lying along it shows too)
            if visible(p, f_tip) {
                f.arrow(p, f_tip, 4.0, GRAD_F);
            }
            if visible(p, along_tip) {
                f.dashed(along_tip, f_tip, 1.4, GRAD_F.a(0.7));
                f.arrow(p, along_tip, 2.2, ALONG);
            }
            if visible(p, g_tip) {
                f.arrow(p, g_tip, 1.8, CURVE);
            }
            for tip in tips {
                room.line(f, p.extend(0.0), tip.extend(0.0));
            }
            room.line(f, along_tip.extend(0.0), f_tip.extend(0.0));
            if let Some(q) = on_screen(f, p.extend(0.0)) {
                room.boxed(q - Vec2::splat(9.0), q + Vec2::splat(9.0));
            }
        }
        if covered {
            return plates;
        }

        // the names: of the arrows, of the curve, of the highest point
        let away = |d: Vec2| Vec2::new(d.x, -d.y);
        let [g_tip, along_tip, f_tip] = tips;
        let long = |f: &Frame, a: Vec2, b: Vec2| ok && on_screen(f, a.extend(0.0)).zip(on_screen(f, b.extend(0.0))).is_some_and(|(a, b)| a.distance(b) > 14.0);
        let wanted = [long(f, p, f_tip), long(f, p, g_tip), long(f, p, along_tip)];
        steady_label(&mut room, f, "grad f", f_tip, r"$\nabla f$", 15.0, GRAD_F, away(f_tip - p), 6.0, wanted[0]);
        steady_label(&mut room, f, "grad g", g_tip, r"$\nabla g$", 15.0, CURVE, away(g_tip - p), 6.0, wanted[1]);
        steady_label(&mut room, f, "along", along_tip, "along the path", 12.0, ALONG, away(along_tip - p), 6.0, wanted[2]);
        let handle = self.kind.at(c, self.kind.handle_t()).as_vec2();
        let name = format!("${} = {}$", self.kind.g_text(), num(c, 2));
        let out = if self.kind.closed() { handle } else { NORMAL.as_vec2() };
        let mut moving = false;
        fading_label(&mut room, f, handle, &name, 13.0, CURVE, away(out), 10.0, &mut self.name_alpha[0], &mut moving);
        match self.best {
            Some(Best::Edge { at, .. }) => {
                let out = if self.kind.closed() { at.p.as_vec2() } else { NORMAL.as_vec2() };
                fading_label(&mut room, f, at.p.as_vec2(), "highest", 12.0, TOP, away(out), 10.0, &mut self.name_alpha[1], &mut moving);
            }
            Some(Best::Inside { p, .. }) => {
                fading_label(&mut room, f, p.as_vec2(), "highest: the hill's top", 12.0, TOP, Vec2::new(0.0, 1.0), 9.0, &mut self.name_alpha[1], &mut moving);
            }
            _ => {}
        }
        // the heights of the contour lines, each where it was put, written where it is
        // clear of all that and fading where something comes over it
        if self.label_alpha.len() != self.levels.labels.len() {
            self.label_alpha = vec![0.0; self.levels.labels.len()];
        }
        let k = 1.0 - (-f.real_dt.max(0.0) / 0.12).exp();
        self.labels_moving = moving;
        for (i, (p, v)) in self.levels.labels.iter().enumerate() {
            let Some(at) = on_screen(f, p.as_vec2().extend(0.0)).filter(|a| a.is_finite()) else { continue };
            let text = level_text(*v, self.levels.step);
            let half = Vec2::new(text_width(&text, 11.0) * 0.5 + 9.0, 10.0);
            let clear = room.clearance(at - half, at + half) >= 2.0 && inside_rect(at, (lay.picture.0, lay.picture.1));
            if clear {
                room.boxed(at - half, at + half);
            }
            let to = if clear { 1.0 } else { 0.0 };
            let a = &mut self.label_alpha[i];
            *a += (to - *a) * k;
            if (to - *a).abs() < 0.01 {
                *a = to;
            } else {
                self.labels_moving = true;
            }
            f.caption_px(at, text, 11.0, pal::TEXT.a(0.8 * alpha * *a), Align2::CENTER_CENTER);
        }
        plates
    }

    /// The panel on the picture: the numbers at the point, f along the curve, the highest
    /// point, and f* against c.
    fn board(&self, f: &mut Frame, b: &Board) {
        hud_panel(f, b.min, b.max);
        // (in the frame a change is made, the board's charts are as blank as the picture:
        // what would be left of them would be taken for the same lines moved)
        if self.blank {
            return;
        }
        let h = self.at_point();
        let small = 12.0;
        let shown = !self.blank && self.fade > 0.05;
        let a = self.fade;
        let value = |v: f64| num(v, 3);

        // the point
        let at = if h.p.is_finite() { format!("${}$", tuple(&[h.p.x, h.p.y], 3)) } else { "none".into() };
        f.text_px(b.point, format!("at the point {at}: $f = {}$", value(h.f)), small, pal::TEXT, Align2::LEFT_TOP);
        let split = if !h.slope.is_finite() {
            r"$\nabla f$ has no value here".to_string()
        } else if h.stationary() {
            let what = if self.along.flat {
                "$f$ does not change along the path"
            } else {
                match self.along.stationary.iter().min_by(|x, y| (x.at.t - h.t).abs().total_cmp(&(y.at.t - h.t).abs())) {
                    Some(s) if s.top => "a top",
                    Some(_) => "a bottom",
                    None => "stationary",
                }
            };
            format!(r"$\nabla f = {}\,\nabla g$: {what}", tint(LAMBDA, &value(h.lambda)))
        } else {
            let sign = if h.slope < 0.0 { "-" } else { "+" };
            format!(r"$\nabla f = {}\,\nabla g {sign} {}\,T$", tint(LAMBDA, &value(h.lambda)), tint(ALONG, &value(h.slope.abs())))
        };
        f.text_px(b.point + Vec2::new(0.0, LINE_H + 1.0), split, small, pal::TEXT, Align2::LEFT_TOP);

        // f along the curve
        let path = if self.inside { "its edge" } else { "the path" };
        let axis = if self.kind.closed() { r"$\theta$" } else { "$s$" };
        f.text_px(b.along_title, format!("$f$ along {path}, against {axis}"), 11.5, pal::TEXT, Align2::LEFT_TOP);
        let [t0, t1] = self.kind.t_range();
        let (x0, x1) = if self.kind.closed() { (-180.0, 180.0) } else { (t0, t1) };
        let to_x = |t: f64| if self.kind.closed() { t.to_degrees() } else { t };
        let chart = Plot::new(b.along, [x0, x1], [self.ranges.get(0) as f64, self.ranges.get(1) as f64]);
        if !chart.tiny() {
            y_ticks(f, &chart);
            let (min, max) = b.along;
            plot::line_of(f, plot::AXES, Vec2::new(min.x, max.y), max, 1.0, pal::DIM.a(0.8));
            let ticks: Vec<f64> = if self.kind.closed() { vec![-180.0, -90.0, 0.0, 90.0, 180.0] } else { vec![-4.0, -2.0, 0.0, 2.0, 4.0] };
            for v in ticks {
                let x = chart.px_x(v);
                plot::line_of(f, plot::AXES, Vec2::new(x, max.y), Vec2::new(x, max.y + 4.0), 1.0, pal::DIM.a(0.8));
                let text = if self.kind.closed() { format!("{}°", num(v, 0)) } else { num(v, 0) };
                f.text_px(Vec2::new(x, max.y + 6.0), text, 10.5, pal::DIM, Align2::CENTER_TOP);
            }
            f.text_px(Vec2::new(max.x, max.y + 18.0), axis, 11.0, pal::DIM, Align2::RIGHT_TOP);
            if shown {
                // (every other sample is plenty for a chart this size)
                let pts: Vec<(f64, f64)> = self.along.samples.iter().step_by(2).map(|s| (to_x(s.t), s.f)).collect();
                let pts = if self.kind.closed() { pts.into_iter().chain(self.along.samples.first().map(|s| (180.0, s.f))).collect() } else { pts };
                chart.path(f, ALONG_CURVE, pts, 1.8, CURVE.a(a));
                for s in &self.along.stationary {
                    chart.dot(f, to_x(s.at.t), s.at.f, 8.0, if s.top { TOP } else { BOTTOM }.a(a));
                }
                for s in self.highest_stationary() {
                    chart.dot(f, to_x(s.t), s.f, 12.0, TOP.a(a));
                }
                if h.f.is_finite() {
                    let x = to_x(if self.kind.closed() { wrap(h.t) } else { h.t });
                    chart.vline(f, plot::MARKS, x, 1.0, pal::WHITE.a(0.5), false);
                }
            }
        }

        // the highest point
        let lambda = self.best_lambda();
        let line = match self.best {
            Some(Best::Edge { at, ties }) if ties > 1 => format!(r"highest: $f^* = {}$, at {ties} points, $\lambda = {}$", value(at.f), tint(LAMBDA, &value(at.lambda))),
            Some(Best::Edge { at, .. }) => format!(r"highest: $f^* = {}$, $\lambda = {}$", value(at.f), tint(LAMBDA, &value(at.lambda))),
            Some(Best::Inside { f: v, .. }) => format!(r"highest: $f^* = {}$ inside, $\lambda = 0$", value(v)),
            Some(Best::Everywhere { f: v }) => format!(r"$f = {}$ all along: every point is highest", value(v)),
            None => "no highest point on the stretch drawn".to_string(),
        };
        f.text_px(b.highest, line, small, TOP, Align2::LEFT_TOP);
        plot::button(f, b.go, "go there", 11.5, false);

        // f* against c
        let region = if self.inside { if self.kind == Kind::Circle { "in the disc" } else { "in the ellipse" } } else { "on the path" };
        f.text_px(b.best_title, format!("the highest $f$ {region}, $f^*$, against $c$"), 11.5, pal::TEXT, Align2::LEFT_TOP);
        let [c0, c1] = self.kind.c_range();
        let chart = Plot::new(b.best, [c0, c1], [self.ranges.get(2) as f64, self.ranges.get(3) as f64]);
        let c = self.glide.shown.c;
        if !chart.tiny() {
            y_ticks(f, &chart);
            chart.x_axis(f, "", 10.5);
            let (_, max) = b.best;
            f.text_px(Vec2::new(max.x, max.y + 18.0), "$c$", 11.0, pal::DIM, Align2::RIGHT_TOP);
            if shown {
                let p = &self.profile;
                if self.inside {
                    let pts = p.cs.iter().zip(&p.on_curve).map(|(c, v)| (*c, *v));
                    dashed_path(f, &chart, BEST_EDGE, pts, 1.4, TOP.a(0.5 * a));
                } else {
                    keep_batch(plot::apart(f, BEST_EDGE, false));
                }
                chart.path(f, BEST_CURVE, p.cs.iter().zip(&p.best).map(|(c, v)| (*c, *v)), 2.0, TOP.a(a));
                if let (Some(best), Some(l)) = (self.best, lambda) {
                    // the line of slope λ through the point (c, f*(c)): it touches the curve
                    let w = 0.18 * (c1 - c0);
                    chart.path(f, TANGENT, [(c - w, best.f() - l * w), (c + w, best.f() + l * w)], 1.6, LAMBDA.a(a));
                    chart.dot(f, c, best.f(), 9.0, TOP.a(a));
                } else {
                    keep_batch(plot::apart(f, TANGENT, false));
                }
                // where c was taken hold of, and the line of slope λ from there
                match self.since.filter(|s| s.best.is_finite() && s.lambda.is_finite() && (s.c - c).abs() > 1e-9) {
                    Some(s) => {
                        chart.path(f, SINCE, [(s.c, s.best), (c, s.best + s.lambda * (c - s.c))], 1.2, LAMBDA.a(0.6 * a));
                        chart.dot(f, s.c, s.best, 7.0, pal::WHITE.a(0.7 * a));
                    }
                    None => keep_batch(plot::apart(f, SINCE, false)),
                }
            }
        }
        let slope = match (self.slope, lambda) {
            (Some(s), Some(l)) => format!(r"slope of $f^*$ at $c = {}$: ${}$; $\lambda = {}$", num(c, 2), value(s), tint(LAMBDA, &value(l))),
            (Some(s), None) => format!(r"slope of $f^*$ at $c = {}$: ${}$", num(c, 2), value(s)),
            _ => format!(r"no slope of $f^*$ at $c = {}$", num(c, 2)),
        };
        f.text_px(b.slope, slope, small, pal::TEXT, Align2::LEFT_TOP);
        if let (Some(s), Some(best)) = (self.since.filter(|s| (s.c - c).abs() > 5e-3 && s.best.is_finite() && s.lambda.is_finite()), self.best) {
            let rise = best.f() - s.best;
            let moved = c - s.c;
            let way = if rise >= 0.0 { "rose" } else { "fell" };
            let text = format!(r"from $c = {}$: $f^*$ {way} by ${}$; $\lambda \times {} = {}$", num(s.c, 2), value(rise.abs()), num(moved, 2), value(s.lambda * moved));
            f.text_px(b.since, text, small, pal::DIM, Align2::LEFT_TOP);
        }
    }
}

/// A dashed polyline on a chart: every other run of three samples.
fn dashed_path(f: &mut Frame, chart: &Plot, kind: u8, pts: impl Iterator<Item = (f64, f64)>, width: f32, color: Color) {
    let pts: Vec<Vec2> = pts.map(|(x, y)| chart.at(x, y)).collect();
    let mut dashes = Vec::with_capacity(pts.len() * 2);
    for (i, p) in pts.iter().enumerate() {
        if i % 6 == 3 {
            dashes.push(Vec2::NAN);
        }
        if i % 6 < 3 || i % 6 == 0 {
            dashes.push(*p);
        }
    }
    plot::path_of(f, kind, &plot::clip(&dashes, chart.rect), width, color);
    keep_batch(plot::apart(f, kind, false));
}

/// The line up a chart's left side with three or four round values marked on it, and
/// faint lines across at them. (The chart kit's own axis marks one value every sixty
/// pixels: on charts this short, often only one.)
fn y_ticks(f: &mut Frame, chart: &Plot) {
    let (min, max) = chart.rect;
    plot::line_of(f, plot::AXES, min, Vec2::new(min.x, max.y), 1.0, pal::DIM.a(0.8));
    let [lo, hi] = chart.y;
    let step = plot::round_step((hi - lo) / 3.5);
    let decimals = (-(step * (1.0 + 1e-9)).log10().floor()).max(0.0).min(6.0) as usize;
    let first = (lo / step - 1e-9).ceil() as i64;
    let last = ((hi / step + 1e-9).floor() as i64).min(first + 8);
    for k in first..=last {
        let v = k as f64 * step;
        let y = chart.px_y(v);
        plot::line_of(f, plot::AXES, Vec2::new(min.x, y), Vec2::new(max.x, y), 1.0, pal::DIM.a(0.14));
        plot::line_of(f, plot::AXES, Vec2::new(min.x - 4.0, y), Vec2::new(min.x, y), 1.0, pal::DIM.a(0.8));
        f.text_px(Vec2::new(min.x - 6.0, y), num(v, decimals), 10.5, pal::DIM, Align2::RIGHT_CENTER);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn expr(text: &str) -> Expr {
        Expr::parse(text, &VARS).unwrap_or_else(|e| panic!("{text}: {e}"))
    }

    fn sqrt5() -> f64 {
        5f64.sqrt()
    }

    /// Bring the scene to rest at 60 frames a second (at most `seconds`).
    fn settle(scene: &mut Lagrange, seconds: f32) {
        for _ in 0..(60.0 * seconds) as usize {
            scene.advance(1.0 / 60.0, 1.0 / 60.0);
            if scene.at_rest() {
                break;
            }
        }
        assert!(
            scene.at_rest(),
            "the scene has not come to rest: {:?} {:?} film {:?} waiting {} change {} fade {} scale {} {} ranges {:?} {}",
            scene.glide.to,
            scene.glide.shown,
            scene.film,
            scene.waiting,
            scene.change.is_some(),
            scene.fade,
            scene.scale,
            scene.scale_shown,
            scene.ranges.shown,
            scene.ranges_moving
        );
    }

    /// The scene as experiment `k`'s button leaves it once its film has played: pressed
    /// after the opening film, as a visitor would.
    fn run(k: usize) -> Lagrange {
        let mut scene = Lagrange::new();
        settle(&mut scene, 20.0);
        scene.set_up(k);
        settle(&mut scene, 20.0);
        scene
    }

    /// The scene from experiment `k`'s button until its film has played, with the point
    /// as drawn where the film starts and in every frame of it.
    fn film(k: usize) -> (Lagrange, Vec<Here>) {
        let mut scene = Lagrange::new();
        settle(&mut scene, 20.0);
        scene.set_up(k);
        let mut frames = Vec::new();
        for _ in 0..60 * 20 {
            let before = scene.at_point();
            scene.advance(1.0 / 60.0, 1.0 / 60.0);
            if let Some(film) = scene.film.filter(|_| !scene.waiting) {
                if frames.is_empty() {
                    // (the film has taken its first step: where it started, which is
                    // where the point was drawn until then, give or take the last hair
                    // of the glide there)
                    let start = here(&scene.expr, scene.kind, film.from.c, film.from.place);
                    assert!((start.p - before.p).length() < 1e-4, "{start:?} {before:?}");
                    frames.push(start);
                }
                frames.push(scene.at_point());
            }
            if scene.at_rest() {
                break;
            }
        }
        assert!(scene.at_rest() && frames.len() > 60, "{} frames of film", frames.len());
        (scene, frames)
    }

    fn three(v: f64) -> String {
        num(v, 3)
    }

    /// The gradient worked out in double precision is the formula's: against central
    /// differences, for every preset and for formulas that use every kind of function.
    #[test]
    fn the_gradient_is_exact() {
        let mut formulas: Vec<&str> = PRESETS.iter().map(|p| p.1).collect();
        formulas.extend([
            "x^3 y - 2 sin(x y)",
            "atan2(y, x + 3)",
            "sqrt(x^2 + y^2 + 1)",
            "(x + 3)^2.5 / (1 + y^2)",
            "tanh(x) + sigmoid(y) + cosh(0.3 x) - sinh(0.2 y)",
            "min(x, y) + max(x, y)^2",
            "exp(-x^2) ln(2 + y^2) + log10(3 + x) + log2(3 + y)",
            "gauss(x) + cos(y) + tan(0.3 x) + atan(y)",
            "asin(0.3 x) + acos(0.3 y)",
            "2^x + x^y + 3",
        ]);
        let mut rng = Rng::new(11);
        for text in formulas {
            let e = expr(text);
            for _ in 0..40 {
                let p = DVec2::new(rng.range(-2.0, 2.0) as f64, rng.range(0.1, 2.0) as f64);
                let d = height(&e, p);
                let h = 1e-6;
                let fd = DVec2::new(
                    (height(&e, p + DVec2::X * h).v - height(&e, p - DVec2::X * h).v) / (2.0 * h),
                    (height(&e, p + DVec2::Y * h).v - height(&e, p - DVec2::Y * h).v) / (2.0 * h),
                );
                if !(d.v.is_finite() && fd.is_finite()) {
                    continue;
                }
                assert!((d.grad() - fd).length() <= 1e-6 * (1.0 + d.grad().length()), "{text} at {p}: {:?} against {fd}", d.grad());
                // and its value is the single-precision formula's
                let single = e.eval(&[p.x as f32, p.y as f32]) as f64;
                assert!((single - d.v).abs() <= 1e-4 * (1.0 + d.v.abs()), "{text} at {p}: {single} against {}", d.v);
            }
        }
    }

    /// Each curve is g = c at every place along it; ∇g is g's gradient and is at right
    /// angles to the curve; the place grows the way the unit arrow T points; and the
    /// place of a point of the curve is the place it came from. The c handle puts the
    /// curve through where it is dragged.
    #[test]
    fn the_curves_are_what_they_say() {
        for kind in KINDS {
            let [lo, hi] = kind.c_range();
            for c in [lo, 0.5 * (lo + hi), hi, 1.0] {
                for i in 0..36 {
                    let t = kind.sample_t(i, 36) * 0.98 + 0.01;
                    let p = kind.at(c, t);
                    assert!((kind.g(p) - c).abs() < 1e-12 * (1.0 + c.abs()), "{kind:?} at c = {c}, t = {t}");
                    let h = 1e-6;
                    let fd = DVec2::new(kind.g(p + DVec2::X * h) - kind.g(p - DVec2::X * h), kind.g(p + DVec2::Y * h) - kind.g(p - DVec2::Y * h)) / (2.0 * h);
                    assert!((kind.grad(p) - fd).length() < 1e-6 * (1.0 + fd.length()), "{kind:?}");
                    let velocity = (kind.at(c, t + h) - kind.at(c, t - h)) / (2.0 * h);
                    let tangent = kind.grad(p).perp().normalize();
                    assert!(velocity.dot(kind.grad(p)).abs() < 1e-6 * velocity.length() * kind.grad(p).length() && velocity.dot(tangent) > 0.0, "{kind:?}");
                    assert!((kind.place_of(c, p) - t).abs() < 1e-9, "{kind:?}: {} against {t}", kind.place_of(c, p));
                }
                let handle = kind.at(c, kind.handle_t());
                assert!((kind.c_through(handle) - c).abs() < 1e-9 * (1.0 + c.abs()), "{kind:?}");
                // ∇g is never 0 on the curves the scene draws: the method always applies
                assert!((0..36).all(|i| kind.grad(kind.at(c, kind.sample_t(i, 36))).length() > 1e-3));
            }
            // c is kept within its range, a c that is not a number too
            assert!(kind.clamp_c(1e9) == hi && kind.clamp_c(-1e9) == lo && kind.clamp_c(f64::NAN).is_finite());
        }
    }

    /// The worked example: f = x + 2y on the circle x² + y² = 1 has its top at
    /// (1, 2)/√5, where f = √5 and λ = √5/2, and its bottom opposite, with λ = −√5/2.
    #[test]
    fn the_worked_example_comes_out() {
        let a = along(&expr("x + 2y"), Kind::Circle, 1.0, SAMPLES);
        assert_eq!(a.stationary.len(), 2);
        let top = a.stationary.iter().find(|s| s.top).unwrap().at;
        let bottom = a.stationary.iter().find(|s| !s.top).unwrap().at;
        assert!((top.p - DVec2::new(1.0, 2.0) / sqrt5()).length() < 1e-12, "{:?}", top.p);
        assert!((top.f - sqrt5()).abs() < 1e-12 && (top.lambda - sqrt5() / 2.0).abs() < 1e-12);
        assert!((bottom.p + DVec2::new(1.0, 2.0) / sqrt5()).length() < 1e-12 && (bottom.lambda + sqrt5() / 2.0).abs() < 1e-12);
        assert!((top.grad_f - top.grad_g * top.lambda).length() < 1e-12);
        assert_eq!(a.highest.map(|h| h.p), Some(top.p));
        assert_eq!(a.ties, 1);
    }

    /// Every stationary point the scene finds, for every preset on every curve at several
    /// c, has ∇f = λ∇g, and is a top or a bottom of f along the curve as it says; the
    /// highest point is the highest of very many places along the curve; and between two
    /// neighbouring stationary points f only rises or only falls (none is missed).
    #[test]
    fn every_stationary_point_has_its_gradients_in_line() {
        for (_, text, _) in PRESETS {
            let e = expr(text);
            for kind in KINDS {
                let [lo, hi] = kind.c_range();
                for c in [lo, 0.3 * lo + 0.7 * hi, 0.5 * (lo + hi), hi, kind.usual_c()] {
                    let a = along(&e, kind, c, SAMPLES);
                    for s in &a.stationary {
                        let h = s.at;
                        assert!((h.grad_f - h.grad_g * h.lambda).length() <= 1e-9 * (1.0 + h.grad_f.length()), "{text} on the {kind:?} at c = {c}: {h:?}");
                        let side = |dt: f64| here(&e, kind, c, h.t + dt).f;
                        let dt = 1e-3;
                        if s.top {
                            assert!(side(dt) <= h.f && side(-dt) <= h.f, "{text}, {kind:?}, {c}: not a top at {}", h.t);
                        } else {
                            assert!(side(dt) >= h.f && side(-dt) >= h.f, "{text}, {kind:?}, {c}: not a bottom at {}", h.t);
                        }
                    }
                    if a.flat {
                        continue;
                    }
                    // the stationary points alternate, top and bottom
                    let mut sorted = a.stationary.clone();
                    sorted.sort_by(|x, y| x.at.t.total_cmp(&y.at.t));
                    for pair in sorted.windows(2) {
                        assert!(pair[0].top != pair[1].top, "{text}, {kind:?}, {c}: two {} in a row", if pair[0].top { "tops" } else { "bottoms" });
                    }
                    // the highest is the highest of 20 000 places
                    let fine = (0..20_000).map(|i| here(&e, kind, c, kind.sample_t(i, 20_000)).f).filter(|v| v.is_finite()).fold(f64::NEG_INFINITY, f64::max);
                    match a.highest {
                        Some(h) => assert!(h.f >= fine - 1e-9 * (1.0 + fine.abs()), "{text}, {kind:?}, {c}: {} against {fine}", h.f),
                        // (on a line, only where f still rises at an end of the stretch)
                        None => assert!(!kind.closed()),
                    }
                }
            }
        }
    }

    /// The multiplier is the slope of the highest value: dλ... d f*/dc, by differences,
    /// is λ at the highest point, for every preset on every curve, wherever there is one
    /// highest point (the two hills' f* has a corner where the highest point jumps from
    /// one top to the other: c near it is left out).
    #[test]
    fn the_multiplier_is_the_slope_of_the_highest_value() {
        let mut checked = 0;
        for (_, text, _) in PRESETS {
            let e = expr(text);
            let tops = hilltops(&e);
            for kind in KINDS {
                let [lo, hi] = kind.c_range();
                for i in 1..12 {
                    let c = lo + (hi - lo) * i as f64 / 12.0;
                    let a = along(&e, kind, c, SAMPLES);
                    let Some(Best::Edge { at, ties }) = best_of(&a, &tops, kind, c, false) else { continue };
                    let slope = slope_of_best(&e, &tops, kind, c, false, &a).unwrap();
                    // (and the same found afresh, from f* a step either side)
                    let step = 1e-4 * (hi - lo);
                    let afresh = (best_value(&e, &tops, kind, c + step, false, SAMPLES).unwrap() - best_value(&e, &tops, kind, c - step, false, SAMPLES).unwrap()) / (2.0 * step);
                    assert!((afresh - at.lambda).abs() <= 1e-5 * (1.0 + at.lambda.abs()), "{text} on the {kind:?} at c = {c}: f* found afresh has slope {afresh}, λ {}", at.lambda);
                    // (a second top nearly as high: the highest point may jump between them)
                    let rival = a.stationary.iter().any(|s| s.top && (s.at.t - at.t).abs() > 1e-6 && (at.f - s.at.f).abs() < 0.05 * (1.0 + at.f.abs()));
                    if rival && ties == 1 {
                        continue;
                    }
                    assert!((slope - at.lambda).abs() <= 1e-5 * (1.0 + at.lambda.abs()), "{text} on the {kind:?} at c = {c}: slope {slope}, λ {}", at.lambda);
                    checked += 1;
                }
            }
        }
        assert!(checked > 60, "only {checked} checked");
    }

    /// With g ≤ c the best point is the dome's top, with λ = 0, once the top is inside
    /// (c > 0.34); before that it is on the edge, with λ > 0, as it must be at a highest
    /// point on the edge. On the edge alone λ is negative once the top is inside.
    #[test]
    fn an_inequality_binds_only_with_a_multiplier_above_zero() {
        let e = expr(PRESETS[DOME].1);
        let tops = hilltops(&e);
        assert_eq!(tops.len(), 1);
        // (the formula's numbers are single-precision ones: 0.3 is 0.30000001)
        assert!((tops[0].0 - DVec2::new(0.5, 0.3)).length() < 1e-7 && (tops[0].1 - 1.0).abs() < 1e-12, "{tops:?}");
        let [lo, hi] = Kind::Circle.c_range();
        for i in 0..=40 {
            let c = lo + (hi - lo) * i as f64 / 40.0;
            let a = along(&e, Kind::Circle, c, SAMPLES);
            let best = best_of(&a, &tops, Kind::Circle, c, true).unwrap();
            let edge = a.highest.unwrap();
            if c < 0.34 {
                assert!(matches!(best, Best::Edge { .. }) && edge.lambda > 0.0, "c = {c}: {best:?}");
            } else {
                assert!(matches!(best, Best::Inside { f, .. } if f == 1.0) && edge.lambda < 0.0, "c = {c}: {best:?}");
                assert!(slope_of_best(&e, &tops, Kind::Circle, c, true, &a) == Some(0.0) || c - 0.34 < 0.01);
            }
        }
        // the two hills have their tops where ∇f = 0, the higher one higher
        let hills = hilltops(&expr(PRESETS[HILLS].1));
        assert_eq!(hills.len(), 2, "{hills:?}");
        for (p, v) in &hills {
            assert!(height(&expr(PRESETS[HILLS].1), *p).grad().length() < 1e-9 && *v > 1.3, "{p} {v}");
        }
    }

    /// Where f does not change along the curve, every point is stationary: f = x² + y²
    /// on a circle (∇f = ∇g, λ = 1 everywhere) and f = x + 2y on the line x + 2y = c.
    /// And where f keeps rising along the line, there is no highest point to show.
    #[test]
    fn flat_and_endless_paths_are_said_to_be_so() {
        for (text, kind, c) in [("x^2 + y^2", Kind::Circle, 2.0), ("x + 2y", Kind::Line, 1.5), ("x^2/4 + y^2", Kind::Ellipse, 1.0)] {
            let a = along(&expr(text), kind, c, SAMPLES);
            assert!(a.flat && a.stationary.is_empty(), "{text}");
            assert!(a.samples.iter().all(|h| h.stationary() && (h.lambda - 1.0).abs() < 1e-12), "{text}");
            assert!(matches!(best_of(&a, &[], kind, c, false), Some(Best::Everywhere { f }) if (f - c).abs() < 1e-12));
        }
        let rising = along(&expr("x + y"), Kind::Line, 1.0, SAMPLES);
        assert!(rising.highest.is_none() && rising.stationary.is_empty() && best_of(&rising, &[], Kind::Line, 1.0, false).is_none());
        assert!(slope_of_best(&expr("x + y"), &[], Kind::Line, 1.0, false, &rising).is_none());
        // the nearest point to 0 on the line x + 2y = 2 is (0.4, 0.8): the smallest answer
        let near = along(&expr(PRESETS[4].1), Kind::Line, 2.0, SAMPLES);
        let h = near.highest.unwrap();
        assert!((h.p - DVec2::new(0.4, 0.8)).length() < 1e-9 && (h.f + 0.8).abs() < 1e-9 && (h.lambda + 0.8).abs() < 1e-9, "{h:?}");
    }

    #[test]
    fn slide_to_the_top_does_what_its_caption_says() {
        let (scene, frames) = film(0);
        assert!(scene.drawn == "x + 2y" && scene.kind == Kind::Circle && scene.glide.shown.c == 1.0 && !scene.inside);
        // it starts where the circle runs straight uphill: all of ∇f is along it
        let first = frames[0];
        assert!(first.lambda.abs() < 1e-9 && (first.slope - sqrt5()).abs() < 1e-9, "{first:?}");
        // less and less of ∇f along the circle as the point climbs
        for pair in frames.windows(2) {
            assert!(pair[1].slope <= pair[0].slope + 1e-12 && pair[1].f >= pair[0].f - 1e-12);
        }
        // at the top none: ∇f = λ∇g with λ = √5/2, f = √5, at (0.447, 0.894)
        let h = scene.at_point();
        assert!(h.slope.abs() < 1e-9 && h.stationary());
        assert_eq!([three(h.p.x), three(h.p.y), three(h.f), three(h.lambda)], ["0.447", "0.894", "2.236", "1.118"]);
        assert!((h.f - sqrt5()).abs() < 1e-9 && (h.lambda - sqrt5() / 2.0).abs() < 1e-9);
        // the yellow arrow is λ times as long as the cyan one, the same way
        assert!((h.grad_f.length() / h.grad_g.length() - h.lambda).abs() < 1e-9 && h.grad_f.dot(h.grad_g) > 0.0);
        // the contour line through the point touches the circle there: nowhere else on
        // the circle is f as high
        assert!(scene.along.samples.iter().all(|s| s.f <= h.f + 1e-12 && (s.f < h.f - 1e-6 || (s.p - h.p).length() < 0.01)));
    }

    #[test]
    fn the_price_does_what_its_caption_says() {
        let scene = run(1);
        assert!(scene.drawn == "x + 2y" && scene.kind == Kind::Circle && (scene.glide.shown.c - 1.1).abs() < 1e-12);
        let since = scene.since.unwrap();
        let best = scene.best.unwrap().f();
        assert_eq!([num(since.c, 2), three(since.best), three(best), three(best - since.best)], ["1.00", "2.236", "2.345", "0.109"]);
        assert_eq!([three(since.lambda), three(since.lambda * 0.1)], ["1.118", "0.112"]);
        // the point has stayed at the top, and λ there now is 1.066: the slope of f*
        let h = scene.at_point();
        assert!(h.stationary() && matches!(scene.best, Some(Best::Edge { at, .. }) if (at.p - h.p).length() < 1e-9));
        assert_eq!(three(h.lambda), "1.066");
        assert!((scene.slope.unwrap() - h.lambda).abs() < 1e-6);
        // the board says the same
        assert!((scene.glide.shown.c - since.c - 0.1).abs() < 1e-12);
    }

    #[test]
    fn every_top_and_bottom_does_what_its_caption_says() {
        let (scene, frames) = film(2);
        assert!(scene.drawn == PRESETS[HILLS].1 && scene.kind == Kind::Circle && scene.glide.shown.c == 1.0);
        let st = &scene.along.stationary;
        assert_eq!((st.iter().filter(|s| s.top).count(), st.iter().filter(|s| !s.top).count()), (2, 2));
        // from the highest
        let first = frames[0];
        assert!(first.stationary() && matches!(scene.best, Some(Best::Edge { at, ties: 1 }) if (at.p - first.p).length() < 1e-9));
        assert_eq!([three(first.f), three(first.lambda)], ["1.429", "0.824"]);
        // through one bottom, with the arrows opposite ways
        let passed: Vec<&Stationary> = st.iter().filter(|s| !s.top && frames.windows(2).any(|w| (w[0].slope > 0.0) != (w[1].slope > 0.0) && (wrap(w[0].t - s.at.t)).abs() < 0.1)).collect();
        assert_eq!(passed.len(), 1);
        let bottom = passed[0].at;
        assert!(bottom.lambda < 0.0 && bottom.grad_f.dot(bottom.grad_g) < 0.0);
        assert_eq!(three(bottom.lambda), "−0.124");
        // to the other top: in line again, and lower
        let h = scene.at_point();
        assert!(h.stationary() && h.lambda > 0.0);
        assert_eq!([three(h.lambda), three(h.f)], ["0.585", "0.934"]);
        assert!(st.iter().any(|s| s.top && (s.at.p - h.p).length() < 1e-9));
        // (nothing else is passed: the film goes the short way round, clockwise)
        let moving: Vec<f64> = frames.iter().map(|h| h.slope).filter(|s| s.abs() > 1e-6).collect();
        assert_eq!(moving.windows(2).filter(|w| (w[0] > 0.0) != (w[1] > 0.0)).count(), 1, "the film crosses one stationary point only");
    }

    #[test]
    fn the_inequality_does_what_its_caption_says() {
        let (scene, frames) = film(3);
        assert!(scene.drawn == PRESETS[DOME].1 && scene.inside && scene.kind == Kind::Circle);
        assert!((scene.glide.shown.c - 1.0).abs() < 1e-12);
        // at the start the top (0.5, 0.3) is outside: the point is the best, with λ = 0.506
        let first = frames[0];
        let e = expr(PRESETS[DOME].1);
        let tops = hilltops(&e);
        let start = best_of(&along(&e, Kind::Circle, 0.15, SAMPLES), &tops, Kind::Circle, 0.15, true).unwrap();
        assert!(matches!(start, Best::Edge { at, .. } if (at.p - first.p).length() < 1e-6 && at.lambda > 0.0));
        assert_eq!(three(first.lambda), "0.506");
        assert!(Kind::Circle.g(tops[0].0) > 0.15 && (Kind::Circle.g(tops[0].0) - 0.34).abs() < 1e-7);
        // f* rises with c until 0.34, and is flat after
        let p = &scene.profile;
        for i in 1..p.cs.len() {
            if p.cs[i] < 0.34 {
                assert!(p.best[i] > p.best[i - 1], "{}", p.cs[i]);
            } else if p.cs[i - 1] > 0.34 {
                assert!(p.best[i] == 1.0 && p.best[i - 1] == 1.0);
            }
        }
        // at the end: the best point is the top, inside, with λ = 0, and nothing gained
        assert!(matches!(scene.best, Some(Best::Inside { p, f }) if (p - DVec2::new(0.5, 0.3)).length() < 1e-7 && f == 1.0));
        assert_eq!(scene.best_lambda(), Some(0.0));
        assert_eq!(scene.slope, Some(0.0));
        // on the circle itself: lower, 0.826, with λ = −0.417
        let edge = scene.along.highest.unwrap();
        assert_eq!([three(edge.f), three(edge.lambda)], ["0.826", "−0.417"]);
        // (the point is at that highest point of the circle, in the direction of the
        // dome's top, 0.3 being a hair more in single precision)
        assert!((scene.at_point().p - edge.p).length() < 1e-6);
    }

    #[test]
    fn touching_twice_does_what_its_caption_says() {
        let (scene, frames) = film(4);
        assert!(scene.drawn == "xy" && scene.kind == Kind::Circle && scene.glide.shown.c == 2.0);
        let Some(Best::Edge { at, ties }) = scene.best else { panic!("{:?}", scene.best) };
        assert_eq!(ties, 2);
        assert!((at.f - 1.0).abs() < 1e-12 && (at.lambda - 0.5).abs() < 1e-12);
        let tops: Vec<Here> = scene.highest_stationary();
        assert_eq!(tops.len(), 2);
        for t in &tops {
            assert!(((t.p - DVec2::ONE).length() < 1e-9 || (t.p + DVec2::ONE).length() < 1e-9) && (t.lambda - 0.5).abs() < 1e-12, "{t:?}");
        }
        // from (1, 1) to (−1, −1), through the lowest point (1, −1)
        assert!((frames[0].p - DVec2::ONE).length() < 1e-9 && (scene.at_point().p + DVec2::ONE).length() < 1e-9);
        let low = scene.along.stationary.iter().filter(|s| !s.top).map(|s| s.at).find(|h| (h.p - DVec2::new(1.0, -1.0)).length() < 1e-9).unwrap();
        assert!((low.f + 1.0).abs() < 1e-12 && (low.lambda + 0.5).abs() < 1e-12);
        assert!(frames.iter().any(|h| (h.p - low.p).length() < 0.05) && frames.iter().all(|h| (h.p - DVec2::new(-1.0, 1.0)).length() > 0.5));
    }

    /// The opening state is the first experiment's end, and the scene comes to rest.
    #[test]
    fn the_scene_opens_on_the_worked_example() {
        let mut scene = Lagrange::new();
        settle(&mut scene, 20.0);
        let h = scene.at_point();
        assert!(h.stationary() && (h.f - sqrt5()).abs() < 1e-9 && scene.drawn == "x + 2y");
        assert!(scene.heading().contains("$f = x + 2y$"));
    }

    /// Whatever the formula, the curve or c, nothing the scene works out is a number that
    /// is not one where it is shown, and the scene comes to rest.
    #[test]
    fn degenerate_input_leaves_everything_finite() {
        for text in ["0", "1/0", "sqrt(-1)", "1e30 x", "x/0 + y", "sqrt(1 - x^2 - y^2)", "abs(x) + abs(y)", "floor(3x) + y", "exp(40 x)", "1/(x^2 + y^2)", "sin(30 x) cos(30 y)", "ln(x)"] {
            for kind in KINDS {
                for inside in [false, true] {
                    let mut scene = Lagrange::new();
                    scene.f.set(text);
                    scene.ask(Change { formula: None, kind, inside, setting: None, film: None, since: None });
                    scene.expr = scene.f.expr.clone();
                    scene.drawn = scene.f.text.clone();
                    scene.formula_changed();
                    settle(&mut scene, 20.0);
                    for c in [kind.c_range()[0], kind.c_range()[1], 1.0] {
                        scene.glide.now(|s| s.c = kind.clamp_c(c));
                        scene.glide_to_highest();
                        settle(&mut scene, 20.0);
                        scene.glide_to_lowest();
                        settle(&mut scene, 20.0);
                        let s = scene.glide.shown;
                        assert!(s.c.is_finite() && s.place.is_finite(), "{text}");
                        assert!(scene.scale.is_finite() && scene.scale > 0.0 && scene.scale_shown.is_finite());
                        assert!((0..4).all(|i| scene.ranges.get(i).is_finite()), "{text}");
                        let p = &scene.profile;
                        assert!(p.cs.iter().all(|c| c.is_finite()));
                        let h = scene.at_point();
                        assert!(h.p.is_finite() && h.grad_g.is_finite(), "{text} {kind:?}");
                        let words = scene.heading();
                        assert!(!words.contains("NaN") && !words.contains("inf"), "{words}");
                    }
                }
            }
        }
    }

    /// The contour line found over a grid passes through the places at its height: for
    /// a plane, a straight line; for a bowl, one closed ring.
    #[test]
    fn contour_lines_are_where_f_has_their_height() {
        let n = 61;
        let (mut xs, mut ys) = (Vec::new(), Vec::new());
        for j in 0..n {
            for i in 0..n {
                xs.push(-3.0 + 6.0 * i as f32 / (n - 1) as f32);
                ys.push(-3.0 + 6.0 * j as f32 / (n - 1) as f32);
            }
        }
        for (text, level, pieces) in [("x + 2y", 1.0f32, 1), ("x^2 + y^2", 2.0, 1), ("xy", 1.0, 2), ("x^2 - y^2", 0.5, 2)] {
            let e = expr(text);
            let zs: Vec<f32> = xs.iter().zip(&ys).map(|(x, y)| e.eval(&[*x, *y])).collect();
            let lines = contour(&xs, &ys, &zs, n, level);
            assert_eq!(lines.len(), pieces, "{text}");
            for line in &lines {
                assert!(line.len() > 20);
                for p in line {
                    assert!((e.eval(&[p.x, p.y]) - level).abs() < 0.02 * (1.0 + level), "{text} at {p}");
                }
            }
        }
        // a saddle exactly at a cell's corners, and values that are not numbers, break nothing
        let zs: Vec<f32> = xs.iter().zip(&ys).map(|(x, y)| if x.abs() < 0.5 { f32::NAN } else { x * y }).collect();
        assert!(contour(&xs, &ys, &zs, n, 0.0).iter().flatten().all(|p| p.is_finite()));
        assert!(contour(&xs, &ys, &zs, n, f32::NAN).is_empty() && contour(&[], &[], &[], 0, 1.0).is_empty());
    }

    /// The arrows' scale is round, keeps the longest arrow within reach, and is no
    /// smaller than it need be.
    #[test]
    fn the_arrow_scale_is_round() {
        for longest in [1e-9, 0.01, 0.3, 1.0, 2.236, 2.9, 7.0, 1e5, f64::NAN, f64::INFINITY, 0.0] {
            let k = arrow_scale(longest);
            assert!(k.is_finite() && k > 0.0, "{longest}");
            if longest.is_finite() && longest > 1e-6 && longest < 1e5 {
                assert!(k * longest <= ARROW_REACH * (1.0 + 1e-9) && k * longest > 0.4 * ARROW_REACH, "{longest}: {k}");
            }
        }
        assert_eq!(arrow_scale(2.236), 0.4);
        assert_eq!(scale_text(0.4), r"arrows drawn at $0.4 \times$ their length");
        assert_eq!(scale_text(0.25), r"arrows drawn at $0.25 \times$ their length");
        assert_eq!(scale_text(2.0), r"arrows drawn at $2 \times$ their length");
    }

    /// The board and the picture fit the canvas, from a phone to a desktop.
    #[test]
    fn the_layout_fits_every_canvas() {
        for (w, h) in [(1080.0, 1000.0), (580.0, 700.0), (390.0, 800.0), (390.0, 560.0), (800.0, 600.0), (1600.0, 1000.0), (240.0, 180.0), (620.0, 480.0), (1300.0, 700.0), (360.0, 640.0)] {
            for inset in [0.0, 46.0] {
                let view = Vec2::new(w, h);
                let lay = Layout::of(view, inset);
                let (min, max) = lay.picture;
                assert!(min.cmpge(Vec2::ZERO).all() && max.cmple(view).all() && (max - min).min_element() > 50.0, "{w}x{h}: {lay:?}");
                if let Some(b) = &lay.board {
                    let within = |q: Vec2| q.cmpge(b.min - Vec2::splat(0.5)).all() && q.cmple(b.max + Vec2::splat(0.5)).all();
                    assert!(b.min.cmpge(Vec2::ZERO).all() && b.max.cmple(view + Vec2::splat(0.5)).all(), "{w}x{h}: {b:?}");
                    for r in [b.along, b.best, b.go] {
                        assert!(within(r.0) && within(r.1), "{w}x{h}: {b:?}");
                    }
                    assert!(within(b.since + Vec2::new(0.0, LINE_H)) && b.along.1.y + BELOW <= b.highest.y + 1.0, "{w}x{h}: {b:?}");
                    assert!(b.min.y >= inset || lay.narrow, "{w}x{h}: under the buttons");
                }
            }
        }
    }

    /// A film ends exactly where it was meant to, and a glide arrives exactly.
    #[test]
    fn films_and_glides_arrive() {
        let from = Setting { c: 1.0, place: 0.3 };
        let to = Setting { c: 1.1, place: -2.0 };
        let mut film = Film { from, to, seconds: 2.0, t: 0.0 };
        assert_eq!(film.now(), from);
        film.t = 2.0;
        assert_eq!(film.now(), to);
        let mut g = Glide::at(from);
        g.to = to;
        for _ in 0..600 {
            g.step(1.0 / 60.0);
        }
        assert!(g.arrived() && g.shown == to);
        // and the scene's place stays within a few turns, however far it was taken round
        let mut scene = Lagrange::new();
        settle(&mut scene, 20.0);
        scene.glide.now(|s| s.place = 50.0);
        settle(&mut scene, 5.0);
        assert!(scene.glide.shown.place.abs() <= PI64 + 1e-9);
    }
}



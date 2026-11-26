//! The Jacobian: a curved map of the plane, seen up close, is a linear map, and its
//! matrix is the matrix of partial derivatives.

use super::*;
use crate::math::expr::Func;
use crate::scenes::stats::plot::{self, Rect, AXES, CURVES, MARKS, OWN};
use glam::{DMat2, DVec2};

pub const DEF: SceneDef = SceneDef {
    id: "jacobian",
    title: "The Jacobian: a curved map up close",
    group: "Calculus",
    blurb: "Zoom in on any smooth map of the plane and it becomes a linear map: its matrix of partial derivatives, and what its determinant says.",
    make: |_| Box::new(Jacobian::new()),
};

// ----- the arithmetic, in double precision -------------------------------------------
//
// The formulas are the library's compiled ones (`Expr`), run on numbers of this file's
// own in f64: `R`, a plain number, and `D2`, a number with its two partial derivatives
// (forward-mode differentiation, exact, as the library's `Dual` is in f32). Seen up
// close the misfit between a map and its linear part is of the size of the patch
// squared: at a zoom of a thousand that is a millionth of the numbers, about what f32
// can tell apart at all.

/// A plain number, in f64.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
struct R(f64);

/// A number with its partial derivatives along x and along y, in f64.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
struct D2 {
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

/// The slope of `f` at `x`, given `y = f(x)` (as the library's f32 version has it).
fn slope64(f: Func, x: f64, y: f64) -> f64 {
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

impl Num for R {
    fn cst(v: f32) -> Self {
        R(v as f64)
    }
    fn value(self) -> f32 {
        self.0 as f32
    }
    fn add(self, o: Self) -> Self {
        R(self.0 + o.0)
    }
    fn sub(self, o: Self) -> Self {
        R(self.0 - o.0)
    }
    fn mul(self, o: Self) -> Self {
        R(self.0 * o.0)
    }
    fn div(self, o: Self) -> Self {
        R(self.0 / o.0)
    }
    fn neg(self) -> Self {
        R(-self.0)
    }
    fn call(self, f: Func) -> Self {
        R(call64(f, self.0))
    }
    fn powi(self, n: i32) -> Self {
        R(self.0.powi(n))
    }
    fn pow(self, o: Self) -> Self {
        R(self.0.powf(o.0))
    }
    fn atan2(self, o: Self) -> Self {
        R(self.0.atan2(o.0))
    }
    fn min(self, o: Self) -> Self {
        R(self.0.min(o.0))
    }
    fn max(self, o: Self) -> Self {
        R(self.0.max(o.0))
    }
}

impl D2 {
    /// The slope `k` along a direction in which the argument changes at rate `d`:
    /// nothing where it does not change at all, however steep `k` is (as `Dual` has it).
    fn along(k: f64, d: f64) -> f64 {
        if d == 0.0 {
            0.0
        } else {
            k * d
        }
    }
}

impl Num for D2 {
    fn cst(v: f32) -> Self {
        D2 { v: v as f64, dx: 0.0, dy: 0.0 }
    }
    fn value(self) -> f32 {
        self.v as f32
    }
    fn add(self, o: Self) -> Self {
        D2 { v: self.v + o.v, dx: self.dx + o.dx, dy: self.dy + o.dy }
    }
    fn sub(self, o: Self) -> Self {
        D2 { v: self.v - o.v, dx: self.dx - o.dx, dy: self.dy - o.dy }
    }
    fn mul(self, o: Self) -> Self {
        D2 { v: self.v * o.v, dx: self.dx * o.v + self.v * o.dx, dy: self.dy * o.v + self.v * o.dy }
    }
    fn div(self, o: Self) -> Self {
        let inv = 1.0 / o.v;
        let q = self.v * inv;
        D2 { v: q, dx: (self.dx - q * o.dx) * inv, dy: (self.dy - q * o.dy) * inv }
    }
    fn neg(self) -> Self {
        D2 { v: -self.v, dx: -self.dx, dy: -self.dy }
    }
    fn call(self, f: Func) -> Self {
        let y = call64(f, self.v);
        let k = slope64(f, self.v, y);
        D2 { v: y, dx: D2::along(k, self.dx), dy: D2::along(k, self.dy) }
    }
    /// d(a^b) = b a^(b−1) da + a^b ln(a) db, each term only where its factor varies.
    fn pow(self, o: Self) -> Self {
        let v = self.v.powf(o.v);
        let by_base = if (self.dx != 0.0 || self.dy != 0.0) && o.v != 0.0 { o.v * self.v.powf(o.v - 1.0) } else { 0.0 };
        let by_exponent = if (o.dx != 0.0 || o.dy != 0.0) && v != 0.0 { v * self.v.ln() } else { 0.0 };
        D2 { v, dx: D2::along(by_base, self.dx) + D2::along(by_exponent, o.dx), dy: D2::along(by_base, self.dy) + D2::along(by_exponent, o.dy) }
    }
    fn atan2(self, o: Self) -> Self {
        let r2 = self.v * self.v + o.v * o.v;
        let (ka, kb) = if r2 > 0.0 { (o.v / r2, -self.v / r2) } else { (0.0, 0.0) };
        D2 { v: self.v.atan2(o.v), dx: ka * self.dx + kb * o.dx, dy: ka * self.dy + kb * o.dy }
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

// ----- maps of the plane ---------------------------------------------------------------

/// A map of the plane, (x, y) to (u, v), as two compiled formulas in x and y.
#[derive(Clone, Debug)]
struct Formulas {
    u: Expr,
    v: Expr,
}

impl Formulas {
    fn parse(u: &str, v: &str) -> Option<Formulas> {
        Some(Formulas { u: Expr::parse(u, &["x", "y"]).ok()?, v: Expr::parse(v, &["x", "y"]).ok()? })
    }

    /// f(q).
    fn at(&self, q: DVec2) -> DVec2 {
        let vars = [R(q.x), R(q.y)];
        DVec2::new(self.u.eval_num(&vars).0, self.v.eval_num(&vars).0)
    }

    /// f(q) and the Jacobian J(q): its first column is (∂u/∂x, ∂v/∂x), where a small
    /// step along x goes, and its second (∂u/∂y, ∂v/∂y).
    fn jet(&self, q: DVec2) -> (DVec2, DMat2) {
        let vars = [D2 { v: q.x, dx: 1.0, dy: 0.0 }, D2 { v: q.y, dx: 0.0, dy: 1.0 }];
        let (u, v) = (self.u.eval_num(&vars), self.v.eval_num(&vars));
        (DVec2::new(u.v, v.v), DMat2::from_cols(DVec2::new(u.dx, v.dx), DVec2::new(u.dy, v.dy)))
    }
}

/// The maps to choose from.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Kind {
    Bend,
    Polar,
    Square,
    Fold,
    Linear,
    Own,
}

/// A part of a plane: its centre and half its width (and height: the pictures are square).
#[derive(Clone, Copy, Debug, PartialEq)]
struct Win {
    c: DVec2,
    h: f64,
}

impl Win {
    const fn of(x: f64, y: f64, h: f64) -> Win {
        Win { c: DVec2::new(x, y), h }
    }

    fn lerp(self, to: Win, b: f64) -> Win {
        Win { c: self.c.lerp(to.c, b), h: self.h + (to.h - self.h) * b }
    }

    /// The nearest point to `q` within `share` of the half-width from the centre.
    fn keep(&self, q: DVec2, share: f64) -> DVec2 {
        let r = self.h * share;
        let q = if q.is_finite() { q } else { self.c };
        DVec2::new(q.x.max(self.c.x - r).min(self.c.x + r), q.y.max(self.c.y - r).min(self.c.y + r))
    }

    /// Where the plane's point `q` is drawn, in the picture `rect` (logical pixels).
    fn px(&self, rect: Rect, q: DVec2) -> Vec2 {
        let mid = (rect.0 + rect.1) * 0.5;
        let k = (rect.1.x - rect.0.x) as f64 * 0.5 / self.h.max(1e-300);
        let d = (q - self.c) * k;
        mid + Vec2::new(d.x as f32, -d.y as f32)
    }

    /// The plane's point drawn at `at` in the picture `rect`.
    fn back(&self, rect: Rect, at: Vec2) -> DVec2 {
        let mid = (rect.0 + rect.1) * 0.5;
        let k = self.h / ((rect.1.x - rect.0.x) as f64 * 0.5).max(1e-6);
        DVec2::new((at.x - mid.x) as f64 * k, -(at.y - mid.y) as f64 * k) + self.c
    }
}

/// One of the maps, as the panel and the picture show it.
struct Preset {
    kind: Kind,
    /// Its button.
    name: &'static str,
    u: &'static str,
    v: &'static str,
    /// The map, its Jacobian and the Jacobian's determinant, written out.
    formula: &'static str,
    jacobian: &'static str,
    det: &'static str,
    /// What the two inputs are called (polar coordinates: r and θ).
    inputs: [&'static str; 2],
    /// The inputs shown at zoom 1, and where p may go.
    domain: Win,
    /// Where p is put when the map is chosen.
    p: [f64; 2],
}

const PRESETS: [Preset; 5] = [
    Preset {
        kind: Kind::Bend,
        name: "a bend",
        u: "x + 0.4 y^2",
        v: "y + 0.4 x^2",
        formula: r"f(x, y) = (x + 0.4y^2,\ y + 0.4x^2)",
        jacobian: r"J = \begin{bmatrix} 1 & 0.8y \\ 0.8x & 1 \end{bmatrix}",
        det: r"\det J = 1 - 0.64xy",
        inputs: ["x", "y"],
        domain: Win::of(0.0, 0.0, 1.2),
        p: [0.6, -0.45],
    },
    Preset {
        kind: Kind::Polar,
        name: "polar coordinates",
        u: "x cos(y)",
        v: "x sin(y)",
        formula: r"f(r, \theta) = (r\cos\theta,\ r\sin\theta)",
        jacobian: r"J = \begin{bmatrix} \cos\theta & -r\sin\theta \\ \sin\theta & r\cos\theta \end{bmatrix}",
        det: r"\det J = r",
        inputs: ["r", r"\theta"],
        domain: Win::of(1.25, 0.0, 1.5),
        p: [2.0, 0.5],
    },
    Preset {
        kind: Kind::Square,
        name: "squaring, $z^2$",
        u: "x^2 - y^2",
        v: "2 x y",
        formula: r"f(x, y) = (x^2 - y^2,\ 2xy)",
        jacobian: r"J = 2\begin{bmatrix} x & -y \\ y & x \end{bmatrix}",
        det: r"\det J = 4(x^2 + y^2)",
        inputs: ["x", "y"],
        domain: Win::of(0.0, 0.0, 1.2),
        p: [0.8, 0.6],
    },
    Preset {
        kind: Kind::Fold,
        name: "a fold",
        u: "x",
        v: "y^2",
        formula: r"f(x, y) = (x,\ y^2)",
        jacobian: r"J = \begin{bmatrix} 1 & 0 \\ 0 & 2y \end{bmatrix}",
        det: r"\det J = 2y",
        inputs: ["x", "y"],
        domain: Win::of(0.0, 0.0, 1.5),
        p: [0.3, 0.0],
    },
    Preset {
        kind: Kind::Linear,
        name: "linear",
        u: "x + 0.5 y",
        v: "0.25 x + y",
        formula: r"f(x, y) = (x + 0.5y,\ 0.25x + y)",
        jacobian: r"J = \begin{bmatrix} 1 & 0.5 \\ 0.25 & 1 \end{bmatrix}",
        det: r"\det J = 0.875",
        inputs: ["x", "y"],
        domain: Win::of(0.0, 0.0, 1.5),
        p: [0.6, 0.5],
    },
];

fn preset(kind: Kind) -> Option<&'static Preset> {
    PRESETS.iter().find(|p| p.kind == kind)
}

/// The map drawn: one map, or on the way from one to another (`b` from 0 to 1), when
/// it is the mixture (1 − b) f₀ + b f₁, whose Jacobian is the same mixture of theirs.
#[derive(Clone, Debug)]
struct Mix {
    from: Formulas,
    to: Formulas,
    b: f64,
}

impl Mix {
    fn at(&self, q: DVec2) -> DVec2 {
        if self.b >= 1.0 {
            return self.to.at(q);
        }
        self.from.at(q).lerp(self.to.at(q), self.b)
    }

    fn jet(&self, q: DVec2) -> (DVec2, DMat2) {
        if self.b >= 1.0 {
            return self.to.jet(q);
        }
        let ((a, ja), (b, jb)) = (self.from.jet(q), self.to.jet(q));
        (a.lerp(b, self.b), ja * (1.0 - self.b) + jb * self.b)
    }
}

// ----- the patch, and how far its two images are apart -----------------------------

/// Points round the square [−1, 1]², `n` to a side, anticlockwise from (−1, −1): the
/// patch's outline, as steps from p in units of h.
fn outline(n: usize) -> Vec<DVec2> {
    let corners = [DVec2::new(-1.0, -1.0), DVec2::new(1.0, -1.0), DVec2::new(1.0, 1.0), DVec2::new(-1.0, 1.0)];
    let mut out = Vec::with_capacity(4 * n);
    for k in 0..4 {
        let (a, b) = (corners[k], corners[(k + 1) % 4]);
        for i in 0..n {
            out.push(a.lerp(b, i as f64 / n as f64));
        }
    }
    out
}

/// The point of the outline of `outline(n)` at `t` (from 0 to 4n, where a whole `t` is
/// that point of it), and the points between.
fn on_outline(t: f64, n: usize) -> DVec2 {
    let corners = [DVec2::new(-1.0, -1.0), DVec2::new(1.0, -1.0), DVec2::new(1.0, 1.0), DVec2::new(-1.0, 1.0)];
    let n = n.max(1) as f64;
    let t = t.rem_euclid(4.0 * n);
    let k = ((t / n).floor() as usize).min(3);
    corners[k].lerp(corners[(k + 1) % 4], (t - k as f64 * n) / n)
}

/// Points on the patch's outline (its picture and its misfit).
const SIDE_POINTS: usize = 24;

/// What the patch of half-side `h` about `p` comes to under the map and under J: the
/// misfit (the furthest a point of the outline lands from where J sends it), where on
/// the outline that is, and the signed area inside the true image.
#[derive(Clone, Copy, Debug, PartialEq)]
struct Fit {
    misfit: f64,
    worst: usize,
    area: f64,
}

/// `steps` is `outline(..)`.
fn fit(map: &Mix, p: DVec2, h: f64, steps: &[DVec2]) -> Option<Fit> {
    let (fp, j) = map.jet(p);
    if !(fp.is_finite() && j.is_finite()) {
        return None;
    }
    let (mut misfit, mut worst, mut twice) = (0.0f64, 0, 0.0f64);
    let mut last = map.at(p + steps[steps.len() - 1] * h) - fp;
    for (i, s) in steps.iter().enumerate() {
        let image = map.at(p + *s * h) - fp;
        let gap = (image - j * (*s * h)).length();
        if !gap.is_finite() {
            return None;
        }
        if gap > misfit {
            (misfit, worst) = (gap, i);
        }
        twice += last.perp_dot(image);
        last = image;
    }
    Some(Fit { misfit, worst, area: 0.5 * twice })
}

/// The misfit as a share of the patch's side, 2h.
fn share(map: &Mix, p: DVec2, h: f64) -> Option<f64> {
    fit(map, p, h, &outline(SIDE_POINTS)).map(|f| f.misfit / (2.0 * h))
}

/// The signed area inside the true image of the patch, divided by the patch's own,
/// 4h². (A finer outline than the picture's: the curved sides are cut into chords.)
fn area_ratio(map: &Mix, p: DVec2, h: f64) -> Option<f64> {
    fit(map, p, h, &outline(96)).map(|f| f.area / (4.0 * h * h))
}

// ----- what J and its determinant say ---------------------------------------------------

/// What J does to the plane near p.
#[derive(Clone, Copy, Debug, PartialEq)]
enum Rank {
    /// det J ≠ 0: a parallelogram, kept the right way round (true) or turned over.
    Two(bool),
    /// det J = 0, J ≠ 0: everything pressed onto a line.
    One,
    /// J = 0: everything pressed to a point.
    Zero,
}

/// What counts as nothing: for det J against the size of J squared, and for J itself
/// against 1 (the maps' numbers are about 1 in size).
const FLAT: f64 = 1e-9;

fn rank(j: DMat2) -> Rank {
    let size: f64 = j.to_cols_array().iter().map(|v| v * v).sum();
    if size <= 1e-18 {
        Rank::Zero
    } else if j.determinant().abs() <= FLAT * size {
        Rank::One
    } else {
        Rank::Two(j.determinant() > 0.0)
    }
}

/// det J in words: what it does to small areas near p.
fn det_words(j: DMat2) -> String {
    let d = j.determinant();
    match rank(j) {
        Rank::Two(true) => format!("small areas near $p$ are scaled by {}", num(d, 2)),
        Rank::Two(false) => format!("small areas near $p$ are scaled by {} and turned over", num(-d, 2)),
        Rank::One => "$J$ presses the plane onto a line, and has no inverse".into(),
        Rank::Zero => "$J = 0$ presses the plane to a point".into(),
    }
}

/// A percentage, to two figures.
fn percent(share: f64) -> String {
    let p = 100.0 * share;
    if !p.is_finite() {
        "?".into()
    } else if p == 0.0 {
        "0".into()
    } else if p >= 9.95 {
        num(p, 0)
    } else if p >= 0.995 {
        num(p, 1)
    } else if p >= 0.09995 {
        num(p, 2)
    } else {
        format!("${}$", fmt(p as f32))
    }
}

// ----- what glides ---------------------------------------------------------------------

/// Where p is, the zoom (as a power of 2), and the patch's half-side at zoom 1.
#[derive(Clone, Copy, Debug, PartialEq)]
struct Setting {
    p: DVec2,
    zoom: f64,
    h0: f64,
}

impl Setting {
    fn numbers(&self) -> [f64; 4] {
        [self.p.x, self.p.y, self.zoom, self.h0]
    }

    fn from_numbers(v: [f64; 4]) -> Setting {
        Setting { p: DVec2::new(v[0], v[1]), zoom: v[2], h0: v[3] }
    }

    /// The share of the zoom-1 window shown: 2 to the power −zoom.
    fn s(&self) -> f64 {
        (-self.zoom).exp2()
    }

    /// The patch's half-side, as zoomed.
    fn h(&self) -> f64 {
        self.h0 * self.s()
    }
}

/// How fast a change is glided to: each of the two eases one after the other covers two
/// thirds of what is left in this many seconds.
const FOLLOW: f64 = 0.09;

/// A setting as asked for and as drawn: the drawn one follows through `between`, two
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
        let (to, mut between, mut shown) = (self.to.numbers(), self.between.numbers(), self.shown.numbers());
        let mut apart = 0.0f64;
        for i in 0..4 {
            between[i] += (to[i] - between[i]) * k;
            shown[i] += (between[i] - shown[i]) * k;
            apart = apart.max((to[i] - between[i]).abs()).max((to[i] - shown[i]).abs());
        }
        // (exactly there in the end: the picture then stops changing)
        if apart < 1e-5 || !apart.is_finite() {
            *self = Glide::at(self.to);
        } else {
            (self.between, self.shown) = (Setting::from_numbers(between), Setting::from_numbers(shown));
        }
    }
}

fn smooth(t: f64) -> f64 {
    let t = t.clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

/// Seconds a change of map takes to glide from the old picture to the new.
const BLEND_SECONDS: f64 = 0.6;
/// The most the pictures are magnified (a power of 2: 1024 times).
const ZOOM_MOST: f64 = 10.0;
/// The least and the most half-side of the patch at zoom 1.
const H0_RANGE: [f64; 2] = [0.05, 0.6];

/// A zoom that plays: from one zoom to another (powers of 2) over some seconds of the
/// scene's time, easing in and out.
#[derive(Clone, Copy, Debug)]
struct Film {
    from: f64,
    to: f64,
    t: f64,
    seconds: f64,
}

// ----- where things go ---------------------------------------------------------------

const PAD: f32 = 12.0;
const BUTTON_H: f32 = 26.0;
/// Room above each picture for its title, beside it for the numbers along its side, and
/// below it for the numbers along its foot.
const TITLE_H: f32 = 22.0;
const SIDE_W: f32 = 46.0;
const FOOT_H: f32 = 17.0;

/// The numbers under the pictures: the buttons, J with its numbers, det J, the misfit.
#[derive(Clone, Copy, Debug, PartialEq)]
struct Board {
    rect: Rect,
    /// In two columns (a wide canvas), or one.
    two: bool,
    text: f32,
    line: f32,
    buttons: [Rect; 2],
    /// Top-left corners of the left column (or the one column) and the right column.
    left: Vec2,
    right: Vec2,
}

impl Board {
    fn height(two: bool) -> f32 {
        let text = if two { 13.0 } else { 12.0 };
        let mat = matrix_size(2, 2, 5, text).y;
        if two {
            2.0 * PAD + (BUTTON_H + 8.0).max(mat + 6.0) + 3.0 * (text + 6.0)
        } else {
            2.0 * PAD + BUTTON_H + 8.0 + mat + 6.0 + 3.0 * (text + 5.0)
        }
    }

    fn place(min: Vec2, width: f32, two: bool) -> Board {
        let text = if two { 13.0 } else { 12.0 };
        let line = text + if two { 6.0 } else { 5.0 };
        let rect = (min, min + Vec2::new(width, Board::height(two)));
        let x = min.x + PAD;
        let y = min.y + PAD;
        let (buttons, left, right) = if two {
            let half = (width - 2.0 * PAD) * 0.5;
            let rx = x + half + 10.0;
            let b = plot::even_row(2, rx, rx + (half - 10.0).min(300.0), y, BUTTON_H, 8.0);
            ([b[0], b[1]], Vec2::new(x, y), Vec2::new(rx, y + BUTTON_H + 8.0))
        } else {
            let b = plot::even_row(2, x, x + (width - 2.0 * PAD).min(300.0), y, BUTTON_H, 8.0);
            let below = Vec2::new(x, y + BUTTON_H + 8.0);
            ([b[0], b[1]], below, below)
        };
        Board { rect, two, text, line, buttons, left, right }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
struct Layout {
    view: Vec2,
    /// One picture above the other (a phone held upright), not side by side.
    stacked: bool,
    /// The two pictures, inputs and outputs, as squares (logical pixels).
    panes: [Rect; 2],
    /// Room for words: titles, numbers along the sides, names (none on a tiny canvas).
    words: bool,
    /// The arrow from the one picture to the other.
    arrow: (Vec2, Vec2),
    board: Option<Board>,
}

impl Layout {
    /// The pictures side by side, or one above the other, whichever makes them larger
    /// (the second only on a canvas taller than it is wide).
    fn of(view: Vec2, inset: f32) -> Layout {
        let wide = Layout::arranged(view, inset, false);
        if view.y > view.x {
            let tall = Layout::arranged(view, inset, true);
            if tall.side() > 1.05 * wide.side() {
                return tall;
            }
        }
        wide
    }

    fn side(&self) -> f32 {
        self.panes[0].1.x - self.panes[0].0.x
    }

    fn arranged(view: Vec2, inset: f32, stacked: bool) -> Layout {
        let view = view.max(Vec2::ONE);
        let words = view.x >= 300.0 && view.y >= 300.0;
        let top = inset.max(0.0) + 6.0;
        let (title, side_w, foot) = if words { (TITLE_H, SIDE_W, FOOT_H) } else { (0.0, 0.0, 0.0) };
        let margin = 8.0;
        let gap = if stacked { title + foot + 8.0 } else { (view.x * 0.06).clamp(40.0, 80.0) };
        // the side of the pictures, with or without the numbers under them
        let side_with = |board_h: f32| {
            let below = if board_h > 0.0 { board_h + 8.0 } else { 0.0 };
            if stacked {
                (view.x - side_w - 2.0 * margin).min((view.y - top - title - gap - foot - below - margin) * 0.5)
            } else {
                ((view.x - 2.0 * side_w - gap - 2.0 * margin) * 0.5).min(view.y - top - title - foot - below - margin)
            }
        };
        let two = !stacked && view.x >= 760.0;
        let mut board_h = if words { Board::height(two) } else { 0.0 };
        if board_h > 0.0 && side_with(board_h) < 150.0 {
            board_h = 0.0;
        }
        let side = side_with(board_h).max(8.0);
        let below = if board_h > 0.0 { board_h + 8.0 } else { 0.0 };
        let (panes, arrow, block_left, block_w) = if stacked {
            let block = title + 2.0 * side + gap + foot + below;
            let y0 = top + ((view.y - top - margin - block) * 0.5).max(0.0);
            let x0 = (side_w + (view.x - side_w - side) * 0.5).max(0.0);
            let first = (Vec2::new(x0, y0 + title), Vec2::new(x0 + side, y0 + title + side));
            let second = (Vec2::new(x0, first.1.y + gap), Vec2::new(x0 + side, first.1.y + gap + side));
            let at = x0 + side - 24.0;
            ([first, second], (Vec2::new(at, first.1.y + foot), Vec2::new(at, second.0.y - 3.0)), margin, view.x - 2.0 * margin)
        } else {
            let block_w = 2.0 * side + gap + 2.0 * side_w;
            let x0 = ((view.x - block_w) * 0.5).max(0.0) + side_w;
            let block = title + side + foot + below;
            let y0 = top + ((view.y - top - margin - block) * 0.5).max(0.0);
            let first = (Vec2::new(x0, y0 + title), Vec2::new(x0 + side, y0 + title + side));
            let second = (Vec2::new(x0 + side + gap, y0 + title), Vec2::new(x0 + 2.0 * side + gap, y0 + title + side));
            let mid = y0 + title + side * 0.5;
            ([first, second], (Vec2::new(first.1.x + 8.0, mid), Vec2::new(second.0.x - 8.0, mid)), x0 - side_w, block_w)
        };
        let board = (board_h > 0.0).then(|| {
            let y = panes[1].1.y + foot + 8.0;
            let (x, w) = if stacked { (margin, view.x - 2.0 * margin) } else { (block_left, block_w) };
            Board::place(Vec2::new(x, y), w, two)
        });
        Layout { view, stacked, panes, words, arrow, board }
    }
}

fn inside_rect(p: Vec2, (min, max): Rect) -> bool {
    p.cmpge(min).all() && p.cmple(max).all()
}

/// The grid's lines across `lo` to `hi`, a picture `pixels` wide, as (value, level,
/// brightness): at the nested steps ..., 5 × 10ⁿ, 2.5 × 10ⁿ, 5 × 10ⁿ⁻¹, 2.5 × 10ⁿ⁻¹, ...
/// (level k + 1 is the step after level k: 5 × 10ⁿ is level −2n, 2.5 × 10ⁿ level
/// −2n + 1), from the first that is at least the whole span. Each line is as bright as
/// the spacing of the coarsest of those steps it is on allows (none closer than 14
/// pixels), so that lines fade in as the picture is magnified instead of appearing all
/// at once.
fn grid_lines(lo: f64, hi: f64, pixels: f64) -> Vec<(f64, i32, f32)> {
    let span = hi - lo;
    if !(span > 0.0 && span.is_finite() && pixels > 1.0) {
        return Vec::new();
    }
    let per = pixels / span;
    let e = (span / 2.5).log10().ceil() as i32;
    let (mut step, mut level) = (2.5 * 10f64.powi(e), -2 * e + 1);
    let mut out = Vec::new();
    let mut coarser: Option<f64> = None;
    for _ in 0..16 {
        let bright = smooth((step * per - 14.0) / 30.0) as f32;
        if bright < 0.02 {
            break;
        }
        let (first, last) = ((lo / step - 1e-9).ceil() as i64, (hi / step + 1e-9).floor() as i64);
        if last - first > 200 {
            break;
        }
        for k in first..=last {
            let v = k as f64 * step;
            // (each line once, at the brightness of the coarsest step it is on)
            let on_coarser = coarser.is_some_and(|c| ((v / c).round() * c - v).abs() < 1e-6 * step);
            if !on_coarser {
                out.push((v, level, bright));
            }
        }
        coarser = Some(step);
        // (an odd level is 2.5 times a power of ten: the next is a fifth of it)
        step /= if level.rem_euclid(2) == 1 { 5.0 } else { 2.0 };
        level += 1;
    }
    out
}

/// One line of the grid: x (axis 0) or y (axis 1) held at a value, the part of it in
/// the region drawn, its level of steps (see `grid_lines`) and its brightness.
#[derive(Clone, Copy, Debug, PartialEq)]
struct Line {
    axis: usize,
    level: i32,
    bright: f32,
    ends: [DVec2; 2],
}

/// The lines of the grid over the region from `lo` to `hi` of the inputs, shown
/// `per_unit` pixels to a unit, in order of axis and value.
fn grid_set(lo: DVec2, hi: DVec2, per_unit: f64) -> Vec<Line> {
    let mut lines = Vec::new();
    for axis in 0..2 {
        let other = 1 - axis;
        for (v, level, bright) in grid_lines(lo[axis], hi[axis], (hi[axis] - lo[axis]) * per_unit) {
            let end = |t: f64| {
                let mut q = DVec2::ZERO;
                q[axis] = v;
                q[other] = t;
                q
            };
            lines.push(Line { axis, level, bright, ends: [end(lo[other]), end(hi[other])] });
        }
    }
    lines
}

/// A set of the grid's lines and their images under the map, kept while nothing they
/// depend on changes (each image is many points of the map).
#[derive(Default)]
struct Grid {
    /// What the lines were worked out for (the region and the scale), and what the
    /// images were (the map and how far it has been glided to).
    key: Option<[u64; 5]>,
    images_key: Option<(u64, u64)>,
    lines: Vec<Line>,
    images: Vec<Vec<DVec2>>,
}

impl Grid {
    /// The lines over the region from `lo` to `hi` at `per_unit` pixels to a unit.
    fn set(&mut self, lo: DVec2, hi: DVec2, per_unit: f64) {
        let key = [lo.x, lo.y, hi.x, hi.y, per_unit].map(f64::to_bits);
        if self.key != Some(key) {
            self.key = Some(key);
            self.lines = grid_set(lo, hi, per_unit);
            self.images_key = None;
        }
    }

    /// Their images under `map` (`version` and how far it is glided tell maps apart).
    fn image(&mut self, map: &Mix, version: u64) {
        let key = (version, map.b.to_bits());
        if self.images_key == Some(key) {
            return;
        }
        self.images_key = Some(key);
        self.images = self.lines.iter().map(|l| (0..=LINE_POINTS).map(|i| map.at(l.ends[0].lerp(l.ends[1], i as f64 / LINE_POINTS as f64))).collect()).collect();
    }
}

/// Samples along each line of the grid's image.
const LINE_POINTS: usize = 96;
/// How near the edge of a picture (pixels) the grid's lines fade out.
const EDGE_FADE: f32 = 28.0;

pub struct Jacobian {
    kind: Kind,
    glide: Glide,
    /// The map drawn (and the one before, while it is glided from).
    mix: Mix,
    /// The inputs shown by the map glided from, and the clock of the glide (0 to 1).
    from_domain: Win,
    clock: f64,
    /// The offset of the inputs shown from where the zoom alone would put them: what a
    /// hand that moved p while the picture was magnified left behind.
    offset: DVec2,
    film: Option<Film>,
    /// Play the film once the glide has arrived.
    film_waiting: Option<Film>,
    /// The formulas typed for "your own map", and the inputs it is shown on.
    own: [ExprField; 2],
    own_domain: Win,
    /// Counts the maps made, so that what is kept for one is not taken for another's.
    version: u64,
    grid: Grid,
    old_grid: Option<Grid>,
    /// Where on the patch's outline the misfit is to be marked, and where it is as drawn
    /// (see `misfit_at`).
    misfit_shown: usize,
    misfit_place: f64,
    dragging: bool,
}

/// What everything drawn in a frame is worked out from.
#[derive(Clone, Copy, Debug)]
struct State {
    s: Setting,
    win_in: Win,
    win_out: Win,
    /// The windows at zoom 1 (the domain p may go in).
    domain: Win,
    fp: DVec2,
    j: DMat2,
}

impl Jacobian {
    fn new() -> Self {
        let first = &PRESETS[0];
        let map = Formulas::parse(first.u, first.v).expect("the presets parse");
        let s = Setting { p: DVec2::from(first.p), zoom: 0.0, h0: 0.35 };
        let mut scene = Jacobian {
            kind: Kind::Bend,
            glide: Glide::at(s),
            mix: Mix { from: map.clone(), to: map, b: 1.0 },
            from_domain: first.domain,
            clock: 1.0,
            offset: DVec2::ZERO,
            film: None,
            film_waiting: None,
            own: [ExprField::new(first.u, &["x", "y"]), ExprField::new(first.v, &["x", "y"])],
            own_domain: first.domain,
            version: 0,
            grid: Grid::default(),
            old_grid: None,
            misfit_shown: 0,
            misfit_place: 0.0,
            dragging: false,
        };
        scene.set_up(0);
        // (opening, the first experiment's picture as it starts: zoom 1, nothing to glide)
        scene.glide = Glide::at(scene.glide.to);
        scene.film_waiting = None;
        scene
    }

    /// The inputs shown at zoom 1 for the map `kind`.
    fn domain_of(&self, kind: Kind) -> Win {
        preset(kind).map_or(self.own_domain, |p| p.domain)
    }

    /// The inputs shown at zoom 1 as drawn (on the way from one map's to another's).
    fn domain(&self) -> Win {
        self.from_domain.lerp(self.domain_of(self.kind), self.mix.b)
    }

    /// Change to the map `kind` (with these formulas for "your own"): the picture glides
    /// from the old map to the new one.
    fn choose(&mut self, kind: Kind, formulas: Formulas) {
        self.from_domain = self.domain();
        let from = Mix { from: self.mix.from.clone(), to: self.mix.to.clone(), b: self.mix.b };
        // (from the map as drawn, which may itself be a mixture: kept as the formulas
        // that are on their way out, while the glide from it is short)
        self.mix = Mix { from: if from.b >= 0.5 { from.to } else { from.from }, to: formulas, b: 0.0 };
        self.clock = 0.0;
        self.kind = kind;
        self.version += 1;
        self.old_grid = Some(std::mem::take(&mut self.grid));
    }

    /// Pick one of the presets: its map, and p where it puts it.
    fn pick(&mut self, kind: Kind) {
        let Some(pre) = preset(kind) else { return };
        let map = Formulas::parse(pre.u, pre.v).expect("the presets parse");
        self.own[0].set(pre.u);
        self.own[1].set(pre.v);
        self.own_domain = pre.domain;
        if self.kind != kind {
            self.choose(kind, map);
        }
        self.glide.to.p = DVec2::from(pre.p);
        (self.film, self.film_waiting) = (None, None);
    }

    /// The formulas typed: a map of the visitor's own.
    fn typed(&mut self) {
        let map = Formulas { u: self.own[0].expr.clone(), v: self.own[1].expr.clone() };
        let domain = self.domain_of(self.kind);
        self.own_domain = domain;
        self.choose(Kind::Own, map);
        self.glide.to.p = domain.keep(self.glide.to.p, 0.95);
        (self.film, self.film_waiting) = (None, None);
    }

    /// Set up experiment `k` of `TRIALS`.
    fn set_up(&mut self, k: usize) {
        let t = &TRIALS[k];
        self.pick(t.kind);
        self.glide.to = Setting { p: DVec2::from(t.p), zoom: 0.0, h0: t.h0 };
        self.film = None;
        self.film_waiting = t.film.map(|to| Film { from: 0.0, to, t: 0.0, seconds: 5.0 });
    }

    /// Ask for another zoom (by the buttons or the slider): glided to.
    fn zoom_to(&mut self, zoom: f64) {
        self.film = None;
        self.film_waiting = None;
        self.glide.to.zoom = zoom.max(0.0).min(ZOOM_MOST);
    }

    fn at_rest(&self) -> bool {
        self.glide.arrived() && self.mix.b >= 1.0 && self.film.is_none() && self.film_waiting.is_none()
    }

    /// Move everything on: `dt` seconds of the scene's time (the film) and `real_dt` of
    /// the clock's (a change of setting glides by the clock, so that it is seen through
    /// even while the scene's time stands still).
    fn advance(&mut self, dt: f64, real_dt: f64) {
        let before = self.glide.shown;
        let b_before = self.mix.b;
        if self.mix.b < 1.0 {
            self.clock = (self.clock + real_dt.max(0.0) / BLEND_SECONDS).min(1.0);
            self.mix.b = if self.clock >= 1.0 { 1.0 } else { smooth(self.clock) };
            if self.mix.b >= 1.0 {
                self.mix.from = self.mix.to.clone();
            }
        }
        self.glide.step(real_dt);
        if let Some(film) = self.film_waiting {
            if self.glide.arrived() && self.mix.b >= 1.0 {
                self.film = Some(film);
                self.film_waiting = None;
            }
        }
        if let Some(mut film) = self.film {
            film.t = (film.t + dt.max(0.0) / film.seconds).min(1.0);
            let z = film.from + (film.to - film.from) * smooth(film.t);
            self.glide.now(|s| s.zoom = z);
            self.film = (film.t < 1.0).then_some(film);
        }
        // the inputs shown follow the zoom: magnified about p, and back to the whole
        // of the map at zoom 1 (see `canonical`); what a hand left behind goes with them
        let (s0, s1) = (before.s(), self.glide.shown.s());
        if s1 < s0 {
            self.offset *= s1 / s0;
        } else if s1 > s0 {
            self.offset *= ((1.0 - s1) / (1.0 - s0).max(1e-12)).max(0.0);
        }
        // and a change of map takes it away
        if b_before < 1.0 {
            self.offset *= ((1.0 - self.mix.b) / (1.0 - b_before)).max(0.0);
        }
        if !self.offset.is_finite() || self.glide.shown.zoom <= 0.0 {
            self.offset = DVec2::ZERO;
        }
    }

    /// What is drawn, worked out from the setting as drawn.
    fn state(&self) -> State {
        let s = self.glide.shown;
        let domain = self.domain();
        let k = s.s();
        let (fp, j) = self.mix.jet(s.p);
        // magnified about p: a point keeps its place in its picture
        let win_in = Win { c: s.p + (domain.c - s.p) * k + self.offset, h: domain.h * k };
        // The outputs shown are as wide as the inputs (the two pictures are drawn to the
        // same scale, so that what the map does to sizes is seen as it is), with f(p) in
        // the middle, where the image of the patch has the most room: magnified about
        // f(p), and following it as p is moved. (Where the map stretches, the image of
        // the grid runs past the edges.)
        let centre = if fp.is_finite() && fp.abs().max_element() < 1e12 { fp } else { win_in.c };
        let win_out = Win { c: centre, h: win_in.h };
        State { s, win_in, win_out, domain, fp, j }
    }

    /// The name of the inputs: x and y, or r and θ.
    fn inputs(&self) -> [&'static str; 2] {
        preset(self.kind).map_or(["x", "y"], |p| p.inputs)
    }

    /// The grid's lines for the picture of `st` (`pixels` wide), and their images: over
    /// the part of the inputs that the region of `region` covers. While the map glides
    /// from one to another, the lines are those the new map will be shown with, and the
    /// old map's lines fade out beside them (`old_grid`): each set stays the same all
    /// the way, so that the picture glides instead of changing its lines as it goes.
    fn grids(&mut self, st: &State, pixels: f32) {
        let per = |w: Win| pixels as f64 / (2.0 * w.h).max(1e-300);
        if self.mix.b < 1.0 {
            let (to, domain) = (self.glide.to, self.domain_of(self.kind));
            let k = to.s();
            let w = Win { c: to.p + (domain.c - to.p) * k, h: domain.h * k };
            let (lo, hi) = region(w, domain);
            self.grid.set(lo, hi, per(w));
        } else {
            self.old_grid = None;
            let (lo, hi) = region(st.win_in, st.domain);
            self.grid.set(lo, hi, per(st.win_in));
        }
        let (map, version) = (&self.mix, self.version);
        self.grid.image(map, version);
        if let Some(old) = &mut self.old_grid {
            old.image(map, version);
        }
    }
}

/// The part of the inputs the grid is drawn over: twice as wide as the inputs shown,
/// so that magnified the outputs shown are filled, and within the map's domain.
fn region(w: Win, domain: Win) -> (DVec2, DVec2) {
    let lo = (w.c - DVec2::splat(2.0 * w.h)).max(domain.c - DVec2::splat(domain.h));
    let hi = (w.c + DVec2::splat(2.0 * w.h)).min(domain.c + DVec2::splat(domain.h));
    (lo, hi.max(lo))
}

// ----- experiments -------------------------------------------------------------------

struct Trial {
    label: &'static str,
    look_for: &'static str,
    kind: Kind,
    p: [f64; 2],
    h0: f64,
    /// A zoom to play to, from 1 (as a power of 2).
    film: Option<f64>,
}

const TRIALS: [Trial; 5] = [
    Trial {
        label: "zoom in: curved becomes straight",
        look_for: r"A film: both pictures are magnified 16 times, about $p$ and about $f(p)$, and the patch shrinks to match. The yellow outline settles onto the dashed parallelogram, and the grid's image straightens with it. The misfit halves each time the side halves: 9.9 % of the side at first, 4.9 % magnified twice, 2.5 % four times, 0.62 % sixteen times. Up close, $f$ moves points as the linear map $J$ moves steps.",
        kind: Kind::Bend,
        p: [0.6, -0.45],
        h0: 0.35,
        film: Some(4.0),
    },
    Trial {
        label: r"polar coordinates: $\det J = r$",
        look_for: r"$f(r, \theta) = (r\cos\theta, r\sin\theta)$ turns the grid of $(r, \theta)$ on the left into circles and rays on the right. At $r = 2$ a step along $r$ (pink) goes straight out from 0 at its own length, and a step along $\theta$ (green) goes round, twice as long: $J$'s columns are 1 and $r$ long, at right angles, so $\det J = r = 2$ and the patch lands with twice its area. Drag $p$ left: at $r = 1$ it keeps its area, and at $r = 0$ the whole line $r = 0$ lands on the point 0, where $\det J = 0$.",
        kind: Kind::Polar,
        p: [2.0, 0.5],
        h0: 0.25,
        film: None,
    },
    Trial {
        label: "squaring keeps squares square",
        look_for: r"$f(x, y) = (x^2 - y^2, 2xy)$ squares the complex number $x + iy$. At $p = (0.8, 0.6)$, $J$'s columns are $(1.6, 1.2)$ and $(-1.2, 1.6)$: as long as each other and at right angles, a turn by 36.9° and a stretch by 2. So the dashed image of the square patch is a square twice as wide, and $\det J = 4(x^2 + y^2) = 4$. Drag $p$ anywhere but 0 and $J$ is still a turn and a stretch: small squares stay squares, and angles are kept. At 0, $J = 0$ and the dashed image is a single point.",
        kind: Kind::Square,
        p: [0.8, 0.6],
        h0: 0.25,
        film: None,
    },
    Trial {
        label: r"a fold, where $\det J = 0$",
        look_for: r"$f(x, y) = (x, y^2)$ folds the plane along the line $y = 0$, where $\det J = 2y = 0$. There $J$ takes a step along $y$ to $0$ (there is no green arrow on the right) and presses the patch onto a segment, dashed. $f$ lays the patch's lower half on its upper half: the yellow strip is filled twice, each of its points coming from two, $(x, y)$ and $(x, -y)$, so nothing can undo $f$ here. The strip is $s^2/4 = 0.16$ high; magnify, and it flattens onto the segment.",
        kind: Kind::Fold,
        p: [0.3, 0.0],
        h0: 0.4,
        film: None,
    },
    Trial {
        label: r"turned over where $\det J < 0$",
        look_for: r"The same fold, below the crease: at $y = -0.75$, $\det J = 2y = -1.5$. The patch lands upside down, as a page does when turned face down: the green arrow, a step up, now points down, clockwise from the pink one instead of anticlockwise, and the turning arrow inside goes round the other way. Its area still grows, by $|\det J| = 1.5$; its signed area, counted negative, is $-1.5$ times the patch's.",
        kind: Kind::Fold,
        p: [0.3, -0.75],
        h0: 0.3,
        film: None,
    },
];

// ----- drawing ---------------------------------------------------------------------------

/// The kinds of line (see `plot::apart`): each in a batch of its own, in the same order
/// in every frame. `eidos motion` follows a batch's points by their places in it, so
/// every curve keeps its number of points whatever part of it is in sight (`steady`,
/// `pieces`), and the grid's lines of each level of steps go in a batch of their own:
/// magnified, the lines of a level only leave the picture, and new ones come in on
/// levels of their own, so that no line ever takes another's place in a batch.
const ARROWS: u8 = MARKS + 2;
const LEVELS: u8 = 5;
const GRID_IN: u8 = OWN;
const GRID_OUT: u8 = OWN + LEVELS;
const TRUE_IMAGE: u8 = OWN + 2 * LEVELS;
const LINEAR: u8 = TRUE_IMAGE + 1;
const OUT_ARROWS: u8 = TRUE_IMAGE + 2;
const TURNING: u8 = TRUE_IMAGE + 3;

/// The batch of the grid's lines of `level`, in the inputs (`out` false) or outputs.
fn grid_kind(level: i32, out: bool) -> u8 {
    (if out { GRID_OUT } else { GRID_IN }) + level.rem_euclid(LEVELS as i32) as u8
}

/// Every batch, made in the same order in each frame: the order they are drawn in, the
/// first underneath.
fn hold_batches(f: &mut Frame) {
    let corner = f.hud(Vec2::ZERO);
    let hold = |f: &mut Frame, kind: u8, dashed: bool| plot::apart(f, kind, dashed).segment(corner, corner, 1.0, Color::CLEAR);
    hold(f, AXES, false);
    for k in 0..2 * LEVELS {
        hold(f, GRID_IN + k, false);
    }
    for kind in [CURVES, TRUE_IMAGE, TURNING] {
        hold(f, kind, false);
    }
    hold(f, LINEAR, true);
    for kind in [ARROWS, OUT_ARROWS, MARKS, plot::BUTTONS] {
        hold(f, kind, false);
    }
    plot::disc(f, Vec2::ZERO, 1.0, Color::CLEAR);
}

/// The part of the segment from `a` to `b` inside `rect`, if any (Liang and Barsky's
/// way: the range of the segment's parameter inside each pair of sides).
fn cut(a: Vec2, b: Vec2, (min, max): Rect) -> Option<(Vec2, Vec2)> {
    if !(a.is_finite() && b.is_finite()) {
        return None;
    }
    let d = b - a;
    let (mut t0, mut t1) = (0.0f32, 1.0f32);
    for (p, q) in [(-d.x, a.x - min.x), (d.x, max.x - a.x), (-d.y, a.y - min.y), (d.y, max.y - a.y)] {
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
    (t0 <= t1).then(|| (a + d * t0, a + d * t1))
}

/// Where a point of the screen that draws nothing goes in a batch: where it is, or far
/// off the picture if it is not a number.
fn unseen(q: Vec2) -> Vec2 {
    if q.is_finite() {
        q
    } else {
        Vec2::splat(-1e5)
    }
}

/// Far off the picture: where a piece that draws nothing goes (on the picture it would
/// seem to come and go as the curve passed the edge; off it, it is nowhere).
const AWAY: Vec2 = Vec2::splat(-1e5);

/// Points of the screen into the batch of `kind`, as one line, each with its width (0:
/// a ghost, which shapes a join and draws nothing).
fn push_line(f: &mut Frame, kind: u8, dashed: bool, pts: &[(Vec2, f32)], color: Color) {
    let world: Vec<(Vec3, f32)> = pts.iter().map(|(q, w)| (f.hud(*q), *w)).collect();
    let batch = plot::apart(f, kind, dashed);
    batch.brk();
    for (p, w) in world {
        batch.push(p, w, color);
    }
    batch.brk();
}

/// A curve of the screen inside `rect`, in the batch of `kind`, fading out within
/// `fade` pixels of its edges: one line of all its points, each where the curve has it,
/// those outside `rect` drawing nothing (nor the segments either side of them, where the
/// curve is faded to next to nothing). So the batch has the same number of points
/// whatever part of the curve is in sight, and every point moves as the curve does.
/// (Cut at the edge, the ends would slide along it, faster the more nearly the curve runs
/// along the edge, and switch from one speed to the other each time another segment
/// crosses it.)
fn faded(f: &mut Frame, kind: u8, pts: &[Vec2], rect: Rect, fade: f32, width: f32, color: Color) {
    let (min, max) = rect;
    let alpha = color.0[3] as f32 / 255.0;
    let line: Vec<(Vec3, f32, Color)> = pts
        .iter()
        .map(|q| {
            let room = (*q - min).min(max - *q).min_element();
            if q.is_finite() && room >= 0.0 {
                (f.hud(*q), width, color.a(alpha * smooth((room / fade.max(1e-3)) as f64) as f32))
            } else {
                (f.hud(unseen(*q)), 0.0, color)
            }
        })
        .collect();
    let batch = plot::apart(f, kind, false);
    batch.brk();
    for (p, w, c) in line {
        batch.push(p, w, c);
    }
    batch.brk();
}

/// A curve of the screen kept inside `rect`, in the batch of `kind`: each of its
/// segments a piece of two points, cut exactly at the edge, or two that draw nothing if
/// it is all outside. So the batch has the same number of points whatever part of the
/// curve is in sight, and every point in sight moves as the curve does: where a
/// segment's end crosses the edge, the end and the place it is cut at are one.
fn steady(f: &mut Frame, kind: u8, pts: &[Vec2], rect: Rect, width: f32, color: Color) {
    pieces(f, kind, false, pts, rect, width, color);
}

/// Straight pieces of the screen, from each point to the next, in the batch of `kind`,
/// each cut exactly at `rect` and each two points (a piece out of sight is two that draw
/// nothing, off the picture).
fn pieces(f: &mut Frame, kind: u8, dashed: bool, pts: &[Vec2], rect: Rect, width: f32, color: Color) {
    let mut out: Vec<(Vec3, f32)> = Vec::with_capacity(2 * pts.len());
    for w in pts.windows(2) {
        let (a, b, width) = match cut(w[0], w[1], rect) {
            Some((a, b)) => (a, b, width),
            None => (AWAY, AWAY, 0.0),
        };
        out.push((f.hud(a), width));
        out.push((f.hud(b), width));
    }
    let batch = plot::apart(f, kind, dashed);
    for pair in out.chunks(2) {
        batch.brk();
        for (p, w) in pair {
            batch.push(*p, *w, color);
        }
    }
    batch.brk();
}

/// An arrow on the screen in the batch of `kind`, its shaft cut at `rect`, with its
/// head (a triangle) if its tip is inside; nothing but ghosts if it has no length.
fn arrow(f: &mut Frame, kind: u8, a: Vec2, b: Vec2, rect: Rect, width: f32, color: Color) {
    let len = (b - a).length();
    let dir = if len.is_finite() && len > 0.5 { (b - a) / len } else { Vec2::ZERO };
    let head = (width * 3.2 + 6.0).min(len * 0.6);
    let end = b - dir * head * 0.8;
    if dir == Vec2::ZERO {
        let at = AWAY;
        push_line(f, kind, false, &[(at, 0.0), (at, 0.0)], color);
        return;
    }
    pieces(f, kind, false, &[a, end], rect, width, color);
    if inside_rect(b, rect) {
        let base = b - dir * head;
        let side = dir.perp() * head * 0.42;
        let [t0, t1, t2] = [b, base + side, base - side].map(|p| f.hud_at(2, p));
        f.dl.tris_alpha.tri(t0, t1, t2, color);
    }
}

/// A filled triangle on the screen, over a picture's backing.
fn fill_tri(f: &mut Frame, a: Vec2, b: Vec2, c: Vec2, color: Color) {
    if a.is_finite() && b.is_finite() && c.is_finite() {
        let [a, b, c] = [a, b, c].map(|p| f.hud_at(1, p));
        f.dl.tris_alpha.tri(a, b, c, color);
    }
}

/// The turning arrow drawn inside the patch: three quarters of a circle round p,
/// anticlockwise, as steps from p in units of h.
fn turning(n: usize) -> Vec<DVec2> {
    (0..=n).map(|i| DVec2::from_angle((-0.15 + 1.5 * i as f64 / n as f64) * std::f64::consts::PI) * 0.5).collect()
}

impl Jacobian {
    /// A picture's backing, frame and the numbers along its sides.
    fn backdrop(&self, f: &mut Frame, lay: &Layout, k: usize, win: Win, names: [&str; 2]) {
        let rect = lay.panes[k];
        hud_panel(f, rect.0, rect.1);
        let (min, max) = rect;
        let corners = [min, Vec2::new(max.x, min.y), max, Vec2::new(min.x, max.y), min];
        plot::path_of(f, AXES, &corners, 1.0, pal::DIM.a(0.45));
        // the plane's own axes, where they are in sight (and out of sight at the edge
        // they left by, unseen: a line in this batch keeps its place from frame to frame)
        let o = win.px(rect, DVec2::ZERO);
        let seen = |v: f32, lo: f32, hi: f32| if v > lo && v < hi { pal::DIM.a(0.35) } else { Color::CLEAR };
        let x = if o.x.is_finite() { o.x.max(min.x).min(max.x) } else { min.x };
        let y = if o.y.is_finite() { o.y.max(min.y).min(max.y) } else { min.y };
        plot::line_of(f, AXES, Vec2::new(x, min.y), Vec2::new(x, max.y), 1.0, seen(o.x, min.x, max.x));
        plot::line_of(f, AXES, Vec2::new(min.x, y), Vec2::new(max.x, y), 1.0, seen(o.y, min.y, max.y));
        if !lay.words {
            return;
        }
        // where each side starts and ends: below the foot, and beside the outer side
        // (left of the inputs, right of the outputs; on a phone both on the left). (Only
        // the ends: numbers along the sides would come and go as the picture moves.)
        let decimals = (2.0 - (2.0 * win.h).log10().floor()).max(1.0).min(8.0) as usize;
        let end = |v: f64| num(v, decimals);
        let (lo, hi) = (win.c - DVec2::splat(win.h), win.c + DVec2::splat(win.h));
        f.text_px(Vec2::new(min.x, max.y + 3.0), end(lo.x), 10.5, pal::DIM, Align2::LEFT_TOP);
        f.text_px(Vec2::new(max.x, max.y + 3.0), end(hi.x), 10.5, pal::DIM, Align2::RIGHT_TOP);
        let left = k == 0 || lay.stacked;
        let (x, top, bottom) = if left { (min.x - 4.0, Align2::RIGHT_TOP, Align2::RIGHT_BOTTOM) } else { (max.x + 4.0, Align2::LEFT_TOP, Align2::LEFT_BOTTOM) };
        f.text_px(Vec2::new(x, min.y), end(hi.y), 10.5, pal::DIM, top);
        f.text_px(Vec2::new(x, max.y), end(lo.y), 10.5, pal::DIM, bottom);
        // the axes' names, at the far end of each side
        f.text_px(Vec2::new(max.x - 4.0, max.y - 3.0), names[0], 12.5, pal::DIM, Align2::RIGHT_BOTTOM);
        f.text_px(Vec2::new(min.x + 5.0, min.y + 3.0), names[1], 12.5, pal::DIM, Align2::LEFT_TOP);
    }

    /// Both pictures: the grid and its image, the patch and its two images, J's columns.
    fn picture(&mut self, f: &mut Frame, lay: &Layout, st: &State) {
        let [a, b] = lay.panes;
        let side = a.1.x - a.0.x;
        let names = self.inputs();
        self.backdrop(f, lay, 0, st.win_in, [&format!("${}$", names[0]), &format!("${}$", names[1])]);
        self.backdrop(f, lay, 1, st.win_out, ["$u$", "$v$"]);
        self.grids(st, side);
        // the grid, and its image (and the old map's, fading out, while the map glides)
        let fade = self.mix.b as f32;
        let sets = [(self.old_grid.as_ref(), 1.0 - fade), (Some(&self.grid), if self.old_grid.is_some() { fade } else { 1.0 })];
        for (grid, share) in sets {
            let Some(grid) = grid else { continue };
            for (line, image) in grid.lines.iter().zip(&grid.images) {
                // (both grids drawn alike, as curves of many points fading out at the edges)
                let along: Vec<Vec2> = (0..=LINE_POINTS).map(|i| st.win_in.px(a, line.ends[0].lerp(line.ends[1], i as f64 / LINE_POINTS as f64))).collect();
                faded(f, grid_kind(line.level, false), &along, a, EDGE_FADE, 1.0, pal::BLUE.a(0.42 * line.bright * share));
                let pts: Vec<Vec2> = image.iter().map(|q| st.win_out.px(b, *q)).collect();
                faded(f, grid_kind(line.level, true), &pts, b, EDGE_FADE, 1.1, pal::BLUE.a(0.62 * line.bright * share));
            }
        }

        let (p, h) = (st.s.p, st.s.h());
        let steps = outline(SIDE_POINTS);
        // the patch, white, with a faint fill
        let patch: Vec<Vec2> = steps.iter().map(|s| st.win_in.px(a, p + *s * h)).collect();
        let mut closed = patch.clone();
        closed.push(patch[0]);
        steady(f, CURVES, &closed, a, 1.8, pal::WHITE.a(0.9));
        let corners = [DVec2::new(-1.0, -1.0), DVec2::new(1.0, -1.0), DVec2::new(1.0, 1.0), DVec2::new(-1.0, 1.0)].map(|c| st.win_in.px(a, p + c * h).max(a.0).min(a.1));
        fill_tri(f, corners[0], corners[1], corners[2], pal::WHITE.a(0.06));
        fill_tri(f, corners[0], corners[2], corners[3], pal::WHITE.a(0.06));
        // steps along x and y, pink and green, and the turning arrow
        let px_in = |q: DVec2| st.win_in.px(a, q);
        arrow(f, ARROWS, px_in(p), px_in(p + DVec2::X * h), a, 2.2, pal::PINK);
        arrow(f, ARROWS, px_in(p), px_in(p + DVec2::Y * h), a, 2.2, pal::GREEN);
        let turn = turning(24);
        let arc: Vec<Vec2> = turn.iter().map(|t| px_in(p + *t * h)).collect();
        self.turning_arrow(f, CURVES, &arc, a, pal::WHITE.a(0.55));

        // the true image of the patch: yellow, filled (a fine mesh, so that where the map
        // folds it over itself it is filled twice)
        let px_out = |q: DVec2| st.win_out.px(b, q);
        let true_image: Vec<Vec2> = steps.iter().map(|s| px_out(self.mix.at(p + *s * h))).collect();
        let m = 8;
        let mesh: Vec<Vec2> = (0..=m).flat_map(|j| (0..=m).map(move |i| (i, j))).map(|(i, j)| px_out(self.mix.at(p + DVec2::new(i as f64 / m as f64 * 2.0 - 1.0, j as f64 / m as f64 * 2.0 - 1.0) * h))).collect();
        for j in 0..m {
            for i in 0..m {
                let at = |i: usize, j: usize| mesh[j * (m + 1) + i].max(b.0).min(b.1);
                fill_tri(f, at(i, j), at(i + 1, j), at(i + 1, j + 1), pal::YELLOW.a(0.18));
                fill_tri(f, at(i, j), at(i + 1, j + 1), at(i, j + 1), pal::YELLOW.a(0.18));
            }
        }
        let mut ring = true_image.clone();
        ring.push(true_image[0]);
        steady(f, TRUE_IMAGE, &ring, b, 2.2, pal::YELLOW);
        let arc_out: Vec<Vec2> = turn.iter().map(|t| px_out(self.mix.at(p + *t * h))).collect();
        self.turning_arrow(f, TURNING, &arc_out, b, pal::WHITE.a(0.55));

        // where J takes the patch: dashed (a parallelogram, a segment or a point)
        let fp = st.fp;
        if fp.is_finite() && st.j.is_finite() {
            let lin: Vec<Vec2> = [DVec2::new(-1.0, -1.0), DVec2::new(1.0, -1.0), DVec2::new(1.0, 1.0), DVec2::new(-1.0, 1.0), DVec2::new(-1.0, -1.0)].iter().map(|c| px_out(fp + st.j * (*c * h))).collect();
            pieces(f, LINEAR, true, &lin, b, 2.0, pal::WHITE);
            // (J = 0 sends the whole patch to the one point f(p): a ring round it)
            let zero = rank(st.j) == Rank::Zero;
            plot::disc(f, px_out(fp), 12.0, if zero { pal::WHITE.a(0.9) } else { Color::CLEAR });
            // J's columns, times h: where the steps along x and y go
            arrow(f, OUT_ARROWS, px_out(fp), px_out(fp + st.j.col(0) * h), b, 2.2, pal::PINK);
            arrow(f, OUT_ARROWS, px_out(fp), px_out(fp + st.j.col(1) * h), b, 2.2, pal::GREEN);
            // the misfit: where the two outlines are furthest apart
            let at = self.misfit_at(p, h, &steps, f.real_dt as f64);
            let (t, l) = (px_out(self.mix.at(p + at * h)), px_out(fp + st.j * (at * h)));
            pieces(f, MARKS, false, &[t, l], b, 2.4, pal::RED);
            plot::disc(f, px_out(fp), 6.0, pal::WHITE);
        }
        // the arrow from the one picture to the other
        if lay.words {
            let (s, e) = lay.arrow;
            arrow(f, ARROWS, s, e, (Vec2::ZERO, lay.view), 1.6, pal::TEXT.a(0.8));
            let (at, align) = if lay.stacked { (Vec2::new(s.x - 8.0, (s.y + e.y) * 0.5), Align2::RIGHT_CENTER) } else { (Vec2::new((s.x + e.x) * 0.5, s.y - 6.0), Align2::CENTER_BOTTOM) };
            f.text_px(at, "$f$", 15.0, pal::TEXT, align);
        }
        if lay.words && side >= 150.0 {
            self.names(f, lay, st, &patch, &true_image);
        }
    }

    /// The names written in the pictures, each where it has room (and keeps it).
    fn names(&self, f: &mut Frame, lay: &Layout, st: &State, patch: &[Vec2], true_image: &[Vec2]) {
        let [a, b] = lay.panes;
        let (p, h) = (st.s.p, st.s.h());
        // a room for each picture: nothing written outside it
        let room_in = |f: &Frame, rect: Rect| {
            let mut room = Room::new(f);
            let v = lay.view;
            let (min, max) = rect;
            for (lo, hi) in [(Vec2::ZERO, Vec2::new(v.x, min.y)), (Vec2::new(0.0, max.y), v), (Vec2::new(0.0, min.y), Vec2::new(min.x, max.y)), (Vec2::new(max.x, min.y), Vec2::new(v.x, max.y))] {
                room.wall(lo, hi);
            }
            room
        };
        let ring = |room: &mut Room, pts: &[Vec2]| {
            for i in 0..pts.len() {
                room.segment_px(pts[i], pts[(i + 1) % pts.len()]);
            }
        };
        let at = |f: &Frame, q: Vec2| f.hud(q);
        // the inputs: p
        let mut room = room_in(f, a);
        ring(&mut room, patch);
        let px_in = |q: DVec2| st.win_in.px(a, q);
        for d in [DVec2::X, DVec2::Y] {
            room.segment_px(px_in(p), px_in(p + d * h));
        }
        let corner = px_in(p + DVec2::new(-h, -h));
        if inside_rect(corner, a) {
            room.label(f, "p", at(f, corner), "patch, side $s$", 12.0, pal::WHITE.a(0.85), Vec2::new(-1.0, 1.0), 4.0, true);
        }
        // the outputs: f(p), the two images of the patch
        if !(st.fp.is_finite() && st.j.is_finite()) {
            return;
        }
        let mut room = room_in(f, b);
        ring(&mut room, true_image);
        let px_out = |q: DVec2| st.win_out.px(b, q);
        let lin = [DVec2::new(-1.0, -1.0), DVec2::new(1.0, -1.0), DVec2::new(1.0, 1.0), DVec2::new(-1.0, 1.0)].map(|c| px_out(st.fp + st.j * (c * h)));
        ring(&mut room, &lin);
        // (no name inside the images of the patch: written over the yellow it is lost)
        let (lo, hi) = true_image.iter().chain(&lin).filter(|q| q.is_finite()).fold((Vec2::splat(f32::MAX), Vec2::splat(f32::MIN)), |(lo, hi), q| (lo.min(*q), hi.max(*q)));
        if lo.x <= hi.x {
            room.wall(lo - Vec2::splat(2.0), hi + Vec2::splat(2.0));
        }
        for k in 0..2 {
            room.segment_px(px_out(st.fp), px_out(st.fp + st.j.col(k) * h));
        }
        let steps = outline(SIDE_POINTS);
        let misfit = fit(&self.mix, p, h, &steps).map(|_| {
            let s = on_outline(self.misfit_place, SIDE_POINTS);
            (px_out(self.mix.at(p + s * h)), px_out(st.fp + st.j * (s * h)))
        });
        if let Some((t, l)) = misfit {
            room.segment_px(t, l);
        }
        // (the yellow name by the corner the patch's top-right corner went to, the dashed
        // one by the corner its bottom-left went to: apart, whatever the map does)
        let top_right = true_image[2 * SIDE_POINTS];
        if inside_rect(top_right, b) {
            let away = (top_right - px_out(st.fp)).normalize_or_zero();
            room.label(f, "yellow", at(f, top_right), r"$f(\text{patch})$", 13.0, pal::YELLOW, away, 6.0, true);
        }
        if inside_rect(lin[0], b) {
            let away = (lin[0] - px_out(st.fp)).normalize_or_zero();
            room.label(f, "dashed", at(f, lin[0]), r"$f(p) + Jh$", 13.0, pal::WHITE, away, 6.0, true);
        }
        // the misfit, where it is long enough to be seen
        if let Some((t, l)) = misfit.filter(|(t, l)| (*t - *l).length() >= 6.0 && inside_rect(*t, b) && inside_rect(*l, b)) {
            let mid = (t + l) * 0.5;
            let away = (mid - px_out(st.fp)).normalize_or_zero();
            room.label(f, "misfit", at(f, mid), "misfit", 12.0, pal::RED, away, 5.0, true);
        }
    }

    /// Where on the outline the misfit is marked, as a step from p in units of h: where
    /// it is largest, or, while the place marked last is within a fiftieth of that,
    /// there still (where two places are about as far apart as each other, as the four
    /// corners are for the bend, the mark would otherwise jump between them from frame to
    /// frame); and when it does move, the mark glides round the outline to its new place,
    /// `dt` seconds nearer to it.
    fn misfit_at(&mut self, p: DVec2, h: f64, steps: &[DVec2], dt: f64) -> DVec2 {
        let n = steps.len();
        if let Some(best) = fit(&self.mix, p, h, steps) {
            let (fp, j) = self.mix.jet(p);
            let kept = self.misfit_shown.min(n - 1);
            let gap = (self.mix.at(p + steps[kept] * h) - fp - j * (steps[kept] * h)).length();
            // (and where the place kept has no misfit to measure, not there)
            if gap.is_nan() || gap < 0.98 * best.misfit {
                self.misfit_shown = best.worst;
            }
        }
        // (the short way round)
        let total = n as f64;
        let to = self.misfit_shown as f64;
        let mut d = (to - self.misfit_place).rem_euclid(total);
        if d > 0.5 * total {
            d -= total;
        }
        if d.abs() < 1e-3 || !self.misfit_place.is_finite() {
            self.misfit_place = to;
        } else {
            self.misfit_place = (self.misfit_place + d * (1.0 - (-dt.max(0.0) / 0.1).exp())).rem_euclid(total);
        }
        on_outline(self.misfit_place, n / 4)
    }

    /// Is the mark of the misfit on its way round the outline?
    fn misfit_moving(&self) -> bool {
        self.misfit_place != self.misfit_shown as f64
    }

    /// A curve with an arrowhead at its end (the turning arrow), kept inside `rect`.
    fn turning_arrow(&self, f: &mut Frame, kind: u8, pts: &[Vec2], rect: Rect, color: Color) {
        steady(f, kind, pts, rect, 1.3, color);
        let n = pts.len();
        if n >= 3 {
            let (tip, from) = (pts[n - 1], pts[n - 3]);
            let d = (tip - from).normalize_or_zero();
            if d != Vec2::ZERO && inside_rect(tip, rect) {
                let side = d.perp() * 3.5;
                let base = tip - d * 8.0;
                let [t0, t1, t2] = [tip, base + side, base - side].map(|p| f.hud_at(2, p));
                f.dl.tris_alpha.tri(t0, t1, t2, color);
            }
        }
    }
}

impl Scene for Jacobian {
    fn view(&self) -> View {
        View::Flat { center: Vec2::ZERO, half_height: 1.0 }
    }

    fn settings(&self, s: &mut ViewSettings) {
        s.grid = false;
        s.bloom = s.bloom.min(0.2);
    }

    fn animated(&self) -> bool {
        !self.at_rest() || self.dragging || self.misfit_moving()
    }

    fn pointer_hint(&self) -> Option<&'static str> {
        Some("white dot: drag the point $p$    buttons under the pictures: zoom in and out")
    }

    fn ui(&mut self, ui: &mut egui::Ui, _: &Gfx) {
        ui::note(ui, r"A map $f$ sends each point $(x, y)$ of the plane on the left to a point $(u, v)$ of the plane on the right, and may bend the plane as it goes: see what it does to the blue grid. Yet where $f$ is smooth, a small square patch round a point $p$ (white) lands on a patch (yellow) that is nearly a parallelogram. Up close, what $f$ does to a small step from $p$ is a linear map, whose matrix $J$ holds the partial derivatives at $p$; the dashed parallelogram is where $J$ sends the patch. For a small step $h$,");
        ui::equation(ui, r"f(p + h) \approx f(p) + Jh, \qquad J = \begin{bmatrix} \partial u/\partial x & \partial u/\partial y \\ \partial v/\partial x & \partial v/\partial y \end{bmatrix}");
        ui::note(ui, r"$J$'s first column is where a step along $x$ goes (pink), its second where a step along $y$ goes (green). What is left over shrinks like the square of the step: halve the patch and the misfit between yellow and dashed halves, as a share of the patch's side. And $\det J$ is the factor by which small areas near $p$ grow.");
        ui::section(ui, "Things to try");
        for (k, trial) in TRIALS.iter().enumerate() {
            if ui::experiment(ui, trial.label, trial.look_for) {
                self.set_up(k);
            }
        }

        ui::section(ui, "The map");
        let mut k = PRESETS.iter().position(|p| p.kind == self.kind).unwrap_or(PRESETS.len());
        let mut names: Vec<&str> = PRESETS.iter().map(|p| p.name).collect();
        names.push("your own");
        if ui::choice(ui, &names, &mut k) {
            match PRESETS.get(k) {
                Some(p) => self.pick(p.kind),
                None => self.typed(),
            }
        }
        if let Some(pre) = preset(self.kind) {
            ui::equation(ui, pre.formula);
            ui::equation(ui, &format!(r"{}, \quad {}", pre.jacobian, pre.det));
        }
        let mut changed = self.own[0].ui(ui, "u =");
        changed |= self.own[1].ui(ui, "v =");
        if changed {
            self.typed();
        }
        ui::note(ui, r"Type $u$ and $v$ in $x$ and $y$ to make the map your own (in polar coordinates $x$ stands for $r$ and $y$ for $\theta$). $J$ is worked out exactly from the formulas, not from differences.");

        ui::section(ui, "Patch and zoom");
        let to = self.glide.to;
        let mut side = (2.0 * to.h0) as f32;
        if ui::slider(ui, "side of the patch at zoom 1", &mut side, (2.0 * H0_RANGE[0]) as f32..=(2.0 * H0_RANGE[1]) as f32) {
            self.glide.now(|s| s.h0 = (side as f64 * 0.5).max(H0_RANGE[0]).min(H0_RANGE[1]));
        }
        let mut times = to.zoom.exp2() as f32;
        if ui::slider_log(ui, "magnified", &mut times, 1.0..=ZOOM_MOST.exp2() as f32) {
            self.zoom_to((times.max(1.0) as f64).log2());
        }
        ui::note(ui, r"Magnifying blows both pictures up, about $p$ and about $f(p)$ alike, and shrinks the patch to match: the patch looks the same size while its side $s$ shrinks, and the yellow outline comes closer to the dashed one.");

        ui::section(ui, "Reading the picture");
        ui::key(
            ui,
            &[
                (pal::WHITE, r"white: the patch, a square of side $s$ round the point $p$ (drag it), with an arrow turning anticlockwise inside; on the right, the dot $f(p)$"),
                (pal::BLUE, "blue: a square grid (left), and where $f$ takes it (right)"),
                (pal::YELLOW, r"yellow: where $f$ takes the patch, and its turning arrow"),
                (pal::WHITE.a(0.7), r"dashed: where $J$ takes the patch, the points $f(p) + Jh$; a segment, or a point, where $\det J = 0$"),
                (pal::PINK, r"pink: a step of $s/2$ along $x$, and where $J$ takes it: $s/2$ times $J$'s first column"),
                (pal::GREEN, r"green: the same along $y$, and $J$'s second column"),
                (pal::RED, "red: the misfit, where the yellow outline and the dashed one are furthest apart"),
            ],
        );

        ui::section(ui, "Words");
        ui::term(ui, "Map of the plane:", r"a rule that sends each point $(x, y)$ to a point $(u, v) = f(x, y)$, given by two formulas, $u(x, y)$ and $v(x, y)$.");
        ui::term(ui, r"Partial derivative $\partial u/\partial x$:", r"how fast $u$ changes as $x$ changes with $y$ held fixed (scene surface).");
        ui::term(ui, "Jacobian matrix:", r"$J$, the four partial derivatives at a point, in the order that makes $Jh$ the change in $(u, v)$ for a small step $h = (h_1, h_2)$: $J$'s first column times $h_1$, plus its second times $h_2$.");
        ui::term(ui, "Linear map:", r"one that sends each step $h$ to $Jh$, for a fixed matrix $J$. It keeps lines straight and parallel lines parallel, and takes squares to parallelograms (scene linmap2).");
        ui::term(ui, "Determinant:", r"$\det J = ad - bc$ for $J = \begin{bmatrix} a & b \\ c & d \end{bmatrix}$: the factor by which $J$ scales areas, negative when it turns them over (scene det).");
        ui::term(ui, "Misfit:", r"the furthest a point of the patch's outline lands from where $J$ sends it, divided by the patch's side $s$. Where $f$'s second derivatives are continuous, for small patches it is at most a fixed number times $s$; and if they are not all $0$ at $p$, halving a small patch halves it.");
        ui::term(ui, "Signed area:", r"area counted negative where an outline goes round clockwise, as that of a patch turned over does. Divided by the patch's own area, the signed area inside the yellow outline is the average of $\det J$ over the patch, and so tends to $\det J$ at $p$ as the patch shrinks.");
        ui::term(ui, "Inverse:", r"a map that undoes $f$. If $f$'s partial derivatives are continuous and $\det J \ne 0$ at $p$, $f$ can be undone near $p$, by a map whose Jacobian is $J^{-1}$. Where $\det J = 0$ no map with a Jacobian can undo it there, and the folds and pinches here cannot be undone at all.");

        ui::section(ui, "Numbers");
        let st = self.state();
        let names = self.inputs();
        ui::readout(ui, "$p$", format!("${}$", tuple(&[st.s.p.x, st.s.p.y], 2)));
        if st.fp.is_finite() && st.j.is_finite() {
            ui::readout(ui, "$f(p)$", format!("${}$", tuple(&[st.fp.x, st.fp.y], 2)));
            let j = st.j;
            ui::readout(ui, "$J$ at $p$", format!("${}$", bmatrix(&[[j.x_axis.x, j.y_axis.x], [j.x_axis.y, j.y_axis.y]], 2)));
            ui::readout(ui, r"$\det J$ at $p$", format!("${}$", num(j.determinant(), 2)));
            ui::note(ui, &det_words(j));
        } else {
            ui::readout(ui, "$f(p)$", "none: $f$ has no value or no slope here");
        }
        let h = st.s.h();
        ui::readout(ui, "side $s$ of the patch", format!("${}$", fmt((2.0 * h) as f32)));
        match (share(&self.mix, st.s.p, h), share(&self.mix, st.s.p, 0.5 * h)) {
            (Some(now), Some(_)) if now < 1e-12 => ui::readout_colored(ui, "misfit", "none: $f$ is linear here", pal::RED),
            (Some(now), Some(half)) => {
                ui::readout_colored(ui, "misfit, as a share of $s$", format!("{} %", percent(now)), pal::RED);
                ui::readout_colored(ui, "with half the side", format!("{} %", percent(half)), pal::RED);
            }
            _ => ui::readout(ui, "misfit", "none"),
        }
        if let Some(ratio) = area_ratio(&self.mix, st.s.p, h) {
            ui::readout(ui, "signed area, yellow ÷ white", format!("${}$", num(ratio, 2)));
        }
        let _ = names;
        ui::see_also(
            ui,
            &[
                ("surface", "the one-row case: the gradient of a surface"),
                ("linmap2", "linear maps of the plane, and their matrices"),
                ("det", "the determinant as a factor of areas"),
                ("changevar", r"$|\det J|$ in an integral: changing variables"),
            ],
        );
    }

    fn frame(&mut self, f: &mut Frame) {
        hold_batches(f);
        let view = f.view_size();
        let lay = Layout::of(view, f.top_inset);
        let covered = view.x < 160.0 || view.y < 120.0;
        // the board's buttons, before anything else takes the click
        if let Some(board) = lay.board.filter(|_| !covered) {
            if plot::pointer(f).is_some_and(|q| inside_rect(q, board.rect)) {
                f.handles.capture();
                f.handles.claim_clicks();
            }
            let z = self.glide.to.zoom;
            if plot::button(f, board.buttons[0], "zoom out", board.text, false) {
                self.zoom_to((z - 1.0).ceil().max(0.0));
            }
            if plot::button(f, board.buttons[1], "zoom in", board.text, false) {
                self.zoom_to((z + 1.0).floor());
            }
        } else {
            plot::line_of(f, plot::BUTTONS, Vec2::ZERO, Vec2::ZERO, 1.0, Color::CLEAR);
        }
        // the handle: p, in the inputs
        let st = self.state();
        let a = lay.panes[0];
        let at = st.win_in.px(a, st.s.p);
        self.dragging = false;
        if inside_rect(at, (a.0 - Vec2::splat(1.0), a.1 + Vec2::splat(1.0))) {
            if let Some(to) = plot::screen_handle(f, 1, at, pal::WHITE) {
                let q = st.win_in.back(a, to.max(a.0).min(a.1));
                let q = st.win_in.keep(st.domain.keep(q, 0.98), 0.97);
                if q.is_finite() {
                    let k = st.s.s();
                    self.offset -= (q - st.s.p) * (1.0 - k);
                    self.glide.now(|s| s.p = q);
                    self.film = None;
                    self.film_waiting = None;
                }
                self.dragging = true;
            }
        }
        self.advance(f.dt as f64, f.real_dt as f64);
        let st = self.state();
        self.picture(f, &lay, &st);
        if covered || !lay.words {
            return;
        }
        self.titles(f, &lay);
        if let Some(b) = &lay.board {
            self.board(f, b, &st);
        }
    }
}

impl Jacobian {
    /// The titles over the two pictures.
    fn titles(&self, f: &mut Frame, lay: &Layout) {
        let [a, b] = lay.panes;
        let names = self.inputs();
        let pair = format!("$({}, {})$", names[0], names[1]);
        let size = 13.0;
        let room_in = if lay.stacked { lay.view.x - a.0.x - 8.0 } else { b.0.x - a.0.x - 12.0 };
        let left = [format!("inputs {pair}, and a patch round $p$"), format!("inputs {pair}")];
        let text = left.iter().find(|t| text_width(t, size) <= room_in).unwrap_or(&left[1]);
        f.text_px(Vec2::new(a.0.x + 1.0, a.0.y - 6.0), text.as_str(), size, pal::TEXT, Align2::LEFT_BOTTOM);
        let room_out = if lay.stacked { lay.arrow.0.x - 20.0 - b.0.x } else { lay.view.x - b.0.x - 8.0 };
        let outputs = format!("outputs $(u, v) = f({}, {})$", names[0], names[1]);
        let right = [preset(self.kind).map(|p| format!("outputs: ${}$", p.formula)).unwrap_or_else(|| outputs.clone()), outputs, "outputs".to_string()];
        let text = right.iter().find(|t| text_width(t, size) <= room_out).unwrap_or(&right[2]);
        f.text_px(Vec2::new(b.0.x + 1.0, b.0.y - 6.0), text.as_str(), size, pal::TEXT, Align2::LEFT_BOTTOM);
    }

    /// The numbers under the pictures.
    fn board(&self, f: &mut Frame, b: &Board, st: &State) {
        let t = b.text;
        let (fp, j, p) = (st.fp, st.j, st.s.p);
        let h = st.s.h();
        let good = fp.is_finite() && j.is_finite();
        let mut y = b.left.y;
        let x = b.left.x;
        let pre = preset(self.kind);
        // J, written out and with its numbers at p, the columns in their colours
        let msize = matrix_size(2, 2, 5, t);
        let mid = y + msize.y * 0.5;
        let mut cx = x;
        if let (true, Some(pre)) = (b.two, pre) {
            let w = f.text_size(&format!("${}$", pre.jacobian), t).x;
            f.text_px(Vec2::new(cx, mid), format!("${}$", pre.jacobian), t, pal::TEXT, Align2::LEFT_CENTER);
            cx += w + 16.0;
        }
        let at_p = "at $p$:  $J =$";
        f.text_px(Vec2::new(cx, mid), at_p, t, pal::TEXT, Align2::LEFT_CENTER);
        cx += f.text_size(at_p, t).x + 6.0;
        if good {
            let cell = |v: f64, c: Color| (format!("${}$", num(v, 2)), c);
            let rows = vec![vec![cell(j.x_axis.x, pal::PINK), cell(j.y_axis.x, pal::GREEN)], vec![cell(j.x_axis.y, pal::PINK), cell(j.y_axis.y, pal::GREEN)]];
            cx += hud_matrix(f, Vec2::new(cx, y), &rows, 5, t, pal::DIM).x + 14.0;
            f.text_px(Vec2::new(cx, mid), format!(r"$\det J = {}$", num(j.determinant(), 2)), t, pal::TEXT, Align2::LEFT_CENTER);
        } else {
            f.text_px(Vec2::new(cx, mid), "none: $f$ has no value or no slope at $p$", t, pal::TEXT, Align2::LEFT_CENTER);
        }
        y += msize.y + 6.0;
        if good {
            f.text_px(Vec2::new(x, y), det_words(j), t, pal::TEXT, Align2::LEFT_TOP);
        }
        if b.two {
            let key = [(pal::YELLOW, "yellow: where $f$ takes the patch"), (pal::WHITE, "dashed: where $J$ takes it"), (pal::RED, "red: the misfit")];
            let mut kx = x;
            let ky = y + 2.0 * b.line;
            for (c, text) in key {
                f.text_px(Vec2::new(kx, ky), text, t - 1.0, c, Align2::LEFT_TOP);
                kx += f.text_size(text, t - 1.0).x + 18.0;
            }
        }
        // the zoom, the misfit, the areas
        let (x, mut y) = if b.two { (b.right.x, b.right.y) } else { (x, y + b.line) };
        let times = st.s.zoom.exp2();
        let zoom = if times < 1.05 { "not magnified".to_string() } else { format!("magnified {} times", num(times, if times < 9.95 { 1 } else { 0 })) };
        f.text_px(Vec2::new(x, y), format!("{zoom}: the patch's side is $s = {}$", fmt((2.0 * h) as f32)), t, pal::TEXT, Align2::LEFT_TOP);
        y += b.line;
        let misfit = match (share(&self.mix, p, h), share(&self.mix, p, 0.5 * h)) {
            (Some(now), Some(_)) if now < 1e-12 => "misfit: none: on this patch $f$ is linear".to_string(),
            (Some(now), Some(half)) => format!("misfit: {} % of $s$; with half the side, {} %", percent(now), percent(half)),
            _ => "misfit: none to measure".to_string(),
        };
        f.text_px(Vec2::new(x, y), misfit, t, pal::RED, Align2::LEFT_TOP);
        y += b.line;
        if b.two {
            if let Some(ratio) = area_ratio(&self.mix, p, h) {
                f.text_px(Vec2::new(x, y), format!(r"signed area inside yellow ÷ white: ${}$", num(ratio, 2)), t, pal::TEXT, Align2::LEFT_TOP);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The scene as an experiment's button leaves it, run at 60 frames a second until
    /// it has come to rest (from the end of another experiment, as a visitor would come).
    fn run(k: usize) -> Jacobian {
        let mut scene = Jacobian::new();
        scene.set_up((k + 2) % TRIALS.len());
        for _ in 0..60 * 20 {
            scene.advance(1.0 / 60.0, 1.0 / 60.0);
        }
        scene.set_up(k);
        for _ in 0..60 * 30 {
            scene.advance(1.0 / 60.0, 1.0 / 60.0);
            if scene.at_rest() {
                break;
            }
        }
        assert!(scene.at_rest(), "experiment {k} has not come to rest");
        scene
    }

    /// The numbers behind the captions (run with --ignored --nocapture).
    #[test]
    #[ignore]
    fn explore() {
        for k in 0..TRIALS.len() {
            let scene = run(k);
            let st = scene.state();
            let (p, h) = (st.s.p, st.s.h());
            let j = st.j;
            println!("experiment {k}: p {p:?} zoom {} h {h} fp {:?}", st.s.zoom, st.fp);
            println!("  J {:?} det {} rank {:?}", j.to_cols_array(), j.determinant(), rank(j));
            println!("  columns {:?} {:?} lengths {} {} angle {}", j.col(0), j.col(1), j.col(0).length(), j.col(1).length(), j.col(0).to_angle().to_degrees());
            for z in [0.0f64, 1.0, 2.0, 3.0, 4.0, 6.0, 10.0] {
                let hz = st.s.h0 * (-z).exp2();
                println!("  zoom 2^{z}: share {:?} area {:?} det {}", share(&scene.mix, p, hz), area_ratio(&scene.mix, p, hz), j.determinant());
            }
            println!("  words: {}", det_words(j));
        }
    }

    /// Formulas of every kind the box takes: the presets', and functions, powers,
    /// quotients and two-argument functions.
    const FORMULAS: [(&str, &str); 10] = [
        ("x + 0.4 y^2", "y + 0.4 x^2"),
        ("x cos(y)", "x sin(y)"),
        ("x^2 - y^2", "2 x y"),
        ("x", "y^2"),
        ("x + 0.5 y", "0.25 x + y"),
        ("exp(0.5 x) cos(y)", "exp(0.5 x) sin(y)"),
        ("sin(x) + y^3", "x / (1 + y^2)"),
        ("sqrt(4 + x^2 + y^2)", "atan2(y, x + 3)"),
        ("tanh(x - y) + ln(2 + cos(x y))", "pow(1.5 + x^2, 0.5 y) - abs(x - 3)"),
        ("sigmoid(x) gauss(y)", "asin(0.3 x) + cosh(0.5 y) + 2^x"),
    ];

    fn formulas(k: usize) -> Formulas {
        Formulas::parse(FORMULAS[k].0, FORMULAS[k].1).expect("parses")
    }

    fn whole(map: Formulas) -> Mix {
        Mix { from: map.clone(), to: map, b: 1.0 }
    }

    fn random_points(seed: u64, n: usize, half: f64) -> Vec<DVec2> {
        let mut rng = Rng::new(seed);
        (0..n).map(|_| DVec2::new(rng.range(-half as f32, half as f32) as f64, rng.range(-half as f32, half as f32) as f64)).collect()
    }

    /// In f64 the formulas give what the library's f32 evaluation gives, to f32's
    /// precision; and J, worked out exactly by the dual numbers, is what the library's
    /// `Dual` gives and what central differences of the map give.
    #[test]
    fn the_jacobian_is_exact_and_agrees_with_differences() {
        let mut checked = 0;
        for k in 0..FORMULAS.len() {
            let map = formulas(k);
            let (u, v) = (Expr::parse(FORMULAS[k].0, &["x", "y"]).unwrap(), Expr::parse(FORMULAS[k].1, &["x", "y"]).unwrap());
            for q in random_points(k as u64 + 1, 60, 1.4) {
                let (fq, j) = map.jet(q);
                assert!((fq - map.at(q)).length() <= 1e-12 * (1.0 + fq.length()) || !fq.is_finite());
                if !(fq.is_finite() && j.is_finite()) {
                    continue;
                }
                let at = [q.x as f32, q.y as f32];
                let qf = DVec2::new(at[0] as f64, at[1] as f64);
                let (fq, j) = map.jet(qf);
                // the library's f32 value and Dual derivatives
                let (fu, ux, uy) = u.grad2(&at, 0, 1);
                let (fv, vx, vy) = v.grad2(&at, 0, 1);
                let close32 = |a: f64, b: f32| (a - b as f64).abs() <= 2e-5 * (1.0 + a.abs());
                assert!(close32(fq.x, fu) && close32(fq.y, fv), "{:?} at {q}: {fq} against {fu}, {fv}", FORMULAS[k]);
                assert!(close32(j.x_axis.x, ux) && close32(j.y_axis.x, uy) && close32(j.x_axis.y, vx) && close32(j.y_axis.y, vy), "{:?} at {q}: {j:?}", FORMULAS[k]);
                // central differences of the map itself: the change for a step along x
                // over twice the step, and along y
                let d = 1e-5;
                let cx = (map.at(qf + DVec2::X * d) - map.at(qf - DVec2::X * d)) / (2.0 * d);
                let cy = (map.at(qf + DVec2::Y * d) - map.at(qf - DVec2::Y * d)) / (2.0 * d);
                let scale = 1.0 + j.to_cols_array().iter().fold(0.0f64, |m, v| m.max(v.abs()));
                assert!((cx - j.col(0)).length() < 1e-7 * scale && (cy - j.col(1)).length() < 1e-7 * scale, "{:?} at {q}: {j:?} against {cx}, {cy}", FORMULAS[k]);
                checked += 1;
            }
        }
        assert!(checked > 500, "{checked}");
    }

    /// The Jacobians and determinants the panel writes out for the maps are theirs.
    #[test]
    fn the_written_jacobians_are_right() {
        for q in random_points(9, 40, 1.4) {
            let (x, y) = (q.x, q.y);
            let rows = |m: DMat2| [m.x_axis.x, m.y_axis.x, m.x_axis.y, m.y_axis.y];
            let near = |a: [f64; 4], b: [f64; 4]| a.iter().zip(&b).all(|(a, b)| (a - b).abs() < 1e-6);
            for pre in &PRESETS {
                let (_, j) = Formulas::parse(pre.u, pre.v).unwrap().jet(q);
                let (want, det) = match pre.kind {
                    Kind::Bend => ([1.0, 0.8 * y, 0.8 * x, 1.0], 1.0 - 0.64 * x * y),
                    Kind::Polar => ([y.cos(), -x * y.sin(), y.sin(), x * y.cos()], x),
                    Kind::Square => ([2.0 * x, -2.0 * y, 2.0 * y, 2.0 * x], 4.0 * (x * x + y * y)),
                    Kind::Fold => ([1.0, 0.0, 0.0, 2.0 * y], 2.0 * y),
                    Kind::Linear => ([1.0, 0.5, 0.25, 1.0], 0.875),
                    Kind::Own => unreachable!(),
                };
                assert!(near(rows(j), want) && (j.determinant() - det).abs() < 1e-6, "{}: {j:?} at {q}", pre.name);
            }
        }
    }

    /// The misfit is of the size of the side squared: as a share of the side it halves
    /// when the side halves (for every map here whose second derivatives are not all 0
    /// at the point, exactly so for the quadratic ones), and it is 0 for a linear map.
    #[test]
    fn the_misfit_halves_when_the_side_halves() {
        let mut checked = 0;
        for k in 0..FORMULAS.len() {
            let map = whole(formulas(k));
            for p in random_points(20 + k as u64, 25, 1.2) {
                let Some(first) = share(&map, p, 0.02) else { continue };
                if k == 4 {
                    assert!(first < 1e-12, "a linear map has a misfit: {first}");
                    continue;
                }
                // the share's ratio for a side and half of it, ever nearer 2
                let ratio = |h: f64| share(&map, p, h).unwrap() / share(&map, p, h / 2.0).unwrap();
                let off: Vec<f64> = [0.004, 0.002, 0.001, 0.0005].iter().map(|h| (ratio(*h) - 2.0).abs()).collect();
                // (the quadratic maps exactly; the others within a hundredth or two, and
                // nearer as the side shrinks)
                let tol = if k == 0 || k == 2 || k == 3 { 1e-6 } else { 0.02 };
                assert!(off[3] < tol && off[3] <= 0.6 * off[0] + 1e-6, "{:?} at {p}: {off:?}", FORMULAS[k]);
                checked += 1;
            }
        }
        assert!(checked > 150, "{checked}");
    }

    /// The signed area inside the true image, over the patch's, tends to det J: exactly
    /// so here for the maps whose det J is linear in x and y (its average over the patch
    /// is its value at the middle), and within the size of the side squared for squaring.
    #[test]
    fn the_signed_area_tends_to_det_j() {
        for k in [0usize, 1, 2, 3, 5, 6] {
            let map = whole(formulas(k));
            for p in random_points(40 + k as u64, 10, 1.1) {
                let (_, j) = map.jet(p);
                let det = j.determinant();
                let gaps: Vec<f64> = [0.2, 0.1, 0.05].iter().map(|h| (area_ratio(&map, p, *h).unwrap() - det).abs()).collect();
                match k {
                    // (the outline is cut into chords: what is left is theirs)
                    0 | 1 | 3 => assert!(gaps.iter().all(|g| *g < 2e-4), "{:?} at {p}: {gaps:?}", FORMULAS[k]),
                    // 4(x² + y²) averages to 4|p|² + 8h²/3
                    2 => {
                        for (h, g) in [0.2, 0.1, 0.05].iter().zip(&gaps) {
                            assert!((g - 8.0 * h * h / 3.0).abs() < 2e-4, "at {p}: {gaps:?}");
                        }
                    }
                    _ => assert!(gaps[2] < 0.3 * gaps[0] + 1e-6, "{:?} at {p}: {gaps:?}", FORMULAS[k]),
                }
            }
        }
    }

    /// Magnifying keeps p where it is in its picture, and f(p) in its; a hand that moves
    /// p while the picture is magnified moves it under the pointer, and the picture
    /// stays still; zooming out again comes back to the whole map.
    #[test]
    fn zooming_keeps_p_in_its_place_and_a_hand_moves_it_under_the_pointer() {
        let rect: Rect = (Vec2::new(100.0, 50.0), Vec2::new(500.0, 450.0));
        let mut scene = Jacobian::new();
        let where_p = |scene: &Jacobian| {
            let st = scene.state();
            (st.win_in.px(rect, st.s.p), st.win_out.px(rect, st.fp))
        };
        let (first, first_out) = where_p(&scene);
        let mut last_h = scene.state().win_in.h;
        for z in [1.0, 2.0, 3.5, 6.0, 9.0] {
            scene.zoom_to(z);
            for _ in 0..240 {
                scene.advance(1.0 / 60.0, 1.0 / 60.0);
                let (now, now_out) = where_p(&scene);
                assert!((now - first).length() < 1e-3 && (now_out - first_out).length() < 1e-3, "{now} against {first}");
            }
            let h = scene.state().win_in.h;
            assert!(h < last_h && scene.at_rest());
            last_h = h;
        }
        // a hand, magnified 512 times: p follows the pointer, the window stays
        let st = scene.state();
        let before = st.win_in;
        let to = first + Vec2::new(-90.0, 40.0);
        let q = st.win_in.back(rect, to);
        scene.offset -= (q - st.s.p) * (1.0 - st.s.s());
        scene.glide.now(|s| s.p = q);
        let st = scene.state();
        assert!((st.win_in.c - before.c).length() < 1e-12 * before.h.max(1e-300) + 1e-15 && st.win_in.h == before.h);
        assert!((st.win_in.px(rect, st.s.p) - to).length() < 1e-3, "{}", st.win_in.px(rect, st.s.p));
        // and all the way out: the whole map, with nothing left of the hand
        scene.zoom_to(0.0);
        for _ in 0..600 {
            scene.advance(1.0 / 60.0, 1.0 / 60.0);
        }
        let st = scene.state();
        let whole = PRESETS[0].domain;
        assert!(scene.at_rest() && (st.win_in.c - whole.c).length() < 1e-12 && st.win_in.h == whole.h && scene.offset == DVec2::ZERO, "{:?}", st.win_in);
    }

    /// Whatever is typed or asked for, every number drawn is finite or left out, the
    /// words say "none" where there is nothing, and the scene comes to rest.
    #[test]
    fn degenerate_input_leaves_everything_finite() {
        let typed = [
            ("1/x", "1/y"),
            ("sqrt(x)", "ln(y)"),
            ("0", "0"),
            ("x", "x"),
            ("y^2", "x^2"),
            ("1e30 x", "y"),
            ("sqrt(-1 - x^2)", "y"),
            ("floor(3x)", "sign(y)"),
            ("x^3", "y^3"),
        ];
        for (u, v) in typed {
            for p in [DVec2::ZERO, DVec2::new(0.7, -0.3), DVec2::new(-1.2, 1.2)] {
                let mut scene = Jacobian::new();
                scene.own[0].set(u);
                scene.own[1].set(v);
                scene.typed();
                scene.glide.to.p = p;
                scene.zoom_to(3.0);
                for _ in 0..60 * 10 {
                    scene.advance(1.0 / 60.0, 1.0 / 60.0);
                }
                assert!(scene.at_rest(), "{u}, {v}");
                let st = scene.state();
                assert!(st.win_in.c.is_finite() && st.win_in.h > 0.0 && st.win_out.c.is_finite() && st.win_out.h > 0.0, "{u}, {v} at {p}: {st:?}");
                assert!(st.s.p.is_finite() && scene.own_domain.h.is_finite());
                let h = st.s.h();
                for text in [det_words(st.j), percent(share(&scene.mix, st.s.p, h).unwrap_or(f64::NAN))] {
                    assert!(!text.contains("NaN") && !text.contains("inf"), "{u}, {v}: {text}");
                }
                scene.grids(&st, 400.0);
                assert!(scene.old_grid.is_none() && scene.grid.lines.len() == scene.grid.images.len());
                for (line, image) in scene.grid.lines.iter().zip(&scene.grid.images) {
                    assert!(line.ends.iter().all(|e| e.is_finite()) && line.bright.is_finite() && image.len() == LINE_POINTS + 1);
                }
            }
        }
        // J = 0, J of rank one, and J of rank two, told apart
        assert_eq!(rank(DMat2::ZERO), Rank::Zero);
        assert_eq!(rank(DMat2::from_cols(DVec2::new(1.0, 0.0), DVec2::ZERO)), Rank::One);
        assert_eq!(rank(DMat2::from_cols(DVec2::new(1.0, 2.0), DVec2::new(2.0, 4.0))), Rank::One);
        assert_eq!(rank(DMat2::from_cols(DVec2::new(0.0, 1.0), DVec2::new(1.0, 0.0))), Rank::Two(false));
        assert_eq!(rank(DMat2::IDENTITY * 1e-3), Rank::Two(true));
    }

    /// The pictures and the numbers fit the canvas, from a phone to a desktop.
    #[test]
    fn the_layout_fits_every_canvas() {
        for (w, h) in [(1080.0, 1000.0), (580.0, 700.0), (390.0, 800.0), (390.0, 560.0), (800.0, 600.0), (1600.0, 1000.0), (240.0, 180.0), (620.0, 480.0), (1300.0, 700.0), (868.0, 700.0)] {
            for inset in [0.0, 46.0] {
                let view = Vec2::new(w, h);
                let lay = Layout::of(view, inset);
                for (min, max) in lay.panes {
                    assert!(min.cmpge(Vec2::ZERO).all() && max.cmple(view).all() && (max.x - min.x - (max.y - min.y)).abs() < 1e-3, "{w}x{h}: {lay:?}");
                    assert!(min.y >= inset || !lay.words, "{w}x{h}: under the buttons");
                }
                let [a, b] = lay.panes;
                assert!(a.1.y <= b.0.y || a.1.x <= b.0.x, "{w}x{h}: the pictures overlap");
                if let Some(board) = &lay.board {
                    assert!(board.rect.0.y >= a.1.y.max(b.1.y) && board.rect.1.cmple(view + Vec2::splat(0.5)).all(), "{w}x{h}: {board:?}");
                    assert!(board.buttons.iter().all(|r| inside_rect(r.0, board.rect) && inside_rect(r.1, board.rect)), "{w}x{h}");
                }
            }
        }
    }

    /// The grid's lines: each once, none closer than 14 pixels, the coarser ones at least
    /// as bright.
    #[test]
    fn the_grid_lines_are_each_drawn_once_and_fade_in() {
        for (lo, hi, pixels) in [(-1.2, 1.2, 470.0), (0.6, 0.9, 470.0), (-0.25, 2.75, 300.0), (0.71234, 0.71301, 400.0), (-4.6, 4.6, 260.0)] {
            let lines = grid_lines(lo, hi, pixels);
            let mut values: Vec<f64> = lines.iter().map(|l| l.0).collect();
            values.sort_by(f64::total_cmp);
            values.dedup_by(|a, b| (*a - *b).abs() < 1e-12);
            assert_eq!(values.len(), lines.len());
            let per = pixels / (hi - lo);
            for w in values.windows(2) {
                assert!((w[1] - w[0]) * per >= 14.0 - 1e-6, "{lo}..{hi}: {w:?}");
            }
            // the coarser a line's step, the brighter it is; a step five times another's
            // is never darker
            for a in &lines {
                for b in &lines {
                    if a.1 < b.1 {
                        assert!(a.2 >= b.2, "{lo}..{hi}: {a:?} {b:?}");
                    }
                }
            }
        }
    }

    /// The opening picture is the first experiment's, before its film plays.
    #[test]
    fn the_scene_opens_on_the_bend_at_zoom_one() {
        let scene = Jacobian::new();
        let st = scene.state();
        assert!(scene.kind == Kind::Bend && st.s.zoom == 0.0 && st.s.p == DVec2::from(TRIALS[0].p) && st.s.h0 == TRIALS[0].h0 && scene.at_rest());
        assert!((st.win_in.c - PRESETS[0].domain.c).length() < 1e-12 && st.win_in.h == PRESETS[0].domain.h);
        assert_eq!(percent(share(&scene.mix, st.s.p, st.s.h()).unwrap()), "9.9");
    }

    /// How far round `pts` go about `centre`: positive anticlockwise.
    fn winding(pts: &[DVec2], centre: DVec2) -> f64 {
        pts.windows(2).map(|w| (w[0] - centre).perp_dot(w[1] - centre)).sum()
    }

    #[test]
    fn zooming_in_does_what_its_caption_says() {
        let scene = run(0);
        let st = scene.state();
        assert!(scene.kind == Kind::Bend && st.s.zoom == 4.0 && st.s.h0 == 0.35 && st.s.p == DVec2::from(TRIALS[0].p));
        // the misfit at 1, 2, 4 and 16 times: halving as the side halves
        let shares: Vec<f64> = [0.0, 1.0, 2.0, 4.0].iter().map(|z: &f64| share(&scene.mix, st.s.p, st.s.h0 * (-z).exp2()).unwrap()).collect();
        assert_eq!(shares.iter().map(|s| percent(*s)).collect::<Vec<_>>(), ["9.9", "4.9", "2.5", "0.62"]);
        for w in shares.windows(2) {
            assert!((w[0] / w[1] - if w[1] == shares[3] { 4.0 } else { 2.0 }).abs() < 1e-9, "{shares:?}");
        }
        // the grid's image straightens: the bend of the image of a line across the inputs
        // shown, against the outputs shown, is sixteen times less at the end
        let bend = |s: f64| {
            let (w, out) = (PRESETS[0].domain.h * s, PRESETS[0].domain.h * s);
            let p = st.s.p;
            let (a, m, b) = (scene.mix.at(p - DVec2::X * w), scene.mix.at(p), scene.mix.at(p + DVec2::X * w));
            (m - (a + b) * 0.5).length() / out
        };
        assert!((bend(1.0) / bend(1.0 / 16.0) - 16.0).abs() < 1e-6, "{} {}", bend(1.0), bend(1.0 / 16.0));
        // and it played as a film, the zoom easing in and out by the scene's own time
        let mut scene = Jacobian::new();
        scene.set_up(0);
        let mut zooms = Vec::new();
        for _ in 0..60 * 8 {
            scene.advance(1.0 / 60.0, 1.0 / 60.0);
            zooms.push(scene.glide.shown.zoom);
        }
        assert!(zooms.windows(2).all(|w| w[1] >= w[0]) && zooms[zooms.len() - 1] == 4.0);
        let mut paused = Jacobian::new();
        paused.set_up(0);
        for _ in 0..60 * 8 {
            paused.advance(0.0, 1.0 / 60.0);
        }
        assert!(paused.glide.shown.zoom == 0.0 && paused.film.is_some(), "a paused scene plays no film");
    }

    #[test]
    fn polar_coordinates_do_what_their_caption_says() {
        let scene = run(1);
        let st = scene.state();
        assert!(scene.kind == Kind::Polar && st.s.p == DVec2::new(2.0, 0.5));
        let (c0, c1) = (st.j.col(0), st.j.col(1));
        // a step along r goes straight out from 0 at its own length; along θ, round, r times as long
        assert!((c0.length() - 1.0).abs() < 1e-12 && c0.perp_dot(st.fp).abs() < 1e-12 && c0.dot(st.fp) > 0.0);
        assert!((c1.length() - 2.0).abs() < 1e-12 && c0.dot(c1).abs() < 1e-12);
        assert!((st.j.determinant() - 2.0).abs() < 1e-12 && num(area_ratio(&scene.mix, st.s.p, st.s.h()).unwrap(), 2) == "2.00");
        // at r = 1 the patch keeps its area; at r = 0 the line r = 0 lands on 0
        assert_eq!(num(area_ratio(&scene.mix, DVec2::new(1.0, 0.5), st.s.h()).unwrap(), 2), "1.00");
        let (_, j0) = scene.mix.jet(DVec2::new(0.0, 0.5));
        assert!(rank(j0) == Rank::One && j0.determinant() == 0.0);
        for t in [-1.4, 0.0, 0.5, 1.4] {
            assert_eq!(scene.mix.at(DVec2::new(0.0, t)), DVec2::ZERO);
        }
        // (and r = 0 is in the inputs shown, where p can be dragged)
        assert!(PRESETS[1].domain.c.x - PRESETS[1].domain.h < 0.0);
    }

    #[test]
    fn squaring_does_what_its_caption_says() {
        let scene = run(2);
        let st = scene.state();
        assert!(scene.kind == Kind::Square && st.s.p == DVec2::new(0.8, 0.6));
        let (c0, c1) = (st.j.col(0), st.j.col(1));
        assert_eq!([num(c0.x, 1), num(c0.y, 1), num(c1.x, 1), num(c1.y, 1)], ["1.6", "1.2", "−1.2", "1.6"]);
        assert!((c0.length() - c1.length()).abs() < 1e-12 && c0.dot(c1).abs() < 1e-12 && (c0.length() - 2.0).abs() < 1e-12);
        assert_eq!(num(c0.to_angle().to_degrees(), 1), "36.9");
        assert!((st.j.determinant() - 4.0).abs() < 1e-12);
        // anywhere but 0, J is a turn and a stretch; at 0 it is 0
        for p in random_points(3, 50, 1.5) {
            let (_, j) = scene.mix.jet(p);
            assert!(j.x_axis.x == j.y_axis.y && j.x_axis.y == -j.y_axis.x && j.determinant() > 0.0, "{p}");
        }
        assert_eq!(rank(scene.mix.jet(DVec2::ZERO).1), Rank::Zero);
    }

    #[test]
    fn the_fold_does_what_its_caption_says() {
        let scene = run(3);
        let st = scene.state();
        assert!(scene.kind == Kind::Fold && st.s.p == DVec2::new(0.3, 0.0) && st.s.h0 == 0.4);
        assert!(st.j.col(1) == DVec2::ZERO && rank(st.j) == Rank::One && st.j.determinant() == 0.0);
        // each point of the strip comes from two, and the strip is s²/4 = 0.16 high
        for q in random_points(4, 20, 1.0) {
            assert_eq!(scene.mix.at(q), scene.mix.at(DVec2::new(q.x, -q.y)));
        }
        let h = st.s.h();
        let heights: Vec<f64> = outline(SIDE_POINTS).iter().map(|s| scene.mix.at(st.s.p + *s * h).y).collect();
        let (lo, hi) = (heights.iter().copied().fold(f64::MAX, f64::min), heights.iter().copied().fold(f64::MIN, f64::max));
        assert!(lo.abs() < 1e-15 && num(hi, 2) == "0.16" && num(2.0 * h, 1) == "0.8" && (hi - (2.0 * h).powi(2) / 4.0).abs() < 1e-12);
        // magnified, the strip flattens: its height against the outputs shown halves with the side
        let flat = |s: f64| (st.s.h0 * s).powi(2) / (PRESETS[3].domain.h * s);
        assert!((flat(0.5) / flat(1.0) - 0.5).abs() < 1e-12);
    }

    #[test]
    fn turned_over_does_what_its_caption_says() {
        let scene = run(4);
        let st = scene.state();
        let (p, h) = (st.s.p, st.s.h());
        assert!(scene.kind == Kind::Fold && p == DVec2::new(0.3, -0.75) && (st.j.determinant() + 1.5).abs() < 1e-12 && rank(st.j) == Rank::Two(false));
        // the patch keeps clear of the crease
        assert!(p.y + h < 0.0);
        // its bottom edge lands above its top edge; the green arrow points down
        assert!(scene.mix.at(p + DVec2::new(0.0, -h)).y > scene.mix.at(p + DVec2::new(0.0, h)).y);
        assert!(st.j.col(1).y < 0.0 && st.j.col(1).x == 0.0 && st.j.col(0).perp_dot(st.j.col(1)) < 0.0);
        // the turning arrow goes round the other way
        let arc: Vec<DVec2> = turning(24).iter().map(|t| p + *t * h).collect();
        let image: Vec<DVec2> = arc.iter().map(|q| scene.mix.at(*q)).collect();
        assert!(winding(&arc, p) > 0.0 && winding(&image, st.fp) < 0.0);
        assert_eq!(num(area_ratio(&scene.mix, p, h).unwrap(), 2), "−1.50");
        assert!(det_words(st.j).contains("turned over") && det_words(st.j).contains("1.50"));
    }
}

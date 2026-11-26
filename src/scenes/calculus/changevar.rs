//! Changing variables in an integral. Describe a region by other coordinates,
//! (x, y) = T(u, v), and each small cell Δu by Δv of a rectangle lands as a patch of
//! area about |det J| Δu Δv: so the integral over the region is the integral over the
//! rectangle with that factor in it. Two planes show the cells before and after T, each
//! tinted by how much T stretches it; under them, the sum of f over the cells worked
//! out three ways as the cells shrink.

use super::*;
use crate::scenes::stats::plot::{self, Plot, Rect};
use glam::{DMat2, DVec2};
use std::f64::consts::{PI as PI64, TAU as TAU64};

pub const DEF: SceneDef = SceneDef {
    id: "changevar",
    title: "Changing variables in an integral",
    group: "Calculus",
    blurb: "Why polar coordinates bring an extra $r$: a map stretches each small patch by the determinant of its Jacobian.",
    make: |_| Box::new(ChangeVar::new()),
};

// ----- the maps, in double precision --------------------------------------------------

/// The most cells along each side of the rectangle.
const N_MAX: usize = 32;
/// The radius R of the disc and of the ellipse: the least and the most.
const R_LEAST: f64 = 0.5;
const R_MOST: f64 = 3.0;

/// The linear map, by rows: (u, v) to (1.5u + v, 0.5u − v). Its determinant is −2.
const LINEAR: [f64; 4] = [1.5, 1.0, 0.5, -1.0];
/// The ellipse's half-axes at r = 1: (x, y) = (2r cos t, r sin t).
const ELLIPSE: [f64; 2] = [2.0, 1.0];

fn linear() -> DMat2 {
    DMat2::from_cols(DVec2::new(LINEAR[0], LINEAR[2]), DVec2::new(LINEAR[1], LINEAR[3]))
}

/// The ways of describing a region by a rectangle of other coordinates.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Map {
    /// (r, t) to (r cos t, r sin t): the disc of radius R from [0, R] × [0, 2π].
    Polar,
    /// (u, v) to A(u, v): a parallelogram from the unit square.
    Linear,
    /// (r, t) to (2r cos t, r sin t): an ellipse from [0, R] × [0, 2π].
    Ellipse,
    /// (u, v) to (u² − v², 2uv), the square of the complex number u + iv: from the
    /// unit square, the region between two parabolas above the x axis.
    Squaring,
}

const MAPS: [Map; 4] = [Map::Polar, Map::Linear, Map::Ellipse, Map::Squaring];

impl Map {
    fn index(self) -> usize {
        self as usize
    }

    fn name(self) -> &'static str {
        match self {
            Map::Polar => "polar",
            Map::Linear => "linear",
            Map::Ellipse => "ellipse",
            Map::Squaring => "squaring",
        }
    }

    /// Whether the rectangle's size is the radius R.
    fn radial(self) -> bool {
        matches!(self, Map::Polar | Map::Ellipse)
    }

    /// The names of the rectangle's two coordinates.
    fn vars(self) -> [&'static str; 2] {
        if self.radial() {
            ["r", "t"]
        } else {
            ["u", "v"]
        }
    }

    /// The rectangle: [u0, u1, v0, v1].
    fn rect(self, r: f64) -> [f64; 4] {
        if self.radial() {
            [0.0, r, 0.0, TAU64]
        } else {
            [0.0, 1.0, 0.0, 1.0]
        }
    }

    /// T itself.
    fn to(self, p: DVec2) -> DVec2 {
        match self {
            Map::Polar => DVec2::from_angle(p.y) * p.x,
            Map::Ellipse => DVec2::from_angle(p.y) * p.x * DVec2::from(ELLIPSE),
            Map::Linear => linear() * p,
            Map::Squaring => DVec2::new(p.x * p.x - p.y * p.y, 2.0 * p.x * p.y),
        }
    }

    /// J, the matrix of T's partial derivatives: its columns are where small steps
    /// along u and along v go.
    fn jacobian(self, p: DVec2) -> DMat2 {
        let (s, c) = p.y.sin_cos();
        match self {
            Map::Polar => DMat2::from_cols(DVec2::new(c, s), DVec2::new(-p.x * s, p.x * c)),
            Map::Ellipse => DMat2::from_cols(DVec2::new(ELLIPSE[0] * c, ELLIPSE[1] * s), DVec2::new(-ELLIPSE[0] * p.x * s, ELLIPSE[1] * p.x * c)),
            Map::Linear => linear(),
            Map::Squaring => DMat2::from_cols(DVec2::new(2.0 * p.x, 2.0 * p.y), DVec2::new(-2.0 * p.y, 2.0 * p.x)),
        }
    }

    /// The point of the (u, v) plane that T takes to q: the one with t in [0, 2π), or
    /// for squaring the one with u ≥ 0 (for dragging in the (x, y) plane).
    fn back(self, q: DVec2) -> DVec2 {
        let polar = |q: DVec2| DVec2::new(q.length(), q.y.atan2(q.x).rem_euclid(TAU64));
        match self {
            Map::Polar => polar(q),
            Map::Ellipse => polar(q / DVec2::from(ELLIPSE)),
            Map::Linear => linear().inverse() * q,
            Map::Squaring => DVec2::from_angle(0.5 * q.y.atan2(q.x)) * q.length().sqrt(),
        }
    }

    /// T as the picture writes it.
    fn formula(self) -> &'static str {
        match self {
            Map::Polar => r"T(r, t) = (r\cos t,\ r\sin t)",
            Map::Linear => r"T(u, v) = (1.5u + v,\ 0.5u - v)",
            Map::Ellipse => r"T(r, t) = (2r\cos t,\ r\sin t)",
            Map::Squaring => r"T(u, v) = (u^2 - v^2,\ 2uv)",
        }
    }

    /// The factor, as a formula.
    fn factor(self) -> &'static str {
        match self {
            Map::Polar => r"|\det J| = r",
            Map::Linear => r"\det J = -2",
            Map::Ellipse => r"|\det J| = 2r",
            Map::Squaring => r"|\det J| = 4(u^2 + v^2)",
        }
    }

    /// A cell's size in the (u, v) plane, as a formula.
    fn cell(self) -> &'static str {
        if self.radial() {
            r"\Delta r\,\Delta t"
        } else {
            r"\Delta u\,\Delta v"
        }
    }

    /// The same in an integral.
    fn d(self) -> &'static str {
        if self.radial() {
            r"dr\,dt"
        } else {
            r"du\,dv"
        }
    }

    /// |det J| as a formula, in the integral over the rectangle.
    fn factor_in(self) -> &'static str {
        match self {
            Map::Polar => "r",
            Map::Ellipse => "2r",
            Map::Linear => "2",
            Map::Squaring => r"4(u^2 + v^2)",
        }
    }

    /// The two integral signs of the integral over the rectangle, with its limits.
    fn limits(self) -> &'static str {
        if self.radial() {
            r"\int_0^{2\pi}\!\!\int_0^R"
        } else {
            r"\int_0^1\!\!\int_0^1"
        }
    }

    /// The top of the colour scale: the most T stretches a cell anywhere on the
    /// rectangle at the largest radius.
    fn most_stretch(self) -> f64 {
        match self {
            Map::Polar => R_MOST,
            Map::Ellipse => ELLIPSE[0] * ELLIPSE[1] * R_MOST,
            Map::Linear => 2.0,
            Map::Squaring => 8.0,
        }
    }

    /// The part of the (u, v) plane shown, [x0, x1, y0, y1]. For linear and squaring it
    /// is as large as the (x, y) plane's, so that the two are drawn to one scale and the
    /// stretch can be seen; r and t are a length and an angle, and their rectangle is
    /// drawn to fill its square. (A radial map's has room for the largest radius.)
    fn uv_box(self) -> [f64; 4] {
        match self {
            Map::Polar | Map::Ellipse => [-0.25, R_MOST + 0.25, -0.5, TAU64 + 0.5],
            Map::Linear => [-1.1, 2.1, -1.1, 2.1],
            Map::Squaring => [-0.75, 1.75, -0.75, 1.75],
        }
    }

    /// The part of the (x, y) plane shown.
    fn xy_box(self) -> [f64; 4] {
        match self {
            Map::Polar => [-3.35, 3.35, -3.35, 3.35],
            Map::Ellipse => [-6.5, 6.5, -6.5, 6.5],
            Map::Linear => [-0.35, 2.85, -1.85, 1.35],
            Map::Squaring => [-1.25, 1.25, -0.25, 2.25],
        }
    }
}

/// What is integrated.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Fun {
    /// f = 1: the integral is the region's area.
    One,
    /// The bell, e^(−(x² + y²)).
    Bell,
    /// x² + y².
    Squares,
}

const FUNS: [Fun; 3] = [Fun::One, Fun::Bell, Fun::Squares];

impl Fun {
    fn at(self, q: DVec2) -> f64 {
        match self {
            Fun::One => 1.0,
            Fun::Bell => (-q.length_squared()).exp(),
            Fun::Squares => q.length_squared(),
        }
    }

    /// f's formula.
    fn formula(self) -> &'static str {
        match self {
            Fun::One => "1",
            Fun::Bell => r"e^{-(x^2 + y^2)}",
            Fun::Squares => r"x^2 + y^2",
        }
    }

    /// What its button says.
    fn button(self) -> &'static str {
        match self {
            Fun::One => "$f = 1$",
            Fun::Bell => "the bell",
            Fun::Squares => "$x^2 + y^2$",
        }
    }
}

/// f(T(u, v)) as a formula in the rectangle's coordinates (none for f = 1).
fn composed(map: Map, fun: Fun) -> &'static str {
    match (fun, map) {
        (Fun::One, _) => "",
        (Fun::Bell, Map::Polar) => r"e^{-r^2}",
        (Fun::Bell, Map::Ellipse) => r"e^{-r^2(4\cos^2 t + \sin^2 t)}",
        (Fun::Bell, Map::Linear) => r"e^{-(1.5u + v)^2 - (0.5u - v)^2}",
        (Fun::Bell, Map::Squaring) => r"e^{-(u^2 + v^2)^2}",
        (Fun::Squares, Map::Polar) => r"r^2",
        (Fun::Squares, Map::Ellipse) => r"r^2(4\cos^2 t + \sin^2 t)",
        (Fun::Squares, Map::Linear) => r"((1.5u + v)^2 + (0.5u - v)^2)",
        (Fun::Squares, Map::Squaring) => r"(u^2 + v^2)^2",
    }
}

/// The rule written out for this map and this function: the integral over the region
/// as an integral over the rectangle.
fn rule(map: Map, fun: Fun) -> String {
    let left = match fun {
        Fun::One => r"\iint dx\,dy".to_string(),
        Fun::Bell => format!(r"\iint {}\,dx\,dy", fun.formula()),
        Fun::Squares => format!(r"\iint ({})\,dx\,dy", fun.formula()),
    };
    let (g, factor, d) = (composed(map, fun), map.factor_in(), map.d());
    let inner = if g.is_empty() {
        format!(r"{factor}\,{d}")
    } else if map == Map::Linear {
        format!(r"{factor}\,{g}\,{d}")
    } else if factor.starts_with(|c: char| c.is_ascii_digit()) {
        format!(r"{g} \cdot {factor}\,{d}")
    } else {
        format!(r"{g}\,{factor}\,{d}")
    };
    format!("{left} = {} {inner}", map.limits())
}

// ----- integrals ------------------------------------------------------------------------

/// Gauss–Legendre's 20 points on [−1, 1] and their weights, found once by Newton's
/// method on the Legendre polynomial of degree 20.
fn legendre() -> &'static [(f64, f64); 20] {
    static NODES: std::sync::OnceLock<[(f64, f64); 20]> = std::sync::OnceLock::new();
    NODES.get_or_init(|| {
        const N: usize = 20;
        // P_N and its slope at x
        let at = |x: f64| {
            let (mut p0, mut p1) = (1.0, x);
            for k in 2..=N {
                let k = k as f64;
                (p0, p1) = (p1, ((2.0 * k - 1.0) * x * p1 - (k - 1.0) * p0) / k);
            }
            (p1, N as f64 * (x * p1 - p0) / (x * x - 1.0))
        };
        let mut out = [(0.0, 0.0); N];
        for (i, node) in out.iter_mut().enumerate() {
            let mut x = (PI64 * (i as f64 + 0.75) / (N as f64 + 0.5)).cos();
            for _ in 0..60 {
                let (p, dp) = at(x);
                let step = p / dp;
                x -= step;
                if step.abs() < 1e-16 {
                    break;
                }
            }
            let (_, dp) = at(x);
            *node = (x, 2.0 / ((1.0 - x * x) * dp * dp));
        }
        out
    })
}

/// The integral of g over the rectangle [u0, u1] × [v0, v1]: cut into `panels` ×
/// `panels` pieces, each with Gauss–Legendre's 20 × 20 points. For the smooth
/// integrands here that is right to far more digits than are shown.
fn integrate(rect: [f64; 4], panels: usize, g: impl Fn(DVec2) -> f64) -> f64 {
    let [u0, u1, v0, v1] = rect;
    let panels = panels.max(1);
    let (hu, hv) = ((u1 - u0) / panels as f64, (v1 - v0) / panels as f64);
    let nodes = legendre();
    let mut total = 0.0;
    for pj in 0..panels {
        for pi in 0..panels {
            let (cu, cv) = (u0 + (pi as f64 + 0.5) * hu, v0 + (pj as f64 + 0.5) * hv);
            for &(y, wy) in nodes {
                for &(x, wx) in nodes {
                    total += wx * wy * g(DVec2::new(cu + 0.5 * hu * x, cv + 0.5 * hv * y));
                }
            }
        }
    }
    total * 0.25 * hu * hv
}

/// The pieces each way the quadratures here cut a rectangle into.
const PANELS: usize = 6;

/// The integral of f over the region, ∫∫ f dx dy, with its closed form where it has one
/// (the formula as the picture writes it). Where it has none, it is the integral over
/// the rectangle of f(T) |det J|, by quadrature: the rule the scene shows, which the
/// tests check against an integral worked out in the (x, y) plane alone.
fn exact(map: Map, fun: Fun, r: f64) -> (f64, Option<&'static str>) {
    let [a, b] = ELLIPSE;
    let closed = match (map, fun) {
        (Map::Polar, Fun::One) => Some((PI64 * r * r, r"\pi R^2")),
        (Map::Polar, Fun::Bell) => Some((PI64 * (1.0 - (-r * r).exp()), r"\pi(1 - e^{-R^2})")),
        (Map::Polar, Fun::Squares) => Some((PI64 * r.powi(4) / 2.0, r"\pi R^4/2")),
        (Map::Ellipse, Fun::One) => Some((PI64 * a * b * r * r, r"2\pi R^2")),
        (Map::Ellipse, Fun::Squares) => Some((PI64 * a * b * (a * a + b * b) * r.powi(4) / 4.0, r"5\pi R^4/2")),
        (Map::Linear, Fun::One) => Some((linear().determinant().abs(), r"|\det A|")),
        (Map::Squaring, Fun::One) => Some((8.0 / 3.0, r"8/3")),
        (Map::Squaring, Fun::Squares) => Some((96.0 / 35.0, r"96/35")),
        _ => None,
    };
    match closed {
        Some((v, text)) => (v, Some(text)),
        None => (integrate(map.rect(r), PANELS, |p| fun.at(map.to(p)) * map.jacobian(p).determinant().abs()), None),
    }
}

/// What the sum without the factor tends to as the cells shrink: the integral over the
/// rectangle of f(T(u, v)) alone, another number altogether.
fn unfactored(map: Map, fun: Fun, r: f64) -> f64 {
    integrate(map.rect(r), PANELS, |p| fun.at(map.to(p)))
}

// ----- the cells ------------------------------------------------------------------------

/// The cell i along u and j along v of n × n, as [u0, u1, v0, v1].
fn cell_of(rect: [f64; 4], n: usize, i: usize, j: usize) -> [f64; 4] {
    let [u0, u1, v0, v1] = rect;
    let n = n.max(1);
    let (du, dv) = ((u1 - u0) / n as f64, (v1 - v0) / n as f64);
    [u0 + i as f64 * du, u0 + (i + 1) as f64 * du, v0 + j as f64 * dv, v0 + (j + 1) as f64 * dv]
}

/// A cell's area in the (u, v) plane, Δu Δv, with n × n cells.
fn cell_size(map: Map, r: f64, n: usize) -> f64 {
    let [u0, u1, v0, v1] = map.rect(r);
    (u1 - u0) * (v1 - v0) / (n.max(1) * n.max(1)) as f64
}

/// The outline of a cell of the (u, v) plane (or of shares of the way across it), `m`
/// points a side, anticlockwise from its corner (u0, v0): along the bottom, up the
/// right, back along the top, down the left.
fn outline(c: [f64; 4], m: usize, mut put: impl FnMut(DVec2)) {
    let [u0, u1, v0, v1] = c;
    let m = m.max(1);
    for side in 0..4 {
        for s in 0..m {
            let t = s as f64 / m as f64;
            put(match side {
                0 => DVec2::new(u0 + (u1 - u0) * t, v0),
                1 => DVec2::new(u1, v0 + (v1 - v0) * t),
                2 => DVec2::new(u1 - (u1 - u0) * t, v1),
                _ => DVec2::new(u0, v1 - (v1 - v0) * t),
            });
        }
    }
}

/// The area enclosed by a closed polygon, by the shoelace formula: positive if it goes
/// round anticlockwise.
fn shoelace(pts: &[DVec2]) -> f64 {
    let n = pts.len();
    (0..n).map(|i| pts[i].perp_dot(pts[(i + 1) % n])).sum::<f64>() * 0.5
}

/// The points a side of each cell's outline has, with n cells along a side: enough
/// that the cells' outlines together go round each edge of the region in 48 steps.
fn points_a_side(n: usize) -> usize {
    48usize.div_ceil(n.max(1)).max(2)
}

/// The area of the image of the cell `c`, worked out from its outline alone, with no
/// Jacobian in it: the outline as a polygon of `k` points a side and again of 2k, by
/// the shoelace formula, and the two combined so that the polygon's error, which falls
/// as 1/k², cancels (Richardson's way). Negative if T turns the cell over.
fn cell_area(map: Map, c: [f64; 4], k: usize) -> f64 {
    let mut fine = Vec::with_capacity(8 * k);
    outline(c, 2 * k, |p| fine.push(map.to(p)));
    let coarse: Vec<DVec2> = fine.iter().step_by(2).copied().collect();
    (4.0 * shoelace(&fine) - shoelace(&coarse)) / 3.0
}

/// The signed areas of the images of every cell, for every number of cells along a
/// side from 1 to `N_MAX`: `by_n[n − 1][j n + i]` for the cell i along u and j along
/// v. A radial map's are at radius 1: at radius R every area is R² times as large.
struct Areas {
    by_n: Vec<Vec<f64>>,
}

impl Areas {
    fn of(map: Map) -> Areas {
        let rect = map.rect(1.0);
        let by_n = (1..=N_MAX)
            .map(|n| {
                let k = points_a_side(n);
                (0..n * n).map(|c| cell_area(map, cell_of(rect, n, c % n, c / n), k)).collect()
            })
            .collect();
        Areas { by_n }
    }

    /// The signed area of the image of the cell (i, j) of n × n, at radius r.
    fn get(&self, map: Map, r: f64, n: usize, i: usize, j: usize) -> f64 {
        let scale = if map.radial() { r * r } else { 1.0 };
        let n = n.max(1).min(N_MAX);
        self.by_n[n - 1].get(j.min(n - 1) * n + i.min(n - 1)).copied().unwrap_or(0.0) * scale
    }
}

/// How much T stretches the cell (i, j) of n × n: the area of its image over its own.
/// (By the factor at its centre until the areas are worked out.)
fn stretch(map: Map, r: f64, n: usize, i: usize, j: usize, areas: Option<&Areas>) -> f64 {
    let size = cell_size(map, r, n);
    match areas {
        Some(a) if size > 0.0 => a.get(map, r, n, i, j).abs() / size,
        _ => {
            let [u0, u1, v0, v1] = cell_of(map.rect(r), n, i, j);
            map.jacobian(DVec2::new(0.5 * (u0 + u1), 0.5 * (v0 + v1))).determinant().abs()
        }
    }
}

/// The sum of f over the cells, three ways, f taken at the image of each cell's
/// centre: times the true area of the cell's image, in the (x, y) plane; times
/// |det J| Δu Δv, in the (u, v) plane; and times Δu Δv alone, the factor left out.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
struct Sums {
    true_areas: f64,
    factor: f64,
    plain: f64,
}

fn sums(map: Map, fun: Fun, r: f64, n: usize, areas: &Areas) -> Sums {
    let rect = map.rect(r);
    let n = n.max(1).min(N_MAX);
    let size = cell_size(map, r, n);
    let mut s = Sums::default();
    for j in 0..n {
        for i in 0..n {
            let [u0, u1, v0, v1] = cell_of(rect, n, i, j);
            let centre = DVec2::new(0.5 * (u0 + u1), 0.5 * (v0 + v1));
            let value = fun.at(map.to(centre));
            s.true_areas += value * areas.get(map, r, n, i, j).abs();
            s.factor += value * map.jacobian(centre).determinant().abs() * size;
            s.plain += value * size;
        }
    }
    s
}

/// The cell of n × n that holds the point at the shares `s` of the way across.
fn picked(s: DVec2, n: usize) -> (usize, usize) {
    let n = n.max(1);
    // (a share that is not a number picks the first cell: max swallows a NaN)
    let at = |s: f64| ((s * n as f64).floor().max(0.0) as usize).min(n - 1);
    (at(s.x), at(s.y))
}

/// The point at the shares `s` of the way across the rectangle.
fn point(map: Map, r: f64, s: DVec2) -> DVec2 {
    let [u0, u1, v0, v1] = map.rect(r);
    DVec2::new(u0 + (u1 - u0) * s.x, v0 + (v1 - v0) * s.y)
}

/// The shares of the way across the rectangle that the point p is, kept inside it.
fn share(map: Map, r: f64, p: DVec2) -> DVec2 {
    let [u0, u1, v0, v1] = map.rect(r);
    let inside = |v: f64| if v.is_finite() { v.max(0.0).min(1.0 - 1e-9) } else { 0.5 };
    DVec2::new(inside((p.x - u0) / (u1 - u0)), inside((p.y - v0) / (v1 - v0)))
}

/// What is said about the picked cell.
#[derive(Clone, Copy, Debug)]
struct Picked {
    /// Its size in the (u, v) plane, the signed area of its image, |det J| at its
    /// centre, and f there.
    size: f64,
    area: f64,
    det: f64,
    value: f64,
}

fn picked_numbers(map: Map, fun: Fun, r: f64, n: usize, s: DVec2, areas: Option<&Areas>) -> Picked {
    let (i, j) = picked(s, n);
    let [u0, u1, v0, v1] = cell_of(map.rect(r), n, i, j);
    let centre = DVec2::new(0.5 * (u0 + u1), 0.5 * (v0 + v1));
    let size = cell_size(map, r, n);
    let area = match areas {
        Some(a) => a.get(map, r, n, i, j),
        None => cell_area(map, [u0, u1, v0, v1], points_a_side(n)),
    };
    Picked { size, area, det: map.jacobian(centre).determinant(), value: fun.at(map.to(centre)) }
}

// ----- numbers as the picture writes them ----------------------------------------------

/// The decimals that give a number of v's size `digits` significant figures.
fn decimals(v: f64, digits: i32) -> usize {
    if !(v.is_finite() && v != 0.0) {
        return (digits - 1).max(0) as usize;
    }
    (digits - 1 - v.abs().log10().floor() as i32).clamp(0, 8) as usize
}

/// A number to `digits` significant figures, with a true minus sign.
fn sig(v: f64, digits: i32) -> String {
    num(v, decimals(v, digits))
}

/// The same with the zeros at the end of its decimals left off: for the factors of a
/// product written out, which are given a figure more than the product, so that the
/// product of the numbers as written rounds to the product as written (but where the
/// product falls within a hair of the edge between two roundings).
fn trimmed(v: f64, digits: i32) -> String {
    let text = sig(v, digits);
    if text.contains('.') {
        text.trim_end_matches('0').trim_end_matches('.').to_string()
    } else {
        text
    }
}

fn smooth(t: f32) -> f32 {
    let t = if t.is_finite() { t.max(0.0).min(1.0) } else { 1.0 };
    t * t * (3.0 - 2.0 * t)
}

/// The colour of a cell T stretches by the share `s` of the scale's top: dark blue for
/// little, a pale violet for much.
fn tint(s: f32) -> Color {
    let s = if s.is_finite() { s.max(0.0).min(1.0) } else { 0.0 };
    let (low, mid, high) = (Color::hex(0x26305A), Color::hex(0x5B49B0), Color::hex(0xCDBFFF));
    if s < 0.5 {
        low.mix(mid, 2.0 * s)
    } else {
        mid.mix(high, 2.0 * s - 1.0)
    }
}

/// The colours of the three sums (with the true areas, with the factor and without
/// it) and of the exact value.
const TRUE_AREAS: Color = pal::YELLOW;
const FACTOR: Color = pal::CYAN;
const PLAIN: Color = pal::RED;
const EXACT: Color = pal::WHITE;

// ----- the picture as drawn -------------------------------------------------------------

/// Where a point of a plane is on the screen: the part [x0, x1, y0, y1] of the plane
/// shown in the square `rect` of the screen.
fn to_screen(b: [f64; 4], (min, max): Rect, p: DVec2) -> Vec2 {
    let x = (p.x - b[0]) / (b[1] - b[0]);
    let y = (p.y - b[2]) / (b[3] - b[2]);
    Vec2::new(min.x + x as f32 * (max.x - min.x), max.y - y as f32 * (max.y - min.y))
}

fn from_screen(b: [f64; 4], (min, max): Rect, q: Vec2) -> DVec2 {
    let x = ((q.x - min.x) / (max.x - min.x).max(1e-3)) as f64;
    let y = ((max.y - q.y) / (max.y - min.y).max(1e-3)) as f64;
    DVec2::new(b[0] + x * (b[1] - b[0]), b[2] + y * (b[3] - b[2]))
}

/// The point at the shares `s` across the rectangle, on the screen: in the (u, v)
/// plane (`plane` 0), or where T takes it in the (x, y) plane (1).
fn place(map: Map, r: f64, planes: &[Rect; 2], plane: usize, s: DVec2) -> Vec2 {
    let p = point(map, r, s);
    if plane == 0 {
        to_screen(map.uv_box(), planes[0], p)
    } else {
        to_screen(map.xy_box(), planes[1], map.to(p))
    }
}

/// The cells as drawn: under one map, or part of the way from one to another (`blend`
/// from 0 to 1, eased as it is drawn), at radius `r`, n × n.
#[derive(Clone, Copy, Debug, PartialEq)]
struct Look {
    from: Map,
    to: Map,
    blend: f32,
    r: f64,
    n: usize,
}

impl Look {
    fn eased(&self) -> f32 {
        if self.from == self.to {
            1.0
        } else {
            smooth(self.blend)
        }
    }

    /// Something worked out for each of the two maps, taken as far from the first to
    /// the second as the glide has gone.
    fn mix(&self, g: impl Fn(Map) -> Vec2) -> Vec2 {
        let b = self.eased();
        if b >= 1.0 {
            g(self.to)
        } else if b <= 0.0 {
            g(self.from)
        } else {
            g(self.from).lerp(g(self.to), b)
        }
    }

    fn mix_value(&self, g: impl Fn(Map) -> f32) -> f32 {
        let b = self.eased();
        if b >= 1.0 {
            g(self.to)
        } else {
            g(self.from) + (g(self.to) - g(self.from)) * b
        }
    }

    /// Where the point at the shares `s` across the rectangle is drawn.
    fn at(&self, planes: &[Rect; 2], plane: usize, s: DVec2) -> Vec2 {
        self.mix(|m| place(m, self.r, planes, plane, s))
    }
}

/// Seconds a glide from one map to another takes, and a fade from one number of cells
/// to another.
const MORPH: f32 = 0.6;
const FADE: f32 = 0.35;

/// Line kinds with a batch each (see `plot::apart`): the cells' outlines (as many
/// points as there are cells), those of the picture fading out, the picked cell, the
/// planes' ticks (as many as the map has), the dashed lines, and the chart's marks up
/// its side (as many as its top has).
const GRID: u8 = plot::OWN;
const GRID_OLD: u8 = plot::OWN + 1;
const PICK: u8 = plot::OWN + 2;
const TICKS: u8 = plot::OWN + 3;
const DASHED: u8 = plot::OWN + 4;
const CHART_TICKS: u8 = plot::OWN + 5;

/// Every batch of lines, made in the same order in every frame.
fn in_order(f: &mut Frame) {
    for kind in [plot::AXES, plot::CURVES, plot::MARKS, GRID, GRID_OLD, PICK, TICKS, CHART_TICKS] {
        plot::apart(f, kind, false);
    }
    plot::apart(f, DASHED, true);
}

/// A batch of lines with nothing in it this frame is given a point that cannot be seen:
/// one left empty is dropped, and every batch after it would be compared, point for
/// point, with the one before it in the last frame.
fn keep_batches(f: &mut Frame) {
    for (kind, dashed) in [(plot::AXES, false), (plot::CURVES, false), (plot::MARKS, false), (GRID, false), (GRID_OLD, false), (PICK, false), (TICKS, false), (CHART_TICKS, false), (DASHED, true)] {
        let batch = plot::apart(f, kind, dashed);
        if batch.pts.len() <= 1 {
            batch.segment(Vec3::ZERO, Vec3::ZERO, 0.0, Color::CLEAR);
        }
    }
}

// ----- experiments ----------------------------------------------------------------------

struct Trial {
    label: &'static str,
    look_for: &'static str,
    map: Map,
    fun: Fun,
    n: usize,
    r: f64,
    /// The cell picked, by the shares of the way across the rectangle of its centre.
    pick: [f64; 2],
}

const TRIALS: [Trial; 5] = [
    Trial {
        label: "polar cells are wider further out",
        look_for: r"Every cell of the rectangle is the same, $\Delta r\,\Delta t = 0.375 \times \pi/4 = 0.2945$, but $T$ fans them round the disc. A cell lands as a piece of a ring, $\Delta r$ wide and $r\,\Delta t$ long round, so its area is $r\,\Delta r\,\Delta t$ (exactly, with $r$ at its middle): $|\det J| = r$. The picked cell of the outer ring, at $r = 2.81$, has 0.8283, fifteen times a wedge of the innermost ring, at $r = 0.19$, which has 0.0552. The tint says the same: dark at the centre, bright at the rim.",
        map: Map::Polar,
        fun: Fun::One,
        n: 8,
        r: 3.0,
        pick: [15.0 / 16.0, 5.0 / 16.0],
    },
    Trial {
        label: "a linear map stretches every cell alike",
        look_for: r"$T(u, v) = (1.5u + v,\ 0.5u - v)$ is linear: $J$ is its matrix $A$ wherever the cell is. Every cell of the square lands as a parallelogram, all of one tint, of area $|\det A|\,\Delta u\,\Delta v = 2\,\Delta u\,\Delta v$, so the region's area is 2, twice the square's. That is the determinant as an area factor (scene det). And $\det A = -2$: the map turns cells over, the green side now clockwise from the pink one, but an area is never negative. The factor is $|\det J|$.",
        map: Map::Linear,
        fun: Fun::One,
        n: 6,
        r: 3.0,
        pick: [7.0 / 12.0, 5.0 / 12.0],
    },
    Trial {
        label: "the area of a disc",
        look_for: r"Add up the cells' areas: $\sum r\,\Delta r\,\Delta t = \pi R^2 = 7.0686$ for $R = 1.5$, and exactly that at every $n$, even with these 16 large cells, since $r$ at a cell's middle times $\Delta r\,\Delta t$ is its piece of ring's area, $(r_2^2 - r_1^2)\,\Delta t/2$. So the yellow and cyan lines of the chart lie on the white one, and in the limit $\int_0^{2\pi}\!\int_0^R r\,dr\,dt = \pi R^2$. Without the factor the sum is the rectangle's area, $2\pi R = 9.4248$. The picked cell is a wedge: along the edge $r = 0$, $\det J = 0$, and the whole edge lands on the one point $0$. Nothing is wrong there: an edge has no area to lose.",
        map: Map::Polar,
        fun: Fun::One,
        n: 4,
        r: 1.5,
        pick: [1.0 / 8.0, 3.0 / 8.0],
    },
    Trial {
        label: r"the bell, and where $\sqrt{\pi}$ comes from",
        look_for: r"The bell $e^{-(x^2 + y^2)}$ over the disc of radius 3. In polar coordinates it is $e^{-r^2}$, and with the factor $r$ it has an antiderivative, $-e^{-r^2}/2$, so $\int_0^{2\pi}\!\int_0^3 e^{-r^2}\,r\,dr\,dt = \pi(1 - e^{-9}) = 3.1412$. The sums close in on it as the cells shrink: 3.1790 at $n = 8$, 3.1435 at 32. The disc holds all but $\pi e^{-9} = 0.0004$ of the bell; over the whole plane its integral is $\pi$. Since $e^{-(x^2 + y^2)} = e^{-x^2} e^{-y^2}$, that is also $(\int e^{-x^2}\,dx)^2$, so $\int e^{-x^2}\,dx = \sqrt{\pi} = 1.7725$ over the whole line; with $x = s/\sqrt{2}$ that makes $\int e^{-s^2/2}\,ds = \sqrt{2\pi}$, the constant in the normal distribution. Without the factor the sum heads for 5.5682 instead.",
        map: Map::Polar,
        fun: Fun::Bell,
        n: 8,
        r: 3.0,
        pick: [5.0 / 16.0, 3.0 / 16.0],
    },
    Trial {
        label: "forget the factor, and get it wrong",
        look_for: r"Squaring, $T(u, v) = (u^2 - v^2,\ 2uv)$, carries the unit square onto the region between two parabolas and the $x$ axis, of area $8/3 = 2.6667$. Left out, the factor $|\det J| = 4(u^2 + v^2)$ leaves $\sum \Delta u\,\Delta v$, the square's own area: 1 at every $n$, so the red line stays flat as the cells shrink. It is not a rough answer that improves; it is the exact answer to another question, $\iint 1\,du\,dv$. The factor runs from 0 at the corner $(0, 0)$, where the first cell shrinks to a 24th of its size, to 8 at $(1, 1)$, where the picked one grows 7.04 times; on average it is $8/3$. With it, the sum is 2.6562 at $n = 8$ ($8/3 - 1/96$), closing in on 2.6667.",
        map: Map::Squaring,
        fun: Fun::One,
        n: 8,
        r: 3.0,
        pick: [15.0 / 16.0, 15.0 / 16.0],
    },
];

// ----- where things go ------------------------------------------------------------------

#[derive(Clone, Debug)]
struct Layout {
    /// A phone held upright: everything one under another, smaller.
    narrow: bool,
    /// The size of the words on the buttons and in the headings, and of the small print.
    text: f32,
    small: f32,
    /// The buttons: the maps, then the functions, and where "map:" and "f:" are
    /// written before them (on a wide canvas).
    maps: Vec<Rect>,
    funs: Vec<Rect>,
    leads: Option<[Vec2; 2]>,
    /// The two planes, (u, v) and (x, y): squares of the screen, their headings above.
    planes: Option<[Rect; 2]>,
    headings: [Vec2; 2],
    /// The colour scale (its strip), the lines about the picked cell (top-left and
    /// width), the sums (the same) and the chart.
    scale: Option<Rect>,
    cell: Option<(Vec2, f32)>,
    sums: Option<(Vec2, f32)>,
    /// The size of the sums' words and numbers.
    sums_size: f32,
    chart: Option<Rect>,
}

/// The room right of a plane for its axis's name.
const NAME_W: f32 = 12.0;
/// The room below a plane for its tick numbers, and beside it on the left.
const TICKS_H: f32 = 18.0;
const TICKS_W: f32 = 34.0;
/// How tall a line of the sums is, and of the lines about the picked cell.
fn line_h(size: f32) -> f32 {
    size + 8.0
}


/// Rectangles for buttons side by side from (x, y), each as wide as its words need.
fn button_row(words: &[&str], x: f32, y: f32, h: f32, size: f32, gap: f32) -> Vec<Rect> {
    let mut left = x;
    words
        .iter()
        .map(|w| {
            let width = text_width(w, size) + 18.0;
            let r = (Vec2::new(left, y), Vec2::new(left + width, y + h));
            left += width + gap;
            r
        })
        .collect()
}

impl Layout {
    fn of(view: Vec2, inset: f32) -> Layout {
        let view = view.max(Vec2::ONE);
        let narrow = view.x < 620.0;
        let (text, small) = if narrow { (11.5, 10.5) } else { (12.5, 12.0) };
        let side = if narrow { 10.0 } else { 16.0 };
        let top = inset.max(0.0) + if narrow { 6.0 } else { 12.0 };
        let h = if narrow { 26.0 } else { 28.0 };
        let mut lay = Layout {
            narrow,
            text,
            small,
            maps: Vec::new(),
            funs: Vec::new(),
            leads: None,
            planes: None,
            headings: [Vec2::ZERO; 2],
            scale: None,
            cell: None,
            sums: None,
            sums_size: if narrow { 10.5 } else { 13.0 },
            chart: None,
        };
        if view.x < 200.0 || view.y < 150.0 {
            return lay;
        }
        // the buttons: on a phone a row of maps and a row of functions, each filling the
        // width; wider, each row as long as its words, on one line if they fit
        let below_buttons;
        if narrow {
            lay.maps = plot::even_row(4, side, view.x - side, top, h, 5.0);
            lay.funs = plot::even_row(3, side, view.x - side, top + h + 5.0, h, 5.0);
            below_buttons = top + 2.0 * h + 5.0 + 12.0;
        } else {
            let lead = [text_width("map:", text) + 10.0, text_width("$f$:", text) + 10.0];
            lay.maps = button_row(&MAPS.map(Map::name), side + lead[0], top, h, text, 6.0);
            let end = lay.maps.last().map_or(side, |r| r.1.x);
            let words = FUNS.map(Fun::button);
            let funs_w: f32 = words.iter().map(|w| text_width(w, text) + 18.0 + 6.0).sum();
            let (fx, fy) = if end + 28.0 + lead[1] + funs_w <= view.x - side { (end + 28.0, top) } else { (side, top + h + 8.0) };
            lay.funs = button_row(&words, fx + lead[1], fy, h, text, 6.0);
            lay.leads = Some([Vec2::new(side, top + h * 0.5), Vec2::new(fx, fy + h * 0.5)]);
            below_buttons = fy + h + 16.0;
        }
        // the planes, side by side, as large as the room leaves them
        let heading_h = if narrow { 18.0 } else { 22.0 };
        let (ticks_w, ticks_h, gap) = if narrow { (24.0, 15.0, 12.0) } else { (TICKS_W, TICKS_H, 30.0) };
        let planes_top = below_buttons + heading_h;
        // under the planes: their tick numbers, the colour scale, two lines on the cell
        let under = ticks_h + 6.0 + line_h(small) + 2.0 * line_h(small) + 8.0;
        // and under those the sums (five lines), and the chart beside or below them
        let sums_h = 5.0 * line_h(lay.sums_size) + 4.0;
        let chart_least = if narrow { 70.0 } else { 120.0 };
        let block = if narrow { sums_h + 14.0 + 24.0 + chart_least + 26.0 } else { (sums_h).max(26.0 + chart_least + 30.0) + 8.0 };
        let across = ((view.x - 2.0 * side - 2.0 * ticks_w - gap - NAME_W) * 0.5).floor();
        let fit = |below: f32| (view.y - 8.0 - planes_top - under - below).floor();
        let mut p = across.min(fit(block)).min(if narrow { 260.0 } else { 520.0 });
        let mut with_block = true;
        if p < if narrow { 110.0 } else { 160.0 } {
            // too little height for everything: the planes and what is under them alone
            // (the sums are in the side panel)
            p = across.min(fit(0.0)).min(if narrow { 260.0 } else { 520.0 });
            with_block = false;
        }
        if p < 50.0 {
            return lay;
        }
        let width = 2.0 * (ticks_w + p) + gap;
        let x0 = ((view.x - width - NAME_W) * 0.5).floor().max(side * 0.5);
        let left = (Vec2::new(x0 + ticks_w, planes_top), Vec2::new(x0 + ticks_w + p, planes_top + p));
        let right = (Vec2::new(left.1.x + gap + ticks_w, planes_top), Vec2::new(left.1.x + gap + ticks_w + p, planes_top + p));
        lay.planes = Some([left, right]);
        lay.headings = [Vec2::new(x0, below_buttons), Vec2::new(right.0.x - ticks_w, below_buttons)];
        // the colour scale, under the (x, y) plane's numbers, on its right
        let scale_y = planes_top + p + ticks_h + 6.0 + line_h(small) * 0.5;
        let strip_w = (p * 0.42).max(50.0);
        lay.scale = Some((Vec2::new(right.1.x - strip_w - 16.0, scale_y - 4.0), Vec2::new(right.1.x - 16.0, scale_y + 4.0)));
        lay.cell = Some((Vec2::new(x0, planes_top + p + ticks_h + 6.0 + line_h(small)), width));
        if !with_block {
            return lay;
        }
        let block_top = planes_top + p + under + 4.0;
        if narrow {
            lay.sums = Some((Vec2::new(side, block_top), view.x - 2.0 * side));
            let chart_top = block_top + sums_h + 14.0 + 24.0;
            // (no taller than it needs to be: a phone held upright has height to spare)
            let bottom = (view.y - 8.0 - 26.0).min(chart_top + 230.0);
            if bottom - chart_top >= 50.0 {
                lay.chart = Some((Vec2::new(side + 30.0, chart_top), Vec2::new(view.x - side - 4.0, bottom)));
            }
        } else {
            let sums_w = (width * 0.42).max(330.0).min(width);
            lay.sums = Some((Vec2::new(x0, block_top), sums_w));
            let chart_left = x0 + sums_w + 30.0 + 34.0;
            let chart_top = block_top + 26.0;
            let bottom = (view.y - 8.0 - 30.0).min(chart_top + ((x0 + width - chart_left) * 0.55).max(140.0).min(320.0));
            if x0 + width - chart_left >= 140.0 && bottom - chart_top >= 60.0 {
                lay.chart = Some((Vec2::new(chart_left, chart_top), Vec2::new(x0 + width, bottom)));
            }
        }
        lay
    }
}

fn inside_rect(p: Vec2, (min, max): Rect) -> bool {
    p.cmpge(min).all() && p.cmple(max).all()
}

// ----- the scene ------------------------------------------------------------------------

/// The numbers for the setting asked for, worked out when it changes.
#[derive(Clone, Debug)]
struct Numbers {
    /// The map, the function and the radius they are for.
    key: (Map, Fun, u64),
    /// The three sums with 1 to `N_MAX` cells along a side.
    series: Vec<Sums>,
    exact: f64,
    closed: Option<&'static str>,
    /// What the sum without the factor tends to.
    without: f64,
}

impl Numbers {
    fn of(map: Map, fun: Fun, r: f64, areas: &Areas) -> Numbers {
        let (exact, closed) = exact(map, fun, r);
        Numbers { key: (map, fun, r.to_bits()), series: (1..=N_MAX).map(|n| sums(map, fun, r, n, areas)).collect(), exact, closed, without: unfactored(map, fun, r) }
    }

    fn at(&self, n: usize) -> Sums {
        self.series.get(n.max(1) - 1).copied().unwrap_or_default()
    }

    /// The top of the chart: a round number a little above everything on it.
    fn top(&self) -> f64 {
        let most = self.series.iter().flat_map(|s| [s.true_areas, s.factor, s.plain]).chain([self.exact, self.without]).filter(|v| v.is_finite()).fold(0.0f64, f64::max);
        let want = (most * 1.08).max(1e-6);
        let power = 10f64.powf(want.log10().floor());
        let lead = [1.0, 1.2, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0].into_iter().find(|l| l * power >= want * (1.0 - 1e-9)).unwrap_or(10.0);
        lead * power
    }
}

pub struct ChangeVar {
    map: Map,
    fun: Fun,
    n: usize,
    r: f64,
    /// The point that picks a cell, by the shares of the way across the rectangle.
    pick: DVec2,
    /// The map the picture is gliding from, and how far it has got (1: there).
    from: Map,
    blend: f32,
    /// The radius as drawn, following the one asked for.
    radius: plot::Glide,
    /// After a change in the number of cells: the picture as it was, fading out while
    /// the new one fades in (`fade` from 0 to 1).
    old: Option<Look>,
    fade: f32,
    /// The picked cell as drawn, [s0, s1, t0, t1] in shares, and the places of the two
    /// white dots, in the (u, v) plane and in the (x, y) plane: they glide to where they
    /// are going. (The dot in the hand is where the hand is; the other follows. Where a
    /// small move in one plane is a large one in the other, near r = 0 or across the
    /// seam t = 0, the other glides there.)
    cell: plot::Glide,
    handles: [plot::Glide; 2],
    /// The chart as drawn: the three sums for 1 to `N_MAX` cells a side, the exact
    /// value, the limit without the factor, the chart's top and where the mark for n is.
    chart: plot::Glide,
    moving: bool,
    /// The cells' areas, map by map, worked out when a map is first drawn.
    areas: [Option<Areas>; 4],
    numbers: Numbers,
}

impl ChangeVar {
    fn new() -> ChangeVar {
        let t = &TRIALS[0];
        let areas = Areas::of(t.map);
        let numbers = Numbers::of(t.map, t.fun, t.r, &areas);
        let mut scene = ChangeVar {
            map: t.map,
            fun: t.fun,
            n: t.n,
            r: t.r,
            pick: DVec2::from(t.pick),
            from: t.map,
            blend: 1.0,
            radius: plot::Glide::default(),
            old: None,
            fade: 1.0,
            cell: plot::Glide::default(),
            handles: [plot::Glide::default(), plot::Glide::default()],
            chart: plot::Glide::default(),
            moving: false,
            areas: [None, None, None, None],
            numbers,
        };
        scene.areas[t.map.index()] = Some(areas);
        scene.radius.snap(&[scene.r as f32]);
        scene.cell.snap(&scene.cell_target());
        for h in &mut scene.handles {
            h.snap(&[scene.pick.x as f32, scene.pick.y as f32]);
        }
        scene.chart.snap(&scene.chart_target());
        scene
    }

    /// Set up experiment `k` of `TRIALS`.
    fn set_up(&mut self, k: usize) {
        let t = &TRIALS[k.min(TRIALS.len() - 1)];
        self.set_map(t.map);
        self.fun = t.fun;
        self.set_n(t.n);
        self.set_r(t.r);
        self.pick = DVec2::from(t.pick);
    }

    /// Another map: glided to from the one drawn, or faded to if the picture is in the
    /// middle of a change already.
    fn set_map(&mut self, m: Map) {
        if m == self.map {
            return;
        }
        if self.old.is_none() && self.blend >= 1.0 {
            (self.from, self.blend) = (self.map, 0.0);
            self.map = m;
        } else {
            self.crossfade();
            self.map = m;
            self.from = m;
        }
    }

    /// Another number of cells: the new grid fades in over the old one.
    fn set_n(&mut self, n: usize) {
        let n = n.max(1).min(N_MAX);
        if n == self.n {
            return;
        }
        self.crossfade();
        self.n = n;
    }

    fn set_r(&mut self, r: f64) {
        self.r = if r.is_finite() { r.max(R_LEAST).min(R_MOST) } else { R_MOST };
    }

    /// Start a fade from the picture as it is drawn now to the one asked for.
    fn crossfade(&mut self) {
        self.old = Some(self.look());
        self.fade = 0.0;
        (self.from, self.blend) = (self.map, 1.0);
    }

    /// The picture as it is drawn now.
    fn look(&self) -> Look {
        Look { from: self.from, to: self.map, blend: self.blend, r: self.radius.get(0) as f64, n: self.n }
    }

    fn areas_of(&self, m: Map) -> Option<&Areas> {
        self.areas[m.index()].as_ref()
    }

    /// The picked cell as it is to be drawn.
    fn cell_target(&self) -> [f32; 4] {
        let (i, j) = picked(self.pick, self.n);
        let n = self.n as f32;
        [i as f32 / n, (i + 1) as f32 / n, j as f32 / n, (j + 1) as f32 / n]
    }

    /// The chart as it is to be drawn.
    fn chart_target(&self) -> Vec<f32> {
        let s = &self.numbers.series;
        let mut out: Vec<f32> = Vec::with_capacity(3 * N_MAX + 4);
        out.extend(s.iter().map(|s| s.true_areas as f32));
        out.extend(s.iter().map(|s| s.factor as f32));
        out.extend(s.iter().map(|s| s.plain as f32));
        out.extend([self.numbers.exact as f32, self.numbers.without as f32, self.numbers.top() as f32, (self.n as f32).log2()]);
        out
    }

    fn picked_numbers(&self) -> Picked {
        picked_numbers(self.map, self.fun, self.r, self.n, self.pick, self.areas_of(self.map))
    }

    /// Move everything on by `dt` seconds of the clock's (a change of setting glides by
    /// the clock, so that it is seen through even while the scene's time stands still).
    fn advance(&mut self, dt: f32) {
        let dt = if dt.is_finite() { dt.max(0.0) } else { 0.0 };
        // the areas of every map drawn, and the numbers for the setting asked for
        let mut wanted = vec![self.map, self.from];
        if let Some(old) = self.old {
            wanted.extend([old.from, old.to]);
        }
        for m in wanted {
            if self.areas[m.index()].is_none() {
                self.areas[m.index()] = Some(Areas::of(m));
            }
        }
        if self.numbers.key != (self.map, self.fun, self.r.to_bits()) {
            if let Some(areas) = self.areas_of(self.map) {
                self.numbers = Numbers::of(self.map, self.fun, self.r, areas);
            }
        }
        let mut moving = false;
        if self.blend < 1.0 {
            self.blend = (self.blend + dt / MORPH).min(1.0);
            if self.blend >= 1.0 || !self.blend.is_finite() {
                (self.blend, self.from) = (1.0, self.map);
            }
            moving = true;
        }
        if self.old.is_some() {
            self.fade += dt / FADE;
            if self.fade >= 1.0 || !self.fade.is_finite() {
                (self.fade, self.old) = (1.0, None);
            }
            moving = true;
        }
        moving |= self.radius.follow(&[self.r as f32], dt, false);
        let cell = self.cell_target();
        moving |= self.cell.follow(&cell, dt, false);
        for h in &mut self.handles {
            moving |= h.follow(&[self.pick.x as f32, self.pick.y as f32], dt, false);
        }
        let chart = self.chart_target();
        moving |= self.chart.follow(&chart, dt, false);
        self.moving = moving;
    }

    fn at_rest(&self) -> bool {
        !self.moving && self.blend >= 1.0 && self.old.is_none()
    }

    /// The shares across the rectangle of a point of the screen in the (u, v) plane,
    /// and of one in the (x, y) plane (the point T takes there).
    fn share_uv(&self, planes: &[Rect; 2], at: Vec2) -> DVec2 {
        share(self.map, self.r, from_screen(self.map.uv_box(), planes[0], at))
    }

    fn share_xy(&self, planes: &[Rect; 2], at: Vec2) -> DVec2 {
        share(self.map, self.r, self.map.back(from_screen(self.map.xy_box(), planes[1], at)))
    }
}

impl Scene for ChangeVar {
    fn view(&self) -> View {
        View::Flat { center: Vec2::ZERO, half_height: 1.0 }
    }

    fn settings(&self, s: &mut ViewSettings) {
        s.grid = false;
        s.bloom = s.bloom.min(0.2);
    }

    fn animated(&self) -> bool {
        !self.at_rest()
    }

    fn pointer_hint(&self) -> Option<&'static str> {
        Some("white dots: drag to pick a cell, or click a cell    buttons over the picture: click")
    }

    fn ui(&mut self, ui: &mut egui::Ui, _: &Gfx) {
        ui::note(ui, r"An integral over a region is a sum of $f$ times small areas. Describe the region by other coordinates, $(x, y) = T(u, v)$, and a rectangle of $(u, v)$, cut into cells, is carried onto it. A small cell, $\Delta u$ by $\Delta v$, lands as a small patch that is nearly a parallelogram, of area about $|\det J|\,\Delta u\,\Delta v$, where $J$ is the Jacobian of $T$. So");
        ui::equation(ui, r"\iint_{\text{region}} f\,dx\,dy = \iint_{\text{rectangle}} f(T(u, v))\,|\det J|\,du\,dv");
        ui::note(ui, r"On the left the rectangle, on the right the region, each cell tinted by how much $T$ stretches it. Under them, the sum of $f$ over the cells is worked out three ways, and charted as the cells shrink: two ways close in on the exact integral, and the one without the factor, in general, on another number.");

        ui::section(ui, "Things to try");
        for (k, trial) in TRIALS.iter().enumerate() {
            if ui::experiment(ui, trial.label, trial.look_for) {
                self.set_up(k);
            }
        }

        ui::section(ui, "The map");
        let mut m = self.map.index();
        if ui::choice(ui, &MAPS.map(Map::name), &mut m) {
            self.set_map(MAPS[m.min(3)]);
        }
        ui::note(ui, &format!("${}$, and ${}$. So here", self.map.formula(), self.map.factor()));
        ui::equation(ui, &rule(self.map, self.fun));
        let mut k = FUNS.iter().position(|g| *g == self.fun).unwrap_or(0);
        ui::section(ui, "The function");
        if ui::choice(ui, &FUNS.map(Fun::button), &mut k) {
            self.fun = FUNS[k.min(2)];
        }
        ui::note(ui, &format!("$f(x, y) = {}$, taken at the image of each cell's centre.", self.fun.formula()));
        let mut n = self.n;
        if ui::slider_int(ui, "cells along each side, $n$", &mut n, 1..=N_MAX) {
            self.set_n(n);
        }
        if self.map.radial() {
            let mut r = self.r as f32;
            if ui::slider(ui, "radius $R$", &mut r, R_LEAST as f32..=R_MOST as f32) {
                self.set_r(r as f64);
            }
            ui::note(ui, r"The rectangle is $0 \le r \le R$, $0 \le t \le 2\pi$, so the region is the disc (or the ellipse) out to $R$.");
        }

        ui::section(ui, "Reading the picture");
        ui::key(
            ui,
            &[
                (tint(0.8), r"cells: tinted by how much $T$ stretches each, the area of its image over its own area $\Delta u\,\Delta v$ (the scale is under the right-hand plane)"),
                (TRUE_AREAS, "yellow: the picked cell and its image (drag either white dot, or click a cell); on the chart, the sum with the true areas of the images"),
                (pal::PINK, r"pink and green: the picked cell's sides, $\Delta u$ along $u$ and $\Delta v$ along $v$, and where $J$ takes them"),
                (FACTOR, r"cyan, dashed: the parallelogram those span, of area $|\det J|\,\Delta u\,\Delta v$ with $J$ at the cell's centre; on the chart, the sum with the factor"),
                (PLAIN, "red: the sum without the factor; dashed, the number it tends to"),
                (EXACT, "white, dashed: the exact integral"),
            ],
        );

        ui::section(ui, "Words");
        ui::term(ui, "Change of variables:", r"the integral of $f\,dx\,dy$ over a region is that of $f(T(u, v))\,|\det J|\,du\,dv$ over the rectangle $T$ carries onto it, for $f$ continuous and $T$ smooth and one to one inside the rectangle. On its edge $T$ may fail to be, as at $r = 0$, where a whole edge lands on one point: an edge has no area, so the integral loses nothing.");
        ui::term(ui, "Jacobian $J$:", r"the matrix of $T$'s partial derivatives. Its columns, $\partial T/\partial u$ and $\partial T/\partial v$, are where small steps along $u$ and along $v$ go: near a point $p$, $T(p + h) \approx T(p) + J h$ (scene jacobian).");
        ui::term(ui, r"Determinant, $\det J$:", "the factor by which $J$ scales areas, negative where it turns shapes over (scene det).");
        ui::term(ui, "Polar coordinates:", r"a point named by its distance $r$ from $0$ and its angle $t$ from the $x$ axis, in radians: $(x, y) = (r\cos t, r\sin t)$, with $|\det J| = r$.");
        ui::term(ui, "The bell:", r"$e^{-(x^2 + y^2)}$, the normal distribution's shape in two dimensions (scene gauss2).");
        ui::term(ui, "One variable:", r"the same rule on a line. If $x = g(u)$ carries an interval of $u$ one to one onto one of $x$, a step $du$ becomes a step of length $|g'(u)|\,du$, so $\int f(x)\,dx = \int f(g(u))\,|g'(u)|\,du$ over the two intervals.");
        ui::term(ui, "Exact:", "in closed form where the picture writes one; otherwise by a quadrature (Gauss–Legendre's rule, 20 points each way on each of 36 pieces), right to more digits than are shown.");

        ui::section(ui, "Numbers");
        let s = self.numbers.at(self.n);
        let d = decimals(self.numbers.exact.abs().max(s.factor.abs()), 5);
        let c = self.picked_numbers();
        ui::readout(ui, "cells", format!("${n} \\times {n}$", n = self.n));
        ui::readout(ui, &format!("picked cell, ${}$", self.map.cell()), format!("${}$", sig(c.size, 4)));
        ui::readout_colored(ui, "area of its image", format!("${}$", sig(c.area.abs(), 4)), TRUE_AREAS);
        ui::readout_colored(ui, &format!(r"$|\det J|\,{}$", self.map.cell()), format!("${}$", sig(c.det.abs() * c.size, 4)), FACTOR);
        ui::readout(ui, "$f$ at its centre", format!("${}$", sig(c.value, 4)));
        ui::readout_colored(ui, r"$\sum f \cdot \text{area}$", format!("${}$", num(s.true_areas, d)), TRUE_AREAS);
        ui::readout_colored(ui, &format!(r"$\sum f\,|\det J|\,{}$", self.map.cell()), format!("${}$", num(s.factor, d)), FACTOR);
        ui::readout_colored(ui, &format!(r"$\sum f\,{}$", self.map.cell()), format!("${}$", num(s.plain, d)), PLAIN);
        ui::readout_colored(ui, r"exact, $\iint f\,dx\,dy$", format!("${}$", num(self.numbers.exact, d)), EXACT);
        ui::readout_colored(ui, &format!(r"without the factor, in the limit: $\iint f\,{}$", self.map.d()), format!("${}$", num(self.numbers.without, d)), PLAIN);
        ui::see_also(
            ui,
            &[
                ("jacobian", "the Jacobian: a curved map up close"),
                ("det", "the determinant as the factor by which a linear map scales areas"),
                ("ftc", "integrals in one variable: the fundamental theorem"),
                ("gauss2", "the bell in two dimensions: the normal distribution"),
            ],
        );
    }

    fn frame(&mut self, f: &mut Frame) {
        let view = f.view_size();
        // (a canvas squeezed to a sliver is one the controls have opened over, on a phone)
        let covered = view.x < 160.0;
        let lay = Layout::of(view, f.top_inset);
        if !covered {
            // the picture is for its buttons, its handles and its cells: no moving the view
            f.handles.capture();
            f.handles.claim_clicks();
            self.buttons(f, &lay);
            self.hands(f, &lay);
        }
        self.advance(f.real_dt);
        if covered {
            return;
        }
        in_order(f);
        if let Some(planes) = lay.planes {
            if let Some(old) = self.old {
                self.cells(f, &planes, &old, 1.0 - smooth(self.fade), GRID_OLD);
            }
            let alpha = if self.old.is_some() { smooth(self.fade) } else { 1.0 };
            self.cells(f, &planes, &self.look(), alpha, GRID);
            self.plane_axes(f, &lay, &planes);
            self.picked_cell(f, &lay, &planes);
        }
        self.words(f, &lay);
        self.sums_table(f, &lay);
        self.chart(f, &lay);
        keep_batches(f);
    }
}

impl ChangeVar {
    /// The buttons over the picture: the maps and the functions.
    fn buttons(&mut self, f: &mut Frame, lay: &Layout) {
        if let Some([a, b]) = lay.leads {
            f.text_px(a, "map:", lay.text, pal::DIM, Align2::LEFT_CENTER);
            f.text_px(b, "$f$:", lay.text, pal::DIM, Align2::LEFT_CENTER);
        }
        for (k, r) in lay.maps.iter().enumerate() {
            if plot::button(f, *r, MAPS[k].name(), lay.text, MAPS[k] == self.map) {
                self.set_map(MAPS[k]);
            }
        }
        for (k, r) in lay.funs.iter().enumerate() {
            if plot::button(f, *r, FUNS[k].button(), lay.text, FUNS[k] == self.fun) {
                self.fun = FUNS[k];
            }
        }
    }

    /// The two white dots, which pick a cell, and clicks on a plane, which pick the cell
    /// there.
    fn hands(&mut self, f: &mut Frame, lay: &Layout) {
        let Some(planes) = lay.planes else { return };
        let look = self.look();
        for plane in 0..2 {
            let at = DVec2::new(self.handles[plane].get(0) as f64, self.handles[plane].get(1) as f64);
            let dot = look.at(&planes, plane, at);
            if !dot.is_finite() {
                continue;
            }
            if let Some(to) = plot::screen_handle(f, 1 + plane as u32, dot, pal::WHITE) {
                let s = if plane == 0 { self.share_uv(&planes, to) } else { self.share_xy(&planes, to) };
                self.pick = s;
                self.handles[plane].snap(&[s.x as f32, s.y as f32]);
            }
        }
        if f.clicked_canvas() {
            if let Some(p) = plot::pointer(f) {
                if inside_rect(p, planes[0]) {
                    self.pick = self.share_uv(&planes, p);
                } else if inside_rect(p, planes[1]) {
                    self.pick = self.share_xy(&planes, p);
                }
            }
        }
    }

    /// The cells of a look, in both planes: tinted, and outlined, at the strength `alpha`.
    fn cells(&self, f: &mut Frame, planes: &[Rect; 2], look: &Look, alpha: f32, kind: u8) {
        let n = look.n.max(1);
        let tints: Vec<Color> = (0..n * n)
            .map(|c| {
                let (i, j) = (c % n, c / n);
                let s = look.mix_value(|m| (stretch(m, look.r, n, i, j, self.areas_of(m)) / m.most_stretch()) as f32);
                tint(s).a(0.92 * alpha)
            })
            .collect();
        // the (u, v) plane: a quad a cell
        let corner = |a: usize, b: usize, plane: usize| look.at(planes, plane, DVec2::new(a as f64 / n as f64, b as f64 / n as f64));
        let uv: Vec<Vec3> = (0..=n).flat_map(|b| (0..=n).map(move |a| (a, b))).map(|(a, b)| f.hud_at(1, corner(a, b, 0))).collect();
        for j in 0..n {
            for i in 0..n {
                let at = |a: usize, b: usize| uv[b * (n + 1) + a];
                f.dl.tris_alpha.quad(at(i, j), at(i + 1, j), at(i + 1, j + 1), at(i, j + 1), tints[j * n + i]);
            }
        }
        // the (x, y) plane: each cell in pieces fine enough to follow its curved sides
        let s = 24usize.div_ceil(n).max(1);
        let fine = n * s;
        let xy: Vec<Vec3> =
            (0..=fine).flat_map(|b| (0..=fine).map(move |a| (a, b))).map(|(a, b)| f.hud_at(1, look.at(planes, 1, DVec2::new(a as f64 / fine as f64, b as f64 / fine as f64)))).collect();
        for b in 0..fine {
            for a in 0..fine {
                let at = |a: usize, b: usize| xy[b * (fine + 1) + a];
                let color = tints[(b / s) * n + a / s];
                f.dl.tris_alpha.quad(at(a, b), at(a + 1, b), at(a + 1, b + 1), at(a, b + 1), color);
            }
        }
        // the outlines: straight in the (u, v) plane, the images of those lines in the
        // (x, y) plane, the edges of the rectangle and of the region brighter. (A radial
        // map's region is edged by r = R alone: t = 0 and t = 2π land on one seam, and
        // r = 0 on one point.)
        let m = points_a_side(n);
        let steps = n * m;
        let line = Color::hex(0x0A0E1C).a(0.62 * alpha);
        let edge = pal::TEXT.a(0.55 * alpha);
        let mut lines: Vec<(Vec<Vec3>, f32, Color)> = Vec::with_capacity(4 * (n + 1));
        for k in 0..=n {
            let share = k as f64 / n as f64;
            for vertical in [true, false] {
                let rim = k == 0 || k == n;
                let style = |rim: bool| if rim { (1.4, edge) } else { (1.0, line) };
                let (w, c) = style(rim);
                let s = |t: f64| if vertical { DVec2::new(share, t) } else { DVec2::new(t, share) };
                lines.push((vec![f.hud(look.at(planes, 0, s(0.0))), f.hud(look.at(planes, 0, s(1.0)))], w, c));
                let (w, c) = style(rim && (!look.to.radial() || (vertical && k == n)));
                lines.push(((0..=steps).map(|q| f.hud(look.at(planes, 1, s(q as f64 / steps as f64)))).collect(), w, c));
            }
        }
        let batch = plot::apart(f, kind, false);
        for (pts, w, c) in lines {
            batch.polyline(pts, w, c);
        }
    }

    /// The frames of the planes, their ticks and numbers and the names of their axes:
    /// for the map drawn (and, while the picture glides from one map to another, the
    /// other one's fading out).
    fn plane_axes(&self, f: &mut Frame, lay: &Layout, planes: &[Rect; 2]) {
        for (min, max) in planes {
            let c = [*min, Vec2::new(max.x, min.y), *max, Vec2::new(min.x, max.y), *min];
            plot::path_of(f, plot::AXES, &c, 1.0, pal::DIM.a(0.35));
        }
        let look = self.look();
        let b = look.eased();
        // (while the picture glides from one map to another, the old numbers fade out in
        // the first half of the way and the new ones in over the second, never the two
        // over each other)
        let maps: Vec<(Map, f32)> =
            if look.from == look.to || b >= 1.0 { vec![(look.to, 1.0)] } else { vec![(look.from, (1.0 - 2.0 * b).max(0.0)), (look.to, (2.0 * b - 1.0).max(0.0))] };
        let size = lay.small - 1.0;
        for (m, alpha) in maps {
            for (plane, (min, max)) in planes.iter().enumerate() {
                let bx = if plane == 0 { m.uv_box() } else { m.xy_box() };
                let names = if plane == 0 { m.vars() } else { ["x", "y"] };
                // across: round numbers (in the (u, v) plane of a radial map, r's)
                let across = plot::Ticks::of(bx[0], bx[1], max.x - min.x);
                for v in across.values() {
                    let x = to_screen(bx, (*min, *max), DVec2::new(v, bx[2])).x;
                    plot::line_of(f, TICKS, Vec2::new(x, max.y), Vec2::new(x, max.y + 4.0), 1.0, pal::DIM.a(0.8 * alpha));
                    f.text_px(Vec2::new(x, max.y + 5.0), across.name(v), size, pal::DIM.a(alpha), Align2::CENTER_TOP);
                }
                // up: round numbers, or t's quarter turns
                let up: Vec<(f64, String)> = if plane == 0 && m.radial() {
                    let quarters: &[(f64, &str)] =
                        if lay.narrow { &[(0.0, "0"), (PI64, r"\pi"), (TAU64, r"2\pi")] } else { &[(0.0, "0"), (0.5 * PI64, r"\pi/2"), (PI64, r"\pi"), (1.5 * PI64, r"3\pi/2"), (TAU64, r"2\pi")] };
                    quarters.iter().map(|(v, t)| (*v, format!("${t}$"))).collect()
                } else {
                    let t = plot::Ticks::of(bx[2], bx[3], max.y - min.y);
                    t.values().map(|v| (v, t.name(v))).collect()
                };
                for (v, text) in up {
                    let y = to_screen(bx, (*min, *max), DVec2::new(bx[0], v)).y;
                    // (none at the very top, where the axis's name is)
                    if y - min.y < 20.0 {
                        continue;
                    }
                    plot::line_of(f, TICKS, Vec2::new(min.x - 4.0, y), Vec2::new(min.x, y), 1.0, pal::DIM.a(0.8 * alpha));
                    f.text_px(Vec2::new(min.x - 6.0, y), text, size, pal::DIM.a(alpha), Align2::RIGHT_CENTER);
                }
                f.text_px(Vec2::new(max.x + 5.0, max.y), format!("${}$", names[0]), lay.small, pal::TEXT.a(0.85 * alpha), Align2::LEFT_CENTER);
                f.text_px(Vec2::new(min.x - 6.0, min.y + 1.0), format!("${}$", names[1]), lay.small, pal::TEXT.a(0.85 * alpha), Align2::RIGHT_TOP);
                // the (x, y) plane's axes through 0, faint
                if plane == 1 {
                    let zero = to_screen(bx, (*min, *max), DVec2::ZERO);
                    if zero.y > min.y && zero.y < max.y {
                        plot::line_of(f, TICKS, Vec2::new(min.x, zero.y), Vec2::new(max.x, zero.y), 1.0, pal::DIM.a(0.25 * alpha));
                    }
                    if zero.x > min.x && zero.x < max.x {
                        plot::line_of(f, TICKS, Vec2::new(zero.x, min.y), Vec2::new(zero.x, max.y), 1.0, pal::DIM.a(0.25 * alpha));
                    }
                }
            }
        }
    }

    /// The picked cell in both planes: its outline and its image's (yellow), its sides
    /// and where J takes them (pink and green), and the parallelogram those span (cyan).
    fn picked_cell(&self, f: &mut Frame, lay: &Layout, planes: &[Rect; 2]) {
        let look = self.look();
        let c = [0, 1, 2, 3].map(|k| self.cell.get(k) as f64);
        let mut uv = Vec::with_capacity(96);
        let mut xy = Vec::with_capacity(96);
        outline(c, 24, |s| {
            uv.push(look.at(planes, 0, s));
            xy.push(look.at(planes, 1, s));
        });
        // J at the cell's centre, applied to its sides, from the image of the centre
        let centre = DVec2::new(0.5 * (c[0] + c[1]), 0.5 * (c[2] + c[3]));
        let corner = |m: Map, a: f64, b: f64| {
            let [u0, u1, v0, v1] = m.rect(look.r);
            let step = DVec2::new((u1 - u0) * (c[1] - c[0]) * a, (v1 - v0) * (c[3] - c[2]) * b);
            let p = point(m, look.r, centre);
            to_screen(m.xy_box(), planes[1], m.to(p) + m.jacobian(p) * step)
        };
        let para = [(-0.5, -0.5), (0.5, -0.5), (0.5, 0.5), (-0.5, 0.5)].map(|(a, b)| look.mix(|m| corner(m, a, b)));
        // a light wash over the picked cell, under its outline
        for pts in [&uv, &xy] {
            let middle = pts.iter().fold(Vec2::ZERO, |s, p| s + *p) / pts.len().max(1) as f32;
            if !middle.is_finite() {
                continue;
            }
            let m = f.hud_at(2, middle);
            for k in 0..pts.len() {
                let (a, b) = (f.hud_at(2, pts[k]), f.hud_at(2, pts[(k + 1) % pts.len()]));
                f.dl.tris_alpha.tri(m, a, b, TRUE_AREAS.a(0.1));
            }
        }
        // (the outline's 96 points: the bottom from 0 to 24, the left edge from 72 down
        // to the corner again)
        let arrows = [(uv[0], uv[24], pal::PINK), (uv[0], uv[72], pal::GREEN), (para[0], para[1], pal::PINK), (para[0], para[3], pal::GREEN)];
        let to_hud = |f: &Frame, p: &[Vec2]| -> Vec<Vec3> { p.iter().map(|q| f.hud(*q)).collect() };
        let (uv3, xy3) = (to_hud(f, &uv), to_hud(f, &xy));
        let para3 = to_hud(f, &para);
        let heads: Vec<(Vec3, Vec3, Vec3, Vec3, Color)> = arrows
            .iter()
            .map(|(a, b, col)| {
                let d = (*b - *a).normalize_or_zero();
                let len = (*b - *a).length();
                let head = (len * 0.35).min(8.0);
                let side = Vec2::new(-d.y, d.x) * head * 0.5;
                (f.hud(*a), f.hud(*b), f.hud(*b - d * head + side), f.hud(*b - d * head - side), *col)
            })
            .collect();
        let batch = plot::apart(f, PICK, false);
        batch.closed(&uv3, 2.2, TRUE_AREAS);
        batch.closed(&xy3, 2.2, TRUE_AREAS);
        for (a, b, l, r, col) in heads {
            batch.segment(a, b, 2.2, col);
            batch.segment(b, l, 2.2, col);
            batch.segment(b, r, 2.2, col);
        }
        let dashed = plot::apart(f, DASHED, true);
        for k in 0..4 {
            dashed.segment(para3[k], para3[(k + 1) % 4], 1.6, FACTOR);
        }
        // the sides' names, beside the picked cell's in the (u, v) plane (where the planes
        // are large)
        if planes[0].1.x - planes[0].0.x >= 260.0 {
            let [du, dv] = if self.map.radial() { [r"$\Delta r$", r"$\Delta t$"] } else { [r"$\Delta u$", r"$\Delta v$"] };
            let (bottom, left) = ((uv[0] + uv[24]) * 0.5, (uv[0] + uv[72]) * 0.5);
            if bottom.is_finite() && left.is_finite() {
                f.caption_px(bottom + Vec2::new(0.0, 6.0), du, lay.small, pal::PINK, Align2::CENTER_TOP);
                f.caption_px(left - Vec2::new(6.0, 0.0), dv, lay.small, pal::GREEN, Align2::RIGHT_CENTER);
            }
        }
    }

    /// The words on the picture: the planes' headings, the colour scale and the lines
    /// about the picked cell.
    fn words(&self, f: &mut Frame, lay: &Layout) {
        let Some(planes) = lay.planes else { return };
        let (map, size) = (self.map, lay.text);
        let [u, v] = map.vars();
        let n = self.n;
        let left = if lay.narrow { format!("$({u}, {v})$: ${n} \\times {n}$ cells") } else { format!("the rectangle of $({u}, {v})$, cut into ${n} \\times {n}$ cells") };
        let right = if lay.narrow { format!("${}$", map.formula()) } else { format!("where $T$ puts them: ${}$", map.formula()) };
        for (k, text) in [left, right].into_iter().enumerate() {
            // (a heading too long for its plane is drawn smaller, to fit)
            let room = planes[k].1.x - lay.headings[k].x + 8.0;
            let w = f.text_size(&text, size).x;
            let s = if w > room && w > 0.0 { (size * room / w).max(8.0) } else { size };
            f.text_px(lay.headings[k], text, s, pal::TEXT, Align2::LEFT_TOP);
        }
        // the colour scale
        if let Some((min, max)) = lay.scale {
            let strips = 30;
            for k in 0..strips {
                let (a, b) = (k as f32 / strips as f32, (k + 1) as f32 / strips as f32);
                let x = |t: f32| min.x + (max.x - min.x) * t;
                hud_rect(f, Vec2::new(x(a), min.y), Vec2::new(x(b) + 0.5, max.y), tint((a + b) * 0.5));
            }
            let mid = (min.y + max.y) * 0.5;
            let small = lay.small - 0.5;
            f.text_px(Vec2::new(min.x - 5.0, mid), "0", small, pal::DIM, Align2::RIGHT_CENTER);
            f.text_px(Vec2::new(max.x + 5.0, mid), num(map.most_stretch(), 0), small, pal::DIM, Align2::LEFT_CENTER);
            let name = if lay.narrow { "stretch:".to_string() } else { format!(r"stretch, area $\div\ {}$:", map.cell()) };
            f.text_px(Vec2::new(min.x - 22.0, mid), name, small, pal::DIM, Align2::RIGHT_CENTER);
        }
        // the picked cell
        if let Some((at, width)) = lay.cell {
            let c = self.picked_numbers();
            let small = lay.small;
            let cell = map.cell();
            let (size, area, det, product) = (trimmed(c.size, 5), sig(c.area.abs(), 4), trimmed(c.det.abs(), 5), sig(c.det.abs() * c.size, 4));
            let first = if lay.narrow {
                format!(r"picked: ${cell} = {size}$, its image ${area}$")
            } else {
                format!(r"yellow: the picked cell, ${cell} = {size}$, and its image, of area ${area}$")
            };
            let over = if c.det < 0.0 { r", turned over: $\det J < 0$" } else { "" };
            let second = if lay.narrow {
                format!(r"dashed: $|\det J|\,{cell} = {det} \times {size} = {product}${over}")
            } else {
                format!(r"dashed: $J$ applied to its sides, of area $|\det J|\,{cell} = {det} \times {size} = {product}${over}")
            };
            for (k, (text, color)) in [(first, TRUE_AREAS), (second, FACTOR)].into_iter().enumerate() {
                let w = f.text_size(&text, small).x;
                let s = if w > width && w > 0.0 { (small * width / w).max(8.0) } else { small };
                f.text_px(at + Vec2::new(0.0, k as f32 * line_h(small)), text, s, color, Align2::LEFT_TOP);
            }
        }
    }

    /// The three sums at the number of cells asked for, and the exact value.
    fn sums_table(&self, f: &mut Frame, lay: &Layout) {
        let Some((at, width)) = lay.sums else { return };
        let small = lay.sums_size;
        let s = self.numbers.at(self.n);
        let d = decimals(self.numbers.exact.abs().max(s.factor.abs()), 5);
        let cell = self.map.cell();
        let n = self.n;
        let radius = if self.map.radial() { format!(" ($R = {}$)", trimmed(self.r, 3)) } else { String::new() };
        let heading = format!(r"$f = {}$, summed over the ${n} \times {n}$ cells{radius}:", self.fun.formula());
        let exact = match self.numbers.closed {
            Some(form) => format!(r"exact: $\iint f\,dx\,dy = {form}$"),
            None => r"exact: $\iint f\,dx\,dy$".to_string(),
        };
        let rows = if lay.narrow {
            [
                (r"$\sum f \cdot \text{area of its image}$".to_string(), s.true_areas, TRUE_AREAS),
                (format!(r"$\sum f\,|\det J|\,{cell}$"), s.factor, FACTOR),
                (format!(r"$\sum f\,{cell}$, no factor"), s.plain, PLAIN),
                (exact, self.numbers.exact, EXACT),
            ]
        } else {
            [
                (r"$\sum f \cdot (\text{area of the cell's image})$".to_string(), s.true_areas, TRUE_AREAS),
                (format!(r"$\sum f \cdot |\det J|\,{cell}$"), s.factor, FACTOR),
                (format!(r"$\sum f \cdot {cell}$, the factor left out"), s.plain, PLAIN),
                (exact, self.numbers.exact, EXACT),
            ]
        };
        f.text_px(at, heading, small, pal::TEXT, Align2::LEFT_TOP);
        let h = line_h(small);
        for (k, (text, value, color)) in rows.into_iter().enumerate() {
            let y = at.y + (k + 1) as f32 * h + 4.0;
            hud_rect(f, Vec2::new(at.x, y + 2.0), Vec2::new(at.x + 9.0, y + small - 1.0), color);
            f.text_px(Vec2::new(at.x + 15.0, y), text, small, pal::TEXT.a(0.9), Align2::LEFT_TOP);
            f.text_px(Vec2::new(at.x + width, y), format!("${}$", num(value, d)), small, color, Align2::RIGHT_TOP);
        }
        // the rule, for this map and this function (where there is room under the sums)
        let y = at.y + 5.0 * h + 18.0;
        if !lay.narrow && y + 60.0 <= f.view_size().y {
            f.text_px(Vec2::new(at.x, y), "the integral, in the two sets of coordinates:", small - 1.0, pal::DIM, Align2::LEFT_TOP);
            let formula = format!(r"$\displaystyle {}$", rule(self.map, self.fun));
            let size = small + 1.0;
            let w = f.text_size(&formula, size).x;
            let size = if w > width && w > 0.0 { (size * width / w).max(8.0) } else { size };
            f.text_px(Vec2::new(at.x, y + 22.0), formula, size, pal::TEXT, Align2::LEFT_TOP);
        }
    }

    /// The chart of the three sums against the number of cells along a side.
    fn chart(&self, f: &mut Frame, lay: &Layout) {
        let Some(rect) = lay.chart else { return };
        let get = |k: usize| self.chart.get(k) as f64;
        let top = get(3 * N_MAX + 2).max(1e-6);
        let p = Plot::new(rect, [0.0, (N_MAX as f64).log2()], [0.0, top]);
        if p.tiny() {
            return;
        }
        let (min, max) = rect;
        let small = lay.small - 0.5;
        f.text_px(Vec2::new(min.x - 30.0, min.y - 22.0), "the three sums as the cells shrink", small + 0.5, pal::TEXT, Align2::LEFT_TOP);
        // the values up the side: those of the top the chart is going to, so that while
        // its top glides they glide with it, as many all the way (a mark above the top
        // as it is now waits at the top, unnamed)
        let ticks = plot::Ticks::of(0.0, self.numbers.top(), p.height() * 1.6);
        plot::line_of(f, plot::AXES, min, Vec2::new(min.x, max.y), 1.0, pal::DIM.a(0.8));
        for v in ticks.values() {
            let y = p.px_y(v).max(min.y).min(max.y);
            plot::line_of(f, CHART_TICKS, Vec2::new(min.x - 4.0, y), Vec2::new(max.x, y), 1.0, pal::DIM.a(0.14));
            if v <= top * (1.0 + 1e-6) {
                f.text_px(Vec2::new(min.x - 6.0, y), ticks.name(v), small - 0.5, pal::DIM, Align2::RIGHT_CENTER);
            }
        }
        plot::line_of(f, plot::AXES, Vec2::new(min.x, max.y), max, 1.0, pal::DIM.a(0.8));
        for e in 0..=5 {
            let x = p.px_x(e as f64);
            plot::line_of(f, plot::AXES, Vec2::new(x, max.y), Vec2::new(x, max.y + 4.0), 1.0, pal::DIM.a(0.8));
            f.text_px(Vec2::new(x, max.y + 5.0), format!("{}", 1 << e), small - 0.5, pal::DIM, Align2::CENTER_TOP);
        }
        f.text_px(Vec2::new(max.x, max.y + 7.0 + small), "cells along each side, $n$", small - 0.5, pal::DIM, Align2::RIGHT_TOP);
        // the exact value and what the sum without the factor tends to, dashed
        let (exact, without) = (get(3 * N_MAX), get(3 * N_MAX + 1));
        for (v, color) in [(exact, EXACT), (without, PLAIN)] {
            let y = p.px_y(v).max(min.y).min(max.y);
            plot::dashed_of(f, DASHED, Vec2::new(min.x, y), Vec2::new(max.x, y), 1.4, color.a(0.8));
        }
        let (ye, yw) = (p.px_y(exact), p.px_y(without));
        // (named at the right end, the two names kept apart where the lines are close)
        let apart = (ye - yw).abs() < 16.0;
        let names = [(ye, "exact", EXACT, !apart || ye <= yw), (yw, "without the factor", PLAIN, !apart || yw < ye)];
        for (y, name, color, above) in names {
            let (off, align) = if above { (-3.0, Align2::RIGHT_BOTTOM) } else { (3.0, Align2::RIGHT_TOP) };
            f.text_px(Vec2::new(max.x - 4.0, y.max(min.y + 12.0).min(max.y - 2.0) + off), name, small - 0.5, color, align);
        }
        // the three sums, n = 1 to N_MAX: cyan wide under yellow narrow, so that where
        // the two agree both are seen
        // (kept inside the chart while its top glides to a new one, so that no line is
        // cut at its edge: a cut adds points, and a batch whose points come and go
        // cannot be followed from frame to frame)
        let x_of = |k: usize| ((k + 1) as f64).log2();
        let inside = |v: f64| if v.is_finite() { v.max(0.0).min(top) } else { 0.0 };
        for (series, width, color) in [(2usize, 2.0, PLAIN), (1, 3.6, FACTOR), (0, 1.6, TRUE_AREAS)] {
            let pts: Vec<(f64, f64)> = (0..N_MAX).map(|k| (x_of(k), inside(get(series * N_MAX + k)))).collect();
            p.path(f, plot::CURVES, pts, width, color);
        }
        // the number of cells asked for, and the sums there
        let xn = get(3 * N_MAX + 3).max(0.0).min((N_MAX as f64).log2());
        plot::line_of(f, plot::MARKS, Vec2::new(p.px_x(xn), min.y), Vec2::new(p.px_x(xn), max.y), 1.0, pal::DIM.a(0.6));
        let along = |series: usize| {
            // (between the two n either side of the mark's place)
            let k = ((2f64.powf(xn) - 1.0).floor().max(0.0) as usize).min(N_MAX - 1);
            let k1 = (k + 1).min(N_MAX - 1);
            let (x0, x1) = (x_of(k), x_of(k1));
            let t = if x1 > x0 { ((xn - x0) / (x1 - x0)).max(0.0).min(1.0) } else { 0.0 };
            get(series * N_MAX + k) + (get(series * N_MAX + k1) - get(series * N_MAX + k)) * t
        };
        for (series, color) in [(2usize, PLAIN), (1, FACTOR), (0, TRUE_AREAS)] {
            p.dot(f, xn, inside(along(series)), if lay.narrow { 6.0 } else { 7.5 }, color);
        }
        f.text_px(Vec2::new(p.px_x(xn) + 4.0, min.y + 2.0), format!("$n = {}$", self.n), small - 0.5, pal::DIM, Align2::LEFT_TOP);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Relative difference, with a floor for numbers near 0.
    fn rel(a: f64, b: f64) -> f64 {
        (a - b).abs() / b.abs().max(1e-12)
    }

    /// The integral of g over [a, b] by Gauss–Legendre on `panels` pieces.
    fn line(a: f64, b: f64, panels: usize, g: impl Fn(f64) -> f64) -> f64 {
        let h = (b - a) / panels as f64;
        (0..panels).map(|k| legendre().iter().map(|(x, w)| w * g(a + h * (k as f64 + 0.5 + 0.5 * x))).sum::<f64>() * 0.5 * h).sum()
    }

    /// Points spread over a map's rectangle, the edges and corners among them.
    fn points(map: Map, r: f64) -> Vec<DVec2> {
        let mut rng = Rng::new(11);
        let mut out: Vec<DVec2> = (0..200).map(|_| point(map, r, DVec2::new(rng.f32() as f64, rng.f32() as f64))).collect();
        out.extend([0.0, 0.5, 1.0].iter().flat_map(|a| [0.0, 0.5, 1.0].map(|b| point(map, r, DVec2::new(*a, b)))));
        out
    }

    /// Gauss–Legendre's rule integrates every polynomial of degree up to 39 exactly.
    #[test]
    fn the_quadrature_is_exact_for_polynomials() {
        let w: f64 = legendre().iter().map(|n| n.1).sum();
        assert!((w - 2.0).abs() < 1e-14, "{w}");
        for d in [0, 1, 2, 5, 12, 39] {
            let got = integrate([0.0, 1.0, 0.0, 2.0], 1, |p| p.x.powi(d) * p.y);
            assert!((got - 2.0 / (d as f64 + 1.0)).abs() < 1e-13, "degree {d}: {got}");
        }
    }

    /// J is T's derivative: each column against a central difference of T, and its
    /// determinant is the one each map's formula says; `back` undoes T inside the
    /// rectangle; and near a point T is J to first order (the error falls as h²).
    #[test]
    fn the_jacobian_is_the_maps_derivative() {
        for map in MAPS {
            for p in points(map, 2.3) {
                let j = map.jacobian(p);
                let h = 1e-6;
                for (k, e) in [DVec2::X, DVec2::Y].into_iter().enumerate() {
                    let diff = (map.to(p + e * h) - map.to(p - e * h)) / (2.0 * h);
                    assert!((diff - j.col(k)).length() < 1e-6 * (1.0 + j.col(k).length()), "{map:?} at {p}: column {k} {:?} against {diff}", j.col(k));
                }
                let want = match map {
                    Map::Polar => p.x,
                    Map::Ellipse => 2.0 * p.x,
                    Map::Linear => -2.0,
                    Map::Squaring => 4.0 * p.length_squared(),
                };
                assert!((j.determinant() - want).abs() < 1e-12 * (1.0 + want.abs()), "{map:?} at {p}");
                // (inside the rectangle, away from r = 0 and from the seam t = 0)
                let s = share(map, 2.3, p);
                if s.min_element() > 1e-3 && s.max_element() < 1.0 - 1e-3 {
                    assert!((map.back(map.to(p)) - p).length() < 1e-9, "{map:?} at {p}: {}", map.back(map.to(p)));
                }
                // T(p + h) − T(p) − J h is of the size of h²
                let d = DVec2::new(0.3, -0.7);
                let miss = |h: f64| (map.to(p + d * h) - map.to(p) - j * (d * h)).length();
                if miss(1e-2) > 1e-12 {
                    assert!((miss(1e-2) / miss(5e-3) - 4.0).abs() < 0.2, "{map:?} at {p}: {} then {}", miss(1e-2), miss(5e-3));
                }
            }
        }
    }

    /// The true areas come from the cells' outlines alone (no Jacobian in them), and
    /// are right to far more than the digits shown: against the area of a piece of
    /// ring, (r₂² − r₁²) Δt / 2 (stretched by 2 for the ellipse), against det A Δu Δv
    /// (negative: the linear map turns cells over), and for squaring against the
    /// boundary integral ½∮(x dy − y dx) worked out by hand; and against a plain
    /// polygon of 4000 points a side.
    #[test]
    fn the_true_areas_are_right() {
        for map in MAPS {
            let areas = Areas::of(map);
            for n in [1, 2, 3, 5, 8, 13, 32] {
                for r in [R_LEAST, 1.7, R_MOST] {
                    let size = cell_size(map, r, n);
                    for j in 0..n {
                        for i in 0..n {
                            let [u0, u1, v0, v1] = cell_of(map.rect(r), n, i, j);
                            let want = match map {
                                Map::Polar => (u1 * u1 - u0 * u0) * (v1 - v0) / 2.0,
                                Map::Ellipse => 2.0 * (u1 * u1 - u0 * u0) * (v1 - v0) / 2.0,
                                Map::Linear => -2.0 * size,
                                Map::Squaring => 4.0 * ((u1.powi(3) - u0.powi(3)) / 3.0 * (v1 - v0) + (u1 - u0) * (v1.powi(3) - v0.powi(3)) / 3.0),
                            };
                            let got = areas.get(map, r, n, i, j);
                            assert!(rel(got, want) < 1e-6, "{map:?} n {n} R {r} cell ({i}, {j}): {got} against {want}");
                        }
                    }
                }
            }
            // a plain polygon, very fine, agrees with the polygon refined
            for c in [[0.0, 1.0, 0.0, 1.0], [0.2, 0.9, 0.1, 0.6], [0.0, 0.25, 3.0, 4.5]] {
                let mut pts = Vec::new();
                outline(c, 4000, |p| pts.push(map.to(p)));
                let plain = shoelace(&pts);
                assert!(rel(cell_area(map, c, 48), plain) < 1e-6, "{map:?} {c:?}: {} against {plain}", cell_area(map, c, 48));
            }
        }
    }

    /// The closed forms the picture quotes are right (against the quadrature of
    /// f(T) |det J| over the rectangle), and the integrals without a closed form are
    /// right too, worked out in the (x, y) plane alone, with no change of variables in
    /// two dimensions; what the sums without the factor tend to is the integral of
    /// f(T) over the rectangle (against values worked out by hand).
    #[test]
    fn the_exact_values_are_right() {
        for map in MAPS {
            for fun in FUNS {
                for r in [R_LEAST, 1.5, R_MOST] {
                    let (e, closed) = exact(map, fun, r);
                    let q = integrate(map.rect(r), PANELS, |p| fun.at(map.to(p)) * map.jacobian(p).determinant().abs());
                    assert!(rel(e, q) < 1e-10, "{map:?} {fun:?} R {r}: {e} against {q}");
                    if closed.is_none() {
                        // in the (x, y) plane: for each x (or y) the region's other
                        // coordinate runs between two edges
                        let bell = |x: f64, y: f64| fun.at(DVec2::new(x, y));
                        let direct = match map {
                            Map::Linear => {
                                // the parallelogram (0, 0), (1.5, 0.5), (2.5, −0.5), (1, −1):
                                // between its lower edges and its upper ones
                                let low = |x: f64| if x <= 1.0 { -x } else { -1.0 + (x - 1.0) * 0.5 / 1.5 };
                                let high = |x: f64| if x <= 1.5 { x / 3.0 } else { 0.5 - (x - 1.5) };
                                [(0.0, 1.0), (1.0, 1.5), (1.5, 2.5)].iter().map(|(a, b)| line(*a, *b, 4, |x| line(low(x), high(x), 4, |y| bell(x, y)))).sum::<f64>()
                            }
                            Map::Squaring => line(0.0, 2.0, 8, |y| line(y * y / 4.0 - 1.0, 1.0 - y * y / 4.0, 8, |x| bell(x, y))),
                            // x = 2R sin φ: the one-variable rule only, to keep the ends
                            // of the ellipse smooth
                            Map::Ellipse => line(-PI64 / 2.0, PI64 / 2.0, 16, |phi| {
                                let x = 2.0 * r * phi.sin();
                                let h = r * phi.cos();
                                2.0 * r * phi.cos() * line(-h, h, 8, |y| bell(x, y))
                            }),
                            Map::Polar => unreachable!(),
                        };
                        assert!(rel(e, direct) < 1e-9, "{map:?} {fun:?} R {r}: {e} against {direct} in the plane");
                    }
                    let erf = |x: f64| 2.0 / PI64.sqrt() * line(0.0, x, 8, |t| (-t * t).exp());
                    let without = match (map, fun) {
                        (Map::Polar | Map::Ellipse, Fun::One) => TAU64 * r,
                        (Map::Polar, Fun::Bell) => PI64.powf(1.5) * erf(r),
                        (Map::Polar, Fun::Squares) => TAU64 * r.powi(3) / 3.0,
                        (Map::Ellipse, Fun::Squares) => PI64 * 5.0 * r.powi(3) / 3.0,
                        (Map::Linear | Map::Squaring, Fun::One) => 1.0,
                        (Map::Linear, Fun::Squares) => 2.5 / 3.0 + 2.0 / 3.0 + 2.0 * 1.0 / 4.0,
                        (Map::Squaring, Fun::Squares) => 28.0 / 45.0,
                        _ => unfactored(map, fun, r),
                    };
                    assert!(rel(unfactored(map, fun, r), without) < 1e-10, "{map:?} {fun:?} R {r}: {} against {without}", unfactored(map, fun, r));
                }
            }
        }
        assert!((PI64.powf(1.5) * 0.9999779095030014 - unfactored(Map::Polar, Fun::Bell, 3.0)).abs() < 1e-12);
    }

    /// The rule: as the cells shrink, the sum with the true areas and the sum with the
    /// factor both close in on the exact integral, and the sum without the factor on
    /// the integral of f(T) over the rectangle.
    #[test]
    fn the_sums_close_in_on_what_the_chart_says() {
        for map in MAPS {
            let areas = Areas::of(map);
            for fun in FUNS {
                for r in [R_LEAST, 1.5, R_MOST] {
                    let s = sums(map, fun, r, N_MAX, &areas);
                    let (e, _) = exact(map, fun, r);
                    assert!(rel(s.true_areas, e) < 2e-3 && rel(s.factor, e) < 2e-3, "{map:?} {fun:?} R {r}: {s:?} against {e}");
                    assert!(rel(s.plain, unfactored(map, fun, r)) < 2e-3, "{map:?} {fun:?} R {r}: {s:?}");
                    // with f = 1 the true areas add up to the region's area at every n
                    if fun == Fun::One {
                        for n in [1, 2, 7, N_MAX] {
                            assert!(rel(sums(map, fun, r, n, &areas).true_areas, e) < 1e-6, "{map:?} R {r} n {n}");
                        }
                    }
                }
            }
        }
    }

    /// The scene as experiment k's button leaves it, run at 60 frames a second until
    /// it has come to rest, from where another experiment left it.
    fn run(k: usize) -> ChangeVar {
        let mut scene = ChangeVar::new();
        scene.set_up((k + 2) % TRIALS.len());
        for _ in 0..30 {
            scene.advance(1.0 / 60.0);
        }
        scene.set_up(k);
        for _ in 0..60 * 10 {
            scene.advance(1.0 / 60.0);
            if scene.at_rest() {
                break;
            }
        }
        assert!(scene.at_rest(), "experiment {k} has not come to rest");
        scene
    }

    /// The scene opens as the first experiment sets it up.
    #[test]
    fn the_scene_opens_on_the_first_experiment() {
        let (opened, first) = (ChangeVar::new(), run(0));
        assert!(opened.map == first.map && opened.fun == first.fun && opened.n == first.n && opened.r == first.r && opened.pick == first.pick);
    }

    #[test]
    fn polar_cells_do_what_their_caption_says() {
        let scene = run(0);
        let (map, r, n) = (scene.map, scene.r, scene.n);
        let areas = scene.areas_of(map).expect("the areas are worked out");
        let size = cell_size(map, r, n);
        assert_eq!([num(r / n as f64, 3), sig(size, 4), num(PI64 / 4.0 * 0.375, 4)], ["0.375", "0.2945", "0.2945"]);
        assert!((size - 0.375 * PI64 / 4.0).abs() < 1e-15);
        // the picked cell is of the outer ring, at r = 2.81, area 0.8283; the innermost
        // ring's wedges are at r = 0.19, area 0.0552, a fifteenth
        let (i, j) = picked(scene.pick, n);
        let c = scene.picked_numbers();
        assert!(i == n - 1 && c.det > 0.0);
        assert_eq!([num(c.det, 2), sig(c.area, 4)], ["2.81", "0.8283"]);
        let inner = areas.get(map, r, n, 0, j);
        assert_eq!([num(0.5 * r / n as f64, 2), sig(inner, 3)], ["0.19", "0.0552"]);
        assert!((c.area / inner - 15.0).abs() < 1e-5, "{}", c.area / inner);
        // each a piece of ring, of area r Δr Δt exactly, r at its middle
        for i in 0..n {
            let mid = (i as f64 + 0.5) * r / n as f64;
            assert!(rel(areas.get(map, r, n, i, j), mid * size) < 1e-6);
            // and the tint brighter further out
            if i > 0 {
                assert!(stretch(map, r, n, i, j, Some(areas)) > stretch(map, r, n, i - 1, j, Some(areas)));
            }
        }
    }

    #[test]
    fn the_linear_map_does_what_its_caption_says() {
        let scene = run(1);
        let (map, n) = (scene.map, scene.n);
        assert_eq!(map, Map::Linear);
        let a = linear();
        assert!(a.determinant() == -2.0 && a * DVec2::new(1.0, 0.0) == DVec2::new(1.5, 0.5) && a * DVec2::new(0.0, 1.0) == DVec2::new(1.0, -1.0));
        let areas = scene.areas_of(map).expect("the areas are worked out");
        let size = cell_size(map, 3.0, n);
        // every cell lands with twice its area, turned over: one tint everywhere
        for j in 0..n {
            for i in 0..n {
                let got = areas.get(map, scene.r, n, i, j);
                assert!((got + 2.0 * size).abs() < 1e-12, "{got}");
                assert!((stretch(map, scene.r, n, i, j, Some(areas)) - 2.0).abs() < 1e-9);
            }
        }
        // the green side clockwise from the pink one
        let j = map.jacobian(DVec2::new(0.3, 0.6));
        assert!(DVec2::X.perp_dot(DVec2::Y) > 0.0 && j.col(0).perp_dot(j.col(1)) < 0.0);
        // the region's area is 2, twice the square's, at every n
        let s = scene.numbers.at(n);
        let d = decimals(scene.numbers.exact, 5);
        assert_eq!([num(s.true_areas, d), num(s.factor, d), num(s.plain, d), num(scene.numbers.exact, d)], ["2.0000", "2.0000", "1.0000", "2.0000"]);
        assert_eq!(scene.numbers.closed, Some(r"|\det A|"));
    }

    #[test]
    fn the_disc_does_what_its_caption_says() {
        let scene = run(2);
        let (map, r, n) = (scene.map, scene.r, scene.n);
        assert!(map == Map::Polar && r == 1.5 && n * n == 16 && scene.fun == Fun::One);
        let d = decimals(scene.numbers.exact, 5);
        assert_eq!([num(PI64 * r * r, d), num(TAU64 * r, d)], ["7.0686", "9.4248"]);
        // exactly πR² at every n, and the yellow and cyan lines on the white one
        for (k, s) in scene.numbers.series.iter().enumerate() {
            assert!(rel(s.factor, PI64 * r * r) < 1e-13 && rel(s.true_areas, PI64 * r * r) < 1e-6, "n = {}: {s:?}", k + 1);
            assert!(rel(s.plain, TAU64 * r) < 1e-13);
        }
        // the picked cell is a wedge, on the edge r = 0, where det J = 0 and the whole
        // edge lands on the point 0
        let (i, _) = picked(scene.pick, n);
        assert_eq!(i, 0);
        for t in [0.0, 1.0, 2.5, TAU64] {
            let p = DVec2::new(0.0, t);
            assert!(map.jacobian(p).determinant() == 0.0 && map.to(p).length() == 0.0);
        }
    }

    #[test]
    fn the_bell_does_what_its_caption_says() {
        let scene = run(3);
        assert!(scene.map == Map::Polar && scene.fun == Fun::Bell && scene.r == 3.0 && scene.n == 8);
        let nums = &scene.numbers;
        let d = decimals(nums.exact, 5);
        assert_eq!([num(nums.exact, d), num(PI64 * (1.0 - (-9.0f64).exp()), 4)], ["3.1412", "3.1412"]);
        assert_eq!([num(nums.at(8).factor, d), num(nums.at(32).factor, d)], ["3.1790", "3.1435"]);
        assert_eq!([num(nums.at(8).true_areas, d), num(nums.at(32).true_areas, d)], ["3.1790", "3.1435"]);
        assert_eq!([num(PI64 * (-9.0f64).exp(), 4), num(PI64.sqrt(), 4)], ["0.0004", "1.7725"]);
        assert_eq!([num(nums.at(8).plain, d), num(nums.without, d)], ["5.5682", "5.5682"]);
        // e^(−r²) r integrates to π(1 − e^(−9)) over the disc; over the line e^(−x²)
        // integrates to √π, and e^(−s²/2) to √(2π)
        assert!((TAU64 * line(0.0, 3.0, 4, |r| (-r * r).exp() * r) - nums.exact).abs() < 1e-12);
        assert!((line(-12.0, 12.0, 24, |x| (-x * x).exp()) - PI64.sqrt()).abs() < 1e-12);
        assert!((line(-16.0, 16.0, 32, |s| (-s * s / 2.0).exp()) - TAU64.sqrt()).abs() < 1e-12);
        // and the sums close in as the cells shrink
        assert!((nums.at(32).factor - nums.exact).abs() < (nums.at(8).factor - nums.exact).abs());
    }

    #[test]
    fn forgetting_the_factor_does_what_its_caption_says() {
        let scene = run(4);
        let (map, n) = (scene.map, scene.n);
        assert!(map == Map::Squaring && scene.fun == Fun::One && n == 8);
        let nums = &scene.numbers;
        let d = decimals(nums.exact, 5);
        assert_eq!([num(nums.exact, d), num(nums.at(8).factor, d)], ["2.6667", "2.6562"]);
        // (the midpoint rule falls short of each of the two squares' integrals by h²/12)
        assert!((nums.at(8).factor - (8.0 / 3.0 - 1.0 / 96.0)).abs() < 1e-14);
        assert_eq!(nums.closed, Some("8/3"));
        // without the factor, the square's area at every n
        assert!(nums.series.iter().all(|s| (s.plain - 1.0).abs() < 1e-12) && (nums.without - 1.0).abs() < 1e-12);
        // the factor from 0 at (0, 0) to 8 at (1, 1), 8/3 on average; the first cell
        // shrinks to a 24th, the picked one grows 7.04 times
        let areas = scene.areas_of(map).expect("the areas are worked out");
        assert!(map.jacobian(DVec2::ZERO).determinant() == 0.0 && map.jacobian(DVec2::ONE).determinant() == 8.0);
        assert!(rel(integrate(map.rect(1.0), 1, |p| map.jacobian(p).determinant()), 8.0 / 3.0) < 1e-13);
        assert!(rel(stretch(map, 1.0, n, 0, 0, Some(areas)), 1.0 / 24.0) < 1e-6);
        let (i, j) = picked(scene.pick, n);
        assert_eq!([i, j], [n - 1, n - 1]);
        assert_eq!(num(stretch(map, 1.0, n, i, j, Some(areas)), 2), "7.04");
        // closing in
        assert!((nums.at(32).factor - nums.exact).abs() < (nums.at(8).factor - nums.exact).abs() / 10.0);
    }

    /// The rule as the picture writes it out for each map and function: f(T(u, v)) as
    /// written (read off the formulas by hand here) is f(T(u, v)), and the factor
    /// written is |det J|.
    #[test]
    fn the_rule_as_written_is_the_rule() {
        let q = |a: f64, b: f64| a * a + b * b;
        for map in MAPS {
            for p in points(map, 2.0) {
                let (u, v) = (p.x, p.y);
                let factor = match map.factor_in() {
                    "r" => u,
                    "2r" => 2.0 * u,
                    "2" => 2.0,
                    _ => 4.0 * q(u, v),
                };
                assert!((factor - map.jacobian(p).determinant().abs()).abs() < 1e-12, "{map:?}");
                for fun in FUNS {
                    let written = match (fun, map) {
                        (Fun::One, _) => 1.0,
                        (Fun::Bell, Map::Polar) => (-u * u).exp(),
                        (Fun::Bell, Map::Ellipse) => (-u * u * (4.0 * v.cos().powi(2) + v.sin().powi(2))).exp(),
                        (Fun::Bell, Map::Linear) => (-(1.5 * u + v).powi(2) - (0.5 * u - v).powi(2)).exp(),
                        (Fun::Bell, Map::Squaring) => (-q(u, v).powi(2)).exp(),
                        (Fun::Squares, Map::Polar) => u * u,
                        (Fun::Squares, Map::Ellipse) => u * u * (4.0 * v.cos().powi(2) + v.sin().powi(2)),
                        (Fun::Squares, Map::Linear) => (1.5 * u + v).powi(2) + (0.5 * u - v).powi(2),
                        (Fun::Squares, Map::Squaring) => q(u, v).powi(2),
                    };
                    let truth = fun.at(map.to(p));
                    assert!((written - truth).abs() < 1e-12 * (1.0 + truth.abs()), "{map:?} {fun:?} at {p}: {written} against {truth}");
                    assert!(composed(map, fun).is_empty() == (fun == Fun::One) && crate::typeset::inspect(&format!("${}$", rule(map, fun))).error.is_none());
                }
            }
        }
    }

    /// The one-variable rule in the words: x = g(u) = 2 − u² carries [0, 1] onto
    /// [1, 2] (backwards), and ∫ f dx over [1, 2] is ∫ f(g(u)) |g'(u)| du over [0, 1].
    #[test]
    fn the_one_variable_rule_holds() {
        let f = |x: f64| x * x + (3.0 * x).sin();
        let lhs = line(1.0, 2.0, 2, f);
        let rhs = line(0.0, 1.0, 2, |u| f(2.0 - u * u) * (2.0 * u).abs());
        assert!((lhs - rhs).abs() < 1e-12, "{lhs} {rhs}");
    }

    /// Whatever is asked, every number drawn or read stays finite, and the scene comes
    /// to rest: every map, the fewest and the most cells, the smallest and the largest
    /// radius, a point on every edge and corner, and one that is not a number.
    #[test]
    fn degenerate_settings_leave_everything_finite() {
        let planes = [(Vec2::new(40.0, 100.0), Vec2::new(340.0, 400.0)), (Vec2::new(400.0, 100.0), Vec2::new(700.0, 400.0))];
        // (one scene, its settings changed one after another, as a visitor would)
        let mut scene = ChangeVar::new();
        for map in MAPS {
            for fun in FUNS {
                for n in [1, N_MAX, 0, 1000] {
                    for r in [R_LEAST, R_MOST, f64::NAN, 50.0] {
                        scene.set_map(map);
                        scene.fun = fun;
                        scene.set_n(n);
                        scene.set_r(r);
                        for pick in [DVec2::ZERO, DVec2::ONE, DVec2::new(1.0 - 1e-12, 0.0), DVec2::NAN, DVec2::new(-3.0, 7.0)] {
                            scene.pick = share(map, scene.r, point(map, scene.r, pick));
                            for _ in 0..60 * 3 {
                                scene.advance(1.0 / 60.0);
                                if scene.at_rest() {
                                    break;
                                }
                            }
                            assert!(scene.at_rest(), "{map:?} {n} {r}");
                            assert!((1..=N_MAX).contains(&scene.n) && scene.r >= R_LEAST && scene.r <= R_MOST);
                            let c = scene.picked_numbers();
                            assert!(c.size.is_finite() && c.area.is_finite() && c.det.is_finite() && c.value.is_finite(), "{map:?} {c:?}");
                            assert!(scene.numbers.series.iter().all(|s| s.true_areas.is_finite() && s.factor.is_finite() && s.plain.is_finite()));
                            assert!(scene.numbers.exact.is_finite() && scene.numbers.without.is_finite() && scene.numbers.top() > 0.0);
                            assert!(scene.chart_target().iter().all(|v| v.is_finite()));
                            let look = scene.look();
                            for s in [DVec2::ZERO, DVec2::ONE, DVec2::new(0.5, 0.0), scene.pick] {
                                assert!(look.at(&planes, 0, s).is_finite() && look.at(&planes, 1, s).is_finite(), "{map:?} at {s}");
                            }
                            // the dots dragged anywhere, the centre of the (x, y) plane among them
                            for at in [Vec2::new(550.0, 250.0), to_screen(map.xy_box(), planes[1], DVec2::ZERO), Vec2::new(-1e6, 1e6), Vec2::NAN] {
                                let (a, b) = (scene.share_uv(&planes, at), scene.share_xy(&planes, at));
                                assert!(a.is_finite() && b.is_finite() && a.min_element() >= 0.0 && b.max_element() < 1.0, "{map:?} {at}: {a} {b}");
                            }
                            for text in [sig(c.area, 4), trimmed(c.det, 5), trimmed(c.size, 5), num(scene.numbers.exact, 4)] {
                                assert!(!text.contains('?') && !text.contains("NaN") && !text.contains("inf"), "{text}");
                            }
                        }
                    }
                }
            }
        }
    }

    /// A change of map glides the picture from the one map to the other, starting where
    /// it was and ending where it should be; a change of the number of cells fades.
    #[test]
    fn changes_glide_and_fade() {
        let planes = [(Vec2::new(40.0, 100.0), Vec2::new(340.0, 400.0)), (Vec2::new(400.0, 100.0), Vec2::new(700.0, 400.0))];
        let mut scene = ChangeVar::new();
        let s = DVec2::new(0.3, 0.7);
        let before = scene.look().at(&planes, 1, s);
        scene.set_map(Map::Squaring);
        assert!(scene.old.is_none() && scene.look().at(&planes, 1, s) == before);
        let mut last = before;
        for _ in 0..120 {
            scene.advance(1.0 / 60.0);
            let now = scene.look().at(&planes, 1, s);
            assert!((now - last).length() < 30.0, "{last} to {now}");
            last = now;
        }
        assert!(scene.at_rest() && last == place(Map::Squaring, scene.r, &planes, 1, s));
        scene.set_n(3);
        assert!(scene.old.is_some_and(|o| o.n == 8) && scene.look().n == 3);
        for _ in 0..60 {
            scene.advance(1.0 / 60.0);
        }
        assert!(scene.old.is_none() && scene.at_rest());
    }

    /// The picture's parts fit the canvas, from a phone to a desktop, and keep off one
    /// another.
    #[test]
    fn the_layout_fits_every_canvas() {
        for (w, h) in [(1080.0, 1000.0), (740.0, 700.0), (390.0, 800.0), (390.0, 560.0), (800.0, 390.0), (1600.0, 1000.0), (240.0, 180.0), (620.0, 480.0), (1300.0, 700.0), (2400.0, 1400.0)] {
            for inset in [0.0, 46.0] {
                let view = Vec2::new(w, h);
                let lay = Layout::of(view, inset);
                let within = |r: Rect| r.0.cmpge(Vec2::ZERO).all() && r.1.cmple(view + Vec2::splat(0.5)).all();
                for r in lay.maps.iter().chain(&lay.funs) {
                    assert!(within(*r) && r.0.y >= inset, "{w}x{h}: a button at {r:?}");
                }
                let Some([a, b]) = lay.planes else { continue };
                assert!(within(a) && within(b) && a.1.x < b.0.x, "{w}x{h}: {a:?} {b:?}");
                assert!((a.1 - a.0).x == (a.1 - a.0).y && (b.1 - b.0).x == (b.1 - b.0).y);
                let bottom_of_planes = a.1.y + 15.0;
                if let Some((at, _)) = lay.cell {
                    assert!(at.y > bottom_of_planes && at.y + 2.0 * line_h(lay.small) <= h, "{w}x{h}: the lines on the cell at {at}");
                }
                if let Some((at, width)) = lay.sums {
                    assert!(at.y > bottom_of_planes && at.x + width <= w + 0.5 && at.y + 5.0 * line_h(lay.sums_size) <= h, "{w}x{h}: the sums at {at}");
                }
                if let Some(c) = lay.chart {
                    assert!(within(c) && c.0.y > bottom_of_planes && c.1.y + 20.0 <= h, "{w}x{h}: the chart at {c:?}");
                    if let Some((at, width)) = lay.sums {
                        let sums = (at, at + Vec2::new(width, 5.0 * line_h(lay.sums_size)));
                        assert!(sums.1.x < c.0.x - 30.0 || sums.1.y < c.0.y - 20.0, "{w}x{h}: the sums run into the chart");
                    }
                }
            }
        }
    }

    /// The numbers behind the captions (run with --ignored --nocapture).
    #[test]
    #[ignore]
    fn explore() {
        for map in MAPS {
            let areas = Areas::of(map);
            for fun in FUNS {
                for r in [1.5, 3.0] {
                    if !map.radial() && r != 3.0 {
                        continue;
                    }
                    let (e, form) = exact(map, fun, r);
                    println!("{map:?} {fun:?} R={r}: exact {e:.6} ({form:?}), without the factor {:.6}", unfactored(map, fun, r));
                    for n in [1, 2, 4, 8, 16, 24, 32] {
                        let s = sums(map, fun, r, n, &areas);
                        println!("   n={n:2}: true {:.6}  factor {:.6}  plain {:.6}", s.true_areas, s.factor, s.plain);
                    }
                }
            }
        }
        for t in &TRIALS {
            let c = picked_numbers(t.map, t.fun, t.r, t.n, DVec2::from(t.pick), None);
            println!("{}: {c:?} stretch {}", t.label, c.area / c.size);
        }
    }
}

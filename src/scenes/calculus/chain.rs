//! The chain rule: do one map and then another, and the small changes they make
//! multiply. Three planes in a row, x to y to z: a small step at p pushed forwards
//! through the Jacobians, and the gradient of a final number L pulled backwards through
//! their transposes. And the same with one variable, on three number lines.

use super::*;
use crate::scenes::backprop::{BACKWARD, FORWARD, LOSS};
use crate::scenes::stats::plot::{self, Rect};
use glam::{DMat2, DVec2};

pub const DEF: SceneDef = SceneDef {
    id: "chain",
    title: "The chain rule: Jacobians multiply",
    group: "Calculus",
    blurb: "One map after another: small changes pass through each in turn, forwards as columns and backwards as rows, which is all that backpropagation is.",
    make: |_| Box::new(ChainRule::new()),
};

// (the colours of the course on backpropagation, which this scene leads into: what goes
// forwards cyan, what comes back pink, the final number orange; and, as in the scene on
// Jacobians, where a small step really goes yellow)
const TRUE: Color = pal::YELLOW;

/// The grid, and where the maps carry it.
const GRID_COLOUR: Color = Color::hex(0x6F85B5);

// ----- the maps, in double precision ---------------------------------------------------

/// The maps of the plane that f and g can be.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Kind {
    Wave,
    Swirl,
    Layer,
    Linear,
}

const KINDS: [Kind; 4] = [Kind::Wave, Kind::Swirl, Kind::Layer, Kind::Linear];

/// The wave: how far it moves a point, and how fast it waves.
const WAVE: f64 = 0.35;
const WAVE_K: f64 = 1.5;
/// The swirl: how far it turns the centre (radians); further out it turns less.
const SWIRL: f64 = 1.6;
/// The tanh layer, tanh(Wv + b) unit by unit: its weights (by rows) and biases.
const LAYER_W: [[f64; 2]; 2] = [[1.45, -0.9], [0.8, 1.25]];
const LAYER_B: [f64; 2] = [0.15, -0.15];
/// The linear map, by rows.
const LINEAR: [[f64; 2]; 2] = [[1.2, -0.5], [0.4, 0.9]];

/// A matrix by its rows, the way matrices are written.
fn rows(m: [[f64; 2]; 2]) -> DMat2 {
    DMat2::from_cols(DVec2::new(m[0][0], m[1][0]), DVec2::new(m[0][1], m[1][1]))
}

/// The matrix a bᵀ.
fn outer(a: DVec2, b: DVec2) -> DMat2 {
    DMat2::from_cols(a * b.x, a * b.y)
}

impl Kind {
    fn name(self) -> &'static str {
        match self {
            Kind::Wave => "wave",
            Kind::Swirl => "swirl",
            Kind::Layer => "tanh layer",
            Kind::Linear => "linear",
        }
    }

    /// The map at the point v.
    fn at(self, v: DVec2) -> DVec2 {
        match self {
            Kind::Wave => DVec2::new(v.x + WAVE * (WAVE_K * v.y).sin(), v.y + WAVE * (WAVE_K * v.x).sin()),
            Kind::Swirl => DMat2::from_angle(SWIRL * (-0.5 * v.length_squared()).exp()) * v,
            Kind::Layer => {
                let a = rows(LAYER_W) * v + DVec2::from(LAYER_B);
                DVec2::new(a.x.tanh(), a.y.tanh())
            }
            Kind::Linear => rows(LINEAR) * v,
        }
    }

    /// Its Jacobian at v, worked out exactly (the tests hold it against differences).
    fn jacobian(self, v: DVec2) -> DMat2 {
        match self {
            Kind::Wave => {
                let k = WAVE * WAVE_K;
                rows([[1.0, k * (WAVE_K * v.y).cos()], [k * (WAVE_K * v.x).cos(), 1.0]])
            }
            Kind::Swirl => {
                // v turned by θ(v) = SWIRL e^(−|v|²/2): d(R(θ)v) = R(θ) dv + R(θ) J₀ v dθ,
                // with J₀ the quarter turn and dθ = −θ v · dv
                let theta = SWIRL * (-0.5 * v.length_squared()).exp();
                DMat2::from_angle(theta) * (DMat2::IDENTITY - outer(v.perp(), v) * theta)
            }
            Kind::Layer => {
                let a = rows(LAYER_W) * v + DVec2::from(LAYER_B);
                let (s, t) = (a.x.tanh(), a.y.tanh());
                // each unit's slope, 1 − tanh², times its row of W
                DMat2::from_diagonal(DVec2::new(1.0 - s * s, 1.0 - t * t)) * rows(LAYER_W)
            }
            Kind::Linear => rows(LINEAR),
        }
    }

    /// Its formula, for the panel, in the letter given (x or y).
    fn formula(self, map: &str, v: &str) -> String {
        match self {
            Kind::Wave => format!(r"${map}({v}) = ({v}_1 + 0.35 \sin 1.5{v}_2,\ {v}_2 + 0.35 \sin 1.5{v}_1)$"),
            Kind::Swirl => format!(r"${map}({v})$ is ${v}$ turned by the angle $1.6\, e^{{-|{v}|^2/2}}$: a lot near $0$, hardly at all far out"),
            Kind::Layer => format!(
                r"${map}({v}) = \tanh(W{v} + b)$, unit by unit, with $W = \begin{{bmatrix}} 1.45 & -0.9 \\ 0.8 & 1.25 \end{{bmatrix}}$ and $b = (0.15, -0.15)$: a layer of a neural network"
            ),
            Kind::Linear => format!(r"${map}({v}) = M{v}$ with $M = \begin{{bmatrix}} 1.2 & -0.5 \\ 0.4 & 0.9 \end{{bmatrix}}$: its Jacobian is $M$ everywhere"),
        }
    }
}

/// A map as drawn: a weighted sum of the kinds, which glides from one kind to another
/// when another is chosen (one weight is 1 and the others 0 once it has arrived). Its
/// Jacobian is the same sum of theirs, so what is drawn while it glides is still a map
/// with its own exact Jacobian.
#[derive(Clone, Copy, Debug, PartialEq)]
struct Map([f64; 4]);

impl Map {
    fn of(kind: Kind) -> Map {
        Map(KINDS.map(|k| if k == kind { 1.0 } else { 0.0 }))
    }

    fn at(&self, v: DVec2) -> DVec2 {
        KINDS.iter().zip(self.0).filter(|(_, w)| *w != 0.0).fold(DVec2::ZERO, |s, (k, w)| s + k.at(v) * w)
    }

    fn jacobian(&self, v: DVec2) -> DMat2 {
        KINDS.iter().zip(self.0).filter(|(_, w)| *w != 0.0).fold(DMat2::ZERO, |s, (k, w)| s + k.jacobian(v) * w)
    }

    /// The kind it is, once it has arrived.
    fn kind(&self) -> Option<Kind> {
        KINDS.iter().zip(self.0).find(|(_, w)| *w == 1.0).map(|(k, _)| *k)
    }

    fn bits(&self) -> [u64; 4] {
        self.0.map(f64::to_bits)
    }
}

/// The most a 2×2 matrix lengthens a step: its largest singular value.
fn stretch(m: DMat2) -> f64 {
    let s = m.x_axis.length_squared() + m.y_axis.length_squared();
    let d = m.determinant();
    (0.5 * (s + (s * s - 4.0 * d * d).max(0.0).sqrt())).max(0.0).sqrt()
}

/// Everything the chain rule says at the point p, for a step dx there and a gradient u
/// of the final number at the end.
#[derive(Clone, Copy, Debug)]
struct At {
    p: DVec2,
    y: DVec2,
    z: DVec2,
    jf: DMat2,
    jg: DMat2,
    /// J = J_g(f(p)) J_f(p).
    j: DMat2,
    /// Forwards: the step and where the Jacobians carry it.
    dx: DVec2,
    dy: DVec2,
    dz: DVec2,
    /// Backwards: the gradient at z, and where the transposes carry it back.
    u: DVec2,
    gy: DVec2,
    gx: DVec2,
}

fn chain(f: &Map, g: &Map, p: DVec2, dx: DVec2, u: DVec2) -> At {
    let y = f.at(p);
    let z = g.at(y);
    let (jf, jg) = (f.jacobian(p), g.jacobian(y));
    let dy = jf * dx;
    let gy = jg.transpose() * u;
    At { p, y, z, jf, jg, j: jg * jf, dx, dy, dz: jg * dy, u, gy, gx: jf.transpose() * gy }
}

impl At {
    /// The effect of the step on L worked out forwards: u · (J_g (J_f dx)).
    fn forwards(&self) -> f64 {
        self.u.dot(self.dz)
    }

    /// The same backwards: (J_fᵀ (J_gᵀ u)) · dx.
    fn backwards(&self) -> f64 {
        self.gx.dot(self.dx)
    }
}

/// How much L = u · z really changes when x goes from p to p + dx.
fn true_change(f: &Map, g: &Map, p: DVec2, dx: DVec2, u: DVec2) -> f64 {
    u.dot(g.at(f.at(p + dx)) - g.at(f.at(p)))
}

// ----- one variable --------------------------------------------------------------------

/// f(x) = x³ and g(y) = arctan y, and their slopes. At x = 1 the slopes are 3 and 1/2.
fn f1(x: f64) -> f64 {
    x * x * x
}

fn df1(x: f64) -> f64 {
    3.0 * x * x
}

fn g1(y: f64) -> f64 {
    y.atan()
}

fn dg1(y: f64) -> f64 {
    1.0 / (1.0 + y * y)
}

/// The same with one variable: (p, f(p), g(f(p))), the slopes f′(p) and g′(f(p)), the
/// step dx and its pushes, and the gradient u at the end and its pulls.
#[derive(Clone, Copy, Debug)]
struct At1 {
    p: f64,
    y: f64,
    z: f64,
    df: f64,
    dg: f64,
    dx: f64,
    dy: f64,
    dz: f64,
    u: f64,
    gy: f64,
    gx: f64,
}

fn chain1(p: f64, dx: f64, u: f64) -> At1 {
    let (y, df) = (f1(p), df1(p));
    let (z, dg) = (g1(y), dg1(y));
    At1 { p, y, z, df, dg, dx, dy: df * dx, dz: dg * df * dx, u, gy: dg * u, gx: df * dg * u }
}

/// The furthest p may go either way, with one variable.
const P1_MAX: f64 = 1.25;
const DX1_MAX: f64 = 0.5;
const U1_MAX: f64 = 1.5;

// ----- the planes ----------------------------------------------------------------------

/// The first plane's grid: the square |x₁|, |x₂| ≤ `GRID_HALF`, with this many lines each
/// way, each sampled at so many points (enough for the curves that the maps make of them).
const GRID_HALF: f64 = 2.0;
const GRID_LINES: usize = 9;
const GRID_SAMPLES: usize = 61;
/// How far the step's tip and the gradient's may be from where they start.
const DX_MAX: f64 = 1.2;
const U_MAX: f64 = 1.5;

/// The grid's lines in the first plane: those along x₂ first, then those along x₁.
fn grid_lines() -> Vec<Vec<DVec2>> {
    let at = |k: usize, n: usize| -GRID_HALF + 2.0 * GRID_HALF * k as f64 / (n - 1) as f64;
    let mut lines = Vec::with_capacity(2 * GRID_LINES);
    for i in 0..GRID_LINES {
        lines.push((0..GRID_SAMPLES).map(|s| DVec2::new(at(i, GRID_LINES), at(s, GRID_SAMPLES))).collect());
    }
    for i in 0..GRID_LINES {
        lines.push((0..GRID_SAMPLES).map(|s| DVec2::new(at(s, GRID_SAMPLES), at(i, GRID_LINES))).collect());
    }
    lines
}

/// The square that holds the points, with a margin: (centre, half its side).
fn window_of<'a>(points: impl Iterator<Item = &'a DVec2>) -> (DVec2, f64) {
    let (mut lo, mut hi) = (DVec2::splat(f64::MAX), DVec2::splat(f64::MIN));
    for p in points.filter(|p| p.is_finite()) {
        lo = lo.min(*p);
        hi = hi.max(*p);
    }
    if !(lo.x <= hi.x && lo.y <= hi.y) {
        return (DVec2::ZERO, 1.0);
    }
    let half = 0.5 * (hi - lo).max_element();
    ((lo + hi) * 0.5, (half * 1.07).max(0.05))
}

/// The grid, where the maps carry it, and each plane's window, kept for as long as the
/// maps are the same (to the bit).
struct Grids {
    of: [u64; 8],
    lines: [Vec<Vec<DVec2>>; 3],
    windows: [(DVec2, f64); 3],
}

impl Grids {
    fn of(f: &Map, g: &Map) -> Grids {
        let x = grid_lines();
        let y: Vec<Vec<DVec2>> = x.iter().map(|l| l.iter().map(|v| f.at(*v)).collect()).collect();
        let z: Vec<Vec<DVec2>> = y.iter().map(|l| l.iter().map(|v| g.at(*v)).collect()).collect();
        let windows = [(DVec2::ZERO, GRID_HALF * 1.07), window_of(y.iter().flatten()), window_of(z.iter().flatten())];
        let mut of = [0u64; 8];
        of[..4].copy_from_slice(&f.bits());
        of[4..].copy_from_slice(&g.bits());
        Grids { of, lines: [x, y, z], windows }
    }
}

/// A plane as it is drawn: the rectangle of the screen it fills, and the point of the
/// plane at its middle and the pixels to a unit there (the same across and up, so that
/// right angles look right).
#[derive(Clone, Copy, Debug)]
struct Plane {
    rect: Rect,
    centre: DVec2,
    scale: f64,
}

impl Plane {
    fn fit(rect: Rect, (centre, half): (DVec2, f64)) -> Plane {
        let side = (rect.1 - rect.0).min_element().max(1.0) as f64;
        Plane { rect, centre, scale: 0.5 * side / half.max(1e-6) }
    }

    fn mid(&self) -> Vec2 {
        (self.rect.0 + self.rect.1) * 0.5
    }

    /// Where the point v of the plane is on the screen.
    fn px(&self, v: DVec2) -> Vec2 {
        let d = (v - self.centre) * self.scale;
        self.mid() + Vec2::new(d.x as f32, -d.y as f32)
    }

    /// The point of the plane at a place on the screen.
    fn unit(&self, q: Vec2) -> DVec2 {
        let d = q - self.mid();
        self.centre + DVec2::new(d.x as f64, -d.y as f64) / self.scale
    }

    /// The part of the plane the rectangle shows: its lowest corner and its highest.
    fn shown(&self) -> (DVec2, DVec2) {
        let (a, b) = (self.unit(self.rect.0), self.unit(self.rect.1));
        (a.min(b), a.max(b))
    }

    fn contains(&self, q: Vec2) -> bool {
        plot::within(q, self.rect, 0.0)
    }
}

// ----- level lines ---------------------------------------------------------------------

/// Nodes across the shorter side of a plane where the final number is worked out for
/// its level lines.
const NODES: usize = 40;

/// A piece of a level line in one cell of the grid: from where it crosses one edge of
/// the cell to where it crosses another, with the larger values on its left; and those
/// two edges, by number, by which it is joined to the pieces before and after it.
#[derive(Clone, Copy, Debug)]
struct Piece {
    a: DVec2,
    b: DVec2,
    from: usize,
    to: usize,
}

/// The pieces of the level line φ = c in the cells (i, j) of a grid of values that it may
/// cross: `values[j * (nx + 1) + i]` is φ at `lo + (i, j) * cell`. Marching squares: in each cell the line crosses the edges
/// whose ends lie on either side of c, at the place found by straight interpolation; a
/// cell with all four edges crossed is decided by the value at its middle.
fn contour(values: &[f64], nx: usize, lo: DVec2, cell: DVec2, c: f64, cells: &[(usize, usize)]) -> Vec<Piece> {
    let node = |i: usize, j: usize| values[j * (nx + 1) + i];
    // (an edge's number: the node it starts from, and whether it goes up or across)
    let across = |i: usize, j: usize| 2 * (j * (nx + 1) + i);
    let up = |i: usize, j: usize| 2 * (j * (nx + 1) + i) + 1;
    let mut out = Vec::new();
    for &(i, j) in cells {
        {
            // (a node exactly at the level counts as a hair above it: the line then never
            // passes through a node, where it would cross two edges at once and the pieces
            // either side of it would not be joined)
            let v = [node(i, j), node(i + 1, j), node(i + 1, j + 1), node(i, j + 1)].map(|x| if x == c { c + 4.0 * f64::EPSILON * c.abs().max(1.0) } else { x });
            if !v.iter().all(|x| x.is_finite()) {
                continue;
            }
            let (least, most) = v.iter().fold((f64::MAX, f64::MIN), |(a, b), x| (a.min(*x), b.max(*x)));
            if !(c > least && c <= most) {
                continue;
            }
            let corner = |k: usize| lo + cell * DVec2::new((i + (k == 1 || k == 2) as usize) as f64, (j + (k >= 2) as usize) as f64);
            let edges = [across(i, j), up(i + 1, j), across(i, j + 1), up(i, j)];
            let above = v.map(|x| x > c);
            // where it crosses edge k, from corner k to corner k + 1
            let cross = |k: usize| {
                let (a, b) = (k, (k + 1) % 4);
                corner(a) + (corner(b) - corner(a)) * ((c - v[a]) / (v[b] - v[a]))
            };
            // the piece across edges k and m, turned so that `corner(high)` is on its left
            // if that corner is above c, on its right if not
            let mut piece = |k: usize, m: usize, high: usize| {
                let (p, q) = (cross(k), cross(m));
                let left = (q - p).perp_dot(corner(high) - p) > 0.0;
                out.push(if left == above[high] { Piece { a: p, b: q, from: edges[k], to: edges[m] } } else { Piece { a: q, b: p, from: edges[m], to: edges[k] } });
            };
            let crossed: Vec<usize> = (0..4).filter(|k| above[*k] != above[(k + 1) % 4]).collect();
            match crossed.len() {
                2 => {
                    // (any corner will do to tell the sides apart; one off the line)
                    let high = (0..4).max_by(|a, b| (v[*a] - c).abs().total_cmp(&(v[*b] - c).abs())).unwrap_or(0);
                    piece(crossed[0], crossed[1], high);
                }
                4 => {
                    // cut off the two corners on the other side from the middle
                    let middle = 0.25 * v.iter().sum::<f64>() > c;
                    for k in (0..4).filter(|k| above[*k] != middle) {
                        piece((k + 3) % 4, k, k);
                    }
                }
                _ => {}
            }
        }
    }
    out
}

/// The pieces joined into whole lines, each running with the larger values on its left,
/// in the order their first pieces were found.
fn chains(pieces: &[Piece]) -> Vec<Vec<DVec2>> {
    let mut starting: std::collections::HashMap<usize, usize> = std::collections::HashMap::with_capacity(pieces.len());
    let mut ending: std::collections::HashMap<usize, usize> = std::collections::HashMap::with_capacity(pieces.len());
    for (n, p) in pieces.iter().enumerate() {
        starting.insert(p.from, n);
        ending.insert(p.to, n);
    }
    let mut used = vec![false; pieces.len()];
    let mut out = Vec::new();
    for first in 0..pieces.len() {
        if used[first] {
            continue;
        }
        // back to where this line begins (or round a closed one, to where it began)
        let mut begin = first;
        while let Some(&before) = ending.get(&pieces[begin].from) {
            if before == first || used[before] {
                break;
            }
            begin = before;
        }
        let mut line = vec![pieces[begin].a];
        let mut at = begin;
        loop {
            used[at] = true;
            line.push(pieces[at].b);
            match starting.get(&pieces[at].to) {
                Some(&next) if !used[next] => at = next,
                _ => break,
            }
        }
        // a closed line starts at its lowest point (and leftmost of those), wherever it
        // was first found: so that where its points are drawn moves smoothly as it does
        let n = line.len();
        if n > 3 && line[0] == line[n - 1] {
            let low = (0..n - 1).min_by(|a, b| line[*a].y.total_cmp(&line[*b].y).then(line[*a].x.total_cmp(&line[*b].x))).unwrap_or(0);
            line.pop();
            line.rotate_left(low);
            line.push(line[0]);
        }
        out.push(line);
    }
    out
}

/// A line drawn with `n` points evenly spaced along it (none for a line of no length):
/// the same number of points however the line was found, each moving smoothly as the
/// line does.
fn resample(line: &[DVec2], n: usize) -> Vec<DVec2> {
    let mut along = Vec::with_capacity(line.len());
    let mut total = 0.0;
    for (k, p) in line.iter().enumerate() {
        if k > 0 {
            total += p.distance(line[k - 1]);
        }
        along.push(total);
    }
    if !(total > 0.0 && total.is_finite()) || n < 2 {
        return Vec::new();
    }
    let mut out = Vec::with_capacity(n);
    let mut k = 1;
    for i in 0..n {
        let want = total * i as f64 / (n - 1) as f64;
        while k + 1 < line.len() && along[k] < want {
            k += 1;
        }
        let span = (along[k] - along[k - 1]).max(1e-300);
        out.push(line[k - 1].lerp(line[k], ((want - along[k - 1]) / span).max(0.0).min(1.0)));
    }
    out
}

/// Where the segment from `a` (in the box `lo..hi`) to `b` leaves the box: `b` itself if
/// it does not.
fn to_edge(a: DVec2, b: DVec2, lo: DVec2, hi: DVec2) -> DVec2 {
    let d = b - a;
    let mut t = 1.0f64;
    for axis in 0..2 {
        if d[axis] > 0.0 && b[axis] > hi[axis] {
            t = t.min((hi[axis] - a[axis]) / d[axis]);
        } else if d[axis] < 0.0 && b[axis] < lo[axis] {
            t = t.min((lo[axis] - a[axis]) / d[axis]);
        }
    }
    a + d * t.max(0.0)
}

/// Points to a straight level line in the last plane, as drawn.
const STRAIGHT_POINTS: usize = 48;

/// Points to a level line as drawn.
const LINE_POINTS: usize = 40;

/// The level line of φ through `start`, both ways from it, as long as it stays in the
/// box `lo..hi`: steps of `h` along it (at right angles to the gradient, taken at the
/// middle of the step), each followed by a step back onto it along the gradient. `phi`
/// gives φ and its gradient. Empty where the gradient is too small to say which way the
/// line goes.
fn trace(phi: impl Fn(DVec2) -> (f64, DVec2), start: DVec2, h: f64, steps: usize, lo: DVec2, hi: DVec2) -> Vec<DVec2> {
    let (c, g0) = phi(start);
    let tiny = 1e-9;
    if !(c.is_finite() && g0.is_finite() && g0.length() > tiny) {
        return Vec::new();
    }
    let along = |g: DVec2, was: DVec2| {
        let t = g.perp().normalize_or_zero();
        if t.dot(was) < 0.0 {
            -t
        } else {
            t
        }
    };
    let mut halves: [Vec<DVec2>; 2] = [Vec::new(), Vec::new()];
    for (side, half) in halves.iter_mut().enumerate() {
        let mut v = start;
        let mut was = g0.perp().normalize() * if side == 0 { 1.0 } else { -1.0 };
        for n in 0..steps {
            let (_, g) = phi(v);
            if g.length().is_nan() || g.length() <= tiny {
                break;
            }
            let (_, gm) = phi(v + along(g, was) * (0.5 * h));
            if gm.length().is_nan() || gm.length() <= tiny {
                break;
            }
            let t = along(gm, was);
            let mut w = v + t * h;
            for _ in 0..2 {
                let (value, gw) = phi(w);
                if gw.length_squared() > tiny * tiny && value.is_finite() {
                    w -= gw * ((value - c) / gw.length_squared());
                }
            }
            if !w.is_finite() {
                break;
            }
            half.push(w);
            // (out of the box: that point is the last; a closed line round: done)
            if w.cmplt(lo).any() || w.cmpgt(hi).any() || (n > 8 && w.distance(start) < 0.6 * h) {
                break;
            }
            (was, v) = (t, w);
        }
    }
    let [ahead, behind] = halves;
    behind.into_iter().rev().chain([start]).chain(ahead).collect()
}

/// The level lines of L in a plane, and what they were found for.
#[derive(Default)]
struct Levels {
    /// The maps and the part of the plane the values are for, and the values: g(f(v))
    /// or g(v) at the nodes, from which u · that is L there for any u.
    nodes_of: Vec<u64>,
    nodes: Vec<DVec2>,
    nx: usize,
    ny: usize,
    lo: DVec2,
    cell: DVec2,
    /// The gradient u the lines are for, and the lines, each with `LINE_POINTS` points.
    lines_of: Vec<u64>,
    lines: Vec<Vec<DVec2>>,
    /// The line through the point, and what it was traced for.
    through_of: Vec<u64>,
    through: Vec<DVec2>,
}

/// The values L takes on its level lines: every multiple of `step`, the same in every
/// plane (they are the same number, carried back). `step` is chosen so that in the last
/// plane the lines are an eighth of its window apart, whatever u is.
fn level_step(u: DVec2, half_z: f64) -> f64 {
    u.length() * 2.0 * half_z / 8.0
}

impl Levels {
    /// Bring the lines up to date for the plane shown by `plane`, the map from it to the
    /// last plane (`to_end`: g∘f or g) and the gradient u at the end.
    fn update(&mut self, plane: &Plane, key: &[u64], to_end: impl Fn(DVec2) -> DVec2, u: DVec2, step: f64) {
        // (found a little beyond what is shown, and cut at the edge when drawn: a line
        // then comes into sight from outside, rather than starting up at the edge)
        let (lo, hi) = plane.shown();
        let margin = (hi - lo) * 0.12;
        let (lo, hi) = (lo - margin, hi + margin);
        let mut nodes_of = key.to_vec();
        nodes_of.extend([lo.x, lo.y, hi.x, hi.y].map(f64::to_bits));
        if self.nodes_of != nodes_of {
            let size = hi - lo;
            let cell = size.min_element().max(1e-9) / NODES as f64;
            let (nx, ny) = (((size.x / cell).round() as usize).clamp(1, 4 * NODES), ((size.y / cell).round() as usize).clamp(1, 4 * NODES));
            let cell = DVec2::new(size.x / nx as f64, size.y / ny as f64);
            self.nodes = (0..=ny).flat_map(|j| (0..=nx).map(move |i| (i, j))).map(|(i, j)| to_end(lo + cell * DVec2::new(i as f64, j as f64))).collect();
            (self.nx, self.ny, self.lo, self.cell, self.nodes_of) = (nx, ny, lo, cell, nodes_of);
            self.lines_of.clear();
        }
        let lines_of = vec![u.x.to_bits(), u.y.to_bits(), step.to_bits()];
        if self.lines_of != lines_of {
            self.lines.clear();
            if step > 0.0 && step.is_finite() {
                let values: Vec<f64> = self.nodes.iter().map(|n| u.dot(*n)).collect();
                let (least, most) = values.iter().filter(|v| v.is_finite()).fold((f64::MAX, f64::MIN), |(a, b), v| (a.min(*v), b.max(*v)));
                if least <= most {
                    // each level's cells, found in one pass over the grid: those whose
                    // corners' values reach from below it up to it or beyond
                    let first = (least / step).ceil() as i64;
                    let count = (((most / step).floor() as i64).min(first + 80) - first + 1).max(0) as usize;
                    let mut cells: Vec<Vec<(usize, usize)>> = vec![Vec::new(); count];
                    let nx = self.nx;
                    for j in 0..self.ny {
                        for i in 0..nx {
                            let v = [values[j * (nx + 1) + i], values[j * (nx + 1) + i + 1], values[(j + 1) * (nx + 1) + i + 1], values[(j + 1) * (nx + 1) + i]];
                            if !v.iter().all(|x| x.is_finite()) {
                                continue;
                            }
                            let (low, high) = v.iter().fold((f64::MAX, f64::MIN), |(a, b), x| (a.min(*x), b.max(*x)));
                            let from = ((low / step).floor() as i64 + 1).max(first);
                            let to = ((high / step).floor() as i64).min(first + count as i64 - 1);
                            for k in from..=to {
                                cells[(k - first) as usize].push((i, j));
                            }
                        }
                    }
                    for (n, cells) in cells.iter().enumerate() {
                        let pieces = contour(&values, nx, self.lo, self.cell, (first + n as i64) as f64 * step, cells);
                        self.lines.extend(chains(&pieces).iter().map(|l| resample(l, LINE_POINTS)).filter(|l| !l.is_empty()));
                    }
                }
            }
            self.lines_of = lines_of;
        }
    }

    /// The line through the point `at`, traced afresh only when something it depends on
    /// has changed.
    fn through(&mut self, plane: &Plane, key: &[u64], at: DVec2, phi: impl Fn(DVec2) -> (f64, DVec2)) -> &[DVec2] {
        let (lo, hi) = plane.shown();
        let mut of = key.to_vec();
        of.extend([at.x, at.y, lo.x, lo.y, hi.x, hi.y].map(f64::to_bits));
        if self.through_of != of {
            let h = (hi - lo).min_element() / 80.0;
            let steps = (((hi - lo).length() / h) as usize + 4).min(600);
            let mut line = trace(phi, at, h, steps, lo, hi);
            // (its ends where it leaves the box, exactly: then its length, and so where
            // its evenly spaced points fall, changes smoothly as the point moves)
            let n = line.len();
            if n >= 2 {
                line[n - 1] = to_edge(line[n - 2], line[n - 1], lo, hi);
                line[0] = to_edge(line[1], line[0], lo, hi);
            }
            self.through = resample(&line, 4 * LINE_POINTS);
            self.through_of = of;
        }
        &self.through
    }
}

// ----- the setting, and how it glides ----------------------------------------------------

/// Everything a change of setting glides: the point, the step and the gradient, in two
/// variables and in one, and the two maps.
#[derive(Clone, Copy, Debug, PartialEq)]
struct Setting {
    p: DVec2,
    dx: DVec2,
    u: DVec2,
    p1: f64,
    dx1: f64,
    u1: f64,
    f: Map,
    g: Map,
}

const NUMBERS: usize = 17;

impl Setting {
    fn numbers(&self) -> [f64; NUMBERS] {
        let mut v = [0.0; NUMBERS];
        v[..9].copy_from_slice(&[self.p.x, self.p.y, self.dx.x, self.dx.y, self.u.x, self.u.y, self.p1, self.dx1, self.u1]);
        v[9..13].copy_from_slice(&self.f.0);
        v[13..].copy_from_slice(&self.g.0);
        v
    }

    fn from_numbers(v: [f64; NUMBERS]) -> Setting {
        Setting {
            p: DVec2::new(v[0], v[1]),
            dx: DVec2::new(v[2], v[3]),
            u: DVec2::new(v[4], v[5]),
            p1: v[6],
            dx1: v[7],
            u1: v[8],
            f: Map([v[9], v[10], v[11], v[12]]),
            g: Map([v[13], v[14], v[15], v[16]]),
        }
    }
}

/// How fast a change of setting is glided to: each of the two eases one after the other
/// covers two thirds of what is left in this many seconds.
const FOLLOW: f64 = 0.1;

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
        if self.arrived() || dt.is_nan() || dt <= 0.0 {
            return;
        }
        let k = 1.0 - (-dt / FOLLOW).exp();
        let (to, mut between, mut shown) = (self.to.numbers(), self.between.numbers(), self.shown.numbers());
        let mut apart = 0.0f64;
        for i in 0..NUMBERS {
            between[i] += (to[i] - between[i]) * k;
            shown[i] += (between[i] - shown[i]) * k;
            apart = apart.max((to[i] - between[i]).abs()).max((to[i] - shown[i]).abs());
        }
        // (exactly there in the end: the picture then stops changing)
        if apart < 2e-4 || !apart.is_finite() {
            *self = Glide::at(self.to);
        } else {
            (self.between, self.shown) = (Setting::from_numbers(between), Setting::from_numbers(shown));
        }
    }
}

/// What the picture shows of the two sweeps.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Show {
    Forwards,
    Backwards,
    Both,
}

impl Show {
    fn forwards(self) -> bool {
        self != Show::Backwards
    }

    fn backwards(self) -> bool {
        self != Show::Forwards
    }
}

const SHOWS: [Show; 3] = [Show::Forwards, Show::Backwards, Show::Both];

/// Seconds the picture takes to fade from one variable to two, or back.
const MIX_SECONDS: f32 = 0.35;

/// Seconds the sweeps take for each of their stages: one plane's arrow growing.
const STAGE: f32 = 0.7;

/// How much of each arrow the sweeps have drawn `t` seconds in (all of it if they are not
/// playing): forwards in the planes x, y, z, then backwards in x, y, z. Forwards goes
/// from x to z; backwards from z to x, after forwards if both are shown.
fn reveal_at(show: Show, t: Option<f32>) -> [f32; 6] {
    let grown = |from: f32| match t {
        None => 1.0,
        Some(t) => smooth((t - from) / STAGE),
    };
    let back_from = if show.forwards() { 3.0 * STAGE } else { 0.0 };
    let mut out = [0.0; 6];
    for k in 0..3 {
        if show.forwards() {
            out[k] = grown(k as f32 * STAGE);
        }
        if show.backwards() {
            out[3 + k] = grown(back_from + (2 - k) as f32 * STAGE);
        }
    }
    out
}

/// How long the sweeps take.
fn film_length(show: Show) -> f32 {
    if show == Show::Both {
        6.0 * STAGE
    } else {
        3.0 * STAGE
    }
}

fn smooth(t: f32) -> f32 {
    let t = t.max(0.0).min(1.0);
    t * t * (3.0 - 2.0 * t)
}

// ----- experiments ---------------------------------------------------------------------

struct Trial {
    label: &'static str,
    look_for: &'static str,
    one: bool,
    show: Show,
    f: Kind,
    g: Kind,
    p: [f64; 2],
    dx: [f64; 2],
    u: [f64; 2],
    p1: f64,
    dx1: f64,
    u1: f64,
}

/// The setting the scene opens with, and the one the experiments in two variables start
/// from (the first experiment's one variable aside).
const START: Trial = Trial {
    label: "",
    look_for: "",
    one: false,
    show: Show::Both,
    f: Kind::Wave,
    g: Kind::Layer,
    p: [-0.35, 0.45],
    dx: [0.3, 0.12],
    u: [0.5, 0.15],
    p1: 1.0,
    dx1: 0.15,
    u1: 1.0,
};

const TRIALS: [Trial; 5] = [
    Trial {
        label: "slopes multiply",
        look_for: r"One variable: $f(x) = x^3$, then $g(y) = \arctan y$, at $p = 1$. A small step $\mathrm{d}x$ there (cyan) becomes a step $f'(1) = 3$ times as long at $f(p) = 1$, and then one half as long again, since $g'(1) = 1/2$: $3 \times 1/2 = 1.5$ times $\mathrm{d}x$ in all. That is the chain rule, $(g \circ f)'(p) = g'(f(p))\,f'(p)$: the rates multiply. The yellow bands are where the step really goes: $0.52$ along $y$ where the slope says $0.45$, since $x^3$ curves upwards. The shorter the step, the closer the two.",
        one: true,
        show: Show::Forwards,
        ..START
    },
    Trial {
        label: "a small step pushed forwards",
        look_for: r"The cyan step $\mathrm{d}x = (0.30, 0.12)$ at $p$ is carried to $J_f\,\mathrm{d}x = (0.35, 0.26)$ at $f(p)$, then to $J_g J_f\,\mathrm{d}x = (0.25, 0.60)$ at $g(f(p))$, each Jacobian taken where the point has got to. That is $J\,\mathrm{d}x$ with $J = J_g J_f$. The yellow curves are where the step really goes: they end $0.01$ and $0.08$ from the arrows' tips. Drag the cyan tip to halve the step and those misses fall to less than a third: for small steps a miss shrinks as the square of the step.",
        show: Show::Forwards,
        ..START
    },
    Trial {
        label: "a gradient pulled back",
        look_for: r"Let $L = u \cdot z$, a number that depends on where the chain ends. Its gradient is $u = (0.50, 0.15)$ (pink), at right angles to its straight level lines (orange). As a function of $y$, $L = u \cdot g(y)$: its level lines bend, and its gradient at $f(p)$ is $J_g^\top u = (0.79, -0.23)$, with the transpose (not $J_g u$, which is $(0.54, 0.58)$). As a function of $x$, its gradient at $p$ is $J_f^\top J_g^\top u = (0.68, 0.09)$. Each pink arrow is at right angles to the bright level line through its point.",
        show: Show::Backwards,
        ..START
    },
    Trial {
        label: "the same number from either end",
        look_for: r"How much does the step $\mathrm{d}x$ change $L$? Forwards: push the step to the end and take $u \cdot \bigl(J_g (J_f\,\mathrm{d}x)\bigr) = 0.216$. Backwards: pull $u$ back to the start and take $\bigl(J_f^\top (J_g^\top u)\bigr) \cdot \mathrm{d}x = 0.216$. The same number, as it must be: $u \cdot (Av) = (A^\top u) \cdot v$ for any matrix $A$. It is the change to first order; the true change is $0.203$. Forwards answers for this one $\mathrm{d}x$; backwards, once it has $J_f^\top J_g^\top u$, for every $\mathrm{d}x$ at once.",
        ..START
    },
    Trial {
        label: "one flat link: the chain goes flat",
        look_for: r"Here the two units of the tanh layer get the inputs $-2.6$ and $-2.5$, far out where tanh is flat (saturated): their slopes $1 - \tanh^2$ are $0.021$ and $0.024$, and $J_g$ stretches no step by more than $0.036$. $J_f$ stretches by up to $1.16$, but a product can stretch by no more than its factors allow, $\|J\| \le 0.036 \times 1.16$: here it is $0.041$. The step reaches $z$ as a dot, and the gradient comes back to $x$ less than a twentieth of the size of $u$. One flat link flattens the whole chain: the vanishing gradient, in miniature.",
        p: [-2.0, -0.45],
        ..START
    },
];

fn setting_of(t: &Trial) -> Setting {
    Setting { p: t.p.into(), dx: t.dx.into(), u: t.u.into(), p1: t.p1, dx1: t.dx1, u1: t.u1, f: Map::of(t.f), g: Map::of(t.g) }
}

// ----- the scene -----------------------------------------------------------------------

pub struct ChainRule {
    glide: Glide,
    /// One variable (three number lines) or two (three planes), as asked; and how far the
    /// picture has gone from two to one (it fades from the one to the other, both drawn
    /// on the way).
    one: bool,
    mix: f32,
    show: Show,
    /// The sweeps: seconds into them while they play, and whether they wait for a glide
    /// to arrive before they start.
    film: Option<f32>,
    film_waits: bool,
    /// How much of each arrow is drawn (see `reveal_at`), following what the sweeps ask.
    reveal: plot::Glide,
    grids: Option<Grids>,
    /// The level lines in the first plane and in the middle one.
    levels: [Levels; 2],
}

impl ChainRule {
    fn new() -> ChainRule {
        let mut scene = ChainRule {
            glide: Glide::at(setting_of(&START)),
            one: START.one,
            mix: START.one as u8 as f32,
            show: START.show,
            film: None,
            film_waits: false,
            reveal: plot::Glide::default(),
            grids: None,
            levels: [Levels::default(), Levels::default()],
        };
        // (opening, everything is drawn: the sweeps play when asked for)
        scene.reveal.snap(&reveal_at(START.show, None));
        scene
    }

    /// Set up experiment `k`: glide to its setting, then play the sweeps.
    fn set_up(&mut self, k: usize) {
        let t = &TRIALS[k];
        self.glide.to = setting_of(t);
        self.one = t.one;
        self.show = t.show;
        self.play();
    }

    /// Play the sweeps from the start, once the picture has glided to where it is going.
    fn play(&mut self) {
        self.film = None;
        self.film_waits = true;
    }

    /// The maps as asked for (each glides to it).
    fn choose(&mut self, which: usize, kind: Kind) {
        let map = Map::of(kind);
        if which == 0 {
            self.glide.to.f = map;
        } else {
            self.glide.to.g = map;
        }
    }

    fn shown(&self) -> Setting {
        self.glide.shown
    }

    /// Everything the chain rule says, as drawn.
    fn at(&self) -> At {
        let s = self.shown();
        chain(&s.f, &s.g, s.p, s.dx, s.u)
    }

    fn at1(&self) -> At1 {
        let s = self.shown();
        chain1(s.p1, s.dx1, s.u1)
    }

    /// A hand has moved something: the sweeps stop, and everything shown is drawn whole.
    fn hand(&mut self) {
        self.film = None;
        self.film_waits = false;
    }

    /// Move everything on: `dt` seconds of the scene's time (the sweeps) and `real_dt` of
    /// the clock's (a glide answers the visitor, and goes on while the scene's time
    /// stands still).
    fn advance(&mut self, dt: f32, real_dt: f32) {
        self.glide.step(real_dt as f64);
        // one variable gives way to two (or back): out, then in
        let target = self.one as u8 as f32;
        if self.mix != target {
            let step = real_dt.max(0.0) / MIX_SECONDS;
            self.mix = if self.mix < target { (self.mix + step).min(target) } else { (self.mix - step).max(target) };
        }
        if self.film_waits && self.glide.arrived() && self.settled() {
            (self.film_waits, self.film) = (false, Some(0.0));
        } else if let Some(t) = self.film {
            let t = t + dt.max(0.0);
            self.film = (t < film_length(self.show)).then_some(t);
        }
        let targets = if self.film_waits { [0.0; 6] } else { reveal_at(self.show, self.film) };
        self.reveal.follow(&targets, real_dt, false);
    }

    fn at_rest(&self) -> bool {
        self.glide.arrived() && self.film.is_none() && !self.film_waits && self.settled() && self.reveal.shown == reveal_at(self.show, None)
    }

    /// Whether the picture has finished fading from one variable to two, or back.
    fn settled(&self) -> bool {
        self.mix == self.one as u8 as f32
    }

    /// Whether the number lines are drawn (rather than the planes) where only one can be:
    /// the board, the panel's key.
    fn one_shown(&self) -> bool {
        self.mix >= 0.5
    }

    /// How much of arrow `k` is drawn (0, 1, 2 forwards in x, y, z; 3, 4, 5 backwards).
    fn grown(&self, k: usize) -> f64 {
        self.reveal.get(k).max(0.0).min(1.0) as f64
    }
}

// ----- where things go -----------------------------------------------------------------

/// A number line with one variable: where 0 is on the screen, the way the values grow
/// along it and the side the arrows are drawn on (unit vectors on the screen), pixels to
/// a unit, and the values it runs over.
#[derive(Clone, Copy, Debug)]
struct Line {
    zero: Vec2,
    along: Vec2,
    side: Vec2,
    scale: f32,
    range: [f64; 2],
}

impl Line {
    fn px(&self, v: f64) -> Vec2 {
        self.zero + self.along * (v as f32 * self.scale)
    }

    fn value(&self, q: Vec2) -> f64 {
        ((q - self.zero).dot(self.along) / self.scale.max(1e-6)) as f64
    }
}

/// The values the three number lines run over: x within ±`P1_MAX`, and what f and g make
/// of that.
fn ranges1() -> [[f64; 2]; 3] {
    let m = P1_MAX;
    [[-m, m], [f1(-m), f1(m)], [g1(f1(-m)), g1(f1(m))]]
}

/// The words on the buttons on the board.
const BUTTONS: [&str; 4] = ["forwards", "backwards", "both", "▶ sweeps"];

/// The widest a number in a matrix on the board can be, in characters (a minus sign
/// counted as two: in a formula it is nearly twice as wide as a digit).
const CELL: usize = 7;

/// The screen panel beside or under the picture: its rectangle, the sizes of its numbers
/// and words, where the matrices go (or the slopes, with one variable), the two sweeps
/// worked out in a column each, the line on what L really does, and the buttons.
#[derive(Clone, Debug)]
struct Board {
    rect: Rect,
    text: f32,
    small: f32,
    line_h: f32,
    /// The short form, for a phone: each sweep's result on a line, without the steps.
    short: bool,
    mats: Vec2,
    /// Where the forwards column starts, and the backwards one (in the short form, the
    /// two lines).
    cols: [Vec2; 2],
    truth: Vec2,
    buttons: Vec<Rect>,
}

impl Board {
    /// The board with its top-left corner at `min`, `width` across (`one`: for one
    /// variable, with the three slopes where the matrices go).
    fn at(min: Vec2, width: f32, short: bool, one: bool) -> Board {
        let pad = if short { 10.0 } else { 14.0 };
        let small = if short { 11.5 } else { 13.0 };
        let line_h = if short { 16.0 } else { 19.0 };
        // the matrices' numbers as large as lets the three fit across, up to 14
        let across = |s: f32| 3.0 * matrix_size(2, 2, CELL, s).x + 30.0;
        let mut text = 14.0f32;
        while across(text) > width - 2.0 * pad && text > 8.0 {
            text -= 0.5;
        }
        // (with one variable, the short form has a line of slopes where the matrices go)
        let mat_h = if one && short { 4.0 } else { matrix_size(2, 2, CELL, text).y };
        let x = min.x + pad;
        let mats = Vec2::new(x, min.y + pad + small + 4.0);
        // the columns beside the matrices where there is room, else under them
        let column_w = 290.0;
        // (under each matrix, how far it can stretch a step)
        let stretch_h = if short { 0.0 } else { line_h };
        let left = if one { 340.0 } else { across(text) };
        let beside = !short && width >= left + 24.0 + 2.0 * column_w + pad;
        let (cols, below) = if short {
            let y = mats.y + mat_h + 8.0;
            ([Vec2::new(x, y), Vec2::new(x, y + line_h)], y + 2.0 * line_h)
        } else if beside {
            let cx = x + left + 24.0;
            let y = min.y + pad;
            ([Vec2::new(cx, y), Vec2::new(cx + column_w, y)], (y + 5.0 * line_h).max(mats.y + mat_h + stretch_h) + 8.0)
        } else {
            let y = mats.y + mat_h + stretch_h + 10.0;
            let w = ((width - 2.0 * pad) * 0.5).max(10.0);
            ([Vec2::new(x, y), Vec2::new(x + w, y)], y + 5.0 * line_h + 8.0)
        };
        let truth = Vec2::new(x, below);
        let by = truth.y + line_h + if short { 6.0 } else { 10.0 };
        let bh = if short { 26.0 } else { 28.0 };
        let right = (x + 560.0).min(min.x + width - pad);
        let buttons = plot::even_row(BUTTONS.len(), x, right, by, bh, 8.0);
        Board { rect: (min, Vec2::new(min.x + width, by + bh + pad)), text, small, line_h, short, mats, cols, truth, buttons }
    }

    fn height(&self) -> f32 {
        self.rect.1.y - self.rect.0.y
    }
}

/// How the three planes are laid out.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Arrange {
    /// Side by side, the board under them: a wide canvas.
    Row,
    /// x and y side by side, z under y and the board beside z: a squarer canvas, where
    /// this makes the planes larger.
    Ell,
    /// One under another, the board under them: a phone held upright.
    Column,
}

#[derive(Clone, Debug)]
struct Layout {
    arrange: Arrange,
    planes: [Rect; 3],
    /// Where each plane's name goes (its top-left corner), inside the plane on a phone.
    titles: [Vec2; 3],
    /// The arrows from plane to plane, for f and then for g.
    arrows: [(Vec2, Vec2); 2],
    board: Option<Board>,
    /// One variable: the number lines, where their names go, and the board under them.
    lines: [Line; 3],
    line_names: [Vec2; 3],
    board1: Option<Board>,
    /// The size of the names on the picture.
    text: f32,
    /// Whether there is room for the caption along the foot of the picture, with two
    /// variables and with one; and, in an L, where the key goes instead.
    caption: bool,
    caption1: bool,
    key: Option<Vec2>,
}

/// The key's height in an L, under the board.
const KEY_H: f32 = 6.0 * 18.0;

impl Layout {
    fn narrow(&self) -> bool {
        self.arrange == Arrange::Column
    }

    fn of(view: Vec2, inset: f32) -> Layout {
        let top = inset + 10.0;
        if view.x < 620.0 && view.y >= 520.0 {
            return Self::column(view, top);
        }
        let pad = 16.0;
        let title_h = 24.0;
        // (its words fit on two lines from about this width on)
        let caption = view.y >= 300.0 && view.x >= 760.0;
        let caption_h = if caption { 46.0 } else { 8.0 };
        // side by side, with the board under them if there is room for it: the whole
        // board, or else its short form
        let gap = 46.0;
        let across = (view.x - 2.0 * pad - 2.0 * gap) / 3.0;
        let width = 3.0 * across + 2.0 * gap;
        let room = view.y - top - title_h - caption_h;
        let fits = |short: bool, least: f32| {
            let h = Board::at(Vec2::ZERO, width, short, false).height();
            ((room - h - 14.0).min(across) >= least).then_some((short, h))
        };
        let row_board = fits(false, 110.0).or_else(|| fits(true, 90.0));
        let row_side = across.min(row_board.map_or(room, |(_, h)| room - h - 14.0)).max(10.0);

        // or in an L, with the board in the corner left over, if that makes them larger
        let (gap_x, gap_y) = (46.0, 40.0);
        // (the key goes in that corner too, under the board, and nothing along the foot)
        let ell_side = ((view.x - 2.0 * pad - gap_x) * 0.5).min((view.y - top - title_h - gap_y - 8.0) * 0.5).max(10.0);
        let ell_board = Board::at(Vec2::ZERO, ell_side, false, false);
        let ell = ell_side > 1.1 * row_side && ell_board.height() + 14.0 + KEY_H <= ell_side + title_h && ell_side >= 200.0;

        let (arrange, planes, titles, arrows, board, key) = if ell {
            let left = (view.x - 2.0 * ell_side - gap_x) * 0.5;
            let right = left + ell_side + gap_x;
            let y0 = top + title_h;
            let y1 = y0 + ell_side + gap_y;
            let planes = [(Vec2::new(left, y0), Vec2::new(left + ell_side, y0 + ell_side)), (Vec2::new(right, y0), Vec2::new(right + ell_side, y0 + ell_side)), (Vec2::new(right, y1), Vec2::new(right + ell_side, y1 + ell_side))];
            let titles = planes.map(|r| r.0 + Vec2::new(0.0, -title_h + 2.0));
            let mid = y0 + ell_side * 0.5;
            let down = right + ell_side - 36.0;
            let arrows = [(Vec2::new(left + ell_side + 8.0, mid), Vec2::new(right - 8.0, mid)), (Vec2::new(down, y0 + ell_side + 4.0), Vec2::new(down, y1 - 4.0))];
            let board = Board::at(Vec2::new(left, y1 - title_h), ell_side, false, false);
            let key = board.rect.1 * Vec2::new(0.0, 1.0) + Vec2::new(left + 14.0, 14.0);
            (Arrange::Ell, planes, titles, arrows, Some(board), Some(key))
        } else {
            let side = row_side;
            let left = (view.x - 3.0 * side - 2.0 * gap) * 0.5;
            let y0 = top + title_h;
            let planes = [0, 1, 2].map(|k| {
                let x = left + k as f32 * (side + gap);
                (Vec2::new(x, y0), Vec2::new(x + side, y0 + side))
            });
            let titles = planes.map(|r| r.0 + Vec2::new(0.0, -title_h + 2.0));
            let mid = y0 + side * 0.5;
            let arrows = [0, 1].map(|k| (Vec2::new(planes[k].1.x + 8.0, mid), Vec2::new(planes[k + 1].0.x - 8.0, mid)));
            let board = row_board.map(|(short, _)| Board::at(Vec2::new(left, y0 + side + 14.0), 3.0 * side + 2.0 * gap, short, false));
            (Arrange::Row, planes, titles, arrows, board, None)
        };

        // one variable: the board across the foot, the lines above it
        let caption1_h = if caption { 46.0 } else { 8.0 };
        let board1 = [(false, 300.0), (true, 220.0)].into_iter().find_map(|(short, least)| {
            let h = Board::at(Vec2::ZERO, view.x - 2.0 * pad, short, true).height();
            (view.y - top - caption1_h - h >= least).then(|| Board::at(Vec2::new(pad, view.y - caption1_h - h), view.x - 2.0 * pad, short, true))
        });
        let picture_bottom = board1.as_ref().map_or(view.y - caption1_h, |b| b.rect.0.y - 6.0);
        let lines = Self::lines(false, (Vec2::new(0.0, top), Vec2::new(view.x, picture_bottom)));
        let line_names = lines.map(|l| l.px(l.range[0] - 0.1) + Vec2::new(-14.0, 0.0));
        Layout { arrange, planes, titles, arrows, board, key, lines, line_names, board1, text: 14.0, caption: caption && !ell, caption1: caption }
    }

    /// A phone held upright: the planes one under another (and the number lines side by
    /// side), the board under them.
    fn column(view: Vec2, top: f32) -> Layout {
        let pad = 10.0;
        let gap = 18.0;
        let board_h = Board::at(Vec2::ZERO, view.x - 2.0 * pad, true, false).height();
        let room = view.y - top - 8.0 - 2.0 * gap;
        let with_board = room - board_h - 10.0 >= 3.0 * 90.0;
        let h = (if with_board { room - board_h - 10.0 } else { room } / 3.0).max(10.0);
        let planes = [0, 1, 2].map(|k| {
            let y = top + k as f32 * (h + gap);
            (Vec2::new(pad, y), Vec2::new(view.x - pad, y + h))
        });
        let titles = planes.map(|r| r.0 + Vec2::new(8.0, 6.0));
        let x = view.x - pad - 40.0;
        let arrows = [0, 1].map(|k| (Vec2::new(x, planes[k].1.y + 2.0), Vec2::new(x, planes[k + 1].0.y - 2.0)));
        let bottom = planes[2].1.y;
        let board = with_board.then(|| Board::at(Vec2::new(pad, bottom + 10.0), view.x - 2.0 * pad, true, false));
        let lines = Self::lines(true, (Vec2::new(0.0, top), Vec2::new(view.x, bottom)));
        let line_names = lines.map(|l| l.px(l.range[1] + 0.1) + Vec2::new(0.0, -8.0));
        Layout { arrange: Arrange::Column, planes, titles, arrows, board: board.clone(), key: None, lines, line_names, board1: board, text: 12.5, caption: false, caption1: false }
    }

    /// The three number lines in the rectangle `area`: across it, one under another, on a
    /// wide canvas; up it, side by side, on a phone. All at the same scale, so that the
    /// lengths of a step and of its images can be compared by eye, with their 0s in line.
    fn lines(narrow: bool, (min, max): Rect) -> [Line; 3] {
        let ranges = ranges1();
        let widest = ranges.iter().map(|r| r[1] - r[0]).fold(0.0f64, f64::max) as f32 * 1.08;
        let size = max - min;
        if narrow {
            let length = (size.y - 60.0).max(10.0);
            let scale = length / widest;
            let mid = min.y + 34.0 + length * 0.5;
            let xs = [0.2, 0.5, 0.8].map(|s| min.x + s * size.x);
            [0, 1, 2].map(|k| Line { zero: Vec2::new(xs[k], mid), along: Vec2::new(0.0, -1.0), side: Vec2::new(-1.0, 0.0), scale, range: ranges[k] })
        } else {
            let length = (size.x - 220.0).max(10.0);
            let scale = length / widest;
            let mid = min.x + 140.0 + length * 0.5;
            let ys = [0.18, 0.5, 0.82].map(|s| min.y + 24.0 + (size.y - 48.0) * s);
            [0, 1, 2].map(|k| Line { zero: Vec2::new(mid, ys[k]), along: Vec2::X, side: Vec2::NEG_Y, scale, range: ranges[k] })
        }
    }
}

// ----- drawing ---------------------------------------------------------------------------

/// The kinds of line drawn, each in a batch of its own (see `plot::apart`): those whose
/// number of points changes from frame to frame are apart from those whose does not, so
/// that `eidos motion` can follow the rest point by point.
const FRAMES: u8 = plot::AXES;
const GRID: u8 = plot::CURVES;
/// (Each plane's level lines, its line through the point and its yellow curve have a
/// batch of their own: a batch that holds two of them could keep its number of points
/// while one gains what the other loses, and be compared point for point with a different
/// line. And the number lines have batches of their own, so that what is drawn with one
/// variable is never compared with what was drawn with two.)
const LEVELS: u8 = plot::OWN;
const THROUGH: u8 = plot::OWN + 3;
const TRUE_PATH: u8 = plot::OWN + 6;
const ARROWS: u8 = plot::OWN + 9;
const FAN: u8 = plot::OWN + 10;
const FRAMES1: u8 = plot::OWN + 11;
const ARROWS1: u8 = plot::OWN + 12;
const TRUE1: u8 = plot::OWN + 13;

/// Every kind of line this scene draws, asked for in the same order at the start of every
/// frame, each with one point that cannot be seen: a batch left out of a frame would move
/// every later one up a place, and `eidos motion` would take one for another.
fn hold_batches(f: &mut Frame) {
    let corner = f.hud(Vec2::ZERO);
    for kind in [FRAMES, GRID, plot::BUTTONS].into_iter().chain(LEVELS..=TRUE1) {
        plot::apart(f, kind, false).segment(corner, corner, 1.0, Color::CLEAR);
    }
    plot::disc(f, Vec2::ZERO, 1.0, Color::CLEAR);
}

/// A colour with its opacity scaled by `k`.
fn faded(c: Color, k: f32) -> Color {
    c.a(c.0[3] as f32 / 255.0 * k.max(0.0).min(1.0))
}

/// An arrow on the screen from `a` to `b`, kept inside `rect`: its shaft in the batch of
/// arrows, its head a small triangle (left off if the tip is outside).
fn arrow(f: &mut Frame, kind: u8, a: Vec2, b: Vec2, rect: Rect, width: f32, c: Color) {
    let len = (b - a).length();
    if !(len.is_finite() && a.is_finite()) {
        return;
    }
    let head = (width * 3.0 + 5.0).min(len * 0.55);
    let dir = (b - a).normalize_or_zero();
    let shaft_end = b - dir * (head * 0.8);
    let pts = plot::clip(&[a, shaft_end], rect);
    plot::path_of(f, kind, &pts, width, c);
    if len > 2.0 && plot::within(b, rect, 0.5) {
        let side = dir.perp() * (head * 0.42);
        let base = b - dir * head;
        let [p, q, r] = [b, base + side, base - side].map(|v| f.hud_at(2, v));
        f.dl.tris_alpha.tri(p, q, r, c);
    }
}

/// A place on the screen kept a few pixels inside a rectangle: where a handle whose
/// point lies beyond a plane's edge is drawn (at the edge, where its arrow is cut).
fn inside(q: Vec2, (min, max): Rect) -> Vec2 {
    let m = Vec2::splat(4.0);
    q.max(min + m).min(max - m)
}

/// A point of the scene (z = 0) that shows at a place on the screen, for the label placer.
fn world(f: &Frame, q: Vec2) -> Option<Vec3> {
    f.cam.pick_ground(q * f.px)
}

/// How many decimals the numbers of a matrix or vector on the board get: two, or three
/// where even the largest of them is small (all of them alike, so that they line up).
fn decimals(largest: f64) -> usize {
    if largest < 0.0995 {
        3
    } else {
        2
    }
}

/// A matrix's numbers for the board, row by row, as formulas.
fn cells(m: DMat2, color: Color) -> Vec<Vec<(String, Color)>> {
    let d = decimals(m.to_cols_array().iter().fold(0.0f64, |a, v| a.max(v.abs())));
    (0..2).map(|i| (0..2).map(|j| (format!("${}$", num(m.col(j)[i], d)), color)).collect()).collect()
}

/// A mark's number on a number line: as few digits as say it, and a true minus sign.
fn tick(v: f64) -> String {
    let t = num(v, 2);
    if t.contains('.') {
        t.trim_end_matches('0').trim_end_matches('.').to_string()
    } else {
        t
    }
}

/// A vector for the board, as markup without the dollars: (0.25, −1.30).
fn pair(v: DVec2) -> String {
    let d = decimals(v.abs().max_element());
    format!("({}, {})", num(v.x, d), num(v.y, d))
}

impl Scene for ChainRule {
    fn view(&self) -> View {
        View::Flat { center: Vec2::ZERO, half_height: 1.0 }
    }

    fn settings(&self, s: &mut ViewSettings) {
        s.grid = false;
        s.bloom = s.bloom.min(0.25);
    }

    fn animated(&self) -> bool {
        !self.at_rest()
    }

    fn pointer_hint(&self) -> Option<&'static str> {
        Some("drag the white dot $p$, the cyan tip of $\\mathrm{d}x$ and the pink tip of $u$    buttons under the picture: click")
    }

    fn ui(&mut self, ui: &mut egui::Ui, _: &Gfx) {
        ui::note(
            ui,
            r"Do one map and then another: $z = g(f(x))$. Near a point $p$, a small step $\mathrm{d}x$ becomes about $J_f\,\mathrm{d}x$ at $f(p)$, where $J_f$ is the Jacobian of $f$, its matrix of slopes; and that becomes about $J_g J_f\,\mathrm{d}x$ at $g(f(p))$. So the Jacobian of the whole is the product of the Jacobians, each taken at the point the one before delivers:",
        );
        ui::equation(ui, r"J_{g \circ f}(p) = J_g\bigl(f(p)\bigr)\,J_f(p)");
        ui::note(ui, r"Read the other way, the same product carries the gradient $u$ of a final number $L$ back from $z$ to $x$, through the transposes $J_g^\top$ and then $J_f^\top$. That backward sweep is all that backpropagation is.");

        ui::section(ui, "Things to try");
        for (k, trial) in TRIALS.iter().enumerate() {
            if ui::experiment(ui, trial.label, trial.look_for) {
                self.set_up(k);
            }
        }

        ui::section(ui, "Variables");
        let mut one = self.one as usize;
        if ui::choice(ui, &["two: three planes", "one: three number lines"], &mut one) {
            self.one = one == 1;
        }
        ui::note(ui, r"With one variable, $f(x) = x^3$ and $g(y) = \arctan y$, and each Jacobian is a single slope.");

        ui::section(ui, "Sweeps");
        let mut k = SHOWS.iter().position(|s| *s == self.show).unwrap_or(2);
        if ui::choice(ui, &["forwards", "backwards", "both"], &mut k) {
            self.show = SHOWS[k];
            self.hand();
        }
        if ui.button("▶ play the sweeps").clicked() {
            self.play();
        }
        ui::note(ui, r"Forwards: a small step $\mathrm{d}x$ at $p$, carried to the end. Backwards: the gradient of $L$ at the end, carried back to the start. Playing draws them a plane at a time, in the order they are worked out.");

        ui::section(ui, "The maps");
        for (which, (map, letter)) in [("f", "x"), ("g", "y")].into_iter().enumerate() {
            let asked = if which == 0 { self.glide.to.f } else { self.glide.to.g };
            let mut k = KINDS.iter().position(|k| Some(*k) == asked.kind()).unwrap_or(0);
            ui::note(ui, &format!("${map}$:"));
            let names: Vec<&str> = KINDS.iter().map(|k| k.name()).collect();
            if ui::choice(ui, &names, &mut k) {
                self.choose(which, KINDS[k]);
            }
            ui::note(ui, &KINDS[k].formula(map, letter));
        }

        ui::section(ui, "Reading the picture");
        if self.one {
            ui::key(
                ui,
                &[
                    (pal::WHITE, "white: $p$ (drag it), and $f(p)$ and $g(f(p))$ on the lines below it"),
                    (FORWARD, r"cyan: a small step $\mathrm{d}x$ (drag its tip), then $f'(p)$ times it, then $g'(f(p))$ times that"),
                    (TRUE, "yellow: where the step really goes, from line to line"),
                    (BACKWARD, r"pink: $u$, the slope of $L = uz$ (drag its tip), then $g'(f(p))$ times it, then $f'(p)$ times that"),
                    (GRID_COLOUR, "grey: where $f$ carries the marks of the line $x$, and where $g$ carries those"),
                ],
            );
        } else {
            ui::key(
                ui,
                &[
                    (pal::WHITE, "white: the point $p$ (drag it), and $f(p)$ and $g(f(p))$, where the maps carry it"),
                    (FORWARD, r"cyan: a small step $\mathrm{d}x$ at $p$ (drag its tip), and $J_f\,\mathrm{d}x$ and $J_g J_f\,\mathrm{d}x$, where the Jacobians carry it"),
                    (TRUE, r"yellow: where the step really goes: $f(p + s\,\mathrm{d}x)$, then $g(f(p + s\,\mathrm{d}x))$, for $s$ from 0 to 1"),
                    (BACKWARD, r"pink: $u$, the gradient of $L = u \cdot z$ (drag its tip); $J_g^\top u$ and $J_f^\top J_g^\top u$, the gradients of $L$ as it depends on $y$ and on $x$"),
                    (LOSS, "orange: level lines of $L$, at the same values in every plane; bright: the one through the point"),
                    (GRID_COLOUR, "grey: a square grid in the first plane, and where $f$ and then $g$ carry it"),
                ],
            );
        }

        ui::section(ui, "Words");
        ui::term(ui, "Jacobian:", r"$J_f(p)$, the matrix of the slopes of $f$ at $p$: row $i$, column $j$ holds $\partial f_i / \partial x_j$. Near $p$, $f(p + \mathrm{d}x) \approx f(p) + J_f(p)\,\mathrm{d}x$. With one variable it is the slope $f'(p)$.");
        ui::term(ui, "Chain rule:", r"the Jacobian of $g \circ f$ (first $f$, then $g$) is $J_g(f(p))\,J_f(p)$; with one variable, $(g \circ f)'(p) = g'(f(p))\,f'(p)$.");
        ui::term(ui, "Gradient:", r"$\nabla L$, the slopes of a number $L$ along each axis. It points the way $L$ rises fastest, at right angles to the level lines, and a small step $\mathrm{d}v$ changes $L$ by about $\nabla L \cdot \mathrm{d}v$.");
        ui::term(ui, "Level line:", "the points where $L$ takes one value.");
        ui::term(ui, "Transpose:", r"$J^\top$, the matrix $J$ with its rows written as columns. For any vectors, $u \cdot (Jv) = (J^\top u) \cdot v$: that is why the gradient of $L(g(y))$ is $J_g^\top u$, and why the backward sweep multiplies by transposes.");
        ui::term(
            ui,
            "Forwards, backwards:",
            r"forwards (forward mode) pushes one step through and finds $J\,\mathrm{d}x$: with $\mathrm{d}x$ along an axis, one column of $J$. Backwards (reverse mode) pulls one gradient back and finds $J^\top u$: with $u$ along an axis, one row of $J$. For a function of $n$ inputs and one output, backwards finds all $n$ slopes in one sweep, where forwards needs $n$ sweeps, one for each input. That is why the gradient of a network, which may have millions of weights and has one loss, is worked out backwards.",
        );
        ui::term(ui, "Stretch:", r"$\|J\|$, the most $J$ lengthens any step (its largest singular value). $\|J_g J_f\| \le \|J_g\|\,\|J_f\|$: a product is no larger than its factors allow, so one flat link flattens the whole chain.");
        ui::term(ui, "Saturated:", r"a tanh unit whose input is far from 0: its output is near $1$ or $-1$, and its slope, $1 - \tanh^2$, is near 0.");

        ui::section(ui, "Numbers");
        if self.one {
            let a = self.at1();
            ui::readout(ui, "$p$", format!("${}$", num(a.p, 2)));
            ui::readout(ui, "$f'(p)$, $g'(f(p))$", format!("${}$, ${}$", num(a.df, 3), num(a.dg, 3)));
            ui::readout(ui, r"$(g \circ f)'(p)$", format!("${}$", num(a.df * a.dg, 3)));
            ui::readout_colored(ui, "forwards", format!("${}$", num(a.u * a.dz, 4)), FORWARD);
            ui::readout_colored(ui, "backwards", format!("${}$", num(a.gx * a.dx, 4)), BACKWARD);
            ui::readout(ui, "the true change of $L$", format!("${}$", num(a.u * (g1(f1(a.p + a.dx)) - a.z), 4)));
        } else {
            let at = self.at();
            let s = self.shown();
            ui::readout(ui, "$p$", format!("${}$", tuple(&[at.p.x, at.p.y], 2)));
            ui::readout(ui, "$f(p)$", format!("${}$", tuple(&[at.y.x, at.y.y], 2)));
            ui::readout(ui, "$g(f(p))$", format!("${}$", tuple(&[at.z.x, at.z.y], 2)));
            ui::readout_colored(ui, r"forwards, $u \cdot J\,\mathrm{d}x$", format!("${}$", num(at.forwards(), 4)), FORWARD);
            ui::readout_colored(ui, r"backwards, $J^\top u \cdot \mathrm{d}x$", format!("${}$", num(at.backwards(), 4)), BACKWARD);
            ui::readout(ui, "the true change of $L$", format!("${}$", num(true_change(&s.f, &s.g, s.p, s.dx, s.u), 4)));
            ui::readout(ui, r"$\|J_f\|$, $\|J_g\|$, $\|J\|$", format!("${}$, ${}$, ${}$", num(stretch(at.jf), 3), num(stretch(at.jg), 3), num(stretch(at.j), 3)));
        }
        ui::see_also(ui, &[("jacobian", "one map up close: the Jacobian"), ("compose", "matrix products: linear maps one after another"), ("autodiff", "the backward sweep on any formula"), ("backprop1", "the backward sweep in the smallest network")]);
    }

    fn frame(&mut self, f: &mut Frame) {
        hold_batches(f);
        let view = f.view_size();
        let lay = Layout::of(view, f.top_inset);
        // (a canvas squeezed to a sliver is one the controls have opened over, on a phone)
        let tiny = view.x < 160.0 || view.y < 160.0;
        if !tiny {
            self.board_input(f, &lay);
        }
        self.advance(f.dt, f.real_dt);
        if tiny {
            return;
        }
        // (while one gives way to the other, both are drawn, the one fading out as the
        // other fades in)
        if self.mix < 1.0 {
            self.picture2(f, &lay, smooth(1.0 - self.mix));
        }
        if self.mix > 0.0 {
            self.picture1(f, &lay, smooth(self.mix));
        }
        if let Some(b) = self.board_of(&lay) {
            self.board(f, b, smooth((2.0 * self.mix - 1.0).abs()));
        }
    }
}

impl ChainRule {
    /// The board for what is drawn: under the planes, or under the number lines.
    fn board_of<'a>(&self, lay: &'a Layout) -> Option<&'a Board> {
        if self.one_shown() {
            lay.board1.as_ref()
        } else {
            lay.board.as_ref()
        }
    }

    /// The board's buttons (their clicks taken before the scene moves on).
    fn board_input(&mut self, f: &mut Frame, lay: &Layout) {
        let Some(b) = self.board_of(lay) else { return };
        if plot::pointer(f).is_some_and(|p| plot::within(p, b.rect, 0.0)) {
            f.handles.capture();
            f.handles.claim_clicks();
        }
        for (k, rect) in b.buttons.iter().enumerate() {
            let on = k < 3 && SHOWS[k] == self.show;
            if plot::button(f, *rect, BUTTONS[k], b.small, on) {
                if k < 3 {
                    self.show = SHOWS[k];
                    self.hand();
                } else {
                    self.play();
                }
            }
        }
    }

    /// The grid and the windows of the planes, for the maps as drawn (kept while the
    /// maps stay the same), and the planes as they are drawn on this canvas.
    fn planes(&mut self, lay: &Layout) -> [Plane; 3] {
        let s = self.shown();
        let mut of = [0u64; 8];
        of[..4].copy_from_slice(&s.f.bits());
        of[4..].copy_from_slice(&s.g.bits());
        if self.grids.as_ref().is_none_or(|g| g.of != of) {
            self.grids = Some(Grids::of(&s.f, &s.g));
        }
        let windows = self.grids.as_ref().map_or([(DVec2::ZERO, 1.0); 3], |g| g.windows);
        [0, 1, 2].map(|k| Plane::fit(lay.planes[k], windows[k]))
    }

    /// Two variables: the three planes.
    fn picture2(&mut self, f: &mut Frame, lay: &Layout, alpha: f32) {
        let planes = self.planes(lay);
        let size = lay.text;

        // the handles: the step's tip and the point in the first plane, the gradient's
        // tip in the last (the step's first: where it lies on p, a press pulls it out)
        let s = self.shown();
        // (a tip can be taken hold of once its arrow is drawn; and nothing while the
        // picture fades from one variable to two)
        let still = self.settled() && !self.one;
        if self.show.forwards() && self.grown(0) > 0.5 && still {
            if let Some(q) = plot::screen_handle(f, 1, inside(planes[0].px(s.p + s.dx), planes[0].rect), FORWARD) {
                let q = q.max(planes[0].rect.0).min(planes[0].rect.1);
                let dx = (planes[0].unit(q) - s.p).clamp_length_max(DX_MAX);
                self.glide.now(|t| t.dx = dx);
                self.hand();
            }
        }
        if let Some(q) = still.then(|| plot::screen_handle(f, 2, planes[0].px(s.p), pal::WHITE)).flatten() {
            let p = planes[0].unit(q).clamp(DVec2::splat(-GRID_HALF), DVec2::splat(GRID_HALF));
            self.glide.now(|t| t.p = p);
            self.hand();
        }
        if self.show.backwards() && self.grown(5) > 0.5 && still {
            let z = s.g.at(s.f.at(s.p));
            if let Some(q) = plot::screen_handle(f, 3, inside(planes[2].px(z + s.u), planes[2].rect), BACKWARD) {
                let q = q.max(planes[2].rect.0).min(planes[2].rect.1);
                let u = (planes[2].unit(q) - z).clamp_length_max(U_MAX);
                self.glide.now(|t| t.u = u);
                self.hand();
            }
        }
        let s = self.shown();
        let at = self.at();
        let (map_f, map_g, u) = (s.f, s.g, s.u);
        let grown: [f64; 6] = [0, 1, 2, 3, 4, 5].map(|k| self.grown(k));

        // each plane: its backing, frame and axes, and the grid as the maps carry it
        for (k, pl) in planes.iter().enumerate() {
            plot::fill(f, pl.rect, faded(Color::hex(0x0C1222), 0.9 * alpha));
            let (min, max) = pl.rect;
            let edge = [min, Vec2::new(max.x, min.y), max, Vec2::new(min.x, max.y), min];
            plot::path_of(f, FRAMES, &edge, 1.0, faded(pal::DIM, 0.5 * alpha));
            // (the axes through 0, drawn at the edge and unseen when 0 is out of sight,
            // so that the batch keeps its points)
            let o = pl.px(DVec2::ZERO);
            let (ox, oy) = (o.x.max(min.x).min(max.x), o.y.max(min.y).min(max.y));
            let seen = |inside: bool| if inside { faded(pal::DIM, 0.35 * alpha) } else { Color::CLEAR };
            plot::line_of(f, FRAMES, Vec2::new(min.x, oy), Vec2::new(max.x, oy), 1.0, seen(o.y > min.y && o.y < max.y));
            plot::line_of(f, FRAMES, Vec2::new(ox, min.y), Vec2::new(ox, max.y), 1.0, seen(o.x > min.x && o.x < max.x));
            if let Some(grids) = &self.grids {
                let c = faded(GRID_COLOUR, 0.45 * alpha);
                for line in &grids.lines[k] {
                    let pts: Vec<Vec3> = line.iter().map(|v| f.hud(pl.px(*v))).collect();
                    plot::apart(f, GRID, false).polyline(pts, 1.0, c);
                }
            }
        }

        // the level lines of L: in the last plane straight, at right angles to u; in the
        // others, where L = u · g(f(x)) and u · g(y) take the same values
        let half_z = self.grids.as_ref().map_or(1.0, |g| g.windows[2].1);
        let step = level_step(u, half_z);
        let flat_l = u.length().is_nan() || u.length() <= 1e-9;
        let family = |k: usize| faded(LOSS, 0.3 * alpha * grown[3 + k] as f32);
        let bright = |k: usize| faded(LOSS, 0.95 * alpha * grown[3 + k] as f32);
        let mut busy: [Vec<(Vec2, Vec2)>; 3] = Default::default();
        if !flat_l {
            // the last plane
            let pl = planes[2];
            let (lo, hi) = pl.shown();
            let reach = (hi - lo).length();
            let along = u.perp().normalize();
            let line = |c: f64| {
                let foot = u * (c / u.length_squared());
                plot::clip(&[pl.px(foot - along * reach), pl.px(foot + along * reach)], pl.rect)
            };
            // (the same list of lines whichever way u points, as many as could reach the
            // plane's corners, each with the same number of points: those beyond the edge
            // are kept to the edge and unseen, so that every point moves smoothly as the
            // lines turn, into sight and out of it)
            let corners = [lo, hi, DVec2::new(lo.x, hi.y), DVec2::new(hi.x, lo.y)];
            let apart = step / u.length();
            let most = corners.iter().fold(0.0f64, |m, v| m.max(v.length()));
            let count = ((most / apart).ceil() as i64 + 1).min(60);
            let centre = (lo + hi) * 0.5;
            let half = (hi - lo).length() * 0.5;
            for i in -count..=count {
                let c = i as f64 * step;
                let foot = centre - u * ((u.dot(centre) - c) / u.length_squared());
                let pts: Vec<(Vec3, Color)> = (0..STRAIGHT_POINTS)
                    .map(|n| {
                        let q = pl.px(foot + along * (half * (2.0 * n as f64 / (STRAIGHT_POINTS - 1) as f64 - 1.0)));
                        (f.hud(q.max(pl.rect.0).min(pl.rect.1)), if pl.contains(q) { family(2) } else { Color::CLEAR })
                    })
                    .collect();
                let batch = plot::apart(f, LEVELS + 2, false);
                batch.brk();
                for (q, c) in pts {
                    batch.push(q, 1.0, c);
                }
                batch.brk();
            }
            let through = line(u.dot(at.z));
            plot::path_of(f, THROUGH + 2, &through, 1.8, bright(2));
            busy[2].extend(through.windows(2).filter(|w| w[0].is_finite() && w[1].is_finite()).map(|w| (w[0], w[1])));
            // the middle plane and the first
            let mut key: Vec<u64> = map_f.bits().into_iter().chain(map_g.bits()).collect();
            key.extend([u.x.to_bits(), u.y.to_bits()]);
            for k in [1, 0] {
                let pl = planes[k];
                let levels = &mut self.levels[k];
                if k == 1 {
                    levels.update(&pl, &key[4..8], |v| map_g.at(v), u, step);
                } else {
                    levels.update(&pl, &key[..8], |v| map_g.at(map_f.at(v)), u, step);
                }
                let c = family(k);
                for line in &levels.lines {
                    // (the part beyond the edge is kept to the edge and unseen, rather than
                    // cut off: so that each line keeps its number of points as it moves)
                    let pts: Vec<(Vec3, Color)> = line
                        .iter()
                        .map(|v| {
                            let q = pl.px(*v);
                            let seen = pl.contains(q);
                            (f.hud(q.max(pl.rect.0).min(pl.rect.1)), if seen { c } else { Color::CLEAR })
                        })
                        .collect();
                    let batch = plot::apart(f, LEVELS + k as u8, false);
                    batch.brk();
                    for (q, c) in pts {
                        batch.push(q, 1.0, c);
                    }
                    batch.brk();
                }
                let line = if k == 1 {
                    levels.through(&pl, &key, at.y, |v| (u.dot(map_g.at(v)), map_g.jacobian(v).transpose() * u))
                } else {
                    levels.through(&pl, &key, at.p, |v| {
                        let y = map_f.at(v);
                        (u.dot(map_g.at(y)), map_f.jacobian(v).transpose() * (map_g.jacobian(y).transpose() * u))
                    })
                };
                let pts = plot::clip(&line.iter().map(|v| pl.px(*v)).collect::<Vec<_>>(), pl.rect);
                plot::path_of(f, THROUGH + k as u8, &pts, 1.8, bright(k));
                busy[k].extend(pts.windows(2).step_by(3).filter(|w| w[0].is_finite() && w[1].is_finite()).map(|w| (w[0], w[1])));
            }
        }

        // where the step really goes (yellow), as far as the sweep has carried it
        for (k, pl) in planes.iter().enumerate().skip(1) {
            let r = grown[k];
            let pts: Vec<Vec2> = (0..=24)
                .map(|i| {
                    let x = at.p + at.dx * (r * i as f64 / 24.0);
                    let y = map_f.at(x);
                    pl.px(if k == 1 { y } else { map_g.at(y) })
                })
                .collect();
            let pts = plot::clip(&pts, pl.rect);
            plot::path_of(f, TRUE_PATH + k as u8, &pts, 5.0, faded(TRUE, 0.75 * alpha * (r as f32 * 4.0).min(1.0)));
            busy[k].extend(pts.windows(2).step_by(2).filter(|w| w[0].is_finite() && w[1].is_finite()).map(|w| (w[0], w[1])));
        }

        // the arrows: forwards, the step and its pushes; backwards, the gradient and its pulls
        let points = [at.p, at.y, at.z];
        let pushes = [at.dx, at.dy, at.dz];
        let pulls = [at.gx, at.gy, at.u];
        let mut tips = [[None; 2]; 3];
        for k in 0..3 {
            let pl = planes[k];
            let from = pl.px(points[k]);
            for (side, (v, r, c)) in [(pushes[k], grown[k], FORWARD), (pulls[k], grown[3 + k], BACKWARD)].into_iter().enumerate() {
                let tip = pl.px(points[k] + v * r);
                arrow(f, ARROWS, from, tip, pl.rect, 2.6, faded(c, alpha * (r as f32 * 4.0).min(1.0)));
                busy[k].push((from, tip));
                if r > 0.5 {
                    tips[k][side] = Some(tip);
                }
            }
            // (p is its handle, but for while the picture fades)
            if (k > 0 || !still) && pl.contains(from) {
                plot::disc(f, from, if k > 0 { 10.0 } else { 12.0 }, faded(pal::WHITE, alpha));
            }
        }

        // the planes' names, and the maps between them
        let names = ["$x$", "$y = f(x)$", "$z = g(y)$"];
        let asked = [None, self.glide.to.f.kind(), self.glide.to.g.kind()];
        for k in 0..3 {
            let at = lay.titles[k];
            let title = match asked[k] {
                Some(kind) if !lay.narrow() && lay.planes[k].1.x - lay.planes[k].0.x >= 260.0 => format!("{}, with {} the {}", names[k], ["", "$f$", "$g$"][k], kind.name()),
                _ => names[k].to_string(),
            };
            if lay.narrow() {
                f.caption_px(at, title, 12.5, faded(pal::TEXT, alpha), Align2::LEFT_TOP);
            } else {
                f.text_px(at, title, 14.0, faded(pal::TEXT, alpha), Align2::LEFT_TOP);
            }
        }
        for (k, (from, to)) in lay.arrows.into_iter().enumerate() {
            let name = ["$f$", "$g$"][k];
            hud_arrow(f, from, to, 1.4, faded(pal::DIM, alpha));
            let (at, align) = if (to - from).x.abs() > (to - from).y.abs() { (Vec2::new((from.x + to.x) * 0.5, from.y - 6.0), Align2::CENTER_BOTTOM) } else { (Vec2::new(from.x + 8.0, (from.y + to.y) * 0.5), Align2::LEFT_CENTER) };
            f.text_px(at, name, if lay.narrow() { 13.0 } else { 15.0 }, faded(pal::TEXT, alpha), align);
        }
        if flat_l {
            let pl = planes[2];
            f.caption_px(Vec2::new(pl.mid().x, pl.rect.1.y - 8.0), "$u = 0$: $L$ is the same everywhere", size - 1.5, faded(LOSS, alpha), Align2::CENTER_BOTTOM);
        }

        // the names beside what they name, each kept inside its plane
        let view = f.view_size();
        let short = lay.narrow();
        let fwd_names = ["$\\mathrm{d}x$", "$J_f\\,\\mathrm{d}x$", "$J_g J_f\\,\\mathrm{d}x$"];
        let bwd_names = ["$J_f^\\top J_g^\\top u$", "$J_g^\\top u$", "$u$"];
        let point_names = ["$p$", "$f(p)$", "$g(f(p))$"];
        for (k, pl) in planes.iter().enumerate() {
            // (a plane too small to hold names beside its arrows has none)
            if (pl.rect.1 - pl.rect.0).min_element() < 100.0 {
                continue;
            }
            let mut room = Room::new(f);
            let (min, max) = pl.rect;
            room.wall(Vec2::ZERO, Vec2::new(view.x, min.y + 1.0));
            room.wall(Vec2::new(0.0, max.y - 1.0), view);
            room.wall(Vec2::ZERO, Vec2::new(min.x + 1.0, view.y));
            room.wall(Vec2::new(max.x - 1.0, 0.0), view);
            if lay.narrow() {
                let t = lay.titles[k];
                room.wall(t - Vec2::splat(4.0), t + Vec2::new(f.text_size(names[k], 12.5).x + 8.0, 20.0));
            }
            for (a, b) in &busy[k] {
                room.segment_px(*a, *b);
            }
            let from = pl.px(points[k]);
            let mut away = Vec2::ZERO;
            for tip in tips[k].iter().flatten() {
                away -= (*tip - from).normalize_or_zero();
            }
            let alpha_of = |r: f64| alpha * ((r as f32 - 0.5) * 4.0).max(0.0).min(1.0);
            if pl.contains(from) {
                if let Some(w) = world(f, from) {
                    let towards = if away == Vec2::ZERO { Vec2::new(-1.0, 1.0) } else { away };
                    room.label(f, ("point", k), w, point_names[k], size, faded(pal::TEXT, alpha), towards, 9.0, false);
                }
            }
            for (side, (names, c, r)) in [(fwd_names, FORWARD, grown[k]), (bwd_names, BACKWARD, grown[3 + k])].into_iter().enumerate() {
                // (on a phone only the two that are set by hand are named: the board says the rest)
                if short && !((side == 0 && k == 0) || (side == 1 && k == 2)) {
                    continue;
                }
                let Some(tip) = tips[k][side] else { continue };
                if tip.distance(from) < 12.0 || !pl.contains(tip) {
                    continue;
                }
                if let Some(w) = world(f, tip) {
                    room.label(f, (["forwards", "backwards"][side], k), w, names[k], size, faded(c, alpha_of(r)), tip - from, 7.0, true);
                }
            }
        }

        if lay.caption && self.board_of(lay).is_some() {
            let parts = [
                "white: $p$, and where $f$ and $g$ carry it",
                r"cyan: a small step $\mathrm{d}x$, carried forwards by the Jacobians",
                "yellow: where the step really goes",
                r"pink: the gradient $u$ of $L = u \cdot z$, carried back by the transposes",
                "orange: level lines of $L$",
                "grey: a grid, and where the maps carry it",
            ];
            bottom_caption(f, &parts, 12.0);
        }
        if let Some(at) = lay.key {
            let entries = [
                (pal::WHITE, "$p$, and where $f$ and then $g$ carry it"),
                (FORWARD, r"a small step $\mathrm{d}x$, carried forwards by the Jacobians"),
                (TRUE, "where the step really goes"),
                (BACKWARD, r"the gradient $u$ of $L = u \cdot z$, carried back by the transposes"),
                (LOSS, "level lines of $L$"),
                (GRID_COLOUR, "a grid, and where the maps carry it"),
            ];
            let entries = entries.map(|(c, t)| (faded(c, alpha), t));
            hud_key(f, at, &entries, 12.0);
        }
    }

    /// One variable: the three number lines.
    fn picture1(&mut self, f: &mut Frame, lay: &Layout, alpha: f32) {
        let lines = lay.lines;
        let s = self.shown();
        let lift = |line: &Line, out: f32| line.side * out;
        // the handles: the step's tip, the point, and the gradient's tip (none while the
        // picture fades from two variables to one)
        let still = self.settled() && self.one;
        if self.show.forwards() && self.grown(0) > 0.5 && still {
            let l = lines[0];
            if let Some(q) = plot::screen_handle(f, 4, l.px(s.p1 + s.dx1) + lift(&l, 10.0), FORWARD) {
                let dx = (l.value(q) - s.p1).max(-DX1_MAX).min(DX1_MAX);
                self.glide.now(|t| t.dx1 = dx);
                self.hand();
            }
        }
        if let Some(q) = still.then(|| plot::screen_handle(f, 5, lines[0].px(s.p1), pal::WHITE)).flatten() {
            let p = lines[0].value(q).max(-P1_MAX).min(P1_MAX);
            self.glide.now(|t| t.p1 = p);
            self.hand();
        }
        if self.show.backwards() && self.grown(5) > 0.5 && still {
            let l = lines[2];
            let z = g1(f1(s.p1));
            if let Some(q) = plot::screen_handle(f, 6, l.px(z + s.u1) + lift(&l, 22.0), BACKWARD) {
                let u = (l.value(q) - z).max(-U1_MAX).min(U1_MAX);
                self.glide.now(|t| t.u1 = u);
                self.hand();
            }
        }
        let a = self.at1();
        let grown: [f64; 6] = [0, 1, 2, 3, 4, 5].map(|k| self.grown(k));
        let view = f.view_size();
        // (one more dot than the planes draw, unseen: the dots of the number lines are then
        // never compared with the planes' as the one picture gives way to the other)
        plot::disc(f, Vec2::ZERO, 1.0, Color::CLEAR);
        let screen = (Vec2::ZERO, view);

        // the lines, their marks and the marks' numbers
        let marks: [(f64, usize); 3] = [(0.25, 2), (0.5, 2), (0.25, 2)];
        for (k, l) in lines.iter().enumerate() {
            let (lo, hi) = (l.range[0] - 0.1, l.range[1] + 0.1);
            plot::line_of(f, FRAMES1, l.px(lo), l.px(hi), 1.6, faded(pal::DIM, 0.8 * alpha));
            let (step, every) = marks[k];
            let first = (l.range[0] / step).ceil() as i64;
            let last = (l.range[1] / step).floor() as i64;
            for i in first..=last {
                let v = i as f64 * step;
                let q = l.px(v);
                let named = i % every as i64 == 0;
                let reach = if named { 5.0 } else { 3.0 };
                plot::line_of(f, FRAMES1, q - l.side * reach, q + l.side * reach, 1.0, faded(pal::DIM, 0.8 * alpha));
                if named {
                    let at = q - l.side * 9.0;
                    let align = if lay.narrow() { Align2::LEFT_CENTER } else { Align2::CENTER_TOP };
                    f.text_px(at, tick(v), 10.5, faded(pal::DIM, alpha), align);
                }
            }
            let name = ["$x$", "$y = x^3$", r"$z = \arctan y$"][k];
            let align = if lay.narrow() { Align2::CENTER_BOTTOM } else { Align2::RIGHT_CENTER };
            f.text_px(lay.line_names[k], name, lay.text, faded(pal::TEXT, alpha), align);
        }

        // where f and g carry the marks of x: a fan from each line to the next
        let fan = faded(GRID_COLOUR, 0.35 * alpha);
        for i in -5..=5 {
            let t = i as f64 * 0.25;
            let (x, y, z) = (lines[0].px(t), lines[1].px(f1(t)), lines[2].px(g1(f1(t))));
            plot::line_of(f, FAN, x, y, 1.0, fan);
            plot::line_of(f, FAN, y, z, 1.0, fan);
        }

        // where the step really goes: a yellow band from each line to the next, and the
        // step's true image on each line
        let ends = |s: f64| {
            let x = a.p + a.dx * s;
            [x, f1(x), g1(f1(x))]
        };
        let (start, end) = (ends(0.0), ends(1.0));
        for k in 1..3 {
            let r = grown[k] as f32;
            let (l0, l1) = (lines[k - 1], lines[k]);
            let (p0, q0, p1, q1) = (l0.px(start[k - 1]), l0.px(end[k - 1]), l1.px(start[k]), l1.px(end[k]));
            let clip = |p: Vec2| p.max(Vec2::ZERO).min(view);
            let (p0, q0, p1, q1) = (clip(p0), clip(q0), clip(p1), clip(q1));
            let fill = faded(TRUE, 0.11 * alpha * r);
            let [c0, c1, c2, c3] = [p0, q0, q1, p1].map(|v| f.hud_at(1, v));
            f.dl.tris_alpha.tri(c0, c1, c2, fill);
            f.dl.tris_alpha.tri(c0, c2, c3, fill);
            let edge = faded(TRUE, 0.55 * alpha * r);
            plot::line_of(f, TRUE1, p0, p1, 1.0, edge);
            plot::line_of(f, TRUE1, q0, q1, 1.0, edge);
            plot::path_of(f, TRUE1, &plot::clip(&[p1, q1], screen), 4.0, faded(TRUE, 0.9 * alpha * r));
        }

        // the arrows beside the lines: forwards nearer, backwards further out
        let points = [a.p, a.y, a.z];
        let pushes = [a.dx, a.dy, a.dz];
        let pulls = [a.gx, a.gy, a.u];
        let mut tips = [[None; 2]; 3];
        for (k, l) in lines.iter().enumerate() {
            for (side, (v, r, c, out)) in [(pushes[k], grown[k], FORWARD, 10.0), (pulls[k], grown[3 + k], BACKWARD, 22.0)].into_iter().enumerate() {
                let from = l.px(points[k]) + lift(l, out);
                let tip = l.px(points[k] + v * r) + lift(l, out);
                arrow(f, ARROWS1, from, tip, screen, 2.6, faded(c, alpha * (r as f32 * 4.0).min(1.0)));
                if r > 0.5 {
                    tips[k][side] = Some((from, tip));
                }
            }
            if k > 0 || !still {
                plot::disc(f, l.px(points[k]), if k > 0 { 10.0 } else { 12.0 }, faded(pal::WHITE, alpha));
            }
        }

        // the names: the slopes between the lines, and the arrows
        let mut room = Room::new(f);
        if let Some(b) = &lay.board {
            room.wall(b.rect.0 - Vec2::splat(4.0), b.rect.1 + Vec2::splat(4.0));
        }
        for (k, l) in lines.iter().enumerate() {
            room.segment_px(l.px(l.range[0] - 0.1), l.px(l.range[1] + 0.1));
            room.boxed(lay.line_names[k] - Vec2::new(60.0, 10.0), lay.line_names[k] + Vec2::new(4.0, 10.0));
            for (from, tip) in tips[k].iter().flatten() {
                room.segment_px(*from, *tip);
            }
        }
        let size = lay.text;
        let slopes = [format!("$f'(p) = {}$", num(a.df, 2)), format!("$g'(f(p)) = {}$", num(a.dg, 2))];
        for k in 0..2 {
            let r = grown[k + 1] as f32;
            if r < 0.5 {
                continue;
            }
            // beside the band, half way between the lines
            let mid = (lines[k].px(end[k]) + lines[k + 1].px(end[k + 1])) * 0.5;
            if let Some(w) = world(f, mid) {
                let towards = if lay.narrow() { Vec2::new(0.0, -1.0) } else { Vec2::X };
                room.label(f, ("slope", k), w, &slopes[k], size, faded(pal::TEXT, alpha), towards, 10.0, false);
            }
        }
        let fwd_names = [r"$\mathrm{d}x$", r"$f'(p)\,\mathrm{d}x$", r"$g'(f(p))\,f'(p)\,\mathrm{d}x$"];
        let bwd_names = [r"$f'(p)\,g'(f(p))\,u$", r"$g'(f(p))\,u$", "$u$"];
        for (k, l) in lines.iter().enumerate() {
            for (side, (names, c)) in [(fwd_names, FORWARD), (bwd_names, BACKWARD)].into_iter().enumerate() {
                if lay.narrow() && !((side == 0 && k == 0) || (side == 1 && k == 2)) {
                    continue;
                }
                let Some((from, tip)) = tips[k][side] else { continue };
                if let Some(w) = world(f, tip) {
                    let towards = l.side + (tip - from).normalize_or_zero();
                    room.label(f, (["forwards", "backwards"][side], k), w, names[k], size - 1.0, faded(c, alpha), towards, 6.0, true);
                }
            }
        }
        if lay.caption1 && self.board_of(lay).is_some() {
            let parts = [
                r"cyan: a small step $\mathrm{d}x$, and the slopes times it",
                "yellow: where the step really goes",
                "pink: the slope of $L = uz$, carried back",
                "grey: where $f$ and $g$ carry the marks of $x$",
            ];
            bottom_caption(f, &parts, 12.0);
        }
    }

    /// The board: the Jacobians and their product (or, with one variable, the slopes),
    /// the two sweeps worked out with their numbers, and what the step really does to L.
    /// (Its buttons are drawn with its input.)
    fn board(&self, f: &mut Frame, b: &Board, alpha: f32) {
        hud_panel(f, b.rect.0, b.rect.1);
        let (text, small) = (b.text, b.small);
        let c = |c: Color| faded(c, alpha);
        // each sweep's column: its heading, then its lines, the last one the number it ends with
        let mut cols: [Vec<String>; 2] = Default::default();
        let truth;
        if self.one_shown() {
            let a = self.at1();
            let slopes = [format!("$f'(p) = 3p^2 = {}$", num(a.df, 2)), format!("$g'(f(p)) = 1/(1 + f(p)^2) = {}$", num(a.dg, 2)), format!(r"$(g \circ f)'(p) = g'(f(p))\,f'(p) = {}$", num(a.df * a.dg, 2))];
            if b.short {
                let line = format!(r"$f'(p) = {}$, $g'(f(p)) = {}$, $(g \circ f)'(p) = {}$", num(a.df, 2), num(a.dg, 2), num(a.df * a.dg, 2));
                f.text_px(b.mats + Vec2::new(0.0, -b.small), line, small, c(pal::TEXT), Align2::LEFT_TOP);
            } else {
                for (i, line) in slopes.into_iter().enumerate() {
                    f.text_px(b.mats + Vec2::new(0.0, -b.small - 2.0 + i as f32 * b.line_h * 1.15), line, small + 0.5, c(pal::TEXT), Align2::LEFT_TOP);
                }
            }
            let change = a.u * (g1(f1(a.p + a.dx)) - a.z);
            if b.short {
                cols[0].push(format!("forwards: ${}$", num(a.u * a.dz, 3)));
                cols[1].push(format!("backwards: ${}$", num(a.gx * a.dx, 3)));
            } else {
                cols[0] = vec![
                    "forwards: the step, slope by slope".into(),
                    format!(r"$\mathrm{{d}}x = {}$", num(a.dx, 3)),
                    format!(r"$f'(p)\,\mathrm{{d}}x = {}$", num(a.dy, 3)),
                    format!(r"$g'(f(p))\,f'(p)\,\mathrm{{d}}x = {}$", num(a.dz, 3)),
                    format!("times $u$: ${}$", num(a.u * a.dz, 3)),
                ];
                cols[1] = vec![
                    "backwards: the slope of $L$, carried back".into(),
                    format!("$u = {}$", num(a.u, 3)),
                    format!(r"$g'(f(p))\,u = {}$", num(a.gy, 3)),
                    format!(r"$f'(p)\,g'(f(p))\,u = {}$", num(a.gx, 3)),
                    format!(r"times $\mathrm{{d}}x$: ${}$", num(a.gx * a.dx, 3)),
                ];
            }
            truth = format!("what the step really does to $L = uz$: ${}$", num(change, 3));
        } else {
            let at = self.at();
            let s = self.shown();
            // J_g J_f = J, each with its name above it and how far it can stretch a step under it
            let msize = matrix_size(2, 2, CELL, text);
            let mut x = b.mats.x;
            let mid = b.mats.y + msize.y * 0.5;
            let names = if b.short { ["$J_g$", "$J_f$", "$J$"] } else { ["$J_g$ at $f(p)$", "$J_f$ at $p$", "$J = J_g J_f$"] };
            for (k, m) in [at.jg, at.jf, at.j].into_iter().enumerate() {
                if k == 2 {
                    f.text_px(Vec2::new(x + 9.0, mid), "$=$", text, c(pal::TEXT), Align2::CENTER_CENTER);
                    x += 20.0;
                }
                let centre = x + msize.x * 0.5;
                f.text_px(Vec2::new(centre, b.mats.y - 3.0), names[k], small, c(pal::TEXT), Align2::CENTER_BOTTOM);
                hud_matrix(f, Vec2::new(x, b.mats.y), &cells(m, c(pal::TEXT)), CELL, text, c(pal::DIM));
                if !b.short {
                    f.text_px(Vec2::new(centre, b.mats.y + msize.y + 3.0), format!(r"stretch ${}$", num(stretch(m), decimals(stretch(m)))), small - 1.0, c(pal::DIM), Align2::CENTER_TOP);
                }
                x += msize.x + 5.0;
            }
            let change = true_change(&s.f, &s.g, s.p, s.dx, s.u);
            if b.short {
                cols[0].push(format!(r"forwards: $u \cdot J_g J_f\,\mathrm{{d}}x = {}$", num(at.forwards(), 3)));
                cols[1].push(format!(r"backwards: $J_f^\top J_g^\top u \cdot \mathrm{{d}}x = {}$", num(at.backwards(), 3)));
            } else {
                cols[0] = vec![
                    "forwards: a column at a time".into(),
                    format!(r"$\mathrm{{d}}x = {}$", pair(at.dx)),
                    format!(r"$J_f\,\mathrm{{d}}x = {}$", pair(at.dy)),
                    format!(r"$J_g J_f\,\mathrm{{d}}x = {}$", pair(at.dz)),
                    format!(r"$u \cdot J_g J_f\,\mathrm{{d}}x = {}$", num(at.forwards(), 3)),
                ];
                cols[1] = vec![
                    "backwards: a row at a time".into(),
                    format!("$u = {}$", pair(at.u)),
                    format!(r"$J_g^\top u = {}$", pair(at.gy)),
                    format!(r"$J_f^\top J_g^\top u = {}$", pair(at.gx)),
                    format!(r"$J_f^\top J_g^\top u \cdot \mathrm{{d}}x = {}$", num(at.backwards(), 3)),
                ];
            }
            truth = format!("what the step really does to $L$: ${}$", num(change, 3));
        }
        for (k, col) in cols.iter().enumerate() {
            let hue = [FORWARD, BACKWARD][k];
            let n = col.len();
            for (i, line) in col.iter().enumerate() {
                // (the heading and the number the sweep ends with in its colour, the steps between paler)
                let color = if b.short || i == 0 || i + 1 == n { hue } else { hue.mix(pal::TEXT, 0.55) };
                f.text_px(b.cols[k] + Vec2::new(0.0, i as f32 * b.line_h), line.as_str(), small, c(color), Align2::LEFT_TOP);
            }
        }
        f.text_px(b.truth, truth, small, c(pal::DIM.mix(pal::TEXT, 0.3)), Align2::LEFT_TOP);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A Jacobian by central differences, to hold the exact ones against.
    fn by_differences(m: impl Fn(DVec2) -> DVec2, v: DVec2) -> DMat2 {
        let h = 1e-6;
        let col = |e: DVec2| (m(v + e * h) - m(v - e * h)) / (2.0 * h);
        DMat2::from_cols(col(DVec2::X), col(DVec2::Y))
    }

    /// A gradient by central differences.
    fn gradient(l: impl Fn(DVec2) -> f64, v: DVec2) -> DVec2 {
        let h = 1e-6;
        DVec2::new((l(v + DVec2::X * h) - l(v - DVec2::X * h)) / (2.0 * h), (l(v + DVec2::Y * h) - l(v - DVec2::Y * h)) / (2.0 * h))
    }

    fn size(m: DMat2) -> f64 {
        m.to_cols_array().iter().fold(0.0f64, |s, v| s.max(v.abs()))
    }

    /// Points, steps and gradients to try things with, the same in every run.
    fn samples(n: usize, seed: u64) -> Vec<(DVec2, DVec2, DVec2)> {
        let mut rng = Rng::new(seed);
        let mut r = |a: f64| rng.range(-a as f32, a as f32) as f64;
        (0..n).map(|_| (DVec2::new(r(2.0), r(2.0)), DVec2::new(r(1.0), r(1.0)), DVec2::new(r(1.5), r(1.5)))).collect()
    }

    /// Every map there is: each kind, and some on their way from one to another.
    fn maps() -> Vec<Map> {
        let mut maps: Vec<Map> = KINDS.iter().map(|k| Map::of(*k)).collect();
        maps.push(Map([0.3, 0.7, 0.0, 0.0]));
        maps.push(Map([0.0, 0.25, 0.5, 0.25]));
        maps
    }

    /// The Jacobians written out are the maps' own, to the accuracy of differences.
    #[test]
    fn the_jacobians_are_exact() {
        for m in maps() {
            for (v, ..) in samples(200, 1) {
                let (exact, near) = (m.jacobian(v), by_differences(|w| m.at(w), v));
                assert!(size(exact - near) < 1e-8 * size(exact).max(1.0), "{m:?} at {v:?}: {exact:?} against {near:?}");
            }
        }
        // the swirl keeps areas, and turns the centre by its whole angle
        for (v, ..) in samples(50, 2) {
            assert!((Kind::Swirl.jacobian(v).determinant() - 1.0).abs() < 1e-12);
        }
        assert!((Kind::Swirl.jacobian(DVec2::ZERO) - DMat2::from_angle(SWIRL)).to_cols_array().iter().all(|e| e.abs() < 1e-15));
    }

    /// J_g(f(p)) J_f(p) is the Jacobian of g after f, for every pair of maps.
    #[test]
    fn the_chain_rule_holds() {
        for f in maps() {
            for g in maps() {
                for (p, dx, u) in samples(40, 3) {
                    let at = chain(&f, &g, p, dx, u);
                    let near = by_differences(|w| g.at(f.at(w)), p);
                    assert!(size(at.j - near) < 1e-7 * size(near).max(1.0), "{f:?} then {g:?} at {p:?}");
                    // and the pushes are the product's: J dx
                    assert!((at.dz - at.j * dx).length() < 1e-12 * dx.length().max(1.0) * size(at.j).max(1.0));
                }
            }
        }
    }

    /// Forwards and backwards give the same number, u · (J_g J_f dx) = (J_fᵀ J_gᵀ u) · dx,
    /// to rounding, at random points and in random directions.
    #[test]
    fn forwards_and_backwards_give_the_same_number() {
        for f in maps() {
            for g in maps() {
                for (p, dx, u) in samples(60, 4) {
                    let at = chain(&f, &g, p, dx, u);
                    let scale = u.length() * dx.length() * size(at.jf).max(1.0) * size(at.jg).max(1.0);
                    assert!((at.forwards() - at.backwards()).abs() <= 1e-14 * scale.max(1e-300) + 1e-300, "{} against {}", at.forwards(), at.backwards());
                    // the backward sweep is the product's transpose: Jᵀu
                    assert!((at.gx - at.j.transpose() * u).length() < 1e-12 * scale.max(1.0));
                }
            }
        }
    }

    /// The pink arrows are the gradients of L = u · z as it depends on y and on x: J_gᵀu
    /// and J_fᵀJ_gᵀu, against differences of L itself. (And not J_g u.)
    #[test]
    fn the_pulled_back_gradients_are_the_gradients_of_l() {
        for f in maps() {
            for g in maps() {
                for (p, dx, u) in samples(30, 5) {
                    let at = chain(&f, &g, p, dx, u);
                    let gy = gradient(|y| u.dot(g.at(y)), at.y);
                    let gx = gradient(|x| u.dot(g.at(f.at(x))), p);
                    assert!((at.gy - gy).length() < 1e-7 * u.length().max(1.0), "{:?} against {gy:?}", at.gy);
                    assert!((at.gx - gx).length() < 1e-7 * u.length().max(1.0), "{:?} against {gx:?}", at.gx);
                }
            }
        }
        // the first-order change, against the true one: the miss shrinks as the square of the step
        let s = setting_of(&START);
        let miss = |k: f64| (true_change(&s.f, &s.g, s.p, s.dx * k, s.u) - chain(&s.f, &s.g, s.p, s.dx * k, s.u).forwards()).abs();
        for k in [0.004, 0.001] {
            let ratio = miss(k) / miss(k / 2.0);
            assert!((3.8..4.2).contains(&ratio), "{ratio}");
        }
    }

    /// The traced level line through a point keeps L's value there, and leaves the point at
    /// right angles to the gradient; and the pieces of the other level lines lie on their
    /// levels.
    #[test]
    fn each_gradient_is_at_right_angles_to_its_level_line() {
        let s = setting_of(&START);
        let (f, g, u) = (s.f, s.g, s.u);
        let at = chain(&f, &g, s.p, s.dx, u);
        let phi_y = |v: DVec2| (u.dot(g.at(v)), g.jacobian(v).transpose() * u);
        let phi_x = |v: DVec2| {
            let y = f.at(v);
            (u.dot(g.at(y)), f.jacobian(v).transpose() * (g.jacobian(y).transpose() * u))
        };
        let (lo, hi) = (DVec2::splat(-2.5), DVec2::splat(2.5));
        for (phi, start, grad) in [(&phi_y as &dyn Fn(DVec2) -> (f64, DVec2), at.y, at.gy), (&phi_x, at.p, at.gx)] {
            let line = trace(phi, start, 0.04, 200, lo, hi);
            assert!(line.len() > 40, "{}", line.len());
            let c = phi(start).0;
            assert!(line.iter().all(|v| (phi(*v).0 - c).abs() < 1e-9), "off its level");
            // the point is on it, and the line leaves it at right angles to the gradient
            let i = line.iter().position(|v| *v == start).expect("the point");
            let tangent = (line[i + 1] - line[i - 1]).normalize();
            assert!(tangent.dot(grad.normalize()).abs() < 2e-3, "{}", tangent.dot(grad.normalize()));
        }
        // the level lines found by marching squares lie on their levels, with the larger
        // values on their left
        let plane = Plane::fit((Vec2::ZERO, Vec2::splat(400.0)), (DVec2::ZERO, 2.2));
        let mut levels = Levels::default();
        let step = level_step(u, 1.07);
        let l = |v: DVec2| u.dot(g.at(f.at(v)));
        levels.update(&plane, &[1], |v| g.at(f.at(v)), u, step);
        assert!(levels.lines.len() > 8 && levels.lines.iter().all(|line| line.len() == LINE_POINTS));
        for line in &levels.lines {
            let k = (l(line[0]) / step).round();
            assert!(line.iter().all(|v| (l(*v) - k * step).abs() < 0.03 * step), "a line off its level");
            let (a, b) = (line[LINE_POINTS / 2 - 1], line[LINE_POINTS / 2]);
            let mid = (a + b) * 0.5;
            let left = (b - a).perp().normalize() * 0.01;
            assert!(l(mid + left) > l(mid - left), "the larger values on the right");
        }
    }

    /// Pieces of a level line are joined into whole lines, running with the larger values
    /// on their left, and drawn with as many points as asked for: the circle |v| = 1 of
    /// |v|², one closed line, going round the way that keeps the inside on its right.
    #[test]
    fn level_lines_are_joined_whole() {
        let (n, lo) = (40, DVec2::splat(-2.0));
        let cell = DVec2::splat(0.1);
        let values: Vec<f64> = (0..=n).flat_map(|j| (0..=n).map(move |i| (i, j))).map(|(i, j)| (lo + cell * DVec2::new(i as f64, j as f64)).length_squared()).collect();
        let all: Vec<(usize, usize)> = (0..n).flat_map(|j| (0..n).map(move |i| (i, j))).collect();
        let pieces = contour(&values, n, lo, cell, 1.0, &all);
        let lines = chains(&pieces);
        assert_eq!(lines.len(), 1, "{} lines", lines.len());
        let line = &lines[0];
        assert!(line[0].distance(*line.last().unwrap_or(&DVec2::ZERO)) < 1e-12, "not closed");
        assert!(line.iter().all(|v| (v.length() - 1.0).abs() < 0.01));
        // the larger values outside, on the left: the line goes round clockwise
        let area: f64 = line.windows(2).map(|w| w[0].perp_dot(w[1])).sum::<f64>() * 0.5;
        assert!(area < -3.0, "{area}");
        let even = resample(line, LINE_POINTS);
        assert_eq!(even.len(), LINE_POINTS);
        let gaps: Vec<f64> = even.windows(2).map(|w| w[0].distance(w[1])).collect();
        let (least, most) = gaps.iter().fold((f64::MAX, 0.0f64), |(a, b), g| (a.min(*g), b.max(*g)));
        assert!(most < 1.05 * least, "{least} {most}");
        // a line of no length, and none at all
        assert!(resample(&[DVec2::ONE, DVec2::ONE], 10).is_empty() && resample(&[], 10).is_empty());
        assert!(chains(&[]).is_empty());
        // a box edge: the segment stops where it leaves the box
        assert_eq!(to_edge(DVec2::ZERO, DVec2::new(3.0, 1.5), DVec2::splat(-1.0), DVec2::splat(1.0)), DVec2::new(1.0, 0.5));
        assert_eq!(to_edge(DVec2::ZERO, DVec2::splat(0.5), DVec2::splat(-1.0), DVec2::splat(1.0)), DVec2::splat(0.5));
    }

    /// The scene as an experiment's button leaves it, run at 60 frames a second until it
    /// is at rest (the glide arrived and the sweeps played), or for half a minute.
    fn run(k: usize) -> ChainRule {
        let mut scene = ChainRule::new();
        scene.set_up(k);
        for _ in 0..1800 {
            scene.advance(1.0 / 60.0, 1.0 / 60.0);
            if scene.at_rest() {
                break;
            }
        }
        assert!(scene.at_rest(), "experiment {k} does not come to rest");
        scene
    }

    fn pair2(v: DVec2) -> String {
        format!("({}, {})", num(v.x, 2), num(v.y, 2))
    }

    #[test]
    fn slopes_multiply_as_the_caption_says() {
        let scene = run(0);
        assert!(scene.one_shown() && scene.show == Show::Forwards);
        let a = scene.at1();
        assert!(a.p == 1.0 && a.y == 1.0 && a.df == 3.0 && a.dg == 0.5 && a.df * a.dg == 1.5);
        // slopes against differences, anywhere on the line
        for x in [-1.2, -0.4, 0.0, 0.3, 1.0, 1.25] {
            let h = 1e-6;
            assert!((df1(x) - (f1(x + h) - f1(x - h)) / (2.0 * h)).abs() < 1e-8);
            assert!((dg1(x) - (g1(x + h) - g1(x - h)) / (2.0 * h)).abs() < 1e-8);
            let whole = |t: f64| g1(f1(t));
            assert!((dg1(f1(x)) * df1(x) - (whole(x + h) - whole(x - h)) / (2.0 * h)).abs() < 1e-8);
        }
        // the step's pushes, and where it really goes along y
        assert!((a.dy - 3.0 * a.dx).abs() < 1e-15 && (a.dz - 1.5 * a.dx).abs() < 1e-15);
        assert_eq!((num(f1(a.p + a.dx) - a.y, 2).as_str(), num(a.dy, 2).as_str()), ("0.52", "0.45"));
        // the shorter the step, the closer the two (relative to the step)
        let off = |dx: f64| ((f1(1.0 + dx) - 1.0) - 3.0 * dx).abs() / dx;
        assert!(off(0.15) > off(0.075) && off(0.075) > off(0.0375));
    }

    #[test]
    fn a_step_pushed_forwards_does_what_its_caption_says() {
        let scene = run(1);
        assert!(!scene.one_shown() && scene.show == Show::Forwards);
        let at = scene.at();
        let s = scene.shown();
        assert_eq!((pair2(at.dx), pair2(at.dy), pair2(at.dz)), ("(0.30, 0.12)".into(), "(0.35, 0.26)".into(), "(0.25, 0.60)".into()));
        assert!((at.dz - at.j * at.dx).length() < 1e-15);
        let miss = |k: f64| {
            let dx = s.dx * k;
            ((s.f.at(s.p + dx) - at.y - at.jf * dx).length(), (s.g.at(s.f.at(s.p + dx)) - at.z - at.j * dx).length())
        };
        let (my, mz) = miss(1.0);
        assert_eq!((num(my, 2).as_str(), num(mz, 2).as_str()), ("0.01", "0.08"));
        // halved, the misses fall to less than a third; and for small steps, to a quarter
        let (hy, hz) = miss(0.5);
        assert!(hy / my < 1.0 / 3.0 && hz / mz < 1.0 / 3.0, "{} {}", hy / my, hz / mz);
        let ((sy, sz), (ty, tz)) = (miss(0.02), miss(0.01));
        assert!((0.23..0.27).contains(&(ty / sy)) && (0.23..0.27).contains(&(tz / sz)), "{} {}", ty / sy, tz / sz);
    }

    #[test]
    fn a_gradient_pulled_back_does_what_its_caption_says() {
        let scene = run(2);
        assert!(!scene.one_shown() && scene.show == Show::Backwards);
        let at = scene.at();
        assert_eq!((pair2(at.u), pair2(at.gy), pair2(at.jg * at.u), pair2(at.gx)), ("(0.50, 0.15)".into(), "(0.79, −0.23)".into(), "(0.54, 0.58)".into(), "(0.68, 0.09)".into()));
    }

    #[test]
    fn either_end_gives_the_same_number_as_the_caption_says() {
        let scene = run(3);
        assert!(!scene.one_shown() && scene.show == Show::Both);
        let at = scene.at();
        let s = scene.shown();
        assert_eq!((num(at.forwards(), 3).as_str(), num(at.backwards(), 3).as_str()), ("0.216", "0.216"));
        assert_eq!(num(true_change(&s.f, &s.g, s.p, s.dx, s.u), 3), "0.203");
    }

    #[test]
    fn one_flat_link_flattens_the_chain_as_the_caption_says() {
        let scene = run(4);
        let at = scene.at();
        let input = rows(LAYER_W) * at.y + DVec2::from(LAYER_B);
        assert_eq!((num(input.x, 1).as_str(), num(input.y, 1).as_str()), ("−2.6", "−2.5"));
        let slope = |a: f64| 1.0 - a.tanh().powi(2);
        assert_eq!((num(slope(input.x), 3).as_str(), num(slope(input.y), 3).as_str()), ("0.021", "0.024"));
        let (g, f, j) = (stretch(at.jg), stretch(at.jf), stretch(at.j));
        assert_eq!((num(g, 3).as_str(), num(f, 2).as_str(), num(j, 3).as_str()), ("0.036", "1.16", "0.041"));
        assert!(j <= g * f && j <= 0.036 * 1.16);
        // the stretch, against the most J lengthens any unit step
        let most = (0..3600).map(|i| (at.j * DVec2::from_angle(i as f64 * std::f64::consts::TAU / 3600.0)).length()).fold(0.0, f64::max);
        assert!((most - j).abs() < 1e-6 * j);
        // the step reaches z as a dot, and the gradient comes back small, against the
        // opening's
        assert!(at.dz.length() < 0.015 && at.gx.length() < at.u.length() / 20.0);
        let usual = chain(&Map::of(START.f), &Map::of(START.g), START.p.into(), START.dx.into(), START.u.into());
        assert!(usual.gx.length() > 0.8 * usual.u.length() && stretch(usual.jg) > 1.2);
    }

    /// What every experiment shows lies in its plane's window: the points, and the tips of
    /// the arrows (the step's and its pushes, the gradients).
    #[test]
    fn every_experiment_is_in_its_planes() {
        for (k, t) in TRIALS.iter().enumerate() {
            let s = setting_of(t);
            let at = chain(&s.f, &s.g, s.p, s.dx, s.u);
            let windows = Grids::of(&s.f, &s.g).windows;
            let within = |v: DVec2, plane: usize| {
                let (c, half) = windows[plane];
                (v - c).abs().max_element() <= half
            };
            for (plane, (point, tips)) in [(at.p, [at.dx, at.gx]), (at.y, [at.dy, at.gy]), (at.z, [at.dz, at.u])].into_iter().enumerate() {
                assert!(within(point, plane), "trial {k}, plane {plane}");
                for tip in tips {
                    assert!(within(point + tip, plane), "trial {k}, plane {plane}: {:?}", point + tip);
                }
            }
        }
    }

    /// The sweeps draw forwards from x to z, then backwards from z to x, a stage at a time.
    #[test]
    fn the_sweeps_go_forwards_then_back() {
        let first_full = |show: Show, k: usize| (0..2000).map(|i| i as f32 * 0.005).find(|t| reveal_at(show, Some(*t))[k] >= 1.0);
        let times: Vec<f32> = (0..6).map(|k| first_full(Show::Both, k).unwrap_or(f32::MAX)).collect();
        assert!(times[0] < times[1] && times[1] < times[2], "{times:?}");
        assert!(times[2] < times[5] && times[5] < times[4] && times[4] < times[3], "{times:?}");
        assert!(times[3] <= film_length(Show::Both) + 1e-3);
        // backwards alone starts at once, from z
        assert!(first_full(Show::Backwards, 5) < first_full(Show::Backwards, 3));
        assert_eq!(reveal_at(Show::Backwards, Some(0.3))[..3], [0.0; 3]);
        assert_eq!(reveal_at(Show::Forwards, None), [1.0, 1.0, 1.0, 0.0, 0.0, 0.0]);
    }

    /// Every experiment comes to rest, and a change of map glides there through maps that
    /// are each a whole map (their weights add up to 1).
    #[test]
    fn every_change_glides_and_comes_to_rest() {
        for k in 0..TRIALS.len() {
            run(k);
        }
        let mut scene = run(3);
        scene.choose(0, Kind::Swirl);
        scene.choose(1, Kind::Linear);
        let mut frames = 0;
        while !scene.at_rest() && frames < 1800 {
            scene.advance(1.0 / 60.0, 1.0 / 60.0);
            let s = scene.shown();
            assert!((s.f.0.iter().sum::<f64>() - 1.0).abs() < 1e-12 && (s.g.0.iter().sum::<f64>() - 1.0).abs() < 1e-12);
            frames += 1;
        }
        assert!(frames > 10 && frames < 120, "{frames}");
        assert_eq!((scene.shown().f.kind(), scene.shown().g.kind()), (Some(Kind::Swirl), Some(Kind::Linear)));
    }

    /// Degenerate input: no gradient (u = 0), no step, the point at a corner, the zero
    /// matrix: every number stays finite and nothing is traced that cannot be.
    #[test]
    fn degenerate_input_stays_finite() {
        for f in maps() {
            for g in maps() {
                for p in [DVec2::splat(GRID_HALF), DVec2::splat(-GRID_HALF), DVec2::ZERO] {
                    for (dx, u) in [(DVec2::ZERO, DVec2::ZERO), (DVec2::ZERO, DVec2::X), (DVec2::X * DX_MAX, DVec2::ZERO), (DVec2::Y, DVec2::splat(U_MAX))] {
                        let at = chain(&f, &g, p, dx, u);
                        let all = [at.y, at.z, at.dx, at.dy, at.dz, at.gy, at.gx];
                        assert!(all.iter().all(|v| v.is_finite()) && at.j.is_finite() && stretch(at.j).is_finite());
                        assert!(at.forwards().is_finite() && true_change(&f, &g, p, dx, u).is_finite());
                    }
                }
            }
        }
        assert_eq!(stretch(DMat2::ZERO), 0.0);
        assert!((stretch(DMat2::IDENTITY) - 1.0).abs() < 1e-15 && (stretch(rows([[0.0, 3.0], [0.0, 0.0]])) - 3.0).abs() < 1e-12);
        // no gradient: no level lines and nothing traced
        let plane = Plane::fit((Vec2::ZERO, Vec2::splat(300.0)), (DVec2::ZERO, 2.0));
        let mut levels = Levels::default();
        levels.update(&plane, &[0], |v| v, DVec2::ZERO, 0.0);
        assert!(levels.lines.is_empty());
        assert!(trace(|_| (0.0, DVec2::ZERO), DVec2::ZERO, 0.1, 50, DVec2::splat(-1.0), DVec2::splat(1.0)).is_empty());
        assert!(trace(|_| (f64::NAN, DVec2::NAN), DVec2::ZERO, 0.1, 50, DVec2::splat(-1.0), DVec2::splat(1.0)).is_empty());
        // a window of nothing, or of points that are not numbers
        assert_eq!(window_of([].iter()), (DVec2::ZERO, 1.0));
        assert_eq!(window_of([DVec2::NAN].iter()), (DVec2::ZERO, 1.0));
        assert!(window_of([DVec2::ONE].iter()).1 > 0.0);
        // one variable, at the ends
        for (p, dx, u) in [(P1_MAX, DX1_MAX, U1_MAX), (-P1_MAX, -DX1_MAX, 0.0), (0.0, 0.0, 0.0)] {
            let a = chain1(p, dx, u);
            assert!([a.y, a.z, a.df, a.dg, a.dy, a.dz, a.gy, a.gx].iter().all(|v| v.is_finite()));
        }
        // at p = 0, f is flat, and so is the chain
        assert_eq!(chain1(0.0, 0.1, 1.0).gx, 0.0);
    }

    /// The planes, the number lines and the board fit the canvas, from a phone to a
    /// desktop, and do not lie on one another.
    #[test]
    fn the_layout_fits_every_canvas() {
        let apart = |a: Rect, b: Rect| a.1.x <= b.0.x || b.1.x <= a.0.x || a.1.y <= b.0.y || b.1.y <= a.0.y;
        for (w, h) in [(1078.0, 1000.0), (1600.0, 1000.0), (1300.0, 700.0), (390.0, 800.0), (390.0, 560.0), (800.0, 600.0), (620.0, 480.0), (240.0, 180.0), (1920.0, 1080.0)] {
            for inset in [0.0, 46.0] {
                let view = Vec2::new(w, h);
                let lay = Layout::of(view, inset);
                let inside = |r: Rect| r.0.cmpge(Vec2::splat(-0.5)).all() && r.1.cmple(view + Vec2::splat(0.5)).all() && r.0.y >= inset;
                for (k, r) in lay.planes.iter().enumerate() {
                    assert!(inside(*r) && (r.1 - r.0).min_element() > 5.0, "{w}x{h}: plane {k} {r:?}");
                    for other in &lay.planes[k + 1..] {
                        assert!(apart(*r, *other), "{w}x{h}: planes on one another");
                    }
                    if let Some(b) = &lay.board {
                        assert!(apart(*r, b.rect), "{w}x{h}: the board on plane {k}");
                    }
                }
                for b in lay.board.iter().chain(&lay.board1) {
                    assert!(inside(b.rect), "{w}x{h}: {:?}", b.rect);
                    assert!(b.buttons.iter().all(|r| inside(*r) && r.1.y <= b.rect.1.y && r.1.x > r.0.x + 30.0), "{w}x{h}: buttons");
                }
                // the number lines: on the canvas, and clear of the board under them
                for l in &lay.lines {
                    for v in [l.range[0], l.range[1]] {
                        let q = l.px(v);
                        assert!(q.cmpge(Vec2::ZERO).all() && q.cmple(view).all(), "{w}x{h}: a line's end {q:?}");
                        if let Some(b) = &lay.board1 {
                            assert!(!plot::within(q, b.rect, 0.0), "{w}x{h}: a line under the board");
                        }
                    }
                }
                if h >= 900.0 && w >= 1000.0 {
                    assert!(lay.board.is_some() && lay.board1.is_some() && (lay.planes[0].1 - lay.planes[0].0).x > 300.0, "{w}x{h}: {:?}", lay.arrange);
                }
                if w == 390.0 && h == 800.0 {
                    assert!(lay.arrange == Arrange::Column && lay.board.is_some());
                }
            }
        }
    }
}

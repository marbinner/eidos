//! Too few equations: underdetermined systems, the smallest answer measured in
//! different ways, and why the 1-norm finds sparse ones.

use super::*;
use glam::{DVec2, DVec3};

pub const DEF: SceneDef = SceneDef {
    id: "underdet",
    title: "Too few equations",
    group: "Linear algebra: equations",
    blurb: "Many answers to choose from: the shortest, the sparsest, and why the 1-norm finds sparse ones.",
    make: |_| Box::new(Underdet::create()),
};

/// Experiments: (button, what to look for). What each one sets is in `Underdet::set_up`.
const TRIALS: [(&str, &str); 5] = [
    (
        "the shortest answer",
        "Size measured as ordinary length ($p = 2$): the ball is a disc, and it first touches the line at the foot of the perpendicular from $0$ (the corner mark), $(0.4, 0.8)$. That is the answer the pseudo-inverse gives. Neither coordinate is $0$.",
    ),
    (
        "the sparsest answer",
        "Size measured as $|x| + |y|$ ($p = 1$): the ball is a diamond with its corners on the axes, and it first touches the line at a corner, $(0, 1)$: $x$ is exactly $0$. The disc's answer $(0.4, 0.8)$ is shorter, but its $|x| + |y|$ is $1.2$.",
    ),
    (
        "a tie",
        "The line $x + y = 1.5$ runs parallel to an edge of the diamond, so the whole edge touches at once: every point from $(1.5, 0)$ to $(0, 1.5)$ has $|x| + |y| = 1.5$. The smallest answer is not unique, but the two ends of the edge are still sparse.",
    ),
    (
        "in space: one equation",
        "One equation in three unknowns has a whole plane of answers. The 1-norm ball is now an octahedron, and it touches the plane at a corner, $(0, 0, 1)$: two of the three coordinates are exactly $0$. The sphere's answer, $(0.214, 0.429, 0.643)$, has none.",
    ),
    (
        "in space: two equations",
        "Two planes cross in a line of answers. The octahedron touches the line on one of its edges, at $(0, 0.75, 0.5)$: one coordinate is $0$, two are not. With $m$ equations, a smallest answer for the 1-norm can always be found with at most $m$ coordinates that are not $0$ (in a tie, others have more).",
    ),
];

/// The equation the scene opens with in the plane, and the two it has in space.
const FLAT_START: ([f64; 2], f64) = ([1.0, 2.0], 2.0);
const SPACE_START: [([f64; 3], f64); 2] = [([1.0, 2.0, 3.0], 3.0), ([2.0, 2.0, -1.0], 1.0)];

/// Seconds the ball takes to grow from nothing to the touching size.
const GROW_TIME: f32 = 2.2;

/// How far along the line, either side of its point nearest 0, the two dots are put.
const DOT_SPREAD: f64 = 1.5;

/// The colours of the two equations (a line or plane each) and of the answers of both.
const FIRST: Color = pal::BLUE;
const SECOND: Color = pal::ORANGE;
const BOTH: Color = pal::WHITE;
/// Coordinates that are exactly zero, in the table of answers.
const ZERO: Color = pal::GREEN;

// ----- the mathematics ----------------------------------------------------------------

/// How the size of a point is measured.
#[derive(Clone, Copy, Debug, PartialEq)]
enum Norm {
    /// (|x|ᵖ + |y|ᵖ + |z|ᵖ)^(1/p). (Below p = 1 this is not a true norm, since its ball
    /// is not convex, but the same formula still measures a size.)
    P(f64),
    /// The largest of |x|, |y|, |z|: what the p-norm comes to as p grows (p = ∞).
    Max,
}

impl Norm {
    /// The three classic ways of measuring, in the order the table lists them.
    const CLASSIC: [Norm; 3] = [Norm::P(1.0), Norm::P(2.0), Norm::Max];

    fn size(self, x: DVec3) -> f64 {
        let c = x.abs();
        let big = c.max_element();
        match self {
            Norm::Max => big,
            // (scaled by the largest coordinate first, so that a large p cannot overflow)
            Norm::P(p) if big > 0.0 && big.is_finite() => big * ((c.x / big).powf(p) + (c.y / big).powf(p) + (c.z / big).powf(p)).powf(1.0 / p),
            Norm::P(_) => big,
        }
    }

    /// Is the ball convex (p ≥ 1)? Then everything between two smallest answers is one too.
    fn convex(self) -> bool {
        !matches!(self, Norm::P(p) if p < 1.0)
    }

    fn name(self) -> String {
        match self {
            Norm::Max => "max-norm".to_string(),
            Norm::P(1.0) => "1-norm".to_string(),
            Norm::P(2.0) => "2-norm".to_string(),
            Norm::P(p) => format!("$p = {}$", short(p)),
        }
    }

    /// The colour of its ball: pink for p = 1, cyan for 2, yellow for ∞, purple at
    /// 0.5, and blends between them.
    fn color(self) -> Color {
        match self {
            Norm::Max => pal::YELLOW,
            Norm::P(1.0) => pal::PINK,
            Norm::P(2.0) => pal::CYAN,
            Norm::P(p) if p < 1.0 => pal::PURPLE.mix(pal::PINK, ((p - 0.5) / 0.5) as f32),
            Norm::P(p) if p < 2.0 => pal::PINK.mix(pal::CYAN, (p - 1.0) as f32),
            Norm::P(p) => pal::CYAN.mix(pal::YELLOW, (1.0 - 2.0 / p) as f32),
        }
    }
}

/// One or two equations in two or three unknowns: equation k says
/// `rows[k] · (x, y, z) = rhs[k]`. With two unknowns there is no z, and a third
/// coefficient is ignored.
#[derive(Clone, Copy, Debug, PartialEq)]
struct System {
    unknowns: usize,
    equations: usize,
    rows: [DVec3; 2],
    rhs: [f64; 2],
}

impl System {
    fn flat(a: [f64; 2], b: f64) -> System {
        System { unknowns: 2, equations: 1, rows: [DVec3::new(a[0], a[1], 0.0), DVec3::ZERO], rhs: [b, 0.0] }
    }

    fn space(equations: usize, eqs: [([f64; 3], f64); 2]) -> System {
        System { unknowns: 3, equations, rows: [DVec3::from(eqs[0].0), DVec3::from(eqs[1].0)], rhs: [eqs[0].1, eqs[1].1] }
    }

    /// Row `k` as it counts: without a third coefficient in the plane.
    fn row(&self, k: usize) -> DVec3 {
        let r = self.rows[k];
        if self.unknowns == 2 {
            DVec3::new(r.x, r.y, 0.0)
        } else {
            r
        }
    }
}

/// What the equations come to, once those that say nothing (0 = 0) are dropped and
/// two that say the same thing are counted once.
#[derive(Clone, Copy, Debug, PartialEq)]
enum Reduced {
    /// No point satisfies them, and why.
    Contradiction(Why),
    /// Every point does (all coefficients and right-hand sides are 0).
    Everything,
    /// One equation (`twice`: the two given are the same plane).
    One { row: DVec3, b: f64, twice: bool },
    /// Two independent equations: their answers are a line.
    Two { rows: [DVec3; 2], rhs: [f64; 2] },
}

#[derive(Clone, Copy, Debug, PartialEq)]
enum Why {
    /// An equation with no coefficients says 0 = b, with b not 0.
    ZeroIsNotB,
    /// Two parallel planes, apart.
    Parallel,
}

fn reduce(sys: &System) -> Reduced {
    let mut kept: Vec<(DVec3, f64)> = Vec::new();
    for k in 0..sys.equations.min(2) {
        let (r, b) = (sys.row(k), sys.rhs[k]);
        if r.length() <= 1e-12 {
            if b != 0.0 {
                return Reduced::Contradiction(Why::ZeroIsNotB);
            }
        } else {
            kept.push((r, b));
        }
    }
    match kept[..] {
        [] => Reduced::Everything,
        [(row, b)] => Reduced::One { row, b, twice: false },
        [(r1, b1), (r2, b2), ..] => {
            if r1.cross(r2).length() > 1e-9 * r1.length() * r2.length() {
                return Reduced::Two { rows: [r1, r2], rhs: [b1, b2] };
            }
            // the second row is a multiple of the first: the same plane, or a parallel one
            let k = r1.dot(r2) / r1.length_squared();
            if (b2 - k * b1).abs() <= 1e-9 * (b2.abs() + (k * b1).abs()) {
                Reduced::One { row: r1, b: b1, twice: true }
            } else {
                Reduced::Contradiction(Why::Parallel)
            }
        }
    }
}

/// The smallest answers of a system in one norm.
#[derive(Clone, Debug, PartialEq)]
enum Smallest {
    /// There is no answer at all.
    Nothing,
    /// The least size, and the points that have it: one, or several in a tie. With a
    /// convex ball (p ≥ 1) every point of the segment or polygon they span has it
    /// too; below p = 1 only these points do.
    At { size: f64, points: Vec<DVec3> },
}

impl Smallest {
    /// The answer, if there is exactly one.
    fn point(&self) -> Option<DVec3> {
        match self {
            Smallest::At { points, .. } if points.len() == 1 => Some(points[0]),
            _ => None,
        }
    }

    fn size(&self) -> Option<f64> {
        match self {
            Smallest::At { size, .. } => Some(*size),
            Smallest::Nothing => None,
        }
    }

    fn points(&self) -> &[DVec3] {
        match self {
            Smallest::At { points, .. } => points,
            Smallest::Nothing => &[],
        }
    }
}

fn smallest(sys: &System, norm: Norm) -> Smallest {
    let origin = || Smallest::At { size: 0.0, points: vec![DVec3::ZERO] };
    match reduce(sys) {
        Reduced::Contradiction(_) => Smallest::Nothing,
        Reduced::Everything => origin(),
        Reduced::One { row, b, .. } => one_equation(row, b, sys.unknowns, norm),
        Reduced::Two { rows: [r1, r2], rhs: [b1, b2] } => {
            if b1 == 0.0 && b2 == 0.0 {
                return origin();
            }
            // the line of answers: its point nearest 0 (in the plane of the two rows,
            // where both equations hold) and its direction, across both rows
            let d = r1.cross(r2);
            let x0 = (r2.cross(d) * b1 + d.cross(r1) * b2) / d.length_squared();
            on_line(x0, d.normalize(), norm)
        }
    }
}

/// The smallest answers of `a · x = b` (a not zero) in `n` unknowns, from Hölder's
/// inequality: |b| = |a · x| ≤ ‖a‖_q ‖x‖_p with 1/p + 1/q = 1, so no answer is smaller
/// than |b| / ‖a‖_q, and the answers that reach it are written down directly.
fn one_equation(a: DVec3, b: f64, n: usize, norm: Norm) -> Smallest {
    if b == 0.0 {
        return Smallest::At { size: 0.0, points: vec![DVec3::ZERO] };
    }
    let big = (0..n).map(|i| a[i].abs()).fold(0.0, f64::max);
    let axis = |i: usize| {
        let mut x = DVec3::ZERO;
        x[i] = b / a[i];
        x
    };
    let points = match norm {
        // q = ∞: the bound is |b| / max |aᵢ|, reached on the axis of the largest
        // coefficient. Below p = 1 those axis points are smallest too: the size is then
        // concave between the axes, so its least is at one of them.
        Norm::P(p) if p <= 1.0 => (0..n).filter(|&i| a[i].abs() >= big * (1.0 - 1e-9)).map(axis).collect(),
        // 1 < q < ∞: xᵢ is a multiple of sign(aᵢ) |aᵢ|^(q − 1), the one unique answer
        Norm::P(p) => {
            let q = p / (p - 1.0);
            let w = DVec3::from_array(std::array::from_fn(|i| if i < n { a[i].signum() * (a[i].abs() / big).powf(q - 1.0) } else { 0.0 }));
            let total: f64 = (0..n).map(|i| (a[i].abs() / big).powf(q)).sum();
            vec![w * (b / (big * total))]
        }
        // q = 1: every coordinate the same size t = |b| / Σ|aᵢ|, with the sign that
        // helps. A coordinate without a coefficient can be anything up to t either way:
        // then a whole edge (or face) of the cube touches.
        Norm::Max => {
            let t = b / (0..n).map(|i| a[i].abs()).sum::<f64>();
            let free: Vec<usize> = (0..n).filter(|&i| a[i].abs() <= 1e-12 * big).collect();
            let fixed = DVec3::from_array(std::array::from_fn(|i| if i < n && !free.contains(&i) { t * a[i].signum() } else { 0.0 }));
            // the corners of that edge or face, in order round it
            let signs: &[[f64; 2]] = match free.len() {
                0 => &[[0.0, 0.0]],
                1 => &[[1.0, 0.0], [-1.0, 0.0]],
                _ => &[[1.0, 1.0], [1.0, -1.0], [-1.0, -1.0], [-1.0, 1.0]],
            };
            signs
                .iter()
                .map(|s| {
                    let mut x = fixed;
                    for (k, &i) in free.iter().take(2).enumerate() {
                        x[i] = s[k] * t.abs();
                    }
                    x
                })
                .collect()
        }
    };
    let size = points.first().map_or(0.0, |&x| norm.size(x));
    Smallest::At { size, points }
}

/// The smallest points of the line `x0 + t d` (d a unit vector).
fn on_line(x0: DVec3, d: DVec3, norm: Norm) -> Smallest {
    let at = |t: f64| x0 + d * t;
    let mut points: Vec<(f64, DVec3)> = match norm {
        // Between the places where a coordinate is zero the 1-norm is straight and the
        // p-norm below 1 is concave (as a power of the size), and both grow without
        // end: the least is at one of those places.
        Norm::P(p) if p <= 1.0 => (0..3)
            .filter(|&i| d[i] != 0.0)
            .map(|i| {
                let t = -x0[i] / d[i];
                let mut x = at(t);
                x[i] = 0.0;
                (t, x)
            })
            .collect(),
        // The size is strictly convex along the line, so its slope crosses zero once:
        // found by halving. No answer is further from x0 than 2 ‖x0‖ / ‖d‖ (the
        // triangle inequality), which brackets it.
        Norm::P(p) => {
            let slope = |t: f64| {
                let y = at(t);
                (0..3).map(|i| d[i] * y[i].signum() * y[i].abs().powf(p - 1.0)).sum::<f64>()
            };
            let reach = 2.0 * norm.size(x0) / norm.size(d);
            let (mut lo, mut hi) = (-reach, reach);
            for _ in 0..200 {
                let mid = 0.5 * (lo + hi);
                if slope(mid) > 0.0 {
                    hi = mid;
                } else {
                    lo = mid;
                }
                if hi - lo <= 1e-15 * reach {
                    break;
                }
            }
            let t = 0.5 * (lo + hi);
            vec![(t, at(t))]
        }
        // The largest coordinate is straight between the places where two coordinates
        // are the same size (yᵢ = ±yⱼ): the least is at one of those.
        Norm::Max => {
            let mut ts = Vec::new();
            for i in 0..3 {
                for j in i + 1..3 {
                    if d[i] != d[j] {
                        ts.push(-(x0[i] - x0[j]) / (d[i] - d[j]));
                    }
                    if d[i] != -d[j] {
                        ts.push(-(x0[i] + x0[j]) / (d[i] + d[j]));
                    }
                }
            }
            ts.into_iter().map(|t| (t, at(t))).collect()
        }
    };
    let least = points.iter().map(|(_, x)| norm.size(*x)).fold(f64::INFINITY, f64::min);
    if !least.is_finite() {
        return Smallest::At { size: norm.size(x0), points: vec![x0] };
    }
    // every candidate as small as the least, once each, in order along the line
    let scale = x0.length().max(least).max(1e-300);
    points.retain(|(_, x)| norm.size(*x) <= least * (1.0 + 1e-9) + 1e-15 * scale);
    points.sort_by(|a, b| a.0.total_cmp(&b.0));
    points.dedup_by(|a, b| (a.1 - b.1).length() <= 1e-9 * scale);
    // with a convex ball the ties fill the segment between the first and the last
    if norm.convex() && points.len() > 2 {
        points.drain(1..points.len() - 1);
    }
    Smallest::At { size: least, points: points.into_iter().map(|(_, x)| x).collect() }
}

/// Is the coordinate `v` of a point zero (as far as the arithmetic can tell)?
fn is_zero(v: f64, x: DVec3) -> bool {
    v.abs() <= 1e-12 * x.abs().max_element()
}

/// How many of the first `n` coordinates of `x` are zero.
fn zeros(x: DVec3, n: usize) -> usize {
    (0..n).filter(|&i| is_zero(x[i], x)).count()
}

/// The minimum-norm answer written as the pseudo-inverse writes it, x = Aᵀ(AAᵀ)⁻¹b:
/// for the read-outs, and for the tests to hold the 2-norm's answer against.
fn pseudo_inverse(sys: &System) -> Option<DVec3> {
    match reduce(sys) {
        Reduced::Contradiction(_) => None,
        Reduced::Everything => Some(DVec3::ZERO),
        Reduced::One { row, b, .. } => Some(row * (b / row.length_squared())),
        Reduced::Two { rows: [r1, r2], rhs: [b1, b2] } => {
            // AAᵀ is the 2×2 matrix of dot products of the rows
            let (g11, g12, g22) = (r1.dot(r1), r1.dot(r2), r2.dot(r2));
            let det = g11 * g22 - g12 * g12;
            let y = DVec2::new(g22 * b1 - g12 * b2, g11 * b2 - g12 * b1) / det;
            Some(r1 * y.x + r2 * y.y)
        }
    }
}

// ----- words and numbers ----------------------------------------------------------------

/// A number to at most two decimals, without trailing zeros: 2, 0.5, 1.23.
fn short(v: f64) -> String {
    let text = format!("{:.2}", v);
    let text = if text.contains('.') { text.trim_end_matches('0').trim_end_matches('.').to_string() } else { text };
    if text == "-0" {
        "0".to_string()
    } else {
        text.replace('-', "−")
    }
}

/// An equation as it would be written by hand: "x + 2y = 2", "2x − y + 0.5z = 1".
fn equation(row: DVec3, b: f64, n: usize) -> String {
    let mut text = String::new();
    for (i, name) in ["x", "y", "z"].iter().enumerate().take(n) {
        let c = (row[i] * 100.0).round() / 100.0;
        if c == 0.0 {
            continue;
        }
        let size = if c.abs() == 1.0 { String::new() } else { short(c.abs()) };
        let sign = match (text.is_empty(), c < 0.0) {
            (true, true) => "−",
            (true, false) => "",
            (false, true) => " − ",
            (false, false) => " + ",
        };
        text += &format!("{sign}{size}{name}");
    }
    if text.is_empty() {
        text.push('0');
    }
    format!("{text} = {}", short(b))
}

/// A point's coordinates, "(0, 1.000)": zeros written as a plain 0.
fn coords(x: DVec3, n: usize) -> String {
    let parts: Vec<String> = (0..n).map(|i| if is_zero(x[i], x) { "0".to_string() } else { fmt(x[i] as f32) }).collect();
    format!("({})", parts.join(", "))
}

/// Which coordinates of `x` are zero, as a sentence.
fn zero_sentence(x: DVec3, n: usize) -> String {
    let names: Vec<&str> = (0..n).filter(|&i| is_zero(x[i], x)).map(|i| ["x", "y", "z"][i]).collect();
    match names[..] {
        [] => "No coordinate is $0$.".to_string(),
        [one] => format!("${one}$ is exactly $0$."),
        [one, two] => format!("${one}$ and ${two}$ are exactly $0$."),
        _ => "Every coordinate is $0$.".to_string(),
    }
}

// ----- the scene ------------------------------------------------------------------------

struct Underdet {
    /// Three unknowns (in space) rather than two (in the plane).
    space: bool,
    /// The equation in the plane.
    flat: System,
    /// The two dots on its line, which the visitor drags (they define the line).
    dots: [DVec2; 2],
    /// The equations in space: one or two of them.
    solid: System,
    /// The p of the p-norm, unless `max` (p = ∞).
    p: f32,
    max: bool,
    /// The ball's size as a share of the size at which it first touches the answers.
    size: f32,
    /// How far the growing animation has got (0 to 1), and whether it is playing.
    growth: f32,
    growing: bool,
    /// Show the other classic norms' answers beside the chosen one.
    compare: bool,
    /// The ball in space, made when it is first drawn, and the norm it was filled for.
    surf: Option<Surface>,
    surf_norm: Option<Norm>,
    /// Whether the canvas is a phone's, held upright (the table goes below the
    /// picture), and whether the camera should move to a new view.
    narrow: bool,
    new_view: bool,
}

impl Underdet {
    /// (Nothing here needs the graphics device, so the tests can make one too.)
    fn create() -> Self {
        let mut s = Underdet {
            space: false,
            flat: System::flat(FLAT_START.0, FLAT_START.1),
            dots: [DVec2::ZERO; 2],
            solid: System::space(1, SPACE_START),
            p: 1.0,
            max: false,
            size: 0.0,
            growth: 0.0,
            growing: true,
            compare: true,
            surf: None,
            surf_norm: None,
            narrow: false,
            new_view: false,
        };
        s.place_dots();
        s
    }

    /// Set up one of the experiments in `TRIALS`, and grow the ball from nothing.
    fn set_up(&mut self, k: usize) {
        let (space, p) = match k {
            0 => (false, 2.0),
            1 | 2 => (false, 1.0),
            _ => (true, 1.0),
        };
        (self.p, self.max, self.compare) = (p, false, true);
        self.space = space;
        match k {
            0 | 1 => self.flat = System::flat(FLAT_START.0, FLAT_START.1),
            2 => self.flat = System::flat([1.0, 1.0], 1.5),
            3 => self.solid = System::space(1, SPACE_START),
            _ => self.solid = System::space(2, SPACE_START),
        }
        self.place_dots();
        // (and the camera back to where the lesson is seen best)
        self.new_view = true;
        self.replay();
    }

    fn replay(&mut self) {
        (self.size, self.growth, self.growing) = (0.0, 0.0, true);
    }

    /// Move the growing animation on by `dt` seconds of scene time. It slows as the
    /// ball comes up to the answers, so that the moment of touching can be seen.
    fn advance(&mut self, dt: f32) {
        if self.growing {
            self.growth = (self.growth + dt / GROW_TIME).min(1.0);
            let g = self.growth;
            self.size = 1.0 - (1.0 - g) * (1.0 - g);
            if g >= 1.0 {
                (self.size, self.growing) = (1.0, false);
            }
        }
    }

    fn norm(&self) -> Norm {
        if self.max {
            Norm::Max
        } else {
            Norm::P(self.p as f64)
        }
    }

    fn system(&self) -> &System {
        if self.space {
            &self.solid
        } else {
            &self.flat
        }
    }

    fn n(&self) -> usize {
        self.system().unknowns
    }

    /// Has the ball grown to the touching size (or beyond)?
    fn touching(&self) -> bool {
        self.size >= 1.0 - 1e-6
    }

    /// The norms the table and the picture show: the chosen one first, then those of
    /// the classic three that are not it (all of them only if `compare`).
    fn shown_norms(&self) -> Vec<Norm> {
        let mut norms = vec![self.norm()];
        if self.compare {
            norms.extend(Norm::CLASSIC.into_iter().filter(|&c| c != self.norm()));
        }
        norms
    }

    /// The direction along the line in the plane, and its point nearest 0.
    fn flat_line(&self) -> Option<(DVec2, DVec2)> {
        let a = self.flat.row(0).truncate();
        let len = a.length();
        (len > 1e-12).then(|| (DVec2::new(-a.y, a.x) / len, a * (self.flat.rhs[0] / (len * len))))
    }

    /// Put the two dots on the line, either side of its point nearest 0.
    fn place_dots(&mut self) {
        self.dots = match self.flat_line() {
            Some((d, foot)) => [foot + d * DOT_SPREAD, foot - d * DOT_SPREAD],
            None => [DVec2::new(-1.0, 1.5), DVec2::new(1.5, 0.3)],
        };
    }

    /// Dot `k` has been dragged to `q`: the line now runs through it and the other
    /// dot. Its coefficients keep their length (the equation changes as little as it
    /// can), and turn the short way.
    fn move_dot(&mut self, k: usize, q: DVec2) {
        let other = self.dots[1 - k];
        let a = self.flat.row(0).truncate();
        let len = if a.length() > 1e-12 { a.length() } else { 1.0 };
        let along = q - other;
        let new_a = if along.length() > 1e-6 {
            let n = DVec2::new(-along.y, along.x).normalize() * len;
            if n.dot(a) < 0.0 {
                -n
            } else {
                n
            }
        } else if a.length() > 1e-12 {
            // the dots on top of each other say nothing about the direction: keep it
            a
        } else {
            DVec2::new(0.0, len)
        };
        self.flat.rows[0] = new_a.extend(0.0);
        self.flat.rhs[0] = new_a.dot(q);
        self.dots[k] = q;
        // (the other dot is on the new line already, unless the two were on top of
        // each other: then it is moved onto it)
        let off = (new_a.dot(other) - self.flat.rhs[0]) / (len * len);
        self.dots[1 - k] = other - new_a * off;
    }

    fn the_view(&self) -> View {
        match (self.space, self.narrow) {
            (false, false) => View::Fit { center: Vec2::new(0.35, 0.45), half: Vec2::new(1.95, 1.95) },
            // (lifted, to leave the lower part of the screen to the table of answers)
            (false, true) => View::Fit { center: Vec2::new(0.4, -0.3), half: Vec2::new(1.65, 1.65) },
            // In space, from where the lesson is seen best. One equation: from a little
            // above the plane, nearly along it, so that the octahedron is seen touching
            // it from below at its top corner. Two: nearly along the edge the line
            // touches, from a little above the plane through the line that only touches
            // the octahedron (5x + 8y + 8z = 10 for the experiment's equations), so that
            // the edge is seen end-on, as a corner, and the line rests on it: the same
            // picture as a diamond's corner touching a line in the plane.
            (true, narrow) => {
                let (yaw, pitch, target) = if self.solid.equations == 1 { (-0.25, 0.28, Vec3::new(0.0, 0.2, 0.35)) } else { (-1.8, 0.72, Vec3::new(0.36, 0.575, 0.575)) };
                let (target, dist) = if narrow { (target - Vec3::Z * 0.9, 13.5) } else { (target, 7.2) };
                View::Orbit { target, yaw, pitch, dist }
            }
        }
    }

    /// What the picture shows, in a sentence or two, for the caption under it.
    fn story(&self) -> String {
        let sys = self.system();
        let n = sys.unknowns;
        let norm = self.norm();
        let set = match (n, reduce(sys)) {
            (2, _) => "line",
            (_, Reduced::Two { .. }) => "line",
            _ => "plane",
        };
        match reduce(sys) {
            Reduced::Contradiction(Why::ZeroIsNotB) => return "An equation with every coefficient $0$ says $0 = b$, and $b$ is not $0$: no point solves it, so there is no answer to choose from.".to_string(),
            Reduced::Contradiction(Why::Parallel) => return "The two planes are parallel and apart: no point lies on both, so there is no answer to choose from.".to_string(),
            Reduced::Everything => return "Every coefficient is $0$ and so is every $b$: every point is an answer, and the smallest is $0$ itself.".to_string(),
            _ => {}
        }
        let ans = smallest(sys, norm);
        let Smallest::At { size, points } = &ans else { return String::new() };
        let twice = if matches!(reduce(sys), Reduced::One { twice: true, .. }) { "The two equations describe the same plane. " } else { "" };
        if *size == 0.0 {
            return format!("{twice}$b = 0$: the {set} passes through $0$, so $0$ itself is the smallest answer, however size is measured.");
        }
        let name = norm.name();
        let r = *size * self.size as f64;
        if !self.touching() {
            return format!("{twice}The {name} ball, growing: size ${}$. No answer is this small yet.", fmt(r as f32));
        }
        if self.size > 1.0 + 1e-6 {
            let bright = if set == "line" { "the bright part of the line is every answer of size up to" } else { "it pokes through the plane, round every answer of size up to" };
            return format!("{twice}The ball is larger than the touching size (${}$): {bright} ${}$.", fmt(*size as f32), fmt(r as f32));
        }
        match points[..] {
            [x] => format!("{twice}The {name} ball first touches the {set} at ${}$: the smallest answer in this norm. {}", coords(x, n), zero_sentence(x, n)),
            [x, y] if norm.convex() => format!("{twice}A tie: the whole segment from ${}$ to ${}$ touches at once, and every point of it is a smallest answer.", coords(x, n), coords(y, n)),
            _ if norm.convex() => format!("{twice}A tie: a whole face of the ball touches at once, and every point of it is a smallest answer."),
            _ => {
                let list: Vec<String> = points.iter().map(|&x| format!("${}$", coords(x, n))).collect();
                format!("{twice}A tie: the ball touches at {} separate points at once, {}.", points.len(), list.join(" and "))
            }
        }
    }
}

// ----- drawing --------------------------------------------------------------------------

/// The boundary of the ball of size `r` in the plane, `k` points round it (k a
/// multiple of 4, so that the corners of the diamond are among them).
fn outline(norm: Norm, r: f64, k: usize) -> Vec<Vec2> {
    match norm {
        Norm::Max => [(1.0, 1.0), (-1.0, 1.0), (-1.0, -1.0), (1.0, -1.0)].map(|(x, y)| Vec2::new(x, y) * r as f32).to_vec(),
        Norm::P(p) => (0..k)
            .map(|i| {
                let (s, c) = (i as f64 / k as f64 * std::f64::consts::TAU).sin_cos();
                // |x|ᵖ + |y|ᵖ = cos² + sin² = 1
                let e = 2.0 / p;
                (DVec2::new(c.signum() * c.abs().powf(e), s.signum() * s.abs().powf(e)) * r).as_vec2()
            })
            .collect(),
    }
}

/// A point of the unit ball's surface in space, for the grid parameters `u` (round
/// the z axis) and `v` (from the bottom to the top), both 0 to 1. Below p = 2 it is a
/// superellipsoid, which crowds its points towards the axes, where the corners and
/// spikes are; the grid's lines include the coordinate planes, so that at p = 1 every
/// cell lies on one face of the octahedron. From p = 2 on, an even grid on the sphere
/// pushed out along its rays onto the ball: the superellipsoid would leave the
/// middles of the nearly flat faces with almost no points, which shows.
fn unit_ball_point(p: f32, u: f32, v: f32) -> Vec3 {
    let (su, cu) = ((u * 2.0 - 1.0) * PI).sin_cos();
    let (sv, cv) = ((v - 0.5) * PI).sin_cos();
    if p < 2.0 {
        let e = 2.0 / p;
        let pow = |c: f32| c.signum() * c.abs().powf(e);
        Vec3::new(pow(cv) * pow(cu), pow(cv) * pow(su), pow(sv))
    } else {
        let x = Vec3::new(cv * cu, cv * su, sv);
        x / Norm::P(p as f64).size(x.as_dvec3()).max(1e-12) as f32
    }
}

/// The max-norm's ball, the cube of half-width `r`: its faces, each with its own
/// tone as if lit from above and to one side, drawn from the back to the front
/// because they are see-through.
fn cube(f: &mut Frame, r: f32, color: Color) {
    let light = Vec3::new(0.45, -0.3, 0.85).normalize();
    let eye = f.cam.eye;
    let mut faces: Vec<(Vec3, Vec3, Vec3)> = Vec::new();
    for k in 0..3 {
        let (n, a, b) = (Vec3::AXES[k], Vec3::AXES[(k + 1) % 3], Vec3::AXES[(k + 2) % 3]);
        faces.push((n * r, a * r, b * r));
        faces.push((-n * r, a * r, b * r));
    }
    faces.sort_by(|p, q| q.0.distance(eye).total_cmp(&p.0.distance(eye)));
    for (c, a, b) in faces {
        let tone = 0.45 + 0.75 * c.normalize_or_zero().dot(light).max(0.0);
        let shade = Color::from_linear(color.to_linear() * tone).a(0.3);
        f.dl.tris_alpha.quad(c - a - b, c + a - b, c + a + b, c - a + b, shade);
    }
}

/// The parts of the segment from `a` to `b` (as t from 0 to 1) inside the
/// rectangle `lo..hi`, if any.
fn clip(a: Vec2, b: Vec2, lo: Vec2, hi: Vec2) -> Option<(f32, f32)> {
    let (mut t0, mut t1) = (0.0f32, 1.0f32);
    for k in 0..2 {
        let (p, d) = (a[k], b[k] - a[k]);
        if d.abs() < 1e-12 {
            if p < lo[k] || p > hi[k] {
                return None;
            }
        } else {
            let (s0, s1) = ((lo[k] - p) / d, (hi[k] - p) / d);
            t0 = t0.max(s0.min(s1));
            t1 = t1.min(s0.max(s1));
        }
    }
    (t0 < t1).then_some((t0, t1))
}

/// The words of a text, each formula whole however many spaces it has (a formula is
/// not broken across lines).
fn words(text: &str) -> Vec<&str> {
    let bytes = text.as_bytes();
    let (mut words, mut start, mut inside, mut i) = (Vec::new(), 0, false, 0);
    while i < bytes.len() {
        match bytes[i] {
            // (an escaped dollar is no edge of a formula)
            b'\\' => i += 1,
            b'$' => inside = !inside,
            b' ' if !inside => {
                words.push(&text[start..i]);
                start = i + 1;
            }
            _ => {}
        }
        i += 1;
    }
    words.push(&text[start.min(text.len())..]);
    words
}

/// Text broken into lines no wider than `room` logical pixels.
fn wrap(text: &str, size: f32, room: f32) -> Vec<String> {
    let mut lines = vec![String::new()];
    // (the last line's width so far: each word, formulas and all, is measured once)
    let mut width = 0.0;
    for word in words(text) {
        let w = text_width(word, size);
        let last = lines.last_mut().expect("there is always a line");
        if last.is_empty() {
            last.push_str(word);
            width = w;
        } else if width + w + size * 0.47 > room {
            lines.push(word.to_string());
            width = w;
        } else {
            last.push(' ');
            last.push_str(word);
            width += size * 0.47 + w;
        }
    }
    lines
}

/// The size of the caption's text.
const CAPTION: f32 = 12.5;

/// The caption's lines: its text word-wrapped to the width of the view (none at all in
/// a view too small to spare the room: a phone's with the controls open over it, say).
fn caption_lines(view: Vec2, text: &str, size: f32) -> Vec<String> {
    if view.y < 300.0 || view.x < 300.0 || text.is_empty() {
        return Vec::new();
    }
    wrap(text, size, view.x - 36.0)
}

/// The caption along the bottom of the view, in these lines.
fn caption(f: &mut Frame, lines: &[String], size: f32) {
    let view = f.view_size();
    let step = size + 7.0;
    for (k, line) in lines.iter().enumerate() {
        let up = (lines.len() - 1 - k) as f32 * step;
        f.caption_px(Vec2::new(18.0, view.y - 18.0 - up), line.as_str(), size, pal::DIM, Align2::LEFT_BOTTOM);
    }
}

/// How much of the bottom of the view the caption in these lines takes.
fn caption_height(lines: &[String], size: f32) -> f32 {
    if lines.is_empty() {
        0.0
    } else {
        lines.len() as f32 * (size + 7.0) + 18.0
    }
}

/// The numbers along the axes of the flat picture, as `Frame::axes2` writes them,
/// but leaving out any that would fall under the table or the caption (`axes2`
/// writes them over everything).
fn ticks(f: &mut Frame, keep_clear: &[(Vec2, Vec2)]) {
    let (lo, hi) = f.cam.flat_bounds();
    let view = f.view_size();
    let wpp = (hi.y - lo.y) / view.y;
    if !(wpp.is_finite() && wpp > 0.0 && view.x >= 1.0) {
        return;
    }
    let step = nice_step(wpp * 95.0);
    let margin = 16.0 * wpp;
    // (max then min rather than clamp: in a view narrower than the margins the
    // bounds cross)
    let y_anchor = 0.0f32.max(lo.y + margin * 1.6).min(hi.y - margin * 0.4);
    let x_anchor = 0.0f32.max(lo.x + margin * 0.4).min(hi.x - margin * 2.4);
    let edge = 20.0 * wpp;
    let values = |lo: f32, hi: f32| {
        let first = (lo / step).ceil();
        let count = (((hi - lo) / step).floor().max(0.0) as usize).min(400);
        (0..=count).map(move |i| (first + i as f32) * step).filter(move |v| *v <= hi && v.abs() > step * 0.01)
    };
    let free = |f: &Frame, p: Vec2, off: Vec2| {
        on_screen(f, p.extend(0.0)).is_some_and(|s| {
            let s = s + off;
            !keep_clear.iter().any(|(a, b)| s.x > a.x - 24.0 && s.x < b.x + 4.0 && s.y > a.y - 10.0 && s.y < b.y + 10.0)
        })
    };
    for x in values(lo.x, hi.x).filter(|x| *x > lo.x + edge && *x < hi.x - edge) {
        let p = Vec2::new(x, y_anchor);
        if free(f, p, Vec2::new(0.0, 11.0)) {
            f.text_off(p, Vec2::new(0.0, 5.0), fmt_tick(x, step), 12.0, pal::DIM, Align2::CENTER_TOP);
        }
    }
    for y in values(lo.y, hi.y).filter(|y| *y > lo.y + edge * 0.6 && *y < hi.y - edge * 0.6) {
        let p = Vec2::new(x_anchor, y);
        if free(f, p, Vec2::new(14.0, 0.0)) {
            f.text_off(p, Vec2::new(6.0, 0.0), fmt_tick(y, step), 12.0, pal::DIM, Align2::LEFT_CENTER);
        }
    }
}

/// A direction in the plane of a flat scene as it points on the screen (y down).
fn screen_dir(d: Vec2) -> Vec2 {
    Vec2::new(d.x, -d.y).normalize_or(Vec2::X)
}

/// How to align a label put beside something in the screen direction `n` from it, so
/// that the label reaches away from it.
fn facing(n: Vec2) -> Align2 {
    match (n.x > 0.35, n.x < -0.35, n.y > 0.35, n.y < -0.35) {
        (true, _, true, _) => Align2::LEFT_TOP,
        (true, _, _, true) => Align2::LEFT_BOTTOM,
        (true, _, _, _) => Align2::LEFT_CENTER,
        (_, true, true, _) => Align2::RIGHT_TOP,
        (_, true, _, true) => Align2::RIGHT_BOTTOM,
        (_, true, _, _) => Align2::RIGHT_CENTER,
        (_, _, true, _) => Align2::CENTER_TOP,
        _ => Align2::CENTER_BOTTOM,
    }
}

/// The places on the screen already given to labels (and to anything labels must not
/// cover), in logical pixels: each label goes at the first of its candidate spots that
/// is inside the view and clear of them.
struct Board {
    view: Vec2,
    taken: Vec<(Vec2, Vec2)>,
}

impl Board {
    fn new(view: Vec2, keep_clear: &[(Vec2, Vec2)]) -> Board {
        Board { view, taken: keep_clear.to_vec() }
    }

    /// The box a label `wh` in size (with its margin) takes, anchored at `at` with
    /// `align`.
    fn box_of(at: Vec2, align: Align2, wh: Vec2) -> (Vec2, Vec2) {
        let shift = |a: egui::Align| match a {
            egui::Align::Min => 0.0,
            egui::Align::Center => 0.5,
            egui::Align::Max => 1.0,
        };
        let min = at - wh * Vec2::new(shift(align.x()), shift(align.y()));
        (min, min + wh)
    }

    /// Put the label at the first clear spot; if none is, at the first spot inside the
    /// view if it is `needed`, and otherwise nowhere. On a dark plate if it is
    /// `needed`: a label that must be there may have to lie over lines.
    fn place(&mut self, f: &mut Frame, spots: &[(Vec2, Align2)], text: impl Into<String>, size: f32, color: Color, needed: bool) {
        let text = text.into();
        // (measured once: a formula is worked out to be measured)
        let wh = Vec2::new(text_width(&text, size) + 6.0, size + 6.0);
        let clear = |b: &(Vec2, Vec2), taken: &[(Vec2, Vec2)]| {
            b.0.x >= 2.0 && b.0.y >= 2.0 && b.1.x <= self.view.x - 2.0 && b.1.y <= self.view.y - 2.0 && taken.iter().all(|t| b.1.x <= t.0.x || b.0.x >= t.1.x || b.1.y <= t.0.y || b.0.y >= t.1.y)
        };
        let found = spots.iter().find(|(at, align)| clear(&Self::box_of(*at, *align, wh), &self.taken));
        // (a label that must be there goes at the first spot inside the view, if no
        // spot is clear of the others)
        let inside = || spots.iter().find(|(at, align)| clear(&Self::box_of(*at, *align, wh), &[]));
        let Some(&(at, align)) = found.or_else(|| inside().filter(|_| needed)) else { return };
        self.taken.push(Self::box_of(at, align, wh));
        if needed {
            f.caption_px(at, text, size, color, align);
        } else {
            f.text_px(at, text, size, color, align);
        }
    }
}

/// The table of answers: one row per norm (its colour, name, coordinates and least
/// size), the zero coordinates picked out. Its size in logical pixels.
fn table_size(rows: usize, n: usize) -> Vec2 {
    Vec2::new(NAME_W + COL_W * (n as f32 + 1.0) + 24.0, 34.0 + ROW_H * rows as f32)
}

const NAME_W: f32 = 82.0;
const COL_W: f32 = 62.0;
const ROW_H: f32 = 19.0;

fn draw_table(f: &mut Frame, at: Vec2, rows: &[(Norm, Smallest)], n: usize) {
    let size = table_size(rows.len(), n);
    hud_panel(f, at, at + size);
    let left = at.x + 12.0;
    let col = |k: usize| left + NAME_W + COL_W * (k as f32 + 1.0);
    let head = at.y + 15.0;
    f.text_px(Vec2::new(left, head), "smallest answer", 11.0, pal::MUTED, Align2::LEFT_CENTER);
    for (k, name) in ["$x$", "$y$", "$z$"].iter().take(n).chain(std::iter::once(&"size")).enumerate() {
        f.text_px(Vec2::new(col(k), head), *name, 11.0, pal::MUTED, Align2::RIGHT_CENTER);
    }
    for (r, (norm, ans)) in rows.iter().enumerate() {
        let y = at.y + 34.0 + ROW_H * (r as f32 + 0.5);
        let chosen = r == 0;
        hud_rect(f, Vec2::new(left, y - 5.0), Vec2::new(left + 10.0, y + 5.0), norm.color());
        f.text_px(Vec2::new(left + 16.0, y), norm.name(), 12.0, if chosen { pal::WHITE } else { pal::DIM }, Align2::LEFT_CENTER);
        let phrase = match ans {
            Smallest::Nothing => Some("no answer".to_string()),
            Smallest::At { points, .. } if points.len() == 1 => None,
            Smallest::At { points, .. } if !norm.convex() => Some(format!("a tie: {} points", points.len())),
            Smallest::At { points, .. } if points.len() == 2 => Some("a tie: a segment".to_string()),
            Smallest::At { .. } => Some("a tie: a face".to_string()),
        };
        let text_color = if chosen { pal::TEXT } else { pal::DIM };
        match (phrase, ans.point()) {
            (None, Some(x)) => {
                for k in 0..n {
                    let (text, c) = if is_zero(x[k], x) { ("$0$".to_string(), ZERO) } else { (format!("${}$", fmt(x[k] as f32)), text_color) };
                    f.text_px(Vec2::new(col(k), y), text, 12.0, c, Align2::RIGHT_CENTER);
                }
            }
            (Some(text), _) => f.text_px(Vec2::new(col(0) - COL_W + 8.0, y), text, 12.0, text_color, Align2::LEFT_CENTER),
            _ => {}
        }
        if let Some(s) = ans.size() {
            f.text_px(Vec2::new(col(n), y), format!("${}$", fmt(s as f32)), 12.0, text_color, Align2::RIGHT_CENTER);
        }
    }
}

impl Underdet {
    /// Where the table goes: the top-left corner on a wide canvas, under the picture
    /// (above the caption) on a phone held upright.
    fn table_at(&self, f: &Frame, rows: usize, caption_h: f32) -> Vec2 {
        let view = f.view_size();
        let size = table_size(rows, self.n());
        if self.narrow {
            Vec2::new(((view.x - size.x) * 0.5).max(4.0), view.y - caption_h - size.y - 10.0)
        } else {
            Vec2::new(16.0, f.top_inset + 16.0)
        }
    }

    fn draw_flat(&mut self, f: &mut Frame, answers: &[(Norm, Smallest)], keep_clear: &[(Vec2, Vec2)]) {
        let norm = self.norm();
        let mut board = Board::new(f.view_size(), keep_clear);
        for dot in self.dots {
            if let Some(p) = on_screen(f, dot.as_vec2().extend(0.0)) {
                board.taken.push((p - Vec2::splat(9.0), p + Vec2::splat(9.0)));
            }
        }
        let (lo, hi) = f.cam.flat_bounds();
        let wpp = (hi.y - lo.y) / f.view_size().y.max(1.0);
        let line = self.flat_line();

        // the other classic balls, faint, each at the size where it touches
        for (other, ans) in answers.iter().skip(1) {
            if let Some(r) = ans.size() {
                let ring: Vec<Vec3> = outline(*other, r, 360).into_iter().map(|p| p.extend(0.0)).collect();
                f.dl.lines.closed(&ring, 1.3, other.color().a(0.55));
            }
        }
        // the ball of the chosen norm, at its present size
        let (main, color) = (&answers[0].1, norm.color());
        let full = main.size().unwrap_or(0.0);
        let r = full * self.size as f64;
        if r > 0.0 {
            let ring = outline(norm, r, 360);
            for i in 0..ring.len() {
                f.dl.tris_alpha.tri(Vec3::ZERO, ring[i].extend(0.0), ring[(i + 1) % ring.len()].extend(0.0), color.a(0.12));
            }
            let closed: Vec<Vec3> = ring.iter().map(|p| p.extend(0.0)).collect();
            f.dl.lines.closed(&closed, 2.2, color);
        }

        // the line of answers, across the whole view
        let mut far_side = Vec2::Y;
        if let Some((d, foot)) = line {
            let reach = ((lo - foot.as_vec2()).length().max((hi - foot.as_vec2()).length()) + 1.0) as f64;
            let (a, b) = ((foot - d * reach).as_vec2(), (foot + d * reach).as_vec2());
            f.line(a, b, 2.4, FIRST);
            // the side of the line away from 0, where the labels go
            let normal = self.flat.row(0).truncate().normalize_or_zero().as_vec2();
            far_side = if self.flat.rhs[0] < 0.0 { -normal } else { normal };
            if far_side == Vec2::ZERO {
                far_side = Vec2::Y;
            }
            // larger than the touching size: the answers inside the ball, picked out
            if self.size > 1.0 + 1e-6 && r > 0.0 {
                if let Some((t0, t1)) = clip(a, b, lo, hi) {
                    let mut piece: Vec<Vec2> = Vec::new();
                    for i in 0..=600 {
                        let t = t0 + (t1 - t0) * i as f32 / 600.0;
                        let p = a + (b - a) * t;
                        if norm.size(p.as_dvec2().extend(0.0)) <= r {
                            piece.push(p);
                        } else if !piece.is_empty() {
                            f.path(std::mem::take(&mut piece), 5.0, color);
                        }
                    }
                    if !piece.is_empty() {
                        f.path(piece, 5.0, color);
                    }
                }
            }
            // its equation, near the end of the line in view that is further from the
            // balls, on the side away from 0
            if let Some((t0, t1)) = clip(a, b, lo, hi) {
                let (near, far) = if (a + (b - a) * t0).length() > (a + (b - a) * t1).length() { (t1, t0) } else { (t0, t1) };
                let n = screen_dir(far_side);
                let spots: Vec<(Vec2, Align2)> =
                    (1..19).filter_map(|k| on_screen(f, (a + (b - a) * (far + (near - far) * k as f32 * 0.05)).extend(0.0)).map(|p| (p + n * 12.0, facing(n)))).collect();
                board.place(f, &spots, format!("${}$", equation(self.flat.row(0), self.flat.rhs[0], 2)), 15.0, FIRST, false);
            }
        }

        // the 2-norm's answer is the foot of the perpendicular from 0: show the right angle
        let two = answers.iter().find(|(m, _)| *m == Norm::P(2.0)).and_then(|(_, a)| a.point());
        if let (Some(x2), Some((d, _))) = (two, line) {
            let x2 = x2.truncate().as_vec2();
            if x2.length() > 1e-6 && (norm == Norm::P(2.0) || self.compare) {
                f.dashed(Vec2::ZERO, x2, 1.4, Norm::P(2.0).color().a(0.8));
                right_angle(f, x2.extend(0.0), -x2.extend(0.0), d.as_vec2().extend(0.0), 13.0 * wpp, pal::WHITE.a(0.75));
            }
        }

        // where each ball touches: the other norms' answers small, in their colours,
        // and the chosen norm's bright, once its ball is there
        let mut labels: Vec<(Vec2, String, Color, bool)> = Vec::new();
        for (other, ans) in answers.iter().skip(1) {
            let pts: Vec<Vec2> = ans.points().iter().map(|x| x.truncate().as_vec2()).collect();
            if pts.len() == 2 && other.convex() {
                f.line(pts[0], pts[1], 3.0, other.color());
            }
            for &p in &pts {
                f.dot(p, 8.0, other.color());
            }
            if let Some(&p) = pts.first() {
                labels.push((p, other.name(), other.color(), false));
            }
        }
        if self.touching() {
            let pts: Vec<Vec2> = main.points().iter().map(|x| x.truncate().as_vec2()).collect();
            if pts.len() == 2 && norm.convex() {
                f.glow_path(pts.iter().copied(), 5.0, pal::WHITE.a(0.8));
                f.line(pts[0], pts[1], 4.0, pal::WHITE);
            }
            for &p in &pts {
                f.glow(p, 26.0, pal::WHITE.a(0.7));
                f.dot(p, 11.0, pal::WHITE);
            }
            for (i, x) in main.points().iter().enumerate() {
                let mut text = format!("${}$", coords(*x, 2));
                if pts.len() == 1 && zeros(*x, 2) > 0 && *x != DVec3::ZERO {
                    text = format!("{text}   {}", zero_sentence(*x, 2).trim_end_matches('.'));
                }
                labels.insert(i, (pts[i], text, pal::WHITE, true));
            }
        }
        // each on the side of the line away from 0, slid along it if need be (the
        // other norms' are left out where there is no room: the table has them)
        let (n, s) = (screen_dir(far_side), screen_dir(Vec2::new(far_side.y, -far_side.x)));
        for (p, text, c, needed) in labels {
            let Some(at) = on_screen(f, p.extend(0.0)) else { continue };
            let spots: Vec<(Vec2, Align2)> = [0.0, 1.0, -1.0, 2.0, -2.0, 3.0, -3.0].iter().map(|k| (at + n * 13.0 + s * (k * 22.0), facing(n))).collect();
            board.place(f, &spots, text, 12.5, c, needed);
        }

        // the ball's name, on its far side from where it touches
        if r > 0.0 {
            let toward = main.points().iter().fold(DVec2::ZERO, |acc, x| acc + x.truncate()).as_vec2();
            let away = -toward.normalize_or(-Vec2::Y);
            let mut spots: Vec<(Vec2, Align2)> = Vec::new();
            for turn in [0.0f32, 0.5, -0.5, 1.0, -1.0, 1.6, -1.6] {
                // the point of the rim straight out from 0 in that direction
                let dir = Vec2::from_angle(turn).rotate(away);
                let rim = dir * (r / norm.size(dir.as_dvec2().extend(0.0)).max(1e-12)) as f32;
                if let Some(at) = on_screen(f, rim.extend(0.0)) {
                    let n = screen_dir(dir);
                    spots.push((at + n * 14.0, facing(n)));
                }
            }
            board.place(f, &spots, format!("{} ball, size ${}$", norm.name(), fmt(r as f32)), 13.0, color, true);
        }

        // the dots that move the line
        for k in 0..2 {
            let mut q = self.dots[k].as_vec2();
            if f.drag2(1 + k as u32, &mut q, FIRST) {
                self.move_dot(k, q.as_dvec2());
            }
        }
        ticks(f, keep_clear);
    }

    fn draw_space(&mut self, f: &mut Frame, answers: &[(Norm, Smallest)], keep_clear: &[(Vec2, Vec2)]) {
        let norm = self.norm();
        let color = norm.color();
        let main = &answers[0].1;
        let r = main.size().unwrap_or(0.0) * self.size as f64;

        // the ball: a lit, see-through surface, and its outline in the three
        // coordinate planes (at p = 1, the octahedron's twelve edges)
        if self.surf.is_none() {
            let mut s = Surface::new(f.gfx, 129, 65);
            s.style.cmap = None;
            s.style.opacity = 0.32;
            s.style.specular = 0.25;
            s.style.wrap = [true, false];
            self.surf = Some(s);
        }
        if let (Some(surf), Norm::P(p)) = (&mut self.surf, norm) {
            if self.surf_norm != Some(norm) {
                surf.fill(|u, v| unit_ball_point(p as f32, u, v).extend(0.0).to_array());
                self.surf_norm = Some(norm);
            }
            surf.style.color = color;
            surf.style.model = Mat4::from_scale(Vec3::splat(r as f32));
            if r > 0.0 {
                f.surface(surf);
            }
        }
        if r > 0.0 {
            let rf = r as f32;
            match norm {
                Norm::Max => {
                    cube(f, rf, color);
                    for (a, b) in CUBE_EDGES {
                        f.line(Vec3::from(a) * rf, Vec3::from(b) * rf, 1.6, color.a(0.9));
                    }
                }
                Norm::P(_) => {
                    let ring = outline(norm, 1.0, 240);
                    for k in 0..3 {
                        let pts: Vec<Vec3> = ring
                            .iter()
                            .map(|p| match k {
                                0 => Vec3::new(p.x, p.y, 0.0),
                                1 => Vec3::new(p.x, 0.0, p.y),
                                _ => Vec3::new(0.0, p.x, p.y),
                            } * rf)
                            .collect();
                        f.dl.lines.closed(&pts, 1.5, color.a(0.85));
                    }
                }
            }
        }

        // the equations' planes, and the line where both hold: with one equation a
        // square of its plane round its point nearest 0; with two, a strip of each
        // round the line, so that the two can be seen crossing in it
        let reduced = reduce(&self.solid);
        let centre = pseudo_inverse(&self.solid).unwrap_or(DVec3::ZERO).as_vec3();
        let line = match reduced {
            Reduced::Two { rows: [r1, r2], .. } => Some(r1.cross(r2).normalize().as_vec3()),
            _ => None,
        };
        let mut planes: Vec<(Vec<Vec3>, String, Color)> = Vec::new();
        for k in 0..self.solid.equations {
            let row = self.solid.row(k);
            if row.length() <= 1e-12 {
                continue;
            }
            let n = row.normalize().as_vec3();
            let tint = if k == 0 { FIRST } else { SECOND };
            let (c, u, v, half) = match line {
                Some(d) => (centre, d, n.cross(d), Vec2::new(2.6, 0.9)),
                None => {
                    let u = n.any_orthonormal_vector();
                    ((row * (self.solid.rhs[k] / row.length_squared())).as_vec3(), u, n.cross(u), Vec2::splat(1.7))
                }
            };
            pane(f, c, u * half.x, v * half.y, tint, if line.is_some() { 0.07 } else { 0.08 });
            // the places its equation may be written: its corners, then along its edges
            let spots = [(1.0, 1.0), (1.0, -1.0), (-1.0, 1.0), (-1.0, -1.0), (0.5, 1.0), (0.5, -1.0), (-0.5, 1.0), (-0.5, -1.0), (0.0, 1.0), (0.0, -1.0)];
            planes.push((spots.map(|(a, b)| c + u * (a * half.x * 0.9) + v * (b * half.y * 0.9)).to_vec(), format!("${}$", equation(row, self.solid.rhs[k], 3)), tint));
        }
        if let Some(d) = line {
            f.line(centre - d * 3.6, centre + d * 3.6, 2.6, BOTH);
            // larger than the touching size: the answers inside the ball, picked out
            if self.size > 1.0 + 1e-6 && r > 0.0 {
                let mut piece: Vec<Vec3> = Vec::new();
                for i in 0..=600 {
                    let p = centre + d * (7.2 * i as f32 / 600.0 - 3.6);
                    if norm.size(p.as_dvec3()) <= r {
                        piece.push(p);
                    } else if !piece.is_empty() {
                        f.path(std::mem::take(&mut piece), 5.0, color);
                    }
                }
                if !piece.is_empty() {
                    f.path(piece, 5.0, color);
                }
            }
        }

        // the 2-norm's answer, and the right angle it makes with the answers
        let two = answers.iter().find(|(m, _)| *m == Norm::P(2.0)).and_then(|(_, a)| a.point());
        if let Some(x2) = two {
            let x2 = x2.as_vec3();
            if x2.length() > 1e-6 {
                f.dashed(Vec3::ZERO, x2, 1.4, Norm::P(2.0).color().a(0.8));
                let along = line.unwrap_or_else(|| x2.normalize().any_orthonormal_vector());
                right_angle(f, x2, -x2, along, 0.16, pal::WHITE.a(0.75));
            }
        }

        // where each ball touches
        let mut names: Vec<(Vec3, String, Color, bool)> = Vec::new();
        for (other, ans) in answers.iter().skip(1) {
            let pts: Vec<Vec3> = ans.points().iter().map(|x| x.as_vec3()).collect();
            if other.convex() && pts.len() == 2 {
                f.line(pts[0], pts[1], 3.0, other.color());
            }
            for &p in &pts {
                f.sphere(p, 0.08, other.color());
            }
            if let Some(&p) = pts.first() {
                names.push((p, other.name(), other.color(), false));
            }
        }
        if self.touching() {
            let pts: Vec<Vec3> = main.points().iter().map(|x| x.as_vec3()).collect();
            match pts.len() {
                2 if norm.convex() => f.line(pts[0], pts[1], 4.0, pal::WHITE),
                3 | 4 if norm.convex() => {
                    f.dl.tris_alpha.tri(pts[0], pts[1], pts[2], pal::WHITE.a(0.4));
                    if pts.len() == 4 {
                        f.dl.tris_alpha.tri(pts[0], pts[2], pts[3], pal::WHITE.a(0.4));
                    }
                }
                _ => {}
            }
            for &p in &pts {
                f.glow(p, 26.0, pal::WHITE.a(0.7));
                f.sphere(p, 0.11, pal::WHITE);
            }
            for (i, x) in main.points().iter().enumerate() {
                let mut text = format!("${}$", coords(*x, 3));
                if pts.len() == 1 && zeros(*x, 3) > 0 && *x != DVec3::ZERO {
                    text = format!("{text}   {}", zero_sentence(*x, 3).trim_end_matches('.'));
                }
                names.insert(i, (pts[i], text, pal::WHITE, true));
            }
        }
        f.axes3(Vec3::splat(2.3));

        // the labels: the answers first, each beside its point wherever there is room
        // (the other norms' are left out where there is none: the table has them),
        // then each plane's equation at one of its corners
        let mut board = Board::new(f.view_size(), keep_clear);
        for (p, ..) in &names {
            if let Some(at) = on_screen(f, *p) {
                board.taken.push((at - Vec2::splat(7.0), at + Vec2::splat(7.0)));
            }
        }
        let around = [(1.0, -1.0), (1.0, 1.0), (-1.0, -1.0), (-1.0, 1.0), (1.0, 0.0), (-1.0, 0.0), (0.0, -1.0), (0.0, 1.0)].map(|(x, y)| Vec2::new(x, y));
        for (p, text, c, needed) in names {
            let Some(at) = on_screen(f, p) else { continue };
            let spots: Vec<(Vec2, Align2)> = around.iter().map(|&n| (at + n.normalize() * 13.0, facing(n))).collect();
            board.place(f, &spots, text, 12.5, c, needed);
        }
        for (corners, text, c) in planes {
            let mut spots: Vec<(Vec2, Align2)> = corners.iter().filter_map(|&q| on_screen(f, q)).map(|at| (at, Align2::CENTER_CENTER)).collect();
            // (the corners highest on the screen first)
            let corners = spots.len().min(4);
            spots[..corners].sort_by(|a, b| a.0.y.total_cmp(&b.0.y));
            board.place(f, &spots, text, 13.0, c, true);
        }
        if let Some(d) = line {
            let ends: Vec<(Vec2, Align2)> = [centre + d * 3.0, centre - d * 3.0, centre + d * 2.2, centre - d * 2.2]
                .iter()
                .filter_map(|&q| on_screen(f, q))
                .flat_map(|at| [(at + Vec2::new(0.0, -10.0), Align2::CENTER_BOTTOM), (at + Vec2::new(0.0, 10.0), Align2::CENTER_TOP)])
                .collect();
            board.place(f, &ends, "both hold on this line", 12.5, BOTH, true);
        }
    }
}

/// A flat piece of a plane in space: the parallelogram round `centre` reaching `u`
/// and `v` either way, see-through (a fill of opacity `fill`), with its outline and
/// three lines across it each way. (Like `sheet` in this topic's `mod.rs`, but faint
/// enough for a ball behind it to be seen, and two of them crossing to be told apart.)
fn pane(f: &mut Frame, centre: Vec3, u: Vec3, v: Vec3, color: Color, fill: f32) {
    let q = |s: f32, t: f32| centre + u * s + v * t;
    f.dl.tris_alpha.quad(q(-1.0, -1.0), q(1.0, -1.0), q(1.0, 1.0), q(-1.0, 1.0), color.a(fill));
    f.dl.lines.closed(&[q(-1.0, -1.0), q(1.0, -1.0), q(1.0, 1.0), q(-1.0, 1.0)], 1.3, color.a(0.7));
    for k in 1..4 {
        let s = k as f32 / 2.0 - 1.0;
        f.line(q(s, -1.0), q(s, 1.0), 1.0, color.a(0.28));
        f.line(q(-1.0, s), q(1.0, s), 1.0, color.a(0.28));
    }
}

/// A coefficient of an equation, to drag or type: like `entry` in this topic's
/// `mod.rs`, but narrower and without trailing zeros, so that three of them and the
/// right-hand side fit across a phone's panel.
fn coefficient(ui: &mut egui::Ui, v: &mut f32) -> egui::Response {
    let text = |v: f64, _| {
        let t = format!("{v:.2}");
        let t = t.trim_end_matches('0').trim_end_matches('.');
        if t == "-0" {
            "0".to_string()
        } else {
            t.to_string()
        }
    };
    ui.add(egui::DragValue::new(v).speed(0.01).range(-100.0..=100.0).custom_formatter(text).min_size(egui::vec2(34.0, 18.0)))
}

/// The twelve edges of the cube of half-width 1, as pairs of corners.
const CUBE_EDGES: [([f32; 3], [f32; 3]); 12] = [
    ([-1., -1., -1.], [1., -1., -1.]),
    ([-1., 1., -1.], [1., 1., -1.]),
    ([-1., -1., 1.], [1., -1., 1.]),
    ([-1., 1., 1.], [1., 1., 1.]),
    ([-1., -1., -1.], [-1., 1., -1.]),
    ([1., -1., -1.], [1., 1., -1.]),
    ([-1., -1., 1.], [-1., 1., 1.]),
    ([1., -1., 1.], [1., 1., 1.]),
    ([-1., -1., -1.], [-1., -1., 1.]),
    ([1., -1., -1.], [1., -1., 1.]),
    ([-1., 1., -1.], [-1., 1., 1.]),
    ([1., 1., -1.], [1., 1., 1.]),
];

impl Scene for Underdet {
    fn view(&self) -> View {
        self.the_view()
    }

    fn take_view(&mut self) -> Option<View> {
        std::mem::take(&mut self.new_view).then(|| self.the_view())
    }

    fn settings(&self, s: &mut ViewSettings) {
        s.grid_alpha = 0.12;
    }

    /// Only the growing ball moves by itself.
    fn animated(&self) -> bool {
        self.growing
    }

    fn ui(&mut self, ui: &mut egui::Ui, _: &Gfx) {
        ui::note(ui, "One equation in two unknowns, $a_1 x + a_2 y = b$, is solved by every point of a line. With fewer equations than unknowns there are always infinitely many answers (or none), and one has to be chosen. A natural choice is the smallest; but size can be measured in more than one way. Grow a ball of all the points up to some size until it first touches the answers: where it touches is the smallest answer in that measure.");

        ui::section(ui, "Things to try");
        for (k, trial) in TRIALS.iter().enumerate() {
            if ui::experiment(ui, trial.0, trial.1) {
                self.set_up(k);
            }
        }

        ui::section(ui, "Unknowns and equations");
        let mut mode = self.space as usize;
        if ui::choice(ui, &["two unknowns (the plane)", "three (space)"], &mut mode) {
            self.space = mode == 1;
            self.new_view = true;
        }
        if self.space {
            let mut count = self.solid.equations - 1;
            if ui::choice(ui, &["one equation", "two equations"], &mut count) {
                self.solid.equations = count + 1;
                self.new_view = true;
            }
        }
        let names = ["x", "y", "z"];
        let n = self.n();
        let mut typed = false;
        let sys = if self.space { &mut self.solid } else { &mut self.flat };
        egui::Grid::new("equations").spacing([3.0, 4.0]).min_col_width(8.0).show(ui, |ui| {
            for k in 0..sys.equations {
                for (i, name) in names.iter().enumerate().take(n) {
                    let mut v = sys.rows[k][i] as f32;
                    if coefficient(ui, &mut v).changed() {
                        sys.rows[k][i] = v as f64;
                        typed = true;
                    }
                    // (the sign spaced as in the equation: "{}" stands for what follows it)
                    let text = if i + 1 < n { format!("${name} + {{}}$") } else { format!("${name} = {{}}$") };
                    ui.label(ui::label_text(ui, &text, 12.5, None, false));
                }
                let mut v = sys.rhs[k] as f32;
                if coefficient(ui, &mut v).changed() {
                    sys.rhs[k] = v as f64;
                    typed = true;
                }
                ui.end_row();
            }
        });
        if typed && !self.space {
            self.place_dots();
        }
        ui::note(ui, if self.space { "Drag a number sideways, or click it and type." } else { "Drag a number sideways, or click it and type; or drag the blue dots on the line." });

        ui::section(ui, "Measuring size");
        let mut p = self.p;
        if ui::slider_log(ui, "$p$", &mut p, 0.5..=8.0) {
            (self.p, self.max) = (p, false);
        }
        let ball = if self.space { ["$1$ (octahedron)", "$2$ (sphere)", r"$\infty$ (cube)"] } else { ["$1$ (diamond)", "$2$ (disc)", r"$\infty$ (square)"] };
        if let Some(i) = ui::buttons(ui, &ball) {
            match i {
                0 => (self.p, self.max) = (1.0, false),
                1 => (self.p, self.max) = (2.0, false),
                _ => self.max = true,
            }
        }
        ui::note(ui, "The $p$-norm of $(x, y)$ is");
        ui::equation(ui, r"\left( |x|^p + |y|^p \right)^{1/p}");
        ui::note(ui, r"$p = 2$ is the ordinary length, $p = 1$ is $|x| + |y|$, and as $p$ grows it comes to the larger of $|x|$ and $|y|$ ($p = \infty$, the max-norm). The max-norm does the opposite of the 1-norm: it spreads the answer out, with as many coordinates as it can of the same, largest size (with one equation, every one of them, if no coefficient is $0$).");
        ui::note(ui, "Below $p = 1$ the ball is pinched in between spikes on the axes, so it is no longer convex. Its answers are sparse too; but with many unknowns no quick way is known to find them, while the 1-norm's can be found quickly.");

        ui::section(ui, "The ball");
        ui.horizontal(|ui| {
            if ui.button("grow it again").clicked() {
                self.replay();
            }
            ui::checkbox(ui, &mut self.compare, r"compare $p = 1, 2, \infty$");
        });
        if ui::slider(ui, "size (1: touching)", &mut self.size, 0.0..=1.5) {
            self.growing = false;
        }
        ui::note(ui, "As a share of the size at which the ball first touches the answers. Beyond 1 it holds every answer up to that size.");

        ui::section(ui, "Reading the picture");
        let norm = self.norm();
        let mut key: Vec<(Color, &str)> = match (self.space, self.solid.equations) {
            (false, _) => vec![(FIRST, "blue line: every answer of the equation (drag its dots)")],
            (true, 1) => vec![(FIRST, "blue sheet: part of the plane of answers")],
            _ => vec![(FIRST, "blue and orange strips: parts of the two planes, one for each equation"), (BOTH, "white line: where they cross, the answers of both")],
        };
        key.extend([
            (norm.color(), "filled ball: every point up to the size shown, in the chosen norm"),
            (pal::WHITE, "white dot: where it first touches, the smallest answer"),
            (Norm::P(1.0).color(), "pink: the 1-norm, $|x| + |y|$ (a diamond; in space an octahedron)"),
            (Norm::P(2.0).color(), "cyan: the 2-norm, the length (a disc or sphere), with the right angle its answer makes"),
            (Norm::Max.color(), "yellow: the max-norm, the largest of $|x|$ and $|y|$ (a square or cube)"),
            (Norm::P(0.5).color(), "other $p$: the colours between these (purple below 1)"),
            (ZERO, "green in the table: a coordinate that is exactly $0$"),
        ]);
        ui::key(ui, &key);

        ui::section(ui, "Words");
        ui::term(ui, "Underdetermined:", "fewer equations than unknowns. If there is an answer at all, there are infinitely many.");
        ui::term(ui, "Norm:", "a way of measuring the size of a vector (here the $p$-norms). Its ball is every point of size at most $r$.");
        ui::term(ui, "Convex:", "containing the whole straight segment between any two of its points.");
        ui::term(ui, "Sparse:", "with many coordinates exactly $0$.");
        ui::term(ui, "Pseudo-inverse:", r"with the coefficients as a matrix $A$, $x = A^\top (A A^\top)^{-1} b$ is the answer of least 2-norm, the minimum-norm solution: the shortest answer.");

        ui::section(ui, "Numbers");
        let sys = *self.system();
        let ans = smallest(&sys, norm);
        let label = format!("smallest answer ({})", norm.name());
        match (&ans, ans.point()) {
            (Smallest::Nothing, _) => ui::readout_colored(ui, &label, "none", norm.color()),
            (_, Some(x)) => {
                ui::readout_colored(ui, &label, format!("${}$", coords(x, n)), norm.color());
                ui::readout(ui, "coordinates exactly $0$", format!("{} of {n}", zeros(x, n)));
            }
            (Smallest::At { points, .. }, None) => ui::readout_colored(ui, &label, format!("a tie: {} ends", points.len()), norm.color()),
        }
        if let Some(s) = ans.size() {
            ui::readout(ui, "its size", format!("${}$", fmt(s as f32)));
        }
        if let Some(x) = pseudo_inverse(&sys) {
            ui::readout_colored(ui, "pseudo-inverse answer", format!("${}$", coords(x, n)), Norm::P(2.0).color());
        }
        ui::see_also(ui, &[("sensing", "the same idea with a hundred unknowns"), ("overdet", "too many equations: the opposite case"), ("subspaces", "the null space: what can be added to an answer without the equations noticing"), ("sae", "a 1-norm penalty finding sparse features")]);
    }

    fn frame(&mut self, f: &mut Frame) {
        self.advance(f.dt);
        let view = f.view_size();
        let narrow = view.x < 620.0 && view.y >= 560.0;
        if narrow != self.narrow {
            self.narrow = narrow;
            self.new_view = true;
        }
        let sys = *self.system();
        let answers: Vec<(Norm, Smallest)> = self.shown_norms().into_iter().map(|m| (m, smallest(&sys, m))).collect();
        // the caption and the table first take their places, which labels keep clear of
        // (the caption broken into lines once, for its height and for drawing it)
        let story = caption_lines(view, &self.story(), CAPTION);
        let caption_h = caption_height(&story, CAPTION);
        let table = (view.y >= 300.0 && view.x >= 300.0).then(|| {
            let at = self.table_at(f, answers.len(), caption_h);
            (at, at + table_size(answers.len(), sys.unknowns))
        });
        let mut keep_clear: Vec<(Vec2, Vec2)> = table.into_iter().collect();
        // (and the application's buttons, where they lie over the top of the picture)
        if f.top_inset > 0.0 {
            keep_clear.push((Vec2::ZERO, Vec2::new(view.x, f.top_inset)));
        }
        if caption_h > 0.0 {
            keep_clear.push((Vec2::new(0.0, view.y - caption_h), view));
        }
        if self.space {
            self.draw_space(f, &answers, &keep_clear);
        } else {
            self.draw_flat(f, &answers, &keep_clear);
        }
        caption(f, &story, CAPTION);
        if let Some((at, _)) = table {
            draw_table(f, at, &answers, sys.unknowns);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn trial(k: usize) -> Underdet {
        let mut s = Underdet::create();
        s.set_up(k);
        s
    }

    /// The experiment as its button leaves it, with the ball grown to the touching size.
    fn grown(k: usize) -> Underdet {
        let mut s = trial(k);
        for _ in 0..1000 {
            s.advance(1.0 / 60.0);
        }
        assert!(!s.growing && s.touching());
        s
    }

    fn answer(s: &Underdet, norm: Norm) -> Smallest {
        smallest(s.system(), norm)
    }

    fn close(a: DVec3, b: DVec3) -> bool {
        (a - b).length() < 1e-9
    }

    /// The captions of the "things to try", held to account.
    #[test]
    fn the_experiments_show_what_their_captions_say() {
        // the shortest answer: the disc touches at the foot of the perpendicular,
        // (0.4, 0.8), which is what the pseudo-inverse gives; no zero coordinate
        let s = grown(0);
        assert!(!s.space && s.norm() == Norm::P(2.0));
        let x = answer(&s, Norm::P(2.0)).point().expect("one answer");
        assert!(close(x, DVec3::new(0.4, 0.8, 0.0)) && close(x, pseudo_inverse(&s.flat).unwrap()), "{x}");
        let (d, _) = s.flat_line().unwrap();
        assert!(x.truncate().dot(d).abs() < 1e-12, "not at a right angle");
        assert_eq!(zeros(x, 2), 0);

        // the sparsest answer: the diamond touches at its corner (0, 1), x exactly 0;
        // the disc's answer is shorter but has |x| + |y| = 1.2
        let s = grown(1);
        assert!(s.norm() == Norm::P(1.0) && s.flat == grown(0).flat);
        let x = answer(&s, Norm::P(1.0)).point().expect("one answer");
        assert!(x == DVec3::new(0.0, 1.0, 0.0) && zeros(x, 2) == 1, "{x}");
        let disc = answer(&s, Norm::P(2.0)).point().unwrap();
        assert!(disc.length() < x.length() && fmt(Norm::P(1.0).size(disc) as f32) == "1.200");
        assert_eq!(coords(x, 2), "(0, 1.000)");

        // a tie: x + y = 1.5, the whole edge from (1.5, 0) to (0, 1.5) touches, and
        // every point of it has |x| + |y| = 1.5
        let s = grown(2);
        assert_eq!(equation(s.flat.row(0), s.flat.rhs[0], 2), "x + y = 1.5");
        let tie = answer(&s, Norm::P(1.0));
        let ends = tie.points();
        assert!(ends.len() == 2 && tie.size() == Some(1.5), "{tie:?}");
        assert!(ends.contains(&DVec3::new(1.5, 0.0, 0.0)) && ends.contains(&DVec3::new(0.0, 1.5, 0.0)));
        for t in [0.1, 0.37, 0.5, 0.9] {
            let m = ends[0].lerp(ends[1], t);
            assert!((Norm::P(1.0).size(m) - 1.5).abs() < 1e-12 && (m.x + m.y - 1.5).abs() < 1e-12);
        }
        assert!(ends.iter().all(|&e| zeros(e, 2) == 1));
        assert!(s.story().starts_with("A tie: the whole segment from $(1.500, 0)$ to $(0, 1.500)$"), "{}", s.story());

        // in space, one equation: the octahedron touches the plane at (0, 0, 1), two
        // zeros; the sphere's answer (0.214, 0.429, 0.643) has none
        let s = grown(3);
        assert!(s.space && s.solid.equations == 1 && s.norm() == Norm::P(1.0));
        let x = answer(&s, Norm::P(1.0)).point().unwrap();
        assert!(x == DVec3::Z && zeros(x, 3) == 2, "{x}");
        let ball = answer(&s, Norm::P(2.0)).point().unwrap();
        assert!(coords(ball, 3) == "(0.214, 0.429, 0.643)" && zeros(ball, 3) == 0, "{ball}");

        // in space, two equations: the octahedron touches the line on an edge, at
        // (0, 0.75, 0.5): one zero, two not (at most m = 2 of them)
        let s = grown(4);
        assert!(s.space && s.solid.equations == 2);
        let x = answer(&s, Norm::P(1.0)).point().unwrap();
        assert!(close(x, DVec3::new(0.0, 0.75, 0.5)) && x.x == 0.0 && zeros(x, 3) == 1, "{x}");
        // (on an edge: on one coordinate plane, and on the octahedron of its own size)
        assert!((Norm::P(1.0).size(x) - 1.25).abs() < 1e-12);
        // the other answers there have no zero coordinate
        for norm in [Norm::P(2.0), Norm::Max] {
            assert_eq!(zeros(answer(&s, norm).point().unwrap(), 3), 0);
        }
    }

    /// "The max-norm spreads the answer evenly, every coordinate as large as the
    /// largest" (when every coefficient counts); "below p = 1 its answer is on an axis
    /// again"; with m equations a smallest answer for the 1-norm can always be found
    /// with at most m coordinates that are not zero (the ends of a tie; the points
    /// between them have more); and, in the plane, the diamond touches at a corner
    /// unless the line is parallel to an edge.
    #[test]
    fn the_claims_about_each_norm_hold_for_any_equations() {
        let mut rng = Rng::new(17);
        let mut c = || (rng.normal() as f64 * 1.5 * 100.0).round() / 100.0;
        let mut checked = 0;
        for case in 0..400 {
            let m = 1 + case % 2;
            let n = if m == 2 { 3 } else { 2 + (case / 2) % 2 };
            let sys = if n == 2 {
                System::flat([c(), c()], c())
            } else {
                System::space(m, [([c(), c(), c()], c()), ([c(), c(), c()], c())])
            };
            if sys.rhs[..m].iter().all(|b| *b == 0.0) {
                continue;
            }
            let Smallest::At { points, .. } = smallest(&sys, Norm::P(1.0)) else { continue };
            // the 1-norm: some smallest answer has at most m non-zero coordinates, and
            // so do those below p = 1 ("its answers are sparse too")
            assert!(points.iter().all(|&x| n - zeros(x, n) <= m), "{sys:?}: {points:?}");
            for p in [0.5, 0.8] {
                assert!(smallest(&sys, Norm::P(p)).points().iter().all(|&x| n - zeros(x, n) <= m), "{sys:?}: p = {p}");
            }
            // the max-norm: at least n - m + 1 coordinates share the largest size
            if let Some(x) = smallest(&sys, Norm::Max).point() {
                let t = x.abs().max_element();
                let largest = (0..n).filter(|&i| (x[i].abs() - t).abs() <= 1e-9 * t).count();
                assert!(largest > n - m, "{sys:?}: {x}");
            }
            if m == 1 {
                let rows = sys.row(0);
                let big = (0..n).map(|i| rows[i].abs()).fold(0.0, f64::max);
                let tied = (0..n).filter(|&i| rows[i].abs() == big).count();
                assert!(tied > 1 || (points.len() == 1 && zeros(points[0], n) == n - 1), "{sys:?}: {points:?}");
                // below 1, the same axis points
                for p in [0.5, 0.8] {
                    assert_eq!(smallest(&sys, Norm::P(p)).points(), &points[..], "{sys:?} p = {p}");
                }
                // the max-norm: every coordinate is as large as the largest (when
                // every coefficient counts; one that is 0 leaves a tie, tested below)
                if (0..n).all(|i| rows[i] != 0.0) {
                    let x = smallest(&sys, Norm::Max).point().expect("one answer");
                    let t = x.abs().max_element();
                    assert!((0..n).all(|i| (x[i].abs() - t).abs() < 1e-12 * t), "{sys:?}: {x}");
                }
            }
            checked += 1;
        }
        assert!(checked > 300, "{checked} checked");
    }

    /// Every way of finding the smallest answers, held against a search along the
    /// line or over the plane of answers: nothing found by the search is smaller,
    /// the search comes within a hair of it, and the answers solve the equations.
    #[test]
    fn the_smallest_answers_are_smallest() {
        let mut rng = Rng::new(3);
        let mut c = || (rng.normal() as f64 * 1.2 * 20.0).round() / 20.0;
        let norms = [Norm::P(0.5), Norm::P(0.8), Norm::P(1.0), Norm::P(1.5), Norm::P(2.0), Norm::P(3.0), Norm::P(8.0), Norm::Max];
        let mut systems: Vec<System> = TRIALS.iter().enumerate().map(|(k, _)| *trial(k).system()).collect();
        for case in 0..36 {
            let m = 1 + case % 2;
            let n = if m == 2 { 3 } else { 2 + (case / 2) % 2 };
            systems.push(if n == 2 { System::flat([c(), c()], c() + 0.3) } else { System::space(m, [([c(), c(), c()], c() + 0.3), ([c(), c(), c()], c())]) });
        }
        for sys in systems {
            for norm in norms {
                let Smallest::At { size, points } = smallest(&sys, norm) else { continue };
                for &x in &points {
                    for k in 0..sys.equations {
                        assert!((sys.row(k).dot(x) - sys.rhs[k]).abs() < 1e-9 * (1.0 + x.length()), "{sys:?} {norm:?}: {x} is not an answer");
                    }
                    assert!((norm.size(x) - size).abs() <= 1e-9 * size.max(1e-12), "{sys:?} {norm:?}");
                    assert!(sys.unknowns == 3 || x.z == 0.0);
                }
                let found = search(&sys, norm);
                assert!(found >= size * (1.0 - 1e-9), "{sys:?} {norm:?}: the search found {found}, below {size}");
                assert!(found <= size * (1.0 + 2e-4) + 1e-9, "{sys:?} {norm:?}: the search found only {found}, against {size}");
            }
        }
    }

    /// The least size over the answers of `sys`, by search: along the line of answers
    /// (one equation in the plane, two in space), or over the plane (one in space),
    /// on a grid first and then closer in round each place lower than its neighbours.
    /// (Every smallest answer is within 6.2 times the shortest one's length of it: it
    /// is no longer than its size, which is at most √3 · 3 = 5.2 times that length
    /// even for p = 0.5.)
    fn search(sys: &System, norm: Norm) -> f64 {
        let x0 = pseudo_inverse(sys).unwrap();
        let reach = 7.0 * x0.length().max(0.1);
        let (u, v) = match reduce(sys) {
            Reduced::Two { rows: [r1, r2], .. } => (r1.cross(r2).normalize(), DVec3::ZERO),
            Reduced::One { row, .. } if sys.unknowns == 2 => (DVec3::new(-row.y, row.x, 0.0).normalize(), DVec3::ZERO),
            Reduced::One { row, .. } => {
                let u = row.normalize().any_orthonormal_vector();
                (u, row.normalize().cross(u))
            }
            _ => unreachable!(),
        };
        let flat = v == DVec3::ZERO;
        let f = |s: f64, t: f64| norm.size(x0 + u * s + v * t);
        let k: i32 = if flat { 3000 } else { 110 };
        let step = reach / k as f64;
        let side = 2 * k + 1;
        let (rows, cols) = (if flat { 1 } else { side }, side);
        let grid: Vec<f64> = (0..rows * cols).map(|n| f((n % cols - k) as f64 * step, if flat { 0.0 } else { (n / cols - k) as f64 * step })).collect();
        let mut best: Vec<(f64, f64, f64)> = Vec::new();
        for j in 0..rows {
            for i in 0..cols {
                let here = grid[(j * cols + i) as usize];
                let lowest = (-1..=1).all(|dj| {
                    (-1..=1).all(|di| {
                        let (a, b) = (i + di, j + dj);
                        !(0..cols).contains(&a) || !(0..rows).contains(&b) || grid[(b * cols + a) as usize] >= here
                    })
                });
                if lowest {
                    best.push((here, (i - k) as f64 * step, if flat { 0.0 } else { (j - k) as f64 * step }));
                }
            }
        }
        best.sort_by(|a, b| a.0.total_cmp(&b.0));
        best.truncate(12);
        let mut least = f64::INFINITY;
        for (_, mut s, mut t) in best {
            let mut h = step;
            for _ in 0..40 {
                let (mut bs, mut bt, mut bv) = (s, t, f(s, t));
                for i in -4..=4 {
                    for j in if flat { 0..=0 } else { -4..=4 } {
                        let (a, b) = (s + i as f64 * h * 0.5, t + j as f64 * h * 0.5);
                        let val = f(a, b);
                        if val < bv {
                            (bs, bt, bv) = (a, b, val);
                        }
                    }
                }
                (s, t) = (bs, bt);
                h *= 0.6;
            }
            least = least.min(f(s, t));
        }
        least
    }

    /// The degenerate equations: no coefficients, b = 0, parallel planes, the same
    /// plane twice, the dots on top of each other. Every number drawn or read out is
    /// finite, and the caption says what has happened.
    #[test]
    fn degenerate_equations_are_said_plainly() {
        let mut s = Underdet::create();
        s.size = 1.0;
        s.growing = false;
        // through 0: the answer is 0 in every norm
        s.flat = System::flat([1.0, 2.0], 0.0);
        for norm in [Norm::P(0.5), Norm::P(1.0), Norm::P(2.0), Norm::Max] {
            assert_eq!(smallest(&s.flat, norm), Smallest::At { size: 0.0, points: vec![DVec3::ZERO] });
        }
        assert!(s.story().starts_with("$b = 0$"), "{}", s.story());
        // no coefficients: 0 = 1 has no answer, 0 = 0 has every point
        s.flat = System::flat([0.0, 0.0], 1.0);
        assert!(smallest(&s.flat, Norm::P(1.0)) == Smallest::Nothing && s.story().contains("no point solves it"));
        s.flat = System::flat([0.0, 0.0], 0.0);
        assert!(s.story().contains("every point is an answer"));
        // below p = 1 the ball is not convex: the point halfway between two of its
        // corners is outside it
        let half = Norm::P(0.5).size(DVec3::new(0.5, 0.5, 0.0));
        assert!(half > 1.0 && Norm::P(1.0).size(DVec3::new(0.5, 0.5, 0.0)) == 1.0, "{half}");
        // the max-norm with a line parallel to an axis: a whole edge of the square
        s.flat = System::flat([0.0, 2.0], 1.0);
        s.max = true;
        let tie = smallest(&s.flat, Norm::Max);
        assert!(tie.points() == [DVec3::new(0.5, 0.5, 0.0), DVec3::new(-0.5, 0.5, 0.0)], "{tie:?}");
        assert!(s.story().starts_with("A tie: the whole segment"), "{}", s.story());
        // below 1 with |a1| = |a2|: two separate points
        (s.max, s.p) = (false, 0.6);
        s.flat = System::flat([1.0, -1.0], 1.0);
        assert_eq!(smallest(&s.flat, Norm::P(0.6)).points().len(), 2);
        assert!(s.story().contains("2 separate points"), "{}", s.story());

        // in space: parallel planes, the same plane twice, a row of zeros
        s.space = true;
        s.p = 1.0;
        s.solid = System::space(2, [([1.0, 2.0, 3.0], 3.0), ([2.0, 4.0, 6.0], 1.0)]);
        assert!(smallest(&s.solid, Norm::P(1.0)) == Smallest::Nothing && s.story().contains("parallel"));
        s.solid = System::space(2, [([1.0, 2.0, 3.0], 3.0), ([-2.0, -4.0, -6.0], -6.0)]);
        assert!(smallest(&s.solid, Norm::P(1.0)).point() == Some(DVec3::Z) && s.story().contains("same plane"));
        s.solid = System::space(2, [([1.0, 2.0, 3.0], 3.0), ([0.0, 0.0, 0.0], 0.0)]);
        assert_eq!(smallest(&s.solid, Norm::P(1.0)).point(), Some(DVec3::Z));
        // the max-norm on a plane with two coefficients 0: a whole face of the cube
        s.solid = System::space(1, [([0.0, 0.0, 2.0], 1.0), ([0.0; 3], 0.0)]);
        assert_eq!(smallest(&s.solid, Norm::Max).points().len(), 4);
        // all three coefficients equal: a whole face of the octahedron
        s.solid = System::space(1, [([1.0, 1.0, 1.0], 1.0), ([0.0; 3], 0.0)]);
        assert_eq!(smallest(&s.solid, Norm::P(1.0)).points().len(), 3);
        assert!(s.story().contains("a whole face"), "{}", s.story());

        // the dots: dragged on top of each other, then apart again
        let mut s = Underdet::create();
        let other = s.dots[1];
        s.move_dot(0, other);
        assert!(s.flat.row(0).is_finite() && s.flat.rhs[0].is_finite() && s.dots.iter().all(|d| d.is_finite()));
        s.move_dot(0, other + DVec2::new(1.0, 1.0));
        let a = s.flat.row(0).truncate();
        assert!((a.dot(s.dots[0]) - s.flat.rhs[0]).abs() < 1e-9 && (a.dot(s.dots[1]) - s.flat.rhs[0]).abs() < 1e-9);
        assert!((a.length() - 5f64.sqrt()).abs() < 1e-9, "the coefficients keep their length");
    }

    /// Dragging a dot turns and moves the line through both dots, and the answers
    /// follow; and the ball's growth is counted in scene time, whatever the frame rate.
    #[test]
    fn the_line_follows_the_dots_and_the_ball_grows_in_scene_time() {
        let mut s = Underdet::create();
        s.move_dot(1, DVec2::new(3.0, 0.0));
        s.move_dot(0, DVec2::new(0.0, 1.0));
        let (a, b) = (s.flat.row(0), s.flat.rhs[0]);
        assert!((a.x * 3.0 - b).abs() < 1e-12 && (a.y - b).abs() < 1e-12 && (a.length() - 5f64.sqrt()).abs() < 1e-12, "{a} {b}");
        assert_eq!(smallest(&s.flat, Norm::P(1.0)).point(), Some(DVec3::new(0.0, 1.0, 0.0)));
        for fps in [30.0f32, 60.0, 120.0, 240.0] {
            let mut s = Underdet::create();
            let frames = (GROW_TIME * 0.5 * fps).round() as usize;
            for _ in 0..frames {
                s.advance(1.0 / fps);
            }
            assert!((s.size - 0.75).abs() < 1e-3 && s.growing, "{fps}: {}", s.size);
            for _ in 0..frames + 2 {
                s.advance(1.0 / fps);
            }
            assert!(s.size == 1.0 && !s.growing);
        }
    }

    /// The words used for numbers and equations.
    #[test]
    fn equations_and_points_are_written_as_by_hand() {
        assert_eq!(equation(DVec3::new(1.0, 2.0, 3.0), 3.0, 3), "x + 2y + 3z = 3");
        assert_eq!(equation(DVec3::new(2.0, 2.0, -1.0), 1.0, 3), "2x + 2y − z = 1");
        assert_eq!(equation(DVec3::new(-0.5, 0.0, 0.0), -1.25, 2), "−0.5x = −1.25");
        assert_eq!(equation(DVec3::ZERO, 0.0, 2), "0 = 0");
        assert_eq!(coords(DVec3::new(0.0, 0.75, 0.5), 3), "(0, 0.750, 0.500)");
        assert_eq!(Norm::P(3.5).name(), "$p = 3.5$");
    }

    /// Every caption, name and label the scene can write is markup that parses, with
    /// no word set in italic: for equations of every kind (degenerate ones too), every
    /// way of measuring, and the ball growing, touching and larger.
    #[test]
    fn every_text_is_markup_that_parses() {
        let fine = |text: &str| crate::typeset::inspect(text) == Default::default();
        let flat = [([1.0, 2.0], 2.0), ([1.0, 1.0], 1.5), ([1.0, 2.0], 0.0), ([0.0, 0.0], 1.0), ([0.0, 0.0], 0.0), ([0.0, 2.0], 1.0), ([1.0, -1.0], 1.0), ([-0.5, 1e-5], -1.25)];
        let solid = [
            System::space(1, SPACE_START),
            System::space(2, SPACE_START),
            System::space(2, [([1.0, 2.0, 3.0], 3.0), ([2.0, 4.0, 6.0], 1.0)]),
            System::space(2, [([1.0, 2.0, 3.0], 3.0), ([-2.0, -4.0, -6.0], -6.0)]),
            System::space(1, [([0.0, 0.0, 2.0], 1.0), ([0.0; 3], 0.0)]),
            System::space(1, [([1.0, 1.0, 1.0], 1.0), ([0.0; 3], 0.0)]),
        ];
        let systems = flat.iter().map(|(a, b)| System::flat(*a, *b)).chain(solid);
        let mut s = Underdet::create();
        s.growing = false;
        for sys in systems {
            s.space = sys.unknowns == 3;
            if s.space {
                s.solid = sys;
            } else {
                s.flat = sys;
            }
            for k in 0..sys.equations {
                assert!(fine(&format!("${}$", equation(sys.row(k), sys.rhs[k], sys.unknowns))), "{sys:?}");
            }
            for (p, max) in [(0.5, false), (0.6, false), (1.0, false), (1.5, false), (2.0, false), (3.5, false), (1.0, true)] {
                (s.p, s.max) = (p, max);
                assert!(fine(&s.norm().name()));
                for x in smallest(&sys, s.norm()).points() {
                    assert!(fine(&format!("${}$", coords(*x, sys.unknowns))) && fine(&zero_sentence(*x, sys.unknowns)));
                }
                for size in [0.4, 1.0, 1.3] {
                    s.size = size;
                    assert!(fine(&s.story()), "{}", s.story());
                }
            }
        }
    }

    /// The caption is broken between words, never inside a formula, however narrow.
    #[test]
    fn the_caption_keeps_its_formulas_whole() {
        let text = r"The 1-norm ball first touches the line at $(0, 0.750, 0.500)$: the smallest answer in this norm. $x$ is exactly $0$, and \$ is a dollar.";
        assert_eq!(words(text)[8], "$(0, 0.750, 0.500)$:");
        for room in [10.0, 40.0, 120.0, 300.0, 2000.0] {
            let lines = wrap(text, CAPTION, room);
            assert_eq!(lines.join(" "), text);
            assert!(lines.iter().all(|l| crate::typeset::inspect(l).error.is_none()), "{room}: {lines:?}");
        }
    }
}

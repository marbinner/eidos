//! Too many equations: an overdetermined system seen as lines in the plane, and three
//! ways of choosing the point that misses them least.

use super::*;
use glam::DVec2;

pub const DEF: SceneDef = SceneDef {
    id: "overdet",
    title: "Too many equations",
    group: "Linear algebra: equations",
    blurb: "Lines that do not meet at one point: least squares and two other ways to settle.",
    make: |_| Box::new(Overdet::create()),
};

/// At most this many equations.
const MOST: usize = 6;

/// Two lines closer to parallel than this (the sine of the angle between them) are
/// taken to be parallel: their crossing would be more than ten million times further
/// off than they are apart.
const PARALLEL: f64 = 1e-7;

/// The part of the plane the experiments happen in, which the picture is fitted to.
const REGION: (Vec2, Vec2) = (Vec2::new(-2.3, -2.3), Vec2::new(2.3, 2.5));

/// The colour of the equations' lines and their handles.
const LINE: Color = pal::BLUE;

/// The colour of how far the white point misses each line.
const MISS: Color = pal::PINK;

/// The lighter blue of the lines' numbers and of the arrows at their turning handles.
fn light() -> Color {
    pal::BLUE.mix(pal::WHITE, 0.45)
}

/// One equation a₁x + a₂y = b, kept with (a₁, a₂) of length 1, so that how much a
/// point misses it is its distance from the line. Its two handles lie on the line:
/// the grip, `along` from the point of the line nearest (0, 0), and the turning
/// handle, `arm` from the grip (which way along the line is the sign of `arm`).
#[derive(Clone, Copy, Debug, PartialEq)]
struct Line {
    a: DVec2,
    b: f64,
    along: f64,
    arm: f64,
}

impl Line {
    /// The line a₁x + a₂y = b (scaled so that (a₁, a₂) has length 1), with its grip
    /// at the point of it nearest `near`.
    fn new([a1, a2, b, gx, gy]: [f64; 5]) -> Line {
        let len = DVec2::new(a1, a2).length();
        let mut line = Line { a: DVec2::new(a1, a2) / len, b: b / len, along: 0.0, arm: 0.8 };
        line.along = line.dir().dot(DVec2::new(gx, gy));
        // the turning handle on the side of the grip away from the middle of things
        let middle = ((REGION.0 + REGION.1) * 0.5).as_dvec2();
        if line.dir().dot(line.grip() - middle) < 0.0 {
            line.arm = -line.arm;
        }
        line
    }

    /// Along the line: the way from the grip to the turning handle if `arm` is
    /// positive, the other way if it is negative.
    fn dir(&self) -> DVec2 {
        DVec2::new(self.a.y, -self.a.x)
    }

    fn grip(&self) -> DVec2 {
        self.a * self.b + self.dir() * self.along
    }

    fn handle(&self) -> DVec2 {
        self.grip() + self.dir() * self.arm
    }

    /// How much the point `p` misses the equation: a₁x + a₂y − b. Since (a₁, a₂) has
    /// length 1, its size is the distance from `p` to the line.
    fn miss(&self, p: DVec2) -> f64 {
        self.a.dot(p) - self.b
    }

    /// The point of the line nearest `p`: where the miss is measured to.
    fn foot(&self, p: DVec2) -> DVec2 {
        p - self.a * self.miss(p)
    }

    /// Moved without turning, so that the grip is at `p`.
    fn slide_to(&mut self, p: DVec2) {
        if p.is_finite() {
            self.b = self.a.dot(p);
            self.along = self.dir().dot(p);
        }
    }

    /// Turned about the grip, towards `p`. (Too near the grip to say which way, it
    /// stays as it is; the handle keeps a distance from the grip that can be gripped.)
    fn turn_to(&mut self, p: DVec2) {
        let grip = self.grip();
        let d = p - grip;
        let len = d.length();
        if !(len > 0.05 && len.is_finite()) {
            return;
        }
        let d = d / len * self.arm.signum();
        self.a = DVec2::new(-d.y, d.x);
        self.b = self.a.dot(grip);
        self.along = d.dot(grip);
        self.arm = len.max(0.45).min(4.0) * self.arm.signum();
    }
}

/// The three ways of saying how much a point misses the equations altogether.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Measure {
    /// The sum of the squared misses.
    Squares,
    /// The sum of their sizes.
    Absolute,
    /// The largest of their sizes.
    Largest,
}

const MEASURES: [Measure; 3] = [Measure::Squares, Measure::Absolute, Measure::Largest];

impl Measure {
    fn total(self, lines: &[Line], p: DVec2) -> f64 {
        match self {
            Measure::Squares => lines.iter().map(|l| l.miss(p).powi(2)).sum(),
            Measure::Absolute => lines.iter().map(|l| l.miss(p).abs()).sum(),
            Measure::Largest => lines.iter().map(|l| l.miss(p).abs()).fold(0.0, f64::max),
        }
    }

    fn color(self) -> Color {
        match self {
            Measure::Squares => pal::CYAN,
            Measure::Absolute => pal::ORANGE,
            Measure::Largest => pal::GREEN,
        }
    }

    /// The name of its best point.
    fn name(self) -> &'static str {
        match self {
            Measure::Squares => "least squares",
            Measure::Absolute => "least absolute values",
            Measure::Largest => "least maximum",
        }
    }

    /// The same, short enough to stand beside a dot.
    fn short(self) -> &'static str {
        match self {
            Measure::Squares => "least squares",
            Measure::Absolute => "least absolute",
            Measure::Largest => "least maximum",
        }
    }

    /// Its formula, with the dollars.
    fn symbol(self) -> &'static str {
        match self {
            Measure::Squares => r"$\sum r^2$",
            Measure::Absolute => r"$\sum |r|$",
            Measure::Largest => r"$\max |r|$",
        }
    }

    fn index(self) -> usize {
        match self {
            Measure::Squares => 0,
            Measure::Absolute => 1,
            Measure::Largest => 2,
        }
    }
}

/// Where a measure is least.
#[derive(Clone, Debug, PartialEq)]
struct Best {
    /// The point marked: one of the best.
    at: DVec2,
    /// The least total there is.
    total: f64,
    /// All the points that do as well.
    all: Ties,
}

#[derive(Clone, Debug, PartialEq)]
enum Ties {
    /// Only `at`.
    One,
    /// Every point of the segment (two corners) or polygon with these corners; `at`
    /// is their middle.
    Corners(Vec<DVec2>),
    /// Every point p with lo ≤ n·p ≤ hi: a line if lo = hi, otherwise the strip
    /// between two lines. Only when all the lines are parallel, `n` being the
    /// direction across them; `at` is the point of the middle line nearest (0, 0).
    Band { n: DVec2, lo: f64, hi: f64 },
}

/// The direction across the lines, if they are all parallel.
fn all_parallel(lines: &[Line]) -> Option<DVec2> {
    let n = lines.first()?.a;
    lines.iter().all(|l| n.perp_dot(l.a).abs() < PARALLEL).then_some(n)
}

/// Where two lines cross, if they do.
fn crossing(l: &Line, m: &Line) -> Option<DVec2> {
    let det = l.a.perp_dot(m.a);
    if det.abs() < PARALLEL {
        return None;
    }
    Some(DVec2::new(l.b * m.a.y - m.b * l.a.y, l.a.x * m.b - m.a.x * l.b) / det)
}

/// The point that makes `measure` least for these lines.
fn best(lines: &[Line], measure: Measure) -> Best {
    let Some(n) = all_parallel(lines) else {
        return match measure {
            Measure::Squares => {
                let at = least_squares(lines);
                Best { at, total: measure.total(lines, at), all: Ties::One }
            }
            Measure::Absolute => best_of(lines, measure, crossings(lines)),
            Measure::Largest => {
                let mut candidates = crossings(lines);
                candidates.extend(equidistant(lines));
                best_of(lines, measure, candidates)
            }
        };
    };
    if lines.is_empty() {
        return Best { at: DVec2::ZERO, total: 0.0, all: Ties::One };
    }
    // All parallel: only the position across them matters. Each line is n·p = c, and
    // a point p misses it by |n·p − c|, so this is the same question in one dimension.
    let mut c: Vec<f64> = lines.iter().map(|l| if l.a.dot(n) >= 0.0 { l.b } else { -l.b }).collect();
    c.sort_by(f64::total_cmp);
    let k = c.len();
    let (lo, hi) = match measure {
        Measure::Squares => {
            let mean = c.iter().sum::<f64>() / k as f64;
            (mean, mean)
        }
        // the middle one, or anywhere between the middle two
        Measure::Absolute => (c[(k - 1) / 2], c[k / 2]),
        Measure::Largest => {
            let mid = 0.5 * (c[0] + c[k - 1]);
            (mid, mid)
        }
    };
    let at = n * (0.5 * (lo + hi));
    Best { at, total: measure.total(lines, at), all: Ties::Band { n, lo, hi } }
}

/// Least squares, for lines that are not all parallel: the (x, y) that makes
/// Σ (aᵢ·(x, y) − bᵢ)² least.
///
/// The normal equations AᵀA x = Aᵀb say what it is, but solved as they stand they
/// square how sensitive the problem is: with two lines a fraction of a degree apart
/// most of the digits would be lost. So the two columns of A are made orthonormal
/// instead (Gram–Schmidt, twice over, the larger column first), and x found from
/// that, as `lsq` does.
fn least_squares(lines: &[Line]) -> DVec2 {
    let col = |k: usize| -> Vec<f64> { lines.iter().map(|l| l.a[k]).collect() };
    let b: Vec<f64> = lines.iter().map(|l| l.b).collect();
    let dot = |u: &[f64], v: &[f64]| u.iter().zip(v).map(|(x, y)| x * y).sum::<f64>();
    let (c1, c2) = (col(0), col(1));
    let swap = dot(&c2, &c2) > dot(&c1, &c1);
    let (u, v) = if swap { (c2, c1) } else { (c1, c2) };
    let r11 = dot(&u, &u).sqrt();
    if r11.is_nan() || r11 <= 0.0 {
        return DVec2::ZERO;
    }
    let q1: Vec<f64> = u.iter().map(|x| x / r11).collect();
    let mut r12 = dot(&q1, &v);
    let mut w: Vec<f64> = v.iter().zip(&q1).map(|(x, q)| x - r12 * q).collect();
    let again = dot(&q1, &w);
    w.iter_mut().zip(&q1).for_each(|(x, q)| *x -= again * q);
    r12 += again;
    let r22 = dot(&w, &w).sqrt();
    let (x1, x2) = if r22 > 1e-14 * r11 {
        let x2 = w.iter().zip(&b).map(|(x, y)| x * y).sum::<f64>() / (r22 * r22);
        ((dot(&q1, &b) - r12 * x2) / r11, x2)
    } else {
        (dot(&q1, &b) / r11, 0.0)
    };
    if swap {
        DVec2::new(x2, x1)
    } else {
        DVec2::new(x1, x2)
    }
}

/// How far from the least-squares point a newly added equation's line passes.
const NEAR: f64 = 0.3;

const PI_64: f64 = std::f64::consts::PI;

/// How far each b may be off, for the nudged answers.
const NUDGE: f64 = 0.1;

/// Where least squares can land if every b is off by up to `NUDGE` either way, the
/// lines not all parallel: the corners of a polygon round the least-squares point,
/// in turn. The answer moves in proportion to each b, so the polygon is the sum of
/// one segment for each equation (a zonotope).
fn nudged(lines: &[Line]) -> Vec<DVec2> {
    if all_parallel(lines).is_some() {
        return Vec::new();
    }
    let centre = least_squares(lines);
    // how far the answer moves when one b moves by NUDGE, each turned to point
    // upwards so that they can be taken in order of angle
    let mut steps: Vec<DVec2> = (0..lines.len())
        .map(|i| {
            let moved: Vec<Line> = lines.iter().enumerate().map(|(j, l)| Line { b: if i == j { NUDGE } else { 0.0 }, ..*l }).collect();
            least_squares(&moved)
        })
        .map(|g| if g.y < 0.0 || (g.y == 0.0 && g.x < 0.0) { -g } else { g })
        .collect();
    steps.sort_by(|p, q| p.y.atan2(p.x).total_cmp(&q.y.atan2(q.x)));
    let mut at = centre - steps.iter().copied().sum::<DVec2>();
    let mut corners = Vec::with_capacity(2 * steps.len());
    for (k, &g) in steps.iter().chain(steps.iter()).enumerate() {
        corners.push(at);
        at += if k < steps.len() { g * 2.0 } else { g * -2.0 };
    }
    corners
}

/// Every point where two of the lines cross.
fn crossings(lines: &[Line]) -> Vec<DVec2> {
    let mut out = Vec::new();
    for i in 0..lines.len() {
        for j in i + 1..lines.len() {
            out.extend(crossing(&lines[i], &lines[j]));
        }
    }
    out
}

/// Every point equally far from three of the lines, for every three and every
/// choice of the sides it is on: where sᵢ(aᵢ·p − bᵢ) = t for all three (s = ±1).
/// Turning every side over gives the same point, so the first is kept at +1.
fn equidistant(lines: &[Line]) -> Vec<DVec2> {
    let mut out = Vec::new();
    let k = lines.len();
    for i in 0..k {
        for j in i + 1..k {
            for l in j + 1..k {
                for (sj, sl) in [(1.0, 1.0), (1.0, -1.0), (-1.0, 1.0), (-1.0, -1.0)] {
                    let rows = [(lines[i], 1.0), (lines[j], sj), (lines[l], sl)].map(|(line, s)| [s * line.a.x, s * line.a.y, -1.0, s * line.b]);
                    out.extend(solve3(rows));
                }
            }
        }
    }
    out
}

/// The (x, y) of the solution of three equations in x, y and t, each row
/// [coefficients of x, y, t | right-hand side], if there is just one.
fn solve3(m: [[f64; 4]; 3]) -> Option<DVec2> {
    let det3 = |c: [usize; 3]| {
        let e = |r: usize, k: usize| m[r][c[k]];
        e(0, 0) * (e(1, 1) * e(2, 2) - e(1, 2) * e(2, 1)) - e(0, 1) * (e(1, 0) * e(2, 2) - e(1, 2) * e(2, 0)) + e(0, 2) * (e(1, 0) * e(2, 1) - e(1, 1) * e(2, 0))
    };
    let det = det3([0, 1, 2]);
    if det.abs() < 1e-12 {
        return None;
    }
    let p = DVec2::new(det3([3, 1, 2]), det3([0, 3, 2])) / det;
    p.is_finite().then_some(p)
}

/// The best of the candidates for a measure whose best points are among them, and
/// all those that tie with it.
fn best_of(lines: &[Line], measure: Measure, candidates: Vec<DVec2>) -> Best {
    let totals: Vec<(f64, DVec2)> = candidates.into_iter().map(|p| (measure.total(lines, p), p)).filter(|t| t.0.is_finite()).collect();
    let least = totals.iter().map(|t| t.0).fold(f64::INFINITY, f64::min);
    if !least.is_finite() {
        return Best { at: DVec2::ZERO, total: measure.total(lines, DVec2::ZERO), all: Ties::One };
    }
    // (ties are exact in exact arithmetic: parallel lines, a symmetric figure)
    let close = 1e-9 * least.max(1.0);
    let mut tied: Vec<DVec2> = Vec::new();
    for (total, p) in totals {
        if total <= least + close && !tied.iter().any(|q| q.distance(p) <= 1e-9 * (1.0 + p.length())) {
            tied.push(p);
        }
    }
    if tied.len() < 2 {
        let at = tied.first().copied().unwrap_or_default();
        return Best { at, total: least, all: Ties::One };
    }
    let corners = hull(tied);
    let at = corners.iter().copied().sum::<DVec2>() / corners.len() as f64;
    Best { at, total: measure.total(lines, at), all: Ties::Corners(corners) }
}

/// The corners of the smallest convex polygon holding all the points, in turn
/// (two, if they lie on one line).
fn hull(mut pts: Vec<DVec2>) -> Vec<DVec2> {
    pts.sort_by(|p, q| p.x.total_cmp(&q.x).then(p.y.total_cmp(&q.y)));
    if pts.len() < 3 {
        return pts;
    }
    let turn = |o: DVec2, a: DVec2, b: DVec2| (a - o).perp_dot(b - o);
    let mut lower: Vec<DVec2> = Vec::new();
    for &p in &pts {
        while lower.len() >= 2 && turn(lower[lower.len() - 2], lower[lower.len() - 1], p) <= 0.0 {
            lower.pop();
        }
        lower.push(p);
    }
    let mut upper: Vec<DVec2> = Vec::new();
    for &p in pts.iter().rev() {
        while upper.len() >= 2 && turn(upper[upper.len() - 2], upper[upper.len() - 1], p) <= 0.0 {
            upper.pop();
        }
        upper.push(p);
    }
    lower.pop();
    upper.pop();
    lower.extend(upper);
    lower
}

// ----- level curves -------------------------------------------------------------------

/// A convex polygon cut out of a box: its corners in turn, each with whether the edge
/// from it to the next corner is a real edge (true) or part of the box's own edge.
type Poly = Vec<(DVec2, bool)>;

fn boxed(lo: DVec2, hi: DVec2) -> Poly {
    vec![(lo, false), (DVec2::new(hi.x, lo.y), false), (hi, false), (DVec2::new(lo.x, hi.y), false)]
}

/// The part of the polygon where n·p ≤ c.
fn cut(poly: &Poly, n: DVec2, c: f64) -> Poly {
    let k = poly.len();
    let mut out = Vec::with_capacity(k + 2);
    for i in 0..k {
        let (p, real) = poly[i];
        let q = poly[(i + 1) % k].0;
        let (dp, dq) = (n.dot(p) - c, n.dot(q) - c);
        if dp <= 0.0 {
            out.push((p, real));
            if dq > 0.0 {
                out.push((p + (q - p) * (dp / (dp - dq)), true));
            }
        } else if dq <= 0.0 {
            out.push((p + (q - p) * (dp / (dp - dq)), real));
        }
    }
    out
}

/// The real edges of a cut polygon, as runs of corners.
fn real_runs(poly: &Poly) -> Vec<Vec<DVec2>> {
    let k = poly.len();
    if k < 2 {
        return Vec::new();
    }
    // (starting just after an edge of the box, so that no run is split in two)
    let start = (0..k).find(|&i| !poly[i].1).map_or(0, |i| (i + 1) % k);
    let (mut runs, mut run) = (Vec::new(), Vec::new());
    for j in 0..k {
        let i = (start + j) % k;
        if poly[i].1 {
            if run.is_empty() {
                run.push(poly[i].0);
            }
            run.push(poly[(i + 1) % k].0);
        } else if !run.is_empty() {
            runs.push(std::mem::take(&mut run));
        }
    }
    if !run.is_empty() {
        runs.push(run);
    }
    runs
}

/// The region where a measure that is a largest or a sum of sizes is at most `c`,
/// within the box: an intersection of half-planes, since |r| ≤ c is two of them,
/// and a sum of sizes is the largest of the sums with every choice of signs.
fn sublevel(lines: &[Line], measure: Measure, c: f64, lo: DVec2, hi: DVec2) -> Poly {
    let mut poly = boxed(lo, hi);
    match measure {
        Measure::Largest => {
            for l in lines {
                poly = cut(&poly, l.a, c + l.b);
                poly = cut(&poly, -l.a, c - l.b);
            }
        }
        _ => {
            for signs in 0..1u32 << lines.len() {
                let (mut n, mut k) = (DVec2::ZERO, c);
                for (i, l) in lines.iter().enumerate() {
                    let s = if signs >> i & 1 == 1 { -1.0 } else { 1.0 };
                    n += l.a * s;
                    k += l.b * s;
                }
                if n.length_squared() < 1e-20 {
                    // (the slopes cancel: either everywhere or nowhere)
                    if k < 0.0 {
                        return Vec::new();
                    }
                    continue;
                }
                // (most of these half-planes hold the whole polygon, which cutting
                // would only copy)
                if poly.iter().all(|(p, _)| n.dot(*p) - k <= 0.0) {
                    continue;
                }
                poly = cut(&poly, n, k);
                if poly.is_empty() {
                    break;
                }
            }
        }
    }
    poly
}

/// The part of the line through `p` along `d` inside the box, if any (Liang–Barsky).
fn line_in_box(p: DVec2, d: DVec2, lo: DVec2, hi: DVec2) -> Option<(DVec2, DVec2)> {
    clip_param(p, d, lo, hi, f64::NEG_INFINITY, f64::INFINITY)
}

/// The part of the segment from `a` to `b` inside the box, if any.
fn segment_in_box(a: DVec2, b: DVec2, lo: DVec2, hi: DVec2) -> Option<(DVec2, DVec2)> {
    clip_param(a, b - a, lo, hi, 0.0, 1.0)
}

fn clip_param(p: DVec2, d: DVec2, lo: DVec2, hi: DVec2, mut s0: f64, mut s1: f64) -> Option<(DVec2, DVec2)> {
    for k in 0..2 {
        if d[k].abs() < 1e-300 {
            if p[k] < lo[k] || p[k] > hi[k] {
                return None;
            }
        } else {
            let (a, b) = ((lo[k] - p[k]) / d[k], (hi[k] - p[k]) / d[k]);
            s0 = s0.max(a.min(b));
            s1 = s1.min(a.max(b));
        }
    }
    (s0 <= s1 && s0.is_finite() && s1.is_finite()).then(|| (p + d * s0, p + d * s1))
}

/// The level curves of a measure within the box: the curve where it totals `c`, as
/// runs of points. For the sum of squares it is an ellipse round the least-squares
/// point (a pair of lines if the lines are all parallel); for the other two, the
/// edge of a polygon.
fn level_curve(lines: &[Line], measure: Measure, best: &Best, c: f64, lo: DVec2, hi: DVec2) -> Vec<Vec<DVec2>> {
    if measure != Measure::Squares {
        return real_runs(&sublevel(lines, measure, c, lo, hi));
    }
    let rho2 = c - best.total;
    if rho2.is_nan() || rho2 <= 0.0 {
        return Vec::new();
    }
    // Σ rᵢ² = (p − p*)ᵀ M (p − p*) + least, with M = Σ aᵢ aᵢᵀ
    if let Ties::Band { n, lo: t, .. } = best.all {
        let off = (rho2 / lines.len() as f64).sqrt();
        let d = DVec2::new(n.y, -n.x);
        return [t - off, t + off].iter().filter_map(|&s| line_in_box(n * s, d, lo, hi)).map(|(a, b)| vec![a, b]).collect();
    }
    let (p, q, r) = lines.iter().fold((0.0, 0.0, 0.0), |(p, q, r), l| (p + l.a.x * l.a.x, q + l.a.y * l.a.y, r + l.a.x * l.a.y));
    let angle = 0.5 * (2.0 * r).atan2(p - q);
    let (v1, v2) = (DVec2::from_angle(angle), DVec2::from_angle(angle).perp());
    let l1 = p * angle.cos().powi(2) + 2.0 * r * angle.sin() * angle.cos() + q * angle.sin().powi(2);
    let l2 = (p + q - l1).max(1e-30);
    let (s1, s2) = ((rho2 / l1.max(1e-30)).sqrt(), (rho2 / l2).sqrt());
    let pts: Vec<DVec2> = round().iter().map(|&(cos, sin)| best.at + v1 * (s1 * cos) + v2 * (s2 * sin)).collect();
    // each piece cut to the box, so that nothing drawn is far from the view
    let (mut runs, mut run): (Vec<Vec<DVec2>>, Vec<DVec2>) = (Vec::new(), Vec::new());
    for w in pts.windows(2) {
        match segment_in_box(w[0], w[1], lo, hi) {
            Some((a, b)) => {
                if run.last().is_none_or(|&last: &DVec2| last.distance(a) > 1e-12 * (1.0 + a.length())) {
                    if !run.is_empty() {
                        runs.push(std::mem::take(&mut run));
                    }
                    run.push(a);
                }
                run.push(b);
            }
            None => {
                if !run.is_empty() {
                    runs.push(std::mem::take(&mut run));
                }
            }
        }
    }
    if !run.is_empty() {
        runs.push(run);
    }
    runs
}

/// The cosine and sine of the angles the ellipses are drawn at, 160 steps round: the
/// same for every ellipse, of which a frame draws as many as forty.
fn round() -> &'static [(f64, f64); 161] {
    static ROUND: std::sync::OnceLock<[(f64, f64); 161]> = std::sync::OnceLock::new();
    ROUND.get_or_init(|| {
        let steps = 160;
        std::array::from_fn(|i| {
            let t = std::f64::consts::TAU * i as f64 / steps as f64;
            (t.cos(), t.sin())
        })
    })
}

/// Faint level curves as drawn (each run of points with its colour), kept with what
/// they were drawn for.
type Kept = Option<(Vec<u64>, Vec<(Color, Vec<Vec2>)>)>;

/// The chosen measure's faint level curves, fainter further out, as drawn: those in
/// `kept` if they were drawn for the same lines, measure and box (to the bit), else
/// worked out afresh and kept. Each is up to forty curves of up to 160 points, cut to
/// the box; the white point is not among what they depend on, so while it is dragged
/// they are drawn as they were.
fn contours<'a>(kept: &'a mut Kept, lines: &[Line], m: Measure, best: &Best, lo: DVec2, hi: DVec2) -> &'a [(Color, Vec<Vec2>)] {
    let of: Vec<u64> = lines.iter().flat_map(|l| [l.a.x, l.a.y, l.b]).chain([lo.x, lo.y, hi.x, hi.y]).map(f64::to_bits).chain([m.index() as u64]).collect();
    if kept.as_ref().is_some_and(|k| k.0 != of) {
        *kept = None;
    }
    let (_, runs) = kept.get_or_insert_with(|| {
        let totals = levels(lines, m, best, lo, hi);
        let count = totals.len().max(1) as f32;
        let mut runs = Vec::new();
        for (k, c) in totals.into_iter().enumerate() {
            let alpha = 0.26 - 0.16 * k as f32 / count;
            runs.extend(level_curve(lines, m, best, c, lo, hi).into_iter().map(|run| (m.color().a(alpha), run.into_iter().map(|p| p.as_vec2()).collect())));
        }
        (of, runs)
    });
    runs
}

/// The totals at which the faint level curves are drawn: evenly spaced out from the
/// best point, about `step` apart where the total climbs fastest (further apart
/// elsewhere), as far as the box reaches, and never more than forty.
fn levels(lines: &[Line], measure: Measure, best: &Best, lo: DVec2, hi: DVec2) -> Vec<f64> {
    let step = 0.3;
    let corners = [lo, DVec2::new(hi.x, lo.y), hi, DVec2::new(lo.x, hi.y)];
    let farthest = corners.iter().map(|&p| measure.total(lines, p)).fold(0.0, f64::max);
    let n = lines.len() as f64;
    // the larger of the two numbers λ for which M = Σ aᵢ aᵢᵀ stretches some direction
    // by λ: the ellipses' shortest half-axes are ρ/√λ₁
    let (p, q, r) = lines.iter().fold((0.0, 0.0, 0.0), |(p, q, r), l| (p + l.a.x * l.a.x, q + l.a.y * l.a.y, r + l.a.x * l.a.y));
    let l1 = 0.5 * (p + q) + (0.25 * (p - q).powi(2) + r * r).sqrt();
    let level = |k: f64| match measure {
        Measure::Squares => best.total + (k * step).powi(2) * l1,
        // (a sum of n sizes climbs by at most n per unit, the largest by at most 1)
        Measure::Absolute => best.total + k * step * 0.5 * n,
        Measure::Largest => best.total + k * step,
    };
    (1..=40).map(|k| level(k as f64)).take_while(|&c| c < farthest).collect()
}

// ----- the experiments ------------------------------------------------------------------

/// A way to set the scene up: the equations (a₁, a₂, b, and a point their grip is
/// put nearest to), the white point, the measure shown, and whether the nudged
/// answers are.
struct Setup {
    lines: &'static [[f64; 5]],
    trial: [f64; 2],
    measure: Measure,
    nudge: bool,
}

/// Three lines making a triangle. Lines 1 and 2 cross at its lowest corner, which
/// line 3 misses by 2.66.
const TRIANGLE: [[f64; 5]; 3] = [[0.96, 0.28, -1.5, -1.86, 1.02], [-0.28, 0.96, -1.6, 0.67, -1.47], [0.6, 0.8, 0.5, 1.21, -0.28]];

/// Four lines that nearly meet, which the scene opens with.
const FOUR: [[f64; 5]; 4] = [[0.28, 0.96, -0.94, -1.44, -0.56], [0.8, -0.6, 0.2, 1.05, 1.06], [0.6, 0.8, 0.2, 1.19, -0.64], [-0.28, 0.96, 0.78, -1.34, 0.42]];

/// What the scene opens with.
const OPENING: Setup = Setup { lines: &FOUR, trial: [-0.2, -0.5], measure: Measure::Squares, nudge: false };

/// Four lines that nearly meet at (−1, −0.6), and a fifth far from them.
const ONE_WILD: [[f64; 5]; 5] = [
    [0.0, 1.0, -0.6, 1.2, -0.6],
    [0.8, 0.6, -1.2, -0.29, -1.62],
    [-0.6, 0.8, 0.2, 0.57, 0.68],
    [0.96, -0.28, -0.85, -0.45, 1.48],
    [0.6, 0.8, 1.5, 1.57, 0.7],
];

const PARALLEL_PAIR: [[f64; 5]; 2] = [[0.6, 0.8, -0.6, -1.37, 0.28], [0.6, 0.8, 1.4, 1.27, 0.8]];

/// Two lines 5° apart, crossing at (0.5, 0.3).
const NEARLY_PARALLEL: [[f64; 5]; 2] = [[0.0, 1.0, 0.3, -1.2, 0.3], [0.087_155_742_747_658_17, 0.996_194_698_091_745_5, 0.342_436_280_801_352_6, 1.22, 0.24]];

/// Experiments: (button, what to look for, set-up).
const TRIALS: [(&str, &str, Setup); 5] = [
    (
        "two equations: one answer",
        "Two lines cross at one point, and that point solves both equations: it misses neither, so all three totals are 0 there, and the three best points are one and the same, the crossing. Drag the white point onto it.",
        Setup { lines: &[TRIANGLE[0], TRIANGLE[1]], trial: [-0.9, -0.3], measure: Measure::Squares, nudge: false },
    ),
    (
        "a third equation: no answer",
        "Line 3 misses the crossing of lines 1 and 2 by 2.66, so no point lies on all three lines: the system has no solution, and the three best points part. Least absolute values stays at that crossing, on two of the lines. Least maximum is the centre of the green circle, the largest that fits inside the triangle: it is 1.11 from all three lines.",
        Setup { lines: &TRIANGLE, trial: [-0.9, -0.3], measure: Measure::Largest, nudge: false },
    ),
    (
        "one wild equation",
        "Lines 1 to 4 nearly meet; line 5 is 2.6 away from them. At the white point its square is over thirty times the other four together, and least squares gives way to it: the cyan point is 0.95 from where lines 1 to 4 alone put it, the green one 1.38. The orange point has not moved at all. It stays on lines 2 and 4 however far line 5 goes: drag line 5 and see.",
        Setup { lines: &ONE_WILD, trial: [-0.85, -0.78], measure: Measure::Squares, nudge: false },
    ),
    (
        "parallel lines",
        "Lines 1 and 2 never meet, so no point solves both. For least squares (and least maximum) every point of the line halfway between them is best: a whole line of answers, and the dot is only the one nearest $(0, 0)$. For absolute values every point between the lines is best, the white point too: there its two misses add up to 2, the distance between the lines.",
        Setup { lines: &PARALLEL_PAIR, trial: [-1.4, 1.5], measure: Measure::Squares, nudge: false },
    ),
    (
        "nearly parallel",
        "Lines 1 and 2 are only 5° apart. Change line 2's $b$ by 0.1 (in the panel) and the crossing moves 1.15 along them. If each $b$ may be off by 0.1, the answer can be anywhere in the long cyan sliver, up to 2.29 away. Small errors in the equations make large errors in the answer: the system is ill-conditioned.",
        Setup { lines: &NEARLY_PARALLEL, trial: [-1.2, 1.5], measure: Measure::Squares, nudge: true },
    ),
];

// ----- the layout ---------------------------------------------------------------------

/// The table of equations and totals: its width, and the heights of its parts.
const TABLE_W: f32 = 300.0;
const ROW: f32 = 17.0;
const HEAD: f32 = 19.0;
const PAD: f32 = 10.0;

/// The table's height with `n` equations.
fn table_height(n: usize) -> f32 {
    PAD + HEAD + ROW * n as f32 + 12.0 + HEAD + ROW * 3.0 + PAD
}

/// Pixels per unit with the region fitted into a part of the canvas (its corners).
fn scale_into((lo, hi): (Vec2, Vec2)) -> f32 {
    let (rmin, rmax) = REGION;
    let room = (hi - lo).max(Vec2::splat(40.0));
    (room.x / (rmax.x - rmin.x)).min(room.y / (rmax.y - rmin.y)).max(1e-3)
}

/// Where the table and the picture go on a canvas of `view` logical pixels, with
/// `inset` of its top under the application's own buttons.
#[derive(Clone, Copy, Debug, PartialEq)]
struct Layout {
    /// The table below the picture, across the canvas (a phone held upright).
    narrow: bool,
    /// The table's top-left corner and size, for the most equations (none on a
    /// canvas too small to spare the room).
    table: Option<(Vec2, Vec2)>,
    /// The part of the canvas the picture's region is fitted into.
    picture: (Vec2, Vec2),
}

impl Layout {
    fn of(view: Vec2, inset: f32) -> Layout {
        let top = inset.max(0.0) + 10.0;
        let tall = table_height(MOST);
        let none = Layout { narrow: false, table: None, picture: (Vec2::new(10.0, top), (view - Vec2::splat(10.0)).max(Vec2::new(20.0, top + 10.0))) };
        if view.x < 640.0 {
            if view.x < 300.0 || view.y < tall + 300.0 {
                return none;
            }
            let at = Vec2::new(10.0, view.y - 10.0 - tall);
            Layout { narrow: true, table: Some((at, Vec2::new(view.x - 20.0, tall))), picture: (Vec2::new(14.0, top + 8.0), Vec2::new(view.x - 14.0, at.y - 16.0)) }
        } else {
            if view.y < top + tall + 10.0 {
                return none;
            }
            // the picture beside the table or below it, whichever lets it be larger
            let at = Vec2::new(12.0, top + 2.0);
            let end = Vec2::new(view.x - 20.0, view.y - 20.0);
            let beside = (Vec2::new(at.x + TABLE_W + 30.0, top + 10.0), end);
            let below = (Vec2::new(20.0, at.y + tall + 16.0), end);
            let picture = if scale_into(below) > scale_into(beside) { below } else { beside };
            Layout { narrow: false, table: Some((at, Vec2::new(TABLE_W, tall))), picture }
        }
    }

    /// The table as drawn with `n` equations: its top-left corner and size. (Below
    /// the picture it keeps to the foot of the canvas.)
    fn table_now(&self, n: usize) -> Option<(Vec2, Vec2)> {
        self.table.map(|(at, size)| {
            let h = table_height(n);
            let top = if self.narrow { at.y + size.y - h } else { at.y };
            (Vec2::new(at.x, top), Vec2::new(size.x, h))
        })
    }

    /// The row of the table for a measure's totals: its top-left and bottom-right
    /// corners.
    fn total_row(&self, n: usize, measure: Measure) -> Option<(Vec2, Vec2)> {
        let (at, size) = self.table_now(n)?;
        let y = at.y + PAD + HEAD + ROW * n as f32 + 12.0 + HEAD + ROW * measure.index() as f32;
        Some((Vec2::new(at.x, y - 2.0), Vec2::new(at.x + size.x, y + ROW - 2.0)))
    }
}

// ----- the scene ----------------------------------------------------------------------

pub struct Overdet {
    lines: [Line; MOST],
    /// How many of them are in use.
    n: usize,
    /// The white point.
    trial: DVec2,
    measure: Measure,
    levels: bool,
    /// Whether to show where least squares lands if every b is off by up to `NUDGE`.
    nudge: bool,
    /// The size of the canvas last drawn on, and how much of its top the
    /// application's buttons cover (where the table goes, and with it the picture).
    canvas: Vec2,
    inset: f32,
    new_view: bool,
    /// The faint level curves as last drawn (see `contours`).
    contours: Kept,
}

impl Overdet {
    /// (Nothing here needs the graphics device, so the tests can make one too.)
    fn create() -> Self {
        let mut s = Overdet {
            lines: [Line::new([0.0, 1.0, 0.0, 0.0, 0.0]); MOST],
            n: 3,
            trial: DVec2::ZERO,
            measure: Measure::Squares,
            levels: true,
            nudge: false,
            canvas: Vec2::new(1078.0, 1000.0),
            inset: 0.0,
            new_view: false,
            contours: None,
        };
        s.set_up(&OPENING);
        s
    }

    /// Set the scene up: the equations, the white point, the measure, the level
    /// curves on, and the view back to the whole region.
    fn set_up(&mut self, setup: &Setup) {
        for (line, &numbers) in self.lines.iter_mut().zip(setup.lines) {
            *line = Line::new(numbers);
        }
        self.n = setup.lines.len().clamp(2, MOST);
        self.trial = DVec2::from(setup.trial);
        self.measure = setup.measure;
        self.levels = true;
        self.nudge = setup.nudge;
        self.new_view = true;
    }

    /// Use `n` equations. Each one added is a new line near the least-squares point
    /// of those already there, missing it by `NEAR`, its direction as unlike theirs
    /// as can be: one more equation that nearly agrees with the others.
    fn set_count(&mut self, n: usize) {
        let n = n.clamp(2, MOST);
        while self.n < n {
            let lines = self.used();
            let at = best(lines, Measure::Squares).at;
            // (the angle of a line's direction across, half a turn being the same line)
            let unlike = |t: f64| lines.iter().map(|l| (t - l.a.y.atan2(l.a.x)).rem_euclid(PI_64)).map(|d| d.min(PI_64 - d)).fold(f64::INFINITY, f64::min);
            let turn = (0..72).map(|k| k as f64 * PI_64 / 72.0).max_by(|&s, &t| unlike(s).total_cmp(&unlike(t))).unwrap_or(0.0);
            let a = DVec2::from_angle(turn);
            let off = if self.n.is_multiple_of(2) { NEAR } else { -NEAR };
            let grip = at + a * off + a.perp() * 1.3;
            self.lines[self.n] = Line::new([a.x, a.y, a.dot(at) + off, grip.x, grip.y]);
            self.n += 1;
        }
        self.n = n;
    }

    fn used(&self) -> &[Line] {
        &self.lines[..self.n.clamp(1, MOST)]
    }

    fn the_view(&self) -> View {
        let layout = Layout::of(self.canvas, self.inset);
        let (pmin, pmax) = layout.picture;
        let (rmin, rmax) = REGION;
        let per_unit = scale_into(layout.picture);
        // the region's middle goes to the middle of the picture's part of the canvas
        let shift = ((pmin + pmax) - self.canvas) * 0.5 / per_unit;
        let middle = (rmin + rmax) * 0.5;
        View::Fit { center: Vec2::new(middle.x - shift.x, middle.y + shift.y), half: self.canvas * 0.5 / per_unit }
    }

    /// Draw the table: the equations with how much the white point misses each, and
    /// the three totals, at the white point and at their best.
    fn table(&self, f: &mut Frame, layout: &Layout, bests: &[Best; 3]) {
        let n = self.n;
        let Some((at, size)) = layout.table_now(n) else { return };
        hud_panel(f, at, at + size);
        let (left, right) = (at.x + 12.0, at.x + size.x - 12.0);
        let mut y = at.y + PAD;
        f.text_px(Vec2::new(left, y), "the equations", 11.5, pal::MUTED, Align2::LEFT_TOP);
        f.text_px(Vec2::new(right, y), "white point misses by", 11.5, pal::MUTED, Align2::RIGHT_TOP);
        y += HEAD;
        for (i, l) in self.used().iter().enumerate() {
            f.text_px(Vec2::new(left, y), (i + 1).to_string(), 12.5, light(), Align2::LEFT_TOP);
            f.text_px(Vec2::new(left + 16.0, y), format!("${}$", equation_text(l)), 12.5, pal::TEXT, Align2::LEFT_TOP);
            f.text_px(Vec2::new(right, y), format!("${}$", signed_text(l.miss(self.trial), 2)), 12.5, MISS, Align2::RIGHT_TOP);
            y += ROW;
        }
        y += 12.0;
        let (col1, col2) = (right - 78.0, right);
        f.text_px(Vec2::new(left, y), "in all", 11.5, pal::MUTED, Align2::LEFT_TOP);
        f.text_px(Vec2::new(col1, y), "white point", 11.5, pal::MUTED, Align2::RIGHT_TOP);
        f.text_px(Vec2::new(col2, y), "least", 11.5, pal::MUTED, Align2::RIGHT_TOP);
        y += HEAD;
        for m in MEASURES {
            let chosen = m == self.measure;
            let c = m.color();
            if chosen {
                hud_rect(f, Vec2::new(at.x + 4.0, y - 2.0), Vec2::new(at.x + size.x - 4.0, y + ROW - 2.0), c.a(0.12));
            }
            hud_rect_at(f, 2, Vec2::new(left, y + 2.0), Vec2::new(left + 10.0, y + 12.0), c);
            let name = if chosen { format!("{}   shown", m.symbol()) } else { m.symbol().to_string() };
            f.text_px(Vec2::new(left + 18.0, y), name, 12.5, if chosen { c } else { pal::DIM }, Align2::LEFT_TOP);
            f.text_px(Vec2::new(col1, y), format!("${}$", total_text(m.total(self.used(), self.trial))), 12.5, pal::TEXT, Align2::RIGHT_TOP);
            f.text_px(Vec2::new(col2, y), format!("${}$", total_text(bests[m.index()].total)), 12.5, c, Align2::RIGHT_TOP);
            y += ROW;
        }
    }
}

/// A number to `decimals` places, with a true minus sign (and none on a zero).
fn num(v: f64, decimals: usize) -> String {
    if !v.is_finite() {
        return "?".into();
    }
    let text = if v.abs() >= 1e5 { format!("{:.2e}", v.abs()) } else { format!("{:.*}", decimals, v.abs()) };
    if v < 0.0 && text.chars().any(|c| ('1'..='9').contains(&c)) {
        format!("−{text}")
    } else {
        text
    }
}

/// The same with a plus sign on a positive number.
fn signed_text(v: f64, decimals: usize) -> String {
    let text = num(v, decimals);
    if v > 0.0 && text.chars().any(|c| ('1'..='9').contains(&c)) {
        format!("+{text}")
    } else {
        text
    }
}

/// A total as read out: what is left of a zero by rounding is shown as 0.
fn total_text(v: f64) -> String {
    if v.abs() < 1e-9 {
        "0".into()
    } else {
        fmt(v as f32)
    }
}

/// An equation as markup, to go between dollars: 0.80x − 0.60y = 1.20.
fn equation_text(l: &Line) -> String {
    let second = num(l.a.y, 2);
    let (op, second) = match second.strip_prefix('−') {
        Some(size) => ("−", size.to_string()),
        None => ("+", second),
    };
    format!("{}x {op} {second}y = {}", num(l.a.x, 2), num(l.b, 2))
}

/// The left-hand side, for the panel (where b can be typed), with the dollars: the
/// sign spaced as in the equation, "{}" standing for what follows it.
fn left_side(l: &Line) -> String {
    let text = equation_text(l);
    format!("${} = {{}}$", text.split(" = ").next().unwrap_or(""))
}

impl Scene for Overdet {
    fn view(&self) -> View {
        self.the_view()
    }

    fn take_view(&mut self) -> Option<View> {
        std::mem::take(&mut self.new_view).then(|| self.the_view())
    }

    fn settings(&self, s: &mut ViewSettings) {
        s.grid_alpha = 0.1;
    }

    fn animated(&self) -> bool {
        false
    }

    fn pointer_hint(&self) -> Option<&'static str> {
        Some("ringed blue dot: slide a line    blue dot with arrows: turn it    white dot: try a point    tap a total: use that measure    drag elsewhere: move    scroll: zoom")
    }

    fn ui(&mut self, ui: &mut egui::Ui, _: &Gfx) {
        ui::note(ui, "An equation in two unknowns, like $0.6x + 0.8y = 2$, holds on a line: every point $(x, y)$ of the line satisfies it. Two lines usually cross at one point, which satisfies both. A third line usually misses that point, and then no point satisfies all three: there are more equations than unknowns, and the system has no solution. What is left is to find the point that misses the lines least. But least in what sense? Drag a line's ringed dot to slide it, its dot with arrows to turn it, and the white dot to try any point.");
        ui::note(ui, "How much a point misses an equation $a_1 x + a_2 y = b$ is its residual");
        ui::equation(ui, "r = a_1 x + a_2 y - b.");
        ui::note(ui, "Every equation here is scaled so that $(a_1, a_2)$ has length 1, and then $|r|$ is exactly the distance from the point to the line: the pink segments. The picture and the algebra measure the same thing.");

        ui::section(ui, "Things to try");
        for (label, look_for, setup) in &TRIALS {
            if ui::experiment(ui, label, look_for) {
                self.set_up(setup);
            }
        }

        ui::section(ui, "Measure");
        let mut k = self.measure.index();
        if ui::choice(ui, &[r"$\sum r^2$  squares", r"$\sum |r|$  absolute values", r"$\max |r|$  largest"], &mut k) {
            self.measure = MEASURES[k];
        }
        ui::note(
            ui,
            match self.measure {
                Measure::Squares => r"Least squares makes the sum of the squared misses least: the total area of the pink squares. It has a formula, the normal equations $A^\top A x = A^\top b$, and is the usual choice.",
                Measure::Absolute => "Least absolute values makes the sum of the misses' sizes least: the total length of the pink segments. Its best point lies on two of the lines, and how far away the others are makes no difference to it.",
                Measure::Largest => "Least maximum makes the largest miss as small as it can be. Its best point is usually equally far from the three worst lines. The pink circle has the white point's largest miss as its radius: every line comes into it, and the worst just touches it.",
            },
        );
        ui.checkbox(&mut self.levels, "level curves");
        ui::note(ui, "Each faint curve joins the points where the chosen total is the same. The bright one passes through the white point: every point inside it does better.");
        let nudge = ui::label_text(ui, &format!("each $b$ off by up to {NUDGE}"), 12.5, None, false);
        ui.checkbox(&mut self.nudge, nudge);
        ui::note(ui, &format!("The cyan patch: everywhere the least-squares point can go if every $b$ is changed by at most {NUDGE} either way. Small and round: the answer is safe from small errors. Long and thin: it is not."));

        ui::section(ui, "Equations");
        let mut count = self.n - 2;
        if ui::choice(ui, &["2", "3", "4", "5", "6"], &mut count) {
            self.set_count(count + 2);
        }
        ui::note(ui, "An equation added comes in near the least-squares point, missing it a little: one more that nearly agrees.");
        for i in 0..self.n {
            let line = &mut self.lines[i];
            ui.horizontal(|ui| {
                ui.label(egui::RichText::new(format!("{}", i + 1)).size(12.5).color(light().egui()));
                ui.label(ui::label_text(ui, &left_side(line), 12.5, Some(pal::TEXT), false));
                let mut b = line.b;
                if ui.add(egui::DragValue::new(&mut b).speed(0.01).fixed_decimals(2).range(-1e4..=1e4).clamp_existing_to_range(false)).changed() && b.is_finite() {
                    line.b = b.max(-1e4).min(1e4);
                }
            });
        }
        ui::note(ui, "Change $b$ to slide a line: since $(a_1, a_2)$ has length 1, the line moves by exactly as much as $b$ changes. Its ringed dot slides it too, and its dot with arrows turns it.");

        ui::section(ui, "Reading the picture");
        let level_key = format!("faint curves, in the chosen measure's colour: where its total ({}) is the same", self.measure.symbol());
        ui::key(
            ui,
            &[
                (LINE, "blue lines: the equations, numbered as in the table; the ringed dot on each slides it, the dot with arrows turns it"),
                (pal::WHITE, "white dot: a point to try"),
                (MISS, r"pink: how far it misses each line, at right angles; for $\sum r^2$ the square on each miss, for $\max |r|$ the circle of the largest"),
                (pal::CYAN, "cyan: the least-squares point"),
                (pal::ORANGE, "orange: the least-absolute-values point"),
                (pal::GREEN, r"green: the least-maximum point, and (for $\max |r|$) the circle round it that touches the worst lines"),
                (pal::CYAN.a(0.5), "cyan patch (if asked for): where least squares can go if each $b$ is a little off"),
                (self.measure.color().a(0.6), &level_key),
            ],
        );

        ui::section(ui, "Words");
        ui::term(ui, "Overdetermined:", "more equations than unknowns. Usually no point satisfies them all.");
        ui::term(ui, "Residual $r$:", "how much a point misses an equation, $a_1 x + a_2 y - b$: zero on the line, positive on one side of it, negative on the other.");
        ui::term(ui, "Normal equations:", r"$A^\top A x = A^\top b$, where $A$ has a row $(a_1, a_2)$ for each equation. They say that $\sum r\,(a_1, a_2) = 0$: at the least-squares point the pink segments, taken as arrows from the lines to the point, add up to nothing. So that point is the average of the segments' feet on the lines.");
        ui::term(ui, "Level curve:", "the points where a total has one value, like a contour line on a map.");
        ui::term(ui, "Ill-conditioned:", "a system whose answer moves a lot when its numbers change a little.");

        ui::section(ui, "Numbers");
        let lines = self.used();
        for m in MEASURES {
            let b = best(lines, m);
            ui::readout_colored(ui, m.name(), format!("$({}, {})$", num(b.at.x, 3), num(b.at.y, 3)), m.color());
            match &b.all {
                Ties::One => {}
                Ties::Corners(c) if c.len() == 2 => ui::note(ui, "Every point of the segment drawn does as well; the dot marks its middle."),
                Ties::Corners(_) => ui::note(ui, "Every point of the patch drawn does as well; the dot marks its middle."),
                Ties::Band { lo, hi, .. } if lo == hi => ui::note(ui, "The lines are parallel: every point of the line drawn does as well. The dot marks the one nearest $(0, 0)$."),
                Ties::Band { .. } => ui::note(ui, "The lines are parallel: every point of the strip drawn does as well. The dot marks the point of its middle line nearest $(0, 0)$."),
            }
        }
        ui::readout(ui, "white point", format!("$({}, {})$", num(self.trial.x, 3), num(self.trial.y, 3)));
        let pull = normal_residue(lines);
        ui::readout(ui, r"$\sum r\,(a_1, a_2)$ at the cyan point", format!("$({}, {})$", signed(pull.x as f32, 4), signed(pull.y as f32, 4)));
        ui::note(ui, "The normal equations: this stays zero however the lines are dragged.");
        ui::see_also(ui, &[("lsq", "the same problem as a projection, from the columns' side"), ("regress", "fitting a curve to points: one equation per point"), ("underdet", "the opposite: too few equations"), ("elim", "as many equations as unknowns, solved a step at a time")]);
    }

    fn frame(&mut self, f: &mut Frame) {
        // the table's place follows the canvas, and the picture's place follows it
        // (but not into the sliver left beside the controls open on a phone: the
        // picture is as it was when they close)
        let view = f.view_size();
        let layout = Layout::of(view, f.top_inset);
        let squeezed = view.x < 200.0 || view.y < 150.0;
        if !squeezed && (layout != Layout::of(self.canvas, self.inset) || (view - self.canvas).abs().max_element() > 24.0) {
            (self.canvas, self.inset) = (view, f.top_inset);
            self.new_view = true;
        }

        // a tap on a row of totals chooses that measure
        if let Some(at) = f.ptr.pos.map(|p| p / f.px) {
            let row = MEASURES.into_iter().find(|&m| layout.total_row(self.n, m).is_some_and(|(lo, hi)| at.cmpge(lo).all() && at.cmple(hi).all()));
            if let Some(m) = row {
                f.handles.claim_clicks();
                if f.clicked_canvas() {
                    self.measure = m;
                }
            }
        }

        // the handles: each line's grip slides it, its turning handle turns it
        let n = self.n.clamp(2, MOST);
        for i in 0..n {
            let mut grip = self.lines[i].grip().as_vec2();
            if f.drag2(10 + i as u32, &mut grip, LINE) {
                self.lines[i].slide_to(grip.as_dvec2());
            }
            let mut handle = self.lines[i].handle().as_vec2();
            if f.drag2(20 + i as u32, &mut handle, LINE) {
                self.lines[i].turn_to(handle.as_dvec2());
            }
        }
        let mut trial = self.trial.as_vec2();
        if f.drag2(1, &mut trial, pal::WHITE) && trial.is_finite() {
            self.trial = trial.as_dvec2();
        }

        let lines = &self.lines[..n];
        let bests = MEASURES.map(|m| best(lines, m));
        let (lo, hi) = f.cam.flat_bounds();
        let wpp = (hi.y - lo.y) / view.y.max(1.0);
        let margin = (hi - lo) * 0.05;
        let (blo, bhi) = ((lo - margin).as_dvec2(), (hi + margin).as_dvec2());
        let to2 = |p: DVec2| p.as_vec2();

        // the chosen measure's level curves, fainter further out, and the one
        // through the white point
        let m = self.measure;
        let chosen = &bests[m.index()];
        if self.levels {
            for (color, run) in contours(&mut self.contours, lines, m, chosen, blo, bhi) {
                f.path(run.iter().copied(), 1.2, *color);
            }
            let here = m.total(lines, self.trial);
            if here > chosen.total + 1e-9 * (1.0 + here) {
                let runs = level_curve(lines, m, chosen, here, blo, bhi);
                for run in &runs {
                    f.path(run.iter().copied().map(to2), 1.8, m.color().a(0.75));
                }
                // named where there is room (not on a phone): as high on the curve
                // as it can be while well clear of the white point, outside the curve
                if !layout.narrow && view.x >= 560.0 {
                    let text = format!("same {} as the white point", m.symbol());
                    // (measured once, as it is drawn: the frame keeps what it measures)
                    let half = f.text_size(&text, 11.5).x * 0.5;
                    let room = (Vec2::new(30.0 + half, f.top_inset + 40.0), view - Vec2::new(30.0 + half, 40.0));
                    let table = layout.table_now(n);
                    let white = on_screen(f, to2(self.trial).extend(0.0)).unwrap_or_default();
                    let mut spot: Option<(f32, Vec2)> = None;
                    for w in runs.iter().flat_map(|run| run.windows(2)) {
                        for k in 0..8 {
                            let Some(s) = on_screen(f, to2(w[0].lerp(w[1], k as f64 / 8.0)).extend(0.0)) else { continue };
                            let free = s.cmpge(room.0).all() && s.cmple(room.1).all() && !table.is_some_and(|(at, size)| s.cmpge(at - Vec2::splat(20.0)).all() && s.cmple(at + size + Vec2::splat(20.0)).all());
                            // (clear of the white point first; then the higher the better)
                            let score = s.distance(white).min(110.0) * 100.0 - s.y;
                            if free && s.distance(white) > 60.0 && spot.is_none_or(|(best, _)| score > best) {
                                spot = Some((score, s));
                            }
                        }
                    }
                    if let (Some((_, s)), Some(centre)) = (spot, on_screen(f, to2(chosen.at).extend(0.0))) {
                        let (offset, align) = if s.y > centre.y { (Vec2::new(0.0, 7.0), Align2::CENTER_TOP) } else { (Vec2::new(0.0, -7.0), Align2::CENTER_BOTTOM) };
                        f.caption_px(s + offset, text, 11.5, m.color(), align);
                    }
                }
            }
        }

        // where the best points are not alone
        for (m, b) in MEASURES.iter().zip(&bests) {
            let c = m.color();
            match &b.all {
                Ties::One => {}
                Ties::Corners(corners) if corners.len() == 2 => f.line(to2(corners[0]), to2(corners[1]), 6.0, c.a(0.55)),
                Ties::Corners(corners) => {
                    let first = to2(corners[0]).extend(0.0);
                    for w in corners[1..].windows(2) {
                        f.dl.tris_alpha.tri(first, to2(w[0]).extend(0.0), to2(w[1]).extend(0.0), c.a(0.22));
                    }
                    let mut ring: Vec<Vec2> = corners.iter().map(|&p| to2(p)).collect();
                    ring.push(ring[0]);
                    f.path(ring, 2.0, c.a(0.7));
                }
                Ties::Band { n: across, lo: t0, hi: t1 } => {
                    let along = DVec2::new(across.y, -across.x);
                    if t1 - t0 > 1e-12 {
                        let strip = cut(&cut(&boxed(blo, bhi), *across, *t1), -*across, -*t0);
                        if let Some(&(first, _)) = strip.first() {
                            for w in strip[1..].windows(2) {
                                f.dl.tris_alpha.tri(to2(first).extend(0.0), to2(w[0].0).extend(0.0), to2(w[1].0).extend(0.0), c.a(0.045));
                            }
                        }
                    }
                    for t in [*t0, *t1] {
                        if let Some((a, b)) = line_in_box(*across * t, along, blo, bhi) {
                            if *m == Measure::Largest {
                                f.dashed(to2(a), to2(b), 2.0, c);
                            } else {
                                f.line(to2(a), to2(b), if *m == Measure::Squares { 4.0 } else { 1.6 }, c.a(0.8));
                            }
                        }
                    }
                }
            }
        }

        // where least squares lands if every b is off by up to NUDGE
        let mut nudge_label = None;
        if self.nudge {
            let corners = nudged(lines);
            if corners.len() >= 3 && corners.iter().all(|c| c.is_finite() && c.distance(bests[0].at) < 1e4) {
                let centre = to2(bests[0].at).extend(0.0);
                for i in 0..corners.len() {
                    f.dl.tris_alpha.tri(centre, to2(corners[i]).extend(0.0), to2(corners[(i + 1) % corners.len()]).extend(0.0), pal::CYAN.a(0.2));
                }
                let mut ring: Vec<Vec2> = corners.iter().map(|&c| to2(c)).collect();
                ring.push(ring[0]);
                f.path(ring, 1.6, pal::CYAN.a(0.85));
                // named beside the corner furthest out that is in view
                let centre = on_screen(f, centre).unwrap_or_default();
                let seen = |s: &Vec2| s.cmpge(Vec2::new(40.0, f.top_inset + 30.0)).all() && s.cmple(view - Vec2::splat(40.0)).all();
                nudge_label = corners.iter().filter_map(|&c| on_screen(f, to2(c).extend(0.0))).filter(seen).max_by(|a, b| a.distance(centre).total_cmp(&b.distance(centre))).map(|s| (s, s.x < centre.x));
            }
        }

        // the equations, with a ring round each grip and, at each turning handle, a
        // short arc of the circle it turns on
        for (i, l) in lines.iter().enumerate() {
            if let Some((a, b)) = line_in_box(l.a * l.b, l.dir(), blo, bhi) {
                f.line(to2(a), to2(b), 2.2, LINE);
            }
            f.dot(to2(l.grip()), 24.0, LINE.a(0.35));
            let spread = (24.0 * wpp as f64 / l.arm.abs()).min(0.8);
            let arc: Vec<Vec2> = (-6..=6).map(|k| to2(l.grip() + DVec2::from_angle(spread * k as f64 / 6.0).rotate(l.dir()) * l.arm)).collect();
            f.path(arc[2..11].iter().copied(), 1.5, light().a(0.75));
            f.arrow(arc[3], arc[0], 1.5, light().a(0.75));
            f.arrow(arc[9], arc[12], 1.5, light().a(0.75));
            // its number, beside the grip on the side where its residual is positive
            let side = Vec2::new(l.a.x as f32, -l.a.y as f32) * 21.0;
            f.text_off(to2(l.grip()), side, (i + 1).to_string(), 14.0, light(), Align2::CENTER_CENTER);
        }

        // how far the white point misses each line: at right angles to it, with the
        // square on that miss, standing on the line on the side clearer of the others
        let p = self.trial;
        let feet: Vec<DVec2> = lines.iter().map(|l| l.foot(p)).collect();
        let worst = lines.iter().map(|l| l.miss(p).abs()).fold(0.0, f64::max);
        for (i, l) in lines.iter().enumerate() {
            let (r, foot) = (l.miss(p), feet[i]);
            if m == Measure::Squares && r.abs() < 1e3 {
                let corners = square_on(lines, i, p).map(|q| to2(q).extend(0.0));
                f.dl.tris_alpha.quad(corners[0], corners[1], corners[2], corners[3], MISS.a(0.13));
                f.dl.lines.closed(&corners, 1.2, MISS.a(0.55));
            }
            let width = if m == Measure::Largest && r.abs() >= worst * (1.0 - 1e-9) { 3.4 } else { 2.0 };
            f.line(to2(p), to2(foot), width, MISS);
            let mark = 8.0 * wpp;
            if (r.abs() as f32) > 2.5 * mark {
                let (toward, along) = ((p - foot).as_vec2().extend(0.0), l.dir().as_vec2().extend(0.0));
                right_angle(f, to2(foot).extend(0.0), toward, along, mark, MISS.a(0.8));
            }
        }
        if m == Measure::Largest {
            if worst > 1e-9 && worst < 1e4 {
                f.circle(to2(p), Vec3::Z, worst as f32, 1.4, MISS.a(0.8));
            }
            let b = &bests[2];
            if matches!(b.all, Ties::One) && b.total > 1e-9 && b.total < 1e4 {
                f.circle(to2(b.at), Vec3::Z, b.total as f32, 1.6, pal::GREEN.a(0.8));
            }
        }

        // the best points, the biggest underneath, so that where they coincide they
        // show as rings round each other; named beside them, or at the edge of the
        // view if they are off it
        let backing = Color::hex(0x080B14);
        for (m, size) in MEASURES.into_iter().zip([19.0, 13.5, 8.5]) {
            let at = to2(bests[m.index()].at);
            f.dot(at, size + 3.0, backing);
            f.dot(at, size, m.color());
        }
        let inside = (Vec2::new(30.0, f.top_inset + 30.0), view - Vec2::splat(30.0));
        let mut labels = Vec::new();
        let dots: Vec<Vec2> = bests.iter().filter_map(|b| on_screen(f, to2(b.at).extend(0.0))).collect();
        for m in MEASURES {
            let b = &bests[m.index()];
            let Some(s) = on_screen(f, to2(b.at).extend(0.0)) else { continue };
            let name = match &b.all {
                Ties::One => m.short().to_string(),
                _ => format!("{}: one of many", m.short()),
            };
            if s.cmpge(inside.0).all() && s.cmple(inside.1).all() {
                // beside the dot, on the left if on the right it would cover another
                let width = text_width(&name, 12.0) + 10.0;
                let covers = |x0: f32| dots.iter().any(|&d| d.distance(s) > 3.0 && d.x > x0 - 10.0 && d.x < x0 + width + 10.0 && (d.y - s.y).abs() < 20.0);
                let x = if covers(s.x + 14.0) && !covers(s.x - 14.0 - width) { -14.0 - width } else { 14.0 };
                labels.push((s + Vec2::new(x, -9.0), name, m.color()));
            } else if inside.0.cmplt(inside.1).all() {
                let edge = s.max(inside.0).min(inside.1);
                let d = (s - edge).normalize_or_zero();
                hud_arrow(f, edge - d * 22.0, edge, 2.2, m.color());
                labels.push((edge - d * 22.0 + Vec2::new(-d.x.max(0.0) * text_width(&name, 12.0) - 4.0, 6.0), name, m.color()));
            }
        }
        if let Some((s, left)) = nudge_label {
            let text = format!("each $b$ off by up to {NUDGE}");
            let x = if left { -14.0 - f.text_size(&text, 12.0).x - 8.0 } else { 10.0 };
            labels.push((s + Vec2::new(x, 4.0), text, pal::CYAN));
        }
        plates(f, labels, 12.0);

        self.table(f, &layout, &bests);
        axes_clear_of(f, layout.table_now(n));
    }
}

/// The numbers along the axes, as `Frame::axes2` writes them, but for those that
/// would lie on the table (`avoid`: its top-left corner and size), where they would
/// be written over its own numbers.
fn axes_clear_of(f: &mut Frame, avoid: Option<(Vec2, Vec2)>) {
    let (lo, hi) = f.cam.flat_bounds();
    let view = f.view_size();
    let wpp = (hi.y - lo.y) / view.y;
    if !(wpp.is_finite() && wpp > 0.0 && view.x >= 1.0) {
        return;
    }
    let step = nice_step(wpp * 95.0);
    let margin = 16.0 * wpp;
    let y_anchor = 0.0f32.max(lo.y + margin * 1.6).min(hi.y - margin * 0.4);
    let x_anchor = 0.0f32.max(lo.x + margin * 0.4).min(hi.x - margin * 2.4);
    let edge = 20.0 * wpp;
    let ticks = |lo: f32, hi: f32| {
        let first = (lo / step).ceil();
        let count = (((hi - lo) / step).floor().max(0.0) as usize).min(400);
        (0..=count).map(move |i| (first + i as f32) * step).filter(move |v| *v <= hi)
    };
    let covered = |f: &Frame, p: Vec2| {
        let at = on_screen(f, p.extend(0.0)).unwrap_or(Vec2::NAN);
        avoid.is_some_and(|(corner, size)| at.cmpge(corner - Vec2::splat(30.0)).all() && at.cmple(corner + size + Vec2::splat(12.0)).all())
    };
    for x in ticks(lo.x, hi.x) {
        let at = Vec2::new(x, y_anchor);
        if x.abs() > step * 0.01 && x > lo.x + edge && x < hi.x - edge && !covered(f, at) {
            f.text_off(at, Vec2::new(0.0, 5.0), fmt_tick(x, step), 12.0, pal::DIM, Align2::CENTER_TOP);
        }
    }
    for y in ticks(lo.y, hi.y) {
        let at = Vec2::new(x_anchor, y);
        if y.abs() > step * 0.01 && y > lo.y + edge * 0.6 && y < hi.y - edge * 0.6 && !covered(f, at) {
            f.text_off(at, Vec2::new(6.0, 0.0), fmt_tick(y, step), 12.0, pal::DIM, Align2::LEFT_CENTER);
        }
    }
}

/// The square on how much `p` misses line `i`: its first side the pink segment from
/// `p` to the line, its second along the line, on whichever side keeps it clearer
/// of the other misses.
fn square_on(lines: &[Line], i: usize, p: DVec2) -> [DVec2; 4] {
    let l = &lines[i];
    let foot = l.foot(p);
    let clear = |side: DVec2| {
        let middle = ((p + foot + side) * 0.5).as_vec2();
        lines.iter().enumerate().filter(|&(j, _)| j != i).map(|(_, m)| seg_dist(middle, p.as_vec2(), m.foot(p).as_vec2())).fold(f32::MAX, f32::min)
    };
    let side = l.dir() * l.miss(p).abs();
    let side = if clear(-side) > clear(side) { -side } else { side };
    [p, foot, foot + side, p + side]
}

/// Σ r (a₁, a₂) at the least-squares point: zero, by the normal equations.
fn normal_residue(lines: &[Line]) -> DVec2 {
    let at = best(lines, Measure::Squares).at;
    lines.iter().fold(DVec2::ZERO, |s, l| s + l.a * l.miss(at))
}

/// Names on plates at screen positions, each moved down just far enough that no two
/// cover each other.
fn plates(f: &mut Frame, mut labels: Vec<(Vec2, String, Color)>, size: f32) {
    let line = size + 10.0;
    labels.sort_by(|a, b| a.0.y.total_cmp(&b.0.y));
    // (each measured once: one with a formula in it exactly, as it is drawn)
    let widths: Vec<f32> = labels.iter().map(|(_, text, _)| if crate::typeset::has_math(text) { f.text_size(text, size).x } else { text_width(text, size) }).collect();
    for i in 1..labels.len() {
        for j in 0..i {
            let (above, here) = (labels[j].0, labels[i].0);
            let reach = (if above.x <= here.x { widths[j] } else { widths[i] }) + 14.0;
            if (above.x - here.x).abs() < reach && here.y < above.y + line {
                labels[i].0.y = above.y + line;
            }
        }
    }
    for (at, text, color) in labels {
        f.caption_px(at, text, size, color, Align2::LEFT_TOP);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn scene(setup: &Setup) -> Overdet {
        let mut s = Overdet::create();
        s.set_up(setup);
        s
    }

    /// Every set-up a button (or the opening) makes.
    fn setups() -> Vec<(&'static str, &'static Setup)> {
        let mut all: Vec<(&str, &Setup)> = vec![("opening", &OPENING)];
        all.extend(TRIALS.iter().map(|t| (t.0, &t.2)));
        all
    }

    /// `n` lines at random: any direction, b from −2 to 2.
    fn random_lines(rng: &mut Rng, n: usize) -> Vec<Line> {
        (0..n)
            .map(|_| {
                let turn = rng.range(0.0, TAU) as f64;
                Line::new([turn.cos(), turn.sin(), rng.range(-2.0, 2.0) as f64, 0.0, 0.0])
            })
            .collect()
    }

    /// The experiments' lines and a few dozen at random, two to six of them.
    fn cases() -> Vec<Vec<Line>> {
        let mut rng = Rng::new(11);
        let mut all: Vec<Vec<Line>> = setups().iter().map(|(_, s)| scene(s).used().to_vec()).collect();
        all.extend((0..30).map(|k| random_lines(&mut rng, 2 + k % 5)));
        all
    }

    /// Is `p` inside the closed polygon `ring` (by the number of edges a ray from it
    /// crosses)?
    fn inside(ring: &[DVec2], p: DVec2) -> bool {
        let mut odd = false;
        for i in 0..ring.len() {
            let (a, b) = (ring[i], ring[(i + 1) % ring.len()]);
            if (a.y > p.y) != (b.y > p.y) && p.x < a.x + (p.y - a.y) / (b.y - a.y) * (b.x - a.x) {
                odd = !odd;
            }
        }
        odd
    }

    /// Whatever an experiment is about lies in the part of the plane the picture is
    /// fitted to, so that it is in view on every canvas: the handles, the white
    /// point and the best points.
    #[test]
    fn every_experiment_is_in_view() {
        let (lo, hi) = (REGION.0.as_dvec2(), REGION.1.as_dvec2());
        let within = |p: DVec2| p.cmpge(lo).all() && p.cmple(hi).all();
        for (name, setup) in setups() {
            let s = scene(setup);
            for (i, l) in s.used().iter().enumerate() {
                assert!(within(l.grip()) && within(l.handle()), "{name}: line {} has its handles at {} and {}", i + 1, l.grip(), l.handle());
            }
            assert!(within(s.trial), "{name}: the white point");
            for m in MEASURES {
                let b = best(s.used(), m);
                assert!(within(b.at), "{name}: {} at {}", m.name(), b.at);
            }
        }
    }

    /// The three best points against a search: no point of a fine grid round the
    /// point found, nor of a coarse grid over a wide square, does better. Where the
    /// best points are many, every corner of them, and every point of the line or
    /// strip, does as well.
    #[test]
    fn no_point_of_a_fine_grid_beats_the_best_points() {
        let mut ties = 0;
        for lines in cases() {
            for m in MEASURES {
                let b = best(&lines, m);
                let close = 1e-9 * (1.0 + b.total);
                assert!((m.total(&lines, b.at) - b.total).abs() < close, "{}: the total read out is not the one at the point", m.name());
                let mut least = f64::INFINITY;
                for i in -300..=300 {
                    for j in -300..=300 {
                        least = least.min(m.total(&lines, b.at + DVec2::new(i as f64, j as f64) * 0.01));
                    }
                }
                for i in -100..=100 {
                    for j in -100..=100 {
                        least = least.min(m.total(&lines, DVec2::new(i as f64, j as f64) * 0.1));
                    }
                }
                assert!(least >= b.total - close, "{} on {lines:?}: the grid found {least}, the solver {}", m.name(), b.total);
                match b.all {
                    Ties::One => {}
                    Ties::Corners(corners) => {
                        ties += 1;
                        assert!(corners.iter().all(|&q| (m.total(&lines, q) - b.total).abs() < 1e-8 * (1.0 + b.total)));
                    }
                    Ties::Band { n, lo, hi } => {
                        ties += 1;
                        let along = DVec2::new(n.y, -n.x);
                        for t in [lo, hi, 0.5 * (lo + hi)] {
                            for s in [-3.0, 0.0, 5.0] {
                                assert!((m.total(&lines, n * t + along * s) - b.total).abs() < 1e-8 * (1.0 + b.total));
                            }
                        }
                    }
                }
            }
        }
        // (the parallel lines' experiment has whole lines and a strip of best points)
        assert!(ties >= 3, "{ties}");
    }

    /// "Its best point lies on two of the lines, and how far away the others are
    /// makes no difference to it": wherever the least-absolute-values point is
    /// alone, it is on two lines at least, and pushing a line it is not on further
    /// away leaves it where it is.
    #[test]
    fn least_absolute_values_lies_on_two_lines_and_ignores_the_others() {
        let mut rng = Rng::new(3);
        let mut seen = 0;
        for k in 0..300 {
            let lines = random_lines(&mut rng, 3 + k % 4);
            let b = best(&lines, Measure::Absolute);
            if b.all != Ties::One {
                continue;
            }
            seen += 1;
            assert!(lines.iter().filter(|l| l.miss(b.at).abs() < 1e-9).count() >= 2, "{lines:?}");
            for i in 0..lines.len() {
                let r = lines[i].miss(b.at);
                if r.abs() < 1e-6 {
                    continue;
                }
                for far in [0.5, 3.0, 40.0] {
                    let mut moved = lines.clone();
                    moved[i].b -= r.signum() * far;
                    let again = best(&moved, Measure::Absolute).at;
                    assert!(again.distance(b.at) < 1e-9, "{lines:?}: line {i} pushed {far} further moves it to {again}");
                }
            }
        }
        assert!(seen > 280, "{seen}");
    }

    /// "Its best point is usually equally far from the three worst lines", and no line
    /// is further: wherever the least-maximum point is alone among three lines or
    /// more, which it is in nearly all random cases (not with parallel lines, say).
    /// The green circle round it, with that distance as its radius, so touches three
    /// lines and crosses or touches all the others.
    #[test]
    fn least_maximum_is_equally_far_from_the_three_worst_lines() {
        let mut rng = Rng::new(4);
        let mut seen = 0;
        for k in 0..300 {
            let lines = random_lines(&mut rng, 3 + k % 4);
            let b = best(&lines, Measure::Largest);
            if b.all != Ties::One {
                continue;
            }
            seen += 1;
            let worst = lines.iter().filter(|l| (l.miss(b.at).abs() - b.total).abs() < 1e-9 * (1.0 + b.total)).count();
            assert!(worst >= 3 && lines.iter().all(|l| l.miss(b.at).abs() <= b.total + 1e-12), "{lines:?}: {worst} lines at the largest distance");
        }
        assert!(seen > 280, "{seen}");
    }

    /// The normal equations hold at the least-squares point, Σ r (a₁, a₂) = 0, and
    /// read out as 0 however the lines are dragged: lines anywhere in a wide square,
    /// and pairs of lines down to a millionth of a radian apart (where solving
    /// AᵀA x = Aᵀb as it stands would lose most of the digits). The point is the
    /// average of the feet of its misses.
    #[test]
    fn least_squares_solves_the_normal_equations() {
        let mut rng = Rng::new(5);
        let mut all = cases();
        for k in 0..60 {
            let mut lines = random_lines(&mut rng, 2 + k % 5);
            for l in &mut lines {
                l.b *= 50.0;
            }
            all.push(lines);
        }
        for angle in [0.3f64, 1e-2, 1e-4, 1e-6] {
            for third in [false, true] {
                let mut lines = vec![Line::new([0.0, 1.0, 0.4, 0.0, 0.0]), Line::new([angle.sin(), angle.cos(), -0.3, 0.0, 0.0])];
                if third {
                    lines.push(Line::new([1.0, 0.2, 1.0, 0.0, 0.0]));
                }
                all.push(lines);
            }
        }
        for lines in all {
            let pull = normal_residue(&lines);
            let read = (signed(pull.x as f32, 4), signed(pull.y as f32, 4));
            assert!(read == ("0".to_string(), "0".to_string()), "{lines:?}: Σ r a = {pull}");
            if all_parallel(&lines).is_none() {
                let at = best(&lines, Measure::Squares).at;
                let feet = lines.iter().map(|l| l.foot(at)).sum::<DVec2>() / lines.len() as f64;
                assert!(feet.distance(at) < 1e-6 * (1.0 + at.length()), "{lines:?}: the feet average {feet}, the point is {at}");
            }
        }
    }

    /// What the picture draws for the white point is what the panel says: each pink
    /// segment meets its line at a right angle and is |r| long; each square has
    /// that segment for a side, its next side along the line, and area r²; the
    /// circle has the largest miss for its radius, and every line comes into it.
    #[test]
    fn the_pink_marks_are_the_misses() {
        let mut rng = Rng::new(6);
        for lines in cases() {
            for _ in 0..5 {
                let p = DVec2::new(rng.range(-3.0, 3.0) as f64, rng.range(-3.0, 3.0) as f64);
                let worst = Measure::Largest.total(&lines, p);
                for (i, l) in lines.iter().enumerate() {
                    let (r, foot) = (l.miss(p), l.foot(p));
                    assert!(l.miss(foot).abs() < 1e-12 && ((p - foot).length() - r.abs()).abs() < 1e-12 && (p - foot).perp_dot(l.a).abs() < 1e-12);
                    let [q0, q1, q2, q3] = square_on(&lines, i, p);
                    let area = 0.5 * ((q1 - q0).perp_dot(q2 - q0) + (q2 - q0).perp_dot(q3 - q0)).abs();
                    assert!(q0 == p && q1 == foot && l.miss(q2).abs() < 1e-9 && (area - r * r).abs() < 1e-9 * (1.0 + r * r), "{area} against {}", r * r);
                    assert!(r.abs() <= worst);
                }
                assert!(lines.iter().any(|l| l.miss(p).abs() == worst));
            }
        }
    }

    /// "The bright one passes through the white point: every point inside it does
    /// better." Every point of the curve drawn has the white point's total, and of
    /// the points round about, those inside the curve total less and those outside
    /// more (where the curve closes within the box).
    #[test]
    fn inside_the_bright_curve_every_point_does_better() {
        let mut rng = Rng::new(7);
        let (lo, hi) = (DVec2::splat(-60.0), DVec2::splat(60.0));
        let mut closed = 0;
        for lines in cases() {
            for m in MEASURES {
                let b = best(&lines, m);
                let p = DVec2::new(rng.range(-2.0, 2.0) as f64, rng.range(-2.0, 2.0) as f64);
                let here = m.total(&lines, p);
                let runs = level_curve(&lines, m, &b, here, lo, hi);
                for q in runs.iter().flatten() {
                    assert!((m.total(&lines, *q) - here).abs() < 1e-7 * (1.0 + here), "{}: {} on the curve against {here}", m.name(), m.total(&lines, *q));
                }
                if let [ring] = runs.as_slice() {
                    if ring.first() == ring.last() || ring.first().is_some_and(|f| ring.last().is_some_and(|l| f.distance(*l) < 1e-9)) {
                        closed += 1;
                        for i in -20..=20 {
                            for j in -20..=20 {
                                let q = b.at + DVec2::new(i as f64, j as f64) * 0.17;
                                let total = m.total(&lines, q);
                                if (total - here).abs() > 1e-3 * (1.0 + here) {
                                    assert_eq!(inside(ring, q), total < here, "{}: {q} totals {total}, the curve {here}", m.name());
                                }
                            }
                        }
                    }
                }
                // the faint ones too lie where the measure has their totals (but for
                // the ends of an ellipse's pieces cut off at the box, which are on
                // its chords)
                for c in levels(&lines, m, &b, DVec2::splat(-3.0), DVec2::splat(3.0)) {
                    for q in level_curve(&lines, m, &b, c, DVec2::splat(-3.0), DVec2::splat(3.0)).iter().flatten().filter(|q| q.abs().max_element() < 3.0 - 1e-9) {
                        assert!((m.total(&lines, *q) - c).abs() < 1e-7 * (1.0 + c), "{}: {} against {c}", m.name(), m.total(&lines, *q));
                    }
                }
            }
        }
        assert!(closed > 60, "{closed}");
    }

    /// The cyan patch is where least squares lands with every b off by up to
    /// NUDGE: the answer for any such change is inside it, and its corners are the
    /// answers for changes of ±NUDGE in every b.
    #[test]
    fn the_nudged_answers_fill_the_cyan_patch() {
        let mut rng = Rng::new(8);
        for lines in cases() {
            let corners = nudged(&lines);
            if all_parallel(&lines).is_some() {
                assert!(corners.is_empty());
                continue;
            }
            assert_eq!(corners.len(), 2 * lines.len());
            let answer = |signs: &[f64]| {
                let moved: Vec<Line> = lines.iter().zip(signs).map(|(l, s)| Line { b: l.b + s * NUDGE, ..*l }).collect();
                best(&moved, Measure::Squares).at
            };
            let size = corners.iter().map(|c| c.distance(best(&lines, Measure::Squares).at)).fold(0.0, f64::max);
            for _ in 0..40 {
                let signs: Vec<f64> = lines.iter().map(|_| rng.range(-0.999, 0.999) as f64).collect();
                let at = answer(&signs);
                // (a little inside: shrink the patch a hair about its middle)
                let middle = best(&lines, Measure::Squares).at;
                let grown: Vec<DVec2> = corners.iter().map(|&c| middle + (c - middle) * (1.0 + 1e-9)).collect();
                assert!(inside(&grown, at), "{lines:?}: {at} outside the patch");
            }
            for &c in &corners {
                let hit = (0..1u32 << lines.len()).any(|bits| {
                    let signs: Vec<f64> = (0..lines.len()).map(|i| if bits >> i & 1 == 1 { 1.0 } else { -1.0 }).collect();
                    answer(&signs).distance(c) < 1e-9 * (1.0 + size)
                });
                assert!(hit, "{lines:?}: corner {c} is no answer");
            }
        }
    }

    /// The handles and the panel move the lines as they say: the grip slides a line
    /// without turning it; the turning handle turns it about the grip, towards the
    /// pointer; and since (a₁, a₂) has length 1, a change of b moves the line by
    /// exactly that much.
    #[test]
    fn the_handles_slide_and_turn_their_lines() {
        for (name, setup) in setups() {
            let s = scene(setup);
            for &l in s.used() {
                let mut slid = l;
                slid.slide_to(DVec2::new(0.7, -1.3));
                assert!(slid.a == l.a && slid.grip().distance(DVec2::new(0.7, -1.3)) < 1e-12, "{name}");
                let mut turned = l;
                let to = l.grip() + DVec2::new(-0.3, 1.1);
                turned.turn_to(to);
                assert!(turned.grip().distance(l.grip()) < 1e-12 && turned.miss(to).abs() < 1e-12 && turned.handle().distance(to) < 1e-12, "{name}");
                assert!((turned.a.length() - 1.0).abs() < 1e-12);
                for nowhere in [l.grip(), DVec2::NAN, DVec2::INFINITY] {
                    let (mut a, mut b) = (l, l);
                    a.turn_to(nowhere);
                    b.slide_to(nowhere);
                    assert!(a == l && (b == l || nowhere == l.grip()), "{name}: a handle sent to {nowhere} changed the line");
                }
                let mut shifted = l;
                shifted.b += 0.37;
                let p = l.grip();
                assert!((shifted.foot(p).distance(p) - 0.37).abs() < 1e-12);
            }
        }
    }

    /// "An equation added comes in near the least-squares point, missing it a
    /// little": by NEAR, its direction at least a few degrees from every other
    /// line's (as far as six lines allow, 30°).
    #[test]
    fn an_added_equation_nearly_agrees() {
        for (name, setup) in setups() {
            let mut s = scene(setup);
            while s.n < MOST {
                let before = best(s.used(), Measure::Squares).at;
                s.set_count(s.n + 1);
                let (new, old) = s.used().split_last().expect("lines");
                assert!((new.miss(before).abs() - NEAR).abs() < 1e-12, "{name}: {}", new.miss(before));
                let apart = old.iter().map(|l| l.a.perp_dot(new.a).abs().asin().to_degrees()).fold(90.0, f64::min);
                assert!(apart > 14.0, "{name}: the new line is {apart}° from another");
                assert!(new.grip().distance(before) < 1.5 && new.handle().is_finite());
            }
            s.set_count(2);
            assert_eq!(s.used().len(), 2);
        }
    }

    /// The captions of the "things to try", held to account.
    #[test]
    fn the_experiments_show_what_their_captions_say() {
        // two equations: the crossing solves both, every total is 0 there, and the
        // three best points are the crossing
        let two = scene(&TRIALS[0].2);
        let lines = two.used();
        let cross = crossing(&lines[0], &lines[1]).expect("they cross");
        assert_eq!(lines.len(), 2);
        for m in MEASURES {
            let b = best(lines, m);
            assert!(b.all == Ties::One && b.at.distance(cross) < 1e-12 && total_text(b.total) == "0", "{}: {b:?}", m.name());
            assert!(m.total(lines, two.trial) > 0.5);
        }

        // a third equation: it misses the crossing of 1 and 2 by 2.66; the best
        // points part; least absolute values stays at that crossing; least maximum
        // is 1.11 from all three lines, inside the triangle (the centre of the
        // largest circle inside it)
        let three = scene(&TRIALS[1].2);
        let lines = three.used();
        assert!(lines.len() == 3 && three.measure == Measure::Largest);
        let miss = lines[2].miss(crossing(&lines[0], &lines[1]).unwrap_or_default());
        assert_eq!(num(miss.abs(), 2), "2.66");
        let points = MEASURES.map(|m| best(lines, m).at);
        assert!(points[0].distance(points[2]) > 0.3 && points[0].distance(points[1]) > 1.0 && points[1].distance(points[2]) > 1.0, "{points:?}");
        assert!(points[1].distance(crossing(&lines[0], &lines[1]).unwrap_or_default()) < 1e-12);
        let centre = best(lines, Measure::Largest);
        assert!(num(centre.total, 2) == "1.11" && lines.iter().all(|l| (l.miss(centre.at).abs() - centre.total).abs() < 1e-12));
        for i in 0..3 {
            let (j, k) = ((i + 1) % 3, (i + 2) % 3);
            let corner = crossing(&lines[j], &lines[k]).unwrap_or_default();
            assert!(lines[i].miss(centre.at) * lines[i].miss(corner) > 0.0, "the green point is not inside the triangle");
        }
        assert!(best(lines, Measure::Largest).total > 1.0, "a point on all three lines");

        // one wild equation: 2.6 from where lines 1 to 4 nearly meet; its square
        // over thirty times the others together at the white point; least squares
        // 0.95 and least maximum 1.38 from the four lines' own best points; least
        // absolute values not moved, on lines 2 and 4, however far line 5 goes
        let wild = scene(&TRIALS[2].2);
        let (all, four) = (wild.used(), &wild.lines[..4]);
        assert_eq!(all.len(), 5);
        let orange = best(all, Measure::Absolute).at;
        assert_eq!(num(all[4].miss(orange).abs(), 1), "2.6");
        let squares: Vec<f64> = all.iter().map(|l| l.miss(wild.trial).powi(2)).collect();
        let ratio = squares[4] / squares[..4].iter().sum::<f64>();
        assert!(ratio > 30.0 && ratio < 40.0, "{ratio}");
        let shift = |m: Measure| best(all, m).at.distance(best(four, m).at);
        assert!(num(shift(Measure::Squares), 2) == "0.95" && num(shift(Measure::Largest), 2) == "1.38", "{} {}", shift(Measure::Squares), shift(Measure::Largest));
        assert!(shift(Measure::Absolute) < 1e-12 && all[1].miss(orange).abs() < 1e-12 && all[3].miss(orange).abs() < 1e-12);
        for far in [0.5, 2.0, 10.0, 100.0] {
            let mut further = all.to_vec();
            further[4].b += far;
            assert!(best(&further, Measure::Absolute).at.distance(orange) < 1e-12);
            assert!(best(&further, Measure::Squares).at.distance(best(four, Measure::Squares).at) > shift(Measure::Squares));
        }

        // parallel lines: no crossing; least squares and least maximum: the whole
        // middle line, the dot its point nearest (0, 0); least absolute values: the
        // whole strip, the white point in it, its misses adding up to 2
        let parallel = scene(&TRIALS[3].2);
        let lines = parallel.used();
        assert!(crossings(lines).is_empty() && all_parallel(lines).is_some());
        for m in [Measure::Squares, Measure::Largest] {
            let b = best(lines, m);
            let Ties::Band { n, lo, hi } = b.all else { panic!("{}: {:?}", m.name(), b.all) };
            assert!(lo == hi && (lo - 0.5 * (lines[0].b + lines[1].b)).abs() < 1e-12 && b.at.perp_dot(n).abs() < 1e-12);
        }
        let strip = best(lines, Measure::Absolute);
        assert!(matches!(strip.all, Ties::Band { lo, hi, .. } if (hi - lo - 2.0).abs() < 1e-12));
        assert!((Measure::Absolute.total(lines, parallel.trial) - 2.0).abs() < 1e-12 && (strip.total - 2.0).abs() < 1e-12);
        let mut rng = Rng::new(9);
        for _ in 0..50 {
            let t = rng.range(-0.6, 1.4) as f64;
            let p = lines[0].a * t + lines[0].dir() * rng.range(-5.0, 5.0) as f64;
            assert!((Measure::Absolute.total(lines, p) - 2.0).abs() < 1e-12);
        }

        // nearly parallel: 5° apart; b₂ changed by 0.1 moves the crossing 1.15; with
        // each b off by up to 0.1 the answer can be up to 2.29 away, in a sliver
        let near = scene(&TRIALS[4].2);
        let lines = near.used();
        assert!(near.nudge && (lines[0].a.angle_to(lines[1].a).abs().to_degrees() - 5.0).abs() < 1e-9);
        let at = best(lines, Measure::Squares).at;
        let mut moved = lines.to_vec();
        moved[1].b += 0.1;
        assert_eq!(num(best(&moved, Measure::Squares).at.distance(at), 2), "1.15");
        let sliver = nudged(lines);
        let far = sliver.iter().map(|c| c.distance(at)).fold(0.0, f64::max);
        let narrow = sliver.iter().map(|c| (c - at).dot(lines[0].a).abs()).fold(0.0, f64::max);
        assert!(num(far, 2) == "2.29" && narrow < 0.11, "{far} {narrow}");
        // (and for the lines the scene opens with, a patch eight times smaller)
        let open = scene(&OPENING);
        let open_far = nudged(open.used()).iter().map(|c| c.distance(best(open.used(), Measure::Squares).at)).fold(0.0, f64::max);
        assert!(open_far * 8.0 < far, "{open_far}");
    }

    /// Lines on top of each other, all parallel, all through one point, far away,
    /// and with ties: every number drawn or read out is finite, and the picture says
    /// what has happened.
    #[test]
    fn degenerate_lines_leave_everything_finite() {
        let line = |a1: f64, a2: f64, b: f64| Line::new([a1, a2, b, 0.0, 0.0]);
        let cases: Vec<Vec<Line>> = vec![
            vec![line(0.6, 0.8, 1.0), line(0.6, 0.8, 1.0)],
            vec![line(0.6, 0.8, 1.0), line(-0.6, -0.8, -1.0)],
            (0..6).map(|i| line(0.0, 1.0, i as f64 * 0.3 - 0.4)).collect(),
            (0..5).map(|i| line(0.0, if i % 2 == 0 { 1.0 } else { -1.0 }, i as f64 * 0.3)).collect(),
            (0..6).map(|i| line((i as f64).cos(), (i as f64).sin(), 0.0)).collect(),
            vec![line(1.0, 0.0, 1e4), line(0.0, 1.0, -1e4), line(1.0, 1.0, 0.0)],
            vec![line(0.0, 1.0, 0.0), line(0.0, 1.0, 1.0), line(1.0, 0.0, 0.0)],
            vec![line(0.0, 1.0, 0.0), line(0.0, 1.0, 1.0), line(1.0, 0.0, 0.0), line(1.0, 0.0, 1.0)],
            vec![line(0.0, 1.0, 0.3), line(1e-9, 1.0, 0.31)],
        ];
        let (lo, hi) = (DVec2::new(-4.0, -3.0), DVec2::new(4.0, 3.0));
        let mut seen = Vec::new();
        for lines in &cases {
            for m in MEASURES {
                let b = best(lines, m);
                assert!(b.at.is_finite() && b.total.is_finite(), "{lines:?}: {}: {b:?}", m.name());
                seen.push(std::mem::discriminant(&b.all));
                for c in levels(lines, m, &b, lo, hi).into_iter().chain([b.total + 0.5]) {
                    assert!(level_curve(lines, m, &b, c, lo, hi).iter().flatten().all(|p| p.is_finite()));
                }
            }
            assert!(nudged(lines).iter().all(|p| p.is_finite()));
            assert!(normal_residue(lines).is_finite());
            for l in lines {
                for text in [equation_text(l), left_side(l), signed_text(l.miss(DVec2::ZERO), 2)] {
                    assert!(!text.contains("NaN") && !text.contains("inf"), "{text}");
                }
            }
        }
        // two parallel lines and one across them: for absolute values every point of
        // the third line between them is best, for the largest miss every point of
        // the middle line within 0.5 of the third; each marked at its middle
        let across = &cases[6];
        let (sum, most) = (best(across, Measure::Absolute), best(across, Measure::Largest));
        assert!(matches!(&sum.all, Ties::Corners(c) if c.len() == 2) && sum.at.distance(DVec2::new(0.0, 0.5)) < 1e-12, "{sum:?}");
        assert!(matches!(&most.all, Ties::Corners(c) if c.len() == 2 && c.iter().all(|p| (p.x.abs() - 0.5).abs() < 1e-12)) && most.at.distance(DVec2::new(0.0, 0.5)) < 1e-12, "{most:?}");
        // lines through one point: the system is solved there, by every measure
        let star = &cases[4];
        for m in MEASURES {
            let b = best(star, m);
            assert!(b.at.length() < 1e-12 && total_text(b.total) == "0");
        }
        // two crossing pairs of parallel lines: for absolute values the whole square
        // between them, its middle marked; ties of every kind were met
        let square = best(&cases[7], Measure::Absolute);
        assert!(matches!(&square.all, Ties::Corners(c) if c.len() == 4) && square.at.distance(DVec2::splat(0.5)) < 1e-12, "{square:?}");
        for kind in [Ties::One, Ties::Corners(Vec::new()), Ties::Band { n: DVec2::X, lo: 0.0, hi: 0.0 }] {
            assert!(seen.contains(&std::mem::discriminant(&kind)), "{kind:?} never met");
        }
        assert_eq!(num(-0.0001, 2), "0.00");
        assert_eq!(num(f64::NAN, 2), "?");
    }

    /// Every text the scene works out is markup that parses, with no word set in
    /// italic: the equations and their left-hand sides, misses and totals of every size
    /// (in the form computers write very large and very small numbers too, and for
    /// numbers that are not finite), the points read out, and the measures' formulas.
    #[test]
    fn every_text_is_markup_that_parses() {
        let fine = |text: &str| crate::typeset::inspect(text) == Default::default();
        let line = |a1: f64, a2: f64, b: f64| Line::new([a1, a2, b, 0.0, 0.0]);
        let mut lines: Vec<Line> = cases().into_iter().flatten().collect();
        lines.extend([line(0.0, 1.0, 0.0), line(-1.0, 0.0, -2e6), line(1.0, -1e-9, 3e5), line(0.6, -0.8, -0.0001)]);
        for l in &lines {
            for text in [format!("${}$", equation_text(l)), left_side(l)] {
                assert!(fine(&text), "{text}");
            }
        }
        for v in [0.0, -0.0001, 0.004, -0.4, 2.135, 1e-7, 3e-4, -12.0, 4.2e4, 2e5, -7e9] {
            for text in [signed_text(v, 2), total_text(v), num(v, 3), signed(v as f32, 4)] {
                assert!(fine(&format!("${text}$")), "{text}");
            }
            assert!(fine(&format!("$({}, {})$", num(v, 3), num(-v, 3))));
        }
        // (the totals and the normal equations' sum are finite whatever the lines:
        // `degenerate_lines_leave_everything_finite`; a coordinate that is not is "?")
        for v in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            for text in [signed_text(v, 2), num(v, 3)] {
                assert!(fine(&format!("${text}$")), "{text}");
            }
        }
        for m in MEASURES {
            for text in [m.symbol().to_string(), format!("{}   shown", m.symbol()), format!("same {} as the white point", m.symbol())] {
                assert!(fine(&text), "{text}");
            }
        }
        for (label, look_for, _) in &TRIALS {
            assert!(fine(label) && fine(look_for), "{look_for}");
        }
        assert_eq!(equation_text(&line(-0.28, 0.96, -1.6)), "−0.28x + 0.96y = −1.60");
        assert_eq!(left_side(&line(0.8, -0.6, 0.2)), "$0.80x − 0.60y = {}$");
    }

    /// The table and the picture keep to the canvas and clear of each other,
    /// whatever its shape, and a total's row lies in the table.
    #[test]
    fn the_table_and_the_picture_fit_every_canvas() {
        for x in [240.0, 300.0, 390.0, 578.0, 639.0, 640.0, 800.0, 1078.0, 1600.0, 2560.0] {
            for y in [180.0, 390.0, 600.0, 800.0, 1000.0, 1440.0] {
                for inset in [0.0, 46.0] {
                    let view = Vec2::new(x, y);
                    let layout = Layout::of(view, inset);
                    let (p0, p1) = layout.picture;
                    assert!(p0.is_finite() && p1.is_finite() && p1.x > p0.x && p1.y > p0.y, "{view}: {layout:?}");
                    if let Some((at, size)) = layout.table {
                        assert!(at.cmpge(Vec2::ZERO).all() && (at + size).cmple(view).all() && at.y >= inset, "{view}: {layout:?}");
                        let apart = p0.x >= at.x + size.x || p1.x <= at.x || p0.y >= at.y + size.y || p1.y <= at.y;
                        assert!(apart, "{view}: the picture's part covers the table");
                        for n in 2..=MOST {
                            let (t0, ts) = layout.table_now(n).unwrap_or_default();
                            for m in MEASURES {
                                let (r0, r1) = layout.total_row(n, m).unwrap_or_default();
                                assert!(r0.cmpge(t0).all() && r1.cmple(t0 + ts).all(), "{view}, {n} equations: {m:?}'s row");
                            }
                        }
                    }
                    let mut s = Overdet::create();
                    (s.canvas, s.inset) = (view, inset);
                    let View::Fit { center, half } = s.the_view() else { panic!() };
                    assert!(center.is_finite() && half.is_finite() && half.min_element() > 0.0);
                }
            }
        }
    }

    /// The level curves kept from frame to frame are never old ones: after every kind
    /// of change (a line slid, turned, or its b typed; another measure; another number
    /// of equations; an experiment; the view moved or zoomed) they are exactly, to the
    /// bit, the curves worked out afresh; and moving the white point, which they do not
    /// depend on, keeps them.
    #[test]
    fn the_level_curves_kept_are_the_ones_worked_out_afresh() {
        let mut s = Overdet::create();
        let (mut lo, mut hi) = (DVec2::new(-3.0, -2.6), DVec2::new(3.2, 2.9));
        let mut kept: Kept = None;
        let bits = |c: &[(Color, Vec<Vec2>)]| c.iter().map(|(color, run)| (*color, run.iter().map(|p| [p.x.to_bits(), p.y.to_bits()]).collect::<Vec<_>>())).collect::<Vec<_>>();
        let changes: [fn(&mut Overdet, &mut DVec2, &mut DVec2); 12] = [
            |_, _, _| {},
            |s, _, _| s.trial += DVec2::new(0.4, -0.3),
            |s, _, _| s.lines[1].slide_to(DVec2::new(0.3, 0.9)),
            |s, _, _| {
                let at = s.lines[2].handle() + DVec2::new(0.2, 0.5);
                s.lines[2].turn_to(at)
            },
            |s, _, _| s.lines[0].b += 0.1,
            |s, _, _| s.measure = Measure::Absolute,
            |s, _, _| s.measure = Measure::Largest,
            |s, _, _| s.set_count(6),
            |s, _, _| s.set_count(3),
            |_, lo, hi| (*lo, *hi) = (*lo + DVec2::new(0.5, 0.1), *hi + DVec2::new(0.5, 0.1)),
            |_, lo, hi| (*lo, *hi) = (*lo * 3.0, *hi * 3.0),
            |s, _, _| s.set_up(&TRIALS[3].2),
        ];
        for (i, change) in changes.into_iter().enumerate() {
            change(&mut s, &mut lo, &mut hi);
            let (lines, m) = (s.used(), s.measure);
            let b = best(lines, m);
            let before = kept.as_ref().map(|k| bits(&k.1));
            let now = bits(contours(&mut kept, lines, m, &b, lo, hi));
            assert_eq!(now, bits(contours(&mut None, lines, m, &b, lo, hi)), "change {i}");
            assert!(!now.is_empty(), "change {i}");
            if i == 1 {
                assert_eq!(before, Some(now), "the white point moved");
            }
        }
    }

    /// The numbers behind the captions (run with --ignored --nocapture).
    #[test]
    #[ignore]
    fn explore() {
        for (name, setup) in setups() {
            let s = scene(setup);
            println!("{name}");
            for (i, l) in s.used().iter().enumerate() {
                println!("  {} {}   grip {:.2?} handle {:.2?}   miss at white {:.3}", i + 1, equation_text(l), l.grip(), l.handle(), l.miss(s.trial));
            }
            let c = crossings(s.used());
            println!("  crossings {:.3?}", c);
            for m in MEASURES {
                let b = best(s.used(), m);
                println!("  {:<22} at {:.4?} total {:.4} {:?}", m.name(), b.at, b.total, b.all);
                let misses: Vec<String> = s.used().iter().map(|l| format!("{:.4}", l.miss(b.at))).collect();
                println!("      misses {}", misses.join(" "));
            }
        }
        // the wild equation: with and without it, and moved further off
        let wild = scene(&TRIALS[2].2);
        for (label, lines) in [("lines 1-4", wild.lines[..4].to_vec()), ("lines 1-5", wild.lines[..5].to_vec())] {
            let pts: Vec<String> = MEASURES.iter().map(|&m| format!("{:.4?}", best(&lines, m).at)).collect();
            println!("{label}: {}", pts.join("  "));
        }
        for far in [0.0, 1.0, 3.0, 10.0] {
            let mut lines = wild.lines[..5].to_vec();
            lines[4].b += far;
            let four = &wild.lines[..4];
            let moved: Vec<String> = MEASURES.iter().map(|&m| format!("{:.4}", best(&lines, m).at.distance(best(four, m).at))).collect();
            println!("line 5 {far} further: moved from the four-line points by {}", moved.join(" "));
        }
        let sq: Vec<f64> = wild.used().iter().map(|l| l.miss(wild.trial).powi(2)).collect();
        println!("squares at the white point {sq:.4?}");
        // the nearly parallel pair: the crossing's move, and the sliver
        let near = scene(&TRIALS[4].2);
        let at = best(near.used(), Measure::Squares).at;
        let mut moved = near.lines[..2].to_vec();
        moved[1].b += 0.1;
        println!("nearly parallel: angle {:.4} deg, crossing moves {:.4}", near.lines[0].a.angle_to(near.lines[1].a).to_degrees(), best(&moved, Measure::Squares).at.distance(at));
        let corners = nudged(near.used());
        println!("sliver corners {:.4?}, furthest {:.4}", corners, corners.iter().map(|c| c.distance(at)).fold(0.0, f64::max));
        let open = scene(&OPENING);
        println!("opening sliver furthest {:.4}", nudged(open.used()).iter().map(|c| c.distance(best(open.used(), Measure::Squares).at)).fold(0.0, f64::max));
        // the most a frame computes: six lines, every measure's best point, and the
        // level curves of each measure in a view of the region and of a wider one
        let six = Overdet::create().lines;
        for (lo, hi) in [(DVec2::new(-4.0, -3.0), DVec2::new(4.0, 3.0)), (DVec2::splat(-40.0), DVec2::splat(40.0))] {
            for m in MEASURES {
                let start = std::time::Instant::now();
                let mut pieces = 0;
                for _ in 0..100 {
                    let bests = MEASURES.map(|m| best(&six, m));
                    let b = &bests[m.index()];
                    for c in levels(&six, m, b, lo, hi).into_iter().chain([b.total + 1.0]) {
                        pieces += level_curve(&six, m, b, c, lo, hi).len();
                    }
                }
                println!("{:<22} box {hi}: {:.3} ms a frame, {} pieces", m.name(), start.elapsed().as_secs_f64() * 10.0, pieces / 100);
            }
        }
    }
}

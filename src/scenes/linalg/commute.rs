//! Matrices that commute: B and then A against A and then B, the commutator AB − BA
//! that is the difference, why two matrices commute when they keep each other's
//! eigenlines, why maps near the identity nearly commute, and two turns of a book in
//! space.

use super::*;
use crate::scenes::stats::plot::{self, Plot};
use glam::{DMat2, DMat3, DQuat, DVec2, DVec3};

pub const DEF: SceneDef = SceneDef {
    id: "commute",
    title: "Matrices that commute",
    group: SPECIAL,
    blurb: "When one map after another is the same in either order, and what the difference measures when it is not.",
    make: |_| Box::new(Commute::create()),
};

// ----- the arithmetic, in double precision -------------------------------------------

/// The largest entry either way that a matrix may be typed or chosen with.
const ENTRY_MAX: f64 = 10.0;
/// What counts as exactly zero for a number the size of the entries: the rounding of
/// a few products, and far below the hundredths the entries are typed in.
const EXACT: f64 = 1e-9;
/// Two matrices commute, as far as the numbers can tell, when AB − BA is no larger than
/// this share of ‖A‖ ‖B‖. (Larger than `EXACT` by enough that a matrix taken to be a
/// multiple of I, its other part `EXACT` of its size, commutes with everything here too.)
const COMMUTE_TOL: f64 = 1e-8;
/// The smallest e for the small maps I + eA and I + eB.
const E_MIN: f64 = 0.01;
/// Seconds each of the two steps of the film takes.
const STAGE_SECONDS: f64 = 1.6;

/// A matrix by its rows, [[a, b], [c, d]], the way matrices are written.
fn rows([a, b, c, d]: [f64; 4]) -> DMat2 {
    DMat2::from_cols(DVec2::new(a, c), DVec2::new(b, d))
}

/// A turn of the plane by `degrees`, anticlockwise.
fn turn(degrees: f64) -> DMat2 {
    DMat2::from_angle(degrees.to_radians())
}

fn trace(m: DMat2) -> f64 {
    m.x_axis.x + m.y_axis.y
}

/// The size of the largest entry.
fn size(m: DMat2) -> f64 {
    m.to_cols_array().iter().fold(0.0f64, |s, v| s.max(v.abs()))
}

/// ‖M‖: the square root of the sum of the squares of the entries.
fn norm(m: DMat2) -> f64 {
    m.to_cols_array().iter().map(|v| v * v).sum::<f64>().sqrt()
}

/// The commutator AB − BA.
fn bracket(a: DMat2, b: DMat2) -> DMat2 {
    a * b - b * a
}

/// Whether AB = BA, as far as numbers of their size can tell.
fn commute(a: DMat2, b: DMat2) -> bool {
    norm(bracket(a, b)) <= COMMUTE_TOL * norm(a) * norm(b)
}

/// (tr A / 2)² − det A, a quarter of the discriminant of λ² − (tr A) λ + det A:
/// positive for two real eigenvalues, negative for a complex pair, zero for a double
/// one. Worked out as ((a − d)/2)² + bc, so that a double eigenvalue gives 0 exactly.
fn quarter_disc(m: DMat2) -> f64 {
    let half = 0.5 * (m.x_axis.x - m.y_axis.y);
    half * half + m.y_axis.x * m.x_axis.y
}

/// The eigenvalues of a matrix and the lines through 0 it keeps.
#[derive(Clone, Copy, Debug, PartialEq)]
enum Eigen {
    /// Two real eigenvalues, the larger first.
    Real { l1: f64, l2: f64, lines: Lines },
    /// A complex pair re ± im i (im > 0): no line is kept.
    Complex { re: f64, im: f64 },
}

/// The eigenlines of a matrix with real eigenvalues, as unit directions.
#[derive(Clone, Copy, Debug, PartialEq)]
enum Lines {
    /// One for each eigenvalue, the larger's first.
    Two(DVec2, DVec2),
    /// A double eigenvalue with a single line (a shear is in it).
    One(DVec2),
    /// Every line through 0 (the matrix is a multiple of I).
    Every,
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

fn eigen(a: DMat2) -> Eigen {
    let scale = size(a);
    let s = 0.5 * trace(a);
    let q = quarter_disc(a);
    let flat = EXACT * scale * scale;
    if q < -flat {
        return Eigen::Complex { re: s, im: (-q).sqrt() };
    }
    let b = a - DMat2::IDENTITY * s;
    if q <= flat {
        let lines = if size(b) <= EXACT * scale { Lines::Every } else { Lines::One(null_direction(b)) };
        return Eigen::Real { l1: s, l2: s, lines };
    }
    let r = q.sqrt();
    Eigen::Real { l1: s + r, l2: s - r, lines: Lines::Two(null_direction(b - DMat2::IDENTITY * r), null_direction(b + DMat2::IDENTITY * r)) }
}

/// Whether `m` sends the line through 0 along the unit vector `v` into itself (as a
/// multiple of each of its points, 0 included).
fn keeps(m: DMat2, v: DVec2) -> bool {
    (m * v).perp_dot(v).abs() <= 1e-9 * size(m)
}

/// How far `m` turns the line along `v` off itself: 0 if it keeps it.
fn off_line(m: DMat2, v: DVec2) -> f64 {
    (m * v).perp_dot(v).abs()
}

/// The numbers a and b with B = aI + bA, for an A that is not a multiple of I and a B
/// that commutes with it (for any other B, the nearest such matrix's). A − (tr A/2) I,
/// the rest of A, is then a multiple of B − (tr B/2) I, the rest of B.
fn span_of(a: DMat2, b: DMat2) -> (f64, f64) {
    let (na, nb) = (a - DMat2::IDENTITY * (0.5 * trace(a)), b - DMat2::IDENTITY * (0.5 * trace(b)));
    let dot = |p: DMat2, q: DMat2| p.to_cols_array().iter().zip(q.to_cols_array()).map(|(x, y)| x * y).sum::<f64>();
    let beta = dot(nb, na) / dot(na, na).max(1e-300);
    (0.5 * (trace(b) - beta * trace(a)), beta)
}

/// Why the first map commutes with the second, or does not: what the first's
/// eigenvalues allow.
#[derive(Clone, Copy, Debug, PartialEq)]
enum Why {
    /// The first is a multiple of I: every matrix commutes with it.
    Scalar,
    /// The first has two eigenlines, and the second keeps this many of them (both
    /// exactly when they commute), and which one if it keeps one.
    Lines(usize, Option<usize>),
    /// The first keeps no line (`complex`) or one alone: the matrices that commute with
    /// it are aI + bA, and the second is one of them, with these a and b, or is not.
    Span { complex: bool, ab: Option<(f64, f64)> },
}

#[cfg(test)]
fn why(a: DMat2, b: DMat2) -> Why {
    why_given(a, b, commute(a, b))
}

/// The same, told whether they commute. (The small maps P = I + eA and Q = I + eB
/// commute exactly when A and B do, and A and B, of the size they are typed at, are
/// what tells it best: P and Q are close to I and to each other.)
fn why_given(a: DMat2, b: DMat2, together: bool) -> Why {
    match eigen(a) {
        Eigen::Real { lines: Lines::Every, .. } => Why::Scalar,
        // (that commuting matrices keep both lines is the theorem; the numbers only
        // say which line is kept when they do not commute, and then it is at most one)
        Eigen::Real { lines: Lines::Two(v1, v2), .. } => match (together, keeps(b, v1), keeps(b, v2)) {
            (true, ..) => Why::Lines(2, None),
            (false, true, false) => Why::Lines(1, Some(0)),
            (false, false, true) => Why::Lines(1, Some(1)),
            (false, true, true) => Why::Lines(1, Some(if off_line(b, v1) <= off_line(b, v2) { 0 } else { 1 })),
            (false, false, false) => Why::Lines(0, None),
        },
        Eigen::Real { lines: Lines::One(_), .. } => Why::Span { complex: false, ab: together.then(|| span_of(a, b)) },
        Eigen::Complex { .. } => Why::Span { complex: true, ab: together.then(|| span_of(a, b)) },
    }
}

/// "+ 1.41" or "− 0.50": a coefficient written after another term.
fn signed_term(v: f64) -> String {
    let n = num(v.abs(), 2);
    if v < 0.0 && n != "0.00" {
        format!("- {n}")
    } else {
        format!("+ {n}")
    }
}

impl Why {
    /// In words, for the maps named `p` and `q` (A and B, or P and Q).
    fn says(self, p: &str, q: &str) -> String {
        match self {
            Why::Scalar => format!("${p}$ is a multiple of $I$: every matrix commutes with it"),
            Why::Lines(2, _) => format!("${q}$ keeps both of ${p}$'s eigenlines"),
            Why::Lines(1, Some(0)) => format!("${q}$ keeps ${p}$'s orange eigenline, but not its purple one"),
            Why::Lines(1, _) => format!("${q}$ keeps ${p}$'s purple eigenline, but not its orange one"),
            Why::Lines(..) => format!("${q}$ keeps neither of ${p}$'s eigenlines"),
            Why::Span { complex, ab } => {
                let lead = if complex { format!("${p}$ keeps no line") } else { format!("${p}$ keeps one line alone") };
                match ab {
                    Some((x, y)) => format!("{lead}: only the matrices $aI + b{p}$ commute with it, and ${q} = {} I {} {p}$", num(x, 2), signed_term(y)),
                    None => format!("{lead}: only the matrices $aI + b{p}$ commute with it, and ${q}$ is not one"),
                }
            }
        }
    }
}

// ----- the film ----------------------------------------------------------------------

fn smooth(t: f64) -> f64 {
    let t = t.clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

/// Does the map turn the plane (complex eigenvalues, or two negative ones)? Sliding
/// every point straight to its place would then cut across the turn and shrink
/// everything on the way (a half turn would pass through a single point).
fn turns(m: DMat2) -> bool {
    let det = m.determinant();
    det > 0.0 && trace(m) < 2.0 * det.sqrt()
}

/// Where the plane is a share `s` of the way through the map `m`: every point slides
/// straight to its place, except that a map that turns the plane turns it by the turn
/// in it (the angle of m = R(θ) P, P symmetric), while the rest slides straight.
fn along(m: DMat2, s: f64) -> DMat2 {
    if s >= 1.0 {
        return m;
    }
    if s <= 0.0 {
        return DMat2::IDENTITY;
    }
    if !turns(m) {
        return DMat2::IDENTITY * (1.0 - s) + m * s;
    }
    let theta = (m.x_axis.y - m.y_axis.x).atan2(m.x_axis.x + m.y_axis.y);
    let rest = DMat2::from_angle(-theta) * m;
    DMat2::from_angle(theta * s) * (DMat2::IDENTITY * (1.0 - s) + rest * s)
}

/// The plane at `progress`: `first` acts from 0 to 1, then `second` from 1 to 2.
fn at(first: DMat2, second: DMat2, progress: f64) -> DMat2 {
    if progress <= 1.0 {
        along(first, smooth(progress))
    } else {
        along(second, smooth(progress - 1.0)) * first
    }
}

/// The F painted on the unit square, as three bars (corners anticlockwise), and its
/// outline. Lopsided on purpose: turned, it is still an F; mirrored, it is not.
const F_BARS: [[DVec2; 4]; 3] = [
    [DVec2::new(0.18, 0.12), DVec2::new(0.36, 0.12), DVec2::new(0.36, 0.88), DVec2::new(0.18, 0.88)],
    [DVec2::new(0.36, 0.70), DVec2::new(0.82, 0.70), DVec2::new(0.82, 0.88), DVec2::new(0.36, 0.88)],
    [DVec2::new(0.36, 0.42), DVec2::new(0.66, 0.42), DVec2::new(0.66, 0.58), DVec2::new(0.36, 0.58)],
];
const F_OUTLINE: [DVec2; 10] = [
    DVec2::new(0.18, 0.12),
    DVec2::new(0.36, 0.12),
    DVec2::new(0.36, 0.42),
    DVec2::new(0.66, 0.42),
    DVec2::new(0.66, 0.58),
    DVec2::new(0.36, 0.58),
    DVec2::new(0.36, 0.70),
    DVec2::new(0.82, 0.70),
    DVec2::new(0.82, 0.88),
    DVec2::new(0.18, 0.88),
];
const SQUARE: [DVec2; 4] = [DVec2::ZERO, DVec2::X, DVec2::ONE, DVec2::Y];

// ----- space --------------------------------------------------------------------------

/// A turn of space by `degrees` about axis `k` (0, 1: x, y), by the right-hand rule.
fn about(k: usize, degrees: f64) -> DMat3 {
    let r = degrees.to_radians();
    if k == 0 {
        DMat3::from_rotation_x(r)
    } else {
        DMat3::from_rotation_y(r)
    }
}

/// The book at `progress` through two turns by θ: about x and then about y
/// (`x_first`), or the other way round.
fn book_pose(theta: f64, progress: f64, x_first: bool) -> DMat3 {
    let (k1, k2) = if x_first { (0, 1) } else { (1, 0) };
    if progress <= 1.0 {
        about(k1, theta * smooth(progress))
    } else {
        about(k2, theta * smooth(progress - 1.0)) * about(k1, theta)
    }
}

/// The two ends: R_y R_x (x first) and R_x R_y.
fn space_ends(theta: f64) -> (DMat3, DMat3) {
    (about(1, theta) * about(0, theta), about(0, theta) * about(1, theta))
}

/// The angle (degrees) of the one turn that takes the rotation `a` to `b`, read from
/// a quaternion (which stays exact for small angles, where the trace does not).
fn apart(a: DMat3, b: DMat3) -> f64 {
    let q = DQuat::from_mat3(&(a.transpose() * b));
    let n = q.length();
    if !(n > 0.0 && n.is_finite()) {
        return 0.0;
    }
    let q = q / n;
    (2.0 * DVec3::new(q.x, q.y, q.z).length().atan2(q.w.abs())).to_degrees()
}

fn norm3(m: DMat3) -> f64 {
    m.to_cols_array().iter().map(|v| v * v).sum::<f64>().sqrt()
}

/// The book: half its width (x), its height (y) and its thickness (z), lying with
/// its cover up and its spine along the left edge.
const BOOK: DVec3 = DVec3::new(0.5, 0.7, 0.12);
/// How far either side of the middle the two books stand, along the screen's width.
const BOOKS_APART: f32 = 1.35;
/// How far each book's axes are drawn either way.
const AXIS_LEN: f32 = 1.05;

/// The direction the space is seen from at first: in front, to the right and above.
const YAW: f32 = -1.05;
const PITCH: f32 = 0.42;
/// (The tangent of half the camera's field of view.)
const TAN_HALF_FOV: f32 = 0.3203;

/// The camera's right and up directions in the view space is seen from at first.
fn screen_axes() -> (Vec3, Vec3) {
    let right = Vec3::new(-YAW.sin(), YAW.cos(), 0.0);
    let up = Vec3::new(-PITCH.sin() * YAW.cos(), -PITCH.sin() * YAW.sin(), PITCH.cos());
    (right, up)
}

// ----- experiments and quick choices -------------------------------------------------

/// What an experiment does once its film has played.
#[derive(Clone, Copy, Debug, PartialEq)]
enum Then {
    Rest,
    /// Turn B's axes (all of B, turned with them) by this many degrees.
    TurnB(f64),
    /// Halve e this many times.
    Halve(usize),
}

struct Trial {
    label: &'static str,
    look_for: &'static str,
    space: bool,
    /// The matrices, by rows.
    a: [f64; 4],
    b: [f64; 4],
    small: bool,
    theta: f64,
    then: Then,
}

const QUARTER: [f64; 4] = [0.0, -1.0, 1.0, 0.0];
const C45: f64 = std::f64::consts::FRAC_1_SQRT_2;

const TRIALS: [Trial; 5] = [
    Trial {
        label: "a turn and an uneven stretch",
        look_for: r"$A$ stretches along the $x$ axis, to 1.6 times, and squashes along $y$, to 0.6; $B$ turns a quarter turn. $B$, then $A$ (yellow) turns the letter and then stretches it across: it ends wide and low. $A$, then $B$ (white) stretches it first and stands the stretch upright: it ends tall and narrow. $AB \ne BA$, and the red arrows from one end to the other are the columns of $AB - BA$, of size 1.41. Why: a quarter turn keeps no line, so $B$ moves both of $A$'s eigenlines, the two axes.",
        space: false,
        a: [1.6, 0.0, 0.0, 0.6],
        b: QUARTER,
        small: false,
        theta: 90.0,
        then: Then::Rest,
    },
    Trial {
        label: "two turns",
        look_for: r"$A$ turns by 45°, $B$ by 90°. Either way round the letter ends turned by 135°: $AB = BA$, and the yellow end lies exactly on the white one. A turn by anything but 0° or 180° keeps no line (its eigenvalues are complex), so there are no eigenlines to compare. The matrices that commute with such a turn are exactly $aI + bA$, the ones that turn the plane and stretch it evenly: here $B = -I + 1.41A$. No uneven stretch is among them.",
        space: false,
        a: [C45, -C45, C45, C45],
        b: QUARTER,
        small: false,
        theta: 90.0,
        then: Then::Rest,
    },
    Trial {
        label: "the same axes, then turned",
        look_for: r"$A$ and $B$ both stretch along the $x$ and $y$ axes, by different amounts: $B$'s blue eigenlines lie on $A$'s, and the two orders end on the same letter, $AB = BA$. Then $B$'s axes turn by 30°: its lines leave $A$'s, and the ends part (the red gap, of size 0.92). Drag the blue dot to turn them back. Both matrices are symmetric, and then $(AB)^\top = B^\top A^\top = BA$: their product is symmetric exactly when they commute.",
        space: false,
        a: [2.0, 0.0, 0.0, 0.5],
        b: [0.6, 0.0, 0.0, 1.6],
        small: false,
        theta: 90.0,
        then: Then::TurnB(30.0),
    },
    Trial {
        label: "small maps nearly commute",
        look_for: r"Two shears, along $x$ ($A$) and along $y$ ($B$), made small: $P = I + eA$ and $Q = I + eB$. Multiplied out, the terms in $e$ cancel, and $PQ - QP = e^2(AB - BA)$ exactly. At $e = 1$ the gap is 1.41; each halving of $e$ divides it by 4, to 0.35, 0.088, and 0.022 at $e = 1/8$. On logarithmic scales that is a straight line of slope 2 (the chart). The nearer two maps are to the identity, the more nearly they commute.",
        space: false,
        a: [0.0, 1.0, 0.0, 0.0],
        b: [0.0, 0.0, 1.0, 0.0],
        small: true,
        theta: 90.0,
        then: Then::Halve(3),
    },
    Trial {
        label: "in space: two quarter turns of a book",
        look_for: r"A book turned a quarter turn about the $x$ axis and then a quarter turn about $y$ (yellow), against the same two turns in the other order (white). They end facing different ways: one turn of 120° takes one end to the other, and $R_y R_x \ne R_x R_y$. Make $\theta$ smaller and the ends come closer, the gap falling as $\theta^2$ for small angles. (At 180° they meet again: two half turns about the $x$ and $y$ axes commute.)",
        space: true,
        a: [1.6, 0.0, 0.0, 0.6],
        b: QUARTER,
        small: false,
        theta: 90.0,
        then: Then::Rest,
    },
];

/// Quick choices of A: (button, the matrix by rows).
const A_PRESETS: [(&str, [f64; 4]); 5] = [
    ("stretch", [2.0, 0.0, 0.0, 0.5]),
    ("turn", QUARTER),
    ("shear", [1.0, 1.0, 0.0, 1.0]),
    ("slanted lines", [1.5, 1.0, 0.0, 0.5]),
    ("$2I$", [2.0, 0.0, 0.0, 2.0]),
];

/// Quick choices of B, the last four made from A: (button, what it is).
#[derive(Clone, Copy, Debug, PartialEq)]
enum BChoice {
    Fixed([f64; 4]),
    Inverse,
    Square,
    Poly,
    Scalar,
}

const B_PRESETS: [(&str, BChoice); 7] = [
    ("stretch", BChoice::Fixed([0.6, 0.0, 0.0, 1.6])),
    ("turn", BChoice::Fixed(QUARTER)),
    ("shear", BChoice::Fixed([1.0, 0.0, 1.0, 1.0])),
    ("$A^{-1}$", BChoice::Inverse),
    ("$A^2$", BChoice::Square),
    ("$A^2 - 3A$", BChoice::Poly),
    ("$3I$", BChoice::Scalar),
];

/// The matrix a choice of B is, for this A (none if it has no inverse, or entries
/// beyond what can be typed).
fn b_choice(c: BChoice, a: DMat2) -> Option<DMat2> {
    let m = match c {
        BChoice::Fixed(r) => rows(r),
        BChoice::Inverse => {
            let det = a.determinant();
            if det.abs() <= EXACT * size(a) * size(a) {
                return None;
            }
            a.inverse()
        }
        BChoice::Square => a * a,
        BChoice::Poly => a * a - a * 3.0,
        BChoice::Scalar => DMat2::IDENTITY * 3.0,
    };
    (m.is_finite() && size(m) <= ENTRY_MAX).then_some(m)
}

// ----- the setting, and how a change of it is followed ----------------------------------

/// What glides when a button changes it: the two matrices, e, how far between A and
/// B and the small maps the film's maps are (`w`, 0 or 1 but on the way), how far the
/// film has got (0 to 2), and the angle of the turns in space (degrees).
#[derive(Clone, Copy, Debug, PartialEq)]
struct Setting {
    a: DMat2,
    b: DMat2,
    e: f64,
    w: f64,
    p: f64,
    theta: f64,
}

/// Where e is among a setting's numbers (it is followed by its own size).
const E_AT: usize = 8;

impl Setting {
    fn numbers(&self) -> [f64; 12] {
        let (a, b) = (self.a.to_cols_array(), self.b.to_cols_array());
        [a[0], a[1], a[2], a[3], b[0], b[1], b[2], b[3], self.e, self.w, self.p, self.theta]
    }

    fn from_numbers(v: [f64; 12]) -> Setting {
        Setting { a: DMat2::from_cols_array(&[v[0], v[1], v[2], v[3]]), b: DMat2::from_cols_array(&[v[4], v[5], v[6], v[7]]), e: v[8], w: v[9], p: v[10], theta: v[11] }
    }

    /// The two maps the film composes: A and B, or the small maps P = I + eA and
    /// Q = I + eB (and between the two while one gives way to the other).
    fn maps(&self) -> (DMat2, DMat2) {
        let small = |m: DMat2| m + (DMat2::IDENTITY + m * (self.e - 1.0)) * self.w;
        (small(self.a), small(self.b))
    }
}

/// How fast a change of setting is glided to: each of the two eases one after the
/// other covers two thirds of what is left in this many seconds.
const FOLLOW: f64 = 0.09;

/// A setting as asked for, and as drawn: the drawn one follows it through `between`,
/// two eases one after the other, so that it sets off gently and arrives exactly.
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

    /// Move the drawn setting `dt` seconds nearer to the one asked for.
    fn step(&mut self, dt: f64) {
        if self.arrived() || dt.is_nan() || dt <= 0.0 {
            return;
        }
        let k = 1.0 - (-dt / FOLLOW).exp();
        let (to, mut between, mut shown) = (self.to.numbers(), self.between.numbers(), self.shown.numbers());
        let mut apart = 0.0f64;
        for i in 0..to.len() {
            between[i] += (to[i] - between[i]) * k;
            shown[i] += (between[i] - shown[i]) * k;
            // (e is measured against its own size: it is followed down to a hundredth)
            let scale = if i == E_AT { to[i].abs().max(E_MIN) } else { 1.0 };
            apart = apart.max((to[i] - between[i]).abs() / scale).max((to[i] - shown[i]).abs() / scale);
        }
        // (exactly there in the end: the picture then stops changing)
        if apart < 2e-4 || !apart.is_finite() {
            *self = Glide::at(self.to);
        } else {
            (self.between, self.shown) = (Setting::from_numbers(between), Setting::from_numbers(shown));
        }
    }
}

/// What is still to come of an experiment, a step at a time.
#[derive(Clone, Copy, Debug, PartialEq)]
enum Act {
    /// Glide to this setting (an experiment's, once the film is back at its start).
    Aim(Setting),
    /// The film, from where it is to the end.
    Play,
    /// A pause of so many seconds.
    Hold(f64),
    /// B's axes turned by `by` degrees over `secs` seconds; `done` so far, from `from`.
    TurnB { from: Option<DMat2>, by: f64, secs: f64, done: f64 },
    /// Halve e (the glide then takes it there).
    Halve,
}

/// Seconds of pause between the film's end and what an experiment does next, and
/// between one halving of e and the next.
const PAUSE: f64 = 0.8;
const TURN_SECONDS: f64 = 2.4;

// ----- where things go -------------------------------------------------------------------

/// The screen panel's width on a wide canvas (logical pixels), and its inner margin.
const PANEL_W: f32 = 330.0;
const PAD: f32 = 12.0;
/// The widest number a matrix on the panel holds, in characters (`hud_matrix`).
const CELL: usize = 5;
/// The rows of the panel.
const HEAD_H: f32 = 28.0;
const LINE_H: f32 = 18.0;
const CHART_H: f32 = 96.0;

/// What the panel shows: the plane's matrices (with the chart of the small maps), or
/// the turns in space.
#[derive(Clone, Copy, Debug, PartialEq)]
enum Shows {
    Plane { small: bool },
    Space,
}

#[derive(Clone, Copy, Debug)]
struct Layout {
    /// A phone held upright: the panel goes below the picture.
    narrow: bool,
    /// The part of the canvas the picture has to itself.
    picture: (Vec2, Vec2),
    /// The screen panel, if the canvas can spare the room (the numbers are in the side
    /// panel too), and the size of its numbers.
    panel: Option<(Vec2, Vec2)>,
    text: f32,
}

/// The height of a matrix of `n` rows on the panel at text size `s`.
fn matrix_h(n: usize, s: f32) -> f32 {
    matrix_size(n, n, CELL, s).y
}

/// How tall the panel's contents are, at text size `s`.
fn panel_height(shows: Shows, s: f32, narrow: bool) -> f32 {
    let caption = if narrow { LINE_H + 4.0 } else { 0.0 };
    match shows {
        Shows::Plane { small } => {
            let rows = 3.0 * (matrix_h(2, s) + 10.0);
            let chart = if small { LINE_H + 6.0 + 22.0 + CHART_H + 30.0 } else { 0.0 };
            2.0 * PAD + HEAD_H + rows + chart + caption
        }
        Shows::Space => 2.0 * PAD + HEAD_H + 3.0 * (matrix_h(3, s) + 10.0) + 2.0 * LINE_H + caption,
    }
}

impl Layout {
    /// (On a phone the panel's height glides from what it was to what it is to be, as
    /// the chart comes and goes: `height` is where it has got to, if not there.)
    fn of(view: Vec2, inset: f32, shows: Shows, height: Option<f32>) -> Layout {
        let narrow = view.x < 620.0 && view.y >= 560.0;
        let text = if narrow { 12.0 } else { 12.5 };
        let whole = Layout { narrow, picture: (Vec2::new(0.0, inset), view), panel: None, text };
        if narrow {
            // (whether it fits is decided by the larger of the two, so that it does
            // not come and go on the way)
            let full = panel_height(shows, text, true);
            let h = height.unwrap_or(full);
            // (the picture keeps at least the top 42 %)
            let top = (view.y - h).max(view.y * 0.42);
            if view.y - (view.y - h.max(full)).max(view.y * 0.42) < h.max(full) - 1.0 || view.x < 330.0 {
                return whole;
            }
            return Layout { narrow, picture: (Vec2::new(0.0, inset), Vec2::new(view.x, top)), panel: Some((Vec2::new(0.0, top), view)), text };
        }
        let top = inset + 12.0;
        let h = panel_height(shows, text, false);
        if view.x < PANEL_W + 360.0 || top + h + 12.0 > view.y {
            return whole;
        }
        let min = Vec2::new(view.x - PANEL_W - 12.0, top);
        Layout { narrow, picture: (Vec2::new(0.0, inset), Vec2::new(min.x - 8.0, view.y)), panel: Some((min, Vec2::new(view.x - 12.0, top + h))), text }
    }

    /// The part of the canvas the scene's frame is fitted into: the picture's, less
    /// the heading at its top and the caption at its foot.
    fn frame_room(&self) -> (Vec2, Vec2) {
        let (min, max) = self.picture;
        let foot = if self.narrow || self.panel.is_none() { 10.0 } else { 46.0 };
        (min + Vec2::new(14.0, 64.0), max - Vec2::new(14.0, foot))
    }
}

/// Room kept round the picture for the names written beside things, logical pixels.
const NAME_ROOM: Vec2 = Vec2::new(70.0, 40.0);
/// The closest the view comes, in logical pixels per unit.
const MOST_PER_UNIT: f32 = 200.0;

fn inside_rect(p: Vec2, (min, max): (Vec2, Vec2)) -> bool {
    p.cmpge(min).all() && p.cmple(max).all()
}

/// A number for a matrix on the screen, as a formula, in five characters at most: two
/// decimals, then one, then none.
fn cell(v: f64) -> String {
    let a = v.abs();
    let number = if !v.is_finite() {
        "?".into()
    } else if a < 9.995 {
        num(v, 2)
    } else if a < 99.95 {
        num(v, 1)
    } else {
        num(v, 0)
    };
    format!("${number}$")
}

/// A size for the screen: two decimals, or two significant figures when it is
/// smaller than a tenth (and 0 only when it is 0).
fn amount(d: f64) -> String {
    if !d.is_finite() {
        "?".into()
    } else if d == 0.0 || d >= 0.0995 {
        num(d, 2)
    } else {
        figures(d)
    }
}

/// A positive number to two significant figures: 1.41, 0.014, 1.4e−4 (set as a power
/// of ten in a formula).
fn figures(v: f64) -> String {
    if !(v.is_finite() && v > 0.0) {
        return num(v, 2);
    }
    if v >= 0.001 {
        let decimals = (1.0 - v.log10().floor()).clamp(0.0, 4.0) as usize;
        num(v, decimals)
    } else {
        format!("{v:.1e}").replace('-', "−")
    }
}

/// An angle in degrees for the screen, to a tenth (two places below 10°).
fn deg(v: f64) -> String {
    format!("{}°", num(v, if v.abs() < 9.995 { 2 } else { 1 }))
}

/// A matrix's rows of numbers as `hud_matrix` takes them, all in one colour.
fn entries2(m: DMat2, c: Color) -> Vec<Vec<(String, Color)>> {
    (0..2).map(|i| (0..2).map(|j| (cell(m.col(j)[i]), c)).collect()).collect()
}

fn entries3(m: DMat3, c: Color) -> Vec<Vec<(String, Color)>> {
    (0..3).map(|i| (0..3).map(|j| (cell(m.col(j)[i]), c)).collect()).collect()
}

/// The caption's lines as last broken, kept with what they were broken for.
struct Caption {
    of: (Vec<&'static str>, f32),
    lines: Vec<String>,
    widest: f32,
}

/// `parts` side by side as long as they fit across `room`, on further lines when they
/// do not, and the widest line's width.
fn caption_lines(f: &Frame, parts: &[&str], room: f32, size: f32) -> (Vec<String>, f32) {
    let gap = "      ";
    let measure = |t: &str| f.text_size(t, size).x;
    let mut lines: Vec<String> = Vec::new();
    for part in parts {
        let fits = lines.last().is_some_and(|last| measure(last) + text_width(gap, size) + measure(part) <= room);
        match lines.last_mut() {
            Some(last) if fits => {
                last.push_str(gap);
                last.push_str(part);
            }
            _ => lines.push(part.to_string()),
        }
    }
    let widest = lines.iter().map(|l| measure(l)).fold(0.0, f32::max);
    (lines, widest)
}

/// A text broken into lines no wider than `room` at `size` (between words, never
/// inside a formula).
fn broken(f: &Frame, text: &str, room: f32, size: f32) -> Vec<String> {
    let mut words: Vec<String> = Vec::new();
    let mut inside = false;
    let mut word = String::new();
    for c in text.chars() {
        if c == '$' {
            inside = !inside;
        }
        if c == ' ' && !inside {
            words.push(std::mem::take(&mut word));
        } else {
            word.push(c);
        }
    }
    words.push(word);
    // (each word measured alone, which the frame keeps from one frame to the next: a
    // heading whose numbers change as something glides is broken afresh in every frame)
    let space = f.text_size("x x", size).x - f.text_size("xx", size).x;
    let mut lines: Vec<(String, f32)> = Vec::new();
    for w in words {
        let width = f.text_size(&w, size).x;
        match lines.last_mut() {
            Some((last, used)) if *used + space + width <= room => {
                last.push(' ');
                last.push_str(&w);
                *used += space + width;
            }
            _ => lines.push((w, width)),
        }
    }
    lines.into_iter().map(|(line, _)| line).collect()
}

/// The heading's lines as last broken, kept with what they were broken for.
struct Heading {
    of: (String, String, f32),
    first: (Vec<String>, f32),
    second: (Vec<String>, f32),
    widest: f32,
}

// ----- the parts of the picture drawn as lines ----------------------------------------------

/// The parts whose number of points changes as the picture moves, each in a batch of
/// its own (different from the usual one by a hair of its width), so that the parts
/// that stay the same can still be compared point for point from one frame to the
/// next: that is how `eidos motion` tells smooth motion from jumps.
#[derive(Clone, Copy, PartialEq)]
enum Part {
    /// The eigenlines, which come and go with the eigenvalues.
    Eigen,
    /// B's, dashed.
    EigenB,
    /// The outlines after the first step, there once it has been taken.
    Stops,
    /// The red arrows, there at the film's end.
    Gap,
    /// The axes of space and the books.
    Space,
}

fn lines_for<'a>(f: &'a mut Frame, part: Part) -> &'a mut crate::gfx::draw::LineBatch {
    let k = 1.0 + 1e-4 * (1 + part as u8) as f32;
    let dash = if part == Part::EigenB { 9.0 } else { 0.0 };
    f.dl.lines_with(LineKind::Blend, LineStyle { width_scale: k, dash, duty: 0.55, ..Default::default() })
}

/// An arrow from `a` to `b` in the batch of `part`, its head a fixed size on the screen.
fn arrow_in(f: &mut Frame, part: Part, a: Vec2, b: Vec2, width: f32, c: Color) {
    let d = b - a;
    let len = d.length();
    if len.is_nan() || len <= 1e-6 || !a.is_finite() || !b.is_finite() {
        return;
    }
    let dir = d / len;
    let wpp = f.cam.world_per_px(b.extend(0.0)) * f.px;
    let head = ((width * 3.2 + 5.0) * wpp).min(len * 0.6);
    let side = dir.perp() * (head * 0.42);
    let base = b - dir * head;
    f.dl.tris_alpha.tri(b.extend(0.0), (base + side).extend(0.0), (base - side).extend(0.0), c);
    lines_for(f, part).segment(a.extend(0.0), (b - dir * (head * 0.8)).extend(0.0), width, c);
}

/// The part of the line through 0 along `v` inside the box `lo..hi`.
fn line_in(v: DVec2, lo: Vec2, hi: Vec2) -> Option<(Vec2, Vec2)> {
    let (lo, hi) = (lo.as_dvec2(), hi.as_dvec2());
    let reach = (hi - lo).length() + lo.length() + hi.length();
    let (a, b) = (-v * reach, v * reach);
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
    (t0 < t1).then(|| ((a + d * t0).as_vec2(), (a + d * t1).as_vec2()))
}

// ----- the scene -------------------------------------------------------------------------

pub struct Commute {
    glide: Glide,
    /// In space, or in the plane.
    space: bool,
    /// The small maps I + eA and I + eB in place of A and B.
    small: bool,
    eigenlines: bool,
    /// What is still to come of an experiment (or of the play button).
    acts: Vec<Act>,
    /// A drag of the orange dot (0) or the blue one (1): the matrix as it was when the
    /// drag began, and the angle of the dot then.
    turning: Option<(usize, DMat2, f64)>,
    /// The eigenline directions drawn last frame (A's two, B's two): a line's name and
    /// the blue dot keep their end of it as the matrix changes.
    last_lines: [Option<DVec2>; 4],
    /// The matrices as they were when a glide to others began: their eigenlines fade
    /// out as the new ones fade in, rather than swinging through all that lies between
    /// (where two eigenvalues meet, the lines turn faster than the eye can follow).
    lines_from: Option<(DMat2, DMat2)>,
    caption: Option<Caption>,
    heading: Option<Heading>,
    /// The canvas last drawn on and how much of its top the application's buttons
    /// cover: they decide where the panel goes, and so where the picture is.
    canvas: Vec2,
    inset: f32,
    measured: bool,
    new_view: bool,
    /// The height of the panel on a phone, as it glides to a new one.
    panel_h: Option<f32>,
}

impl Commute {
    /// (Nothing here needs the graphics device, so the tests can make one too.)
    fn create() -> Self {
        let start = Setting { a: DMat2::IDENTITY, b: DMat2::IDENTITY, e: 1.0, w: 0.0, p: 0.0, theta: 90.0 };
        let mut scene = Commute {
            glide: Glide::at(start),
            space: false,
            small: false,
            eigenlines: true,
            acts: Vec::new(),
            turning: None,
            last_lines: [None; 4],
            lines_from: None,
            caption: None,
            heading: None,
            canvas: Vec2::new(1080.0, 1000.0),
            inset: 0.0,
            measured: false,
            new_view: false,
            panel_h: None,
        };
        scene.set_up(0);
        // (opening, there is nothing to glide from)
        if let Some(Act::Aim(s)) = scene.acts.first().copied() {
            scene.glide = Glide::at(s);
            scene.acts.remove(0);
        }
        scene.new_view = false;
        scene
    }

    /// Set up experiment `k` of `TRIALS`: the film glides back to its start, then the
    /// matrices to the experiment's (so that nothing but the eigenlines and the numbers
    /// change while they do), and then it plays.
    fn set_up(&mut self, k: usize) {
        let t = &TRIALS[k];
        (self.space, self.small, self.eigenlines) = (t.space, t.small, true);
        self.glide.to.p = 0.0;
        let to = Setting { a: rows(t.a), b: rows(t.b), e: 1.0, w: t.small as u8 as f64, p: 0.0, theta: t.theta };
        self.acts = vec![Act::Aim(to), Act::Play];
        match t.then {
            Then::Rest => {}
            Then::TurnB(by) => self.acts.extend([Act::Hold(PAUSE), Act::TurnB { from: None, by, secs: TURN_SECONDS, done: 0.0 }]),
            Then::Halve(n) => {
                for _ in 0..n {
                    self.acts.extend([Act::Hold(PAUSE), Act::Halve]);
                }
            }
        }
        self.turning = None;
        self.new_view = true;
    }

    /// Play the film from the start (the picture glides back to it first).
    fn replay(&mut self) {
        self.by_hand();
        self.glide.to.p = 0.0;
        self.acts = vec![Act::Play];
    }

    /// Anything the visitor changes by hand ends what an experiment was still to do,
    /// but for the setting it was on its way to, which is gone to at once, and the
    /// film, which plays on.
    fn by_hand(&mut self) {
        let mut film = false;
        if let Some(Act::Aim(to)) = self.acts.first().copied() {
            self.lines_from = Some((self.glide.shown.a, self.glide.shown.b));
            self.glide.to = Setting { p: self.glide.to.p, ..to };
            self.acts.remove(0);
        }
        if let Some(Act::Play) = self.acts.first() {
            film = true;
        }
        self.acts.clear();
        if film {
            self.acts.push(Act::Play);
        }
    }

    /// The small maps in place of A and B, or A and B again: the picture glides from
    /// the ones to the others.
    fn set_small(&mut self, small: bool) {
        self.by_hand();
        if small != self.small {
            self.new_view = true;
        }
        self.small = small;
        self.glide.to.w = small as u8 as f64;
    }

    /// Ask for new matrices by a button or a typed number: the picture glides to them.
    fn aim(&mut self, a: DMat2, b: DMat2) {
        self.lines_from = Some((self.glide.shown.a, self.glide.shown.b));
        self.glide.to.a = a;
        self.glide.to.b = b;
        self.by_hand();
        self.new_view = true;
    }

    fn shown(&self) -> Setting {
        self.glide.shown
    }

    /// The maps the plane's film composes, as drawn.
    fn maps(&self) -> (DMat2, DMat2) {
        self.shown().maps()
    }

    /// Whether they are the small maps (or nearer them than A and B, on the way).
    fn small_shown(&self) -> bool {
        self.shown().w >= 0.5
    }

    /// Their names.
    fn names(&self) -> (&'static str, &'static str) {
        if self.small_shown() {
            ("P", "Q")
        } else {
            ("A", "B")
        }
    }

    /// Move everything on: `dt` seconds of the scene's time (the film, the pauses, the
    /// turning of B) and `real_dt` of the clock's (a change of setting glides by the
    /// clock, so that it is seen through even while the scene's time stands still).
    fn advance(&mut self, dt: f64, real_dt: f64) {
        self.glide.step(real_dt);
        let dt = if dt.is_finite() { dt.max(0.0) } else { 0.0 };
        // (each act waits for the glide to arrive: a new setting is reached first)
        while self.glide.arrived() {
            let Some(act) = self.acts.first_mut() else { break };
            match act {
                Act::Aim(to) => {
                    let to = *to;
                    self.lines_from = Some((self.glide.shown.a, self.glide.shown.b));
                    self.glide.to = to;
                    self.acts.remove(0);
                }
                Act::Play => {
                    let p = (self.glide.to.p + dt / STAGE_SECONDS).min(2.0);
                    self.glide.now(|s| s.p = p);
                    if p >= 2.0 {
                        self.acts.remove(0);
                    }
                    break;
                }
                Act::Hold(left) => {
                    *left -= dt;
                    if *left > 0.0 {
                        break;
                    }
                    self.acts.remove(0);
                    break;
                }
                Act::TurnB { from, by, secs, done } => {
                    let start = *from.get_or_insert(self.glide.to.b);
                    *done = (*done + dt).min(*secs);
                    let r = turn(*by * smooth(*done / *secs));
                    let b = r * start * r.transpose();
                    let finished = *done >= *secs;
                    self.glide.now(|s| s.b = b);
                    if finished {
                        self.acts.remove(0);
                    }
                    break;
                }
                Act::Halve => {
                    self.glide.to.e = (self.glide.to.e * 0.5).max(E_MIN);
                    self.acts.remove(0);
                }
            }
        }
    }

    fn shows(&self) -> Shows {
        if self.space {
            Shows::Space
        } else {
            Shows::Plane { small: self.small }
        }
    }

    /// Everything the film draws in the plane, from first frame to last, for the
    /// setting asked for (with small maps, for the largest e: the view stays put as e
    /// changes, and the gap is seen to close).
    fn bounds(&self) -> (Vec2, Vec2) {
        // (the setting an experiment is on its way to, if it is)
        let mut s = match self.acts.first() {
            Some(Act::Aim(to)) => *to,
            _ => self.glide.to,
        };
        if self.small {
            (s.e, s.w) = (1.0, 1.0);
        }
        // (and where B will have turned to, if an experiment is still to turn it)
        let mut settings = vec![s];
        for act in &self.acts {
            if let Act::TurnB { from, by, .. } = act {
                let r = turn(*by);
                settings.push(Setting { b: r * from.unwrap_or(s.b) * r.transpose(), ..s });
            }
        }
        let mut lo = DVec2::splat(-0.2);
        let mut hi = DVec2::splat(1.2);
        for s in settings {
            let (m1, m2) = s.maps();
            for i in 0..=48 {
                let p = i as f64 / 24.0;
                for m in [at(m2, m1, p), at(m1, m2, p)] {
                    for c in SQUARE {
                        let q = m * c;
                        if q.is_finite() {
                            (lo, hi) = (lo.min(q), hi.max(q));
                        }
                    }
                }
            }
        }
        (lo.as_vec2(), hi.as_vec2())
    }

    /// The canonical view: the film's frame in the middle of the part of the canvas
    /// the picture has to itself; in space, both books.
    fn the_view(&self) -> View {
        let lay = Layout::of(self.canvas, self.inset, self.shows(), None);
        let (min, max) = lay.frame_room();
        let room = (max - min - NAME_ROOM).max(Vec2::splat(60.0));
        if self.space {
            let (right, up) = screen_axes();
            // the two books and their axes, as wide and tall as they are seen
            let size = Vec2::new(2.0 * (BOOKS_APART + 0.92 * AXIS_LEN) + 0.3, 2.0 * AXIS_LEN + 0.6);
            let per_unit = (room.x / size.x).min(room.y / size.y).max(1.0);
            let dist = (self.canvas.y * 0.5 / per_unit / TAN_HALF_FOV).clamp(2.0, 80.0);
            let per_unit = self.canvas.y * 0.5 / (dist * TAN_HALF_FOV);
            let shift = ((min + max) * 0.5 - self.canvas * 0.5) / per_unit;
            return View::Orbit { target: -right * shift.x + up * shift.y, yaw: YAW, pitch: PITCH, dist };
        }
        let (lo, hi) = self.bounds();
        let size = (hi - lo).max(Vec2::splat(2.0));
        let per_unit = (room.x / size.x).min(room.y / size.y).min(MOST_PER_UNIT).max(1e-3);
        // (screen y runs down, the scene's up)
        let off = ((min + max) * 0.5 - self.canvas * 0.5) / per_unit;
        let centre = (lo + hi) * 0.5;
        View::Fit { center: Vec2::new(centre.x - off.x, centre.y + off.y), half: self.canvas.max(Vec2::ONE) * (0.5 / per_unit) }
    }

    /// What the film is doing, in words.
    fn stage(&self) -> String {
        let p = self.shown().p;
        let (a, b) = self.names();
        if self.space {
            return if p < 1e-4 {
                "before either turn".into()
            } else if p < 1.0 - 1e-4 {
                "1st turn: yellow about $x$, white about $y$".into()
            } else if p <= 1.0 + 1e-4 {
                "after the 1st turn".into()
            } else if p < 2.0 - 1e-4 {
                "2nd turn: yellow about $y$, white about $x$".into()
            } else {
                "after both turns".into()
            };
        }
        if p < 1e-4 {
            "before either map".into()
        } else if p < 1.0 - 1e-4 {
            format!("step 1: ${b}$ acts on the yellow, ${a}$ on the white")
        } else if p <= 1.0 + 1e-4 {
            "after the first step".into()
        } else if p < 2.0 - 1e-4 {
            format!("step 2: ${a}$ acts on the yellow, ${b}$ on the white")
        } else {
            "after both steps".into()
        }
    }

    /// The heading's first line: whether the two orders agree.
    fn verdict(&self) -> String {
        if self.space {
            let (r1, r2) = space_ends(self.shown().theta);
            return if norm3(r1 - r2) <= COMMUTE_TOL * 3.0 {
                r"$R_y R_x = R_x R_y$: the two orders end the same".into()
            } else {
                format!(r"$R_y R_x \ne R_x R_y$: the ends are {} apart", deg(apart(r1, r2)))
            };
        }
        let s = self.shown();
        let (a, b) = self.names();
        // (PQ − QP is e² times AB − BA, and so is 0 exactly when that is)
        if commute(s.a, s.b) {
            format!("${a}{b} = {b}{a}$: the order does not matter")
        } else {
            format!(r"${a}{b} \ne {b}{a}$: the order matters")
        }
    }

    /// The heading's second line: why.
    fn reason(&self) -> String {
        if self.space {
            return "each book is turned about the axes of space, not its own".into();
        }
        let s = self.shown();
        // (with the small maps all there: on the way to them, P and Q are neither)
        if s.w == 1.0 {
            let c = norm(bracket(s.a, s.b));
            return if commute(s.a, s.b) {
                "$AB = BA$, so $PQ = QP$ for every $e$: $PQ - QP = e^2(AB - BA)$".into()
            } else {
                format!(r"$PQ - QP = e^2(AB - BA)$: the gap is $e^2 = {}$ times $\|AB - BA\| = {}$", figures(s.e * s.e), amount(c))
            };
        }
        let (p, q) = self.names();
        let (m1, m2) = self.maps();
        why_given(m1, m2, commute(s.a, s.b)).says(p, q)
    }
}

/// The four numbers of a 2×2 matrix to drag or type, within ±`ENTRY_MAX`. Returns
/// (changed, typed): a number typed in glides the picture to it, one dragged moves it
/// at once.
fn matrix_edit(ui: &mut egui::Ui, id: &str, m: &mut DMat2) -> (bool, bool) {
    let (mut changed, mut typed) = (false, false);
    let mut a = [[m.x_axis.x, m.y_axis.x], [m.x_axis.y, m.y_axis.y]];
    egui::Grid::new(id).spacing([6.0, 4.0]).show(ui, |ui| {
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

/// The eigenvalues in a few symbols: "$2.00$ and $0.50$", "$\pm 1.00i$".
fn eigen_text(e: Eigen) -> String {
    match e {
        Eigen::Real { l1, lines: Lines::One(_) | Lines::Every, .. } => format!("${}$ twice", num(l1, 2)),
        Eigen::Real { l1, l2, .. } => format!("${}$ and ${}$", num(l1, 2), num(l2, 2)),
        Eigen::Complex { re, im } if re.abs() < 5e-3 => format!(r"$\pm {}i$", num(im, 2)),
        Eigen::Complex { re, im } => format!(r"${} \pm {}i$", num(re, 2), num(im, 2)),
    }
}

impl Scene for Commute {
    fn view(&self) -> View {
        self.the_view()
    }

    fn take_view(&mut self) -> Option<View> {
        std::mem::take(&mut self.new_view).then(|| self.the_view())
    }

    fn settings(&self, s: &mut ViewSettings) {
        s.grid_alpha = 0.09;
        // (the plane's axes are drawn by the scene, and in space each book has its own)
        s.axis_alpha = 0.0;
    }

    fn animated(&self) -> bool {
        !self.glide.arrived() || !self.acts.is_empty() || self.panel_h.is_some_and(|h| h != panel_height(self.shows(), 12.0, true))
    }

    fn ui(&mut self, ui: &mut egui::Ui, _: &Gfx) {
        ui::note(ui, r"Doing $B$ and then $A$ is the map $AB$ (the one on the right acts first); doing $A$ and then $B$ is $BA$. Usually the two differ, and the difference $AB - BA$ is a matrix of its own, the commutator. $A$ and $B$ commute when it is $0$. The picture plays both orders on the same letter (yellow: $B$, then $A$; white: $A$, then $B$) and marks the gap between the two ends in red. When $A$ has two eigenlines (orange and purple), they commute exactly when $B$ keeps both, that is, when they are $B$'s eigenlines (blue) too.");

        ui::section(ui, "Things to try");
        for (k, trial) in TRIALS.iter().enumerate() {
            if ui::experiment(ui, trial.label, trial.look_for) {
                self.set_up(k);
            }
        }

        ui::section(ui, "Where");
        let mut place = self.space as usize;
        if ui::choice(ui, &["in the plane", "in space"], &mut place) {
            self.space = place == 1;
            self.by_hand();
            self.replay();
            self.new_view = true;
        }

        if self.space {
            ui::section(ui, "The two turns");
            let mut theta = self.glide.to.theta as f32;
            if ui::slider(ui, r"angle $\theta$ (degrees)", &mut theta, 0.0..=180.0) {
                self.by_hand();
                self.glide.now(|s| s.theta = theta as f64);
            }
            if ui.button("▶ play both orders").clicked() {
                self.replay();
            }
            ui::note(ui, r"Both books are turned by $\theta$ about the $x$ axis and by $\theta$ about the $y$ axis of space: the yellow one about $x$ first, the white one about $y$ first.");
        } else {
            ui::section(ui, "The two orders");
            if ui.button("▶ play both orders").clicked() {
                self.replay();
            }
            let mut p = self.glide.to.p as f32;
            if ui::slider(ui, "progress", &mut p, 0.0..=2.0) {
                self.by_hand();
                self.acts.clear();
                self.glide.now(|s| s.p = p as f64);
            }
            ui::note(ui, "0 is the letter as it was, 1 after the first map of each order, 2 after both.");

            for name in ["A", "B"] {
                ui::section(ui, &format!("Matrix ${name}$"));
                let to = self.glide.to;
                if name == "A" {
                    let labels: Vec<&str> = A_PRESETS.iter().map(|p| p.0).collect();
                    if let Some(i) = ui::buttons(ui, &labels) {
                        self.aim(rows(A_PRESETS[i].1), to.b);
                    }
                } else {
                    let mut hit = None;
                    ui.horizontal_wrapped(|ui| {
                        for (i, (label, choice)) in B_PRESETS.iter().enumerate() {
                            let button = egui::Button::new(ui::label_text(ui, label, 12.5, None, false));
                            if ui.add_enabled(b_choice(*choice, to.a).is_some(), button).clicked() {
                                hit = Some(i);
                            }
                        }
                    });
                    if let Some(b) = hit.and_then(|i| b_choice(B_PRESETS[i].1, to.a)) {
                        self.aim(to.a, b);
                    }
                }
                let mut m = if name == "A" { to.a } else { to.b };
                let (changed, typed) = ui.push_id(name, |ui| matrix_edit(ui, name, &mut m)).inner;
                if changed {
                    let (a, b) = if name == "A" { (m, to.b) } else { (to.a, m) };
                    if typed {
                        self.aim(a, b);
                    } else {
                        self.by_hand();
                        self.glide.now(|s| (s.a, s.b) = (a, b));
                    }
                }
            }
            ui::note(ui, r"The last four choices of $B$ are made from $A$: its inverse, its square, a polynomial in it and a multiple of $I$. Each of them commutes with $A$, whatever $A$ is. Drag the orange dot on $A$'s eigenline, or the blue one on $B$'s, to turn all of that matrix: its lines turn with it, its eigenvalues stay. Near the other's line it is put exactly on it.");

            ui::section(ui, "Small maps");
            let mut small = self.small;
            if ui::checkbox(ui, &mut small, r"use $P = I + eA$ and $Q = I + eB$").changed() {
                self.set_small(small);
            }
            let mut e = self.glide.to.e as f32;
            if ui::slider_log(ui, "$e$", &mut e, E_MIN as f32..=1.0) {
                self.set_small(true);
                self.glide.now(|s| s.e = e as f64);
            }
            if ui.add(egui::Button::new(ui::label_text(ui, "halve $e$", 12.5, None, false))).clicked() {
                self.set_small(true);
                self.glide.to.e = (self.glide.to.e * 0.5).max(E_MIN);
            }
            ui::note(ui, r"For small $e$, $P$ and $Q$ are near the identity. Multiplied out, $PQ - QP = e^2(AB - BA)$, so the gap falls as $e^2$; the chart on the picture shows it on logarithmic scales.");

            ui::section(ui, "Show");
            ui::checkbox(ui, &mut self.eigenlines, "the eigenlines");
        }

        ui::section(ui, "Reading the picture");
        if self.space {
            ui::key(
                ui,
                &[
                    (pal::YELLOW, r"yellow book: turned about $x$ and then about $y$, the map $R_y R_x$"),
                    (pal::WHITE, r"white book: turned about $y$ and then about $x$, the map $R_x R_y$"),
                    (pal::DIM, "grey lines: the axes of space $x$, $y$, $z$, through each book; the one a book is being turned about is in its colour; faint dashed: each book before any turn"),
                    (pal::RED, r"red: the panel's $R_y R_x - R_x R_y$, how far the two ends are apart"),
                ],
            );
        } else {
            let (a, b) = self.names();
            let yellow = format!("yellow letter and square: ${b}$, then ${a}$, ending at ${a}{b}$; faint: after the first step");
            let white = format!("white letter and square: ${a}$, then ${b}$, ending at ${b}{a}$; faint: after the first step");
            ui::key(
                ui,
                &[
                    (pal::YELLOW, &yellow),
                    (pal::WHITE, &white),
                    (pal::DIM, "faint grey: the letter before either map"),
                    (pal::RED, &format!("red arrows: from where ${b}{a}$ sends $(1, 0)$ and $(0, 1)$ to where ${a}{b}$ does, the columns of ${a}{b} - {b}{a}$")),
                    (pal::ORANGE, r"orange and purple: $A$'s eigenlines, with its eigenvalues $\lambda$; the orange dot turns $A$"),
                    (pal::BLUE, r"blue, dashed: $B$'s eigenlines, with its eigenvalues $\mu$; the blue dot turns $B$"),
                ],
            );
        }

        ui::section(ui, "Words");
        ui::term(ui, "Commute:", r"$A$ and $B$ commute if $AB = BA$: $B$ and then $A$ sends every point where $A$ and then $B$ does.");
        ui::term(ui, "Commutator:", r"$AB - BA$, often written $[A, B]$. It sends $x$ to $ABx - BAx$, the gap between where the two orders take $x$. Its trace is always $0$, since $\tr AB = \tr BA$: so it is never $I$.");
        ui::term(ui, r"Size $\|M\|$:", "the square root of the sum of the squares of the entries of $M$.");
        ui::term(ui, "Eigenline:", r"a line through $0$ that a matrix keeps: each point $v$ on it goes to $\lambda v$ ($v$ is an eigenvector, $\lambda$ its eigenvalue).");
        ui::term(ui, "The eigenline test:", r"if $A$ has two different real eigenvalues, $B$ commutes with $A$ exactly when $B$ keeps both of $A$'s eigenlines. (If $AB = BA$ and $Av = \lambda v$, then $A(Bv) = BAv = \lambda Bv$: $Bv$ is $0$ or an eigenvector for $\lambda$ too, on the same line.) In a basis along those lines both are then diagonal.");
        ui::term(ui, "Trace:", r"$\tr M$, the sum of the diagonal entries.");
        ui::term(ui, "Polynomial in $A$:", r"a sum of multiples of powers of $A$, such as $A^2 - 3A$. Each commutes with $A$, as does $A^{-1}$ when $A$ has an inverse. For a $2 \times 2$ matrix $A$ that is not a multiple of $I$, the matrices that commute with it are exactly $aI + bA$.");

        ui::section(ui, "Numbers");
        let s = self.shown();
        if self.space {
            let (r1, r2) = space_ends(s.theta);
            ui::readout(ui, r"$\theta$", deg(s.theta));
            ui::readout_colored(ui, r"$\|R_y R_x - R_x R_y\|$", format!("${}$", amount(norm3(r1 - r2))), pal::RED);
            ui::readout(ui, "the ends apart by", deg(apart(r1, r2)));
        } else {
            let (m1, m2) = self.maps();
            let (a, b) = self.names();
            let c = bracket(m1, m2);
            ui::readout_colored(ui, &format!(r"$\|{a}{b} - {b}{a}\|$"), format!("${}$", amount(norm(c))), pal::RED);
            ui::readout(ui, &format!(r"$\tr({a}{b} - {b}{a})$"), format!("${}$", num(trace(c), 2)));
            ui::readout_colored(ui, "eigenvalues of $A$", eigen_text(eigen(s.a)), pal::ORANGE);
            ui::readout_colored(ui, "eigenvalues of $B$", eigen_text(eigen(s.b)), pal::BLUE);
            if self.small_shown() {
                ui::readout(ui, "$e$", format!("${}$", num(s.e, 3)));
                ui::readout(ui, r"$e^2 \|AB - BA\|$", format!("${}$", amount(s.e * s.e * norm(bracket(s.a, s.b)))));
            }
        }
        ui::see_also(ui, &[("compose", "one map after another, and why the order matters"), ("eigen", "finding eigenvalues and eigenlines"), ("rotations", "turns of space, and three angles"), ("expsum", r"$e^{A + B} = e^A e^B$ when $A$ and $B$ commute")]);
    }

    fn frame(&mut self, f: &mut Frame) {
        // the panel's place follows the canvas, and the picture's place follows the panel's
        let view = f.view_size();
        let shows = self.shows();
        // (the phone's panel glides to its new height when the chart comes or goes)
        let full = panel_height(shows, 12.0, true);
        let k = 1.0 - (-f.real_dt.max(0.0) / 0.1).exp();
        let h = self.panel_h.map_or(full, |h| h + (full - h) * k);
        self.panel_h = Some(if (h - full).abs() < 0.3 || !h.is_finite() { full } else { h });
        let lay = Layout::of(view, f.top_inset, shows, self.panel_h);
        let before = Layout::of(self.canvas, self.inset, shows, None);
        // (a canvas squeezed to a sliver is one the controls have opened over, on a
        // phone: the picture is not seen then, and is left as it was for when they close)
        let covered = view.x < 160.0;
        let changed = !covered && (lay.narrow != before.narrow || lay.panel.is_some() != before.panel.is_some() || (view - self.canvas).abs().max_element() > 24.0);
        let first = !covered && !self.measured && (view - self.canvas).abs().max_element() > 24.0;
        if !covered && (changed || !self.measured) {
            self.new_view |= changed || first;
            (self.canvas, self.measured) = (view, true);
        }
        if !covered {
            self.new_view |= (f.top_inset - self.inset).abs() > 1.0;
            self.inset = f.top_inset;
        }

        let pointer = f.ptr.pos.map(|p| p / f.px);
        let mut replay = false;
        if let Some(panel) = lay.panel.filter(|_| !covered) {
            if pointer.is_some_and(|p| inside_rect(p, panel)) {
                // the panel is for its own button: no moving the view from there, and
                // two quick clicks are two clicks
                f.handles.capture();
                f.handles.claim_clicks();
            }
            let (min, max) = panel;
            let at = (Vec2::new(max.x - PAD - 66.0, min.y + PAD - 2.0), Vec2::new(max.x - PAD, min.y + PAD + 22.0));
            replay = plot::button(f, at, "▶ play", 12.5, false);
        }
        if replay {
            self.replay();
        }
        self.advance(f.dt as f64, f.real_dt as f64);

        let labels_before = f.dl.labels.len();
        if self.space {
            self.space_picture(f, &lay);
        } else {
            self.plane_picture(f, &lay);
        }
        if covered {
            // (the sliver left beside the controls is the picture alone)
            f.dl.labels.truncate(labels_before);
            return;
        }
        if let Some(p) = lay.panel {
            // nothing written into the picture may show through the panel
            let (lo, hi) = (p.0 - Vec2::splat(2.0), p.1 + Vec2::splat(2.0));
            let mut k = 0;
            f.dl.labels.retain(|l| {
                k += 1;
                k <= labels_before || l.backed || !inside_rect(l.pos, (lo, hi))
            });
            if self.space {
                self.space_panel(f, &lay, p);
            } else {
                self.plane_panel(f, &lay, p);
            }
        }
    }
}

impl Commute {
    /// The heading at the top of the picture, the verdict and the reason, broken into
    /// lines to fit. Returns the box it takes.
    fn heading(&mut self, f: &mut Frame, lay: &Layout) -> (Vec2, Vec2) {
        let at = lay.picture.0 + Vec2::new(16.0, 14.0);
        let room = (lay.picture.1.x - lay.picture.0.x - 32.0).max(60.0);
        let (first, second) = (self.verdict(), self.reason());
        let (big, small) = if lay.narrow { (14.0, 11.5) } else { (15.0, 12.5) };
        let of = (first.clone(), second.clone(), room);
        if self.heading.as_ref().is_none_or(|h| h.of != of) {
            let one = broken(f, &first, room, big);
            let two = broken(f, &second, room, small);
            let widest = one.iter().map(|l| f.text_size(l, big).x).chain(two.iter().map(|l| f.text_size(l, small).x)).fold(0.0, f32::max);
            self.heading = Some(Heading { of, first: (one, big), second: (two, small), widest });
        }
        let Some(h) = &self.heading else { return (at, at) };
        let mut y = at.y;
        for line in &h.first.0 {
            f.caption_px(Vec2::new(at.x, y), line.as_str(), h.first.1, pal::TEXT, Align2::LEFT_TOP);
            y += h.first.1 + 7.0;
        }
        y += 2.0;
        for line in &h.second.0 {
            f.caption_px(Vec2::new(at.x, y), line.as_str(), h.second.1, pal::DIM, Align2::LEFT_TOP);
            y += h.second.1 + 6.0;
        }
        (at - Vec2::splat(6.0), Vec2::new(at.x + h.widest.min(room) + 8.0, y + 2.0))
    }

    /// A caption along the bottom of the picture (on a wide canvas), naming what has no
    /// name of its own there. Returns the box it takes.
    fn caption(&mut self, f: &mut Frame, lay: &Layout, parts: &[&'static str]) -> Option<(Vec2, Vec2)> {
        if lay.narrow || lay.panel.is_none() || f.view_size().y < 300.0 {
            return None;
        }
        let size = 12.0;
        let (left, right, bottom) = (lay.picture.0.x, lay.picture.1.x, lay.picture.1.y - 16.0);
        let room = right - left - 36.0;
        let of = (parts.to_vec(), room);
        if self.caption.as_ref().is_none_or(|c| c.of != of) {
            let (lines, widest) = caption_lines(f, parts, room, size);
            self.caption = Some(Caption { of, lines, widest });
        }
        let c = self.caption.as_ref()?;
        let count = c.lines.len();
        for (k, line) in c.lines.iter().enumerate() {
            let up = (count - 1 - k) as f32 * (size + 7.0);
            f.caption_px(Vec2::new(left + 18.0, bottom - up), line.as_str(), size, pal::DIM, Align2::LEFT_BOTTOM);
        }
        Some((Vec2::new(left + 10.0, bottom - count as f32 * (size + 7.0) - 4.0), Vec2::new(left + 26.0 + c.widest, bottom + 6.0)))
    }

    /// The dots on A's first eigenline (orange) and on B's (blue): dragging one turns
    /// all of its matrix, its eigenlines with it, its eigenvalues as they were. (First,
    /// so that everything is drawn where it now is.) Returns where the dots are drawn
    /// on the screen, for the names to keep off.
    fn handles(&mut self, f: &mut Frame, lay: &Layout) -> Vec<Vec2> {
        if !f.handles.dragging() {
            self.turning = None;
        }
        let s = self.shown();
        let moving = s.a != self.glide.to.a || s.b != self.glide.to.b;
        let mut drawn = Vec::new();
        if !self.eigenlines || (moving && !f.handles.dragging()) {
            return drawn;
        }
        let (lo, hi) = f.cam.flat_bounds();
        let wpp = (hi.y - lo.y) / f.view_size().y.max(1.0);
        for (k, reach_px, colour) in [(0usize, 105.0f32, pal::ORANGE), (1, 150.0, pal::BLUE)] {
            let m = if k == 0 { s.a } else { s.b };
            let v = match eigen(m) {
                Eigen::Real { lines: Lines::Two(v, _) | Lines::One(v), .. } => v,
                _ => continue,
            };
            // (the end of the line kept for it)
            let v = if self.last_lines[2 * k].is_some_and(|was| was.dot(v) < 0.0) { -v } else { v };
            let mut dot = (v * (reach_px * wpp) as f64).as_vec2();
            let hidden = lay.panel.is_some_and(|r| on_screen(f, dot.extend(0.0)).is_some_and(|q| inside_rect(q, r)));
            if hidden && !f.handles.dragging() {
                continue;
            }
            let moved = f.drag2(1 + k as u32, &mut dot, colour);
            drawn.extend(on_screen(f, dot.extend(0.0)));
            if moved && dot.length() > 1e-6 {
                let angle = (dot.y as f64).atan2(dot.x as f64);
                let (_, from, start) = *self.turning.get_or_insert((k, m, v.y.atan2(v.x)));
                let other = if k == 0 { s.b } else { s.a };
                let r = DMat2::from_angle(snapped(angle - start, other, from));
                let turned = r * from * r.transpose();
                self.by_hand();
                self.glide.now(|s| if k == 0 { s.a = turned } else { s.b = turned });
            }
        }
        drawn
    }

    /// The plane: the F carried both ways, the eigenlines, the gap, and their names.
    fn plane_picture(&mut self, f: &mut Frame, lay: &Layout) {
        let dots = self.handles(f, lay);
        let (lo, hi) = f.cam.flat_bounds();
        let wpp = (hi.y - lo.y) / f.view_size().y.max(1.0);
        let s = self.shown();
        let (m1, m2) = self.maps();
        let (n1, n2) = self.names();

        // (the batches of the parts, made in the same order every frame)
        for part in [Part::Eigen, Part::EigenB, Part::Stops, Part::Gap] {
            lines_for(f, part);
        }
        let mut room = Room::new(f);
        keep_ticks_clear(f, &mut room);
        for dot in dots {
            room.boxed(dot - Vec2::splat(9.0), dot + Vec2::splat(9.0));
        }
        if let Some((min, max)) = lay.panel {
            room.wall(min - Vec2::splat(4.0), max + Vec2::splat(4.0));
        }
        let head = self.heading(f, lay);
        room.wall(head.0, head.1);
        let parts: &[&'static str] = &["faint grey: the letter before either map", r"orange, purple: $A$'s eigenlines; dashed blue: $B$'s", r"red: the gap between the two ends at $(1, 0)$ and $(0, 1)$"];
        if let Some(plate) = self.caption(f, lay, parts) {
            room.wall(plate.0, plate.1);
        }

        // the axes of the plane
        for (a, b) in [(Vec2::new(lo.x, 0.0), Vec2::new(hi.x, 0.0)), (Vec2::new(0.0, lo.y), Vec2::new(0.0, hi.y))] {
            f.line(a, b, 1.2, pal::DIM.a(0.45));
        }

        // the eigenlines: A's as wide pale bands, B's dashed on top, so that a line
        // they share shows as both. Those of the matrices as they are; or, while they
        // glide to new ones, those of the old fading out and of the new fading in.
        let to = (self.glide.to.a, self.glide.to.b);
        if (s.a, s.b) == to {
            self.lines_from = None;
        }
        let share = self.lines_from.map(|(fa, fb)| {
            let total = size(to.0 - fa).max(size(to.1 - fb));
            let left = size(to.0 - s.a).max(size(to.1 - s.b));
            if total > 0.0 {
                (1.0 - left / total).clamp(0.0, 1.0)
            } else {
                1.0
            }
        });
        let (sets, strength) = match (self.lines_from, share) {
            (Some(from), Some(u)) if u < 1.0 => ([from, to], [smooth(1.0 - 2.0 * u) as f32, smooth(2.0 * u - 1.0) as f32]),
            _ => ([(s.a, s.b), (s.a, s.b)], [1.0, 0.0]),
        };
        // (the set named, and whose lines keep their ends: the one more there)
        let named = (strength[1] > strength[0]) as usize;
        let lines_of = |m: DMat2, l: [Color; 2]| -> Vec<(DVec2, f64, Color)> {
            match eigen(m) {
                Eigen::Real { l1, l2, lines: Lines::Two(v1, v2) } => vec![(v1, l1, l[0]), (v2, l2, l[1])],
                Eigen::Real { l1, lines: Lines::One(v), .. } => vec![(v, l1, l[0])],
                _ => Vec::new(),
            }
        };
        let (box_lo, box_hi) = (lo - (hi - lo), hi + (hi - lo));
        let mut lines: [Vec<(DVec2, f64, Color)>; 2] = [Vec::new(), Vec::new()];
        for (set, (a, b)) in sets.iter().enumerate() {
            if strength[set] <= 0.0 {
                continue;
            }
            let mut these = [lines_of(*a, [pal::ORANGE, pal::PURPLE]), lines_of(*b, [pal::BLUE, pal::BLUE])];
            // (a line's end, for its name and the blue dot, stays the end it was as
            // the matrix changes)
            for (k, list) in these.iter_mut().enumerate() {
                for (i, (v, _, _)) in list.iter_mut().enumerate() {
                    let was = self.last_lines[2 * k + i].unwrap_or(DVec2::new(1.0, 0.3));
                    if was.dot(*v) < 0.0 {
                        *v = -*v;
                    }
                    if set == named {
                        self.last_lines[2 * k + i] = Some(*v);
                    }
                }
                if set == named {
                    for i in list.len()..2 {
                        self.last_lines[2 * k + i] = None;
                    }
                }
            }
            if self.eigenlines {
                for (k, list) in these.iter().enumerate() {
                    for &(v, _, c) in list {
                        if let Some((p, q)) = line_in(v, box_lo, box_hi) {
                            // (names keep off A's lines; B's, thin dashes, may pass behind
                            // a name: as B is turned, its line would sweep a name before it
                            // from place to place)
                            if k == 0 {
                                lines_for(f, Part::Eigen).segment(p.extend(0.0), q.extend(0.0), 3.5, c.a(0.22 * strength[set]));
                                room.line(f, p.extend(0.0), q.extend(0.0));
                            } else {
                                lines_for(f, Part::EigenB).segment(p.extend(0.0), q.extend(0.0), 1.8, c.a(0.95 * strength[set]));
                            }
                        }
                    }
                }
            }
            if set == named {
                lines = these;
            }
        }
        let named_strength = strength[named];

        // the F before either map
        let outline = |m: DMat2| -> Vec<Vec3> { F_OUTLINE.iter().map(|p| (m * *p).as_vec2().extend(0.0)).collect() };
        let square = |m: DMat2| -> [Vec3; 4] { SQUARE.map(|p| (m * p).as_vec2().extend(0.0)) };
        f.dl.lines.closed(&outline(DMat2::IDENTITY), 1.1, pal::DIM.a(0.5));
        f.dl.lines.closed(&square(DMat2::IDENTITY), 1.0, pal::DIM.a(0.35));

        // where each order is after its first step, once it has been taken
        let p = s.p;
        // (and faint once both steps are taken: then the two ends are what matter)
        let stop = smooth((p - 1.0) / 0.25) as f32;
        let faded = stop * (1.0 - 0.8 * smooth((p - 1.75) / 0.25) as f32);
        if stop > 0.0 {
            for (m, c) in [(m2, pal::YELLOW), (m1, pal::WHITE)] {
                lines_for(f, Part::Stops).closed(&outline(m), 1.0, c.a(0.45 * faded));
                lines_for(f, Part::Stops).closed(&square(m), 1.0, c.a(0.3 * faded));
            }
        }

        // the two orders as they are now: yellow, B first; white, A first
        let (yellow, white) = (at(m2, m1, p), at(m1, m2, p));
        let fill = |f: &mut Frame, m: DMat2, square_alpha: f32, bars_alpha: f32, c: Color| {
            let q = square(m);
            f.dl.tris_alpha.quad(q[0], q[1], q[2], q[3], c.a(square_alpha));
            for bar in F_BARS {
                let [p0, p1, p2, p3] = bar.map(|p| (m * p).as_vec2().extend(0.0));
                f.dl.tris_alpha.quad(p0, p1, p2, p3, c.a(bars_alpha));
            }
        };
        fill(f, yellow, 0.12, 0.5, pal::YELLOW);
        fill(f, white, 0.0, 0.16, pal::WHITE);
        f.dl.lines.closed(&square(yellow), 1.6, pal::YELLOW.a(0.9));
        f.dl.lines.closed(&outline(yellow), 1.4, pal::YELLOW);
        f.dl.lines.closed(&square(white), 1.4, pal::WHITE.a(0.8));
        f.dl.lines.closed(&outline(white), 1.2, pal::WHITE.a(0.9));
        for m in [yellow, white] {
            room.shape(f, &square(m));
            let q = square(m);
            for k in 0..4 {
                room.line(f, q[k], q[(k + 1) % 4]);
            }
        }

        // the gap at the end: from where BA sends (1, 0) and (0, 1) to where AB does
        let gap = smooth((p - 1.75) / 0.25) as f32;
        let together = commute(s.a, s.b);
        let mut red_at = None;
        if gap > 0.0 && !together {
            for e in [DVec2::X, DVec2::Y] {
                let (from, to) = ((white * e).as_vec2(), (yellow * e).as_vec2());
                arrow_in(f, Part::Gap, from, to, 2.4, pal::RED.a(gap));
                room.line(f, from.extend(0.0), to.extend(0.0));
                if red_at.is_none() && (to - from).length() > 18.0 * wpp {
                    red_at = Some((from + to) * 0.5);
                }
            }
        }

        // the names, most important first
        let away = |p: Vec2| Vec2::new(p.x, -p.y);
        let seen = |f: &Frame, p: Vec2| p.is_finite() && on_screen(f, p.extend(0.0)).is_some_and(|q| inside_rect(q, (Vec2::ZERO, f.view_size())));
        let corner = |m: DMat2| (m * DVec2::ONE).as_vec2();
        let ends_meet = together && p >= 2.0 - 1e-6;
        let y_text = if ends_meet { format!("${n1}{n2} = {n2}{n1}$") } else { format!("${n1}{n2}$: ${n2}$, then ${n1}$") };
        let (yc, wc) = (corner(yellow), corner(white));
        if seen(f, yc) {
            room.label(f, "yellow", yc.extend(0.0), &y_text, 14.0, pal::YELLOW, away(yc - wc * 0.3), 8.0, false);
        }
        // (the white end named too, unless it is so near the yellow that the two names
        // would be written over each other: as two orders that commute come together)
        let near = on_screen(f, yc.extend(0.0)).zip(on_screen(f, wc.extend(0.0))).is_some_and(|(p, q)| p.distance(q) < 40.0);
        if !ends_meet && !near && seen(f, wc) {
            room.label(f, "white", wc.extend(0.0), &format!("${n2}{n1}$: ${n1}$, then ${n2}$"), 14.0, pal::WHITE, away(wc - yc * 0.3), 8.0, false);
        }
        if let Some(mid) = red_at.filter(|m| seen(f, *m)) {
            room.label(f, "gap", mid.extend(0.0), &format!("${n1}{n2} - {n2}{n1}$"), 13.0, pal::RED, Vec2::new(1.0, 1.0), 8.0, true);
        } else if gap > 0.5 && !together {
            // (a gap too small to see is still there: its size, where it would be)
            let at = (yellow * DVec2::X).as_vec2();
            let text = format!(r"gap $\|{n1}{n2} - {n2}{n1}\| = {}$", amount(norm(bracket(m1, m2))));
            room.label(f, "gap", at.extend(0.0), &text, 12.5, pal::RED, Vec2::new(1.0, 0.6), 8.0, false);
        }
        if stop > 0.5 && p < 1.9 {
            for (name, m, c, text) in [("after first", m2, pal::YELLOW, format!("after ${n2}$")), ("after second", m1, pal::WHITE, format!("after ${n1}$"))] {
                let q = corner(m);
                if seen(f, q) && (q - yc).length() > 30.0 * wpp && (q - wc).length() > 30.0 * wpp {
                    room.label(f, name, q.extend(0.0), &text, 12.0, c.mix(pal::DIM, 0.35), away(q), 6.0, true);
                }
            }
        }
        if p > 0.3 {
            room.label(f, "start", Vec3::new(0.5, 0.06, 0.0), "start", 11.0, pal::DIM, Vec2::new(0.0, 1.0), 4.0, true);
        }
        if self.eigenlines {
            let (min, max) = lay.frame_room();
            // each line named near where it leaves the picture, at the end kept for it
            // (one place for each, whose side is kept from frame to frame: a name that
            // looked for room along its line hopped from place to place as the letters
            // passed it)
            let end_of = |f: &Frame, v: DVec2| -> Option<(Vec2, Vec2, Vec2)> {
                let (o, tip) = (on_screen(f, Vec3::ZERO)?, on_screen(f, v.as_vec2().extend(0.0))?);
                let d = (tip - o).normalize_or_zero();
                let (_, end) = line_in(d.as_dvec2(), min - o, max - o)?;
                let up = if d.perp().y < 0.0 { d.perp() } else { -d.perp() };
                Some((o + end, d, up))
            };
            let symbols = [[r"\lambda_1", r"\lambda_2"], [r"\mu_1", r"\mu_2"]];
            // (each name: its number, the place it is written from on the screen, the
            // way off its line, its words and its colour)
            let mut names: Vec<(usize, Vec2, Vec2, String, Color)> = Vec::new();
            for (k, set) in lines.iter().enumerate() {
                let single = set.len() == 1;
                for (i, &(v, l, c)) in set.iter().enumerate() {
                    let symbol = if single { &symbols[k][0][..symbols[k][0].len() - 2] } else { symbols[k][i] };
                    let Some((end, d, up)) = end_of(f, v) else { continue };
                    let at = end - d * 30.0;
                    // a line of B's that leaves the picture where one of A's does (the
                    // same line, or nearly) is named in A's name: two names in one
                    // place would be written over each other
                    if k == 1 {
                        if let Some(shared) = names.iter_mut().find(|n| n.0 < 2 && n.1.distance(at) < 26.0) {
                            shared.3 = format!("{}, ${}$", shared.3, tint(pal::BLUE, &format!("{symbol} = {}", num(l, 2))));
                            continue;
                        }
                    }
                    names.push((2 * k + i, at, up, format!("${symbol} = {}$", num(l, 2)), c));
                }
            }
            for (number, at, up, text, c) in names {
                let Some(w) = f.cam.pick_ground(at * f.px) else { continue };
                // (far enough off the line that the whole box clears it, however the
                // line slants)
                let half = f.text_size(&text, 13.0) * 0.5 + Vec2::new(4.0, 0.0);
                let gap = 7.0 + half.x * up.x.abs() * (1.0 - up.x.abs()) + half.y * up.y.abs() * (1.0 - up.y.abs());
                room.label(f, ("eigen", number as u32), w, &text, 13.0, c.a(named_strength), up, gap, false);
            }
        }
        f.axes2();
    }

    /// The panel on the screen for the plane: the matrices, the commutator and its
    /// size, and with small maps the chart of the gap against e.
    fn plane_panel(&self, f: &mut Frame, lay: &Layout, (min, max): (Vec2, Vec2)) {
        hud_panel(f, min, max);
        let s = self.shown();
        let (m1, m2) = self.maps();
        let (n1, n2) = self.names();
        let t = lay.text;
        let x = min.x + PAD;
        let mut y = min.y + PAD;
        f.text_px(Vec2::new(x, y + 10.0), self.stage(), 12.0, pal::TEXT, Align2::LEFT_CENTER);
        y += HEAD_H;

        let mh = matrix_h(2, t);
        let lead = |f: &mut Frame, right: f32, y: f32, text: &str, c: Color| f.text_px(Vec2::new(right, y + mh * 0.5), text, t, c, Align2::RIGHT_CENTER);
        // (the columns the same for A and B as for P and Q, so that nothing moves as the
        // one pair gives way to the other)
        let widest = |a: &str, b: &str| text_width(a, t).max(text_width(b, t));
        let first_col = x + widest("$AB =$", "$PQ =$") + 6.0;
        let mw = matrix_size(2, 2, CELL, t).x;
        let second_lead = first_col + mw + 14.0 + widest("$BA =$", "$QP =$");
        let second_col = second_lead + 6.0;
        let c = bracket(m1, m2);
        let rows: [[(String, DMat2, Color); 2]; 2] = [[(format!("${n1} =$"), m1, pal::TEXT), (format!("${n2} =$"), m2, pal::BLUE)], [(format!("${n1}{n2} =$"), m1 * m2, pal::YELLOW), (format!("${n2}{n1} =$"), m2 * m1, pal::WHITE)]];
        for row in rows {
            for (k, (name, m, colour)) in row.into_iter().enumerate() {
                let (right, col) = if k == 0 { (first_col - 6.0, first_col) } else { (second_lead, second_col) };
                lead(f, right, y, &name, colour);
                hud_matrix(f, Vec2::new(col, y), &entries2(m, colour), CELL, t, pal::DIM);
            }
            y += mh + 10.0;
        }
        // the commutator, its size and its trace
        let name = format!("${n1}{n2} - {n2}{n1} =$");
        let w = widest("$AB - BA =$", "$PQ - QP =$");
        f.text_px(Vec2::new(x, y + mh * 0.5), name.as_str(), t, pal::RED, Align2::LEFT_CENTER);
        let col = x + w + 6.0;
        hud_matrix(f, Vec2::new(col, y), &entries2(c, pal::RED), CELL, t, pal::RED.mix(pal::DIM, 0.5));
        let after = col + mw + 10.0;
        f.text_px(Vec2::new(after, y + mh * 0.5 - 9.0), format!("size ${}$", amount(norm(c))), 12.0, pal::RED, Align2::LEFT_CENTER);
        f.text_px(Vec2::new(after, y + mh * 0.5 + 9.0), format!("trace ${}$", num(trace(c), 2)), 12.0, pal::DIM, Align2::LEFT_CENTER);
        y += mh + 10.0;

        if self.small {
            f.text_px(Vec2::new(x, y), format!(r"$P = I + eA$, $Q = I + eB$, $e = {}$", num(s.e, 3)), 12.0, pal::TEXT, Align2::LEFT_TOP);
            y += LINE_H + 6.0;
            f.text_px(Vec2::new(x, y), r"the gap $\|PQ - QP\|$ against $e$, both scales logarithmic", 11.5, pal::DIM, Align2::LEFT_TOP);
            y += 22.0;
            let rect = (Vec2::new(x + 54.0, y), Vec2::new(max.x - PAD - 6.0, y + CHART_H));
            self.chart(f, rect, s);
        }
        if lay.narrow {
            f.text_px(Vec2::new(x, max.y - 8.0), r"$\lambda$: $A$'s eigenlines   $\mu$: $B$'s, dashed   red: the gap", 11.0, pal::DIM, Align2::LEFT_BOTTOM);
        }
    }

    /// The gap ‖PQ − QP‖ = e² ‖AB − BA‖ against e, both on logarithmic scales: a
    /// straight line of slope 2, and the dot where e is.
    fn chart(&self, f: &mut Frame, rect: (Vec2, Vec2), s: Setting) {
        let c = norm(bracket(s.a, s.b));
        let commuting = commute(s.a, s.b) || c.is_nan() || c <= 0.0;
        // (the gap from e = 0.01 to 1 spans four powers of ten, wherever they are: the
        // range follows ‖AB − BA‖ smoothly, so that the line stays put as it changes)
        let lc = if commuting { 0.0 } else { c.log10() };
        let plot = Plot::new(rect, [-2.0, 0.0], [lc - 4.35, lc + 0.35]);
        hud_rect(f, rect.0, rect.1, Color::hex(0x10172A));
        if plot.tiny() {
            return;
        }
        let (lo, hi) = rect;
        // the powers of ten, across and up
        for k in 0..=2 {
            let x = plot.px_x(-(k as f64));
            plot::line_of(f, plot::AXES, Vec2::new(x, hi.y), Vec2::new(x, hi.y + 4.0), 1.0, pal::DIM);
            plot::line_of(f, plot::AXES, Vec2::new(x, lo.y), Vec2::new(x, hi.y), 1.0, pal::DIM.a(0.18));
            let name = ["$1$", "$0.1$", "$0.01$"][k];
            f.text_px(Vec2::new(x, hi.y + 5.0), name, 10.5, pal::DIM, Align2::CENTER_TOP);
        }
        plot::line_of(f, plot::AXES, Vec2::new(lo.x, hi.y), hi, 1.0, pal::DIM.a(0.8));
        plot::line_of(f, plot::AXES, lo, Vec2::new(lo.x, hi.y), 1.0, pal::DIM.a(0.8));
        f.text_px(Vec2::new(hi.x - 3.0, hi.y - 3.0), "$e$", 11.5, pal::TEXT, Align2::RIGHT_BOTTOM);
        if commuting {
            f.text_px((lo + hi) * 0.5, r"$AB = BA$: the gap is $0$ for every $e$", 11.5, pal::TEXT, Align2::CENTER_CENTER);
            return;
        }
        // the gap's own values at e = 1, 0.1 and 0.01, a hundredth of the one before
        // (at places that stay put as the gap changes)
        for k in 0..=4 {
            let y = plot.px_y(lc - k as f64);
            plot::line_of(f, plot::OWN, Vec2::new(lo.x - 4.0, y), Vec2::new(lo.x, y), 1.0, pal::DIM);
            plot::line_of(f, plot::OWN, Vec2::new(lo.x, y), Vec2::new(hi.x, y), 1.0, pal::DIM.a(if k % 2 == 0 { 0.22 } else { 0.1 }));
            if k % 2 == 0 {
                f.text_px(Vec2::new(lo.x - 6.0, y), format!("${}$", figures(c * 10f64.powi(-k))), 10.5, pal::DIM, Align2::RIGHT_CENTER);
            }
        }
        // gap = e² c: log gap = 2 log e + log c
        let line = |x: f64| 2.0 * x + lc;
        plot.path(f, plot::CURVES, [(-2.0, line(-2.0)), (0.0, line(0.0))], 2.0, pal::RED);
        let ex = s.e.max(E_MIN).log10();
        plot.dot(f, ex, line(ex), 9.0, pal::RED);
        // halving e: a quarter of the gap
        let (x0, x1) = (-1.25, -1.25 + 2f64.log10());
        let (p0, p1, p2) = (plot.at(x0, line(x0)), plot.at(x1, line(x0)), plot.at(x1, line(x1)));
        plot::line_of(f, plot::MARKS, p0, p1, 1.0, pal::DIM);
        plot::line_of(f, plot::MARKS, p1, p2, 1.0, pal::DIM);
        f.text_px(p1 + Vec2::new(5.0, 2.0), r"$e \times 2$: gap $\times 4$, slope $2$", 10.5, pal::DIM, Align2::LEFT_TOP);
    }

    /// The panel on the screen in space: the two ends as matrices and their difference.
    fn space_panel(&self, f: &mut Frame, lay: &Layout, (min, max): (Vec2, Vec2)) {
        hud_panel(f, min, max);
        let s = self.shown();
        let t = lay.text;
        let x = min.x + PAD;
        let mut y = min.y + PAD;
        f.text_px(Vec2::new(x, y + 10.0), self.stage(), 12.0, pal::TEXT, Align2::LEFT_CENTER);
        y += HEAD_H;
        let (r1, r2) = space_ends(s.theta);
        let mh = matrix_h(3, t);
        let names = [(r"$R_y R_x =$", r1, pal::YELLOW), (r"$R_x R_y =$", r2, pal::WHITE), (r"$R_y R_x - R_x R_y =$", r1 - r2, pal::RED)];
        let lead_w = text_width(names[2].0, t);
        for (name, m, c) in names {
            f.text_px(Vec2::new(x + lead_w, y + mh * 0.5), name, t, c, Align2::RIGHT_CENTER);
            hud_matrix(f, Vec2::new(x + lead_w + 6.0, y), &entries3(m, c), CELL, t, pal::DIM);
            y += mh + 10.0;
        }
        f.text_px(Vec2::new(x, y), format!(r"$\theta = {}$: size ${}$, the ends {} apart", deg(s.theta).replace('°', r"^\circ"), amount(norm3(r1 - r2)), deg(apart(r1, r2))), 12.0, pal::RED, Align2::LEFT_TOP);
        if lay.narrow {
            f.text_px(Vec2::new(x, max.y - 8.0), r"yellow: $x$, then $y$   white: $y$, then $x$", 11.0, pal::DIM, Align2::LEFT_BOTTOM);
        }
    }

    /// Space: the two books, each with the axes of space through it, turned in their
    /// two orders.
    fn space_picture(&mut self, f: &mut Frame, lay: &Layout) {
        self.heading(f, lay);
        let s = self.shown();
        let (right, up) = screen_axes();
        let p = s.p;
        let step = if p <= 1.0 { 0 } else { 1 };
        let busy = p > 1e-4 && p < 2.0 - 1e-4;
        for (k, (x_first, colour)) in [(true, pal::YELLOW), (false, pal::WHITE)].into_iter().enumerate() {
            let centre = right * if k == 0 { -BOOKS_APART } else { BOOKS_APART };
            // the axis turned about now: x first for the yellow book, y first for the white
            let turning = if busy { Some(if x_first == (step == 0) { 0 } else { 1 }) } else { None };
            for (axis, name) in ["$x$", "$y$", "$z$"].into_iter().enumerate() {
                let d = [Vec3::X, Vec3::Y, Vec3::Z][axis] * AXIS_LEN;
                let on = turning == Some(axis);
                let c = if on { colour } else { pal::DIM.a(0.7) };
                lines_for(f, Part::Space).segment(centre - d, centre + d, if on { 2.6 } else { 1.2 }, c);
                f.text_off(centre + d * 1.08, Vec2::ZERO, name, 13.0, if on { colour } else { pal::DIM }, Align2::CENTER_CENTER);
            }
            let m = book_pose(s.theta, p, x_first).as_mat3();
            book_outline(f, centre, Mat3::IDENTITY, pal::DIM.a(0.45));
            book(f, centre, m, colour);
            let title = if x_first { r"about $x$, then $y$: $R_y R_x$" } else { r"about $y$, then $x$: $R_x R_y$" };
            if let Some(at) = on_screen(f, centre + up * (AXIS_LEN + 0.32)) {
                f.text_px(at, title, 13.5, colour, Align2::CENTER_BOTTOM);
            }
        }
    }
}

/// How far to turn B (radians), asked to turn it by `turn`: exactly onto one of A's
/// eigenlines where that brings one of B's within `SNAP` of it, so that a hand can
/// find the place where they commute. (B turned by a half turn more is the same B.)
fn snapped(turn: f64, a: DMat2, b: DMat2) -> f64 {
    let angles = |m: DMat2| -> Vec<f64> {
        match eigen(m) {
            Eigen::Real { lines: Lines::Two(v1, v2), .. } => vec![v1.y.atan2(v1.x), v2.y.atan2(v2.x)],
            Eigen::Real { lines: Lines::One(v), .. } => vec![v.y.atan2(v.x)],
            _ => Vec::new(),
        }
    };
    let pi = std::f64::consts::PI;
    // the difference of two angles of lines, taken into −90°..90°
    let wrap = |d: f64| (d + 0.5 * pi).rem_euclid(pi) - 0.5 * pi;
    let mut best = (SNAP, turn);
    for a in angles(a) {
        for b in angles(b) {
            let off = wrap(turn - (a - b));
            if off.abs() < best.0 {
                best = (off.abs(), turn - off);
            }
        }
    }
    best.1
}

/// How near (radians) one of B's eigenlines must be turned to one of A's to be put on it.
const SNAP: f64 = 0.03;

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

/// The book's eight corners, turned by `m` and stood at `centre`.
fn book_corners(centre: Vec3, m: Mat3) -> [Vec3; 8] {
    let h = BOOK.as_vec3();
    let mut out = [Vec3::ZERO; 8];
    for (i, c) in out.iter_mut().enumerate() {
        let s = Vec3::new(if i & 1 == 0 { -1.0 } else { 1.0 }, if i & 2 == 0 { -1.0 } else { 1.0 }, if i & 4 == 0 { -1.0 } else { 1.0 });
        *c = centre + m * (h * s);
    }
    out
}

/// The book's six faces, as corners (in `book_corners`' numbering), with their
/// shares of the book's colour: the cover, the back, the spine, the edges of the pages.
const FACES: [([usize; 4], f32); 6] = [([4, 5, 7, 6], 1.0), ([0, 2, 3, 1], 0.55), ([0, 4, 6, 2], 0.75), ([1, 3, 7, 5], 0.3), ([2, 6, 7, 3], 0.3), ([0, 1, 5, 4], 0.3)];
const EDGES: [(usize, usize); 12] = [(0, 1), (2, 3), (4, 5), (6, 7), (0, 2), (1, 3), (4, 6), (5, 7), (0, 4), (1, 5), (2, 6), (3, 7)];

/// The book turned by `m` at `centre`: see-through faces shaded as if lit from above,
/// drawn back to front, its outline, and the F printed on its cover.
fn book(f: &mut Frame, centre: Vec3, m: Mat3, colour: Color) {
    let c = book_corners(centre, m);
    let light = Vec3::new(0.35, -0.3, 0.88).normalize();
    let eye = f.cam.eye;
    let mut faces: Vec<([Vec3; 4], f32)> = FACES.iter().map(|(q, share)| (q.map(|i| c[i]), *share)).collect();
    let far = |q: &[Vec3; 4]| ((q[0] + q[1] + q[2] + q[3]) * 0.25).distance(eye);
    faces.sort_by(|p, q| far(&q.0).total_cmp(&far(&p.0)));
    for ([a, b, cc, d], share) in faces {
        let tone = (0.45 + 0.55 * (b - a).cross(d - a).normalize_or_zero().dot(light).abs()) * (0.4 + 0.6 * share);
        f.dl.tris_alpha.quad(a, b, cc, d, Color::from_linear(colour.to_linear() * tone).a(0.62));
    }
    for (i, j) in EDGES {
        lines_for(f, Part::Space).segment(c[i], c[j], 1.3, colour.a(0.9));
    }
    // the F on the cover, a hair above it, in ink
    let h = BOOK.as_vec3();
    let on_cover = |p: DVec2| centre + m * Vec3::new(-h.x * 0.8 + 1.6 * h.x * p.x as f32, -h.y * 0.8 + 1.6 * h.y * p.y as f32, h.z * 1.02);
    for bar in F_BARS {
        let [p0, p1, p2, p3] = bar.map(on_cover);
        f.dl.tris_alpha.quad(p0, p1, p2, p3, Color::hex(0x1A2236).a(0.92));
    }
}

/// The book's outline alone, dashed: where it starts.
fn book_outline(f: &mut Frame, centre: Vec3, m: Mat3, colour: Color) {
    let c = book_corners(centre, m);
    for (i, j) in EDGES {
        f.dashed(c[i], c[j], 1.0, colour);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const SMALL_TRIAL: usize = 3;

    fn close(m: DMat2, n: DMat2, tol: f64) -> bool {
        (m - n).to_cols_array().iter().all(|v| v.abs() <= tol)
    }

    fn random(rng: &mut Rng, scale: f64) -> DMat2 {
        rows([0; 4].map(|_| rng.range(-1.0, 1.0) as f64 * scale))
    }

    /// The scene as an experiment's button leaves it, run at 60 frames a second until
    /// it has done everything (or for `seconds` at most).
    fn run(k: usize, seconds: f64) -> Commute {
        let mut scene = Commute::create();
        scene.set_up(k);
        for _ in 0..(seconds * 60.0) as usize {
            scene.advance(1.0 / 60.0, 1.0 / 60.0);
            if !scene.animated() {
                break;
            }
        }
        assert!(!scene.animated(), "experiment {k} still going after {seconds} s");
        scene
    }

    /// The arithmetic the scene stands on: the trace of a commutator is 0; small maps'
    /// gap is e² times the commutator, exactly; whatever is made from A commutes with it.
    #[test]
    fn commutators_behave_as_the_panel_says() {
        let mut rng = Rng::new(3);
        for _ in 0..500 {
            let (a, b) = (random(&mut rng, 3.0), random(&mut rng, 3.0));
            let c = bracket(a, b);
            assert!(trace(c).abs() < 1e-12 * norm(a) * norm(b) + 1e-14, "{a:?} {b:?}");
            assert!(!close(c, DMat2::IDENTITY, 0.5), "AB - BA came out as I");
            for e in [1.0, 0.5, 0.1, 0.013] {
                let (p, q) = (DMat2::IDENTITY + a * e, DMat2::IDENTITY + b * e);
                assert!(close(bracket(p, q), c * (e * e), 1e-12 * (1.0 + norm(c))), "{a:?} {b:?} at {e}");
            }
            // a polynomial in A, A's inverse, multiples of I commute with A
            for m in [a * a, a * a - a * 3.0, DMat2::IDENTITY * 3.0, a * a * a + a * 0.5] {
                assert!(commute(a, m), "{a:?} {m:?}");
            }
            if a.determinant().abs() > 0.05 {
                assert!(commute(a, a.inverse()));
            }
        }
    }

    /// The eigenline test: for an A with two different real eigenvalues, B commutes
    /// with A exactly when B keeps both of A's eigenlines; and for any A that is not a
    /// multiple of I, the matrices that commute with it are aI + bA, and no others.
    #[test]
    fn what_commutes_with_a_is_what_the_heading_says() {
        let mut rng = Rng::new(7);
        let mut checked = [0usize; 4];
        for _ in 0..2000 {
            let a = random(&mut rng, 2.0);
            let (x, y) = (rng.range(-2.0, 2.0) as f64, rng.range(-2.0, 2.0) as f64);
            let along = DMat2::IDENTITY * x + a * y;
            let other = random(&mut rng, 2.0);
            for b in [along, other] {
                let w = why(a, b);
                let c = commute(a, b);
                match eigen(a) {
                    Eigen::Real { lines: Lines::Two(v1, v2), .. } => {
                        // B keeps both lines exactly when they commute
                        if !c {
                            assert!(!(keeps(b, v1) && keeps(b, v2)), "{a:?} {b:?}");
                        }
                        assert_eq!(c, matches!(w, Why::Lines(2, _)), "{a:?} {b:?}");
                        if c {
                            assert!(off_line(b, v1) < 1e-9 * size(b).max(1.0) && off_line(b, v2) < 1e-9 * size(b).max(1.0), "{a:?} {b:?}");
                        }
                        checked[c as usize] += 1;
                    }
                    Eigen::Complex { .. } | Eigen::Real { lines: Lines::One(_), .. } => {
                        assert_eq!(c, matches!(w, Why::Span { ab: Some(_), .. }));
                        if let Why::Span { ab: Some((p, q)), .. } = w {
                            assert!(close(DMat2::IDENTITY * p + a * q, b, 1e-9 * size(b).max(1.0)), "{a:?} {b:?}: {p} {q}");
                        }
                        checked[2 + c as usize] += 1;
                    }
                    _ => {}
                }
                // aI + bA always commutes; a random B (almost) never does
                if b == along {
                    assert!(c, "{a:?} {b:?}");
                }
            }
            // the matrices that commute with A form a plane (I and A), not more: the
            // map X -> AX - XA has rank 2
            let basis = [rows([1.0, 0.0, 0.0, 0.0]), rows([0.0, 1.0, 0.0, 0.0]), rows([0.0, 0.0, 1.0, 0.0]), rows([0.0, 0.0, 0.0, 1.0])];
            let images: Vec<[f64; 4]> = basis.iter().map(|x| bracket(a, *x).to_cols_array()).collect();
            let rank = rank4(&images);
            assert_eq!(rank, 2, "{a:?}");
        }
        assert!(checked.iter().all(|n| *n > 100), "{checked:?}");
        // a multiple of I commutes with everything
        for _ in 0..50 {
            let b = random(&mut rng, 3.0);
            assert!(commute(DMat2::IDENTITY * 2.0, b) && why(DMat2::IDENTITY * 2.0, b) == Why::Scalar);
        }
    }

    /// The rank of up to four vectors of four numbers, by elimination.
    fn rank4(v: &[[f64; 4]]) -> usize {
        let mut m: Vec<[f64; 4]> = v.to_vec();
        let mut rank = 0;
        for col in 0..4 {
            let Some(pivot) = (rank..m.len()).max_by(|&i, &j| m[i][col].abs().total_cmp(&m[j][col].abs())) else { break };
            if m[pivot][col].abs() < 1e-9 {
                continue;
            }
            m.swap(rank, pivot);
            for i in 0..m.len() {
                if i != rank {
                    let k = m[i][col] / m[rank][col];
                    for j in 0..4 {
                        m[i][j] -= k * m[rank][j];
                    }
                }
            }
            rank += 1;
        }
        rank
    }

    /// The film starts at the F and ends at the products, the yellow at AB and the
    /// white at BA, and a turn is turned rather than squashed on the way.
    #[test]
    fn the_film_ends_at_the_two_products() {
        let mut rng = Rng::new(11);
        for _ in 0..300 {
            let (a, b) = (random(&mut rng, 2.0), random(&mut rng, 2.0));
            assert!(close(at(b, a, 0.0), DMat2::IDENTITY, 0.0) && close(at(a, b, 0.0), DMat2::IDENTITY, 0.0));
            assert!(close(at(b, a, 1.0), b, 1e-15) && close(at(a, b, 1.0), a, 1e-15));
            assert!(close(at(b, a, 2.0), a * b, 1e-12) && close(at(a, b, 2.0), b * a, 1e-12));
        }
        // a quarter turn turns: every step of the way keeps lengths
        for i in 0..=20 {
            let m = along(turn(90.0), i as f64 / 20.0);
            assert!((m * DVec2::new(0.3, 0.8)).length() - DVec2::new(0.3, 0.8).length() < 1e-12);
        }
    }

    /// The captions of the "things to try", held to account.
    #[test]
    fn a_turn_and_an_uneven_stretch_do_what_the_caption_says() {
        let s = run(0, 30.0);
        let Setting { a, b, p, .. } = s.shown();
        assert!(!s.space && !s.small && p == 2.0);
        // A stretches x to 1.6 and y to 0.6; B is a quarter turn
        assert!(close(a, rows([1.6, 0.0, 0.0, 0.6]), 0.0) && close(b, turn(90.0), 1e-15));
        let (ab, ba) = (a * b, b * a);
        // yellow (AB) ends wide and low, white (BA) tall and narrow
        let extent = |m: DMat2| SQUARE.iter().map(|c| (m * *c).abs()).fold(DVec2::ZERO, |e, q| e.max(q));
        let (wide, tall) = (extent(ab), extent(ba));
        assert!(wide.x > 1.5 && wide.y < 0.7 && tall.x < 0.7 && tall.y > 1.5, "{wide} {tall}");
        // the red arrows from BA's ends to AB's are the columns of AB - BA, size 1.41
        let c = bracket(a, b);
        for e in [DVec2::X, DVec2::Y] {
            assert!(((ab * e - ba * e) - c * e).length() < 1e-15);
        }
        assert!(!commute(a, b) && amount(norm(c)) == "1.41");
        // a quarter turn keeps no line; B moves both of A's eigenlines, the axes
        assert!(matches!(eigen(b), Eigen::Complex { .. }));
        assert!(matches!(eigen(a), Eigen::Real { lines: Lines::Two(v1, v2), .. } if v1.y.abs() < 1e-15 && v2.x.abs() < 1e-15));
        assert_eq!(why(a, b), Why::Lines(0, None));
        assert!(s.verdict().contains(r"\ne"));
    }

    #[test]
    fn two_turns_do_what_the_caption_says() {
        let s = run(1, 30.0);
        let Setting { a, b, .. } = s.shown();
        // 45 and 90 degrees; either way 135; AB = BA, the ends on each other
        assert!(close(a, turn(45.0), 1e-15) && close(b, turn(90.0), 1e-15));
        assert!(close(a * b, turn(135.0), 1e-12) && close(b * a, turn(135.0), 1e-12));
        assert!(commute(a, b) && norm(bracket(a, b)) < 1e-15 && s.verdict().contains("does not matter"));
        // no line kept: complex eigenvalues, for any turn but 0 and 180 degrees
        assert!(matches!(eigen(a), Eigen::Complex { .. }) && matches!(eigen(b), Eigen::Complex { .. }));
        for d in [1.0, 30.0, 90.0, 135.0, 179.0, 181.0, 300.0] {
            assert!(matches!(eigen(turn(d)), Eigen::Complex { .. }), "{d}");
        }
        assert!(matches!(eigen(turn(180.0)), Eigen::Real { lines: Lines::Every, .. }));
        // B = -I + 1.41 A, a turn and an even stretch
        let Why::Span { complex: true, ab: Some((x, y)) } = why(a, b) else { panic!("{:?}", why(a, b)) };
        assert!(num(x, 2) == "−1.00" && num(y, 2) == "1.41", "{x} {y}");
        // what commutes with a turn (not by 0 or 180 degrees) turns and stretches evenly:
        // [[p, -q], [q, p]]; no uneven stretch is among them
        let mut rng = Rng::new(2);
        for d in [45.0, 10.0, 90.0, 135.0, 200.0, 359.0] {
            let a = turn(d);
            for _ in 0..200 {
                let m = random(&mut rng, 2.0);
                let even = (m.x_axis.x - m.y_axis.y).abs() < 1e-12 && (m.y_axis.x + m.x_axis.y).abs() < 1e-12;
                assert_eq!(commute(a, m), even);
                let (p, q) = (rng.range(-2.0, 2.0) as f64, rng.range(-2.0, 2.0) as f64);
                assert!(commute(a, rows([p, -q, q, p])));
                if (p - q).abs() > 0.01 {
                    assert!(!commute(a, rows([p, 0.0, 0.0, q])), "an uneven stretch commuted with a turn");
                }
            }
        }
    }

    #[test]
    fn the_same_axes_then_turned_do_what_the_caption_says() {
        // at the end of the film, before the turn: the same lines, AB = BA
        let mut s = Commute::create();
        s.set_up(2);
        let mut seen_together = false;
        for _ in 0..(30 * 60) {
            s.advance(1.0 / 60.0, 1.0 / 60.0);
            let sh = s.shown();
            if sh.p == 2.0 && s.acts.first().is_some_and(|a| matches!(a, Act::Hold(_))) {
                assert!(commute(sh.a, sh.b) && why(sh.a, sh.b) == Why::Lines(2, None));
                seen_together = true;
            }
            if !s.animated() {
                break;
            }
        }
        assert!(seen_together);
        let Setting { a, b, .. } = s.shown();
        assert!(close(a, rows([2.0, 0.0, 0.0, 0.5]), 0.0));
        // B is the stretch by 0.6 and 1.6 along axes turned by 30 degrees
        let r = turn(30.0);
        assert!(close(b, r * rows([0.6, 0.0, 0.0, 1.6]) * r.transpose(), 1e-12), "{b:?}");
        let Eigen::Real { l1, l2, lines: Lines::Two(v1, _) } = eigen(b) else { panic!() };
        assert!((l1 - 1.6).abs() < 1e-12 && (l2 - 0.6).abs() < 1e-12);
        assert!((v1.y.atan2(v1.x).to_degrees().rem_euclid(180.0) - 120.0).abs() < 1e-6, "{v1}");
        // the ends part: size 0.92
        assert!(!commute(a, b) && amount(norm(bracket(a, b))) == "0.92");
        // both symmetric; the product symmetric exactly when they commute
        assert!(close(b, b.transpose(), 1e-15));
        assert!(close((a * b).transpose(), b * a, 1e-15) && !close(a * b, (a * b).transpose(), 0.1));
        let s0 = rows([0.6, 0.0, 0.0, 1.6]);
        assert!(close(a * s0, (a * s0).transpose(), 0.0));
        let mut rng = Rng::new(4);
        for _ in 0..300 {
            let (p, q) = (random(&mut rng, 2.0), random(&mut rng, 2.0));
            let (p, q) = (p + p.transpose(), q + q.transpose());
            let symmetric = norm(p * q - (p * q).transpose()) <= COMMUTE_TOL * norm(p) * norm(q);
            assert_eq!(symmetric, commute(p, q));
        }
    }

    #[test]
    fn small_maps_do_what_the_caption_says() {
        let s = run(SMALL_TRIAL, 60.0);
        let Setting { a, b, e, .. } = s.shown();
        assert!(s.small && e == 0.125);
        // two shears, along x and along y
        assert_eq!(a * DVec2::new(0.3, 0.7), DVec2::new(0.7, 0.0));
        assert_eq!(b * DVec2::new(0.3, 0.7), DVec2::new(0.0, 0.3));
        let gap = |e: f64| norm(bracket(DMat2::IDENTITY + a * e, DMat2::IDENTITY + b * e));
        assert!(amount(gap(1.0)) == "1.41" && num(gap(0.5), 2) == "0.35" && num(gap(0.25), 3) == "0.088" && num(gap(0.125), 3) == "0.022");
        for e in [1.0, 0.5, 0.25, 0.125, 0.03] {
            assert!((gap(e / 2.0) * 4.0 - gap(e)).abs() < 1e-14, "{e}");
        }
        // the halvings went 1, 1/2, 1/4, 1/8: e was seen at each
        let mut s = Commute::create();
        s.set_up(SMALL_TRIAL);
        let mut stops = Vec::new();
        for _ in 0..(60 * 60) {
            s.advance(1.0 / 60.0, 1.0 / 60.0);
            let e = s.shown().e;
            if s.glide.arrived() && stops.last() != Some(&e) {
                stops.push(e);
            }
        }
        assert_eq!(stops, vec![1.0, 0.5, 0.25, 0.125]);
    }

    #[test]
    fn two_quarter_turns_of_a_book_do_what_the_caption_says() {
        let s = run(4, 30.0);
        assert!(s.space && s.shown().theta == 90.0 && s.shown().p == 2.0);
        let (r1, r2) = space_ends(90.0);
        assert_eq!(deg(apart(r1, r2)), "120.0°");
        assert!(norm3(r1 - r2) > 2.0);
        // the books end where the two products are
        assert!(norm3(book_pose(90.0, 2.0, true) - r1) < 1e-12 && norm3(book_pose(90.0, 2.0, false) - r2) < 1e-12);
        // the gap falls as theta squared for small angles: halving theta quarters it
        for th in [10.0, 5.0, 2.0] {
            let gap = |t: f64| {
                let (p, q) = space_ends(t);
                norm3(p - q)
            };
            let ratio = gap(th) / gap(th / 2.0);
            assert!((ratio - 4.0).abs() < 0.02, "{th}: {ratio}");
            let (p, q) = space_ends(th);
            let angle = apart(p, q) / apart(space_ends(th / 2.0).0, space_ends(th / 2.0).1);
            assert!((angle - 4.0).abs() < 0.02, "{th}: {angle}");
        }
        // they meet again at 180 degrees, and not between
        let (p, q) = space_ends(180.0);
        assert!(norm3(p - q) < 1e-12);
        for th in (1..180).map(|t| t as f64) {
            let (p, q) = space_ends(th);
            assert!(norm3(p - q) > 1e-4, "{th}");
        }
    }

    /// Every choice of B made from A commutes with it, and is offered only when it
    /// exists and can be typed.
    #[test]
    fn the_choices_made_from_a_commute_with_it() {
        let mut rng = Rng::new(9);
        let mut mats: Vec<DMat2> = A_PRESETS.iter().map(|p| rows(p.1)).collect();
        mats.extend((0..200).map(|_| random(&mut rng, 3.0)));
        mats.push(DMat2::ZERO);
        mats.push(rows([1.0, 2.0, 2.0, 4.0]));
        for a in mats {
            for (_, c) in B_PRESETS.iter().filter(|p| !matches!(p.1, BChoice::Fixed(_))) {
                if let Some(b) = b_choice(*c, a) {
                    assert!(commute(a, b) && size(b) <= ENTRY_MAX, "{a:?} {c:?}");
                }
            }
            assert_eq!(b_choice(BChoice::Inverse, a).is_none(), a.determinant().abs() <= EXACT * size(a) * size(a) || size(a.inverse()) > ENTRY_MAX);
        }
    }

    /// Degenerate input: every number drawn or read out is finite, and the words still
    /// say something true.
    #[test]
    fn degenerate_input_leaves_everything_finite() {
        let mats = [DMat2::ZERO, DMat2::IDENTITY * 1e-13, rows([0.0, 1.0, 0.0, 0.0]), rows([ENTRY_MAX; 4]), rows([-ENTRY_MAX, ENTRY_MAX, -ENTRY_MAX, ENTRY_MAX]), rows([1.0, 1.0, 1.0, 1.0]), DMat2::IDENTITY * -ENTRY_MAX, turn(180.0)];
        for a in mats {
            for b in mats {
                for small in [false, true] {
                    let mut s = Commute::create();
                    s.small = small;
                    s.glide = Glide::at(Setting { a, b, e: E_MIN, w: small as u8 as f64, p: 1.3, theta: 0.0 });
                    let (m1, m2) = s.maps();
                    for p in [0.0, 0.5, 1.0, 1.5, 2.0] {
                        assert!(at(m2, m1, p).is_finite() && at(m1, m2, p).is_finite());
                    }
                    for text in [s.verdict(), s.reason(), s.stage(), amount(norm(bracket(m1, m2))), eigen_text(eigen(a))] {
                        assert!(!text.contains("NaN") && !text.contains("inf") && !text.contains('?'), "{text}");
                    }
                    let (lo, hi) = s.bounds();
                    assert!(lo.is_finite() && hi.is_finite());
                    assert!(matches!(s.the_view(), View::Fit { center, half } if center.is_finite() && half.is_finite()));
                }
            }
        }
        for theta in [0.0, 1e-9, 90.0, 180.0] {
            let (p, q) = space_ends(theta);
            assert!(apart(p, q).is_finite() && norm3(p - q).is_finite());
        }
    }

    /// The panel and the picture fit the canvas, from a phone to a desktop.
    #[test]
    fn the_layout_fits_every_canvas() {
        for (w, h) in [(1080.0, 1000.0), (580.0, 700.0), (390.0, 800.0), (390.0, 560.0), (800.0, 600.0), (1600.0, 1000.0), (240.0, 180.0), (620.0, 480.0)] {
            for inset in [0.0, 46.0] {
                for shows in [Shows::Plane { small: false }, Shows::Plane { small: true }, Shows::Space] {
                    let view = Vec2::new(w, h);
                    let lay = Layout::of(view, inset, shows, None);
                    let (min, max) = lay.picture;
                    assert!(min.cmpge(Vec2::ZERO).all() && max.cmple(view).all() && (max - min).min_element() > 50.0, "{w}x{h}: {lay:?}");
                    if let Some((pmin, pmax)) = lay.panel {
                        assert!(pmin.cmpge(Vec2::ZERO).all() && pmax.cmple(view + Vec2::splat(0.5)).all(), "{w}x{h}: {lay:?}");
                        assert!(pmax.y - pmin.y >= panel_height(shows, lay.text, lay.narrow) - 0.5, "{w}x{h}: {lay:?}");
                        assert!(pmin.y >= inset || lay.narrow, "{w}x{h}: under the buttons");
                    }
                }
            }
        }
    }

    /// Dragging a dot turns all of its matrix: the eigenvalues stay, the lines turn by
    /// the angle dragged; and near one of the other's lines it is put exactly on it.
    #[test]
    fn a_dot_turns_its_matrix_and_finds_where_they_commute() {
        let a = rows([2.0, 0.0, 0.0, 0.5]);
        let b0 = rows([0.6, 0.0, 0.0, 1.6]);
        let r30 = turn(30.0);
        let b = r30 * b0 * r30.transpose();
        for t in [0.1, -0.7, 1.3, 2.9] {
            let r = DMat2::from_angle(t);
            let m = r * b * r.transpose();
            let (Eigen::Real { l1, l2, lines: Lines::Two(v1, _) }, Eigen::Real { lines: Lines::Two(w1, _), .. }) = (eigen(m), eigen(b)) else { panic!() };
            assert!((l1 - 1.6).abs() < 1e-12 && (l2 - 0.6).abs() < 1e-12);
            assert!((v1.perp_dot(r * w1)).abs() < 1e-12, "{t}");
        }
        // turned back by nearly 30 degrees, it is put exactly back, and they commute
        let pi = std::f64::consts::PI;
        for near in [-30.0f64, -29.0, -31.5, 60.0 - 0.8, 150.0 + 1.2] {
            let t = snapped(near.to_radians(), a, b);
            let r = DMat2::from_angle(t);
            let m = r * b * r.transpose();
            assert!(commute(a, m) && why(a, m) == Why::Lines(2, None), "{near}: {t}");
            assert!(((t + pi / 6.0) / (pi / 2.0)).fract().abs().min(1.0 - ((t + pi / 6.0) / (pi / 2.0)).fract().abs()) < 1e-12, "{near}");
        }
        // further off, it is left where it is put
        for far in [-20.0f64, 10.0, -45.0] {
            assert_eq!(snapped(far.to_radians(), a, b), far.to_radians());
            let r = DMat2::from_angle(far.to_radians());
            assert!(!commute(a, r * b * r.transpose()));
        }
        // nothing to snap to for a turn, which keeps no line
        assert_eq!(snapped(0.01, turn(90.0), b), 0.01);
    }

    /// What the heading says, in each experiment, as it ends.
    #[test]
    fn the_heading_says_what_is_so() {
        let said = |k: usize| {
            let s = run(k, 60.0);
            (s.verdict(), s.reason())
        };
        assert_eq!(said(0), (r"$AB \ne BA$: the order matters".to_string(), "$B$ keeps neither of $A$'s eigenlines".to_string()));
        assert_eq!(said(1), ("$AB = BA$: the order does not matter".to_string(), "$A$ keeps no line: only the matrices $aI + bA$ commute with it, and $B = −1.00 I + 1.41 A$".to_string()));
        assert_eq!(said(2).1, "$B$ keeps neither of $A$'s eigenlines");
        assert_eq!(said(3), (r"$PQ \ne QP$: the order matters".to_string(), r"$PQ - QP = e^2(AB - BA)$: the gap is $e^2 = 0.016$ times $\|AB - BA\| = 1.41$".to_string()));
        assert_eq!(said(4).0, r"$R_y R_x \ne R_x R_y$: the ends are 120.0° apart");
        // B = −I + 1.41 A is B
        assert!(close(DMat2::IDENTITY * -1.0 + turn(45.0) * 2f64.sqrt(), turn(90.0), 1e-15));
        // one line kept, and which
        let a = rows([2.0, 0.0, 0.0, 0.5]);
        assert_eq!(why(a, rows([1.0, 1.0, 0.0, 2.0])), Why::Lines(1, Some(0)));
        assert_eq!(why(a, rows([1.0, 0.0, 1.0, 2.0])), Why::Lines(1, Some(1)));
        assert!(Why::Lines(1, Some(0)).says("A", "B").contains("orange eigenline, but not its purple"));
    }

    /// The heading never contradicts itself: whatever the matrices, with A and B or with
    /// the small maps at any e (or on the way to them), the first line says the order
    /// does not matter exactly when the second gives a reason why it does not.
    #[test]
    fn the_verdict_and_the_reason_agree() {
        let mut rng = Rng::new(13);
        let mut pairs: Vec<(DMat2, DMat2)> = (0..300).map(|_| (random(&mut rng, 2.0), random(&mut rng, 2.0))).collect();
        for _ in 0..150 {
            let a = random(&mut rng, 2.0);
            pairs.extend([(a, a * a - a * 2.0 + DMat2::IDENTITY), (DMat2::IDENTITY * 1.5, a), (a, DMat2::IDENTITY * -0.5), (a, a)]);
        }
        pairs.extend([(rows([2.0, 0.0, 0.0, 0.5]), rows([1.0, 1.0, 0.0, 2.0])), (rows([1.0, 1.0, 0.0, 1.0]), rows([1.0, 2.0, 0.0, 1.0])), (turn(30.0), turn(100.0)), (turn(30.0), rows([2.0, 0.0, 0.0, 1.0]))]);
        for (a, b) in pairs {
            for (w, e) in [(0.0, 1.0), (1.0, 1.0), (1.0, 0.01), (0.6, 0.3)] {
                let mut s = Commute::create();
                s.small = w > 0.0;
                s.glide = Glide::at(Setting { a, b, e, w, p: 2.0, theta: 90.0 });
                let (verdict, reason) = (s.verdict(), s.reason());
                let together = verdict.contains("does not matter");
                assert_eq!(together, commute(a, b), "{verdict}");
                let because = reason.contains("keeps both") || reason.contains("every matrix commutes") || reason.contains("for every $e$") || (reason.contains("commute with it, and") && !reason.contains("is not one"));
                assert_eq!(together, because, "{verdict} / {reason}");
            }
        }
    }
}

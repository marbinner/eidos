//! Five ways to take a matrix apart: one 2×2 matrix A, written five ways as a product
//! of simpler maps (LU, QR, its eigenvectors, the SVD, the polar form), each played as
//! a film of its factors acting one after another, the right-hand one first. Every
//! film ends in the same picture: the plane as A takes it.

use super::*;
use glam::{DMat2, DVec2};

pub const DEF: SceneDef = SceneDef {
    id: "factor",
    title: "Five ways to take a matrix apart",
    group: FACTOR,
    blurb: "LU, QR, eigenvectors, the SVD and the polar form of one matrix, each as a film of simple steps.",
    make: |_| Box::new(Factor::new()),
};

// ----- the arithmetic, in double precision -------------------------------------------

/// The largest entry of A either way: the pictures of the films then stay within a few
/// units, but for LU with a small first pivot, whose factors are as large as they are.
const ENTRY_MAX: f64 = 3.0;
/// What counts as exactly zero, against the size of the entries (squared, for a
/// determinant or a discriminant).
const EXACT: f64 = 1e-9;
/// A column dragged by its handle is put on the nearest multiple of this, so that a
/// zero (a₁₁ = 0, a column on an axis, two columns in line) can be reached by hand.
const SNAP: f64 = 0.05;
/// A factor nearer the identity than this, in every entry, takes no time in its film:
/// nothing would be seen moving.
const IDLE: f64 = 1e-12;

/// A matrix by its rows, [[a, b], [c, d]], the way matrices are written.
fn rows([a, b, c, d]: [f64; 4]) -> DMat2 {
    DMat2::from_cols(DVec2::new(a, c), DVec2::new(b, d))
}

/// The size of the largest entry.
fn size(m: DMat2) -> f64 {
    m.to_cols_array().iter().fold(0.0f64, |s, v| s.max(v.abs()))
}

fn trace(m: DMat2) -> f64 {
    m.x_axis.x + m.y_axis.y
}

/// The turn by `theta` radians, anticlockwise.
fn turn(theta: f64) -> DMat2 {
    DMat2::from_angle(theta)
}

/// An angle brought into (−π, π].
fn wrapped_angle(theta: f64) -> f64 {
    let pi = std::f64::consts::PI;
    let t = (theta + pi).rem_euclid(2.0 * pi) - pi;
    if t <= -pi {
        t + 2.0 * pi
    } else {
        t
    }
}

/// (tr A / 2)² − det A, a quarter of the discriminant of λ² − (tr A) λ + det A, worked
/// out as ((a − d)/2)² + bc so that a double eigenvalue gives 0 exactly.
fn quarter_disc(m: DMat2) -> f64 {
    let half = 0.5 * (m.x_axis.x - m.y_axis.y);
    half * half + m.y_axis.x * m.x_axis.y
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

/// The eigenvalues of A and the lines they keep.
#[derive(Clone, Copy, Debug, PartialEq)]
enum Eigen {
    /// Two real eigenvalues, the larger first.
    Real { l1: f64, l2: f64, lines: Lines },
    /// A complex pair re ± im i (im > 0).
    Complex { re: f64, im: f64 },
}

/// The eigenlines of a matrix with real eigenvalues, as unit directions.
#[derive(Clone, Copy, Debug, PartialEq)]
enum Lines {
    /// One for each eigenvalue, the larger's first.
    Two(DVec2, DVec2),
    /// A double eigenvalue with a single line (a shear is in it).
    One(DVec2),
    /// Every line through 0 (A is a multiple of I).
    Every,
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

/// The eigenvalues in a formula: "$2.00$ and $0.50$", "$1.00$ twice", "$0.60 \pm 0.80i$".
fn eigen_text(e: Eigen) -> String {
    match e {
        Eigen::Real { l1, lines: Lines::One(_) | Lines::Every, .. } => format!("${}$ twice", num(l1, 2)),
        Eigen::Real { l1, l2, .. } => format!("${}$ and ${}$", num(l1, 2), num(l2, 2)),
        Eigen::Complex { re, im } => format!(r"${} \pm {}i$", num(re, 2), num(im, 2)),
    }
}

/// An angle for the screen, in degrees to a tenth, with no "−0.0°".
fn degrees(radians: f64) -> String {
    let d = radians.to_degrees();
    format!("{:.1}°", if d.abs() < 0.05 { 0.0 } else { d })
}

/// A turn in words: "26.6° anticlockwise", "11.3° clockwise".
fn turn_words(theta: f64) -> String {
    let way = if theta < 0.0 { "clockwise" } else { "anticlockwise" };
    format!("{} {way}", degrees(theta.abs()))
}

/// The direction of a mirror's line, an angle in [0°, 180°).
fn line_angle(theta: f64) -> f64 {
    theta.rem_euclid(std::f64::consts::PI)
}

// ----- the five factorisations --------------------------------------------------------

/// The five, in the order of the tabs.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Kind {
    Lu,
    Qr,
    Eigen,
    Svd,
    Polar,
}

const KINDS: [Kind; 5] = [Kind::Lu, Kind::Qr, Kind::Eigen, Kind::Svd, Kind::Polar];

impl Kind {
    /// The words on its tab.
    fn tab(self) -> &'static str {
        match self {
            Kind::Lu => "LU",
            Kind::Qr => "QR",
            Kind::Eigen => "eigen",
            Kind::Svd => "SVD",
            Kind::Polar => "polar",
        }
    }

    /// What it is, as the panel's title says it (after "A = …:").
    fn says(self) -> &'static str {
        match self {
            Kind::Lu => "elimination as a product",
            Kind::Qr => "Gram–Schmidt as a product",
            Kind::Eigen => "a stretch along the eigenvectors",
            Kind::Svd => "turn, stretch, turn",
            Kind::Polar => "a stretch, then a turn",
        }
    }

    /// When it exists, when it is unique, and what it is for, a short line each (the
    /// panel on the picture; the side panel says more).
    fn facts(self) -> [&'static str; 3] {
        match self {
            Kind::Lu => [r"exists if $a_{11} \ne 0$ (if not, swap the rows first)", r"unique: whenever $a_{11} \ne 0$", "for: solving $Ax = b$"],
            Kind::Qr => ["exists: always", r"unique: if $A$ is invertible ($r_{11}, r_{22} > 0$)", "for: least squares, perpendicular bases"],
            Kind::Eigen => ["exists: if there are two independent eigenvectors", "real: only if the eigenvalues are real", "for: powers $A^k$, and $e^{At}$"],
            Kind::Svd => ["exists: always", r"unique: $\Sigma$; $U$, $V$ but for signs if $\sigma_1 > \sigma_2 > 0$", "for: lengths, rank, low-rank approximation"],
            Kind::Polar => ["exists: always", "unique: $P$; $Q$ too if $A$ is invertible", "for: the nearest orthogonal matrix, $Q$"],
        }
    }
}

/// How a factor is played: how the plane gets from where it was to where the factor
/// takes it.
#[derive(Clone, Copy, Debug, PartialEq)]
enum Motion {
    /// Every point slides straight to its place: a shear, a stretch along the axes or
    /// along two lines at right angles (a negative stretch passes through flat and
    /// opens out flipped), or a mirror, which folds the plane flat onto its line and
    /// opens it out on the other side.
    Slide,
    /// A turn by this angle (radians), the short way round.
    Turn(f64),
    /// Any other matrix (V and V⁻¹, or A in one step): the turn in it turns while the
    /// rest slides (`along`).
    Blend,
}

/// Does `m` turn the plane: does it turn every direction (complex eigenvalues), or send
/// the plane half way round (two negative eigenvalues)? Sliding every point straight to
/// its place would then cut across the turn and shrink everything on the way.
fn turns(m: DMat2) -> bool {
    let (det, tr) = (m.determinant(), trace(m));
    det > 0.0 && tr < 2.0 * det.sqrt()
}

/// A share `s` of the way from I to `m`: straight, or (for a map that turns) turning by
/// the turn in it (the angle of its polar form m = R(θ) S, S symmetric) while the rest,
/// S, slides. For det m > 0 the plane never passes through flat on the way.
fn along(m: DMat2, s: f64) -> DMat2 {
    if !turns(m) {
        return DMat2::IDENTITY * (1.0 - s) + m * s;
    }
    let theta = (m.x_axis.y - m.y_axis.x).atan2(m.x_axis.x + m.y_axis.y);
    let rest = turn(-theta) * m;
    turn(theta * s) * (DMat2::IDENTITY * (1.0 - s) + rest * s)
}

/// A share `s` (0 to 1) of the way through a factor `m` that moves as `motion` says.
fn partial(m: DMat2, motion: Motion, s: f64) -> DMat2 {
    if s >= 1.0 {
        return m;
    }
    if s <= 0.0 {
        return DMat2::IDENTITY;
    }
    match motion {
        Motion::Slide => DMat2::IDENTITY * (1.0 - s) + m * s,
        Motion::Turn(theta) => turn(theta * s),
        Motion::Blend => along(m, s),
    }
}

/// One factor of a product.
#[derive(Clone, Debug)]
struct Part {
    /// Its name, a formula without the dollars: "L", "V^{-1}", "\Sigma".
    name: &'static str,
    m: DMat2,
    motion: Motion,
    /// What it does to the plane, for the heading while it acts.
    does: String,
    /// The direction of its mirror's line (radians), if it is a mirror (det = −1).
    mirror: Option<f64>,
}

impl Part {
    fn new(name: &'static str, m: DMat2, motion: Motion, does: String) -> Part {
        Part { name, m, motion, does, mirror: None }
    }

    /// Is it I, to rounding? It then takes no time in its film.
    fn idle(&self) -> bool {
        size(self.m - DMat2::IDENTITY) <= IDLE
    }
}

/// The two arrows that ride along in some films (orange and purple): where they start,
/// and what they are called once each factor has acted.
#[derive(Clone, Debug)]
struct Arrows {
    start: [DVec2; 2],
    /// `names[k]`: once the first k factors to act have acted.
    names: Vec<[&'static str; 2]>,
}

/// One of the five for one matrix.
#[derive(Clone, Debug)]
struct Split {
    kind: Kind,
    /// The product as it is written for this matrix: "LU", or "PLU" with a swap.
    product: &'static str,
    /// The factors, left to right as written (none if there are none).
    parts: Vec<Part>,
    /// Why there are none (only the eigenvectors can fail).
    missing: Option<String>,
    arrows: Option<Arrows>,
}

impl Split {
    /// The product of the factors, as worked out from them.
    fn rebuilt(&self) -> Option<DMat2> {
        (self.missing.is_none()).then(|| self.parts.iter().fold(DMat2::IDENTITY, |p, part| p * part.m))
    }

    /// The equation as a formula: "A = LU".
    fn equation(&self) -> String {
        format!("$A = {}$", self.product)
    }
}

fn split(kind: Kind, a: DMat2) -> Split {
    match kind {
        Kind::Lu => lu(a),
        Kind::Qr => qr(a),
        Kind::Eigen => eigen_split(a),
        Kind::Svd => svd_split(a),
        Kind::Polar => polar_split(a),
    }
}

/// A = LU, L lower triangular with 1s on its diagonal and U upper triangular: the
/// elimination of the scene on elimination written as a product. If a₁₁ = 0 and a₂₁ is
/// not, no multiple of row 1 can clear a₂₁: the rows are swapped first, PA = LU, and
/// A = PLU (a swap undoes itself).
fn lu(a: DMat2) -> Split {
    let s = size(a);
    let zero = |v: f64| v.abs() <= EXACT * s;
    let [a11, a21, a12, a22] = a.to_cols_array();
    let swap = zero(a11) && !zero(a21);
    let ([t1, t2], [b1, b2]) = if swap { ([a21, a22], [a11, a12]) } else { ([a11, a12], [a21, a22]) };
    // (a first column of zeros has nothing under its pivot place to clear)
    let l21 = if zero(t1) { 0.0 } else { b1 / t1 };
    let u = rows([t1, t2, 0.0, b2 - l21 * t2]);
    let l = rows([1.0, 0.0, l21, 1.0]);
    let u_does = format!(r"$U$, upper triangular: $(1, 0)$ goes to ${}$, on the $x$ axis, and $(0, 1)$ to ${}$", tuple(&[t1, 0.0], 2), tuple(&[t2, b2 - l21 * t2], 2));
    let l_does = format!(r"$L$, a shear: each point $(x, y)$ moves straight up or down to $(x, y + \ell_{{21}} x)$, with $\ell_{{21}} = {}$", num(l21, 2));
    let mut parts = Vec::new();
    if swap {
        let does = r"$P$, the swap of the rows: as a map it takes $(x, y)$ to $(y, x)$, a mirror in the line $y = x$".to_string();
        parts.push(Part { mirror: Some(std::f64::consts::FRAC_PI_4), ..Part::new("P", rows([0.0, 1.0, 1.0, 0.0]), Motion::Slide, does) });
    }
    parts.push(Part::new("L", l, Motion::Slide, l_does));
    parts.push(Part::new("U", u, Motion::Slide, u_does));
    Split { kind: Kind::Lu, product: if swap { "PLU" } else { "LU" }, parts, missing: None, arrows: None }
}

/// A = QR, Q orthogonal and R upper triangular with no negative number on its diagonal
/// (positive if A is invertible: then the two are unique). Q is the turn that takes
/// (1, 0) onto the direction of a₁ (of a₂ if a₁ = 0), with a mirror after it if A turns
/// the plane over.
fn qr(a: DMat2) -> Split {
    let s = size(a);
    let (a1, a2) = (a.x_axis, a.y_axis);
    let mut theta = if a1.length() > EXACT * s {
        a1.y.atan2(a1.x)
    } else if a2.length() > EXACT * s {
        a2.y.atan2(a2.x)
    } else {
        0.0
    };
    if theta.abs() <= IDLE {
        theta = 0.0;
    }
    let mut q = turn(theta);
    let mut r = q.transpose() * a;
    // (what is left below the diagonal is rounding)
    r.x_axis.y = 0.0;
    let mirror = r.y_axis.y < -EXACT * s;
    if mirror {
        q.y_axis = -q.y_axis;
        r.y_axis.y = -r.y_axis.y;
    }
    let (r11, r12, r22) = (r.x_axis.x, r.y_axis.x, r.y_axis.y);
    let r_does = if r11 > EXACT * s {
        format!(r"$R$, upper triangular: $(1, 0)$ stays on the $x$ axis, stretched to $r_{{11}} = |a_1| = {}$, and $(0, 1)$ goes to ${}$", num(r11, 2), tuple(&[r12, r22], 2))
    } else {
        format!(r"$R$, upper triangular: $a_1 = 0$, so $(1, 0)$ goes to $0$, and $(0, 1)$ to ${}$", tuple(&[r12, r22], 2))
    };
    let q_part = if mirror {
        let line = line_angle(0.5 * theta);
        let does = format!(r"$Q$ is a mirror in the line at {} ($\det A < 0$, and $R$ keeps its diagonal positive)", degrees(line));
        Part { mirror: Some(line), ..Part::new("Q", q, Motion::Slide, does) }
    } else {
        Part::new("Q", q, Motion::Turn(theta), format!("$Q$ turns everything {}", turn_words(theta)))
    };
    Split { kind: Kind::Qr, product: "QR", parts: vec![q_part, Part::new("R", r, Motion::Slide, r_does)], missing: None, arrows: None }
}

/// A = V D V⁻¹: V's columns are unit eigenvectors (the larger eigenvalue's first,
/// pointing right, and the second's turned so that det V > 0), D the eigenvalues.
/// None with real numbers if the eigenvalues are complex, and none at all if there is
/// only one eigenline.
fn eigen_split(a: DMat2) -> Split {
    let none = |why: String| Split { kind: Kind::Eigen, product: r"VDV^{-1}", parts: Vec::new(), missing: Some(why), arrows: None };
    let names = vec![["$v_1$", "$v_2$"], ["$e_1$", "$e_2$"], [r"$\lambda_1 e_1$", r"$\lambda_2 e_2$"], [r"$\lambda_1 v_1$", r"$\lambda_2 v_2$"]];
    match eigen(a) {
        Eigen::Complex { re, im } => none(format!(r"no real eigenvectors: $A$ turns every line through $0$ off itself ($\lambda = {} \pm {}i$)", num(re, 2), num(im, 2))),
        Eigen::Real { l1, lines: Lines::One(_), .. } => {
            none(format!(r"one eigenline only ($\lambda = {}$ twice): $V$ needs two independent eigenvectors, and there is one", num(l1, 2)))
        }
        Eigen::Real { l1, lines: Lines::Every, .. } => {
            let d_does = format!(r"every direction is an eigenvector, so $V = I$, and $D = A$ takes every point $x$ to ${} x$", num(l1, 2));
            let i = DMat2::IDENTITY;
            let parts = vec![
                Part::new("V", i, Motion::Blend, String::new()),
                Part::new("D", i * l1, Motion::Slide, d_does),
                Part::new("V^{-1}", i, Motion::Blend, String::new()),
            ];
            Split { kind: Kind::Eigen, product: r"VDV^{-1}", parts, missing: None, arrows: Some(Arrows { start: [DVec2::X, DVec2::Y], names }) }
        }
        Eigen::Real { l1, l2, lines: Lines::Two(v1, v2) } => {
            let v1 = if v1.x < 0.0 || (v1.x == 0.0 && v1.y < 0.0) { -v1 } else { v1 };
            let v2 = if v1.perp_dot(v2) < 0.0 { -v2 } else { v2 };
            let v = DMat2::from_cols(v1, v2);
            let d = DMat2::from_diagonal(DVec2::new(l1, l2));
            let flip = |l: f64, axis: &str| if l < 0.0 && num(l, 2) != "0.00" { format!(", a flip along the {axis} axis") } else { String::new() };
            let d_does = format!(r"$D$ stretches along the axes: by $\lambda_1 = {}$ along $x$ and $\lambda_2 = {}$ along $y${}{}", num(l1, 2), num(l2, 2), flip(l1, "$x$"), flip(l2, "$y$"));
            let parts = vec![
                Part::new("V", v, Motion::Blend, r"$V$ carries the axes back onto the eigenvectors: $\lambda_1 e_1$ to $\lambda_1 v_1$, $\lambda_2 e_2$ to $\lambda_2 v_2$".into()),
                Part::new("D", d, Motion::Slide, d_does),
                Part::new("V^{-1}", v.inverse(), Motion::Blend, r"$V^{-1}$ carries the eigenvectors $v_1$ and $v_2$ onto the axes: from here on the coordinates are along them".into()),
            ];
            Split { kind: Kind::Eigen, product: r"VDV^{-1}", parts, missing: None, arrows: Some(Arrows { start: [v1, v2], names }) }
        }
    }
}

/// A = U Σ Vᵀ with σ₁ ≥ σ₂ ≥ 0 and V a turn, by the smallest angle that does it (V = I
/// if σ₁ = σ₂, when any V would do); U is a turn too unless det A < 0, when it holds
/// the mirror.
#[derive(Clone, Copy, Debug)]
struct Svd {
    u: DMat2,
    sigma: [f64; 2],
    v: DMat2,
    /// The angle of V (radians, in (−90°, 90°]), and of U's first column.
    theta_v: f64,
    theta_u: f64,
    mirror: bool,
}

fn svd(a: DMat2) -> Svd {
    let s = size(a);
    let ata = a.transpose() * a;
    let (p, q, r) = (ata.x_axis.x, ata.y_axis.x, ata.y_axis.y);
    // (AᵀA is a multiple of I, to rounding: every direction is stretched alike)
    let even = (p - r).hypot(2.0 * q) <= EXACT * (p + r);
    let mut theta_v = if even { 0.0 } else { 0.5 * (2.0 * q).atan2(p - r) };
    if theta_v.abs() <= IDLE {
        theta_v = 0.0;
    }
    let v = turn(theta_v);
    let det = a.determinant();
    let s1 = (a * v.x_axis).length();
    let s2 = if s1 > 0.0 { (det.abs() / s1).min(s1) } else { 0.0 };
    let u1 = if s1 > EXACT * s { (a * v.x_axis) / s1 } else { DVec2::X };
    let mut theta_u = u1.y.atan2(u1.x);
    if theta_u.abs() <= IDLE {
        theta_u = 0.0;
    }
    let mirror = det < -EXACT * s * s;
    let u1 = DVec2::from_angle(theta_u);
    let u = DMat2::from_cols(u1, if mirror { -u1.perp() } else { u1.perp() });
    Svd { u, sigma: [s1, s2], v, theta_v, theta_u, mirror }
}

fn svd_split(a: DMat2) -> Split {
    let d = svd(a);
    let [s1, s2] = d.sigma;
    let u_part = if d.mirror {
        let line = line_angle(0.5 * d.theta_u);
        let does = format!(r"$U$ is a mirror in the line at {}, taking the axes to $u_1$ and $u_2$ ($\det A < 0$: the mirror has to be somewhere, and $\Sigma$ is never negative)", degrees(line));
        Part { mirror: Some(line), ..Part::new("U", d.u, Motion::Slide, does) }
    } else {
        Part::new("U", d.u, Motion::Turn(d.theta_u), format!("$U$ turns the axes onto $u_1$ and $u_2$: {}", turn_words(d.theta_u)))
    };
    let sigma_does = format!(r"$\Sigma$ stretches along the axes: by $\sigma_1 = {}$ along $x$ and $\sigma_2 = {}$ along $y$", num(s1, 2), num(s2, 2));
    let parts = vec![
        u_part,
        Part::new(r"\Sigma", DMat2::from_diagonal(DVec2::new(s1, s2)), Motion::Slide, sigma_does),
        Part::new(r"V^\top", d.v.transpose(), Motion::Turn(-d.theta_v), format!(r"$V^\top$ turns $v_1$ and $v_2$ onto the axes: {}", turn_words(-d.theta_v))),
    ];
    let names = vec![["$v_1$", "$v_2$"], ["$e_1$", "$e_2$"], [r"$\sigma_1 e_1$", r"$\sigma_2 e_2$"], [r"$\sigma_1 u_1$", r"$\sigma_2 u_2$"]];
    Split { kind: Kind::Svd, product: r"U\Sigma V^\top", parts, missing: None, arrows: Some(Arrows { start: [d.v.x_axis, d.v.y_axis], names }) }
}

/// A = QP: P = VΣVᵀ, symmetric with no negative eigenvalue (a stretch along v₁ and v₂,
/// at right angles), and Q = UVᵀ, orthogonal: the turn, or turn and mirror, in A.
fn polar_split(a: DMat2) -> Split {
    let d = svd(a);
    let [s1, s2] = d.sigma;
    let mut p = d.v * DMat2::from_diagonal(DVec2::new(s1, s2)) * d.v.transpose();
    // (symmetric exactly, not to rounding)
    let off = 0.5 * (p.y_axis.x + p.x_axis.y);
    (p.y_axis.x, p.x_axis.y) = (off, off);
    let q = d.u * d.v.transpose();
    let q_part = if d.mirror {
        let line = line_angle(0.5 * (d.theta_u + d.theta_v));
        let does = format!(r"$Q$ is a mirror in the line at {}: $\det A < 0$, and $P$ has no negative eigenvalue", degrees(line));
        Part { mirror: Some(line), ..Part::new("Q", q, Motion::Slide, does) }
    } else {
        let theta = wrapped_angle(d.theta_u - d.theta_v);
        Part::new("Q", q, Motion::Turn(theta), format!("$Q$ turns everything {}", turn_words(theta)))
    };
    let p_does = format!(r"$P$ stretches by $\sigma_1 = {}$ along $v_1$ and by $\sigma_2 = {}$ along $v_2$: two lines at right angles, which stay where they are", num(s1, 2), num(s2, 2));
    let names = vec![["$v_1$", "$v_2$"], [r"$\sigma_1 v_1$", r"$\sigma_2 v_2$"], [r"$\sigma_1 u_1$", r"$\sigma_2 u_2$"]];
    Split { kind: Kind::Polar, product: "QP", parts: vec![q_part, Part::new("P", p, Motion::Slide, p_does)], missing: None, arrows: Some(Arrows { start: [d.v.x_axis, d.v.y_axis], names }) }
}

// ----- a film of the factors ---------------------------------------------------------

/// The factors of one product in the order they act (right to left), as a film: each
/// takes one share of it, but one that is I, which takes none. (If there are no
/// factors, the film is A in one step, which is all that can be shown then.)
struct Film {
    /// Each step's matrix, how it moves, and its share (1, or 0).
    steps: Vec<(DMat2, Motion, f64)>,
    /// The plane once every step before step k has acted.
    before: Vec<DMat2>,
    end: DMat2,
    /// How many steps take a share.
    total: f64,
}

impl Film {
    fn of(split: &Split, a: DMat2) -> Film {
        let steps: Vec<(DMat2, Motion, f64)> = if split.missing.is_some() {
            vec![(a, Motion::Blend, if size(a - DMat2::IDENTITY) <= IDLE { 0.0 } else { 1.0 })]
        } else {
            split.parts.iter().rev().map(|p| (p.m, p.motion, if p.idle() { 0.0 } else { 1.0 })).collect()
        };
        let mut before = Vec::with_capacity(steps.len());
        let mut m = DMat2::IDENTITY;
        for (f, _, _) in &steps {
            before.push(m);
            m = *f * m;
        }
        let total = steps.iter().map(|s| s.2).sum();
        Film { steps, before, end: m, total }
    }

    /// Where the film is at `phi` (0 to 1): (k, s), every step before step k has acted
    /// and step k has acted the share s of itself; k = the number of steps once all
    /// have.
    fn place(&self, phi: f64) -> (usize, f64) {
        let n = self.steps.len();
        if self.total <= 0.0 || phi >= 1.0 || !phi.is_finite() {
            return (n, 0.0);
        }
        let p = phi.max(0.0) * self.total;
        let mut done = 0.0;
        for (k, &(_, _, share)) in self.steps.iter().enumerate() {
            if share > 0.0 {
                if p < done + share {
                    return (k, (p - done) / share);
                }
                done += share;
            } else if p <= done {
                // (a step that does nothing, not yet passed)
                return (k, 0.0);
            }
        }
        (n, 0.0)
    }

    /// The plane at `phi`: each step eased in and out, so that the film rests a
    /// moment between two factors.
    fn at(&self, phi: f64) -> DMat2 {
        let (k, s) = self.place(phi);
        match self.steps.get(k) {
            Some(&(m, motion, _)) => partial(m, motion, smoothstep(s as f32) as f64) * self.before[k],
            None => self.end,
        }
    }

    /// The steps that take a share, by their place in `steps`.
    fn shown(&self) -> impl Iterator<Item = usize> + '_ {
        (0..self.steps.len()).filter(|&k| self.steps[k].2 > 0.0)
    }
}

// ----- experiments and quick choices -------------------------------------------------

struct Trial {
    label: &'static str,
    look_for: &'static str,
    /// The matrix, by rows.
    a: [f64; 4],
    /// The films it plays, one after another.
    tabs: &'static [Kind],
}

const FIRST: [f64; 4] = [1.0, 1.0, 0.5, 1.5];

const TRIALS: [Trial; 5] = [
    Trial {
        label: "the same map, five films",
        look_for: r"One matrix, its five films one after another. Each starts from the plane as it is and ends on the faint outline, where $A$ takes it. LU: a shear along $x$ ($U$), then one along $y$ ($L$). QR: $R$ keeps $(1, 0)$ on the $x$ axis, then $Q$ turns 26.6°. Eigenvectors: a stretch by 2 along $(1, 1)$ and by 0.5 along $(-2, 1)$. The SVD and the polar form: stretches by 2.07 and 0.48 along two lines at right angles.",
        a: FIRST,
        tabs: &KINDS,
    },
    Trial {
        label: "a turn",
        look_for: r"$A$ turns the plane 53.1° anticlockwise and stretches nothing. No line through $0$ stays on itself, so there are no real eigenvectors ($\lambda = 0.6 \pm 0.8i$): this film cannot be made, and $A$ is shown as one step. The other four tabs can: in QR, $Q = A$ and $R = I$; in the SVD, $\sigma_1 = \sigma_2 = 1$; in the polar form, $P = I$ and $Q = A$; and LU builds the turn out of two triangular maps.",
        a: [0.6, -0.8, 0.8, 0.6],
        tabs: &[Kind::Eigen],
    },
    Trial {
        label: "a zero in the corner",
        look_for: r"$a_{11} = 0$: elimination has no first pivot, and no multiple of row 1, which starts with 0, can clear the 1 under it, so $A$ has no LU. Swap the rows first: $PA = LU$, and $A = PLU$. Here $PA$ is triangular already, so $L = I$: $U$ acts, and then $P$, the swap, folds the plane over the line $y = x$: a mirror, as $\det A = -1.5 < 0$ needs.",
        a: [0.0, 1.5, 1.0, 0.5],
        tabs: &[Kind::Lu],
    },
    Trial {
        label: "a shear",
        look_for: r"$\lambda = 1$ twice, but only the $x$ axis stays on itself: one eigenline (orange). $V$ needs two independent eigenvectors and there is one, so there is no $VDV^{-1}$, real or complex, and $A$ is shown as one step. The other four exist: $A$ is upper triangular already, so LU and QR leave it as it is ($L = I$, $Q = I$), and the SVD stretches by 1.62 and 0.62.",
        a: [1.0, 1.0, 0.0, 1.0],
        tabs: &[Kind::Eigen],
    },
    Trial {
        label: "symmetric, positive definite",
        look_for: r"$A^\top = A$, and both eigenvalues are positive (2 and 0.5). Its eigenvectors $(1, 1)$ and $(-1, 1)$ are at right angles, so $V$ is a turn (45°) and $V^{-1} = V^\top$: the eigenvector film and the SVD film are one film, with $U = V$ and $\Sigma = D$. Then the polar form: $Q = I$ and $P = A$, the same stretch along the same two lines, in one step.",
        a: [1.25, 0.75, 0.75, 1.25],
        tabs: &[Kind::Eigen, Kind::Svd, Kind::Polar],
    },
];

/// Quick choices of A: (button, the matrix by rows, what it shows).
const PRESETS: [(&str, [f64; 4], &str); 4] = [
    ("the first", FIRST, "Every one of the five exists for this matrix, with real numbers, and each film is different."),
    (
        "a mirror",
        [1.2, 0.6, 0.8, -0.6],
        r"$\det A = -1.2 < 0$: $A$ turns the plane over (the letter comes out mirrored), so one of the factors must turn it over too. In QR that is $Q$, a mirror ($R$ keeps its diagonal positive); in the SVD, $U$ ($\Sigma$ is never negative, and $V$ is kept a turn); in the polar form, $Q$. In LU it is $U$, with a negative pivot $u_{22} = -1$; with the eigenvectors, $D$, with a negative eigenvalue $\lambda_2 = -0.84$.",
    ),
    (
        "a squash",
        [1.0, 0.5, 2.0, 1.0],
        r"$\det A = 0$: the columns are in line, and every point lands on one line. LU has $u_{22} = 0$, QR $r_{22} = 0$, the eigenvalues are 2 and 0, and $\sigma_2 = 0$. The polar form's $P$ is still unique, but $Q$ is not: a turn and a mirror both work, and this scene picks the turn.",
    ),
    (
        "diagonal",
        [2.0, 0.0, 0.0, 0.5],
        r"A stretch along the axes is already each of the five: in every one, each factor is $I$ but one, which is $A$ itself ($U$ in LU, $R$ in QR, $D$ with the eigenvectors, $\Sigma$ in the SVD, $P$ in the polar form), and every film is that one stretch.",
    ),
];

// ----- what glides when a button changes it ------------------------------------------

/// How fast a new matrix is glided to: each of the two eases one after the other
/// covers two thirds of what is left in this many seconds.
const FOLLOW: f64 = 0.07;

/// The matrix as asked for, and as drawn: the drawn one follows it through `between`,
/// two eases one after the other, so that it sets off gently and arrives exactly.
#[derive(Clone, Copy, Debug)]
struct Glide {
    to: DMat2,
    between: DMat2,
    shown: DMat2,
}

impl Glide {
    fn at(a: DMat2) -> Glide {
        Glide { to: a, between: a, shown: a }
    }

    fn arrived(&self) -> bool {
        self.shown == self.to && self.between == self.to
    }

    /// Move the drawn matrix `dt` seconds nearer to the one asked for.
    fn step(&mut self, dt: f64) {
        if self.arrived() || dt <= 0.0 {
            return;
        }
        let k = 1.0 - (-dt / FOLLOW).exp();
        self.between += (self.to - self.between) * k;
        self.shown += (self.between - self.shown) * k;
        // (exactly there in the end: the picture then stops changing)
        let apart = size(self.to - self.between).max(size(self.to - self.shown));
        if apart < 2e-4 || !apart.is_finite() {
            *self = Glide::at(self.to);
        }
    }
}

/// The matrix with every entry within ±`ENTRY_MAX` (and 0 for one that is not a number).
fn within_limits(m: DMat2) -> DMat2 {
    DMat2::from_cols_array(&m.to_cols_array().map(|v| if v.is_finite() { v.max(-ENTRY_MAX).min(ENTRY_MAX) } else { 0.0 }))
}

// ----- the screen panel's layout -------------------------------------------------------

/// Rows of the panel, in logical pixels.
const TAB_H: f32 = 28.0;
const TITLE_H: f32 = 18.0;
const NAME_H: f32 = 16.0;
const LINE_H: f32 = 16.0;
const BAR_H: f32 = 28.0;
/// The widest number in a matrix's cell, in characters (`cell` writes at most five).
const CELL: usize = 5;
/// The panel's width on a wide canvas.
const PANEL_W: f32 = 350.0;

#[derive(Clone, Debug)]
struct Panel {
    min: Vec2,
    max: Vec2,
    /// The size of the numbers in the matrices.
    text: f32,
    tabs: [(Vec2, Vec2); 5],
    title: Vec2,
    /// Top-left of the factors' names, over their matrices.
    factors: Vec2,
    /// Top-left of the row with the product and A, and of the line under it.
    check: Vec2,
    agree: Vec2,
    /// The two ends of the film's track, at its middle; the play button.
    bar: (Vec2, Vec2),
    play: (Vec2, Vec2),
    facts: Vec2,
    width: f32,
}

#[derive(Clone, Debug)]
struct Layout {
    /// A phone held upright: the panel goes below the picture.
    narrow: bool,
    /// The part of the canvas the picture has to itself.
    picture: (Vec2, Vec2),
    /// None on a canvas too small to spare the room (the numbers are in the side panel).
    panel: Option<Panel>,
}

/// The names written before the product and before A, on the panel.
const LEADS: [&str; 2] = [r"$U\Sigma V^\top =$", "$A =$"];

impl Layout {
    fn of(view: Vec2, inset: f32) -> Layout {
        let narrow = view.x < 620.0 && view.y >= 560.0;
        let pad = if narrow { 10.0 } else { 12.0 };
        let width = if narrow { view.x } else { PANEL_W };
        let inner = width - 2.0 * pad;
        // the numbers' size: as large as lets three matrices fit across, and the product
        // and A beside each other, up to 13
        let matrix_w = |s: f32| matrix_size(2, 2, CELL, s).x;
        let leads = text_width(LEADS[0], 1.0) + text_width(LEADS[1], 1.0);
        let fits = |s: f32| 3.0 * matrix_w(s) + 2.0 * 12.0 <= inner && leads * s + 2.0 * matrix_w(s) + 2.0 * 8.0 + 18.0 <= inner;
        let mut text = 13.0f32;
        while !fits(text) && text > 9.0 {
            text -= 0.5;
        }
        let mat_h = matrix_size(2, 2, CELL, text).y;
        let height = 2.0 * pad + TAB_H + 8.0 + TITLE_H + 6.0 + NAME_H + mat_h + 8.0 + mat_h + 4.0 + LINE_H + 8.0 + BAR_H + 8.0 + 3.0 * LINE_H;
        let whole = Layout { narrow, picture: (Vec2::new(0.0, inset), view), panel: None };
        let place = |min: Vec2| {
            let x = min.x + pad;
            let mut y = min.y + pad;
            let tab_w = inner / 5.0;
            let tabs = [0, 1, 2, 3, 4].map(|k| (Vec2::new(x + k as f32 * tab_w + 1.5, y), Vec2::new(x + (k + 1) as f32 * tab_w - 1.5, y + TAB_H)));
            y += TAB_H + 8.0;
            let title = Vec2::new(x, y);
            y += TITLE_H + 6.0;
            let factors = Vec2::new(x, y);
            y += NAME_H + mat_h + 8.0;
            let check = Vec2::new(x, y);
            y += mat_h + 4.0;
            let agree = Vec2::new(x, y);
            y += LINE_H + 8.0;
            let play = (Vec2::new(min.x + width - pad - 30.0, y), Vec2::new(min.x + width - pad, y + BAR_H));
            let bar = (Vec2::new(x + 8.0, y + BAR_H * 0.5), Vec2::new(play.0.x - 16.0, y + BAR_H * 0.5));
            y += BAR_H + 8.0;
            let facts = Vec2::new(x, y);
            Panel { min, max: Vec2::new(min.x + width, min.y + height), text, tabs, title, factors, check, agree, bar, play, facts, width }
        };
        if narrow {
            // below the picture, which keeps at least the top 42 %
            let top = view.y - height;
            if top < view.y * 0.42 || view.y < 300.0 {
                return whole;
            }
            let mut panel = place(Vec2::new(0.0, top));
            panel.max = view;
            return Layout { narrow, picture: (Vec2::new(0.0, inset), Vec2::new(view.x, top)), panel: Some(panel) };
        }
        let top = inset + 12.0;
        if view.x < 600.0 || top + height + 12.0 > view.y {
            return whole;
        }
        let panel = place(Vec2::new(view.x - width - 12.0, top));
        let picture = (Vec2::new(0.0, inset), Vec2::new(panel.min.x - 8.0, view.y));
        Layout { narrow, picture, panel: Some(panel) }
    }

    /// The part of the canvas the scene's frame is fitted into: the picture's, less the
    /// heading at its top and the caption at its foot.
    fn frame_room(&self) -> (Vec2, Vec2) {
        let (min, max) = self.picture;
        let foot = if self.narrow { 30.0 } else { 44.0 };
        (min + Vec2::new(12.0, 70.0), max - Vec2::new(12.0, foot))
    }
}

/// The part of the scene the view is fitted to: what the films of every experiment
/// show lies in it. (One for all of them: the view stays put as one gives way to the
/// next.)
const FRAME_CENTRE: Vec2 = Vec2::new(0.36, 0.28);
const FRAME_HALF: Vec2 = Vec2::new(2.6, 2.2);

// ----- the scene -----------------------------------------------------------------------

/// Seconds each factor takes to act, and rests at the end of a film before the next
/// film of a tour; and the seconds a factor takes to be wound back, and a whole
/// rewind at the least.
const STEP_SECONDS: f64 = 1.4;
const REST: f32 = 1.0;
const REWIND_STEP: f64 = 0.45;
const REWIND_LEAST: f64 = 0.25;

/// The film being wound back to its start, by the clock: from where, how long it has
/// been going and how long it takes (it eases in and out), and what to do there.
#[derive(Clone, Debug)]
struct Rewind {
    from: f64,
    spent: f64,
    takes: f64,
    next: Next,
}

/// What to do once the film has been wound back to its start.
#[derive(Clone, Debug)]
struct Next {
    tab: Kind,
    /// A new matrix to glide to there (the picture is then I, whatever the matrix).
    a: Option<DMat2>,
    /// The films to play after that one.
    tour: Vec<Kind>,
}

pub struct Factor {
    glide: Glide,
    tab: Kind,
    /// How far the film has got: 0 is the plane as it is, 1 the plane as A takes it.
    phi: f64,
    playing: bool,
    /// Winding the film back to its start, and what then.
    rewind: Option<Rewind>,
    /// Play once the glide to a new matrix has arrived.
    play_when_there: bool,
    /// The films still to play, in order, and how long is left resting at the end of
    /// the last one before the next begins.
    tour: Vec<Kind>,
    rest: Option<f32>,
    /// Where the two dragged columns are asked to be (they glide there), and whether
    /// the film's knob is held.
    dragged: [DVec2; 2],
    holding: bool,
    /// Where the orange and purple arrows start, as drawn and as they should be: a new
    /// film's arrows turn to their places before it plays, instead of jumping there.
    arrows: Option<[DVec2; 2]>,
    arrows_to: Option<[DVec2; 2]>,
    /// The heading's and the caption's lines as last broken, with what for.
    heading: Option<Kept>,
    caption: Option<Kept>,
    /// The canvas last drawn on and how much of its top the application's buttons
    /// cover: they decide where the panel goes, and so where the picture is.
    canvas: Vec2,
    inset: f32,
    measured: bool,
    new_view: bool,
}

/// Lines of text as last broken, kept with what they were broken for (the text, and
/// the room across).
struct Kept {
    of: (String, u32),
    lines: Vec<String>,
}

impl Factor {
    fn new() -> Self {
        let a = rows(FIRST);
        Factor {
            glide: Glide::at(a),
            tab: Kind::Lu,
            // (opening, the first film plays)
            phi: 0.0,
            playing: false,
            rewind: None,
            play_when_there: true,
            tour: Vec::new(),
            rest: None,
            dragged: [a.x_axis, a.y_axis],
            holding: false,
            arrows: None,
            arrows_to: None,
            heading: None,
            caption: None,
            canvas: Vec2::new(1080.0, 1000.0),
            inset: 0.0,
            measured: false,
            new_view: false,
        }
    }

    /// The matrix as drawn.
    fn a(&self) -> DMat2 {
        self.glide.shown
    }

    fn split(&self) -> Split {
        split(self.tab, self.a())
    }

    /// Go to the start of the film (winding it back there, if it is not at it), and
    /// then do `next`.
    fn go(&mut self, next: Next) {
        self.playing = false;
        self.play_when_there = false;
        self.rest = None;
        if self.phi <= 0.0 {
            self.arrive(next);
        } else {
            let total = Film::of(&self.split(), self.a()).total;
            let takes = (REWIND_STEP * total * self.phi).max(REWIND_LEAST);
            self.rewind = Some(Rewind { from: self.phi, spent: 0.0, takes, next });
        }
    }

    /// At the start of the film: switch to the next one, glide to its matrix, and play
    /// it once there.
    fn arrive(&mut self, next: Next) {
        self.phi = 0.0;
        self.tab = next.tab;
        if let Some(a) = next.a {
            self.glide.to = within_limits(a);
            self.dragged = [self.glide.to.x_axis, self.glide.to.y_axis];
        }
        self.tour = next.tour;
        self.play_when_there = true;
    }

    /// Set up experiment `k` of `TRIALS`.
    fn set_up(&mut self, k: usize) {
        let trial = &TRIALS[k];
        let (first, rest) = trial.tabs.split_first().map_or((self.tab, &[][..]), |(f, r)| (*f, r));
        self.go(Next { tab: first, a: Some(rows(trial.a)), tour: rest.to_vec() });
    }

    /// Show another of the five, from its start.
    fn choose(&mut self, kind: Kind) {
        self.tour.clear();
        self.go(Next { tab: kind, a: None, tour: Vec::new() });
    }

    /// A new matrix by a button or a typed number: its film, from the start.
    fn aim(&mut self, a: DMat2) {
        self.tour.clear();
        self.go(Next { tab: self.tab, a: Some(a), tour: Vec::new() });
    }

    /// A change the visitor's hand makes now (a dragged column or number, the knob):
    /// whatever was about to happen does not.
    fn hand(&mut self) {
        (self.rewind, self.play_when_there, self.rest) = (None, false, None);
        self.tour.clear();
    }

    /// Play from where the film is (from its start, if it is at its end).
    fn play(&mut self) {
        if self.phi >= 1.0 {
            self.choose(self.tab);
        } else {
            self.hand();
            self.playing = true;
        }
    }

    fn pause(&mut self) {
        self.hand();
        self.playing = false;
    }

    fn busy(&self) -> bool {
        self.playing || self.play_when_there || self.rewind.is_some() || self.rest.is_some()
    }

    /// Move everything on: `dt` seconds of the scene's time (the film playing, the rest
    /// between two films of a tour) and `real_dt` of the clock's (a new matrix glides,
    /// and the film winds back, by the clock: they answer a click, also while the
    /// scene's time stands still).
    fn advance(&mut self, dt: f32, real_dt: f32) {
        let (dt, real) = (dt.max(0.0) as f64, real_dt.max(0.0) as f64);
        self.glide.step(real);
        if let Some(mut r) = self.rewind.take() {
            r.spent += real;
            let u = (r.spent / r.takes).min(1.0);
            self.phi = r.from * (1.0 - u * u * (3.0 - 2.0 * u));
            if u >= 1.0 {
                self.arrive(r.next);
            } else {
                self.rewind = Some(r);
            }
        }
        self.arrows_to = self.split().arrows.filter(|_| self.split().missing.is_none()).map(|ar| ar.start);
        self.arrows = match (self.arrows, self.arrows_to) {
            (Some(drawn), Some(to)) => Some(turn_towards(drawn, to, real)),
            (_, to) => to,
        };
        if self.play_when_there && self.glide.arrived() && self.arrows == self.arrows_to {
            (self.play_when_there, self.playing) = (false, true);
        }
        if self.playing {
            let total = Film::of(&self.split(), self.a()).total;
            self.phi = if total > 0.0 { (self.phi + dt / (total * STEP_SECONDS)).min(1.0) } else { 1.0 };
            if self.phi >= 1.0 {
                self.playing = false;
                if !self.tour.is_empty() {
                    self.rest = Some(REST);
                }
            }
        }
        if let Some(left) = self.rest {
            let left = left - dt as f32;
            if left > 0.0 {
                self.rest = Some(left);
            } else {
                self.rest = None;
                let mut tour = std::mem::take(&mut self.tour).into_iter();
                if let Some(tab) = tour.next() {
                    self.go(Next { tab, a: None, tour: tour.collect() });
                }
            }
        }
    }

    /// The canonical view: the picture's frame in the middle of the part of the canvas
    /// it has to itself.
    fn the_view(&self) -> View {
        let lay = Layout::of(self.canvas, self.inset);
        let (min, max) = lay.frame_room();
        let room = (max - min).max(Vec2::splat(60.0));
        let per_unit = (room.x / (2.0 * FRAME_HALF.x)).min(room.y / (2.0 * FRAME_HALF.y)).max(1e-3);
        // (screen y runs down, the scene's up)
        let off = ((min + max) * 0.5 - self.canvas * 0.5) / per_unit;
        View::Fit { center: Vec2::new(FRAME_CENTRE.x - off.x, FRAME_CENTRE.y + off.y), half: self.canvas.max(Vec2::ONE) * (0.5 / per_unit) }
    }

    /// The panel's handles and buttons: the tabs, the play button, and the film's track
    /// with its knob.
    fn panel_input(&mut self, f: &mut Frame, panel: &Panel) {
        let pointer = f.ptr.pos.map(|p| p / f.px);
        let clicked = f.clicked_canvas();
        let (b0, b1) = panel.bar;
        let knob = b0 + (b1 - b0) * self.phi.clamp(0.0, 1.0) as f32;
        // (a knob held still is held: the film does not play on under the finger)
        self.holding &= f.handles.dragging();
        if self.holding {
            self.pause();
        }
        if let Some(mut w) = f.cam.pick_ground(knob * f.px).map(|p| p.truncate()) {
            if f.drag2(3, &mut w, pal::WHITE) {
                if let Some(p) = on_screen(f, w.extend(0.0)) {
                    self.pause();
                    self.holding = true;
                    self.phi = ((p.x - b0.x) / (b1.x - b0.x).max(1.0)).clamp(0.0, 1.0) as f64;
                }
            }
        }
        let Some(p) = pointer.filter(|_| clicked) else { return };
        if let Some(k) = panel.tabs.iter().position(|r| inside_rect(p, *r)) {
            self.choose(KINDS[k]);
        } else if inside_rect(p, panel.play) {
            if self.busy() {
                self.pause();
            } else {
                self.play();
            }
        } else if inside_rect(p, (b0 - Vec2::new(8.0, 12.0), b1 + Vec2::new(8.0, 12.0))) {
            self.pause();
            self.phi = ((p.x - b0.x) / (b1.x - b0.x).max(1.0)).clamp(0.0, 1.0) as f64;
        }
    }
}

/// How fast the arrows turn to a new film's places: two thirds of the way in this many
/// seconds.
const ARROWS_TURN: f64 = 0.07;

/// Unit arrows turned `dt` seconds nearer to `to`, the short way round (there exactly,
/// once they are within a hair of it).
fn turn_towards(drawn: [DVec2; 2], to: [DVec2; 2], dt: f64) -> [DVec2; 2] {
    let k = 1.0 - (-dt.max(0.0) / ARROWS_TURN).exp();
    [0, 1].map(|i| {
        let (from, aim) = (drawn[i], to[i]);
        let apart = from.angle_to(aim);
        if !apart.is_finite() || apart.abs() < 1e-3 {
            aim
        } else {
            DVec2::from_angle(from.y.atan2(from.x) + apart * k)
        }
    })
}

fn inside_rect(p: Vec2, (min, max): (Vec2, Vec2)) -> bool {
    p.cmpge(min).all() && p.cmple(max).all()
}

/// The four numbers of a 2×2 matrix to drag or type, within ±`ENTRY_MAX`. Returns
/// (changed, typed): a number typed in plays the new matrix's film from the start, one
/// dragged moves the picture at once.
fn matrix_edit(ui: &mut egui::Ui, m: &mut DMat2) -> (bool, bool) {
    let (mut changed, mut typed) = (false, false);
    let mut a = [[m.x_axis.x, m.y_axis.x], [m.x_axis.y, m.y_axis.y]];
    egui::Grid::new("factor_a").spacing([6.0, 4.0]).show(ui, |ui| {
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

impl Scene for Factor {
    fn view(&self) -> View {
        self.the_view()
    }

    fn take_view(&mut self) -> Option<View> {
        std::mem::take(&mut self.new_view).then(|| self.the_view())
    }

    fn settings(&self, s: &mut ViewSettings) {
        s.grid_alpha = 0.08;
        s.axis_alpha = 0.35;
    }

    fn animated(&self) -> bool {
        self.busy() || !self.glide.arrived() || self.arrows != self.arrows_to
    }

    fn ui(&mut self, ui: &mut egui::Ui, _: &Gfx) {
        ui::note(ui, r"A factorisation writes one matrix as a product of simpler ones, each a map that can be seen: triangular (a shear and stretches along the axes), orthogonal (a turn, or a mirror) or diagonal (a stretch along the axes). Read a product from the right: in $A = LU$, $U$ acts first and then $L$. The tabs on the picture are five ways of taking the same $A$ apart; each plays as a film of its factors, and every film ends in the same picture, the plane as $A$ takes it. Each has a scene of its own (elimination, Gram–Schmidt, change of basis, the SVD, the polar form): this one puts them side by side.");

        ui::section(ui, "Things to try");
        for (k, trial) in TRIALS.iter().enumerate() {
            if ui::experiment(ui, trial.label, trial.look_for) {
                self.set_up(k);
            }
        }

        ui::section(ui, "Factorisation");
        let mut k = KINDS.iter().position(|x| *x == self.tab).unwrap_or(0);
        let names: Vec<&str> = KINDS.iter().map(|k| k.tab()).collect();
        if ui::choice(ui, &names, &mut k) {
            self.choose(KINDS[k]);
        }
        ui.horizontal(|ui| {
            if ui.button(if self.busy() { "❚❚ pause" } else { "▶ play" }).clicked() {
                if self.busy() {
                    self.pause();
                } else {
                    self.play();
                }
            }
            if ui.button("start again").clicked() {
                self.choose(self.tab);
            }
        });
        let mut phi = self.phi as f32;
        if ui::slider(ui, "film", &mut phi, 0.0..=1.0) {
            self.pause();
            self.phi = phi as f64;
        }
        ui::note(ui, "0 is the plane as it is, 1 the plane as $A$ takes it; each factor that moves anything has an equal share between. The knob under the matrices on the picture does the same, and the tabs above them choose the factorisation.");

        ui::section(ui, "Matrix $A$");
        let names: Vec<&str> = PRESETS.iter().map(|p| p.0).collect();
        if let Some(i) = ui::buttons(ui, &names) {
            self.aim(rows(PRESETS[i].1));
        }
        let mut a = self.glide.to;
        let (changed, typed) = matrix_edit(ui, &mut a);
        if changed {
            if typed {
                self.aim(a);
            } else {
                self.hand();
                self.glide = Glide::at(within_limits(a));
                self.dragged = [self.glide.to.x_axis, self.glide.to.y_axis];
            }
        }
        if let Some(p) = PRESETS.iter().find(|p| rows(p.1) == self.glide.to) {
            ui::note(ui, p.2);
        }
        ui::note(ui, "Or drag the pink and green dots, the columns of $A$: every film follows. Type a number to play the film of the new matrix from the start.");

        ui::section(ui, "Reading the picture");
        let mut key = vec![
            (pal::YELLOW, "yellow: the unit circle, and the unit square with a letter on it (lopsided, so that a mirror shows), carried by the factors that have acted so far; the faint outline is where $A$ takes them, and every film ends on it"),
            (pal::DIM, "dashed grey: the circle and the square at the start; the faint grid is the square grid, carried along"),
            (pal::PINK, r"pink: $e_1 = (1, 0)$ carried along; it ends on $a_1$, the first column of $A$ (the dot: drag it)"),
            (pal::GREEN, r"green: $e_2 = (0, 1)$ carried along; it ends on $a_2$, the second column (drag it too)"),
        ];
        match self.tab {
            Kind::Eigen => key.push((pal::ORANGE, r"orange and purple: the eigenvectors $v_1$ and $v_2$ of $A$, carried along (dashed: the lines they span, the eigenlines)")),
            Kind::Svd | Kind::Polar => key.push((pal::ORANGE, r"orange and purple: $v_1$ and $v_2$, the directions $A$ stretches most and least, carried along")),
            _ => {}
        }
        key.push((pal::TEXT, "dashed white line: a mirror's line, while a mirror acts"));
        ui::key(ui, &key);

        ui::section(ui, "Words");
        ui::term(ui, "Factorisation:", r"a matrix written as a product of simpler ones. Read it from the right: in $A = LU$, $U$ acts first, then $L$.");
        ui::term(ui, "Triangular:", r"zeros on one side of the diagonal. Upper triangular ($U$, $R$): zeros below it, so $(1, 0)$ stays on the $x$ axis. Lower triangular ($L$): zeros above, so $(0, 1)$ stays on the $y$ axis.");
        ui::term(ui, "Orthogonal:", r"$Q^\top Q = I$: it keeps every length and every angle. In the plane it is a turn ($\det Q = 1$) or a mirror in a line through $0$ ($\det Q = -1$).");
        ui::term(ui, "Diagonal:", "zeros off the diagonal: a stretch along the $x$ and $y$ axes by the two diagonal numbers (a negative one is a flip).");
        ui::term(ui, "Pivot:", r"a number on the diagonal of $U$. Elimination divides by the first, $a_{11}$, to clear the number under it; if $a_{11} = 0$ and that number is not, the rows are swapped first ($P$), and the first pivot is $a_{21}$.");
        ui::term(ui, "Eigenvector:", r"a $v$ with $Av = \lambda v$: a direction $A$ only stretches, by the eigenvalue $\lambda$. $V$ has two of them as its columns, $D$ their eigenvalues on its diagonal.");
        ui::term(ui, "Singular values:", r"$\sigma_1 \ge \sigma_2 \ge 0$, the most and the least $A$ stretches any direction. $v_1$ and $v_2$, at right angles, are those directions, and $Av_1 = \sigma_1 u_1$, $Av_2 = \sigma_2 u_2$.");
        ui::term(ui, r"Transpose, $V^\top$:", r"rows and columns swapped. For an orthogonal $V$ it undoes $V$: $V^\top = V^{-1}$.");
        ui::term(ui, "Positive definite:", r"for a symmetric $A$ ($A^\top = A$): $x \cdot Ax > 0$ for every $x \ne 0$, which is both eigenvalues positive.");

        ui::section(ui, "The five side by side");
        ui::term(ui, "$A = LU$:", r"$L$ lower triangular with ones on its diagonal, $U$ upper triangular. For a $2 \times 2$ matrix it exists, and is unique, if $a_{11} \ne 0$; with the rows swapped first where needed, $PA = LU$ always exists. For solving $Ax = b$: two triangular systems, each solved one unknown at a time.");
        ui::term(ui, "$A = QR$:", r"$Q$ orthogonal, $R$ upper triangular. Always exists; unique if $A$ is invertible and the diagonal of $R$ is kept positive, as here. For least squares, and for perpendicular bases (Gram–Schmidt).");
        ui::term(ui, r"$A = VDV^{-1}$:", r"eigenvectors in $V$, eigenvalues in $D$. Exists if $A$ has two independent eigenvectors, which it has whenever its two eigenvalues differ, though they may be complex; real only if the eigenvalues are real. If the two eigenvalues differ, it is unique but for their order and the lengths and signs of the eigenvectors (here: the larger first, length 1, $v_1$ pointing right (or up, if it is upright), $\det V > 0$). For powers, $A^k = VD^kV^{-1}$, and for $e^{At} = Ve^{Dt}V^{-1}$.");
        ui::term(ui, r"$A = U\Sigma V^\top$:", r"$U$, $V$ orthogonal, $\Sigma$ diagonal with $\sigma_1 \ge \sigma_2 \ge 0$. Always exists; $\Sigma$ is unique, and $U$ and $V$ are too, but for signs, if $\sigma_1 > \sigma_2 > 0$. Here $V$ is kept a turn, so if $\det A < 0$ the mirror is in $U$. For lengths ($\sigma_1$ is the most $A$ stretches), rank (how many $\sigma$ are not $0$) and the best low-rank approximation.");
        ui::term(ui, "$A = QP$:", r"$Q$ orthogonal, $P$ symmetric with no negative eigenvalue. Always exists; $P$ is unique, and $Q$ too if $A$ is invertible. For the nearest orthogonal matrix to $A$ (by the sum of squared differences of entries), which is $Q$, a turn if $\det A > 0$; and for splitting a deformation into a stretch and a turn.");

        ui::section(ui, "Numbers");
        let a = self.a();
        let s = self.split();
        ui::readout(ui, r"$\det A$", format!("${}$", num(a.determinant(), 2)));
        ui::readout(ui, "eigenvalues", eigen_text(eigen(a)));
        let d = svd(a);
        ui::readout(ui, "singular values", format!("${}$ and ${}$", num(d.sigma[0], 2), num(d.sigma[1], 2)));
        match s.rebuilt() {
            Some(p) => ui::readout(ui, &format!("${}$ against $A$, largest difference", s.product), format!("${}$", tiny(size(p - a)))),
            None => ui::readout(ui, &format!("${}$", s.product), "none, with real numbers"),
        }
        ui::see_also(ui, &[("elim", "elimination, step by step: where $L$ and $U$ come from"), ("gram", "Gram–Schmidt: where $Q$ and $R$ come from"), ("svd", "the SVD on its own"), ("polar", "the polar form on its own")]);
    }

    fn frame(&mut self, f: &mut Frame) {
        // the panel's place follows the canvas, and the picture's place follows the panel's
        let view = f.view_size();
        let lay = Layout::of(view, f.top_inset);
        let before = Layout::of(self.canvas, self.inset);
        // (a canvas squeezed to a sliver is one the controls have opened over, on a
        // phone: the picture is left as it was for when they close)
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
                // the panel is for its own buttons and knob: no moving the view from
                // there, and two quick clicks are two clicks
                f.handles.capture();
                f.handles.claim_clicks();
            }
            self.panel_input(f, panel);
        }
        self.advance(f.dt, f.real_dt);

        let labels_before = f.dl.labels.len();
        let split = self.split();
        let film = Film::of(&split, self.a());
        let plates = self.picture(f, &lay, &split, &film);
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
        if let Some(p) = &lay.panel {
            let (lo, hi) = (p.min - Vec2::splat(2.0), p.max + Vec2::splat(2.0));
            let mut k = 0;
            f.dl.labels.retain(|l| {
                k += 1;
                k <= labels_before || l.backed || !inside_rect(l.pos, (lo, hi))
            });
            self.panel(f, p, &split, &film);
        }
    }
}

/// A difference for the screen: 0 when it is 0, two figures in powers of ten when it
/// is small (as rounding leaves it), two decimals otherwise.
fn tiny(d: f64) -> String {
    if d == 0.0 || d >= 0.005 || !d.is_finite() {
        num(d, 2)
    } else {
        format!("{d:.1e}")
    }
}

/// A number for a matrix on the screen, as a formula, in five characters at most (so
/// that the brackets never move as the numbers change): two decimals, then one, then
/// none, and a power of ten beyond 9999 (drawn smaller if it is wider than its cell).
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

/// How wide a text is on the screen: one with mathematics in it exactly, words alone
/// as `text_width` judges (and that too before the fonts are in place).
fn width(f: &Frame, text: &str, size: f32) -> f32 {
    if crate::typeset::has_math(text) {
        crate::typeset::measure(f.ctx, text, size).map_or_else(|| text_width(text, size), |s| s.x)
    } else {
        text_width(text, size)
    }
}

/// A matrix of numbers on the panel, as `hud_matrix` draws it, with a number too wide
/// for its cell (a large power of ten) drawn smaller, to fit.
fn matrix(f: &mut Frame, top_left: Vec2, rows: &[Vec<(String, Color)>], size: f32, bracket: Color) -> Vec2 {
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
    let total = hud_matrix(f, top_left, &fitted, CELL, size, bracket);
    for (i, j, text, color, s) in smaller {
        let (lo, hi) = matrix_entry(top_left, CELL, size, i, j);
        f.text_px((lo + hi) * 0.5, text, s, color, Align2::CENTER_CENTER);
    }
    total
}

/// A matrix's entries for `matrix`, the columns in the colours given.
fn entries(m: DMat2, cols: [Color; 2]) -> Vec<Vec<(String, Color)>> {
    (0..2).map(|i| (0..2).map(|j| (cell(m.col(j)[i]), cols[j])).collect()).collect()
}

/// `text` broken into lines no wider than `room` pixels at `size`, kept in `kept` while
/// the text and the room are the same.
fn lines_of(kept: &mut Option<Kept>, text: &str, size: f32, room: f32) -> Vec<String> {
    let of = (text.to_string(), (room * 4.0) as u32);
    if let Some(k) = kept.as_ref().filter(|k| k.of == of) {
        return k.lines.clone();
    }
    let mut lines: Vec<String> = Vec::new();
    for word in words_in(text) {
        match lines.last_mut() {
            Some(line) if text_width(&format!("{line} {word}"), size) <= room => {
                line.push(' ');
                line.push_str(word);
            }
            _ => lines.push(word.to_string()),
        }
    }
    *kept = Some(Kept { of, lines: lines.clone() });
    lines
}

/// `parts` side by side, as long as they fit across `room` pixels at `size`, on further
/// lines when they do not; kept in `kept` while the parts and the room are the same.
fn caption_lines(kept: &mut Option<Kept>, parts: &[&str], size: f32, room: f32) -> Vec<String> {
    let gap = "      ";
    let of = (parts.join("\n"), (room * 4.0) as u32);
    if let Some(k) = kept.as_ref().filter(|k| k.of == of) {
        return k.lines.clone();
    }
    let mut lines: Vec<String> = Vec::new();
    for part in parts {
        match lines.last_mut() {
            Some(line) if text_width(&format!("{line}{gap}{part}"), size) <= room => {
                line.push_str(gap);
                line.push_str(part);
            }
            _ => lines.push(part.to_string()),
        }
    }
    *kept = Some(Kept { of, lines: lines.clone() });
    lines
}

/// The words of a text, a formula between dollars counting as one word.
fn words_in(text: &str) -> Vec<&str> {
    let mut out = Vec::new();
    let (mut start, mut inside) = (0, false);
    for (i, c) in text.char_indices() {
        match c {
            '$' => inside = !inside,
            ' ' if !inside => {
                if i > start {
                    out.push(&text[start..i]);
                }
                start = i + 1;
            }
            _ => {}
        }
    }
    if start < text.len() {
        out.push(&text[start..]);
    }
    out
}

// ----- the picture -----------------------------------------------------------------------

/// The F painted on the unit square, as three bars (corners anticlockwise). Lopsided on
/// purpose: turned, it is still an F; mirrored, it is not.
const F_BARS: [[DVec2; 4]; 3] = [
    [DVec2::new(0.18, 0.12), DVec2::new(0.36, 0.12), DVec2::new(0.36, 0.88), DVec2::new(0.18, 0.88)],
    [DVec2::new(0.36, 0.70), DVec2::new(0.82, 0.70), DVec2::new(0.82, 0.88), DVec2::new(0.36, 0.88)],
    [DVec2::new(0.36, 0.42), DVec2::new(0.66, 0.42), DVec2::new(0.66, 0.58), DVec2::new(0.36, 0.58)],
];

/// Points round the unit circle.
const CIRCLE: usize = 96;

/// The colour of the F.
fn letter_colour() -> Color {
    pal::YELLOW.mix(pal::WHITE, 0.45)
}

/// The colour of the carried grid.
fn grid_colour() -> Color {
    pal::DIM.mix(pal::BLUE, 0.25)
}

impl Factor {
    /// Everything drawn in the scene's own coordinates, and the names written beside
    /// it. Returns the boxes of the heading and of the caption, which are written over it.
    fn picture(&mut self, f: &mut Frame, lay: &Layout, split: &Split, film: &Film) -> [Option<(Vec2, Vec2)>; 2] {
        let (lo, hi) = f.cam.flat_bounds();
        // what is drawn is cut off a view's width beyond the view, however large it is
        let span = (hi - lo).as_dvec2();
        let (box_lo, box_hi) = (lo.as_dvec2() - span, hi.as_dvec2() + span);
        let a = self.a();
        let m = film.at(self.phi);
        let (acted, _) = film.place(self.phi);

        // the handles: A's columns, where the pink and green arrows end (not under the panel)
        let under_panel = |f: &Frame, p: DVec2| lay.panel.as_ref().is_some_and(|panel| on_screen(f, p.as_vec2().extend(0.0)).is_some_and(|q| inside_rect(q, (panel.min, panel.max))));
        for (k, c) in [pal::PINK, pal::GREEN].into_iter().enumerate() {
            let col = a.col(k);
            if under_panel(f, col) && !f.handles.dragging() {
                continue;
            }
            let mut w = col.as_vec2();
            if f.drag2(1 + k as u32, &mut w, c) {
                // (to the nearest multiple of SNAP: the picture glides there)
                let to = (w.as_dvec2() / SNAP).round() * SNAP;
                let to = to.clamp(DVec2::splat(-ENTRY_MAX), DVec2::splat(ENTRY_MAX));
                if to != self.dragged[k] {
                    self.dragged[k] = to;
                    self.hand();
                    self.playing = false;
                    self.glide.to = DMat2::from_cols(self.dragged[0], self.dragged[1]);
                }
            }
        }

        // the square grid, carried along (faint)
        let n = 6;
        for i in -n..=n {
            let t = i as f64;
            let (w, alpha) = if i == 0 { (1.3, 0.4) } else { (1.0, 0.2) };
            for (p, q) in [(DVec2::new(t, -n as f64), DVec2::new(t, n as f64)), (DVec2::new(-n as f64, t), DVec2::new(n as f64, t))] {
                if let Some((p, q)) = clip(m * p, m * q, box_lo, box_hi) {
                    f.line(p.as_vec2(), q.as_vec2(), w, grid_colour().a(alpha));
                }
            }
        }

        // where everything starts (dashed), and where every film ends (faint)
        let ring: Vec<DVec2> = (0..=CIRCLE).map(|i| DVec2::from_angle(i as f64 / CIRCLE as f64 * std::f64::consts::TAU)).collect();
        f.dashed_path(ring.iter().map(|p| p.as_vec2()), 1.2, pal::DIM.a(0.7));
        let unit = [DVec2::ZERO, DVec2::X, DVec2::ONE, DVec2::Y];
        for k in 0..4 {
            f.dashed(unit[k].as_vec2(), unit[(k + 1) % 4].as_vec2(), 1.2, pal::DIM.a(0.7));
        }
        let ghost: Vec<DVec2> = ring.iter().map(|p| a * *p).collect();
        for run in runs(&ghost, box_lo, box_hi) {
            f.path(run, 1.2, pal::YELLOW.a(0.3));
        }
        let corners = unit.map(|p| a * p);
        for run in runs(&[corners[0], corners[1], corners[2], corners[3], corners[0]], box_lo, box_hi) {
            f.path(run, 1.2, pal::YELLOW.a(0.3));
        }

        // the eigenlines of A, in the eigenvectors' film
        let reach = (hi - lo).length() as f64 * 2.0;
        if split.kind == Kind::Eigen {
            let lines: Vec<(DVec2, Color)> = match eigen(a) {
                Eigen::Real { lines: Lines::Two(v1, v2), .. } => vec![(v1, pal::ORANGE), (v2, pal::PURPLE)],
                Eigen::Real { lines: Lines::One(v), .. } => vec![(v, pal::ORANGE)],
                _ => Vec::new(),
            };
            for (v, c) in lines {
                if let Some((p, q)) = clip(-v * reach, v * reach, box_lo, box_hi) {
                    f.dashed(p.as_vec2(), q.as_vec2(), 1.4, c.a(0.6));
                }
            }
        }

        // a mirror's line, while the mirror acts
        let acting = film.steps.get(acted).map(|_| split.parts.len().saturating_sub(1 + acted));
        let mirror = acting.and_then(|i| split.parts.get(i)).filter(|_| split.missing.is_none()).and_then(|p| p.mirror);
        if let Some(angle) = mirror {
            let d = DVec2::from_angle(angle);
            if let Some((p, q)) = clip(-d * reach, d * reach, box_lo, box_hi) {
                f.dashed(p.as_vec2(), q.as_vec2(), 1.4, pal::TEXT.a(0.7));
            }
        }

        // the square and its F, and the circle, as the film has carried them
        let square = unit.map(|p| m * p);
        let cut = clip_polygon(&square, box_lo, box_hi);
        for i in 1..cut.len().saturating_sub(1) {
            f.dl.tris_alpha.tri(cut[0].as_vec2().extend(0.0), cut[i].as_vec2().extend(0.0), cut[i + 1].as_vec2().extend(0.0), pal::YELLOW.a(0.12));
        }
        for bar in F_BARS {
            let cut = clip_polygon(&bar.map(|p| m * p), box_lo, box_hi);
            for i in 1..cut.len().saturating_sub(1) {
                f.dl.tris_alpha.tri(cut[0].as_vec2().extend(0.0), cut[i].as_vec2().extend(0.0), cut[i + 1].as_vec2().extend(0.0), letter_colour().a(0.85));
            }
        }
        for run in runs(&[square[0], square[1], square[2], square[3], square[0]], box_lo, box_hi) {
            f.path(run, 1.5, pal::YELLOW.a(0.8));
        }
        let carried: Vec<DVec2> = ring.iter().map(|p| m * *p).collect();
        for run in runs(&carried, box_lo, box_hi) {
            f.path(run, 2.2, pal::YELLOW);
        }

        // e₁ and e₂ carried along, and the two arrows that ride with some films
        for (k, c) in [pal::PINK, pal::GREEN].into_iter().enumerate() {
            arrow_in(f, DVec2::ZERO, m.col(k), box_lo, box_hi, 2.8, c);
        }
        let arrows = split.arrows.as_ref().filter(|_| split.missing.is_none());
        let tips: Vec<(DVec2, &'static str, Color)> = arrows
            .map(|ar| {
                let names = ar.names.get(acted).or(ar.names.last()).copied().unwrap_or(["", ""]);
                // (from where they are drawn starting, which is where they start but
                // while a new film's arrows turn to their places)
                let start = self.arrows.unwrap_or(ar.start);
                [0, 1].map(|i| (m * start[i], names[i], [pal::ORANGE, pal::PURPLE][i])).to_vec()
            })
            .unwrap_or_default();
        for &(tip, _, c) in &tips {
            arrow_in(f, DVec2::ZERO, tip, box_lo, box_hi, 2.4, c);
        }

        // the names, out of the way of what is drawn
        let mut room = Room::new(f);
        if let Some(p) = &lay.panel {
            room.wall(p.min - Vec2::splat(4.0), p.max + Vec2::splat(4.0));
        }
        let heading = self.heading(f, lay, split, film);
        room.wall(heading.0, heading.1);
        let plate = self.caption(f, lay);
        if let Some(plate) = plate {
            room.wall(plate.0, plate.1);
        }
        for pair in carried.windows(2).step_by(4) {
            room.line(f, pair[0].as_vec2().extend(0.0), pair[1].as_vec2().extend(0.0));
        }
        for k in 0..4 {
            room.line(f, square[k].as_vec2().extend(0.0), square[(k + 1) % 4].as_vec2().extend(0.0));
        }
        for bar in F_BARS {
            let c = bar.map(|p| (m * p).as_vec2().extend(0.0));
            for k in 0..4 {
                room.line(f, c[k], c[(k + 1) % 4]);
            }
        }
        for k in 0..2 {
            room.line(f, Vec3::ZERO, m.col(k).as_vec2().extend(0.0));
        }
        for &(tip, _, _) in &tips {
            room.line(f, Vec3::ZERO, tip.as_vec2().extend(0.0));
        }
        let seen = |f: &Frame, p: DVec2| p.is_finite() && on_screen(f, p.as_vec2().extend(0.0)).is_some_and(|q| inside_rect(q, (lay.picture.0, lay.picture.1)));
        let away = |p: DVec2| Vec2::new(p.x as f32, -(p.y as f32));
        for (k, (name, c)) in [("$a_1$", pal::PINK), ("$a_2$", pal::GREEN)].into_iter().enumerate() {
            let col = a.col(k);
            if seen(f, col) && !under_panel(f, col) {
                room.label(f, Tag("column", k as u32), col.as_vec2().extend(0.0), name, 15.0, c, away(col), 10.0, false);
            }
        }
        for (i, &(tip, name, c)) in tips.iter().enumerate() {
            if seen(f, tip) && !name.is_empty() {
                room.label(f, Tag("arrow", i as u32), tip.as_vec2().extend(0.0), name, 14.0, c, away(tip), 8.0, false);
            }
        }
        if let Some(angle) = mirror {
            // (out along the line, at whichever end is nearer the middle of the picture)
            let d = DVec2::from_angle(angle);
            let centre = FRAME_CENTRE.as_dvec2();
            let at = if (d * 2.2 - centre).length() <= (-d * 2.2 - centre).length() { d * 2.2 } else { -d * 2.2 };
            if seen(f, at) {
                room.label(f, Tag("mirror", 0), at.as_vec2().extend(0.0), "the mirror's line", 13.0, pal::TEXT, away(d.perp()), 8.0, false);
            }
        }
        [Some(heading), plate]
    }

    /// The heading at the top of the picture: which step of the film this is, and in a
    /// line or two under it what that step does. Returns the box it takes.
    fn heading(&mut self, f: &mut Frame, lay: &Layout, split: &Split, film: &Film) -> (Vec2, Vec2) {
        let (first, second) = heading_lines(split, film, self.phi);
        let at = lay.picture.0 + Vec2::new(16.0, 12.0);
        let room = (lay.picture.1.x - lay.picture.0.x - 32.0).max(40.0);
        let size = if text_width(&first, 15.0) <= room { 15.0 } else { 13.0 };
        let small = 12.5;
        let lines = lines_of(&mut self.heading, &second, small, room);
        let color = if split.missing.is_some() { pal::RED } else { pal::TEXT };
        f.caption_px(at, first.as_str(), size, color, Align2::LEFT_TOP);
        let mut y = at.y + size + 8.0;
        let mut wide = text_width(&first, size);
        // (a canvas too small for more has the first line alone)
        let more = if f.view_size().y < 300.0 { 0 } else { 3 };
        for line in lines.iter().take(more) {
            f.caption_px(Vec2::new(at.x, y), line.as_str(), small, pal::DIM, Align2::LEFT_TOP);
            wide = wide.max(text_width(line, small));
            y += small + 6.0;
        }
        (at - Vec2::splat(6.0), Vec2::new(at.x + wide.min(room) + 12.0, y + 2.0))
    }

    /// A caption along the bottom of the picture naming what has no name of its own
    /// there. Returns the box it takes (none on a canvas too small to spare the room).
    fn caption(&mut self, f: &mut Frame, lay: &Layout) -> Option<(Vec2, Vec2)> {
        if f.view_size().y < 300.0 {
            return None;
        }
        let parts: &[&str] = if lay.narrow {
            &["dashed: the start", "faint yellow: the end, $A$"]
        } else {
            &["dashed: the circle and the square at the start", "faint yellow: where $A$ takes them, the end of every film"]
        };
        let (left, right) = (lay.picture.0.x + 18.0, lay.picture.1.x - 18.0);
        let size = 12.0;
        let lines = caption_lines(&mut self.caption, parts, size, right - left);
        let bottom = lay.picture.1.y - if lay.narrow { 8.0 } else { 16.0 };
        let count = lines.len();
        let mut wide = 0.0f32;
        for (k, line) in lines.iter().enumerate() {
            let up = (count - 1 - k) as f32 * (size + 7.0);
            f.caption_px(Vec2::new(left, bottom - up), line.as_str(), size, pal::DIM, Align2::LEFT_BOTTOM);
            wide = wide.max(text_width(line, size));
        }
        Some((Vec2::new(left - 8.0, bottom - count as f32 * (size + 7.0) - 4.0), Vec2::new(left + wide + 8.0, bottom + 6.0)))
    }

    /// The screen panel: the five tabs, the factors written out (the one acting lit),
    /// their product beside A, the film's track, and what the factorisation is for.
    fn panel(&self, f: &mut Frame, p: &Panel, split: &Split, film: &Film) {
        hud_panel(f, p.min, p.max);
        let s = p.text;
        // the tabs
        for (k, kind) in KINDS.iter().enumerate() {
            let (lo, hi) = p.tabs[k];
            let on = *kind == self.tab;
            hud_rect(f, lo, hi, if on { Color::hex(0x24304A) } else { Color::hex(0x121828) });
            hud_outline(f, lo, hi, 1.0, if on { pal::TEXT } else { pal::FAINT });
            f.text_px((lo + hi) * 0.5, kind.tab(), 13.0, if on { pal::WHITE } else { pal::DIM }, Align2::CENTER_CENTER);
        }
        // the title
        let says = if split.product == "PLU" { "elimination after a row swap" } else { split.kind.says() };
        let title = format!("{}: {}", split.equation(), says);
        let title_size = if text_width(&title, 13.5) <= p.width - 24.0 { 13.5 } else { 12.0 };
        f.text_px(p.title, title, title_size, pal::TEXT, Align2::LEFT_TOP);

        // the factors, the one acting lit
        let (acted, share) = film.place(self.phi);
        let msize = matrix_size(2, 2, CELL, s);
        let inner = p.width - 2.0 * (p.factors.x - p.min.x);
        let a = self.a();
        match &split.missing {
            Some(why) => {
                let mut kept = None;
                let mut y = p.factors.y + 4.0;
                for line in lines_of(&mut kept, why, 12.0, inner).iter().take(4) {
                    f.text_px(Vec2::new(p.factors.x, y), line.as_str(), 12.0, pal::RED, Align2::LEFT_TOP);
                    y += 17.0;
                }
            }
            None => {
                let n = split.parts.len();
                let gap = ((inner - n as f32 * msize.x) / (n as f32 - 1.0).max(1.0)).min(24.0);
                let row = n as f32 * msize.x + (n as f32 - 1.0) * gap;
                let mut x = p.factors.x + (inner - row) * 0.5;
                for (i, part) in split.parts.iter().enumerate() {
                    // the factor in step k of the film (right to left)
                    let step = n - 1 - i;
                    let state = if part.idle() {
                        0
                    } else if step < acted || acted >= film.steps.len() {
                        2
                    } else if step == acted && (share > 0.0 || self.playing) {
                        3
                    } else {
                        1
                    };
                    let (name_c, num_c) = match state {
                        3 => (pal::WHITE, pal::WHITE),
                        2 => (pal::TEXT, pal::TEXT),
                        _ => (pal::DIM, pal::DIM),
                    };
                    let top = Vec2::new(x, p.factors.y + NAME_H);
                    if state == 3 {
                        hud_rect(f, top - Vec2::new(4.0, NAME_H + 2.0), top + msize + Vec2::new(4.0, 2.0), Color::hex(0x24304A));
                    }
                    f.text_px(Vec2::new(x + msize.x * 0.5, p.factors.y), format!("${}$", part.name), 13.0, name_c, Align2::CENTER_TOP);
                    matrix(f, top, &entries(part.m, [num_c, num_c]), s, name_c);
                    x += msize.x + gap;
                }
            }
        }

        // the product beside A: the same to rounding
        let mid = p.check.y + msize.y * 0.5;
        let mut x = p.check.x;
        let columns = [pal::PINK, pal::GREEN];
        if let Some(product) = split.rebuilt() {
            let lead = format!("${} =$", split.product);
            f.text_px(Vec2::new(x, mid), lead.as_str(), s, pal::TEXT, Align2::LEFT_CENTER);
            x += text_width(&lead, s) + 8.0;
            x += matrix(f, Vec2::new(x, p.check.y), &entries(product, columns), s, pal::DIM).x + 18.0;
            let diff = size(product - a);
            let agree = if diff == 0.0 { "the same numbers exactly".to_string() } else { format!("the same but for rounding: at most ${}$ apart", tiny(diff)) };
            f.text_px(p.agree, agree, 11.5, pal::DIM, Align2::LEFT_TOP);
        } else {
            f.text_px(p.agree, "no factors to multiply", 11.5, pal::DIM, Align2::LEFT_TOP);
        }
        f.text_px(Vec2::new(x, mid), LEADS[1], s, pal::TEXT, Align2::LEFT_CENTER);
        x += text_width(LEADS[1], s) + 8.0;
        matrix(f, Vec2::new(x, p.check.y), &entries(a, columns), s, pal::DIM);

        // the film's track: a segment for each factor that acts, named under it
        let (b0, b1) = p.bar;
        hud_line(f, b0, b1, 2.0, pal::DIM);
        let shown: Vec<usize> = film.shown().collect();
        let count = shown.len().max(1) as f32;
        for (j, &k) in shown.iter().enumerate() {
            let (x0, x1) = (b0.x + (b1.x - b0.x) * j as f32 / count, b0.x + (b1.x - b0.x) * (j + 1) as f32 / count);
            hud_line(f, Vec2::new(x0, b0.y - 4.0), Vec2::new(x0, b0.y + 4.0), 1.2, pal::DIM);
            let name = match (&split.missing, split.parts.len().checked_sub(1 + k).and_then(|i| split.parts.get(i))) {
                (None, Some(part)) => format!("${}$", part.name),
                _ => "$A$".to_string(),
            };
            let c = if k < acted || acted >= film.steps.len() { pal::TEXT } else { pal::DIM };
            f.text_px(Vec2::new((x0 + x1) * 0.5, b0.y + 6.0), name, 11.0, c, Align2::CENTER_TOP);
        }
        hud_line(f, Vec2::new(b1.x, b0.y - 4.0), Vec2::new(b1.x, b0.y + 4.0), 1.2, pal::DIM);
        let knob = b0 + (b1 - b0) * self.phi.clamp(0.0, 1.0) as f32;
        hud_line(f, b0, knob, 3.0, pal::WHITE.a(0.8));
        let (lo, hi) = p.play;
        hud_rect(f, lo, hi, Color::hex(0x1A2236));
        hud_outline(f, lo, hi, 1.0, pal::DIM);
        f.text_px((lo + hi) * 0.5, if self.busy() { "❚❚" } else { "▶" }, 14.0, pal::TEXT, Align2::CENTER_CENTER);

        // when it exists, when it is unique, what it is for
        for (k, line) in split.kind.facts().iter().enumerate() {
            f.text_px(p.facts + Vec2::new(0.0, k as f32 * LINE_H), *line, 11.5, pal::DIM, Align2::LEFT_TOP);
        }
    }
}

/// The heading's two lines: which step of the film this is (or that it is at its start
/// or its end), and what that step does.
fn heading_lines(split: &Split, film: &Film, phi: f64) -> (String, String) {
    let (acted, share) = film.place(phi);
    let n = film.steps.len();
    let total = film.total as usize;
    if let Some(why) = &split.missing {
        let first = format!(r"no real ${}$: $A$ in one step", split.product);
        return (first, why.clone());
    }
    if acted >= n {
        return (format!("{}: every factor has acted", split.equation()), "the plane as $A$ takes it: the same picture at the end of every film".into());
    }
    if phi <= 0.0 && share <= 0.0 {
        return (format!("{}: the start, the plane as it is", split.equation()), "the factor on the right acts first: press play, or drag the knob under the matrices".into());
    }
    let number = film.shown().filter(|&k| k <= acted).count().max(1);
    let part = &split.parts[split.parts.len() - 1 - acted];
    (format!("step {number} of {total}: ${}$", part.name), part.does.clone())
}

// ----- drawing what may run far off --------------------------------------------------

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
    // (an end inside the box is that end exactly, not a + (b − a) rounded)
    (t0 <= t1).then(|| (if t0 <= 0.0 { a } else { a + d * t0 }, if t1 >= 1.0 { b } else { a + d * t1 }))
}

/// The stretches of a path that lie in the box `lo..hi`.
fn runs(points: &[DVec2], lo: DVec2, hi: DVec2) -> Vec<Vec<Vec2>> {
    let (mut out, mut run): (Vec<Vec<Vec2>>, Vec<Vec2>) = (Vec::new(), Vec::new());
    for pair in points.windows(2) {
        match clip(pair[0], pair[1], lo, hi) {
            Some((p, q)) => {
                if p != pair[0] && run.len() >= 2 {
                    out.push(std::mem::take(&mut run));
                }
                if run.is_empty() {
                    run.push(p.as_vec2());
                }
                run.push(q.as_vec2());
                if q != pair[1] {
                    out.push(std::mem::take(&mut run));
                }
            }
            None => {
                if run.len() >= 2 {
                    out.push(std::mem::take(&mut run));
                }
                run.clear();
            }
        }
    }
    if run.len() >= 2 {
        out.push(run);
    }
    out.retain(|r| r.len() >= 2);
    out
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

/// An arrow from `a` to `b`, the part of it in the box `lo..hi`: without its head if
/// the head is outside.
fn arrow_in(f: &mut Frame, a: DVec2, b: DVec2, lo: DVec2, hi: DVec2, width: f32, c: Color) {
    let Some((p, q)) = clip(a, b, lo, hi) else { return };
    if q == b {
        f.arrow(p.as_vec2(), q.as_vec2(), width, c);
    } else {
        f.line(p.as_vec2(), q.as_vec2(), width, c);
    }
}

/// The box a label takes on the screen, nearly: its text as wide as it is reckoned to be.
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
        size(m - n) <= tol
    }

    fn orthogonal(q: DMat2) -> bool {
        close(q.transpose() * q, DMat2::IDENTITY, 1e-12)
    }

    /// Every matrix the tests try: the experiments', the quick choices', hundreds at
    /// random, and the degenerate ones.
    fn matrices() -> Vec<DMat2> {
        let mut rng = Rng::new(7);
        let mut all: Vec<DMat2> = TRIALS.iter().map(|t| rows(t.a)).chain(PRESETS.iter().map(|p| rows(p.1))).collect();
        all.extend((0..400).map(|_| rows([0; 4].map(|_| rng.range(-3.0, 3.0) as f64))));
        // on the grid the handles snap to, where zeros and equalities are exact
        all.extend((0..400).map(|_| rows([0; 4].map(|_| (rng.range(-3.0, 3.0) as f64 / SNAP).round() * SNAP))));
        all.extend([
            DMat2::ZERO,
            DMat2::IDENTITY,
            DMat2::IDENTITY * -1.0,
            DMat2::IDENTITY * 2.5,
            rows([0.0, 0.0, 0.0, 1.0]),
            rows([0.0, 1.0, 0.0, 0.0]),
            rows([0.0, 0.0, 1.0, 0.0]),
            rows([0.0, 1.0, 1.0, 0.0]),
            rows([0.0, -2.0, 0.0, 3.0]),
            rows([1e-12, 1.0, 1.0, 1.0]),
            rows([1e-6, 1.0, 1.0, 1.0]),
            rows([0.01, 3.0, 3.0, -3.0]),
            rows([1.0, 2.0, 2.0, 4.0]),
            rows([-1.0, 0.0, 0.0, -1.0]),
            rows([0.0, -1.0, 1.0, 0.0]),
            rows([2.0, -1.0, 1.0, 2.0]),
            rows([1.0, 0.0, 0.0, -1.0]),
            rows([3.0, 3.0, 3.0, 3.0]),
            rows([-3.0, 3.0, -3.0, 3.0]),
            rows([1.0, 1e-13, 0.0, 1.0]),
        ]);
        all
    }

    /// Each factorisation multiplies back to A, its factors are of the kinds it says,
    /// and the mirror is where the panel says it is.
    #[test]
    fn every_factorisation_multiplies_back_to_a() {
        for a in matrices() {
            let scale = size(a).max(1.0);
            let det = a.determinant();
            let mirror = det < -EXACT * size(a) * size(a);
            for kind in KINDS {
                let s = split(kind, a);
                if let Some(product) = s.rebuilt() {
                    assert!(close(product, a, 1e-12 * scale * scale.max(size(s.parts.iter().fold(DMat2::IDENTITY, |p, q| p * q.m)))), "{kind:?} of {a:?}: {product:?}");
                }
                for part in &s.parts {
                    assert!(part.m.is_finite(), "{kind:?} of {a:?}: {part:?}");
                }
                let m = |name: &str| s.parts.iter().find(|p| p.name == name).map(|p| p.m).unwrap();
                match kind {
                    Kind::Lu => {
                        let (l, u) = (m("L"), m("U"));
                        assert!(l.x_axis.x == 1.0 && l.y_axis.y == 1.0 && l.y_axis.x == 0.0 && u.x_axis.y == 0.0, "{a:?}");
                        let swapped = s.product == "PLU";
                        assert_eq!(swapped, a.x_axis.x.abs() <= EXACT * size(a) && a.x_axis.y.abs() > EXACT * size(a), "{a:?}");
                    }
                    Kind::Qr => {
                        let (q, r) = (m("Q"), m("R"));
                        assert!(orthogonal(q) && r.x_axis.y == 0.0 && r.x_axis.x >= 0.0 && r.y_axis.y >= -EXACT * size(a), "{a:?}: {r:?}");
                        assert_eq!(q.determinant() < 0.0, mirror, "{a:?}");
                        // a positive diagonal whenever A is invertible
                        if det.abs() > 1e-6 * scale * scale {
                            assert!(r.x_axis.x > 0.0 && r.y_axis.y > 0.0);
                        }
                    }
                    Kind::Eigen => match (eigen(a), &s.missing) {
                        (Eigen::Complex { .. }, Some(_)) | (Eigen::Real { lines: Lines::One(_), .. }, Some(_)) => {}
                        (Eigen::Real { l1, l2, .. }, None) => {
                            let (v, d, vi) = (m("V"), m("D"), m("V^{-1}"));
                            assert!(d.x_axis.y == 0.0 && d.y_axis.x == 0.0 && d.x_axis.x == l1 && d.y_axis.y == l2 && l1 >= l2);
                            assert!(v.determinant() > 0.0 && close(v * vi, DMat2::IDENTITY, 1e-9 * size(vi).max(1.0)), "{a:?}");
                            for (k, l) in [(0, l1), (1, l2)] {
                                let x = v.col(k);
                                assert!((x.length() - 1.0).abs() < 1e-12 && (a * x - x * l).length() < 1e-9 * scale, "{a:?}: {x:?}");
                            }
                        }
                        other => panic!("{a:?}: {other:?}"),
                    },
                    Kind::Svd => {
                        let (u, sig, vt) = (m("U"), m(r"\Sigma"), m(r"V^\top"));
                        let (s1, s2) = (sig.x_axis.x, sig.y_axis.y);
                        assert!(orthogonal(u) && orthogonal(vt) && vt.determinant() > 0.0 && s1 >= s2 && s2 >= 0.0, "{a:?}");
                        assert_eq!(u.determinant() < 0.0, mirror, "{a:?}");
                        // the most and least A stretches any direction
                        for k in 0..360 {
                            let x = DVec2::from_angle((k as f64).to_radians());
                            let l = (a * x).length();
                            assert!(l <= s1 * (1.0 + 1e-12) + 1e-12 && l >= s2 * (1.0 - 1e-9) - 1e-12, "{a:?}: {l} beyond {s1}, {s2}");
                        }
                    }
                    Kind::Polar => {
                        let (q, p) = (m("Q"), m("P"));
                        assert!(orthogonal(q) && p.x_axis.y == p.y_axis.x, "{a:?}");
                        assert_eq!(q.determinant() < 0.0, mirror, "{a:?}");
                        // P's eigenvalues are A's singular values: none negative
                        let e = eigen(p);
                        assert!(matches!(e, Eigen::Real { l2, .. } if l2 >= -1e-12 * scale), "{a:?}: {e:?}");
                    }
                }
            }
        }
    }

    /// A mirror's line is the line the mirror keeps: each point on it stays put.
    #[test]
    fn a_mirror_keeps_its_line() {
        for a in matrices() {
            for kind in KINDS {
                for part in split(kind, a).parts {
                    if let Some(angle) = part.mirror {
                        let d = DVec2::from_angle(angle);
                        assert!((part.m * d - d).length() < 1e-12 && part.m.determinant() < 0.0 && orthogonal(part.m), "{kind:?} of {a:?}");
                    } else if part.m.determinant() < -1e-12 * size(a).max(1.0).powi(2) {
                        // (no other factor's determinant is negative but a triangular or
                        // diagonal one, which flips along an axis as it slides)
                        assert!(part.motion == Motion::Slide && (part.m.x_axis.y == 0.0 || part.m.y_axis.x == 0.0), "{kind:?} of {a:?}: {part:?}");
                    }
                }
            }
        }
    }

    /// Every film starts at I and ends on A; on its way each factor's part of the film
    /// is a smooth path from I to it, and a turn or a blend never passes through flat.
    #[test]
    fn every_film_goes_from_i_to_a_smoothly() {
        for a in matrices() {
            for kind in KINDS {
                let s = split(kind, a);
                let film = Film::of(&s, a);
                let scale = size(film.end).max(1.0);
                assert!(close(film.at(1.0), film.end, 0.0) && close(film.end, a, 1e-12 * scale * scale.max(4.0)), "{kind:?} of {a:?}");
                assert!(close(film.at(0.0), DMat2::IDENTITY, 0.0) || film.total == 0.0, "{kind:?} of {a:?}: {:?}", film.at(0.0));
                let mut last = film.at(0.0);
                let biggest = (0..=200).map(|i| size(film.at(i as f64 / 200.0))).fold(1.0f64, f64::max);
                for i in 1..=2000 {
                    let now = film.at(i as f64 / 2000.0);
                    assert!(now.is_finite());
                    assert!(size(now - last) <= 0.02 * biggest, "{kind:?} of {a:?}: a jump at {i}");
                    last = now;
                }
                for &(m, motion, _) in &film.steps {
                    if matches!(motion, Motion::Turn(_)) || (motion == Motion::Blend && m.determinant() > 0.0) {
                        for i in 1..100 {
                            assert!(partial(m, motion, i as f64 / 100.0).determinant() > 0.0, "{kind:?} of {a:?}");
                        }
                    }
                }
            }
        }
    }

    /// The scene as an experiment's button leaves it, run at 60 frames a second until
    /// it has played every film it plays (or for `seconds` at most), with each film's
    /// tab noted as it played.
    fn run(k: usize, seconds: f32) -> (Factor, Vec<Kind>) {
        let mut scene = Factor::new();
        scene.set_up(k);
        let mut played = Vec::new();
        for _ in 0..(seconds * 60.0) as usize {
            scene.advance(1.0 / 60.0, 1.0 / 60.0);
            if scene.playing && played.last() != Some(&scene.tab) {
                played.push(scene.tab);
            }
            if !scene.busy() && scene.glide.arrived() {
                break;
            }
        }
        (scene, played)
    }

    #[test]
    fn the_same_map_plays_five_films_that_end_alike() {
        let (scene, played) = run(0, 60.0);
        assert_eq!(played, KINDS.to_vec());
        assert!(scene.phi == 1.0 && !scene.busy() && scene.glide.to == rows(FIRST));
        let a = scene.a();
        for kind in KINDS {
            let s = split(kind, a);
            assert!(s.missing.is_none() && close(Film::of(&s, a).at(1.0), a, 1e-14));
        }
        // LU: a shear along x, then one along y
        let lu = split(Kind::Lu, a);
        assert!(close(lu.parts[0].m, rows([1.0, 0.0, 0.5, 1.0]), 0.0) && close(lu.parts[1].m, rows([1.0, 1.0, 0.0, 1.0]), 0.0));
        // QR: R keeps (1, 0) on the x axis; Q turns 26.6 degrees
        let qr = split(Kind::Qr, a);
        assert!((qr.parts[1].m * DVec2::X).y == 0.0 && qr.parts[0].motion == Motion::Turn(0.5f64.atan2(1.0)) && degrees(0.5f64.atan2(1.0)) == "26.6°");
        // eigenvectors: 2 along (1, 1), 0.5 along (-2, 1)
        let e = split(Kind::Eigen, a);
        let v = e.parts[0].m;
        assert!(e.parts[1].m == DMat2::from_diagonal(DVec2::new(2.0, 0.5)));
        assert!(v.x_axis.perp_dot(DVec2::ONE).abs() < 1e-12 && v.y_axis.perp_dot(DVec2::new(-2.0, 1.0)).abs() < 1e-12 && v.y_axis.x < 0.0);
        // SVD and polar: 2.07 and 0.48 along two lines at right angles
        let d = svd(a);
        assert!(num(d.sigma[0], 2) == "2.07" && num(d.sigma[1], 2) == "0.48" && d.v.x_axis.dot(d.v.y_axis).abs() < 1e-15);
    }

    #[test]
    fn a_turn_has_no_real_eigenvector_film() {
        let (scene, played) = run(1, 30.0);
        assert_eq!(played, vec![Kind::Eigen]);
        let a = scene.a();
        // a turn by 53.1 degrees anticlockwise, stretching nothing
        assert!(orthogonal(a) && a.determinant() > 0.0 && degrees(a.x_axis.y.atan2(a.x_axis.x)) == "53.1°");
        assert_eq!(eigen(a), Eigen::Complex { re: 0.6, im: 0.8 });
        for k in 0..36 {
            let x = DVec2::from_angle((k as f64 * 5.0).to_radians());
            assert!(x.perp_dot(a * x).abs() > 0.79, "a line stays on itself");
        }
        let e = split(Kind::Eigen, a);
        assert!(e.missing.is_some() && e.parts.is_empty());
        // shown as one step, which is A
        let film = Film::of(&e, a);
        assert!(film.steps.len() == 1 && close(film.at(1.0), a, 0.0) && scene.phi == 1.0);
        // QR: Q = A, R = I; SVD: both 1; polar: P = I, Q = A; LU: two triangular maps
        let qr = split(Kind::Qr, a);
        assert!(close(qr.parts[0].m, a, 1e-15) && close(qr.parts[1].m, DMat2::IDENTITY, 1e-15));
        let d = svd(a);
        assert!((d.sigma[0] - 1.0).abs() < 1e-15 && (d.sigma[1] - 1.0).abs() < 1e-15);
        let p = split(Kind::Polar, a);
        assert!(close(p.parts[1].m, DMat2::IDENTITY, 1e-15) && close(p.parts[0].m, a, 1e-15));
        let lu = split(Kind::Lu, a);
        assert!(lu.parts.len() == 2 && lu.parts.iter().all(|p| !p.idle()));
    }

    #[test]
    fn a_zero_in_the_corner_needs_a_swap() {
        let (scene, played) = run(2, 30.0);
        assert_eq!(played, vec![Kind::Lu]);
        let a = scene.a();
        assert!(a.x_axis.x == 0.0 && a.x_axis.y == 1.0 && a.determinant() == -1.5);
        // no LU at all: l11 u11 = a11 = 0 and l21 u11 = a21 = 1 cannot both hold with a
        // lower triangular L that is invertible, and A is
        let s = split(Kind::Lu, a);
        assert_eq!(s.product, "PLU");
        let (p, l, u) = (s.parts[0].m, s.parts[1].m, s.parts[2].m);
        assert!(p == rows([0.0, 1.0, 1.0, 0.0]) && l == DMat2::IDENTITY && u == p * a && u.x_axis.y == 0.0);
        // the film: U, then P (L = I takes no time), a mirror in y = x
        let film = Film::of(&s, a);
        assert_eq!(film.total, 2.0);
        assert!(s.parts[0].mirror == Some(std::f64::consts::FRAC_PI_4) && p.determinant() == -1.0);
    }

    #[test]
    fn a_shear_has_one_eigenline() {
        let (scene, played) = run(3, 30.0);
        assert_eq!(played, vec![Kind::Eigen]);
        let a = scene.a();
        assert!(matches!(eigen(a), Eigen::Real { l1, l2, lines: Lines::One(v) } if l1 == 1.0 && l2 == 1.0 && v.y == 0.0));
        let e = split(Kind::Eigen, a);
        assert!(e.missing.is_some());
        // upper triangular already: L = I, Q = I
        let lu = split(Kind::Lu, a);
        assert!(lu.parts[0].m == DMat2::IDENTITY && lu.parts[1].m == a);
        let qr = split(Kind::Qr, a);
        assert!(qr.parts[0].m == DMat2::IDENTITY && qr.parts[1].m == a);
        let d = svd(a);
        assert!(num(d.sigma[0], 2) == "1.62" && num(d.sigma[1], 2) == "0.62");
    }

    #[test]
    fn a_symmetric_positive_definite_matrix_makes_one_film() {
        let (scene, played) = run(4, 60.0);
        assert_eq!(played, vec![Kind::Eigen, Kind::Svd, Kind::Polar]);
        let a = scene.a();
        assert!(a == a.transpose() && matches!(eigen(a), Eigen::Real { l1, l2, .. } if l1 == 2.0 && l2 == 0.5));
        let (e, s) = (split(Kind::Eigen, a), split(Kind::Svd, a));
        let v = e.parts[0].m;
        assert!(orthogonal(v) && degrees(v.x_axis.y.atan2(v.x_axis.x)) == "45.0°" && close(e.parts[2].m, v.transpose(), 1e-15));
        assert!(v.x_axis.perp_dot(DVec2::ONE).abs() < 1e-15 && v.y_axis.perp_dot(DVec2::new(-1.0, 1.0)).abs() < 1e-15);
        // U = V and Sigma = D, and the two films are one, frame by frame
        assert!(close(s.parts[0].m, v, 1e-15) && close(s.parts[1].m, e.parts[1].m, 1e-15));
        let (fe, fs) = (Film::of(&e, a), Film::of(&s, a));
        for i in 0..=300 {
            let phi = i as f64 / 300.0;
            assert!(close(fe.at(phi), fs.at(phi), 1e-12), "at {phi}");
        }
        // polar: Q = I, P = A, in one step
        let p = split(Kind::Polar, a);
        assert!(p.parts[0].idle() && close(p.parts[1].m, a, 1e-15) && Film::of(&p, a).total == 1.0);
    }

    /// What the quick choices' notes say.
    #[test]
    fn the_quick_choices_do_what_their_notes_say() {
        let a = rows(PRESETS[0].1);
        assert!(KINDS.iter().all(|k| split(*k, a).missing.is_none()));
        // a mirror: in Q, U, Q; a negative pivot u22 = -1; lambda2 = -0.84
        let a = rows(PRESETS[1].1);
        assert!(num(a.determinant(), 2) == "−1.20");
        let lu = split(Kind::Lu, a);
        assert!((lu.parts[1].m.y_axis.y + 1.0).abs() < 1e-15 && lu.parts[1].m.x_axis.x > 0.0);
        assert!(split(Kind::Qr, a).parts[0].mirror.is_some() && split(Kind::Svd, a).parts[0].mirror.is_some() && split(Kind::Polar, a).parts[0].mirror.is_some());
        assert!(matches!(eigen(a), Eigen::Real { l1, l2, .. } if l1 > 0.0 && num(l2, 2) == "−0.84"));
        // the one factor that turns the plane over, in each
        for (kind, which) in [(Kind::Lu, "U"), (Kind::Qr, "Q"), (Kind::Eigen, "D"), (Kind::Svd, "U"), (Kind::Polar, "Q")] {
            for part in split(kind, a).parts {
                assert_eq!(part.m.determinant() < 0.0, part.name == which, "{kind:?}: {}", part.name);
            }
        }
        let r = split(Kind::Qr, a).parts[1].m;
        assert!(r.x_axis.x > 0.0 && r.y_axis.y > 0.0 && svd(a).v.determinant() > 0.0);
        // a squash: u22 = 0, r22 = 0, eigenvalues 2 and 0, sigma2 = 0; two orthogonal Q
        let a = rows(PRESETS[2].1);
        assert!(a.determinant() == 0.0 && split(Kind::Lu, a).parts[1].m.y_axis.y == 0.0);
        assert!(split(Kind::Qr, a).parts[1].m.y_axis.y.abs() < 1e-15 && svd(a).sigma[1] < 1e-15);
        assert!(matches!(eigen(a), Eigen::Real { l1, l2, .. } if l1 == 2.0 && l2 == 0.0));
        let p = split(Kind::Polar, a);
        let (q, pp) = (p.parts[0].m, p.parts[1].m);
        assert!(q.determinant() > 0.0);
        let d = svd(a);
        let mirror_q = DMat2::from_cols(d.u.x_axis, -d.u.y_axis) * d.v.transpose();
        assert!(close(mirror_q * pp, a, 1e-14) && mirror_q.determinant() < 0.0);
        // diagonal: every one is the one stretch
        let a = rows(PRESETS[3].1);
        for kind in KINDS {
            let s = split(kind, a);
            let film = Film::of(&s, a);
            assert!(film.total == 1.0, "{kind:?}");
            let busy: Vec<&Part> = s.parts.iter().filter(|p| !p.idle()).collect();
            assert!(busy.len() == 1 && busy[0].m == a, "{kind:?}");
        }
    }

    /// The claims of "the five side by side" that a test can hold.
    #[test]
    fn the_table_says_what_is_so() {
        let mut rng = Rng::new(5);
        for _ in 0..200 {
            let a = rows([0; 4].map(|_| rng.range(-3.0, 3.0) as f64));
            // the polar form's Q is the nearest orthogonal matrix to A
            let q = split(Kind::Polar, a).parts[0].m;
            let dist = |m: DMat2| (m - a).to_cols_array().iter().map(|v| v * v).sum::<f64>();
            for k in 0..720 {
                let r = turn((k as f64 * 0.5).to_radians());
                assert!(dist(r) >= dist(q) - 1e-12 && dist(r * rows([1.0, 0.0, 0.0, -1.0])) >= dist(q) - 1e-12, "{a:?}");
            }
            // A^k = V D^k V^-1, when there is one
            let e = split(Kind::Eigen, a);
            if e.missing.is_none() {
                let (v, d, vi) = (e.parts[0].m, e.parts[1].m, e.parts[2].m);
                let a3 = a * a * a;
                assert!(close(v * d * d * d * vi, a3, 1e-9 * size(a3).max(1.0) * size(vi).max(1.0)), "{a:?}");
            }
        }
        // two different eigenvalues: always two independent eigenvectors (here real)
        let a = rows([1.0, 5.0, 0.0, 1.0 + 1e-3]);
        assert!(split(Kind::Eigen, a).missing.is_none());
        // LU with a zero first column: many
        let a = rows([0.0, 2.0, 0.0, 3.0]);
        for l in [0.0, 1.0, -2.5] {
            let (ll, u) = (rows([1.0, 0.0, l, 1.0]), rows([0.0, 2.0, 0.0, 3.0 - 2.0 * l]));
            assert!(ll * u == a);
        }
    }

    /// Changing the factorisation winds the film back to its start and plays the new
    /// one from there: the picture never jumps, at the change or anywhere else.
    #[test]
    fn a_new_tab_winds_back_and_plays_without_a_jump() {
        let mut scene = Factor::new();
        // half way through a film, and at its end
        for start in [0.5, 1.0] {
            scene.tab = Kind::Svd;
            scene.phi = start;
            scene.choose(Kind::Lu);
            assert!(scene.rewind.is_some() && scene.tab == Kind::Svd);
            let mut last = Film::of(&scene.split(), scene.a()).at(scene.phi);
            let mut wound = false;
            for _ in 0..900 {
                scene.advance(1.0 / 60.0, 1.0 / 60.0);
                let now = Film::of(&scene.split(), scene.a()).at(scene.phi);
                assert!(size(now - last) < 0.1, "{now:?} after {last:?}");
                wound |= scene.tab == Kind::Lu && scene.phi < 0.02;
                last = now;
            }
            assert!(wound && scene.tab == Kind::Lu && scene.phi == 1.0 && !scene.busy());
        }
    }

    /// Through the whole tour of the five films, everything drawn moves a little in
    /// each frame: the plane, and the orange and purple arrows, which turn to a new
    /// film's places before it plays (from the eigenvectors' film to the SVD's they
    /// start along other directions).
    #[test]
    fn the_tour_moves_evenly() {
        let mut scene = Factor::new();
        scene.set_up(0);
        let tips = |s: &Factor| {
            let film = Film::of(&s.split(), s.a());
            (film.at(s.phi), s.arrows.map(|start| start.map(|v| film.at(s.phi) * v)))
        };
        let (mut plane, mut arrows) = tips(&scene);
        let mut most = 0.0f64;
        for _ in 0..60 * 60 {
            scene.advance(1.0 / 60.0, 1.0 / 60.0);
            let (p, a) = tips(&scene);
            most = most.max(size(p - plane));
            if let (Some(a), Some(b)) = (a, arrows) {
                for i in 0..2 {
                    most = most.max((a[i] - b[i]).length());
                }
            }
            (plane, arrows) = (p, a);
        }
        // (the most is in winding a film back, which goes about four times as fast as
        // playing it: 0.12 of a unit in a frame at the fastest)
        assert!(!scene.busy() && scene.tab == Kind::Polar && most < 0.125, "{most}");
    }

    /// Dragging a column puts it on the grid of `SNAP`, within the limits, and the
    /// matrix glides there.
    #[test]
    fn a_dragged_column_snaps_and_glides() {
        let mut g = Glide::at(rows(FIRST));
        g.to = rows([0.0, 1.0, 1.0, 0.5]);
        let mut last = g.shown;
        for _ in 0..200 {
            g.step(1.0 / 60.0);
            assert!(size(g.shown - last) < 0.1);
            last = g.shown;
        }
        assert!(g.arrived() && g.shown == g.to);
        assert!(within_limits(rows([9.0, -9.0, f64::NAN, 1.0])) == rows([ENTRY_MAX, -ENTRY_MAX, 0.0, 1.0]));
    }

    /// Degenerate input: every number drawn or written stays finite.
    #[test]
    fn degenerate_input_leaves_everything_finite() {
        for a in matrices() {
            for kind in KINDS {
                let s = split(kind, a);
                let film = Film::of(&s, a);
                for i in 0..=50 {
                    let m = film.at(i as f64 / 50.0);
                    assert!(m.is_finite() && m.as_mat2().is_finite(), "{kind:?} of {a:?}");
                    let (first, second) = heading_lines(&s, &film, i as f64 / 50.0);
                    for text in [first, second] {
                        assert!(!text.contains("NaN") && !text.contains("inf") && !text.contains('?'), "{text}");
                    }
                }
                for part in &s.parts {
                    let cells: Vec<String> = part.m.to_cols_array().iter().map(|v| cell(*v)).collect();
                    assert!(cells.iter().all(|c| !c.contains('?')), "{cells:?}");
                }
            }
        }
    }

    /// What each experiment shows lies in the part of the scene the view is fitted to:
    /// every frame of every film it plays.
    #[test]
    fn every_experiment_is_in_view() {
        let (lo, hi) = ((FRAME_CENTRE - FRAME_HALF).as_dvec2(), (FRAME_CENTRE + FRAME_HALF).as_dvec2());
        let inside = |p: DVec2| p.cmpge(lo - DVec2::splat(1e-9)).all() && p.cmple(hi + DVec2::splat(1e-9)).all();
        for trial in TRIALS.iter() {
            let a = rows(trial.a);
            for kind in trial.tabs {
                let film = Film::of(&split(*kind, a), a);
                for i in 0..=200 {
                    let m = film.at(i as f64 / 200.0);
                    for k in 0..48 {
                        let p = m * DVec2::from_angle(k as f64 / 48.0 * std::f64::consts::TAU);
                        assert!(inside(p), "{}, {kind:?}: {p:?}", trial.label);
                    }
                    for p in [DVec2::X, DVec2::Y, DVec2::ONE] {
                        assert!(inside(m * p), "{}, {kind:?}: {:?}", trial.label, m * p);
                    }
                }
            }
        }
    }

    /// The panel and the picture fit the canvas, from a phone to a desktop, and the
    /// lines of the panel fit across it.
    #[test]
    fn the_layout_fits_every_canvas() {
        for (w, h) in [(1080.0, 1000.0), (580.0, 700.0), (390.0, 800.0), (390.0, 560.0), (800.0, 600.0), (1600.0, 1000.0), (240.0, 180.0), (620.0, 480.0), (700.0, 700.0)] {
            for inset in [0.0, 46.0] {
                let view = Vec2::new(w, h);
                let lay = Layout::of(view, inset);
                let (min, max) = lay.picture;
                assert!(min.cmpge(Vec2::ZERO).all() && max.cmple(view).all() && (max - min).min_element() > 50.0, "{w}x{h}: {lay:?}");
                if let Some(p) = &lay.panel {
                    let within = |q: Vec2| q.cmpge(p.min - Vec2::splat(0.5)).all() && q.cmple(p.max + Vec2::splat(0.5)).all();
                    assert!(p.min.cmpge(Vec2::ZERO).all() && p.max.cmple(view + Vec2::splat(0.5)).all(), "{w}x{h}: {p:?}");
                    assert!(p.tabs.iter().all(|t| within(t.0) && within(t.1)) && within(p.play.1) && p.bar.1.x > p.bar.0.x + 120.0, "{w}x{h}: {p:?}");
                    assert!(within(p.facts + Vec2::new(0.0, 3.0 * LINE_H)), "{w}x{h}: {p:?}");
                    let room = p.width - 2.0 * (p.facts.x - p.min.x);
                    for kind in KINDS {
                        for line in kind.facts() {
                            assert!(text_width(line, 11.5) <= room, "{w}x{h}: {line}");
                        }
                    }
                }
            }
        }
    }
}

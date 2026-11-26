//! A zoo of special matrices: each kind of matrix is defined by one equation, and that
//! equation decides both what the map can do to the plane and where its eigenvalues can
//! lie in the complex plane.

use super::*;
use crate::scenes::stats::plot::{self, Plot};
use glam::{DMat2, DVec2};

pub const DEF: SceneDef = SceneDef {
    id: "special",
    title: "A zoo of special matrices",
    group: SPECIAL,
    blurb: "Symmetric, orthogonal, triangular, projections and more: what each kind does to the plane, and where its eigenvalues can lie.",
    make: |_| Box::new(Special::new()),
};

// ----- the arithmetic, in double precision -------------------------------------------

/// The largest entry either way that dragging or typing can give A or B.
const ENTRY_MAX: f64 = 3.0;
/// What counts as exactly zero, for a number the size of the entries (squared, for a
/// product of two of them).
const EXACT: f64 = 1e-9;
/// The quarter turn, anticlockwise.
const J: DMat2 = DMat2::from_cols(DVec2::new(0.0, 1.0), DVec2::new(-1.0, 0.0));
/// The swap of the two coordinates.
const SWAP: DMat2 = DMat2::from_cols(DVec2::new(0.0, 1.0), DVec2::new(1.0, 0.0));

/// A matrix by its rows, [[a, b], [c, d]], the way matrices are written.
fn rows([a, b, c, d]: [f64; 4]) -> DMat2 {
    DMat2::from_cols(DVec2::new(a, c), DVec2::new(b, d))
}

/// The entries by rows: a₁₁, a₁₂, a₂₁, a₂₂.
fn entries(m: DMat2) -> [f64; 4] {
    [m.x_axis.x, m.y_axis.x, m.x_axis.y, m.y_axis.y]
}

fn trace(m: DMat2) -> f64 {
    m.x_axis.x + m.y_axis.y
}

/// The size of the largest entry.
fn size(m: DMat2) -> f64 {
    m.to_cols_array().iter().fold(0.0f64, |s, v| s.max(v.abs()))
}

/// (tr A / 2)² − det A: positive for two real eigenvalues, negative for a complex
/// pair, zero for a double one. Worked out as ((a − d)/2)² + bc, so that a matrix with
/// a double eigenvalue gives 0 exactly.
fn quarter_disc(m: DMat2) -> f64 {
    let half = 0.5 * (m.x_axis.x - m.y_axis.y);
    half * half + m.y_axis.x * m.x_axis.y
}

/// The smallest eigenvalue of the symmetric part (M + Mᵀ)/2: the least x·Mx over the
/// unit circle.
fn least_form(m: DMat2) -> f64 {
    let [a, b, c, d] = entries(m);
    let h = 0.5 * (b + c);
    0.5 * (a + d) - (0.25 * (a - d) * (a - d) + h * h).sqrt()
}

/// The eigenvalues of a matrix and the lines they keep.
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

/// The two eigenvalues as complex numbers (re, im), the first the one drawn orange.
fn eigenvalues(e: Eigen) -> [(f64, f64); 2] {
    match e {
        Eigen::Real { l1, l2, .. } => [(l1, 0.0), (l2, 0.0)],
        Eigen::Complex { re, im } => [(re, im), (re, -im)],
    }
}

/// The eigenvalues in a formula: "$\lambda = 2.00$ and $0.50$", "$\lambda = 0.60 \pm
/// 0.80i$", in the colours of the dots that show them.
fn eigen_text(e: Eigen) -> String {
    let (o, p) = (pal::ORANGE, pal::PURPLE);
    match e {
        Eigen::Real { l1, lines: Lines::One(_) | Lines::Every, .. } => format!(r"$\lambda = {}$ twice", tint(o, &num(l1, 2))),
        Eigen::Real { l1, l2, .. } => format!(r"$\lambda = {}$ and ${}$", tint(o, &num(l1, 2)), tint(p, &num(l2, 2))),
        Eigen::Complex { re, im } if num(re, 2) == "0.00" => format!(r"$\lambda = \pm {}i$", num(im, 2)),
        Eigen::Complex { re, im } => format!(r"$\lambda = {} \pm {}i$", num(re, 2), num(im, 2)),
    }
}

/// The matrix with every entry within ±`ENTRY_MAX` (and 0 for one that is not a
/// number).
fn within_limits(m: DMat2) -> DMat2 {
    DMat2::from_cols_array(&m.to_cols_array().map(|v| if v.is_finite() { v.max(-ENTRY_MAX).min(ENTRY_MAX) } else { 0.0 }))
}

/// The inverse, if there is one: none when the determinant is nothing beside the
/// entries' size.
fn inverse(a: DMat2) -> Option<DMat2> {
    let s = size(a);
    let det = a.determinant();
    (s > 0.0 && det.abs() > EXACT * s * s).then(|| a.inverse()).filter(|m| m.is_finite())
}

// ----- the kinds ---------------------------------------------------------------------

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Kind {
    Diagonal,
    Triangular,
    Symmetric,
    PosDef,
    Skew,
    Orthogonal,
    Normal,
    Projection,
    Nilpotent,
    Shear,
    Markov,
}

/// The kinds, in the order they are offered.
const KINDS: [Kind; 11] = [
    Kind::Diagonal,
    Kind::Triangular,
    Kind::Symmetric,
    Kind::PosDef,
    Kind::Skew,
    Kind::Orthogonal,
    Kind::Normal,
    Kind::Projection,
    Kind::Nilpotent,
    Kind::Shear,
    Kind::Markov,
];

/// Where a kind's eigenvalues can lie, in the complex plane.
#[derive(Clone, Copy, Debug, PartialEq)]
enum Region {
    /// The real line.
    Real,
    /// The real line right of 0 (0 itself left out).
    Positive,
    /// The imaginary line.
    Imaginary,
    /// The unit circle.
    Circle,
    /// Only these points of the real line.
    Points(&'static [f64]),
    /// 1, and the real line from −1 to 1 for the other.
    Shares,
    /// Anywhere at all.
    Anywhere,
}

impl Region {
    /// Its name, as the chart writes it.
    fn name(self) -> &'static str {
        match self {
            Region::Real => "the real line",
            Region::Positive => r"the real line right of $0$",
            Region::Imaginary => "the imaginary line",
            Region::Circle => "the unit circle",
            Region::Points([0.0, 1.0]) => r"$0$ and $1$ only",
            Region::Points([0.0]) => r"$0$ only",
            Region::Points(_) => r"$1$ only",
            Region::Shares => r"$1$, and the other in $[-1, 1]$",
            Region::Anywhere => "anywhere",
        }
    }

    /// Whether the eigenvalue re + im i lies in it, to within `tol`.
    fn holds(self, (re, im): (f64, f64), tol: f64) -> bool {
        match self {
            Region::Real => im.abs() <= tol,
            Region::Positive => im.abs() <= tol && re > 0.0,
            Region::Imaginary => re.abs() <= tol,
            Region::Circle => (re.hypot(im) - 1.0).abs() <= tol,
            Region::Points(points) => im.abs() <= tol && points.iter().any(|p| (re - p).abs() <= tol),
            Region::Shares => im.abs() <= tol && re >= -1.0 - tol && re <= 1.0 + tol,
            Region::Anywhere => true,
        }
    }
}

impl Kind {
    fn name(self) -> &'static str {
        match self {
            Kind::Diagonal => "diagonal",
            Kind::Triangular => "upper triangular",
            Kind::Symmetric => "symmetric",
            Kind::PosDef => "positive definite",
            Kind::Skew => "skew-symmetric",
            Kind::Orthogonal => "orthogonal",
            Kind::Normal => "normal",
            Kind::Projection => "projection",
            Kind::Nilpotent => "nilpotent",
            Kind::Shear => "shear",
            Kind::Markov => "Markov",
        }
    }

    /// Its name on its button.
    fn chip(self) -> &'static str {
        match self {
            Kind::Triangular => "triangular",
            k => k.name(),
        }
    }

    /// "a symmetric matrix", "an orthogonal matrix", "a projection".
    fn a_name(self) -> String {
        let article = if matches!(self, Kind::Orthogonal | Kind::Skew | Kind::Triangular) { "an" } else { "a" };
        match self {
            Kind::Projection | Kind::Shear => format!("a {}", self.name()),
            Kind::Markov => "a Markov matrix".into(),
            _ => format!("{article} {} matrix", self.name()),
        }
    }

    fn region(self) -> Region {
        match self {
            Kind::Diagonal | Kind::Triangular | Kind::Symmetric => Region::Real,
            Kind::PosDef => Region::Positive,
            Kind::Skew => Region::Imaginary,
            Kind::Orthogonal => Region::Circle,
            Kind::Normal => Region::Anywhere,
            Kind::Projection => Region::Points(&[0.0, 1.0]),
            Kind::Nilpotent => Region::Points(&[0.0]),
            Kind::Shear => Region::Points(&[1.0]),
            Kind::Markov => Region::Shares,
        }
    }

    /// The defining equation, for the matrix written `n` (`A`, `AB`, `A + B`, ...).
    fn equation(self, n: &str) -> String {
        let p = paren(n);
        let (e12, e21) = (entry(n, 12), entry(n, 21));
        match self {
            Kind::Diagonal => format!("${e12} = {e21} = 0$"),
            Kind::Triangular => format!("${e21} = 0$"),
            Kind::Symmetric => format!(r"${p}^\top = {n}$"),
            Kind::PosDef => format!(r"${p}^\top = {n}$, $x \cdot {p}x > 0$"),
            Kind::Skew => format!(r"${p}^\top = -{p}$"),
            Kind::Orthogonal => format!(r"${p}^\top {p} = I$"),
            Kind::Normal => format!(r"${p}{p}^\top = {p}^\top {p}$"),
            Kind::Projection => format!("${p}^2 = {n}$"),
            Kind::Nilpotent => format!("${p}^2 = 0$"),
            Kind::Shear => format!(r"$({n} - I)^2 = 0$, ${n} \ne I$"),
            Kind::Markov => r"entries $\ge 0$, columns add to $1$".into(),
        }
    }

    /// What its eigenvalues and eigenlines are, in a few words.
    fn eigen_words(self) -> &'static str {
        match self {
            Kind::Diagonal => "its eigenvalues are its diagonal entries, and the axes are eigenlines",
            Kind::Triangular => r"its eigenvalues are its diagonal entries, $a_{11}$ and $a_{22}$",
            Kind::Symmetric => "its eigenvalues are real, and its eigenlines at right angles",
            Kind::PosDef => "its eigenvalues are positive, and its eigenlines at right angles",
            Kind::Skew => r"$x \cdot Ax = 0$ for every $x$; its eigenvalues lie on the imaginary line",
            Kind::Orthogonal => "it keeps every length and angle; its eigenvalues lie on the unit circle",
            Kind::Normal => "in the plane: symmetric, or a turn and a scaling; eigenvectors at right angles (complex ones, for a turn)",
            Kind::Projection => r"applied twice it does nothing more; its eigenvalues are $0$ and $1$",
            Kind::Nilpotent => r"its only eigenvalue is $0$; unless $A = 0$, it has a single eigenline",
            Kind::Shear => r"its only eigenvalue is $1$, with a single eigenline; it keeps areas",
            Kind::Markov => r"one eigenvalue is exactly $1$; the other lies between $-1$ and $1$",
        }
    }

    /// Whether a sum, a product and an inverse of the kind are of the kind, in a few
    /// words each (what `closure` says in full).
    fn closure_short(self) -> [&'static str; 3] {
        match self {
            Kind::Diagonal | Kind::Triangular => ["always", "always", "where it exists"],
            Kind::Symmetric => ["always", r"exactly when $AB = BA$", "where it exists"],
            Kind::PosDef => ["always", r"exactly when $AB = BA$", "always"],
            Kind::Skew => ["always", r"only if it is $0$", "where it exists"],
            Kind::Orthogonal => ["not always", "always", "always"],
            Kind::Normal => ["not always", "not always", "where it exists"],
            Kind::Projection => [r"exactly when $AB = BA = 0$", r"if $AB = BA$; not always", r"only $I$ has one"],
            Kind::Nilpotent => ["not always", "not always", "never"],
            Kind::Shear => ["never", "along one line: a shear or $I$", "always"],
            Kind::Markov => ["never (an average is)", "always", r"only for $I$ and the swap"],
        }
    }

    /// What stays true of the kind under a sum, a product or an inverse.
    fn closure(self, op: Op) -> &'static str {
        use Kind::*;
        match (op, self) {
            (Op::Alone, _) => "",
            (Op::Sum, Diagonal) => "A sum of diagonal matrices is diagonal.",
            (Op::Sum, Triangular) => "A sum of upper triangular matrices is upper triangular.",
            (Op::Sum, Symmetric) => r"A sum of symmetric matrices is symmetric: $(A + B)^\top = A^\top + B^\top$.",
            (Op::Sum, PosDef) => r"A sum of positive definite matrices is positive definite: $x \cdot (A + B)x = x \cdot Ax + x \cdot Bx$.",
            (Op::Sum, Skew) => "A sum of skew-symmetric matrices is skew-symmetric.",
            (Op::Sum, Orthogonal) => r"A sum of orthogonal matrices need not be orthogonal: $I + I = 2I$ doubles every length.",
            (Op::Sum, Normal) => "A sum of normal matrices need not be normal.",
            (Op::Sum, Projection) => r"$A + B$ is a projection only if $AB = BA = 0$.",
            (Op::Sum, Nilpotent) => "A sum of nilpotent matrices need not be nilpotent.",
            (Op::Sum, Shear) => r"A sum of two shears is never a shear: its diagonal adds up to $4$, a shear's to $2$.",
            (Op::Sum, Markov) => r"A sum of Markov matrices is not one: its columns add to $2$. Their average is one.",
            (Op::Product, Diagonal) => "A product of diagonal matrices is diagonal, and the same in either order.",
            (Op::Product, Triangular) => "A product of upper triangular matrices is upper triangular.",
            (Op::Product, Symmetric) => r"A product of symmetric matrices is symmetric exactly when they commute: $(AB)^\top = BA$.",
            (Op::Product, PosDef) => "A product of positive definite matrices is positive definite exactly when they commute (its eigenvalues are positive either way).",
            (Op::Product, Skew) => r"In the plane a product of two skew-symmetric matrices is $(aJ)(bJ) = -abI$: symmetric, and skew only if it is $0$.",
            (Op::Product, Orthogonal) => r"A product of orthogonal matrices is always orthogonal: $(AB)^\top AB = B^\top A^\top A B = I$.",
            (Op::Product, Normal) => "A product of normal matrices need not be normal.",
            (Op::Product, Projection) => "A product of projections is a projection if they commute, and need not be otherwise.",
            (Op::Product, Nilpotent) => "A product of nilpotent matrices need not be nilpotent.",
            (Op::Product, Shear) => r"A product of two shears along the same line is a shear (or $I$); along different lines it is not.",
            (Op::Product, Markov) => "A product of Markov matrices is a Markov matrix.",
            (Op::Inverse, Diagonal) => r"The inverse of a diagonal matrix, where there is one, is diagonal: $1/a_{11}$ and $1/a_{22}$.",
            (Op::Inverse, Triangular) => "The inverse of an upper triangular matrix, where there is one, is upper triangular.",
            (Op::Inverse, Symmetric) => "The inverse of a symmetric matrix, where there is one, is symmetric.",
            (Op::Inverse, PosDef) => r"The inverse of a positive definite matrix is positive definite: its eigenvalues are $1/\lambda$.",
            (Op::Inverse, Skew) => r"In the plane the inverse of $wJ$ ($w \ne 0$) is $-J/w$: skew-symmetric too.",
            (Op::Inverse, Orthogonal) => r"The inverse of an orthogonal matrix is its transpose, $A^{-1} = A^\top$, and orthogonal too.",
            (Op::Inverse, Normal) => "The inverse of a normal matrix, where there is one, is normal.",
            (Op::Inverse, Projection) => r"A projection has no inverse unless it is $I$: it sends a whole line to $0$.",
            (Op::Inverse, Nilpotent) => r"A nilpotent matrix never has an inverse: it sends a whole line to $0$.",
            (Op::Inverse, Shear) => r"The inverse of a shear slides back along the same line: $(I + N)^{-1} = I - N$.",
            (Op::Inverse, Markov) => r"The inverse, where there is one, has columns adding to $1$ but a negative entry, unless $A$ is $I$ or the swap.",
        }
    }

    /// The matrix a kind starts with.
    fn default_a(self) -> DMat2 {
        rows(match self {
            Kind::Diagonal => [1.5, 0.0, 0.0, 0.6],
            Kind::Triangular => [1.2, 0.8, 0.0, 0.6],
            Kind::Symmetric => [1.7, 0.6, 0.6, 0.8],
            Kind::PosDef => [1.3, 0.4, 0.4, 0.7],
            Kind::Skew => [0.0, -1.2, 1.2, 0.0],
            Kind::Orthogonal => [0.6, -0.8, 0.8, 0.6],
            Kind::Normal => [0.9, -0.6, 0.6, 0.9],
            Kind::Projection => [0.8, 0.4, 0.4, 0.2],
            Kind::Nilpotent => [0.8, -1.6, 0.4, -0.8],
            Kind::Shear => [1.0, 1.0, 0.0, 1.0],
            Kind::Markov => [0.8, 0.3, 0.2, 0.7],
        })
    }

    /// The second matrix of the kind it starts with, for sums and products.
    fn default_b(self) -> DMat2 {
        rows(match self {
            Kind::Diagonal => [-0.8, 0.0, 0.0, 1.4],
            Kind::Triangular => [0.7, -0.6, 0.0, 1.3],
            Kind::Symmetric => [1.5, -0.5, -0.5, -0.5],
            Kind::PosDef => [0.9, -0.5, -0.5, 1.1],
            Kind::Skew => [0.0, 0.8, -0.8, 0.0],
            Kind::Orthogonal => [0.6, 0.8, 0.8, -0.6],
            Kind::Normal => [1.2, 0.5, 0.5, -0.4],
            Kind::Projection => [1.0, -1.0, 0.0, 0.0],
            Kind::Nilpotent => [0.0, 1.0, 0.0, 0.0],
            Kind::Shear => [1.0, 0.0, 0.8, 1.0],
            Kind::Markov => [0.5, 0.9, 0.5, 0.1],
        })
    }
}

/// The matrix written `n`, in brackets if it is more than a letter: `A`, `(AB)`.
fn paren(n: &str) -> String {
    if n == "A" {
        n.to_string()
    } else {
        format!("({n})")
    }
}

/// The entry ij of the matrix written `n`: `a_{12}`, `(AB)_{12}`.
fn entry(n: &str, ij: u32) -> String {
    if n == "A" {
        format!("a_{{{ij}}}")
    } else {
        format!("({n})_{{{ij}}}")
    }
}

/// How far `m` is from being of the kind (0 when it is, to rounding), and whether it
/// is.
fn test(kind: Kind, m: DMat2) -> (f64, bool) {
    let s = size(m).max(1.0);
    let (lin, quad) = (EXACT * s, EXACT * s * s);
    let [a, b, c, d] = entries(m);
    let id = DMat2::IDENTITY;
    let t = m.transpose();
    match kind {
        Kind::Diagonal => {
            let off = b.abs().max(c.abs());
            (off, off <= lin)
        }
        Kind::Triangular => (c.abs(), c.abs() <= lin),
        Kind::Symmetric => ((b - c).abs(), (b - c).abs() <= lin),
        Kind::PosDef => {
            let (asym, low) = ((b - c).abs(), least_form(m));
            (asym.max(-low).max(0.0), asym <= lin && low > lin)
        }
        Kind::Skew => {
            let off = a.abs().max(d.abs()).max((b + c).abs());
            (off, off <= lin)
        }
        Kind::Orthogonal => {
            let off = size(t * m - id);
            (off, off <= quad)
        }
        Kind::Normal => {
            let off = size(m * t - t * m);
            (off, off <= quad)
        }
        Kind::Projection => {
            let off = size(m * m - m);
            (off, off <= quad)
        }
        Kind::Nilpotent => {
            let off = size(m * m);
            (off, off <= quad)
        }
        Kind::Shear => {
            let n = m - id;
            let off = size(n * n);
            (off, off <= quad && size(n) > EXACT)
        }
        Kind::Markov => {
            let off = (a + c - 1.0).abs().max((b + d - 1.0).abs()).max(-a.min(b).min(c).min(d)).max(0.0);
            (off, off <= lin)
        }
    }
}

fn is(kind: Kind, m: DMat2) -> bool {
    test(kind, m).1
}

/// The sides of the kind's equation that are worked out from `m` (named `n`), to be
/// written out as matrices beside it.
fn sides(kind: Kind, m: DMat2, n: &str) -> Vec<(String, DMat2)> {
    let p = paren(n);
    let t = m.transpose();
    match kind {
        Kind::Symmetric | Kind::PosDef => vec![(format!(r"${p}^\top$"), t)],
        Kind::Skew => vec![(format!(r"${p}^\top$"), t), (format!("$-{p}$"), -m)],
        Kind::Orthogonal => vec![(format!(r"${p}^\top {p}$"), t * m)],
        Kind::Normal => vec![(format!(r"${p}{p}^\top$"), m * t), (format!(r"${p}^\top {p}$"), t * m)],
        Kind::Projection | Kind::Nilpotent => vec![(format!("${p}^2$"), m * m)],
        Kind::Shear => {
            let k = m - DMat2::IDENTITY;
            vec![(format!("$({n} - I)^2$"), k * k)]
        }
        Kind::Diagonal | Kind::Triangular | Kind::Markov => Vec::new(),
    }
}

/// A difference for the screen: two decimals, or two figures in powers of ten when it
/// is smaller than that.
fn small(d: f64) -> String {
    if d == 0.0 || d >= 0.005 || !d.is_finite() {
        num(d, 2)
    } else {
        format!("{d:.1e}").replace('-', "−")
    }
}

/// The verdict on the kind's equation for `m` (named `n`): its first line, whether it
/// holds, and a second line of what more there is to say.
fn verdict(kind: Kind, m: DMat2, n: &str) -> (String, bool, Option<String>) {
    let (off, holds) = test(kind, m);
    let p = paren(n);
    let ok = r" $\checkmark$";
    let by = format!(": they differ by up to ${}$", small(off));
    let line = |eq: String, ne: String| if holds { format!("{eq}{ok}") } else { format!("{ne}{by}") };
    let [a, b, c, d] = entries(m);
    let first = match kind {
        Kind::Diagonal => {
            let (e12, e21) = (entry(n, 12), entry(n, 21));
            if holds {
                format!("${e12} = {e21} = 0${ok}")
            } else {
                format!("${e12} = {}$, ${e21} = {}$: not both $0$", num(b, 2), num(c, 2))
            }
        }
        Kind::Triangular => {
            let e21 = entry(n, 21);
            if holds {
                format!("${e21} = 0${ok}")
            } else {
                format!("${e21} = {}$, not $0$", num(c, 2))
            }
        }
        Kind::Symmetric | Kind::PosDef => {
            let off = (b - c).abs();
            if off <= EXACT * size(m).max(1.0) {
                format!(r"${p}^\top = {n}${ok}")
            } else {
                format!(r"${p}^\top \ne {n}$: they differ by up to ${}$", small(off))
            }
        }
        Kind::Skew => line(format!(r"${p}^\top = -{p}$"), format!(r"${p}^\top \ne -{p}$")),
        Kind::Orthogonal => line(format!(r"${p}^\top {p} = I$"), format!(r"${p}^\top {p} \ne I$")),
        Kind::Normal => line(format!(r"${p}{p}^\top = {p}^\top {p}$"), format!(r"${p}{p}^\top \ne {p}^\top {p}$")),
        Kind::Projection => line(format!("${p}^2 = {n}$"), format!(r"${p}^2 \ne {n}$")),
        Kind::Nilpotent => line(format!("${p}^2 = 0$"), format!(r"${p}^2 \ne 0$")),
        Kind::Shear => {
            if holds {
                format!(r"$({n} - I)^2 = 0$, ${n} \ne I${ok}")
            } else if size(m - DMat2::IDENTITY) <= EXACT {
                format!("${n} = I$: nothing moves")
            } else {
                format!(r"$({n} - I)^2 \ne 0$: an entry is ${}$", small(off))
            }
        }
        Kind::Markov => {
            let (s1, s2, low) = (a + c, b + d, a.min(b).min(c).min(d));
            let sums = if (s1 - 1.0).abs().max((s2 - 1.0).abs()) <= EXACT * size(m).max(1.0) { r"columns add to $1$".to_string() } else { format!("columns add to ${}$ and ${}$", num(s1, 2), num(s2, 2)) };
            let signs = if low >= 0.0 { r"no entry below $0$".to_string() } else { format!("an entry is ${}$", num(low, 2)) };
            format!("{sums}, {signs}{}", if holds { ok } else { "" })
        }
    };
    let second = match kind {
        Kind::PosDef => {
            let low = least_form(m);
            let rel = if low > EXACT * size(m).max(1.0) { r"> 0$ $\checkmark" } else { r"\le 0" };
            Some(format!(r"least $x \cdot {p}x$ with $|x| = 1$: ${} {rel}$", num(low, 2)))
        }
        Kind::Orthogonal if holds => Some(if m.determinant() > 0.0 { format!(r"a rotation: $\det {p} = 1$") } else { format!(r"a reflection: $\det {p} = -1$") }),
        Kind::Nilpotent if holds => Some(if size(m) <= EXACT { format!("here ${n} = 0$ itself") } else { format!(r"and ${n} \ne 0$") }),
        Kind::Normal if holds => Some(if (b - c).abs() <= EXACT * size(m).max(1.0) { "here: symmetric".to_string() } else { r"here: a turn and a scaling, $aI + bJ$".to_string() }),
        _ => None,
    };
    (first, holds, second)
}

/// What the matrix does to the plane, in words, with its numbers.
fn does(kind: Kind, m: DMat2) -> String {
    let [a, b, c, d] = entries(m);
    let degrees = |r: f64| {
        let v = r.to_degrees();
        format!("{:.1}°", if v.abs() < 0.05 { 0.0 } else { v })
    };
    let e = eigen(m);
    match kind {
        Kind::Diagonal => format!("multiplies the $x$ axis by ${}$ and the $y$ axis by ${}$", num(a, 2), num(d, 2)),
        Kind::Triangular => format!("keeps the $x$ axis on itself, multiplying it by ${}$", num(a, 2)),
        Kind::Symmetric | Kind::PosDef => match e {
            Eigen::Real { l1, l2, lines: Lines::Two(..) } => {
                let both = if kind == Kind::PosDef { ", both positive" } else { "" };
                format!(r"stretches along two perpendicular lines, by $\lambda = {}$ and ${}${both}", num(l1, 2), num(l2, 2))
            }
            Eigen::Real { l1, .. } => format!("multiplies every vector by ${}$", num(l1, 2)),
            Eigen::Complex { .. } => "has complex eigenvalues".into(),
        },
        Kind::Skew => {
            let w = c;
            if w.abs() <= EXACT {
                "$A = 0$: it sends everything to $0$".into()
            } else {
                let way = if w > 0.0 { "anticlockwise" } else { "clockwise" };
                format!("turns every arrow a quarter turn {way} and multiplies its length by ${}$", num(w.abs(), 2))
            }
        }
        Kind::Orthogonal => {
            if m.determinant() > 0.0 {
                let angle = c.atan2(a);
                if angle.to_degrees().abs() < 0.05 {
                    "the identity: nothing moves".into()
                } else {
                    format!("turns the plane by {}, keeping every length", degrees(angle))
                }
            } else {
                let mut mirror = 0.5 * c.atan2(a);
                if mirror < 0.0 {
                    mirror += std::f64::consts::PI;
                }
                format!("reflects the plane in the line at {} to the $x$ axis", degrees(mirror))
            }
        }
        Kind::Normal => {
            if (b - c).abs() <= EXACT * size(m).max(1.0) {
                match e {
                    Eigen::Real { l1, l2, lines: Lines::Two(..) } => format!(r"symmetric here: stretches along two perpendicular lines, by $\lambda = {}$ and ${}$", num(l1, 2), num(l2, 2)),
                    _ => format!("multiplies every vector by ${}$", num(a, 2)),
                }
            } else {
                let (r, angle) = (a.hypot(c), c.atan2(a));
                format!("turns the plane by {} and multiplies every length by ${}$", degrees(angle), num(r, 2))
            }
        }
        Kind::Projection => "drops every point along the purple line onto the orange one".into(),
        Kind::Nilpotent if size(m) <= EXACT => "$A = 0$: it sends everything to $0$".into(),
        Kind::Nilpotent => "squashes the plane onto the orange line, and sends that line to $0$".into(),
        Kind::Shear => "keeps the orange line point for point, and slides every other point along it".into(),
        Kind::Markov => r"maps pairs of shares that add to $1$ (the segment from $(1, 0)$ to $(0, 1)$) to pairs that add to $1$".into(),
    }
}

// ----- keeping a matrix of its kind ----------------------------------------------------

/// A after its column `j` is dragged to `p`: that column goes as near `p` as the kind
/// allows, and the rest changes only as much as it must for A to stay of its kind. Every
/// entry stays within ±`ENTRY_MAX`.
fn fit(kind: Kind, a: DMat2, j: usize, p: DVec2) -> DMat2 {
    let p = if p.is_finite() { p.clamp(DVec2::splat(-ENTRY_MAX), DVec2::splat(ENTRY_MAX)) } else { a.col(j.min(1)) };
    let a = within_limits(a);
    match (kind, j) {
        // (the ones that are not the same with the two coordinates swapped, and those
        // too simple to need it)
        (Kind::Diagonal, 0) => DMat2::from_cols(DVec2::new(p.x, 0.0), DVec2::new(0.0, a.y_axis.y)),
        (Kind::Diagonal, _) => DMat2::from_cols(DVec2::new(a.x_axis.x, 0.0), DVec2::new(0.0, p.y)),
        (Kind::Triangular, 0) => DMat2::from_cols(DVec2::new(p.x, 0.0), a.y_axis),
        (Kind::Triangular, _) => DMat2::from_cols(DVec2::new(a.x_axis.x, 0.0), p),
        (Kind::Markov, _) => {
            // the nearest point of the segment from (1, 0) to (0, 1)
            let t = (0.5 * (p.x - p.y + 1.0)).clamp(0.0, 1.0);
            let mut m = a;
            *m.col_mut(j.min(1)) = DVec2::new(t, 1.0 - t);
            // (the other column too, should A have come from elsewhere)
            let k = 1 - j.min(1);
            let t = (0.5 * (m.col(k).x - m.col(k).y + 1.0)).clamp(0.0, 1.0);
            *m.col_mut(k) = DVec2::new(t, 1.0 - t);
            m
        }
        (_, 0) => fit_first(kind, a, p),
        // the second column dragged is the first of the matrix with its coordinates
        // swapped (every other kind is still itself after that swap)
        _ => SWAP * fit_first(kind, SWAP * a * SWAP, DVec2::new(p.y, p.x)) * SWAP,
    }
}

/// The smallest positive eigenvalue a positive definite matrix may be dragged to.
const LEAST_POSITIVE: f64 = 0.05;

/// `fit` for the first column.
fn fit_first(kind: Kind, a: DMat2, p: DVec2) -> DMat2 {
    let l = ENTRY_MAX;
    // the sign a coordinate keeps when it must stay off 0: its own, or else the one it had
    let side = |v: f64, was: f64| if v != 0.0 { v.signum() } else if was < 0.0 { -1.0 } else { 1.0 };
    match kind {
        Kind::Symmetric => rows([p.x, p.y, p.y, a.y_axis.y]),
        Kind::Skew => {
            let w = p.y;
            rows([0.0, -w, w, 0.0])
        }
        Kind::Orthogonal => {
            let u = if p.length() > 1e-6 { p.normalize() } else { a.x_axis.try_normalize().unwrap_or(DVec2::X) };
            let s = if a.determinant() < 0.0 { -1.0 } else { 1.0 };
            DMat2::from_cols(u, J * u * s)
        }
        Kind::Normal => {
            // the nearer of the two kinds a normal matrix of the plane can be: symmetric,
            // or a turn and a scaling (the column dragged in both)
            let dragged = DMat2::from_cols(p, a.y_axis);
            let sym = rows([p.x, p.y, p.y, a.y_axis.y]);
            let turn = DMat2::from_cols(p, J * p);
            let apart = |m: DMat2| (m - dragged).to_cols_array().iter().map(|v| v * v).sum::<f64>();
            if apart(turn) < apart(sym) {
                turn
            } else {
                sym
            }
        }
        Kind::PosDef => {
            let mu = LEAST_POSITIVE;
            let a11 = p.x.max(2.0 * mu).min(l);
            // (a₁₁ − μ)(a₂₂ − μ) ≥ a₂₁² keeps the smaller eigenvalue at μ or above, with
            // a₂₂ no larger than the limit
            let reach = ((a11 - mu) * (l - mu)).sqrt();
            let a21 = p.y.max(-reach).min(reach);
            let a22 = a.y_axis.y.max(mu + a21 * a21 / (a11 - mu)).min(l);
            rows([a11, a21, a21, a22])
        }
        Kind::Projection => {
            // Ae₁ = p keeps the line through p, and sends e₁ − p (the direction from p
            // to (1, 0)) to 0: A = p nᵀ / (n · p), n = (p_y, 1 − p_x), n · p = p_y. The
            // second column p (1 − p_x)/p_y grows without end as p nears the x axis
            // (there the two lines meet): p is kept far enough off it for that to stay
            // within the limit.
            let x = p.x.max(1.0 - l).min(l);
            let y = side(p.y, a.x_axis.y) * p.y.abs().max((x * (1.0 - x)).abs() / l).max(1e-6);
            let q = DVec2::new(x, y);
            DMat2::from_cols(q, q * ((1.0 - x) / y))
        }
        Kind::Nilpotent => {
            let n = nilpotent_from(p, a.x_axis.y, l);
            within_limits(n)
        }
        Kind::Shear => {
            // A = I + N, with N nilpotent and Ne₁ = p − e₁ (kept small enough for A to
            // stay within the limit)
            let n = nilpotent_from(p - DVec2::X, a.x_axis.y, l - 1.0);
            DMat2::IDENTITY + n
        }
        // (handled in `fit`)
        Kind::Diagonal | Kind::Triangular | Kind::Markov => a,
    }
}

/// The nilpotent matrix whose first column is `p` (or as near `p` as entries within
/// ±`l` allow). Its image and the line it sends to 0 are both the line through p, so
/// N = p mᵀ with m ⊥ p and m₁ = 1: the second column is −p p_x / p_y, which grows
/// without end as p nears the x axis; p is kept far enough off it. `was` is the
/// second entry of the first column before, whose sign is kept when p is on the axis.
fn nilpotent_from(p: DVec2, was: f64, l: f64) -> DMat2 {
    let x = p.x.max(-l).min(l);
    let sign = if p.y != 0.0 {
        p.y.signum()
    } else if was < 0.0 {
        -1.0
    } else {
        1.0
    };
    let y = sign * p.y.abs().min(l).max(x * x / l).max(1e-6);
    let q = DVec2::new(x, y);
    DMat2::from_cols(q, q * (-x / y))
}

/// Another matrix of the kind, at random.
fn random(kind: Kind, rng: &mut Rng) -> DMat2 {
    let mut r = |lo: f64, hi: f64| lo + (hi - lo) * rng.f32() as f64;
    let turn = |angle: f64| DMat2::from_angle(angle);
    let pi = std::f64::consts::PI;
    let m = match kind {
        Kind::Diagonal => {
            let (x, y, s) = (r(0.4, 1.8), r(0.3, 1.6), r(0.0, 1.0));
            rows([if s < 0.2 { -x } else { x }, 0.0, 0.0, if s > 0.8 { -y } else { y }])
        }
        Kind::Triangular => rows([r(0.4, 1.7), r(-1.2, 1.2), 0.0, r(-1.0, 1.4)]),
        Kind::Symmetric | Kind::PosDef => {
            let (big, other, angle) = (r(0.9, 1.9), if kind == Kind::PosDef { r(0.25, 0.9) } else { r(-1.2, 1.2) }, r(0.0, pi));
            let q = turn(angle);
            q * DMat2::from_diagonal(DVec2::new(big, other)) * q.transpose()
        }
        Kind::Skew => {
            let w = r(0.5, 1.5) * if r(0.0, 1.0) < 0.5 { -1.0 } else { 1.0 };
            J * w
        }
        Kind::Orthogonal => {
            if r(0.0, 1.0) < 0.5 {
                turn(r(0.35, 2.8) * if r(0.0, 1.0) < 0.5 { -1.0 } else { 1.0 })
            } else {
                // the reflection in the line at the angle φ
                let phi = r(0.0, pi);
                rows([(2.0 * phi).cos(), (2.0 * phi).sin(), (2.0 * phi).sin(), -(2.0 * phi).cos()])
            }
        }
        Kind::Normal => {
            if r(0.0, 1.0) < 0.5 {
                let (s, angle) = (r(0.6, 1.5), r(0.35, 2.4) * if r(0.0, 1.0) < 0.5 { -1.0 } else { 1.0 });
                turn(angle) * s
            } else {
                let q = turn(r(0.0, pi));
                q * DMat2::from_diagonal(DVec2::new(r(0.8, 1.8), r(-1.2, 0.6))) * q.transpose()
            }
        }
        Kind::Projection => {
            // the line kept, and the line sent to 0, at least 35° apart
            let kept = r(0.0, pi);
            let lost = kept + r(0.6, pi - 0.6);
            let (u, w) = (DVec2::from_angle(kept), DVec2::from_angle(lost));
            let n = w.perp();
            DMat2::from_cols(u * (n.x / n.dot(u)), u * (n.y / n.dot(u)))
        }
        Kind::Nilpotent | Kind::Shear => {
            let u = DVec2::from_angle(r(0.0, pi));
            let c = r(0.5, 1.1) * if r(0.0, 1.0) < 0.5 { -1.0 } else { 1.0 };
            let n = DMat2::from_cols(u * (c * (J * u).x), u * (c * (J * u).y));
            if kind == Kind::Shear {
                DMat2::IDENTITY + n
            } else {
                n
            }
        }
        Kind::Markov => {
            let (p, q) = (r(0.1, 0.95), r(0.05, 0.9));
            rows([p, q, 1.0 - p, 1.0 - q])
        }
    };
    within_limits(m)
}

/// Another matrix of the kind, not too like `was` (or as unlike it as a few tries find).
fn another(kind: Kind, was: DMat2, rng: &mut Rng) -> DMat2 {
    let mut best = random(kind, rng);
    for _ in 0..12 {
        if size(best - was) > 0.35 && is(kind, best) {
            break;
        }
        best = random(kind, rng);
    }
    best
}

// ----- two at once ---------------------------------------------------------------------

/// What is done with A (and B): it alone, a sum, a product, or its inverse.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Op {
    Alone,
    Sum,
    Product,
    Inverse,
}

const OPS: [Op; 4] = [Op::Alone, Op::Sum, Op::Product, Op::Inverse];

impl Op {
    /// Its button.
    fn label(self) -> &'static str {
        match self {
            Op::Alone => "$A$ alone",
            Op::Sum => "$A + B$",
            Op::Product => "$AB$",
            Op::Inverse => "$A^{-1}$",
        }
    }

    /// The matrix it makes, as written (without dollars).
    fn name(self) -> &'static str {
        match self {
            Op::Alone => "A",
            Op::Sum => "A + B",
            Op::Product => "AB",
            Op::Inverse => "A^{-1}",
        }
    }

    fn uses_b(self) -> bool {
        matches!(self, Op::Sum | Op::Product)
    }

    /// "a sum of two symmetric matrices", "the inverse of a symmetric matrix".
    fn of(self, kind: Kind) -> String {
        let plural = plural(kind);
        match self {
            Op::Alone => kind.a_name(),
            Op::Sum => format!("a sum of two {plural}"),
            Op::Product => format!("a product of two {plural}"),
            Op::Inverse => format!("the inverse of {}", kind.a_name()),
        }
    }
}

/// "symmetric matrices", "projections".
fn plural(kind: Kind) -> String {
    match kind {
        Kind::Projection => "projections".to_string(),
        Kind::Shear => "shears".to_string(),
        Kind::Markov => "Markov matrices".to_string(),
        k => format!("{} matrices", k.name()),
    }
}

/// The matrix the picture is about: A, A + B, AB or A⁻¹ (none if A has no inverse).
fn combine(op: Op, a: DMat2, b: DMat2) -> Option<DMat2> {
    match op {
        Op::Alone => Some(a),
        Op::Sum => Some(a + b),
        Op::Product => Some(a * b),
        Op::Inverse => inverse(a),
    }
}

/// The map the picture shows `power` steps in: Mᵏ when the power is a whole number k,
/// and between two, the matrices of the steps either side mixed in proportion.
fn powered(m: DMat2, power: f64) -> DMat2 {
    let p = power.clamp(0.0, 2.0);
    let k = (p.floor() as i32).min(1);
    let t = p - k as f64;
    let (lo, hi) = if k == 0 { (DMat2::IDENTITY, m) } else { (m, m * m) };
    lo + (hi - lo) * t
}

// ----- where things go -----------------------------------------------------------------

/// The part of the scene the view is fitted to (its centre, and half its size).
const FRAME_CENTRE: Vec2 = Vec2::new(0.2, 0.2);
const FRAME_HALF: Vec2 = Vec2::new(2.2, 1.85);

/// The screen panel's parts, in logical pixels.
#[derive(Clone, Debug)]
struct Panel {
    min: Vec2,
    max: Vec2,
    /// The kinds' buttons, in the order of `KINDS`.
    chips: Vec<(Vec2, Vec2)>,
    /// The line with the kind's equation (and, at its right end, what the bright
    /// buttons mean).
    title: Vec2,
    /// Where the line saying what the bright buttons mean starts (at its left, its middle
    /// height): after the last button if there is room, or on a line of its own.
    note: Vec2,
    /// How far down the verdict's lines may go.
    lines_end: f32,
    /// The matrices: the size of their numbers, how many to a row, where the first
    /// one's name goes, and how far apart they are across and down.
    text: f32,
    per_row: usize,
    cells: Vec2,
    step: Vec2,
    /// Where the table of what stays true of the kind goes (none on a phone).
    table: Option<Vec2>,
    chart_title: Vec2,
    chart: (Vec2, Vec2),
    eigen_line: Vec2,
    /// The buttons: another A; another B (when there is a B).
    another: (Vec2, Vec2),
    another_b: (Vec2, Vec2),
}

/// Rows of the panel, in logical pixels.
const CHIP_SIZE: f32 = 10.5;
const CHIP_H: f32 = 20.0;
const CHIP_GAP: f32 = 4.0;
const TITLE_H: f32 = 20.0;
const LINE_H: f32 = 17.0;
const LABEL_H: f32 = 17.0;
const BUTTON_H: f32 = 24.0;
/// The lines of the table of what stays true of the kind.
const TABLE_LINE: f32 = 16.0;
/// The lines of the verdict reserved under the matrices.
const VERDICT_LINES: usize = 3;
/// The same on a phone, where room is short (a third line, when there is one, is
/// written only where a row of matrices is free).
const VERDICT_LINES_NARROW: usize = 2;
/// The most characters a number in a matrix takes (a minus sign counted as two).
const CELL: usize = 6;
/// The panel's width beside the picture.
const PANEL_W: f32 = 316.0;

/// The size of the line under the kinds' buttons, and what it says.
const NOTE_SIZE: f32 = 10.5;

fn chips_note(op: Op) -> String {
    format!("bright: every kind ${}$ is", op.name())
}

/// The kinds' buttons flowed into rows across `width`: their rectangles, from (0, 0),
/// and the height they take.
fn chip_layout(width: f32) -> (Vec<(Vec2, Vec2)>, f32) {
    let (mut x, mut y) = (0.0f32, 0.0f32);
    let mut out = Vec::with_capacity(KINDS.len());
    for kind in KINDS {
        let w = text_width(kind.chip(), CHIP_SIZE) + 13.0;
        if x > 0.0 && x + w > width {
            (x, y) = (0.0, y + CHIP_H + CHIP_GAP);
        }
        out.push((Vec2::new(x, y), Vec2::new(x + w, y + CHIP_H)));
        x += w + CHIP_GAP;
    }
    (out, y + CHIP_H)
}

/// How wide a matrix of numbers is at the size `s`.
fn matrix_w(s: f32) -> f32 {
    matrix_size(2, 2, CELL, s).x
}

/// The size of the matrices' numbers and how many go across `width`: three to a row
/// if they can be read so, otherwise two as large as fit.
fn matrix_fit(width: f32) -> (f32, usize) {
    let gap = 14.0;
    for (per_row, least) in [(3usize, 11.5f32), (2, 10.0)] {
        let mut s = 13.5f32;
        while s >= least {
            if per_row as f32 * matrix_w(s) + (per_row - 1) as f32 * gap <= width {
                return (s, per_row);
            }
            s -= 0.5;
        }
    }
    (10.0, 1)
}

#[derive(Clone, Debug)]
struct Layout {
    /// A phone held upright: the panel goes below the picture.
    narrow: bool,
    /// The part of the canvas the picture has to itself.
    picture: (Vec2, Vec2),
    /// None on a canvas too small to spare the room (the numbers are in the side panel).
    panel: Option<Panel>,
    /// Where the caption goes: its left and right, and its foot.
    caption: Option<(f32, f32, f32)>,
}

impl Layout {
    fn of(view: Vec2, inset: f32, two: bool) -> Layout {
        let narrow = view.x < 620.0 && view.y >= 560.0;
        let pad = if narrow { 10.0 } else { 12.0 };
        let width = if narrow { view.x } else { PANEL_W };
        let inner = width - 2.0 * pad;
        let (chips, mut chips_h) = chip_layout(inner);
        let last = chips.last().map_or(Vec2::ZERO, |c| Vec2::new(c.1.x, (c.0.y + c.1.y) * 0.5));
        let beside = last.x + 10.0 + text_width(&chips_note(Op::Sum), NOTE_SIZE) <= inner;
        let note_at = if beside { last + Vec2::new(10.0, 0.0) } else { Vec2::new(0.0, chips_h + 4.0 + LINE_H * 0.5) };
        if !beside {
            chips_h += 4.0 + LINE_H;
        }
        let (text, per_row) = matrix_fit(inner);
        let mat_h = matrix_size(2, 2, CELL, text).y;
        // (A, the sides of its equation; and B and what is made of the two, for two at once)
        let cells: usize = if two { 5 } else { 3 };
        let rows = cells.div_ceil(per_row);
        let cell_h = LABEL_H + mat_h + 8.0;
        let verdict_h = if narrow { VERDICT_LINES_NARROW } else { VERDICT_LINES } as f32 * LINE_H;
        let below = 6.0 + LINE_H + 8.0 + BUTTON_H + pad;
        let without = pad + chips_h + 8.0 + TITLE_H + 4.0 + rows as f32 * cell_h + verdict_h + 6.0 + LINE_H + 4.0;
        // (beside the picture, if it is tall enough, there is room for the table of what
        // stays true of the kind, with a chart of a good size under it)
        let table_h = if !narrow && inset + 24.0 + without + 4.0 * TABLE_LINE + 6.0 + 150.0 + below <= view.y { 4.0 * TABLE_LINE + 6.0 } else { 0.0 };
        let above = without + table_h;
        let whole = |caption| Layout { narrow, picture: (Vec2::new(0.0, inset), view), panel: None, caption };
        let place = |min: Vec2, width: f32, chart_h: f32| {
            let x = min.x + pad;
            let mut y = min.y + pad;
            let chips = chips.iter().map(|(lo, hi)| (*lo + Vec2::new(x, y), *hi + Vec2::new(x, y))).collect();
            let note = note_at + Vec2::new(x, y);
            y += chips_h + 8.0;
            let title = Vec2::new(x, y);
            y += TITLE_H + 4.0;
            let cells_at = Vec2::new(x, y);
            let across = if per_row > 1 { (inner - matrix_w(text)) / (per_row - 1) as f32 } else { 0.0 };
            y += rows as f32 * cell_h + verdict_h;
            let lines_end = y;
            let table = (table_h > 0.0).then_some(Vec2::new(x, y));
            y += table_h + 6.0;
            let chart_title = Vec2::new(x, y);
            y += LINE_H + 4.0;
            let chart = (Vec2::new(x + 2.0, y), Vec2::new(min.x + width - pad - 2.0, y + chart_h));
            y += chart_h + 6.0;
            let eigen_line = Vec2::new(x, y);
            y += LINE_H + 8.0;
            let half = (inner - 8.0) * 0.5;
            let another = (Vec2::new(x, y), Vec2::new(x + half, y + BUTTON_H));
            let another_b = (Vec2::new(x + half + 8.0, y), Vec2::new(x + inner, y + BUTTON_H));
            let max = Vec2::new(min.x + width, y + BUTTON_H + pad);
            Panel { min, max, chips, title, note, lines_end, text, per_row, cells: cells_at, step: Vec2::new(across, cell_h), table, chart_title, chart, eigen_line, another, another_b }
        };
        if narrow {
            // the panel below the picture, with the caption at its foot
            let caption_h = 19.0 + 8.0;
            let chart_h = ((inner - 4.0) * 0.3).clamp(80.0, 120.0);
            let rest = above + below + caption_h;
            // (the picture keeps at least the top 40 %)
            let top = (view.y - rest - chart_h).max(view.y * 0.4);
            let chart_h = (view.y - top - rest).min(chart_h);
            if chart_h < 60.0 {
                return whole(Some((0.0, view.x, view.y - 10.0)));
            }
            let mut panel = place(Vec2::new(0.0, top), view.x, chart_h);
            panel.max.y = view.y;
            return Layout { narrow, picture: (Vec2::new(0.0, inset), Vec2::new(view.x, top)), panel: Some(panel), caption: Some((0.0, view.x, view.y - 10.0)) };
        }
        let top = inset + 12.0;
        let chart_h = ((inner - 4.0) * 0.66).min(view.y - 12.0 - top - above - below);
        if view.x < 600.0 || chart_h < 90.0 {
            return whole(Some((0.0, view.x, view.y - 16.0)));
        }
        let panel = place(Vec2::new(view.x - width - 12.0, top), width, chart_h);
        let picture = (Vec2::new(0.0, inset), Vec2::new(panel.min.x - 8.0, view.y));
        Layout { narrow, picture, panel: Some(panel), caption: Some((0.0, picture.1.x, view.y - 16.0)) }
    }

    /// The part of the canvas the scene's frame is fitted into: the picture's, less the
    /// heading at its top and the caption at its foot.
    fn frame_room(&self) -> (Vec2, Vec2) {
        let (min, max) = self.picture;
        let foot = if self.narrow { 6.0 } else { 56.0 };
        let head = if self.narrow { 70.0 } else { 64.0 };
        (min + Vec2::new(12.0, head), max - Vec2::new(12.0, foot))
    }
}

fn inside_rect(p: Vec2, (min, max): (Vec2, Vec2)) -> bool {
    p.cmpge(min).all() && p.cmple(max).all()
}

/// How wide a text is on the screen: one with mathematics in it exactly (and kept from
/// frame to frame), words alone as `text_width` judges. None for one with mathematics
/// in it before the fonts are in place (the first frame).
fn measured(f: &Frame, text: &str, size: f32) -> Option<f32> {
    if crate::typeset::has_math(text) {
        crate::typeset::measure(f.ctx, text, size).map(|s| s.x)
    } else {
        Some(text_width(text, size))
    }
}

/// The words of a text, a formula kept whole.
fn words_of(text: &str) -> Vec<&str> {
    let mut out = Vec::new();
    let (mut start, mut inside) = (0usize, false);
    let bytes = text.as_bytes();
    let mut i = 0;
    while i < bytes.len() {
        match bytes[i] {
            b'\\' => i += 1,
            b'$' => inside = !inside,
            b' ' if !inside => {
                if i > start {
                    out.push(&text[start..i]);
                }
                start = i + 1;
            }
            _ => {}
        }
        i += 1;
    }
    if start < text.len() {
        out.push(&text[start..]);
    }
    out
}

/// A text broken into lines no wider than `room` (each as wide as `measure` says; none
/// if it cannot say), and the widest line's width.
fn wrap_lines(text: &str, room: f32, measure: impl Fn(&str) -> Option<f32>) -> Option<(Vec<String>, f32)> {
    let mut lines: Vec<String> = Vec::new();
    for word in words_of(text) {
        let joined = lines.last().map(|last| format!("{last} {word}"));
        match (lines.last_mut(), joined) {
            (Some(last), Some(joined)) if measure(&joined)? <= room => *last = joined,
            _ => lines.push(word.to_string()),
        }
    }
    let mut widest = 0.0f32;
    for line in &lines {
        widest = widest.max(measure(line)?);
    }
    Some((lines, widest))
}

/// `parts` side by side as long as they fit across `room`, on further lines when they
/// do not, and the widest line's width (each text as wide as `measure` says; none if it
/// cannot say).
fn caption_lines(parts: &[&str], room: f32, measure: impl Fn(&str) -> Option<f32>) -> Option<(Vec<String>, f32)> {
    let gap = "      ";
    let mut lines: Vec<String> = Vec::new();
    for part in parts {
        let joined = lines.last().map(|last| format!("{last}{gap}{part}"));
        match (lines.last_mut(), joined) {
            (Some(last), Some(joined)) if measure(&joined)? <= room => *last = joined,
            _ => lines.push(part.to_string()),
        }
    }
    let mut widest = 0.0f32;
    for line in &lines {
        widest = widest.max(measure(line)?);
    }
    Some((lines, widest))
}

/// Lines of text as last broken, kept with what they were broken for.
struct Broken {
    of: (String, u32, u32),
    lines: Vec<String>,
    widest: f32,
}

/// `text` broken to `room` at `size`, from `kept` if it was broken so before.
fn broken(f: &Frame, text: &str, room: f32, size: f32, kept: &mut Option<Broken>) -> (Vec<String>, f32) {
    let of = (text.to_string(), room.to_bits(), size.to_bits());
    if let Some(k) = kept.as_ref().filter(|k| k.of == of) {
        return (k.lines.clone(), k.widest);
    }
    match wrap_lines(text, room, |t| measured(f, t, size)) {
        Some((lines, widest)) => {
            *kept = Some(Broken { of, lines: lines.clone(), widest });
            (lines, widest)
        }
        // (the formulas' fonts not in place yet: the first frame)
        None => wrap_lines(text, room, |t| Some(text_width(t, size))).unwrap_or_default(),
    }
}

// ----- drawing what may run far off ----------------------------------------------------

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
    (t0 <= t1).then(|| (if t0 <= 0.0 { a } else { a + d * t0 }, if t1 >= 1.0 { b } else { a + d * t1 }))
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

/// A path's points kept to the box `lo..hi` (the box reaches a view's width beyond the
/// view on every side, so whatever is moved lies outside the view): unlike cutting the
/// path at the box, this keeps its number of points the same however it moves.
fn within(points: &[DVec2], lo: DVec2, hi: DVec2) -> Vec<Vec3> {
    points.iter().map(|p| if p.is_finite() { p.clamp(lo, hi) } else { DVec2::ZERO }.as_vec2().extend(0.0)).collect()
}

/// The F painted on the unit square, as three bars (corners anticlockwise), and its
/// outline: the letter of "One map after another". Lopsided on purpose: turned, it is
/// still an F; mirrored, it is not.
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

/// Points round the unit circle.
const CIRCLE_POINTS: usize = 96;

fn unit_circle() -> Vec<DVec2> {
    (0..=CIRCLE_POINTS).map(|i| DVec2::from_angle(i as f64 / CIRCLE_POINTS as f64 * std::f64::consts::TAU)).collect()
}

/// The parts of the picture drawn as lines, each in a batch of its own when its number
/// of points may change from frame to frame (`eidos motion` compares the batches that
/// stay the same point for point).
#[derive(Clone, Copy, PartialEq)]
enum Part {
    Fixed,
    /// The eigenlines and the corner where they meet at a right angle, which come and go.
    Eigen,
    /// The handles' tracks and the factors' arrows, which come and go with the kind.
    Factors,
    /// The point x and its image.
    Point,
    /// The letter after one step, while a second is shown.
    Steps,
}

fn lines_for<'a>(f: &'a mut Frame, part: Part) -> &'a mut crate::gfx::draw::LineBatch {
    if part == Part::Fixed {
        return &mut f.dl.lines;
    }
    f.dl.lines_with(LineKind::Blend, LineStyle { width_scale: 1.0 + 1e-4 * part as u8 as f32, ..Default::default() })
}

/// The same, dashed as `Frame::dashed` dashes (the pattern afresh at every segment).
fn dashed_for<'a>(f: &'a mut Frame, part: Part) -> &'a mut crate::gfx::draw::LineBatch {
    f.dl.lines_with(LineKind::Blend, LineStyle { width_scale: 1.0 + 1e-4 * part as u8 as f32, dash: 9.0, duty: 0.55, ..Default::default() })
}

/// An arrow from `a` to `b` as `Frame::arrow` draws one, its shaft in the batch of
/// `part`, cut to the box `lo..hi` (without its head if the head is outside).
#[allow(clippy::too_many_arguments)]
fn arrow_in(f: &mut Frame, part: Part, a: DVec2, b: DVec2, lo: DVec2, hi: DVec2, width: f32, c: Color) {
    let Some((p, q)) = clip(a, b, lo, hi) else { return };
    let (p3, q3) = (p.as_vec2().extend(0.0), q.as_vec2().extend(0.0));
    let d = q3 - p3;
    let len = d.length();
    if len < 1e-9 {
        return;
    }
    let dir = d / len;
    let mut end = q3;
    if q == b {
        let wpp = f.cam.world_per_px(q3) * f.px;
        let head = ((width * 3.2 + 5.0) * wpp).min(len * 0.6);
        let side = dir.cross(Vec3::Z).normalize_or_zero() * (head * 0.42);
        let base = q3 - dir * head;
        f.dl.tris_alpha.tri(q3, base + side, base - side, c);
        end = q3 - dir * (head * 0.8);
    }
    lines_for(f, part).segment(p3, end, width, c);
}

fn inside_box(p: DVec2, lo: DVec2, hi: DVec2) -> bool {
    p.is_finite() && p.cmpge(lo).all() && p.cmple(hi).all()
}

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

/// The direction on the screen away from 0 through the point `p` of the scene (screen y
/// runs down).
fn away(p: DVec2) -> Vec2 {
    Vec2::new(p.x as f32, -(p.y as f32))
}

/// Where the line through 0 along `v` leaves the rectangle `rect` of the screen at the
/// end `v` points to, a little inside it, and the side of the line that is up on the
/// screen (none if 0 is not in view, or the line misses the rectangle).
fn end_on_screen(f: &Frame, v: DVec2, rect: (Vec2, Vec2)) -> Option<(Vec2, Vec2)> {
    let (o, tip) = (on_screen(f, Vec3::ZERO)?, on_screen(f, v.as_vec2().extend(0.0))?);
    let d = (tip - o).normalize_or_zero().as_dvec2();
    if d == DVec2::ZERO {
        return None;
    }
    let (_, end) = clip(o.as_dvec2() - d * 1e5, o.as_dvec2() + d * 1e5, rect.0.as_dvec2(), rect.1.as_dvec2())?;
    let up = if d.perp().y < 0.0 { d.perp() } else { -d.perp() };
    Some(((end - d * 30.0).as_vec2(), up.as_vec2()))
}

/// Whether two boxes on the screen meet.
fn overlap(a: (Vec2, Vec2), b: (Vec2, Vec2)) -> bool {
    a.0.x < b.1.x && b.0.x < a.1.x && a.0.y < b.1.y && b.0.y < a.1.y
}

// ----- the scene -----------------------------------------------------------------------

/// What glides when a button changes it: the two matrices, the white point, and how
/// many times the map is applied.
#[derive(Clone, Copy, Debug, PartialEq)]
struct Setting {
    a: DMat2,
    b: DMat2,
    x: DVec2,
    power: f64,
}

impl Setting {
    fn numbers(&self) -> [f64; 11] {
        let (a, b) = (self.a.to_cols_array(), self.b.to_cols_array());
        [a[0], a[1], a[2], a[3], b[0], b[1], b[2], b[3], self.x.x, self.x.y, self.power]
    }

    fn from_numbers(v: [f64; 11]) -> Setting {
        Setting { a: DMat2::from_cols_array(&[v[0], v[1], v[2], v[3]]), b: DMat2::from_cols_array(&[v[4], v[5], v[6], v[7]]), x: DVec2::new(v[8], v[9]), power: v[10] }
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
        if self.arrived() || dt <= 0.0 {
            return;
        }
        let k = 1.0 - (-dt / FOLLOW).exp();
        let (to, mut between, mut shown) = (self.to.numbers(), self.between.numbers(), self.shown.numbers());
        let mut apart = 0.0f64;
        for i in 0..to.len() {
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

/// The steps an experiment plays, I to A to A²: seconds each step takes, and the pause
/// at A between them.
const STEP_SECONDS: f32 = 1.3;
const STEP_PAUSE: f32 = 0.8;

/// How many times the map is applied, `s` seconds into the steps.
fn steps_at(s: f32) -> f64 {
    let first = smoothstep(s / STEP_SECONDS);
    let second = smoothstep((s - STEP_SECONDS - STEP_PAUSE) / STEP_SECONDS);
    (first + second) as f64
}

fn steps_length() -> f32 {
    2.0 * STEP_SECONDS + STEP_PAUSE
}

/// The loop the first column is dragged round in the first experiment: its radius,
/// and the seconds it takes.
const LOOP_RADIUS: f64 = 0.4;
const LOOP_SECONDS: f32 = 6.0;

/// Where the dragged column is, `s` seconds round the loop that starts at `from` (it
/// sets off upwards and comes back there).
fn loop_at(from: DVec2, s: f32) -> DVec2 {
    let angle = -std::f64::consts::FRAC_PI_2 + std::f64::consts::TAU * smoothstep(s / LOOP_SECONDS) as f64;
    let centre = from + DVec2::new(0.0, LOOP_RADIUS);
    centre + DVec2::from_angle(angle) * LOOP_RADIUS
}

struct Trial {
    label: &'static str,
    look_for: &'static str,
    kind: Kind,
    op: Op,
    a: [f64; 4],
    b: [f64; 4],
    /// The white point, if it is shown.
    x: Option<[f64; 2]>,
    /// Play the steps I, A, A².
    steps: bool,
    /// Drag the first column round a loop.
    tour: bool,
}

const TRIALS: [Trial; 5] = [
    Trial {
        label: "symmetric: right angles",
        look_for: r"$A^\top = A$. Its eigenlines (orange and purple) cross at a right angle, and the circle becomes an ellipse with its axes along them, stretched by the eigenvalues $\lambda = 2$ and $0.5$: real numbers, as a symmetric matrix's always are. Watch the pink arrow being dragged round a loop: the green one follows to keep $A$ symmetric, and the right angle stays. Then drag either arrow yourself.",
        kind: Kind::Symmetric,
        op: Op::Alone,
        a: [1.7, 0.6, 0.6, 0.8],
        b: [1.5, -0.5, -0.5, -0.5],
        x: None,
        steps: false,
        tour: true,
    },
    Trial {
        label: "orthogonal: a mirror",
        look_for: r"$A^\top A = I$: every length and angle is kept, so the circle lands on itself and the letter keeps its size and shape. This one is a reflection ($\det A = -1$), and the letter comes out mirrored: it reads backwards. Points on the mirror line, through $(2, 1)$, stay put ($\lambda = 1$, orange); points on the line across it are reversed ($\lambda = -1$, purple). Both eigenvalues lie on the unit circle, as every orthogonal matrix's do: a turn by $\theta$ has the pair $\cos\theta \pm i \sin\theta$, on the same circle.",
        kind: Kind::Orthogonal,
        op: Op::Alone,
        a: [0.6, 0.8, 0.8, -0.6],
        b: [0.6, 0.8, 0.8, -0.6],
        x: None,
        steps: false,
        tour: false,
    },
    Trial {
        label: "skew: always at right angles",
        look_for: r"In the plane every skew-symmetric matrix is a multiple of the quarter turn $J$: here $A = 1.2J$, a quarter turn and a stretch by 1.2. So $Ax$ is at right angles to $x$, whatever $x$ is: $x \cdot Ax = 0$. Drag the white point $x$ and watch. No line is kept, and the eigenvalues are $\pm 1.2i$, on the imaginary line.",
        kind: Kind::Skew,
        op: Op::Alone,
        a: [0.0, -1.2, 1.2, 0.0],
        b: [0.0, 0.8, -0.8, 0.0],
        x: Some([1.3, 0.5]),
        steps: false,
        tour: false,
    },
    Trial {
        label: "nilpotent: twice is nothing",
        look_for: r"The first step of $A$ squashes the whole plane onto the orange line, and the letter goes flat; the second step sends that line to $0$. So $A^2 = 0$, though $A$ itself is not $0$. Its only eigenvalue is $0$, and the orange line is its only eigenline: the line everything lands on is the line it sends to $0$. (A projection squashes the plane onto a line too, but there the second step changes nothing: $A^2 = A$.)",
        kind: Kind::Nilpotent,
        op: Op::Alone,
        a: [0.8, -1.6, 0.4, -0.8],
        b: [0.0, 1.0, 0.0, 0.0],
        x: None,
        steps: true,
        tour: false,
    },
    Trial {
        label: "a product of two symmetric",
        look_for: r"$A$ and $B$ (blue) are both symmetric, but $AB$ is not: $(AB)^\top = B^\top A^\top = BA$, and here $BA \ne AB$. The product even turns every direction: its eigenvalues $0.88 \pm 0.99i$ are off the real line, where a symmetric matrix's never are. Two symmetric matrices have a symmetric product exactly when they commute. Choose orthogonal among the kinds: a product of two orthogonal matrices always is one.",
        kind: Kind::Symmetric,
        op: Op::Product,
        a: [1.0, 0.5, 0.5, -1.5],
        b: [1.5, -0.5, -0.5, -0.5],
        x: None,
        steps: false,
        tour: false,
    },
];

/// How fast an eigenline's name follows its line's end: the time in which it covers
/// two thirds of the way (seconds).
const NAME_EASE: f32 = 0.05;

/// The handles: the columns of A and of B, and the white point.
const HANDLE_A: [u32; 2] = [1, 2];
const HANDLE_B: [u32; 2] = [3, 4];
const HANDLE_X: u32 = 5;

/// Where the white point starts.
const X_START: DVec2 = DVec2::new(1.3, 0.5);

pub struct Special {
    kind: Kind,
    op: Op,
    glide: Glide,
    /// Seconds into the steps I, A, A², while they play (they start once the glide to a
    /// new setting has arrived).
    steps: Option<f32>,
    /// Seconds round the loop the first column is dragged round, while it goes, and
    /// where it set off from.
    tour: Option<(f32, DVec2)>,
    show_x: bool,
    eigenlines: bool,
    /// The eigenlines' directions as last named: a line's name keeps its end. And where
    /// on the screen each name was put, which it follows through an ease (and whether
    /// it is still on its way, or must be put afresh: the view has changed).
    last_lines: [Option<DVec2>; 2],
    anchors: [Option<Vec2>; 2],
    anchors_moving: bool,
    anchors_reset: bool,
    rng: Rng,
    /// The heading's lines and the caption's, as last broken.
    heading: [Option<Broken>; 2],
    caption: Option<Broken>,
    /// The canvas last drawn on and how much of its top the application's buttons
    /// cover: they decide where the panel goes, and so where the picture is.
    canvas: Vec2,
    inset: f32,
    measured: bool,
    new_view: bool,
}

impl Special {
    fn new() -> Self {
        let kind = Kind::Symmetric;
        Special {
            kind,
            op: Op::Alone,
            glide: Glide::at(Setting { a: kind.default_a(), b: kind.default_b(), x: X_START, power: 1.0 }),
            steps: None,
            tour: None,
            show_x: false,
            eigenlines: true,
            last_lines: [None; 2],
            anchors: [None; 2],
            anchors_moving: false,
            anchors_reset: false,
            rng: Rng::new(7),
            heading: [None, None],
            caption: None,
            canvas: Vec2::new(1080.0, 1000.0),
            inset: 0.0,
            measured: false,
            new_view: false,
        }
    }

    /// Set up experiment `k` of `TRIALS`: every setting it depends on, glided to.
    fn set_up(&mut self, k: usize) {
        let trial = &TRIALS[k];
        let two = self.op != Op::Alone;
        self.kind = trial.kind;
        self.op = trial.op;
        self.new_view |= two != (self.op != Op::Alone);
        let a = rows(trial.a);
        self.glide.to = Setting { a, b: rows(trial.b), x: DVec2::from(trial.x.unwrap_or([X_START.x, X_START.y])), power: if trial.steps { 0.0 } else { 1.0 } };
        self.show_x = trial.x.is_some();
        self.eigenlines = true;
        self.steps = trial.steps.then_some(0.0);
        self.tour = trial.tour.then_some((0.0, a.x_axis));
    }

    /// Choose a kind: A stays if it is of that kind already (B too), and otherwise
    /// glides to the kind's own.
    fn choose(&mut self, kind: Kind) {
        self.kind = kind;
        let to = &mut self.glide.to;
        if !is(kind, to.a) {
            to.a = kind.default_a();
        }
        if !is(kind, to.b) {
            to.b = kind.default_b();
        }
        (self.steps, self.tour) = (None, None);
        to.power = to.power.round();
    }

    /// Choose what is done with A and B.
    fn choose_op(&mut self, op: Op) {
        self.new_view |= (op == Op::Alone) != (self.op == Op::Alone);
        self.op = op;
        self.tour = None;
    }

    /// Another A of the kind (or B, if `second`).
    fn another(&mut self, second: bool) {
        let was = if second { self.glide.to.b } else { self.glide.to.a };
        let m = another(self.kind, was, &mut self.rng);
        if second {
            self.glide.to.b = m;
        } else {
            self.glide.to.a = m;
        }
        (self.steps, self.tour) = (None, None);
    }

    /// The setting as drawn.
    fn shown(&self) -> Setting {
        self.glide.shown
    }

    /// Move everything on: `dt` seconds of the scene's time (the steps, the loop) and
    /// `real_dt` of the clock's (a change of setting glides by the clock, so that it is
    /// seen through even while the scene's time stands still).
    fn advance(&mut self, dt: f32, real_dt: f32) {
        self.glide.step(real_dt as f64);
        if let Some(s) = self.steps {
            if self.glide.arrived() {
                let s = s + dt.max(0.0);
                let power = steps_at(s);
                self.glide.now(|g| g.power = power);
                self.steps = (s < steps_length()).then_some(s);
            }
        }
        if let Some((s, from)) = self.tour {
            if self.glide.arrived() {
                let s = s + dt.max(0.0);
                let a = fit(self.kind, self.glide.to.a, 0, loop_at(from, s.min(LOOP_SECONDS)));
                self.glide.now(|g| g.a = a);
                self.tour = (s < LOOP_SECONDS).then_some((s, from));
            }
        }
    }

    /// The canonical view: the picture's frame in the middle of the part of the canvas
    /// it has to itself.
    fn the_view(&self) -> View {
        let lay = Layout::of(self.canvas, self.inset, self.op != Op::Alone);
        let (min, max) = lay.frame_room();
        let room = (max - min).max(Vec2::splat(60.0));
        let per_unit = (room.x / (2.0 * FRAME_HALF.x)).min(room.y / (2.0 * FRAME_HALF.y)).max(1e-3);
        // (screen y runs down, the scene's up)
        let off = ((min + max) * 0.5 - self.canvas * 0.5) / per_unit;
        View::Fit { center: Vec2::new(FRAME_CENTRE.x - off.x, FRAME_CENTRE.y + off.y), half: self.canvas.max(Vec2::ONE) * (0.5 / per_unit) }
    }

    /// The matrix under test as asked for (none if it is an inverse that does not exist).
    fn tested(&self) -> Option<DMat2> {
        combine(self.op, self.glide.to.a, self.glide.to.b)
    }

    /// The name the picture gives the map it draws: `A`, `AB`, `A^2`, ...
    fn drawn_name(&self, power: f64) -> String {
        let n = self.op.name();
        let k = power.round() as i32;
        match k {
            0 => String::new(),
            1 => n.to_string(),
            _ => format!("{}^{k}", paren(n)),
        }
    }
}

/// The four numbers of a 2×2 matrix to drag or type, within ±`ENTRY_MAX`. Returns the
/// column changed and its new value, and whether it was typed (a number dragged moves
/// the picture at once, one typed glides it there).
fn matrix_edit(ui: &mut egui::Ui, m: DMat2) -> Option<(usize, DVec2, bool)> {
    let mut out = None;
    let mut a = [[m.x_axis.x, m.y_axis.x], [m.x_axis.y, m.y_axis.y]];
    egui::Grid::new("special_a").spacing([6.0, 4.0]).show(ui, |ui| {
        for i in 0..2 {
            for j in 0..2 {
                let r = ui.add(egui::DragValue::new(&mut a[i][j]).speed(0.01).fixed_decimals(2).range(-ENTRY_MAX..=ENTRY_MAX));
                if r.changed() {
                    out = Some((j, DVec2::new(a[0][j], a[1][j]), !r.dragged()));
                }
            }
            ui.end_row();
        }
    });
    out
}

/// The label of one of egui's own widgets (a button, a checkbox), typeset.
fn typeset_label(ui: &egui::Ui, text: &str) -> egui::WidgetText {
    crate::typeset::seen(text);
    ui::label_text(ui, text, 12.5, None, false)
}

impl Scene for Special {
    fn view(&self) -> View {
        self.the_view()
    }

    fn take_view(&mut self) -> Option<View> {
        std::mem::take(&mut self.new_view).then(|| self.the_view())
    }

    fn settings(&self, s: &mut ViewSettings) {
        s.grid_alpha = 0.10;
        s.axis_alpha = 0.35;
    }

    fn animated(&self) -> bool {
        !self.glide.arrived() || self.steps.is_some() || self.tour.is_some() || self.anchors_moving
    }

    fn ui(&mut self, ui: &mut egui::Ui, _: &Gfx) {
        ui::note(ui, r"Each kind of matrix is defined by an equation: symmetric is $A^\top = A$, orthogonal is $A^\top A = I$, a projection is $A^2 = A$ (three kinds add a condition). That equation decides what the map can do to the plane (the yellow letter and circle show it) and where its eigenvalues can lie: they are the dots in the complex plane beside the picture or below it, and the bright line or points are where this kind's must be. Pick a kind and drag the tips of the arrows: the matrix stays of its kind.");

        ui::section(ui, "Things to try");
        for (k, trial) in TRIALS.iter().enumerate() {
            if ui::experiment(ui, trial.label, trial.look_for) {
                self.set_up(k);
            }
        }

        ui::section(ui, "Kind");
        let names: Vec<&str> = KINDS.iter().map(|k| k.name()).collect();
        let mut chosen = KINDS.iter().position(|k| *k == self.kind).unwrap_or(0);
        if ui::choice(ui, &names, &mut chosen) {
            self.choose(KINDS[chosen]);
        }
        ui::note(ui, &format!("{}: {}. {}.", self.kind.equation("A"), does(self.kind, self.glide.to.a), capitalised(self.kind.eigen_words())));
        if ui.button(typeset_label(ui, "another of this kind")).clicked() {
            self.another(false);
        }

        ui::section(ui, "Matrix $A$");
        if let Some((j, col, typed)) = matrix_edit(ui, self.glide.to.a) {
            let a = fit(self.kind, self.glide.to.a, j, col);
            if typed {
                self.glide.to.a = a;
            } else {
                self.glide.now(|s| s.a = a);
            }
            (self.steps, self.tour) = (None, None);
        }
        ui::note(ui, "Drag a number sideways or type one: the other entries follow, as little as they can, so that $A$ stays of its kind. A number the kind does not allow springs back.");

        ui::section(ui, "Two at once");
        let ops: Vec<&str> = OPS.iter().map(|o| o.label()).collect();
        let mut op = OPS.iter().position(|o| *o == self.op).unwrap_or(0);
        if ui::choice(ui, &ops, &mut op) {
            self.choose_op(OPS[op]);
        }
        if self.op == Op::Alone {
            ui::note(ui, r"Is a sum, a product or an inverse of this kind of the same kind? Choose one: the picture and the equation are then about it, and $B$ is a second matrix of the kind (its arrows blue; drag their tips).");
        } else {
            ui::note(ui, self.kind.closure(self.op));
            if self.op.uses_b() && ui.button(typeset_label(ui, "another $B$")).clicked() {
                self.another(true);
            }
        }

        ui::section(ui, "Steps");
        let mut power = self.glide.to.power as f32;
        if ui::slider(ui, "times the map is applied", &mut power, 0.0..=2.0) {
            self.glide.now(|s| s.power = power as f64);
            (self.steps, self.tour) = (None, None);
        }
        if ui.button(typeset_label(ui, "play: $I$, then once, then twice")).clicked() {
            self.glide.to.power = 0.0;
            self.steps = Some(0.0);
            self.tour = None;
        }
        ui::note(ui, r"From the plane as it was (0) to the map applied once (1) and twice (2). Applied any other number of times, the columns of $A$ itself are the white arrows, and their tips can be dragged too.");

        ui::section(ui, "Show");
        ui::checkbox(ui, &mut self.eigenlines, "the eigenlines");
        ui::checkbox(ui, &mut self.show_x, r"the white point $x$ and $Ax$");

        ui::section(ui, "Reading the picture");
        let mut key = vec![
            (pal::YELLOW, "yellow: a letter and the unit circle after the map; dashed: before it"),
            (pal::PINK, r"pink: where $(1, 0)$ lands, the first column of the map; drag its tip"),
            (pal::GREEN, r"green: where $(0, 1)$ lands, the second column"),
            (pal::ORANGE, r"orange and purple, dashed: the eigenlines, each with its $\lambda$; a corner where they meet at a right angle"),
        ];
        if self.op.uses_b() {
            key.push((pal::BLUE, r"blue: the columns of $B$; drag their tips"));
        }
        if self.op != Op::Alone {
            key.push((pal::WHITE, r"white arrows: the columns of $A$; drag their tips"));
        }
        if self.show_x {
            key.push((pal::WHITE, r"white dot: the point $x$ (drag it), and the arrow to where the map sends it"));
        }
        key.push((pal::TEXT, "the complex plane, beside the picture or below it: the eigenvalues as dots, and the bright line or points where this kind's can lie; a red ring round a dot that lies outside them"));
        ui::key(ui, &key);

        ui::section(ui, "Words");
        ui::term(ui, r"Transpose $A^\top$:", r"$A$ flipped over its diagonal: its rows become columns, $(A^\top)_{ij} = a_{ji}$. For every $x$ and $y$, $x \cdot Ay = A^\top x \cdot y$.");
        ui::term(ui, "Eigenvalue, eigenline:", r"$Av = \lambda v$: along an eigenline the map only stretches, by $\lambda$ (a negative $\lambda$ reverses it).");
        ui::term(ui, "Complex eigenvalues:", r"when no line is kept, the eigenvalues are a pair $a \pm bi$, mirror images across the real line. A real matrix's complex eigenvalues always come in such pairs.");
        ui::term(ui, "$I$, $J$:", r"$I$ leaves every point where it is; $J$ turns every vector a quarter turn anticlockwise, $J^2 = -I$.");
        ui::term(ui, "Commute:", r"$AB = BA$: the two maps give the same result in either order.");
        for kind in KINDS {
            ui::term(ui, &format!("{}:", capitalised(kind.name())), kind_words(kind));
        }

        ui::section(ui, "Numbers");
        let a = self.glide.to.a;
        ui::readout(ui, "kind", self.kind.name());
        ui::readout(ui, r"$\det A$, $\tr A$", format!("${}, {}$", num(a.determinant(), 2), num(trace(a), 2)));
        ui::readout(ui, "eigenvalues of $A$", eigen_text(eigen(a)));
        if self.op != Op::Alone {
            let n = self.op.name();
            match self.tested() {
                Some(c) => {
                    ui::readout(ui, &format!("eigenvalues of ${n}$"), eigen_text(eigen(c)));
                    ui::readout(ui, &format!("${n}$ is {}", self.kind.name()), if is(self.kind, c) { "yes" } else { "no" });
                }
                None => ui::readout(ui, &format!("${n}$"), r"none: $\det A = 0$"),
            }
        }
        ui::readout(ui, "off the equation by", format!("${}$", self.tested().map_or("?".to_string(), |c| small(test(self.kind, c).0))));
        ui::see_also(ui, &[("symskew", "every matrix as symmetric plus skew-symmetric"), ("project", "projections and mirrors"), ("commute", "when $AB = BA$"), ("eigen", "finding eigenvalues")]);
    }

    fn frame(&mut self, f: &mut Frame) {
        // the panel's place follows the canvas, and the picture's place follows the panel's
        let view = f.view_size();
        let two = self.op != Op::Alone;
        let lay = Layout::of(view, f.top_inset, two);
        let before = Layout::of(self.canvas, self.inset, two);
        // (a canvas squeezed to a sliver is one the controls have opened over, on a
        // phone: the picture is not seen then, and is left as it was for when they close)
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
        // (the names of the eigenlines are put afresh on a canvas of another shape)
        (self.anchors_moving, self.anchors_reset) = (false, changed);

        // Every batch of lines of the scene's own is made in the same order in every
        // frame, and kept in its place in the list even in a frame that draws nothing
        // in it (two breaks, which draw nothing, are what an empty batch would not have:
        // it would be dropped, and made again at the end of the list): `eidos motion`
        // compares the batches of one frame with the next by their places in the list.
        let keep = |batch: &mut crate::gfx::draw::LineBatch| batch.pts.extend([crate::gfx::draw::BREAK; 2]);
        for part in [Part::Eigen, Part::Factors, Part::Point, Part::Steps] {
            keep(lines_for(f, part));
        }
        keep(dashed_for(f, Part::Eigen));
        for kind in [plot::AXES, plot::MARKS, plot::BUTTONS] {
            keep(plot::apart(f, kind, false));
        }
        f.dl.points_with(PointKind::Overlay, PointStyle { outline_px: 0.0, ..Default::default() });

        let pointer = f.ptr.pos.map(|p| p / f.px);
        if let Some(panel) = lay.panel.as_ref().filter(|_| !covered) {
            if pointer.is_some_and(|p| inside_rect(p, (panel.min, panel.max))) {
                // the panel is for its own buttons: no moving the view from there
                f.handles.capture();
                f.handles.claim_clicks();
            }
        }
        self.advance(f.dt, f.real_dt);

        let labels_before = f.dl.labels.len();
        let plates = self.picture(f, &lay);
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
        if let Some(p) = &lay.panel {
            // nothing written into the picture may show through the panel
            let (lo, hi) = (p.min - Vec2::splat(2.0), p.max + Vec2::splat(2.0));
            let mut k = 0;
            f.dl.labels.retain(|l| {
                k += 1;
                k <= labels_before || l.backed || !inside_rect(l.pos, (lo, hi))
            });
            self.panel(f, p);
        }
    }
}

/// A word with its first letter a capital.
fn capitalised(s: &str) -> String {
    let mut c = s.chars();
    match c.next() {
        Some(first) => first.to_uppercase().chain(c).collect(),
        None => String::new(),
    }
}

/// What each kind is, for the panel's words.
fn kind_words(kind: Kind) -> &'static str {
    match kind {
        Kind::Diagonal => r"$0$ off the diagonal: each axis is stretched on its own.",
        Kind::Triangular => r"$0$ below the diagonal, $a_{21} = 0$: the $x$ axis is kept.",
        Kind::Symmetric => r"$A^\top = A$: a stretch along two perpendicular eigenlines.",
        Kind::PosDef => r"symmetric, with $x \cdot Ax > 0$ for every $x \ne 0$: both stretches are positive.",
        Kind::Skew => r"$A^\top = -A$: in the plane $wJ$, a quarter turn and a stretch.",
        Kind::Orthogonal => r"$A^\top A = I$: lengths and angles are kept; a rotation or a reflection.",
        Kind::Normal => r"$AA^\top = A^\top A$: eigenvectors at right angles (complex ones allowed).",
        Kind::Projection => r"$A^2 = A$: each point dropped onto one line along another.",
        Kind::Nilpotent => r"some power of it is $0$; in the plane, $A^2 = 0$.",
        Kind::Shear => r"$(A - I)^2 = 0$, $A \ne I$: one line kept point for point, every other point sliding along it.",
        Kind::Markov => r"entries $\ge 0$ and columns adding to $1$: shares go to shares.",
    }
}

impl Special {
    /// Everything drawn in the scene's own coordinates, and the names written beside it.
    /// Returns the boxes of the heading and of the caption, which are written over it.
    fn picture(&mut self, f: &mut Frame, lay: &Layout) -> [Option<(Vec2, Vec2)>; 2] {
        let (lo, hi) = f.cam.flat_bounds();
        // what is drawn is cut off a view's width beyond the view, whatever the matrices
        let span = (hi - lo).as_dvec2();
        let (box_lo, box_hi) = (lo.as_dvec2() - span, hi.as_dvec2() + span);
        let wpp = ((hi.y - lo.y) / f.view_size().y.max(1.0)) as f64;

        let shown = self.shown();
        let two = self.op != Op::Alone;
        let tested = combine(self.op, shown.a, shown.b);
        let drawn = tested.map(|c| powered(c, shown.power));
        // the arrows of A itself are the map's own columns when the map is A
        let a_is_drawn = !two && (shown.power - 1.0).abs() < 1e-9;


        let mut room = Room::new(f);
        keep_ticks_clear(f, &mut room);
        if let Some(p) = &lay.panel {
            room.wall(p.min - Vec2::splat(4.0), p.max + Vec2::splat(4.0));
        }
        let heading = self.heading(f, lay);
        room.wall(heading.0, heading.1);
        let mut plates = [Some(heading), None];
        if let Some((left, right, bottom)) = lay.caption {
            let parts: &[&str] = if lay.narrow { &["yellow: the letter and the circle after the map", "dashed: before"] } else { &["yellow: a letter and the unit circle, after the map", "dashed: before it"] };
            if let Some(plate) = self.caption(f, parts, (left, right), bottom) {
                room.wall(plate.0, plate.1);
                plates[1] = Some(plate);
            }
        }

        // where each handle may go, where the kind does not leave it free
        if a_is_drawn || two {
            self.tracks(f, shown.a);
        }

        // the unit circle and the F, before (dashed) and after the map
        let circle = unit_circle();
        // (dashes of the circle laid along it in the scene's own units, so that they stay
        // where they are as the view moves; the F's straight edges dashed as lines are)
        for pair in circle.chunks_exact(2) {
            f.line(pair[0].as_vec2(), pair[1].as_vec2(), 1.0, pal::YELLOW.a(0.35));
        }
        for k in 0..F_OUTLINE.len() {
            f.dashed(F_OUTLINE[k].as_vec2(), F_OUTLINE[(k + 1) % F_OUTLINE.len()].as_vec2(), 1.0, pal::YELLOW.a(0.4));
        }
        // (after a first step, where it was after one step stays faint, to compare the second with)
        if let Some(c) = tested.filter(|_| shown.power > 1.0) {
            lines_for(f, Part::Steps).polyline(within(&F_OUTLINE.iter().chain(&F_OUTLINE[..1]).map(|p| c * *p).collect::<Vec<_>>(), box_lo, box_hi), 1.0, pal::YELLOW.a(0.3));
        }
        if let Some(m) = drawn {
            f.path(within(&circle.iter().map(|p| m * *p).collect::<Vec<_>>(), box_lo, box_hi), 2.0, pal::YELLOW.a(0.9));
            for bar in F_BARS {
                let cut = clip_polygon(&bar.map(|p| m * p), box_lo, box_hi);
                for i in 1..cut.len().saturating_sub(1) {
                    f.dl.tris_alpha.tri(cut[0].as_vec2().extend(0.0), cut[i].as_vec2().extend(0.0), cut[i + 1].as_vec2().extend(0.0), pal::YELLOW.a(0.32));
                }
            }
            let outline: Vec<DVec2> = F_OUTLINE.iter().chain(&F_OUTLINE[..1]).map(|p| m * *p).collect();
            f.path(within(&outline, box_lo, box_hi), 1.6, pal::YELLOW.a(0.95));
            room.shape(f, &F_OUTLINE.map(|p| (m * p).as_vec2().extend(0.0)));
            for pair in outline.windows(2) {
                room.line(f, pair[0].as_vec2().extend(0.0), pair[1].as_vec2().extend(0.0));
            }
            for pair in circle.windows(4).step_by(3) {
                room.line(f, (m * pair[0]).as_vec2().extend(0.0), (m * pair[3]).as_vec2().extend(0.0));
            }
        }

        // the eigenlines of the matrix under test, with the right angle where they make one
        let reach = (hi - lo).length() as f64 * 2.0;
        let mut names: Vec<(Tag, Vec<Vec3>, String, Color, Vec2)> = Vec::new();
        if let (Some(c), true) = (tested, self.eigenlines) {
            if let Eigen::Real { l1, l2, lines } = eigen(c) {
                let list: Vec<(f64, DVec2, Color)> = match lines {
                    Lines::Two(v1, v2) => vec![(l1, v1, pal::ORANGE), (l2, v2, pal::PURPLE)],
                    Lines::One(v) => vec![(l1, v, pal::ORANGE)],
                    Lines::Every => Vec::new(),
                };
                for i in list.len()..2 {
                    (self.last_lines[i], self.anchors[i]) = (None, None);
                }
                for (i, (l, v, col)) in list.iter().enumerate() {
                    if let Some((p, q)) = clip(-*v * reach, *v * reach, box_lo, box_hi) {
                        dashed_for(f, Part::Eigen).segment(p.as_vec2().extend(0.0), q.as_vec2().extend(0.0), 1.6, col.a(0.85));
                        room.line(f, p.as_vec2().extend(0.0), q.as_vec2().extend(0.0));
                    }
                    // named near where it leaves the picture: at the end it was named at
                    // before (a line new to the picture at its end on the right)
                    let mut v = *v;
                    let fresh = self.last_lines[i].is_none();
                    let was = self.last_lines[i].unwrap_or(DVec2::new(1.0, 1e-3));
                    if was.dot(v) < 0.0 {
                        v = -v;
                    }
                    self.last_lines[i] = Some(v);
                    let text = if matches!(lines, Lines::One(_)) { format!(r"$\lambda = {}$ twice", num(*l, 2)) } else { format!(r"$\lambda = {}$", num(*l, 2)) };
                    if let Some((at, up)) = end_on_screen(f, v, lay.frame_room()) {
                        // (where a line turns fast, its end far out sweeps along the edge:
                        // the name follows it through a short ease, rather than leaping
                        // with it; a line new to the picture is named where it is)
                        let at = match self.anchors[i].filter(|_| !fresh && !self.anchors_reset) {
                            Some(was) => {
                                let next = was + (at - was) * (1.0 - (-f.real_dt.max(0.0) / NAME_EASE).exp());
                                if next.distance(at) < 0.5 || !next.is_finite() {
                                    at
                                } else {
                                    self.anchors_moving = true;
                                    next
                                }
                            }
                            None => at,
                        };
                        self.anchors[i] = Some(at);
                        if let Some(w) = f.cam.pick_ground(at * f.px) {
                            names.push((Tag("eigen", i as u32), vec![w], text, *col, up));
                        }
                    }
                }
                // (in the quarter between the lines' ends that are named, which keep their
                // sides as the lines turn)
                if let (Lines::Two(v1, v2), [Some(u1), Some(u2)]) = (lines, self.last_lines) {
                    let (v1, v2) = (if u1.dot(v1) < 0.0 { -v1 } else { v1 }, if u2.dot(v2) < 0.0 { -v2 } else { v2 });
                    if v1.dot(v2).abs() < 1e-6 {
                        let mark = (16.0 * wpp) as f32;
                        lines_for(f, Part::Eigen).polyline(
                            {
                                let (a, b) = (v1.as_vec2() * mark, v2.as_vec2() * mark);
                                [a, a + b, b].map(|p| p.extend(0.0))
                            },
                            1.4,
                            pal::TEXT.a(0.75),
                        );
                    }
                }
            } else {
                (self.last_lines, self.anchors) = ([None; 2], [None; 2]);
            }
        }

        // the columns of the map drawn: where (1, 0) and (0, 1) land
        if let Some(m) = drawn {
            let label = self.drawn_name(shown.power);
            for (j, col) in [(0, pal::PINK), (1, pal::GREEN)] {
                let tip = m.col(j);
                arrow_in(f, Part::Fixed, DVec2::ZERO, tip, box_lo, box_hi, 3.0, col);
                room.line(f, Vec3::ZERO, tip.as_vec2().extend(0.0));
                let hat = if j == 0 { r"\hat\imath" } else { r"\hat\jmath" };
                if !a_is_drawn && inside_box(tip, box_lo, box_hi) {
                    f.dot(tip.as_vec2(), 7.0, col);
                }
                names.push((Tag("column", j as u32), vec![tip.as_vec2().extend(0.0)], format!("${label}{hat}$"), col, away(tip)));
            }
        } else {
            names.clear();
        }

        // the handles: A's columns (the map's own arrows, or white ones of their own), B's
        let mut factors: Vec<(usize, u32, DVec2, Color, String)> = Vec::new();
        for j in 0..2 {
            let hat = if j == 0 { r"\hat\imath" } else { r"\hat\jmath" };
            let colour = if a_is_drawn { [pal::PINK, pal::GREEN][j] } else { pal::WHITE };
            factors.push((j, HANDLE_A[j], shown.a.col(j), colour, format!("$A{hat}$")));
            if self.op.uses_b() {
                factors.push((j + 2, HANDLE_B[j], shown.b.col(j), pal::BLUE, format!("$B{hat}$")));
            }
        }
        // (A's own white arrows fade in as the steps leave the map A itself, on which
        // they would lie, and their names come once they are seen)
        let seen = if two { 1.0 } else { smoothstep(((shown.power - 1.0).abs() / 0.35) as f32) };
        for (which, id, tip, colour, text) in factors {
            if !a_is_drawn {
                arrow_in(f, Part::Factors, DVec2::ZERO, tip, box_lo, box_hi, 1.6, colour.a(0.8 * seen));
                room.line(f, Vec3::ZERO, tip.as_vec2().extend(0.0));
                if seen > 0.6 {
                    names.push((Tag("factor", which as u32), vec![tip.as_vec2().extend(0.0)], text, colour, away(tip)));
                }
            }
            let mut grip = tip.as_vec2();
            if !inside_box(tip, box_lo, box_hi) {
                continue;
            }
            if f.drag2(id, &mut grip, colour) {
                let (j, second) = (which % 2, which >= 2);
                let p = grip.as_dvec2();
                let kind = self.kind;
                if second {
                    let b = fit(kind, self.glide.to.b, j, p);
                    self.glide.now(|s| s.b = b);
                } else {
                    let a = fit(kind, self.glide.to.a, j, p);
                    self.glide.now(|s| s.a = a);
                }
                (self.steps, self.tour) = (None, None);
            }
        }

        // the white point and where the map sends it
        if self.show_x {
            let mut x = shown.x.as_vec2();
            if f.drag2(HANDLE_X, &mut x, pal::WHITE) {
                let x = x.as_dvec2().clamp(DVec2::splat(-4.0), DVec2::splat(4.0));
                self.glide.now(|s| s.x = x);
            }
            let x = self.shown().x;
            lines_for(f, Part::Point).segment(Vec3::ZERO, x.as_vec2().extend(0.0), 1.2, pal::WHITE.a(0.45));
            room.line(f, Vec3::ZERO, x.as_vec2().extend(0.0));
            names.push((Tag("x", 0), vec![x.as_vec2().extend(0.0)], "$x$".into(), pal::TEXT, away(x)));
            if let Some(m) = drawn {
                let mx = m * x;
                arrow_in(f, Part::Point, DVec2::ZERO, mx, box_lo, box_hi, 2.4, pal::WHITE);
                room.line(f, Vec3::ZERO, mx.as_vec2().extend(0.0));
                let n = self.drawn_name(shown.power);
                names.push((Tag("x", 1), vec![mx.as_vec2().extend(0.0)], format!("${n}x$"), pal::WHITE, away(mx)));
                // the corner where x and Mx meet at a right angle
                if x.length() > 1e-9 && mx.length() > 1e-9 && (x.dot(mx) / (x.length() * mx.length())).abs() < 1e-6 {
                    let mark = (12.0 * wpp) as f32;
                    let (a, b) = (x.normalize().as_vec2() * mark, mx.normalize().as_vec2() * mark);
                    lines_for(f, Part::Point).polyline([a, a + b, b].map(|p| p.extend(0.0)), 1.4, pal::WHITE.a(0.8));
                }
            }
        }

        // a map that sends everything to 0 says so, where everything is
        if let Some(m) = drawn.filter(|m| size(*m) <= EXACT) {
            let _ = m;
            let n = self.drawn_name(shown.power);
            room.label(f, Tag("zero", 0), Vec3::ZERO, &format!("${n} = 0$: everything at $0$"), 14.0, pal::YELLOW, Vec2::new(-1.0, -1.0), 18.0, false);
        }

        // the names, each where it has room (none for an arrow too short to see)
        let origin = on_screen(f, Vec3::ZERO);
        let seen = |f: &Frame, p: Vec3| on_screen(f, p).is_some_and(|q| inside_rect(q, (Vec2::ZERO, f.view_size())));
        for (tag, points, text, colour, towards) in names {
            let points: Vec<Vec3> = points.into_iter().filter(|p| seen(f, *p)).collect();
            let short = tag.0 != "eigen" && tag != Tag("x", 0) && points.first().and_then(|p| on_screen(f, *p)).zip(origin).is_some_and(|(q, o)| q.distance(o) < 14.0);
            if points.is_empty() || short {
                continue;
            }
            let (size, gap) = if tag.0 == "eigen" {
                // (an eigenline's name beside the line, its box clear of the line by 8
                // pixels however the line slants: `Room::label` puts the box's middle
                // `gap` along `towards` and then half its size further)
                let half = f.text_size(&text, 13.0) * 0.5 + Vec2::new(4.0, 0.0);
                let n = towards.normalize_or_zero();
                let reach = half.x * n.x.abs() + half.y * n.y.abs();
                let pushed = half.x * n.x * n.x + half.y * n.y * n.y;
                (13.0, (reach - pushed).max(0.0) + 8.0)
            } else {
                (14.5, 10.0)
            };
            room.label_any(f, tag, &points, &text, size, colour, towards, gap, false);
        }
        plates
    }

    /// The lines along which a handle can move, for a kind that does not leave it free.
    fn tracks(&self, f: &mut Frame, _a: DMat2) {
        let faint = |c: Color| c.a(0.4);
        let long = 3.4f32;
        let (pink, green) = if self.op == Op::Alone && (self.shown().power - 1.0).abs() < 1e-9 { (pal::PINK, pal::GREEN) } else { (pal::WHITE, pal::WHITE) };
        let batch = lines_for(f, Part::Factors);
        match self.kind {
            Kind::Diagonal => {
                batch.segment(Vec3::new(-long, 0.0, 0.0), Vec3::new(long, 0.0, 0.0), 2.0, faint(pink));
                batch.segment(Vec3::new(0.0, -long, 0.0), Vec3::new(0.0, long, 0.0), 2.0, faint(green));
            }
            Kind::Triangular => batch.segment(Vec3::new(-long, 0.0, 0.0), Vec3::new(long, 0.0, 0.0), 2.0, faint(pink)),
            Kind::Skew => {
                batch.segment(Vec3::new(0.0, -long, 0.0), Vec3::new(0.0, long, 0.0), 2.0, faint(pink));
                batch.segment(Vec3::new(-long, 0.0, 0.0), Vec3::new(long, 0.0, 0.0), 2.0, faint(green));
            }
            Kind::Markov => batch.segment(Vec3::new(1.0, 0.0, 0.0), Vec3::new(0.0, 1.0, 0.0), 3.0, faint(pal::TEXT)),
            _ => {}
        }
    }

    /// The heading at the top of the picture: what the map does, and in a second line
    /// what follows for its eigenvalues (or, for two at once, what stays true). Returns
    /// the box it takes.
    fn heading(&mut self, f: &mut Frame, lay: &Layout) -> (Vec2, Vec2) {
        let at = lay.picture.0 + Vec2::new(16.0, 12.0);
        let room = (lay.picture.1.x - lay.picture.0.x - 32.0).max(40.0);
        let (first, second) = self.heading_text(lay.narrow);
        // (a picture too small for more has the first line alone, smaller)
        let tiny = lay.picture.1.y - lay.picture.0.y < 300.0;
        let (big, small) = if tiny {
            (12.0, 11.0)
        } else if lay.narrow {
            (13.5, 11.5)
        } else {
            (15.0, 12.5)
        };
        // (on a phone, what follows for the eigenvalues is left to the panel below)
        let second = if tiny || (lay.narrow && self.op == Op::Alone) { String::new() } else { second };
        let (one, w1) = broken(f, &first, room, big, &mut self.heading[0]);
        let (two, w2) = broken(f, &second, room, small, &mut self.heading[1]);
        let mut y = at.y;
        for line in &one {
            f.caption_px(Vec2::new(at.x, y), line.as_str(), big, pal::TEXT, Align2::LEFT_TOP);
            y += big + 7.0;
        }
        y += 1.0;
        for line in &two {
            f.caption_px(Vec2::new(at.x, y), line.as_str(), small, pal::DIM, Align2::LEFT_TOP);
            y += small + 7.0;
        }
        (at - Vec2::splat(6.0), Vec2::new(at.x + w1.max(w2) + 12.0, y + 2.0))
    }

    /// The heading's two lines.
    fn heading_text(&self, narrow: bool) -> (String, String) {
        let kind = self.kind;
        let to = self.glide.to;
        match self.op {
            Op::Alone => (format!("{}: {}", capitalised(kind.name()), does(kind, to.a)), format!("{}.", capitalised(kind.eigen_words()))),
            op => {
                let n = op.name();
                let first = match self.tested() {
                    None => r"$A$ has no inverse: $\det A = 0$".to_string(),
                    Some(c) => {
                        let verdict = if is(kind, c) { format!("{} too", kind.name()) } else { format!("not {}", kind.name()) };
                        // (on a phone the second line says what it is of)
                        if narrow {
                            format!("${n}$: {verdict}")
                        } else {
                            format!("${n}$, {}: {verdict}", op.of(kind))
                        }
                    }
                };
                (first, kind.closure(op).to_string())
            }
        }
    }

    /// A caption along the bottom of the picture, on a dark plate, naming what has no
    /// name of its own there: `parts` side by side as long as they fit across `(left,
    /// right)`, on further lines above when they do not, the last ending at `bottom`.
    /// Returns the box it takes (none on a canvas too small to spare the room).
    fn caption(&mut self, f: &mut Frame, parts: &[&str], (left, right): (f32, f32), bottom: f32) -> Option<(Vec2, Vec2)> {
        let size: f32 = 12.0;
        if f.view_size().y < 300.0 {
            return None;
        }
        let room = right - left - 36.0;
        let of = (parts.join("|"), room.to_bits(), size.to_bits());
        let (lines, widest) = match self.caption.as_ref().filter(|k| k.of == of) {
            Some(kept) => (kept.lines.clone(), kept.widest),
            None => match caption_lines(parts, room, |t| measured(f, t, size)) {
                Some((lines, widest)) => {
                    self.caption = Some(Broken { of, lines: lines.clone(), widest });
                    (lines, widest)
                }
                None => caption_lines(parts, room, |t| Some(text_width(t, size))).unwrap_or_default(),
            },
        };
        let count = lines.len();
        for (k, line) in lines.into_iter().enumerate() {
            let up = (count - 1 - k) as f32 * (size + 7.0);
            f.caption_px(Vec2::new(left + 18.0, bottom - up), line, size, pal::DIM, Align2::LEFT_BOTTOM);
        }
        Some((Vec2::new(left + 10.0, bottom - count as f32 * (size + 7.0) - 4.0), Vec2::new(left + 26.0 + widest, bottom + 6.0)))
    }

    /// The screen panel: the kinds, the equation with both sides worked out, the
    /// matrices, the complex plane with the eigenvalues, and buttons.
    fn panel(&mut self, f: &mut Frame, p: &Panel) {
        hud_panel(f, p.min, p.max);
        let kind = self.kind;
        let to = self.glide.to;
        let tested = self.tested();

        // the kinds: the one chosen outlined, every kind the matrix is bright
        let mut clicked = None;
        for (k, rect) in KINDS.iter().zip(&p.chips) {
            let member = tested.is_some_and(|c| is(*k, c));
            if chip(f, *rect, k.chip(), *k == kind, member) {
                clicked = Some(*k);
            }
        }
        let n = self.op.name();
        let mid = p.title.y + TITLE_H * 0.5;
        f.text_px(p.note, chips_note(self.op), NOTE_SIZE, pal::DIM, Align2::LEFT_CENTER);

        // the equation, the matrices, and the sides worked out
        f.text_px(Vec2::new(p.title.x, mid), format!("{}: {}", kind.name(), kind.equation(n)), 14.0, pal::TEXT, Align2::LEFT_CENTER);
        let mut cells: Vec<(String, DMat2, [Color; 2])> = vec![(String::from("$A$"), to.a, if self.op == Op::Alone { [pal::PINK, pal::GREEN] } else { [pal::TEXT, pal::TEXT] })];
        if self.op.uses_b() {
            cells.push(("$B$".into(), to.b, [pal::BLUE.mix(pal::TEXT, 0.25); 2]));
        }
        if let (true, Some(c)) = (self.op != Op::Alone, tested) {
            cells.push((format!("${n}$"), c, [pal::PINK, pal::GREEN]));
        }
        if let Some(c) = tested {
            for (name, m) in sides(kind, c, n) {
                cells.push((name, m, [pal::TEXT, pal::TEXT]));
            }
        }
        let s = p.text;
        for (i, (name, m, colours)) in cells.iter().enumerate() {
            let (row, col) = (i / p.per_row, i % p.per_row);
            let at = p.cells + Vec2::new(col as f32 * p.step.x, row as f32 * p.step.y);
            let w = matrix_w(s);
            f.text_px(at + Vec2::new(w * 0.5, LABEL_H * 0.5), name.as_str(), 12.5, pal::TEXT, Align2::CENTER_CENTER);
            let rows: Vec<Vec<(String, Color)>> = (0..2).map(|r| (0..2).map(|c| (cell(m.col(c)[r]), colours[c])).collect()).collect();
            matrix(f, at + Vec2::new(0.0, LABEL_H), &rows, s);
        }
        // the verdict, under the matrices
        let mut y = p.cells.y + cells.len().div_ceil(p.per_row) as f32 * p.step.y - 2.0;
        let line = |f: &mut Frame, y: &mut f32, text: String, colour: Color| {
            if *y + LINE_H <= p.lines_end + 1.0 {
                f.text_px(Vec2::new(p.cells.x, *y + LINE_H * 0.5), text, 12.0, colour, Align2::LEFT_CENTER);
            }
            *y += LINE_H;
        };
        match tested {
            Some(c) => {
                let (first, holds, second) = verdict(kind, c, n);
                line(f, &mut y, first, if holds { pal::TEXT } else { pal::RED });
                if let Some(second) = second {
                    line(f, &mut y, second, pal::DIM);
                }
                if self.show_x {
                    let m = powered(c, to.power);
                    let name = self.drawn_name(to.power);
                    let dot = to.x.dot(m * to.x);
                    let text = if name.is_empty() { format!(r"$x \cdot x = {}$ at the white point", num(dot, 2)) } else { format!(r"$x \cdot {}x = {}$ at the white point", paren(&name), num(dot, 2)) };
                    line(f, &mut y, text, pal::DIM);
                }
            }
            None => line(f, &mut y, r"$A$ has no inverse: $\det A = 0$".into(), pal::RED),
        }

        // whether a sum, a product and an inverse are of the kind too
        if let Some(at) = p.table {
            f.text_px(at + Vec2::new(0.0, TABLE_LINE * 0.5), format!("of the kind too? (two {})", plural(kind)), 11.5, pal::TEXT, Align2::LEFT_CENTER);
            for (k, (what, answer)) in ["$A + B$", "$AB$", "$A^{-1}$"].iter().zip(kind.closure_short()).enumerate() {
                let y = at.y + (k + 1) as f32 * TABLE_LINE + TABLE_LINE * 0.5;
                let colour = if matches!(answer, "always" | "where it exists") { pal::TEXT } else { pal::DIM };
                f.text_px(Vec2::new(at.x + 8.0, y), *what, 11.5, pal::DIM, Align2::LEFT_CENTER);
                f.text_px(Vec2::new(at.x + 62.0, y), answer, 11.5, colour, Align2::LEFT_CENTER);
            }
        }

        // the complex plane
        self.chart(f, p);

        // the buttons
        let label = if self.op.uses_b() { "another $A$" } else { "another of this kind" };
        if plot::button(f, p.another, label, 12.0, false) {
            self.another(false);
        }
        if self.op.uses_b() && plot::button(f, p.another_b, "another $B$", 12.0, false) {
            self.another(true);
        }
        if let Some(k) = clicked {
            self.choose(k);
        }
    }

    /// The complex plane: the region where the kind's eigenvalues can lie, and the
    /// eigenvalues as dots (of A and B as rings, for two at once).
    fn chart(&self, f: &mut Frame, p: &Panel) {
        let (min, max) = p.chart;
        let kind = self.kind;
        let shown = self.shown();
        let title = if self.op == Op::Alone { "eigenvalues, in the complex plane".to_string() } else { format!("eigenvalues of ${}$ (dots), of $A$ and $B$ (rings)", self.op.name()) };
        let title = if self.op == Op::Inverse { r"eigenvalues of $A^{-1}$ (dots), of $A$ (rings)".to_string() } else { title };
        f.text_px(p.chart_title + Vec2::new(0.0, LINE_H * 0.5), title, 11.5, pal::TEXT, Align2::LEFT_CENTER);
        hud_rect(f, min, max, Color::hex(0x10172A));
        // one unit the same across and up, the middle at 0
        let (w, h) = (max.x - min.x, max.y - min.y);
        let unit = (w / 5.6).min(h / 3.4).max(1.0) as f64;
        let (hx, hy) = (w as f64 * 0.5 / unit, h as f64 * 0.5 / unit);
        let plot = Plot::new(p.chart, [-hx, hx], [-hy, hy]);
        // the axes, the unit circle, and the whole numbers along them
        plot::line_of(f, plot::AXES, plot.at(-hx, 0.0), plot.at(hx, 0.0), 1.0, pal::DIM.a(0.7));
        plot::line_of(f, plot::AXES, plot.at(0.0, -hy), plot.at(0.0, hy), 1.0, pal::DIM.a(0.7));
        let ring: Vec<Vec2> = (0..=72).map(|i| {
            let t = i as f64 / 72.0 * std::f64::consts::TAU;
            plot.at(t.cos(), t.sin())
        }).collect();
        for pair in ring.windows(2).step_by(2) {
            plot::line_of(f, plot::AXES, pair[0], pair[1], 1.0, pal::DIM.a(0.45));
        }
        for k in [-2i32, -1, 1, 2] {
            let x = k as f64;
            if x.abs() < hx - 0.2 {
                plot::line_of(f, plot::AXES, plot.at(x, 0.0) - Vec2::new(0.0, 3.0), plot.at(x, 0.0) + Vec2::new(0.0, 3.0), 1.0, pal::DIM.a(0.8));
                f.text_px(plot.at(x, 0.0) + Vec2::new(0.0, 5.0), format!("${}$", num(x, 0)), 10.0, pal::DIM, Align2::CENTER_TOP);
            }
            if x.abs() < hy - 0.15 {
                plot::line_of(f, plot::AXES, plot.at(0.0, x) - Vec2::new(3.0, 0.0), plot.at(0.0, x) + Vec2::new(3.0, 0.0), 1.0, pal::DIM.a(0.8));
                let name = match k {
                    1 => "$i$".to_string(),
                    -1 => "$-i$".to_string(),
                    k => format!("${k}i$").replace('-', "−"),
                };
                f.text_px(plot.at(0.0, x) + Vec2::new(5.0, 0.0), name, 10.0, pal::DIM, Align2::LEFT_CENTER);
            }
        }
        // where this kind's eigenvalues can lie
        let bright = Color::hex(0xC8D4F0).a(0.42);
        let band = 5.0;
        match kind.region() {
            Region::Real => plot::line_of(f, plot::MARKS, plot.at(-hx, 0.0), plot.at(hx, 0.0), band, bright),
            Region::Positive => plot::line_of(f, plot::MARKS, plot.at(0.0, 0.0) + Vec2::new(4.0, 0.0), plot.at(hx, 0.0), band, bright),
            Region::Imaginary => plot::line_of(f, plot::MARKS, plot.at(0.0, -hy), plot.at(0.0, hy), band, bright),
            Region::Circle => plot::path_of(f, plot::MARKS, &ring, band, bright),
            Region::Points(points) => {
                for x in points {
                    plot::disc(f, plot.at(*x, 0.0), 18.0, bright);
                }
            }
            Region::Shares => {
                plot::line_of(f, plot::MARKS, plot.at(-1.0, 0.0), plot.at(1.0, 0.0), band, bright);
                plot::disc(f, plot.at(1.0, 0.0), 18.0, bright);
            }
            Region::Anywhere => plot::fill(f, p.chart, bright.a(0.1)),
        }
        // the region's name, in a corner
        f.text_px(Vec2::new(max.x - 5.0, min.y + 4.0), kind.region().name(), 10.5, pal::TEXT.a(0.8), Align2::RIGHT_TOP);
        // the eigenvalues: of A and B as rings, of the matrix under test as dots
        let at = |(re, im): (f64, f64)| {
            let q = plot.at(re, im);
            let inside = within_chart(q, p.chart);
            (q.max(min + Vec2::splat(3.0)).min(max - Vec2::splat(3.0)), inside)
        };
        if self.op != Op::Alone {
            let mut rings = vec![(shown.a, pal::WHITE)];
            if self.op.uses_b() {
                rings.push((shown.b, pal::BLUE));
            }
            for (m, colour) in rings {
                for l in eigenvalues(eigen(m)) {
                    let (q, _) = at(l);
                    hud_circle(f, q, 5.0, 1.5, colour.a(0.9));
                }
            }
        }
        match combine(self.op, shown.a, shown.b) {
            Some(c) => {
                let e = eigen(c);
                let tol = 1e-6 * size(c).max(1.0);
                for (l, colour) in eigenvalues(e).into_iter().zip([pal::ORANGE, pal::PURPLE]).rev() {
                    let (q, inside) = at(l);
                    // (one where this kind's cannot be is ringed in red)
                    if !kind.region().holds(l, tol) {
                        hud_circle(f, q, 9.0, 1.6, pal::RED);
                    }
                    plot::disc(f, q, if inside { 10.0 } else { 7.0 }, colour);
                }
                let tested = self.tested().map(eigen).unwrap_or(e);
                f.text_px(p.eigen_line + Vec2::new(0.0, LINE_H * 0.5), eigen_text(tested), 12.5, pal::TEXT, Align2::LEFT_CENTER);
            }
            None => f.text_px(p.eigen_line + Vec2::new(0.0, LINE_H * 0.5), "no inverse, so no eigenvalues to show", 12.0, pal::DIM, Align2::LEFT_CENTER),
        }
    }
}

fn within_chart(q: Vec2, (min, max): (Vec2, Vec2)) -> bool {
    q.is_finite() && q.cmpge(min).all() && q.cmple(max).all()
}

/// A number for a matrix on the screen, as a formula, in five characters at most:
/// two decimals, then one, then none, and a power of ten beyond 9999.
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

/// A matrix of numbers on the panel, as `hud_matrix` draws it, with a number too wide for
/// its cell drawn smaller, to fit: the brackets stay where they are whatever the numbers.
fn matrix(f: &mut Frame, top_left: Vec2, rows: &[Vec<(String, Color)>], size: f32) -> Vec2 {
    let room = matrix_cell(CELL, size).x - 2.0;
    let mut fitted = rows.to_vec();
    let mut smaller = Vec::new();
    for (i, row) in fitted.iter_mut().enumerate() {
        for (j, (text, color)) in row.iter_mut().enumerate() {
            if !text.contains('e') {
                continue;
            }
            let wide = measured(f, text, size).unwrap_or_else(|| text_width(text, size));
            if wide > room {
                smaller.push((i, j, std::mem::take(text), *color, size * room / wide));
            }
        }
    }
    let total = hud_matrix(f, top_left, &fitted, CELL, size, pal::DIM);
    for (i, j, text, color, s) in smaller {
        let (lo, hi) = matrix_entry(top_left, CELL, size, i, j);
        f.text_px((lo + hi) * 0.5, text, s, color, Align2::CENTER_CENTER);
    }
    total
}

/// One of the kinds' buttons: outlined if it is the kind chosen, its name bright if the
/// matrix is of that kind. Returns true in the frame it is clicked.
fn chip(f: &mut Frame, (min, max): (Vec2, Vec2), text: &str, on: bool, member: bool) -> bool {
    let over = plot::pointer(f).is_some_and(|q| plot::within(q, (min, max), 0.0));
    let backing = match (on, over) {
        (true, _) => Color::hex(0x2B3D66),
        (false, true) => Color::hex(0x2A3350),
        _ => Color::hex(0x161C2B),
    };
    hud_rect(f, min, max, backing);
    let edge = [min, Vec2::new(max.x, min.y), max, Vec2::new(min.x, max.y), min];
    plot::path_of(f, plot::BUTTONS, &edge, if on { 1.4 } else { 1.0 }, pal::LINK.a(if on { 0.95 } else if over { 0.7 } else { 0.3 }));
    let colour = if member { pal::WHITE } else { pal::DIM.a(0.8) };
    f.text_px((min + max) * 0.5, text, CHIP_SIZE, colour, Align2::CENTER_CENTER);
    if over {
        f.handles.claim_clicks();
        f.handles.capture();
    }
    let clicked = over && f.ptr.clicked;
    if clicked {
        f.ptr.clicked = false;
    }
    clicked
}

#[cfg(test)]
mod tests {
    use super::*;

    fn close(m: DMat2, n: DMat2, tol: f64) -> bool {
        (m - n).to_cols_array().iter().all(|v| v.abs() <= tol)
    }

    /// A kind's own two matrices and `n` more at random.
    fn members(kind: Kind, n: usize, seed: u64) -> Vec<DMat2> {
        let mut rng = Rng::new(seed);
        let mut out = vec![kind.default_a(), kind.default_b()];
        out.extend((0..n).map(|_| random(kind, &mut rng)));
        out
    }

    fn commute(a: DMat2, b: DMat2) -> bool {
        size(a * b - b * a) <= 1e-9 * size(a).max(1.0) * size(b).max(1.0)
    }

    /// The scene as an experiment's button leaves it, run at 60 frames a second until
    /// everything has come to rest.
    fn run(k: usize) -> Special {
        let mut scene = Special::new();
        scene.set_up(k);
        for _ in 0..60 * 30 {
            scene.advance(1.0 / 60.0, 1.0 / 60.0);
            if !scene.animated() {
                break;
            }
        }
        assert!(!scene.animated(), "experiment {k} is still moving");
        scene
    }

    /// The signed area of a polygon, positive if its corners run anticlockwise.
    fn area(corners: &[DVec2]) -> f64 {
        (0..corners.len()).map(|i| corners[i].perp_dot(corners[(i + 1) % corners.len()])).sum::<f64>() * 0.5
    }

    /// Every matrix a kind starts with, makes at random, or is dragged to is of that
    /// kind, finite, and within the limits, wherever the column is dragged: on the axes,
    /// at 0, beyond the limits, nowhere at all.
    #[test]
    fn every_kind_keeps_its_matrices_in_it() {
        let mut rng = Rng::new(5);
        for kind in KINDS {
            assert!(is(kind, kind.default_a()) && is(kind, kind.default_b()), "{kind:?}");
            for a in members(kind, 200, 11) {
                assert!(is(kind, a) && size(a) <= ENTRY_MAX, "{kind:?}: {a:?}");
                let awkward = [DVec2::ZERO, DVec2::X, DVec2::Y, DVec2::new(0.4, 0.0), DVec2::new(0.0, -0.7), DVec2::new(-2.0, 1e-12), DVec2::new(1.0, 1e-300), DVec2::splat(9.0), DVec2::splat(-9.0), DVec2::NAN, DVec2::new(f64::INFINITY, 1.0)];
                let random = (0..20).map(|_| DVec2::new(rng.range(-4.0, 4.0) as f64, rng.range(-4.0, 4.0) as f64));
                for p in awkward.into_iter().chain(random) {
                    for j in 0..2 {
                        let m = fit(kind, a, j, p);
                        assert!(m.is_finite() && size(m) <= ENTRY_MAX + 1e-12 && is(kind, m), "{kind:?}: {a:?}, column {j} dragged to {p:?}, gives {m:?}");
                    }
                }
            }
        }
    }

    /// The column dragged goes where it is dragged when the kind allows that, and
    /// otherwise to the nearest place it allows; and a drag in small steps moves the
    /// matrix in small steps.
    #[test]
    fn a_dragged_column_goes_as_near_as_the_kind_allows() {
        let mut rng = Rng::new(9);
        for _ in 0..500 {
            let p = DVec2::new(rng.range(-2.8, 2.8) as f64, rng.range(-2.8, 2.8) as f64);
            let sym = Kind::Symmetric.default_a();
            assert!(fit(Kind::Symmetric, sym, 0, p).x_axis == p && fit(Kind::Symmetric, sym, 1, p).y_axis == p);
            let tri = Kind::Triangular.default_a();
            assert!(fit(Kind::Triangular, tri, 1, p).y_axis == p && fit(Kind::Triangular, tri, 0, p).x_axis == DVec2::new(p.x, 0.0));
            let diag = Kind::Diagonal.default_a();
            assert!(fit(Kind::Diagonal, diag, 0, p).x_axis == DVec2::new(p.x, 0.0) && fit(Kind::Diagonal, diag, 1, p).y_axis == DVec2::new(0.0, p.y));
            assert!(fit(Kind::Skew, Kind::Skew.default_a(), 0, p).x_axis == DVec2::new(0.0, p.y));
            for a in [Kind::Orthogonal.default_a(), Kind::Orthogonal.default_b()] {
                for j in 0..2 {
                    let m = fit(Kind::Orthogonal, a, j, p);
                    assert!((m.col(j) - p.normalize()).length() < 1e-12 && m.determinant().signum() == a.determinant().signum());
                }
            }
            // the nearest point of the segment of shares
            let m = fit(Kind::Markov, Kind::Markov.default_a(), 0, p);
            let c = m.x_axis;
            assert!(c.x >= 0.0 && c.y >= 0.0 && (c.x + c.y - 1.0).abs() < 1e-12);
            for t in 0..=20 {
                let q = DVec2::new(t as f64 / 20.0, 1.0 - t as f64 / 20.0);
                assert!((p - c).length() <= (p - q).length() + 1e-12);
            }
            // where the rank-one kinds and the positive definite ones are free to go
            let pd = Kind::PosDef.default_a();
            if p.x >= 0.1 && p.y.abs() <= ((p.x - LEAST_POSITIVE) * (ENTRY_MAX - LEAST_POSITIVE)).sqrt() {
                assert!(fit(Kind::PosDef, pd, 0, p).x_axis == p);
            }
            if p.x >= 1.0 - ENTRY_MAX && p.y.abs() >= (p.x * (1.0 - p.x)).abs() / ENTRY_MAX {
                assert!(fit(Kind::Projection, Kind::Projection.default_a(), 0, p).x_axis == p);
            }
            if p.y.abs() >= p.x * p.x / ENTRY_MAX {
                assert!(fit(Kind::Nilpotent, Kind::Nilpotent.default_a(), 0, p).x_axis == p);
            }
            let q = p - DVec2::X;
            if q.x.abs() <= ENTRY_MAX - 1.0 && q.y.abs() <= ENTRY_MAX - 1.0 && q.y.abs() >= q.x * q.x / (ENTRY_MAX - 1.0) {
                assert!((fit(Kind::Shear, Kind::Shear.default_a(), 0, p).x_axis - p).length() < 1e-12);
            }
        }
        // a drag round a loop clear of the axes, in small steps: no jumps
        for kind in KINDS {
            for j in 0..2 {
                let mut a = kind.default_a();
                for i in 0..=2000 {
                    let p = DVec2::new(0.8, 0.9) + DVec2::from_angle(i as f64 * std::f64::consts::TAU / 2000.0) * 0.5;
                    let next = fit(kind, a, j, p);
                    if i > 0 {
                        assert!(size(next - a) < 0.05, "{kind:?}, column {j}: a jump at step {i}: {a:?} to {next:?}");
                    }
                    a = next;
                }
            }
        }
    }

    /// Each kind's eigenvalues lie where the picture says they can, and the eigenlines
    /// are what the panel says.
    #[test]
    fn the_eigenvalues_lie_where_each_kind_allows() {
        let mut rng = Rng::new(21);
        for kind in KINDS {
            for a in members(kind, 300, 31) {
                let tol = 1e-9 * size(a).max(1.0);
                let e = eigen(a);
                for l in eigenvalues(e) {
                    assert!(kind.region().holds(l, tol), "{kind:?}: {a:?} has the eigenvalue {l:?}, not on {}", kind.region().name());
                }
                let [a11, _, a21, a22] = entries(a);
                match kind {
                    Kind::Diagonal | Kind::Triangular => {
                        let [(l1, _), (l2, _)] = eigenvalues(e);
                        assert!(a21 == 0.0 && ((l1 - a11).abs() + (l2 - a22).abs() < 1e-12 || (l1 - a22).abs() + (l2 - a11).abs() < 1e-12), "{a:?}");
                    }
                    Kind::Symmetric | Kind::PosDef => match e {
                        Eigen::Real { lines: Lines::Two(v1, v2), .. } => assert!(v1.dot(v2).abs() < 1e-9, "{a:?}"),
                        Eigen::Real { lines: Lines::Every, .. } => {}
                        _ => panic!("{kind:?}: {a:?} {e:?}"),
                    },
                    Kind::Skew => {
                        for _ in 0..10 {
                            let x = DVec2::new(rng.range(-3.0, 3.0) as f64, rng.range(-3.0, 3.0) as f64);
                            assert!(x.dot(a * x).abs() < 1e-12, "{a:?}");
                        }
                        assert!(close(a, J * a21, 0.0), "in the plane a skew matrix is wJ: {a:?}");
                    }
                    Kind::Orthogonal => {
                        if a.determinant() < 0.0 {
                            assert!(matches!(e, Eigen::Real { l1, l2, lines: Lines::Two(v1, v2) } if (l1 - 1.0).abs() < 1e-12 && (l2 + 1.0).abs() < 1e-12 && v1.dot(v2).abs() < 1e-9), "{a:?} {e:?}");
                        }
                        for _ in 0..10 {
                            let x = DVec2::new(rng.range(-3.0, 3.0) as f64, rng.range(-3.0, 3.0) as f64);
                            assert!(((a * x).length() - x.length()).abs() < 1e-12);
                        }
                    }
                    Kind::Normal => match e {
                        Eigen::Real { lines: Lines::Two(v1, v2), .. } => assert!(v1.dot(v2).abs() < 1e-9, "{a:?}"),
                        Eigen::Real { lines: Lines::Every, .. } => {}
                        Eigen::Real { lines: Lines::One(_), .. } => panic!("a normal matrix with one eigenline: {a:?}"),
                        // the complex eigenvectors (a₁₂, λ − a₁₁) and their conjugates are
                        // at right angles: v₁² + v₂² = 0 (as complex numbers)
                        Eigen::Complex { re, im } => {
                            let (v1, v2) = ((a.y_axis.x, 0.0), (re - a11, im));
                            let square = (v1.0 * v1.0 + v2.0 * v2.0 - v2.1 * v2.1, 2.0 * v2.0 * v2.1);
                            assert!(square.0.abs() + square.1.abs() < 1e-9 * (1.0 + v1.0 * v1.0 + im * im), "{a:?}: {square:?}");
                        }
                    },
                    Kind::Projection => assert!(close(a * a, a, 1e-12)),
                    Kind::Nilpotent | Kind::Shear => {
                        let Eigen::Real { l1, lines: Lines::One(v), .. } = e else { panic!("{kind:?} {a:?} {e:?}") };
                        let keep = if kind == Kind::Shear { 1.0 } else { 0.0 };
                        assert!((l1 - keep).abs() < 1e-12 && (a * v - v * keep).length() < 1e-9, "{a:?}");
                        // everything lands on that line (nilpotent), or slides along it (shear)
                        for _ in 0..10 {
                            let x = DVec2::new(rng.range(-3.0, 3.0) as f64, rng.range(-3.0, 3.0) as f64);
                            assert!((a * x - x * keep).perp_dot(v).abs() < 1e-9, "{a:?}");
                        }
                        if kind == Kind::Shear {
                            assert!((a.determinant() - 1.0).abs() < 1e-9);
                        }
                    }
                    Kind::Markov => {
                        let Eigen::Real { l1, l2, .. } = e else { panic!("{a:?}") };
                        assert!((l1 - 1.0).abs() < 1e-12 && (l2 - (trace(a) - 1.0)).abs() < 1e-12 && (-1.0 - 1e-12..=1.0 + 1e-12).contains(&l2), "{a:?}");
                    }
                }
            }
        }
        // a rotation's eigenvalues are a complex pair on the unit circle
        for angle in [0.3, 1.0, 2.0, -2.5] {
            let Eigen::Complex { re, im } = eigen(DMat2::from_angle(angle)) else { panic!() };
            assert!((re - angle.cos()).abs() < 1e-12 && (im - angle.sin().abs()).abs() < 1e-12);
        }
    }

    /// What the panel and the table say stays true of each kind under a sum, a product
    /// and an inverse, tried on many pairs.
    #[test]
    fn what_stays_true_is_what_the_panel_says() {
        let mut rng = Rng::new(41);
        let pairs = |kind: Kind, rng: &mut Rng| -> Vec<(DMat2, DMat2)> {
            let mut out = vec![(kind.default_a(), kind.default_b())];
            out.extend((0..300).map(|_| (random(kind, rng), random(kind, rng))));
            out
        };
        let id = DMat2::IDENTITY;
        let swap = SWAP;
        for kind in KINDS {
            let all = pairs(kind, &mut rng);
            let [sum, product, inverse_short] = kind.closure_short();
            let holds = |op: Op, a: DMat2, b: DMat2| combine(op, a, b).map(|c| is(kind, c));
            let count = |op: Op| all.iter().filter(|(a, b)| holds(op, *a, *b) == Some(true)).count();
            // sums
            match sum {
                "always" => assert_eq!(count(Op::Sum), all.len(), "{kind:?}"),
                "never" => assert_eq!(count(Op::Sum), 0, "{kind:?}"),
                _ => {}
            }
            // products
            if product == "always" {
                assert_eq!(count(Op::Product), all.len(), "{kind:?}");
            }
            // inverses
            for (a, _) in &all {
                let inv = inverse(*a);
                match inverse_short {
                    "always" => assert!(inv.is_some_and(|m| is(kind, m)), "{kind:?}: {a:?}"),
                    "where it exists" => assert!(inv.is_none_or(|m| is(kind, m)), "{kind:?}: {a:?}"),
                    "never" => assert!(inv.is_none(), "{kind:?}: {a:?}"),
                    _ => {}
                }
            }
            match kind {
                Kind::Diagonal => {
                    assert!(all.iter().all(|(a, b)| close(*a * *b, *b * *a, 0.0)));
                    for (a, _) in &all {
                        let [p, _, _, s] = entries(*a);
                        assert!(inverse(*a).is_some_and(|m| close(m, rows([1.0 / p, 0.0, 0.0, 1.0 / s]), 1e-12)));
                    }
                }
                Kind::Symmetric | Kind::PosDef => {
                    // exactly when they commute: pairs that do not, and pairs built to
                    for (a, b) in &all {
                        assert_eq!(is(kind, *a * *b), commute(*a, *b), "{kind:?}: {a:?} {b:?}");
                        // (the eigenvalues of a product of positive definite ones are positive either way)
                        if kind == Kind::PosDef {
                            assert!(eigenvalues(eigen(*a * *b)).iter().all(|(re, im)| *im == 0.0 && *re > 0.0), "{a:?} {b:?}");
                        }
                    }
                    for _ in 0..100 {
                        let q = DMat2::from_angle(rng.range(0.0, 3.2) as f64);
                        let low = if kind == Kind::PosDef { 0.2 } else { -1.5 };
                        let d = |rng: &mut Rng| DMat2::from_diagonal(DVec2::new(rng.range(low, 1.5) as f64, rng.range(low, 1.5) as f64));
                        let (a, b) = (q * d(&mut rng) * q.transpose(), q * d(&mut rng) * q.transpose());
                        assert!(is(kind, a) && is(kind, b) && commute(a, b) && is(kind, a * b), "{a:?} {b:?}");
                    }
                    assert!(count(Op::Product) < all.len());
                }
                Kind::Skew => {
                    for (a, b) in &all {
                        let p = *a * *b;
                        assert!(close(p, id * -(a.x_axis.y * b.x_axis.y), 1e-12) && is(Kind::Symmetric, p));
                        assert_eq!(is(kind, p), size(p) <= 1e-9, "{a:?} {b:?}");
                    }
                    assert!(is(kind, J * 0.0 * J));
                    for (a, _) in &all {
                        let w = a.x_axis.y;
                        assert!(inverse(*a).is_some_and(|m| close(m, J * (-1.0 / w), 1e-12)));
                    }
                }
                Kind::Orthogonal => {
                    // not always: I + I doubles lengths, yet a turn by 0 and one by 120° add to a turn
                    assert!(!is(kind, id + id) && is(kind, DMat2::from_angle(0.0) + DMat2::from_angle(std::f64::consts::TAU / 3.0)));
                    for (a, _) in &all {
                        assert!(close(inverse(*a).unwrap(), a.transpose(), 1e-12));
                    }
                }
                Kind::Normal => {
                    // not always, either way: a symmetric matrix plus a turn and a scaling,
                    // or times one, need not be normal; two symmetric ones add to one that is
                    let (s, t) = (rows([2.0, 0.0, 0.0, 1.0]), J);
                    assert!(is(kind, s) && is(kind, t) && !is(kind, s + t) && !is(kind, s * t));
                    assert!(is(kind, s + s) && is(kind, t * t) && count(Op::Sum) < all.len() && count(Op::Product) < all.len());
                }
                Kind::Projection => {
                    // a sum exactly when AB = BA = 0; a product if they commute, not always
                    for (a, b) in &all {
                        let zero = size(*a * *b) <= 1e-9 && size(*b * *a) <= 1e-9;
                        assert_eq!(is(kind, *a + *b), zero, "{a:?} {b:?}");
                        if commute(*a, *b) {
                            assert!(is(kind, *a * *b));
                        }
                        assert!(inverse(*a).is_none());
                    }
                    let p = kind.default_a();
                    assert!(is(kind, p + (id - p)) && is(kind, p * (id - p)) && is(kind, p * p) && is(kind, id) && inverse(id) == Some(id));
                    assert!(!is(kind, kind.default_a() * kind.default_b()));
                }
                Kind::Nilpotent => {
                    // not always: along different lines no, along the same line yes
                    let (n, m) = (kind.default_a(), kind.default_b());
                    assert!(!is(kind, n + m) && !is(kind, n * m) && is(kind, n + n * 0.5) && is(kind, n * (n * 0.5)));
                }
                Kind::Shear => {
                    for (a, b) in &all {
                        assert!((trace(*a + *b) - 4.0).abs() < 1e-9);
                        // along the same line: a shear, or I; along different lines: not
                        let same = matches!((eigen(*a), eigen(*b)), (Eigen::Real { lines: Lines::One(u), .. }, Eigen::Real { lines: Lines::One(v), .. }) if u.perp_dot(v).abs() < 1e-9);
                        assert_eq!(is(kind, *a * *b) || close(*a * *b, id, 1e-9), same, "{a:?} {b:?}");
                        let n = *a - id;
                        assert!(close(inverse(*a).unwrap(), id - n, 1e-12));
                    }
                    let a = kind.default_a();
                    assert!(is(kind, a * fit(kind, a, 1, DVec2::new(-0.4, 1.0))) && close(a * (id * 2.0 - a), id, 1e-12));
                }
                Kind::Markov => {
                    assert_eq!(count(Op::Sum), 0);
                    for (a, b) in &all {
                        assert!(is(kind, (*a + *b) * 0.5));
                        if let Some(inv) = inverse(*a) {
                            let [p, q, r, s] = entries(inv);
                            assert!((p + r - 1.0).abs() < 1e-9 && (q + s - 1.0).abs() < 1e-9 && p.min(q).min(r).min(s) < 0.0, "{a:?}");
                        }
                    }
                    assert!(is(kind, inverse(id).unwrap()) && is(kind, inverse(swap).unwrap()) && is(kind, swap));
                }
                Kind::Triangular => {}
            }
        }
    }

    /// The captions of the "things to try", held to account.
    #[test]
    fn symmetric_right_angles_does_what_its_caption_says() {
        let scene = run(0);
        let a = scene.shown().a;
        assert!(scene.kind == Kind::Symmetric && scene.op == Op::Alone && is(Kind::Symmetric, a));
        let Eigen::Real { l1, l2, lines: Lines::Two(v1, v2) } = eigen(a) else { panic!("{:?}", eigen(a)) };
        assert!((l1 - 2.0).abs() < 1e-12 && (l2 - 0.5).abs() < 1e-12 && v1.dot(v2).abs() < 1e-12);
        // the ellipse's axes along the eigenlines, stretched by 2 and 0.5
        let lengths: Vec<f64> = unit_circle().iter().map(|p| (a * *p).length()).collect();
        let (most, least) = lengths.iter().fold((0.0f64, f64::MAX), |(m, l), v| (m.max(*v), l.min(*v)));
        assert!((most - 2.0).abs() < 1e-3 && (least - 0.5).abs() < 1e-3 && ((a * v1).length() - 2.0).abs() < 1e-12);
        // the loop: at every frame A is symmetric (the green column following the pink),
        // its eigenlines at right angles, and the pink column where the loop has it
        let mut scene = Special::new();
        scene.set_up(0);
        let start = scene.glide.to.a;
        let mut moved = 0.0f64;
        let mut frames = 0;
        while scene.animated() && frames < 60 * 30 {
            scene.advance(1.0 / 60.0, 1.0 / 60.0);
            frames += 1;
            let a = scene.shown().a;
            assert!(is(Kind::Symmetric, a) && a.y_axis.x == a.x_axis.y);
            if let Eigen::Real { lines: Lines::Two(v1, v2), .. } = eigen(a) {
                assert!(v1.dot(v2).abs() < 1e-9);
            }
            if let Some((s, from)) = scene.tour {
                assert!((a.x_axis - loop_at(from, s)).length() < 1e-12);
            }
            moved = moved.max(size(a - start));
        }
        assert!(moved > 0.5 && close(scene.shown().a, start, 1e-9), "{moved}");
    }

    #[test]
    fn the_mirror_does_what_its_caption_says() {
        let scene = run(1);
        let a = scene.shown().a;
        assert!(is(Kind::Orthogonal, a) && (a.determinant() + 1.0).abs() < 1e-12);
        // the circle lands on itself, and the F keeps its shape but is mirrored
        assert!(unit_circle().iter().all(|p| ((a * *p).length() - 1.0).abs() < 1e-12));
        assert!(area(&F_OUTLINE) > 0.0 && (area(&F_OUTLINE.map(|p| a * p)) + area(&F_OUTLINE)).abs() < 1e-12);
        // the mirror line through (2, 1) stays put; the line across it is reversed
        let Eigen::Real { l1, l2, lines: Lines::Two(v1, v2) } = eigen(a) else { panic!() };
        assert!((l1 - 1.0).abs() < 1e-12 && (l2 + 1.0).abs() < 1e-12);
        assert!(v1.perp_dot(DVec2::new(2.0, 1.0)).abs() < 1e-12 && v2.perp_dot(DVec2::new(-1.0, 2.0)).abs() < 1e-12);
        assert!(eigenvalues(eigen(a)).iter().all(|l| Region::Circle.holds(*l, 1e-12)));
        assert_eq!(does(Kind::Orthogonal, a), "reflects the plane in the line at 26.6° to the $x$ axis");
    }

    #[test]
    fn the_skew_matrix_does_what_its_caption_says() {
        let scene = run(2);
        let s = scene.shown();
        assert!(scene.show_x && is(Kind::Skew, s.a) && close(s.a, J * 1.2, 0.0));
        let mut rng = Rng::new(2);
        for _ in 0..200 {
            let x = DVec2::new(rng.range(-4.0, 4.0) as f64, rng.range(-4.0, 4.0) as f64);
            assert!(x.dot(s.a * x).abs() < 1e-12);
        }
        assert!(s.x.dot(s.a * s.x) == 0.0);
        assert_eq!(eigen(s.a), Eigen::Complex { re: 0.0, im: 1.2 });
        assert_eq!(eigen_text(eigen(s.a)), r"$\lambda = \pm 1.20i$");
    }

    #[test]
    fn the_nilpotent_steps_do_what_their_caption_says() {
        let scene = run(3);
        let s = scene.shown();
        let a = s.a;
        assert!(s.power == 2.0 && is(Kind::Nilpotent, a) && size(a) > 0.5 && size(a * a) == 0.0);
        assert!(size(powered(a, 2.0)) == 0.0 && close(powered(a, 1.0), a, 0.0) && close(powered(a, 0.0), DMat2::IDENTITY, 0.0));
        // the first step lands everything on the orange line, which the second sends to 0
        let Eigen::Real { l1, lines: Lines::One(v), .. } = eigen(a) else { panic!("{:?}", eigen(a)) };
        assert!(l1 == 0.0 && (a * v).length() < 1e-12);
        for p in F_OUTLINE {
            assert!((a * p).perp_dot(v).abs() < 1e-12);
        }
        // the steps go from I to A to A², stopping at A between
        assert!(steps_at(0.0) == 0.0 && steps_at(STEP_SECONDS) == 1.0 && steps_at(STEP_SECONDS + STEP_PAUSE) == 1.0 && steps_at(steps_length()) == 2.0);
        // a projection squashes onto a line too, and the second step changes nothing
        for p in members(Kind::Projection, 50, 3) {
            assert!(close(powered(p, 2.0), p, 1e-12) && close(powered(p, 1.5), p, 1e-12));
        }
    }

    #[test]
    fn the_symmetric_product_does_what_its_caption_says() {
        let scene = run(4);
        let s = scene.shown();
        let (a, b) = (s.a, s.b);
        assert!(scene.op == Op::Product && is(Kind::Symmetric, a) && is(Kind::Symmetric, b));
        let ab = a * b;
        assert!(!is(Kind::Symmetric, ab) && close(ab.transpose(), b * a, 1e-15) && !commute(a, b));
        let Eigen::Complex { re, im } = eigen(ab) else { panic!() };
        assert!(num(re, 2) == "0.88" && num(im, 2) == "0.99", "{re} {im}");
        assert_eq!(verdict(Kind::Symmetric, ab, "AB").0, r"$(AB)^\top \ne AB$: they differ by up to $2.25$");
        // every direction is turned, the same way round
        let turns: Vec<f64> = (0..180).map(|d| DVec2::from_angle((d as f64).to_radians())).map(|d| d.perp_dot(ab * d)).collect();
        assert!(turns.iter().all(|t| *t > 0.0) || turns.iter().all(|t| *t < 0.0));
        // and two orthogonal matrices always have an orthogonal product
        let mut rng = Rng::new(4);
        for _ in 0..200 {
            assert!(is(Kind::Orthogonal, random(Kind::Orthogonal, &mut rng) * random(Kind::Orthogonal, &mut rng)));
        }
    }

    /// What the heading says each kind's own matrix does, with its numbers.
    #[test]
    fn the_headings_say_what_the_matrices_do() {
        let say = |k: Kind| does(k, k.default_a());
        assert_eq!(say(Kind::Diagonal), "multiplies the $x$ axis by $1.50$ and the $y$ axis by $0.60$");
        assert_eq!(say(Kind::Triangular), "keeps the $x$ axis on itself, multiplying it by $1.20$");
        assert!(Kind::Triangular.default_a() * DVec2::X == DVec2::new(1.2, 0.0));
        assert_eq!(say(Kind::Symmetric), r"stretches along two perpendicular lines, by $\lambda = 2.00$ and $0.50$");
        assert_eq!(say(Kind::PosDef), r"stretches along two perpendicular lines, by $\lambda = 1.50$ and $0.50$, both positive");
        assert_eq!(say(Kind::Skew), "turns every arrow a quarter turn anticlockwise and multiplies its length by $1.20$");
        assert_eq!(say(Kind::Orthogonal), "turns the plane by 53.1°, keeping every length");
        assert!(close(Kind::Orthogonal.default_a(), DMat2::from_angle(53.13010235415598f64.to_radians()), 1e-12));
        assert_eq!(say(Kind::Normal), "turns the plane by 33.7° and multiplies every length by $1.08$");
        let n = Kind::Normal.default_a();
        assert!(close(n, DMat2::from_angle(0.6f64.atan2(0.9)) * 0.9f64.hypot(0.6), 1e-12));
        // the orange line is the line kept (λ = 1), the purple one the line sent to 0
        let p = Kind::Projection.default_a();
        let Eigen::Real { l1, l2, lines: Lines::Two(v1, v2) } = eigen(p) else { panic!() };
        assert!((l1 - 1.0).abs() < 1e-12 && l2.abs() < 1e-12 && (p * v1 - v1).length() < 1e-12 && (p * v2).length() < 1e-12);
        // Markov: the shares segment goes into itself
        let m = Kind::Markov.default_a();
        for t in 0..=10 {
            let q = m * DVec2::new(t as f64 / 10.0, 1.0 - t as f64 / 10.0);
            assert!(q.x >= 0.0 && q.y >= 0.0 && (q.x + q.y - 1.0).abs() < 1e-12);
        }
        // every heading is finished and says something
        for kind in KINDS {
            assert!(!kind.closure(Op::Alone).contains('.') && !kind.eigen_words().is_empty());
            for op in [Op::Sum, Op::Product, Op::Inverse] {
                assert!(kind.closure(op).ends_with('.'), "{kind:?} {op:?}");
            }
        }
    }

    /// The verdict holds exactly when the matrix is of the kind, and says so in
    /// finite numbers.
    #[test]
    fn the_verdict_is_the_test() {
        let mut rng = Rng::new(13);
        let mut cases: Vec<DMat2> = (0..300).map(|_| rows([0; 4].map(|_| rng.range(-3.0, 3.0) as f64))).collect();
        for kind in KINDS {
            cases.extend(members(kind, 20, 17));
        }
        cases.extend([DMat2::ZERO, DMat2::IDENTITY, SWAP, J, rows([1e-12, 0.0, 0.0, 0.0]), rows([3.0; 4])]);
        for m in cases {
            for kind in KINDS {
                for n in ["A", "A + B", "AB", "A^{-1}"] {
                    let (first, holds, second) = verdict(kind, m, n);
                    assert_eq!(holds, is(kind, m), "{kind:?} {m:?}");
                    for text in [Some(first), second, Some(does(kind, m)), Some(kind.equation(n))].into_iter().flatten() {
                        assert!(!text.contains("NaN") && !text.contains("inf") && !text.contains('?'), "{text}");
                    }
                }
            }
        }
    }

    /// Degenerate input: every number drawn or read out stays finite.
    #[test]
    fn degenerate_input_leaves_everything_finite() {
        let mats = [DMat2::ZERO, DMat2::IDENTITY * 1e-13, rows([1.0, 2.0, 0.5, 1.0]), rows([ENTRY_MAX; 4]), rows([f64::NAN, 1.0, 0.0, 1.0]), rows([1e300, 0.0, 0.0, 1.0])];
        for a in mats {
            for b in mats {
                for op in OPS {
                    let c = combine(op, within_limits(a), within_limits(b));
                    if let Some(c) = c {
                        for power in [0.0, 0.5, 1.0, 1.7, 2.0, -1.0, 9.0] {
                            assert!(powered(c, power).is_finite());
                        }
                        let e = eigen(c);
                        assert!(eigenvalues(e).iter().all(|(re, im)| re.is_finite() && im.is_finite()));
                        assert!(!eigen_text(e).contains("NaN"));
                    }
                }
            }
            for kind in KINDS {
                for j in 0..2 {
                    assert!(fit(kind, a, j, DVec2::new(0.3, 0.2)).is_finite());
                }
            }
        }
        assert!(inverse(DMat2::ZERO).is_none() && inverse(rows([1.0, 2.0, 0.5, 1.0])).is_none() && inverse(Kind::Nilpotent.default_a()).is_none());
        assert!(combine(Op::Inverse, Kind::Projection.default_a(), DMat2::ZERO).is_none());
    }

    /// Every experiment and every kind comes to rest, and then the scene asks for no
    /// more frames.
    #[test]
    fn everything_comes_to_rest() {
        for k in 0..TRIALS.len() {
            let _ = run(k);
        }
        for kind in KINDS {
            let mut scene = Special::new();
            scene.choose(kind);
            scene.another(false);
            scene.choose_op(Op::Product);
            scene.another(true);
            for _ in 0..600 {
                scene.advance(1.0 / 60.0, 1.0 / 60.0);
            }
            assert!(!scene.animated() && is(kind, scene.glide.to.a) && is(kind, scene.glide.to.b), "{kind:?}");
        }
    }

    /// The panel and the picture fit the canvas, from a phone to a desktop, and the
    /// panel's parts stay inside it, one below another.
    #[test]
    fn the_layout_fits_every_canvas() {
        for (w, h) in [(1080.0, 1000.0), (580.0, 700.0), (390.0, 800.0), (390.0, 560.0), (360.0, 640.0), (800.0, 600.0), (1600.0, 1000.0), (240.0, 180.0), (620.0, 480.0), (1300.0, 700.0)] {
            for inset in [0.0, 46.0] {
                for two in [false, true] {
                    let view = Vec2::new(w, h);
                    let lay = Layout::of(view, inset, two);
                    let (min, max) = lay.picture;
                    assert!(min.cmpge(Vec2::ZERO).all() && max.cmple(view).all() && (max - min).min_element() > 50.0, "{w}x{h}: {lay:?}");
                    if let Some(p) = &lay.panel {
                        let within = |q: Vec2| q.cmpge(p.min - Vec2::splat(0.5)).all() && q.cmple(p.max + Vec2::splat(0.5)).all();
                        assert!(p.min.cmpge(Vec2::ZERO).all() && p.max.cmple(view + Vec2::splat(0.5)).all(), "{w}x{h}: {p:?}");
                        assert!(p.chips.iter().all(|(lo, hi)| within(*lo) && within(*hi)) && within(p.chart.0) && within(p.chart.1) && within(p.another.1) && within(p.another_b.1), "{w}x{h}: {p:?}");
                        let rows = (if two { 5 } else { 3 } as usize).div_ceil(p.per_row);
                        let last = p.cells + Vec2::new((p.per_row - 1) as f32 * p.step.x + matrix_w(p.text), rows as f32 * p.step.y);
                        assert!(within(last) && last.y <= p.lines_end && p.lines_end <= p.chart_title.y && p.chart.1.y < p.another.0.y, "{w}x{h}: {p:?}");
                        assert!(p.chips.iter().all(|c| c.1.y < p.title.y) && p.note.y < p.title.y, "{w}x{h}");
                        assert!(p.min.y >= inset || lay.narrow, "{w}x{h}: under the buttons");
                    }
                }
            }
        }
    }
}

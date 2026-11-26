//! A zoo of special matrices: thirty kinds of 2 × 2 matrix, each defined by an equation
//! (or a condition), in seven families. The equation decides what the map can do to the
//! plane and where its eigenvalues can lie in the complex plane; a map of the zoo shows
//! which kinds lie inside which, and any matrix can be asked which kinds it is.

use super::*;
use crate::scenes::stats::plot::{self, Plot};
use glam::{DMat2, DVec2};

pub const DEF: SceneDef = SceneDef {
    id: "special",
    title: "A zoo of special matrices",
    group: SPECIAL,
    blurb: "Thirty kinds of $2 \\times 2$ matrix, from symmetric and orthogonal to nilpotent and Markov: what each does to the plane, where its eigenvalues lie, and which kinds contain which.",
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

/// Mⁿ, by repeated products (n is small here).
fn power(m: DMat2, n: u32) -> DMat2 {
    (0..n).fold(DMat2::IDENTITY, |p, _| p * m)
}

/// The most steps a root of I may take to come back: the scene looks for Aⁿ = I with n
/// from 1 to this.
const ROOT_MAX: u32 = 12;

/// The smallest n ≤ `ROOT_MAX` with Mⁿ as near I as rounding allows, and how far Mⁿ is
/// from I at the n where it comes nearest (that n, and its distance, if none is near
/// enough).
fn order(m: DMat2) -> (u32, f64, bool) {
    let tol = EXACT * size(m).max(1.0).powi(2);
    let mut best = (1, f64::INFINITY);
    let mut p = DMat2::IDENTITY;
    for n in 1..=ROOT_MAX {
        p *= m;
        let off = size(p - DMat2::IDENTITY);
        if off <= tol {
            return (n, off, true);
        }
        if off < best.1 {
            best = (n, off);
        }
    }
    (best.0, if best.1.is_finite() { best.1 } else { f64::MAX }, false)
}

/// The Gershgorin discs of a matrix, by rows: about each diagonal entry, with the size
/// of the other entry of its row as the radius. Every eigenvalue lies in one of them.
fn discs(m: DMat2) -> [(f64, f64); 2] {
    let [a, b, c, d] = entries(m);
    [(a, b.abs()), (d, c.abs())]
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

/// An angle in degrees for the screen, to a tenth, with no "−0.0°".
fn degrees(r: f64) -> String {
    let v = r.to_degrees();
    format!("{:.1}°", if v.abs() < 0.05 { 0.0 } else { v })
}

// ----- the kinds ---------------------------------------------------------------------

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Kind {
    // by where the zeros are
    Scalar,
    Diagonal,
    Upper,
    Lower,
    // by symmetry
    Symmetric,
    Skew,
    Normal,
    ScaledRotation,
    // by what the map keeps
    Orthogonal,
    Rotation,
    Reflection,
    DetOne,
    // by what its powers do
    Projection,
    OrthProjection,
    Involution,
    Nilpotent,
    Shear,
    Root,
    // by sign
    PosDef,
    PosSemidef,
    Indefinite,
    Positive,
    Markov,
    Doubly,
    Dominant,
    // by rank
    Singular,
    RankOne,
    // with names of their own
    Permutation,
    Companion,
    JordanBlock,
    /// No kind: any matrix at all, free to drag, with every kind it is of lit.
    Any,
}

/// The kinds, family by family, in the order they are offered.
const KINDS: [Kind; 30] = [
    Kind::Scalar,
    Kind::Diagonal,
    Kind::Upper,
    Kind::Lower,
    Kind::Symmetric,
    Kind::Skew,
    Kind::Normal,
    Kind::ScaledRotation,
    Kind::Orthogonal,
    Kind::Rotation,
    Kind::Reflection,
    Kind::DetOne,
    Kind::Projection,
    Kind::OrthProjection,
    Kind::Involution,
    Kind::Nilpotent,
    Kind::Shear,
    Kind::Root,
    Kind::PosDef,
    Kind::PosSemidef,
    Kind::Indefinite,
    Kind::Positive,
    Kind::Markov,
    Kind::Doubly,
    Kind::Dominant,
    Kind::Singular,
    Kind::RankOne,
    Kind::Permutation,
    Kind::Companion,
    Kind::JordanBlock,
];

/// The families the kinds are found by.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Family {
    Zeros,
    Symmetry,
    Keeps,
    Powers,
    Signs,
    Rank,
    Named,
}

const FAMILIES: [Family; 7] = [Family::Zeros, Family::Symmetry, Family::Keeps, Family::Powers, Family::Signs, Family::Rank, Family::Named];

impl Family {
    fn kinds(self) -> &'static [Kind] {
        let range = match self {
            Family::Zeros => 0..4,
            Family::Symmetry => 4..8,
            Family::Keeps => 8..12,
            Family::Powers => 12..18,
            Family::Signs => 18..25,
            Family::Rank => 25..27,
            Family::Named => 27..30,
        };
        &KINDS[range]
    }

    /// Its name on its tab.
    fn tab(self) -> &'static str {
        match self {
            Family::Zeros => "zeros",
            Family::Symmetry => "symmetry",
            Family::Keeps => "what it keeps",
            Family::Powers => "powers",
            Family::Signs => "signs",
            Family::Rank => "rank",
            Family::Named => "named",
        }
    }

    /// What its kinds have in common.
    fn heading(self) -> &'static str {
        match self {
            Family::Zeros => "by where the zeros are",
            Family::Symmetry => "by symmetry",
            Family::Keeps => "by what the map keeps",
            Family::Powers => "by what doing it again does",
            Family::Signs => "by the signs of the entries, or of the eigenvalues",
            Family::Rank => "by rank: how much of the plane is left",
            Family::Named => "with names of their own",
        }
    }
}

/// Where a kind's eigenvalues can lie, in the complex plane.
#[derive(Clone, Copy, Debug, PartialEq)]
enum Region {
    /// The real line.
    Real,
    /// One real number, twice.
    Twice,
    /// The real line right of 0 (0 itself left out).
    Positive,
    /// The real line from 0 rightwards, 0 included.
    NonNegative,
    /// The real line, one each side of 0.
    Sides,
    /// The imaginary line.
    Imaginary,
    /// The unit circle.
    Circle,
    /// Only these points of the real line.
    Points(&'static [f64]),
    /// 1, and the real line from −1 to 1 for the other.
    Shares,
    /// λ and 1/λ on the real line, or a pair on the unit circle.
    DetOne,
    /// The roots of 1: points of the unit circle a whole share of a turn round.
    Roots,
    /// The Gershgorin discs of the matrix itself.
    Discs,
    /// 0, and the other on the real line.
    ZeroAndReal,
    /// The real line, the larger positive and larger in size.
    Perron,
    /// Anywhere at all.
    Anywhere,
}

impl Region {
    /// Its name, as the chart writes it.
    fn name(self) -> &'static str {
        match self {
            Region::Real => "the real line",
            Region::Twice => "one real number, twice",
            Region::Positive => r"the real line right of $0$",
            Region::NonNegative => r"the real line from $0$ on",
            Region::Sides => r"real, one each side of $0$",
            Region::Imaginary => "the imaginary line",
            Region::Circle => "the unit circle",
            Region::Points([0.0, 1.0]) => r"$0$ and $1$ only",
            Region::Points([0.0]) => r"$0$ only",
            Region::Points([1.0, -1.0]) => r"$1$ and $-1$ only",
            Region::Points(_) => r"$1$ only",
            Region::Shares => r"$1$, and the other in $[-1, 1]$",
            Region::DetOne => r"$\lambda$ and $1/\lambda$, or on the circle",
            Region::Roots => r"roots of $1$ on the unit circle",
            Region::Discs => "the Gershgorin discs",
            Region::ZeroAndReal => r"$0$, and the trace",
            Region::Perron => "real; the larger positive",
            Region::Anywhere => "anywhere",
        }
    }

    /// Whether the eigenvalue re + im i of `m` lies in it, to within `tol`.
    fn holds(self, (re, im): (f64, f64), tol: f64, m: DMat2) -> bool {
        let real = im.abs() <= tol;
        match self {
            Region::Real | Region::Twice | Region::ZeroAndReal | Region::Perron => real,
            Region::Positive => real && re > 0.0,
            Region::NonNegative => real && re >= -tol,
            Region::Sides => real && re.abs() > tol,
            Region::Imaginary => re.abs() <= tol,
            Region::Circle => (re.hypot(im) - 1.0).abs() <= tol,
            Region::Points(points) => real && points.iter().any(|p| (re - p).abs() <= tol),
            Region::Shares => real && re >= -1.0 - tol && re <= 1.0 + tol,
            Region::DetOne => (real && re.abs() > tol) || (re.hypot(im) - 1.0).abs() <= tol,
            Region::Roots => {
                // on the circle, and a whole number of n-ths of a turn round for some n
                let turns = im.atan2(re) / std::f64::consts::TAU;
                (re.hypot(im) - 1.0).abs() <= tol && (1..=ROOT_MAX).any(|n| (turns * n as f64 - (turns * n as f64).round()).abs() <= tol * n as f64 * 10.0)
            }
            Region::Discs => discs(m).iter().any(|(c, r)| (re - c).hypot(im) <= r + tol),
            Region::Anywhere => true,
        }
    }
}

impl Kind {
    fn family(self) -> Family {
        FAMILIES.into_iter().find(|f| f.kinds().contains(&self)).unwrap_or(Family::Zeros)
    }

    fn name(self) -> &'static str {
        match self {
            Kind::Scalar => r"multiple of $I$",
            Kind::Diagonal => "diagonal",
            Kind::Upper => "upper triangular",
            Kind::Lower => "lower triangular",
            Kind::Symmetric => "symmetric",
            Kind::Skew => "skew-symmetric",
            Kind::Normal => "normal",
            Kind::ScaledRotation => "scaled rotation",
            Kind::Orthogonal => "orthogonal",
            Kind::Rotation => "rotation",
            Kind::Reflection => "reflection",
            Kind::DetOne => r"determinant $1$",
            Kind::Projection => "projection",
            Kind::OrthProjection => "orthogonal projection",
            Kind::Involution => "involution",
            Kind::Nilpotent => "nilpotent",
            Kind::Shear => "shear",
            Kind::Root => r"root of $I$",
            Kind::PosDef => "positive definite",
            Kind::PosSemidef => "positive semidefinite",
            Kind::Indefinite => "indefinite",
            Kind::Positive => "positive",
            Kind::Markov => "Markov",
            Kind::Doubly => "doubly stochastic",
            Kind::Dominant => "diagonally dominant",
            Kind::Singular => "singular",
            Kind::RankOne => "rank one",
            Kind::Permutation => "permutation",
            Kind::Companion => "companion",
            Kind::JordanBlock => "Jordan block",
            Kind::Any => "any matrix",
        }
    }

    /// Its name on its button.
    fn chip(self) -> &'static str {
        match self {
            Kind::Positive => "positive entries",
            k => k.name(),
        }
    }

    /// Its name on the map of the zoo, in one or two lines.
    fn map_name(self) -> [&'static str; 2] {
        match self {
            Kind::Scalar => ["multiple", r"of $I$"],
            Kind::Upper => ["upper", "triangular"],
            Kind::Lower => ["lower", "triangular"],
            Kind::Skew => ["skew-", "symmetric"],
            Kind::ScaledRotation => ["scaled", "rotation"],
            Kind::DetOne => [r"$\det A = 1$", ""],
            Kind::OrthProjection => ["orthogonal", "projection"],
            Kind::PosDef => ["positive", "definite"],
            Kind::PosSemidef => ["positive", "semidefinite"],
            Kind::Positive => ["positive", "entries"],
            Kind::Doubly => ["doubly", "stochastic"],
            Kind::Dominant => ["diagonally", "dominant"],
            Kind::JordanBlock => ["Jordan", "block"],
            k => [k.name(), ""],
        }
    }

    /// "a symmetric matrix", "an orthogonal matrix", "a projection".
    fn a_name(self) -> String {
        match self {
            Kind::Scalar => r"a multiple of $I$".into(),
            Kind::Upper | Kind::Orthogonal | Kind::Indefinite => format!("an {} matrix", self.name()),
            Kind::OrthProjection | Kind::Involution => format!("an {}", self.name()),
            Kind::ScaledRotation | Kind::Rotation | Kind::Reflection | Kind::Projection | Kind::Shear | Kind::Root | Kind::JordanBlock => format!("a {}", self.name()),
            Kind::DetOne => r"a matrix of determinant $1$".into(),
            Kind::RankOne => "a matrix of rank one".into(),
            Kind::Any => "any matrix".into(),
            _ => format!("a {} matrix", self.name()),
        }
    }

    /// "symmetric matrices", "projections".
    fn plural(self) -> String {
        match self {
            Kind::Scalar => r"multiples of $I$".into(),
            Kind::ScaledRotation | Kind::Rotation | Kind::Reflection | Kind::Projection | Kind::OrthProjection | Kind::Involution | Kind::Shear | Kind::JordanBlock => format!("{}s", self.name()),
            Kind::Root => r"roots of $I$".into(),
            Kind::DetOne => r"matrices of determinant $1$".into(),
            Kind::RankOne => "matrices of rank one".into(),
            Kind::Any => "matrices".into(),
            _ => format!("{} matrices", self.name()),
        }
    }

    fn region(self) -> Region {
        match self {
            Kind::Diagonal | Kind::Upper | Kind::Lower | Kind::Symmetric => Region::Real,
            Kind::Scalar | Kind::JordanBlock => Region::Twice,
            Kind::PosDef => Region::Positive,
            Kind::PosSemidef => Region::NonNegative,
            Kind::Indefinite => Region::Sides,
            Kind::Skew => Region::Imaginary,
            Kind::Orthogonal | Kind::Rotation => Region::Circle,
            Kind::Reflection | Kind::Involution | Kind::Permutation => Region::Points(&[1.0, -1.0]),
            Kind::DetOne => Region::DetOne,
            Kind::Root => Region::Roots,
            Kind::Normal | Kind::ScaledRotation | Kind::Companion | Kind::Any => Region::Anywhere,
            Kind::Projection | Kind::OrthProjection => Region::Points(&[0.0, 1.0]),
            Kind::Nilpotent => Region::Points(&[0.0]),
            Kind::Shear => Region::Points(&[1.0]),
            Kind::Markov | Kind::Doubly => Region::Shares,
            Kind::Dominant => Region::Discs,
            Kind::Singular | Kind::RankOne => Region::ZeroAndReal,
            Kind::Positive => Region::Perron,
        }
    }

    /// How many numbers can be chosen freely in a matrix of the kind: the dimension of
    /// the set, where it is a smooth family.
    fn free(self) -> &'static str {
        match self {
            Kind::Scalar | Kind::Skew | Kind::Orthogonal | Kind::Rotation | Kind::Reflection | Kind::Doubly | Kind::JordanBlock => "1",
            Kind::Diagonal | Kind::ScaledRotation | Kind::Nilpotent | Kind::Shear | Kind::Markov | Kind::Companion => "2",
            Kind::Upper | Kind::Lower | Kind::Symmetric | Kind::DetOne | Kind::PosDef | Kind::PosSemidef | Kind::Indefinite | Kind::Singular | Kind::RankOne => "3",
            Kind::Positive | Kind::Dominant | Kind::Any => "4",
            Kind::Normal => r"3 (the scaled rotations among them: 2)",
            Kind::Projection => r"2 (and $0$ and $I$ besides)",
            Kind::OrthProjection => r"1 (and $0$ and $I$ besides)",
            Kind::Involution => r"2 (and $I$ and $-I$ besides)",
            Kind::Root => r"2, for each $n$ and share of a turn",
            Kind::Permutation => "0: two matrices only",
        }
    }

    /// The scene where the kind turns up, and what that scene is, in a few words.
    fn home(self) -> (&'static str, &'static str) {
        match self {
            Kind::Scalar => ("commute", "matrices that commute"),
            Kind::Diagonal => ("basis", "change of basis, where the eigenvector basis makes a matrix diagonal"),
            Kind::Upper => ("schur", "Schur form, upper triangular in a perpendicular basis"),
            Kind::Lower => ("factor", r"$A = LU$, lower triangular times upper"),
            Kind::Symmetric => ("spectral", "the spectral theorem"),
            Kind::Skew => ("symskew", "every matrix as symmetric plus skew-symmetric"),
            Kind::Normal => ("symskew", r"symmetric and skew parts, which commute exactly when $A$ is normal"),
            Kind::ScaledRotation => ("expm", "spirals, from the matrix exponential"),
            Kind::Orthogonal => ("gram", "Gram–Schmidt, which makes orthogonal matrices"),
            Kind::Rotation => ("rotations", "rotations in 3D"),
            Kind::Reflection => ("project", "projections and reflections"),
            Kind::DetOne => ("det", "determinants, how a map scales areas"),
            Kind::Projection => ("project", "projections and reflections"),
            Kind::OrthProjection => ("lsq", "projection and least squares"),
            Kind::Involution => ("project", "mirrors, straight and slanted"),
            Kind::Nilpotent => ("jordan", "Jordan blocks, when eigenvectors run out"),
            Kind::Shear => ("linmap2", "linear maps, a shear among them"),
            Kind::Root => ("powers", "powers of a matrix"),
            Kind::PosDef => ("cholesky", "Cholesky, a triangular square root"),
            Kind::PosSemidef => ("pca", "covariance matrices and principal components"),
            Kind::Indefinite => ("quadform", "quadratic forms, bowls and saddles"),
            Kind::Positive => ("powers", "powers of a matrix, and the eigenvector they find"),
            Kind::Markov => ("powers", "a Markov chain settling, step by step"),
            Kind::Doubly => ("powers", "a Markov chain settling, step by step"),
            Kind::Dominant => ("eigen", "finding eigenvalues"),
            Kind::Singular => ("subspaces", r"the four subspaces, and what a matrix sends to $0$"),
            Kind::RankOne => ("lowrank", "a matrix as a sum of rank-one pieces"),
            Kind::Permutation => ("factor", r"row swaps, $PA = LU$"),
            Kind::Companion => ("patterns", "larger companion matrices, and other patterns"),
            Kind::JordanBlock => ("jordan", "Jordan blocks, when eigenvectors run out"),
            Kind::Any => ("patterns", "larger matrices, known by their pattern"),
        }
    }

    /// The defining equation, for the matrix written `n` (`A`, `AB`, `A + B`, ...).
    fn equation(self, n: &str) -> String {
        let p = paren(n);
        let e = |ij: u32| entry(n, ij);
        match self {
            Kind::Scalar => format!("${} = {} = 0$, ${} = {}$", e(12), e(21), e(11), e(22)),
            Kind::Diagonal => format!("${} = {} = 0$", e(12), e(21)),
            Kind::Upper => format!("${} = 0$", e(21)),
            Kind::Lower => format!("${} = 0$", e(12)),
            Kind::Symmetric => format!(r"${p}^\top = {n}$"),
            Kind::Skew => format!(r"${p}^\top = -{p}$"),
            Kind::Normal => format!(r"${p}{p}^\top = {p}^\top {p}$"),
            Kind::ScaledRotation => format!("${} = {}$, ${} = -{}$", e(11), e(22), e(21), e(12)),
            Kind::Orthogonal => format!(r"${p}^\top {p} = I$"),
            Kind::Rotation => format!(r"${p}^\top {p} = I$, $\det {p} = 1$"),
            Kind::Reflection => format!(r"${p}^\top {p} = I$, $\det {p} = -1$"),
            Kind::DetOne => format!(r"$\det {p} = 1$"),
            Kind::Projection => format!("${p}^2 = {n}$"),
            Kind::OrthProjection => format!(r"${p}^2 = {n}$, ${p}^\top = {n}$"),
            Kind::Involution => format!("${p}^2 = I$"),
            Kind::Nilpotent => format!("${p}^2 = 0$"),
            Kind::Shear => format!(r"$({n} - I)^2 = 0$, ${n} \ne I$"),
            Kind::Root => format!(r"${p}^n = I$ for a whole $n \le {ROOT_MAX}$"),
            Kind::PosDef => format!(r"${p}^\top = {n}$, $x \cdot {p}x > 0$"),
            Kind::PosSemidef => format!(r"${p}^\top = {n}$, $x \cdot {p}x \ge 0$"),
            Kind::Indefinite => format!(r"${p}^\top = {n}$, $\det {p} < 0$"),
            Kind::Positive => r"every entry $> 0$".into(),
            Kind::Markov => r"entries $\ge 0$, columns add to $1$".into(),
            Kind::Doubly => r"entries $\ge 0$, rows and columns add to $1$".into(),
            Kind::Dominant => format!("$|{}| > |{}|$, $|{}| > |{}|$", e(11), e(12), e(22), e(21)),
            Kind::Singular => format!(r"$\det {p} = 0$"),
            Kind::RankOne => format!(r"$\det {p} = 0$, ${n} \ne 0$"),
            Kind::Permutation => r"$I$, or $S$, the swap".into(),
            Kind::Companion => format!("${} = 0$, ${} = 1$", e(11), e(21)),
            Kind::JordanBlock => format!("${} = 0$, ${} = 1$, ${} = {}$", e(21), e(12), e(11), e(22)),
            Kind::Any => "no equation".into(),
        }
    }

    /// What its eigenvalues and eigenlines are, in a few words.
    fn eigen_words(self) -> &'static str {
        match self {
            Kind::Scalar => r"every direction is an eigenvector, and it commutes with every matrix",
            Kind::Diagonal => "its eigenvalues are its diagonal entries, and the axes are eigenlines",
            Kind::Upper => r"its eigenvalues are its diagonal entries, $a_{11}$ and $a_{22}$; the $x$ axis is an eigenline",
            Kind::Lower => r"its eigenvalues are its diagonal entries, $a_{11}$ and $a_{22}$; the $y$ axis is an eigenline",
            Kind::Symmetric => "its eigenvalues are real, and its eigenlines at right angles",
            Kind::Skew => r"$x \cdot Ax = 0$ for every $x$; its eigenvalues lie on the imaginary line",
            Kind::Normal => "in the plane: symmetric, or a scaled rotation; eigenvectors at right angles (complex ones, for a turn)",
            Kind::ScaledRotation => r"$aI + bJ$ is the complex number $a + bi$ as a matrix; its eigenvalues are $a \pm bi$",
            Kind::Orthogonal => "it keeps every length and angle; its eigenvalues lie on the unit circle",
            Kind::Rotation => r"its eigenvalues $\cos\theta \pm i \sin\theta$ lie on the unit circle",
            Kind::Reflection => r"its eigenvalues are $1$, along the mirror, and $-1$, across it",
            Kind::DetOne => r"its eigenvalues multiply to $1$: $\lambda$ and $1/\lambda$ on the real line, or a pair on the unit circle",
            Kind::Projection => r"applied twice it does nothing more; its eigenvalues are $0$ and $1$ ($0$ and $I$ have one of them twice)",
            Kind::OrthProjection => r"its eigenvalues are $0$ and $1$ ($0$ and $I$ have one of them twice), and its eigenlines are at right angles",
            Kind::Involution => r"applied twice it undoes itself; its eigenvalues are $1$ and $-1$ ($I$ and $-I$ have one twice)",
            Kind::Nilpotent => r"its only eigenvalue is $0$; unless $A = 0$, it has a single eigenline",
            Kind::Shear => r"its only eigenvalue is $1$, with a single eigenline; it keeps areas",
            Kind::Root => r"its eigenvalues are $n$-th roots of $1$, on the unit circle",
            Kind::PosDef => "its eigenvalues are positive, and its eigenlines at right angles",
            Kind::PosSemidef => "its eigenvalues are real and none is negative; its eigenlines are at right angles",
            Kind::Indefinite => "one eigenvalue is positive and one negative, on eigenlines at right angles: a saddle",
            Kind::Positive => "both eigenvalues are real; the larger is positive and larger in size than the other, and its eigenline runs through the positive quarter",
            Kind::Markov => r"one eigenvalue is exactly $1$; the other lies between $-1$ and $1$",
            Kind::Doubly => r"its eigenvalues are $1$ and $2a_{11} - 1$",
            Kind::Dominant => r"each eigenvalue lies in a Gershgorin disc, and $0$ in neither, so $\det A \ne 0$",
            Kind::Singular => r"one eigenvalue is $0$; the other is the trace",
            Kind::RankOne => r"$A = uv^\top$; its eigenvalues are $0$ and $v \cdot u$, which is the trace",
            Kind::Permutation => r"$I$ has the eigenvalue $1$ twice; the swap has $1$ and $-1$",
            Kind::Companion => r"its eigenvalues are the roots of $\lambda^2 - a_{22}\lambda - a_{12}$, read off its last column; every matrix but a multiple of $I$ is one in a suitable basis",
            Kind::JordanBlock => r"its only eigenvalue is its diagonal entry, with a single eigenline, the $x$ axis",
            Kind::Any => "every kind it is of is lit",
        }
    }

    /// The matrix a kind starts with.
    fn default_a(self) -> DMat2 {
        rows(match self {
            Kind::Scalar => [1.3, 0.0, 0.0, 1.3],
            Kind::Diagonal => [1.5, 0.0, 0.0, 0.6],
            Kind::Upper => [1.2, 0.8, 0.0, 0.6],
            Kind::Lower => [1.2, 0.0, 0.8, 0.6],
            Kind::Symmetric => [1.7, 0.6, 0.6, 0.8],
            Kind::Skew => [0.0, -1.2, 1.2, 0.0],
            Kind::Normal => [0.9, -0.6, 0.6, 0.9],
            Kind::ScaledRotation => [1.2, -0.5, 0.5, 1.2],
            Kind::Orthogonal => [0.6, -0.8, 0.8, 0.6],
            Kind::Rotation => [0.6, -0.8, 0.8, 0.6],
            Kind::Reflection => [0.6, 0.8, 0.8, -0.6],
            Kind::DetOne => [1.5, 0.5, 0.4, 0.8],
            Kind::Projection => [0.6, 0.6, 0.4, 0.4],
            Kind::OrthProjection => [0.8, 0.4, 0.4, 0.2],
            Kind::Involution => [1.0, 1.0, 0.0, -1.0],
            Kind::Nilpotent => [0.8, -1.6, 0.4, -0.8],
            Kind::Shear => [1.0, 1.0, 0.0, 1.0],
            Kind::Root => [1.0, -1.0, 1.0, 0.0],
            Kind::PosDef => [1.3, 0.4, 0.4, 0.7],
            Kind::PosSemidef => [1.6, 0.8, 0.8, 0.4],
            Kind::Indefinite => [1.0, 1.2, 1.2, -0.5],
            Kind::Positive => [0.5, 1.5, 1.0, 1.0],
            Kind::Markov => [0.8, 0.3, 0.2, 0.7],
            Kind::Doubly => [0.8, 0.2, 0.2, 0.8],
            Kind::Dominant => [1.5, 0.8, -0.5, -1.2],
            Kind::Singular => [1.2, 0.6, 0.8, 0.4],
            Kind::RankOne => [1.2, -0.6, 0.6, -0.3],
            Kind::Permutation => [0.0, 1.0, 1.0, 0.0],
            Kind::Companion => [0.0, -0.5, 1.0, 1.5],
            Kind::JordanBlock => [1.2, 1.0, 0.0, 1.2],
            Kind::Any => [1.2, 0.5, -0.4, 0.8],
        })
    }

    /// The second matrix of the kind it starts with, for sums and products.
    fn default_b(self) -> DMat2 {
        rows(match self {
            Kind::Scalar => [-0.6, 0.0, 0.0, -0.6],
            Kind::Diagonal => [-0.8, 0.0, 0.0, 1.4],
            Kind::Upper => [0.7, -0.6, 0.0, 1.3],
            Kind::Lower => [0.7, 0.0, -0.6, 1.3],
            Kind::Symmetric => [1.5, -0.5, -0.5, -0.5],
            Kind::Skew => [0.0, 0.8, -0.8, 0.0],
            Kind::Normal => [1.2, 0.5, 0.5, -0.4],
            Kind::ScaledRotation => [0.5, -1.0, 1.0, 0.5],
            Kind::Orthogonal => [0.6, 0.8, 0.8, -0.6],
            Kind::Rotation => [0.0, -1.0, 1.0, 0.0],
            Kind::Reflection => [-0.6, 0.8, 0.8, 0.6],
            Kind::DetOne => [0.8, -0.6, 1.0, 0.5],
            Kind::Projection => [1.0, -1.0, 0.0, 0.0],
            Kind::OrthProjection => [0.5, -0.5, -0.5, 0.5],
            Kind::Involution => [0.0, 1.0, 1.0, 0.0],
            Kind::Nilpotent => [0.0, 1.0, 0.0, 0.0],
            Kind::Shear => [1.0, 0.0, 0.8, 1.0],
            Kind::Root => [0.0, -1.0, 1.0, -1.0],
            Kind::PosDef => [0.9, -0.5, -0.5, 1.1],
            Kind::PosSemidef => [1.0, 0.3, 0.3, 0.6],
            Kind::Indefinite => [0.5, -1.0, -1.0, 0.2],
            Kind::Positive => [1.0, 0.4, 0.6, 0.8],
            Kind::Markov => [0.5, 0.9, 0.5, 0.1],
            Kind::Doubly => [0.3, 0.7, 0.7, 0.3],
            Kind::Dominant => [1.0, -0.6, 0.3, 0.9],
            Kind::Singular => [0.5, -1.0, -0.5, 1.0],
            Kind::RankOne => [0.4, 0.8, 0.5, 1.0],
            Kind::Permutation => [1.0, 0.0, 0.0, 1.0],
            Kind::Companion => [0.0, -1.0, 1.0, 0.0],
            Kind::JordanBlock => [0.5, 1.0, 0.0, 0.5],
            Kind::Any => [0.6, -0.8, 0.9, 0.4],
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
    let det = m.determinant();
    let at_most = |off: f64, tol: f64| (off, off <= tol);
    match kind {
        Kind::Scalar => at_most(b.abs().max(c.abs()).max((a - d).abs()), lin),
        Kind::Diagonal => at_most(b.abs().max(c.abs()), lin),
        Kind::Upper => at_most(c.abs(), lin),
        Kind::Lower => at_most(b.abs(), lin),
        Kind::Symmetric => at_most((b - c).abs(), lin),
        Kind::PosDef | Kind::PosSemidef => {
            let (asym, low) = ((b - c).abs(), least_form(m));
            let positive = if kind == Kind::PosDef { low > lin } else { low >= -lin };
            (asym.max(-low).max(0.0), asym <= lin && positive)
        }
        Kind::Indefinite => {
            let asym = (b - c).abs();
            (asym.max(det.max(0.0)), asym <= lin && det < -quad)
        }
        Kind::Skew => at_most(a.abs().max(d.abs()).max((b + c).abs()), lin),
        Kind::ScaledRotation => at_most((a - d).abs().max((b + c).abs()), lin),
        Kind::Orthogonal => at_most(size(t * m - id), quad),
        Kind::Rotation => at_most(size(t * m - id).max((det - 1.0).abs()), quad),
        Kind::Reflection => at_most(size(t * m - id).max((det + 1.0).abs()), quad),
        Kind::DetOne => at_most((det - 1.0).abs(), quad),
        Kind::Normal => at_most(size(m * t - t * m), quad),
        Kind::Projection => at_most(size(m * m - m), quad),
        Kind::OrthProjection => {
            let (idem, asym) = (size(m * m - m), (b - c).abs());
            (idem.max(asym), idem <= quad && asym <= lin)
        }
        Kind::Involution => at_most(size(m * m - id), quad),
        Kind::Nilpotent => at_most(size(m * m), quad),
        Kind::Shear => {
            let n = m - id;
            let off = size(n * n);
            (off, off <= quad && size(n) > EXACT)
        }
        Kind::Root => {
            let (_, off, holds) = order(m);
            (off, holds)
        }
        Kind::Positive => {
            let low = a.min(b).min(c).min(d);
            ((-low).max(0.0), low > lin)
        }
        Kind::Markov | Kind::Doubly => {
            let mut off = (a + c - 1.0).abs().max((b + d - 1.0).abs()).max(-a.min(b).min(c).min(d)).max(0.0);
            if kind == Kind::Doubly {
                off = off.max((a + b - 1.0).abs());
            }
            at_most(off, lin)
        }
        Kind::Dominant => {
            let margin = (a.abs() - b.abs()).min(d.abs() - c.abs());
            ((-margin).max(0.0), margin > lin)
        }
        Kind::Singular => at_most(det.abs(), quad),
        Kind::RankOne => (det.abs(), det.abs() <= quad && size(m) > lin),
        Kind::Permutation => at_most(size(m - id).min(size(m - SWAP)), lin),
        Kind::Companion => at_most(a.abs().max((c - 1.0).abs()), lin),
        Kind::JordanBlock => at_most(c.abs().max((b - 1.0).abs()).max((a - d).abs()), lin),
        Kind::Any => (0.0, true),
    }
}

fn is(kind: Kind, m: DMat2) -> bool {
    test(kind, m).1
}

/// The kinds a matrix is of, as a mask over `KINDS`.
fn kinds_of(m: DMat2) -> [bool; 30] {
    KINDS.map(|k| is(k, m))
}

/// The sides of the kind's equation that are worked out from `m` (named `n`), to be
/// written out as matrices beside it.
fn sides(kind: Kind, m: DMat2, n: &str) -> Vec<(String, DMat2)> {
    let p = paren(n);
    let t = m.transpose();
    match kind {
        Kind::Symmetric | Kind::PosDef | Kind::PosSemidef | Kind::Indefinite => vec![(format!(r"${p}^\top$"), t)],
        Kind::Skew => vec![(format!(r"${p}^\top$"), t), (format!("$-{p}$"), -m)],
        Kind::Orthogonal | Kind::Rotation | Kind::Reflection => vec![(format!(r"${p}^\top {p}$"), t * m)],
        Kind::Normal => vec![(format!(r"${p}{p}^\top$"), m * t), (format!(r"${p}^\top {p}$"), t * m)],
        Kind::Projection | Kind::Nilpotent | Kind::Involution => vec![(format!("${p}^2$"), m * m)],
        Kind::OrthProjection => vec![(format!("${p}^2$"), m * m), (format!(r"${p}^\top$"), t)],
        Kind::Shear => {
            let k = m - DMat2::IDENTITY;
            vec![(format!("$({n} - I)^2$"), k * k)]
        }
        Kind::Root => {
            let (k, _, _) = order(m);
            vec![(format!("${p}^{{{k}}}$"), power(m, k))]
        }
        _ => Vec::new(),
    }
}

/// A difference for the screen: two decimals, or two figures in powers of ten when it
/// is smaller than that.
fn small(d: f64) -> String {
    if d == 0.0 || d >= 0.005 || !d.is_finite() {
        num(d.min(1e9), 2)
    } else {
        format!("{d:.1e}").replace('-', "−")
    }
}

/// The share of a full turn, k/n in lowest terms, that the angle θ (radians, in
/// (0, π]) is, for n up to `ROOT_MAX`.
fn share_of_turn(theta: f64) -> Option<(u32, u32)> {
    let turns = theta / std::f64::consts::TAU;
    (1..=ROOT_MAX).find_map(|n| {
        let k = (turns * n as f64).round();
        ((turns * n as f64 - k).abs() < 1e-6 && k >= 1.0).then_some((k as u32, n))
    })
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
    let e = |ij: u32| entry(n, ij);
    let s = size(m).max(1.0);
    let det = m.determinant();
    let orthogonal = size(m.transpose() * m - DMat2::IDENTITY) <= EXACT * s * s;
    let first = match kind {
        Kind::Scalar => {
            if holds {
                format!("${} = {} = 0$, ${} = {}${ok}", e(12), e(21), e(11), e(22))
            } else if b.abs().max(c.abs()) <= EXACT * s {
                format!("${} = {}$, ${} = {}$: not equal", e(11), num(a, 2), e(22), num(d, 2))
            } else {
                format!("${} = {}$, ${} = {}$: not both $0$", e(12), num(b, 2), e(21), num(c, 2))
            }
        }
        Kind::Diagonal => {
            if holds {
                format!("${} = {} = 0${ok}", e(12), e(21))
            } else {
                format!("${} = {}$, ${} = {}$: not both $0$", e(12), num(b, 2), e(21), num(c, 2))
            }
        }
        Kind::Upper | Kind::Lower => {
            let (ij, v) = if kind == Kind::Upper { (21, c) } else { (12, b) };
            if holds {
                format!("${} = 0${ok}", e(ij))
            } else {
                format!("${} = {}$, not $0$", e(ij), num(v, 2))
            }
        }
        Kind::Symmetric | Kind::PosDef | Kind::PosSemidef | Kind::Indefinite => {
            let off = (b - c).abs();
            if off <= EXACT * s {
                format!(r"${p}^\top = {n}${ok}")
            } else {
                format!(r"${p}^\top \ne {n}$: they differ by up to ${}$", small(off))
            }
        }
        Kind::Skew => line(format!(r"${p}^\top = -{p}$"), format!(r"${p}^\top \ne -{p}$")),
        Kind::ScaledRotation => {
            if holds {
                format!("${} = {}$, ${} = -{}${ok}", e(11), e(22), e(21), e(12))
            } else {
                format!("${} - {} = {}$, ${} + {} = {}$: not both $0$", e(11), e(22), num(a - d, 2), e(21), e(12), num(b + c, 2))
            }
        }
        Kind::Orthogonal => line(format!(r"${p}^\top {p} = I$"), format!(r"${p}^\top {p} \ne I$")),
        Kind::Rotation | Kind::Reflection => {
            let want = if kind == Kind::Rotation { "1" } else { "-1" };
            if holds {
                format!(r"${p}^\top {p} = I$, $\det {p} = {want}${ok}")
            } else if orthogonal {
                format!(r"${p}^\top {p} = I$, but $\det {p} = {}$", num(det.signum(), 0))
            } else {
                format!(r"${p}^\top {p} \ne I$: they differ by up to ${}$", small(size(m.transpose() * m - DMat2::IDENTITY)))
            }
        }
        Kind::DetOne | Kind::Singular | Kind::RankOne => {
            let want = if kind == Kind::DetOne { "1" } else { "0" };
            if holds {
                format!(r"$\det {p} = {want}${ok}")
            } else if kind == Kind::RankOne && size(m) <= EXACT * s {
                format!("${n} = 0$: rank $0$, not $1$")
            } else {
                format!(r"$\det {p} = {}$, not ${want}$", num(det, 2))
            }
        }
        Kind::Normal => line(format!(r"${p}{p}^\top = {p}^\top {p}$"), format!(r"${p}{p}^\top \ne {p}^\top {p}$")),
        Kind::Projection => line(format!("${p}^2 = {n}$"), format!(r"${p}^2 \ne {n}$")),
        Kind::OrthProjection => {
            let idem = size(m * m - m) <= EXACT * s * s;
            if holds {
                format!(r"${p}^2 = {n}$, ${p}^\top = {n}${ok}")
            } else if idem {
                format!(r"${p}^2 = {n}$, but ${p}^\top \ne {n}$: slanted")
            } else {
                format!(r"${p}^2 \ne {n}$: they differ by up to ${}$", small(size(m * m - m)))
            }
        }
        Kind::Involution => line(format!("${p}^2 = I$"), format!(r"${p}^2 \ne I$")),
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
        Kind::Root => {
            let (k, near, _) = order(m);
            if holds {
                format!("${p}^{{{k}}} = I${ok}")
            } else {
                format!(r"${p}^n \ne I$ for $n \le {ROOT_MAX}$: nearest, ${p}^{{{k}}}$, by ${}$", small(near))
            }
        }
        Kind::Positive => {
            let low = a.min(b).min(c).min(d);
            if holds {
                format!(r"every entry $> 0$ (the least is ${}$){ok}", num(low, 2))
            } else {
                format!(r"an entry is ${}$: not above $0$", num(low, 2))
            }
        }
        Kind::Markov | Kind::Doubly => {
            let (s1, s2, low) = (a + c, b + d, a.min(b).min(c).min(d));
            let near = |v: f64| (v - 1.0).abs() <= EXACT * s;
            let sums = if near(s1) && near(s2) { r"columns add to $1$".to_string() } else { format!("columns add to ${}$ and ${}$", num(s1, 2), num(s2, 2)) };
            let rows_too = match kind {
                Kind::Doubly if near(a + b) && near(c + d) => r", rows too".to_string(),
                Kind::Doubly => format!(", rows to ${}$ and ${}$", num(a + b, 2), num(c + d, 2)),
                _ => String::new(),
            };
            let signs = if low >= 0.0 { r"no entry below $0$".to_string() } else { format!("an entry is ${}$", num(low, 2)) };
            format!("{sums}{rows_too}, {signs}{}", if holds { ok } else { "" })
        }
        Kind::Dominant => {
            if holds {
                format!("$|{}| > |{}|$, $|{}| > |{}|${ok}", e(11), e(12), e(22), e(21))
            } else if a.abs() - b.abs() <= d.abs() - c.abs() {
                format!(r"$|{}| = {} \le |{}| = {}$", e(11), num(a.abs(), 2), e(12), num(b.abs(), 2))
            } else {
                format!(r"$|{}| = {} \le |{}| = {}$", e(22), num(d.abs(), 2), e(21), num(c.abs(), 2))
            }
        }
        Kind::Permutation => {
            if holds {
                if size(m - DMat2::IDENTITY) <= EXACT * s {
                    format!("${n} = I${ok}")
                } else {
                    format!("${n} = S$, the swap{ok}")
                }
            } else {
                format!(r"neither $I$ nor $S$: off by ${}$", small(off))
            }
        }
        Kind::Companion => {
            if holds {
                format!("${} = 0$, ${} = 1${ok}", e(11), e(21))
            } else {
                format!("${} = {}$, ${} = {}$: not $0$ and $1$", e(11), num(a, 2), e(21), num(c, 2))
            }
        }
        Kind::JordanBlock => {
            if holds {
                format!("${} = 0$, ${} = 1$, ${} = {}${ok}", e(21), e(12), e(11), e(22))
            } else {
                format!("${} = {}$, ${} = {}$, ${} - {} = {}$", e(21), num(c, 2), e(12), num(b, 2), e(11), e(22), num(a - d, 2))
            }
        }
        Kind::Any => "any matrix: no equation is kept".into(),
    };
    let second = match kind {
        Kind::PosDef | Kind::PosSemidef => {
            let low = least_form(m);
            let rel = match kind {
                Kind::PosDef if low > EXACT * s => r"> 0$ $\checkmark",
                Kind::PosDef => r"\le 0",
                _ if low >= -EXACT * s => r"\ge 0$ $\checkmark",
                _ => r"< 0",
            };
            Some(format!(r"least $x \cdot {p}x$ with $|x| = 1$: ${} {rel}$", num(low, 2)))
        }
        Kind::Indefinite => {
            let rel = if det < -EXACT * s * s { r"< 0$ $\checkmark" } else { r"\ge 0" };
            Some(format!(r"$\det {p} = {} {rel}$", num(det, 2)))
        }
        Kind::Orthogonal if holds => Some(if det > 0.0 { format!(r"a rotation: $\det {p} = 1$") } else { format!(r"a reflection: $\det {p} = -1$") }),
        Kind::Nilpotent if holds => Some(if size(m) <= EXACT { format!("here ${n} = 0$ itself") } else { format!(r"and ${n} \ne 0$") }),
        Kind::Normal if holds => Some(if (b - c).abs() <= EXACT * s { "here: symmetric".to_string() } else { r"here: a scaled rotation, $aI + bJ$".to_string() }),
        Kind::ScaledRotation if holds => Some(format!("the complex number ${} {} {}i$", num(a, 2), if c < 0.0 { "-" } else { "+" }, num(c.abs(), 2))),
        Kind::Involution if holds => Some(
            if size(m - DMat2::IDENTITY) <= EXACT * s {
                format!("here ${n} = I$ itself")
            } else if size(m + DMat2::IDENTITY) <= EXACT * s {
                format!("here ${n} = -I$: a half turn")
            } else if orthogonal {
                r"and $A^\top A = I$: a mirror".to_string()
            } else {
                r"but $A^\top A \ne I$: a slanted mirror".to_string()
            },
        ),
        Kind::Root if holds => {
            let (k, _, _) = order(m);
            Some(match eigen(m) {
                Eigen::Complex { re, im } => {
                    let (turn, of) = share_of_turn(im.atan2(re)).unwrap_or((1, k));
                    let frame = if orthogonal { "a rotation" } else { "in a slanted frame" };
                    format!("a turn by {turn}/{of} of a full turn, {frame}")
                }
                _ if k == 1 => "the identity".to_string(),
                _ if size(m + DMat2::IDENTITY) <= EXACT * s => r"$-I$: a half turn".to_string(),
                _ => r"an involution: eigenvalues $1$ and $-1$".to_string(),
            })
        }
        Kind::Dominant => Some(if holds { r"so $0$ lies in neither Gershgorin disc".to_string() } else { "so the discs may hold 0".to_string() }),
        Kind::Positive => Some("(not the same as positive definite)".to_string()),
        Kind::Companion => Some(format!(r"$\lambda^2 - {}\lambda {} {} = 0$", num(d, 2), if b > 0.0 { "-" } else { "+" }, num(b.abs(), 2))),
        Kind::RankOne if holds => {
            let (u, v) = rank_one_factors(m);
            Some(format!(r"$u = {}$, $v = {}$", tuple(&[u.x, u.y], 2), tuple(&[v.x, v.y], 2)))
        }
        _ => None,
    };
    (first, holds, second)
}

/// A rank-one matrix as u vᵀ: u along its longer column, of length 1 (and pointing to
/// the right, or up), v what it must then be.
fn rank_one_factors(m: DMat2) -> (DVec2, DVec2) {
    let col = if m.x_axis.length_squared() >= m.y_axis.length_squared() { m.x_axis } else { m.y_axis };
    let mut u = col.try_normalize().unwrap_or(DVec2::X);
    if u.x < 0.0 || (u.x == 0.0 && u.y < 0.0) {
        u = -u;
    }
    // v_j = u · (column j), since column j = u v_j
    (u, DVec2::new(u.dot(m.x_axis), u.dot(m.y_axis)))
}

/// What the matrix does to the plane, in words, with its numbers.
fn does(kind: Kind, m: DMat2) -> String {
    let [a, b, c, d] = entries(m);
    let e = eigen(m);
    let s = size(m).max(1.0);
    let zero = size(m) <= EXACT;
    let identity = size(m - DMat2::IDENTITY) <= EXACT * s;
    let stretch = |both: &str| match e {
        Eigen::Real { l1, l2, lines: Lines::Two(..) } => format!(r"stretches along two perpendicular lines, by $\lambda = {}$ and ${}${both}", num(l1, 2), num(l2, 2)),
        Eigen::Real { l1, .. } => format!("multiplies every vector by ${}$", num(l1, 2)),
        Eigen::Complex { .. } => "has complex eigenvalues".into(),
    };
    let turn = |what: &str| {
        let angle = c.atan2(a);
        if angle.to_degrees().abs() < 0.05 {
            "the identity: nothing moves".to_string()
        } else {
            format!("turns the plane by {}{what}", degrees(angle))
        }
    };
    let mirror = || {
        let mut line = 0.5 * c.atan2(a);
        if line < 0.0 {
            line += std::f64::consts::PI;
        }
        format!("reflects the plane in the line at {} to the $x$ axis", degrees(line))
    };
    match kind {
        _ if zero && matches!(kind, Kind::Scalar | Kind::Diagonal | Kind::Skew | Kind::Nilpotent | Kind::Singular | Kind::Projection | Kind::OrthProjection | Kind::ScaledRotation | Kind::Symmetric | Kind::PosSemidef | Kind::Normal) => {
            "$A = 0$: it sends everything to $0$".into()
        }
        Kind::Scalar => format!("multiplies every vector by ${}$", num(a, 2)),
        Kind::Diagonal => format!("multiplies the $x$ axis by ${}$ and the $y$ axis by ${}$", num(a, 2), num(d, 2)),
        Kind::Upper => format!("keeps the $x$ axis on itself, multiplying it by ${}$", num(a, 2)),
        Kind::Lower => format!("keeps the $y$ axis on itself, multiplying it by ${}$", num(d, 2)),
        Kind::Symmetric => stretch(""),
        Kind::PosDef => stretch(", both positive"),
        Kind::PosSemidef => stretch(", neither negative"),
        Kind::Indefinite => match e {
            Eigen::Real { l1, l2, .. } => format!(r"stretches the orange line by $\lambda = {}$ and reverses the purple one, at right angles to it, by ${}$", num(l1, 2), num(l2, 2)),
            Eigen::Complex { .. } => "has complex eigenvalues".into(),
        },
        Kind::Skew => {
            if c.abs() <= EXACT {
                "$A = 0$: it sends everything to $0$".into()
            } else {
                let way = if c > 0.0 { "anticlockwise" } else { "clockwise" };
                format!("turns every arrow a quarter turn {way} and multiplies its length by ${}$", num(c.abs(), 2))
            }
        }
        Kind::Orthogonal | Kind::Reflection if det_sign(m) < 0.0 => mirror(),
        Kind::Orthogonal | Kind::Rotation => turn(", keeping every length"),
        Kind::Reflection => mirror(),
        Kind::Normal | Kind::ScaledRotation if (b - c).abs() <= EXACT * s => stretch(""),
        Kind::Normal | Kind::ScaledRotation => format!("turns the plane by {} and multiplies every length by ${}$", degrees(c.atan2(a)), num(a.hypot(c), 2)),
        Kind::DetOne => "keeps every area, and does not turn shapes over".into(),
        Kind::Projection | Kind::OrthProjection if identity => r"$A = I$: nothing moves".into(),
        Kind::Projection => "drops every point along the purple line onto the orange one".into(),
        Kind::OrthProjection => "drops every point straight onto the orange line, at right angles to it".into(),
        Kind::Involution | Kind::Root | Kind::Permutation if identity => r"$A = I$: nothing moves".into(),
        Kind::Involution | Kind::Root if size(m + DMat2::IDENTITY) <= EXACT * s => r"$A = -I$: turns everything half a turn".into(),
        Kind::Involution if size(m.transpose() * m - DMat2::IDENTITY) <= EXACT * s * s => format!("{}: a mirror", mirror()),
        Kind::Involution => "keeps the orange line and reverses the purple one: done twice, nothing has moved".into(),
        Kind::Root => {
            let (k, _, _) = order(m);
            match e {
                Eigen::Complex { .. } => format!("brings every point back where it started after ${k}$ steps"),
                _ => "keeps the orange line and reverses the purple one: done twice, nothing has moved".into(),
            }
        }
        Kind::Nilpotent => "squashes the plane onto the orange line, and sends that line to $0$".into(),
        Kind::Shear => "keeps the orange line point for point, and slides every other point along it".into(),
        Kind::Positive => match e {
            Eigen::Real { l1, .. } => format!(r"stretches most along the orange line, by $\lambda = {}$, which runs through the positive quarter", num(l1, 2)),
            Eigen::Complex { .. } => "has complex eigenvalues".into(),
        },
        Kind::Markov => r"maps pairs of shares that add to $1$ (the segment from $(1, 0)$ to $(0, 1)$) to pairs that add to $1$".into(),
        Kind::Doubly => format!(r"maps shares to shares and keeps $(1, 1)$: each new coordinate is ${}$ of one old one and ${}$ of the other", num(a, 2), num(b, 2)),
        Kind::Dominant => "sends no direction to $0$: each diagonal entry outweighs the rest of its row".into(),
        Kind::Singular => "squashes the plane onto a line".into(),
        Kind::RankOne => r"squashes the plane onto one line: $Ax = (v \cdot x)\,u$".into(),
        Kind::Permutation => r"swaps the two coordinates: a mirror in the line $y = x$".into(),
        Kind::Companion => format!("sends $(1, 0)$ to $(0, 1)$, and $(0, 1)$ to its last column, ${}$", tuple(&[b, d], 2)),
        Kind::JordanBlock => format!(r"multiplies every point by $\lambda = {}$, and then slides it along the $x$ direction by its height", num(a, 2)),
        Kind::Any => String::new(),
    }
}

/// The sign of the determinant (1 for a matrix with none to speak of).
fn det_sign(m: DMat2) -> f64 {
    if m.determinant() < 0.0 {
        -1.0
    } else {
        1.0
    }
}

// ----- what survives sums, products, inverses and transposes ---------------------------

/// What is done with A (and B): it alone, a sum, a product, its inverse, its transpose.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Op {
    Alone,
    Sum,
    Product,
    Inverse,
    Transpose,
}

const OPS: [Op; 5] = [Op::Alone, Op::Sum, Op::Product, Op::Inverse, Op::Transpose];

impl Op {
    /// Its button.
    fn label(self) -> &'static str {
        match self {
            Op::Alone => "$A$ alone",
            Op::Sum => "$A + B$",
            Op::Product => "$AB$",
            Op::Inverse => "$A^{-1}$",
            Op::Transpose => r"$A^\top$",
        }
    }

    /// The matrix it makes, as written (without dollars).
    fn name(self) -> &'static str {
        match self {
            Op::Alone => "A",
            Op::Sum => "A + B",
            Op::Product => "AB",
            Op::Inverse => "A^{-1}",
            Op::Transpose => r"A^\top",
        }
    }

    fn uses_b(self) -> bool {
        matches!(self, Op::Sum | Op::Product)
    }

    /// "a sum of two symmetric matrices", "the inverse of a symmetric matrix".
    fn of(self, kind: Kind) -> String {
        let plural = kind.plural();
        match self {
            Op::Alone => kind.a_name(),
            Op::Sum => format!("a sum of two {plural}"),
            Op::Product => format!("a product of two {plural}"),
            Op::Inverse => format!("the inverse of {}", kind.a_name()),
            Op::Transpose => format!("the transpose of {}", kind.a_name()),
        }
    }
}

/// The matrix the picture is about: A, A + B, AB, A⁻¹ (none if A has no inverse) or Aᵀ.
fn combine(op: Op, a: DMat2, b: DMat2) -> Option<DMat2> {
    match op {
        Op::Alone => Some(a),
        Op::Sum => Some(a + b),
        Op::Product => Some(a * b),
        Op::Inverse => inverse(a),
        Op::Transpose => Some(a.transpose()),
    }
}

/// What a sum, a product, an inverse or a transpose of the kind can be counted on to be.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Keeps {
    /// Always of the kind.
    Always,
    /// Of the kind whenever it exists (an inverse).
    WhereItExists,
    /// Never of the kind (an inverse may exist, and is not).
    Never,
    /// An inverse never exists.
    NoInverse,
    /// Sometimes: under the condition the table states.
    Sometimes(&'static str),
}

impl Keeps {
    fn short(self) -> &'static str {
        match self {
            Keeps::Always => "always",
            Keeps::WhereItExists => "where it exists",
            Keeps::Never => "never",
            Keeps::NoInverse => "never: there is no inverse",
            Keeps::Sometimes(when) => when,
        }
    }
}

impl Kind {
    /// Whether a sum, a product, an inverse and a transpose of the kind are of the kind
    /// (what `closure` says in full).
    fn keeps(self, op: Op) -> Keeps {
        use Keeps::*;
        use Kind::*;
        let [sum, product, inv, transpose] = match self {
            Scalar | Diagonal => [Always, Always, WhereItExists, Always],
            Upper | Lower => [Always, Always, WhereItExists, Sometimes("only if diagonal")],
            Symmetric => [Always, Sometimes(r"exactly when $AB = BA$"), WhereItExists, Always],
            Skew => [Always, Sometimes(r"only if it is $0$"), WhereItExists, Always],
            Normal => [Sometimes("not always"), Sometimes("not always"), WhereItExists, Always],
            ScaledRotation => [Always, Always, WhereItExists, Always],
            Orthogonal | Rotation => [Sometimes("not always"), Always, Always, Always],
            Reflection => [Sometimes("if the mirrors are 60° apart"), Never, Always, Always],
            DetOne => [Sometimes("not always"), Always, Always, Always],
            Projection => [Sometimes(r"exactly when $AB = BA = 0$"), Sometimes(r"if $AB = BA$; not always"), Sometimes(r"only $I$ has one"), Always],
            OrthProjection => [Sometimes(r"exactly when $AB = 0$"), Sometimes(r"exactly when $AB = BA$"), Sometimes(r"only $I$ has one"), Always],
            Involution => [Sometimes("not always"), Sometimes(r"exactly when $AB = BA$"), Always, Always],
            Nilpotent => [Sometimes("not always"), Sometimes("not always"), NoInverse, Always],
            Shear => [Never, Sometimes("along one line: a shear or $I$"), Always, Always],
            Root => [Sometimes("not always"), Sometimes("not always"), Always, Always],
            PosDef => [Always, Sometimes(r"exactly when $AB = BA$"), Always, Always],
            PosSemidef => [Always, Sometimes(r"exactly when $AB = BA$"), WhereItExists, Always],
            Indefinite => [Sometimes("not always"), Never, Always, Always],
            Positive => [Always, Always, Never, Always],
            Markov => [Sometimes("never; an average is"), Always, Sometimes(r"only for $I$ and the swap"), Sometimes("only if doubly stochastic")],
            Doubly => [Sometimes("never; an average is"), Always, Sometimes(r"only for $I$ and the swap"), Always],
            Dominant => [Sometimes("not always"), Sometimes("not always"), Sometimes("exists; not always"), Sometimes("not always")],
            Singular => [Sometimes("not always"), Always, NoInverse, Always],
            RankOne => [Sometimes("not always"), Sometimes(r"rank one, or $0$"), NoInverse, Always],
            Permutation => [Never, Always, Always, Always],
            Companion => [Never, Sometimes("rarely"), Sometimes("only for the swap"), Sometimes(r"only if $a_{12} = 1$")],
            JordanBlock => [Never, Sometimes(r"if $\lambda + \mu = 1$"), Never, Never],
            Any => [Always; 4],
        };
        match op {
            Op::Sum => sum,
            Op::Product => product,
            Op::Inverse => inv,
            Op::Transpose | Op::Alone => transpose,
        }
    }

    /// What stays true of the kind under a sum, a product, an inverse or a transpose.
    fn closure(self, op: Op) -> &'static str {
        use Kind::*;
        match (op, self) {
            (Op::Alone, _) | (_, Any) => "",
            (Op::Sum, Scalar) => r"A sum of multiples of $I$ is one: $aI + bI = (a + b)I$.",
            (Op::Sum, Diagonal) => "A sum of diagonal matrices is diagonal.",
            (Op::Sum, Upper) => "A sum of upper triangular matrices is upper triangular.",
            (Op::Sum, Lower) => "A sum of lower triangular matrices is lower triangular.",
            (Op::Sum, Symmetric) => r"A sum of symmetric matrices is symmetric: $(A + B)^\top = A^\top + B^\top$.",
            (Op::Sum, Skew) => "A sum of skew-symmetric matrices is skew-symmetric.",
            (Op::Sum, Normal) => "A sum of normal matrices need not be normal.",
            (Op::Sum, ScaledRotation) => r"A sum of scaled rotations is one: $(aI + bJ) + (cI + dJ) = (a + c)I + (b + d)J$, as complex numbers add.",
            (Op::Sum, Orthogonal) => r"A sum of orthogonal matrices need not be orthogonal: $I + I = 2I$ doubles every length.",
            (Op::Sum, Rotation) => r"A sum of rotations need not be one: $I + I = 2I$ doubles every length (though turns by 0° and by 120° add to the turn by 60°).",
            (Op::Sum, Reflection) => "A sum of two reflections is a reflection only when their mirrors are 60° apart.",
            (Op::Sum, DetOne) => r"A sum need not have determinant $1$: $\det(I + I) = 4$.",
            (Op::Sum, Projection) => r"$A + B$ is a projection only if $AB = BA = 0$.",
            (Op::Sum, OrthProjection) => r"$A + B$ is an orthogonal projection exactly when $AB = 0$: onto lines at right angles, it is $I$.",
            (Op::Sum, Involution) => r"A sum of involutions need not be one: $I + I = 2I$.",
            (Op::Sum, Nilpotent) => "A sum of nilpotent matrices need not be nilpotent.",
            (Op::Sum, Shear) => r"A sum of two shears is never a shear: its diagonal adds up to $4$, a shear's to $2$.",
            (Op::Sum, Root) => r"A sum of roots of $I$ need not be one: $I + I = 2I$.",
            (Op::Sum, PosDef) => r"A sum of positive definite matrices is positive definite: $x \cdot (A + B)x = x \cdot Ax + x \cdot Bx$.",
            (Op::Sum, PosSemidef) => r"A sum of positive semidefinite matrices is one: $x \cdot (A + B)x = x \cdot Ax + x \cdot Bx \ge 0$.",
            (Op::Sum, Indefinite) => r"A sum of indefinite matrices need not be indefinite: $A + (-A) = 0$.",
            (Op::Sum, Positive) => "A sum of positive matrices is positive: each entry is a sum of two positive numbers.",
            (Op::Sum, Markov) => r"A sum of Markov matrices is not one: its columns add to $2$. Their average is one.",
            (Op::Sum, Doubly) => r"A sum of doubly stochastic matrices is not one: its rows and columns add to $2$. Their average is one.",
            (Op::Sum, Dominant) => r"A sum of diagonally dominant matrices need not be one: the diagonals can cancel, as in $I + (-I) = 0$.",
            (Op::Sum, Singular) => r"A sum of singular matrices need not be singular: $\diag(1, 0) + \diag(0, 1) = I$.",
            (Op::Sum, RankOne) => r"A sum of two matrices of rank one is most often of rank two: $\diag(1, 0) + \diag(0, 1) = I$.",
            (Op::Sum, Permutation) => r"A sum of permutation matrices is not one: its rows add to $2$.",
            (Op::Sum, Companion) => r"A sum of companion matrices is not one: its $a_{21}$ is $2$.",
            (Op::Sum, JordanBlock) => r"A sum of Jordan blocks is not one: its $a_{12}$ is $2$.",
            (Op::Product, Scalar) => r"A product of multiples of $I$ is one: $(aI)(bI) = abI$.",
            (Op::Product, Diagonal) => "A product of diagonal matrices is diagonal, and the same in either order.",
            (Op::Product, Upper) => "A product of upper triangular matrices is upper triangular.",
            (Op::Product, Lower) => "A product of lower triangular matrices is lower triangular.",
            (Op::Product, Symmetric) => r"A product of symmetric matrices is symmetric exactly when they commute: $(AB)^\top = BA$.",
            (Op::Product, Skew) => r"In the plane a product of two skew-symmetric matrices is $(aJ)(bJ) = -abI$: symmetric, and skew only if it is $0$.",
            (Op::Product, Normal) => "A product of normal matrices need not be normal.",
            (Op::Product, ScaledRotation) => "A product of scaled rotations is one, the same in either order: the angles add and the sizes multiply, as complex numbers multiply.",
            (Op::Product, Orthogonal) => r"A product of orthogonal matrices is always orthogonal: $(AB)^\top AB = B^\top A^\top A B = I$.",
            (Op::Product, Rotation) => "A product of rotations is the rotation by the sum of their angles, in either order.",
            (Op::Product, Reflection) => r"A product of two reflections is never a reflection: its determinant is $(-1)(-1) = 1$. It is the rotation by twice the angle between the mirrors.",
            (Op::Product, DetOne) => r"$\det AB = \det A \det B = 1$: a product of matrices of determinant $1$ has determinant $1$.",
            (Op::Product, Projection) => "A product of projections is a projection if they commute, and need not be otherwise.",
            (Op::Product, OrthProjection) => "A product of orthogonal projections is one exactly when they commute; otherwise it is not even a projection.",
            (Op::Product, Involution) => r"$(AB)^2 = I$ exactly when $AB = BA$: then $ABAB = AABB = I$.",
            (Op::Product, Nilpotent) => "A product of nilpotent matrices need not be nilpotent.",
            (Op::Product, Shear) => r"A product of two shears along the same line is a shear (or $I$); along different lines it is not.",
            (Op::Product, Root) => r"A product of roots of $I$ need not be one: two mirrors make the turn by twice the angle between them, which comes back to $I$ only for some angles.",
            (Op::Product, PosDef) => "A product of positive definite matrices is positive definite exactly when they commute (its eigenvalues are positive either way).",
            (Op::Product, PosSemidef) => "A product of positive semidefinite matrices is one exactly when they commute; otherwise it is not symmetric.",
            (Op::Product, Indefinite) => r"A product of two indefinite matrices is never indefinite: $\det AB = \det A \det B > 0$.",
            (Op::Product, Positive) => "A product of positive matrices is positive: each entry is a sum of products of positive numbers.",
            (Op::Product, Markov) => "A product of Markov matrices is a Markov matrix.",
            (Op::Product, Doubly) => "A product of doubly stochastic matrices is doubly stochastic.",
            (Op::Product, Dominant) => r"A product need not be diagonally dominant: the square of the shear with $a_{12} = 0.9$ has $a_{12} = 1.8$, beside $a_{11} = 1$.",
            (Op::Product, Singular) => r"$\det AB = \det A \det B = 0$: a product is singular if either factor is.",
            (Op::Product, RankOne) => r"$(uv^\top)(wz^\top) = (v \cdot w)\,uz^\top$: of rank one again, or $0$ if $v \cdot w = 0$.",
            (Op::Product, Permutation) => r"A product of permutation matrices is one: two shuffles make a shuffle. In the plane $S^2 = I$.",
            (Op::Product, Companion) => r"$AB$ is a companion matrix only when $A$'s last column is $(0, 1)$.",
            (Op::Product, JordanBlock) => r"Two Jordan blocks with diagonals $\lambda$ and $\mu$ have a product with $a_{12} = \lambda + \mu$: a Jordan block only when that is $1$.",
            (Op::Inverse, Scalar) => r"The inverse of $cI$, where there is one ($c \ne 0$), is $(1/c)I$.",
            (Op::Inverse, Diagonal) => r"The inverse of a diagonal matrix, where there is one, is diagonal: $1/a_{11}$ and $1/a_{22}$.",
            (Op::Inverse, Upper) => "The inverse of an upper triangular matrix, where there is one, is upper triangular.",
            (Op::Inverse, Lower) => "The inverse of a lower triangular matrix, where there is one, is lower triangular.",
            (Op::Inverse, Symmetric) => "The inverse of a symmetric matrix, where there is one, is symmetric.",
            (Op::Inverse, Skew) => r"In the plane the inverse of $wJ$ ($w \ne 0$) is $-J/w$: skew-symmetric too.",
            (Op::Inverse, Normal) => "The inverse of a normal matrix, where there is one, is normal.",
            (Op::Inverse, ScaledRotation) => r"The inverse of $aI + bJ$, where there is one, turns back and divides by the size, as $1/(a + bi)$ does: a scaled rotation.",
            (Op::Inverse, Orthogonal) => r"The inverse of an orthogonal matrix is its transpose, $A^{-1} = A^\top$, and orthogonal too.",
            (Op::Inverse, Rotation) => "The inverse of a rotation turns back by the same angle: a rotation.",
            (Op::Inverse, Reflection) => "A reflection is its own inverse: reflecting twice puts everything back.",
            (Op::Inverse, DetOne) => r"$\det A^{-1} = 1/\det A = 1$.",
            (Op::Inverse, Projection) => r"A projection has no inverse unless it is $I$: it sends a whole line to $0$.",
            (Op::Inverse, OrthProjection) => r"An orthogonal projection has no inverse unless it is $I$: it sends a whole line to $0$.",
            (Op::Inverse, Involution) => r"An involution is its own inverse: $A^2 = I$ says $A^{-1} = A$.",
            (Op::Inverse, Nilpotent) => r"A nilpotent matrix never has an inverse: it sends a whole line to $0$.",
            (Op::Inverse, Shear) => r"The inverse of a shear slides back along the same line: $(I + N)^{-1} = I - N$.",
            (Op::Inverse, Root) => r"If $A^n = I$, then $A^{-1} = A^{n - 1}$, and $(A^{-1})^n = I$: a root of $I$ with the same $n$.",
            (Op::Inverse, PosDef) => r"The inverse of a positive definite matrix is positive definite: its eigenvalues are $1/\lambda$.",
            (Op::Inverse, PosSemidef) => "Where there is an inverse, the matrix is positive definite, and so is its inverse.",
            (Op::Inverse, Indefinite) => r"The inverse has the eigenvalues $1/\lambda$, of the same signs: indefinite too.",
            (Op::Inverse, Positive) => r"The inverse of a positive matrix, where there is one, has a negative entry: its entries are $a_{22}$, $-a_{12}$, $-a_{21}$ and $a_{11}$, each divided by $\det A$.",
            (Op::Inverse, Markov) => r"The inverse, where there is one, has columns adding to $1$ but a negative entry, unless $A$ is $I$ or the swap.",
            (Op::Inverse, Doubly) => r"The inverse, where there is one, has a negative entry, unless $A$ is $I$ or the swap.",
            (Op::Inverse, Dominant) => r"A diagonally dominant matrix always has an inverse ($0$ is in neither disc), but the inverse need not be diagonally dominant.",
            (Op::Inverse, Singular) => r"A singular matrix has no inverse: it sends a whole line to $0$.",
            (Op::Inverse, RankOne) => r"A matrix of rank one has no inverse: it sends a whole line to $0$.",
            (Op::Inverse, Permutation) => r"In the plane each is its own inverse: $I^{-1} = I$, $S^{-1} = S$.",
            (Op::Inverse, Companion) => r"The inverse of a companion matrix is one only for the swap, $a_{12} = 1$ and $a_{22} = 0$.",
            (Op::Inverse, JordanBlock) => r"The inverse of a Jordan block, where there is one, has $a_{12} = -1/\lambda^2$: never $1$, so it is not one (it is upper triangular).",
            (Op::Transpose, Scalar) => r"A multiple of $I$ is its own transpose.",
            (Op::Transpose, Diagonal) => "A diagonal matrix is its own transpose.",
            (Op::Transpose, Upper) => "The transpose of an upper triangular matrix is lower triangular; it is upper triangular too only if it is diagonal.",
            (Op::Transpose, Lower) => "The transpose of a lower triangular matrix is upper triangular; it is lower triangular too only if it is diagonal.",
            (Op::Transpose, Symmetric) => "A symmetric matrix is its own transpose.",
            (Op::Transpose, Skew) => r"The transpose of a skew-symmetric matrix is $-A$: skew-symmetric too.",
            (Op::Transpose, Normal) => r"The transpose of a normal matrix is normal: $AA^\top = A^\top A$ reads the same for $A^\top$.",
            (Op::Transpose, ScaledRotation) => r"The transpose of $aI + bJ$ is $aI - bJ$: the same turn the other way, as $a - bi$ is to $a + bi$.",
            (Op::Transpose, Orthogonal) => "The transpose of an orthogonal matrix is its inverse, and orthogonal too.",
            (Op::Transpose, Rotation) => "The transpose of a rotation is its inverse: the turn back.",
            (Op::Transpose, Reflection) => "A reflection is its own transpose, and its own inverse.",
            (Op::Transpose, DetOne) => r"$\det A^\top = \det A = 1$.",
            (Op::Transpose, Projection) => r"The transpose of a projection is a projection, $(A^\top)^2 = (A^2)^\top = A^\top$; onto another line, unless $A$ is symmetric.",
            (Op::Transpose, OrthProjection) => "An orthogonal projection is its own transpose.",
            (Op::Transpose, Involution) => r"$(A^\top)^2 = (A^2)^\top = I$: the transpose of an involution is one.",
            (Op::Transpose, Nilpotent) => r"$(A^\top)^2 = (A^2)^\top = 0$: the transpose of a nilpotent matrix is nilpotent.",
            (Op::Transpose, Shear) => "The transpose of a shear is a shear along the line at right angles to its own.",
            (Op::Transpose, Root) => r"$(A^\top)^n = (A^n)^\top = I$: the transpose of a root of $I$ is one.",
            (Op::Transpose, PosDef) => "A positive definite matrix is its own transpose.",
            (Op::Transpose, PosSemidef) => "A positive semidefinite matrix is its own transpose.",
            (Op::Transpose, Indefinite) => "An indefinite matrix is its own transpose.",
            (Op::Transpose, Positive) => "The transpose has the same entries: positive.",
            (Op::Transpose, Markov) => r"The transpose of a Markov matrix has rows adding to $1$; it is one only if $A$'s rows add to $1$ too (doubly stochastic).",
            (Op::Transpose, Doubly) => "A doubly stochastic matrix of the plane is symmetric: its own transpose.",
            (Op::Transpose, Dominant) => "The transpose need not be diagonally dominant: it is dominant by columns, and need not be by rows.",
            (Op::Transpose, Singular) => r"$\det A^\top = \det A = 0$.",
            (Op::Transpose, RankOne) => r"$(uv^\top)^\top = vu^\top$: of rank one too.",
            (Op::Transpose, Permutation) => r"In the plane each is its own transpose: $I^\top = I$, $S^\top = S$.",
            (Op::Transpose, Companion) => r"The transpose of a companion matrix is one only when $a_{12} = 1$.",
            (Op::Transpose, JordanBlock) => r"The transpose of a Jordan block is lower triangular, with its $1$ below the diagonal: not a Jordan block.",
        }
    }
}

/// The map the picture shows `power` steps in: Mᵏ when the power is a whole number k,
/// and between two, the matrices of the steps either side mixed in proportion.
fn powered(m: DMat2, power: f64) -> DMat2 {
    let p = if power.is_finite() { power.max(0.0).min(ROOT_MAX as f64) } else { 0.0 };
    let k = (p.floor() as u32).min(ROOT_MAX - 1);
    let t = p - k as f64;
    let lo = power_of(m, k);
    let hi = lo * m;
    let out = lo + (hi - lo) * t;
    if out.is_finite() {
        out
    } else {
        DMat2::ZERO
    }
}

/// Mᵏ, kept finite: entries far beyond anything drawn are cut back (a matrix with
/// entries up to 3 raised to the twelfth power can reach 6¹²).
fn power_of(m: DMat2, k: u32) -> DMat2 {
    let mut p = DMat2::IDENTITY;
    for _ in 0..k {
        p *= m;
        if size(p) > 1e6 {
            p = DMat2::from_cols_array(&p.to_cols_array().map(|v| v.max(-1e6).min(1e6)));
        }
    }
    p
}

// ----- keeping a matrix of its kind ----------------------------------------------------

/// The smallest positive eigenvalue a positive definite matrix may be dragged to.
const LEAST_POSITIVE: f64 = 0.05;
/// How far from the edge of their kinds the drags keep the kinds defined by a strict
/// inequality: the least entry of a positive matrix, how much a diagonal entry of a
/// dominant one outweighs the rest of its row, how far below 0 an indefinite one's
/// determinant stays.
const MARGIN: f64 = 0.05;
/// The range of p₁₁ for the projection P behind an involution 2P − I: |p₁₁(1 − p₁₁)| ≤
/// 1.5, so that a₂₁ = 2p₂₁ stays within the limits.
const INVOLUTION_X: (f64, f64) = (-0.82, 1.82);
/// The shortest first column a matrix of determinant 1 may be dragged to: its second
/// column is then at least 1/0.5 = 2 long, and still within the limits.
const LEAST_COLUMN: f64 = 0.5;

/// A after its column `j` is dragged to `p`: that column goes as near `p` as the kind
/// allows, and the rest changes only as much as it must for A to stay of its kind. Every
/// entry stays within ±`ENTRY_MAX`.
fn fit(kind: Kind, a: DMat2, j: usize, p: DVec2) -> DMat2 {
    let j = j.min(1);
    let p = if p.is_finite() { p.clamp(DVec2::splat(-ENTRY_MAX), DVec2::splat(ENTRY_MAX)) } else { a.col(j) };
    let p = if p.is_finite() { p } else { DVec2::ZERO };
    let a = within_limits(a);
    let fixed = |m: DMat2| within_limits(m);
    match (kind, j) {
        // (the ones that are not the same with the two coordinates swapped, and those
        // too simple to need it)
        (Kind::Any, _) => {
            let mut m = a;
            *m.col_mut(j) = p;
            m
        }
        (Kind::Scalar, 0) => DMat2::IDENTITY * p.x,
        (Kind::Scalar, _) => DMat2::IDENTITY * p.y,
        (Kind::Diagonal, 0) => DMat2::from_cols(DVec2::new(p.x, 0.0), DVec2::new(0.0, a.y_axis.y)),
        (Kind::Diagonal, _) => DMat2::from_cols(DVec2::new(a.x_axis.x, 0.0), DVec2::new(0.0, p.y)),
        (Kind::Upper, 0) => DMat2::from_cols(DVec2::new(p.x, 0.0), a.y_axis),
        (Kind::Upper, _) => DMat2::from_cols(DVec2::new(a.x_axis.x, 0.0), p),
        (Kind::Lower, 0) => DMat2::from_cols(p, DVec2::new(0.0, a.y_axis.y)),
        (Kind::Lower, _) => DMat2::from_cols(DVec2::new(a.x_axis.x, a.x_axis.y), DVec2::new(0.0, p.y)),
        (Kind::Markov, _) => {
            // the nearest point of the segment from (1, 0) to (0, 1)
            let t = (0.5 * (p.x - p.y + 1.0)).clamp(0.0, 1.0);
            let mut m = a;
            *m.col_mut(j) = DVec2::new(t, 1.0 - t);
            // (the other column too, should A have come from elsewhere)
            let k = 1 - j;
            let t = (0.5 * (m.col(k).x - m.col(k).y + 1.0)).clamp(0.0, 1.0);
            *m.col_mut(k) = DVec2::new(t, 1.0 - t);
            m
        }
        (Kind::Doubly, _) => {
            // [p, 1 − p; 1 − p, p]: the first column (p, 1 − p), the second (1 − p, p),
            // each on the segment of shares
            let (along, across) = if j == 0 { (p.x, p.y) } else { (p.y, p.x) };
            let t = (0.5 * (along - across + 1.0)).clamp(0.0, 1.0);
            rows([t, 1.0 - t, 1.0 - t, t])
        }
        (Kind::Positive, _) => {
            let mut m = DMat2::from_cols_array(&a.to_cols_array().map(|v| v.max(MARGIN)));
            *m.col_mut(j) = p.max(DVec2::splat(MARGIN));
            m
        }
        (Kind::Permutation, _) => {
            // the nearer of the two: the column dragged towards where I has it, or
            // towards where the swap has it
            let home = if j == 0 { DVec2::X } else { DVec2::Y };
            if p.distance_squared(home) <= p.distance_squared(DVec2::new(home.y, home.x)) {
                DMat2::IDENTITY
            } else {
                SWAP
            }
        }
        (Kind::Companion, 0) => DMat2::from_cols(DVec2::Y, a.y_axis),
        (Kind::Companion, _) => DMat2::from_cols(DVec2::Y, p),
        (Kind::JordanBlock, 0) => rows([p.x, 1.0, 0.0, p.x]),
        (Kind::JordanBlock, _) => rows([p.y, 1.0, 0.0, p.y]),
        (_, 0) => fixed(fit_first(kind, a, p)),
        // the second column dragged is the first of the matrix with its coordinates
        // swapped (every other kind is still itself after that swap)
        _ => fixed(SWAP * fit_first(kind, SWAP * a * SWAP, DVec2::new(p.y, p.x)) * SWAP),
    }
}

/// `fit` for the first column.
fn fit_first(kind: Kind, a: DMat2, p: DVec2) -> DMat2 {
    let l = ENTRY_MAX;
    // the sign a coordinate keeps when it must stay off 0: its own, or else the one it had
    let side = |v: f64, was: f64| if v != 0.0 { v.signum() } else if was < 0.0 { -1.0 } else { 1.0 };
    let unit = |p: DVec2| if p.length() > 1e-6 { p.normalize() } else { a.x_axis.try_normalize().unwrap_or(DVec2::X) };
    match kind {
        Kind::Symmetric => rows([p.x, p.y, p.y, a.y_axis.y]),
        Kind::Skew => {
            let w = p.y;
            rows([0.0, -w, w, 0.0])
        }
        Kind::ScaledRotation => DMat2::from_cols(p, J * p),
        Kind::Orthogonal => {
            let u = unit(p);
            let s = if a.determinant() < 0.0 { -1.0 } else { 1.0 };
            DMat2::from_cols(u, J * u * s)
        }
        Kind::Rotation => {
            let u = unit(p);
            DMat2::from_cols(u, J * u)
        }
        Kind::Reflection => {
            let u = unit(p);
            DMat2::from_cols(u, -(J * u))
        }
        Kind::Normal => {
            // the nearer of the two kinds a normal matrix of the plane can be: symmetric,
            // or a scaled rotation (the column dragged in both)
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
        Kind::DetOne => {
            // the first column where it is dragged (no shorter than LEAST_COLUMN); the
            // second the nearest point of the line det[p, q] = 1 to where it was, kept
            // within the limits by sliding along that line
            let u = unit(p);
            let p = if p.length() >= LEAST_COLUMN { p } else { u * LEAST_COLUMN };
            let foot = p.perp() / p.length_squared();
            let t0 = (a.y_axis - foot).dot(u);
            let (mut lo, mut hi) = (f64::NEG_INFINITY, f64::INFINITY);
            for (f, d) in [(foot.x, u.x), (foot.y, u.y)] {
                if d.abs() > 1e-12 {
                    let (t1, t2) = ((-l - f) / d, (l - f) / d);
                    lo = lo.max(t1.min(t2));
                    hi = hi.min(t1.max(t2));
                }
            }
            let t = if lo <= hi { t0.max(lo).min(hi) } else { 0.0 };
            DMat2::from_cols(p, foot + u * t)
        }
        Kind::PosDef | Kind::PosSemidef => {
            let mu = if kind == Kind::PosDef { LEAST_POSITIVE } else { 0.0 };
            let a11 = p.x.max(2.0 * mu).min(l);
            // (a₁₁ − μ)(a₂₂ − μ) ≥ a₂₁² keeps the smaller eigenvalue at μ or above, with
            // a₂₂ no larger than the limit
            let reach = ((a11 - mu) * (l - mu)).max(0.0).sqrt();
            let a21 = p.y.max(-reach).min(reach);
            let least = if a11 - mu > 0.0 { mu + a21 * a21 / (a11 - mu) } else { mu };
            let a22 = a.y_axis.y.max(least).min(l);
            rows([a11, a21, a21, a22])
        }
        Kind::Indefinite => {
            // a₁₁a₂₂ − a₂₁² ≤ −MARGIN, a₂₂ moved as little as that needs; a₂₁ kept far
            // enough from 0 for some a₂₂ within the limits to do (and, where a₁₁ is near
            // 0, far enough that a₂₂ need not leap from one side to the other)
            let a11 = p.x;
            let need = (2.0 * MARGIN - l * a11.abs()).max(0.0).sqrt();
            let a21 = side(p.y, a.x_axis.y) * p.y.abs().max(need);
            let bound = (a21 * a21 - MARGIN) / a11;
            let a22 = if a11 > 0.0 {
                a.y_axis.y.min(bound).max(-l)
            } else if a11 < 0.0 {
                a.y_axis.y.max(bound).min(l)
            } else {
                a.y_axis.y
            };
            rows([a11, a21, a21, a22])
        }
        Kind::Projection => projection_from(p, a.x_axis.y, 1.0 - l, l, l),
        Kind::OrthProjection => {
            // uuᵀ has the first column u₁u: on the circle through 0 and (1, 0)
            let centre = DVec2::new(0.5, 0.0);
            let off = p - centre;
            let was = a.x_axis - centre;
            let angle = if off.length() > 1e-9 { off.y.atan2(off.x) } else { was.y.atan2(was.x) };
            let u = DVec2::from_angle(0.5 * angle);
            DMat2::from_cols(u * u.x, u * u.y)
        }
        Kind::Involution => {
            // A = 2P − I, P a projection: P's first column is (p + (1, 0))/2, with
            // |p₁₁(1 − p₁₁)| ≤ 1.5 and |p₁₂| ≤ 1, so that A's entries are within ±3
            let q = (p + DVec2::X) * 0.5;
            let half = (a + DMat2::IDENTITY) * 0.5;
            projection_from(q, half.x_axis.y, INVOLUTION_X.0, INVOLUTION_X.1, 1.0) * 2.0 - DMat2::IDENTITY
        }
        Kind::Root => {
            let det = a.determinant();
            match eigen(a) {
                Eigen::Complex { re, im } if det > 0.0 => {
                    // keep the turn: trace 2 cos θ and determinant 1, the first column
                    // where it is dragged (kept far enough off the x axis for a₁₂)
                    let theta = im.atan2(re);
                    let (k, n) = share_of_turn(theta).unwrap_or((1, 4));
                    let (s, c) = (std::f64::consts::TAU * k as f64 / n as f64).sin_cos();
                    // (a₁₁ near enough c for a₁₂ to fit within the limits, and a₂₂ = 2c − a₁₁ too)
                    let w = (l * l - s * s).max(0.0).sqrt();
                    let a11 = p.x.max((c - w).max(2.0 * c - l).max(-l)).min((c + w).min(2.0 * c + l).min(l));
                    let reach = ((a11 - c) * (a11 - c) + s * s) / l;
                    let a21 = side(p.y, a.x_axis.y) * p.y.abs().max(reach).min(l);
                    rows([a11, -((a11 - c) * (a11 - c) + s * s) / a21, a21, 2.0 * c - a11])
                }
                // an involution (I and −I among them): keep it one
                _ => fit_first(Kind::Involution, a, p),
            }
        }
        Kind::Nilpotent => within_limits(nilpotent_from(p, a.x_axis.y, l)),
        Kind::Shear => {
            // A = I + N, with N nilpotent and Ne₁ = p − e₁ (kept small enough for A to
            // stay within the limit)
            let n = nilpotent_from(p - DVec2::X, a.x_axis.y, l - 1.0);
            DMat2::IDENTITY + n
        }
        Kind::Dominant => {
            // |a₁₁| ≥ |a₁₂| + MARGIN and |a₂₂| ≥ |a₂₁| + MARGIN: a₁₁ where it is dragged
            // (no nearer 0 than MARGIN), a₁₂ cut back to fit under it, a₂₂ pushed out to
            // stay above a₂₁
            let a11 = side(p.x, a.x_axis.x) * p.x.abs().max(MARGIN);
            let room = a11.abs() - MARGIN;
            let a12 = a.y_axis.x.max(-room).min(room);
            let a21 = p.y.max(-(l - MARGIN)).min(l - MARGIN);
            let a22 = side(a.y_axis.y, 1.0) * a.y_axis.y.abs().max(a21.abs() + MARGIN).min(l);
            rows([a11, a12, a21, a22])
        }
        Kind::Singular | Kind::RankOne => {
            // the second column moved onto the line of the first (left as it was where
            // the first is 0, which squashes everything onto it already)
            if p.length() <= 1e-9 {
                let col = if kind == Kind::RankOne && a.y_axis.length() <= 1e-9 { a.x_axis.try_normalize().unwrap_or(DVec2::X) * 0.05 } else { DVec2::ZERO };
                DMat2::from_cols(col, a.y_axis)
            } else {
                let u = p.normalize();
                DMat2::from_cols(p, u * u.dot(a.y_axis))
            }
        }
        // (handled in `fit`)
        _ => a,
    }
}

/// The projection whose first column is `p`, or as near it as the limits allow: the
/// first entry within `lo..hi`, and the entry `a₁₂` no larger in size than `off`. Ae₁ =
/// p keeps the line through p, and sends e₁ − p (the direction from p to (1, 0)) to 0:
/// A = p nᵀ / (n · p), n = (p_y, 1 − p_x), n · p = p_y. The second column p (1 − p_x)/p_y
/// grows without end as p nears the x axis (there the two lines meet): p is kept far
/// enough off it. `was` is the second entry of the first column before, whose sign is
/// kept when p is on the axis.
fn projection_from(p: DVec2, was: f64, lo: f64, hi: f64, off: f64) -> DMat2 {
    let x = p.x.max(lo).min(hi);
    let sign = if p.y != 0.0 {
        p.y.signum()
    } else if was < 0.0 {
        -1.0
    } else {
        1.0
    };
    let y = sign * p.y.abs().max((x * (1.0 - x)).abs() / off).max(1e-6);
    let q = DVec2::new(x, y);
    DMat2::from_cols(q, q * ((1.0 - x) / y))
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
        Kind::Scalar => DMat2::IDENTITY * (r(0.4, 1.8) * if r(0.0, 1.0) < 0.3 { -1.0 } else { 1.0 }),
        Kind::Diagonal => {
            let (x, y, s) = (r(0.4, 1.8), r(0.3, 1.6), r(0.0, 1.0));
            rows([if s < 0.2 { -x } else { x }, 0.0, 0.0, if s > 0.8 { -y } else { y }])
        }
        Kind::Upper => rows([r(0.4, 1.7), r(-1.2, 1.2), 0.0, r(-1.0, 1.4)]),
        Kind::Lower => rows([r(0.4, 1.7), 0.0, r(-1.2, 1.2), r(-1.0, 1.4)]),
        Kind::Symmetric | Kind::PosDef | Kind::PosSemidef | Kind::Indefinite => {
            let big = r(0.9, 1.9);
            let other = match kind {
                Kind::PosDef => r(0.25, 0.9),
                Kind::PosSemidef if r(0.0, 1.0) < 0.3 => 0.0,
                Kind::PosSemidef => r(0.0, 0.9),
                Kind::Indefinite => -r(0.3, 1.5),
                _ => r(-1.2, 1.2),
            };
            let q = turn(r(0.0, pi));
            q * DMat2::from_diagonal(DVec2::new(big, other)) * q.transpose()
        }
        Kind::Skew => J * (r(0.5, 1.5) * if r(0.0, 1.0) < 0.5 { -1.0 } else { 1.0 }),
        Kind::ScaledRotation => turn(r(0.3, 2.8) * if r(0.0, 1.0) < 0.5 { -1.0 } else { 1.0 }) * r(0.5, 1.6),
        Kind::Orthogonal | Kind::Rotation | Kind::Reflection => {
            let mirror = match kind {
                Kind::Rotation => false,
                Kind::Reflection => true,
                _ => r(0.0, 1.0) < 0.5,
            };
            if mirror {
                // the reflection in the line at the angle φ
                let phi = r(0.0, pi);
                rows([(2.0 * phi).cos(), (2.0 * phi).sin(), (2.0 * phi).sin(), -(2.0 * phi).cos()])
            } else {
                turn(r(0.35, 2.8) * if r(0.0, 1.0) < 0.5 { -1.0 } else { 1.0 })
            }
        }
        Kind::DetOne => {
            // a matrix with a determinant well clear of 0, made positive and scaled to 1
            let mut m = rows([r(-1.5, 1.5), r(-1.5, 1.5), r(-1.5, 1.5), r(-1.5, 1.5)]);
            for _ in 0..20 {
                if m.determinant().abs() > 0.4 {
                    break;
                }
                m = rows([r(-1.5, 1.5), r(-1.5, 1.5), r(-1.5, 1.5), r(-1.5, 1.5)]);
            }
            if m.determinant() < 0.0 {
                m.x_axis = -m.x_axis;
            }
            let d = m.determinant().max(0.4);
            m * (1.0 / d.sqrt())
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
        Kind::Involution => {
            // 2P − I for a projection P, tried until its entries are within the limits
            for _ in 0..20 {
                let m = random(Kind::Projection, rng) * 2.0 - DMat2::IDENTITY;
                if size(m) <= ENTRY_MAX {
                    return m;
                }
            }
            Kind::Involution.default_a()
        }
        Kind::Projection => {
            // the line kept, and the line sent to 0, at least 35° apart
            let kept = r(0.0, pi);
            let lost = kept + r(0.6, pi - 0.6);
            let (u, w) = (DVec2::from_angle(kept), DVec2::from_angle(lost));
            let n = w.perp();
            DMat2::from_cols(u * (n.x / n.dot(u)), u * (n.y / n.dot(u)))
        }
        Kind::OrthProjection => {
            let u = DVec2::from_angle(r(0.0, pi));
            DMat2::from_cols(u * u.x, u * u.y)
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
        Kind::Root => {
            // a turn by k/n of a full turn, seen in a slanted frame S (or an involution)
            if r(0.0, 1.0) < 0.2 {
                return random(Kind::Involution, rng);
            }
            let n = 3 + (r(0.0, 1.0) * 6.0).floor().min(5.0) as u32;
            let k = if n >= 5 && r(0.0, 1.0) < 0.4 { 2 } else { 1 };
            let s = rows([r(0.7, 1.3), r(-0.5, 0.5), r(-0.5, 0.5), r(0.7, 1.3)]);
            let m = s * turn(std::f64::consts::TAU * k as f64 / n as f64) * s.inverse();
            // (rebuilt from its first column, so that its trace and determinant are
            // exactly those of the turn)
            fit_first(Kind::Root, turn(std::f64::consts::TAU * k as f64 / n as f64), m.x_axis)
        }
        Kind::Positive => rows([r(0.1, 1.9), r(0.1, 1.9), r(0.1, 1.9), r(0.1, 1.9)]),
        Kind::Markov => {
            let (p, q) = (r(0.1, 0.95), r(0.05, 0.9));
            rows([p, q, 1.0 - p, 1.0 - q])
        }
        Kind::Doubly => {
            let p = r(0.05, 0.95);
            rows([p, 1.0 - p, 1.0 - p, p])
        }
        Kind::Dominant => {
            let (a, d) = (r(0.6, 2.0) * if r(0.0, 1.0) < 0.3 { -1.0 } else { 1.0 }, r(0.6, 2.0) * if r(0.0, 1.0) < 0.3 { -1.0 } else { 1.0 });
            rows([a, a.abs() * r(-0.9, 0.9), d.abs() * r(-0.9, 0.9), d])
        }
        Kind::Singular | Kind::RankOne => {
            let u = DVec2::from_angle(r(0.0, 2.0 * pi)) * r(0.6, 1.3);
            let v = DVec2::from_angle(r(0.0, 2.0 * pi)) * r(0.6, 1.3);
            DMat2::from_cols(u * v.x, u * v.y)
        }
        Kind::Permutation => {
            if r(0.0, 1.0) < 0.5 {
                DMat2::IDENTITY
            } else {
                SWAP
            }
        }
        Kind::Companion => rows([0.0, r(-1.5, 1.5), 1.0, r(-2.0, 2.0)]),
        Kind::JordanBlock => {
            let l = r(0.3, 1.8) * if r(0.0, 1.0) < 0.3 { -1.0 } else { 1.0 };
            rows([l, 1.0, 0.0, l])
        }
        Kind::Any => rows([r(-2.0, 2.0), r(-2.0, 2.0), r(-2.0, 2.0), r(-2.0, 2.0)]),
    };
    within_limits(m)
}

/// Another matrix of the kind, not too like `was` (or as unlike it as a few tries find).
fn another(kind: Kind, was: DMat2, rng: &mut Rng) -> DMat2 {
    if kind == Kind::Permutation {
        // (there are only two)
        return if size(was - DMat2::IDENTITY) <= EXACT { SWAP } else { DMat2::IDENTITY };
    }
    let mut best = random(kind, rng);
    for _ in 0..12 {
        if size(best - was) > 0.35 && is(kind, best) {
            break;
        }
        best = random(kind, rng);
    }
    best
}

// ----- the map of the zoo --------------------------------------------------------------

/// Which kinds contain which: (smaller, larger), every matrix of the first being one of
/// the second, for each pair with no kind between them (the rest follow along the
/// lines). Each is proved in the primer's way and held by a test below.
const EDGES: [(Kind, Kind); 30] = [
    (Kind::Scalar, Kind::Diagonal),
    (Kind::Scalar, Kind::ScaledRotation),
    (Kind::Diagonal, Kind::Upper),
    (Kind::Diagonal, Kind::Lower),
    (Kind::Diagonal, Kind::Symmetric),
    (Kind::JordanBlock, Kind::Upper),
    (Kind::Symmetric, Kind::Normal),
    (Kind::ScaledRotation, Kind::Normal),
    (Kind::Orthogonal, Kind::Normal),
    (Kind::Skew, Kind::ScaledRotation),
    (Kind::Rotation, Kind::ScaledRotation),
    (Kind::Rotation, Kind::Orthogonal),
    (Kind::Rotation, Kind::DetOne),
    (Kind::Shear, Kind::DetOne),
    (Kind::Reflection, Kind::Orthogonal),
    (Kind::Reflection, Kind::Indefinite),
    (Kind::Reflection, Kind::Involution),
    (Kind::Permutation, Kind::Orthogonal),
    (Kind::Permutation, Kind::Involution),
    (Kind::Permutation, Kind::Doubly),
    (Kind::Involution, Kind::Root),
    (Kind::Indefinite, Kind::Symmetric),
    (Kind::PosSemidef, Kind::Symmetric),
    (Kind::PosDef, Kind::PosSemidef),
    (Kind::OrthProjection, Kind::PosSemidef),
    (Kind::OrthProjection, Kind::Projection),
    (Kind::Doubly, Kind::Symmetric),
    (Kind::Doubly, Kind::Markov),
    (Kind::Nilpotent, Kind::Singular),
    (Kind::RankOne, Kind::Singular),
];

/// The map's places, in the order of `KINDS`: across a wide box `MAP_WIDE_BOX` (larger
/// kinds higher), and down a tall one `MAP_TALL_BOX` for a phone. Laid out so that no
/// two lines cross and no line passes through a box.
const MAP_WIDE_BOX: (f32, f32) = (11.0, 5.85);
const MAP_WIDE: [(f32, f32); 30] = [
    (4.30, 3.24), // multiple of I
    (5.29, 2.51), // diagonal
    (4.14, 1.39), // upper triangular
    (5.29, 1.42), // lower triangular
    (7.13, 1.13), // symmetric
    (3.28, 3.47), // skew-symmetric
    (2.57, 0.30), // normal
    (3.00, 2.43), // scaled rotation
    (0.53, 2.20), // orthogonal
    (2.26, 3.25), // rotation
    (2.61, 4.62), // reflection
    (2.16, 1.58), // det A = 1
    (10.49, 3.65), // projection
    (9.87, 4.59), // orthogonal projection
    (7.01, 3.58), // involution
    (10.20, 1.31), // nilpotent
    (1.69, 2.27), // shear
    (6.99, 2.71), // root of I
    (9.25, 3.62), // positive definite
    (9.82, 2.51), // positive semidefinite
    (5.76, 3.35), // indefinite
    (7.59, 5.30), // positive entries
    (8.63, 2.62), // Markov
    (8.19, 3.85), // doubly stochastic
    (8.70, 5.30), // diagonally dominant
    (9.66, 0.42), // singular
    (9.02, 1.36), // rank one
    (2.27, 5.47), // permutation
    (9.91, 5.53), // companion
    (4.20, 2.34), // Jordan block
];
const MAP_TALL_BOX: (f32, f32) = (4.4, 12.4);
const MAP_TALL: [(f32, f32); 30] = [
    (1.65, 4.25), // multiple of I
    (1.66, 3.26), // diagonal
    (2.95, 1.39), // upper triangular
    (1.59, 2.18), // lower triangular
    (0.43, 1.98), // symmetric
    (2.79, 4.49), // skew-symmetric
    (3.51, 0.25), // normal
    (3.35, 3.45), // scaled rotation
    (3.96, 5.21), // orthogonal
    (3.30, 6.27), // rotation
    (3.43, 9.24), // reflection
    (2.34, 5.51), // det A = 1
    (1.45, 9.49), // projection
    (1.13, 10.51), // orthogonal projection
    (2.08, 8.38), // involution
    (1.99, 11.15), // nilpotent
    (2.88, 7.23), // shear
    (1.71, 7.33), // root of I
    (0.30, 11.13), // positive definite
    (0.44, 9.47), // positive semidefinite
    (1.31, 5.30), // indefinite
    (1.01, 11.93), // positive entries
    (1.41, 6.49), // Markov
    (0.94, 8.13), // doubly stochastic
    (1.98, 11.99), // diagonally dominant
    (2.23, 10.35), // singular
    (2.87, 11.32), // rank one
    (3.92, 11.09), // permutation
    (3.31, 12.01), // companion
    (2.98, 2.40), // Jordan block
];

/// Where a kind sits on the map, in the units of its box.
fn map_place(kind: Kind, tall: bool) -> (f32, f32) {
    let k = index(kind);
    if tall {
        MAP_TALL[k]
    } else {
        MAP_WIDE[k]
    }
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
    /// The tabs: "any matrix", then the seven families.
    tabs: Vec<(Vec2, Vec2)>,
    /// Where the family's heading goes (its left, at its middle height); none on a phone.
    heading: Option<Vec2>,
    /// Each family's kinds' buttons, in the order of `Family::kinds`.
    chips: Vec<Vec<(Vec2, Vec2)>>,
    /// The line with the kind's equation.
    title: Vec2,
    /// How far down the verdict's lines may go.
    lines_end: f32,
    /// The matrices: the size of their numbers, how many to a row, where the first
    /// one's name goes, and how far apart they are across and down (none on a short
    /// panel, which has the title and the verdict alone).
    text: f32,
    per_row: usize,
    cells: Vec2,
    step: Vec2,
    rows: usize,
    /// The line saying how many numbers are free (none where room is short).
    facts: Option<Vec2>,
    /// Where the table of what stays true of the kind goes (none where room is short).
    table: Option<Vec2>,
    /// The complex plane, its title and the line under it (none on a short panel).
    chart: Option<ChartPlace>,
    /// The buttons: another A; another B (when there is a B).
    another: Option<((Vec2, Vec2), (Vec2, Vec2))>,
}

#[derive(Clone, Copy, Debug)]
struct ChartPlace {
    title: Vec2,
    rect: (Vec2, Vec2),
    line: Vec2,
}

/// Rows of the panel, in logical pixels.
const TAB_SIZE: f32 = 10.5;
const TAB_H: f32 = 25.0;
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
/// The least room the picture needs for the plane and the map one above the other.
const BOTH_W: f32 = 560.0;
const BOTH_H: f32 = 800.0;
/// The tabs' names: "any matrix" and the families'.
fn tab_names() -> [&'static str; 8] {
    let mut out = ["any matrix"; 8];
    for (k, f) in FAMILIES.iter().enumerate() {
        out[k + 1] = f.tab();
    }
    out
}

/// The kinds' buttons of a family flowed into rows across `width`: their rectangles,
/// from (0, 0), and how many rows they take.
fn chip_layout(kinds: &[Kind], width: f32) -> (Vec<(Vec2, Vec2)>, usize) {
    let (mut x, mut y, mut rows) = (0.0f32, 0.0f32, 1usize);
    let mut out = Vec::with_capacity(kinds.len());
    for kind in kinds {
        let w = text_width(kind.chip(), CHIP_SIZE) + 13.0;
        if x > 0.0 && x + w > width {
            (x, y, rows) = (0.0, y + CHIP_H + CHIP_GAP, rows + 1);
        }
        out.push((Vec2::new(x, y), Vec2::new(x + w, y + CHIP_H)));
        x += w + CHIP_GAP;
    }
    (out, rows)
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

/// What the picture's part of the canvas shows.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Show {
    /// The plane alone (no room for the map beside it, and the map not asked for).
    Plane,
    /// The plane, with the map of the zoo under it.
    Both,
    /// The map alone.
    Map,
}

#[derive(Clone, Debug)]
struct Layout {
    /// A phone held upright: the panel goes below the picture.
    narrow: bool,
    show: Show,
    /// The part of the canvas the plane has to itself (where it is shown).
    picture: (Vec2, Vec2),
    /// The map of the zoo, where it is shown.
    map: Option<(Vec2, Vec2)>,
    /// None on a canvas too small to spare the room (the numbers are in the side panel).
    panel: Option<Panel>,
    /// Where the caption goes: its left and right, and its foot.
    caption: Option<(f32, f32, f32)>,
}

impl Layout {
    /// The layout for a canvas of the size `view` whose top `inset` the application's
    /// buttons cover, with two matrices or one (`two`), and the map asked for alone
    /// or not (`map_alone`).
    fn of(view: Vec2, inset: f32, two: bool, map_alone: bool) -> Layout {
        let narrow = view.x < 620.0 && view.y >= 560.0;
        let pad = if narrow { 10.0 } else { 12.0 };
        let width = if narrow { view.x } else { PANEL_W };
        let inner = width - 2.0 * pad;
        // the tabs, two rows of four
        let tab_w = (inner - 3.0 * CHIP_GAP) / 4.0;
        let tabs: Vec<(Vec2, Vec2)> = (0..8)
            .map(|k| {
                let at = Vec2::new((k % 4) as f32 * (tab_w + CHIP_GAP), (k / 4) as f32 * (TAB_H + CHIP_GAP));
                (at, at + Vec2::new(tab_w, TAB_H))
            })
            .collect();
        let tabs_h = 2.0 * TAB_H + CHIP_GAP;
        // the kinds' buttons: room for the family that needs the most rows
        let laid: Vec<(Vec<(Vec2, Vec2)>, usize)> = FAMILIES.iter().map(|f| chip_layout(f.kinds(), inner)).collect();
        let chip_rows = laid.iter().map(|(_, r)| *r).max().unwrap_or(1);
        let chips_h = chip_rows as f32 * (CHIP_H + CHIP_GAP) - CHIP_GAP;
        let heading_h = if narrow { 0.0 } else { LINE_H };
        let (text, per_row) = matrix_fit(inner);
        let mat_h = matrix_size(2, 2, CELL, text).y;
        // (A, the sides of its equation; and B and what is made of the two, for two at once)
        let cells: usize = if two { 5 } else { 3 };
        let rows = cells.div_ceil(per_row);
        let cell_h = LABEL_H + mat_h + 8.0;
        let verdict_h = if narrow { VERDICT_LINES_NARROW } else { VERDICT_LINES } as f32 * LINE_H;
        let top_part = pad + tabs_h + 6.0 + heading_h + chips_h + 8.0 + TITLE_H + 4.0;
        let below = 6.0 + LINE_H + 8.0 + BUTTON_H + pad;
        let without = top_part + rows as f32 * cell_h + verdict_h + 6.0 + LINE_H + 4.0;
        let place = |min: Vec2, width: f32, chart_h: f32, extras: bool, short: bool| {
            let x = min.x + pad;
            let mut y = min.y + pad;
            let tabs = tabs.iter().map(|(lo, hi)| (*lo + Vec2::new(x, y), *hi + Vec2::new(x, y))).collect();
            y += tabs_h + 6.0;
            let heading = (heading_h > 0.0).then_some(Vec2::new(x, y + LINE_H * 0.5));
            y += heading_h;
            let chips = laid.iter().map(|(rects, _)| rects.iter().map(|(lo, hi)| (*lo + Vec2::new(x, y), *hi + Vec2::new(x, y))).collect()).collect();
            y += chips_h + 8.0;
            let title = Vec2::new(x, y);
            y += TITLE_H + 4.0;
            let cells_at = Vec2::new(x, y);
            let across = if per_row > 1 { (inner - matrix_w(text)) / (per_row - 1) as f32 } else { 0.0 };
            if short {
                // (the verdict's first line straight under the title)
                let max = Vec2::new(min.x + width, y + LINE_H + pad);
                return Panel { min, max, tabs, heading, chips, title, lines_end: y + LINE_H, text, per_row, cells: cells_at, step: Vec2::new(across, cell_h), rows: 0, facts: None, table: None, chart: None, another: None };
            }
            y += rows as f32 * cell_h + verdict_h;
            let lines_end = y;
            let facts = extras.then_some(Vec2::new(x, y));
            y += if extras { LINE_H + 2.0 } else { 0.0 };
            let table = extras.then_some(Vec2::new(x, y));
            y += if extras { 5.0 * TABLE_LINE + 6.0 } else { 0.0 };
            y += 6.0;
            let chart_title = Vec2::new(x, y);
            y += LINE_H + 4.0;
            let rect = (Vec2::new(x + 2.0, y), Vec2::new(min.x + width - pad - 2.0, y + chart_h));
            y += chart_h + 6.0;
            let line = Vec2::new(x, y);
            y += LINE_H + 8.0;
            let half = (inner - 8.0) * 0.5;
            let another = ((Vec2::new(x, y), Vec2::new(x + half, y + BUTTON_H)), (Vec2::new(x + half + 8.0, y), Vec2::new(x + inner, y + BUTTON_H)));
            let max = Vec2::new(min.x + width, y + BUTTON_H + pad);
            Panel { min, max, tabs, heading, chips, title, lines_end, text, per_row, cells: cells_at, step: Vec2::new(across, cell_h), rows, facts, table, chart: Some(ChartPlace { title: chart_title, rect, line }), another: Some(another) }
        };
        let whole = |caption| Layout { narrow, show: Show::Plane, picture: (Vec2::new(0.0, inset), view), map: None, panel: None, caption };
        if narrow {
            if map_alone {
                // the map above, and under it the panel cut short: the tabs, the kinds,
                // the title and the verdict
                let short_h = top_part + LINE_H + pad;
                let top = view.y - short_h;
                if top - inset < 240.0 {
                    return whole(None);
                }
                let mut panel = place(Vec2::new(0.0, top), view.x, 0.0, false, true);
                panel.max.y = view.y;
                let map = (Vec2::new(6.0, inset + 6.0), Vec2::new(view.x - 6.0, top - 6.0));
                return Layout { narrow, show: Show::Map, picture: (Vec2::new(0.0, inset), Vec2::new(view.x, top)), map: Some(map), panel: Some(panel), caption: None };
            }
            // the panel below the picture, with the caption at the picture's foot
            let chart_h = ((inner - 4.0) * 0.3).clamp(80.0, 110.0);
            let rest = without + below;
            // (the picture keeps at least the top 38 %)
            let top = (view.y - rest - chart_h).max(view.y * 0.38);
            let chart_h = (view.y - top - rest).min(chart_h);
            if chart_h < 60.0 {
                return whole(Some((0.0, view.x, view.y - 10.0)));
            }
            let mut panel = place(Vec2::new(0.0, top), view.x, chart_h, false, false);
            panel.max.y = view.y;
            return Layout { narrow, show: Show::Plane, picture: (Vec2::new(0.0, inset), Vec2::new(view.x, top)), map: None, panel: Some(panel), caption: Some((0.0, view.x, top - 8.0)) };
        }
        let top = inset + 12.0;
        // (beside the picture, if it is tall enough, there is room for the line of free
        // numbers and the table of what stays true of the kind, with a chart of a good
        // size under them)
        let extras_h = LINE_H + 2.0 + 5.0 * TABLE_LINE + 6.0;
        let extras = top + without + extras_h + 150.0 + below <= view.y;
        let above = without + if extras { extras_h } else { 0.0 };
        let chart_h = ((inner - 4.0) * 0.66).min(view.y - 12.0 - top - above - below);
        if view.x < 600.0 || chart_h < 90.0 {
            return whole(Some((0.0, view.x, view.y - 16.0)));
        }
        let panel = place(Vec2::new(view.x - width - 12.0, top), width, chart_h, extras, false);
        let region = (Vec2::new(0.0, inset), Vec2::new(panel.min.x - 8.0, view.y));
        let size = region.1 - region.0;
        if map_alone {
            let map = (region.0 + Vec2::new(10.0, 10.0), region.1 - Vec2::new(4.0, 10.0));
            return Layout { narrow, show: Show::Map, picture: region, map: Some(map), panel: Some(panel), caption: None };
        }
        if size.x >= BOTH_W && size.y >= BOTH_H {
            // the plane above, the map below
            let map_h = (size.y * 0.4).clamp(300.0, 400.0);
            let map = (Vec2::new(region.0.x + 10.0, region.1.y - map_h - 10.0), Vec2::new(region.1.x - 4.0, region.1.y - 10.0));
            let picture = (region.0, Vec2::new(region.1.x, map.0.y - 4.0));
            return Layout { narrow, show: Show::Both, picture, map: Some(map), panel: Some(panel), caption: Some((0.0, picture.1.x, picture.1.y - 10.0)) };
        }
        Layout { narrow, show: Show::Plane, picture: region, map: None, panel: Some(panel), caption: Some((0.0, region.1.x, view.y - 16.0)) }
    }

    /// The part of the canvas the scene's frame is fitted into: the picture's, less the
    /// heading at its top and the caption at its foot.
    fn frame_room(&self) -> (Vec2, Vec2) {
        let (min, max) = self.picture;
        // (on a phone the caption, two lines, sits at the picture's foot)
        let foot = if self.narrow { 50.0 } else { 56.0 };
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

/// The steps an experiment plays, I to A to A² (or on to Aⁿ = I for a root of I):
/// seconds each step takes, and the pause between them (shorter where there are many).
fn step_seconds(top: u32) -> (f32, f32) {
    if top <= 2 {
        (1.3, 0.8)
    } else {
        (0.8, 0.3)
    }
}

/// How many times the map is applied, `s` seconds into steps that go up to `top`.
fn steps_at(s: f32, top: u32) -> f64 {
    let (step, pause) = step_seconds(top);
    (0..top).map(|k| smoothstep((s - k as f32 * (step + pause)) / step) as f64).sum()
}

fn steps_length(top: u32) -> f32 {
    let (step, pause) = step_seconds(top);
    top as f32 * step + top.saturating_sub(1) as f32 * pause
}

/// A column of A carried along a straight line by an experiment: which column, from
/// where to where, in how many seconds, and how far it has gone.
#[derive(Clone, Copy, Debug)]
struct Tour {
    column: usize,
    from: DVec2,
    to: DVec2,
    seconds: f32,
    s: f32,
}

impl Tour {
    fn at(&self, s: f32) -> DVec2 {
        self.from.lerp(self.to, smoothstep(s / self.seconds) as f64)
    }
}

struct Trial {
    label: &'static str,
    look_for: &'static str,
    kind: Kind,
    /// The family whose kinds the panel shows.
    family: Family,
    op: Op,
    a: [f64; 4],
    b: [f64; 4],
    /// The white point, if it is shown.
    x: Option<[f64; 2]>,
    /// Play the steps I, A, A².
    steps: bool,
    /// Carry a column of A along a line: the column, where to, in how many seconds.
    tour: Option<(usize, [f64; 2], f32)>,
    /// The Gershgorin discs.
    discs: bool,
    /// The map of the zoo in place of the plane.
    map_alone: bool,
}

const TRIALS: [Trial; 5] = [
    Trial {
        label: "a complex number as a matrix",
        look_for: r"A scaled rotation $aI + bJ$ is the complex number $a + bi$ written as a matrix. Here $A$ is $1 + 0.5i$ (size 1.12, angle 26.6°) and $B$, in blue, is $1.2 + 0.9i$ (size 1.5, angle 36.9°). Their product $AB$ turns the letter by 63.4°, the sum of 26.6° and 36.9°, and stretches it by 1.68, the product of 1.12 and 1.5: the angles add and the sizes multiply, as when complex numbers multiply. In the complex plane each matrix's eigenvalue $a + bi$ is its own complex number, and the dot of $AB$ sits at $(1 + 0.5i)(1.2 + 0.9i) = 0.75 + 1.5i$. And $BA = AB$: scaled rotations commute.",
        kind: Kind::ScaledRotation,
        family: Family::Symmetry,
        op: Op::Product,
        a: [1.0, -0.5, 0.5, 1.0],
        b: [1.2, -0.9, 0.9, 1.2],
        x: None,
        steps: false,
        tour: None,
        discs: false,
        map_alone: false,
    },
    Trial {
        label: "an involution that is not a mirror",
        look_for: r"$A^2 = I$: done twice, the map puts everything back, so $A$ is its own inverse. Watch the steps: after one the letter is flipped, slanted and stretched; after two it is home. Yet $A$ is no mirror: it does not keep lengths ($A^\top A \ne I$; it sends $(0, 1)$ to $(1, -1)$, which is longer). It keeps the orange line, the $x$ axis ($\lambda = 1$), and reverses the purple one, through $(1, -2)$ ($\lambda = -1$), and the two are not at right angles. So involution and root of $I$ are lit, and orthogonal and reflection are not.",
        kind: Kind::Involution,
        family: Family::Powers,
        op: Op::Alone,
        a: [1.0, 1.0, 0.0, -1.0],
        b: [0.0, 1.0, 1.0, 0.0],
        x: None,
        steps: true,
        tour: None,
        discs: false,
        map_alone: false,
    },
    Trial {
        label: "Gershgorin discs",
        look_for: r"Each Gershgorin disc is centred on a diagonal entry, with the size of the other entry of its row as its radius, and every eigenvalue lies in one of the discs. Here $a_{12}$ grows from 0 to 2, and the first disc with it. While $a_{12} < 1$ the matrix is diagonally dominant: $0$ lies in neither disc, so $0$ is not an eigenvalue and $A$ has an inverse. Past 1 the first disc takes in $0$, and the light of diagonally dominant goes out. At $a_{12} = 2$ an eigenvalue reaches $0$: $A$ is singular, and the plane is squashed onto a line.",
        kind: Kind::Any,
        family: Family::Signs,
        op: Op::Alone,
        a: [1.0, 0.0, 0.5, 1.0],
        b: [0.6, -0.8, 0.9, 0.4],
        x: None,
        steps: false,
        tour: Some((1, [2.0, 1.0], 6.0)),
        discs: true,
        map_alone: false,
    },
    Trial {
        label: "positive entries",
        look_for: r"Every entry of $A$ is above $0$: positive entries, which is not positive definite ($A$ is not even symmetric). Then both eigenvalues are real, and the larger, $\lambda = 2$, is positive and larger in size than the other, $-0.5$. Its eigenline (orange) runs through the positive quarter of the plane: the eigenvector $(1, 1)$ has both entries positive. The other eigenline, through $(3, -2)$, does not. Watch the steps: $A^2$ stretches the orange line by 4 and the purple one by only 0.25, sixteen times less.",
        kind: Kind::Positive,
        family: Family::Signs,
        op: Op::Alone,
        a: [0.5, 1.5, 1.0, 1.0],
        b: [1.0, 0.4, 0.6, 0.8],
        x: None,
        steps: true,
        tour: None,
        discs: false,
        map_alone: false,
    },
    Trial {
        label: "one matrix, many kinds",
        look_for: r"The swap $S$ sends $(x, y)$ to $(y, x)$, and passes eleven of the thirty tests at once. It is a permutation; a reflection (in the line $y = x$), so orthogonal; symmetric, so normal; indefinite (its eigenvalues are $1$ and $-1$); an involution and a root of $I$ ($S^2 = I$); doubly stochastic, so Markov (every row and column adds to $1$); and a companion matrix ($a_{11} = 0$, $a_{21} = 1$). On the map each of these is lit, and so is every line between two of them. Go back to the plane and drag a column, and watch which lights go out.",
        kind: Kind::Any,
        family: Family::Named,
        op: Op::Alone,
        a: [0.0, 1.0, 1.0, 0.0],
        b: [0.6, -0.8, 0.9, 0.4],
        x: None,
        steps: false,
        tour: None,
        discs: false,
        map_alone: true,
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
    /// The family whose kinds the panel shows.
    family: Family,
    op: Op,
    glide: Glide,
    /// Seconds into the steps I, A, A², ... while they play (they start once the glide
    /// to a new setting has arrived), and how many steps they go to.
    steps: Option<(f32, u32)>,
    /// A column carried along a line, while it goes.
    tour: Option<Tour>,
    show_x: bool,
    eigenlines: bool,
    /// The Gershgorin discs in the complex plane, for any kind.
    discs: bool,
    /// The map of the zoo in place of the plane.
    map_alone: bool,
    /// The eigenlines' directions as last named: a line's name keeps its end. And where
    /// on the screen each name was put, which it follows through an ease (and whether
    /// it is still on its way, or must be put afresh: the view has changed).
    last_lines: [Option<DVec2>; 2],
    anchors: [Option<Vec2>; 2],
    anchors_moving: bool,
    anchors_reset: bool,
    rng: Rng,
    /// The heading's lines, the caption's and the map's title's, as last broken.
    heading: [Option<Broken>; 2],
    caption: Option<Broken>,
    map_title: Option<Broken>,
    /// The canvas last drawn on and how much of its top the application's buttons
    /// cover: they decide where the panel goes, and so where the picture is.
    canvas: Vec2,
    inset: f32,
    measured: bool,
    new_view: bool,
    /// The titles of the scenes the kinds turn up in.
    titles: Vec<(&'static str, &'static str)>,
}

impl Special {
    fn new() -> Self {
        let kind = Kind::Symmetric;
        let homes: Vec<&str> = KINDS.iter().map(|k| k.home().0).collect();
        let titles = crate::scenes::registry().into_iter().filter(|d| homes.contains(&d.id)).map(|d| (d.id, d.title)).collect();
        Special {
            kind,
            family: kind.family(),
            op: Op::Alone,
            glide: Glide::at(Setting { a: kind.default_a(), b: kind.default_b(), x: X_START, power: 1.0 }),
            steps: None,
            tour: None,
            show_x: false,
            eigenlines: true,
            discs: false,
            map_alone: false,
            last_lines: [None; 2],
            anchors: [None; 2],
            anchors_moving: false,
            anchors_reset: false,
            rng: Rng::new(7),
            heading: [None, None],
            caption: None,
            map_title: None,
            canvas: Vec2::new(1080.0, 1000.0),
            inset: 0.0,
            measured: false,
            new_view: false,
            titles,
        }
    }

    /// Set up experiment `k` of `TRIALS`: every setting it depends on, glided to.
    fn set_up(&mut self, k: usize) {
        let trial = &TRIALS[k];
        let (two, alone) = (self.op != Op::Alone, self.map_alone);
        self.kind = trial.kind;
        self.family = trial.family;
        self.op = trial.op;
        self.map_alone = trial.map_alone;
        self.new_view |= two != (self.op != Op::Alone) || alone != self.map_alone;
        let a = rows(trial.a);
        self.glide.to = Setting { a, b: rows(trial.b), x: DVec2::from(trial.x.unwrap_or([X_START.x, X_START.y])), power: if trial.steps { 0.0 } else { 1.0 } };
        self.show_x = trial.x.is_some();
        self.eigenlines = true;
        self.discs = trial.discs;
        self.steps = trial.steps.then(|| (0.0, self.steps_top(a)));
        self.tour = trial.tour.map(|(column, to, seconds)| Tour { column, from: a.col(column), to: DVec2::from(to), seconds, s: 0.0 });
    }

    /// How many steps the steps go to for the matrix `a`: twice, or for a root of I,
    /// until it is back at I.
    fn steps_top(&self, a: DMat2) -> u32 {
        match (self.kind, self.op) {
            (Kind::Root, Op::Alone) if is(Kind::Root, a) => order(a).0.max(2),
            _ => 2,
        }
    }

    /// Choose a kind: A stays if it is of that kind already (B too), and otherwise
    /// glides to the kind's own.
    fn choose(&mut self, kind: Kind) {
        self.kind = kind;
        if kind != Kind::Any {
            self.family = kind.family();
        }
        let to = &mut self.glide.to;
        if !is(kind, to.a) {
            to.a = kind.default_a();
        }
        if !is(kind, to.b) {
            to.b = kind.default_b();
        }
        (self.steps, self.tour) = (None, None);
        to.power = to.power.round().min(2.0);
    }

    /// Choose what is done with A and B.
    fn choose_op(&mut self, op: Op) {
        self.new_view |= (op == Op::Alone) != (self.op == Op::Alone);
        self.op = op;
        self.tour = None;
    }

    /// Show the map of the zoo in place of the plane, or the plane again.
    fn set_map_alone(&mut self, on: bool) {
        self.new_view |= on != self.map_alone;
        self.map_alone = on;
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
        self.glide.to.power = self.glide.to.power.round().min(2.0);
    }

    /// The setting as drawn.
    fn shown(&self) -> Setting {
        self.glide.shown
    }

    /// Move everything on: `dt` seconds of the scene's time (the steps, the tour) and
    /// `real_dt` of the clock's (a change of setting glides by the clock, so that it is
    /// seen through even while the scene's time stands still).
    fn advance(&mut self, dt: f32, real_dt: f32) {
        self.glide.step(real_dt as f64);
        if let Some((s, top)) = self.steps {
            if self.glide.arrived() {
                let s = s + dt.max(0.0);
                let power = steps_at(s, top);
                self.glide.now(|g| g.power = power);
                self.steps = (s < steps_length(top)).then_some((s, top));
            }
        }
        if let Some(mut tour) = self.tour {
            if self.glide.arrived() {
                tour.s += dt.max(0.0);
                let a = fit(self.kind, self.glide.to.a, tour.column, tour.at(tour.s.min(tour.seconds)));
                self.glide.now(|g| g.a = a);
                self.tour = (tour.s < tour.seconds).then_some(tour);
            }
        }
    }

    fn layout(&self, view: Vec2, inset: f32) -> Layout {
        Layout::of(view, inset, self.op != Op::Alone, self.map_alone)
    }

    /// The canonical view: the picture's frame in the middle of the part of the canvas
    /// it has to itself.
    fn the_view(&self) -> View {
        let lay = self.layout(self.canvas, self.inset);
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
            _ => format!("{}^{{{k}}}", paren(n)),
        }
    }

    /// The title of the scene with this id.
    fn title_of(&self, id: &str) -> &'static str {
        self.titles.iter().find(|(i, _)| *i == id).map_or("", |(_, t)| *t)
    }

    /// The links at the foot of the panel: where the chosen kind turns up, then the
    /// zoo's other scenes.
    fn links(&self) -> Vec<(&'static str, String)> {
        let (home, says) = self.kind.home();
        let first = if self.kind == Kind::Any { says.to_string() } else { format!("where {} turn up: {says}", self.kind.plural()) };
        let mut out = vec![(home, first)];
        for (id, text) in [("patterns", "the zoo of larger matrices, known by their pattern"), ("symskew", "every matrix as symmetric plus skew-symmetric"), ("project", "projections and mirrors"), ("commute", r"when $AB = BA$")] {
            if out.len() < 4 && !out.iter().any(|(i, _)| *i == id) {
                out.push((id, text.to_string()));
            }
        }
        out
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

/// A word with its first letter a capital.
fn capitalised(s: &str) -> String {
    let mut c = s.chars();
    match c.next() {
        Some(first) => first.to_uppercase().chain(c).collect(),
        None => String::new(),
    }
}

/// The names of the kinds lit in `lit`, joined: "symmetric, normal and orthogonal".
fn named_list(lit: &[bool; 30]) -> String {
    let names: Vec<&str> = KINDS.iter().zip(lit).filter(|(_, on)| **on).map(|(k, _)| k.chip()).collect();
    match names.len() {
        0 => "none of the thirty kinds".to_string(),
        1 => names[0].to_string(),
        n => format!("{} and {}", names[..n - 1].join(", "), names[n - 1]),
    }
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
        ui::note(ui, r"A kind of matrix is defined by an equation, or a condition: symmetric is $A^\top = A$, orthogonal is $A^\top A = I$, a projection is $A^2 = A$. The equation decides what the map can do to the plane (the yellow letter and circle show it) and where its eigenvalues can lie (the dots in the complex plane, inside the bright region). Thirty kinds of $2 \times 2$ matrix are here, in seven families, and the map of the zoo shows which kinds lie inside which. Pick a kind and drag the arrows' tips: the matrix stays of its kind. Or pick any matrix, and see which kinds it is.");

        ui::section(ui, "Things to try");
        for (k, trial) in TRIALS.iter().enumerate() {
            if ui::experiment(ui, trial.label, trial.look_for) {
                self.set_up(k);
            }
        }

        ui::section(ui, "Kind");
        let tabs = tab_names();
        let mut tab = if self.kind == Kind::Any { 0 } else { FAMILIES.iter().position(|f| *f == self.family).map_or(0, |p| p + 1) };
        if ui::choice(ui, &tabs, &mut tab) {
            if tab == 0 {
                self.choose(Kind::Any);
            } else if let Some(f) = FAMILIES.get(tab - 1) {
                self.family = *f;
                if let Some(first) = f.kinds().first().filter(|_| !f.kinds().contains(&self.kind)) {
                    self.choose(*first);
                }
            }
        }
        if self.kind == Kind::Any {
            ui::note(ui, "No kind is kept: drag the arrows or type the numbers freely. Every kind the matrix is of is lit, on the panel's buttons and on the map.");
        } else {
            let kinds = self.family.kinds();
            let names: Vec<&str> = kinds.iter().map(|k| k.chip()).collect();
            let mut chosen = kinds.iter().position(|k| *k == self.kind).unwrap_or(0);
            if ui::choice(ui, &names, &mut chosen) {
                self.choose(kinds[chosen]);
            }
            let (home, _) = self.kind.home();
            ui::note(ui, &format!("{}: {}. {}. Free numbers: {}. It turns up in “{}”, linked at the foot of this panel.", self.kind.equation("A"), does(self.kind, self.glide.to.a), capitalised(self.kind.eigen_words()), self.kind.free(), self.title_of(home)));
        }
        if ui.button(typeset_label(ui, if self.kind == Kind::Any { "another matrix" } else { "another of this kind" })).clicked() {
            self.another(false);
        }

        ui::section(ui, "Matrix $A$");
        if let Some((j, col, typed)) = matrix_edit(ui, self.glide.to.a) {
            let a = fit(self.kind, self.glide.to.a, j, col);
            if typed || self.kind == Kind::Permutation {
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
            ui::note(ui, r"Is a sum, a product, an inverse or a transpose of this kind of the same kind? Choose one: the picture and the equation are then about it, and $B$ is a second matrix of the kind (its arrows blue; drag their tips).");
        } else {
            ui::note(ui, self.kind.closure(self.op));
            if self.op.uses_b() && ui.button(typeset_label(ui, "another $B$")).clicked() {
                self.another(true);
            }
        }

        ui::section(ui, "Steps");
        let top = self.steps_top(self.glide.to.a);
        let mut power = self.glide.to.power as f32;
        if ui::slider(ui, "times the map is applied", &mut power, 0.0..=top as f32) {
            self.glide.now(|s| s.power = power as f64);
            (self.steps, self.tour) = (None, None);
        }
        let play = if top > 2 { format!("play: $I$, then once, twice, up to {top} times") } else { "play: $I$, then once, then twice".to_string() };
        if ui.button(typeset_label(ui, &play)).clicked() {
            self.glide.to.power = 0.0;
            self.steps = Some((0.0, top));
            self.tour = None;
        }
        ui::note(ui, r"From the plane as it was (0) to the map applied once (1) and twice (2); a root of $I$ goes on until it is back. Applied any other number of times, the columns of $A$ itself are the white arrows, and their tips can be dragged too.");

        ui::section(ui, "Show");
        ui::checkbox(ui, &mut self.eigenlines, "the eigenlines");
        ui::checkbox(ui, &mut self.show_x, r"the white point $x$ and $Ax$");
        ui::checkbox(ui, &mut self.discs, "the Gershgorin discs, for any kind");
        let mut alone = self.map_alone;
        if ui::checkbox(ui, &mut alone, "the map of the zoo in place of the plane").changed() {
            self.set_map_alone(alone);
        }

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
        key.push((pal::TEXT, "the complex plane, beside the picture or below it: the eigenvalues as dots, and the bright line, points or discs where this kind's can lie; a red ring round a dot that lies outside them"));
        key.push((pal::LINK, "the map of the zoo: each kind joined by a line to the larger kinds just above it that contain it; lit, the kinds the matrix is; outlined, the kind chosen (click one to choose it)"));
        ui::key(ui, &key);

        ui::section(ui, "Words");
        ui::term(ui, r"Transpose $A^\top$:", r"$A$ flipped over its diagonal: its rows become columns, $(A^\top)_{ij} = a_{ji}$. For every $x$ and $y$, $x \cdot Ay = A^\top x \cdot y$.");
        ui::term(ui, "Eigenvalue, eigenline:", r"$Av = \lambda v$: along an eigenline the map only stretches, by $\lambda$ (a negative $\lambda$ reverses it).");
        ui::term(ui, "Complex eigenvalues:", r"when no line is kept, the eigenvalues are a pair $a \pm bi$, mirror images across the real line. A real matrix's complex eigenvalues always come in such pairs.");
        ui::term(ui, "$I$, $J$, $S$:", r"$I$ leaves every point where it is; $J$ turns every vector a quarter turn anticlockwise, $J^2 = -I$; $S$ swaps the two coordinates.");
        ui::term(ui, "Commute:", r"$AB = BA$: the two maps give the same result in either order.");
        ui::term(ui, "Free numbers:", "how many of a matrix's numbers can be chosen freely, the rest then being fixed by the kind: 4 for any matrix, 3 for a symmetric one, 1 for a rotation.");
        ui::term(ui, "Gershgorin disc:", "a disc in the complex plane centred on a diagonal entry, its radius the sizes of the other entries of its row added up. Every eigenvalue lies in one of them.");
        ui::term(ui, "Contains:", "one kind contains another when every matrix of the second is one of the first: every rotation is orthogonal. The map draws a line for each such pair with no kind between them.");
        ui::term(ui, "Positive, positive definite:", r"positive means every entry is above $0$; positive definite means symmetric with $x \cdot Ax > 0$ for every $x \ne 0$. Neither implies the other.");

        ui::section(ui, "Numbers");
        let a = self.glide.to.a;
        ui::readout(ui, "kind", self.kind.name());
        ui::readout(ui, r"$\det A$, $\tr A$", format!("${}, {}$", num(a.determinant(), 2), num(trace(a), 2)));
        ui::readout(ui, "eigenvalues of $A$", eigen_text(eigen(a)));
        let lit = kinds_of(a);
        ui::readout(ui, "kinds $A$ is of", format!("{} of 30", lit.iter().filter(|v| **v).count()));
        if self.op != Op::Alone {
            let n = self.op.name();
            match self.tested() {
                Some(c) => {
                    ui::readout(ui, &format!("eigenvalues of ${n}$"), eigen_text(eigen(c)));
                    if self.kind != Kind::Any {
                        ui::readout(ui, &format!("${n}$ is {}", self.kind.name()), if is(self.kind, c) { "yes" } else { "no" });
                    }
                }
                None => ui::readout(ui, &format!("${n}$"), r"none: $\det A = 0$"),
            }
        }
        if self.kind != Kind::Any {
            ui::readout(ui, "off the equation by", format!("${}$", self.tested().map_or("?".to_string(), |c| small(test(self.kind, c).0))));
        }
        let links = self.links();
        let links: Vec<(&'static str, &str)> = links.iter().map(|(id, text)| (*id, text.as_str())).collect();
        ui::see_also(ui, &links);
    }

    fn frame(&mut self, f: &mut Frame) {
        // the panel's place follows the canvas, and the picture's place follows the panel's
        let view = f.view_size();
        let lay = self.layout(view, f.top_inset);
        let before = self.layout(self.canvas, self.inset);
        // (a canvas squeezed to a sliver is one the controls have opened over, on a
        // phone: the picture is not seen then, and is left as it was for when they close)
        let covered = view.x < 160.0;
        let changed = !covered && (lay.narrow != before.narrow || lay.show != before.show || lay.panel.is_some() != before.panel.is_some() || (view - self.canvas).abs().max_element() > 24.0);
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
        for kind in [plot::AXES, plot::MARKS, plot::BUTTONS, MAP_EDGES, MAP_BOXES, DISCS] {
            keep(plot::apart(f, kind, false));
        }
        f.dl.points_with(PointKind::Overlay, PointStyle { outline_px: 0.0, ..Default::default() });

        let pointer = f.ptr.pos.map(|p| p / f.px);
        let walls: Vec<(Vec2, Vec2)> = lay.panel.iter().map(|p| (p.min, p.max)).chain(lay.map).collect();
        if !covered && pointer.is_some_and(|p| walls.iter().any(|w| inside_rect(p, *w))) {
            // the panel and the map are for their own buttons: no moving the view from there
            f.handles.capture();
            f.handles.claim_clicks();
        }
        self.advance(f.dt, f.real_dt);

        let labels_before = f.dl.labels.len();
        let plates = if lay.show == Show::Map { [None, None] } else { self.picture(f, &lay) };
        let numbers_from = f.dl.labels.len();
        if lay.show != Show::Map {
            f.axes2();
        }
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
        // nothing written into the picture may show through the panel or the map
        let mut k = 0;
        f.dl.labels.retain(|l| {
            k += 1;
            k <= labels_before || l.backed || !walls.iter().any(|(lo, hi)| inside_rect(l.pos, (*lo - Vec2::splat(2.0), *hi + Vec2::splat(2.0))))
        });
        let lit = self.tested().map_or([false; 30], kinds_of);
        if let Some(rect) = lay.map {
            self.map(f, rect, &lit, lay.show == Show::Map);
        }
        // the switch between the plane and the map, in the picture's corner
        let corner = |rect: (Vec2, Vec2), w: f32| (Vec2::new(rect.1.x - w - 10.0, rect.0.y + 8.0), Vec2::new(rect.1.x - 10.0, rect.0.y + 8.0 + 22.0));
        match (lay.show, lay.map) {
            (Show::Map, Some(rect)) => {
                if plot::button(f, corner(rect, 120.0), "back to the plane", 11.5, false) {
                    self.set_map_alone(false);
                }
            }
            (Show::Both, Some(rect)) => {
                if plot::button(f, corner(rect, 100.0), "the map alone", 11.5, false) {
                    self.set_map_alone(true);
                }
            }
            _ => {
                let (lo, hi) = lay.picture;
                let at = (Vec2::new(hi.x - 130.0, hi.y - 34.0), Vec2::new(hi.x - 10.0, hi.y - 12.0));
                if lay.panel.is_some() && hi.x - lo.x > 300.0 && plot::button(f, at, "the map of the zoo", 11.5, false) {
                    self.set_map_alone(true);
                }
            }
        }
        if let Some(p) = &lay.panel {
            self.panel(f, p, &lit);
        }
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
        if let Some((lo, hi)) = lay.map {
            room.wall(lo - Vec2::splat(4.0), hi + Vec2::splat(4.0));
        }
        let heading = self.heading(f, lay);
        room.wall(heading.0, heading.1);
        let mut plates = [Some(heading), None];
        if let Some((left, right, bottom)) = lay.caption {
            let parts: &[&str] = if lay.narrow { &["yellow: the letter and the circle after the map", "dashed: before"] } else { &["yellow: a letter and the unit circle, after the map", "dashed: before it"] };
            // (clear of the button to the map in the corner)
            let right = if lay.show == Show::Plane && lay.panel.is_some() { right - 140.0 } else { right };
            if let Some(plate) = self.caption(f, parts, (left, right), bottom) {
                room.wall(plate.0, plate.1);
                plates[1] = Some(plate);
            }
        }
        if lay.show == Show::Plane && lay.panel.is_some() {
            let hi = lay.picture.1;
            room.wall(Vec2::new(hi.x - 136.0, hi.y - 40.0), hi);
        }

        // where each handle may go, where the kind does not leave it free
        if a_is_drawn || two {
            self.tracks(f);
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
        // (after a first step, where it was one step before stays faint, to compare with)
        if let Some(c) = tested.filter(|_| shown.power > 1.0) {
            let before = powered(c, shown.power.floor().max(1.0) - if shown.power.fract() == 0.0 { 1.0 } else { 0.0 });
            lines_for(f, Part::Steps).polyline(within(&F_OUTLINE.iter().chain(&F_OUTLINE[..1]).map(|p| before * *p).collect::<Vec<_>>(), box_lo, box_hi), 1.0, pal::YELLOW.a(0.3));
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
                // (a permutation leaps from I to the swap and back: the picture glides)
                let leaps = kind == Kind::Permutation;
                if second {
                    let b = fit(kind, self.glide.to.b, j, p);
                    if leaps {
                        self.glide.to.b = b;
                    } else {
                        self.glide.now(|s| s.b = b);
                    }
                } else {
                    let a = fit(kind, self.glide.to.a, j, p);
                    if leaps {
                        self.glide.to.a = a;
                    } else {
                        self.glide.now(|s| s.a = a);
                    }
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
        if drawn.is_some_and(|m| size(m) <= EXACT) {
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
    fn tracks(&self, f: &mut Frame) {
        let faint = |c: Color| c.a(0.4);
        let long = 3.4f32;
        let (pink, green) = if self.op == Op::Alone && (self.shown().power - 1.0).abs() < 1e-9 { (pal::PINK, pal::GREEN) } else { (pal::WHITE, pal::WHITE) };
        let batch = lines_for(f, Part::Factors);
        let x_axis = (Vec3::new(-long, 0.0, 0.0), Vec3::new(long, 0.0, 0.0));
        let y_axis = (Vec3::new(0.0, -long, 0.0), Vec3::new(0.0, long, 0.0));
        let shares = (Vec3::new(1.0, 0.0, 0.0), Vec3::new(0.0, 1.0, 0.0));
        let ring = |centre: Vec2| -> Vec<Vec3> { (0..=48).map(|i| (centre + Vec2::from_angle(i as f32 / 48.0 * std::f32::consts::TAU) * 0.5).extend(0.0)).collect() };
        match self.kind {
            Kind::Diagonal | Kind::Scalar => {
                batch.segment(x_axis.0, x_axis.1, 2.0, faint(pink));
                batch.segment(y_axis.0, y_axis.1, 2.0, faint(green));
            }
            Kind::Upper => batch.segment(x_axis.0, x_axis.1, 2.0, faint(pink)),
            Kind::Lower => batch.segment(y_axis.0, y_axis.1, 2.0, faint(green)),
            Kind::Skew => {
                batch.segment(y_axis.0, y_axis.1, 2.0, faint(pink));
                batch.segment(x_axis.0, x_axis.1, 2.0, faint(green));
            }
            Kind::JordanBlock => {
                batch.segment(x_axis.0, x_axis.1, 2.0, faint(pink));
                batch.segment(Vec3::new(1.0, -long, 0.0), Vec3::new(1.0, long, 0.0), 2.0, faint(green));
            }
            Kind::Markov | Kind::Doubly => batch.segment(shares.0, shares.1, 3.0, faint(pal::TEXT)),
            Kind::OrthProjection => {
                batch.polyline(ring(Vec2::new(0.5, 0.0)), 2.0, faint(pink));
                batch.polyline(ring(Vec2::new(0.0, 0.5)), 2.0, faint(green));
            }
            _ => {}
        }
    }

    /// The heading at the top of the picture: what the map does, and in a second line
    /// what follows for its eigenvalues (or, for two at once, what stays true; for any
    /// matrix, the kinds it is). Returns the box it takes.
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
        let second = if tiny || (lay.narrow && self.op == Op::Alone && self.kind != Kind::Any) { String::new() } else { second };
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
        if kind == Kind::Any {
            let n = self.op.name();
            return match self.tested() {
                None => (r"$A$ has no inverse: $\det A = 0$".to_string(), String::new()),
                Some(c) => {
                    let lit = kinds_of(c);
                    let count = lit.iter().filter(|v| **v).count();
                    let of = if count == 1 { "1 kind".to_string() } else { format!("{count} kinds") };
                    (format!("Any matrix: ${n}$ is of {of}"), capitalised(&named_list(&lit)))
                }
            };
        }
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

    /// The screen panel: the families and their kinds, the equation with both sides
    /// worked out, the matrices, what survives, the complex plane with the eigenvalues,
    /// and buttons.
    fn panel(&mut self, f: &mut Frame, p: &Panel, lit: &[bool; 30]) {
        hud_panel(f, p.min, p.max);
        let kind = self.kind;
        let to = self.glide.to;
        let tested = self.tested();
        let mut clicked: Option<Kind> = None;
        let mut family_clicked: Option<Option<Family>> = None;

        // the tabs: any matrix, and the families, each with a light for each of its kinds
        for (k, (rect, name)) in p.tabs.iter().zip(tab_names()).enumerate() {
            let (on, pips): (bool, Vec<bool>) = if k == 0 {
                (kind == Kind::Any, Vec::new())
            } else {
                let fam = FAMILIES[k - 1];
                (fam == self.family, fam.kinds().iter().map(|kd| lit[index(*kd)]).collect())
            };
            if tab(f, *rect, name, on, k == 0 && kind == Kind::Any, &pips) {
                family_clicked = Some(if k == 0 { None } else { Some(FAMILIES[k - 1]) });
            }
        }
        if let Some(at) = p.heading {
            let words = if kind == Kind::Any { format!("{}: lit, the kinds $A$ is", capitalised(self.family.heading())) } else { capitalised(self.family.heading()) };
            f.text_px(at, words, 10.5, pal::DIM, Align2::LEFT_CENTER);
        }
        // the kinds of the family shown: the one chosen outlined, every kind the matrix is bright
        for (k, rect) in self.family.kinds().iter().zip(&p.chips[FAMILIES.iter().position(|x| *x == self.family).unwrap_or(0)]) {
            if chip(f, *rect, k.chip(), *k == kind, lit[index(*k)]) {
                clicked = Some(*k);
            }
        }
        let n = self.op.name();
        let mid = p.title.y + TITLE_H * 0.5;

        // the equation, the matrices, and the sides worked out
        let title = if kind == Kind::Any { format!("any matrix: ${n}$ is of {} kinds", lit.iter().filter(|v| **v).count()) } else { format!("{}: {}", kind.name(), kind.equation(n)) };
        // (smaller where it is long; and where even that is too wide, the name alone: the
        // equation is in the verdict under the matrices)
        let room = p.max.x - p.title.x - 10.0;
        let fits = |size: f32| f.text_size(&title, size).x <= room;
        let (title, size) = if fits(14.0) {
            (title, 14.0)
        } else if fits(12.0) {
            (title, 12.0)
        } else {
            (format!("{}:", kind.name()), 14.0)
        };
        f.text_px(Vec2::new(p.title.x, mid), title, size, pal::TEXT, Align2::LEFT_CENTER);
        let line = |f: &mut Frame, y: &mut f32, text: String, colour: Color| {
            if *y + LINE_H <= p.lines_end + 1.0 {
                f.text_px(Vec2::new(p.cells.x, *y + LINE_H * 0.5), text, 12.0, colour, Align2::LEFT_CENTER);
            }
            *y += LINE_H;
        };
        if p.rows == 0 {
            // (a panel cut short: the verdict's first line alone)
            let mut y = p.cells.y;
            if let Some(c) = tested {
                let (first, holds, _) = verdict(kind, c, n);
                line(f, &mut y, first, if holds { pal::TEXT } else { pal::RED });
            }
        } else {
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
            for (i, (name, m, colours)) in cells.iter().enumerate().take(p.rows * p.per_row) {
                let (row, col) = (i / p.per_row, i % p.per_row);
                let at = p.cells + Vec2::new(col as f32 * p.step.x, row as f32 * p.step.y);
                let w = matrix_w(s);
                f.text_px(at + Vec2::new(w * 0.5, LABEL_H * 0.5), name.as_str(), 12.5, pal::TEXT, Align2::CENTER_CENTER);
                let rows: Vec<Vec<(String, Color)>> = (0..2).map(|r| (0..2).map(|c| (cell(m.col(c)[r]), colours[c])).collect()).collect();
                matrix(f, at + Vec2::new(0.0, LABEL_H), &rows, s);
            }
            // the verdict, under the matrices
            let mut y = p.cells.y + cells.len().min(p.rows * p.per_row).div_ceil(p.per_row) as f32 * p.step.y - 2.0;
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
        }

        // how many numbers are free, and whether a sum, a product, an inverse and a
        // transpose are of the kind too
        if let Some(at) = p.facts {
            let text = if kind == Kind::Any { "free numbers: 4, every one".to_string() } else { format!("free numbers: {}", kind.free()) };
            f.text_px(at + Vec2::new(0.0, LINE_H * 0.5), text, 11.5, pal::DIM, Align2::LEFT_CENTER);
        }
        if let Some(at) = p.table {
            if kind == Kind::Any {
                f.text_px(at + Vec2::new(0.0, TABLE_LINE * 0.5), "of the kind too? (choose a kind to see)", 11.5, pal::DIM, Align2::LEFT_CENTER);
            } else {
                f.text_px(at + Vec2::new(0.0, TABLE_LINE * 0.5), format!("of the kind too? (two {})", kind.plural()), 11.5, pal::TEXT, Align2::LEFT_CENTER);
                for (k, op) in [Op::Sum, Op::Product, Op::Inverse, Op::Transpose].into_iter().enumerate() {
                    let y = at.y + (k + 1) as f32 * TABLE_LINE + TABLE_LINE * 0.5;
                    let answer = kind.keeps(op);
                    let colour = if matches!(answer, Keeps::Always | Keeps::WhereItExists) { pal::TEXT } else { pal::DIM };
                    f.text_px(Vec2::new(at.x + 8.0, y), op.label().replace(r"$A$ alone", ""), 11.5, pal::DIM, Align2::LEFT_CENTER);
                    f.text_px(Vec2::new(at.x + 62.0, y), answer.short(), 11.5, colour, Align2::LEFT_CENTER);
                }
            }
        }

        // the complex plane
        if let Some(place) = p.chart {
            self.chart(f, &place);
        }

        // the buttons
        if let Some((a_rect, b_rect)) = p.another {
            let label = if self.op.uses_b() { "another $A$" } else if kind == Kind::Any { "another matrix" } else { "another of this kind" };
            if plot::button(f, a_rect, label, 12.0, false) {
                self.another(false);
            }
            if self.op.uses_b() && plot::button(f, b_rect, "another $B$", 12.0, false) {
                self.another(true);
            }
        }
        if let Some(choice) = family_clicked {
            match choice {
                None => self.choose(Kind::Any),
                Some(fam) => self.family = fam,
            }
        }
        if let Some(k) = clicked {
            self.choose(k);
        }
    }

    /// The complex plane: the region where the kind's eigenvalues can lie, and the
    /// eigenvalues as dots (of A and B as rings, for two at once).
    fn chart(&self, f: &mut Frame, place: &ChartPlace) {
        let (min, max) = place.rect;
        let kind = self.kind;
        let shown = self.shown();
        let title = match self.op {
            Op::Alone => "eigenvalues, in the complex plane".to_string(),
            Op::Inverse | Op::Transpose => format!("eigenvalues of ${}$ (dots), of $A$ (rings)", self.op.name()),
            _ => format!("eigenvalues of ${}$ (dots), of $A$ and $B$ (rings)", self.op.name()),
        };
        f.text_px(place.title + Vec2::new(0.0, LINE_H * 0.5), title, 11.5, pal::TEXT, Align2::LEFT_CENTER);
        hud_rect(f, min, max, Color::hex(0x10172A));
        // one unit the same across and up, the middle at 0
        let (w, h) = (max.x - min.x, max.y - min.y);
        let unit = (w / 5.6).min(h / 3.4).max(1.0) as f64;
        let (hx, hy) = (w as f64 * 0.5 / unit, h as f64 * 0.5 / unit);
        let plot = Plot::new(place.rect, [-hx, hx], [-hy, hy]);
        // the axes, the unit circle, and the whole numbers along them
        plot::line_of(f, plot::AXES, plot.at(-hx, 0.0), plot.at(hx, 0.0), 1.0, pal::DIM.a(0.7));
        plot::line_of(f, plot::AXES, plot.at(0.0, -hy), plot.at(0.0, hy), 1.0, pal::DIM.a(0.7));
        let ring: Vec<Vec2> = (0..=72)
            .map(|i| {
                let t = i as f64 / 72.0 * std::f64::consts::TAU;
                plot.at(t.cos(), t.sin())
            })
            .collect();
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
        let tested = combine(self.op, shown.a, shown.b);
        // where this kind's eigenvalues can lie
        let bright = Color::hex(0xC8D4F0).a(0.42);
        let band = 5.0;
        let region = kind.region();
        match region {
            Region::Real | Region::Twice | Region::Perron | Region::ZeroAndReal => {
                plot::line_of(f, plot::MARKS, plot.at(-hx, 0.0), plot.at(hx, 0.0), band, bright);
                if region == Region::ZeroAndReal {
                    plot::disc(f, plot.at(0.0, 0.0), 18.0, bright);
                }
            }
            Region::Positive | Region::NonNegative => plot::line_of(f, plot::MARKS, plot.at(0.0, 0.0) + Vec2::new(if region == Region::Positive { 4.0 } else { 0.0 }, 0.0), plot.at(hx, 0.0), band, bright),
            Region::Sides => {
                plot::line_of(f, plot::MARKS, plot.at(-hx, 0.0), plot.at(0.0, 0.0) - Vec2::new(4.0, 0.0), band, bright);
                plot::line_of(f, plot::MARKS, plot.at(0.0, 0.0) + Vec2::new(4.0, 0.0), plot.at(hx, 0.0), band, bright);
            }
            Region::Imaginary => plot::line_of(f, plot::MARKS, plot.at(0.0, -hy), plot.at(0.0, hy), band, bright),
            Region::Circle => plot::path_of(f, plot::MARKS, &ring, band, bright),
            Region::DetOne => {
                plot::path_of(f, plot::MARKS, &ring, band, bright);
                plot::line_of(f, plot::MARKS, plot.at(-hx, 0.0), plot.at(0.0, 0.0) - Vec2::new(4.0, 0.0), band, bright);
                plot::line_of(f, plot::MARKS, plot.at(0.0, 0.0) + Vec2::new(4.0, 0.0), plot.at(hx, 0.0), band, bright);
            }
            Region::Roots => {
                // the n-th roots of 1 for this matrix's n (the circle faintly)
                plot::path_of(f, plot::MARKS, &ring, 1.5, bright.a(0.25));
                let n = tested.map_or(ROOT_MAX, |c| order(c).0.max(1));
                for k in 0..n {
                    let t = std::f64::consts::TAU * k as f64 / n as f64;
                    plot::disc(f, plot.at(t.cos(), t.sin()), 12.0, bright);
                }
            }
            Region::Points(points) => {
                for x in points {
                    plot::disc(f, plot.at(*x, 0.0), 18.0, bright);
                }
            }
            Region::Shares => {
                plot::line_of(f, plot::MARKS, plot.at(-1.0, 0.0), plot.at(1.0, 0.0), band, bright);
                plot::disc(f, plot.at(1.0, 0.0), 18.0, bright);
            }
            Region::Discs => {}
            Region::Anywhere => plot::fill(f, place.rect, bright.a(0.1)),
        }
        // the Gershgorin discs: the region itself for a dominant kind, or on request
        if let Some(c) = tested.filter(|_| self.discs || region == Region::Discs) {
            for (k, (centre, radius)) in discs(c).into_iter().enumerate() {
                let r = radius.min(1e3);
                // (the part inside the chart: a point outside breaks the line)
                let outline: Vec<Vec2> = (0..=64)
                    .map(|i| {
                        let t = i as f64 / 64.0 * std::f64::consts::TAU;
                        let q = plot.at(centre + r * t.cos(), r * t.sin());
                        if within_chart(q, place.rect) {
                            q
                        } else {
                            Vec2::NAN
                        }
                    })
                    .collect();
                let colour = if region == Region::Discs { bright } else { pal::CYAN.a(0.55) };
                plot::path_of(f, DISCS, &outline, if region == Region::Discs { 3.0 } else { 1.6 }, colour);
                // (named at its top, where that is in the chart)
                let label = plot.at(centre, r) + Vec2::new(0.0, -2.0);
                if within_chart(label - Vec2::new(0.0, 12.0), place.rect) && within_chart(label, place.rect) {
                    f.text_px(label, format!("row {}", k + 1), 9.5, if region == Region::Discs { pal::TEXT.a(0.8) } else { pal::CYAN.a(0.85) }, Align2::CENTER_BOTTOM);
                }
            }
        }
        // the region's name, in a corner
        let corner = if self.discs && region != Region::Discs { format!("{}; cyan: the Gershgorin discs", region.name()) } else { region.name().to_string() };
        f.text_px(Vec2::new(max.x - 5.0, min.y + 4.0), corner, 10.5, pal::TEXT.a(0.8), Align2::RIGHT_TOP);
        // the eigenvalues: of A and B as rings, of the matrix under test as dots
        let at = |(re, im): (f64, f64)| {
            let q = plot.at(re, im);
            let inside = within_chart(q, place.rect);
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
        match tested {
            Some(c) => {
                let e = eigen(c);
                let tol = 1e-6 * size(c).max(1.0);
                for (l, colour) in eigenvalues(e).into_iter().zip([pal::ORANGE, pal::PURPLE]).rev() {
                    let (q, inside) = at(l);
                    // (one where this kind's cannot be is ringed in red)
                    if !region.holds(l, tol, c) {
                        hud_circle(f, q, 9.0, 1.6, pal::RED);
                    }
                    plot::disc(f, q, if inside { 10.0 } else { 7.0 }, colour);
                }
                let shown_e = self.tested().map(eigen).unwrap_or(e);
                f.text_px(place.line + Vec2::new(0.0, LINE_H * 0.5), eigen_text(shown_e), 12.5, pal::TEXT, Align2::LEFT_CENTER);
            }
            None => f.text_px(place.line + Vec2::new(0.0, LINE_H * 0.5), "no inverse, so no eigenvalues to show", 12.0, pal::DIM, Align2::LEFT_CENTER),
        }
    }

    /// The map of the zoo in the rectangle `rect`: the kinds as boxes, each joined by a
    /// line to the kinds just above it that contain it; lit, the kinds the matrix is.
    /// A click on a kind chooses it.
    fn map(&mut self, f: &mut Frame, rect: (Vec2, Vec2), lit: &[bool; 30], alone: bool) {
        let (min, max) = rect;
        hud_panel(f, min, max);
        let narrow = max.x - min.x < 520.0;
        // the title, and what the lines mean
        let title = "The map of the zoo";
        f.text_px(min + Vec2::new(12.0, 14.0), title, 13.5, pal::TEXT, Align2::LEFT_CENTER);
        let count = lit.iter().filter(|v| **v).count();
        let name = if self.op == Op::Alone { "A".to_string() } else { self.op.name().to_string() };
        let lit_words = match count {
            0 => format!("${name}$ is none of them"),
            1 => format!("Lit: the one kind ${name}$ is"),
            _ => format!("Lit: the {count} kinds ${name}$ is"),
        };
        let says = format!(r"a line runs up from each kind to the larger kinds just above it that contain it. {lit_words}. Click a kind to choose it.");
        let room = (max.x - min.x - 24.0 - if alone || !narrow { 130.0 } else { 110.0 }).max(80.0);
        let (lines, _) = broken(f, &says, room, 10.5, &mut self.map_title);
        let mut y = min.y + 32.0;
        for l in &lines {
            f.text_px(Vec2::new(min.x + 12.0, y), l.as_str(), 10.5, pal::DIM, Align2::LEFT_CENTER);
            y += 14.0;
        }
        let tall = (max.y - min.y) > (max.x - min.x) * 1.1;
        let area = (Vec2::new(min.x + 8.0, y + 2.0), max - Vec2::new(8.0, 8.0));
        let size = if narrow { 9.5 } else { 10.5 };
        let boxes: Vec<(Vec2, Vec2)> = KINDS.iter().map(|k| map_box(*k, area, tall, size)).collect();
        let pointer = plot::pointer(f);
        let hovered = pointer.and_then(|q| boxes.iter().position(|b| plot::within(q, *b, 2.0)));
        // the lines, cut where they meet the boxes
        for (small, large) in EDGES {
            let (i, j) = (index(small), index(large));
            let (a, b) = (boxes[i], boxes[j]);
            let (p, q) = ((a.0 + a.1) * 0.5, (b.0 + b.1) * 0.5);
            let (p, q) = (leave_box(p, q, a), leave_box(q, p, b));
            let both = lit[i] && lit[j];
            let near = hovered.is_some_and(|h| h == i || h == j);
            let (width, colour) = if near {
                (1.8, pal::LINK.a(0.9))
            } else if both {
                (1.8, pal::TEXT.a(0.85))
            } else {
                (1.0, pal::DIM.a(0.4))
            };
            plot::line_of(f, MAP_EDGES, p, q, width, colour);
        }
        // the kinds
        let mut clicked = None;
        for (k, kind) in KINDS.iter().enumerate() {
            let (lo, hi) = boxes[k];
            let on = lit[k];
            let chosen = *kind == self.kind;
            let over = hovered == Some(k);
            let backing = match (on, over) {
                (true, _) => Color::hex(0x2B3D66),
                (false, true) => Color::hex(0x222B44),
                _ => Color::hex(0x141A28),
            };
            hud_rect(f, lo, hi, backing);
            let edge = [lo, Vec2::new(hi.x, lo.y), hi, Vec2::new(lo.x, hi.y), lo];
            let (width, colour) = if chosen {
                (1.8, pal::LINK)
            } else if over {
                (1.2, pal::LINK.a(0.7))
            } else if on {
                (1.0, pal::TEXT.a(0.6))
            } else {
                (1.0, pal::DIM.a(0.35))
            };
            plot::path_of(f, MAP_BOXES, &edge, width, colour);
            let text = if on { pal::WHITE } else { pal::DIM.a(0.85) };
            let [one, two] = kind.map_name();
            let middle = (lo + hi) * 0.5;
            if two.is_empty() {
                f.text_px(middle, one, size, text, Align2::CENTER_CENTER);
            } else {
                let half = (size + 2.0) * 0.5;
                f.text_px(middle - Vec2::new(0.0, half), one, size, text, Align2::CENTER_CENTER);
                f.text_px(middle + Vec2::new(0.0, half), two, size, text, Align2::CENTER_CENTER);
            }
            if over {
                f.handles.claim_clicks();
                f.handles.capture();
                if f.ptr.clicked {
                    f.ptr.clicked = false;
                    clicked = Some(*kind);
                }
            }
        }
        if let Some(k) = clicked {
            self.choose(k);
        }
    }
}

/// The batches of lines of the map's lines, of its boxes' outlines, and of the
/// Gershgorin discs.
const MAP_EDGES: u8 = plot::OWN;
const MAP_BOXES: u8 = plot::OWN + 1;
const DISCS: u8 = plot::OWN + 2;

/// A kind's place in `KINDS`.
fn index(kind: Kind) -> usize {
    KINDS.iter().position(|k| *k == kind).unwrap_or(0)
}

/// Where the segment from `p` (inside `rect`) to `q` leaves the rectangle.
fn leave_box(p: Vec2, q: Vec2, (lo, hi): (Vec2, Vec2)) -> Vec2 {
    let d = q - p;
    let mut t = 1.0f32;
    for (v, dv, a, b) in [(p.x, d.x, lo.x, hi.x), (p.y, d.y, lo.y, hi.y)] {
        if dv > 1e-6 {
            t = t.min((b - v) / dv);
        } else if dv < -1e-6 {
            t = t.min((a - v) / dv);
        }
    }
    p + d * t.max(0.0).min(1.0)
}

/// A kind's box on the map drawn in `area`, its names at the size `size`.
fn map_box(kind: Kind, (lo, hi): (Vec2, Vec2), tall: bool, size: f32) -> (Vec2, Vec2) {
    let (x, y) = map_place(kind, tall);
    let (w, h) = if tall { MAP_TALL_BOX } else { MAP_WIDE_BOX };
    let centre = lo + (hi - lo) * Vec2::new(x / w, y / h);
    let [one, two] = kind.map_name();
    let width = text_width(one, size).max(text_width(two, size)) + 12.0;
    let height = if two.is_empty() { size + 9.0 } else { 2.0 * size + 11.0 };
    let half = Vec2::new(width, height) * 0.5;
    (centre - half, centre + half)
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
    let backing = match (on, member, over) {
        (true, _, _) => Color::hex(0x2B3D66),
        (false, _, true) => Color::hex(0x2A3350),
        (false, true, false) => Color::hex(0x1E2840),
        _ => Color::hex(0x161C2B),
    };
    hud_rect(f, min, max, backing);
    let edge = [min, Vec2::new(max.x, min.y), max, Vec2::new(min.x, max.y), min];
    plot::path_of(f, plot::BUTTONS, &edge, if on { 1.4 } else { 1.0 }, pal::LINK.a(if on { 0.95 } else if over { 0.7 } else { 0.3 }));
    let colour = if member { pal::WHITE } else { pal::DIM.a(0.8) };
    f.text_px((min + max) * 0.5, text, CHIP_SIZE, colour, Align2::CENTER_CENTER);
    clicked_in(f, over)
}

/// One of the tabs: a family (with a light under its name for each of its kinds, lit if
/// the matrix is of that kind), or "any matrix". Returns true in the frame it is clicked.
fn tab(f: &mut Frame, (min, max): (Vec2, Vec2), text: &str, on: bool, free: bool, pips: &[bool]) -> bool {
    let over = plot::pointer(f).is_some_and(|q| plot::within(q, (min, max), 0.0));
    let backing = match (on, over) {
        (true, _) => Color::hex(0x2B3D66),
        (false, true) => Color::hex(0x2A3350),
        _ => Color::hex(0x1A2133),
    };
    hud_rect(f, min, max, backing);
    let edge = [min, Vec2::new(max.x, min.y), max, Vec2::new(min.x, max.y), min];
    plot::path_of(f, plot::BUTTONS, &edge, if on { 1.4 } else { 1.0 }, pal::LINK.a(if on { 0.95 } else if over { 0.7 } else { 0.35 }));
    let mid = (min + max) * 0.5;
    let text_at = if pips.is_empty() { mid } else { Vec2::new(mid.x, min.y + 9.0) };
    f.text_px(text_at, text, TAB_SIZE, if on || free { pal::WHITE } else { pal::TEXT.a(0.85) }, Align2::CENTER_CENTER);
    let gap = 6.0;
    let first = mid.x - gap * (pips.len() as f32 - 1.0) * 0.5;
    for (k, on) in pips.iter().enumerate() {
        let at = Vec2::new(first + gap * k as f32, max.y - 6.0);
        plot::disc(f, at, if *on { 5.0 } else { 3.5 }, if *on { pal::WHITE } else { pal::DIM.a(0.5) });
    }
    clicked_in(f, over)
}

/// A click on a button under the pointer: taken from the view, and reported once.
fn clicked_in(f: &mut Frame, over: bool) -> bool {
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

    /// A few matrices of a kind that random draws seldom or never give: the edges of
    /// the kind (0, I, −I, the swap, a matrix on the boundary of an inequality).
    fn specials(kind: Kind) -> Vec<DMat2> {
        let id = DMat2::IDENTITY;
        let candidates = [
            DMat2::ZERO,
            id,
            -id,
            SWAP,
            -SWAP,
            J,
            rows([1.0, 0.0, 0.0, 0.0]),
            rows([0.0, 0.0, 0.0, 1.0]),
            rows([0.0, 1.0, 0.0, 0.0]),
            rows([1.0, 1.0, 0.0, 1.0]),
            rows([2.0, 0.0, 0.0, 0.5]),
            rows([0.5, 0.5, 0.5, 0.5]),
            rows([1.0, -1.0, 0.0, 0.0]),
            rows([0.0, -1.0, 1.0, -1.0]),
            rows([1.0, -1.0, 1.0, 0.0]),
            rows([0.0, 0.0, 1.0, 1.0]),
            DMat2::from_angle(std::f64::consts::TAU / 5.0),
        ];
        candidates.into_iter().filter(|m| is(kind, *m)).collect()
    }

    /// A kind's own two matrices, its special ones, and `n` more at random.
    fn members(kind: Kind, n: usize, seed: u64) -> Vec<DMat2> {
        let mut rng = Rng::new(seed);
        let mut out = vec![kind.default_a(), kind.default_b()];
        out.extend(specials(kind));
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
        for _ in 0..60 * 40 {
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

    fn lit_names(m: DMat2) -> Vec<Kind> {
        KINDS.iter().copied().filter(|k| is(*k, m)).collect()
    }

    /// Every matrix a kind starts with, makes at random, or is dragged to is of that
    /// kind, finite, and within the limits, wherever the column is dragged: on the axes,
    /// at 0, beyond the limits, nowhere at all.
    #[test]
    fn every_kind_keeps_its_matrices_in_it() {
        let mut rng = Rng::new(5);
        for kind in KINDS.into_iter().chain([Kind::Any]) {
            assert!(is(kind, kind.default_a()) && is(kind, kind.default_b()), "{kind:?}");
            for a in members(kind, 200, 11) {
                assert!(is(kind, a) && size(a) <= ENTRY_MAX, "{kind:?}: {a:?}");
                let awkward = [DVec2::ZERO, DVec2::X, DVec2::Y, DVec2::new(0.4, 0.0), DVec2::new(0.0, -0.7), DVec2::new(-2.0, 1e-12), DVec2::new(1.0, 1e-300), DVec2::new(0.5, 0.0), DVec2::splat(9.0), DVec2::splat(-9.0), DVec2::NAN, DVec2::new(f64::INFINITY, 1.0)];
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
            let tri = Kind::Upper.default_a();
            assert!(fit(Kind::Upper, tri, 1, p).y_axis == p && fit(Kind::Upper, tri, 0, p).x_axis == DVec2::new(p.x, 0.0));
            let low = Kind::Lower.default_a();
            assert!(fit(Kind::Lower, low, 0, p).x_axis == p && fit(Kind::Lower, low, 1, p).y_axis == DVec2::new(0.0, p.y));
            let diag = Kind::Diagonal.default_a();
            assert!(fit(Kind::Diagonal, diag, 0, p).x_axis == DVec2::new(p.x, 0.0) && fit(Kind::Diagonal, diag, 1, p).y_axis == DVec2::new(0.0, p.y));
            assert!(fit(Kind::Skew, Kind::Skew.default_a(), 0, p).x_axis == DVec2::new(0.0, p.y));
            assert!(fit(Kind::ScaledRotation, Kind::ScaledRotation.default_a(), 0, p).x_axis == p && fit(Kind::ScaledRotation, Kind::ScaledRotation.default_a(), 1, p).y_axis == p);
            assert!(fit(Kind::Any, Kind::Any.default_a(), 1, p).y_axis == p && fit(Kind::Companion, Kind::Companion.default_a(), 1, p).y_axis == p);
            for (kind, a) in [(Kind::Orthogonal, Kind::Orthogonal.default_a()), (Kind::Orthogonal, Kind::Orthogonal.default_b()), (Kind::Rotation, Kind::Rotation.default_a()), (Kind::Reflection, Kind::Reflection.default_a())] {
                for j in 0..2 {
                    let m = fit(kind, a, j, p);
                    assert!((m.col(j) - p.normalize()).length() < 1e-12 && m.determinant().signum() == a.determinant().signum());
                }
            }
            // the nearest point of the segment of shares
            for kind in [Kind::Markov, Kind::Doubly] {
                let m = fit(kind, kind.default_a(), 0, p);
                let c = m.x_axis;
                assert!(c.x >= 0.0 && c.y >= 0.0 && (c.x + c.y - 1.0).abs() < 1e-12);
                for t in 0..=20 {
                    let q = DVec2::new(t as f64 / 20.0, 1.0 - t as f64 / 20.0);
                    assert!((p - c).length() <= (p - q).length() + 1e-12);
                }
            }
            // the nearest point of the circle an orthogonal projection's column lies on
            let c = fit(Kind::OrthProjection, Kind::OrthProjection.default_a(), 0, p).x_axis;
            let centre = DVec2::new(0.5, 0.0);
            assert!(((c - centre).length() - 0.5).abs() < 1e-12 && ((p - centre).normalize() - (c - centre).normalize()).length() < 1e-9);
            // where the kinds with room to move are free to go
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
            if p.length() >= LEAST_COLUMN {
                for kind in [Kind::DetOne, Kind::Singular, Kind::RankOne] {
                    assert!(fit(kind, kind.default_a(), 0, p).x_axis == p, "{kind:?}");
                }
            }
            if p.min_element() >= MARGIN {
                assert!(fit(Kind::Positive, Kind::Positive.default_a(), 0, p).x_axis == p);
            }
            if p.x.abs() >= MARGIN && p.y.abs() <= ENTRY_MAX - MARGIN {
                assert!(fit(Kind::Dominant, Kind::Dominant.default_a(), 0, p).x_axis == p);
            }
            if p.y.abs() >= (2.0 * MARGIN).sqrt() {
                assert!(fit(Kind::Indefinite, Kind::Indefinite.default_a(), 0, p).x_axis == p);
            }
        }
        // a drag round a loop clear of the axes, in small steps: no jumps (a
        // permutation has nowhere between its two matrices, and leaps)
        for kind in KINDS.into_iter().filter(|k| *k != Kind::Permutation) {
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
        let mut points = Rng::new(22);
        let mut point = || DVec2::new(points.range(-3.0, 3.0) as f64, points.range(-3.0, 3.0) as f64);
        for kind in KINDS {
            for a in members(kind, 300, 31) {
                let tol = 1e-9 * size(a).max(1.0);
                let e = eigen(a);
                let [l1, l2] = eigenvalues(e);
                for l in [l1, l2] {
                    assert!(kind.region().holds(l, tol, a), "{kind:?}: {a:?} has the eigenvalue {l:?}, not in {}", kind.region().name());
                }
                let [a11, a12, a21, a22] = entries(a);
                let diagonal_ones = || (l1.0 - a11).abs() + (l2.0 - a22).abs() < 1e-12 || (l1.0 - a22).abs() + (l2.0 - a11).abs() < 1e-12;
                match kind {
                    Kind::Scalar => assert!(matches!(e, Eigen::Real { lines: Lines::Every, .. }) && (0..10).all(|_| commute(a, rows([0; 4].map(|_| rng.range(-2.0, 2.0) as f64))))),
                    Kind::Diagonal | Kind::Upper | Kind::Lower => assert!(l1.1 == 0.0 && diagonal_ones(), "{a:?}"),
                    Kind::JordanBlock => assert!(matches!(e, Eigen::Real { l1, lines: Lines::One(v), .. } if l1 == a11 && v.y.abs() < 1e-12), "{a:?} {e:?}"),
                    Kind::Symmetric | Kind::PosDef | Kind::PosSemidef | Kind::Indefinite | Kind::OrthProjection => match e {
                        Eigen::Real { lines: Lines::Two(v1, v2), l1, l2 } => {
                            assert!(v1.dot(v2).abs() < 1e-9, "{a:?}");
                            if kind == Kind::Indefinite {
                                assert!(l1 > 0.0 && l2 < 0.0);
                            }
                        }
                        Eigen::Real { lines: Lines::Every, .. } => assert!(kind != Kind::Indefinite),
                        _ => panic!("{kind:?}: {a:?} {e:?}"),
                    },
                    Kind::Skew => {
                        for _ in 0..10 {
                            let x = point();
                            assert!(x.dot(a * x).abs() < 1e-12, "{a:?}");
                        }
                        assert!(close(a, J * a21, 0.0), "in the plane a skew matrix is wJ: {a:?}");
                    }
                    Kind::ScaledRotation => assert!(e == Eigen::Complex { re: a11, im: a21.abs() } || (a21 == 0.0 && l1 == (a11, 0.0)), "{a:?} {e:?}"),
                    Kind::Orthogonal | Kind::Rotation | Kind::Reflection => {
                        if a.determinant() < 0.0 {
                            assert!(matches!(e, Eigen::Real { l1, l2, lines: Lines::Two(v1, v2) } if (l1 - 1.0).abs() < 1e-12 && (l2 + 1.0).abs() < 1e-12 && v1.dot(v2).abs() < 1e-9), "{a:?} {e:?}");
                        }
                        for _ in 0..10 {
                            let x = point();
                            assert!(((a * x).length() - x.length()).abs() < 1e-12);
                        }
                    }
                    Kind::DetOne => assert!((l1.0 * l2.0 - l1.1 * l2.1 - 1.0).abs() < 1e-9),
                    Kind::Normal => match e {
                        Eigen::Real { lines: Lines::Two(v1, v2), .. } => assert!(v1.dot(v2).abs() < 1e-9, "{a:?}"),
                        Eigen::Real { lines: Lines::Every, .. } => {}
                        Eigen::Real { lines: Lines::One(_), .. } => panic!("a normal matrix with one eigenline: {a:?}"),
                        // the complex eigenvectors (a₁₂, λ − a₁₁) and their conjugates are
                        // at right angles: v₁² + v₂² = 0 (as complex numbers)
                        Eigen::Complex { re, im } => {
                            let (v1, v2) = ((a12, 0.0), (re - a11, im));
                            let square = (v1.0 * v1.0 + v2.0 * v2.0 - v2.1 * v2.1, 2.0 * v2.0 * v2.1);
                            assert!(square.0.abs() + square.1.abs() < 1e-9 * (1.0 + v1.0 * v1.0 + im * im), "{a:?}: {square:?}");
                        }
                    },
                    Kind::Projection => assert!(close(a * a, a, 1e-12)),
                    Kind::Involution => assert!(close(a * a, DMat2::IDENTITY, 1e-12)),
                    Kind::Root => {
                        let (n, _, holds) = order(a);
                        assert!(holds && close(power(a, n), DMat2::IDENTITY, 1e-9), "{a:?}");
                        for l in [l1, l2] {
                            // λⁿ = 1
                            let (r, t) = (l.0.hypot(l.1), l.1.atan2(l.0) * n as f64);
                            assert!((r - 1.0).abs() < 1e-9 && (t.cos() - 1.0).abs() < 1e-6, "{a:?} {l:?} {n}");
                        }
                    }
                    Kind::Nilpotent | Kind::Shear if size(a) > 0.0 => {
                        let Eigen::Real { l1, lines: Lines::One(v), .. } = e else { panic!("{kind:?} {a:?} {e:?}") };
                        let keep = if kind == Kind::Shear { 1.0 } else { 0.0 };
                        assert!((l1 - keep).abs() < 1e-12 && (a * v - v * keep).length() < 1e-9, "{a:?}");
                        // everything lands on that line (nilpotent), or slides along it (shear)
                        for _ in 0..10 {
                            let x = point();
                            assert!((a * x - x * keep).perp_dot(v).abs() < 1e-9, "{a:?}");
                        }
                        assert!(kind == Kind::Nilpotent || (a.determinant() - 1.0).abs() < 1e-9);
                    }
                    Kind::Positive => {
                        // real, the larger positive and larger in size, its eigenvector positive
                        let Eigen::Real { l1, l2, lines: Lines::Two(v, _) } = e else { panic!("{a:?} {e:?}") };
                        assert!(l1 > 0.0 && l1 > l2.abs() && v.x * v.y > 0.0, "{a:?} {e:?}");
                    }
                    Kind::Markov | Kind::Doubly => {
                        let Eigen::Real { l1, l2, .. } = e else { panic!("{a:?}") };
                        assert!((l1 - 1.0).abs() < 1e-12 && (l2 - (trace(a) - 1.0)).abs() < 1e-12 && (-1.0 - 1e-12..=1.0 + 1e-12).contains(&l2), "{a:?}");
                        if kind == Kind::Doubly {
                            assert!((l2 - (2.0 * a11 - 1.0)).abs() < 1e-12);
                            assert!(matches!(e, Eigen::Real { lines: Lines::Two(v1, v2), .. } if v1.dot(v2).abs() < 1e-9) || matches!(e, Eigen::Real { lines: Lines::Every, .. }), "{a:?} {e:?}");
                        }
                    }
                    Kind::Dominant => {
                        assert!(discs(a).iter().all(|(c, r)| c.abs() > *r) && a.determinant() != 0.0, "{a:?}");
                    }
                    Kind::Singular | Kind::RankOne => {
                        assert!(l1.1 == 0.0 && (l1.0.abs().min(l2.0.abs()) < 1e-9) && ((l1.0 + l2.0) - trace(a)).abs() < 1e-9, "{a:?} {e:?}");
                        if kind == Kind::RankOne {
                            let (u, v) = rank_one_factors(a);
                            assert!(close(DMat2::from_cols(u * v.x, u * v.y), a, 1e-12) && (v.dot(u) - trace(a)).abs() < 1e-12);
                        }
                    }
                    Kind::Permutation => assert!(close(a, DMat2::IDENTITY, 0.0) || close(a, SWAP, 0.0)),
                    Kind::Companion => {
                        // the roots of λ² − a₂₂λ − a₁₂ (t = a₂₂, d = −a₁₂)
                        for l in [l1, l2] {
                            let (re, im) = l;
                            let (sq_re, sq_im) = (re * re - im * im, 2.0 * re * im);
                            assert!((sq_re - a22 * re - a12).abs() < 1e-9 && (sq_im - a22 * im).abs() < 1e-9, "{a:?} {l:?}");
                        }
                    }
                    _ => {}
                }
            }
        }
        // a rotation's eigenvalues are a complex pair on the unit circle
        for angle in [0.3, 1.0, 2.0, -2.5] {
            let Eigen::Complex { re, im } = eigen(DMat2::from_angle(angle)) else { panic!() };
            assert!((re - angle.cos()).abs() < 1e-12 && (im - angle.sin().abs()).abs() < 1e-12);
        }
        // every eigenvalue of any matrix lies in a Gershgorin disc
        for _ in 0..2000 {
            let m = rows([0; 4].map(|_| rng.range(-3.0, 3.0) as f64));
            for l in eigenvalues(eigen(m)) {
                assert!(Region::Discs.holds(l, 1e-9, m), "{m:?}");
            }
        }
    }

    /// Whether every matrix of `small` is one of `large`, as far as the map is
    /// concerned: along its lines.
    fn reaches(small: Kind, large: Kind) -> bool {
        small == large || EDGES.iter().any(|(s, l)| *s == small && reaches(*l, large))
    }

    /// Every line of the map holds: every matrix of the smaller kind tried (its own, its
    /// special ones, hundreds at random) passes the larger kind's test. Every other
    /// inclusion follows along the lines, and where the map has no path from one kind
    /// up to another, a matrix of the first is found that is not of the second.
    #[test]
    fn the_map_shows_which_kinds_contain_which() {
        for (small, large) in EDGES {
            for m in members(small, 2000, 3) {
                assert!(is(large, m), "{small:?} in {large:?}: not {m:?}");
            }
            // (and the line is not one that two others make already)
            assert!(!KINDS.iter().any(|k| *k != small && *k != large && reaches(small, *k) && reaches(*k, large)), "{small:?} to {large:?} goes through another kind");
        }
        for small in KINDS {
            let all = members(small, 400, 17);
            for large in KINDS {
                if reaches(small, large) {
                    assert!(all.iter().all(|m| is(large, *m)), "{small:?} {large:?}");
                } else {
                    assert!(all.iter().any(|m| !is(large, *m)), "the map has no path from {small:?} up to {large:?}, yet every one tried is");
                }
            }
        }
        // the lines a reader might expect, and why they are not there
        let not = |small: Kind, large: Kind, m: DMat2| assert!(is(small, m) && !is(large, m), "{small:?} {large:?} {m:?}");
        not(Kind::Orthogonal, Kind::Symmetric, J);
        not(Kind::Normal, Kind::Symmetric, J);
        not(Kind::Projection, Kind::Symmetric, Kind::Projection.default_a());
        not(Kind::Involution, Kind::Orthogonal, rows([1.0, 1.0, 0.0, -1.0]));
        not(Kind::Root, Kind::Orthogonal, rows([1.0, -1.0, 1.0, 0.0]));
        not(Kind::DetOne, Kind::Orthogonal, Kind::Shear.default_a());
        not(Kind::Shear, Kind::Normal, Kind::Shear.default_a());
        not(Kind::Positive, Kind::PosDef, rows([1.0, 2.0, 2.0, 1.0]));
        not(Kind::PosDef, Kind::Positive, rows([1.0, -0.5, -0.5, 1.0]));
        not(Kind::Markov, Kind::Symmetric, Kind::Markov.default_a());
        not(Kind::Markov, Kind::Positive, DMat2::IDENTITY);
        not(Kind::RankOne, Kind::Projection, Kind::RankOne.default_a());
        not(Kind::Nilpotent, Kind::Upper, Kind::Nilpotent.default_a());
        not(Kind::JordanBlock, Kind::Shear, Kind::JordanBlock.default_a());
        not(Kind::Dominant, Kind::Symmetric, Kind::Dominant.default_a());
        not(Kind::Permutation, Kind::Rotation, SWAP);
        not(Kind::Permutation, Kind::Reflection, DMat2::IDENTITY);
        not(Kind::Permutation, Kind::Companion, DMat2::IDENTITY);
        not(Kind::Singular, Kind::RankOne, DMat2::ZERO);
        not(Kind::Rotation, Kind::Root, DMat2::from_angle(1.0));
        not(Kind::Reflection, Kind::DetOne, SWAP);
    }

    /// The map is drawn so that no two lines cross, no line passes through a box, and
    /// no two boxes overlap, at the sizes it is drawn at.
    #[test]
    fn the_map_can_be_read() {
        let mut problems = Vec::new();
        for (tall, area, size) in [(false, (Vec2::new(0.0, 0.0), Vec2::new(720.0, 300.0)), 10.5), (false, (Vec2::new(0.0, 0.0), Vec2::new(1000.0, 700.0)), 10.5), (true, (Vec2::new(0.0, 0.0), Vec2::new(370.0, 540.0)), 9.5), (true, (Vec2::new(0.0, 0.0), Vec2::new(420.0, 800.0)), 10.5)] {
            let boxes: Vec<(Vec2, Vec2)> = KINDS.iter().map(|k| map_box(*k, area, tall, size)).collect();
            for i in 0..30 {
                if !(boxes[i].0.cmpge(area.0).all() && boxes[i].1.cmple(area.1).all()) {
                    problems.push(format!("{tall} {area:?}: {:?} outside", KINDS[i]));
                }
                for j in i + 1..30 {
                    let (a, b) = (boxes[i], boxes[j]);
                    let apart = a.1.x + 4.0 <= b.0.x || b.1.x + 4.0 <= a.0.x || a.1.y + 4.0 <= b.0.y || b.1.y + 4.0 <= a.0.y;
                    if !apart {
                        problems.push(format!("{tall} {area:?}: {:?} and {:?} overlap", KINDS[i], KINDS[j]));
                    }
                }
            }
            let seg = |(s, l): (Kind, Kind)| {
                let (a, b) = (boxes[index(s)], boxes[index(l)]);
                let (p, q) = ((a.0 + a.1) * 0.5, (b.0 + b.1) * 0.5);
                (leave_box(p, q, a), leave_box(q, p, b))
            };
            let o = |p: Vec2, q: Vec2, r: Vec2| (q - p).perp_dot(r - p);
            for (i, e) in EDGES.iter().enumerate() {
                let (p, q) = seg(*e);
                // the smaller kind lower down
                if p.y <= q.y {
                    problems.push(format!("{tall} {area:?}: {e:?} runs downwards"));
                }
                for f in &EDGES[i + 1..] {
                    if e.0 == f.0 || e.0 == f.1 || e.1 == f.0 || e.1 == f.1 {
                        continue;
                    }
                    let (r, s) = seg(*f);
                    if o(p, q, r) * o(p, q, s) < 0.0 && o(r, s, p) * o(r, s, q) < 0.0 {
                        problems.push(format!("{tall} {area:?}: {e:?} crosses {f:?}"));
                    }
                }
                for (k, b) in boxes.iter().enumerate() {
                    if KINDS[k] == e.0 || KINDS[k] == e.1 {
                        continue;
                    }
                    let (lo, hi) = (b.0 - Vec2::splat(3.0), b.1 + Vec2::splat(3.0));
                    let inside = (0..=200).any(|t| {
                        let x = p.lerp(q, t as f32 / 200.0);
                        x.cmpge(lo).all() && x.cmple(hi).all()
                    });
                    if inside {
                        problems.push(format!("{tall} {area:?}: {e:?} passes through {:?}", KINDS[k]));
                    }
                }
            }
        }
        assert!(problems.is_empty(), "{}", problems.join("\n"));
    }

    /// The count of free numbers is the dimension of the kind: 4 less the number of
    /// independent equations it puts on the entries, worked out from the slopes of
    /// those equations at matrices of the kind.
    #[test]
    fn the_free_numbers_are_the_dimension() {
        // the equations each kind puts on [a₁₁, a₁₂, a₂₁, a₂₂] near the matrix `at`
        fn equations(kind: Kind, at: DMat2, m: DMat2) -> Vec<f64> {
            let [a, b, c, d] = entries(m);
            let id = DMat2::IDENTITY;
            let flat = |x: DMat2| entries(x).to_vec();
            match kind {
                Kind::Scalar => vec![b, c, a - d],
                Kind::Diagonal => vec![b, c],
                Kind::Upper => vec![c],
                Kind::JordanBlock => vec![c, b - 1.0, a - d],
                Kind::Lower => vec![b],
                Kind::Symmetric | Kind::PosDef | Kind::PosSemidef | Kind::Indefinite => vec![b - c],
                Kind::Skew => vec![a, d, b + c],
                Kind::ScaledRotation => vec![a - d, b + c],
                Kind::Normal if is(Kind::Symmetric, at) => vec![b - c],
                Kind::Normal => vec![a - d, b + c],
                Kind::Orthogonal | Kind::Rotation | Kind::Reflection => flat(m.transpose() * m - id),
                Kind::DetOne => vec![m.determinant() - 1.0],
                Kind::Projection => flat(m * m - m),
                Kind::OrthProjection => [flat(m * m - m), vec![b - c]].concat(),
                Kind::Involution => flat(m * m - id),
                Kind::Nilpotent => flat(m * m),
                Kind::Shear => flat((m - id) * (m - id)),
                Kind::Root => flat(power(m, order(at).0) - id),
                Kind::Markov => vec![a + c - 1.0, b + d - 1.0],
                Kind::Doubly => vec![a + c - 1.0, b + d - 1.0, a + b - 1.0],
                Kind::Singular | Kind::RankOne => vec![m.determinant()],
                Kind::Permutation => flat(m - at),
                Kind::Companion => vec![a, c - 1.0],
                Kind::Positive | Kind::Dominant | Kind::Any => Vec::new(),
            }
        }
        // the rank of the equations' slopes, by elimination
        fn rank(mut rows: Vec<[f64; 4]>) -> usize {
            let scale = rows.iter().flatten().fold(0.0f64, |s, v| s.max(v.abs())).max(1e-300);
            let mut r = 0;
            for col in 0..4 {
                let Some(p) = (r..rows.len()).max_by(|i, j| rows[*i][col].abs().total_cmp(&rows[*j][col].abs())) else { break };
                if rows[p][col].abs() <= 1e-6 * scale {
                    continue;
                }
                rows.swap(r, p);
                for i in 0..rows.len() {
                    if i != r {
                        let f = rows[i][col] / rows[r][col];
                        let pivot = rows[r];
                        for k in 0..4 {
                            rows[i][k] -= f * pivot[k];
                        }
                    }
                }
                r += 1;
            }
            r
        }
        for kind in KINDS {
            let leading: usize = kind.free()[..1].parse().unwrap();
            let mut rng = Rng::new(77);
            let mut tried = 0;
            for _ in 0..40 {
                let at = random(kind, &mut rng);
                // (the few matrices set apart in the count: 0, I and −I)
                if size(at) < 1e-9 || size(at - DMat2::IDENTITY) < 1e-9 || size(at + DMat2::IDENTITY) < 1e-9 {
                    continue;
                }
                let h = 1e-6;
                let base = equations(kind, at, at);
                let mut slopes: Vec<[f64; 4]> = vec![[0.0; 4]; base.len()];
                for (k, delta) in [rows([h, 0.0, 0.0, 0.0]), rows([0.0, h, 0.0, 0.0]), rows([0.0, 0.0, h, 0.0]), rows([0.0, 0.0, 0.0, h])].into_iter().enumerate() {
                    let (up, down) = (equations(kind, at, at + delta), equations(kind, at, at - delta));
                    for i in 0..base.len() {
                        slopes[i][k] = (up[i] - down[i]) / (2.0 * h);
                    }
                }
                // (the normal ones are two families: the symmetric, 3, and the scaled rotations, 2)
                let stated = if kind == Kind::Normal && !is(Kind::Symmetric, at) { 2 } else { leading };
                assert_eq!(4 - rank(slopes), stated, "{kind:?} at {at:?}");
                tried += 1;
            }
            assert!(tried > 10, "{kind:?}");
        }
        // the scaled rotations among the normal matrices: 2
        assert!(Kind::Normal.free().contains("scaled rotations among them: 2"));
    }

    /// What the panel and the table say stays true of each kind under a sum, a product,
    /// an inverse and a transpose: "always", "never", "where it exists" tried on
    /// hundreds of pairs, and each condition and "not always" held by examples both
    /// ways.
    #[test]
    fn what_stays_true_is_what_the_panel_says() {
        let id = DMat2::IDENTITY;
        let mirror = |phi: f64| rows([(2.0 * phi).cos(), (2.0 * phi).sin(), (2.0 * phi).sin(), -(2.0 * phi).cos()]);
        let turn = DMat2::from_angle;
        let third = std::f64::consts::TAU / 3.0;
        for kind in KINDS {
            let mut rng = Rng::new(41);
            let all = members(kind, 150, 43);
            let mut pairs: Vec<(DMat2, DMat2)> = Vec::new();
            for a in &all {
                pairs.push((*a, *a));
                pairs.push((*a, random(kind, &mut rng)));
            }
            for op in [Op::Sum, Op::Product, Op::Inverse, Op::Transpose] {
                let unary = matches!(op, Op::Inverse | Op::Transpose);
                let results: Vec<Option<DMat2>> = if unary { all.iter().map(|a| combine(op, *a, *a)).collect() } else { pairs.iter().map(|(a, b)| combine(op, *a, *b)).collect() };
                let of_kind = results.iter().filter(|r| r.is_some_and(|m| is(kind, m))).count();
                match kind.keeps(op) {
                    Keeps::Always => assert!(results.iter().all(|r| r.is_some_and(|m| is(kind, m))), "{kind:?} {op:?}"),
                    Keeps::WhereItExists => assert!(results.iter().all(|r| r.is_none_or(|m| is(kind, m))) && results.iter().any(|r| r.is_some()), "{kind:?} {op:?}"),
                    Keeps::Never => assert_eq!(of_kind, 0, "{kind:?} {op:?}"),
                    Keeps::NoInverse => assert!(results.iter().all(|r| r.is_none()), "{kind:?} {op:?}"),
                    Keeps::Sometimes(_) => {}
                }
                assert!(kind.closure(op).ends_with('.'), "{kind:?} {op:?}");
            }
        }
        // the conditions, tried on many pairs, and examples each way of "not always"
        let both = |kind: Kind, op: Op, yes: (DMat2, DMat2), no: (DMat2, DMat2)| {
            assert!(matches!(kind.keeps(op), Keeps::Sometimes(_)), "{kind:?} {op:?}");
            let unary = matches!(op, Op::Inverse | Op::Transpose);
            for (pair, want) in [(yes, true), (no, false)] {
                assert!(is(kind, pair.0) && (unary || is(kind, pair.1)), "{kind:?} {op:?} {pair:?}");
                assert_eq!(combine(op, pair.0, pair.1).is_some_and(|m| is(kind, m)), want, "{kind:?} {op:?} {pair:?}");
            }
        };
        let s = |k: Kind| k.default_a();
        let t = |k: Kind| k.default_b();
        both(Kind::Upper, Op::Transpose, (rows([1.5, 0.0, 0.0, 0.6]), id), (s(Kind::Upper), id));
        both(Kind::Lower, Op::Transpose, (rows([1.5, 0.0, 0.0, 0.6]), id), (s(Kind::Lower), id));
        both(Kind::Normal, Op::Sum, (rows([2.0, 0.0, 0.0, 1.0]), rows([2.0, 0.0, 0.0, 1.0])), (rows([2.0, 0.0, 0.0, 1.0]), J));
        both(Kind::Normal, Op::Product, (J, J), (rows([2.0, 0.0, 0.0, 1.0]), J));
        for kind in [Kind::Orthogonal, Kind::Rotation, Kind::DetOne, Kind::Root] {
            both(kind, Op::Sum, (id, turn(third)), (id, id));
        }
        both(Kind::Reflection, Op::Sum, (mirror(0.3), mirror(0.3 + third / 2.0)), (mirror(0.3), mirror(0.8)));
        both(Kind::Involution, Op::Sum, (mirror(0.3), mirror(0.3 + third / 2.0)), (id, id));
        both(Kind::Involution, Op::Product, (s(Kind::Involution), s(Kind::Involution)), (s(Kind::Involution), SWAP));
        both(Kind::Root, Op::Product, (turn(0.4 * std::f64::consts::PI), turn(0.4 * std::f64::consts::PI)), (mirror(0.0), mirror(0.5)));
        both(Kind::Indefinite, Op::Sum, (s(Kind::Indefinite), s(Kind::Indefinite)), (s(Kind::Indefinite), -s(Kind::Indefinite)));
        let shear9 = rows([1.0, 0.9, 0.0, 1.0]);
        let lopsided = rows([1.0, 0.9, 0.0, 0.5]);
        both(Kind::Dominant, Op::Sum, (id, id), (id, -id));
        both(Kind::Dominant, Op::Product, (id, shear9), (shear9, shear9));
        both(Kind::Dominant, Op::Inverse, (id, id), (lopsided, lopsided));
        both(Kind::Dominant, Op::Transpose, (id, id), (lopsided, lopsided));
        assert!(members(Kind::Dominant, 300, 5).iter().all(|m| inverse(*m).is_some()));
        both(Kind::Singular, Op::Sum, (s(Kind::Singular), s(Kind::Singular)), (rows([1.0, 0.0, 0.0, 0.0]), rows([0.0, 0.0, 0.0, 1.0])));
        both(Kind::RankOne, Op::Sum, (s(Kind::RankOne), s(Kind::RankOne)), (rows([1.0, 0.0, 0.0, 0.0]), rows([0.0, 0.0, 0.0, 1.0])));
        both(Kind::RankOne, Op::Product, (s(Kind::RankOne), s(Kind::RankOne)), (rows([1.0, 0.0, 0.0, 0.0]), rows([0.0, 0.0, 0.0, 1.0])));
        both(Kind::Nilpotent, Op::Sum, (s(Kind::Nilpotent), s(Kind::Nilpotent) * 0.5), (s(Kind::Nilpotent), t(Kind::Nilpotent)));
        both(Kind::Nilpotent, Op::Product, (s(Kind::Nilpotent), s(Kind::Nilpotent) * 0.5), (s(Kind::Nilpotent), t(Kind::Nilpotent)));
        both(Kind::Shear, Op::Product, (s(Kind::Shear), fit(Kind::Shear, s(Kind::Shear), 1, DVec2::new(-0.4, 1.0))), (s(Kind::Shear), t(Kind::Shear)));
        both(Kind::Companion, Op::Product, (rows([0.0, 0.0, 1.0, 1.0]), s(Kind::Companion)), (s(Kind::Companion), s(Kind::Companion)));
        both(Kind::Companion, Op::Inverse, (SWAP, SWAP), (s(Kind::Companion), s(Kind::Companion)));
        both(Kind::Companion, Op::Transpose, (rows([0.0, 1.0, 1.0, 0.7]), id), (s(Kind::Companion), id));
        both(Kind::JordanBlock, Op::Product, (rows([0.3, 1.0, 0.0, 0.3]), rows([0.7, 1.0, 0.0, 0.7])), (s(Kind::JordanBlock), t(Kind::JordanBlock)));
        both(Kind::Markov, Op::Inverse, (SWAP, SWAP), (s(Kind::Markov), s(Kind::Markov)));
        both(Kind::Markov, Op::Transpose, (s(Kind::Doubly), id), (s(Kind::Markov), id));
        both(Kind::Doubly, Op::Inverse, (id, id), (s(Kind::Doubly), s(Kind::Doubly)));
        both(Kind::Projection, Op::Inverse, (id, id), (s(Kind::Projection), s(Kind::Projection)));
        both(Kind::OrthProjection, Op::Inverse, (id, id), (s(Kind::OrthProjection), s(Kind::OrthProjection)));
        // "exactly when" and "only if", over many pairs
        let mut rng = Rng::new(8);
        for _ in 0..400 {
            for kind in [Kind::Symmetric, Kind::PosDef, Kind::PosSemidef, Kind::OrthProjection, Kind::Involution] {
                let (a, b) = (random(kind, &mut rng), random(kind, &mut rng));
                assert_eq!(is(kind, a * b), commute(a, b), "{kind:?}: {a:?} {b:?}");
                if kind == Kind::OrthProjection {
                    assert_eq!(is(kind, a + b), size(a * b) <= 1e-9, "{a:?} {b:?}");
                }
            }
            let (a, b) = (random(Kind::Projection, &mut rng), random(Kind::Projection, &mut rng));
            let zero = size(a * b) <= 1e-9 && size(b * a) <= 1e-9;
            assert_eq!(is(Kind::Projection, a + b), zero, "{a:?} {b:?}");
            if commute(a, b) {
                assert!(is(Kind::Projection, a * b));
            }
            let (a, b) = (random(Kind::Skew, &mut rng), random(Kind::Skew, &mut rng));
            assert!(close(a * b, DMat2::IDENTITY * -(a.x_axis.y * b.x_axis.y), 1e-12) && !is(Kind::Skew, a * b));
            let (a, b) = (random(Kind::Reflection, &mut rng), random(Kind::Reflection, &mut rng));
            let apart = (0.5 * (a.x_axis.y.atan2(a.x_axis.x) - b.x_axis.y.atan2(b.x_axis.x))).to_degrees().rem_euclid(180.0);
            assert_eq!(is(Kind::Reflection, a + b), (apart - 60.0).abs() < 1e-6 || (apart - 120.0).abs() < 1e-6, "{apart}");
            let (a, b) = (random(Kind::Shear, &mut rng), random(Kind::Shear, &mut rng));
            let same = matches!((eigen(a), eigen(b)), (Eigen::Real { lines: Lines::One(u), .. }, Eigen::Real { lines: Lines::One(v), .. }) if u.perp_dot(v).abs() < 1e-9);
            assert_eq!(is(Kind::Shear, a * b) || close(a * b, id, 1e-9), same, "{a:?} {b:?}");
            let (a, b) = (random(Kind::RankOne, &mut rng), random(Kind::RankOne, &mut rng));
            assert!(is(Kind::RankOne, a * b) || size(a * b) <= 1e-9);
            let (a, b) = (random(Kind::Companion, &mut rng), random(Kind::Companion, &mut rng));
            assert_eq!(is(Kind::Companion, a * b), a.y_axis == DVec2::Y, "{a:?}");
            assert_eq!(inverse(a).is_some_and(|m| is(Kind::Companion, m)), close(a, SWAP, 1e-12));
            assert_eq!(is(Kind::Companion, a.transpose()), (a.y_axis.x - 1.0).abs() <= 1e-9);
            let (a, b) = (random(Kind::JordanBlock, &mut rng), random(Kind::JordanBlock, &mut rng));
            assert_eq!(is(Kind::JordanBlock, a * b), (a.x_axis.x + b.x_axis.x - 1.0).abs() <= 1e-9);
            let a = random(Kind::Markov, &mut rng);
            assert_eq!(is(Kind::Markov, a.transpose()), is(Kind::Doubly, a));
            assert!(is(Kind::Markov, (a + random(Kind::Markov, &mut rng)) * 0.5));
            for kind in [Kind::Markov, Kind::Doubly] {
                let a = random(kind, &mut rng);
                if let Some(inv) = inverse(a) {
                    let [p, q, r, s] = entries(inv);
                    assert!((p + r - 1.0).abs() < 1e-9 && (q + s - 1.0).abs() < 1e-9 && p.min(q).min(r).min(s) < 0.0, "{a:?}");
                }
            }
            let a = random(Kind::Positive, &mut rng);
            assert!(inverse(a).is_none_or(|m| m.to_cols_array().iter().any(|v| *v < 0.0)));
            let (a, b) = (random(Kind::Upper, &mut rng), random(Kind::Indefinite, &mut rng));
            assert_eq!(is(Kind::Upper, a.transpose()), is(Kind::Diagonal, a));
            assert!((b * random(Kind::Indefinite, &mut rng)).determinant() > 0.0);
        }
        // the ones every member shows
        for kind in [Kind::Markov, Kind::Doubly] {
            assert!(is(kind, id) && is(kind, SWAP) && is(kind, inverse(SWAP).unwrap()));
        }
        assert!(is(Kind::Projection, id) && inverse(id) == Some(id));
        // every matrix but a multiple of I is a companion matrix in the basis v, Av, for
        // any v that is not an eigenvector
        for _ in 0..300 {
            let m = rows([0; 4].map(|_| rng.range(-2.0, 2.0) as f64));
            let v = DVec2::new(rng.range(-1.0, 1.0) as f64, rng.range(-1.0, 1.0) as f64);
            let basis = DMat2::from_cols(v, m * v);
            if basis.determinant().abs() > 0.05 {
                assert!(is(Kind::Companion, basis.inverse() * m * basis) || size(basis.inverse() * m * basis - rows([0.0, -m.determinant(), 1.0, trace(m)])) < 1e-6, "{m:?}");
            }
        }
        // the product of two complex numbers, as scaled rotations
        let (a, b) = (s(Kind::ScaledRotation), t(Kind::ScaledRotation));
        assert!(close(a * b, b * a, 1e-15));
    }

    /// The captions of the "things to try", held to account.
    #[test]
    fn the_complex_numbers_do_what_their_caption_says() {
        let scene = run(0);
        let s = scene.shown();
        let (a, b) = (s.a, s.b);
        assert!(scene.kind == Kind::ScaledRotation && scene.op == Op::Product && is(Kind::ScaledRotation, a) && is(Kind::ScaledRotation, b));
        // A = 1 + 0.5i, B = 1.2 + 0.9i
        assert!(close(a, DMat2::IDENTITY + J * 0.5, 0.0) && close(b, DMat2::IDENTITY * 1.2 + J * 0.9, 0.0));
        let size_of = |m: DMat2| m.x_axis.length();
        let angle_of = |m: DMat2| degrees(m.x_axis.y.atan2(m.x_axis.x));
        assert!(num(size_of(a), 2) == "1.12" && num(size_of(b), 2) == "1.50" && angle_of(a) == "26.6°" && angle_of(b) == "36.9°");
        let ab = a * b;
        assert!(close(ab, rows([0.75, -1.5, 1.5, 0.75]), 1e-15) && close(ab, b * a, 1e-15));
        assert!(angle_of(ab) == "63.4°" && num(size_of(ab), 2) == "1.68" && num(size_of(a) * size_of(b), 2) == "1.68");
        // the letter turned by 63.4° and stretched by 1.68: AB = |AB| R(θ)
        assert!(close(ab, DMat2::from_angle(ab.x_axis.y.atan2(ab.x_axis.x)) * size_of(ab), 1e-12));
        assert_eq!(eigen(ab), Eigen::Complex { re: 0.75, im: 1.5 });
        assert_eq!(eigen(a), Eigen::Complex { re: 1.0, im: 0.5 });
        // (and inside the chart: up to 1.7 either way)
        assert!(heading_ok(&scene));
    }

    fn heading_ok(scene: &Special) -> bool {
        let (first, second) = scene.heading_text(false);
        !first.is_empty() && !first.contains("NaN") && !second.contains("NaN")
    }

    #[test]
    fn the_slanted_involution_does_what_its_caption_says() {
        let scene = run(1);
        let s = scene.shown();
        let a = s.a;
        assert!(scene.kind == Kind::Involution && s.power == 2.0 && close(a, rows([1.0, 1.0, 0.0, -1.0]), 0.0));
        assert!(close(a * a, DMat2::IDENTITY, 0.0) && close(powered(a, 2.0), DMat2::IDENTITY, 0.0));
        // after one step the letter is flipped (area reversed), slanted and stretched
        let once = F_OUTLINE.map(|p| a * p);
        assert!(area(&once) < 0.0 && (area(&once) + area(&F_OUTLINE)).abs() < 1e-12);
        assert!(close(a * DMat2::IDENTITY, a, 0.0) && (a * DVec2::Y - DVec2::new(1.0, -1.0)).length() == 0.0 && (a * DVec2::Y).length() > 1.0);
        assert!(!is(Kind::Orthogonal, a) && !is(Kind::Reflection, a) && is(Kind::Involution, a) && is(Kind::Root, a));
        let Eigen::Real { l1, l2, lines: Lines::Two(v1, v2) } = eigen(a) else { panic!() };
        assert!(l1 == 1.0 && l2 == -1.0 && v1.perp_dot(DVec2::X).abs() < 1e-12 && v2.perp_dot(DVec2::new(1.0, -2.0)).abs() < 1e-12 && v1.dot(v2).abs() > 0.1);
        // the steps went from I to A to A² = I
        assert!(steps_at(0.0, 2) == 0.0 && steps_at(steps_length(2), 2) == 2.0);
        assert_eq!(scene.steps_top(a), 2);
    }

    #[test]
    fn the_gershgorin_discs_do_what_their_caption_says() {
        let mut scene = Special::new();
        scene.set_up(2);
        assert!(scene.kind == Kind::Any && scene.discs && scene.family == Family::Signs);
        let mut seen = Vec::new();
        for _ in 0..60 * 20 {
            scene.advance(1.0 / 60.0, 1.0 / 60.0);
            // (once the glide to the experiment's matrix has arrived and the tour is under way)
            if !scene.glide.arrived() {
                continue;
            }
            let a = scene.shown().a;
            let [a11, a12, a21, a22] = entries(a);
            // a₁₂ alone moves, from 0 to 2
            assert!(a11 == 1.0 && a21 == 0.5 && a22 == 1.0 && (-1e-12..=2.0 + 1e-12).contains(&a12), "{a:?}");
            for l in eigenvalues(eigen(a)) {
                assert!(Region::Discs.holds(l, 1e-9, a), "{a:?}");
            }
            let dominant = is(Kind::Dominant, a);
            if a12 < 0.999 {
                assert!(dominant && discs(a).iter().all(|(c, r)| c.abs() > *r) && a.determinant() > 0.0, "{a:?}");
            }
            if a12 > 1.001 {
                assert!(!dominant && discs(a)[0].1 > discs(a)[0].0.abs(), "{a:?}");
            }
            seen.push(a12);
            if !scene.animated() {
                break;
            }
        }
        assert!(seen.windows(2).all(|w| w[1] >= w[0]) && seen.first().is_some_and(|v| *v < 0.05));
        let a = scene.shown().a;
        assert!(close(a, rows([1.0, 2.0, 0.5, 1.0]), 0.0) && is(Kind::Singular, a) && !is(Kind::Dominant, a));
        let Eigen::Real { l1, l2, .. } = eigen(a) else { panic!() };
        assert!(l1 == 2.0 && l2 == 0.0);
        assert!(lit_names(a).contains(&Kind::Singular) && !lit_names(rows([1.0, 0.0, 0.5, 1.0])).contains(&Kind::Singular) && lit_names(rows([1.0, 0.0, 0.5, 1.0])).contains(&Kind::Dominant));
    }

    #[test]
    fn the_positive_matrix_does_what_its_caption_says() {
        let scene = run(3);
        let a = scene.shown().a;
        assert!(scene.kind == Kind::Positive && is(Kind::Positive, a) && !is(Kind::Symmetric, a) && !is(Kind::PosDef, a));
        let Eigen::Real { l1, l2, lines: Lines::Two(v1, v2) } = eigen(a) else { panic!() };
        assert!(l1 == 2.0 && l2 == -0.5 && v1.perp_dot(DVec2::ONE).abs() < 1e-12 && v2.perp_dot(DVec2::new(3.0, -2.0)).abs() < 1e-12);
        let a2 = powered(a, 2.0);
        assert!((a2 * DVec2::ONE - DVec2::ONE * 4.0).length() < 1e-12 && (a2 * DVec2::new(3.0, -2.0) - DVec2::new(3.0, -2.0) * 0.25).length() < 1e-12 && 4.0 / 0.25 == 16.0);
        assert_eq!(scene.shown().power, 2.0);
    }

    #[test]
    fn the_swap_does_what_its_caption_says() {
        let scene = run(4);
        let a = scene.shown().a;
        assert!(scene.kind == Kind::Any && scene.map_alone && scene.family == Family::Named && a == SWAP);
        let lit = lit_names(a);
        let said = [Kind::Permutation, Kind::Reflection, Kind::Orthogonal, Kind::Symmetric, Kind::Normal, Kind::Indefinite, Kind::Involution, Kind::Root, Kind::Doubly, Kind::Markov, Kind::Companion];
        assert_eq!(lit.len(), 11);
        assert!(said.iter().all(|k| lit.contains(k)), "{lit:?}");
        assert!(a * DVec2::new(0.3, 0.7) == DVec2::new(0.7, 0.3) && does(Kind::Reflection, a) == "reflects the plane in the line at 45.0° to the $x$ axis");
        assert!(matches!(eigen(a), Eigen::Real { l1, l2, .. } if l1 == 1.0 && l2 == -1.0));
        assert_eq!(scene.heading_text(false).0, "Any matrix: $A$ is of 11 kinds");
    }

    /// What the heading says each kind's own matrix does, with its numbers.
    #[test]
    fn the_headings_say_what_the_matrices_do() {
        let say = |k: Kind| does(k, k.default_a());
        assert_eq!(say(Kind::Diagonal), "multiplies the $x$ axis by $1.50$ and the $y$ axis by $0.60$");
        assert_eq!(say(Kind::Upper), "keeps the $x$ axis on itself, multiplying it by $1.20$");
        assert_eq!(say(Kind::Lower), "keeps the $y$ axis on itself, multiplying it by $0.60$");
        assert!(Kind::Upper.default_a() * DVec2::X == DVec2::new(1.2, 0.0) && Kind::Lower.default_a() * DVec2::Y == DVec2::new(0.0, 0.6));
        assert_eq!(say(Kind::Symmetric), r"stretches along two perpendicular lines, by $\lambda = 2.00$ and $0.50$");
        assert_eq!(say(Kind::PosDef), r"stretches along two perpendicular lines, by $\lambda = 1.50$ and $0.50$, both positive");
        assert_eq!(say(Kind::PosSemidef), r"stretches along two perpendicular lines, by $\lambda = 2.00$ and $0.00$, neither negative");
        assert_eq!(say(Kind::Skew), "turns every arrow a quarter turn anticlockwise and multiplies its length by $1.20$");
        assert_eq!(say(Kind::Orthogonal), "turns the plane by 53.1°, keeping every length");
        assert_eq!(say(Kind::Rotation), "turns the plane by 53.1°, keeping every length");
        assert!(close(Kind::Orthogonal.default_a(), DMat2::from_angle(53.13010235415598f64.to_radians()), 1e-12));
        assert_eq!(say(Kind::Reflection), "reflects the plane in the line at 26.6° to the $x$ axis");
        assert_eq!(does(Kind::Orthogonal, Kind::Orthogonal.default_b()), "reflects the plane in the line at 26.6° to the $x$ axis");
        assert!((Kind::Reflection.default_a() * DVec2::new(2.0, 1.0) - DVec2::new(2.0, 1.0)).length() < 1e-12);
        assert_eq!(say(Kind::Normal), "turns the plane by 33.7° and multiplies every length by $1.08$");
        assert_eq!(say(Kind::ScaledRotation), "turns the plane by 22.6° and multiplies every length by $1.30$");
        let n = Kind::Normal.default_a();
        assert!(close(n, DMat2::from_angle(0.6f64.atan2(0.9)) * 0.9f64.hypot(0.6), 1e-12));
        assert_eq!(say(Kind::Scalar), "multiplies every vector by $1.30$");
        assert_eq!(say(Kind::JordanBlock), r"multiplies every point by $\lambda = 1.20$, and then slides it along the $x$ direction by its height");
        assert_eq!(say(Kind::Companion), "sends $(1, 0)$ to $(0, 1)$, and $(0, 1)$ to its last column, $(−0.50, 1.50)$");
        assert_eq!(say(Kind::Indefinite), r"stretches the orange line by $\lambda = 1.67$ and reverses the purple one, at right angles to it, by $-1.17$".replace('-', "−"));
        assert_eq!(say(Kind::Positive), r"stretches most along the orange line, by $\lambda = 2.00$, which runs through the positive quarter");
        assert_eq!(say(Kind::Doubly), r"maps shares to shares and keeps $(1, 1)$: each new coordinate is $0.80$ of one old one and $0.20$ of the other");
        assert_eq!(say(Kind::Root), "brings every point back where it started after $6$ steps");
        assert_eq!(order(Kind::Root.default_a()).0, 6);
        assert_eq!(order(Kind::Root.default_b()).0, 3);
        // the orange line is the line kept (λ = 1), the purple one the line sent to 0
        for kind in [Kind::Projection, Kind::OrthProjection] {
            let p = kind.default_a();
            let Eigen::Real { l1, l2, lines: Lines::Two(v1, v2) } = eigen(p) else { panic!() };
            assert!((l1 - 1.0).abs() < 1e-12 && l2.abs() < 1e-12 && (p * v1 - v1).length() < 1e-12 && (p * v2).length() < 1e-12);
            assert_eq!(v1.dot(v2).abs() < 1e-9, kind == Kind::OrthProjection);
        }
        // shares go into shares (Markov), and (1, 1) is kept too (doubly stochastic)
        for kind in [Kind::Markov, Kind::Doubly] {
            let m = kind.default_a();
            for t in 0..=10 {
                let q = m * DVec2::new(t as f64 / 10.0, 1.0 - t as f64 / 10.0);
                assert!(q.x >= 0.0 && q.y >= 0.0 && (q.x + q.y - 1.0).abs() < 1e-12);
            }
        }
        assert!(Kind::Doubly.default_a() * DVec2::ONE == DVec2::ONE);
        // the involution keeps the x axis and reverses (1, −2)
        assert_eq!(say(Kind::Involution), "keeps the orange line and reverses the purple one: done twice, nothing has moved");
        // a nilpotent matrix's first step lands everything on the line it sends to 0
        let a = Kind::Nilpotent.default_a();
        let Eigen::Real { l1, lines: Lines::One(v), .. } = eigen(a) else { panic!() };
        assert!(l1 == 0.0 && (a * v).length() < 1e-12 && size(a * a) == 0.0 && F_OUTLINE.iter().all(|p| (a * *p).perp_dot(v).abs() < 1e-12));
        assert!(size(powered(a, 2.0)) == 0.0 && close(powered(a, 1.0), a, 0.0) && close(powered(a, 0.0), DMat2::IDENTITY, 0.0));
        // every heading is finished and says something
        for kind in KINDS {
            assert!(!kind.closure(Op::Alone).contains('.') && !kind.eigen_words().is_empty() && !does(kind, kind.default_a()).is_empty(), "{kind:?}");
            for op in [Op::Sum, Op::Product, Op::Inverse, Op::Transpose] {
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
            for kind in KINDS.into_iter().chain([Kind::Any]) {
                for n in ["A", "A + B", "AB", "A^{-1}", r"A^\top"] {
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
                        for power in [0.0, 0.5, 1.0, 1.7, 2.0, -1.0, 9.0, 12.0, 40.0, f64::NAN] {
                            assert!(powered(c, power).is_finite());
                        }
                        let e = eigen(c);
                        assert!(eigenvalues(e).iter().all(|(re, im)| re.is_finite() && im.is_finite()));
                        assert!(!eigen_text(e).contains("NaN"));
                        let (_, off, _) = order(c);
                        assert!(off.is_finite());
                    }
                }
            }
            for kind in KINDS.into_iter().chain([Kind::Any]) {
                for j in 0..2 {
                    assert!(fit(kind, a, j, DVec2::new(0.3, 0.2)).is_finite());
                }
                assert!(random(kind, &mut Rng::new(1)).is_finite());
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
        for kind in KINDS.into_iter().chain([Kind::Any]) {
            let mut scene = Special::new();
            scene.choose(kind);
            scene.another(false);
            scene.choose_op(Op::Product);
            scene.another(true);
            scene.steps = Some((0.0, scene.steps_top(scene.glide.to.a)));
            for _ in 0..60 * 30 {
                scene.advance(1.0 / 60.0, 1.0 / 60.0);
            }
            assert!(!scene.animated() && is(kind, scene.glide.to.a) && is(kind, scene.glide.to.b), "{kind:?}");
        }
        // a root of I plays its steps all the way back to I
        let mut scene = Special::new();
        scene.choose(Kind::Root);
        let a = scene.glide.to.a;
        let top = scene.steps_top(a);
        assert_eq!(top, 6);
        scene.steps = Some((0.0, top));
        for _ in 0..60 * 30 {
            scene.advance(1.0 / 60.0, 1.0 / 60.0);
        }
        assert!(scene.shown().power == 6.0 && close(powered(a, 6.0), DMat2::IDENTITY, 1e-12));
    }

    /// The panel, the picture and the map fit the canvas, from a phone to a desktop,
    /// and the panel's parts stay inside it, one below another.
    #[test]
    fn the_layout_fits_every_canvas() {
        for (w, h) in [(1080.0, 1000.0), (580.0, 700.0), (390.0, 800.0), (390.0, 560.0), (360.0, 640.0), (800.0, 600.0), (1600.0, 1000.0), (240.0, 180.0), (620.0, 480.0), (1300.0, 700.0), (1400.0, 1100.0)] {
            for inset in [0.0, 46.0] {
                for two in [false, true] {
                    for alone in [false, true] {
                        let view = Vec2::new(w, h);
                        let lay = Layout::of(view, inset, two, alone);
                        let (min, max) = lay.picture;
                        assert!(min.cmpge(Vec2::ZERO).all() && max.cmple(view).all() && (max - min).min_element() > 50.0, "{w}x{h}: {lay:?}");
                        if let Some((lo, hi)) = lay.map {
                            assert!(lo.cmpge(Vec2::ZERO).all() && hi.cmple(view).all() && (hi - lo).min_element() > 200.0, "{w}x{h}: {lay:?}");
                            if let Some(p) = &lay.panel {
                                assert!(!overlap((lo, hi), (p.min, p.max)), "{w}x{h}: the map under the panel");
                            }
                            if lay.show == Show::Both {
                                assert!(lay.picture.1.y <= lo.y, "{w}x{h}: the map under the plane");
                            }
                        }
                        assert_eq!(lay.show == Show::Map, alone && lay.map.is_some(), "{w}x{h}");
                        if let Some(p) = &lay.panel {
                            let within = |q: Vec2| q.cmpge(p.min - Vec2::splat(0.5)).all() && q.cmple(p.max + Vec2::splat(0.5)).all();
                            assert!(p.min.cmpge(Vec2::ZERO).all() && p.max.cmple(view + Vec2::splat(0.5)).all(), "{w}x{h}: {p:?}");
                            assert!(p.tabs.iter().chain(p.chips.iter().flatten()).all(|(lo, hi)| within(*lo) && within(*hi)), "{w}x{h}: {p:?}");
                            let chips_end = p.chips.iter().flatten().fold(0.0f32, |m, c| m.max(c.1.y));
                            assert!(p.tabs.iter().all(|t| t.1.y < chips_end) && chips_end < p.title.y, "{w}x{h}");
                            if let (Some(chart), Some((a, b))) = (p.chart, p.another) {
                                let last = p.cells + Vec2::new((p.per_row - 1) as f32 * p.step.x + matrix_w(p.text), p.rows as f32 * p.step.y);
                                assert!(within(chart.rect.0) && within(chart.rect.1) && within(a.1) && within(b.1), "{w}x{h}: {p:?}");
                                assert!(within(last) && last.y <= p.lines_end && p.lines_end <= chart.title.y && chart.rect.1.y < a.0.y, "{w}x{h}: {p:?}");
                                if let Some(t) = p.table {
                                    assert!(t.y >= p.lines_end && t.y + 5.0 * TABLE_LINE <= chart.title.y);
                                }
                            }
                            assert!(p.min.y >= inset || lay.narrow, "{w}x{h}: under the buttons");
                            if let Some((_, _, foot)) = lay.caption {
                                assert!(foot <= p.min.y || !lay.narrow, "{w}x{h}: the caption over the panel");
                            }
                        }
                    }
                }
            }
        }
    }
}

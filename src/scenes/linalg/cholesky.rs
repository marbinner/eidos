//! Cholesky: a symmetric matrix that is positive definite is L Lᵀ for exactly one lower
//! triangular L with a positive diagonal, found in three steps. L carries a round cloud
//! of independent normal numbers to a cloud whose covariance is A, and the unit circle
//! to the ellipse xᵀA⁻¹x = 1. The recipe meets a number that is not positive under a
//! root exactly when A is not positive definite; and L is one square root of A among
//! many, all L times a turn or a mirror.

use super::*;
use glam::{DMat2, DVec2};

pub const DEF: SceneDef = SceneDef {
    id: "cholesky",
    title: "Cholesky: a triangular square root",
    group: FACTOR,
    blurb: "A positive definite matrix as a triangular matrix times its transpose: elimination made half the work, and how correlated numbers are made.",
    make: |_| Box::new(Cholesky::new()),
};

// ----- the arithmetic, in double precision -------------------------------------------

/// A number this small beside the matrix's own is taken for 0: rounding leaves crumbs
/// of about 10⁻¹⁶ times them where there should be nothing.
const EXACT: f64 = 1e-12;

/// The smallest and the largest spread σ either way, and the largest size of r.
const S_MIN: f64 = 0.3;
const S_MAX: f64 = 1.5;
const R_MAX: f64 = 1.4;
/// How far out the white point z may be dragged, either way.
const Z_MAX: f64 = 2.8;

/// A symmetric 2×2 matrix, by its three numbers.
#[derive(Clone, Copy, Debug, PartialEq)]
struct Sym {
    a11: f64,
    a21: f64,
    a22: f64,
}

impl Sym {
    /// The covariance matrix of two numbers with spreads (standard deviations) `s1` and
    /// `s2` and correlation `r`.
    fn of(s1: f64, s2: f64, r: f64) -> Sym {
        Sym { a11: s1 * s1, a21: r * s1 * s2, a22: s2 * s2 }
    }

    fn mat(self) -> DMat2 {
        DMat2::from_cols(DVec2::new(self.a11, self.a21), DVec2::new(self.a21, self.a22))
    }

    /// The size of its largest number.
    fn size(self) -> f64 {
        self.a11.abs().max(self.a21.abs()).max(self.a22.abs())
    }

    fn det(self) -> f64 {
        self.a11 * self.a22 - self.a21 * self.a21
    }

    /// xᵀAx.
    #[cfg(test)]
    fn form(self, x: DVec2) -> f64 {
        self.a11 * x.x * x.x + 2.0 * self.a21 * x.x * x.y + self.a22 * x.y * x.y
    }

    /// The eigenvalues, the larger first, and a unit eigenvector of the larger.
    fn eigen(self) -> (f64, f64, DVec2) {
        let (mid, half) = (0.5 * (self.a11 + self.a22), 0.5 * (self.a11 - self.a22));
        let rad = half.hypot(self.a21);
        // (each of these is across one row of A − λ₁I: the longer is the better made)
        let (u, v) = (DVec2::new(rad + half, self.a21), DVec2::new(self.a21, rad - half));
        let e = if u.length_squared() >= v.length_squared() { u } else { v };
        (mid + rad, mid - rad, if e.length_squared() > 0.0 { e.normalize() } else { DVec2::X })
    }
}

/// What the recipe makes of a symmetric matrix, step by step:
///
/// 1. l₁₁ = √a₁₁,  2. l₂₁ = a₂₁ / l₁₁,  3. l₂₂ = √(a₂₂ − l₂₁²).
#[derive(Clone, Copy, Debug, PartialEq)]
enum Chol {
    /// Both numbers under the roots positive: A is positive definite, and this L, with
    /// its positive diagonal, is the only one.
    Definite { l11: f64, l21: f64, l22: f64 },
    /// A 0 under the last root (to rounding): A is semidefinite, and l₂₂ = 0.
    Flat { l11: f64, l21: f64 },
    /// The last root is of a negative number, `under`: A is not positive definite, and
    /// has no real L.
    Fails3 { l11: f64, l21: f64, under: f64 },
    /// The first root is of a number that is not positive (a₁₁ itself), so there is
    /// nothing to divide by in step 2.
    Fails1 { a11: f64 },
}

fn cholesky(a: Sym) -> Chol {
    if a.a11 <= EXACT * a.size() || !a.a11.is_finite() {
        return Chol::Fails1 { a11: a.a11 };
    }
    let l11 = a.a11.sqrt();
    let l21 = a.a21 / l11;
    let under = a.a22 - l21 * l21;
    // (a difference of two numbers of about the size of these: 0 to rounding below this)
    let zero = EXACT * a.size().max(l21 * l21);
    if under > zero {
        Chol::Definite { l11, l21, l22: under.sqrt() }
    } else if under >= -zero {
        Chol::Flat { l11, l21 }
    } else {
        Chol::Fails3 { l11, l21, under }
    }
}

/// The lower triangular matrix with these three numbers.
fn lower(l11: f64, l21: f64, l22: f64) -> DMat2 {
    DMat2::from_cols(DVec2::new(l11, l21), DVec2::new(0.0, l22))
}

impl Chol {
    /// L, if the recipe finished.
    fn l(self) -> Option<DMat2> {
        match self {
            Chol::Definite { l11, l21, l22 } => Some(lower(l11, l21, l22)),
            Chol::Flat { l11, l21 } => Some(lower(l11, l21, 0.0)),
            _ => None,
        }
    }

    /// The number under the last root, a₂₂ − l₂₁² (none if the recipe stopped before).
    fn under(self) -> Option<f64> {
        match self {
            Chol::Definite { l22, .. } => Some(l22 * l22),
            Chol::Flat { .. } => Some(0.0),
            Chol::Fails3 { under, .. } => Some(under),
            Chol::Fails1 { .. } => None,
        }
    }
}

/// The turn by `theta` (radians, anticlockwise).
fn turn(theta: f64) -> DMat2 {
    let (s, c) = theta.sin_cos();
    DMat2::from_cols(DVec2::new(c, s), DVec2::new(-s, c))
}

/// The angle of the turn Q that takes L to the symmetric square root: S = LQ. Then
/// Lᵀ = QS, the polar form of Lᵀ, whose turn is by −atan2(l₂₁, l₁₁ + l₂₂): within a
/// right angle either way, since l₁₁ + l₂₂ > 0.
fn root_turn(l: DMat2) -> f64 {
    -(l.x_axis.y.atan2(l.x_axis.x + l.y_axis.y))
}

/// The symmetric square root of a semidefinite A: the S with Sᵀ = S, S² = A and no
/// negative eigenvalue (A's eigenvectors, the roots of A's eigenvalues). For 2×2,
/// S = (A + √det A · I) / √(tr A + 2√det A). None if A is not semidefinite.
fn sym_root(a: Sym) -> Option<DMat2> {
    let (_, l2, _) = a.eigen();
    if l2 < -EXACT * a.size() {
        return None;
    }
    let d = a.det().max(0.0).sqrt();
    let t = a.a11 + a.a22 + 2.0 * d;
    if t <= 0.0 {
        return Some(DMat2::ZERO);
    }
    Some((a.mat() + DMat2::IDENTITY * d) * (1.0 / t.sqrt()))
}

/// 0 to 1, slow at both ends.
fn ease(t: f64) -> f64 {
    let t = t.clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

/// L played as two steps: at `stage` 0 nothing has happened (I); by 1 the plane is
/// stretched along the axes, by l₁₁ across and l₂₂ up; by 2 it is sheared, each point
/// raised by m = l₂₁/l₁₁ times its first number, and the whole is L.
fn film(l: DMat2, stage: f64) -> DMat2 {
    let (l11, l21, l22) = (l.x_axis.x, l.x_axis.y, l.y_axis.y);
    let (a, b) = (ease(stage), ease(stage - 1.0));
    let stretch = DMat2::from_diagonal(DVec2::new(1.0 + (l11 - 1.0) * a, 1.0 + (l22 - 1.0) * a));
    let m = if l11 > 0.0 { l21 / l11 } else { 0.0 };
    let shear = DMat2::from_cols(DVec2::new(1.0, m * b), DVec2::Y);
    shear * stretch
}

/// The multiplier elimination uses, a₂₁/a₁₁, which is l₂₁/l₁₁.
fn multiplier(a: Sym) -> f64 {
    if a.a11 != 0.0 {
        a.a21 / a.a11
    } else {
        0.0
    }
}

// ----- the cloud ---------------------------------------------------------------------

/// How many pairs z the cloud has, and the seed they are drawn with.
const SAMPLE: usize = 400;
const SEED: u64 = 7;

/// The pairs of independent standard normal numbers.
fn sample() -> Vec<DVec2> {
    let mut rng = Rng::new(SEED);
    (0..SAMPLE)
        .map(|_| {
            let (a, b) = (rng.normal(), rng.normal());
            DVec2::new(a as f64, b as f64)
        })
        .collect()
}

/// The sample covariance of points (their means taken out, divided by n − 1).
fn covariance(points: &[DVec2]) -> DMat2 {
    let n = points.len().max(2) as f64;
    let mean = points.iter().fold(DVec2::ZERO, |s, p| s + *p) / points.len().max(1) as f64;
    let (mut xx, mut xy, mut yy) = (0.0, 0.0, 0.0);
    for p in points {
        let d = *p - mean;
        (xx, xy, yy) = (xx + d.x * d.x, xy + d.x * d.y, yy + d.y * d.y);
    }
    DMat2::from_cols(DVec2::new(xx, xy), DVec2::new(xy, yy)) * (1.0 / (n - 1.0))
}

// ----- experiments -------------------------------------------------------------------

struct Trial {
    label: &'static str,
    look_for: &'static str,
    s1: f64,
    s2: f64,
    r: f64,
    /// Show the symmetric square root (reached from L, through L times smaller turns).
    symmetric: bool,
    /// Play the two steps of L from the start.
    steps: bool,
    /// A second r, glided to after a pause.
    then_r: Option<f64>,
}

/// Where every experiment puts the white point z.
const Z0: DVec2 = DVec2::new(-1.5, 0.5);

const TRIALS: [Trial; 5] = [
    Trial {
        label: "correlation from independence",
        look_for: r"$\sigma_1 = \sigma_2 = 1$ and $r = 0.8$. The recipe gives $l_{11} = 1$, $l_{21} = 0.8$ and $l_{22} = \sqrt{1 - 0.64} = 0.6$, so $x_1 = z_1$ and $x_2 = 0.8 z_1 + 0.6 z_2$. Watch the two steps: the round cloud is squeezed to 0.6 of its height, then sheared, each point raised by 0.8 times its $x_1$. Each $x$ has variance $1$ ($0.8^2 + 0.6^2 = 1$), and the $z_1$ they share makes their covariance $0.8$.",
        s1: 1.0,
        s2: 1.0,
        r: 0.8,
        symmetric: false,
        steps: true,
        then_r: None,
    },
    Trial {
        label: "nearly a line, then a line",
        look_for: r"$r = 0.99$: $l_{22} = \sqrt{1 - 0.9801} = 0.14$, and the cloud is a needle along $x_2 = x_1$. Then $r = 1$: the last root is $\sqrt{0} = 0$, the green column is gone, and every $x$ lands on that line. $A$ is now semidefinite, not definite: $x^\top A x = 0$ for $x = (-1, 1)$. $L$ still exists (with $a_{11} > 0$ it is the only one with no negative number on its diagonal), but it squashes the plane flat and has no inverse.",
        s1: 1.0,
        s2: 1.0,
        r: 0.99,
        symmetric: false,
        steps: false,
        then_r: Some(1.0),
    },
    Trial {
        label: "not positive definite",
        look_for: r"$r = 1.2$, past any correlation. Steps 1 and 2 still go through: the first column is $(1, 1.2)$, its tip above the box. Step 3 needs $\sqrt{1 - 1.44} = \sqrt{-0.44}$, and no real number squared is negative: there is no real $L$ and no ellipse. That $-0.44$ is $x^\top A x$ for $x = (-1.2, 1)$, so $A$ is not positive definite (its eigenvalues are $2.2$ and $-0.2$). Trying the recipe is a cheap test: it meets a number that is not positive under a root exactly when $A$ is not positive definite.",
        s1: 1.0,
        s2: 1.0,
        r: 1.2,
        symmetric: false,
        steps: false,
        then_r: None,
    },
    Trial {
        label: "two square roots",
        look_for: r"The same $A$ as in the first. Its symmetric square root $S$ with no negative eigenvalue (0.89 on the diagonal, 0.45 off it) has $SS^\top = S^2 = A$ too: the ellipse stays where it is, and $x = Sz$ has the same covariance. But the marks slide round the ellipse. $S = LQ$, where $Q$ turns the plane 26.6° clockwise, so $S$ sends each point of the circle where $L$ sends that point turned 26.6° clockwise. Every matrix on the way, $L$ times a smaller turn, is a square root of $A$ as well.",
        s1: 1.0,
        s2: 1.0,
        r: 0.8,
        symmetric: true,
        steps: false,
        then_r: None,
    },
    Trial {
        label: "the first step is elimination",
        look_for: r"$a_{11} = 1.44$, $a_{21} = 0.72$, $a_{22} = 1$. Elimination takes $0.72/1.44 = 0.5$ times row 1 from row 2, which leaves the pivots $1.44$ and $1 - 0.5 \times 0.72 = 0.64$. Their square roots, $1.2$ and $0.8$, are $l_{11}$ and $l_{22}$, and the multiplier $0.5$ is $l_{21}/l_{11}$. So $L$ is the stretch by the roots of the pivots followed by the shear by the multiplier: the two steps played here.",
        s1: 1.2,
        s2: 1.0,
        r: 0.6,
        symmetric: false,
        steps: true,
        then_r: None,
    },
];

// ----- what glides -------------------------------------------------------------------

/// What glides when a button changes it: the matrix (as σ₁, σ₂ and r), how far the
/// steps of L have got, and the white point z.
#[derive(Clone, Copy, Debug, PartialEq)]
struct Setting {
    s1: f64,
    s2: f64,
    r: f64,
    stage: f64,
    z: DVec2,
}

impl Setting {
    fn numbers(&self) -> [f64; 6] {
        [self.s1, self.s2, self.r, self.stage, self.z.x, self.z.y]
    }

    fn from_numbers(v: [f64; 6]) -> Setting {
        Setting { s1: v[0], s2: v[1], r: v[2], stage: v[3], z: DVec2::new(v[4], v[5]) }
    }

    fn a(&self) -> Sym {
        Sym::of(self.s1, self.s2, self.r)
    }
}

/// How fast a change of setting is glided to: each of the two eases one after the other
/// covers two thirds of what is left in this many seconds.
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
        for i in 0..6 {
            between[i] += (to[i] - between[i]) * k;
            shown[i] += (between[i] - shown[i]) * k;
            apart = apart.max((to[i] - between[i]).abs()).max((to[i] - shown[i]).abs());
        }
        // (exactly there in the end: the picture then stops changing. Near r = ±1 a step
        // of 10⁻⁴ in r is one of 0.014 in l₂₂ = σ₂√(1 − r²), so the last step is a small one)
        if apart < 1e-6 || !apart.is_finite() {
            *self = Glide::at(self.to);
        } else {
            (self.between, self.shown) = (Setting::from_numbers(between), Setting::from_numbers(shown));
        }
    }
}

/// Seconds for each of the two steps of L as they play, for the change from L to the
/// symmetric root, and for the pause before an experiment's second stop.
const STEP_TIME: f64 = 1.4;
const BLEND_TIME: f64 = 1.8;
const HOLD: f64 = 1.8;

// ----- where things go ---------------------------------------------------------------

/// Each picture shows its plane from −2.5 to 2.5 either way (a point of the cloud has a
/// number beyond that about once in forty).
const HALF: f64 = 2.5;
/// Rows of the numbers under the pictures, in logical pixels.
const BUTTON_H: f32 = 26.0;
const PAD: f32 = 12.0;
const NAMES_H: f32 = 15.0;
/// The widest number in a matrix there, in characters (as `hud_matrix` counts them: a
/// minus sign, nearly twice a digit's width in a formula, counted as two); one fewer in
/// a panel too narrow for that, where the cells are just wide enough for "−0.80".
fn cell_chars(width: f32) -> usize {
    if width >= 380.0 {
        5
    } else {
        4
    }
}

/// The buttons on the numbers' panel (and shorter, for a narrow one), and the third
/// one's label while the steps play.
const BUTTONS: [[&str; 3]; 2] = [["$L$, triangular", "$S$, symmetric", "▶ the steps of $L$"], ["$L$", "$S$", "▶ steps"]];
const PAUSE: &str = "❚❚ pause";

/// From one of the two planes (z's or x's) to the stage, and back.
#[derive(Clone, Copy, Debug, PartialEq)]
struct Map {
    /// Where the plane's 0 is on the stage, and stage units per unit of the plane.
    centre: Vec2,
    k: f32,
}

impl Map {
    fn to(&self, p: DVec2) -> Vec2 {
        self.centre + p.as_vec2() * self.k
    }

    fn back(&self, w: Vec2) -> DVec2 {
        ((w - self.centre) / self.k.max(1e-6)).as_dvec2()
    }

    /// Does the plane's point lie within the picture?
    fn shows(&self, p: DVec2) -> bool {
        p.is_finite() && p.abs().max_element() <= HALF
    }
}

/// The numbers under the pictures: the buttons, the matrices, the recipe.
#[derive(Clone, Copy, Debug, PartialEq)]
struct Hud {
    min: Vec2,
    max: Vec2,
    /// In two columns (the matrices on the left, the recipe on the right), or one.
    two: bool,
    /// The size of the numbers and of the recipe's lines, and the recipe's line height.
    text: f32,
    line: f32,
    buttons: [(Vec2, Vec2); 3],
    labels: [&'static str; 3],
    /// A matrix's cells, in characters (`cell_chars`).
    cell: usize,
    /// Top-left corners: the row of matrices, the line under it (two columns only),
    /// and the recipe's first line.
    mats: Vec2,
    note: Option<Vec2>,
    recipe: Vec2,
}

impl Hud {
    /// Text size and line height for a panel this wide.
    fn sizes(width: f32, two: bool) -> (f32, f32) {
        let text = if two && width >= 1000.0 {
            14.0
        } else if two && width >= 900.0 {
            13.0
        } else if width >= 360.0 {
            12.0
        } else {
            11.0
        };
        (text, text + if two { 6.0 } else { 5.0 })
    }

    /// How tall it is, at this width.
    fn height(width: f32) -> f32 {
        let two = width >= 760.0;
        let (text, line) = Hud::sizes(width, two);
        let mats = matrix_size(2, 2, cell_chars(width), text).y + NAMES_H;
        if two {
            // (the recipe's column: a heading, three steps, a verdict of up to two lines, z)
            (BUTTON_H + 10.0 + mats + 6.0 + line).max(7.0 * line) + 2.0 * PAD
        } else {
            BUTTON_H + 8.0 + mats + 6.0 + 5.0 * line + 2.0 * PAD
        }
    }

    fn place(min: Vec2, width: f32) -> Hud {
        let two = width >= 760.0;
        let (text, line) = Hud::sizes(width, two);
        let max = min + Vec2::new(width, Hud::height(width));
        let x = min.x + PAD;
        let mut y = min.y + PAD;
        let row = |labels: [&str; 3]| {
            let widths = [text_width(labels[0], 12.5), text_width(labels[1], 12.5), text_width(labels[2], 12.5).max(text_width(PAUSE, 12.5))];
            let mut buttons = [(Vec2::ZERO, Vec2::ZERO); 3];
            let mut bx = x;
            for (k, w) in widths.iter().enumerate() {
                // (a gap between the switch and the steps)
                if k == 2 {
                    bx += 10.0;
                }
                buttons[k] = (Vec2::new(bx, y), Vec2::new(bx + w + 20.0, y + BUTTON_H));
                bx += w + 26.0;
            }
            buttons
        };
        let mut labels = BUTTONS[0];
        let mut buttons = row(labels);
        if buttons[2].1.x > max.x - PAD {
            labels = BUTTONS[1];
            buttons = row(labels);
        }
        y += BUTTON_H + if two { 10.0 } else { 8.0 };
        let mats = Vec2::new(x, y);
        let mats_h = matrix_size(2, 2, cell_chars(width), text).y + NAMES_H;
        y += mats_h + 6.0;
        let (note, recipe) = if two {
            // the recipe in a column of its own, from the top
            let left = mats_width(text, cell_chars(width)) + 34.0;
            (Some(Vec2::new(x, y)), Vec2::new(x + left, min.y + PAD))
        } else {
            (None, Vec2::new(x, y))
        };
        Hud { min, max, two, text, line, buttons, labels, cell: cell_chars(width), mats, note, recipe }
    }

    /// How wide the recipe's column is.
    fn recipe_width(&self) -> f32 {
        self.max.x - PAD - self.recipe.x
    }

    /// What the recipe found, on one line if it fits across the column, else on two.
    fn verdict(&self, a: Sym) -> Vec<(String, Color)> {
        let long = verdict(a, false);
        if long.iter().all(|(text, _)| text_width(text, self.text - 0.5) <= self.recipe_width()) {
            long
        } else {
            verdict(a, true)
        }
    }
}

/// How wide the row "A = [ ] = [ ][ ]" is at a text size, with cells of so many characters.
fn mats_width(text: f32, cell: usize) -> f32 {
    let m = matrix_size(2, 2, cell, text).x;
    text_width("$A =$", text) + 8.0 + m + 8.0 + text_width("$=$", text) + 8.0 + m + 6.0 + m
}

#[derive(Clone, Copy, Debug, PartialEq)]
struct Layout {
    view: Vec2,
    /// One picture above the other (a phone held upright), not side by side.
    narrow: bool,
    /// Logical pixels per unit of the stage, and per unit of either plane.
    ppu: f32,
    unit: f32,
    /// The two pictures, z's and x's, as squares on the canvas (logical pixels).
    panes: [(Vec2, Vec2); 2],
    /// Room for words: titles, names, the numbers (none on a canvas too small).
    words: bool,
    /// The arrow from the one picture to the other (logical pixels).
    between: (Vec2, Vec2),
    hud: Option<Hud>,
}

impl Layout {
    /// The pictures side by side, or one above the other, whichever makes them larger
    /// (the second only on a canvas taller than it is wide: a phone held upright).
    fn of(view: Vec2, inset: f32) -> Layout {
        let wide = Layout::arranged(view, inset, false);
        if view.y > view.x {
            let tall = Layout::arranged(view, inset, true);
            if tall.unit > 1.05 * wide.unit {
                return tall;
            }
        }
        wide
    }

    fn arranged(view: Vec2, inset: f32, narrow: bool) -> Layout {
        let view = view.max(Vec2::ONE);
        let ppu = (view.y / STAGE_H).min(view.x / 2.0).max(1e-3);
        let words = view.x >= 300.0 && view.y >= 330.0;
        let top = inset.max(0.0) + 8.0;
        let avail = (view.y - top - 8.0).max(1.0);
        // (a title above each picture: one above the other, the second's goes in the gap
        // between them, beside the arrow from the first to the second)
        let title = if words { if narrow { 20.0 } else { 24.0 } } else { 0.0 };
        let gap = if narrow { title + 12.0 } else { (view.x * 0.07).clamp(36.0, 90.0) };
        let width = if narrow { view.x - 16.0 } else { view.x - 24.0 };
        let side_with = |hud_h: f32| {
            let room = if hud_h > 0.0 { hud_h + 12.0 } else { 0.0 };
            if narrow {
                ((avail - title - gap - room) * 0.5).min(view.x - 20.0)
            } else {
                ((view.x - 24.0 - gap) * 0.5).min(avail - title - room)
            }
        };
        // the numbers, if they leave the pictures room enough
        let mut hud_h = if words { Hud::height(width) } else { 0.0 };
        if hud_h > 0.0 && side_with(hud_h) < 150.0 {
            hud_h = 0.0;
        }
        let side = side_with(hud_h).max(8.0);
        let room = if hud_h > 0.0 { hud_h + 12.0 } else { 0.0 };
        let block = if narrow { title + 2.0 * side + gap + room } else { title + side + room };
        let y0 = top + ((avail - block) * 0.5).max(0.0);
        let (panes, between, hud_at) = if narrow {
            let x0 = ((view.x - side) * 0.5).max(0.0);
            let y = y0 + title;
            let first = (Vec2::new(x0, y), Vec2::new(x0 + side, y + side));
            let second = (Vec2::new(x0, y + side + gap), Vec2::new(x0 + side, y + 2.0 * side + gap));
            // (the arrow at the right of the gap, the second title at its left)
            let at = x0 + side - 30.0;
            ([first, second], (Vec2::new(at, first.1.y + 3.0), Vec2::new(at, second.0.y - 3.0)), Vec2::new(8.0, second.1.y + 12.0))
        } else {
            let x0 = ((view.x - 2.0 * side - gap) * 0.5).max(0.0);
            let y = y0 + title;
            let first = (Vec2::new(x0, y), Vec2::new(x0 + side, y + side));
            let second = (Vec2::new(x0 + side + gap, y), Vec2::new(x0 + 2.0 * side + gap, y + side));
            let mid = y + side * 0.5;
            ([first, second], (Vec2::new(first.1.x + 6.0, mid), Vec2::new(second.0.x - 6.0, mid)), Vec2::new(12.0, first.1.y + 12.0))
        };
        let hud = (hud_h > 0.0).then(|| Hud::place(hud_at, width));
        Layout { view, narrow, ppu, unit: side / (2.0 * HALF as f32), panes, words, between, hud }
    }

    /// How far across each picture's title may run: to the next title, the arrow between
    /// the pictures, or the canvas's edge.
    fn title_room(&self) -> [f32; 2] {
        let [(a0, a1), (b0, _)] = self.panes;
        if self.narrow {
            [self.view.x - a0.x - 10.0, self.between.0.x - 30.0 - b0.x]
        } else {
            [b0.x - a0.x - 14.0, self.view.x - b0.x - 10.0].map(|w| w.max(a1.x - a0.x))
        }
    }

    /// A point of the canvas (logical pixels) on the stage.
    fn stage(&self, px: Vec2) -> Vec2 {
        Vec2::new(px.x - self.view.x * 0.5, self.view.y * 0.5 - px.y) / self.ppu
    }

    /// From picture `k`'s plane (0: z's, 1: x's) to the stage.
    fn map(&self, k: usize) -> Map {
        let (lo, hi) = self.panes[k];
        Map { centre: self.stage((lo + hi) * 0.5), k: self.unit / self.ppu }
    }
}

fn inside_rect(p: Vec2, (min, max): (Vec2, Vec2)) -> bool {
    p.cmpge(min).all() && p.cmple(max).all()
}

// ----- the scene ---------------------------------------------------------------------

pub struct Cholesky {
    glide: Glide,
    /// The steps of L playing; and waiting to play until the glide to a new setting,
    /// and the change back to L, are done.
    playing: bool,
    play_when_there: bool,
    /// From L (0) to the symmetric root (1): where the change has got to, where it is
    /// going, and where it goes after that (an experiment that starts it again).
    blend: f64,
    blend_to: f64,
    blend_then: Option<f64>,
    /// An experiment's second r, and the seconds before it is glided to.
    then: Option<(f64, f64)>,
    /// The pairs z, and their sample covariance.
    zs: Vec<DVec2>,
    cz: DMat2,
}

/// Which square root is drawn.
#[derive(Clone, Copy, Debug, PartialEq)]
enum Root {
    L,
    S,
    /// L times a turn, on the way from the one to the other.
    Between,
}

impl Cholesky {
    fn new() -> Self {
        let zs = sample();
        let cz = covariance(&zs);
        let start = Setting { s1: 1.0, s2: 1.0, r: 0.8, stage: 2.0, z: Z0 };
        let mut scene = Cholesky {
            glide: Glide::at(start),
            playing: false,
            play_when_there: false,
            blend: 0.0,
            blend_to: 0.0,
            blend_then: None,
            then: None,
            zs,
            cz,
        };
        scene.set_up(0);
        // (opening, the first experiment's picture as it ends: nothing to play or glide)
        scene.glide = Glide::at(Setting { stage: 2.0, ..scene.glide.to });
        (scene.playing, scene.play_when_there) = (false, false);
        scene
    }

    /// Set up experiment `k` of `TRIALS`.
    fn set_up(&mut self, k: usize) {
        let t = &TRIALS[k];
        self.glide.to = Setting { s1: t.s1, s2: t.s2, r: t.r, stage: if t.steps { 0.0 } else { 2.0 }, z: Z0 };
        self.playing = false;
        self.play_when_there = t.steps;
        self.then = t.then_r.map(|r| (HOLD, r));
        if t.symmetric {
            // (from L, so that the way there is seen: back first if it is not at L)
            (self.blend_to, self.blend_then) = if self.blend > 0.0 { (0.0, Some(1.0)) } else { (1.0, None) };
        } else {
            (self.blend_to, self.blend_then) = (0.0, None);
        }
    }

    /// A change made by hand (a handle, a slider): at once, and the picture shows all
    /// of L (the steps, if they were playing, glide to their end).
    fn by_hand(&mut self, change: impl Fn(&mut Setting)) {
        self.glide.now(change);
        self.glide.to.stage = 2.0;
        (self.playing, self.play_when_there, self.then) = (false, false, None);
    }

    /// Show L, or the symmetric root.
    fn choose(&mut self, symmetric: bool) {
        (self.blend_to, self.blend_then) = (if symmetric { 1.0 } else { 0.0 }, None);
        if symmetric {
            (self.playing, self.play_when_there) = (false, false);
            self.glide.to.stage = 2.0;
        }
    }

    /// Play the steps of L from the start (back to L first), or pause them.
    fn play_steps(&mut self) {
        if self.playing || self.play_when_there {
            (self.playing, self.play_when_there) = (false, false);
            return;
        }
        if cholesky(self.glide.to.a()).l().is_none() {
            return;
        }
        (self.blend_to, self.blend_then) = (0.0, None);
        if self.glide.to.stage >= 2.0 {
            self.glide.to.stage = 0.0;
        }
        self.play_when_there = true;
    }

    /// Move everything on: `dt` seconds of the scene's time (the steps, the change of
    /// root, the pause) and `real_dt` of the clock's (a change of setting glides by the
    /// clock, so that it is seen through even while the scene's time stands still).
    fn advance(&mut self, dt: f32, real_dt: f32) {
        self.glide.step(real_dt as f64);
        let dt = if dt.is_finite() { dt.max(0.0) as f64 } else { 0.0 };
        // the change of root, at an even pace, once the matrix is there
        if self.glide.arrived() && self.blend != self.blend_to {
            let step = dt / BLEND_TIME;
            self.blend = if self.blend < self.blend_to { (self.blend + step).min(self.blend_to) } else { (self.blend - step).max(self.blend_to) };
            if self.blend == self.blend_to {
                if let Some(next) = self.blend_then.take() {
                    self.blend_to = next;
                }
            }
        }
        let settled = self.glide.arrived() && self.blend == self.blend_to;
        if self.play_when_there && settled {
            (self.play_when_there, self.playing) = (false, true);
        }
        if self.playing {
            let stage = (self.glide.to.stage + dt / STEP_TIME).min(2.0);
            self.glide.now(|s| s.stage = stage);
            if stage >= 2.0 || cholesky(self.glide.to.a()).l().is_none() {
                self.playing = false;
            }
        }
        if let Some((left, r)) = self.then {
            if settled && !self.playing && !self.play_when_there {
                let left = left - dt;
                if left <= 0.0 {
                    self.glide.to.r = r;
                    self.then = None;
                } else {
                    self.then = Some((left, r));
                }
            }
        }
    }

    /// The setting as drawn.
    fn shown(&self) -> Setting {
        self.glide.shown
    }

    /// How far from L to the symmetric root the drawn root is (0 to 1, eased).
    fn blend_shown(&self) -> f64 {
        ease(self.blend)
    }

    fn root(&self) -> Root {
        match self.blend_shown() {
            b if b <= 0.0 => Root::L,
            b if b >= 1.0 => Root::S,
            _ => Root::Between,
        }
    }

    /// The square root drawn, with all of L's steps: L times the turn the change of root
    /// has got to (none if A has no real L).
    fn root_matrix(&self) -> Option<DMat2> {
        let l = cholesky(self.shown().a()).l()?;
        Some(l * turn(self.blend_shown() * root_turn(l)))
    }

    /// What the picture of x shows: the root drawn, as far as L's steps have got.
    fn drawn(&self) -> Option<DMat2> {
        let s = self.shown();
        let l = cholesky(s.a()).l()?;
        Some(film(l, s.stage) * turn(self.blend_shown() * root_turn(l)))
    }

    fn settled(&self) -> bool {
        self.glide.arrived() && self.blend == self.blend_to && !self.playing && !self.play_when_there && self.then.is_none()
    }
}

// ----- texts -------------------------------------------------------------------------

/// A number in a matrix on the screen, as a formula.
fn cell(v: f64) -> String {
    format!("${}$", num(v, 2))
}

/// An angle in degrees, in words: "26.6° clockwise".
fn turn_words(theta: f64) -> String {
    let d = theta.to_degrees();
    if d.abs() < 0.05 {
        "0°".to_string()
    } else {
        format!("{:.1}° {}", d.abs(), if d < 0.0 { "clockwise" } else { "anticlockwise" })
    }
}

/// The recipe's three steps with this matrix's numbers, each with its colour; `short`
/// leaves out the steps' formulas and keeps the numbers.
fn recipe_lines(a: Sym, short: bool) -> [(String, Color); 3] {
    let c = cholesky(a);
    let n = |v: f64| num(v, 2);
    let first = |l11: f64| if short { format!(r"1.  $l_{{11}} = \sqrt{{{}}} = {}$", n(a.a11), n(l11)) } else { format!(r"1.  $l_{{11}} = \sqrt{{a_{{11}}}} = \sqrt{{{}}} = {}$", n(a.a11), n(l11)) };
    let second = |l11: f64, l21: f64| {
        if short {
            format!(r"2.  $l_{{21}} = {} / {} = {}$", n(a.a21), n(l11), n(l21))
        } else {
            format!(r"2.  $l_{{21}} = a_{{21}} / l_{{11}} = {} / {} = {}$", n(a.a21), n(l11), n(l21))
        }
    };
    let root = |l21: f64| if short { format!(r"\sqrt{{{} - {}}}", n(a.a22), n(l21 * l21)) } else { format!(r"\sqrt{{a_{{22}} - l_{{21}}^2}} = \sqrt{{{} - {}}}", n(a.a22), n(l21 * l21)) };
    match c {
        Chol::Definite { l11, l21, l22 } => [(first(l11), pal::TEXT), (second(l11, l21), pal::TEXT), (format!(r"3.  $l_{{22}} = {} = {}$", root(l21), n(l22)), pal::TEXT)],
        Chol::Flat { l11, l21 } => [(first(l11), pal::TEXT), (second(l11, l21), pal::TEXT), (format!(r"3.  $l_{{22}} = {} = \sqrt{{0}} = 0$", root(l21)), pal::YELLOW)],
        Chol::Fails3 { l11, l21, under } => {
            [(first(l11), pal::TEXT), (second(l11, l21), pal::TEXT), (format!(r"3.  $l_{{22}} = {} = \sqrt{{{}}}$: no real number", root(l21), n(under)), pal::RED)]
        }
        Chol::Fails1 { a11 } => [
            (format!(r"1.  $l_{{11}} = \sqrt{{{}}}$: no positive root", n(a11)), pal::RED),
            ("2.  nothing to divide by".to_string(), pal::DIM),
            ("3.  —".to_string(), pal::DIM),
        ],
    }
}

/// What the recipe found, in one or two lines (`short`: for a narrow panel), each
/// with its colour.
fn verdict(a: Sym, short: bool) -> Vec<(String, Color)> {
    let m = multiplier(a);
    match cholesky(a) {
        Chol::Definite { .. } if short => vec![("positive definite: this $L$, with a".to_string(), pal::TEXT), ("positive diagonal, is the only one".to_string(), pal::TEXT)],
        Chol::Definite { .. } => vec![("positive definite: this $L$ is the only one with a positive diagonal".to_string(), pal::TEXT)],
        Chol::Flat { .. } if short => vec![("semidefinite: $l_{22} = 0$, and $L$ squashes".to_string(), pal::YELLOW), ("the plane onto a line".to_string(), pal::YELLOW)],
        Chol::Flat { .. } => vec![("semidefinite: $l_{22} = 0$, and $L$ squashes the plane onto a line".to_string(), pal::YELLOW)],
        Chol::Fails3 { under, .. } => {
            let x = format!(r"$x^\top A x = {}$ for $x = ({}, 1)$", num(under, 2), num(-m, 2));
            if short {
                vec![("not positive definite:".to_string(), pal::RED), (x, pal::RED)]
            } else {
                vec![(format!("not positive definite: {x}"), pal::RED)]
            }
        }
        Chol::Fails1 { .. } => vec![("not positive definite: no real $L$".to_string(), pal::RED)],
    }
}

impl Cholesky {
    /// The title of the picture of x (`short`: for a narrow canvas), and its colour.
    fn x_title(&self, short: bool) -> (String, Color) {
        let s = self.shown();
        let c = cholesky(s.a());
        let Some(l) = c.l() else {
            return (if short { "no real $L$: no such $x$" } else { "no real $L$ for this $A$, so no such $x$" }.to_string(), pal::RED);
        };
        if s.stage < 2.0 && self.root() == Root::L {
            return if s.stage <= 0.0 {
                ("before the steps: $x = z$".to_string(), pal::TEXT)
            } else if s.stage <= 1.0 {
                let words = if short { "1 of 2: stretch".to_string() } else { format!("step 1 of 2: stretch, by ${}$ across and ${}$ up", num(l.x_axis.x, 2), num(l.y_axis.y, 2)) };
                (words, pal::TEXT)
            } else {
                let words = if short { "2 of 2: shear".to_string() } else { format!("step 2 of 2: shear, raising each point by ${}$ times its $x_1$", num(multiplier(s.a()), 2)) };
                (words, pal::TEXT)
            };
        }
        let flat = matches!(c, Chol::Flat { .. });
        let words = match (self.root(), short) {
            (Root::L, _) if flat => "$x = Lz$: every point on one line".to_string(),
            (Root::L, true) => "$x = Lz$: covariance $A$".to_string(),
            (Root::L, false) => r"$x = Lz$, with covariance $LL^\top = A$".to_string(),
            (Root::S, true) => "$x = Sz$: covariance $A$".to_string(),
            (Root::S, false) => r"$x = Sz$, with covariance $SS^\top = A$ too".to_string(),
            (Root::Between, true) => "$x = LQz$: covariance $A$".to_string(),
            (Root::Between, false) => format!(r"$x = LQz$, $Q$ a turn by {}: covariance $A$ still", turn_words(self.blend_shown() * root_turn(l))),
        };
        (words, pal::TEXT)
    }

    /// The name of the root drawn, and of its transpose.
    fn root_names(&self) -> (&'static str, &'static str) {
        match self.root() {
            Root::L => ("$L$", r"$L^\top$"),
            Root::S => ("$S$", r"$S^\top = S$"),
            Root::Between => ("$LQ$", r"$(LQ)^\top$"),
        }
    }
}

// ----- drawing -----------------------------------------------------------------------

/// The cloud's colour: a pale grey-blue, no colour of the key's.
fn cloud_colour() -> Color {
    pal::TEXT.mix(pal::BLUE, 0.3)
}

/// The points round the unit circle, and the marks on it (every 30°).
const RING: usize = 96;
const MARKS: usize = 12;

/// The point of the unit circle at a share `t` of the way round.
fn on_circle(t: f64) -> DVec2 {
    DVec2::from_angle(t * std::f64::consts::TAU)
}

/// The cloud of x, whose points come and go at the picture's edges and with L itself: a
/// batch of its own (different from the usual one by a hair of its outline), so that
/// the rest can still be compared point for point from one frame to the next.
fn cloud_apart<'a>(f: &'a mut Frame) -> &'a mut crate::gfx::draw::PointBatch {
    f.dl.points_with(PointKind::Disc, PointStyle { outline_px: 1e-4, ..Default::default() })
}

impl Cholesky {
    /// Everything in the two pictures, and the names written beside it.
    fn picture(&mut self, f: &mut Frame, lay: &Layout, room: &mut Option<Room>) {
        let s = self.shown();
        let a = s.a();
        let c = cholesky(a);
        let b = self.drawn();
        let (zm, xm) = (lay.map(0), lay.map(1));
        let unit_px = lay.unit;

        for (k, map) in [zm, xm].into_iter().enumerate() {
            backdrop(f, lay, k, map, room);
        }

        // the cloud: z on the left, x = Bz on the right
        for z in &self.zs {
            if zm.shows(*z) {
                f.dot(zm.to(*z), 4.0, cloud_colour().a(0.55));
            }
        }
        if let Some(b) = b {
            // (one beyond the frame is kept at its edge, unseen: so the batch keeps its
            // number of points, and is compared point for point as the cloud moves)
            for z in &self.zs {
                let x = b * *z;
                let colour = if xm.shows(x) { cloud_colour().a(0.55) } else { Color::CLEAR };
                cloud_apart(f).push(xm.to(x.clamp(DVec2::splat(-HALF), DVec2::splat(HALF))).extend(0.0), 4.0, colour);
            }
        }

        // the box: x₁ within ±σ₁, x₂ within ±σ₂, which the ellipse touches
        let corner = |sx: f64, sy: f64| xm.to(DVec2::new(sx * s.s1, sy * s.s2));
        let box_corners = [corner(1.0, 1.0), corner(-1.0, 1.0), corner(-1.0, -1.0), corner(1.0, -1.0)];
        for i in 0..4 {
            f.dashed(box_corners[i], box_corners[(i + 1) % 4], 1.2, pal::DIM.a(0.8));
            if let Some(room) = room {
                room.line(f, box_corners[i].extend(0.0), box_corners[(i + 1) % 4].extend(0.0));
            }
        }

        // the circle and its image, with the marks and the thicker quarter
        ring(f, zm, DMat2::IDENTITY, room);
        if let Some(b) = b {
            ring(f, xm, b, room);
        }

        // the columns: of I on the left, of the root drawn on the right
        let arrows = |f: &mut Frame, map: Map, m: DMat2, room: &mut Option<Room>| {
            for (col, colour) in [(m.x_axis, pal::PINK), (m.y_axis, pal::GREEN)] {
                let (from, to) = (map.to(DVec2::ZERO), map.to(col));
                f.arrow(from, to, 3.0, colour);
                if let Some(room) = room {
                    room.line(f, from.extend(0.0), to.extend(0.0));
                }
            }
        };
        arrows(f, zm, DMat2::IDENTITY, room);
        match (b, c) {
            (Some(b), _) => arrows(f, xm, b, room),
            // (no L: the first column still comes out of steps 1 and 2)
            (None, Chol::Fails3 { l11, l21, .. }) => {
                let (from, to) = (xm.to(DVec2::ZERO), xm.to(DVec2::new(l11, l21)));
                f.arrow(from, to, 3.0, pal::PINK);
                if let Some(room) = room {
                    room.line(f, from.extend(0.0), to.extend(0.0));
                }
            }
            _ => {}
        }

        // x = Bz, where the white z goes
        let x = b.map(|b| b * s.z);
        if let Some(x) = x.filter(|x| xm.shows(*x)) {
            f.dl.points_with(PointKind::Disc, PointStyle { outline_px: 1.25 + 2e-4, ..Default::default() }).push(xm.to(x).extend(0.0), 10.0, pal::WHITE);
        }

        // the arrow from one picture to the other
        let (from, to) = (lay.stage(lay.between.0), lay.stage(lay.between.1));
        let colour = if b.is_some() { pal::TEXT.a(0.8) } else { pal::RED.a(0.8) };
        f.arrow(from, to, 1.6, colour);

        let Some(room) = room else { return };
        if !lay.words || unit_px < 22.0 {
            return;
        }
        // (no name on the white dots, which are drawn over everything)
        let mut dots = vec![zm.to(s.z), xm.to(DVec2::new(s.s1, s.r * s.s2)), xm.to(DVec2::new(s.s1, s.s2))];
        dots.extend(x.filter(|x| xm.shows(*x)).map(|x| xm.to(x)));
        for dot in dots {
            if let Some(p) = on_screen(f, dot.extend(0.0)) {
                room.boxed(p - Vec2::splat(8.0), p + Vec2::splat(8.0));
            }
        }
        let name = |f: &mut Frame, room: &mut Room, tag: (&'static str, u32), at: Vec2, text: &str, size: f32, colour: Color, towards: Vec2, gap: f32| {
            room.label(f, tag, at.extend(0.0), text, size, colour, towards, gap, false);
        };
        // (a direction in the plane, as a direction on the screen: y runs down there)
        let screen = |v: DVec2| {
            let v = Vec2::new(v.x as f32, -(v.y as f32)).normalize_or_zero();
            if v.is_finite() {
                v
            } else {
                Vec2::ZERO
            }
        };
        // what the arrow between the pictures stands for
        let (root_name, _) = self.root_names();
        let mid = (lay.between.0 + lay.between.1) * 0.5;
        let off = if lay.narrow { Vec2::new(-10.0, 0.0) } else { Vec2::new(0.0, -14.0) };
        let align = if lay.narrow { Align2::RIGHT_CENTER } else { Align2::CENTER_BOTTOM };
        let between_name = if b.is_some() { root_name } else { "no $L$" };
        f.text_off(lay.stage(mid), off, between_name, 15.0, colour, align);

        // the left: the columns of I, the circle, z
        name(f, room, ("e1", 0), zm.to(DVec2::X), "$(1, 0)$", 13.0, pal::PINK, Vec2::new(1.0, 0.7), 9.0);
        name(f, room, ("e2", 0), zm.to(DVec2::Y), "$(0, 1)$", 13.0, pal::GREEN, Vec2::new(0.7, -1.0), 9.0);
        let at = on_circle(0.375);
        name(f, room, ("circle", 0), zm.to(at), "unit circle", 12.5, pal::YELLOW, screen(at), 9.0);
        name(f, room, ("z", 0), zm.to(s.z), "$z$", 15.0, pal::WHITE, screen(s.z), 11.0);

        // the right: each name outside the ellipse, along its outward normal there
        // (for a point B u of it, u on the circle, that is B⁻ᵀu: the gradient of xᵀA⁻¹x)
        let outward = |b: DMat2, u: DVec2| {
            let n = if b.determinant().abs() > 1e-9 { b.inverse().transpose() * u } else { b * u };
            screen(n)
        };
        let root = self.root();
        let full = s.stage >= 2.0 || root != Root::L;
        if let Some(b) = b {
            let (n1, n2) = match root {
                Root::L => (r"$(l_{11}, l_{21})$", r"$(0, l_{22})$"),
                Root::S => (r"$(s_{11}, s_{21})$", r"$(s_{12}, s_{22})$"),
                Root::Between => (r"$LQ(1, 0)$", r"$LQ(0, 1)$"),
            };
            if full {
                name(f, room, ("c1", 0), xm.to(b.x_axis), n1, 13.0, pal::PINK, outward(b, DVec2::X), 10.0);
                if b.y_axis.length() * unit_px as f64 >= 4.0 {
                    name(f, room, ("c2", 0), xm.to(b.y_axis), n2, 13.0, pal::GREEN, outward(b, DVec2::Y), 10.0);
                } else {
                    name(f, room, ("c2", 0), xm.to(DVec2::ZERO), r"$l_{22} = 0$", 13.0, pal::GREEN, Vec2::new(-1.0, -0.6), 10.0);
                }
            }
            // (at its lowest point, where it touches the box: the same for every root; a
            // semidefinite A has no inverse, and the circle is pressed flat. Not in a small
            // picture, where it would only crowd the others: the title says what it is)
            let roomy = unit_px >= 60.0;
            if roomy {
                let text = if matches!(c, Chol::Flat { .. }) { "the circle, squashed flat" } else { r"$x^\top A^{-1} x = 1$" };
                name(f, room, ("ellipse", 0), xm.to(DVec2::new(-s.r * s.s1, -s.s2)), text, 13.0, pal::YELLOW, Vec2::new(-0.3, 1.0), 9.0);
            }
            if let Some(x) = x.filter(|x| xm.shows(*x)) {
                let text = match root {
                    Root::L if full && roomy => "$x = Lz$",
                    Root::S if roomy => "$x = Sz$",
                    _ => "$x$",
                };
                name(f, room, ("x", 0), xm.to(x), text, 14.0, pal::WHITE, screen(x), 11.0);
            }
        } else if let Chol::Fails3 { under, l11, l21 } = c {
            name(f, room, ("c1", 0), xm.to(DVec2::new(l11, l21)), r"$(l_{11}, l_{21})$", 13.0, pal::PINK, Vec2::new(1.0, 0.0), 11.0);
            let text = format!(r"$l_{{22}} = \sqrt{{{}}}$: not a real number", num(under, 2));
            name(f, room, ("fail", 0), xm.to(DVec2::new(0.0, -1.4)), &text, 13.0, pal::RED, Vec2::ZERO, 0.0);
        }
        name(f, room, ("s1", 0), corner(1.0, -1.0), r"$\sigma_1$", 13.0, pal::DIM, Vec2::new(0.4, 1.0), 6.0);
        name(f, room, ("s2", 0), corner(-1.0, 1.0), r"$\sigma_2$", 13.0, pal::DIM, Vec2::new(-1.0, -0.4), 6.0);
    }

    /// The handles: where the ellipse touches the box's right side (r, and σ₁), the
    /// box's corner (σ₁ and σ₂), and the white point z.
    fn handles(&mut self, f: &mut Frame, lay: &Layout) {
        let s = self.shown();
        let (zm, xm) = (lay.map(0), lay.map(1));
        // (the touching point first: where it lies on the corner, at r = ±1, it is the
        // one a drag takes, so that the two can be parted)
        let mut w = xm.to(DVec2::new(s.s1, s.r * s.s2));
        if f.drag2(1, &mut w, pal::WHITE) {
            let p = xm.back(w);
            if p.is_finite() {
                let (s1, r) = (p.x.clamp(S_MIN, S_MAX), (p.y / s.s2).clamp(-R_MAX, R_MAX));
                self.by_hand(|g| (g.s1, g.r) = (s1, r));
            }
        }
        let mut w = xm.to(DVec2::new(s.s1, s.s2));
        if f.drag2(2, &mut w, pal::WHITE) {
            let p = xm.back(w);
            if p.is_finite() {
                let (s1, s2) = (p.x.clamp(S_MIN, S_MAX), p.y.clamp(S_MIN, S_MAX));
                self.by_hand(|g| (g.s1, g.s2) = (s1, s2));
            }
        }
        let mut w = zm.to(s.z);
        if f.drag2(3, &mut w, pal::WHITE) {
            let p = zm.back(w);
            if p.is_finite() {
                let z = p.clamp(DVec2::splat(-Z_MAX), DVec2::splat(Z_MAX));
                // (moving z changes nothing else: the steps go on as they were)
                self.glide.now(|g| g.z = z);
            }
        }
    }

    /// The numbers under the pictures: the buttons, A = BBᵀ, the recipe and what it found.
    fn hud(&self, f: &mut Frame, h: &Hud) {
        hud_panel(f, h.min, h.max);
        let s = self.shown();
        let a = s.a();
        let c = cholesky(a);
        // the buttons
        let busy = self.playing || self.play_when_there;
        for (k, (lo, hi)) in h.buttons.iter().enumerate() {
            let (on, live) = match k {
                0 => (self.blend_to == 0.0, true),
                1 => (self.blend_to == 1.0, true),
                _ => (busy, c.l().is_some()),
            };
            hud_rect(f, *lo, *hi, if on { Color::hex(0x24345A) } else { Color::hex(0x161D30) });
            hud_outline(f, *lo, *hi, 1.0, if on { pal::LINK } else { pal::DIM.a(0.7) });
            let label = if k == 2 && busy { PAUSE } else { h.labels[k] };
            f.text_px((*lo + *hi) * 0.5, label, 12.5, if live { pal::TEXT } else { pal::DIM }, Align2::CENTER_CENTER);
        }

        // A = BBᵀ, B's columns (and Bᵀ's rows) in the colours of the arrows
        let t = h.text;
        let cw = h.cell;
        let msize = matrix_size(2, 2, cw, t);
        let mid = h.mats.y + msize.y * 0.5;
        let mut x = h.mats.x;
        f.text_px(Vec2::new(x, mid), "$A =$", t, pal::TEXT, Align2::LEFT_CENTER);
        x += text_width("$A =$", t) + 8.0;
        let am = a.mat();
        let rows = |m: DMat2, colour: &dyn Fn(usize, usize) -> Color| -> Vec<Vec<(String, Color)>> { (0..2).map(|i| (0..2).map(|j| (cell(m.col(j)[i]), colour(i, j))).collect()).collect() };
        hud_matrix(f, Vec2::new(x, h.mats.y), &rows(am, &|_, _| pal::TEXT), cw, t, pal::DIM);
        x += msize.x + 8.0;
        let names_y = h.mats.y + msize.y + 2.0;
        match self.root_matrix() {
            Some(b) => {
                f.text_px(Vec2::new(x, mid), "$=$", t, pal::TEXT, Align2::LEFT_CENTER);
                x += text_width("$=$", t) + 8.0;
                let colours = [pal::PINK, pal::GREEN];
                let (n1, n2) = self.root_names();
                hud_matrix(f, Vec2::new(x, h.mats.y), &rows(b, &|_, j| colours[j]), cw, t, pal::DIM);
                f.text_px(Vec2::new(x + msize.x * 0.5, names_y), n1, 11.5, pal::DIM, Align2::CENTER_TOP);
                x += msize.x + 6.0;
                hud_matrix(f, Vec2::new(x, h.mats.y), &rows(b.transpose(), &|i, _| colours[i]), cw, t, pal::DIM);
                f.text_px(Vec2::new(x + msize.x * 0.5, names_y), n2, 11.5, pal::DIM, Align2::CENTER_TOP);
            }
            None => {
                f.text_px(Vec2::new(x, mid), r"$\ne LL^\top$ for any real $L$", t, pal::RED, Align2::LEFT_CENTER);
            }
        }

        // the line under the matrices: elimination's numbers, or the turn from L
        if let Some(at) = h.note {
            let text = match (self.root(), c.l()) {
                (Root::L, Some(l)) => format!(
                    r"elimination: multiplier $a_{{21}}/a_{{11}} = {}$, pivots ${} = l_{{11}}^2$ and ${} = l_{{22}}^2$",
                    num(multiplier(a), 2),
                    num(a.a11, 2),
                    num(l.y_axis.y * l.y_axis.y, 2)
                ),
                (_, Some(l)) => format!(r"$S = LQ$, $Q$ a turn by {}; on the way, $L$ times part of it", turn_words(root_turn(l))),
                (_, None) => format!(r"elimination: multiplier ${}$, pivots ${}$ and ${}$: one is not positive", num(multiplier(a), 2), num(a.a11, 2), num(c.under().unwrap_or(0.0), 2)),
            };
            f.text_px(at, text, t - 1.0, pal::DIM, Align2::LEFT_TOP);
        }

        // the recipe, and what it found
        let mut y = h.recipe.y;
        if h.two {
            f.text_px(Vec2::new(h.recipe.x, y), "the recipe, with these numbers:", t - 1.0, pal::DIM, Align2::LEFT_TOP);
            y += h.line;
        }
        for (text, colour) in recipe_lines(a, !h.two) {
            f.text_px(Vec2::new(h.recipe.x, y), text, t, colour, Align2::LEFT_TOP);
            y += h.line;
        }
        for (text, colour) in h.verdict(a) {
            f.text_px(Vec2::new(h.recipe.x, y), text, t - 0.5, colour, Align2::LEFT_TOP);
            y += h.line - if h.two { 0.0 } else { 1.0 };
        }
        if h.two {
            let line = match self.root_matrix() {
                Some(b) => {
                    let name = match self.root() {
                        Root::L => "L",
                        Root::S => "S",
                        Root::Between => "LQ",
                    };
                    format!(r"$z = {} \;\mapsto\; x = {name}z = {}$", tuple(&[s.z.x, s.z.y], 2), tuple(&[(b * s.z).x, (b * s.z).y], 2))
                }
                None => format!(r"$z = {}$: no real $L$ to take it anywhere", tuple(&[s.z.x, s.z.y], 2)),
            };
            f.text_px(Vec2::new(h.recipe.x, y), line, t - 0.5, pal::WHITE, Align2::LEFT_TOP);
        }
    }

    /// Clicks on the numbers' buttons.
    fn hud_input(&mut self, f: &Frame, h: &Hud) {
        let Some(p) = f.ptr.pos.map(|p| p / f.px) else { return };
        if !f.clicked_canvas() {
            return;
        }
        if inside_rect(p, h.buttons[0]) {
            self.choose(false);
        } else if inside_rect(p, h.buttons[1]) {
            self.choose(true);
        } else if inside_rect(p, h.buttons[2]) {
            self.play_steps();
        }
    }
}

/// A picture's frame, its axes with their names, and numbers along them.
fn backdrop(f: &mut Frame, lay: &Layout, k: usize, map: Map, room: &mut Option<Room>) {
    let h = HALF;
    let c = [DVec2::new(-h, -h), DVec2::new(h, -h), DVec2::new(h, h), DVec2::new(-h, h)].map(|p| map.to(p));
    f.dl.tris_alpha.quad(c[0].extend(0.0), c[1].extend(0.0), c[2].extend(0.0), c[3].extend(0.0), Color::hex(0x080B14).a(0.55));
    f.dl.lines.closed(&c.map(|p| p.extend(0.0)), 1.0, pal::DIM.a(0.45));
    let axis = pal::DIM.a(0.4);
    f.line(map.to(DVec2::new(-h, 0.0)), map.to(DVec2::new(h, 0.0)), 1.0, axis);
    f.line(map.to(DVec2::new(0.0, -h)), map.to(DVec2::new(0.0, h)), 1.0, axis);
    let wpp = 1.0 / lay.ppu;
    for v in [-2.0, -1.0, 1.0, 2.0] {
        let (p, q) = (map.to(DVec2::new(v, 0.0)), map.to(DVec2::new(0.0, v)));
        f.line(p - Vec2::new(0.0, 3.0 * wpp), p + Vec2::new(0.0, 3.0 * wpp), 1.0, axis);
        f.line(q - Vec2::new(3.0 * wpp, 0.0), q + Vec2::new(3.0 * wpp, 0.0), 1.0, axis);
    }
    if let Some(room) = room {
        for (a, b) in [(c[0], c[1]), (c[1], c[2]), (c[2], c[3]), (c[3], c[0])] {
            room.line(f, a.extend(0.0), b.extend(0.0));
        }
        room.line(f, map.to(DVec2::new(-h, 0.0)).extend(0.0), map.to(DVec2::new(h, 0.0)).extend(0.0));
        room.line(f, map.to(DVec2::new(0.0, -h)).extend(0.0), map.to(DVec2::new(0.0, h)).extend(0.0));
    }
    if !lay.words || lay.unit < 22.0 {
        return;
    }
    let names = [["$z_1$", "$z_2$"], ["$x_1$", "$x_2$"]][k];
    let ends = [map.to(DVec2::new(h, 0.0)), map.to(DVec2::new(0.0, h))];
    f.text_off(ends[0], Vec2::new(-4.0, -4.0), names[0], 13.0, pal::DIM, Align2::RIGHT_BOTTOM);
    f.text_off(ends[1], Vec2::new(6.0, 4.0), names[1], 13.0, pal::DIM, Align2::LEFT_TOP);
    // (numbers at ±2 only: at ±1 they would lie on the box's sides whenever σ = 1)
    for v in [-2.0, 2.0] {
        let text = num(v, 0);
        f.text_off(map.to(DVec2::new(v, 0.0)), Vec2::new(0.0, 5.0), text.as_str(), 10.5, pal::DIM.a(0.8), Align2::CENTER_TOP);
        f.text_off(map.to(DVec2::new(0.0, v)), Vec2::new(-6.0, 0.0), text, 10.5, pal::DIM.a(0.8), Align2::RIGHT_CENTER);
    }
    if let Some(room) = room {
        let box_at = |f: &Frame, p: Vec2, lo: Vec2, hi: Vec2| on_screen(f, p.extend(0.0)).map(|s| (s + lo, s + hi));
        for v in [-2.0, 2.0] {
            if let Some((lo, hi)) = box_at(f, map.to(DVec2::new(v, 0.0)), Vec2::new(-9.0, 4.0), Vec2::new(9.0, 17.0)) {
                room.boxed(lo, hi);
            }
            if let Some((lo, hi)) = box_at(f, map.to(DVec2::new(0.0, v)), Vec2::new(-24.0, -7.0), Vec2::new(-5.0, 7.0)) {
                room.boxed(lo, hi);
            }
        }
        for (end, lo, hi) in [(ends[0], Vec2::new(-24.0, -22.0), Vec2::new(-2.0, -2.0)), (ends[1], Vec2::new(4.0, 2.0), Vec2::new(26.0, 22.0))] {
            if let Some((lo, hi)) = box_at(f, end, lo, hi) {
                room.boxed(lo, hi);
            }
        }
    }
}

/// The unit circle carried by `m` into a picture: the curve, the marks every 30°, and
/// the quarter from (1, 0) to (0, 1) drawn thicker, so that where each point goes shows.
fn ring(f: &mut Frame, map: Map, m: DMat2, room: &mut Option<Room>) {
    let pts: Vec<Vec3> = (0..RING).map(|i| map.to(m * on_circle(i as f64 / RING as f64)).extend(0.0)).collect();
    f.dl.lines.closed(&pts, 2.0, pal::YELLOW.a(0.9));
    f.path((0..=24).map(|i| map.to(m * on_circle(i as f64 / 96.0))), 4.5, pal::YELLOW);
    for i in 0..MARKS {
        f.dot(map.to(m * on_circle(i as f64 / MARKS as f64)), 6.5, pal::YELLOW);
    }
    if let Some(room) = room {
        for i in 0..24 {
            room.line(f, pts[i * 4], pts[(i * 4 + 4) % RING]);
        }
    }
}

impl Scene for Cholesky {
    fn view(&self) -> View {
        STAGE_VIEW
    }

    fn settings(&self, s: &mut ViewSettings) {
        s.grid = false;
        s.bloom = s.bloom.min(0.2);
    }

    fn animated(&self) -> bool {
        !self.settled()
    }

    fn pointer_hint(&self) -> Option<&'static str> {
        Some("white dots: drag $z$, the corner of the box, or where the ellipse touches its side    drag elsewhere: move    scroll: zoom")
    }

    fn ui(&mut self, ui: &mut egui::Ui, _: &Gfx) {
        ui::note(ui, r"A symmetric matrix $A$ is positive definite when $x^\top A x > 0$ for every $x \ne 0$. Covariance matrices are, unless some sum of multiples of the numbers never varies (one always twice another, say). Every such $A$ is");
        ui::equation(ui, r"A = LL^\top, \quad L = \begin{bmatrix} l_{11} & 0 \\ l_{21} & l_{22} \end{bmatrix}");
        ui::note(ui, r"for exactly one $L$ with $l_{11}, l_{22} > 0$, found in three steps (written out under the pictures with this $A$'s numbers). What it is for: pairs $z$ of independent standard normal numbers (left) become $x = Lz$ (right), whose covariance is $A$. $L$ carries the unit circle to the ellipse $x^\top A^{-1} x = 1$. And the recipe meets a number that is not positive under a square root exactly when $A$ is not positive definite.");

        ui::section(ui, "Things to try");
        for (k, trial) in TRIALS.iter().enumerate() {
            if ui::experiment(ui, trial.label, trial.look_for) {
                self.set_up(k);
            }
        }

        ui::section(ui, "The matrix $A$");
        ui::equation(ui, r"A = \begin{bmatrix} \sigma_1^2 & r\sigma_1\sigma_2 \\ r\sigma_1\sigma_2 & \sigma_2^2 \end{bmatrix}");
        let to = self.glide.to;
        let (mut s1, mut s2, mut r) = (to.s1 as f32, to.s2 as f32, to.r as f32);
        if ui::slider(ui, r"$\sigma_1$", &mut s1, S_MIN as f32..=S_MAX as f32) {
            self.by_hand(|g| g.s1 = (s1 as f64).clamp(S_MIN, S_MAX));
        }
        if ui::slider(ui, r"$\sigma_2$", &mut s2, S_MIN as f32..=S_MAX as f32) {
            self.by_hand(|g| g.s2 = (s2 as f64).clamp(S_MIN, S_MAX));
        }
        if ui::slider(ui, "$r$", &mut r, -R_MAX as f32..=R_MAX as f32) {
            self.by_hand(|g| g.r = (r as f64).clamp(-R_MAX, R_MAX));
        }
        ui::note(ui, r"$\sigma_1^2$ and $\sigma_2^2$ are the variances of $x_1$ and $x_2$, and $r$ is their correlation, which lies between $-1$ and $1$. Past those ends $A$ is still symmetric, but not positive definite. On the picture, drag the corner of the dashed box ($\sigma_1$, $\sigma_2$), or the point where the ellipse touches its right side, at height $r\sigma_2$.");

        ui::section(ui, "Which square root");
        let mut k = if self.blend_to >= 0.5 { 1 } else { 0 };
        if ui::choice(ui, &["$L$, lower triangular", "$S$, symmetric"], &mut k) {
            self.choose(k == 1);
        }
        ui::note(ui, r"Both have $BB^\top = A$: the same ellipse, and the same covariance for $x = Bz$, but each point other than $0$ lands somewhere else on it (unless $A$ is diagonal, when $S = L$). $S = LQ$ for a turn $Q$, and the change from one to the other passes through $L$ times smaller turns, every one of them a square root of $A$ too.");

        ui::section(ui, "The steps of $L$");
        ui.horizontal(|ui| {
            let busy = self.playing || self.play_when_there;
            if ui.button(if busy { "❚❚ pause" } else { "▶ play the steps" }).clicked() {
                self.play_steps();
            }
        });
        let mut stage = self.glide.to.stage as f32;
        if ui::slider(ui, "step", &mut stage, 0.0..=2.0) {
            (self.playing, self.play_when_there, self.then) = (false, false, None);
            (self.blend_to, self.blend_then) = (0.0, None);
            self.glide.now(|g| g.stage = (stage as f64).clamp(0.0, 2.0));
        }
        ui::note(ui, r"$L$ is a stretch along the axes, by $l_{11}$ across and $l_{22}$ up, followed by a shear that raises each point by $l_{21}/l_{11}$ times its $x_1$. 0 is the plane as it was, 1 after the stretch, 2 after the shear: all of $L$.");

        ui::section(ui, "Reading the picture");
        ui::key(
            ui,
            &[
                (cloud_colour(), r"pale dots: 400 pairs $z$ of independent standard normal numbers (left), and the same pairs as $x$ (right); those beyond the frame are not drawn"),
                (pal::YELLOW, r"yellow: the unit circle, and where the root drawn takes it: the ellipse $x^\top A^{-1} x = 1$. The marks every 30° and the thicker quarter, from $(1, 0)$ to $(0, 1)$, show where each point goes"),
                (pal::PINK, r"pink: $(1, 0)$, and where it goes: the first column of the root, $(l_{11}, l_{21})$ for $L$, at the ellipse's rightmost point"),
                (pal::GREEN, r"green: $(0, 1)$ and the second column: $(0, l_{22})$ for $L$, straight up, where the ellipse crosses the $x_2$ axis"),
                (pal::DIM, r"dashed box: $x_1$ from $-\sigma_1$ to $\sigma_1$, $x_2$ from $-\sigma_2$ to $\sigma_2$; the ellipse touches all four sides"),
                (pal::WHITE, r"white dots, to drag: $z$ (its $x$ follows on the right), the box's corner, and where the ellipse touches the box's right side"),
                (pal::RED, "red: where the recipe breaks"),
            ],
        );

        ui::section(ui, "Words");
        ui::term(ui, "Positive definite:", r"symmetric, with $x^\top A x > 0$ for every $x \ne 0$ (a bowl: scene quadform), which for $2 \times 2$ means $a_{11} > 0$ and $\det A > 0$. Exactly then does the recipe find a real $L$ with a positive diagonal, and that $L$ is the only one.");
        ui::term(ui, "Semidefinite, not definite:", r"$x^\top A x \ge 0$ for every $x$, and $= 0$ for some $x \ne 0$. With $a_{11} > 0$, as here, the recipe meets $\sqrt{0}$ in step 3: $L$ has a 0 on its diagonal and squashes the plane onto a line. (With $a_{11} = 0$ it stops at a division by 0, and an $L$ need not be the only one.)");
        ui::term(ui, "Covariance matrix:", r"for two numbers with mean $0$, the mean values of $x_1^2$, $x_1 x_2$ and $x_2^2$, over all the pairs that could be drawn. For $x = Lz$ it is $L$ times $z$'s, which is $I$, times $L^\top$: $LL^\top = A$.");
        ui::term(ui, "Pivots, multiplier:", r"elimination's numbers (scene elim). Row 2 loses $a_{21}/a_{11}$ times row 1, leaving the pivots $a_{11}$ and $a_{22} - a_{21}^2/a_{11}$: here $l_{11}^2$ and $l_{22}^2$, with the multiplier $l_{21}/l_{11}$. In matrices, $A = L_1 D L_1^\top$, with $L_1$ having $1$ all down its diagonal and the multiplier under it, and $D$ the pivots: so $L = L_1 \sqrt{D}$. A symmetric $A$ needs only $L$, not $L$ and $U$: for an $n \times n$ matrix about $n^3/3$ arithmetic operations, against $2n^3/3$.");
        ui::term(ui, "Square root of $A$:", r"here, any $B$ with $BB^\top = A$. For a positive definite $A$ each is $L$ times an orthogonal $Q$, $B = LQ$, since then $Q = L^{-1}B$ has $QQ^\top = I$. $S$ is the symmetric one with no negative eigenvalue: $A$'s eigenvectors, and the roots of its eigenvalues (scene spectral).");
        ui::term(ui, "Orthogonal:", r"$Q^\top Q = I$: a turn or a mirror. It keeps lengths, so a round cloud of standard normal pairs stays one.");
        ui::term(ui, r"Solving $Ax = b$:", r"with $L$ in hand, solve $Ly = b$ from the top down, then $L^\top x = y$ from the bottom up: two triangular systems.");

        ui::section(ui, "Numbers");
        let s = self.shown();
        let a = s.a();
        let c = cholesky(a);
        ui::readout(ui, "$A$", format!("${}$", bmatrix(&[[a.a11, a.a21], [a.a21, a.a22]], 2)));
        match c.l() {
            Some(l) => ui::readout(ui, "$L$", format!("${}$", bmatrix(&[[l.x_axis.x, 0.0], [l.x_axis.y, l.y_axis.y]], 2))),
            None => ui::readout(ui, "$L$", "none that is real"),
        }
        let (l1, l2, _) = a.eigen();
        ui::readout(ui, r"pivots $a_{11}$, $a_{22} - a_{21}^2/a_{11}$", format!("${}, {}$", num(a.a11, 2), num(c.under().unwrap_or(0.0), 2)));
        ui::readout(ui, "eigenvalues of $A$", format!("${}, {}$", num(l1, 2), num(l2, 2)));
        match (sym_root(a), c.l()) {
            (Some(root), Some(l)) => {
                ui::readout(ui, "$S$", format!("${}$", bmatrix(&[[root.x_axis.x, root.y_axis.x], [root.x_axis.y, root.y_axis.y]], 2)));
                ui::readout(ui, "$Q$ with $S = LQ$", format!("turns {}", turn_words(root_turn(l))));
            }
            _ => ui::readout(ui, "$S$", "none that is real"),
        }
        match self.root_matrix() {
            Some(b) => {
                let cx = b * self.cz * b.transpose();
                ui::readout(ui, "the 400 points' covariance", format!("${}$", bmatrix(&[[cx.x_axis.x, cx.y_axis.x], [cx.x_axis.y, cx.y_axis.y]], 2)));
                ui::note(ui, r"Near $A$ but not equal to it: a sample of 400 is not the whole population, and the $z$'s own sample covariance is not quite $I$.");
                ui::readout(ui, "$x$, from the white $z$", format!("${}$", tuple(&[(b * s.z).x, (b * s.z).y], 2)));
            }
            None => ui::readout(ui, "$x$, from the white $z$", "none"),
        }
        ui::see_also(
            ui,
            &[
                ("elim", "elimination, whose pivots these are"),
                ("spectral", "the symmetric square root, along the eigenvectors"),
                ("gauss2", "correlated normal numbers, and conditioning"),
                ("quadform", r"positive definite: $x^\top A x$ as a bowl"),
            ],
        );
    }

    fn frame(&mut self, f: &mut Frame) {
        let view = f.view_size();
        let lay = Layout::of(view, f.top_inset);
        // (a canvas squeezed to a sliver is one the controls have opened over, on a
        // phone: the pictures alone, without names or numbers)
        let covered = view.x < 160.0;
        let hud = lay.hud.filter(|_| !covered);
        if let Some(h) = &hud {
            if f.ptr.pos.is_some_and(|p| inside_rect(p / f.px, (h.min, h.max))) {
                // the numbers are for their own buttons: no moving the view from there,
                // and two quick clicks are two clicks
                f.handles.capture();
                f.handles.claim_clicks();
            }
            self.hud_input(f, h);
        }
        self.handles(f, &lay);
        self.advance(f.dt, f.real_dt);

        let mut room = (!covered && lay.words).then(|| Room::new(f));
        if let (Some(room), Some(h)) = (&mut room, &hud) {
            room.wall(h.min - Vec2::splat(4.0), h.max + Vec2::splat(4.0));
        }
        // the titles
        if lay.words && !covered {
            let size = if lay.unit >= 70.0 { 13.0 } else { 12.0 };
            // (the longer title where it fits: up to the next title, or the arrow, or the edge)
            let across = lay.title_room();
            let fits = |k: usize, text: &str| text_width(text, size) <= across[k];
            let ztitle = ["$z$: 400 pairs of independent standard normal numbers", "$z$: independent standard normals"].into_iter().find(|t| fits(0, t)).unwrap_or("$z$");
            let xtitle = Some(self.x_title(false)).filter(|t| fits(1, &t.0)).unwrap_or_else(|| self.x_title(true));
            for (k, (text, colour)) in [(ztitle.to_string(), pal::TEXT), xtitle].into_iter().enumerate() {
                let corner = lay.stage(lay.panes[k].0);
                let Some(at) = on_screen(f, corner.extend(0.0)) else { continue };
                let at = at + Vec2::new(2.0, -6.0);
                f.caption_px(at, text.as_str(), size, colour, Align2::LEFT_BOTTOM);
                if let Some(room) = &mut room {
                    room.wall(at - Vec2::new(4.0, size + 6.0), at + Vec2::new(text_width(&text, size) + 8.0, 4.0));
                }
            }
        }
        self.picture(f, &lay, &mut room);
        if let Some(h) = &hud {
            self.hud(f, h);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn close(m: DMat2, n: DMat2, tol: f64) -> bool {
        (m - n).to_cols_array().iter().all(|v| v.abs() <= tol)
    }

    fn sym(m: DMat2) -> Sym {
        Sym { a11: m.x_axis.x, a21: m.x_axis.y, a22: m.y_axis.y }
    }

    /// Random symmetric matrices of every kind: positive definite (made as L Lᵀ, and
    /// with random eigenvalues on random axes), indefinite, negative, with a₁₁ ≤ 0.
    fn matrices(seed: u64, n: usize) -> Vec<Sym> {
        let mut rng = Rng::new(seed);
        let mut out = Vec::new();
        for _ in 0..n {
            let r = |rng: &mut Rng, lo: f32, hi: f32| rng.range(lo, hi) as f64;
            let l = lower(r(&mut rng, 0.05, 3.0), r(&mut rng, -3.0, 3.0), r(&mut rng, 0.05, 3.0));
            out.push(sym(l * l.transpose()));
            let (l1, l2, t) = (r(&mut rng, -2.0, 3.0), r(&mut rng, -2.0, 3.0), r(&mut rng, 0.0, 6.3));
            let q = turn(t);
            out.push(sym(q * DMat2::from_diagonal(DVec2::new(l1, l2)) * q.transpose()));
            out.push(Sym { a11: r(&mut rng, -2.0, 2.0), a21: r(&mut rng, -2.0, 2.0), a22: r(&mut rng, -2.0, 2.0) });
        }
        out
    }

    /// The recipe multiplies back to A, its L has a positive diagonal, and it is the
    /// only such L (the same as Rᵀ from the QR of the symmetric root, made another
    /// way); it finishes exactly for the positive definite matrices.
    #[test]
    fn the_recipe_multiplies_back_and_fails_exactly_when_it_should() {
        let mut definite = 0;
        for a in matrices(3, 400) {
            let (l1, l2, _) = a.eigen();
            let scale = a.size().max(1e-9);
            match cholesky(a) {
                Chol::Definite { l11, l21, l22 } => {
                    definite += 1;
                    assert!(l11 > 0.0 && l22 > 0.0, "{a:?}");
                    let l = lower(l11, l21, l22);
                    assert!(close(l * l.transpose(), a.mat(), 1e-12 * scale), "{a:?}");
                    assert!(l2 > 0.0 && a.a11 > 0.0 && a.det() > 0.0, "{a:?}: eigenvalues {l1}, {l2}");
                    // the only one: Gram–Schmidt on the columns of S gives S = QR with
                    // R's diagonal positive, and A = SᵀS = RᵀR, so Rᵀ is such an L too
                    let s = sym_root(a).expect("a root");
                    let (c1, c2) = (s.x_axis, s.y_axis);
                    let q1 = c1.normalize();
                    let r11 = c1.length();
                    let r12 = q1.dot(c2);
                    let r22 = (c2 - q1 * r12).length();
                    assert!(close(lower(r11, r12, r22), l, 1e-9 * scale.sqrt().max(1.0)), "{a:?}");
                }
                Chol::Flat { .. } => panic!("{a:?}: no random matrix is exactly semidefinite"),
                Chol::Fails3 { l11, l21, under } => {
                    assert!(under < 0.0 && l11 > 0.0 && (l11 * l21 - a.a21).abs() < 1e-12 * scale, "{a:?}");
                    assert!(l2 < 0.0 && a.det() < 0.0, "{a:?}: eigenvalues {l1}, {l2}");
                }
                Chol::Fails1 { a11 } => assert!(a11 <= 0.0 && l2 <= 0.0, "{a:?}"),
            }
        }
        assert!(definite > 400, "{definite}");
    }

    /// The number under the last root is xᵀAx for x = (−m, 1), m the multiplier; and
    /// xᵀAx is a₁₁(x₁ + m x₂)² + (that number) x₂²: elimination, as completing the square.
    #[test]
    fn the_last_number_under_a_root_is_a_value_of_the_form() {
        let mut rng = Rng::new(5);
        for a in matrices(4, 200).into_iter().filter(|a| a.a11 > 0.0) {
            let under = a.a22 - a.a21 * a.a21 / a.a11;
            let m = multiplier(a);
            assert!((a.form(DVec2::new(-m, 1.0)) - under).abs() < 1e-10 * (1.0 + m * m) * a.size(), "{a:?}");
            for _ in 0..5 {
                let x = DVec2::new(rng.range(-2.0, 2.0) as f64, rng.range(-2.0, 2.0) as f64);
                let squares = a.a11 * (x.x + m * x.y).powi(2) + under * x.y * x.y;
                assert!((a.form(x) - squares).abs() < 1e-9 * (1.0 + m * m) * a.size(), "{a:?} at {x:?}");
            }
            if let Some(u) = cholesky(a).under() {
                assert!((u - under).abs() < 1e-9 * a.size().max(under.abs()), "{a:?}");
            }
        }
    }

    /// S is symmetric, squares to A and has no negative eigenvalue; S = LQ with Q the
    /// turn `root_turn` says (within a right angle); and every L times a turn is a square
    /// root of A too. For semidefinite matrices as well.
    #[test]
    fn every_square_root_is_l_times_a_turn() {
        let mut cases: Vec<Sym> = matrices(6, 300).into_iter().filter(|a| matches!(cholesky(*a), Chol::Definite { .. })).collect();
        cases.extend([Sym::of(1.0, 1.0, 1.0), Sym::of(1.3, 0.4, -1.0), Sym::of(1.0, 1.0, 0.0), Sym::of(0.3, 1.5, 0.999)]);
        for a in cases {
            let scale = a.size();
            let s = sym_root(a).expect("a root");
            assert!(s.x_axis.y == s.y_axis.x && close(s * s, a.mat(), 1e-10 * scale), "{a:?}");
            let (e1, e2, _) = sym(s).eigen();
            assert!(e2 >= -1e-9 && e1 >= e2, "{a:?}");
            let l = cholesky(a).l().expect("an L");
            let phi = root_turn(l);
            assert!(phi.abs() < std::f64::consts::FRAC_PI_2, "{a:?}: {phi}");
            // (near a singular A the closed form's √det A is good to about 10⁻⁸ only)
            let tol = if a.det() > 1e-6 { 1e-9 } else { 1e-7 } * scale.sqrt().max(1.0);
            assert!(close(l * turn(phi), s, tol), "{a:?}: {:?} against {s:?}", l * turn(phi));
            for k in 0..=8 {
                let b = l * turn(phi * k as f64 / 8.0);
                assert!(close(b * b.transpose(), a.mat(), 1e-12 * scale.max(1.0)), "{a:?}");
            }
        }
        // a matrix that is not semidefinite has no symmetric root
        assert!(sym_root(Sym::of(1.0, 1.0, 1.2)).is_none());
        // S = L exactly when A is diagonal; otherwise S sends every point but 0 somewhere
        // other than where L sends it (the panel's note on the two roots)
        for (s1, s2, r) in [(1.0, 1.0, 0.0), (1.3, 0.4, 0.0), (1.0, 1.0, 0.8), (0.5, 1.2, -0.3), (1.5, 1.5, 0.05)] {
            let a = Sym::of(s1, s2, r);
            let (l, root) = (cholesky(a).l().expect("an L"), sym_root(a).expect("a root"));
            if r == 0.0 {
                assert!(close(l, root, 1e-15) && root_turn(l) == 0.0);
            } else {
                assert!(root_turn(l) != 0.0);
                for i in 0..36 {
                    let z = on_circle(i as f64 / 36.0);
                    assert!((root * z - l * z).length() > 1e-3, "{a:?} at {z:?}");
                }
            }
        }
    }

    /// The steps run from I through the stretch by l₁₁ and l₂₂ to L.
    #[test]
    fn the_steps_run_from_i_to_l() {
        for a in [Sym::of(1.0, 1.0, 0.8), Sym::of(1.2, 1.0, 0.6), Sym::of(0.3, 1.5, -0.95), Sym::of(1.0, 1.0, 1.0)] {
            let l = cholesky(a).l().expect("an L");
            assert!(close(film(l, 0.0), DMat2::IDENTITY, 0.0));
            assert!(close(film(l, 1.0), DMat2::from_diagonal(DVec2::new(l.x_axis.x, l.y_axis.y)), 1e-15));
            assert!(close(film(l, 2.0), l, 1e-15), "{a:?}");
        }
    }

    /// The circle carried by any square root is the ellipse xᵀA⁻¹x = 1, inside the box
    /// and touching it; L's first column is its rightmost point, its second where it
    /// crosses the x₂ axis.
    #[test]
    fn the_ellipse_is_the_circle_carried_by_any_root() {
        for (s1, s2, r) in [(1.0, 1.0, 0.8), (1.2, 1.0, 0.6), (0.4, 1.5, -0.7), (1.5, 0.3, 0.2)] {
            let a = Sym::of(s1, s2, r);
            let inv = a.mat().inverse();
            let l = cholesky(a).l().expect("an L");
            for b in [l, sym_root(a).expect("a root"), l * turn(0.3)] {
                let mut most = DVec2::ZERO;
                for i in 0..2000 {
                    let x = b * on_circle(i as f64 / 2000.0);
                    assert!((x.dot(inv * x) - 1.0).abs() < 1e-12);
                    most = most.max(x.abs());
                }
                assert!((most.x - s1).abs() < 1e-5 && (most.y - s2).abs() < 1e-5, "{most:?}");
            }
            let first = l.x_axis;
            assert!((first.x - s1).abs() < 1e-15 && (first.y - r * s2).abs() < 1e-15);
            assert!(l.y_axis.x == 0.0 && l.y_axis.y > 0.0);
        }
    }

    /// The scene as an experiment's button leaves it, run at 60 frames a second until
    /// everything has played.
    fn run(k: usize) -> Cholesky {
        let mut scene = Cholesky::new();
        scene.set_up(k);
        for _ in 0..60 * 30 {
            scene.advance(1.0 / 60.0, 1.0 / 60.0);
            if scene.settled() {
                break;
            }
        }
        assert!(scene.settled(), "experiment {k} did not settle");
        scene
    }

    #[test]
    fn correlation_from_independence_does_what_its_caption_says() {
        let scene = run(0);
        let s = scene.shown();
        assert!(s.s1 == 1.0 && s.s2 == 1.0 && s.r == 0.8 && s.stage == 2.0 && scene.root() == Root::L);
        let Chol::Definite { l11, l21, l22 } = cholesky(s.a()) else { panic!() };
        assert!(l11 == 1.0 && (l21 - 0.8).abs() < 1e-15 && (l22 - 0.6).abs() < 1e-15);
        assert!(num(1.0 - l21 * l21, 2) == "0.36" && num(l22, 1) == "0.6");
        // x₁ = z₁ and x₂ = 0.8 z₁ + 0.6 z₂, from the two steps: squeezed to 0.6 of the
        // height, then each point raised by 0.8 times its x₁
        let l = scene.drawn().expect("an L");
        let z = DVec2::new(0.7, -1.3);
        assert!(((l * z) - DVec2::new(z.x, 0.8 * z.x + 0.6 * z.y)).length() < 1e-15);
        let squeezed = film(l, 1.0) * z;
        assert!((squeezed - DVec2::new(z.x, 0.6 * z.y)).length() < 1e-15);
        assert!(((film(l, 2.0) * z) - (squeezed + DVec2::new(0.0, 0.8 * squeezed.x))).length() < 1e-15);
        // variances 1, covariance 0.8
        assert!(close(l * l.transpose(), Sym::of(1.0, 1.0, 0.8).mat(), 1e-15));
        // the steps did play, from the start
        let mut scene = Cholesky::new();
        scene.set_up(0);
        let mut least = 2.0f64;
        for _ in 0..600 {
            scene.advance(1.0 / 60.0, 1.0 / 60.0);
            least = least.min(scene.shown().stage);
        }
        assert!(least < 0.02 && scene.shown().stage == 2.0, "{least}");
    }

    #[test]
    fn nearly_a_line_does_what_its_caption_says() {
        // first r = 0.99
        let mut scene = Cholesky::new();
        scene.set_up(1);
        let mut seen = false;
        for _ in 0..60 * 30 {
            scene.advance(1.0 / 60.0, 1.0 / 60.0);
            if scene.glide.arrived() && scene.shown().r == 0.99 && scene.then.is_some() {
                seen = true;
                let Chol::Definite { l22, .. } = cholesky(scene.shown().a()) else { panic!() };
                assert!(num(l22, 2) == "0.14" && num(0.99 * 0.99, 4) == "0.9801");
            }
            if scene.settled() {
                break;
            }
        }
        assert!(seen);
        // then r = 1: the last root is of 0, every x on the line x₂ = x₁
        let s = scene.shown();
        assert!(s.r == 1.0 && scene.settled());
        let c = cholesky(s.a());
        assert!(matches!(c, Chol::Flat { l11, l21 } if l11 == 1.0 && l21 == 1.0), "{c:?}");
        let l = scene.drawn().expect("an L");
        assert!(l.y_axis == DVec2::ZERO && l.determinant() == 0.0);
        assert!(scene.zs.iter().all(|z| { let x = l * *z; x.x == x.y }));
        // semidefinite: xᵀAx = 0 for (−1, 1), and never below 0
        assert!(s.a().form(DVec2::new(-1.0, 1.0)) == 0.0);
        let (_, l2, _) = s.a().eigen();
        assert!(l2.abs() < 1e-15);
        // L is the only one with no negative number on its diagonal: l₁₁ = √a₁₁ is
        // forced, then l₂₁ = a₂₁/l₁₁, then l₂₂² = 0
        assert!(s.a().a11 > 0.0);
        assert!(recipe_lines(s.a(), false)[2].0.contains(r"\sqrt{0} = 0"));
    }

    #[test]
    fn not_positive_definite_does_what_its_caption_says() {
        let scene = run(2);
        let s = scene.shown();
        assert!(s.r == 1.2 && scene.drawn().is_none() && scene.root_matrix().is_none());
        let a = s.a();
        let Chol::Fails3 { l11, l21, under } = cholesky(a) else { panic!() };
        // steps 1 and 2 go through: the first column (1, 1.2), above the box's top
        assert!(l11 == 1.0 && l21 == 1.2 && l21 > s.s2);
        assert!(num(l21 * l21, 2) == "1.44" && num(under, 2) == "−0.44");
        // −0.44 is xᵀAx at (−1.2, 1); the eigenvalues are 2.2 and −0.2
        assert!((a.form(DVec2::new(-1.2, 1.0)) - under).abs() < 1e-15 && multiplier(a) == 1.2);
        let (l1, l2, _) = a.eigen();
        assert!((l1 - 2.2).abs() < 1e-12 && (l2 + 0.2).abs() < 1e-12);
        assert!(sym_root(a).is_none());
        let lines = recipe_lines(a, false);
        assert!(lines[2].0.contains(r"\sqrt{−0.44}") && lines[2].1 == pal::RED, "{}", lines[2].0);
    }

    #[test]
    fn two_square_roots_does_what_its_caption_says() {
        let scene = run(3);
        let s = scene.shown();
        assert!(s.r == 0.8 && s.s1 == 1.0 && s.s2 == 1.0 && scene.root() == Root::S);
        let a = s.a();
        let root = sym_root(a).expect("a root");
        let drawn = scene.drawn().expect("a root");
        assert!(close(drawn, root, 1e-12));
        assert!(num(root.x_axis.x, 2) == "0.89" && num(root.y_axis.y, 2) == "0.89" && num(root.x_axis.y, 2) == "0.45" && root.x_axis.y == root.y_axis.x);
        assert!(close(root * root.transpose(), a.mat(), 1e-12));
        // S = LQ, Q turning 26.6° clockwise: S takes z where L takes z turned so
        let l = cholesky(a).l().expect("an L");
        let phi = root_turn(l);
        assert!(turn_words(phi) == "26.6° clockwise");
        for i in 0..12 {
            let z = on_circle(i as f64 / 12.0);
            assert!((root * z - l * (turn(phi) * z)).length() < 1e-12);
        }
        // on the way, from the start: every matrix drawn is a square root of A
        let mut scene = Cholesky::new();
        scene.set_up(3);
        let mut between = 0;
        for _ in 0..60 * 10 {
            scene.advance(1.0 / 60.0, 1.0 / 60.0);
            let b = scene.drawn().expect("a root");
            assert!(close(b * b.transpose(), a.mat(), 1e-12));
            between += (scene.root() == Root::Between) as usize;
        }
        assert!(between > 60 && scene.root() == Root::S, "{between}");
        // pressed again at S, it goes back to L first and then to S again
        scene.set_up(3);
        let mut back = false;
        for _ in 0..60 * 10 {
            scene.advance(1.0 / 60.0, 1.0 / 60.0);
            back |= scene.root() == Root::L;
        }
        assert!(back && scene.root() == Root::S);
    }

    #[test]
    fn elimination_does_what_its_caption_says() {
        let scene = run(4);
        let a = scene.shown().a();
        assert!(num(a.a11, 2) == "1.44" && num(a.a21, 2) == "0.72" && a.a22 == 1.0 && scene.root() == Root::L);
        // row 2 − 0.5 × row 1 leaves (0, 0.64)
        let m = multiplier(a);
        assert!((m - 0.5).abs() < 1e-15);
        let row2 = DVec2::new(a.a21, a.a22) - DVec2::new(a.a11, a.a21) * m;
        assert!(row2.x.abs() < 1e-15 && (row2.y - 0.64).abs() < 1e-15 && num(1.0 - 0.5 * 0.72, 2) == "0.64");
        let Chol::Definite { l11, l21, l22 } = cholesky(a) else { panic!() };
        assert!((l11 - 1.2).abs() < 1e-15 && (l22 - 0.8).abs() < 1e-15 && (l21 / l11 - 0.5).abs() < 1e-15);
        assert!((l11 * l11 - a.a11).abs() < 1e-15 && (l22 * l22 - row2.y).abs() < 1e-15);
        // L = the shear by the multiplier times the stretch by the roots of the pivots,
        // and A = L₁ D L₁ᵀ (the panel's words)
        let shear = DMat2::from_cols(DVec2::new(1.0, m), DVec2::Y);
        let stretch = DMat2::from_diagonal(DVec2::new(l11, l22));
        assert!(close(shear * stretch, lower(l11, l21, l22), 1e-15));
        let d = DMat2::from_diagonal(DVec2::new(a.a11, row2.y));
        assert!(close(shear * d * shear.transpose(), a.mat(), 1e-15));
        for a in matrices(8, 100).into_iter().filter(|a| matches!(cholesky(*a), Chol::Definite { .. })) {
            let l1 = DMat2::from_cols(DVec2::new(1.0, multiplier(a)), DVec2::Y);
            let d = DMat2::from_diagonal(DVec2::new(a.a11, a.a22 - a.a21 * a.a21 / a.a11));
            let tol = 1e-12 * a.size().max(1.0) * (1.0 + multiplier(a).powi(2));
            assert!(close(l1 * d * l1.transpose(), a.mat(), tol), "{a:?}");
            assert!(close(l1 * DMat2::from_diagonal(DVec2::new(d.x_axis.x.sqrt(), d.y_axis.y.sqrt())), cholesky(a).l().expect("an L"), tol), "{a:?}");
        }
    }

    /// The cloud is a fair sample: its covariance is near I, and the covariance of the
    /// x drawn is B times it times Bᵀ.
    #[test]
    fn the_cloud_is_round() {
        let zs = sample();
        let c = covariance(&zs);
        assert!(close(c, DMat2::IDENTITY, 0.12), "{c:?}");
        let b = lower(1.0, 0.8, 0.6);
        let xs: Vec<DVec2> = zs.iter().map(|z| b * *z).collect();
        assert!(close(covariance(&xs), b * c * b.transpose(), 1e-12));
    }

    /// Degenerate input: every number drawn or read out is finite, whatever the
    /// matrix, the steps and the change of root.
    #[test]
    fn degenerate_input_leaves_everything_finite() {
        for s1 in [S_MIN, 1.0, S_MAX] {
            for s2 in [S_MIN, S_MAX] {
                for r in [-R_MAX, -1.0, -0.999_999, 0.0, 0.999_999_999, 1.0, 1.000_001, R_MAX] {
                    let mut scene = Cholesky::new();
                    scene.by_hand(|g| (g.s1, g.s2, g.r) = (s1, s2, r));
                    for (stage, blend) in [(0.0, 0.0), (0.5, 0.0), (1.5, 0.0), (2.0, 0.5), (2.0, 1.0)] {
                        scene.glide.now(|g| g.stage = stage);
                        scene.blend = blend;
                        if let Some(b) = scene.drawn() {
                            assert!(b.as_mat2().is_finite() && scene.zs.iter().all(|z| (b * *z).is_finite()));
                        }
                        let a = scene.shown().a();
                        let texts: Vec<String> = recipe_lines(a, false).into_iter().chain(recipe_lines(a, true)).map(|l| l.0).chain(verdict(a, false).into_iter().map(|l| l.0)).chain([scene.x_title(false).0, scene.x_title(true).0]).collect();
                        for t in texts {
                            assert!(!t.contains("NaN") && !t.contains("inf") && !t.contains('?'), "{s1} {s2} {r}: {t}");
                        }
                    }
                }
            }
        }
    }

    /// The pictures and the numbers fit the canvas, from a phone to a desktop, and do
    /// not overlap; every line of the numbers fits across its column.
    #[test]
    fn the_layout_fits_every_canvas() {
        for (w, h) in [(1080.0, 1000.0), (1100.0, 700.0), (1300.0, 900.0), (800.0, 600.0), (580.0, 700.0), (390.0, 800.0), (390.0, 560.0), (360.0, 640.0), (620.0, 480.0), (240.0, 180.0), (1600.0, 1000.0), (300.0, 330.0)] {
            for inset in [0.0, 46.0] {
                let view = Vec2::new(w, h);
                let lay = Layout::of(view, inset);
                let fits = |(lo, hi): (Vec2, Vec2)| lo.cmpge(Vec2::splat(-0.5)).all() && hi.cmple(view + Vec2::splat(0.5)).all();
                for p in lay.panes {
                    assert!(fits(p) && (p.1 - p.0).min_element() > 7.0, "{w}x{h}: {lay:?}");
                }
                assert!(lay.panes[0].1.x <= lay.panes[1].0.x || lay.panes[0].1.y <= lay.panes[1].0.y, "{w}x{h}: the pictures overlap");
                if let Some(hud) = &lay.hud {
                    assert!(fits((hud.min, hud.max)), "{w}x{h}: {hud:?}");
                    assert!(lay.panes.iter().all(|p| p.1.y <= hud.min.y), "{w}x{h}: the numbers over a picture");
                    assert!(lay.panes.iter().all(|p| p.0.y >= inset), "{w}x{h}: under the buttons");
                    for b in hud.buttons {
                        assert!(b.1.x <= hud.max.x && b.1.y <= hud.max.y, "{w}x{h}: {b:?}");
                    }
                    // the widest lines there can be, against the room for them
                    let room = hud.recipe_width();
                    for a in [Sym::of(1.5, 1.5, -0.999), Sym::of(1.5, 1.5, -R_MAX), Sym::of(1.2, 1.0, 0.6), Sym::of(1.0, 1.0, 1.0)] {
                        let mut lines: Vec<(String, f32)> = recipe_lines(a, !hud.two).into_iter().map(|l| (l.0, hud.text)).collect();
                        lines.extend(hud.verdict(a).into_iter().map(|l| (l.0, hud.text - 0.5)));
                        for (line, size) in lines {
                            assert!(text_width(&line, size) <= room, "{w}x{h}: {line} is {} wide, with {room}", text_width(&line, size));
                        }
                    }
                    let matrices_end = hud.mats.x + mats_width(hud.text, hud.cell);
                    assert!(matrices_end <= if hud.two { hud.recipe.x - 10.0 } else { hud.max.x - PAD + 4.0 }, "{w}x{h}: the matrices run to {matrices_end}");
                    let bottom = if hud.two { hud.recipe.y + 7.0 * hud.line } else { hud.recipe.y + 5.0 * hud.line };
                    assert!(bottom <= hud.max.y + 0.5, "{w}x{h}: the numbers run to {bottom}, below {}", hud.max.y);
                }
            }
        }
    }

    /// A drag of the handles, by hand, keeps every number within its limits and shows
    /// all of L.
    #[test]
    fn handles_keep_within_limits() {
        let mut scene = Cholesky::new();
        scene.set_up(0);
        scene.by_hand(|g| (g.s1, g.r) = (9.0f64.clamp(S_MIN, S_MAX), (-7.0f64).clamp(-R_MAX, R_MAX)));
        assert!(scene.glide.to.s1 == S_MAX && scene.glide.to.r == -R_MAX && scene.glide.to.stage == 2.0 && !scene.playing && !scene.play_when_there);
    }
}

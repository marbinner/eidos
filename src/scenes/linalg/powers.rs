//! Powers of a matrix: x, Ax, A²x, … Written in the eigenvectors, every step multiplies
//! each part by its own eigenvalue, and the part whose eigenvalue is largest in size
//! takes over.

use super::*;
use glam::DVec2;

pub const DEF: SceneDef = SceneDef {
    id: "powers",
    title: "Powers of a matrix",
    group: "Linear algebra: eigenvalues",
    blurb: "Apply a map again and again: the largest eigenvalue takes over.",
    make: |_| Box::new(Powers::create()),
};

/// One of the "things to try": everything it depends on.
struct Lesson {
    name: &'static str,
    caption: &'static str,
    /// The matrix by its columns: (a₁₁, a₂₁, a₁₂, a₂₂).
    cols: [f32; 4],
    x0: [f64; 2],
    /// The step its play stops at, and the steps the view is fitted to at the start.
    plan: usize,
    view: usize,
    /// Steps per second.
    rate: f32,
}

/// 0.9 cos 30° and 0.9 sin 30°: a turn of 30° and a shrink to 0.9.
const TURN_C: f32 = 0.779_422_9;
const TURN_S: f32 = 0.45;

const TRIALS: [Lesson; 5] = [
    Lesson {
        name: "the largest eigenvalue wins",
        caption: r"$x_0$ is mostly along $v_2$. Each step multiplies the part along $v_1$ by $\lambda_1 = 1.2$ and the part along $v_2$ by $\lambda_2 = 0.6$, so the purple part halves against the orange one every step: $x$ first falls in towards $0$, then turns and runs out along $v_1$'s line. $x_{10}$ is 0.2° from that line, and grows by 1.196 a step, nearly $\lambda_1$. On the chart the two parts are straight lines, falling and rising by the same factors every step. Every start that is not on $v_2$'s line ends up the same way: running out along $v_1$'s line, longer by 1.2 a step.",
        cols: [1.32, 0.24, -0.36, 0.48],
        x0: [2.0, 3.0],
        plan: 10,
        view: 10,
        rate: 2.5,
    },
    Lesson {
        name: "a spiral into $0$",
        caption: r"$A$ turns the plane by 30° and shrinks it to 0.9 of its size. No line stays on itself: the eigenvalues are complex, $0.9(\cos 30^\circ \pm i \sin 30^\circ)$. Every step turns $x$ by $\theta = 30^\circ$ and multiplies its length by $r = 0.9$, so twelve steps make a full turn and leave $x$ at $0.9^{12} = 0.28$ of its length. Both eigenvalues are smaller than 1 in size: every start spirals into $0$.",
        cols: [TURN_C, TURN_S, -TURN_S, TURN_C],
        x0: [3.0, 0.0],
        plan: 24,
        view: 24,
        rate: 4.0,
    },
    Lesson {
        name: "a Markov chain settles",
        caption: r"Each year 10% of the people in a town move to the country, and 20% of those in the country move to the town: each column of $A$ is how one place's people are shared out, and adds to 1. $x$ is how many million live in the town, $y$ in the country, and all 3 million start in the country. Nobody is lost, so $x$ stays on the line $x + y = 3$, and slides along it to the steady state $(2, 1)$, the eigenvector with $\lambda_1 = 1$. What is left to go shrinks to $\lambda_2 = 0.7$ of itself every step. Drag $x_0$ along the green line: wherever the 3 million start, they end at $(2, 1)$.",
        cols: [0.9, 0.1, 0.2, 0.8],
        x0: [0.0, 3.0],
        plan: 12,
        view: 12,
        rate: 2.5,
    },
    Lesson {
        name: "Fibonacci and the golden ratio",
        caption: r"$A$ sends $(a, b)$ to $(a + b, a)$. From $(1, 0)$ come $(1, 1), (2, 1), (3, 2), (5, 3), (8, 5), \ldots$: the Fibonacci numbers, each the sum of the two before (the powers of $A$ are made of them too). $\lambda_1 = (1 + \sqrt{5})/2 = 1.618\ldots$, the golden ratio, and $\lambda_2 = -0.618\ldots$: the small part changes sign every step, so $x$ zigzags across $v_1$'s line as it closes in on it. $x_8 = (34, 21)$: $34/21 = 1.619$, and the growth per step is already 1.618.",
        cols: [1.0, 1.0, 1.0, 0.0],
        x0: [1.0, 0.0],
        plan: 8,
        view: 8,
        rate: 2.0,
    },
    Lesson {
        name: "a shear grows in a straight line",
        caption: r"$A$ slides every point sideways by its height. $\lambda = 1$ twice, but there is only one eigenvector line, the $x$ axis. $x$ steps along the line $y = 1$, the same distance every time: $x_k = (k, 1)$. That is growth by adding, not by multiplying: $\lVert x_{12} \rVert / \lVert x_{11} \rVert = 1.09$, and the ratio creeps down towards 1. $x$ turns towards the $x$ axis too, but slowly: the angle is about $1/k$ radians, which bends on the log chart instead of falling straight.",
        cols: [1.0, 0.0, 1.0, 1.0],
        x0: [0.0, 1.0],
        plan: 12,
        view: 12,
        rate: 2.5,
    },
];

/// cos 45° = sin 45°, the entries of an eighth of a turn.
const EIGHTH: f32 = std::f32::consts::FRAC_1_SQRT_2;

/// Quick choices of matrix: (button, the matrix by its columns).
const PRESETS: [(&str, [f32; 4]); 4] = [("turn 45°", [EIGHTH, EIGHTH, -EIGHTH, EIGHTH]), ("flip", [0.0, 1.0, 1.0, 0.0]), ("squash", [0.6, 0.4, 0.6, 0.4]), ("zero", [0.0; 4])];

/// The longest run.
const MOST_STEPS: usize = 100;
/// A run also stops once x is this many times as long as x₀: as far out as the
/// picture can follow it (x₀ itself is kept within `REACH` of 0).
const FAR: f64 = 1e6;
const REACH: f64 = 500.0;
/// How long the drawn step takes to follow the step taken: two eases of this many
/// seconds one after the other (shorter when the steps come faster).
const FOLLOW: f64 = 0.085;
/// The longest the axes' numbers are left out while the view glides, seconds.
const QUIET: f32 = 0.6;
/// How long a chart's range takes to glide to a new one, seconds.
const RESCALE: f32 = 0.12;
/// The closest the view comes, logical pixels per unit.
const MOST_PER_UNIT: f32 = 260.0;
/// Starts in the ring, and how many of them are drawn as dots.
const RING: usize = 96;
const RING_DOTS: usize = 16;

/// The colours: one thing each, in the picture, in the charts and in the key.
const X_COL: Color = pal::WHITE;
const V1: Color = pal::ORANGE;
const V2: Color = pal::PURPLE;
const ANGLE: Color = pal::YELLOW;
const RING_COL: Color = pal::CYAN;
const SHARES: Color = pal::GREEN;

// ----- the arithmetic, in double precision ---------------------------------------------

/// A 2×2 matrix by its rows: (a b; c d).
#[derive(Clone, Copy, Debug, PartialEq)]
struct M2 {
    a: f64,
    b: f64,
    c: f64,
    d: f64,
}

impl M2 {
    const I: M2 = M2 { a: 1.0, b: 0.0, c: 0.0, d: 1.0 };

    /// The matrix as edited (single precision, by columns). A number that is not finite
    /// counts as 0, and none is larger in size than the editor allows.
    fn of(m: Mat2) -> M2 {
        let ok = |v: f32| if v.is_finite() { (v as f64).clamp(-100.0, 100.0) } else { 0.0 };
        M2 { a: ok(m.x_axis.x), b: ok(m.y_axis.x), c: ok(m.x_axis.y), d: ok(m.y_axis.y) }
    }

    fn times(self, o: M2) -> M2 {
        M2 { a: self.a * o.a + self.b * o.c, b: self.a * o.b + self.b * o.d, c: self.c * o.a + self.d * o.c, d: self.c * o.b + self.d * o.d }
    }

    fn apply(self, x: DVec2) -> DVec2 {
        DVec2::new(self.a * x.x + self.b * x.y, self.c * x.x + self.d * x.y)
    }

    fn size(self) -> f64 {
        self.a.abs().max(self.b.abs()).max(self.c.abs()).max(self.d.abs())
    }

    fn rows(self) -> [[f64; 2]; 2] {
        [[self.a, self.b], [self.c, self.d]]
    }

    /// Entries ≥ 0 and columns that add to 1: each column says how the share in one
    /// state is passed on in a step.
    fn markov(self) -> bool {
        [self.a, self.b, self.c, self.d].iter().all(|v| *v >= 0.0) && (self.a + self.c - 1.0).abs() < 1e-6 && (self.b + self.d - 1.0).abs() < 1e-6
    }
}

/// What the eigenvalues of a 2×2 matrix are like.
#[derive(Clone, Copy, Debug, PartialEq)]
enum Eigen {
    /// Two different real eigenvalues, each with its line: the larger in size first
    /// (of two the same size, the positive one), with unit eigenvectors.
    Real { l: [f64; 2], v: [DVec2; 2] },
    /// One eigenvalue twice, with only one line: a shear, perhaps scaled.
    Defective { l: f64, v: DVec2 },
    /// A = λI: every line through 0 is an eigenvector line.
    Scalar { l: f64 },
    /// λ = r(cos θ ± i sin θ), 0 < θ < π: no line stays on itself.
    Complex { r: f64, theta: f64 },
}

/// The unit vector along the line that `m − l I` sends to 0: at right angles to its
/// longer row. Pointed to the right (or up), so that it is the same every time.
fn line_of(m: M2, l: f64) -> DVec2 {
    let (r1, r2) = (DVec2::new(m.a - l, m.b), DVec2::new(m.c, m.d - l));
    let r = if r1.length_squared() >= r2.length_squared() { r1 } else { r2 };
    let v = DVec2::new(-r.y, r.x).try_normalize().unwrap_or(DVec2::X);
    if v.x < 0.0 || (v.x == 0.0 && v.y < 0.0) {
        -v
    } else {
        v
    }
}

fn eigen(m: M2) -> Eigen {
    let size = m.size();
    if size == 0.0 || !size.is_finite() {
        return Eigen::Scalar { l: 0.0 };
    }
    let half = (m.a + m.d) * 0.5;
    let det = m.a * m.d - m.b * m.c;
    let disc = half * half - det;
    // The entries are single-precision numbers, whose products are exact here: a
    // repeated eigenvalue comes out with disc 0 or within rounding of it, while two
    // eigenvalues as close as single precision can tell apart are far from that.
    if disc.abs() <= 1e-20 * size * size {
        if (m.a - m.d).abs().max(m.b.abs()).max(m.c.abs()) <= 1e-12 * size {
            return Eigen::Scalar { l: half };
        }
        return Eigen::Defective { l: half, v: line_of(m, half) };
    }
    if disc < 0.0 {
        return Eigen::Complex { r: det.max(0.0).sqrt(), theta: (-disc).sqrt().atan2(half) };
    }
    let s = disc.sqrt();
    // the larger in size without cancellation, and the other from the product
    let big = if half >= 0.0 { half + s } else { half - s };
    let small = det / big;
    Eigen::Real { l: [big, small], v: [line_of(m, big), line_of(m, small)] }
}

impl Eigen {
    /// The line the direction of x turns to, if there is one: v₁ (when one eigenvalue
    /// is larger in size than the other), or the only line there is.
    fn winner(self) -> Option<DVec2> {
        match self {
            Eigen::Real { l, v } if l[0].abs() > l[1].abs() * (1.0 + 1e-9) => Some(v[0]),
            Eigen::Defective { v, .. } => Some(v),
            _ => None,
        }
    }

    /// The two lines, when there are two.
    fn lines(self) -> Option<[DVec2; 2]> {
        match self {
            Eigen::Real { v, .. } => Some(v),
            _ => None,
        }
    }
}

/// x written in the eigenvectors: x = a v₁ + b v₂.
fn parts(v: [DVec2; 2], x: DVec2) -> [f64; 2] {
    let den = v[0].perp_dot(v[1]);
    [x.perp_dot(v[1]) / den, v[0].perp_dot(x) / den]
}

/// The angle between x and the line along `v`, in degrees (0 to 90).
fn angle_to_line(x: DVec2, v: DVec2) -> f64 {
    x.perp_dot(v).abs().atan2(x.dot(v).abs()).to_degrees()
}

/// x₀, Ax₀, A²x₀, … and the powers of A with them.
#[derive(Clone, Debug)]
struct Run {
    /// x₀, x₁, …, xₙ.
    x: Vec<DVec2>,
    /// A⁰, A¹, …, Aⁿ.
    p: Vec<M2>,
    /// Whether it ended because x went further out than the picture can follow
    /// (otherwise it is `MOST_STEPS` long).
    far: bool,
    /// The most characters any power's entries take on the screen (a cross counted as
    /// three): the matrix is written that wide from the first step to the last, so that
    /// its brackets stay put.
    widest: usize,
}

impl Run {
    fn of(m: M2, x0: DVec2) -> Run {
        let limit = FAR * x0.length();
        let (mut x, mut p) = (vec![x0], vec![M2::I]);
        let (mut xk, mut pk, mut far) = (x0, M2::I, false);
        for _ in 0..MOST_STEPS {
            (xk, pk) = (m.apply(xk), m.times(pk));
            x.push(xk);
            p.push(pk);
            // (a number that is not one stops it too)
            if xk.length() > limit || !xk.is_finite() {
                far = true;
                break;
            }
        }
        // (a cross, with the room set round it, takes about three characters' room)
        let widest = p.iter().flat_map(|m| together(&[m.a, m.b, m.c, m.d])).map(|t| t.chars().count() + 2 * t.matches('×').count()).max().unwrap_or(1).max(4);
        Run { x, p, far, widest }
    }

    /// The last step.
    fn last(&self) -> usize {
        self.x.len() - 1
    }

    /// x at step k (the last one there is, beyond the end).
    fn at(&self, k: usize) -> DVec2 {
        self.x[k.min(self.last())]
    }

    fn power(&self, k: usize) -> M2 {
        self.p[k.min(self.last())]
    }

    /// x at a step between two whole ones, on the straight line from one to the next.
    fn between(&self, s: f64) -> DVec2 {
        let s = s.max(0.0).min(self.last() as f64);
        let i = s.floor() as usize;
        self.at(i).lerp(self.at(i + 1), s - i as f64)
    }

    /// |x_k| / |x_{k−1}|: how much the last step stretched x (None at the start, or
    /// when x_{k−1} is 0).
    fn growth(&self, k: usize) -> Option<f64> {
        let k = k.min(self.last());
        (k >= 1 && self.x[k - 1].length() > 0.0).then(|| self.x[k].length() / self.x[k - 1].length())
    }
}

// ----- numbers for the screen ------------------------------------------------------

/// A power of ten as it is written: 10⁶, 10⁻¹⁶.
fn ten_to(e: i32) -> String {
    format!("10{}{}", if e < 0 { "⁻" } else { "" }, sup(e.unsigned_abs() as usize))
}

/// A number to `digits` significant digits, with a true minus sign and no trailing
/// zeros; a whole number as it is; very large and very small ones as 1.2×10⁶. (It goes
/// on the screen inside a formula, which sets the × and the raised digits as
/// mathematics, as it does the numbers that `together` and `point` write.)
fn sig(v: f64, digits: i32) -> String {
    if !v.is_finite() {
        return "?".into();
    }
    let a = v.abs();
    if a == 0.0 {
        return "0".into();
    }
    if a < 1e7 && v.fract() == 0.0 {
        return num(v, 0);
    }
    let trim = |t: String| if t.contains('.') { t.trim_end_matches('0').trim_end_matches('.').to_string() } else { t };
    if !(1e-3..1e5).contains(&a) {
        let mut e = a.log10().floor() as i32;
        let mut m = v / 10f64.powi(e);
        // (9.996 rounds to 10.0: that is 1.0 of the next power)
        if num(m.abs(), (digits - 1).max(0) as usize).starts_with("10") {
            e += 1;
            m /= 10.0;
        }
        return format!("{}×{}", trim(num(m, (digits - 1).max(0) as usize)), ten_to(e));
    }
    let decimals = (digits - 1 - a.log10().floor() as i32).clamp(0, 6) as usize;
    trim(num(v, decimals))
}

/// Numbers that belong together (a point, a matrix), each to the places that give
/// the largest of them three significant digits: a part that rounding has left, a
/// millionth of the rest, then reads 0. Whole numbers are written whole.
fn together(values: &[f64]) -> Vec<String> {
    let big = values.iter().fold(0.0f64, |m, v| m.max(v.abs()));
    if !(1e-3..1e5).contains(&big) {
        return values.iter().map(|v| sig(*v, 3)).collect();
    }
    if values.iter().all(|v| v.fract() == 0.0) {
        return values.iter().map(|v| num(*v, 0)).collect();
    }
    let decimals = (2 - big.log10().floor() as i32).clamp(0, 5) as usize;
    values.iter().map(|v| num(*v, decimals)).map(|t| if t.chars().all(|c| c == '0' || c == '.') { "0".into() } else { t }).collect()
}

/// A point as "(x, y)".
fn point(p: DVec2) -> String {
    let [x, y] = <[String; 2]>::try_from(together(&[p.x, p.y])).unwrap_or_default();
    format!("({x}, {y})")
}

/// An angle in degrees, to three significant digits, to go in a formula. (The degree
/// sign is a raised circle after an empty group, as TeX writes it: it then follows a
/// number written with a power of ten as well as any other.)
fn degrees(d: f64) -> String {
    format!(r"{}{{}}^\circ", sig(d, 3))
}

// ----- the layout on the screen ------------------------------------------------------

/// Where everything goes on a canvas of `view` logical pixels, the top `inset` under
/// the application's buttons: the numbers, the buttons and the charts in a column on
/// the right, or below the picture on a narrow canvas (a phone held upright), or
/// left out on one too small for them.
#[derive(Clone, Copy, Debug, PartialEq)]
struct Layout {
    narrow: bool,
    /// The part of the canvas the picture is fitted into.
    plane: (Vec2, Vec2),
    /// The dark backing behind the buttons, the numbers and the charts.
    backing: Option<(Vec2, Vec2)>,
    /// The row of buttons: its top-left corner, and whether the words are short.
    buttons: Option<(Vec2, bool)>,
    /// The numbers: top-left corner, width, and text size.
    sums: Option<(Vec2, f32, f32)>,
    /// The two charts' plot areas: top-left corner and size.
    charts: [Option<(Vec2, Vec2)>; 2],
}

/// The height the numbers take, for a text size.
fn sums_height(size: f32) -> f32 {
    (size + 6.0) + 2.0 * (size + 8.0) + 6.0 + 4.0 * (size + 5.0)
}

/// The room a chart's title and key take above its plot, and its step numbers below.
const CHART_ABOVE: f32 = 40.0;
const CHART_BELOW: f32 = 18.0;

impl Layout {
    fn of(view: Vec2, inset: f32) -> Layout {
        let narrow = view.x < 620.0 && view.y >= 560.0;
        let none = [None, None];
        if narrow {
            // the lower part of the canvas, under the picture
            let height = (view.y * 0.44).clamp(300.0, 400.0);
            let top = view.y - height;
            let text = 12.0;
            let sums_top = top + 46.0;
            let charts_top = sums_top + sums_height(text) + 8.0 + CHART_ABOVE;
            let plot_h = view.y - CHART_BELOW - 10.0 - charts_top;
            let half = (view.x - 36.0) * 0.5;
            let plot_w = half - 40.0;
            let charts = if plot_h >= 50.0 && plot_w >= 60.0 {
                [Some((Vec2::new(12.0 + 40.0, charts_top), Vec2::new(plot_w, plot_h))), Some((Vec2::new(24.0 + half + 40.0, charts_top), Vec2::new(plot_w, plot_h)))]
            } else {
                none
            };
            return Layout {
                narrow,
                plane: (Vec2::new(0.0, inset), Vec2::new(view.x, top)),
                backing: Some((Vec2::new(0.0, top), view)),
                buttons: Some((Vec2::new(12.0, top + 10.0), true)),
                sums: Some((Vec2::new(12.0, sums_top), view.x - 24.0, text)),
                charts,
            };
        }
        let top = inset.max(0.0) + 16.0;
        if view.x >= 640.0 && view.y >= 380.0 {
            let width = (view.x * 0.3).clamp(270.0, 340.0);
            let x = view.x - width - 16.0;
            let text = 13.0;
            let sums_top = top + 48.0;
            let below_sums = sums_top + sums_height(text) + 12.0;
            let room = view.y - 16.0 - below_sums;
            let block = CHART_ABOVE + CHART_BELOW + 14.0;
            let plot_w = width - 62.0;
            let left = x + 50.0;
            let (charts, end) = if room >= 2.0 * (block + 70.0) {
                let h = ((room - 2.0 * block) * 0.5).min(190.0);
                let first = below_sums + CHART_ABOVE;
                let second = first + h + CHART_BELOW + 14.0 + CHART_ABOVE;
                ([Some((Vec2::new(left, first), Vec2::new(plot_w, h))), Some((Vec2::new(left, second), Vec2::new(plot_w, h)))], second + h + CHART_BELOW)
            } else if room >= block + 60.0 {
                let h = (room - block).min(190.0);
                let first = below_sums + CHART_ABOVE;
                ([Some((Vec2::new(left, first), Vec2::new(plot_w, h))), None], first + h + CHART_BELOW)
            } else {
                (none, below_sums - 6.0)
            };
            return Layout {
                narrow,
                plane: (Vec2::new(0.0, inset), Vec2::new(x - 8.0, view.y)),
                backing: Some((Vec2::new(x, top), Vec2::new(x + width, end + 6.0))),
                // (the short words when the long ones would not fit across)
                buttons: Some((Vec2::new(x + 12.0, top + 10.0), width < BUTTON_LONG.iter().sum::<f32>() + 40.0)),
                sums: Some((Vec2::new(x + 12.0, sums_top), width - 24.0, text)),
                charts,
            };
        }
        // too small for the numbers and the charts (the side panel has them): the
        // buttons alone, over the picture's corner, if even they fit
        let fits = view.x >= 280.0 && view.y >= 160.0;
        Layout {
            narrow,
            plane: (Vec2::new(0.0, if fits { top + 36.0 } else { inset }), view),
            backing: fits.then(|| (Vec2::new(8.0, top - 6.0), Vec2::new(8.0 + BUTTONS_SHORT, top + 34.0))),
            buttons: fits.then_some((Vec2::new(14.0, top), true)),
            sums: None,
            charts: none,
        }
    }

    /// Is the screen point `p` on one of the panels?
    fn covers(&self, p: Vec2) -> bool {
        self.backing.is_some_and(|(lo, hi)| p.cmpge(lo).all() && p.cmple(hi).all())
    }

    /// The three buttons: (top-left, bottom-right) of each.
    fn button_rects(&self) -> Option<[(Vec2, Vec2); 3]> {
        let (at, short) = self.buttons?;
        let widths = if short { BUTTON_SHORT } else { BUTTON_LONG };
        let mut x = at.x;
        Some(widths.map(|w| {
            let r = (Vec2::new(x, at.y), Vec2::new(x + w, at.y + 28.0));
            x += w + 8.0;
            r
        }))
    }
}

const BUTTON_LONG: [f32; 3] = [72.0, 82.0, 128.0];
const BUTTON_SHORT: [f32; 3] = [66.0, 74.0, 74.0];
const BUTTONS_SHORT: f32 = 66.0 + 74.0 + 74.0 + 2.0 * 8.0 + 12.0;

/// A button drawn over the picture (logical pixels): true when it is clicked. One that
/// cannot be used now is drawn dim and ignores clicks. (As in `linalg/gram.rs`.)
fn hud_button(f: &mut Frame, (min, max): (Vec2, Vec2), text: &str, usable: bool, pointer: Option<Vec2>) -> bool {
    let over = pointer.is_some_and(|p| p.cmpge(min).all() && p.cmple(max).all());
    let fill = match (usable, over) {
        (false, _) => Color::hex(0x141925),
        (true, true) => Color::hex(0x33405E),
        (true, false) => Color::hex(0x222B40),
    };
    hud_rect(f, min, max, fill);
    f.text_px((min + max) * 0.5, text, 12.5, if usable { pal::TEXT } else { pal::FAINT }, Align2::CENTER_CENTER);
    usable && over && f.ptr.clicked
}

// ----- the scene ---------------------------------------------------------------------

pub struct Powers {
    a: Mat2,
    x0: DVec2,
    /// The steps taken.
    k: usize,
    /// The step drawn: it follows `k` through two eases (`between` is the first), so
    /// that the dots glide from place to place.
    between: f64,
    shown: f64,
    playing: bool,
    /// Steps per second while playing, and the part of a step owed so far.
    rate: f32,
    owed: f32,
    /// Where play stops of itself, if before the end (an experiment's last step).
    stop_at: Option<usize>,
    /// The steps the view is fitted to, now and after "back to the start"; and the
    /// steps the charts are to show at least.
    view_k: usize,
    home_k: usize,
    plan: usize,
    show_parts: bool,
    ring: bool,
    /// x₀ as drawn: it glides to where an experiment puts it.
    x0_drawn: DVec2,
    /// The eigenvector lines fade in after a new matrix: 0 to 1.
    fresh: f32,
    /// The charts as drawn: the steps across them, and the powers of ten from bottom
    /// to top of each. They glide to new values.
    span: f32,
    ranges: [[f32; 2]; 2],
    /// The run for the matrix and start as they are now, and what it was worked out for.
    run: Run,
    key: (M2, DVec2),
    /// The canvas last drawn on, and how much of its top the application's buttons cover.
    canvas: Vec2,
    inset: f32,
    measured: bool,
    new_view: bool,
    /// While the view glides to one asked for: the numbers along the axes are left
    /// out (they would come and go as the scale changes), for at most this long.
    quiet_axes: f32,
    last_bounds: (Vec2, Vec2),
    /// Whether the next new view is a change of lesson (and not the view following x),
    /// and the step the view last followed x to.
    quiet_next: bool,
    followed: usize,
    /// Whether x₀ was being dragged in the last frame.
    dragging: bool,
    /// Where each name in the picture was last drawn.
    kept: Kept,
}

impl Powers {
    /// (Nothing here needs the graphics device, so the tests can make one too.)
    fn create() -> Self {
        let mut scene = Powers {
            a: Mat2::IDENTITY,
            x0: DVec2::ONE,
            k: 0,
            between: 0.0,
            shown: 0.0,
            playing: false,
            rate: 2.5,
            owed: 0.0,
            stop_at: None,
            view_k: 10,
            home_k: 10,
            plan: 10,
            show_parts: true,
            ring: false,
            x0_drawn: DVec2::ONE,
            fresh: 1.0,
            span: 20.0,
            ranges: [[-1.0, 1.0], [-1.0, 2.0]],
            run: Run::of(M2::I, DVec2::ONE),
            key: (M2::I, DVec2::ONE),
            canvas: Vec2::new(1080.0, 1000.0),
            inset: 0.0,
            measured: false,
            new_view: false,
            quiet_axes: 0.0,
            last_bounds: (Vec2::ZERO, Vec2::ZERO),
            quiet_next: false,
            followed: 0,
            dragging: false,
            kept: Kept::new(),
        };
        scene.set_up(0);
        scene.x0_drawn = scene.x0;
        scene.fresh = 1.0;
        scene.refresh();
        (scene.span, scene.ranges) = (scene.span_wanted(), scene.ranges_wanted());
        scene
    }

    /// Set up one of the experiments in `TRIALS`, and play it from the start.
    fn set_up(&mut self, i: usize) {
        let l = &TRIALS[i];
        self.a = Mat2::from_cols_array(&l.cols);
        self.x0 = DVec2::new(l.x0[0], l.x0[1]);
        (self.view_k, self.home_k, self.plan) = (l.view, l.view, l.plan);
        (self.show_parts, self.ring) = (true, false);
        self.rate = l.rate;
        self.fresh = 0.0;
        self.refresh();
        self.play_from_start(Some(l.plan));
    }

    /// Back to x₀, and play from there (up to `stop`, if given).
    fn play_from_start(&mut self, stop: Option<usize>) {
        self.rewind();
        self.playing = true;
        self.stop_at = stop;
        // (a moment to see x₀ before the first step)
        self.owed = 1.0 - 0.6 * self.rate;
    }

    /// Back to the start: the dot runs back along its path.
    fn rewind(&mut self) {
        self.k = 0;
        self.playing = false;
        self.view_k = self.home_k;
        (self.new_view, self.quiet_next) = (true, true);
    }

    fn step(&mut self) {
        self.refresh();
        if self.k < self.run.last() {
            self.k += 1;
        }
    }

    /// Play on, or pause. At the end of the run, play starts again from x₀.
    fn toggle_play(&mut self) {
        self.refresh();
        if self.playing {
            self.playing = false;
        } else if self.k >= self.run.last() {
            self.play_from_start(None);
        } else {
            (self.playing, self.stop_at, self.owed) = (true, None, 1.0);
        }
    }

    /// A new matrix typed or chosen: the run is played again from x₀.
    fn new_matrix(&mut self) {
        self.refresh();
        self.fresh = 0.0;
        (self.plan, self.view_k, self.home_k) = (10, 10, 10);
        self.play_from_start(Some(10));
    }

    /// Put x₀ on v₂'s line, as exactly as the numbers allow, keeping its length (and
    /// the side of the line it is nearer), and play many steps quickly.
    fn onto_losing_line(&mut self) {
        self.refresh();
        if let Some([_, v2]) = self.losing_line() {
            let len = self.x0.length().max(1e-3);
            let side = if self.x0.dot(v2) < 0.0 { -1.0 } else { 1.0 };
            self.x0 = v2 * (len * side);
            self.refresh();
            // as many steps as it takes for the part along v₁ to win, or better, to
            // grow back to a size that can be seen; played in about six seconds
            let steps = self.steps_to_win();
            (self.plan, self.rate) = (steps, (steps as f32 / 6.0).clamp(4.0, 15.0));
            self.play_from_start(Some(steps));
        }
    }

    /// How many steps to play from a start on v₂'s line: until the part along v₁ is
    /// half as long as x₀ was, and a few more; else until it is the larger part, and
    /// a dozen more; else to the end.
    fn steps_to_win(&self) -> usize {
        let Some(v) = self.losing_line() else { return MOST_STEPS };
        let len = self.x0.length();
        let parts: Vec<[f64; 2]> = self.run.x.iter().map(|x| parts(v, *x)).collect();
        let back = parts.iter().position(|[a, _]| a.abs() >= 0.5 * len).map(|k| k + 6);
        let wins = parts.iter().position(|[a, b]| a.abs() > b.abs()).map(|k| k + 12);
        back.or(wins).unwrap_or(MOST_STEPS).min(self.run.last())
    }

    /// v₁ and v₂, when there are two lines and v₁'s eigenvalue is the larger in size.
    fn losing_line(&self) -> Option<[DVec2; 2]> {
        let e = eigen(self.key.0);
        e.winner().and(e.lines())
    }

    /// For a Markov matrix, the total x + y of a start, which every step keeps (each
    /// column adds to 1): None for any other matrix, or a total of 0.
    fn kept_total(&self, x0: DVec2) -> Option<f64> {
        let total = x0.x + x0.y;
        (self.key.0.markov() && total.abs() > 1e-9).then_some(total)
    }

    /// Where a Markov chain with this total settles: on v₁'s line (λ₁ = 1), with the
    /// same total. None unless the other eigenvalue is smaller than 1 in size.
    fn steady(&self, total: f64) -> Option<DVec2> {
        match self.eigen() {
            Eigen::Real { l, v } if self.key.0.markov() && (l[0] - 1.0).abs() < 1e-6 && l[1].abs() < 1.0 - 1e-9 => Some(v[0] * (total / (v[0].x + v[0].y))),
            _ => None,
        }
    }

    /// Work the run out again if the matrix or the start has changed.
    fn refresh(&mut self) {
        let key = (M2::of(self.a), self.x0);
        if key != self.key || self.run.x.is_empty() {
            self.run = Run::of(key.0, key.1);
            self.key = key;
        }
        self.k = self.k.min(self.run.last());
    }

    fn eigen(&self) -> Eigen {
        eigen(self.key.0)
    }

    /// Move the play on by `dt` seconds of scene time.
    fn advance(&mut self, dt: f32) {
        if !self.playing || dt <= 0.0 {
            return;
        }
        let end = self.stop_at.map_or(self.run.last(), |s| s.min(self.run.last()));
        // (no burst of steps after a long frame)
        self.owed = (self.owed + dt * self.rate).min(2.0);
        while self.owed >= 1.0 && self.k < end {
            self.owed -= 1.0;
            self.k += 1;
        }
        if self.k >= end {
            (self.playing, self.stop_at) = (false, None);
        }
    }

    /// Bring the step drawn `dt` seconds nearer to the step taken; with `at_once`, all
    /// the way.
    fn follow(&mut self, dt: f64, at_once: bool) {
        let target = self.k as f64;
        let tau = FOLLOW.min(0.3 / self.rate.max(0.1) as f64);
        let e = 1.0 - (-dt.max(0.0) / tau).exp();
        self.between += (target - self.between) * e;
        self.shown += (self.between - self.shown) * e;
        // (exactly there in the end, not nearly: the picture then stops changing)
        if at_once || !self.shown.is_finite() || ((target - self.shown).abs() < 2e-3 && (target - self.between).abs() < 2e-3) {
            (self.between, self.shown) = (target, target);
        }
    }

    /// Is anything still on its way?
    fn moving(&self) -> bool {
        self.playing
            || self.shown != self.k as f64
            || self.between != self.k as f64
            || self.x0_drawn != self.x0
            || self.fresh < 1.0
            || self.quiet_axes > 0.0
            || self.span != self.span_wanted()
            || self.ranges != self.ranges_wanted()
    }

    /// The steps across the charts: at least 20 and the experiment's, and the steps
    /// taken, in tens.
    fn span_wanted(&self) -> f32 {
        let most = self.plan.max(self.k + 1).max(20);
        (most.div_ceil(10) * 10) as f32
    }

    /// What each chart plots, for k = 0 to the end of the run: the size of x, and the
    /// sizes of its parts along v₁ and v₂ when there are two lines; and the angle
    /// between x and the line it turns to, when there is one.
    fn series(&self) -> ([Vec<f64>; 3], Option<Vec<f64>>) {
        let x = &self.run.x;
        let size: Vec<f64> = x.iter().map(|p| p.length()).collect();
        let (along1, along2) = match self.eigen().lines() {
            Some(v) => x.iter().map(|p| parts(v, *p)).map(|[a, b]| (a.abs(), b.abs())).unzip(),
            None => (Vec::new(), Vec::new()),
        };
        let angle = self.eigen().winner().map(|v| x.iter().map(|p| if p.length() > 0.0 { angle_to_line(*p, v) } else { f64::NAN }).collect());
        ([size, along1, along2], angle)
    }

    /// The powers of ten each chart should run between, to hold what it plots up to
    /// the end of its span.
    fn ranges_wanted(&self) -> [[f32; 2]; 2] {
        let n = self.span_wanted() as usize;
        let (sizes, angle) = self.series();
        let span_of = |values: &mut dyn Iterator<Item = f64>| {
            let (mut lo, mut hi) = (f64::MAX, f64::MIN);
            for v in values.filter(|v| *v > 0.0 && v.is_finite()) {
                lo = lo.min(v.log10());
                hi = hi.max(v.log10());
            }
            (lo <= hi).then_some((lo, hi))
        };
        let first = span_of(&mut sizes.iter().flat_map(|s| s.iter().take(n + 1).copied()));
        let first = first.map_or([-1.0, 1.0], |(lo, hi)| {
            let hi = hi.ceil().max(lo.floor() + 1.0);
            [lo.floor().max(hi - 24.0) as f32, hi as f32]
        });
        let second = angle.as_ref().and_then(|a| span_of(&mut a.iter().take(n + 1).copied()));
        let second = second.map_or([-1.0, 90f32.log10()], |(lo, _)| [(lo.floor().max(-20.0) as f32).min(0.0), 90f32.log10()]);
        [first, second]
    }

    /// Everything the view should hold: 0, x₀ and the steps up to `upto`, the ring,
    /// and the line of shares of a Markov chain.
    fn fit_box(&self, upto: usize) -> (Vec2, Vec2) {
        let mut pts: Vec<DVec2> = vec![DVec2::ZERO];
        pts.extend(self.run.x.iter().take(upto.min(self.run.last()) + 1));
        if self.ring {
            let r = self.x0.length();
            pts.extend([DVec2::X * r, -DVec2::X * r, DVec2::Y * r, -DVec2::Y * r]);
        }
        if let Some(total) = self.kept_total(self.x0) {
            pts.extend([DVec2::X * total, DVec2::Y * total]);
        }
        let (lo, hi) = pts.iter().filter(|p| p.is_finite()).fold((DVec2::ZERO, DVec2::ZERO), |(lo, hi), p| (lo.min(*p), hi.max(*p)));
        (lo.as_vec2(), hi.as_vec2())
    }

    /// The view that fits the run up to `view_k` (and the next step) into the part of
    /// the canvas the picture has to itself, with room round it for the names.
    fn the_view(&self) -> View {
        let lay = Layout::of(self.canvas, self.inset);
        let (lo, hi) = self.fit_box(self.view_k.max(self.k) + 1);
        let mut size = hi - lo;
        // (not a sliver: a flat run still gets some height, and an empty one some room)
        size = size.max(Vec2::splat(size.max_element() * 0.3)).max(Vec2::splat(1.0));
        let (min, max) = lay.plane;
        let margin = if lay.narrow { Vec2::new(70.0, 70.0) } else { Vec2::new(150.0, 110.0) };
        let room = (max - min - margin).max(Vec2::splat(40.0));
        let per_unit = (room.x / size.x).min(room.y / size.y).min(MOST_PER_UNIT).max(1e-7);
        let centre = (lo + hi) * 0.5;
        let off = ((min + max) * 0.5 - self.canvas * 0.5) / per_unit;
        View::Fit { center: Vec2::new(centre.x - off.x, centre.y + off.y), half: self.canvas.max(Vec2::ONE) * (0.5 / per_unit) }
    }
}

/// Is A a turn times a stretch (a = d, b = −c)? Then every step turns every x by the
/// same angle θ and multiplies every length by r; otherwise x goes round an ellipse,
/// unevenly, θ a step on average.
fn turns_evenly(m: M2) -> bool {
    let tol = 1e-6 * m.size();
    (m.a - m.d).abs() <= tol && (m.b + m.c).abs() <= tol
}

/// What the numbers say about the run at step k: lines of text, each in a colour.
fn sums_lines(run: &Run, m: M2, k: usize) -> Vec<(String, Color)> {
    let e = eigen(m);
    let x = run.at(k);
    let mut lines = vec![(format!("$x_{{{k}}} = A^{{{k}}} x_0 = {}$", point(x)), X_COL)];
    if x == DVec2::ZERO && k > 0 && run.at(0) != DVec2::ZERO {
        lines.push(("$A$ has sent $x$ to $0$, and $0$ stays $0$".into(), pal::TEXT));
    }
    match e {
        Eigen::Real { l, v } => {
            let [a, b] = parts(v, x);
            let sign = |v: f64, first: bool| match (v < 0.0 && sig(v, 3) != "0", first) {
                (true, _) => format!("− {}", sig(-v, 3)),
                (false, true) => sig(v, 3),
                (false, false) => format!("+ {}", sig(v, 3)),
            };
            // (each line goes on from the one above it: its sign is spaced as between
            // two terms, not as the sign of a number; and the factor, which may be
            // negative, is a group, so that its sign is a number's and not a minus)
            lines.push((format!(r"${{}} = {} v_1$   ($\times{{{}}}$ a step)", sign(a, true), sig(l[0], 3)), V1));
            lines.push((format!(r"${{}} {} v_2$   ($\times{{{}}}$ a step)", sign(b, false), sig(l[1], 3)), V2));
        }
        Eigen::Defective { l, .. } => {
            lines.push((format!(r"$\lambda = {}$ twice, but only one eigenvector line", sig(l, 3)), V1));
            lines.push(("so no second part to write $x$ with".into(), pal::DIM));
        }
        Eigen::Scalar { l } => {
            lines.push((format!("$A = {} I$: every line through $0$ is an", sig(l, 3)), pal::TEXT));
            lines.push((format!(r"eigenvector line, and $x$ is just $\times{{{}}}$", sig(l, 3)), pal::DIM));
        }
        Eigen::Complex { r, theta } => {
            let deg = degrees(theta.to_degrees());
            lines.push((format!(r"$\lambda = {} (\cos {deg} \pm i \sin {deg})$", sig(r, 3)), pal::TEXT));
            if turns_evenly(m) {
                lines.push((format!(r"each step: a turn of ${deg}$, length $\times {}$", sig(r, 3)), pal::DIM));
            } else {
                lines.push((format!(r"each step: ${deg}$ round the ellipse, $\times {}$", sig(r, 3)), pal::DIM));
            }
        }
    }
    let growth = match run.growth(k) {
        Some(g) => format!(r"growth this step: $\lVert x_{{{k}}} \rVert / \lVert x_{{{}}} \rVert = {}$", k - 1, sig(g, 4)),
        None if k == 0 => "growth: none yet (step to see it)".into(),
        None => "growth: none ($x$ was $0$)".into(),
    };
    lines.push((growth, pal::DIM));
    lines
}

/// The words for where the run ends, if step k is its end.
fn the_end(run: &Run, k: usize) -> Option<String> {
    (k >= run.last()).then(|| if run.far { format!("the end: $x_{{{k}}}$ is too far out to follow") } else { format!("the end: {MOST_STEPS} steps") })
}

impl Scene for Powers {
    fn view(&self) -> View {
        self.the_view()
    }

    fn take_view(&mut self) -> Option<View> {
        // (the numbers along the axes are left out while the view glides to a new
        // lesson: they would come and go as the scale changes)
        let new = std::mem::take(&mut self.new_view);
        if new && std::mem::take(&mut self.quiet_next) {
            self.quiet_axes = QUIET;
        }
        new.then(|| self.the_view())
    }

    fn settings(&self, s: &mut ViewSettings) {
        s.grid_alpha = 0.1;
    }

    fn animated(&self) -> bool {
        self.moving()
    }

    fn ui(&mut self, ui: &mut egui::Ui, _: &Gfx) {
        self.refresh();
        ui::note(ui, r"Take a start $x_0$ and apply a matrix $A$ to it again and again: $x_1 = Ax_0$, $x_2 = Ax_1 = A^2 x_0$, and so on. Written in $A$'s eigenvectors, $x_0 = c_1 v_1 + c_2 v_2$, every step multiplies each part by its own eigenvalue:");
        ui::equation(ui, r"x_k = A^k x_0 = \textcolor{orange}{c_1 \lambda_1^k v_1} + \textcolor{purple}{c_2 \lambda_2^k v_2}");
        ui::note(ui, r"The part whose $\lambda$ is larger in size takes over. Drag $x_0$, and step with the buttons over the picture.");

        ui::section(ui, "Things to try");
        for (i, l) in TRIALS.iter().enumerate() {
            if ui::experiment(ui, l.name, l.caption) {
                self.set_up(i);
            }
        }

        ui::section(ui, "Steps");
        ui.horizontal_wrapped(|ui| {
            if ui.add_enabled(self.k < self.run.last(), egui::Button::new("step ▶")).clicked() {
                self.step();
            }
            if ui.button(if self.playing { "❚❚ pause" } else { "▶ play" }).clicked() {
                self.toggle_play();
            }
            if ui.button("↺ back to the start").clicked() {
                self.rewind();
            }
        });
        ui::slider_log(ui, "steps per second", &mut self.rate, 0.5..=15.0);
        ui::note(ui, &format!("Play stops by itself after {MOST_STEPS} steps, or sooner once $x$ is a million times as long as $x_0$: the picture can follow it no further."));

        ui::section(ui, "Matrix $A$");
        let mut fresh = false;
        ui.horizontal_wrapped(|ui| {
            for (name, cols) in PRESETS {
                if ui.button(egui::RichText::new(name).size(12.5)).clicked() {
                    self.a = Mat2::from_cols_array(&cols);
                    fresh = true;
                }
            }
        });
        fresh |= mat2_edit(ui, &mut self.a);
        if fresh {
            self.new_matrix();
        }
        ui::note(ui, r"turn 45°: both eigenvalues have size 1; flip: $\lambda = 1$ and $-1$; squash: $\lambda = 1$ and $0$; zero: everything goes to $0$. Type a number to play the steps again; drag one sideways and the dots follow.");

        ui::section(ui, "Start $x_0$");
        let losing = self.losing_line().is_some();
        if ui.add_enabled(losing, egui::Button::new(ui::label_text(ui, "put $x_0$ exactly on $v_2$'s line", 12.5, None, false))).clicked() {
            self.onto_losing_line();
        }
        ui::note(ui, r"Then plays fast. In exact arithmetic $x$ would stay on that line for ever. But the computer keeps about 16 digits, so $x_0$ and every step are rounded, and that usually leaves a part along $v_1$ of about $10^{-16}$ of $x$. Every step it grows by $|\lambda_1| / |\lambda_2|$ against the other part: on the chart the orange line climbs towards the purple one, and given enough steps it wins. (Only when $A$ has two eigenvector lines, and one $\lambda$ larger in size.)");

        ui::section(ui, "Show");
        ui.checkbox(&mut self.show_parts, ui::label_text(ui, r"the two parts, $\textcolor{orange}{c_1 \lambda_1^k v_1}$ and $\textcolor{purple}{c_2 \lambda_2^k v_2}$", 12.5, None, false));
        if ui.checkbox(&mut self.ring, "a ring of starts, carried along").changed() {
            self.new_view = true;
        }
        ui::note(ui, "The ring: every start as far from $0$ as $x_0$, all stepped together, to show what $A$ does to the whole plane.");

        ui::section(ui, "Reading the picture");
        ui::key(
            ui,
            &[
                (X_COL, r"white dots: $x_0, x_1, x_2, \ldots$, joined in order; $x_0$ is the handle"),
                (V1, "orange: $v_1$'s line (dashed), the eigenvector of the eigenvalue largest in size, and the part of $x$ along it"),
                (V2, "purple: $v_2$'s line and the part of $x$ along it; the two parts add up to $x$"),
                (RING_COL, "cyan: the ring of starts, when shown"),
                (SHARES, "green, when $A$ is a Markov matrix: the line of points with the same total $x + y$ as $x_0$, which every step keeps"),
                (pal::DIM, "grey dashed curve, for complex eigenvalues: the ellipse $x$ goes round, which grows or shrinks by $r$ every step"),
                (ANGLE, "yellow, in the lower chart: the angle between $x$ and $v_1$'s line"),
            ],
        );
        ui::section(ui, "Words");
        ui::term(ui, "Eigenvector, eigenvalue:", r"a direction $v$ that $A$ does not turn: $Av = \lambda v$. A vector along it is only multiplied by $\lambda$ (reversed if $\lambda < 0$).");
        ui::term(ui, "Power method:", r"finding the eigenvector of the eigenvalue largest in size by applying $A$ again and again (rescaling $x$ to length 1 now and then, to keep the numbers in range). It closes in by the factor $|\lambda_2| / |\lambda_1|$ a step.");
        ui::term(ui, "Log scale:", "each mark up is ten times the one below, so a number multiplied by the same factor every step draws a straight line.");
        ui::term(ui, "Complex eigenvalues:", r"$\lambda = r(\cos\theta \pm i \sin\theta)$. No direction is kept; each step takes $x$ round an ellipse, onto one $r$ times as large, by an angle that averages $\theta$: exactly $\theta$ at every step when $A$ is a turn times a stretch (as in the spiral), otherwise more at some steps and less at others.");
        ui::term(ui, "Markov matrix:", r"entries $\ge 0$ and columns that add to 1: column $j$ says in what shares whatever is in place $j$ is passed on in one step. Nothing is lost, so the total stays the same.");
        ui::term(ui, "Steady state:", r"what $A$ leaves as it is: the eigenvector with $\lambda = 1$, scaled to the total there is.");

        ui::section(ui, "Numbers");
        let k = self.k;
        ui::readout(ui, "$k$, the steps taken", format!("${k}$"));
        let e = self.eigen();
        match e {
            Eigen::Real { l, .. } => {
                ui::readout_colored(ui, r"$\lambda_1$", format!("${}$", sig(l[0], 4)), V1);
                ui::readout_colored(ui, r"$\lambda_2$", format!("${}$", sig(l[1], 4)), V2);
                ui::readout(ui, r"$|\lambda_2| / |\lambda_1|$, the closing-in factor", if l[0] != 0.0 { format!("${}$", sig(l[1].abs() / l[0].abs(), 4)) } else { "none".into() });
            }
            Eigen::Defective { l, .. } => ui::readout(ui, r"$\lambda$", format!("${}$ twice, one line", sig(l, 4))),
            Eigen::Scalar { l } => ui::readout(ui, r"$\lambda$", format!("${}$ twice, every line", sig(l, 4))),
            Eigen::Complex { r, theta } => {
                ui::readout(ui, r"$\lambda = r(\cos\theta \pm i \sin\theta)$", format!(r"$r = {}$, $\theta = {}$", sig(r, 4), degrees(theta.to_degrees())));
            }
        }
        ui::readout(ui, &format!("$x_{{{k}}}$"), format!("${}$", point(self.run.at(k))));
        ui::readout(ui, &format!(r"$\lVert x_{{{k}}} \rVert$"), format!("${}$", sig(self.run.at(k).length(), 4)));
        if let Some(g) = self.run.growth(k) {
            ui::readout(ui, "growth this step", format!("${}$", sig(g, 5)));
        }
        if let Some(v) = e.winner() {
            let x = self.run.at(k);
            ui::readout_colored(ui, "angle from $v_1$'s line", if x.length() > 0.0 { format!("${}$", degrees(angle_to_line(x, v))) } else { "none".into() }, ANGLE);
        }
        let p = self.run.power(k).rows();
        let entries = |row: [f64; 2]| row.map(|v| sig(v, 4)).join(" & ");
        ui::readout(ui, &format!("$A^{{{k}}}$"), format!(r"$\begin{{bmatrix}} {} \\ {} \end{{bmatrix}}$", entries(p[0]), entries(p[1])));
        if let Some(end) = the_end(&self.run, k) {
            ui::note(ui, &end);
        }
        ui::see_also(ui, &[("eigen", "how eigenvalues are found"), ("basis", "writing a vector in another basis"), ("expm", "flows: $e^{At}$ instead of $A^k$"), ("lyapunov", "growth through many different matrices"), ("linmap2", "one map at a time")]);
    }

    fn frame(&mut self, f: &mut Frame) {
        // the canvas decides where the panels go, and with them where the picture is
        let view = f.view_size();
        // (a canvas squeezed to a sliver is one the controls have opened over, on a
        // phone: the picture is not seen then, and is left as it was for when they close)
        let covered = view.x < 160.0;
        let lay = Layout::of(view, f.top_inset);
        let before = Layout::of(self.canvas, self.inset);
        let changed = !covered && (lay.narrow != before.narrow || lay.plane != before.plane);
        if !covered && (changed || !self.measured) {
            self.new_view |= changed || (view - self.canvas).abs().max_element() > 24.0;
            (self.canvas, self.inset, self.measured) = (view, f.top_inset, true);
        }

        self.refresh();
        let pointer = f.ptr.pos.map(|p| p / f.px);
        if pointer.is_some_and(|p| lay.covers(p)) {
            // the panel is for its buttons: no panning from there, and two quick
            // clicks are two steps, not a reset of the view
            f.handles.capture();
            f.handles.claim_clicks();
        }

        // x₀'s handle first, so that everything is drawn where it now is
        let mut grip = self.x0_drawn.as_vec2();
        let hidden = on_screen(f, grip.extend(0.0)).is_some_and(|p| lay.covers(p));
        if (!hidden || f.handles.dragging()) && f.drag2(1, &mut grip, X_COL) {
            self.x0 = grip.as_dvec2().clamp(DVec2::splat(-REACH), DVec2::splat(REACH));
            self.x0_drawn = self.x0;
            self.refresh();
        }
        // x₀ let go with x now out of the picture: the view brings it back
        let dragging = f.handles.dragging();
        if self.dragging && !dragging {
            let (lo, hi) = lay.plane;
            let now = on_screen(f, self.run.at(self.k).as_vec2().extend(0.0));
            if now.is_none_or(|p| !(p.cmpge(lo).all() && p.cmple(hi).all())) {
                self.view_k = self.view_k.max(self.k);
                self.new_view = true;
            }
        }
        self.dragging = dragging;

        let dt = f.dt;
        self.advance(dt);
        self.follow(dt as f64, dt <= 0.0);
        // what glides by the clock: x₀ to an experiment's start, the lines fading in,
        // the charts to their new span and range
        let real = f.real_dt.max(0.0);
        let ease = |tau: f32| 1.0 - (-real / tau).exp();
        let e = ease(0.09) as f64;
        self.x0_drawn += (self.x0 - self.x0_drawn) * e;
        if (self.x0 - self.x0_drawn).length() < 1e-4 * self.x0.length().max(1e-3) {
            self.x0_drawn = self.x0;
        }
        self.fresh = (self.fresh + real / 0.45).min(1.0);
        let (span, ranges) = (self.span_wanted(), self.ranges_wanted());
        let e = ease(RESCALE);
        self.span += (span - self.span) * e;
        if (span - self.span).abs() < 0.02 {
            self.span = span;
        }
        for (now, want) in self.ranges.iter_mut().zip(ranges) {
            for (a, b) in now.iter_mut().zip(want) {
                *a += (b - *a) * e;
                if (b - *a).abs() < 0.005 {
                    *a = b;
                }
            }
        }

        // The panels' words first and the picture's after them, the axes' numbers
        // last: what comes and goes in one then never moves the others along.
        self.panels(f, &lay, pointer);
        let labels_before = f.dl.labels.len();
        let (lo, hi) = lay.plane;
        let caption = Caption::of(f, &self.caption_parts(lay.narrow), lo.x + 16.0, hi.x - 16.0, hi.y - 14.0);
        let mut kept = std::mem::take(&mut self.kept);
        self.picture(f, &lay, caption.covers(), &mut kept);
        self.kept = kept;

        // a step that would take x out of the picture brings the view along (checked
        // once a step, so that a view zoomed in on purpose is left alone)
        if self.k != self.followed {
            self.followed = self.k;
            let next = self.run.at(self.k + 1).as_vec2();
            let (lo, hi) = (lay.plane.0 + Vec2::splat(30.0), lay.plane.1 - Vec2::splat(30.0));
            let outside = on_screen(f, next.extend(0.0)).is_none_or(|p| !(p.cmpge(lo).all() && p.cmple(hi).all()));
            if outside && self.k + 1 > self.view_k && !f.handles.dragging() {
                self.view_k = (self.k + 1).min(self.run.last());
                self.new_view = true;
            }
        }

        caption.draw(f);
        // (the numbers come back once the view is still, or after at most `QUIET`)
        let bounds = f.cam.flat_bounds();
        if self.quiet_axes > 0.0 {
            let still = bounds == self.last_bounds && self.quiet_axes < QUIET - 0.1;
            self.quiet_axes = if still { 0.0 } else { self.quiet_axes - f.real_dt.max(0.0) };
        }
        self.last_bounds = bounds;
        if self.quiet_axes <= 0.0 {
            f.axes2();
        }
        // nothing written into the picture shows through the panels (the empty places
        // kept for names stay, so that the names keep theirs)
        if let Some((lo, hi)) = lay.backing {
            let (lo, hi) = (lo - Vec2::splat(4.0), hi + Vec2::splat(4.0));
            let mut i = 0;
            f.dl.labels.retain(|l| {
                i += 1;
                i <= labels_before || l.text.is_empty() || !(l.pos.cmpge(lo).all() && l.pos.cmple(hi).all())
            });
        }
    }
}

impl Powers {
    /// The plane: the eigenvector lines, the ring, the dots and their path, the two
    /// parts of x, and the names of all of them.
    fn picture(&self, f: &mut Frame, lay: &Layout, caption: Option<(Vec2, Vec2)>, kept: &mut Kept) {
        let run = &self.run;
        let e = self.eigen();
        let s = self.shown;
        let mut room = Room::new(f);
        for (lo, hi) in [lay.backing, caption].into_iter().flatten() {
            room.wall(lo, hi);
        }
        // the part of the plane in view, in the scene's units
        let (blo, bhi) = f.cam.flat_bounds();
        let (wlo, whi) = {
            let (a, b) = (world_at(f, lay.plane.0 + Vec2::splat(20.0)), world_at(f, lay.plane.1 - Vec2::splat(20.0)));
            (a.min(b), a.max(b))
        };
        let reach = (bhi - blo).length() * 2.0 + 10.0;
        let fade = smoothstep(self.fresh);
        // The names wait until a new picture has nearly settled, and then fade in:
        // a name looking for room while everything moves at once would glide across
        // the picture to it.
        let named = self.fresh >= 0.6;
        let word = smoothstep((self.fresh - 0.6) / 0.4);

        // the eigenvector lines
        let lines: Vec<(DVec2, Color, String)> = match e {
            Eigen::Real { l, v } => vec![(v[0], V1, format!(r"$v_1$: $\lambda_1 = {}$", sig(l[0], 3))), (v[1], V2, format!(r"$v_2$: $\lambda_2 = {}$", sig(l[1], 3)))],
            Eigen::Defective { l, v } => vec![(v, V1, format!(r"$v_1$: $\lambda = {}$, twice; the only line", sig(l, 3)))],
            _ => Vec::new(),
        };
        for (v, c, _) in &lines {
            let d = v.as_vec2() * reach;
            f.dashed(-d, d, 1.6, c.a(0.8 * fade));
            room.line(f, (-d).extend(0.0), d.extend(0.0));
        }

        // a Markov chain: the line its total stays on, and where it settles
        let total = self.kept_total(self.x0_drawn).map(|t| t as f32);
        let steady = total.and_then(|t| self.steady(t as f64)).map(|p| p.as_vec2());
        let total_name = total.map(|t| format!("$x + y = {}$", sig(t as f64, 3)));
        if let Some(t) = total {
            f.dashed(Vec2::X * t, Vec2::Y * t, 2.0, SHARES.a(0.8 * fade));
            room.line(f, Vec3::X * t, Vec3::Y * t);
        }

        // the ring of starts, carried along
        let r0 = self.x0_drawn.length();
        let i = (s.floor() as usize).min(run.last());
        let t = s - i as f64;
        let (pi, pj) = (run.power(i), run.power(i + 1));
        let carried = |p: DVec2| pi.apply(p).lerp(pj.apply(p), t).as_vec2();
        if self.ring && r0 > 0.0 {
            let pts: Vec<Vec3> = (0..RING).map(|j| carried(DVec2::from_angle(j as f64 / RING as f64 * std::f64::consts::TAU) * r0).extend(0.0)).collect();
            f.dl.lines.closed(&pts, 1.6, RING_COL.a(0.75));
            for p in pts.iter().step_by(RING / RING_DOTS) {
                f.dot(*p, 6.0, RING_COL);
            }
            f.circle(Vec2::ZERO, Vec3::Z, r0 as f32, 1.0, RING_COL.a(0.25));
        }

        // for complex eigenvalues, the ellipse x goes round (a circle for a turn)
        let x_now = if i == 0 && t == 0.0 { self.x0_drawn } else { run.between(s) };
        // (as so many dashes whatever its size, each a piece of the curve: dashes
        // measured along the screen would come and go as it shrinks)
        if let (Eigen::Complex { .. }, Some(curve)) = (e, orbit(self.key.0, x_now)) {
            for dash in curve.chunks(4).filter(|c| c.len() == 4) {
                f.path(dash[..3].iter().map(|p| p.as_vec2()), 1.3, pal::DIM.a(0.8 * fade));
            }
        }

        // the path and the dots: x₀ as drawn, then the steps taken
        let at = |j: usize| if j == 0 { self.x0_drawn.as_vec2() } else { run.at(j).as_vec2() };
        let path = f.dl.lines_with(LineKind::Blend, LineStyle::default());
        path.brk();
        for j in 0..=i {
            path.push(at(j).extend(0.0), 1.6, X_COL.a(0.42));
        }
        path.push(x_now.as_vec2().extend(0.0), 1.6, X_COL.a(0.42));
        path.brk();
        for j in 1..=i {
            f.dot(at(j), 7.0, X_COL.a(0.85));
            room.line(f, at(j - 1).extend(0.0), at(j).extend(0.0));
        }
        room.line(f, at(i).extend(0.0), x_now.as_vec2().extend(0.0));

        // the two parts of x: along v₁ and along v₂, adding up to x
        let x_f = x_now.as_vec2();
        let mut part_names: Vec<(usize, Vec<Vec3>, String, Color, bool)> = Vec::new();
        if let (true, Some(v)) = (self.show_parts, e.lines()) {
            let [a, b] = parts(v, x_now);
            let (pa, pb) = ((v[0] * a).as_vec2(), (v[1] * b).as_vec2());
            // (two lines very nearly one have huge parts that nearly cancel: not drawn)
            if pa.length() <= 50.0 * x_f.length() + 1e-9 {
                for (p, c) in [(pa, V1), (pb, V2)] {
                    f.arrow(Vec2::ZERO, p, 2.6, c.a(fade));
                    f.dashed(p, x_f, 1.0, c.a(0.55 * fade));
                    room.line(f, Vec3::ZERO, p.extend(0.0));
                    room.line(f, p.extend(0.0), x_f.extend(0.0));
                }
                // Named later, with the rest: by the middle of the arrow, or beside it on
                // the side away from x, outside the parallelogram. Few places close
                // together, so that a name that has to move never has far to go; and
                // kept from the middle of the arrow itself, which does not jump across
                // when x crosses the arrow's line. (An arrow too short to see is not named.)
                let wpp = f.cam.world_per_px(Vec3::ZERO) * f.px;
                for (n, (p, c, text)) in [(pa, V1, format!(r"$c_1 \lambda_1^{{{}}} v_1$", self.k)), (pb, V2, format!(r"$c_2 \lambda_2^{{{}}} v_2$", self.k))].into_iter().enumerate() {
                    let mut places = vec![(p * 0.5).extend(0.0)];
                    places.extend(beside(f, Vec2::ZERO, p, &[0.5], Some(x_f), &text, 13.0));
                    part_names.push((n, places, text, c, p.length() >= 36.0 * wpp));
                }
            }
        }

        // the steady state of a Markov chain
        if let Some(st) = steady {
            let wpp = f.cam.world_per_px(Vec3::ZERO) * f.px;
            f.circle(st, Vec3::Z, 9.0 * wpp, 2.0, V1);
        }

        // the dot now, and x₀'s handle (drawn by drag2 before) over it
        f.glow(x_f, 26.0, X_COL.a(0.5));
        f.dot(x_f, 11.0, X_COL);

        if !named {
            return;
        }
        // The names, always in the same order, each in a place of its own whether it is
        // written or not (see `name`): first those that are always on the picture,
        // then those that can leave it.
        let out_of = |f: &Frame, p: Vec2| on_screen(f, p.extend(0.0)).zip(on_screen(f, Vec3::ZERO)).map_or(Vec2::new(1.0, -1.0), |(a, b)| a - b);
        let x0f = self.x0_drawn.as_vec2();
        let toward = out_of(f, x0f);
        name(&mut room, f, kept, Tag("x0", 0), &[x0f.extend(0.0)], "$x_0$", 16.0, X_COL.a(word), toward, 10.0, Some(false));
        // the lines: beside them, near the edge of the picture, on whichever side has room
        for (n, (v, c, text)) in lines.iter().enumerate() {
            let d = v.as_vec2();
            let mut places = Vec::new();
            for d in [d, -d] {
                if let Some(t) = exit(d, wlo, whi) {
                    places.extend(beside(f, Vec2::ZERO, d * t, &[0.9, 0.82], None, text, 14.0));
                }
            }
            name(&mut room, f, kept, Tag("line", n as u32), &places, text, 14.0, c.a(word), Vec2::ZERO, 0.0, Some(true));
        }
        if let Some(st) = steady {
            let text = format!("steady state ${}$", point(st.as_dvec2()));
            // (to the right of the point, between the line it is on and the line of
            // the total, which part there)
            if let Some(at) = on_screen(f, st.extend(0.0)) {
                let half = measure(f, &text, 13.0) * 0.5 + 4.0;
                let places: Vec<Vec3> = [8.0, 0.0, 16.0, -10.0].iter().map(|dy| world_at(f, at + Vec2::new(half + 16.0, *dy)).extend(0.0)).collect();
                name(&mut room, f, kept, Tag("steady", 0), &places, &text, 13.0, V1.a(word), Vec2::ZERO, 0.0, Some(true));
            }
        }
        if let (Some(t), Some(text)) = (total, &total_name) {
            let places = beside(f, Vec2::X * t, Vec2::Y * t, &[0.75, 0.85, 0.15, 0.25], None, text, 13.0);
            name(&mut room, f, kept, Tag("shares", 0), &places, text, 13.0, SHARES.a(word), Vec2::ZERO, 0.0, Some(true));
        }
        let toward = out_of(f, x_f);
        let now = format!("$x_{{{}}}$", self.k);
        name(&mut room, f, kept, Tag("xk", 0), &[x_f.extend(0.0)], &now, 16.0, X_COL.a(word), toward, 10.0, (self.k > 0).then_some(false));
        for (n, places, text, c, seen) in &part_names {
            name(&mut room, f, kept, Tag("part", *n as u32), places, text, 13.0, c.a(word), Vec2::ZERO, 0.0, seen.then_some(true));
        }
        // the first few steps, where there is room, once they have been taken (until
        // then their places go along with the dot)
        for j in 1..=4 {
            let p = if j <= i { at(j) } else { x_f };
            let toward = out_of(f, p);
            let text = format!("$x_{{{j}}}$");
            name(&mut room, f, kept, Tag("step", j as u32), &[p.extend(0.0)], &text, 13.0, X_COL.mix(pal::DIM, 0.4).a(word), toward, 7.0, (j <= i && j != self.k).then_some(true));
        }
    }

    /// What the picture shows that has no name written beside it (on a phone, only
    /// the dots: everything else there has its name beside it).
    fn caption_parts(&self, narrow: bool) -> Vec<&'static str> {
        let e = self.eigen();
        let mut parts = vec![r"white: $x_0$, $x_1 = Ax_0$, $x_2 = A^2 x_0$, $\ldots$"];
        if narrow {
            return parts;
        }
        match e {
            Eigen::Real { .. } => {
                parts.push("dashed: the eigenvector lines");
                if self.show_parts {
                    parts.push("arrows: the parts of $x$ along them");
                }
            }
            Eigen::Defective { .. } => parts.push("dashed: the only eigenvector line"),
            Eigen::Complex { .. } => parts.push("grey dashed: the ellipse $x$ goes round"),
            Eigen::Scalar { .. } => {}
        }
        if self.ring {
            parts.push("cyan: a ring of starts, carried along");
        }
        parts
    }

    /// The panels over the picture: the buttons, the numbers, the two charts.
    fn panels(&mut self, f: &mut Frame, lay: &Layout, pointer: Option<Vec2>) {
        if let Some((lo, hi)) = lay.backing {
            hud_panel(f, lo, hi);
        }
        if let Some(rects) = lay.button_rects() {
            let short = lay.buttons.is_some_and(|b| b.1);
            if hud_button(f, rects[0], "step ▶", self.k < self.run.last(), pointer) {
                self.step();
            }
            let play = if self.playing { "❚❚ pause" } else { "▶ play" };
            if hud_button(f, rects[1], play, true, pointer) {
                self.toggle_play();
            }
            if hud_button(f, rects[2], if short { "↺ start" } else { "↺ back to the start" }, self.k > 0 || self.shown > 0.0, pointer) {
                self.rewind();
            }
        }
        if let Some((at, width, size)) = lay.sums {
            self.sums(f, at, width, size);
        }
        let (sizes, angle) = self.series();
        let e = self.eigen();
        let shown = self.shown as f32;
        if let Some(area) = lay.charts[0] {
            let mut series: Vec<(&[f64], Color, &str)> = Vec::new();
            let title;
            let key: Vec<(String, Color)>;
            if let Eigen::Real { l, .. } = e {
                series.push((&sizes[0], X_COL.a(0.5), ""));
                series.push((&sizes[2], V2, ""));
                series.push((&sizes[1], V1, ""));
                title = "how long each part is (log scale)";
                // (shorter where the chart is narrow)
                let part = if area.1.x >= 200.0 { "part " } else { "" };
                key = vec![(format!(r"$v_1$ {part}$\times {}$", sig(l[0].abs(), 3)), V1), (format!(r"$v_2$ {part}$\times {}$", sig(l[1].abs(), 3)), V2), (r"$\lVert x \rVert$".into(), X_COL)];
            } else {
                series.push((&sizes[0], X_COL, ""));
                title = "how long $x$ is (log scale)";
                key = vec![(r"$\lVert x \rVert$".into(), X_COL)];
            }
            chart(f, area, self.span, self.ranges[0], &series, shown, title, &key, "");
        }
        if let Some(area) = lay.charts[1] {
            match &angle {
                Some(a) => {
                    let series = [(&a[..], ANGLE, "")];
                    chart(f, area, self.span, self.ranges[1], &series, shown, "angle between $x$ and $v_1$'s line", &[("degrees (log scale)".into(), ANGLE)], r"{}^\circ");
                }
                None => {
                    let (o, size) = area;
                    let why = match e {
                        Eigen::Complex { .. } => "Complex eigenvalues: no line stays on itself, so $x$ has no line to turn to: it keeps going round.",
                        Eigen::Scalar { .. } => "Every line is an eigenvector line: $x$ stays on the line it starts on.",
                        _ => "The two eigenvalues are the same size: neither part wins, and the direction of $x$ does not settle.",
                    };
                    f.text_px(o + Vec2::new(-38.0, -CHART_ABOVE + 2.0), "angle between $x$ and $v_1$'s line", 12.0, pal::TEXT, Align2::LEFT_TOP);
                    wrapped_px(f, o + Vec2::new(-38.0, 6.0), size.x + 38.0, why, 12.0, pal::DIM);
                }
            }
        }
    }

    /// k, Aᵏ and what x_k is made of, from `at` (top-left), `width` wide.
    fn sums(&self, f: &mut Frame, at: Vec2, width: f32, size: f32) {
        let k = self.k;
        let mut y = at.y;
        let head = match the_end(&self.run, k) {
            Some(end) => format!("$k = {k}$    {end}"),
            None => format!("$k = {k}$"),
        };
        f.text_px(Vec2::new(at.x, y), head, size + 1.0, pal::TEXT, Align2::LEFT_TOP);
        y += size + 6.0;
        let p = self.run.power(k).rows();
        let texts = together(&[p[0][0], p[0][1], p[1][0], p[1][1]]);
        let rows: Vec<Vec<(String, Color)>> = texts.chunks(2).map(|r| r.iter().map(|t| (format!("${t}$"), pal::TEXT)).collect()).collect();
        let widest = self.run.widest;
        let name = format!("$A^{{{k}}} =$");
        let name_w = measure(f, &name, size) + 10.0;
        let msize = matrix_size(2, 2, widest, size);
        f.text_px(Vec2::new(at.x, y + msize.y * 0.5), name, size, pal::TEXT, Align2::LEFT_CENTER);
        hud_matrix(f, Vec2::new(at.x + name_w, y), &rows, widest, size, pal::DIM);
        y += msize.y + 6.0;
        for (line, c) in sums_lines(&self.run, self.key.0, k) {
            // (a line too long for the panel is written smaller)
            let wide = measure(f, &line, size);
            let s = if wide > width { size * (width / wide).max(0.75) } else { size };
            f.text_px(Vec2::new(at.x, y), line, s, c, Align2::LEFT_TOP);
            y += size + 5.0;
        }
    }
}

/// A caption along the bottom of the picture, on a dark plate: `parts` side by side
/// as long as they fit between `left` and `right`, on further lines above when they do
/// not, the last ending at `bottom`. (Like `bottom_caption` in the toolkit, but within
/// the picture's own part of the canvas, and laid out before it is drawn, so that the
/// names in the picture can keep clear of it.)
struct Caption {
    lines: Vec<String>,
    /// The widest line's width.
    widest: f32,
    left: f32,
    bottom: f32,
}

const CAPTION_SIZE: f32 = 12.0;

impl Caption {
    fn of(f: &Frame, parts: &[&str], left: f32, right: f32, bottom: f32) -> Caption {
        let mut lines: Vec<String> = Vec::new();
        let width = |text: &str| measure(f, text, CAPTION_SIZE);
        if f.view_size().y >= 300.0 && right - left >= 120.0 {
            let gap = "      ";
            for part in parts {
                match lines.last_mut() {
                    Some(last) if width(last) + width(gap) + width(part) <= right - left => {
                        last.push_str(gap);
                        last.push_str(part);
                    }
                    _ => lines.push(part.to_string()),
                }
            }
        }
        let widest = lines.iter().map(|l| width(l)).fold(0.0, f32::max);
        Caption { lines, widest, left, bottom }
    }

    /// The box it covers, if anything.
    fn covers(&self) -> Option<(Vec2, Vec2)> {
        let height = self.lines.len() as f32 * (CAPTION_SIZE + 7.0);
        (!self.lines.is_empty()).then(|| (Vec2::new(self.left - 4.0, self.bottom - height - 4.0), Vec2::new(self.left + self.widest + 8.0, self.bottom + 4.0)))
    }

    fn draw(&self, f: &mut Frame) {
        let count = self.lines.len();
        for (k, line) in self.lines.iter().enumerate() {
            let up = (count - 1 - k) as f32 * (CAPTION_SIZE + 7.0);
            f.caption_px(Vec2::new(self.left, self.bottom - up), line.as_str(), CAPTION_SIZE, pal::DIM, Align2::LEFT_BOTTOM);
        }
    }
}

thread_local! {
    /// The widths of texts with mathematics in them, as `measure` found them, and at
    /// what size: the names and lines here are measured every frame, and a width
    /// looked up here costs a fraction of one asked of the typesetter.
    static MEASURED: std::cell::RefCell<std::collections::HashMap<String, (f32, f32)>> = std::cell::RefCell::new(std::collections::HashMap::new());
}

/// How wide a text is drawn, logical pixels: one with mathematics in it measured as it
/// is set (and kept), plain words estimated as before.
fn measure(f: &Frame, text: &str, size: f32) -> f32 {
    if !crate::typeset::has_math(text) {
        return text_width(text, size);
    }
    MEASURED.with(|kept| {
        if let Some(&(at, width)) = kept.borrow().get(text) {
            if at == size {
                return width;
            }
        }
        // (nothing to keep before the fonts are in place: estimated till then)
        let Some(set) = crate::typeset::measure(f.ctx, text, size) else { return text_width(text, size) };
        let mut kept = kept.borrow_mut();
        if kept.len() >= 1024 {
            kept.clear();
        }
        kept.insert(text.to_string(), (size, set.x));
        set.x
    })
}

/// Text in screen space broken into lines no wider than `width` pixels.
fn wrapped_px(f: &mut Frame, at: Vec2, width: f32, text: &str, size: f32, color: Color) {
    let mut lines = vec![String::new()];
    for word in text.split(' ') {
        let last = lines.last_mut().expect("there is always a line");
        if last.is_empty() {
            last.push_str(word);
        } else if measure(f, last, size) + measure(f, word, size) + size * 0.47 > width {
            lines.push(word.to_string());
        } else {
            last.push(' ');
            last.push_str(word);
        }
    }
    for (i, line) in lines.into_iter().enumerate() {
        f.text_px(at + Vec2::new(0.0, i as f32 * (size + 4.0)), line, size, color, Align2::LEFT_TOP);
    }
}

/// How far along `d` the line from 0 goes before it leaves the box `lo..hi` (None if
/// 0 is not in the box).
fn exit(d: Vec2, lo: Vec2, hi: Vec2) -> Option<f32> {
    if !(lo.x < 0.0 && hi.x > 0.0 && lo.y < 0.0 && hi.y > 0.0) {
        return None;
    }
    let mut t = f32::MAX;
    for (dv, l, h) in [(d.x, lo.x, hi.x), (d.y, lo.y, hi.y)] {
        if dv > 0.0 {
            t = t.min(h / dv);
        } else if dv < 0.0 {
            t = t.min(l / dv);
        }
    }
    (t.is_finite() && t > 0.0).then_some(t)
}

/// Name a point of the picture with `Room::label_any` (`write` says whether the name
/// may be left out when there is no room for it; None: leave it out now).
///
/// A name not written still keeps a place, empty, where it was; and a name whose point
/// has left the canvas keeps an empty place that drifts, no faster than the eye can
/// follow, along with the point as far as the edge. Then the names after it are the
/// same names from frame to frame, and a name comes back about where it went. (`kept`
/// holds the point each was last drawn from, on the screen, and its offset from
/// there; a name never drawn yet has no place to keep.)
#[allow(clippy::too_many_arguments)]
fn name(room: &mut Room, f: &mut Frame, kept: &mut Kept, tag: Tag, points: &[Vec3], text: &str, size: f32, color: Color, towards: Vec2, gap: f32, write: Option<bool>) {
    let Tag(word, number) = tag;
    let view = f.view_size();
    let anchor = points.first().and_then(|p| on_screen(f, *p)).filter(|p| p.is_finite());
    let from = match anchor.filter(|p| p.cmpge(Vec2::ZERO).all() && p.cmple(view).all()) {
        Some(from) => {
            let written = write.is_some_and(|optional| room.label_any(f, tag, points, text, size, color, towards, gap, optional));
            if !written {
                let offset = f.places.chosen(word, number).unwrap_or(Vec2::ZERO);
                f.text_placed(word, number, from, offset, "", size, color);
            }
            from
        }
        None => {
            let Some((was, offset)) = kept.get(&(word, number)).copied() else { return };
            let edge = anchor.map_or(was, |p| p.max(Vec2::ZERO).min(view));
            let from = was + (edge - was).clamp_length_max(DRIFT * f.real_dt.max(0.0));
            f.text_placed(word, number, from, offset, "", size, color);
            from
        }
    };
    kept.insert((word, number), (from, f.places.chosen(word, number).unwrap_or(Vec2::ZERO)));
}

/// How fast an empty place kept for a name drifts after its point, logical pixels a
/// second.
const DRIFT: f32 = 500.0;

/// Where each name of the picture was last drawn from, and its offset (see `name`).
type Kept = std::collections::HashMap<(&'static str, u32), (Vec2, Vec2)>;

/// The world point at the screen point `px` (logical pixels) of a flat view.
fn world_at(f: &Frame, px: Vec2) -> Vec2 {
    let (lo, hi) = f.cam.flat_bounds();
    let view = f.view_size().max(Vec2::ONE);
    Vec2::new(lo.x + (hi.x - lo.x) * px.x / view.x, hi.y - (hi.y - lo.y) * px.y / view.y)
}

/// Places for a label beside the segment from `a` to `b` (in the scene) at these
/// shares of the way along, on either side of it, or only on the side away from
/// `away` if that is given: where the label's middle would be for its box to keep
/// just clear of the line, as points of the scene. (Few places, close together: a
/// label that has to move then never has far to go.)
fn beside(f: &Frame, a: Vec2, b: Vec2, shares: &[f32], away: Option<Vec2>, text: &str, size: f32) -> Vec<Vec3> {
    let (Some(sa), Some(sb)) = (on_screen(f, a.extend(0.0)), on_screen(f, b.extend(0.0))) else { return Vec::new() };
    let Some(n) = (sb - sa).perp().try_normalize() else { return Vec::new() };
    // (the box Room keeps clear for a label, and a little more)
    let half = Vec2::new(measure(f, text, size) * 0.5 + 4.0, size * 0.5 + 1.0);
    let off = n.x.abs() * half.x + n.y.abs() * half.y + 6.0;
    let sides: &[f32] = match away.and_then(|p| on_screen(f, p.extend(0.0))) {
        Some(p) if (p - sa).dot(n) > 0.0 => &[-1.0],
        Some(_) => &[1.0],
        None => &[-1.0, 1.0],
    };
    let mut places = Vec::new();
    for share in shares {
        let on = sa.lerp(sb, *share);
        for side in sides {
            places.push(world_at(f, on + n * (off * side)).extend(0.0));
        }
    }
    places
}

/// Points round the curve x goes round, in fours: three drawn and one left out.
const ORBIT: usize = 256;

/// For complex eigenvalues: the closed curve through x that A carries onto a copy of
/// itself r times as large, one θ further round (an ellipse; a circle for a turn).
/// In the real coordinates of an eigenvector u + iw, A turns and scales: going round
/// in those coordinates and back is the curve.
fn orbit(m: M2, x: DVec2) -> Option<Vec<DVec2>> {
    let Eigen::Complex { r, theta } = eigen(m) else { return None };
    let (re, im) = (r * theta.cos(), r * theta.sin());
    // an eigenvector for re + i im, from whichever row of A − λI is larger
    let (u, w) = if m.b.abs() >= m.c.abs() { (DVec2::new(m.b, re - m.a), DVec2::new(0.0, im)) } else { (DVec2::new(re - m.d, m.c), DVec2::new(im, 0.0)) };
    let den = u.perp_dot(w);
    if den == 0.0 || !den.is_finite() || x.length() == 0.0 {
        return None;
    }
    // x in those coordinates
    let (p, q) = (x.perp_dot(w) / den, u.perp_dot(x) / den);
    Some(
        (0..ORBIT)
            .map(|j| {
                let (sn, cs) = (j as f64 / ORBIT as f64 * std::f64::consts::TAU).sin_cos();
                u * (p * cs - q * sn) + w * (p * sn + q * cs)
            })
            .collect(),
    )
}

/// The powers of ten a chart can name, from 10⁻ᴰ to 10ᴰ.
const DECADES: i32 = 40;

/// A power of ten for an axis, as a formula: 1, 10, 0.1, 10⁶, 10⁻⁴ (every one of them
/// a formula, so that they are all in the same digits).
fn decade(e: i32, unit: &str) -> String {
    let text = match e {
        -2 => "0.01".to_string(),
        -1 => "0.1".to_string(),
        0 => "1".to_string(),
        1 => "10".to_string(),
        2 => "100".to_string(),
        _ => format!("10^{{{e}}}"),
    };
    format!("${text}{unit}$")
}

/// A chart on a log scale in screen space. `area` is the plot (top-left corner and
/// size); `span` the steps across it; `range` the powers of ten at its bottom and top;
/// each series its values for k = 0, 1, …, bright up to the step `shown` and faint
/// after it. A title and a key go above, the steps below; `unit` follows the numbers.
#[allow(clippy::too_many_arguments)]
fn chart(f: &mut Frame, (o, size): (Vec2, Vec2), span: f32, range: [f32; 2], series: &[(&[f64], Color, &str)], shown: f32, title: &str, key: &[(String, Color)], unit: &str) {
    let [lo, hi] = range;
    let height = (hi - lo).max(1e-3);
    let x_of = |k: f32| o.x + size.x * k / span.max(1.0);
    let y_of = |log: f32| o.y + size.y * (1.0 - ((log - lo) / height).clamp(0.0, 1.0));
    // the title and the key
    f.text_px(Vec2::new(o.x - 38.0, o.y - CHART_ABOVE + 2.0), title, 12.0, pal::TEXT, Align2::LEFT_TOP);
    let mut x = o.x - 38.0;
    for (text, c) in key {
        hud_rect(f, Vec2::new(x, o.y - 21.0), Vec2::new(x + 9.0, o.y - 12.0), *c);
        f.text_px(Vec2::new(x + 13.0, o.y - 16.5), text.as_str(), 11.0, pal::DIM, Align2::LEFT_CENTER);
        x += 13.0 + measure(f, text, 11.0) + 12.0;
    }
    // The powers of ten, as many named as there is room for, and the steps. Every
    // name has a place of its own kept for it, written or not, so that names coming
    // and going as the range glides never shift the others along. (The lines are
    // thin boxes for the same reason.)
    let first = lo.ceil() as i32;
    let last = hi.floor() as i32;
    let count = (last - first + 1).max(1) as f32;
    let room = (size.y / 22.0).max(1.0);
    let every = [1, 2, 4, 5, 10, 20].into_iter().find(|s| count / *s as f32 <= room).unwrap_or(40);
    for e in -DECADES..=DECADES {
        let y = y_of(e as f32);
        let inside = e >= first && e <= last;
        let named = inside && e.rem_euclid(every) == 0;
        if inside {
            hud_rect(f, Vec2::new(o.x, y - 0.5), Vec2::new(o.x + size.x, y + 0.5), pal::FAINT.a(if named { 0.7 } else { 0.3 }));
        }
        f.text_px(Vec2::new(o.x - 5.0, y), if named { decade(e, unit) } else { String::new() }, 10.5, pal::DIM, Align2::RIGHT_CENTER);
    }
    hud_rect(f, Vec2::new(o.x, o.y + size.y - 0.5), Vec2::new(o.x + size.x, o.y + size.y + 0.5), pal::DIM.a(0.8));
    let tick = if span <= 30.0 { 5 } else if span <= 60.0 { 10 } else { 20 };
    for k in (0..=MOST_STEPS + 20).step_by(5) {
        let shown = k % tick == 0 && k as f32 <= span + 1e-3;
        let x = x_of((k as f32).min(span));
        if shown {
            hud_rect(f, Vec2::new(x - 0.5, o.y + size.y), Vec2::new(x + 0.5, o.y + size.y + 4.0), pal::DIM);
        }
        f.text_px(Vec2::new(x, o.y + size.y + 5.0), if shown { k.to_string() } else { String::new() }, 10.5, pal::DIM, Align2::CENTER_TOP);
    }
    f.text_px(Vec2::new(o.x + size.x + 9.0, o.y + size.y + 5.0), "$k$", 11.0, pal::DIM, Align2::LEFT_TOP);
    // the step drawn
    let now = shown.min(span);
    hud_line(f, Vec2::new(x_of(now), o.y), Vec2::new(x_of(now), o.y + size.y), 1.0, pal::WHITE.a(0.35));
    // The series: each step's piece bright as far as the step drawn has got, and
    // faint beyond. Every piece is always drawn as the same two (one of them of no
    // length, and unseen), so that the lines only ever slide along as the step goes on.
    let most = span.floor() as usize;
    for (values, c, _) in series {
        let point = |k: usize| -> Option<Vec2> {
            let v = *values.get(k)?;
            (v > 0.0 && v.is_finite()).then(|| Vec2::new(x_of(k as f32), y_of(v.log10() as f32)))
        };
        let mut middle = None;
        for k in 0..most.min(values.len().saturating_sub(1)) {
            let (Some(a), Some(b)) = (point(k), point(k + 1)) else { continue };
            let m = a.lerp(b, (now - k as f32).clamp(0.0, 1.0));
            hud_line(f, a, m, 1.9, if m != a { *c } else { c.a(0.0) });
            hud_line(f, m, b, 1.2, if m != b { c.a(0.3) } else { c.a(0.0) });
            if now >= k as f32 && now <= (k + 1) as f32 {
                middle = Some(m);
            }
        }
        // (the value now, ringed; at the start, the first value)
        if let Some(m) = middle.or_else(|| point(0).filter(|_| now < 1.0)) {
            hud_circle(f, m, 2.5, 1.6, *c);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// An experiment as its button sets it up, played at 60 frames a second until it
    /// stops by itself.
    fn run(i: usize) -> Powers {
        let mut scene = Powers::create();
        scene.set_up(i);
        assert!(scene.playing && scene.k == 0);
        for _ in 0..60 * 60 {
            scene.advance(1.0 / 60.0);
            scene.follow(1.0 / 60.0, false);
            if !scene.playing && scene.shown == scene.k as f64 {
                break;
            }
        }
        assert!(!scene.playing, "{} still playing", TRIALS[i].name);
        scene
    }

    /// The two parts of x at step k, along v₁ and v₂.
    fn parts_at(s: &Powers, k: usize) -> [f64; 2] {
        parts(s.eigen().lines().expect("two eigenvector lines"), s.run.at(k))
    }

    /// The angle from one vector to the next, in degrees, anticlockwise.
    fn turn(a: DVec2, b: DVec2) -> f64 {
        a.perp_dot(b).atan2(a.dot(b)).to_degrees()
    }

    fn close(a: f64, b: f64, tol: f64) -> bool {
        (a - b).abs() <= tol
    }

    /// The opening lesson: each part multiplied by its own eigenvalue, the smaller one
    /// losing half its share every step, x falling in and then running out along v₁,
    /// 0.2° from that line at x₁₀ and growing by 1.196 a step, and straight lines on
    /// the log chart. (The scene opens on it.)
    #[test]
    fn the_largest_eigenvalue_wins() {
        let s = run(0);
        assert_eq!(s.k, 10);
        assert_eq!(Powers::create().key, s.key, "the scene opens on this lesson");
        let Eigen::Real { l, v } = s.eigen() else { panic!("{:?}", s.eigen()) };
        assert!(close(l[0], 1.2, 1e-6) && close(l[1], 0.6, 1e-6), "{l:?}");
        // mostly along v₂ to begin with
        let [a0, b0] = parts_at(&s, 0);
        assert!(b0.abs() > 4.0 * a0.abs(), "{a0} {b0}");
        for k in 0..10 {
            let ([a, b], [a1, b1]) = (parts_at(&s, k), parts_at(&s, k + 1));
            assert!(close(a1 / a, l[0], 1e-9) && close(b1 / b, l[1], 1e-9), "step {k}");
            // the purple part halves against the orange one
            assert!(close((b1 / a1) / (b / a), 0.5, 1e-6), "step {k}");
        }
        // in towards 0 for four steps, then out
        let len: Vec<f64> = (0..=10).map(|k| s.run.at(k).length()).collect();
        assert!((0..4).all(|k| len[k + 1] < len[k]) && (4..10).all(|k| len[k + 1] > len[k]), "{len:?}");
        let angle = angle_to_line(s.run.at(10), v[0]);
        assert!(close(angle, 0.2, 0.006), "{angle}");
        assert_eq!(sig(s.run.growth(10).expect("a step"), 4), "1.196");
        // straight on the log chart: the same rise or fall every step
        let (sizes, _) = s.series();
        for series in &sizes[1..] {
            for k in 0..18 {
                let d = |k: usize| series[k + 1].log10() - series[k].log10();
                assert!(close(d(k), d(k + 1), 1e-9), "step {k}");
            }
        }
        // every start off v₂'s line ends up the same way (a ring of them, a degree apart)
        for d in 0..360 {
            let start = DVec2::from_angle((d as f64 + 0.5).to_radians()) * 3.0;
            let run = Run::of(s.key.0, start);
            let k = 60.min(run.last());
            let (x, before) = (run.at(k), run.at(k - 1));
            assert!(angle_to_line(x, v[0]) < 1e-3 && close(x.length() / before.length(), 1.2, 1e-4) && x.length() > start.length(), "{start}");
        }
        // the numbers say so too
        let said: Vec<String> = sums_lines(&s.run, s.key.0, 10).into_iter().map(|l| l.0).collect();
        assert!(said[1].contains(r"\times{1.2}$ a step") && said[2].contains(r"\times{0.6}$ a step") && said[3].ends_with("= 1.196$"), "{said:?}");
    }

    /// The spiral: complex eigenvalues 0.9(cos 30° ± i sin 30°); every step turns x by
    /// 30° and multiplies its length by 0.9; twelve steps make a full turn and leave
    /// 0.28 of the length; every start spirals in.
    #[test]
    fn a_turn_and_a_shrink_make_a_spiral() {
        let s = run(1);
        assert_eq!(s.k, 24);
        let m = s.key.0;
        let Eigen::Complex { r, theta } = s.eigen() else { panic!("{:?}", s.eigen()) };
        assert!(close(r, 0.9, 1e-6) && close(theta.to_degrees(), 30.0, 1e-4), "{r} {theta}");
        assert!(turns_evenly(m) && s.eigen().winner().is_none() && s.eigen().lines().is_none());
        for d in 0..36 {
            let x = DVec2::from_angle((d as f64 * 10.0).to_radians());
            let y = m.apply(x);
            assert!(close(y.length(), 0.9, 1e-6) && close(turn(x, y), 30.0, 1e-4), "{x}");
        }
        for k in 0..24 {
            let (x, y) = (s.run.at(k), s.run.at(k + 1));
            assert!(close(y.length() / x.length(), 0.9, 1e-6) && close(turn(x, y), 30.0, 1e-4), "step {k}");
        }
        let (x0, x12) = (s.run.at(0), s.run.at(12));
        assert!(turn(x0, x12).abs() < 1e-3 && sig(x12.length() / x0.length(), 2) == "0.28", "{x12}");
        // every start in a ring goes the same way
        for d in 0..36 {
            let start = DVec2::from_angle((d as f64 * 10.0).to_radians()) * 3.0;
            let end = Run::of(m, start).at(24);
            assert!(close(end.length(), 3.0 * 0.9f64.powi(24), 1e-5), "{start}");
        }
        let said: Vec<String> = sums_lines(&s.run, m, 24).into_iter().map(|l| l.0).collect();
        assert!(said[2].contains(r"a turn of $30{}^\circ$, length $\times 0.9$"), "{said:?}");
        // and the curve drawn round 0 is the circle through x
        let curve = orbit(m, s.run.at(5)).expect("complex eigenvalues");
        assert!(curve.iter().all(|p| close(p.length(), s.run.at(5).length(), 1e-6)));
    }

    /// The Markov chain: columns that add to 1 (10% one way, 20% the other); the
    /// total stays 3; x settles at (2, 1) on v₁'s line, λ₁ = 1, closing in by 0.7 a
    /// step, from any start on the line.
    #[test]
    fn a_markov_chain_settles() {
        let s = run(2);
        assert_eq!(s.k, 12);
        let m = s.key.0;
        assert!(m.markov() && close(m.c, 0.1, 1e-7) && close(m.b, 0.2, 1e-7), "{m:?}");
        assert_eq!(s.x0, DVec2::new(0.0, 3.0));
        assert!(s.run.x.iter().all(|x| close(x.x + x.y, 3.0, 1e-5)));
        let Eigen::Real { l, v } = s.eigen() else { panic!() };
        assert!(close(l[0], 1.0, 1e-6) && close(l[1], 0.7, 1e-6), "{l:?}");
        let steady = s.steady(3.0).expect("a steady state");
        assert!(steady.distance(DVec2::new(2.0, 1.0)) < 1e-5 && steady.perp_dot(v[0]).abs() < 1e-9, "{steady}");
        let target = DVec2::new(2.0, 1.0);
        // (to the digits shown: the matrix is kept in single precision, so its λ₁ is
        // 1 to about eight digits, and (2, 1) is its steady state to about as many)
        for k in 0..20 {
            let ratio = s.run.at(k + 1).distance(target) / s.run.at(k).distance(target);
            assert!(close(ratio, 0.7, 1e-4), "step {k}: {ratio}");
        }
        for i in 0..=6 {
            let start = DVec2::new(i as f64 * 0.5, 3.0 - i as f64 * 0.5);
            let end = Run::of(m, start).at(100);
            assert!(end.distance(target) < 1e-4, "{start}: {end}");
        }
    }

    /// Fibonacci: (a, b) to (a + b, a); the numbers in x and in the powers; the golden
    /// ratio and −0.618; the small part changing sign every step; x₈ = (34, 21), whose
    /// ratio is 1.619 and growth 1.618.
    #[test]
    fn fibonacci_and_the_golden_ratio() {
        let s = run(3);
        assert_eq!(s.k, 8);
        let m = s.key.0;
        assert_eq!(m.apply(DVec2::new(5.0, 3.0)), DVec2::new(8.0, 5.0));
        let fib = [0.0, 1.0, 1.0, 2.0, 3.0, 5.0, 8.0, 13.0, 21.0, 34.0, 55.0];
        for k in 0..=8 {
            assert_eq!(s.run.at(k), DVec2::new(fib[k + 1], fib[k]));
            if k >= 1 {
                assert_eq!(s.run.power(k).rows(), [[fib[k + 1], fib[k]], [fib[k], fib[k - 1]]]);
            }
        }
        let Eigen::Real { l, v } = s.eigen() else { panic!() };
        let phi = (1.0 + 5f64.sqrt()) / 2.0;
        assert!(close(l[0], phi, 1e-12) && close(l[1], -1.0 / phi, 1e-12) && sig(l[1], 3) == "−0.618", "{l:?}");
        for k in 0..8 {
            let ([_, b], [_, b1]) = (parts_at(&s, k), parts_at(&s, k + 1));
            assert!(b * b1 < 0.0, "step {k}");
            assert!(v[0].perp_dot(s.run.at(k)) * v[0].perp_dot(s.run.at(k + 1)) < 0.0, "x crosses v₁'s line at step {k}");
        }
        assert_eq!((sig(34.0 / 21.0, 4), sig(s.run.growth(8).expect("a step"), 4)), ("1.619".to_string(), "1.618".to_string()));
    }

    /// The shear: λ = 1 twice with one line, the x axis; xₖ = (k, 1); growth 1.09 at
    /// step 12 and falling towards 1; an angle of about 1/k radians, which does not
    /// fall by a fixed factor.
    #[test]
    fn a_shear_grows_by_adding() {
        let s = run(4);
        assert_eq!(s.k, 12);
        let m = s.key.0;
        assert_eq!(m.apply(DVec2::new(0.5, 2.0)), DVec2::new(2.5, 2.0));
        assert_eq!(s.eigen(), Eigen::Defective { l: 1.0, v: DVec2::X });
        for k in 0..=s.run.last() {
            assert_eq!(s.run.at(k), DVec2::new(k as f64, 1.0));
        }
        assert_eq!(sig(s.run.growth(12).expect("a step"), 3), "1.09");
        let growth: Vec<f64> = (2..=100).map(|k| s.run.growth(k).expect("a step")).collect();
        assert!(growth.windows(2).all(|g| g[1] < g[0]) && growth.iter().all(|g| *g > 1.0), "{growth:?}");
        let (_, angle) = s.series();
        let angle = angle.expect("one line to turn to");
        for k in 5..=100 {
            assert!(close(angle[k].to_radians() * k as f64, 1.0, 0.015), "step {k}: {}", angle[k]);
        }
        // a fixed factor would be a straight line on the log chart: here the factor creeps up to 1
        let factor: Vec<f64> = (1..60).map(|k| angle[k + 1] / angle[k]).collect();
        assert!(factor.windows(2).all(|f| f[1] > f[0]), "{factor:?}");
    }

    /// x₀ put on v₂'s line, for the first lesson's matrix: the part along v₁ starts
    /// near 10⁻¹⁶ of x, not at 0, grows by |λ₁|/|λ₂| a step against the other, and
    /// wins within the steps played. For Fibonacci's, it grows back to be seen. And
    /// "usually": a diagonal A keeps x exactly on its line.
    #[test]
    fn a_start_on_the_losing_line_is_not_kept_there() {
        let mut s = Powers::create();
        s.onto_losing_line();
        let Eigen::Real { l, v } = s.eigen() else { panic!() };
        assert!(s.x0.perp_dot(v[1]).abs() < 1e-15 * s.x0.length() && s.playing && s.stop_at == Some(s.plan));
        let size = |k: usize| parts(v, s.run.at(k)).map(f64::abs);
        let first = size(0)[0] / s.x0.length();
        assert!(first < 1e-15, "{first}");
        assert!(first > 1e-17, "{first}");
        for k in 15..50 {
            let gain = (size(k + 1)[0] / size(k + 1)[1]) / (size(k)[0] / size(k)[1]);
            assert!(close(gain, l[0] / l[1], 1e-3), "step {k}: {gain}");
        }
        let wins = (0..=s.plan).find(|&k| size(k)[0] > size(k)[1]).expect("v₁'s part wins");
        assert!((50..=60).contains(&wins) && s.plan == wins + 12, "{wins} {}", s.plan);

        let mut fib = Powers::create();
        fib.set_up(3);
        fib.onto_losing_line();
        // (here x₀ has no part along v₁ at all; the first step's rounding gives it one)
        let v = fib.eigen().lines().expect("lines");
        let k = (0..20).find(|&k| parts(v, fib.run.at(k))[0] != 0.0).expect("a part along v₁");
        let first = parts(v, fib.run.at(k))[0].abs() / fib.run.at(k).length();
        assert!(first > 1e-17 && first < 1e-15, "step {k}: {first}");
        let back = fib.run.at(fib.plan);
        let v1 = fib.eigen().lines().expect("lines")[0];
        assert!(back.length() > fib.x0.length() && angle_to_line(back, v1) < 1e-3 && fib.plan < MOST_STEPS, "{back} after {}", fib.plan);

        let mut diagonal = Powers::create();
        diagonal.a = Mat2::from_cols(Vec2::new(1.25, 0.0), Vec2::new(0.0, 0.5));
        diagonal.onto_losing_line();
        assert!(diagonal.run.x.iter().all(|x| x.x == 0.0), "{:?}", &diagonal.run.x[..5]);
    }

    /// The matrices offered by name do what the note under them says.
    #[test]
    fn the_quick_matrices_are_what_they_say() {
        let of = |name: &str| M2::of(Mat2::from_cols_array(&PRESETS.iter().find(|p| p.0 == name).expect("a preset").1));
        let Eigen::Complex { r, theta } = eigen(of("turn 45°")) else { panic!() };
        assert!(close(r, 1.0, 1e-6) && close(theta.to_degrees(), 45.0, 1e-4));
        let flip = eigen(of("flip"));
        assert!(matches!(flip, Eigen::Real { l, .. } if l == [1.0, -1.0]) && flip.winner().is_none());
        let x = DVec2::new(2.0, 3.0);
        let run = Run::of(of("flip"), x);
        assert!(run.at(2) == x && run.at(1) == DVec2::new(3.0, 2.0), "no winner: it goes back and forth");
        let Eigen::Real { l, v } = eigen(of("squash")) else { panic!() };
        assert!(close(l[0], 1.0, 1e-6) && l[1].abs() < 1e-6);
        assert!(angle_to_line(Run::of(of("squash"), x).at(1), v[0]) < 1e-4, "one step lands on v₁'s line");
        let zero = Run::of(of("zero"), x);
        assert!(zero.at(1) == DVec2::ZERO && zero.growth(1) == Some(0.0) && zero.growth(2).is_none());
        let said: Vec<String> = sums_lines(&zero, of("zero"), 3).into_iter().map(|l| l.0).collect();
        assert!(said.iter().any(|l| l.contains("sent $x$ to $0$")) && said.iter().any(|l| l.contains("none ($x$ was $0$)")), "{said:?}");
    }

    /// Where a run ends: after a hundred steps, or as soon as x is more than a
    /// million times as long as x₀; the step after that is never drawn.
    #[test]
    fn a_run_stops_where_the_note_says() {
        let mut s = Powers::create();
        s.refresh();
        let n = s.run.last();
        let len = |k: usize| s.run.at(k).length() / s.x0.length();
        assert!(s.run.far && len(n) > FAR && len(n - 1) <= FAR && n < MOST_STEPS, "{n}");
        assert!(the_end(&s.run, n).is_some_and(|t| t.contains("too far out")) && the_end(&s.run, n - 1).is_none());
        s.playing = true;
        s.stop_at = None;
        for _ in 0..10_000 {
            s.advance(0.05);
        }
        assert!(!s.playing && s.k == n);
        s.step();
        assert_eq!(s.k, n);
        let markov = run(2);
        assert!(!markov.run.far && markov.run.last() == MOST_STEPS);
        assert!(the_end(&markov.run, MOST_STEPS).is_some_and(|t| t.contains("100 steps")));
    }

    /// The dot drawn follows the step taken: it gets there exactly, never overshoots,
    /// and takes the same time whatever the display's rate; back to the start, it runs
    /// back the same way.
    #[test]
    fn the_dot_glides_to_each_step() {
        let mut ends = Vec::new();
        for hz in [30.0, 60.0, 144.0] {
            let mut s = Powers::create();
            (s.playing, s.k, s.between, s.shown) = (false, 3, 3.0, 3.0);
            s.step();
            let (mut t, mut last) = (0.0, 3.0);
            while s.shown != 4.0 {
                s.follow(1.0 / hz, false);
                t += 1.0 / hz;
                assert!(s.shown >= last && s.shown <= 4.0 && t < 2.0, "{hz} Hz: {} after {t}", s.shown);
                last = s.shown;
            }
            ends.push(t);
            s.rewind();
            while s.shown != 0.0 {
                s.follow(1.0 / hz, false);
                assert!(s.shown >= 0.0 && s.shown <= last);
                last = s.shown;
            }
        }
        assert!(ends.iter().all(|t| (t - ends[1]).abs() < 0.06), "{ends:?}");
        // with the scene's time stopped, it is there at once
        let mut s = Powers::create();
        s.step();
        s.follow(0.0, true);
        assert_eq!(s.shown, s.k as f64);
    }

    /// Degenerate input: the zero matrix, a start at 0, the largest entries there are,
    /// a matrix whose two lines are nearly one, numbers that are not numbers, and A = I.
    /// Every number the picture and the panel use stays finite, and nothing panics.
    #[test]
    fn degenerate_input_leaves_everything_finite() {
        let mats = [
            Mat2::ZERO,
            Mat2::IDENTITY,
            Mat2::from_cols(Vec2::new(100.0, -100.0), Vec2::new(100.0, 100.0)),
            Mat2::from_cols(Vec2::new(-100.0, 0.0), Vec2::new(0.0, -100.0)),
            Mat2::from_cols(Vec2::new(1.0, 1e-6), Vec2::new(1.0, 1.0)),
            Mat2::from_cols(Vec2::new(1.0, 0.0), Vec2::new(1e-7, 1.0)),
            Mat2::from_cols(Vec2::new(f32::NAN, 0.0), Vec2::new(f32::INFINITY, 1.0)),
            Mat2::from_cols(Vec2::new(0.0, 0.0), Vec2::new(1.0, 0.0)),
            Mat2::from_cols(Vec2::new(1e-30, 0.0), Vec2::new(0.0, 1e-30)),
        ];
        for a in mats {
            for x0 in [DVec2::new(2.0, 3.0), DVec2::ZERO, DVec2::new(REACH, -REACH), DVec2::new(1e-9, 0.0)] {
                let mut s = Powers::create();
                (s.a, s.x0) = (a, x0);
                s.refresh();
                let e = s.eigen();
                assert!(format!("{e:?}").chars().all(|c| c != 'N'), "{a}: {e:?}");
                assert!(s.run.x.iter().all(|x| x.is_finite()) && s.run.p.iter().all(|p| p.rows().iter().flatten().all(|v| v.is_finite())), "{a} {x0}");
                assert!(s.run.x.iter().all(|x| x.length() <= FAR * x0.length() * 200.0 + 1e-300));
                let (sizes, angle) = s.series();
                assert!(sizes.iter().flatten().chain(angle.iter().flatten()).all(|v| v.is_finite() || v.is_nan()));
                let r = s.ranges_wanted();
                assert!(r.iter().flatten().all(|v| v.is_finite()) && r.iter().all(|[lo, hi]| lo < hi), "{a} {x0}: {r:?}");
                let (lo, hi) = s.fit_box(MOST_STEPS);
                assert!(lo.is_finite() && hi.is_finite());
                let View::Fit { center, half } = s.the_view() else { panic!() };
                assert!(center.is_finite() && half.is_finite() && half.min_element() > 0.0, "{a} {x0}");
                for k in [0, 1, 2, s.run.last()] {
                    for (line, _) in sums_lines(&s.run, s.key.0, k) {
                        assert!(!line.contains("NaN") && !line.contains("inf") && !line.contains('?'), "{a} {x0}: {line}");
                    }
                }
                if let Some(curve) = orbit(s.key.0, s.run.at(3)) {
                    assert!(curve.iter().all(|p| p.is_finite()));
                }
                s.onto_losing_line();
                assert!(s.x0.is_finite() && s.run.x.iter().all(|x| x.is_finite()));
                for _ in 0..200 {
                    s.advance(0.1);
                    s.follow(0.1, false);
                }
                assert!(s.shown.is_finite() && s.k <= s.run.last());
            }
        }
        // the layouts, for every kind of window
        for (w, h) in [(1600.0, 1000.0), (1078.0, 1000.0), (390.0, 800.0), (800.0, 390.0), (240.0, 180.0), (1.0, 1.0), (0.0, 0.0)] {
            let lay = Layout::of(Vec2::new(w, h), 46.0);
            let corners = [Some(lay.plane), lay.backing].into_iter().flatten().flat_map(|(a, b)| [a, b]);
            assert!(corners.chain(lay.charts.iter().flatten().flat_map(|(a, b)| [*a, *b])).all(|p| p.is_finite()), "{w}x{h}");
        }
    }

    /// Numbers as they are written on the screen.
    #[test]
    fn numbers_are_written_plainly() {
        assert_eq!(sig(1.2000000626, 3), "1.2");
        assert_eq!(sig(-0.013, 3), "−0.013");
        assert_eq!(sig(3.88e-8, 3), "3.88×10⁻⁸");
        assert_eq!(sig(9.9996e7, 3), "1×10⁸");
        assert_eq!(sig(999600.0, 3), "999600");
        assert_eq!(sig(-1.453712e5, 3), "−1.45×10⁵");
        assert_eq!(sig(89.0, 3), "89");
        assert_eq!(sig(f64::NAN, 3), "?");
        assert_eq!(point(DVec2::new(0.2392993785839923, -1.1633368497687702e-7)), "(0.239, 0)");
        assert_eq!(point(DVec2::new(1.368, 1.296)), "(1.37, 1.30)");
        assert_eq!(point(DVec2::new(34.0, 21.0)), "(34, 21)");
        assert_eq!(together(&[0.0798, 3.88e-8, -3.88e-8, 0.0798]), ["0.0798", "0", "0", "0.0798"]);
        assert_eq!(decade(-16, ""), "$10^{-16}$");
        assert_eq!(decade(0, r"{}^\circ"), r"$1{}^\circ$");
    }

    /// The numbers behind the captions (run with --ignored --nocapture).
    #[test]
    #[ignore]
    fn explore() {
        for (i, l) in TRIALS.iter().enumerate() {
            let s = run(i);
            println!("{}: k = {}, x = {:?}, eigen {:?}", l.name, s.k, s.run.at(s.k), s.eigen());
            for line in sums_lines(&s.run, s.key.0, s.k) {
                println!("    {}", line.0);
            }
            let (sizes, angle) = s.series();
            for k in 0..=s.k.min(14) {
                println!("    k {k:2}  |x| {:.5}  a {:.4e}  b {:.4e}  angle {:?}  growth {:?}", sizes[0][k], sizes[1].get(k).unwrap_or(&f64::NAN), sizes[2].get(k).unwrap_or(&f64::NAN), angle.as_ref().map(|a| a[k]), s.run.growth(k));
            }
        }
        // x0 on the losing line, for the first lesson's matrix and the others
        for i in [0, 2, 3] {
            let mut s = Powers::create();
            s.set_up(i);
            s.onto_losing_line();
            let (sizes, _) = s.series();
            let cross = (0..sizes[1].len()).find(|&k| sizes[1][k] > sizes[2][k]);
            println!("{} on v2: a0/|x0| {:.3e}, crossing at {:?}, run {} far {}", TRIALS[i].name, sizes[1][0] / sizes[0][0], cross, s.run.last(), s.run.far);
            for k in (0..=s.run.last()).step_by(5) {
                println!("    k {k:3}  a {:.3e}  b {:.3e}", sizes[1][k], sizes[2][k]);
            }
        }
        // where the runs end
        for i in 0..TRIALS.len() {
            let mut s = Powers::create();
            s.set_up(i);
            println!("{}: the run ends at {} (far {})", TRIALS[i].name, s.run.last(), s.run.far);
        }
    }
}

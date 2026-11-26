//! Quadratic forms: a symmetric matrix as a surface over the plane.

use super::*;
use glam::DVec2;
use std::ops::RangeInclusive;

pub const DEF: SceneDef = SceneDef {
    id: "quadform",
    title: "Quadratic forms",
    group: "Linear algebra: eigenvalues",
    blurb: "A symmetric matrix as a bowl, a saddle or a trough, with its eigenvectors as the axes.",
    make: |_| Box::new(QuadForm::create()),
};

// ----- colours, each with one meaning ----------------------------------------------------

/// λ₁, the larger eigenvalue: its eigenvector, and everything drawn along it.
const V1: Color = pal::ORANGE;
/// λ₂, the smaller, likewise.
const V2: Color = pal::PURPLE;
/// The unit circle lifted to its heights (the ring), x on it, and its height.
const RING: Color = pal::WHITE;
/// The level curves xᵀAx = 1 and xᵀAx = −1.
const LEVEL: Color = pal::YELLOW;
/// Gradient descent's path.
const DESCENT: Color = pal::CYAN;
/// The coordinate axes x₁ and x₂, and the turned axes y₁ and y₂.
const AXES: Color = Color::hex(0xC9D2E3);
/// The heights, as colours on the floor.
const CMAP: Cmap = Cmap::Slate;
/// The surface itself.
const SURFACE: Color = Color::hex(0x6F8FC0);

// ----- the picture's measurements --------------------------------------------------------

/// The surface stands on the unit disc, so that its rim is the unit circle lifted to
/// its heights: the ring.
const SURF_R: f64 = 1.0;
/// The map on the floor reaches further: far enough for the level curve xᵀAx = 1 to
/// close whenever both eigenvalues are at least 1/1.7² ≈ 0.35.
const FLOOR_R: f64 = 1.7;
/// The tallest the surface is drawn (in units of the plane): a form with larger heights
/// is drawn with its heights scaled down, by halves, and the picture says so.
const TALL: f64 = 2.2;
/// How far below the surface's lowest point the floor lies.
const GAP: f32 = 0.7;
/// Vertices of the surfaces round (the first and last the same) and outwards.
const ROUND: u32 = 121;
const OUT: u32 = 25;
/// Samples of the ring, and of each level curve (on the floor, and on the surface).
const RING_N: usize = 240;
const LEVEL_N: usize = 160;
const LEVEL_TOP_N: usize = 96;

/// Seconds a change of matrix takes when an experiment asks for one.
const MORPH_TIME: f32 = 1.4;
/// Seconds the axes take to turn onto the eigenvectors.
const TURN_TIME: f32 = 2.2;
/// Seconds x takes to go once round the circle.
const SWEEP_TIME: f32 = 8.0;
/// How quickly the drawn scale of the heights glides to a new one (seconds for two
/// thirds of the way).
const SCALE_GLIDE: f32 = 0.12;
/// Gradient descent's steps: how many are drawn, and how long each is, as a share of
/// the longest that does not throw it out of the steepest direction.
const DESCENT_STEPS: usize = 30;
const DESCENT_SHARE: f64 = 0.9;

/// How high the picture is seen from, and the tangent of half the camera's field of
/// view. (Which way round it is seen from depends on the experiment: see `Trial::side`.)
const PITCH: f32 = 0.55;
const TAN_HALF_FOV: f32 = 0.3203;

// ----- the form --------------------------------------------------------------------------

/// The symmetric matrix A = [[a, b], [b, c]], and with it the quadratic form
/// xᵀAx = a x₁² + 2b x₁x₂ + c x₂².
#[derive(Clone, Copy, Debug, PartialEq)]
struct Form {
    a: f64,
    b: f64,
    c: f64,
}

/// The eigenvalues of a form, λ₁ ≥ λ₂, and the angle of λ₁'s eigenvector from the x₁
/// axis, in (−90°, 90°]; λ₂'s is a right angle further round. No angle when λ₁ = λ₂:
/// then every direction is an eigenvector.
#[derive(Clone, Copy, Debug, PartialEq)]
struct Spectrum {
    l1: f64,
    l2: f64,
    theta: Option<f64>,
}

/// What counts as nothing beside numbers of size `size`: rounding, not a value.
fn negligible(v: f64, size: f64) -> bool {
    v.abs() <= 1e-9 * size
}

impl Form {
    /// The form with eigenvalues `l1` and `l2` along the directions at `theta` and a right
    /// angle further round: A = QΛQᵀ, Q the rotation by `theta`.
    fn of(l1: f64, l2: f64, theta: f64) -> Form {
        let (s, c) = theta.sin_cos();
        let size = l1.abs().max(l2.abs());
        let clean = |v: f64| if negligible(v, size) { 0.0 } else { v };
        Form { a: clean(l1 * c * c + l2 * s * s), b: clean((l1 - l2) * s * c), c: clean(l1 * s * s + l2 * c * c) }
    }

    /// The height xᵀAx above the point x.
    fn at(&self, x: DVec2) -> f64 {
        self.a * x.x * x.x + 2.0 * self.b * x.x * x.y + self.c * x.y * x.y
    }

    /// Ax.
    fn times(&self, x: DVec2) -> DVec2 {
        DVec2::new(self.a * x.x + self.b * x.y, self.b * x.x + self.c * x.y)
    }

    /// The same form in the coordinates along axes turned by `alpha`: RᵀAR.
    fn turned(&self, alpha: f64) -> Form {
        let (s, c) = alpha.sin_cos();
        let (u, v) = (DVec2::new(c, s), DVec2::new(-s, c));
        let size = self.size();
        let clean = |v: f64| if negligible(v, size) { 0.0 } else { v };
        Form { a: clean(u.dot(self.times(u))), b: clean(u.dot(self.times(v))), c: clean(v.dot(self.times(v))) }
    }

    fn size(&self) -> f64 {
        self.a.abs().max(self.b.abs()).max(self.c.abs())
    }

    fn det(&self) -> f64 {
        self.a * self.c - self.b * self.b
    }

    fn is_finite(&self) -> bool {
        self.a.is_finite() && self.b.is_finite() && self.c.is_finite()
    }

    /// The eigenvalues and the axes, in closed form: for a symmetric 2 × 2 matrix the
    /// eigenvalues are the middle (a + c)/2 plus and minus the radius √(((a − c)/2)² + b²),
    /// and the axes are at half the angle of the point ((a − c)/2, b).
    fn spectrum(&self) -> Spectrum {
        let size = self.size();
        let (middle, half) = (0.5 * (self.a + self.c), 0.5 * (self.a - self.c));
        let radius = half.hypot(self.b);
        // (the one of the two that is not a difference of nearly equal numbers, and the
        // other from the determinant, their product)
        let (mut l1, mut l2) = if middle >= 0.0 {
            let l1 = middle + radius;
            (l1, if l1 != 0.0 { self.det() / l1 } else { 0.0 })
        } else {
            let l2 = middle - radius;
            (self.det() / l2, l2)
        };
        l2 = l2.min(l1);
        if negligible(l1, size) {
            l1 = 0.0;
        }
        if negligible(l2, size) {
            l2 = 0.0;
        }
        let theta = (!negligible(radius, size)).then(|| 0.5 * self.b.atan2(half));
        Spectrum { l1, l2, theta }
    }
}

impl Spectrum {
    /// The larger of the two sizes.
    fn size(&self) -> f64 {
        self.l1.abs().max(self.l2.abs())
    }
}

/// The unit vector at angle `t`.
fn unit(t: f64) -> DVec2 {
    DVec2::new(t.cos(), t.sin())
}

/// The angle `t` taken into (−90°, 90°]: an axis is a line, and a line turned half a
/// turn is the same line.
fn axis_angle(t: f64) -> f64 {
    let half = std::f64::consts::PI;
    let mut t = t.rem_euclid(half);
    if t > 0.5 * half {
        t -= half;
    }
    t
}

/// What the surface is.
#[derive(Clone, Copy, Debug, PartialEq)]
enum Kind {
    Bowl,
    Dome,
    Saddle,
    Trough,
    Ridge,
    Flat,
}

fn kind(s: &Spectrum) -> Kind {
    let sign = |v: f64| if v > 0.0 { 1 } else if v < 0.0 { -1 } else { 0 };
    match (sign(s.l1), sign(s.l2)) {
        (0, 0) => Kind::Flat,
        (_, 1) => Kind::Bowl,
        (-1, _) => Kind::Dome,
        (1, -1) => Kind::Saddle,
        (1, 0) => Kind::Trough,
        _ => Kind::Ridge,
    }
}

/// The verdict: what the surface is, in a word or two, and why.
fn verdict(s: &Spectrum) -> (&'static str, &'static str) {
    let round = s.theta.is_none();
    match kind(s) {
        Kind::Bowl if round => ("A round bowl.", "Both eigenvalues are positive and equal: it rises the same way in every direction, and every direction is an eigenvector."),
        Kind::Bowl => ("A bowl.", "Both eigenvalues are positive: every way out of $0$ goes up (positive definite)."),
        Kind::Dome if round => ("A round upside-down bowl.", "Both eigenvalues are negative and equal: it falls the same way in every direction, and every direction is an eigenvector."),
        Kind::Dome => ("An upside-down bowl.", "Both eigenvalues are negative: every way out of $0$ goes down (negative definite)."),
        Kind::Saddle => ("A saddle.", r"One eigenvalue is positive and one negative: up along $\lambda_1$'s axis, down along $\lambda_2$'s (indefinite)."),
        Kind::Trough => ("A trough.", r"$\lambda_2 = 0$: flat all along $\lambda_2$'s axis, up across it (positive semidefinite)."),
        Kind::Ridge => ("A ridge: an upside-down trough.", r"$\lambda_1 = 0$: flat all along $\lambda_1$'s axis, down across it (negative semidefinite)."),
        Kind::Flat => ("Flat.", r"$A = 0$: every height is $0$, and every direction is an eigenvector, with $\lambda = 0$."),
    }
}

/// A number as the picture and the panel write it: to two places, or, if it is not 0
/// but would round to 0.00, to as many as show that it is not.
fn val(v: f64) -> String {
    if v != 0.0 && v.abs() < 0.005 && v.is_finite() {
        let places = ((-v.abs().log10()).ceil() as usize).clamp(3, 12);
        num(v, places)
    } else {
        num(v, 2)
    }
}

/// The form written out with its present numbers, in the coordinates `names`, as markup
/// (to go between dollars): "1.70 x_1^2 + 1.20 x_1 x_2 + 0.80 x_2^2". Without its mixed
/// term if `mixed` is false.
fn written(m: &Form, names: [&str; 2], mixed: bool) -> String {
    let [p, q] = names;
    let mut terms: Vec<(f64, String)> = vec![(m.a, format!("{p}^2"))];
    if mixed {
        terms.push((2.0 * m.b, format!("{p} {q}")));
    }
    terms.push((m.c, format!("{q}^2")));
    let mut out = String::new();
    for (k, (coef, name)) in terms.iter().enumerate() {
        let text = val(coef.abs());
        let negative = *coef < 0.0 && text.chars().any(|c| c.is_ascii_digit() && c != '0');
        match (k, negative) {
            (0, true) => out += &format!("−{text} {name}"),
            (0, false) => out += &format!("{text} {name}"),
            (_, true) => out += &format!(" − {text} {name}"),
            (_, false) => out += &format!(" + {text} {name}"),
        }
    }
    out
}

/// One piece of a level curve: the point at each value of its parameter, and the
/// stretch of the parameter that lies inside the disc.
type Piece = (Box<dyn Fn(f64) -> DVec2>, f64, f64);

/// The points of the plane where xᵀAx = `level`, out to `reach` from 0, as curves to
/// draw (a hyperbola has two branches; a trough's level set is two straight lines),
/// for a form with spectrum `s` and its first axis at `angle`. Each curve is `n` points
/// evenly spaced along it (round an ellipse, the first and last the same), in the same
/// order and the same way along whatever the kind of curve: so a curve that changes a
/// little is drawn a little changed, point for point. An ellipse stretched until the
/// rim cuts it in two becomes, as λ₂ reaches 0, the two straight lines of a trough and
/// then a saddle's hyperbola, without a jump.
fn level_curves(s: &Spectrum, angle: f64, level: f64, reach: f64, n: usize) -> Vec<Vec<DVec2>> {
    // in the coordinates along the axes, p y₁² + q y₂² = 1; with a and b the distances
    // at which a curve crosses (or would cross) the axes, where it is inside the disc
    // can be said exactly
    let (p, q) = (s.l1 / level, s.l2 / level);
    let (u, v) = (unit(angle), unit(angle + 0.5 * std::f64::consts::PI));
    let r2 = reach * reach;
    let half_pi = 0.5 * std::f64::consts::PI;
    // each piece: the point at each value of a parameter, and the stretch of it inside
    let mut pieces: Vec<Piece> = Vec::new();
    if p > 0.0 && q > 0.0 {
        let (a, b) = (1.0 / p.sqrt(), 1.0 / q.sqrt());
        let at = move |t: f64| u * (a * t.cos()) + v * (b * t.sin());
        if a <= reach && b <= reach {
            pieces.push((Box::new(at), 0.0, std::f64::consts::TAU));
        } else if b > reach && a <= reach {
            // cut where it runs out along λ₂'s axis: the arcs round y₁ = −a and y₁ = a
            let t0 = ((r2 - a * a) / (b * b - a * a)).max(0.0).sqrt().min(1.0).asin();
            pieces.push((Box::new(at), std::f64::consts::PI - t0, std::f64::consts::PI + t0));
            pieces.push((Box::new(at), -t0, t0));
        } else if a > reach && b <= reach {
            let t0 = ((r2 - b * b) / (a * a - b * b)).max(0.0).sqrt().min(1.0).asin();
            pieces.push((Box::new(at), -half_pi - t0, -half_pi + t0));
            pieces.push((Box::new(at), half_pi - t0, half_pi + t0));
        }
    } else if p > 0.0 && q <= 0.0 {
        // two branches across λ₁'s axis, at y₁ = ±a: straight lines if q = 0
        let a = 1.0 / p.sqrt();
        if a <= reach {
            for side in [-1.0, 1.0] {
                if q < 0.0 {
                    let b = 1.0 / (-q).sqrt();
                    let top = ((r2 - a * a) / (a * a + b * b)).sqrt().asinh();
                    pieces.push((Box::new(move |t: f64| u * (side * a * t.cosh()) + v * (b * t.sinh())), -top, top));
                } else {
                    let top = (r2 - a * a).sqrt();
                    pieces.push((Box::new(move |t: f64| u * (side * a) + v * t), -top, top));
                }
            }
        }
    } else if q > 0.0 && p <= 0.0 {
        // two branches across λ₂'s axis, at y₂ = ±b
        let b = 1.0 / q.sqrt();
        if b <= reach {
            for side in [-1.0, 1.0] {
                if p < 0.0 {
                    let a = 1.0 / (-p).sqrt();
                    let top = ((r2 - b * b) / (a * a + b * b)).sqrt().asinh();
                    pieces.push((Box::new(move |t: f64| u * (a * t.sinh()) + v * (side * b * t.cosh())), -top, top));
                } else {
                    let top = (r2 - b * b).sqrt();
                    pieces.push((Box::new(move |t: f64| u * t + v * (side * b)), -top, top));
                }
            }
        }
    }
    let mut out = Vec::new();
    for (at, lo, hi) in pieces {
        // n points evenly spaced along the stretch: its length measured on a fine
        // polyline, and each point put back exactly on the curve
        let fine = 256;
        let params: Vec<f64> = (0..=fine).map(|i| lo + (hi - lo) * i as f64 / fine as f64).collect();
        // (each point of the fine polyline worked out once, not once for each of the two
        // steps it ends)
        let points: Vec<DVec2> = params.iter().map(|t| at(*t)).collect();
        let mut lengths = vec![0.0];
        for w in points.windows(2) {
            let last = *lengths.last().unwrap_or(&0.0);
            lengths.push(last + (w[1] - w[0]).length());
        }
        let total = *lengths.last().unwrap_or(&0.0);
        // (a stretch of no length, or not a number: nothing to draw)
        if total.is_nan() || total <= 1e-9 {
            continue;
        }
        let mut j = 0;
        let mut curve = Vec::with_capacity(n);
        for i in 0..n {
            let want = total * i as f64 / (n - 1).max(1) as f64;
            while j + 1 < fine && lengths[j + 1] < want {
                j += 1;
            }
            let span = (lengths[j + 1] - lengths[j]).max(1e-300);
            let t = params[j] + (params[j + 1] - params[j]) * ((want - lengths[j]) / span).clamp(0.0, 1.0);
            // (exactly on the curve, and within the disc despite rounding at the ends)
            let x = at(t);
            curve.push(if x.length() > reach { x * (reach / x.length()) } else { x });
        }
        // all the same way along: along λ₂'s axis where a piece runs that way, else λ₁'s
        let along = curve[n - 1] - curve[0];
        let w = if along.dot(v).abs() >= along.dot(u).abs() { v } else { u };
        if along.dot(w) < 0.0 {
            curve.reverse();
        }
        out.push(curve);
    }
    out
}

/// The level curves the picture draws, xᵀAx = 1 and xᵀAx = −1, each out to the floor's
/// rim and out to the ring, with what they were made from. `level_curves` reads nothing
/// but the two eigenvalues and the angle of the axes, so the curves are made again only
/// when one of those three numbers changes (to the last bit): in every frame while the
/// matrix changes, in none while x moves or the view turns.
struct Levels {
    made_from: [u64; 3],
    /// For 1 and then −1: the curves on the floor, and those on the surface.
    curves: [(Curves, Curves); 2],
}

/// Curves as `level_curves` makes them: each a list of points.
type Curves = Vec<Vec<DVec2>>;

impl Levels {
    fn key(s: &Spectrum, angle: f64) -> [u64; 3] {
        [s.l1.to_bits(), s.l2.to_bits(), angle.to_bits()]
    }

    /// The curves for the spectrum `s` with its first axis at `angle`: `kept` if it was
    /// made from those, else made afresh.
    fn of(kept: Option<Levels>, s: &Spectrum, angle: f64) -> Levels {
        let key = Levels::key(s, angle);
        kept.filter(|l| l.made_from == key).unwrap_or_else(|| Levels {
            made_from: key,
            curves: [1.0, -1.0].map(|level| (level_curves(s, angle, level, FLOOR_R, LEVEL_N), level_curves(s, angle, level, SURF_R, LEVEL_TOP_N))),
        })
    }
}

/// Gradient descent on the form from `start`: steps x ← x − h ∇(xᵀAx) = x − 2hAx. Along
/// each eigenvector that multiplies x's part by 1 − 2hλ, which must stay above −1 or
/// the steps grow: h < 1/λ. h is the share `DESCENT_SHARE` of that for the largest
/// |λ|. Stops when it leaves the disc of radius `reach`.
fn descent(m: &Form, start: DVec2, steps: usize, reach: f64) -> Vec<DVec2> {
    let size = m.spectrum().size();
    let mut path = vec![start];
    if size <= 0.0 {
        return path;
    }
    let h = DESCENT_SHARE / size;
    let mut x = start;
    for _ in 0..steps {
        x -= 2.0 * h * m.times(x);
        if !x.is_finite() || x.length() > reach {
            break;
        }
        path.push(x);
    }
    path
}

/// The scale the heights are drawn at: 1, or the largest half, quarter, ... that keeps
/// the surface within `TALL` of the plane.
fn scale_for(s: &Spectrum) -> f64 {
    let tallest = s.size() * SURF_R * SURF_R;
    let mut scale = 1.0;
    while tallest * scale > TALL && scale > 1e-9 {
        scale *= 0.5;
    }
    scale
}

/// How the picture says what scale its heights are drawn at.
fn scale_words(scale: f64) -> Option<String> {
    if scale >= 1.0 {
        return None;
    }
    let inverse = (1.0 / scale).round();
    Some(match inverse as u64 {
        2 => "heights drawn at half their size".to_string(),
        4 => "heights drawn at a quarter of their size".to_string(),
        k => format!("heights drawn at 1/{k} of their size"),
    })
}

// ----- the experiments -------------------------------------------------------------------

/// One of the "things to try": the matrix (a, b, c), where x starts on the circle (its
/// angle in degrees), which way the picture is seen, and what is switched on.
struct Trial {
    label: &'static str,
    look_for: &'static str,
    form: [f64; 3],
    phi: f64,
    /// Where λ₁'s axis lies on the screen, in degrees from the screen's rightward
    /// direction (positive: turned away from the eye). Across the screen, a bowl shows
    /// its steep profile as a U; at 45°, a saddle shows both its rise and its fall.
    side: f64,
    sweep: bool,
    turned: bool,
    descent: bool,
}

const TRIALS: [Trial; 5] = [
    Trial {
        label: "round the circle",
        look_for: r"$x$ goes once round the unit circle, and the ring takes it up and down twice. Its height is greatest, $2.00 = \lambda_1$, each time $x$ crosses $\lambda_1$'s axis (orange), and least, $0.50 = \lambda_2$, where it crosses $\lambda_2$'s axis (purple), a right angle further round: the top and the bottom of the graph. Drag $x$ yourself to stop it.",
        form: [1.7, 0.6, 0.8],
        phi: 0.0,
        side: -12.0,
        sweep: true,
        turned: false,
        descent: false,
    },
    Trial {
        label: "a saddle",
        look_for: r"$\lambda_1 = 1$ and $\lambda_2 = -1$: up along the orange axis, down along the purple one, and the ring dips below $0$ twice. The height is $0$ along two lines across the floor (dashed, pale), halfway between the axes. The level curves $x^\top A x = 1$ and $x^\top A x = -1$ are hyperbolas: the first never meets the purple axis, the second never meets the orange.",
        form: [0.6, 0.8, -0.6],
        phi: 60.0,
        side: 45.0,
        sweep: false,
        turned: false,
        descent: false,
    },
    Trial {
        label: "a trough",
        look_for: r"$\lambda_2 = 0$: along the purple axis the height stays $0$, so the bottom of the trough is a whole line, not a single point (positive semidefinite: never below $0$, and $0$ all along that line). $\det A = \lambda_1 \lambda_2 = 0$. The level curve $x^\top A x = 1$ has come apart into two straight lines, $1/\sqrt{\lambda_1} = 0.82$ either side of the bottom.",
        form: [1.2, 0.6, 0.3],
        phi: 60.0,
        side: 45.0,
        sweep: false,
        turned: false,
        descent: false,
    },
    Trial {
        label: "a narrow valley",
        look_for: r"$\lambda_1 = 3$ and $\lambda_2 = 0.1$: at the same distance from the bottom, the valley is thirty times steeper across than along. Gradient descent (cyan) steps downhill from $x$, its steps nine tenths of the size at which they would start to grow across the valley. It bounces from side to side, and along the gentle floor it creeps: after 30 steps it is still 0.13 from the bottom, where in the opening bowl ($\lambda_1/\lambda_2 = 4$) the same rule from the same start gets within 0.001.",
        form: [2.42, 1.16, 0.68],
        phi: 85.0,
        side: -12.0,
        sweep: false,
        turned: false,
        descent: true,
    },
    Trial {
        label: "turn the axes",
        look_for: r"The axes $x_1$ and $x_2$ turn by 26.6° until they lie along the eigenvectors (and take their colours). In coordinates $y_1, y_2$ along the turned axes the mixed term is gone: $1.70 x_1^2 + 1.20 x_1 x_2 + 0.80 x_2^2$ is $2.00 y_1^2 + 0.50 y_2^2$, and the matrix in those coordinates, $Q^\top A Q$, has the eigenvalues on its diagonal and $0$ off it: $A = Q \Lambda Q^\top$, the spectral theorem.",
        form: [1.7, 0.6, 0.8],
        phi: 60.0,
        side: -12.0,
        sweep: false,
        turned: true,
        descent: false,
    },
];

/// The experiment the scene opens as (without its sweep).
const OPENING: usize = 0;

// ----- where things go on the canvas -----------------------------------------------------

/// Sizes of the two screen panels: the card with the matrix, the form and the verdict,
/// and the graph of the ring's height.
const CARD_W: f32 = 400.0;
const GRAPH_W: f32 = 370.0;
const GRAPH_H: f32 = 172.0;
/// The graph's height below a phone's picture, where every pixel counts.
const GRAPH_H_NARROW: f32 = 146.0;
/// The type size of the card's matrices.
const CARD_TEXT: f32 = 13.5;

/// The card's height at a given width: the most its contents can take, whichever
/// verdict it shows (fixed for the width, so that nothing in it moves when what it
/// says changes). Worked out once for a width: the layout is decided more than once a
/// frame, and the formulas in the verdicts are measured from the fonts' metrics.
fn card_height(width: f32) -> f32 {
    thread_local! {
        static KEPT: std::cell::Cell<Option<(u32, f32)>> = const { std::cell::Cell::new(None) };
    }
    if let Some((of, height)) = KEPT.get() {
        if of == width.to_bits() {
            return height;
        }
    }
    let longest = [
        Spectrum { l1: 1.0, l2: 1.0, theta: None },
        Spectrum { l1: 2.0, l2: 1.0, theta: Some(0.0) },
        Spectrum { l1: -1.0, l2: -1.0, theta: None },
        Spectrum { l1: -1.0, l2: -2.0, theta: Some(0.0) },
        Spectrum { l1: 1.0, l2: -1.0, theta: Some(0.0) },
        Spectrum { l1: 1.0, l2: 0.0, theta: Some(0.0) },
        Spectrum { l1: 0.0, l2: -1.0, theta: Some(0.0) },
        Spectrum { l1: 0.0, l2: 0.0, theta: None },
    ]
    .iter()
    .map(|s| line_count(verdict(s).1, 12.0, width - 24.0))
    .max()
    .unwrap_or(1);
    let matrix = matrix_size(2, 2, 5, CARD_TEXT).y;
    let height = 10.0 + matrix + 6.0 + 19.0 + 19.0 + 21.0 + 19.0 + 16.0 * longest as f32 + 8.0 + 24.0 + 10.0;
    KEPT.set(Some((width.to_bits(), height)));
    height
}

#[derive(Clone, Copy, Debug, PartialEq)]
struct Layout {
    /// A phone held upright: the panels go below the picture.
    narrow: bool,
    card: Option<(Vec2, Vec2)>,
    graph: Option<(Vec2, Vec2)>,
    /// The part of the canvas the picture has to itself.
    picture: (Vec2, Vec2),
}

impl Layout {
    /// For a canvas of `view` logical pixels, `inset` of whose top lies under the
    /// application's buttons.
    fn of(view: Vec2, inset: f32) -> Layout {
        let whole = (Vec2::new(0.0, inset.min(view.y * 0.5)), view.max(Vec2::ONE));
        if view.x < 620.0 && view.y >= 560.0 {
            // a phone held upright: the card and the graph one above the other, below the picture
            let width = view.x - 16.0;
            let card_h = card_height(width);
            let graph_top = view.y - 8.0 - GRAPH_H_NARROW;
            let card_top = graph_top - 6.0 - card_h;
            let picture = (Vec2::new(0.0, whole.0.y), Vec2::new(view.x, (card_top - 4.0).max(whole.0.y + 60.0)));
            return Layout { narrow: true, card: Some((Vec2::new(8.0, card_top), Vec2::new(8.0 + width, card_top + card_h))), graph: Some((Vec2::new(8.0, graph_top), Vec2::new(8.0 + width, graph_top + GRAPH_H_NARROW))), picture };
        }
        let top = 14.0f32.max(inset + 8.0);
        let card_h = card_height(CARD_W);
        if view.x >= CARD_W + GRAPH_W + 48.0 && view.y >= top + card_h + 360.0 {
            // wide and tall enough: both along the top, the picture below
            let card = (Vec2::new(14.0, top), Vec2::new(14.0 + CARD_W, top + card_h));
            let graph = (Vec2::new(view.x - 14.0 - GRAPH_W, top), Vec2::new(view.x - 14.0, top + GRAPH_H));
            return Layout { narrow: false, card: Some(card), graph: Some(graph), picture: (Vec2::new(0.0, top + card_h.max(GRAPH_H)), view) };
        }
        if view.x >= CARD_W + 220.0 && view.y >= top + card_h + 20.0 {
            // otherwise down the left, the picture to their right (the graph if there is room)
            let card = (Vec2::new(14.0, top), Vec2::new(14.0 + CARD_W, top + card_h));
            let graph_top = top + card_h + 10.0;
            let graph = (view.y >= graph_top + GRAPH_H + 10.0).then(|| (Vec2::new(14.0, graph_top), Vec2::new(14.0 + GRAPH_W.min(CARD_W), graph_top + GRAPH_H)));
            return Layout { narrow: false, card: Some(card), graph, picture: (Vec2::new(14.0 + CARD_W, whole.0.y), view) };
        }
        // too small to spare the room: the numbers are in the side panel as well
        Layout { narrow: false, card: None, graph: None, picture: whole }
    }

    fn covers(&self, p: Vec2) -> bool {
        [self.card, self.graph].iter().flatten().any(|(min, max)| p.cmpge(*min).all() && p.cmple(*max).all())
    }
}

/// The camera's right and up directions, seen from `yaw`.
fn screen_axes(yaw: f32) -> (Vec3, Vec3) {
    let right = Vec3::new(-yaw.sin(), yaw.cos(), 0.0);
    let up = Vec3::new(-PITCH.sin() * yaw.cos(), -PITCH.sin() * yaw.sin(), PITCH.cos());
    (right, up)
}

/// The direction to see from for λ₁'s axis at `theta` to lie at `side` (both in
/// radians) from the screen's rightward direction.
fn yaw_for(theta: f64, side: f64) -> f32 {
    (theta - 0.5 * std::f64::consts::PI - side) as f32
}

// ----- the scene -------------------------------------------------------------------------

/// A change of matrix on its way: eigenvalues and angle from where it started, the
/// matrix it ends at, and x's angle at both ends.
#[derive(Clone, Copy, Debug)]
struct Morph {
    from: (f64, f64, f64),
    to: Form,
    phi: (f64, f64),
    t: f32,
}

pub struct QuadForm {
    form: Form,
    /// The angle of λ₁'s eigenvector when it last had one: when λ₁ = λ₂ every direction
    /// is an eigenvector, and the axes stay where they were.
    axis: f64,
    /// Where x is on the unit circle: its angle from the x₁ axis.
    phi: f64,
    morph: Option<Morph>,
    /// The axes asked to lie along the eigenvectors, and how far they have turned (0 to 1).
    turned: bool,
    turn: f32,
    /// x going once round the circle: where it set off, and how far it has got (0 to 1).
    sweep: Option<(f64, f32)>,
    descent: bool,
    /// The scale the heights are drawn at (gliding to `scale_for` the matrix).
    scale: f32,
    surf: Option<Surface>,
    floor: Option<Surface>,
    /// What the two surfaces were last filled for.
    filled: Option<(Form, f32)>,
    /// The level curves as last made (see `Levels`).
    levels: Option<Levels>,
    /// The canvas last drawn on, and how much of its top the application's buttons
    /// cover: they decide where the panels go, and with them the picture.
    canvas: Vec2,
    inset: f32,
    measured: bool,
    new_view: bool,
    /// Which way round the picture is seen from (see `Trial::side`).
    yaw: f32,
    names: Names,
    /// The verdict's reason as last broken into lines, with what it was broken for: the
    /// reason and the room across.
    why: Option<((&'static str, f32), Vec<String>)>,
    /// While x is dragged: its angle less that of the point under the pointer when it
    /// was grabbed.
    grab: Option<f64>,
}

impl QuadForm {
    /// (Nothing here needs the graphics device, so the tests can make one too.)
    fn create() -> Self {
        let t = &TRIALS[OPENING];
        let form = Form { a: t.form[0], b: t.form[1], c: t.form[2] };
        let s = form.spectrum();
        QuadForm {
            form,
            axis: s.theta.unwrap_or(0.0),
            phi: 60f64.to_radians(),
            morph: None,
            turned: false,
            turn: 0.0,
            sweep: None,
            descent: false,
            scale: scale_for(&s) as f32,
            surf: None,
            floor: None,
            filled: None,
            levels: None,
            canvas: Vec2::new(1078.0, 1000.0),
            inset: 0.0,
            measured: false,
            new_view: false,
            yaw: yaw_for(s.theta.unwrap_or(0.0), t.side.to_radians()),
            names: Names::default(),
            why: None,
            grab: None,
        }
    }

    /// Set up one of the experiments in `TRIALS`: the matrix and x move there together,
    /// and what it then plays (a turn of the axes, a sweep of x) starts once they have.
    fn set_up(&mut self, k: usize) {
        let t = &TRIALS[k];
        let to = Form { a: t.form[0], b: t.form[1], c: t.form[2] };
        self.go_to(to, t.phi.to_radians());
        self.turned = t.turned;
        self.sweep = t.sweep.then_some((t.phi.to_radians(), 0.0));
        self.descent = t.descent;
        self.yaw = yaw_for(to.spectrum().theta.unwrap_or(self.axis), t.side.to_radians());
        self.new_view = true;
    }

    /// Move to the matrix `to` and x to the angle `phi`, by way of the eigenvalues and
    /// the angle of the axes: what is on the way is a form too (a bowl on its way to a
    /// saddle passes through a trough).
    fn go_to(&mut self, to: Form, phi: f64) {
        let s = self.form.spectrum();
        let from = (s.l1, s.l2, s.theta.unwrap_or(self.axis));
        // (x the shorter way round)
        let turn = (phi - self.phi + std::f64::consts::PI).rem_euclid(std::f64::consts::TAU) - std::f64::consts::PI;
        // (nothing to move: nothing to wait for)
        self.morph = (to != self.form || turn.abs() > 1e-12).then_some(Morph { from, to, phi: (self.phi, self.phi + turn), t: 0.0 });
    }

    /// The matrix set directly (a slider): whatever was on its way stops.
    fn set_form(&mut self, form: Form) {
        if form.is_finite() {
            self.form = form;
        }
        self.morph = None;
        if let Some(theta) = self.form.spectrum().theta {
            self.axis = theta;
        }
    }

    fn spectrum(&self) -> Spectrum {
        self.form.spectrum()
    }

    /// The angle of λ₁'s axis, also when every direction is one.
    fn angle(&self) -> f64 {
        self.spectrum().theta.unwrap_or(self.axis)
    }

    /// The angle the coordinate axes are drawn at: on their way from x₁, x₂ to the
    /// eigenvectors.
    fn alpha(&self) -> f64 {
        smoothstep(self.turn) as f64 * self.angle()
    }

    fn moving(&self) -> bool {
        self.morph.is_some() || self.sweep.is_some() || self.turn != if self.turned { 1.0 } else { 0.0 }
    }

    /// Move everything that plays on by `dt` seconds of scene time: a change of matrix
    /// first, then the turn of the axes and the sweep of x.
    fn advance(&mut self, dt: f32) {
        let mut dt = dt.max(0.0);
        if let Some(m) = self.morph.as_mut() {
            m.t += dt / MORPH_TIME;
            let m = *m;
            if m.t >= 1.0 {
                self.form = m.to;
                self.phi = m.phi.1;
                self.morph = None;
                // (the time left over goes to what comes next)
                dt = (m.t - 1.0) * MORPH_TIME;
            } else {
                let k = smoothstep(m.t) as f64;
                let s = m.to.spectrum();
                let (l1, l2, from) = m.from;
                let to = s.theta.unwrap_or(from);
                // (the axes the shorter way round: a line turned half a turn is the same line)
                let turn = axis_angle(to - from);
                self.form = Form::of(l1 + (s.l1 - l1) * k, l2 + (s.l2 - l2) * k, from + turn * k);
                self.phi = m.phi.0 + (m.phi.1 - m.phi.0) * k;
            }
            if let Some(theta) = self.form.spectrum().theta {
                self.axis = theta;
            }
            if self.morph.is_some() {
                return;
            }
        }
        let goal = if self.turned { 1.0 } else { 0.0 };
        if self.turn < goal {
            self.turn = (self.turn + dt / TURN_TIME).min(1.0);
        } else if self.turn > goal {
            self.turn = (self.turn - dt / TURN_TIME).max(0.0);
        }
        if let Some((start, t)) = self.sweep.as_mut() {
            *t += dt / SWEEP_TIME;
            if *t >= 1.0 {
                self.phi = start.rem_euclid(std::f64::consts::TAU);
                self.sweep = None;
            } else {
                self.phi = *start + std::f64::consts::TAU * smoothstep(*t) as f64;
            }
        }
    }

    /// The scale the heights should be drawn at.
    fn scale_goal(&self) -> f32 {
        scale_for(&self.spectrum()) as f32
    }

    /// The lowest and highest heights of the picture as drawn with `form` at `scale`: the
    /// floor and the top of the surface (in world units).
    fn extent(form: &Form, scale: f32) -> (f32, f32) {
        let s = form.spectrum();
        let r2 = SURF_R * SURF_R;
        let low = (s.l2.min(0.0) * r2) as f32 * scale;
        let high = (s.l1.max(0.0) * r2) as f32 * scale;
        (low - GAP, high)
    }

    /// Where the floor is drawn now.
    fn floor_z(&self) -> f32 {
        Self::extent(&self.form, self.scale).0
    }

    /// The view that fits the picture the matrix is going to (an experiment's, once it
    /// has moved there) into the part of the canvas the panels leave free.
    fn the_view(&self) -> View {
        let lay = Layout::of(self.canvas, self.inset);
        let (right, up) = screen_axes(self.yaw);
        let form = self.morph.map_or(self.form, |m| m.to);
        let scale = scale_for(&form.spectrum()) as f32;
        let (floor, _) = Self::extent(&form, scale);
        let mut points: Vec<Vec3> = Vec::new();
        for k in 0..32 {
            let t = k as f64 / 32.0 * std::f64::consts::TAU;
            let x = unit(t);
            points.push(Vec3::new((x.x * FLOOR_R) as f32, (x.y * FLOOR_R) as f32, floor));
            let r = x * SURF_R;
            points.push(Vec3::new(r.x as f32, r.y as f32, form.at(r) as f32 * scale));
        }
        let (mut lo, mut hi) = (Vec2::splat(f32::MAX), Vec2::splat(f32::MIN));
        for p in points.iter().filter(|p| p.is_finite()) {
            let s = Vec2::new(p.dot(right), p.dot(up));
            (lo, hi) = (lo.min(s), hi.max(s));
        }
        let middle = (lo + hi) * 0.5;
        // (room round it for the names)
        let size = (hi - lo).max(Vec2::splat(0.5)) * 1.06 + Vec2::new(0.9, 0.5);
        let (min, max) = lay.picture;
        // (and above the caption along the bottom)
        let caption = if self.canvas.y >= 300.0 && !lay.narrow { 34.0 } else { 0.0 };
        // (and room at the top and the bottom for the names)
        let (min, max) = (min + Vec2::new(0.0, 42.0), max - Vec2::new(0.0, 16.0));
        let region = (max - min - Vec2::new(0.0, caption)).max(Vec2::splat(60.0));
        let per_unit = (region.x / size.x).min(region.y / size.y).max(1.0);
        let dist = (self.canvas.y * 0.5 / per_unit / TAN_HALF_FOV).clamp(2.0, 80.0);
        let per_unit = self.canvas.y * 0.5 / (dist * TAN_HALF_FOV);
        // the middle of the picture to the middle of its part of the canvas
        let shift = ((min + max - Vec2::new(0.0, caption)) * 0.5 - self.canvas * 0.5) / per_unit;
        let target = right * middle.x + up * middle.y - right * shift.x + up * shift.y;
        View::Orbit { target, yaw: self.yaw, pitch: PITCH, dist }
    }

    /// x as it is: on the unit circle.
    fn x(&self) -> DVec2 {
        unit(self.phi)
    }

    /// The level curves for the matrix as it is now, taken out of the scene (`picture`
    /// puts them back once it has drawn them).
    fn levels_now(&mut self) -> Levels {
        Levels::of(self.levels.take(), &self.spectrum(), self.angle())
    }
}

/// A button drawn over the picture (logical pixels): true when it is clicked.
fn hud_button(f: &mut Frame, min: Vec2, max: Vec2, text: &str, pointer: Option<Vec2>) -> bool {
    let over = pointer.is_some_and(|p| p.cmpge(min).all() && p.cmple(max).all());
    hud_rect(f, min, max, if over { Color::hex(0x33405E) } else { Color::hex(0x222B40) });
    f.text_px((min + max) * 0.5, text, 12.5, pal::TEXT, Align2::CENTER_CENTER);
    over && f.ptr.clicked
}

/// A slider row in the panel's usual form (`ui::slider`), for a number that other
/// controls also set: a number beyond the ends of the track is left as it is (the knob
/// waits at the end), where `ui::slider` would pull it in. Typed numbers are kept to the
/// range.
fn free_slider(ui: &mut egui::Ui, label: &str, v: &mut f64, range: RangeInclusive<f64>, suffix: &str) -> bool {
    let (mut typed, mut dragged) = (*v, *v);
    let mut changed = false;
    crate::typeset::seen(label);
    ui.horizontal(|ui| {
        ui.label(ui::label_text(ui, label, 12.5, Some(pal::DIM), false));
        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
            let number = egui::DragValue::new(&mut typed).speed(0.01).range(range.clone()).clamp_existing_to_range(false).suffix(suffix).custom_formatter(|x, _| val(x));
            changed |= ui.add(number).changed();
        });
    });
    ui.scope(|ui| {
        ui.spacing_mut().slider_width = ui.available_width();
        changed |= ui.add(egui::Slider::new(&mut dragged, range).clamping(egui::SliderClamping::Never).show_value(false)).changed();
    });
    if typed != *v {
        *v = typed;
    } else if dragged != *v {
        *v = dragged;
    }
    changed
}

/// How quickly a name fades in or out (seconds for two thirds of the way).
const NAME_FADE: f32 = 0.06;

/// The names on the picture that come and go: behind the surface as the view turns,
/// crowded out, or not wanted where x says the same. Each fades in and out rather than
/// popping, and is drawn in every frame (unseen while faded out), at the place it last
/// had, so that the names never trade places with each other.
#[derive(Default)]
struct Names {
    shown: std::collections::HashMap<(&'static str, u32), f32>,
    /// Whether one is still on its way in or out, or to a new side.
    fading: bool,
    /// Where the name that moves with x is drawn, from x on the screen, and the side
    /// it is going to.
    beside_x: Option<(Vec2, Vec2)>,
}

/// How slowly the name that moves with x glides to a new side (seconds for two thirds
/// of the way): slower than names that look for room usually do, since it may have to
/// change sides all the while x is dragged round.
const X_NAME_GLIDE: f32 = 0.16;

impl Names {
    /// How far the name `tag` is faded in, after `dt` more seconds of being wanted
    /// (`show`) or not.
    fn fade(&mut self, tag: (&'static str, u32), show: bool, dt: f32) -> f32 {
        let alpha = self.shown.entry(tag).or_insert(if show { 1.0 } else { 0.0 });
        let goal = if show { 1.0 } else { 0.0 };
        *alpha += (goal - *alpha) * (1.0 - (-dt.max(0.0) / NAME_FADE).exp());
        if (*alpha - goal).abs() < 0.01 || dt <= 0.0 {
            *alpha = goal;
        } else {
            self.fading = true;
        }
        *alpha
    }

    /// Name the point `at` that moves with x, on the clearest side (the side asked for,
    /// or the one it has, or the nearest to that with room, as `Room::label` chooses),
    /// gliding there when the side changes.
    #[allow(clippy::too_many_arguments)]
    fn name_moving(&mut self, f: &mut Frame, room: &mut Room, at: Vec3, text: &str, size: f32, color: Color, towards: Vec2, gap: f32) {
        let Some(from) = on_screen(f, at) else { return };
        let half = half_box(f, text, size);
        let place = |side: Vec2| side * gap + Vec2::new(side.x * half.x, side.y * half.y);
        let want = towards.normalize_or_zero();
        let mut sides: Vec<Vec2> = (0..8).map(|k| Vec2::from_angle(k as f32 * std::f32::consts::TAU / 8.0)).collect();
        sides.sort_by(|p, q| q.dot(want).total_cmp(&p.dot(want)));
        sides.insert(0, want);
        let kept = self.beside_x.map(|(_, side)| side);
        let room_at = |room: &Room, side: Vec2| room.room_for(from, from + place(side), half);
        let side = if room_at(room, want) >= 5.0 {
            want
        } else if let Some(side) = kept.filter(|s| room_at(room, *s) >= 3.0) {
            side
        } else {
            sides.iter().copied().map(|s| (room_at(room, s), s)).fold((f32::MIN, want), |best, c| if c.0 > best.0 + 0.5 { c } else { best }).1
        };
        let goal = place(side);
        let drawn = match self.beside_x {
            Some((drawn, _)) if f.real_dt > 0.0 => {
                let d = drawn + (goal - drawn) * (1.0 - (-f.real_dt / X_NAME_GLIDE).exp());
                if (goal - d).length() < 0.5 || !d.is_finite() {
                    goal
                } else {
                    self.fading = true;
                    d
                }
            }
            _ => goal,
        };
        self.beside_x = Some((drawn, side));
        room.boxed(from + goal - half, from + goal + half);
        f.text_px(from + drawn, text, size, color, Align2::CENTER_CENTER);
    }

    /// Name the scene's point `at` with `text`, in `color` faded as far as it is: placed
    /// by `room` while it is wanted (and, with `first`, only on the side towards
    /// `towards`, or not at all: for a name that matters less than what crowds it, one
    /// that moves round its point is worse than none), and while it fades out, left
    /// where it was.
    #[allow(clippy::too_many_arguments)]
    fn name(&mut self, f: &mut Frame, room: &mut Room, tag: (&'static str, u32), at: Vec3, text: &str, size: f32, color: Color, towards: Vec2, gap: f32, show: bool, first: bool) -> bool {
        let Some(from) = on_screen(f, at) else { return false };
        // (the box as `Room::label_any` measures it, on the side asked for)
        let half = half_box(f, text, size);
        let side = towards.normalize_or_zero();
        let wanted = from + side * gap + Vec2::new(side.x * half.x, side.y * half.y);
        // (whether it would fit matters only for a name that is wanted: most of the
        // highest and lowest points' places are not, and the search is the dear part)
        let fits = !first || !show || room.room_for(from, wanted, half) >= 3.0;
        let alpha = self.fade(tag, show && fits, f.real_dt);
        if show && fits && alpha > 0.0 {
            if first {
                room.boxed(wanted - half, wanted + half);
                f.text_placed(tag.0, tag.1, from, wanted - from, text, size, color.a(color.0[3] as f32 / 255.0 * alpha));
            } else {
                room.label(f, tag, at, text, size, color.a(color.0[3] as f32 / 255.0 * alpha), towards, gap, false);
            }
        } else {
            // (all but gone, it is written as nothing: the shadow under every name is
            // drawn as dark however faint the name is)
            let offset = f.places.chosen(tag.0, tag.1).unwrap_or(wanted - from);
            let text = if alpha < 0.05 { "" } else { text };
            f.text_placed(tag.0, tag.1, from, offset, text, size, color.a(color.0[3] as f32 / 255.0 * alpha));
        }
        show && fits
    }
}

/// A filled disc in screen space, on top of the panels (`radius` in logical pixels).
fn hud_disc(f: &mut Frame, centre: Vec2, radius: f32, color: Color) {
    let n = 16;
    let middle = f.hud_at(2, centre);
    for k in 0..n {
        let (a, b) = (Vec2::from_angle(k as f32 / n as f32 * std::f32::consts::TAU), Vec2::from_angle((k + 1) as f32 / n as f32 * std::f32::consts::TAU));
        let (a, b) = (f.hud_at(2, centre + a * radius), f.hud_at(2, centre + b * radius));
        f.dl.tris_alpha.tri(middle, a, b, color);
    }
}

/// The words of a text, a formula between dollars one word however many spaces it has:
/// a line is never broken inside one.
fn words(text: &str) -> Vec<&str> {
    let bytes = text.as_bytes();
    let (mut words, mut start, mut inside, mut i) = (Vec::new(), 0, false, 0);
    while i < bytes.len() {
        match bytes[i] {
            // (a `\$` is a dollar sign, not the end of a formula)
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

/// Text broken into lines no wider than `width` logical pixels (as `text_width` judges
/// each word, a formula from the fonts' metrics).
fn wrap(text: &str, size: f32, width: f32) -> Vec<String> {
    let space = size * 0.47;
    let mut lines = vec![String::new()];
    // (the last line's width)
    let mut wide = 0.0;
    for word in words(text) {
        let w = text_width(word, size);
        let last = lines.last_mut().expect("there is always a line");
        if last.is_empty() {
            last.push_str(word);
            wide = w;
        } else if wide + w + space > width {
            lines.push(word.to_string());
            wide = w;
        } else {
            last.push(' ');
            last.push_str(word);
            wide += space + w;
        }
    }
    lines
}

/// How many lines `wrap` breaks the text into, without making them.
fn line_count(text: &str, size: f32, width: f32) -> usize {
    // (the same sums as `wrap` makes, in the same order)
    let space = size * 0.47;
    let (mut lines, mut empty, mut wide) = (1, true, 0.0);
    for word in words(text) {
        let w = text_width(word, size);
        if empty {
            (empty, wide) = (word.is_empty(), w);
        } else if wide + w + space > width {
            (lines, empty, wide) = (lines + 1, word.is_empty(), w);
        } else {
            wide += space + w;
        }
    }
    lines
}

/// How wide a text is on the screen: one with mathematics in it exactly (and kept from
/// frame to frame, as `Frame::text_size` keeps it), words alone as `text_width` judges.
/// None for one with mathematics in it before the fonts are in place (the first frame).
fn measured(f: &Frame, text: &str, size: f32) -> Option<f32> {
    if crate::typeset::has_math(text) {
        crate::typeset::measure(f.ctx, text, size).map(|s| s.x)
    } else {
        Some(text_width(text, size))
    }
}

/// The same, and before the fonts are in place as `text_width` judges it.
fn width(f: &Frame, text: &str, size: f32) -> f32 {
    measured(f, text, size).unwrap_or_else(|| text_width(text, size))
}

/// Half the box a name takes, as `Room::label` measures it: a name with mathematics in
/// it exactly, one of words alone a little wider than it is likely to be.
fn half_box(f: &Frame, text: &str, size: f32) -> Vec2 {
    if crate::typeset::has_math(text) {
        f.text_size(text, size) * 0.5 + Vec2::new(4.0, 0.0)
    } else {
        Vec2::new(text_width(text, size) * 0.5 + 4.0, size * 0.5 + 1.0)
    }
}

/// The label of one of egui's own widgets (a button, a checkbox), typeset: they set
/// their labels as plain text.
fn typeset_label(ui: &egui::Ui, text: &str) -> egui::WidgetText {
    crate::typeset::seen(text);
    ui::label_text(ui, text, 12.5, None, false)
}

impl Scene for QuadForm {
    fn view(&self) -> View {
        self.the_view()
    }

    fn take_view(&mut self) -> Option<View> {
        std::mem::take(&mut self.new_view).then(|| self.the_view())
    }

    fn settings(&self, s: &mut ViewSettings) {
        s.grid = false;
    }

    fn animated(&self) -> bool {
        self.moving() || self.scale != self.scale_goal() || self.names.fading
    }

    fn ui(&mut self, ui: &mut egui::Ui, _: &Gfx) {
        ui::note(ui, "A symmetric matrix $A$ gives every point $x$ of the plane a height:");
        ui::equation(ui, r"x^\top A x = a x_1^2 + 2b x_1 x_2 + c x_2^2");
        ui::note(ui, "Over the plane that is a bowl, a saddle or a trough, and the matrix's eigenvectors are its axes: along each, the height is its eigenvalue times the squared distance from $0$. The white ring is the unit circle lifted to its heights; drag $x$ round it. Near its lowest point a smooth function of two numbers usually looks like one of these bowls, which is why they matter.");

        ui::section(ui, "Things to try");
        for (k, trial) in TRIALS.iter().enumerate() {
            if ui::experiment(ui, trial.label, trial.look_for) {
                self.set_up(k);
            }
        }

        ui::section(ui, "The matrix");
        let mut m = self.form;
        let mut changed = free_slider(ui, "$a$", &mut m.a, -4.0..=4.0, "");
        changed |= free_slider(ui, "$b$", &mut m.b, -4.0..=4.0, "");
        changed |= free_slider(ui, "$c$", &mut m.c, -4.0..=4.0, "");
        if changed {
            self.set_form(m);
        }
        ui::equation(ui, r"A = \begin{bmatrix} a & b \\ b & c \end{bmatrix}");
        ui::note(ui, "$A$ is symmetric: $b$ stands in both corners, and the form's $x_1 x_2$ term is $2b$.");
        ui::section(ui, "Or by its eigenvalues");
        let s = self.spectrum();
        let (mut l1, mut l2, mut degrees) = (s.l1, s.l2, self.angle().to_degrees());
        let first = free_slider(ui, r"$\lambda_1$", &mut l1, -4.0..=4.0, "");
        let second = free_slider(ui, r"$\lambda_2$", &mut l2, -4.0..=4.0, "");
        let turned = free_slider(ui, r"angle of $\lambda_1$'s axis", &mut degrees, -90.0..=90.0, "°");
        // (a number typed in that is not one is ignored)
        if (first || second || turned) && l1.is_finite() && l2.is_finite() && degrees.is_finite() {
            // (λ₁ is the larger: one dragged past the other takes it along)
            if first {
                l2 = l2.min(l1);
            } else if second {
                l1 = l1.max(l2);
            }
            self.set_form(Form::of(l1, l2, degrees.to_radians()));
            // (kept also when every direction is an eigenvector, and the matrix cannot say it)
            self.axis = degrees.to_radians();
        }
        ui::note(ui, r"$\lambda_1$ is the larger: drag one past the other and it takes the other along. $\lambda_2$'s axis is at right angles to $\lambda_1$'s.");
        ui::section(ui, "Show");
        ui.horizontal_wrapped(|ui| {
            if ui.button(if self.turned { "turn the axes back" } else { "turn the axes onto the eigenvectors" }).clicked() {
                self.turned = !self.turned;
            }
            if ui.button(typeset_label(ui, "$x$ once round the circle")).clicked() {
                self.sweep = Some((self.phi, 0.0));
            }
        });
        ui.checkbox(&mut self.descent, typeset_label(ui, "gradient descent from $x$"));
        ui::note(ui, r"Each step goes straight downhill, $x \leftarrow x - h \nabla(x^\top A x)$, with $h$ nine tenths of the size at which the steps would start to grow ($h = 0.9/|\lambda|$ for the $\lambda$ furthest from $0$).");

        ui::section(ui, "Reading the picture");
        ui::key(
            ui,
            &[
                (SURFACE, r"the surface: the height $x^\top A x$ above each point $x$ of the plane, with level curves (thin dark lines)"),
                (CMAP.color(0.75), "the floor below: the same heights as colours, paler higher, with the same level curves"),
                (V1, r"orange: the eigenvector of $\lambda_1$, the larger eigenvalue: its axis on the floor, and the surface above it, a parabola $\lambda_1 t^2$"),
                (V2, r"purple: the same for $\lambda_2$"),
                (RING, "white ring: the unit circle lifted to its heights; $x$ is the dot on it, and its height is also in the graph"),
                (LEVEL, r"yellow: the level curve $x^\top A x = 1$ (and $x^\top A x = -1$), on the floor and at its height on the surface"),
                (AXES, "pale arrows: the axes $x_1$ and $x_2$ (turned: $y_1$ and $y_2$)"),
                (pal::TEXT.a(0.55), "dashed pale lines (a saddle): where the height is $0$"),
                (DESCENT, "cyan (when shown): gradient descent from $x$"),
            ],
        );

        ui::section(ui, "Words");
        ui::term(ui, "Quadratic form:", r"$x^\top A x = a x_1^2 + 2b x_1 x_2 + c x_2^2$: a number for every $x$, made from a symmetric matrix $A$. Twice as far out, four times as high.");
        ui::term(ui, "Symmetric matrix:", "one that equals its transpose. Its eigenvalues are real, and its eigenvectors are at right angles.");
        ui::term(ui, "Eigenvector, eigenvalue:", r"$Av = \lambda v$: a direction that $A$ does not turn, only stretches by $\lambda$.");
        ui::term(ui, "Positive definite:", r"$x^\top A x > 0$ for every $x$ but $0$: both eigenvalues positive. Negative definite: both negative. Indefinite: one of each. Semidefinite: one of them is $0$.");
        ui::term(ui, "Rayleigh quotient:", r"$x^\top A x / x^\top x$, the height over the unit circle (where $x^\top x = 1$): never above $\lambda_1$ nor below $\lambda_2$.");
        ui::term(ui, "Level curve:", r"the points of one height. For a bowl, $x^\top A x = 1$ is an ellipse with half-axes $1/\sqrt{\lambda_1}$ and $1/\sqrt{\lambda_2}$ along the eigenvectors; for a saddle, a hyperbola.");
        ui::term(ui, "Spectral theorem:", r"a symmetric $A$ is $Q \Lambda Q^\top$, $Q$'s columns its eigenvectors (at right angles), $\Lambda$ its eigenvalues on the diagonal. In the coordinates $y = Q^\top x$ the form is $\lambda_1 y_1^2 + \lambda_2 y_2^2$.");
        ui::term(ui, r"Gradient $\nabla$:", r"an arrow pointing straight uphill, as long as the slope is steep. $\nabla(x^\top A x) = 2Ax$.");
        ui::term(ui, "Hessian:", "the matrix of second derivatives of a function of several numbers. Near a minimum the function is nearly its value there plus the quadratic form of half its Hessian: a bowl, and a narrow one slows gradient descent.");

        ui::section(ui, "Numbers");
        let s = self.spectrum();
        let m = self.form;
        // (the values as formulas, every one alike: one of them is a point)
        ui::readout(ui, "$a$, $b$, $c$", format!("${}, {}, {}$", val(m.a), val(m.b), val(m.c)));
        ui::readout_colored(ui, r"$\lambda_1$", format!("${}$", val(s.l1)), V1);
        ui::readout_colored(ui, r"$\lambda_2$", format!("${}$", val(s.l2)), V2);
        ui::readout(ui, r"angle of $\lambda_1$'s axis", if s.theta.is_some() { format!("${}°$", num(self.angle().to_degrees(), 1)) } else { "any: every direction".to_string() });
        ui::readout(ui, r"$\tr A = a + c = \lambda_1 + \lambda_2$", format!("${}$", val(m.a + m.c)));
        ui::readout(ui, r"$\det A = ac - b^2 = \lambda_1 \lambda_2$", format!("${}$", val(m.det())));
        if s.l2 > 0.0 {
            ui::readout(ui, r"$\lambda_1 / \lambda_2$", format!("${}$", val(s.l1 / s.l2)));
        }
        let x = self.x();
        ui::readout_colored(ui, "$x$", format!("$({}, {})$", val(x.x), val(x.y)), RING);
        ui::readout_colored(ui, r"its height $x^\top A x$", format!("${}$", val(m.at(x))), RING);
        if let Some(words) = scale_words(self.scale_goal() as f64) {
            ui::note(ui, &format!("These heights are large: the picture has its {words}. The numbers are the true ones."));
        }
        ui::see_also(ui, &[("eigen", "how eigenvalues are found"), ("surface", "any surface, with its tangent plane"), ("descent", "optimisers in narrow valleys"), ("pca", "the bowl of a cloud of points: principal components")]);
    }

    fn frame(&mut self, f: &mut Frame) {
        self.advance(f.dt);
        // the drawn scale of the heights glides to a new one (by the clock: it answers the hand)
        let goal = self.scale_goal();
        if self.scale != goal {
            let k = 1.0 - (-f.real_dt.max(0.0) / SCALE_GLIDE).exp();
            self.scale += (goal - self.scale) * k;
            if !self.scale.is_finite() || (self.scale - goal).abs() < 1e-3 * goal || f.real_dt <= 0.0 {
                self.scale = goal;
            }
        }

        // the panels' places follow the canvas, and the picture's place follows theirs
        let view = f.view_size();
        let lay = Layout::of(view, f.top_inset);
        // (nearly always the same canvas as last time, and so the same layout)
        let before = if (self.canvas, self.inset) == (view, f.top_inset) { lay } else { Layout::of(self.canvas, self.inset) };
        // (a canvas squeezed to a sliver is one the controls have opened over, on a
        // phone: the picture is not seen then, and is left as it was for when they close)
        let covered = view.x < 160.0;
        let changed = !covered && (lay.narrow != before.narrow || lay.card.is_some() != before.card.is_some() || (view - self.canvas).abs().max_element() > 24.0);
        let first = !covered && !self.measured && (view - self.canvas).abs().max_element() > 24.0;
        self.new_view |= changed || first;
        if !covered && (changed || !self.measured) {
            self.canvas = view;
            self.measured = true;
        }
        self.inset = f.top_inset;
        let pointer = f.ptr.pos.map(|p| p / f.px);
        if pointer.is_some_and(|p| lay.covers(p)) {
            // the panels are for reading and their button: no turning the view from there
            f.handles.capture();
            f.handles.claim_clicks();
        }

        self.drag_x(f, &lay);
        self.fill_surfaces(f);
        self.picture(f, &lay);
        if let Some(at) = lay.card {
            self.card(f, at, pointer);
        }
        if let Some(at) = lay.graph {
            self.graph(f, at);
        }
        if !lay.narrow {
            let mut parts = vec!["floor: the heights as colours, paler higher", "white ring: the unit circle, lifted to its heights"];
            let words = scale_words(self.scale as f64);
            if let Some(w) = words.as_deref() {
                parts.push(w);
            }
            bottom_caption(f, &parts, 12.0);
        }
    }
}

impl QuadForm {
    /// The point on the ring above x at angle `t`, as drawn.
    fn ring_point(&self, t: f64) -> Vec3 {
        let x = unit(t);
        Vec3::new(x.x as f32, x.y as f32, self.form.at(x) as f32 * self.scale)
    }

    /// x is dragged round the ring by the angle of the point under the pointer on a level
    /// plane at the ring's middle height. (Not the plane through x itself: that rises and
    /// falls as x goes round, and the angle found on it would chase its own tail. Nor the
    /// nearest point of the ring on the screen, which leaps from the near side to the far
    /// one where the two are about as near.)
    fn drag_x(&mut self, f: &mut Frame, lay: &Layout) {
        let mut p = self.ring_point(self.phi);
        let hidden = on_screen(f, p).is_some_and(|s| lay.covers(s));
        if !f.handles.dragging() {
            self.grab = None;
        }
        if hidden && !f.handles.dragging() {
            return;
        }
        if !f.drag3(1, &mut p, Some(Vec3::Z), RING) {
            return;
        }
        let s = self.spectrum();
        let middle = Vec3::new(0.0, 0.0, (0.5 * (s.l1 + s.l2)) as f32 * self.scale);
        let Some(hit) = f.ptr.pos.and_then(|q| f.cam.pick_plane(q, middle, Vec3::Z)) else { return };
        if hit.truncate().length() < 1e-6 || !hit.is_finite() {
            return;
        }
        // (x keeps the angle it was grabbed at from the pointer's: it does not leap to it)
        let under = f64::from(hit.y).atan2(f64::from(hit.x));
        let grab = *self.grab.get_or_insert(self.phi - under);
        let t = under + grab;
        // (the angle carried on from where x was, not wrapped back: a sweep or a drag
        // goes on round, and the graph and the numbers take it modulo a turn)
        let along = (t - self.phi + std::f64::consts::PI).rem_euclid(std::f64::consts::TAU) - std::f64::consts::PI;
        self.phi += along;
        self.sweep = None;
        if let Some(m) = self.morph.as_mut() {
            m.phi = (self.phi, self.phi);
        }
    }

    /// The surface and the floor, filled again when the matrix or the drawn scale has changed.
    fn fill_surfaces(&mut self, f: &mut Frame) {
        let (form, scale) = (self.form, self.scale);
        let s = form.spectrum();
        let floor_z = self.floor_z();
        // (outwards first and round second: the shader's normals, made from the two
        // directions in that order, then point up, and the top is the side lit as the front)
        let surf = self.surf.get_or_insert_with(|| Surface::new(f.gfx, OUT, ROUND));
        let floor = self.floor.get_or_insert_with(|| Surface::new(f.gfx, OUT, ROUND));
        if self.filled != Some((form, scale)) {
            for (radius, z_scale, out) in [(SURF_R, scale, &mut *surf), (FLOOR_R, 0.0, &mut *floor)] {
                let (nx, ny) = out.size();
                let data = out.data_mut();
                for j in 0..ny {
                    // (the last row round is the first again, to the last bit: a hair
                    // apart, the two edges leave a crack the floor shows through)
                    let d = unit(std::f64::consts::TAU * (j % (ny - 1)) as f64 / (ny - 1) as f64);
                    for i in 0..nx {
                        let x = d * (radius * i as f64 / (nx - 1) as f64);
                        let q = form.at(x);
                        data[j * nx + i] = [x.x as f32, x.y as f32, q as f32 * z_scale, q as f32];
                    }
                }
            }
            self.filled = Some((form, scale));
        }
        // one colour scale for both: the heights the floor reaches
        let r2 = FLOOR_R * FLOOR_R;
        let (mut lo, mut hi) = ((s.l2.min(0.0) * r2) as f32, (s.l1.max(0.0) * r2) as f32);
        if hi - lo < 1e-6 {
            (lo, hi) = (-1.0, 1.0);
        }
        // (level curves as close together as the surface needs: a few across its heights)
        let span = ((s.l1.max(0.0) - s.l2.min(0.0)) * SURF_R * SURF_R) as f32;
        let contour = if span > 1e-9 { nice_step(span / 5.0) } else { 0.0 };
        for (k, out) in [&mut *surf, &mut *floor].into_iter().enumerate() {
            let st = &mut out.style;
            st.cmap = Some(CMAP);
            st.range = [lo, hi];
            st.contour = contour;
            // (not joined round: with the rows round joined, the shader's normals along
            // the seam come out pointing down, and a thin dark wedge shows there; the
            // first and last rows are the same points, and one-sided normals do)
            st.wrap = [false, false];
            if k == 0 {
                st.cmap = None;
                st.color = SURFACE;
                st.contour_alpha = 0.45;
                st.specular = 0.12;
                st.model = Mat4::IDENTITY;
            } else {
                st.contour_alpha = 0.55;
                st.lit = 0.0;
                st.model = Mat4::from_translation(Vec3::new(0.0, 0.0, floor_z));
            }
        }
        f.surface(surf);
        f.surface(floor);
    }

    /// Is the point `p` hidden from the eye behind the surface? (The surface is all that
    /// stands in the way: it is the only thing drawn solid above the floor.) The line of
    /// sight is followed from the eye, and it is hidden if it passes through the surface
    /// on the way, from above it to below or the other way.
    fn hidden(&self, f: &Frame, p: Vec3) -> bool {
        let eye = f.cam.eye;
        let n = 64;
        let mut side: Option<bool> = None;
        // (not quite all the way: a point on the surface is not hidden by itself)
        for i in 0..=n {
            let r = eye + (p - eye) * (0.985 * i as f32 / n as f32);
            let x = DVec2::new(r.x as f64, r.y as f64);
            // (a sight line that leaves the disc below the rim has passed through the
            // surface just inside it: the samples just outside count as at the rim)
            if x.length() > SURF_R * 1.04 {
                side = None;
                continue;
            }
            let x = if x.length() > SURF_R { x * (SURF_R / x.length()) } else { x };
            let above = r.z >= self.form.at(x) as f32 * self.scale;
            if side.is_some_and(|s| s != above) {
                return true;
            }
            side = Some(above);
        }
        false
    }

    /// Everything drawn in the scene's space, and the names on it.
    fn picture(&mut self, f: &mut Frame, lay: &Layout) {
        let s = self.spectrum();
        let form = self.form;
        let scale = self.scale;
        let fz = self.floor_z();
        let angle = self.angle();
        let quarter = 0.5 * std::f64::consts::PI;
        let (v1, v2) = (unit(angle), unit(angle + quarter));
        let on_floor = |p: DVec2| Vec3::new(p.x as f32, p.y as f32, fz);
        let lifted = |p: DVec2| Vec3::new(p.x as f32, p.y as f32, form.at(p) as f32 * scale);
        let mut room = Room::new(f);
        for (min, max) in [lay.card, lay.graph].into_iter().flatten() {
            room.wall(min, max);
        }
        let view = f.view_size();
        if !lay.narrow && view.y >= 300.0 {
            // (the caption along the bottom)
            room.wall(Vec2::new(0.0, view.y - 40.0), view);
        }
        let round = s.theta.is_none();

        // the floor's rim and the unit circle on it
        let rim: Vec<Vec3> = (0..96).map(|i| on_floor(unit(i as f64 / 96.0 * std::f64::consts::TAU) * FLOOR_R)).collect();
        f.dl.lines.closed(&rim, 1.0, pal::DIM.a(0.5));
        let circle: Vec<Vec3> = (0..RING_N).map(|i| on_floor(unit(i as f64 / RING_N as f64 * std::f64::consts::TAU))).collect();
        f.dl.lines.closed(&circle, 1.3, RING.a(0.45));
        for k in (0..RING_N).step_by(8) {
            room.line(f, circle[k], circle[(k + 8) % RING_N]);
        }
        for k in (0..96).step_by(4) {
            room.line(f, rim[k], rim[(k + 4) % 96]);
        }

        // the coordinate axes, turned as far as they have got (and, once on the
        // eigenvectors, in their colours)
        let alpha = self.alpha();
        let arrived = if round { 0.0 } else { smoothstep(self.turn) };
        let axes = [(unit(alpha), AXES.mix(V1, arrived)), (unit(alpha + quarter), AXES.mix(V2, arrived))];
        let reach = FLOOR_R + 0.18;
        for (d, c) in axes {
            let (a, b) = (on_floor(-d * FLOOR_R), on_floor(d * reach));
            f.arrow(a, b, 1.6, c.a(0.9));
            room.line(f, a, b);
        }

        // where the height is 0, if not only at 0: two lines for a saddle (a trough's
        // is its own axis, drawn next)
        if kind(&s) == Kind::Saddle {
            let (p, q) = (s.l1.sqrt(), (-s.l2).sqrt());
            for side in [1.0, -1.0] {
                let d = (v1 * q + v2 * (side * p)).normalize();
                let (a, b) = (on_floor(-d * FLOOR_R), on_floor(d * FLOOR_R));
                f.dashed(a, b, 1.2, pal::TEXT.a(0.55));
                room.line(f, a, b);
            }
        }

        // the eigenvectors' axes on the floor, and the parabolas λt² above them
        if !round {
            for (v, c) in [(v1, V1), (v2, V2)] {
                let (a, b) = (on_floor(-v * FLOOR_R), on_floor(v * FLOOR_R));
                f.dashed(a, b, 1.8, c);
                room.line(f, a, b);
                let n = 64;
                let pts: Vec<Vec3> = (0..=n).map(|i| v * (SURF_R * (2.0 * i as f64 / n as f64 - 1.0))).map(lifted).collect();
                f.path(pts.iter().copied(), 2.6, c);
                for k in (0..n).step_by(8) {
                    room.line(f, pts[k], pts[(k + 8).min(n)]);
                }
            }
        }

        // the level curves xᵀAx = 1 and = −1: on the floor, and at their height on the
        // surface, where it reaches. Each is named where it is nearest the eye (and not
        // under the surface).
        let toward = unit(f64::from(self.yaw));
        let mut level_marks: Vec<(Vec3, &str)> = Vec::new();
        let levels = self.levels_now();
        for ((level, text), (on_the_floor, on_the_surface)) in [(1.0, r"$x^\top A x = 1$"), (-1.0, r"$x^\top A x = -1$")].into_iter().zip(&levels.curves) {
            let mut nearest: Option<DVec2> = None;
            for curve in on_the_floor {
                let pts: Vec<Vec3> = curve.iter().map(|p| on_floor(*p)).collect();
                f.path(pts.iter().copied(), 2.2, LEVEL);
                for k in (0..pts.len() - 1).step_by(10) {
                    room.line(f, pts[k], pts[(k + 10).min(pts.len() - 1)]);
                }
                for p in curve.iter().filter(|p| p.length() > SURF_R * 1.05) {
                    if nearest.is_none_or(|n| p.dot(toward) > n.dot(toward)) {
                        nearest = Some(*p);
                    }
                }
            }
            for curve in on_the_surface {
                f.path(curve.iter().map(|p| Vec3::new(p.x as f32, p.y as f32, level as f32 * scale)), 2.0, LEVEL.a(0.9));
            }
            if let Some(p) = nearest {
                level_marks.push((on_floor(p), text));
            }
        }
        self.levels = Some(levels);

        // the ring, its highest and lowest points, and x on it
        let ring: Vec<Vec3> = (0..RING_N).map(|i| self.ring_point(i as f64 / RING_N as f64 * std::f64::consts::TAU)).collect();
        f.dl.lines.closed(&ring, 2.8, RING);
        for k in (0..RING_N).step_by(6) {
            room.line(f, ring[k], ring[(k + 6) % RING_N]);
        }
        if !round {
            for (v, c) in [(v1, V1), (v2, V2)] {
                for side in [1.0, -1.0] {
                    let top = lifted(v * side);
                    f.dot(top, 10.0, c);
                    f.dashed(top, on_floor(v * side), 1.0, c.a(0.5));
                }
            }
        }
        let x = self.x();
        let (top, foot) = (lifted(x), on_floor(x));
        f.dashed(top, foot, 1.4, RING.a(0.7));
        f.dot(foot, 8.0, RING);
        room.line(f, top, foot);
        if let Some(at) = on_screen(f, top) {
            room.boxed(at - Vec2::splat(9.0), at + Vec2::splat(9.0));
        }

        // gradient descent from x: on the surface, and its shadow on the floor
        if self.descent {
            let path = descent(&form, x, DESCENT_STEPS, FLOOR_R);
            let floor_path: Vec<Vec3> = path.iter().map(|p| on_floor(*p)).collect();
            f.path(floor_path.iter().copied(), 1.4, DESCENT.a(0.7));
            f.dl.lines.brk();
            for p in &path {
                if p.length() <= SURF_R {
                    let q = lifted(*p);
                    f.dl.lines.push(q, 2.2, DESCENT);
                    f.dot(q + Vec3::Z * 0.01, 5.0, DESCENT);
                } else {
                    f.dl.lines.brk();
                }
            }
            f.dl.lines.brk();
            for w in path.windows(2).filter(|w| w[1].length() <= SURF_R) {
                room.line(f, lifted(w[0]), lifted(w[1]));
            }
        }

        // the names: first those of what stays put, then x's height, which moves with
        // it, then the ring's highest and lowest points, which make way for it. Those
        // the surface hides fade out.
        let outwards = |f: &Frame, p: Vec3| match (on_screen(f, Vec3::new(0.0, 0.0, p.z)), on_screen(f, p)) {
            (Some(a), Some(b)) if (b - a).length() > 1.0 => b - a,
            _ => Vec2::new(1.0, -1.0),
        };
        let mut names = std::mem::take(&mut self.names);
        names.fading = false;
        if !round {
            // the eigenvalues, at the end of each axis nearer the eye (just beyond the
            // floor's rim, where no level curve reaches); once the axes turn onto the
            // eigenvectors their arrows take those ends, and the names go to the others
            let turning = self.turned || self.turn > 0.0;
            for (k, (v, l, c, name)) in [(v1, s.l1, V1, r"\lambda_1"), (v2, s.l2, V2, r"\lambda_2")].into_iter().enumerate() {
                let near = if turning { -v } else if v.dot(toward) >= 0.0 { v } else { -v };
                let at = on_floor(near * (FLOOR_R + 0.06));
                let dir = outwards(f, at);
                let show = !self.hidden(f, at);
                names.name(f, &mut room, ("eigen", k as u32), at, &format!("${name} = {}$", val(l)), 14.0, c, dir, 6.0, show, false);
            }
        }
        let axis_names = if self.turn > 0.0 { ["$y_1$", "$y_2$"] } else { ["$x_1$", "$x_2$"] };
        for (k, (d, c)) in axes.iter().enumerate() {
            let at = on_floor(*d * reach);
            let dir = outwards(f, at);
            let show = !self.hidden(f, at);
            names.name(f, &mut room, ("axis", k as u32), at, axis_names[k], 14.0, *c, dir, 4.0, show, false);
        }
        for (k, (at, text)) in level_marks.iter().enumerate() {
            let dir = outwards(f, *at);
            let show = !self.hidden(f, *at);
            names.name(f, &mut room, ("level", k as u32), *at, text, 13.0, LEVEL, dir, 7.0, show, false);
        }
        if kind(&s) == Kind::Saddle {
            let (p, q) = (s.l1.sqrt(), (-s.l2).sqrt());
            let ends = [1.0, -1.0].map(|side| (v1 * q + v2 * (side * p)).normalize());
            let end = [ends[0], -ends[0], ends[1], -ends[1]].into_iter().max_by(|a, b| a.dot(toward).total_cmp(&b.dot(toward))).unwrap_or(ends[0]);
            let at = on_floor(end * FLOOR_R * 0.92);
            let dir = outwards(f, at);
            let show = !self.hidden(f, at);
            names.name(f, &mut room, ("zero", 0), at, "height $0$", 12.0, pal::TEXT, dir, 6.0, show, false);
        }
        if round {
            let says = if kind(&s) == Kind::Flat { "flat: every height is $0$" } else { "the same height all round: every direction is an eigenvector" };
            // (on the floor's near edge)
            let at = on_floor(toward * FLOOR_R);
            room.label(f, "round", at, says, 13.0, pal::TEXT, Vec2::new(0.0, 1.0), 10.0, false);
        }
        let dir = outwards(f, top);
        names.name_moving(f, &mut room, top, &format!(r"$x^\top A x = {}$", val(form.at(x))), 15.0, RING, dir, 14.0);
        if !round {
            // the highest and lowest points, named beside the one nearer the eye, or
            // failing that the other: straight out from the middle, or a little to one
            // side, each a place of its own that the name fades into or out of, so that
            // it never slides round its point. Faded out as x comes near (it says the
            // height there itself).
            for (k, (v, l, c, word)) in [(v1, s.l1, V1, "highest"), (v2, s.l2, V2, "lowest")].into_iter().enumerate() {
                let near = if v.dot(toward) >= 0.0 { v } else { -v };
                let text = format!("{word} ${}$", val(l));
                let mut named = false;
                for (e, end) in [near, -near].into_iter().enumerate() {
                    let at = lifted(end);
                    let apart = (axis_angle(self.phi - end.y.atan2(end.x))).abs() as f32;
                    let dir = outwards(f, at);
                    let shown = apart > 0.45 && !self.hidden(f, at);
                    for (j, turn) in [0.0f32, 0.7, -0.7].into_iter().enumerate() {
                        let tag = ("mark", (k * 6 + e * 3 + j) as u32);
                        // (paler: the name may lie over the surface)
                        named |= names.name(f, &mut room, tag, at, &text, 13.0, c.mix(pal::WHITE, 0.3), Vec2::from_angle(turn).rotate(dir), 11.0, shown && !named, true);
                    }
                }
            }
        }
        let dir = outwards(f, foot);
        let show = !self.hidden(f, foot);
        names.name(f, &mut room, ("x", 0), foot, "$x$", 15.0, RING, dir, 8.0, show, true);
        self.names = names;
    }

    /// The card: the matrix, the form written out, the eigenvalues and the verdict, and
    /// the button that turns the axes.
    fn card(&mut self, f: &mut Frame, (min, max): (Vec2, Vec2), pointer: Option<Vec2>) {
        hud_panel(f, min, max);
        let s = self.spectrum();
        let m = self.form;
        let turned = m.turned(self.alpha());
        let x0 = min.x + 12.0;
        let mut y = min.y + 10.0;
        let size = CARD_TEXT;
        // (while the axes turn, the turned matrix's numbers pass through every value on
        // their way: they are written to two places, so that the cells keep their width)
        let moving = self.turn > 0.0 && self.turn < 1.0;
        let write = |v: f64| if moving { num(v, 2) } else { val(v) };
        // (a minus sign counted as two characters: in a formula it is nearly twice as
        // wide as a digit)
        let chars = |t: String| t.chars().map(|c| if c == '−' { 2 } else { 1 }).sum::<usize>();
        let entries = [val(m.a), val(m.b), val(m.c), write(turned.a), write(turned.b), write(turned.c), num(turned.a, 2), num(turned.c, 2)].map(chars);
        // (and room for a minus sign whether or not there is one, so that a number
        // passing through 0 on its way somewhere does not widen the matrix)
        let widest = entries.iter().copied().max().unwrap_or(6).max(6);
        let msize = matrix_size(2, 2, widest, size);
        // (the numbers set as formulas, as the names of the matrices are)
        let rows = |m: &Form, diag: [Color; 2], off: Color, write: &dyn Fn(f64) -> String| {
            let cell = |v: f64, c: Color| (format!("${}$", write(v)), c);
            vec![vec![cell(m.a, diag[0]), cell(m.b, off)], vec![cell(m.b, off), cell(m.c, diag[1])]]
        };
        f.text_px(Vec2::new(x0, y + msize.y * 0.5), "$A =$", 14.0, pal::TEXT, Align2::LEFT_CENTER);
        let mx = x0 + width(f, "$A =$", 14.0) + 6.0;
        hud_matrix(f, Vec2::new(mx, y), &rows(&m, [pal::TEXT; 2], pal::TEXT, &val), widest, size, pal::DIM);
        // (QᵀAQ has its place beside A whether or not it is there, so that nothing moves
        // when it comes)
        let qx = mx + msize.x + 14.0;
        let qm = qx + width(f, r"$Q^\top A Q =$", 14.0) + 6.0;
        if self.turn > 0.0 {
            // the same form in the turned coordinates, its corners fading to 0
            let fade = 1.0 - smoothstep(self.turn);
            let diag = [pal::TEXT.mix(V1, 1.0 - fade), pal::TEXT.mix(V2, 1.0 - fade)];
            f.text_px(Vec2::new(qx, y + msize.y * 0.5), r"$Q^\top A Q =$", 14.0, pal::TEXT, Align2::LEFT_CENTER);
            hud_matrix(f, Vec2::new(qm, y), &rows(&turned, diag, pal::TEXT.mix(pal::DIM, 1.0 - fade), &write), widest, size, pal::DIM);
        }
        // the eigenvalues, under the matrices
        y += msize.y + 6.0;
        f.text_px(Vec2::new(x0, y), format!(r"$\lambda_1 = {}$", val(s.l1)), 13.0, V1, Align2::LEFT_TOP);
        f.text_px(Vec2::new(x0 + 110.0, y), format!(r"$\lambda_2 = {}$", val(s.l2)), 13.0, V2, Align2::LEFT_TOP);
        if let Some(theta) = s.theta {
            f.text_px(Vec2::new(x0 + 220.0, y), format!(r"$\lambda_1$'s axis at {}°", num(theta.to_degrees(), 1)), 12.0, pal::DIM, Align2::LEFT_TOP);
        }
        y += 19.0;
        // the form, and in the turned coordinates (its "=" under the first line's: the
        // width of xᵀAx and the space TeX puts before a relation, 5/18 of a quad)
        f.text_px(Vec2::new(x0, y), format!(r"$x^\top A x = {}$", written(&m, ["x_1", "x_2"], true)), 13.0, pal::TEXT, Align2::LEFT_TOP);
        y += 19.0;
        if self.turn > 0.0 {
            let gone = self.turn >= 1.0 && turned.b == 0.0;
            let equals = x0 + width(f, r"$x^\top A x$", 13.0) + 13.0 * crate::typeset::TEXT_SCALE * 5.0 / 18.0;
            f.text_px(Vec2::new(equals, y), format!("$= {}$", written(&turned, ["y_1", "y_2"], !gone)), 13.0, pal::TEXT, Align2::LEFT_TOP);
        } else if s.theta.is_some() && m.b != 0.0 {
            f.text_px(Vec2::new(x0, y), "(turn the axes to write it without its $x_1 x_2$ term)", 12.0, pal::DIM, Align2::LEFT_TOP);
        }
        y += 21.0;
        // the verdict, and why
        let (word, why) = verdict(&s);
        f.text_px(Vec2::new(x0, y), word, 14.0, pal::WHITE, Align2::LEFT_TOP);
        y += 19.0;
        // (broken into lines again only when the verdict or the room changes: the
        // formulas in it are measured from the fonts' metrics)
        let room = max.x - x0 - 12.0;
        if self.why.as_ref().is_none_or(|(of, _)| *of != (why, room)) {
            self.why = Some(((why, room), wrap(why, 12.0, room)));
        }
        for line in self.why.iter().flat_map(|(_, lines)| lines) {
            f.text_px(Vec2::new(x0, y), line.as_str(), 12.0, pal::DIM, Align2::LEFT_TOP);
            y += 16.0;
        }
        let label = if self.turned { "turn the axes back" } else { "turn the axes onto the eigenvectors" };
        let bottom = max.y - 10.0;
        let width = text_width(label, 12.5) + 24.0;
        if hud_button(f, Vec2::new(x0, bottom - 24.0), Vec2::new(x0 + width, bottom), label, pointer) {
            self.turned = !self.turned;
        }
    }

    /// The graph of the ring's height against the angle of x, with the eigenvalues as its
    /// top and bottom.
    fn graph(&self, f: &mut Frame, (min, max): (Vec2, Vec2)) {
        hud_panel(f, min, max);
        let s = self.spectrum();
        f.text_px(min + Vec2::new(12.0, 8.0), r"height of the ring $x^\top A x$, against $x$'s angle", 12.0, pal::TEXT, Align2::LEFT_TOP);
        let plot_min = min + Vec2::new(40.0, 32.0);
        let plot_max = max - Vec2::new(92.0, 26.0);
        let size = plot_max - plot_min;
        if size.x < 20.0 || size.y < 20.0 {
            return;
        }
        let pad = 0.12 * (s.l1 - s.l2).max(0.2 * s.size()).max(0.1);
        let (lo, hi) = (s.l2.min(0.0).min(s.l2) - pad, s.l1.max(0.0) + pad);
        let px = |t: f64, h: f64| Vec2::new(plot_min.x + size.x * (t / std::f64::consts::TAU) as f32, plot_max.y - size.y * ((h - lo) / (hi - lo)) as f32);
        // 0, and the eigenvalues as the top and the bottom
        if lo < 0.0 && hi > 0.0 {
            hud_line(f, px(0.0, 0.0), px(std::f64::consts::TAU, 0.0), 1.0, pal::FAINT);
            f.text_px(px(0.0, 0.0) - Vec2::new(6.0, 0.0), "0", 11.0, pal::DIM, Align2::RIGHT_CENTER);
        }
        let dashes = LineStyle { dash: 6.0, ..Default::default() };
        for (l, c, name) in [(s.l1, V1, r"\lambda_1"), (s.l2, V2, r"\lambda_2")] {
            let (a, b) = (px(0.0, l), px(std::f64::consts::TAU, l));
            let (a3, b3) = (f.hud(a), f.hud(b));
            f.dl.lines_with(LineKind::Overlay, dashes).segment(a3, b3, 1.2, c);
            // (equal, they share one line and one name)
            let text = if s.theta.is_none() { format!(r"$\lambda_1 = \lambda_2 = {}$", val(l)) } else { format!("${name} = {}$", val(l)) };
            if s.theta.is_some() || name == r"\lambda_2" {
                // (beside the end of its line; a name longer than the room there, the
                // shared one, just inside the end instead, on the side of the line with
                // more of the graph: the curve is the line itself then)
                if width(f, &text, 12.0) <= max.x - b.x - 10.0 {
                    f.text_px(b + Vec2::new(6.0, 0.0), text, 12.0, c, Align2::LEFT_CENTER);
                } else if b.y < (plot_min.y + plot_max.y) * 0.5 {
                    f.text_px(Vec2::new(max.x - 6.0, b.y + 5.0), text, 12.0, c, Align2::RIGHT_TOP);
                } else {
                    f.text_px(Vec2::new(max.x - 6.0, b.y - 5.0), text, 12.0, c, Align2::RIGHT_BOTTOM);
                }
            }
        }
        // the angles of the axes, where the curve touches the top and the bottom
        let angle = self.angle().rem_euclid(std::f64::consts::PI);
        if s.theta.is_some() {
            for k in 0..4 {
                let t = (angle + k as f64 * 0.5 * std::f64::consts::PI).rem_euclid(std::f64::consts::TAU);
                let c = if k % 2 == 0 { V1 } else { V2 };
                hud_line(f, px(t, lo), px(t, hi), 1.0, c.a(0.35));
            }
        }
        // the angle along the bottom
        for (k, text) in ["0°", "90°", "180°", "270°", "360°"].iter().enumerate() {
            let t = k as f64 * 0.5 * std::f64::consts::PI;
            hud_line(f, px(t, lo), px(t, lo) + Vec2::new(0.0, 4.0), 1.0, pal::DIM);
            f.text_px(px(t, lo) + Vec2::new(0.0, 6.0), *text, 11.0, pal::DIM, Align2::CENTER_TOP);
        }
        hud_line(f, px(0.0, lo), px(std::f64::consts::TAU, lo), 1.0, pal::DIM.a(0.8));
        // the curve, and x on it
        let n = 180;
        let m = self.form;
        hud_path(f, (0..=n).map(|i| i as f64 / n as f64 * std::f64::consts::TAU).map(|t| px(t, m.at(unit(t)))), 2.0, RING);
        // x on it: the angle goes round, so near 0° x leaves one end of the graph as it
        // comes back in at the other, faded at the ends
        let t = self.phi.rem_euclid(std::f64::consts::TAU);
        for turn in [-1.0, 0.0, 1.0] {
            let at = t + turn * std::f64::consts::TAU;
            let beyond = ((-at).max(at - std::f64::consts::TAU) / (0.04 * std::f64::consts::TAU)).max(0.0);
            if beyond < 1.0 {
                hud_disc(f, px(at, m.at(unit(t))), 4.5, RING.a(1.0 - beyond as f32));
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// An experiment as its button sets it up, played to the end at 60 frames a second.
    fn trial(k: usize) -> QuadForm {
        let mut scene = QuadForm::create();
        scene.set_up(k);
        for _ in 0..2000 {
            scene.advance(1.0 / 60.0);
        }
        assert!(!scene.moving(), "experiment {k} is still moving");
        scene
    }

    /// Random symmetric matrices, the experiments' and some awkward ones.
    fn forms() -> Vec<Form> {
        let mut rng = Rng::new(3);
        let mut out: Vec<Form> = TRIALS.iter().map(|t| Form { a: t.form[0], b: t.form[1], c: t.form[2] }).collect();
        out.extend((0..400).map(|_| Form { a: rng.normal() as f64 * 2.0, b: rng.normal() as f64 * 2.0, c: rng.normal() as f64 * 2.0 }));
        out.extend([Form { a: 1.0, b: 0.0, c: 1.0 }, Form { a: -2.0, b: 0.0, c: -2.0 }, Form { a: 0.0, b: 0.0, c: 1.0 }, Form { a: 0.0, b: 1e-7, c: 0.0 }, Form { a: 4.0, b: 4.0, c: 4.0 }, Form { a: 1e-6, b: 0.0, c: 0.0 }]);
        out
    }

    /// The closed form agrees with the general method (`eig_sym`), rebuilds the matrix as
    /// QΛQᵀ, and its eigenvectors are at right angles and are sent along themselves.
    #[test]
    fn the_spectrum_rebuilds_the_matrix() {
        for m in forms() {
            let s = m.spectrum();
            let size = m.size().max(1e-300);
            assert!(s.l1 >= s.l2, "{m:?}: {s:?}");
            let (values, _) = crate::math::linalg::eig_sym(&[m.a, m.b, m.b, m.c], 2);
            let mut sorted = values.clone();
            sorted.sort_by(|a, b| b.total_cmp(a));
            assert!((sorted[0] - s.l1).abs() < 1e-9 * size && (sorted[1] - s.l2).abs() < 1e-9 * size, "{m:?}: {s:?} against {values:?}");
            let theta = s.theta.unwrap_or(0.0);
            let back = Form::of(s.l1, s.l2, theta);
            assert!((back.a - m.a).abs() < 1e-9 * size && (back.b - m.b).abs() < 1e-9 * size && (back.c - m.c).abs() < 1e-9 * size, "{m:?} {s:?} {back:?}");
            let (v1, v2) = (unit(theta), unit(theta + 0.5 * std::f64::consts::PI));
            assert!(v1.dot(v2).abs() < 1e-15);
            assert!((m.times(v1) - v1 * s.l1).length() < 1e-9 * size && (m.times(v2) - v2 * s.l2).length() < 1e-9 * size, "{m:?}");
            assert!(theta > -0.5 * std::f64::consts::PI && theta <= 0.5 * std::f64::consts::PI);
            // the trace and the determinant, as the panel says
            assert!((m.a + m.c - s.l1 - s.l2).abs() < 1e-9 * size && (m.det() - s.l1 * s.l2).abs() < 1e-9 * size * size);
        }
    }

    /// "Along each eigenvector, the height is its eigenvalue times the squared distance
    /// from 0" (the opening paragraph), and "twice as far out, four times as high".
    #[test]
    fn along_an_eigenvector_the_height_is_lambda_times_the_squared_distance() {
        for m in forms() {
            let s = m.spectrum();
            let size = m.size().max(1e-300);
            let theta = s.theta.unwrap_or(0.0);
            for (v, l) in [(unit(theta), s.l1), (unit(theta + 0.5 * std::f64::consts::PI), s.l2)] {
                for t in [-1.7, -0.5, 0.3, 1.0, 2.0] {
                    assert!((m.at(v * t) - l * t * t).abs() < 1e-9 * size * t * t, "{m:?} at {t}");
                }
            }
            let x = DVec2::new(0.3, -0.8);
            assert!((m.at(x * 2.0) - 4.0 * m.at(x)).abs() < 1e-12 * size);
        }
    }

    /// The Rayleigh quotient: round the unit circle the height is never above λ₁ nor
    /// below λ₂, and reaches them at the eigenvectors.
    #[test]
    fn round_the_ring_the_height_lies_between_the_eigenvalues() {
        for m in forms() {
            let s = m.spectrum();
            let size = m.size().max(1e-300);
            let heights: Vec<f64> = (0..3600).map(|i| m.at(unit(i as f64 / 3600.0 * std::f64::consts::TAU))).collect();
            let (lo, hi) = heights.iter().fold((f64::MAX, f64::MIN), |(lo, hi), h| (lo.min(*h), hi.max(*h)));
            assert!(hi <= s.l1 + 1e-9 * size && lo >= s.l2 - 1e-9 * size, "{m:?}: {lo}..{hi} against {s:?}");
            assert!(s.l1 - hi < 1e-5 * size && lo - s.l2 < 1e-5 * size, "{m:?}: {lo}..{hi} against {s:?}");
            if let Some(theta) = s.theta {
                assert!((m.at(unit(theta)) - s.l1).abs() < 1e-9 * size && (m.at(unit(theta + 0.5 * std::f64::consts::PI)) - s.l2).abs() < 1e-9 * size);
            }
        }
    }

    /// The level curves drawn are where the height is ±1, the ellipse's half-axes are
    /// 1/√λ along the eigenvectors, and a hyperbola never meets the axis of the other sign.
    #[test]
    fn the_level_curves_drawn_are_at_their_heights() {
        let mut ellipses = 0;
        for m in forms() {
            let s = m.spectrum();
            let theta = s.theta.unwrap_or(0.0);
            let (v1, v2) = (unit(theta), unit(theta + 0.5 * std::f64::consts::PI));
            for level in [1.0, -1.0] {
                let curves = level_curves(&s, theta, level, FLOOR_R, LEVEL_N);
                for c in &curves {
                    assert!(c.len() >= 2 && c.iter().all(|p| p.is_finite() && p.length() <= FLOOR_R + 1e-12), "{m:?}");
                    for p in c {
                        assert!((m.at(*p) - level).abs() < 1e-9, "{m:?}: {} at {p} on the curve {level}", m.at(*p));
                    }
                }
                // a curve exactly when some direction reaches the level within the floor
                let reaches = (0..720).any(|i| {
                    let d = unit(i as f64 / 720.0 * std::f64::consts::TAU);
                    let q = m.at(d);
                    q * level > 0.0 && (level / q).sqrt() < FLOOR_R * 0.99
                });
                assert_eq!(reaches, !curves.is_empty(), "{m:?} level {level}");
                if s.l2 > 0.0 && level > 0.0 && 1.0 / s.l2.sqrt() < FLOOR_R && s.theta.is_some() {
                    // a closed ellipse: furthest out 1/√λ₂ along v2, nearest 1/√λ₁ along v1
                    ellipses += 1;
                    assert_eq!(curves.len(), 1, "{m:?}");
                    let c = &curves[0];
                    assert!(c.len() == LEVEL_N && (c[0] - c[LEVEL_N - 1]).length() < 1e-9, "{m:?}: not closed");
                    // it passes through ±v₁/√λ₁ and ±v₂/√λ₂ (within half a pixel as the
                    // scene opens, some 250 to a unit), and no point is nearer 0 than the
                    // first or further than the second
                    let passes = |p: DVec2| c.windows(2).map(|w| seg_dist(p.as_vec2(), w[0].as_vec2(), w[1].as_vec2())).fold(f32::MAX, f32::min) < 2e-3;
                    for side in [1.0, -1.0] {
                        assert!(passes(v1 * side / s.l1.sqrt()) && passes(v2 * side / s.l2.sqrt()), "{m:?}");
                    }
                    assert!(c.iter().all(|p| p.length() >= 1.0 / s.l1.sqrt() - 1e-9 && p.length() <= 1.0 / s.l2.sqrt() + 1e-9), "{m:?}");
                    // evenly spaced along it
                    let steps: Vec<f64> = c.windows(2).map(|w| (w[1] - w[0]).length()).collect();
                    let (lo, hi) = steps.iter().fold((f64::MAX, 0.0f64), |(lo, hi), d| (lo.min(*d), hi.max(*d)));
                    assert!(hi < 1.02 * lo, "{m:?}: steps from {lo} to {hi}");
                }
                if s.l1 > 0.0 && s.l2 < 0.0 {
                    // the hyperbola xᵀAx = 1 keeps clear of λ₂'s axis, = −1 of λ₁'s
                    let (keep, of) = if level > 0.0 { (1.0 / s.l1.sqrt(), v1) } else { (1.0 / (-s.l2).sqrt(), v2) };
                    for p in curves.iter().flatten() {
                        assert!(p.dot(of).abs() >= keep - 1e-9, "{m:?}: {p} comes within {} of the axis", p.dot(of).abs());
                    }
                }
            }
        }
        assert!(ellipses > 25, "only {ellipses} ellipses were looked at");
    }

    /// A level curve that changes a little is drawn a little changed, point for point:
    /// as λ₂ goes from a little above 0 through 0 to a little below, the ellipse cut by
    /// the rim, the trough's two lines and the saddle's hyperbola are drawn alike.
    #[test]
    fn a_level_curve_changes_point_for_point() {
        let theta = 0.4636;
        let at = |l2: f64| level_curves(&Form::of(1.5, l2, theta).spectrum(), theta, 1.0, FLOOR_R, LEVEL_N);
        let mut before = at(0.02);
        for l2 in [0.01, 0.001, 1e-5, 0.0, -1e-5, -0.001, -0.01] {
            let now = at(l2);
            assert!(now.len() == 2 && before.len() == 2, "{l2}: {} pieces", now.len());
            let most = now.iter().zip(&before).flat_map(|(a, b)| a.iter().zip(b).map(|(p, q)| (*p - *q).length())).fold(0.0, f64::max);
            assert!(most < 0.05, "{l2}: a point moved {most}");
            before = now;
        }
    }

    /// Turning the axes onto the eigenvectors takes the mixed term away: QᵀAQ is Λ, the
    /// 0s off its diagonal are 0, not rounding, and the form written in y has no y₁y₂.
    #[test]
    fn turning_the_axes_takes_the_mixed_term_away() {
        for m in forms() {
            let s = m.spectrum();
            let size = m.size().max(1e-300);
            let t = m.turned(s.theta.unwrap_or(0.0));
            assert!(t.b == 0.0 && (t.a - s.l1).abs() < 1e-9 * size && (t.c - s.l2).abs() < 1e-9 * size, "{m:?}: {t:?} against {s:?}");
            // half way there, the mixed term is still there (unless there is none at all)
            if s.theta.is_some_and(|theta| theta.abs() > 0.1) {
                assert!(m.turned(0.5 * s.theta.unwrap()).b != 0.0);
            }
            // turning keeps the heights: the same point, in the turned coordinates
            let alpha = 0.7;
            let x = DVec2::new(0.4, -1.1);
            let y = DVec2::new(x.dot(unit(alpha)), x.dot(unit(alpha + 0.5 * std::f64::consts::PI)));
            assert!((m.turned(alpha).at(y) - m.at(x)).abs() < 1e-12 * size);
        }
        assert_eq!(written(&Form { a: 2.0, b: 0.0, c: 0.5 }, ["y_1", "y_2"], false), "2.00 y_1^2 + 0.50 y_2^2");
        assert_eq!(written(&Form { a: -0.6, b: -0.8, c: 0.0 }, ["x_1", "x_2"], true), "−0.60 x_1^2 − 1.60 x_1 x_2 + 0.00 x_2^2");
        assert_eq!(written(&Form { a: 0.003, b: -0.0001, c: 1.0 }, ["x_1", "x_2"], true), "0.003 x_1^2 − 0.0002 x_1 x_2 + 1.00 x_2^2");
    }

    /// The captions of the "things to try", held to account, each as its button sets it up.
    #[test]
    fn the_experiments_show_what_their_captions_say() {
        // round the circle: x goes once round, the ring up and down twice, the top
        // 2.00 = λ₁ where x crosses λ₁'s axis and the bottom 0.50 = λ₂ a right angle on
        let mut scene = QuadForm::create();
        scene.set_up(0);
        let (mut angles, mut heights) = (Vec::new(), Vec::new());
        for _ in 0..3000 {
            scene.advance(1.0 / 60.0);
            if scene.morph.is_none() && scene.sweep.is_some() {
                angles.push(scene.phi);
                heights.push(scene.form.at(scene.x()));
            }
        }
        assert!(!scene.moving());
        let s = scene.spectrum();
        let theta = s.theta.unwrap();
        assert!(val(s.l1) == "2.00" && val(s.l2) == "0.50" && kind(&s) == Kind::Bowl, "{s:?}");
        let travelled = angles.last().unwrap() - angles[0];
        assert!((travelled - std::f64::consts::TAU).abs() < 0.01 && angles.windows(2).all(|w| w[1] >= w[0]), "{travelled}");
        let (mut tops, mut bottoms) = (Vec::new(), Vec::new());
        for i in 1..heights.len() - 1 {
            if heights[i] >= heights[i - 1] && heights[i] > heights[i + 1] {
                tops.push((angles[i], heights[i]));
            }
            if heights[i] <= heights[i - 1] && heights[i] < heights[i + 1] {
                bottoms.push((angles[i], heights[i]));
            }
        }
        assert!(tops.len() == 2 && bottoms.len() == 2, "{tops:?} {bottoms:?}");
        let off = |t: f64, axis: f64| axis_angle(t - axis).abs();
        for (t, h) in &tops {
            assert!(val(*h) == "2.00" && off(*t, theta) < 0.01, "a top {h} at {}°", t.to_degrees());
        }
        for (t, h) in &bottoms {
            assert!(val(*h) == "0.50" && off(*t, theta + 0.5 * std::f64::consts::PI) < 0.01, "a bottom {h} at {}°", t.to_degrees());
        }

        // a saddle: 1 and −1; the ring below 0 on two arcs; 0 along two lines halfway
        // between the axes; hyperbolas (checked in general above)
        let scene = trial(1);
        let s = scene.spectrum();
        assert!(s.l1 == 1.0 && s.l2 == -1.0 && kind(&s) == Kind::Saddle && val(s.l2) == "−1.00", "{s:?}");
        let theta = s.theta.unwrap();
        let below: Vec<bool> = (0..720).map(|i| scene.form.at(unit(i as f64 / 720.0 * std::f64::consts::TAU)) < 0.0).collect();
        let arcs = (0..720).filter(|&i| below[i] && !below[(i + 719) % 720]).count();
        assert_eq!(arcs, 2);
        for side in [1.0, -1.0] {
            let d = unit(theta + side * 0.25 * std::f64::consts::PI);
            assert!(scene.form.at(d).abs() < 1e-12 && scene.form.at(d * 1.5).abs() < 1e-12);
        }
        assert!(level_curves(&s, theta, 1.0, FLOOR_R, LEVEL_N).len() == 2 && level_curves(&s, theta, -1.0, FLOOR_R, LEVEL_N).len() == 2);

        // a trough: λ₂ = 0, flat along its axis, never below 0, det 0, and the level
        // curve two straight lines 0.82 either side
        let scene = trial(2);
        let s = scene.spectrum();
        let (v1, v2) = (unit(s.theta.unwrap()), unit(s.theta.unwrap() + 0.5 * std::f64::consts::PI));
        assert!(s.l2 == 0.0 && val(s.l1) == "1.50" && kind(&s) == Kind::Trough && scene.form.det().abs() < 1e-15, "{s:?}");
        for t in [-1.5, -0.4, 0.7, 1.6] {
            assert!(scene.form.at(v2 * t).abs() < 1e-12);
        }
        assert!((0..720).all(|i| scene.form.at(unit(i as f64 / 720.0 * std::f64::consts::TAU)) >= -1e-15));
        let lines = level_curves(&s, s.theta.unwrap(), 1.0, FLOOR_R, LEVEL_N);
        assert_eq!(lines.len(), 2);
        for line in &lines {
            let across: Vec<f64> = line.iter().map(|p| p.dot(v1)).collect();
            assert!(across.iter().all(|a| (a.abs() - 1.0 / 1.5f64.sqrt()).abs() < 1e-9) && num(across[0].abs(), 2) == "0.82", "{across:?}");
            // straight: along the purple axis
            assert!((line[line.len() - 1] - line[0]).normalize().dot(v2).abs() > 1.0 - 1e-12);
        }

        // a narrow valley: 3 and 0.1, slopes thirty times apart at one distance; steps
        // nine tenths of the size at which they grow; bouncing; 0.13 from the bottom
        // after 30 steps, where the opening bowl's are within 0.001
        let scene = trial(3);
        let s = scene.spectrum();
        let (v1, v2) = (unit(s.theta.unwrap()), unit(s.theta.unwrap() + 0.5 * std::f64::consts::PI));
        assert!(val(s.l1) == "3.00" && val(s.l2) == "0.10" && scene.descent, "{s:?}");
        let slope = |v: DVec2| (scene.form.times(v) * 2.0).length();
        assert!((slope(v1 * 0.6) / slope(v2 * 0.6) - 30.0).abs() < 1e-6);
        let path = descent(&scene.form, scene.x(), DESCENT_STEPS, FLOOR_R);
        assert_eq!(path.len(), DESCENT_STEPS + 1);
        let across: Vec<f64> = path.iter().map(|p| p.dot(v1)).collect();
        assert!(across.windows(2).all(|w| w[0] * w[1] < 0.0), "it does not bounce: {across:?}");
        for w in across.windows(2) {
            assert!((w[1] / w[0] + 0.8).abs() < 1e-9, "the part across the valley is not multiplied by −0.8");
        }
        let along: Vec<f64> = path.iter().map(|p| p.dot(v2)).collect();
        assert!(along.windows(2).all(|w| w[1] < w[0] && w[1] > 0.9 * w[0]), "it does not creep: {along:?}");
        let last = path[DESCENT_STEPS].length();
        assert_eq!(num(last, 2), "0.13");
        let opening = Form { a: TRIALS[OPENING].form[0], b: TRIALS[OPENING].form[1], c: TRIALS[OPENING].form[2] };
        let bowl = descent(&opening, scene.x(), DESCENT_STEPS, FLOOR_R);
        let os = opening.spectrum();
        assert!(bowl[DESCENT_STEPS].length() < 0.001 && bowl[DESCENT_STEPS].length() > 0.0005 && val(os.l1 / os.l2) == "4.00", "{}", bowl[DESCENT_STEPS].length());
        // (the step: 1 − 2hλ₁ = −0.8 is nine tenths of the way from 1 to −1)
        assert_eq!(scale_words(scale_for(&s)).as_deref(), Some("heights drawn at half their size"));

        // turn the axes: by 26.6°, onto the eigenvectors; the form without its mixed
        // term, and QᵀAQ diagonal with the eigenvalues
        let scene = trial(4);
        assert!(scene.turned && scene.turn == 1.0);
        assert_eq!(num(scene.alpha().to_degrees(), 1), "26.6");
        assert_eq!(scene.alpha(), scene.spectrum().theta.unwrap());
        let t = scene.form.turned(scene.alpha());
        assert_eq!(written(&scene.form, ["x_1", "x_2"], true), "1.70 x_1^2 + 1.20 x_1 x_2 + 0.80 x_2^2");
        assert!(t.b == 0.0 && val(t.a) == "2.00" && val(t.c) == "0.50", "{t:?}");
        assert_eq!(written(&t, ["y_1", "y_2"], t.b != 0.0), "2.00 y_1^2 + 0.50 y_2^2");

        // what each says it is
        let kinds: Vec<Kind> = (0..TRIALS.len()).map(|k| kind(&trial(k).spectrum())).collect();
        assert_eq!(kinds, [Kind::Bowl, Kind::Saddle, Kind::Trough, Kind::Bowl, Kind::Bowl]);
    }

    /// A change of matrix is drawn moving evenly: no frame jumps, the eigenvalues go
    /// straight from where they were to where they end (a bowl to a saddle passes through
    /// a trough), and it ends exactly at the experiment's matrix.
    #[test]
    fn a_change_of_matrix_moves_evenly() {
        for hz in [60.0f32, 144.0] {
            let mut scene = QuadForm::create();
            scene.set_up(1);
            let mut last = (scene.form, scene.phi);
            let (mut most, mut frames, mut troughs) = (0.0f64, 0, 0);
            while scene.morph.is_some() {
                scene.advance(1.0 / hz);
                let step = (scene.form.a - last.0.a).abs().max((scene.form.b - last.0.b).abs()).max((scene.form.c - last.0.c).abs()).max((scene.phi - last.1).abs());
                most = most.max(step);
                let s = scene.spectrum();
                if s.l2.abs() < 0.05 {
                    troughs += 1;
                }
                last = (scene.form, scene.phi);
                frames += 1;
            }
            assert!(frames as f32 >= MORPH_TIME * hz - 1.0 && most < 6.0 / (MORPH_TIME * hz) as f64, "{frames} frames, the largest step {most}");
            assert!(troughs > 0, "it never passed through a trough");
            assert_eq!(scene.form, Form { a: 0.6, b: 0.8, c: -0.6 });
        }
    }

    /// Degenerate matrices: 0, a round bowl, a ridge, very large and very small ones, a
    /// typed-in extreme. Every number drawn or read out is finite, and the picture says
    /// what has happened.
    #[test]
    fn degenerate_matrices_leave_everything_finite() {
        let cases = [
            (Form { a: 0.0, b: 0.0, c: 0.0 }, Kind::Flat),
            (Form { a: 1.0, b: 0.0, c: 1.0 }, Kind::Bowl),
            (Form { a: -0.5, b: 0.0, c: -0.5 }, Kind::Dome),
            (Form { a: -1.2, b: -0.6, c: -0.3 }, Kind::Ridge),
            (Form { a: 1.2, b: 0.6, c: 0.3 }, Kind::Trough),
            (Form { a: 4.0, b: 4.0, c: 4.0 }, Kind::Trough),
            (Form { a: 100.0, b: -100.0, c: 100.0 }, Kind::Trough),
            (Form { a: 1e-7, b: 0.0, c: 2e-7 }, Kind::Bowl),
            (Form { a: 0.003, b: 0.0, c: -0.002 }, Kind::Saddle),
        ];
        for (m, want) in cases {
            let s = m.spectrum();
            assert_eq!(kind(&s), want, "{m:?}: {s:?}");
            let (word, why) = verdict(&s);
            assert!(!word.is_empty() && !why.is_empty());
            if s.theta.is_none() {
                assert!(why.contains("every direction is an eigenvector"), "{m:?}: {why}");
            }
            let mut scene = QuadForm::create();
            scene.set_form(m);
            for _ in 0..200 {
                scene.advance(1.0 / 60.0);
            }
            let scale = scale_for(&s);
            assert!(scale > 0.0 && scale <= 1.0 && s.size() * scale * SURF_R * SURF_R <= TALL + 1e-12, "{m:?}: {scale}");
            let (low, high) = QuadForm::extent(&m, scale as f32);
            assert!(low.is_finite() && high.is_finite() && low < high);
            for level in [1.0, -1.0] {
                assert!(level_curves(&s, scene.angle(), level, FLOOR_R, LEVEL_N).iter().flatten().all(|p| p.is_finite()));
            }
            assert!(descent(&m, scene.x(), DESCENT_STEPS, FLOOR_R).iter().all(|p| p.is_finite()));
            for text in [val(s.l1), val(s.l2), val(m.det()), written(&m, ["x_1", "x_2"], true), written(&m.turned(scene.alpha()), ["y_1", "y_2"], true)] {
                assert!(!text.contains("NaN") && !text.contains("inf") && !text.contains('?'), "{m:?}: {text}");
            }
            assert!(matches!(scene.the_view(), View::Orbit { target, dist, .. } if target.is_finite() && dist.is_finite() && dist > 0.0), "{m:?}");
        }
        // large numbers are drawn smaller, and the picture says so
        let big = Form { a: 4.0, b: 4.0, c: 4.0 }.spectrum();
        assert!(big.l1 == 8.0 && scale_for(&big) == 0.25 && scale_words(0.25).as_deref() == Some("heights drawn at a quarter of their size"));
        assert_eq!(scale_words(1.0), None);
        assert_eq!(scale_words(1.0 / 16.0).as_deref(), Some("heights drawn at 1/16 of their size"));
        // a number that is not 0 is never written as 0
        assert_eq!((val(0.003), val(-0.0004), val(0.0), val(-1.5), val(2.0)), ("0.003".into(), "−0.0004".into(), "0.00".into(), "−1.50".into(), "2.00".into()));
        // a round bowl keeps its axes where they were, and so does the angle slider
        let mut scene = QuadForm::create();
        let before = scene.angle();
        scene.set_form(Form::of(1.0, 1.0, 1.0));
        assert!(scene.spectrum().theta.is_none() && scene.angle() == before);
    }

    /// The two ways in agree: the eigenvalue sliders give the matrix with those
    /// eigenvalues and that angle, and reading them back gives what was set.
    #[test]
    fn the_sliders_keep_the_two_descriptions_in_step() {
        let mut rng = Rng::new(5);
        for _ in 0..300 {
            let l1 = rng.range(-4.0, 4.0) as f64;
            let l2 = rng.range(-4.0, 4.0) as f64;
            let (l1, l2) = (l1.max(l2), l1.min(l2));
            let theta = rng.range(-89.0, 90.0) as f64;
            let mut scene = QuadForm::create();
            scene.set_form(Form::of(l1, l2, theta.to_radians()));
            let s = scene.spectrum();
            assert!((s.l1 - l1).abs() < 1e-9 && (s.l2 - l2).abs() < 1e-9, "{l1} {l2}: {s:?}");
            if l1 - l2 > 1e-6 {
                assert!((scene.angle().to_degrees() - theta).abs() < 1e-6, "{theta}: {}", scene.angle().to_degrees());
            }
        }
    }

    /// The level curves are kept from frame to frame and made again whenever what they
    /// hang on changes. Whatever was done (the matrix's sliders, the eigenvalues' and the
    /// angle's, the angle turned with a round bowl, which every angle fits, an experiment
    /// playing its way in frame by frame), the curves drawn are the ones made afresh; and
    /// while only x moves they are not made again.
    #[test]
    fn the_level_curves_follow_the_matrix() {
        // (as `picture` takes the curves and puts them back, against curves made
        // afresh; true if they were made afresh)
        fn draw(scene: &mut QuadForm) -> bool {
            let before = scene.levels.as_ref().map(|l| l.made_from);
            let levels = scene.levels_now();
            let (s, angle) = (scene.spectrum(), scene.angle());
            let fresh = [1.0, -1.0].map(|level| (level_curves(&s, angle, level, FLOOR_R, LEVEL_N), level_curves(&s, angle, level, SURF_R, LEVEL_TOP_N)));
            assert!(levels.curves == fresh, "{:?} at {}", scene.form, angle);
            let made = before != Some(levels.made_from);
            scene.levels = Some(levels);
            made
        }
        let mut scene = QuadForm::create();
        let mut made = usize::from(draw(&mut scene));
        // the matrix's sliders
        for m in forms() {
            scene.set_form(m);
            made += usize::from(draw(&mut scene));
        }
        // the eigenvalues' and the angle's sliders, a round bowl among them
        for (l1, l2, degrees) in [(2.0, 0.5, 30.0), (2.0, 0.5, -60.0), (1.0, 1.0, 10.0), (1.0, 1.0, 75.0), (0.8, -1.2, 75.0), (1.5, 0.0, 75.0), (-0.5, -0.5, -20.0)] {
            scene.set_form(Form::of(l1, l2, f64::to_radians(degrees)));
            scene.axis = f64::to_radians(degrees);
            assert!(draw(&mut scene), "{l1} {l2} {degrees}: not made afresh");
        }
        // every experiment, from every other, frame by frame
        for before in 0..TRIALS.len() {
            for k in 0..TRIALS.len() {
                let mut scene = trial(before);
                draw(&mut scene);
                scene.set_up(k);
                for _ in 0..240 {
                    scene.advance(1.0 / 60.0);
                    draw(&mut scene);
                }
            }
        }
        // x dragged round: the curves stay as they were made
        let once = made;
        for i in 0..100 {
            scene.phi = i as f64 * 0.1;
            made += usize::from(draw(&mut scene));
        }
        assert_eq!(made, once, "made again while only x moved");
    }

    /// The card's height counts the verdicts' lines without making them, and counts as
    /// many as `wrap` makes.
    #[test]
    fn lines_are_counted_as_they_are_made() {
        let mut texts: Vec<&str> = vec!["", "one", "two  spaces", " leading", "trailing ", "x"];
        texts.extend(TRIALS.iter().map(|t| t.look_for));
        for (l1, l2, round) in [(1.0, 1.0, true), (2.0, 1.0, false), (-1.0, -1.0, true), (-1.0, -2.0, false), (1.0, -1.0, false), (1.0, 0.0, false), (0.0, -1.0, false), (0.0, 0.0, true)] {
            texts.push(verdict(&Spectrum { l1, l2, theta: (!round).then_some(0.0) }).1);
        }
        for text in texts {
            for size in [11.5, 12.0, 13.5] {
                for width in (0..120).map(|i| i as f32 * 5.3) {
                    assert_eq!(line_count(text, size, width), wrap(text, size, width).len(), "{text:?} at {width}");
                }
            }
        }
    }

    /// The card and the graph stay on the canvas and off each other and the picture's
    /// part, whatever its shape, and the view fitted to it is a view.
    #[test]
    fn the_panels_fit_every_canvas() {
        for (x, y) in [(1078.0f32, 1000.0f32), (1100.0, 700.0), (578.0, 700.0), (390.0, 800.0), (360.0, 640.0), (844.0, 390.0), (640.0, 480.0), (240.0, 180.0), (2000.0, 1300.0), (8.0, 8.0)] {
            let view = Vec2::new(x, y);
            let inset = if x < 900.0 { 46.0 } else { 0.0 };
            let lay = Layout::of(view, inset);
            let boxes: Vec<(Vec2, Vec2)> = [lay.card, lay.graph].into_iter().flatten().collect();
            for (min, max) in &boxes {
                assert!(min.x >= 0.0 && min.y >= inset.min(y) && max.x <= x && max.y <= y && max.x > min.x, "{x}x{y}: {min} {max}");
            }
            if let [a, b] = boxes[..] {
                assert!(a.1.x <= b.0.x || b.1.x <= a.0.x || a.1.y <= b.0.y || b.1.y <= a.0.y, "{x}x{y}: the card and the graph overlap");
            }
            let (min, max) = lay.picture;
            assert!(min.is_finite() && max.is_finite() && max.x > min.x && max.y > min.y, "{x}x{y}: {min} {max}");
            let mut scene = QuadForm::create();
            (scene.canvas, scene.inset) = (view, inset);
            for k in 0..TRIALS.len() {
                scene.set_up(k);
                assert!(matches!(scene.the_view(), View::Orbit { target, dist, .. } if target.is_finite() && dist.is_finite() && dist > 0.0), "{x}x{y}");
            }
        }
        // the desktop and the phone have both panels
        assert!(Layout::of(Vec2::new(1078.0, 1000.0), 0.0).graph.is_some() && Layout::of(Vec2::new(390.0, 800.0), 46.0).graph.is_some());
    }

    /// The numbers behind the captions (run with --ignored --nocapture).
    #[test]
    #[ignore]
    fn explore() {
        for k in 0..TRIALS.len() {
            let s = trial(k);
            let sp = s.spectrum();
            println!("{}: {:?} theta {:?} deg, kind {:?}", TRIALS[k].label, sp, sp.theta.map(f64::to_degrees), kind(&sp));
            println!("   written {}", written(&s.form, ["x_1", "x_2"], true));
            let t = s.form.turned(s.alpha());
            println!("   turned {:?} alpha {} -> {}", t, s.alpha().to_degrees(), written(&t, ["y_1", "y_2"], t.b != 0.0));
            for (lev, name) in [(1.0, "+1"), (-1.0, "-1")] {
                let c = level_curves(&sp, s.angle(), lev, FLOOR_R, LEVEL_N);
                println!("   level {name}: {} pieces", c.len());
            }
            let x = s.x();
            for steps in [5, 10, 20, 30, 40, 60, 100] {
                let path = descent(&s.form, x, steps, 100.0);
                let end = *path.last().unwrap();
                println!("   descent {steps} steps from phi {:.1}: end {:?} dist {:.4} height {:.4}", s.phi.to_degrees(), end, end.length(), s.form.at(end));
            }
        }
        // the same rule in the opening bowl, from the valley's start
        let bowl = Form { a: 1.7, b: 0.6, c: 0.8 };
        let x = unit(85f64.to_radians());
        for steps in [5, 10, 20, 30] {
            let path = descent(&bowl, x, steps, 100.0);
            let end = *path.last().unwrap();
            println!("bowl descent {steps}: dist {:.6}", end.length());
        }
        println!("1/sqrt(1.5) = {}", 1.0 / 1.5f64.sqrt());
    }
}

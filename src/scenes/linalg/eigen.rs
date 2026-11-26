//! Finding eigenvalues: the directions a matrix does not turn, and the numbers λ for
//! which A − λI squashes the plane flat.

use super::*;
use glam::{DMat2, DVec2};
use std::f64::consts::{PI as PI64, TAU as TAU64};

pub const DEF: SceneDef = SceneDef {
    id: "eigen",
    title: "Finding eigenvalues",
    group: "Linear algebra: eigenvalues",
    blurb: r"Sweep a vector round the circle to see where $Ax$ lines up with $x$, and slide $\lambda$ until $A - \lambda I$ collapses the plane.",
    make: |_| Box::new(Eigen::create()),
};

// ----- the matrices ------------------------------------------------------------------

/// A matrix as it is written, row by row: [a, b, c, d] is [[a, b], [c, d]].
fn rows([a, b, c, d]: [f32; 4]) -> Mat2 {
    Mat2::from_cols(Vec2::new(a, c), Vec2::new(b, d))
}

/// The same in double precision, where the arithmetic is done.
fn wide(m: Mat2) -> DMat2 {
    DMat2::from_cols(m.x_axis.as_dvec2(), m.y_axis.as_dvec2())
}

const START: [f32; 4] = [1.0, 2.0, 1.0, 0.0];
const SYMMETRIC: [f32; 4] = [1.0, 2.0, 2.0, -2.0];
/// A turn by 60°: cos 60° = 0.5, sin 60° = √3/2.
const TURN60: [f32; 4] = [0.5, -0.866_025_4, 0.866_025_4, 0.5];
const SHEAR: [f32; 4] = [1.0, 1.0, 0.0, 1.0];

/// Quick choices for A: (button, the matrix by rows).
const PRESETS: [(&str, [f32; 4]); 8] = [
    ("start", START),
    ("symmetric", SYMMETRIC),
    ("turn", TURN60),
    ("shear", SHEAR),
    ("$2I$", [2.0, 0.0, 0.0, 2.0]),
    ("$I$", [1.0, 0.0, 0.0, 1.0]),
    ("singular", [1.0, 2.0, 0.5, 1.0]),
    ("$0$", [0.0; 4]),
];

/// What an experiment sets: the matrix (by rows), where x is (degrees round from the
/// x axis), λ, whether x sweeps round, and how λ gets there: in `lambda_secs` seconds
/// from where it is, or, if `lambda_via` is given, from there (where it goes first).
#[derive(Clone, Copy, Debug)]
struct Setup {
    a: [f32; 4],
    x_deg: f64,
    lambda: f64,
    sweep: bool,
    lambda_via: Option<f64>,
    lambda_secs: f32,
}

/// The direction of the first kept line of `START`, (2, 1): where the second
/// experiment puts x.
const ON_V1: f64 = 26.565_051_177_077_99;

/// Experiments: (button, what to look for, what it sets).
const TRIALS: [(&str, &str, Setup); 5] = [
    (
        r"sweep $x$ round",
        r"$x$ goes twice round the unit circle, and $A$ turns it, mostly. Four times a turn it stops: where $x$ lies on a dashed line, $Ax$ lies on it too. On the orange line $Ax$ is twice $x$ ($\lambda_1 = 2$); on the purple one it is $x$ reversed ($\lambda_2 = -1$), a turn of 180°. The chart under the plane reaches 0° and ±180° at just those places. After its two turns $x$ comes to rest at the next stop, on the purple line.",
        Setup { a: START, x_deg: 60.0, lambda: 0.5, sweep: true, lambda_via: None, lambda_secs: MORPH },
    ),
    (
        r"slide $\lambda$ to a root",
        r"$\lambda$ slides from 3 to 2. The dot runs down the parabola: $\det(A - \lambda I)$, the area of the yellow square, falls from 4 to 0. The square gets thinner until, at $\lambda = 2$, the whole plane is folded flat onto a line. The orange arrow, where $A - \lambda I$ sends $v_1$ (the orange ring), shrinks to the point $0$: $(A - 2I)v_1 = Av_1 - 2v_1 = 0$. So 2 is an eigenvalue, and $v_1$, on the orange line where $x$ is now, an eigenvector.",
        Setup { a: START, x_deg: ON_V1, lambda: 2.0, sweep: false, lambda_via: Some(3.0), lambda_secs: SLIDE },
    ),
    (
        "a turn",
        r"$A$ turns every arrow by 60° and keeps its length. The chart is flat at 60°: it never reaches 0° or ±180°, and the sweep never pauses: it goes twice round and comes to rest where it set off. No direction is kept. The parabola $\lambda^2 - \lambda + 1$ stays above 0: at its lowest, at $\lambda = 0.5$, it is 0.75, so $A - \lambda I$ never flattens the plane. Its roots are not real numbers but $0.5 \pm 0.87i$.",
        Setup { a: TURN60, x_deg: 0.0, lambda: 0.5, sweep: true, lambda_via: None, lambda_secs: MORPH },
    ),
    (
        "a shear",
        r"The parabola, $\lambda^2 - 2\lambda + 1 = (\lambda - 1)^2$, only touches 0, at $\lambda = 1$: a double root. $A - I$ folds the plane onto the $x$ axis and crushes the $x$ axis itself to $0$. That is the only direction kept: the sweep stops twice a turn, not four times, and the chart only touches 0°. The identity $I$ has the same parabola and keeps every direction (preset $I$): a double root does not say how many directions are kept.",
        Setup { a: SHEAR, x_deg: 60.0, lambda: 1.0, sweep: true, lambda_via: None, lambda_secs: MORPH },
    ),
    (
        "a symmetric matrix",
        r"The entry above the diagonal equals the one below: $A$ is symmetric. Its eigenvalues are real, $2$ and $-3$, and its dashed lines cross at a right angle. They are the long and short axes of the ellipse $Ax$ traces: $Ax$ is longest (3) on the purple line and shortest (2) on the orange one. Every symmetric matrix has real eigenvalues and eigenvectors at right angles.",
        Setup { a: SYMMETRIC, x_deg: 100.0, lambda: 0.0, sweep: true, lambda_via: None, lambda_secs: MORPH },
    ),
];

/// Seconds a change of setting takes to play (a new matrix, x or λ put elsewhere).
const MORPH: f32 = 1.0;
/// Seconds λ takes to slide to its root in the second experiment.
const SLIDE: f32 = 3.0;

/// The sweep: radians a second at full speed, how fast it gathers and loses speed
/// (radians a second, each second), and how long it rests on a kept direction.
const SWEEP_SPEED: f64 = TAU64 / 6.0;
const SWEEP_BRAKE: f64 = 2.0;
const REST: f32 = 1.2;

/// How long a glide that answers the hand takes: the time in which it covers two
/// thirds of the way (seconds), and what counts as there.
const GLIDE: f32 = 0.045;
const ARRIVED: f64 = 1e-6;
/// How long the frames of the three parts take to fit a new matrix.
const REFIT: f32 = 0.18;

/// How near x must come to a kept direction to click onto it (radians: about 3°), and
/// λ to a root (logical pixels along the graph).
const SNAP_ANGLE: f64 = 0.055;
const SNAP_PX: f32 = 9.0;

/// The colours of the two eigenvalues and their lines: the larger, and the smaller.
const EIG: [Color; 2] = [pal::ORANGE, pal::PURPLE];
/// The turn from x to Ax.
const TURN_COL: Color = pal::GREEN;
/// Ax.
const AX_COL: Color = pal::CYAN;

/// The size of a matrix's entries: what counts as zero is judged against it.
fn magnitude(m: DMat2) -> f64 {
    m.to_cols_array().iter().fold(0.0f64, |s, v| s.max(v.abs()))
}

fn trace(m: DMat2) -> f64 {
    m.x_axis.x + m.y_axis.y
}

/// What the roots of det(A − λI) = λ² − (trace A) λ + det A are, and with them the
/// directions A keeps (each a unit vector, pointing right or straight up).
#[derive(Clone, Copy, Debug, PartialEq)]
enum Kept {
    /// Two different real roots, the larger first, each with its line.
    Two([(f64, DVec2); 2]),
    /// One root, twice, and only one line.
    One(f64, DVec2),
    /// One root, twice, and A = λI: every direction is kept.
    Every(f64),
    /// No real root: re ± i im, and no direction kept.
    Turned { re: f64, im: f64 },
}

impl Kept {
    /// The kept lines (none for `Every`, which keeps them all), with their eigenvalues.
    fn lines(&self) -> Vec<(f64, DVec2)> {
        match *self {
            Kept::Two(two) => two.to_vec(),
            Kept::One(l, v) => vec![(l, v)],
            _ => Vec::new(),
        }
    }

    /// The real roots, the larger first (a double one once).
    fn roots(&self) -> Vec<f64> {
        match *self {
            Kept::Two([(a, _), (b, _)]) => vec![a, b],
            Kept::One(l, _) | Kept::Every(l) => vec![l],
            Kept::Turned { .. } => Vec::new(),
        }
    }
}

/// The direction a matrix of rank 1 or less sends to 0: across its longer row.
fn crushed(r: DMat2) -> DVec2 {
    let (r1, r2) = (DVec2::new(r.x_axis.x, r.y_axis.x), DVec2::new(r.x_axis.y, r.y_axis.y));
    let row = if r1.length_squared() >= r2.length_squared() { r1 } else { r2 };
    upright(row.perp().try_normalize().unwrap_or(DVec2::X))
}

/// A direction as the unit vector that points right (or straight up): a line has two.
fn upright(v: DVec2) -> DVec2 {
    if v.x < -1e-12 || (v.x.abs() <= 1e-12 && v.y < 0.0) {
        -v
    } else {
        v
    }
}

fn kept(m: DMat2) -> Kept {
    let s = magnitude(m);
    if s == 0.0 || !s.is_finite() {
        return Kept::Every(0.0);
    }
    let (t, d) = (trace(m), m.determinant());
    let c = t * 0.5;
    let disc = c * c - d;
    // (the entries are what was typed, exactly; a double root typed as such comes out
    // exactly double, and anything closer than a millionth of the entries is one)
    if disc.abs() <= 1e-12 * s * s {
        let r = m - DMat2::IDENTITY * c;
        return if magnitude(r) <= 1e-9 * s { Kept::Every(c) } else { Kept::One(c, crushed(r)) };
    }
    if disc < 0.0 {
        return Kept::Turned { re: c, im: (-disc).sqrt() };
    }
    // (the root further from 0 first, then the other from the product: no cancelling)
    let q = c + if c >= 0.0 { 1.0 } else { -1.0 } * disc.sqrt();
    let (l1, l2) = if q >= d / q { (q, d / q) } else { (d / q, q) };
    let line = |l: f64| (l, crushed(m - DMat2::IDENTITY * l));
    Kept::Two([line(l1), line(l2)])
}

/// The unit arrow at angle θ.
fn unit(theta: f64) -> DVec2 {
    DVec2::new(theta.cos(), theta.sin())
}

/// An angle brought into (−π, π].
fn wrap(a: f64) -> f64 {
    let w = (a + PI64).rem_euclid(TAU64) - PI64;
    if w <= -PI64 {
        w + TAU64
    } else {
        w
    }
}

/// How far A turns the unit arrow at angle θ: the signed angle from x to Ax, in
/// (−π, π]. None where Ax = 0, which has no direction.
fn turn(m: DMat2, theta: f64) -> Option<f64> {
    let x = unit(theta);
    let ax = m * x;
    (ax.length() > 1e-9 * magnitude(m).max(1e-300)).then(|| x.perp_dot(ax).atan2(x.dot(ax))).map(wrap)
}

/// Where the sweep stops: the angles round the circle (in [0, 2π)) at which x lies on
/// a kept line, two to a line.
fn stops(k: &Kept) -> Vec<f64> {
    let mut out: Vec<f64> = k.lines().iter().flat_map(|(_, v)| {
        let a = v.y.atan2(v.x).rem_euclid(TAU64);
        [a, (a + PI64).rem_euclid(TAU64)]
    }).collect();
    out.sort_by(f64::total_cmp);
    out
}

/// The value of det(A − λI) = λ² − tλ + d.
fn poly(t: f64, d: f64, l: f64) -> f64 {
    l * l - t * l + d
}

// ----- numbers as written ------------------------------------------------------------

/// A number with up to two decimals and no trailing zeros: 2, −1.5, 0.87.
fn short(v: f64) -> String {
    let text = num(v, 2);
    if text.contains('.') {
        text.trim_end_matches('0').trim_end_matches('.').to_string()
    } else {
        text
    }
}

/// `short`, in brackets if it is negative (a factor in a product).
fn factor(v: f64) -> String {
    let t = short(v);
    if t.starts_with('−') {
        format!("({t})")
    } else {
        t
    }
}

/// The polynomial λ² − tλ + d as written (markup, to go between dollars): λ² − λ − 2,
/// λ² + 1; (λ − 1)² is not attempted.
fn poly_text(t: f64, d: f64) -> String {
    let mut text = r"\lambda^2".to_string();
    let b = -t;
    let bt = short(b.abs());
    if bt != "0" {
        text += if b < 0.0 { " - " } else { " + " };
        if bt != "1" {
            text += &bt;
        }
        text += r"\lambda";
    }
    let dt = short(d.abs());
    if dt != "0" {
        text += if d < 0.0 { " - " } else { " + " };
        text += &dt;
    }
    text
}

/// "(a − λ)(d − λ) − b·c" with the numbers of A (markup, to go between dollars).
fn product_text(m: DMat2) -> String {
    let (a, b, c, d) = (m.x_axis.x, m.y_axis.x, m.x_axis.y, m.y_axis.y);
    format!(r"({} - \lambda)({} - \lambda) - {} \cdot {}", short(a), short(d), factor(b), factor(c))
}

/// An eigenvalue's name (markup): λ₁, λ₂, or plain λ when there is one.
fn name(k: &Kept, i: usize) -> &'static str {
    match k {
        Kept::Two(_) => [r"\lambda_1", r"\lambda_2"][i.min(1)],
        _ => r"\lambda",
    }
}

/// A − λI with the number for λ (markup): A − 2I, A + 1.5I, A − I, and A itself for
/// λ = 0.
fn minus_lambda(l: f64) -> String {
    match short(l.abs()).as_str() {
        "0" => "A".to_string(),
        "1" => format!("A {} I", if l < 0.0 { "+" } else { "-" }),
        t => format!("A {} {t}I", if l < 0.0 { "+" } else { "-" }),
    }
}

/// Where A − λI sends the unit arrow `vi` (markup) of a kept line, `gap` = λᵢ − λ, as
/// a label: "v₁ ↦ −4v₁", "v₁ ↦ 0" (`scale` is the size of the entries, what counts as
/// 0 is judged by it).
fn sent(vi: &str, gap: f64, scale: f64) -> String {
    let image = match short(gap).as_str() {
        _ if gap.abs() <= 1e-9 * scale => "0".to_string(),
        "0" => format!("{}{vi}", num(gap, 3)),
        "1" => vi.to_string(),
        "−1" => format!("-{vi}"),
        g => format!("{g}{vi}"),
    };
    format!(r"${vi} \mapsto {image}$")
}

/// "Ax = 2x", "Ax = −x", "Ax = 0", as a label.
fn ax_is(l: f64) -> String {
    match short(l).as_str() {
        "0" => "$Ax = 0$".to_string(),
        "1" => "$Ax = x$".to_string(),
        "−1" => "$Ax = -x$".to_string(),
        t => format!("$Ax = {t}x$"),
    }
}

/// The smallest of 1, 1.25, 1.5, 2, 2.5, 3, 4, 5, 6, 8 times a power of ten that is at
/// least `v`: frames that change only in steps, so that a small change of the matrix
/// leaves them where they are.
fn nice_up(v: f64) -> f64 {
    let v = v.max(1e-6);
    let p = 10f64.powf(v.log10().floor());
    [1.0, 1.25, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0].iter().map(|k| k * p).find(|x| *x >= v * (1.0 - 1e-12)).unwrap_or(10.0 * p)
}

/// The window the graph of det(A − λI) shows, [λ from, λ to, value from, value to]:
/// both real roots well inside, the lowest point and the value 0 inside, and the arms
/// up to the top corners.
fn window(t: f64, d: f64) -> [f64; 4] {
    let c = t * 0.5;
    let disc = c * c - d;
    let s = disc.abs().sqrt();
    let w = nice_up((2.0 * s).max(1.5));
    let q = w * 0.25;
    let centre = (c / q).round() * q;
    let (lo, hi) = (centre - w, centre + w);
    let arms = poly(t, d, lo).min(poly(t, d, hi));
    let low = (-disc).min(0.0);
    let top = arms.max(low + 1.0);
    [lo, hi, low - 0.14 * (top - low), top]
}

/// How far the plane of x and Ax reaches from its middle to its edge.
fn reach(m: DMat2) -> f64 {
    let s = svd2(m.as_mat2()).s.x as f64;
    nice_up(1.12 * s.max(1.0))
}

// ----- the layout ----------------------------------------------------------------------

/// A square part of the picture: a plane, centred on its origin.
#[derive(Clone, Copy, Debug, PartialEq)]
struct Plane {
    centre: Vec2,
    half: f32,
    /// How far it reaches from its middle, in its own units.
    reach: f32,
}

impl Plane {
    fn per_unit(&self) -> f32 {
        self.half / self.reach.max(1e-6)
    }

    fn at(&self, p: DVec2) -> Vec2 {
        self.centre + p.as_vec2() * self.per_unit()
    }

    fn of(&self, w: Vec2) -> DVec2 {
        ((w - self.centre) / self.per_unit()).as_dvec2()
    }

    fn lo(&self) -> Vec2 {
        self.centre - Vec2::splat(self.half)
    }

    fn hi(&self) -> Vec2 {
        self.centre + Vec2::splat(self.half)
    }
}

/// A rectangular part: a chart or a graph, with the window of values it shows.
#[derive(Clone, Copy, Debug, PartialEq)]
struct Graph {
    min: Vec2,
    max: Vec2,
    /// [x from, x to, y from, y to].
    win: [f64; 4],
}

impl Graph {
    fn at(&self, x: f64, y: f64) -> Vec2 {
        let [x0, x1, y0, y1] = self.win;
        let u = ((x - x0) / (x1 - x0).max(1e-12)) as f32;
        let v = ((y - y0) / (y1 - y0).max(1e-12)) as f32;
        self.min + (self.max - self.min) * Vec2::new(u, v)
    }

    fn x_of(&self, w: f32) -> f64 {
        let [x0, x1, ..] = self.win;
        x0 + ((w - self.min.x) / (self.max.x - self.min.x).max(1e-9)) as f64 * (x1 - x0)
    }
}

/// Where the parts go, in the scene's units (the stage: five high, centred on the
/// origin), with the sizes of text that fit them.
#[derive(Clone, Copy, Debug, PartialEq)]
struct Layout {
    /// One column (a phone held upright) rather than two.
    narrow: bool,
    /// Logical pixels per unit at the view the scene opens with.
    ppu: f32,
    /// The plane of x and Ax: centre and half its side.
    a: (Vec2, f32),
    /// The chart of the turn, under it: its corners.
    chart: (Vec2, Vec2),
    /// The plane of A − λI.
    b: (Vec2, f32),
    /// The graph of det(A − λI).
    graph: (Vec2, Vec2),
    /// Text size.
    text: f32,
    /// Room for the parts' titles above them (logical pixels).
    title: f32,
    /// Whether there is room for words at all (not in a sliver of a window, such as
    /// the strip a phone's open controls leave beside them).
    words: bool,
}

impl Layout {
    fn of(view: Vec2, inset: f32) -> Layout {
        let view = view.max(Vec2::splat(1.0));
        // (as `STAGE_VIEW` is fitted to the canvas)
        let ppu = (view.y / STAGE_H).min(view.x / 2.0).max(1e-3);
        let world = |p: Vec2| Vec2::new((p.x - view.x * 0.5) / ppu, (view.y * 0.5 - p.y) / ppu);
        let narrow = view.x < 0.75 * view.y;
        let small = view.x < 600.0 || view.y < 600.0;
        let text = if small { 11.5 } else { 13.0 };
        let margin = if small { 10.0 } else { 16.0 };
        // a title row: a matrix written on the screen (two rows of numbers)
        let title = 2.0 * (text + 8.0) + 8.0;
        let ticks = text + 6.0;
        let top = inset.max(0.0) + margin;
        let bottom = view.y - margin;
        // (room at the chart's left for its scale)
        let lab = text_width("−180°", text - 2.0) + 10.0;
        let (a, chart, b, graph);
        if narrow {
            // one column: the plane, the chart under it, then the two small parts side by side
            let gap = 14.0;
            let w = (view.x - 2.0 * margin).max(20.0);
            let pair = ((w - gap) * 0.5).max(10.0);
            let chart_share = 0.2;
            let fixed = title + 8.0 + ticks * 0.6 + 10.0 + ticks + 12.0 + title + pair + 2.0 * ticks;
            let side = w.min((bottom - top - fixed) / (1.0 + chart_share)).max(10.0);
            let left = (view.x - side) * 0.5;
            let mut y = top + title;
            a = (Vec2::new(left, y), side);
            y += side + 10.0 + ticks * 0.6;
            let ch = (side * chart_share).max(10.0);
            chart = (Vec2::new(left.max(margin + lab).min(left + side * 0.5), y), Vec2::new(left + side, y + ch));
            y += ch + ticks + 12.0 + title;
            // the pair across the full width, under whatever the plane left
            let pair = pair.min((bottom - 2.0 * ticks - y).max(10.0));
            let lx = (view.x - 2.0 * pair - gap) * 0.5;
            b = (Vec2::new(lx, y), pair);
            graph = (Vec2::new(lx + pair + gap, y), Vec2::new(lx + 2.0 * pair + gap, y + pair));
        } else {
            // two columns: the plane over its chart; the plane of A − λI over the graph
            let gap = if small { 22.0 } else { 40.0 };
            let w = (view.x - 2.0 * margin - gap - lab).max(20.0);
            let room = (bottom - top).max(20.0);
            let (least_chart, most_chart, least_graph, most_graph) = (0.2, 0.34, 0.62, 0.9);
            let left_fixed = title + 10.0 + ticks + ticks;
            let right_fixed = title + 14.0 + 2.0 * (text + 6.0) + 8.0 + 2.0 * ticks;
            let la = (w * 0.58).min((room - left_fixed) / (1.0 + least_chart)).max(10.0);
            let rb = (w - la).min((room - right_fixed) / (1.0 + least_graph)).max(10.0);
            // what height is left goes to the chart and the graph, then evenly above and below
            let ch = (room - left_fixed - la).max(least_chart * la).min(most_chart * la);
            let gh = (room - right_fixed - rb).max(least_graph * rb).min(most_graph * rb);
            let down = ((room - (left_fixed + la + ch).max(right_fixed + rb + gh)) * 0.5).max(0.0);
            let left = margin + lab + (w - la - rb).max(0.0) * 0.5;
            let y = top + down + title;
            a = (Vec2::new(left, y), la);
            let cy = y + la + 10.0 + ticks;
            chart = (Vec2::new(left, cy), Vec2::new(left + la, cy + ch));
            let rx = left + la + gap;
            b = (Vec2::new(rx, y), rb);
            let gy = y + rb + 14.0 + 2.0 * (text + 6.0) + 8.0;
            graph = (Vec2::new(rx, gy), Vec2::new(rx + rb, gy + gh));
        }
        // (screen y runs down, the scene's y up)
        let square = |(min, side): (Vec2, f32)| {
            let c = world(min + Vec2::splat(side * 0.5));
            (c, side * 0.5 / ppu)
        };
        let rect = |(min, max): (Vec2, Vec2)| {
            let (p, q) = (world(min), world(max));
            (p.min(q), p.max(q))
        };
        let words = a.1 >= 150.0 && view.x >= 240.0;
        Layout { narrow, ppu, a: square(a), chart: rect(chart), b: square(b), graph: rect(graph), text, title, words }
    }
}

// ----- drawing helpers -------------------------------------------------------------------

/// The part of the segment a–b inside the box lo–hi, if any.
fn clip(a: Vec2, b: Vec2, lo: Vec2, hi: Vec2) -> Option<(Vec2, Vec2)> {
    let d = b - a;
    let (mut t0, mut t1) = (0.0f32, 1.0f32);
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
    (t0 <= t1).then(|| (a + d * t0, a + d * t1))
}

/// A straight line drawn as a thin filled strip, `width` logical pixels across.
///
/// Used for lines cut off at the edge of a part of the picture (the grid of A − λI,
/// the chart's curve where it leaves at ±180° and comes back at the other edge). The
/// picture moves evenly, but the cut ends of a line that turns slide along the edge
/// far faster than anything else moves, and a point of a wrapped curve goes from one
/// edge to the other: `eidos motion`, which follows lines and points, would take both
/// for jumps. A strip is a fill, which it does not follow.
fn strip(f: &mut Frame, a: Vec2, b: Vec2, width: f32, color: Color) {
    let d = b - a;
    let len = d.length();
    if len.is_nan() || len <= 1e-9 || !a.is_finite() || !b.is_finite() {
        return;
    }
    let n = d.perp() / len * (0.5 * width / px_per_unit(f));
    f.dl.tris_alpha.quad((a - n).extend(0.0), (b - n).extend(0.0), (b + n).extend(0.0), (a + n).extend(0.0), color);
}

/// The part of the segment a–b inside the box lo–hi, as a strip.
fn clipped_strip(f: &mut Frame, a: Vec2, b: Vec2, lo: Vec2, hi: Vec2, width: f32, color: Color) {
    if let Some((p, q)) = clip(a, b, lo, hi) {
        strip(f, p, q, width, color);
    }
}

/// A convex polygon cut down to the part inside a box (Sutherland–Hodgman).
fn clip_polygon(points: &[Vec2], lo: Vec2, hi: Vec2) -> Vec<Vec2> {
    let mut out = points.to_vec();
    let edges: [(Vec2, f32); 4] = [(Vec2::X, lo.x), (-Vec2::X, -hi.x), (Vec2::Y, lo.y), (-Vec2::Y, -hi.y)];
    for (n, k) in edges {
        let input = std::mem::take(&mut out);
        let inside = |p: Vec2| p.dot(n) >= k;
        for i in 0..input.len() {
            let (p, q) = (input[i], input[(i + 1) % input.len()]);
            if inside(p) {
                out.push(p);
            }
            if inside(p) != inside(q) {
                let (dp, dq) = (p.dot(n) - k, q.dot(n) - k);
                let t = dp / (dp - dq);
                if t.is_finite() {
                    out.push(p + (q - p) * t);
                }
            }
        }
    }
    out
}

/// The labels of one part, kept inside it: a `Room` walled off everywhere else.
fn room_in(f: &Frame, lo: Vec2, hi: Vec2) -> Room {
    let mut room = Room::new(f);
    let (Some(p), Some(q)) = (on_screen(f, lo.extend(0.0)), on_screen(f, hi.extend(0.0))) else { return room };
    let (min, max) = (p.min(q), p.max(q));
    let view = f.view_size();
    let big = view.max_element() * 4.0 + 1e4;
    room.wall(Vec2::new(-big, -big), Vec2::new(big, min.y));
    room.wall(Vec2::new(-big, max.y), Vec2::new(big, big));
    room.wall(Vec2::new(-big, min.y), Vec2::new(min.x, max.y));
    room.wall(Vec2::new(max.x, min.y), Vec2::new(big, max.y));
    room
}

/// Logical pixels per unit of the scene, as it is zoomed now.
fn px_per_unit(f: &Frame) -> f32 {
    1.0 / (f.cam.world_per_px(Vec3::ZERO) * f.px).max(1e-9)
}

/// How wide a label is: words by their number of characters, as `text_width` counts
/// them, and a text with formulas in it exactly (measured once and kept by the frame,
/// where `text_width` would work its formulas out again at every call).
fn width(f: &Frame, text: &str, size: f32) -> f32 {
    if crate::typeset::has_math(text) {
        f.text_size(text, size).x
    } else {
        text_width(text, size)
    }
}

/// The text that fits `room` logical pixels, the longest first (the last if none).
fn fitting<'a>(f: &Frame, options: &[&'a str], room: f32, size: f32) -> &'a str {
    options.iter().copied().find(|t| width(f, t, size) <= room).unwrap_or(options[options.len() - 1])
}

// ----- the scene --------------------------------------------------------------------

/// What the sweep is doing.
#[derive(Clone, Copy, Debug, PartialEq)]
enum Sweep {
    Off,
    /// Going round: seconds since it set off (it gathers speed from rest), and its turns.
    Going(f32, Turns),
    /// Resting on a kept direction: seconds left, and its turns.
    Resting(f32, Turns),
}

/// A sweep goes twice round, so that every stop is seen twice, and then comes to rest:
/// where it set off (an angle), and how far it has still to go (radians).
#[derive(Clone, Copy, Debug, PartialEq)]
struct Turns {
    from: f64,
    left: f64,
}

impl Sweep {
    /// A sweep from x at the angle `from`, already `since` seconds under way.
    fn set_off(from: f64, since: f32) -> Sweep {
        Sweep::Going(since, Turns { from, left: 2.0 * TAU64 })
    }
}

/// A value on its way from one setting to the next, by the scene's time.
#[derive(Clone, Copy, Debug, PartialEq)]
struct Move {
    from: f64,
    /// Seconds so far, and in all.
    t: f32,
    secs: f32,
}

impl Move {
    fn share(&self) -> f64 {
        smoothstep(self.t / self.secs.max(1e-6)) as f64
    }
}

pub struct Eigen {
    /// The matrix as set, and the one the picture is on its way from (with how far
    /// along it is, in seconds), after a new one is chosen.
    a: Mat2,
    a_from: Mat2,
    a_t: Option<f32>,
    /// Where x is, as an angle round the circle, as set; on its way there after an
    /// experiment; and as drawn, which glides after a click onto a kept direction.
    theta: f64,
    theta_move: Option<Move>,
    theta_drawn: f64,
    sweep: Sweep,
    /// λ, the same way.
    lambda: f64,
    lambda_move: Option<Move>,
    lambda_drawn: f64,
    /// Where λ goes next, and in how many seconds, once it is where it is going.
    lambda_then: Option<(f64, f32)>,
    /// The frames of the parts as drawn, gliding to fit a new matrix: how far the
    /// planes reach, and the graph's window.
    reach_drawn: f64,
    window_drawn: [f64; 4],
    /// Which way along each kept line its arrow pointed in the last frame: a line has
    /// two directions, and an arrow drawn on it should not flip from one to the other
    /// while the matrix changes.
    pointing: [DVec2; 2],
    /// The chart's curve as last drawn (see `TurnCurve`).
    curve: Option<TurnCurve>,
}

/// The chart's curve, the turn from x to Ax for x all round the circle: its pieces, each
/// cut to the chart, with the matrix and the chart they were made for. They hang on
/// nothing else, so they are made again only when one of those changes (to the last
/// bit): in every frame while a new matrix plays its way in, in none while x sweeps or
/// is dragged, or λ moves. Only the strips' width on the screen is worked out afresh.
struct TurnCurve {
    made_for: [u64; 12],
    pieces: Vec<(Vec2, Vec2)>,
}

impl TurnCurve {
    fn key(m: DMat2, g: &Graph) -> [u64; 12] {
        let [a, c, b, d] = m.to_cols_array().map(f64::to_bits);
        let [w0, w1, w2, w3] = g.win.map(f64::to_bits);
        let bits = |v: f32| u64::from(v.to_bits());
        [a, b, c, d, bits(g.min.x), bits(g.min.y), bits(g.max.x), bits(g.max.y), w0, w1, w2, w3]
    }

    /// The curve of `m`'s turns (whose kept lines are `k`) on the chart `g`: `kept` if
    /// it was made for those, else made afresh.
    fn of(kept: Option<TurnCurve>, m: DMat2, k: &Kept, g: &Graph) -> TurnCurve {
        let key = TurnCurve::key(m, g);
        if let Some(c) = kept.filter(|c| c.made_for == key) {
            return c;
        }
        // where it passes ±180° the curve leaves at one edge and comes back at the other
        let n = 360;
        let (bottom, top) = (g.at(0.0, -PI64).y, g.at(0.0, PI64).y);
        let (wlo, whi) = (Vec2::new(g.min.x, bottom), Vec2::new(g.max.x, top));
        // (and where A crushes x to 0 the turn has no value: there it jumps by 180°,
        // and the curve is broken rather than joined across)
        let crushed: Vec<f64> = k.lines().iter().filter(|(l, _)| *l == 0.0).flat_map(|(_, v)| {
            let a = v.y.atan2(v.x).rem_euclid(TAU64);
            [a, (a + PI64).rem_euclid(TAU64)]
        }).collect();
        let mut pieces = Vec::with_capacity(2 * n);
        let mut before: Option<(f64, f64)> = None;
        for i in 0..=n {
            let th = TAU64 * i as f64 / n as f64;
            let now = turn(m, th);
            if before.is_some_and(|(t0, _)| crushed.iter().any(|c| *c >= t0 && *c <= th)) {
                before = None;
            }
            if let (Some((t0, p0)), Some(p1)) = (before, now) {
                let step = wrap(p1 - p0);
                pieces.extend(clip(g.at(t0, p0), g.at(th, p0 + step), wlo, whi));
                if (p0 + step).abs() > PI64 {
                    pieces.extend(clip(g.at(t0, p1 - step), g.at(th, p1), wlo, whi));
                }
            }
            before = now.map(|p| (th, p));
        }
        TurnCurve { made_for: key, pieces }
    }
}

impl Eigen {
    /// (Nothing here needs the graphics device, so the tests can make one too.)
    fn create() -> Self {
        let a = rows(START);
        let m = wide(a);
        let mut s = Eigen {
            a,
            a_from: a,
            a_t: None,
            theta: 60f64.to_radians(),
            theta_move: None,
            theta_drawn: 60f64.to_radians(),
            // (already under way as the scene opens)
            sweep: Sweep::set_off(60f64.to_radians(), 1.0),
            lambda: 3.0,
            lambda_move: None,
            lambda_drawn: 3.0,
            lambda_then: None,
            reach_drawn: reach(m),
            window_drawn: window(trace(m), m.determinant()),
            pointing: [DVec2::X, DVec2::X],
            curve: None,
        };
        for (i, (_, v)) in kept(m).lines().iter().enumerate().take(2) {
            s.pointing[i] = *v;
        }
        s
    }

    /// Set up one of the experiments in `TRIALS`: everything glides to its place.
    fn set_up(&mut self, k: usize) {
        let s = TRIALS[k].2;
        self.set_matrix(rows(s.a));
        self.theta_move = Some(Move { from: self.theta_now(), t: 0.0, secs: MORPH });
        self.theta = s.x_deg.to_radians().rem_euclid(TAU64);
        let from = self.lambda_now();
        (self.lambda, self.lambda_then) = match s.lambda_via {
            Some(via) => {
                self.lambda_move = Some(Move { from, t: 0.0, secs: MORPH });
                (via, Some((s.lambda, s.lambda_secs)))
            }
            None => {
                self.lambda_move = Some(Move { from, t: 0.0, secs: s.lambda_secs });
                (s.lambda, None)
            }
        };
        self.sweep = if s.sweep { Sweep::set_off(self.theta, 0.0) } else { Sweep::Off };
    }

    /// A new matrix, which the picture plays its way to.
    fn set_matrix(&mut self, m: Mat2) {
        self.a_from = self.matrix_now();
        self.a = m;
        self.a_t = Some(0.0);
    }

    /// The matrix as drawn now.
    fn matrix_now(&self) -> Mat2 {
        match self.a_t {
            Some(t) => {
                let s = smoothstep(t / MORPH);
                self.a_from * (1.0 - s) + self.a * s
            }
            None => self.a,
        }
    }

    fn theta_now(&self) -> f64 {
        match self.theta_move {
            Some(mv) => mv.from + wrap(self.theta - mv.from) * mv.share(),
            None => self.theta_drawn,
        }
    }

    fn lambda_now(&self) -> f64 {
        match self.lambda_move {
            Some(mv) => mv.from + (self.lambda - mv.from) * mv.share(),
            None => self.lambda_drawn,
        }
    }

    /// The graph's window for the matrix as set.
    fn target_window(&self) -> [f64; 4] {
        let m = wide(self.a);
        window(trace(m), m.determinant())
    }

    /// Move everything on: `dt` seconds of the scene's time (what plays and sweeps)
    /// and `real_dt` of the clock's (what answers the hand).
    fn advance(&mut self, dt: f32, real_dt: f32) {
        // the matrix
        if let Some(t) = self.a_t {
            let t = t + dt;
            self.a_t = (t < MORPH).then_some(t);
        }
        // x: played to its place, then swept or glided
        if let Some(mut mv) = self.theta_move {
            mv.t += dt;
            self.theta_move = (mv.t < mv.secs).then_some(mv);
            if self.theta_move.is_none() {
                self.theta_drawn = self.theta;
            }
        }
        if self.theta_move.is_none() {
            let swept = self.sweep != Sweep::Off;
            self.advance_sweep(dt);
            // (the frame a sweep ends in too: its last step is drawn as it is taken,
            // as the steps onto every stop on the way are, not glided)
            if swept || self.sweep != Sweep::Off {
                self.theta_drawn = self.theta;
            } else {
                let k = 1.0 - (-(real_dt.max(0.0) / GLIDE) as f64).exp();
                self.theta_drawn += wrap(self.theta - self.theta_drawn) * k;
                if !self.theta_drawn.is_finite() || wrap(self.theta - self.theta_drawn).abs() < ARRIVED {
                    self.theta_drawn = self.theta;
                }
            }
        }
        // λ, the same way
        if let Some(mut mv) = self.lambda_move {
            mv.t += dt;
            self.lambda_move = (mv.t < mv.secs).then_some(mv);
            if self.lambda_move.is_none() {
                self.lambda_drawn = self.lambda;
                if let Some((to, secs)) = self.lambda_then.take() {
                    self.lambda_move = Some(Move { from: self.lambda, t: 0.0, secs });
                    self.lambda = to;
                }
            }
        } else {
            let k = 1.0 - (-(real_dt.max(0.0) / GLIDE) as f64).exp();
            self.lambda_drawn += (self.lambda - self.lambda_drawn) * k;
            let scale = self.window_drawn[1] - self.window_drawn[0];
            if !self.lambda_drawn.is_finite() || (self.lambda - self.lambda_drawn).abs() < ARRIVED * scale.max(1.0) {
                self.lambda_drawn = self.lambda;
            }
        }
        // the frames of the parts fit the matrix as set
        let m = wide(self.a);
        let (want_reach, want) = (reach(m), self.target_window());
        let k = 1.0 - (-(real_dt.max(0.0) / REFIT) as f64).exp();
        self.reach_drawn += (want_reach - self.reach_drawn) * k;
        if !self.reach_drawn.is_finite() || (want_reach - self.reach_drawn).abs() < 1e-5 * want_reach {
            self.reach_drawn = want_reach;
        }
        let span = (want[1] - want[0]).max(want[3] - want[2]).max(1e-9);
        for (drawn, to) in self.window_drawn.iter_mut().zip(want) {
            *drawn += (to - *drawn) * k;
            if !drawn.is_finite() || (to - *drawn).abs() < 1e-5 * span {
                *drawn = to;
            }
        }
        // λ stays inside the graph
        let [lo, hi, ..] = want;
        if self.lambda < lo || self.lambda > hi {
            let inside = self.lambda.max(lo).min(hi);
            if inside.is_finite() {
                self.lambda_move = Some(Move { from: self.lambda_now(), t: 0.0, secs: MORPH * 0.5 });
                (self.lambda, self.lambda_then) = (inside, None);
            }
        }
    }

    /// Carry x round, stopping for a moment on each kept direction; after two turns,
    /// come to rest on the next one (or, if there is none, where it set off), braking
    /// into it as into every stop on the way.
    fn advance_sweep(&mut self, dt: f32) {
        match self.sweep {
            Sweep::Off => {}
            Sweep::Resting(left, turns) => {
                let left = left - dt;
                self.sweep = if left > 0.0 { Sweep::Resting(left, turns) } else { Sweep::Going(0.0, turns) };
            }
            Sweep::Going(since, turns) => {
                let since = since + dt;
                let stops = stops(&kept(wide(self.matrix_now())));
                let ahead = stops.iter().map(|s| (s - self.theta).rem_euclid(TAU64)).filter(|r| *r > 1e-9).fold(f64::INFINITY, f64::min);
                // (the last stop: the first at or past the end of the two turns)
                let last = ahead >= turns.left - 1e-9;
                let goal = if ahead.is_finite() { ahead } else { turns.left.max(0.0) };
                let speed = SWEEP_SPEED.min(SWEEP_BRAKE * since as f64).min((2.0 * SWEEP_BRAKE * goal).sqrt());
                let step = speed * dt as f64;
                if step >= goal {
                    self.theta = (self.theta + goal).rem_euclid(TAU64);
                    // (exactly on the line, not nearly; or exactly where it set off, unless
                    // the kept lines went away while it was on its way past the end of its
                    // turns to the next, when it stops where it is)
                    if let Some(s) = stops.iter().min_by(|a, b| wrap(*a - self.theta).abs().total_cmp(&wrap(*b - self.theta).abs())) {
                        self.theta = *s;
                    } else if turns.left > 0.0 {
                        self.theta = turns.from;
                    }
                    self.sweep = if last { Sweep::Off } else { Sweep::Resting(REST, Turns { left: turns.left - goal, ..turns }) };
                } else {
                    self.theta = (self.theta + step).rem_euclid(TAU64);
                    self.sweep = Sweep::Going(since, Turns { left: turns.left - step, ..turns });
                }
            }
        }
    }

    /// The kept lines, each pointing the way it pointed in the last frame.
    fn oriented(&mut self, k: &Kept) -> Vec<(f64, DVec2)> {
        let mut lines = k.lines();
        for (i, (_, v)) in lines.iter_mut().enumerate().take(2) {
            if v.dot(self.pointing[i]) < 0.0 {
                *v = -*v;
            }
            self.pointing[i] = *v;
        }
        lines
    }

    /// Is anything still on its way?
    fn moving(&self) -> bool {
        self.sweep != Sweep::Off
            || self.a_t.is_some()
            || self.theta_move.is_some()
            || self.lambda_move.is_some()
            || self.lambda_then.is_some()
            || self.theta_drawn != self.theta
            || self.lambda_drawn != self.lambda
            || self.reach_drawn != reach(wide(self.a))
            || self.window_drawn != self.target_window()
    }

    /// x put on the circle by the hand at angle `psi`: onto a kept direction if it
    /// is near one.
    fn put_x(&mut self, psi: f64) {
        let stops = stops(&kept(wide(self.a)));
        let near = stops.iter().copied().min_by(|a, b| wrap(a - psi).abs().total_cmp(&wrap(b - psi).abs()));
        self.theta = match near {
            Some(s) if wrap(s - psi).abs() < SNAP_ANGLE => s,
            _ => psi.rem_euclid(TAU64),
        };
        self.theta_move = None;
        self.sweep = Sweep::Off;
    }

    /// λ put by the hand: onto a real root within `snap` of it.
    fn put_lambda(&mut self, l: f64, snap: f64) {
        let roots = kept(wide(self.a)).roots();
        let near = roots.iter().copied().min_by(|a, b| (a - l).abs().total_cmp(&(b - l).abs()));
        self.lambda = match near {
            Some(r) if (r - l).abs() < snap => r,
            _ => l,
        };
        (self.lambda_move, self.lambda_then) = (None, None);
    }
}

impl Scene for Eigen {
    fn view(&self) -> View {
        STAGE_VIEW
    }

    fn settings(&self, s: &mut ViewSettings) {
        s.grid = false;
        s.bloom = s.bloom.min(0.35);
    }

    /// Moving while x sweeps round, and while a change plays or glides to its place.
    fn animated(&self) -> bool {
        self.moving()
    }

    fn pointer_hint(&self) -> Option<&'static str> {
        Some(r"white dot on the circle: move $x$    white dot on the $\lambda$ axis: choose $\lambda$    drag elsewhere: move the picture    scroll: zoom")
    }

    fn ui(&mut self, ui: &mut egui::Ui, _: &Gfx) {
        ui::note(ui, r"A matrix $A$ turns most arrows. An eigenvector is one it does not turn:");
        ui::equation(ui, r"Ax = \lambda x,");
        ui::note(ui, r"the same arrow stretched by a number $\lambda$, its eigenvalue (reversed if $\lambda$ is below 0). The big plane carries $x$ round the circle; the dashed lines are where $Ax$ lines up with it. To find $\lambda$ without searching: $Ax = \lambda x$ says $(A - \lambda I)x = 0$, so at an eigenvalue $A - \lambda I$ crushes a line to $0$ and flattens the plane (the small plane). Its determinant, the yellow area, is then 0: the eigenvalues are the roots of the parabola $\det(A - \lambda I)$ (the graph).");

        ui::section(ui, "Things to try");
        for (k, trial) in TRIALS.iter().enumerate() {
            if ui::experiment(ui, trial.0, trial.1) {
                self.set_up(k);
            }
        }

        ui::section(ui, "The matrix $A$");
        let names: Vec<&str> = PRESETS.iter().map(|p| p.0).collect();
        if let Some(i) = ui::buttons(ui, &names) {
            self.set_matrix(rows(PRESETS[i].1));
        }
        let before = self.matrix_now();
        let mut m = self.a;
        if mat2_edit(ui, &mut m) {
            self.a = m;
            self.a_from = before;
            self.a_t = Some(0.0);
        } else if m != self.a {
            self.a = m;
        }
        ui::note(ui, "Written as on paper:");
        ui::equation(ui, r"A = \begin{bmatrix} a & b \\ c & d \end{bmatrix}");
        ui::note(ui, r"Type a number and the picture plays the change; drag one sideways and it follows. $2I$ doubles every arrow; $I$ leaves every arrow as it is; singular squashes the plane onto a line.");

        ui::section(ui, r"$x$ and $\lambda$");
        let mut sweeping = self.sweep != Sweep::Off;
        let label = r"sweep $x$ round the circle";
        crate::typeset::seen(label);
        if ui.checkbox(&mut sweeping, ui::label_text(ui, label, 12.5, None, false)).changed() {
            self.sweep = if sweeping { Sweep::set_off(self.theta, 0.0) } else { Sweep::Off };
        }
        ui::note(ui, r"$x$ stops for a moment on each dashed line, where $Ax$ lines up with it, and after two turns comes to rest on the next one (or where it set off, if there are none). Or drag the white dot on the circle: it clicks onto a dashed line when it comes near one.");
        let [lo, hi, ..] = self.target_window();
        let mut l = self.lambda as f32;
        if ui::slider(ui, r"$\lambda$", &mut l, lo as f32..=hi as f32) {
            self.lambda = l as f64;
            (self.lambda_move, self.lambda_then) = (None, None);
        }
        ui::note(ui, r"Or drag the white dot on the graph's $\lambda$ axis: it clicks onto a root when it comes near one.");

        ui::section(ui, "Reading the picture");
        ui::key(
            ui,
            &[
                (pal::WHITE, r"white: the unit arrow $x$, on the unit circle (grey); white dots are handles"),
                (AX_COL, r"cyan: $Ax$, and the ellipse it traces as $x$ goes round"),
                (TURN_COL, r"green: the turn from $x$ to $Ax$, as an arc, and on the chart for every $x$"),
                (EIG[0], r"orange: the larger eigenvalue $\lambda_1$: its root on the graph, its line of eigenvectors (dashed), the unit arrow $v_1$ on that line (the ring), and where $A - \lambda I$ sends $v_1$: $(\lambda_1 - \lambda)v_1$ (the arrow from $0$)"),
                (EIG[1], r"purple: the smaller eigenvalue $\lambda_2$, the same with $v_2$"),
                (pal::BLUE, r"blue grid: the square grid after $A - \lambda I$"),
                (pal::TEXT, r"light curve: $\det(A - \lambda I)$ for every $\lambda$, a parabola; its roots, where it meets 0, are the eigenvalues"),
                (pal::YELLOW, r"yellow: the unit square after $A - \lambda I$; its area is $\det(A - \lambda I)$, the height of the graph at $\lambda$"),
                (pal::PINK, r"pink instead: $\det(A - \lambda I)$ is below 0 ($A - \lambda I$ flips the plane over)"),
            ],
        );

        ui::section(ui, "Words");
        ui::term(ui, "Eigenvector:", r"an arrow $x$, not $0$, that $A$ does not turn: $Ax = \lambda x$. Every arrow on its line is one too.");
        ui::term(ui, r"Eigenvalue $\lambda$:", r"how many times $x$ is stretched along an eigenvector. Below 0, $Ax$ points the other way; 0, $Ax = 0$.");
        ui::term(ui, "$I$, the identity:", r"the matrix that changes nothing. $\lambda I$ stretches every arrow by $\lambda$, so $(A - \lambda I)x = Ax - \lambda x$.");
        ui::term(ui, "Determinant:", "the area the unit square becomes, below 0 if the plane is flipped over. It is 0 exactly when the plane is squashed flat, onto a line or a point.");
        ui::term(ui, "Characteristic polynomial:", r"$\det(A - \lambda I) = (a - \lambda)(d - \lambda) - bc = \lambda^2 - (a + d)\lambda + (ad - bc)$. Its roots are the eigenvalues.");
        ui::term(ui, "Trace:", r"$a + d$, the sum of the diagonal. It is the sum of the eigenvalues, and $\det A$ is their product.");
        ui::term(ui, "Double root:", r"the parabola only touches 0: $\lambda_1 = \lambda_2$.");
        ui::term(ui, "Complex roots:", r"when the parabola stays above 0 the roots are $p \pm qi$, with $i^2 = -1$: no real $\lambda$, and no direction is kept. What $p$ and $q$ do is for the scenes on powers and on flows.");
        ui::term(ui, "Symmetric:", r"equal to its own mirror image across the diagonal: $b = c$.");

        ui::section(ui, "Numbers");
        let m = wide(self.a);
        let (t, d) = (trace(m), m.determinant());
        let k = kept(m);
        // (every value a formula, as the polynomial, the points and the complex roots
        // must be: so that the numbers of the list are all set alike)
        let point = |v: DVec2| format!("$({}, {})$", num(v.x, 2), num(v.y, 2));
        let value = |v: f64| format!("${}$", short(v));
        ui::readout(ui, r"$\tr A = a + d$", value(t));
        ui::readout(ui, r"$\det A = ad - bc$", value(d));
        ui::readout(ui, r"$\det(A - \lambda I)$", format!("${}$", poly_text(t, d)));
        match k {
            Kept::Two([(l1, v1), (l2, v2)]) => {
                ui::readout_colored(ui, r"$\lambda_1$", value(l1), EIG[0]);
                ui::readout_colored(ui, r"$\lambda_2$", value(l2), EIG[1]);
                ui::readout_colored(ui, r"$v_1$, on $\lambda_1$'s line", point(v1), EIG[0]);
                ui::readout_colored(ui, r"$v_2$, on $\lambda_2$'s line", point(v2), EIG[1]);
                ui::readout(ui, r"$\lambda_1 + \lambda_2$, $\lambda_1 \lambda_2$", format!("${}, {}$", short(l1 + l2), short(l1 * l2)));
            }
            Kept::One(l, v) => {
                ui::readout_colored(ui, r"$\lambda$ (a double root)", value(l), EIG[0]);
                ui::readout_colored(ui, "$v$, on its only line", point(v), EIG[0]);
            }
            Kept::Every(l) => {
                ui::readout_colored(ui, r"$\lambda$ (a double root)", value(l), EIG[0]);
                ui::readout(ui, "eigenvectors", "every arrow but $0$");
                ui::note(ui, &format!("$A$ is ${}$ times $I$: it stretches every arrow by the same factor, so every direction is kept.", short(l)));
            }
            Kept::Turned { re, im } => {
                ui::readout(ui, "eigenvalues", format!(r"${} \pm {}i$", short(re), short(im)));
                ui::readout(ui, "eigenvectors", "none that are real");
                ui::note(ui, r"Complex eigenvalues: $\det(A - \lambda I)$ is never 0 for a real $\lambda$, and $A$ turns every direction.");
            }
        }
        if d.abs() <= 1e-12 * magnitude(m).powi(2) && magnitude(m) > 0.0 {
            ui::note(ui, r"$\det A = 0$: $A$ itself squashes the plane, so $\lambda = 0$ is an eigenvalue, and the line $A$ crushes to $0$ is its eigenvector.");
        }
        let x = unit(self.theta);
        let ax = m * x;
        ui::readout(ui, "$x$", point(x));
        ui::readout_colored(ui, "$Ax$", point(ax), AX_COL);
        let turned = turn(m, self.theta).map_or("none: $Ax = 0$".to_string(), |a| format!("${}°$", num(a.to_degrees(), 1)));
        ui::readout_colored(ui, "turn from $x$ to $Ax$", turned, TURN_COL);
        ui::readout(ui, r"$\lambda$", value(self.lambda));
        ui::readout(ui, r"$\det(A - \lambda I)$", value(poly(t, d, self.lambda)));
        ui::see_also(
            ui,
            &[
                // (a link with a formula in it is set on one line, which this one would
                // not fit: the matrix is named in words)
                ("powers", "what the eigenvalues do when the matrix is applied again and again"),
                ("expm", "flows: complex eigenvalues as turning"),
                ("quadform", "symmetric matrices and their axes"),
                ("basis", "eigenvectors as a basis: $A$ made diagonal"),
            ],
        );
    }

    fn frame(&mut self, f: &mut Frame) {
        self.advance(f.dt, f.real_dt);
        let view = f.view_size();
        let lay = Layout::of(view, f.top_inset);
        let m = wide(self.matrix_now());
        let k = kept(m);
        let theta = self.theta_now();
        let lambda = self.lambda_now();
        // (wide enough for the ellipse whatever the glide is doing)
        let sigma = svd2(m.as_mat2()).s.x;
        let r = (self.reach_drawn as f32).max(1.04 * sigma).max(1.04);
        let plane_a = Plane { centre: lay.a.0, half: lay.a.1, reach: r };
        let plane_b = Plane { centre: lay.b.0, half: lay.b.1, reach: r };
        let mut win = self.window_drawn;
        // (the lowest point and the value 0 always inside, whatever the glide is doing)
        let (t, d) = (trace(m), m.determinant());
        let low = -(t * t * 0.25 - d);
        let pad = 0.06 * (win[3] - win[2]).abs().max(1e-6);
        win[0] = win[0].min(t * 0.5 - 0.05 * (win[1] - win[0]));
        win[1] = win[1].max(t * 0.5 + 0.05 * (win[1] - win[0]));
        win[2] = win[2].min(low.min(0.0) - pad);
        win[3] = win[3].max(low.max(0.0) + pad);
        let graph = Graph { min: lay.graph.0, max: lay.graph.1, win };
        let chart = Graph { min: lay.chart.0, max: lay.chart.1, win: [0.0, TAU64, -PI64, PI64] };

        let big_enough = |half: f32| half * lay.ppu >= 24.0;
        let lines = self.oriented(&k);
        let labels_before = f.dl.labels.len();
        if big_enough(plane_a.half) {
            self.draw_plane(f, &lay, plane_a, m, &k, &lines, theta);
            self.draw_chart(f, &lay, chart, m, &k, theta);
        }
        if big_enough(plane_b.half) {
            self.draw_crush(f, &lay, plane_b, m, &k, &lines, lambda);
            self.draw_graph(f, &lay, graph, m, &k, lambda);
        }
        // (in a sliver of a window the parts are drawn without their words)
        if !lay.words {
            f.dl.labels.truncate(labels_before);
        }
    }
}

// ----- the four parts ------------------------------------------------------------------

impl Eigen {
    /// The plane of x and Ax. `lines` are the kept lines, each pointing the way it
    /// pointed in the last frame.
    #[allow(clippy::too_many_arguments)]
    fn draw_plane(&mut self, f: &mut Frame, lay: &Layout, p: Plane, m: DMat2, k: &Kept, lines: &[(f64, DVec2)], theta: f64) {
        let (lo, hi) = (p.lo(), p.hi());
        inset(f, lo, hi);
        frame_rect(f, lo, hi, 1.0, pal::FAINT);
        let r = p.reach as f64;
        let mut room = room_in(f, lo, hi);
        f.line(p.at(DVec2::new(-r, 0.0)), p.at(DVec2::new(r, 0.0)), 1.0, pal::FAINT.a(0.9));
        f.line(p.at(DVec2::new(0.0, -r)), p.at(DVec2::new(0.0, r)), 1.0, pal::FAINT.a(0.9));
        let wpp = 1.0 / px_per_unit(f);
        let size = lay.text;
        let screen = |f: &Frame, w: Vec2| on_screen(f, w.extend(0.0));
        let out = |f: &Frame, w: Vec2| match (screen(f, p.centre), screen(f, w)) {
            (Some(a), Some(b)) if (b - a).length() > 1e-3 => b - a,
            _ => Vec2::new(1.0, -1.0),
        };

        // the kept lines, brighter while x is on one or near it, each named at its end
        // (where neither x nor Ax ever comes: they stay inside the ellipse)
        let x = unit(theta);
        let lit = |v: DVec2| {
            let off = x.perp_dot(v).abs().min(1.0).asin();
            (-(off / 0.05).powi(2)).exp() as f32
        };
        let mut ends = Vec::new();
        for (i, (_, v)) in lines.iter().enumerate() {
            let c = EIG[i];
            let glow = lit(*v);
            if let Some((a, b)) = clip(p.at(-*v * 4.0 * r), p.at(*v * 4.0 * r), lo, hi) {
                f.dashed(a, b, 1.5 + 0.6 * glow, c.a(0.6 + 0.35 * glow));
                f.glow_path([a, b], 2.0, c.a(0.3 * glow));
                room.line(f, a.extend(0.0), b.extend(0.0));
                // (far enough in from the edge for the name to fit beside the line,
                // even where the line runs into a corner)
                ends.push((i, b - (b - a).normalize_or_zero() * 44.0 * wpp));
            }
        }
        for (i, end) in ends {
            let (l, v) = lines[i];
            let text = format!("${} = {}$", name(k, i), short(l));
            let across = out(f, p.at(v)).perp();
            let side = if across.y > 0.0 { -across } else { across };
            room.label(f, ("eig", i), end.extend(0.0), &text, size, EIG[i], side, 12.0, false);
        }

        // the unit circle, and the ellipse Ax traces as x goes round
        f.circle(p.centre, Vec3::Z, p.per_unit(), 1.2, pal::DIM.a(0.75));
        let ellipse: Vec<Vec2> = (0..=144).map(|i| p.at(m * unit(i as f64 / 144.0 * TAU64))).collect();
        f.path(ellipse.iter().copied(), 1.4, AX_COL.a(0.5));

        // x, Ax, and the turn from one to the other
        let ax = m * x;
        let (xw, axw) = (p.at(x), p.at(ax));
        // (the arc draws in to nothing as the turn nears 180°, where it would otherwise
        // flip from one side to the other)
        let full_arc = (34.0 * wpp / p.per_unit()) as f64;
        let shrink = |phi: f64| smoothstep(((PI64 - phi.abs()) / 0.45) as f32) as f64;
        let arc_r = turn(m, theta).map_or(full_arc, |phi| full_arc * shrink(phi));
        let mut arc_mid = None;
        if let Some(phi) = turn(m, theta) {
            let n = 28;
            let pts: Vec<Vec2> = (0..=n).map(|i| p.at(unit(theta + phi * i as f64 / n as f64) * arc_r)).collect();
            f.path(pts.iter().copied(), 1.8, TURN_COL);
            for w in pts.windows(2).step_by(4) {
                room.line(f, w[0].extend(0.0), w[1].extend(0.0));
            }
            arc_mid = Some((theta + phi * 0.5, phi));
        }
        f.arrow(p.centre, axw, 3.2, AX_COL);
        f.arrow(p.centre, xw, 3.2, pal::WHITE);
        room.line(f, p.centre.extend(0.0), axw.extend(0.0));
        room.line(f, p.centre.extend(0.0), xw.extend(0.0));
        // (a right angle where the kept lines cross at one)
        if let [(_, v1), (_, v2)] = lines[..] {
            if v1.dot(v2).abs() < 1e-6 {
                let corner = (14.0 * wpp).min(0.3 * p.per_unit());
                right_angle(f, p.centre.extend(0.0), v1.as_vec2().extend(0.0), v2.as_vec2().extend(0.0), corner, pal::WHITE.a(0.8));
            }
        }
        let mut handle = xw;
        if f.drag2(1, &mut handle, pal::WHITE) {
            let q = p.of(handle);
            if q.length() > 1e-9 {
                self.put_x(q.y.atan2(q.x));
            }
        }

        // what x is doing: on a kept line exactly, Ax = λx, said at Ax
        let on_line = lines.iter().enumerate().find(|(_, (_, v))| x.perp_dot(*v).abs() < 1e-9);
        let every = matches!(k, Kept::Every(_));
        let ax_text = match (on_line, k) {
            (Some((_, (l, _))), _) => ax_is(*l),
            (None, Kept::Every(l)) => ax_is(*l),
            _ => "$Ax$".to_string(),
        };
        let ax_colour = on_line.map_or(AX_COL, |(i, _)| EIG[i]);
        let dir_x = out(f, xw);
        room.label(f, "x", xw.extend(0.0), "$x$", size + 3.0, pal::WHITE, dir_x, 9.0, false);
        if ax.length() > 1e-9 {
            let dir = out(f, axw);
            room.label(f, "Ax", axw.extend(0.0), &ax_text, size + 2.0, ax_colour, dir, 8.0, false);
        } else {
            room.label(f, "Ax", p.centre.extend(0.0), &ax_text, size + 2.0, ax_colour, Vec2::new(-1.0, 1.0), 10.0, false);
        }
        if let (Some((mid, phi)), None, false, true) = (arc_mid, on_line, every, arc_r > 0.6 * full_arc) {
            // beside the arc, as near it as there is room; where that is crowded, the
            // number alone
            let deg = phi.to_degrees();
            let text = if deg.abs() >= 179.95 { "turned 180°".to_string() } else { format!("turned {}°", num(deg.abs(), 0)) };
            let along: Vec<Vec3> = [1.15, 1.6, 2.2].iter().map(|s| p.at(unit(mid) * arc_r * *s).extend(0.0)).collect();
            let dir = out(f, along[0].truncate());
            if !room.label_any(f, "turn", &along, &text, size, TURN_COL, dir, 4.0, true) {
                let number = format!("{}°", num(deg.abs(), 0));
                room.label_any(f, "turn", &along, &number, size, TURN_COL, dir, 3.0, false);
            }
        }
        // the circle and the ellipse, each named at a place of its own where there is
        // room (and not at all where there is none)
        for i in 0..24 {
            let (u, w) = (unit(i as f64 / 24.0 * TAU64), unit((i + 1) as f64 / 24.0 * TAU64));
            room.line(f, p.at(u).extend(0.0), p.at(w).extend(0.0));
        }
        for w in ellipse.windows(2).step_by(6) {
            room.line(f, w[0].extend(0.0), w[1].extend(0.0));
        }
        let near = |a: f64| [a, a + 0.25, a - 0.25];
        let on_circle: Vec<Vec3> = near(4.1).iter().map(|a| p.at(unit(*a)).extend(0.0)).collect();
        let dir = out(f, on_circle[0].truncate());
        room.label_any(f, "circle", &on_circle, "unit circle", size - 1.0, pal::DIM, dir, 5.0, true);
        // (not where it has shrunk to the point 0, for A = 0)
        let on_ellipse: Vec<Vec3> = near(2.2).iter().map(|a| p.at(m * unit(*a)).extend(0.0)).collect();
        let dir = out(f, on_ellipse[0].truncate());
        if svd2(m.as_mat2()).s.x * p.per_unit() > 20.0 * wpp {
            room.label_any(f, "ellipse", &on_ellipse, "$Ax$ as $x$ goes round", size - 1.0, AX_COL.a(0.85), dir, 5.0, true);
        }

        // over the plane: A itself, and what it keeps
        let Some(corner) = on_screen(f, Vec3::new(lo.x, hi.y, 0.0)).filter(|_| lay.words) else { return };
        let entries = [[m.x_axis.x, m.y_axis.x], [m.x_axis.y, m.y_axis.y]].map(|row| row.map(short));
        // (each number set as a formula, as "A =" is, which takes a character's room
        // more than its characters; room for −0.87 whatever is written now, so that
        // the brackets stay put as the numbers change)
        let widest = entries.iter().flatten().map(|t| t.chars().count()).max().unwrap_or(1).max(5) + 1;
        let rows_text: Vec<Vec<(String, Color)>> = entries.iter().map(|row| row.iter().map(|t| (format!("${t}$"), pal::TEXT)).collect()).collect();
        let msize = matrix_size(2, 2, widest, size);
        let top = corner.y - 8.0 - msize.y;
        f.text_px(Vec2::new(corner.x, top + msize.y * 0.5), "$A =$", size + 1.0, pal::TEXT, Align2::LEFT_CENTER);
        let mx = corner.x + width(f, "$A =$", size + 1.0) + 8.0;
        hud_matrix(f, Vec2::new(mx, top), &rows_text, widest, size, pal::DIM);
        let says = match k {
            Kept::Two(_) => ["$Ax$ lines up with $x$ on two lines (dashed)", "two lines kept (dashed)", "two lines kept"],
            Kept::One(..) => ["$Ax$ lines up with $x$ on one line only", "one line kept (dashed)", "one line kept"],
            Kept::Every(_) => [r"every direction is kept: $Ax = \lambda x$ for every $x$", "every direction kept", "all kept"],
            Kept::Turned { .. } => ["no direction is kept: $A$ turns every $x$", "every $x$ is turned", "none kept"],
        };
        let after = mx + msize.x + 14.0;
        let room_px = (lay.a.1 * 2.0) * px_per_unit(f) - (after - corner.x);
        f.text_px(Vec2::new(after, top + msize.y * 0.5), fitting(f, &says, room_px, size), size, pal::DIM, Align2::LEFT_CENTER);
    }

    /// The chart of the turn from x to Ax, against where x is on the circle.
    fn draw_chart(&mut self, f: &mut Frame, lay: &Layout, g: Graph, m: DMat2, k: &Kept, theta: f64) {
        let (lo, hi) = (g.min, g.max);
        inset(f, lo, hi);
        frame_rect(f, lo, hi, 1.0, pal::FAINT);
        let size = lay.text - 1.0;
        // lined up (0°) and reversed (±180°, the top and the bottom edge)
        f.line(g.at(0.0, 0.0), g.at(TAU64, 0.0), 1.0, pal::DIM.a(0.8));
        for y in [-PI64, PI64] {
            f.line(g.at(0.0, y), g.at(TAU64, y), 1.0, pal::DIM.a(0.5));
        }
        // where x is on a kept line
        let lines = k.lines();
        for (i, (l, v)) in lines.iter().enumerate() {
            let a = v.y.atan2(v.x).rem_euclid(TAU64);
            for s in [a, (a + PI64).rem_euclid(TAU64)] {
                f.dashed(g.at(s, -PI64), g.at(s, PI64), 1.0, EIG[i].a(0.55));
                let y = if *l > 0.0 {
                    Some(0.0)
                } else if *l < 0.0 {
                    Some(PI64)
                } else {
                    None
                };
                if let Some(y) = y {
                    f.dot(g.at(s, y), 8.0, EIG[i]);
                    if y != 0.0 {
                        f.dot(g.at(s, -y), 8.0, EIG[i]);
                    }
                }
            }
        }
        // the turn at every place round the circle
        let curve = TurnCurve::of(self.curve.take(), m, k, &g);
        for (p, q) in &curve.pieces {
            strip(f, *p, *q, 1.9, TURN_COL);
        }
        self.curve = Some(curve);
        if (0..8).all(|i| turn(m, i as f64 * 0.4).is_none()) {
            f.text(g.at(PI64, PI64 * 0.5), "$Ax = 0$ for every $x$: no turn to draw", size, pal::TEXT, Align2::CENTER_CENTER);
        }
        // where x is now (a strip, as the curve: going past 360° it comes back at 0°)
        let th = theta.rem_euclid(TAU64);
        strip(f, g.at(th, -PI64), g.at(th, PI64), 1.4, pal::WHITE.a(0.75));
        // its scales
        let ppu = px_per_unit(f);
        let tick = |f: &mut Frame, w: Vec2, text: &str, align: Align2| f.text(w, text, size - 1.0, pal::DIM, align);
        for (deg, text) in [(0.0, "0°"), (90.0, "90°"), (180.0, "180°"), (270.0, "270°"), (360.0, "360°")] {
            tick(f, g.at(f64::to_radians(deg), -PI64) - Vec2::new(0.0, 4.0 / ppu), text, Align2::CENTER_TOP);
        }
        for (y, text) in [(PI64, "180°"), (0.0, "0°"), (-PI64, "−180°")] {
            tick(f, g.at(0.0, y) - Vec2::new(5.0 / ppu, 0.0), text, Align2::RIGHT_CENTER);
        }
        let width = (hi.x - lo.x) * ppu;
        let title = fitting(f, &["the turn from $x$ to $Ax$, for $x$ all round the circle", "the turn from $x$ to $Ax$", "turn"], width, size);
        f.text(Vec2::new(lo.x, hi.y) + Vec2::new(0.0, 4.0 / ppu), title, size, TURN_COL, Align2::LEFT_BOTTOM);
        let right = fitting(f, &["0°: lined up    ±180°: reversed", "0°: lined up", ""], width - self::width(f, title, size) - 20.0, size - 1.0);
        f.text(Vec2::new(hi.x, hi.y) + Vec2::new(0.0, 4.0 / ppu), right, size - 1.0, pal::DIM, Align2::RIGHT_BOTTOM);
    }

    /// The plane as A − λI leaves it.
    #[allow(clippy::too_many_arguments)]
    fn draw_crush(&mut self, f: &mut Frame, lay: &Layout, p: Plane, m: DMat2, k: &Kept, lines: &[(f64, DVec2)], lambda: f64) {
        let (lo, hi) = (p.lo(), p.hi());
        inset(f, lo, hi);
        frame_rect(f, lo, hi, 1.0, pal::FAINT);
        let r = p.reach as f64;
        let mut room = room_in(f, lo, hi);
        f.line(p.at(DVec2::new(-r, 0.0)), p.at(DVec2::new(r, 0.0)), 1.0, pal::FAINT.a(0.9));
        f.line(p.at(DVec2::new(0.0, -r)), p.at(DVec2::new(0.0, r)), 1.0, pal::FAINT.a(0.9));
        let b = m - DMat2::IDENTITY * lambda;
        let det = b.determinant();
        let scale = magnitude(m).max(lambda.abs()).max(1e-12);
        let flat = det.abs() <= 1e-9 * scale * scale;

        // the square grid after A − λI, as far as it reaches
        let n = 12;
        for i in -n..=n {
            let (s, e) = (i as f64, n as f64);
            let w = if i == 0 { 1.8 } else { 1.0 };
            let c = pal::BLUE.a(if i == 0 { 0.85 } else { 0.32 });
            clipped_strip(f, p.at(b * DVec2::new(s, -e)), p.at(b * DVec2::new(s, e)), lo, hi, w, c);
            clipped_strip(f, p.at(b * DVec2::new(-e, s)), p.at(b * DVec2::new(e, s)), lo, hi, w, c);
        }

        // the unit square after it: its area is det(A − λI)
        let colour = if det >= 0.0 || flat { pal::YELLOW } else { pal::PINK };
        let corners = [DVec2::ZERO, b.x_axis, b.x_axis + b.y_axis, b.y_axis].map(|q| p.at(q));
        let inside = clip_polygon(&corners, lo, hi);
        for i in 1..inside.len().saturating_sub(1) {
            f.dl.tris_alpha.tri(inside[0].extend(0.0), inside[i].extend(0.0), inside[i + 1].extend(0.0), colour.a(0.25));
        }
        for i in 0..4 {
            clipped_strip(f, corners[i], corners[(i + 1) % 4], lo, hi, 1.6, colour.a(0.9));
            if let Some((a, c)) = clip(corners[i], corners[(i + 1) % 4], lo, hi) {
                room.line(f, a.extend(0.0), c.extend(0.0));
            }
        }

        // the lines A keeps: A − λI multiplies each by λᵢ − λ. A ring marks the unit
        // arrow vᵢ along it, the arrow is where A − λI sends it.
        let mut rings = Vec::new();
        let mut tips = Vec::new();
        let ring = 4.5 / px_per_unit(f);
        for (i, (l, v)) in lines.iter().enumerate() {
            if let Some((a, c)) = clip(p.at(-*v * 4.0 * r), p.at(*v * 4.0 * r), lo, hi) {
                f.dashed(a, c, 1.3, EIG[i].a(0.55));
                room.line(f, a.extend(0.0), c.extend(0.0));
            }
            f.circle(p.at(*v), Vec3::Z, ring, 1.6, EIG[i]);
            if let Some(q) = on_screen(f, p.at(*v).extend(0.0)) {
                room.boxed(q - Vec2::splat(6.0), q + Vec2::splat(6.0));
            }
            rings.push((i, *v));
            let tip = *v * (l - lambda);
            // (shortened to the edge if it runs out of the plane)
            let most = 0.95 * r;
            let tip = if tip.abs().max_element() > most { tip * (most / tip.abs().max_element()) } else { tip };
            f.arrow(p.centre, p.at(tip), 2.8, EIG[i]);
            room.line(f, p.centre.extend(0.0), p.at(tip).extend(0.0));
            tips.push((i, *l, *v, tip));
        }
        f.dot(p.centre, 6.0, pal::WHITE.a(0.8));

        // what A − λI does to each kept line, and what the grid is, in the corner (where
        // they stay put however the picture changes)
        let size = lay.text;
        let mut legend: Vec<(String, Color)> = Vec::new();
        for (i, l, _, _) in &tips {
            let vi = match k {
                Kept::Two(_) => ["v_1", "v_2"][(*i).min(1)],
                _ => "v",
            };
            legend.push((sent(vi, l - lambda, scale), EIG[*i]));
        }
        if !lay.narrow {
            legend.push((r"blue: the grid after $A - \lambda I$".to_string(), pal::BLUE.mix(pal::WHITE, 0.4)));
        }
        if let Some(corner) = on_screen(f, lo.extend(0.0)) {
            let line = size + 6.0;
            let widest = legend.iter().map(|(t, _)| width(f, t, size - 1.0)).fold(0.0, f32::max);
            let top = corner.y - 8.0 - line * legend.len() as f32;
            room.wall(Vec2::new(corner.x, top - 4.0), Vec2::new(corner.x + widest + 24.0, corner.y));
            for (n, (text, colour)) in legend.into_iter().enumerate() {
                f.caption_px(Vec2::new(corner.x + 8.0, top + line * (n as f32 + 0.5)), text, size - 1.0, colour, Align2::LEFT_CENTER);
            }
        }
        // the rings' names, where there is room, then the area (a name set as a formula
        // is taller than the word was, and stands a little further off its ring, clear
        // of the dashed line through it)
        for (i, v) in rings {
            let vi = match k {
                Kept::Two(_) => ["$v_1$", "$v_2$"][i.min(1)],
                _ => "$v$",
            };
            let at = p.at(v).extend(0.0);
            let dir = match (on_screen(f, p.centre.extend(0.0)), on_screen(f, p.at(v).extend(0.0))) {
                (Some(a), Some(c)) => (c - a).perp(),
                _ => Vec2::new(1.0, -1.0),
            };
            room.label(f, ("ring", i), at, vi, size, EIG[i], dir, 13.0, true);
        }
        let area_text = if flat { "area $0$".to_string() } else { format!("area ${}$", short(det)) };
        // (in the middle of the square while it has room, beside it when it is thin)
        let middle = p.at((b.x_axis + b.y_axis) * 0.5).extend(0.0);
        room.label(f, "area", middle, &area_text, size, colour.mix(pal::WHITE, 0.3), Vec2::ZERO, 8.0, false);
        if flat {
            let says = if magnitude(b) <= 1e-9 * scale { r"$A - \lambda I = 0$: everything lands on $0$" } else { "flat: everything lands on one line" };
            let at = Vec3::new(p.centre.x, lo.y, 0.0);
            room.label(f, "flat", at, says, size, pal::TEXT, Vec2::new(0.0, -1.0), 6.0, true);
        }

        // over it: the matrix A − λI, with λ
        let Some(corner) = on_screen(f, Vec3::new(lo.x, hi.y, 0.0)).filter(|_| lay.words) else { return };
        let entries = [[b.x_axis.x, b.y_axis.x], [b.x_axis.y, b.y_axis.y]].map(|row| row.map(short));
        // (each number set as a formula, as the name is, which takes a character's
        // room more than its characters; room for −0.87 whatever is written now, so
        // that the brackets stay put as the numbers change)
        let widest = entries.iter().flatten().map(|t| t.chars().count()).max().unwrap_or(1).max(5) + 1;
        let ent = |t: &String, c: Color| (format!("${t}$"), c);
        let rows_text: Vec<Vec<(String, Color)>> = vec![vec![ent(&entries[0][0], pal::WHITE), ent(&entries[0][1], pal::DIM)], vec![ent(&entries[1][0], pal::DIM), ent(&entries[1][1], pal::WHITE)]];
        let msize = matrix_size(2, 2, widest, size);
        let top = corner.y - 8.0 - msize.y;
        let head = r"$A - \lambda I =$";
        f.text_px(Vec2::new(corner.x, top + msize.y * 0.5), head, size + 1.0, pal::TEXT, Align2::LEFT_CENTER);
        let mx = corner.x + width(f, head, size + 1.0) + 8.0;
        hud_matrix(f, Vec2::new(mx, top), &rows_text, widest, size, pal::DIM);
        let after = mx + msize.x + 12.0;
        let said = format!(r"$\lambda = {}$", short(lambda));
        let right = corner.x + 2.0 * p.half * px_per_unit(f);
        if after + width(f, &said, size + 1.0) <= right {
            f.text_px(Vec2::new(after, top + msize.y * 0.5), said, size + 1.0, pal::WHITE, Align2::LEFT_CENTER);
        }
    }

    /// The graph of det(A − λI) against λ.
    fn draw_graph(&mut self, f: &mut Frame, lay: &Layout, g: Graph, m: DMat2, k: &Kept, lambda: f64) {
        let (lo, hi) = (g.min, g.max);
        inset(f, lo, hi);
        frame_rect(f, lo, hi, 1.0, pal::FAINT);
        let (t, d) = (trace(m), m.determinant());
        let [l0, l1, y0, y1] = g.win;
        let ppu = px_per_unit(f);
        let size = lay.text;
        let width = (hi.x - lo.x) * ppu;
        let mut room = room_in(f, lo, hi);

        // the axis, the value 0
        f.line(g.at(l0, 0.0), g.at(l1, 0.0), 1.2, pal::DIM);
        room.line(f, g.at(l0, 0.0).extend(0.0), g.at(l1, 0.0).extend(0.0));
        // the parabola over the part of the window it is in, coloured by its sign
        let c = t * 0.5;
        let reach = (y1 - poly(t, d, c)).max(0.0).sqrt();
        let (a, b) = ((c - reach).max(l0), (c + reach).min(l1));
        let n = 120;
        let mut samples: Vec<f64> = (0..=n).map(|i| a + (b - a) * i as f64 / n as f64).collect();
        samples.extend(k.roots().into_iter().filter(|r| *r > a && *r < b));
        samples.sort_by(f64::total_cmp);
        for w in samples.windows(2) {
            let (p, q) = (w[0], w[1]);
            let (yp, yq) = (poly(t, d, p).min(y1), poly(t, d, q).min(y1));
            let mid = poly(t, d, (p + q) * 0.5);
            let colour = if mid >= 0.0 { pal::YELLOW } else { pal::PINK };
            let (pa, pb, qa, qb) = (g.at(p, 0.0), g.at(p, yp), g.at(q, 0.0), g.at(q, yq));
            f.dl.tris_alpha.quad(pa.extend(0.0), qa.extend(0.0), qb.extend(0.0), pb.extend(0.0), colour.a(0.13));
        }
        let curve: Vec<Vec2> = (0..=n).map(|i| a + (b - a) * i as f64 / n as f64).map(|l| g.at(l, poly(t, d, l).min(y1))).collect();
        f.path(curve.iter().copied(), 2.2, pal::TEXT);
        for w in curve.windows(2).step_by(4) {
            room.line(f, w[0].extend(0.0), w[1].extend(0.0));
        }

        // the scale along the bottom
        let step = nice_step(((l1 - l0) / 5.0) as f32) as f64;
        let mut v = (l0 / step).ceil() * step;
        while v <= l1 + 1e-9 {
            if (v - l0) > 0.02 * (l1 - l0) && (l1 - v) > 0.02 * (l1 - l0) {
                f.text(g.at(v, y0) - Vec2::new(0.0, 3.0 / ppu), short(v), size - 2.0, pal::DIM, Align2::CENTER_TOP);
                f.line(g.at(v, 0.0) - Vec2::new(0.0, 3.0 / ppu), g.at(v, 0.0) + Vec2::new(0.0, 3.0 / ppu), 1.0, pal::DIM);
            }
            v += step;
        }
        f.text(g.at(l1, 0.0) + Vec2::new(-4.0 / ppu, 3.0 / ppu), r"$\lambda$", size + 1.0, pal::TEXT, Align2::RIGHT_BOTTOM);

        // the roots, in their colours
        let roots = k.roots();
        for (i, r) in roots.iter().enumerate() {
            f.dot(g.at(*r, 0.0), 10.0, EIG[i]);
        }

        // λ now: its handle on the axis, and the height of the graph above it
        let value = poly(t, d, lambda);
        let colour = if value >= 0.0 || value.abs() < 1e-12 { pal::YELLOW } else { pal::PINK };
        let top = g.at(lambda, value.max(y0).min(y1));
        f.line(g.at(lambda, 0.0), top, 3.0, colour);
        f.dot(top, 8.0, colour);
        room.line(f, g.at(lambda, 0.0).extend(0.0), top.extend(0.0));
        let mut handle = g.at(lambda, 0.0);
        if f.drag2(2, &mut handle, pal::WHITE) {
            let l = g.x_of(handle.x).max(l0).min(l1);
            let snap = SNAP_PX as f64 / ppu as f64 * (l1 - l0) / (hi.x - lo.x).max(1e-6) as f64;
            self.put_lambda(l, snap);
        }

        // the names of the roots, and the height of the graph at λ written at the top,
        // between the arms, where it stays put as λ moves (set as formulas they are
        // wider than their characters: in a narrow graph, a phone's or a small
        // window's, two roots are named without their numbers, which the dashed lines'
        // names give, and the height is a size smaller, to stay between the arms)
        let narrow = width < 260.0;
        let det_size = if narrow { size } else { size + 1.0 };
        for (i, r) in roots.iter().enumerate() {
            let text = match k {
                Kept::Two(_) if narrow => format!("${}$", name(k, i)),
                Kept::Two(_) => format!("${} = {}$", name(k, i), short(*r)),
                _ => format!(r"$\lambda = {}$, twice", short(*r)),
            };
            let at = g.at(*r, 0.0).extend(0.0);
            room.label_any(f, ("root", i), &[at], &text, size, EIG[i], Vec2::new(0.0, -1.0), 8.0, false);
        }
        let top_middle = Vec3::new((lo.x + hi.x) * 0.5, hi.y, 0.0);
        let flat = roots.iter().any(|r| (r - lambda).abs() <= 1e-9 * (l1 - l0).abs().max(1e-9));
        let det_text = format!(r"$\det({}) = {}$", minus_lambda(lambda), if flat { "0".to_string() } else { short(value) });
        if let Some(at) = on_screen(f, top_middle) {
            f.text_px(at + Vec2::new(0.0, 8.0), det_text, det_size, colour.mix(pal::WHITE, 0.3), Align2::CENTER_TOP);
            room.wall(at + Vec2::new(-90.0, 4.0), at + Vec2::new(90.0, 12.0 + size));
        }
        if let Kept::Turned { re, .. } = k {
            let long = format!("no real root: never below {}", short(poly(t, d, *re)));
            let text = if text_width(&long, size) <= 0.7 * width { long } else { "no real root".to_string() };
            if let Some(at) = on_screen(f, top_middle) {
                f.text_px(at + Vec2::new(0.0, 14.0 + size), text, size, pal::TEXT, Align2::CENTER_TOP);
            }
        }

        // over it: the polynomial, worked out (the second line's = under the first's)
        let Some(corner) = on_screen(f, Vec3::new(lo.x, hi.y, 0.0)) else { return };
        let line1 = format!(r"$\det(A - \lambda I) = {}$", product_text(m));
        let line2 = format!("$= {}$", poly_text(t, d));
        let (one, two, indent) = if self::width(f, &line1, size) <= width {
            let indent = self::width(f, r"$\det(A - \lambda I) =$", size) - self::width(f, "$=$", size);
            (line1, line2, indent)
        } else {
            (r"$\det(A - \lambda I)$".to_string(), line2, 0.0)
        };
        f.text_px(corner - Vec2::new(0.0, 2.0 * (size + 6.0) + 2.0), one, size, pal::TEXT, Align2::LEFT_TOP);
        f.text_px(corner - Vec2::new(-indent, size + 6.0), two, size, pal::TEXT, Align2::LEFT_TOP);
        // and under it, what the roots add and multiply to
        let Some(foot) = on_screen(f, Vec3::new(lo.x, lo.y, 0.0)) else { return };
        let sums = match k {
            Kept::Two([(a, _), (b, _)]) => format!(r"$\lambda_1 + \lambda_2 = {} = \tr A \qquad \lambda_1 \lambda_2 = {} = \det A$", short(a + b), short(a * b)),
            Kept::One(l, _) | Kept::Every(l) => format!(r"$\lambda + \lambda = {} = \tr A \qquad \lambda \cdot \lambda = {} = \det A$", short(2.0 * l), short(l * l)),
            Kept::Turned { re, im } => format!(r"roots ${} \pm {}i$: sum ${} = \tr A$, product ${} = \det A$", short(*re), short(*im), short(2.0 * re), short(re * re + im * im)),
        };
        let shorter = match k {
            Kept::Turned { re, im } => format!(r"roots ${} \pm {}i$", short(*re), short(*im)),
            _ => format!(r"sum ${} = \tr A$, product ${} = \det A$", short(t), short(d)),
        };
        let said = if self::width(f, &sums, size - 1.0) <= width { sums } else { shorter };
        f.text_px(foot + Vec2::new(0.0, size + 8.0), said, size - 1.0, pal::DIM, Align2::LEFT_TOP);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const FRAME: f32 = 1.0 / 60.0;

    /// An experiment as its button sets it up, played until what it sets has arrived
    /// (x, λ and the matrix: a sweep goes on for its two turns), at 60 frames a second.
    fn run(k: usize) -> Eigen {
        let mut s = Eigen::create();
        s.set_up(k);
        for _ in 0..(5.0 / FRAME) as usize {
            if s.a_t.is_none() && s.theta_move.is_none() && s.lambda_move.is_none() {
                break;
            }
            s.advance(FRAME, FRAME);
        }
        assert!(s.a_t.is_none() && s.theta_move.is_none() && s.lambda_move.is_none(), "{} has not arrived", TRIALS[k].0);
        s
    }

    fn m_of(s: &Eigen) -> DMat2 {
        wide(s.matrix_now())
    }

    /// Where the sweep rests in one whole turn of x from where it is now: the angles,
    /// each with the eigenvalue of the line it is on (none if it is on no kept line).
    fn rests_in_a_turn(s: &mut Eigen) -> Vec<(f64, Option<f64>)> {
        let (mut gone, mut resting, mut rests) = (0.0, false, Vec::new());
        let mut seconds = 0.0;
        while gone < TAU64 {
            let before = s.theta;
            s.advance(FRAME, FRAME);
            seconds += FRAME;
            assert!(seconds < 60.0, "the sweep has not gone round in a minute");
            gone += (s.theta - before).rem_euclid(TAU64);
            let now = matches!(s.sweep, Sweep::Resting(..));
            if now && !resting {
                let m = m_of(s);
                let x = unit(s.theta);
                let on = kept(m).lines().iter().find(|(_, v)| x.perp_dot(*v).abs() < 1e-12).map(|(l, _)| *l);
                rests.push((s.theta, on));
            }
            resting = now;
        }
        rests
    }

    /// What a sweep did, played to its end.
    #[derive(Debug)]
    struct Swept {
        /// Where it set off; how far it went (radians); where it rested on the way, with
        /// how far it had gone by then; and how far x was drawn to move in each frame.
        from: f64,
        gone: f64,
        rests: Vec<(f64, f64)>,
        steps: Vec<f64>,
    }

    /// Play the scene at `hz` frames a second until nothing moves: a sweep to its end.
    fn sweep_to_the_end(s: &mut Eigen, hz: f32) -> Swept {
        let dt = 1.0 / hz;
        let (Sweep::Going(_, turns) | Sweep::Resting(_, turns)) = s.sweep else { panic!("no sweep") };
        let mut out = Swept { from: turns.from, gone: 0.0, rests: Vec::new(), steps: Vec::new() };
        let (mut resting, mut seconds) = (false, 0.0);
        while s.moving() {
            let (before, drawn, sweeping) = (s.theta, s.theta_now(), s.sweep != Sweep::Off && s.theta_move.is_none());
            s.advance(dt, dt);
            seconds += dt;
            assert!(seconds < 60.0, "still moving after a minute");
            out.gone += wrap(s.theta - before);
            if sweeping {
                out.steps.push(wrap(s.theta_now() - drawn));
            }
            let now = matches!(s.sweep, Sweep::Resting(..));
            if now && !resting {
                out.rests.push((s.theta, out.gone));
            }
            resting = now;
        }
        out
    }

    /// The turn from x to Ax at every tenth of a degree round the circle.
    fn turns(m: DMat2) -> Vec<(f64, Option<f64>)> {
        (0..3600).map(|i| (i as f64 / 3600.0 * TAU64, turn(m, i as f64 / 3600.0 * TAU64))).collect()
    }

    /// The signed area of the unit square after `b`, from its corners.
    fn square_area(b: DMat2) -> f64 {
        let c = [DVec2::ZERO, b.x_axis, b.x_axis + b.y_axis, b.y_axis];
        (0..4).map(|i| c[i].perp_dot(c[(i + 1) % 4])).sum::<f64>() * 0.5
    }

    /// The first: the sweep rests four times a turn, on the orange line (Ax = 2x, a
    /// turn of 0) and on the purple one (Ax = −x, a turn of 180°), and only there.
    #[test]
    fn the_sweep_stops_where_ax_lines_up_with_x() {
        let mut s = run(0);
        assert_eq!(s.a, rows(START));
        let m = m_of(&s);
        let Kept::Two([(l1, v1), (l2, v2)]) = kept(m) else { panic!("{:?}", kept(m)) };
        assert!((l1 - 2.0).abs() < 1e-12 && (l2 + 1.0).abs() < 1e-12, "{l1} {l2}");
        let rests = rests_in_a_turn(&mut s);
        assert_eq!(rests.len(), 4, "{rests:?}");
        assert_eq!(rests.iter().filter(|r| r.1 == Some(l1)).count(), 2, "{rests:?}");
        assert_eq!(rests.iter().filter(|r| r.1 == Some(l2)).count(), 2, "{rests:?}");
        for (theta, l) in &rests {
            let x = unit(*theta);
            let l = l.expect("rests on a kept line");
            assert!((m * x - x * l).length() < 1e-12, "Ax = {l}x at {}°", theta.to_degrees());
            let t = turn(m, *theta).expect("Ax is not 0");
            assert!(if l > 0.0 { t.abs() < 1e-9 } else { (t.abs() - PI64).abs() < 1e-9 }, "turn {t} at λ = {l}");
            assert_eq!(ax_is(l), if l > 0.0 { "$Ax = 2x$" } else { "$Ax = -x$" });
        }
        // the dashed lines are those of the rests, and the orange one is λ₁ = 2
        let on = |v: DVec2| rests.iter().any(|(t, _)| unit(*t).perp_dot(v).abs() < 1e-12);
        assert!(on(v1) && on(v2));
        // and everywhere else A turns x: the chart reaches 0 and ±180° only there
        for (theta, t) in turns(m) {
            let t = t.expect("A has no line it crushes");
            let near = rests.iter().any(|(r, _)| wrap(theta - r).abs() < 1e-3);
            if !near {
                assert!(t.abs() > 1e-4 && PI64 - t.abs() > 1e-4, "{t} at {}°", theta.to_degrees());
            }
        }
        // twice round, and then at rest at the next stop, on the purple line: Ax = −x
        let mut s = Eigen::create();
        s.set_up(0);
        let swept = sweep_to_the_end(&mut s, 60.0);
        let x = unit(s.theta);
        assert!(swept.rests.iter().all(|(_, gone)| *gone < 2.0 * TAU64) && swept.gone >= 2.0 * TAU64, "{} after rests {:?}", swept.gone, swept.rests);
        assert!(x.perp_dot(v2).abs() < 1e-12 && (m * x + x).length() < 1e-12 && ax_is(l2) == "$Ax = -x$", "at rest at {}°", s.theta.to_degrees());
        assert!(TRIALS[0].1.contains("twice round") && TRIALS[0].1.contains("After its two turns $x$ comes to rest at the next stop, on the purple line."));
        assert_eq!(EIG[1], pal::PURPLE);
    }

    /// The second: λ goes to 3, then glides from 3 to 2 while det(A − λI) falls from 4
    /// to 0 all the way; at 2 the plane is flat, A − 2I crushes v₁ to 0, and x is on
    /// v₁'s line. (From the opening, where λ is 3, and from where the first leaves it.)
    #[test]
    fn sliding_lambda_to_a_root_flattens_the_plane() {
        for before in [None, Some(0)] {
            let mut s = match before {
                Some(k) => run(k),
                None => Eigen::create(),
            };
            s.set_up(1);
            let m = wide(s.a);
            let (t, d) = (trace(m), m.determinant());
            assert_eq!(poly(t, d, 3.0), 4.0);
            for _ in 0..(MORPH / FRAME) as usize + 1 {
                s.advance(FRAME, FRAME);
            }
            assert_eq!(s.lambda_now(), 3.0, "λ is at 3 when the slide begins");
            let mut last = f64::INFINITY;
            for _ in 0..(SLIDE / FRAME) as usize + 4 {
                s.advance(FRAME, FRAME);
                let l = s.lambda_now();
                let b = m - DMat2::IDENTITY * l;
                let area = square_area(b);
                assert!((area - poly(t, d, l)).abs() < 1e-12 && (b.determinant() - area).abs() < 1e-12);
                assert!(area <= last && area >= 0.0, "{area} after {last}");
                last = area;
            }
            assert!(s.lambda_move.is_none() && s.lambda_then.is_none() && s.lambda_now() == 2.0 && s.lambda == 2.0, "{:?} {:?} {} {}", s.lambda_move, s.lambda_then, s.lambda_now(), s.lambda);
            let b = m - DMat2::IDENTITY * 2.0;
            assert_eq!(b.determinant(), 0.0);
            assert!(magnitude(b) > 0.5, "flat onto a line, not onto a point");
            let Kept::Two([(l1, v1), _]) = kept(m) else { panic!() };
            assert_eq!(l1, 2.0);
            assert!((b * v1).length() < 1e-12, "{}", b * v1);
            assert_eq!(sent("v_1", l1 - 2.0, magnitude(m)), r"$v_1 \mapsto 0$");
            // every point lands on one line: the columns of A − 2I are in line
            assert!(b.x_axis.perp_dot(b.y_axis).abs() < 1e-12);
            // x is on the orange line
            assert!(unit(s.theta_now()).perp_dot(v1).abs() < 1e-12, "x at {}°", s.theta_now().to_degrees());
            // and then nothing moves
            assert!(!s.moving());
        }
    }

    /// The third: a turn by 60°, the same for every x; the sweep never pauses, goes
    /// twice round and comes to rest where it set off; the parabola λ² − λ + 1 is never
    /// below 0.75, its lowest at λ = 0.5; the roots 0.5 ± 0.87i.
    #[test]
    fn a_turn_keeps_no_direction() {
        let mut s = run(2);
        let m = m_of(&s);
        for (theta, t) in turns(m) {
            let t = t.expect("Ax is never 0");
            assert!((t.to_degrees() - 60.0).abs() < 1e-5, "{} at {}°", t.to_degrees(), theta.to_degrees());
            assert!(((m * unit(theta)).length() - 1.0).abs() < 1e-6);
        }
        assert!(rests_in_a_turn(&mut s).is_empty());
        let mut s = Eigen::create();
        s.set_up(2);
        let swept = sweep_to_the_end(&mut s, 60.0);
        assert!(swept.rests.is_empty() && (swept.gone - 2.0 * TAU64).abs() < 1e-9 && s.theta == swept.from && swept.from == 0.0, "{swept:?}");
        assert!(TRIALS[2].1.contains("the sweep never pauses: it goes twice round and comes to rest where it set off"));
        let (t, d) = (trace(m), m.determinant());
        assert_eq!(poly_text(t, d), r"\lambda^2 - \lambda + 1");
        let Kept::Turned { re, im } = kept(m) else { panic!("{:?}", kept(m)) };
        assert_eq!((short(re), short(im)), ("0.5".to_string(), "0.87".to_string()));
        assert_eq!(s.lambda, 0.5);
        let lowest = poly(t, d, 0.5);
        assert_eq!(short(lowest), "0.75");
        for i in 0..=1000 {
            let l = -20.0 + 40.0 * i as f64 / 1000.0;
            assert!(poly(t, d, l) >= lowest - 1e-12, "{l}");
            assert!((m - DMat2::IDENTITY * l).determinant() > 0.74);
        }
    }

    /// The fourth: a double root at 1 and one line only (the x axis); A − I lands
    /// everything on the x axis and the x axis on 0; two rests a turn; the chart only
    /// touches 0; and I has the same parabola with every direction kept.
    #[test]
    fn a_shear_keeps_one_direction() {
        let mut s = run(3);
        let m = m_of(&s);
        let (t, d) = (trace(m), m.determinant());
        assert_eq!(poly_text(t, d), r"\lambda^2 - 2\lambda + 1");
        assert_eq!(kept(m), Kept::One(1.0, DVec2::X));
        assert_eq!(s.lambda, 1.0);
        let b = m - DMat2::IDENTITY;
        for p in [DVec2::new(0.3, 1.7), DVec2::new(-2.0, 0.5), DVec2::new(1.0, -1.0)] {
            assert_eq!((b * p).y, 0.0);
        }
        assert_eq!(b * DVec2::X, DVec2::ZERO);
        let rests = rests_in_a_turn(&mut s);
        assert_eq!(rests.len(), 2, "{rests:?}");
        assert!(rests.iter().all(|(theta, l)| *l == Some(1.0) && unit(*theta).y.abs() < 1e-12));
        for (theta, turned) in turns(m) {
            let turned = turned.expect("Ax is never 0");
            assert!(turned <= 1e-12, "{} at {}°", turned.to_degrees(), theta.to_degrees());
            if unit(theta).y.abs() > 1e-6 {
                assert!(turned < 0.0);
            }
        }
        // the identity: the same parabola, and every direction kept
        let i = wide(rows(PRESETS[5].1));
        assert_eq!((PRESETS[5].0, trace(i), i.determinant()), ("$I$", t, d));
        assert_eq!(kept(i), Kept::Every(1.0));
    }

    /// The fifth: real eigenvalues 2 and −3, lines at right angles, and Ax longest (3)
    /// on the purple line and shortest (2) on the orange one.
    #[test]
    fn a_symmetric_matrix_has_lines_at_right_angles() {
        let s = run(4);
        let m = m_of(&s);
        assert_eq!(m.y_axis.x, m.x_axis.y);
        let Kept::Two([(l1, v1), (l2, v2)]) = kept(m) else { panic!("{:?}", kept(m)) };
        assert!((l1 - 2.0).abs() < 1e-12 && (l2 + 3.0).abs() < 1e-12);
        assert!(v1.dot(v2).abs() < 1e-12);
        let lengths: Vec<(f64, f64)> = (0..3600).map(|i| i as f64 / 3600.0 * TAU64).map(|t| (t, (m * unit(t)).length())).collect();
        let longest = lengths.iter().copied().fold((0.0, 0.0), |a, b| if b.1 > a.1 { b } else { a });
        let shortest = lengths.iter().copied().fold((0.0, f64::MAX), |a, b| if b.1 < a.1 { b } else { a });
        assert!((longest.1 - 3.0).abs() < 1e-5 && unit(longest.0).perp_dot(v2).abs() < 2e-3, "{longest:?}");
        assert!((shortest.1 - 2.0).abs() < 1e-5 && unit(shortest.0).perp_dot(v1).abs() < 2e-3, "{shortest:?}");
        assert!((m * v2).length() == 3.0 || ((m * v2).length() - 3.0).abs() < 1e-12);
    }

    /// "Every symmetric matrix has real eigenvalues and eigenvectors at right angles";
    /// and for every matrix: the kept lines are kept (Av = λv), A − λI crushes each to
    /// 0 at its root, the roots add up to the trace and multiply to the determinant,
    /// det(A − λI) is (a − λ)(d − λ) − bc, and where the roots are complex no x is
    /// lined up with Ax.
    #[test]
    fn what_the_panel_says_of_every_matrix() {
        let mut rng = Rng::new(11);
        let (mut real, mut complex) = (0, 0);
        for n in 0..800 {
            let e: Vec<f32> = (0..4).map(|_| rng.normal() * 1.6).collect();
            let mut m = wide(rows([e[0], e[1], e[2], e[3]]));
            if n % 2 == 0 {
                m.y_axis.x = m.x_axis.y;
                let k = kept(m);
                let Kept::Two([(_, v1), (_, v2)]) = k else { panic!("symmetric {m}: {k:?}") };
                assert!(v1.dot(v2).abs() < 1e-9, "{m}");
            }
            let (t, d) = (trace(m), m.determinant());
            let size = magnitude(m).max(1.0);
            let (a, b, c, dd) = (m.x_axis.x, m.y_axis.x, m.x_axis.y, m.y_axis.y);
            for l in [-3.0, -0.5, 0.0, 1.25, 4.0] {
                assert!(((a - l) * (dd - l) - b * c - poly(t, d, l)).abs() < 1e-9 * size * size);
                assert!(((m - DMat2::IDENTITY * l).determinant() - poly(t, d, l)).abs() < 1e-9 * size * size);
            }
            match kept(m) {
                Kept::Two([(l1, v1), (l2, v2)]) => {
                    real += 1;
                    assert!(l1 > l2);
                    for (l, v) in [(l1, v1), (l2, v2)] {
                        assert!((v.length() - 1.0).abs() < 1e-12);
                        assert!((m * v - v * l).length() < 1e-9 * size, "{m}: λ = {l}");
                        assert!(((m - DMat2::IDENTITY * l) * v).length() < 1e-9 * size);
                        assert!(poly(t, d, l).abs() < 1e-9 * size * size);
                    }
                    assert!((l1 + l2 - t).abs() < 1e-9 * size && (l1 * l2 - d).abs() < 1e-9 * size * size);
                }
                Kept::Turned { re, im } => {
                    complex += 1;
                    assert!((2.0 * re - t).abs() < 1e-9 * size && (re * re + im * im - d).abs() < 1e-9 * size * size);
                    for (theta, turned) in turns(m) {
                        let turned = turned.expect("Ax = 0 needs a real root, 0");
                        assert!(turned.abs() > 1e-6 && PI64 - turned.abs() > 1e-6, "{m} at {theta}");
                    }
                    let lowest = poly(t, d, re);
                    assert!(lowest > 0.0 && (lowest - im * im).abs() < 1e-9 * size * size);
                }
                other => panic!("{m}: {other:?}"),
            }
        }
        assert!(real > 400 && complex > 50, "{real} real, {complex} complex");
    }

    /// The numbers on the picture: the area of the square after A − λI is det(A − λI),
    /// the height of the graph; the arrows are (λᵢ − λ)vᵢ; the sums under the graph;
    /// and the formulas as written.
    #[test]
    fn the_picture_writes_what_it_shows() {
        let m = wide(rows(START));
        let (t, d) = (trace(m), m.determinant());
        assert_eq!(poly_text(t, d), r"\lambda^2 - \lambda - 2");
        assert_eq!(product_text(m), r"(1 - \lambda)(0 - \lambda) - 2 \cdot 1");
        for l in [-2.0, -1.0, 0.0, 0.5, 1.7, 2.0, 3.0] {
            let b = m - DMat2::IDENTITY * l;
            assert!((square_area(b) - poly(t, d, l)).abs() < 1e-12);
        }
        // the opening: λ = 3, the area 4, v₁ → −v₁ and v₂ → −4v₂
        let s = Eigen::create();
        assert_eq!(s.lambda, 3.0);
        assert_eq!(poly(t, d, 3.0), 4.0);
        let Kept::Two([(l1, v1), (l2, v2)]) = kept(m) else { panic!() };
        let b = m - DMat2::IDENTITY * 3.0;
        assert!((b * v1 - v1 * (l1 - 3.0)).length() < 1e-12 && (b * v2 - v2 * (l2 - 3.0)).length() < 1e-12);
        assert_eq!(sent("v_1", l1 - 3.0, 2.0), r"$v_1 \mapsto -v_1$");
        assert_eq!(sent("v_2", l2 - 3.0, 2.0), r"$v_2 \mapsto −4v_2$");
        assert_eq!(sent("v", 1.5, 2.0), r"$v \mapsto 1.5v$");
        assert_eq!(sent("v", 1.0, 2.0), r"$v \mapsto v$");
        assert_eq!((l1 + l2, l1 * l2), (1.0, -2.0));
        assert_eq!((short(l1 + l2), short(l1 * l2)), ("1".to_string(), "−2".to_string()));
        assert_eq!(minus_lambda(3.0), "A - 3I");
        assert_eq!(minus_lambda(-1.5), "A + 1.5I");
        assert_eq!(minus_lambda(1.0), "A - I");
        assert_eq!(minus_lambda(0.0), "A");
        assert_eq!(poly_text(-1.0, -6.0), r"\lambda^2 + \lambda - 6");
        assert_eq!(poly_text(0.0, 0.0), r"\lambda^2");
        assert_eq!(ax_is(0.0), "$Ax = 0$");
        assert_eq!(short(-0.004), "0");
        // the symmetric one, the shear and the turn, as their titles write them
        assert_eq!(product_text(wide(rows(SYMMETRIC))), r"(1 - \lambda)(−2 - \lambda) - 2 \cdot 2");
        assert_eq!(product_text(wide(rows(SHEAR))), r"(1 - \lambda)(1 - \lambda) - 1 \cdot 0");
        assert_eq!(product_text(wide(rows(TURN60))), r"(0.5 - \lambda)(0.5 - \lambda) - (−0.87) \cdot 0.87");
    }

    /// The degenerate matrices: 0 and multiples of I keep every direction; a singular
    /// one has the root 0, with the line it crushes; a double root typed as such is
    /// one; and nothing the picture is made of stops being a finite number, for these
    /// or for entries as large or as small as can be typed.
    #[test]
    fn degenerate_matrices_are_said_for_what_they_are() {
        assert_eq!(kept(DMat2::ZERO), Kept::Every(0.0));
        assert_eq!(kept(wide(rows([2.0, 0.0, 0.0, 2.0]))), Kept::Every(2.0));
        let singular = wide(rows(PRESETS[6].1));
        assert_eq!(singular.determinant(), 0.0);
        let k = kept(singular);
        let Kept::Two([(l1, _), (l2, v2)]) = k else { panic!("{k:?}") };
        assert_eq!((l1, l2), (2.0, 0.0));
        assert!((singular * v2).length() < 1e-12);
        let nilpotent = wide(rows([0.0, 1.0, 0.0, 0.0]));
        assert_eq!(kept(nilpotent), Kept::One(0.0, DVec2::X));
        let cases = [
            rows([0.0; 4]),
            rows([1.0, 0.0, 0.0, 1.0]),
            rows([-2.0, 0.0, 0.0, -2.0]),
            rows(PRESETS[6].1),
            rows([0.0, 1.0, 0.0, 0.0]),
            rows([100.0, -100.0, 100.0, 100.0]),
            rows([100.0, 100.0, 100.0, 100.0]),
            rows([-100.0, 100.0, -100.0, -100.0]),
            rows([1e-7, 0.0, 0.0, -1e-7]),
            rows([1e-7, 1e-7, 0.0, 1e-7]),
            rows([3e-38, 0.0, 0.0, 0.0]),
            rows([1.0, 1.0, 1.0, 1.0]),
            rows([0.0, -1.0, 1.0, 0.0]),
        ];
        for a in cases {
            let m = wide(a);
            let k = kept(m);
            let ok = |v: f64| v.is_finite();
            match k {
                Kept::Two([(l1, v1), (l2, v2)]) => assert!(ok(l1) && ok(l2) && v1.is_finite() && v2.is_finite(), "{a}"),
                Kept::One(l, v) => assert!(ok(l) && v.is_finite()),
                Kept::Every(l) => assert!(ok(l)),
                Kept::Turned { re, im } => assert!(ok(re) && ok(im) && im > 0.0),
            }
            assert!(stops(&k).iter().all(|s| s.is_finite()));
            let w = window(trace(m), m.determinant());
            assert!(w.iter().all(|v| v.is_finite()) && w[0] < w[1] && w[2] < 0.0 && w[3] > 0.0, "{a}: {w:?}");
            for r in k.roots() {
                assert!(r > w[0] && r < w[1], "{a}: root {r} outside {w:?}");
            }
            let r = reach(m);
            assert!(r.is_finite() && r >= 1.0, "{a}: {r}");
            for (_, t) in turns(m).iter().step_by(37) {
                assert!(t.is_none_or(|t| t.is_finite()));
            }
            // the scene with it, every part moved on
            let mut s = Eigen::create();
            s.set_matrix(a);
            s.sweep = Sweep::set_off(s.theta, 0.0);
            for _ in 0..240 {
                s.advance(FRAME, FRAME);
            }
            s.put_x(1.0);
            s.put_lambda(0.3, 0.1);
            for _ in 0..60 {
                s.advance(FRAME, FRAME);
            }
            assert!(s.theta_now().is_finite() && s.lambda_now().is_finite() && s.reach_drawn.is_finite() && s.window_drawn.iter().all(|v| v.is_finite()), "{a}");
            assert!(!s.moving(), "{a}: still moving");
        }
    }

    /// The hand: x clicks onto a kept line within about 3°, and not further off; λ onto
    /// a root within the reach it is given.
    #[test]
    fn handles_click_onto_what_they_are_near() {
        let mut s = Eigen::create();
        let k = kept(wide(s.a));
        let first = stops(&k)[0];
        s.put_x(first + 0.03);
        assert_eq!(s.theta, first);
        assert_eq!(s.sweep, Sweep::Off);
        s.put_x(first - 0.05);
        assert_eq!(s.theta, first);
        s.put_x(first + 0.08);
        assert!((s.theta - (first + 0.08)).abs() < 1e-12);
        s.put_lambda(2.04, 0.05);
        assert_eq!(s.lambda, 2.0);
        s.put_lambda(-0.97, 0.05);
        assert_eq!(s.lambda, -1.0);
        s.put_lambda(1.5, 0.05);
        assert_eq!(s.lambda, 1.5);
        // and the drawn x glides there, and is there in the end
        for _ in 0..60 {
            s.advance(FRAME, FRAME);
        }
        assert_eq!((s.theta_now(), s.lambda_now()), (s.theta, s.lambda));
    }

    /// Every experiment comes to rest, whatever was on screen before (one that sweeps
    /// once its two turns are done); a matrix typed in plays its way there and stops.
    #[test]
    fn changes_play_and_come_to_rest() {
        for before in 0..TRIALS.len() {
            for k in 0..TRIALS.len() {
                let mut s = run(before);
                s.set_up(k);
                for _ in 0..(6.0 / FRAME) as usize {
                    s.advance(FRAME, FRAME);
                }
                let setup = TRIALS[k].2;
                assert_eq!(s.a, rows(setup.a));
                assert_eq!(s.matrix_now(), s.a);
                assert_eq!(s.lambda_now(), setup.lambda, "{} after {}", TRIALS[k].0, TRIALS[before].0);
                assert_eq!(s.sweep != Sweep::Off, setup.sweep);
                if !setup.sweep {
                    assert!(!s.moving(), "{} after {}", TRIALS[k].0, TRIALS[before].0);
                    assert!((s.theta_now() - setup.x_deg.to_radians()).abs() < 1e-12);
                }
                for _ in 0..(40.0 / FRAME) as usize {
                    s.advance(FRAME, FRAME);
                }
                assert!(!s.moving() && !s.animated(), "{} after {}: still moving", TRIALS[k].0, TRIALS[before].0);
            }
        }
    }

    /// Every sweep ends by itself, at any frame rate: from the opening, after every
    /// experiment that sweeps, and with every quick choice of A swept as the "sweep"
    /// box sets it off. x goes at least twice round, resting on every stop at least
    /// twice, and then comes to rest at the first stop at or past the end of its two
    /// turns, braking into it as into every stop on the way; with no stop to make, it
    /// comes to rest exactly where it set off. Then nothing moves, and the scene asks
    /// for no more frames; the box, or an experiment, sets it off again.
    #[test]
    fn every_sweep_ends_after_two_turns() {
        let starts = || {
            let mut out: Vec<(String, Eigen)> = vec![("the opening".to_string(), Eigen::create())];
            for (k, (name, ..)) in TRIALS.iter().enumerate().filter(|(_, t)| t.2.sweep) {
                let mut s = Eigen::create();
                s.set_up(k);
                out.push((name.to_string(), s));
            }
            for (name, a) in PRESETS {
                let mut s = Eigen::create();
                s.set_matrix(rows(a));
                for _ in 0..(MORPH / FRAME) as usize + 2 {
                    s.advance(FRAME, FRAME);
                }
                s.sweep = Sweep::set_off(s.theta, 0.0);
                out.push((format!("the preset {name}"), s));
            }
            out
        };
        for hz in [30.0, 60.0, 144.0] {
            let dt = 1.0 / hz;
            for (name, mut s) in starts() {
                let swept = sweep_to_the_end(&mut s, hz);
                let m = m_of(&s);
                let k = kept(m);
                let stops = stops(&k);
                assert!(swept.gone >= 2.0 * TAU64 - 1e-9, "{name} at {hz} Hz: only {}° round", swept.gone.to_degrees());
                // every stop rested on twice, and the last rest the first stop past two turns
                for stop in &stops {
                    let seen = swept.rests.iter().filter(|(at, _)| at == stop).count() + usize::from(s.theta == *stop);
                    assert!(seen >= 2, "{name} at {hz} Hz: the stop at {}° seen {seen} times", stop.to_degrees());
                }
                assert!(swept.rests.iter().all(|(_, gone)| *gone < 2.0 * TAU64), "{name} at {hz} Hz: {:?}", swept.rests);
                let x = unit(s.theta);
                if stops.is_empty() {
                    assert!(s.theta == swept.from && (swept.gone - 2.0 * TAU64).abs() < 1e-9, "{name} at {hz} Hz: at rest at {}°, set off from {}°", s.theta.to_degrees(), swept.from.to_degrees());
                } else {
                    assert!(stops.contains(&s.theta) && k.lines().iter().any(|(_, v)| x.perp_dot(*v).abs() < 1e-12), "{name} at {hz} Hz: at rest at {}°", s.theta.to_degrees());
                    for (l, v) in k.lines() {
                        if x.perp_dot(v).abs() < 1e-12 {
                            assert!((m * x - x * l).length() < 1e-12 * magnitude(m).max(1.0), "{name}: Ax = {l}x");
                        }
                    }
                }
                // braking into the end: drawn steps that only shrink, ending with a small one
                let n = swept.steps.len();
                let last = &swept.steps[n - (0.4 * hz) as usize..];
                assert!(last.windows(2).all(|w| w[1] <= w[0] + 1e-12) && last.iter().all(|d| *d >= 0.0), "{name} at {hz} Hz: {last:?}");
                assert!(swept.steps.iter().all(|d| *d <= SWEEP_SPEED * dt as f64 + 1e-12), "{name} at {hz} Hz: a step faster than the sweep goes");
                assert!(last[last.len() - 1] < 0.1 * SWEEP_SPEED * dt as f64, "{name} at {hz} Hz: the last step {}", last[last.len() - 1]);
                // and then still, asking for nothing
                assert!(!s.animated() && s.sweep == Sweep::Off && s.theta_now() == s.theta);
                let at = s.theta;
                for _ in 0..(5.0 * hz) as usize {
                    s.advance(dt, dt);
                }
                assert!(!s.animated() && s.theta == at && s.theta_now() == at, "{name} at {hz} Hz");
                // set off again, by the box and by an experiment
                s.sweep = Sweep::set_off(s.theta, 0.0);
                assert!(s.animated());
                assert!(sweep_to_the_end(&mut s, hz).gone >= 2.0 * TAU64 - 1e-9 && !s.animated());
                s.set_up(0);
                assert!(s.animated() && sweep_to_the_end(&mut s, hz).gone >= 2.0 * TAU64 - 1e-9 && !s.animated());
            }
        }
        // a new matrix while it sweeps: still twice round, and then at rest
        let mut s = Eigen::create();
        for _ in 0..(5.0 / FRAME) as usize {
            s.advance(FRAME, FRAME);
        }
        s.set_matrix(rows(TURN60));
        let swept = sweep_to_the_end(&mut s, 60.0);
        assert!(swept.gone > TAU64 && !s.animated());
    }

    /// The chart's curve is kept from frame to frame, and made again whenever the
    /// matrix or the chart's place changes: whatever the inputs did (a quick choice and
    /// the way it plays in, a number typed, an experiment, a window of another shape),
    /// what is drawn is what would be made afresh.
    #[test]
    fn the_charts_curve_follows_the_matrix_and_the_window() {
        // (as `draw_chart` keeps it, against one made afresh; true if it was made afresh)
        fn check(s: &Eigen, view: Vec2, kept_curve: &mut Option<TurnCurve>) -> bool {
            let lay = Layout::of(view, 0.0);
            let g = Graph { min: lay.chart.0, max: lay.chart.1, win: [0.0, TAU64, -PI64, PI64] };
            let m = m_of(s);
            let k = kept(m);
            let before = kept_curve.as_ref().map(|c| c.made_for);
            let curve = TurnCurve::of(kept_curve.take(), m, &k, &g);
            let fresh = TurnCurve::of(None, m, &k, &g);
            assert!(curve.pieces == fresh.pieces && curve.made_for == fresh.made_for, "{m} in {view}");
            let made = before != Some(curve.made_for);
            *kept_curve = Some(curve);
            made
        }
        let mut s = Eigen::create();
        let mut rng = Rng::new(7);
        let mut kept_curve: Option<TurnCurve> = None;
        let mut made = 0;
        let views = [Vec2::new(1100.0, 620.0), Vec2::new(390.0, 800.0), Vec2::new(1600.0, 1000.0)];
        for (n, view) in views.iter().enumerate() {
            // the window changes; then quick choices, each played in, frame by frame
            made += usize::from(check(&s, *view, &mut kept_curve));
            for (_, a) in PRESETS.iter().skip(n) {
                s.set_matrix(rows(*a));
                for _ in 0..(MORPH / FRAME) as usize + 2 {
                    s.advance(FRAME, FRAME);
                    made += usize::from(check(&s, *view, &mut kept_curve));
                }
            }
            // numbers typed (the matrix as it is set, at once), and the experiments
            for _ in 0..20 {
                let e: Vec<f32> = (0..4).map(|_| rng.normal() * 1.5).collect();
                (s.a, s.a_t) = (rows([e[0], e[1], e[2], e[3]]), None);
                made += usize::from(check(&s, *view, &mut kept_curve));
            }
            for k in 0..TRIALS.len() {
                s.set_up(k);
                for _ in 0..90 {
                    s.advance(FRAME, FRAME);
                    made += usize::from(check(&s, *view, &mut kept_curve));
                }
            }
        }
        // and while nothing it hangs on changes (x sweeping, λ moving), it is not made again
        let before = made;
        s.set_up(1);
        for _ in 0..(MORPH / FRAME) as usize + 2 {
            s.advance(FRAME, FRAME);
        }
        made += usize::from(check(&s, views[0], &mut kept_curve));
        let once = made;
        for _ in 0..240 {
            s.advance(FRAME, FRAME);
            made += usize::from(check(&s, views[0], &mut kept_curve));
        }
        assert!(once > before && made == once, "made {} times while only λ moved", made - once);
    }

    /// The parts of the picture are where the layout puts them, inside the stage and
    /// clear of each other, in a desktop window, a phone's, and a small one.
    #[test]
    fn the_layout_fits_every_window() {
        for (view, inset) in [(Vec2::new(1080.0, 1000.0), 0.0), (Vec2::new(580.0, 700.0), 0.0), (Vec2::new(390.0, 800.0), 52.0), (Vec2::new(800.0, 390.0), 52.0), (Vec2::new(240.0, 180.0), 40.0), (Vec2::new(1.0, 1.0), 0.0)] {
            let lay = Layout::of(view, inset);
            let all = [lay.a.0, lay.chart.0, lay.chart.1, lay.b.0, lay.graph.0, lay.graph.1];
            assert!(all.iter().all(|p| p.is_finite()) && lay.a.1 > 0.0 && lay.b.1 > 0.0 && lay.ppu > 0.0, "{view}: {lay:?}");
            if view.x >= 390.0 && view.y >= 390.0 {
                let a = (lay.a.0 - Vec2::splat(lay.a.1), lay.a.0 + Vec2::splat(lay.a.1));
                let b = (lay.b.0 - Vec2::splat(lay.b.1), lay.b.0 + Vec2::splat(lay.b.1));
                let boxes = [a, lay.chart, b, lay.graph];
                let half = Vec2::new(view.x, view.y) * 0.5 / lay.ppu;
                for (i, p) in boxes.iter().enumerate() {
                    assert!(p.0.cmpge(-half).all() && p.1.cmple(half).all(), "{view}: part {i} {p:?} off the stage {half}");
                    for q in &boxes[..i] {
                        let apart = p.0.x >= q.1.x || q.0.x >= p.1.x || p.0.y >= q.1.y || q.0.y >= p.1.y;
                        assert!(apart, "{view}: parts {p:?} and {q:?} overlap");
                    }
                }
                // (the plane is the largest part)
                assert!(lay.a.1 >= lay.b.1, "{view}");
            }
        }
    }
}

//! Flows and the matrix exponential: let Ax be the velocity of the point x, and the
//! whole plane flows; following the flow for a time t is a single matrix, e^(At).

use super::*;
use glam::{DMat2, DVec2};

pub const DEF: SceneDef = SceneDef {
    id: "expm",
    title: "The matrix exponential",
    group: EXPONENTIAL,
    blurb: "Let $Ax$ be a velocity: saddles, spirals and centres, and the map that carries every point along for a time $t$.",
    make: |g| Box::new(Expm::new(Some(g))),
};

// ----- the arithmetic, in double precision -------------------------------------------

/// The longest time the flow is followed for: one full turn of the matrix that turns.
const T_MAX: f64 = std::f64::consts::TAU;
/// The largest entry of A either way. With t up to `T_MAX` no eigenvalue is beyond 8
/// in size, so nothing drawn or written is beyond e^(8 × 2π), about 10²², times the
/// start: far from overflow, even in single precision.
const ENTRY_MAX: f64 = 4.0;
/// How far out the start x₀ may be dragged.
const X0_MAX: f64 = 40.0;
/// The most terms of the series after I.
const N_MAX: usize = 20;
/// What counts as exactly zero, for a number of the size of the entries (squared, for
/// a determinant).
const EXACT: f64 = 1e-9;

/// A matrix by its rows, [[a, b], [c, d]], the way matrices are written.
fn rows([a, b, c, d]: [f64; 4]) -> DMat2 {
    DMat2::from_cols(DVec2::new(a, c), DVec2::new(b, d))
}

fn trace(m: DMat2) -> f64 {
    m.x_axis.x + m.y_axis.y
}

/// The size of the largest entry.
fn size(m: DMat2) -> f64 {
    m.to_cols_array().iter().fold(0.0f64, |s, v| s.max(v.abs()))
}

/// (tr A / 2)² − det A, a quarter of the discriminant of λ² − (tr A) λ + det A:
/// positive for two real eigenvalues, negative for a complex pair, zero for a double
/// one. Worked out as ((a − d)/2)² + bc, so that a matrix with a double eigenvalue
/// gives 0 exactly, rather than the difference of two nearly equal numbers.
fn quarter_disc(m: DMat2) -> f64 {
    let half = 0.5 * (m.x_axis.x - m.y_axis.y);
    half * half + m.y_axis.x * m.x_axis.y
}

/// e^(At), in closed form. A = sI + B with s = tr A / 2, and B, the rest, squares to
/// qI (q = `quarter_disc`). So e^(At) = e^(st) (c I + g B), where c = cosh(√q t) and
/// g = sinh(√q t) / √q if q > 0, the same with cos and sin of √−q t if q < 0, and
/// c = 1, g = t if q = 0. Near q t² = 0 the three are one power series, and are summed
/// as one: nothing is ever divided by a number near zero.
fn expm(a: DMat2, t: f64) -> DMat2 {
    let s = 0.5 * trace(a);
    let b = a - DMat2::IDENTITY * s;
    let q = quarter_disc(a);
    let z = q * t * t;
    let (c, g) = if z.abs() < 1.0 {
        // c = Σ zᵏ/(2k)!, g = t Σ zᵏ/(2k + 1)!: twelve terms are exact to the last digit
        let (mut c, mut g, mut tc, mut tg) = (1.0, 1.0, 1.0, 1.0);
        for k in 1..=12 {
            let k = k as f64;
            tc *= z / ((2.0 * k - 1.0) * (2.0 * k));
            tg *= z / ((2.0 * k) * (2.0 * k + 1.0));
            c += tc;
            g += tg;
        }
        (c, g * t)
    } else if q > 0.0 {
        let r = q.sqrt();
        ((r * t).cosh(), (r * t).sinh() / r)
    } else {
        let r = (-q).sqrt();
        ((r * t).cos(), (r * t).sin() / r)
    };
    (DMat2::IDENTITY * c + b * g) * (s * t).exp()
}

/// The terms of the series for e^(At)x, from the 0th to the `n`th: x, tAx, (tA)²x/2!, …
fn terms(a: DMat2, t: f64, x: DVec2, n: usize) -> Vec<DVec2> {
    let mut out = vec![x];
    let mut term = x;
    for k in 1..=n {
        term = a * term * (t / k as f64);
        out.push(term);
    }
    out
}

/// I + At + (At)²/2! + … + (At)ⁿ/n!.
fn partial_sum(a: DMat2, t: f64, n: usize) -> DMat2 {
    let (mut sum, mut term) = (DMat2::IDENTITY, DMat2::IDENTITY);
    for k in 1..=n {
        term = term * a * (t / k as f64);
        sum += term;
    }
    sum
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

/// What kind of flow a matrix makes: where it is on the map of trace and determinant.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Kind {
    /// A = 0.
    Still,
    /// Both eigenvalues 0, A not 0: every point slides along one direction.
    Shear,
    /// det A = 0, tr A ≠ 0: a line of points at rest, the rest flowing in to it or out from it.
    RestIn,
    RestOut,
    Saddle,
    Sink,
    Source,
    /// A double eigenvalue with one eigenline.
    ImproperSink,
    ImproperSource,
    /// A multiple of I: every line an eigenline.
    StarSink,
    StarSource,
    SpiralSink,
    SpiralSource,
    Centre,
}

fn kind(a: DMat2) -> Kind {
    let scale = size(a);
    // (entries are typed to hundredths, or come from a matrix that is: one of a
    // trillionth is what is left of zero)
    if scale < 1e-12 {
        return Kind::Still;
    }
    let (tr, det, q) = (trace(a), a.determinant(), quarter_disc(a));
    let zero_tr = tr.abs() <= EXACT * scale;
    let flat = EXACT * scale * scale;
    let sign = |inward: Kind, outward: Kind| if tr < 0.0 { inward } else { outward };
    if det.abs() <= flat {
        return if zero_tr { Kind::Shear } else { sign(Kind::RestIn, Kind::RestOut) };
    }
    if det < 0.0 {
        Kind::Saddle
    } else if q < -flat {
        if zero_tr {
            Kind::Centre
        } else {
            sign(Kind::SpiralSink, Kind::SpiralSource)
        }
    } else if q <= flat {
        match eigen(a) {
            Eigen::Real { lines: Lines::Every, .. } => sign(Kind::StarSink, Kind::StarSource),
            _ => sign(Kind::ImproperSink, Kind::ImproperSource),
        }
    } else {
        sign(Kind::Sink, Kind::Source)
    }
}

impl Kind {
    /// Its name, as the picture's heading gives it.
    fn name(self) -> &'static str {
        match self {
            Kind::Still => "no flow",
            Kind::Shear => "a shear flow",
            Kind::RestIn | Kind::RestOut => "a line of rest points",
            Kind::Saddle => "a saddle",
            Kind::Sink => "a sink",
            Kind::Source => "a source",
            Kind::ImproperSink => "an improper sink",
            Kind::ImproperSource => "an improper source",
            Kind::StarSink => "a star sink",
            Kind::StarSource => "a star source",
            Kind::SpiralSink => "a spiral sink",
            Kind::SpiralSource => "a spiral source",
            Kind::Centre => "a centre",
        }
    }

    /// What the flow does, in a few words.
    fn says(self) -> &'static str {
        match self {
            Kind::Still => "$A = 0$: no point moves",
            Kind::Shear => "the points of one line stay put; every other point slides along beside it, faster the further out it is",
            Kind::RestIn => "the points of one line stay put; every other point flows straight in to it",
            Kind::RestOut => "the points of one line stay put; every other point flows straight out from it",
            Kind::Saddle => "in along one eigenline, out along the other",
            Kind::Sink => "everything flows in to $0$",
            Kind::Source => "everything flows out from $0$",
            Kind::ImproperSink => "everything flows in to $0$, turning to come in along the one eigenline",
            Kind::ImproperSource => "everything flows out from $0$, turning to leave along the one eigenline",
            Kind::StarSink => "everything flows straight in to $0$, from every direction",
            Kind::StarSource => "everything flows straight out from $0$, in every direction",
            Kind::SpiralSink => "everything turns round $0$ and flows in to it",
            Kind::SpiralSource => "everything turns round $0$ and flows out from it",
            Kind::Centre => "everything goes round a closed loop, for ever",
        }
    }
}

/// The eigenvalues in words and formulas: "$\lambda = 1.00$ and $−1.00$", "$\lambda = −0.25
/// \pm 1.50i$"; without the "λ =" if not `named`.
fn eigen_text(e: Eigen, k: Kind, named: bool) -> String {
    let l = if named { r"\lambda = " } else { "" };
    match e {
        Eigen::Real { l1, lines: Lines::One(_) | Lines::Every, .. } => format!("${l}{}$ twice", num(l1, 2)),
        Eigen::Real { l1, l2, .. } => format!("${l}{}$ and ${}$", num(l1, 2), num(l2, 2)),
        Eigen::Complex { im, .. } if k == Kind::Centre => format!(r"${l}\pm {}i$", num(im, 2)),
        Eigen::Complex { re, im } => format!(r"${l}{} \pm {}i$", num(re, 2), num(im, 2)),
    }
}

/// A number for a matrix on the screen, as a formula, in five characters at most (so
/// that the brackets never move as the numbers change): two decimals, then one, then
/// none, and a power of ten beyond 9999 (set as 1 × 10⁴: drawn smaller if it is wider
/// than its cell, see `matrix`).
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

/// The widest `cell` can be, in characters of its number, a minus sign counted as two
/// (in a formula it is nearly twice as wide as a digit).
const CELL: usize = 6;

/// A distance for the screen: two decimals, or two figures in powers of ten when it is
/// smaller than that (and 0 only when it is 0).
fn distance(d: f64) -> String {
    if d == 0.0 || d >= 0.005 || !d.is_finite() {
        short(d)
    } else {
        format!("{d:.1e}").replace('-', "−")
    }
}

/// A number for the screen: two decimals and a true minus sign, and in powers of ten
/// once it is too long for that.
fn short(v: f64) -> String {
    if v.is_finite() && v.abs() >= 9999.5 {
        format!("{v:.1e}").replace('-', "−")
    } else {
        num(v, 2)
    }
}

// ----- the map of every flow -------------------------------------------------------

/// The part of the plane of (trace, determinant) that the map shows.
const MAP_TR: [f64; 2] = [-3.0, 3.0];
const MAP_DET: [f64; 2] = [-1.5, 3.0];

/// The matrix of trace τ and determinant δ that a drag on the map starts from when
/// A is a multiple of I: [[τ/2, −1], [δ − τ²/4, τ/2]]. At τ = 0, δ = 1 it is J, the
/// quarter turn.
fn canonical(tau: f64, det: f64) -> DMat2 {
    rows([0.5 * tau, -1.0, det - 0.25 * tau * tau, 0.5 * tau])
}

/// The matrices a drag on the map moves A among: one for each trace τ and determinant
/// δ, changing smoothly with them, with A itself at A's own place.
#[derive(Clone, Copy, Debug)]
enum Family {
    /// P C(τ, δ) P⁻¹ (C is `canonical`): the same flows, seen in the coordinates P.
    /// Every matrix but a multiple of I is one of these for some P, so a drag starts
    /// where A is. Crossing the parabola, its two eigenlines close up into one and
    /// give way to turning.
    Similar { p: DMat2, p_inv: DMat2 },
    /// A multiple of I is no other matrix seen in other coordinates. From one: τ/2 I
    /// plus √(τ²/4 − δ) diag(1, −1) below the parabola, √(δ − τ²/4) J above it.
    Scalar,
}

fn family_for(a: DMat2) -> Family {
    let b = a - DMat2::IDENTITY * (0.5 * trace(a));
    if size(b) <= EXACT * size(a).max(1e-12) {
        return Family::Scalar;
    }
    // A P = P C for P = [−B p₂, p₂], whatever p₂ is, as long as it is not along an
    // eigenvector of B. Of the directions p₂ can take, the one that makes P most nearly
    // a turn and an even stretch: the family is then A, distorted as little as it can be.
    let mut best = (f64::MIN, DMat2::IDENTITY);
    for k in 0..90 {
        let p2 = DVec2::from_angle(k as f64 * std::f64::consts::PI / 90.0);
        let p1 = -(b * p2);
        let p = DMat2::from_cols(p1, p2);
        let evenness = p.determinant().abs() / (p1.length_squared() + p2.length_squared());
        if evenness > best.0 {
            best = (evenness, p);
        }
    }
    Family::Similar { p: best.1, p_inv: best.1.inverse() }
}

impl Family {
    fn at(self, tau: f64, det: f64) -> DMat2 {
        let m = match self {
            Family::Similar { p, p_inv } => p * canonical(tau, det) * p_inv,
            Family::Scalar => {
                let d = 0.25 * tau * tau - det;
                let rest = if d >= 0.0 { rows([1.0, 0.0, 0.0, -1.0]) * d.sqrt() } else { rows([0.0, -1.0, 1.0, 0.0]) * (-d).sqrt() };
                DMat2::IDENTITY * (0.5 * tau) + rest
            }
        };
        within_limits(m)
    }
}

/// The matrix with every entry within ±`ENTRY_MAX` (and 0 for one that is not a number).
fn within_limits(m: DMat2) -> DMat2 {
    DMat2::from_cols_array(&m.to_cols_array().map(|v| if v.is_finite() { v.max(-ENTRY_MAX).min(ENTRY_MAX) } else { 0.0 }))
}

/// The tour of the map an experiment takes: a saddle, a sink, a spiral sink, a centre,
/// a spiral source, a source, and the saddle again, as (trace, determinant).
const TOUR: [(f64, f64); 7] = [(0.0, -1.0), (-2.4, 0.9), (-1.0, 2.0), (0.0, 2.0), (1.0, 2.0), (2.4, 0.9), (0.0, -1.0)];
/// Seconds spent at each stop, and on the way from one to the next.
const TOUR_STAY: f32 = 1.4;
const TOUR_LEG: f32 = 2.0;

fn tour_length() -> f32 {
    (TOUR.len() - 1) as f32 * (TOUR_STAY + TOUR_LEG) + TOUR_STAY
}

/// Where on the map the tour is, `s` seconds in.
fn tour_at(s: f32) -> (f64, f64) {
    let per = TOUR_STAY + TOUR_LEG;
    let k = ((s.max(0.0) / per).floor() as usize).min(TOUR.len() - 1);
    let into = s - k as f32 * per;
    if k + 1 >= TOUR.len() || into <= TOUR_STAY {
        return TOUR[k];
    }
    let u = smoothstep((into - TOUR_STAY) / TOUR_LEG) as f64;
    let (a, b) = (TOUR[k], TOUR[k + 1]);
    (a.0 + (b.0 - a.0) * u, a.1 + (b.1 - a.1) * u)
}

// ----- experiments and quick choices -------------------------------------------------

struct Trial {
    label: &'static str,
    look_for: &'static str,
    /// The matrix, by rows.
    a: [f64; 4],
    x0: [f64; 2],
    /// The time it is set to, and where playing on from there stops (if it plays).
    t: f64,
    play_to: Option<f64>,
    /// The series, counted up a term at a time.
    series: bool,
    /// The tour of the map.
    tour: bool,
}

const TRIALS: [Trial; 5] = [
    Trial {
        label: "a saddle",
        look_for: r"$\lambda = 1$ and $-1$, one of each sign. Points on the orange line move straight out, $e^t$ times as far at time $t$; points on the purple line move straight in, $e^{-t}$ times as far. Every other point does both: $x_0$ starts near the purple line, is drawn in towards $0$, then swept out along the orange one (the faint path ahead). The square is stretched one way and squashed the other just as much: its area stays $1$, since $\tr A = 0$.",
        a: [1.0, -1.0, 0.0, -1.0],
        x0: [-0.8, -2.4],
        t: 0.0,
        play_to: Some(1.5),
        series: false,
        tour: false,
    },
    Trial {
        label: "a spiral sink",
        look_for: r"$\lambda = -0.25 \pm 1.5i$. Every point turns round $0$ at 1.5 radians per unit of time (the imaginary part) while its distance shrinks as $e^{-0.25t}$ (the real part). At $t = 2\pi/1.5 = 4.19$, one full turn, $x_0$ is back on the ray it started on, at 0.35 of its distance. The square turns and shrinks with it: its area is $e^{t \tr A} = e^{-0.5t}$, 0.12 by then.",
        a: [-0.25, -1.5, 1.5, -0.25],
        x0: [2.4, 0.0],
        t: 0.0,
        play_to: Some(std::f64::consts::TAU / 1.5),
        series: false,
        tour: false,
    },
    Trial {
        label: "the matrix that turns",
        look_for: r"$J = \begin{bmatrix} 0 & -1 \\ 1 & 0 \end{bmatrix}$ turns every arrow a quarter turn, so the velocity $Jx$ is always at right angles to $x$, and every point goes round a circle. $e^{tJ}$ is the turn by the angle $t$: the square has turned a quarter at $t = \pi/2 = 1.57$, and all the way round at $t = 2\pi = 6.28$. $\lambda = \pm i$: real part $0$, so nothing grows or shrinks, and the area stays $1$.",
        a: [0.0, -1.0, 1.0, 0.0],
        x0: [1.8, 1.0],
        t: 0.0,
        play_to: Some(std::f64::consts::TAU),
        series: false,
        tour: false,
    },
    Trial {
        label: "the series, a term at a time",
        look_for: r"$x_0 = (0, -0.75)$ and $t = \pi$, a half turn: $e^{\pi J} x_0 = -x_0$ ($e^{i\pi} = -1$, in the plane). The red arrows are the terms $(tJ)^k x_0 / k!$, added one at a time. The first is one straight step along the velocity; each next one is turned a quarter and $\pi/k$ times as long. The sum swings out to 3.22 away from $-x_0$, then closes in: within 0.01 of it from $N = 10$ on.",
        a: [0.0, -1.0, 1.0, 0.0],
        x0: [0.0, -0.75],
        t: std::f64::consts::PI,
        play_to: None,
        series: true,
        tour: false,
    },
    Trial {
        label: "round the map of all flows",
        look_for: r"The dot on the map ($\tr A$ across, $\det A$ up) visits each kind of flow, and the picture follows. Below $\det A = 0$: a saddle. Above it, left of $\tr A = 0$ everything flows in, right of it out. Above the parabola $\det A = (\tr A)^2/4$ the eigenvalues are complex and the flow turns; on the line $\tr A = 0$ between the two spirals it goes round closed loops. Then drag the dot yourself.",
        a: [0.0, -1.0, -1.0, 0.0],
        x0: [1.5, 1.0],
        t: 0.8,
        play_to: None,
        series: false,
        tour: true,
    },
];

/// Quick choices of A: (button, the matrix by rows, what it shows).
const PRESETS: [(&str, [f64; 4], &str); 9] = [
    ("sink", [-1.0, 0.5, 0.5, -1.0], r"$\lambda = -0.5$ and $-1.5$, both negative: everything flows in to $0$. The part along the $\lambda = -1.5$ line dies faster, so in the end every point not on that line comes in along the $\lambda = -0.5$ line."),
    ("source", [1.0, 0.5, 0.5, 1.0], r"$\lambda = 1.5$ and $0.5$, both positive: everything flows out, and every point not on the $\lambda = 0.5$ line in the end leaves along the $\lambda = 1.5$ line, whose part grows faster."),
    ("saddle", [1.0, -1.0, 0.0, -1.0], r"$\lambda = 1$ and $-1$: in along the purple line, out along the orange one."),
    ("spiral sink", [-0.25, -1.5, 1.5, -0.25], r"$\lambda = -0.25 \pm 1.5i$: turning at the rate 1.5 and shrinking at the rate 0.25."),
    ("spiral source", [0.15, -1.5, 1.5, 0.15], r"$\lambda = 0.15 \pm 1.5i$: turning at the rate 1.5 and growing at the rate 0.15."),
    ("centre", [1.0, -2.0, 1.0, -1.0], r"$\tr A = 0$ and $\det A = 1$, so $\lambda = \pm i$: points go round ellipses, and every one is back where it started at $t = 2\pi$."),
    ("one eigenline", [-1.0, 1.0, 0.0, -1.0], r"$\lambda = -1$ twice, with a single eigenline: $e^{At} = e^{-t}(I + tN)$, with $N = A + I$, and that $t e^{-t}$ makes points swing round to come in along the one line."),
    ("a line at rest", [-0.5, -0.5, -0.5, -0.5], r"$\det A = 0$, so $\lambda = 0$ and $-1$. Every point on the $\lambda = 0$ line stays put; every other point flows straight to it along the other eigenline."),
    ("shear", [0.0, 1.0, 0.0, 0.0], r"$\lambda = 0$ twice, one eigenline. $A^2 = 0$, so $e^{At} = I + At$ exactly: every point slides sideways at a speed equal to its height, and the $x$ axis stays put."),
];

// ----- the streaks -------------------------------------------------------------------

/// The grid of cells the streaks are at home in, this many each way.
const GRID: usize = 28;
/// A row of every streak is laid each so much scene time, and between rows the newest
/// row follows the particles (`TrailSet::follow`): a streak moves in every frame, on
/// any display, and is the same stretch of motion on all of them.
const ROW_EVERY: f32 = 1.0 / 20.0;
/// Rows in a streak: its head, six rows behind it (the last 0.3 seconds), and one more
/// that has faded out by the time it is dropped.
const ROWS: u32 = 8;
/// The longest stretch of scene time a frame's streaks move through: a very long frame
/// falls behind instead.
const MOST_ROWS: f32 = 4.0;
/// How long a particle lives at most (one that hardly moves), and at least (one that
/// crosses the picture in a moment).
const LIFE_MOST: f32 = 3.5;
const LIFE_LEAST: f32 = 0.35;
/// How far a particle goes in its life, as a share of the picture's smaller side: its
/// life is as long as that takes (within the two limits above), so that fast and slow
/// flows alike keep every part of the picture as full of streaks as every other.
const REACH: f32 = 0.2;
/// How long a particle takes to fade in after it is born, and out before it dies (a
/// short-lived one, a share of its life instead).
const FADE_IN: f32 = 0.5;
const FADE_OUT: f32 = 0.7;
/// How bright a particle is that hardly moves, against one that does; and how far on
/// the screen (pixels, in its whole life) counts as moving.
const STILL: f32 = 0.35;
const MOVING_PX: f32 = 24.0;

fn streak_colour() -> Color {
    pal::BLUE.mix(pal::WHITE, 0.35)
}

/// The faint streaks: particles carried by the flow, each with a short tail that is
/// brightest at its head.
///
/// Each has a cell of its own of a grid over the picture, and a home at a random place
/// in it. It is born up the flow from its home, passes through it in the middle of
/// its life, when it is brightest, and dies as far down the flow: so the streaks are
/// spread evenly over the picture whatever the flow is, also where everything rushes
/// towards one point or away from it. (Born at home with a few seconds to live, as
/// they once were, the particles of a sink all ended in the middle and left the rest
/// of the picture empty.) Each fades in as it is born and out before it dies, so that
/// none of them blinks on or off.
struct Flow {
    pos: Vec<Vec2>,
    age: Vec<f32>,
    life: Vec<f32>,
    /// How bright it is at its brightest (less, if it hardly moves).
    bright: Vec<f32>,
    /// Born since the row was last written; and its streak cut in the newest row: no
    /// streak is drawn from where a particle died to where it is born.
    fresh: Vec<bool>,
    cut: Vec<bool>,
    row: Vec<LinePoint>,
    /// (None in the tests, which have no graphics device.)
    trails: Option<TrailSet>,
    /// Scene time since the last whole row was laid; and whether the newest row is
    /// that whole row (the next to be written is then a new one).
    owed: f32,
    whole: bool,
    rng: Rng,
}

impl Flow {
    fn new(gfx: Option<&Gfx>) -> Flow {
        let trails = gfx.map(|g| {
            let mut trails = TrailSet::new(g, (GRID * GRID) as u32, ROWS, true);
            trails.style.intensity = 1.0;
            trails.style.fade_power = 1.3;
            trails.style.taper = 0.6;
            trails
        });
        Flow { pos: Vec::new(), age: Vec::new(), life: Vec::new(), bright: Vec::new(), fresh: Vec::new(), cut: Vec::new(), row: Vec::new(), trails, owed: 0.0, whole: false, rng: Rng::new(5) }
    }

    /// Particle `i` born again: at home at a random place in its cell of the rectangle
    /// `lo..hi`, to pass there in the middle of a life long enough to go `REACH` of
    /// the picture, or as near that as the limits on a life let it.
    fn born(&mut self, i: usize, a: DMat2, lo: Vec2, hi: Vec2) {
        let cell = Vec2::new((i % GRID) as f32, (i / GRID % GRID) as f32);
        let at = (cell + Vec2::new(self.rng.f32(), self.rng.f32())) / GRID as f32;
        let home = (lo + (hi - lo) * at).as_dvec2();
        let size = (hi - lo).min_element().max(1e-6) as f64;
        let reach = REACH as f64 * size;
        // as long as the reach takes at the speed at home; less, where the flow would
        // carry it further than that before or after (near a point everything runs to)
        let speed = (a * home).length();
        let mut life = if speed * LIFE_MOST as f64 > reach { reach / speed } else { LIFE_MOST as f64 };
        let ends = |life: f64| (expm(a, -0.5 * life) * home, expm(a, 0.5 * life) * home);
        for _ in 0..8 {
            let (from, to) = ends(life);
            if life <= LIFE_LEAST as f64 || (from - home).length().max((to - home).length()) <= 0.75 * reach {
                break;
            }
            life *= 0.7;
        }
        let life = life.clamp(LIFE_LEAST as f64, LIFE_MOST as f64) * self.rng.range(0.8, 1.2) as f64;
        let (from, to) = ends(life);
        // (how far it goes on the screen, in pixels of a picture a thousand high)
        let moved = ((to - from).length() / size) as f32 * 1000.0;
        self.pos[i] = from.as_vec2();
        (self.age[i], self.life[i], self.fresh[i]) = (0.0, life as f32, true);
        self.bright[i] = STILL + (1.0 - STILL) * smoothstep(moved / MOVING_PX);
    }

    /// A new set, each part of the way through its life.
    fn scatter(&mut self, a: DMat2, lo: Vec2, hi: Vec2) {
        let n = GRID * GRID;
        self.pos = vec![Vec2::ZERO; n];
        (self.age, self.life, self.bright) = (vec![0.0; n], vec![1.0; n], vec![1.0; n]);
        (self.fresh, self.cut) = (vec![true; n], vec![false; n]);
        self.row = vec![crate::gfx::draw::BREAK; n];
        for i in 0..n {
            self.born(i, a, lo, hi);
            let age = self.rng.range(0.0, self.life[i]);
            self.pos[i] = (expm(a, age as f64) * self.pos[i].as_dvec2()).as_vec2();
            self.age[i] = age;
        }
        (self.owed, self.whole) = (0.0, false);
        if let Some(trails) = &mut self.trails {
            trails.clear();
        }
    }

    /// Carry every particle along the flow of `a` for the time `h`. Those that have
    /// lived their time (or have left all bounds) are born again.
    fn carry(&mut self, a: DMat2, h: f32, lo: Vec2, hi: Vec2) {
        let m = expm(a, h as f64).as_mat2();
        let far = 1e3 * (hi - lo).max_element().max(1e-6);
        for i in 0..self.pos.len() {
            let p = m * self.pos[i];
            self.age[i] += h;
            if self.age[i] >= self.life[i] || !p.is_finite() || p.abs().max_element() > far {
                self.born(i, a, lo, hi);
            } else {
                self.pos[i] = p;
            }
        }
    }

    /// Where the particles are, into the streaks' newest row: a new row if the one
    /// before was laid whole (returns true), the same row again otherwise.
    fn write(&mut self, colour: Color, gfx: Option<&Gfx>) -> bool {
        let new = std::mem::take(&mut self.whole);
        for i in 0..self.pos.len() {
            // (a cut in a row is kept for as long as the row is the newest: the next
            // row may then go on from the particle's new place)
            if new {
                self.cut[i] = false;
            }
            if std::mem::take(&mut self.fresh[i]) {
                self.cut[i] = true;
            }
            let (age, life) = (self.age[i], self.life[i]);
            let fade = smoothstep(age / FADE_IN.min(0.3 * life)) * smoothstep((life - age) / FADE_OUT.min(0.35 * life));
            let q = self.pos[i];
            // (a negative width cuts the streak)
            self.row[i] = LinePoint { pos: [q.x, q.y, 0.0], width: if self.cut[i] { -1.0 } else { 1.5 }, color: colour.a(0.5 * fade * self.bright[i]) };
        }
        if let (Some(trails), Some(gfx)) = (&mut self.trails, gfx) {
            trails.row_mut().copy_from_slice(&self.row);
            if new {
                trails.commit(gfx);
            } else {
                trails.update(gfx);
            }
        }
        new
    }

    /// Move the streaks on by `dt` of scene time along the flow of `a`, over the
    /// rectangle `lo..hi`: a row laid whenever `ROW_EVERY` has gone by, and the newest
    /// row left where the particles now are.
    fn advance(&mut self, a: DMat2, dt: f32, lo: Vec2, hi: Vec2, colour: Color, gfx: Option<&Gfx>) {
        if self.pos.is_empty() {
            self.scatter(a, lo, hi);
        }
        let mut left = if dt.is_finite() { dt.clamp(0.0, MOST_ROWS * ROW_EVERY) } else { 0.0 };
        let mut written = false;
        loop {
            let to_row = ROW_EVERY - self.owed;
            if left < to_row {
                break;
            }
            self.carry(a, to_row, lo, hi);
            self.write(colour, gfx);
            (self.whole, self.owed, written) = (true, 0.0, true);
            left -= to_row;
        }
        if left > 1e-6 {
            self.carry(a, left, lo, hi);
            self.write(colour, gfx);
            self.owed += left;
            written = true;
        }
        if !written && self.trails.as_ref().is_some_and(|t| !t.started()) {
            // (the first frame of all, if no time has passed in it: something to show)
            self.write(colour, gfx);
        }
        if let Some(trails) = &mut self.trails {
            trails.follow(if self.whole { 1.0 } else { self.owed / ROW_EVERY });
        }
    }
}

// ----- where things go ---------------------------------------------------------------

/// The part of the scene the view is fitted to (its centre, and half its size): the
/// paths, squares and sums of every experiment lie in it. (One for all of them: the
/// view stays put as one experiment gives way to the next.)
const FRAME_CENTRE: Vec2 = Vec2::new(0.3, 0.3);
const FRAME_HALF: Vec2 = Vec2::new(3.4, 2.9);

/// The screen panel's parts, in logical pixels.
#[derive(Clone, Debug)]
struct Panel {
    /// Its dark backing.
    min: Vec2,
    max: Vec2,
    /// The size of the numbers in the matrices.
    text: f32,
    /// Where the line saying what eᴬᵗ is goes (not on a phone).
    head: Option<Vec2>,
    /// Top-left corner of the row of matrices, and of the lines under it.
    mats: Vec2,
    lines: Vec2,
    /// Top-left of the series' matrix, when it is shown.
    series: Vec2,
    /// The two ends of the time track, at its middle.
    bar: (Vec2, Vec2),
    /// The play button.
    play: (Vec2, Vec2),
    map_title: Vec2,
    map: (Vec2, Vec2),
}

/// Rows of the panel, in logical pixels.
const HEAD_H: f32 = 18.0;
const LINE_H: f32 = 17.0;
const BAR_H: f32 = 28.0;

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
    fn of(view: Vec2, inset: f32, series: bool) -> Layout {
        let narrow = view.x < 620.0 && view.y >= 560.0;
        let pad = if narrow { 10.0 } else { 12.0 };
        let width = if narrow { view.x } else { 316.0 };
        // the numbers' size: as large as lets "A = [ ]  eᴬᵗ = [ ]" fit across, up to 14
        let [a_name, e_name] = lead_widths();
        let across = |s: f32| (a_name + e_name) * s + 2.0 * matrix_size(2, 2, CELL, s).x + 28.0;
        let mut text = 14.0f32;
        while across(text) > width - 2.0 * pad && text > 10.0 {
            text -= 0.5;
        }
        let mat_h = matrix_size(2, 2, CELL, text).y;
        // (on a phone the time is on the track, and the heading's line is left out)
        let head_h = if narrow { 0.0 } else { HEAD_H + 6.0 };
        // everything above the map's title, from the panel's top
        // (the series' sum is a matrix on a wide canvas, a line of words on a phone)
        let series_h = match (series, narrow) {
            (false, _) => 0.0,
            (true, false) => 8.0 + LINE_H + mat_h,
            (true, true) => 8.0 + LINE_H,
        };
        let above = pad + head_h + mat_h + 8.0 + 2.0 * LINE_H + series_h + 8.0 + BAR_H + 10.0;
        let whole = |caption| Layout { narrow, picture: (Vec2::new(0.0, inset), view), panel: None, caption };
        let place = |min: Vec2, width: f32, map_h: f32| {
            let x = min.x + pad;
            let mut y = min.y + pad;
            let head = (!narrow).then_some(Vec2::new(x, y));
            y += head_h;
            let mats = Vec2::new(x, y);
            y += mat_h + 8.0;
            let lines = Vec2::new(x, y);
            y += 2.0 * LINE_H;
            let series_at = Vec2::new(x, y + 8.0);
            y += series_h;
            y += 8.0;
            let play = (Vec2::new(min.x + width - pad - 30.0, y), Vec2::new(min.x + width - pad, y + BAR_H));
            let bar = (Vec2::new(x + 14.0, y + BAR_H * 0.5), Vec2::new(play.0.x - 74.0, y + BAR_H * 0.5));
            y += BAR_H + 10.0;
            let map_title = Vec2::new(x, y);
            y += LINE_H + 2.0;
            let map = (Vec2::new(x + 4.0, y), Vec2::new(min.x + width - pad - 4.0, y + map_h));
            Panel { min, max: Vec2::new(min.x + width, map.1.y + pad + 4.0), text, head, mats, lines, series: series_at, bar, play, map_title, map }
        };
        if narrow {
            // the panel below the picture, with the caption at its foot
            let caption_h = CAPTION_LINES as f32 * 19.0 + 8.0;
            let rest = above + LINE_H + 2.0 + 6.0 + caption_h;
            let map_h = ((view.x - 2.0 * pad - 8.0) * 0.4).clamp(100.0, 150.0);
            // (the picture keeps at least the top 44 %)
            let below = (view.y - rest - map_h).max(view.y * 0.44);
            let map_h = (view.y - below - rest).min(map_h);
            if map_h < 60.0 {
                return whole(Some((0.0, view.x, view.y - 10.0)));
            }
            let mut panel = place(Vec2::new(0.0, below), view.x, map_h);
            panel.max = view;
            return Layout { narrow, picture: (Vec2::new(0.0, inset), Vec2::new(view.x, below)), panel: Some(panel), caption: Some((0.0, view.x, view.y - 10.0)) };
        }
        let top = inset + 12.0;
        let map_h = ((width - 2.0 * pad - 8.0) * 0.68).min(view.y - 12.0 - top - above - LINE_H - 2.0 - pad - 4.0);
        if view.x < 600.0 || map_h < 90.0 {
            return whole(Some((0.0, view.x, view.y - 16.0)));
        }
        let panel = place(Vec2::new(view.x - width - 12.0, top), width, map_h);
        let picture = (Vec2::new(0.0, inset), Vec2::new(panel.min.x - 8.0, view.y));
        Layout { narrow, picture, panel: Some(panel), caption: Some((0.0, picture.1.x, view.y - 16.0)) }
    }

    /// The part of the canvas the scene's frame is fitted into: the picture's, less the
    /// heading at its top and the caption at its foot.
    fn frame_room(&self) -> (Vec2, Vec2) {
        let (min, max) = self.picture;
        let foot = if self.narrow { 6.0 } else { 52.0 };
        (min + Vec2::new(12.0, 52.0), max - Vec2::new(12.0, foot))
    }
}

/// The most lines the caption takes on a phone.
const CAPTION_LINES: usize = 3;

/// The names written before the panel's matrices.
const LEADS: [&str; 2] = ["$A =$", "$e^{At} =$"];

/// Their widths at a size of 1 (a width is in proportion to the size), worked out once:
/// the layout is decided several times a frame, at several sizes.
fn lead_widths() -> [f32; 2] {
    static WIDTHS: std::sync::OnceLock<[f32; 2]> = std::sync::OnceLock::new();
    *WIDTHS.get_or_init(|| LEADS.map(|name| text_width(name, 1.0)))
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

/// A matrix of numbers on the panel, as `hud_matrix` draws it, with a number too wide
/// for its cell (a large power of ten, set as 4 × 10²¹) drawn smaller, to fit: so that
/// the brackets stay where they are whatever the numbers.
fn matrix(f: &mut Frame, top_left: Vec2, rows: &[Vec<(String, Color)>], widest: usize, size: f32) -> Vec2 {
    let room = matrix_cell(widest, size).x - 2.0;
    let mut fitted = rows.to_vec();
    let mut smaller = Vec::new();
    for (i, row) in fitted.iter_mut().enumerate() {
        for (j, (text, color)) in row.iter_mut().enumerate() {
            // (only a power of ten can be too wide)
            if !text.contains('e') {
                continue;
            }
            let wide = width(f, text, size);
            if wide > room {
                smaller.push((i, j, std::mem::take(text), *color, size * room / wide));
            }
        }
    }
    let total = hud_matrix(f, top_left, &fitted, widest, size, pal::DIM);
    for (i, j, text, color, s) in smaller {
        let (lo, hi) = matrix_entry(top_left, widest, size, i, j);
        f.text_px((lo + hi) * 0.5, text, s, color, Align2::CENTER_CENTER);
    }
    total
}

fn inside_rect(p: Vec2, (min, max): (Vec2, Vec2)) -> bool {
    p.cmpge(min).all() && p.cmple(max).all()
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

/// A path's points kept to the box `lo..hi`. For a path of short steps, beginning in
/// view, whatever is moved lies outside the view (the box reaches a view's width beyond
/// it on every side): unlike cutting the path at the box, this keeps the number of
/// points the same as the path swings in and out of it.
fn within(points: &[DVec2], lo: DVec2, hi: DVec2) -> Vec<Vec3> {
    points.iter().map(|p| p.clamp(lo, hi).as_vec2().extend(0.0)).collect()
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
        // (the crossing has the bound itself for its coordinate across, not the sum of a
        // far corner and a step back from it, which for a corner 10²⁰ away is all rounding)
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

/// The points along the path of `x` under the flow of `a`, from time `from` to time
/// `to` in steps no longer than `step`: each the last carried on by e^(A h), so that
/// a long path costs one small matrix product a point.
fn path(a: DMat2, x: DVec2, from: f64, to: f64, step: f64) -> Vec<DVec2> {
    let n = (((to - from) / step.max(1e-6)).ceil().max(1.0) as usize).min(4000);
    let h = (to - from) / n as f64;
    let m = expm(a, h);
    let mut p = expm(a, from) * x;
    let mut out = Vec::with_capacity(n + 1);
    out.push(p);
    for _ in 0..n {
        p = m * p;
        out.push(p);
    }
    out
}

/// A path from t = 0 to `T_MAX` on a fixed grid of times, so that as t moves on, the
/// points already drawn stay where they are and only the end moves.
struct Track {
    /// The time between points, and the points.
    h: f64,
    points: Vec<DVec2>,
}

/// Points on a `Track`: a fiftieth of a radian apart for the fastest turning any A
/// allowed can do (eigenvalues up to 8 in size), and the same number whatever A is,
/// so that a path that changes shape keeps its points.
const TRACK_POINTS: usize = 1600;

impl Track {
    fn of(a: DMat2, x: DVec2) -> Track {
        let points = path(a, x, 0.0, T_MAX, T_MAX / TRACK_POINTS as f64);
        Track { h: T_MAX / (points.len() - 1).max(1) as f64, points }
    }

    /// The last grid point at or before the time t.
    fn before(&self, t: f64) -> usize {
        ((t / self.h).floor().max(0.0) as usize).min(self.points.len() - 1)
    }

    /// The path up to time t, ending at `end`, where it is then.
    fn upto(&self, t: f64, end: DVec2) -> Vec<DVec2> {
        let k = self.before(t);
        self.points[..=k].iter().copied().chain([end]).collect()
    }

    /// The path from time t, starting at `start`, where it is then.
    fn ahead(&self, t: f64, start: DVec2) -> Vec<DVec2> {
        let k = self.before(t);
        [start].into_iter().chain(self.points[k + 1..].iter().copied()).collect()
    }
}

/// The paths drawn, each 1600 steps of the flow, kept with what they were worked out
/// for, to the bit: those of (1, 0) and (0, 1), which depend on A alone, and the
/// start's, which depends on A and x₀. Playing moves only t along them, and dragging
/// x₀ leaves A as it was.
#[derive(Default)]
struct Paths {
    columns: Option<([u64; 4], [Track; 2])>,
    start: Option<([u64; 6], Track)>,
}

/// A matrix's entries, as bits.
fn bits(a: DMat2) -> [u64; 4] {
    a.to_cols_array().map(f64::to_bits)
}

impl Paths {
    fn columns(&mut self, a: DMat2) -> &[Track; 2] {
        let of = bits(a);
        if self.columns.as_ref().is_some_and(|c| c.0 != of) {
            self.columns = None;
        }
        &self.columns.get_or_insert_with(|| (of, [Track::of(a, DVec2::X), Track::of(a, DVec2::Y)])).1
    }

    fn start(&mut self, a: DMat2, x0: DVec2) -> &Track {
        let [p, q, r, s] = bits(a);
        let of = [p, q, r, s, x0.x.to_bits(), x0.y.to_bits()];
        if self.start.as_ref().is_some_and(|c| c.0 != of) {
            self.start = None;
        }
        &self.start.get_or_insert_with(|| (of, Track::of(a, x0))).1
    }
}

/// Where the names in the picture went in the last frame, and everything that decided
/// it.
///
/// Placing a name is a search: up to nine places beside its point, each measured
/// against everything drawn, hundreds of short segments of the paths among them. In
/// most frames nothing moves but the streaks, which the names do not keep clear of;
/// then the names go where they went before, and the room for them is not even
/// measured. What decides it (`names_of`) is all that the search reads, to the bit:
/// the setting drawn, what is shown, the eigenlines' sides, the camera, the canvas,
/// and where the frame remembers putting each name (`Frame::places`), which a name
/// that has just moved changes, so that it is placed afresh once more.
struct Names {
    of: Vec<u64>,
    placed: Vec<Placed>,
    /// The eigenlines' sides as the names left them (a new line is named at the end
    /// with more room, and keeps it).
    lines: [Option<DVec2>; 2],
}

/// A name as it was put: the arguments of `Frame::text_placed`.
struct Placed {
    name: &'static str,
    number: u32,
    from: Vec2,
    offset: Vec2,
    text: String,
    size: f32,
    color: Color,
}

/// How large on the screen (pixels) the velocity's arrow and the carried square (corner
/// to corner) must be to be named: smaller, and the name would sit on the thing itself
/// and on its neighbours' names.
const NAMED_FROM: f32 = 34.0;
const AREA_NAMED_FROM: f32 = 64.0;
/// The least distance on the screen (pixels) between two of the marks that show which
/// way the flow goes along an eigenline.
const CHEVRONS_APART: f32 = 22.0;
/// And how large the unit must be on the screen for either to be named at all.
const UNIT_FROM: f32 = 24.0;

/// The furthest off its line (pixels) an eigenline's name is put to keep clear of the
/// carried square.
const NAME_OFF_LINE: f32 = 46.0;

/// How far a shape on the screen (its corners in order) rises above the line through
/// `at` along `along`, on the side `up`, within `half` of `at` along the line: the most
/// of it, less than zero if it lies below the line there, and far below if it is not
/// there at all.
fn rises(shape: &[Vec2], at: Vec2, along: Vec2, up: Vec2, half: f32) -> f32 {
    let local = |p: Vec2| Vec2::new((p - at).dot(along), (p - at).dot(up));
    let mut most = f32::MIN;
    for k in 0..shape.len() {
        let (p, q) = (local(shape[k]), local(shape[(k + 1) % shape.len()]));
        let (lo, hi) = (p.x.min(q.x).max(-half), p.x.max(q.x).min(half));
        if lo > hi {
            continue;
        }
        // (the edge's height at the two ends of the part of it that is within reach)
        for s in [lo, hi] {
            let height = if (q.x - p.x).abs() < 1e-6 { p.y.max(q.y) } else { p.y + (q.y - p.y) * (s - p.x) / (q.x - p.x) };
            most = most.max(height);
        }
    }
    most
}

/// The names the picture can have, as the frame tells them apart.
const NAMES: [(&str, u32); 7] = [("x0", 0), ("at", 0), ("velocity", 0), ("area", 0), ("eigen", 0), ("eigen", 1), ("sum", 0)];

/// Everything the names' places depend on, as bits (see `Names`).
fn names_of(scene: &Expm, f: &Frame) -> Vec<u64> {
    let s = scene.shown();
    let mut of: Vec<u64> = s.numbers().map(f64::to_bits).to_vec();
    of.extend([scene.series as u64, scene.eigenlines as u64, scene.terms as u64, scene.terms_shown.to_bits()]);
    for line in scene.last_lines {
        of.extend(line.map_or([u64::MAX; 3], |v| [0, v.x.to_bits(), v.y.to_bits()]));
    }
    let cam = f.cam;
    let (lo, hi) = cam.flat_bounds();
    let floats = cam.view_proj.to_cols_array().into_iter().chain(cam.inv_view_proj.to_cols_array()).chain([lo.x, lo.y, hi.x, hi.y, cam.viewport.x, cam.viewport.y, f.px, f.top_inset]);
    of.extend(floats.chain(f.view_size().to_array()).map(|v| v.to_bits() as u64));
    for (name, number) in NAMES {
        of.extend(f.places.chosen(name, number).map_or([u64::MAX; 2], |p| [p.x.to_bits() as u64, p.y.to_bits() as u64]));
    }
    of
}

/// Whether the names go where they went in the last frame, `again` saying that nothing
/// they depend on has changed. (In the tests a run that places them afresh in every
/// frame is the yardstick for one that keeps them.)
fn keep_names(again: bool) -> bool {
    #[cfg(test)]
    let again = tests::kept(again);
    again
}

/// The room for the names (`Room`), or nothing in a frame that puts them where they
/// were (`Names`), where measuring it would be work for nothing.
struct Busy(Option<Room>);

impl Busy {
    fn wall(&mut self, min: Vec2, max: Vec2) {
        if let Some(room) = &mut self.0 {
            room.wall(min, max);
        }
    }

    fn line(&mut self, f: &Frame, a: Vec3, b: Vec3) {
        if let Some(room) = &mut self.0 {
            room.line(f, a, b);
        }
    }

    fn shape(&mut self, f: &Frame, corners: &[Vec3]) {
        if let Some(room) = &mut self.0 {
            room.shape(f, corners);
        }
    }
}

// ----- the scene -----------------------------------------------------------------------

/// What glides when a button changes it: the matrix, the time and the start.
#[derive(Clone, Copy, Debug, PartialEq)]
struct Setting {
    a: DMat2,
    t: f64,
    x0: DVec2,
}

impl Setting {
    fn numbers(&self) -> [f64; 7] {
        let a = self.a.to_cols_array();
        [a[0], a[1], a[2], a[3], self.t, self.x0.x, self.x0.y]
    }

    fn from_numbers(v: [f64; 7]) -> Setting {
        Setting { a: DMat2::from_cols_array(&[v[0], v[1], v[2], v[3]]), t: v[4], x0: DVec2::new(v[5], v[6]) }
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
        for i in 0..7 {
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

/// Seconds between one term of the series and the next, as an experiment counts them up.
const TERM_EVERY: f32 = 0.9;
/// The number of terms an experiment counts up to.
const COUNT_TO: usize = 12;

pub struct Expm {
    glide: Glide,
    playing: bool,
    /// Where playing stops.
    stop_at: f64,
    /// Play once the glide to a new setting has arrived.
    play_when_there: bool,
    series: bool,
    /// Terms of the series after I, and how many are drawn (gliding to it).
    terms: usize,
    terms_shown: f64,
    /// Counting the terms up: seconds until the next one.
    counting: Option<f32>,
    /// The tour of the map: seconds into it (it starts once the glide to its first
    /// stop has arrived).
    tour: Option<f32>,
    /// What a drag on the map moves A among, while one is going on.
    family: Option<Family>,
    particles: bool,
    eigenlines: bool,
    flow: Flow,
    /// The eigenline directions drawn last frame: a line's arrow keeps its side.
    last_lines: [Option<DVec2>; 2],
    /// The paths last worked out, where the names went in the last frame, and the
    /// caption's lines and the heading's sizes.
    paths: Paths,
    names: Option<Names>,
    caption: Option<Caption>,
    heading: Option<Heading>,
    /// The canvas last drawn on and how much of its top the application's buttons
    /// cover: they decide where the panel goes, and so where the picture is.
    canvas: Vec2,
    inset: f32,
    measured: bool,
    new_view: bool,
}

impl Expm {
    fn new(gfx: Option<&Gfx>) -> Self {
        let mut scene = Expm {
            glide: Glide::at(Setting { a: DMat2::IDENTITY, t: 0.0, x0: DVec2::ONE }),
            playing: false,
            stop_at: T_MAX,
            play_when_there: false,
            series: false,
            terms: 0,
            terms_shown: 0.0,
            counting: None,
            tour: None,
            family: None,
            particles: true,
            eigenlines: true,
            flow: Flow::new(gfx),
            last_lines: [None; 2],
            paths: Paths::default(),
            names: None,
            caption: None,
            heading: None,
            canvas: Vec2::new(1080.0, 1000.0),
            inset: 0.0,
            measured: false,
            new_view: false,
        };
        scene.set_up(0);
        // (opening, there is nothing to glide from, and the view starts as it should be)
        scene.glide = Glide::at(scene.glide.to);
        scene.new_view = false;
        scene
    }

    /// Set up experiment `k` of `TRIALS`: glide to its setting, then play it.
    fn set_up(&mut self, k: usize) {
        let trial = &TRIALS[k];
        self.glide.to = Setting { a: rows(trial.a), t: trial.t, x0: DVec2::from(trial.x0) };
        self.playing = false;
        self.play_when_there = trial.play_to.is_some();
        self.stop_at = trial.play_to.unwrap_or(T_MAX);
        self.series = trial.series;
        if trial.series {
            (self.terms, self.terms_shown, self.counting) = (0, 0.0, Some(TERM_EVERY));
        } else {
            self.counting = None;
        }
        self.tour = trial.tour.then_some(0.0);
        self.family = None;
        (self.particles, self.eigenlines) = (true, true);
    }

    /// Ask for a new matrix by a button or a typed number: the picture glides to it.
    fn aim_matrix(&mut self, a: DMat2) {
        self.glide.to.a = within_limits(a);
        self.tour = None;
    }

    /// The setting as drawn.
    fn shown(&self) -> Setting {
        self.glide.shown
    }

    /// Play from where t is (from 0, if it is at the end).
    fn play(&mut self) {
        self.tour = None;
        self.stop_at = T_MAX;
        if self.glide.to.t >= T_MAX - 1e-9 {
            self.glide.to.t = 0.0;
            self.play_when_there = true;
        } else {
            self.playing = true;
        }
    }

    /// Move everything on: `dt` seconds of the scene's time (playing, counting, the
    /// tour) and `real_dt` of the clock's (a change of setting glides by the clock, so
    /// that it is seen through even while the scene's time stands still).
    fn advance(&mut self, dt: f32, real_dt: f32) {
        self.glide.step(real_dt as f64);
        if self.play_when_there && self.glide.arrived() {
            (self.play_when_there, self.playing) = (false, true);
        }
        if self.playing {
            let t = (self.glide.to.t + dt as f64).min(self.stop_at);
            self.glide.now(|s| s.t = t);
            if t >= self.stop_at {
                self.playing = false;
            }
        }
        if let Some(left) = self.counting {
            if self.glide.arrived() {
                let left = left - dt;
                if left <= 0.0 {
                    self.terms = (self.terms + 1).min(N_MAX);
                    self.counting = (self.terms < COUNT_TO).then_some(left + TERM_EVERY);
                } else {
                    self.counting = Some(left);
                }
            }
        }
        // the drawn number of terms glides to the asked for one: a new arrow grows
        let to = self.terms as f64;
        if self.terms_shown != to {
            let k = 1.0 - (-(real_dt as f64) / 0.12).exp();
            self.terms_shown += (to - self.terms_shown) * k;
            if (to - self.terms_shown).abs() < 1e-3 {
                self.terms_shown = to;
            }
        }
        if let Some(s) = self.tour {
            if self.glide.arrived() {
                let s = s + dt;
                let (tau, det) = tour_at(s);
                self.glide.now(|g| g.a = canonical(tau, det));
                self.tour = (s < tour_length()).then_some(s);
            }
        }
    }

    /// The canonical view: the picture's frame in the middle of the part of the
    /// canvas it has to itself.
    fn the_view(&self) -> View {
        let lay = Layout::of(self.canvas, self.inset, self.series);
        let (min, max) = lay.frame_room();
        let room = (max - min).max(Vec2::splat(60.0));
        let (centre, half) = (FRAME_CENTRE, FRAME_HALF);
        let per_unit = (room.x / (2.0 * half.x)).min(room.y / (2.0 * half.y)).max(1e-3);
        // (screen y runs down, the scene's up)
        let off = ((min + max) * 0.5 - self.canvas * 0.5) / per_unit;
        View::Fit { center: Vec2::new(centre.x - off.x, centre.y + off.y), half: self.canvas.max(Vec2::ONE) * (0.5 / per_unit) }
    }

    /// The panel's handles and buttons: the time knob, the play button, the track and
    /// the dot on the map.
    fn panel_input(&mut self, f: &mut Frame, panel: &Panel) {
        let pointer = f.ptr.pos.map(|p| p / f.px);
        let clicked = f.clicked_canvas();
        if !f.handles.dragging() {
            self.family = None;
        }
        // the knob on the time track
        let (b0, b1) = panel.bar;
        let shown = self.shown();
        let knob = b0 + (b1 - b0) * (shown.t / T_MAX).clamp(0.0, 1.0) as f32;
        if let Some(mut w) = f.cam.pick_ground(knob * f.px).map(|p| p.truncate()) {
            if f.drag2(2, &mut w, pal::WHITE) {
                if let Some(p) = on_screen(f, w.extend(0.0)) {
                    let t = (((p.x - b0.x) / (b1.x - b0.x).max(1.0)) as f64 * T_MAX).clamp(0.0, T_MAX);
                    self.glide.now(|s| s.t = t);
                    (self.playing, self.play_when_there, self.tour) = (false, false, None);
                }
            }
        }
        if let Some(p) = pointer.filter(|_| clicked) {
            if inside_rect(p, panel.play) {
                if self.playing || self.play_when_there {
                    (self.playing, self.play_when_there) = (false, false);
                } else {
                    self.play();
                }
            } else if inside_rect(p, (b0 - Vec2::new(8.0, 12.0), b1 + Vec2::new(8.0, 12.0))) {
                // a click on the track: t glides there
                self.glide.to.t = (((p.x - b0.x) / (b1.x - b0.x).max(1.0)) as f64 * T_MAX).clamp(0.0, T_MAX);
                (self.playing, self.play_when_there, self.tour) = (false, false, None);
            } else if inside_rect(p, panel.map) {
                // a click on the map: A glides to that place, within the family of the matrix asked for
                let (tau, det) = map_value(panel.map, p);
                self.aim_matrix(family_for(self.glide.to.a).at(tau, det));
            }
        }
        // the dot on the map, at the drawn matrix's place
        let a = shown.a;
        let dot = map_point(panel.map, trace(a), a.determinant());
        if let Some(mut w) = f.cam.pick_ground(dot * f.px).map(|p| p.truncate()) {
            if f.drag2(3, &mut w, pal::WHITE) {
                if let Some(p) = on_screen(f, w.extend(0.0)) {
                    let (tau, det) = map_value(panel.map, p);
                    // (from where it is drawn: a drag stops any glide where it has got to)
                    let family = *self.family.get_or_insert_with(|| family_for(a));
                    let m = family.at(tau, det);
                    self.glide.now(|s| s.a = m);
                    self.tour = None;
                }
            }
        }
    }
}

/// Where on the map rectangle `(min, max)` a trace and determinant are (kept to its edges).
fn map_point((min, max): (Vec2, Vec2), tau: f64, det: f64) -> Vec2 {
    let u = ((tau - MAP_TR[0]) / (MAP_TR[1] - MAP_TR[0])).clamp(0.0, 1.0) as f32;
    let v = ((det - MAP_DET[0]) / (MAP_DET[1] - MAP_DET[0])).clamp(0.0, 1.0) as f32;
    Vec2::new(min.x + (max.x - min.x) * u, max.y - (max.y - min.y) * v)
}

/// The trace and determinant at a point of the map (kept to its edges).
fn map_value((min, max): (Vec2, Vec2), p: Vec2) -> (f64, f64) {
    let u = ((p.x - min.x) / (max.x - min.x).max(1.0)).clamp(0.0, 1.0) as f64;
    let v = ((max.y - p.y) / (max.y - min.y).max(1.0)).clamp(0.0, 1.0) as f64;
    (MAP_TR[0] + (MAP_TR[1] - MAP_TR[0]) * u, MAP_DET[0] + (MAP_DET[1] - MAP_DET[0]) * v)
}

/// The four numbers of a 2×2 matrix to drag or type, within ±`ENTRY_MAX`. Returns
/// (changed, typed): a number typed in glides the picture to it, one dragged moves it
/// at once.
fn matrix_edit(ui: &mut egui::Ui, m: &mut DMat2) -> (bool, bool) {
    let (mut changed, mut typed) = (false, false);
    let mut a = [[m.x_axis.x, m.y_axis.x], [m.x_axis.y, m.y_axis.y]];
    egui::Grid::new("expm_a").spacing([6.0, 4.0]).show(ui, |ui| {
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

/// The label of one of egui's own widgets (a button, a checkbox), typeset: they set
/// their labels as plain text.
fn typeset_label(ui: &egui::Ui, text: &str) -> egui::WidgetText {
    crate::typeset::seen(text);
    ui::label_text(ui, text, 12.5, None, false)
}

impl Scene for Expm {
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
        let moving = size(self.shown().a) > 1e-12;
        (self.particles && moving) || self.playing || self.play_when_there || !self.glide.arrived() || self.counting.is_some() || self.tour.is_some() || self.terms_shown != self.terms as f64
    }

    fn ui(&mut self, ui: &mut egui::Ui, _: &Gfx) {
        ui::note(ui, r"Let the velocity of a point $x$ be $Ax$:");
        ui::equation(ui, r"\frac{\mathrm{d}x}{\mathrm{d}t} = Ax");
        ui::note(ui, r"Then the whole plane flows (the faint streaks). Following the flow from a start $x_0$ for a time $t$ ends at $e^{At} x_0$, and $e^{At}$ is a matrix: the map that carries every point along for that time at once. Its columns are where $(1, 0)$ and $(0, 1)$ have flowed to; the yellow square shows what it does to areas. The eigenvalues of $A$ decide what kind of flow it is; the small map with the matrices places every $2 \times 2$ matrix by its trace and determinant.");

        ui::section(ui, "Things to try");
        for (k, trial) in TRIALS.iter().enumerate() {
            if ui::experiment(ui, trial.label, trial.look_for) {
                self.set_up(k);
            }
        }

        ui::section(ui, "Time");
        ui.horizontal(|ui| {
            let busy = self.playing || self.play_when_there;
            if ui.button(if busy { "❚❚ pause" } else { "▶ play" }).clicked() {
                if busy {
                    (self.playing, self.play_when_there) = (false, false);
                } else {
                    self.play();
                }
            }
            if ui.button(typeset_label(ui, "back to $t = 0$")).clicked() {
                self.glide.to.t = 0.0;
                (self.playing, self.play_when_there, self.tour) = (false, false, None);
            }
        });
        let mut t = self.glide.to.t as f32;
        if ui::slider(ui, "time $t$", &mut t, 0.0..=T_MAX as f32) {
            self.glide.now(|s| s.t = t as f64);
            (self.playing, self.play_when_there, self.tour) = (false, false, None);
        }
        ui::note(ui, "How long the flow is followed for. Play runs it on at one unit of time a second; the knob under the matrices on the picture does the same as this slider.");

        ui::section(ui, "Matrix $A$");
        let names: Vec<&str> = PRESETS.iter().map(|p| p.0).collect();
        if let Some(i) = ui::buttons(ui, &names) {
            self.aim_matrix(rows(PRESETS[i].1));
        }
        let mut a = self.glide.to.a;
        let (changed, typed) = matrix_edit(ui, &mut a);
        if changed {
            if typed {
                self.aim_matrix(a);
            } else {
                self.glide.now(|s| s.a = a);
                self.tour = None;
            }
        }
        if let Some(p) = PRESETS.iter().find(|p| rows(p.1) == self.glide.to.a) {
            ui::note(ui, p.2);
        }
        ui::note(ui, "Or drag the dot on the map of all flows: $A$ changes to the trace and determinant of the place you drag it to, keeping as much of its shape as it can.");

        ui::section(ui, "The series");
        ui::equation(ui, r"I + At + \frac{(At)^2}{2!} + \cdots + \frac{(At)^N}{N!}");
        ui.checkbox(&mut self.series, typeset_label(ui, "show this sum acting on $x_0$"));
        if ui::slider_int(ui, "$N$, the last power", &mut self.terms, 0..=N_MAX) {
            self.counting = None;
            self.series = true;
        }
        ui::note(ui, r"The red arrows are the terms $x_0, tAx_0, (tA)^2 x_0/2!, \ldots$ laid end to end: where they end is the sum up to $N$, beside $e^{At} x_0$, where the flow takes $x_0$.");

        ui::section(ui, "Show");
        ui.checkbox(&mut self.particles, "the flow, as moving streaks");
        ui.checkbox(&mut self.eigenlines, "the eigenlines");

        ui::section(ui, "Reading the picture");
        let mut key = vec![
            (streak_colour(), "faint streaks: the flow; each point moves with the velocity $Ax$, its head bright and its tail fading"),
            (pal::ORANGE, r"orange and purple, dashed: the eigenlines (when $\lambda$ is real), with the flow's direction along each; the arrow on each is a unit vector carried for the time $t$, $e^{\lambda t}$ times as long. Solid: $\lambda = 0$, a line of points at rest"),
            (pal::YELLOW, r"yellow: the unit square (dashed) carried for the time $t$, its image under $e^{At}$; its area is $\det e^{At}$"),
            (pal::PINK, "pink: the path of $(1, 0)$; where it has got to (the dot) is the first column of $e^{At}$"),
            (pal::GREEN, "green: the path of $(0, 1)$, and the second column"),
            (pal::WHITE, r"white: the start $x_0$ (drag it), its path, and $e^{At} x_0$ where it has got to; dots at $t = 1, 2, 3, \ldots$; faint: the path still ahead"),
            (pal::CYAN, "cyan arrow: the velocity $Ax$ there"),
        ];
        if self.series {
            key.push((pal::RED, "red: the terms of the series, end to end, and their sum"));
        }
        key.push((pal::TEXT, r"the map: every kind of flow, by $\tr A$ and $\det A$; the dot is $A$ (drag it)"));
        ui::key(ui, &key);

        ui::section(ui, "Words");
        ui::term(ui, "Flow:", r"what $\mathrm{d}x/\mathrm{d}t = Ax$ does to the plane: at every moment each point moves with the velocity $A$ times where it is.");
        ui::term(ui, "$e^{At}$:", r"the matrix that carries every point along the flow for a time $t$: $x(t) = e^{At} x_0$. It is $I + At + (At)^2/2! + (At)^3/3! + \cdots$, and flowing for $s$ and then for $t$ is flowing for $s + t$: $e^{As} e^{At} = e^{A(s + t)}$.");
        ui::term(ui, "Eigenline:", r"a line through $0$ along an eigenvector $v$ of $A$, where $Av = \lambda v$. A point on it stays on it and is $e^{\lambda t}$ times as far out at time $t$: it grows if $\lambda > 0$, shrinks if $\lambda < 0$.");
        ui::term(ui, r"Complex $\lambda = a \pm bi$:", "no line stays put: the flow turns at the rate $b$ and grows ($a > 0$) or shrinks ($a < 0$) at the rate $a$.");
        ui::term(ui, "Trace, determinant:", r"$\tr A$, the sum of the diagonal, is $\lambda_1 + \lambda_2$; $\det A$ is $\lambda_1 \lambda_2$. Areas grow at the rate $\tr A$: $\det e^{At} = e^{t \tr A}$.");
        ui::term(ui, "Sink, source, saddle:", "everything flows in to $0$; out from it; in along one eigenline and out along the other. A spiral turns as it goes; a centre goes round closed loops.");
        ui::term(ui, "Flows and powers:", r"the dots on the white path, at $t = 1, 2, 3, \ldots$, are the powers of the matrix $e^A$ applied to $x_0$. Its eigenvalues $e^\lambda$ are smaller than 1 in size exactly when $\lambda$ has a negative real part: that is the test for shrinking here, where for powers of $A$ it is $|\lambda| < 1$.");

        ui::section(ui, "Numbers");
        let s = self.shown();
        let (e, k) = (eigen(s.a), kind(s.a));
        let et = expm(s.a, s.t);
        ui::readout(ui, "kind", k.name().trim_start_matches("a ").trim_start_matches("an "));
        // (the values as formulas, every one alike: eigenvalues may be complex, and the
        // sizes may be written in powers of ten)
        ui::readout(ui, "eigenvalues", eigen_text(e, k, false));
        ui::readout(ui, r"$\tr A$, $\det A$", format!("${}, {}$", num(trace(s.a), 2), num(s.a.determinant(), 2)));
        ui::readout(ui, "$t$", format!("${}$", num(s.t, 2)));
        ui::readout(ui, r"$\det e^{At}$", format!("${}$", short(et.determinant())));
        ui::readout(ui, r"$e^{t \tr A}$", format!("${}$", short((s.t * trace(s.a)).exp())));
        ui::readout(ui, r"$\lVert e^{At} x_0 \rVert$", format!("${}$", short((et * s.x0).length())));
        if self.series {
            let sum = partial_sum(s.a, s.t, self.terms) * s.x0;
            ui::readout(ui, &format!("sum to $N = {}$, away from $e^{{At}} x_0$", self.terms), format!("${}$", distance((sum - et * s.x0).length())));
        }
        ui::see_also(ui, &[("powers", "powers of a matrix: steps instead of a flow"), ("eigen", "finding eigenvalues"), ("field", "flows that are not linear"), ("lyapunov", "how fast a nudge grows, in a deep network")]);
    }

    fn frame(&mut self, f: &mut Frame) {
        // the panel's place follows the canvas, and the picture's place follows the panel's
        let view = f.view_size();
        let lay = Layout::of(view, f.top_inset, self.series);
        let before = Layout::of(self.canvas, self.inset, self.series);
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

        let pointer = f.ptr.pos.map(|p| p / f.px);
        if let Some(panel) = lay.panel.as_ref().filter(|_| !covered) {
            if pointer.is_some_and(|p| inside_rect(p, (panel.min, panel.max))) {
                // the panel is for its own handles and buttons: no moving the view from
                // there, and two quick clicks are two clicks
                f.handles.capture();
                f.handles.claim_clicks();
            }
            self.panel_input(f, panel);
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
            // (the sliver left beside the controls is the flow alone: its names and
            // numbers would only be written over one another)
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
            self.panel(f, p);
        }
    }
}

impl Expm {
    /// Everything drawn in the scene's own coordinates, and the names written beside it.
    /// Returns the boxes of the heading and of the caption, which are written over it.
    fn picture(&mut self, f: &mut Frame, lay: &Layout) -> [Option<(Vec2, Vec2)>; 2] {
        let (lo, hi) = f.cam.flat_bounds();
        let wpp = (hi.y - lo.y) / f.view_size().y.max(1.0);
        // what is drawn is cut off a view's width beyond the view, whatever the flow does
        let span = (hi - lo).as_dvec2();
        let (box_lo, box_hi) = (lo.as_dvec2() - span, hi.as_dvec2() + span);

        // the start, which can be dragged (unless it lies under the panel)
        let mut x0 = self.shown().x0.as_vec2();
        let under_panel = lay.panel.as_ref().is_some_and(|p| on_screen(f, x0.extend(0.0)).is_some_and(|q| inside_rect(q, (p.min, p.max))));
        if (!under_panel || f.handles.dragging()) && f.drag2(1, &mut x0, pal::WHITE) {
            let x = x0.as_dvec2().clamp(DVec2::splat(-X0_MAX), DVec2::splat(X0_MAX));
            self.glide.now(|s| s.x0 = x);
        }

        let Setting { a, t, x0 } = self.shown();
        let e = expm(a, t);
        let (eig, k) = (eigen(a), kind(a));

        // the streaks, over all of the view (the panel covers only a corner of a wide one)
        if self.particles && k != Kind::Still {
            self.flow.advance(a, f.dt, lo, hi, streak_colour(), Some(f.gfx));
            if let Some(trails) = &mut self.flow.trails {
                f.trails(trails);
            }
        }

        // (the batches of the parts, made in the same order every frame, those that may
        // be empty last)
        for part in [Part::Start, Part::Columns, Part::Eigen, Part::Series] {
            lines_for(f, part);
        }
        // The names go where they went in the last frame if nothing they depend on has
        // changed (`Names`); only otherwise is the room for them measured and searched.
        let names_of = names_of(self, f);
        let again = keep_names(self.names.as_ref().is_some_and(|n| n.of == names_of));
        let mut room = Busy((!again).then(|| Room::new(f)));
        if let Some(room) = &mut room.0 {
            keep_ticks_clear(f, room);
        }
        if let Some(p) = &lay.panel {
            room.wall(p.min - Vec2::splat(4.0), p.max + Vec2::splat(4.0));
        }
        // the heading, at the top of the picture
        let heading = self.heading(f, lay);
        room.wall(heading.0, heading.1);
        let mut plates = [Some(heading), None];
        if let Some((left, right, bottom)) = lay.caption {
            let parts = if lay.narrow {
                ["streaks: the flow", "dashed: the unit square at $t = 0$", "pink, green: the paths of $(1, 0)$, $(0, 1)$", r"white dots: $t = 1, 2, 3, \ldots$"]
            } else {
                [r"streaks: the flow, $\mathrm{d}x/\mathrm{d}t = Ax$", "dashed square: the unit square at $t = 0$", "pink and green: the paths of $(1, 0)$ and $(0, 1)$, ending at the columns of $e^{At}$", r"dots on the white path: $t = 1, 2, 3, \ldots$"]
            };
            if let Some(plate) = caption(f, &parts, lay.narrow, (left, right), bottom, &mut self.caption) {
                room.wall(plate.0, plate.1);
                plates[1] = Some(plate);
            }
        }

        // the eigenlines, with the flow's direction along them, and the unit vector
        // on each carried for the time t
        let reach = (hi - lo).length() as f64;
        let mut lines: Vec<(f64, DVec2, Color)> = match eig {
            Eigen::Real { l1, l2, lines: Lines::Two(v1, v2) } => vec![(l1, v1, pal::ORANGE), (l2, v2, pal::PURPLE)],
            Eigen::Real { l1, lines: Lines::One(v), .. } => vec![(l1, v, pal::ORANGE)],
            _ => Vec::new(),
        };
        // (a line's arrow, and its name, stay on the side they were on as the matrix
        // changes; for a line new to the picture, the side is chosen with its name)
        let fresh = [0, 1].map(|i| self.last_lines[i].is_none());
        for (i, (_, v, _)) in lines.iter_mut().enumerate() {
            let was = self.last_lines[i].unwrap_or(DVec2::new(1.0, 1e-6));
            if was.dot(*v) < 0.0 {
                *v = -*v;
            }
            self.last_lines[i] = Some(*v);
        }
        for i in lines.len()..2 {
            self.last_lines[i] = None;
        }
        if self.eigenlines {
            for &(l, v, c) in &lines {
                if let Some((p, q)) = clip(-v * reach, v * reach, box_lo, box_hi) {
                    if l.abs() < 1e-9 * size(a).max(1e-12) {
                        // a line of rest points: solid
                        lines_for(f, Part::Eigen).segment(p.as_vec2().extend(0.0), q.as_vec2().extend(0.0), 2.0, c.a(0.8));
                    } else {
                        f.dashed(p.as_vec2(), q.as_vec2(), 1.6, c.a(0.85));
                    }
                    room.line(f, p.as_vec2().extend(0.0), q.as_vec2().extend(0.0));
                }
                // which way the flow goes along it: chevrons, out if λ > 0 and in if λ < 0
                // (smaller as λ nears 0, where they turn round)
                if l.abs() >= 1e-9 * size(a).max(1e-12) {
                    let grown = (l.abs() / 0.25).min(1.0) as f32;
                    // (further apart in a view zoomed out, where they would lie on one another)
                    let spread = (CHEVRONS_APART * wpp / 0.7).max(1.0) as f64;
                    for at in [0.55 * spread, 1.25 * spread, 2.1 * spread] {
                        for side in [-1.0, 1.0] {
                            let p = v * (at * side);
                            let dir = v * (side * l.signum());
                            chevron(f, p.as_vec2(), dir.as_vec2(), wpp * grown, c);
                        }
                    }
                }
                // the unit vector along it, carried for the time t: e^(λt) times as long
                let tip = v * (l * t).exp();
                arrow_in(f, Part::Eigen, DVec2::ZERO, tip, box_lo, box_hi, 2.6, c);
            }
        }

        // the unit square at the start, and carried for the time t
        let unit = [DVec2::ZERO, DVec2::X, DVec2::ONE, DVec2::Y];
        for k in 0..4 {
            f.dashed(unit[k].as_vec2(), unit[(k + 1) % 4].as_vec2(), 1.2, pal::YELLOW.a(0.45));
        }
        let image: Vec<DVec2> = unit.iter().map(|p| e * *p).collect();
        let cut = clip_polygon(&image, box_lo, box_hi);
        for i in 1..cut.len().saturating_sub(1) {
            f.dl.tris_alpha.tri(cut[0].as_vec2().extend(0.0), cut[i].as_vec2().extend(0.0), cut[i + 1].as_vec2().extend(0.0), pal::YELLOW.a(0.2));
        }
        for run in runs(&[image[0], image[1], image[2], image[3], image[0]], box_lo, box_hi) {
            room.shape(f, &[image[0], image[1], image[2], image[3]].map(|p| p.as_vec2().extend(0.0)));
            for pair in run.windows(2) {
                room.line(f, pair[0].extend(0.0), pair[1].extend(0.0));
            }
            f.path(run, 1.6, pal::YELLOW.a(0.85));
        }

        // the paths of (1, 0) and (0, 1): where they have got to are the columns
        for (track, (start, c)) in self.paths.columns(a).iter().zip([(DVec2::X, pal::PINK), (DVec2::Y, pal::GREEN)]) {
            let run = within(&track.upto(t, e * start), box_lo, box_hi);
            lines_for(f, Part::Columns).polyline(run, 1.3, c.a(0.55));
            let end = e * start;
            if inside_box(end, box_lo, box_hi) {
                f.dot(end.as_vec2(), 8.0, c);
            }
        }

        // the start's path so far, the path ahead (faint), and where it is at whole times
        let at = e * x0;
        let track = self.paths.start(a, x0);
        let travelled = within(&track.upto(t, at), box_lo, box_hi);
        if room.0.is_some() {
            for pair in travelled.windows(6).step_by(5) {
                room.line(f, pair[0], pair[5]);
            }
        }
        lines_for(f, Part::Start).polyline(travelled, 2.4, pal::WHITE.a(0.9));
        // (faint rather than dashed: dashes measured from a start that moves would
        // slide along it)
        lines_for(f, Part::Start).polyline(within(&track.ahead(t, at), box_lo, box_hi), 1.2, pal::WHITE.a(0.3));
        let whole = expm(a, 1.0);
        let mut p = x0;
        for _ in 1..=(t.floor() as usize).min(6) {
            p = whole * p;
            if inside_box(p, box_lo, box_hi) {
                // (a batch of their own: they come one at a time)
                f.dl.points_with(PointKind::Disc, PointStyle { outline_px: 1.25 + 1e-4, ..Default::default() }).push(p.as_vec2().extend(0.0), 6.0, pal::WHITE.a(0.8));
            }
        }
        // the velocity there (not while the series is shown: its first term is the
        // velocity at the start, and the picture is busy enough)
        let velocity = if self.series { DVec2::ZERO } else { a * at };
        arrow_in(f, Part::Fixed, at, at + velocity, box_lo, box_hi, 2.4, pal::CYAN);
        if inside_box(at, box_lo, box_hi) {
            f.dot(at.as_vec2(), 11.0, pal::WHITE);
        }
        room.line(f, at.as_vec2().extend(0.0), (at + velocity).as_vec2().extend(0.0));

        // the series: its terms end to end, the last one growing in
        let mut sum_at = None;
        if self.series {
            let parts = terms(a, t, x0, N_MAX);
            let mut end = x0;
            let shown = self.terms_shown.clamp(0.0, N_MAX as f64);
            for (k, term) in parts.iter().enumerate().skip(1) {
                let share = (shown - (k - 1) as f64).clamp(0.0, 1.0);
                if share <= 0.0 {
                    break;
                }
                let next = end + *term * share;
                arrow_in(f, Part::Series, end, next, box_lo, box_hi, 2.0, pal::RED);
                room.line(f, end.as_vec2().extend(0.0), next.as_vec2().extend(0.0));
                end = next;
            }
            if inside_box(end, box_lo, box_hi) {
                f.dot(end.as_vec2(), 9.0, pal::RED);
            }
            sum_at = Some(end);
        }

        // the names: where they went in the last frame, if nothing they depend on has
        // changed since
        let Some(mut room) = room.0 else {
            if let Some(names) = &self.names {
                for p in &names.placed {
                    f.text_placed(p.name, p.number, p.from, p.offset, p.text.as_str(), p.size, p.color);
                }
                self.last_lines = names.lines;
            }
            return plates;
        };
        // (each name noted as it is placed: where it went, from which point, as the
        // frame was told; none kept if one is not of that kind)
        let mut placed = Some(Vec::new());
        let mut name = |f: &mut Frame, room: &mut Room, (name, number): (&'static str, u32), at: Vec3, text: &str, size: f32, color: Color, towards: Vec2, gap: f32| {
            let drawn = room.label(f, Tag(name, number), at, text, size, color, towards, gap, false);
            match (drawn, on_screen(f, at), f.places.chosen(name, number)) {
                (true, Some(from), Some(offset)) => {
                    if let Some(list) = &mut placed {
                        list.push(Placed { name, number, from, offset, text: text.to_string(), size, color });
                    }
                }
                _ => placed = None,
            }
        };
        let away = |p: DVec2| Vec2::new(p.x as f32, -(p.y as f32));
        // (only what is in view is named)
        let seen = |f: &Frame, p: DVec2| p.is_finite() && on_screen(f, p.as_vec2().extend(0.0)).is_some_and(|q| inside_rect(q, (Vec2::ZERO, f.view_size())));
        if seen(f, x0) {
            name(f, &mut room, ("x0", 0), x0.as_vec2().extend(0.0), "$x_0$", 16.0, pal::TEXT, away(x0), 10.0);
        }
        if seen(f, at) {
            name(f, &mut room, ("at", 0), at.as_vec2().extend(0.0), "$e^{At} x_0$", 15.0, pal::WHITE, away(at) - away(velocity) * 0.3, 10.0);
        }
        // how far apart two points of the scene are on the screen
        let apart_px = |f: &Frame, p: DVec2, q: DVec2| on_screen(f, p.as_vec2().extend(0.0)).zip(on_screen(f, q.as_vec2().extend(0.0))).map_or(0.0, |(p, q)| p.distance(q));
        // (beside the arrow rather than at its tip, where it may meet other things; and
        // not beside an arrow too short to be seen as one: far out in a zoomed-out view,
        // or where the flow has all but stopped)
        let mid = at + velocity * 0.6;
        // (in a view zoomed far out, everything about the start is in one spot: only
        // the points themselves are named there)
        let roomy = apart_px(f, DVec2::ZERO, DVec2::X) >= UNIT_FROM;
        if roomy && seen(f, mid) && apart_px(f, at, at + velocity) >= NAMED_FROM {
            let across = away(velocity.perp());
            let towards = if across.y < 0.0 { -across } else { across };
            name(f, &mut room, ("velocity", 0), mid.as_vec2().extend(0.0), "velocity $Ax$", 13.0, pal::CYAN, towards, 6.0);
        }
        // the area, in the middle of the square or beside it
        let area = format!("area ${}$", short(e.determinant()));
        let centre = (image[0] + image[2]) * 0.5;
        // (not in a square too small on the screen to hold it)
        if roomy && seen(f, centre) && apart_px(f, image[0], image[2]).max(apart_px(f, image[1], image[3])) >= AREA_NAMED_FROM {
            name(f, &mut room, ("area", 0), centre.as_vec2().extend(0.0), &area, 13.0, pal::YELLOW, Vec2::ZERO, 0.0);
        }
        if self.eigenlines {
            // each line named near where it leaves the picture, at the end its arrow
            // points to (a single place, which moves smoothly as the line turns)
            let (min, max) = lay.frame_room();
            // where the name of the line along `v` goes: near where it leaves the
            // picture at that end, and the side of the line that is up on the screen
            let end_of = |f: &Frame, v: DVec2| -> Option<(Vec2, Vec2, Vec2)> {
                let (o, tip) = (on_screen(f, Vec3::ZERO)?, on_screen(f, v.as_vec2().extend(0.0))?);
                let d = (tip - o).normalize_or_zero().as_dvec2();
                let (_, end) = clip(o.as_dvec2() - d * 1e5, o.as_dvec2() + d * 1e5, min.as_dvec2(), max.as_dvec2())?;
                let up = if d.perp().y < 0.0 { d.perp() } else { -d.perp() };
                Some(((end - d * 28.0).as_vec2(), up.as_vec2(), d.as_vec2()))
            };
            // (the carried square on the screen: it stretches out along a line that grows,
            // and the line's name keeps above it)
            let square: Vec<Vec2> = image.iter().filter_map(|p| on_screen(f, p.as_vec2().extend(0.0))).collect();
            for (i, (l, v, c)) in lines.iter().enumerate() {
                let text = if l.abs() < 1e-9 * size(a).max(1e-12) { r"$\lambda = 0$: at rest".to_string() } else { format!(r"$\lambda = {}$", num(*l, 2)) };
                let mut v = *v;
                if fresh[i] {
                    // a new line is named at whichever end has more room (and keeps it)
                    let half = Vec2::new(text_width(&text, 13.0) * 0.5 + 4.0, 7.5);
                    let clear = |f: &Frame, v: DVec2| end_of(f, v).map_or(f32::MIN, |(at, up, _)| room.room_for(at, at + up * (8.0 + half.y), half));
                    if clear(f, -v) > clear(f, v) {
                        v = -v;
                        self.last_lines[i] = Some(v);
                    }
                }
                let Some((at, up, along)) = end_of(f, v) else { continue };
                if let Some(w) = f.cam.pick_ground(at * f.px) {
                    // beside the line: just off it, or above the square where the square
                    // rises over the line there (if that is not too far off to be the
                    // line's name still)
                    let half = text_width(&text, 13.0) * 0.5 + 8.0;
                    let rise = rises(&square, at, along, up, half) + 6.0;
                    let gap = if rise <= NAME_OFF_LINE { rise.max(8.0) } else { 8.0 };
                    name(f, &mut room, ("eigen", i as u32), w, &text, 13.0, *c, up, gap);
                }
            }
        }
        if let Some(end) = sum_at {
            // (named while it is apart from eᴬᵗx₀: once there, that name is its name; and
            // not before the first term, when it is x₀ itself)
            let apart = on_screen(f, end.as_vec2().extend(0.0)).zip(on_screen(f, at.as_vec2().extend(0.0))).is_some_and(|(p, q)| p.distance(q) > 24.0);
            if seen(f, end) && apart && self.terms > 0 {
                let text = format!("sum to $N = {}$", self.terms);
                name(f, &mut room, ("sum", 0), end.as_vec2().extend(0.0), &text, 13.0, pal::RED, away(end - at), 8.0);
            }
        }
        self.names = placed.map(|placed| Names { of: names_of, placed, lines: self.last_lines });
        plates
    }

    /// The heading at the top of the picture: the kind of flow and its eigenvalues, and
    /// in a line under it what that means. Returns the box it takes.
    fn heading(&mut self, f: &mut Frame, lay: &Layout) -> (Vec2, Vec2) {
        let a = self.shown().a;
        let (e, k) = (eigen(a), kind(a));
        let at = lay.picture.0 + Vec2::new(16.0, 14.0);
        let first = format!("{}: {}", k.name(), eigen_text(e, k, true));
        let second = k.says();
        let room = (lay.picture.1.x - lay.picture.0.x - 32.0).max(40.0);
        // (the sizes chosen for these lines in this room before, if they were)
        let kept = self.heading.as_ref().filter(|h| (h.of.0.as_str(), h.of.1, h.of.2) == (first.as_str(), second, room)).map(|h| h.sizes);
        let (size, small, wide) = match kept {
            Some(sizes) => sizes,
            None => match heading_sizes(&first, second, room, |t, s| measured(f, t, s)) {
                Some(sizes) => {
                    self.heading = Some(Heading { of: (first.clone(), second, room), sizes });
                    sizes
                }
                // (the formulas' fonts not in place yet: the first frame)
                None => heading_sizes(&first, second, room, |t, s| Some(width(f, t, s))).unwrap_or((13.0, 11.0, room)),
            },
        };
        f.caption_px(at, first.as_str(), size, pal::TEXT, Align2::LEFT_TOP);
        f.caption_px(at + Vec2::new(0.0, size + 8.0), second, small, pal::DIM, Align2::LEFT_TOP);
        (at - Vec2::splat(6.0), at + Vec2::new(wide + 12.0, size + small + 16.0))
    }

    /// The screen panel: A and e^(At) as matrices, what has become of areas and of the
    /// eigenlines, the series' sum, the time track, and the map of all flows.
    fn panel(&self, f: &mut Frame, p: &Panel) {
        hud_panel(f, p.min, p.max);
        let Setting { a, t, .. } = self.shown();
        let e = expm(a, t);
        let (eig, k) = (eigen(a), kind(a));
        let s = p.text;
        if let Some(head) = p.head {
            f.text_px(head, format!("$e^{{At}}$: the flow for the time $t = {}$", num(t, 2)), 13.0, pal::TEXT, Align2::LEFT_TOP);
        }

        // A and e^(At), the columns of e^(At) in the colours of the paths they end
        let entries = |m: DMat2, cols: [Color; 2]| -> Vec<Vec<(String, Color)>> { (0..2).map(|i| (0..2).map(|j| (cell(m.col(j)[i]), cols[j])).collect()).collect() };
        let a_rows = entries(a, [pal::TEXT, pal::TEXT]);
        let e_rows = entries(e, [pal::PINK, pal::GREEN]);
        let widest = CELL;
        let msize = matrix_size(2, 2, widest, s);
        let mid = p.mats.y + msize.y * 0.5;
        let lead = lead_widths();
        let mut x = p.mats.x;
        f.text_px(Vec2::new(x, mid), LEADS[0], s, pal::TEXT, Align2::LEFT_CENTER);
        x += lead[0] * s + 8.0;
        x += matrix(f, Vec2::new(x, p.mats.y), &a_rows, widest, s).x + 12.0;
        f.text_px(Vec2::new(x, mid), LEADS[1], s, pal::TEXT, Align2::LEFT_CENTER);
        x += lead[1] * s + 8.0;
        matrix(f, Vec2::new(x, p.mats.y), &e_rows, widest, s);

        // areas, and the eigenlines
        let small = 12.0;
        f.text_px(p.lines, format!(r"area: $\det e^{{At}} = e^{{t \tr A}} = {}$", short(e.determinant())), small, pal::YELLOW, Align2::LEFT_TOP);
        let y = p.lines.y + LINE_H;
        // (the numbers in columns of their own: the width of text is only estimated)
        let (one, two) = (p.lines.x + 128.0, p.lines.x + 200.0);
        match eig {
            Eigen::Real { l1, l2, lines } => {
                let name = if matches!(lines, Lines::Two(..)) { r"eigenlines: $e^{\lambda t} =$" } else { r"eigenline: $e^{\lambda t} =$" };
                f.text_px(Vec2::new(p.lines.x, y), name, small, pal::DIM, Align2::LEFT_TOP);
                f.text_px(Vec2::new(one, y), format!("${}$", short((l1 * t).exp())), small, pal::ORANGE, Align2::LEFT_TOP);
                match lines {
                    Lines::Two(..) => f.text_px(Vec2::new(two, y), format!("${}$", short((l2 * t).exp())), small, pal::PURPLE, Align2::LEFT_TOP),
                    Lines::Every => f.text_px(Vec2::new(two, y), "on every line", small, pal::DIM, Align2::LEFT_TOP),
                    Lines::One(_) => {}
                }
            }
            Eigen::Complex { re, im } => {
                f.text_px(Vec2::new(p.lines.x, y), r"$\lambda = a \pm bi$: $e^{at} =$", small, pal::DIM, Align2::LEFT_TOP);
                f.text_px(Vec2::new(one, y), format!("${}$", short((re * t).exp())), small, pal::TEXT, Align2::LEFT_TOP);
                f.text_px(Vec2::new(two - 14.0, y), format!("$bt = {}$", num(im * t, 2)), small, pal::TEXT, Align2::LEFT_TOP);
            }
        }

        // the series' sum: how far from eᴬᵗx₀ it takes x₀, and (with room) the matrix
        if self.series {
            let n = self.terms;
            let sum = partial_sum(a, t, n);
            let off = distance((sum * self.shown().x0 - e * self.shown().x0).length());
            if p.head.is_some() {
                f.text_px(p.series, format!(r"$I + At + \cdots + (At)^N/N!$, $N = {n}$:"), small, pal::RED, Align2::LEFT_TOP);
                let sum_rows = entries(sum, [pal::RED.mix(pal::TEXT, 0.4), pal::RED.mix(pal::TEXT, 0.4)]);
                let at = p.series + Vec2::new(lead[0] * s + 8.0, LINE_H);
                let size = matrix(f, at, &sum_rows, widest, s);
                let x = at.x + size.x + 12.0;
                f.text_px(Vec2::new(x, at.y + size.y * 0.5 - 8.0), "takes $x_0$ to within", small, pal::DIM, Align2::LEFT_CENTER);
                f.text_px(Vec2::new(x, at.y + size.y * 0.5 + 8.0), format!("${off}$ of $e^{{At}} x_0$"), small, pal::DIM, Align2::LEFT_CENTER);
            } else {
                f.text_px(p.series, format!("the sum to $N = {n}$ ends ${off}$ from $e^{{At}} x_0$"), small, pal::RED, Align2::LEFT_TOP);
            }
        }

        // the time track: its knob is a handle (drawn with the panel's input)
        let (b0, b1) = p.bar;
        f.text_px(b0 - Vec2::new(12.0, 0.0), "$t$", 14.0, pal::TEXT, Align2::CENTER_CENTER);
        hud_line(f, b0, b1, 2.0, pal::DIM);
        for (v, name) in [(0.0, "$0$"), (std::f64::consts::PI, r"$\pi$"), (T_MAX, r"$2\pi$")] {
            let x = b0.x + (b1.x - b0.x) * (v / T_MAX) as f32;
            hud_line(f, Vec2::new(x, b0.y - 4.0), Vec2::new(x, b0.y + 4.0), 1.2, pal::DIM);
            f.text_px(Vec2::new(x, b0.y + 7.0), name, 10.5, pal::DIM, Align2::CENTER_TOP);
        }
        let knob = b0 + (b1 - b0) * (t / T_MAX).clamp(0.0, 1.0) as f32;
        hud_line(f, b0, knob, 3.0, pal::WHITE.a(0.8));
        f.text_px(Vec2::new(b1.x + 14.0, b0.y), format!("$t = {}$", num(t, 2)), 13.0, pal::TEXT, Align2::LEFT_CENTER);
        let (lo, hi) = p.play;
        hud_rect(f, lo, hi, Color::hex(0x1A2236));
        hud_outline(f, lo, hi, 1.0, pal::DIM);
        let busy = self.playing || self.play_when_there;
        f.text_px((lo + hi) * 0.5, if busy { "❚❚" } else { "▶" }, 14.0, pal::TEXT, Align2::CENTER_CENTER);

        self.map(f, p, a, k);
    }

    /// The map of all flows: trace across, determinant up, the regions named, and A on it.
    fn map(&self, f: &mut Frame, p: &Panel, a: DMat2, k: Kind) {
        let (min, max) = p.map;
        f.text_px(p.map_title, r"every flow, by $\tr A$ (across) and $\det A$ (up)", 11.5, pal::TEXT, Align2::LEFT_TOP);
        hud_rect(f, min, max, Color::hex(0x10172A));
        let at = |tau: f64, det: f64| map_point(p.map, tau, det);
        // det = 0, and tr = 0 above it (the centres)
        hud_line(f, at(MAP_TR[0], 0.0), at(MAP_TR[1], 0.0), 1.2, pal::DIM);
        hud_line(f, at(0.0, 0.0), at(0.0, MAP_DET[1]), 1.2, pal::DIM);
        // the parabola det = tr²/4
        let tau_max = (4.0 * MAP_DET[1]).sqrt().min(MAP_TR[1]);
        hud_path(f, (0..=60).map(|i| { let tau = -tau_max + 2.0 * tau_max * i as f64 / 60.0; at(tau, 0.25 * tau * tau) }), 1.4, pal::TEXT.a(0.8));
        // the regions' names, the one A is in brighter
        let names: [(&str, f64, f64, &[Kind]); 6] = [
            ("saddle", 1.6, -0.75, &[Kind::Saddle]),
            ("sink", -2.25, 0.45, &[Kind::Sink]),
            ("source", 2.25, 0.45, &[Kind::Source]),
            ("spiral sink", -1.45, 2.35, &[Kind::SpiralSink]),
            ("spiral source", 1.45, 2.35, &[Kind::SpiralSource]),
            ("centre", 0.0, 2.2, &[Kind::Centre]),
        ];
        for (name, tau, det, kinds) in names {
            let here = kinds.contains(&k);
            let c = if here { pal::TEXT } else { pal::DIM };
            if name == "centre" {
                f.caption_px(at(tau, det), name, 10.5, c, Align2::CENTER_CENTER);
            } else {
                f.text_px(at(tau, det), name, 10.5, c, Align2::CENTER_CENTER);
            }
        }
        f.text_px(at(MAP_TR[1], 0.0) + Vec2::new(-3.0, -3.0), r"$\tr A$", 10.5, pal::DIM, Align2::RIGHT_BOTTOM);
        f.text_px(at(0.0, MAP_DET[1]) + Vec2::new(5.0, 2.0), r"$\det A$", 10.5, pal::DIM, Align2::LEFT_TOP);
        let off = !(MAP_TR[0]..=MAP_TR[1]).contains(&trace(a)) || !(MAP_DET[0]..=MAP_DET[1]).contains(&a.determinant());
        if off {
            f.text_px(Vec2::new(min.x + 4.0, max.y - 4.0), "$A$ is off the map: shown at its edge", 10.5, pal::DIM, Align2::LEFT_BOTTOM);
        }
    }
}

/// The caption's lines as last broken, and how wide the widest is, kept with what they
/// were broken for: whose parts (a phone's or a wider canvas's) and the room across.
/// (Its formulas are measured exactly, which is work for every frame if not kept.)
struct Caption {
    of: (bool, f32),
    lines: Vec<String>,
    widest: f32,
}

/// `parts` side by side as long as they fit across `room`, on further lines when they
/// do not, and the widest line's width, each text as wide as `measure` says (none if it
/// cannot say).
fn caption_lines(parts: &[&str], room: f32, size: f32, measure: impl Fn(&str) -> Option<f32>) -> Option<(Vec<String>, f32)> {
    let gap = "      ";
    let mut lines: Vec<String> = Vec::new();
    for part in parts {
        let fits = match lines.last() {
            Some(last) => measure(last)? + text_width(gap, size) + measure(part)? <= room,
            None => false,
        };
        match lines.last_mut() {
            Some(last) if fits => {
                last.push_str(gap);
                last.push_str(part);
            }
            _ => lines.push(part.to_string()),
        }
    }
    let mut widest = 0.0f32;
    for line in &lines {
        widest = widest.max(measure(line)?);
    }
    Some((lines, widest))
}

/// A caption along the bottom of the picture, on a dark plate, naming what has no name
/// of its own there: `parts` side by side as long as they fit across `(left, right)`,
/// on further lines above when they do not, the last ending at `bottom`; the lines
/// kept in `kept` for as long as the parts (`narrow`, a phone's) and the room are the
/// same. Returns the box it takes (none on a canvas too small to spare the room).
fn caption(f: &mut Frame, parts: &[&str], narrow: bool, (left, right): (f32, f32), bottom: f32, kept: &mut Option<Caption>) -> Option<(Vec2, Vec2)> {
    let size = 12.0;
    if f.view_size().y < 300.0 {
        return None;
    }
    let room = right - left - 36.0;
    if kept.as_ref().is_none_or(|c| c.of != (narrow, room)) {
        // (none kept before the formulas' fonts are in place, the first frame)
        *kept = caption_lines(parts, room, size, |t| measured(f, t, size)).map(|(lines, widest)| Caption { of: (narrow, room), lines, widest });
    }
    let (lines, widest) = match kept {
        Some(c) => (c.lines.clone(), c.widest),
        None => caption_lines(parts, room, size, |t| Some(width(f, t, size))).unwrap_or_default(),
    };
    let count = lines.len();
    for (k, line) in lines.into_iter().enumerate() {
        let up = (count - 1 - k) as f32 * (size + 7.0);
        f.caption_px(Vec2::new(left + 18.0, bottom - up), line, size, pal::DIM, Align2::LEFT_BOTTOM);
    }
    Some((Vec2::new(left + 10.0, bottom - count as f32 * (size + 7.0) - 4.0), Vec2::new(left + 26.0 + widest, bottom + 6.0)))
}

/// The heading's sizes as last chosen (its first line's and its second's, and how wide
/// the wider is), kept with what they were chosen for: the two lines and the room.
struct Heading {
    of: (String, &'static str, f32),
    sizes: (f32, f32, f32),
}

/// The sizes of the heading's two lines: each as large as fits across `room` (as wide
/// as `measure` says; none if it cannot say), and how wide the wider then is.
fn heading_sizes(first: &str, second: &str, room: f32, measure: impl Fn(&str, f32) -> Option<f32>) -> Option<(f32, f32, f32)> {
    let size = if measure(first, 15.0)? <= room { 15.0 } else { 13.0 };
    let small = if measure(second, 12.5)? <= room { 12.5 } else { 11.0 };
    Some((size, small, measure(first, size)?.max(measure(second, small)?).min(room)))
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

/// A small chevron at `p` pointing along `dir`, its size fixed on the screen.
fn chevron(f: &mut Frame, p: Vec2, dir: Vec2, wpp: f32, c: Color) {
    let d = dir.normalize_or_zero() * (6.0 * wpp);
    if d == Vec2::ZERO {
        return;
    }
    let n = d.perp();
    lines_for(f, Part::Eigen).polyline([p - d + n, p, p - d - n].map(|q| q.extend(0.0)), 1.8, c);
}

/// The parts of the picture drawn as lines.
#[derive(Clone, Copy, PartialEq)]
enum Part {
    /// What keeps its number of points from frame to frame (in the usual batch).
    Fixed,
    /// The eigenlines' arrows and marks, which come and go with the lines.
    Eigen,
    /// The paths of (1, 0) and (0, 1), which grow with t.
    Columns,
    /// The start's path, so far and ahead.
    Start,
    /// The series' arrows, which come one at a time.
    Series,
}

/// The line batch a part is drawn in. A part whose number of points changes as the
/// picture moves has a batch of its own (different from the usual one by a hair of
/// its width, and nothing else), so that from one frame to the next the parts that
/// stay the same can still be compared point for point: that is how `eidos motion`
/// tells smooth motion from jumps.
fn lines_for<'a>(f: &'a mut Frame, part: Part) -> &'a mut crate::gfx::draw::LineBatch {
    if part == Part::Fixed {
        return &mut f.dl.lines;
    }
    f.dl.lines_with(LineKind::Blend, LineStyle { width_scale: 1.0 + 1e-4 * part as u8 as f32, ..Default::default() })
}

/// An arrow from `a` to `b` (as `Frame::arrow` draws one, its shaft in the batch of
/// `part`), the part of it in the box `lo..hi`: without its head if the head is outside.
#[allow(clippy::too_many_arguments)]
fn arrow_in(f: &mut Frame, part: Part, a: DVec2, b: DVec2, lo: DVec2, hi: DVec2, width: f32, c: Color) {
    let Some((p, q)) = clip(a, b, lo, hi) else { return };
    let (p, q) = (p.as_vec2().extend(0.0), q.as_vec2().extend(0.0));
    let d = q - p;
    let len = d.length();
    if len < 1e-9 {
        return;
    }
    let dir = d / len;
    let mut end = q;
    if clip(a, b, lo, hi).is_some_and(|(_, q)| q == b) {
        let wpp = f.cam.world_per_px(q) * f.px;
        let head = ((width * 3.2 + 5.0) * wpp).min(len * 0.6);
        let side = dir.cross(Vec3::Z).normalize_or_zero() * (head * 0.42);
        let base = q - dir * head;
        f.dl.tris_alpha.tri(q, base + side, base - side, c);
        end = q - dir * (head * 0.8);
    }
    lines_for(f, part).segment(p, end, width, c);
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

fn inside_box(p: DVec2, lo: DVec2, hi: DVec2) -> bool {
    p.is_finite() && p.cmpge(lo).all() && p.cmple(hi).all()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::cell::Cell;

    thread_local! {
        /// Whether names may be kept from frame to frame (`keep_names`), and in how
        /// many frames they have been.
        static KEEP: Cell<bool> = const { Cell::new(true) };
        static KEPT: Cell<u32> = const { Cell::new(0) };
    }

    pub(super) fn kept(again: bool) -> bool {
        let keep = again && KEEP.get();
        KEPT.set(KEPT.get() + keep as u32);
        keep
    }

    /// What a frame drew: its points, batch by batch, as bits, and its labels.
    type Drawn = (Vec<Vec<[u32; 2]>>, Vec<String>);

    /// The scene in the whole application, without drawing, through everything that
    /// moves its names: the opening's play, the streaks alone, drags of x₀, of the
    /// time's knob and of the dot on the map, an experiment, the window made smaller
    /// and then a phone's. Every frame's points and labels: where they are (to the
    /// bit) and what they say. (None where nothing can draw, if that is allowed.)
    fn run_drawn() -> Option<Vec<Drawn>> {
        use crate::headless::{NoAdapter, Stage};
        let mut stage = match Stage::showing("expm") {
            Ok(stage) => stage,
            Err(e) if e.is::<NoAdapter>() && std::env::var("EIDOS_ALLOW_NO_GPU").is_ok_and(|v| v == "1") => return None,
            Err(e) => panic!("no application to run the scene in: {e}"),
        };
        stage.app.set_dry(true);
        let button = |at: egui::Pos2, pressed| egui::Event::PointerButton { pos: at, button: egui::PointerButton::Primary, pressed, modifiers: Default::default() };
        // drags (first frame, last frame, from, to) and a click
        let drags = [(200, 230, (487.0, 785.0), (600.0, 700.0)), (270, 290, (1050.0, 155.0), (1120.0, 155.0)), (330, 360, (1140.0, 369.0), (1100.0, 300.0))];
        let click = (400, egui::pos2(1410.0, 726.0));
        let mut drawn = Vec::new();
        for frame in 0..700u32 {
            let size = match frame {
                520..560 => egui::vec2(1300.0, 900.0),
                560..620 => egui::vec2(390.0, 800.0),
                _ => egui::vec2(1600.0, 1000.0),
            };
            let screen = egui::Rect::from_min_size(egui::Pos2::ZERO, size);
            let mut raw = egui::RawInput { screen_rect: Some(screen), time: Some(frame as f64 / 60.0), predicted_dt: 1.0 / 60.0, focused: true, ..Default::default() };
            for (first, last, from, to) in drags {
                let (from, to) = (egui::pos2(from.0, from.1), egui::pos2(to.0, to.1));
                if frame + 1 == first {
                    raw.events.push(egui::Event::PointerMoved(from));
                } else if frame == first {
                    raw.events.extend([egui::Event::PointerMoved(from), button(from, true)]);
                } else if frame > first && frame < last {
                    raw.events.push(egui::Event::PointerMoved(from + (to - from) * ((frame - first) as f32 / (last - 1 - first) as f32)));
                } else if frame == last {
                    raw.events.push(button(to, false));
                }
            }
            if frame + 2 == click.0 || frame + 1 == click.0 {
                raw.events.push(egui::Event::PointerMoved(click.1));
            } else if frame == click.0 || frame == click.0 + 1 {
                raw.events.push(button(click.1, frame == click.0));
            }
            let out = stage.frame(raw);
            stage.done(&out.textures_delta);
            let points = stage.app.drawn_points().iter().map(|batch| batch.iter().map(|p| [p.x.to_bits(), p.y.to_bits()]).collect()).collect();
            drawn.push((points, stage.app.drawn_labels().iter().map(|l| l.to_string()).collect()));
        }
        Some(drawn)
    }

    /// Names put where they went in the last frame (`Names`) are where they would
    /// have been put afresh: the same run, once placing every name afresh in every
    /// frame and once keeping them, draws the same points and labels in every frame.
    #[test]
    fn names_kept_are_where_they_would_be_put_afresh() {
        KEEP.set(false);
        let afresh = run_drawn();
        KEEP.set(true);
        KEPT.set(0);
        let kept = run_drawn();
        let (Some(afresh), Some(kept)) = (afresh, kept) else { return };
        for (frame, (a, k)) in afresh.iter().zip(&kept).enumerate() {
            assert!(a == k, "frame {frame}: {:?} against {:?}", a.1, k.1);
        }
        // (kept in the frames in which only the streaks move: most of those at rest)
        assert!(KEPT.get() > 250, "names kept in only {} frames", KEPT.get());
    }

    /// The paths kept from frame to frame are the paths of A and x₀ as they now are,
    /// to the bit, however either has changed.
    #[test]
    fn the_paths_kept_are_the_paths_of_a_and_x0() {
        let same = |p: &Track, q: &Track| p.h.to_bits() == q.h.to_bits() && p.points.len() == q.points.len() && p.points.iter().zip(&q.points).all(|(a, b)| a.to_array().map(f64::to_bits) == b.to_array().map(f64::to_bits));
        let (saddle, spiral) = (rows(TRIALS[0].a), rows(TRIALS[1].a));
        let (x, y) = (DVec2::new(-0.8, -2.4), DVec2::new(1.5, 1.0));
        let mut paths = Paths::default();
        for (a, x0) in [(saddle, x), (saddle, x), (saddle, y), (spiral, y), (spiral, x), (saddle * -1.0, x), (DMat2::ZERO, x), (saddle, x + DVec2::new(1e-15, 0.0))] {
            let columns = paths.columns(a);
            assert!(same(&columns[0], &Track::of(a, DVec2::X)) && same(&columns[1], &Track::of(a, DVec2::Y)), "{a:?}");
            assert!(same(paths.start(a, x0), &Track::of(a, x0)), "{a:?} {x0:?}");
        }
    }

    /// e^(At) by its series, summed until the terms are nothing (for checking).
    fn by_series(a: DMat2, t: f64) -> DMat2 {
        partial_sum(a, t, 80)
    }

    fn close(m: DMat2, n: DMat2, tol: f64) -> bool {
        (m - n).to_cols_array().iter().all(|v| v.abs() <= tol)
    }

    /// The closed form agrees with the series for every kind of matrix, on both sides
    /// of a double eigenvalue and right at it; flowing for s and then t is flowing
    /// for s + t; and areas grow at the rate tr A.
    #[test]
    fn the_exponential_is_right_everywhere() {
        let mut rng = Rng::new(11);
        let mut cases: Vec<DMat2> = TRIALS.iter().map(|t| rows(t.a)).chain(PRESETS.iter().map(|p| rows(p.1))).collect();
        cases.extend((0..300).map(|_| rows([0; 4].map(|_| rng.range(-2.0, 2.0) as f64))));
        // a double eigenvalue, and a hair either side of it
        for eps in [0.0, 1e-14, -1e-14, 1e-8, -1e-8, 1e-3, -1e-3] {
            cases.push(rows([-1.0, 1.0, eps, -1.0]));
            cases.push(rows([0.5, 2.0, eps - 0.125, 0.0]));
        }
        cases.push(DMat2::ZERO);
        cases.push(DMat2::IDENTITY * -0.7);
        for a in cases {
            for t in [0.0, 0.3, 1.0, 2.5, -0.8] {
                let e = expm(a, t);
                let s = by_series(a, t);
                let scale = size(s).max(1.0);
                assert!(close(e, s, 1e-11 * scale), "{a:?} at {t}: {e:?} against {s:?}");
                assert!((e.determinant() - (t * trace(a)).exp()).abs() <= 1e-11 * scale * scale, "{a:?} at {t}");
                assert!(close(expm(a, t) * expm(a, 0.7), expm(a, t + 0.7), 1e-10 * scale * size(expm(a, 0.7)).max(1.0)));
            }
        }
        // the matrix that turns: e^(tJ) is the turn by t
        let j = rows([0.0, -1.0, 1.0, 0.0]);
        for t in [0.0, 0.5, PI as f64 / 2.0, 2.0, std::f64::consts::PI, 5.0, T_MAX] {
            let (s, c) = t.sin_cos();
            assert!(close(expm(j, t), rows([c, -s, s, c]), 1e-14), "{t}");
        }
        // the largest numbers that can be asked for stay finite, in single precision too
        for a in [rows([ENTRY_MAX; 4]), rows([ENTRY_MAX, -ENTRY_MAX, -ENTRY_MAX, ENTRY_MAX]), rows([-ENTRY_MAX; 4])] {
            let e = expm(a, T_MAX);
            assert!(e.as_mat2().is_finite() && (e * DVec2::splat(X0_MAX)).as_vec2().is_finite());
            let worst = terms(a, T_MAX, DVec2::splat(X0_MAX), N_MAX).iter().fold(0.0f64, |m, v| m.max(v.length()));
            assert!(worst < 1e30, "{worst}");
            assert!(partial_sum(a, T_MAX, N_MAX).as_mat2().is_finite());
        }
    }

    /// The kinds, and the eigenlines: every line drawn is kept by A, with its λ.
    #[test]
    fn each_matrix_is_put_in_its_place() {
        let expect = [
            ([1.0, -1.0, 0.0, -1.0], Kind::Saddle),
            ([-0.25, -1.5, 1.5, -0.25], Kind::SpiralSink),
            ([0.15, -1.5, 1.5, 0.15], Kind::SpiralSource),
            ([0.0, -1.0, 1.0, 0.0], Kind::Centre),
            ([1.0, -2.0, 1.0, -1.0], Kind::Centre),
            ([-1.0, 0.5, 0.5, -1.0], Kind::Sink),
            ([1.0, 0.5, 0.5, 1.0], Kind::Source),
            ([-1.0, 1.0, 0.0, -1.0], Kind::ImproperSink),
            ([1.0, 1.0, 0.0, 1.0], Kind::ImproperSource),
            ([-2.0, 0.0, 0.0, -2.0], Kind::StarSink),
            ([0.3, 0.0, 0.0, 0.3], Kind::StarSource),
            ([-0.5, -0.5, -0.5, -0.5], Kind::RestIn),
            ([0.0, 0.0, 0.0, 2.0], Kind::RestOut),
            ([0.0, 1.0, 0.0, 0.0], Kind::Shear),
            ([0.0; 4], Kind::Still),
        ];
        for (m, want) in expect {
            let a = rows(m);
            assert_eq!(kind(a), want, "{m:?}");
            match eigen(a) {
                Eigen::Real { l1, l2, lines } => {
                    assert!(l1 >= l2 && (l1 + l2 - trace(a)).abs() < 1e-12 && (l1 * l2 - a.determinant()).abs() < 1e-12, "{m:?}");
                    let keeps = |l: f64, v: DVec2| (a * v - v * l).length() < 1e-9 && (v.length() - 1.0).abs() < 1e-12;
                    match lines {
                        Lines::Two(v1, v2) => assert!(keeps(l1, v1) && keeps(l2, v2) && v1.perp_dot(v2).abs() > 1e-3, "{m:?}"),
                        Lines::One(v) => assert!(keeps(l1, v), "{m:?}"),
                        Lines::Every => assert!(close(a, DMat2::IDENTITY * l1, 1e-12), "{m:?}"),
                    }
                }
                Eigen::Complex { re, im } => assert!((2.0 * re - trace(a)).abs() < 1e-12 && (re * re + im * im - a.determinant()).abs() < 1e-12 && im > 0.0, "{m:?}"),
            }
        }
    }

    /// A drag on the map starts where A is, moves it to the trace and determinant
    /// asked for, and smoothly: a small move of the dot is a small change of A.
    #[test]
    fn a_drag_on_the_map_moves_a_where_it_is_asked() {
        let mut rng = Rng::new(3);
        let mut starts: Vec<DMat2> = PRESETS.iter().map(|p| rows(p.1)).collect();
        starts.extend((0..100).map(|_| rows([0; 4].map(|_| rng.range(-1.5, 1.5) as f64))));
        starts.push(DMat2::IDENTITY * -1.0);
        starts.push(DMat2::ZERO);
        for a in starts {
            let fam = family_for(a);
            assert!(close(fam.at(trace(a), a.determinant()), a, 1e-9 * size(a).max(1.0)), "{a:?}");
            let mut last = fam.at(trace(a), a.determinant());
            // across the whole map, in small steps
            for i in 1..=200 {
                let s = i as f64 / 200.0;
                let (tau, det) = (trace(a) + (2.5 - trace(a)) * s, a.determinant() + (2.6 - a.determinant()) * s);
                let m = fam.at(tau, det);
                if size(m) < ENTRY_MAX * 0.99 {
                    assert!((trace(m) - tau).abs() < 1e-9 && (m.determinant() - det).abs() < 1e-9, "{a:?} at {tau}, {det}: {m:?}");
                }
                assert!(size(m - last) < 0.25, "{a:?}: a jump at {tau}, {det}");
                last = m;
            }
        }
    }

    /// The scene as an experiment's button leaves it, run at 60 frames a second until
    /// it has played (or for `seconds` at most).
    fn run(k: usize, seconds: f32) -> Expm {
        let mut scene = Expm::new(None);
        scene.set_up(k);
        for _ in 0..(seconds * 60.0) as usize {
            scene.advance(1.0 / 60.0, 1.0 / 60.0);
            if scene.glide.arrived() && !scene.playing && !scene.play_when_there && scene.counting.is_none() && scene.tour.is_none() && scene.terms_shown == scene.terms as f64 {
                break;
            }
        }
        scene
    }

    fn angle(v: DVec2) -> f64 {
        v.y.atan2(v.x).to_degrees()
    }

    /// The captions of the "things to try", held to account.
    #[test]
    fn the_saddle_does_what_its_caption_says() {
        let scene = run(0, 30.0);
        let Setting { a, t, x0 } = scene.shown();
        assert!((t - 1.5).abs() < 1e-12 && !scene.playing);
        assert_eq!(kind(a), Kind::Saddle);
        let Eigen::Real { l1, l2, lines: Lines::Two(v1, v2) } = eigen(a) else { panic!("{:?}", eigen(a)) };
        assert!(l1 == 1.0 && l2 == -1.0);
        // orange is the larger λ: points on it go straight out by e^t, on purple in by e^-t
        for s in [0.5, 1.0, 1.75] {
            assert!((expm(a, s) * v1 - v1 * s.exp()).length() < 1e-12 && (expm(a, s) * v2 - v2 * (-s).exp()).length() < 1e-12);
        }
        // x0 starts near the purple line, is drawn in towards 0 and then swept out along
        // orange: nearest 0 part of the way along, and from there ever closer to the
        // orange line, which the path ahead (to t = 2 pi) reaches
        let off = |p: DVec2, v: DVec2| (angle(p) - angle(v)).abs().min((angle(p) - angle(-v)).abs());
        let mut way = path(a, x0, 0.0, t, 0.01);
        let now = way.len() - 1;
        way.extend(path(a, x0, t, T_MAX, 0.01).into_iter().skip(1));
        let dist: Vec<f64> = way.iter().map(|p| p.length()).collect();
        let nearest = dist.iter().enumerate().min_by(|a, b| a.1.total_cmp(b.1)).map(|(i, _)| i).unwrap_or(0);
        assert!(off(x0, v2) < 10.0 && off(x0, v1) > 60.0, "{} {}", off(x0, v2), off(x0, v1));
        assert!(nearest > 20 && nearest + 20 < now, "{nearest} of {now}");
        assert!((way[now] - expm(a, t) * x0).length() < 1e-9);
        assert!(way[nearest..].windows(2).all(|w| off(w[1], v1) <= off(w[0], v1) + 1e-9));
        assert!(off(way[now], v1) < 25.0 && off(*way.last().unwrap(), v1) < 0.1, "{} {}", off(way[now], v1), off(*way.last().unwrap(), v1));
        // the area stays 1
        assert!((expm(a, t).determinant() - 1.0).abs() < 1e-12 && trace(a) == 0.0);
    }

    #[test]
    fn the_spiral_sink_does_what_its_caption_says() {
        let scene = run(1, 30.0);
        let Setting { a, t, x0 } = scene.shown();
        assert_eq!(kind(a), Kind::SpiralSink);
        assert_eq!(eigen(a), Eigen::Complex { re: -0.25, im: 1.5 });
        assert!((t - std::f64::consts::TAU / 1.5).abs() < 1e-12 && num(t, 2) == "4.19");
        // every point turns at 1.5 a unit of time and shrinks as e^(-0.25 t)
        for s in [0.3, 1.0, 2.0] {
            for p in [DVec2::new(1.0, 0.0), DVec2::new(-0.4, 2.0)] {
                let q = expm(a, s) * p;
                assert!((q.length() - p.length() * (-0.25 * s).exp()).abs() < 1e-12);
                let turn = p.angle_to(q) - 1.5 * s;
                let pi = std::f64::consts::PI;
                assert!(((turn + pi).rem_euclid(2.0 * pi) - pi).abs() < 1e-9, "{turn}");
            }
        }
        // one full turn: back on its ray, at 0.35 of the distance; area 0.12
        let end = expm(a, t) * x0;
        assert!(end.y.abs() < 1e-12 && end.x > 0.0 && num(end.length() / x0.length(), 2) == "0.35");
        assert!(num(expm(a, t).determinant(), 2) == "0.12" && (expm(a, t).determinant() - (-0.5 * t).exp()).abs() < 1e-12);
    }

    #[test]
    fn the_turning_matrix_does_what_its_caption_says() {
        let scene = run(2, 30.0);
        let Setting { a, t, x0 } = scene.shown();
        assert_eq!(kind(a), Kind::Centre);
        assert_eq!(eigen(a), Eigen::Complex { re: 0.0, im: 1.0 });
        assert!((t - T_MAX).abs() < 1e-12);
        // Jx is at right angles to x, so points go round circles; e^(tJ) is the turn by t
        for p in [DVec2::new(1.0, 0.3), x0] {
            assert!((a * p).dot(p).abs() < 1e-15);
            for s in [0.4, 1.0, 3.0, t] {
                assert!(((expm(a, s) * p).length() - p.length()).abs() < 1e-12);
            }
        }
        assert!(close(expm(a, std::f64::consts::FRAC_PI_2), rows([0.0, -1.0, 1.0, 0.0]), 1e-15) && num(std::f64::consts::FRAC_PI_2, 2) == "1.57");
        assert!(close(expm(a, t), DMat2::IDENTITY, 1e-14) && num(T_MAX, 2) == "6.28");
        assert!((expm(a, 1.3).determinant() - 1.0).abs() < 1e-14);
    }

    #[test]
    fn the_series_does_what_its_caption_says() {
        let mut scene = run(3, 60.0);
        assert!(scene.series && scene.terms == COUNT_TO && scene.terms_shown == COUNT_TO as f64);
        let Setting { a, t, x0 } = scene.shown();
        assert!(t == std::f64::consts::PI && x0 == DVec2::new(0.0, -0.75));
        let exact = expm(a, t) * x0;
        assert!((exact + x0).length() < 1e-14);
        let parts = terms(a, t, x0, N_MAX);
        // the first term is a straight step along the velocity at x0; each next is the
        // last turned a quarter and pi/k times as long
        assert!((parts[1] - a * x0 * t).length() < 1e-15);
        for k in 2..=N_MAX {
            let (last, this) = (parts[k - 1], parts[k]);
            assert!(last.dot(this).abs() < 1e-12 * last.length() * this.length() && last.perp_dot(this) > 0.0);
            assert!((this.length() / last.length() - t / k as f64).abs() < 1e-12);
        }
        // out to 3.22 at the most, then in: within 0.01 from N = 10 on, and not before
        let off: Vec<f64> = (0..=N_MAX).map(|n| (partial_sum(a, t, n) * x0 - exact).length()).collect();
        let most = off.iter().cloned().fold(0.0, f64::max);
        assert!(num(most, 2) == "3.22" && off[2] == most, "{off:?}");
        assert!(off[9] > 0.01 && off[10..].iter().all(|e| *e < 0.01), "{off:?}");
        // the sum drawn is the sum of the terms drawn
        let drawn = parts.iter().skip(1).take(scene.terms).fold(x0, |s, p| s + *p);
        assert!((drawn - partial_sum(a, t, scene.terms) * x0).length() < 1e-12);
        // and moving N by hand stops the counting
        scene.counting = Some(1.0);
        scene.terms = 3;
        scene.counting = None;
        scene.advance(1.0, 1.0);
        assert_eq!(scene.terms, 3);
    }

    /// What each experiment is about lies in the part of the scene the view is fitted
    /// to: the start, where it ends, and (for the series) every partial sum.
    #[test]
    fn every_experiment_is_in_view() {
        let (lo, hi) = ((FRAME_CENTRE - FRAME_HALF).as_dvec2(), (FRAME_CENTRE + FRAME_HALF).as_dvec2());
        let inside = |p: DVec2| p.cmpge(lo).all() && p.cmple(hi).all();
        for k in 0..TRIALS.len() {
            let scene = run(k, 60.0);
            let Setting { a, t, x0 } = scene.shown();
            assert!(inside(x0) && inside(expm(a, t) * x0), "trial {k}");
            if scene.series {
                assert!((0..=N_MAX).all(|n| inside(partial_sum(a, t, n) * x0)), "the series");
            }
        }
    }

    #[test]
    fn the_tour_goes_round_every_kind_of_flow() {
        let mut scene = Expm::new(None);
        scene.set_up(4);
        let mut seen: Vec<Kind> = Vec::new();
        for _ in 0..(60.0 * (tour_length() + 5.0)) as usize {
            scene.advance(1.0 / 60.0, 1.0 / 60.0);
            let k = kind(scene.shown().a);
            if seen.last() != Some(&k) {
                seen.push(k);
            }
        }
        assert!(scene.tour.is_none());
        // in the order of the caption, with the boundaries crossed on the way
        let order = [Kind::Saddle, Kind::Sink, Kind::SpiralSink, Kind::Centre, Kind::SpiralSource, Kind::Source, Kind::Saddle];
        let mut at = 0;
        for k in &seen {
            if at < order.len() && *k == order[at] {
                at += 1;
            }
        }
        assert_eq!(at, order.len(), "{seen:?}");
        assert!(seen.iter().all(|k| order.contains(k) || matches!(k, Kind::RestIn | Kind::RestOut | Kind::ImproperSink | Kind::ImproperSource)), "{seen:?}");
        // every stop is where the map names it
        for (tau, det) in TOUR {
            let m = canonical(tau, det);
            assert!((trace(m) - tau).abs() < 1e-15 && (m.determinant() - det).abs() < 1e-15);
        }
    }

    /// What the quick choices' notes say.
    #[test]
    fn the_quick_choices_do_what_their_notes_say() {
        let a = |k: usize| rows(PRESETS[k].1);
        let direction = |m: DMat2, p: DVec2, t: f64| (expm(m, t) * p).normalize();
        let along = |u: DVec2, v: DVec2| u.perp_dot(v.normalize()).abs();
        // sink: in to 0, in the end along the lambda = -0.5 line (1, 1)
        assert_eq!(eigen(a(0)), Eigen::Real { l1: -0.5, l2: -1.5, lines: eigen(a(0)).lines() });
        for p in [DVec2::new(2.0, -1.0), DVec2::new(-0.3, 1.2)] {
            // (the other part is e^(-12) times as large, relative to this one, by t = 12)
            assert!(along(direction(a(0), p, 12.0), DVec2::ONE) < 1e-4 && (expm(a(0), 12.0) * p).length() < 0.01);
            assert!(along(direction(a(1), p, 12.0), DVec2::ONE) < 1e-4 && (expm(a(1), 12.0) * p).length() > 100.0);
        }
        assert!(matches!(eigen(a(1)), Eigen::Real { l1, l2, .. } if l1 == 1.5 && l2 == 0.5));
        assert!(matches!(eigen(a(2)), Eigen::Real { l1, l2, .. } if l1 == 1.0 && l2 == -1.0));
        assert_eq!(eigen(a(3)), Eigen::Complex { re: -0.25, im: 1.5 });
        assert_eq!(eigen(a(4)), Eigen::Complex { re: 0.15, im: 1.5 });
        // centre: lambda = +-i, every point back at t = 2 pi
        assert_eq!(eigen(a(5)), Eigen::Complex { re: 0.0, im: 1.0 });
        assert!(close(expm(a(5), T_MAX), DMat2::IDENTITY, 1e-13));
        // one eigenline: e^(At) = e^(-t)(I + tN), and points come in along the line
        let n = a(6) + DMat2::IDENTITY;
        for t in [0.5, 2.0] {
            assert!(close(expm(a(6), t), (DMat2::IDENTITY + n * t) * (-t).exp(), 1e-14));
        }
        assert!(matches!(eigen(a(6)), Eigen::Real { l1, lines: Lines::One(v), .. } if l1 == -1.0 && along(v, DVec2::X) < 1e-12));
        assert!(along(direction(a(6), DVec2::new(0.2, 1.0), 40.0), DVec2::X) < 0.03);
        // a line at rest: lambda 0 and -1; points on the first stay, others flow straight to it
        assert!(matches!(eigen(a(7)), Eigen::Real { l1, l2, .. } if l1 == 0.0 && l2 == -1.0));
        let rest = DVec2::new(1.0, -1.0);
        assert!((expm(a(7), 3.0) * rest - rest).length() < 1e-15);
        let p = DVec2::new(1.5, 0.5);
        for t in [0.5, 1.0, 4.0] {
            assert!(along(expm(a(7), t) * p - p, DVec2::ONE) < 1e-12);
        }
        // shear: e^(At) = I + At, sliding sideways at a speed equal to the height
        for t in [0.5, 3.0] {
            assert!(close(expm(a(8), t), DMat2::IDENTITY + a(8) * t, 0.0));
        }
        assert_eq!(a(8) * DVec2::new(0.3, 1.7), DVec2::new(1.7, 0.0));
    }

    impl Eigen {
        fn lines(&self) -> Lines {
            match self {
                Eigen::Real { lines, .. } => *lines,
                Eigen::Complex { .. } => Lines::Every,
            }
        }
    }

    /// The dots at t = 1, 2, 3, … are the powers of e^A applied to x₀, and the
    /// eigenvalues of e^A are e^λ: smaller than 1 in size exactly when the real part of λ
    /// is negative.
    #[test]
    fn the_whole_times_are_powers_of_e_to_the_a() {
        let mut rng = Rng::new(17);
        for _ in 0..200 {
            let a = rows([0; 4].map(|_| rng.range(-1.5, 1.5) as f64));
            let (m, x0) = (expm(a, 1.0), DVec2::new(rng.range(-2.0, 2.0) as f64, rng.range(-2.0, 2.0) as f64));
            let mut p = x0;
            for k in 1..=6 {
                p = m * p;
                let exact = expm(a, k as f64) * x0;
                assert!((p - exact).length() < 1e-12 * exact.length().max(1.0) * 10.0, "{a:?} at {k}");
            }
            // the sizes of the eigenvalues of e^A, against those of e^λ, e^(real part of λ)
            let sizes = match eigen(m) {
                Eigen::Real { l1, l2, .. } => [l1.abs(), l2.abs()],
                Eigen::Complex { re, im } => [re.hypot(im); 2],
            };
            let want = match eigen(a) {
                Eigen::Real { l1, l2, .. } => [l1.exp(), l2.exp()],
                Eigen::Complex { re, .. } => [re.exp(); 2],
            };
            for (s, w) in sizes.iter().zip(want) {
                assert!((s - w).abs() < 1e-6 * w.max(1.0), "{a:?}: {sizes:?} against {want:?}");
            }
            let shrinks = match eigen(a) {
                Eigen::Real { l1, .. } => l1 < 0.0,
                Eigen::Complex { re, .. } => re < 0.0,
            };
            assert_eq!(shrinks, sizes.iter().all(|s| *s < 1.0), "{a:?}");
        }
    }

    /// Clipping keeps what is inside and nothing else, and a polygon cut to a box has
    /// the area of what is inside.
    #[test]
    fn what_runs_far_off_is_cut_at_the_box() {
        let (lo, hi) = (DVec2::splat(-1.0), DVec2::splat(1.0));
        assert_eq!(clip(DVec2::new(-3.0, 0.0), DVec2::new(3.0, 0.0), lo, hi), Some((DVec2::new(-1.0, 0.0), DVec2::new(1.0, 0.0))));
        assert_eq!(clip(DVec2::new(-3.0, 2.0), DVec2::new(3.0, 2.0), lo, hi), None);
        assert_eq!(clip(DVec2::new(f64::NAN, 0.0), DVec2::ZERO, lo, hi), None);
        let big = [DVec2::ZERO, DVec2::new(1e20, 0.0), DVec2::new(1e20, 1e20), DVec2::new(0.0, 1e20)];
        let cut = clip_polygon(&big, lo, hi);
        let area: f64 = (0..cut.len()).map(|i| cut[i].perp_dot(cut[(i + 1) % cut.len()])).sum::<f64>() * 0.5;
        assert!((area - 1.0).abs() < 1e-9, "{cut:?}");
        let pts: Vec<DVec2> = (0..50).map(|i| DVec2::new(-2.0 + i as f64 * 0.1, 0.0)).collect();
        let r = runs(&pts, lo, hi);
        assert!(r.len() == 1 && r[0].first().unwrap().x == -1.0 && r[0].last().unwrap().x == 1.0, "{r:?}");
    }

    /// Degenerate input: every number drawn or read out is finite, whatever A, t and
    /// x0 are, and the particles too.
    #[test]
    fn degenerate_input_leaves_everything_finite() {
        let mats = [DMat2::ZERO, DMat2::IDENTITY * 1e-13, rows([0.0, 1.0, 0.0, 0.0]), rows([ENTRY_MAX; 4]), rows([-ENTRY_MAX, ENTRY_MAX, -ENTRY_MAX, ENTRY_MAX]), rows([1.0, 1.0, 1.0, 1.0]), DMat2::IDENTITY * -ENTRY_MAX];
        for a in mats {
            for t in [0.0, 1e-9, 1.0, T_MAX] {
                for x0 in [DVec2::ZERO, DVec2::splat(X0_MAX), DVec2::new(-X0_MAX, 3.0)] {
                    let e = expm(a, t);
                    assert!(e.as_mat2().is_finite() && (e * x0).as_vec2().is_finite() && (a * (e * x0)).as_vec2().is_finite());
                    let track = Track::of(a, x0);
                    assert!(track.points.iter().all(|p| p.as_vec2().is_finite()) && track.points.len() == TRACK_POINTS + 1);
                    assert!(track.upto(t, e * x0).len() + track.ahead(t, e * x0).len() == TRACK_POINTS + 3);
                    assert!(terms(a, t, x0, N_MAX).iter().all(|p| p.as_vec2().is_finite()));
                    assert!(!short(e.determinant()).contains("NaN") && !short(e.determinant()).contains("inf"));
                }
            }
            let mut flow = Flow::new(None);
            for k in 0..400 {
                // (frames of every length, a stopped one and one with no length among them)
                let dt = [1.0 / 60.0, 1.0 / 144.0, 0.0, 0.3, f32::NAN, 1.0 / 30.0][k % 6];
                flow.advance(a, dt, Vec2::splat(-3.0), Vec2::splat(3.0), streak_colour(), None);
                assert!(flow.pos.iter().all(|p| p.is_finite()) && flow.owed >= 0.0 && flow.owed < ROW_EVERY);
                assert!(flow.row.iter().all(|r| r.pos.iter().all(|v| v.is_finite()) && r.color.0[3] <= 128));
            }
            assert!(family_for(a).at(0.0, 1.0).as_mat2().is_finite());
        }
    }

    /// The panel and the picture fit the canvas, from a phone to a desktop, and the map
    /// and the time track stay inside the panel.
    #[test]
    fn a_shape_rises_over_a_line_as_far_as_it_does() {
        let (along, up) = (Vec2::X, Vec2::NEG_Y);
        // a triangle standing on the line from x = 0 to 100, its top 40 above x = 50
        let triangle = [Vec2::new(0.0, 0.0), Vec2::new(100.0, 0.0), Vec2::new(50.0, -40.0)];
        assert!((rises(&triangle, Vec2::new(50.0, 0.0), along, up, 10.0) - 40.0).abs() < 1e-3);
        assert!((rises(&triangle, Vec2::new(25.0, 0.0), along, up, 5.0) - 24.0).abs() < 1e-3);
        assert!((rises(&triangle, Vec2::new(0.0, 0.0), along, up, 5.0) - 4.0).abs() < 1e-3);
        // beside it, nothing; hanging below the line, less than nothing
        assert!(rises(&triangle, Vec2::new(200.0, 0.0), along, up, 10.0) == f32::MIN);
        assert!(rises(&triangle, Vec2::new(50.0, -60.0), along, up, 10.0) < 0.0);
        assert!(rises(&[], Vec2::ZERO, along, up, 10.0) == f32::MIN);
    }

    /// The flows the streaks are tried on: every kind there is, and the fastest.
    fn flows() -> Vec<(&'static str, DMat2)> {
        let mut flows: Vec<(&'static str, DMat2)> = PRESETS.iter().map(|(name, m, _)| (*name, rows(*m))).collect();
        flows.push(("a fast sink", rows([-4.0, 0.0, 0.0, -3.0])));
        flows.push(("a fast spiral", rows([-1.0, -4.0, 4.0, -1.0])));
        flows
    }

    /// The streaks fill the picture evenly, whatever the flow: over a while, each of
    /// sixteen parts of it has about its share of them (as born at home with seconds
    /// to live, a sink's all ended in the middle). And a particle that is not among
    /// the fastest stays within reach of its home.
    #[test]
    fn the_streaks_fill_the_picture_whatever_the_flow() {
        let (lo, hi) = (Vec2::new(-4.0, -4.6), Vec2::new(6.8, 4.6));
        for (name, a) in flows() {
            let mut flow = Flow::new(None);
            let mut parts = [0.0f64; 16];
            let mut homes: Vec<Option<Vec2>> = vec![None; GRID * GRID];
            let reach = REACH * (hi - lo).min_element();
            for frame in 0..900 {
                flow.advance(a, 1.0 / 60.0, lo, hi, streak_colour(), None);
                for i in 0..flow.pos.len() {
                    // (the middle of its life is its home)
                    if homes[i].is_none() && flow.age[i] >= 0.5 * flow.life[i] {
                        homes[i] = Some(flow.pos[i]);
                    }
                    if flow.age[i] < 1.0 / 30.0 {
                        homes[i] = None;
                    }
                    if let (Some(home), true) = (homes[i], flow.life[i] > 1.3 * LIFE_LEAST) {
                        assert!(flow.pos[i].distance(home) <= 1.1 * reach, "{name}: a particle {} from its home, with {reach} to go", flow.pos[i].distance(home));
                    }
                    let at = (flow.pos[i] - lo) / (hi - lo);
                    if frame >= 240 && at.cmpge(Vec2::ZERO).all() && at.cmplt(Vec2::ONE).all() {
                        // (as bright as it is drawn, but for what it loses by being still)
                        let alpha = flow.row[i].color.0[3] as f64 / flow.bright[i] as f64;
                        parts[(at.x * 4.0) as usize + 4 * (at.y * 4.0) as usize] += alpha;
                    }
                }
            }
            let mean = parts.iter().sum::<f64>() / 16.0;
            for (k, part) in parts.iter().enumerate() {
                assert!((0.6..=1.45).contains(&(part / mean)), "{name}: part {k} of the picture has {:.2} of its share of the streaks", part / mean);
            }
        }
    }

    /// No streak is drawn from where a particle died to where it is born: between two
    /// rows that both hold a point of a streak, the particle has not been born again.
    /// (The rows as the GPU has them: a row written again replaces the newest.)
    #[test]
    fn a_streak_never_joins_two_lives() {
        for (name, a) in flows() {
            for dt in [1.0 / 60.0, 1.0 / 144.0, 1.0 / 24.0, 0.11] {
                let mut flow = Flow::new(None);
                // each particle's last rows: (drawn, how many lives it has had)
                let mut rows_of: Vec<Vec<(bool, u32)>> = vec![Vec::new(); GRID * GRID];
                let mut lives = vec![0u32; GRID * GRID];
                let (lo, hi) = (Vec2::splat(-4.0), Vec2::splat(4.0));
                if flow.pos.is_empty() {
                    flow.scatter(a, lo, hi);
                }
                for _ in 0..600 {
                    // one frame, as `advance` goes through it, noting every row written
                    let mut left = dt;
                    let mut steps = Vec::new();
                    let mut owed = flow.owed;
                    while left >= ROW_EVERY - owed {
                        steps.push((ROW_EVERY - owed, true));
                        left -= ROW_EVERY - owed;
                        owed = 0.0;
                    }
                    if left > 1e-6 {
                        steps.push((left, false));
                    }
                    for (h, whole) in steps {
                        let before = flow.age.clone();
                        flow.carry(a, h, lo, hi);
                        let new = flow.write(streak_colour(), None);
                        (flow.whole, flow.owed) = (whole, if whole { 0.0 } else { flow.owed + h });
                        for i in 0..flow.pos.len() {
                            lives[i] += (flow.age[i] < before[i]) as u32;
                            let entry = (flow.row[i].width > 0.0, lives[i]);
                            match rows_of[i].last_mut() {
                                Some(last) if !new => *last = entry,
                                _ => rows_of[i].push(entry),
                            }
                            let n = rows_of[i].len();
                            if n >= 2 && rows_of[i][n - 1].0 && rows_of[i][n - 2].0 {
                                assert!(rows_of[i][n - 1].1 == rows_of[i][n - 2].1, "{name}, {dt}: a streak drawn from one life of particle {i} to the next");
                            }
                        }
                    }
                }
                // (and the streaks are drawn at all: most rows hold a point)
                let drawn = rows_of.iter().flatten().filter(|r| r.0).count() as f64 / rows_of.iter().map(Vec::len).sum::<usize>() as f64;
                assert!(drawn > 0.5, "{name}, {dt}: only {drawn:.2} of the rows are drawn");
            }
        }
    }

    #[test]
    fn the_layout_fits_every_canvas() {
        for (w, h) in [(1080.0, 1000.0), (580.0, 700.0), (390.0, 800.0), (390.0, 560.0), (800.0, 600.0), (1600.0, 1000.0), (240.0, 180.0), (620.0, 480.0)] {
            for inset in [0.0, 46.0] {
                for series in [false, true] {
                    let view = Vec2::new(w, h);
                    let lay = Layout::of(view, inset, series);
                    let (min, max) = lay.picture;
                    assert!(min.cmpge(Vec2::ZERO).all() && max.cmple(view).all() && (max - min).min_element() > 50.0, "{w}x{h}: {lay:?}");
                    if let Some(p) = &lay.panel {
                        let within = |q: Vec2| q.cmpge(p.min - Vec2::splat(0.5)).all() && q.cmple(p.max + Vec2::splat(0.5)).all();
                        assert!(p.min.cmpge(Vec2::ZERO).all() && p.max.cmple(view + Vec2::splat(0.5)).all(), "{w}x{h}: {p:?}");
                        assert!(within(p.map.0) && within(p.map.1) && within(p.play.0) && within(p.play.1) && within(p.bar.0) && p.bar.1.x > p.bar.0.x + 60.0, "{w}x{h}: {p:?}");
                        assert!(p.min.y >= inset || lay.narrow, "{w}x{h}: under the buttons");
                    }
                }
            }
        }
    }
}

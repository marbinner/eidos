//! Symmetric and skew-symmetric parts: every square matrix is, in exactly one way, a
//! symmetric matrix plus a skew-symmetric one, A = S + K. Read A as a velocity field
//! (the velocity of the point x is Ax, as in the scene on the matrix exponential), and
//! S is a stretch along two axes at right angles while K spins every point at one
//! rate: the velocity at every point is the sum of the two.

use super::*;
use glam::{DMat2, DVec2};
use std::f64::consts::{PI as PI64, TAU as TAU64};

pub const DEF: SceneDef = SceneDef {
    id: "symskew",
    title: "Symmetric and skew-symmetric parts",
    group: SPECIAL,
    blurb: "Every square matrix is a stretch plus a spin: its symmetric and skew-symmetric parts, and what each does.",
    make: |g| Box::new(SymSkew::new(Some(g))),
};

// ----- the arithmetic, in double precision -------------------------------------------

/// The largest entry of A either way, as typed; and the largest stretch or spin rate
/// the sliders of the parts reach (with all of them at an end, no entry is beyond 4).
const ENTRY_MAX: f64 = 4.0;
const RATE_MAX: f64 = 2.0;
/// The pictures show the plane this far out from 0 each way.
const REACH: f64 = 2.5;
/// The longest time the yellow circle is carried for, and its radius: a small circle
/// about 0, which leaves room round it for x and its velocities.
const T_MAX: f64 = 2.0;
const RADIUS: f64 = 0.7;
/// What counts as exactly zero, for a number of the size of the entries (squared, for
/// a determinant).
const EXACT: f64 = 1e-9;

/// The colours of the three matrices: their numbers, their pictures' frames and their
/// velocities. A's is the colour of a velocity (as in the scene on flows).
const A_COL: Color = pal::CYAN;
const S_COL: Color = pal::GREEN;
const K_COL: Color = pal::PINK;
/// The colours of the eigenlines: the larger eigenvalue's, and the smaller's.
const EIG: [Color; 2] = [pal::ORANGE, pal::PURPLE];

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

/// The sum of the squares of the entries: the squared length of the matrix taken as a
/// list of four numbers.
fn squares(m: DMat2) -> f64 {
    m.to_cols_array().iter().map(|v| v * v).sum()
}

/// J, the quarter turn: it turns every arrow a right angle anticlockwise.
fn quarter_turn() -> DMat2 {
    rows([0.0, -1.0, 1.0, 0.0])
}

/// The two parts of a matrix, and what they are made of.
#[derive(Clone, Copy, Debug)]
struct Parts {
    /// S = (A + Aᵀ)/2 and K = (A − Aᵀ)/2.
    s: DMat2,
    k: DMat2,
    /// The spin rate: K = wJ, w = (a₂₁ − a₁₂)/2.
    w: f64,
    /// The stretch rates, S's eigenvalues, the larger first.
    s1: f64,
    s2: f64,
    /// The direction of the axis of s₁ (none when s₁ = s₂: then every direction is one).
    axis: Option<DVec2>,
}

impl Parts {
    /// Half the spread of the stretch rates, (s₁ − s₂)/2: as fast as the stretch turns
    /// any direction (see `turning`).
    fn half_spread(&self) -> f64 {
        0.5 * (self.s1 - self.s2)
    }
}

/// A split into its symmetric part and its skew-symmetric part.
fn split(a: DMat2) -> Parts {
    let t = a.transpose();
    let (s, k) = ((a + t) * 0.5, (a - t) * 0.5);
    let mean = 0.5 * trace(s);
    // S = mean·I + half·[[cos 2φ, sin 2φ], [sin 2φ, −cos 2φ]], φ the angle of s₁'s axis
    let (p, q) = (0.5 * (s.x_axis.x - s.y_axis.y), s.y_axis.x);
    let half = p.hypot(q);
    let axis = (half > EXACT * size(a).max(1e-12)).then(|| DVec2::from_angle(0.5 * q.atan2(p)));
    Parts { s, k, w: k.x_axis.y, s1: mean + half, s2: mean - half, axis }
}

/// The matrix whose symmetric part stretches at the rates s₁ and s₂, s₁'s axis at the
/// angle `angle` (radians), and whose skew part spins at the rate w.
fn compose(s1: f64, s2: f64, angle: f64, w: f64) -> DMat2 {
    let (m, h) = (0.5 * (s1 + s2), 0.5 * (s1 - s2));
    let (sin, cos) = (2.0 * angle).sin_cos();
    rows([m + h * cos, h * sin - w, h * sin + w, m - h * cos])
}

/// How fast the flow of `m` turns the direction at the angle θ (radians a unit of time,
/// anticlockwise): the velocity's part across the unit arrow u there, (Ju)·(Mu). For
/// K = wJ it is w for every θ; for S it is −((s₁ − s₂)/2) sin 2(θ − φ), towards s₁'s axis.
fn turning(m: DMat2, theta: f64) -> f64 {
    let u = DVec2::from_angle(theta);
    u.perp().dot(m * u)
}

/// (tr A / 2)² − det A, a quarter of the discriminant of λ² − (tr A) λ + det A:
/// positive for two real eigenvalues, negative for a complex pair, zero for a double
/// one. Worked out as ((a − d)/2)² + bc, so that a double eigenvalue gives 0 exactly.
/// (For A = S + wJ it is ((s₁ − s₂)/2)² − w².)
fn quarter_disc(m: DMat2) -> f64 {
    let half = 0.5 * (m.x_axis.x - m.y_axis.y);
    half * half + m.y_axis.x * m.x_axis.y
}

/// e^(Mt), in closed form (as in the scene on flows). M = sI + B with s = tr M / 2,
/// and B squares to qI (q = `quarter_disc`), so e^(Mt) = e^(st) (c I + g B), with
/// c = cosh(√q t), g = sinh(√q t)/√q for q > 0, cos and sin of √−q t for q < 0; near
/// q t² = 0 the three are one power series, summed as one.
fn expm(m: DMat2, t: f64) -> DMat2 {
    let s = 0.5 * trace(m);
    let b = m - DMat2::IDENTITY * s;
    let q = quarter_disc(m);
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

/// The eigenvalues of a matrix and the lines it keeps.
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

/// What kind of flow a matrix makes (as the scene on flows names them).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Kind {
    Still,
    Shear,
    RestIn,
    RestOut,
    Saddle,
    Sink,
    Source,
    ImproperSink,
    ImproperSource,
    StarSink,
    StarSource,
    SpiralSink,
    SpiralSource,
    Centre,
}

fn kind(a: DMat2) -> Kind {
    let scale = size(a);
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
    fn name(self) -> &'static str {
        match self {
            Kind::Still => "no flow",
            Kind::Shear => "a shear flow",
            Kind::RestIn | Kind::RestOut => "a line at rest",
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
}

/// The eigenvalues as a formula: "$\lambda = 1.00$ and $−1.00$", "$\lambda = −0.25 \pm 1.50i$".
fn eigen_text(e: Eigen) -> String {
    match e {
        Eigen::Real { l1, lines: Lines::One(_) | Lines::Every, .. } => format!(r"$\lambda = {}$ twice", num(l1, 2)),
        Eigen::Real { l1, l2, .. } => format!(r"$\lambda = {}$ and ${}$", num(l1, 2), num(l2, 2)),
        Eigen::Complex { re, im } if re.abs() < 0.005 => format!(r"$\lambda = \pm {}i$", num(im, 2)),
        Eigen::Complex { re, im } => format!(r"$\lambda = {} \pm {}i$", num(re, 2), num(im, 2)),
    }
}

/// Which part wins: the spin (A's eigenvalues are complex, and the flow turns), the
/// stretch (two real eigenvalues, two lines kept), or neither (one eigenvalue twice).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Verdict {
    Spin,
    Stretch,
    Tie,
}

fn verdict(a: DMat2) -> Verdict {
    match eigen(a) {
        Eigen::Complex { .. } => Verdict::Spin,
        Eigen::Real { l1, l2, .. } if l1 == l2 => Verdict::Tie,
        Eigen::Real { .. } => Verdict::Stretch,
    }
}

/// Two numbers written to as many decimals (two to four) as tell them apart.
fn apart(u: f64, v: f64) -> (String, String) {
    for d in 2..=4 {
        let (p, q) = (num(u, d), num(v, d));
        if p != q {
            return (p, q);
        }
    }
    (num(u, 2), num(v, 2))
}

/// The spin against the stretch, as a formula: "$|w| = 1.20 > (s_1 − s_2)/2 = 1.00$".
fn contest(a: DMat2) -> String {
    let p = split(a);
    let (w, h) = (p.w.abs(), p.half_spread());
    let (ws, hs) = apart(w, h);
    match verdict(a) {
        Verdict::Tie => format!(r"$|w| = (s_1 - s_2)/2 = {}$", num(h, 2)),
        // (the two differ beyond the fourth decimal: say so without a false "=")
        _ if ws == hs => format!(r"$|w| \approx (s_1 - s_2)/2 = {hs}$"),
        Verdict::Spin => format!(r"$|w| = {ws} > (s_1 - s_2)/2 = {hs}$"),
        Verdict::Stretch => format!(r"$|w| = {ws} < (s_1 - s_2)/2 = {hs}$"),
    }
}

/// A number for a matrix on the screen, as a formula, in five characters at most (so
/// that the brackets never move as the numbers change).
fn cell(v: f64) -> String {
    let a = v.abs();
    let number = if !v.is_finite() {
        "?".to_string()
    } else if a < 9.995 {
        num(v, 2)
    } else {
        num(v, 1)
    };
    format!("${number}$")
}

/// The widest `cell` can be, in characters of its number (a minus sign counted as two).
const CELL: usize = 6;

// ----- experiments and quick choices -------------------------------------------------

/// The matrix the scene opens with: a stretch at the rates 0.8 and −1.2 along axes
/// tilted by 26.6° (the direction (2, 1)), and a spin w = 1.2 that beats it.
const START: [f64; 4] = [0.4, -0.4, 2.0, -0.8];
/// The point x it opens with, 1.25 from 0.
const START_X: [f64; 2] = [1.0, 0.75];
/// How long the yellow circle is carried for, to begin with.
const START_T: f64 = 0.5;

/// What plays once an experiment's setting is in place.
#[derive(Clone, Copy, Debug, PartialEq)]
enum Act {
    Nothing,
    /// x goes once round the circle about 0 it is on.
    Round,
    /// A changes into its transpose.
    Transpose,
    /// The spin is turned up from 0 through `SPIN_STOPS`, S held as it is.
    Spin,
}

struct Trial {
    label: &'static str,
    look_for: &'static str,
    /// The matrix, by rows.
    a: [f64; 4],
    x: [f64; 2],
    t: f64,
    act: Act,
    shadows: bool,
}

const TRIALS: [Trial; 5] = [
    Trial {
        label: "a pure spin",
        look_for: r"$S = 0$, so $A = K$: no stretch at all. Every point turns round $0$ at the same rate, $w = 0.8$ radians per unit of time, anticlockwise: a centre, whose paths are circles. The velocity $Kx$ is at right angles to $x$ (the corner at $x$), so $x \cdot Ax = 0$ at every $x$. In the time $t = 1$ the yellow circle turns into itself by 0.8 radians (46°) and keeps its area, since $\tr K = 0$. No line is kept: $\lambda = \pm 0.8i$.",
        a: [0.0, -0.8, 0.8, 0.0],
        x: [1.25, 0.5],
        t: 1.0,
        act: Act::Nothing,
        shadows: false,
    },
    Trial {
        label: "a shear: half stretch, half spin",
        look_for: r"$A = \begin{bmatrix} 0 & 1 \\ 0 & 0 \end{bmatrix}$ slides each point sideways, as fast as it is high. Its $S$ has $0.5$ above and below the diagonal: it stretches along the 45° line at the rate $s_1 = 0.5$ and squeezes along the line at right angles to it at $s_2 = -0.5$. Its $K$ turns everything clockwise at $0.5$ ($w = -0.5$). On the $x$ axis $Sx$ and $Kx$ cancel, and the axis stays put; on the $y$ axis they add up to the slide. Here $|w| = (s_1 - s_2)/2$ exactly: a tie, and $\lambda = 0$ twice, with one eigenline.",
        a: [0.0, 1.0, 0.0, 0.0],
        x: [0.6, 1.0],
        t: 0.8,
        act: Act::Nothing,
        shadows: false,
    },
    Trial {
        label: "the quadratic form ignores the spin",
        look_for: r"$x$ goes once round a circle about $0$. The dashed line from the tip of $Ax$, at right angles to the line of $x$, runs through the tip of $Sx$: the side between them is $Kx$, at right angles to $x$. So $Ax$ and $Sx$ cast the same shadow on the line of $x$, and $x \cdot Ax = x \cdot Sx$ at every $x$: largest, $1.25 = s_1 |x|^2$, where $x$ crosses $S$'s orange line, and least, $-1.88 = s_2 |x|^2$, on its purple line. $A$'s own eigenvalues are complex and play no part: the quadratic form belongs to $S$.",
        a: START,
        x: START_X,
        t: START_T,
        act: Act::Round,
        shadows: true,
    },
    Trial {
        label: "transpose: the spin turns the other way",
        look_for: r"$A$ changes into its transpose $A^\top$, its entries mirrored across the diagonal. $S$ does not change at all; $K$ becomes $-K$, the same spin the other way round: $A^\top = S - K$. The flow of $A$ spiralled in anticlockwise; that of $A^\top$ spirals in clockwise, with the same eigenvalues, $-0.20 \pm 0.66i$ (a matrix and its transpose have the same trace and determinant). Half-way, where $w$ passes through $0$, $A$ is $S$ for a moment, with its eigenlines at right angles.",
        a: START,
        x: START_X,
        t: START_T,
        act: Act::Transpose,
        shadows: false,
    },
    Trial {
        label: "stretch against spin",
        look_for: r"$S$ stays as it is, with stretch rates $s_1 = 0.4$ and $s_2 = -1.6$, and the spin $w$ is turned up from $0$. At first $K = 0$: a pure stretch, $A = S$, a saddle with its eigenlines at right angles. As $w$ grows, $A$'s eigenlines swing towards each other: past $w = 0.8$ both eigenvalues are negative and the saddle has become a sink; at $w = 1 = (s_1 - s_2)/2$ the two lines meet; beyond that no line is kept, and the flow spirals in. In the chart (on a wide screen), $A$'s curve, $K$'s level line plus $S$'s wave, meets $0$ twice, then only touches it, then stays above it. In general $A$'s eigenvalues are complex exactly when $|w| > (s_1 - s_2)/2$.",
        a: [0.0, 0.8, 0.8, -1.2],
        x: [1.0, 0.6],
        t: START_T,
        act: Act::Spin,
        shadows: false,
    },
];

/// The spin rates the last experiment stops at: a pure stretch, a sink, the two lines
/// met, a spiral.
const SPIN_STOPS: [f64; 4] = [0.0, 0.9, 1.0, 1.4];
/// Seconds it rests at each, and takes from one to the next.
const SPIN_STAY: f32 = 1.8;
const SPIN_LEG: f32 = 2.2;
/// Seconds x takes to go round, and A to become its transpose.
const ROUND_SECS: f32 = 9.0;
const TRANSPOSE_SECS: f32 = 3.5;

fn spin_length() -> f32 {
    (SPIN_STOPS.len() - 1) as f32 * (SPIN_STAY + SPIN_LEG) + SPIN_STAY
}

/// The spin rate `s` seconds into the last experiment.
fn spin_at(s: f32) -> f64 {
    let per = SPIN_STAY + SPIN_LEG;
    let k = ((s.max(0.0) / per).floor() as usize).min(SPIN_STOPS.len() - 1);
    let into = s - k as f32 * per;
    if k + 1 >= SPIN_STOPS.len() || into <= SPIN_STAY {
        return SPIN_STOPS[k];
    }
    let u = smoothstep((into - SPIN_STAY) / SPIN_LEG) as f64;
    SPIN_STOPS[k] + (SPIN_STOPS[k + 1] - SPIN_STOPS[k]) * u
}

/// Quick choices of A: (button, the matrix by rows, what it shows).
const PRESETS: [(&str, [f64; 4], &str); 5] = [
    ("start", START, r"Stretch rates $0.8$ and $-1.2$ along axes tilted by 26.6°, and a spin $w = 1.2$ that beats them: a spiral sink."),
    ("symmetric", [1.0, 0.5, 0.5, -0.5], r"$A^\top = A$: $K = 0$ and $A = S$, a stretch along two lines at right angles."),
    ("skew", [0.0, -1.0, 1.0, 0.0], r"$A^\top = -A$: $S = 0$ and $A = K$, a pure spin. Every point goes round a circle, at the rate $1$."),
    ("shear", [0.0, 1.0, 0.0, 0.0], "Half stretch, half spin: on the $x$ axis they cancel, and the axis stays put."),
    (
        "even squeeze",
        [-0.3, -1.0, 1.0, -0.3],
        r"$S = -0.3I$ squeezes every direction alike, so it has no axes of its own and every line is an eigenline of $S$. Then $S$ and $K$ commute, and the flow of $A$ for a time $t$ is the flow of $S$ and then that of $K$: a turn by $t$, shrunk by $e^{-0.3t}$.",
    ),
];

// ----- the streaks ---------------------------------------------------------------------

/// The grid of cells the streaks of one picture are at home in, this many each way.
const GRID: usize = 15;
/// A row of every streak is laid each so much scene time, and between rows the newest
/// row follows the particles (`TrailSet::follow`), as in the scene on flows.
const ROW_EVERY: f32 = 1.0 / 20.0;
const ROWS: u32 = 8;
/// The longest stretch of scene time a frame's streaks move through.
const MOST_ROWS: f32 = 4.0;
/// How long a particle lives at most and at least, and how far it goes in its life (a
/// share of the picture's side).
const LIFE_MOST: f32 = 3.5;
const LIFE_LEAST: f32 = 0.35;
const GOES: f32 = 0.2;
/// How long a particle takes to fade in and out.
const FADE_IN: f32 = 0.5;
const FADE_OUT: f32 = 0.7;
/// How bright a particle is that hardly moves, against one that does; and how far
/// (pixels of a picture a thousand high, in its whole life) counts as moving.
const STILL: f32 = 0.35;
const MOVING_PX: f32 = 24.0;

/// The faint streaks of one picture: particles carried by its flow, each with a short
/// tail. (The particles of the scene on flows, kept to a square: each is born up the
/// flow from a home of its own in a grid over the square, passes through it in mid-life
/// and dies as far down the flow, so that the streaks cover the square evenly whatever
/// the flow; outside the square a streak is not drawn.)
struct Flow {
    /// Where the particles are, in the picture's own units.
    pos: Vec<Vec2>,
    age: Vec<f32>,
    life: Vec<f32>,
    bright: Vec<f32>,
    fresh: Vec<bool>,
    cut: Vec<bool>,
    row: Vec<LinePoint>,
    /// (None in the tests, which have no graphics device.)
    trails: Option<TrailSet>,
    owed: f32,
    whole: bool,
    rng: Rng,
}

impl Flow {
    fn new(gfx: Option<&Gfx>, seed: u64) -> Flow {
        let trails = gfx.map(|g| {
            let mut trails = TrailSet::new(g, (GRID * GRID) as u32, ROWS, true);
            trails.style.intensity = 1.0;
            trails.style.fade_power = 1.3;
            trails.style.taper = 0.6;
            trails
        });
        Flow { pos: Vec::new(), age: Vec::new(), life: Vec::new(), bright: Vec::new(), fresh: Vec::new(), cut: Vec::new(), row: Vec::new(), trails, owed: 0.0, whole: false, rng: Rng::new(seed) }
    }

    /// Particle `i` born again, to pass its home (a random place in its cell of the
    /// square) in the middle of a life long enough to go `GOES` of the square's side.
    fn born(&mut self, i: usize, a: DMat2) {
        let (lo, hi) = (Vec2::splat(-REACH as f32), Vec2::splat(REACH as f32));
        let cell = Vec2::new((i % GRID) as f32, (i / GRID % GRID) as f32);
        let at = (cell + Vec2::new(self.rng.f32(), self.rng.f32())) / GRID as f32;
        let home = (lo + (hi - lo) * at).as_dvec2();
        let side = 2.0 * REACH;
        let reach = GOES as f64 * side;
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
        let moved = ((to - from).length() / side) as f32 * 1000.0;
        self.pos[i] = from.as_vec2();
        (self.age[i], self.life[i], self.fresh[i]) = (0.0, life as f32, true);
        self.bright[i] = STILL + (1.0 - STILL) * smoothstep(moved / MOVING_PX);
    }

    /// A new set, each part of the way through its life.
    fn scatter(&mut self, a: DMat2) {
        let n = GRID * GRID;
        self.pos = vec![Vec2::ZERO; n];
        (self.age, self.life, self.bright) = (vec![0.0; n], vec![1.0; n], vec![1.0; n]);
        (self.fresh, self.cut) = (vec![true; n], vec![false; n]);
        self.row = vec![crate::gfx::draw::BREAK; n];
        for i in 0..n {
            self.born(i, a);
            let age = self.rng.range(0.0, self.life[i]);
            self.pos[i] = (expm(a, age as f64) * self.pos[i].as_dvec2()).as_vec2();
            self.age[i] = age;
        }
        (self.owed, self.whole) = (0.0, false);
        if let Some(trails) = &mut self.trails {
            trails.clear();
        }
    }

    /// Carry every particle along the flow for the time `h`; those that have lived
    /// their time, or gone far off, are born again.
    fn carry(&mut self, a: DMat2, h: f32) {
        let m = expm(a, h as f64).as_mat2();
        let far = 1e3 * REACH as f32;
        for i in 0..self.pos.len() {
            let p = m * self.pos[i];
            self.age[i] += h;
            if self.age[i] >= self.life[i] || !p.is_finite() || p.abs().max_element() > far {
                self.born(i, a);
            } else {
                self.pos[i] = p;
            }
        }
    }

    /// Where the particles are, into the newest row (on the scene's plane, through
    /// `plane`): a new row if the one before was laid whole (returns true).
    fn write(&mut self, colour: Color, plane: Plane, gfx: Option<&Gfx>) -> bool {
        let new = std::mem::take(&mut self.whole);
        let edge = REACH as f32;
        for i in 0..self.pos.len() {
            if new {
                self.cut[i] = false;
            }
            if std::mem::take(&mut self.fresh[i]) {
                self.cut[i] = true;
            }
            let (age, life) = (self.age[i], self.life[i]);
            let fade = smoothstep(age / FADE_IN.min(0.3 * life)) * smoothstep((life - age) / FADE_OUT.min(0.35 * life));
            let q = self.pos[i];
            // (a negative width leaves a gap: none is drawn outside the square)
            let outside = q.abs().max_element() > edge;
            let w = plane.at(q.as_dvec2());
            self.row[i] = LinePoint { pos: [w.x, w.y, 0.0], width: if self.cut[i] || outside { -1.0 } else { 1.4 }, color: colour.a(0.5 * fade * self.bright[i]) };
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

    /// Move the streaks on by `dt` of scene time along the flow of `a`.
    fn advance(&mut self, a: DMat2, dt: f32, colour: Color, plane: Plane, gfx: Option<&Gfx>) {
        if self.pos.is_empty() {
            self.scatter(a);
        }
        let mut left = if dt.is_finite() { dt.clamp(0.0, MOST_ROWS * ROW_EVERY) } else { 0.0 };
        let mut written = false;
        loop {
            let to_row = ROW_EVERY - self.owed;
            if left < to_row {
                break;
            }
            self.carry(a, to_row);
            self.write(colour, plane, gfx);
            (self.whole, self.owed, written) = (true, 0.0, true);
            left -= to_row;
        }
        if left > 1e-6 {
            self.carry(a, left);
            self.write(colour, plane, gfx);
            self.owed += left;
            written = true;
        }
        if !written && self.trails.as_ref().is_some_and(|t| !t.started()) {
            self.write(colour, plane, gfx);
        }
        if let Some(trails) = &mut self.trails {
            trails.follow(if self.whole { 1.0 } else { self.owed / ROW_EVERY });
        }
    }

    /// Forget the streaks' history (the pictures have moved on the canvas), keeping the
    /// particles.
    fn forget(&mut self) {
        if let Some(trails) = &mut self.trails {
            trails.clear();
        }
        self.cut.iter_mut().for_each(|c| *c = true);
    }
}

// ----- where things go ---------------------------------------------------------------

/// One picture: a square on the scene's plane, showing the plane of x out to `REACH`
/// each way from 0.
#[derive(Clone, Copy, Debug, PartialEq)]
struct Plane {
    centre: Vec2,
    half: f32,
}

impl Plane {
    /// Units of the scene per unit of the picture.
    fn scale(&self) -> f32 {
        self.half / REACH as f32
    }

    fn at(&self, p: DVec2) -> Vec2 {
        self.centre + p.as_vec2() * self.scale()
    }

    fn of(&self, w: Vec2) -> DVec2 {
        ((w - self.centre) / self.scale().max(1e-9)).as_dvec2()
    }

    fn lo(&self) -> Vec2 {
        self.centre - Vec2::splat(self.half)
    }

    fn hi(&self) -> Vec2 {
        self.centre + Vec2::splat(self.half)
    }
}

/// Where the three pictures go, in the scene's units (the stage: five high, centred on
/// the origin), with the sizes of text that fit them.
#[derive(Clone, Copy, Debug, PartialEq)]
struct Layout {
    /// A phone held upright: A's picture over the other two, the equation on top.
    narrow: bool,
    /// The pictures of A, S and K.
    planes: [Plane; 3],
    /// Text sizes: of words, and of the matrices' numbers.
    text: f32,
    numbers: f32,
    /// Whether there is room for words at all.
    words: bool,
    /// The chart of how fast each flow turns each line (its corners), where there is
    /// room for it: under the pictures on a canvas wide and tall enough.
    chart: Option<(Vec2, Vec2)>,
}

/// The chart's height on the screen: at most, and at least (or none at all).
const CHART_MOST: f32 = 170.0;
const CHART_LEAST: f32 = 80.0;

impl Layout {
    fn line(&self) -> f32 {
        self.text + 6.0
    }

    fn of(view: Vec2, inset: f32) -> Layout {
        let view = view.max(Vec2::ONE);
        // (as `STAGE_VIEW` is fitted to the canvas)
        let ppu = (view.y / STAGE_H).min(view.x / 2.0).max(1e-3);
        let world = |p: Vec2| Vec2::new((p.x - view.x * 0.5) / ppu, (view.y * 0.5 - p.y) / ppu);
        let narrow = view.x < 0.75 * view.y;
        let small = view.x < 640.0 || view.y < 560.0;
        let text = if small { 11.5 } else { 13.5 };
        let numbers = if small { 11.0 } else { 14.0 };
        let line = text + 6.0;
        let mat_h = matrix_size(2, 2, CELL, numbers).y;
        let margin = if small { 10.0 } else { 16.0 };
        let top = inset.max(0.0) + margin;
        let bottom = view.y - margin;
        let mut squares = [(Vec2::ZERO, 10.0f32); 3];
        let mut chart = None;
        let bare;
        if narrow {
            // the equation on top; A's picture under it; S's and K's side by side
            let gap = 14.0;
            let w = (view.x - 2.0 * margin).max(20.0);
            let fixed = mat_h + 10.0 + line + 6.0 + line + 8.0 + line + 2.0 * line + 6.0 + 2.0 * line;
            // (on a short canvas both rows give up height, A's picture staying the larger)
            let room = bottom - top - fixed;
            let pair = ((w - gap) * 0.5).min(room * 0.4).max(10.0);
            let a_side = w.min(room - pair).max(10.0);
            bare = a_side < 90.0;
            let mut y = top + mat_h + 10.0 + line;
            squares[0] = (Vec2::new((view.x - a_side) * 0.5, y), a_side);
            y += a_side + 6.0 + line + 8.0 + line;
            let left = (view.x - 2.0 * pair - gap) * 0.5;
            squares[1] = (Vec2::new(left, y), pair);
            squares[2] = (Vec2::new(left + pair + gap, y), pair);
        } else {
            // side by side, each under its matrix: A = S + K
            let gap = if small { 30.0 } else { 44.0 };
            let head = line + mat_h + 10.0;
            let foot = 6.0 + 2.0 * line + 10.0 + 3.0 * line;
            // (the chart's own rows: its title above it, its angles under it)
            let chart_rows = line + 4.0 + line + 10.0;
            let side = ((view.x - 2.0 * margin - 2.0 * gap) / 3.0).min(bottom - top - head - foot).max(10.0);
            bare = side < 70.0;
            let spare = bottom - top - head - side - foot - chart_rows;
            let chart_h = (spare >= CHART_LEAST && !small).then_some(spare.min(CHART_MOST));
            let block = head + side + foot + chart_h.map_or(0.0, |h| h + chart_rows);
            let y = top + ((bottom - top - block) * 0.5).max(0.0) + head;
            let left = (view.x - 3.0 * side - 2.0 * gap) * 0.5;
            for (i, square) in squares.iter_mut().enumerate() {
                *square = (Vec2::new(left + i as f32 * (side + gap), y), side);
            }
            if let Some(h) = chart_h {
                let cy = y + side + 6.0 + 2.0 * line + 10.0 + line + 4.0;
                // (room at its left for the numbers of its scale)
                let (p, q) = (world(Vec2::new(left + 34.0, cy)), world(Vec2::new(left + 3.0 * side + 2.0 * gap, cy + h)));
                chart = Some((p.min(q), p.max(q)));
            }
        }
        if bare {
            // (a canvas with no room for the words round the pictures: the three
            // pictures alone, as large as they go, side by side)
            let gap = 8.0;
            let side = ((view.x - 2.0 * margin - 2.0 * gap) / 3.0).min(bottom - top).max(10.0);
            let (left, y) = ((view.x - 3.0 * side - 2.0 * gap) * 0.5, top + ((bottom - top - side) * 0.5).max(0.0));
            for (i, square) in squares.iter_mut().enumerate() {
                *square = (Vec2::new(left + i as f32 * (side + gap), y), side);
            }
            chart = None;
        }
        let plane = |(min, side): (Vec2, f32)| Plane { centre: world(min + Vec2::splat(side * 0.5)), half: side * 0.5 / ppu };
        let words = !bare && squares[1].1 >= 100.0 && view.x >= 240.0;
        Layout { narrow, planes: squares.map(plane), text, numbers, words, chart }
    }
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
    (t0 <= t1).then(|| (if t0 <= 0.0 { a } else { a + d * t0 }, if t1 >= 1.0 { b } else { a + d * t1 }))
}

/// The stretches of a path that lie in the box `lo..hi`.
fn runs(points: &[DVec2], lo: DVec2, hi: DVec2) -> Vec<Vec<DVec2>> {
    let (mut out, mut run): (Vec<Vec<DVec2>>, Vec<DVec2>) = (Vec::new(), Vec::new());
    for pair in points.windows(2) {
        match clip(pair[0], pair[1], lo, hi) {
            Some((p, q)) => {
                if p != pair[0] && run.len() >= 2 {
                    out.push(std::mem::take(&mut run));
                }
                if run.is_empty() {
                    run.push(p);
                }
                run.push(q);
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

/// The square of the picture, in its own units.
fn bounds() -> (DVec2, DVec2) {
    (DVec2::splat(-REACH), DVec2::splat(REACH))
}

fn in_square(p: DVec2) -> bool {
    p.is_finite() && p.abs().max_element() <= REACH
}

/// The parts of the picture drawn as lines.
#[derive(Clone, Copy, PartialEq)]
enum Part {
    /// What keeps its number of points from frame to frame (the usual batch).
    Fixed,
    /// The velocities and the parallelogram, which are cut at the square's edge.
    Arrows,
    /// The yellow circle's image and the paths of its points, cut likewise.
    Carried,
    /// The eigenlines and their dashes, which come and go.
    Eigen,
    /// The shadows on the line of x, when shown.
    Shadow,
    /// The small arrows of the velocity field, when shown.
    Field,
    /// The corners that mark right angles, which come and go.
    Corners,
}

/// The line batch a part is drawn in: one of its own for a part whose number of points
/// changes as the picture moves (different from the usual one by a hair of its width),
/// so that `eidos motion` can still compare the parts that stay the same point for
/// point. `dashed` gives the dashed kind.
fn lines_for<'a>(f: &'a mut Frame, part: Part, dashed: bool) -> &'a mut crate::gfx::draw::LineBatch {
    if part == Part::Fixed && !dashed {
        return &mut f.dl.lines;
    }
    let (dash, duty) = if dashed { (9.0, 0.55) } else { (0.0, 0.5) };
    f.dl.lines_with(LineKind::Blend, LineStyle { width_scale: 1.0 + 1e-4 * part as u8 as f32, dash, duty, ..Default::default() })
}

/// An arrow from `a` to `b` in the picture `plane` (its own units), the part of it in
/// the square: without its head if the head is outside.
fn arrow_in(f: &mut Frame, plane: Plane, part: Part, a: DVec2, b: DVec2, width: f32, c: Color) {
    let (lo, hi) = bounds();
    let Some((p, q)) = clip(a, b, lo, hi) else { return };
    let whole = q == b;
    let (p, q) = (plane.at(p).extend(0.0), plane.at(q).extend(0.0));
    let full = (plane.at(b) - plane.at(a)).length();
    let d = q - p;
    let len = d.length();
    if len < 1e-9 || full < 1e-9 {
        return;
    }
    let dir = d / len;
    // (the head's size goes by the whole arrow, and the shaft stops short of where the
    // head would be also when the head is cut off at the square's edge: so the shaft's
    // end moves on smoothly as the tip crosses the edge)
    let wpp = f.cam.world_per_px(q) * f.px;
    let head = ((width * 3.2 + 5.0) * wpp).min(full * 0.6);
    if whole {
        let side = dir.cross(Vec3::Z).normalize_or_zero() * (head * 0.42);
        let base = q - dir * head;
        f.dl.tris_alpha.tri(q, base + side, base - side, c);
    }
    let end = q - dir * (head * 0.8).min(len);
    lines_for(f, part, false).segment(p, end, width, c);
}

/// The small square corner that marks a right angle at `at` between the directions
/// `d1` and `d2` (in the scene), sides `size` long, in a batch of its own.
fn corner(f: &mut Frame, at: Vec2, d1: Vec2, d2: Vec2, size: f32, color: Color) {
    let (a, b) = (d1.normalize_or_zero() * size, d2.normalize_or_zero() * size);
    lines_for(f, Part::Corners, false).polyline([at + a, at + a + b, at + b].map(|p| p.extend(0.0)), 1.4, color);
}

/// A segment of the picture (its own units), cut at the square's edge.
#[allow(clippy::too_many_arguments)]
fn segment_in(f: &mut Frame, plane: Plane, part: Part, dashed: bool, a: DVec2, b: DVec2, width: f32, c: Color) {
    let (lo, hi) = bounds();
    if let Some((p, q)) = clip(a, b, lo, hi) {
        lines_for(f, part, dashed).segment(plane.at(p).extend(0.0), plane.at(q).extend(0.0), width, c);
    }
}

/// The labels of one picture, kept inside it: a `Room` walled off everywhere else.
fn room_in(f: &Frame, plane: Plane) -> Room {
    let mut room = Room::new(f);
    let (Some(p), Some(q)) = (on_screen(f, plane.lo().extend(0.0)), on_screen(f, plane.hi().extend(0.0))) else { return room };
    let (min, max) = (p.min(q), p.max(q));
    let big = f.view_size().max_element() * 4.0 + 1e4;
    room.wall(Vec2::new(-big, -big), Vec2::new(big, min.y));
    room.wall(Vec2::new(-big, max.y), Vec2::new(big, big));
    room.wall(Vec2::new(-big, min.y), Vec2::new(min.x, max.y));
    room.wall(Vec2::new(max.x, min.y), Vec2::new(big, max.y));
    room
}

/// A picture's square on the screen (logical pixels), if it is in view.
fn on_screen_rect(f: &Frame, plane: Plane) -> Option<(Vec2, Vec2)> {
    let (p, q) = (on_screen(f, plane.lo().extend(0.0))?, on_screen(f, plane.hi().extend(0.0))?);
    Some((p.min(q), p.max(q)))
}

/// How wide a label is: words by their number of characters, a text with formulas in
/// it exactly.
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

/// Where a label goes beside a line: through the scene's point `at` along `along` (in
/// the scene), far enough off it on the screen that the label's box clears it, first
/// on the side nearer to the screen direction `prefer`, then on the other side. (A
/// label is placed off a point by a fixed gap in each of eight directions; beside a
/// slanting line that is not enough, and its box would cross the line.)
fn beside(f: &Frame, at: Vec2, along: Vec2, prefer: Vec2, text: &str, size: f32, gap: f32) -> [Option<Vec3>; 2] {
    let Some(p) = on_screen(f, at.extend(0.0)) else { return [None, None] };
    let d = Vec2::new(along.x, -along.y).normalize_or_zero();
    let mut n = d.perp();
    if n.dot(prefer) < 0.0 {
        n = -n;
    }
    let half = f.text_size(text, size) * 0.5 + Vec2::new(4.0, 1.0);
    let off = n.x.abs() * half.x + n.y.abs() * half.y + gap;
    let back = |q: Vec2| f.cam.pick_ground(q * f.px).filter(|w| w.is_finite());
    [back(p + n * off), back(p - n * off)]
}

/// Where a label goes past the tip `tip` of an arrow along `along` (in the scene): on
/// the line of the arrow, far enough beyond the tip that its box clears it.
fn beyond(f: &Frame, tip: Vec2, along: Vec2, text: &str, size: f32, gap: f32) -> Option<Vec3> {
    let p = on_screen(f, tip.extend(0.0))?;
    let d = Vec2::new(along.x, -along.y).normalize_or_zero();
    let half = f.text_size(text, size) * 0.5 + Vec2::new(4.0, 1.0);
    let off = d.x.abs() * half.x + d.y.abs() * half.y + gap;
    f.cam.pick_ground((p + d * off) * f.px).filter(|w| w.is_finite())
}

/// A name at `centre` (in the scene), carried along by `anchor`, the point it names:
/// kept out of the room left for names placed after it, slid inside the picture if it
/// runs over its edge, and left out if its middle is not in the picture.
#[allow(clippy::too_many_arguments)]
fn put(f: &mut Frame, room: &mut Room, (name, number): (&'static str, u32), plane: Plane, anchor: Vec3, centre: Vec3, text: &str, size: f32, colour: Color) {
    if plane.of(centre.truncate()).abs().max_element() > REACH {
        return;
    }
    let (Some(a), Some(c), Some((min, max))) = (on_screen(f, anchor), on_screen(f, centre), on_screen_rect(f, plane)) else { return };
    let half = f.text_size(text, size) * 0.5;
    // (slid in, if need be, to lie wholly inside the picture)
    let (lo, hi) = (min + half + Vec2::splat(3.0), max - half - Vec2::splat(3.0));
    if lo.x > hi.x || lo.y > hi.y {
        return;
    }
    let c = c.max(lo).min(hi);
    room.boxed(c - half, c + half);
    f.text_placed(name, number, a, c - a, text, size, colour);
}

// ----- the scene -----------------------------------------------------------------------

/// What glides when a button changes it: the matrix, the point x and the time the
/// circle is carried for.
#[derive(Clone, Copy, Debug, PartialEq)]
struct Setting {
    a: DMat2,
    x: DVec2,
    t: f64,
}

impl Setting {
    fn numbers(&self) -> [f64; 7] {
        let a = self.a.to_cols_array();
        [a[0], a[1], a[2], a[3], self.x.x, self.x.y, self.t]
    }

    fn from_numbers(v: [f64; 7]) -> Setting {
        Setting { a: DMat2::from_cols_array(&[v[0], v[1], v[2], v[3]]), x: DVec2::new(v[4], v[5]), t: v[6] }
    }
}

/// How fast a change of setting is glided to: each of two eases one after the other
/// covers two thirds of what is left in this many seconds.
const FOLLOW: f64 = 0.09;

/// A setting as asked for, and as drawn: the drawn one follows it through `between`,
/// so that it sets off gently and arrives exactly (as in the scene on flows).
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
        if apart < 2e-4 || !apart.is_finite() {
            *self = Glide::at(self.to);
        } else {
            (self.between, self.shown) = (Setting::from_numbers(between), Setting::from_numbers(shown));
        }
    }
}

/// What is playing: seconds into it, and what it started from.
#[derive(Clone, Copy, Debug, PartialEq)]
enum Play {
    Idle,
    /// x once round the circle about 0 through where it started.
    Round(f32, DVec2),
    /// A becoming its transpose.
    Transpose(f32, DMat2),
    /// The spin turned up, S held.
    Spin(f32, DMat2),
}

/// The matrix with every entry within ±`ENTRY_MAX` (and 0 for one that is not a number).
fn within_limits(m: DMat2) -> DMat2 {
    DMat2::from_cols_array(&m.to_cols_array().map(|v| if v.is_finite() { v.max(-ENTRY_MAX).min(ENTRY_MAX) } else { 0.0 }))
}

pub struct SymSkew {
    glide: Glide,
    play: Play,
    /// Waiting for the glide to an experiment's setting to arrive before playing.
    waiting: bool,
    /// The shadows of Ax and Sx on the line of x.
    shadows: bool,
    /// The flow as small arrows rather than streaks.
    arrows: bool,
    eigenlines: bool,
    flows: [Flow; 3],
    /// The angle of s₁'s axis as last set (for the slider, while S has no axes of its own).
    angle: f64,
    /// The eigenlines of A and of S as drawn last frame: a line's name keeps its end.
    last_lines: [[Option<DVec2>; 2]; 2],
    /// Where the pictures were last frame (the streaks' history is let go when they move).
    laid: Option<[Plane; 3]>,
}

impl SymSkew {
    fn new(gfx: Option<&Gfx>) -> Self {
        let start = Setting { a: rows(START), x: DVec2::from(START_X), t: START_T };
        SymSkew {
            glide: Glide::at(start),
            play: Play::Idle,
            waiting: false,
            shadows: false,
            arrows: false,
            eigenlines: true,
            flows: [Flow::new(gfx, 5), Flow::new(gfx, 6), Flow::new(gfx, 7)],
            angle: split(rows(START)).axis.map_or(0.0, |v| v.y.atan2(v.x)),
            last_lines: [[None; 2]; 2],
            laid: None,
        }
    }

    /// Set up experiment `k`: glide to its setting, then play what it plays.
    fn set_up(&mut self, k: usize) {
        let trial = &TRIALS[k];
        let (a, x) = (rows(trial.a), DVec2::from(trial.x));
        self.glide.to = Setting { a, x, t: trial.t };
        self.play = match trial.act {
            Act::Nothing => Play::Idle,
            Act::Round => Play::Round(0.0, x),
            Act::Transpose => Play::Transpose(0.0, a),
            Act::Spin => Play::Spin(0.0, a),
        };
        self.waiting = true;
        (self.shadows, self.arrows, self.eigenlines) = (trial.shadows, false, true);
    }

    /// Ask for a new matrix by a button or a typed number: the picture glides to it.
    fn aim_matrix(&mut self, a: DMat2) {
        self.glide.to.a = within_limits(a);
        self.stop_matrix_play();
    }

    /// A hand has changed A: whatever was changing it stops.
    fn stop_matrix_play(&mut self) {
        if matches!(self.play, Play::Transpose(..) | Play::Spin(..)) {
            (self.play, self.waiting) = (Play::Idle, false);
        }
    }

    fn shown(&self) -> Setting {
        self.glide.shown
    }

    fn playing(&self) -> bool {
        self.play != Play::Idle
    }

    /// Move everything on: `dt` seconds of the scene's time (what plays) and `real_dt`
    /// of the clock's (a change of setting glides by the clock, so that it is seen
    /// through even while the scene's time stands still).
    fn advance(&mut self, dt: f32, real_dt: f32) {
        self.glide.step(real_dt as f64);
        if self.waiting {
            if !self.glide.arrived() {
                return;
            }
            self.waiting = false;
        }
        let dt = if dt.is_finite() { dt.max(0.0) } else { 0.0 };
        match self.play {
            Play::Idle => {}
            Play::Round(s, from) => {
                let s = s + dt;
                let turn = TAU64 * smoothstep(s / ROUND_SECS) as f64;
                let x = DVec2::from_angle(turn).rotate(from);
                self.glide.now(|g| g.x = x);
                self.play = if s >= ROUND_SECS { Play::Idle } else { Play::Round(s, from) };
            }
            Play::Transpose(s, from) => {
                let s = s + dt;
                let u = smoothstep(s / TRANSPOSE_SECS) as f64;
                let a = from + (from.transpose() - from) * u;
                self.glide.now(|g| g.a = a);
                self.play = if s >= TRANSPOSE_SECS { Play::Idle } else { Play::Transpose(s, from) };
            }
            Play::Spin(s, base) => {
                let s = s + dt;
                let a = base + quarter_turn() * spin_at(s);
                self.glide.now(|g| g.a = a);
                self.play = if s >= spin_length() { Play::Idle } else { Play::Spin(s, base) };
            }
        }
    }
}

/// The four numbers of a 2×2 matrix to drag or type, within ±`ENTRY_MAX`. Returns
/// (changed, typed): a number typed in glides the picture to it, one dragged moves it
/// at once.
fn matrix_edit(ui: &mut egui::Ui, m: &mut DMat2) -> (bool, bool) {
    let (mut changed, mut typed) = (false, false);
    let mut a = [[m.x_axis.x, m.y_axis.x], [m.x_axis.y, m.y_axis.y]];
    egui::Grid::new("symskew_a").spacing([6.0, 4.0]).show(ui, |ui| {
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

/// What a picture is of.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Which {
    A,
    S,
    K,
}

impl Which {
    const ALL: [Which; 3] = [Which::A, Which::S, Which::K];

    fn colour(self) -> Color {
        match self {
            Which::A => A_COL,
            Which::S => S_COL,
            Which::K => K_COL,
        }
    }

    fn matrix(self, a: DMat2, p: &Parts) -> DMat2 {
        match self {
            Which::A => a,
            Which::S => p.s,
            Which::K => p.k,
        }
    }

    /// The velocity's name.
    fn velocity(self) -> &'static str {
        match self {
            Which::A => "$Ax$",
            Which::S => "$Sx$",
            Which::K => "$Kx$",
        }
    }

    fn streak_colour(self) -> Color {
        self.colour().mix(pal::WHITE, 0.5)
    }
}

impl Scene for SymSkew {
    fn view(&self) -> View {
        STAGE_VIEW
    }

    fn settings(&self, s: &mut ViewSettings) {
        s.grid = false;
        s.bloom = s.bloom.min(0.35);
    }

    fn animated(&self) -> bool {
        let moving = size(self.shown().a) > 1e-12;
        (!self.arrows && moving) || self.playing() || self.waiting || !self.glide.arrived()
    }

    fn pointer_hint(&self) -> Option<&'static str> {
        Some(r"white dot: move $x$ (the same point in all three pictures)    drag elsewhere: move the picture    scroll: zoom")
    }

    fn ui(&mut self, ui: &mut egui::Ui, _: &Gfx) {
        ui::note(ui, r"Every square matrix $A$ is, in exactly one way, a symmetric matrix plus a skew-symmetric one:");
        ui::equation(ui, r"A = S + K, \quad S = \frac{A + A^\top}{2}, \quad K = \frac{A - A^\top}{2}.");
        ui::note(ui, r"Let $Ax$ be the velocity of the point $x$, as in the scene on flows. Then $S$ is a stretch: along two lines at right angles (its eigenlines, orange and purple) points move straight out or in, at the rates $s_1$ and $s_2$. $K$ is a spin: every point turns round $0$ at the same rate $w$. The velocity at every point is the sum of the two: $Ax = Sx + Kx$, the parallelogram at the white point.");

        ui::section(ui, "Things to try");
        for (k, trial) in TRIALS.iter().enumerate() {
            if ui::experiment(ui, trial.label, trial.look_for) {
                self.set_up(k);
            }
        }

        ui::section(ui, "The matrix $A$");
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
                self.stop_matrix_play();
            }
        }
        if ui::buttons(ui, &[r"transpose it: $A \to A^\top$"]).is_some() {
            let t = self.glide.to.a.transpose();
            self.aim_matrix(t);
        }
        if let Some(p) = PRESETS.iter().find(|p| rows(p.1) == self.glide.to.a) {
            ui::note(ui, p.2);
        }

        ui::section(ui, "Or by its two parts");
        let parts = split(self.glide.to.a);
        if let Some(v) = parts.axis {
            self.angle = v.y.atan2(v.x);
        }
        let (mut s1, mut s2, mut w) = (parts.s1 as f32, parts.s2 as f32, parts.w as f32);
        let mut deg = self.angle.rem_euclid(PI64).to_degrees() as f32;
        let rate = RATE_MAX as f32;
        let c1 = ui::slider(ui, "$s_1$, the larger stretch rate", &mut s1, -rate..=rate);
        let c2 = ui::slider(ui, "$s_2$, the smaller", &mut s2, -rate..=rate);
        let c3 = ui::slider(ui, "angle of $s_1$'s axis (°)", &mut deg, 0.0..=180.0);
        let spun = ui::slider(ui, "spin rate $w$", &mut w, -rate..=rate);
        if c1 || c2 || c3 {
            // (one dragged past the other takes it along: s₁ is the larger)
            if c1 && s1 < s2 {
                s2 = s1;
            }
            if c2 && s2 > s1 {
                s1 = s2;
            }
            self.angle = (deg as f64).to_radians();
            let a = compose(s1 as f64, s2 as f64, self.angle, parts.w);
            self.glide.now(|s| s.a = a);
            self.stop_matrix_play();
        } else if spun {
            // (S exactly as it was: only K changes)
            let a = parts.s + quarter_turn() * w as f64;
            self.glide.now(|s| s.a = a);
            self.stop_matrix_play();
        }
        ui::note(ui, r"Three numbers make $S$ (two stretch rates and the angle of its axes) and one makes $K$: the four numbers of $A$.");

        ui::section(ui, "Show");
        let mut t = self.glide.to.t as f32;
        if ui::slider(ui, "time $t$", &mut t, 0.0..=T_MAX as f32) {
            self.glide.now(|s| s.t = t as f64);
        }
        ui::note(ui, r"How long the yellow circle is carried along by each flow.");
        let mut style = self.arrows as usize;
        if ui::choice(ui, &["moving streaks", "small arrows"], &mut style) {
            self.arrows = style == 1;
        }
        ui::note(ui, "The flow as streaks that move with it, or as the velocity at points of a grid, drawn to the same scale in all three pictures.");
        ui::checkbox(ui, &mut self.eigenlines, "the eigenlines");
        ui::checkbox(ui, &mut self.shadows, r"the shadows of $Ax$ and $Sx$ on the line of $x$");

        ui::section(ui, "Reading the picture");
        ui::key(
            ui,
            &[
                (A_COL, r"cyan: $A$, its numbers and its picture; the arrow $Ax$ is the velocity at $x$, the diagonal of the parallelogram of $Sx$ and $Kx$"),
                (S_COL, r"green: $S$, the symmetric part, and its velocity $Sx$"),
                (K_COL, r"pink: $K$, the skew-symmetric part, and its velocity $Kx$, at right angles to $x$ (the corner)"),
                (EIG[0], r"orange and purple, dashed: the eigenlines of each picture's matrix, along which points move straight out or in; $S$'s are always at right angles, $K$ has none (unless $K = 0$)"),
                (pal::YELLOW, r"yellow: a small circle about $0$ (dashed) and where the flow carries it in the time $t$ (solid), with the paths of twelve of its points from where they set off (the dim dots); the larger dot set off on the $x$ axis"),
                (pal::WHITE, "white: the point $x$ (drag it; it is the same point in all three pictures); dashed, when shown: the line of $x$, and the shadows on it"),
                (Which::A.streak_colour(), "faint streaks: each matrix's flow (or small arrows: its velocity at points of a grid)"),
                (pal::TEXT, r"the chart (on a wide screen): how fast each flow turns the line through $0$ at each angle $\theta$, up for anticlockwise; $K$'s is level at $w$, $S$'s a wave between $\pm(s_1 - s_2)/2$ (dashed) that is $0$ on $S$'s axes, $A$'s their sum, $0$ on $A$'s eigenlines (the dots); the white line is the line of $x$"),
            ],
        );

        ui::section(ui, "Words");
        ui::term(ui, r"Transpose $A^\top$:", r"$A$ mirrored across its diagonal: the rows of $A$ are the columns of $A^\top$.");
        ui::term(ui, "Symmetric:", r"equal to its transpose, $S^\top = S$. Its eigenvalues are real, and it has two eigenlines at right angles (every line, if the two eigenvalues are equal): a stretch along two perpendicular axes.");
        ui::term(ui, "Skew-symmetric:", r"equal to minus its transpose, $K^\top = -K$, so its diagonal is $0$. In the plane it is $wJ$, a multiple of the quarter turn $J = \begin{bmatrix} 0 & -1 \\ 1 & 0 \end{bmatrix}$.");
        ui::term(ui, "Only one way:", r"if $A = S' + K'$ with $S'$ symmetric and $K'$ skew, then $A^\top = S' - K'$; adding and subtracting the two gives $S' = S$ and $K' = K$.");
        ui::term(ui, r"Stretch rates $s_1, s_2$:", r"the eigenvalues of $S$. On $S$'s axes a point moves straight out (rate above $0$) or in (below $0$), at that rate times its distance.");
        ui::term(ui, "Spin rate $w$:", r"$Kx = wJx$: every point turns round $0$ at $w$ radians per unit of time, anticlockwise if $w > 0$. $w = (a_{21} - a_{12})/2$.");
        ui::term(ui, "Stretch against spin:", r"the spin turns every direction at the rate $w$; the stretch turns a direction towards $s_1$'s axis at a rate up to $(s_1 - s_2)/2$ (at 45° from the axes). Where the two turnings cancel, a line is kept: an eigenline of $A$ (the chart). $A$'s eigenvalues are $\frac{s_1 + s_2}{2} \pm \sqrt{((s_1 - s_2)/2)^2 - w^2}$, complex exactly when $|w| > (s_1 - s_2)/2$.");
        ui::term(ui, "Saddle, sink, spiral, centre:", "the kinds of flow, as the scene on flows names them: in along one eigenline and out along the other; everything flows in to $0$; in to $0$ while turning; round closed loops.");
        ui::term(ui, r"Quadratic form $x \cdot Ax$:", r"a number for each $x$. $Kx$ is at right angles to $x$, so $x \cdot Kx = 0$ and $x \cdot Ax = x \cdot Sx$: the form sees only $S$.");
        ui::term(ui, "Trace:", r"the sum of the diagonal. Areas change at the rate $\tr A$: the yellow area is multiplied by $e^{t \tr A}$. $\tr K = 0$, so $\tr A = \tr S$.");
        ui::term(ui, r"Sum of squares $\|A\|^2$:", r"the four entries squared and added. Taken as lists of four numbers, $S$ and $K$ are at right angles (their entries multiplied pair by pair add up to $0$), so $\|A\|^2 = \|S\|^2 + \|K\|^2$, and $S$ is the symmetric matrix nearest to $A$: $\|A - S\|^2$ is the least.");
        ui::term(ui, "In space:", r"a $3 \times 3$ matrix splits the same way, $S$ with 6 numbers of its own and $K$ with 3; then $Kx = \omega \times x$, a spin about the axis $\omega$.");
        ui::term(ui, "Velocities, not journeys:", r"$Ax = Sx + Kx$ at each moment, but flowing by $A$ for a time $t$ is in general not flowing by $S$ and then by $K$. The two agree for every $t$ exactly when $S$ and $K$ commute, $SK = KS$ (in the plane: $K = 0$, or $S$ a multiple of $I$); for a short time they agree to first order in $t$.");
        ui::term(ui, "Polar form:", r"$A = QP$, a stretch $P$ and then a turn $Q$ (with a mirror, if $\det A < 0$), is a product, not a sum. For a matrix near $I$ the two meet: $I + \varepsilon A$ has $Q \approx I + \varepsilon K$ and $P \approx I + \varepsilon S$, to first order in $\varepsilon$.");

        ui::section(ui, "Numbers");
        let s = self.shown();
        let p = split(s.a);
        let value = |v: f64| format!("${}$", num(v, 2));
        ui::readout_colored(ui, "$s_1$", value(p.s1), EIG[0]);
        ui::readout_colored(ui, "$s_2$", value(p.s2), EIG[1]);
        let axis = p.axis.map_or("any: $s_1 = s_2$".to_string(), |v| format!("${}°$", num(v.y.atan2(v.x).rem_euclid(PI64).to_degrees(), 1)));
        ui::readout(ui, "angle of $s_1$'s axis", axis);
        ui::readout_colored(ui, "$w$", value(p.w), K_COL);
        ui::readout_colored(ui, "eigenvalues of $A$", eigen_text(eigen(s.a)).replace(r"\lambda = ", ""), A_COL);
        ui::readout(ui, "spin against stretch", contest(s.a));
        ui::readout(ui, r"$\tr A = \tr S$", value(trace(s.a)));
        ui::readout(ui, "$x$", format!("${}$", tuple(&[s.x.x, s.x.y], 2)));
        ui::readout(ui, r"$x \cdot Ax$, $x \cdot Sx$", format!("${}, {}$", num(s.x.dot(s.a * s.x), 2), num(s.x.dot(p.s * s.x), 2)));
        ui::readout(ui, r"$x \cdot Kx$", value(s.x.dot(p.k * s.x)));
        ui::readout(ui, r"$\|A\|^2$, $\|S\|^2 + \|K\|^2$", format!("${}, {}$", num(squares(s.a), 2), num(squares(p.s) + squares(p.k), 2)));
        ui::see_also(
            ui,
            &[
                ("expm", "flows: the velocity $Ax$ followed in time"),
                ("quadform", "the quadratic form of a symmetric matrix"),
                ("polar", "polar form: a turn times a stretch"),
                ("expsum", "two flows at once, or one after the other"),
            ],
        );
    }

    fn frame(&mut self, f: &mut Frame) {
        self.advance(f.dt, f.real_dt);
        let view = f.view_size();
        let lay = Layout::of(view, f.top_inset);
        // (a canvas squeezed to a sliver is one the controls have opened over, on a
        // phone: the pictures are drawn without their words)
        let covered = view.x < 160.0;
        if !covered {
            if self.laid.is_some_and(|l| l != lay.planes) {
                for flow in &mut self.flows {
                    flow.forget();
                }
            }
            self.laid = Some(lay.planes);
        }
        let s = self.shown();
        let parts = split(s.a);
        let labels_before = f.dl.labels.len();
        // (the batches of the parts, made in the same order every frame)
        for part in [Part::Arrows, Part::Carried, Part::Field, Part::Corners] {
            lines_for(f, part, false);
        }
        for part in [Part::Eigen, Part::Shadow, Part::Arrows] {
            lines_for(f, part, true);
        }
        // the arrows' scale: the longest velocity on the grid, in all three pictures, as long as 0.8 of a cell
        let field_scale = self.arrows.then(|| field_scale(s.a, &parts));
        for which in Which::ALL {
            self.picture(f, &lay, which, s, &parts, field_scale);
        }
        self.chart(f, &lay, s, &parts);
        if covered || !lay.words {
            f.dl.labels.truncate(labels_before);
            return;
        }
        self.words(f, &lay, s, &parts);
    }
}

/// The small arrows of the field: a grid of this many points each way.
const FIELD: usize = 9;

/// The length of a velocity on the screen per unit of velocity, for the small arrows:
/// the longest on the grid in any of the three pictures 0.8 of a cell long.
fn field_scale(a: DMat2, p: &Parts) -> f64 {
    let cell = 2.0 * REACH / FIELD as f64;
    let mut most = 0.0f64;
    for m in [a, p.s, p.k] {
        for (i, j) in (0..FIELD).flat_map(|i| (0..FIELD).map(move |j| (i, j))) {
            let q = DVec2::new(-REACH + cell * (i as f64 + 0.5), -REACH + cell * (j as f64 + 0.5));
            most = most.max((m * q).length());
        }
    }
    if most > 1e-12 {
        0.8 * cell / most
    } else {
        0.0
    }
}

impl SymSkew {
    /// One of the three pictures: its flow, its eigenlines, the yellow circle carried,
    /// and x with its velocity (in A's picture, the parallelogram of the two parts).
    fn picture(&mut self, f: &mut Frame, lay: &Layout, which: Which, s: Setting, parts: &Parts, field_scale: Option<f64>) {
        let plane = lay.planes[which as usize];
        let (lo, hi) = (plane.lo(), plane.hi());
        if plane.half * pixels_per_unit(f) < 12.0 {
            return;
        }
        let m = which.matrix(s.a, parts);
        let colour = which.colour();
        inset(f, lo, hi);
        frame_rect(f, lo, hi, 1.2, colour.a(0.45));
        let r = REACH;
        f.line(plane.at(DVec2::new(-r, 0.0)), plane.at(DVec2::new(r, 0.0)), 1.0, pal::FAINT.a(0.9));
        f.line(plane.at(DVec2::new(0.0, -r)), plane.at(DVec2::new(0.0, r)), 1.0, pal::FAINT.a(0.9));
        let still = size(m) <= 1e-9 * size(s.a).max(1e-12) || size(m) < 1e-12;

        // the flow: streaks, or small arrows
        match field_scale {
            None if !still => {
                let flow = &mut self.flows[which as usize];
                flow.advance(m, f.dt, which.streak_colour(), plane, Some(f.gfx));
                if let Some(trails) = &mut flow.trails {
                    f.trails(trails);
                }
            }
            Some(k) => {
                let cell = 2.0 * REACH / FIELD as f64;
                for (i, j) in (0..FIELD).flat_map(|i| (0..FIELD).map(move |j| (i, j))) {
                    let q = DVec2::new(-REACH + cell * (i as f64 + 0.5), -REACH + cell * (j as f64 + 0.5));
                    let v = m * q * k;
                    if v.length() > 1e-9 {
                        arrow_in(f, plane, Part::Field, q - v * 0.5, q + v * 0.5, 1.3, which.streak_colour().a(0.6));
                    }
                }
            }
            None => {}
        }

        let mut room = room_in(f, plane);
        let wpp = 1.0 / pixels_per_unit(f).max(1e-6);
        let unit_px = plane.scale() / wpp;

        // the eigenlines: A's (if real), S's (always at right angles), K's (none)
        let lines: Vec<(f64, DVec2, Color, String)> = match which {
            Which::A => match eigen(m) {
                Eigen::Real { l1, l2, lines: Lines::Two(v1, v2) } if !still => {
                    vec![(l1, v1, EIG[0], format!(r"$\lambda_1 = {}$", num(l1, 2))), (l2, v2, EIG[1], format!(r"$\lambda_2 = {}$", num(l2, 2)))]
                }
                Eigen::Real { l1, lines: Lines::One(v), .. } if !still => vec![(l1, v, EIG[0], format!(r"$\lambda = {}$", num(l1, 2)))],
                _ => Vec::new(),
            },
            Which::S => match parts.axis {
                Some(v) if !still => {
                    vec![(parts.s1, v, EIG[0], format!("$s_1 = {}$", num(parts.s1, 2))), (parts.s2, v.perp(), EIG[1], format!("$s_2 = {}$", num(parts.s2, 2)))]
                }
                _ => Vec::new(),
            },
            Which::K => Vec::new(),
        };
        let mut named = Vec::new();
        if self.eigenlines && which != Which::K {
            let slot = (which == Which::S) as usize;
            for (i, (_, v, c, text)) in lines.iter().enumerate() {
                // (a line's name stays at the end it was at as the matrix changes; a line
                // new to the picture is named at its upper end, or, the second, at the end
                // away from the first's: two lines that close up meet at opposite ends)
                let mut v = *v;
                let flip = match (self.last_lines[slot][i], i) {
                    (Some(was), _) => was.dot(v) < 0.0,
                    (None, 1) => self.last_lines[slot][0].map_or(v.y < 0.0, |first| first.dot(v) > 1e-9 || (first.dot(v).abs() <= 1e-9 && v.y < 0.0)),
                    (None, _) => v.y < 0.0 || (v.y == 0.0 && v.x < 0.0),
                };
                if flip {
                    v = -v;
                }
                self.last_lines[slot][i] = Some(v);
                segment_in(f, plane, Part::Eigen, true, -v * (2.0 * r), v * (2.0 * r), 1.5, c.a(0.85));
                if let Some((p, q)) = clip(-v * (2.0 * r), v * (2.0 * r), bounds().0, bounds().1) {
                    room.line(f, plane.at(p).extend(0.0), plane.at(q).extend(0.0));
                }
                named.push((i, v, *c, text.clone()));
            }
            for i in lines.len()..2 {
                self.last_lines[slot][i] = None;
            }
            if which == Which::S && lines.len() == 2 {
                let mark = (12.0 * wpp).min(0.3 * plane.scale());
                corner(f, plane.centre, lines[0].1.as_vec2(), lines[1].1.as_vec2(), mark, pal::WHITE.a(0.7));
            }
        }

        // the small circle, and where the flow carries it in the time t, with the paths
        // of twelve of its points from where they set off (the small dim dots)
        let (blo, bhi) = bounds();
        let round = |i: usize, n: usize| DVec2::from_angle(i as f64 / n as f64 * TAU64) * RADIUS;
        let circle: Vec<Vec2> = (0..=96).map(|i| plane.at(round(i, 96))).collect();
        f.dashed_path(circle.iter().map(|p| p.extend(0.0)), 1.2, pal::YELLOW.a(0.6));
        let e = expm(m, s.t);
        let image: Vec<DVec2> = (0..=96).map(|i| e * round(i, 96)).collect();
        let cut = clip_polygon(&image[..96], blo, bhi);
        for i in 1..cut.len().saturating_sub(1) {
            f.dl.tris_alpha.tri(plane.at(cut[0]).extend(0.0), plane.at(cut[i]).extend(0.0), plane.at(cut[i + 1]).extend(0.0), pal::YELLOW.a(0.1));
        }
        for run in runs(&image, blo, bhi) {
            lines_for(f, Part::Carried, false).polyline(run.iter().map(|p| plane.at(*p).extend(0.0)), 1.7, pal::YELLOW.a(0.9));
        }
        for k in (0..96).step_by(8) {
            room.line(f, plane.at(image[k]).extend(0.0), plane.at(image[k + 8]).extend(0.0));
        }
        let steps: Vec<DMat2> = (0..=12).map(|j| expm(m, s.t * j as f64 / 12.0)).collect();
        for k in 0..12 {
            let start = round(k, 12);
            f.dot(plane.at(start), 4.0, pal::YELLOW.a(0.45));
            let path: Vec<DVec2> = steps.iter().map(|step| *step * start).collect();
            for run in runs(&path, blo, bhi) {
                lines_for(f, Part::Carried, false).polyline(run.iter().map(|p| plane.at(*p).extend(0.0)), if k == 0 { 1.8 } else { 1.2 }, pal::YELLOW.a(0.8));
            }
            let end = e * start;
            if in_square(end) {
                let dot = if k == 0 { 9.0 } else { 5.5 };
                f.dl.points_with(PointKind::Disc, PointStyle { outline_px: 1.25 + 1e-4, ..Default::default() }).push(plane.at(end).extend(0.0), dot, pal::YELLOW);
            }
        }

        // x, which can be dragged in any of the three pictures
        let x = s.x;
        let mut handle = plane.at(x);
        if f.drag2(1 + which as u32, &mut handle, pal::WHITE) {
            let q = plane.of(handle).clamp(DVec2::splat(-0.96 * REACH), DVec2::splat(0.96 * REACH));
            self.glide.now(|g| g.x = q);
            if matches!(self.play, Play::Round(..)) {
                (self.play, self.waiting) = (Play::Idle, false);
            }
        }
        let v = m * x;
        let (sx, kx) = (parts.s * x, parts.k * x);
        // the shadows of the velocity's tip on the line of x (A's and S's: the same)
        let unit = x.normalize_or_zero();
        if self.shadows && unit != DVec2::ZERO && which != Which::K {
            segment_in(f, plane, Part::Shadow, true, -unit * (2.0 * r), unit * (2.0 * r), 1.1, pal::WHITE.a(0.4));
            let foot = x + unit * unit.dot(v);
            segment_in(f, plane, Part::Shadow, true, x + v, foot, 1.2, colour.a(0.8));
            segment_in(f, plane, Part::Shadow, false, x, foot, 3.2, pal::WHITE.a(0.55));
            if (foot - x).length() * unit_px as f64 > 10.0 && (x + v - foot).length() * unit_px as f64 > 10.0 && in_square(foot) {
                let mark = 8.0 * wpp;
                corner(f, plane.at(foot), (-unit * unit.dot(v)).as_vec2(), (x + v - foot).as_vec2(), mark, pal::WHITE.a(0.6));
            }
        }
        if which == Which::A {
            // the parallelogram: Sx and Kx from x, and the sides that close it at x + Ax
            segment_in(f, plane, Part::Arrows, true, x + sx, x + v, 1.1, K_COL.a(0.6));
            segment_in(f, plane, Part::Arrows, true, x + kx, x + v, 1.1, S_COL.a(0.6));
            arrow_in(f, plane, Part::Arrows, x, x + sx, 2.0, S_COL);
            arrow_in(f, plane, Part::Arrows, x, x + kx, 2.0, K_COL);
        }
        if which == Which::K && kx.length() * unit_px as f64 > 10.0 && unit != DVec2::ZERO {
            // the corner between x and Kx: always a right angle
            let mark = 9.0 * wpp;
            segment_in(f, plane, Part::Shadow, true, DVec2::ZERO, x, 1.0, pal::WHITE.a(0.35));
            corner(f, plane.at(x), (-unit).as_vec2(), kx.as_vec2(), mark, pal::WHITE.a(0.75));
        }
        arrow_in(f, plane, Part::Arrows, x, x + v, 2.6, colour);
        for (p, q) in [(x, x + v), (x, x + sx), (x, x + kx)] {
            if let Some((p, q)) = clip(p, q, blo, bhi) {
                room.line(f, plane.at(p).extend(0.0), plane.at(q).extend(0.0));
            }
        }

        // the names, each in a place that moves smoothly with what it names: the
        // velocities past their tips; the eigenlines beside them near the end each keeps
        // (on the side a quarter turn anticlockwise from it); then x, where there is room
        let away = |p: DVec2| Vec2::new(p.x as f32, -(p.y as f32));
        let named_px = |len: f64| len * unit_px as f64 >= 22.0;
        let mut tips = vec![("v", which as u32, v, colour, which.velocity(), lay.text)];
        if which == Which::A {
            tips.push(("sx", 0, sx, S_COL, "$Sx$", lay.text - 1.0));
            tips.push(("kx", 0, kx, K_COL, "$Kx$", lay.text - 1.0));
        }
        for (tag, number, part, c, text, size) in tips {
            if named_px(part.length()) && in_square(x + part) {
                let tip = plane.at(x + part);
                if let Some(centre) = beyond(f, tip, part.as_vec2(), text, size, 4.0) {
                    put(f, &mut room, (tag, number), plane, tip.extend(0.0), centre, text, size, c);
                }
            }
        }
        for (i, v, c, text) in named {
            let size = lay.text - 1.0;
            let end = plane.at(v * (REACH * 0.8));
            let side = Vec2::new(-v.y as f32, -v.x as f32);
            if let [Some(centre), _] = beside(f, end, v.as_vec2(), side, &text, size, 4.0) {
                let tag: &'static str = if which == Which::A { "eigA" } else { "eigS" };
                put(f, &mut room, (tag, i as u32), plane, end.extend(0.0), centre, &text, size, c);
            }
        }
        if in_square(x) {
            room.label(f, ("x", which as u32), plane.at(x).extend(0.0), "$x$", lay.text + 2.0, pal::WHITE, away(x) - away(v) * 0.5, 9.0, false);
        }
        if still {
            let text = match which {
                Which::A => "$A = 0$: nothing moves",
                Which::S => "$S = 0$: no stretch",
                Which::K => "$K = 0$: no spin",
            };
            if let Some(c) = on_screen(f, plane.at(DVec2::new(0.0, -0.45 * REACH)).extend(0.0)) {
                f.caption_px(c, text, lay.text, pal::TEXT, Align2::CENTER_CENTER);
            }
        }
    }

    /// The words round the pictures: each one's name and matrix, A = S + K written out,
    /// and under them what each part does and how they meet.
    fn words(&self, f: &mut Frame, lay: &Layout, s: Setting, p: &Parts) {
        let rects: Vec<(Vec2, Vec2)> = lay.planes.iter().filter_map(|plane| on_screen_rect(f, *plane)).collect();
        let [ra, rs, rk] = rects[..] else { return };
        let (size, numbers, line) = (lay.text, lay.numbers, lay.line());
        let msize = matrix_size(2, 2, CELL, numbers);
        let entries = |m: DMat2, c: Color| -> Vec<Vec<(String, Color)>> { (0..2).map(|i| (0..2).map(|j| (cell(m.col(j)[i]), c)).collect()).collect() };
        let mats = [entries(s.a, A_COL), entries(p.s, S_COL), entries(p.k, K_COL)];
        let titles: [[&str; 2]; 3] = [
            [r"$A$: the velocity is $Ax$", "$A$"],
            [r"$S = (A + A^\top)/2$: stretch", "$S$: stretch"],
            [r"$K = (A - A^\top)/2$: spin", "$K$: spin"],
        ];
        let sign = |f: &mut Frame, at: Vec2, text: &str| f.text_px(at, text, numbers + 5.0, pal::TEXT, Align2::CENTER_CENTER);
        let ra_wide = ra.1.x - ra.0.x;
        if lay.narrow {
            // the equation on top, A = S + K, across the width of A's picture
            let sign_w = 26.0;
            let whole = 3.0 * msize.x + 2.0 * sign_w;
            let left = (ra.0.x + ra.1.x) * 0.5 - whole * 0.5;
            let top = ra.0.y - line - 10.0 - msize.y;
            let mid = top + msize.y * 0.5;
            for (k, rows) in mats.iter().enumerate() {
                let x = left + k as f32 * (msize.x + sign_w);
                hud_matrix(f, Vec2::new(x, top), rows, CELL, numbers, pal::DIM);
                if k < 2 {
                    sign(f, Vec2::new(x + msize.x + sign_w * 0.5, mid), ["$=$", "$+$"][k]);
                }
            }
            let title_a = fitting(f, &[titles[0][0], titles[0][1]], ra_wide, size);
            f.text_px(Vec2::new(ra.0.x, ra.0.y - 4.0), title_a, size, A_COL, Align2::LEFT_BOTTOM);
            for (k, r) in [(1, rs), (2, rk)] {
                let wide = r.1.x - r.0.x;
                let title = fitting(f, &[titles[k][0], titles[k][1]], wide, size);
                f.text_px(Vec2::new(r.0.x, r.0.y - 4.0), title, size, Which::ALL[k].colour(), Align2::LEFT_BOTTOM);
            }
        } else {
            // each matrix over its picture, with = and + between them
            let top = ra.0.y - 8.0 - msize.y;
            let mid = top + msize.y * 0.5;
            for (k, r) in [ra, rs, rk].into_iter().enumerate() {
                let wide = r.1.x - r.0.x;
                let x = (r.0.x + r.1.x) * 0.5 - msize.x * 0.5;
                hud_matrix(f, Vec2::new(x, top), &mats[k], CELL, numbers, pal::DIM);
                let title = fitting(f, &[titles[k][0], titles[k][1]], wide, size);
                f.text_px(Vec2::new(r.0.x, top - 6.0), title, size, Which::ALL[k].colour(), Align2::LEFT_BOTTOM);
            }
            sign(f, Vec2::new((ra.1.x + rs.0.x) * 0.5, mid), "$=$");
            sign(f, Vec2::new((rs.1.x + rk.0.x) * 0.5, mid), "$+$");
        }

        // under each picture: what its matrix does, and what it does to areas
        let a = s.a;
        let kind_a = kind(a);
        let under_a = format!("{}: {}", kind_a.name(), eigen_text(eigen(a)));
        let stretch = match p.axis {
            Some(_) => format!("${}$, ${}$", tint(EIG[0], &format!("s_1 = {}", num(p.s1, 2))), tint(EIG[1], &format!("s_2 = {}", num(p.s2, 2)))),
            None => format!("$s_1 = s_2 = {}$", num(p.s1, 2)),
        };
        let spin = if p.w.abs() < 0.005 {
            "$w = 0$: no spin".to_string()
        } else {
            format!("$w = {}$, {}", num(p.w, 2), if p.w > 0.0 { "anticlockwise" } else { "clockwise" })
        };
        let area = |tr: f64, name: &str| -> [String; 2] {
            let times = (s.t * tr).exp();
            [format!(r"$\tr {name} = {}$: yellow area $\times {}$", num(tr, 2), num(times, 2)), format!(r"area $\times {}$", num(times, 2))]
        };
        let lines: [[String; 2]; 3] = [
            [under_a, area(trace(a), "A")[0].clone()],
            [format!("stretch rates {stretch}"), area(trace(p.s), "S")[0].clone()],
            [format!("spin rate {spin}"), r"$\tr K = 0$: yellow area $\times 1$".to_string()],
        ];
        let short: [[String; 2]; 3] = [[eigen_text(eigen(a)), area(trace(a), "A")[1].clone()], [stretch.clone(), area(trace(p.s), "S")[1].clone()], [spin.clone(), r"area $\times 1$".to_string()]];
        for (k, r) in [ra, rs, rk].into_iter().enumerate() {
            let wide = r.1.x - r.0.x + 8.0;
            let shown = if lay.narrow && k == 0 { 1 } else { 2 };
            for j in 0..shown {
                let text = fitting(f, &[lines[k][j].as_str(), short[k][j].as_str()], wide, size - 0.5);
                f.text_px(Vec2::new(r.0.x, r.1.y + 5.0 + j as f32 * line), text, size - 0.5, pal::DIM, Align2::LEFT_TOP);
            }
        }

        // across the foot: the chart's title, the contest, and the quadratic form
        let mut bottom = rs.1.y.max(rk.1.y).max(if lay.narrow { 0.0 } else { ra.1.y }) + 5.0 + 2.0 * line + 6.0;
        let left = ra.0.x.min(rs.0.x);
        let room = ra.1.x.max(rk.1.x) - left;
        if let Some((lo, hi)) = lay.chart {
            if let (Some(p), Some(q)) = (on_screen(f, lo.extend(0.0)), on_screen(f, hi.extend(0.0))) {
                let (min, max) = (p.min(q), p.max(q));
                let title = [
                    r"how fast each flow turns the line through $0$ at the angle $\theta$ (up: anticlockwise); where $A$'s curve meets $0$, that line is kept: an eigenline of $A$",
                    r"how fast each flow turns the line at the angle $\theta$; where $A$'s curve meets $0$, the line is kept",
                    r"how fast each flow turns the line at the angle $\theta$",
                ];
                let text = fitting(f, &title, max.x - left, size);
                f.text_px(Vec2::new(left, min.y - 4.0), text, size, pal::TEXT, Align2::LEFT_BOTTOM);
                bottom = max.y + line + 6.0;
            }
        }
        let verdict_words = match (verdict(a), eigen(a)) {
            (_, Eigen::Real { lines: Lines::Every, .. }) => "every direction is kept",
            (Verdict::Spin, _) => "the spin wins: no line is kept, and the flow turns",
            (Verdict::Stretch, _) => "the stretch wins: two lines are kept",
            (Verdict::Tie, _) => "a tie: one line is kept",
        };
        let first = [
            format!("{}: {verdict_words}", contest(a)),
            format!("{}: {}", contest(a), verdict_words.split(':').next().unwrap_or(verdict_words)),
        ];
        let x = s.x;
        let second = [
            format!(r"$x \cdot Ax = x \cdot Sx = {}$, and $x \cdot Kx = 0$: $Kx$ is at right angles to $x$", num(x.dot(a * x), 2)),
            format!(r"$x \cdot Ax = x \cdot Sx = {}$, $x \cdot Kx = 0$", num(x.dot(a * x), 2)),
        ];
        let mut y = bottom;
        for options in [&first, &second] {
            let text = fitting(f, &[options[0].as_str(), options[1].as_str()], room, size);
            f.text_px(Vec2::new(left, y), text, size, pal::TEXT, Align2::LEFT_TOP);
            y += line;
        }
        if !lay.narrow {
            let third = format!(r"yellow: a small circle (dashed) and where each flow carries it in the time $t = {}$; $\tr A = \tr S$, so $A$ and $S$ change areas alike", num(s.t, 2));
            let text = fitting(f, &[third.as_str(), "yellow: a small circle (dashed), and where each flow carries it in the time $t$"], room, size);
            f.text_px(Vec2::new(left, y), text, size, pal::DIM, Align2::LEFT_TOP);
        }
    }
}

impl SymSkew {
    /// The chart under the pictures: how fast each flow turns the line through 0 at
    /// each angle θ. K turns every line at w (a level line); S turns each towards its
    /// axis of s₁, at most (s₁ − s₂)/2 (a wave between the dashed bounds), and keeps
    /// its two axes; A turns each line at the sum, and keeps those where the sum is 0.
    fn chart(&self, f: &mut Frame, lay: &Layout, s: Setting, p: &Parts) {
        let Some((lo, hi)) = lay.chart else { return };
        inset(f, lo, hi);
        frame_rect(f, lo, hi, 1.0, pal::FAINT);
        let (a, w, h) = (s.a, p.w, p.half_spread());
        let top = (1.25 * (w.abs() + h)).max(1.0);
        let at = |theta: f64, rate: f64| Vec2::new(lo.x + (hi.x - lo.x) * (theta / PI64) as f32, (lo.y + hi.y) * 0.5 + (hi.y - lo.y) * 0.5 * (rate / top) as f32);
        // 0, and the bounds of the stretch's turning, ±(s₁ − s₂)/2
        f.line(at(0.0, 0.0), at(PI64, 0.0), 1.0, pal::DIM.a(0.9));
        for bound in [h, -h] {
            lines_for(f, Part::Fixed, true).segment(at(0.0, bound).extend(0.0), at(PI64, bound).extend(0.0), 1.0, S_COL.a(0.5));
        }
        // the three: K level, S a wave, A their sum
        let n = 180;
        let theta = |i: usize| PI64 * i as f64 / n as f64;
        let curve = |m: DMat2| (0..=n).map(move |i| at(theta(i), turning(m, theta(i))).extend(0.0)).collect::<Vec<Vec3>>();
        f.path(curve(p.k), 2.0, K_COL);
        f.path(curve(p.s), 1.8, S_COL);
        f.path(curve(a), 2.6, A_COL);
        // the lines S keeps (its axes) and those A keeps (its eigenlines), as angles
        let angle = |v: DVec2| v.y.atan2(v.x).rem_euclid(PI64);
        if let Some(v) = p.axis {
            for (dir, c) in [(v, EIG[0]), (v.perp(), EIG[1])] {
                let th = angle(dir);
                lines_for(f, Part::Eigen, true).segment(at(th, -top).extend(0.0), at(th, top).extend(0.0), 1.1, c.a(0.45));
            }
        }
        let kept: Vec<(DVec2, Color)> = match eigen(a) {
            Eigen::Real { lines: Lines::Two(v1, v2), .. } => vec![(v1, EIG[0]), (v2, EIG[1])],
            Eigen::Real { lines: Lines::One(v), .. } => vec![(v, EIG[0])],
            _ => Vec::new(),
        };
        for (v, c) in kept {
            f.dl.points_with(PointKind::Disc, PointStyle { outline_px: 1.5 + 1e-4, ..Default::default() }).push(at(angle(v), 0.0).extend(0.0), 10.0, c);
        }
        // where x is: its line, and the three rates there. (Drawn as fills: as x goes
        // round, its line leaves the chart at 180° and comes back at 0°, the same line,
        // which a line or a point drawn there would make a leap from edge to edge.)
        if s.x.length() > 1e-9 {
            let th = angle(s.x);
            let wpp = 1.0 / pixels_per_unit(f).max(1e-6);
            let (p0, p1) = (at(th, -top), at(th, top));
            let half = Vec2::new(0.5 * wpp, 0.0);
            f.dl.tris_alpha.quad((p0 - half).extend(0.0), (p0 + half).extend(0.0), (p1 + half).extend(0.0), (p1 - half).extend(0.0), pal::WHITE.a(0.35));
            for (m, c) in [(p.k, K_COL), (p.s, S_COL), (a, A_COL)] {
                let centre = at(th, turning(m, th));
                let ring: Vec<Vec2> = (0..=12).map(|i| centre + Vec2::from_angle(i as f32 * TAU / 12.0) * (3.5 * wpp)).collect();
                for pair in ring.windows(2) {
                    f.dl.tris_alpha.tri(centre.extend(0.0), pair[0].extend(0.0), pair[1].extend(0.0), c);
                }
            }
        }
        // the angles, and the names at the right-hand end
        let size = lay.text - 1.5;
        for (k, name) in ["0°", "45°", "90°", "135°", "180°"].iter().enumerate() {
            if let Some(q) = on_screen(f, at(k as f64 * PI64 / 4.0, -top).extend(0.0)) {
                f.text_px(q + Vec2::new(0.0, 4.0), *name, size, pal::DIM, Align2::CENTER_TOP);
            }
        }
        if let Some(q) = on_screen(f, at(0.0, -top).extend(0.0)) {
            f.text_px(q + Vec2::new(-16.0, 4.0), r"$\theta$", size + 1.0, pal::DIM, Align2::RIGHT_TOP);
        }
        if let Some(q) = on_screen(f, at(0.0, 0.0).extend(0.0)) {
            f.text_px(q - Vec2::new(6.0, 0.0), "$0$", size, pal::DIM, Align2::RIGHT_CENTER);
        }
        // the names, each beside its curve at a place of its own near the right-hand
        // end (S's below its curve, the others above), and the bound's at the left
        for (m, k, below, text, c) in [(a, 0.97, false, "$A$", A_COL), (p.s, 0.91, true, "$S$", S_COL), (p.k, 0.85, false, "$K$", K_COL)] {
            if let Some(q) = on_screen(f, at(PI64 * k, turning(m, PI64 * k)).extend(0.0)) {
                let (off, align) = if below { (Vec2::new(0.0, 5.0), Align2::CENTER_TOP) } else { (Vec2::new(0.0, -5.0), Align2::CENTER_BOTTOM) };
                f.text_px(q + off, text, size + 1.0, c, align);
            }
        }
        if h > 1e-9 {
            if let Some(q) = on_screen(f, at(0.015 * PI64, h).extend(0.0)) {
                f.text_px(q + Vec2::new(0.0, -3.0), r"$(s_1 - s_2)/2$", size, S_COL, Align2::LEFT_BOTTOM);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn close(m: DMat2, n: DMat2, tol: f64) -> bool {
        (m - n).to_cols_array().iter().all(|v| v.abs() <= tol)
    }

    /// Matrices to try things on: random ones, the experiments' and the quick choices',
    /// and the awkward ones.
    fn matrices(count: usize, seed: u64) -> Vec<DMat2> {
        let mut rng = Rng::new(seed);
        let mut out: Vec<DMat2> = TRIALS.iter().map(|t| rows(t.a)).chain(PRESETS.iter().map(|p| rows(p.1))).collect();
        out.extend((0..count).map(|_| rows([0; 4].map(|_| rng.range(-3.0, 3.0) as f64))));
        out.extend([DMat2::ZERO, DMat2::IDENTITY, DMat2::IDENTITY * -0.7, quarter_turn(), rows([0.0, 1.0, 0.0, 0.0]), rows([ENTRY_MAX; 4]), rows([1.0, 2.0, 3.0, 4.0])]);
        out
    }

    /// S is symmetric, K skew, and they add up to A; built back from its rates, its
    /// angle and its spin, S + K is A again.
    #[test]
    fn the_split_is_exact_and_comes_back() {
        for a in matrices(400, 1) {
            let p = split(a);
            assert!(close(p.s, p.s.transpose(), 0.0) && close(p.k, -p.k.transpose(), 0.0), "{a:?}");
            assert!(close(p.s + p.k, a, 1e-15 * size(a).max(1.0)), "{a:?}");
            assert!(close(p.k, quarter_turn() * p.w, 0.0) && p.w == 0.5 * (a.x_axis.y - a.y_axis.x));
            // A transposed is S − K
            assert!(close(a.transpose(), p.s - p.k, 1e-15 * size(a).max(1.0)));
            // the stretch rates are S's eigenvalues, on its axes
            assert!(p.s1 >= p.s2 && (p.s1 + p.s2 - trace(p.s)).abs() < 1e-12 && (p.s1 * p.s2 - p.s.determinant()).abs() < 1e-12, "{a:?}");
            if let Some(v) = p.axis {
                assert!((p.s * v - v * p.s1).length() < 1e-12 && (p.s * v.perp() - v.perp() * p.s2).length() < 1e-12, "{a:?}");
                let back = compose(p.s1, p.s2, v.y.atan2(v.x), p.w);
                assert!(close(back, a, 1e-12 * size(a).max(1.0)), "{a:?}: {back:?}");
            } else {
                assert!(close(p.s, DMat2::IDENTITY * p.s1, 1e-9 * size(a).max(1.0)));
            }
        }
    }

    /// The spin adds nothing to x·Ax (Kx is at right angles to x), areas grow at the
    /// rate tr A = tr S, and the parts are at right angles as lists of numbers, so that
    /// their sums of squares add up, and S is the symmetric matrix nearest to A.
    #[test]
    fn what_follows_from_the_split() {
        let mut rng = Rng::new(2);
        for a in matrices(300, 3) {
            let p = split(a);
            let scale = size(a).max(1.0);
            for _ in 0..20 {
                let x = DVec2::new(rng.range(-3.0, 3.0) as f64, rng.range(-3.0, 3.0) as f64);
                assert!(x.dot(p.k * x).abs() < 1e-14 * scale * x.length_squared());
                assert!((x.dot(a * x) - x.dot(p.s * x)).abs() < 1e-13 * scale * x.length_squared());
                // every point turns round 0 at the same rate w under K
                if x.length() > 1e-3 {
                    assert!((x.perp_dot(p.k * x) / x.length_squared() - p.w).abs() < 1e-13 * scale);
                }
            }
            assert!(trace(p.k) == 0.0 && (trace(a) - trace(p.s)).abs() < 1e-15 * scale);
            for t in [0.3, 1.0] {
                let area = |m: DMat2| expm(m, t).determinant();
                assert!((area(a) - (t * trace(a)).exp()).abs() < 1e-9 * area(a).max(1.0), "{a:?}");
                assert!((area(a) - area(p.s)).abs() < 1e-9 * area(a).max(1.0) && (area(p.k) - 1.0).abs() < 1e-12);
            }
            // at right angles: S·K = 0 entry by entry, and the squares add up
            let dot: f64 = p.s.to_cols_array().iter().zip(p.k.to_cols_array()).map(|(u, v)| u * v).sum();
            assert!(dot.abs() < 1e-14 * scale * scale);
            assert!((squares(a) - squares(p.s) - squares(p.k)).abs() < 1e-12 * scale * scale);
            // S is nearer to A than any other symmetric matrix
            for _ in 0..20 {
                let (e, g) = (rng.range(-1.0, 1.0) as f64, rng.range(-1.0, 1.0) as f64);
                let other = p.s + rows([e, g, g, -e * 0.3]) * 0.1;
                assert!(squares(a - other) > squares(a - p.s));
            }
        }
    }

    /// A's eigenvalues are (s₁ + s₂)/2 ± √(((s₁ − s₂)/2)² − w²): complex exactly when
    /// |w| > (s₁ − s₂)/2, one twice when the two are equal. And a direction is kept
    /// exactly where the stretch's turning and the spin's cancel.
    #[test]
    fn the_eigenvalues_are_complex_exactly_when_the_spin_wins() {
        let mut checked = [0; 3];
        for a in matrices(3000, 4) {
            let p = split(a);
            let (m, h, w) = (0.5 * (p.s1 + p.s2), p.half_spread(), p.w);
            let gap = h * h - w * w;
            assert!((quarter_disc(a) - gap).abs() < 1e-12 * size(a).max(1.0).powi(2), "{a:?}");
            let margin = 1e-6 * size(a).max(1.0).powi(2);
            match eigen(a) {
                Eigen::Complex { re, im } => {
                    assert!(w.abs() > h && gap < 0.0, "{a:?}");
                    assert!((re - m).abs() < 1e-12 && (im - (-gap).sqrt()).abs() < 1e-9, "{a:?}");
                    assert_eq!(verdict(a), Verdict::Spin);
                    checked[0] += 1;
                }
                Eigen::Real { l1, l2, .. } => {
                    assert!(gap > -margin, "{a:?}");
                    if gap > margin {
                        let r = gap.sqrt();
                        assert!((l1 - (m + r)).abs() < 1e-9 && (l2 - (m - r)).abs() < 1e-9, "{a:?}");
                        assert_eq!(verdict(a), Verdict::Stretch);
                        checked[1] += 1;
                    }
                }
            }
            // how fast the flow turns each direction: w, less the stretch's turning
            for k in 0..12 {
                let theta = k as f64 * PI64 / 12.0 + 0.1;
                let phi = p.axis.map_or(0.0, |v| v.y.atan2(v.x));
                assert!((turning(p.k, theta) - w).abs() < 1e-13 * size(a).max(1.0));
                assert!((turning(p.s, theta) + h * (2.0 * (theta - phi)).sin()).abs() < 1e-12 * size(a).max(1.0), "{a:?}");
                assert!((turning(a, theta) - turning(p.s, theta) - turning(p.k, theta)).abs() < 1e-13 * size(a).max(1.0));
            }
        }
        // ties, exactly on the boundary: the shear, and S + wJ with |w| = (s₁ − s₂)/2
        for a in [rows([0.0, 1.0, 0.0, 0.0]), compose(0.4, -1.6, 0.3, 1.0), compose(0.4, -1.6, 0.3, -1.0), compose(2.0, 0.0, 1.0, 1.0)] {
            assert_eq!(verdict(a), Verdict::Tie, "{a:?}");
            checked[2] += 1;
        }
        assert!(checked.iter().all(|&n| n > 0), "{checked:?}");
    }

    /// The chart's curves: A's turning is S's plus K's; it is 0 exactly on A's
    /// eigenlines (S's on S's axes); and it never meets 0 when the spin wins.
    #[test]
    fn the_chart_is_zero_on_the_eigenlines() {
        for a in matrices(500, 8) {
            let p = split(a);
            let scale = size(a).max(1.0);
            match eigen(a) {
                Eigen::Real { lines: Lines::Two(u, v), .. } => {
                    for d in [u, v] {
                        assert!(turning(a, d.y.atan2(d.x)).abs() < 1e-9 * scale, "{a:?}");
                    }
                }
                Eigen::Complex { .. } => {
                    let low = (0..720).map(|i| turning(a, i as f64 * PI64 / 720.0) * p.w.signum()).fold(f64::MAX, f64::min);
                    assert!(low > 0.0, "{a:?}");
                }
                _ => {}
            }
            if let Some(v) = p.axis {
                for d in [v, v.perp()] {
                    assert!(turning(p.s, d.y.atan2(d.x)).abs() < 1e-12 * scale);
                }
            }
        }
    }

    /// S and K commute exactly when K = 0 or S is a multiple of I, and only then is
    /// flowing by A for a time the flow of S followed by that of K; for a short time the
    /// two agree to first order.
    #[test]
    fn the_parts_flows_compose_only_when_they_commute() {
        for a in matrices(300, 5) {
            let p = split(a);
            let scale = size(a).max(1.0);
            let commute = close(p.s * p.k, p.k * p.s, 1e-12 * scale * scale);
            let simple = p.w.abs() < 1e-12 * scale || p.axis.is_none();
            assert_eq!(commute, simple, "{a:?}");
            let t = 0.7;
            let gap = size(expm(a, t) - expm(p.s, t) * expm(p.k, t));
            if commute {
                assert!(gap < 1e-9 * size(expm(a, t)).max(1.0), "{a:?}: {gap}");
            } else if p.w.abs() > 0.1 && p.half_spread() > 0.1 {
                assert!(gap > 1e-4, "{a:?}: {gap}");
            }
            // short times: the gap falls as t², one order faster than the flow moves
            let gap_at = |t: f64| size(expm(a, t) - expm(p.s, t) * expm(p.k, t));
            if !commute {
                let (g1, g2) = (gap_at(1e-3), gap_at(5e-4));
                assert!(g1 < 1e-5 * scale * scale && (g1 / g2 - 4.0).abs() < 0.05, "{a:?}: {g1} {g2}");
            }
        }
    }

    /// Polar form near I: I + εA = QP with Q ≈ I + εK and P ≈ I + εS, errors of order ε².
    #[test]
    fn polar_form_near_the_identity_is_the_split() {
        // the square root of a symmetric 2×2 matrix with positive eigenvalues, in closed form
        let sqrt = |m: DMat2| {
            let r = m.determinant().sqrt();
            (m + DMat2::IDENTITY * r) * (1.0 / (trace(m) + 2.0 * r).sqrt())
        };
        for a in matrices(100, 6) {
            let p = split(a);
            let mut last = None;
            for eps in [1e-2, 5e-3] {
                let m = DMat2::IDENTITY + a * eps;
                let big_p = sqrt(m.transpose() * m);
                let q = m * big_p.inverse();
                assert!(close(q.transpose() * q, DMat2::IDENTITY, 1e-9) && close(q * big_p, m, 1e-12));
                let err = size(big_p - DMat2::IDENTITY - p.s * eps).max(size(q - DMat2::IDENTITY - p.k * eps));
                assert!(err < 3.0 * eps * eps * size(a).max(1.0).powi(2), "{a:?}: {err} at {eps}");
                if let Some(before) = last {
                    if before > 1e-10 {
                        assert!(err < 0.3 * before, "{a:?}: {before} then {err}");
                    }
                }
                last = Some(err);
            }
        }
    }

    /// In space: a skew-symmetric K is the cross product with ω = (k₃₂, k₁₃, k₂₁).
    #[test]
    fn in_space_the_spin_is_a_cross_product() {
        let mut rng = Rng::new(9);
        for _ in 0..100 {
            let a = glam::DMat3::from_cols_array(&[0; 9].map(|_| rng.range(-2.0, 2.0) as f64));
            let k = (a - a.transpose()) * 0.5;
            let omega = glam::DVec3::new(k.y_axis.z, k.z_axis.x, k.x_axis.y);
            let x = glam::DVec3::new(rng.range(-2.0, 2.0) as f64, rng.range(-2.0, 2.0) as f64, rng.range(-2.0, 2.0) as f64);
            assert!((k * x - omega.cross(x)).length() < 1e-14);
        }
    }

    /// The closed-form exponential agrees with its series.
    #[test]
    fn the_exponential_is_right() {
        let series = |a: DMat2, t: f64| {
            let (mut sum, mut term) = (DMat2::IDENTITY, DMat2::IDENTITY);
            for k in 1..80 {
                term = term * a * (t / k as f64);
                sum += term;
            }
            sum
        };
        for a in matrices(200, 7) {
            for t in [0.0, 0.25, 1.0, -0.6] {
                let (e, s) = (expm(a, t), series(a, t));
                assert!(close(e, s, 1e-10 * size(s).max(1.0)), "{a:?} at {t}");
            }
        }
    }

    /// The scene as an experiment's button leaves it, run at 60 frames a second until
    /// it has played.
    fn run(k: usize, seconds: f32) -> SymSkew {
        let mut scene = SymSkew::new(None);
        scene.set_up(k);
        for _ in 0..(seconds * 60.0) as usize {
            scene.advance(1.0 / 60.0, 1.0 / 60.0);
            if scene.glide.arrived() && !scene.playing() && !scene.waiting {
                break;
            }
        }
        scene
    }

    #[test]
    fn a_pure_spin_does_what_its_caption_says() {
        let scene = run(0, 30.0);
        let Setting { a, x, t } = scene.shown();
        let p = split(a);
        assert!(p.s == DMat2::ZERO && p.w == 0.8 && close(a, p.k, 0.0) && t == 1.0);
        assert_eq!(kind(a), Kind::Centre);
        assert!(matches!(eigen(a), Eigen::Complex { re, im } if re == 0.0 && (im - 0.8).abs() < 1e-15));
        assert_eq!(eigen_text(eigen(a)), r"$\lambda = \pm 0.80i$");
        // Kx at right angles to x; the circle turned by wt = 0.8 radians, 46°, area kept
        assert!(x.dot(a * x).abs() < 1e-15);
        let (sin, cos) = 0.8f64.sin_cos();
        assert!(close(expm(a, t), rows([cos, -sin, sin, cos]), 1e-14));
        assert_eq!(num(0.8f64.to_degrees(), 0), "46");
        assert!((expm(a, t).determinant() - 1.0).abs() < 1e-14 && trace(a) == 0.0);
        // every point goes round a circle at the rate 0.8
        for q in [DVec2::new(1.0, 0.0), DVec2::new(-0.3, 2.0)] {
            assert!(((expm(a, 2.7) * q).length() - q.length()).abs() < 1e-13);
        }
    }

    #[test]
    fn the_shear_does_what_its_caption_says() {
        let scene = run(1, 30.0);
        let a = scene.shown().a;
        let p = split(a);
        assert!(close(p.s, rows([0.0, 0.5, 0.5, 0.0]), 0.0) && p.w == -0.5);
        assert!(p.s1 == 0.5 && p.s2 == -0.5);
        let axis = p.axis.expect("an axis");
        assert!((axis.y.atan2(axis.x).to_degrees() - 45.0).abs() < 1e-12);
        // each point slides sideways as fast as it is high
        for h in [-1.0, 0.5, 2.0] {
            assert_eq!(a * DVec2::new(0.7, h), DVec2::new(h, 0.0));
        }
        // on the x axis S and K cancel; on the y axis they add up to the slide
        for u in [-2.0, 0.4, 1.5] {
            let (on_x, on_y) = (DVec2::new(u, 0.0), DVec2::new(0.0, u));
            assert!(p.s * on_x + p.k * on_x == DVec2::ZERO && p.s * on_x != DVec2::ZERO);
            assert!(p.s * on_y == p.k * on_y && p.s * on_y + p.k * on_y == a * on_y);
        }
        // a tie: λ = 0 twice, one eigenline (the x axis)
        assert!(p.w.abs() == p.half_spread() && verdict(a) == Verdict::Tie);
        assert!(matches!(eigen(a), Eigen::Real { l1, l2, lines: Lines::One(v) } if l1 == 0.0 && l2 == 0.0 && v.y.abs() < 1e-15));
        assert_eq!(contest(a), r"$|w| = (s_1 - s_2)/2 = 0.50$");
    }

    #[test]
    fn the_quadratic_form_does_what_its_caption_says() {
        let mut scene = SymSkew::new(None);
        scene.set_up(2);
        assert!(scene.shadows);
        let p = split(rows(START));
        let (mut most, mut least) = ((f64::MIN, DVec2::ZERO), (f64::MAX, DVec2::ZERO));
        let mut went = 0.0f64;
        let mut last = None;
        for _ in 0..(60.0 * (ROUND_SECS + 5.0)) as usize {
            scene.advance(1.0 / 60.0, 1.0 / 60.0);
            let Setting { a, x, .. } = scene.shown();
            // the tips of Ax and Sx have the same shadow on the line of x
            let unit = x.normalize();
            assert!((unit.dot(a * x) - unit.dot(p.s * x)).abs() < 1e-12);
            assert!((x.length() - 1.25).abs() < 1e-12 || !scene.glide.arrived());
            let q = x.dot(a * x);
            if q > most.0 {
                most = (q, x);
            }
            if q < least.0 {
                least = (q, x);
            }
            if let Some(l) = last {
                went += unit.angle_to(l).abs();
            }
            last = Some(unit);
        }
        assert!(!scene.playing() && (went - TAU64).abs() < 0.02, "went {went}");
        assert!(scene.shown().x.distance(DVec2::from(START_X)) < 1e-9);
        // largest s1 |x|^2 = 1.25 on the orange line, least s2 |x|^2 = -1.88 on the purple
        assert!(p.s1 == 0.8 && (p.s2 + 1.2).abs() < 1e-15);
        assert!((most.0 - 1.25).abs() < 1e-3 && num(1.25, 2) == "1.25" && num(p.s1 * 1.5625, 2) == "1.25");
        assert!((least.0 + 1.875).abs() < 1e-3 && num(p.s2 * 1.5625, 2) == "−1.88");
        let axis = p.axis.unwrap();
        assert!(most.1.normalize().perp_dot(axis).abs() < 0.02 && least.1.normalize().dot(axis).abs() < 0.02);
        // A's own eigenvalues are complex
        assert_eq!(verdict(rows(START)), Verdict::Spin);
    }

    #[test]
    fn the_transpose_does_what_its_caption_says() {
        let mut scene = SymSkew::new(None);
        scene.set_up(3);
        let a0 = rows(START);
        let p0 = split(a0);
        let mut signs = Vec::new();
        for _ in 0..(60.0 * (TRANSPOSE_SECS + 5.0)) as usize {
            scene.advance(1.0 / 60.0, 1.0 / 60.0);
            let p = split(scene.shown().a);
            // S does not change at all on the way
            assert!(close(p.s, p0.s, 1e-15));
            if signs.last() != Some(&(p.w > 0.0)) {
                signs.push(p.w > 0.0);
            }
        }
        let a = scene.shown().a;
        assert!(!scene.playing() && close(a, a0.transpose(), 1e-15));
        assert!(p0.w == 1.2 && (split(a).w + 1.2).abs() < 1e-15 && signs == [true, false]);
        // the same eigenvalues, -0.20 ± 0.66i; it spirals in, anticlockwise before, clockwise after
        assert_eq!(eigen_text(eigen(a0)), r"$\lambda = -0.20 \pm 0.66i$".replace('-', "−"));
        let (Eigen::Complex { re: r1, im: i1 }, Eigen::Complex { re: r2, im: i2 }) = (eigen(a), eigen(a0)) else { panic!("{:?}", eigen(a)) };
        assert!((r1 - r2).abs() < 1e-15 && (i1 - i2).abs() < 1e-14);
        assert!(kind(a0) == Kind::SpiralSink && kind(a) == Kind::SpiralSink);
        for k in 0..24 {
            let theta = k as f64 * TAU64 / 24.0;
            assert!(turning(a0, theta) > 0.0 && turning(a, theta) < 0.0);
        }
        // half-way A is S, its eigenlines at right angles
        let half = a0 + (a0.transpose() - a0) * 0.5;
        assert!(close(half, p0.s, 1e-15));
        assert!(matches!(eigen(half), Eigen::Real { lines: Lines::Two(u, v), .. } if u.dot(v).abs() < 1e-12));
    }

    #[test]
    fn stretch_against_spin_does_what_its_caption_says() {
        let mut scene = SymSkew::new(None);
        scene.set_up(4);
        let base = rows(TRIALS[4].a);
        let p0 = split(base);
        assert!((p0.s1 - 0.4).abs() < 1e-15 && (p0.s2 + 1.6).abs() < 1e-15 && p0.w == 0.0 && (p0.half_spread() - 1.0).abs() < 1e-15);
        let mut seen: Vec<Kind> = Vec::new();
        let mut last_angle = f64::MAX;
        for _ in 0..(60.0 * (spin_length() + 5.0)) as usize {
            scene.advance(1.0 / 60.0, 1.0 / 60.0);
            if scene.waiting {
                continue;
            }
            let a = scene.shown().a;
            let p = split(a);
            // S stays as it is
            assert!(close(p.s, p0.s, 1e-15));
            let k = kind(a);
            if seen.last() != Some(&k) {
                seen.push(k);
            }
            // the eigenlines swing towards each other as w grows
            if let Eigen::Real { lines: Lines::Two(u, v), .. } = eigen(a) {
                let angle = u.dot(v).abs().min(1.0).acos();
                if p.w > 0.0 {
                    assert!(angle <= last_angle + 1e-9, "{angle} after {last_angle}");
                }
                last_angle = angle;
            }
        }
        assert!(!scene.playing());
        // (a line at rest, det A = 0, for the moment w passes 0.8, if a frame falls on it)
        seen.retain(|k| *k != Kind::RestIn);
        assert_eq!(seen, [Kind::Saddle, Kind::Sink, Kind::ImproperSink, Kind::SpiralSink]);
        // the stops: a saddle with lines at right angles; past 0.8 a sink; at 1 a tie; a spiral
        let at = |w: f64| base + quarter_turn() * w;
        assert!(matches!(eigen(at(0.0)), Eigen::Real { lines: Lines::Two(u, v), .. } if u.dot(v).abs() < 1e-12) && kind(at(0.0)) == Kind::Saddle);
        assert!(kind(at(0.79)) == Kind::Saddle && kind(at(0.81)) == Kind::Sink && kind(at(0.9)) == Kind::Sink);
        assert!(matches!(eigen(at(0.8)), Eigen::Real { l1, .. } if l1.abs() < 1e-12));
        assert!(verdict(at(1.0)) == Verdict::Tie && kind(at(1.0)) == Kind::ImproperSink);
        assert!(verdict(at(1.4)) == Verdict::Spin && kind(at(1.4)) == Kind::SpiralSink && verdict(at(0.99)) == Verdict::Stretch);
        assert_eq!(scene.shown().a, at(1.4));
        // the chart: A's curve is K's level line plus S's wave; it meets 0 twice, then
        // touches it, then stays above it
        let crossings = |m: DMat2| (0..720).filter(|i| {
            let (u, v) = (turning(m, *i as f64 * PI64 / 720.0), turning(m, (*i + 1) as f64 * PI64 / 720.0));
            u.signum() != v.signum()
        }).count();
        let lowest = |m: DMat2| (0..=3600).map(|i| turning(m, i as f64 * PI64 / 3600.0)).fold(f64::MAX, f64::min);
        for w in [0.0, 0.5, 0.9, 0.99] {
            assert_eq!(crossings(at(w)), 2, "{w}");
        }
        assert!(lowest(at(1.0)).abs() < 1e-6 && lowest(at(1.0)) > -1e-12);
        assert!(lowest(at(1.4)) > 0.39 && crossings(at(1.4)) == 0 && crossings(at(1.01)) == 0);
    }

    /// The quick choices' notes.
    #[test]
    fn the_quick_choices_do_what_their_notes_say() {
        let p = split(rows(START));
        assert!(p.s1 == 0.8 && (p.s2 + 1.2).abs() < 1e-15 && p.w == 1.2);
        assert_eq!(num(p.axis.unwrap().y.atan2(p.axis.unwrap().x).to_degrees(), 1), "26.6");
        assert_eq!(kind(rows(START)), Kind::SpiralSink);
        let sym = split(rows(PRESETS[1].1));
        assert!(sym.k == DMat2::ZERO && matches!(eigen(rows(PRESETS[1].1)), Eigen::Real { lines: Lines::Two(u, v), .. } if u.dot(v).abs() < 1e-12));
        let skew = rows(PRESETS[2].1);
        assert!(split(skew).s == DMat2::ZERO && kind(skew) == Kind::Centre && split(skew).w == 1.0);
        let shear = rows(PRESETS[3].1);
        assert!(split(shear).s * DVec2::X + split(shear).k * DVec2::X == DVec2::ZERO);
        let even = rows(PRESETS[4].1);
        let pe = split(even);
        assert!(pe.axis.is_none() && pe.s == DMat2::IDENTITY * -0.3 && close(pe.s * pe.k, pe.k * pe.s, 0.0));
        for t in [0.5f64, 2.0] {
            let (sin, cos) = t.sin_cos();
            assert!(close(expm(even, t), rows([cos, -sin, sin, cos]) * (-0.3 * t).exp(), 1e-14));
            assert!(close(expm(even, t), expm(pe.s, t) * expm(pe.k, t), 1e-14));
        }
    }

    /// Degenerate input: every number drawn or read out is finite, whatever A, t and x.
    #[test]
    fn degenerate_input_leaves_everything_finite() {
        let mats = [DMat2::ZERO, DMat2::IDENTITY * 1e-13, rows([0.0, 1.0, 0.0, 0.0]), rows([ENTRY_MAX; 4]), rows([-ENTRY_MAX, ENTRY_MAX, -ENTRY_MAX, ENTRY_MAX]), DMat2::IDENTITY * -ENTRY_MAX, quarter_turn() * ENTRY_MAX];
        for a in mats {
            let p = split(a);
            assert!([p.s1, p.s2, p.w].iter().all(|v| v.is_finite()));
            for t in [0.0, 1e-9, 1.0, T_MAX] {
                for m in [a, p.s, p.k] {
                    let e = expm(m, t);
                    assert!(e.as_mat2().is_finite());
                    let image: Vec<DVec2> = (0..=96).map(|i| e * DVec2::from_angle(i as f64 / 96.0 * TAU64)).collect();
                    let (lo, hi) = bounds();
                    assert!(runs(&image, lo, hi).iter().flatten().all(|q| q.as_vec2().is_finite() && q.abs().max_element() <= REACH + 1e-9));
                    assert!(clip_polygon(&image[..96], lo, hi).iter().all(|q| q.as_vec2().is_finite()));
                }
            }
            for text in [contest(a), eigen_text(eigen(a)), cell(a.x_axis.x)] {
                assert!(!text.contains("NaN") && !text.contains("inf") && !text.contains('?'), "{text}");
            }
            assert!(field_scale(a, &p).is_finite());
            let mut flow = Flow::new(None, 1);
            let plane = Plane { centre: Vec2::ZERO, half: 1.0 };
            for k in 0..300 {
                let dt = [1.0 / 60.0, 1.0 / 144.0, 0.0, 0.3, f32::NAN, 1.0 / 30.0][k % 6];
                flow.advance(a, dt, pal::WHITE, plane, None);
                assert!(flow.pos.iter().all(|q| q.is_finite()) && flow.owed >= 0.0 && flow.owed < ROW_EVERY);
                assert!(flow.row.iter().all(|r| r.pos.iter().all(|v| v.is_finite())));
            }
        }
    }

    /// The streaks fill the square evenly, whatever the flow, and none is drawn outside it.
    #[test]
    fn the_streaks_fill_the_square_and_stay_in_it() {
        let plane = Plane { centre: Vec2::new(3.0, -1.0), half: 2.0 };
        for a in [rows(START), split(rows(START)).s, split(rows(START)).k, rows([-3.0, 0.0, 0.0, -2.5]), rows([0.0, 1.0, 0.0, 0.0])] {
            let mut flow = Flow::new(None, 3);
            let mut parts = [0.0f64; 16];
            for frame in 0..900 {
                flow.advance(a, 1.0 / 60.0, pal::WHITE, plane, None);
                for (i, point) in flow.row.iter().enumerate() {
                    let w = Vec2::new(point.pos[0], point.pos[1]);
                    let q = plane.of(w);
                    if point.width > 0.0 {
                        assert!(q.abs().max_element() <= REACH + 1e-4, "{a:?}: drawn at {q:?}");
                    }
                    let at = ((flow.pos[i].as_dvec2() + DVec2::splat(REACH)) / (2.0 * REACH)).as_vec2();
                    if frame >= 240 && at.cmpge(Vec2::ZERO).all() && at.cmplt(Vec2::ONE).all() {
                        let alpha = point.color.0[3] as f64 / flow.bright[i] as f64;
                        parts[(at.x * 4.0) as usize + 4 * (at.y * 4.0) as usize] += alpha;
                    }
                }
            }
            let mean = parts.iter().sum::<f64>() / 16.0;
            for (k, part) in parts.iter().enumerate() {
                assert!((0.55..=1.5).contains(&(part / mean)), "{a:?}: part {k} has {:.2} of its share", part / mean);
            }
        }
    }

    #[test]
    fn the_layout_fits_every_canvas() {
        for (w, h) in [(1080.0, 1000.0), (1300.0, 900.0), (390.0, 800.0), (390.0, 560.0), (800.0, 600.0), (1600.0, 1000.0), (240.0, 180.0), (620.0, 480.0), (580.0, 700.0)] {
            for inset in [0.0, 46.0] {
                let view = Vec2::new(w, h);
                let lay = Layout::of(view, inset);
                let ppu = (h / STAGE_H).min(w / 2.0);
                let px = |p: Vec2| Vec2::new(p.x * ppu + w * 0.5, h * 0.5 - p.y * ppu);
                for (k, plane) in lay.planes.iter().enumerate() {
                    let (a, b) = (px(plane.lo()), px(plane.hi()));
                    let (min, max) = (a.min(b), a.max(b));
                    assert!(min.x >= -0.5 && max.x <= w + 0.5 && min.y >= inset - 0.5 && max.y <= h + 0.5, "{w}x{h} +{inset}: picture {k} at {min:?}..{max:?}");
                }
                // no two pictures overlap
                for i in 0..3 {
                    for j in i + 1..3 {
                        let (p, q) = (lay.planes[i], lay.planes[j]);
                        let apart = (p.centre - q.centre).abs();
                        assert!(apart.x >= p.half + q.half - 1e-4 || apart.y >= p.half + q.half - 1e-4, "{w}x{h}: {i} and {j}");
                    }
                }
            }
        }
    }
}

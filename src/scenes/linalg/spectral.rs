//! Symmetric matrices and the spectral theorem. A symmetric matrix stretches the plane
//! along two axes at right angles, its eigenvectors; written out, it is the sum of the
//! projections onto those axes, each times its eigenvalue. A film builds Ax that way
//! (split x into its shadows on the axes, stretch each by its eigenvalue, add), and the
//! same film with other numbers on the same axes gives A², A⁻¹ and √A. A skew part,
//! added, breaks the right angle, and then the eigenlines themselves.

use super::*;
use glam::{DMat2, DVec2};

pub const DEF: SceneDef = SceneDef {
    id: "spectral",
    title: "Symmetric matrices: the spectral theorem",
    group: FACTOR,
    blurb: "A symmetric matrix stretches along perpendicular axes: it is a sum of projections, each times an eigenvalue.",
    make: |_| Box::new(Spectral::new()),
};

// ----- the arithmetic, in double precision -------------------------------------------

/// The largest eigenvalue either way that the sliders and the dots allow.
const L_MAX: f64 = 2.5;
/// The largest skew part either way.
const K_MAX: f64 = 1.5;
/// How far out x may be dragged.
const X_MAX: f64 = 4.0;
/// A skew part smaller than this is none: what a glide back to 0 leaves, or a slider a
/// hair from 0. (Its square is far above what `eigen` takes for nothing.)
const K_NONE: f64 = 1e-4;
/// A dragged eigenvalue this near 0, or the other one, is taken to be exactly that:
/// so that a squash and a double eigenvalue can be reached by hand.
const SNAP: f64 = 0.03;
/// Seconds each of the film's three steps takes.
const STEP: f64 = 1.5;
/// The orange axis's angle in most of the experiments: the one of (0.8, 0.6), which
/// makes the projections' entries short (0.64, 0.48, 0.36).
const ANGLE: f64 = 0.643_501_108_793_284_4;

/// The quarter turn: the skew part is k J.
const J: DMat2 = DMat2::from_cols(DVec2::new(0.0, 1.0), DVec2::new(-1.0, 0.0));

/// u vᵀ.
fn outer(u: DVec2, v: DVec2) -> DMat2 {
    DMat2::from_cols(u * v.x, u * v.y)
}

/// The size of the largest entry.
fn size(m: DMat2) -> f64 {
    m.to_cols_array().iter().fold(0.0f64, |s, v| s.max(v.abs()))
}

fn ease(t: f64) -> f64 {
    let t = if t.is_nan() { 0.0 } else { t.clamp(0.0, 1.0) };
    t * t * (3.0 - 2.0 * t)
}

/// Everything the picture is drawn from: what the sliders, the dots and the film set.
#[derive(Clone, Copy, Debug, PartialEq)]
struct Setting {
    /// The eigenvalues of the symmetric part: the orange axis's, then the purple's.
    l: [f64; 2],
    /// The angle of the orange axis, in radians. Any angle, not only within a half
    /// turn: the dots stay on whichever side of 0 they were dragged to.
    theta: f64,
    /// The skew part: A = S + k J.
    k: f64,
    x: DVec2,
    /// How far the film has got: 0 (x alone), 1 (its shadows), 2 (each stretched), 3 (added).
    stage: f64,
}

impl Setting {
    /// The unit vectors along the orange and the purple axis, a quarter turn apart.
    fn axes(&self) -> [DVec2; 2] {
        let q = DVec2::from_angle(self.theta);
        [q, q.perp()]
    }

    /// The symmetric part, λ₁ q₁ q₁ᵀ + λ₂ q₂ q₂ᵀ.
    fn sym(&self) -> DMat2 {
        let [q1, q2] = self.axes();
        outer(q1, q1) * self.l[0] + outer(q2, q2) * self.l[1]
    }

    fn a(&self) -> DMat2 {
        self.sym() + J * self.k
    }

    fn symmetric(&self) -> bool {
        self.k == 0.0
    }

    fn numbers(&self) -> [f64; 7] {
        [self.l[0], self.l[1], self.theta, self.k, self.x.x, self.x.y, self.stage]
    }

    fn from_numbers(v: [f64; 7]) -> Setting {
        // (what a glide back to a symmetric matrix leaves of the skew part is none)
        let k = if v[3].abs() < K_NONE { 0.0 } else { v[3] };
        Setting { l: [v[0], v[1]], theta: v[2], k, x: DVec2::new(v[4], v[5]), stage: v[6] }
    }
}

/// The eigenvalues of A and the lines they keep.
#[derive(Clone, Copy, Debug, PartialEq)]
enum Eigen {
    /// Two eigenlines: each one's eigenvalue `mu` and unit direction `v`, and `w`, the
    /// rows of V⁻¹ (V has the v's for columns), which read off how much of each line
    /// a vector is made of: x = (w₁·x) v₁ + (w₂·x) v₂. For a symmetric matrix w = v,
    /// and the shadows are dropped at right angles.
    Two { mu: [f64; 2], v: [DVec2; 2], w: [DVec2; 2] },
    /// The two eigenlines have met: one eigenvalue, twice, and a single line.
    One { mu: f64, v: DVec2 },
    /// No real eigenvalue: re ± im i.
    Complex { re: f64, im: f64 },
}

/// The eigenvalues and eigenlines of A = S + kJ, in closed form.
///
/// In the coordinates of S's axes A is [[λ₁, −k], [k, λ₂]]. With m = (λ₁ + λ₂)/2 and
/// d = (λ₁ − λ₂)/2 its eigenvalues are m ± √(d² − k²): two real ones while |k| < |d|,
/// one (twice) at |k| = |d|, a complex pair beyond. The eigenvector of the one that
/// becomes λ₁ as k goes to 0 is (|d| + r, ±k), the other's (±k, |d| + r), with
/// r = √(d² − k²): each leans towards the other by the same angle, and the angle
/// between the two lines is arccos(|k|/|d|).
fn eigen(s: &Setting) -> Eigen {
    let [q1, q2] = s.axes();
    if s.symmetric() {
        return Eigen::Two { mu: s.l, v: [q1, q2], w: [q1, q2] };
    }
    let (m, d) = (0.5 * (s.l[0] + s.l[1]), 0.5 * (s.l[0] - s.l[1]));
    let quarter = d * d - s.k * s.k;
    let scale = s.l[0].abs().max(s.l[1].abs()).max(s.k.abs());
    let flat = 1e-12 * scale * scale;
    if quarter < -flat {
        return Eigen::Complex { re: m, im: (-quarter).sqrt() };
    }
    let sign = if d < 0.0 { -1.0 } else { 1.0 };
    let big = d.abs();
    let unit = |v: DVec2| {
        let v = v.normalize_or_zero();
        if v == DVec2::ZERO {
            q1
        } else {
            v
        }
    };
    if quarter <= flat {
        return Eigen::One { mu: m, v: unit(q1 * big + q2 * (sign * s.k)) };
    }
    let r = quarter.sqrt();
    let v1 = unit(q1 * (big + r) + q2 * (sign * s.k));
    let v2 = unit(q1 * (sign * s.k) + q2 * (big + r));
    let det = v1.perp_dot(v2);
    // the eigenvalue further from 0 as m ± r, the other as det A over it: m ∓ r would
    // lose its digits to cancellation when it is small
    let mut mu = [m + sign * r, m - sign * r];
    let far = if mu[0].abs() >= mu[1].abs() { 0 } else { 1 };
    if mu[far] != 0.0 {
        mu[1 - far] = (s.l[0] * s.l[1] + s.k * s.k) / mu[far];
    }
    Eigen::Two { mu, v: [v1, v2], w: [DVec2::new(v2.y, -v2.x) / det, DVec2::new(-v1.y, v1.x) / det] }
}

/// The angle between the two eigenlines, in degrees (0 to 90).
fn lines_apart(v: [DVec2; 2]) -> f64 {
    v[0].dot(v[1]).abs().min(1.0).acos().to_degrees()
}

/// What is applied to x: A itself, or a function of it, which acts on the eigenvalues
/// and keeps the eigenlines.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Fun {
    Same,
    Square,
    Inverse,
    Root,
}

const FUNS: [Fun; 4] = [Fun::Same, Fun::Square, Fun::Inverse, Fun::Root];

impl Fun {
    /// What it does to one eigenvalue; nothing where it has no (real) value.
    fn of(self, l: f64) -> Option<f64> {
        match self {
            Fun::Same => Some(l),
            Fun::Square => Some(l * l),
            Fun::Inverse => (l != 0.0).then(|| 1.0 / l),
            Fun::Root => (l >= 0.0).then(|| l.sqrt()),
        }
    }

    /// Its name, as markup to go between dollars.
    fn name(self) -> &'static str {
        match self {
            Fun::Same => "A",
            Fun::Square => "A^2",
            Fun::Inverse => "A^{-1}",
            Fun::Root => r"\sqrt{A}",
        }
    }

    /// The same done to the eigenvalue λᵢ (i = 1 or 2), as markup.
    fn on(self, i: usize) -> String {
        match self {
            Fun::Same => format!(r"\lambda_{i}"),
            Fun::Square => format!(r"\lambda_{i}^2"),
            Fun::Inverse => format!(r"\lambda_{i}^{{-1}}"),
            Fun::Root => format!(r"\sqrt{{\lambda_{i}}}"),
        }
    }
}

/// Why a function of A cannot be built along the eigenlines.
#[derive(Clone, Copy, Debug, PartialEq)]
enum Why {
    /// There are not two eigenlines.
    NoLines,
    /// Eigenvalue i is 0 (no inverse) or negative (no square root on these lines).
    Zero(usize),
    Negative(usize),
}

/// f(A) and f of each eigenvalue, if the picture can build it along two eigenlines.
/// f(A) = f(μ₁) P₁ + f(μ₂) P₂, worked out without the projections, which grow without
/// bound as two eigenlines close: A², A⁻¹, and √A = (A + √(μ₁μ₂) I)/(√μ₁ + √μ₂) (its
/// square is A + 2√(μ₁μ₂) A/(√μ₁ + √μ₂) … by Cayley–Hamilton, A² = (tr A) A − (det A) I,
/// which makes it A; see the test).
fn applied(s: &Setting, e: &Eigen, fun: Fun) -> Result<(DMat2, [f64; 2]), Why> {
    let Eigen::Two { mu, .. } = e else { return Err(Why::NoLines) };
    let mut fs = [0.0; 2];
    for i in 0..2 {
        fs[i] = match fun.of(mu[i]) {
            Some(v) => v,
            None if fun == Fun::Inverse => return Err(Why::Zero(i)),
            None => return Err(Why::Negative(i)),
        };
    }
    let a = s.a();
    let m = match fun {
        Fun::Same => a,
        Fun::Square => a * a,
        // (A⁻¹ = (tr A I − A)/det A, with det A = λ₁λ₂ + k² from the setting itself: from
        // the entries it would be a difference of two nearly equal numbers when A is
        // nearly flat)
        Fun::Inverse => (DMat2::IDENTITY * (s.l[0] + s.l[1]) - a) * (1.0 / (s.l[0] * s.l[1] + s.k * s.k)),
        Fun::Root => {
            let t = fs[0] + fs[1];
            if t > 0.0 {
                (a + DMat2::IDENTITY * (fs[0] * fs[1])) * (1.0 / t)
            } else {
                DMat2::ZERO
            }
        }
    };
    if !(m.is_finite() && fs.iter().all(|v| v.is_finite())) {
        return Err(Why::Zero(if mu[0].abs() <= mu[1].abs() { 0 } else { 1 }));
    }
    Ok((m, fs))
}

/// The map drawn at a stage of the film: the identity until the stretch, then the
/// straight blend from I to f(A). Along each eigenline that is the factor going evenly
/// from 1 to f(μ): Σ (1 + (f(μᵢ) − 1) s) Pᵢ = (1 − s) I + s f(A), since P₁ + P₂ = I.
fn film(f_a: DMat2, stage: f64) -> DMat2 {
    let s = ease(stage - 1.0);
    DMat2::IDENTITY * (1.0 - s) + f_a * s
}

/// The film's construction for the point x, along two eigenlines.
#[derive(Clone, Copy, Debug)]
struct Built {
    /// The shadows Pᵢx = cᵢvᵢ, where x = c₁v₁ + c₂v₂ (for a symmetric matrix cᵢ = qᵢ·x,
    /// the shadows' signed lengths), and the same stretched by the factor reached so far.
    feet: [DVec2; 2],
    scaled: [DVec2; 2],
    /// The two stretched shadows added: where the drawn map takes x.
    corner: DVec2,
}

fn build(v: [DVec2; 2], w: [DVec2; 2], fs: [f64; 2], x: DVec2, stage: f64) -> Built {
    let s2 = ease(stage - 1.0);
    let c = [w[0].dot(x), w[1].dot(x)];
    let feet = [v[0] * c[0], v[1] * c[1]];
    let g = [1.0 + (fs[0] - 1.0) * s2, 1.0 + (fs[1] - 1.0) * s2];
    let scaled = [feet[0] * g[0], feet[1] * g[1]];
    Built { feet, scaled, corner: scaled[0] + scaled[1] }
}

// ----- experiments --------------------------------------------------------------------

/// What an experiment moves once it has been set up.
#[derive(Clone, Copy, Debug, PartialEq)]
enum Ramp {
    /// The axes turn by an eighth of a turn, clockwise (that way round x's shadows
    /// stay on the sides of 0 away from the dots).
    Turn,
    /// The skew part grows from 0 past the point where the eigenlines meet.
    Twist,
}

impl Ramp {
    /// (seconds in, value): the turn's angle added to where it started, the twist's k.
    fn keys(self) -> &'static [(f32, f64)] {
        match self {
            Ramp::Turn => &[(0.0, 0.0), (3.5, -std::f64::consts::FRAC_PI_4)],
            Ramp::Twist => &[(0.0, 0.0), (3.5, 0.75), (4.9, 0.75), (6.6, 1.0)],
        }
    }

    fn length(self) -> f32 {
        self.keys().last().map_or(0.0, |k| k.0)
    }

    /// Its value `t` seconds in: eased from each key to the next.
    fn at(self, t: f32) -> f64 {
        let keys = self.keys();
        for pair in keys.windows(2) {
            let ((t0, v0), (t1, v1)) = (pair[0], pair[1]);
            if t <= t1 {
                return v0 + (v1 - v0) * ease(((t - t0) / (t1 - t0).max(1e-6)) as f64);
            }
        }
        keys.last().map_or(0.0, |k| k.1)
    }
}

/// A ramp being run: which, how far into it (seconds), and the angle it turns from.
#[derive(Clone, Copy, Debug)]
struct Running {
    ramp: Ramp,
    t: f32,
    from: f64,
}

struct Trial {
    label: &'static str,
    look_for: &'static str,
    l: [f64; 2],
    theta: f64,
    k: f64,
    x: [f64; 2],
    fun: Fun,
    /// The step the film is set to, and whether it plays on from there.
    stage: f64,
    play: bool,
    ramp: Option<Ramp>,
}

/// x on the line the eigenlines of the twist close in on: 1.5 times the unit vector at
/// 45° from the orange axis (`the_twist_does_what_its_caption_says` checks it).
const ON_THE_MEETING_LINE: [f64; 2] = [0.212_132_034_355_964_26, 1.484_924_240_491_749_8];

const TRIALS: [Trial; 5] = [
    Trial {
        label: "a sum of two projections",
        look_for: r"$x = (0.4, 1.55)$ is split into its shadows on the two axes, dropped at right angles: $P_1 x = (1, 0.75)$ on the orange axis and $P_2 x = (-0.6, 0.8)$ on the purple one, which add up to $x$ again. Each is multiplied by its eigenvalue, 2 and 0.5, and the two are added: $Ax = (1.7, 1.9)$. The matrices on the picture say the same for every $x$ at once: $A = 2P_1 + 0.5P_2$.",
        l: [2.0, 0.5],
        theta: ANGLE,
        k: 0.0,
        x: [0.4, 1.55],
        fun: Fun::Same,
        stage: 0.0,
        play: true,
        ramp: None,
    },
    Trial {
        label: "equal eigenvalues",
        look_for: r"$\lambda_1 = \lambda_2 = 1.5$, so $A = 1.5I$: every direction is an eigenvector, and the circle becomes a circle 1.5 times as large. The axes turn by 45°; the shadows of $x$ change as they turn, yet the stretched shadows always add up to the same $Ax = 1.5x$, and the numbers of $A$ do not change. Any two axes at right angles will do. $P_1$ and $P_2$ change as they turn, and always add up to $I$.",
        l: [1.5, 1.5],
        theta: ANGLE,
        k: 0.0,
        x: [0.4, 1.55],
        fun: Fun::Same,
        stage: 3.0,
        play: false,
        ramp: Some(Ramp::Turn),
    },
    Trial {
        label: "a flip, and a squash",
        look_for: r"$\lambda_2 = -0.5$: along the purple axis everything is turned round and halved. Watch the stretch: the purple factor goes from 1 down to $-0.5$, and where it passes 0 the map is $1.67\,P_1$, a projection times a number, which squashes the whole plane flat onto the orange axis. Then the plane opens out mirrored: the letter comes out mirror-written, $\det A = \lambda_1 \lambda_2 = -1$, and $Ax$ lands on the other side of the orange axis from $x$.",
        l: [2.0, -0.5],
        theta: ANGLE,
        k: 0.0,
        x: [0.4, 1.55],
        fun: Fun::Same,
        stage: 0.0,
        play: true,
        ramp: None,
    },
    Trial {
        label: "break the symmetry",
        look_for: r"A skew part $kJ$ is added, $k$ from 0 to 1, and the two corner entries come apart: $0.72 - k$ above the diagonal, $0.72 + k$ below. The eigenlines lean towards each other like closing scissors, $\arccos(2k/(\lambda_1 - \lambda_2))$ apart: 48.2° at $k = 0.5$. At $k = 0.75 = (\lambda_1 - \lambda_2)/2$ they meet in one line, and past it no line is kept at all: at $k = 1$ the eigenvalues are $1.25 \pm 0.66i$. $x$ is on the line they close in on; anywhere else, its two parts along them would grow without bound as they meet.",
        l: [2.0, 0.5],
        theta: ANGLE,
        k: 0.0,
        x: ON_THE_MEETING_LINE,
        fun: Fun::Same,
        stage: 3.0,
        play: false,
        ramp: Some(Ramp::Twist),
    },
    Trial {
        label: "the inverse and the square root",
        look_for: r"The same axes and the same shadows, other numbers. $A^{-1}$ multiplies the orange shadow by $1/2.25 = 0.44$ and the purple one by $1/0.64 = 1.56$: it undoes what $A$ does along each axis, so it undoes $A$, and its ellipse is long where the ellipse of $A$ (faint) is short. Then choose $\sqrt{A}$ under “apply”: $\sqrt{2.25} = 1.5$ and $\sqrt{0.64} = 0.8$, and doing it twice gives $A$. The inverse needs no eigenvalue to be 0; this square root needs none to be negative.",
        l: [2.25, 0.64],
        theta: ANGLE,
        k: 0.0,
        x: [0.4, 1.55],
        fun: Fun::Inverse,
        stage: 1.0,
        play: true,
        ramp: None,
    },
];

// ----- gliding from one setting to the next ------------------------------------------------

/// How fast a change of setting is glided to: each of the two eases one after the other
/// covers two thirds of what is left in this many seconds (as in `expm`).
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
        // (no time, or a time that is not a number, moves nothing)
        if self.arrived() || dt.is_nan() || dt <= 0.0 {
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

// ----- the scene ------------------------------------------------------------------------------

pub struct Spectral {
    glide: Glide,
    playing: bool,
    /// Play once the glide to a new setting has arrived (and the function asked for is
    /// the one drawn).
    play_when_there: bool,
    /// The function asked for, and the one drawn: a new one is drawn from the moment
    /// the film is back at its shadows, before the stretch, so that it never jumps.
    fun: Fun,
    drawn_fun: Fun,
    /// The step the film glides on to once a new function is drawn (an experiment that
    /// shows a function of A at the end of its film, set up while another was drawn:
    /// the film goes back to the shadows first, then on with the new numbers).
    then_stage: Option<f64>,
    ramp: Option<Running>,
    /// The canvas last drawn on and how much of its top the application's buttons
    /// cover: they decide where the panel goes, and so where the picture is.
    canvas: Vec2,
    inset: f32,
    measured: bool,
    new_view: bool,
}

impl Spectral {
    fn new() -> Self {
        let first = Setting { l: [1.0, 1.0], theta: ANGLE, k: 0.0, x: DVec2::ONE, stage: 0.0 };
        let mut scene = Spectral {
            glide: Glide::at(first),
            playing: false,
            play_when_there: false,
            fun: Fun::Same,
            drawn_fun: Fun::Same,
            then_stage: None,
            ramp: None,
            canvas: Vec2::new(1080.0, 1000.0),
            inset: 0.0,
            measured: false,
            new_view: false,
        };
        scene.set_up(0);
        // (opening, there is nothing to glide from: the film starts at once)
        scene.glide = Glide::at(scene.glide.to);
        scene.drawn_fun = scene.fun;
        scene
    }

    /// Set up experiment `k` of `TRIALS`: glide to its setting, then play it.
    fn set_up(&mut self, k: usize) {
        let trial = &TRIALS[k];
        // (the axes turn the short way round to the angle asked for, at most half a turn)
        let now = self.glide.shown.theta;
        let tau = std::f64::consts::TAU;
        let theta = trial.theta + tau * ((now - trial.theta) / tau).round();
        // (a function other than the one drawn comes in at the shadows: see `then_stage`)
        let rewind = trial.fun != self.drawn_fun && trial.stage > 1.0;
        let stage = if rewind { 1.0 } else { trial.stage };
        self.then_stage = rewind.then_some(trial.stage);
        self.glide.to = Setting { l: trial.l, theta, k: trial.k, x: DVec2::from(trial.x), stage };
        self.fun = trial.fun;
        self.playing = false;
        self.play_when_there = trial.play;
        self.ramp = trial.ramp.map(|ramp| Running { ramp, t: 0.0, from: theta });
    }

    /// Play the film from where it is (from the start, if it has ended).
    fn play(&mut self) {
        if self.glide.to.stage >= 3.0 - 1e-9 {
            self.glide.to.stage = 0.0;
            self.play_when_there = true;
        } else {
            self.playing = true;
        }
    }

    fn pause(&mut self) {
        (self.playing, self.play_when_there) = (false, false);
    }

    /// Ask for another function of A: the film goes back to the shadows and stretches
    /// them by the new numbers.
    fn choose(&mut self, fun: Fun) {
        if fun == self.fun {
            return;
        }
        self.fun = fun;
        self.playing = false;
        self.then_stage = None;
        self.glide.to.stage = self.glide.to.stage.min(1.0);
        self.play_when_there = true;
    }

    /// Where the dot of axis `i` is drawn: at −λᵢqᵢ, an end of the ellipse's half-axis
    /// (where A takes −qᵢ), on the side of 0 away from x's shadows in the experiments,
    /// so that it is not mistaken for a point of the film.
    fn dot_at(s: &Setting, i: usize) -> DVec2 {
        -s.axes()[i] * s.l[i]
    }

    /// A dot on an axis dragged to `at`: the axes turn to point at it, and that axis's
    /// eigenvalue becomes how far from 0 it is (negative past 0, on the side of qᵢ).
    fn drag_axis(&mut self, i: usize, at: DVec2) {
        let h = -at;
        let mut s = self.glide.shown;
        let q = s.axes()[i];
        let len = h.length();
        // (near 0 the direction is not to be trusted: there only the eigenvalue changes)
        if len > 0.05 {
            let mut dir = h / len;
            if dir.dot(q) < 0.0 {
                dir = -dir;
            }
            s.theta += q.perp_dot(dir).atan2(q.dot(dir));
        }
        let q = s.axes()[i];
        let mut l = h.dot(q).clamp(-L_MAX, L_MAX);
        if l.abs() < SNAP {
            l = 0.0;
        }
        if (l - s.l[1 - i]).abs() < SNAP {
            l = s.l[1 - i];
        }
        s.l[i] = l;
        self.glide = Glide::at(s);
        self.ramp = None;
    }

    /// Move everything on: `dt` seconds of the scene's time (the film, a ramp) and
    /// `real_dt` of the clock's (a change of setting glides by the clock, so that it is
    /// seen through even while the scene's time stands still).
    fn advance(&mut self, dt: f32, real_dt: f32) {
        self.glide.step(real_dt as f64);
        if self.drawn_fun != self.fun && self.glide.shown.stage <= 1.0 {
            self.drawn_fun = self.fun;
        }
        if self.drawn_fun == self.fun && self.glide.arrived() {
            if let Some(stage) = self.then_stage.take() {
                self.glide.to.stage = stage;
            }
        }
        if self.play_when_there && self.glide.arrived() && self.drawn_fun == self.fun {
            (self.play_when_there, self.playing) = (false, true);
        }
        let dt = if dt.is_finite() { dt.max(0.0) } else { 0.0 };
        if self.playing {
            let stage = (self.glide.to.stage + dt as f64 / STEP).min(3.0);
            self.glide.now(|s| s.stage = stage);
            if stage >= 3.0 {
                self.playing = false;
            }
        }
        if let Some(mut run) = self.ramp {
            if self.glide.arrived() {
                run.t += dt;
                let v = run.ramp.at(run.t);
                match run.ramp {
                    Ramp::Turn => self.glide.now(|s| s.theta = run.from + v),
                    Ramp::Twist => self.glide.now(|s| s.k = if v.abs() < K_NONE { 0.0 } else { v }),
                }
                self.ramp = (run.t < run.ramp.length()).then_some(run);
            }
        }
    }

    fn shown(&self) -> Setting {
        self.glide.shown
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

    /// The screen panel's play button, its track (a click sets the step) and the knob
    /// on the track (a handle).
    fn panel_input(&mut self, f: &mut Frame, panel: &Panel) {
        let pointer = f.ptr.pos.map(|p| p / f.px);
        let clicked = f.clicked_canvas();
        let (b0, b1) = panel.bar;
        let along = |p: Vec2| ((p.x - b0.x) / (b1.x - b0.x).max(1.0)).clamp(0.0, 1.0) as f64 * 3.0;
        let filmed = matches!(eigen(&self.shown()), Eigen::Two { .. });
        if filmed {
            let knob = b0 + (b1 - b0) * (self.shown().stage / 3.0).clamp(0.0, 1.0) as f32;
            if let Some(mut w) = f.cam.pick_ground(knob * f.px).map(|p| p.truncate()) {
                if f.drag2(4, &mut w, pal::WHITE) {
                    if let Some(p) = on_screen(f, w.extend(0.0)) {
                        let stage = along(p);
                        self.glide.now(|s| s.stage = stage);
                        self.pause();
                    }
                }
            }
        }
        if let Some(p) = pointer.filter(|_| clicked) {
            if inside_rect(p, panel.play) && filmed {
                if self.playing || self.play_when_there {
                    self.pause();
                } else {
                    self.play();
                }
            } else if filmed && inside_rect(p, (b0 - Vec2::new(8.0, 12.0), b1 + Vec2::new(8.0, 12.0))) {
                self.glide.to.stage = along(p);
                self.pause();
            }
        }
    }
}

// ----- where things go ----------------------------------------------------------------

/// The part of the plane the view is fitted to (its centre, and half its size): every
/// experiment's x, its image and the ellipse lie in it. (One for all of them: the view
/// stays put as one experiment gives way to the next.)
const FRAME_CENTRE: Vec2 = Vec2::new(0.2, 0.35);
const FRAME_HALF: Vec2 = Vec2::new(2.25, 2.1);

/// The screen panel's parts, in logical pixels.
#[derive(Clone, Debug)]
struct Panel {
    /// Its dark backing.
    min: Vec2,
    max: Vec2,
    /// The size of the numbers in the matrices.
    text: f32,
    /// Where its heading goes.
    head: Vec2,
    /// The left edge of the column of matrices, and the tops of its three rows.
    column: f32,
    tops: [f32; 3],
    /// Where the column to the right of the matrices starts.
    right: f32,
    /// The two ends of the film's track, at its middle, and the play button.
    bar: (Vec2, Vec2),
    play: (Vec2, Vec2),
}

const PANEL_W: f32 = 360.0;
const HEAD_H: f32 = 24.0;
const BAR_H: f32 = 26.0;
const TICKS_H: f32 = 18.0;
/// The most lines the caption takes on a phone.
const CAPTION_LINES: usize = 3;

/// The widest a number in a matrix on the panel is, in characters, a minus sign counted
/// as two (see `cell`).
const CELL: usize = 6;

/// The names written before the matrices' column, the widest of each kind.
const LEADS: [&str; 3] = [r"$\sqrt{A} =$", "$A^{-1} =$", "$= -2.50$"];

/// The widths of `LEADS` and of "A =" at a size of 1, worked out once.
fn lead_widths() -> [f32; 4] {
    static WIDTHS: std::sync::OnceLock<[f32; 4]> = std::sync::OnceLock::new();
    *WIDTHS.get_or_init(|| [LEADS[0], LEADS[1], LEADS[2], "$A =$"].map(|t| text_width(t, 1.0)))
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
    fn of(view: Vec2, inset: f32) -> Layout {
        let narrow = view.x < 620.0 && view.y >= 560.0;
        let pad = if narrow { 10.0 } else { 12.0 };
        let width = if narrow { view.x } else { PANEL_W };
        // the numbers' size: as large as lets the widest row fit across, up to 13.5
        let w = lead_widths();
        let lead = |s: f32| w[0].max(w[1]).max(w[2]) * s;
        let across = |s: f32| lead(s) + 8.0 + matrix_size(2, 2, CELL, s).x + 14.0 + w[3] * s + 8.0 + matrix_size(2, 2, CELL, s).x;
        let mut text = 13.5f32;
        while across(text) > width - 2.0 * pad && text > 9.5 {
            text -= 0.5;
        }
        let mat_h = matrix_size(2, 2, CELL, text).y;
        let tall = pad + HEAD_H + 3.0 * mat_h + 22.0 + 12.0 + BAR_H + TICKS_H + pad;
        let place = |min: Vec2, width: f32| {
            let x = min.x + pad;
            let head = Vec2::new(x, min.y + pad);
            let column = x + lead(text) + 8.0;
            let top = min.y + pad + HEAD_H;
            let tops = [top, top + mat_h + 10.0, top + 2.0 * mat_h + 22.0];
            let right = column + matrix_size(2, 2, CELL, text).x + 14.0;
            let y = tops[2] + mat_h + 12.0;
            let play = (Vec2::new(x, y), Vec2::new(x + 30.0, y + BAR_H));
            let bar = (Vec2::new(x + 52.0, y + BAR_H * 0.5), Vec2::new(min.x + width - pad - 30.0, y + BAR_H * 0.5));
            Panel { min, max: Vec2::new(min.x + width, y + BAR_H + TICKS_H + pad), text, head, column, tops, right, bar, play }
        };
        let whole = |caption| Layout { narrow, picture: (Vec2::new(0.0, inset), view), panel: None, caption };
        if narrow {
            // the panel below the picture, with the caption at its foot
            let caption_h = CAPTION_LINES as f32 * 19.0 + 8.0;
            let below = (view.y - tall - caption_h).max(view.y * 0.5);
            if below + tall + caption_h > view.y + 0.5 {
                return whole(Some((0.0, view.x, view.y - 10.0)));
            }
            let mut panel = place(Vec2::new(0.0, below), view.x);
            panel.max = view;
            return Layout { narrow, picture: (Vec2::new(0.0, inset), Vec2::new(view.x, below)), panel: Some(panel), caption: Some((0.0, view.x, view.y - 10.0)) };
        }
        let top = inset + 12.0;
        if view.x < 640.0 || top + tall > view.y - 60.0 {
            return whole(Some((0.0, view.x, view.y - 16.0)));
        }
        let panel = place(Vec2::new(view.x - width - 12.0, top), width);
        let picture = (Vec2::new(0.0, inset), Vec2::new(panel.min.x - 8.0, view.y));
        Layout { narrow, picture, panel: Some(panel), caption: Some((0.0, picture.1.x, view.y - 16.0)) }
    }

    /// The part of the canvas the scene's frame is fitted into: the picture's, less the
    /// heading at its top and the caption at its foot.
    fn frame_room(&self) -> (Vec2, Vec2) {
        let (min, max) = self.picture;
        let foot = if self.narrow { 6.0 } else { 46.0 };
        (min + Vec2::new(12.0, 56.0), max - Vec2::new(12.0, foot))
    }
}

fn inside_rect(p: Vec2, (min, max): (Vec2, Vec2)) -> bool {
    p.cmpge(min).all() && p.cmple(max).all()
}

/// A number for a matrix on the screen, as a formula, in five characters at most (so
/// that the brackets never move as the numbers change): two decimals, then one, then
/// none, and a power of ten beyond 9999 (as in `expm`).
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

/// How wide a text is on the screen: one with mathematics in it exactly, words alone as
/// `text_width` judges (and that, before the fonts are in place, for both).
fn width(f: &Frame, text: &str, size: f32) -> f32 {
    if crate::typeset::has_math(text) {
        crate::typeset::measure(f.ctx, text, size).map_or_else(|| text_width(text, size), |s| s.x)
    } else {
        text_width(text, size)
    }
}

/// A matrix of numbers on the panel, as `hud_matrix` draws it, with a number too wide
/// for its cell (a large power of ten) drawn smaller, to fit (as in `expm`).
fn matrix(f: &mut Frame, top_left: Vec2, m: DMat2, colors: [Color; 2], size: f32) -> Vec2 {
    let room = matrix_cell(CELL, size).x - 2.0;
    let mut rows: Vec<Vec<(String, Color)>> = (0..2).map(|i| (0..2).map(|j| (cell(m.col(j)[i]), colors[j])).collect()).collect();
    let mut smaller = Vec::new();
    for (i, row) in rows.iter_mut().enumerate() {
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
    let total = hud_matrix(f, top_left, &rows, CELL, size, pal::DIM);
    for (i, j, text, color, s) in smaller {
        let (lo, hi) = matrix_entry(top_left, CELL, size, i, j);
        f.text_px((lo + hi) * 0.5, text, s, color, Align2::CENTER_CENTER);
    }
    total
}

/// Rings round a matrix's two corner entries: faint when they are equal, red when not.
fn ring_corners(f: &mut Frame, top_left: Vec2, m: DMat2, size: f32) {
    let equal = (m.y_axis.x - m.x_axis.y).abs() <= 1e-9 * size_of(m);
    let color = if equal { pal::TEXT.a(0.35) } else { pal::RED.a(0.9) };
    for (i, j) in [(0, 1), (1, 0)] {
        let (lo, hi) = matrix_entry(top_left, CELL, size, i, j);
        hud_outline(f, lo + Vec2::new(3.0, 2.0), hi - Vec2::new(3.0, 2.0), 1.2, color);
    }
}

fn size_of(m: DMat2) -> f64 {
    size(m).max(1e-300)
}

// ----- drawing what may run far off ------------------------------------------------

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

/// A path's points kept to the box `lo..hi` (the same number of points however far
/// the path runs off).
fn within(points: &[DVec2], lo: DVec2, hi: DVec2) -> Vec<Vec3> {
    points.iter().map(|p| if p.is_finite() { p.clamp(lo, hi) } else { DVec2::ZERO }.as_vec2().extend(0.0)).collect()
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

/// The parts of the picture drawn as lines.
#[derive(Clone, Copy, PartialEq)]
enum Part {
    /// What keeps its number of points from frame to frame (in the usual batch).
    Fixed,
    /// The eigenlines and their marks, which come and go with the lines.
    Eigen,
    /// The film's construction, there only while there are two eigenlines.
    Build,
    /// Every line through 0, while the eigenvalues are equal.
    Fan,
    /// The circle and the letter as they were (always the same points).
    Before,
}

/// The line batch a part is drawn in, solid or dashed. A part whose number of points
/// can change from frame to frame has a batch of its own (different from the usual one
/// by a hair of its width, and nothing else), so that from one frame to the next the
/// parts that stay the same can still be compared point for point, as `eidos motion`
/// does (see `lines_for` in `expm`).
fn lines_for<'a>(f: &'a mut Frame, part: Part, dashed: bool) -> &'a mut crate::gfx::draw::LineBatch {
    if part == Part::Fixed && !dashed {
        return &mut f.dl.lines;
    }
    if part == Part::Fixed {
        return &mut f.dl.dashed;
    }
    let (dash, duty) = if dashed { (9.0, 0.55) } else { (0.0, 0.5) };
    f.dl.lines_with(LineKind::Blend, LineStyle { width_scale: 1.0 + 1e-4 * part as u8 as f32, dash, duty, ..Default::default() })
}

/// An arrow from `a` to `b` in the batch of `part`, the part of it in the box `lo..hi`:
/// without its head if the head is outside.
#[allow(clippy::too_many_arguments)]
fn arrow_in(f: &mut Frame, part: Part, a: DVec2, b: DVec2, lo: DVec2, hi: DVec2, width: f32, c: Color) {
    let Some((p, q)) = clip(a, b, lo, hi) else { return };
    let ends = (p.as_vec2().extend(0.0), q.as_vec2().extend(0.0));
    let d = ends.1 - ends.0;
    let len = d.length();
    if len < 1e-9 || c.0[3] == 0 {
        return;
    }
    let dir = d / len;
    let mut end = ends.1;
    if q == b {
        let wpp = f.cam.world_per_px(ends.1) * f.px;
        let head = ((width * 3.2 + 5.0) * wpp).min(len * 0.6);
        let side = dir.cross(Vec3::Z).normalize_or_zero() * (head * 0.42);
        let base = ends.1 - dir * head;
        f.dl.tris_alpha.tri(ends.1, base + side, base - side, c);
        end = ends.1 - dir * (head * 0.8);
    }
    lines_for(f, part, false).segment(ends.0, end, width, c);
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

/// The letter F, in the circle's lower left: three rectangles (each by its lower left
/// and upper right corners), and its outline, corners in order.
const F_AT: DVec2 = DVec2::new(-0.55, -0.62);
const F_RECTS: [[f64; 4]; 3] = [[0.0, 0.0, 0.12, 0.6], [0.12, 0.48, 0.4, 0.6], [0.12, 0.24, 0.3, 0.35]];
const F_OUTLINE: [[f64; 2]; 10] = [[0.0, 0.0], [0.12, 0.0], [0.12, 0.24], [0.3, 0.24], [0.3, 0.35], [0.12, 0.35], [0.12, 0.48], [0.4, 0.48], [0.4, 0.6], [0.0, 0.6]];

/// Points round the unit circle.
const ROUND: usize = 96;

/// The unit circle and the letter's outline as they were, dashed. The dashes are laid
/// in the plane's own units, a fixed number of them, and not by their length on the
/// screen (as `Frame::dashed_path` lays them): so that they move with the view as one,
/// rather than sliding along the curve as it is zoomed, which `eidos motion` sees as
/// points leaping.
fn before_dashes(f: &mut Frame, outline: &[DVec2]) {
    let batch = lines_for(f, Part::Before, false);
    let colour = shape_colour().a(0.45);
    let dashes = 60;
    for k in 0..dashes {
        let (a, b) = (k as f64 / dashes as f64, (k as f64 + 0.55) / dashes as f64);
        let arc = (0..=3).map(|j| DVec2::from_angle((a + (b - a) * j as f64 / 3.0) * std::f64::consts::TAU).as_vec2().extend(0.0));
        batch.polyline(arc, 1.2, colour);
    }
    for (i, p) in outline.iter().enumerate() {
        let q = outline[(i + 1) % outline.len()];
        // (about 0.07 to a dash and its gap, a whole number of them on each edge)
        let n = ((q - *p).length() / 0.07).round().max(1.0) as usize;
        for k in 0..n {
            let (a, b) = (k as f64 / n as f64, (k as f64 + 0.55) / n as f64);
            batch.segment(p.lerp(q, a).as_vec2().extend(0.0), p.lerp(q, b).as_vec2().extend(0.0), 1.1, colour);
        }
    }
}

fn circle_point(i: usize) -> DVec2 {
    DVec2::from_angle(i as f64 / ROUND as f64 * std::f64::consts::TAU)
}

/// The yellow of the shapes the map carries.
fn shape_colour() -> Color {
    pal::YELLOW
}

/// The colour of the single line two eigenlines become when they meet.
fn met_colour() -> Color {
    pal::ORANGE.mix(pal::PURPLE, 0.5)
}

impl Scene for Spectral {
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
        self.playing || self.play_when_there || !self.glide.arrived() || self.ramp.is_some() || self.drawn_fun != self.fun || self.then_stage.is_some()
    }

    fn ui(&mut self, ui: &mut egui::Ui, _: &Gfx) {
        ui::note(ui, r"A matrix is symmetric when it is its own mirror image across the diagonal, $A^\top = A$: the entry in row 1, column 2 is the one in row 2, column 1. Every symmetric matrix of real numbers is a stretch along two axes at right angles, its eigenvectors $q_1$ and $q_2$, by its eigenvalues $\lambda_1$ and $\lambda_2$ (a negative one also turns its axis round). Written out, it is a sum of two projections:");
        ui::equation(ui, r"A = \lambda_1 q_1 q_1^\top + \lambda_2 q_2 q_2^\top = \lambda_1 P_1 + \lambda_2 P_2");
        ui::note(ui, "So to apply $A$ to $x$: split $x$ into its shadows on the two axes, multiply each by its eigenvalue, and add. The film on the picture does that. Drag $x$ and the orange and purple dots.");

        ui::section(ui, "Things to try");
        for (k, trial) in TRIALS.iter().enumerate() {
            if ui::experiment(ui, trial.label, trial.look_for) {
                self.set_up(k);
            }
        }

        ui::section(ui, "The film");
        ui.horizontal(|ui| {
            let busy = self.playing || self.play_when_there;
            if ui.button(if busy { "❚❚ pause" } else { "▶ play" }).clicked() {
                if busy {
                    self.pause();
                } else {
                    self.play();
                }
            }
            if ui.button("from the start").clicked() {
                self.glide.to.stage = 0.0;
                self.play_when_there = true;
                self.playing = false;
            }
        });
        let mut stage = self.glide.to.stage as f32;
        if ui::slider(ui, "step", &mut stage, 0.0..=3.0) {
            self.glide.now(|s| s.stage = stage as f64);
            self.pause();
        }
        ui::note(ui, r"0: $x$ alone. 1: its shadows $P_1 x$ and $P_2 x$ on the axes. 2: each multiplied by its eigenvalue. 3: the two added, $Ax$.");

        ui::section(ui, "Apply");
        let names: Vec<String> = FUNS.iter().map(|f| format!("${}$", f.name())).collect();
        let names: Vec<&str> = names.iter().map(String::as_str).collect();
        let mut which = FUNS.iter().position(|f| *f == self.fun).unwrap_or(0);
        if ui::choice(ui, &names, &mut which) {
            self.choose(FUNS[which]);
        }
        ui::note(ui, r"A function of $A$ keeps the axes and acts on the eigenvalues: $f(A) = f(\lambda_1) P_1 + f(\lambda_2) P_2$. $A^{-1}$ needs no eigenvalue to be 0, and the square root with $\sqrt{\lambda} \ge 0$ on these axes needs none to be negative.");

        ui::section(ui, "The matrix");
        let to = self.glide.to;
        let mut l1 = to.l[0] as f32;
        if ui::slider(ui, r"$\lambda_1$, along the orange axis", &mut l1, -L_MAX as f32..=L_MAX as f32) {
            self.glide.now(|s| s.l[0] = typed(l1));
        }
        let mut l2 = to.l[1] as f32;
        if ui::slider(ui, r"$\lambda_2$, along the purple axis", &mut l2, -L_MAX as f32..=L_MAX as f32) {
            self.glide.now(|s| s.l[1] = typed(l2));
        }
        let mut degrees = (to.theta.to_degrees().rem_euclid(180.0)) as f32;
        if ui::slider(ui, "angle of the orange axis, in degrees", &mut degrees, 0.0..=180.0) {
            // (the nearest angle that draws the same axes, so that the dots do not jump
            // to the other side of 0)
            let want = (degrees as f64).to_radians();
            let pi = std::f64::consts::PI;
            let theta = want + pi * ((to.theta - want) / pi).round();
            self.glide.now(|s| s.theta = theta);
            self.ramp = None;
        }
        let mut k = to.k as f32;
        if ui::slider(ui, "skew part $k$", &mut k, -K_MAX as f32..=K_MAX as f32) {
            let k = if (k as f64).abs() < K_NONE { 0.0 } else { k as f64 };
            self.glide.now(|s| s.k = k);
            self.ramp = None;
        }
        ui::note(ui, r"The dots on the picture set the same: each is $|\lambda_i|$ from $0$ on its axis, where the ellipse of $A$ ends. Drag one round to turn the axes, in or out to change its eigenvalue, through $0$ to make it negative. The skew part adds $kJ$, where $J$ turns every arrow a quarter turn: the corner entries become $b - k$ and $b + k$, and for $k \ne 0$ the matrix is not symmetric (the dots are then hidden, until $k$ is 0 again).");

        ui::section(ui, "Reading the picture");
        let mut key = vec![
            (shape_colour(), r"yellow: the unit circle and a letter (dashed), and where the map takes them (solid). Under $A$ the circle becomes an ellipse with half-axes $|\lambda_1|$ and $|\lambda_2|$ along the eigenlines"),
            (pal::ORANGE, r"orange and purple, dashed: the eigenlines. The dots on them (drag them) are $|\lambda_1|$ and $|\lambda_2|$ from $0$, where the ellipse of $A$ ends. The arrows along them are the shadows $P_1 x$ and $P_2 x$ (faint) and the shadows times the eigenvalues"),
            (pal::WHITE, r"white: $x$ (drag it) and $Ax$; dashed: how $x$ is split into its shadows, and how the stretched shadows add up"),
        ];
        if !self.shown().symmetric() {
            key.push((pal::DIM, "grey, dashed: the axes the eigenlines lay on before the skew part was added"));
        }
        if self.drawn_fun != Fun::Same {
            key.push((shape_colour().a(0.5), "faint yellow ellipse: where $A$ itself takes the circle"));
        }
        key.push((pal::RED, "red: the corner entries of a matrix that is not symmetric"));
        ui::key(ui, &key);

        ui::section(ui, "Words");
        ui::term(ui, "Symmetric:", r"$A^\top = A$: the matrix is its own mirror image across the diagonal.");
        ui::term(ui, "Spectral theorem:", r"a real symmetric matrix has real eigenvalues, and eigenvectors that can be chosen at right angles (they are at right angles whenever their eigenvalues differ). So $A = Q \Lambda Q^\top$: the eigenvectors are the columns of $Q$, the eigenvalues are on the diagonal of $\Lambda$.");
        ui::term(ui, "Why at right angles:", r"for a symmetric matrix $(Av) \cdot w = v \cdot (Aw)$. If $Av = \lambda_1 v$ and $Aw = \lambda_2 w$, that says $\lambda_1 (v \cdot w) = \lambda_2 (v \cdot w)$, so $v \cdot w = 0$ when $\lambda_1 \ne \lambda_2$.");
        ui::term(ui, r"Projection $P_i = q_i q_i^\top$:", r"for a unit vector $q_i$, it sends $x$ to its shadow $(q_i \cdot x)\, q_i$ on the line of $q_i$. Doing it twice changes nothing: $P_i^2 = P_i$. Here $P_1 + P_2 = I$ and $P_1 P_2 = 0$.");
        ui::term(ui, "Orthogonal matrix $Q$:", r"its columns are unit vectors at right angles, so $Q^\top Q = I$: $Q^\top$ undoes $Q$.");
        ui::term(ui, "Skew-symmetric:", r"$K^\top = -K$, as $kJ$ is: it turns every arrow a quarter turn and multiplies its length by $|k|$.");

        ui::section(ui, "Numbers");
        let s = self.shown();
        let e = eigen(&s);
        let a = s.a();
        ui::readout(ui, "eigenvalues", eigen_text(e));
        ui::readout(
            ui,
            "angle between the eigenlines",
            match e {
                Eigen::Two { v, .. } => format!("{:.1}°", lines_apart(v)),
                Eigen::One { .. } => "0° (one line)".into(),
                Eigen::Complex { .. } => "no eigenlines".into(),
            },
        );
        ui::readout(ui, r"$\det A$, $\tr A$", format!("${}, {}$", num(a.determinant(), 2), num(a.x_axis.x + a.y_axis.y, 2)));
        ui::readout(ui, "$x$", format!("${}$", tuple(&[s.x.x, s.x.y], 2)));
        let ax = a * s.x;
        ui::readout(ui, "$Ax$", format!("${}$", tuple(&[ax.x, ax.y], 2)));
        if let Eigen::Two { w, .. } = e {
            let label = if s.symmetric() { r"shadows: $q_1 \cdot x$, $q_2 \cdot x$" } else { "parts along the eigenlines" };
            ui::readout(ui, label, format!("${}, {}$", num(w[0].dot(s.x), 2), num(w[1].dot(s.x), 2)));
        }
        if self.fun != Fun::Same {
            let value = match applied(&s, &e, self.fun) {
                Ok((m, _)) => {
                    let y = m * s.x;
                    format!("${}$", tuple(&[y.x, y.y], 2))
                }
                Err(_) => "none drawn".into(),
            };
            ui::readout(ui, &format!("${}x$", self.fun.name()), value);
        }
        ui::see_also(
            ui,
            &[
                ("quadform", r"the bowl $x^\top A x$ of a symmetric matrix"),
                ("project", "projections, at right angles and not"),
                ("symskew", "any matrix as a symmetric part plus a skew part"),
                ("svd", r"any matrix: $A = U \Sigma V^\top$, two sets of axes"),
            ],
        );
    }

    fn frame(&mut self, f: &mut Frame) {
        // the panel's place follows the canvas, and the picture's place follows the panel's
        let view = f.view_size();
        let lay = Layout::of(view, f.top_inset);
        let before = Layout::of(self.canvas, self.inset);
        // (a canvas squeezed to a sliver is one the controls have opened over, on a
        // phone: the picture is not seen then, and the film waits for them to close)
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
        self.advance(if covered { 0.0 } else { f.dt }, f.real_dt);

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

/// An eigenvalue from a slider: one within a thousandth of 0 is 0 (so that 1/λ, drawn,
/// stays within a thousand).
fn typed(l: f32) -> f64 {
    let l = (l as f64).clamp(-L_MAX, L_MAX);
    if l.abs() < 1e-3 {
        0.0
    } else {
        l
    }
}

/// The eigenvalues in words and formulas.
fn eigen_text(e: Eigen) -> String {
    match e {
        Eigen::Two { mu, .. } => format!("${}$ and ${}$", num(mu[0], 2), num(mu[1], 2)),
        Eigen::One { mu, .. } => format!("${}$ twice", num(mu, 2)),
        Eigen::Complex { re, im } => format!(r"${} \pm {}i$", num(re, 2), num(im, 2)),
    }
}

impl Spectral {
    /// Everything drawn in the plane, and the names written beside it. Returns the
    /// boxes of the heading and of the caption, which are written over it.
    fn picture(&mut self, f: &mut Frame, lay: &Layout) -> [Option<(Vec2, Vec2)>; 2] {
        let (lo, hi) = f.cam.flat_bounds();
        // what is drawn is cut off a view's width beyond the view, whatever the map does
        let span = (hi - lo).as_dvec2();
        let (box_lo, box_hi) = (lo.as_dvec2() - span, hi.as_dvec2() + span);
        let px = f.view_size().y / (hi.y - lo.y).max(1e-6);
        let wpp = 1.0 / px.max(1e-6);

        // the handles: x, and (while A is symmetric) the dots on the axes
        let under_panel = |f: &Frame, p: Vec2| lay.panel.as_ref().is_some_and(|pn| on_screen(f, p.extend(0.0)).is_some_and(|q| inside_rect(q, (pn.min, pn.max))));
        let mut x = self.shown().x.as_vec2();
        if (!under_panel(f, x) || f.handles.dragging()) && f.drag2(1, &mut x, pal::WHITE) {
            let x = x.as_dvec2().clamp(DVec2::splat(-X_MAX), DVec2::splat(X_MAX));
            self.glide.now(|s| s.x = x);
        }
        if self.shown().symmetric() && self.glide.to.symmetric() {
            for (i, c) in [pal::ORANGE, pal::PURPLE].into_iter().enumerate() {
                let s = self.shown();
                let mut h = Spectral::dot_at(&s, i).as_vec2();
                if (!under_panel(f, h) || f.handles.dragging()) && f.drag2(2 + i as u32, &mut h, c) {
                    self.drag_axis(i, h.as_dvec2());
                }
            }
        }

        let s = self.shown();
        let a = s.a();
        let e = eigen(&s);
        let made = applied(&s, &e, self.drawn_fun);
        // what is drawn: f(A) along the eigenlines if it can be built, A itself if not
        let (f_a, fs) = match (made, e) {
            (Ok(v), _) => v,
            (Err(_), Eigen::Two { mu, .. }) => (a, mu),
            (Err(_), _) => (a, [1.0, 1.0]),
        };
        let shown_fun = if made.is_ok() { self.drawn_fun } else { Fun::Same };
        let m = match e {
            Eigen::Two { .. } => film(f_a, s.stage),
            _ => a,
        };

        // (the batches of the parts, made in the same order every frame)
        for part in [Part::Before, Part::Eigen, Part::Build, Part::Fan] {
            lines_for(f, part, false);
            lines_for(f, part, true);
        }

        let mut room = Room::new(f);
        keep_ticks_clear(f, &mut room);
        if let Some(p) = &lay.panel {
            room.wall(p.min - Vec2::splat(4.0), p.max + Vec2::splat(4.0));
        }
        let heading = self.heading(f, lay, &s, &e, made.err(), shown_fun, fs);
        room.wall(heading.0, heading.1);
        let mut plates = [Some(heading), None];
        if let Some((left, right, bottom)) = lay.caption {
            let parts: &[&str] = if lay.narrow {
                &["dashed: the unit circle, a letter", "yellow: where they go", "drag the dots"]
            } else {
                &["dashed: the unit circle and a letter", "solid yellow: where the map takes them", "drag the white dot $x$, and the orange and purple dots"]
            };
            if let Some(plate) = caption(f, parts, (left, right), bottom) {
                room.wall(plate.0, plate.1);
                plates[1] = Some(plate);
            }
        }

        let reach = span.length() * 2.0;
        let [q1, q2] = s.axes();
        // the axes the eigenlines lay on before the skew part (grey), once they have left them
        if !s.symmetric() {
            for q in [q1, q2] {
                if let Some((p, q)) = clip(-q * reach, q * reach, box_lo, box_hi) {
                    lines_for(f, Part::Eigen, true).segment(p.as_vec2().extend(0.0), q.as_vec2().extend(0.0), 1.2, pal::DIM.a(0.6));
                }
            }
        }
        // the eigenlines
        let lines: Vec<(DVec2, Color)> = match e {
            Eigen::Two { v, .. } => vec![(v[0], pal::ORANGE), (v[1], pal::PURPLE)],
            Eigen::One { v, .. } => vec![(v, met_colour())],
            Eigen::Complex { .. } => Vec::new(),
        };
        for &(v, c) in &lines {
            if let Some((p, q)) = clip(-v * reach, v * reach, box_lo, box_hi) {
                lines_for(f, Part::Eigen, true).segment(p.as_vec2().extend(0.0), q.as_vec2().extend(0.0), 1.6, c.a(0.85));
                room.line(f, p.as_vec2().extend(0.0), q.as_vec2().extend(0.0));
            }
        }
        // equal eigenvalues: every line through 0 is an eigenline
        let every = s.symmetric() && s.l[0] == s.l[1];
        if every {
            // (out to a little beyond the circle's image: enough to be read as lines
            // through 0, not so far as to cross the whole picture)
            let out = s.l[0].abs().max(1.0) + 0.6;
            for j in [1, 2, 4, 5] {
                let d = DVec2::from_angle(s.theta + j as f64 * std::f64::consts::PI / 6.0);
                if let Some((p, q)) = clip(-d * out, d * out, box_lo, box_hi) {
                    lines_for(f, Part::Fan, true).segment(p.as_vec2().extend(0.0), q.as_vec2().extend(0.0), 1.1, pal::ORANGE.mix(pal::PURPLE, 0.5).a(0.4));
                }
            }
        }

        // the unit circle and the letter, as they were (dashed) and as the map has taken them
        let circle: Vec<DVec2> = (0..=ROUND).map(circle_point).collect();
        let outline: Vec<DVec2> = F_OUTLINE.iter().map(|p| F_AT + DVec2::from(*p)).collect();
        before_dashes(f, &outline);
        if self.drawn_fun != Fun::Same && made.is_ok() {
            // A's own ellipse, faint, to compare with
            let ghost: Vec<DVec2> = circle.iter().map(|p| a * *p).collect();
            lines_for(f, Part::Fixed, false).polyline(within(&ghost, box_lo, box_hi), 1.2, shape_colour().a(0.35));
        } else {
            // (the same number of points in the usual batch either way)
            let ghost: Vec<DVec2> = circle.iter().map(|p| m * *p).collect();
            lines_for(f, Part::Fixed, false).polyline(within(&ghost, box_lo, box_hi), 1.2, shape_colour().a(0.0));
        }
        let ellipse: Vec<DVec2> = circle.iter().map(|p| m * *p).collect();
        lines_for(f, Part::Fixed, false).polyline(within(&ellipse, box_lo, box_hi), 2.2, shape_colour().a(0.95));
        for r in F_RECTS {
            let corners = [DVec2::new(r[0], r[1]), DVec2::new(r[2], r[1]), DVec2::new(r[2], r[3]), DVec2::new(r[0], r[3])].map(|p| m * (F_AT + p));
            let cut = clip_polygon(&corners, box_lo, box_hi);
            for i in 1..cut.len().saturating_sub(1) {
                f.dl.tris_alpha.tri(cut[0].as_vec2().extend(0.0), cut[i].as_vec2().extend(0.0), cut[i + 1].as_vec2().extend(0.0), shape_colour().a(0.3));
            }
        }
        let carried: Vec<DVec2> = outline.iter().chain(outline.first()).map(|p| m * *p).collect();
        lines_for(f, Part::Fixed, false).polyline(within(&carried, box_lo, box_hi), 1.8, shape_colour().a(0.95));
        for pair in ellipse.windows(4).step_by(3) {
            room.line(f, pair[0].as_vec2().extend(0.0), pair[3].as_vec2().extend(0.0));
        }
        for pair in carried.windows(2) {
            room.line(f, pair[0].as_vec2().extend(0.0), pair[1].as_vec2().extend(0.0));
        }

        // the angle between the eigenlines at 0: a right angle, or an arc and its size
        let corner_px = 13.0;
        let mut angle_label = None;
        match e {
            Eigen::Two { v, .. } if s.symmetric() => {
                let size = corner_px * wpp;
                let (a1, a2) = (v[0].as_vec2() * size, v[1].as_vec2() * size);
                lines_for(f, Part::Eigen, false).polyline([a1, a1 + a2, a2].map(|p| p.extend(0.0)), 1.4, pal::WHITE.a(0.8));
            }
            Eigen::Two { v, .. } => {
                // between v₁ and whichever of ±v₂ makes the smaller angle
                let other = if v[0].dot(v[1]) < 0.0 { -v[1] } else { v[1] };
                let (from, turn) = (v[0].to_angle(), v[0].perp_dot(other).atan2(v[0].dot(other)));
                let r = 34.0 * wpp as f64;
                let arc: Vec<Vec3> = (0..=16).map(|j| (DVec2::from_angle(from + turn * j as f64 / 16.0) * r).as_vec2().extend(0.0)).collect();
                lines_for(f, Part::Eigen, false).polyline(arc, 1.4, pal::WHITE.a(0.8));
                let mid = DVec2::from_angle(from + 0.5 * turn) * r;
                angle_label = Some((mid, format!("{:.1}°", lines_apart(v))));
            }
            _ => {}
        }

        // the film's construction
        let x = s.x;
        let mut built = None;
        if let Eigen::Two { v, w, .. } = e {
            let b = build(v, w, fs, x, s.stage);
            built = Some(b);
            let (s1, s2, s3) = (ease(s.stage), ease(s.stage - 1.0), ease(s.stage - 2.0));
            let colours = [pal::ORANGE, pal::PURPLE];
            // the drop lines, from x to each foot (along the other line: at right angles
            // to this one when A is symmetric)
            for i in 0..2 {
                let to = x + (b.feet[i] - x) * s1;
                if let Some((p, q)) = clip(x, to, box_lo, box_hi) {
                    lines_for(f, Part::Build, true).segment(p.as_vec2().extend(0.0), q.as_vec2().extend(0.0), 1.3, pal::WHITE.a(0.55));
                    room.line(f, p.as_vec2().extend(0.0), q.as_vec2().extend(0.0));
                }
            }
            if s.symmetric() {
                let alpha = ease((s.stage - 0.75) / 0.25) as f32;
                let size = 9.0 * wpp;
                for i in 0..2 {
                    let foot = b.feet[i];
                    let (along, up) = (if foot.length() > 1e-9 { -foot } else { v[i] }, x - foot);
                    if up.length() * px as f64 > 2.5 * 9.0 && foot.length() * px as f64 > 2.5 * 9.0 {
                        let (d1, d2) = (along.normalize().as_vec2() * size, up.normalize().as_vec2() * size);
                        let at = foot.as_vec2();
                        lines_for(f, Part::Build, false).polyline([at + d1, at + d1 + d2, at + d2].map(|p| p.extend(0.0)), 1.2, pal::WHITE.a(0.8 * alpha));
                    }
                }
            }
            // the shadows (faint once the stretch is under way), and the stretched shadows
            let shadow_alpha = ease((s.stage - 0.55) / 0.45) * (1.0 - 0.65 * s2);
            for i in 0..2 {
                arrow_in(f, Part::Build, DVec2::ZERO, b.feet[i], box_lo, box_hi, 2.2, colours[i].a(shadow_alpha as f32));
            }
            if s.stage > 1.0 {
                for i in 0..2 {
                    arrow_in(f, Part::Build, DVec2::ZERO, b.scaled[i], box_lo, box_hi, 3.0, colours[i]);
                    room.line(f, Vec3::ZERO, b.scaled[i].as_vec2().extend(0.0));
                }
            }
            for i in 0..2 {
                room.line(f, Vec3::ZERO, b.feet[i].as_vec2().extend(0.0));
            }
            // the sum: each stretched shadow carried to the other's tip
            if s3 > 0.0 {
                for i in 0..2 {
                    let from = b.scaled[i];
                    let to = from + b.scaled[1 - i] * s3;
                    if let Some((p, q)) = clip(from, to, box_lo, box_hi) {
                        lines_for(f, Part::Build, true).segment(p.as_vec2().extend(0.0), q.as_vec2().extend(0.0), 1.3, colours[1 - i].a(0.8));
                        room.line(f, p.as_vec2().extend(0.0), q.as_vec2().extend(0.0));
                    }
                }
                arrow_in(f, Part::Build, DVec2::ZERO, b.corner, box_lo, box_hi, 2.8, pal::WHITE.a(s3 as f32));
                room.line(f, Vec3::ZERO, b.corner.as_vec2().extend(0.0));
            }
        } else {
            // no eigenlines to build along: A x, straight
            arrow_in(f, Part::Build, DVec2::ZERO, a * x, box_lo, box_hi, 2.8, pal::WHITE);
            room.line(f, Vec3::ZERO, (a * x).as_vec2().extend(0.0));
        }
        // x itself, as an arrow from 0 (its dot is the handle)
        arrow_in(f, Part::Fixed, DVec2::ZERO, x, box_lo, box_hi, 1.6, pal::WHITE.a(0.55));
        room.line(f, Vec3::ZERO, x.as_vec2().extend(0.0));

        // ----- the names -----
        let away = |p: DVec2| Vec2::new(p.x as f32, -(p.y as f32));
        let seen = |f: &Frame, p: DVec2| p.is_finite() && on_screen(f, p.as_vec2().extend(0.0)).is_some_and(|q| inside_rect(q, (Vec2::ZERO, f.view_size())));
        let long_enough = |p: DVec2| p.length() * px as f64 > 26.0;
        // the eigenlines: beside their dots while A is symmetric (the dots are where the
        // eigenvalues are set), near where each leaves the picture when it is not
        let (fmin, fmax) = lay.frame_room();
        let names: Vec<String> = match e {
            Eigen::Two { mu, .. } => (0..2).map(|i| format!(r"$\lambda_{} = {}$", i + 1, num(mu[i], 2))).collect(),
            Eigen::One { mu, .. } => vec![format!(r"$\lambda = {}$, twice", num(mu, 2))],
            Eigen::Complex { .. } => Vec::new(),
        };
        for (i, (&(v, c), text)) in lines.iter().zip(&names).enumerate() {
            if s.symmetric() {
                let dot = Spectral::dot_at(&s, i);
                let out = if dot.length() * px as f64 > 4.0 { dot } else { -v };
                if seen(f, dot) {
                    room.label(f, Tag("line", i as u32), dot.as_vec2().extend(0.0), text, 13.0, c, away(out), 11.0, false);
                }
                continue;
            }
            let Some(o) = on_screen(f, Vec3::ZERO) else { continue };
            let Some(tip) = on_screen(f, (-v).as_vec2().extend(0.0)) else { continue };
            let d = (tip - o).normalize_or_zero().as_dvec2();
            let Some((_, end)) = clip(o.as_dvec2() - d * 1e5, o.as_dvec2() + d * 1e5, fmin.as_dvec2(), fmax.as_dvec2()) else { continue };
            let at = (end - d * 30.0).as_vec2();
            let up = if d.perp().y < 0.0 { d.perp() } else { -d.perp() };
            if let Some(w) = f.cam.pick_ground(at * f.px) {
                room.label(f, Tag("line", i as u32), w, text, 13.0, c, up.as_vec2(), 8.0, false);
            }
        }
        if let Some((at, text)) = angle_label {
            if seen(f, at) {
                room.label(f, "angle", at.as_vec2().extend(0.0), &text, 12.0, pal::TEXT, away(at), 4.0, true);
            }
        }
        if seen(f, x) {
            room.label(f, "x", x.as_vec2().extend(0.0), "$x$", 16.0, pal::TEXT, away(x), 10.0, false);
        }
        let fun = shown_fun.name();
        if let Some(b) = built {
            let colours = [pal::ORANGE, pal::PURPLE];
            let s2 = ease(s.stage - 1.0);
            for i in 0..2 {
                let other = b.feet[1 - i];
                if s.stage > 0.75 && s2 < 0.5 && seen(f, b.feet[i]) && long_enough(b.feet[i]) {
                    room.label(f, Tag("shadow", i as u32), b.feet[i].as_vec2().extend(0.0), &format!("$P_{} x$", i + 1), 14.0, colours[i], away(-other), 8.0, false);
                }
                if s2 >= 0.5 && seen(f, b.scaled[i]) && long_enough(b.scaled[i]) {
                    let text = format!("${} P_{} x$", shown_fun.on(i + 1), i + 1);
                    room.label(f, Tag("stretched", i as u32), b.scaled[i].as_vec2().extend(0.0), &text, 14.0, colours[i], away(-b.scaled[1 - i]), 11.0, false);
                }
            }
            if s.stage > 2.5 && seen(f, b.corner) {
                room.label(f, "image", b.corner.as_vec2().extend(0.0), &format!("${fun}x$"), 16.0, pal::WHITE, away(b.corner), 10.0, false);
            }
        } else if seen(f, a * x) {
            room.label(f, "image", (a * x).as_vec2().extend(0.0), "$Ax$", 16.0, pal::WHITE, away(a * x), 10.0, false);
        }
        plates
    }

    /// The heading at the top of the picture: what kind of matrix A is, and in a line
    /// under it the step of the film, or what has become of the eigenlines. Returns the
    /// box it takes.
    #[allow(clippy::too_many_arguments)]
    fn heading(&self, f: &mut Frame, lay: &Layout, s: &Setting, e: &Eigen, why: Option<Why>, fun: Fun, fs: [f64; 2]) -> (Vec2, Vec2) {
        let at = lay.picture.0 + Vec2::new(16.0, 14.0);
        let (first, mut second, mut warn) = heading_text(s, e, fun, fs);
        if let Some(why) = why.filter(|_| self.drawn_fun != Fun::Same) {
            second = why_text(why, self.drawn_fun, e);
            warn = true;
        }
        let room = (lay.picture.1.x - lay.picture.0.x - 32.0).max(40.0);
        let size = if width(f, &first, 15.0) <= room { 15.0 } else { 13.0 };
        let small = if width(f, &second, 12.5) <= room { 12.5 } else { 11.0 };
        // (on a narrow canvas a line still too wide is broken between words)
        let (firsts, seconds) = (lines_within(f, &first, size, room), lines_within(f, &second, small, room));
        let mut wide = 0.0f32;
        let mut y = at.y;
        for (lines, size, colour, gap) in [(firsts, size, pal::TEXT, 8.0), (seconds, small, if warn { pal::RED } else { pal::DIM }, 4.0)] {
            for line in lines {
                wide = wide.max(width(f, &line, size));
                f.caption_px(Vec2::new(at.x, y), line, size, colour, Align2::LEFT_TOP);
                y += size + 4.0;
            }
            y += gap - 4.0;
        }
        (at - Vec2::splat(6.0), Vec2::new(at.x + wide.min(room) + 12.0, y + 6.0))
    }

    /// The screen panel: A (and the function of it chosen) written out, its sum of
    /// projections with their numbers, and the film's track.
    fn panel(&self, f: &mut Frame, p: &Panel) {
        hud_panel(f, p.min, p.max);
        let s = self.shown();
        let a = s.a();
        let e = eigen(&s);
        let size = p.text;
        let small = 12.0;
        let made = applied(&s, &e, self.drawn_fun);
        let fun = if made.is_ok() { self.drawn_fun } else { Fun::Same };
        let head = if s.symmetric() { r"$A$ is symmetric: $A^\top = A$" } else { r"$A$ is not symmetric: $A^\top \ne A$" };
        f.text_px(p.head, head, 13.0, pal::TEXT, Align2::LEFT_TOP);

        let mat_h = matrix_size(2, 2, CELL, size).y;
        let mid = |row: usize| p.tops[row] + mat_h * 0.5;
        let lead = |f: &mut Frame, row: usize, text: &str, color: Color| f.text_px(Vec2::new(p.column - 8.0, mid(row)), text, size, color, Align2::RIGHT_CENTER);
        // the first row: the matrix being taken apart, and (for a function of A) A beside it
        let (first, cols) = match made {
            Ok((m, _)) if fun != Fun::Same => (m, [pal::TEXT, pal::TEXT]),
            _ => (a, [pal::PINK, pal::GREEN]),
        };
        lead(f, 0, &format!("${} =$", fun.name()), pal::TEXT);
        matrix(f, Vec2::new(p.column, p.tops[0]), first, cols, size);
        ring_corners(f, Vec2::new(p.column, p.tops[0]), first, size);
        if fun != Fun::Same {
            let w = width(f, "$A =$", size);
            f.text_px(Vec2::new(p.right, mid(0)), "$A =$", size, pal::TEXT, Align2::LEFT_CENTER);
            let at = Vec2::new(p.right + w + 8.0, p.tops[0]);
            matrix(f, at, a, [pal::PINK, pal::GREEN], size);
            ring_corners(f, at, a, size);
        } else if let Eigen::Two { mu, .. } = e {
            for i in 0..2 {
                let c = [pal::ORANGE, pal::PURPLE][i];
                f.text_px(Vec2::new(p.right, mid(0) + (i as f32 - 0.5) * (size + 6.0)), format!(r"$\lambda_{} = {}$", i + 1, num(mu[i], 2)), small, c, Align2::LEFT_CENTER);
            }
        }
        // the sum of projections, a term to a row
        match (e, made) {
            (Eigen::Two { v, w, .. }, _) => {
                let fs = match made {
                    Ok((_, fs)) => fs,
                    Err(_) => match e {
                        Eigen::Two { mu, .. } => mu,
                        _ => [0.0; 2],
                    },
                };
                for i in 0..2 {
                    let c = [pal::ORANGE, pal::PURPLE][i];
                    // (a negative second term is written as one taken away)
                    let sign = match (i, fs[i] < 0.0) {
                        (0, _) => "=",
                        (_, false) => r"+\;",
                        (_, true) => r"-\;",
                    };
                    let coefficient = cell(if i == 1 { fs[i].abs() } else { fs[i] });
                    lead(f, i + 1, &format!("${sign} {}$", tint(c, coefficient.trim_matches('$'))), pal::TEXT);
                    matrix(f, Vec2::new(p.column, p.tops[i + 1]), outer(v[i], w[i]), [c, c], size);
                    let note = if s.symmetric() { format!(r"$P_{} = q_{} q_{}^\top$", i + 1, i + 1, i + 1) } else { format!("$P_{}$: cast along {}", i + 1, ["purple", "orange"][i]) };
                    f.text_px(Vec2::new(p.right, mid(i + 1)), note, small, pal::DIM, Align2::LEFT_CENTER);
                }
            }
            (Eigen::One { .. }, _) => {
                f.text_px(Vec2::new(p.column, mid(1) - 9.0), "the eigenlines have met:", small, pal::RED, Align2::LEFT_CENTER);
                f.text_px(Vec2::new(p.column, mid(1) + 9.0), "no sum of two projections", small, pal::RED, Align2::LEFT_CENTER);
            }
            (Eigen::Complex { .. }, _) => {
                f.text_px(Vec2::new(p.column, mid(1) - 9.0), "no real eigenvalue, no eigenline:", small, pal::RED, Align2::LEFT_CENTER);
                f.text_px(Vec2::new(p.column, mid(1) + 9.0), "no sum of projections", small, pal::RED, Align2::LEFT_CENTER);
            }
        }

        // the film's track: its knob is a handle (drawn with the panel's input)
        let filmed = matches!(e, Eigen::Two { .. });
        let (b0, b1) = p.bar;
        let ink = if filmed { pal::DIM } else { pal::FAINT };
        hud_line(f, b0, b1, 2.0, ink);
        for (k, name) in ["$x$", "split", "stretch", "add"].into_iter().enumerate() {
            let x = b0.x + (b1.x - b0.x) * k as f32 / 3.0;
            hud_line(f, Vec2::new(x, b0.y - 4.0), Vec2::new(x, b0.y + 4.0), 1.2, ink);
            f.text_px(Vec2::new(x, b0.y + 8.0), name, 11.0, ink, Align2::CENTER_TOP);
        }
        if filmed {
            let knob = b0 + (b1 - b0) * (s.stage / 3.0).clamp(0.0, 1.0) as f32;
            hud_line(f, b0, knob, 3.0, pal::WHITE.a(0.8));
        }
        let (lo, hi) = p.play;
        hud_rect(f, lo, hi, Color::hex(0x1A2236));
        hud_outline(f, lo, hi, 1.0, ink);
        let busy = self.playing || self.play_when_there;
        f.text_px((lo + hi) * 0.5, if busy { "❚❚" } else { "▶" }, 14.0, if filmed { pal::TEXT } else { pal::FAINT }, Align2::CENTER_CENTER);
    }
}

/// The heading's two lines, and whether the second is a warning.
fn heading_text(s: &Setting, e: &Eigen, fun: Fun, fs: [f64; 2]) -> (String, String, bool) {
    let n = |v: f64| num(v, 2);
    match *e {
        Eigen::Two { mu, v, .. } => {
            let first = if !s.symmetric() {
                format!(r"not symmetric: $\lambda_1 = {}$ and $\lambda_2 = {}$, eigenlines {:.1}° apart", n(mu[0]), n(mu[1]), lines_apart(v))
            } else if mu[0] == mu[1] {
                format!(r"symmetric, $\lambda_1 = \lambda_2 = {}$: $A = {}\,I$", n(mu[0]), n(mu[0]))
            } else {
                format!(r"symmetric: $\lambda_1 = {}$ and $\lambda_2 = {}$, on axes at right angles", n(mu[0]), n(mu[1]))
            };
            let along = if s.symmetric() { "the two axes" } else { "the two eigenlines" };
            let f_on = |i: usize| fun.on(i);
            let second = if s.stage < 1.0 {
                format!("1 / 3   split $x$ into its shadows on {along}")
            } else if s.stage < 2.0 {
                // (with the factors reached so far, which go evenly from 1 to the numbers)
                let so_far = |i: usize| num(1.0 + (fs[i] - 1.0) * ease(s.stage - 1.0), 2);
                let by = if fun == Fun::Same { "its eigenvalue".to_string() } else { format!("${}$ and ${}$", f_on(1), f_on(2)) };
                format!(r"2 / 3   multiply each shadow by {by}: so far $\times {}$ and $\times {}$", so_far(0), so_far(1))
            } else if s.stage < 3.0 {
                format!("3 / 3   add them: ${}x$", fun.name())
            } else if s.symmetric() && mu[0] == mu[1] {
                "every direction is an eigenvector: any two axes at right angles will do".to_string()
            } else if fun != Fun::Same {
                format!("${}x = {} P_1 x + {} P_2 x$", fun.name(), f_on(1), f_on(2))
            } else if !s.symmetric() {
                r"$Ax = \lambda_1 P_1 x + \lambda_2 P_2 x$, with the shadows cast along the other line".to_string()
            } else if let Some(i) = (0..2).find(|&i| mu[i] == 0.0) {
                if mu[1 - i] == 0.0 {
                    "$A = 0$: everything is sent to $0$".to_string()
                } else {
                    format!(r"$\lambda_{} = 0$: the plane is squashed onto the {} axis, and $A$ has no inverse", i + 1, ["purple", "orange"][i])
                }
            } else if mu[0] < 0.0 && mu[1] < 0.0 {
                "both eigenvalues negative: flipped along both axes, which is a half turn and a stretch".to_string()
            } else if let Some(i) = (0..2).find(|&i| mu[i] < 0.0) {
                format!(r"$\lambda_{} < 0$: a flip along the {} axis, and $\det A < 0$", i + 1, ["orange", "purple"][i])
            } else {
                r"$Ax = \lambda_1 P_1 x + \lambda_2 P_2 x$, for every $x$".to_string()
            };
            (first, second, false)
        }
        Eigen::One { mu, .. } => (
            format!(r"not symmetric: $\lambda = {}$ twice, and a single eigenline", n(mu)),
            "the two eigenlines have met: there is one, and a vector off it cannot be split along eigenlines".to_string(),
            true,
        ),
        Eigen::Complex { re, im } => (
            format!(r"not symmetric: $\lambda = {} \pm {}i$", n(re), n(im)),
            "no real eigenvalue: no line through $0$ is kept, and there is nothing to split $x$ along".to_string(),
            true,
        ),
    }
}

/// Why the function asked for is not drawn, in a line.
fn why_text(why: Why, fun: Fun, e: &Eigen) -> String {
    let name = fun.name();
    match why {
        Why::NoLines => format!("${name}$ is not drawn: there are not two eigenlines to build it along"),
        Why::Zero(i) => format!(r"no ${name}$: $\lambda_{} = 0$ has no reciprocal (the picture shows $A$)", i + 1),
        Why::Negative(i) => {
            let value = match e {
                Eigen::Two { mu, .. } => num(mu[i], 2),
                _ => String::new(),
            };
            format!(r"no square root on these axes: $\lambda_{} = {value}$ has none (the picture shows $A$)", i + 1)
        }
    }
}

/// A text broken between words (never inside a formula) into lines no wider than
/// `room` pixels at `size`, as `width` measures them.
fn lines_within(f: &Frame, text: &str, size: f32, room: f32) -> Vec<String> {
    if width(f, text, size) <= room {
        return vec![text.to_string()];
    }
    // the words, each with its formulas whole
    let mut words: Vec<String> = Vec::new();
    let (mut word, mut inside) = (String::new(), false);
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
    let mut lines: Vec<String> = Vec::new();
    for word in words {
        match lines.last_mut() {
            Some(line) if width(f, &format!("{line} {word}"), size) <= room => {
                line.push(' ');
                line.push_str(&word);
            }
            _ => lines.push(word),
        }
    }
    lines
}

/// A caption along the bottom of the picture, on a dark plate: `parts` side by side as
/// long as they fit across `(left, right)`, on further lines above when they do not, the
/// last ending at `bottom`. Returns the box it takes (none on a canvas too small).
fn caption(f: &mut Frame, parts: &[&str], (left, right): (f32, f32), bottom: f32) -> Option<(Vec2, Vec2)> {
    let size = 12.0;
    if f.view_size().y < 300.0 {
        return None;
    }
    let room = right - left - 36.0;
    let gap = "      ";
    let mut lines: Vec<String> = Vec::new();
    for part in parts {
        match lines.last_mut() {
            Some(last) if width(f, last, size) + width(f, gap, size) + width(f, part, size) <= room => {
                last.push_str(gap);
                last.push_str(part);
            }
            _ => lines.push(part.to_string()),
        }
    }
    let widest = lines.iter().map(|l| width(f, l, size)).fold(0.0f32, f32::max);
    let count = lines.len();
    for (k, line) in lines.into_iter().enumerate() {
        let up = (count - 1 - k) as f32 * (size + 7.0);
        f.caption_px(Vec2::new(left + 18.0, bottom - up), line, size, pal::DIM, Align2::LEFT_BOTTOM);
    }
    Some((Vec2::new(left + 10.0, bottom - count as f32 * (size + 7.0) - 4.0), Vec2::new(left + 26.0 + widest, bottom + 6.0)))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A matrix by its rows, [[a, b], [c, d]], the way matrices are written.
    fn rows([a, b, c, d]: [f64; 4]) -> DMat2 {
        DMat2::from_cols(DVec2::new(a, c), DVec2::new(b, d))
    }

    fn close(m: DMat2, n: DMat2, tol: f64) -> bool {
        (m - n).to_cols_array().iter().all(|v| v.abs() <= tol)
    }

    fn setting(l: [f64; 2], theta: f64, k: f64) -> Setting {
        Setting { l, theta, k, x: DVec2::new(0.4, 1.55), stage: 3.0 }
    }

    /// Settings of every kind: random ones, and the edges (equal eigenvalues, a zero, a
    /// negative one, the meeting point, just either side of it, A = 0).
    fn settings() -> Vec<Setting> {
        let mut rng = Rng::new(7);
        let mut out: Vec<Setting> = (0..400)
            .map(|i| {
                let k = if i % 3 == 0 { 0.0 } else { rng.range(-1.5, 1.5) as f64 };
                setting([rng.range(-2.5, 2.5) as f64, rng.range(-2.5, 2.5) as f64], rng.range(-7.0, 7.0) as f64, k)
            })
            .collect();
        for theta in [0.0, ANGLE, 2.0, -3.5] {
            for l in [[2.0, 0.5], [1.5, 1.5], [2.0, 0.0], [0.0, 0.0], [2.0, -0.5], [-1.0, -2.0], [0.5, 2.0]] {
                for k in [0.0, 1e-4, 0.3, 0.75, 0.75 - 1e-9, 0.75 + 1e-9, -0.75, 1.0] {
                    out.push(setting(l, theta, k));
                }
            }
        }
        out
    }

    /// The closed form is right: every eigenline drawn is kept by A with its
    /// eigenvalue, the w's read off how much of each line a vector is made of, the
    /// lines meet exactly where |k| = |λ₁ − λ₂|/2, and their angle is arccos(2|k|/|λ₁ − λ₂|).
    #[test]
    fn the_eigenlines_are_right_everywhere() {
        for s in settings() {
            let a = s.a();
            let (tr, det) = (a.x_axis.x + a.y_axis.y, a.determinant());
            let d = 0.5 * (s.l[0] - s.l[1]);
            let scale = size(a).max(1.0);
            match eigen(&s) {
                Eigen::Two { mu, v, w } => {
                    assert!(s.k.abs() < d.abs() + 1e-6, "{s:?}");
                    assert!((mu[0] + mu[1] - tr).abs() < 1e-12 * scale && (mu[0] * mu[1] - det).abs() < 1e-11 * scale * scale, "{s:?}");
                    for i in 0..2 {
                        assert!((v[i].length() - 1.0).abs() < 1e-12 && (a * v[i] - v[i] * mu[i]).length() < 1e-9 * scale, "{s:?}: line {i}");
                        for j in 0..2 {
                            let want = if i == j { 1.0 } else { 0.0 };
                            assert!((w[i].dot(v[j]) - want).abs() < 1e-6 * (1.0 + w[i].length()), "{s:?}");
                        }
                    }
                    if s.symmetric() {
                        // at right angles, and the projections are symmetric
                        assert!(v[0].dot(v[1]).abs() < 1e-15 && w == v, "{s:?}");
                    } else if d.abs() > 1e-9 {
                        let want = (s.k.abs() / d.abs()).min(1.0).acos().to_degrees();
                        assert!((lines_apart(v) - want).abs() < 1e-6, "{s:?}: {} against {want}", lines_apart(v));
                        // as k goes to 0 the lines go back to S's axes, with λ₁ and λ₂
                        let near = Setting { k: s.k * 1e-3, ..s };
                        if let (Eigen::Two { mu: m2, v: v2, .. }, true) = (eigen(&near), near.k.abs() < 1e-3 * d.abs()) {
                            let [q1, q2] = s.axes();
                            assert!((m2[0] - s.l[0]).abs() < 0.01 * d.abs().max(1.0) && v2[0].dot(q1).abs() > 0.999 && v2[1].dot(q2).abs() > 0.999, "{s:?}");
                        }
                    }
                }
                Eigen::One { mu, v } => {
                    assert!((s.k.abs() - d.abs()).abs() < 1e-6, "{s:?}");
                    assert!((2.0 * mu - tr).abs() < 1e-12 * scale && (a * v - v * mu).length() < 1e-5 * scale, "{s:?}");
                    // (A − μI)² = 0: a double eigenvalue
                    let n = a - DMat2::IDENTITY * mu;
                    assert!(size(n * n) < 1e-9 * scale * scale, "{s:?}");
                }
                Eigen::Complex { re, im } => {
                    assert!(s.k.abs() > d.abs() - 1e-6 && im > 0.0, "{s:?}");
                    assert!((2.0 * re - tr).abs() < 1e-12 * scale && (re * re + im * im - det).abs() < 1e-11 * scale * scale, "{s:?}");
                }
            }
        }
    }

    /// The spectral theorem, as the scene draws it: for a symmetric A, A = λ₁P₁ + λ₂P₂
    /// with Pᵢ = qᵢqᵢᵀ symmetric, Pᵢ² = Pᵢ, P₁P₂ = 0, P₁ + P₂ = I; and in general (two
    /// eigenlines) A = Σ μᵢ vᵢwᵢᵀ, with the vᵢwᵢᵀ not symmetric once k ≠ 0.
    #[test]
    fn a_is_the_sum_of_its_projections() {
        for s in settings() {
            let a = s.a();
            if let Eigen::Two { mu, v, w } = eigen(&s) {
                let p = [outer(v[0], w[0]), outer(v[1], w[1])];
                let scale = size(p[0]).max(size(p[1])).max(1.0);
                assert!(close(p[0] * mu[0] + p[1] * mu[1], a, 1e-9 * scale * size(a).max(1.0)), "{s:?}");
                assert!(close(p[0] + p[1], DMat2::IDENTITY, 1e-9 * scale) && close(p[0] * p[0], p[0], 1e-9 * scale * scale), "{s:?}");
                if s.symmetric() {
                    assert!(close(a, a.transpose(), 1e-15) && close(p[0], p[0].transpose(), 0.0) && close(p[0] * p[1], DMat2::ZERO, 1e-15), "{s:?}");
                    let q = DMat2::from_cols(v[0], v[1]);
                    assert!(close(q.transpose() * q, DMat2::IDENTITY, 1e-15) && close(q * DMat2::from_diagonal(DVec2::from(mu)) * q.transpose(), a, 1e-14), "{s:?}");
                } else {
                    assert!(!close(a, a.transpose(), 1e-6) && !close(p[0], p[0].transpose(), 1e-9), "{s:?}");
                }
            }
        }
    }

    /// f(A), built without the projections, is f(μ₁)P₁ + f(μ₂)P₂, and is what its name
    /// says: A A, the inverse, a square root; it is offered exactly when every f(μ) is real.
    #[test]
    fn a_function_of_a_acts_on_the_eigenvalues() {
        for s in settings() {
            let (a, e) = (s.a(), eigen(&s));
            for fun in FUNS {
                match (applied(&s, &e, fun), e) {
                    (Ok((m, fs)), Eigen::Two { mu, v, w }) => {
                        let scale = size(m).max(1.0);
                        let by_lines = outer(v[0], w[0]) * fs[0] + outer(v[1], w[1]) * fs[1];
                        let p = size(outer(v[0], w[0])).max(1.0);
                        assert!(close(by_lines, m, 1e-9 * scale * p), "{s:?} {fun:?}: {by_lines:?} against {m:?}");
                        let (f1, f2) = (fun.of(mu[0]).unwrap(), fun.of(mu[1]).unwrap());
                        assert!(fs == [f1, f2]);
                        match fun {
                            Fun::Same => assert!(m == a),
                            Fun::Square => assert!(close(m, a * a, 0.0)),
                            Fun::Inverse => assert!(close(m * a, DMat2::IDENTITY, 1e-9 * scale * size(a).max(1.0))),
                            Fun::Root => assert!(close(m * m, a, 1e-9 * scale * scale) && fs.iter().all(|v| *v >= 0.0), "{s:?}"),
                        }
                        if s.symmetric() {
                            assert!(close(m, m.transpose(), 1e-12 * scale), "{s:?} {fun:?}");
                        }
                    }
                    (Err(why), Eigen::Two { mu, .. }) => match why {
                        Why::Zero(i) => assert!(fun == Fun::Inverse && mu[i] == 0.0, "{s:?}"),
                        Why::Negative(i) => assert!(fun == Fun::Root && mu[i] < 0.0, "{s:?}"),
                        Why::NoLines => panic!("{s:?}"),
                    },
                    (Err(why), _) => assert_eq!(why, Why::NoLines),
                    (Ok(_), _) => panic!("built along lines that are not there: {s:?}"),
                }
            }
        }
    }

    /// The film: at the stretch's start the map is I and the stretched shadows are the
    /// shadows, which add up to x; at its end the map is f(A), and the sum is f(A)x. In
    /// between each shadow is stretched evenly, and the map is the straight blend.
    #[test]
    fn the_film_ends_at_f_of_a() {
        for s in settings() {
            let e = eigen(&s);
            let Eigen::Two { v, w, .. } = e else { continue };
            for fun in FUNS {
                let Ok((m, fs)) = applied(&s, &e, fun) else { continue };
                let p = size(outer(v[0], w[0])).max(1.0);
                for stage in [0.0, 0.5, 1.0, 1.3, 1.5, 2.0, 2.7, 3.0] {
                    let b = build(v, w, fs, s.x, stage);
                    let map = film(m, stage);
                    let scale = size(map).max(1.0) * p * s.x.length().max(1.0);
                    assert!((b.feet[0] + b.feet[1] - s.x).length() < 1e-9 * scale, "{s:?}");
                    assert!((b.corner - map * s.x).length() < 1e-8 * scale, "{s:?} {fun:?} at {stage}");
                    if stage <= 1.0 {
                        assert!(map == DMat2::IDENTITY && (b.corner - s.x).length() < 1e-9 * scale);
                    }
                    if stage >= 2.0 {
                        assert!(map == m);
                    }
                }
            }
        }
    }

    /// The scene as an experiment's button leaves it, run at 60 frames a second until
    /// it has come to rest (or for `seconds` at most), with a moment of `before` first.
    fn run(k: usize, seconds: f32) -> Spectral {
        let mut scene = Spectral::new();
        scene.set_up(k);
        for _ in 0..(seconds * 60.0) as usize {
            scene.advance(1.0 / 60.0, 1.0 / 60.0);
            if !scene.animated() {
                break;
            }
        }
        scene
    }

    #[test]
    fn the_film_does_what_its_caption_says() {
        let scene = run(0, 30.0);
        assert!(!scene.animated());
        let s = scene.shown();
        assert!(s.stage == 3.0 && s.symmetric());
        assert!((DVec2::from_angle(s.theta) - DVec2::new(0.8, 0.6)).length() < 1e-15);
        let a = s.a();
        assert!(close(a, rows([1.46, 0.72, 0.72, 1.04]), 1e-14), "{a:?}");
        let Eigen::Two { mu, v, w } = eigen(&s) else { panic!() };
        assert!(mu == [2.0, 0.5]);
        let b = build(v, w, mu, s.x, 0.99);
        // at right angles: the feet are where x drops perpendicularly
        for i in 0..2 {
            assert!((s.x - b.feet[i]).dot(v[i]).abs() < 1e-14);
        }
        assert!((b.feet[0] - DVec2::new(1.0, 0.75)).length() < 1e-14 && (b.feet[1] - DVec2::new(-0.6, 0.8)).length() < 1e-14);
        assert!((b.feet[0] + b.feet[1] - s.x).length() < 1e-14 && s.x == DVec2::new(0.4, 1.55));
        let b = build(v, w, mu, s.x, 3.0);
        assert!((b.scaled[0] - DVec2::new(2.0, 1.5)).length() < 1e-14 && (b.scaled[1] - DVec2::new(-0.3, 0.4)).length() < 1e-14);
        assert!((b.corner - DVec2::new(1.7, 1.9)).length() < 1e-14 && (a * s.x - b.corner).length() < 1e-14);
        // A = 2 P1 + 0.5 P2
        assert!(close(outer(v[0], v[0]) * 2.0 + outer(v[1], v[1]) * 0.5, a, 1e-15));
    }

    #[test]
    fn equal_eigenvalues_do_what_their_caption_says() {
        let mut scene = Spectral::new();
        scene.set_up(1);
        let from = scene.ramp.map(|r| r.from).unwrap_or(f64::NAN);
        let mut shadows = Vec::new();
        for _ in 0..60 * 12 {
            scene.advance(1.0 / 60.0, 1.0 / 60.0);
            let s = scene.shown();
            if scene.glide.arrived() {
                // A = 1.5 I whatever the axes, and the stretched shadows add up to 1.5 x
                assert!(close(s.a(), DMat2::IDENTITY * 1.5, 1e-15), "{:?}", s.a());
                let Eigen::Two { mu, v, w } = eigen(&s) else { panic!() };
                assert!(mu == [1.5, 1.5] && s.stage == 3.0);
                let b = build(v, w, mu, s.x, s.stage);
                assert!((b.corner - s.x * 1.5).length() < 1e-14);
                assert!(close(outer(v[0], v[0]) + outer(v[1], v[1]), DMat2::IDENTITY, 1e-15));
                // (each shadow stays on the side of 0 away from its dot)
                assert!(w[0].dot(s.x) > 0.1 && w[1].dot(s.x) > 0.1, "{s:?}");
                shadows.push(b.feet[0]);
            }
            if !scene.animated() {
                break;
            }
        }
        assert!(!scene.animated());
        // the axes turned by 45 degrees, and the shadows changed
        let turned = scene.shown().theta - from;
        assert!((turned + std::f64::consts::FRAC_PI_4).abs() < 1e-12 && (DVec2::from_angle(from) - DVec2::new(0.8, 0.6)).length() < 1e-12, "{turned}");
        let (first, last) = (shadows.first().unwrap(), shadows.last().unwrap());
        assert!((*first - *last).length() > 0.5);
        assert!((scene.shown().a() * scene.shown().x - DVec2::new(0.6, 2.325)).length() < 1e-14);
    }

    #[test]
    fn the_flip_does_what_its_caption_says() {
        let scene = run(2, 30.0);
        let s = scene.shown();
        assert!(s.stage == 3.0 && !scene.animated());
        let a = s.a();
        assert!(close(a, rows([1.1, 1.2, 1.2, 0.4]), 1e-14) && (a.determinant() + 1.0).abs() < 1e-14);
        let Eigen::Two { mu, v, .. } = eigen(&s) else { panic!() };
        assert!(mu == [2.0, -0.5]);
        // where the purple factor passes 0 the map is 1.67 P1: flat onto the orange axis
        let (m, fs) = applied(&s, &eigen(&s), Fun::Same).unwrap();
        let s2 = 2.0 / 3.0;
        let at = DMat2::IDENTITY * (1.0 - s2) + m * s2;
        assert!(close(at, outer(v[0], v[0]) * (5.0 / 3.0), 1e-14) && num(5.0 / 3.0, 2) == "1.67" && at.determinant().abs() < 1e-14);
        // and the film does pass there: the factor goes evenly from 1 to -0.5
        let factor = |stage: f64| 1.0 + (fs[1] - 1.0) * ease(stage - 1.0);
        assert!(factor(1.0) == 1.0 && factor(2.0) == -0.5 && (1..100).any(|i| factor(1.0 + i as f64 / 100.0).abs() < 0.02));
        // Ax lands on the other side of the orange axis from x; the F is mirror-written
        let side = |p: DVec2| v[0].perp_dot(p).signum();
        assert!(side(s.x) != side(a * s.x) && (a * s.x - DVec2::new(2.3, 1.1)).length() < 1e-14);
        let area = |pts: &[DVec2]| (0..pts.len()).map(|i| pts[i].perp_dot(pts[(i + 1) % pts.len()])).sum::<f64>() * 0.5;
        let f_before: Vec<DVec2> = F_OUTLINE.iter().map(|p| F_AT + DVec2::from(*p)).collect();
        let f_after: Vec<DVec2> = f_before.iter().map(|p| a * *p).collect();
        assert!(area(&f_before) > 0.0 && area(&f_after) < 0.0 && (area(&f_after) + area(&f_before)).abs() < 1e-12);
    }

    #[test]
    fn the_twist_does_what_its_caption_says() {
        let mut scene = Spectral::new();
        scene.set_up(3);
        let (mut seen_two, mut seen_one, mut seen_complex) = (false, false, false);
        let mut last_angle = 90.0;
        let mut biggest_part = 0.0f64;
        for _ in 0..60 * 15 {
            scene.advance(1.0 / 60.0, 1.0 / 60.0);
            let s = scene.shown();
            if !scene.glide.arrived() {
                continue;
            }
            let a = s.a();
            // the corner entries are 0.72 - k and 0.72 + k
            assert!((a.y_axis.x - (0.72 - s.k)).abs() < 1e-14 && (a.x_axis.y - (0.72 + s.k)).abs() < 1e-14);
            match eigen(&s) {
                Eigen::Two { v, w, .. } => {
                    assert!(!seen_one && !seen_complex);
                    seen_two = true;
                    let apart = lines_apart(v);
                    assert!(apart <= last_angle + 1e-9 && (apart - (2.0 * s.k / 1.5).min(1.0).acos().to_degrees()).abs() < 1e-9);
                    last_angle = apart;
                    biggest_part = biggest_part.max(w[0].dot(s.x).abs()).max(w[1].dot(s.x).abs());
                }
                Eigen::One { v, mu } => {
                    seen_one = true;
                    assert!((s.k - 0.75).abs() < 1e-9 && mu == 1.25, "{s:?} {mu}");
                    // x is on the line they close in on
                    assert!(v.perp_dot(s.x).abs() < 1e-9 && (s.x.length() - 1.5).abs() < 1e-14);
                }
                Eigen::Complex { re, im } => {
                    seen_complex = true;
                    assert!(s.k > 0.75 && re == 1.25 && (im - (s.k * s.k - 0.5625).sqrt()).abs() < 1e-14);
                }
            }
            if !scene.animated() {
                break;
            }
        }
        assert!(seen_two && seen_one && seen_complex && !scene.animated());
        // the parts along the lines stay bounded on the line they close in on
        assert!(biggest_part < 1.1, "{biggest_part}");
        let s = scene.shown();
        assert!(s.k == 1.0 && eigen_text(eigen(&s)) == r"$1.25 \pm 0.66i$");
        assert!(num((2.0 * 0.5f64 / 1.5).acos().to_degrees(), 1) == "48.2");
        // anywhere else, the parts grow without bound as the lines meet
        let off = Setting { x: DVec2::new(1.0, 0.0), ..s };
        let parts = |k: f64| match eigen(&Setting { k, ..off }) {
            Eigen::Two { w, .. } => w[0].dot(off.x).abs(),
            _ => f64::NAN,
        };
        assert!(parts(0.75 - 1e-4) > 30.0 * parts(0.5) && parts(0.75 - 1e-8) > 30.0 * parts(0.75 - 1e-4));
    }

    #[test]
    fn the_inverse_does_what_its_caption_says() {
        let mut scene = run(4, 30.0);
        let s = scene.shown();
        assert!(s.stage == 3.0 && scene.drawn_fun == Fun::Inverse && !scene.animated());
        let e = eigen(&s);
        let (m, fs) = applied(&s, &e, Fun::Inverse).unwrap();
        assert!(close(m * s.a(), DMat2::IDENTITY, 1e-14));
        assert!(num(fs[0], 2) == "0.44" && num(fs[1], 2) == "1.56" && fs[1] == 1.5625);
        // long where A's ellipse is short, along the same axes
        let Eigen::Two { v, .. } = e else { panic!() };
        assert!((m * v[0]).length() < (s.a() * v[0]).length() && (m * v[1]).length() > (s.a() * v[1]).length());
        // the square root: 1.5 and 0.8, and twice is A
        scene.choose(Fun::Root);
        for _ in 0..60 * 10 {
            scene.advance(1.0 / 60.0, 1.0 / 60.0);
        }
        assert!(scene.drawn_fun == Fun::Root && scene.shown().stage == 3.0);
        let (r, fs) = applied(&s, &e, Fun::Root).unwrap();
        assert!(fs == [1.5, 0.8] && close(r * r, s.a(), 1e-14));
        // and neither is offered where its condition fails
        let zero = Setting { l: [2.25, 0.0], ..s };
        assert_eq!(applied(&zero, &eigen(&zero), Fun::Inverse).err(), Some(Why::Zero(1)));
        let negative = Setting { l: [2.25, -0.64], ..s };
        assert_eq!(applied(&negative, &eigen(&negative), Fun::Root).err(), Some(Why::Negative(1)));
    }

    /// A change of function never makes the picture jump: the new one is drawn only
    /// once the film is back at its shadows, where every function draws the same.
    #[test]
    fn a_new_function_comes_in_at_the_shadows() {
        let mut scene = run(0, 30.0);
        scene.choose(Fun::Inverse);
        let mut drawn = scene.drawn_fun;
        let mut switched = 0;
        for _ in 0..60 * 10 {
            let before = scene.shown();
            scene.advance(1.0 / 60.0, 1.0 / 60.0);
            if scene.drawn_fun != drawn {
                let s = scene.shown();
                let e = eigen(&s);
                let (old, new) = (applied(&s, &e, drawn).unwrap().0, applied(&s, &e, scene.drawn_fun).unwrap().0);
                // (the switch is made at the shadows, and the film may play on a step
                // in the same frame: the point drawn moves by no more than that step)
                let moved = (film(new, s.stage) * s.x - film(old, before.stage) * before.x).length();
                assert!(before.stage > 1.0 && s.stage < 1.02 && moved < 0.02, "{before:?} {s:?} {moved}");
                (drawn, switched) = (scene.drawn_fun, switched + 1);
            }
        }
        assert!(switched == 1 && scene.drawn_fun == Fun::Inverse && scene.shown().stage == 3.0 && !scene.animated());
    }

    /// Dragging a dot: round, it turns the axes and keeps its eigenvalue; through 0,
    /// the eigenvalue changes sign and the axes stay; near 0 or the other eigenvalue
    /// it snaps to it.
    #[test]
    fn the_dots_set_the_axes_and_the_eigenvalues() {
        let mut scene = run(0, 30.0);
        let start = scene.shown();
        // round a circle of radius 2 from the orange dot's place: a whole turn
        for j in 1..=200 {
            let at = -DVec2::from_angle(start.theta + j as f64 / 200.0 * std::f64::consts::TAU) * 2.0;
            scene.drag_axis(0, at);
            let s = scene.shown();
            assert!((s.l[0] - 2.0).abs() < 1e-12 && (Spectral::dot_at(&s, 0) - at).length() < 1e-12, "{j}");
        }
        assert!(close(scene.shown().a(), start.a(), 1e-12));
        // straight through 0 to the other side
        let q = scene.shown().axes()[0];
        for j in 0..=100 {
            let t = 2.0 - 3.0 * j as f64 / 100.0;
            scene.drag_axis(0, -(q * t + q.perp() * 1e-3));
            let l = scene.shown().l[0];
            let want = if t.abs() < SNAP {
                0.0
            } else if (t - 0.5).abs() < SNAP {
                0.5
            } else {
                t
            };
            // (a hair from where it snaps, either is right)
            let edge = [SNAP, -SNAP, 0.5 - SNAP, 0.5 + SNAP].iter().any(|e| (t - e).abs() < 2e-3);
            assert!(edge || (l - want).abs() < 1e-3, "{t}: {l}");
            assert!(scene.shown().axes()[0].dot(q) > 0.999);
        }
        assert!((scene.shown().l[0] + 1.0).abs() < 1e-3);
        // near the other eigenvalue, the same
        scene.drag_axis(0, -q * 0.51);
        assert!(scene.shown().l[0] == 0.5);
    }

    /// Degenerate input: every number drawn or read out stays finite.
    #[test]
    fn degenerate_input_leaves_everything_finite() {
        let mut cases = settings();
        cases.push(Setting { l: [1e-300, 2.0], theta: 0.0, k: 0.0, x: DVec2::splat(X_MAX), stage: 3.0 });
        cases.push(Setting { l: [0.0, 0.0], theta: 1.0, k: 0.0, x: DVec2::ZERO, stage: 3.0 });
        cases.push(Setting { l: [L_MAX, -L_MAX], theta: 1.0, k: K_MAX, x: DVec2::splat(-X_MAX), stage: 3.0 });
        for s in cases {
            let e = eigen(&s);
            for fun in FUNS {
                for stage in [0.0, 1.2, 2.5, 3.0] {
                    let s = Setting { stage, ..s };
                    if let (Ok((m, fs)), Eigen::Two { v, w, .. }) = (applied(&s, &e, fun), e) {
                        let b = build(v, w, fs, s.x, stage);
                        assert!(film(m, stage).is_finite() && b.corner.is_finite() && b.scaled.iter().all(|p| p.is_finite()), "{s:?} {fun:?}");
                    }
                }
            }
            let text = eigen_text(e);
            assert!(!text.contains('?') && !text.contains("NaN"), "{s:?}: {text}");
            let (first, second, _) = heading_text(&s, &e, Fun::Same, [1.0, 1.0]);
            assert!(!(first + &second).contains("NaN"));
        }
    }

    /// Every experiment's x and where the map takes it lie in the part of the plane the
    /// view is fitted to.
    #[test]
    fn every_experiment_is_in_view() {
        let (lo, hi) = ((FRAME_CENTRE - FRAME_HALF).as_dvec2(), (FRAME_CENTRE + FRAME_HALF).as_dvec2());
        let inside = |p: DVec2| p.cmpge(lo).all() && p.cmple(hi).all();
        for k in 0..TRIALS.len() {
            let scene = run(k, 30.0);
            let s = scene.shown();
            let e = eigen(&s);
            let m = applied(&s, &e, scene.drawn_fun).map_or(s.a(), |v| v.0);
            assert!(inside(s.x) && inside(m * s.x), "trial {k}");
            assert!(inside(m * (F_AT + DVec2::new(0.4, 0.6))) && inside(m * F_AT), "trial {k}");
        }
    }

    /// The panel and the picture fit the canvas, from a phone to a desktop.
    #[test]
    fn the_layout_fits_every_canvas() {
        for (w, h) in [(1080.0, 1000.0), (580.0, 700.0), (390.0, 800.0), (390.0, 560.0), (800.0, 600.0), (1600.0, 1000.0), (240.0, 180.0), (620.0, 480.0), (1100.0, 700.0)] {
            for inset in [0.0, 46.0] {
                let view = Vec2::new(w, h);
                let lay = Layout::of(view, inset);
                let (min, max) = lay.picture;
                assert!(min.cmpge(Vec2::ZERO).all() && max.cmple(view).all() && (max - min).min_element() > 50.0, "{w}x{h}: {lay:?}");
                if let Some(p) = &lay.panel {
                    assert!(p.min.cmpge(Vec2::ZERO).all() && p.max.cmple(view + Vec2::splat(0.5)).all(), "{w}x{h}: {p:?}");
                    let need = p.right + text_width("$A =$", p.text) + 8.0 + matrix_size(2, 2, CELL, p.text).x;
                    assert!(need <= p.max.x + 0.5 && p.bar.1.x > p.bar.0.x + 120.0, "{w}x{h}: {p:?}");
                    assert!(p.min.y >= inset || lay.narrow, "{w}x{h}: under the buttons");
                }
            }
        }
    }

    /// Each experiment shows the same thing whatever came before it.
    #[test]
    fn an_experiment_ends_the_same_whatever_came_before() {
        for k in 0..TRIALS.len() {
            let alone = run(k, 30.0).shown();
            for before in 0..TRIALS.len() {
                let mut scene = run(before, 30.0);
                scene.drag_axis(1, DVec2::new(-1.3, 0.4));
                scene.set_up(k);
                for _ in 0..60 * 30 {
                    scene.advance(1.0 / 60.0, 1.0 / 60.0);
                    if !scene.animated() {
                        break;
                    }
                }
                let s = scene.shown();
                assert!(close(s.a(), alone.a(), 1e-12) && s.x == alone.x && s.stage == alone.stage && scene.drawn_fun == TRIALS[k].fun, "{before} then {k}");
                assert!((DVec2::from_angle(s.theta) - DVec2::from_angle(alone.theta)).length() < 1e-12, "{before} then {k}");
            }
        }
    }
}

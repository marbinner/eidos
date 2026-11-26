//! Spin into rotation: in space a skew-symmetric matrix K is a spin, Kx = w × x, the
//! velocity of the point x when all of space turns about the line of w at the rate |w|.
//! Followed for a time the spin has turned space: e^(tK) is a rotation, Rodrigues'
//! formula folds its series into three terms, and every rotation is e^K for a spin K.

use super::subspaces::{Inputs, Still};
use super::*;
use glam::{DMat3, DVec3};
use std::f64::consts::{FRAC_PI_2, PI as HALF_TURN, TAU as WHOLE_TURN};

pub const DEF: SceneDef = SceneDef {
    id: "expskew",
    title: "Spin into rotation",
    group: EXPONENTIAL,
    blurb: "A skew-symmetric matrix is a spin, and its exponential is the rotation that spin makes: Rodrigues' formula, term by term.",
    make: |_| Box::new(ExpSkew::create()),
};

// ----- the arithmetic, in double precision -------------------------------------------

/// The longest spin there is here (radians in a unit of time): half a turn. Every
/// rotation is the turn of a spin no longer than this.
const W_MAX: f64 = HALF_TURN;
/// The longest each of two spins may be, so that their sum is never longer than `W_MAX`.
const W_PART_MAX: f64 = FRAC_PI_2;
/// How far from 0 the marked point may be dragged.
const P_MAX: f64 = 2.0;
/// The most terms of the series after I.
const N_MAX: usize = 20;

/// The skew-symmetric matrix of the spin w: the K with Kx = w × x for every x.
fn skew(w: DVec3) -> DMat3 {
    DMat3::from_cols(DVec3::new(0.0, w.z, -w.y), DVec3::new(-w.z, 0.0, w.x), DVec3::new(w.y, -w.x, 0.0))
}

/// The spin of the part of `m` that is not symmetric, (m − mᵀ)/2: for a skew-symmetric
/// m, the w with m = skew(w).
fn vee(m: DMat3) -> DVec3 {
    DVec3::new(m.y_axis.z - m.z_axis.y, m.z_axis.x - m.x_axis.z, m.x_axis.y - m.y_axis.x) * 0.5
}

fn trace(m: DMat3) -> f64 {
    m.x_axis.x + m.y_axis.y + m.z_axis.z
}

/// sin x / x, at 0 and near it too.
fn sinc(x: f64) -> f64 {
    if x.abs() < 1e-4 {
        1.0 - x * x / 6.0
    } else {
        x.sin() / x
    }
}

/// e^K for K = skew(w), by Rodrigues' formula. With θ = |w| and N = K/θ (the spin of
/// the same axis at the rate 1), N³ = −N, and the series I + K + K²/2! + … folds into
/// I + sin θ N + (1 − cos θ) N². It is worked out with half the angle, as
/// I + 2 cos(θ/2) V + 2 V² for V = sin(θ/2) N (sin θ = 2 sin(θ/2) cos(θ/2) and
/// 1 − cos θ = 2 sin²(θ/2)): no entry of V or V² is larger than 1, so the columns come
/// out of length 1 and at right angles to the last digit or two, at every angle, and a
/// small θ keeps all its digits, which 1 − cos θ loses.
fn exp_spin(w: DVec3) -> DMat3 {
    let theta = w.length();
    let v = skew(w * (0.5 * sinc(0.5 * theta)));
    DMat3::IDENTITY + v * (2.0 * (0.5 * theta).cos()) + v * v * 2.0
}

/// The terms of the series for e^(tK) p, from the 0th to the nth: p, tKp, (tK)²p/2!, …
fn terms(w: DVec3, t: f64, p: DVec3, n: usize) -> Vec<DVec3> {
    let mut out = Vec::with_capacity(n + 1);
    let mut term = p;
    out.push(p);
    for k in 1..=n {
        term = w.cross(term) * (t / k as f64);
        out.push(term);
    }
    out
}

/// I + tK + (tK)²/2! + … + (tK)ⁿ/n!.
fn partial_sum(w: DVec3, t: f64, n: usize) -> DMat3 {
    let k = skew(w * t);
    let (mut sum, mut term) = (DMat3::IDENTITY, DMat3::IDENTITY);
    for i in 1..=n {
        term = term * k * (1.0 / i as f64);
        sum += term;
    }
    sum
}

/// A rotation read back as a spin: the angle of its turn and its axis.
#[derive(Clone, Copy, Debug)]
struct Spin {
    /// From 0 to π.
    theta: f64,
    /// Of length 1, by the right-hand rule; None if there is no turn at all.
    axis: Option<DVec3>,
    /// A half turn: R − Rᵀ is 0 and says nothing, and the axis was read from the
    /// symmetric part, which gives its line but not which way along it (both ways are
    /// the same turn there).
    half: bool,
}

/// From a rotation R back to its spin. The angle from the diagonal, tr R = 1 + 2 cos θ,
/// and the axis from the part that is not symmetric, R − Rᵀ = 2 sin θ N, while sin θ
/// is not small. Nearer a half turn the axis is read from the symmetric part instead,
/// (R + Rᵀ)/2 − cos θ I = (1 − cos θ) n nᵀ, whose largest column lies along n; R − Rᵀ
/// then only says which way along it, until at the half turn it says nothing, and the
/// way nearer `prefer` is taken.
fn log_rotation(r: DMat3, prefer: DVec3) -> Spin {
    let none = Spin { theta: 0.0, axis: None, half: false };
    let c = 0.5 * (trace(r) - 1.0);
    let v = vee(r);
    let s = v.length();
    if !(c.is_finite() && s.is_finite()) {
        return none;
    }
    // (from the sine and the cosine together, not from either alone: exact at both ends)
    let theta = s.atan2(c);
    if theta <= 1e-14 {
        return none;
    }
    if c > -0.5 {
        return Spin { theta, axis: Some(v / s), half: false };
    }
    let m = (r + r.transpose()) * 0.5 - DMat3::IDENTITY * c;
    let diag = [m.x_axis.x, m.y_axis.y, m.z_axis.z];
    let j = if diag[0] >= diag[1] && diag[0] >= diag[2] {
        0
    } else if diag[1] >= diag[2] {
        1
    } else {
        2
    };
    let col = m.col(j);
    let len = col.length();
    if !(len > 0.0 && len.is_finite()) {
        return none;
    }
    let n = col / len;
    let along = v.dot(n);
    let half = along.abs() < 1e-13;
    let flip = if half { prefer.dot(n) < 0.0 } else { along < 0.0 };
    Spin { theta, axis: Some(if flip { -n } else { n }), half }
}

/// How far apart two rotations are: the angle of the turn from one to the other (degrees).
fn apart(a: DMat3, b: DMat3) -> f64 {
    log_rotation(a.transpose() * b, DVec3::X).theta.to_degrees()
}

/// The largest entry of MᵀM − I: 0 when the columns have length 1 and are at right
/// angles (the entries of MᵀM are the dot products of the columns).
fn off_orthonormal(m: DMat3) -> f64 {
    (m.transpose() * m - DMat3::IDENTITY).to_cols_array().iter().fold(0.0, |w, v| w.max(v.abs()))
}

/// The turn of `w1` and then the turn of `w2`, as far as it has got at the time `t`
/// from 0 to 1: w1's in the first half (at twice its rate), then w2's.
fn one_then_other(w1: DVec3, w2: DVec3, t: f64) -> DMat3 {
    let first = (2.0 * t).clamp(0.0, 1.0);
    let second = (2.0 * t - 1.0).clamp(0.0, 1.0);
    exp_spin(w2 * second) * exp_spin(w1 * first)
}

/// An angle in degrees for the screen, to one decimal.
fn deg(degrees: f64) -> String {
    format!("{}°", num(degrees, 1))
}

/// The same as markup, to go in a formula.
fn deg_tex(degrees: f64) -> String {
    format!("{}^\\circ", num(degrees, 1))
}

/// A point or a direction for the screen, as markup: (x, y, z).
fn coords(v: DVec3, decimals: usize) -> String {
    format!("({}, {}, {})", num(v.x, decimals), num(v.y, decimals), num(v.z, decimals))
}

/// The spin w and its parts: the axis n = w/|w| (None for no spin) and the rate |w|.
fn axis_of(w: DVec3) -> Option<DVec3> {
    let len = w.length();
    (len > 1e-9 && len.is_finite()).then(|| w / len)
}

/// `v` no longer than `most`.
fn at_most(v: DVec3, most: f64) -> DVec3 {
    let len = v.length();
    if !len.is_finite() {
        DVec3::ZERO
    } else if len > most {
        v * (most / len)
    } else {
        v
    }
}

// ----- experiments ---------------------------------------------------------------------

/// What the picture shows besides the turn itself.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Mode {
    /// The turn alone.
    Turn,
    /// The terms of the series from the point, end to end.
    Series,
    /// Rodrigues' three pieces from the point.
    Pieces,
    /// Two spins: their sum, and their turns one after the other.
    Sum,
    /// A rotation given, and the spin read back from it.
    Log,
}

const MODES: [(Mode, &str); 5] = [(Mode::Turn, "the turn"), (Mode::Series, "the series"), (Mode::Pieces, "three pieces"), (Mode::Sum, "two spins"), (Mode::Log, "back to the spin")];

struct Trial {
    label: &'static str,
    look_for: &'static str,
    mode: Mode,
    /// Played from t = 0 to 1 once the picture has glided there (or else left at t = 1).
    play: bool,
}

const TRIALS: [Trial; 5] = [
    Trial {
        label: "one spin, followed to the end",
        look_for: r"$w$ points along $(1, 1, 1)$ and is 2.09 long: a spin of 2.09 radians (120°) in a unit of time about that line. The velocity $w \times x$ of every point (cyan) is at right angles to the axis and to the point's own arrow from $0$, so each point keeps its height along the axis and its distance from $0$: it goes round a circle about the axis. By $t = 1$ space has turned 120°: $p = (1.1, -0.5, 0.3)$ is at $(0.3, 1.1, -0.5)$, the arrow that lay along $x$ lies along $y$, the one along $y$ along $z$ and the one along $z$ along $x$, and the cube is back on its own outline. All the way $R^\top R = I$ and $\det R = 1$.",
        mode: Mode::Turn,
        play: true,
    },
    Trial {
        label: "the series, a term at a time",
        look_for: r"The terms $(tK)^k p / k!$ of $e^{tK} p$ at $t = 1$, end to end (red), from $p = (1, 0, 0)$, one more every second. The first, $Kp = w \times p$, is a straight step along the velocity at $p$; each after it is the one before turned a quarter turn about the axis and $2.09/k$ times as long. The sum swings out to 1.58 away from $e^K p$, then spirals in: within 0.01 of it from $N = 7$ on. It ends at $(0, 1, 0)$, the tip of the pink arrow: $e^K p$ is the first column of $e^K$.",
        mode: Mode::Series,
        play: false,
    },
    Trial {
        label: "three pieces are enough",
        look_for: r"Write $K = |w|\,N$, with $N$ the spin of rate 1 about the same axis $n$. Then $N^3 = -N$, so every power of $N$ is $\pm N$ or $\pm N^2$, and the endless series folds into three terms: $e^{\theta N} = I + \sin\theta\, N + (1 - \cos\theta)\, N^2$, with $\theta = t|w|$. From $p$, the step $(1 - \cos\theta)\, N^2 p$ (purple) goes straight in at the axis and leaves $\cos\theta$ of $p$'s part across it; the step $\sin\theta\, Np$ (cyan) is at right angles to that, along $n \times p$; together they end on the circle. The part of $p$ along the axis stays; the part across it turns, $\cos\theta$ of itself plus $\sin\theta$ of $n \times p$. At 120° the steps are 1.5 and 0.866 times $p$'s distance from the axis: the first goes through the axis and half as far again.",
        mode: Mode::Pieces,
        play: true,
    },
    Trial {
        label: "two spins added, two turns made",
        look_for: r"$w_1$ is a quarter turn about $y$ and $w_2$ one about $z$. At every point the velocities add, $w_1 \times x + w_2 \times x = (w_1 + w_2) \times x$, so both spins at once are one spin, $w_1 + w_2$: 127.3° about the line through $(0, 1, 1)$. Made one after the other, $w_1$'s turn and then $w_2$'s make $e^{K_2} e^{K_1}$ (blue): 120° about the line through $(-1, 1, 1)$. The two end 62.4° apart: $e^{K_1 + K_2} \ne e^{K_2} e^{K_1}$. Drag one dot onto the line of the other arrow and the gap closes: spins about the same axis commute.",
        mode: Mode::Sum,
        play: true,
    },
    Trial {
        label: "from a rotation back to its spin",
        look_for: r"The white outline is a rotation $R$: a quarter turn about $x$, then one about $z$. Its angle is in its diagonal: $\tr R = 0 = 1 + 2\cos\theta$, so $\theta = 120^\circ$. Its axis is in the part that is not symmetric: $R - R^\top = 2\sin\theta\, N$, which gives $n = (1, 1, 1)/\sqrt{3}$. Played, the spin $w = \theta n$ lands exactly on $R$. It is not the only spin that does: 480° about $n$, or 240° about $-n$, end there too. At a half turn (set the angle to 180°) $R - R^\top = 0$, and the axis has to be read from $R + I = 2nn^\top$ instead.",
        mode: Mode::Log,
        play: true,
    },
];
const ONE_TRIAL: usize = 0;
const SERIES_TRIAL: usize = 1;
const PIECES_TRIAL: usize = 2;
const SUM_TRIAL: usize = 3;
const LOG_TRIAL: usize = 4;

/// A turn of `degrees` about the axis of space `k` (0, 1, 2: x, y, z), made directly,
/// not from a spin.
fn about(k: usize, degrees: f64) -> DMat3 {
    let r = degrees.to_radians();
    match k {
        0 => DMat3::from_rotation_x(r),
        1 => DMat3::from_rotation_y(r),
        _ => DMat3::from_rotation_z(r),
    }
}

/// The spin of the first three experiments: 120° about the line through (1, 1, 1).
fn diagonal_spin() -> DVec3 {
    DVec3::ONE.normalize() * (WHOLE_TURN / 3.0)
}

/// The two quarter turns of the experiment with two spins: about y, and about z.
fn spin_pair() -> (DVec3, DVec3) {
    (DVec3::Y * FRAC_PI_2, DVec3::Z * FRAC_PI_2)
}

/// The rotation the last experiment reads its spin back from: a quarter turn about x,
/// then one about z.
fn given_rotation() -> DMat3 {
    about(2, 90.0) * about(0, 90.0)
}

/// The marked point of the experiments (but the series').
const POINT: DVec3 = DVec3::new(1.1, -0.5, 0.3);

/// What experiment `k` sets: the spin, the second spin (two spins), the point and the
/// time.
fn setting_of(k: usize) -> Setting {
    let (w1, w2) = spin_pair();
    match k {
        SERIES_TRIAL => Setting { w: diagonal_spin(), w2: DVec3::ZERO, p: DVec3::X, t: 1.0 },
        SUM_TRIAL => Setting { w: w1 + w2, w2, p: POINT, t: 1.0 },
        LOG_TRIAL => {
            let back = log_rotation(given_rotation(), DVec3::ONE);
            Setting { w: back.axis.unwrap_or(DVec3::Z) * back.theta, w2: DVec3::ZERO, p: POINT, t: 1.0 }
        }
        ONE_TRIAL | PIECES_TRIAL => Setting { w: diagonal_spin(), w2: DVec3::ZERO, p: POINT, t: 1.0 },
        _ => opening(),
    }
}

/// What the scene opens with: 110° about a line leaning off the upright, played.
fn opening() -> Setting {
    Setting { w: DVec3::new(0.45, 0.25, 1.0).normalize() * 110f64.to_radians(), w2: DVec3::ZERO, p: DVec3::new(1.05, -0.35, 0.25), t: 0.0 }
}

// ----- what glides -----------------------------------------------------------------------

/// What glides when a button changes it: the spin (the sum, with two), the second
/// spin, the marked point and the time.
#[derive(Clone, Copy, Debug, PartialEq)]
struct Setting {
    w: DVec3,
    w2: DVec3,
    p: DVec3,
    t: f64,
}

impl Setting {
    fn numbers(&self) -> [f64; 10] {
        [self.w.x, self.w.y, self.w.z, self.w2.x, self.w2.y, self.w2.z, self.p.x, self.p.y, self.p.z, self.t]
    }

    fn from_numbers(v: [f64; 10]) -> Setting {
        Setting { w: DVec3::new(v[0], v[1], v[2]), w2: DVec3::new(v[3], v[4], v[5]), p: DVec3::new(v[6], v[7], v[8]), t: v[9] }
    }

    /// The first of two spins (the spin less the second).
    fn w1(&self) -> DVec3 {
        self.w - self.w2
    }
}

/// How long a change of setting takes to glide to (seconds): as long as the scene
/// "rotations" takes to glide back to the start of a play.
const GLIDE_SECONDS: f64 = 0.8;

/// Easing in and out, from 0 at 0 to 1 at 1.
fn ease(u: f64) -> f64 {
    let u = u.clamp(0.0, 1.0);
    u * u * (3.0 - 2.0 * u)
}

/// A setting as asked for, and as drawn: the drawn one goes from where it was to the
/// one asked for in `GLIDE_SECONDS`, setting off gently and arriving exactly.
#[derive(Clone, Copy, Debug)]
struct Glide {
    from: Setting,
    to: Setting,
    shown: Setting,
    /// How far it has got (0 to 1).
    done: f64,
}

impl Glide {
    fn at(s: Setting) -> Glide {
        Glide { from: s, to: s, shown: s, done: 1.0 }
    }

    fn arrived(&self) -> bool {
        self.done >= 1.0
    }

    /// Ask for a change of the setting: the drawing glides to it from where it is now.
    fn aim(&mut self, change: impl Fn(&mut Setting)) {
        let mut to = self.to;
        change(&mut to);
        if to != self.to {
            (self.from, self.to, self.done) = (self.shown, to, 0.0);
        }
    }

    /// Change part of the setting at once, as asked for and as drawn: what a hand moves.
    fn now(&mut self, change: impl Fn(&mut Setting)) {
        for s in [&mut self.from, &mut self.to, &mut self.shown] {
            change(s);
        }
    }

    /// Move the drawn setting on by `dt` seconds.
    fn step(&mut self, dt: f64) {
        if self.arrived() || dt.is_nan() || dt <= 0.0 {
            return;
        }
        self.done = (self.done + dt / GLIDE_SECONDS).min(1.0);
        if self.arrived() {
            self.shown = self.to;
            return;
        }
        let (e, from, to) = (ease(self.done), self.from.numbers(), self.to.numbers());
        self.shown = Setting::from_numbers(std::array::from_fn(|i| from[i] + (to[i] - from[i]) * e));
    }
}

/// Seconds that playing takes from t = 0 to t = 1.
const PLAY_SECONDS: f64 = 2.5;
/// Seconds between one term of the series and the next, as an experiment counts them
/// up; and the number of terms it counts up to.
const TERM_EVERY: f32 = 1.0;
const COUNT_TO: usize = 12;

// ----- what is drawn, and where ------------------------------------------------------------

/// The cube of side 1 about 0 that the turn carries: its corners, by the bits of their
/// number (x, y, z), its edges and its faces.
const HALF: f32 = 0.5;
const EDGES: [(usize, usize); 12] = [(0, 1), (2, 3), (4, 5), (6, 7), (0, 2), (1, 3), (4, 6), (5, 7), (0, 4), (1, 5), (2, 6), (3, 7)];
const FACES: [[usize; 4]; 6] = [[0, 2, 6, 4], [1, 3, 7, 5], [0, 1, 5, 4], [2, 3, 7, 6], [0, 1, 3, 2], [4, 5, 7, 6]];

fn corner(i: usize) -> Vec3 {
    let s = |bit: usize| if i & bit == 0 { -HALF } else { HALF };
    Vec3::new(s(1), s(2), s(4))
}

/// Points on a sphere about 0, evenly spread, where the velocities are drawn: this
/// many, this far out, each arrow this share of the velocity's length.
const FIELD_N: usize = 84;
const FIELD_R: f32 = 1.35;
const FIELD_SCALE: f32 = 1.0 / 6.0;

fn field_points() -> Vec<Vec3> {
    let golden = std::f32::consts::PI * (3.0 - 5f32.sqrt());
    (0..FIELD_N)
        .map(|i| {
            let z = 1.0 - (2 * i + 1) as f32 / FIELD_N as f32;
            let r = (1.0 - z * z).max(0.0).sqrt();
            let a = golden * i as f32;
            Vec3::new(r * a.cos(), r * a.sin(), z) * FIELD_R
        })
        .collect()
}

/// How far the axis is drawn either way, and the axes of space.
const AXIS_LEN: f32 = 2.4;
const SPACE_LEN: f32 = 2.0;
/// Points on the arc a point turns along, and on its whole circle.
const ARC_POINTS: usize = 64;

/// The colours, each with one meaning.
const COLS: [Color; 3] = pal::AXIS;
const SPIN: Color = pal::ORANGE;
const CUBE: Color = pal::YELLOW;
const SPEED: Color = pal::CYAN;
const POINTS: Color = pal::WHITE;
const SERIES: Color = pal::RED;
const ACROSS: Color = pal::PURPLE;
const OTHER: Color = pal::BLUE;
const GAP: Color = pal::RED;

/// The parts of the picture drawn as lines.
#[derive(Clone, Copy, PartialEq)]
enum Part {
    /// What keeps its number of points from frame to frame (in the usual batch).
    Fixed,
    /// The axis and the spin's arrow, which are not there without a spin.
    Axis,
    /// The marked point's circle, arc and spokes, which are not there for a point on
    /// the axis.
    Point,
    /// The series' arrows, which come one at a time.
    Series,
    /// What only one of the other ways of showing it draws.
    Extra,
}

/// The line batch a part is drawn in. A part whose number of points can change as the
/// picture moves has a batch of its own (different from the usual one by a hair of its
/// width, and nothing else), so that from one frame to the next the parts that stay the
/// same can still be compared point for point: that is how `eidos motion` tells smooth
/// motion from jumps.
fn lines_for<'a>(f: &'a mut Frame, part: Part) -> &'a mut crate::gfx::draw::LineBatch {
    if part == Part::Fixed {
        return &mut f.dl.lines;
    }
    f.dl.lines_with(LineKind::Blend, LineStyle { width_scale: 1.0 + 1e-4 * part as u8 as f32, ..Default::default() })
}

/// An arrow from `a` to `b`, as `Frame::arrow` draws one, its shaft in the batch of
/// `part`: always a shaft and a head, also for an arrow of no length (drawn faint as it
/// shrinks, and not at all when it has none), so that the number of points drawn stays
/// the same.
fn arrow(f: &mut Frame, part: Part, a: Vec3, b: Vec3, width: f32, color: Color) {
    let d = b - a;
    let len = d.length();
    let dir = if len > 1e-9 && len.is_finite() { d / len } else { Vec3::X };
    let wpp = f.cam.world_per_px(b) * f.px;
    let head = ((width * 3.2 + 5.0) * wpp).min(len * 0.6);
    let to_cam = (f.cam.eye - b).normalize_or_zero();
    let mut side = dir.cross(to_cam);
    if side.length() < 1e-4 {
        side = dir.cross(f.cam.up);
    }
    let side = side.normalize_or_zero() * (head * 0.42);
    let base = b - dir * head;
    // (an arrow a few pixels long is a dot more than an arrow)
    let fade = (len / (6.0 * wpp).max(1e-9)).min(1.0);
    let color = color.a(color.0[3] as f32 / 255.0 * fade);
    lines_for(f, part).segment(a, b - dir * (head * 0.8), width, color);
    f.dl.tris_alpha.tri(b, base + side, base - side, color);
}

/// The cube carried by `m`: see-through faces shaded as if lit from above, drawn back to
/// front, and its edges.
fn cube(f: &mut Frame, m: Mat3) {
    let light = Vec3::new(0.35, -0.3, 0.88).normalize();
    let eye = f.cam.eye;
    let mut faces: Vec<[Vec3; 4]> = FACES.iter().map(|q| q.map(|i| m * corner(i))).collect();
    let far = |q: &[Vec3; 4]| ((q[0] + q[2]) * 0.5).distance(eye);
    faces.sort_by(|p, q| far(q).total_cmp(&far(p)));
    for q in faces {
        let normal = (q[1] - q[0]).cross(q[3] - q[0]).normalize_or_zero();
        let tone = 0.45 + 0.55 * normal.dot(light).abs();
        f.dl.tris_alpha.quad(q[0], q[1], q[2], q[3], Color::from_linear(CUBE.to_linear() * tone).a(0.16));
    }
    for (a, b) in EDGES {
        f.line(m * corner(a), m * corner(b), 1.6, CUBE.a(0.9));
    }
}

/// The cube's outline alone, dashed: where it was, or where it is to go.
fn outline(f: &mut Frame, m: Mat3, width: f32, color: Color) {
    for (a, b) in EDGES {
        f.dashed(m * corner(a), m * corner(b), width, color);
    }
}

/// The circle a point `p` goes round under a spin about the unit axis `n`: its centre
/// on the axis, and the two arrows across the axis from it (`p`'s own part across, and
/// that turned a quarter turn), so that the point at angle φ is c + cos φ r + sin φ s.
/// None for a point on the axis, which does not move.
fn circle_of(n: DVec3, p: DVec3) -> Option<(DVec3, DVec3, DVec3)> {
    let c = n * n.dot(p);
    let r = p - c;
    (r.length() > 1e-6).then(|| (c, r, n.cross(r)))
}

// ----- the layout ----------------------------------------------------------------------

/// Where the panel of numbers goes on a canvas of `view` logical pixels, `inset` of
/// whose top lies under the application's buttons: over the picture's top-left corner,
/// or below the picture on a canvas narrow and tall enough (a phone held upright).
/// Left out where the canvas cannot spare the room.
#[derive(Clone, Copy, Debug, PartialEq)]
struct Layout {
    narrow: bool,
    panel: Option<(Vec2, Vec2)>,
    /// Where the picture's part of the canvas ends at the bottom.
    below: f32,
}

/// The panel's width, and its height when it says the most it can.
const PANEL_W: f32 = 392.0;
const PANEL_H: f32 = 300.0;
/// The panel's rows: the size of the matrices' numbers, the room for the names before
/// them, and the lines of words.
const MAT: f32 = 12.5;
const LEAD: f32 = 50.0;
const LINE: f32 = 17.0;

impl Layout {
    /// (`height`: the panel's height on a wide canvas. On a phone it is always the
    /// largest, so that the picture above it stays put.)
    fn of(view: Vec2, inset: f32, height: f32) -> Layout {
        if view.x < 620.0 && view.y >= 600.0 {
            let top = view.y - PANEL_H - 6.0;
            Layout { narrow: true, panel: Some((Vec2::new(6.0, top), Vec2::new(view.x - 6.0, view.y - 6.0))), below: top - 4.0 }
        } else {
            let top = 14.0f32.max(inset + 8.0);
            let fits = view.x >= PANEL_W + 300.0 && view.y >= top + PANEL_H + 120.0;
            Layout { narrow: false, panel: fits.then(|| (Vec2::new(14.0, top), Vec2::new(14.0 + PANEL_W, top + height))), below: view.y }
        }
    }

    fn covers(&self, p: Vec2) -> bool {
        self.panel.is_some_and(|(min, max)| inside_rect(p, (min, max)))
    }
}

fn inside_rect(p: Vec2, (min, max): (Vec2, Vec2)) -> bool {
    p.cmpge(min).all() && p.cmple(max).all()
}

/// Where the parts of the panel go.
struct Parts {
    /// The play button, the two ends of the time track (at its middle), where the time
    /// is written.
    play: (Vec2, Vec2),
    track: (Vec2, Vec2),
    time: Vec2,
    /// Top-left corners of the two matrices, and where the lines under them start.
    k_at: Vec2,
    r_at: Vec2,
    lines: Vec2,
}

fn parts((min, max): (Vec2, Vec2)) -> Parts {
    let (x, y) = (min.x + 12.0, min.y + 10.0);
    let play = (Vec2::new(x, y), Vec2::new(x + 30.0, y + 26.0));
    let track = (Vec2::new(x + 56.0, y + 13.0), Vec2::new(max.x - 86.0, y + 13.0));
    let time = Vec2::new(max.x - 12.0, y + 13.0);
    let k_at = Vec2::new(x + LEAD, y + 40.0);
    let r_at = Vec2::new(x + LEAD, k_at.y + matrix_size(3, 3, 6, MAT).y + 24.0);
    let lines = Vec2::new(x, r_at.y + matrix_size(3, 3, 7, MAT).y + 10.0);
    Parts { play, track, time, k_at, r_at, lines }
}

/// How far from the origin the picture reaches (world units): on a phone, a little less,
/// letting the ends of the axis and of a long spin run to the edge of the canvas, for a
/// larger picture. And the room kept round it for the names written at its edges
/// (logical pixels).
const REACH_DRAWN: f32 = 2.3;
const REACH_NARROW: f32 = 1.95;
const LABEL_ROOM: f32 = 16.0;

/// The round part of the canvas the picture is fitted into: its centre and radius, in
/// logical pixels. As large as fits below the application's buttons and above the
/// caption (and above the panel, on a phone) and, on a wide canvas, clear of the panel
/// in its corner.
fn stage(canvas: Vec2, inset: f32, lay: &Layout) -> (Vec2, f32) {
    let top = inset + 4.0;
    let bottom = if lay.narrow { lay.below } else { canvas.y } - caption_height(canvas);
    let bottom = bottom.max(top + 60.0);
    let middle = Vec2::new(canvas.x * 0.5, (top + bottom) * 0.5);
    // (of circles about as large, the one nearest the middle: a canvas wider than it is
    // tall has a row of them)
    let score = |c: Vec2, r: f32| r - 0.01 * (c - middle).length();
    let mut best = (middle, 30.0f32);
    for i in 0..=48 {
        for j in 0..=48 {
            let c = Vec2::new(canvas.x * i as f32 / 48.0, top + (bottom - top) * j as f32 / 48.0);
            let mut r = c.x.min(canvas.x - c.x).min(c.y - top).min(bottom - c.y);
            if let (false, Some((lo, hi))) = (lay.narrow, lay.panel) {
                let nearest = c.max(lo).min(hi);
                r = r.min((c - nearest).length() - 8.0);
            }
            if r > 30.0 && score(c, r) > score(best.0, best.1) {
                best = (c, r);
            }
        }
    }
    best
}

/// The parts of the caption along the foot of the picture, naming what has no name of
/// its own there.
const CAPTION: [&str; 3] = ["grey: the axes of space", "dashed: the cube at $t = 0$", "cyan arrows: velocities, a sixth of their length"];

/// How much of the canvas's foot the caption takes (the picture keeps clear of it).
fn caption_height(canvas: Vec2) -> f32 {
    if canvas.y < 300.0 {
        return 0.0;
    }
    caption_lines(canvas.x, &CAPTION).len() as f32 * CAPTION_LINE + 18.0
}

const CAPTION_SIZE: f32 = 12.0;
const CAPTION_LINE: f32 = 19.0;

/// The caption's parts set side by side for as long as they fit across, on further
/// lines when they do not.
fn caption_lines(width: f32, parts: &[&str]) -> Vec<String> {
    let (room, gap) = (width - 36.0, "      ");
    let mut lines: Vec<String> = Vec::new();
    for part in parts {
        match lines.last_mut() {
            Some(last) if text_width(last, CAPTION_SIZE) + text_width(gap, CAPTION_SIZE) + text_width(part, CAPTION_SIZE) <= room => {
                last.push_str(gap);
                last.push_str(part);
            }
            _ => lines.push(part.to_string()),
        }
    }
    lines
}

/// The caption, on a dark plate, its last line ending at `bottom`.
fn caption(f: &mut Frame, bottom: f32) {
    let view = f.view_size();
    if view.y < 300.0 {
        return;
    }
    let lines = caption_lines(view.x, &CAPTION);
    let count = lines.len();
    for (k, line) in lines.into_iter().enumerate() {
        let up = (count - 1 - k) as f32 * CAPTION_LINE;
        f.caption_px(Vec2::new(18.0, bottom - up), line, CAPTION_SIZE, pal::DIM, Align2::LEFT_BOTTOM);
    }
}

/// The direction the scene is seen from at first: in front, a little to the right and
/// above, so that the line through (1, 1, 1) is seen at about 50° to the line of sight
/// and the circles about it as round ellipses.
const YAW: f32 = -0.245;
const PITCH: f32 = 0.594;
/// (The tangent of half the camera's field of view.)
const TAN_HALF_FOV: f32 = 0.3203;

/// The camera's right and up directions in the view the scene opens with.
fn screen_axes() -> (Vec3, Vec3) {
    let right = Vec3::new(-YAW.sin(), YAW.cos(), 0.0);
    let up = Vec3::new(-PITCH.sin() * YAW.cos(), -PITCH.sin() * YAW.sin(), PITCH.cos());
    (right, up)
}

/// The play button over the picture (logical pixels).
fn hud_button(f: &mut Frame, (min, max): (Vec2, Vec2), text: &str, over: bool) {
    let fill = if over { Color::hex(0x33405E) } else { Color::hex(0x222B40) };
    hud_rect(f, min, max, fill);
    f.text_px((min + max) * 0.5, text, 13.0, pal::TEXT, Align2::CENTER_CENTER);
}

// ----- the scene -----------------------------------------------------------------------

pub struct ExpSkew {
    glide: Glide,
    mode: Mode,
    playing: bool,
    /// Play once the glide to a new setting has arrived.
    play_when_there: bool,
    /// Terms of the series after I, and how many are drawn (gliding to it); counting
    /// them up: seconds until the next one.
    terms: usize,
    terms_shown: f64,
    counting: Option<f32>,
    /// Whether the velocities on the sphere are drawn.
    field: bool,
    /// The axis kept for when there is no spin: the one an angle given again turns about.
    axis: DVec3,
    /// The time track on the picture being dragged.
    scrub: bool,
    /// Where the velocities are drawn.
    sphere: Vec<Vec3>,
    /// The canvas last drawn on, and how much of its top the application's buttons
    /// cover: they decide where the panel goes, and with it where the picture goes.
    canvas: Vec2,
    inset: f32,
    measured: bool,
    new_view: bool,
    /// The names in the picture as they were written last, kept while nothing they are
    /// placed from changes (see `Still`).
    names: Still,
}

impl ExpSkew {
    /// (Nothing here needs the graphics device, so the tests can make one too.)
    fn create() -> Self {
        let start = opening();
        ExpSkew {
            glide: Glide::at(start),
            mode: Mode::Turn,
            playing: false,
            play_when_there: true,
            terms: 4,
            terms_shown: 4.0,
            counting: None,
            field: true,
            axis: axis_of(start.w).unwrap_or(DVec3::Z),
            scrub: false,
            sphere: field_points(),
            canvas: Vec2::new(1080.0, 1000.0),
            inset: 0.0,
            measured: false,
            new_view: false,
            names: Still::default(),
        }
    }

    /// Set up experiment `k` of `TRIALS`: glide to its setting, then play it.
    fn set_up(&mut self, k: usize) {
        let Some(trial) = TRIALS.get(k) else { return };
        self.set_mode(trial.mode);
        let to = setting_of(k);
        self.glide.aim(|s| *s = Setting { t: if trial.play { 0.0 } else { to.t }, ..to });
        (self.playing, self.play_when_there, self.scrub) = (false, trial.play, false);
        if trial.mode == Mode::Series {
            (self.terms, self.terms_shown, self.counting) = (0, 0.0, Some(TERM_EVERY));
        } else {
            self.counting = None;
        }
        // (two spins are busy enough without them)
        self.field = trial.mode != Mode::Sum;
    }

    /// Show something else. Two spins start from the second at nothing, so that the
    /// first is the whole spin at first and the second grows out of 0.
    fn set_mode(&mut self, mode: Mode) {
        if mode != Mode::Sum || self.mode != Mode::Sum {
            // (outside two spins the second means nothing, and is kept at 0)
            self.glide.now(|s| s.w2 = DVec3::ZERO);
        }
        if mode == Mode::Sum && self.mode != Mode::Sum {
            let (w1, w2) = spin_pair();
            self.glide.aim(|s| (s.w, s.w2) = (w1 + w2, w2));
        }
        self.mode = mode;
    }

    /// The setting as drawn.
    fn shown(&self) -> Setting {
        self.glide.shown
    }

    /// Play from where t is (from 0, if it is at the end).
    fn play(&mut self) {
        if self.glide.to.t >= 1.0 - 1e-9 {
            self.glide.aim(|s| s.t = 0.0);
            self.play_when_there = true;
        } else {
            self.playing = true;
        }
    }

    fn toggle_play(&mut self) {
        if self.playing || self.play_when_there {
            (self.playing, self.play_when_there) = (false, false);
        } else {
            self.play();
        }
    }

    /// Move everything on: `dt` seconds of the scene's time (playing, counting) and
    /// `real_dt` of the clock's (a change of setting glides by the clock, so that it is
    /// seen through even while the scene's time stands still).
    fn advance(&mut self, dt: f32, real_dt: f32) {
        self.glide.step(real_dt as f64);
        if self.play_when_there && self.glide.arrived() {
            (self.play_when_there, self.playing) = (false, true);
        }
        if self.playing {
            let t = (self.glide.to.t + dt.max(0.0) as f64 / PLAY_SECONDS).min(1.0);
            self.glide.now(|s| s.t = t);
            if t >= 1.0 {
                self.playing = false;
            }
        }
        if let Some(left) = self.counting {
            if self.glide.arrived() {
                let left = left - dt.max(0.0);
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
            let k = 1.0 - (-(real_dt.max(0.0) as f64) / 0.12).exp();
            self.terms_shown += (to - self.terms_shown) * k;
            if (to - self.terms_shown).abs() < 1e-3 || !self.terms_shown.is_finite() {
                self.terms_shown = to;
            }
        }
        if let Some(n) = axis_of(self.glide.to.w) {
            self.axis = n;
        }
    }

    /// The rotation drawn now, e^(tK).
    fn rotation(&self) -> DMat3 {
        let s = self.shown();
        exp_spin(s.w * s.t)
    }

    /// The angle turned by now, θ = t|w| (radians).
    fn turned(&self) -> f64 {
        let s = self.shown();
        s.t * s.w.length()
    }

    /// The view that fits the picture into its round part of the canvas (`stage`).
    fn the_view(&self) -> View {
        let lay = Layout::of(self.canvas, self.inset, PANEL_H);
        let (right, up) = screen_axes();
        let (centre, radius) = stage(self.canvas, self.inset, &lay);
        let reach = if lay.narrow { REACH_NARROW } else { REACH_DRAWN };
        let per_unit = ((radius - LABEL_ROOM) / reach).max(1.0);
        let dist = (self.canvas.y * 0.5 / per_unit / TAN_HALF_FOV).clamp(2.0, 60.0);
        let per_unit = self.canvas.y * 0.5 / (dist * TAN_HALF_FOV);
        // move the middle of the picture to the middle of its part of the canvas
        let shift = (centre - self.canvas * 0.5) / per_unit;
        View::Orbit { target: -right * shift.x + up * shift.y, yaw: YAW, pitch: PITCH, dist }
    }

    /// The series' sum as drawn: the terms so far end to end, the last growing in.
    fn series_drawn(&self) -> Vec<DVec3> {
        let s = self.shown();
        let parts = terms(s.w, s.t, s.p, N_MAX);
        let shown = self.terms_shown.clamp(0.0, N_MAX as f64);
        let mut out = vec![s.p];
        let mut end = s.p;
        for (k, term) in parts.iter().enumerate().skip(1) {
            let share = (shown - (k - 1) as f64).clamp(0.0, 1.0);
            if share <= 0.0 {
                break;
            }
            end += *term * share;
            out.push(end);
        }
        out
    }

    /// The lines of the panel under its matrices, each with its colour.
    fn panel_lines(&self) -> Vec<(String, Color)> {
        let s = self.shown();
        let theta = self.turned();
        let mut lines = Vec::new();
        match axis_of(s.w) {
            Some(n) => lines.push((format!(r"$\theta = t\,|w| = {}$ about $n = {}$", deg_tex(theta.to_degrees()), coords(n, 3)), SPIN)),
            None => lines.push(("no spin, $w = 0$: nothing turns, $R = I$".to_string(), SPIN)),
        }
        match self.mode {
            Mode::Turn => {
                let r = self.rotation();
                lines.push((format!(r"$\tr R = {} = 1 + 2\cos\theta$", num(trace(r), 3)), pal::DIM));
            }
            Mode::Series => {
                let exact = self.rotation() * s.p;
                let off = (partial_sum(s.w, s.t, self.terms) * s.p - exact).length();
                lines.push((format!(r"sum to $N = {}$ ends ${}$ from $e^{{tK}} p$", self.terms, distance(off)), SERIES));
            }
            Mode::Pieces => {
                lines.push((r"$e^{\theta N} = I + \sin\theta\, N + (1 - \cos\theta)\, N^2$".to_string(), pal::TEXT));
                lines.push((format!(r"${} = {}$, ${} = {}$", tint(SPEED, r"\sin\theta"), num(theta.sin(), 3), tint(ACROSS, r"1 - \cos\theta"), num(1.0 - theta.cos(), 3)), pal::TEXT));
            }
            Mode::Sum => {
                let (w1, w2) = (s.w1(), s.w2);
                let both = exp_spin(s.w);
                let made = one_then_other(w1, w2, 1.0);
                let one = log_rotation(both, s.w);
                let other = log_rotation(made, DVec3::ONE);
                let say = |spin: Spin| match spin.axis {
                    Some(n) => format!("{} about ${}$", deg(spin.theta.to_degrees()), coords(n, 3)),
                    None => "no turn".to_string(),
                };
                lines.push((format!("$e^{{K_1 + K_2}}$: {}", say(one)), SPIN));
                lines.push((format!("$e^{{K_2}} e^{{K_1}}$: {}", say(other)), OTHER));
                lines.push((format!("at $t = 1$ they are {} apart", deg(apart(both, made))), GAP));
            }
            Mode::Log => lines.extend(self.log_lines()),
        }
        lines
    }

    /// The rotation the spin is read back from, in the last way of showing it: the turn
    /// of the whole spin asked for.
    fn given(&self) -> DMat3 {
        exp_spin(self.glide.to.w)
    }

    /// How the spin is read back from that rotation, line by line.
    fn log_lines(&self) -> Vec<(String, Color)> {
        let r = self.given();
        let back = log_rotation(r, self.axis);
        let mut lines = vec![(format!(r"$\tr R = {} = 1 + 2\cos\theta$: $\theta = {}$", num(trace(r), 3), deg_tex(back.theta.to_degrees())), pal::TEXT)];
        match back.axis {
            None => lines.push((r"$R = I$: no turn, so no axis".to_string(), pal::TEXT)),
            Some(n) if back.half => lines.push((format!(r"$R - R^\top = 0$, and $R + I = 2nn^\top$: $n = {}$", coords(n, 3)), pal::TEXT)),
            Some(n) => lines.push((format!(r"$(R - R^\top)/2 = \sin\theta\, N$: $n = {}$", coords(n, 3)), pal::TEXT)),
        }
        lines
    }

    /// How tall the panel is, for what it says now.
    fn panel_height(&self) -> f32 {
        let p = parts((Vec2::ZERO, Vec2::new(PANEL_W, PANEL_H)));
        p.lines.y + self.panel_lines().len() as f32 * LINE + 8.0
    }
}

/// A distance for the screen: three decimals, or two figures with a power of ten when
/// it is smaller than that (and 0 only when it is 0).
fn distance(d: f64) -> String {
    if d == 0.0 || d >= 0.0005 || !d.is_finite() {
        num(d, 3)
    } else {
        format!("{d:.1e}")
    }
}

/// The label of one of egui's own widgets (a button), typeset.
fn typeset_label(ui: &egui::Ui, text: &str) -> egui::WidgetText {
    crate::typeset::seen(text);
    ui::label_text(ui, text, 12.5, None, false)
}

impl Scene for ExpSkew {
    fn view(&self) -> View {
        self.the_view()
    }

    fn take_view(&mut self) -> Option<View> {
        std::mem::take(&mut self.new_view).then(|| self.the_view())
    }

    fn animated(&self) -> bool {
        self.playing || self.play_when_there || !self.glide.arrived() || self.counting.is_some() || self.terms_shown != self.terms as f64
    }

    fn settings(&self, s: &mut ViewSettings) {
        s.grid_alpha = 0.06;
        // (the axes of space are drawn by the scene)
        s.axis_alpha = 0.0;
    }

    fn ui(&mut self, ui: &mut egui::Ui, _: &Gfx) {
        ui::note(ui, r"A skew-symmetric matrix $K$ ($K^\top = -K$) in space is a spin: $Kx = w \times x$ for one arrow $w$, and that is the velocity of the point $x$ when all of space turns about the line of $w$ at the rate $|w|$, in radians per unit of time. Follow those velocities for a time $t$ and space has turned by the angle $t|w|$: that turn is the matrix $R = e^{tK}$, a rotation. Drag the white dot at the tip of the orange arrow $w$ to change the spin, and the white dot $p$ to choose the point that is followed.");

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
                self.toggle_play();
            }
            if ui.button(typeset_label(ui, "back to $t = 0$")).clicked() {
                self.glide.aim(|s| s.t = 0.0);
                (self.playing, self.play_when_there) = (false, false);
            }
        });
        let mut t = self.glide.to.t as f32;
        if ui::slider(ui, "time $t$", &mut t, 0.0..=1.0) {
            self.glide.now(|s| s.t = t as f64);
            (self.playing, self.play_when_there) = (false, false);
        }
        ui::note(ui, r"How much of the turn is done: by the time $t$ space has turned by $t|w|$. Play runs $t$ from 0 to 1 in 2.5 seconds; the track on the picture does the same as this slider.");

        ui::section(ui, "The spin $w$");
        if self.mode == Mode::Sum {
            ui::note(ui, r"Here the spin is $w = w_1 + w_2$: drag the white dots at the tips of the two thin arrows (each up to a quarter turn).");
        } else {
            let mut degrees = self.glide.to.w.length().to_degrees() as f32;
            if ui::slider(ui, "angle of the whole turn, $|w|$ (degrees)", &mut degrees, 0.0..=180.0) {
                let w = self.axis * (degrees.max(0.0).min(180.0) as f64).to_radians();
                self.glide.now(|s| s.w = w);
            }
            let names = ["about $x$", "about $y$", "about $z$", "about $(1, 1, 1)$"];
            if let Some(i) = ui::buttons(ui, &names) {
                let n = [DVec3::X, DVec3::Y, DVec3::Z, DVec3::ONE.normalize()][i];
                let len = self.glide.to.w.length();
                let len = if len > 1e-9 { len } else { FRAC_PI_2 };
                self.glide.aim(|s| s.w = n * len);
            }
            ui::note(ui, r"The direction of $w$ is the axis, by the right-hand rule (thumb along $w$, fingers the way space turns); its length is the angle of the whole turn, in radians. Drag its tip anywhere.");
        }

        ui::section(ui, "Show");
        let mut k = MODES.iter().position(|(m, _)| *m == self.mode).unwrap_or(0);
        let labels: Vec<&str> = MODES.iter().map(|(_, l)| *l).collect();
        if ui::choice(ui, &labels, &mut k) {
            self.set_mode(MODES[k].0);
            self.counting = None;
        }
        if self.mode == Mode::Series {
            if ui::slider_int(ui, "$N$, the last power", &mut self.terms, 0..=N_MAX) {
                self.counting = None;
            }
            ui::note(ui, r"The red arrows are the terms $p, tKp, (tK)^2 p/2!, \ldots$ laid end to end: they end at the sum up to the $N$th.");
        }
        ui::checkbox(ui, &mut self.field, "the velocities on a sphere");

        ui::section(ui, "Reading the picture");
        let mut key: Vec<(Color, &str)> = vec![
            (COLS[0], "pink, green and blue arrows $Re_1, Re_2, Re_3$: where one step along $x$, $y$ and $z$ has turned to, the columns of $R = e^{tK}$"),
            (CUBE, r"yellow: a cube of side 1 about $0$, turned by $R$; dashed grey: where it was at $t = 0$"),
            (SPIN, r"orange arrow $w$: the spin (drag the white dot at its tip). Its line (dashed) is the axis, where nothing moves; its length is the angle of the whole turn"),
            (SPEED, r"cyan arrows: the velocity $w \times x$ at points of a sphere, and of the moving point, each a sixth of its length"),
            (POINTS, r"white: the point $p$ (drag it), the circle it goes round, the arc it has gone along, and where it is now, $e^{tK} p$"),
        ];
        match self.mode {
            Mode::Series => key.push((SERIES, r"red: the terms $(tK)^k p / k!$ of the series, end to end, and their sum")),
            Mode::Pieces => {
                key.push((ACROSS, r"purple step: $(1 - \cos\theta)\, N^2 p$, from $p$ straight in at the axis"));
                key.push((SPEED, r"cyan step: $\sin\theta\, Np$, at right angles to it, along $n \times p$"));
            }
            Mode::Sum => {
                key.push((SPIN, "thin orange arrows $w_1$ and $w_2$ (drag their dots), added to make $w$"));
                key.push((OTHER, r"blue: $e^{K_2} e^{K_1}$, the turn of $w_1$ and then that of $w_2$, and the way it takes $p$"));
                key.push((GAP, "red: how far apart the two take $p$"));
            }
            Mode::Log => key.push((POINTS, "white dashed outline: the rotation $R$ the spin is read back from")),
            Mode::Turn => {}
        }
        ui::key(ui, &key);

        ui::section(ui, "Words");
        ui::term(ui, "Skew-symmetric:", r"a square matrix with $K^\top = -K$: zeros down the diagonal, and each entry minus the one across the diagonal from it. In space it is made of three numbers $w = (a, b, c)$, and $Kx = w \times x$.");
        ui::term(ui, "Spin:", r"all of space turning about one line through $0$ at a steady rate. The velocity of each point is $w \times x$: at right angles to the axis and to the point's arrow, and $|w|$ times its distance from the axis.");
        ui::term(ui, "Right-hand rule:", "with the right thumb along $w$, the fingers curl the way space turns.");
        ui::term(ui, "$e^{tK}$:", r"$I + tK + (tK)^2/2! + \cdots$: where following the velocities $Kx$ for the time $t$ takes each point. Its transpose is $e^{tK^\top} = e^{-tK}$, its inverse, and its determinant is $e^{t \tr K} = e^0 = 1$: it is a rotation.");
        ui::term(ui, "Rodrigues' formula:", r"$e^{\theta N} = I + \sin\theta\, N + (1 - \cos\theta)\, N^2$, for $N$ the spin of rate 1 about an axis $n$: true because $N^3 = -N$.");
        ui::term(ui, r"Orthogonal, $R^\top R = I$:", r"the columns have length 1 and are at right angles. With $\det R = 1$ too, $R$ is a rotation.");
        ui::term(ui, "Logarithm:", r"a spin whose exponential is a given rotation. Every rotation has one no longer than $\pi$, and only one but for a half turn, where $\pi n$ and $-\pi n$ both serve; longer spins along the same line, a whole turn $2\pi$ apart, give the same rotation too.");
        ui::term(ui, "In the plane:", r"the spin of rate 1 about $z$ turns the plane $z = 0$ as $J$ does there, and on it $e^{tK}$ is $e^{tJ} = \cos t\, I + \sin t\, J$.");

        ui::section(ui, "Numbers");
        let s = self.shown();
        let r = self.rotation();
        let theta = self.turned();
        ui::readout_colored(ui, r"$|w|$, the angle of the whole turn", format!("{} rad ({})", num(s.w.length(), 3), deg(s.w.length().to_degrees())), SPIN);
        ui::readout_colored(ui, r"its axis $n = w/|w|$", axis_of(s.w).map_or("none: no spin".to_string(), |n| format!("${}$", coords(n, 3))), SPIN);
        ui::readout(ui, r"$t$, and $\theta = t|w|$ turned by then", format!("{}, {}", num(s.t, 2), deg(theta.to_degrees())));
        ui::readout(ui, r"$\det R$", num(r.determinant(), 6));
        let off = off_orthonormal(r);
        // (as a bound, which takes one of a few texts, rather than digits that change in
        // every frame of a play and would be set afresh in each)
        let off_text = match off {
            0.0 => "0".to_string(),
            v if v < 1e-15 => r"$< 10^{-15}$".to_string(),
            v if v < 1e-14 => r"$< 10^{-14}$".to_string(),
            v => format!("${v:.1e}$"),
        };
        ui::readout(ui, r"largest entry of $R^\top R - I$", off_text);
        ui::readout(ui, r"$\tr R$ and $1 + 2\cos\theta$", format!("{}, {}", num(trace(r), 3), num(1.0 + 2.0 * theta.cos(), 3)));
        match self.mode {
            Mode::Series => {
                let off = (partial_sum(s.w, s.t, self.terms) * s.p - r * s.p).length();
                ui::readout_colored(ui, &format!("sum to $N = {}$, away from $e^{{tK}} p$", self.terms), format!("${}$", distance(off)), SERIES);
            }
            Mode::Sum => {
                let gap = apart(exp_spin(s.w), one_then_other(s.w1(), s.w2, 1.0));
                ui::readout_colored(ui, r"$e^{K_1 + K_2}$ and $e^{K_2} e^{K_1}$ apart by", deg(gap), GAP);
            }
            Mode::Log => {
                let back = log_rotation(self.given(), self.axis);
                ui::readout(ui, r"read back: $\theta$", deg(back.theta.to_degrees()));
            }
            _ => {}
        }
        ui::see_also(ui, &[("expm", "the same in the plane: flows and their exponentials"), ("rotations", "rotations, and the one axis every one has"), ("symskew", "every matrix a symmetric part plus a spin"), ("expsum", r"$e^{A + B}$ against $e^A e^B$")]);
    }

    fn frame(&mut self, f: &mut Frame) {
        // the panel's place follows the canvas, and the picture's place follows the panel's
        let view = f.view_size();
        let lay = Layout::of(view, f.top_inset, self.panel_height());
        let before = Layout::of(self.canvas, self.inset, PANEL_H);
        // (a canvas squeezed to a sliver is one the controls have opened over, on a
        // phone: the picture is not seen then, and is left as it was for when they close)
        let covered = view.x < 160.0;
        let changed = !covered && (lay.narrow != before.narrow || lay.panel.is_some() != before.panel.is_some() || (view - self.canvas).abs().max_element() > 24.0);
        let first = !covered && !self.measured && (view - self.canvas).abs().max_element() > 24.0;
        self.new_view |= changed || first;
        if !covered && (changed || !self.measured) {
            self.canvas = view;
            self.measured = true;
        }
        if !covered {
            self.new_view |= (f.top_inset - self.inset).abs() > 1.0;
            self.inset = f.top_inset;
        }
        let pointer = f.ptr.pos.map(|p| p / f.px);
        if pointer.is_some_and(|p| lay.covers(p)) || self.scrub {
            // the panel is for its own track and button: no turning the view from there,
            // and two quick clicks are two clicks, not a reset of the view
            f.handles.capture();
            f.handles.claim_clicks();
        }
        if !covered {
            self.panel_input(f, &lay, pointer);
        }
        self.advance(f.dt, f.real_dt);
        if !covered {
            self.handles(f, &lay);
        }

        let labels_before = f.dl.labels.len();
        let mut names = std::mem::take(&mut self.names);
        self.picture(f, &lay, &mut names);
        self.names = names;
        if covered {
            // (the sliver left beside the controls is the picture alone: its names, the
            // panel and the caption would only be written over one another)
            f.dl.labels.truncate(labels_before);
            return;
        }
        self.panel(f, &lay, pointer);
        let bottom = if lay.narrow { lay.below - 4.0 } else { view.y - 18.0 };
        caption(f, bottom);
    }
}

impl ExpSkew {
    /// The time track's knob and the play button on the panel over the picture.
    fn panel_input(&mut self, f: &mut Frame, lay: &Layout, pointer: Option<Vec2>) {
        if !f.ptr.down {
            self.scrub = false;
        }
        let Some(rect) = lay.panel else {
            self.scrub = false;
            return;
        };
        let p = parts(rect);
        let Some(at) = pointer else { return };
        let (a, b) = p.track;
        if f.ptr.pressed && inside_rect(at, (a - Vec2::new(12.0, 13.0), b + Vec2::new(12.0, 13.0))) {
            self.scrub = true;
        }
        if self.scrub {
            let t = ((at.x - a.x) / (b.x - a.x).max(1.0)).clamp(0.0, 1.0) as f64;
            self.glide.now(|s| s.t = t);
            (self.playing, self.play_when_there) = (false, false);
        }
        if f.ptr.clicked && inside_rect(at, p.play) {
            self.toggle_play();
        }
    }

    /// The dots that can be dragged: the tip of the spin (or of each of two spins) and
    /// the marked point. (One under the panel is left alone unless it is being dragged.)
    fn handles(&mut self, f: &mut Frame, lay: &Layout) {
        let usable = |f: &Frame, p: Vec3| !on_screen(f, p).is_some_and(|q| lay.covers(q)) || f.handles.dragging();
        let s = self.shown();
        if self.mode == Mode::Sum {
            let mut tip = s.w1().as_vec3();
            if usable(f, tip) && f.drag3(1, &mut tip, None, POINTS) {
                let w1 = at_most(tip.as_dvec3(), W_PART_MAX);
                self.glide.now(|s| s.w = w1 + s.w2);
            }
            let mut tip = s.w2.as_vec3();
            if usable(f, tip) && f.drag3(3, &mut tip, None, POINTS) {
                let w2 = at_most(tip.as_dvec3(), W_PART_MAX);
                self.glide.now(|s| {
                    s.w = s.w1() + w2;
                    s.w2 = w2;
                });
            }
        } else {
            let mut tip = s.w.as_vec3();
            if usable(f, tip) && f.drag3(1, &mut tip, None, POINTS) {
                let w = at_most(tip.as_dvec3(), W_MAX);
                self.glide.now(|s| s.w = w);
            }
        }
        let mut p = s.p.as_vec3();
        if usable(f, p) && f.drag3(2, &mut p, None, POINTS) {
            let p = at_most(p.as_dvec3(), P_MAX);
            self.glide.now(|s| s.p = p);
        }
    }

    /// Everything in space, and the names written beside it.
    fn picture(&self, f: &mut Frame, lay: &Layout, names: &mut Still) {
        let mut room = Room::new(f);
        if let Some((min, max)) = lay.panel {
            room.wall(min - Vec2::splat(4.0), max + Vec2::splat(4.0));
        }
        let s = self.shown();
        let r = self.rotation();
        let m = r.as_mat3();
        let theta = self.turned();
        let n = axis_of(s.w);
        let outwards = |f: &Frame, from: Vec3, to: Vec3| match (on_screen(f, from), on_screen(f, to)) {
            (Some(o), Some(p)) if (p - o).length() > 1.0 => p - o,
            _ => Vec2::new(1.0, -1.0),
        };

        // the axes of space
        for k in 0..3 {
            let d = Vec3::AXES[k];
            f.line(-d * SPACE_LEN, d * SPACE_LEN, 1.0, pal::DIM.a(0.5));
        }
        // the cube as it was
        outline(f, Mat3::IDENTITY, 1.1, pal::DIM.a(0.75));

        // the velocities on the sphere
        let wf = s.w.as_vec3();
        if self.field {
            // (fainter on the far side of the sphere, which is seen through the near one)
            let eye = f.cam.eye;
            for x in &self.sphere {
                let facing = x.normalize_or_zero().dot((eye - *x).normalize_or_zero());
                arrow(f, Part::Extra, *x, *x + wf.cross(*x) * FIELD_SCALE, 1.2, SPEED.a(0.2 + 0.25 * (facing + 1.0)));
            }
        }

        // the axis and the spin, or the two spins and their sum
        if let Some(n) = n {
            let nf = n.as_vec3();
            f.dashed(-nf * AXIS_LEN, nf * AXIS_LEN, 1.4, SPIN.a(0.8));
            room.line(f, -nf * AXIS_LEN, nf * AXIS_LEN);
            arrow(f, Part::Axis, Vec3::ZERO, wf, 3.2, SPIN);
            room.line(f, Vec3::ZERO, wf);
        }
        let (w1, w2) = (s.w1().as_vec3(), s.w2.as_vec3());
        if self.mode == Mode::Sum {
            for part in [w1, w2] {
                arrow(f, Part::Extra, Vec3::ZERO, part, 2.2, SPIN.a(0.7));
                room.line(f, Vec3::ZERO, part);
            }
            f.dashed(w1, wf, 1.2, SPIN.a(0.6));
            f.dashed(w2, wf, 1.2, SPIN.a(0.6));
        }

        // the cube turned, and its columns
        cube(f, m);
        for (a, b) in EDGES {
            room.line(f, m * corner(a), m * corner(b));
        }
        let cols = [m.x_axis, m.y_axis, m.z_axis];
        for k in 0..3 {
            arrow(f, Part::Fixed, Vec3::ZERO, cols[k], 3.0, COLS[k]);
            room.line(f, Vec3::ZERO, cols[k]);
        }

        // the marked point: its circle about the axis, the arc it has turned along so
        // far, where it is, and its velocity there
        let now = r * s.p;
        let circle = n.and_then(|n| circle_of(n, s.p));
        if let Some((c, rr, ss)) = circle {
            let at = |phi: f64| (c + rr * phi.cos() + ss * phi.sin()).as_vec3();
            for i in 0..ARC_POINTS {
                let (a, b) = (i as f64 / ARC_POINTS as f64 * WHOLE_TURN, (i + 1) as f64 / ARC_POINTS as f64 * WHOLE_TURN);
                f.dashed(at(a), at(b), 1.1, POINTS.a(0.35));
            }
            let arc: Vec<Vec3> = (0..=ARC_POINTS).map(|i| at(theta * i as f64 / ARC_POINTS as f64)).collect();
            lines_for(f, Part::Point).polyline(arc.iter().copied(), 2.2, POINTS.a(0.9));
            for pair in arc.windows(8).step_by(7) {
                room.line(f, pair[0], pair[7]);
            }
            // the spokes from the axis to where the point started and to where it is
            let cf = c.as_vec3();
            lines_for(f, Part::Point).segment(cf, s.p.as_vec3(), 1.0, POINTS.a(0.4));
            lines_for(f, Part::Point).segment(cf, now.as_vec3(), 1.0, POINTS.a(0.4));
            room.line(f, cf, s.p.as_vec3());
            room.line(f, cf, now.as_vec3());
        }
        let nowf = now.as_vec3();
        let velocity = wf.cross(nowf) * FIELD_SCALE;
        arrow(f, Part::Fixed, nowf, nowf + velocity, 2.4, SPEED);
        room.line(f, nowf, nowf + velocity);
        f.dot(nowf, 10.0, POINTS);

        // the extras of each way of showing it
        let mut extra_names: Vec<(&'static str, Vec<Vec3>, String, Color, Vec2)> = Vec::new();
        match self.mode {
            Mode::Series => {
                let sums = self.series_drawn();
                for pair in sums.windows(2) {
                    let (a, b) = (pair[0].as_vec3(), pair[1].as_vec3());
                    arrow(f, Part::Series, a, b, 2.0, SERIES);
                    room.line(f, a, b);
                }
                let end = sums.last().copied().unwrap_or(s.p).as_vec3();
                f.dl.points_with(PointKind::Disc, PointStyle { outline_px: 1.25 + 1e-4, ..Default::default() }).push(end, 8.0, SERIES);
                let apart = on_screen(f, end).zip(on_screen(f, nowf)).is_some_and(|(p, q)| p.distance(q) > 24.0);
                if self.terms > 0 && apart {
                    extra_names.push(("sum", vec![end], format!("sum to $N = {}$", self.terms), SERIES, outwards(f, nowf, end)));
                }
            }
            Mode::Pieces => {
                // first in along the spoke, straight at the axis, which leaves cos θ of
                // the part across it; then across, along n × p
                let np = s.w.cross(s.p) / s.w.length().max(1e-12);
                let inward = n.map_or(DVec3::ZERO, |n| n.cross(n.cross(s.p))) * (1.0 - theta.cos());
                let (p0, p1) = (s.p.as_vec3(), (s.p + inward).as_vec3());
                let p2 = (s.p + inward + np * theta.sin()).as_vec3();
                arrow(f, Part::Extra, p0, p1, 2.6, ACROSS);
                arrow(f, Part::Extra, p1, p2, 2.6, SPEED);
                room.line(f, p0, p1);
                room.line(f, p1, p2);
                // (named near p: further along, the step runs through the cube)
                let near_p = [0.25, 0.15, 0.4].map(|u| p0.lerp(p1, u)).to_vec();
                extra_names.push(("cos", near_p, r"$(1 - \cos\theta)\, N^2 p$".to_string(), ACROSS, outwards(f, p2, p0.lerp(p1, 0.25))));
                extra_names.push(("sin", vec![(p1 + p2) * 0.5], r"$\sin\theta\, Np$".to_string(), SPEED, outwards(f, p0, (p1 + p2) * 0.5)));
            }
            Mode::Sum => {
                let ghost = one_then_other(s.w1(), s.w2, s.t);
                outline(f, ghost.as_mat3(), 1.6, OTHER.a(0.95));
                let gm = ghost.as_mat3();
                for (a, b) in EDGES {
                    room.line(f, gm * corner(a), gm * corner(b));
                }
                // the way it takes the point: all of it faint, and solid as far as it has got
                let path: Vec<Vec3> = (0..=ARC_POINTS).map(|i| (one_then_other(s.w1(), s.w2, i as f64 / ARC_POINTS as f64) * s.p).as_vec3()).collect();
                for pair in path.windows(2) {
                    f.dashed(pair[0], pair[1], 1.1, OTHER.a(0.5));
                }
                let upto: Vec<Vec3> = (0..=ARC_POINTS).map(|i| (one_then_other(s.w1(), s.w2, s.t * i as f64 / ARC_POINTS as f64) * s.p).as_vec3()).collect();
                lines_for(f, Part::Extra).polyline(upto.iter().copied(), 2.0, OTHER.a(0.9));
                let there = (ghost * s.p).as_vec3();
                f.dot(there, 9.0, OTHER);
                lines_for(f, Part::Extra).segment(nowf, there, 1.8, GAP);
                room.line(f, nowf, there);
                extra_names.push(("other", vec![there], r"$e^{K_2} e^{K_1} p$".to_string(), OTHER, outwards(f, nowf, there)));
            }
            Mode::Log => {
                let given = self.given().as_mat3();
                outline(f, given, 1.8, POINTS.a(0.85));
                for (a, b) in EDGES {
                    room.line(f, given * corner(a), given * corner(b));
                }
                // (named at the corner furthest out on the screen, beyond the cube)
                let o = on_screen(f, Vec3::ZERO).unwrap_or(Vec2::ZERO);
                let out = |p: Vec3| on_screen(f, p).map_or(0.0, |q| (q - o).length());
                let far = (0..8).map(|i| given * corner(i)).fold(None, |best: Option<Vec3>, p| if best.is_none_or(|b| out(p) > out(b) + 0.5) { Some(p) } else { best });
                if let Some(far) = far {
                    extra_names.push(("given", vec![far], "$R$, given".to_string(), POINTS, outwards(f, Vec3::ZERO, far)));
                }
            }
            Mode::Turn => {}
        }

        // the names, the most important first; written as they were in the last frame
        // if nothing they are placed from has changed since and they had come to rest
        self.inputs(names.begin(f));
        if names.settle(f) {
            names.replay(f, &mut room);
            return;
        }
        // (smaller on a phone, where the picture is small; the names of what stays put
        // first, so that what moves keeps clear of them rather than they of it, and the
        // angle, which the panel gives too, only where it has room)
        let size = |s: f32| if lay.narrow { s * 0.88 } else { s };
        for k in 0..3 {
            let d = Vec3::AXES[k] * SPACE_LEN;
            names.label(&mut room, f, ("space", k), d, ["$x$", "$y$", "$z$"][k], size(12.5), pal::DIM, outwards(f, Vec3::ZERO, d), 5.0, false);
        }
        if let Some(n) = n {
            let end = -n.as_vec3() * AXIS_LEN;
            names.label(&mut room, f, "axis", end, "axis", size(12.5), SPIN, outwards(f, Vec3::ZERO, end), 6.0, false);
            let text = if self.mode == Mode::Sum { "$w = w_1 + w_2$" } else { "$w$" };
            names.label(&mut room, f, "w", wf, text, size(15.0), SPIN, outwards(f, Vec3::ZERO, wf), 10.0, false);
        } else {
            names.label(&mut room, f, "w", wf, "no spin: $w = 0$", size(13.0), SPIN, Vec2::new(1.0, -1.0), 12.0, false);
        }
        if self.mode == Mode::Sum {
            names.label(&mut room, f, "w1", w1, "$w_1$", size(14.0), SPIN, outwards(f, Vec3::ZERO, w1), 8.0, false);
            names.label(&mut room, f, "w2", w2, "$w_2$", size(14.0), SPIN, outwards(f, Vec3::ZERO, w2), 8.0, false);
        }
        let pf = s.p.as_vec3();
        let still = circle.is_none();
        let apart_px = on_screen(f, pf).zip(on_screen(f, nowf)).map_or(0.0, |(a, b)| a.distance(b));
        if apart_px > 20.0 {
            names.label(&mut room, f, "p", pf, "$p$", size(15.0), POINTS, outwards(f, Vec3::ZERO, pf), 9.0, false);
        }
        // (where the point is at the tip of a column, the two share a name)
        let shared = (0..3).find(|&k| (cols[k] - nowf).length() < 1e-3);
        let text = match shared {
            _ if still && n.is_some() => "$p$ is on the axis: it stays".to_string(),
            Some(k) if apart_px > 20.0 => format!("$e^{{tK}} p = Re_{}$", k + 1),
            Some(k) => format!("$p = Re_{}$", k + 1),
            None if apart_px > 20.0 => "$e^{tK} p$".to_string(),
            None => "$p$".to_string(),
        };
        names.label(&mut room, f, "now", nowf, &text, size(14.0), POINTS, outwards(f, Vec3::ZERO, nowf), 9.0, false);
        for (name, at, text, color, towards) in extra_names {
            names.label_any(&mut room, f, name, &at, &text, size(13.0), color, towards, 8.0, false);
        }
        for k in (0..3).filter(|k| shared != Some(*k)) {
            let dir = outwards(f, Vec3::ZERO, cols[k]);
            names.label_any(&mut room, f, ("column", k), &[cols[k], cols[k] * 0.8, cols[k] * 0.6], &format!("$Re_{}$", k + 1), size(14.0), COLS[k], dir, 6.0, false);
        }
        if let (Some((c, rr, ss)), true) = (circle, theta > 0.15) {
            // (beside the arc on its outer side, about half-way along it)
            let out = |share: f64| (c + (rr * (share * theta).cos() + ss * (share * theta).sin()) * 1.12).as_vec3();
            let points = [out(0.5), out(0.38), out(0.62), out(0.26), out(0.74)];
            let dir = outwards(f, c.as_vec3(), points[0]);
            names.label_any(&mut room, f, "theta", &points, &format!(r"$\theta = {}$", deg_tex(theta.to_degrees())), size(13.0), POINTS, dir, 4.0, true);
        }
        names.end(f, false);
    }

    /// Everything the picture and its names are drawn from (see `Still`).
    fn inputs(&self, k: &mut Inputs) {
        k.f64s(&self.shown().numbers()).f64s(&[self.terms_shown]).int(self.terms as u64).int(self.mode as u64).int(self.field as u64);
        k.f32s(&[self.canvas.x, self.canvas.y, self.inset]).int(self.measured as u64);
    }

    /// The panel over the picture: the time, the two matrices and what they say.
    fn panel(&self, f: &mut Frame, lay: &Layout, pointer: Option<Vec2>) {
        let Some((min, max)) = lay.panel else { return };
        hud_panel(f, min, max);
        let p = parts((min, max));
        let s = self.shown();

        // the play button and the time track
        let busy = self.playing || self.play_when_there;
        hud_button(f, p.play, if busy { "❚❚" } else { "▶" }, pointer.is_some_and(|at| inside_rect(at, p.play)));
        let (a, b) = p.track;
        f.text_px(a - Vec2::new(14.0, 0.0), "$t$", 14.0, pal::TEXT, Align2::CENTER_CENTER);
        hud_line(f, a, b, 2.0, pal::DIM);
        for (v, name) in [(0.0f32, "$0$"), (1.0, "$1$")] {
            let x = a.x + (b.x - a.x) * v;
            hud_line(f, Vec2::new(x, a.y - 4.0), Vec2::new(x, a.y + 4.0), 1.2, pal::DIM);
            f.text_px(Vec2::new(x, a.y + 6.0), name, 10.5, pal::DIM, Align2::CENTER_TOP);
        }
        let knob = a + (b - a) * (s.t as f32).clamp(0.0, 1.0);
        hud_line(f, a, knob, 3.0, POINTS.a(0.8));
        f.dl.overlay_discs.push(f.hud_at(2, knob), 12.0, POINTS);
        f.text_px(p.time, format!("$t = {}$", num(s.t, 2)), 13.0, pal::TEXT, Align2::RIGHT_CENTER);

        // K, in the spin's colour, and what it says
        let k = skew(s.w);
        let entries = |m: DMat3, decimals: usize, color: &dyn Fn(usize) -> Color| -> Vec<Vec<(String, Color)>> {
            (0..3).map(|i| (0..3).map(|j| (format!("${}$", num(m.col(j)[i], decimals)), color(j))).collect()).collect()
        };
        let total = hud_matrix(f, p.k_at, &entries(k, 2, &|_| SPIN), 6, MAT, pal::DIM);
        f.text_px(Vec2::new(p.k_at.x - 6.0, p.k_at.y + total.y * 0.5), "$K =$", 14.0, pal::TEXT, Align2::RIGHT_CENTER);
        let right = p.k_at.x + total.x + 14.0;
        let small = 12.0;
        f.text_px(Vec2::new(right, p.k_at.y + 6.0), r"$K^\top = -K$", small, pal::TEXT, Align2::LEFT_TOP);
        f.text_px(Vec2::new(right, p.k_at.y + 6.0 + LINE), r"$Kx = w \times x$", small, SPIN, Align2::LEFT_TOP);
        f.text_px(Vec2::new(right, p.k_at.y + 6.0 + 2.0 * LINE), format!("$|w| = {}$", num(s.w.length(), 2)), small, SPIN, Align2::LEFT_TOP);

        // R = e^(tK), its columns in the arrows' colours, and the checks
        for j in 0..3 {
            let (lo, hi) = matrix_entry(p.r_at, 7, MAT, 0, j);
            f.text_px(Vec2::new((lo.x + hi.x) * 0.5, p.r_at.y - 3.0), format!("$Re_{}$", j + 1), 12.0, COLS[j], Align2::CENTER_BOTTOM);
        }
        let r = self.rotation();
        let total = hud_matrix(f, p.r_at, &entries(r, 3, &|j| COLS[j]), 7, MAT, pal::DIM);
        f.text_px(Vec2::new(p.r_at.x - 6.0, p.r_at.y + total.y * 0.5), "$R =$", 14.0, pal::TEXT, Align2::RIGHT_CENTER);
        f.text_px(Vec2::new(p.r_at.x - 6.0, p.r_at.y + total.y * 0.5 + 17.0), "$e^{tK}$", 11.5, pal::DIM, Align2::RIGHT_CENTER);
        let right = p.r_at.x + total.x + 14.0;
        let ok = off_orthonormal(r) < 1e-14;
        f.text_px(Vec2::new(right, p.r_at.y + 6.0), if ok { r"$R^\top R = I$" } else { r"$R^\top R \ne I$" }, small, pal::TEXT, Align2::LEFT_TOP);
        f.text_px(Vec2::new(right, p.r_at.y + 6.0 + LINE), r"to within $10^{-14}$", small, pal::DIM, Align2::LEFT_TOP);
        f.text_px(Vec2::new(right, p.r_at.y + 6.0 + 2.0 * LINE), format!(r"$\det R = {}$", num(r.determinant(), 3)), small, pal::TEXT, Align2::LEFT_TOP);

        // what the matrices say
        let mut y = p.lines.y;
        for (text, color) in self.panel_lines() {
            f.text_px(Vec2::new(p.lines.x, y), text, small, color, Align2::LEFT_TOP);
            y += LINE;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The largest entry of a matrix, in size.
    fn size(m: DMat3) -> f64 {
        m.to_cols_array().iter().fold(0.0, |w, v| w.max(v.abs()))
    }

    fn close(a: DMat3, b: DMat3) -> f64 {
        size(a - b)
    }

    fn random_unit(rng: &mut Rng) -> DVec3 {
        loop {
            let v = DVec3::new(rng.normal() as f64, rng.normal() as f64, rng.normal() as f64);
            if v.length() > 1e-3 {
                return v.normalize();
            }
        }
    }

    /// Spins of every kind: random axes with random lengths up to half a turn, the axes
    /// of space and the diagonals, and lengths of none, tiny ones, and half a turn and a
    /// hair less.
    fn spins() -> Vec<DVec3> {
        let mut rng = Rng::new(7);
        let mut axes = vec![DVec3::X, DVec3::Y, DVec3::Z, DVec3::ONE.normalize(), DVec3::new(1.0, -1.0, 1e-9).normalize(), DVec3::new(-0.2, 0.3, -1.0).normalize()];
        axes.extend((0..120).map(|_| random_unit(&mut rng)));
        let mut out = Vec::new();
        for (i, n) in axes.iter().enumerate() {
            for len in [0.0, 1e-12, 1e-7, 1e-3, 0.4, 1.0, FRAC_PI_2, 2.5, W_MAX - 1e-7, W_MAX] {
                out.push(*n * len);
            }
            out.push(*n * rng.range(0.0, W_MAX as f32) as f64 * if i % 2 == 0 { 1.0 } else { -1.0 });
        }
        out.extend([diagonal_spin(), opening().w, spin_pair().0 + spin_pair().1]);
        out
    }

    /// Rodrigues' formula is the series, for many axes and angles: summed until its
    /// terms are nothing, the series gives the same matrix, up to half a turn and on to
    /// a whole one (t up to 2); and the three-term form is the same again. N³ = −N,
    /// which is why.
    #[test]
    fn rodrigues_formula_is_the_series() {
        let mut checked = 0;
        for w in spins() {
            for t in [0.0, 0.25, 0.5, 1.0, 1.5, 2.0, -0.7] {
                let series = partial_sum(w, t, 60);
                let exact = exp_spin(w * t);
                assert!(close(exact, series) < 1e-13 * size(series).max(1.0), "{w} at {t}: {exact} against {series}");
                checked += 1;
            }
            if let Some(n) = axis_of(w) {
                let nn = skew(n);
                assert!(close(nn * nn * nn, -nn) < 1e-15, "{n}");
                let k = skew(w);
                assert!(close(k * k * k, -k * w.length_squared()) < 1e-14 * w.length().powi(3).max(1e-300) + 1e-300);
                let theta = w.length();
                let three = DMat3::IDENTITY + nn * theta.sin() + nn * nn * (1.0 - theta.cos());
                assert!(close(three, exp_spin(w)) < 1e-15, "{w}");
            }
        }
        assert!(checked > 5000, "{checked}");
        // and against glam's own turn about an axis, made another way
        let mut rng = Rng::new(31);
        for _ in 0..300 {
            let (n, a) = (random_unit(&mut rng), rng.range(-3.1, 3.1) as f64);
            assert!(close(exp_spin(n * a), DMat3::from_axis_angle(n, a)) < 1e-14);
        }
    }

    /// e^K is a rotation: its transpose is e^(Kᵀ) = e^(−K), its inverse; it keeps
    /// lengths; its determinant is e^(tr K) = 1. Following the spin for s and then for
    /// t is following it for s + t. And the panel's claim: RᵀR = I to within 10⁻¹⁴
    /// (the rounding of the last digits of numbers near 1 is a few times 10⁻¹⁶).
    #[test]
    fn every_turn_of_a_spin_is_a_rotation() {
        let mut worst = 0.0f64;
        for w in spins() {
            let k = skew(w);
            assert!(close(k.transpose(), -k) == 0.0 && trace(k) == 0.0);
            for t in [0.0, 0.1, 0.33, 0.5, 0.77, 1.0] {
                let r = exp_spin(w * t);
                worst = worst.max(off_orthonormal(r));
                assert!((r.determinant() - 1.0).abs() < 1e-14 && (r.determinant() - (t * trace(k)).exp()).abs() < 1e-14);
                assert!(close(r.transpose(), exp_spin(-w * t)) < 1e-15 && close(r.transpose(), exp_spin(vee(k.transpose()) * t)) < 1e-15);
                assert!(close(exp_spin(w * t) * exp_spin(w * 0.4), exp_spin(w * (t + 0.4))) < 1e-14);
            }
        }
        assert!(worst < 5e-15, "the panel says to within 1e-14: {worst}");
    }

    /// Which way a positive angle turns: by the right-hand rule, as the scene "vectors"
    /// has it. A spin along z moves the point on x towards y (the velocity w × x is +y),
    /// and a quarter turn about z takes x to y, y to −x; about x, y to z; about y, z to x.
    #[test]
    fn a_positive_spin_turns_by_the_right_hand_rule() {
        assert_eq!(DVec3::Z.cross(DVec3::X), DVec3::Y);
        assert_eq!(skew(DVec3::Z) * DVec3::X, DVec3::Y);
        let quarter = |n: DVec3| exp_spin(n * FRAC_PI_2);
        assert!((quarter(DVec3::Z) * DVec3::X - DVec3::Y).length() < 1e-15 && (quarter(DVec3::Z) * DVec3::Y + DVec3::X).length() < 1e-15);
        assert!((quarter(DVec3::X) * DVec3::Y - DVec3::Z).length() < 1e-15 && (quarter(DVec3::Y) * DVec3::Z - DVec3::X).length() < 1e-15);
        // a little of a spin about z: x moves into positive y, and the turn agrees with
        // the turns of the scene "rotations", made directly
        assert!((exp_spin(DVec3::Z * 0.1) * DVec3::X).y > 0.0);
        for k in 0..3 {
            for degrees in [10.0, 90.0, -45.0, 180.0] {
                assert!(close(exp_spin(DVec3::AXES[k] * f64::to_radians(degrees)), about(k, degrees)) < 1e-15);
            }
        }
    }

    /// From a rotation back to its spin: the spin comes back exactly for every spin up to
    /// half a turn, also near no turn and near a half turn, where R − Rᵀ is small; at a
    /// half turn the axis comes from R + I, either way along it (the way preferred), and
    /// it says so; no turn has no axis; what is not a number has none either.
    #[test]
    fn the_logarithm_reads_back_the_spin() {
        for w in spins() {
            let r = exp_spin(w);
            let back = log_rotation(r, w);
            let len = w.length();
            match back.axis {
                None => assert!(len < 1e-12, "{w}: {back:?}"),
                Some(n) => {
                    assert!((back.theta - len).abs() < 1e-12, "{w}: {back:?}");
                    assert!((n * back.theta - w).length() < 1e-9, "{w}: {back:?}");
                    assert!(close(exp_spin(n * back.theta), r) < 1e-14);
                    // (a half turn is said to be one only within a hair of it)
                    assert!(!back.half || W_MAX - back.theta < 1e-12, "{w}: {back:?}");
                }
            }
            // read the other way at a half turn, if that is preferred
            if (len - W_MAX).abs() < 1e-14 {
                let other = log_rotation(r, -w);
                let n = other.axis.expect("a half turn has an axis");
                assert!(other.half && (n * W_MAX + w).length() < 1e-9 && close(exp_spin(n * W_MAX), r) < 1e-14, "{w}: {other:?}");
                // and R + I = 2 n nᵀ there
                let nnt = DMat3::from_cols(n * n.x, n * n.y, n * n.z);
                assert!(close(r + DMat3::IDENTITY, nnt * 2.0) < 1e-14);
            }
        }
        assert!(log_rotation(DMat3::IDENTITY, DVec3::X).axis.is_none());
        // three half turns about x, y, z are no turn at all
        let none = log_rotation(about(2, 180.0) * about(1, 180.0) * about(0, 180.0), DVec3::X);
        assert!(none.axis.is_none() || none.theta < 1e-12, "{none:?}");
        let bad = DMat3::from_cols(DVec3::splat(f64::NAN), DVec3::Y, DVec3::Z);
        assert!(log_rotation(bad, DVec3::X).axis.is_none() && log_rotation(bad, DVec3::X).theta == 0.0);
    }

    /// In the plane the same thing is the scene "expm"'s: a spin of rate 1 about z turns
    /// the plane z = 0 as J does, and there e^(tK) is e^(tJ) = cos t I + sin t J; z stays.
    #[test]
    fn in_the_plane_it_is_the_turn_of_expm() {
        for t in [0.0, 0.5, 1.0, FRAC_PI_2, 2.0, HALF_TURN, 5.0] {
            let r = exp_spin(DVec3::Z * t);
            let (s, c) = t.sin_cos();
            // columns: (cos t, sin t, 0), (−sin t, cos t, 0), (0, 0, 1)
            assert!((r.x_axis - DVec3::new(c, s, 0.0)).length() < 1e-14 && (r.y_axis - DVec3::new(-s, c, 0.0)).length() < 1e-14 && r.z_axis == DVec3::Z, "{t}");
        }
        let k = skew(DVec3::Z);
        assert!(k.x_axis == DVec3::Y && k.y_axis == -DVec3::X && k.z_axis == DVec3::ZERO);
    }

    /// The scene as an experiment's button leaves it, from a scene that has been played
    /// with first (so that the button must set everything itself), run at 60 frames a
    /// second until nothing moves by itself.
    fn run(k: usize) -> ExpSkew {
        let mut s = ExpSkew::create();
        s.set_mode(Mode::Pieces);
        s.glide.now(|g| {
            g.w = DVec3::new(0.3, -1.0, 0.4);
            g.p = DVec3::new(-0.4, 0.2, 1.3);
            g.t = 0.4;
        });
        (s.field, s.terms, s.terms_shown) = (false, 3, 3.0);
        s.set_up(k);
        for _ in 0..4000 {
            s.advance(1.0 / 60.0, 1.0 / 60.0);
            if !s.animated() {
                return s;
            }
        }
        panic!("experiment {k} still moving after 4000 frames");
    }

    /// The cube's corners, turned.
    fn corners(m: DMat3) -> Vec<DVec3> {
        (0..8).map(|i| m * corner(i).as_dvec3()).collect()
    }

    #[test]
    fn one_spin_does_what_its_caption_says() {
        let s = run(ONE_TRIAL);
        let set = s.shown();
        let w = set.w;
        assert!(s.mode == Mode::Turn && s.field && set.t == 1.0 && set.p == POINT && w == diagonal_spin());
        // 2.09 long along (1, 1, 1): 2.09 radians, 120 degrees, in a unit of time
        assert!(num(w.length(), 2) == "2.09" && (w.normalize() - DVec3::ONE.normalize()).length() < 1e-15 && deg(w.length().to_degrees()) == "120.0°");
        // the velocity is at right angles to the axis and to the point's arrow
        let mut rng = Rng::new(4);
        for _ in 0..50 {
            let x = random_unit(&mut rng) * rng.range(0.1, 2.0) as f64;
            let v = w.cross(x);
            assert!(v.dot(w).abs() < 1e-14 && v.dot(x).abs() < 1e-14 && (v.length() - w.length() * (x - w.normalize() * w.normalize().dot(x)).length()).abs() < 1e-13);
        }
        // along the play, p keeps its height along the axis and its distance from 0,
        // and every turn drawn is a rotation
        let n = w.normalize();
        let mut play = ExpSkew::create();
        play.set_up(ONE_TRIAL);
        let mut frames = 0;
        while play.animated() {
            play.advance(1.0 / 60.0, 1.0 / 60.0);
            let r = play.rotation();
            let q = r * POINT;
            assert!((q.dot(n) - POINT.dot(n)).abs() < 1e-12 && (q.length() - POINT.length()).abs() < 1e-12);
            assert!(off_orthonormal(r) < 1e-14 && (r.determinant() - 1.0).abs() < 1e-14);
            frames += 1;
        }
        assert!(frames > 100);
        // by t = 1: turned 120 degrees; p at (0.3, 1.1, -0.5); x's arrow along y, y's
        // along z, z's along x; the cube on its own outline
        let r = s.rotation();
        assert!(deg(s.turned().to_degrees()) == "120.0°" && (r * POINT - DVec3::new(0.3, 1.1, -0.5)).length() < 1e-14);
        assert!((r * DVec3::X - DVec3::Y).length() < 1e-15 && (r * DVec3::Y - DVec3::Z).length() < 1e-15 && (r * DVec3::Z - DVec3::X).length() < 1e-15);
        let before = corners(DMat3::IDENTITY);
        assert!(corners(r).iter().all(|c| before.iter().any(|b| (*b - *c).length() < 1e-6)));
        assert!(off_orthonormal(r) < 1e-14 && num(r.determinant(), 3) == "1.000");
    }

    #[test]
    fn the_series_does_what_its_caption_says() {
        let s = run(SERIES_TRIAL);
        let set = s.shown();
        let (w, t, p) = (set.w, set.t, set.p);
        assert!(s.mode == Mode::Series && t == 1.0 && p == DVec3::X && w == diagonal_spin());
        assert!(s.terms == COUNT_TO && s.terms_shown == COUNT_TO as f64 && TERM_EVERY == 1.0);
        let n = w.normalize();
        let parts = terms(w, t, p, N_MAX);
        // the first is the velocity at p; each after it the one before turned a quarter
        // turn about the axis, and 2.09/k times as long, across the axis
        assert!((parts[1] - w.cross(p)).length() < 1e-15);
        for k in 2..=N_MAX {
            let (last, this) = (parts[k - 1], parts[k]);
            assert!(last.dot(n).abs() < 1e-15 && this.dot(n).abs() < 1e-15 && last.dot(this).abs() < 1e-14 * last.length().max(1e-300));
            assert!(n.cross(last).dot(this) > 0.0 && (this.length() / last.length() - w.length() / k as f64).abs() < 1e-12, "{k}");
        }
        // out to 1.58 away, after the first term; then in; within 0.01 from N = 7 on
        let exact = s.rotation() * p;
        let off: Vec<f64> = (0..=N_MAX).map(|n| (partial_sum(w, t, n) * p - exact).length()).collect();
        let most = off.iter().cloned().fold(0.0, f64::max);
        assert!(num(most, 2) == "1.58" && off[1] == most && off[1..].windows(2).all(|w| w[1] < w[0]), "{off:?}");
        assert!(off[6] > 0.01 && off[7..].iter().all(|e| *e < 0.01), "{off:?}");
        // it ends at (0, 1, 0), the first column of e^K
        assert!((exact - DVec3::Y).length() < 1e-15 && (exact - s.rotation().x_axis).length() == 0.0);
        // the sum drawn is the sum of the terms drawn, and the panel says how far it is
        let drawn = s.series_drawn();
        assert!(drawn.len() == COUNT_TO + 1 && (drawn[COUNT_TO] - partial_sum(w, t, COUNT_TO) * p).length() < 1e-14);
        assert!(s.panel_lines().iter().any(|(l, _)| l.starts_with("sum to $N = 12$ ends")));
    }

    #[test]
    fn three_pieces_do_what_their_caption_says() {
        let s = run(PIECES_TRIAL);
        let set = s.shown();
        let (w, p) = (set.w, set.p);
        assert!(s.mode == Mode::Pieces && set.t == 1.0 && p == POINT && w == diagonal_spin());
        let n = w.normalize();
        let nn = skew(n);
        assert!(close(skew(w), nn * w.length()) < 1e-15 && close(nn * nn * nn, -nn) < 1e-15);
        // every power of N is ±N or ±N²
        let mut power = nn;
        for k in 2..10 {
            power *= nn;
            let want = match k % 4 {
                0 => -(nn * nn),
                1 => nn,
                2 => nn * nn,
                _ => -nn,
            };
            assert!(close(power, want) < 1e-14, "{k}");
        }
        // the two steps: from p straight in at the axis, which leaves cos θ of p's part
        // across it, and then at right angles to that, along n × p; they end on the
        // circle, where the turn takes p; the part along the axis stays and the part
        // across turns, cos of itself and sin of n × p
        let theta = s.turned();
        let (c, r, _) = circle_of(n, p).expect("p is off the axis");
        let tangent = nn * p * theta.sin();
        let inward = nn * nn * p * (1.0 - theta.cos());
        assert!(inward.normalize().dot(-r.normalize()) > 1.0 - 1e-14 && inward.dot(n).abs() < 1e-14);
        assert!((p + inward - (c + r * theta.cos())).length() < 1e-14);
        assert!(tangent.dot(r).abs() < 1e-14 && tangent.dot(n).abs() < 1e-14 && (tangent.normalize() - n.cross(p).normalize()).length() < 1e-14);
        let end = p + inward + tangent;
        assert!((end - s.rotation() * p).length() < 1e-14 && ((end - c).length() - r.length()).abs() < 1e-14);
        assert!((end - (c + r * theta.cos() + n.cross(p) * theta.sin())).length() < 1e-14);
        // (past a quarter turn the first step goes through the axis: here half as far again)
        assert!(inward.length() > r.length() && ((inward.length() - r.length()) / r.length() - 0.5).abs() < 1e-12);
        // at 120 degrees: 1.5 and 0.866 of p's distance from the axis
        assert!(deg(theta.to_degrees()) == "120.0°" && num(tangent.length() / r.length(), 3) == "0.866" && num(inward.length() / r.length(), 3) == "1.500");
        assert!(s.panel_lines().iter().any(|(l, _)| l.contains("= 0.866$") && l.contains("= 1.500$")));
    }

    #[test]
    fn two_spins_do_what_their_caption_says() {
        let s = run(SUM_TRIAL);
        let set = s.shown();
        let (w1, w2) = (set.w1(), set.w2);
        assert!(s.mode == Mode::Sum && !s.field && set.t == 1.0);
        assert!((w1 - DVec3::Y * FRAC_PI_2).length() < 1e-15 && (w2 - DVec3::Z * FRAC_PI_2).length() < 1e-15);
        // the velocities add
        let mut rng = Rng::new(9);
        for _ in 0..20 {
            let x = random_unit(&mut rng);
            assert!((w1.cross(x) + w2.cross(x) - set.w.cross(x)).length() < 1e-15);
        }
        // at once: 127.3 degrees about (0, 1, 1)
        let both = exp_spin(set.w);
        let one = log_rotation(both, set.w);
        assert!(deg(one.theta.to_degrees()) == "127.3°" && (one.axis.unwrap() - DVec3::new(0.0, 1.0, 1.0).normalize()).length() < 1e-14, "{one:?}");
        // one after the other: the quarter turn about y, then the one about z; 120
        // degrees about (-1, 1, 1)
        let made = one_then_other(w1, w2, 1.0);
        assert!(close(made, about(2, 90.0) * about(1, 90.0)) < 1e-15);
        let other = log_rotation(made, DVec3::ONE);
        assert!(deg(other.theta.to_degrees()) == "120.0°" && (other.axis.unwrap() - DVec3::new(-1.0, 1.0, 1.0).normalize()).length() < 1e-14, "{other:?}");
        // 62.4 degrees apart; the panel says so
        assert_eq!(deg(apart(both, made)), "62.4°");
        assert!(s.panel_lines().iter().any(|(l, _)| l == "at $t = 1$ they are 62.4° apart"));
        // spins about one axis commute: the gap closes
        for (a, b) in [(DVec3::Z * 0.7, DVec3::Z * 1.2), (DVec3::ONE.normalize() * 1.5, DVec3::ONE.normalize() * -0.4), (DVec3::X * 1.5, DVec3::X * 1.5)] {
            assert!(apart(exp_spin(a + b), one_then_other(a, b, 1.0)) < 1e-6 && close(exp_spin(a + b), exp_spin(b) * exp_spin(a)) < 1e-14);
        }
        // the film of one then the other starts at no turn and is w1's turn half-way
        assert!(close(one_then_other(w1, w2, 0.0), DMat3::IDENTITY) == 0.0 && close(one_then_other(w1, w2, 0.5), exp_spin(w1)) < 1e-15);
    }

    #[test]
    fn the_logarithm_does_what_its_caption_says() {
        let mut s = run(LOG_TRIAL);
        let set = s.shown();
        assert!(s.mode == Mode::Log && set.t == 1.0);
        // the rotation given: a quarter turn about x, then one about z
        let r = given_rotation();
        assert!(close(r, about(2, 90.0) * about(0, 90.0)) == 0.0);
        // tr R = 0 = 1 + 2 cos 120
        assert!(trace(r).abs() < 1e-15 && (1.0 + 2.0 * 120f64.to_radians().cos()).abs() < 1e-15);
        // R − Rᵀ = 2 sin θ N, N the spin of (1, 1, 1)/√3
        let n = DVec3::ONE.normalize();
        assert!(close(r - r.transpose(), skew(n) * (2.0 * 120f64.to_radians().sin())) < 1e-15);
        // the spin read back, played, lands exactly on R; and the scene's spin is it
        assert!((set.w - n * (WHOLE_TURN / 3.0)).length() < 1e-14 && close(s.given(), r) < 1e-15 && close(s.rotation(), r) < 1e-15);
        let lines = s.panel_lines();
        assert!(lines.iter().any(|(l, _)| l == r"$\tr R = 0.000 = 1 + 2\cos\theta$: $\theta = 120.0^\circ$"), "{lines:?}");
        assert!(lines.iter().any(|(l, _)| l == r"$(R - R^\top)/2 = \sin\theta\, N$: $n = (0.577, 0.577, 0.577)$"), "{lines:?}");
        // 480 degrees about n, or 240 about −n, land there too
        assert!(close(exp_spin(n * 480f64.to_radians()), r) < 1e-14 && close(exp_spin(-n * 240f64.to_radians()), r) < 1e-14);
        // at a half turn (the angle's slider at its end) R − Rᵀ = 0, and the axis is
        // read from R + I = 2nnᵀ instead, as the panel says
        let w = s.axis * (180.0f32 as f64).to_radians();
        s.glide.now(|g| g.w = w);
        let half = s.given();
        assert!(close(half - half.transpose(), DMat3::ZERO) < 1e-15);
        let back = log_rotation(half, s.axis);
        assert!(back.half && (back.axis.unwrap() - n).length() < 1e-14, "{back:?}");
        assert!(s.panel_lines().iter().any(|(l, _)| l == r"$R - R^\top = 0$, and $R + I = 2nn^\top$: $n = (0.577, 0.577, 0.577)$"), "{:?}", s.panel_lines());
    }

    /// The opening: a spin played from no turn to all of it, which then rests.
    #[test]
    fn the_scene_opens_by_playing_a_spin() {
        let mut s = ExpSkew::create();
        assert!(s.animated() && s.shown().t == 0.0 && s.mode == Mode::Turn);
        let mut frames = 0;
        while s.animated() && frames < 1000 {
            s.advance(1.0 / 60.0, 1.0 / 60.0);
            frames += 1;
        }
        assert!(!s.animated() && s.shown().t == 1.0 && (frames as f64 - 60.0 * PLAY_SECONDS).abs() < 3.0, "{frames}");
    }

    /// Each play moves on evenly at any display rate (no frame turns the cube or moves
    /// the point by much, also in the glide to its start), ends where it should, and then
    /// rests; nothing moves while no time passes.
    #[test]
    fn the_plays_move_smoothly_and_end_where_they_should() {
        for hz in [60.0f32, 144.0] {
            for k in 0..TRIALS.len() {
                let mut s = ExpSkew::create();
                s.set_up((k + 2) % TRIALS.len());
                for _ in 0..400 {
                    s.advance(1.0 / hz, 1.0 / hz);
                }
                s.set_up(k);
                let (mut before, mut point, mut most, mut frames) = (s.rotation(), s.rotation() * s.shown().p, 0.0f64, 0);
                while s.animated() {
                    s.advance(1.0 / hz, 1.0 / hz);
                    let (now, at) = (s.rotation(), s.rotation() * s.shown().p);
                    most = most.max(apart(before, now)).max((at - point).length().to_degrees());
                    (before, point) = (now, at);
                    frames += 1;
                    assert!(frames < 5000);
                }
                let want = setting_of(k);
                assert!(close(s.rotation(), exp_spin(want.w * want.t)) < 1e-14 && s.shown() == s.glide.to, "experiment {k}");
                // (the fastest is the glide back from a whole turn of 120 degrees to the
                // start of the next, as fast as the scene "rotations" glides back: at most
                // about 230 degrees a second, the point at 1.2 from the axis 1.2 times that)
                assert!(most < 300.0 / hz as f64, "experiment {k} at {hz} Hz: {most} in a frame");
            }
        }
        let mut s = ExpSkew::create();
        s.set_up(ONE_TRIAL);
        let still = s.shown();
        s.advance(0.0, 0.0);
        assert_eq!(s.shown(), still);
    }

    /// Degenerate input is normal input: no spin, a point at 0 or on the axis, the
    /// longest spin, two spins of nothing or in line, the series at its longest: every
    /// number drawn or read out is finite, and the panel says what has happened.
    #[test]
    fn degenerate_input_leaves_everything_finite() {
        let cases = [
            (DVec3::ZERO, DVec3::ZERO, DVec3::new(1.0, 0.0, 0.0)),
            (DVec3::Z * 1.0, DVec3::ZERO, DVec3::ZERO),
            (DVec3::Z * 1.0, DVec3::ZERO, DVec3::Z * 1.5),
            (DVec3::ONE.normalize() * W_MAX, DVec3::ZERO, DVec3::new(P_MAX, 0.0, 0.0)),
            (DVec3::new(-1.0, 2.0, 0.5).normalize() * W_MAX, DVec3::new(-1.0, 2.0, 0.5).normalize() * W_PART_MAX, DVec3::new(-2.0, 0.0, 0.0)),
            (DVec3::ZERO, DVec3::ZERO, DVec3::ZERO),
            (DVec3::X * 1e-13, DVec3::X * 1e-14, DVec3::Y * 1e-13),
        ];
        for (w, w2, p) in cases {
            for (mode, _) in MODES {
                let mut s = ExpSkew::create();
                s.set_mode(mode);
                (s.terms, s.terms_shown) = (N_MAX, N_MAX as f64);
                for t in [0.0, 0.5, 1.0] {
                    s.glide = Glide::at(Setting { w, w2, p, t });
                    s.advance(1.0 / 60.0, 1.0 / 60.0);
                    assert!(s.rotation().is_finite() && s.series_drawn().iter().all(|q| q.is_finite()));
                    assert!(one_then_other(s.shown().w1(), s.shown().w2, t).is_finite() && s.given().is_finite() && s.axis.is_finite());
                    for (line, _) in s.panel_lines() {
                        assert!(!line.contains('?') && !line.contains("NaN") && !line.contains("inf"), "{line}");
                    }
                    assert!(s.panel_height() <= PANEL_H, "{mode:?}");
                }
                if w == DVec3::ZERO {
                    assert!(s.panel_lines()[0].0.starts_with("no spin"));
                }
            }
        }
        // a point on the axis has no circle; the whole of it is drawn for any other
        assert!(circle_of(DVec3::Z, DVec3::Z * 1.5).is_none() && circle_of(DVec3::Z, DVec3::ZERO).is_none() && circle_of(DVec3::Z, DVec3::X).is_some());
        // what a drag can do is kept to its limits
        assert!(at_most(DVec3::splat(1e30), W_MAX).length() <= W_MAX + 1e-12 && at_most(DVec3::splat(f64::NAN), 1.0) == DVec3::ZERO);
    }

    /// The panel and the picture share every canvas: the panel stays inside it and clear
    /// of the application's buttons, holds its longest text, the round part for the
    /// picture stays inside the canvas and off the panel, and the view made for it is a
    /// view.
    #[test]
    fn the_canvas_is_shared_on_every_screen() {
        for (x, y) in [(1080.0f32, 1000.0f32), (1100.0, 700.0), (390.0, 800.0), (360.0, 640.0), (600.0, 900.0), (844.0, 390.0), (640.0, 480.0), (240.0, 180.0), (2000.0, 1300.0), (8.0, 8.0)] {
            let view = Vec2::new(x, y);
            let inset = if x < 900.0 { 46.0 } else { 0.0 };
            let lay = Layout::of(view, inset, PANEL_H);
            if let Some((lo, hi)) = lay.panel {
                assert!(lo.x >= 0.0 && lo.y >= inset && hi.x <= x && hi.y <= y, "{x}x{y}: {lo} {hi}");
                let p = parts((lo, hi));
                assert!(p.lines.y + 3.0 * LINE <= hi.y && p.track.1.x > p.track.0.x + 80.0 && p.time.x <= hi.x, "{x}x{y}");
            }
            let (c, r) = stage(view, inset, &lay);
            assert!(r > 0.0 && c.is_finite(), "{x}x{y}");
            if x >= 200.0 && y >= 150.0 {
                assert!(c.x - r >= -1e-3 && c.x + r <= x + 1e-3 && c.y - r >= inset && c.y + r <= y, "{x}x{y}: {c} {r}");
                if let Some((lo, hi)) = lay.panel.filter(|_| !lay.narrow) {
                    assert!((c - c.max(lo).min(hi)).length() >= r, "{x}x{y}: the picture runs under the panel");
                }
                if lay.narrow {
                    assert!(c.y + r <= lay.below, "{x}x{y}");
                }
            }
            let mut s = ExpSkew::create();
            (s.canvas, s.inset) = (view, inset);
            for k in 0..TRIALS.len() {
                s.set_up(k);
                assert!(matches!(s.the_view(), View::Orbit { target, dist, .. } if target.is_finite() && dist.is_finite() && dist > 0.0), "{x}x{y}");
                assert!(s.panel_height() <= PANEL_H);
                // every line of the panel fits across a phone's, with numbers as long as
                // they come (signs on every one)
                for _ in 0..400 {
                    s.advance(1.0 / 30.0, 1.0 / 30.0);
                }
                if let Some((lo, hi)) = Layout::of(Vec2::new(360.0, 700.0), 46.0, PANEL_H).panel {
                    let room = hi.x - lo.x - 24.0;
                    let flip = |text: &str| text.replace("(0.", "(−0.").replace(", 0.", ", −0.").replace("= 0.", "= −0.");
                    for (line, _) in s.panel_lines() {
                        assert!(text_width(&flip(&line), 12.0) <= room, "experiment {k}: {line} is {} wide, with {room} to go in", text_width(&flip(&line), 12.0));
                    }
                }
            }
        }
    }

    /// The names kept from frame to frame (`Still`) follow every change: run side by
    /// side with a copy that places every name afresh in every frame, the scene draws
    /// exactly the same in every frame, through drags of the spin's tip, of the point,
    /// of the time track and of the view, the play button, every experiment, a pause, a
    /// phone's window, and the scene opened again.
    #[test]
    fn what_is_kept_follows_every_change() {
        use super::super::subspaces::tried::{labels_follow, Step::*};
        let steps = [
            Rest(200),
            Drag([1068.0, 401.0], [1010.0, 330.0]),
            Rest(30),
            Drag([932.0, 665.0], [880.0, 700.0]),
            Rest(30),
            Drag([400.0, 37.0], [500.0, 37.0]),
            Rest(20),
            Click([271.0, 37.0]),
            Rest(200),
            Drag([1100.0, 850.0], [1180.0, 820.0]),
            Rest(40),
            Try(0),
            Rest(250),
            Try(1),
            Rest(400),
            Try(2),
            Rest(250),
            Try(3),
            Rest(250),
            Try(4),
            Rest(250),
            Pause,
            Click([271.0, 37.0]),
            Rest(20),
            Pause,
            Rest(30),
            Size([390.0, 800.0]),
            Rest(40),
            Size([1600.0, 1000.0]),
            Rest(30),
            Away,
            Rest(30),
        ];
        if let Some(again) = labels_follow("expskew", &steps) {
            assert!(again > 300, "the names were written again as they were only {again} times");
        }
    }
}

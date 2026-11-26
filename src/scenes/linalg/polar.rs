//! Polar form: every square matrix is a stretch along two perpendicular axes followed
//! by a turn or a mirror, A = QP, as every complex number is a size and a turn.

use super::*;
use crate::scenes::stats::plot::{self, Plot};
use glam::{DMat2, DVec2};
use std::f64::consts::PI as PI64;

pub const DEF: SceneDef = SceneDef {
    id: "polar",
    title: "Polar form: a stretch, then a turn",
    group: FACTOR,
    blurb: "Every square matrix is a turn (or a mirror) times a symmetric stretch, as every complex number is a direction times a size.",
    make: |_| Box::new(PolarForm::new()),
};

// ----- the arithmetic, in double precision -------------------------------------------

/// The largest entry of A either way (the handles and the editor keep to it).
const ENTRY_MAX: f64 = 3.0;

/// A matrix by its rows, [[a, b], [c, d]], the way matrices are written.
fn rows([a, b, c, d]: [f64; 4]) -> DMat2 {
    DMat2::from_cols(DVec2::new(a, c), DVec2::new(b, d))
}

/// The sum of the squares of the entries: the square of the distance from 0.
fn squares(m: DMat2) -> f64 {
    m.to_cols_array().iter().map(|v| v * v).sum()
}

/// The distance between two matrices: the square root of the sum of the squared
/// differences of their entries.
fn distance(a: DMat2, b: DMat2) -> f64 {
    squares(a - b).sqrt()
}

fn smooth(t: f64) -> f64 {
    let t = t.clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

/// The turn by the angle t.
fn turn(t: f64) -> DMat2 {
    let (s, c) = t.sin_cos();
    rows([c, -s, s, c])
}

fn outer(u: DVec2, v: DVec2) -> DMat2 {
    DMat2::from_cols(u * v.x, u * v.y)
}

/// An angle in (−π, π].
fn wrapped(t: f64) -> f64 {
    let w = (t + PI64).rem_euclid(2.0 * PI64) - PI64;
    if w <= -PI64 {
        w + 2.0 * PI64
    } else {
        w
    }
}

/// What the orthogonal factor is: a turn by an angle, or a mirror in the line through 0
/// at an angle.
#[derive(Clone, Copy, Debug, PartialEq)]
enum Orth {
    Turn(f64),
    Mirror(f64),
}

/// How far the orthogonal factor is pinned down.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Pinned {
    /// A is invertible: one Q, Q = AP⁻¹.
    One,
    /// A presses the plane onto a line: Q has only to send v₁ to u₁, and one turn and
    /// one mirror do that.
    Two,
    /// A = 0: every Q.
    Any,
}

/// A = QP = P′Q.
#[derive(Clone, Copy, Debug)]
struct Polar {
    q: DMat2,
    /// √(AᵀA), and P′ = √(AAᵀ) = QPQᵀ.
    p: DMat2,
    p2: DMat2,
    /// The singular values, σ₁ ≥ σ₂ ≥ 0: P's eigenvalues.
    s1: f64,
    s2: f64,
    /// P's unit eigenvector for σ₁ (any direction, if σ₁ = σ₂); the one for σ₂ is a
    /// quarter turn anticlockwise from it.
    v1: DVec2,
    orth: Orth,
    pinned: Pinned,
    /// σ₁ = σ₂: P stretches every direction alike.
    round: bool,
}

/// What counts as nothing: for det A against the size of A squared, and for A itself.
const FLAT: f64 = 1e-9;
const ZERO: f64 = 1e-24;

/// The polar form, in closed form. With s = σ₁ + σ₂ = √(‖A‖² + 2|det A|),
/// P = (AᵀA + |det A| I)/s, whose eigenvalues are (σᵢ² + σ₁σ₂)/(σ₁ + σ₂) = σᵢ; and Q is
/// A plus or minus its matrix of cofactors, scaled to be orthogonal: A + cof A is a
/// multiple of a turn, A − cof A of a mirror, and which is wanted is the sign of det A.
/// For A that is flat (det A = 0, A ≠ 0) both are, and `mirror_when_flat` chooses.
fn polar(a: DMat2, mirror_when_flat: bool) -> Polar {
    let n2 = squares(a);
    let det = a.determinant();
    if n2 <= ZERO {
        return Polar {
            q: DMat2::IDENTITY,
            p: DMat2::ZERO,
            p2: DMat2::ZERO,
            s1: 0.0,
            s2: 0.0,
            v1: DVec2::X,
            orth: Orth::Turn(0.0),
            pinned: Pinned::Any,
            round: true,
        };
    }
    let flat = det.abs() <= FLAT * n2;
    let sign = if flat {
        if mirror_when_flat {
            -1.0
        } else {
            1.0
        }
    } else {
        det.signum()
    };
    let [a11, a21, a12, a22] = a.to_cols_array();
    let cof = rows([a22, -a21, -a12, a11]);
    // (|det (A ± cof A)| = ‖A‖² ± 2 det A, which is above 0 for the sign chosen)
    let q = (a + cof * sign) * (1.0 / (n2 + 2.0 * sign * det).max(ZERO).sqrt());
    let sum = (n2 + 2.0 * det.abs()).sqrt();
    let diff = (n2 - 2.0 * det.abs()).max(0.0).sqrt();
    let (s1, s2) = (0.5 * (sum + diff), 0.5 * (sum - diff));
    let id = DMat2::IDENTITY * det.abs();
    let p = (a.transpose() * a + id) * (1.0 / sum);
    let p2 = (a * a.transpose() + id) * (1.0 / sum);
    // the eigenvector of the larger eigenvalue of [[p, r], [r, s]] is at half the angle
    // of (p − s, 2r)
    let round = diff <= 1e-7 * sum;
    let phi = if round { 0.0 } else { 0.5 * (2.0 * p.y_axis.x).atan2(p.x_axis.x - p.y_axis.y) };
    let v1 = DVec2::from_angle(phi);
    let angle = q.x_axis.y.atan2(q.x_axis.x);
    let orth = if q.determinant() > 0.0 { Orth::Turn(angle) } else { Orth::Mirror(0.5 * angle) };
    Polar { q, p, p2, s1, s2, v1, orth, pinned: if flat { Pinned::Two } else { Pinned::One }, round }
}

impl Polar {
    /// Where Q sends (1, 0), as an angle: Q's place on the chart.
    fn q_angle(&self) -> f64 {
        self.q.x_axis.y.atan2(self.q.x_axis.x)
    }

    /// ‖A − Q‖ = ‖QP − Q‖ = ‖P − I‖: all of it the stretch.
    fn gap(&self) -> f64 {
        ((self.s1 - 1.0).powi(2) + (self.s2 - 1.0).powi(2)).sqrt()
    }

    /// The other Q of a flat A: the mirror if Q is the turn, the turn if it is the mirror.
    fn other_q(&self, a: DMat2) -> Option<DMat2> {
        (self.pinned == Pinned::Two).then(|| polar(a, matches!(self.orth, Orth::Turn(_))).q)
    }
}

/// How far A is from the turn by t, and from the mirror that sends (1, 0) to the angle
/// t: ‖A − O‖² = ‖A‖² + 2 − 2 tr(OᵀA), and tr(OᵀA) is a cos t + b sin t.
fn to_turn(a: DMat2, t: f64) -> f64 {
    let [a11, a21, a12, a22] = a.to_cols_array();
    (squares(a) + 2.0 - 2.0 * ((a11 + a22) * t.cos() + (a21 - a12) * t.sin())).max(0.0).sqrt()
}

fn to_mirror(a: DMat2, t: f64) -> f64 {
    let [a11, a21, a12, a22] = a.to_cols_array();
    (squares(a) + 2.0 - 2.0 * ((a11 - a22) * t.cos() + (a21 + a12) * t.sin())).max(0.0).sqrt()
}

/// The nearest turn: its angle (none if every turn is as near as every other) and its
/// distance. (a cos t + b sin t is largest at the angle of (a, b).)
fn nearest_turn(a: DMat2) -> (Option<f64>, f64) {
    let [a11, a21, a12, a22] = a.to_cols_array();
    let (x, y) = (a11 + a22, a21 - a12);
    let r = x.hypot(y);
    ((r > 1e-9 * squares(a).sqrt().max(1e-12)).then(|| y.atan2(x)), (squares(a) + 2.0 - 2.0 * r).max(0.0).sqrt())
}

// ----- the two films -----------------------------------------------------------------

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Order {
    /// A = QP: P first.
    StretchFirst,
    /// A = P′Q: Q first.
    TurnFirst,
}

/// Seconds each of the two steps takes to play.
const STEP_SECONDS: f64 = 1.6;

/// Q a share s of the way: a turn by s times its angle; a mirror as the plane turned
/// over its line like a page, seen from above, I − (1 − cos πs) n nᵀ for the line's
/// normal n (flat half way, the mirror at the end).
fn q_part(pol: &Polar, s: f64) -> DMat2 {
    match pol.orth {
        Orth::Turn(t) => turn(t * s),
        Orth::Mirror(line) => {
            let n = DVec2::new(-line.sin(), line.cos());
            DMat2::IDENTITY - outer(n, n) * (1.0 - (PI64 * s).cos())
        }
    }
}

/// A stretch a share s of the way: I + s(P − I), the same axes all the way.
fn stretch_part(p: DMat2, s: f64) -> DMat2 {
    DMat2::IDENTITY + (p - DMat2::IDENTITY) * s
}

/// The plane at `progress` (0 to 2) through the film of `order`: the first step from 0
/// to 1, the second from 1 to 2, and A itself at the end.
fn staged(a: DMat2, pol: &Polar, order: Order, progress: f64) -> DMat2 {
    if progress >= 2.0 {
        return a;
    }
    let (first, second) = (smooth(progress), smooth(progress - 1.0));
    match (order, progress <= 1.0) {
        (Order::StretchFirst, true) => stretch_part(pol.p, first),
        (Order::StretchFirst, false) => q_part(pol, second) * pol.p,
        (Order::TurnFirst, true) => q_part(pol, first),
        (Order::TurnFirst, false) => stretch_part(pol.p2, second) * pol.q,
    }
}

/// The plane after the first step: P, or Q.
fn halfway(pol: &Polar, order: Order) -> DMat2 {
    match order {
        Order::StretchFirst => pol.p,
        Order::TurnFirst => pol.q,
    }
}

// ----- the shapes carried ------------------------------------------------------------

/// An F inside the unit circle, as three bars (corners anticlockwise), and its outline.
/// Lopsided on purpose: turned, it is still an F; mirrored, it is not.
const F_BARS: [[[f64; 2]; 4]; 3] = [
    [[-0.448, -0.532], [-0.196, -0.532], [-0.196, 0.532], [-0.448, 0.532]],
    [[-0.196, 0.28], [0.448, 0.28], [0.448, 0.532], [-0.196, 0.532]],
    [[-0.196, -0.112], [0.224, -0.112], [0.224, 0.112], [-0.196, 0.112]],
];
const F_OUTLINE: [[f64; 2]; 10] = [
    [-0.448, -0.532],
    [-0.196, -0.532],
    [-0.196, -0.112],
    [0.224, -0.112],
    [0.224, 0.112],
    [-0.196, 0.112],
    [-0.196, 0.28],
    [0.448, 0.28],
    [0.448, 0.532],
    [-0.448, 0.532],
];

/// Points on the unit circle.
const RING: usize = 120;

fn ring_point(i: usize) -> DVec2 {
    DVec2::from_angle(i as f64 / RING as f64 * 2.0 * PI64)
}

fn v2f(p: DVec2) -> Vec3 {
    Vec3::new(p.x as f32, p.y as f32, 0.0)
}

// ----- experiments -------------------------------------------------------------------

struct Trial {
    label: &'static str,
    look_for: &'static str,
    /// The matrix, by rows.
    a: [f64; 4],
    order: Order,
}

/// The matrix the scene opens with, and the first two experiments' (by rows).
const FIRST: [f64; 4] = [0.7, -0.3, 1.4, 1.0];

const TRIALS: [Trial; 5] = [
    Trial {
        label: "stretch, then turn",
        look_for: r"First $P$: the circle is stretched along the orange axis $v_1$, by $\sigma_1 = 1.77$, and squeezed along the purple one, by $\sigma_2 = 0.63$, and the orange and purple arrows stay on their dashed lines: a stretch does not turn its own axes. Then $Q$ turns everything by 45.0°, the $\text{F}$ with it, and the picture is $A$'s. The faint ellipse and $\text{F}$ are the plane after $P$ alone.",
        a: FIRST,
        order: Order::StretchFirst,
    },
    Trial {
        label: "turn, then stretch",
        look_for: r"The same $A$ the other way round, $A = P'Q$. First the same turn $Q$, by 45.0°; then $P'$ stretches by the same $\sigma_1 = 1.77$ and $\sigma_2 = 0.63$, along other axes: $u_1$ and $u_2$ (dashed), which are $v_1$ and $v_2$ turned by 45.0°, to 75.1° and 165.1°. So $P' = QPQ^\top$, and the film ends where the first one did.",
        a: FIRST,
        order: Order::TurnFirst,
    },
    Trial {
        label: "the nearest turn",
        look_for: r"This $A$ is nearly a turn: its columns are 1.08 and 0.89 long instead of 1, and 83° apart instead of 90°. $Q$ is the turn nearest to it, by 30.5°: the lowest point of the chart. What is left, $P$, is nearly $I$ ($\sigma_1 = 1.10$, $\sigma_2 = 0.87$), and the distance from $A$ to $Q$ is all stretch: $\|A - Q\| = \|P - I\| = 0.16$. So a turn whose numbers have drifted a little (with $\det A$ still positive) is made a turn again: replace $A$ by $Q$.",
        a: [0.9, -0.4, 0.6, 0.8],
        order: Order::TurnFirst,
    },
    Trial {
        label: "a mirror inside",
        look_for: r"The matrix of the first two experiments, followed by the mirror in the $y$ axis (the signs of its top row changed): it turns the $\text{F}$ over, and $\det A = -1.12$. No turn can do that, so $Q$ is a mirror, in the blue line at 67.5°, while $P$ is the same stretch as before. The chart's lowest point is on the dashed curve: $Q$, at $0.86$. The nearest turn is further off, at $1.81$: the turn by 74.7°, which takes $v_1$ to $u_1$ but $v_2$ to $-u_2$.",
        a: [-0.7, 0.3, 1.4, 1.0],
        order: Order::StretchFirst,
    },
    Trial {
        label: "a squash",
        look_for: r#"The second column is half the first, so $A$ presses the plane onto a line: $\det A = 0$ and $\sigma_2 = 0$. $P = \sqrt{A^\top A}$ is still the only possible $P$, and presses the circle flat along $v_1$. But all $Q$ must do is take $v_1$ to $u_1$: nothing is left along $v_2$ to show what it does there. The turn by 29.7° does it, and so does the mirror in the line at 41.4° (tick "the other $Q$"): both give $A = QP$, and both curves of the chart come down to $1.17$. $Q$ is not pinned down."#,
        a: [0.8, 0.4, 1.2, 0.6],
        order: Order::StretchFirst,
    },
];

// ----- what glides --------------------------------------------------------------------

/// The matrix and how far the film has got.
#[derive(Clone, Copy, Debug, PartialEq)]
struct Setting {
    a: DMat2,
    progress: f64,
}

impl Setting {
    fn numbers(&self) -> [f64; 5] {
        let a = self.a.to_cols_array();
        [a[0], a[1], a[2], a[3], self.progress]
    }

    fn from_numbers(v: [f64; 5]) -> Setting {
        Setting { a: DMat2::from_cols_array(&[v[0], v[1], v[2], v[3]]), progress: v[4] }
    }
}

/// How fast a change is glided to: each of the two eases one after the other covers two
/// thirds of what is left in this many seconds.
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

    /// Whether the drawn setting is within `near` of the one asked for, in every number.
    fn within(&self, near: f64) -> bool {
        let (to, between, shown) = (self.to.numbers(), self.between.numbers(), self.shown.numbers());
        (0..5).all(|i| (to[i] - between[i]).abs() < near && (to[i] - shown[i]).abs() < near)
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
        for i in 0..5 {
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

/// An angle as drawn, following the one asked for the same way, the short way round:
/// so that a direction that is only defined up to a half turn (an axis) or that swings
/// fast (near σ₁ = σ₂, where every direction is nearly an axis) turns smoothly.
#[derive(Clone, Copy, Debug)]
struct Turning {
    to: f64,
    between: f64,
    shown: f64,
}

impl Turning {
    fn at(t: f64) -> Turning {
        Turning { to: t, between: t, shown: t }
    }

    fn arrived(&self) -> bool {
        self.shown == self.to && self.between == self.to
    }

    /// Ask for the angle `t`, or for the same direction a multiple of `period` round
    /// from it, whichever is nearest to the angle as drawn.
    fn aim(&mut self, t: f64, period: f64) {
        if !t.is_finite() {
            return;
        }
        let k = ((self.shown - t) / period).round();
        let to = t + k * period;
        if to != self.to {
            self.to = to;
        }
    }

    fn step(&mut self, dt: f64) {
        if self.arrived() || dt <= 0.0 {
            return;
        }
        let k = 1.0 - (-dt / (0.6 * FOLLOW)).exp();
        self.between += (self.to - self.between) * k;
        self.shown += (self.between - self.shown) * k;
        if (self.to - self.shown).abs().max((self.to - self.between).abs()) < 1e-5 || !self.shown.is_finite() {
            *self = Turning::at(self.to);
        }
    }

    /// The angle as drawn, kept within a few turns of 0 (a direction that has gone round
    /// and round is the same direction).
    fn settle(&mut self) {
        if self.arrived() && self.to.abs() > 4.0 * PI64 {
            *self = Turning::at(wrapped(self.to));
        }
    }
}

// ----- where things go ---------------------------------------------------------------

/// The part of the plane the view is fitted to (its centre, and half its size): every
/// experiment's ellipse, handles and names lie in it.
const FRAME_CENTRE: Vec2 = Vec2::new(0.0, 0.1);
const FRAME_HALF: Vec2 = Vec2::new(2.25, 2.0);

/// The width of a number's cell in the matrices on the screen, in characters (a minus
/// sign counted as two: in a formula it is nearly twice as wide as a digit).
const CELL: usize = 6;
const PAD: f32 = 12.0;
const LINE_H: f32 = 17.0;
const BUTTON_H: f32 = 26.0;
/// The width of the panel on a wide canvas.
const BOARD_W: f32 = 372.0;

/// The panel drawn on the picture: A = QP as numbers, what Q and P are, the buttons
/// that choose and play the film, and the chart of how far A is from each turn and
/// mirror. In logical pixels.
#[derive(Clone, Debug)]
struct Board {
    min: Vec2,
    max: Vec2,
    /// The size of the numbers in the matrices.
    text: f32,
    /// The top-left of the row of matrices (their names are above it).
    mats: Vec2,
    lines: Vec2,
    buttons: [plot::Rect; 3],
    title: Vec2,
    /// The chart's title and the key: on one line each where they fit, else on two.
    title_lines: usize,
    key_lines: usize,
    chart: plot::Rect,
    /// The lines of the key at the foot (on a phone, where the picture has no room).
    key: Option<Vec2>,
}

#[derive(Clone, Debug)]
struct Layout {
    /// A phone held upright: the panel goes below the picture.
    narrow: bool,
    /// The part of the canvas the picture has to itself.
    picture: (Vec2, Vec2),
    /// None on a canvas too small to spare the room (the numbers are in the side panel).
    board: Option<Board>,
}

/// The lines of the key that goes under a phone's panel (the picture's caption elsewhere).
const KEY_LINES: [&str; 2] = ["faint: the plane after the first step", "dashed: the stretch's axes"];

/// The chart's title, as two lines.
const TITLE_LINES: [&str; 2] = [r"how far $A$ is from the turn by $t$ (solid)", r"and from the mirror in the line at $t/2$ (dashed)"];

/// What goes between two parts of a line set side by side.
const APART: &str = "      ";

/// The number of lines `parts` take at `size` in `room` across: one if they fit side by
/// side (with `gap` between), else one each.
fn lines_needed(parts: &[&str], gap: &str, size: f32, room: f32) -> usize {
    let one: f32 = parts.iter().map(|p| text_width(p, size)).sum::<f32>() + text_width(gap, size) * (parts.len() as f32 - 1.0);
    if one <= room {
        1
    } else {
        parts.len()
    }
}

impl Layout {
    fn of(view: Vec2, inset: f32) -> Layout {
        let narrow = view.x < 620.0 && view.y >= 560.0;
        let pad = if narrow { 10.0 } else { PAD };
        let width = if narrow { view.x } else { BOARD_W };
        // the numbers' size: as large as lets three matrices fit across, up to 13
        let across = |s: f32| 3.0 * matrix_size(2, 2, CELL, s).x + 40.0;
        let mut text = 13.0f32;
        while across(text) > width - 2.0 * pad && text > 9.0 {
            text -= 0.5;
        }
        let mat_h = matrix_size(2, 2, CELL, text).y;
        let room = width - 2.0 * pad - 4.0;
        let title_lines = lines_needed(&TITLE_LINES, " ", 11.5, room);
        let key_lines = if narrow { lines_needed(&KEY_LINES, APART, 11.5, room) } else { 0 };
        let key_h = if narrow { key_lines as f32 * 16.0 + 4.0 } else { 0.0 };
        // everything above the chart, from the panel's top, and below it
        let above = pad + LINE_H + 2.0 + mat_h + 8.0 + 3.0 * LINE_H + 8.0 + BUTTON_H + 12.0 + title_lines as f32 * LINE_H + 2.0;
        let below = 34.0 + key_h + pad;
        let whole = Layout { narrow, picture: (Vec2::new(0.0, inset), view), board: None };
        let place = |min: Vec2, width: f32, chart_h: f32| {
            let x = min.x + pad;
            let mut y = min.y + pad;
            y += LINE_H + 2.0;
            let mats = Vec2::new(x, y);
            y += mat_h + 8.0;
            let lines = Vec2::new(x, y);
            y += 3.0 * LINE_H + 8.0;
            let right = min.x + width - pad;
            let play_w = 64.0;
            let films = plot::even_row(2, x, right - play_w - 8.0, y, BUTTON_H, 8.0);
            let buttons = [films[0], films[1], (Vec2::new(right - play_w, y), Vec2::new(right, y + BUTTON_H))];
            y += BUTTON_H + 12.0;
            let title = Vec2::new(x, y);
            y += title_lines as f32 * LINE_H + 2.0;
            let chart = (Vec2::new(x + 30.0, y), Vec2::new(right - 8.0, y + chart_h));
            let key = narrow.then(|| Vec2::new(x, chart.1.y + 34.0));
            Board { min, max: Vec2::new(min.x + width, chart.1.y + below), text, mats, lines, buttons, title, title_lines, key_lines, chart, key }
        };
        if narrow {
            let chart_h = ((view.x - 2.0 * pad - 40.0) * 0.3).clamp(70.0, 110.0);
            // (the picture keeps at least the top 40 %)
            let top = (view.y - above - chart_h - below).max(view.y * 0.4);
            let chart_h = (view.y - top - above - below).min(chart_h);
            if chart_h < 50.0 {
                return whole;
            }
            let mut board = place(Vec2::new(0.0, top), view.x, chart_h);
            board.max = view;
            return Layout { narrow, picture: (Vec2::new(0.0, inset), Vec2::new(view.x, top)), board: Some(board) };
        }
        let top = inset + 12.0;
        let chart_h = ((width - 2.0 * pad - 38.0) * 0.5).min(view.y - 12.0 - top - above - below);
        if view.x < 640.0 || chart_h < 70.0 {
            return whole;
        }
        let board = place(Vec2::new(view.x - width - 12.0, top), width, chart_h);
        let picture = (Vec2::new(0.0, inset), Vec2::new(board.min.x - 8.0, view.y));
        Layout { narrow, picture, board: Some(board) }
    }

    /// The part of the canvas the scene's frame is fitted into: the picture's, less the
    /// heading at its top and the caption at its foot.
    fn frame_room(&self) -> (Vec2, Vec2) {
        let (min, max) = self.picture;
        let foot = if self.narrow { 8.0 } else { 44.0 };
        (min + Vec2::new(12.0, 44.0), max - Vec2::new(12.0, foot))
    }
}

fn inside_rect(p: Vec2, (min, max): (Vec2, Vec2)) -> bool {
    p.cmpge(min).all() && p.cmple(max).all()
}

/// A number for a matrix on the screen, as a formula: two decimals and a true minus sign.
fn cell(v: f64) -> String {
    format!("${}$", num(v, 2))
}

/// An angle in degrees, to a tenth, with no "−0.0" for one that rounds to nothing.
fn degrees(t: f64) -> String {
    let d = t.to_degrees();
    if d.abs() < 0.05 {
        "0.0°".into()
    } else {
        format!("{}°", num(d, 1))
    }
}

/// The parts of the picture drawn as lines that come and go, each in a batch of its own
/// (different from the usual one by a hair of its width), so that from one frame to the
/// next the parts that stay the same can still be compared point for point: that is how
/// `eidos motion` tells smooth motion from jumps.
#[derive(Clone, Copy, PartialEq)]
enum Part {
    /// The arc of Q's turn (drawn when Q is a turn, once the turn has begun).
    Arc = 1,
    /// Q's mirror line (drawn when Q is a mirror).
    Mirror = 2,
    /// The plane after the first step (drawn once the second has begun).
    Ghost = 3,
}

const PARTS: [Part; 3] = [Part::Arc, Part::Mirror, Part::Ghost];

fn lines_for<'a>(f: &'a mut Frame, part: Part) -> &'a mut crate::gfx::draw::LineBatch {
    f.dl.lines_with(LineKind::Blend, LineStyle { width_scale: 1.0 + 1e-4 * part as u8 as f32, ..Default::default() })
}

/// A batch of lines that has nothing in it this frame is given a point that cannot be
/// seen: a batch that is left empty is dropped, and every batch drawn after it would
/// then be compared with the one before it in the frame before (see `Part`).
fn keep_batch(batch: &mut crate::gfx::draw::LineBatch) {
    if batch.pts.len() <= 1 {
        batch.segment(Vec3::ZERO, Vec3::ZERO, 0.0, Color::CLEAR);
    }
}

/// The line kinds of the chart (see `plot::apart`).
const MIRRORS: u8 = plot::OWN;
const MARKS: u8 = plot::OWN + 1;

// ----- the scene ---------------------------------------------------------------------

pub struct PolarForm {
    glide: Glide,
    /// A matrix asked for (and the film to show it in), waiting for the film to be
    /// wound back to its start: at the start the plane is the same whatever A is.
    queued: Option<(DMat2, Order)>,
    order: Order,
    playing: bool,
    /// Play once the glide has arrived.
    play_when_there: bool,
    /// For a flat A, take the mirror for Q instead of the turn.
    other: bool,
    /// The direction of v₁ as drawn, and of the stretch's orange axis.
    v_angle: Turning,
    axis_angle: Turning,
    /// The top of the chart as drawn (it follows the largest distance on it, rounded up).
    ceiling: f64,
    ceiling_to: f64,
    /// Q's place on the chart as drawn, (angle in degrees, distance): it follows Q at
    /// once, but glides across when Q jumps (from a turn to a mirror as det A goes
    /// through 0), and `marker_moving` while it does.
    marker: plot::Glide,
    marker_moving: bool,
    /// The canvas last drawn on and how much of its top the application's buttons
    /// cover: they decide where the panel goes, and so where the picture is.
    canvas: Vec2,
    inset: f32,
    measured: bool,
    new_view: bool,
}

/// The tops the chart may have: the least that leaves room above the highest curve.
const CEILINGS: [f64; 8] = [2.0, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0, 12.0];

fn ceiling_for(a: DMat2) -> f64 {
    let pol = polar(a, false);
    // the highest point of either curve: √((σ₁ + 1)² + (σ₂ + 1)²)
    let top = ((pol.s1 + 1.0).powi(2) + (pol.s2 + 1.0).powi(2)).sqrt();
    CEILINGS.iter().copied().find(|c| *c >= top * 1.04).unwrap_or(CEILINGS[CEILINGS.len() - 1])
}

impl PolarForm {
    fn new() -> Self {
        let a = rows(FIRST);
        let pol = polar(a, false);
        let mut scene = PolarForm {
            glide: Glide::at(Setting { a, progress: 0.0 }),
            queued: None,
            order: Order::StretchFirst,
            playing: true,
            play_when_there: false,
            other: false,
            v_angle: Turning::at(pol.v1.to_angle()),
            axis_angle: Turning::at(pol.v1.to_angle()),
            ceiling: ceiling_for(a),
            ceiling_to: ceiling_for(a),
            marker: plot::Glide::default(),
            marker_moving: false,
            canvas: Vec2::new(1080.0, 1000.0),
            inset: 0.0,
            measured: false,
            new_view: false,
        };
        scene.set_up(0);
        // (opening, there is nothing to wind back: the film plays from the start)
        if let Some((a, order)) = scene.queued.take() {
            scene.glide = Glide::at(Setting { a, progress: 0.0 });
            scene.order = order;
        }
        (scene.playing, scene.play_when_there) = (true, false);
        scene
    }

    /// Set up experiment `k` of `TRIALS`: wind the film back, glide to its matrix,
    /// then play its film.
    fn set_up(&mut self, k: usize) {
        let trial = &TRIALS[k];
        self.other = false;
        self.ask(rows(trial.a), trial.order);
    }

    /// Ask for a new matrix or film (a button, a typed number): the film is wound back
    /// to its start, A glides there to the new matrix, and the new film plays.
    fn ask(&mut self, a: DMat2, order: Order) {
        self.queued = Some((within_limits(a), order));
        self.glide.to.progress = 0.0;
        self.playing = false;
        self.play_when_there = true;
    }

    /// The matrix as it is asked for, the last of what is queued.
    fn asked(&self) -> DMat2 {
        self.queued.map_or(self.glide.to.a, |(a, _)| a)
    }

    fn asked_order(&self) -> Order {
        self.queued.map_or(self.order, |(_, o)| o)
    }

    /// A moved by hand (a handle, a number dragged): at once, and the film taken to its
    /// end, where the picture is the whole of A.
    fn hand(&mut self, a: DMat2) {
        let a = within_limits(a);
        if let Some((_, order)) = self.queued.take() {
            self.order = order;
        }
        self.glide.now(|s| s.a = a);
        self.glide.to.progress = 2.0;
        (self.playing, self.play_when_there) = (false, false);
    }

    /// Play from the start (or on from where the film is, if it is part of the way).
    fn play(&mut self) {
        if self.glide.to.progress >= 2.0 - 1e-9 {
            self.glide.to.progress = 0.0;
            self.play_when_there = true;
        } else {
            self.playing = true;
        }
    }

    fn shown(&self) -> Setting {
        self.glide.shown
    }

    /// The polar form of the matrix as drawn.
    fn polar(&self) -> Polar {
        polar(self.shown().a, self.other)
    }

    /// Move everything on: `dt` seconds of the scene's time (the film) and `real_dt` of
    /// the clock's (a change of setting glides by the clock, so that it is seen through
    /// even while the scene's time stands still).
    fn advance(&mut self, dt: f32, real_dt: f32) {
        self.glide.step(real_dt as f64);
        // a matrix waits for the film to be wound back; the last ten-thousandth of the
        // way back moves nothing that can be seen (smoothstep of 0.004 is 5e-5)
        if self.glide.to.progress == 0.0 && self.glide.shown.progress.max(self.glide.between.progress) < 0.004 {
            if let Some((a, order)) = self.queued.take() {
                self.glide.now(|s| s.progress = 0.0);
                self.glide.to.a = a;
                self.order = order;
            }
        }
        // and the film plays once the new matrix is all but there (the rest of its
        // glide goes on as it plays)
        if self.play_when_there && self.queued.is_none() && self.glide.within(0.01) {
            (self.play_when_there, self.playing) = (false, true);
        }
        if self.playing {
            let p = (self.glide.to.progress + dt as f64 / STEP_SECONDS).min(2.0);
            self.glide.now(|s| s.progress = p);
            if p >= 2.0 {
                self.playing = false;
            }
        }
        let pol = self.polar();
        self.v_angle.aim(pol.v1.to_angle(), if pol.round { 0.5 * PI64 } else { PI64 });
        self.v_angle.step(real_dt as f64);
        self.v_angle.settle();
        let axis = match self.order {
            Order::StretchFirst => self.v_angle.to,
            Order::TurnFirst => (pol.q * DVec2::from_angle(self.v_angle.to)).to_angle(),
        };
        self.axis_angle.aim(axis, if pol.round { 0.5 * PI64 } else { PI64 });
        self.axis_angle.step(real_dt as f64);
        self.axis_angle.settle();
        let to = [pol.q_angle().to_degrees() as f32, pol.gap() as f32];
        let was = [self.marker.get(0), self.marker.get(1)];
        // (a turn whose angle goes round past a half turn is the same turn: no glide)
        let wrap = (to[0] - was[0]).abs() > 300.0;
        if self.marker.shown.is_empty() || wrap || !(self.marker_moving || (to[0] - was[0]).abs() > 8.0) {
            self.marker.snap(&to);
            self.marker_moving = false;
        } else {
            self.marker_moving = self.marker.follow(&to, real_dt, false);
        }
        self.ceiling_to = ceiling_for(self.shown().a);
        if self.ceiling != self.ceiling_to {
            let k = 1.0 - (-(real_dt as f64) / 0.12).exp();
            self.ceiling += (self.ceiling_to - self.ceiling) * k;
            if (self.ceiling - self.ceiling_to).abs() < 1e-3 {
                self.ceiling = self.ceiling_to;
            }
        }
    }

    /// v₁ and v₂ as drawn.
    fn axes_drawn(&self) -> (DVec2, DVec2) {
        let v1 = DVec2::from_angle(self.v_angle.shown);
        (v1, v1.perp())
    }

    fn at_rest(&self) -> bool {
        self.glide.arrived() && self.queued.is_none() && !self.playing && !self.play_when_there && self.v_angle.arrived() && self.axis_angle.arrived() && self.ceiling == self.ceiling_to && !self.marker_moving
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

    /// What the step the film is in is called, for the picture's heading.
    fn stage_text(&self) -> String {
        let p = self.shown().progress;
        let q = if matches!(self.polar().orth, Orth::Mirror(_)) { "the mirror $Q$" } else { "the turn $Q$" };
        let (form, first, second) = match self.order {
            Order::StretchFirst => ("$A = QP$", r"the stretch $P$", q),
            Order::TurnFirst => ("$A = P'Q$", q, r"the stretch $P'$"),
        };
        if p <= 1e-9 {
            match self.order {
                Order::StretchFirst => format!("{form}: the plane before $P$ and $Q$"),
                Order::TurnFirst => format!("{form}: the plane before $Q$ and $P'$"),
            }
        } else if p < 1.0 {
            format!("1 of 2: {first}")
        } else if p < 2.0 {
            format!("2 of 2: {second}")
        } else {
            format!("{form}: {first}, then {second}")
        }
    }
}

/// The matrix with every entry within ±`ENTRY_MAX` (and 0 for one that is not a number).
fn within_limits(m: DMat2) -> DMat2 {
    DMat2::from_cols_array(&m.to_cols_array().map(|v| if v.is_finite() { v.max(-ENTRY_MAX).min(ENTRY_MAX) } else { 0.0 }))
}

/// The four numbers of a 2×2 matrix to drag or type, within ±`ENTRY_MAX`. Returns
/// (changed, typed): a number typed in is a new matrix, played from the start; one
/// dragged moves the picture at once.
fn matrix_edit(ui: &mut egui::Ui, m: &mut DMat2) -> (bool, bool) {
    let (mut changed, mut typed) = (false, false);
    let mut a = [[m.x_axis.x, m.y_axis.x], [m.x_axis.y, m.y_axis.y]];
    egui::Grid::new("polar_a").spacing([6.0, 4.0]).show(ui, |ui| {
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

/// What Q is, in words, for the panel and the picture.
fn q_words(pol: &Polar, a: DMat2) -> String {
    let what = |orth: Orth| match orth {
        Orth::Turn(t) => format!("turn by {}", degrees(t)),
        Orth::Mirror(l) => format!("mirror in the line at {}", degrees(l)),
    };
    match pol.pinned {
        Pinned::Any => r"$A = 0$, so $P = 0$, and every turn or mirror will do for $Q$".into(),
        Pinned::Two => {
            let other = polar(a, !matches!(pol.orth, Orth::Mirror(_))).orth;
            format!("$Q$: {}, or {}", what(pol.orth), what(other))
        }
        Pinned::One => format!("$Q$: {}", what(pol.orth)),
    }
}

impl Scene for PolarForm {
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
        !self.at_rest()
    }

    fn ui(&mut self, ui: &mut egui::Ui, _: &Gfx) {
        ui::note(ui, "Every square matrix $A$ can be written as a stretch followed by a turn or a mirror:");
        ui::equation(ui, r"A = \textcolor{blue}{Q} P");
        ui::note(ui, r"The stretch $P$ is symmetric with no negative eigenvalues: it stretches the plane along two perpendicular axes, $\textcolor{orange}{v_1}$ and $\textcolor{purple}{v_2}$, by $\sigma_1$ and $\sigma_2$, the singular values of $A$, and does not turn them. $\textcolor{blue}{Q}$ keeps every length and angle: it is a turn if $\det A > 0$ and a mirror if $\det A < 0$. In the same way a complex number is written as $z = e^{i\theta} r$: a turn times a size. Watch the $\text{F}$ and the circle go through the two steps.");

        ui::section(ui, "Things to try");
        for (k, trial) in TRIALS.iter().enumerate() {
            if ui::experiment(ui, trial.label, trial.look_for) {
                self.set_up(k);
            }
        }

        ui::section(ui, "Film");
        let mut order = (self.asked_order() == Order::TurnFirst) as usize;
        if ui::choice(ui, &["$P$ first: $A = QP$", "$Q$ first: $A = P'Q$"], &mut order) {
            let order = if order == 1 { Order::TurnFirst } else { Order::StretchFirst };
            self.ask(self.asked(), order);
        }
        ui.horizontal(|ui| {
            let busy = self.playing || self.play_when_there;
            if ui.button(if busy { "❚❚ pause" } else { "▶ play" }).clicked() {
                if busy {
                    (self.playing, self.play_when_there) = (false, false);
                } else {
                    self.play();
                }
            }
        });
        let mut p = self.glide.to.progress as f32;
        if ui::slider(ui, "progress", &mut p, 0.0..=2.0) {
            if let Some((a, order)) = self.queued.take() {
                self.glide.now(|s| s.a = a);
                self.order = order;
            }
            self.glide.now(|s| s.progress = p as f64);
            (self.playing, self.play_when_there) = (false, false);
        }
        ui::note(ui, "0 is the plane as it was, 1 after the first step, 2 after both: the whole of $A$.");

        ui::section(ui, "Matrix $A$");
        let mut a = self.asked();
        let (changed, typed) = matrix_edit(ui, &mut a);
        if changed {
            if typed {
                self.ask(a, self.asked_order());
            } else {
                self.hand(a);
            }
        }
        ui::note(ui, "Or drag the pink and green dots, the columns of $A$: the film then goes to its end, the whole of $A$, and play shows it again from the start.");
        let pol = self.polar();
        if pol.pinned == Pinned::Two || self.other {
            if ui::checkbox(ui, &mut self.other, r"the other $Q$ (only for $\det A = 0$)").changed() {
                // (the film again, with the other Q)
                self.ask(self.asked(), self.asked_order());
            }
            ui::note(ui, r"When $A$ squashes the plane flat, a turn and a mirror both serve as $Q$: this takes the other one, and plays the film with it.");
        }

        ui::section(ui, "Reading the picture");
        ui::key(
            ui,
            &[
                (pal::YELLOW, r"yellow: the unit circle and an $\text{F}$, carried through the two steps; faint: as they were after the first step"),
                (pal::PINK, r"pink and green: where $(1, 0)$ and $(0, 1)$ are carried; the dots are the columns of $A$, $a_1$ and $a_2$ (drag them)"),
                (pal::ORANGE, r"orange and purple arrows: $v_1$ and $v_2$, carried; dashed lines: the axes of the stretch, $v_1$ and $v_2$ for $P$, $u_1 = Qv_1$ and $u_2 = Qv_2$ for $P'$"),
                (pal::BLUE, "blue: $Q$, by its angle or its mirror line; on the chart, how far $A$ is from each turn (solid) and each mirror (dashed)"),
                (pal::RED, r"red: the distance $\|A - Q\|$, at the chart's lowest point"),
            ],
        );

        ui::section(ui, "Words");
        ui::term(ui, "Polar form:", r"$A = QP$ with $Q$ orthogonal and $P$ symmetric with no negative eigenvalues. $P = \sqrt{A^\top A}$ is the only such $P$, and if $A$ is invertible, $Q = AP^{-1}$ is the only such $Q$. The other order, $A = P'Q$, has the same $Q$ and $P' = QPQ^\top = \sqrt{AA^\top}$.");
        ui::term(ui, "Orthogonal:", r"$Q^\top Q = I$: $Q$ keeps every length and angle. In the plane it is a turn ($\det Q = 1$) or a mirror in a line through $0$ ($\det Q = -1$).");
        ui::term(ui, "Symmetric:", r"$P^\top = P$. Such a matrix stretches the plane along two perpendicular axes, its eigenvectors, by its eigenvalues.");
        ui::term(ui, "Singular values:", r"$\sigma_1 \ge \sigma_2 \ge 0$, the eigenvalues of $P$: the most and the least $A$ stretches any direction. From the SVD $A = U \Sigma V^\top$: $P = V \Sigma V^\top$ and $Q = UV^\top$.");
        ui::term(ui, r"Distance $\|A - B\|$:", "the square root of the sum of the squares of the differences of the four entries.");
        ui::term(ui, "Symmetric and skew parts:", r"$A = S + K$ with $S = (A + A^\top)/2$ and $K = (A - A^\top)/2$: a sum, where the polar form is a product. Near $I$ they agree: for $A = I + \varepsilon E$ with $\varepsilon$ small, $P = I + \varepsilon S_E$ and $Q = I + \varepsilon K_E$, up to terms in $\varepsilon^2$, where $S_E$ and $K_E$ are the parts of $E$.");

        ui::section(ui, "Numbers");
        let a = self.shown().a;
        let pol = self.polar();
        ui::readout(ui, r"$\det A$", format!("${}$", num(a.determinant(), 2)));
        let q = match (pol.pinned, pol.orth) {
            (Pinned::Any, _) => "any turn or mirror".to_string(),
            (_, Orth::Turn(t)) => format!("turn by {}", degrees(t)),
            (_, Orth::Mirror(l)) => format!("mirror, line at {}", degrees(l)),
        };
        ui::readout_colored(ui, "$Q$", q, pal::BLUE);
        ui::readout_colored(ui, r"$\sigma_1$", format!("${}$", num(pol.s1, 2)), pal::ORANGE);
        ui::readout_colored(ui, r"$\sigma_2$", format!("${}$", num(pol.s2, 2)), pal::PURPLE);
        ui::readout_colored(ui, r"$\|A - Q\| = \|P - I\|$", format!("${}$", num(pol.gap(), 2)), pal::RED);
        if a.determinant() < 0.0 && pol.pinned == Pinned::One {
            if let (Some(t), d) = nearest_turn(a) {
                ui::readout(ui, "nearest turn", format!("by {}, at ${}$", degrees(t), num(d, 2)));
            }
        }
        ui::see_also(ui, &[("svd", r"the SVD, $A = U \Sigma V^\top$, that the two factors come from"), ("spectral", "a symmetric matrix is a stretch along perpendicular axes"), ("symskew", "a matrix as a sum: symmetric plus skew"), ("factor", "five ways to take a matrix apart")]);
    }

    fn frame(&mut self, f: &mut Frame) {
        // the panel's place follows the canvas, and the picture's place follows the panel's
        let view = f.view_size();
        let lay = Layout::of(view, f.top_inset);
        let before = Layout::of(self.canvas, self.inset);
        // (a canvas squeezed to a sliver is one the controls have opened over, on a
        // phone: the picture is not seen then, and is left as it was for when they close)
        let covered = view.x < 160.0;
        let changed = !covered && (lay.narrow != before.narrow || lay.board.is_some() != before.board.is_some() || (view - self.canvas).abs().max_element() > 24.0);
        if !covered && (changed || !self.measured) {
            self.new_view |= changed;
            (self.canvas, self.measured) = (view, true);
        }
        if !covered {
            self.new_view |= (f.top_inset - self.inset).abs() > 1.0;
            self.inset = f.top_inset;
        }

        let pointer = plot::pointer(f);
        if let Some(b) = lay.board.as_ref().filter(|_| !covered) {
            if pointer.is_some_and(|p| inside_rect(p, (b.min, b.max))) {
                // the panel is for its own buttons: no moving the view from there
                f.handles.capture();
                f.handles.claim_clicks();
            }
        }
        // the board's buttons (drawn with the board, after the picture; their clicks
        // are taken here, before the scene moves on)
        if let Some(b) = lay.board.as_ref().filter(|_| !covered) {
            if let Some(p) = pointer.filter(|_| f.ptr.clicked) {
                if inside_rect(p, b.buttons[0]) {
                    self.ask(self.asked(), Order::StretchFirst);
                } else if inside_rect(p, b.buttons[1]) {
                    self.ask(self.asked(), Order::TurnFirst);
                } else if inside_rect(p, b.buttons[2]) {
                    if self.playing || self.play_when_there {
                        (self.playing, self.play_when_there) = (false, false);
                    } else {
                        self.play();
                    }
                }
            }
        }

        // the handles: A's columns (not where the panel covers them)
        let a = self.shown().a;
        for (k, c) in [pal::PINK, pal::GREEN].into_iter().enumerate() {
            let mut col = a.col(k).as_vec2();
            let under = lay.board.as_ref().is_some_and(|b| on_screen(f, col.extend(0.0)).is_some_and(|q| inside_rect(q, (b.min, b.max))));
            if (!under || f.handles.dragging()) && f.drag2(1 + k as u32, &mut col, c) {
                let mut m = self.shown().a;
                *m.col_mut(k) = col.as_dvec2();
                self.hand(m);
            }
        }
        self.advance(f.dt, f.real_dt);

        let labels_before = f.dl.labels.len();
        let plates = self.picture(f, &lay, covered);
        let numbers_from = f.dl.labels.len();
        f.axes2();
        // no number of the axes is written over the heading or the caption
        let mut k = 0;
        f.dl.labels.retain(|l| {
            k += 1;
            k <= numbers_from || !plates.iter().flatten().any(|plate| overlap(label_box(l), *plate))
        });
        if covered {
            // (the sliver left beside the controls is the picture alone)
            f.dl.labels.truncate(labels_before);
            return;
        }
        // nothing written into the picture may show through the panel
        if let Some(b) = &lay.board {
            let (lo, hi) = (b.min - Vec2::splat(2.0), b.max + Vec2::splat(2.0));
            let mut k = 0;
            f.dl.labels.retain(|l| {
                k += 1;
                k <= labels_before || l.backed || !inside_rect(l.pos, (lo, hi))
            });
            self.board(f, b);
        }
    }
}

impl PolarForm {
    /// Everything drawn in the plane, and the names written beside it. Returns the boxes
    /// of the heading and of the caption, which are written over it.
    fn picture(&mut self, f: &mut Frame, lay: &Layout, covered: bool) -> [Option<(Vec2, Vec2)>; 2] {
        let Setting { a, progress } = self.shown();
        let pol = self.polar();
        let m = staged(a, &pol, self.order, progress);
        let (v1, v2) = self.axes_drawn();
        let (lo, hi) = f.cam.flat_bounds();
        let reach = (hi - lo).length() as f64 + 10.0;
        // (the batches of the parts that come and go, in the same order every frame)
        for part in PARTS {
            lines_for(f, part);
        }

        // where it started: the unit circle, faint
        let start: Vec<Vec3> = (0..RING).map(|i| v2f(ring_point(i))).collect();
        f.dl.lines.closed(&start, 1.0, pal::DIM.a(0.45));

        // the stretch's axes, dashed, through 0
        let axis = DVec2::from_angle(self.axis_angle.shown);
        for (d, c) in [(axis, pal::ORANGE), (axis.perp(), pal::PURPLE)] {
            f.dashed(v2f(-d * reach), v2f(d * reach), 1.3, c.a(0.6));
        }

        // the plane after the first step, faint (it fades in as the second begins)
        let half = halfway(&pol, self.order);
        let fade = 0.55 * smooth((progress - 1.0) / 0.35) as f32;
        if fade > 0.0 {
            let ghost: Vec<Vec3> = (0..RING).map(|i| v2f(half * ring_point(i))).collect();
            lines_for(f, Part::Ghost).closed(&ghost, 1.2, pal::YELLOW.a(0.6 * fade));
            let ghost_f: Vec<Vec3> = F_OUTLINE.iter().map(|p| v2f(half * DVec2::from(*p))).collect();
            lines_for(f, Part::Ghost).closed(&ghost_f, 1.0, pal::YELLOW.a(0.7 * fade));
            for bar in F_BARS {
                let [p0, p1, p2, p3] = bar.map(|p| v2f(half * DVec2::from(p)));
                f.dl.tris_alpha.quad(p0, p1, p2, p3, pal::YELLOW.a(0.12 * fade));
            }
        }
        if fade > 0.01 {
            for (k, c) in [pal::PINK, pal::GREEN].into_iter().enumerate() {
                f.dot(v2f(half.col(k)), 7.0, c.a(fade));
            }
        }

        // Q's turn, as an arc from v₁ round to where Q takes it; or its mirror line
        let q_share = match self.order {
            Order::StretchFirst => smooth(progress - 1.0),
            Order::TurnFirst => smooth(progress),
        };
        let mut arc_label = None;
        match pol.orth {
            // (none before the turn begins: an arc of no length has nowhere to be)
            Orth::Turn(t) if pol.pinned != Pinned::Any && t.abs() >= 0.5f64.to_radians() && q_share > 0.0 => {
                // (on the path of the orange arrow's tip as Q turns it: σ₁ from the
                // centre if P came first, 1 if Q comes first)
                let from = v1.to_angle();
                let r = match self.order {
                    Order::StretchFirst => pol.s1,
                    Order::TurnFirst => 1.0,
                }
                .max(0.3);
                let pts: Vec<Vec3> = (0..=40).map(|i| v2f(DVec2::from_angle(from + t * q_share * i as f64 / 40.0) * r)).collect();
                lines_for(f, Part::Arc).polyline(pts, 1.8, pal::BLUE);
                if q_share > 0.02 {
                    // its head: a small triangle, the size fixed on the screen
                    let end = from + t * q_share;
                    let wpp = (hi.y - lo.y) as f64 / f.view_size().y.max(1.0) as f64;
                    let tip = DVec2::from_angle(end) * r;
                    let along = DVec2::from_angle(end).perp() * t.signum();
                    let head = (10.0 * wpp).min(0.6 * r * (t * q_share).abs());
                    let base = tip - along * head;
                    let side = along.perp() * (head * 0.45);
                    f.dl.tris_alpha.tri(v2f(tip), v2f(base + side), v2f(base - side), pal::BLUE);
                    arc_label = Some((DVec2::from_angle(from + 0.5 * t * q_share) * r, format!("$Q$: turn by {}", degrees(t))));
                }
            }
            Orth::Mirror(line) => {
                // (named far out along it, on the side that is up on the screen)
                let d = DVec2::from_angle(line);
                let d = if d.y < 0.0 || (d.y == 0.0 && d.x < 0.0) { -d } else { d };
                let bright = 0.35 + 0.6 * smooth(q_share * 4.0) as f32;
                lines_for(f, Part::Mirror).segment(v2f(-d * reach), v2f(d * reach), 2.0, pal::BLUE.a(bright));
                arc_label = Some((d * 1.85, "$Q$: mirror".to_string()));
            }
            _ => {}
        }

        for part in PARTS {
            keep_batch(lines_for(f, part));
        }

        // the circle and the F, carried
        let ring: Vec<Vec3> = (0..RING).map(|i| v2f(m * ring_point(i))).collect();
        f.dl.lines.closed(&ring, 2.4, pal::YELLOW);
        for bar in F_BARS {
            let [p0, p1, p2, p3] = bar.map(|p| v2f(m * DVec2::from(p)));
            f.dl.tris_alpha.quad(p0, p1, p2, p3, pal::YELLOW.a(0.3));
        }
        let outline: Vec<Vec3> = F_OUTLINE.iter().map(|p| v2f(m * DVec2::from(*p))).collect();
        f.dl.lines.closed(&outline, 1.4, pal::YELLOW.a(0.9));

        // where (1, 0) and (0, 1) are carried, and v₁ and v₂
        for (k, c) in [pal::PINK, pal::GREEN].into_iter().enumerate() {
            f.arrow(Vec2::ZERO, (m * DVec2::AXES[k]).as_vec2(), 2.0, c.a(0.85));
        }
        let tips = [m * v1, m * v2];
        for (tip, c) in tips.iter().zip([pal::ORANGE, pal::PURPLE]) {
            f.arrow(Vec2::ZERO, tip.as_vec2(), 3.0, c);
        }

        // the heading, at the top of the picture, and the caption at its foot
        let heading_at = lay.picture.0 + Vec2::new(16.0, 14.0);
        let text = self.stage_text();
        let size = 15.0;
        let wide = f.text_size(&text, size).x;
        f.caption_px(heading_at, text, size, pal::TEXT, Align2::LEFT_TOP);
        let heading = (heading_at - Vec2::splat(6.0), heading_at + Vec2::new(wide + 12.0, size + 10.0));
        let mut plates = [Some(heading), None];
        if !lay.narrow && f.view_size().y >= 300.0 {
            let parts = [r"yellow: the unit circle and an $\text{F}$, carried along", "faint: the plane after the first step", "dashed: the stretch's axes"];
            let (left, bottom) = (lay.picture.0.x + 18.0, lay.picture.1.y - 16.0);
            let line = parts.join("      ");
            let w = f.text_size(&line, 12.0).x;
            if w <= lay.picture.1.x - lay.picture.0.x - 36.0 {
                f.caption_px(Vec2::new(left, bottom), line, 12.0, pal::DIM, Align2::LEFT_BOTTOM);
                plates[1] = Some((Vec2::new(left - 8.0, bottom - 20.0), Vec2::new(left + w + 8.0, bottom + 6.0)));
            }
        }
        if covered {
            return plates;
        }

        // the names, keeping clear of what is drawn and of one another
        let mut room = Room::new(f);
        if let Some(b) = &lay.board {
            room.wall(b.min - Vec2::splat(4.0), b.max + Vec2::splat(4.0));
        }
        for plate in plates.iter().flatten() {
            room.wall(plate.0, plate.1);
        }
        for i in 0..RING / 4 {
            room.line(f, ring[4 * i], ring[(4 * i + 4) % RING]);
        }
        for i in 0..outline.len() {
            room.line(f, outline[i], outline[(i + 1) % outline.len()]);
        }
        for k in 0..2 {
            room.line(f, Vec3::ZERO, v2f(m * DVec2::AXES[k]));
            room.line(f, Vec3::ZERO, v2f(tips[k]));
        }
        let away = |p: DVec2| Vec2::new(p.x as f32, -(p.y as f32));
        let px = |f: &Frame, p: DVec2, q: DVec2| on_screen(f, v2f(p)).zip(on_screen(f, v2f(q))).map_or(0.0, |(p, q)| p.distance(q));
        // v₁ and v₂ as far as the film has got, by what they have become
        let done = if progress >= 2.0 { 2 } else if progress >= 1.0 { 1 } else { 0 };
        for (k, c) in [pal::ORANGE, pal::PURPLE].into_iter().enumerate() {
            if px(f, DVec2::ZERO, tips[k]) >= 12.0 {
                let name = arrow_name(self.order, k, done);
                room.label(f, ("v", k), v2f(tips[k]), name, 15.0, c, away(tips[k]), 8.0, false);
            }
        }
        if let Some((at, text)) = arc_label {
            // (on a phone Q's name alone: the panel just below says what it is)
            let (text, size) = if lay.narrow { ("$Q$", 15.0) } else { (text.as_str(), 13.0) };
            room.label(f, "q", v2f(at), text, size, pal::BLUE, away(at), 10.0, false);
        }
        // the columns of A, at the handles
        for (k, (name, c)) in [("$a_1$", pal::PINK), ("$a_2$", pal::GREEN)].into_iter().enumerate() {
            let col = a.col(k);
            if on_screen(f, v2f(col)).is_some_and(|q| inside_rect(q, (Vec2::ZERO, f.view_size()))) {
                room.label(f, ("col", k), v2f(col), name, 15.0, c, away(col), 12.0, false);
            }
        }
        plates
    }

    /// The panel on the picture: A = QP as numbers, what Q and P are, the film's buttons
    /// and the chart.
    fn board(&self, f: &mut Frame, b: &Board) {
        hud_panel(f, b.min, b.max);
        let a = self.shown().a;
        let pol = self.polar();
        let s = b.text;
        let small = 12.0;

        // A = Q P (or P′ Q), the columns of A in the colours of the arrows that end there
        let entries = |m: DMat2, cols: [Color; 2]| -> Vec<Vec<(String, Color)>> { (0..2).map(|i| (0..2).map(|j| (cell(m.col(j)[i]), cols[j])).collect()).collect() };
        let msize = matrix_size(2, 2, CELL, s);
        let mid = b.mats.y + msize.y * 0.5;
        let q_rows = entries(pol.q, [pal::BLUE, pal::BLUE]);
        let p_rows = entries(if self.order == Order::StretchFirst { pol.p } else { pol.p2 }, [pal::TEXT, pal::TEXT]);
        let factors = match self.order {
            Order::StretchFirst => [("$Q$", q_rows, pal::BLUE), ("$P$", p_rows, pal::TEXT)],
            Order::TurnFirst => [("$P'$", p_rows, pal::TEXT), ("$Q$", q_rows, pal::BLUE)],
        };
        let mut x = b.mats.x;
        let name_y = b.mats.y - 4.0;
        f.text_px(Vec2::new(x + msize.x * 0.5, name_y), "$A$", 13.0, pal::TEXT, Align2::CENTER_BOTTOM);
        x += hud_matrix(f, Vec2::new(x, b.mats.y), &entries(a, [pal::PINK, pal::GREEN]), CELL, s, pal::DIM).x;
        f.text_px(Vec2::new(x + 13.0, mid), "$=$", s, pal::TEXT, Align2::CENTER_CENTER);
        x += 26.0;
        for (name, rows, c) in factors {
            f.text_px(Vec2::new(x + msize.x * 0.5, name_y), name, 13.0, c, Align2::CENTER_BOTTOM);
            x += hud_matrix(f, Vec2::new(x, b.mats.y), &rows, CELL, s, pal::DIM).x + 10.0;
        }

        // what Q is, what P is, and how far A is from Q
        let mut y = b.lines.y;
        f.text_px(Vec2::new(b.lines.x, y), q_words(&pol, a), small, pal::BLUE, Align2::LEFT_TOP);
        y += LINE_H;
        let along = match self.order {
            Order::StretchFirst => ["$v_1$", "$v_2$"],
            Order::TurnFirst => ["$u_1$", "$u_2$"],
        };
        let stretch = if pol.pinned == Pinned::Any {
            "$P = 0$: everything is pressed to $0$".to_string()
        } else if pol.round {
            format!(r"$P = {} I$: the same stretch in every direction", num(pol.s1, 2))
        } else {
            let name = if self.order == Order::StretchFirst { "$P$" } else { "$P'$" };
            format!(r"{name}: by $\textcolor{{orange}}{{\sigma_1 = {}}}$ along {}, $\textcolor{{purple}}{{\sigma_2 = {}}}$ along {}", num(pol.s1, 2), along[0], num(pol.s2, 2), along[1])
        };
        f.text_px(Vec2::new(b.lines.x, y), stretch, small, pal::TEXT, Align2::LEFT_TOP);
        y += LINE_H;
        let p_name = if self.order == Order::StretchFirst { "P" } else { "P'" };
        f.text_px(Vec2::new(b.lines.x, y), format!(r"$\|A - Q\| = \|{p_name} - I\| = {}$", num(pol.gap(), 2)), small, pal::RED, Align2::LEFT_TOP);

        // the film's buttons
        let busy = self.playing || self.play_when_there;
        let order = self.asked_order();
        plot::button(f, b.buttons[0], "$P$ first: $A = QP$", 12.0, order == Order::StretchFirst);
        plot::button(f, b.buttons[1], "$Q$ first: $A = P'Q$", 12.0, order == Order::TurnFirst);
        plot::button(f, b.buttons[2], if busy { "❚❚" } else { "▶ play" }, 12.0, false);

        self.chart(f, b, a, &pol);
        if let Some(at) = b.key {
            if b.key_lines == 1 {
                f.text_px(at, KEY_LINES.join(APART), 11.5, pal::DIM, Align2::LEFT_TOP);
            } else {
                for (i, line) in KEY_LINES.iter().enumerate() {
                    f.text_px(at + Vec2::new(0.0, i as f32 * 16.0), *line, 11.5, pal::DIM, Align2::LEFT_TOP);
                }
            }
        }
    }

    /// The chart: how far A is from the turn by t (solid) and from the mirror that sends
    /// (1, 0) to the angle t (dashed), for every t; Q at the lowest point.
    fn chart(&self, f: &mut Frame, b: &Board, a: DMat2, pol: &Polar) {
        if b.title_lines == 1 {
            f.text_px(b.title, TITLE_LINES.join(" "), 11.5, pal::TEXT, Align2::LEFT_TOP);
        } else {
            for (i, line) in TITLE_LINES.iter().enumerate() {
                f.text_px(b.title + Vec2::new(0.0, i as f32 * LINE_H), *line, 11.5, pal::TEXT, Align2::LEFT_TOP);
            }
        }
        let chart = Plot::new(b.chart, [-180.0, 180.0], [0.0, self.ceiling]);
        if chart.tiny() {
            return;
        }
        chart.y_grid(f);
        chart.y_axis(f, "", 10.5);
        // the angle along the bottom, in degrees, every quarter turn
        let (min, max) = b.chart;
        plot::line_of(f, plot::AXES, Vec2::new(min.x, max.y), max, 1.0, pal::DIM.a(0.8));
        for t in [-180.0, -90.0, 0.0, 90.0, 180.0] {
            let x = chart.px_x(t);
            plot::line_of(f, plot::AXES, Vec2::new(x, max.y), Vec2::new(x, max.y + 4.0), 1.0, pal::DIM.a(0.8));
            f.text_px(Vec2::new(x, max.y + 6.0), format!("{}°", num(t, 0)), 10.5, pal::DIM, Align2::CENTER_TOP);
        }
        f.text_px(Vec2::new(max.x, max.y + 18.0), "$t$", 11.0, pal::DIM, Align2::RIGHT_TOP);
        const N: usize = 181;
        let at = |i: usize| -180.0 + 360.0 * i as f64 / (N - 1) as f64;
        chart.path(f, plot::CURVES, (0..N).map(|i| (at(i), to_turn(a, at(i).to_radians()))), 1.8, pal::BLUE);
        // dashed: every other run of four steps
        let pts: Vec<Vec2> = (0..N).map(|i| chart.at(at(i), to_mirror(a, at(i).to_radians())).max(min).min(max)).collect();
        let mut dashes = Vec::with_capacity(N + N / 4);
        for (i, p) in pts.iter().enumerate() {
            if i % 8 == 4 {
                dashes.push(Vec2::NAN);
            }
            if i % 8 < 4 || i % 8 == 0 {
                dashes.push(*p);
            }
        }
        plot::path_of(f, MIRRORS, &dashes, 1.6, pal::BLUE.a(0.8));

        // Q, at the lowest point; and for a flat A the other Q, as low, or for a mirror
        // the nearest turn. Each name beside its dot, on the side away from the other
        // dot (and from the chart's edge).
        let gap_text = format!(r"$Q$: ${}$", num(pol.gap(), 2));
        if pol.pinned == Pinned::Any {
            f.text_px((min + max) * 0.5 - Vec2::new(0.0, 10.0), "both flat: every $Q$ is as near", 11.5, pal::TEXT, Align2::CENTER_BOTTOM);
        } else {
            let at = |t: f64, d: f64| chart.at(t.to_degrees(), d).max(min).min(max);
            let q = at((self.marker.get(0) as f64).to_radians(), self.marker.get(1) as f64);
            let second = if let Some(other) = pol.other_q(a) {
                Some((at(other.x_axis.y.atan2(other.x_axis.x), distance(a, other)), "or this one".to_string(), pal::WHITE.a(0.7), pal::DIM))
            } else if matches!(pol.orth, Orth::Mirror(_)) {
                match nearest_turn(a) {
                    (Some(t), d) => Some((at(t, d), format!("nearest turn: ${}$", num(d, 2)), pal::BLUE, pal::BLUE)),
                    _ => None,
                }
            } else {
                None
            };
            let middle = (min.x + max.x) * 0.5;
            let q_side = match &second {
                Some((p, ..)) if (p.x - q.x).abs() > 1.0 => (q.x - p.x).signum(),
                _ => -(q.x - middle).signum(),
            };
            let name = |f: &mut Frame, p: Vec2, side: f32, text: &str, c: Color| {
                // (turned inwards where the chart's edge is too near)
                let side = if (side < 0.0 && p.x - min.x < 70.0) || (side > 0.0 && max.x - p.x < 70.0) { -side } else { side };
                let align = if side > 0.0 { Align2::LEFT_BOTTOM } else { Align2::RIGHT_BOTTOM };
                f.caption_px(p + Vec2::new(8.0 * side, -5.0), text, 11.5, c, align);
            };
            plot::line_of(f, MARKS, Vec2::new(q.x, max.y), q, 2.0, pal::RED.a(0.85));
            plot::disc(f, q, 8.0, pal::WHITE);
            name(f, q, q_side, &gap_text, pal::TEXT);
            if let Some((p, text, dot, c)) = second {
                plot::disc(f, p, 7.0, dot);
                name(f, p, -q_side, &text, c);
            }
        }
        // (for A = 0 there is no lowest point to mark)
        keep_batch(plot::apart(f, MARKS, false));
    }
}

/// What the orange (k = 0) and purple (k = 1) arrows are called once `done` steps of
/// the film of `order` have been played.
fn arrow_name(order: Order, k: usize, done: usize) -> &'static str {
    const STRETCH_FIRST: [[&str; 3]; 2] = [["$v_1$", r"$\sigma_1 v_1$", r"$\sigma_1 u_1$"], ["$v_2$", r"$\sigma_2 v_2$", r"$\sigma_2 u_2$"]];
    const TURN_FIRST: [[&str; 3]; 2] = [["$v_1$", "$u_1$", r"$\sigma_1 u_1$"], ["$v_2$", "$u_2$", r"$\sigma_2 u_2$"]];
    let names = if order == Order::StretchFirst { &STRETCH_FIRST } else { &TURN_FIRST };
    names[k.min(1)][done.min(2)]
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

#[cfg(test)]
mod tests {
    use super::*;

    /// The mirror that sends (1, 0) to the direction at the angle t (in the line at t/2).
    fn mirror(t: f64) -> DMat2 {
        let (s, c) = t.sin_cos();
        rows([c, s, s, -c])
    }

    fn close(m: DMat2, n: DMat2, tol: f64) -> bool {
        (m - n).to_cols_array().iter().all(|v| v.abs() <= tol)
    }

    fn size(m: DMat2) -> f64 {
        m.to_cols_array().iter().fold(0.0f64, |s, v| s.max(v.abs()))
    }

    /// Random matrices of every size, and the awkward ones: flat, nearly flat, a multiple
    /// of a turn or a mirror (σ₁ = σ₂), zero, tiny, and the largest the handles allow.
    fn matrices() -> Vec<DMat2> {
        let mut rng = Rng::new(7);
        let mut out: Vec<DMat2> = TRIALS.iter().map(|t| rows(t.a)).collect();
        for k in 0..400 {
            let scale = [0.05, 0.5, 1.0, 2.0][k % 4];
            out.push(rows([0; 4].map(|_| rng.normal() as f64 * scale)));
        }
        out.extend([
            rows([1.0, 2.0, 2.0, 4.0]),
            rows([0.0, 1.0, 0.0, 0.0]),
            rows([1.0, 0.0, 0.0, 0.0]),
            rows([1.0, 2.0, 2.0, 4.0 + 1e-7]),
            rows([1.0, 2.0, 2.0, 4.0 - 1e-7]),
            turn(0.7) * 1.5,
            mirror(2.0) * 0.4,
            DMat2::IDENTITY,
            DMat2::IDENTITY * -1.0,
            rows([1.0, 0.0, 0.0, -1.0]),
            DMat2::IDENTITY * 1e-9,
            rows([ENTRY_MAX, ENTRY_MAX, ENTRY_MAX, -ENTRY_MAX]),
            rows([ENTRY_MAX; 4]),
            rows([-ENTRY_MAX, ENTRY_MAX, -ENTRY_MAX, -ENTRY_MAX]),
        ]);
        out
    }

    /// A = QP = P′Q, Q orthogonal (a turn if det A > 0, a mirror if det A < 0), P and
    /// P′ symmetric with eigenvalues σ₁ ≥ σ₂ ≥ 0 along v₁ and its quarter turn, P′ =
    /// QPQᵀ; σ₁ and σ₂ are the singular values (σ₁² + σ₂² = ‖A‖², σ₁σ₂ = |det A|); and
    /// for an invertible A, Q = AP⁻¹.
    #[test]
    fn the_polar_form_is_right_for_every_matrix() {
        for a in matrices() {
            for other in [false, true] {
                let pol = polar(a, other);
                let scale = size(a).max(1e-12);
                let tol = 1e-12 * scale.max(1.0);
                assert!(close(pol.q * pol.p, a, tol) && close(pol.p2 * pol.q, a, tol), "{a:?}: {pol:?}");
                assert!(close(pol.q.transpose() * pol.q, DMat2::IDENTITY, 1e-12), "{a:?}");
                assert!(close(pol.p, pol.p.transpose(), 0.0) && close(pol.p2, pol.p2.transpose(), 0.0));
                assert!(close(pol.q * pol.p * pol.q.transpose(), pol.p2, tol), "{a:?}");
                // P = √(AᵀA), P′ = √(AAᵀ)
                let tol2 = 1e-12 * scale.max(1.0).powi(2);
                assert!(close(pol.p * pol.p, a.transpose() * a, tol2) && close(pol.p2 * pol.p2, a * a.transpose(), tol2), "{a:?}");
                // σ₁ and σ₂ are the most and the least A stretches any direction
                for k in 0..36 {
                    let x = DVec2::from_angle(k as f64 * PI64 / 36.0);
                    assert!((a * x).length() <= pol.s1 + tol && (a * x).length() >= pol.s2 - tol, "{a:?}");
                }
                assert!(pol.s1 >= pol.s2 && pol.s2 >= 0.0);
                let (v1, v2) = (pol.v1, pol.v1.perp());
                assert!((pol.p * v1 - v1 * pol.s1).length() < tol && (pol.p * v2 - v2 * pol.s2).length() < tol, "{a:?}");
                assert!((pol.s1 * pol.s1 + pol.s2 * pol.s2 - squares(a)).abs() < 1e-12 * scale.max(1.0).powi(2));
                assert!((pol.s1 * pol.s2 - a.determinant().abs()).abs() < 1e-12 * scale.max(1.0).powi(2));
                let det = a.determinant();
                match pol.pinned {
                    Pinned::One => {
                        assert!(matches!(pol.orth, Orth::Turn(_)) == (det > 0.0), "{a:?}");
                        assert!(close(a * pol.p.inverse(), pol.q, 1e-9 * (scale / pol.s2).max(1.0)), "{a:?}");
                    }
                    Pinned::Two => assert!(matches!(pol.orth, Orth::Mirror(_)) == other && det.abs() <= FLAT * squares(a)),
                    Pinned::Any => assert!(squares(a) <= ZERO && pol.q == DMat2::IDENTITY),
                }
                // the angle the panel gives Q is Q's
                let want = match pol.orth {
                    Orth::Turn(t) => turn(t),
                    Orth::Mirror(line) => mirror(2.0 * line),
                };
                assert!(close(want, pol.q, 1e-12), "{a:?}");
            }
        }
    }

    /// Q is the orthogonal matrix nearest to A, at the distance ‖P − I‖, and on the
    /// chart's curve of its own kind at its own angle. If det A < 0 the nearest turn is
    /// another matrix, at √((σ₁ − 1)² + (σ₂ + 1)²): the turn that takes v₁ to u₁ and v₂
    /// to −u₂.
    #[test]
    fn q_is_the_nearest_orthogonal_matrix() {
        let grid: Vec<f64> = (0..=3600).map(|i| -PI64 + 2.0 * PI64 * i as f64 / 3600.0).collect();
        for a in matrices() {
            let pol = polar(a, false);
            let gap = pol.gap();
            assert!((distance(a, pol.q) - gap).abs() < 1e-12 * size(a).max(1.0), "{a:?}");
            for t in &grid {
                assert!((distance(a, turn(*t)) - to_turn(a, *t)).abs() < 1e-9 && (distance(a, mirror(*t)) - to_mirror(a, *t)).abs() < 1e-9);
                assert!(to_turn(a, *t) >= gap - 1e-9 && to_mirror(a, *t) >= gap - 1e-9, "{a:?} at {t}");
            }
            let on_curve = match pol.orth {
                Orth::Turn(_) => to_turn(a, pol.q_angle()),
                Orth::Mirror(_) => to_mirror(a, pol.q_angle()),
            };
            assert!((on_curve - gap).abs() < 1e-7, "{a:?}");
            if a.determinant() < 0.0 && pol.pinned == Pinned::One {
                let (Some(t), d) = nearest_turn(a) else { continue };
                assert!((d - ((pol.s1 - 1.0).powi(2) + (pol.s2 + 1.0).powi(2)).sqrt()).abs() < 1e-9, "{a:?}");
                assert!(grid.iter().all(|s| to_turn(a, *s) >= d - 1e-9) && (to_turn(a, t) - d).abs() < 1e-9);
                assert!(d > gap);
                let (v1, v2) = (pol.v1, pol.v1.perp());
                assert!((turn(t) * v1 - pol.q * v1).length() < 1e-9 && (turn(t) * v2 + pol.q * v2).length() < 1e-9, "{a:?}");
            }
        }
    }

    /// A flat A has two Qs, a turn and a mirror, which both give A = QP and are as near
    /// to A as each other; P is the same for both. For A = 0 every Q will do.
    #[test]
    fn a_flat_matrix_has_two_qs() {
        for a in [rows(TRIALS[4].a), rows([1.0, 2.0, 2.0, 4.0]), rows([0.0, 1.0, 0.0, 0.0]), rows([-0.3, 0.0, 0.9, 0.0])] {
            let (one, two) = (polar(a, false), polar(a, true));
            assert!(one.pinned == Pinned::Two && two.pinned == Pinned::Two);
            assert!(matches!(one.orth, Orth::Turn(_)) && matches!(two.orth, Orth::Mirror(_)));
            assert!(close(one.p, two.p, 1e-15) && one.s2 < 1e-9);
            for q in [one.q, two.q] {
                assert!(close(q * one.p, a, 1e-12) && close(q.transpose() * q, DMat2::IDENTITY, 1e-12));
            }
            assert!((distance(a, one.q) - distance(a, two.q)).abs() < 1e-9);
            assert!(close(one.other_q(a).unwrap_or(DMat2::ZERO), two.q, 1e-15));
        }
        let zero = polar(DMat2::ZERO, false);
        assert!(zero.pinned == Pinned::Any && zero.p == DMat2::ZERO && zero.gap() == 2f64.sqrt());
        for t in [0.0, 1.0, 2.5] {
            assert!(close(turn(t) * zero.p, DMat2::ZERO, 0.0) && (to_turn(DMat2::ZERO, t) - 2f64.sqrt()).abs() < 1e-15 && (to_mirror(DMat2::ZERO, t) - 2f64.sqrt()).abs() < 1e-15);
        }
    }

    /// Each film starts at I and ends at A, and passes through P or Q at its middle; the
    /// stretch keeps its axes all the way, the turn keeps lengths, and the mirror is
    /// the mirror at the end of its step.
    #[test]
    fn the_films_go_from_i_through_a_factor_to_a() {
        for a in matrices() {
            let pol = polar(a, false);
            for order in [Order::StretchFirst, Order::TurnFirst] {
                assert!(close(staged(a, &pol, order, 0.0), DMat2::IDENTITY, 0.0));
                assert!(close(staged(a, &pol, order, 1.0), halfway(&pol, order), 1e-12 * size(a).max(1.0)), "{a:?}");
                assert!(close(staged(a, &pol, order, 2.0 - 1e-9), a, 1e-7 * size(a).max(1.0)), "{a:?}");
                assert_eq!(staged(a, &pol, order, 2.0), a);
                for i in 0..=40 {
                    assert!(staged(a, &pol, order, i as f64 / 20.0).is_finite());
                }
            }
            let (v1, v2) = (pol.v1, pol.v1.perp());
            for i in 0..=10 {
                let s = i as f64 / 10.0;
                assert!((stretch_part(pol.p, s) * v1).perp_dot(v1).abs() < 1e-12 * size(a).max(1.0) && (stretch_part(pol.p, s) * v2).perp_dot(v2).abs() < 1e-12 * size(a).max(1.0));
                if let Orth::Turn(_) = pol.orth {
                    assert!((q_part(&pol, s) * DVec2::new(0.3, -1.2)).length() - DVec2::new(0.3, -1.2).length() < 1e-12);
                }
            }
            assert!(close(q_part(&pol, 1.0), pol.q, 1e-12), "{a:?}");
        }
    }

    /// The scene as an experiment's button leaves it, run at 60 frames a second until it
    /// has come to rest.
    fn run(k: usize) -> PolarForm {
        let mut scene = PolarForm::new();
        // (from the end of another film, as a visitor would come to it)
        for _ in 0..400 {
            scene.advance(1.0 / 60.0, 1.0 / 60.0);
        }
        scene.set_up(k);
        for _ in 0..60 * 20 {
            scene.advance(1.0 / 60.0, 1.0 / 60.0);
            if scene.at_rest() {
                break;
            }
        }
        assert!(scene.at_rest(), "experiment {k} has not come to rest");
        scene
    }

    fn angle_of(v: DVec2) -> String {
        degrees(v.to_angle())
    }

    /// The captions of the "things to try", held to account.
    #[test]
    fn stretch_then_turn_does_what_its_caption_says() {
        let scene = run(0);
        let Setting { a, progress } = scene.shown();
        assert!(a == rows(FIRST) && progress == 2.0 && scene.order == Order::StretchFirst);
        let pol = scene.polar();
        assert_eq!([num(pol.s1, 2), num(pol.s2, 2)], ["1.77", "0.63"]);
        assert!(matches!(pol.orth, Orth::Turn(t) if degrees(t) == "45.0°"));
        // the arrows on the axes stay on their lines all through the stretch
        let (v1, v2) = scene.axes_drawn();
        assert!((v1 - pol.v1).length() < 1e-12 || (v1 + pol.v1).length() < 1e-12);
        for i in 0..=20 {
            let m = staged(a, &pol, Order::StretchFirst, i as f64 / 20.0);
            assert!((m * v1).perp_dot(v1).abs() < 1e-12 && (m * v2).perp_dot(v2).abs() < 1e-12);
        }
        // then Q turns all of it, and the faint shape is the plane after P alone
        assert!(close(staged(a, &pol, Order::StretchFirst, 2.0), turn(PI64 / 4.0) * halfway(&pol, Order::StretchFirst), 1e-12));
        assert!(close(halfway(&pol, Order::StretchFirst), pol.p, 0.0));
    }

    #[test]
    fn turn_then_stretch_does_what_its_caption_says() {
        let scene = run(1);
        let Setting { a, progress } = scene.shown();
        assert!(a == rows(FIRST) && progress == 2.0 && scene.order == Order::TurnFirst);
        let pol = scene.polar();
        assert!(matches!(pol.orth, Orth::Turn(t) if degrees(t) == "45.0°") && close(halfway(&pol, Order::TurnFirst), pol.q, 0.0));
        // P' stretches by the same amounts along u1 = Qv1 and u2 = Qv2, at 75.1 and 165.1 degrees
        let (u1, u2) = (pol.q * pol.v1, pol.q * pol.v1.perp());
        assert!((pol.p2 * u1 - u1 * pol.s1).length() < 1e-12 && (pol.p2 * u2 - u2 * pol.s2).length() < 1e-12);
        assert_eq!([angle_of(u1), angle_of(u2)], ["75.1°", "165.1°"]);
        // the dashed axes drawn are those
        let axis = DVec2::from_angle(scene.axis_angle.shown);
        assert!(axis.perp_dot(u1).abs() < 1e-9);
        assert!(close(pol.p2, pol.q * pol.p * pol.q.transpose(), 1e-12) && close(staged(a, &pol, Order::TurnFirst, 2.0), staged(a, &pol, Order::StretchFirst, 2.0), 0.0));
    }

    #[test]
    fn the_nearest_turn_does_what_its_caption_says() {
        let scene = run(2);
        let a = scene.shown().a;
        let pol = scene.polar();
        let (c1, c2) = (a.col(0), a.col(1));
        assert_eq!([num(c1.length(), 2), num(c2.length(), 2), num(c1.angle_to(c2).to_degrees(), 0)], ["1.08", "0.89", "83"]);
        assert!(matches!(pol.orth, Orth::Turn(t) if degrees(t) == "30.5°"));
        assert_eq!([num(pol.s1, 2), num(pol.s2, 2), num(pol.gap(), 2), num(distance(a, pol.q), 2)], ["1.10", "0.87", "0.16", "0.16"]);
        // Q is the lowest point of the chart, of both curves
        let lowest = (0..=3600).map(|i| -PI64 + 2.0 * PI64 * i as f64 / 3600.0).map(|t| to_turn(a, t).min(to_mirror(a, t))).fold(f64::MAX, f64::min);
        assert!((lowest - pol.gap()).abs() < 1e-4 && lowest >= pol.gap() - 1e-12);
    }

    #[test]
    fn the_mirror_does_what_its_caption_says() {
        let scene = run(3);
        let a = scene.shown().a;
        let pol = scene.polar();
        // the first matrix followed by the mirror in the y axis
        assert!(a == rows([-1.0, 0.0, 0.0, 1.0]) * rows(FIRST) && num(a.determinant(), 2) == "−1.12");
        assert!(matches!(pol.orth, Orth::Mirror(l) if degrees(l) == "67.5°"));
        assert!(close(pol.p, polar(rows(FIRST), false).p, 1e-15));
        // it turns the F over: its corners go round the other way
        let turning = |m: DMat2| (0..F_OUTLINE.len()).map(|i| (m * DVec2::from(F_OUTLINE[i])).perp_dot(m * DVec2::from(F_OUTLINE[(i + 1) % F_OUTLINE.len()]))).sum::<f64>();
        assert!(turning(DMat2::IDENTITY) > 0.0 && turning(a) < 0.0);
        // the lowest point is on the dashed curve, at 0.86; the nearest turn at 1.81, by 74.7 degrees
        assert!((to_mirror(a, pol.q_angle()) - pol.gap()).abs() < 1e-9 && num(pol.gap(), 2) == "0.86");
        let (Some(t), d) = nearest_turn(a) else { panic!("no nearest turn") };
        assert_eq!([num(d, 2), degrees(t)], ["1.81", "74.7°"]);
        let (v1, v2) = (pol.v1, pol.v1.perp());
        assert!((turn(t) * v1 - pol.q * v1).length() < 1e-12 && (turn(t) * v2 + pol.q * v2).length() < 1e-12);
    }

    #[test]
    fn the_squash_does_what_its_caption_says() {
        let scene = run(4);
        let a = scene.shown().a;
        assert!(a.col(1) == a.col(0) * 0.5 && a.determinant().abs() < 1e-15);
        let pol = scene.polar();
        assert!(pol.pinned == Pinned::Two && pol.s2 < 1e-9 && num(pol.s2, 2) == "0.00");
        // P presses the circle flat along v1
        assert!((0..RING).all(|i| (pol.p * ring_point(i)).perp_dot(pol.v1).abs() < 1e-9));
        // the turn by 29.7 degrees and the mirror in the line at 41.4 both take v1 to u1
        let u1 = a * pol.v1 / pol.s1;
        let mirror = polar(a, true);
        assert!(matches!(pol.orth, Orth::Turn(t) if degrees(t) == "29.7°") && matches!(mirror.orth, Orth::Mirror(l) if degrees(l) == "41.4°"));
        for q in [pol.q, mirror.q] {
            assert!((q * pol.v1 - u1).length() < 1e-12 && close(q * pol.p, a, 1e-12));
        }
        // and both curves come down to 1.17
        let lowest = |g: fn(DMat2, f64) -> f64| (0..=7200).map(|i| g(a, -PI64 + 2.0 * PI64 * i as f64 / 7200.0)).fold(f64::MAX, f64::min);
        assert_eq!([num(lowest(to_turn), 2), num(lowest(to_mirror), 2)], ["1.17", "1.17"]);
        // ticking "the other Q" plays it with the mirror, which ends in the same place
        let mut scene = scene;
        scene.other = true;
        assert!(matches!(scene.polar().orth, Orth::Mirror(_)) && close(staged(a, &scene.polar(), Order::StretchFirst, 2.0 - 1e-9), a, 1e-7));
    }

    /// The words: near I the polar form and the symmetric and skew parts agree to first
    /// order; and the SVD gives the two factors.
    #[test]
    fn the_words_are_true() {
        let mut rng = Rng::new(5);
        for _ in 0..50 {
            let e = rows([0; 4].map(|_| rng.normal() as f64));
            let (s, k) = ((e + e.transpose()) * 0.5, (e - e.transpose()) * 0.5);
            for eps in [1e-2, 1e-3] {
                let pol = polar(DMat2::IDENTITY + e * eps, false);
                // (what is left is of the size of ε²)
                let left = size(pol.p - DMat2::IDENTITY - s * eps).max(size(pol.q - DMat2::IDENTITY - k * eps));
                assert!(left < 3.0 * eps * eps * size(e).max(1.0).powi(2), "{e:?} at {eps}: {left}");
            }
        }
        // from an SVD, P = V Σ Vᵀ and Q = U Vᵀ
        for a in matrices().into_iter().filter(|a| a.determinant().abs() > 1e-6) {
            let pol = polar(a, false);
            let v = DMat2::from_cols(pol.v1, pol.v1.perp());
            let u = DMat2::from_cols(a * pol.v1 / pol.s1, a * pol.v1.perp() / pol.s2);
            let sigma = DMat2::from_diagonal(DVec2::new(pol.s1, pol.s2));
            assert!(close(u * sigma * v.transpose(), a, 1e-9 * size(a).max(1.0)) && close(u.transpose() * u, DMat2::IDENTITY, 1e-9));
            assert!(close(v * sigma * v.transpose(), pol.p, 1e-9 * size(a).max(1.0)) && close(u * v.transpose(), pol.q, 1e-9), "{a:?}");
        }
    }

    /// Whatever is asked, by buttons, typed numbers or hands, every number drawn stays
    /// finite and the scene comes to rest.
    #[test]
    fn degenerate_input_leaves_everything_finite_and_comes_to_rest() {
        for a in matrices().into_iter().chain([rows([f64::NAN, 1.0, 0.0, 1.0]), rows([1e30, 0.0, 0.0, -1e30])]) {
            for order in [Order::StretchFirst, Order::TurnFirst] {
                let mut scene = PolarForm::new();
                scene.ask(a, order);
                for _ in 0..60 * 20 {
                    scene.advance(1.0 / 60.0, 1.0 / 60.0);
                    if scene.at_rest() {
                        break;
                    }
                }
                assert!(scene.at_rest(), "{a:?}");
                let shown = scene.shown().a;
                assert!(size(shown) <= ENTRY_MAX && shown.is_finite());
                let pol = scene.polar();
                for i in 0..=20 {
                    let m = staged(shown, &pol, order, i as f64 / 10.0);
                    assert!(m.is_finite() && (m * DVec2::ONE).as_vec2().is_finite());
                }
                assert!(scene.ceiling.is_finite() && scene.ceiling >= ((pol.s1 + 1.0).powi(2) + (pol.s2 + 1.0).powi(2)).sqrt());
                assert!(scene.v_angle.shown.is_finite() && scene.axis_angle.shown.is_finite());
                for t in [-3.0, 0.0, 1.0] {
                    assert!(to_turn(shown, t).is_finite() && to_mirror(shown, t).is_finite());
                }
                let words = [q_words(&pol, shown), degrees(pol.q_angle()), num(pol.gap(), 2), num(pol.s1, 2), num(pol.s2, 2)].join(" ");
                assert!(!words.contains("NaN") && !words.contains("inf") && !words.contains('?'), "{a:?}: {words}");
                // a hand on a column, then let go: at rest again, at the end of the film
                scene.hand(shown * 0.5 + rows([0.0, 0.1, 0.0, 0.0]));
                for _ in 0..60 * 5 {
                    scene.advance(1.0 / 60.0, 1.0 / 60.0);
                }
                assert!(scene.at_rest() && scene.shown().progress == 2.0);
            }
        }
    }

    /// A new matrix waits until the film is back at its start, where the plane is the
    /// same whatever A is: the picture never jumps from one matrix's film to another's.
    #[test]
    fn a_new_matrix_waits_for_the_film_to_be_wound_back() {
        let mut scene = PolarForm::new();
        for _ in 0..400 {
            scene.advance(1.0 / 60.0, 1.0 / 60.0);
        }
        let first = scene.shown().a;
        scene.set_up(3);
        let mut last = staged(first, &scene.polar(), scene.order, scene.shown().progress);
        let mut moved = false;
        for _ in 0..600 {
            scene.advance(1.0 / 60.0, 1.0 / 60.0);
            if scene.shown().a != first && !moved {
                // (the first frame of the new matrix is at the start of the film)
                assert!(scene.shown().progress == 0.0 && scene.order == Order::StretchFirst, "{}", scene.shown().progress);
                moved = true;
            }
            let now = staged(scene.shown().a, &scene.polar(), scene.order, scene.shown().progress);
            // (the rewind and the films are eases: never a jump from one picture to another)
            assert!(size(now - last) < 0.2, "{now:?} after {last:?}");
            last = now;
        }
        assert!(scene.at_rest() && scene.shown().a == rows(TRIALS[3].a));
    }

    /// The panel and the picture fit the canvas, from a phone to a desktop, and the chart
    /// and the buttons stay inside the panel.
    #[test]
    fn the_layout_fits_every_canvas() {
        for (w, h) in [(1080.0, 1000.0), (580.0, 700.0), (390.0, 800.0), (390.0, 560.0), (800.0, 600.0), (1600.0, 1000.0), (240.0, 180.0), (620.0, 480.0), (1300.0, 700.0)] {
            for inset in [0.0, 46.0] {
                let view = Vec2::new(w, h);
                let lay = Layout::of(view, inset);
                let (min, max) = lay.picture;
                assert!(min.cmpge(Vec2::ZERO).all() && max.cmple(view).all() && (max - min).min_element() > 50.0, "{w}x{h}: {lay:?}");
                if let Some(b) = &lay.board {
                    let within = |q: Vec2| q.cmpge(b.min - Vec2::splat(0.5)).all() && q.cmple(b.max + Vec2::splat(0.5)).all();
                    assert!(b.min.cmpge(Vec2::ZERO).all() && b.max.cmple(view + Vec2::splat(0.5)).all(), "{w}x{h}: {b:?}");
                    assert!(within(b.chart.0) && within(b.chart.1 + Vec2::new(0.0, 30.0)) && b.buttons.iter().all(|r| within(r.0) && within(r.1)), "{w}x{h}: {b:?}");
                    assert!(b.min.y >= inset || lay.narrow, "{w}x{h}: under the buttons");
                    let across = 3.0 * matrix_size(2, 2, CELL, b.text).x + 46.0;
                    assert!(b.mats.x + across <= b.max.x, "{w}x{h}: the matrices run past the panel");
                }
            }
        }
    }
}

//! Projections and reflections: a matrix that does nothing more the second time
//! (P² = P), which splits every point into the part it keeps and the part it kills;
//! and the mirror made from it, H = 2P − I, which undoes itself (H² = I).

use super::*;
use glam::{DMat2, DVec2};
use std::f64::consts::{FRAC_PI_2, PI as PI64, TAU as TAU64};

pub const DEF: SceneDef = SceneDef {
    id: "project",
    title: "Projections and reflections",
    group: SPECIAL,
    blurb: "Matrices that do nothing more the second time, and matrices that undo themselves: shadows and mirrors.",
    make: |_| Box::new(Project::create()),
};

// ----- the arithmetic, in double precision -------------------------------------------

/// The least angle between `w` and the line (radians). Along the line itself there is
/// no projection at all: a point off the line, moved along `w`, never reaches it. Near
/// it, P stretches by up to 1/sin θ, 28.65 at this angle.
const LEAST: f64 = 2.0 * PI64 / 180.0;
/// `w` snaps to the perpendicular (the orthogonal projection) within this angle of it.
const SNAP: f64 = 4.0 * PI64 / 180.0;
/// Below this angle between `w` and the line, the picture warns that P is large.
const WARN: f64 = 20.0 * PI64 / 180.0;

/// How far from 0 the knobs that turn the line (and the mirrors) and `w` are.
const KNOB_U: f64 = 3.0;
const KNOB_W: f64 = 1.6;
/// How far the point x may be dragged, and q along the line.
const X_MAX: f64 = 6.0;
const Q_MAX: f64 = 3.6;

/// A matrix by its rows, [[a, b], [c, d]], the way matrices are written.
#[cfg(test)]
fn rows([a, b, c, d]: [f64; 4]) -> DMat2 {
    DMat2::from_cols(DVec2::new(a, c), DVec2::new(b, d))
}

/// The projection onto the line along `u` that sends `w` to 0 (unit vectors, not along
/// one line): P = u nᵀ / (n · u), with n at right angles to w. Then Pu = u, Pw = 0
/// (since n · w = 0), and P² = u (nᵀu) nᵀ / (n · u)² = P.
fn projection(u: DVec2, w: DVec2) -> DMat2 {
    let n = w.perp();
    let d = n.dot(u);
    DMat2::from_cols(u * (n.x / d), u * (n.y / d))
}

/// The orthogonal projection onto the line along `u`: u uᵀ / (uᵀ u), symmetric to the
/// last bit (the formula above, with n along u, is so only up to rounding).
fn orthogonal(u: DVec2) -> DMat2 {
    let l = u.length_squared();
    let (a, b, c) = (u.x * u.x / l, u.x * u.y / l, u.y * u.y / l);
    DMat2::from_cols(DVec2::new(a, b), DVec2::new(b, c))
}

/// The reflection made from a projection: H = 2P − I keeps what P keeps and reverses
/// what P kills, and H² = 4P² − 4P + I = I.
fn reflection(p: DMat2) -> DMat2 {
    p * 2.0 - DMat2::IDENTITY
}

/// The size of the largest entry.
fn size(m: DMat2) -> f64 {
    m.to_cols_array().iter().fold(0.0f64, |s, v| s.max(v.abs()))
}

/// Is the matrix symmetric, as far as its numbers can tell?
fn symmetric(m: DMat2) -> bool {
    (m.y_axis.x - m.x_axis.y).abs() <= 1e-9 * size(m).max(1.0)
}

/// The angle between two lines (as an angle between directions, any size), in
/// (−π/2, π/2].
fn between_lines(a: f64) -> f64 {
    let r = a.rem_euclid(PI64);
    if r > FRAC_PI_2 {
        r - PI64
    } else {
        r
    }
}

/// `target` moved by whole periods to lie as near `now` as it can: an angle that
/// glides there turns the short way.
fn nearest(now: f64, target: f64, period: f64) -> f64 {
    target + period * ((now - target) / period).round()
}

/// The angle of `w` from the line when its knob is dragged to the direction `angle`:
/// kept on the knob's side of the line, between `LEAST` and π − `LEAST` (it cannot
/// cross the line, along which there is no projection), and at exactly π/2 within
/// `SNAP` of it.
fn rel_for(line: f64, angle: f64) -> f64 {
    let r = rel_unsnapped(line, angle);
    if (r - FRAC_PI_2).abs() < SNAP {
        FRAC_PI_2
    } else {
        r
    }
}

/// The same, without the snap to the perpendicular.
fn rel_unsnapped(line: f64, angle: f64) -> f64 {
    let r = (angle - line).rem_euclid(TAU64);
    let r = if r > PI64 {
        // (dragged past the line: held at whichever end of it is nearer)
        if r > 1.5 * PI64 {
            LEAST
        } else {
            PI64 - LEAST
        }
    } else {
        r
    };
    if r.is_finite() {
        r.max(LEAST).min(PI64 - LEAST)
    } else {
        FRAC_PI_2
    }
}

fn smooth(t: f64) -> f64 {
    let t = t.clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

/// What the picture shows: one line (a projection and its mirror), or two mirrors.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Mode {
    Line,
    Mirrors,
}

/// What glides when a button changes it.
#[derive(Clone, Copy, Debug, PartialEq)]
struct Setting {
    /// The line's direction, an angle (radians): the knob u is on it.
    line: f64,
    /// The angle of `w` from `u`, between `LEAST` and π − `LEAST`; exactly π/2 for the
    /// orthogonal projection.
    rel: f64,
    x: DVec2,
    /// Where q is on the line: q = s u, for s = this.
    q: f64,
    /// The second mirror's direction, an angle (radians).
    mirror: f64,
}

impl Setting {
    fn numbers(&self) -> [f64; 6] {
        [self.line, self.rel, self.x.x, self.x.y, self.q, self.mirror]
    }

    fn from_numbers(v: [f64; 6]) -> Setting {
        Setting { line: v[0], rel: v[1], x: DVec2::new(v[2], v[3]), q: v[4], mirror: v[5] }
    }

    fn u(&self) -> DVec2 {
        DVec2::from_angle(self.line)
    }

    fn w(&self) -> DVec2 {
        DVec2::from_angle(self.line + self.rel)
    }

    fn orthogonal(&self) -> bool {
        (self.rel - FRAC_PI_2).abs() < 1e-12
    }

    /// The angle between `w` and the line, in [`LEAST`, π/2].
    fn theta(&self) -> f64 {
        self.rel.min(PI64 - self.rel)
    }

    fn p(&self) -> DMat2 {
        if self.orthogonal() {
            orthogonal(self.u())
        } else {
            projection(self.u(), self.w())
        }
    }

    fn h(&self) -> DMat2 {
        reflection(self.p())
    }

    /// The reflections in the two mirrors (true mirrors, whatever `w` is).
    fn h1(&self) -> DMat2 {
        reflection(orthogonal(self.u()))
    }

    fn h2(&self) -> DMat2 {
        reflection(orthogonal(DVec2::from_angle(self.mirror)))
    }

    /// The angle from mirror 1 to mirror 2, in (−π/2, π/2].
    fn apart(&self) -> f64 {
        between_lines(self.mirror - self.line)
    }

    /// Where on the line the point nearest to x is: its foot is this times u.
    fn foot(&self) -> f64 {
        self.x.dot(self.u())
    }

    fn step(&self, s: Step) -> DMat2 {
        match s {
            Step::P => self.p(),
            Step::H => self.h(),
            Step::Mirror1 => self.h1(),
            Step::Mirror2 => self.h2(),
        }
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

    fn step(&mut self, dt: f64) {
        if self.arrived() || dt <= 0.0 {
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
        // (exactly there in the end: the picture then stops changing)
        if apart < 2e-4 || !apart.is_finite() {
            *self = Glide::at(self.to);
        } else {
            (self.between, self.shown) = (Setting::from_numbers(between), Setting::from_numbers(shown));
        }
    }
}

// ----- the films: a map acting twice ----------------------------------------------------

/// One map a film shows acting.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Step {
    P,
    H,
    Mirror1,
    Mirror2,
}

/// Seconds each map takes to act, and the pause between the two.
const STEP_SECONDS: f64 = 1.8;
const PAUSE: f64 = 0.7;
const FILM_SECONDS: f64 = 2.0 * STEP_SECONDS + PAUSE;

/// How far a film is `t` seconds in: 0 to 1 while the first map acts, 1 during the
/// pause, 1 to 2 while the second acts.
fn progress(t: f64) -> f64 {
    if t <= STEP_SECONDS {
        t.max(0.0) / STEP_SECONDS
    } else if t <= STEP_SECONDS + PAUSE {
        1.0
    } else {
        (1.0 + (t - STEP_SECONDS - PAUSE) / STEP_SECONDS).min(2.0)
    }
}

/// The map the plane has been carried through at `progress`, the maps `m` acting one
/// after the other. Each point moves in a straight line from where it is to where the
/// map sends it: under a projection, along `w` to the line; under a reflection, along
/// `w` through the line and as far again (half way, every point is at its shadow,
/// since (I + H)/2 = P).
fn carried(m: [DMat2; 2], progress: f64) -> DMat2 {
    let along = |m: DMat2, s: f64| DMat2::IDENTITY + (m - DMat2::IDENTITY) * smooth(s);
    if progress <= 1.0 {
        along(m[0], progress)
    } else {
        along(m[1], progress - 1.0) * m[0]
    }
}

/// How long a jump of the carried F (a new film, or none) takes to glide.
const MORPH_SECONDS: f64 = 0.5;

/// The sweep of q along the line, in an experiment: (seconds in, how far from the foot
/// of x), eased from each to the next.
const SWEEP: [(f64, f64); 3] = [(0.0, -3.0), (2.4, 1.4), (4.0, 0.0)];

fn sweep_at(t: f64) -> f64 {
    for k in 0..SWEEP.len() - 1 {
        let ((t0, a), (t1, b)) = (SWEEP[k], SWEEP[k + 1]);
        if t <= t1 {
            return a + (b - a) * smooth((t - t0) / (t1 - t0));
        }
    }
    SWEEP[SWEEP.len() - 1].1
}

// ----- the letter F ------------------------------------------------------------------

/// The F, in a box of its own from (0, 0) to (1, 1): its outline, and the three bars
/// it is filled with. Lopsided on purpose: turned, it is still an F; mirrored, it is not.
const F_OUTLINE: [[f64; 2]; 10] = [[0.0, 0.0], [0.3, 0.0], [0.3, 0.4], [0.75, 0.4], [0.75, 0.62], [0.3, 0.62], [0.3, 0.76], [1.0, 0.76], [1.0, 1.0], [0.0, 1.0]];
const F_BARS: [[[f64; 2]; 4]; 3] = [
    [[0.0, 0.0], [0.3, 0.0], [0.3, 1.0], [0.0, 1.0]],
    [[0.3, 0.76], [1.0, 0.76], [1.0, 1.0], [0.3, 1.0]],
    [[0.3, 0.4], [0.75, 0.4], [0.75, 0.62], [0.3, 0.62]],
];
/// The F's width and height in the plane.
const F_SIZE: [f64; 2] = [0.85, 1.15];

/// Where the F's middle is: to the left, above the line, in the picture of one line
/// (where its mirror image and its shadow stay in view however `w` slants); and above,
/// between the mirrors' reflections, in the picture of two.
fn f_centre(mode: Mode) -> DVec2 {
    match mode {
        Mode::Line => DVec2::new(-2.4, 0.0),
        Mode::Mirrors => DVec2::from_angle(1.35) * 2.1,
    }
}

/// The points that are carried by a film: the F's outline (10), its bars' corners
/// (12), and x.
const CARRIED: usize = 23;

fn f_points(mode: Mode) -> Vec<DVec2> {
    let c = f_centre(mode);
    let place = |[a, b]: [f64; 2]| c + DVec2::new((a - 0.5) * F_SIZE[0], (b - 0.5) * F_SIZE[1]);
    F_OUTLINE.iter().map(|p| place(*p)).chain(F_BARS.iter().flatten().map(|p| place(*p))).collect()
}

// ----- the experiments ---------------------------------------------------------------

/// Experiments: (button, what to look for). What each sets is `trial`.
const TRIALS: [(&str, &str); 5] = [
    (
        "the shadow of the shadow",
        r"Watch $P$ act: every point slides straight across to the line, along $w$, and the letter $F$ lands flat on its shadow $PF$. Then $P$ again: nothing moves, since every point is on the line already and $P$ keeps the line. So $P^2 = P$, and the screen shows the same four numbers twice. $P$ keeps $u$ ($\lambda = 1$) and sends $w$ to $0$ ($\lambda = 0$), and $\tr P = 1 + 0 = 1$: one line kept.",
    ),
    (
        "the nearest point",
        r"$q$ slides along the line. Its distance to $x$ (dashed, and charted under the matrices) is least, $1.79$, where $q$ passes $Px$: there $x - q$ meets the line at a right angle. So the orthogonal projection finds the nearest point of the line, which is what least squares does onto a plane. Drag $q$ yourself; then slant $w$, and $Px$ moves away from the nearest point.",
    ),
    (
        "a slanted projection",
        r"$w$ is now $45°$ from the line, not $90°$: every point slides along $w$, and the leftover $x - Px$ runs along $w$ too. Still $P^2 = P$ (watch it act twice), but $P^\top \ne P$: the numbers are not the same across the diagonal. And $P$ can lengthen: $|x| = 1.58$ but $|Px| = 2.24$, outside the dashed circle. That is $1/\sin 45° = 1.41$ times as long, the most at this angle. Turn $w$ towards the line, and the most $P$ can stretch, $1/\sin\theta$, grows without limit as the angle $\theta$ nears $0$ (this scene stops at 2°, where it is 28.65).",
    ),
    (
        "a mirror, twice",
        r"$H = 2P - I$ keeps the line and reverses the direction across it: $Hx = Px - (x - Px)$. This line is halfway between $x = (1.5, 2)$ and the $x$ axis, so $H$ sends $x$ onto the axis, to $(2.50, 0)$, at the same length: a Householder reflection, the step QR is built from in practice. Watch $H$ act: half way, every point is at its shadow, since $(I + H)/2 = P$. Then $H$ again: everything comes back, $H^2 = I$.",
    ),
    (
        "two mirrors make a turn",
        r"Mirror 2 is $0.30$ radians ($17.2°$) round from mirror 1. The letter $F$ is reflected in mirror 1, then in mirror 2. Each reflection turns it over; the two flips cancel, and what is left is a turn about $0$ by twice the angle between the mirrors: $H_2 H_1$ turns by $0.60$ radians ($34.4°$). In the other order, $H_1 H_2$, it turns the other way. Drag mirror 2: the turn stays twice the angle.",
    ),
];

/// What experiment `k` sets: the picture, the setting, the film, and q (shown, and swept).
struct Trial {
    mode: Mode,
    setting: Setting,
    film: Option<[Step; 2]>,
    q: bool,
    sweep: bool,
}

/// The line the scene starts with, along (2, 1): its projection is [[0.8, 0.4], [0.4, 0.2]].
fn first_line() -> f64 {
    1.0f64.atan2(2.0)
}

fn opening() -> Setting {
    let s = Setting { line: first_line(), rel: FRAC_PI_2, x: DVec2::new(1.0, 2.5), q: 0.0, mirror: 0.5 };
    Setting { q: s.foot() + SWEEP[0].1, ..s }
}

fn trial(k: usize) -> Trial {
    let open = opening();
    let line = |setting, film| Trial { mode: Mode::Line, setting, film, q: false, sweep: false };
    match k {
        0 => line(open, Some([Step::P, Step::P])),
        1 => Trial { q: true, sweep: true, ..line(open, None) },
        // w along (−3, 1), at 135° from u: P = [[0.4, 1.2], [0.2, 0.6]]; x at right
        // angles to w, which P stretches most
        2 => line(Setting { rel: 0.75 * PI64, x: DVec2::new(0.5, 1.5), ..open }, Some([Step::P, Step::P])),
        // x = (1.5, 2) at twice the line's angle: H sends it to (2.5, 0)
        3 => line(Setting { x: DVec2::new(1.5, 2.0), ..open }, Some([Step::H, Step::H])),
        _ => Trial { mode: Mode::Mirrors, setting: Setting { line: 0.2, mirror: 0.5, ..open }, film: Some([Step::Mirror1, Step::Mirror2]), q: false, sweep: false },
    }
}

// ----- where things go ---------------------------------------------------------------

/// The part of the scene the view is fitted to: what every experiment draws lies in it.
const FRAME_CENTRE: Vec2 = Vec2::new(0.35, -0.05);
const FRAME_HALF: Vec2 = Vec2::new(3.45, 2.85);
/// Above and to the right of this point of the frame nothing is drawn that matters in
/// any experiment (only the orange line runs on there): the panel may cover it.
const CORNER: Vec2 = Vec2::new(2.0, 1.7);

/// The screen panel's parts, in logical pixels.
#[derive(Clone, Debug)]
struct Panel {
    min: Vec2,
    max: Vec2,
    /// The size of the matrices' numbers.
    text: f32,
    /// Each matrix's name (the middle of its top) and its top-left corner.
    names: [Vec2; 4],
    mats: [Vec2; 4],
    /// The top-left of the lines under the matrices.
    lines: Vec2,
    /// The chart of |x − q|: its title, and its box.
    chart: Option<(Vec2, (Vec2, Vec2))>,
}

const PAD: f32 = 12.0;
const NAME_H: f32 = 19.0;
const LINE_H: f32 = 18.0;
const LINES: usize = 5;
const CHART_H: f32 = 84.0;
/// The widest number a matrix holds, in characters (a minus counted as two).
const CELL: usize = 6;
/// The panel's width on a wide canvas.
const PANEL_W: f32 = 300.0;

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

/// The most lines the caption takes on a phone.
const CAPTION_LINES: usize = 3;

impl Layout {
    fn of(view: Vec2, inset: f32, chart: bool) -> Layout {
        let narrow = view.x < 620.0 && view.y >= 560.0;
        let width = if narrow { view.x } else { PANEL_W };
        // the numbers' size: as large as lets two matrices side by side fit, up to 14
        let mut text = 14.0f32;
        while 2.0 * matrix_size(2, 2, CELL, text).x + 20.0 > width - 2.0 * PAD && text > 9.0 {
            text -= 0.5;
        }
        let mat = matrix_size(2, 2, CELL, text);
        let height = 2.0 * PAD + 2.0 * (NAME_H + mat.y + 8.0) + 6.0 + LINES as f32 * LINE_H + if chart { 8.0 + LINE_H + CHART_H } else { 0.0 };
        let place = |min: Vec2, width: f32| {
            let gap = 20.0;
            let left = min.x + (width - (2.0 * mat.x + gap)) * 0.5;
            let xs = [left, left + mat.x + gap];
            let mut y = min.y + PAD;
            let (mut names, mut mats) = ([Vec2::ZERO; 4], [Vec2::ZERO; 4]);
            for row in 0..2 {
                for col in 0..2 {
                    names[2 * row + col] = Vec2::new(xs[col] + mat.x * 0.5, y);
                    mats[2 * row + col] = Vec2::new(xs[col], y + NAME_H);
                }
                y += NAME_H + mat.y + 8.0;
            }
            y += 6.0;
            let lines = Vec2::new(min.x + PAD, y);
            y += LINES as f32 * LINE_H;
            let chart = chart.then(|| {
                y += 8.0;
                let title = Vec2::new(min.x + PAD, y);
                y += LINE_H;
                let chart = (Vec2::new(min.x + PAD + 6.0, y), Vec2::new(min.x + width - PAD - 6.0, y + CHART_H));
                y += CHART_H;
                (title, chart)
            });
            Panel { min, max: Vec2::new(min.x + width, y + PAD), text, names, mats, lines, chart }
        };
        let whole = |caption| Layout { narrow, picture: (Vec2::new(0.0, inset), view), panel: None, caption };
        if narrow {
            // the panel below the picture, with the caption at its foot
            let caption_h = CAPTION_LINES as f32 * 19.0 + 10.0;
            let below = view.y - height - caption_h;
            // (the picture keeps at least the top 45 %)
            if below < view.y * 0.45 {
                return whole(Some((0.0, view.x, view.y - 10.0)));
            }
            let mut panel = place(Vec2::new(0.0, below), view.x);
            panel.max = view;
            return Layout { narrow, picture: (Vec2::new(0.0, inset), Vec2::new(view.x, below)), panel: Some(panel), caption: Some((0.0, view.x, view.y - 10.0)) };
        }
        let top = inset + 12.0;
        if view.x < 640.0 || top + height + 12.0 > view.y {
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
        let foot = if self.narrow { 8.0 } else { 52.0 };
        (min + Vec2::new(12.0, 56.0), max - Vec2::new(12.0, foot))
    }
}

/// A number for a matrix on the screen, as a formula, in five characters at most (so
/// that the brackets never move as the numbers change).
fn cell(v: f64) -> String {
    let a = v.abs();
    let number = if !v.is_finite() {
        "?".into()
    } else if a < 9.995 {
        num(v, 2)
    } else if a < 99.95 {
        num(v, 1)
    } else {
        num(v, 0)
    };
    format!("${number}$")
}

/// An angle in degrees, to a tenth.
fn degrees(radians: f64) -> String {
    format!("{:.1}°", radians.to_degrees())
}

fn inside_rect(p: Vec2, (min, max): (Vec2, Vec2)) -> bool {
    p.cmpge(min).all() && p.cmple(max).all()
}

// ----- the scene ---------------------------------------------------------------------

pub struct Project {
    mode: Mode,
    glide: Glide,
    /// The film: the two maps it shows acting, how many seconds in it is, whether it
    /// plays, and whether it is to play once the glide to its setting has arrived.
    film: Option<[Step; 2]>,
    film_t: f64,
    playing: bool,
    play_when_there: bool,
    /// The carried points as last drawn, and, after a jump (a new film, or none), where
    /// they were then and how far the glide from there has got (0 to 1).
    drawn: Vec<DVec2>,
    morph: Option<(Vec<DVec2>, f64)>,
    /// The second point q on the line, and its sweep (seconds in).
    q_on: bool,
    sweep: Option<f64>,
    /// The canvas last drawn on and how much of its top the application's buttons
    /// cover: they decide where the panel goes, and so where the picture is.
    canvas: Vec2,
    inset: f32,
    measured: bool,
    new_view: bool,
    /// Whether the panel had the chart when the view was last fitted (it is taller then).
    charted: bool,
}

impl Project {
    /// (Nothing here needs the graphics device, so the tests can make one too.)
    fn create() -> Self {
        Project {
            mode: Mode::Line,
            glide: Glide::at(opening()),
            film: None,
            film_t: 0.0,
            playing: false,
            play_when_there: false,
            drawn: Vec::new(),
            morph: None,
            q_on: false,
            sweep: None,
            canvas: Vec2::new(1080.0, 1000.0),
            inset: 0.0,
            measured: false,
            new_view: false,
            charted: false,
        }
    }

    fn shown(&self) -> Setting {
        self.glide.shown
    }

    /// The carried points about to jump: glide them from where they are drawn.
    fn morph_from_here(&mut self) {
        if self.drawn.len() == CARRIED {
            self.morph = Some((self.drawn.clone(), 0.0));
        }
    }

    /// Set up experiment `k` of `TRIALS`: glide to its setting, then play its film.
    fn set_up(&mut self, k: usize) {
        let t = trial(k);
        self.morph_from_here();
        self.mode = t.mode;
        let now = self.glide.to;
        let mut to = t.setting;
        to.line = nearest(now.line, to.line, TAU64);
        to.mirror = nearest(now.mirror, to.mirror, TAU64);
        self.glide.to = to;
        (self.film, self.film_t, self.playing, self.play_when_there) = (t.film, 0.0, false, t.film.is_some());
        self.q_on = t.q;
        self.sweep = t.sweep.then_some(0.0);
    }

    /// Show the maps `steps` acting, from the start.
    fn play(&mut self, steps: [Step; 2]) {
        self.morph_from_here();
        (self.film, self.film_t, self.playing, self.play_when_there) = (Some(steps), 0.0, false, true);
    }

    /// Everything back where it started: no film.
    fn rewind(&mut self) {
        self.morph_from_here();
        (self.film, self.film_t, self.playing, self.play_when_there) = (None, 0.0, false, false);
    }

    fn set_mode(&mut self, mode: Mode) {
        if mode == self.mode {
            return;
        }
        self.rewind();
        self.mode = mode;
        self.sweep = None;
        if mode == Mode::Mirrors {
            // (the second mirror a little round from the first, wherever that is)
            self.glide.to.mirror = nearest(self.glide.shown.mirror, self.glide.to.line + 0.3, TAU64);
        }
    }

    /// The film's two maps, as they are now (they follow the knobs).
    fn film_maps(&self, s: &Setting) -> Option<[DMat2; 2]> {
        self.film.map(|steps| steps.map(|k| s.step(k)))
    }

    /// The map the plane has been carried through so far.
    fn carried(&self, s: &Setting) -> DMat2 {
        match self.film_maps(s) {
            Some(m) => carried(m, progress(self.film_t)),
            None => DMat2::IDENTITY,
        }
    }

    /// The carried points where the film has them now: the F and x.
    fn carried_points(&self, s: &Setting) -> Vec<DVec2> {
        let m = self.carried(s);
        f_points(self.mode).into_iter().chain([s.x]).map(|p| m * p).collect()
    }

    /// The carried points as drawn: on their way from where they jumped from, if they did.
    fn carried_drawn(&self, s: &Setting) -> Vec<DVec2> {
        let now = self.carried_points(s);
        match &self.morph {
            Some((from, k)) if from.len() == now.len() => {
                let k = smooth(*k);
                from.iter().zip(&now).map(|(a, b)| *a + (*b - *a) * k).collect()
            }
            _ => now,
        }
    }

    /// Move everything on: `dt` seconds of the scene's time (the film, the sweep) and
    /// `real_dt` of the clock's (a change of setting glides by the clock, so that it is
    /// seen through even while the scene's time stands still).
    fn advance(&mut self, dt: f64, real_dt: f64) {
        let dt = if dt.is_finite() { dt.max(0.0) } else { 0.0 };
        let real_dt = if real_dt.is_finite() { real_dt.max(0.0) } else { 0.0 };
        self.glide.step(real_dt);
        if let Some((_, k)) = &mut self.morph {
            *k += real_dt / MORPH_SECONDS;
            if *k >= 1.0 {
                self.morph = None;
            }
        }
        let ready = self.glide.arrived() && self.morph.is_none();
        if self.play_when_there && ready {
            (self.play_when_there, self.playing) = (false, true);
        }
        if self.playing {
            self.film_t = (self.film_t + dt).min(FILM_SECONDS);
            self.playing = self.film_t < FILM_SECONDS;
        }
        if let (Some(t), true) = (self.sweep, ready) {
            let t = t + dt;
            let q = (self.glide.to.foot() + sweep_at(t)).clamp(-Q_MAX, Q_MAX);
            self.glide.now(|s| s.q = q);
            self.sweep = (t < SWEEP[SWEEP.len() - 1].0).then_some(t);
        }
    }

    /// The canonical view: the picture's frame in the middle of the part of the
    /// canvas it has to itself, or (where that shows it larger) over the whole canvas,
    /// with the panel over the frame's top right-hand corner, where nothing is drawn
    /// that matters (`CORNER`).
    fn the_view(&self) -> View {
        let lay = Layout::of(self.canvas, self.inset, self.chart());
        let (min, max) = lay.frame_room();
        let size = FRAME_HALF * 2.0;
        let fit = |(min, max): (Vec2, Vec2)| ((max - min).max(Vec2::splat(60.0)) / size).min_element().max(1e-3);
        let mut best = (fit((min, max)), (min, max));
        if let (Some(p), false) = (&lay.panel, lay.narrow) {
            let whole = (Vec2::new(12.0, min.y), Vec2::new(self.canvas.x - 12.0, max.y));
            let lo = FRAME_CENTRE - FRAME_HALF;
            // (the panel's left edge and its foot, with a margin)
            let (left, foot) = (p.min.x - 8.0, p.max.y + 8.0);
            let scale = fit(whole).min((left - whole.0.x) / (CORNER.x - lo.x)).min((whole.1.y - foot) / (CORNER.y - lo.y));
            if scale > best.0 {
                // as near the middle as it can be while the corner keeps clear of the panel
                let w = size * scale;
                let x0 = ((whole.0.x + whole.1.x - w.x) * 0.5).min(left - (CORNER.x - lo.x) * scale).max(whole.0.x);
                let bottom = ((whole.0.y + whole.1.y + w.y) * 0.5).max(foot + (CORNER.y - lo.y) * scale).min(whole.1.y);
                best = (scale, (Vec2::new(x0, bottom - w.y), Vec2::new(x0 + w.x, bottom)));
            }
        }
        let (per_unit, (min, max)) = best;
        // (screen y runs down, the scene's up)
        let off = ((min + max) * 0.5 - self.canvas * 0.5) / per_unit;
        View::Fit { center: Vec2::new(FRAME_CENTRE.x - off.x, FRAME_CENTRE.y + off.y), half: self.canvas.max(Vec2::ONE) * (0.5 / per_unit) }
    }

    /// Whether the panel has the chart of |x − q|.
    fn chart(&self) -> bool {
        self.mode == Mode::Line && self.q_on
    }

    /// What the film is doing, in a few words: None if there is no film.
    fn stage(&self) -> Option<String> {
        let steps = self.film?;
        let t = self.film_t;
        let (first, pause, second) = (t < STEP_SECONDS, t < STEP_SECONDS + PAUSE, t < FILM_SECONDS);
        let not_yet = t <= 0.0;
        Some(
            match steps {
                [Step::H, Step::H] => {
                    if not_yet || first {
                        r"1 of 2: $H$ acts: along $w$, through the line, as far again"
                    } else if pause {
                        r"after $H$: the mirror image"
                    } else if second {
                        r"2 of 2: $H$ again: everything comes back"
                    } else {
                        r"$H^2 = I$: the mirror undoes itself"
                    }
                }
                [Step::Mirror1, Step::Mirror2] => {
                    if not_yet || first {
                        r"1 of 2: $H_1$, the reflection in mirror 1"
                    } else if pause {
                        r"after $H_1$: $F$ is turned over"
                    } else if second {
                        r"2 of 2: $H_2$, the reflection in mirror 2"
                    } else {
                        r"$H_2 H_1$: turned over twice, which is a turn"
                    }
                }
                _ => {
                    if not_yet || first {
                        r"1 of 2: $P$ acts: every point slides along $w$ to the line"
                    } else if pause {
                        r"after $P$: every point is on the line"
                    } else if second {
                        r"2 of 2: $P$ again: nothing moves"
                    } else {
                        r"$P^2 = P$: the shadow of the shadow is the shadow"
                    }
                }
            }
            .to_string(),
        )
    }

    /// The heading's two lines, and whether the second is a warning.
    fn heading_lines(&self) -> (String, String, bool) {
        let s = self.shown();
        if self.mode == Mode::Mirrors {
            let a = s.apart();
            let first = format!(r"two mirrors, $\alpha = {}$ radians apart", num(a.abs(), 2));
            let second = self.stage().unwrap_or_else(|| format!(r"$H_2 H_1$ is a turn by $2\alpha = {}$", num(2.0 * a, 2)));
            return (first, second, false);
        }
        let first = if s.orthogonal() {
            r"an orthogonal projection $P$, and its mirror $H$".to_string()
        } else {
            format!(r"a slanted projection: $w$ at ${}$ to the line", degrees(s.theta()))
        };
        if s.theta() < WARN {
            return (first, format!(r"$w$ nearly along the line: $P$ stretches by up to $1/\sin\theta = {}$", num(1.0 / s.theta().sin(), 2)), true);
        }
        let second = self.stage().unwrap_or_else(|| {
            if s.orthogonal() {
                r"$P^2 = P$ and $P^\top = P$; $H^2 = I$ and $H^\top H = I$".to_string()
            } else {
                r"$P^2 = P$ but $P^\top \ne P$; $H^2 = I$ but $H^\top H \ne I$".to_string()
            }
        });
        (first, second, false)
    }

    /// The panel's lines under the matrices: (text, colour).
    fn panel_lines(&self) -> Vec<(String, Color)> {
        let s = self.shown();
        if self.mode == Mode::Mirrors {
            let a = s.apart();
            return vec![
                (format!(r"mirror 1 to mirror 2: $\alpha = {}$ (${}$)", num(a, 2), degrees(a)), pal::ORANGE),
                (format!(r"$H_2 H_1$ turns by $2\alpha = {}$ (${}$)", num(2.0 * a, 2), degrees(2.0 * a)), pal::YELLOW),
                (format!(r"$H_1 H_2$ turns by ${}$: the other way", num(-2.0 * a, 2)), pal::DIM),
                (format!(r"$\det H_1 = \det H_2 = {}$, $\det H_2 H_1 = {}$", num(s.h1().determinant(), 0), num((s.h2() * s.h1()).determinant(), 0)), pal::TEXT),
                (r"each undoes itself: $H_1^2 = H_2^2 = I$".to_string(), pal::DIM),
            ];
        }
        let (p, h) = (s.p(), s.h());
        let (x, px, hx) = (s.x, p * s.x, h * s.x);
        let sym = symmetric(p);
        let theta = s.theta();
        vec![
            (if sym { r"$P^2 = P$, and $P^\top = P$: symmetric" } else { r"$P^2 = P$, but $P^\top \ne P$: not symmetric" }.to_string(), pal::ORANGE),
            (if sym { r"$H^2 = I$, and $H^\top H = I$: lengths kept" } else { r"$H^2 = I$, but $H^\top H \ne I$: lengths change" }.to_string(), pal::BLUE),
            (r"$P$ keeps $u$ ($\lambda = 1$), kills $w$ ($\lambda = 0$); $\tr P = 1$".to_string(), pal::TEXT),
            (format!(r"$|x| = {}$, $|Px| = {}$, $|Hx| = {}$", num(x.length(), 2), num(px.length(), 2), num(hx.length(), 2)), pal::TEXT),
            if s.orthogonal() {
                (r"$w$ at right angles: $|Px| \le |x|$ for every $x$".to_string(), pal::DIM)
            } else {
                (format!(r"$w$ at ${}$: $P$ stretches by up to ${}$", degrees(theta), num(1.0 / theta.sin(), 2)), if theta < WARN { pal::RED } else { pal::DIM })
            },
        ]
    }
}

/// The four numbers of a matrix on the panel, in one colour.
fn entries(m: DMat2, c: Color) -> Vec<Vec<(String, Color)>> {
    (0..2).map(|i| (0..2).map(|j| (cell(m.col(j)[i]), c)).collect()).collect()
}

impl Scene for Project {
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
        !self.glide.arrived() || self.playing || self.play_when_there || self.morph.is_some() || self.sweep.is_some()
    }

    fn ui(&mut self, ui: &mut egui::Ui, _: &Gfx) {
        ui::note(ui, r"A projection is a matrix $P$ that does nothing more the second time: $P^2 = P$. Here $P$ moves every point $x$ along a direction $w$ until it meets the orange line: that is its shadow $Px$. The leftover, $x - Px$, is the part of $x$ that $P$ kills (sends to $0$). The mirror $H = 2P - I$ moves $x$ twice as far, through the line:");
        ui::equation(ui, r"x = Px + (x - Px), \quad Hx = Px - (x - Px)");
        ui::note(ui, "Drag $x$, the knob $u$ to turn the line, and the knob $w$ to slant the direction $P$ kills (it snaps to the perpendicular, the orthogonal projection).");

        ui::section(ui, "Things to try");
        for (k, (label, look_for)) in TRIALS.iter().enumerate() {
            if ui::experiment(ui, label, look_for) {
                self.set_up(k);
            }
        }

        ui::section(ui, "Show");
        let mut mode = (self.mode == Mode::Mirrors) as usize;
        if ui::choice(ui, &["one line", "two mirrors"], &mut mode) {
            self.set_mode(if mode == 1 { Mode::Mirrors } else { Mode::Line });
        }
        let buttons: &[&str] = if self.mode == Mode::Line { &["apply $P$ twice", "apply $H$ twice", "back to the start"] } else { &["reflect in both mirrors", "back to the start"] };
        match (ui::buttons(ui, buttons), self.mode) {
            (Some(0), Mode::Line) => self.play([Step::P, Step::P]),
            (Some(1), Mode::Line) => self.play([Step::H, Step::H]),
            (Some(0), Mode::Mirrors) => self.play([Step::Mirror1, Step::Mirror2]),
            (Some(_), _) => self.rewind(),
            _ => {}
        }
        ui::note(ui, "Each film shows the map acting twice on the letter $F$ (and on $x$), every point moving straight to where the map sends it.");
        if self.mode == Mode::Line {
            ui.horizontal_wrapped(|ui| {
                if ui.button(ui::label_text(ui, "$w$ at right angles", 12.5, None, false)).clicked() {
                    self.glide.to.rel = FRAC_PI_2;
                }
                if ui::checkbox(ui, &mut self.q_on, "a point $q$ on the line").changed() {
                    self.sweep = None;
                }
            });
            ui::note(ui, r"$w$ at right angles makes $P$ the orthogonal projection, $uu^\top / (u^\top u)$. Drag $q$ along the line to see how far it is from $x$.");
        } else {
            ui::note(ui, "Drag the knobs on the mirrors to turn them.");
        }

        ui::section(ui, "Reading the picture");
        if self.mode == Mode::Line {
            let white = if self.q_on {
                r"white: $x$ (drag it), and $q$ on the line (drag it), dashed to $x$; the dashed circle round $x$ touches the line at the point nearest to $x$"
            } else {
                r"white: $x$ (drag it); the dashed circle round $0$ passes through $x$; a small dot, $x$ as a film carries it"
            };
            ui::key(
                ui,
                &[
                    (pal::ORANGE, r"orange: the line through $0$ along $u$, which $P$ keeps (drag the knob $u$ to turn it); the arrow on it is $Px$"),
                    (pal::PURPLE, r"purple, dashed: the line along $w$, which $P$ sends to $0$ (drag the knob $w$)"),
                    (pal::WHITE, white),
                    (pal::RED, r"red: the leftover $x - Px$, along $w$; dashed, the same reversed, from $Px$ on to $Hx$"),
                    (pal::BLUE, r"blue: $Hx$ and $HF$, the mirror images of $x$ and of the letter $F$ under $H = 2P - I$"),
                    (pal::YELLOW, "yellow: the letter $F$, and $PF$, its shadow on the line; dashed, where $F$ was before a film moved it"),
                ],
            );
        } else {
            ui::key(
                ui,
                &[
                    (pal::ORANGE, "orange: the two mirrors, lines through $0$ (drag their knobs); the arc between them is the angle $\\alpha$"),
                    (pal::YELLOW, "yellow: the letter $F$, as the reflections carry it; dashed, where it starts and where the two reflections take it"),
                    (pal::BLUE, "blue: $H_1 F$, the letter after mirror 1 alone"),
                    (pal::WHITE, r"white arc: the turn the two make together, $2\alpha$"),
                ],
            );
        }

        ui::section(ui, "Words");
        ui::term(ui, "Projection:", r"a matrix $P$ with $P^2 = P$. The ones here project the plane onto a line: they keep every point of the line and send one other direction, $w$, to $0$, so every point moves to the line along $w$. (Always through $0$: a linear map.)");
        ui::term(ui, "Leftover:", r"$x - Px = (I - P)x$, what $P$ takes away. $I - P$ is a projection too, onto the line of $w$, sending $u$ to $0$.");
        ui::term(ui, "Orthogonal projection:", r"one whose $w$ is at right angles to the line. Then $P = uu^\top / (u^\top u)$, $P$ is symmetric, and $Px$ is the point of the line nearest to $x$.");
        ui::term(ui, "Symmetric:", r"equal to its own transpose, $P^\top = P$: the same numbers on either side of the diagonal.");
        ui::term(ui, "Slanted projection:", r"one whose $w$ is not at right angles to the line (also called oblique). Still $P^2 = P$, but $P^\top \ne P$, and $Px$ can be longer than $x$.");
        ui::term(ui, "Reflection:", r"$H = 2P - I$: it keeps the line and reverses $w$ (eigenvalues $1$ and $-1$), so $H^2 = I$. A true mirror, keeping every length ($H^\top H = I$), exactly when $P$ is orthogonal.");
        ui::term(ui, "Householder reflection:", r"a true mirror written with the unit vector $n$ at right angles to it: $H = I - 2nn^\top$. With $n$ along $a - b$, for $a$ and $b$ of the same length, it sends $a$ to $b$.");
        ui::term(ui, "Trace:", r"$\tr P$, the sum of the diagonal, which is also the sum of the eigenvalues: for a projection, the number of independent directions it keeps.");

        ui::section(ui, "Numbers");
        let s = self.shown();
        if self.mode == Mode::Line {
            let (p, h) = (s.p(), s.h());
            ui::readout(ui, "angle between $w$ and the line", degrees(s.theta()));
            ui::readout(ui, r"$|x|$, $|Px|$, $|Hx|$", format!("${}, {}, {}$", num(s.x.length(), 2), num((p * s.x).length(), 2), num((h * s.x).length(), 2)));
            ui::readout(ui, r"$|x - Px|$", num((s.x - p * s.x).length(), 2));
            if self.q_on {
                ui::readout(ui, r"$|x - q|$", num((s.x - s.u() * s.q).length(), 2));
            }
            ui::readout(ui, r"$\tr P$, $\det P$", format!("${}, {}$", num(p.x_axis.x + p.y_axis.y, 2), num(p.determinant(), 2)));
            ui::readout(ui, r"$\det H$", format!("${}$", num(h.determinant(), 2)));
        } else {
            let a = s.apart();
            ui::readout(ui, r"$\alpha$, mirror 1 to mirror 2", format!("{} ({})", num(a, 2), degrees(a)));
            ui::readout(ui, r"the turn $H_2 H_1$", format!("{} ({})", num(2.0 * a, 2), degrees(2.0 * a)));
            ui::readout(ui, r"$\det H_2 H_1$", format!("${}$", num((s.h2() * s.h1()).determinant(), 2)));
        }
        ui::see_also(
            ui,
            &[
                ("lsq", "the nearest point of a plane: least squares"),
                ("gram", "Gram–Schmidt: shadows taken away one at a time"),
                ("special", "projections and mirrors among other special matrices"),
                ("pinv", "the pseudoinverse, whose products are projections"),
            ],
        );
    }

    fn frame(&mut self, f: &mut Frame) {
        // the panel's place follows the canvas, and the picture's place follows the panel's
        let view = f.view_size();
        let chart = self.chart();
        let lay = Layout::of(view, f.top_inset, chart);
        let before = Layout::of(self.canvas, self.inset, chart);
        // (a canvas squeezed to a sliver is one the controls have opened over, on a
        // phone: the picture is not seen then, and is left as it was for when they close)
        let covered = view.x < 160.0;
        let changed = !covered && (lay.narrow != before.narrow || lay.panel.is_some() != before.panel.is_some() || (view - self.canvas).abs().max_element() > 24.0);
        if !covered && (changed || !self.measured) {
            self.new_view |= changed;
            (self.canvas, self.measured) = (view, true);
        }
        if !covered {
            self.new_view |= (f.top_inset - self.inset).abs() > 1.0 || chart != self.charted;
            (self.inset, self.charted) = (f.top_inset, chart);
        }
        let pointer = f.ptr.pos.map(|p| p / f.px);
        if let Some(panel) = lay.panel.as_ref().filter(|_| !covered) {
            if pointer.is_some_and(|p| inside_rect(p, (panel.min, panel.max))) {
                // (the panel is for reading: no moving the view from there)
                f.handles.capture();
            }
        }
        self.advance(f.dt as f64, f.real_dt as f64);

        let labels_before = f.dl.labels.len();
        let plates = self.picture(f, &lay);
        let numbers_from = f.dl.labels.len();
        f.axes2();
        // no number of the axes is written over the heading or the caption
        let mut k = 0;
        f.dl.labels.retain(|l| {
            k += 1;
            k <= numbers_from || !plates.iter().any(|plate| inside_rect(l.pos, (plate.0 - Vec2::splat(10.0), plate.1 + Vec2::splat(10.0))))
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

/// The parts of the picture drawn as lines, each in a batch of its own where its
/// number of points can change from frame to frame (so that `eidos motion` can still
/// compare the rest point for point).
#[derive(Clone, Copy, PartialEq)]
enum Part {
    /// What keeps its number of points (in the usual batch).
    Fixed,
    /// The right-angle mark, which comes and goes.
    Corner,
    /// The dashed circle (round 0, or round x while q is shown).
    Circle,
    /// Where the letter started, which a film's start brings.
    Film,
    /// The arrows, which vanish when they have no length.
    Arrows,
    /// Where two mirrors take the letter (dashed).
    Ends,
}

fn lines_for<'a>(f: &'a mut Frame, part: Part) -> &'a mut crate::gfx::draw::LineBatch {
    if part == Part::Fixed {
        return &mut f.dl.lines;
    }
    f.dl.lines_with(LineKind::Blend, LineStyle { width_scale: 1.0 + 1e-4 * part as u8 as f32, ..Default::default() })
}

/// An arrow from `a` to `b` (as `Frame::arrow` draws one), in the batch of `part`.
fn arrow(f: &mut Frame, part: Part, a: DVec2, b: DVec2, width: f32, c: Color) {
    let (p, q) = (a.as_vec2().extend(0.0), b.as_vec2().extend(0.0));
    let d = q - p;
    let len = d.length();
    if len.is_nan() || len <= 1e-9 || !len.is_finite() {
        return;
    }
    let dir = d / len;
    let wpp = f.cam.world_per_px(q) * f.px;
    let head = ((width * 3.2 + 5.0) * wpp).min(len * 0.6);
    let side = dir.cross(Vec3::Z).normalize_or_zero() * (head * 0.42);
    let base = q - dir * head;
    f.dl.tris_alpha.tri(q, base + side, base - side, c);
    lines_for(f, part).segment(p, q - dir * (head * 0.8), width, c);
}

/// The F, through the points `pts` (its outline, then its bars' corners: `f_points`).
fn draw_f(f: &mut Frame, pts: &[DVec2], fill: Color, edge: Color, width: f32) {
    let v: Vec<Vec3> = pts.iter().map(|p| p.as_vec2().extend(0.0)).collect();
    for bar in 0..3 {
        let c = &v[10 + 4 * bar..14 + 4 * bar];
        f.dl.tris_alpha.quad(c[0], c[1], c[2], c[3], fill);
    }
    f.dl.lines.closed(&v[..10], width, edge);
}

/// The middle of the F's box, through the points `pts` (as `f_points` lists them).
fn f_middle(pts: &[DVec2]) -> DVec2 {
    // (the stem's foot and the top bar's far corner are opposite corners of the box)
    (pts[0] + pts[8]) * 0.5
}

/// The F's box through the points `pts`: its four corners, in order round it.
fn f_box(pts: &[DVec2]) -> [DVec2; 4] {
    // (the outline's corners 0, 1 … are the box's at 0 and 9, and 8 is opposite 0;
    // the fourth is the box's bottom right, the image of (1, 0))
    [pts[0], pts[0] + (pts[8] - pts[9]), pts[8], pts[9]]
}

/// Keep labels off the F through the points `pts`: off its box, and off its edges.
fn keep_off(f: &Frame, room: &mut Room, pts: &[DVec2]) {
    let corners = f_box(pts).map(|p| p.as_vec2().extend(0.0));
    room.shape(f, &corners);
    for k in 0..4 {
        room.line(f, corners[k], corners[(k + 1) % 4]);
    }
}

/// How far the F through `pts` reaches from its middle along `d` (a unit vector), in
/// pixels on the screen.
fn half_across_px(f: &Frame, pts: &[DVec2], d: DVec2) -> f32 {
    let mid = f_middle(pts);
    let reach = f_box(pts).iter().map(|c| (*c - mid).dot(d)).fold(0.0f64, f64::max);
    let wpp = f.cam.world_per_px(mid.as_vec2().extend(0.0)) * f.px;
    reach as f32 / wpp.max(1e-9)
}

/// The F's outline, dashed (in a batch of dashed lines of `part`'s own): each edge
/// is dashed from its start, which for straight edges is as good as dashes measured
/// along the whole outline, and is the same number of points in every frame.
fn dashed_f(f: &mut Frame, part: Part, pts: &[DVec2], width: f32, color: Color) {
    let style = LineStyle { dash: 9.0, duty: 0.55, width_scale: 1.0 + 1e-4 * part as u8 as f32, ..Default::default() };
    let batch = f.dl.lines_with(LineKind::Blend, style);
    for k in 0..10 {
        batch.segment(pts[k].as_vec2().extend(0.0), pts[(k + 1) % 10].as_vec2().extend(0.0), width, color);
    }
}

/// Keep labels off a circle (as 32 chords).
fn keep_off_circle(f: &Frame, room: &mut Room, centre: DVec2, r: f64) {
    let at = |k: usize| (centre + DVec2::from_angle(k as f64 / 32.0 * TAU64) * r).as_vec2().extend(0.0);
    for k in 0..32 {
        room.line(f, at(k), at(k + 1));
    }
}

/// A dashed circle, its dashes fixed on the circle rather than measured on the
/// screen: as many points in every frame, so that `eidos motion` can follow them.
fn dashed_circle(f: &mut Frame, centre: DVec2, r: f64, width: f32, color: Color) {
    const DASHES: usize = 72;
    let batch = lines_for(f, Part::Circle);
    for k in 0..DASHES {
        let from = k as f64 / DASHES as f64 * TAU64;
        let pts = (0..=2).map(|i| (centre + DVec2::from_angle(from + i as f64 * 0.275 * TAU64 / DASHES as f64) * r).as_vec2().extend(0.0));
        batch.polyline(pts, width, color);
    }
}

/// Where the segment from `a` to `b` (in the scene) leaves the view, less a margin of
/// `margin` pixels all round: None if it does not start in it, or does not leave it.
fn leaves_view(f: &Frame, a: DVec2, b: DVec2, margin: f32) -> Option<DVec2> {
    let (p, q) = (on_screen(f, a.as_vec2().extend(0.0))?, on_screen(f, b.as_vec2().extend(0.0))?);
    let (lo, hi) = (Vec2::new(margin, f.top_inset + margin), f.view_size() - Vec2::splat(margin));
    if !inside_rect(p, (lo, hi)) {
        return None;
    }
    let d = q - p;
    let mut t = 1.0f32;
    for k in 0..2 {
        if d[k] > 1e-6 {
            t = t.min((hi[k] - p[k]) / d[k]);
        } else if d[k] < -1e-6 {
            t = t.min((lo[k] - p[k]) / d[k]);
        }
    }
    (0.0..1.0).contains(&t).then(|| a + (b - a) * t as f64)
}

/// Write `text` beside the scene's point `at`, on its side `towards` (`gap` pixels off),
/// if there is room there, and not at all if there is not: for a name that would
/// otherwise go looking for room round its point, and jump across it, as what it
/// names is dragged past an edge or another name. (A name already shown needs less
/// room to stay than a new one needs to appear, so that it does not flicker.)
#[allow(clippy::too_many_arguments)]
fn name_or_none(f: &mut Frame, room: &mut Room, tag: &'static str, at: Vec3, text: &str, size: f32, color: Color, towards: Vec2, gap: f32) {
    name_out_to(f, room, tag, at, text, size, color, towards, gap, 1);
}

/// The same, trying `tries` gaps, each 12 pixels further out than the last: for a
/// name beside several lines close together.
#[allow(clippy::too_many_arguments)]
fn name_out_to(f: &mut Frame, room: &mut Room, tag: &'static str, at: Vec3, text: &str, size: f32, color: Color, towards: Vec2, gap: f32, tries: usize) {
    name_beside(f, room, tag, at, text, size, color, &[towards], gap, tries);
}

/// The same with a choice of sides, each tried in turn (and the side it was on in
/// the last frame first, so that it does not swap sides while both have room).
#[allow(clippy::too_many_arguments)]
fn name_beside(f: &mut Frame, room: &mut Room, tag: &'static str, at: Vec3, text: &str, size: f32, color: Color, sides: &[Vec2], gap: f32, tries: usize) {
    let Some(point) = on_screen(f, at) else { return };
    let half = if crate::typeset::has_math(text) { f.text_size(text, size) * 0.5 + Vec2::new(4.0, 0.0) } else { Vec2::new(text_width(text, size) * 0.5 + 4.0, size * 0.5 + 1.0) };
    let was = f.places.chosen(tag, 0);
    let need = if was.is_some() { 2.0 } else { 6.0 };
    let mut sides: Vec<Vec2> = sides.iter().map(|s| s.normalize_or_zero()).collect();
    if let Some(was) = was {
        sides.sort_by(|a, b| b.dot(was).total_cmp(&a.dot(was)));
    }
    for side in sides {
        for k in 0..tries {
            let gap = gap + 12.0 * k as f32;
            if room.room_for(point, point + side * gap + side * half, half) >= need {
                room.label(f, tag, at, text, size, color, side, gap, false);
                return;
            }
        }
    }
}

/// The gap that puts a label of `text` beside a line, on its side `side` (a direction
/// on the screen at right angles to the line), `margin` pixels clear of it.
/// (`Room::label` puts the label's middle `gap` along `side` and then half its size
/// further, axis by axis: beside a slanting line, a wide label so placed has a corner
/// on the line unless the gap allows for it.)
fn gap_beside(f: &Frame, text: &str, size: f32, side: Vec2, margin: f32) -> f32 {
    let n = side.normalize_or_zero();
    let half = if crate::typeset::has_math(text) { f.text_size(text, size) * 0.5 + Vec2::new(4.0, 0.0) } else { Vec2::new(text_width(text, size) * 0.5 + 4.0, size * 0.5 + 1.0) };
    let (ax, ay) = (n.x.abs(), n.y.abs());
    half.x * (ax - ax * ax) + half.y * (ay - ay * ay) + margin
}

impl Project {
    /// Everything drawn in the scene's own coordinates, and the names written beside it.
    /// Returns the boxes of the heading and of the caption, which are written over it.
    fn picture(&mut self, f: &mut Frame, lay: &Layout) -> Vec<(Vec2, Vec2)> {
        let (lo, hi) = f.cam.flat_bounds();
        let reach = ((hi - lo).length() + lo.length().max(hi.length())) as f64;
        let mut room = Room::new(f);
        if let Some(p) = &lay.panel {
            room.wall(p.min - Vec2::splat(4.0), p.max + Vec2::splat(4.0));
        }
        let under_panel = |f: &Frame, p: DVec2| lay.panel.as_ref().is_some_and(|panel| on_screen(f, p.as_vec2().extend(0.0)).is_some_and(|q| inside_rect(q, (panel.min, panel.max))));

        // the handles first, so that what is drawn follows them in the same frame
        let s = self.shown();
        let knob_u = s.u() * KNOB_U;
        let mut k = knob_u.as_vec2();
        let free = |f: &Frame, p: DVec2| !under_panel(f, p) || f.handles.dragging();
        if free(f, knob_u) && f.drag2(2, &mut k, pal::ORANGE) && k.length() > 1e-3 {
            let line = (k.y as f64).atan2(k.x as f64);
            self.glide.now(|g| g.line = line);
            self.sweep = None;
        }
        if self.mode == Mode::Line {
            let mut x = s.x.as_vec2();
            if free(f, s.x) && f.drag2(1, &mut x, pal::WHITE) {
                let x = x.as_dvec2().clamp(DVec2::splat(-X_MAX), DVec2::splat(X_MAX));
                self.glide.now(|g| g.x = x);
                self.sweep = None;
            }
            let knob_w = s.w() * KNOB_W;
            let mut k = knob_w.as_vec2();
            if free(f, knob_w) && f.drag2(3, &mut k, pal::PURPLE) && k.length() > 1e-3 {
                let angle = (k.y as f64).atan2(k.x as f64);
                let (rel, raw) = (rel_for(self.glide.to.line, angle), rel_unsnapped(self.glide.to.line, angle));
                // (w follows the hand, but glides into the perpendicular and out of it
                // again: a snap is a step of up to 4 degrees, too far to take in a frame)
                if rel == raw && (self.glide.shown.rel - raw).abs() < 0.5 * PI64 / 180.0 {
                    self.glide.now(|g| g.rel = rel);
                } else {
                    self.glide.to.rel = rel;
                }
            }
            if self.q_on {
                let at = s.u() * s.q;
                let mut q = at.as_vec2();
                if free(f, at) && f.drag2(4, &mut q, pal::WHITE) {
                    let along = q.as_dvec2().dot(self.glide.to.u()).clamp(-Q_MAX, Q_MAX);
                    self.glide.now(|g| g.q = along);
                    self.sweep = None;
                }
            }
        } else {
            let knob_m = DVec2::from_angle(s.mirror) * KNOB_U;
            let mut k = knob_m.as_vec2();
            if free(f, knob_m) && f.drag2(5, &mut k, pal::ORANGE) && k.length() > 1e-3 {
                let mirror = (k.y as f64).atan2(k.x as f64);
                self.glide.now(|g| g.mirror = mirror);
            }
        }
        let s = self.shown();

        // the heading, at the top of the picture, and the caption at its foot
        let mut plates = Vec::new();
        let heading = self.heading(f, lay);
        room.wall(heading.0, heading.1);
        plates.push(heading);
        if let Some((left, right, bottom)) = lay.caption {
            if let Some(plate) = self.caption(f, (left, right), bottom, lay.narrow) {
                room.wall(plate.0, plate.1);
                plates.push(plate);
            }
        }

        // the carried points (the F and x), as drawn
        let carried = self.carried_drawn(&s);
        self.drawn = carried.clone();
        let home = f_points(self.mode);
        let away = carried.iter().zip(&home).any(|(a, b)| a.distance(*b) > 1e-6);

        match self.mode {
            Mode::Line => self.one_line(f, &mut room, &s, reach, &carried, &home, away),
            Mode::Mirrors => self.two_mirrors(f, &mut room, &s, reach, &carried, &home),
        }
        plates
    }

    /// The picture of one line: the projection and its mirror.
    #[allow(clippy::too_many_arguments)]
    fn one_line(&mut self, f: &mut Frame, room: &mut Room, s: &Setting, reach: f64, carried: &[DVec2], home: &[DVec2], away: bool) {
        let (u, w, p, h) = (s.u(), s.w(), s.p(), s.h());
        let (x, px, hx) = (s.x, p * s.x, h * s.x);
        let v3 = |p: DVec2| p.as_vec2().extend(0.0);

        // the two lines: the one P keeps, and the one it sends to 0
        f.line(v3(-u * reach), v3(u * reach), 1.8, pal::ORANGE.a(0.7));
        f.dashed(v3(-w * reach), v3(w * reach), 1.5, pal::PURPLE.a(0.8));
        room.line(f, v3(-u * reach), v3(u * reach));
        room.line(f, v3(-w * reach), v3(w * reach));
        // the knobs' arrows
        let (knob_u, knob_w) = (u * KNOB_U, w * KNOB_W);
        f.line(Vec3::ZERO, v3(knob_w), 2.6, pal::PURPLE.a(0.9));

        // the circle through x, round 0 (not while q is shown: then the circle round x)
        let r = x.length();
        if !self.q_on {
            dashed_circle(f, DVec2::ZERO, r, 1.2, pal::WHITE.a(0.3));
            keep_off_circle(f, room, DVec2::ZERO, r);
        }

        // the F's mirror image, and its shadow: a stretch of the line
        let mirrored: Vec<DVec2> = home.iter().map(|q| h * *q).collect();
        draw_f(f, &mirrored, pal::BLUE.a(0.16), pal::BLUE.a(0.8), 1.4);
        keep_off(f, room, &mirrored);
        let along: Vec<f64> = home[..10].iter().map(|q| (p * *q).dot(u)).collect();
        let (s0, s1) = along.iter().fold((f64::MAX, f64::MIN), |(a, b), v| (a.min(*v), b.max(*v)));
        f.dl.lines_with(LineKind::Blend, LineStyle { width_scale: 1.0 + 7e-4, ..Default::default() }).segment(v3(u * s0), v3(u * s1), 5.0, pal::YELLOW.a(0.85));

        // the F as the film has carried it, and where it started
        draw_f(f, carried, pal::YELLOW.a(0.22), pal::YELLOW.a(0.9), 1.6);
        keep_off(f, room, carried);
        if away {
            dashed_f(f, Part::Film, home, 1.2, pal::YELLOW.a(0.6));
            keep_off(f, room, home);
        }

        // x, its shadow, the leftover, and the mirror image
        arrow(f, Part::Arrows, DVec2::ZERO, x, 2.0, pal::WHITE.a(0.85));
        arrow(f, Part::Arrows, DVec2::ZERO, px, 3.2, pal::ORANGE);
        arrow(f, Part::Arrows, px, x, 2.4, pal::RED);
        f.dashed(v3(px), v3(hx), 1.6, pal::RED.a(0.7));
        arrow(f, Part::Arrows, DVec2::ZERO, hx, 2.2, pal::BLUE);
        f.dot(v3(px), 9.0, pal::ORANGE);
        f.dot(v3(hx), 9.0, pal::BLUE);
        for (a, b) in [(DVec2::ZERO, x), (DVec2::ZERO, px), (px, x), (px, hx), (DVec2::ZERO, hx)] {
            room.line(f, v3(a), v3(b));
        }
        // the right angle, where the leftover meets the line (if it has the room)
        let wpp = f.cam.world_per_px(Vec3::ZERO) * f.px;
        let corner = 11.0 * wpp as f64;
        if s.orthogonal() && (x - px).length() > 2.0 * corner && px.length() > 2.0 * corner {
            let (a, b) = (-px.normalize() * corner, (x - px).normalize() * corner);
            lines_for(f, Part::Corner).polyline([v3(px + a), v3(px + a + b), v3(px + b)], 1.4, pal::WHITE.a(0.8));
        }

        // x carried by the film
        let xc = carried[CARRIED - 1];
        if xc.distance(x) > 1e-6 {
            f.dl.points_with(PointKind::Disc, PointStyle { outline_px: 1.25 + 1e-4, ..Default::default() }).push(v3(xc), 8.0, pal::WHITE.a(0.9));
        }

        // q, and how far it is from x
        if self.q_on {
            let q = u * s.q;
            f.dashed(v3(q), v3(x), 1.6, pal::WHITE.a(0.75));
            room.line(f, v3(q), v3(x));
            // the circle round x that touches the line, at the nearest point
            let foot = u * s.foot();
            dashed_circle(f, x, (x - foot).length(), 1.2, pal::WHITE.a(0.3));
            keep_off_circle(f, room, x, (x - foot).length());
        }

        // the names
        let away_from = |p: DVec2| Vec2::new(p.x as f32, -(p.y as f32));
        let seen = |f: &Frame, p: DVec2| p.is_finite() && on_screen(f, v3(p)).is_some_and(|q| inside_rect(q, (Vec2::ZERO, f.view_size())));
        if seen(f, x) {
            room.label(f, "x", v3(x), "$x$", 16.0, pal::WHITE, away_from(x - px), 10.0, false);
        }
        if seen(f, px) {
            let q_here = self.q_on && (u * s.q - px).length() < 14.0 * wpp as f64;
            let text = if q_here { "$Px = q$" } else { "$Px$" };
            room.label(f, "Px", v3(px), text, 15.0, pal::ORANGE, away_from(px - x), 9.0, false);
        }
        if seen(f, hx) {
            room.label(f, "Hx", v3(hx), "$Hx$", 15.0, pal::BLUE, away_from(hx - px), 9.0, false);
        }
        // (named near x: Px may be far off, out of the picture)
        let mid = x + (px - x) * (0.5f64).min(90.0 * wpp as f64 / (px - x).length().max(1e-9));
        if seen(f, mid) && (x - px).length() > 30.0 * wpp as f64 {
            let across = away_from((x - px).perp());
            // (on the right of it; the distance from q, which can lie along it, on the left)
            let side = if across.x < 0.0 { -across } else { across };
            room.label(f, "left", v3(mid), "$x - Px$", 14.0, pal::RED, side, gap_beside(f, "$x - Px$", 14.0, side, 6.0), true);
        }
        // the knobs, with what P does to the lines they are on
        if seen(f, knob_u) {
            let side = away_from(u.perp());
            room.label(f, "u", v3(knob_u), "$Pu = u$", 15.0, pal::ORANGE, side, gap_beside(f, "$Pu = u$", 15.0, side, 9.0), false);
        }
        if seen(f, knob_w) {
            room.label(f, "w", v3(knob_w), "$Pw = 0$", 15.0, pal::PURPLE, away_from(w), 9.0, false);
        }
        if self.q_on {
            let q = u * s.q;
            // q named with how far it is from x; on Px, where Px's name says it is q, the
            // distance is written beside the line from it to x
            let on_px = (q - px).length() < 14.0 * wpp as f64;
            let text = format!("$q$: $|x - q| = {}$", num((x - q).length(), 2));
            if seen(f, q) && !on_px {
                // (on the side of the line away from x)
                let across = away_from(u.perp());
                let side = if across.dot(away_from(x - q)) > 0.0 { -across } else { across };
                name_beside(f, room, "q", v3(q), &text, 14.0, pal::WHITE, &[side, -side], gap_beside(f, &text, 14.0, side, 8.0), 3);
            }
            let mid = (q + x) * 0.5;
            if on_px && seen(f, mid) && (q - x).length() > 40.0 * wpp as f64 {
                let across = away_from((x - q).perp());
                let side = if across.x > 0.0 { -across } else { across };
                let text = format!("$|x - q| = {}$", num((x - q).length(), 2));
                name_beside(f, room, "dist", v3(mid), &text, 13.0, pal::WHITE.a(0.85), &[side, -side], gap_beside(f, &text, 13.0, side, 6.0), 4);
            }
        }
        // the F's mirror image, beside it on the side away from 0, and its shadow, under
        // the line on the screen
        let mc = f_middle(&mirrored);
        if seen(f, mc) {
            let out = mc.normalize_or_zero();
            let gap = half_across_px(f, &mirrored, out) + 6.0;
            name_or_none(f, room, "HF", v3(mc), "$HF$", 14.0, pal::BLUE.mix(pal::WHITE, 0.3), away_from(out), gap);
        }
        let sc = u * (0.5 * (s0 + s1));
        if seen(f, sc) {
            // (on the side of the line that is down on the screen)
            let across = away_from(u.perp());
            let down = if across.y > 0.0 { across } else { -across };
            name_or_none(f, room, "shadow", v3(sc), "$PF$", 14.0, pal::YELLOW, down, gap_beside(f, "$PF$", 14.0, down, 8.0));
        }
        // Px and Hx far off the picture (w nearly along the line): how far, where
        // their arrows leave it
        for (tip, tag, name, c) in [(px, "Px far", "Px", pal::ORANGE), (hx, "Hx far", "Hx", pal::BLUE)] {
            if seen(f, tip) {
                continue;
            }
            let text = format!("${name}$, ${}$ from $0$", num(tip.length(), 1));
            let margin = f.text_size(&text, 13.0).x * 0.5 + 24.0;
            if let Some(at) = leaves_view(f, DVec2::ZERO, tip, margin) {
                // (on the side of the arrow towards the middle of the picture)
                let across = away_from(tip.perp());
                let inwards = on_screen(f, v3(at)).map_or(Vec2::ZERO, |p| f.view_size() * 0.5 - p);
                let side = if across.dot(inwards) > 0.0 { across } else { -across };
                name_out_to(f, room, tag, v3(at), &text, 13.0, c, side, gap_beside(f, &text, 13.0, side, 6.0), 5);
            }
        }
    }

    /// The picture of two mirrors, whose reflections one after the other make a turn.
    #[allow(clippy::too_many_arguments)]
    fn two_mirrors(&mut self, f: &mut Frame, room: &mut Room, s: &Setting, reach: f64, carried: &[DVec2], home: &[DVec2]) {
        let v3 = |p: DVec2| p.as_vec2().extend(0.0);
        let (m1, m2) = (s.u(), DVec2::from_angle(s.mirror));
        for (d, width) in [(m1, 1.8), (m2, 1.8)] {
            f.line(v3(-d * reach), v3(d * reach), width, pal::ORANGE.a(0.7));
            room.line(f, v3(-d * reach), v3(d * reach));
        }
        let (h1, h2) = (s.h1(), s.h2());
        let a = s.apart();

        // the arc of the angle between the mirrors, from mirror 1 round to mirror 2
        let arc = |r: f64, from: f64, by: f64| -> Vec<Vec3> { (0..=40).map(|i| v3(DVec2::from_angle(from + by * i as f64 / 40.0) * r)).collect() };
        f.path(arc(1.1, s.line, a), 1.6, pal::ORANGE.a(0.9));

        // the F after mirror 1, and after both (where it ends), and the turn between
        let once: Vec<DVec2> = home.iter().map(|q| h1 * *q).collect();
        let twice: Vec<DVec2> = home.iter().map(|q| h2 * h1 * *q).collect();
        draw_f(f, &once, pal::BLUE.a(0.16), pal::BLUE.a(0.8), 1.4);
        dashed_f(f, Part::Ends, &twice, 1.2, pal::YELLOW.a(0.6));
        let c = f_centre(Mode::Mirrors);
        let turn = arc(c.length() + 0.85, c.y.atan2(c.x), 2.0 * a);
        f.path(turn.iter().copied(), 1.4, pal::WHITE.a(0.6));
        if let [.., p, q] = turn[..] {
            let d = q - p;
            if d.length() > 1e-9 {
                let wpp = f.cam.world_per_px(q) * f.px;
                let dir = d.normalize();
                let side = dir.cross(Vec3::Z) * (4.0 * wpp);
                f.dl.tris_alpha.tri(q + dir * (5.0 * wpp), q - dir * (5.0 * wpp) + side, q - dir * (5.0 * wpp) - side, pal::WHITE.a(0.8));
            }
        }

        // the F as the film has carried it, and where it started (dashed: under it, until
        // a film moves it)
        dashed_f(f, Part::Film, home, 1.2, pal::YELLOW.a(0.6));
        draw_f(f, carried, pal::YELLOW.a(0.22), pal::YELLOW.a(0.9), 1.6);
        for shape in [&once[..], &twice[..], carried] {
            keep_off(f, room, shape);
        }

        // the names
        let seen = |f: &Frame, p: DVec2| p.is_finite() && on_screen(f, v3(p)).is_some_and(|q| inside_rect(q, (Vec2::ZERO, f.view_size())));
        let out = |p: DVec2| Vec2::new(p.x as f32, -(p.y as f32));
        for (pts, text, colour, tag) in [(home, "$F$", pal::YELLOW, "F"), (&once[..], "$H_1 F$", pal::BLUE.mix(pal::WHITE, 0.3), "H1F"), (&twice[..], "$H_2 H_1 F$", pal::YELLOW, "RF")] {
            let c = f_middle(pts);
            if seen(f, c) {
                let gap = half_across_px(f, pts, c.normalize_or_zero()) + 6.0;
                name_or_none(f, room, tag, v3(c), text, 14.0, colour, out(c), gap);
            }
        }
        let mid = DVec2::from_angle(s.line + 0.5 * a) * 1.1;
        if seen(f, mid) {
            room.label(f, "alpha", v3(mid), r"$\alpha$", 15.0, pal::ORANGE, out(mid), 6.0, false);
        }
        let mid = DVec2::from_angle(c.y.atan2(c.x) + a) * (c.length() + 0.85);
        if seen(f, mid) {
            room.label(f, "turn", v3(mid), r"$2\alpha$", 15.0, pal::WHITE, out(mid), 6.0, false);
        }
        for (d, text, tag) in [(m1, "mirror 1", "m1"), (m2, "mirror 2", "m2")] {
            let knob = d * KNOB_U;
            if seen(f, knob) {
                // (on the side away from the other mirror)
                let other = if tag == "m1" { m2 } else { m1 };
                let side = out(if d.perp().dot(other) > 0.0 { -d.perp() } else { d.perp() });
                room.label(f, tag, v3(knob), text, 13.0, pal::ORANGE, side, gap_beside(f, text, 13.0, side, 8.0), false);
            }
        }
    }

    /// The heading at the top of the picture: what the picture is, and what the film is
    /// doing (or a warning). Returns the box it takes.
    fn heading(&self, f: &mut Frame, lay: &Layout) -> (Vec2, Vec2) {
        let (first, second, warning) = self.heading_lines();
        let at = lay.picture.0 + Vec2::new(16.0, 14.0);
        let room = (lay.picture.1.x - lay.picture.0.x - 32.0).max(40.0);
        let size = if text_width(&first, 15.0) <= room { 15.0 } else { 13.0 };
        let small = if text_width(&second, 12.5) <= room { 12.5 } else { 11.0 };
        f.caption_px(at, first.as_str(), size, pal::TEXT, Align2::LEFT_TOP);
        f.caption_px(at + Vec2::new(0.0, size + 8.0), second.as_str(), small, if warning { pal::RED } else { pal::DIM }, Align2::LEFT_TOP);
        let wide = text_width(&first, size).max(text_width(&second, small)).min(room);
        (at - Vec2::splat(6.0), at + Vec2::new(wide + 12.0, size + small + 16.0))
    }

    /// The caption along the foot of the picture, naming what has no name of its own
    /// there. Returns the box it takes (none on a canvas too small to spare the room).
    fn caption(&self, f: &mut Frame, (left, right): (f32, f32), bottom: f32, narrow: bool) -> Option<(Vec2, Vec2)> {
        let size = 12.0;
        if f.view_size().y < 300.0 {
            return None;
        }
        let parts: Vec<&str> = match self.mode {
            Mode::Line => {
                let mut parts = vec!["yellow: the letter $F$, and its shadow $PF$", "blue: $HF$, its mirror image"];
                parts.push(if self.q_on { "dashed: the circle round $x$ that touches the line" } else { "dashed circle: as far from $0$ as $x$" });
                parts
            }
            Mode::Mirrors => vec!["yellow: the letter $F$", "blue: $H_1 F$, after mirror 1", "dashed: where it starts, and where both mirrors take it"],
        };
        let _ = narrow;
        let room = right - left - 36.0;
        let gap = "      ";
        let mut lines: Vec<String> = Vec::new();
        for part in parts {
            match lines.last_mut() {
                Some(last) if text_width(last, size) + text_width(gap, size) + text_width(part, size) <= room => {
                    last.push_str(gap);
                    last.push_str(part);
                }
                _ => lines.push(part.to_string()),
            }
        }
        let count = lines.len();
        let mut widest = 0.0f32;
        for (k, line) in lines.into_iter().enumerate() {
            let up = (count - 1 - k) as f32 * (size + 7.0);
            widest = widest.max(text_width(&line, size));
            f.caption_px(Vec2::new(left + 18.0, bottom - up), line, size, pal::DIM, Align2::LEFT_BOTTOM);
        }
        Some((Vec2::new(left + 10.0, bottom - count as f32 * (size + 7.0) - 4.0), Vec2::new(left + 26.0 + widest, bottom + 6.0)))
    }

    /// The screen panel: the matrices, what they say, and the chart of |x − q|.
    fn panel(&self, f: &mut Frame, p: &Panel) {
        hud_panel(f, p.min, p.max);
        let s = self.shown();
        let mats: [(DMat2, &str, Color); 4] = match self.mode {
            Mode::Line => {
                let pm = s.p();
                [(pm, "$P$", pal::ORANGE), (pm * pm, "$P^2$", pal::ORANGE), (DMat2::IDENTITY - pm, "$I - P$", pal::RED), (s.h(), "$H = 2P - I$", pal::BLUE)]
            }
            Mode::Mirrors => {
                let (h1, h2) = (s.h1(), s.h2());
                [(h1, "$H_1$", pal::BLUE.mix(pal::WHITE, 0.25)), (h2, "$H_2$", pal::TEXT), (h2 * h1, "$H_2 H_1$", pal::YELLOW), (h1 * h2, "$H_1 H_2$", pal::DIM)]
            }
        };
        for (k, (m, name, c)) in mats.iter().enumerate() {
            f.text_px(p.names[k], *name, 13.0, *c, Align2::CENTER_TOP);
            hud_matrix(f, p.mats[k], &entries(*m, *c), CELL, p.text, pal::DIM);
        }
        for (k, (text, c)) in self.panel_lines().into_iter().enumerate() {
            f.text_px(p.lines + Vec2::new(0.0, k as f32 * LINE_H), text, 12.0, c, Align2::LEFT_TOP);
        }
        if let (Some((title, (lo, hi))), Mode::Line) = (p.chart, self.mode) {
            f.text_px(title, r"$|x - q|$, as $q$ slides along the line", 12.0, pal::DIM, Align2::LEFT_TOP);
            hud_rect(f, lo, hi, Color::hex(0x10172A));
            let (u, x) = (s.u(), s.x);
            let dist = |t: f64| (x - u * t).length();
            let top = dist(-Q_MAX).max(dist(Q_MAX)).max(1e-6);
            let at = |t: f64, d: f64| Vec2::new(lo.x + (hi.x - lo.x) * ((t + Q_MAX) / (2.0 * Q_MAX)) as f32, hi.y - 4.0 - (hi.y - lo.y - 8.0) * (d / top).clamp(0.0, 1.0) as f32);
            hud_path(f, (0..=64).map(|i| -Q_MAX + 2.0 * Q_MAX * i as f64 / 64.0).map(|t| at(t, dist(t))), 1.6, pal::WHITE.a(0.8));
            // the nearest point, and where Px is
            let foot = s.foot();
            let least = at(foot, dist(foot));
            hud_line(f, Vec2::new(least.x, hi.y), least, 1.0, pal::DIM);
            let along = (s.p() * x).dot(u);
            let px = Vec2::new(at(along, 0.0).x, hi.y);
            if along.abs() <= Q_MAX {
                hud_line(f, px, px - Vec2::new(0.0, 7.0), 2.4, pal::ORANGE);
            }
            f.text_px(least + Vec2::new(5.0, -3.0), format!("least, ${}$", num(dist(foot), 2)), 11.0, pal::DIM, Align2::LEFT_BOTTOM);
            let q = at(s.q, dist(s.q));
            hud_rect_at(f, 2, q - Vec2::splat(3.5), q + Vec2::splat(3.5), pal::WHITE);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn close(m: DMat2, n: DMat2, tol: f64) -> bool {
        (m - n).to_cols_array().iter().all(|v| v.abs() <= tol)
    }

    /// The scene as an experiment's button leaves it, run at 60 frames a second until
    /// it has come to rest (or for `seconds` at most).
    fn run(k: usize, seconds: f64) -> Project {
        let mut scene = Project::create();
        scene.set_up(k);
        for _ in 0..(seconds * 60.0) as usize {
            scene.advance(1.0 / 60.0, 1.0 / 60.0);
            if !scene.animated() {
                break;
            }
        }
        scene
    }

    /// The projection along any w onto any line: P² = P, Pu = u, Pw = 0, I − P is a
    /// projection too (onto w, along u), and the trace is 1; and it is symmetric only
    /// when w is at right angles to the line.
    #[test]
    fn the_slanted_projection_is_a_projection() {
        let mut rng = Rng::new(7);
        let mut checked = 0;
        for _ in 0..2000 {
            let (a, b) = (rng.range(-7.0, 7.0) as f64, rng.range(-7.0, 7.0) as f64);
            let (u, w) = (DVec2::from_angle(a), DVec2::from_angle(b));
            let sin = u.perp_dot(w).abs();
            if sin < LEAST.sin() {
                continue;
            }
            let p = projection(u, w);
            let scale = 1.0 / sin;
            assert!(close(p * p, p, 1e-12 * scale * scale), "{a} {b}");
            assert!((p * u - u).length() < 1e-12 * scale && (p * w).length() < 1e-12 * scale, "{a} {b}");
            let q = DMat2::IDENTITY - p;
            assert!(close(q * q, q, 1e-12 * scale * scale) && (q * w - w).length() < 1e-12 * scale && (q * u).length() < 1e-12 * scale);
            assert!((p.x_axis.x + p.y_axis.y - 1.0).abs() < 1e-12 && p.determinant().abs() < 1e-12 * scale * scale);
            // the most it stretches anything is 1/sin of the angle between w and the line
            let most = (0..360).map(|k| (p * DVec2::from_angle(k as f64 * PI64 / 180.0)).length()).fold(0.0, f64::max);
            assert!(most <= scale * (1.0 + 1e-12) && most >= scale * 0.999, "{most} against {scale}");
            assert_eq!(symmetric(p), (u.dot(w)).abs() < 1e-9, "{a} {b}");
            // and its mirror undoes itself, keeping lengths only if P is symmetric
            let h = reflection(p);
            assert!(close(h * h, DMat2::IDENTITY, 1e-11 * scale * scale) && (h.determinant() + 1.0).abs() < 1e-11 * scale * scale);
            checked += 1;
        }
        assert!(checked > 1800);
        // the orthogonal one, the formula on the screen, is the same matrix
        for a in [0.0, 0.3, first_line(), 2.0, -1.1] {
            let u = DVec2::from_angle(a);
            assert!(close(orthogonal(u), projection(u, u.perp()), 1e-15));
            assert!(symmetric(orthogonal(u)) && orthogonal(u).y_axis.x == orthogonal(u).x_axis.y);
            let h = reflection(orthogonal(u));
            assert!(close(h.transpose() * h, DMat2::IDENTITY, 1e-15));
            // Householder's form: I − 2nnᵀ with n the unit normal
            let n = u.perp();
            assert!(close(h, DMat2::IDENTITY - DMat2::from_cols(n * n.x, n * n.y) * 2.0, 1e-15));
        }
    }

    /// The knob for w can be dragged anywhere and gives an angle that makes a
    /// projection: never along the line, snapped to the perpendicular near it.
    #[test]
    fn w_never_lies_along_the_line() {
        let line = first_line();
        for k in 0..=3600 {
            let angle = -7.0 + 14.0 * k as f64 / 3600.0;
            let rel = rel_for(line, angle);
            assert!((LEAST..=PI64 - LEAST).contains(&rel), "{angle}: {rel}");
            let s = Setting { rel, ..opening() };
            let p = s.p();
            assert!(p.as_mat2().is_finite() && size(p) <= 1.0 / LEAST.sin() + 1e-9);
            if ((angle - line).rem_euclid(TAU64) - FRAC_PI_2).abs() < SNAP * 0.99 {
                assert!(s.orthogonal(), "{angle}");
            }
        }
        // the warning's number at the least angle
        assert_eq!(num(1.0 / LEAST.sin(), 2), "28.65");
        for bad in [f64::NAN, f64::INFINITY] {
            let rel = rel_for(line, bad);
            assert!(rel.is_finite() && (LEAST..=PI64 - LEAST).contains(&rel));
        }
    }

    #[test]
    fn the_shadow_of_the_shadow_does_what_its_caption_says() {
        let scene = run(0, 30.0);
        let s = scene.shown();
        assert!(scene.film == Some([Step::P, Step::P]) && scene.film_t == FILM_SECONDS && !scene.playing);
        let p = s.p();
        // the four numbers, twice
        let shown = |m: DMat2| entries(m, pal::TEXT).iter().flatten().map(|e| e.0.clone()).collect::<Vec<_>>();
        assert_eq!(shown(p), ["$0.80$", "$0.40$", "$0.40$", "$0.20$"]);
        assert_eq!(shown(p * p), shown(p));
        assert!(close(p * p, p, 1e-15));
        // the first step takes every point along w onto the line; the second moves nothing
        let m = [p, p];
        for pt in f_points(Mode::Line).into_iter().chain([s.x]) {
            let landed = carried(m, 1.0) * pt;
            assert!(landed.perp_dot(s.u()).abs() < 1e-12 && (pt - landed).perp_dot(s.w()).abs() < 1e-12);
            for k in 0..=20 {
                assert!((carried(m, 1.0 + k as f64 / 20.0) * pt - landed).length() < 1e-14);
            }
        }
        // keeps u, kills w, trace 1
        assert!((p * s.u() - s.u()).length() < 1e-15 && (p * s.w()).length() < 1e-15);
        assert!((p.x_axis.x + p.y_axis.y - 1.0).abs() < 1e-15);
    }

    #[test]
    fn the_nearest_point_does_what_its_caption_says() {
        let mut scene = Project::create();
        scene.set_up(1);
        // q sweeps from one side of Px to the other, and back to it
        let mut seen = Vec::new();
        for _ in 0..(30 * 60) {
            scene.advance(1.0 / 60.0, 1.0 / 60.0);
            seen.push(scene.shown().q);
            if !scene.animated() {
                break;
            }
        }
        let s = scene.shown();
        let (u, x, px) = (s.u(), s.x, s.p() * s.x);
        assert!(scene.q_on && scene.sweep.is_none() && s.orthogonal());
        assert!((u * s.q - px).length() < 1e-12, "q ends at Px");
        let foot = px.dot(u);
        assert!(seen.iter().any(|q| *q < foot - 2.9) && seen.iter().any(|q| *q > foot + 1.3));
        // its distance to x is least there, 1.79, and there x − q is at right angles
        let dist = |t: f64| (x - u * t).length();
        let best = (0..=2000).map(|k| -Q_MAX + 2.0 * Q_MAX * k as f64 / 2000.0).min_by(|a, b| dist(*a).total_cmp(&dist(*b))).unwrap();
        assert!((best - foot).abs() < 0.004 && num(dist(foot), 2) == "1.79");
        assert!((x - px).dot(u).abs() < 1e-14);
        // slanted, Px is no longer the nearest point
        let slanted = Setting { rel: 0.75 * PI64, ..s };
        assert!(((slanted.p() * x).dot(u) - foot).abs() > 1.0);
    }

    #[test]
    fn the_slanted_projection_does_what_its_caption_says() {
        let scene = run(2, 30.0);
        let s = scene.shown();
        assert!(!s.orthogonal() && degrees(s.theta()) == "45.0°" && scene.film == Some([Step::P, Step::P]));
        let (p, h, w) = (s.p(), s.h(), s.w());
        assert!(close(p, rows([0.4, 1.2, 0.2, 0.6]), 1e-14) && close(p * p, p, 1e-14));
        assert!(!symmetric(p) && (w - DVec2::new(-3.0, 1.0).normalize()).length() < 1e-14);
        // the leftover runs along w, and P lengthens x, as much as it can at 45 degrees
        let (x, px) = (s.x, p * s.x);
        assert!((x - px).perp_dot(w).abs() < 1e-14);
        assert!(num(x.length(), 2) == "1.58" && num(px.length(), 2) == "2.24");
        assert!(num(px.length() / x.length(), 2) == "1.41" && (px.length() / x.length() - 1.0 / s.theta().sin()).abs() < 1e-12);
        // the mirror is no true mirror
        assert!(!close(h.transpose() * h, DMat2::IDENTITY, 0.1) && close(h * h, DMat2::IDENTITY, 1e-14));
        assert!((h * x).length() > 2.0 * x.length());
        // turned towards the line, P stretches more and more
        let mut last = 1.0 / s.theta().sin();
        for deg in [30.0, 15.0, 5.0, 2.0] {
            let more = 1.0 / Setting { rel: PI64 - deg * PI64 / 180.0, ..s }.theta().sin();
            assert!(more > last);
            last = more;
        }
    }

    #[test]
    fn the_mirror_does_what_its_caption_says() {
        let scene = run(3, 30.0);
        let s = scene.shown();
        let (p, h) = (s.p(), s.h());
        let (x, hx) = (s.x, h * s.x);
        assert!(x == DVec2::new(1.5, 2.0) && s.orthogonal());
        assert!(num(hx.x, 2) == "2.50" && num(hx.y, 2) == "0.00" && (hx.length() - x.length()).abs() < 1e-14);
        // the line is halfway between x and the x axis
        assert!((s.line - 0.5 * x.y.atan2(x.x)).abs() < 1e-14);
        assert!((hx - (p * x - (x - p * x))).length() < 1e-14);
        // Householder: I − 2nnᵀ, n along x − Hx
        let n = (x - hx).normalize();
        assert!(close(h, DMat2::IDENTITY - DMat2::from_cols(n * n.x, n * n.y) * 2.0, 1e-14));
        // half way through H, every point is at its shadow; H again brings it home
        assert!(close((DMat2::IDENTITY + h) * 0.5, p, 1e-15));
        let m = [h, h];
        for pt in f_points(Mode::Line).into_iter().chain([x]) {
            assert!((carried(m, 0.5) * pt - p * pt).length() < 1e-14);
            assert!((carried(m, 1.0) * pt - h * pt).length() < 1e-14);
            assert!((carried(m, 2.0) * pt - pt).length() < 1e-14);
        }
        assert!(scene.film == Some(m.map(|_| Step::H)) && scene.film_t == FILM_SECONDS);
    }

    #[test]
    fn two_mirrors_do_what_their_caption_says() {
        let scene = run(4, 30.0);
        let s = scene.shown();
        assert!(scene.mode == Mode::Mirrors && scene.film == Some([Step::Mirror1, Step::Mirror2]));
        assert!((s.line - 0.2).abs() < 1e-12 && (s.mirror - 0.5).abs() < 1e-12);
        let a = s.apart();
        assert!(num(a, 2) == "0.30" && degrees(a) == "17.2°");
        let turn = |m: DMat2| m.x_axis.y.atan2(m.x_axis.x);
        let (h1, h2) = (s.h1(), s.h2());
        assert!(close(h2 * h1, DMat2::from_angle(0.6), 1e-14) && num(turn(h2 * h1), 2) == "0.60" && degrees(0.6) == "34.4°");
        assert!(close(h1 * h2, DMat2::from_angle(-0.6), 1e-14));
        assert!((h1.determinant() + 1.0).abs() < 1e-14 && ((h2 * h1).determinant() - 1.0).abs() < 1e-14);
        // the film ends with the F turned by 0.6 about 0
        let end = carried([h1, h2], 2.0);
        for pt in f_points(Mode::Mirrors) {
            assert!((end * pt - DMat2::from_angle(0.6) * pt).length() < 1e-14);
        }
        // whatever the angle between the mirrors, the turn is twice it
        let mut rng = Rng::new(9);
        for _ in 0..200 {
            let t = Setting { line: rng.range(-4.0, 4.0) as f64, mirror: rng.range(-4.0, 4.0) as f64, ..s };
            assert!(close(t.h2() * t.h1(), DMat2::from_angle(2.0 * t.apart()), 1e-12));
        }
    }

    /// What each experiment is about lies in the part of the scene the view is fitted
    /// to: x, its shadow and mirror image, the F, its shadow and mirror images, the knobs.
    #[test]
    fn every_experiment_is_in_view() {
        let (lo, hi) = ((FRAME_CENTRE - FRAME_HALF).as_dvec2(), (FRAME_CENTRE + FRAME_HALF).as_dvec2());
        let inside = |p: DVec2| p.cmpge(lo).all() && p.cmple(hi).all();
        for k in 0..TRIALS.len() {
            let scene = run(k, 30.0);
            let s = scene.shown();
            let mut pts: Vec<DVec2> = Vec::new();
            let home = f_points(scene.mode);
            pts.extend(&home);
            pts.push(s.u() * KNOB_U);
            if scene.mode == Mode::Line {
                pts.extend([s.x, s.p() * s.x, s.h() * s.x, s.w() * KNOB_W]);
                pts.extend(home.iter().map(|q| s.h() * *q));
                pts.extend(home.iter().map(|q| s.p() * *q));
            } else {
                pts.push(DVec2::from_angle(s.mirror) * KNOB_U);
                pts.extend(home.iter().map(|q| s.h1() * *q));
                pts.extend(home.iter().map(|q| s.h2() * s.h1() * *q));
            }
            for p in pts {
                assert!(inside(p), "experiment {k}: {p}");
            }
        }
    }

    /// Degenerate input: x at 0, q at the ends, w as near the line as it goes, the
    /// mirrors on top of each other: every number drawn or read out is finite.
    #[test]
    fn degenerate_input_leaves_everything_finite() {
        for rel in [LEAST, PI64 - LEAST, FRAC_PI_2, 1.0] {
            for x in [DVec2::ZERO, DVec2::splat(X_MAX), DVec2::new(-X_MAX, 0.1)] {
                for mirror in [first_line(), first_line() + PI64, 0.0] {
                    let mut scene = Project::create();
                    scene.glide = Glide::at(Setting { rel, x, q: Q_MAX, mirror, ..opening() });
                    for film in [[Step::P, Step::P], [Step::H, Step::H], [Step::Mirror1, Step::Mirror2]] {
                        scene.play(film);
                        for _ in 0..400 {
                            scene.advance(1.0 / 60.0, 1.0 / 60.0);
                            assert!(scene.carried_drawn(&scene.shown()).iter().all(|p| p.as_vec2().is_finite()));
                        }
                    }
                    let lines = scene.panel_lines().iter().chain([scene.heading_lines()].iter().map(|h| &h.0).map(|t| (t.clone(), pal::TEXT)).collect::<Vec<_>>().iter()).map(|l| l.0.clone()).collect::<Vec<_>>().join(" ");
                    assert!(!lines.contains('?') && !lines.contains("NaN") && !lines.contains("inf"), "{lines}");
                    let s = scene.shown();
                    for m in [s.p(), s.h(), s.h1(), s.h2()] {
                        assert!(m.as_mat2().is_finite() && entries(m, pal::TEXT).iter().flatten().all(|e| !e.0.contains('?')));
                    }
                }
            }
        }
    }

    /// The panel and the picture fit the canvas, from a phone to a desktop.
    #[test]
    fn the_layout_fits_every_canvas() {
        for (w, h) in [(1080.0, 1000.0), (580.0, 700.0), (390.0, 800.0), (390.0, 560.0), (800.0, 600.0), (1600.0, 1000.0), (240.0, 180.0), (620.0, 480.0), (1100.0, 700.0)] {
            for inset in [0.0, 46.0] {
                for chart in [false, true] {
                    let view = Vec2::new(w, h);
                    let lay = Layout::of(view, inset, chart);
                    let (min, max) = lay.picture;
                    assert!(min.cmpge(Vec2::ZERO).all() && max.cmple(view).all() && (max - min).min_element() > 50.0, "{w}x{h}: {lay:?}");
                    if let Some(p) = &lay.panel {
                        assert!(p.min.cmpge(Vec2::ZERO).all() && p.max.cmple(view + Vec2::splat(0.5)).all(), "{w}x{h}: {p:?}");
                        assert!(p.min.y >= inset || lay.narrow, "{w}x{h}: under the buttons");
                        let right = p.mats[1].x + matrix_size(2, 2, CELL, p.text).x;
                        assert!(right <= p.max.x - PAD + 0.5 && p.mats[0].x >= p.min.x + PAD - 0.5, "{w}x{h}: {p:?}");
                        if let Some((_, (lo, hi))) = p.chart {
                            assert!(hi.y <= p.max.y && lo.x >= p.min.x && hi.x <= p.max.x);
                        }
                    }
                }
            }
        }
    }

    /// Two mirrors chosen by hand: the second 0.30 round from the first, wherever that
    /// is, and the letter glides to its place in that picture and back.
    #[test]
    fn two_mirrors_by_hand_start_apart() {
        let mut scene = Project::create();
        scene.drawn = scene.carried_points(&scene.shown());
        scene.set_mode(Mode::Mirrors);
        assert!(scene.morph.is_some());
        for _ in 0..120 {
            scene.advance(1.0 / 60.0, 1.0 / 60.0);
        }
        let s = scene.shown();
        assert!(!scene.animated() && (s.apart() - 0.3).abs() < 1e-12 && scene.film.is_none());
        assert!(scene.carried_points(&s).iter().zip(f_points(Mode::Mirrors)).all(|(a, b)| a.distance(b) < 1e-12));
        scene.set_mode(Mode::Line);
        assert_eq!(scene.mode, Mode::Line);
    }

    /// A drag of q and of x stops the sweep; an experiment's film waits for its glide.
    #[test]
    fn the_film_waits_for_the_glide_and_then_plays_once() {
        let mut scene = Project::create();
        scene.set_up(2);
        scene.advance(1.0 / 60.0, 1.0 / 60.0);
        assert!(scene.play_when_there && !scene.playing && scene.film_t == 0.0);
        let scene = run(2, 30.0);
        assert!(!scene.animated() && scene.film_t == FILM_SECONDS);
        // the sweep's path: from 3 before the foot to 1.4 past it, and back
        assert!(sweep_at(0.0) == -3.0 && (sweep_at(2.4) - 1.4).abs() < 1e-12 && sweep_at(100.0) == 0.0);
    }
}


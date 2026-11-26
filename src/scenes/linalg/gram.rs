//! Gram–Schmidt: three vectors made perpendicular one at a time, and the factorisation
//! A = QR that keeps the accounts of it.

use super::*;
use glam::DVec3;

pub const DEF: SceneDef = SceneDef {
    id: "gram",
    title: "Gram–Schmidt",
    group: "Linear algebra: equations",
    blurb: "Straighten three vectors into a perpendicular frame, one shadow at a time.",
    make: |g| Box::new(Gram::new(g)),
};

/// The arrows the scene opens with. (Every coordinate is a multiple of 1/16, so the
/// experiments built from them, such as a₂ − a₁/2, are exactly what they say.)
const START: [[f32; 3]; 3] = [[1.75, 0.375, 0.25], [1.125, 1.625, 0.25], [0.5, 1.0, 1.625]];

/// Experiments: (button, what to look for, the arrows, the step it starts from and
/// the step it plays to, the slanted box shown, the frame of the other order shown).
type Trial = (&'static str, &'static str, [[f32; 3]; 3], usize, usize, bool, bool);
const TRIALS: [Trial; 5] = [
    (
        "already perpendicular",
        "These three are at right angles to each other already. No arrow has a shadow on the ones before it, so nothing is taken away: $R$ is $0$ everywhere above its diagonal, and each $q$ is its $a$ scaled to length 1. The diagonal holds the three lengths.",
        [[2.0, 1.0, 0.25], [-0.75, 1.3125, 0.75], [0.15625, -0.625, 1.25]],
        8,
        8,
        false,
        false,
    ),
    (
        "nearly in line",
        "$a_2$ is only 5° from $a_1$, so its shadow on $q_1$ is nearly all of it: little is left ($r_{22} = 0.157$, against $a_2$'s length of 1.8), and $q_2$ is that little scaled up 6.4 times. Drag $a_2$'s tip a short way: $q_2$ swings round. A nudge of 0.05 can turn $q_2$ by as much as 19°; with the arrows the scene opens with, by 2° at most.",
        [START[0], [1.703, 0.525, 0.254], START[2]],
        1,
        4,
        false,
        false,
    ),
    (
        "three in one plane",
        r"$a_3$ is now $a_2 - a_1/2$, so it lies in the plane of $a_1$ and $a_2$ (the blue sheet). Its two shadows make up all of it: nothing is left, $r_{33} = 0$, and there is no $q_3$. Only two perpendicular directions can be made from these three arrows: the rank is 2, and the box they span is flat ($\det A = 0$).",
        [START[0], START[1], [0.25, 1.4375, 0.125]],
        4,
        7,
        false,
        false,
    ),
    (
        "another order",
        "The same three arrows, made perpendicular a second time in the order $a_3, a_2, a_1$ (the cyan arrows $q_1', q_2', q_3'$). Now $q_1'$ lies along $a_3$, and the frame is a different one, at right angles all the same. The lengths left on the way differ too (see the numbers), but their product does not: it is the volume of the same box.",
        START,
        8,
        8,
        false,
        true,
    ),
    (
        "the box's volume",
        r"The three arrows span a slanted box, and $\lvert \det A \rvert$ is its volume: $r_{11} \times r_{22} \times r_{33}$. That is the length of $a_1$, times the height of $a_2$ above the line of $a_1$ (the last leg of $a_2$'s walk), times the height of $a_3$ above the floor of the box (the last leg of $a_3$'s walk): $\text{length} \times \text{height} \times \text{height}$.",
        START,
        8,
        8,
        true,
        false,
    ),
];

/// What is left of a vector, once its shadows are taken away, counts as nothing when
/// it is shorter than this share of the vector's own length. The arrows' coordinates
/// are kept to about seven significant digits, and rounding them alone can leave an
/// arrow put in the plane of two others a ten-millionth of its length off it: it
/// should still count as in the plane. (Arithmetic in f64 adds far less.)
const NOTHING: f64 = 1e-6;

/// The steps, and what each is called over the picture. Step k (1 to 8) plays as the
/// stage goes from k − 1 to k.
const STEPS: usize = 8;
const TITLES: [&str; STEPS + 1] = [
    "Three vectors $a_1, a_2, a_3$",
    "Scale $a_1$ to length 1: that is $q_1$",
    "Drop $a_2$'s shadow on $q_1$",
    "Take the shadow away from $a_2$",
    "Scale what is left to length 1: $q_2$",
    "Drop $a_3$'s shadows on $q_1$ and $q_2$",
    "Take both shadows away from $a_3$",
    "Scale what is left to length 1: $q_3$",
    "Put back together: $A = QR$",
];

/// Seconds a step takes to play forwards and backwards, and the pause after each one
/// when they are played through.
const STEP_TIME: f32 = 1.4;
const BACK_TIME: f32 = 0.5;
const HOLD: f32 = 1.0;

/// The colours of q₁, q₂, q₃; the vectors they are made from, a₁, a₂, a₃, have paler
/// versions of the same.
const Q: [Color; 3] = [pal::PINK, pal::GREEN, pal::PURPLE];
/// What is left of a vector once its shadows are taken away.
const LEFT: Color = pal::YELLOW;
/// The box the three arrows span.
const BOX: Color = Color::hex(0x8A93B8);
/// The frame made from the same arrows in the order a₃, a₂, a₁.
const OTHER: Color = pal::CYAN;
const NAMES_A: [&str; 3] = ["$a_1$", "$a_2$", "$a_3$"];
const NAMES_Q: [&str; 3] = ["$q_1$", "$q_2$", "$q_3$"];

fn pale(i: usize) -> Color {
    Q[i].mix(pal::DIM, 0.5)
}

/// Gram–Schmidt on three vectors, in f64.
#[derive(Clone, Copy, Debug)]
struct Gs {
    /// The perpendicular unit vectors made, or None where nothing was left to make one.
    q: [Option<DVec3>; 3],
    /// R: r[i][j] = qᵢ · aⱼ above the diagonal (0 where there is no qᵢ), the length of
    /// what was left on it, and 0 below it.
    r: [[f64; 3]; 3],
    /// What was left of each aⱼ once its shadows were taken away.
    left: [DVec3; 3],
    /// The sum of those shadows, where aⱼ's tip drops onto the span of the q's before it.
    foot: [DVec3; 3],
}

impl Gs {
    fn of(a: [DVec3; 3]) -> Gs {
        let mut gs = Gs { q: [None; 3], r: [[0.0; 3]; 3], left: [DVec3::ZERO; 3], foot: [DVec3::ZERO; 3] };
        for j in 0..3 {
            let mut v = a[j];
            // the shadows on the q's made so far, taken away; and then once more what
            // rounding left of them, so that what is left is at right angles to the
            // q's to the last digit even when it is a small difference of large numbers
            for pass in 0..2 {
                for i in 0..j {
                    if let Some(qi) = gs.q[i] {
                        let c = qi.dot(if pass == 0 { a[j] } else { v });
                        gs.r[i][j] += c;
                        v -= qi * c;
                    }
                }
            }
            let len = v.length();
            gs.left[j] = v;
            gs.foot[j] = a[j] - v;
            if len > NOTHING * a[j].length() {
                gs.q[j] = Some(v / len);
                gs.r[j][j] = len;
            }
        }
        gs
    }

    fn of_arrows(a: [Vec3; 3]) -> Gs {
        Gs::of(a.map(|v| v.as_dvec3()))
    }

    /// How many perpendicular directions could be made.
    fn rank(&self) -> usize {
        self.q.iter().filter(|q| q.is_some()).count()
    }

    /// r₁₁ r₂₂ r₃₃: the volume of the box the arrows span.
    fn volume(&self) -> f64 {
        self.r[0][0] * self.r[1][1] * self.r[2][2]
    }
}

/// The signed volume of the box spanned by three arrows: det A.
fn det(a: [DVec3; 3]) -> f64 {
    a[0].dot(a[1].cross(a[2]))
}

/// An entry of R as written: a plain 0 for what is zero but for rounding.
fn entry_text(v: f64, scale: f64) -> String {
    if v.abs() <= 1e-12 * scale {
        "0".into()
    } else {
        fmt(v as f32)
    }
}

/// Which step makes entry (i, j) of R, and so from which stage on it is written.
fn made_by(i: usize, j: usize) -> usize {
    if i == j {
        3 * j + 1
    } else {
        (3 * j).saturating_sub(1)
    }
}

/// The entries of R a step works with, picked out while it plays and after.
fn picked(step: usize) -> &'static [(usize, usize)] {
    match step {
        1 => &[(0, 0)],
        2 | 3 => &[(0, 1)],
        4 => &[(1, 1)],
        5 | 6 => &[(0, 2), (1, 2)],
        7 => &[(2, 2)],
        _ => &[],
    }
}

/// What the shadows of aⱼ lie in: the span of the q's made before it.
fn span_name(gs: &Gs, j: usize) -> &'static str {
    match (j > 0 && gs.q[0].is_some(), j > 1 && gs.q[1].is_some()) {
        (true, true) => "the plane of $q_1$ and $q_2$",
        (true, false) => "the line of $q_1$",
        (false, true) => "the line of $q_2$",
        (false, false) => "nothing yet",
    }
}

/// The two lines under R for a step: what its numbers are and where they come from.
fn detail(step: usize, gs: &Gs, scale: f64) -> [String; 2] {
    let r = |i: usize, j: usize| entry_text(gs.r[i][j], scale);
    let no = |i: usize| gs.q[i].is_none();
    let lines = |a: &str, b: &str| [a.to_string(), b.to_string()];
    match step {
        0 => lines("Gram–Schmidt takes them one at a time.", "next ▶ takes a step; drag a tip to change an arrow."),
        1 if no(0) => lines("$a_1$ is $0$: there is nothing to scale,", "and no $q_1$."),
        1 => [
            "Nothing has been made yet, so all of $a_1$ is left.".into(),
            format!(r"$r_{{11}} = \lVert a_1 \rVert = {}$, and $q_1 = a_1 / r_{{11}}$", r(0, 0)),
        ],
        2 if no(0) => lines("There is no $q_1$ for $a_2$ to cast a shadow on:", "$r_{12} = 0$, and all of $a_2$ is left."),
        2 => [
            fitting(format!(r"$r_{{12}} = q_1 \cdot a_2 = {}$: the shadow is $r_{{12}} q_1$,", r(0, 1)), || format!("$r_{{12}} = {}$: the shadow is $r_{{12}} q_1$,", r(0, 1))),
            "where $a_2$'s tip drops onto the line of $q_1$.".into(),
        ],
        3 if no(0) => lines("Nothing is taken away: all of $a_2$ is left.", "(There is no $q_1$ to be at right angles to.)"),
        3 if no(1) => [
            "Taking the shadow away leaves nothing:".into(),
            format!("$a_2$ lies on {}.", span_name(gs, 1)),
        ],
        3 => lines("What is left, $a_2 - r_{12} q_1$, is at right angles", "to $q_1$. It slides down to start at $0$."),
        4 if no(1) => lines("Nothing is left ($r_{22} = 0$), so there is no $q_2$.", "$a_2$ adds no new direction."),
        4 => [format!(r"$r_{{22}} = \text{{its length}} = {}$, and", r(1, 1)), "$q_2 = (a_2 - r_{12} q_1) / r_{22}$".into()],
        5 | 6 if no(0) && no(1) => lines("There are no $q$'s yet to cast a shadow on:", "$r_{13} = r_{23} = 0$, and all of $a_3$ is left."),
        5 => {
            // (each with where it comes from, if the line has room for that)
            let shadow = |i: usize, whence: bool| match (no(i), whence) {
                (true, _) => format!("$r_{{{}3}} = 0$ (no $q_{}$)", i + 1, i + 1),
                (false, true) => format!(r"$r_{{{}3}} = q_{} \cdot a_3 = {}$", i + 1, i + 1, r(i, 2)),
                (false, false) => format!("$r_{{{}3}} = {}$", i + 1, r(i, 2)),
            };
            let both = |whence: bool| format!("{},  {}:", shadow(0, whence), shadow(1, whence));
            [fitting(both(true), || both(false)), format!("$a_3$'s tip drops onto {}.", span_name(gs, 2))]
        }
        6 if no(2) => [
            "Taking the shadows away leaves nothing:".into(),
            format!("$a_3$ lies {} {}.", if no(0) || no(1) { "on" } else { "in" }, span_name(gs, 2)),
        ],
        6 => ["What is left, $a_3 - r_{13} q_1 - r_{23} q_2$, is at".into(), format!("right angles to {}.", span_name(gs, 2))],
        7 if no(2) => [
            "Nothing is left ($r_{33} = 0$), so there is no $q_3$.".into(),
            format!("Only {} perpendicular directions: rank {}.", ["no", "one", "two", "three"][gs.rank()], gs.rank()),
        ],
        7 => [format!(r"$r_{{33}} = \text{{its length}} = {}$, and", r(2, 2)), "$q_3 = (a_3 - r_{13} q_1 - r_{23} q_2) / r_{33}$".into()],
        _ => [
            "Each column of $R$ rebuilds its $a$ from the $q$'s:".into(),
            format!("$a_3 = {} q_1 + {} q_2 + {} q_3$", r(0, 2), r(1, 2), r(2, 2)).replace("+ -", "− "),
        ],
    }
}

/// How wide a line under R may be.
const LINE_ROOM: f32 = PANEL_W - 28.0;

/// A line under R, or a shorter one in its place where it is too wide for the panel
/// (as a number written with a power of ten can make it).
fn fitting(line: String, shorter: impl FnOnce() -> String) -> String {
    if estimate(&line, PANEL_TEXT) <= LINE_ROOM {
        line
    } else {
        shorter()
    }
}

/// The line about the volume under R, once the diagonal is complete: as much of it as
/// fits in `room` logical pixels.
fn volume_line(gs: &Gs, scale: f64, room: f32) -> String {
    let d: Vec<String> = (0..3).map(|i| entry_text(gs.r[i][i], scale)).collect();
    let volume = entry_text(gs.volume(), scale * scale * scale);
    let flat = if gs.rank() < 3 { ": the box is flat" } else { "" };
    // (the numbers multiplied with a centred dot: with crosses, set as TeX sets them,
    // the whole line no longer fits)
    let long = format!(r"$\lvert \det A \rvert = r_{{11}} r_{{22}} r_{{33}} = {} \cdot {} \cdot {} = {volume}${flat}", d[0], d[1], d[2]);
    let medium = format!(r"$\lvert \det A \rvert = {} \cdot {} \cdot {} = {volume}$", d[0], d[1], d[2]);
    let short = format!(r"$\lvert \det A \rvert = r_{{11}} r_{{22}} r_{{33}} = {volume}$");
    [long, medium].into_iter().find(|t| estimate(t, PANEL_TEXT) <= room).unwrap_or(short)
}

/// The size of the panel's smaller text.
const PANEL_TEXT: f32 = 11.5;

/// The line under R before the diagonal is complete: what its rows and columns are.
const ROWS_AND_COLUMNS: &str = "rows: amounts of $q_1, q_2, q_3$;  columns: $a_1, a_2, a_3$";

/// Where the panel of R and the step buttons goes on a canvas of `view` logical
/// pixels, `inset` of whose top lies under the application's buttons: over the
/// picture's top-left corner, or below the picture on a canvas narrow and tall enough
/// (a phone held upright). Left out where the canvas cannot spare the room.
#[derive(Clone, Copy, Debug, PartialEq)]
struct Layout {
    narrow: bool,
    panel: Option<(Vec2, Vec2)>,
    /// Where the picture's part of the canvas ends at the bottom.
    below: f32,
}

const PANEL_W: f32 = 320.0;
const PANEL_H: f32 = 240.0;

impl Layout {
    fn of(view: Vec2, inset: f32) -> Layout {
        if view.x < 620.0 && view.y >= 560.0 {
            let top = view.y - PANEL_H - 10.0;
            Layout { narrow: true, panel: Some((Vec2::new(8.0, top), Vec2::new(view.x - 8.0, view.y - 10.0))), below: top - 4.0 }
        } else {
            let top = 14.0f32.max(inset + 8.0);
            let fits = view.x >= PANEL_W + 160.0 && view.y >= top + PANEL_H + 60.0;
            Layout { narrow: false, panel: fits.then(|| (Vec2::new(14.0, top), Vec2::new(14.0 + PANEL_W, top + PANEL_H))), below: view.y }
        }
    }

    fn covers(&self, p: Vec2) -> bool {
        self.panel.is_some_and(|(min, max)| p.cmpge(min).all() && p.cmple(max).all())
    }
}

thread_local! {
    /// The widths of texts with mathematics in them, as `measure` and `estimate` found
    /// them, and at what size: the names and lines here are measured every frame, and
    /// a width looked up here costs a fraction of one worked out afresh. (Measured and
    /// estimated widths are kept apart.)
    static MEASURED: std::cell::RefCell<std::collections::HashMap<String, (f32, f32)>> = std::cell::RefCell::new(std::collections::HashMap::new());
    static ESTIMATED: std::cell::RefCell<std::collections::HashMap<String, (f32, f32)>> = std::cell::RefCell::new(std::collections::HashMap::new());
}

/// A width from `table`, or worked out by `work` (and kept, if there is one).
fn kept_width(table: &'static std::thread::LocalKey<std::cell::RefCell<std::collections::HashMap<String, (f32, f32)>>>, text: &str, size: f32, work: impl FnOnce() -> Option<f32>) -> Option<f32> {
    table.with(|kept| {
        if let Some(&(at, width)) = kept.borrow().get(text) {
            if at == size {
                return Some(width);
            }
        }
        let width = work()?;
        let mut kept = kept.borrow_mut();
        if kept.len() >= 1024 {
            kept.clear();
        }
        kept.insert(text.to_string(), (size, width));
        Some(width)
    })
}

/// How wide a text is drawn, logical pixels: one with mathematics in it measured as it
/// is set (and kept), plain words estimated as before.
fn measure(f: &Frame, text: &str, size: f32) -> f32 {
    if !crate::typeset::has_math(text) {
        return text_width(text, size);
    }
    // (nothing to keep before the fonts are in place: estimated till then)
    kept_width(&MEASURED, text, size, || crate::typeset::measure(f.ctx, text, size).map(|s| s.x)).unwrap_or_else(|| text_width(text, size))
}

/// `text_width`, kept: for the lines under R, which are chosen by their width every
/// frame without a frame to measure them by.
fn estimate(text: &str, size: f32) -> f32 {
    if !crate::typeset::has_math(text) {
        return text_width(text, size);
    }
    kept_width(&ESTIMATED, text, size, || Some(text_width(text, size))).unwrap_or(0.0)
}

/// The direction the scene is seen from at first: in front, to the right and above.
const YAW: f32 = -0.85;
const PITCH: f32 = 0.55;
/// (The tangent of half the camera's field of view.)
const TAN_HALF_FOV: f32 = 0.3203;

/// The camera's right and up directions in the view the scene opens with.
fn screen_axes() -> (Vec3, Vec3) {
    let right = Vec3::new(-YAW.sin(), YAW.cos(), 0.0);
    let up = Vec3::new(-PITCH.sin() * YAW.cos(), -PITCH.sin() * YAW.sin(), PITCH.cos());
    (right, up)
}

pub struct Gram {
    a: [Vec3; 3],
    /// How far through the steps the picture is (0 to `STEPS`), and where it is going.
    stage: f32,
    goal: f32,
    /// Playing through with a pause after each step, and what is left of the pause.
    through: bool,
    hold: f32,
    show_box: bool,
    show_other: bool,
    /// The canvas last drawn on, and how much of its top the application's buttons
    /// cover: they decide where the panel goes, and with it where the picture goes.
    canvas: Vec2,
    inset: f32,
    /// Whether a canvas has been drawn on yet (until then `canvas` is a guess).
    measured: bool,
    new_view: bool,
}

impl Gram {
    pub fn new(_: &Gfx) -> Self {
        Self::create()
    }

    /// (Nothing here needs the graphics device, so the tests can make one too.)
    fn create() -> Self {
        Gram {
            a: START.map(Vec3::from),
            stage: STEPS as f32,
            goal: STEPS as f32,
            through: false,
            hold: 0.0,
            show_box: false,
            show_other: false,
            canvas: Vec2::new(1080.0, 1000.0),
            inset: 0.0,
            measured: false,
            new_view: false,
        }
    }

    /// Set up one of the experiments in `TRIALS`.
    fn set_up(&mut self, k: usize) {
        let (_, _, a, from, to, show_box, show_other) = TRIALS[k];
        self.a = a.map(Vec3::from);
        (self.show_box, self.show_other) = (show_box, show_other);
        (self.stage, self.goal, self.through) = (from as f32, to as f32, true);
        self.hold = if from < to { HOLD } else { 0.0 };
        // (and the camera takes in what the lesson shows)
        self.new_view = true;
    }

    fn solve(&self) -> Gs {
        Gs::of_arrows(self.a)
    }

    /// The largest length among the arrows (what "zero but for rounding" is measured against).
    fn scale(&self) -> f64 {
        self.a.iter().map(|a| a.length() as f64).fold(1e-300, f64::max)
    }

    /// One step on: after the steps already asked for, or, while they are being played
    /// through, after the one playing (and there it stops).
    fn next(&mut self) {
        let from = if self.through { self.stage } else { self.goal.max(self.stage) };
        self.goal = (from.floor() + 1.0).min(STEPS as f32);
        (self.through, self.hold) = (false, 0.0);
    }

    fn can_go_on(&self) -> bool {
        (if self.through { self.stage } else { self.goal.max(self.stage) }) < STEPS as f32
    }

    fn back(&mut self) {
        self.goal = (self.goal.min(self.stage).ceil() - 1.0).max(0.0);
        (self.through, self.hold) = (false, 0.0);
    }

    fn replay(&mut self) {
        (self.stage, self.goal, self.through, self.hold) = (0.0, STEPS as f32, true, HOLD * 0.6);
    }

    fn moving(&self) -> bool {
        self.stage != self.goal || self.hold > 0.0
    }

    /// Move the steps on by `dt` seconds of scene time: towards the goal, a step at a
    /// time, pausing after each when playing through. (All the time a frame owes is
    /// spent, across as many steps and pauses as it reaches.)
    fn advance(&mut self, dt: f32) {
        let mut dt = dt.max(0.0);
        while dt > 0.0 && self.moving() {
            if self.hold > 0.0 {
                let wait = self.hold.min(dt);
                self.hold -= wait;
                dt -= wait;
                if self.hold <= 0.0 {
                    self.hold = 0.0;
                }
                continue;
            }
            let forward = self.goal > self.stage;
            let (next, time) = if forward { ((self.stage.floor() + 1.0).min(self.goal), STEP_TIME) } else { ((self.stage.ceil() - 1.0).max(self.goal), BACK_TIME) };
            let need = (next - self.stage).abs() * time;
            if dt >= need {
                self.stage = next;
                dt -= need;
                if self.through && forward && self.stage < self.goal {
                    self.hold = HOLD;
                }
            } else {
                self.stage += if forward { dt / time } else { -dt / time };
                dt = 0.0;
            }
        }
        if !self.moving() {
            self.through = false;
        }
    }

    /// The step being shown: the one playing, or the last one finished.
    fn step(&self) -> usize {
        (self.stage.ceil() as usize).min(STEPS)
    }

    /// What the camera should take in: the middle of the picture, and its size across
    /// and up (world units, as seen from the opening direction), room for the names
    /// included. The arrows, their frame, the walks back to them and the box if shown.
    fn framing(&self) -> (Vec3, Vec2) {
        let (right, up) = screen_axes();
        let gs = self.solve();
        let mut points: Vec<Vec3> = vec![Vec3::ZERO];
        points.extend(self.a);
        points.extend(gs.q.iter().flatten().map(|q| q.as_vec3()));
        points.extend(gs.foot.map(|p| p.as_vec3()));
        // (what is left of each, as it lies when slid down to 0)
        points.extend(gs.left.map(|p| p.as_vec3()));
        if self.show_box {
            let [a, b, c] = self.a;
            points.extend([a + b, a + c, b + c, a + b + c]);
        }
        let (mut lo, mut hi) = (Vec2::splat(f32::MAX), Vec2::splat(f32::MIN));
        for p in points.iter().filter(|p| p.is_finite()) {
            let s = Vec2::new(p.dot(right), p.dot(up));
            (lo, hi) = (lo.min(s), hi.max(s));
        }
        let mid = (lo + hi) * 0.5;
        let size = (hi - lo).max(Vec2::splat(0.5)) * 1.08 + Vec2::new(0.55, 0.45);
        (right * mid.x + up * mid.y, size)
    }

    fn the_view(&self) -> View {
        let lay = Layout::of(self.canvas, self.inset);
        let (right, up) = screen_axes();
        let (middle, size) = self.framing();
        // the part of the canvas the picture has to itself: below the application's
        // buttons, and above the panel on a narrow canvas or to the right of it on a wide one
        // (and above the caption along its bottom)
        let caption = if self.canvas.y >= 300.0 { 30.0 } else { 0.0 };
        let (min, max) = match (lay.narrow, lay.panel) {
            (true, _) => (Vec2::new(0.0, self.inset + 8.0), Vec2::new(self.canvas.x, lay.below - caption)),
            (false, Some((_, corner))) => (Vec2::new(corner.x + 10.0, self.inset), self.canvas - Vec2::Y * caption),
            (false, None) => (Vec2::new(0.0, self.inset), self.canvas - Vec2::Y * caption),
        };
        let region = (max - min).max(Vec2::splat(60.0));
        let per_unit = (region.x / size.x).min(region.y / size.y).max(1.0);
        let dist = (self.canvas.y * 0.5 / per_unit / TAN_HALF_FOV).clamp(2.0, 60.0);
        let per_unit = self.canvas.y * 0.5 / (dist * TAN_HALF_FOV);
        // move the middle of the picture to the middle of its part of the canvas
        let shift = ((min + max) * 0.5 - self.canvas * 0.5) / per_unit;
        View::Orbit { target: middle - right * shift.x + up * shift.y, yaw: YAW, pitch: PITCH, dist }
    }
}

/// A flat sheet in space, like `sheet` in this topic's module, but faded by `alpha`
/// (as it comes into the picture and leaves it).
fn faded_sheet(f: &mut Frame, u: Vec3, v: Vec3, half: f32, lines: i32, color: Color, alpha: f32) {
    let q = |s: f32, t: f32| u * s + v * t;
    f.dl.tris_alpha.quad(q(-half, -half), q(half, -half), q(half, half), q(-half, half), color.a(0.14 * alpha));
    for k in -lines..=lines {
        let s = k as f32 * half / lines.max(1) as f32;
        f.line(q(s, -half), q(s, half), 1.0, color.a(0.32 * alpha));
        f.line(q(-half, s), q(half, s), 1.0, color.a(0.32 * alpha));
    }
}

/// A shadow: some number times a q, drawn dashed in that q's colour on a faint band
/// of it (so that it shows where it lies along the q itself).
fn shadow_leg(f: &mut Frame, from: Vec3, to: Vec3, color: Color, alpha: f32) {
    f.line(from, to, 8.0, color.a(0.2 * alpha));
    f.dashed(from, to, 2.2, color.a(alpha));
}

/// A right angle at `at` between the directions `d1` and `d2`: the corner of a small
/// square, faintly filled.
fn corner(f: &mut Frame, at: Vec3, d1: Vec3, d2: Vec3, size: f32, alpha: f32) {
    // (never larger than half of either side it marks)
    let size = size.min(0.5 * d1.length()).min(0.5 * d2.length());
    if size <= 1e-6 {
        return;
    }
    let (a, b) = (d1.normalize_or_zero() * size, d2.normalize_or_zero() * size);
    f.dl.tris_alpha.quad(at, at + a, at + a + b, at + b, pal::WHITE.a(0.1 * alpha));
    right_angle(f, at, d1, d2, size, pal::WHITE.a(0.8 * alpha));
}

/// The box spanned by `a`, `b`, `c` from the origin: translucent faces, each pair of
/// opposite ones in its own tone so that the shape reads as a solid, drawn back to
/// front, and its twelve edges.
fn slanted_box(f: &mut Frame, a: Vec3, b: Vec3, c: Vec3) {
    let light = Vec3::new(0.55, -0.25, 0.8).normalize();
    let eye = f.cam.eye;
    let mut faces = [(Vec3::ZERO, a, b), (c, a, b), (Vec3::ZERO, a, c), (b, a, c), (Vec3::ZERO, b, c), (a, b, c)];
    let far = |(o, u, v): &(Vec3, Vec3, Vec3)| (*o + (*u + *v) * 0.5).distance(eye);
    faces.sort_by(|p, q| far(q).total_cmp(&far(p)));
    for (o, u, v) in faces {
        let tone = 0.35 + u.cross(v).normalize_or_zero().dot(light).abs().powf(1.5) * 0.65;
        f.dl.tris_alpha.quad(o, o + u, o + u + v, o + v, Color::from_linear(BOX.to_linear() * tone).a(0.3));
    }
    for (o, d) in [(Vec3::ZERO, a), (b, a), (c, a), (b + c, a), (Vec3::ZERO, b), (a, b), (c, b), (a + c, b), (Vec3::ZERO, c), (a, c), (b, c), (a + b, c)] {
        f.line(o, o + d, 1.2, BOX.a(0.7));
    }
}

/// Names placed beside points without covering each other, the panel, or the lines
/// already drawn there (screen segments in logical pixels), and within the view (below
/// `top`, where the application's buttons end).
struct Names<'l> {
    busy: Vec<(Vec2, Vec2)>,
    lay: &'l Layout,
    view: Vec2,
    top: f32,
}

impl Names<'_> {
    fn keep_clear(&mut self, f: &Frame, a: Vec3, b: Vec3) {
        if let (Some(a), Some(b)) = (on_screen(f, a), on_screen(f, b)) {
            self.busy.push((a, b));
        }
    }

    /// Name the point `tip` on whichever of its four diagonal sides is clearest; up
    /// and to the right unless another side is clearly better.
    fn name(&mut self, f: &mut Frame, tip: Vec3, text: &str, size: f32, color: Color) {
        let Some(at) = on_screen(f, tip) else { return };
        let half = Vec2::new(measure(f, text, size) * 0.5, size * 0.5);
        let mut best = (f32::MIN, Vec2::ZERO, Vec2::ZERO, Align2::LEFT_BOTTOM);
        for (side, align) in [
            (Vec2::new(1.0, -1.0), Align2::LEFT_BOTTOM),
            (Vec2::new(-1.0, -1.0), Align2::RIGHT_BOTTOM),
            (Vec2::new(1.0, 1.0), Align2::LEFT_TOP),
            (Vec2::new(-1.0, 1.0), Align2::RIGHT_TOP),
        ] {
            let corner = at + side * 8.0;
            let centre = corner + side * half;
            let ends = [centre - Vec2::X * half.x, centre, centre + Vec2::X * half.x];
            let room = ends.iter().flat_map(|q| self.busy.iter().map(move |(a, b)| seg_dist(*q, *a, *b))).fold(f32::MAX, f32::min);
            // (a side that would put the name on the panel, or partly out of view, is
            // no side at all)
            let (lo, hi) = (centre - half, centre + half);
            let out = lo.x < 2.0 || lo.y < self.top || hi.x > self.view.x - 2.0 || hi.y > self.view.y - 2.0;
            let room = if out || [lo, hi, Vec2::new(lo.x, hi.y), Vec2::new(hi.x, lo.y)].iter().any(|e| self.lay.covers(*e)) { f32::MIN / 2.0 } else { room };
            if room > best.0 + 4.0 {
                best = (room, corner, centre, align);
            }
        }
        if best.0 <= f32::MIN / 2.0 {
            return;
        }
        self.busy.push((best.2 - Vec2::X * half.x, best.2 + Vec2::X * half.x));
        f.text_px(best.1, text, size, color, best.3);
    }
}

/// A button drawn over the picture (logical pixels): true when it is clicked. One that
/// cannot be used now is drawn dim and ignores clicks.
fn hud_button(f: &mut Frame, min: Vec2, max: Vec2, text: &str, usable: bool, pointer: Option<Vec2>) -> bool {
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

impl Gram {
    /// The panel: the step and its buttons, R with the entries of this step picked
    /// out, and what they are.
    fn panel(&mut self, f: &mut Frame, gs: &Gs, lay: &Layout, pointer: Option<Vec2>) {
        let Some((min, max)) = lay.panel else { return };
        hud_panel(f, min, max);
        let scale = self.scale();
        let step = self.step();
        let x = min.x + 14.0;
        f.text_px(Vec2::new(x, min.y + 10.0), format!("Step {step} of {STEPS}"), 11.0, pal::DIM, Align2::LEFT_TOP);
        f.text_px(Vec2::new(x, min.y + 25.0), TITLES[step], 14.0, pal::TEXT, Align2::LEFT_TOP);

        // the buttons
        let y = min.y + 50.0;
        let at_start = self.goal.min(self.stage) <= 0.0;
        if hud_button(f, Vec2::new(x, y), Vec2::new(x + 72.0, y + 26.0), "◀ back", !at_start, pointer) {
            self.back();
        }
        if hud_button(f, Vec2::new(x + 80.0, y), Vec2::new(x + 152.0, y + 26.0), "next ▶", self.can_go_on(), pointer) {
            self.next();
        }
        if hud_button(f, Vec2::new(x + 160.0, y), Vec2::new(x + 250.0, y + 26.0), "↺ replay all", true, pointer) {
            self.replay();
        }

        // R, its rows the q's and its columns the a's
        let top = y + 40.0;
        let (cell, row) = (62.0, 20.0);
        let left = x + 54.0;
        // ("R =" ends clear of the rows' names, as wide as they are set)
        let names = NAMES_Q.iter().map(|q| measure(f, q, 12.5)).fold(0.0, f32::max);
        f.text_px(Vec2::new(left - 12.0 - names - 6.0, top + row * 2.5), "$R =$", 15.0, pal::TEXT, Align2::RIGHT_CENTER);
        for j in 0..3 {
            f.text_px(Vec2::new(left + cell * (j as f32 + 0.5), top + row * 0.5), NAMES_A[j], 12.5, pale(j).mix(pal::WHITE, 0.25), Align2::CENTER_CENTER);
        }
        // brackets
        let (b0, b1) = (Vec2::new(left - 6.0, top + row + 1.0), Vec2::new(left + cell * 3.0 + 6.0, top + row * 4.0 - 1.0));
        for (xb, d) in [(b0.x, 5.0), (b1.x, -5.0)] {
            hud_line(f, Vec2::new(xb + d, b0.y), Vec2::new(xb, b0.y), 1.2, pal::DIM);
            hud_line(f, Vec2::new(xb, b0.y), Vec2::new(xb, b1.y), 1.2, pal::DIM);
            hud_line(f, Vec2::new(xb, b1.y), Vec2::new(xb + d, b1.y), 1.2, pal::DIM);
        }
        let shown = self.stage;
        let now = picked(step);
        for i in 0..3 {
            let yc = top + row * (i as f32 + 1.5);
            f.text_px(Vec2::new(left - 12.0, yc), NAMES_Q[i], 12.5, Q[i], Align2::RIGHT_CENTER);
            for j in 0..3 {
                let c = Vec2::new(left + cell * (j as f32 + 0.5), yc);
                if i > j {
                    f.text_px(c, "$0$", 13.0, pal::FAINT, Align2::CENTER_CENTER);
                    continue;
                }
                // written from the moment the step that makes it starts
                if shown <= made_by(i, j) as f32 - 1.0 {
                    f.text_px(c, "·", 13.0, pal::FAINT, Align2::CENTER_CENTER);
                    continue;
                }
                let this = now.contains(&(i, j));
                if this {
                    hud_rect(f, c - Vec2::new(cell * 0.46, row * 0.46), c + Vec2::new(cell * 0.46, row * 0.46), LEFT.a(0.22));
                }
                // (a number written with a power of ten is set smaller, so as to keep
                // within its cell)
                let text = format!("${}$", entry_text(gs.r[i][j], scale));
                let wide = if text.contains('e') { measure(f, &text, 13.0) / (cell - 6.0) } else { 0.0 };
                let size = if wide > 1.0 { 13.0 / wide.min(1.5) } else { 13.0 };
                f.text_px(c, text, size, if this { pal::WHITE } else { pal::TEXT }, Align2::CENTER_CENTER);
            }
        }

        // what this step's numbers are, and the volume
        let mut y = top + row * 4.0 + 10.0;
        for line in detail(step, gs, scale) {
            f.text_px(Vec2::new(x, y), line, PANEL_TEXT, pal::TEXT, Align2::LEFT_TOP);
            y += 16.0;
        }
        y += 3.0;
        if step >= 7 {
            f.text_px(Vec2::new(x, y), volume_line(gs, scale, max.x - x - 10.0), PANEL_TEXT, pal::DIM, Align2::LEFT_TOP);
        } else {
            f.text_px(Vec2::new(x, y), ROWS_AND_COLUMNS, PANEL_TEXT, pal::DIM, Align2::LEFT_TOP);
        }
    }
}

impl Scene for Gram {
    fn view(&self) -> View {
        self.the_view()
    }

    fn take_view(&mut self) -> Option<View> {
        std::mem::take(&mut self.new_view).then(|| self.the_view())
    }

    fn animated(&self) -> bool {
        self.moving()
    }

    fn settings(&self, s: &mut ViewSettings) {
        s.grid_alpha = 0.07;
    }

    fn ui(&mut self, ui: &mut egui::Ui, _: &Gfx) {
        ui::note(ui, "Gram–Schmidt turns three independent vectors into three perpendicular vectors of length 1 (an orthonormal frame) that span the same space, one at a time. Keep the first one's direction. From each next one, take away its shadows on the ones already made: what is left is at right angles to all of them. Then scale what is left to length 1. Step through it with the buttons over the picture, and drag the tips of $a_1$, $a_2$, $a_3$.");

        ui::section(ui, "Things to try");
        for (k, trial) in TRIALS.iter().enumerate() {
            if ui::experiment(ui, trial.0, trial.1) {
                self.set_up(k);
            }
        }

        ui::section(ui, "Steps");
        ui.horizontal_wrapped(|ui| {
            if ui.button("◀ back").clicked() {
                self.back();
            }
            if ui.button("next ▶").clicked() {
                self.next();
            }
            if ui.button("↺ replay all").clicked() {
                self.replay();
            }
            if ui.button("the end").clicked() {
                (self.stage, self.goal, self.through, self.hold) = (STEPS as f32, STEPS as f32, false, 0.0);
            }
        });
        if ui::slider(ui, "step", &mut self.stage, 0.0..=STEPS as f32) {
            (self.goal, self.through, self.hold) = (self.stage, false, 0.0);
        }
        ui::note(ui, "Next plays one step; replay plays all eight, pausing after each. The same buttons are on the picture, with $R$.");
        if ui.checkbox(&mut self.show_box, "the slanted box the arrows span").changed() {
            self.new_view = true;
        }
        ui.checkbox(&mut self.show_other, ui::label_text(ui, "the frame made in the order $a_3, a_2, a_1$", 12.5, None, false));

        ui::section(ui, "Reading the picture");
        ui::key(
            ui,
            &[
                (pale(0), "pale arrows $a_1, a_2, a_3$: the vectors given; drag their tips"),
                (Q[0], r"bright arrows $\textcolor{pink}{q_1}$ (pink), $\textcolor{green}{q_2}$ (green), $\textcolor{purple}{q_3}$ (purple): the perpendicular vectors of length 1 made from them"),
                (Q[1], "dashed, in a $q$'s colour: a shadow on that $q$, some number times it"),
                (LEFT, "yellow: what is left of a vector once its shadows are taken away"),
                (pal::WHITE, "white corners: right angles"),
                (pal::BLUE, "blue sheet: the plane of $q_1$ and $q_2$, which $a_3$'s shadows lie in"),
                (BOX, "grey box (when shown): the slanted box spanned by $a_1, a_2, a_3$"),
                (pal::DIM, "grey arcs: quarter circles of radius 1 from one $q$'s tip to another's"),
                (OTHER, "cyan arrows $q_1', q_2', q_3'$ (when shown): the frame made in the order $a_3, a_2, a_1$"),
            ],
        );
        ui::section(ui, "Words");
        ui::term(ui, "Shadow:", r"the shadow of $a$ on a vector $q$ of length 1 is $(q \cdot a)\, q$: the arrow from $0$ to the point of $q$'s line nearest to $a$'s tip. What is left, $a - (q \cdot a)\, q$, is at right angles to $q$.");
        ui::term(ui, "Independent:", "no one of the vectors can be made from the others. Three in one plane are not independent.");
        ui::term(ui, "Orthonormal:", "at right angles to each other, and each of length 1.");
        ui::term(ui, "QR factorisation:", "$A = QR$, where $A$ has the $a$'s as its columns, $Q$ has the $q$'s, and $R$, upper triangular, holds the numbers that rebuild each $a$ from the $q$'s.");
        ui::term(ui, "Upper triangular:", "zero below the diagonal. $R$ is, because $a_1$ is made of $q_1$ alone, $a_2$ of $q_1$ and $q_2$, and only $a_3$ needs all three.");
        ui::term(ui, "Rank:", "how many independent directions the arrows span: how many $q$'s can be made.");
        ui::term(ui, "Determinant:", r"$\det A$, the volume of the box the columns span, counted negative if the box is a mirror image.");

        let gs = self.solve();
        let scale = self.scale();
        ui::section(ui, "Numbers");
        for i in 0..3 {
            let text = match gs.q[i] {
                Some(q) => format!("$({}, {}, {})$", fmt(q.x as f32), fmt(q.y as f32), fmt(q.z as f32)),
                None => "none: nothing was left".into(),
            };
            ui::readout_colored(ui, NAMES_Q[i], text, Q[i]);
        }
        let mut worst: f64 = 0.0;
        for i in 0..3 {
            for k in i + 1..3 {
                if let (Some(a), Some(b)) = (gs.q[i], gs.q[k]) {
                    worst = worst.max(a.dot(b).abs());
                }
            }
        }
        ui::readout(ui, r"largest $|q_i \cdot q_j|$, $i \ne j$", format!("${}$", signed(worst as f32, 6)));
        ui::readout(ui, "rank", format!("${}$", gs.rank()));
        let a = self.a.map(|v| v.as_dvec3());
        ui::readout(ui, r"$\lvert \det A \rvert$ (the box's volume)", format!("${}$", entry_text(det(a).abs(), scale * scale * scale)));
        ui::readout(ui, r"$r_{11} \times r_{22} \times r_{33}$", format!("${}$", entry_text(gs.volume(), scale * scale * scale)));
        if self.show_other {
            let other = Gs::of([a[2], a[1], a[0]]);
            let d: Vec<String> = (0..3).map(|i| entry_text(other.r[i][i], scale)).collect();
            ui::readout(ui, "lengths left, order $a_3, a_2, a_1$", format!("${}$", d.join(", ")));
            ui::readout(ui, "their product", format!("${}$", entry_text(other.volume(), scale * scale * scale)));
        }
        ui::note(ui, "What is left of a vector counts as nothing when it is shorter than a millionth of the vector's own length. The arrows' coordinates are kept to about seven digits: an arrow put in the plane of two others can end up a ten-millionth of its length off it by rounding alone, and should still count as in it.");
        ui::see_also(ui, &[("lsq", "projection: taking away a shadow"), ("det", "the volume of a box: the determinant"), ("vectors", "dot products and shadows"), ("svd", "another way to take a matrix apart")]);
    }

    fn frame(&mut self, f: &mut Frame) {
        self.advance(f.dt);
        // the panel's place follows the canvas, and the picture's place follows the panel's
        let view = f.view_size();
        let lay = Layout::of(view, f.top_inset);
        let before = Layout::of(self.canvas, self.inset);
        // (a canvas squeezed to a sliver is one the controls have opened over, on a
        // phone: the picture is not seen then, and is left as it was for when they close)
        let covered = view.x < 160.0;
        let changed = !covered && (lay.narrow != before.narrow || lay.panel.is_some() != before.panel.is_some() || (lay.narrow && (view - self.canvas).abs().max_element() > 24.0));
        // (the first canvas is a guess until the first frame: a different one gets the
        // view made for it)
        let first = !covered && !self.measured && (view - self.canvas).abs().max_element() > 24.0;
        self.new_view |= changed || first;
        if !covered && (changed || !self.measured) {
            self.canvas = view;
            self.measured = true;
        }
        self.inset = f.top_inset;
        let pointer = f.ptr.pos.map(|p| p / f.px);
        if pointer.is_some_and(|p| lay.covers(p)) {
            // the panel is for its buttons: no turning the view from there, and two
            // quick clicks are two steps, not a reset of the view
            f.handles.capture();
            f.handles.claim_clicks();
        }

        // the handles first, so that everything below is drawn where they now are
        // (one under the panel is left alone, unless it is being dragged already)
        for i in 0..3 {
            let hidden = on_screen(f, self.a[i]).is_some_and(|p| lay.covers(p));
            if !hidden || f.handles.dragging() {
                let mut tip = self.a[i];
                if f.drag3(1 + i as u32, &mut tip, None, pale(i)) {
                    self.a[i] = tip;
                }
            }
        }

        let gs = self.solve();
        self.picture(f, &gs, &lay);
        self.panel(f, &gs, &lay, pointer);
        // what has no name of its own in the picture, named along its bottom
        let bottom = if lay.narrow { lay.below - 10.0 } else { view.y - 18.0 };
        caption(f, &self.caption_parts(&gs), bottom);
    }
}

/// A caption along the bottom of the picture, on a dark plate, like `bottom_caption`
/// in the toolkit, but ending at `bottom` (above the panel, on a phone): `parts` side
/// by side as long as they fit across, on further lines above when they do not.
fn caption(f: &mut Frame, parts: &[&str], bottom: f32) {
    let (view, size) = (f.view_size(), 12.0);
    if view.y < 300.0 {
        return;
    }
    let room = view.x - 36.0;
    let gap = "      ";
    let width = |text: &str| measure(f, text, size);
    let mut lines: Vec<String> = Vec::new();
    for part in parts {
        match lines.last_mut() {
            Some(last) if width(last) + width(gap) + width(part) <= room => {
                last.push_str(gap);
                last.push_str(part);
            }
            _ => lines.push(part.to_string()),
        }
    }
    let count = lines.len();
    for (k, line) in lines.into_iter().enumerate() {
        let up = (count - 1 - k) as f32 * (size + 7.0);
        f.caption_px(Vec2::new(18.0, bottom - up), line, size, pal::DIM, Align2::LEFT_BOTTOM);
    }
}

impl Gram {
    /// The names the caption gives, for what the step on screen shows unnamed.
    fn caption_parts(&self, gs: &Gs) -> Vec<&'static str> {
        let step = self.step();
        let mut parts = Vec::new();
        match step {
            0 | 1 => parts.push("pale arrows: the vectors given (drag their tips)"),
            2..=4 if gs.q[0].is_some() => parts.push("dashed pink line: the line of $q_1$"),
            5..=7 if gs.q[0].is_some() && gs.q[1].is_some() => parts.push("blue sheet: the plane of $q_1$ and $q_2$"),
            8 => parts.push("dashed: each $a$ as a walk along the $q$'s (its column of $R$)"),
            _ => {}
        }
        if step >= 2 {
            parts.push("white corners: right angles");
        }
        if step >= 4 && gs.rank() >= 2 {
            parts.push("grey arcs: radius 1");
        }
        if self.show_box {
            parts.push("grey box: the one $a_1, a_2, a_3$ span");
        }
        if self.show_other {
            parts.push("cyan: the frame made in the order $a_3, a_2, a_1$");
        }
        parts
    }
}

impl Gram {
    /// Everything in space: the arrows, the step being played, and the finished frame.
    fn picture(&self, f: &mut Frame, gs: &Gs, lay: &Layout) {
        let s = self.stage;
        let prog = |k: usize| (s - (k as f32 - 1.0)).clamp(0.0, 1.0);
        let ease = |k: usize| smoothstep(prog(k));
        let a = self.a;
        let q: [Option<Vec3>; 3] = gs.q.map(|q| q.map(|v| v.as_vec3()));
        let left = gs.left.map(|v| v.as_vec3());
        let foot = gs.foot.map(|v| v.as_vec3());
        let r = gs.r;
        let mut names = Names { busy: Vec::new(), lay, view: f.view_size(), top: f.top_inset };
        let mut later: Vec<(Vec3, String, f32, Color)> = Vec::new();

        if self.show_box {
            slanted_box(f, a[0], a[1], a[2]);
        }

        // the given arrows
        for i in 0..3 {
            f.arrow(Vec3::ZERO, a[i], 2.2, pale(i));
            names.keep_clear(f, Vec3::ZERO, a[i]);
        }

        // the construction of each q in turn: j = 0 is a₁ scaled, j = 1 and 2 drop
        // their shadows, take them away and scale what is left
        for j in 0..3usize {
            // (the steps that drop its shadows, take them away and scale what is left;
            // a₁ has only the last)
            let (shadow, take, scale) = ((3 * j).saturating_sub(1), 3 * j, 3 * j + 1);
            let started = prog(if j == 0 { scale } else { shadow }) > 0.0;
            // (it fades as the next vector's construction begins, or the last step)
            let fade = 1.0 - smoothstep(prog(3 * j + 2) / 0.3);
            if !started || fade <= 0.0 {
                continue;
            }
            // and dims a little once its q is made
            let dim = fade * (1.0 - 0.45 * ease(scale));
            let tip = a[j];
            let before: Vec<(usize, Vec3)> = (0..j).filter_map(|i| q[i].map(|qi| (i, qi))).collect();
            if j > 0 {
                let ps = ease(shadow);
                let show = (prog(shadow) / 0.25).min(1.0) * fade;
                // what the shadows lie in: the line of q₁, or the plane of q₁ and q₂
                match before.as_slice() {
                    [(i, qi)] => f.dashed(*qi * -3.0, *qi * 3.0, 1.2, Q[*i].a(0.45 * show)),
                    [(_, q1), (_, q2)] => faded_sheet(f, *q1, *q2, 2.6, 4, pal::BLUE, 0.8 * show),
                    _ => {}
                }
                // the shadows, laid end to end from 0, growing as the tip drops
                let mut at = Vec3::ZERO;
                for &(i, qi) in &before {
                    let leg = qi * (r[i][j] as f32 * ps);
                    shadow_leg(f, at, at + leg, Q[i], dim);
                    names.keep_clear(f, at, at + leg);
                    at += leg;
                }
                // the tip dropping onto the line or plane
                f.dashed(tip, tip + (foot[j] - tip) * ps, 1.3, pal::WHITE.a(0.6 * dim));
                names.keep_clear(f, tip, foot[j]);
                let marked = smoothstep((ps - 0.6) / 0.4) * dim;
                let along = if foot[j].length() > 1e-3 { -foot[j] } else { before.first().map_or(Vec3::X, |b| b.1) };
                if marked > 0.0 && left[j].length() > 1e-4 {
                    corner(f, foot[j], tip - foot[j], along, 0.16, marked);
                }
                if ps >= 1.0 && prog(take) <= 0.0 && dim > 0.5 {
                    let shadows = match j {
                        1 => "$r_{12} q_1$",
                        _ => "$r_{13} q_1 + r_{23} q_2$",
                    };
                    later.push((foot[j] * 0.6, shadows.to_string(), 13.0, pal::TEXT.a(dim)));
                }
            }

            // what is left: cut out between the foot and the tip, slid down to 0,
            // then scaled to length 1
            let u = if j == 0 { 1.0 } else { prog(take) };
            if u <= 0.0 {
                continue;
            }
            let nothing = q[j].is_none();
            let appear = (u / 0.35).min(1.0);
            let slide = smoothstep((u - 0.35) / 0.65);
            if j > 0 {
                // where it was cut from, and the shadows taken away from the tip
                f.line(foot[j], tip, 1.6, LEFT.a(0.55 * appear * dim));
                let mut at = tip;
                for &(i, qi) in &before {
                    let leg = -qi * (r[i][j] as f32 * slide);
                    f.dashed(at, at + leg, 1.6, Q[i].a(0.7 * dim));
                    names.keep_clear(f, at, at + leg);
                    at += leg;
                }
                // and, once it is being scaled, how long it was
                if u >= 1.0 && q[j].is_some() && prog(scale) > 0.0 {
                    f.line(Vec3::ZERO, left[j], 1.6, LEFT.a(0.55 * dim));
                }
            }
            if nothing {
                f.dot(tip, 9.0, LEFT.a(fade));
                if fade > 0.5 {
                    later.push((tip, "nothing is left".into(), 13.0, LEFT));
                }
                continue;
            }
            let pc = ease(scale);
            let Some(qj) = q[j] else { continue };
            if pc >= 1.0 {
                continue;
            }
            let base = foot[j] * (1.0 - slide);
            // (from what is left to q itself: no dividing by a length that may be small)
            let arrow = left[j] * (1.0 - pc) + qj * pc;
            let color = LEFT.mix(Q[j], pc).a(if j == 0 { 1.0 } else { appear });
            f.arrow(base, base + arrow, 3.0, color);
            names.keep_clear(f, base, base + arrow);
            // at right angles to the q's before it, wherever it is on its way down
            for &(_, qi) in &before {
                corner(f, base, left[j], qi, 0.16, appear);
            }
            if appear >= 1.0 && pc < 0.5 {
                later.push((base + arrow * 0.5, "what is left".into(), 13.0, LEFT));
            }
        }

        // the q's made so far, and the right angles between them
        let made: Vec<usize> = (0..3).filter(|&i| prog(3 * i + 1) >= 1.0).collect();
        for &i in &made {
            if let Some(qi) = q[i] {
                f.arrow(Vec3::ZERO, qi, 3.6, Q[i]);
                names.keep_clear(f, Vec3::ZERO, qi);
            }
        }
        // (with a quarter of a circle of radius 1 from each tip to the next: they
        // are all of length 1)
        for (n, &i) in made.iter().enumerate() {
            for &k in &made[n + 1..] {
                if let (Some(qi), Some(qk)) = (q[i], q[k]) {
                    corner(f, Vec3::ZERO, qi, qk, 0.14, 0.9);
                    f.path((0..=24).map(|t| { let t = t as f32 / 24.0 * PI * 0.5; qi * t.cos() + qk * t.sin() }), 1.2, pal::DIM.a(0.5));
                }
            }
        }
        // where a q could not be made, say so at the origin
        if let Some(&i) = made.iter().find(|&&i| q[i].is_none()) {
            later.push((Vec3::ZERO, format!("no {}", NAMES_Q[i]), 13.0, LEFT));
        }

        // put back together: each a reached by walking along the q's by the numbers
        // in its column of R
        let walk = smoothstep((prog(STEPS) - 0.3) / 0.7);
        if walk > 0.0 {
            for j in 1..3 {
                let mut at = Vec3::ZERO;
                for i in 0..=j {
                    if let Some(qi) = q[i] {
                        let leg = qi * r[i][j] as f32;
                        shadow_leg(f, at, at + leg, Q[i], 0.75 * walk);
                        names.keep_clear(f, at, at + leg);
                        if i < j {
                            f.dot(at + leg, 5.0, Q[i].a(walk));
                        }
                        if self.show_box && i == j && walk > 0.5 {
                            later.push((at + leg * 0.5, format!("height $r_{{{}{}}}$", j + 1, j + 1), 12.5, Q[j]));
                        }
                        at += leg;
                    }
                }
            }
            if self.show_box && walk > 0.5 {
                later.push((a[0] * 0.5, "length $r_{11}$".into(), 12.5, Q[0]));
            }
        }

        // the frame made in the other order, for comparison
        if self.show_other {
            let other = Gs::of([a[2], a[1], a[0]].map(|v| v.as_dvec3()));
            for i in 0..3 {
                if let Some(p) = other.q[i] {
                    let p = p.as_vec3();
                    f.arrow(Vec3::ZERO, p, 2.0, OTHER.a(0.75));
                    later.push((p, format!("$q_{}'$", i + 1), 14.0, OTHER));
                }
            }
        }

        // the names: the arrows' first, so that they get the better places
        for i in 0..3 {
            names.name(f, a[i], NAMES_A[i], 15.0, pale(i).mix(pal::WHITE, 0.3));
        }
        for &i in &made {
            if let Some(qi) = q[i] {
                names.name(f, qi, NAMES_Q[i], 16.0, Q[i]);
            }
        }
        for (at, text, size, color) in later {
            names.name(f, at, &text, size, color);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn arrows(a: [[f32; 3]; 3]) -> [DVec3; 3] {
        a.map(|v| Vec3::from(v).as_dvec3())
    }

    /// An experiment as its button sets it up, played to rest at 60 frames a second.
    fn run(k: usize) -> Gram {
        let mut scene = Gram::create();
        scene.set_up(k);
        for _ in 0..60 * 60 {
            scene.advance(1.0 / 60.0);
        }
        assert!(!scene.moving());
        scene
    }

    /// The angle in degrees between two directions.
    fn degrees(a: DVec3, b: DVec3) -> f64 {
        a.angle_between(b).to_degrees()
    }

    /// The most that moving a₂ by `nudge` in any direction turns q₂ (degrees). The
    /// worst direction is found by trying many, and the one the geometry says is worst:
    /// at right angles to q₁, and tilted back from what is left so as to just graze
    /// the sphere of nudges round its tip.
    fn worst_swing(a: [DVec3; 3], nudge: f64) -> f64 {
        let gs = Gs::of(a);
        let (Some(q1), Some(q2)) = (gs.q[0], gs.q[1]) else { return f64::NAN };
        let n = q1.cross(q2);
        let tilt = (nudge / gs.r[1][1]).min(1.0);
        let mut ways: Vec<DVec3> = vec![n * (1.0 - tilt * tilt).sqrt() - q2 * tilt];
        let mut rng = Rng::new(9);
        ways.extend((0..4000).map(|_| DVec3::new(rng.normal() as f64, rng.normal() as f64, rng.normal() as f64).normalize()));
        ways.iter().map(|d| Gs::of([a[0], a[1] + *d * nudge, a[2]]).q[1].map_or(0.0, |q| degrees(q, q2))).fold(0.0, f64::max)
    }

    /// The arithmetic, wherever the arrows are (on top of each other, at the origin,
    /// in one plane, very small or large among them): the q's made are of length 1
    /// and at right angles; R is upper triangular, with rᵢⱼ = qᵢ · aⱼ above the
    /// diagonal and the lengths left on it; each aⱼ is rebuilt by column j of R; and
    /// r₁₁ r₂₂ r₃₃ = |det A|.
    #[test]
    fn gram_schmidt_makes_a_frame_and_r_rebuilds_the_arrows() {
        let mut rng = Rng::new(17);
        let mut point = || DVec3::new(rng.normal() as f64, rng.normal() as f64, rng.normal() as f64) * 1.5;
        let mut cases: Vec<[DVec3; 3]> = (0..500).map(|_| [point(), point(), point()]).collect();
        let [a, b, c] = arrows(START);
        cases.extend([
            [DVec3::ZERO, b, c],
            [a, DVec3::ZERO, c],
            [a, b, DVec3::ZERO],
            [DVec3::ZERO; 3],
            [a, a, a],
            [a, a * -2.0, c],
            [a, b, a * 0.3 - b * 1.7],
            [a, b, a + b * 1e-9],
            [a * 1e-20, b * 1e-20, c * 1e-20],
            [a * 1e6, b, c * 1e-6],
        ]);
        cases.extend(TRIALS.iter().map(|t| arrows(t.2)));
        let mut ranks = [0; 4];
        for a in cases {
            let gs = Gs::of(a);
            let scale = a.iter().map(|v| v.length()).fold(1e-300, f64::max);
            for i in 0..3 {
                if let Some(qi) = gs.q[i] {
                    assert!((qi.length() - 1.0).abs() < 1e-12, "{a:?}");
                    for k in i + 1..3 {
                        if let Some(qk) = gs.q[k] {
                            assert!(qi.dot(qk).abs() < 1e-12, "{a:?}: q{i} . q{k} = {}", qi.dot(qk));
                        }
                    }
                }
                for j in 0..3 {
                    let r = gs.r[i][j];
                    assert!(r.is_finite());
                    if i > j {
                        assert_eq!(r, 0.0);
                    } else if i < j {
                        let want = gs.q[i].map_or(0.0, |qi| qi.dot(a[j]));
                        assert!((r - want).abs() <= 1e-12 * scale, "{a:?}: r{i}{j} = {r}, not q{i} . a{j} = {want}");
                    } else {
                        assert!(r >= 0.0 && (r > 0.0) == gs.q[i].is_some(), "{a:?}");
                    }
                }
            }
            for j in 0..3 {
                let rebuilt: DVec3 = (0..3).filter_map(|i| gs.q[i].map(|qi| qi * gs.r[i][j])).sum();
                // (where nothing was left, what was dropped is under the threshold)
                assert!((rebuilt - a[j]).length() <= NOTHING * a[j].length() + 1e-12 * scale, "{a:?}: a{j} is not rebuilt");
                assert!(gs.left[j].is_finite() && gs.foot[j].is_finite());
            }
            let volume = det(a).abs();
            if gs.rank() == 3 {
                assert!((gs.volume() - volume).abs() <= 1e-12 * scale.powi(3), "{a:?}: {} against {volume}", gs.volume());
            } else {
                let lengths: f64 = a.iter().map(|v| v.length()).product();
                assert!(gs.volume() == 0.0 && volume <= 2.0 * NOTHING * lengths + 1e-300, "{a:?}");
            }
            ranks[gs.rank()] += 1;
        }
        assert!(ranks == [1, 1, 7, 506], "{ranks:?}");
    }

    /// The captions of the "things to try", held to account.
    #[test]
    fn the_experiments_show_what_their_captions_say() {
        // already perpendicular: no shadows, R is diagonal (and reads 0 above it),
        // each q is its a scaled, and the diagonal holds the lengths
        let s = run(0);
        let (a, gs) = (s.a.map(|v| v.as_dvec3()), s.solve());
        assert!(a[0].dot(a[1]) == 0.0 && a[0].dot(a[2]) == 0.0 && a[1].dot(a[2]) == 0.0 && s.stage == 8.0);
        for i in 0..3 {
            for j in i + 1..3 {
                assert!(gs.r[i][j].abs() < 1e-15 && entry_text(gs.r[i][j], s.scale()) == "0", "r{i}{j} = {}", gs.r[i][j]);
            }
            assert!(gs.q[i].is_some_and(|q| (q - a[i] / a[i].length()).length() < 1e-15) && (gs.r[i][i] - a[i].length()).abs() < 1e-15);
        }

        // nearly in line: 5 degrees apart, a2 of length 1.8, r22 = 0.157, scaled up
        // 6.4 times; a nudge of 0.05 turns q2 by up to 19 degrees, against 2 at the start
        let s = run(1);
        let (a, gs) = (s.a.map(|v| v.as_dvec3()), s.solve());
        assert!((degrees(a[0], a[1]) - 5.0).abs() < 0.05 && (a[1].length() - 1.8).abs() < 0.001, "{} degrees, {}", degrees(a[0], a[1]), a[1].length());
        assert_eq!((entry_text(gs.r[1][1], s.scale()), format!("{:.1}", 1.0 / gs.r[1][1])), ("0.157".to_string(), "6.4".to_string()));
        assert!(s.stage == 4.0);
        let (near, start) = (worst_swing(a, 0.05), worst_swing(arrows(START), 0.05));
        assert!((18.5..19.0).contains(&near) && (2.0..2.2).contains(&start), "{near} and {start} degrees");

        // three in one plane: a3 = a2 - a1/2 exactly, in the plane; its two shadows
        // make up all of it, nothing is left, no q3, rank 2, det A = 0
        let s = run(2);
        let (a, gs) = (s.a.map(|v| v.as_dvec3()), s.solve());
        assert!(s.a[2] == s.a[1] - s.a[0] * 0.5 && a[0].cross(a[1]).dot(a[2]) == 0.0);
        let shadows = gs.q[0].unwrap_or(DVec3::NAN) * gs.r[0][2] + gs.q[1].unwrap_or(DVec3::NAN) * gs.r[1][2];
        assert!((shadows - a[2]).length() < 1e-15 && gs.q[2].is_none() && gs.rank() == 2 && entry_text(gs.r[2][2], s.scale()) == "0");
        assert!(entry_text(det(a), s.scale().powi(3)) == "0" && s.stage == 7.0);
        assert_eq!(detail(7, &gs, s.scale())[1], "Only two perpendicular directions: rank 2.");

        // another order: q1' lies along a3; a different frame, at right angles all the
        // same; the lengths left differ, their product does not
        let s = run(3);
        let (a, gs) = (s.a.map(|v| v.as_dvec3()), s.solve());
        let other = Gs::of([a[2], a[1], a[0]]);
        assert!(s.show_other && other.rank() == 3 && other.q[0].is_some_and(|q| q.cross(a[2]).length() < 1e-15));
        for i in 0..3 {
            let (p, q) = (other.q[i].unwrap_or(DVec3::NAN), gs.q[i].unwrap_or(DVec3::NAN));
            assert!(degrees(p, q) > 40.0 && (other.r[i][i] - gs.r[i][i]).abs() > 0.1, "{i}: {} degrees apart", degrees(p, q));
            for k in i + 1..3 {
                assert!(p.dot(other.q[k].unwrap_or(DVec3::NAN)).abs() < 1e-15);
            }
        }
        assert!((other.volume() - gs.volume()).abs() < 1e-14 && entry_text(other.volume(), 8.0) == entry_text(gs.volume(), 8.0));

        // the box's volume: |det A| = r11 r22 r33, the length of a1, the height of a2
        // above the line of a1, and the height of a3 above the floor of the box
        let s = run(4);
        let (a, gs) = (s.a.map(|v| v.as_dvec3()), s.solve());
        let floor = a[0].cross(a[1]).length();
        assert!(s.show_box && !s.show_other && s.stage == 8.0);
        assert!((gs.r[0][0] - a[0].length()).abs() < 1e-15 && (gs.r[1][1] - floor / a[0].length()).abs() < 1e-14);
        assert!((gs.r[2][2] - det(a).abs() / floor).abs() < 1e-14 && (gs.volume() - det(a).abs()).abs() < 1e-14);
        // and the walk's last legs are those heights, at right angles to what is under them
        assert!(gs.left[1].dot(a[0]).abs() < 1e-14 && gs.left[2].dot(a[0]).abs() < 1e-14 && gs.left[2].dot(a[1]).abs() < 1e-14);
    }

    /// The steps: next, back and replay go where they say, at any frame rate, with a
    /// pause after each step only when playing through; and the scene says it is
    /// still exactly when nothing moves.
    #[test]
    fn the_steps_play_and_come_to_rest() {
        let mut s = Gram::create();
        assert!(s.stage == 8.0 && !s.moving() && s.step() == 8);
        // nothing more at either end
        s.next();
        assert!(!s.moving());
        // a replay, at several frame rates: the whole takes the same time, and between
        // two steps it waits on the step just done
        let whole = HOLD * 0.6 + STEPS as f32 * STEP_TIME + (STEPS - 1) as f32 * HOLD;
        for hz in [30.0f32, 60.0, 144.0, 240.0] {
            let mut s = Gram::create();
            s.replay();
            let (mut t, dt) = (0.0, 1.0 / hz);
            let mut seen = Vec::new();
            while s.moving() && t < 60.0 {
                s.advance(dt);
                t += dt;
                if (t - (HOLD * 0.6 + 2.0 * STEP_TIME + HOLD * 1.5)).abs() < 0.5 * dt {
                    seen.push(s.stage);
                }
            }
            assert!((t - whole).abs() <= dt * 1.01 && s.stage == 8.0 && !s.through, "{hz} Hz: {t} s, not {whole}");
            assert_eq!(seen, [2.0], "{hz} Hz: not waiting on step 2");
        }
        // next, while the steps are being played through: it stops after the one playing
        let mut s = Gram::create();
        s.replay();
        s.advance(HOLD * 0.6 + STEP_TIME + HOLD + STEP_TIME * 0.5);
        assert!((s.stage - 1.5).abs() < 1e-4 && s.through && s.can_go_on());
        s.next();
        for _ in 0..600 {
            s.advance(1.0 / 60.0);
        }
        assert!(s.stage == 2.0 && !s.moving());
        // one step on, and one back; a second press queues a second step
        let mut s = Gram::create();
        (s.stage, s.goal) = (3.0, 3.0);
        s.next();
        s.advance(STEP_TIME * 0.5);
        assert!(s.stage > 3.4 && s.stage < 3.6 && s.moving());
        s.next();
        s.advance(STEP_TIME * 1.5 + 0.01);
        assert!(s.stage == 5.0 && !s.moving());
        s.back();
        s.advance(BACK_TIME + 0.01);
        assert!(s.stage == 4.0 && !s.moving());
        // paused (no time passes), nothing moves
        s.next();
        s.advance(0.0);
        assert!(s.stage == 4.0 && s.moving());
        // the start is the floor
        (s.stage, s.goal) = (0.0, 0.0);
        s.back();
        assert!(!s.moving() && s.step() == 0);
    }

    /// The panel keeps to every canvas and clear of the application's buttons, and its
    /// lines fit across it whatever the numbers in them.
    #[test]
    fn the_panel_fits_every_canvas() {
        for (x, y) in [(1080.0f32, 1000.0f32), (580.0, 700.0), (390.0, 800.0), (360.0, 640.0), (844.0, 390.0), (1700.0, 800.0), (600.0, 400.0), (240.0, 180.0)] {
            let inset = if x < 900.0 { 46.0 } else { 0.0 };
            let lay = Layout::of(Vec2::new(x, y), inset);
            if let Some((min, max)) = lay.panel {
                assert!(min.x >= 0.0 && min.y >= inset && max.x <= x && max.y <= y && max.y - min.y >= PANEL_H - 1.0, "{x}x{y}: {lay:?}");
                assert!(!lay.narrow || lay.below > inset + 200.0, "{x}x{y}: too little room left for the picture");
            }
            assert!((x, y) != (1080.0, 1000.0) || lay.panel.is_some());
        }
        // the lines, for every step and many arrows, with long numbers among them
        let room = PANEL_W - 28.0;
        let mut sets: Vec<[[f32; 3]; 3]> = TRIALS.iter().map(|t| t.2).collect();
        sets.extend([
            START,
            [[0.0; 3], START[1], START[2]],
            [START[0], START[0], START[2]],
            [START[0], START[1], START[1]],
            [[-12.34, 45.67, -78.9], [-98.7, -65.4, 32.1], [-0.00123, 0.00456, -0.00789]],
            [[-1.234e-5, 4.567e-5, 0.0], [-9.876e-5, -6.543e-5, 3.21e-5], [-0.00123, -0.00456, 98.7]],
        ]);
        for a in sets {
            let gs = Gs::of(arrows(a));
            let scale = arrows(a).iter().map(|v| v.length()).fold(1e-300, f64::max);
            let mut lines: Vec<String> = (0..=STEPS).flat_map(|k| detail(k, &gs, scale)).collect();
            lines.push(volume_line(&gs, scale, room));
            lines.push(ROWS_AND_COLUMNS.into());
            for line in lines {
                assert!(text_width(&line, PANEL_TEXT) <= room && !line.contains("NaN") && !line.contains("inf"), "{a:?}: {line:?}");
            }
        }
        for title in TITLES {
            assert!(text_width(title, 14.0) <= room, "{title}");
        }
    }

    /// Degenerate arrows give finite numbers, views and words, and the words say what
    /// happened.
    #[test]
    fn degenerate_arrows_leave_everything_finite() {
        let [a, b, c] = START.map(Vec3::from);
        for (arrows, rank, says) in [
            ([Vec3::ZERO, b, c], 2, (1, "$a_1$ is $0$: there is nothing to scale,")),
            ([a, a * 2.0, c], 2, (4, "Nothing is left ($r_{22} = 0$), so there is no $q_2$.")),
            ([a, b, a - b], 2, (6, "Taking the shadows away leaves nothing:")),
            ([Vec3::ZERO; 3], 0, (5, "There are no $q$'s yet to cast a shadow on:")),
            ([a, a, a], 1, (7, "Nothing is left ($r_{33} = 0$), so there is no $q_3$.")),
        ] {
            let mut s = Gram::create();
            s.a = arrows;
            let gs = s.solve();
            assert_eq!(gs.rank(), rank, "{arrows:?}");
            assert_eq!(detail(says.0, &gs, s.scale())[0], says.1);
            let (middle, size) = s.framing();
            assert!(middle.is_finite() && size.is_finite());
            for show_box in [false, true] {
                s.show_box = show_box;
                for canvas in [Vec2::new(1080.0, 1000.0), Vec2::new(390.0, 800.0), Vec2::new(240.0, 180.0)] {
                    s.canvas = canvas;
                    let View::Orbit { target, dist, .. } = s.the_view() else { panic!("not a view in space") };
                    assert!(target.is_finite() && dist.is_finite() && dist > 0.0);
                }
            }
        }
    }

    /// The threshold, as the panel explains it: an arrow put in the plane of two others
    /// ends up, once its coordinates are rounded to the seven or so digits they are
    /// kept to, as much as a ten-millionth of its length off that plane, and is still
    /// found to leave nothing.
    #[test]
    fn rounding_leaves_less_than_the_threshold() {
        let mut rng = Rng::new(23);
        let mut worst: f64 = 0.0;
        for _ in 0..5000 {
            let mut point = || Vec3::new(rng.normal(), rng.normal(), rng.normal()) * 2.0;
            let (a1, a2) = (point(), point());
            let (c1, c2) = (rng.normal(), rng.normal());
            // the third arrow worked out in f32, as an experiment or a drag would leave it
            let a3 = a1 * c1 + a2 * c2;
            let a = [a1, a2, a3].map(|v| v.as_dvec3());
            let gs = Gs::of(a);
            // (where a3 is not a small difference of large parts, which multiplies the
            // rounding of the parts)
            let parts = (a1 * c1).length() + (a2 * c2).length();
            if gs.rank() >= 2 && a3.length() >= 0.25 * parts {
                worst = worst.max(gs.left[2].length() / a[2].length());
                assert!(gs.q[2].is_none(), "{a:?}: {} is left", gs.left[2].length() / a[2].length());
            }
        }
        assert!(worst > 5e-8 && worst < 2e-7, "{worst}");
    }

    /// The numbers behind the captions (run with --ignored --nocapture).
    #[test]
    #[ignore]
    fn explore() {
        for k in 0..TRIALS.len() {
            let s = run(k);
            let gs = s.solve();
            println!("{}: R = {:?}, rank {}, det {}, stage {}", TRIALS[k].0, gs.r, gs.rank(), det(s.a.map(|v| v.as_dvec3())), s.stage);
        }
        let a = arrows(TRIALS[1].2);
        println!("nearly in line: angle {}, |a2| {}, worst swing {}, at the start {}", degrees(a[0], a[1]), a[1].length(), worst_swing(a, 0.05), worst_swing(arrows(START), 0.05));
    }
}

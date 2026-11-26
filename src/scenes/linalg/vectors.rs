//! Dot and cross products.

use super::*;
use glam::DVec3;

pub const DEF: SceneDef = SceneDef {
    id: "vectors",
    title: "Dot and cross products",
    group: "Linear algebra: vectors and maps",
    blurb: "Two arrows: the shadow one casts on the other, and the area they span.",
    make: |_| Box::new(Vectors::create()),
};

/// What the picture shows: one product, the other, or both.
const SHOWN: [&str; 3] = ["dot product", "cross product", "both"];
const DOT: usize = 0;
const CROSS: usize = 1;
const BOTH: usize = 2;

/// The colours, each with one meaning.
const A_COL: Color = pal::GREEN;
const B_COL: Color = pal::ORANGE;
/// b's shadow on the line of a, and the dot product it gives.
const SHADOW: Color = pal::YELLOW;
/// The same shadow when it falls behind the start (the dot product is negative).
const BEHIND: Color = pal::PINK;
/// The parallelogram, and the cross product whose length is its area.
const AREA: Color = pal::PURPLE;

/// How far a tip can be taken from the origin.
const REACH: f32 = 4.0;
/// The tips move in steps of this much, unless they are kept at length 1.
const STEP: f32 = 0.1;
/// Seconds a swap takes.
const SWAP_TIME: f32 = 1.8;
/// How quickly an arrow is drawn gliding to where its tip has gone (the time in which
/// it covers about two thirds of the way, in seconds). A dragged tip moves in steps of
/// `STEP`, about twenty pixels at a time: drawn as it stands, the arrow would jump.
const GLIDE: f32 = 0.045;
/// An arrow drawn within this of its tip's place is there (a fiftieth of a pixel).
const ARRIVED: f32 = 1e-4;
/// A length below this is no length at all: the arrow has no direction.
const TINY: f64 = 1e-6;

/// The pair the scene opens with.
const START: ([f32; 3], [f32; 3]) = ([1.8, 0.8, -0.2], [0.4, 1.6, 0.6]);

/// Experiments: (button, what to look for, what is shown). The arrows each one sets
/// are `trial_pair`; the fourth then swaps them, and the fifth keeps them at length 1.
const TRIALS: [(&str, &str, usize); 5] = [
    (
        "at right angles",
        r"$b$ now stands at a right angle to $a$. Its shadow on the line of $a$ shrinks to a single point, the start, so $a \cdot b = 0$: in the sum, the three products cancel. The parallelogram is a rectangle, and no other angle gives arrows of these lengths a larger area: $a \times b$ is as long as it can be, $|a|\,|b| = 3.09$.",
        BOTH,
    ),
    (
        "past a right angle",
        r"$\theta$ is now 124.9°, more than a right angle. $b$'s shadow falls behind the start, where the line of $a$ carries on backwards (pink), so its length counts as negative, and so does the dot product: $a \cdot b = -2.04$, from the sum and from $|a|\,|b| \cos\theta$ alike.",
        DOT,
    ),
    (
        "in line",
        r"$b$ lies along $a$, half as long. Its shadow is $b$ itself, so $a \cdot b = |a|\,|b| = 1.96$. The two span no area: the parallelogram is flat, and $a \times b = 0$, an arrow with no length and so no direction.",
        BOTH,
    ),
    (
        "swap $a$ and $b$",
        r"$a$ and $b$ swing round until each is where the other was. Their lengths and the angle between them stay as they were, and so does $a \cdot b = 1.64$, all the way. But the turn from $a$ to $b$ now goes the other way round, and $a \times b$ ends up pointing the opposite way: $b \times a = -(a \times b)$.",
        BOTH,
    ),
    (
        "unit vectors",
        r"Both arrows now have length 1 (the faint circle) and keep it while you drag. Then $a \cdot b = |a|\,|b| \cos\theta$ is $\cos\theta$ itself, the length of the shadow: here $\theta = 60.0°$ and $a \cdot b = 0.50$. Drag $b$ round: $a \cdot b$ runs from 1 (the same way as $a$) through 0 (at a right angle) to −1 (opposite).",
        DOT,
    ),
];
const SWAP_TRIAL: usize = 3;
const UNIT_TRIAL: usize = 4;

/// The arrows experiment `k` starts from.
fn trial_pair(k: usize) -> (Vec3, Vec3) {
    let a = Vec3::from(START.0);
    match k {
        0 => (a, Vec3::new(-0.5, 1.3, 0.7)),
        1 => (a, Vec3::new(-1.4, 0.8, 0.8)),
        2 => (a, Vec3::new(0.9, 0.4, -0.1)),
        3 => (a, Vec3::new(0.2, 1.8, 0.8)),
        _ => {
            // a at length 1, and b at 60 degrees from it in the plane of the opening pair
            let u = a.normalize();
            let w = (Vec3::from(START.1) - u * u.dot(Vec3::from(START.1))).normalize();
            (u, u * 0.5 + w * (0.75f32).sqrt())
        }
    }
}

/// Everything the picture and the numbers are made of, worked out in double
/// precision from the two arrows.
#[derive(Clone, Copy, Debug)]
struct Pair {
    a: DVec3,
    b: DVec3,
    la: f64,
    lb: f64,
    dot: f64,
    /// a × b: zero when the two are in line or either has no length.
    cross: DVec3,
    /// |a × b|, the area of the parallelogram.
    area: f64,
    /// The angle between them, in radians, when both have a direction.
    theta: Option<f64>,
    /// Both have a direction, and they lie along one line.
    in_line: bool,
}

/// An arrow in double precision. A tip on the grid of tenths is taken at its exact
/// decimal value, so that a · b and a × b come out as the sums written on screen do.
fn exact(v: Vec3) -> DVec3 {
    let tenths = v.as_dvec3() * 10.0;
    if (tenths - tenths.round()).abs().max_element() < 1e-4 {
        tenths.round() / 10.0
    } else {
        v.as_dvec3()
    }
}

/// Is every coordinate a whole number of tenths? (Then one decimal says it exactly.)
fn on_grid(v: Vec3) -> bool {
    let tenths = v.as_dvec3() * 10.0;
    (tenths - tenths.round()).abs().max_element() < 1e-4
}

impl Pair {
    fn of(a: Vec3, b: Vec3) -> Pair {
        let (a, b) = (exact(a), exact(b));
        let (la, lb) = (a.length(), b.length());
        let both = la > TINY && lb > TINY;
        let scale = la * lb;
        let mut cross = a.cross(b);
        let mut dot = a.dot(b);
        // (what is left of a zero after rounding is a zero: at a right angle the
        // shadow is not to turn pink because of the sixteenth decimal)
        if dot.abs() <= 1e-12 * scale {
            dot = 0.0;
        }
        let in_line = both && cross.length() <= 1e-9 * scale;
        if !both || in_line {
            cross = DVec3::ZERO;
        }
        let area = cross.length();
        let theta = both.then(|| area.atan2(dot));
        Pair { a, b, la, lb, dot, cross, area, theta, in_line }
    }

    fn a_dir(&self) -> bool {
        self.la > TINY
    }

    fn b_dir(&self) -> bool {
        self.lb > TINY
    }

    /// The length of b's shadow on the line of a, counted negative when it falls
    /// behind the start: |b| cos θ. None when a has no direction (there is no line).
    fn shadow(&self) -> Option<f64> {
        self.a_dir().then(|| self.dot / self.la)
    }

    /// How far b's tip is from the line of a, |b| sin θ: the height of the
    /// parallelogram over a.
    fn height(&self) -> Option<f64> {
        self.a_dir().then(|| self.area / self.la)
    }

    /// Where b's shadow ends: the point of the line of a nearest to b's tip.
    fn foot(&self) -> DVec3 {
        if self.a_dir() {
            self.a * (self.dot / (self.la * self.la))
        } else {
            DVec3::ZERO
        }
    }

    fn degrees(&self) -> Option<f64> {
        self.theta.map(f64::to_degrees)
    }
}

/// A number as a factor in a written-out product: in brackets if it is negative.
fn factor(v: f64, decimals: usize) -> String {
    let s = num(v, decimals);
    if s.starts_with('−') {
        format!("({s})")
    } else {
        s
    }
}

/// An arrow's coordinates: one decimal on the grid of tenths, two off it.
fn coords(v: Vec3) -> String {
    let d = if on_grid(v) { 1 } else { 2 };
    let e = exact(v);
    format!("({}, {}, {})", num(e.x, d), num(e.y, d), num(e.z, d))
}

fn degrees(v: f64) -> String {
    format!("{}°", num(v, 1))
}

/// The pair part of the way through a swap (`s` from 0 to 1). Both arrows turn
/// together by half a turn about the line halfway between them, which carries the
/// direction of each onto the other's, while their lengths trade over: the angle
/// between them never changes on the way. (An arrow with no length has no direction
/// to turn: then the two simply slide to each other's places.)
fn swapping(a: Vec3, b: Vec3, s: f32) -> (Vec3, Vec3) {
    let (la, lb) = (a.length(), b.length());
    if la < 1e-6 || lb < 1e-6 {
        return (a.lerp(b, s), b.lerp(a, s));
    }
    let (ua, ub) = (a / la, b / lb);
    let mid = ua + ub;
    // (opposite arrows have no line halfway between them: any line across them will do)
    let axis = if mid.length() > 1e-3 { mid.normalize() } else { ua.any_orthonormal_vector() };
    let turn = glam::Quat::from_axis_angle(axis, PI * s);
    (turn * ua * (la + (lb - la) * s), turn * ub * (lb + (la - lb) * s))
}

/// One line of the sums written over the picture: what stands left of the column of
/// = signs (empty on a line that carries on from the one before) and its colour, and
/// the rest and its colour. An empty line parts the two products.
type SumLine = (String, Color, String, Color);

/// Which arrows have no length, as the start of a sentence.
fn lengthless(p: &Pair) -> &'static str {
    match (p.a_dir(), p.b_dir()) {
        (false, false) => "$a$ and $b$ have",
        (false, true) => "$a$ has",
        _ => "$b$ has",
    }
}

/// The two products worked out with the present numbers, both ways, line by line.
fn sums(a: Vec3, b: Vec3, shown: usize) -> Vec<SumLine> {
    let p = Pair::of(a, b);
    let d = if on_grid(a) && on_grid(b) { 1 } else { 2 };
    let mut lines: Vec<SumLine> = Vec::new();
    let mut line = |left: &str, color: Color, rest: String| lines.push((left.to_string(), color, rest, pal::TEXT));
    if shown != CROSS {
        let dot = num(p.dot, 2);
        line(r"$a \cdot b$", SHADOW, "$= a_1 b_1 + a_2 b_2 + a_3 b_3$".to_string());
        let terms: Vec<String> = (0..3).map(|i| format!(r"{} \times {}", factor(p.a[i], d), factor(p.b[i], d))).collect();
        line("", SHADOW, format!("$= {} = {dot}$", terms.join(" + ")));
        match (p.theta, p.shadow()) {
            (Some(theta), Some(shadow)) => {
                line("", SHADOW, format!(r"$= |a|\,|b| \cos\theta = {} \times {} \times {} = {dot}$", num(p.la, 2), num(p.lb, 2), factor(theta.cos(), 3)));
                line("", SHADOW, format!(r"$= |a| \times \text{{shadow}} = {} \times {} = {dot}$", num(p.la, 2), factor(shadow, 2)));
            }
            _ => line(
                "",
                SHADOW,
                match (p.a_dir(), p.b_dir()) {
                    (true, _) => r"$b$ has no length: no angle $\theta$, and its shadow is a point".to_string(),
                    (false, true) => r"$a$ has no length: no angle $\theta$, and no line for a shadow".to_string(),
                    _ => r"$a$ and $b$ have no length: no angle $\theta$, no shadow".to_string(),
                },
            ),
        }
    }
    if shown == BOTH {
        line("", pal::TEXT, String::new());
    }
    if shown != DOT {
        let area = num(p.area, 2);
        line(r"$a \times b$", AREA, "$= (a_2 b_3 - a_3 b_2, a_3 b_1 - a_1 b_3, a_1 b_2 - a_2 b_1)$".to_string());
        line("", AREA, format!("$= ({}, {}, {})$", num(p.cross.x, 2), num(p.cross.y, 2), num(p.cross.z, 2)));
        match (p.theta, p.height()) {
            (Some(theta), Some(height)) => {
                line(r"$|a \times b|$", AREA, format!(r"$= |a|\,|b| \sin\theta = {} \times {} \times {} = {area}$", num(p.la, 2), num(p.lb, 2), num(theta.sin(), 3)));
                let what = if p.in_line { "no area: they are in line" } else { "the area" };
                line("", AREA, format!(r"$= |a| \times \text{{height}} = {} \times {} = {area}$, {what}", num(p.la, 2), num(height, 2)));
            }
            _ => line(r"$|a \times b|$", AREA, format!("$= 0$: {} no length, so no area", lengthless(&p))),
        }
    }
    lines
}

/// How wide `text` is set, measured by a stand-in with every digit a 0: digits are all
/// as wide as one another, and the stand-in stays the same while a tip is dragged and
/// the numbers change (the frame keeps what it has measured, but a text it has not
/// seen is set afresh).
fn width_of(f: &Frame, text: &str, size: f32) -> f32 {
    let stand_in: String = text.chars().map(|c| if c.is_ascii_digit() { '0' } else { c }).collect();
    f.text_size(&stand_in, size).x
}

/// Is the canvas narrow (a phone held upright)? Then the sums go below the picture
/// instead of over its top left corner.
fn narrow_canvas(view: Vec2) -> bool {
    view.x < 620.0 && view.y >= 560.0
}

/// Line height and type size of the sums.
const SUM_SIZE: f32 = 13.0;
const SUM_LINE: f32 = 19.0;
/// The height kept for the sums: the most they can take (both products and the line
/// between), so that the picture need not move when the choice changes.
const SUMS_HEIGHT: f32 = 9.0 * SUM_LINE + 16.0;

/// How the canvas of size `view` (logical pixels), whose top `inset` pixels the
/// application's buttons cover, is shared: the box behind `lines` lines of sums whose
/// widest is `width` (None if the canvas cannot spare the room), and the part left
/// for the picture.
fn layout(view: Vec2, inset: f32, lines: usize, width: f32) -> (Option<(Vec2, Vec2)>, (Vec2, Vec2)) {
    // (a canvas squeezed to nothing still has a part for the picture)
    let whole = (Vec2::new(0.0, inset.min(view.y * 0.5)), view.max(Vec2::ONE));
    if view.x < 300.0 || SUMS_HEIGHT + inset > view.y * 0.45 {
        return (None, whole);
    }
    let height = lines as f32 * SUM_LINE + 16.0;
    if narrow_canvas(view) {
        // along the bottom, the picture above
        let top = view.y - 8.0 - SUMS_HEIGHT;
        let sums = (lines > 0).then(|| (Vec2::new(8.0, view.y - 8.0 - height), Vec2::new(view.x - 8.0, view.y - 8.0)));
        (sums, (Vec2::new(0.0, inset), Vec2::new(view.x, top)))
    } else {
        // in the top left corner, the picture below
        let top = inset + 14.0;
        let sums = (lines > 0).then(|| (Vec2::new(14.0, top), Vec2::new(14.0 + width.min(view.x - 28.0), top + height)));
        // (the sums take only the left part of the top: the picture may reach up
        // beside them)
        (sums, (Vec2::new(0.0, top + SUMS_HEIGHT * 0.5), view))
    }
}

pub struct Vectors {
    a: Vec3,
    b: Vec3,
    /// Where each handle is while it is dragged: the arrow's tip is that point taken
    /// to the nearest tenths (or to length 1), so the handle follows the pointer
    /// smoothly while the tip steps.
    grip: [Vec3; 2],
    /// Where the two arrows are drawn: gliding to a and b when those have stepped
    /// (the numbers are those of a and b themselves, from the first moment).
    drawn: [Vec3; 2],
    shown: usize,
    /// Both arrows kept at length 1.
    unit: bool,
    /// A swap under way: the pair it started from, and how far it has got (0 to 1).
    swap: Option<(Vec3, Vec3, f32)>,
    /// The points the view is framed on.
    home: Vec<Vec3>,
    /// The canvas last drawn on, and how much of its top the application's buttons
    /// cover: the picture's place follows them.
    canvas: Vec2,
    inset: f32,
    new_view: bool,
}

impl Vectors {
    /// (Nothing here needs the graphics device, so the tests can make one too.)
    fn create() -> Self {
        let (a, b) = (Vec3::from(START.0), Vec3::from(START.1));
        let mut s = Vectors { a, b, grip: [a, b], drawn: [a, b], shown: BOTH, unit: false, swap: None, home: Vec::new(), canvas: Vec2::new(1070.0, 1000.0), inset: 0.0, new_view: false };
        s.frame_on_pair();
        s.new_view = false;
        s
    }

    /// Set up one of the experiments in `TRIALS`.
    fn set_up(&mut self, k: usize) {
        (self.a, self.b) = trial_pair(k);
        self.shown = TRIALS[k].2;
        self.unit = k == UNIT_TRIAL;
        self.swap = None;
        self.grip = [self.a, self.b];
        self.drawn = [self.a, self.b];
        self.frame_on_pair();
        if k == SWAP_TRIAL {
            self.start_swap();
            // (a × b ends up pointing the other way: the view has room for both)
            self.home.push(-self.a.cross(self.b));
        }
    }

    /// Have the view show the present pair, with what is drawn for it, as well as it
    /// can (an experiment's picture is framed afresh).
    fn frame_on_pair(&mut self) {
        let p = Pair::of(self.a, self.b);
        let (a, b) = (self.a, self.b);
        let mut points = vec![Vec3::ZERO, a, b];
        if self.shown != CROSS {
            points.push(p.foot().as_vec3());
        }
        if self.shown != DOT {
            points.extend([a + b, p.cross.as_vec3()]);
        }
        if self.unit {
            // the circle of length 1 in their plane
            let u = a.normalize_or_zero();
            let w = (b - u * u.dot(b)).normalize_or_zero();
            points.extend([u, -u, w, -w]);
        }
        self.home = points;
        self.new_view = true;
    }

    fn start_swap(&mut self) {
        self.swap = Some((self.a, self.b, 0.0));
    }

    /// Move a swap on by `dt` seconds of scene time.
    fn advance(&mut self, dt: f32) {
        let Some((a, b, t)) = self.swap.as_mut() else { return };
        *t += dt / SWAP_TIME;
        if *t >= 1.0 {
            // (exactly each other's places at the end, not nearly)
            (self.a, self.b) = (*b, *a);
            self.swap = None;
        } else {
            (self.a, self.b) = swapping(*a, *b, smoothstep(*t));
        }
        self.grip = [self.a, self.b];
        // (a swap is a glide already: it is drawn as it stands)
        self.drawn = [self.a, self.b];
    }

    /// Are the arrows drawn where a and b are?
    fn arrived(&self) -> bool {
        self.drawn == [self.a, self.b]
    }

    /// Bring the arrows as drawn `dt` seconds nearer to a and b; with `at_once`, all
    /// the way.
    fn glide(&mut self, dt: f32, at_once: bool) {
        let k = 1.0 - (-dt.max(0.0) / GLIDE).exp();
        for (drawn, to) in self.drawn.iter_mut().zip([self.a, self.b]) {
            *drawn += (to - *drawn) * k;
            // (exactly there in the end, not nearly: the picture then stops changing)
            if at_once || !drawn.is_finite() || (to - *drawn).length() < ARRIVED {
                *drawn = to;
            }
        }
    }

    /// Where a dragged tip goes when its handle is at `p`: no further out than
    /// `REACH`, and then to the nearest tenths, or to length 1 if lengths are kept
    /// at 1 (`keep` if `p` is the origin, which has no direction).
    fn settle(&self, p: Vec3, keep: Vec3) -> Vec3 {
        let p = if p.length() > REACH { p * (REACH / p.length()) } else { p };
        if self.unit {
            p.try_normalize().unwrap_or(keep)
        } else {
            (p / STEP).round() * STEP
        }
    }

    /// Keep both at length 1 from now on (an arrow with no length is given one).
    fn make_unit(&mut self) {
        self.a = self.a.try_normalize().unwrap_or(Vec3::X);
        self.b = self.b.try_normalize().unwrap_or(if self.a.x.abs() < 0.9 { Vec3::X } else { Vec3::Y });
        self.grip = [self.a, self.b];
    }

    /// The view that fits the pair, with what is drawn for it (`home`), into the part
    /// of the canvas the sums leave free, and no larger than reads well.
    fn the_view(&self) -> View {
        let (yaw, pitch) = (-0.95f32, 0.42f32);
        // the camera's directions: across the screen, up it, and towards the eye
        let right = Vec3::new(-yaw.sin(), yaw.cos(), 0.0);
        let up = Vec3::new(-pitch.sin() * yaw.cos(), -pitch.sin() * yaw.sin(), pitch.cos());
        let eye = right.cross(up);
        let (_, (min, max)) = layout(self.canvas, self.inset, 0, 0.0);
        // (room round the edge for the names: less on a phone, where every pixel counts)
        let margin = if narrow_canvas(self.canvas) { Vec2::new(60.0, 70.0) } else { Vec2::new(110.0, 80.0) };
        let room = (max - min - margin).max(Vec2::splat(40.0));
        let goal = (min + max) * 0.5;
        // pixels on the screen per unit of length across the line of sight, at unit distance
        let focal = self.canvas.y * 0.5 / (0.31f32).tan();
        let points = &self.home;
        let reach = points.iter().map(|p| p.length()).fold(0.5, f32::max);
        let mut target = points.iter().copied().sum::<Vec3>() / points.len().max(1) as f32;
        let mut dist = 8.0f32;
        // seen in perspective, the near side looks larger than the far: fitted by
        // looking at the picture, moving the camera, and looking again
        for _ in 0..12 {
            let from = target + eye * dist;
            let seen: Vec<Vec2> = points
                .iter()
                .map(|p| {
                    let d = *p - from;
                    let depth = (-d.dot(eye)).max(1e-3);
                    self.canvas * 0.5 + Vec2::new(d.dot(right), -d.dot(up)) * (focal / depth)
                })
                .collect();
            let lo = seen.iter().fold(Vec2::MAX, |m, p| m.min(*p));
            let hi = seen.iter().fold(Vec2::MIN, |m, p| m.max(*p));
            let size = (hi - lo).max(Vec2::ONE);
            let scale = (room.x / size.x).min(room.y / size.y).min(680.0 / size.max_element());
            // move the middle of the picture to the middle of the free part, then
            // come nearer or go further by the scale wanted
            let shift = ((lo + hi) * 0.5 - goal) * (dist / focal);
            target += right * shift.x - up * shift.y;
            dist = (dist / scale).max(2.0 * reach);
        }
        View::Orbit { target, yaw, pitch, dist }
    }
}

impl Scene for Vectors {
    fn view(&self) -> View {
        self.the_view()
    }

    fn take_view(&mut self) -> Option<View> {
        std::mem::take(&mut self.new_view).then(|| self.the_view())
    }

    fn settings(&self, s: &mut ViewSettings) {
        s.grid_alpha = 0.07;
    }

    /// Only a swap moves by itself, and an arrow gliding to where its tip was put.
    fn animated(&self) -> bool {
        self.swap.is_some() || !self.arrived()
    }

    fn ui(&mut self, ui: &mut egui::Ui, _: &Gfx) {
        ui::note(ui, r"Two arrows, $\textcolor{green}{a}$ and $\textcolor{orange}{b}$, start from the same point, and there are two ways to multiply them. The dot product $a \cdot b$ is a number: the length of $b$'s shadow on the line of $a$, times the length of $a$. The cross product $a \times b$ is an arrow, at right angles to both, as long as the area of the parallelogram they span. Drag the tips of $a$ and $b$; the sums over the picture work out each product both ways with the numbers as they are.");

        ui::section(ui, "Things to try");
        for (k, trial) in TRIALS.iter().enumerate() {
            if ui::experiment(ui, trial.0, trial.1) {
                self.set_up(k);
            }
        }

        ui::section(ui, "Show");
        ui::choice(ui, &SHOWN, &mut self.shown);
        if ui.checkbox(&mut self.unit, "keep both at length 1").changed() && self.unit {
            self.make_unit();
        }
        ui::note(ui, "Otherwise the tips move in steps of 0.1, so that the sums over the picture can be checked by hand.");
        if ui.add_enabled(self.swap.is_none(), egui::Button::new(ui::label_text(ui, "swap $a$ and $b$", 12.5, None, false))).clicked() {
            self.start_swap();
        }

        ui::section(ui, "Reading the picture");
        let mut key: Vec<(Color, &str)> = vec![(A_COL, "$a$, and the line it lies on (dashed), which goes on both ways"), (B_COL, "$b$")];
        if self.shown != CROSS {
            key.push((SHADOW, "$b$'s shadow on the line of $a$: where $b$'s tip lands when dropped at a right angle (dashed white, with the white corner)"));
            key.push((BEHIND, r"the same shadow when it falls behind the start: then $a \cdot b$ is negative"));
        }
        if self.shown != DOT {
            key.push((AREA, r"the parallelogram $a$ and $b$ span, and the arrow $a \times b$, whose length is its area; the small corners show it at right angles to $a$ and to $b$"));
            key.push((pal::WHITE, r"the arc: the angle $\theta$ from $a$ to $b$; its arrowhead shows which way $a$ turns to reach $b$"));
        } else {
            key.push((pal::WHITE, r"the arc: the angle $\theta$ between $a$ and $b$"));
        }
        if self.unit {
            key.push((pal::DIM, "the faint circle: length 1, in the plane of $a$ and $b$"));
        }
        key.push((pal::FAINT, "thin grey lines: the $x$, $y$ and $z$ axes"));
        ui::key(ui, &key);

        ui::section(ui, "Words");
        ui::term(ui, "Length $|a|$:", r"how long the arrow is, $\sqrt{a_1^2 + a_2^2 + a_3^2}$, where $a_1, a_2, a_3$ are its steps along $x$, $y$ and $z$.");
        ui::term(ui, r"Angle $\theta$:", "the angle between $a$ and $b$, from 0° (the same way) to 180° (opposite ways).");
        ui::term(ui, "Dot product:", r"$a \cdot b = a_1 b_1 + a_2 b_2 + a_3 b_3$, a number. It is also $|a|\,|b| \cos\theta$.");
        ui::term(ui, "Shadow:", r"where $b$'s tip lands when it is dropped onto the line of $a$ at a right angle (also called the projection of $b$ onto $a$). Its length is $|b| \cos\theta$.");
        ui::term(ui, "Cross product:", r"$a \times b$, an arrow at right angles to $a$ and to $b$, of length $|a|\,|b| \sin\theta$, the area of the parallelogram they span.");
        ui::term(ui, "Right-hand rule:", r"curl the fingers of your right hand the way $a$ turns towards $b$: your thumb points along $a \times b$.");
        ui::term(ui, "Cosine similarity:", r"$a \cdot b / (|a|\,|b|) = \cos\theta$, how far two directions agree: 1 the same, 0 at right angles, −1 opposite. For arrows of length 1 it is the dot product itself.");

        ui::section(ui, "Numbers");
        let p = Pair::of(self.a, self.b);
        // (the points are formulas, and so every number here is set as one)
        ui::readout_colored(ui, "$a$", format!("${}$", coords(self.a)), A_COL);
        ui::readout_colored(ui, "$b$", format!("${}$", coords(self.b)), B_COL);
        ui::readout(ui, "$|a|, |b|$", format!("${}, {}$", num(p.la, 2), num(p.lb, 2)));
        ui::readout(ui, r"$\theta$", p.degrees().map_or("none".to_string(), |d| format!("${}$", degrees(d))));
        ui::readout_colored(ui, r"$a \cdot b$", format!("${}$", num(p.dot, 2)), if p.dot < 0.0 { BEHIND } else { SHADOW });
        ui::readout(ui, r"$\cos\theta = a \cdot b / (|a|\,|b|)$", p.theta.map_or("none".to_string(), |t| format!("${}$", num(t.cos(), 3))));
        let c = p.cross;
        ui::readout_colored(ui, r"$a \times b$", format!("$({}, {}, {})$", num(c.x, 2), num(c.y, 2), num(c.z, 2)), AREA);
        ui::readout(ui, r"$b \times a$", format!("$({}, {}, {})$", num(-c.x, 2), num(-c.y, 2), num(-c.z, 2)));
        ui::readout(ui, r"$|a \times b|$, the area", format!("${}$", num(p.area, 2)));
        if !(p.a_dir() && p.b_dir()) {
            ui::note(ui, "An arrow of length 0 has no direction: there is no angle, no shadow and no area, and both products are 0.");
        } else if p.in_line {
            ui::note(ui, r"$a$ and $b$ are in line: they span no area, so $a \times b = 0$, and $a \cdot b = \pm |a|\,|b|$.");
        }
        ui::see_also(ui, &[("lsq", "the shadow on a plane: projection"), ("gram", "making arrows perpendicular by taking shadows away"), ("span", "walks along arrows: everything two or three of them can reach"), ("packing", "directions as far apart as they can be")]);
    }

    fn frame(&mut self, f: &mut Frame) {
        // the canvas decides where the sums go, and with that where the picture is
        let view = f.view_size();
        if (view - self.canvas).abs().max_element() > 24.0 || f.top_inset != self.inset {
            (self.canvas, self.inset) = (view, f.top_inset);
            self.new_view = true;
        }

        self.advance(f.dt);
        if self.swap.is_none() {
            for (k, color) in [A_COL, B_COL].into_iter().enumerate() {
                let mut grip = self.grip[k];
                if f.drag3(1 + k as u32, &mut grip, None, color) {
                    self.grip[k] = grip;
                    let keep = if k == 0 { self.a } else { self.b };
                    let tip = self.settle(grip, keep);
                    if k == 0 {
                        self.a = tip;
                    } else {
                        self.b = tip;
                    }
                }
            }
            if !f.handles.dragging() {
                self.grip = [self.a, self.b];
            }
        } else {
            f.dot(self.a, 12.0, A_COL);
            f.dot(self.b, 12.0, B_COL);
        }
        // The arrows glide to where the tips have stepped. Not while the scene's time
        // stands still (nothing glides then, and a drag must still be seen), nor
        // while tips kept at length 1 are dragged: those do not step, and follow
        // the hand exactly.
        // (By the clock, as the view's own glide is: it answers the hand.)
        let at_once = f.dt <= 0.0 || (self.unit && f.handles.dragging());
        self.glide(f.real_dt, at_once);
        draw(f, self);
    }
}

/// The whole picture.
fn draw(f: &mut Frame, s: &Vectors) {
    // the picture is of the arrows as they are drawn (see `Vectors::drawn`); every
    // number written on it is that of a and b themselves, as in the sums
    let (a, b) = (s.drawn[0], s.drawn[1]);
    let p = Pair::of(a, b);
    let said = Pair::of(s.a, s.b);
    let view = f.view_size();
    let (dot_on, cross_on) = (s.shown != CROSS, s.shown != DOT);
    let wpp = |f: &Frame, at: Vec3| f.cam.world_per_px(at) * f.px;
    let mut room = Room::new(f);

    // the sums first: everything else keeps clear of them (measured as they are set)
    let lines = sums(s.a, s.b, s.shown);
    let col = lines.iter().map(|l| width_of(f, &l.0, SUM_SIZE)).fold(0.0, f32::max);
    let body = lines.iter().map(|l| width_of(f, &l.2, SUM_SIZE)).fold(0.0, f32::max);
    let (panel, _) = layout(view, f.top_inset, lines.len(), col + body + 28.0);
    if let Some(sums) = panel {
        room.wall(sums.0, sums.1);
    }

    // the axes, faint, for the coordinates the sums are made of
    let reach = 3.0;
    for d in Vec3::AXES {
        f.line(-d * reach, d * reach, 1.0, pal::FAINT.a(0.8));
        room.line(f, -d * reach, d * reach);
    }

    let ua = p.a.normalize_or_zero().as_vec3();
    let foot = p.foot().as_vec3();
    let shadow = p.shadow();
    let shadow_col = if p.dot < 0.0 { BEHIND } else { SHADOW };
    let spans = p.a_dir() && p.b_dir() && !p.in_line;
    let cross = p.cross.as_vec3();

    // the line of a, which goes on both ways (the shadow can fall behind the start)
    if p.a_dir() {
        let long = 12.0;
        f.dashed(-ua * long, ua * long, 1.2, A_COL.a(if dot_on { 0.5 } else { 0.3 }));
        room.line(f, -ua * long, ua * long);
    }

    // the parallelogram, see-through
    if cross_on && spans {
        f.dl.tris_alpha.quad(Vec3::ZERO, a, a + b, b, AREA.a(0.22));
        f.line(a, a + b, 1.4, AREA.a(0.75));
        f.line(b, a + b, 1.4, AREA.a(0.75));
        room.line(f, a, a + b);
        room.line(f, b, a + b);
        room.shape(f, &[Vec3::ZERO, a, a + b, b]);
    }

    // b's shadow on the line of a, under a's own arrow
    if dot_on && p.b_dir() && shadow.is_some() {
        f.line(Vec3::ZERO, foot, 9.0, shadow_col.a(0.6));
        room.line(f, Vec3::ZERO, foot);
    }

    // b's tip dropped onto the line of a at a right angle: in the parallelogram,
    // that is its height over a
    if spans {
        // (not when the shadow is next to nothing: the drop is then b itself)
        if foot.length() > 4.0 * wpp(f, foot) {
            f.dashed(b, foot, 1.6, pal::WHITE.a(0.8));
        }
        room.line(f, b, foot);
        let size = 13.0 * wpp(f, foot);
        let height = p.height().unwrap_or(0.0) as f32;
        if height > 2.0 * size {
            let along = if foot.length() > 1.5 * size { -foot } else { ua };
            right_angle(f, foot, along, b - foot, size, pal::WHITE.a(0.9));
        }
    }

    // the angle, as an arc from a to b
    let mut theta_at: Option<Vec3> = None;
    if let Some(theta) = p.theta {
        let theta = theta as f32;
        let r = (0.45f32).min(0.4 * p.la.min(p.lb) as f32);
        // the direction across a in the plane of the two, towards b
        let w = (b - ua * ua.dot(b)).normalize_or_zero();
        if spans && w != Vec3::ZERO {
            let at = |t: f32| (ua * t.cos() + w * t.sin()) * r;
            let n = 40;
            let pts: Vec<Vec3> = (0..=n).map(|i| at(theta * i as f32 / n as f32)).collect();
            f.path(pts.iter().copied(), 1.6, pal::WHITE.a(0.9));
            for k in (0..n).step_by(4) {
                room.line(f, pts[k], pts[(k + 4).min(n)]);
            }
            if cross_on {
                // which way a turns to reach b: the curl of the right hand's fingers
                let back = (13.0 * wpp(f, at(theta)) / r).min(theta * 0.5);
                f.arrow(at(theta - back), at(theta), 1.6, pal::WHITE);
            }
            theta_at = Some(at(theta * 0.5));
        } else {
            theta_at = Some(Vec3::ZERO);
        }
    }

    // a × b stands at a right angle to a and to b: a small corner with each
    if cross_on && spans {
        let size = (17.0 * wpp(f, Vec3::ZERO)).min(0.3 * p.la.min(p.lb).min(p.area) as f32);
        right_angle(f, Vec3::ZERO, cross, a, size, AREA.mix(pal::WHITE, 0.4));
        right_angle(f, Vec3::ZERO, cross, b, size, AREA.mix(pal::WHITE, 0.4));
    }

    // the circle of length 1, when the arrows are kept to it
    let mut unit_circle: Option<Vec3> = None;
    if s.unit {
        let w = (b - ua * ua.dot(b)).try_normalize().unwrap_or_else(|| ua.any_orthonormal_vector());
        let normal = ua.cross(w);
        if normal != Vec3::ZERO {
            f.circle(Vec3::ZERO, normal, 1.0, 1.0, pal::DIM.a(0.6));
            unit_circle = Some(-w);
        }
    }

    // the arrows themselves
    if cross_on && spans {
        f.arrow(Vec3::ZERO, cross, 3.0, AREA);
        room.line(f, Vec3::ZERO, cross);
    }
    f.arrow(Vec3::ZERO, a, 3.4, A_COL);
    f.arrow(Vec3::ZERO, b, 3.4, B_COL);
    room.line(f, Vec3::ZERO, a);
    room.line(f, Vec3::ZERO, b);

    // and their names and numbers, the most important first
    let screen_dir = |f: &Frame, from: Vec3, to: Vec3| match (on_screen(f, from), on_screen(f, to)) {
        (Some(p), Some(q)) => q - p,
        _ => Vec2::ZERO,
    };
    let outwards = |f: &Frame, v: Vec3| if v.length() > 1e-6 { screen_dir(f, Vec3::ZERO, v) } else { Vec2::new(1.0, -1.0) };
    let dir_a = outwards(f, a);
    let dir_b = outwards(f, b);
    room.label(f, "a", a, "$a$", 17.0, A_COL, dir_a, 9.0, false);
    room.label(f, "b", b, "$b$", 17.0, B_COL, dir_b, 9.0, false);
    if cross_on && spans {
        let dir = outwards(f, cross);
        room.label(f, "cross", cross, r"$a \times b$", 15.0, AREA, dir, 8.0, false);
    }
    if let (Some(mid), Some(deg)) = (theta_at, said.degrees().or(p.degrees())) {
        if p.in_line {
            // no arc to put it on: beside the start, across the line
            let text = if cross_on { format!(r"$\theta = {}$: in line, so no area, and $a \times b = 0$", degrees(deg)) } else { format!(r"$\theta = {}$: in line", degrees(deg)) };
            let across = if dir_a.perp().y > 0.0 { dir_a.perp() } else { -dir_a.perp() };
            room.label(f, "angle", mid, &text, 13.0, pal::TEXT, across, 22.0, false);
        } else {
            // beside the arc; where that is crowded (a small picture, a narrow
            // angle), the number alone, and failing that the name alone
            let dir = outwards(f, mid);
            if !room.label_any(f, "angle", &[mid, mid * 1.4], &format!(r"$\theta = {}$", degrees(deg)), 13.0, pal::TEXT, dir, 6.0, true)
                && !room.label_any(f, "angle", &[mid, mid * 1.4, mid * 0.6], &degrees(deg), 13.0, pal::TEXT, dir, 5.0, true)
            {
                room.label_any(f, "angle", &[mid, mid * 1.4], r"$\theta$", 14.0, pal::TEXT, dir, 5.0, false);
            }
        }
    }
    if dot_on && p.b_dir() {
        if let Some(sh) = shadow {
            // beside the middle of the shadow, on the side away from b
            let along = if p.dot < 0.0 { -dir_a } else { dir_a };
            let across = along.perp() * along.perp().dot(screen_dir(f, foot, b)).signum();
            let away = if across == Vec2::ZERO { Vec2::new(0.0, 1.0) } else { -across };
            room.label(f, "shadow", foot * 0.5, &format!("shadow {}", num(said.shadow().unwrap_or(sh), 2)), 13.0, shadow_col, away, 12.0, false);
        }
    }
    if cross_on && spans {
        if let Some(h) = p.height() {
            // beside the middle of the height, on the side away from the start
            let away = if p.dot < 0.0 { -dir_a } else { dir_a };
            room.label(f, "height", (b + foot) * 0.5, &format!("height {}", num(said.height().unwrap_or(h), 2)), 12.5, pal::TEXT, away, 7.0, true);
        }
        // in the middle of the parallelogram, or wherever inside it there is room;
        // a thin one (or one seen edge on) is named from beside its far corner
        let text = format!("area {}", num(said.area, 2));
        let inner: Vec<Vec3> = [(0.5, 0.5), (0.65, 0.5), (0.5, 0.65), (0.35, 0.5), (0.5, 0.35), (0.75, 0.75), (0.75, 0.3), (0.3, 0.75)].iter().map(|(s, t)| a * *s + b * *t).collect();
        let colour = AREA.mix(pal::WHITE, 0.45);
        if !room.label_any(f, "area", &inner, &text, 13.0, colour, Vec2::ZERO, 0.0, true) {
            room.label(f, "area beside", a + b, &text, 13.0, colour, outwards(f, a + b), 8.0, false);
        }
    }
    // what has become of the products when there is nothing to draw
    if !p.a_dir() || !p.b_dir() {
        let says = if p.a_dir() || p.b_dir() { format!("{} length 0: no direction", lengthless(&p)) } else { "$a$ and $b$ have length 0: no directions".to_string() };
        room.label(f, "none", Vec3::ZERO, &says, 13.0, pal::TEXT, Vec2::new(0.0, 1.0), 14.0, false);
    }
    if let Some(at) = unit_circle {
        room.label(f, "unit", at, "length 1", 12.0, pal::DIM, outwards(f, at), 6.0, true);
    }
    for (k, (name, text)) in [("x", "$x$"), ("y", "$y$"), ("z", "$z$")].into_iter().enumerate() {
        let d = Vec3::AXES[k] * reach;
        room.label(f, name, d, text, 12.0, pal::DIM, outwards(f, d), 4.0, true);
    }

    // the sums, set a little smaller where the canvas is too narrow for them (a phone's)
    if let Some((min, max)) = panel {
        hud_panel(f, min, max);
        let room = max.x - min.x - 28.0;
        let shrink = if col + body > room + 0.5 { room / (col + body) } else { 1.0 };
        let size = SUM_SIZE * shrink;
        let x = min.x + 12.0 + col * shrink;
        for (i, (left, left_col, rest, rest_col)) in lines.iter().enumerate() {
            let y = min.y + 8.0 + SUM_LINE * (i as f32 + 0.5);
            if !left.is_empty() {
                f.text_px(Vec2::new(x, y), left.as_str(), size, *left_col, Align2::RIGHT_CENTER);
            }
            f.text_px(Vec2::new(x + 6.0, y), rest.as_str(), size, *rest_col, Align2::LEFT_CENTER);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// An experiment as its button sets it up, played to the end at 60 frames a second.
    fn trial(k: usize) -> Vectors {
        let mut scene = Vectors::create();
        scene.set_up(k);
        for _ in 0..1000 {
            scene.advance(1.0 / 60.0);
        }
        assert!(scene.swap.is_none());
        scene
    }

    fn pair(s: &Vectors) -> Pair {
        Pair::of(s.a, s.b)
    }

    /// The last number on a line of the sums ("$... = 1.88$").
    fn result(line: &SumLine) -> &str {
        line.2.rsplit("= ").next().unwrap_or("").split(',').next().unwrap_or("").trim_end_matches('$')
    }

    /// A tip that has stepped is drawn gliding there: a part of the way each frame,
    /// never past it, exactly there within a quarter of a second, and then at rest.
    /// On a faster display the glide takes as long.
    #[test]
    fn a_stepped_tip_is_drawn_gliding_to_its_place() {
        for hz in [60.0f32, 144.0] {
            let mut s = Vectors::create();
            assert!(s.arrived() && !s.animated());
            let from = s.a;
            s.a += Vec3::new(STEP, 0.0, -STEP);
            assert!(s.animated(), "a glide is movement: frames must keep coming");
            let mut left = (s.a - s.drawn[0]).length();
            let mut frames = 0;
            while !s.arrived() {
                s.glide(1.0 / hz, false);
                let now = (s.a - s.drawn[0]).length();
                assert!(now < left && (s.drawn[0] - from).dot(s.a - from) > 0.0, "frame {frames}: {now} after {left}");
                // (b has not moved, and is not drawn moving)
                assert_eq!(s.drawn[1], s.b);
                left = now;
                frames += 1;
                // two thirds of the way after `GLIDE` seconds
                if frames == (GLIDE * hz).round() as usize {
                    let share = 1.0 - now / (s.a - from).length();
                    assert!((0.55..0.75).contains(&share), "{share} of the way after {frames} frames at {hz} Hz");
                }
                assert!(frames as f32 / hz < 0.4, "still gliding after {frames} frames at {hz} Hz");
            }
            assert!(s.drawn == [s.a, s.b] && !s.animated());
        }
        // an experiment's picture, a swap and a paused scene are drawn as they stand
        let mut s = Vectors::create();
        s.set_up(1);
        assert!(s.arrived());
        s.set_up(SWAP_TRIAL);
        for _ in 0..40 {
            s.advance(1.0 / 60.0);
            assert!(s.swap.is_some() && s.arrived());
        }
        let mut s = Vectors::create();
        s.b = Vec3::new(0.0, 2.0, 0.0);
        s.glide(0.0, true);
        assert!(s.arrived());
        // and nothing moves while no time passes
        s.b = Vec3::new(0.5, 2.0, 0.0);
        let was = s.drawn;
        s.glide(0.0, false);
        assert_eq!(s.drawn, was);
    }

    /// The captions of the "things to try", held to account.
    #[test]
    fn the_experiments_show_what_their_captions_say() {
        // at right angles: the shadow is the start, a . b = 0 and the products in the
        // sum cancel; the parallelogram is a rectangle with the largest area these
        // lengths allow, |a| |b| = 3.09
        let s = trial(0);
        let p = pair(&s);
        assert!(p.dot == 0.0 && p.shadow() == Some(0.0) && p.foot() == DVec3::ZERO && degrees(p.degrees().unwrap()) == "90.0°", "{p:?}");
        let terms: Vec<f64> = (0..3).map(|i| p.a[i] * p.b[i]).collect();
        assert!(terms.iter().filter(|t| t.abs() > 0.1).count() == 3, "{terms:?}");
        assert!((p.area - p.la * p.lb).abs() < 1e-12 && num(p.area, 2) == "3.09" && s.shown == BOTH);
        for degrees in (1..180).filter(|d| *d != 90) {
            assert!(p.la * p.lb * (degrees as f64).to_radians().sin() < p.area - 1e-6);
        }
        let lines = sums(s.a, s.b, s.shown);
        assert!(lines[1..4].iter().all(|l| result(l) == "0.00"), "{lines:?}");

        // past a right angle: 124.9 degrees, the shadow falls behind the start (pink),
        // and a . b = -2.04 from the sum and from |a| |b| cos θ
        let s = trial(1);
        let p = pair(&s);
        assert_eq!(degrees(p.degrees().unwrap()), "124.9°");
        assert!(p.shadow().unwrap() < 0.0 && p.foot().dot(p.a) < 0.0 && num(p.dot, 2) == "−2.04" && s.shown == DOT);
        let lines = sums(s.a, s.b, s.shown);
        assert!(lines.len() == 4 && lines[1..].iter().all(|l| result(l) == "−2.04"), "{lines:?}");

        // in line: b is half of a; its shadow is b itself, a . b = |a| |b| = 1.96;
        // no area, and a x b = 0
        let s = trial(2);
        let p = pair(&s);
        assert!(p.b * 2.0 == p.a && p.in_line && p.area == 0.0 && p.cross == DVec3::ZERO);
        assert!((p.shadow().unwrap() - p.lb).abs() < 1e-12 && (p.foot() - p.b).length() < 1e-12);
        assert!((p.dot - p.la * p.lb).abs() < 1e-12 && num(p.dot, 2) == "1.96" && num(p.la * p.lb, 2) == "1.96");
        assert_eq!(degrees(p.degrees().unwrap()), "0.0°");

        // swap a and b: same lengths, the angle and a . b = 1.64 kept all the way;
        // at the end each is exactly where the other was, and a x b points the other way
        let mut s = Vectors::create();
        s.set_up(3);
        let (a0, b0) = (s.a, s.b);
        let start = pair(&s);
        assert!((start.la - start.lb).abs() < 1e-12 && num(start.dot, 2) == "1.64");
        let mut frames = 0;
        while s.swap.is_some() {
            s.advance(1.0 / 60.0);
            let p = pair(&s);
            assert!((p.dot - start.dot).abs() < 1e-5 && (p.area - start.area).abs() < 1e-5, "frame {frames}: {p:?}");
            assert!((p.theta.unwrap() - start.theta.unwrap()).abs() < 1e-5);
            frames += 1;
        }
        let end = pair(&s);
        assert!(frames > 60 && s.a == b0 && s.b == a0, "{frames} frames");
        assert!(end.cross == -start.cross && end.dot == start.dot && s.shown == BOTH);

        // unit vectors: both of length 1, θ = 60 degrees and a . b = 0.50 = cos θ;
        // dragged round, a . b stays cos θ, from 1 through 0 to -1
        let mut s = trial(4);
        let p = pair(&s);
        assert!(s.unit && (p.la - 1.0).abs() < 1e-6 && (p.lb - 1.0).abs() < 1e-6 && s.shown == DOT);
        assert!(degrees(p.degrees().unwrap()) == "60.0°" && num(p.dot, 2) == "0.50" && (p.dot - p.theta.unwrap().cos()).abs() < 1e-6);
        assert!((p.shadow().unwrap() - p.dot).abs() < 1e-6);
        let a = s.a;
        let across = a.any_orthonormal_vector();
        for (to, want) in [(a * 2.5, "1.00"), (across * 0.3, "0.00"), (-a * 0.7, "−1.00")] {
            s.b = s.settle(to, s.b);
            let p = pair(&s);
            assert!((p.lb - 1.0).abs() < 1e-6 && num(p.dot, 2) == want, "{to}: {p:?}");
        }
        let mut rng = Rng::new(3);
        for _ in 0..200 {
            s.b = s.settle(Vec3::new(rng.normal(), rng.normal(), rng.normal()) * 2.0, s.b);
            let p = pair(&s);
            assert!((p.lb - 1.0).abs() < 1e-6 && (p.dot - p.theta.unwrap().cos()).abs() < 1e-6, "{p:?}");
        }
    }

    /// Random pairs on the grid the tips move on (and off it), and the ones the scene
    /// sets up: the two ways of working out each product agree, the shadow ends where
    /// b's tip lands at a right angle, a x b stands at right angles to both and
    /// follows the right hand, b x a is its opposite, and the sign of a . b says on
    /// which side of a right angle θ is.
    #[test]
    fn the_two_ways_of_working_out_each_product_agree() {
        let mut rng = Rng::new(9);
        let mut pairs: Vec<(Vec3, Vec3)> = (0..TRIALS.len()).map(trial_pair).collect();
        pairs.push((Vec3::from(START.0), Vec3::from(START.1)));
        let mut point = || Vec3::new(rng.normal(), rng.normal(), rng.normal()) * 1.5;
        for k in 0..400 {
            let (a, b) = (point(), point());
            pairs.push(if k % 2 == 0 { ((a / STEP).round() * STEP, (b / STEP).round() * STEP) } else { (a, b) });
        }
        let (mut acute, mut obtuse) = (0, 0);
        for (a, b) in pairs {
            let p = Pair::of(a, b);
            if !(p.a_dir() && p.b_dir()) {
                continue;
            }
            let theta = p.theta.unwrap();
            let scale = p.la * p.lb;
            assert!((p.dot - scale * theta.cos()).abs() < 1e-9 * scale && (p.dot - p.la * p.shadow().unwrap()).abs() < 1e-9 * scale, "{a} {b}");
            assert!((p.area - scale * theta.sin()).abs() < 1e-9 * scale && (p.area - p.la * p.height().unwrap()).abs() < 1e-9 * scale, "{a} {b}");
            assert!((p.b - p.foot()).dot(p.a).abs() < 1e-9 * scale && (p.foot().length() - p.shadow().unwrap().abs()).abs() < 1e-9 * p.lb);
            assert!(((p.b - p.foot()).length() - p.height().unwrap()).abs() < 1e-9 * p.lb);
            assert!(p.cross.dot(p.a).abs() < 1e-9 * scale * p.la && p.cross.dot(p.b).abs() < 1e-9 * scale * p.lb);
            if !p.in_line {
                // right-handed: a, b, a x b turn the way x, y, z do
                assert!(p.a.cross(p.b).dot(p.cross) > 0.0);
                assert!((Pair::of(b, a).cross + p.cross).length() < 1e-12 * scale && Pair::of(b, a).dot == p.dot);
            }
            if p.dot > 0.0 {
                acute += 1;
                assert!(theta < std::f64::consts::FRAC_PI_2);
            } else if p.dot < 0.0 {
                obtuse += 1;
                assert!(theta > std::f64::consts::FRAC_PI_2);
            } else {
                assert!((theta - std::f64::consts::FRAC_PI_2).abs() < 1e-12);
            }
        }
        assert!(acute > 100 && obtuse > 100, "{acute} {obtuse}");
    }

    /// On the grid of tenths the sum is exact: the products written out add up to
    /// the total written, every way of working it out ends on the same number, and
    /// each component of a x b is what its formula gives.
    #[test]
    fn the_sums_on_screen_add_up_as_written() {
        let read = |s: &str| s.trim().trim_matches(|c| c == '(' || c == ')').replace('−', "-").parse::<f64>().unwrap_or(f64::NAN);
        let mut rng = Rng::new(4);
        for k in 0..300 {
            let mut point = || (Vec3::new(rng.normal(), rng.normal(), rng.normal()) * 1.5 / STEP).round() * STEP;
            let (a, b) = if k < TRIALS.len() - 1 { trial_pair(k) } else { (point(), point()) };
            let lines = sums(a, b, BOTH);
            assert_eq!(lines.len(), 9, "{lines:?}");
            // "$= 1.8 \times 0.4 + 0.8 \times 1.6 + (−0.2) \times 0.6 = 1.88$"
            let line = &lines[1].2;
            let parts: Vec<&str> = line.trim_start_matches("$= ").trim_end_matches('$').split(" = ").collect();
            let written: f64 = parts[0].split(" + ").map(|t| t.split(r" \times ").map(read).product::<f64>()).sum();
            assert!(parts.len() == 2 && num(written, 2) == parts[1], "{line}");
            assert!(lines[1..4].iter().all(|l| result(l) == parts[1]), "{lines:?}");
            // the area, both ways, and the components of a x b
            assert_eq!(result(&lines[7]), result(&lines[8]), "{lines:?}");
            let p = Pair::of(a, b);
            let e = (exact(a), exact(b));
            let by_formula = [e.0.y * e.1.z - e.0.z * e.1.y, e.0.z * e.1.x - e.0.x * e.1.z, e.0.x * e.1.y - e.0.y * e.1.x];
            assert_eq!(lines[6].2, format!("$= ({}, {}, {})$", num(by_formula[0], 2), num(by_formula[1], 2), num(by_formula[2], 2)));
            assert!(p.in_line || num(p.area, 2) == result(&lines[7]));
        }
    }

    /// An arrow of no length, two in line either way, two on top of each other,
    /// very short ones: every number drawn or read out is finite, and what the
    /// picture says has happened has.
    #[test]
    fn degenerate_pairs_leave_everything_finite() {
        let a = Vec3::from(START.0);
        let cases = [
            (Vec3::ZERO, a),
            (a, Vec3::ZERO),
            (Vec3::ZERO, Vec3::ZERO),
            (a, a),
            (a, -a),
            (a, a * 2.0),
            (a * 1e-7, a),
            (Vec3::new(1e-7, 0.0, 0.0), Vec3::new(0.0, 1e-7, 0.0)),
            (a, a + Vec3::new(0.0, 0.0, 1e-7)),
            (Vec3::splat(REACH), Vec3::splat(-REACH)),
        ];
        for (a, b) in cases {
            let p = Pair::of(a, b);
            assert!(p.dot.is_finite() && p.cross.is_finite() && p.area.is_finite() && p.foot().is_finite(), "{a} {b}: {p:?}");
            assert!(p.theta.is_none_or(f64::is_finite) && p.shadow().is_none_or(f64::is_finite) && p.height().is_none_or(f64::is_finite));
            // no length: no angle; in line: no area, no a x b
            assert_eq!(p.theta.is_some(), p.a_dir() && p.b_dir(), "{a} {b}");
            if p.in_line || !(p.a_dir() && p.b_dir()) {
                assert!(p.area == 0.0 && p.cross == DVec3::ZERO, "{a} {b}");
            }
            for shown in [DOT, CROSS, BOTH] {
                for (left, _, rest, _) in sums(a, b, shown) {
                    let line = left + &rest;
                    assert!(!line.contains('?') && !line.contains("NaN") && !line.contains("inf"), "{a} {b}: {line}");
                    // (and its mathematics is set, not drawn as written)
                    assert!(crate::typeset::inspect(&line).error.is_none(), "{a} {b}: {line}");
                }
            }
            for k in 0..=10 {
                let (a, b) = swapping(a, b, k as f32 / 10.0);
                assert!(a.is_finite() && b.is_finite());
            }
            let mut s = Vectors::create();
            (s.a, s.b) = (a, b);
            s.make_unit();
            assert!((s.a.length() - 1.0).abs() < 1e-6 && (s.b.length() - 1.0).abs() < 1e-6, "{a} {b}");
            s.frame_on_pair();
            assert!(s.home.iter().all(|p| p.is_finite()));
            assert!(matches!(s.the_view(), View::Orbit { target, dist, .. } if target.is_finite() && dist.is_finite() && dist > 0.0));
        }
        // the two cases the panel names
        assert!(Pair::of(a, a * -2.0).in_line && !Pair::of(Vec3::ZERO, a).in_line);
        // dragging a tip to the origin with lengths kept at 1 leaves it where it was
        let mut s = trial(4);
        let keep = s.b;
        assert_eq!(s.settle(Vec3::ZERO, keep), keep);
        s.unit = false;
        assert_eq!(s.settle(Vec3::ZERO, keep), Vec3::ZERO);
    }

    /// The tips move in tenths, and no further out than `REACH`; kept at length 1,
    /// they stay on the sphere.
    #[test]
    fn dragged_tips_move_in_tenths() {
        let s = Vectors::create();
        let tip = s.settle(Vec3::new(1.234, -0.051, 2.96), Vec3::X);
        assert!(tip == Vec3::new(1.2, -0.1, 3.0) && on_grid(tip) && exact(tip) == DVec3::new(1.2, -0.1, 3.0), "{tip}");
        let far = s.settle(Vec3::new(30.0, 40.0, 0.0), Vec3::X);
        assert!(far == Vec3::new(2.4, 3.2, 0.0), "{far}");
        assert_eq!(coords(tip), "(1.2, −0.1, 3.0)");
        assert_eq!(coords(Vec3::new(0.5, 0.25, 0.0)), "(0.50, 0.25, 0.00)");
    }

    /// The sums and the picture share the canvas without leaving it, whatever its
    /// shape, and the view fitted to it is a view.
    #[test]
    fn the_canvas_is_shared_on_every_screen() {
        for (x, y) in [(1070.0f32, 1000.0f32), (1100.0, 700.0), (390.0, 800.0), (360.0, 640.0), (600.0, 900.0), (844.0, 390.0), (640.0, 480.0), (240.0, 180.0), (2000.0, 1300.0), (8.0, 8.0)] {
            let view = Vec2::new(x, y);
            let inset = if x < 900.0 { 46.0 } else { 0.0 };
            let lines = sums(Vec3::from(START.0), Vec3::from(START.1), BOTH);
            let width = lines.iter().map(|l| text_width(&l.0, SUM_SIZE)).fold(0.0, f32::max) + lines.iter().map(|l| text_width(&l.2, SUM_SIZE)).fold(0.0, f32::max) + 28.0;
            let (sums, (min, max)) = layout(view, inset, lines.len(), width);
            if let Some((lo, hi)) = sums {
                assert!(lo.x >= 0.0 && lo.y >= inset && hi.x <= x && hi.y <= y && hi.y - lo.y >= lines.len() as f32 * SUM_LINE, "{x}x{y}: {lo} {hi}");
                // the sums lie beside or under the picture's part, not across the middle of it
                assert!(hi.y <= min.y + (max.y - min.y) * 0.25 || lo.y >= max.y, "{x}x{y}");
            }
            assert!(min.x >= 0.0 && min.y >= 0.0 && max.x <= x && max.y <= y && max.x > min.x && max.y > min.y, "{x}x{y}");
            let mut s = Vectors::create();
            (s.canvas, s.inset) = (view, inset);
            for k in 0..TRIALS.len() {
                s.set_up(k);
                assert!(matches!(s.the_view(), View::Orbit { target, dist, .. } if target.is_finite() && dist.is_finite() && dist > 0.0), "{x}x{y}");
            }
        }
    }
}

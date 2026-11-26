//! One map after another: matrix products, why their order matters, and inverses.

use super::*;

pub const DEF: SceneDef = SceneDef {
    id: "compose",
    title: "One map after another",
    group: "Linear algebra: vectors and maps",
    blurb: "Matrix products, why their order matters, and what an inverse undoes.",
    make: |_| Box::new(Compose::create()),
};

/// Experiments: (button, what to look for). The matrices each one sets are `trial`.
const TRIALS: [(&str, &str); 5] = [
    (
        "a turn, then a shear",
        r"$B$ turns the plane a quarter turn; then $A$ shears it, sliding every point sideways by its height. The F's long stroke ends lying flat, pointing left. In the other order (purple) the shear tilts the stroke first and the turn carries the tilt along: it ends leaning at 45°. The same two maps, two different ends: $AB \ne BA$.",
    ),
    (
        "two turns",
        "$B$ turns by 30°, $A$ by 60°. The purple outline of the other order lies exactly on the result: turns about the same point add up, $30° + 60° = 90°$, whichever comes first, so $AB = BA$. Stretches along the $x$ and $y$ axes also commute with each other.",
    ),
    (
        "areas multiply",
        r"$B$ doubles every area ($\det B = 2$); $A$ flips the plane over and multiplies areas by 1.5 ($\det A = -1.5$). The square's area goes from 1 to 2 to −3: $\det AB = \det A \times \det B$. The minus sign is the flip: the F ends mirrored. The other order ends elsewhere, with the same area −3.",
    ),
    (
        "a map and its inverse",
        r"$B$ turns the plane by 45° and enlarges it ($\det B = 2$). $A = B^{-1}$ undoes it: everything returns to where it started, so $AB = I$, and $\det A = 1/2$. The purple outline is home too: $BA = I$. (For another $B$, the $B^{-1}$ button under matrix $A$ finds its inverse, if it has one.)",
    ),
    (
        "a squash cannot be undone",
        r"$B$ drops every point straight down onto the $x$ axis ($\det B = 0$). The red dots $p$ and $q$, one above the other, land on the same spot. $A$ then tries to undo it by tripling every height, and nothing moves: every height is now 0, and $3 \times 0 = 0$. No map can part $p$ and $q$ again, since it sends one spot to one place: $B$ has no inverse, and $\det AB = 0$ whatever $A$ is.",
    ),
];

/// A matrix by its columns: where î goes, then where ĵ goes.
fn cols(c: [f32; 4]) -> Mat2 {
    Mat2::from_cols_array(&c)
}

const QUARTER_TURN: [f32; 4] = [0.0, 1.0, -1.0, 0.0];
const SHEAR: [f32; 4] = [1.0, 0.0, 1.0, 1.0];

/// The matrices A and B of experiment `k`.
fn trial(k: usize) -> (Mat2, Mat2) {
    match k {
        0 => (cols(SHEAR), cols(QUARTER_TURN)),
        1 => (Mat2::from_angle(PI / 3.0), Mat2::from_angle(PI / 6.0)),
        2 => (cols([-1.0, 0.5, 0.0, 1.5]), cols([2.0, 0.0, 0.0, 1.0])),
        3 => (cols([0.5, -0.5, 0.5, 0.5]), cols([1.0, 1.0, -1.0, 1.0])),
        _ => (cols([1.0, 0.0, 0.0, 3.0]), cols([1.0, 0.0, 0.0, 0.0])),
    }
}

/// Quick choices for either matrix: (button, the matrix by its columns).
const PRESETS: [(&str, [f32; 4]); 6] = [
    ("turn", QUARTER_TURN),
    ("shear", SHEAR),
    ("stretch", [2.0, 0.0, 0.0, 1.0]),
    ("flip", [-1.0, 0.0, 0.0, 1.0]),
    ("squash", [1.0, 0.0, 0.0, 0.0]),
    ("$I$", [1.0, 0.0, 0.0, 1.0]),
];

/// Seconds each map takes to play.
const STAGE_SECONDS: f32 = 1.6;

/// The F painted on the unit square, as three bars (corners anticlockwise), and its
/// outline. Lopsided on purpose: turned, it is still an F; mirrored, it is not.
const F_BARS: [[Vec2; 4]; 3] = [
    [Vec2::new(0.18, 0.12), Vec2::new(0.36, 0.12), Vec2::new(0.36, 0.88), Vec2::new(0.18, 0.88)],
    [Vec2::new(0.36, 0.70), Vec2::new(0.82, 0.70), Vec2::new(0.82, 0.88), Vec2::new(0.36, 0.88)],
    [Vec2::new(0.36, 0.42), Vec2::new(0.66, 0.42), Vec2::new(0.66, 0.58), Vec2::new(0.36, 0.58)],
];
const F_OUTLINE: [Vec2; 10] = [
    Vec2::new(0.18, 0.12),
    Vec2::new(0.36, 0.12),
    Vec2::new(0.36, 0.42),
    Vec2::new(0.66, 0.42),
    Vec2::new(0.66, 0.58),
    Vec2::new(0.36, 0.58),
    Vec2::new(0.36, 0.70),
    Vec2::new(0.82, 0.70),
    Vec2::new(0.82, 0.88),
    Vec2::new(0.18, 0.88),
];

/// The point midway between the two red points that show a squash.
const MERGE_AT: Vec2 = Vec2::new(-0.6, 0.5);

/// Where the area is written: the corner of the square the F leaves empty.
const AREA_AT: Vec2 = Vec2::new(0.66, 0.22);

/// The closest the view comes, in logical pixels per unit: a picture of small
/// matrices is not blown up past the point where the grid stops looking like one.
const MOST_PER_UNIT: f32 = 150.0;

/// The determinant, as far as the numbers can tell it: 0 when it is no larger than
/// the error in storing the four entries (about one part in ten million each), since
/// the matrix may then be exactly flat as it was typed. (In double precision: the
/// inverse is worked out from it.)
fn det(m: Mat2) -> f64 {
    let [a, b, c, d] = m.to_cols_array().map(f64::from);
    let det = a * d - c * b;
    let size = a.abs().max(b.abs()).max(c.abs()).max(d.abs());
    if det.abs() <= 1e-6 * size * size || !det.is_finite() {
        0.0
    } else {
        det
    }
}

/// The inverse, by its columns, if there is one. Near flat its entries are enormous,
/// which is why it is worked out in double precision.
fn inverse(m: Mat2) -> Option<[f64; 4]> {
    let d = det(m);
    let [a, b, c, e] = m.to_cols_array().map(f64::from);
    let inv = [e / d, -b / d, -c / d, a / d];
    (d != 0.0 && inv.iter().all(|v| v.is_finite())).then_some(inv)
}

/// The inverse, if it has one that the matrix editor can hold (entries within ±100).
fn inverse_to_edit(m: Mat2) -> Option<Mat2> {
    inverse(m).map(|v| v.map(|x| x as f32)).filter(|v| v.iter().all(|x| x.abs() <= 100.0)).map(cols)
}

/// A direction the map sends to 0, if it squashes the plane flat: across its longer
/// row (any direction at all for the zero matrix).
fn null_direction(m: Mat2) -> Option<Vec2> {
    if det(m) != 0.0 {
        return None;
    }
    let (r1, r2) = (m.row(0), m.row(1));
    let r = if r1.length_squared() >= r2.length_squared() { r1 } else { r2 };
    Some(if r.length_squared() > 0.0 { r.perp().normalize() } else { Vec2::Y })
}

/// Does the map turn the plane: does it turn every direction (complex eigenvalues),
/// or send the plane half way round (two negative eigenvalues)? Then sliding every
/// point straight to its place would cut across the turn and shrink everything on
/// the way: a quarter turn would halve the area half way through, and a half turn
/// would pass through a single point.
fn turns(m: Mat2) -> bool {
    let (det, trace) = (m.determinant(), m.x_axis.x + m.y_axis.y);
    det > 0.0 && trace < 2.0 * det.sqrt()
}

/// Where the plane is a share `s` of the way through the map `m`. Every point slides
/// straight to its place, as in the scene on linear maps, which is how a shear or a
/// stretch looks; a map that turns the plane turns it, by the turn in it (the angle
/// of its polar decomposition, m = R(θ) P with P symmetric), while the rest, P,
/// slides straight. Either way the plane never passes through flat unless the map
/// flips it over or flattens it itself.
fn along(m: Mat2, s: f32) -> Mat2 {
    // (the ends exactly, without the rounding of taking the turn apart)
    if s >= 1.0 {
        return m;
    }
    if !turns(m) {
        return Mat2::IDENTITY * (1.0 - s) + m * s;
    }
    let theta = (m.x_axis.y - m.y_axis.x).atan2(m.x_axis.x + m.y_axis.y);
    let rest = Mat2::from_angle(-theta) * m;
    Mat2::from_angle(theta * s) * (Mat2::IDENTITY * (1.0 - s) + rest * s)
}

/// The plane at `progress`: `first` acts from 0 to 1, then `second` from 1 to 2.
fn at(first: Mat2, second: Mat2, progress: f32) -> Mat2 {
    if progress <= 1.0 {
        along(first, smoothstep(progress))
    } else {
        along(second, smoothstep(progress - 1.0)) * first
    }
}

/// A number for the screen: as few digits as say it, a true minus sign, and no "−0".
fn num(v: f64) -> String {
    let a = v.abs();
    let text = if !v.is_finite() {
        return "?".into();
    } else if a < 5e-4 {
        return "0".into();
    } else if a >= 9999.5 {
        format!("{v:.1e}")
    } else if a >= 99.95 {
        format!("{v:.0}")
    } else {
        let t = if a >= 9.995 { format!("{v:.1}") } else { format!("{v:.3}") };
        t.trim_end_matches('0').trim_end_matches('.').to_string()
    };
    text.replace('-', "−")
}

/// A determinant for the screen: like `num`, but a small one that is not 0 says so.
fn num_det(v: f64) -> String {
    if v != 0.0 && v.abs() < 5e-4 {
        format!("{v:.1e}").replace('-', "−")
    } else {
        num(v)
    }
}

/// Is the canvas narrow and tall (a phone held upright)? The equation then goes
/// below the picture instead of over its corner.
fn narrow(view: Vec2) -> bool {
    view.x < 620.0 && view.y >= 560.0
}

/// On a narrow canvas: where the part kept for the equation and the key begins.
fn panel_top(view: Vec2) -> f32 {
    view.y - (view.y * 0.36).clamp(250.0, 330.0)
}

/// How long a line of the equation's panel is, in characters of words (as `Panel::of`
/// measures lines): a formula counts for as many as it is wide, not for its markup.
fn length(text: &str) -> usize {
    if text.contains('$') {
        (text_width(text, 1.0) / text_width("m", 1.0)).ceil() as usize
    } else {
        text.chars().count()
    }
}

/// Room kept around the picture for the names written beside things, logical pixels.
const NAME_ROOM: Vec2 = Vec2::new(80.0, 56.0);

/// Where the caption along the bottom begins (the canvas's foot, if it has none).
fn caption_top(view: Vec2) -> f32 {
    if view.y >= 300.0 {
        view.y - 34.0
    } else {
        view.y
    }
}

/// The part of the canvas the picture has to itself, as its two corners in logical
/// pixels: above the equation on a phone; otherwise below the equation or beside
/// it, whichever shows a picture `size` world units across the larger (on a phone
/// held sideways, beside it). The scale it can be shown at comes with it. `needs`
/// is what the equation has to hold (see `Panel::of`).
fn region(view: Vec2, inset: f32, size: Vec2, needs: (usize, usize)) -> ((Vec2, Vec2), f32) {
    let scale = |(lo, hi): (Vec2, Vec2)| {
        let room = (hi - lo - NAME_ROOM).max(Vec2::splat(60.0));
        (room.x / size.x.max(1e-6)).min(room.y / size.y.max(1e-6)).min(MOST_PER_UNIT).max(1e-3)
    };
    let with_scale = |r: (Vec2, Vec2)| (r, scale(r));
    if narrow(view) {
        return with_scale((Vec2::new(0.0, 52.0f32.max(inset + 6.0)), Vec2::new(view.x, panel_top(view))));
    }
    let (top, bottom) = (inset + 4.0, caption_top(view));
    let Some(p) = Panel::of(view, inset, needs.0, needs.1) else { return with_scale((Vec2::new(0.0, top), Vec2::new(view.x, bottom))) };
    let below = (Vec2::new(0.0, p.min.y + p.size.y), Vec2::new(view.x, bottom.max(p.min.y + p.size.y + 1.0)));
    let beside = (Vec2::new(0.0, top), Vec2::new(p.min.x.max(1.0), bottom));
    if scale(beside) > scale(below) * 1.05 {
        with_scale(beside)
    } else {
        with_scale(below)
    }
}

/// The names written beside things in the picture: each goes on the clearest of the
/// sides offered to it, out of the way of what has been written (or kept clear)
/// before it, and is moved in from the edges rather than cut off by them.
struct Names {
    /// The top and bottom of the part of the canvas names may be written in.
    band: (f32, f32),
    /// Rectangles already taken, logical pixels.
    taken: Vec<(Vec2, Vec2)>,
}

impl Names {
    /// Keep the dot of `radius` pixels at `at` clear of names.
    fn keep(&mut self, f: &Frame, at: Vec2, radius: f32) {
        if let Some(p) = on_screen(f, at.extend(0.0)) {
            self.taken.push((p - Vec2::splat(radius), p + Vec2::splat(radius)));
        }
    }

    /// The clearest of `sides` (an offset in pixels and the alignment there) of the
    /// point `at` for `text`, the earliest of equally clear ones: how much of the
    /// names already written it would cover (in square pixels), its rectangle, and
    /// where and how to write it.
    fn best(&self, f: &Frame, at: Vec2, sides: &[(Vec2, Align2)], text: &str, size: f32) -> Option<(f32, (Vec2, Vec2), Vec2, Align2)> {
        let p = on_screen(f, at.extend(0.0))?;
        let view = f.view_size();
        // (measured by a stand-in with every digit a 0, as wide, and the same while
        // the area changes as the maps play: the frame keeps what it has measured,
        // but a text it has not seen is set afresh)
        let stand_in: String = text.chars().map(|c| if c.is_ascii_digit() { '0' } else { c }).collect();
        let wh = Vec2::new(f.text_size(&stand_in, size).x + 4.0, size + 2.0);
        let mut best: Option<(f32, (Vec2, Vec2), Vec2, Align2)> = None;
        for &(off, align) in sides {
            let k = Vec2::new(align.x().to_factor(), align.y().to_factor());
            let want = p + off - wh * k;
            let min = Vec2::new(want.x.max(6.0).min(view.x - 6.0 - wh.x), want.y.max(self.band.0 + 2.0).min(self.band.1 - 2.0 - wh.y));
            let max = min + wh;
            let covered: f32 = self.taken.iter().map(|(a, b)| (max.min(*b) - min.max(*a)).max(Vec2::ZERO).element_product()).sum();
            if best.is_none_or(|(c, ..)| covered < c) {
                best = Some((covered, (min, max), min + wh * k, align));
            }
        }
        best
    }

    fn put(&mut self, f: &mut Frame, at: Vec2, sides: &[(Vec2, Align2)], text: String, size: f32, color: Color) {
        if let Some((_, rect, pos, align)) = self.best(f, at, sides, &text, size) {
            self.taken.push(rect);
            f.text_px(pos, text, size, color, align);
        }
    }
}

/// The four sides of a point to write a name on, the one facing away from the
/// origin along `out` first, its neighbours next, the side facing back last.
fn sides(out: Vec2) -> [(Vec2, Align2); 4] {
    let (right, up) = (out.x >= -1e-3, out.y >= -1e-3);
    let side = |r: bool, u: bool| {
        let align = match (r, u) {
            (true, true) => Align2::LEFT_BOTTOM,
            (false, true) => Align2::RIGHT_BOTTOM,
            (true, false) => Align2::LEFT_TOP,
            (false, false) => Align2::RIGHT_TOP,
        };
        (Vec2::new(if r { 7.0 } else { -7.0 }, if u { -5.0 } else { 5.0 }), align)
    };
    [side(right, up), side(!right, up), side(right, !up), side(!right, !up)]
}

/// The equation panel's layout: where it is, how large its text is, and where each of
/// the twelve numbers goes.
#[derive(Clone, Debug)]
struct Panel {
    narrow: bool,
    /// Top-left corner and size of its dark backing, logical pixels.
    min: Vec2,
    size: Vec2,
    /// Text size of the numbers; the rest of the text is a little smaller.
    text: f32,
    /// Width of one number's cell, and height of a row.
    cell: Vec2,
    /// Left edge of each matrix's left bracket, and the top of the matrices.
    left: [f32; 3],
    top: f32,
}

/// Inner margin of the panel, and the room between a bracket and its numbers.
const PAD: f32 = 12.0;
const BRACKET: f32 = 7.0;

impl Panel {
    /// The panel for a canvas of size `view` whose top `inset` pixels are under the
    /// application's buttons, holding numbers up to `widest` characters long and
    /// lines of words up to `line` characters. None if the canvas is too small to
    /// spare the room (the numbers are in the side panel as well).
    fn of(view: Vec2, inset: f32, widest: usize, line: usize) -> Option<Panel> {
        let narrow = narrow(view);
        // everything is measured in units of the text size, then scaled to fit
        let layout = |s: f32| {
            let cell = Vec2::new((widest as f32 * s * 0.56 + 8.0).max(2.4 * s), s + 10.0);
            let matrix = 2.0 * cell.x + 2.0 * BRACKET;
            let gap = 1.5 * s;
            let width = (3.0 * matrix + 2.0 * gap).max(text_width(&"m".repeat(line), s * 0.86)) + 2.0 * PAD;
            let height = 2.0 * PAD + (s * 0.86 + 10.0) + (s + 6.0) + 2.0 * cell.y + (s + 10.0) + (s * 0.86 + 10.0) + 2.0 * (s * 0.86 + 6.0) + (s * 0.78 + 6.0);
            (cell, matrix, gap, width, height)
        };
        let room = if narrow { Vec2::new(view.x, view.y - panel_top(view)) } else { Vec2::new(view.x - 32.0, view.y - inset.max(16.0) - 16.0) };
        // (on a narrow canvas the key goes under the equation, in the same region)
        let key = if narrow { KEY_LINES.len() as f32 * 17.0 + 8.0 } else { 0.0 };
        let fits = |s: f32| {
            let (_, _, _, w, h) = layout(s);
            w <= room.x && h + key <= room.y
        };
        // (smaller on a small canvas, where the picture needs the room)
        let mut s: f32 = if narrow { 14.0 } else { (view.y / 66.0).min(view.x / 72.0).clamp(10.0, 15.0) };
        while !fits(s) && s >= 9.0 {
            s *= 0.96;
        }
        if s.is_nan() || s < 9.0 {
            return None;
        }
        let (cell, matrix, gap, width, height) = layout(s);
        let size = if narrow { Vec2::new(view.x, view.y - panel_top(view)) } else { Vec2::new(width, height) };
        // (over the picture's corner only while it leaves most of the picture clear)
        if !narrow && width * height > 0.4 * view.x * view.y {
            return None;
        }
        let min = if narrow { Vec2::new(0.0, panel_top(view)) } else { Vec2::new(view.x - width - 16.0, inset.max(16.0)) };
        let x0 = min.x + (size.x - (3.0 * matrix + 2.0 * gap)) * 0.5;
        let left = [x0, x0 + matrix + gap, x0 + 2.0 * (matrix + gap)];
        let top = min.y + PAD + (s * 0.86 + 10.0) + (s + 6.0);
        Some(Panel { narrow, min, size, text: s, cell, left, top })
    }

    /// The rectangle of number (i, j) of matrix k (0 on the left, 2 the product).
    fn cell(&self, k: usize, i: usize, j: usize) -> (Vec2, Vec2) {
        let min = Vec2::new(self.left[k] + BRACKET + j as f32 * self.cell.x, self.top + i as f32 * self.cell.y);
        (min, min + self.cell)
    }

    /// Below the two rows of numbers.
    fn bottom(&self) -> f32 {
        self.top + 2.0 * self.cell.y
    }
}

/// What the picture shows that has no label of its own: under the picture on a
/// desktop, inside the equation's region on a phone. (The last only while the other
/// order is shown.)
const KEY_LINES: [&str; 3] = ["blue grid: the square grid, carried along", r"dashed: the paths of $\hat\imath$ and $\hat\jmath$", "purple: where the other order ends"];

pub struct Compose {
    a: Mat2,
    b: Mat2,
    /// Which map acts first: B (false: the product is AB) or A (true: BA).
    a_first: bool,
    /// How far the animation has got: 0 is the plane as it was, 1 the end of the
    /// first map, 2 the end of the second.
    progress: f32,
    playing: bool,
    /// Show where the other order ends.
    ghost: bool,
    /// The entry of the product whose sum is written out: (row, column).
    pick: (usize, usize),
    /// The canvas last drawn on, and how much of its top the application's buttons
    /// cover: on a phone they decide where the equation goes, and with it the picture.
    canvas: Vec2,
    inset: f32,
    new_view: bool,
    /// What `panel_needs` found last, and for what: the formulas it measures are the
    /// same frame after frame, and measuring them is not free.
    needs: std::cell::RefCell<(String, (usize, usize))>,
}

impl Compose {
    /// (Nothing here needs the graphics device, so the tests can make one too.)
    fn create() -> Self {
        let mut scene = Compose {
            a: Mat2::IDENTITY,
            b: Mat2::IDENTITY,
            a_first: false,
            progress: 0.0,
            playing: false,
            ghost: true,
            pick: (0, 0),
            canvas: Vec2::new(1080.0, 1000.0),
            inset: 0.0,
            new_view: false,
            needs: Default::default(),
        };
        scene.set_up(0);
        scene
    }

    /// Set up one of the experiments in `TRIALS`, and play it from the start.
    fn set_up(&mut self, k: usize) {
        (self.a, self.b) = trial(k);
        (self.a_first, self.ghost, self.pick) = (false, true, (0, 0));
        self.replay();
        self.new_view = true;
    }

    fn replay(&mut self) {
        self.progress = 0.0;
        self.playing = true;
    }

    /// Move the animation on by `dt` seconds of scene time.
    fn advance(&mut self, dt: f32) {
        if self.playing {
            self.progress += dt / STAGE_SECONDS;
            if self.progress >= 2.0 {
                self.progress = 2.0;
                self.playing = false;
            }
        }
    }

    /// The map that acts first and the one that acts second, with their names.
    fn first(&self) -> (Mat2, &'static str) {
        if self.a_first {
            (self.a, "A")
        } else {
            (self.b, "B")
        }
    }

    fn second(&self) -> (Mat2, &'static str) {
        if self.a_first {
            (self.b, "B")
        } else {
            (self.a, "A")
        }
    }

    /// The single map both make, and its name: the one on the right acts first.
    fn product(&self) -> Mat2 {
        self.second().0 * self.first().0
    }

    fn name(&self) -> String {
        format!("{}{}", self.second().1, self.first().1)
    }

    /// The same two maps in the other order.
    fn other(&self) -> Mat2 {
        self.first().0 * self.second().0
    }

    fn other_name(&self) -> String {
        format!("{}{}", self.first().1, self.second().1)
    }

    /// Do the two maps commute: does the other order end in the same place?
    fn commute(&self) -> bool {
        let (ab, ba) = (self.product(), self.other());
        let size = ab.to_cols_array().iter().chain(ba.to_cols_array().iter()).fold(1.0f32, |m, v| m.max(v.abs()));
        (ab - ba).to_cols_array().iter().all(|v| v.abs() <= 1e-5 * size)
    }

    /// The plane as drawn now.
    fn shown(&self) -> Mat2 {
        at(self.first().0, self.second().0, self.progress)
    }

    /// The two points that end on the same spot, if the maps squash the plane flat:
    /// far apart along a direction sent to 0, either side of a point beside the unit
    /// square (where they and their names keep clear of it).
    fn merging(&self) -> Option<(Vec2, Vec2)> {
        null_direction(self.product()).map(|n| (MERGE_AT - n * 0.9, MERGE_AT + n * 0.9))
    }

    /// The products of row i of the left matrix with column j of the right one, and
    /// their sum: the entry (i, j) of the product, as the panel writes it out.
    fn entry_sum(&self, i: usize, j: usize) -> ([(f64, f64); 2], f64) {
        let (left, right) = (self.second().0, self.first().0);
        let pairs = [0, 1].map(|k| (left.col(k)[i] as f64, right.col(j)[k] as f64));
        (pairs, pairs.iter().map(|(x, y)| x * y).sum())
    }

    /// What the panel says about the stage the animation is at.
    fn stage(&self) -> String {
        let (p, (_, first), (_, second)) = (self.progress, self.first(), self.second());
        if p < 1e-4 {
            "before either map".into()
        } else if p < 1.0 - 1e-4 {
            format!("step 1 of 2: ${first}$ acts")
        } else if p <= 1.0 + 1e-4 {
            format!(r"after ${first}$: its columns are where $\hat\imath$ and $\hat\jmath$ are")
        } else if p < 2.0 - 1e-4 {
            format!("step 2 of 2: ${second}$ acts on what ${first}$ made")
        } else {
            format!("${first}$, then ${second}$: one map, ${}$", self.name())
        }
    }

    /// The longest number the panel will write, and the longest line of words, in
    /// characters (see `length`).
    fn panel_needs(&self) -> (usize, usize) {
        let mats = [self.second().0, self.first().0, self.product()];
        let numbers: Vec<String> = mats.iter().flat_map(|m| m.to_cols_array()).map(|v| num(v as f64)).collect();
        let (i, j) = self.pick;
        let lines = [self.stage(), self.verdict(), self.head(i, j), format!("  {}", self.sum_line(i, j))];
        let asked = format!("{}\n{}", numbers.join(" "), lines.join("\n"));
        {
            let last = self.needs.borrow();
            if last.0 == asked {
                return last.1;
            }
        }
        // (a number set as mathematics is a little wider than its characters, and one
        // written as a power of ten much wider: it counts for as many as it is wide)
        let cell = |t: &String| t.chars().count().max((text_width(&format!("${t}$"), 1.0) / 0.56).ceil() as usize);
        let needs = (numbers.iter().map(cell).max().unwrap_or(1), lines.iter().map(|l| length(l)).max().unwrap_or(0));
        *self.needs.borrow_mut() = (asked, needs);
        needs
    }

    fn key_lines(&self) -> &'static [&'static str] {
        if self.ghost {
            &KEY_LINES
        } else {
            &KEY_LINES[..2]
        }
    }

    /// Whether the other order ends in the same place, in words.
    fn verdict(&self) -> String {
        if self.commute() {
            format!("${} = {}$: the other order ends in the same place", self.other_name(), self.name())
        } else {
            format!(r"${} \ne {}$: the other order ends elsewhere", self.other_name(), self.name())
        }
    }

    /// Which row and column make entry (i, j) of the product.
    fn head(&self, i: usize, j: usize) -> String {
        format!("$({})_{{{}{}}}$ is row {} of ${}$ times column {} of ${}$", self.name(), i + 1, j + 1, i + 1, self.second().1, j + 1, self.first().1)
    }

    /// "= a × b + c × d = e" for entry (i, j).
    fn sum_line(&self, i: usize, j: usize) -> String {
        let ([(a, b), (c, d)], e) = self.entry_sum(i, j);
        let factor = |v: f64| if v < 0.0 && num(v) != "0" { format!("({})", num(v)) } else { num(v) };
        format!(r"$= {} \times {} + {} \times {} = {}$", num(a), factor(b), factor(c), factor(d), num(e))
    }

    /// Everything the animation draws, from the first frame to the last: the unit
    /// square at every stage, and where the other order ends (whether it is shown or
    /// not, so that showing it does not move the view).
    fn bounds(&self) -> (Vec2, Vec2) {
        let (first, second) = (self.first().0, self.second().0);
        let red = self.merging();
        let mut points = vec![Vec2::ZERO, Vec2::ONE];
        let mut square = |m: Mat2| points.extend([m.x_axis, m.y_axis, m.x_axis + m.y_axis]);
        for i in 0..=48 {
            square(at(first, second, i as f32 / 24.0));
        }
        square(self.other());
        if let Some((p, q)) = red {
            points.extend((0..=48).flat_map(|i| [at(first, second, i as f32 / 24.0) * p, at(first, second, i as f32 / 24.0) * q]));
        }
        points.iter().fold((Vec2::ZERO, Vec2::ZERO), |(lo, hi), p| (lo.min(*p), hi.max(*p)))
    }

    /// The view that fits the whole animation into the part of the canvas the
    /// picture has to itself, with room around it for the names of things.
    fn the_view(&self) -> View {
        let (lo, hi) = self.bounds();
        let size = (hi - lo).max(Vec2::splat(2.0));
        let ((min, max), per_unit) = region(self.canvas, self.inset, size, self.panel_needs());
        // the middle of the region, which is not the canvas's, shows the middle of
        // the picture (screen y runs down, world y up)
        let off = ((min + max) * 0.5 - self.canvas * 0.5) / per_unit;
        let centre = (lo + hi) * 0.5;
        View::Fit { center: Vec2::new(centre.x - off.x, centre.y + off.y), half: self.canvas.max(Vec2::ONE) * (0.5 / per_unit) }
    }

    /// The equation [second][first] = [product] on screen, with the determinants
    /// under it and one entry of the product worked out.
    fn equation(&self, f: &mut Frame, p: &Panel) {
        let s = p.text;
        let small = s * 0.86;
        hud_panel(f, p.min, p.min + p.size);
        let mats = [self.second(), self.first(), (self.product(), "")];
        let names = [mats[0].1.to_string(), mats[1].1.to_string(), self.name()].map(|n| format!("${n}$"));
        let x = p.min.x + PAD;
        let mut y = p.min.y + PAD;
        f.text_px(Vec2::new(x, y), self.stage(), small, pal::TEXT, Align2::LEFT_TOP);
        y += small + 10.0;

        // which matrices have acted so far: they are bright, the rest dim
        let acted = [self.progress > 1.0 + 1e-4, self.progress > 1e-4, self.progress >= 2.0 - 1e-4];
        let mid = |k: usize| p.left[k] + BRACKET + p.cell.x;
        for k in 0..3 {
            let c = if acted[k] { pal::TEXT } else { pal::DIM };
            f.text_px(Vec2::new(mid(k), y), names[k].as_str(), s, c, Align2::CENTER_TOP);
            // brackets
            let (l, r) = (p.left[k], p.left[k] + 2.0 * BRACKET + 2.0 * p.cell.x);
            let (t, b) = (p.top + 1.0, p.bottom() - 1.0);
            for (edge, inward) in [(l, 5.0), (r, -5.0)] {
                hud_line(f, Vec2::new(edge, t), Vec2::new(edge, b), 1.4, c);
                hud_line(f, Vec2::new(edge, t), Vec2::new(edge + inward, t), 1.4, c);
                hud_line(f, Vec2::new(edge, b), Vec2::new(edge + inward, b), 1.4, c);
            }
            // the numbers: the columns of the first map and of the product are where
            // î and ĵ land, and have their colours
            let m = mats[k].0;
            for i in 0..2 {
                for j in 0..2 {
                    let colour = if k == 0 {
                        c
                    } else if acted[k] {
                        [pal::PINK, pal::GREEN][j]
                    } else {
                        [pal::PINK, pal::GREEN][j].mix(pal::DIM, 0.6)
                    };
                    let (lo, hi) = p.cell(k, i, j);
                    f.text_px((lo + hi) * 0.5, format!("${}$", num(m.col(j)[i] as f64)), s, colour, Align2::CENTER_CENTER);
                }
            }
        }
        // the row and the column that make the picked entry
        let (i, j) = self.pick;
        let pad = Vec2::splat(2.0);
        let ring = |f: &mut Frame, (lo, hi): (Vec2, Vec2)| hud_outline(f, lo + pad, hi - pad, 1.4, pal::CYAN);
        ring(f, (p.cell(0, i, 0).0, p.cell(0, i, 1).1));
        ring(f, (p.cell(1, 0, j).0, p.cell(1, 1, j).1));
        ring(f, p.cell(2, i, j));
        let gap_mid = |k: usize| (p.left[k] + 2.0 * BRACKET + 2.0 * p.cell.x + p.left[k + 1]) * 0.5;
        let row = p.bottom() + 10.0;
        f.text_px(Vec2::new(gap_mid(1), p.top + p.cell.y), "$=$", s, pal::TEXT, Align2::CENTER_CENTER);
        for k in 0..3 {
            let d = det(mats[k].0);
            f.text_px(Vec2::new(mid(k), row), format!(r"$\det {}$", num_det(d)), small, pal::YELLOW.mix(pal::DIM, if acted[k] { 0.0 } else { 0.5 }), Align2::CENTER_TOP);
        }
        f.text_px(Vec2::new(gap_mid(0), row + small * 0.5), r"$\times$", s * 1.1, pal::TEXT, Align2::CENTER_CENTER);
        f.text_px(Vec2::new(gap_mid(1), row + small * 0.5), "$=$", s, pal::TEXT, Align2::CENTER_CENTER);
        y = row + small + 10.0;
        f.text_px(Vec2::new(x, y), self.verdict(), small, pal::PURPLE, Align2::LEFT_TOP);
        y += small + 10.0;
        f.text_px(Vec2::new(x, y), self.head(i, j), small, pal::CYAN, Align2::LEFT_TOP);
        y += small + 6.0;
        f.text_px(Vec2::new(x + 10.0, y), self.sum_line(i, j), small, pal::TEXT, Align2::LEFT_TOP);
        y += small + 6.0;
        f.text_px(Vec2::new(x, y), format!("click an entry of ${}$ to pick another", self.name()), s * 0.78, pal::DIM, Align2::LEFT_TOP);
        if p.narrow {
            y += s * 0.78 + 14.0;
            for line in self.key_lines() {
                f.text_px(Vec2::new(x, y), *line, 12.0, pal::DIM, Align2::LEFT_TOP);
                y += 17.0;
            }
        }
    }
}

/// The quick choices for one matrix, and the inverse of the other one (when it has
/// one that the editor can hold). Returns true if one was taken.
fn presets(ui: &mut egui::Ui, m: &mut Mat2, other: Mat2, other_name: &str) -> bool {
    let mut taken = false;
    ui.horizontal_wrapped(|ui| {
        for (name, c) in PRESETS {
            if ui.button(ui::label_text(ui, name, 12.5, None, false)).clicked() {
                *m = cols(c);
                taken = true;
            }
        }
        let inv = inverse_to_edit(other);
        let button = egui::Button::new(ui::label_text(ui, &format!("${other_name}^{{-1}}$"), 12.5, None, false));
        if ui.add_enabled(inv.is_some(), button).clicked() {
            if let Some(inv) = inv {
                *m = inv;
                taken = true;
            }
        }
    });
    if inverse(other).is_none() {
        ui::note(ui, &format!(r"${other_name}$ squashes the plane flat ($\det {other_name} = 0$): it has no inverse, so there is no ${other_name}^{{-1}}$ to choose."));
    } else if inverse_to_edit(other).is_none() {
        ui::note(ui, &format!(r"${other_name}$ is so nearly flat that ${other_name}^{{-1}}$ has entries beyond $\pm 100$, more than the numbers here can hold."));
    }
    taken
}

/// The inverse for the list of numbers: a matrix, set on a line of its own, or why
/// there is none.
fn inverse_rows(ui: &mut egui::Ui, m: Mat2, name: &str) {
    match inverse(m) {
        Some(v) => {
            let [a, b, c, d] = v.map(num);
            ui::equation(ui, &format!(r"{name}^{{-1}} = \begin{{bmatrix}} {a} & {c} \\ {b} & {d} \end{{bmatrix}}"));
        }
        None => ui::readout(ui, &format!("${name}^{{-1}}$"), format!(r"none: $\det {name} = 0$")),
    }
}

impl Scene for Compose {
    fn view(&self) -> View {
        self.the_view()
    }

    fn take_view(&mut self) -> Option<View> {
        std::mem::take(&mut self.new_view).then(|| self.the_view())
    }

    fn animated(&self) -> bool {
        self.playing
    }

    fn ui(&mut self, ui: &mut egui::Ui, _: &Gfx) {
        ui::note(ui, r"Doing one linear map $B$ and then another, $A$, is again a single linear map. Its matrix is the product $AB$: the map on the right acts first. Its columns are where $\textcolor{pink}{\hat\imath}$ and $\textcolor{green}{\hat\jmath}$ end up, which is $A$ applied to the columns of $B$. Watch the plane go through $B$ and then $A$; the equation drawn with the picture works out one entry of $AB$.");

        ui::section(ui, "Things to try");
        for (k, trial) in TRIALS.iter().enumerate() {
            if ui::experiment(ui, trial.0, trial.1) {
                self.set_up(k);
            }
        }

        ui::section(ui, "Order");
        let mut order = self.a_first as usize;
        if ui::choice(ui, &["$B$, then $A$ ($AB$)", "$A$, then $B$ ($BA$)"], &mut order) {
            self.a_first = order == 1;
            self.replay();
            self.new_view = true;
        }
        ui::section(ui, "Animation");
        if ui.button("replay").clicked() {
            self.replay();
        }
        if ui::slider(ui, "progress", &mut self.progress, 0.0..=2.0) {
            self.playing = false;
        }
        ui::note(ui, "0 is the plane as it was, 1 after the first map, 2 after both.");

        for (name, other_name) in [("A", "B"), ("B", "A")] {
            ui::section(ui, &format!("Matrix ${name}$"));
            let (mine, other) = if name == "A" { (&mut self.a, self.b) } else { (&mut self.b, self.a) };
            let mut fresh = presets(ui, mine, other, other_name);
            fresh |= ui.push_id(name, |ui| mat2_edit(ui, mine)).inner;
            if fresh {
                self.replay();
                self.new_view = true;
            }
        }
        ui::note(ui, "turn: a quarter turn; shear: $x$ slides by $y$; stretch: $x$ doubles; flip: a mirror in the $y$ axis; squash: onto the $x$ axis; $I$: nothing moves. Type a number to play the maps from the start; drag one sideways and the picture follows it. The dots halfway along the dashed paths can be dragged too: they are the first map's columns.");
        ui::section(ui, "Show");
        ui.checkbox(&mut self.ghost, "where the other order ends");

        ui::section(ui, "Reading the picture");
        ui::key(
            ui,
            &[
                (pal::BLUE, "blue grid: the square grid, carried along"),
                (pal::YELLOW, "yellow: the unit square, carried along; its area is the determinant so far"),
                (pal::WHITE, "white F: painted on the square, so that turns and flips can be told apart"),
                (pal::PINK, r"pink: $\hat\imath = (1, 0)$, carried along; its dot halfway is the first column of the first map, its tip at the end the first column of the product"),
                (pal::GREEN, r"green: $\hat\jmath = (0, 1)$, the same for the second columns"),
                (pal::PURPLE, "purple: where the square and the F end in the other order"),
                (pal::RED, "red dots: two points that end on the same spot (only when the plane is squashed flat)"),
                (pal::CYAN, "cyan boxes: the row and the column whose products add up to one entry"),
            ],
        );
        ui::section(ui, "Words");
        ui::term(ui, "Linear map:", r"a way of moving every point of the plane that keeps the origin in place and lines straight, parallel and evenly spaced. Where $\hat\imath$ and $\hat\jmath$ go, the columns of its matrix, settles where everything goes.");
        ui::term(ui, "Product $AB$:", r#"the matrix of "$B$, then $A$". Its entry in row $i$ and column $j$ is row $i$ of $A$ times column $j$ of $B$: multiply the pairs of numbers and add."#);
        ui::term(ui, "Commute:", "two maps commute if their order does not matter: $AB = BA$.");
        ui::term(ui, "Identity $I$:", r"the map that leaves every point where it is. Its columns are $\hat\imath$ and $\hat\jmath$.");
        ui::term(ui, "Inverse $A^{-1}$:", r"the map that undoes $A$: $A^{-1} A = A A^{-1} = I$. It exists only if $\det A \ne 0$, and $\det A^{-1} = 1/\det A$, so the flatter $A$ squashes, the larger its inverse.");
        ui::term(ui, "Determinant:", r"the factor by which a map multiplies every area, negative if it flips the plane over. $\det AB = \det A \times \det B$.");

        ui::section(ui, "Numbers");
        let name = self.name();
        // (a determinant near 0 is written as a power of ten: a formula)
        ui::readout(ui, r"$\det A$", format!("${}$", num_det(det(self.a))));
        ui::readout(ui, r"$\det B$", format!("${}$", num_det(det(self.b))));
        ui::readout(ui, &format!(r"$\det {name}$"), format!("${}$", num_det(det(self.product()))));
        ui::readout(ui, "$AB = BA$?", if self.commute() { "yes: they commute" } else { "no" });
        inverse_rows(ui, self.a, "A");
        inverse_rows(ui, self.b, "B");
        ui::see_also(ui, &[("linmap2", "one map at a time"), ("svd", "any map as three simple ones in a row"), ("rotations", "turns of space, where the order matters too"), ("fold", "a network's layers: maps one after another, with a bend between")]);
    }

    fn frame(&mut self, f: &mut Frame) {
        self.advance(f.dt);
        let labels_before = f.dl.labels.len();

        // the equation's place follows the canvas, and the picture's place follows it
        // (but not into a sliver: on a phone the controls, when open, leave a strip of
        // the picture beside them, and closing them should find the view as it was)
        let view = f.view_size();
        let changed = (view - self.canvas).abs().max_element() > 24.0 || narrow(view) != narrow(self.canvas) || (f.top_inset - self.inset).abs() > 1.0;
        if changed && view.x >= 200.0 && view.y >= 150.0 {
            (self.canvas, self.inset) = (view, f.top_inset);
            self.new_view = true;
        }
        let (widest, line) = self.panel_needs();
        let panel = Panel::of(view, f.top_inset, widest, line);
        // names are written between the application's buttons and the caption (or
        // the equation, on a phone), and steer clear of the equation elsewhere
        let band = (f.top_inset + 2.0, if narrow(view) { panel_top(view) } else { caption_top(view) });
        let wpp = f.cam.world_per_px(Vec3::ZERO) * f.px;

        // a click on an entry of the product picks the entry that is worked out
        if let (Some(p), Some(pointer)) = (&panel, f.ptr.pos.map(|q| q / f.px)) {
            if pointer.cmpge(p.min).all() && pointer.cmple(p.min + p.size).all() {
                f.handles.claim_clicks();
                if f.clicked_canvas() {
                    for (i, j) in [(0, 0), (0, 1), (1, 0), (1, 1)] {
                        let (lo, hi) = p.cell(2, i, j);
                        if pointer.cmpge(lo).all() && pointer.cmple(hi).all() {
                            self.pick = (i, j);
                        }
                    }
                }
            }
        }

        let ((first, first_name), (second, _)) = (self.first(), self.second());
        let m = self.shown();
        warped_grid(f, m, 9, pal::BLUE);

        // the paths of î and ĵ through both maps, faint
        for (e, c) in [(Vec2::X, pal::PINK), (Vec2::Y, pal::GREEN)] {
            f.dashed_path((0..=48).map(|i| at(first, second, i as f32 / 24.0) * e), 1.4, c.a(0.55));
        }
        // where the other order ends: the square and the F in outline
        let other = self.other();
        if self.ghost {
            let sq = [Vec2::ZERO, other.x_axis, other.x_axis + other.y_axis, other.y_axis];
            f.dashed_path(sq.iter().chain(&sq[..1]).copied(), 1.6, pal::PURPLE);
            for bar in F_BARS {
                let [p0, p1, p2, p3] = bar.map(|p| (other * p).extend(0.0));
                f.dl.tris_alpha.quad(p0, p1, p2, p3, pal::PURPLE.a(0.14));
            }
            f.dl.lines.closed(&F_OUTLINE.map(|p| (other * p).extend(0.0)), 1.1, pal::PURPLE.a(0.75));
        }

        // the unit square with its F, as it is now
        let [o, i_hat, d, j_hat] = [Vec2::ZERO, m.x_axis, m.x_axis + m.y_axis, m.y_axis].map(|p| p.extend(0.0));
        f.dl.tris_alpha.quad(o, i_hat, d, j_hat, pal::YELLOW.a(0.2));
        f.dl.lines.closed(&[o, i_hat, d, j_hat], 1.5, pal::YELLOW.a(0.8));
        for bar in F_BARS {
            let [p0, p1, p2, p3] = bar.map(|p| (m * p).extend(0.0));
            f.dl.tris_alpha.quad(p0, p1, p2, p3, pal::WHITE.a(0.82));
        }

        // the arrows, and halfway along their paths the first map's columns, which
        // are handles: dragging one moves that column of the first map
        for (k, c) in [pal::PINK, pal::GREEN].into_iter().enumerate() {
            f.arrow(Vec2::ZERO, m.col(k), 3.2, c);
        }
        let mut columns = [first.x_axis, first.y_axis];
        for (k, c) in [pal::PINK, pal::GREEN].into_iter().enumerate() {
            if f.drag2(1 + k as u32, &mut columns[k], c) {
                let target = if self.a_first { &mut self.a } else { &mut self.b };
                *target.col_mut(k) = columns[k].max(Vec2::splat(-100.0)).min(Vec2::splat(100.0));
            }
        }

        // two points that end on the same spot, when the plane is squashed flat: with
        // their paths, so that where they started stays in view
        let red = self.merging().map(|(p, q)| (p, q, m * p, m * q));
        if let Some((p, q, sp, sq)) = red {
            for start in [p, q] {
                f.dashed_path((0..=48).map(|i| at(first, second, i as f32 / 24.0) * start), 1.3, pal::RED.a(0.6));
            }
            f.dot(sp, 10.0, pal::RED);
            f.dot(sq, 10.0, pal::RED);
        }

        // the names, most important first: each keeps clear of those before it
        let mut names = Names { band, taken: panel.iter().filter(|p| !p.narrow).map(|p| (p.min, p.min + p.size)).collect() };
        for dot in [first.x_axis, first.y_axis, m.x_axis, m.y_axis] {
            names.keep(f, dot, 7.0);
        }
        if let Some((_, _, sp, sq)) = red {
            names.keep(f, sp, 6.0);
            names.keep(f, sq, 6.0);
        }
        // the arrows: î and ĵ while they are carried, and named for what they have
        // become where the animation rests between the maps and after both
        let p = self.progress;
        let prefix = if (p - 1.0).abs() <= 1e-4 {
            first_name.to_string()
        } else if p >= 2.0 - 1e-4 {
            self.name()
        } else {
            String::new()
        };
        for (k, (hat, c)) in [(r"\hat\imath", pal::PINK), (r"\hat\jmath", pal::GREEN)].into_iter().enumerate() {
            names.put(f, m.col(k), &sides(m.col(k)), format!("${prefix}{hat}$"), 16.0, c);
        }
        // the area, in the corner the F leaves empty (under a square too thin to
        // hold it)
        let longest = [m.x_axis, m.y_axis, m.x_axis + m.y_axis, m.x_axis - m.y_axis].iter().fold(0.0f32, |l, v| l.max(v.length()));
        let thin = m.determinant().abs() / longest.max(1e-9) < 20.0 * wpp;
        let area_sides: &[(Vec2, Align2)] = if thin { &[(Vec2::new(0.0, 8.0), Align2::CENTER_TOP), (Vec2::new(0.0, -8.0), Align2::CENTER_BOTTOM)] } else { &[(Vec2::ZERO, Align2::CENTER_CENTER)] };
        names.put(f, m * AREA_AT, area_sides, format!("area ${}$", num_det(det(m))), 13.0, pal::YELLOW);
        // the first map's columns, unless the arrows' own names are on them
        for (k, (hat, c)) in [(r"\hat\imath", pal::PINK), (r"\hat\jmath", pal::GREEN)].into_iter().enumerate() {
            let col = first.col(k);
            if (m.col(k) - col).length() > 14.0 * wpp {
                names.put(f, col, &sides(col), format!("${first_name}{hat}$"), 13.0, c.mix(pal::DIM, 0.3));
            }
        }
        if let Some((p, q, sp, sq)) = red {
            let left_right = [(Vec2::new(-8.0, 0.0), Align2::RIGHT_CENTER), (Vec2::new(8.0, 0.0), Align2::LEFT_CENTER)];
            if (sp - sq).length() < 8.0 * wpp {
                // (above first: the numbers of the x axis are written under it)
                let around = [(Vec2::new(-8.0, -7.0), Align2::RIGHT_BOTTOM), (Vec2::new(8.0, -7.0), Align2::LEFT_BOTTOM), (Vec2::new(-8.0, 7.0), Align2::RIGHT_TOP), (Vec2::new(8.0, 7.0), Align2::LEFT_TOP)];
                names.put(f, sp, &around, "$p$ and $q$: one spot".into(), 13.0, pal::RED);
            } else {
                names.put(f, sp, &[left_right[1], left_right[0]], "$p$".into(), 14.0, pal::RED);
                names.put(f, sq, &[left_right[1], left_right[0]], "$q$".into(), 14.0, pal::RED);
            }
            // where they started, once they have left
            for (start, now, name) in [(p, sp, "$p$"), (q, sq, "$q$")] {
                if (now - start).length() > 14.0 * wpp {
                    names.put(f, start, &left_right, name.into(), 13.0, pal::RED.mix(pal::DIM, 0.3));
                }
            }
        }
        if self.ghost && !self.commute() {
            // at whichever corner of its square, furthest from the square drawn now
            // first, has room
            let centre = (m.x_axis + m.y_axis) * 0.5;
            let mut corners = [other.x_axis, other.x_axis + other.y_axis, other.y_axis];
            corners.sort_by(|a, b| b.distance(centre).total_cmp(&a.distance(centre)));
            let text = format!("${}$: the other order", self.other_name());
            let size = 13.0;
            let covered = |corner: Vec2| names.best(f, corner, &sides(corner), &text, size).map_or(f32::MAX, |b| b.0);
            let mut best = corners[0];
            for corner in corners {
                if covered(corner) < covered(best) {
                    best = corner;
                }
            }
            names.put(f, best, &sides(best), text, size, pal::PURPLE);
        }
        f.axes2();

        // nothing written into the picture may show through the equation, or under
        // the application's buttons
        let (lo, hi) = panel.as_ref().map_or((Vec2::splat(-1.0), Vec2::splat(-1.0)), |p| (p.min - Vec2::splat(4.0), p.min + p.size + Vec2::splat(4.0)));
        let (mut k, inset) = (0, f.top_inset);
        f.dl.labels.retain(|l| {
            k += 1;
            k <= labels_before || !(l.pos.y < inset || l.pos.cmpge(lo).all() && l.pos.cmple(hi).all())
        });
        if let Some(p) = &panel {
            self.equation(f, p);
        }
        if !panel.as_ref().is_some_and(|p| p.narrow) {
            bottom_caption(f, self.key_lines(), 12.0);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// An experiment as its button sets it up, played to the end at 60 frames a second.
    fn run(k: usize) -> Compose {
        let mut scene = Compose::create();
        scene.set_up(k);
        assert!(scene.playing && scene.progress == 0.0);
        for _ in 0..1000 {
            scene.advance(1.0 / 60.0);
        }
        assert!(!scene.playing && scene.progress == 2.0);
        scene
    }

    /// The same experiment, stopped at `progress`.
    fn at_progress(k: usize, progress: f32) -> Compose {
        let mut scene = run(k);
        scene.progress = progress;
        scene
    }

    fn close(a: Mat2, b: Mat2, tol: f32) -> bool {
        (a - b).to_cols_array().iter().all(|v| v.abs() <= tol)
    }

    /// The angle a vector is turned through by `m`, in degrees, and how much it is stretched.
    fn turn_and_stretch(m: Mat2, v: Vec2) -> (f32, f32) {
        (v.angle_to(m * v).to_degrees(), (m * v).length() / v.length())
    }

    /// The signed area of a polygon, positive if its corners run anticlockwise.
    fn area(corners: &[Vec2]) -> f32 {
        (0..corners.len()).map(|i| corners[i].perp_dot(corners[(i + 1) % corners.len()])).sum::<f32>() * 0.5
    }

    fn random(rng: &mut Rng, scale: f32) -> Mat2 {
        Mat2::from_cols(Vec2::new(rng.normal(), rng.normal()), Vec2::new(rng.normal(), rng.normal())) * scale
    }

    /// Unit vectors every few degrees round the circle.
    fn directions() -> impl Iterator<Item = Vec2> {
        (0..72).map(|d| Vec2::from_angle((d as f32 * 5.0 + 1.0).to_radians()))
    }

    /// The captions of the "things to try", held to account.
    #[test]
    fn the_experiments_show_what_their_captions_say() {
        // a turn, then a shear: B is a quarter turn, A slides every point sideways by
        // its height; the F's long stroke (bottom to top) ends flat and pointing
        // left; in the other order the shear tilts it to (1, 1) and the turn carries
        // that to 45 degrees from flat; AB is not BA
        let s = run(0);
        assert!(directions().all(|v| {
            let (turn, stretch) = turn_and_stretch(s.b, v);
            (turn - 90.0).abs() < 1e-4 && (stretch - 1.0).abs() < 1e-6
        }));
        for (x, y) in [(0.3, 0.8), (-1.0, 2.0), (2.0, -1.5)] {
            assert_eq!(s.a * Vec2::new(x, y), Vec2::new(x + y, y));
        }
        let stroke = F_BARS[0][3] - F_BARS[0][0];
        assert!(stroke.x == 0.0 && stroke.y > 0.7, "the long stroke is upright to begin with");
        let ends = s.product() * stroke;
        assert!(ends.y.abs() < 1e-6 && ends.x < -0.7, "{ends}");
        assert!((s.a * stroke).normalize().distance(Vec2::new(1.0, 1.0).normalize()) < 1e-6);
        let other = s.other() * stroke;
        assert!((other.y.atan2(-other.x).to_degrees() - 45.0).abs() < 1e-4, "{other}");
        assert!(!s.commute() && s.verdict().starts_with(r"$BA \ne AB$"));
        assert!(close(s.shown(), s.product(), 0.0) && s.name() == "AB" && s.other_name() == "BA");

        // two turns: by 30 and 60 degrees; the other order ends exactly on the result,
        // a turn by 90 degrees; stretches along the axes commute too
        let s = run(1);
        for (m, want) in [(s.b, 30.0), (s.a, 60.0), (s.product(), 90.0), (s.other(), 90.0)] {
            assert!(
                directions().all(|v| {
                    let (turn, stretch) = turn_and_stretch(m, v);
                    (turn - want).abs() < 1e-3 && (stretch - 1.0).abs() < 1e-6
                }),
                "{m}"
            );
        }
        assert!(close(s.product(), s.other(), 1e-6) && s.commute() && s.verdict().starts_with("$BA = AB$"));
        let mut rng = Rng::new(5);
        for _ in 0..100 {
            let (p, q) = (Mat2::from_diagonal(Vec2::new(rng.normal(), rng.normal())), Mat2::from_diagonal(Vec2::new(rng.normal(), rng.normal())));
            assert!(close(p * q, q * p, 0.0));
        }

        // areas multiply: B doubles areas, A flips the plane and multiplies them by
        // 1.5; the square's area reads 1, then 2, then -3; the F comes out mirrored;
        // the other order ends elsewhere, with area -3 as well
        let s = run(2);
        assert_eq!([det(s.b), det(s.a)], [2.0, -1.5]);
        for (progress, want) in [(0.0, "1"), (1.0, "2"), (2.0, "−3")] {
            let m = at_progress(2, progress).shown();
            let square = area(&[Vec2::ZERO, m.x_axis, m.x_axis + m.y_axis, m.y_axis]);
            assert_eq!(num_det(det(m)), want);
            assert!((square as f64 - det(m)).abs() < 1e-6, "the yellow square's area at {progress} is {square}");
        }
        assert!(area(&F_OUTLINE) > 0.0 && area(&F_OUTLINE.map(|p| s.product() * p)) < 0.0, "the F is mirrored");
        assert!(!close(s.product(), s.other(), 0.1) && det(s.other()) == -3.0);

        // a map and its inverse: B turns by 45 degrees and enlarges (det 2); A is
        // B's inverse, the very matrix the B⁻¹ button gives; everything returns;
        // AB = BA = I; det A reads 1/2
        let s = run(3);
        assert!(directions().all(|v| {
            let (turn, stretch) = turn_and_stretch(s.b, v);
            (turn - 45.0).abs() < 1e-4 && stretch > 1.0
        }));
        assert_eq!(det(s.b), 2.0);
        assert_eq!(inverse_to_edit(s.b), Some(s.a));
        assert!(s.product() == Mat2::IDENTITY && s.other() == Mat2::IDENTITY && s.shown() == Mat2::IDENTITY);
        assert_eq!(num_det(det(s.a)), "0.5");
        assert!(s.commute() && s.merging().is_none());

        // a squash: B drops every point straight down onto the x axis (det 0); p and
        // q start one above the other and meet at the end of B; tripling heights
        // moves nothing after that; no map parts them; no inverse; det AB = 0 for any A
        let s = run(4);
        for (x, y) in [(0.3, 0.8), (-1.0, 2.0), (2.0, -1.5)] {
            assert_eq!(s.b * Vec2::new(x, y), Vec2::new(x, 0.0));
            assert_eq!(s.a * Vec2::new(x, y), Vec2::new(x, 3.0 * y));
        }
        assert!(det(s.b) == 0.0 && num_det(det(s.b)) == "0" && inverse(s.b).is_none() && inverse_to_edit(s.b).is_none());
        let (p, q) = s.merging().expect("the plane is squashed");
        assert!(p.x == q.x && q.y - p.y > 1.5, "{p} {q}");
        let start = at_progress(4, 0.0).shown();
        assert!((start * p - start * q).length() > 1.5);
        let landed = at_progress(4, 1.0).shown();
        assert_eq!(landed * p, landed * q);
        for progress in [1.0, 1.2, 1.5, 1.9, 2.0] {
            assert_eq!(at_progress(4, progress).shown(), s.b, "something moved at {progress}");
        }
        let mut rng = Rng::new(9);
        for _ in 0..100 {
            let any = random(&mut rng, 2.0);
            assert_eq!(any * s.b * p, any * s.b * q);
            assert_eq!(det(any * s.b), 0.0);
        }
    }

    /// The three maps on screen are what the panel says they are: the plane goes
    /// from I through the first map to the product, the columns of the product are
    /// the second map applied to the first map's columns, each entry is a row times
    /// a column, and the determinants multiply. In either order.
    #[test]
    fn one_map_after_another_is_the_product() {
        let mut rng = Rng::new(11);
        for n in 0..300 {
            let mut scene = Compose::create();
            (scene.a, scene.b, scene.a_first) = (random(&mut rng, 1.5), random(&mut rng, 1.5), n % 2 == 1);
            let ((first, f), (second, g)) = (scene.first(), scene.second());
            assert_eq!(scene.name(), format!("{g}{f}"));
            let product = scene.product();
            assert_eq!(product, if scene.a_first { scene.b * scene.a } else { scene.a * scene.b });
            let size = 1.0 + product.to_cols_array().iter().fold(0.0f32, |m, v| m.max(v.abs()));
            for (progress, want) in [(0.0, Mat2::IDENTITY), (1.0, first), (2.0, product)] {
                scene.progress = progress;
                assert!(close(scene.shown(), want, 1e-5 * size), "{progress}: {} against {want}", scene.shown());
            }
            for k in 0..2 {
                assert!((product.col(k) - second * first.col(k)).length() < 1e-5 * size);
            }
            for (i, j) in [(0, 0), (0, 1), (1, 0), (1, 1)] {
                let (pairs, sum) = scene.entry_sum(i, j);
                assert!((sum - product.col(j)[i] as f64).abs() < 1e-5 * size as f64);
                assert_eq!(pairs.map(|p| p.0), [second.col(0)[i], second.col(1)[i]].map(f64::from));
                assert_eq!(pairs.map(|p| p.1), [first.col(j)[0], first.col(j)[1]].map(f64::from));
                assert!(scene.sum_line(i, j).ends_with(&format!("= {}$", num(sum))));
            }
            let (da, db, dp) = (det(scene.a), det(scene.b), det(product));
            assert!((dp - da * db).abs() < 1e-5 * (1.0 + dp.abs()), "{dp} against {da} × {db}");
            // the words in the panel follow the order
            assert!(scene.head(1, 0).contains(&format!("row 2 of ${g}$ times column 1 of ${f}$")));
        }
    }

    /// A map with a turn in it is played as a turn (every stage a rotation, for a
    /// pure one); a shear or a stretch slides every point straight to its place;
    /// the plane never passes through flat unless the map flips it or flattens it;
    /// and the picture never swings out further than where each map starts and ends.
    #[test]
    fn turns_are_played_as_turns_and_slides_as_slides() {
        for degrees in [10.0f32, 30.0, 60.0, 90.0, 135.0, 179.0, 180.0, -90.0, -150.0] {
            let r = Mat2::from_angle(degrees.to_radians());
            for i in 0..=20 {
                let m = along(r, i as f32 / 20.0);
                assert!((m.x_axis.length() - 1.0).abs() < 1e-5 && m.x_axis.dot(m.y_axis).abs() < 1e-5 && (m.determinant() - 1.0).abs() < 1e-5, "{degrees}°: {m}");
            }
        }
        for m in [cols(SHEAR), cols([2.0, 0.0, 0.0, 1.0]), cols([1.0, 0.0, 0.0, 3.0]), cols([-1.0, 0.0, 0.0, 1.0]), cols([1.0, 0.0, 0.0, 0.0])] {
            assert!(!turns(m), "{m}");
            for i in 0..=20 {
                let s = i as f32 / 20.0;
                assert!(close(along(m, s), Mat2::IDENTITY + (m - Mat2::IDENTITY) * s, 1e-6));
            }
        }
        let mut rng = Rng::new(2);
        let norm = |m: Mat2| m.to_cols_array().iter().map(|v| v * v).sum::<f32>().sqrt();
        for _ in 0..500 {
            let (first, second) = (random(&mut rng, 1.5), random(&mut rng, 1.5));
            if first.determinant() > 1e-3 {
                let low = (0..=100).map(|i| along(first, i as f32 / 100.0).determinant()).fold(f32::MAX, f32::min);
                assert!(low > 0.0, "{first} passes through flat: {low}");
            }
            // stage 1 between I and the first map, stage 2 between the first map
            // and the product (in the size of their entries, measured as a whole)
            let reach = [norm(Mat2::IDENTITY).max(norm(first)), norm(first).max(norm(second * first))];
            for i in 0..=40 {
                let p = i as f32 / 20.0;
                let n = norm(at(first, second, p));
                assert!(n <= reach[(p > 1.0) as usize] * (1.0 + 1e-5), "{first}, {second} at {p}: {n}");
            }
        }
        // the inverse of a nearly flat map still brings the picture home without
        // flinging it out on the way
        let b = cols([1.0, 0.0, 0.0, 0.01]);
        let a = inverse_to_edit(b).expect("within the editor's range");
        assert!(a == cols([1.0, 0.0, 0.0, 100.0]));
        for i in 0..=40 {
            assert!(norm(at(b, a, i as f32 / 20.0)) <= 2f32.sqrt() + 1e-5);
        }
    }

    /// Flat, nearly flat, zero and extreme matrices: every number drawn or written
    /// stays finite and short, and the panel says honestly what has happened.
    #[test]
    fn degenerate_matrices_leave_everything_finite() {
        let cases = [
            Mat2::ZERO,
            cols([1.0, 0.5, 2.0, 1.0]),
            cols([0.7, 1.4, 0.3, 0.6]),
            cols([1.0, 1.0, 1.0, 1.0001]),
            cols([100.0, 100.0, -100.0, 100.0]),
            cols([100.0, 100.0, 100.0, 100.0]),
            cols([-100.0, 0.0, 0.0, -100.0]),
            cols([0.01, 0.0, 0.0, 0.01]),
            cols([100.0, 0.0, 0.0, 0.0001]),
        ];
        for a in cases {
            for b in cases {
                for a_first in [false, true] {
                    let mut scene = Compose::create();
                    (scene.a, scene.b, scene.a_first) = (a, b, a_first);
                    for i in 0..=40 {
                        scene.progress = i as f32 / 20.0;
                        assert!(scene.shown().is_finite(), "{a} {b} at {}", scene.progress);
                    }
                    let (lo, hi) = scene.bounds();
                    assert!(lo.is_finite() && hi.is_finite());
                    for canvas in [Vec2::new(1080.0, 1000.0), Vec2::new(390.0, 800.0), Vec2::new(240.0, 180.0)] {
                        scene.canvas = canvas;
                        let View::Fit { center, half } = scene.the_view() else { panic!("a fitted view") };
                        assert!(center.is_finite() && half.is_finite() && half.min_element() > 0.0);
                    }
                    if let Some((p, q)) = scene.merging() {
                        assert!(p.is_finite() && q.is_finite() && (scene.product() * p - scene.product() * q).length() < 1e-3 * (1.0 + scene.product().abs().to_cols_array().iter().sum::<f32>()));
                    }
                    let text = [scene.stage(), scene.verdict(), scene.head(1, 1), scene.sum_line(0, 1), scene.sum_line(1, 0)].join(" ");
                    assert!(!text.contains("NaN") && !text.contains("inf") && !text.contains('?'), "{text}");
                    // (and its mathematics is set, not drawn as written)
                    assert!(crate::typeset::inspect(&text).error.is_none(), "{text}");
                    for m in [a, b, scene.product()] {
                        for v in m.to_cols_array() {
                            assert!(num(v as f64).chars().count() <= 6, "{v} reads {}", num(v as f64));
                        }
                        assert!(num_det(det(m)).chars().count() <= 8);
                        match inverse(m) {
                            Some(inv) => {
                                assert!(det(m) != 0.0 && inv.iter().all(|v| v.is_finite() && num(*v).chars().count() <= 7), "{m}: {inv:?}");
                                // (in double precision: near flat, single precision
                                // cannot multiply them back to I)
                                let [a, b, c, d] = m.to_cols_array().map(f64::from);
                                let [p, q, r, t] = inv;
                                let product = [p * a + r * b, q * a + t * b, p * c + r * d, q * c + t * d];
                                let size = inv.iter().fold(1.0f64, |s, v| s.max(v.abs())) * a.abs().max(b.abs()).max(c.abs()).max(d.abs());
                                assert!(product.iter().zip([1.0, 0.0, 0.0, 1.0]).all(|(x, y)| (x - y).abs() < 1e-9 * size), "{m}: {product:?}");
                            }
                            None => assert_eq!(det(m), 0.0),
                        }
                    }
                }
            }
        }
        // the flat ones are called flat, and the nearly flat one has an inverse too
        // large for the editor (the button for it is off, and the note says why)
        assert!([0, 1, 2, 5].iter().all(|&k| det(cases[k]) == 0.0));
        assert!(inverse(cases[3]).is_some() && inverse_to_edit(cases[3]).is_none());
        assert!(inverse_to_edit(cases[7]) == Some(cols([100.0, 0.0, 0.0, 100.0])));
    }

    /// The panel's words on inverses, determinants and the quick choices.
    #[test]
    fn the_words_and_the_quick_choices_are_true() {
        let mut rng = Rng::new(4);
        for _ in 0..300 {
            let m = random(&mut rng, 1.2);
            if let Some(inv) = inverse(m) {
                let inv = cols(inv.map(|v| v as f32));
                assert!(close(inv * m, Mat2::IDENTITY, 1e-3) && close(m * inv, Mat2::IDENTITY, 1e-3));
                assert!((det(inv) * det(m) - 1.0).abs() < 1e-3, "det A⁻¹ = 1/det A");
            }
        }
        let preset = |name: &str| cols(PRESETS.iter().find(|p| p.0 == name).expect("a preset").1);
        let v = Vec2::new(0.7, -1.3);
        assert!(turn_and_stretch(preset("turn"), v).0 - 90.0 < 1e-4 && preset("turn").determinant() == 1.0);
        assert_eq!(preset("shear") * v, Vec2::new(v.x + v.y, v.y));
        assert_eq!(preset("stretch") * v, Vec2::new(2.0 * v.x, v.y));
        assert_eq!(preset("flip") * v, Vec2::new(-v.x, v.y));
        assert_eq!(preset("squash") * v, Vec2::new(v.x, 0.0));
        assert_eq!(preset("$I$") * v, v);
        // numbers on screen: no "−0", a true minus, no more digits than they need
        assert_eq!([num(-0.0001), num(-1.0), num(1.0 / 3.0), num(12.26), num(-12345.0), num(250.0), num(f64::NAN)], ["0", "−1", "0.333", "12.3", "−1.2e4", "250", "?"]);
        assert_eq!([num_det(0.0), num_det(-2e-5), num_det(0.5)], ["0", "−2.0e−5", "0.5"]);
    }

    /// What the panel says about the stage the animation is at.
    #[test]
    fn the_stages_are_named_for_what_has_happened() {
        let mut s = Compose::create();
        let names: Vec<String> = [0.0, 0.5, 1.0, 1.5, 2.0]
            .iter()
            .map(|&p| {
                s.progress = p;
                s.stage()
            })
            .collect();
        assert_eq!(
            names,
            ["before either map", "step 1 of 2: $B$ acts", r"after $B$: its columns are where $\hat\imath$ and $\hat\jmath$ are", "step 2 of 2: $A$ acts on what $B$ made", "$B$, then $A$: one map, $AB$"]
        );
        s.a_first = true;
        assert_eq!(s.stage(), "$A$, then $B$: one map, $BA$");
        s.progress = 1.0;
        assert!(close(s.shown(), s.a, 0.0));
    }

    /// The equation keeps to the canvas and clear of the part of it the picture
    /// has, and its numbers keep clear of each other.
    #[test]
    fn the_panel_fits_every_canvas() {
        let canvases = [(1080.0, 1000.0, 0.0), (1600.0, 1000.0, 46.0), (390.0, 800.0, 46.0), (360.0, 640.0, 46.0), (600.0, 900.0, 46.0), (844.0, 390.0, 46.0), (640.0, 480.0, 46.0), (600.0, 400.0, 46.0), (240.0, 180.0, 46.0)];
        let mut shown = 0;
        for (x, y, inset) in canvases {
            let view = Vec2::new(x, y);
            let nominal = Panel::of(view, inset, 4, 44);
            for size in [Vec2::splat(2.4), Vec2::new(4.4, 2.6), Vec2::new(2.0, 4.0)] {
                let ((lo, hi), per_unit) = region(view, inset, size, (4, 44));
                assert!(lo.cmpge(Vec2::new(0.0, inset)).all() && hi.cmple(view).all() && hi.cmpgt(lo).all(), "{view}: the picture's region {lo}..{hi}");
                if let Some(p) = &nominal {
                    let (a, b) = (p.min, p.min + p.size);
                    assert!(hi.x <= a.x + 0.01 || b.x <= lo.x + 0.01 || hi.y <= a.y + 0.01 || b.y <= lo.y + 0.01, "{view}: the picture's region {lo}..{hi} runs under the equation");
                }
                // (the experiments' pictures are at most this size: the scale
                // they are shown at, on any canvas from a small phone up, can be read)
                let least = if x < 360.0 || y < 390.0 {
                    0.0
                } else if x * y < 250_000.0 {
                    40.0
                } else {
                    60.0
                };
                assert!(per_unit > 0.0 && per_unit >= least, "{view}: {size} shown at {per_unit} pixels a unit");
            }
            for (widest, line) in [(1, 30), (5, 47), (6, 60), (7, 70)] {
                let Some(p) = Panel::of(view, inset, widest, line) else { continue };
                shown += 1;
                assert!(p.min.x >= 0.0 && p.min.y >= inset && p.min.x + p.size.x <= x + 0.01 && p.min.y + p.size.y <= y + 0.01, "{view}: {p:?}");
                assert!(p.text >= 9.0 && text_width(&"m".repeat(line), p.text * 0.86) + 2.0 * PAD <= p.size.x + 0.01, "{view}: lines of {line} do not fit");
                let cells: Vec<(Vec2, Vec2)> = (0..3).flat_map(|k| [(0, 0), (0, 1), (1, 0), (1, 1)].map(|(i, j)| p.cell(k, i, j))).collect();
                for (n, a) in cells.iter().enumerate() {
                    assert!(a.0.cmpge(p.min).all() && a.1.cmple(p.min + p.size).all(), "{view}: a number outside the panel: {a:?} {p:?}");
                    assert!(a.1.x - a.0.x >= text_width(&"0".repeat(widest), p.text) * 1.1, "{view}: {widest}-character numbers do not fit their cells");
                    for b in &cells[n + 1..] {
                        assert!(a.1.x <= b.0.x + 0.01 || b.1.x <= a.0.x + 0.01 || a.1.y <= b.0.y + 0.01 || b.1.y <= a.0.y + 0.01, "{view}: numbers overlap");
                    }
                }
            }
        }
        assert!(shown >= 20, "the equation was shown on only {shown} of the canvases");
        assert!(Panel::of(Vec2::new(1080.0, 1000.0), 0.0, 5, 47).is_some() && Panel::of(Vec2::new(390.0, 800.0), 46.0, 5, 47).is_some_and(|p| p.narrow));
    }
}

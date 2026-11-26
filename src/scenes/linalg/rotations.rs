//! Rotations in 3D: turns of space as matrices, the one axis every turn has, why the
//! order of turns matters, three angles and the setting where they lose one, and the
//! mirror image that no turn reaches.

use super::subspaces::{Inputs, Still};
use super::*;
use glam::{DMat3, DQuat, DVec3};

pub const DEF: SceneDef = SceneDef {
    id: "rotations",
    title: "Rotations in 3D",
    group: "Linear algebra: vectors and maps",
    blurb: "Turns of space as matrices: the axis every one of them has, and why their order matters.",
    make: |_| Box::new(Rotations::create()),
};

/// The axes of space the three turns are made about, in the order they are made.
const NAMES: [&str; 3] = ["$x$", "$y$", "$z$"];
const NTH: [&str; 3] = ["1st", "2nd", "3rd"];
/// How far each angle goes either way (degrees). The middle one needs only ±90° for
/// the three to reach every rotation.
const REACH: [f32; 3] = [180.0, 90.0, 180.0];
/// The angles the scene opens with: x first, then y, then z.
const START: [f32; 3] = [30.0, -15.0, 75.0];

/// Experiments: (button, what to look for). What each one sets is in `Rotations::set_up`.
const TRIALS: [(&str, &str); 5] = [
    (
        "order matters",
        r"A quarter turn about $x$ and then one about $y$ (solid), against the same two turns the other way round (purple). $x$ first leaves the nose where it is, and the $y$ turn then points it down; $y$ first points it down at once, and the $x$ turn then swings it round to $y$. They end 120.0° apart: $R_y R_x \ne R_x R_y$.",
    ),
    (
        "two turns make one",
        r"The same two quarter turns, made as one turn instead: 120.0° about the line through $(1, 1, -1)$, in yellow. Every point circles that line by the same angle (the dashed arcs are the paths of the arrow tips), and the aeroplane arrives exactly where the two turns put it (the white outline). The angle is in the diagonal: $\tr R = 0 = 1 + 2\cos 120^\circ$.",
    ),
    (
        "gimbal lock",
        "The 2nd turn, −90° about $y$, points the nose straight up. Now the 1st angle turns the aeroplane about its nose and the 3rd about the $z$ axis, and these are the same line (cyan): either slider spins it the same way, and only their sum counts. No small move of any slider tips the nose towards a wing: one way of turning is lost. Move the 2nd slider off −90° and the lines part.",
    ),
    (
        "a mirror is not a turn",
        r"The green wing folds through the pale sheet, the mirror, to the other side: the mirror image. Its columns still have length 1 and are at right angles ($M^\top M = I$), but $\det M = -1$, and $Me_3$ points against $Me_1 \times Me_2$. Every turn has $\det R = 1$, and $\det M$ changes smoothly as the aeroplane moves, so the way here passes $0$: half-way the aeroplane is flat. Move the sliders: no turn brings the green wing back to the left.",
    ),
    (
        "small turns nearly commute",
        r"Two turns of 10°, about $x$ and about $y$, in both orders. The two ends are only 1.74° apart, against 120° for quarter turns: for small turns the gap is about the product of the two angles in radians, $0.175 \times 0.175 = 0.0305\,\text{rad} = 1.75^\circ$. Halve both to 5° and the gap falls to a quarter, 0.44°.",
    ),
];
const ORDER_TRIAL: usize = 0;
const ONE_TURN_TRIAL: usize = 1;
const LOCK_TRIAL: usize = 2;
const MIRROR_TRIAL: usize = 3;
const SMALL_TRIAL: usize = 4;

/// The colours, each with one meaning. The columns of the matrix: where e₁, e₂, e₃
/// go, which are the aeroplane's nose, its left wing and its fin.
const COLS: [Color; 3] = pal::AXIS;
/// The single turn: its axis, its angle, and the dot that points the axis.
const TURN: Color = pal::YELLOW;
/// The same turns in the other order.
const OTHER: Color = pal::PURPLE;
/// The lines each angle turns the aeroplane about, if it is nudged.
const NUDGE: Color = pal::CYAN;
/// The axis of space the turn being played is about, and where a play ends.
const STEP: Color = pal::WHITE;
/// The mirror the aeroplane is reflected in.
const GLASS: Color = Color::hex(0xC9D3E6);
/// The aeroplane's right wing, which has no colour of its own.
const PAPER: Color = Color::hex(0xB8C0D4);

/// Seconds for one of the turns made one after another, the pause after it, the
/// single turn, the squash into the mirror image, and the glide back to where a play
/// starts from.
const STEP_SECONDS: f32 = 1.5;
const HOLD_SECONDS: f32 = 0.45;
const WHOLE_SECONDS: f32 = 2.6;
const SQUASH_SECONDS: f32 = 2.6;
const LEAD_SECONDS: f32 = 0.8;

/// Where the dot that points the axis sits, and how far the axis is drawn either way.
const HANDLE_R: f32 = 1.2;
const AXIS_LEN: f32 = 1.38;
/// How far the axes of space are drawn.
const SPACE_LEN: f32 = 1.3;
/// How far the lines each angle turns about are drawn.
const NUDGE_LEN: f32 = 1.45;

// ----- the arithmetic, in double precision -----------------------------------------

/// A turn of `degrees` about axis `k` of space (0, 1, 2: x, y, z), by the right-hand
/// rule: with the thumb along the axis, the fingers curl the way it turns.
fn about(k: usize, degrees: f64) -> DMat3 {
    let r = degrees.to_radians();
    match k {
        0 => DMat3::from_rotation_x(r),
        1 => DMat3::from_rotation_y(r),
        _ => DMat3::from_rotation_z(r),
    }
}

/// The turns the angles stand for, in the order they are made (x, then y, then z, or
/// with `reversed` the other way round), leaving out any of no angle at all.
fn steps(angles: [f32; 3], reversed: bool) -> Vec<(usize, f64)> {
    let mut s: Vec<(usize, f64)> = (0..3).map(|k| (k, angles[k] as f64)).filter(|(_, a)| *a != 0.0).collect();
    if reversed {
        s.reverse();
    }
    s
}

/// One turn after another as one matrix: each later turn multiplies from the left.
fn product(steps: &[(usize, f64)]) -> DMat3 {
    steps.iter().fold(DMat3::IDENTITY, |m, &(k, a)| about(k, a) * m)
}

/// R = R_z R_y R_x: the turn about x first, then y, then z.
fn pose(angles: [f32; 3]) -> DMat3 {
    product(&steps(angles, false))
}

/// The same three turns the other way round: z first, then y, then x.
fn other_pose(angles: [f32; 3]) -> DMat3 {
    product(&steps(angles, true))
}

/// A turn of `degrees` about the line through the origin along `axis` (of length 1).
fn rot(axis: DVec3, degrees: f64) -> DMat3 {
    DMat3::from_axis_angle(axis, degrees.to_radians())
}

/// The mirror image: the aeroplane reflected in its own upright plane, the one through
/// its nose and its fin, so that its left wing becomes a right wing.
const MIRROR: DMat3 = DMat3::from_cols(DVec3::X, DVec3::NEG_Y, DVec3::Z);

/// The aeroplane's middle column scaled by `flip`: 1 is the aeroplane, −1 its mirror
/// image, and 0 flat, half-way between.
fn flipped(flip: f64) -> DMat3 {
    DMat3::from_diagonal(DVec3::new(1.0, flip, 1.0))
}

/// The single turn a rotation is.
#[derive(Clone, Copy, Debug)]
struct Turn {
    /// From 0° to 180°.
    degrees: f64,
    /// Its axis, of length 1, by the right-hand rule; None if there is no turn at all.
    axis: Option<DVec3>,
}

/// A rotation's quaternion, worked out from whichever of its four parts is largest,
/// so that nothing is divided by a small number.
fn quat(m: DMat3) -> DQuat {
    let q = DQuat::from_mat3(&m);
    let n = q.length();
    if n > 0.0 && n.is_finite() {
        q / n
    } else {
        DQuat::IDENTITY
    }
}

/// The angle and the axis of the rotation `m`, read from its quaternion. That stays
/// exact near a half turn, where the axis can no longer be read from R − Rᵀ (it is 0
/// there). At a half turn the axis could point either way along its line (the two are
/// the same turn): the way nearer `prefer`.
fn turn_of(m: DMat3, prefer: DVec3) -> Turn {
    let q = quat(m);
    let (v, w) = (DVec3::new(q.x, q.y, q.z), q.w);
    let s = v.length();
    if s.is_nan() || s <= 1e-12 {
        return Turn { degrees: 0.0, axis: None };
    }
    let degrees = (2.0 * s.atan2(w.abs())).to_degrees();
    let mut axis = if w < 0.0 { -v / s } else { v / s };
    if w.abs() < 1e-9 && axis.dot(prefer) < 0.0 {
        axis = -axis;
    }
    Turn { degrees, axis: Some(axis) }
}

/// An angle in degrees, taken into −180° to 180°.
fn wrap(degrees: f64) -> f64 {
    let w = (degrees + 180.0).rem_euclid(360.0) - 180.0;
    if w.is_finite() {
        w
    } else {
        0.0
    }
}

/// The three angles (x first, then y, then z) that make the rotation `m`: every
/// rotation has them, with the middle one from −90° to 90°. Where the middle one is
/// ±90° only the sum or the difference of the other two is settled (gimbal lock): the
/// first is then kept at `first`, and the third makes up the rest.
fn angles_of(m: DMat3, first: f32) -> [f32; 3] {
    // R = Rz(a) Ry(b) Rx(c) has the bottom row (−sin b, cos b sin c, cos b cos c)
    // and the first column (cos a cos b, sin a cos b, −sin b)
    let (r20, r21, r22) = (m.x_axis.z, m.y_axis.z, m.z_axis.z);
    let cb = (r21 * r21 + r22 * r22).sqrt();
    let b = (-r20).atan2(cb);
    let (a, c) = if cb > 1e-9 {
        (m.x_axis.y.atan2(m.x_axis.x), r21.atan2(r22))
    } else {
        // R = Rz(a + c) Ry(−90°) or Rz(a − c) Ry(90°): what is left once the middle
        // turn is undone is a turn about z
        let c = (first as f64).to_radians();
        let rest = m * about(1, b.to_degrees()).transpose();
        let phi = rest.x_axis.y.atan2(rest.x_axis.x);
        (if b > 0.0 { phi + c } else { phi - c }, c)
    };
    let out = [wrap(c.to_degrees()), b.to_degrees(), wrap(a.to_degrees())];
    // (+ 0.0: no slider is to read "−0.0")
    [0, 1, 2].map(|k| if out[k].is_finite() { (out[k] as f32).max(-REACH[k]).min(REACH[k]) + 0.0 } else { 0.0 })
}

/// The line about which a small change of each angle turns the aeroplane as it now
/// stands: the 1st about its own nose (Rz Ry x), the 2nd about Rz y, the 3rd about z
/// itself. They are the axes of a gimbal's three rings.
fn nudge_axes(angles: [f32; 3]) -> [DVec3; 3] {
    let (b, a) = (angles[1] as f64, angles[2] as f64);
    [about(2, a) * about(1, b) * DVec3::X, about(2, a) * DVec3::Y, DVec3::Z]
}

/// The angle between the lines the 1st and the 3rd angle turn about (degrees).
fn lines_apart(angles: [f32; 3]) -> f64 {
    let n = nudge_axes(angles);
    n[0].dot(n[2]).abs().min(1.0).acos().to_degrees()
}

/// How far apart two rotations are: the angle of the turn that takes one to the other.
fn apart(a: DMat3, b: DMat3) -> f64 {
    turn_of(a.transpose() * b, DVec3::X).degrees
}

/// The largest entry of MᵀM − I: 0 when the columns have length 1 and are at right
/// angles to each other (the entries of MᵀM are the dot products of the columns).
fn off_orthonormal(m: DMat3) -> f64 {
    (m.transpose() * m - DMat3::IDENTITY).to_cols_array().iter().fold(0.0, |w, v| w.max(v.abs()))
}

/// An angle in degrees for the screen: one decimal, two below 10°.
fn deg(v: f64) -> String {
    format!("{}°", num(v, if v.abs() < 9.995 { 2 } else { 1 }))
}

/// The same as markup, to go in a formula.
fn deg_tex(v: f64) -> String {
    format!("{}^\\circ", num(v, if v.abs() < 9.995 { 2 } else { 1 }))
}

/// A slider's angle for the screen: as a whole number when it is one.
fn slider_deg(v: f64) -> String {
    format!("{}°", num(v, if (v - v.round()).abs() < 1e-6 { 0 } else { 2 }))
}

/// A direction for the screen: (x, y, z) to three places (markup, to go in a formula).
fn coords(v: DVec3) -> String {
    format!("({}, {}, {})", num(v.x, 3), num(v.y, 3), num(v.z, 3))
}

/// The turns the angles make, in words: "x by 90°, then y by 90°".
fn turns_text(angles: [f32; 3], reversed: bool) -> String {
    let parts: Vec<String> = steps(angles, reversed).iter().map(|&(k, a)| format!("{} by {}", NAMES[k], slider_deg(a))).collect();
    if parts.is_empty() {
        "no turn at all".to_string()
    } else {
        parts.join(", then ")
    }
}

// ----- the aeroplane -----------------------------------------------------------------

/// The aeroplane, in its own coordinates: the nose along x, the left wing along y, the
/// fin up along z. Coloured so that it looks different from every side and its
/// mirror image can be told from it.
const NOSE: Vec3 = Vec3::new(0.78, 0.0, 0.0);
const TAIL: Vec3 = Vec3::new(-0.5, 0.0, 0.0);
const LEFT_TIP: Vec3 = Vec3::new(-0.56, 0.5, 0.0);
const RIGHT_TIP: Vec3 = Vec3::new(-0.56, -0.5, 0.0);
const FIN_FOOT: Vec3 = Vec3::new(-0.2, 0.0, 0.0);
const FIN_TOP: Vec3 = Vec3::new(-0.58, 0.0, 0.36);

/// Its three faces: the left wing, the right wing, the fin.
const FACES: [[Vec3; 3]; 3] = [[NOSE, TAIL, LEFT_TIP], [NOSE, TAIL, RIGHT_TIP], [FIN_FOOT, TAIL, FIN_TOP]];
/// Its outline, as pairs of corners.
const EDGES: [(Vec3, Vec3); 7] = [(NOSE, LEFT_TIP), (LEFT_TIP, TAIL), (TAIL, RIGHT_TIP), (RIGHT_TIP, NOSE), (NOSE, TAIL), (FIN_FOOT, FIN_TOP), (FIN_TOP, TAIL)];

/// The aeroplane carried by `m`: see-through faces shaded as if lit from above, drawn
/// back to front, and its outline.
fn aeroplane(f: &mut Frame, m: Mat3, colors: [Color; 3], alpha: f32, edge: Color, width: f32) {
    let light = Vec3::new(0.35, -0.3, 0.88).normalize();
    let eye = f.cam.eye;
    let mut faces: Vec<([Vec3; 3], Color)> = FACES.iter().zip(colors).map(|(t, c)| (t.map(|p| m * p), c)).collect();
    let far = |t: &[Vec3; 3]| ((t[0] + t[1] + t[2]) / 3.0).distance(eye);
    faces.sort_by(|p, q| far(&q.0).total_cmp(&far(&p.0)));
    for ([a, b, c], color) in faces {
        let tone = 0.5 + 0.5 * (b - a).cross(c - a).normalize_or_zero().dot(light).abs();
        f.dl.tris_alpha.tri(a, b, c, Color::from_linear(color.to_linear() * tone).a(alpha));
    }
    for (a, b) in EDGES {
        f.line(m * a, m * b, width, edge);
    }
}

/// The mirror: the square of the plane through the origin spanned by `u` and `v`
/// (of length 1), `half` either way, faint enough to see the aeroplane through.
fn glass(f: &mut Frame, u: Vec3, v: Vec3, half: f32) {
    let q = |s: f32, t: f32| (u * s + v * t) * half;
    f.dl.tris_alpha.quad(q(-1.0, -1.0), q(1.0, -1.0), q(1.0, 1.0), q(-1.0, 1.0), GLASS.a(0.045));
    for k in -1..=1 {
        let s = k as f32 / 2.0;
        f.line(q(s, -1.0), q(s, 1.0), 1.0, GLASS.a(0.08));
        f.line(q(-1.0, s), q(1.0, s), 1.0, GLASS.a(0.08));
    }
    f.dl.lines.closed(&[q(-1.0, -1.0), q(1.0, -1.0), q(1.0, 1.0), q(-1.0, 1.0)], 1.2, GLASS.a(0.35));
}

/// The aeroplane's outline alone, dashed: where it starts, or where it is going.
fn outline(f: &mut Frame, m: Mat3, width: f32, color: Color) {
    for (a, b) in EDGES {
        f.dashed(m * a, m * b, width, color);
    }
}

// ----- the layout ----------------------------------------------------------------------

/// Where the panel of numbers and buttons goes on a canvas of `view` logical pixels,
/// `inset` of whose top lies under the application's buttons: over the picture's
/// top-left corner, or below the picture on a canvas narrow and tall enough (a phone
/// held upright). Left out where the canvas cannot spare the room.
#[derive(Clone, Copy, Debug, PartialEq)]
struct Layout {
    narrow: bool,
    panel: Option<(Vec2, Vec2)>,
    /// Where the picture's part of the canvas ends at the bottom.
    below: f32,
}

/// The panel's width, and its height when it says the most it can; where its lines
/// start, and how far apart they are.
const PANEL_W: f32 = 352.0;
const PANEL_H: f32 = 292.0;
const PANEL_TOP: f32 = 163.0;
const PANEL_LINE: f32 = 17.0;

impl Layout {
    /// (`height`: the panel's height on a wide canvas. On a phone it is always the
    /// largest, so that the picture above it stays put.)
    fn of(view: Vec2, inset: f32, height: f32) -> Layout {
        if view.x < 620.0 && view.y >= 560.0 {
            let top = view.y - PANEL_H - 6.0;
            Layout { narrow: true, panel: Some((Vec2::new(6.0, top), Vec2::new(view.x - 6.0, view.y - 6.0))), below: top - 4.0 }
        } else {
            let top = 14.0f32.max(inset + 8.0);
            let fits = view.x >= PANEL_W + 300.0 && view.y >= top + PANEL_H + 120.0;
            Layout { narrow: false, panel: fits.then(|| (Vec2::new(14.0, top), Vec2::new(14.0 + PANEL_W, top + height))), below: view.y }
        }
    }

    fn covers(&self, p: Vec2) -> bool {
        self.panel.is_some_and(|(min, max)| p.cmpge(min).all() && p.cmple(max).all())
    }
}

/// Where along the axis the curled arrow of the angle is drawn, and its radius.
const PROTRACTOR_AT: f64 = 0.92;
const PROTRACTOR_R: f64 = 0.24;

/// The direction across the axis `n` that the curled arrow of the angle starts from:
/// square to the axis and to the line of sight the scene opens with, so that the arrow
/// is seen side on, and changing smoothly with the axis. (No such choice can be smooth
/// for every axis: this one fails only for an axis pointing straight at that eye.)
fn protractor_start(n: DVec3) -> DVec3 {
    let (right, up) = screen_axes();
    let across = n.cross(right.cross(up).as_dvec3());
    if across.length() > 1e-6 {
        across.normalize()
    } else {
        n.any_orthonormal_vector()
    }
}

/// How far from the origin the picture reaches (world units), and the room kept round
/// it for the names written at its edges (logical pixels).
const REACH_DRAWN: f32 = 1.45;
const LABEL_ROOM: f32 = 22.0;

/// The round part of the canvas the picture is fitted into: its centre and radius, in
/// logical pixels. As large as fits below the application's buttons and above the
/// caption (and above the panel, on a phone) and, on a wide canvas, clear of the panel
/// in its corner: the picture is round, so it can reach down beside the panel.
fn stage(canvas: Vec2, inset: f32, lay: &Layout) -> (Vec2, f32) {
    let top = inset + 4.0;
    let bottom = if lay.narrow { lay.below } else { canvas.y } - caption_height(canvas);
    let bottom = bottom.max(top + 60.0);
    let mut best = (Vec2::new(canvas.x * 0.5, (top + bottom) * 0.5), 30.0f32);
    for i in 0..=48 {
        for j in 0..=48 {
            let c = Vec2::new(canvas.x * i as f32 / 48.0, top + (bottom - top) * j as f32 / 48.0);
            let mut r = c.x.min(canvas.x - c.x).min(c.y - top).min(bottom - c.y);
            if let (false, Some((lo, hi))) = (lay.narrow, lay.panel) {
                let nearest = c.max(lo).min(hi);
                r = r.min((c - nearest).length() - 8.0);
            }
            if r > best.1 {
                best = (c, r);
            }
        }
    }
    best
}

/// The parts of the caption along the foot of the picture, naming what has no name of
/// its own there (the last only while the arcs are drawn).
const CAPTION: [&str; 3] = ["grey: the axes of space", "faint: before any turn", "dashed arcs: the tips' paths in the single turn"];

/// How much of the canvas's foot the caption takes with all its parts (the picture
/// keeps clear of that, whatever it says).
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

/// The caption, on a dark plate, its last line ending at `bottom` (above the panel, on
/// a phone).
fn caption(f: &mut Frame, parts: &[&str], bottom: f32) {
    let view = f.view_size();
    if view.y < 300.0 {
        return;
    }
    let lines = caption_lines(view.x, parts);
    let count = lines.len();
    for (k, line) in lines.into_iter().enumerate() {
        let up = (count - 1 - k) as f32 * CAPTION_LINE;
        f.caption_px(Vec2::new(18.0, bottom - up), line, CAPTION_SIZE, pal::DIM, Align2::LEFT_BOTTOM);
    }
}

/// The direction the scene is seen from at first: in front, to the right and above.
const YAW: f32 = -1.1;
const PITCH: f32 = 0.35;
/// (The tangent of half the camera's field of view.)
const TAN_HALF_FOV: f32 = 0.3203;

/// The camera's right and up directions in the view the scene opens with.
fn screen_axes() -> (Vec3, Vec3) {
    let right = Vec3::new(-YAW.sin(), YAW.cos(), 0.0);
    let up = Vec3::new(-PITCH.sin() * YAW.cos(), -PITCH.sin() * YAW.sin(), PITCH.cos());
    (right, up)
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

// ----- the scene -----------------------------------------------------------------------

/// What is being played.
#[derive(Clone, Copy, Debug, PartialEq)]
enum Play {
    /// The turns one after another (and the other order beside them, if shown).
    Steps,
    /// The single turn about the axis.
    Whole,
    /// Squashed through flat, into the mirror image or back out of it.
    Squash,
}

/// What was drawn when a play began, which it first glides back from to where it
/// starts (the short way round): the turning part of the aeroplane, of the other
/// order's, and how far it was mirrored (see `flipped`).
#[derive(Clone, Copy, Debug)]
struct Drawn {
    turned: DQuat,
    other: DQuat,
    flip: f64,
}

pub struct Rotations {
    /// The three angles, in degrees: about x (made first), about y, about z (made last).
    angles: [f32; 3],
    /// The axis the yellow dot stands on (length 1). It follows the rotation's own
    /// axis, and stays where it was when there is no turn (so that an angle can be
    /// given to it again).
    axis: DVec3,
    /// The angle of the single turn, degrees (0 to 180), as its slider holds it.
    turn: f32,
    /// Where the dot is: on the axis, `HANDLE_R` from the origin.
    grip: Vec3,
    /// What is shown: the mirror image, the other order, the lines the angles turn about.
    mirror: bool,
    other: bool,
    nudges: bool,
    /// What is being played, and for how long so far (seconds of scene time; below 0
    /// while it glides back to where it starts from, `lead`).
    play: Option<(Play, f32)>,
    lead: Option<Drawn>,
    /// The canvas last drawn on, and how much of its top the application's buttons
    /// cover: they decide where the panel goes, and with it where the picture goes.
    canvas: Vec2,
    inset: f32,
    /// Whether a canvas has been drawn on yet (until then `canvas` is a guess).
    measured: bool,
    new_view: bool,
    /// The names in the picture as they were written last, kept while nothing they are
    /// placed from changes (see `Still`).
    names: Still,
}

impl Rotations {
    /// (Nothing here needs the graphics device, so the tests can make one too.)
    fn create() -> Self {
        let mut s = Rotations {
            angles: START,
            axis: DVec3::Z,
            turn: 0.0,
            grip: Vec3::Z * HANDLE_R,
            mirror: false,
            other: false,
            nudges: false,
            play: None,
            lead: None,
            canvas: Vec2::new(1080.0, 1000.0),
            inset: 0.0,
            measured: false,
            new_view: false,
            names: Still::default(),
        };
        s.follow_angles();
        s
    }

    /// Set up one of the experiments in `TRIALS`, and play it.
    fn set_up(&mut self, k: usize) {
        let (angles, other, nudges, mirror, play) = match k {
            ORDER_TRIAL => ([90.0, 90.0, 0.0], true, false, false, Play::Steps),
            ONE_TURN_TRIAL => ([90.0, 90.0, 0.0], false, false, false, Play::Whole),
            LOCK_TRIAL => ([0.0, -90.0, 40.0], false, true, false, Play::Steps),
            MIRROR_TRIAL => ([0.0, 0.0, -40.0], false, false, true, Play::Squash),
            SMALL_TRIAL => ([10.0, 10.0, 0.0], true, false, false, Play::Steps),
            _ => return,
        };
        let from = self.drawn();
        self.angles = angles;
        (self.other, self.nudges, self.mirror) = (other, nudges, mirror);
        self.follow_angles();
        self.start(play, from);
    }

    /// The angles have changed: the single turn follows them.
    fn follow_angles(&mut self) {
        let t = turn_of(self.pose(), self.axis);
        self.turn = t.degrees as f32;
        if let Some(axis) = t.axis {
            self.axis = axis;
        }
        self.grip = (self.axis * HANDLE_R as f64).as_vec3();
    }

    /// The single turn has been given a new axis or angle: the three angles follow it.
    fn set_turn(&mut self, axis: DVec3, degrees: f32) {
        self.axis = axis;
        self.turn = degrees.max(0.0).min(180.0);
        self.angles = angles_of(rot(axis, self.turn as f64), self.angles[0]);
    }

    /// Show the mirror image or not, by squashing the aeroplane through flat.
    fn set_mirror(&mut self, on: bool) {
        if on != self.mirror {
            let from = self.drawn();
            self.mirror = on;
            self.start(Play::Squash, from);
        }
    }

    /// What is drawn now, for a play to glide back from.
    fn drawn(&self) -> Drawn {
        let turned = quat(self.turned(false));
        let other = if self.other { quat(self.turned(true)) } else { turned };
        Drawn { turned, other, flip: self.flip() }
    }

    /// Where a play starts from: the turning part of the aeroplane and of the other
    /// order's, and how far it is mirrored. The turns start from no turn at all; the
    /// squash from the turned aeroplane, on the side of the mirror it is leaving.
    fn start_of(&self, play: Play) -> (DMat3, DMat3, f64) {
        let side = if self.mirror { -1.0 } else { 1.0 };
        match play {
            Play::Squash => (self.pose(), other_pose(self.angles), -side),
            _ => (DMat3::IDENTITY, DMat3::IDENTITY, side),
        }
    }

    /// Can this be played now?
    fn can(&self, play: Play) -> bool {
        match play {
            Play::Steps => !steps(self.angles, false).is_empty(),
            Play::Whole => !self.mirror && self.whole().axis.is_some(),
            Play::Squash => true,
        }
    }

    /// Play from the start, gliding there first from what was drawn (`from`) if that
    /// is somewhere else.
    fn start(&mut self, play: Play, from: Drawn) {
        if !self.can(play) {
            (self.play, self.lead) = (None, None);
            return;
        }
        let (to, to_other, flip) = self.start_of(play);
        let away = |q: DQuat, m: DMat3| q.dot(quat(m)).abs().min(1.0).acos().to_degrees() * 2.0;
        let far = away(from.turned, to) > 0.5 || (self.other && away(from.other, to_other) > 0.5) || (from.flip - flip).abs() > 1e-9;
        self.lead = far.then_some(from);
        self.play = Some((play, if far { -LEAD_SECONDS } else { 0.0 }));
    }

    /// How long a play lasts (from its start, after any glide back to it).
    fn length(&self, play: Play) -> f32 {
        match play {
            Play::Steps => {
                let n = steps(self.angles, false).len() as f32;
                n * STEP_SECONDS + (n - 1.0).max(0.0) * HOLD_SECONDS
            }
            Play::Whole => WHOLE_SECONDS,
            Play::Squash => SQUASH_SECONDS,
        }
    }

    /// Move a play on by `dt` seconds of scene time.
    fn advance(&mut self, dt: f32) {
        if let Some((play, t)) = self.play {
            let t = t + dt.max(0.0);
            if t < self.length(play) {
                self.play = Some((play, t));
                if t >= 0.0 {
                    self.lead = None;
                }
            } else {
                (self.play, self.lead) = (None, None);
            }
        }
    }

    /// The glide back to the start, if one is under way: what it glides from, and how
    /// far it has got (eased, 0 to 1).
    fn leading(&self) -> Option<(Play, Drawn, f64)> {
        match (self.play, self.lead) {
            (Some((play, t)), Some(from)) if t < 0.0 => Some((play, from, smoothstep(1.0 + t / LEAD_SECONDS) as f64)),
            _ => None,
        }
    }

    /// A play under way past its glide back to the start: what, and for how long so far.
    fn playing(&self) -> Option<(Play, f32)> {
        self.play.filter(|(_, t)| *t >= 0.0)
    }

    /// The rotation the angles make, and its single turn.
    fn pose(&self) -> DMat3 {
        pose(self.angles)
    }

    fn whole(&self) -> Turn {
        turn_of(self.pose(), self.axis)
    }

    /// Which step a play of the turns one after another is at, and how far through it
    /// (eased), at time `t`.
    fn stage(&self, t: f32) -> (usize, f64) {
        let n = steps(self.angles, false).len().max(1);
        let period = STEP_SECONDS + HOLD_SECONDS;
        let k = ((t / period).max(0.0) as usize).min(n - 1);
        let u = (t - k as f32 * period) / STEP_SECONDS;
        (k, smoothstep(u) as f64)
    }

    /// How far round the single turn has got (degrees): all of it, unless it is being
    /// played.
    fn whole_so_far(&self) -> f64 {
        let whole = self.whole().degrees;
        match self.playing() {
            Some((Play::Whole, t)) => whole * smoothstep(t / WHOLE_SECONDS) as f64,
            _ => whole,
        }
    }

    /// The turning part of what is drawn now (`other`: of the other order).
    fn turned(&self, other: bool) -> DMat3 {
        let end = if other { other_pose(self.angles) } else { self.pose() };
        if let Some((play, from, e)) = self.leading() {
            let (to, to_other, _) = self.start_of(play);
            let (a, b) = if other { (from.other, quat(to_other)) } else { (from.turned, quat(to)) };
            return DMat3::from_quat(a.slerp(b, e).normalize());
        }
        match self.playing() {
            Some((Play::Steps, t)) => {
                let list = steps(self.angles, other);
                let (k, e) = self.stage(t);
                match list.get(k) {
                    Some(&(axis, a)) => about(axis, a * e) * product(&list[..k]),
                    None => end,
                }
            }
            Some((Play::Whole, t)) => {
                let turn = turn_of(end, self.axis);
                match turn.axis {
                    Some(axis) => rot(axis, turn.degrees * smoothstep(t / WHOLE_SECONDS) as f64),
                    None => end,
                }
            }
            _ => end,
        }
    }

    /// How far the aeroplane is mirrored now, as the factor its middle column is
    /// multiplied by: 1 itself, −1 its mirror image, 0 flat.
    fn flip(&self) -> f64 {
        let to = if self.mirror { -1.0 } else { 1.0 };
        if let Some((play, from, e)) = self.leading() {
            let (_, _, start) = self.start_of(play);
            return from.flip + (start - from.flip) * e;
        }
        match self.playing() {
            Some((Play::Squash, t)) => -to + 2.0 * to * smoothstep(t / SQUASH_SECONDS) as f64,
            _ => to,
        }
    }

    /// The matrix of the aeroplane as it is drawn now (`other`: the other order's).
    fn shown(&self, other: bool) -> DMat3 {
        self.turned(other) * flipped(self.flip())
    }

    /// Is the matrix drawn now a rotation (not a mirror image, nor on the way to one)?
    fn is_turn(&self) -> bool {
        self.flip() == 1.0
    }

    /// The turn being played now, if the turns are being played one after another:
    /// (which of them, axis of space).
    /// (Not in the pause after it: the next turn is then about another axis.)
    fn playing_step(&self) -> Option<(usize, usize)> {
        match self.playing() {
            Some((Play::Steps, t)) if t - self.stage(t).0 as f32 * (STEP_SECONDS + HOLD_SECONDS) <= STEP_SECONDS => {
                let (k, _) = self.stage(t);
                steps(self.angles, false).get(k).map(|&(axis, _)| (k, axis))
            }
            _ => None,
        }
    }

    /// The three angles of the aeroplane as it is drawn, where it is drawn at some: at
    /// rest, and part of the way through the turns played one after another (the
    /// turns made so far, and part of the one being made). Not on the way back to the
    /// start, nor in the single turn or the squash.
    fn drawn_angles(&self) -> Option<[f32; 3]> {
        if self.lead.is_some() {
            return None;
        }
        match self.playing() {
            None => Some(self.angles),
            Some((Play::Steps, t)) => {
                let (k, e) = self.stage(t);
                let mut angles = [0.0; 3];
                for (i, &(axis, a)) in steps(self.angles, false).iter().enumerate() {
                    angles[axis] = if i < k {
                        a as f32
                    } else if i == k {
                        (a * e) as f32
                    } else {
                        0.0
                    };
                }
                Some(angles)
            }
            Some(_) => None,
        }
    }

    /// Is the picture of the single turn shown: its axis and the dot that points it,
    /// the paths of the arrow tips and the angle? Only while nothing else is (the other
    /// order, the lines the angles turn about), for a picture with one thing to say.
    fn plain(&self) -> bool {
        !self.mirror && !self.other && !self.nudges
    }

    /// Is the single turn drawn now? Not for a mirror image or on the way to one, nor
    /// while the turns are played one by one.
    fn shows_arcs(&self) -> bool {
        self.plain() && self.is_turn() && self.whole().axis.is_some() && self.lead.is_none() && !matches!(self.play, Some((Play::Steps, _)))
    }

    /// The view that fits the picture into its round part of the canvas (`stage`).
    fn the_view(&self) -> View {
        let lay = Layout::of(self.canvas, self.inset, PANEL_H);
        let (right, up) = screen_axes();
        let (centre, radius) = stage(self.canvas, self.inset, &lay);
        let per_unit = ((radius - LABEL_ROOM) / REACH_DRAWN).max(1.0);
        let dist = (self.canvas.y * 0.5 / per_unit / TAN_HALF_FOV).clamp(2.0, 60.0);
        let per_unit = self.canvas.y * 0.5 / (dist * TAN_HALF_FOV);
        // move the middle of the picture to the middle of its part of the canvas
        let shift = (centre - self.canvas * 0.5) / per_unit;
        View::Orbit { target: -right * shift.x + up * shift.y, yaw: YAW, pitch: PITCH, dist }
    }

    /// The lines of the panel under its matrix, each with its colour.
    fn panel_lines(&self) -> Vec<(String, Color)> {
        let m = self.shown(false);
        let turn_ok = self.is_turn();
        let name = if turn_ok { "R" } else { "M" };
        let mut lines = Vec::new();
        if off_orthonormal(m) < 1e-9 {
            lines.push((format!("${name}^\\top {name} = I$: columns of length 1, at right angles"), pal::DIM));
        } else {
            lines.push((format!("${name}^\\top {name} \\ne I$: a column is not of length 1"), pal::DIM));
        }
        let det = m.determinant();
        let what = if turn_ok {
            ""
        } else if det < -1.0 + 1e-9 {
            ": a mirror image, not a turn"
        } else if det.abs() < 5e-4 {
            ": flat"
        } else {
            ": squashed"
        };
        lines.push((format!("$\\det {name} = {}${what}", num(det, 3)), pal::TEXT));
        if turn_ok {
            let whole = turn_of(m, self.axis);
            match whole.axis {
                Some(axis) => {
                    lines.push((format!("one turn of {} about ${}$", deg(whole.degrees), coords(axis)), TURN));
                    let trace = m.x_axis.x + m.y_axis.y + m.z_axis.z;
                    lines.push((format!("$\\tr {name} = {} = 1 + 2\\cos {}$", num(trace, 3), deg_tex(whole.degrees)), TURN));
                }
                None => lines.push(("no turn at all ($R = I$), so no axis".to_string(), TURN)),
            }
        } else {
            let way = if det < 0.0 { "against" } else { "along" };
            lines.push((format!("${name}e_3$ points {way} ${name}e_1 \\times {name}e_2$"), pal::DIM));
        }
        if self.other {
            let gap = apart(self.pose(), other_pose(self.angles));
            let names: Vec<&str> = steps(self.angles, true).iter().map(|&(k, _)| NAMES[k]).collect();
            lines.push((format!("other order ({}): {} away", names.join(", then "), deg(gap)), OTHER));
            if let Some((a, b)) = self.small_pair() {
                lines.push((format!("about angle $\\times$ angle: ${} \\times {}\\,\\text{{rad}} = {}$", num(a, 3), num(b, 3), deg_tex((a * b).to_degrees())), pal::DIM));
            }
        }
        if self.nudges {
            let lines_at = lines_apart(self.angles);
            let text = if lines_at < 0.005 { "the 1st and 3rd angles turn about one line".to_string() } else { format!("the 1st and 3rd angles' lines: {} apart", deg(lines_at)) };
            lines.push((text, NUDGE));
        }
        lines
    }

    /// The two angles (radians), when there are just two turns and both are small: the
    /// gap between the two orders is then about their product.
    fn small_pair(&self) -> Option<(f64, f64)> {
        match steps(self.angles, false)[..] {
            [(_, a), (_, b)] if a.abs() <= 30.0 && b.abs() <= 30.0 => Some((a.to_radians().abs(), b.to_radians().abs())),
            _ => None,
        }
    }

    /// How tall the panel is, for what it says now.
    fn panel_height(&self) -> f32 {
        PANEL_TOP + self.panel_lines().len() as f32 * PANEL_LINE + 10.0
    }
}

impl Scene for Rotations {
    fn view(&self) -> View {
        self.the_view()
    }

    fn take_view(&mut self) -> Option<View> {
        std::mem::take(&mut self.new_view).then(|| self.the_view())
    }

    /// Only a play moves by itself.
    fn animated(&self) -> bool {
        self.play.is_some()
    }

    fn settings(&self, s: &mut ViewSettings) {
        s.grid_alpha = 0.07;
        // (the axes of space are drawn by the scene, as far as they are needed)
        s.axis_alpha = 0.0;
    }

    fn ui(&mut self, ui: &mut egui::Ui, _: &Gfx) {
        ui::note(ui, "A rotation turns all of space about the origin, keeping every length and every angle. Its matrix $R$ has as its columns where the $x$, $y$ and $z$ axes go: the red, green and blue arrows, along the aeroplane's nose, left wing and fin. They have length 1, they are at right angles, and they keep the order of $x$, $y$ and $z$. However a rotation is made, it is one turn about one line, its axis (yellow). Build one from three turns with the sliders, or drag the yellow dot and set the angle.");

        ui::section(ui, "Things to try");
        for (k, trial) in TRIALS.iter().enumerate() {
            if ui::experiment(ui, trial.0, trial.1) {
                self.set_up(k);
            }
        }

        ui::section(ui, "Three turns");
        let mut moved = false;
        for k in 0..3 {
            moved |= ui::slider(ui, &format!("{}: about {} (degrees)", NTH[k], NAMES[k]), &mut self.angles[k], -REACH[k]..=REACH[k]);
        }
        if moved {
            (self.play, self.lead) = (None, None);
            self.follow_angles();
        }
        ui::note(ui, "Each turn is about an axis of space (the grey lines), not of the aeroplane, and they are made in this order: $x$ first, $z$ last. Together they make $R = R_z R_y R_x$ ($R_x$ is the turn about $x$, and so on), the matrix on the right acting first. Every rotation can be made like this.");

        ui::section(ui, "One turn");
        let mut turn = self.turn;
        if ui::slider(ui, r"angle $\theta$ (degrees)", &mut turn, 0.0..=180.0) {
            (self.play, self.lead) = (None, None);
            self.set_turn(self.axis, turn);
        }
        ui::note(ui, "About the yellow axis, by the right-hand rule. Drag the yellow dot to point the axis anywhere (it is there while nothing below is shown): the three angles above follow, and the other way round.");

        ui::section(ui, "Play");
        ui.horizontal_wrapped(|ui| {
            if ui.add_enabled(self.can(Play::Steps), egui::Button::new("▶ the turns in order")).clicked() {
                self.start(Play::Steps, self.drawn());
            }
            if ui.add_enabled(self.can(Play::Whole), egui::Button::new("▶ as one turn")).clicked() {
                self.start(Play::Whole, self.drawn());
            }
        });
        ui::note(ui, "The same buttons are on the picture. Both start from the aeroplane before any turn and end in the same place.");

        ui::section(ui, "Show");
        ui.checkbox(&mut self.other, ui::label_text(ui, "the other order: $z$ first, then $y$, then $x$", 12.5, None, false));
        ui.checkbox(&mut self.nudges, "the line each angle turns it about");
        let mut mirror = self.mirror;
        if ui.checkbox(&mut mirror, "the mirror image").changed() {
            self.set_mirror(mirror);
        }

        ui::section(ui, "Reading the picture");
        let mut key: Vec<(Color, &str)> = vec![
            (COLS[0], "red arrow $Re_1$: where the $x$ axis goes, the 1st column of $R$; the aeroplane's nose points along it"),
            (COLS[1], "green arrow $Re_2$ and the green wing: where $y$ goes, the 2nd column"),
            (COLS[2], "blue arrow $Re_3$ and the blue fin: where $z$ goes, the 3rd column"),
            (TURN, r"yellow line: the axis of the single turn, and the curled arrow round it the angle $\theta$ it turns by (drag the yellow dot to point the axis); the dashed arcs in the arrows' colours are the paths of their tips in that turn"),
            (pal::DIM, "grey lines $x$, $y$, $z$: the axes of space; the faint dashed outline: the aeroplane before any turn"),
            (STEP, "white: the axis of space the turn being played is about, and the outline of where the play ends"),
        ];
        if self.mirror {
            key.push((GLASS, "pale sheet: the mirror, through the aeroplane's nose and fin"));
        }
        if self.other {
            key.push((OTHER, "purple: the same turns in the other order, and its nose"));
        }
        if self.nudges {
            key.push((NUDGE, "cyan lines: what each angle turns the aeroplane about, if it is changed a little"));
        }
        ui::key(ui, &key);

        ui::section(ui, "Words");
        ui::term(ui, "$e_1, e_2, e_3$:", "one step along $x$, along $y$ and along $z$. $Re_1$ is $R$ times $e_1$, the 1st column of $R$, and so on.");
        ui::term(ui, "Orthonormal:", r"of length 1 and at right angles to each other. The columns of $R$ are, which is what $R^\top R = I$ says: the entries of $R^\top R$ are the dot products of the columns.");
        ui::term(ui, r"Transpose $R^\top$:", "$R$ with its rows and columns swapped. For a rotation it is the inverse: the turn back.");
        ui::term(ui, "Determinant:", r"the factor by which a map multiplies volumes, negative if it turns space inside out (a mirror image). Every rotation has $\det R = 1$.");
        ui::term(ui, "Axis:", "the line a turn leaves where it is. A direction $v$ along it is an eigenvector of $R$ with eigenvalue 1: $Rv = v$.");
        ui::term(ui, r"Trace $\tr R$:", r"the sum of the diagonal entries. For a turn by $\theta$ it is $1 + 2\cos\theta$, whatever the axis.");
        ui::term(ui, "Right-hand rule:", "with the right thumb along the axis, the fingers curl the way a positive angle turns.");
        ui::term(ui, "Commute:", "two turns commute if their order does not matter.");
        ui::term(ui, "Gimbal lock:", "the setting of three angles at which two of them turn about the same line, so that one way of turning cannot be made by changing them a little.");

        ui::section(ui, "Numbers");
        let r = self.pose();
        let whole = self.whole();
        let m = if self.mirror { r * MIRROR } else { r };
        ui::readout(ui, if self.mirror { r"$\det M$" } else { r"$\det R$" }, num(m.determinant(), 3));
        // (a number this small is written with a power of ten)
        let off = off_orthonormal(m);
        ui::readout(ui, r"largest entry of $R^\top R - I$", if off == 0.0 { "0".to_string() } else { format!("${off:.1e}$") });
        if self.mirror {
            ui::readout(ui, "single turn", "none: a mirror image");
        } else {
            ui::readout_colored(ui, r"angle $\theta$ of the single turn", deg(whole.degrees), TURN);
            ui::readout_colored(ui, "its axis", whole.axis.map_or("none: no turn".to_string(), |a| format!("${}$", coords(a))), TURN);
            let trace = r.x_axis.x + r.y_axis.y + r.z_axis.z;
            ui::readout(ui, r"$\tr R$", num(trace, 3));
            ui::readout(ui, r"$1 + 2\cos\theta$", num(1.0 + 2.0 * whole.degrees.to_radians().cos(), 3));
        }
        if self.other {
            let o = other_pose(self.angles);
            ui::readout_colored(ui, "the other order ends", format!("{} away", deg(apart(r, o))), OTHER);
            // its matrix: the turns made z first, then y, then x
            let rows: Vec<[f64; 3]> = (0..3).map(|i| [o.x_axis[i], o.y_axis[i], o.z_axis[i]]).collect();
            ui::equation(ui, &format!(r"\textcolor{{purple}}{{R_x R_y R_z}} = {}", bmatrix(&rows, 3)));
        }
        if self.nudges {
            ui::readout_colored(ui, "1st and 3rd lines apart by", deg(lines_apart(self.angles)), NUDGE);
        }
        ui::see_also(ui, &[("compose", "one map after another, in the plane"), ("linmap3", r"any $3 \times 3$ matrix, a rotation among them"), ("gram", "making orthonormal columns"), ("expm", "a turn as a flow")]);
    }

    fn frame(&mut self, f: &mut Frame) {
        self.advance(f.dt);
        // the panel's place follows the canvas, and the picture's place follows the panel's
        let view = f.view_size();
        let lay = Layout::of(view, f.top_inset, self.panel_height());
        let before = Layout::of(self.canvas, self.inset, PANEL_H);
        // (a canvas squeezed to a sliver is one the controls have opened over, on a
        // phone: the picture is not seen then, and is left as it was for when they close)
        let covered = view.x < 160.0;
        let changed = !covered && (lay.narrow != before.narrow || lay.panel.is_some() != before.panel.is_some() || (view - self.canvas).abs().max_element() > 24.0);
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
            // quick clicks are two clicks, not a reset of the view
            f.handles.capture();
            f.handles.claim_clicks();
        }

        // the dot that points the axis, first, so that everything is drawn where it now
        // is (not while a play runs, nor for a mirror image, which has no axis; and
        // one under the panel is left alone unless it is being dragged already)
        if self.play.is_none() && self.plain() {
            let hidden = on_screen(f, self.grip).is_some_and(|p| lay.covers(p));
            if !hidden || f.handles.dragging() {
                let mut grip = self.grip;
                // (the dot stays on its sphere: dragged off it, it goes to the point of
                // the sphere in that direction, so that it is where it is let go)
                if f.drag3(1, &mut grip, None, TURN) && grip.length() > 0.2 * HANDLE_R {
                    self.set_turn(grip.as_dvec3().normalize(), self.turn);
                }
                self.grip = (self.axis * HANDLE_R as f64).as_vec3();
            }
        }

        let mut names = std::mem::take(&mut self.names);
        self.picture(f, &lay, &mut names);
        self.names = names;
        self.panel(f, &lay, pointer);
        // what has no name of its own in the picture, named along its bottom
        let bottom = if lay.narrow { lay.below - 4.0 } else { view.y - 18.0 };
        caption(f, &CAPTION[..if self.shows_arcs() { 3 } else { 2 }], bottom);
    }
}

impl Rotations {
    /// Everything in space.
    fn picture(&self, f: &mut Frame, lay: &Layout, names: &mut Still) {
        let mut room = Room::new(f);
        if let Some((min, max)) = lay.panel {
            room.wall(min - Vec2::splat(4.0), max + Vec2::splat(4.0));
        }
        let m = self.shown(false).as_mat3();
        let rest = self.play.is_none();
        let whole = self.whole();
        let outwards = |f: &Frame, from: Vec3, to: Vec3| match (on_screen(f, from), on_screen(f, to)) {
            (Some(o), Some(p)) if (p - o).length() > 1.0 => p - o,
            _ => Vec2::new(1.0, -1.0),
        };

        // the axes of space, and where the arrows start: one step along each
        let stepping = self.playing_step();
        for k in 0..3 {
            let d = Vec3::AXES[k];
            if stepping.is_some_and(|(_, axis)| axis == k) {
                f.line(-d * SPACE_LEN, d * SPACE_LEN, 2.4, STEP.a(0.9));
            } else {
                f.line(-d * SPACE_LEN, d * SPACE_LEN, 1.1, pal::DIM.a(0.55));
            }
            f.dot(d, 5.0, pal::DIM);
        }
        // (names may cross these faint lines: kept off them, a name would be pushed
        // about whenever what it names passes one)

        // the aeroplane before any turn
        outline(f, Mat3::IDENTITY, 1.1, pal::DIM.a(0.7));
        // the mirror, whenever the aeroplane is mirrored or on its way: the plane
        // through its nose and its fin, which the wings pass through
        let mirror_at = (self.mirror || !self.is_turn()).then(|| {
            let t = self.turned(false);
            ((t * DVec3::X).as_vec3(), (t * DVec3::Z).as_vec3())
        });
        if let Some((u, v)) = mirror_at {
            glass(f, u, v, 0.95);
            room.line(f, -u - v, u - v);
            room.line(f, u - v, u + v);
            room.line(f, u + v, -u + v);
            room.line(f, -u + v, -u - v);
        }

        // the other order, as a ghost, with its nose
        let ghost = self.other.then(|| self.shown(true).as_mat3());
        if let Some(g) = ghost {
            aeroplane(f, g, [OTHER; 3], 0.24, OTHER, 2.2);
            f.arrow(Vec3::ZERO, g.x_axis, 2.4, OTHER);
            for (a, b) in EDGES {
                room.line(f, g * a, g * b);
            }
            room.line(f, Vec3::ZERO, g.x_axis);
        }

        // the single turn: its axis, and the paths of the arrow tips about it
        let mut theta_at: Option<Vec<Vec3>> = None;
        let axis_shown = self.shows_arcs().then_some(whole.axis).flatten();
        if let Some(n) = axis_shown {
            let nf = n.as_vec3();
            // (on the dot's side it ends at the dot, which its name is written beyond)
            f.line(-nf * AXIS_LEN, nf * HANDLE_R, 2.2, TURN.a(0.9));
            room.line(f, -nf * AXIS_LEN, nf * HANDLE_R);
            let so_far = self.whole_so_far();
            // the paths of the arrow tips: the whole of each dashed, and while the turn
            // is played the part travelled so far solid over it. (The dashes are pieces
            // of the path itself, so that they stay where they are on it while the view
            // turns or the path is played.)
            for k in 0..3 {
                let e = DVec3::AXES[k];
                let centre = n * n.dot(e);
                if (e - centre).length() < 0.05 {
                    continue;
                }
                let pieces = ((e - centre).length() * whole.degrees.to_radians() / 0.06).ceil().clamp(4.0, 96.0) as usize;
                let path = |upto: f64| -> Vec<Vec3> { (0..=pieces).map(|i| (rot(n, upto * i as f64 / pieces as f64) * e).as_vec3()).collect() };
                let pts = path(whole.degrees);
                let playing = so_far < whole.degrees;
                if playing {
                    f.path(path(so_far), 1.8, COLS[k].a(0.9));
                }
                for w in pts.windows(2) {
                    f.dashed(w[0], w[1], 1.5, COLS[k].a(if playing { 0.4 } else { 0.75 }));
                    room.line(f, w[0], w[1]);
                }
            }
            // the angle, as a curled arrow round the axis near its dot, from a direction
            // across the axis to where the turn takes it
            if so_far > 0.5 {
                let (c, u) = (n * PROTRACTOR_AT, protractor_start(n));
                let at = |deg: f64| (c + rot(n, deg) * u * PROTRACTOR_R).as_vec3();
                let pts: Vec<Vec3> = (0..=36).map(|i| at(so_far * i as f64 / 36.0)).collect();
                f.path(pts.iter().copied(), 2.0, TURN);
                let back = (so_far * 0.25).min(14.0);
                f.arrow(at(so_far - back), at(so_far), 2.0, TURN);
                let cf = c.as_vec3();
                f.line(cf, at(0.0), 1.0, TURN.a(0.55));
                f.line(cf, at(so_far), 1.0, TURN.a(0.55));
                for w in pts.windows(2) {
                    room.line(f, w[0], w[1]);
                }
                room.line(f, cf, at(0.0));
                room.line(f, cf, at(so_far));
                theta_at = Some(vec![at(so_far * 0.5), at(so_far * 0.3), at(so_far * 0.7)]);
            }
        }

        // the lines each angle turns the aeroplane about
        let at_angles = self.drawn_angles().filter(|_| self.nudges);
        let nudge = at_angles.map(|a| nudge_axes(a).map(|n| n.as_vec3()));
        if let Some(lines) = nudge {
            for n in lines {
                f.dashed(-n * NUDGE_LEN, n * NUDGE_LEN, 1.8, NUDGE.a(0.9));
                room.line(f, -n * NUDGE_LEN, n * NUDGE_LEN);
            }
        }

        // where a play ends
        let end = (!rest).then(|| (self.pose() * flipped(if self.mirror { -1.0 } else { 1.0 })).as_mat3());
        if let Some(end) = end {
            outline(f, end, 1.4, STEP.a(0.55));
        }

        // the aeroplane itself, and its three arrows: the columns
        // (lighter beside the other order, which would otherwise be hidden behind it)
        let alpha = if ghost.is_some() { 0.32 } else { 0.55 };
        aeroplane(f, m, [COLS[1], PAPER, COLS[2]], alpha, pal::WHITE.a(0.85), 1.4);
        for (a, b) in EDGES {
            room.line(f, m * a, m * b);
        }
        for face in FACES {
            room.shape(f, &face.map(|p| m * p));
        }
        let cols = [m.x_axis, m.y_axis, m.z_axis];
        for k in 0..3 {
            f.arrow(Vec3::ZERO, cols[k], 3.0, COLS[k]);
            room.line(f, Vec3::ZERO, cols[k]);
        }

        // no turn at all: the line an angle would turn about, faint, to its dot
        if rest && self.plain() && whole.axis.is_none() {
            f.dashed(Vec3::ZERO, self.grip, 1.2, TURN.a(0.5));
        }
        if rest && self.plain() {
            room.line(f, self.grip, self.grip);
        }

        // the names, the most important first: the axis's beyond its dot, where there
        // is nothing else, and the columns' at their tips (or further down the arrow,
        // if the tip is crowded); written as they were in the last frame if nothing they
        // are placed from has changed since, and they had come to rest (see `Still`)
        self.inputs(names.begin(f));
        if names.settle(f) {
            names.replay(f, &mut room);
            return;
        }
        if let Some(n) = axis_shown {
            let tip = if rest { self.grip } else { n.as_vec3() * HANDLE_R };
            let text = if rest { "axis (drag the dot)" } else { "axis" };
            names.label(&mut room, f, "axis", tip, text, 13.0, TURN, outwards(f, Vec3::ZERO, tip), 10.0, false);
        } else if rest && self.plain() && whole.axis.is_none() {
            names.label(&mut room, f, "axis", self.grip, "no turn, so no axis", 13.0, TURN, outwards(f, Vec3::ZERO, self.grip), 10.0, false);
        }
        let name = if self.is_turn() { "R" } else { "M" };
        for k in 0..3 {
            let dir = outwards(f, Vec3::ZERO, cols[k]);
            names.label_any(&mut room, f, ("column", k), &[cols[k], cols[k] * 0.8, cols[k] * 0.6], &format!("${name}e_{}$", k + 1), 16.0, COLS[k], dir, 8.0, false);
        }
        if let (Some(points), Some(n)) = (theta_at, axis_shown) {
            let c = (n * PROTRACTOR_AT).as_vec3();
            let dir = outwards(f, c, points[0]);
            names.label_any(&mut room, f, "theta", &points, &format!("$\\theta = {}$", deg_tex(self.whole_so_far())), 14.0, TURN, dir, 6.0, false);
        }

        if let Some(g) = ghost {
            let at = g.x_axis;
            names.label(&mut room, f, "other", at, "other order", 13.0, OTHER, outwards(f, Vec3::ZERO, at), 8.0, false);
        }
        if let Some(lines) = nudge {
            let same = at_angles.is_some_and(|a| lines_apart(a) < 0.005);
            for (k, n) in lines.into_iter().enumerate() {
                if same && k == 0 {
                    continue;
                }
                let at = n * NUDGE_LEN;
                let text = if same && k == 2 { "1st and 3rd angles: one line".to_string() } else { format!("{} angle", NTH[k]) };
                names.label(&mut room, f, ("nudge", k), at, &text, 12.5, NUDGE, outwards(f, Vec3::ZERO, at), 6.0, false);
            }
        }
        if let Some(end) = end {
            let at = end * NOSE;
            names.label(&mut room, f, "end", at, "where it ends", 12.0, STEP.a(0.8), outwards(f, Vec3::ZERO, at), 8.0, false);
        }
        if let Some((u, v)) = mirror_at {
            let at = (u + v) * 0.95;
            names.label(&mut room, f, "mirror", at, "mirror", 13.0, GLASS, outwards(f, Vec3::ZERO, at), 6.0, false);
        }
        // (every name above is always written while what it names is drawn, so that
        // names come and go only with what they name)
        for k in 0..3 {
            let d = Vec3::AXES[k] * SPACE_LEN;
            names.label(&mut room, f, ("space", k), d, NAMES[k], 13.0, pal::DIM, outwards(f, Vec3::ZERO, d), 5.0, false);
        }
        // the turn being played, beside the axis it is about
        if let Some((_, axis)) = stepping {
            let d = Vec3::AXES[axis] * SPACE_LEN;
            names.label(&mut room, f, "step", d, &format!("{} turn: about {}", NTH[axis], NAMES[axis]), 13.0, STEP, outwards(f, Vec3::ZERO, d), 22.0, false);
        }
        names.end(f, false);
    }

    /// Everything the picture and its names are drawn from (see `Still`).
    fn inputs(&self, k: &mut Inputs) {
        k.f32s(&self.angles).f64s(&self.axis.to_array()).f32s(&[self.turn]).f32s(&self.grip.to_array());
        k.int(self.mirror as u64).int(self.other as u64).int(self.nudges as u64);
        match self.play {
            None => k.int(0),
            Some((play, t)) => k.int(1 + play as u64).f32s(&[t]),
        };
        match self.lead {
            None => k.int(0),
            Some(Drawn { turned, other, flip }) => k.int(1).f64s(&turned.to_array()).f64s(&other.to_array()).f64s(&[flip]),
        };
        k.f32s(&[self.canvas.x, self.canvas.y, self.inset]).int(self.measured as u64);
    }

    /// The panel over the picture: the buttons, the matrix and what it says.
    fn panel(&mut self, f: &mut Frame, lay: &Layout, pointer: Option<Vec2>) {
        let Some((min, max)) = lay.panel else { return };
        hud_panel(f, min, max);
        let x = min.x + 14.0;
        let mut y = min.y + 12.0;

        // the buttons
        if hud_button(f, Vec2::new(x, y), Vec2::new(x + 168.0, y + 26.0), "▶ the turns in order", self.can(Play::Steps), pointer) {
            self.start(Play::Steps, self.drawn());
        }
        if hud_button(f, Vec2::new(x + 176.0, y), Vec2::new(x + 300.0, y + 26.0), "▶ as one turn", self.can(Play::Whole), pointer) {
            self.start(Play::Whole, self.drawn());
        }
        y += 38.0;
        let made = if self.mirror { format!("mirrored, then {}", turns_text(self.angles, false)) } else { turns_text(self.angles, false) };
        // (smaller where the angles are long numbers, to stay in the panel; counted
        // without the dollars, the names of the axes being single letters)
        let room = max.x - x - 10.0;
        let letters = made.chars().filter(|c| *c != '$').count() as f32;
        let size = [12.5, 11.5, 10.5].into_iter().find(|s| letters * s * 0.52 <= room).unwrap_or(10.0);
        f.text_px(Vec2::new(x, y), made, size, pal::TEXT, Align2::LEFT_TOP);
        y += 24.0;

        // the matrix, its columns in the arrows' colours, with the diagonal ringed
        // while the trace is written under it
        let m = self.shown(false);
        let name = if self.is_turn() { "R" } else { "M" };
        // (cells a character wider than the numbers' six: set as a formula, they are
        // wider than the interface's digits, and need room between them)
        let (size, widest) = (13.0, 7);
        let at = Vec2::new(x + 34.0, y + 16.0);
        for j in 0..3 {
            let (lo, hi) = matrix_entry(at, widest, size, 0, j);
            f.text_px(Vec2::new((lo.x + hi.x) * 0.5, at.y - 3.0), format!("${name}e_{}$", j + 1), 12.0, COLS[j], Align2::CENTER_BOTTOM);
        }
        if self.is_turn() && turn_of(m, self.axis).axis.is_some() {
            for i in 0..3 {
                let (lo, hi) = matrix_entry(at, widest, size, i, i);
                hud_outline(f, lo + Vec2::new(3.0, 2.0), hi - Vec2::new(3.0, 2.0), 1.0, TURN.a(0.5));
            }
        }
        let rows: Vec<Vec<(String, Color)>> = (0..3).map(|i| (0..3).map(|j| (format!("${}$", num(m.col(j)[i], 3)), COLS[j])).collect()).collect();
        let total = hud_matrix(f, at, &rows, widest, size, pal::DIM);
        // (set against the bracket, as wide as it is set)
        f.text_px(Vec2::new(at.x - 6.0, at.y + total.y * 0.5), format!("${name} =$"), 14.0, pal::TEXT, Align2::RIGHT_CENTER);
        y = at.y + total.y + 10.0;

        // what the matrix says
        for (text, color) in self.panel_lines() {
            f.text_px(Vec2::new(x, y), text, 12.0, color, Align2::LEFT_TOP);
            y += PANEL_LINE;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// An experiment as its button sets it up (from the scene as it opens), played to
    /// the end at 60 frames a second.
    fn run(k: usize) -> Rotations {
        let mut s = Rotations::create();
        s.set_up(k);
        assert!(s.play.is_some(), "experiment {k} plays");
        for _ in 0..2000 {
            s.advance(1.0 / 60.0);
        }
        assert!(s.play.is_none() && s.lead.is_none() && !s.animated());
        s
    }

    /// The largest difference between the entries of two matrices.
    fn close(a: DMat3, b: DMat3) -> f64 {
        (a - b).to_cols_array().iter().fold(0.0, |w, v| w.max(v.abs()))
    }

    fn cols(m: DMat3) -> [DVec3; 3] {
        [m.x_axis, m.y_axis, m.z_axis]
    }

    /// Rotations of every kind: from random angles, about random axes by random angles
    /// (half turns and the locked settings among them), and the experiments'.
    fn samples() -> Vec<DMat3> {
        let mut rng = Rng::new(17);
        let mut out = Vec::new();
        for _ in 0..300 {
            let angles = [0, 1, 2].map(|k| rng.range(-REACH[k], REACH[k]));
            out.push(pose(angles));
            let axis = DVec3::new(rng.normal() as f64, rng.normal() as f64, rng.normal() as f64).normalize();
            out.push(rot(axis, rng.range(0.0, 180.0) as f64));
            out.push(rot(axis, 180.0));
        }
        for (b, a, c) in [(-90.0, 40.0, 0.0), (90.0, -30.0, 70.0), (-90.0, 180.0, -180.0), (90.0, 0.0, 0.0)] {
            out.push(pose([c, b, a]));
        }
        out
    }

    /// The captions of the "things to try", held to account.
    #[test]
    fn the_experiments_show_what_their_captions_say() {
        // order matters: x first leaves the nose along x, then the y turn points it
        // down; y first points it down, then the x turn swings it round to y; the two
        // ends are 120.0 degrees apart
        let s = run(ORDER_TRIAL);
        assert!(s.other && !s.nudges && !s.mirror && s.angles == [90.0, 90.0, 0.0]);
        let mut half = Rotations::create();
        half.set_up(ORDER_TRIAL);
        (half.play, half.lead) = (Some((Play::Steps, STEP_SECONDS + 0.5 * HOLD_SECONDS)), None);
        assert!((half.turned(false) * DVec3::X - DVec3::X).length() < 1e-12);
        assert!((half.turned(true) * DVec3::X - DVec3::NEG_Z).length() < 1e-12);
        assert!((s.pose() * DVec3::X - DVec3::NEG_Z).length() < 1e-12);
        assert!((other_pose(s.angles) * DVec3::X - DVec3::Y).length() < 1e-12);
        assert_eq!(deg(apart(s.pose(), other_pose(s.angles))), "120.0°");
        assert!(close(s.pose(), about(1, 90.0) * about(0, 90.0)) < 1e-15 && close(other_pose(s.angles), about(0, 90.0) * about(1, 90.0)) < 1e-15);

        // two turns make one: 120.0 degrees about the line through (1, 1, -1); every
        // point circles it by that angle; the play ends exactly where the two turns do;
        // tr R = 0 = 1 + 2 cos 120
        let s = run(ONE_TURN_TRIAL);
        assert!(!s.other && !s.nudges && !s.mirror && s.angles == [90.0, 90.0, 0.0] && s.shows_arcs());
        let whole = s.whole();
        let n = whole.axis.unwrap();
        assert!(deg(whole.degrees) == "120.0°" && (n - DVec3::new(1.0, 1.0, -1.0).normalize()).length() < 1e-12, "{whole:?}");
        let r = s.pose();
        assert!(close(rot(n, whole.degrees), r) < 1e-12 && (r * n - n).length() < 1e-12);
        let trace = r.x_axis.x + r.y_axis.y + r.z_axis.z;
        assert!(trace.abs() < 1e-12 && (1.0 + 2.0 * 120f64.to_radians().cos()).abs() < 1e-12);
        let mut rng = Rng::new(5);
        for _ in 0..50 {
            let p = DVec3::new(rng.normal() as f64, rng.normal() as f64, rng.normal() as f64);
            let (a, b) = (p - n * n.dot(p), r * p - n * n.dot(r * p));
            assert!((a.angle_between(b).to_degrees() - 120.0).abs() < 1e-6 && (a.length() - b.length()).abs() < 1e-12);
        }

        // gimbal lock: the nose straight up; the 1st and 3rd angles turn about the
        // same line, the nose, and only their sum counts; no small move of any slider
        // tips the nose towards a wing; off -90 degrees the lines part
        let s = run(LOCK_TRIAL);
        assert!(s.nudges && !s.other && s.angles == [0.0, -90.0, 40.0]);
        assert!((s.pose() * DVec3::X - DVec3::Z).length() < 1e-12);
        let lines = nudge_axes(s.angles);
        assert!((lines[0] - DVec3::Z).length() < 1e-12 && lines[2] == DVec3::Z && lines_apart(s.angles) < 1e-6);
        assert!(s.panel_lines().iter().any(|(t, _)| t == "the 1st and 3rd angles turn about one line"));
        for d in [5.0f32, -30.0, 75.0] {
            assert!(close(pose([d, -90.0, 40.0]), pose([0.0, -90.0, 40.0 + d])) < 1e-12, "{d}");
        }
        let wing = s.pose() * DVec3::Y;
        let h = 1e-3;
        for k in 0..3 {
            let mut a = s.angles;
            a[k] += h;
            let mut b = s.angles;
            b[k] -= h;
            // how fast the nose moves, per degree of the slider
            let v = (pose(a) * DVec3::X - pose(b) * DVec3::X) / (2.0 * h as f64);
            assert!(v.dot(wing).abs() < 1e-7, "slider {k}: {v}");
            if k == 1 {
                assert!(v.length() > 0.017, "slider 2 tips the nose: {v}");
            } else {
                assert!(v.length() < 1e-7, "slider {k} leaves the nose where it is: {v}");
            }
        }
        let mut off = s.angles;
        off[1] = -80.0;
        assert_eq!(deg(lines_apart(off)), "10.0°");

        // a mirror is not a turn: columns of length 1 at right angles, det -1, the third
        // against the cross product of the first two; flat half-way; no turn brings the
        // green wing back to the left
        let s = run(MIRROR_TRIAL);
        assert!(s.mirror && !s.other && !s.nudges);
        let m = s.shown(false);
        let [c1, c2, c3] = cols(m);
        assert!(off_orthonormal(m) < 1e-15 && num(m.determinant(), 3) == "−1.000" && (c3.dot(c1.cross(c2)) + 1.0).abs() < 1e-12);
        assert!(s.panel_lines().iter().any(|(t, _)| t == r"$\det M = −1.000$: a mirror image, not a turn"));
        let mut s = Rotations::create();
        s.set_up(MIRROR_TRIAL);
        let mut was = f64::MAX;
        let mut flat = false;
        while s.play.is_some() {
            if s.leading().is_none() {
                let det = s.shown(false).determinant();
                assert!(det <= was + 1e-12 && (det - s.flip()).abs() < 1e-12, "{det} after {was}");
                was = det;
                if let Some((_, t)) = s.playing() {
                    flat |= (t - SQUASH_SECONDS * 0.5).abs() < 0.01 && det.abs() < 0.02;
                }
            }
            s.advance(1.0 / 240.0);
        }
        assert!(flat && was < -0.99 && (s.shown(false).determinant() + 1.0).abs() < 1e-12);
        (s.play, s.lead) = (Some((Play::Squash, SQUASH_SECONDS * 0.5)), None);
        assert!(s.shown(false).determinant().abs() < 1e-12);
        let mut rng = Rng::new(8);
        for _ in 0..200 {
            let angles = [0, 1, 2].map(|k| rng.range(-REACH[k], REACH[k]));
            assert!((pose(angles).determinant() - 1.0).abs() < 1e-12);
            s.angles = angles;
            s.play = None;
            let [nose, wing, fin] = cols(s.shown(false));
            assert!((s.shown(false).determinant() + 1.0).abs() < 1e-12 && wing.dot(fin.cross(nose)) < -0.999, "{angles:?}");
        }

        // small turns nearly commute: 1.74 degrees apart, about 0.175 x 0.175 = 0.0305
        // rad = 1.75 degrees; at 5 degrees each a quarter of that, 0.44
        let s = run(SMALL_TRIAL);
        assert!(s.other && s.angles == [10.0, 10.0, 0.0]);
        let gap = apart(s.pose(), other_pose(s.angles));
        let (a, b) = s.small_pair().unwrap();
        assert!(deg(gap) == "1.74°" && num(a, 3) == "0.175" && num(b, 3) == "0.175" && num(a * b, 4) == "0.0305" && deg((a * b).to_degrees()) == "1.75°");
        let five = apart(pose([5.0, 5.0, 0.0]), other_pose([5.0, 5.0, 0.0]));
        assert!(deg(five) == "0.44°" && (gap / five - 4.0).abs() < 0.02, "{five} {}", gap / five);
        assert!(s.panel_lines().iter().any(|(t, _)| t == r"about angle $\times$ angle: $0.175 \times 0.175\,\text{rad} = 1.75^\circ$"));
        // (and the line is left out for quarter turns, where it does not hold)
        assert!(run(ORDER_TRIAL).small_pair().is_none());
    }

    /// Every rotation keeps lengths and angles; its columns have length 1, are at
    /// right angles and turn in the order of x, y, z (det 1); it is one turn about one
    /// axis, which it leaves where it is; and its trace is 1 + 2 cos of that angle.
    #[test]
    fn every_rotation_is_one_turn_about_one_axis() {
        let mut rng = Rng::new(3);
        for r in samples() {
            let [c1, c2, c3] = cols(r);
            assert!(off_orthonormal(r) < 1e-14 && (r.determinant() - 1.0).abs() < 1e-14 && (c1.cross(c2) - c3).length() < 1e-14);
            let (u, v) = (DVec3::new(rng.normal() as f64, rng.normal() as f64, 0.3), DVec3::new(0.5, rng.normal() as f64, rng.normal() as f64));
            assert!(((r * u).length() - u.length()).abs() < 1e-12 && ((r * u).angle_between(r * v) - u.angle_between(v)).abs() < 1e-7);
            let t = turn_of(r, DVec3::X);
            let n = t.axis.expect("a turn");
            assert!((0.0..=180.0).contains(&t.degrees) && (n.length() - 1.0).abs() < 1e-12);
            assert!((r * n - n).length() < 1e-9 && close(rot(n, t.degrees), r) < 1e-9, "{r}: {t:?}");
            let trace = r.x_axis.x + r.y_axis.y + r.z_axis.z;
            assert!((trace - (1.0 + 2.0 * t.degrees.to_radians().cos())).abs() < 1e-12);
            // the two orders of the same turns end at rotations too
            let angles = angles_of(r, 0.0);
            assert!(off_orthonormal(other_pose(angles)) < 1e-14);
        }
    }

    /// Three angles, the middle one from −90 to 90 degrees, reach every rotation, also
    /// where two of them turn about the same line.
    #[test]
    fn three_angles_reach_every_rotation() {
        for r in samples() {
            for first in [0.0f32, 33.0, -170.0] {
                let angles = angles_of(r, first);
                assert!((0..3).all(|k| angles[k].abs() <= REACH[k]), "{angles:?}");
                assert!(close(pose(angles), r) < 1e-5, "{r}: {angles:?}");
            }
        }
        // at the lock the first angle is kept, and the third makes up the rest
        let locked = pose([30.0, -90.0, 40.0]);
        let angles = angles_of(locked, 30.0);
        assert!(angles_of(DMat3::IDENTITY, 0.0).iter().all(|a| *a == 0.0 && a.is_sign_positive()));
        assert!(angles[1] == -90.0 && (angles[0] - 30.0).abs() < 1e-4 && (angles[2] - 40.0).abs() < 1e-4, "{angles:?}");
        // away from it the lines the angles turn about are three different lines
        for b in [-89.0f32, -45.0, 0.0, 60.0, 89.0] {
            assert!((lines_apart([10.0, b, 20.0]) - (90.0 - b.abs() as f64)).abs() < 1e-6);
        }
        assert!(lines_apart([10.0, 90.0, 20.0]) < 1e-6);
    }

    /// No turn has no axis; a half turn still has one, found in full precision, and
    /// it points the way it pointed before; nearly half a turn, nearly none, and the
    /// sliders at their ends all come out finite and right.
    #[test]
    fn half_turns_and_no_turn() {
        assert!(turn_of(DMat3::IDENTITY, DVec3::X).axis.is_none() && turn_of(pose([0.0; 3]), DVec3::X).degrees == 0.0);
        // three half turns about x, y and z are no turn at all
        let none = turn_of(about(2, 180.0) * about(1, 180.0) * about(0, 180.0), DVec3::X);
        assert!(none.axis.is_none() || none.degrees < 1e-6, "{none:?}");
        let mut rng = Rng::new(21);
        let mut axes = vec![DVec3::X, DVec3::Y, DVec3::Z, DVec3::ONE.normalize(), DVec3::new(1.0, 1.0, 0.0).normalize(), DVec3::new(1.0, -1.0, 1e-9).normalize()];
        axes.extend((0..100).map(|_| DVec3::new(rng.normal() as f64, rng.normal() as f64, rng.normal() as f64).normalize()));
        for n in axes {
            for degrees in [180.0, 179.9999, 0.0001, 1e-8, 90.0] {
                let r = rot(n, degrees);
                let t = turn_of(r, n);
                let got = t.axis.expect("a turn");
                assert!((got - n).length() < 1e-6 && (t.degrees - degrees).abs() < 1e-6, "{n} by {degrees}: {t:?}");
                // and at a half turn the other way along the line, if that is preferred
                if degrees == 180.0 {
                    let back = turn_of(r, -n).axis.unwrap();
                    assert!((back + n).length() < 1e-9 && (r * back - back).length() < 1e-12);
                }
            }
        }
        // the sliders at their ends
        for angles in [[180.0, 90.0, 180.0], [-180.0, -90.0, -180.0], [180.0, -90.0, 0.0], [0.0, 90.0, -180.0]] {
            let mut s = Rotations::create();
            s.angles = angles;
            s.follow_angles();
            assert!(s.axis.is_finite() && s.turn.is_finite() && s.grip.is_finite() && off_orthonormal(s.pose()) < 1e-14);
            for line in s.panel_lines() {
                assert!(!line.0.contains('?') && !line.0.contains("NaN") && !line.0.contains("inf"), "{line:?}");
            }
        }
    }

    /// The axis and its angle, given in place of the three angles: the angles follow
    /// and make that turn; no angle at all is no turn, and the axis is kept for the
    /// next angle; a matrix that is not finite comes to nothing finite-breaking.
    #[test]
    fn the_axis_and_the_angles_keep_in_step() {
        let mut s = Rotations::create();
        let n = DVec3::new(0.3, -0.5, 0.8).normalize();
        for degrees in [0.0f32, 35.0, 120.0, 180.0] {
            s.set_turn(n, degrees);
            assert!(close(s.pose(), rot(n, degrees as f64)) < 1e-5, "{degrees}");
            let t = s.whole();
            if degrees == 0.0 {
                assert!(t.axis.is_none() && s.angles == [0.0; 3] && s.axis == n);
            } else {
                assert!((t.axis.unwrap() - n).length() < 1e-5 && (t.degrees - degrees as f64).abs() < 1e-4, "{t:?}");
            }
        }
        // and the other way: the sliders move the axis
        s.angles = [0.0, 0.0, 50.0];
        s.follow_angles();
        assert!((s.axis - DVec3::Z).length() < 1e-12 && (s.turn - 50.0).abs() < 1e-4 && (s.grip - Vec3::Z * HANDLE_R).length() < 1e-6);
        let bad = DMat3::from_cols(DVec3::splat(f64::NAN), DVec3::Y, DVec3::Z);
        assert!(turn_of(bad, DVec3::X).axis.is_none() && angles_of(bad, 0.0).iter().all(|a| a.is_finite()));
    }

    /// Each play starts by gliding back to its start the short way, moves on smoothly
    /// at any display rate (no frame turns it by much, also where one part of a play
    /// hands over to the next), ends exactly where the angles put it, and then rests.
    #[test]
    fn the_plays_move_smoothly_and_end_where_they_should() {
        for hz in [60.0f32, 144.0] {
            for k in 0..TRIALS.len() {
                let mut s = Rotations::create();
                s.set_up(k);
                assert!(s.lead.is_some(), "experiment {k} starts with a glide back from where the scene opened");
                let mut before = s.shown(false);
                let (mut frames, mut most) = (0, 0.0f64);
                while s.play.is_some() {
                    s.advance(1.0 / hz);
                    let now = s.shown(false);
                    // (the cyan lines are drawn for the angles the aeroplane is drawn at)
                    if let Some(angles) = s.drawn_angles() {
                        assert!(close(pose(angles), s.turned(false)) < 1e-5, "experiment {k}: {angles:?}");
                    }
                    // (in degrees of turn, or as entries of a squashed matrix)
                    let step = if s.is_turn() && before.determinant() > 0.999 { apart(before, now) } else { close(before, now).to_degrees() };
                    most = most.max(step);
                    before = now;
                    frames += 1;
                }
                let end = s.pose() * flipped(if s.mirror { -1.0 } else { 1.0 });
                assert!(s.shown(false) == end && close(before, end) < 1e-12 && !s.animated());
                // the fastest of them, the glide back, turns at most about 340 degrees a second
                assert!(most < 360.0 / hz as f64 && frames as f32 > hz * 2.0, "experiment {k} at {hz} Hz: {most} degrees in a frame, {frames} frames");
            }
        }
        // the single turn keeps its axis and grows its angle from 0 to all of it
        let mut s = Rotations::create();
        let whole = s.whole();
        s.start(Play::Whole, s.drawn());
        let mut was = -1.0;
        while s.play.is_some() {
            if s.leading().is_none() {
                let t = turn_of(s.turned(false), whole.axis.unwrap());
                assert!(t.degrees >= was && t.axis.is_none_or(|a| (a - whole.axis.unwrap()).length() < 1e-9), "{t:?}");
                was = t.degrees;
            }
            s.advance(1.0 / 60.0);
        }
        assert!((was - whole.degrees).abs() < 1.0);
        // nothing moves while no time passes
        let mut s = Rotations::create();
        s.set_up(ORDER_TRIAL);
        let still = s.shown(false);
        s.advance(0.0);
        assert_eq!(s.shown(false), still);
    }

    /// The panel and the picture share every canvas: the panel stays inside it and
    /// clear of the application's buttons, the round part for the picture stays inside
    /// it and off the panel, and the view made for it is a view.
    #[test]
    fn the_canvas_is_shared_on_every_screen() {
        for (x, y) in [(1080.0f32, 1000.0f32), (1100.0, 700.0), (390.0, 800.0), (360.0, 640.0), (600.0, 900.0), (844.0, 390.0), (640.0, 480.0), (240.0, 180.0), (2000.0, 1300.0), (8.0, 8.0)] {
            let view = Vec2::new(x, y);
            let inset = if x < 900.0 { 46.0 } else { 0.0 };
            let lay = Layout::of(view, inset, PANEL_H);
            if let Some((lo, hi)) = lay.panel {
                assert!(lo.x >= 0.0 && lo.y >= inset && hi.x <= x && hi.y <= y, "{x}x{y}: {lo} {hi}");
                assert!(hi.y - lo.y >= PANEL_TOP + 7.0 * PANEL_LINE, "{x}x{y}: the panel holds its longest text");
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
            let mut s = Rotations::create();
            (s.canvas, s.inset) = (view, inset);
            for k in 0..TRIALS.len() {
                s.set_up(k);
                assert!(matches!(s.the_view(), View::Orbit { target, dist, .. } if target.is_finite() && dist.is_finite() && dist > 0.0), "{x}x{y}");
                assert!(s.panel_height() <= PANEL_H);
            }
        }
    }

    /// The names kept from frame to frame (`Still`) follow every change: run side by
    /// side with a copy that places every name afresh in every frame, the scene draws
    /// exactly the same in every frame, through drags of the yellow dot, of the view and
    /// of a slider, both plays, every experiment, a zoom, a pause, a phone's window and
    /// its buttons, and the scene opened again.
    #[test]
    fn what_is_kept_follows_every_change() {
        use super::super::subspaces::tried::{labels_follow, Step::*};
        let steps = [
            Rest(30),
            Drag([1055.0, 296.0], [980.0, 240.0]),
            Rest(30),
            Click([343.0, 38.0]),
            Rest(330),
            Click([497.0, 38.0]),
            Rest(230),
            Drag([1000.0, 820.0], [1100.0, 780.0]),
            Rest(40),
            Scroll([800.0, 500.0], -200.0),
            Rest(30),
            Try(0),
            Rest(400),
            Try(1),
            Rest(250),
            Try(2),
            Rest(120),
            Try(3),
            Rest(220),
            Try(4),
            Rest(60),
            Drag([1470.0, 993.0], [1520.0, 993.0]),
            Rest(30),
            Pause,
            Click([497.0, 38.0]),
            Rest(20),
            Pause,
            Rest(30),
            Size([390.0, 800.0]),
            Rest(30),
            Click([103.0, 527.0]),
            Rest(120),
            Drag([257.0, 117.0], [200.0, 100.0]),
            Rest(30),
            Size([1600.0, 1000.0]),
            Rest(30),
            Away,
            Rest(30),
        ];
        if let Some(again) = labels_follow("rotations", &steps) {
            assert!(again > 200, "the names were written again as they were only {again} times");
        }
    }
}

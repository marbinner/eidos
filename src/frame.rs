//! What a scene sees each frame: time, pointer, camera, and a convenient drawing API
//! on top of the raw draw list (arrows with pixel-sized heads, draggable handles,
//! labelled axes, text anchored to world positions).

use crate::gfx::camera::Camera;
use crate::gfx::color::{pal, Color};
use crate::gfx::draw::{DrawList, Label};
use crate::gfx::{Gfx, Image, Surface, TrailSet};
use egui::Align2;
use glam::{Vec2, Vec3};

/// Anything usable as a world position; 2D points live on the plane z = 0.
pub trait P3: Copy {
    fn p3(self) -> Vec3;
}

impl P3 for Vec3 {
    #[inline]
    fn p3(self) -> Vec3 {
        self
    }
}

impl P3 for Vec2 {
    #[inline]
    fn p3(self) -> Vec3 {
        self.extend(0.0)
    }
}

impl P3 for (f32, f32) {
    #[inline]
    fn p3(self) -> Vec3 {
        Vec3::new(self.0, self.1, 0.0)
    }
}

impl P3 for (f32, f32, f32) {
    #[inline]
    fn p3(self) -> Vec3 {
        Vec3::new(self.0, self.1, self.2)
    }
}

/// Pointer state in target pixels, relative to the viewport.
#[derive(Clone, Copy, Debug, Default)]
pub struct Pointer {
    pub pos: Option<Vec2>,
    pub down: bool,
    pub pressed: bool,
    pub released: bool,
    pub clicked: bool,
}

/// Draggable-handle bookkeeping. "Hot" is decided one frame late (nearest handle to
/// the pointer), which is what lets handles win over camera dragging.
#[derive(Default)]
pub struct Handles {
    active: Option<u32>,
    hot: Option<u32>,
    next_hot: Option<(u32, f32)>,
    grab: Vec3,
    press_age: u32,
    captured: bool,
    capture_next: bool,
    claimed: bool,
    claim_next: bool,
    offered: bool,
    offered_next: bool,
}

impl Handles {
    pub fn begin_frame(&mut self, ptr: &Pointer) {
        self.hot = self.next_hot.take().map(|h| h.0);
        self.captured = std::mem::take(&mut self.capture_next);
        self.claimed = std::mem::take(&mut self.claim_next);
        self.offered = std::mem::take(&mut self.offered_next);
        if ptr.pressed {
            self.press_age = 0;
        } else {
            self.press_age = self.press_age.saturating_add(1);
        }
        if !ptr.down {
            self.active = None;
        }
    }

    /// Called by a scene every frame it wants pointer drags for itself (free-hand
    /// drawing, poking a surface) instead of moving the camera.
    pub fn capture(&mut self) {
        self.capture_next = true;
    }

    pub fn captured(&self) -> bool {
        self.captured
    }

    /// Called by a scene every frame in which a click on empty canvas does something
    /// (adds a point, starts a path): a double-click then reaches the scene as two
    /// clicks, instead of resetting the view (`R` still does that).
    pub fn claim_clicks(&mut self) {
        self.claim_next = true;
    }

    pub fn clicks_claimed(&self) -> bool {
        self.claimed
    }

    /// Did the scene offer a handle to drag, in this frame or the one before? (So
    /// that the hint over the picture only speaks of coloured dots where there are some.)
    pub fn offered(&self) -> bool {
        self.offered || self.offered_next
    }

    /// Is a handle (not just a capture) under the pointer or being dragged?
    pub fn on_handle(&self) -> bool {
        self.active.is_some() || self.hot.is_some()
    }

    pub fn dragging(&self) -> bool {
        self.active.is_some()
    }
}

/// How long a label takes to glide to a new place: the time in which it covers about
/// two thirds of the way, in seconds.
const LABEL_GLIDE: f32 = 0.07;

/// Where the labels that look for a clear place were put, kept from frame to frame.
///
/// A label that looks for room each frame, with nothing to remember the last frame
/// by, jumps to and fro whenever two places are about as clear as each other, which
/// while something is dragged is all the time. Remembering does two things: the
/// scene can ask where a label was ([`Places::chosen`]) and leave it there while that
/// is still clear enough, and a label whose place does change is drawn gliding to
/// it ([`Frame::text_placed`]).
///
/// A place is an offset in logical pixels from a point on the screen that moves with
/// whatever the label names, so a label carried along by its arrow has not moved.
/// Labels are told apart by a name and a number of the scene's choosing.
#[derive(Default)]
pub struct Places {
    kept: std::collections::HashMap<(&'static str, u32), Place>,
    frame: u64,
    gliding: bool,
}

struct Place {
    /// Where the scene last put the label, and where it was last drawn.
    chosen: Vec2,
    drawn: Vec2,
    /// The frame it was last drawn in.
    seen: u64,
}

impl Places {
    pub fn begin_frame(&mut self) {
        // (a label not drawn in the last frame has no place to keep or glide from)
        let last = self.frame;
        self.kept.retain(|_, place| place.seen == last);
        self.frame += 1;
        self.gliding = false;
    }

    /// Where the label was put in the last frame, if it was drawn then.
    pub fn chosen(&self, name: &'static str, number: u32) -> Option<Vec2> {
        self.kept.get(&(name, number)).map(|place| place.chosen)
    }

    /// Is a label still on its way? (Then another frame is wanted, even of a scene at rest.)
    pub fn gliding(&self) -> bool {
        self.gliding
    }

    /// Put the label at `offset`, and say where to draw it now: `dt` seconds nearer
    /// to there than it was drawn last (there at once, if it was not drawn last frame).
    fn put(&mut self, name: &'static str, number: u32, offset: Vec2, dt: f32) -> Vec2 {
        let frame = self.frame;
        let place = self.kept.entry((name, number)).or_insert(Place { chosen: offset, drawn: offset, seen: frame });
        place.chosen = offset;
        place.seen = frame;
        place.drawn += (offset - place.drawn) * (1.0 - (-dt.max(0.0) / LABEL_GLIDE).exp());
        // (exactly there in the end: half a pixel is nothing to glide)
        if !place.drawn.is_finite() || (offset - place.drawn).length() < 0.5 {
            place.drawn = offset;
        } else {
            self.gliding = true;
        }
        place.drawn
    }
}

pub struct Frame<'a> {
    /// The interface's context: its fonts, for measuring text exactly ([`Frame::text_size`]).
    pub ctx: &'a egui::Context,
    pub gfx: &'a Gfx,
    pub dl: &'a mut DrawList,
    pub cam: &'a Camera,
    pub handles: &'a mut Handles,
    /// Where the labels that look for a clear place were put last frame.
    pub places: &'a mut Places,
    /// Scene time in seconds; stops while paused and follows the speed control.
    pub t: f32,
    /// Scene time step for this frame (0 while paused).
    pub dt: f32,
    /// Wall-clock time step.
    pub real_dt: f32,
    pub ptr: Pointer,
    /// Target pixels per logical pixel.
    pub px: f32,
    /// Logical pixels at the top of the view that the application's own buttons and
    /// the scene's title lie over (when the panels are not docked beside the
    /// picture: a narrow screen, or the panels hidden). A scene that draws its own
    /// headings at the top of the view starts below this.
    pub top_inset: f32,
    /// The map from logical pixels to points just in front of the camera, for the
    /// three levels of [`Frame::hud_at`]: for each, the point at pixel (0, 0) and
    /// the steps per pixel across and down. Made by [`Frame::hud_planes`].
    pub hud: [[Vec3; 3]; 3],
}

impl Frame<'_> {
    // ----- lines -----------------------------------------------------------------

    #[inline]
    pub fn line(&mut self, a: impl P3, b: impl P3, width: f32, color: Color) {
        self.dl.lines.segment(a.p3(), b.p3(), width, color);
    }

    pub fn path<P: P3>(&mut self, pts: impl IntoIterator<Item = P>, width: f32, color: Color) {
        self.dl.lines.polyline(pts.into_iter().map(P3::p3), width, color);
    }

    /// A curve that glows (additive, feeds the bloom).
    pub fn glow_path<P: P3>(&mut self, pts: impl IntoIterator<Item = P>, width: f32, color: Color) {
        self.dl.glow_lines.polyline(pts.into_iter().map(P3::p3), width, color);
    }

    #[inline]
    pub fn dashed(&mut self, a: impl P3, b: impl P3, width: f32, color: Color) {
        self.dl.dashed.segment(a.p3(), b.p3(), width, color);
    }

    /// A dashed curve. ([`Frame::dashed`] starts its pattern afresh at every segment,
    /// which is right for a straight line and leaves a curve of short segments
    /// solid; this measures the same pattern along the whole curve as it lies on the
    /// screen.) A point that is not finite, or is behind the camera, breaks the curve.
    pub fn dashed_path<P: P3>(&mut self, pts: impl IntoIterator<Item = P>, width: f32, color: Color) {
        let (cam, period) = (self.cam, 9.0 * self.px);
        let pts = pts.into_iter().map(|p| (p.p3(), cam.project(p.p3())));
        self.dl.lines.dashes(pts, period, 0.55 * period, cam.viewport, width, color);
    }

    /// Sample `f` over `[t0, t1]` into a polyline, breaking at non-finite values.
    pub fn curve(&mut self, t0: f32, t1: f32, n: usize, width: f32, color: Color, mut f: impl FnMut(f32) -> Vec3) {
        let b = &mut self.dl.lines;
        b.brk();
        for i in 0..=n {
            let t = t0 + (t1 - t0) * i as f32 / n as f32;
            let p = f(t);
            if p.is_finite() {
                b.push(p, width, color);
            } else {
                b.brk();
            }
        }
        b.brk();
    }

    /// Arrow from `a` to `b` with a head whose size is fixed in pixels.
    pub fn arrow(&mut self, a: impl P3, b: impl P3, width: f32, color: Color) {
        let (a, b) = (a.p3(), b.p3());
        let d = b - a;
        let len = d.length();
        if len < 1e-9 {
            return;
        }
        let dir = d / len;
        let wpp = self.cam.world_per_px(b) * self.px;
        let head_len = ((width * 3.2 + 5.0) * wpp).min(len * 0.6);
        let head_w = head_len * 0.42;
        let to_cam = if self.cam.is_flat() { Vec3::Z } else { (self.cam.eye - b).normalize_or_zero() };
        let mut side = dir.cross(to_cam);
        if side.length() < 1e-4 {
            side = dir.cross(self.cam.up);
        }
        let side = side.normalize_or_zero();
        let base = b - dir * head_len;
        self.dl.lines.segment(a, b - dir * (head_len * 0.8), width, color);
        self.dl.tris_alpha.tri(b, base + side * head_w, base - side * head_w, color);
    }

    pub fn circle(&mut self, center: impl P3, normal: Vec3, radius: f32, width: f32, color: Color) {
        let c = center.p3();
        let n = normal.normalize_or_zero();
        let u = n.any_orthonormal_vector();
        let v = n.cross(u);
        let pts: Vec<Vec3> = (0..96)
            .map(|i| {
                let a = i as f32 / 96.0 * std::f32::consts::TAU;
                c + (u * a.cos() + v * a.sin()) * radius
            })
            .collect();
        self.dl.lines.closed(&pts, width, color);
    }

    // ----- points ----------------------------------------------------------------

    /// Flat disc, diameter in logical pixels.
    #[inline]
    pub fn dot(&mut self, p: impl P3, size: f32, color: Color) {
        self.dl.discs.push(p.p3(), size, color);
    }

    /// Lit sphere, diameter in world units.
    #[inline]
    pub fn sphere(&mut self, p: impl P3, size: f32, color: Color) {
        self.dl.spheres.push(p.p3(), size, color);
    }

    /// Soft additive glow, diameter in logical pixels.
    #[inline]
    pub fn glow(&mut self, p: impl P3, size: f32, color: Color) {
        self.dl.glows.push(p.p3(), size, color);
    }

    // ----- retained objects ------------------------------------------------------

    pub fn surface(&mut self, s: &mut Surface) {
        let d = s.prepare(self.gfx);
        self.dl.surfaces.push(d);
    }

    /// A whole picture filling the rectangle `min..max` (world units) at depth `z`.
    pub fn image(&mut self, image: &mut Image, min: Vec2, max: Vec2, z: f32) {
        let [w, h] = image.size();
        image.prepare(self.gfx, self.dl, min, max, [0, 0, w, h], z);
    }

    /// The block of pixels `part` = [x, y, width, height] of a picture, filling the
    /// rectangle `min..max`. Consecutive parts of the same picture are one draw call.
    pub fn image_part(&mut self, image: &mut Image, part: [u32; 4], min: Vec2, max: Vec2, z: f32) {
        image.prepare(self.gfx, self.dl, min, max, part, z);
    }

    pub fn trails(&mut self, t: &mut TrailSet) {
        if let Some(d) = t.prepare(self.gfx) {
            self.dl.trails.push(d);
        }
    }

    // ----- text ------------------------------------------------------------------

    /// Text anchored at a world position.
    pub fn text(&mut self, p: impl P3, text: impl Into<String>, size: f32, color: Color, align: Align2) {
        if let Some(s) = self.cam.project(p.p3()) {
            self.dl.labels.push(Label { pos: s / self.px, text: text.into(), size, color, align, backed: false });
        }
    }

    /// Text at a world position, nudged by `offset` logical pixels.
    pub fn text_off(&mut self, p: impl P3, offset: Vec2, text: impl Into<String>, size: f32, color: Color, align: Align2) {
        if let Some(s) = self.cam.project(p.p3()) {
            self.dl.labels.push(Label { pos: s / self.px + offset, text: text.into(), size, color, align, backed: false });
        }
    }

    /// Text at a position in logical pixels from the viewport's top-left corner.
    pub fn text_px(&mut self, pos: Vec2, text: impl Into<String>, size: f32, color: Color, align: Align2) {
        self.dl.labels.push(Label { pos, text: text.into(), size, color, align, backed: false });
    }

    /// Text centred `offset` logical pixels from the screen point `from` (logical
    /// pixels too): for a label that looks for a clear place beside something. `from`
    /// is a point that moves with what the label names, and `name` and `number` tell
    /// the label from the scene's others. When its offset is not what it was in the
    /// last frame, the label is drawn gliding to its new place instead of jumping
    /// there (see [`Places`], which also says where it was: `f.places.chosen`).
    #[allow(clippy::too_many_arguments)]
    pub fn text_placed(&mut self, name: &'static str, number: u32, from: Vec2, offset: Vec2, text: impl Into<String>, size: f32, color: Color) {
        // (by the clock, like the view's own glide: it must go on while the scene's time stands still)
        let drawn = self.places.put(name, number, offset, self.real_dt);
        self.dl.labels.push(Label { pos: from + drawn, text: text.into(), size, color, align: Align2::CENTER_CENTER, backed: false });
    }

    /// Text at a screen position on a dark plate: for a title or caption that lies
    /// over the picture, where curves and axes would otherwise run through it.
    pub fn caption_px(&mut self, pos: Vec2, text: impl Into<String>, size: f32, color: Color, align: Align2) {
        self.dl.labels.push(Label { pos, text: text.into(), size, color, align, backed: true });
    }

    /// The size, in logical pixels, that `text` takes as a label at `size`: exactly as
    /// it is drawn, formulas and all. (Cheap for a text drawn in the last frame, whose
    /// layout is kept; `kit::text_width` is an estimate that needs no frame.)
    pub fn text_size(&self, text: &str, size: f32) -> Vec2 {
        if crate::typeset::has_math(text) {
            if let Some(s) = crate::typeset::measure(self.ctx, text, size) {
                return Vec2::new(s.x, s.y);
            }
        }
        let s = self.ctx.fonts_mut(|f| f.layout_no_wrap(text.to_string(), egui::FontId::proportional(size), egui::Color32::WHITE)).size();
        Vec2::new(s.x, s.y)
    }

    /// A point just in front of the camera that appears at logical pixel `px`
    /// (measured from the top-left corner of the view). Geometry placed with this is
    /// in effect drawn in screen space, in front of the scene: charts and legends
    /// inside a 3D view, which stay put whatever the camera does. The same as
    /// `hud_at(0, px)`.
    pub fn hud(&self, px: Vec2) -> Vec3 {
        self.hud_at(0, px)
    }

    /// The same on one of three flat sheets, each in front of the last, so that
    /// what is drawn on a higher level covers what is on a lower one: 0 for the
    /// backing of a panel, 1 for what lies on it, 2 for marks on that.
    pub fn hud_at(&self, level: usize, px: Vec2) -> Vec3 {
        let [origin, across, down] = self.hud[level.min(2)];
        origin + across * px.x + down * px.y
    }

    /// The three sheets of [`Frame::hud_at`] for a camera: worked out once a frame,
    /// after which a point costs two multiplications instead of a ray through the
    /// camera. `px` is target pixels per logical pixel.
    pub fn hud_planes(cam: &Camera, px: f32) -> [[Vec3; 3]; 3] {
        // (points at a fixed share of the way from the near plane to the far one
        // lie on a plane, in a flat view and in perspective alike)
        let ends = |at: Vec2| {
            let ndc = Vec2::new(at.x / cam.viewport.x.max(1.0) * 2.0 - 1.0, 1.0 - at.y / cam.viewport.y.max(1.0) * 2.0);
            (cam.inv_view_proj.project_point3(ndc.extend(0.0)), cam.inv_view_proj.project_point3(ndc.extend(1.0)))
        };
        let (near, far) = ends(cam.viewport * 0.5);
        let depth = (far - near).length().max(1e-12);
        [0.6, 0.45, 0.3].map(|level| {
            let share = cam.near * level / depth;
            let point = |at: Vec2| {
                let (a, b) = ends(at);
                a + (b - a) * share
            };
            let origin = point(Vec2::ZERO);
            [origin, (point(Vec2::new(1000.0, 0.0)) - origin) / 1000.0 * px, (point(Vec2::new(0.0, 1000.0)) - origin) / 1000.0 * px]
        })
    }

    /// Viewport size in logical pixels.
    pub fn view_size(&self) -> Vec2 {
        self.cam.viewport / self.px
    }

    // ----- interaction -----------------------------------------------------------

    /// Pointer position on the plane z = 0.
    pub fn mouse(&self) -> Option<Vec2> {
        self.ptr.pos.and_then(|p| self.cam.pick_ground(p)).map(|p| p.truncate())
    }

    /// True when the canvas (not a handle) was clicked without dragging.
    pub fn clicked_canvas(&self) -> bool {
        self.ptr.clicked && !self.handles.on_handle()
    }

    fn drag_impl(&mut self, id: u32, p: &mut Vec3, normal: Vec3, color: Color, size: f32) -> bool {
        let Some(sp) = self.cam.project(*p) else { return false };
        self.handles.offered_next = true;
        if let Some(ptr) = self.ptr.pos {
            let d = (ptr - sp).length() / self.px;
            if d < size * 0.5 + 9.0 && self.handles.next_hot.is_none_or(|(_, best)| d < best) {
                self.handles.next_hot = Some((id, d));
            }
        }
        let hot = self.handles.hot == Some(id);
        if hot && self.ptr.down && self.handles.press_age <= 1 && self.handles.active.is_none() {
            self.handles.active = Some(id);
            let hit = self.ptr.pos.and_then(|q| self.cam.pick_plane(q, *p, normal)).unwrap_or(*p);
            self.handles.grab = *p - hit;
        }
        let active = self.handles.active == Some(id);
        let mut changed = false;
        if active {
            if let Some(hit) = self.ptr.pos.and_then(|q| self.cam.pick_plane(q, *p, normal)) {
                let np = hit + self.handles.grab;
                if np.is_finite() && np != *p {
                    *p = np;
                    changed = true;
                }
            }
        }
        let s = if active {
            size + 5.0
        } else if hot {
            size + 3.0
        } else {
            size
        };
        if active || hot {
            self.dl.glows.push(*p, s * 3.2, color.a(0.55));
        }
        self.dl.overlay_discs.push(*p, s, color);
        changed
    }

    /// A draggable point on the plane z = 0. Returns true while it is being moved.
    pub fn drag2(&mut self, id: u32, p: &mut Vec2, color: Color) -> bool {
        let mut q = p.extend(0.0);
        let changed = self.drag_impl(id, &mut q, Vec3::Z, color, 12.0);
        *p = q.truncate();
        changed
    }

    /// A draggable point in space, moved within the plane facing the camera (or the
    /// plane with the given normal).
    pub fn drag3(&mut self, id: u32, p: &mut Vec3, normal: Option<Vec3>, color: Color) -> bool {
        self.drag3_sized(id, p, normal, color, 12.0)
    }

    /// The same with a dot of another size (logical pixels across), for a picture with
    /// many handles close together.
    pub fn drag3_sized(&mut self, id: u32, p: &mut Vec3, normal: Option<Vec3>, color: Color, size: f32) -> bool {
        let n = normal.unwrap_or(-self.cam.fwd);
        self.drag_impl(id, p, n, color, size)
    }

    // ----- axes ------------------------------------------------------------------

    /// Tick labels for a 2D view. The grid itself is drawn by the renderer.
    pub fn axes2(&mut self) {
        let (lo, hi) = self.cam.flat_bounds();
        let view = self.view_size();
        let wpp = (hi.y - lo.y) / view.y;
        // a view squeezed to nothing (a panel open over a small phone's screen, a
        // window being minimised) has no room for numbers, and no finite scale
        if !(wpp.is_finite() && wpp > 0.0 && view.x >= 1.0) {
            return;
        }
        let step = nice_step(wpp * 95.0);
        let col = pal::DIM;
        let margin = 16.0 * wpp;
        // x labels sit just under the x axis, or cling to the edge when it is off-screen
        // (max then min rather than clamp: in a view narrower than the margins the
        // bounds cross, and clamp would panic)
        let y_anchor = 0.0f32.max(lo.y + margin * 1.6).min(hi.y - margin * 0.4);
        let x_anchor = 0.0f32.max(lo.x + margin * 0.4).min(hi.x - margin * 2.4);
        // a number too near the edge would be cut off: leave it out
        let edge = 20.0 * wpp;
        let ticks = |lo: f32, hi: f32| tick_values(lo, hi, step);
        // No number is written over a caption or a heading: those written so far on a
        // dark plate (`caption_px`), each with the room its plate takes.
        let plates: Vec<egui::Rect> = self
            .dl
            .labels
            .iter()
            .filter(|l| l.backed)
            .map(|l| {
                let size = self.text_size(&l.text, l.size);
                l.align.anchor_size(egui::pos2(l.pos.x, l.pos.y), egui::vec2(size.x, size.y)).expand2(egui::vec2(9.0, 5.0))
            })
            .collect();
        let numbers_from = self.dl.labels.len();
        for x in ticks(lo.x, hi.x) {
            if x.abs() > step * 0.01 && x > lo.x + edge && x < hi.x - edge {
                self.text_off(Vec2::new(x, y_anchor), Vec2::new(0.0, 5.0), fmt_tick(x, step), 12.0, col, Align2::CENTER_TOP);
            }
        }
        for y in ticks(lo.y, hi.y) {
            if y.abs() > step * 0.01 && y > lo.y + edge * 0.6 && y < hi.y - edge * 0.6 {
                self.text_off(Vec2::new(x_anchor, y), Vec2::new(6.0, 0.0), fmt_tick(y, step), 12.0, col, Align2::LEFT_CENTER);
            }
        }
        if !plates.is_empty() {
            let mut k = 0;
            self.dl.labels.retain(|l| {
                k += 1;
                // (a number's box, nearly: its figures are about 7 pixels wide at this size)
                let size = egui::vec2(7.0 * l.text.chars().count() as f32, 14.0);
                let at = l.align.anchor_size(egui::pos2(l.pos.x, l.pos.y), size);
                k <= numbers_from || !plates.iter().any(|plate| plate.intersects(at))
            });
        }
    }

    /// Coloured x/y/z axes from the origin with ticks and labels (the axes' names are
    /// variables, and typeset as such).
    pub fn axes3(&mut self, extent: Vec3) {
        let names = ["$x$", "$y$", "$z$"];
        for k in 0..3 {
            let mut dir = Vec3::ZERO;
            dir[k] = 1.0;
            let e = extent[k];
            let c = pal::AXIS[k];
            self.line(-dir * e, Vec3::ZERO, 1.2, c.a(0.35));
            self.arrow(Vec3::ZERO, dir * e, 2.0, c);
            self.text_off(dir * e, Vec2::new(0.0, -10.0), names[k], 15.0, c, Align2::CENTER_BOTTOM);
            let step = nice_step(e / 2.6);
            let mut v = step;
            while v < e * 0.92 {
                for s in [-1.0, 1.0] {
                    let p = dir * (v * s);
                    self.dot(p, 4.0, c.a(0.9));
                    if s > 0.0 {
                        self.text_off(p, Vec2::new(0.0, 7.0), fmt_tick(v, step), 11.0, pal::DIM, Align2::CENTER_TOP);
                    }
                }
                v += step;
            }
        }
    }
}

/// The smallest "nice" number (1, 2 or 5 times a power of ten) that is at least `x`.
pub fn nice_step(x: f32) -> f32 {
    let x = x.max(1e-30);
    let p = 10f32.powf(x.log10().floor());
    let f = x / p;
    let m = if f <= 1.0 {
        1.0
    } else if f <= 2.0 {
        2.0
    } else if f <= 5.0 {
        5.0
    } else {
        10.0
    };
    m * p
}

/// The multiples of `step` from `lo` to `hi`, where the 2D axes put their numbers.
/// Counted, not stepped to: far from the origin and zoomed far in, a step can be too
/// small to move the coordinate at all.
fn tick_values(lo: f32, hi: f32, step: f32) -> impl Iterator<Item = f32> {
    let first = (lo / step).ceil();
    let count = (((hi - lo) / step).floor().max(0.0) as usize).min(400);
    (0..=count).map(move |i| (first + i as f32) * step).filter(move |v| *v <= hi)
}

/// Format a tick value with just enough digits for its spacing: neighbouring ticks
/// never read the same, also in the scientific form used for very large and very
/// small numbers (100005 with a step of 5 is 1.00005e5, not 1.0e5).
pub fn fmt_tick(v: f32, step: f32) -> String {
    let a = v.abs();
    if a != 0.0 && !(1e-4..1e5).contains(&a) {
        // the digits from the leading one down to the step's
        let digits = ((a as f64).log10().floor() - (step as f64).log10().floor()).max(1.0).min(9.0) as usize;
        return format!("{v:.digits$e}");
    }
    let decimals = (-step.log10().floor()).clamp(0.0, 9.0) as usize;
    format!("{v:.decimals$}")
}

/// Compact number formatting for read-outs.
/// A whole number with its thousands set apart: 384 000.
pub fn count(n: usize) -> String {
    let digits = n.to_string();
    let mut out = String::new();
    for (i, c) in digits.chars().enumerate() {
        if i > 0 && (digits.len() - i).is_multiple_of(3) {
            out.push('\u{2009}');
        }
        out.push(c);
    }
    out
}

/// A signed number to `decimals` places, with a true minus sign (as `kit::num` writes
/// it: a hyphen is not one) and a plain "0" for anything that rounds to zero (rather
/// than "−0.0000").
pub fn signed(v: f32, decimals: usize) -> String {
    if (v.abs() as f64) < 0.5 * 10f64.powi(-(decimals as i32)) {
        "0".to_string()
    } else {
        format!("{v:+.decimals$}").replacen('-', "−", 1)
    }
}

pub fn fmt(v: f32) -> String {
    let a = v.abs();
    if a == 0.0 {
        "0".into()
    } else if !(1e-3..1e4).contains(&a) {
        format!("{v:.2e}")
    } else if a >= 100.0 {
        format!("{v:.1}")
    } else if a >= 10.0 {
        format!("{v:.2}")
    } else {
        format!("{v:.3}")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A label's place is remembered for one frame; a new place is glided to, most of
    /// the way in a fifth of a second and exactly there soon after, whatever the
    /// display's rate; and a label that was not drawn last frame starts where it is put.
    #[test]
    fn a_label_glides_to_a_new_place() {
        for hz in [60.0f32, 144.0] {
            let mut places = Places::default();
            places.begin_frame();
            assert_eq!(places.chosen("a", 0), None);
            let first = Vec2::new(10.0, -8.0);
            assert_eq!(places.put("a", 0, first, 1.0 / hz), first);
            assert!(!places.gliding());
            // told apart by name and by number
            assert_eq!(places.put("a", 1, Vec2::ONE, 1.0 / hz), Vec2::ONE);
            places.begin_frame();
            assert_eq!((places.chosen("a", 0), places.chosen("a", 1), places.chosen("b", 0)), (Some(first), Some(Vec2::ONE), None));
            // a new place, sixty pixels off: nearer each frame, never past it
            let to = Vec2::new(-50.0, -8.0);
            let (mut left, mut frames) = (60.0f32, 0);
            loop {
                let drawn = places.put("a", 0, to, 1.0 / hz);
                assert_eq!(places.chosen("a", 0), Some(to));
                let now = (to - drawn).length();
                assert!(now < left && drawn.y == -8.0 && drawn.x >= to.x, "frame {frames}: {drawn}");
                left = now;
                frames += 1;
                if (frames as f32 / hz - 0.2).abs() < 0.5 / hz {
                    assert!(left < 0.1 * 60.0 && left > 0.02 * 60.0, "{left} px left after {frames} frames at {hz} Hz");
                }
                if !places.gliding() {
                    break;
                }
                assert!(frames as f32 / hz < 0.6, "still gliding after {frames} frames at {hz} Hz");
                places.begin_frame();
            }
            assert_eq!(places.put("a", 0, to, 1.0 / hz), to);
            // left out of a frame, it has no place to glide from
            places.begin_frame();
            places.begin_frame();
            assert_eq!(places.chosen("a", 0), None);
            assert_eq!(places.put("a", 0, first, 1.0 / hz), first);
            assert!(!places.gliding());
            // and nothing that is not a number is ever drawn
            places.begin_frame();
            assert!(places.put("a", 0, Vec2::new(f32::NAN, 0.0), 1.0 / hz).x.is_nan());
            places.begin_frame();
            assert_eq!(places.put("a", 0, to, 1.0 / hz), to);
        }
    }

    /// A signed number has a true minus sign, a plus sign, or is a plain 0.
    #[test]
    fn signed_numbers_have_true_minus_signs() {
        assert_eq!(signed(-0.2, 1), "−0.2");
        assert_eq!(signed(0.25, 2), "+0.25");
        assert_eq!(signed(-0.00004, 3), "0");
        assert_eq!(signed(-1.5e3, 0), "−1500");
    }

    /// The sheets that screen-space drawing lies on are finite, and put a pixel where
    /// it belongs, however far a 3D view is zoomed in or out and whatever the window.
    #[test]
    fn the_screen_sheets_hold_at_any_zoom() {
        use crate::gfx::camera::{Camera, View};
        // (the second target is where random input once took the view: far off to one
        // side, and then zoomed in as far as it went, which left the sheets at NaN)
        let targets = [Vec3::new(0.0, 0.0, 0.5), Vec3::new(72.6, 59.3, -35.7), Vec3::new(-4000.0, 0.0, 1.0)];
        for (viewport, target) in
            [Vec2::new(1600.0, 1000.0), Vec2::new(600.0, 480.0), Vec2::new(240.0, 180.0), Vec2::new(8.0, 8.0)].into_iter().flat_map(|v| targets.map(|t| (v, t)))
        {
            for (factor, times) in [(1.0f32, 0), (5.0, 12), (0.2, 12), (1.25, 200), (0.8, 200)] {
                let mut cam = Camera::new(View::Orbit { target, yaw: 0.9, pitch: 0.5, dist: 8.0 });
                cam.update(1.0 / 60.0, viewport);
                for _ in 0..times {
                    cam.zoom(factor, None);
                    // (frames of a fifteenth of a second, as on a slow machine)
                    cam.update(1.0 / 15.0, viewport);
                    let planes = Frame::hud_planes(&cam, 2.0);
                    assert!(planes.iter().flatten().all(|v| v.is_finite()), "{viewport} at {target} zoomed by {factor}: {planes:?} at distance {}", cam.dist());
                    // the middle of the window, on the top sheet, is seen in the middle
                    let [origin, across, down] = planes[2];
                    let mid = origin + across * (viewport.x * 0.25) + down * (viewport.y * 0.25);
                    let seen = cam.project(mid).expect("in front of the camera");
                    assert!(
                        (seen - viewport * 0.5).length() < 0.02 * viewport.max_element() + 0.5,
                        "{viewport} at {target} zoomed by {factor}: the middle is seen at {seen}, distance {}",
                        cam.dist()
                    );
                }
            }
        }
    }

    /// Wherever the 2D axes can be taken (any centre, zoomed in as far as the camera
    /// allows, in a short window and a tall one), no two neighbouring numbers on an
    /// axis read the same.
    #[test]
    fn neighbouring_ticks_never_read_the_same() {
        let mut checked = 0;
        for magnitude in -7..=8 {
            for mantissa in [1.0f32, 0.97336, 4.6, 9.9993] {
                for sign in [1.0f32, -1.0] {
                    let c = sign * mantissa * 10f32.powi(magnitude);
                    // (the camera's own limit, gfx/camera.rs: no closer than this)
                    let closest = 1e-6f32.max(1e-4 * c.abs());
                    for view in [180.0f32, 700.0, 2160.0] {
                        let mut half = closest;
                        while half < 1e3 * c.abs().max(1.0) {
                            let step = nice_step(2.0 * half / view * 95.0);
                            let ticks: Vec<f32> = tick_values(c - half, c + half, step).filter(|v| v.abs() > step * 0.01).collect();
                            for pair in ticks.windows(2) {
                                let (a, b) = (fmt_tick(pair[0], step), fmt_tick(pair[1], step));
                                assert!(pair[0] == pair[1] || a != b, "{} and {} both read {a} (step {step}, centre {c})", pair[0], pair[1]);
                                checked += 1;
                            }
                            half *= 3.7;
                        }
                    }
                }
            }
        }
        assert!(checked > 10_000, "only {checked} pairs of ticks were looked at");
        // the reviewer's case: x³ near 46, zoomed in
        let near: Vec<String> = [99990.0, 99995.0, 100000.0, 100005.0, 100010.0].iter().map(|v| fmt_tick(*v, 5.0)).collect();
        assert_eq!(near, ["99990", "99995", "1.00000e5", "1.00005e5", "1.00010e5"]);
        // and where the step is coarse, the short form stays
        assert_eq!(fmt_tick(200000.0, 100000.0), "2.0e5");
        assert_eq!(fmt_tick(5e-5, 5e-5), "5.0e-5");
    }
}

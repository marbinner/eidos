//! The per-frame draw list. Scenes append primitives; the renderer uploads each batch
//! with one buffer write and draws it with one instanced call. Vectors keep their
//! capacity between frames, so a steady scene allocates nothing.

use super::color::Color;
use bytemuck::{Pod, Zeroable};
use glam::{Vec2, Vec3};

/// One polyline vertex (20 bytes). `width` is in logical pixels; 0 marks a ghost
/// vertex (shapes a join but is not drawn) and a negative width breaks the line.
///
/// Nothing that is not a finite number is ever sent to the GPU (what a GPU does with
/// such a position is its own business, and on some it is a streak across the
/// screen). The batches see to it: a line stops at such a point and starts again
/// after it, so a curve can simply be given NaN where it has no value; a point, a
/// triangle, a picture or a label placed there is left out, and counted (see
/// [`DrawList::left_out`]).
#[derive(Clone, Copy, Pod, Zeroable)]
#[repr(C)]
pub struct LinePoint {
    pub pos: [f32; 3],
    pub width: f32,
    pub color: Color,
}

pub const BREAK: LinePoint = LinePoint { pos: [0.0; 3], width: -1.0, color: Color::CLEAR };

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LineStyle {
    pub width_scale: f32,
    /// Brightness multiplier; values above 1 feed the bloom.
    pub intensity: f32,
    /// Pulls lines towards the camera so they do not z-fight with surfaces.
    pub depth_bias: f32,
    /// Trails only: how quickly brightness falls off with age.
    pub fade_power: f32,
    /// Dash period in logical pixels, 0 for solid. The pattern starts afresh at
    /// every segment, so it suits straight lines; a curve of short segments comes
    /// out solid (dash those with `Frame::dashed_path`).
    pub dash: f32,
    pub duty: f32,
    /// Trails only: 0..1, how much the width narrows with age.
    pub taper: f32,
    pub tint: [f32; 4],
}

impl Default for LineStyle {
    fn default() -> Self {
        LineStyle { width_scale: 1.0, intensity: 1.0, depth_bias: 2e-4, fade_power: 1.5, dash: 0.0, duty: 0.5, taper: 0.0, tint: [1.0; 4] }
    }
}

pub struct LineBatch {
    pub pts: Vec<LinePoint>,
    pub style: LineStyle,
}

impl LineBatch {
    pub fn new(style: LineStyle) -> Self {
        LineBatch { pts: vec![BREAK], style }
    }

    pub fn clear(&mut self) {
        self.pts.clear();
        self.pts.push(BREAK);
    }

    /// One more point of the current polyline. A point that is not a finite number
    /// ends the line there instead: the curve comes out with a gap.
    #[inline]
    pub fn push(&mut self, p: Vec3, width: f32, color: Color) {
        if p.is_finite() && width.is_finite() {
            self.pts.push(LinePoint { pos: p.to_array(), width, color });
        } else {
            self.brk();
        }
    }

    /// End the current polyline.
    #[inline]
    pub fn brk(&mut self) {
        if self.pts.last().is_none_or(|p| p.width >= 0.0) {
            self.pts.push(BREAK);
        }
    }

    #[inline]
    pub fn segment(&mut self, a: Vec3, b: Vec3, width: f32, color: Color) {
        self.brk();
        self.push(a, width, color);
        self.push(b, width, color);
        self.brk();
    }

    pub fn polyline<I: IntoIterator<Item = Vec3>>(&mut self, pts: I, width: f32, color: Color) {
        self.brk();
        for p in pts {
            self.push(p, width, color);
        }
        self.brk();
    }

    /// A dashed curve, as solid pieces. `pts` gives each point with where it falls on
    /// the screen (in pixels; `None` if it does not, which breaks the curve), and the
    /// pattern (`period` pixels to a dash and its gap, the first `on` of them lit) is
    /// measured along the curve as it lies there, so the dashes are even however
    /// finely the curve is cut into segments. Only what is inside `view` (the screen's
    /// size in the same pixels) is walked: a line that runs far off it costs nothing.
    pub fn dashes<I: IntoIterator<Item = (Vec3, Option<Vec2>)>>(&mut self, pts: I, period: f32, on: f32, view: Vec2, width: f32, color: Color) {
        self.brk();
        if !(period > 0.5 && on > 0.0 && on < period) {
            return;
        }
        let mut last: Option<(Vec3, Vec2)> = None;
        let (mut along, mut lit) = (0.0f32, false);
        for (p, s) in pts {
            let Some(s) = s.filter(|s| s.is_finite() && p.is_finite()) else {
                (last, along, lit) = (None, 0.0, false);
                self.brk();
                continue;
            };
            if let Some((q, t)) = last {
                let len = (s - t).length();
                // the stretch of this segment that is on the screen (with a period to spare)
                let (mut from, mut to) = (0.0f32, len);
                for (a, d, hi) in [(t.x, s.x - t.x, view.x), (t.y, s.y - t.y, view.y)] {
                    let (lo, hi) = (-period, hi + period);
                    if d.abs() < 1e-9 {
                        if a < lo || a > hi {
                            to = from;
                        }
                    } else {
                        let (u0, u1) = ((lo - a) / d * len, (hi - a) / d * len);
                        from = from.max(u0.min(u1));
                        to = to.min(u0.max(u1));
                    }
                }
                if to > from && len > 0.0 {
                    if from > 0.0 && lit {
                        self.brk();
                        lit = false;
                    }
                    let (mut done, start) = (from, along + from);
                    while done < to {
                        let phase = (start + (done - from)).rem_euclid(period);
                        let now = phase < on;
                        let step = (if now { on - phase } else { period - phase }).max(1e-3).min(to - done);
                        if now {
                            if !lit {
                                self.brk();
                                self.push(q.lerp(p, done / len), width, color);
                            }
                            self.push(q.lerp(p, (done + step) / len), width, color);
                        } else if lit {
                            self.brk();
                        }
                        lit = now;
                        done += step;
                    }
                    if to < len && lit {
                        self.brk();
                        lit = false;
                    }
                } else if lit {
                    self.brk();
                    lit = false;
                }
                along = (along + len).rem_euclid(period);
            }
            last = Some((p, s));
        }
        self.brk();
    }

    /// A closed loop with a proper join where it meets itself.
    pub fn closed(&mut self, pts: &[Vec3], width: f32, color: Color) {
        let n = pts.len();
        if n < 3 {
            return;
        }
        self.brk();
        self.push(pts[n - 1], 0.0, color);
        for &p in pts {
            self.push(p, width, color);
        }
        self.push(pts[0], width, color);
        self.push(pts[1], 0.0, color);
        self.brk();
    }

    pub(crate) fn finish(&mut self) {
        self.brk();
    }

    pub(crate) fn instances(&self) -> u32 {
        self.pts.len().saturating_sub(3) as u32
    }
}

/// One point sprite (20 bytes). `size` is a diameter.
#[derive(Clone, Copy, Pod, Zeroable)]
#[repr(C)]
pub struct PointInst {
    pub pos: [f32; 3],
    pub size: f32,
    pub color: Color,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PointStyle {
    pub size_scale: f32,
    pub intensity: f32,
    pub outline_px: f32,
    /// Sizes are world units instead of pixels.
    pub world_size: bool,
    pub outline: Color,
}

impl Default for PointStyle {
    fn default() -> Self {
        PointStyle { size_scale: 1.0, intensity: 1.0, outline_px: 0.0, world_size: false, outline: Color::hex(0x0B0E14) }
    }
}

pub struct PointBatch {
    pub pts: Vec<PointInst>,
    pub style: PointStyle,
    /// Points left out this frame because their place or size was not a finite number.
    left_out: usize,
}

impl PointBatch {
    pub fn new(style: PointStyle) -> Self {
        PointBatch { pts: Vec::new(), style, left_out: 0 }
    }

    #[inline]
    pub fn push(&mut self, p: Vec3, size: f32, color: Color) {
        if p.is_finite() && size.is_finite() {
            self.pts.push(PointInst { pos: p.to_array(), size, color });
        } else {
            self.left_out += 1;
        }
    }

    pub fn clear(&mut self) {
        self.pts.clear();
        self.left_out = 0;
    }
}

/// One filled-triangle vertex (20 bytes). A zero normal means unlit.
#[derive(Clone, Copy, Pod, Zeroable)]
#[repr(C)]
pub struct TriVert {
    pub pos: [f32; 3],
    pub color: Color,
    pub normal: [i8; 4],
}

fn pack_normal(n: Vec3) -> [i8; 4] {
    let n = n.normalize_or_zero() * 127.0;
    [n.x as i8, n.y as i8, n.z as i8, 0]
}

#[derive(Default)]
pub struct TriBatch {
    pub v: Vec<TriVert>,
}

impl TriBatch {
    /// Take out the triangles with a corner that is not a finite number; returns how
    /// many there were. (Nearly always none, which one pass over the corners finds.)
    fn drop_unfinite(&mut self) -> usize {
        let fine = |v: &TriVert| v.pos.iter().all(|x| x.is_finite());
        if self.v.iter().all(fine) {
            return 0;
        }
        let before = self.v.len() / 3;
        let kept: Vec<TriVert> = self.v.chunks_exact(3).filter(|t| t.iter().all(fine)).flatten().copied().collect();
        self.v = kept;
        before - self.v.len() / 3
    }

    #[inline]
    pub fn vert(&mut self, p: Vec3, color: Color, normal: [i8; 4]) {
        self.v.push(TriVert { pos: p.to_array(), color, normal });
    }

    /// Unlit triangle.
    #[inline]
    pub fn tri(&mut self, a: Vec3, b: Vec3, c: Vec3, color: Color) {
        self.vert(a, color, [0; 4]);
        self.vert(b, color, [0; 4]);
        self.vert(c, color, [0; 4]);
    }

    /// Unlit quad a-b-c-d (in order around the perimeter).
    #[inline]
    pub fn quad(&mut self, a: Vec3, b: Vec3, c: Vec3, d: Vec3, color: Color) {
        self.tri(a, b, c, color);
        self.tri(a, c, d, color);
    }

    /// Lit quad; the normal is taken from the winding.
    pub fn quad_lit(&mut self, a: Vec3, b: Vec3, c: Vec3, d: Vec3, color: Color) {
        let n = pack_normal((b - a).cross(d - a));
        for p in [a, b, c, a, c, d] {
            self.vert(p, color, n);
        }
    }

    /// Quad with a colour per corner (gradients).
    pub fn quad_grad(&mut self, p: [Vec3; 4], c: [Color; 4]) {
        for i in [0, 1, 2, 0, 2, 3] {
            self.vert(p[i], c[i], [0; 4]);
        }
    }

    /// Axis-aligned rectangle in the plane z = `z`.
    pub fn rect(&mut self, min: Vec2, max: Vec2, z: f32, color: Color) {
        self.quad(Vec3::new(min.x, min.y, z), Vec3::new(max.x, min.y, z), Vec3::new(max.x, max.y, z), Vec3::new(min.x, max.y, z), color);
    }

    /// Lit parallelepiped spanned by three edge vectors from `o`.
    pub fn parallelepiped(&mut self, o: Vec3, a: Vec3, b: Vec3, c: Vec3, color: Color) {
        // orient faces outwards regardless of handedness
        let flip = a.cross(b).dot(c) < 0.0;
        let mut face = |p: Vec3, u: Vec3, v: Vec3| {
            if flip {
                self.quad_lit(p, p + v, p + u + v, p + u, color);
            } else {
                self.quad_lit(p, p + u, p + u + v, p + v, color);
            }
        };
        face(o, b, a);
        face(o + c, a, b);
        face(o, a, c);
        face(o + b, c, a);
        face(o, c, b);
        face(o + a, b, c);
    }
}

/// A piece of text placed over the rendered image.
pub struct Label {
    /// Position in logical points relative to the viewport's top-left corner.
    pub pos: Vec2,
    pub text: String,
    pub size: f32,
    pub color: Color,
    pub align: egui::Align2,
    /// Draw a dark plate behind the text (for titles and captions over the picture).
    pub backed: bool,
}

/// A retained GPU object queued for drawing this frame. `stamp` is (object id,
/// content version): the renderer uses it to tell whether anything changed.
pub(crate) struct SurfaceDraw {
    pub bind: wgpu::BindGroup,
    pub verts: u32,
    pub opaque: bool,
    pub stamp: [u64; 2],
}

pub(crate) struct TrailDraw {
    pub buf: wgpu::Buffer,
    pub bind: wgpu::BindGroup,
    pub cols: u32,
    pub rows: u32,
    pub additive: bool,
    pub stamp: [u64; 2],
}

/// One rectangle showing part of an image: corners in world units, the part of the
/// texture it shows, depth and opacity. 40 bytes.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, bytemuck::Pod, bytemuck::Zeroable)]
pub(crate) struct ImageRect {
    pub rect: [f32; 4],
    pub uv: [f32; 4],
    pub z: f32,
    pub alpha: f32,
}

/// A run of rectangles (`first..first + count` in the draw list's `image_rects`)
/// that all show the same image.
pub(crate) struct ImageDraw {
    pub bind: wgpu::BindGroup,
    pub stamp: [u64; 2],
    pub first: u32,
    pub count: u32,
}

/// How a line batch is composited.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LineKind {
    /// Alpha-blended, hidden behind nearer geometry.
    Blend,
    /// Added to what is behind it (glows), hidden behind nearer geometry.
    Additive,
    /// Alpha-blended on top of everything.
    Overlay,
}

/// How a point batch is drawn.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PointKind {
    Disc,
    Sphere,
    Glow,
    /// Discs on top of everything.
    Overlay,
}

pub struct DrawList {
    /// Alpha-blended, depth-tested lines.
    pub lines: LineBatch,
    pub dashed: LineBatch,
    /// Additive lines that bloom.
    pub glow_lines: LineBatch,
    /// Drawn last, ignoring depth (handles, gizmos).
    pub overlay: LineBatch,
    /// Flat anti-aliased discs with an optional outline.
    pub discs: PointBatch,
    /// Lit spheres with true depth.
    pub spheres: PointBatch,
    /// Soft additive glows.
    pub glows: PointBatch,
    /// Discs drawn last, ignoring depth.
    pub overlay_discs: PointBatch,
    /// Depth-writing fills.
    pub tris: TriBatch,
    /// Translucent fills.
    pub tris_alpha: TriBatch,
    pub labels: Vec<Label>,
    pub(crate) surfaces: Vec<SurfaceDraw>,
    pub(crate) trails: Vec<TrailDraw>,
    pub(crate) images: Vec<ImageDraw>,
    pub(crate) image_rects: Vec<ImageRect>,
    /// Batches with styles of the scene's own choosing (see [`DrawList::lines_with`]).
    pub(crate) extra_lines: Vec<(LineKind, LineBatch)>,
    pub(crate) extra_points: Vec<(PointKind, PointBatch)>,
    /// Triangles, pictures and labels left out this frame (see [`DrawList::left_out`]).
    pub(crate) left_out: usize,
}

impl Default for DrawList {
    fn default() -> Self {
        let (line, point) = DrawList::built_in_styles();
        DrawList {
            lines: LineBatch::new(line[0]),
            dashed: LineBatch::new(line[1]),
            glow_lines: LineBatch::new(line[2]),
            overlay: LineBatch::new(line[3]),
            discs: PointBatch::new(point[0]),
            spheres: PointBatch::new(point[1]),
            glows: PointBatch::new(point[2]),
            overlay_discs: PointBatch::new(point[3]),
            tris: TriBatch::default(),
            tris_alpha: TriBatch::default(),
            labels: Vec::new(),
            surfaces: Vec::new(),
            trails: Vec::new(),
            images: Vec::new(),
            image_rects: Vec::new(),
            extra_lines: Vec::new(),
            extra_points: Vec::new(),
            left_out: 0,
        }
    }
}

impl DrawList {
    /// The styles of the built-in batches: lines, dashed, glow_lines, overlay; and
    /// discs, spheres, glows, overlay_discs. (A scene may change a batch's style for
    /// the frame; `clear` puts these back.)
    fn built_in_styles() -> ([LineStyle; 4], [PointStyle; 4]) {
        (
            [
                LineStyle::default(),
                LineStyle { dash: 9.0, duty: 0.55, ..Default::default() },
                LineStyle { intensity: 2.2, ..Default::default() },
                LineStyle { depth_bias: 0.0, ..Default::default() },
            ],
            [
                PointStyle { outline_px: 1.25, ..Default::default() },
                PointStyle { world_size: true, ..Default::default() },
                PointStyle { intensity: 2.0, ..Default::default() },
                PointStyle { outline_px: 1.5, ..Default::default() },
            ],
        )
    }

    /// The built-in line batches, in the order of [`DrawList::built_in_styles`].
    pub(crate) fn built_in_lines(&self) -> [&LineBatch; 4] {
        [&self.lines, &self.dashed, &self.glow_lines, &self.overlay]
    }

    /// The built-in point batches, in the same order.
    pub(crate) fn built_in_points(&self) -> [&PointBatch; 4] {
        [&self.discs, &self.spheres, &self.glows, &self.overlay_discs]
    }

    /// Nothing more will be added this frame: close every line batch, and take out
    /// what cannot be drawn. (Doing it twice changes nothing.)
    pub(crate) fn finish(&mut self) {
        for batch in [&mut self.lines, &mut self.dashed, &mut self.glow_lines, &mut self.overlay] {
            batch.finish();
        }
        self.extra_lines.iter_mut().for_each(|(_, b)| b.finish());
        self.left_out += self.tris.drop_unfinite() + self.tris_alpha.drop_unfinite();
        let labels = self.labels.len();
        self.labels.retain(|l| l.pos.is_finite() && l.size.is_finite());
        self.left_out += labels - self.labels.len();
    }

    /// How many things the scene asked for this frame at places (or of sizes) that
    /// are not finite numbers, and that were therefore left out: points, triangles,
    /// pictures and labels. Gaps in lines are not counted: they are how a curve
    /// without a value somewhere is drawn. Anything counted here means the scene
    /// divided by zero somewhere and did not notice.
    pub fn left_out(&self) -> usize {
        self.left_out + self.built_in_points().into_iter().chain(self.extra_points.iter().map(|(_, b)| b)).map(|b| b.left_out).sum::<usize>()
    }

    /// Empty every batch, keeping allocations and restoring default styles.
    pub fn clear(&mut self) {
        let (line, point) = DrawList::built_in_styles();
        for (batch, style) in [&mut self.lines, &mut self.dashed, &mut self.glow_lines, &mut self.overlay].into_iter().zip(line) {
            batch.clear();
            batch.style = style;
        }
        for (batch, style) in [&mut self.discs, &mut self.spheres, &mut self.glows, &mut self.overlay_discs].into_iter().zip(point) {
            batch.clear();
            batch.style = style;
        }
        self.left_out = 0;
        self.tris.v.clear();
        self.tris_alpha.v.clear();
        self.labels.clear();
        self.surfaces.clear();
        self.trails.clear();
        self.images.clear();
        self.image_rects.clear();
        // custom batches survive (with their allocations) for as long as they keep being used
        self.extra_lines.retain(|(_, b)| b.pts.len() > 1);
        self.extra_lines.iter_mut().for_each(|(_, b)| b.clear());
        self.extra_points.retain(|(_, b)| !b.pts.is_empty());
        self.extra_points.iter_mut().for_each(|(_, b)| b.clear());
    }

    /// A line batch with any style: dash pattern, width scale, brightness, tint. All
    /// lines pushed with the same kind and style this frame share one draw call. The
    /// built-in batches (`lines`, `dashed`, `glow_lines`, `overlay`) cover the common
    /// cases; this is for everything else.
    pub fn lines_with(&mut self, kind: LineKind, style: LineStyle) -> &mut LineBatch {
        let i = match self.extra_lines.iter().position(|(k, b)| *k == kind && b.style == style) {
            Some(i) => i,
            None => {
                self.extra_lines.push((kind, LineBatch::new(style)));
                self.extra_lines.len() - 1
            }
        };
        &mut self.extra_lines[i].1
    }

    /// A point batch with any style (outline, size units, brightness).
    pub fn points_with(&mut self, kind: PointKind, style: PointStyle) -> &mut PointBatch {
        let i = match self.extra_points.iter().position(|(k, b)| *k == kind && b.style == style) {
            Some(i) => i,
            None => {
                self.extra_points.push((kind, PointBatch::new(style)));
                self.extra_points.len() - 1
            }
        };
        &mut self.extra_points[i].1
    }

    /// Rough primitive count for the on-screen statistics.
    pub fn stats(&self) -> (usize, usize, usize) {
        let segs = self.built_in_lines().into_iter().chain(self.extra_lines.iter().map(|(_, b)| b)).map(|b| b.pts.len()).sum::<usize>();
        let pts = self.built_in_points().into_iter().chain(self.extra_points.iter().map(|(_, b)| b)).map(|b| b.pts.len()).sum::<usize>();
        let tris =
            (self.tris.v.len() + self.tris_alpha.v.len()) / 3 + self.surfaces.iter().map(|s| s.verts as usize / 3).sum::<usize>() + self.image_rects.len() * 2;
        let trail = self.trails.iter().map(|t| (t.cols * t.rows) as usize).sum::<usize>();
        (segs + trail, pts, tris)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A dashed curve is lit for the same share of its length, in pieces of the same
    /// length, however finely it is cut into segments; and a stretch that runs off
    /// the screen is not walked.
    #[test]
    fn dashes_follow_the_curve_and_not_its_segments() {
        let view = Vec2::new(400.0, 300.0);
        let lit = |segments: usize, length: f32| {
            let mut batch = LineBatch::new(LineStyle::default());
            let pts = (0..=segments).map(|i| {
                let x = length * i as f32 / segments as f32;
                (Vec3::new(x, 0.0, 0.0), Some(Vec2::new(x, 150.0)))
            });
            batch.dashes(pts, 10.0, 6.0, view, 2.0, Color::WHITE);
            // the pieces: runs of points between breaks
            let mut pieces = Vec::new();
            let mut run: Option<(f32, f32)> = None;
            for p in &batch.pts {
                if p.width < 0.0 {
                    pieces.extend(run.take().map(|(a, b)| b - a));
                } else {
                    run = Some(run.map_or((p.pos[0], p.pos[0]), |(a, _)| (a, p.pos[0])));
                }
            }
            (pieces, batch.pts.len())
        };
        for segments in [1, 7, 300, 4000] {
            let (pieces, _) = lit(segments, 300.0);
            assert_eq!(pieces.len(), 30, "{segments} segments");
            assert!(pieces.iter().all(|l| (l - 6.0).abs() < 0.05), "{segments} segments: {pieces:?}");
        }
        // a line a million pixels long is only dashed where it can be seen
        let (pieces, points) = lit(1, 1e6);
        assert!(pieces.len() <= 43 && points < 200, "{} pieces, {points} points", pieces.len());
        // and a point that is nowhere breaks the curve without ending it
        let mut batch = LineBatch::new(LineStyle::default());
        let at = |x: f32| (Vec3::new(x, 0.0, 0.0), Some(Vec2::new(x, 10.0)));
        batch.dashes([at(0.0), at(50.0), (Vec3::NAN, None), at(100.0), at(150.0)], 10.0, 6.0, view, 2.0, Color::WHITE);
        assert!(batch.pts.iter().all(|p| p.pos.iter().all(|x| x.is_finite())));
        assert!(batch.pts.iter().any(|p| p.pos[0] > 100.0) && !batch.pts.iter().any(|p| p.pos[0] > 50.5 && p.pos[0] < 99.5));
    }

    /// What is not a finite number never reaches the GPU: a line stops there and
    /// starts again, and anything else placed there is left out and counted.
    #[test]
    fn nothing_unfinite_is_drawn() {
        let mut dl = DrawList::default();
        let at = |x: f32| Vec3::new(x, 0.0, 0.0);
        dl.lines.polyline([at(0.0), at(1.0), at(f32::NAN), at(3.0), at(4.0), at(f32::INFINITY)], 2.0, Color::WHITE);
        dl.discs.push(at(1.0), 4.0, Color::WHITE);
        dl.discs.push(at(f32::NAN), 4.0, Color::WHITE);
        dl.glows.push(at(1.0), f32::INFINITY, Color::WHITE);
        dl.tris.tri(at(0.0), at(1.0), at(2.0), Color::WHITE);
        dl.tris.tri(at(0.0), at(f32::NAN), at(2.0), Color::WHITE);
        dl.tris.tri(at(5.0), at(6.0), at(7.0), Color::WHITE);
        dl.labels.push(Label {
            pos: Vec2::new(f32::NAN, 1.0),
            text: "nowhere".into(),
            size: 12.0,
            color: Color::WHITE,
            align: egui::Align2::LEFT_TOP,
            backed: false,
        });
        dl.finish();
        dl.finish();
        let finite = |p: &[f32; 3]| p.iter().all(|x| x.is_finite());
        assert!(dl.lines.pts.iter().all(|p| finite(&p.pos)));
        // two pieces, 0-1 and 3-4, with a break between them and one at each end
        let widths: Vec<f32> = dl.lines.pts.iter().map(|p| p.width).collect();
        assert_eq!(widths, [-1.0, 2.0, 2.0, -1.0, 2.0, 2.0, -1.0]);
        assert_eq!((dl.discs.pts.len(), dl.glows.pts.len(), dl.labels.len()), (1, 0, 0));
        assert_eq!(dl.tris.v.len(), 6);
        assert_eq!(dl.tris.v[3].pos, [5.0, 0.0, 0.0]);
        assert_eq!(dl.left_out(), 4, "a disc, a glow, a triangle and a label");
        dl.clear();
        assert_eq!(dl.left_out(), 0);
    }

    /// Custom batches are found again by kind and style, and are forgotten once a
    /// frame goes by without them.
    #[test]
    fn custom_batches_are_shared_and_forgotten() {
        let mut dl = DrawList::default();
        let dashes = LineStyle { dash: 14.0, ..Default::default() };
        dl.lines_with(LineKind::Blend, dashes).segment(Vec3::ZERO, Vec3::X, 1.0, Color::WHITE);
        dl.lines_with(LineKind::Blend, dashes).segment(Vec3::ZERO, Vec3::Y, 1.0, Color::WHITE);
        dl.lines_with(LineKind::Overlay, dashes).segment(Vec3::ZERO, Vec3::Z, 1.0, Color::WHITE);
        assert_eq!(dl.extra_lines.len(), 2, "one batch per kind and style");
        dl.clear();
        assert_eq!(dl.extra_lines.len(), 2, "kept (empty) for the next frame");
        dl.lines_with(LineKind::Blend, dashes).segment(Vec3::ZERO, Vec3::X, 1.0, Color::WHITE);
        dl.clear();
        assert_eq!(dl.extra_lines.len(), 1, "the one that went unused is gone");
    }
}

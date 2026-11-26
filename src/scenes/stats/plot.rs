//! What the statistics scenes draw with: charts that stay put on the screen, buttons on
//! the picture, marks that can be dragged along a chart, and numbers that glide.
//!
//! Everything here is in logical pixels from the top-left corner of the picture, y
//! downwards (see `kit.rs`, "screen space"), so a scene lays its charts out for the
//! window it is given: beside each other on a desktop, one under another on a phone.
//!
//! A chart is a [`Plot`]: a rectangle of the screen and the ranges of x and y shown in
//! it. It draws its own backing, axes and ticks, and curves, bars, shaded areas, dots
//! and lines given in the data's own units, kept inside its rectangle. Lines are drawn
//! in batches by kind ([`AXES`], [`CURVES`], ...; see [`apart`]): `eidos motion` follows
//! a batch's points by their places in it, so anything whose number of points changes
//! from frame to frame (a histogram that fills, a path that grows) goes in a kind of
//! its own, [`OWN`] and above.

// (a kit: a scene uses the parts it needs)
#![allow(dead_code)]

use crate::scenes::prelude::*;

/// A rectangle of the screen: its top-left and bottom-right corners.
pub type Rect = (Vec2, Vec2);

pub fn within(p: Vec2, (min, max): Rect, margin: f32) -> bool {
    p.x >= min.x - margin && p.x <= max.x + margin && p.y >= min.y - margin && p.y <= max.y + margin
}

/// Where the pointer is, in logical pixels of the picture.
pub fn pointer(f: &Frame) -> Option<Vec2> {
    f.ptr.pos.map(|p| p / f.px)
}

// ----- lines by kind -------------------------------------------------------------------

/// Axes, ticks and frames: the same number of points in every frame.
pub const AXES: u8 = 0;
/// Curves sampled at a fixed number of points. (A curve that crosses its chart's edge
/// is cut there, which adds points and breaks: one that moves across an edge from
/// frame to frame belongs in a kind of its own, `OWN` and up, like anything else whose
/// number of points changes. And a kind that is drawn in some frames and not in others
/// moves every later batch up a place: draw it in every frame, empty if need be.)
pub const CURVES: u8 = 1;
/// Marks that come and go singly (a threshold, a mean).
pub const MARKS: u8 = 2;
/// The outlines of buttons.
pub const BUTTONS: u8 = 3;
/// The first kind a scene may use for something of its own whose number of points
/// changes (`OWN`, `OWN + 1`, ... up to 31).
pub const OWN: u8 = 8;

/// The batch of lines, on top of everything, for one kind of line alone. (The fading
/// is for trails: in a line batch it only tells one batch from another.)
pub fn apart<'b>(f: &'b mut Frame<'_>, kind: u8, dashed: bool) -> &'b mut crate::gfx::draw::LineBatch {
    let fade_power = 60.0 + kind.min(31) as f32;
    let style = if dashed {
        LineStyle { dash: 7.0, duty: 0.55, depth_bias: 0.0, fade_power, ..Default::default() }
    } else {
        LineStyle { depth_bias: 0.0, fade_power, ..Default::default() }
    };
    f.dl.lines_with(LineKind::Overlay, style)
}

/// A line in screen space, in the batch of its kind.
pub fn line_of(f: &mut Frame, kind: u8, a: Vec2, b: Vec2, width: f32, color: Color) {
    if !(a.is_finite() && b.is_finite()) {
        return;
    }
    let (a, b) = (f.hud(a), f.hud(b));
    apart(f, kind, false).segment(a, b, width, color);
}

/// The same, dashed.
pub fn dashed_of(f: &mut Frame, kind: u8, a: Vec2, b: Vec2, width: f32, color: Color) {
    if !(a.is_finite() && b.is_finite()) {
        return;
    }
    let (a, b) = (f.hud(a), f.hud(b));
    apart(f, kind, true).segment(a, b, width, color);
}

/// A polyline in screen space, in the batch of its kind; a point that is not a number
/// breaks it.
pub fn path_of(f: &mut Frame, kind: u8, pts: &[Vec2], width: f32, color: Color) {
    for piece in pts.split(|p| !p.is_finite()).filter(|piece| piece.len() >= 2) {
        let points: Vec<Vec3> = piece.iter().map(|p| f.hud(*p)).collect();
        apart(f, kind, false).polyline(points, width, color);
    }
}

/// A translucent rectangle in screen space, over a panel's backing.
pub fn fill(f: &mut Frame, (min, max): Rect, color: Color) {
    if !(min.is_finite() && max.is_finite()) {
        return;
    }
    let c = [min, Vec2::new(max.x, min.y), max, Vec2::new(min.x, max.y)].map(|p| f.hud_at(1, p));
    f.dl.tris_alpha.quad(c[0], c[1], c[2], c[3], color);
}

/// A dot in screen space, on top of everything.
pub fn disc(f: &mut Frame, at: Vec2, size: f32, color: Color) {
    if at.is_finite() {
        let p = f.hud(at);
        f.dl.points_with(PointKind::Overlay, PointStyle { outline_px: 0.0, ..Default::default() }).push(p, size, color);
    }
}

// ----- charts --------------------------------------------------------------------------

/// Round values along an axis from `lo` to `hi` drawn over `pixels`: about one every
/// sixty pixels.
#[derive(Clone, Copy, Debug)]
pub struct Ticks {
    pub step: f64,
    first: i64,
    last: i64,
}

impl Ticks {
    pub fn of(lo: f64, hi: f64, pixels: f32) -> Ticks {
        let span = hi - lo;
        if !span.is_finite() || span <= 0.0 || pixels.is_nan() || pixels <= 0.0 {
            return Ticks { step: 1.0, first: 0, last: -1 };
        }
        let step = round_step(span / (pixels as f64 / 60.0).max(1.0));
        let first = (lo / step - 1e-9).ceil() as i64;
        let last = ((hi / step + 1e-9).floor() as i64).min(first + 60);
        Ticks { step, first, last }
    }

    pub fn values(&self) -> impl Iterator<Item = f64> + '_ {
        (self.first..=self.last).map(|k| k as f64 * self.step)
    }

    /// A tick's value written with just the digits its spacing needs.
    pub fn name(&self, v: f64) -> String {
        let decimals = (-(self.step * (1.0 + 1e-9)).log10().floor()).max(0.0).min(6.0) as usize;
        num(v, decimals)
    }
}

/// The round number (1, 2 or 5 times a power of ten) at or above `x`. (Worked out in
/// f64: a step that went through f32 is a hair off, and 1 / 0.2 then counts 4 ticks.)
pub fn round_step(x: f64) -> f64 {
    let x = x.max(1e-300);
    let power = 10f64.powf(x.log10().floor());
    let lead = x / power;
    let round = if lead <= 1.0 + 1e-9 {
        1.0
    } else if lead <= 2.0 + 1e-9 {
        2.0
    } else if lead <= 5.0 + 1e-9 {
        5.0
    } else {
        10.0
    };
    round * power
}

/// A chart on the screen: the rectangle the data is drawn in (axis names and tick
/// numbers go outside it: leave about 34 pixels to its left and 30 below), and the
/// ranges of x and y that fill it.
#[derive(Clone, Copy, Debug)]
pub struct Plot {
    pub rect: Rect,
    pub x: [f64; 2],
    pub y: [f64; 2],
}

impl Plot {
    /// (Ranges that are empty or not numbers are widened to something drawable.)
    pub fn new(rect: Rect, x: [f64; 2], y: [f64; 2]) -> Plot {
        let range = |[a, b]: [f64; 2]| {
            if a.is_finite() && b.is_finite() && b > a {
                [a, b]
            } else if a.is_finite() {
                [a - 0.5, a + 0.5]
            } else {
                [0.0, 1.0]
            }
        };
        Plot { rect, x: range(x), y: range(y) }
    }

    pub fn width(&self) -> f32 {
        self.rect.1.x - self.rect.0.x
    }

    pub fn height(&self) -> f32 {
        self.rect.1.y - self.rect.0.y
    }

    /// Too small to draw anything in.
    pub fn tiny(&self) -> bool {
        !(self.width() >= 24.0 && self.height() >= 18.0)
    }

    /// Where the value x is across the screen (not kept inside the chart).
    pub fn px_x(&self, x: f64) -> f32 {
        self.rect.0.x + self.width() * ((x - self.x[0]) / (self.x[1] - self.x[0])) as f32
    }

    /// Where the value y is down the screen (not kept inside the chart).
    pub fn px_y(&self, y: f64) -> f32 {
        self.rect.1.y - self.height() * ((y - self.y[0]) / (self.y[1] - self.y[0])) as f32
    }

    pub fn at(&self, x: f64, y: f64) -> Vec2 {
        Vec2::new(self.px_x(x), self.px_y(y))
    }

    /// The value of x at a place across the screen.
    pub fn x_at(&self, px: f32) -> f64 {
        self.x[0] + (self.x[1] - self.x[0]) * ((px - self.rect.0.x) / self.width().max(1e-6)) as f64
    }

    /// The value of y at a place down the screen.
    pub fn y_at(&self, px: f32) -> f64 {
        self.y[0] + (self.y[1] - self.y[0]) * ((self.rect.1.y - px) / self.height().max(1e-6)) as f64
    }

    /// The values under the pointer, if it is over the chart.
    pub fn under_pointer(&self, f: &Frame) -> Option<(f64, f64)> {
        pointer(f).filter(|p| within(*p, self.rect, 0.0)).map(|p| (self.x_at(p.x), self.y_at(p.y)))
    }

    /// The dark backing behind the chart and what is written round it: `left`, `below`,
    /// `above` and `right` pixels beyond the rectangle.
    pub fn backing(&self, f: &mut Frame, left: f32, below: f32, above: f32, right: f32) {
        hud_panel(f, self.rect.0 - Vec2::new(left, above), self.rect.1 + Vec2::new(right, below));
    }

    /// The line along the bottom with its ticks and their numbers, and the axis's name
    /// under the middle (`""` for none).
    pub fn x_axis(&self, f: &mut Frame, name: &str, size: f32) {
        if self.tiny() {
            return;
        }
        let (min, max) = self.rect;
        line_of(f, AXES, Vec2::new(min.x, max.y), max, 1.0, pal::DIM.a(0.8));
        let ticks = Ticks::of(self.x[0], self.x[1], self.width());
        for v in ticks.values() {
            let x = self.px_x(v);
            line_of(f, AXES, Vec2::new(x, max.y), Vec2::new(x, max.y + 4.0), 1.0, pal::DIM.a(0.8));
            f.text_px(Vec2::new(x, max.y + 6.0), ticks.name(v), size, pal::DIM, Align2::CENTER_TOP);
        }
        if !name.is_empty() {
            f.text_px(Vec2::new((min.x + max.x) * 0.5, max.y + 8.0 + size), name, size, pal::DIM, Align2::CENTER_TOP);
        }
    }

    /// The line up the left side with its ticks and their numbers, and the axis's name
    /// above its top (`""` for none).
    pub fn y_axis(&self, f: &mut Frame, name: &str, size: f32) {
        if self.tiny() {
            return;
        }
        let (min, max) = self.rect;
        line_of(f, AXES, min, Vec2::new(min.x, max.y), 1.0, pal::DIM.a(0.8));
        let ticks = Ticks::of(self.y[0], self.y[1], self.height() * 1.6);
        for v in ticks.values() {
            let y = self.px_y(v);
            line_of(f, AXES, Vec2::new(min.x - 4.0, y), Vec2::new(min.x, y), 1.0, pal::DIM.a(0.8));
            f.text_px(Vec2::new(min.x - 6.0, y), ticks.name(v), size, pal::DIM, Align2::RIGHT_CENTER);
        }
        if !name.is_empty() {
            f.text_px(Vec2::new(min.x - 30.0, min.y - 6.0), name, size, pal::DIM, Align2::LEFT_BOTTOM);
        }
    }

    /// What the chart shows, written above it: over the y axis's name, starting where
    /// the tick numbers start. (Leave about 44 pixels above the rectangle for the two.)
    pub fn title(&self, f: &mut Frame, text: &str, size: f32) {
        f.text_px(Vec2::new(self.rect.0.x - 30.0, self.rect.0.y - 24.0), text, size, pal::TEXT, Align2::LEFT_BOTTOM);
    }

    /// Words at (x, y) in the data's units, if that is inside the chart.
    #[allow(clippy::too_many_arguments)]
    pub fn text(&self, f: &mut Frame, x: f64, y: f64, offset: Vec2, text: &str, size: f32, color: Color, align: Align2) {
        let at = self.at(x, y);
        if at.is_finite() && within(at, self.rect, 0.5) {
            f.text_px(at + offset, text, size, color, align);
        }
    }

    /// Faint lines across the chart at the ticks of the y axis.
    pub fn y_grid(&self, f: &mut Frame) {
        if self.tiny() {
            return;
        }
        let (min, max) = self.rect;
        for v in Ticks::of(self.y[0], self.y[1], self.height() * 1.6).values() {
            let y = self.px_y(v);
            line_of(f, AXES, Vec2::new(min.x, y), Vec2::new(max.x, y), 1.0, pal::DIM.a(0.14));
        }
    }

    /// Points given in the data's units, as a polyline kept inside the chart: where it
    /// leaves the rectangle it is cut at the edge.
    pub fn path(&self, f: &mut Frame, kind: u8, points: impl IntoIterator<Item = (f64, f64)>, width: f32, color: Color) {
        let pts: Vec<Vec2> = points.into_iter().map(|(x, y)| self.at(x, y)).collect();
        path_of(f, kind, &clip(&pts, self.rect), width, color);
    }

    /// The curve y = g(x) across the whole chart, sampled at `n` points.
    pub fn curve(&self, f: &mut Frame, kind: u8, n: usize, width: f32, color: Color, g: impl Fn(f64) -> f64) {
        let n = n.max(2);
        let [a, b] = self.x;
        self.path(f, kind, (0..n).map(|i| a + (b - a) * i as f64 / (n - 1) as f64).map(|x| (x, g(x))), width, color);
    }

    /// The area between y = g(x) and y = `base`, for x from `from` to `to` (in either
    /// order), sampled at `n` points: a tail of a distribution, the probability of an
    /// interval.
    #[allow(clippy::too_many_arguments)]
    pub fn area(&self, f: &mut Frame, from: f64, to: f64, base: f64, n: usize, color: Color, g: impl Fn(f64) -> f64) {
        // (an end that is not a number shades nothing: `min` and `max` would quietly
        // take the chart's own edge in its place)
        if from.is_nan() || to.is_nan() {
            return;
        }
        let (a, b) = (from.min(to).max(self.x[0]), from.max(to).min(self.x[1]));
        if self.tiny() || b.is_nan() || a.is_nan() || b <= a {
            return;
        }
        let n = n.max(2);
        let (top, bottom) = (self.rect.0.y, self.rect.1.y);
        let floor = self.px_y(base).max(top).min(bottom);
        let column = |x: f64| {
            let y = g(x);
            (self.px_x(x), if y.is_finite() { self.px_y(y).max(top).min(bottom) } else { floor })
        };
        let mut last = column(a);
        for i in 1..n {
            let next = column(a + (b - a) * i as f64 / (n - 1) as f64);
            let c = [Vec2::new(last.0, floor), Vec2::new(last.0, last.1), Vec2::new(next.0, next.1), Vec2::new(next.0, floor)].map(|p| f.hud_at(1, p));
            f.dl.tris_alpha.quad(c[0], c[1], c[2], c[3], color);
            last = next;
        }
    }

    /// The band between y = lower(x) and y = upper(x), for x from `from` to `to`,
    /// sampled at `n` points: an interval round a fitted line, a range of curves.
    #[allow(clippy::too_many_arguments)]
    pub fn band(&self, f: &mut Frame, from: f64, to: f64, n: usize, color: Color, lower: impl Fn(f64) -> f64, upper: impl Fn(f64) -> f64) {
        if from.is_nan() || to.is_nan() {
            return;
        }
        let (a, b) = (from.min(to).max(self.x[0]), from.max(to).min(self.x[1]));
        if self.tiny() || b.is_nan() || a.is_nan() || b <= a {
            return;
        }
        let n = n.max(2);
        let (top, bottom) = (self.rect.0.y, self.rect.1.y);
        let column = |x: f64| {
            let (lo, hi) = (lower(x), upper(x));
            (lo.is_finite() && hi.is_finite()).then(|| (self.px_x(x), self.px_y(hi).max(top).min(bottom), self.px_y(lo).max(top).min(bottom)))
        };
        let mut last = column(a);
        for i in 1..n {
            let next = column(a + (b - a) * i as f64 / (n - 1) as f64);
            if let (Some(l), Some(r)) = (last, next) {
                let c = [Vec2::new(l.0, l.2), Vec2::new(l.0, l.1), Vec2::new(r.0, r.1), Vec2::new(r.0, r.2)].map(|p| f.hud_at(1, p));
                f.dl.tris_alpha.quad(c[0], c[1], c[2], c[3], color);
            }
            last = next;
        }
    }

    /// One bar from `x0` to `x1`, from y = `base` up (or down) to `y`, with `gap`
    /// pixels left free on each side.
    #[allow(clippy::too_many_arguments)]
    pub fn bar(&self, f: &mut Frame, x0: f64, x1: f64, base: f64, y: f64, gap: f32, color: Color) {
        if self.tiny() || !(x0.is_finite() && x1.is_finite() && y.is_finite() && base.is_finite()) {
            return;
        }
        let (min, max) = self.rect;
        let (a, b) = (self.px_x(x0.min(x1)).max(min.x), self.px_x(x0.max(x1)).min(max.x));
        let (a, b) = if b - a > 2.0 * gap + 1.0 { (a + gap, b - gap) } else { (a, b) };
        let (top, bottom) = (self.px_y(y.max(base)).max(min.y), self.px_y(y.min(base)).min(max.y));
        if b > a && bottom > top {
            fill(f, (Vec2::new(a, top), Vec2::new(b, bottom)), color);
        }
    }

    /// Bars side by side from `lo` on, each `width` wide (in the data's units), with
    /// these heights from y = 0: a histogram, or the probabilities of whole numbers
    /// (then `lo` is half a unit below the first of them and `width` is 1).
    pub fn bars(&self, f: &mut Frame, lo: f64, width: f64, heights: &[f64], gap: f32, color: Color) {
        for (i, h) in heights.iter().enumerate() {
            self.bar(f, lo + i as f64 * width, lo + (i + 1) as f64 * width, 0.0, *h, gap, color);
        }
    }

    /// A line up the whole chart at x (nothing if x is outside it).
    pub fn vline(&self, f: &mut Frame, kind: u8, x: f64, width: f32, color: Color, dashed: bool) {
        if !(x >= self.x[0] && x <= self.x[1]) {
            return;
        }
        let (a, b) = (Vec2::new(self.px_x(x), self.rect.0.y), Vec2::new(self.px_x(x), self.rect.1.y));
        if dashed {
            dashed_of(f, kind, a, b, width, color);
        } else {
            line_of(f, kind, a, b, width, color);
        }
    }

    /// A line across the whole chart at y (nothing if y is outside it).
    pub fn hline(&self, f: &mut Frame, kind: u8, y: f64, width: f32, color: Color, dashed: bool) {
        if !(y >= self.y[0] && y <= self.y[1]) {
            return;
        }
        let (a, b) = (Vec2::new(self.rect.0.x, self.px_y(y)), Vec2::new(self.rect.1.x, self.px_y(y)));
        if dashed {
            dashed_of(f, kind, a, b, width, color);
        } else {
            line_of(f, kind, a, b, width, color);
        }
    }

    /// A dot at (x, y), if that is inside the chart.
    pub fn dot(&self, f: &mut Frame, x: f64, y: f64, size: f32, color: Color) {
        let at = self.at(x, y);
        if at.is_finite() && within(at, self.rect, 0.5) {
            disc(f, at, size, color);
        }
    }

    /// A mark that can be dragged along the chart: a dot at (x, y) which the pointer
    /// can take hold of. Returns the values where the pointer has taken it while it is
    /// being dragged (kept inside the chart), `None` otherwise. `id` tells this handle
    /// from the scene's others.
    pub fn handle(&self, f: &mut Frame, id: u32, x: f64, y: f64, color: Color) -> Option<(f64, f64)> {
        let at = self.at(x, y);
        if self.tiny() || !at.is_finite() {
            return None;
        }
        let (min, max) = self.rect;
        let to = screen_handle(f, id, at.max(min).min(max), color)?;
        let to = to.max(min).min(max);
        Some((self.x_at(to.x), self.y_at(to.y)))
    }
}

/// A polyline kept inside a rectangle: each of its segments is cut to the part of it
/// that lies inside (a segment may cross the rectangle with both its ends outside),
/// and where the line leaves the rectangle it is broken (a point that is not a number).
pub fn clip(pts: &[Vec2], (min, max): Rect) -> Vec<Vec2> {
    // (half a pixel's grace, so that a line along the edge is not cut away)
    let (min, max) = (min - Vec2::splat(0.5), max + Vec2::splat(0.5));
    let mut out: Vec<Vec2> = Vec::with_capacity(pts.len() + 8);
    let mut open = false;
    for pair in pts.windows(2) {
        let (a, b) = (pair[0], pair[1]);
        let cut = if a.is_finite() && b.is_finite() { cut_segment(a, b, min, max) } else { None };
        match cut {
            Some((p, q)) => {
                // (the same point as the last one drawn: the line goes on)
                if !(open && out.last().is_some_and(|last| (*last - p).length_squared() < 1e-6)) {
                    if !out.is_empty() {
                        out.push(Vec2::NAN);
                    }
                    out.push(p);
                }
                out.push(q);
                // it goes on only if this segment ended inside, at its own end
                open = (q - b).length_squared() < 1e-6;
            }
            None => open = false,
        }
    }
    out
}

/// The part of the segment from `a` to `b` inside the rectangle, if any (Liang and
/// Barsky's way: the range of the segment's parameter inside each pair of edges).
fn cut_segment(a: Vec2, b: Vec2, min: Vec2, max: Vec2) -> Option<(Vec2, Vec2)> {
    let d = b - a;
    let (mut t0, mut t1) = (0.0f32, 1.0f32);
    for (p, q) in [(-d.x, a.x - min.x), (d.x, max.x - a.x), (-d.y, a.y - min.y), (d.y, max.y - a.y)] {
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
    (t0 <= t1).then(|| (a + d * t0, a + d * t1))
}

/// A draggable dot at a point of the screen: where the pointer has taken it, while it
/// is being dragged.
pub fn screen_handle(f: &mut Frame, id: u32, at: Vec2, color: Color) -> Option<Vec2> {
    let world = f.cam.pick_ground(at * f.px)?;
    let mut grip = world.truncate();
    if f.drag2(id, &mut grip, color) {
        on_screen(f, grip.extend(0.0))
    } else {
        None
    }
}

// ----- buttons on the picture ----------------------------------------------------------

/// A button on the picture: `on` draws it as the one chosen of several. Returns true in
/// the frame it is clicked. (The click is the button's: it does not also reach the
/// scene as a click on the picture, and a press on it does not move the view.)
pub fn button(f: &mut Frame, (min, max): Rect, text: &str, size: f32, on: bool) -> bool {
    if !(max.x - min.x >= 2.0 && max.y - min.y >= 2.0) {
        return false;
    }
    let over = pointer(f).is_some_and(|p| within(p, (min, max), 0.0));
    let backing = match (on, over) {
        (true, _) => Color::hex(0x2B3D66),
        (false, true) => Color::hex(0x2A3350),
        _ => Color::hex(0x161C2B),
    };
    hud_rect(f, min, max, backing);
    let edge = [min, Vec2::new(max.x, min.y), max, Vec2::new(min.x, max.y), min];
    path_of(
        f,
        BUTTONS,
        &edge,
        if on { 1.4 } else { 1.0 },
        pal::LINK.a(if on {
            0.95
        } else if over {
            0.7
        } else {
            0.35
        }),
    );
    f.text_px((min + max) * 0.5, text, size, if on { pal::WHITE } else { pal::TEXT.a(0.85) }, Align2::CENTER_CENTER);
    if over {
        f.handles.claim_clicks();
        f.handles.capture();
    }
    // (the click is used up: `Frame::clicked_canvas` is false for the rest of the frame.
    // A scene that acts on clicks on the picture draws its buttons first.)
    let clicked = over && f.ptr.clicked;
    if clicked {
        f.ptr.clicked = false;
    }
    clicked
}

/// Rectangles for buttons side by side from (x, y), each as wide as its words need,
/// `h` high, `gap` apart.
pub fn button_row(f: &Frame, words: &[&str], x: f32, y: f32, h: f32, size: f32, gap: f32) -> Vec<Rect> {
    let mut left = x;
    words
        .iter()
        .map(|w| {
            let width = f.text_size(w, size).x + 18.0;
            let r = (Vec2::new(left, y), Vec2::new(left + width, y + h));
            left += width + gap;
            r
        })
        .collect()
}

/// `k` rectangles of equal width filling x0 to x1, `h` high, `gap` apart.
pub fn even_row(k: usize, x0: f32, x1: f32, y: f32, h: f32, gap: f32) -> Vec<Rect> {
    let k = k.max(1);
    let width = ((x1 - x0 - gap * (k - 1) as f32) / k as f32).max(0.0);
    (0..k).map(|i| (Vec2::new(x0 + i as f32 * (width + gap), y), Vec2::new(x0 + i as f32 * (width + gap) + width, y + h))).collect()
}

// ----- numbers that glide --------------------------------------------------------------

/// Numbers shown gliding to where they have gone: what is drawn follows what was
/// computed through two short eases, so that a picture whose numbers arrive in steps
/// (a count, a new sample, a setting changed) moves in every frame and comes exactly
/// to rest. Keep the true numbers apart from these: captions and read-outs say the
/// true ones.
#[derive(Clone, Debug, Default)]
pub struct Glide {
    between: Vec<f32>,
    pub shown: Vec<f32>,
}

impl Glide {
    /// How long one of the two eases takes (seconds), and how near is there.
    const EASE: f32 = 0.07;
    const THERE: f32 = 1e-4;

    /// Bring the numbers `dt` seconds nearer to `to`. With `keep`, a list that has
    /// grown or shrunk keeps what it had in common; otherwise it starts where it is
    /// going. Says whether anything is still on its way (the scene is then animated).
    pub fn follow(&mut self, to: &[f32], dt: f32, keep: bool) -> bool {
        if self.shown.len() != to.len() {
            if keep && !self.shown.is_empty() {
                let n = self.shown.len().min(to.len());
                self.shown.truncate(n);
                self.between.truncate(n);
                self.shown.extend_from_slice(&to[n..]);
                self.between.extend_from_slice(&to[n..]);
            } else {
                self.snap(to);
            }
        }
        if self.shown == to {
            return false;
        }
        let k = 1.0 - (-dt.max(0.0) / Self::EASE).exp();
        // (how near is there, by the size of the numbers: a number in the thousands
        // cannot come within a ten-thousandth of another in f32, and would glide for ever)
        let (mut far, mut moved) = (false, false);
        for ((b, s), t) in self.between.iter_mut().zip(self.shown.iter_mut()).zip(to) {
            let (b0, s0) = (*b, *s);
            *b += (t - *b) * k;
            *s += (*b - *s) * k;
            moved |= *b != b0 || *s != s0;
            let near = Self::THERE * t.abs().max(1.0);
            far |= !((t - *s).abs() <= near && (t - *b).abs() <= near);
        }
        // (exactly there in the end, not nearly: the picture then stops changing; and
        // if a frame's step was too small to change anything, this is as near as it gets)
        if !far || !moved {
            self.snap(to);
            return false;
        }
        true
    }

    pub fn snap(&mut self, to: &[f32]) {
        self.shown = to.to_vec();
        self.between = to.to_vec();
    }

    /// The `i`th number as shown (0 beyond the list).
    pub fn get(&self, i: usize) -> f32 {
        self.shown.get(i).copied().unwrap_or(0.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ticks_are_round_and_inside() {
        let t = Ticks::of(-0.3, 2.2, 300.0);
        let v: Vec<f64> = t.values().collect();
        assert_eq!(t.step, 0.5);
        assert_eq!(v, vec![0.0, 0.5, 1.0, 1.5, 2.0]);
        assert_eq!(t.name(1.5), "1.5");
        assert_eq!(Ticks::of(0.0, 100.0, 240.0).name(50.0), "50");
        // a range that ends on a tick keeps it, and a step of a tenth is written with one decimal
        let unit = Ticks::of(0.0, 1.0, 300.0);
        assert_eq!(unit.values().map(|v| unit.name(v)).collect::<Vec<_>>(), vec!["0.0", "0.2", "0.4", "0.6", "0.8", "1.0"]);
        let tenths = Ticks::of(0.0, 0.5, 300.0);
        assert_eq!(tenths.values().map(|v| tenths.name(v)).collect::<Vec<_>>(), vec!["0.0", "0.1", "0.2", "0.3", "0.4", "0.5"]);
        assert_eq!(Ticks::of(1.0, 1.0, 240.0).values().count(), 0);
        assert_eq!(Ticks::of(0.0, f64::NAN, 240.0).values().count(), 0);
    }

    #[test]
    fn a_plot_puts_values_where_its_ranges_say() {
        let p = Plot::new((Vec2::new(100.0, 50.0), Vec2::new(300.0, 150.0)), [0.0, 4.0], [0.0, 1.0]);
        assert_eq!(p.at(0.0, 0.0), Vec2::new(100.0, 150.0));
        assert_eq!(p.at(4.0, 1.0), Vec2::new(300.0, 50.0));
        assert!((p.x_at(200.0) - 2.0).abs() < 1e-9 && (p.y_at(100.0) - 0.5).abs() < 1e-9);
        // a range that is no range is still drawable
        let q = Plot::new(p.rect, [2.0, 2.0], [f64::NAN, 1.0]);
        assert!(q.at(2.0, 0.5).is_finite());
    }

    #[test]
    fn a_line_is_cut_at_the_edge_of_its_chart() {
        let r = (Vec2::new(0.0, 0.0), Vec2::new(10.0, 10.0));
        let near = |a: Vec2, b: Vec2| (a - b).length() < 0.6;
        // up and out through the top, and back in: two pieces
        let cut = clip(&[Vec2::new(0.0, 5.0), Vec2::new(4.0, -5.0), Vec2::new(8.0, 5.0)], r);
        assert_eq!(cut.len(), 5, "{cut:?}");
        assert_eq!(cut[0], Vec2::new(0.0, 5.0));
        assert!(near(cut[1], Vec2::new(2.0, 0.0)) && cut[2].is_nan() && near(cut[3], Vec2::new(6.0, 0.0)), "{cut:?}");
        assert_eq!(cut[4], Vec2::new(8.0, 5.0));
        // wholly outside: nothing to draw
        assert!(clip(&[Vec2::new(-5.0, -5.0), Vec2::new(-1.0, -9.0)], r).is_empty());
        // both ends outside and the middle inside: the part inside is kept
        let across = clip(&[Vec2::new(-10.0, 5.0), Vec2::new(20.0, 5.0)], r);
        assert_eq!(across.len(), 2, "{across:?}");
        assert!(near(across[0], Vec2::new(0.0, 5.0)) && near(across[1], Vec2::new(10.0, 5.0)), "{across:?}");
        // a line wholly inside comes through as it is, with no breaks
        let inside = [Vec2::new(1.0, 1.0), Vec2::new(5.0, 9.0), Vec2::new(9.0, 2.0)];
        assert_eq!(clip(&inside, r), inside.to_vec());
        // a break in the line stays a break
        let broken = clip(&[Vec2::new(1.0, 1.0), Vec2::new(2.0, 2.0), Vec2::NAN, Vec2::new(3.0, 3.0), Vec2::new(4.0, 4.0)], r);
        assert_eq!(broken.iter().filter(|p| p.is_nan()).count(), 1, "{broken:?}");
        assert_eq!(broken.len(), 5);
    }

    #[test]
    fn a_glide_arrives_exactly_and_then_rests() {
        let mut g = Glide::default();
        assert!(!g.follow(&[1.0, 2.0], 0.016, false));
        assert_eq!(g.shown, vec![1.0, 2.0]);
        let mut frames = 0;
        while g.follow(&[3.0, 2.0], 1.0 / 60.0, false) {
            frames += 1;
            assert!(g.get(0) > 1.0 && g.get(0) < 3.0);
            assert!(frames < 200);
        }
        assert!(frames > 5);
        assert_eq!(g.shown, vec![3.0, 2.0]);
        // a longer list keeps what it had
        g.follow(&[3.0, 2.0, 7.0], 0.016, true);
        assert_eq!(g.shown, vec![3.0, 2.0, 7.0]);
        // large numbers arrive too, at every display rate (they once stalled a few
        // steps of f32 short and kept the scene moving for ever)
        for hz in [60.0, 75.0, 120.0, 144.0, 240.0] {
            for (from, to) in [(900.0, 1000.0), (200.0, 260.0), (0.0, 123_456.0), (1e6, 1.0)] {
                let mut g = Glide::default();
                g.snap(&[from]);
                let mut frames = 0;
                while g.follow(&[to], 1.0 / hz, false) {
                    frames += 1;
                    assert!(frames < 5000, "{from} to {to} at {hz} Hz never arrives");
                }
                assert_eq!(g.shown, vec![to]);
            }
        }
    }
}

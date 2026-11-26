//! The scene toolkit: drawing helpers that more than one topic needs. Everything
//! here is re-exported by the prelude. Put a helper here when a second scene wants it.

use crate::frame::Frame;
use crate::gfx::camera::View;
use crate::gfx::color::{pal, Color};
use crate::typeset::has_math;
use glam::{Vec2, Vec3};

/// Is the step from sample `i - 1` to sample `i` a leap across a pole or a jump, not a
/// stretch of the curve? Told by the samples either side: the curve was heading one
/// way, leaps back the other way by more than it was moving, and then carries on as
/// before. That is `tan x` at its poles and `1/x` at zero however far the view is
/// zoomed out, and the fall of a sawtooth; a smooth curve, however steep, never
/// does it. (The riser of a staircase, flat on both sides, is left drawn.)
fn leaps(ys: &[f32], i: usize) -> bool {
    if i < 2 || i + 1 >= ys.len() {
        return false;
    }
    let (before, jump, after) = (ys[i - 1] - ys[i - 2], ys[i] - ys[i - 1], ys[i + 1] - ys[i]);
    jump * before < 0.0 && jump * after < 0.0 && jump.abs() > before.abs().max(after.abs())
}

/// Push `ys` over `xs` as a polyline, breaking at gaps and at vertical asymptotes.
pub fn plot_samples(f: &mut Frame, xs: &[f32], ys: &[f32], width: f32, color: Color, glow: bool) {
    let (lo, hi) = f.cam.flat_bounds();
    let span = (hi.y - lo.y).max(1e-9);
    let b = if glow { &mut f.dl.glow_lines } else { &mut f.dl.lines };
    b.brk();
    let mut prev = f32::NAN;
    for (i, (&x, &y)) in xs.iter().zip(ys).enumerate() {
        if !y.is_finite() || (y - prev).abs() > span * 6.0 || leaps(ys, i) {
            b.brk();
        }
        if y.is_finite() {
            // keep far-off-screen points from producing gigantic segments
            b.push(Vec3::new(x, y.clamp(lo.y - span * 4.0, hi.y + span * 4.0), 0.0), width, color);
        }
        prev = y;
    }
    b.brk();
}

/// Shade the area between a curve and the level `base`: the curve as points from
/// one end to the other, in the plane z = 0.
pub fn fill_under(f: &mut Frame, curve: impl IntoIterator<Item = Vec2>, base: f32, color: Color) {
    let mut before: Option<Vec2> = None;
    for p in curve {
        if let Some(q) = before {
            f.dl.tris_alpha.quad(Vec3::new(q.x, base, 0.0), Vec3::new(p.x, base, 0.0), p.extend(0.0), q.extend(0.0), color);
        }
        before = Some(p);
    }
}

/// Histogram bars standing on `base`, with a soft vertical gradient.
pub fn bars(f: &mut Frame, x0: f32, width: f32, heights: &[f32], base: f32, color: Color) {
    for (i, &h) in heights.iter().enumerate() {
        if h <= 0.0 {
            continue;
        }
        let (a, b) = (x0 + width * i as f32 + width * 0.06, x0 + width * (i + 1) as f32 - width * 0.06);
        let p = [Vec3::new(a, base, 0.0), Vec3::new(b, base, 0.0), Vec3::new(b, base + h, 0.0), Vec3::new(a, base + h, 0.0)];
        f.dl.tris_alpha.quad_grad(p, [color.a(0.25), color.a(0.25), color.a(0.8), color.a(0.8)]);
    }
}

/// A small line chart in world coordinates (used for loss curves). The vertical
/// range fits the data.
pub fn chart(f: &mut Frame, origin: Vec2, size: Vec2, series: &[(&[f32], Color)], log: bool) -> Vec<Vec2> {
    chart_in(f, origin, size, series, log, None)
}

/// The same with a fixed vertical range, if one is given. Returns where each curve
/// ends, so that it can be named right there (see [`spread_labels`]).
pub fn chart_in(f: &mut Frame, origin: Vec2, size: Vec2, series: &[(&[f32], Color)], log: bool, range: Option<[f32; 2]>) -> Vec<Vec2> {
    f.dl.tris_alpha.rect(origin - Vec2::splat(0.05), origin + size + Vec2::splat(0.05), 0.0, Color::hex(0x0A0D16).a(0.8));
    f.line(origin, origin + Vec2::new(size.x, 0.0), 1.0, pal::DIM.a(0.8));
    let tr = |v: f32| if log { v.max(1e-6).ln() } else { v };
    let (mut lo, mut hi) = (f32::MAX, f32::MIN);
    for (s, _) in series {
        for &v in s.iter().filter(|v| v.is_finite()) {
            lo = lo.min(tr(v));
            hi = hi.max(tr(v));
        }
    }
    if let Some([a, b]) = range {
        (lo, hi) = (tr(a), tr(b));
    }
    if lo >= hi {
        hi = lo + 1.0;
    }
    let mut ends = Vec::with_capacity(series.len());
    for (s, c) in series {
        let n = s.len();
        let at = |i: usize, v: f32| origin + Vec2::new(size.x * i as f32 / (n.max(2) - 1) as f32, size.y * ((tr(v) - lo) / (hi - lo)).clamp(0.0, 1.0));
        ends.push(s.iter().enumerate().rev().find(|(_, v)| v.is_finite()).map_or(origin, |(i, &v)| at(i, v)));
        if n < 2 {
            continue;
        }
        f.path(s.iter().enumerate().filter(|(_, v)| v.is_finite()).map(|(i, &v)| at(i, v)), 1.8, *c);
    }
    ends
}

/// A caption along the bottom of the view, on a dark plate, naming what the picture
/// shows that has no label of its own: `parts` side by side for as long as they fit
/// across the view, and on further lines above when they do not (a phone held
/// upright). Left out of a view too small to spare the room.
pub fn bottom_caption(f: &mut Frame, parts: &[&str], size: f32) {
    let view = f.view_size();
    if view.y < 300.0 {
        return;
    }
    let room = view.x - 36.0;
    let gap = "      ";
    // (with mathematics in a part, every part is measured exactly as it is drawn)
    let exact = parts.iter().any(|p| has_math(p));
    let width = |text: &str| if exact { f.text_size(text, size).x } else { text_width(text, size) };
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
        f.caption_px(Vec2::new(18.0, view.y - 18.0 - up), line, size, pal::DIM, egui::Align2::LEFT_BOTTOM);
    }
}

/// Name things right where they are: each label is drawn at its screen position
/// (logical pixels), moved down just far enough that no two cover each other.
pub fn spread_labels(f: &mut Frame, mut labels: Vec<(Vec2, String, Color)>, size: f32, align: egui::Align2) {
    let line = size + 3.0;
    labels.sort_by(|a, b| a.0.y.total_cmp(&b.0.y));
    let width = |text: &str| if has_math(text) { f.text_size(text, size).x.max(4.0 * size * 0.62) } else { (text.chars().count().max(4) as f32) * size * 0.62 };
    for i in 1..labels.len() {
        for j in 0..i {
            let (above, here) = (labels[j].0, labels[i].0);
            // two left-aligned labels collide if the one further left reaches the other
            let reach = if above.x <= here.x { width(&labels[j].1) } else { width(&labels[i].1) };
            if (above.x - here.x).abs() < reach && here.y < above.y + line {
                labels[i].0.y = above.y + line;
            }
        }
    }
    for (at, text, color) in labels {
        f.text_px(at, text, size, color, align);
    }
}

/// Where a point of the scene is on screen, in logical pixels (if it is in view).
pub fn on_screen(f: &Frame, p: Vec3) -> Option<Vec2> {
    f.cam.project(p).map(|s| s / f.px)
}

pub const SUB: [&str; 10] = ["₀", "₁", "₂", "₃", "₄", "₅", "₆", "₇", "₈", "₉"];

/// A number as subscript digits.
pub fn sub(n: usize) -> String {
    n.to_string().chars().map(|c| SUB[c as usize - '0' as usize]).collect()
}

pub const SUP: [&str; 10] = ["⁰", "¹", "²", "³", "⁴", "⁵", "⁶", "⁷", "⁸", "⁹"];

/// A number as superscript digits.
pub fn sup(n: usize) -> String {
    n.to_string().chars().map(|c| SUP[c as usize - '0' as usize]).collect()
}

/// A dark backing panel for an inset (a small chart or legend drawn inside a 2D view),
/// in world coordinates.
pub fn inset(f: &mut Frame, min: Vec2, max: Vec2) {
    f.dl.tris_alpha.rect(min, max, 0.0, Color::hex(0x0A0D16).a(0.88));
}

// ----- screen space ------------------------------------------------------------------
//
// Panels that must stay put whatever the camera does (2D or 3D) are drawn in logical
// pixels measured from the top-left corner of the view, y downwards. Text goes with
// `Frame::text_px`.

fn hud_quad(f: &mut Frame, min: Vec2, max: Vec2, color: Color, level: usize) {
    let c = [min, Vec2::new(max.x, min.y), max, Vec2::new(min.x, max.y)].map(|p| f.hud_at(level, p));
    f.dl.tris.quad(c[0], c[1], c[2], c[3], color);
}

/// A filled rectangle in screen space, in front of any panel.
pub fn hud_rect(f: &mut Frame, min: Vec2, max: Vec2, color: Color) {
    hud_quad(f, min, max, color, 1);
}

/// A filled rectangle in screen space on a level of your choosing (see
/// [`Frame::hud_at`]): 2 for a mark on top of what `hud_rect` drew.
pub fn hud_rect_at(f: &mut Frame, level: usize, min: Vec2, max: Vec2, color: Color) {
    hud_quad(f, min, max, color, level);
}

/// The dark backing of a screen-space panel.
pub fn hud_panel(f: &mut Frame, min: Vec2, max: Vec2) {
    hud_quad(f, min, max, Color::hex(0x080B14).a(0.95), 0);
}

/// A line in screen space, on top of everything.
pub fn hud_line(f: &mut Frame, a: Vec2, b: Vec2, width: f32, color: Color) {
    let (a, b) = (f.hud(a), f.hud(b));
    f.dl.overlay.segment(a, b, width, color);
}

/// A small line chart in screen space: `origin` is its top-left corner. The vertical
/// range fits the data unless one is given.
pub fn hud_chart(f: &mut Frame, origin: Vec2, size: Vec2, series: &[(&[f32], Color)], log: bool, range: Option<[f32; 2]>) {
    hud_panel(f, origin - Vec2::splat(8.0), origin + size + Vec2::splat(8.0));
    hud_line(f, origin + Vec2::new(0.0, size.y), origin + size, 1.0, pal::DIM.a(0.8));
    let tr = |v: f32| if log { v.max(1e-6).ln() } else { v };
    let (mut lo, mut hi) = (f32::MAX, f32::MIN);
    for (s, _) in series {
        for &v in s.iter().filter(|v| v.is_finite()) {
            lo = lo.min(tr(v));
            hi = hi.max(tr(v));
        }
    }
    if let Some([a, b]) = range {
        (lo, hi) = (tr(a), tr(b));
    }
    if lo >= hi {
        hi = lo + 1.0;
    }
    for (s, c) in series {
        let n = s.len();
        if n < 2 {
            continue;
        }
        let pts: Vec<Vec3> = s
            .iter()
            .enumerate()
            .filter(|(_, v)| v.is_finite())
            .map(|(i, &v)| f.hud(origin + Vec2::new(size.x * i as f32 / (n - 1) as f32, size.y * (1.0 - ((tr(v) - lo) / (hi - lo)).clamp(0.0, 1.0)))))
            .collect();
        f.dl.overlay.polyline(pts, 1.8, *c);
    }
}

/// A colour key in screen space: one swatch and meaning per row, starting at
/// `top_left`. Returns the height used. (Draw a `hud_panel` behind it first if it
/// lies over the picture.)
pub fn hud_key(f: &mut Frame, top_left: Vec2, entries: &[(Color, &str)], size: f32) -> f32 {
    let row = size + 6.0;
    for (i, (color, meaning)) in entries.iter().enumerate() {
        let at = top_left + Vec2::new(0.0, i as f32 * row);
        hud_rect(f, at + Vec2::new(0.0, 1.0), at + Vec2::new(size - 1.0, size), *color);
        f.text_px(at + Vec2::new(size + 6.0, size * 0.5), *meaning, size, pal::TEXT, egui::Align2::LEFT_CENTER);
    }
    entries.len() as f32 * row
}

/// A matrix of coloured cells in screen space (a heat map): `rows` by `cols` cells
/// filling the rectangle from `top_left` of the given `size`, cell (i, j) coloured by
/// `color(i, j)`; row 0 is at the top.
pub fn hud_heat(f: &mut Frame, top_left: Vec2, size: Vec2, rows: usize, cols: usize, mut color: impl FnMut(usize, usize) -> Color) {
    let cell = Vec2::new(size.x / cols.max(1) as f32, size.y / rows.max(1) as f32);
    // a hair of space between cells while they are big enough to afford it
    let gap = Vec2::new(if cell.x > 6.0 { 1.0 } else { 0.0 }, if cell.y > 6.0 { 1.0 } else { 0.0 });
    for i in 0..rows {
        for j in 0..cols {
            let at = top_left + Vec2::new(j as f32 * cell.x, i as f32 * cell.y);
            hud_rect(f, at, at + cell - gap, color(i, j));
        }
    }
}

/// An arrow in screen space (logical pixels), on top of everything.
pub fn hud_arrow(f: &mut Frame, a: Vec2, b: Vec2, width: f32, color: Color) {
    hud_line(f, a, b, width, color);
    let d = (b - a).normalize_or_zero();
    let n = Vec2::new(-d.y, d.x);
    let head = (4.0 + 2.5 * width).min((b - a).length() * 0.5);
    hud_line(f, b, b - d * head + n * head * 0.5, width, color);
    hud_line(f, b, b - d * head - n * head * 0.5, width, color);
}

/// Is the view too narrow (a phone held upright) for a picture laid out side by
/// side? Scenes drawn in world units then start zoomed in on their main picture and
/// leave the rest to be reached by dragging; text keeps its size, so it stays legible.
pub fn narrow_view(f: &Frame) -> bool {
    f.view_size().x < 560.0
}

// ----- a picture that follows the shape of the canvas ---------------------------------
//
// Flat scenes can lay themselves out in a rectangle five units high and as wide as the canvas makes it,
// centred on the origin: wide on a desktop, tall on a phone. The camera always shows
// exactly that rectangle to begin with, so a layout worked out from its width fills
// the canvas whatever its shape, and zooming and panning still work as usual.

/// The height of that rectangle.
pub const STAGE_H: f32 = 5.0;

/// The view that shows it (its width follows from the canvas).
pub const STAGE_VIEW: View = View::Fit { center: Vec2::ZERO, half: Vec2::new(1.0, 2.5) };

/// The rectangle's width on this canvas.
pub fn stage_width(f: &Frame) -> f32 {
    let view = f.view_size();
    (STAGE_H * view.x / view.y.max(1.0)).max(2.0)
}

/// Logical pixels per unit as the picture is now zoomed.
pub fn pixels_per_unit(f: &Frame) -> f32 {
    let (lo, hi) = f.cam.flat_bounds();
    f.view_size().y / (hi.y - lo.y).max(1e-6)
}

/// How many logical pixels a line of text takes, near enough to decide whether it fits,
/// with no frame at hand: by its number of characters, and a formula exactly (from the
/// fonts' metrics). With a frame, `f.text_size` measures exactly. (A text with a formula
/// in it is measured once and kept for as long as it is asked about in every frame:
/// laying a formula out in every frame for every label was a scene's whole cost.)
pub fn text_width(text: &str, size: f32) -> f32 {
    if !has_math(text) {
        return text.chars().count() as f32 * size * 0.47;
    }
    let key = key_of(&(text, size.to_bits()));
    if let Some(width) = WIDTHS.with(|kept| kept.borrow_mut().find(key, text)) {
        return width;
    }
    let width = crate::typeset::estimate_width(text, size, 0.47);
    WIDTHS.with(|kept| kept.borrow_mut().put(key, text, width));
    width
}

// ----- what is kept from frame to frame -----------------------------------------------
//
// Measuring text with formulas in it means laying the formulas out: too dear to do for
// every word of a note in every frame, and the text is mostly the same from one frame to
// the next. What `wrapped` and `text_width` work out is kept, for as long as it is asked
// for in every frame; a text not asked about in a whole frame (no longer shown, or
// changed) is let go.

thread_local! {
    static FRAME: std::cell::Cell<u64> = const { std::cell::Cell::new(0) };
    static WIDTHS: std::cell::RefCell<Kept<f32>> = std::cell::RefCell::new(Kept::default());
    static WRAPPED: std::cell::RefCell<Kept<Lines>> = std::cell::RefCell::new(Kept::default());
}

/// A text's lines as `wrapped` draws them, kept.
type Lines = std::rc::Rc<Wrapped>;

#[cfg(test)]
thread_local! {
    /// How many times a text has been broken into lines afresh (for the tests).
    pub static WRAPS: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
}

/// A new frame begins (the application says so): what was kept and not asked for in
/// the frame before is let go at the next ask.
pub fn next_frame() {
    FRAME.with(|f| f.set(f.get() + 1));
}

/// Results kept by a hash of what they were worked out from, with the text among it to
/// make sure of a match.
struct Kept<V> {
    /// The frame `now` is for; what was asked for in it, and in the frame before.
    frame: u64,
    now: std::collections::HashMap<u64, (Box<str>, V)>,
    before: std::collections::HashMap<u64, (Box<str>, V)>,
}

impl<V> Default for Kept<V> {
    fn default() -> Self {
        Kept { frame: 0, now: Default::default(), before: Default::default() }
    }
}

/// The most a `Kept` holds from one frame: where no frames are counted (a unit test),
/// it is emptied when it gets this full.
const KEPT_MOST: usize = 1024;

fn key_of(what: &impl std::hash::Hash) -> u64 {
    use std::hash::Hasher;
    let mut h = std::collections::hash_map::DefaultHasher::new();
    what.hash(&mut h);
    h.finish()
}

impl<V: Clone> Kept<V> {
    fn turn(&mut self) {
        let frame = FRAME.with(std::cell::Cell::get);
        if frame != self.frame {
            self.before = std::mem::take(&mut self.now);
            // (frames went by without a single ask: nothing was asked for in the last)
            if frame != self.frame + 1 {
                self.before.clear();
            }
            self.frame = frame;
        }
    }

    fn find(&mut self, key: u64, text: &str) -> Option<V> {
        self.turn();
        if let Some((check, value)) = self.now.get(&key) {
            return (&**check == text).then(|| value.clone());
        }
        let (check, value) = self.before.remove(&key)?;
        if &*check != text {
            return None;
        }
        self.now.insert(key, (check, value.clone()));
        Some(value)
    }

    fn put(&mut self, key: u64, text: &str, value: V) {
        self.turn();
        if self.now.len() >= KEPT_MOST {
            self.now.clear();
        }
        self.now.insert(key, (text.into(), value));
    }
}

/// The words of a text, each with its formulas whole (a formula is not broken
/// across lines here).
fn words_of(text: &str) -> Vec<&str> {
    if !has_math(text) {
        return text.split(' ').collect();
    }
    let mut words = Vec::new();
    let (mut start, mut inside) = (0, false);
    let bytes = text.as_bytes();
    let mut i = 0;
    while i < bytes.len() {
        match bytes[i] {
            b'\\' => i += 1,
            b'$' => inside = !inside,
            b' ' if !inside => {
                words.push(&text[start..i]);
                start = i + 1;
            }
            _ => {}
        }
        i += 1;
    }
    words.push(&text[start.min(text.len())..]);
    words
}

/// Text broken into lines no wider than `width` units, the first line's top-left
/// corner at `at`. Returns how many lines it took.
///
/// Lines are `size + 3` pixels apart, and a line that holds a formula taller than the
/// words (a root, a fraction, a superscript) further, by just as much as it needs to
/// keep clear of the lines either side ([`wrapped_height`] says how tall that makes
/// it). A text with formulas in it is broken into lines once, and kept so while it is
/// drawn in every frame at the same size and width.
pub fn wrapped(f: &mut Frame, at: Vec2, width: f32, text: &str, size: f32, color: Color) -> usize {
    let set = wrap(f, width, text, size);
    for (line, y) in &set.lines {
        f.text_off(at, Vec2::new(0.0, *y), line.as_str(), size, color, egui::Align2::LEFT_TOP);
    }
    set.lines.len()
}

/// How many pixels tall `wrapped` draws `text`: `size + 3` for each line, and the room
/// made for formulas taller than the words.
pub fn wrapped_height(f: &Frame, width: f32, text: &str, size: f32) -> f32 {
    wrap(f, width, text, size).height
}

/// A text's lines as `wrapped` draws them, each with how far below the first line's
/// top it is, and how tall they are together.
struct Wrapped {
    lines: Vec<(String, f32)>,
    height: f32,
}

fn wrap(f: &Frame, width: f32, text: &str, size: f32) -> Lines {
    let room = width * pixels_per_unit(f);
    if has_math(text) {
        let key = key_of(&(text, size.to_bits(), room.to_bits(), f.ctx.pixels_per_point().to_bits()));
        if let Some(kept) = WRAPPED.with(|kept| kept.borrow_mut().find(key, text)) {
            return kept;
        }
        // (before the fonts are in place, the lines are broken afresh each frame)
        if let Some(set) = wrap_math(f, text, size, room) {
            WRAPPED.with(|kept| kept.borrow_mut().put(key, text, set.clone()));
            return set;
        }
    }
    let exact = has_math(text);
    let measure = |f: &Frame, text: &str| if exact { f.text_size(text, size).x } else { text_width(text, size) };
    let mut lines = vec![String::new()];
    for word in words_of(text) {
        let Some(last) = lines.last_mut() else { break };
        if last.is_empty() {
            last.push_str(word);
        } else if measure(f, last) + measure(f, word) + size * 0.47 > room {
            lines.push(word.to_string());
        } else {
            last.push(' ');
            last.push_str(word);
        }
    }
    let height = lines.len() as f32 * (size + 3.0);
    std::rc::Rc::new(Wrapped { lines: lines.into_iter().enumerate().map(|(i, line)| (line, i as f32 * (size + 3.0))).collect(), height })
}

/// A text with formulas in it broken into lines no wider than `room` pixels, as
/// `wrapped` breaks it; none before the fonts are in place. Each word is measured
/// once, and a line by the sum of its words and the spaces between them.
fn wrap_math(f: &Frame, text: &str, size: f32, room: f32) -> Option<Lines> {
    #[cfg(test)]
    WRAPS.with(|n| n.set(n.get() + 1));
    let measure = |text: &str| if has_math(text) { crate::typeset::measure(f.ctx, text, size).map(|s| s.x) } else { Some(f.text_size(text, size).x) };
    let space = measure("x x")? - measure("xx")?;
    let mut lines: Vec<(String, f32)> = Vec::new();
    for word in words_of(text) {
        let width = measure(word)?;
        match lines.last_mut() {
            Some((line, used)) if *used + width + size * 0.47 <= room => {
                line.push(' ');
                line.push_str(word);
                *used += space + width;
            }
            _ => lines.push((word.to_string(), width)),
        }
    }
    // each line a pitch below the last, and further where a formula needs the room
    // (what a line's galley says: `over`, how far its top stands above a plain line's,
    // and `rise` and `drop`, the room its formulas need beyond plain text's)
    let mut placed = Vec::with_capacity(lines.len());
    let (mut top, mut drop) = (0.0f32, 0.0f32);
    for (k, (line, _)) in lines.into_iter().enumerate() {
        let (over, rise, below) = match has_math(&line) {
            true => crate::typeset::label(f.ctx, &line, size).map(|set| (set.over, set.rise, set.drop))?,
            false => (0.0, 0.0, 0.0),
        };
        top = if k == 0 { over } else { top + size + 3.0 + drop + rise };
        placed.push((line, top - over));
        drop = below;
    }
    let height = if placed.is_empty() { 0.0 } else { top + size + 3.0 + drop };
    Some(std::rc::Rc::new(Wrapped { lines: placed, height }))
}

/// The outline of a square panel.
pub fn frame_square(f: &mut Frame, centre: Vec2, half: f32, width: f32, color: Color) {
    let c = [Vec2::new(-half, -half), Vec2::new(half, -half), Vec2::new(half, half), Vec2::new(-half, half)].map(|c| (centre + c).extend(0.0));
    f.dl.lines.closed(&c, width, color);
}

/// The outline of a rectangle in screen space, on top of everything.
pub fn hud_outline(f: &mut Frame, min: Vec2, max: Vec2, width: f32, color: Color) {
    let c = [min, Vec2::new(max.x, min.y), max, Vec2::new(min.x, max.y)];
    for i in 0..4 {
        hud_line(f, c[i], c[(i + 1) % 4], width, color);
    }
}

/// A path in screen space, on top of everything.
pub fn hud_path(f: &mut Frame, points: impl Iterator<Item = Vec2>, width: f32, color: Color) {
    let points: Vec<Vec3> = points.map(|p| f.hud(p)).collect();
    f.dl.overlay.polyline(points, width, color);
}

/// A circle in screen space, on top of everything.
pub fn hud_circle(f: &mut Frame, centre: Vec2, radius: f32, width: f32, color: Color) {
    let sides = ((radius * 0.9) as usize).clamp(16, 96);
    let turn = std::f32::consts::TAU / sides as f32;
    hud_path(f, (0..=sides).map(|i| centre + Vec2::from_angle(i as f32 * turn) * radius), width, color);
}

/// The outline of a rectangle in a flat scene (world units).
pub fn frame_rect(f: &mut Frame, min: Vec2, max: Vec2, width: f32, color: Color) {
    let c = [min, Vec2::new(max.x, min.y), max, Vec2::new(min.x, max.y)].map(|c| c.extend(0.0));
    f.dl.lines.closed(&c, width, color);
}

/// A colour scale drawn as a strip in a flat scene (world units): `cmap` from left
/// to right, `half` wide either side of `centre` and `height` tall, with a label at
/// each end in a colour of its own and, if not empty, one under the middle.
#[allow(clippy::too_many_arguments)]
pub fn ramp_key(f: &mut Frame, centre: Vec2, half: f32, height: f32, cmap: crate::gfx::color::Cmap, low: (&str, Color), high: (&str, Color), middle: &str) {
    let table = cmap.table();
    let strips = 36;
    let x = |t: f32| centre.x - half + 2.0 * half * t;
    let (bottom, top) = (centre.y - height * 0.5, centre.y + height * 0.5);
    for i in 0..strips {
        let (a, b) = (i as f32 / strips as f32, (i + 1) as f32 / strips as f32);
        f.dl.tris_alpha.rect(Vec2::new(x(a), bottom), Vec2::new(x(b), top), 0.0, table[crate::gfx::color::table_index((a + b) * 0.5)]);
    }
    f.text((centre.x - half - 0.08, centre.y), low.0, 11.0, low.1, egui::Align2::RIGHT_CENTER);
    f.text((centre.x + half + 0.08, centre.y), high.0, 11.0, high.1, egui::Align2::LEFT_CENTER);
    if !middle.is_empty() {
        f.text((centre.x, bottom - 0.03), middle, 10.5, pal::DIM, egui::Align2::CENTER_TOP);
    }
}

// ----- histories ----------------------------------------------------------------------

/// What a chart of a long run is drawn from: at most `capacity` points, however
/// many values are pushed. When it is full, neighbours are averaged in pairs and
/// from then on each new point is the average of twice as many values, so the points
/// always stand for equal stretches of the run and the chart's horizontal axis stays
/// true to it.
#[derive(Clone, Debug)]
pub struct History {
    points: Vec<f32>,
    capacity: usize,
    /// Values per point at present, and the part of the next point collected so far.
    every: u32,
    sum: f32,
    count: u32,
}

impl History {
    pub fn new(capacity: usize) -> Self {
        History { points: Vec::new(), capacity: (capacity.max(2) + 1) & !1, every: 1, sum: 0.0, count: 0 }
    }

    pub fn push(&mut self, value: f32) {
        self.sum += value;
        self.count += 1;
        if self.count < self.every {
            return;
        }
        self.points.push(self.sum / self.count as f32);
        (self.sum, self.count) = (0.0, 0);
        if self.points.len() >= self.capacity {
            self.points = self.points.chunks_exact(2).map(|pair| (pair[0] + pair[1]) * 0.5).collect();
            self.every *= 2;
        }
    }

    /// `value` pushed `n` times over: a reading that stands for `n` steps of a run.
    /// It goes into every point those `n` values reach, so that each point still
    /// stands for the same number of steps however unevenly the readings come.
    pub fn push_n(&mut self, value: f32, n: u64) {
        let mut left = n;
        while left > 0 {
            let take = left.min((self.every - self.count) as u64);
            self.sum += value * take as f32;
            self.count += take as u32;
            left -= take;
            if self.count < self.every {
                return;
            }
            self.points.push(self.sum / self.count as f32);
            (self.sum, self.count) = (0.0, 0);
            if self.points.len() >= self.capacity {
                self.points = self.points.chunks_exact(2).map(|pair| (pair[0] + pair[1]) * 0.5).collect();
                self.every *= 2;
            }
        }
    }

    /// The points, oldest first.
    pub fn points(&self) -> &[f32] {
        &self.points
    }

    /// How many pushed values each point stands for.
    pub fn every(&self) -> u32 {
        self.every
    }

    pub fn is_empty(&self) -> bool {
        self.points.is_empty()
    }

    pub fn clear(&mut self) {
        self.points.clear();
        (self.every, self.sum, self.count) = (1, 0.0, 0);
    }
}

// ----- numbers and names --------------------------------------------------------------

/// A number to `decimals` places, with a true minus sign, and a plain 0 (to those
/// places) for anything that rounds to zero; "?" for a number that is not finite.
pub fn num(v: f64, decimals: usize) -> String {
    if !v.is_finite() {
        return "?".to_string();
    }
    let digits = format!("{:.*}", decimals, v.abs());
    if v < 0.0 && digits.chars().any(|c| c.is_ascii_digit() && c != '0') {
        format!("−{digits}")
    } else {
        digits
    }
}

/// Part of a formula in a colour of the scene's (the colour of the arrow it names):
/// `\textcolor{#rrggbb}{body}`, to go between dollars.
pub fn tint(color: Color, body: &str) -> String {
    let [r, g, b, a] = color.egui().to_srgba_unmultiplied();
    let alpha = if a == 255 { String::new() } else { format!("{a:02x}") };
    format!("\\textcolor{{#{r:02x}{g:02x}{b:02x}{alpha}}}{{{body}}}")
}

/// A matrix of numbers as markup, to go between dollars: `\begin{bmatrix} 1.50 & −0.50
/// \\ … \end{bmatrix}`, each number to `decimals` places (as [`num`] writes them).
pub fn bmatrix<T: Copy + Into<f64>, R: AsRef<[T]>>(rows: &[R], decimals: usize) -> String {
    matrix_markup("bmatrix", rows, decimals)
}

/// The same in round brackets.
pub fn pmatrix<T: Copy + Into<f64>, R: AsRef<[T]>>(rows: &[R], decimals: usize) -> String {
    matrix_markup("pmatrix", rows, decimals)
}

/// A column vector of numbers, in square brackets, as markup.
pub fn column<T: Copy + Into<f64>>(values: &[T], decimals: usize) -> String {
    let rows: Vec<[T; 1]> = values.iter().map(|v| [*v]).collect();
    matrix_markup("bmatrix", &rows, decimals)
}

/// Numbers in round brackets, separated by commas: a point, `(1.50, −0.50)`.
pub fn tuple<T: Copy + Into<f64>>(values: &[T], decimals: usize) -> String {
    let numbers: Vec<String> = values.iter().map(|v| num((*v).into(), decimals)).collect();
    format!("({})", numbers.join(", "))
}

fn matrix_markup<T: Copy + Into<f64>, R: AsRef<[T]>>(env: &str, rows: &[R], decimals: usize) -> String {
    let rows: Vec<String> = rows.iter().map(|row| row.as_ref().iter().map(|v| num((*v).into(), decimals)).collect::<Vec<_>>().join(" & ")).collect();
    format!("\\begin{{{env}}} {} \\end{{{env}}}", rows.join(" \\\\ "))
}

/// An activation's name as a panel shows it, the ones that are formulas typeset
/// (`Act::name` is the plain name, for code and tests).
pub fn activation_name(act: crate::math::nn::Act) -> &'static str {
    use crate::math::nn::Act;
    match act {
        Act::Hermite2 => r"$(x^2 - 1)/\sqrt{2}$",
        Act::Hermite3 => r"$(x^3 - 3x)/\sqrt{6}$",
        Act::ReluSquared => "ReLU$^2$",
        other => other.name(),
    }
}

/// Distance from `p` to the segment from `a` to `b`.
pub fn seg_dist(p: Vec2, a: Vec2, b: Vec2) -> f32 {
    let ab = b - a;
    let t = ((p - a).dot(ab) / ab.length_squared().max(1e-12)).clamp(0.0, 1.0);
    p.distance(a + ab * t)
}

/// What tells one label of a scene from its others, from frame to frame: a word, and
/// a number for the labels of things there are several of. `"tip"` and `("tip", 2)`
/// both pass for one.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Tag(pub &'static str, pub u32);

impl From<&'static str> for Tag {
    fn from(name: &'static str) -> Tag {
        Tag(name, 0)
    }
}

impl From<(&'static str, u32)> for Tag {
    fn from((name, number): (&'static str, u32)) -> Tag {
        Tag(name, number)
    }
}

impl From<(&'static str, usize)> for Tag {
    fn from((name, number): (&'static str, usize)) -> Tag {
        Tag(name, number as u32)
    }
}

/// Where labels can go without covering a line or one another: what is drawn so
/// far, as segments on the screen (logical pixels).
pub struct Room {
    busy: Vec<(Vec2, Vec2)>,
    /// Flat shapes on the screen (corners in order round them).
    shapes: Vec<Vec<Vec2>>,
    /// Boxes on the screen no label may enter at all (the sums, the buttons).
    walls: Vec<(Vec2, Vec2)>,
    view: Vec2,
}

/// Distance from the segment `a`–`b` to the box `min`–`max` (0 if it crosses it).
pub fn seg_box_dist(a: Vec2, b: Vec2, min: Vec2, max: Vec2) -> f32 {
    // does the segment pass through the box? (clipped against each pair of sides)
    let (mut t0, mut t1) = (0.0f32, 1.0f32);
    let d = b - a;
    let mut crosses = true;
    for (p, q) in [(-d.x, a.x - min.x), (d.x, max.x - a.x), (-d.y, a.y - min.y), (d.y, max.y - a.y)] {
        if p == 0.0 {
            if q < 0.0 {
                crosses = false;
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
    if crosses && t0 <= t1 {
        return 0.0;
    }
    let to_box = |p: Vec2| (p - p.clamp(min, max)).length();
    let corners = [min, Vec2::new(max.x, min.y), max, Vec2::new(min.x, max.y)];
    corners.iter().map(|c| seg_dist(*c, a, b)).fold(to_box(a).min(to_box(b)), f32::min)
}

impl Room {
    /// An empty picture the size of the view. (The strip under the application's
    /// buttons, at the top, is walled off from the start.)
    pub fn new(f: &Frame) -> Room {
        let view = f.view_size();
        let mut room = Room { busy: Vec::new(), shapes: Vec::new(), walls: Vec::new(), view };
        if f.top_inset > 0.0 {
            room.wall(Vec2::ZERO, Vec2::new(view.x, f.top_inset));
        }
        room
    }

    /// Keep labels out of a box on the screen altogether (a panel, a row of buttons).
    pub fn wall(&mut self, min: Vec2, max: Vec2) {
        self.walls.push((min, max));
    }

    /// Keep labels off the segment from `a` to `b` on the screen (logical pixels).
    pub fn segment_px(&mut self, a: Vec2, b: Vec2) {
        self.busy.push((a, b));
    }

    /// Keep labels off the segment from `a` to `b` (in the scene).
    pub fn line(&mut self, f: &Frame, a: Vec3, b: Vec3) {
        if let (Some(a), Some(b)) = (on_screen(f, a), on_screen(f, b)) {
            self.busy.push((a, b));
        }
    }

    /// Keep labels off a box on the screen.
    pub fn boxed(&mut self, min: Vec2, max: Vec2) {
        let c = [min, Vec2::new(max.x, min.y), max, Vec2::new(min.x, max.y)];
        for i in 0..4 {
            self.busy.push((c[i], c[(i + 1) % 4]));
        }
        self.busy.push((Vec2::new(min.x, (min.y + max.y) * 0.5), Vec2::new(max.x, (min.y + max.y) * 0.5)));
    }

    /// How clear the box is: its distance from everything busy, and less than nothing
    /// if it is not all on the screen.
    pub fn clearance(&self, min: Vec2, max: Vec2) -> f32 {
        let off = (-min.x).max(-min.y).max(max.x - self.view.x).max(max.y - self.view.y);
        if off > 0.0 {
            return -off;
        }
        if self.walls.iter().any(|(lo, hi)| min.x < hi.x && max.x > lo.x && min.y < hi.y && max.y > lo.y) {
            return -1.0;
        }
        // The nearest of the busy segments. One whose own bounding box is clearly
        // further from this box than the nearest found so far cannot be the nearest,
        // and is passed over without its distance being worked out: the same answer,
        // found several times sooner where much is drawn (this search was most of
        // what several scenes cost in a frame).
        let mut nearest = f32::MAX;
        for (a, b) in &self.busy {
            let gap = (a.min(*b) - max).max(min - a.max(*b)).max(Vec2::ZERO);
            if gap.length_squared() > nearest * nearest * 1.001 {
                continue;
            }
            nearest = nearest.min(seg_box_dist(*a, *b, min, max));
        }
        nearest
    }

    /// Keep the labels of things outside the scene's flat shape with these corners
    /// (in order round it) outside it too.
    pub fn shape(&mut self, f: &Frame, corners: &[Vec3]) {
        let on: Vec<Vec2> = corners.iter().filter_map(|c| on_screen(f, *c)).collect();
        if on.len() == corners.len() {
            self.shapes.push(on);
        }
    }

    /// How much room a label's box centred at `centre` has, as the name of the point
    /// drawn at `at` (a point outside a shape is not named from inside it).
    pub fn room_for(&self, at: Vec2, centre: Vec2, half: Vec2) -> f32 {
        let (min, max) = (centre - half, centre + half);
        let room = self.clearance(min, max);
        let enters = |s: &Vec<Vec2>| !inside(at, s) && [min, max, Vec2::new(min.x, max.y), Vec2::new(max.x, min.y), centre].iter().any(|c| inside(*c, s));
        if self.shapes.iter().any(enters) {
            room.min(0.0)
        } else {
            room
        }
    }

    /// Write `text` beside the scene's point `at`, `gap` pixels off it, on whichever
    /// side is clear, trying first the side towards `towards` (a direction on the
    /// screen; zero tries the point itself first). Returns false if no side was clear
    /// and `optional` asked for nothing to be written then. `name` tells the label
    /// from the scene's others, from frame to frame (see `label_any`): a word, or a
    /// word and a number, ("tip", 2).
    #[allow(clippy::too_many_arguments)]
    pub fn label(
        &mut self,
        f: &mut Frame,
        name: impl Into<Tag>,
        at: Vec3,
        text: &str,
        size: f32,
        color: Color,
        towards: Vec2,
        gap: f32,
        optional: bool,
    ) -> bool {
        self.label_any(f, name, &[at], text, size, color, towards, gap, optional)
    }

    /// The same for the first of several points that has a clear side.
    ///
    /// A label keeps its place from frame to frame where it can. Its first choice
    /// (the side asked for, of the first point) is taken whenever it is clear, and
    /// moves smoothly as the picture does. Failing that it stays where it was last
    /// frame, carried along by the first point, for as long as it has a little room
    /// there; only then does it look for a new place, and it is drawn gliding to it
    /// (`Frame::text_placed`). Looking afresh every frame, it would jump to and fro
    /// between two places about as clear as each other all the while an arrow is
    /// dragged.
    #[allow(clippy::too_many_arguments)]
    pub fn label_any(
        &mut self,
        f: &mut Frame,
        name: impl Into<Tag>,
        points: &[Vec3],
        text: &str,
        size: f32,
        color: Color,
        towards: Vec2,
        gap: f32,
        optional: bool,
    ) -> bool {
        // (a little wider than the text is likely to be: a label must not touch a line;
        // one with mathematics in it is measured exactly)
        let half =
            if has_math(text) { f.text_size(text, size) * 0.5 + Vec2::new(4.0, 0.0) } else { Vec2::new(text_width(text, size) * 0.5 + 4.0, size * 0.5 + 1.0) };
        let want = towards.normalize_or_zero();
        let mut sides: Vec<Vec2> = (0..8).map(|k| Vec2::from_angle(k as f32 * std::f32::consts::TAU / 8.0)).collect();
        sides.sort_by(|p, q| q.dot(want).total_cmp(&p.dot(want)));
        // (the side asked for exactly, before the nearest of the eight)
        sides.insert(0, want);
        // the point the label's place is kept from, and where that puts it now
        let from = points.first().and_then(|p| on_screen(f, *p));
        let Tag(name, number) = name.into();
        let kept = from.zip(f.places.chosen(name, number)).map(|(from, offset)| (from, from + offset));
        let mut best: Option<(f32, Vec2)> = None;
        'points: for (n, at) in points.iter().filter_map(|p| on_screen(f, *p)).enumerate() {
            for (k, side) in sides.iter().enumerate() {
                // the box's nearest point to `at` is about `gap` away along `side`
                let centre = if *side == Vec2::ZERO { at } else { at + *side * gap + Vec2::new(side.x * half.x, side.y * half.y) };
                let room = self.room_for(at, centre, half);
                if best.is_none_or(|b| room > b.0) {
                    best = Some((room, centre));
                }
                if room >= 5.0 {
                    break 'points;
                }
                // the first choice is not clear: where it was, if there is still room
                if let (0, 0, Some((from, was))) = (n, k, kept) {
                    let room = self.room_for(from, was, half);
                    if room >= 3.0 {
                        best = Some((room, was));
                        break 'points;
                    }
                }
            }
        }
        let Some((room, centre)) = best else { return false };
        if optional && room < 3.0 {
            return false;
        }
        self.boxed(centre - half, centre + half);
        match from {
            Some(from) => f.text_placed(name, number, from, centre - from, text, size, color),
            None => f.text_px(centre, text, size, color, egui::Align2::CENTER_CENTER),
        }
        true
    }
}

/// Is `p` inside the convex polygon with these corners (in order, either way round)?
/// (Asked several times for every place a label tries: nothing is allocated.)
pub fn inside(p: Vec2, corners: &[Vec2]) -> bool {
    let n = corners.len();
    if n < 3 {
        return false;
    }
    // on the same side of every edge, turning left all the way round or right
    let (mut left, mut right) = (true, true);
    for (i, a) in corners.iter().enumerate() {
        let turn = (corners[(i + 1) % n] - *a).perp_dot(p - *a);
        left &= turn > 0.0;
        right &= turn < 0.0;
    }
    left || right
}

// ----- a matrix written on the screen -------------------------------------------------

/// The room between a bracket and the numbers of a matrix written with `hud_matrix`.
const BRACKET: f32 = 7.0;

/// The size of one number's cell in such a matrix: wide enough for `widest` characters
/// of text size `size`.
pub fn matrix_cell(widest: usize, size: f32) -> Vec2 {
    Vec2::new((widest as f32 * size * 0.56 + 8.0).max(2.4 * size), size + 8.0)
}

/// The size of the whole matrix, brackets and all.
pub fn matrix_size(rows: usize, cols: usize, widest: usize, size: f32) -> Vec2 {
    let cell = matrix_cell(widest, size);
    Vec2::new(cols as f32 * cell.x + 2.0 * BRACKET, rows as f32 * cell.y)
}

/// The rectangle of the number in row `i` and column `j` (to ring it, or to tell
/// whether the pointer is on it).
pub fn matrix_entry(top_left: Vec2, widest: usize, size: f32, i: usize, j: usize) -> (Vec2, Vec2) {
    let cell = matrix_cell(widest, size);
    let min = top_left + Vec2::new(BRACKET + j as f32 * cell.x, i as f32 * cell.y);
    (min, min + cell)
}

/// A matrix written in screen space between square brackets, its top-left corner at
/// `top_left` (logical pixels): `rows` of numbers as text, each with its colour. Every
/// cell is wide enough for `widest` characters, so matrices side by side line up if
/// they are given the same `widest`. Returns the size taken (`matrix_size` says it
/// beforehand). Draw a `hud_panel` behind it first where it lies over the picture.
pub fn hud_matrix(f: &mut Frame, top_left: Vec2, rows: &[Vec<(String, Color)>], widest: usize, size: f32, bracket: Color) -> Vec2 {
    let cols = rows.iter().map(Vec::len).max().unwrap_or(0);
    let total = matrix_size(rows.len(), cols, widest, size);
    let (t, b) = (top_left.y + 1.0, top_left.y + total.y - 1.0);
    for (edge, inward) in [(top_left.x, 5.0), (top_left.x + total.x, -5.0)] {
        hud_line(f, Vec2::new(edge, t), Vec2::new(edge, b), 1.4, bracket);
        hud_line(f, Vec2::new(edge, t), Vec2::new(edge + inward, t), 1.4, bracket);
        hud_line(f, Vec2::new(edge, b), Vec2::new(edge + inward, b), 1.4, bracket);
    }
    for (i, row) in rows.iter().enumerate() {
        for (j, (text, color)) in row.iter().enumerate() {
            let (lo, hi) = matrix_entry(top_left, widest, size, i, j);
            f.text_px((lo + hi) * 0.5, text.as_str(), size, *color, egui::Align2::CENTER_CENTER);
        }
    }
    total
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Where a plotted curve is cut: at every pole of tan and of 1/x, at whatever zoom
    /// (the samples near a pole are no larger than the sampling makes them, so no
    /// test of size alone can find it), and at the falls of a sawtooth; nowhere along
    /// a smooth curve, however steep or wiggly.
    #[test]
    fn a_plotted_curve_is_cut_at_its_poles_and_nowhere_else() {
        let cuts = |f: &dyn Fn(f32) -> f32, lo: f32, hi: f32, n: usize| {
            let ys: Vec<f32> = (0..n).map(|i| f(lo + (hi - lo) * (i as f32 + 0.37) / n as f32)).collect();
            (0..n).filter(|&i| leaps(&ys, i)).count()
        };
        // the poles of tan in (-20, 20) are the odd multiples of pi/2 up to 11 pi/2: twelve
        for n in [300, 1200, 5000] {
            assert_eq!(cuts(&|x| x.tan(), -20.0, 20.0, n), 12, "tan with {n} samples");
            assert_eq!(cuts(&|x| 1.0 / x, -100.0, 100.0, n), 1, "1/x with {n} samples");
            assert_eq!(cuts(&|x| x - x.floor(), -3.0, 3.0, n), 5, "a sawtooth with {n} samples");
            for (name, f) in [
                ("a steep line", &(|x: f32| 1e4 * x) as &dyn Fn(f32) -> f32),
                ("a sine", &|x: f32| (7.0 * x).sin()),
                ("a narrow bell", &|x: f32| (-(40.0 * x).powi(2)).exp()),
                ("1/x^2", &|x: f32| 1.0 / (x * x)),
                ("a cubic", &|x: f32| x * x * x - 30.0 * x),
                ("|x|", &|x: f32| x.abs()),
                ("a staircase", &|x: f32| x.floor()),
            ] {
                assert_eq!(cuts(f, -20.0, 20.0, n), 0, "{name} with {n} samples");
            }
        }
    }

    /// However long the run, point k stands for the same stretch of it: pushing the
    /// step number itself, the points must lie evenly from start to finish.
    #[test]
    fn a_history_stays_evenly_spaced() {
        let mut h = History::new(100);
        for step in 0..12_345 {
            h.push(step as f32);
            assert!(h.points().len() < 100);
        }
        let (points, every) = (h.points(), h.every() as f32);
        assert!(points.len() >= 48, "{} points", points.len());
        for (k, &p) in points.iter().enumerate() {
            // the mean of steps k * every .. (k + 1) * every
            assert!((p - (k as f32 * every + (every - 1.0) * 0.5)).abs() < 1e-2 * every, "point {k}: {p}");
        }
        h.clear();
        assert!(h.is_empty() && h.every() == 1);
    }

    /// The label placer's geometry: a segment through a box is at no distance from
    /// it, and the inside of a shape is told from the outside.
    #[test]
    fn labels_find_what_they_would_cover() {
        let (lo, hi) = (Vec2::new(0.0, 0.0), Vec2::new(10.0, 4.0));
        assert_eq!(seg_box_dist(Vec2::new(-5.0, 2.0), Vec2::new(15.0, 2.0), lo, hi), 0.0);
        assert_eq!(seg_box_dist(Vec2::new(5.0, -3.0), Vec2::new(5.0, 9.0), lo, hi), 0.0);
        assert!((seg_box_dist(Vec2::new(-5.0, 7.0), Vec2::new(15.0, 7.0), lo, hi) - 3.0).abs() < 1e-6);
        assert!((seg_box_dist(Vec2::new(13.0, 0.0), Vec2::new(13.0, 4.0), lo, hi) - 3.0).abs() < 1e-6);
        let square = [Vec2::ZERO, Vec2::X, Vec2::ONE, Vec2::Y];
        assert!(inside(Vec2::splat(0.5), &square) && !inside(Vec2::new(1.5, 0.5), &square));
        assert!(inside(Vec2::splat(0.5), &[Vec2::ZERO, Vec2::Y, Vec2::ONE, Vec2::X]));
        // (on an edge, not a shape, or not a number: not inside)
        assert!(!inside(Vec2::new(0.5, 0.0), &square) && !inside(Vec2::ZERO, &square[..2]) && !inside(Vec2::NAN, &square));
    }

    /// What is kept from frame to frame is found again in the next frame and the one
    /// after, for as long as it is asked for; a frame without an ask lets it go; a text
    /// is never taken for another.
    #[test]
    fn what_is_kept_lasts_while_it_is_asked_for() {
        let mut kept: Kept<u32> = Kept::default();
        next_frame();
        kept.put(1, "one", 10);
        kept.put(2, "two", 20);
        assert_eq!((kept.find(1, "one"), kept.find(1, "uno"), kept.find(3, "three")), (Some(10), None, None));
        for _ in 0..5 {
            next_frame();
            assert_eq!(kept.find(1, "one"), Some(10), "asked for in every frame");
        }
        // "two" was not asked for in the frames since: gone
        assert_eq!(kept.find(2, "two"), None);
        next_frame();
        next_frame();
        assert_eq!(kept.find(1, "one"), None, "a frame without it");
        // the widths of texts with formulas are kept the same way, and are what they were
        let text = r"$\sqrt{x^2 + y^2}$ and words";
        let first = text_width(text, 13.0);
        assert!(WIDTHS.with(|w| w.borrow_mut().find(key_of(&(text, 13.0f32.to_bits())), text)) == Some(first));
        assert_eq!(text_width(text, 13.0), crate::typeset::estimate_width(text, 13.0, 0.47));
    }

    #[test]
    fn numbers_are_written_plainly() {
        assert_eq!(num(-0.004, 2), "0.00");
        assert_eq!(num(-1.5, 1), "−1.5");
        assert_eq!(num(2.0, 0), "2");
        assert_eq!(num(f64::NAN, 2), "?");
        assert_eq!(Tag::from("tip"), Tag("tip", 0));
        assert_eq!(Tag::from(("tip", 2usize)), Tag("tip", 2));
        // a matrix's cells tile it between the brackets
        let size = matrix_size(2, 3, 5, 14.0);
        let (lo, _) = matrix_entry(Vec2::ZERO, 5, 14.0, 0, 0);
        let (_, hi) = matrix_entry(Vec2::ZERO, 5, 14.0, 1, 2);
        assert!((lo.x - (size.x - hi.x)).abs() < 1e-4 && lo.y == 0.0 && (hi.y - size.y).abs() < 1e-4);
    }
}

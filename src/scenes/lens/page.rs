//! Pages of charts: the layout scaffolding the scenes of this topic share.
//!
//! Two of the scenes (Lyapunov exponents, what averaging keeps) are pages of charts.
//! A page is laid out in "design pixels" (x to the right, y downwards, the size of the
//! view) and drawn in world coordinates at one unit to the design pixel, so that the
//! usual panning and zooming work on it. When the view changes shape, the page is
//! laid out afresh and the camera told. The third scene (the belief filter) draws in
//! screen space instead, and shares only [`Rect`].

use crate::scenes::prelude::*;

/// A point of the page (design pixels, y downwards) as a point of the world.
pub fn world(p: Vec2) -> Vec2 {
    Vec2::new(p.x, -p.y)
}

/// The size the page is laid out for, and whether the camera has yet to hear of it.
pub struct Page {
    pub size: Vec2,
    /// How much of the top of the page the application's own buttons cover.
    top: f32,
    fresh: bool,
}

impl Page {
    pub fn new() -> Self {
        // (the size of the picture in a window of 1500 by 940 with both panels open)
        Page { size: Vec2::new(978.0, 940.0), top: 0.0, fresh: false }
    }

    /// The view that shows the whole page, one design pixel to the pixel.
    pub fn view(&self) -> View {
        View::Fit { center: world(self.size * 0.5), half: self.size * 0.5 }
    }

    /// Follow the size of the view; call at the start of every frame.
    pub fn fit(&mut self, f: &Frame) {
        let size = f.view_size();
        self.top = f.top_inset;
        if size.min_element() > 60.0 && (size - self.size).abs().max_element() > 1.0 {
            self.size = size;
            self.fresh = true;
        }
    }

    pub fn take_view(&mut self) -> Option<View> {
        std::mem::take(&mut self.fresh).then(|| self.view())
    }

    /// The part of the page that charts may use: clear of the buttons that lie over
    /// the top of the picture when the panels are not beside it.
    pub fn usable(&self) -> Rect {
        Rect::new(10.0, self.top.max(10.0), self.size.x - 10.0, self.size.y - 10.0)
    }
}

/// A rectangle in pixels of the picture, y downwards: design pixels of a [`Page`],
/// or logical pixels of the view for a scene drawn in screen space.
#[derive(Clone, Copy, Debug)]
pub struct Rect {
    pub min: Vec2,
    pub max: Vec2,
}

impl Rect {
    /// From corner to corner (never less than a pixel either way).
    pub fn new(x0: f32, y0: f32, x1: f32, y1: f32) -> Self {
        Rect { min: Vec2::new(x0, y0), max: Vec2::new(x1.max(x0 + 1.0), y1.max(y0 + 1.0)) }
    }

    /// From its top-left corner and its size.
    pub fn sized(x: f32, y: f32, w: f32, h: f32) -> Self {
        Rect { min: Vec2::new(x, y), max: Vec2::new(x + w, y + h) }
    }

    pub fn w(&self) -> f32 {
        self.max.x - self.min.x
    }

    pub fn h(&self) -> f32 {
        self.max.y - self.min.y
    }

    pub fn center(&self) -> Vec2 {
        (self.min + self.max) * 0.5
    }

    /// The same rectangle with `d` pixels taken off every side.
    pub fn shrink(&self, d: f32) -> Rect {
        Rect::new(self.min.x + d, self.min.y + d, self.max.x - d, self.max.y - d)
    }

    /// The dark panel behind a chart of a page.
    pub fn back(&self, f: &mut Frame) {
        inset(f, world(Vec2::new(self.min.x, self.max.y)), world(Vec2::new(self.max.x, self.min.y)));
    }

    /// A filled rectangle of a page (a bar, a swatch).
    pub fn fill(&self, f: &mut Frame, color: Color) {
        f.dl.tris_alpha.rect(world(Vec2::new(self.min.x, self.max.y)), world(Vec2::new(self.max.x, self.min.y)), 0.0, color);
    }
}

// Text on a page. (The toolkit's `wrapped` breaks a text to the room there is on
// screen at the present zoom and steps down in screen pixels. A page instead needs to
// know, before anything is drawn, how many design pixels a text will take, so that the
// chart under it can be given the rest; so it breaks lines for the design size, with
// the toolkit's estimate of a text's width.)

/// `text_width`, remembered for texts with formulas in them: measuring a formula means
/// laying it out, too dear to do for every word of every heading in every frame, and
/// the words are mostly the same from one frame to the next.
pub fn width(text: &str, size: f32) -> f32 {
    thread_local! {
        // (by size, then by text, so that a text is looked up without copying it)
        static KNOWN: std::cell::RefCell<std::collections::HashMap<u32, std::collections::HashMap<String, f32>>> = std::cell::RefCell::new(std::collections::HashMap::new());
    }
    if !crate::typeset::has_math(text) {
        return text_width(text, size);
    }
    KNOWN.with(|known| {
        let mut known = known.borrow_mut();
        let known = known.entry(size.to_bits()).or_default();
        if let Some(w) = known.get(text) {
            return *w;
        }
        // (a text with a number in it is a new text whenever the number changes)
        if known.len() >= 256 {
            known.clear();
        }
        let w = text_width(text, size);
        known.insert(text.to_string(), w);
        w
    })
}

/// The words of a text with formulas in it, each formula whole: lines are broken
/// between them, never inside a formula.
fn words(text: &str) -> Vec<&str> {
    let (mut words, mut start, mut inside) = (Vec::new(), 0, false);
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

/// Text broken into lines of at most `room` pixels, by the toolkit's estimate; a text
/// with formulas in it as it was broken the last time it was drawn in that room.
fn wrap(text: &str, room: f32, size: f32) -> Vec<String> {
    // (by room and size, then by text)
    type Broken = std::collections::HashMap<(u32, u32), std::collections::HashMap<String, Vec<String>>>;
    thread_local! {
        static BROKEN: std::cell::RefCell<Broken> = std::cell::RefCell::new(Broken::new());
    }
    if !crate::typeset::has_math(text) {
        return break_lines(text, room, size);
    }
    BROKEN.with(|broken| {
        let mut broken = broken.borrow_mut();
        let broken = broken.entry((room.to_bits(), size.to_bits())).or_default();
        if let Some(lines) = broken.get(text) {
            return lines.clone();
        }
        if broken.len() >= 64 {
            broken.clear();
        }
        let lines = break_lines(text, room, size);
        broken.insert(text.to_string(), lines.clone());
        lines
    })
}

/// Text broken into lines of at most `room` pixels, by the toolkit's estimate.
fn break_lines(text: &str, room: f32, size: f32) -> Vec<String> {
    let mut lines = vec![String::new()];
    if !crate::typeset::has_math(text) {
        for word in text.split(' ') {
            let line = lines.last_mut().unwrap();
            if !line.is_empty() && text_width(line, size) + text_width(word, size) + size * 0.3 > room {
                lines.push(word.to_string());
            } else {
                if !line.is_empty() {
                    line.push(' ');
                }
                line.push_str(word);
            }
        }
        return lines;
    }
    // The same with formulas in the text, each word measured once: the last line's
    // width so far is kept, words and the spaces between them, as the estimate of the
    // whole line would come out.
    let space = text_width(" ", size);
    let mut used = 0.0;
    for word in words(text) {
        let w = width(word, size);
        let line = lines.last_mut().unwrap();
        if !line.is_empty() && used + w + size * 0.3 > room {
            lines.push(word.to_string());
            used = w;
        } else {
            if !line.is_empty() {
                line.push(' ');
                used += space;
            }
            line.push_str(word);
            used += w;
        }
    }
    lines
}

fn write_lines(f: &mut Frame, at: Vec2, lines: &[String], size: f32, color: Color) -> f32 {
    let step = size + 3.0;
    for (i, line) in lines.iter().enumerate() {
        f.text(world(at + Vec2::new(0.0, i as f32 * step)), line.as_str(), size, color, Align2::LEFT_TOP);
    }
    lines.len() as f32 * step
}

/// Wrapped text with its top-left corner at `at`. Returns the height it took.
pub fn write(f: &mut Frame, at: Vec2, width: f32, text: &str, size: f32, color: Color) -> f32 {
    write_lines(f, at, &wrap(text, width, size), size, color)
}

/// Wrapped text standing at the foot of `inner`, its last line ending `lift` pixels
/// above the bottom edge. Returns the height it took, so that what is drawn above it
/// can keep clear.
pub fn write_foot(f: &mut Frame, inner: Rect, lift: f32, text: &str, size: f32, color: Color) -> f32 {
    let lines = wrap(text, inner.w(), size);
    let height = lines.len() as f32 * (size + 3.0);
    write_lines(f, Vec2::new(inner.min.x, inner.max.y - lift - height), &lines, size, color)
}

/// A colour key that flows along rows of at most `room` pixels: a short line of
/// each colour and what it stands for. Returns the height it took. (The toolkit's
/// `hud_key` is a column of swatches in screen space; this one belongs to a page.)
pub fn key(f: &mut Frame, at: Vec2, room: f32, entries: &[(Color, &str)], size: f32) -> f32 {
    let row = size + 5.0;
    let (mut x, mut y) = (0.0, 0.0);
    for (color, text) in entries {
        let need = 22.0 + width(text, size) + 14.0;
        if x > 0.0 && x + need > room {
            x = 0.0;
            y += row;
        }
        let p = at + Vec2::new(x, y);
        f.line(world(p + Vec2::new(0.0, size * 0.6)), world(p + Vec2::new(16.0, size * 0.6)), 3.0, *color);
        f.text(world(p + Vec2::new(21.0, 0.0)), *text, size, pal::TEXT.a(0.92), Align2::LEFT_TOP);
        x += need;
    }
    y + row
}

/// The head of a panel of a page: its dark backing is drawn, and then its title, its
/// colour key and any caption are written one under the other. `inner` is the panel
/// inside its margin; `y` is where the next thing goes.
pub struct Panel {
    pub inner: Rect,
    pub y: f32,
}

impl Panel {
    pub fn open(f: &mut Frame, r: Rect, margin: f32) -> Panel {
        r.back(f);
        let inner = r.shrink(margin);
        Panel { inner, y: inner.min.y }
    }

    /// A wrapped line of words, then `gap` pixels.
    pub fn text(&mut self, f: &mut Frame, text: &str, size: f32, color: Color, gap: f32) {
        self.y += write(f, Vec2::new(self.inner.min.x, self.y), self.inner.w(), text, size, color) + gap;
    }

    /// A colour key, then `gap` pixels.
    pub fn key(&mut self, f: &mut Frame, entries: &[(Color, &str)], gap: f32) {
        self.y += key(f, Vec2::new(self.inner.min.x, self.y), self.inner.w(), entries, 11.0) + gap;
    }
}

/// A chart: the rectangle of the page it fills and the range of numbers it shows.
pub struct Plot {
    pub area: Rect,
    pub x: [f32; 2],
    pub y: [f32; 2],
}

impl Plot {
    /// Where the numbers (x, y) are drawn.
    pub fn at(&self, x: f32, y: f32) -> Vec2 {
        let tx = (x - self.x[0]) / (self.x[1] - self.x[0]);
        let ty = (y - self.y[0]) / (self.y[1] - self.y[0]);
        world(Vec2::new(self.area.min.x + tx * self.area.w(), self.area.max.y - ty * self.area.h()))
    }

    /// Horizontal rules at round values, numbered down the left side, the one at zero
    /// brighter than the rest.
    pub fn rules(&self, f: &mut Frame) {
        let span = self.y[1] - self.y[0];
        // (a range that is no range, or not a number, has no round values in it)
        if !(span.is_finite() && span > 0.0) {
            return;
        }
        let step = nice_step(span / (self.area.h() / 34.0).clamp(2.0, 7.0));
        // counted, not stepped to: a step too small to move the value would never arrive
        let first = (self.y[0] / step).ceil();
        let count = ((span / step).floor().max(0.0) as usize).min(40);
        for v in (0..=count).map(|i| (first + i as f32) * step).filter(|v| *v <= self.y[1] + step * 1e-3) {
            let zero = v.abs() < step * 1e-3;
            let color = if zero { pal::DIM.a(0.9) } else { pal::FAINT.a(0.45) };
            f.line(self.at(self.x[0], v), self.at(self.x[1], v), 1.0, color);
            f.text(self.at(self.x[0], v) - Vec2::new(5.0, 0.0), fmt_tick(if zero { 0.0 } else { v }, step), 10.5, pal::DIM, Align2::RIGHT_CENTER);
        }
    }

    /// A curve through the points given, cut off where it leaves the vertical range.
    pub fn curve(&self, f: &mut Frame, points: impl IntoIterator<Item = (f32, f32)>, width: f32, color: Color) {
        let (lo, hi) = (self.y[0], self.y[1]);
        let inside = |y: f32| y >= lo && y <= hi;
        // the point of the segment a-b at height y
        let cut = |a: (f32, f32), b: (f32, f32), y: f32| (a.0 + (b.0 - a.0) * (y - a.1) / (b.1 - a.1), y);
        let mut pieces: Vec<Vec<Vec3>> = vec![Vec::new()];
        let mut prev: Option<(f32, f32)> = None;
        for p in points {
            if !p.1.is_finite() {
                pieces.push(Vec::new());
                prev = None;
                continue;
            }
            match prev {
                None => {
                    if inside(p.1) {
                        pieces.last_mut().unwrap().push(self.at(p.0, p.1).extend(0.0));
                    }
                }
                Some(q) => {
                    let edge = |y: f32| if y > hi { hi } else { lo };
                    match (inside(q.1), inside(p.1)) {
                        (true, true) => pieces.last_mut().unwrap().push(self.at(p.0, p.1).extend(0.0)),
                        (true, false) => {
                            let c = cut(q, p, edge(p.1));
                            pieces.last_mut().unwrap().push(self.at(c.0, c.1).extend(0.0));
                            pieces.push(Vec::new());
                        }
                        (false, true) => {
                            let c = cut(q, p, edge(q.1));
                            pieces.push(vec![self.at(c.0, c.1).extend(0.0), self.at(p.0, p.1).extend(0.0)]);
                        }
                        (false, false) => {
                            // right across the chart from below to above, or back
                            if (q.1 < lo) != (p.1 < lo) {
                                let (c, d) = (cut(q, p, edge(q.1)), cut(q, p, edge(p.1)));
                                pieces.push(vec![self.at(c.0, c.1).extend(0.0), self.at(d.0, d.1).extend(0.0)]);
                                pieces.push(Vec::new());
                            }
                        }
                    }
                }
            }
            prev = Some(p);
        }
        for piece in pieces {
            if piece.len() >= 2 {
                f.dl.lines.polyline(piece, width, color);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A text with formulas in it is broken between words, never inside a formula,
    /// and each line of more than one word keeps to the room (as a line without
    /// formulas does: to within the part of a space that the rule allows over).
    #[test]
    fn lines_break_between_formulas_never_inside_one() {
        let text = r"the answers $K(c, \tau) v$, each first turned back by $\omega\tau$ as the tuned lens does (bold: $\cos$ part, faint: $\sin$ part)";
        for room in [40.0, 90.0, 150.0, 260.0, 900.0] {
            let lines = wrap(text, room, 10.5);
            assert_eq!(lines.join(" "), text, "room {room}");
            for line in &lines {
                assert!(crate::typeset::inspect(line).error.is_none(), "room {room}: {line:?}");
                assert!(words(line).len() == 1 || width(line, 10.5) <= room + 0.2 * 10.5, "room {room}: {line:?}");
            }
        }
        assert_eq!(wrap(text, 900.0, 10.5).len(), 1);
    }
}

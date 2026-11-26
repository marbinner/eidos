//! Low-rank approximation: a picture as a sum of pieces, each a column pattern times a
//! row pattern, the biggest first; the first k of them are the best picture of rank k.

use super::*;
use crate::clock::Budget;

pub const DEF: SceneDef = SceneDef {
    id: "lowrank",
    title: "Low-rank approximation",
    group: "Linear algebra: data and dimensions",
    blurb: "A picture as a sum of simple pieces, the biggest first: how few does it take?",
    make: |_| Box::new(LowRank::new()),
};

/// The pictures are this many pixels on a side.
const N: usize = 64;
/// Pieces added per second while the pieces are being added one by one (at least: a
/// long run goes faster, so that all 64 take twelve seconds).
const RATE: f32 = 3.0;
/// How long a change of k takes to glide: the time to about two thirds of the way (s).
const GLIDE: f32 = 0.1;

/// The colours: one thing each.
const KEPT: Color = pal::GREEN;
const DROPPED: Color = pal::PURPLE;
const K_MARK: Color = pal::YELLOW;
/// Signed values (the pieces, their patterns, what is left): blue below zero, red above.
const SIGNED: Cmap = Cmap::IceFire;
const GREY: Cmap = Cmap::Grays;
/// What is left is drawn at full colour at ± this (a quarter of the way from black to white).
const LEFT_FULL: f64 = 0.25;

/// Everything an experiment sets.
#[derive(Clone, Copy, Debug)]
struct Setup {
    pic: Pic,
    /// The noise added (the standard deviation of each pixel's), and its seed.
    noise: f32,
    seed: u64,
    k: usize,
}

const fn setup(pic: Pic, noise: f32, k: usize) -> Setup {
    Setup { pic, noise, seed: 1, k }
}

/// The scene opens on the second of these. Every claim here is checked by a test of
/// the same setting (and of other noise where chance is involved).
const TRY: [(&str, &str, Setup); 5] = [
    (
        "a plaid is one piece",
        r"Every row of this plaid is the same row pattern, brighter or darker: the picture is one column pattern times one row pattern. So it is a single piece. $\sigma_1 = 25.6$ and every other $\sigma$ is $0$; the first piece alone is the picture, and what is left is rounding, $10^{-13}$ or less. 129 numbers hold all 4096.",
        setup(Pic::Plaid, 0.0, 1),
    ),
    (
        "a few pieces carry a picture",
        "Soft blobs. The sizes fall fast, 15.6, 8.8, 4.7, 2.2, 1.4, 0.5, …: five pieces leave 3 % of the picture, in 645 numbers instead of 4096. Watch each piece put back what the ones before it missed.",
        setup(Pic::Blobs, 0.0, 5),
    ),
    (
        "a line: the hardest picture",
        r"One pixel on each row and column: the identity matrix. Every $\sigma$ is $1$, so no piece is bigger than any other, and $k$ pieces leave $\sqrt{64 - k}$ of the line's size 8: eight leave 94 %. Here each piece is one pixel of the line, 129 numbers for one pixel. (With equal $\sigma$ the pieces could be chosen in many other ways, all exactly as good.)",
        setup(Pic::Line, 0.0, 8),
    ),
    (
        "noise does not compress",
        "Every pixel a random grey. The first piece is about the average grey; after it the sizes fall slowly and evenly (4.6, 4.3, 4.3, 4.2, … and still 3.0 at the sixteenth), so each piece takes away only a sliver. Sixteen pieces, half as many numbers as the picture, still leave 31 % of it.",
        setup(Pic::Static, 0.0, 16),
    ),
    (
        "structure plus noise: cut it off",
        r"The blobs with noise added to every pixel. Three bars stand above a floor of small ones, which is the noise. Keep the three and drop the floor: $A_3$ is nearer the clean blobs than $A$ is, by less than two fifths as much, and no other $k$ comes as near. What is left, $A - A_3$, is mostly the noise itself.",
        setup(Pic::Blobs, 0.2, 3),
    ),
];

pub struct LowRank {
    /// The picture chosen (None: drawn on), the noise on it and its seed.
    pic: Option<Pic>,
    noise: f32,
    seed: u64,
    /// The picture without noise (what is drawn on), and with it: A.
    clean: Vec<f64>,
    a: Vec<f64>,
    /// Counts the changes of A.
    version: u64,
    /// The decomposition last finished, and a count of them.
    dec: Dec,
    dec_id: u64,
    /// The decomposition being worked out; whether it is of a new picture (then the
    /// pieces are added from none once it is done); whether A has changed again since
    /// it started.
    job: Option<Jacobi>,
    fresh: bool,
    stale: bool,
    /// Pieces kept, and as drawn (gliding to k, or rising while they are added).
    k: usize,
    t: f32,
    /// Adding the pieces one by one, up to this many.
    build: Option<f32>,
    /// The pen: 0 white, 1 black; where the stroke being drawn is (in pixels of the picture).
    pen: usize,
    stroke: Option<Vec2>,
    /// The sizes σ as the chart draws them (gliding to a new picture's), and its top.
    sig: Vec<f32>,
    top: f32,
    /// All the pictures in one, repainted when what they show changes.
    image: Option<Image>,
    painted: Option<(u64, u64, u32)>,
    /// The first ⌊t⌋ pieces added up, for the decomposition with this count; and the
    /// first k, for the numbers.
    cache: (usize, u64, Vec<f64>),
    whole: (usize, u64, Vec<f64>),
    /// The first t pieces as last painted (for the pixel pointed at).
    shown: Vec<f64>,
}

impl LowRank {
    fn new() -> LowRank {
        let mut s = LowRank {
            pic: None,
            noise: 0.0,
            seed: 1,
            clean: vec![0.0; N * N],
            a: vec![0.0; N * N],
            version: 0,
            dec: Dec::of(&[0.0; N * N], N, N),
            dec_id: 0,
            job: None,
            fresh: false,
            stale: false,
            k: 0,
            t: 0.0,
            build: None,
            pen: 0,
            stroke: None,
            sig: vec![0.0; N],
            top: 1.0,
            image: None,
            painted: None,
            cache: (0, u64::MAX, Vec::new()),
            whole: (0, u64::MAX, Vec::new()),
            shown: Vec::new(),
        };
        s.apply(TRY[1].2);
        s
    }

    /// Set an experiment up.
    fn apply(&mut self, s: Setup) {
        (self.noise, self.seed, self.k) = (s.noise, s.seed, s.k.min(N));
        self.choose(s.pic);
    }

    /// Show one of the pictures, with the noise as it is set.
    fn choose(&mut self, pic: Pic) {
        self.pic = Some(pic);
        self.clean = picture(pic);
        self.new_picture();
    }

    /// A different picture: decomposed afresh, from the axes (so that it comes out the
    /// same whatever was shown before), and its pieces added from none up to k.
    fn new_picture(&mut self) {
        self.mix();
        self.job = Some(Jacobi::new(&self.a, N, N, None));
        (self.fresh, self.stale, self.stroke) = (true, false, None);
        self.t = 0.0;
        self.build = Some(self.k as f32);
    }

    /// The same picture changed a little (drawn on, more or less noise): decomposed
    /// again from the directions the last decomposition ended with (which usually
    /// saves a few sweeps), and shown with the pieces kept as they were.
    fn touched(&mut self) {
        self.mix();
        if self.job.is_some() {
            self.stale = true;
        } else {
            self.job = Some(Jacobi::new(&self.a, N, N, Some(&self.dec.directions)));
        }
    }

    /// A = the picture plus the noise.
    fn mix(&mut self) {
        self.a = if self.noise > 0.0 { self.clean.iter().zip(noise(self.noise as f64, self.seed)).map(|(x, e)| x + e).collect() } else { self.clean.clone() };
        self.version += 1;
    }

    /// Carry the decomposition on within `budget`.
    fn work(&mut self, budget: &Budget) {
        let Some(job) = &mut self.job else { return };
        if !job.run(budget) {
            return;
        }
        // (a new picture's patterns owe nothing to the last one's signs)
        let dec = job.finish(if self.fresh { None } else { Some(&self.dec) });
        self.dec = dec;
        self.dec_id += 1;
        if self.dec_id == 1 {
            // (the first picture's bars have nothing to glide from)
            self.sig = self.dec.s.iter().map(|x| *x as f32).collect();
            self.top = self.dec.top();
        }
        (self.job, self.fresh) = (None, false);
        if std::mem::take(&mut self.stale) {
            self.job = Some(Jacobi::new(&self.a, N, N, Some(&self.dec.directions)));
        }
    }

    /// Work the decomposition out to the end at once (the tests).
    #[cfg(test)]
    fn settle(&mut self) {
        while self.job.is_some() {
            self.work(&Budget::new(f64::INFINITY));
        }
    }

    /// Pieces added per second while adding them up to `end`.
    fn rate(end: f32) -> f32 {
        RATE.max(end / 12.0)
    }

    /// Move the picture on: the pieces being added (by scene time), k gliding, the
    /// chart's bars gliding to a new picture's sizes (by the clock: they answer a click).
    fn advance(&mut self, dt: f32, real_dt: f32) {
        let ease = 1.0 - (-real_dt.max(0.0) / GLIDE).exp();
        match self.build {
            // (a new picture's pieces wait for its decomposition)
            Some(_) if self.fresh => {}
            Some(end) => {
                self.t = (self.t + Self::rate(end) * dt.max(0.0)).min(end);
                if self.t >= end {
                    self.build = None;
                }
            }
            None => {
                let goal = self.k as f32;
                self.t += (goal - self.t) * ease;
                // (exactly there in the end: the picture then stops changing)
                if (goal - self.t).abs() < 2e-3 || !self.t.is_finite() {
                    self.t = goal;
                }
            }
        }
        if !self.fresh {
            let top = self.dec.top();
            let close = 2e-4 * top.max(self.top);
            for (shown, s) in self.sig.iter_mut().zip(&self.dec.s) {
                *shown += (*s as f32 - *shown) * ease;
                if (*shown - *s as f32).abs() < close || !shown.is_finite() {
                    *shown = *s as f32;
                }
            }
            self.top += (top - self.top) * ease;
            if (self.top - top).abs() < close || !self.top.is_finite() {
                self.top = top;
            }
        }
    }

    fn bars_still(&self) -> bool {
        self.fresh || (self.top == self.dec.top() && self.sig.iter().zip(&self.dec.s).all(|(a, b)| *a == *b as f32))
    }

    /// Add the pieces one by one, from none, up to the last that is not zero.
    fn play(&mut self) {
        let end = self.dec.nonzero().max(1).min(N);
        (self.t, self.k, self.build) = (0.0, end, Some(end as f32));
    }

    /// How far the first `k` pieces are from A, and from the picture without its
    /// noise. (Kept from one call to the next while k and the decomposition stay the
    /// same: the numbers on show ask for them every frame.)
    fn misses(&mut self, k: usize) -> (f64, f64) {
        if self.whole.0 != k || self.whole.1 != self.dec_id || self.whole.2.len() != N * N {
            self.whole = (k, self.dec_id, self.dec.approx(k));
        }
        (distance(&self.dec.a, &self.whole.2), distance(&self.clean, &self.whole.2))
    }

    /// The first t pieces, a share of the next one for a t between whole numbers.
    fn approx_shown(&mut self) -> Vec<f64> {
        let t = self.t.clamp(0.0, N as f32);
        let whole = (t.floor() as usize).min(N);
        if self.cache.0 != whole || self.cache.1 != self.dec_id || self.cache.2.len() != N * N {
            self.cache = (whole, self.dec_id, self.dec.approx(whole));
        }
        let mut out = self.cache.2.clone();
        let share = (t - whole as f32) as f64;
        if share > 0.0 {
            self.dec.add_piece(whole, share, &mut out);
        }
        out
    }

    /// The piece shown alone (counting from 0): the one being added, or the last one
    /// kept (the first, before any is).
    fn piece_shown(&self) -> usize {
        ((self.t - 1e-4).ceil().max(1.0) as usize - 1).min(N - 1)
    }

    /// How many pieces the numbers on show speak of: k, as drawn.
    fn k_shown(&self) -> usize {
        ((self.t - 1e-4).ceil().max(0.0) as usize).min(N)
    }

    /// Paint the pictures into the image: A, the first pieces, what is left, the piece
    /// alone, and its two patterns as strips.
    fn paint(&mut self, gfx: &Gfx) {
        let key = (self.version, self.dec_id, self.t.to_bits());
        if self.painted == Some(key) && self.image.is_some() {
            return;
        }
        let approx = self.approx_shown();
        self.shown = approx.clone();
        let (w, h) = (4 * N + 1, N + 1);
        let mut image = self.image.take().unwrap_or_else(|| Image::new(gfx, w as u32, h as u32));
        image.ensure_size(gfx, w as u32, h as u32);
        if image.size() == [w as u32, h as u32] {
            let (grey, signed) = (GREY.table(), SIGNED.table());
            let shade = |v: f64| grey[table_index(v as f32)];
            let sign = |v: f64| signed[table_index(0.5 + 0.47 * v.clamp(-1.0, 1.0) as f32)];
            let p = self.piece_shown();
            // (the patterns of a piece down at rounding are rounding too: not drawn)
            let live = p < self.dec.nonzero();
            let (u, v) = (self.dec.u_col(p), self.dec.v_col(p));
            let most = |x: &[f64]| x.iter().fold(0.0f64, |b, v| b.max(v.abs())).max(1e-300);
            let (mu, mv) = (most(u), most(v));
            let pixels = image.pixels_mut();
            for i in 0..N {
                // (pixel rows count from the bottom, matrix rows from the top)
                let row = &mut pixels[(N - 1 - i) * w..(N - i) * w];
                for j in 0..N {
                    let at = i * N + j;
                    row[j] = shade(self.a[at]);
                    row[N + j] = shade(approx[at]);
                    row[2 * N + j] = sign((self.dec.a[at] - approx[at]) / LEFT_FULL);
                    row[3 * N + j] = if live { sign(u[i] * v[j] / (mu * mv)) } else { sign(0.0) };
                }
                row[4 * N] = if live { sign(u[i] / mu) } else { sign(0.0) };
            }
            let top = &mut pixels[N * w..(N + 1) * w];
            for (j, px) in top.iter_mut().enumerate() {
                *px = if j < N && live { sign(v[j] / mv) } else if j < N { sign(0.0) } else { Color::CLEAR };
            }
            self.painted = Some(key);
        }
        self.image = Some(image);
    }

    /// Draw on the picture A with the pen: a drag that starts on it paints along the
    /// way (white or black, a soft round brush), and A is decomposed again as it goes.
    fn pen_input(&mut self, f: &mut Frame, lay: &Layout) {
        let (lo, hi) = lay.cell(0);
        let side = lay.side;
        let at = f.mouse();
        let over = at.is_some_and(|p| p.x >= lo.x && p.x <= hi.x && p.y >= lo.y && p.y <= hi.y);
        if over || self.stroke.is_some() {
            f.handles.capture();
        }
        if !f.ptr.down || f.handles.dragging() {
            self.stroke = None;
            if over {
                // where the pen would draw
                if let Some(p) = at {
                    f.circle(p, Vec3::Z, 2.0 * side / N as f32, 1.2, pal::WHITE.a(0.6));
                }
            }
            return;
        }
        let Some(p) = at else { return };
        let here = Vec2::new((p.x - lo.x) / side * N as f32, (hi.y - p.y) / side * N as f32);
        let from = match self.stroke {
            Some(last) => last,
            None if f.ptr.pressed && over => here,
            None => return,
        };
        self.stroke = Some(here);
        if self.brush(from, here) {
            (self.pic, self.build) = (None, None);
            self.touched();
        }
    }

    /// Paint along the segment from `from` to `to` (in pixels of the picture). Returns
    /// whether any pixel changed.
    fn brush(&mut self, from: Vec2, to: Vec2) -> bool {
        const R: f32 = 1.9;
        let value = if self.pen == 0 { 1.0 } else { 0.0 };
        let (lo, hi) = (from.min(to) - Vec2::splat(R + 1.0), from.max(to) + Vec2::splat(R + 1.0));
        let range = |a: f32, b: f32| (a.max(0.0).floor() as usize)..(b.max(0.0).ceil() as usize).min(N);
        let mut changed = false;
        for i in range(lo.y, hi.y) {
            for j in range(lo.x, hi.x) {
                let d = seg_dist(Vec2::new(j as f32 + 0.5, i as f32 + 0.5), from, to);
                let weight = (R + 0.5 - d).clamp(0.0, 1.0) as f64;
                let px = &mut self.clean[i * N + j];
                let new = *px + (value - *px) * weight;
                if weight > 0.0 && (new - *px).abs() > 1e-6 {
                    *px = new;
                    changed = true;
                }
            }
        }
        changed
    }

    /// Where the pointer is, for the numbers of what it points at: nowhere while it is
    /// busy drawing or dragging the handle (the marks would jump from pixel to pixel
    /// and from bar to bar under it).
    fn pointer(&self, f: &Frame) -> Option<Vec2> {
        if self.stroke.is_some() || f.handles.dragging() {
            None
        } else {
            f.mouse()
        }
    }

    /// The size left out by the first k pieces, as a formula (to go between dollars):
    /// `\sqrt{\sigma_{6}^2 + \cdots + \sigma_{64}^2}`.
    fn left_formula(&self, k: usize) -> String {
        let last = self.dec.s.len();
        match last.saturating_sub(k) {
            0 => r"\text{nothing}".to_string(),
            1 => format!(r"\sqrt{{\sigma_{{{last}}}^2}}"),
            2 => format!(r"\sqrt{{\sigma_{{{}}}^2 + \sigma_{{{last}}}^2}}", k + 1),
            _ => format!(r"\sqrt{{\sigma_{{{}}}^2 + \cdots + \sigma_{{{last}}}^2}}", k + 1),
        }
    }
}

/// A size: to a tenth from 10 up, to two decimals from 0.1, to two figures below
/// that, and as a power of ten below a thousandth (where it is usually rounding). As a
/// formula, to go between dollars.
fn size_text(v: f64) -> String {
    if !v.is_finite() || v <= 0.0 {
        "0".to_string()
    } else if v >= 10.0 {
        format!("{v:.1}")
    } else if v >= 0.1 {
        format!("{v:.2}")
    } else if v >= 1e-3 {
        let decimals = (1.0 - v.log10().floor()) as usize;
        format!("{v:.decimals$}")
    } else {
        let e = v.log10().floor();
        let mantissa = (v / 10f64.powf(e)).round() as i64;
        let (mantissa, e) = if mantissa >= 10 { (1, e + 1.0) } else { (mantissa, e) };
        format!(r"{mantissa} \times 10^{{-{}}}", (-e) as usize)
    }
}

/// The first of `options` that fits in `room` units at this zoom (the last if none
/// does), measured as drawn: a formula is not as wide as its markup.
fn fits<'a>(f: &Frame, options: &[&'a str], room: f32, ppu: f32, size: f32) -> &'a str {
    options.iter().copied().find(|t| f.text_size(t, size).x < room * ppu).unwrap_or_else(|| options.last().copied().unwrap_or_default())
}

/// Where everything goes, in the units of the stage (five high, as wide as the canvas
/// makes it, centred on 0).
struct Layout {
    wide: bool,
    /// Text size, a line of it in units, and units per pixel.
    size: f32,
    line: f32,
    px: f32,
    /// The three pictures of the sum, A = A_k + (A − A_k): the side of each and their
    /// top-left corners; the room between them.
    side: f32,
    cells: [Vec2; 3],
    gap: f32,
    /// The piece alone: its top-left corner (the strips lie outside it, above and to
    /// the left), its side, and the strips' thickness.
    piece: Vec2,
    pside: f32,
    strip: f32,
    /// The chart: bottom-left corner and size.
    chart: Vec2,
    chart_size: Vec2,
}

impl Layout {
    fn of(w: f32, ppu: f32, top_inset: f32) -> Layout {
        let px = 1.0 / ppu.max(1e-3);
        let wide = w >= 3.3;
        let size = if wide { 12.5 } else { 11.0 };
        let line = (size + 6.0) * px;
        let top = 2.5 - (top_inset + if wide { 10.0 } else { 4.0 }) * px;
        let bottom = -2.5 + 6.0 * px;
        let strip = (if wide { 12.0 } else { 9.0 }) * px;
        // from the top: the sum's titles, its pictures, the lines under them, a gap;
        // the titles of the piece and the chart, the strip, the piece; under the
        // chart its numbers and key
        let under = if wide { 3.0 } else { 2.0 };
        let (margin, gap) = if wide { (24.0 * px, 40.0 * px) } else { (10.0 * px, 18.0 * px) };
        let side_across = ((w - 2.0 * margin - 2.0 * gap) / 3.0).max(0.05);
        let (side, pside, chart_h, spare);
        if wide {
            let fixed = (1.3 + under + 0.8 + 1.3 + 2.4) * line + strip + 3.0 * px;
            // the piece as large as the pictures of the sum, and the chart beside it
            side = ((top - bottom - fixed) * 0.5).min(side_across).max(0.05);
            pside = side;
            chart_h = pside + strip + 3.0 * px;
            spare = (top - bottom - fixed - 2.0 * side).max(0.0);
        } else {
            // the piece with words beside it, then the chart across the whole width
            let fixed = (1.3 + under + 0.6 + 1.3 + 0.6 + 1.3 + 2.4) * line + strip + 3.0 * px;
            side = side_across;
            let rest = (top - bottom - fixed - side).max(0.1);
            chart_h = (rest * 0.45).clamp(0.05, 200.0 * px);
            pside = (rest - chart_h).min(0.55 * w).max(0.05);
            spare = (rest - chart_h - pside).max(0.0);
        }
        // (what is left over is shared out above and below)
        let top = top - spare * 0.4;
        let left = -(3.0 * side + 2.0 * gap) * 0.5;
        let right = -left;
        let row = top - 1.3 * line;
        let cells = [0, 1, 2].map(|c| Vec2::new(left + c as f32 * (side + gap), row));
        let strip_top = row - side - (under + 0.8 + 1.3) * line + if wide { 0.0 } else { 0.2 * line };
        let piece_top = strip_top - strip - 3.0 * px;
        let piece = Vec2::new(left + 30.0 * px + strip + 3.0 * px, piece_top);
        let (chart, chart_size) = if wide {
            let chart_left = piece.x + pside + 70.0 * px;
            (Vec2::new(chart_left, strip_top - chart_h), Vec2::new((right - chart_left).max(0.2), chart_h))
        } else {
            let chart_top = piece_top - pside - (0.6 + 1.3) * line;
            let chart_left = left + 34.0 * px;
            (Vec2::new(chart_left, chart_top - chart_h), Vec2::new((right - chart_left).max(0.2), chart_h))
        };
        Layout { wide, size, line, px, side, cells, gap, piece, pside, strip, chart, chart_size }
    }

    /// The corners (bottom-left, top-right) of picture `c` of the sum.
    fn cell(&self, c: usize) -> (Vec2, Vec2) {
        let tl = self.cells[c.min(2)];
        (Vec2::new(tl.x, tl.y - self.side), Vec2::new(tl.x + self.side, tl.y))
    }
}

impl LowRank {
    /// The sum A = A_k + (A − A_k): three pictures, with what each is and what it costs.
    fn draw_sum(&mut self, f: &mut Frame, lay: &Layout, ppu: f32) {
        let k = self.k_shown();
        let (size, line, px) = (lay.size, lay.line, lay.px);
        let scale = size_of(&self.dec.a);
        // the pictures
        if let Some(image) = &mut self.image {
            for c in 0..3 {
                let (lo, hi) = lay.cell(c);
                f.image_part(image, [(c * N) as u32, 0, N as u32, N as u32], lo, hi, 0.0);
            }
        }
        if self.a.iter().all(|v| *v == 0.0) {
            let (lo, hi) = lay.cell(0);
            f.text((lo + hi) * 0.5, fits(f, &["blank: drag here to draw", "drag to draw", ""], lay.side, ppu, size), size, pal::DIM, Align2::CENTER_CENTER);
        }
        let frames = [pal::DIM.a(0.7), KEPT.a(0.8), DROPPED.a(0.8)];
        for (c, color) in frames.into_iter().enumerate() {
            let (lo, hi) = lay.cell(c);
            frame_rect(f, lo - Vec2::splat(px), hi + Vec2::splat(px), 1.2, color);
        }
        // the pixel pointed at, in all three: the sum holds pixel by pixel
        let pointed = self.pointer(f).and_then(|p| {
            (0..3).find_map(|c| {
                let (lo, hi) = lay.cell(c);
                pixel_at(p, lo, hi)
            })
        });
        if let (Some((i, j)), true) = (pointed, self.shown.len() == N * N) {
            for c in 0..3 {
                let (lo, hi) = lay.cell(c);
                ring_pixel(f, lo, hi, i, j, pal::WHITE);
            }
            let (a, kept) = (self.dec.a[i * N + j], self.shown[i * N + j]);
            let left = a - kept;
            let sign = if num(left, 2).starts_with('−') { "−" } else { "+" };
            let words = format!("row {}, column {}:  ${} = {} {sign} {}$", i + 1, j + 1, num(a, 2), num(kept, 2), num(left.abs(), 2));
            if let (Some(at), true) = (f.mouse().and_then(|p| on_screen(f, p.extend(0.0))), lay.wide || f.view_size().x >= 300.0) {
                f.caption_px(at + Vec2::new(0.0, -14.0), words, size, pal::TEXT, Align2::CENTER_BOTTOM);
            }
        }
        // "=" and "+" between them (on a phone smaller: the narrow gap would hold the
        // signs, set as formulas, with no room either side)
        let sign_size = if lay.wide { size + 6.0 } else { size + 2.0 };
        for (c, sign) in [(0, "$=$"), (1, "$+$")] {
            let (_, hi) = lay.cell(c);
            f.text((hi.x + lay.gap * 0.5, hi.y - lay.side * 0.5), sign, sign_size, pal::TEXT, Align2::CENTER_CENTER);
        }
        // what each is, above it
        let room = lay.side + lay.gap * 0.8;
        let noisy = self.noise > 0.0;
        let titles = [
            if noisy { vec!["the picture $A$, with noise".to_string(), "the picture $A$".to_string(), "$A$".to_string()] } else { vec!["the picture $A$".to_string(), "$A$".to_string()] },
            if k == 0 {
                vec!["no pieces yet: $A_0 = 0$".to_string(), "$A_0 = 0$".to_string()]
            } else if k == 1 {
                vec!["the first piece: $A_1$".to_string(), "first: $A_1$".to_string(), "$A_1$".to_string()]
            } else {
                vec![format!("the first {k} pieces: $A_{{{k}}}$"), format!("first {k}: $A_{{{k}}}$"), format!("$A_{{{k}}}$")]
            },
            vec![format!("what is left: $A - A_{{{k}}}$"), format!("left: $A - A_{{{k}}}$"), format!("$A - A_{{{k}}}$")],
        ];
        let colours = [pal::TEXT, KEPT, DROPPED];
        for c in 0..3 {
            let options: Vec<&str> = titles[c].iter().map(String::as_str).collect();
            let (lo, hi) = lay.cell(c);
            f.text((lo.x, hi.y + 3.0 * px), fits(f, &options, room, ppu, size + 1.0), size + 1.0, colours[c], Align2::LEFT_BOTTOM);
        }
        // what each costs or measures, below it
        let below = |c: usize, l: f32| {
            let (lo, _) = lay.cell(c);
            Vec2::new(lo.x, lo.y - 4.0 * px - l * line)
        };
        let small = size - 1.5;
        let mn = N * N;
        let numbers = k * (2 * N + 1);
        f.text(below(0, 0.0), fits(f, &[&format!(r"${N} \times {N}$ pixels: {} numbers", count(mn)), &format!("{} numbers", count(mn))], room, ppu, small), small, pal::DIM, Align2::LEFT_TOP);
        let cost = if k == 0 {
            vec!["0 numbers".to_string()]
        } else {
            let sum = format!(r"{k} \times ({N} + {N} + 1) = {}", count(numbers));
            let more = numbers >= mn;
            vec![
                format!("${sum}$ numbers{}", if more { ": more than $A$" } else { "" }),
                format!("${sum}{}$", if more { " > 4096" } else { "" }),
                format!("{} numbers", count(numbers)),
            ]
        };
        let options: Vec<&str> = cost.iter().map(String::as_str).collect();
        f.text(below(1, 0.0), fits(f, &options, room, ppu, small), small, pal::DIM, Align2::LEFT_TOP);
        if lay.wide && k > 0 {
            let each = if k == 1 { r"($\sigma$, $u$ and $v$ of the piece)" } else { r"($\sigma$, $u$ and $v$ of each piece)" };
            f.text(below(1, 1.0), fits(f, &[each, ""], room, ppu, small), small, pal::DIM, Align2::LEFT_TOP);
        }
        // the size of what is left: by the formula, and measured
        let left = self.dec.left_after(k.min(N));
        let share = if scale > 0.0 { 100.0 * left / scale } else { 0.0 };
        let formula = self.left_formula(k);
        let size_words = if self.dec.rank == 0 {
            vec!["nothing: $A$ is blank".to_string()]
        } else if k >= self.dec.rank {
            let measured = self.misses(k).0;
            vec![format!("nothing: size ${}$ (rounding)", size_text(measured)), format!("size ${}$", size_text(measured))]
        } else {
            vec![format!("size ${formula} = {}$", size_text(left)), format!("size ${}$", size_text(left))]
        };
        let options: Vec<&str> = size_words.iter().map(String::as_str).collect();
        f.text(below(2, 0.0), fits(f, &options, room, ppu, small), small, pal::DIM, Align2::LEFT_TOP);
        if self.dec.rank > 0 && k < self.dec.rank {
            let share_words = format!("{} of $A$'s size ${}$", percent(share), size_text(scale));
            f.text(below(2, 1.0), fits(f, &[&share_words, &percent(share), ""], room, ppu, small), small, pal::DIM, Align2::LEFT_TOP);
        }
        if noisy && lay.wide {
            // how far each is from the picture without its noise
            let was = distance(&self.dec.a, &self.clean);
            let now = self.misses(k).1;
            let words = format!("from the clean picture: ${}$ ($A$: ${}$)", size_text(now), size_text(was));
            f.text(below(1, 2.0), fits(f, &[&words, ""], room, ppu, small), small, pal::DIM, Align2::LEFT_TOP);
        }
    }

    /// The piece on its own, with its column pattern down the left and its row pattern
    /// along the top.
    fn draw_piece(&mut self, f: &mut Frame, lay: &Layout, ppu: f32) {
        let (size, px) = (lay.size, lay.px);
        let p = self.piece_shown();
        let (o, side, strip) = (lay.piece, lay.pside, lay.strip);
        let live = p < self.dec.nonzero();
        let ps = p + 1;
        if let Some(image) = &mut self.image {
            f.image_part(image, [(3 * N) as u32, 0, N as u32, N as u32], Vec2::new(o.x, o.y - side), Vec2::new(o.x + side, o.y), 0.0);
            let (ul, vb) = (o.x - strip - 3.0 * px, o.y + 3.0 * px);
            f.image_part(image, [(4 * N) as u32, 0, 1, N as u32], Vec2::new(ul, o.y - side), Vec2::new(ul + strip, o.y), 0.0);
            f.image_part(image, [0, N as u32, N as u32, 1], Vec2::new(o.x, vb), Vec2::new(o.x + side, vb + strip), 0.0);
            frame_rect(f, Vec2::new(ul, o.y - side), Vec2::new(ul + strip, o.y), 1.0, pal::DIM.a(0.7));
            frame_rect(f, Vec2::new(o.x, vb), Vec2::new(o.x + side, vb + strip), 1.0, pal::DIM.a(0.7));
        }
        frame_rect(f, Vec2::new(o.x, o.y - side), Vec2::new(o.x + side, o.y), 1.0, pal::DIM.a(0.7));
        // the names of the patterns, at the left
        f.text((o.x - strip - 8.0 * px, o.y - side * 0.5), format!("$u_{{{ps}}}$"), size, pal::TEXT, Align2::RIGHT_CENTER);
        let top = o.y + strip + 3.0 * px;
        f.text((o.x - strip - 8.0 * px, top - strip * 0.5), format!(r"$v_{{{ps}}}^\top$"), size, pal::TEXT, Align2::RIGHT_CENTER);
        // the title, above
        let sigma = self.dec.s.get(p).copied().unwrap_or(0.0);
        let title_at = Vec2::new(o.x - strip - 30.0 * px, top + 5.0 * px);
        let room = if lay.wide { lay.chart.x - title_at.x - 20.0 * px } else { lay.cells[2].x + lay.side - title_at.x };
        let added = if self.t <= 1e-4 { " (not added yet)" } else { "" };
        let title = if live {
            let long = format!(r"piece {ps} alone: $\sigma_{{{ps}}} u_{{{ps}}} v_{{{ps}}}^\top$, $\sigma_{{{ps}}} = {}${added}", size_text(sigma));
            let mid = format!(r"piece {ps}: $\sigma_{{{ps}}} u_{{{ps}}} v_{{{ps}}}^\top$, $\sigma_{{{ps}}} = {}$", size_text(sigma));
            let short = format!(r"piece {ps}: $\sigma_{{{ps}}} = {}$", size_text(sigma));
            fits(f, &[&long, &mid, &short], room, ppu, size + 1.0).to_string()
        } else {
            format!(r"piece {ps}: $\sigma_{{{ps}}} = {}$", size_text(sigma))
        };
        f.text(title_at, title, size + 1.0, pal::TEXT, Align2::LEFT_BOTTOM);
        if !live {
            let words = if self.dec.rank == 0 {
                "nothing: the picture is blank"
            } else if sigma > 0.0 {
                "too small to draw: rounding"
            } else {
                "nothing: no pieces are left"
            };
            f.text((o.x + side * 0.5, o.y - side * 0.5), fits(f, &[words, "nothing"], side, ppu, size - 1.0), size - 1.0, pal::DIM, Align2::CENTER_CENTER);
        }
        // the pixel pointed at: its row's entry of u times its column's of v, times σ
        let small = size - 1.5;
        let pointed = self.pointer(f).and_then(|p| pixel_at(p, Vec2::new(o.x, o.y - side), Vec2::new(o.x + side, o.y)));
        let mut said = None;
        // (the factors and what they come to, to be written in a few short lines beside
        // the piece on a phone)
        let mut said_short = None;
        if let (Some((i, j)), true) = (pointed, live) {
            let (ul, vb) = (o.x - strip - 3.0 * px, o.y + 3.0 * px);
            let cell = side / N as f32;
            let (y, x) = (o.y - (i as f32 + 0.5) * cell, o.x + (j as f32 + 0.5) * cell);
            ring_pixel(f, Vec2::new(o.x, o.y - side), Vec2::new(o.x + side, o.y), i, j, pal::WHITE);
            frame_rect(f, Vec2::new(ul, y - cell * 0.5), Vec2::new(ul + strip, y + cell * 0.5), 1.5, pal::WHITE);
            frame_rect(f, Vec2::new(x - cell * 0.5, vb), Vec2::new(x + cell * 0.5, vb + strip), 1.5, pal::WHITE);
            f.dashed((ul + strip, y), (x - cell * 0.5, y), 1.0, pal::WHITE.a(0.6));
            f.dashed((x, vb), (x, y + cell * 0.5), 1.0, pal::WHITE.a(0.6));
            let (ui_, vj) = (self.dec.u_col(p)[i], self.dec.v_col(p)[j]);
            let value = ui_ * vj * sigma;
            // (a factor below zero in brackets: "× (−0.18)")
            let factor = |v: f64| {
                let t = num(v, 3);
                if t.starts_with('−') {
                    format!("({t})")
                } else {
                    t
                }
            };
            let (product, result) = (format!(r"{} \times {} \times {}", num(ui_, 3), factor(vj), num(sigma, 2)), num(value, 3));
            said = Some([format!(r"row {}, column {}:  $u \times v \times \sigma = {product} = {result}$", i + 1, j + 1), format!("${product} = {result}$")]);
            said_short = Some((num(ui_, 3), factor(vj), num(sigma, 2), result));
        }
        // how to read it
        let words = said.unwrap_or_else(|| {
            [
                r"each pixel: its row's colour in $u$ times its column's in $v^\top$ ($\text{blue} \times \text{blue} = \text{red}$), times $\sigma$".to_string(),
                r"$\text{pixel} = u \times v^\top \times \sigma$   ($\text{blue} \times \text{blue} = \text{red}$)".to_string(),
            ]
        });
        if lay.wide {
            let at = Vec2::new(o.x - strip - 30.0 * px, o.y - side - 5.0 * px);
            let room = lay.chart.x - at.x - 10.0 * px;
            f.text(at, fits(f, &[&words[0], &words[1], ""], room, ppu, small), small, pal::DIM, Align2::LEFT_TOP);
        } else {
            // beside the piece, in a few short lines
            let at = Vec2::new(o.x + side + 12.0 * px, o.y);
            let room = lay.cells[2].x + lay.side - at.x;
            let lines = ["each pixel:", "its row's colour", "in $u$, times its", r"column's in $v^\top$,", r"times $\sigma$", r"($\text{blue} \times \text{blue}$", r" $= \text{red}$)"];
            if let Some((a, b, c, result)) = &said_short {
                // (the product on a line of its own if it fits, otherwise in two; a line
                // that goes on from the one above starts with its sign, as "{} = …")
                let two = [format!(r"${a} \times {b} \times {c}$"), format!("${{}} = {result}$")];
                let three = [format!(r"${a} \times {b}$"), format!(r"${{}} \times {c}$"), format!("${{}} = {result}$")];
                let said: &[String] = if f.text_size(&two[0], small).x <= room * ppu { &two } else { &three };
                for (i, l) in said.iter().enumerate() {
                    f.text(at - Vec2::new(0.0, i as f32 * (small + 4.0) * px), l.as_str(), small, pal::TEXT, Align2::LEFT_TOP);
                }
            } else if room * ppu > text_width("its row's colour", small) {
                for (i, l) in lines.iter().enumerate() {
                    f.text(at - Vec2::new(0.0, i as f32 * (small + 4.0) * px), *l, small, pal::DIM, Align2::LEFT_TOP);
                }
            }
        }
    }

    /// The sizes of the pieces, largest first: the kept ones in one colour, the dropped
    /// in another, k marked (and a handle to move it).
    fn draw_chart(&mut self, f: &mut Frame, lay: &Layout, ppu: f32) {
        let (o, s) = (lay.chart, lay.chart_size);
        let (size, line, px) = (lay.size, lay.line, lay.px);
        let small = size - 1.5;
        let n = self.sig.len().max(1);
        let x_of = |i: f32| o.x + s.x * i / n as f32;
        let top = self.top.max(1e-9);
        if self.dec.nonzero() == 0 && !self.fresh {
            f.text(o + s * 0.5, fits(f, &[r"every $\sigma$ is $0$: the picture is blank", r"every $\sigma$ is $0$"], s.x, ppu, small), small, pal::DIM, Align2::CENTER_CENTER);
        }
        f.line(o, o + Vec2::new(s.x, 0.0), 1.0, pal::DIM.a(0.8));
        let bar = s.x / n as f32;
        let gap = if bar * ppu > 4.0 { (0.18 * bar).min(1.5 * px) } else { 0.0 };
        for (i, &v) in self.sig.iter().enumerate() {
            let h = s.y * (v / top).clamp(0.0, 1.0);
            if h <= 0.0 {
                continue;
            }
            let kept = (self.t - i as f32).clamp(0.0, 1.0);
            let color = DROPPED.mix(KEPT, kept);
            f.dl.tris_alpha.rect(Vec2::new(x_of(i as f32) + gap * 0.5, o.y), Vec2::new(x_of(i as f32 + 1.0) - gap * 0.5, o.y + h), 0.0, color.a(0.9));
        }
        // the bar pointed at: its size; a click there keeps the pieces up to it
        let over = self.pointer(f).filter(|p| p.x >= o.x && p.x < o.x + s.x && p.y >= o.y - 2.0 * px && p.y <= o.y + s.y);
        if let Some(p) = over {
            f.handles.claim_clicks();
            let i = (((p.x - o.x) / bar).floor().max(0.0) as usize).min(n - 1);
            let h = s.y * (self.sig[i] / top).clamp(0.0, 1.0);
            frame_rect(f, Vec2::new(x_of(i as f32), o.y), Vec2::new(x_of(i as f32 + 1.0), o.y + h.max(2.0 * px)), 1.2, pal::WHITE);
            if let Some(at) = on_screen(f, Vec3::new(x_of(i as f32 + 0.5), o.y + h, 0.0)) {
                let words = format!(r"$\sigma_{{{}}} = {}$", i + 1, size_text(self.dec.s[i]));
                f.caption_px(at + Vec2::new(0.0, -8.0), words, small, pal::TEXT, Align2::CENTER_BOTTOM);
            }
            if f.clicked_canvas() {
                self.k = i + 1;
                self.build = None;
            }
        }
        // the axes' numbers
        for i in [1usize, 16, 32, 48, 64] {
            f.text((x_of(i as f32 - 0.5), o.y - 3.0 * px), i.to_string(), small, pal::DIM, Align2::CENTER_TOP);
        }
        let step = nice_step(top / 3.0);
        let mut v = 0.0;
        while v <= top * 1.001 {
            let y = o.y + s.y * v / top;
            f.text((o.x - 6.0 * px, y), fmt_tick(v, step), small, pal::DIM, Align2::RIGHT_CENTER);
            if v > 0.0 {
                // (a thin strip rather than a line: lines come and go as the scale
                // glides, and the strip is drawn with the bars)
                f.dl.tris_alpha.rect(Vec2::new(o.x, y - 0.5 * px), Vec2::new(o.x + s.x, y + 0.5 * px), 0.0, pal::FAINT.a(0.35));
            }
            v += step;
        }
        // k: between the kept bars and the dropped ones, with a handle on the axis
        let at = x_of(self.t);
        f.line((at, o.y), (at, o.y + s.y + 4.0 * px), 1.6, K_MARK.a(0.9));
        let mut knob = Vec2::new(at, o.y);
        if f.drag2(1, &mut knob, K_MARK) {
            let k = ((knob.x - o.x) / bar.max(1e-6)).round().max(0.0).min(n as f32) as usize;
            self.build = None;
            if k != self.k {
                self.k = k;
            }
        }
        // k, inside the chart at the top of its line: to the right of it, or to the
        // left near the right-hand end
        let k = self.k_shown();
        let label = format!("$k = {k}$");
        if let Some(mark) = on_screen(f, Vec3::new(at, o.y + s.y, 0.0)) {
            let room = on_screen(f, (o + s).extend(0.0)).map_or(f32::MAX, |end| end.x - mark.x);
            let wide = text_width(&label, size) + 14.0;
            let (x, align) = if room > wide { (mark.x + 6.0, Align2::LEFT_TOP) } else { (mark.x - 6.0, Align2::RIGHT_TOP) };
            f.caption_px(Vec2::new(x, mark.y + 2.0), label, size, K_MARK, align);
        }
        // its title, and the key under it
        let back = if lay.wide { 40.0 } else { 30.0 } * px;
        let title = fits(f, &[r"$\sigma_1 \ge \sigma_2 \ge \cdots$: the size of each piece, the largest first", r"the size $\sigma$ of each piece", r"$\sigma$"], s.x + back, ppu, size + 1.0);
        f.text(o + Vec2::new(-back, s.y + 5.0 * px), title, size + 1.0, pal::TEXT, Align2::LEFT_BOTTOM);
        let key = [(KEPT, format!("kept: they add up to $A_{{{k}}}$")), (DROPPED, format!("dropped: they add up to $A - A_{{{k}}}$"))];
        let key_y = o.y - 3.0 * px - 1.5 * line;
        let mut x = o.x - 30.0 * px;
        for (color, name) in &key {
            swatch(f, Vec2::new(x, key_y), 9.0 * px, *color);
            f.text((x + 13.0 * px, key_y), name.as_str(), small, *color, Align2::LEFT_CENTER);
            x += (text_width(name, small) + 34.0) * px;
        }
    }
}

/// The pixel (row, column) of a picture drawn from `lo` (bottom-left) to `hi` that the
/// point `p` is on, if it is on it.
fn pixel_at(p: Vec2, lo: Vec2, hi: Vec2) -> Option<(usize, usize)> {
    let inside = p.x >= lo.x && p.x < hi.x && p.y > lo.y && p.y <= hi.y;
    let cell = |d: f32, len: f32| ((d / len.max(1e-9) * N as f32).floor().max(0.0) as usize).min(N - 1);
    inside.then(|| (cell(hi.y - p.y, hi.y - lo.y), cell(p.x - lo.x, hi.x - lo.x)))
}

/// The outline of pixel (i, j) of a picture drawn from `lo` to `hi`.
fn ring_pixel(f: &mut Frame, lo: Vec2, hi: Vec2, i: usize, j: usize, color: Color) {
    let cell = (hi - lo) / N as f32;
    let min = Vec2::new(lo.x + j as f32 * cell.x, hi.y - (i + 1) as f32 * cell.y);
    frame_rect(f, min, min + cell, 1.5, color);
}

/// A filled square `side` wide, its left edge's middle at `at`: a swatch for a key.
fn swatch(f: &mut Frame, at: Vec2, side: f32, color: Color) {
    f.dl.tris_alpha.rect(Vec2::new(at.x, at.y - side * 0.5), Vec2::new(at.x + side, at.y + side * 0.5), 0.0, color);
}

/// A share in per cent, with one decimal below 10 %.
fn percent(p: f64) -> String {
    if p < 0.05 {
        "under 0.1 %".to_string()
    } else if p < 10.0 {
        format!("{p:.1} %")
    } else {
        format!("{p:.0} %")
    }
}

impl Scene for LowRank {
    fn view(&self) -> View {
        STAGE_VIEW
    }

    fn settings(&self, s: &mut ViewSettings) {
        s.grid = false;
        s.bloom = s.bloom.min(0.2);
    }

    /// Moving: while the decomposition is being worked out, while the pieces are being
    /// added or k glides, and while the bars glide to a new picture's sizes.
    fn animated(&self) -> bool {
        self.job.is_some() || self.build.is_some() || self.t != self.k as f32 || !self.bars_still() || self.stroke.is_some()
    }

    fn pointer_hint(&self) -> Option<&'static str> {
        Some("drag on the picture $A$: draw on it    point at a pixel or a bar: its numbers    drag the yellow dot, or click a bar: pieces kept")
    }

    fn ui(&mut self, ui: &mut egui::Ui, _: &Gfx) {
        ui::note(ui, "A grey picture is a grid of numbers: a matrix. The singular value decomposition writes it as a sum of pieces,");
        ui::equation(ui, r"A = \sigma_1 u_1 v_1^\top + \sigma_2 u_2 v_2^\top + \cdots");
        ui::note(
            ui,
            r"each one column pattern $u$ times one row pattern $v^\top$, times a size $\sigma$, the biggest first. Keep the first $k$ pieces and drop the rest: no picture made of $k$ pieces is closer to $A$, and the size of what is dropped is exactly",
        );
        ui::equation(ui, r"\sqrt{\sigma_{k+1}^2 + \sigma_{k+2}^2 + \cdots}");

        ui::section(ui, "Things to try");
        for (label, look_for, setup) in TRY {
            if ui::experiment(ui, label, look_for) {
                self.apply(setup);
            }
        }

        ui::section(ui, "The picture");
        let names: Vec<&str> = PICS.iter().map(|p| p.1).collect();
        let mut chosen = self.pic.and_then(|p| PICS.iter().position(|q| q.0 == p)).unwrap_or(usize::MAX);
        if ui::choice(ui, &names, &mut chosen) {
            if let Some(&(pic, _)) = PICS.get(chosen) {
                self.choose(pic);
            }
        }
        ui.horizontal_wrapped(|ui| {
            ui::choice(ui, &["white pen", "black pen"], &mut self.pen);
            if ui.button("clear").clicked() {
                self.pic = None;
                self.clean = vec![0.0; N * N];
                self.new_picture();
            }
        });
        ui::note(ui, "Drag across the picture $A$ to draw on it: it is taken apart again as you draw.");
        if ui::slider(ui, "noise", &mut self.noise, 0.0..=0.4) {
            self.touched();
        }
        if ui.button("new noise").clicked() {
            self.seed += 1;
            self.touched();
        }
        ui::note(ui, "A random number added to every pixel; the slider sets how large they typically are (their standard deviation).");

        ui::section(ui, "Pieces kept");
        // (while the pieces are being added, the slider shows how many are in so far)
        let mut k = if self.build.is_some() { self.k_shown() } else { self.k };
        if ui::slider_int(ui, "pieces kept, $k$", &mut k, 0..=N) {
            self.k = k;
            self.build = None;
        }
        let playing = self.build.is_some();
        if ui.button(if playing { "❚❚ pause" } else { "▶ add them one by one" }).clicked() {
            if playing {
                self.build = None;
                self.k = self.t.round() as usize;
            } else {
                self.play();
            }
        }
        ui::note(ui, "Or drag the yellow dot under the chart, or click a bar: the pieces up to it are kept.");

        ui::section(ui, "Reading the picture");
        ui::key(
            ui,
            &[
                (GREY.color(0.75), "grey pictures: $A$ and $A_k$, black 0 to white 1 (beyond that, drawn black or white)"),
                (SIGNED.table()[table_index(0.1)], r"blue below zero, red above, dark zero: what is left, the piece alone and its patterns $u$ and $v^\top$"),
                (KEPT, "green: the pieces kept, whose sum is $A_k$"),
                (DROPPED, "purple: the pieces dropped, whose sum is $A - A_k$"),
                (K_MARK, "yellow: $k$, the number kept (drag it)"),
            ],
        );
        ui::note(ui, r"What is left is drawn at full colour at $\pm 0.25$. The piece alone is drawn as bright as it goes, whatever its size: its size is $\sigma$. Point at a pixel for its numbers: in the sum, $A = A_k + \text{what is left}$; in the piece, $u \times v \times \sigma$, with its entry of $u$ and of $v$ marked on the strips.");

        ui::section(ui, "Words");
        ui::term(ui, "Piece (rank one):", r"a column pattern $u$ times a row pattern $v^\top$, times a size $\sigma$: pixel $(i, j)$ is $\sigma u_i v_j$. Every row is the row pattern, scaled.");
        ui::term(ui, r"Singular values $\sigma_1 \ge \sigma_2 \ge \cdots$:", r"the sizes of the pieces: $u$ and $v$ each have length 1, so the piece $\sigma u v^\top$ has size $\sigma$.");
        ui::term(ui, "Rank:", r"how many pieces a picture needs, the number of $\sigma$ that are not zero; $A_k$, a sum of $k$ pieces, has rank $k$ (at most). Counted here: the $\sigma$ larger than $\sigma_1 / 10^{10}$ (below that they are rounding, or too small to matter).");
        ui::term(ui, "Size of a picture:", "the square root of the sum of the squares of its pixels, its length as a list of 4096 numbers.");
        ui::term(ui, "Eckart–Young:", "the theorem that the first $k$ pieces are the closest picture of rank $k$, measured by that size.");

        ui::section(ui, "Numbers");
        let k = self.k_shown();
        let scale = size_of(&self.dec.a);
        let left = self.dec.left_after(k);
        let (measured, from_clean) = self.misses(k);
        // (the sizes in a formula each: a size down at rounding is a power of ten)
        let size = |v: f64| format!("${}$", size_text(v));
        ui::readout(ui, "pieces kept, $k$", format!("${k}$"));
        ui::readout(ui, "pieces not zero (rank of $A$)", format!("${}$", self.dec.nonzero()));
        ui::readout_colored(ui, "size of $A - A_k$, measured", size(measured), DROPPED);
        let formula = if k >= N { "by the formula (nothing dropped)".to_string() } else { format!("${}$ by the formula", self.left_formula(k)) };
        ui::readout_colored(ui, &formula, size(left), DROPPED);
        ui::readout(ui, "size of $A$", size(scale));
        ui::readout(ui, "numbers kept, $k(64 + 64 + 1)$", format!("{} of {}", count(k * (2 * N + 1)), count(N * N)));
        if k > 0 && k < N {
            ui::readout(ui, &format!(r"a rival: $A$ in ${k} \times {k}$ blocks, leaves"), size(distance(&self.dec.a, &blocky(&self.dec.a, N, N, k))));
            ui::note(ui, r"A picture of $k \times k$ big pixels, each the average of $A$ under it, is also a sum of $k$ pieces (its rows come in $k$ kinds). It never leaves less than the first $k$ pieces do: no picture of rank $k$ is closer to $A$.");
        }
        if self.noise > 0.0 {
            ui::readout(ui, "$A$ from the clean picture", size(distance(&self.dec.a, &self.clean)));
            ui::readout_colored(ui, "$A_k$ from the clean picture", size(from_clean), KEPT);
        }
        ui::see_also(
            ui,
            &[
                ("svd", r"the SVD of a $2 \times 2$ matrix, step by step"),
                ("pca", "principal components: the same pieces for a cloud of points"),
                ("dft", "the discrete Fourier transform: fixed patterns, waves, instead of the picture's own"),
                ("sae", "sparse autoencoders: pieces learned from a network's activations"),
            ],
        );
    }

    fn frame(&mut self, f: &mut Frame) {
        // (the decomposition answers the settings, so it goes on while paused)
        self.work(&crate::clock::frame_budget(f.real_dt, 0.004));
        self.advance(f.dt, f.real_dt);
        let (w, ppu) = (stage_width(f), pixels_per_unit(f));
        let lay = Layout::of(w, ppu, f.top_inset);
        self.pen_input(f, &lay);
        self.paint(f.gfx);
        let words = f.dl.labels.len();
        self.draw_sum(f, &lay, ppu);
        self.draw_piece(f, &lay, ppu);
        self.draw_chart(f, &lay, ppu);
        // a canvas squeezed to a sliver (the controls open over a phone's), or one far
        // too small, has no room to read anything: the pictures stay, the words piled
        // on them do not
        let view = f.view_size();
        if view.x < 200.0 || view.y < 260.0 {
            f.dl.labels.truncate(words);
        }
    }
}

// ----- the decomposition ----------------------------------------------------------------

/// The singular value decomposition of an m × n picture by one-sided Jacobi rotations,
/// worked out a little at a time so that no frame waits for all of it.
///
/// (`crate::math::linalg::svd` does the same in one go, and is what this is tested
/// against. It is not used here for two reasons. A picture with many zero singular
/// values, a disc for instance, keeps it turning rounding noise for all of its sixty
/// sweeps, 50 ms for 64 × 64; here a column that has come down to rounding is left
/// alone. And a picture being drawn on changes a little at a time: started from the
/// last answer's directions, this needs a few sweeps fewer, 8 instead of 11 for a
/// stroke across the house.)
struct Jacobi {
    m: usize,
    n: usize,
    /// The picture (row-major) this is the decomposition of.
    a: Vec<f64>,
    /// A V, column by column (column j is `w[j * m..(j + 1) * m]`), turned until its
    /// columns are at right angles: then they are σⱼ uⱼ.
    w: Vec<f64>,
    /// V, column by column.
    v: Vec<f64>,
    /// The next pair of columns to turn, and the sweep it belongs to.
    p: usize,
    q: usize,
    sweeps: u32,
    /// The largest cosine between two columns met in this sweep so far.
    off: f64,
    /// A column whose squared length is below this is rounding, not a direction.
    floor: f64,
    done: bool,
}

/// Sweeps after which the decomposition stops even if it has not settled (it settles in
/// fewer than twenty for every picture here: `the_decomposition_is_right`).
const MOST_SWEEPS: u32 = 60;

impl Jacobi {
    /// Start on the picture `a` (m × n, row-major), from the directions `start` (an
    /// n × n orthogonal matrix, column by column), or from the axes if none is given.
    fn new(a: &[f64], m: usize, n: usize, start: Option<&[f64]>) -> Jacobi {
        let v = match start {
            Some(v) if v.len() == n * n => v.to_vec(),
            _ => (0..n * n).map(|k| if k / n == k % n { 1.0 } else { 0.0 }).collect(),
        };
        let mut w = vec![0.0; m * n];
        for j in 0..n {
            let col = &v[j * n..(j + 1) * n];
            for i in 0..m {
                let row = &a[i * n..(i + 1) * n];
                w[j * m + i] = row.iter().zip(col).map(|(x, y)| x * y).sum();
            }
        }
        let total: f64 = a.iter().map(|x| x * x).sum();
        Jacobi { m, n, a: a.to_vec(), w, v, p: 0, q: 1, sweeps: 0, off: 0.0, floor: 1e-26 * total, done: n < 2 }
    }

    /// Turn pairs of columns until done or until `budget` is spent. Returns true once done.
    fn run(&mut self, budget: &crate::clock::Budget) -> bool {
        let (m, n) = (self.m, self.n);
        let mut since_check = 0;
        while !self.done {
            let (p, q) = (self.p, self.q);
            let (wp, wq) = two_columns(&mut self.w, m, p, q);
            let (mut alpha, mut beta, mut gamma) = (0.0, 0.0, 0.0);
            for (x, y) in wp.iter().zip(wq.iter()) {
                alpha += x * x;
                beta += y * y;
                gamma += x * y;
            }
            // (a column of rounding noise is never orthogonal to anything, and needs
            // no direction: it is left alone)
            if alpha > self.floor && beta > self.floor && gamma.abs() > 1e-15 * (alpha * beta).sqrt() {
                self.off = self.off.max(gamma.abs() / (alpha * beta).sqrt());
                let zeta = (beta - alpha) / (2.0 * gamma);
                let t = zeta.signum() / (zeta.abs() + (1.0 + zeta * zeta).sqrt());
                let c = 1.0 / (1.0 + t * t).sqrt();
                let s = c * t;
                turn(wp, wq, c, s);
                let (vp, vq) = two_columns(&mut self.v, n, p, q);
                turn(vp, vq, c, s);
            }
            // the next pair, row by row
            self.q += 1;
            if self.q >= n {
                self.p += 1;
                self.q = self.p + 1;
                if self.q >= n {
                    self.sweeps += 1;
                    self.done = self.off < 1e-13 || self.sweeps >= MOST_SWEEPS;
                    (self.p, self.q, self.off) = (0, 1, 0.0);
                }
            }
            since_check += 1;
            if since_check >= 32 {
                since_check = 0;
                if budget.spent() {
                    break;
                }
            }
        }
        self.done
    }

    /// The decomposition, once done: the singular values largest first, with their
    /// patterns, signed so that each piece's patterns point the way the `like` ones do
    /// (the last decomposition, so that a picture drawn on does not flicker between a
    /// pattern and its negative), or else so that they mostly add up to more than nothing.
    fn finish(&self, like: Option<&Dec>) -> Dec {
        let (m, n) = (self.m, self.n);
        let r = m.min(n);
        let lengths: Vec<f64> = (0..n).map(|j| self.w[j * m..(j + 1) * m].iter().map(|x| x * x).sum::<f64>().sqrt()).collect();
        let mut order: Vec<usize> = (0..n).collect();
        // (a stable sort: equal singular values keep the order of their columns)
        order.sort_by(|&x, &y| lengths[y].total_cmp(&lengths[x]));
        let biggest = lengths.iter().fold(0.0f64, |b, l| b.max(*l));
        let (mut s, mut u, mut v) = (vec![0.0; r], vec![0.0; r * m], vec![0.0; r * n]);
        for (k, &j) in order.iter().take(r).enumerate() {
            let length = lengths[j];
            // (what is left of a direction that came down to rounding is no singular value)
            if length <= 1e-13 * biggest || length == 0.0 {
                continue;
            }
            let uk: Vec<f64> = self.w[j * m..(j + 1) * m].iter().map(|x| x / length).collect();
            let vk = &self.v[j * n..(j + 1) * n];
            let sum: f64 = uk.iter().sum::<f64>() + vk.iter().sum::<f64>();
            let flip = match like.filter(|d| d.m == m && d.n == n && k < d.rank) {
                Some(d) => {
                    let along: f64 = vk.iter().zip(d.v_col(k)).map(|(x, y)| x * y).sum();
                    if along.abs() > 0.3 {
                        along < 0.0
                    } else {
                        sum < 0.0
                    }
                }
                None => sum < 0.0,
            };
            let sign = if flip { -1.0 } else { 1.0 };
            s[k] = length;
            u[k * m..(k + 1) * m].iter_mut().zip(&uk).for_each(|(o, x)| *o = sign * x);
            v[k * n..(k + 1) * n].iter_mut().zip(vk).for_each(|(o, x)| *o = sign * x);
        }
        let rank = s.iter().filter(|x| **x > 0.0).count();
        Dec { m, n, a: self.a.clone(), s, u, v, rank, sweeps: self.sweeps, directions: self.v.clone() }
    }
}

/// Columns `p` < `q` of a matrix stored column by column, columns `len` long, both
/// at once.
fn two_columns(x: &mut [f64], len: usize, p: usize, q: usize) -> (&mut [f64], &mut [f64]) {
    let (head, tail) = x.split_at_mut(q * len);
    (&mut head[p * len..(p + 1) * len], &mut tail[..len])
}

/// Turn the pair of columns (x, y) by the rotation with cosine `c` and sine `s`.
fn turn(x: &mut [f64], y: &mut [f64], c: f64, s: f64) {
    for (a, b) in x.iter_mut().zip(y.iter_mut()) {
        let (p, q) = (*a, *b);
        *a = c * p - s * q;
        *b = s * p + c * q;
    }
}

/// A picture and its decomposition A = σ₁u₁v₁ᵀ + σ₂u₂v₂ᵀ + …, largest σ first.
struct Dec {
    m: usize,
    n: usize,
    a: Vec<f64>,
    /// min(m, n) singular values, largest first (zero where the picture has no more).
    s: Vec<f64>,
    /// Column patterns u (m long each) and row patterns v (n long each), piece by piece.
    u: Vec<f64>,
    v: Vec<f64>,
    /// How many singular values are not zero (to rounding).
    rank: usize,
    /// How many sweeps it took.
    #[cfg_attr(not(test), allow(dead_code))]
    sweeps: u32,
    /// The directions as the rotations left them (unsorted): where the next
    /// decomposition of a slightly changed picture starts from.
    directions: Vec<f64>,
}

impl Dec {
    fn u_col(&self, k: usize) -> &[f64] {
        &self.u[k * self.m..(k + 1) * self.m]
    }

    fn v_col(&self, k: usize) -> &[f64] {
        &self.v[k * self.n..(k + 1) * self.n]
    }

    /// Add `weight` times piece `k` (counting from 0) to `out` (m × n, row-major).
    fn add_piece(&self, k: usize, weight: f64, out: &mut [f64]) {
        if k >= self.rank {
            return;
        }
        let w = weight * self.s[k];
        let v = self.v_col(k);
        for (i, ui) in self.u_col(k).iter().enumerate() {
            let c = w * ui;
            for (o, vj) in out[i * self.n..(i + 1) * self.n].iter_mut().zip(v) {
                *o += c * vj;
            }
        }
    }

    /// The first `k` pieces added up: the best picture of rank k.
    fn approx(&self, k: usize) -> Vec<f64> {
        let mut out = vec![0.0; self.m * self.n];
        for i in 0..k.min(self.rank) {
            self.add_piece(i, 1.0, &mut out);
        }
        out
    }

    /// How many pieces are not zero: larger than a ten-billionth of the first. (The
    /// rest of the 64, for a picture like the blobs, are not zero but far too small to
    /// matter, and further down they are rounding.)
    fn nonzero(&self) -> usize {
        let first = self.s.first().copied().unwrap_or(0.0);
        self.s.iter().filter(|x| **x > 1e-10 * first).count()
    }

    /// The top of the chart of the σ: the first (1 for a blank picture, which has none).
    fn top(&self) -> f32 {
        if self.nonzero() == 0 {
            1.0
        } else {
            self.s[0] as f32
        }
    }

    /// √(σ²ₖ₊₁ + σ²ₖ₊₂ + …): what the first k pieces leave, by the formula.
    fn left_after(&self, k: usize) -> f64 {
        self.s.iter().skip(k).map(|x| x * x).sum::<f64>().sqrt()
    }

    /// The decomposition worked out at once (the tests, and headless runs).
    fn of(a: &[f64], m: usize, n: usize) -> Dec {
        let mut job = Jacobi::new(a, m, n, None);
        job.run(&crate::clock::Budget::new(f64::INFINITY));
        job.finish(None)
    }
}

/// A rival of rank k (at most): the picture in k × k big pixels, each the average of
/// the pixels of A under it (the bands as even as whole pixels allow). Its rows come in
/// k kinds, one for each band of rows, so it is a sum of k pieces at most.
fn blocky(a: &[f64], m: usize, n: usize, k: usize) -> Vec<f64> {
    if k == 0 {
        return vec![0.0; m * n];
    }
    let band = |len: usize, i: usize| (i * k.min(len)) / len;
    let (kr, kc) = (k.min(m), k.min(n));
    let (mut sums, mut counts) = (vec![0.0; kr * kc], vec![0usize; kr * kc]);
    for i in 0..m {
        for j in 0..n {
            let b = band(m, i) * kc + band(n, j);
            sums[b] += a[i * n + j];
            counts[b] += 1;
        }
    }
    (0..m * n).map(|at| {
        let b = band(m, at / n) * kc + band(n, at % n);
        sums[b] / counts[b].max(1) as f64
    }).collect()
}

/// The size of a picture: the square root of the sum of the squares of its pixels.
fn size_of(x: &[f64]) -> f64 {
    x.iter().map(|v| v * v).sum::<f64>().sqrt()
}

/// The size of the difference of two pictures.
fn distance(x: &[f64], y: &[f64]) -> f64 {
    x.iter().zip(y).map(|(a, b)| (a - b) * (a - b)).sum::<f64>().sqrt()
}

// ----- the pictures ---------------------------------------------------------------------

/// The pictures to choose from.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Pic {
    Plaid,
    Blobs,
    Letter,
    House,
    Line,
    Static,
}

const PICS: [(Pic, &str); 6] = [(Pic::Plaid, "plaid"), (Pic::Blobs, "blobs"), (Pic::Letter, "letter"), (Pic::House, "house"), (Pic::Line, "line"), (Pic::Static, "random")];

/// Brightness of one stripe pattern, for the plaid: a dull ground with bands across it.
fn stripes(i: usize, bands: &[(usize, usize, f64)]) -> f64 {
    bands.iter().find(|(a, b, _)| (*a..*b).contains(&i)).map_or(0.3, |(_, _, v)| *v)
}

/// The distance from `p` to the segment from `a` to `b` (f64).
fn to_segment(p: glam::DVec2, a: glam::DVec2, b: glam::DVec2) -> f64 {
    let ab = b - a;
    let t = ((p - a).dot(ab) / ab.length_squared().max(1e-12)).clamp(0.0, 1.0);
    p.distance(a + ab * t)
}

/// Is `p` inside the triangle?
fn in_triangle(p: glam::DVec2, t: [glam::DVec2; 3]) -> bool {
    let side = |a: glam::DVec2, b: glam::DVec2| (b - a).perp_dot(p - a);
    let (x, y, z) = (side(t[0], t[1]), side(t[1], t[2]), side(t[2], t[0]));
    (x >= 0.0 && y >= 0.0 && z >= 0.0) || (x <= 0.0 && y <= 0.0 && z <= 0.0)
}

/// A picture drawn by `shade` (x to the right and y up, both from −1 to 1), each pixel
/// the average of 4 × 4 points in it, so that edges are smooth.
fn painted(shade: impl Fn(glam::DVec2) -> f64) -> Vec<f64> {
    let mut a = vec![0.0; N * N];
    for i in 0..N {
        for j in 0..N {
            let mut sum = 0.0;
            for si in 0..4 {
                for sj in 0..4 {
                    let x = -1.0 + 2.0 * (j as f64 + (sj as f64 + 0.5) / 4.0) / N as f64;
                    let y = 1.0 - 2.0 * (i as f64 + (si as f64 + 0.5) / 4.0) / N as f64;
                    sum += shade(glam::DVec2::new(x, y));
                }
            }
            a[i * N + j] = sum / 16.0;
        }
    }
    a
}

/// The picture `pic`, N × N, row-major from the top, 0 black and 1 white.
fn picture(pic: Pic) -> Vec<f64> {
    use glam::DVec2 as P;
    match pic {
        // one column pattern times one row pattern
        Pic::Plaid => {
            let rows = [(5, 12, 1.0), (17, 19, 0.65), (27, 38, 0.9), (44, 46, 0.6), (52, 59, 0.8)];
            let cols = [(3, 8, 0.75), (14, 25, 1.0), (31, 33, 0.6), (39, 47, 0.85), (55, 61, 0.7)];
            (0..N * N).map(|k| stripes(k / N, &rows) * stripes(k % N, &cols)).collect()
        }
        // soft round and oval blobs
        Pic::Blobs => painted(|p| {
            let blob = |c: P, sx: f64, sy: f64, angle: f64, height: f64| {
                let d = P::from_angle(-angle).rotate(p - c);
                height * (-0.5 * ((d.x / sx).powi(2) + (d.y / sy).powi(2))).exp()
            };
            blob(P::new(-0.42, 0.38), 0.24, 0.24, 0.0, 0.95) + blob(P::new(0.32, -0.12), 0.42, 0.14, 0.6, 0.85) + blob(P::new(-0.35, -0.55), 0.13, 0.13, 0.0, 0.7)
        }),
        // a capital A: two slanted strokes and a straight one
        Pic::Letter => painted(|p| {
            let (top, left, right) = (P::new(0.0, 0.78), P::new(-0.52, -0.78), P::new(0.52, -0.78));
            let bar = to_segment(p, P::new(-0.27, -0.12), P::new(0.27, -0.12));
            let d = to_segment(p, left, top).min(to_segment(p, top, right)).min(bar);
            if d < 0.09 {
                1.0
            } else {
                0.0
            }
        }),
        // a house under a sun
        Pic::House => painted(|p| {
            let in_rect = |x0: f64, y0: f64, x1: f64, y1: f64| p.x >= x0 && p.x <= x1 && p.y >= y0 && p.y <= y1;
            if in_rect(-0.36, -0.6, -0.16, -0.18) {
                0.2
            } else if in_rect(-0.02, -0.22, 0.2, -0.02) {
                0.95
            } else if in_rect(-0.6, -0.6, 0.34, 0.02) {
                0.65
            } else if in_triangle(p, [P::new(-0.74, 0.02), P::new(0.48, 0.02), P::new(-0.13, 0.5)]) {
                0.45
            } else if p.distance(P::new(0.55, 0.6)) < 0.2 {
                1.0
            } else if p.y < -0.6 {
                0.3
            } else {
                0.08 + 0.12 * (p.y + 1.0)
            }
        }),
        // the identity matrix, drawn
        Pic::Line => (0..N * N).map(|k| if k / N == k % N { 1.0 } else { 0.0 }).collect(),
        // every pixel a random grey
        Pic::Static => {
            let mut rng = Rng::new(17);
            (0..N * N).map(|_| rng.range(0.0, 1.0) as f64).collect()
        }
    }
}

/// Noise to add to a picture: each pixel a normal random number times `level`.
fn noise(level: f64, seed: u64) -> Vec<f64> {
    let mut rng = Rng::new(seed);
    (0..N * N)
        .map(|_| {
            // (Box and Muller)
            let (u1, u2) = (rng.range(1e-9, 1.0) as f64, rng.range(0.0, 1.0) as f64);
            level * (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos()
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The decomposition is right: it agrees with `linalg::svd`, its patterns are at
    /// right angles and of length one, and its pieces add up to the picture; for every
    /// picture, and for random matrices of other shapes.
    #[test]
    fn the_decomposition_is_right() {
        let mut cases: Vec<(Vec<f64>, usize, usize)> = PICS.iter().map(|(p, _)| (picture(*p), N, N)).collect();
        let mut rng = Rng::new(5);
        for (m, n) in [(7usize, 5usize), (5, 7), (12, 12), (1, 4), (4, 1), (2, 2)] {
            cases.push(((0..m * n).map(|_| rng.range(-1.0, 1.0) as f64).collect(), m, n));
        }
        cases.push((vec![0.0; N * N], N, N));
        for (a, m, n) in cases {
            let d = Dec::of(&a, m, n);
            let (_, s, _) = crate::math::linalg::svd(&a, m, n);
            let scale = size_of(&a).max(1e-300);
            for (x, y) in d.s.iter().zip(&s) {
                assert!((x - y).abs() < 1e-10 * scale, "{m}×{n}: {:?} against {:?}", &d.s[..4.min(d.s.len())], &s[..4.min(s.len())]);
            }
            assert!(d.s.windows(2).all(|w| w[0] >= w[1]));
            assert!(distance(&d.approx(m.min(n)), &a) < 1e-10 * scale.max(1.0), "{m}×{n}");
            // (the patterns of pieces far smaller than the first are only as exact as
            // rounding leaves them: those are not looked at)
            let big = d.s.iter().filter(|x| **x > 1e-8 * d.s[0]).count();
            for k in 0..big {
                for l in 0..big {
                    let uu: f64 = d.u_col(k).iter().zip(d.u_col(l)).map(|(x, y)| x * y).sum();
                    let vv: f64 = d.v_col(k).iter().zip(d.v_col(l)).map(|(x, y)| x * y).sum();
                    let want = if k == l { 1.0 } else { 0.0 };
                    assert!((uu - want).abs() < 1e-9 && (vv - want).abs() < 1e-9, "{m}×{n}: {k}, {l}: {uu} {vv}");
                }
            }
            assert!(d.sweeps < 20, "{m}×{n}: {} sweeps", d.sweeps);
        }
    }

    /// The scene set up as experiment `k` (with noise of another seed, if given), its
    /// decomposition worked out and its pieces added, as the scene does it at 60 frames
    /// a second.
    fn run(k: usize, seed: u64) -> LowRank {
        let mut s = LowRank::new();
        s.apply(Setup { seed, ..TRY[k].2 });
        s.settle();
        for _ in 0..3000 {
            s.advance(1.0 / 60.0, 1.0 / 60.0);
            if !s.animated() {
                break;
            }
        }
        assert!(!s.animated(), "experiment {k} comes to rest");
        assert_eq!((s.t, s.k_shown()), (TRY[k].2.k as f32, TRY[k].2.k));
        s
    }

    fn share(d: &Dec, k: usize) -> f64 {
        d.left_after(k) / size_of(&d.a)
    }

    /// The scene opens on "a few pieces carry a picture", adding its five pieces one by
    /// one, evenly, three a second.
    #[test]
    fn it_opens_adding_the_pieces_evenly() {
        let mut s = LowRank::new();
        assert_eq!((s.pic, s.k, s.noise), (Some(Pic::Blobs), 5, 0.0));
        // the pieces wait for the decomposition
        s.advance(0.1, 0.1);
        assert_eq!(s.t, 0.0);
        s.settle();
        let mut seen = vec![s.t];
        while s.build.is_some() {
            s.advance(1.0 / 60.0, 1.0 / 60.0);
            seen.push(s.t);
        }
        let steps: Vec<f32> = seen.windows(2).map(|w| w[1] - w[0]).collect();
        // (all but the last, which is what was left)
        assert!(steps[..steps.len() - 1].iter().all(|d| (d - RATE / 60.0).abs() < 1e-5), "{steps:?}");
        assert!(seen.len() as f32 / 60.0 > 1.6 && seen.len() as f32 / 60.0 < 1.75, "{} frames", seen.len());
        assert_eq!(s.t, 5.0);
        // and once there, nothing moves
        for _ in 0..60 {
            s.advance(1.0 / 60.0, 1.0 / 60.0);
        }
        assert!(!s.animated());
    }

    /// "A plaid is one piece": every row the same pattern, scaled; σ₁ = 25.6 and the
    /// rest 0; the first piece is the picture, to 10⁻¹³ or less; 129 numbers.
    #[test]
    fn a_plaid_is_one_piece() {
        let s = run(0, 1);
        let a = &s.a;
        for i in 0..N {
            let scale = a[i * N] / a[0];
            assert!((0..N).all(|j| (a[i * N + j] - scale * a[j]).abs() < 1e-12), "row {i}");
        }
        let d = &s.dec;
        assert!((d.s[0] - 25.6).abs() < 0.05, "{}", d.s[0]);
        assert_eq!((d.rank, d.nonzero()), (1, 1));
        assert!(d.s[1..].iter().all(|x| *x == 0.0));
        let left = distance(&d.approx(1), a);
        assert!(left < 1e-13 && d.left_after(1) == 0.0, "{left}");
        assert_eq!(2 * N + 1, 129);
    }

    /// "A few pieces carry a picture": 15.6, 8.8, 4.7, 2.2, 1.4, 0.5; five pieces leave
    /// 3 % (to the nearest per cent), in 645 numbers.
    #[test]
    fn a_few_pieces_carry_a_picture() {
        let s = run(1, 1);
        let d = &s.dec;
        for (x, want) in d.s.iter().zip([15.6, 8.8, 4.7, 2.2, 1.4, 0.5]) {
            assert!((x - want).abs() < 0.05, "{:?}", &d.s[..6]);
        }
        let left = share(d, 5);
        assert!((0.025..0.035).contains(&left), "{left}");
        assert_eq!(5 * (2 * N + 1), 645);
    }

    /// "A line: the hardest picture": the identity; every σ is 1; k pieces leave
    /// √(64 − k) of 8, so eight leave 94 %; each piece found is one pixel of the line;
    /// and any other eight directions do exactly as well.
    #[test]
    fn a_line_is_the_hardest_picture() {
        let s = run(2, 1);
        let d = &s.dec;
        assert!((0..N * N).all(|at| s.a[at] == if at / N == at % N { 1.0 } else { 0.0 }));
        assert!(d.s.iter().all(|x| (x - 1.0).abs() < 1e-12));
        for k in 0..=N {
            assert!((d.left_after(k) - ((N - k) as f64).sqrt()).abs() < 1e-10);
        }
        assert_eq!(format!("{:.0}", 100.0 * share(d, 8)), "94");
        // the first eight pieces put back the first eight pixels of the line, and no others
        let a8 = d.approx(8);
        for at in 0..N * N {
            let want = if at / N == at % N && at / N < 8 { 1.0 } else { 0.0 };
            assert!((a8[at] - want).abs() < 1e-12, "pixel {at}: {}", a8[at]);
        }
        // eight other directions, at random: the picture they make leaves just as much
        let mut rng = Rng::new(9);
        let mut q: Vec<f64> = (0..N * 8).map(|_| rng.range(-1.0, 1.0) as f64).collect();
        crate::math::linalg::orthonormalise(&mut q, N, 8);
        let other: Vec<f64> = (0..N * N).map(|at| (0..8).map(|c| q[(at / N) * 8 + c] * q[(at % N) * 8 + c]).sum()).collect();
        assert!((distance(&other, &s.a) - 56f64.sqrt()).abs() < 1e-10);
        assert!(distance(&other, &a8) > 1.0, "and it is a different picture");
    }

    /// "Noise does not compress": the first piece is about the average grey (within a
    /// ninth of its size); then 4.6, 4.3, 4.3, 4.2, …, 3.0 at the sixteenth; sixteen
    /// pieces, half the numbers, leave 31 %.
    #[test]
    fn noise_does_not_compress() {
        let s = run(3, 1);
        let d = &s.dec;
        let mean = s.a.iter().sum::<f64>() / (N * N) as f64;
        let mut first = vec![0.0; N * N];
        d.add_piece(0, 1.0, &mut first);
        assert!(distance(&first, &vec![mean; N * N]) < size_of(&first) / 9.0);
        for (x, want) in d.s[1..5].iter().zip([4.6, 4.3, 4.3, 4.2]) {
            assert!((x - want).abs() < 0.05, "{:?}", &d.s[1..5]);
        }
        assert!((d.s[15] - 3.0).abs() < 0.05, "{}", d.s[15]);
        // slowly and evenly: no step from one to the next of more than a tenth of σ₂
        assert!(d.s[1..].windows(2).all(|w| w[0] - w[1] < 0.1 * d.s[1]));
        assert_eq!(format!("{:.0}", 100.0 * share(d, 16)), "31");
        let numbers = 16 * (2 * N + 1);
        assert!((numbers as f64 / (N * N) as f64 - 0.5).abs() < 0.01);
    }

    /// "Structure plus noise": three bars above a floor; A₃ is nearer the clean blobs
    /// than A by less than two fifths as much, and no other k is as near; what is left
    /// is mostly the noise. For eight draws of the noise.
    #[test]
    fn cutting_off_the_small_pieces_removes_noise() {
        for seed in 1..=8 {
            let s = run(4, seed);
            let d = &s.dec;
            assert!(d.s[2] > 1.35 * d.s[3] && d.s[3] < 1.2 * d.s[4], "seed {seed}: {:?}", &d.s[..6]);
            let before = distance(&d.a, &s.clean);
            let near: Vec<f64> = (0..=N).map(|k| distance(&d.approx(k), &s.clean)).collect();
            assert!(near[3] < 0.4 * before && near[3] > 0.3 * before, "seed {seed}: {} of {before}", near[3]);
            assert!((0..=N).all(|k| k == 3 || near[k] > near[3]), "seed {seed}");
            let left: Vec<f64> = d.a.iter().zip(d.approx(3)).map(|(x, y)| x - y).collect();
            let noise: Vec<f64> = d.a.iter().zip(&s.clean).map(|(x, c)| x - c).collect();
            let cos = left.iter().zip(&noise).map(|(x, y)| x * y).sum::<f64>() / (size_of(&left) * size_of(&noise));
            assert!(cos > 0.9, "seed {seed}: {cos}");
        }
    }

    /// The size of what the first k pieces leave is √(σ²ₖ₊₁ + …), and no picture of
    /// rank k is closer to A: not the picture in k × k blocks (the rival in the
    /// numbers), not the best one with any other k row patterns, and not the first k
    /// pieces nudged. For every picture, with and without noise.
    #[test]
    fn nothing_of_rank_k_is_closer() {
        let mut rng = Rng::new(3);
        let mut pictures: Vec<Vec<f64>> = PICS.iter().map(|(p, _)| picture(*p)).collect();
        pictures.push(picture(Pic::House).iter().zip(noise(0.15, 4)).map(|(x, e)| x + e).collect());
        for a in &pictures {
            let d = Dec::of(a, N, N);
            let scale = size_of(a);
            for k in [1usize, 2, 3, 5, 8, 16, 32, 63] {
                let best = distance(a, &d.approx(k));
                assert!((best - d.left_after(k)).abs() < 1e-9 * scale, "{best} against {}", d.left_after(k));
                assert!(distance(a, &blocky(a, N, N, k)) >= best - 1e-9 * scale, "k {k}");
                for _ in 0..4 {
                    // the best picture with these row patterns: A Q Qᵀ
                    let mut q: Vec<f64> = (0..N * k).map(|_| rng.range(-1.0, 1.0) as f64).collect();
                    crate::math::linalg::orthonormalise(&mut q, N, k);
                    let aq: Vec<f64> = (0..N * k).map(|at| (0..N).map(|j| a[(at / k) * N + j] * q[j * k + at % k]).sum()).collect();
                    let rival: Vec<f64> = (0..N * N).map(|at| (0..k).map(|c| aq[(at / N) * k + c] * q[(at % N) * k + c]).sum()).collect();
                    assert!(distance(a, &rival) >= best - 1e-9 * scale);
                    // the kept pieces, each nudged a little
                    let mut nudged = vec![0.0; N * N];
                    for p in 0..k.min(d.rank) {
                        let u: Vec<f64> = d.u_col(p).iter().map(|x| x + 0.01 * rng.range(-1.0, 1.0) as f64).collect();
                        let v: Vec<f64> = d.v_col(p).iter().map(|x| x + 0.01 * rng.range(-1.0, 1.0) as f64).collect();
                        for (at, out) in nudged.iter_mut().enumerate() {
                            *out += d.s[p] * u[at / N] * v[at % N];
                        }
                    }
                    assert!(distance(a, &nudged) >= best - 1e-9 * scale);
                }
            }
        }
    }

    /// The rival of the numbers is of rank k at most (its rows come in k kinds).
    #[test]
    fn the_blocky_rival_has_rank_k() {
        for k in [1usize, 3, 7, 8, 30] {
            let b = blocky(&picture(Pic::House), N, N, k);
            let d = Dec::of(&b, N, N);
            assert!(d.nonzero() <= k, "k {k}: {}", d.nonzero());
        }
    }

    /// Drawing paints along the stroke and nowhere else, keeps inside the picture
    /// whatever the pointer does, and the picture is taken apart again from where the
    /// last decomposition ended: the same answer as a fresh one, for a stroke across
    /// the house in fewer sweeps; and the pieces kept stay as they were.
    #[test]
    fn drawing_is_taken_apart_again() {
        let mut s = LowRank::new();
        s.apply(setup(Pic::House, 0.0, 5));
        s.settle();
        let before = s.clean.clone();
        assert!(s.brush(Vec2::new(10.0, 40.0), Vec2::new(30.0, 40.0)));
        let changed: Vec<usize> = (0..N * N).filter(|&at| s.clean[at] != before[at]).collect();
        assert!(changed.iter().all(|&at| (at / N).abs_diff(40) <= 3 && (8..=32).contains(&(at % N))), "{changed:?}");
        assert!(changed.iter().any(|&at| s.clean[at] == 1.0));
        (s.pic, s.build) = (None, None);
        s.touched();
        assert_eq!(s.job.as_ref().map(|j| j.v.clone()), Some(s.dec.directions.clone()));
        s.settle();
        let fresh = Dec::of(&s.a, N, N);
        assert!(s.dec.s.iter().zip(&fresh.s).all(|(x, y)| (x - y).abs() < 1e-10 * fresh.s[0]));
        assert!(s.dec.sweeps < fresh.sweeps, "{} against {}", s.dec.sweeps, fresh.sweeps);
        assert_eq!(s.k, 5);
        // strokes off the edges, through the corners, and from nowhere
        for (a, b) in [(Vec2::new(-50.0, -50.0), Vec2::new(200.0, 300.0)), (Vec2::new(63.9, 0.0), Vec2::new(64.5, 70.0)), (Vec2::splat(f32::NAN), Vec2::ZERO)] {
            s.brush(a, b);
        }
        assert!(s.clean.iter().all(|v| (0.0..=1.0).contains(v)));
        s.touched();
        s.settle();
        let fresh = Dec::of(&s.a, N, N);
        assert!(s.dec.s.iter().zip(&fresh.s).all(|(x, y)| (x - y).abs() < 1e-10 * fresh.s[0]));
    }

    /// The degenerate cases: a blank picture (no pieces at all), k = 0 and k = 64,
    /// "add them one by one" on each, and every layout: nothing panics and every
    /// number is finite.
    #[test]
    fn the_ends_are_fine() {
        let mut s = run(1, 1);
        s.pic = None;
        s.clean = vec![0.0; N * N];
        s.new_picture();
        s.settle();
        assert_eq!((s.dec.rank, s.dec.nonzero(), s.dec.left_after(0)), (0, 0, 0.0));
        assert!(s.dec.s.iter().all(|x| *x == 0.0));
        s.play();
        for _ in 0..600 {
            s.advance(1.0 / 60.0, 1.0 / 60.0);
        }
        assert!(!s.animated() && s.t.is_finite() && s.approx_shown().iter().all(|v| *v == 0.0));
        for (k, pic) in [(0, Pic::House), (N, Pic::House), (N, Pic::Line), (0, Pic::Plaid), (N, Pic::Static)] {
            s.apply(setup(pic, 0.4, k));
            s.settle();
            for _ in 0..1200 {
                s.advance(1.0 / 60.0, 1.0 / 60.0);
            }
            assert!(!s.animated() && s.t == k as f32);
            let shown = s.approx_shown();
            assert!(shown.iter().all(|v| v.is_finite()));
            if k == N {
                assert!(distance(&shown, &s.a) < 1e-10 * size_of(&s.a), "{pic:?}");
            }
            assert!(s.piece_shown() < N && s.misses(k).0.is_finite() && s.dec.left_after(k).is_finite());
            s.play();
            s.advance(0.5, 0.5);
            assert!(s.t > 0.0 && s.t <= N as f32);
        }
        for (w, ppu, inset) in [(5.39, 200.0, 0.0), (2.44, 160.0, 46.0), (6.67, 36.0, 46.0), (0.44, 160.0, 46.0), (2.0, 1.0, 46.0), (8.0, 200.0, 0.0)] {
            let lay = Layout::of(w, ppu, inset);
            let numbers = [lay.size, lay.line, lay.px, lay.side, lay.gap, lay.pside, lay.strip];
            assert!(numbers.iter().all(|v| v.is_finite() && *v > 0.0), "{w} {ppu}");
            assert!(lay.cells.iter().chain([&lay.piece, &lay.chart, &lay.chart_size]).all(|v| v.is_finite()));
        }
    }

    /// "Add them one by one": from none, evenly, to the last piece that is not zero,
    /// whatever k was; all 64 of the line in twelve seconds.
    #[test]
    fn adding_them_one_by_one() {
        for (k, end, seconds) in [(1usize, 22usize, 22.0 / 3.0), (2, 64, 12.0)] {
            let mut s = run(k, 1);
            assert_eq!(s.dec.nonzero(), end);
            s.play();
            assert_eq!((s.t, s.k), (0.0, end));
            let mut frames = 0;
            while s.build.is_some() {
                let before = s.t;
                s.advance(1.0 / 60.0, 1.0 / 60.0);
                assert!(s.t > before && s.t - before <= LowRank::rate(end as f32) / 60.0 + 1e-5);
                frames += 1;
            }
            assert!((frames as f32 / 60.0 - seconds).abs() < 0.05, "{frames} frames");
            assert_eq!(s.t, end as f32);
        }
    }

    /// The words that come with sizes: a power of ten for rounding, and the formula.
    #[test]
    fn sizes_are_written_plainly() {
        assert_eq!(size_text(5.2e-14), r"5 \times 10^{-14}");
        assert_eq!(size_text(0.0), "0");
        assert_eq!(size_text(0.52), "0.52");
        assert_eq!(size_text(18.7), "18.7");
        assert_eq!(size_text(0.0043), "0.0043");
        assert_eq!(size_text(0.05), "0.050");
        assert_eq!(size_text(9.6e-5), r"1 \times 10^{-4}");
        let s = run(1, 1);
        assert_eq!(s.left_formula(5), r"\sqrt{\sigma_{6}^2 + \cdots + \sigma_{64}^2}");
        assert_eq!(s.left_formula(62), r"\sqrt{\sigma_{63}^2 + \sigma_{64}^2}");
        assert_eq!(s.left_formula(63), r"\sqrt{\sigma_{64}^2}");
        assert_eq!(s.left_formula(64), r"\text{nothing}");
        // (and every one of them is markup that parses, with no word in italic)
        let fine = |formula: String| crate::typeset::inspect(&format!("${formula}$")) == Default::default();
        assert!([0, 5, 62, 63, 64].into_iter().all(|k| fine(s.left_formula(k))));
        assert!([5.2e-14, 9.6e-5, 0.0, 18.7].into_iter().all(|v| fine(size_text(v))));
        assert_eq!(percent(2.8), "2.8 %");
        assert_eq!(percent(2.7e-11), "under 0.1 %");
        assert_eq!(percent(94.0), "94 %");
    }

    /// The numbers behind the captions (run with --ignored --nocapture).
    #[test]
    #[ignore]
    fn explore() {
        for (pic, name) in PICS {
            let a = picture(pic);
            let t = std::time::Instant::now();
            let d = Dec::of(&a, N, N);
            let took = t.elapsed().as_secs_f64() * 1000.0;
            let total = size_of(&a);
            let s: Vec<String> = d.s.iter().take(16).map(|x| format!("{x:.2}")).collect();
            let errs: Vec<String> = [1usize, 2, 3, 4, 5, 6, 8, 10, 12, 16, 20, 32].iter().map(|&k| format!("{k}:{:.1}%", 100.0 * d.left_after(k) / total)).collect();
            let rival: Vec<String> = [1usize, 2, 3, 5, 8, 16].iter().map(|&k| format!("{k}:{:.2}/{:.2}", distance(&a, &blocky(&a, N, N, k)), d.left_after(k))).collect();
            println!("{name:<8} {took:.2} ms, {} sweeps, rank {} ({} not zero), size {total:.2}\n   σ {}\n   err {}\n   rival {}", d.sweeps, d.rank, d.nonzero(), s.join(" "), errs.join(" "), rival.join(" "));
            // warm start from a slightly changed picture
            let mut b = a.clone();
            for j in 20..28 {
                b[30 * N + j] = 1.0;
            }
            let mut job = Jacobi::new(&b, N, N, Some(&d.directions));
            let t = std::time::Instant::now();
            job.run(&crate::clock::Budget::new(f64::INFINITY));
            println!("   warm start after a stroke: {} sweeps, {:.2} ms", job.sweeps, t.elapsed().as_secs_f64() * 1000.0);
        }
        let random = Dec::of(&picture(Pic::Static), N, N);
        let mean = picture(Pic::Static).iter().sum::<f64>() / (N * N) as f64;
        let mut first = vec![0.0; N * N];
        random.add_piece(0, 1.0, &mut first);
        println!("random: first piece from the flat mean {:.3} of {:.3}; σ16/σ2 {:.3}, σ64 {:.3}", distance(&first, &vec![mean; N * N]), size_of(&first), random.s[15] / random.s[1], random.s[63]);
        for seed in 1..=8u64 {
            let mut sc = LowRank::new();
            sc.apply(Setup { seed, ..TRY[4].2 });
            sc.settle();
            let d = &sc.dec;
            let nz = noise(0.2, seed);
            let before = distance(&d.a, &sc.clean);
            let after = distance(&d.approx(3), &sc.clean);
            let left: Vec<f64> = d.a.iter().zip(d.approx(3)).map(|(x, y)| x - y).collect();
            let cos = left.iter().zip(&nz).map(|(x, y)| x * y).sum::<f64>() / (size_of(&left) * size_of(&nz));
            let s: Vec<String> = d.s.iter().take(8).map(|x| format!("{x:.2}")).collect();
            let best = (1..=8).map(|k| (distance(&d.approx(k), &sc.clean) / before, k)).fold((9.0, 0), |b, x| if x.0 < b.0 { x } else { b });
            println!("noise seed {seed}: {before:.2} -> {after:.2} ({:.3}), cos {cos:.3}, best k {} ({:.3}), σ {}", after / before, best.1, best.0, s.join(" "));
        }
    }
}

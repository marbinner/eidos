//! Text and formulas made into one egui `Galley`, wrapped to a width, and kept.
//!
//! Each glyph of a formula is laid out by egui on its own (so that egui's font atlas,
//! hinting and sub-pixel placement serve it as they serve all text) and its row is
//! placed where the formula's layout says; words between formulas are laid out by egui
//! in runs, exactly as a label would lay them out; rules and drawn strokes (fraction
//! bars, roots, tall brackets) go in one more row, as a mesh with its own edges
//! smoothed. The result is one `Galley`, painted like any other text: one shape, in
//! whatever colour the caller gives (parts given a colour of their own keep it).
//!
//! What is built is kept, keyed by the text, its sizes and colours and the width: a
//! still text costs a lookup a frame, and so does one that comes back (the panel of a
//! scene opened a second time, a paragraph scrolled out of sight and in again), until
//! more is kept than `KEPT` paragraphs, when the ones longest unused go. A galley
//! points into egui's font atlas, which egui sometimes starts afresh; when it does,
//! everything kept is dropped (egui's own cache of laid-out text is dropped at the same
//! moment, which is how that is noticed).
//!
//! What building asks of egui again and again with the same answer is kept too (see
//! [`Memo`]): the width of a word, a font's measures, a formula's letter laid out.
//! Setting a paragraph is then mostly looking things up: every word used to be laid out
//! by egui twice to be measured, and every letter of every formula once to be placed.

use super::layout::{self, Item};
use super::parse::{self, Node};
use super::{pieces, Face, Piece, MONO_SCALE, TEXT_SCALE};
use egui::epaint::text::{Galley, LayoutJob, PlacedRow, TextFormat};
use egui::epaint::{Mesh, Vertex, WHITE_UV};
use egui::{pos2, vec2, Color32, FontFamily, FontId, Pos2, Rect, Vec2};
use std::collections::HashMap;
use std::hash::{Hash, Hasher};
use std::sync::{Arc, Mutex};

/// A run of text in one look: its size, colour and font, and the room before it.
/// `Color32::PLACEHOLDER` leaves the colour to whoever paints it.
#[derive(Clone, Copy, Debug)]
pub struct Span<'a> {
    pub text: &'a str,
    pub size: f32,
    pub color: Color32,
    pub mono: bool,
    /// Points of space before it (after the span before it).
    pub gap: f32,
}

impl<'a> Span<'a> {
    pub fn new(text: &'a str, size: f32, color: Color32) -> Self {
        Span { text, size, color, mono: false, gap: 0.0 }
    }
}

/// How a text is set.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Mode {
    /// Lines no wider than this many points, broken between words and, inside a
    /// formula, after a relation or an operator.
    Wrap(f32),
    /// One line (or one for each line break in the text): a label.
    Line,
    /// A formula in display style, on lines of its own within this width, centred (the
    /// whole text is the formula, with or without dollars; `\\` starts a new line).
    Display(f32),
}

/// A text set and ready to paint.
#[derive(Debug)]
pub struct Paragraph {
    pub galley: Arc<Galley>,
    /// Where each span was put (a rectangle for each line it is on), to tell what the
    /// pointer is over.
    pub spans: Vec<Vec<Rect>>,
    /// How far the galley's top stands above where a line of plain text at its size
    /// would have its top (0 unless a formula on its first line is taller than the
    /// text's line, which a label holds whole).
    pub over: f32,
    /// How much room a formula on the first line needs above it, and one on the last
    /// line below it, beyond what plain text has: what lines set one under another
    /// must be moved apart by so that neither reaches into the other (0 for a line
    /// whose formulas reach no higher and no deeper than the words' own ink).
    pub rise: f32,
    pub drop: f32,
}

/// A label: `text` at `size` on one line (or one per line break), its colour left to
/// the painter. `None` until the fonts are in place (the first frame).
pub fn label(ctx: &egui::Context, text: &str, size: f32) -> Option<Arc<Paragraph>> {
    paragraph(ctx, &[Span::new(text, size, Color32::PLACEHOLDER)], Mode::Line)
}

/// The size `text` takes as a label at `size`.
pub fn measure(ctx: &egui::Context, text: &str, size: f32) -> Option<Vec2> {
    label(ctx, text, size).map(|p| p.galley.size())
}

/// Spans of text set together: kept from the last pass if they were set then.
pub fn paragraph(ctx: &egui::Context, spans: &[Span], mode: Mode) -> Option<Arc<Paragraph>> {
    let ppp = ctx.pixels_per_point();
    let key = key(spans, mode, ppp);
    let shared = ctx.data_mut(|d| d.get_temp_mut_or_default::<Shared>(egui::Id::unique("typeset")).clone());
    let mut cache = shared.lock().unwrap_or_else(|e| e.into_inner());
    let pass = cache.begin(ctx);
    if !cache.ready {
        return None;
    }
    if let Some(entry) = cache.entries.get_mut(&key) {
        if same(&entry.check, spans) {
            entry.used = pass;
            return Some(entry.value.clone());
        }
    }
    let template = cache.template.clone()?;
    let began = crate::clock::now();
    let memo = cache.memo.for_ratio(ppp);
    let value = Arc::new(ctx.fonts_mut(|fonts| Builder { fonts, ppp, template, memo }.build(spans, mode)));
    BUILT.with(|b| {
        let (n, s) = b.get();
        b.set((n + 1, s + crate::clock::now() - began));
    });
    let check = spans.iter().map(|s| s.text).collect::<Vec<_>>().join("\u{0}").into_boxed_str();
    cache.entries.insert(key, Entry { check, used: pass, value: value.clone() });
    Some(value)
}

thread_local! {
    /// How many paragraphs have been set since [`take_built`] was last asked, and the
    /// seconds that took (for `eidos profile`).
    static BUILT: std::cell::Cell<(u32, f64)> = const { std::cell::Cell::new((0, 0.0)) };
}

/// The paragraphs set afresh (not found kept) since this was last asked, and the
/// milliseconds they took.
pub fn take_built() -> (u32, f64) {
    let (n, s) = BUILT.with(|b| b.replace((0, 0.0)));
    (n, s * 1e3)
}

/// How many paragraphs are being kept (for the tests of what is kept and what goes).
#[cfg(test)]
pub(super) fn kept(ctx: &egui::Context) -> usize {
    let shared = ctx.data_mut(|d| d.get_temp_mut_or_default::<Shared>(egui::Id::unique("typeset")).clone());
    let cache = shared.lock().unwrap_or_else(|e| e.into_inner());
    cache.entries.len()
}

#[cfg(test)]
pub(super) const KEPT_AT_MOST: usize = KEPT;

type Shared = Arc<Mutex<Cache>>;

#[derive(Default)]
struct Cache {
    pass: Option<u64>,
    /// Whether the mathematics fonts are installed yet, and the family that says so.
    ready: bool,
    family: Option<FontFamily>,
    /// A galley laid out by egui and asked for again in every pass: egui's cache gives
    /// back the same one for as long as its font atlas is the same.
    template: Option<Arc<Galley>>,
    entries: HashMap<u64, Entry>,
    memo: Memo,
}

/// As many paragraphs as this are kept whether or not they are being drawn: some
/// thirty scenes' panels. (A label whose text changes every frame, a number being
/// counted, leaves one behind each frame: they are what goes first.)
const KEPT: usize = 1200;

/// What a builder has asked of egui before, and would be told the same again: kept
/// from paragraph to paragraph and from frame to frame, while egui's font atlas and
/// the pixel ratio are what they were.
#[derive(Default)]
struct Memo {
    ppp: f32,
    /// The width of a word, or of a word with the spaces after it, in a font.
    widths: HashMap<(FontFamily, u32), HashMap<Box<str>, f32>>,
    widths_kept: usize,
    /// A font's measures at a size.
    metrics: HashMap<(FontFamily, u32), TextMetrics>,
    /// How far a letter's ink reaches above and below its baseline, at the size
    /// [`LARGE`], in the monospace font or the other: for letters not in `ink::KNOWN`.
    inks: HashMap<(bool, char), [f32; 2]>,
    /// One letter laid out: by the letter, its font and size, its colour, and the
    /// part of a pixel it is moved along by.
    letters: HashMap<(char, FontFamily, u32, Color32, u32), Arc<Galley>>,
}

impl Memo {
    /// No more than this many widths, and this many letters, are kept: beyond that
    /// the table starts again (texts that change every frame would fill it for ever).
    const WIDTHS: usize = 30_000;
    const LETTERS: usize = 8_000;

    /// The memo, emptied first if it was made for another pixel ratio.
    fn for_ratio(&mut self, ppp: f32) -> &mut Memo {
        if self.ppp != ppp {
            *self = Memo { ppp, ..Memo::default() };
        }
        self
    }
}

struct Entry {
    /// The spans' texts, to make sure of a match.
    check: Box<str>,
    /// The last pass it was asked for in.
    used: u64,
    value: Arc<Paragraph>,
}

impl Cache {
    /// Once a pass: drop what was not drawn in the last one, and everything if egui's
    /// font atlas was started afresh. Returns the pass.
    fn begin(&mut self, ctx: &egui::Context) -> u64 {
        let pass = ctx.cumulative_pass_nr();
        if self.pass == Some(pass) {
            return pass;
        }
        self.pass = Some(pass);
        let family = self.family.get_or_insert_with(|| FontFamily::Name(super::Face::MathItalic.family().into()));
        self.ready = ctx.fonts(|f| f.definitions().families.contains_key(family));
        let template = ctx.fonts_mut(|f| f.layout_job(LayoutJob::simple_singleline("x".into(), FontId::proportional(10.0), Color32::PLACEHOLDER)));
        if !self.template.as_ref().is_some_and(|t| Arc::ptr_eq(t, &template)) {
            self.entries.clear();
            self.memo = Memo::default();
        }
        self.template = Some(template);
        if self.entries.len() > KEPT {
            // the half longest unused goes (never what the last pass drew)
            let mut used: Vec<u64> = self.entries.values().map(|e| e.used).collect();
            let middle = used.len() / 2;
            let (_, &mut median, _) = used.select_nth_unstable(middle);
            self.entries.retain(|_, e| e.used + 1 >= pass || e.used > median);
            if self.entries.len() > KEPT {
                // (all of it was drawn in one pass, or nearly: the old rule)
                self.entries.retain(|_, e| e.used + 1 >= pass);
            }
        }
        pass
    }
}

fn key(spans: &[Span], mode: Mode, ppp: f32) -> u64 {
    let mut h = std::collections::hash_map::DefaultHasher::new();
    match mode {
        Mode::Wrap(w) => (0u8, w.to_bits()).hash(&mut h),
        Mode::Line => 1u8.hash(&mut h),
        Mode::Display(w) => (2u8, w.to_bits()).hash(&mut h),
    }
    ppp.to_bits().hash(&mut h);
    for s in spans {
        (s.text, s.size.to_bits(), s.color, s.mono, s.gap.to_bits()).hash(&mut h);
    }
    h.finish()
}

/// Are these the spans whose texts, joined, are `check`?
fn same(check: &str, spans: &[Span]) -> bool {
    let mut rest = check;
    for (i, s) in spans.iter().enumerate() {
        let Some(after) = rest.strip_prefix(s.text) else { return false };
        rest = after;
        if i + 1 < spans.len() {
            let Some(after) = rest.strip_prefix('\u{0}') else { return false };
            rest = after;
        }
    }
    rest.is_empty()
}

/// The size at which a letter is drawn to see how far its ink reaches.
pub(super) const LARGE: f32 = 200.0;

/// How far the ink of `ch` reaches above and below its baseline at the size [`LARGE`],
/// found by having egui draw it (which puts its picture, at that size, in the font
/// atlas: see `ink::KNOWN` for the letters this is not needed for).
pub(super) fn drawn_ink(fonts: &mut egui::epaint::text::FontsView<'_>, ch: char, mono: bool) -> [f32; 2] {
    let font = if mono { FontId::monospace(LARGE) } else { FontId::proportional(LARGE) };
    let mut job = LayoutJob { break_on_newline: false, ..Default::default() };
    job.append(ch.encode_utf8(&mut [0; 4]), 0.0, TextFormat { font_id: font, color: Color32::PLACEHOLDER, ..Default::default() });
    let galley = fonts.layout_job(job);
    let mut ink = [0.0f32; 2];
    for glyph in galley.rows.iter().flat_map(|r| &r.glyphs).filter(|g| !g.uv_rect.is_nothing()) {
        ink[0] = ink[0].max(-glyph.uv_rect.offset.y);
        ink[1] = ink[1].max(glyph.uv_rect.offset.y + glyph.uv_rect.size.y);
    }
    ink
}

/// Where the parts of a paragraph go before they are made into rows.
struct Token {
    span: usize,
    kind: Kind,
    /// Its width, and the space after it if the line goes on, in points.
    width: f32,
    glue: f32,
    /// May a line end after it (and what kind of place for a break is that)? Must it?
    breakable: bool,
    rank: layout::Break,
    newline: bool,
    /// Which formula of the paragraph it is part of, counting from 1 (0: none).
    formula: u32,
}

enum Kind {
    /// A word, and the spaces after it.
    Word(String, String),
    /// A piece of a formula, at `em` points to the em.
    Math(layout::Bx, f32),
    /// Nothing but room (the gap between spans).
    Gap,
}

/// The interface font's measures at one size.
#[derive(Clone, Copy)]
struct TextMetrics {
    /// From the top of a line to the baseline, and from the baseline to the next line.
    ascent: f32,
    descent: f32,
    /// How far the words' own ink reaches above the baseline (ascenders, brackets),
    /// and below it (descenders).
    ink_up: f32,
    ink_down: f32,
}

struct Builder<'f, 'v, 'm> {
    fonts: &'f mut egui::epaint::text::FontsView<'v>,
    ppp: f32,
    template: Arc<Galley>,
    memo: &'m mut Memo,
}

impl Builder<'_, '_, '_> {
    fn font(span: &Span) -> FontId {
        if span.mono {
            FontId::monospace(span.size)
        } else {
            FontId::proportional(span.size)
        }
    }

    fn run(&mut self, text: &str, font: FontId, color: Color32, leading: f32) -> Arc<Galley> {
        let mut job = LayoutJob { break_on_newline: false, ..Default::default() };
        job.append(text, leading, TextFormat { font_id: font, color, ..Default::default() });
        self.fonts.layout_job(job)
    }

    fn width(&mut self, text: &str, font: &FontId) -> f32 {
        let key = (font.family.clone(), font.size.to_bits());
        if let Some(width) = self.memo.widths.get(&key).and_then(|of| of.get(text)) {
            return *width;
        }
        let width = self.run(text, font.clone(), Color32::PLACEHOLDER, 0.0).rect.width();
        if self.memo.widths_kept >= Memo::WIDTHS {
            self.memo.widths.clear();
            self.memo.widths_kept = 0;
        }
        self.memo.widths.entry(key).or_default().insert(text.into(), width);
        self.memo.widths_kept += 1;
        width
    }

    /// Words of a formula (`\text`) as the layout wants them measured: set in the
    /// span's own font at its size, their width and their ink's height and depth, in
    /// ems of the formula (`em` points).
    fn words_box(&mut self, text: &str, bold: bool, span: &Span, em: f32) -> [f32; 3] {
        let width = self.width(text, &Self::font(span)) + if bold { layout::embolden(span.size) } else { 0.0 };
        let [height, depth] = self.ink(text, span);
        [width / em, height / em, depth / em]
    }

    fn text_metrics(&mut self, span: &Span) -> TextMetrics {
        let font = Self::font(span);
        let key = (font.family.clone(), font.size.to_bits());
        if let Some(metrics) = self.memo.metrics.get(&key) {
            return *metrics;
        }
        let galley = self.run("x", font, Color32::PLACEHOLDER, 0.0);
        let (ascent, height) =
            galley.rows.first().map_or((span.size * 0.93, span.size * 1.15), |r| (r.glyphs.first().map_or(span.size * 0.93, |g| g.pos.y), r.size.y));
        let [ink_up, ink_down] = self.ink("dl(gp)", span);
        let metrics = TextMetrics { ascent, descent: height - ascent, ink_up, ink_down };
        self.memo.metrics.insert(key, metrics);
        metrics
    }

    /// How far the ink of `text` in the span's font reaches above and below the
    /// baseline, in points. (Measured large, where the whole pixels a glyph's picture
    /// is rounded out to are a small part of it.)
    fn ink(&mut self, text: &str, span: &Span) -> [f32; 2] {
        // Letter by letter: a letter's picture reaches as far up and down wherever in
        // a word it stands, so the text's reach is the furthest of its letters'. The
        // common letters are in a table (`ink::KNOWN`); another is drawn once, alone,
        // and remembered. (Laid out as words, every letter was drawn at this size once
        // for each part of a pixel it happened to start at, up to four times: most of
        // what egui's font atlas held was these, a hundred times the size of the
        // letters anyone reads, and making them was most of the work of showing a
        // scene's text for the first time.)
        let (mut up, mut down) = (0.0f32, 0.0f32);
        for ch in text.chars() {
            let ink = match super::ink::known(ch, span.mono) {
                Some(ink) => ink,
                None => {
                    let key = (span.mono, ch);
                    match self.memo.inks.get(&key) {
                        Some(ink) => *ink,
                        None => {
                            let ink = drawn_ink(self.fonts, ch, span.mono);
                            self.memo.inks.insert(key, ink);
                            ink
                        }
                    }
                }
            };
            up = up.max(ink[0]);
            down = down.max(ink[1]);
        }
        [up * span.size / LARGE, down * span.size / LARGE]
    }

    /// Split the spans into words and pieces of formulas.
    fn tokens(&mut self, spans: &[Span], mode: Mode) -> Vec<Token> {
        let mut tokens: Vec<Token> = Vec::new();
        let mut formulas = 0;
        for (k, span) in spans.iter().enumerate() {
            if span.gap > 0.0 && k > 0 {
                if let Some(last) = tokens.last_mut() {
                    last.breakable = true;
                }
                tokens.push(Token {
                    span: k,
                    kind: Kind::Gap,
                    width: span.gap,
                    glue: 0.0,
                    breakable: false,
                    rank: layout::Break::None,
                    newline: false,
                    formula: 0,
                });
            }
            let font = Self::font(span);
            let em = span.size * if span.mono { MONO_SCALE } else { TEXT_SCALE };
            let display = matches!(mode, Mode::Display(_));
            // a text that does not parse is set as it is written
            // (a display's lines as its writer broke them, with `\\`)
            let parts: Vec<Piece> = if display {
                let formula = span.text.trim();
                let formula = formula.strip_prefix('$').and_then(|f| f.strip_suffix('$')).unwrap_or(formula);
                super::display_lines(formula).into_iter().map(Piece::Math).collect()
            } else if super::has_math(span.text) {
                pieces(span.text).unwrap_or_else(|_| vec![Piece::Text(span.text.into())])
            } else {
                vec![Piece::Text(span.text.into())]
            };
            let parsed: Option<Vec<Option<parse::Parsed>>> =
                parts.iter().map(|p| if let Piece::Math(f) = p { parse::parse(f).ok().map(Some) } else { Some(None) }).collect();
            let (parts, parsed) = match parsed {
                Some(parsed) => (parts, parsed),
                None => (vec![Piece::Text(span.text.into())], vec![None]),
            };
            let count = parts.len();
            for (n, (part, parsed)) in parts.iter().zip(parsed).enumerate() {
                match (part, parsed) {
                    (_, Some(parsed)) => {
                        formulas += 1;
                        // its words measured first (the layout has no fonts)
                        let mut words: Vec<((&str, bool), [f32; 3])> = Vec::new();
                        parse::walk(&parsed.nodes, &mut |node| {
                            if let Node::Text { text, bold } = node {
                                if !words.iter().any(|(k, _)| *k == (text.as_str(), *bold)) {
                                    words.push(((text.as_str(), *bold), [0.0; 3]));
                                }
                            }
                        });
                        for ((text, bold), size) in &mut words {
                            *size = self.words_box(text, *bold, span, em);
                        }
                        let measure = |text: &str, bold: bool| words.iter().find(|(k, _)| *k == (text, bold)).map_or([0.0; 3], |(_, size)| *size);
                        for chunk in layout::layout(&parsed.nodes, display, &measure) {
                            let (width, glue) = (chunk.bx.width * em, chunk.glue * em);
                            tokens.push(Token {
                                span: k,
                                kind: Kind::Math(chunk.bx, em),
                                width,
                                glue,
                                breakable: chunk.breakable,
                                rank: chunk.kind,
                                newline: false,
                                formula: formulas,
                            });
                        }
                        if display && n + 1 < count {
                            if let Some(last) = tokens.last_mut() {
                                last.newline = true;
                            }
                        }
                    }
                    (Piece::Text(text), None) => self.words(text, k, &font, &mut tokens),
                    (Piece::Math(_), None) => {}
                }
            }
        }
        tokens
    }

    /// Words, each with the spaces after it; a line break ends a line.
    fn words(&mut self, text: &str, span: usize, font: &FontId, tokens: &mut Vec<Token>) {
        for (n, line) in text.split('\n').enumerate() {
            if n > 0 {
                match tokens.last_mut() {
                    Some(last) if !last.newline => last.newline = true,
                    _ => tokens.push(Token {
                        span,
                        kind: Kind::Gap,
                        width: 0.0,
                        glue: 0.0,
                        breakable: false,
                        rank: layout::Break::None,
                        newline: true,
                        formula: 0,
                    }),
                }
            }
            let mut rest = line;
            // spaces at the start belong to what came before
            let lead = rest.len() - rest.trim_start_matches(' ').len();
            if lead > 0 {
                let glue = self.width(&rest[..lead], font);
                match tokens.last_mut() {
                    Some(last) if !last.newline => {
                        last.glue += glue;
                        last.breakable = true;
                    }
                    _ => tokens.push(Token {
                        span,
                        kind: Kind::Gap,
                        width: glue,
                        glue: 0.0,
                        breakable: false,
                        rank: layout::Break::None,
                        newline: false,
                        formula: 0,
                    }),
                }
                rest = &rest[lead..];
            }
            while !rest.is_empty() {
                let end = rest.find(' ').unwrap_or(rest.len());
                let spaces_end = end + (rest[end..].len() - rest[end..].trim_start_matches(' ').len());
                let (word, spaces) = (&rest[..end], &rest[end..spaces_end]);
                let width = self.width(word, font);
                let glue = if spaces.is_empty() { 0.0 } else { self.width(&rest[..spaces_end], font) - width };
                tokens.push(Token {
                    span,
                    kind: Kind::Word(word.to_string(), spaces.to_string()),
                    width,
                    glue,
                    breakable: !spaces.is_empty(),
                    rank: layout::Break::None,
                    newline: false,
                    formula: 0,
                });
                rest = &rest[spaces_end..];
            }
        }
    }

    fn build(mut self, spans: &[Span], mode: Mode) -> Paragraph {
        let tokens = self.tokens(spans, mode);
        let width = match mode {
            Mode::Wrap(w) | Mode::Display(w) => w.max(1.0),
            Mode::Line => f32::INFINITY,
        };
        let lines = match mode {
            Mode::Display(_) => break_display(&tokens, width),
            _ => break_lines(&tokens, width),
        };
        let metrics: Vec<TextMetrics> = spans.iter().map(|s| self.text_metrics(s)).collect();
        let mut rows: Vec<PlacedRow> = Vec::new();
        let mut marks = Marks::default();
        let mut placed: Vec<Vec<Rect>> = vec![Vec::new(); spans.len()];
        let mut top = 0.0f32;
        let mut widest = 0.0f32;
        let (mut over, mut rise, mut drop) = (0.0f32, 0.0f32, 0.0f32);
        for (n, line) in lines.iter().enumerate() {
            let line = &tokens[line.clone()];
            let (first, last) = (n == 0, n + 1 == lines.len());
            // how high the line must be: the text's own line, and more where a formula
            // reaches higher or deeper than the words' own ink does
            let (mut ascent, mut descent, mut ink_up, mut ink_down) = (0.0f32, 0.0f32, 0.0f32, 0.0f32);
            for m in line.iter().map(|t| metrics[t.span]).chain(line.is_empty().then(|| metrics.first().copied()).flatten()) {
                ascent = ascent.max(m.ascent);
                descent = descent.max(m.descent);
                ink_up = ink_up.max(m.ink_up);
                ink_down = ink_down.max(m.ink_down);
            }
            let (mut up, mut down, mut high, mut deep) = (0.0f32, 0.0f32, 0.0f32, 0.0f32);
            for t in line.iter() {
                if let Kind::Math(bx, em) = &t.kind {
                    // (no higher than its own brackets, a formula is as tall as words:
                    // Computer Modern's stand a little above the interface font's)
                    up = up.max(bx.height * em - ink_up.max(0.75 * em));
                    down = down.max(bx.depth * em - ink_down.max(0.25 * em));
                    high = high.max(bx.height * em);
                    deep = deep.max(bx.depth * em);
                    if let Mode::Display(_) = mode {
                        // (the lines of a display have room between them)
                        let room = 0.25 * spans[t.span].size;
                        up = up.max(bx.height * em + room - ascent);
                        down = down.max(bx.depth * em + room - descent);
                    }
                }
            }
            let (up, down) = (up.max(0.0), down.max(0.0));
            if first {
                rise = up;
                // (a label holds all of its first line and no more: its top is the
                // formula's, if that stands out of the text's line)
                over = if mode == Mode::Line { (high - ascent).max(0.0) } else { up };
            }
            if last {
                drop = down;
            }
            // Lines that hold a formula taller than the words are moved apart by just so
            // much that it is as far from the line above as the words' ascenders are
            // (and from the line below as their descenders), and no more; lines without
            // one are spaced as plain text is.
            ascent += if first && mode == Mode::Line { over } else { up };
            descent += if last && mode == Mode::Line { (deep - descent).max(0.0) } else { down };
            let baseline = self.px(top + ascent);
            let length: f32 = line.iter().enumerate().map(|(i, t)| t.width + if i + 1 < line.len() { t.glue } else { 0.0 }).sum();
            // (every line of a display centred)
            let mut x = match mode {
                Mode::Display(w) => ((w - length) * 0.5).max(0.0),
                _ => 0.0,
            };
            let mut i = 0;
            while i < line.len() {
                let t = &line[i];
                let span = &spans[t.span];
                let start = x;
                match &t.kind {
                    Kind::Word(..) => {
                        // consecutive words of one span are one run, as egui sets text
                        let mut text = String::new();
                        let mut j = i;
                        while let Some(Token { kind: Kind::Word(word, spaces), span, .. }) = line.get(j) {
                            if *span != t.span {
                                break;
                            }
                            text.push_str(word);
                            if matches!(line.get(j + 1), Some(Token { kind: Kind::Word(..), span: s, .. }) if *s == t.span) {
                                text.push_str(spaces);
                            }
                            j += 1;
                        }
                        let (x0, leading) = self.split(x);
                        let galley = self.run(&text, Self::font(span), span.color, leading);
                        if let Some(row) = galley.rows.first() {
                            let y = row.glyphs.first().map_or(metrics[t.span].ascent, |g| g.pos.y);
                            rows.push(PlacedRow { pos: pos2(row.pos.x + x0, baseline - y), row: row.row.clone(), ends_with_newline: false });
                        }
                        x += galley.rect.width() - leading;
                        if j < line.len() {
                            x += line[j - 1].glue;
                        }
                        placed[t.span].push(Rect::from_min_max(pos2(start, top), pos2(x, top + ascent + descent)));
                        i = j;
                        continue;
                    }
                    Kind::Math(bx, em) => {
                        for item in &bx.items {
                            self.item(item, *em, pos2(x, baseline), span, &mut rows, &mut marks);
                        }
                    }
                    Kind::Gap => {}
                }
                x += t.width;
                placed[t.span].push(Rect::from_min_max(pos2(start, top), pos2(x, top + ascent + descent)));
                if i + 1 < line.len() {
                    x += t.glue;
                }
                i += 1;
            }
            widest = widest.max(x);
            top = baseline + descent;
        }
        if let Some(row) = marks.row(&self.template) {
            rows.push(row);
        }
        // spans placed on several lines: one rectangle a line
        for rects in &mut placed {
            rects.dedup_by(|b, a| {
                if (a.min.y - b.min.y).abs() < 0.01 {
                    *a = a.union(*b);
                    true
                } else {
                    false
                }
            });
        }
        // (no wider than the room it was set for, though rounding may have put the end of
        // a line a hair past it, or a word longer than the room runs over: a widget
        // must take no more room than it was given, or it widens its panel)
        let width = match mode {
            Mode::Display(w) => w.max(1.0),
            Mode::Wrap(w) => widest.min(w.max(1.0)),
            Mode::Line => widest,
        };
        let size = vec2(width, top);
        let plain: String = spans.iter().map(|s| super::plain(s.text)).collect::<Vec<_>>().join(" ");
        Paragraph { galley: self.galley(rows, size, plain), spans: placed, over, rise, drop }
    }

    /// `x` in points as a whole number of pixels and the rest (in points, to a quarter
    /// of a pixel): a glyph is placed by the first and rendered at the second.
    fn split(&self, x: f32) -> (f32, f32) {
        let px = x * self.ppp;
        if self.ppp >= 1.5 {
            // (on a dense display, the nearest whole pixel: see where the application
            // turns egui's own quarter-pixel placing off)
            return (px.round() / self.ppp, 0.0);
        }
        let whole = px.floor();
        let quarters = ((px - whole) * 4.0).round();
        let (whole, quarters) = if quarters >= 4.0 { (whole + 1.0, 0.0) } else { (whole, quarters) };
        (whole / self.ppp, quarters * 0.25 / self.ppp)
    }

    /// `y` on a whole pixel.
    fn px(&self, y: f32) -> f32 {
        (y * self.ppp).round() / self.ppp
    }

    /// A run of text in `font`, its baseline's left end at `target`.
    fn place_run(&mut self, text: &str, font: FontId, color: Color32, target: Pos2, rows: &mut Vec<PlacedRow>) {
        let (x0, leading) = self.split(target.x);
        let galley = self.run(text, font, color, leading);
        if let Some(row) = galley.rows.first() {
            let y = row.glyphs.first().map_or(0.0, |g| g.pos.y);
            rows.push(PlacedRow { pos: pos2(row.pos.x + x0, self.px(target.y) - y), row: row.row.clone(), ends_with_newline: false });
        }
    }

    /// One thing a formula in `span` draws, its baseline's left end at `origin`.
    fn item(&mut self, item: &Item, em: f32, origin: Pos2, span: &Span, rows: &mut Vec<PlacedRow>, marks: &mut Marks) {
        let text_color = span.color;
        let at = |p: Vec2| pos2(origin.x + p.x * em, origin.y - p.y * em);
        match item {
            Item::Glyph { face, ch, size, at: p, color } => {
                let font = FontId::new(em * size, FontFamily::Name(face.family().into()));
                let (color, target) = (color.unwrap_or(text_color), at(*p));
                // (one letter, in one of a few sizes and colours, a quarter of a pixel
                // along or not: laid out once and found again ever after)
                let (x0, leading) = self.split(target.x);
                let key = (*ch, font.family.clone(), font.size.to_bits(), color, leading.to_bits());
                let galley = match self.memo.letters.get(&key) {
                    Some(galley) => galley.clone(),
                    None => {
                        let galley = self.run(ch.encode_utf8(&mut [0; 4]), font, color, leading);
                        if self.memo.letters.len() >= Memo::LETTERS {
                            self.memo.letters.clear();
                        }
                        self.memo.letters.insert(key, galley.clone());
                        galley
                    }
                };
                if let Some(row) = galley.rows.first() {
                    let y = row.glyphs.first().map_or(0.0, |g| g.pos.y);
                    rows.push(PlacedRow { pos: pos2(row.pos.x + x0, self.px(target.y) - y), row: row.row.clone(), ends_with_newline: false });
                }
            }
            Item::Words { text, size, at: p, color, bold } => {
                let font = FontId { size: span.size * size, ..Self::font(span) };
                let target = at(*p);
                self.place_run(text, font.clone(), color.unwrap_or(text_color), target, rows);
                if *bold {
                    let nudge = vec2(layout::embolden(font.size), 0.0);
                    self.place_run(text, font, color.unwrap_or(text_color), target + nudge, rows);
                }
            }
            Item::Rule { min, max, color } => {
                let (a, b) = (at(*min), at(*max));
                // on whole pixels, and at least one thick: crisp, as TeX's rules are
                let one = 1.0 / self.ppp;
                let (mut y0, mut y1) = (self.px(b.y), self.px(a.y));
                if y1 - y0 < one {
                    let middle = (a.y + b.y) * 0.5;
                    y0 = self.px(middle - one * 0.5);
                    y1 = y0 + one;
                }
                let (mut x0, mut x1) = (self.px(a.x), self.px(b.x));
                if x1 - x0 < one {
                    let middle = (a.x + b.x) * 0.5;
                    x0 = self.px(middle - one * 0.5);
                    x1 = x0 + one;
                }
                marks.rect(Rect::from_min_max(pos2(x0, y0), pos2(x1, y1)), color.unwrap_or(text_color));
            }
            Item::Stroke { points, color } => {
                let points: Vec<(Pos2, f32)> = points.iter().map(|(p, w)| (at(*p), w * em)).collect();
                marks.stroke(&points, color.unwrap_or(text_color), 1.0 / self.ppp);
            }
        }
    }

    /// The rows made into one galley, `size` points large.
    fn galley(&self, rows: Vec<PlacedRow>, size: Vec2, plain: String) -> Arc<Galley> {
        let mut galley: Galley = (*self.template).clone();
        let mut bounds = Rect::NOTHING;
        let (mut vertices, mut indices) = (0, 0);
        for row in &rows {
            bounds = bounds.union(row.visuals.mesh_bounds.translate(row.pos.to_vec2()));
            vertices += row.visuals.mesh.vertices.len();
            indices += row.visuals.mesh.indices.len();
        }
        let font = galley.job.sections.first().map_or(FontId::default(), |s| s.format.font_id.clone());
        galley.job = Arc::new(LayoutJob::simple_singleline(if plain.is_empty() { " ".into() } else { plain }, font, Color32::PLACEHOLDER));
        galley.rows = rows;
        galley.elided = false;
        galley.rect = Rect::from_min_size(Pos2::ZERO, size);
        galley.mesh_bounds = if bounds.is_positive() { bounds } else { Rect::from_min_size(Pos2::ZERO, Vec2::ZERO) };
        galley.num_vertices = vertices;
        galley.num_indices = indices;
        galley.pixels_per_point = self.ppp;
        Arc::new(galley)
    }
}

/// Lines: each a range of tokens, as many on a line as fit, broken at the last place a
/// line may end before one would run over (and run over where there is none). A
/// formula is broken (after a relation or an operator) only if it must be: if it fits
/// on a line of its own, the line ends before it instead.
fn break_lines(tokens: &[Token], width: f32) -> Vec<std::ops::Range<usize>> {
    let inside = |b: usize| tokens[b].formula != 0 && tokens.get(b + 1).is_some_and(|t| t.formula == tokens[b].formula);
    let mut lines = Vec::new();
    let mut start = 0;
    while start < tokens.len() {
        let mut x = 0.0;
        let mut end = tokens.len();
        let (mut last_break, mut last_between) = (None, None);
        for i in start..tokens.len() {
            let left = if i > start { x + tokens[i - 1].glue } else { 0.0 };
            if i > start && left + tokens[i].width > width + 0.01 {
                if let Some(b) = last_break {
                    end = b + 1;
                    // (not inside a formula that would fit on the next line whole)
                    if let (true, Some(q)) = (inside(b), last_between) {
                        let f = tokens[b].formula;
                        let whole: f32 = tokens.iter().filter(|t| t.formula == f).map(|t| t.width + t.glue).sum();
                        if q < b && whole <= width {
                            end = q + 1;
                        }
                    }
                    break;
                }
            }
            x = left + tokens[i].width;
            if tokens[i].newline {
                end = i + 1;
                break;
            }
            if tokens[i].breakable {
                last_break = Some(i);
                if !inside(i) {
                    last_between = Some(i);
                }
            }
        }
        lines.push(start..end);
        start = end;
    }
    if lines.is_empty() {
        lines.push(0..0);
    }
    lines
}

/// A display's lines: where its writer broke it (`\\`), and where a line is still wider
/// than `width`, where a mathematician would break it: between two formulas set side by
/// side, before a relation, or before a binary operator, in that order of preference,
/// as late as the line allows; never so early that the line holds less than a third of
/// the width while another place would do, nor before the first relation of a line
/// written as one (which would leave its left side alone). A line that cannot be broken
/// to fit is broken at its first place, or runs over.
fn break_display(tokens: &[Token], width: f32) -> Vec<std::ops::Range<usize>> {
    use layout::Break;
    let length =
        |from: usize, to: usize| -> f32 { tokens[from..to].iter().enumerate().map(|(i, t)| t.width + if from + i + 1 < to { t.glue } else { 0.0 }).sum() };
    let mut lines = Vec::new();
    let mut start = 0;
    while start < tokens.len() {
        // the line as written
        let end = (start..tokens.len()).find(|&i| tokens[i].newline).map_or(tokens.len(), |i| i + 1);
        let first_relation = (start..end).find(|&b| tokens[b].rank == Break::Relation);
        let mut from = start;
        while from < end {
            if length(from, end) <= width + 0.01 {
                lines.push(from..end);
                break;
            }
            let fitting: Vec<usize> = (from..end - 1).filter(|&b| tokens[b].breakable && length(from, b + 1) <= width + 0.01).collect();
            let fair = |b: &usize| Some(*b) != first_relation && length(from, b + 1) >= width / 3.0;
            let chosen = fitting
                .iter()
                .copied()
                .filter(fair)
                .max_by_key(|&b| (tokens[b].rank, b))
                .or_else(|| fitting.last().copied())
                .or_else(|| (from..end - 1).find(|&b| tokens[b].breakable));
            match chosen {
                Some(b) => {
                    lines.push(from..b + 1);
                    from = b + 1;
                }
                None => {
                    lines.push(from..end);
                    break;
                }
            }
        }
        start = end;
    }
    if lines.is_empty() {
        lines.push(0..0);
    }
    lines
}

/// Rules and strokes, as one mesh: solid triangles first (the ones a shadow's colour
/// replaces), then the soft edges of the strokes, fading to nothing over a pixel.
#[derive(Default)]
struct Marks {
    solid: Vec<Vertex>,
    edges: Vec<Vertex>,
    /// Indices; those of edge vertices are counted from `EDGE`, and moved past the
    /// solid ones at the end.
    indices: Vec<u32>,
}

const EDGE: u32 = 1 << 30;

impl Marks {
    fn vertex(&mut self, pos: Pos2, color: Color32, solid: bool) -> u32 {
        let v = Vertex { pos, uv: WHITE_UV, color };
        if solid {
            self.solid.push(v);
            self.solid.len() as u32 - 1
        } else {
            self.edges.push(v);
            EDGE + self.edges.len() as u32 - 1
        }
    }

    fn quad(&mut self, [a, b, c, d]: [u32; 4]) {
        self.indices.extend_from_slice(&[a, b, c, a, c, d]);
    }

    fn rect(&mut self, r: Rect, color: Color32) {
        let corners = [r.left_top(), r.right_top(), r.right_bottom(), r.left_bottom()].map(|p| self.vertex(p, color, true));
        self.quad(corners);
    }

    /// A stroke through `points`, each with its width there, its edges smoothed over
    /// `pixel` points.
    fn stroke(&mut self, points: &[(Pos2, f32)], color: Color32, pixel: f32) {
        if points.len() < 2 || points.iter().any(|(p, w)| !p.is_finite() || !w.is_finite()) {
            return;
        }
        let n = points.len();
        let tangent = |i: usize| {
            let (a, b) = (points[i.saturating_sub(1)].0, points[(i + 1).min(n - 1)].0);
            let d = b - a;
            if d.length() > 1e-6 {
                d / d.length()
            } else {
                Vec2::X
            }
        };
        let mut sides = Vec::with_capacity(n);
        for i in 0..n {
            let (p, w) = points[i];
            let t = tangent(i);
            let normal = vec2(-t.y, t.x);
            // (the solid part is a pixel narrower than the stroke, and its edges a pixel wide)
            let inner = (w * 0.5 - pixel * 0.5).max(0.0);
            let outer = w.max(pixel) * 0.5 + pixel * 0.5;
            let ends = if i == 0 {
                -t * pixel * 0.5
            } else if i == n - 1 {
                t * pixel * 0.5
            } else {
                Vec2::ZERO
            };
            let ids = [
                self.vertex(p + normal * outer + ends, Color32::TRANSPARENT, false),
                self.vertex(p + normal * inner, color, true),
                self.vertex(p - normal * inner, color, true),
                self.vertex(p - normal * outer + ends, Color32::TRANSPARENT, false),
            ];
            sides.push(ids);
        }
        for i in 0..n - 1 {
            let (a, b) = (sides[i], sides[i + 1]);
            self.quad([a[1], b[1], b[2], a[2]]);
            self.quad([a[0], b[0], b[1], a[1]]);
            self.quad([a[2], b[2], b[3], a[3]]);
        }
        // the soft ends
        for (s, along) in [(sides[0], -tangent(0)), (sides[n - 1], tangent(n - 1))] {
            let p1 = self.edge_pos(s[1]) + along * pixel;
            let p2 = self.edge_pos(s[2]) + along * pixel;
            let e1 = self.vertex(p1, Color32::TRANSPARENT, false);
            let e2 = self.vertex(p2, Color32::TRANSPARENT, false);
            self.quad([s[1], e1, e2, s[2]]);
        }
    }

    fn edge_pos(&self, id: u32) -> Pos2 {
        if id >= EDGE {
            self.edges[(id - EDGE) as usize].pos
        } else {
            self.solid[id as usize].pos
        }
    }

    /// The marks as a row of the galley, made from the template's (whose glyphs are
    /// dropped): in galley coordinates, at the galley's corner.
    fn row(self, template: &Galley) -> Option<PlacedRow> {
        if self.solid.is_empty() {
            return None;
        }
        let first = template.rows.first()?;
        let solid = self.solid.len() as u32;
        let indices = self.indices.iter().map(|&i| if i >= EDGE { i - EDGE + solid } else { i }).collect();
        let mut mesh = Mesh { indices, vertices: self.solid, ..Default::default() };
        mesh.vertices.extend(self.edges);
        let mut bounds = Rect::NOTHING;
        for v in &mesh.vertices {
            bounds.extend_with(v.pos);
        }
        let mut row = (*first.row).clone();
        row.glyphs.clear();
        row.size = bounds.max.to_vec2().max(Vec2::ZERO);
        row.visuals.mesh = mesh;
        row.visuals.mesh_bounds = bounds;
        row.visuals.glyph_index_start = 0;
        row.visuals.glyph_vertex_range = 0..solid as usize;
        Some(PlacedRow { pos: Pos2::ZERO, row: Arc::new(row), ends_with_newline: false })
    }
}

/// The double-struck form of a capital letter (`\mathbb`): ℝ for R.
fn blackboard(c: char) -> char {
    match c {
        'C' => 'ℂ',
        'H' => 'ℍ',
        'N' => 'ℕ',
        'P' => 'ℙ',
        'Q' => 'ℚ',
        'R' => 'ℝ',
        'Z' => 'ℤ',
        _ => char::from_u32(0x1D538 + (c as u32 - 'A' as u32)).unwrap_or(c),
    }
}

/// A formula's tree written out in plain Unicode (see [`super::plain`]).
pub fn plain_nodes(nodes: &[Node], out: &mut String) {
    for node in nodes {
        match node {
            // (a blackboard capital is written as one: ℝ, not R)
            Node::Sym { ch, face: Face::Ams, .. } if ch.is_ascii_uppercase() => out.push(blackboard(*ch)),
            Node::Sym { ch, .. } => out.push(match ch {
                '\u{E131}' => 'ı',
                '\u{E237}' => 'ȷ',
                c => *c,
            }),
            Node::Neg(ch) => {
                out.push(*ch);
                out.push('\u{338}');
            }
            Node::BigOp { ch, .. } => out.push(*ch),
            Node::Func { name, .. } => out.push_str(name),
            Node::Text { text, .. } => out.push_str(text),
            Node::Group(list) | Node::Color { body: list, .. } | Node::Style { body: list, .. } | Node::Line { body: list, .. } => plain_nodes(list, out),
            Node::Scripts { base, sup, sub } => {
                plain_nodes(std::slice::from_ref(base), out);
                for (mark, list) in [('_', sub), ('^', sup)] {
                    if let Some(list) = list {
                        out.push(mark);
                        let mut inner = String::new();
                        plain_nodes(list, &mut inner);
                        if inner.chars().count() > 1 {
                            out.push('(');
                            out.push_str(&inner);
                            out.push(')');
                        } else {
                            out.push_str(&inner);
                        }
                    }
                }
            }
            Node::Frac { num, den, .. } => {
                out.push('(');
                plain_nodes(num, out);
                out.push_str(")/(");
                plain_nodes(den, out);
                out.push(')');
            }
            Node::Sqrt { body, .. } => {
                out.push_str("√(");
                plain_nodes(body, out);
                out.push(')');
            }
            Node::Delimited { left, right, body } => {
                out.extend(*left);
                plain_nodes(body, out);
                out.extend(*right);
            }
            Node::Middle(d) | Node::Big { delim: d, .. } => out.extend(*d),
            Node::Matrix { rows, .. } => {
                out.push('[');
                for (i, row) in rows.iter().enumerate() {
                    if i > 0 {
                        out.push_str("; ");
                    }
                    for (j, cell) in row.iter().enumerate() {
                        if j > 0 {
                            out.push_str(", ");
                        }
                        plain_nodes(cell, out);
                    }
                }
                out.push(']');
            }
            Node::Accent { body, .. } => plain_nodes(body, out),
            Node::Space(mu) => {
                if *mu > 2.0 {
                    out.push(' ');
                }
            }
        }
    }
}

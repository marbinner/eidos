//! Small widgets shared by the scene control panels.

use crate::gfx::color::{pal, Color};
use crate::math::expr::Expr;
use crate::typeset::{self, has_math, Mode, Span};
use egui::RichText;

// Every function here that puts words in a panel takes text with mathematics between
// dollar signs (see `typeset`); text without a dollar sign is shown exactly as it
// always was.

/// Spans of text with mathematics in them, set as one paragraph as wide as the panel.
/// `None` only before the fonts are in place (the first frame), when the caller shows
/// the text as it is.
fn typeset_spans(ui: &mut egui::Ui, spans: &[Span]) -> Option<(egui::Response, std::sync::Arc<typeset::Paragraph>)> {
    let set = typeset::paragraph(ui.ctx(), spans, Mode::Wrap(ui.available_width()))?;
    let response = ui.add(egui::Label::new(egui::WidgetText::Galley(set.galley.clone())));
    Some((response, set))
}

/// A text as a widget's label: typeset on one line if it has mathematics in it, as
/// `RichText` (exactly as before) if not. `Color32::PLACEHOLDER` (`None`) leaves the
/// colour to the widget. The text is noted as shown (`typeset::seen`), so that the lint
/// and the scene check read it, whatever widget it is given to.
pub fn label_text(ui: &egui::Ui, text: &str, size: f32, color: Option<Color>, mono: bool) -> egui::WidgetText {
    typeset::seen(text);
    widget_text(ui, text, size, color, mono)
}

/// The same without noting the text: for the application's own texts (the scene list,
/// the links on to the next scene), which are no scene's, and for a text made from one
/// that has been noted already (a link's arrow added, a heading in capitals).
pub(crate) fn widget_text(ui: &egui::Ui, text: &str, size: f32, color: Option<Color>, mono: bool) -> egui::WidgetText {
    if has_math(text) {
        let span = Span { text, size, color: color.map_or(egui::Color32::PLACEHOLDER, Color::egui), mono, gap: 0.0 };
        if let Some(set) = typeset::paragraph(ui.ctx(), &[span], Mode::Line) {
            return egui::WidgetText::Galley(set.galley.clone());
        }
    }
    let mut rich = RichText::new(text).size(size);
    if let Some(color) = color {
        rich = rich.color(color.egui());
    }
    if mono {
        rich = rich.monospace();
    }
    rich.into()
}

/// Words that wrap to the panel's width, at `size` in `color`, mathematics and all.
pub fn words(ui: &mut egui::Ui, text: &str, size: f32, color: Color) -> egui::Response {
    typeset::seen(text);
    words_unnoted(ui, text, size, color)
}

/// The same without noting the text as shown (for the application's own words).
pub(crate) fn words_unnoted(ui: &mut egui::Ui, text: &str, size: f32, color: Color) -> egui::Response {
    let Some(key) = later::key(ui, &[text], size) else { return set_words(ui, text, size, color) };
    if later::wait(key) {
        return room_for(ui, text.chars().count(), size);
    }
    let response = set_words(ui, text, size, color);
    later::set(key);
    response
}

/// Room for a text of `letters` letters that is to be set in a later frame, by a
/// guess at its height (see [`later`]); and another frame asked for, to set it in.
fn room_for(ui: &mut egui::Ui, letters: usize, size: f32) -> egui::Response {
    let width = ui.available_width().max(1.0);
    let lines = (letters as f32 * size * 0.5 / width).ceil().max(1.0);
    ui.ctx().request_repaint();
    ui.allocate_exact_size(egui::vec2(width, lines * size * 1.3), egui::Sense::hover()).1
}

/// `set` now, or room for it and `set` in a later frame, if it is a paragraph below
/// what is in sight that this frame has no time left for (see [`later`]). `texts` are
/// what it says, by which it is known.
fn now_or_later(ui: &mut egui::Ui, texts: &[&str], size: f32, set: impl FnOnce(&mut egui::Ui)) {
    let Some(key) = later::key(ui, texts, size) else { return set(ui) };
    if later::wait(key) {
        room_for(ui, texts.iter().map(|t| t.chars().count()).sum(), size);
        return;
    }
    set(ui);
    later::set(key);
}

fn set_words(ui: &mut egui::Ui, text: &str, size: f32, color: Color) -> egui::Response {
    if has_math(text) {
        if let Some((response, _)) = typeset_spans(ui, &[Span::new(text, size, color.egui())]) {
            return response;
        }
    }
    ui.label(RichText::new(text).size(size).color(color.egui()))
}

/// Text below what is in sight is not all set in the frame that first meets it.
///
/// A panel's text is laid out whole, in sight or not, to know how long the panel is,
/// and the first time that is most of the work of opening a scene: 10 to 20 ms where
/// a frame has 16. So a paragraph that starts below the bottom of what is showing, and
/// has not been set before at this width, is set at once only while the frame is
/// young; after that it is given room by a guess and set in a later frame (one more
/// each frame at the least, and as many as fit in a frame's first milliseconds). What
/// is in sight, or above it, is always set at once: nothing a visitor is looking at
/// moves. Only where the application asks for it ([`defer_unseen_text`]): for the tools
/// that make pictures and check scenes, and in the tests, every text is set in the
/// frame that first meets it.
mod later {
    use std::cell::RefCell;
    use std::collections::HashSet;
    use std::hash::{Hash, Hasher};

    /// A frame older than this many seconds sets no more text that is out of sight
    /// (beyond one paragraph, so that it is all set in the end whatever frames cost).
    const YOUNG: f64 = 0.004;

    #[derive(Default)]
    struct State {
        /// Whether any text waits at all (see [`on`]).
        on: bool,
        /// When this frame began, and how many unseen paragraphs it has set.
        began: f64,
        set_now: u32,
        /// Paragraphs set before, by text, size and width.
        done: HashSet<u64>,
    }

    thread_local! {
        static STATE: RefCell<State> = RefCell::new(State::default());
    }

    /// Let text wait, or not: it does only where the application asks for it (in a
    /// window or a browser), so that the tools and the tests, which lay a panel out
    /// and look at it, get every text in the frame that first meets it.
    pub fn on(on: bool) {
        STATE.with(|s| s.borrow_mut().on = on);
    }

    /// A frame begins (the application says so before it lays anything out).
    pub fn frame_begins() {
        STATE.with(|s| {
            let mut s = s.borrow_mut();
            s.began = crate::clock::now();
            s.set_now = 0;
            if s.done.len() > 60_000 {
                s.done.clear();
            }
        });
    }

    /// What a paragraph about to be laid out at the cursor is known by, if it starts
    /// below what is in sight (and so may wait); `None` if it must be set now.
    pub fn key(ui: &egui::Ui, texts: &[&str], size: f32) -> Option<u64> {
        if !STATE.with(|s| s.borrow().on) || ui.cursor().min.y <= ui.clip_rect().max.y {
            return None;
        }
        let mut h = std::collections::hash_map::DefaultHasher::new();
        (texts, size.to_bits(), ui.available_width().to_bits(), ui.ctx().pixels_per_point().to_bits()).hash(&mut h);
        Some(h.finish())
    }

    /// Whether the paragraph should wait for a later frame.
    pub fn wait(key: u64) -> bool {
        STATE.with(|s| {
            let s = s.borrow();
            !s.done.contains(&key) && s.set_now >= 1 && crate::clock::now() - s.began > YOUNG
        })
    }

    /// The paragraph has been set.
    pub fn set(key: u64) {
        STATE.with(|s| {
            let mut s = s.borrow_mut();
            if s.done.insert(key) {
                s.set_now += 1;
            }
        });
    }
}

/// A frame begins: see [`later`].
pub(crate) fn frame_begins() {
    later::frame_begins();
}

/// Whether a paragraph below what is in sight may be set a frame or two after the
/// frame that first meets it (see `later`): asked for where a visitor is watching,
/// in a window or a browser. Not otherwise: every text is then set at once.
pub fn defer_unseen_text(on: bool) {
    later::on(on);
}

#[cfg(test)]
mod later_tests {
    use super::later;

    /// A paragraph out of sight waits only when the frame is no longer young, when
    /// the frame has set one such already, and when it has never been set: once set,
    /// it is never made to wait again (or its room would go back to a guess).
    #[test]
    fn unseen_text_waits_for_a_young_frame_once_and_never_again() {
        later::on(true);
        later::frame_begins();
        // a young frame sets what it meets
        assert!(!later::wait(1));
        later::set(1);
        assert!(!later::wait(2));
        std::thread::sleep(std::time::Duration::from_millis(8));
        // an old one that has set one does not start another
        assert!(later::wait(2));
        // but what has been set before is set at once, whatever the frame's age
        assert!(!later::wait(1));
        // the next frame, old before it meets any: one paragraph all the same
        later::frame_begins();
        std::thread::sleep(std::time::Duration::from_millis(8));
        assert!(!later::wait(2));
        later::set(2);
        assert!(later::wait(3));
        assert!(!later::wait(2));
    }
}

/// A link made of a typeset label: underlined under its baseline when the pointer is
/// on it (egui's own underline would run under every glyph's own row).
pub fn link_label(ui: &mut egui::Ui, text: egui::WidgetText) -> egui::Response {
    let egui::WidgetText::Galley(galley) = text else { return ui.link(text) };
    let response = ui.add(egui::Label::new(egui::WidgetText::Galley(galley.clone())).sense(egui::Sense::click()));
    if response.hovered() {
        let y = response.rect.bottom() - 2.0;
        ui.painter().hline(response.rect.x_range(), y, egui::Stroke::new(1.0, pal::LINK.egui()));
        ui.set_cursor_icon(egui::CursorIcon::PointingHand);
    }
    response
}

/// Text shown while the pointer is over a widget, mathematics and all. (Noted as shown
/// only while it is.)
pub fn hover(response: egui::Response, text: &str) -> egui::Response {
    if has_math(text) {
        response.on_hover_ui(|ui| {
            words(ui, text, 12.5, pal::TEXT);
        })
    } else {
        if response.hovered() {
            typeset::seen(text);
        }
        response.on_hover_text(text)
    }
}

/// A checkbox, its label taking mathematics like every other text here (egui's own sets
/// its label as plain text).
pub fn checkbox(ui: &mut egui::Ui, on: &mut bool, text: &str) -> egui::Response {
    let text = label_text(ui, text, 12.5, None, false);
    ui.checkbox(on, text)
}

/// Spans with mathematics in them as one wrapped paragraph, the span numbered `link`
/// (if any) a link. Says whether the link was clicked; `None` before the fonts are in
/// place, when the caller shows the text as it is.
pub fn linked_words(ui: &mut egui::Ui, spans: &[Span], link: Option<usize>) -> Option<bool> {
    for span in spans {
        typeset::seen(span.text);
    }
    let links: Vec<usize> = link.into_iter().collect();
    linked_spans(ui, spans, &links).map(|clicked| clicked.is_some())
}

/// Spans set as one wrapped paragraph, those numbered in `links` links (each wrapping
/// as words do, underlined on every line it is on while the pointer is on it). Which
/// was clicked, if any; `None` before the fonts are in place.
fn linked_spans(ui: &mut egui::Ui, spans: &[Span], links: &[usize]) -> Option<Option<usize>> {
    let (response, set) = typeset_spans(ui, spans)?;
    let mut clicked = None;
    for &k in links {
        // (one place to point at for each line the link is on, all one link)
        let parts: Vec<egui::Response> = set
            .spans
            .get(k)
            .into_iter()
            .flatten()
            .enumerate()
            .map(|(i, rect)| ui.interact(rect.translate(response.rect.min.to_vec2()), response.id.with(("link", k, i)), egui::Sense::click()))
            .collect();
        if parts.iter().any(egui::Response::hovered) {
            for part in &parts {
                ui.painter().hline(part.rect.x_range(), part.rect.bottom() - 2.0, egui::Stroke::new(1.0, pal::LINK.egui()));
            }
            ui.set_cursor_icon(egui::CursorIcon::PointingHand);
        }
        if parts.iter().any(egui::Response::clicked) {
            clicked = Some(k);
        }
    }
    Some(clicked)
}

pub fn section(ui: &mut egui::Ui, title: &str) {
    ui.add_space(8.0);
    typeset::seen(title);
    if has_math(title) {
        // (the words in capitals, the mathematics as it is)
        let upper: String = match typeset::pieces(title) {
            Ok(parts) => parts
                .iter()
                .map(|p| match p {
                    typeset::Piece::Text(t) => t.to_uppercase().replace('$', "\\$"),
                    typeset::Piece::Math(m) => format!("${m}$"),
                })
                .collect(),
            Err(_) => title.to_string(),
        };
        ui.label(widget_text(ui, &upper, 10.5, Some(pal::MUTED), false));
        return;
    }
    ui.label(RichText::new(title.to_uppercase()).size(10.5).strong().color(pal::MUTED.egui()));
}

/// Explanatory text.
pub fn note(ui: &mut egui::Ui, text: &str) {
    words(ui, text, 12.0, pal::DIM);
}

/// A formula on a line of its own, in display style (fractions full size, limits above
/// and below), centred in the panel. `\\` starts a new line; a line wider than the panel
/// is broken where a mathematician would break it (between two formulas set side by
/// side, before a relation, before an operator), every line centred. The text is the
/// formula, with or without dollars around it.
pub fn equation(ui: &mut egui::Ui, text: &str) {
    equation_at(ui, text, 12.5);
}

/// The same at another size, and without noting the text as shown (for the
/// application's own texts: a scene's primer).
pub(crate) fn equation_at(ui: &mut egui::Ui, text: &str, size: f32) {
    let noted = size == 12.5;
    let formula = text.trim();
    let formula = formula.strip_prefix('$').and_then(|f| f.strip_suffix('$')).unwrap_or(formula);
    // (each line as a formula of its own: `\\` is no formula's part)
    if noted {
        for line in typeset::display_lines(formula) {
            typeset::seen(&format!("${line}$"));
        }
    }
    let span = Span::new(formula, size, pal::TEXT.egui());
    ui.add_space(3.0);
    match typeset::paragraph(ui.ctx(), &[span], Mode::Display(ui.available_width())) {
        Some(set) => {
            ui.add(egui::Label::new(egui::WidgetText::Galley(set.galley.clone())));
        }
        None => {
            let lines: Vec<String> = typeset::display_lines(formula).iter().map(|line| typeset::plain(&format!("${line}$"))).collect();
            ui.label(RichText::new(lines.join("\n")).size(size).color(pal::TEXT.egui()));
        }
    }
    ui.add_space(3.0);
}

fn slider_row(ui: &mut egui::Ui, label: &str, value: impl FnOnce(&mut egui::Ui) -> bool, bar: impl FnOnce(&mut egui::Ui) -> bool) -> bool {
    // name on the left, editable number on the right, full-width track underneath:
    // nothing depends on the label's length, so rows never overflow the panel
    let mut changed = false;
    ui.horizontal(|ui| {
        ui.label(label_text(ui, label, 12.5, Some(pal::DIM), false));
        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
            changed |= value(ui);
        });
    });
    ui.scope(|ui| {
        ui.spacing_mut().slider_width = ui.available_width();
        changed |= bar(ui);
    });
    changed
}

/// A slider row shows one value in two widgets (a number to type and a bar to
/// drag), each working on a copy: whichever copy changed is the new value.
fn take_changed<T: PartialEq + Copy>(v: &mut T, typed: T, dragged: T) {
    if typed != *v {
        *v = typed;
    } else if dragged != *v {
        *v = dragged;
    }
}

fn slider_f32(ui: &mut egui::Ui, label: &str, v: &mut f32, range: std::ops::RangeInclusive<f32>, log: bool) -> bool {
    let (lo, hi) = (*range.start(), *range.end());
    let speed = if log { (v.abs() * 0.02).max(1e-7) } else { (hi - lo) * 0.004 };
    // no more digits than the range gives meaning to (a value that is being animated
    // would otherwise show five)
    let decimals = if hi - lo >= 2.0 { 2 } else { 3 };
    let mut a = *v;
    let mut b = *v;
    let changed = slider_row(
        ui,
        label,
        |ui| {
            let value = egui::DragValue::new(&mut a).speed(speed).range(lo..=hi);
            // a logarithmic slider spans decades: show what is needed and no padding
            ui.add(if log { value.custom_formatter(|v, _| tidy(v)) } else { value.max_decimals(decimals) }).changed()
        },
        |ui| ui.add(egui::Slider::new(&mut b, range).logarithmic(log).show_value(false)).changed(),
    );
    take_changed(v, a, b);
    changed
}

/// A number with up to three significant digits after its leading zeros, and no
/// trailing zeros: 0.02, 0.00125, 1.5, 20, 0.
fn tidy(v: f64) -> String {
    if v == 0.0 {
        return "0".to_string();
    }
    let digits = (2 - v.abs().log10().floor() as i32).clamp(0, 7) as usize;
    let text = format!("{v:.digits$}");
    if text.contains('.') {
        text.trim_end_matches('0').trim_end_matches('.').to_string()
    } else {
        text
    }
}

pub fn slider(ui: &mut egui::Ui, label: &str, v: &mut f32, range: std::ops::RangeInclusive<f32>) -> bool {
    slider_f32(ui, label, v, range, false)
}

pub fn slider_log(ui: &mut egui::Ui, label: &str, v: &mut f32, range: std::ops::RangeInclusive<f32>) -> bool {
    slider_f32(ui, label, v, range, true)
}

/// A logarithmic slider that can also be exactly zero: the left end is 0, and the
/// scale runs from `smallest` to `largest` after it. For "off or any strength" settings.
pub fn slider_log_or_zero(ui: &mut egui::Ui, label: &str, v: &mut f32, smallest: f32, largest: f32) -> bool {
    let speed = (v.abs() * 0.02).max(smallest * 0.2);
    let mut a = *v;
    let mut b = *v;
    let changed = slider_row(
        ui,
        label,
        |ui| ui.add(egui::DragValue::new(&mut a).speed(speed).range(0.0..=largest).custom_formatter(|v, _| tidy(v))).changed(),
        |ui| ui.add(egui::Slider::new(&mut b, 0.0..=largest).logarithmic(true).smallest_positive(smallest as f64).show_value(false)).changed(),
    );
    take_changed(v, a, b);
    changed
}

fn slider_usize(ui: &mut egui::Ui, label: &str, v: &mut usize, range: std::ops::RangeInclusive<usize>, log: bool) -> bool {
    let (lo, hi) = (*range.start(), *range.end());
    let speed = if log { (*v as f32 * 0.02).max(0.2) } else { ((hi - lo) as f32 * 0.004).max(0.05) };
    let mut a = *v;
    let mut b = *v;
    let changed = slider_row(
        ui,
        label,
        |ui| ui.add(egui::DragValue::new(&mut a).speed(speed).range(lo..=hi)).changed(),
        |ui| ui.add(egui::Slider::new(&mut b, range).logarithmic(log).show_value(false)).changed(),
    );
    take_changed(v, a, b);
    changed
}

pub fn slider_int(ui: &mut egui::Ui, label: &str, v: &mut usize, range: std::ops::RangeInclusive<usize>) -> bool {
    slider_usize(ui, label, v, range, false)
}

pub fn slider_int_log(ui: &mut egui::Ui, label: &str, v: &mut usize, range: std::ops::RangeInclusive<usize>) -> bool {
    slider_usize(ui, label, v, range, true)
}

/// One row of numbers in a panel: an optional coloured mark, the label, and the
/// value in monospace at the right-hand edge.
fn readout_row(ui: &mut egui::Ui, mark: Option<RichText>, label: egui::WidgetText, value: String) {
    // (in a row of other things, the read-out is a column of its own: its number may
    // need a line of its own under its name, and in a row that would be beside it,
    // over the name)
    if ui.layout().is_horizontal() {
        ui.vertical(|ui| readout_lines(ui, mark, label, value));
    } else {
        readout_lines(ui, mark, label, value);
    }
}

fn readout_lines(ui: &mut egui::Ui, mark: Option<RichText>, label: egui::WidgetText, value: String) {
    let tall = has_math(&value);
    let value = label_text(ui, &value, 12.5, Some(pal::TEXT), true);
    // (a name and a number too long to share the line, in a narrow panel, would be
    // written over each other: the number then goes on a line of its own)
    let size =
        |ui: &egui::Ui, text: &egui::WidgetText| text.clone().into_galley(ui, Some(egui::TextWrapMode::Extend), f32::INFINITY, egui::TextStyle::Body).size();
    let needed = size(ui, &label).x + size(ui, &value).x + if mark.is_some() { 30.0 } else { 14.0 };
    let shared = needed <= ui.available_width();
    let height = size(ui, &value).y;
    ui.horizontal(|ui| {
        // (a value with a formula in it may be taller than a line, a matrix or a
        // fraction: the row is made as tall as it first, so that the name and the mark
        // are centred on it, and the row reads as one line)
        if shared && tall {
            ui.set_min_height(height);
        }
        if let Some(mark) = mark {
            ui.label(mark);
        }
        ui.label(label);
        if shared {
            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                ui.label(value.clone());
            });
        }
    });
    if !shared {
        ui.with_layout(egui::Layout::right_to_left(egui::Align::TOP), |ui| {
            ui.label(value);
        });
    }
}

/// "label   value" on one line, value in monospace.
pub fn readout(ui: &mut egui::Ui, label: &str, value: impl Into<String>) {
    let label = label_text(ui, label, 12.5, Some(pal::DIM), false);
    readout_row(ui, None, label, value.into());
}

/// "■ name   value", the name in the colour its curve or marker has in the view: for
/// lists where the reader must match each row to a colour.
pub fn readout_series(ui: &mut egui::Ui, name: &str, value: impl Into<String>, color: Color) {
    typeset::seen(name);
    let name = if has_math(name) { label_text(ui, name, 13.0, Some(color), false) } else { RichText::new(name).size(13.0).strong().color(color.egui()).into() };
    readout_row(ui, Some(RichText::new("■").size(14.0).color(color.egui())), name, value.into());
}

/// "● label   value": a read-out that belongs to something drawn in that colour.
pub fn readout_colored(ui: &mut egui::Ui, label: &str, value: impl Into<String>, color: Color) {
    let label = label_text(ui, label, 12.5, Some(pal::DIM), false);
    readout_row(ui, Some(RichText::new("●").size(11.0).color(color.egui())), label, value.into());
}

/// A wrapping row of buttons; returns the index of the one clicked.
pub fn buttons(ui: &mut egui::Ui, labels: &[&str]) -> Option<usize> {
    let mut hit = None;
    ui.horizontal_wrapped(|ui| {
        for (i, l) in labels.iter().enumerate() {
            if ui.button(label_text(ui, l, 12.5, None, false)).clicked() {
                hit = Some(i);
            }
        }
    });
    hit
}

/// A wrapping row of mutually exclusive choices. Returns true when the choice changed.
pub fn choice(ui: &mut egui::Ui, labels: &[&str], current: &mut usize) -> bool {
    let mut changed = false;
    ui.horizontal_wrapped(|ui| {
        for (i, l) in labels.iter().enumerate() {
            if ui.selectable_label(*current == i, label_text(ui, l, 12.5, None, false)).clicked() && *current != i {
                *current = i;
                changed = true;
            }
        }
    });
    changed
}

/// A text box holding an expression. It keeps the last expression that compiled, so
/// the picture never disappears while you are half-way through typing.
pub struct ExprField {
    pub text: String,
    pub expr: Expr,
    pub error: Option<String>,
    vars: &'static [&'static str],
}

impl ExprField {
    pub fn new(text: &str, vars: &'static [&'static str]) -> Self {
        let expr = Expr::parse(text, vars).unwrap_or_else(|_| Expr::parse("0", vars).unwrap());
        ExprField { text: text.to_string(), expr, error: None, vars }
    }

    /// Replace the text programmatically (presets).
    pub fn set(&mut self, text: &str) {
        self.text = text.to_string();
        self.compile();
    }

    fn compile(&mut self) {
        match Expr::parse(&self.text, self.vars) {
            Ok(e) => {
                self.expr = e;
                self.error = None;
            }
            Err(e) => self.error = Some(e),
        }
    }

    /// Returns true when a new valid expression was entered. The label before the box
    /// ("f(x) =") is typeset as the formula it is ([`field_label`]); what is typed in
    /// the box is something to type, and stays as it is.
    pub fn ui(&mut self, ui: &mut egui::Ui, label: &str) -> bool {
        let mut changed = false;
        ui.horizontal(|ui| {
            let label = field_label(label);
            if label.is_empty() {
                ui.label(RichText::new(label).monospace().size(13.0).color(pal::DIM.egui()));
            } else {
                ui.label(label_text(ui, &label, 13.0, Some(pal::DIM), false));
            }
            let edit = egui::TextEdit::singleline(&mut self.text).font(egui::FontId::monospace(13.0)).desired_width(f32::INFINITY);
            if ui.add(edit).changed() {
                self.compile();
                changed = self.error.is_none();
            }
        });
        if let Some(e) = &self.error {
            ui.label(RichText::new(e).size(11.5).color(pal::RED.egui()));
        }
        changed
    }
}

/// The label of a formula box as a formula: "f(x) =" is `$f(x) =$`, "dx/dt =" is
/// `$\mathrm{d}x/\mathrm{d}t =$`, and a word, "loss =", is `$\text{loss} =$`. A label
/// with mathematics in it already is left as it is, and so is an empty one.
pub fn field_label(label: &str) -> String {
    if has_math(label) || label.trim().is_empty() {
        return label.to_string();
    }
    let (name, equals) = match label.trim().strip_suffix('=') {
        Some(name) => (name.trim(), " ="),
        None => (label.trim(), ""),
    };
    let letters = |s: &str| !s.is_empty() && s.chars().all(|c| c.is_ascii_alphabetic());
    let name = match name.split_once('/') {
        // a derivative: dx/dt
        Some((top, bottom)) if top.len() == 2 && bottom.len() == 2 && top.starts_with('d') && bottom.starts_with('d') && letters(top) && letters(bottom) => {
            format!(r"\mathrm{{d}}{}/\mathrm{{d}}{}", &top[1..], &bottom[1..])
        }
        // a word, not a product of letters
        _ if letters(name) && name.len() > 1 => format!(r"\text{{{name}}}"),
        _ => name.to_string(),
    };
    let formula = format!("${name}{equals}$");
    // (anything it cannot read stays as it was written)
    if typeset::inspect(&formula).error.is_some() {
        return label.to_string();
    }
    formula
}

// ----- links between scenes -----------------------------------------------------------

thread_local! {
    static WANTED: std::cell::Cell<Option<&'static str>> = const { std::cell::Cell::new(None) };
}

/// What a scene's panel was seen to contain of the parts every panel should have
/// (README, "The panel"), while this is being collected.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct PanelParts {
    /// How many "things to try" ([`experiment`]).
    pub experiments: usize,
    /// Whether it has a colour key ([`key`]).
    pub key: bool,
    /// Every scene id its "see also" lists name ([`see_also`]).
    pub links: Vec<&'static str>,
}

thread_local! {
    static PARTS: std::cell::RefCell<Option<PanelParts>> = const { std::cell::RefCell::new(None) };
}

/// Note something about the panel being drawn, if anyone is collecting.
fn noting(note: impl FnOnce(&mut PanelParts)) {
    PARTS.with(|p| {
        if let Some(parts) = p.borrow_mut().as_mut() {
            note(parts);
        }
    });
}

/// Start (or stop) collecting what panels contain, and return what was collected so
/// far: how the scene check finds a panel without its standard parts, and links that
/// lead nowhere.
pub fn collect_panel(on: bool) -> PanelParts {
    PARTS.with(|p| p.replace(on.then(PanelParts::default)).unwrap_or_default())
}

thread_local! {
    static PRESS: std::cell::Cell<Option<usize>> = const { std::cell::Cell::new(None) };
}

/// Have the `k`-th "thing to try" of the next panel drawn (counting from 0) answer
/// as if it had been clicked: for the tools that try every experiment without
/// knowing where its button is. Only while the panel is being collected
/// ([`collect_panel`]), which is what counts them.
pub fn press_experiment(k: usize) {
    PRESS.with(|p| p.set(Some(k)));
}

/// The same, for the links alone.
pub fn collect_links(on: bool) -> Vec<&'static str> {
    collect_panel(on).links
}

/// The scene a link asked for since this was last called (the application polls it).
pub fn take_wanted_scene() -> Option<&'static str> {
    WANTED.with(|w| w.take())
}

/// "See also" links to other scenes: `(scene id, what the link says)`. Clicking one
/// switches scene. A link to a scene that does not exist does nothing.
pub fn see_also(ui: &mut egui::Ui, links: &[(&'static str, &str)]) {
    noting(|parts| {
        for (id, _) in links {
            if !parts.links.contains(id) {
                parts.links.push(id);
            }
        }
    });
    // (every link's words are noted as shown, the hidden ones' too: a mistake in one
    // is then found now, not on the day the scene it leads to is written)
    for (_, text) in links {
        typeset::seen(text);
    }
    // (a link to a scene that is planned and not yet written is shown when that scene
    // is there: a course's scenes link to each other and are not all written at once)
    let there: Vec<(&'static str, &str)> = links.iter().copied().filter(|(id, _)| crate::scenes::written(id)).collect();
    let links = &there[..];
    if links.is_empty() {
        return;
    }
    section(ui, "See also");
    // (with mathematics in them, the links are one typeset paragraph, each wrapping as
    // words do: a formula set as a label on one line, a long link ran past the panel)
    if links.iter().any(|(_, text)| has_math(text)) {
        let texts: Vec<String> = links.iter().map(|(_, text)| format!("→ {text}")).collect();
        let spans: Vec<Span> =
            texts.iter().enumerate().map(|(i, text)| Span { gap: if i > 0 { 8.0 } else { 0.0 }, ..Span::new(text, 12.5, pal::LINK.egui()) }).collect();
        let all: Vec<usize> = (0..links.len()).collect();
        for (_, text) in links {
            typeset::seen(text);
        }
        if let Some(clicked) = linked_spans(ui, &spans, &all) {
            if let Some(&(id, _)) = clicked.and_then(|k| links.get(k)) {
                WANTED.with(|w| w.set(Some(id)));
            }
            return;
        }
    }
    ui.horizontal_wrapped(|ui| {
        for (id, text) in links {
            typeset::seen(text);
            let text = format!("→ {text}");
            if link_label(ui, widget_text(ui, &text, 12.5, Some(pal::LINK), false)).clicked() {
                WANTED.with(|w| w.set(Some(id)));
            }
        }
    });
}

/// One entry of a "things to try" list: a button that sets the experiment up, with a
/// line underneath saying what to look for. Returns true when the button is clicked.
pub fn experiment(ui: &mut egui::Ui, label: &str, look_for: &str) -> bool {
    let mut nth = None;
    noting(|parts| {
        nth = Some(parts.experiments);
        parts.experiments += 1;
    });
    // (a tool may ask for this one to be pressed: see `press_experiment`)
    let pressed = nth.is_some() && PRESS.with(|p| p.get() == nth && p.take().is_some());
    typeset::seen(label);
    let clicked = ui.button(widget_text(ui, &format!("▶ {label}"), 12.5, None, false)).clicked() || pressed;
    words(ui, look_for, 11.5, pal::DIM);
    ui.add_space(3.0);
    clicked
}

/// A colour key for the picture: one "■ meaning" row per colour.
pub fn key(ui: &mut egui::Ui, entries: &[(Color, &str)]) {
    noting(|parts| parts.key = true);
    for (color, meaning) in entries {
        typeset::seen(meaning);
        now_or_later(ui, &["■", meaning], 12.0, |ui| {
            if has_math(meaning)
                && typeset_spans(ui, &[Span::new("■", 13.0, color.egui()), Span { gap: 8.0, ..Span::new(meaning, 12.0, pal::DIM.egui()) }]).is_some()
            {
                return;
            }
            ui.horizontal_wrapped(|ui| {
                ui.label(RichText::new("■").size(13.0).color(color.egui()));
                ui.label(RichText::new(*meaning).size(12.0).color(pal::DIM.egui()));
            });
        });
    }
}

/// A term being defined: the term in bright text, its meaning after it.
pub fn term(ui: &mut egui::Ui, term: &str, meaning: &str) {
    typeset::seen(term);
    typeset::seen(meaning);
    now_or_later(ui, &[term, meaning], 12.0, |ui| {
        if (has_math(term) || has_math(meaning))
            && typeset_spans(ui, &[Span::new(term, 12.0, pal::TEXT.egui()), Span { gap: 5.0, ..Span::new(meaning, 12.0, pal::DIM.egui()) }]).is_some()
        {
            return;
        }
        ui.horizontal_wrapped(|ui| {
            ui.spacing_mut().item_spacing.x = 5.0;
            ui.label(RichText::new(term).size(12.0).strong().color(pal::TEXT.egui()));
            ui.label(RichText::new(meaning).size(12.0).color(pal::DIM.egui()));
        });
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The scene check finds links that lead nowhere by collecting what "see also"
    /// lists name; nothing is collected unless it asks.
    /// A context with the fonts in place, and a pass of it.
    fn with_fonts() -> egui::Context {
        let ctx = egui::Context::default();
        let _ = ctx.run_ui(egui::RawInput::default(), |ui| crate::app::apply_style(ui.ctx()));
        ctx
    }

    /// The texts a pass painted, with where.
    fn painted(ctx: &egui::Context, add: impl Fn(&mut egui::Ui)) -> Vec<(String, egui::Rect)> {
        let mut out = egui::FullOutput::default();
        for _ in 0..3 {
            out = ctx.run_ui(egui::RawInput::default(), |ui| add(ui));
        }
        out.shapes
            .iter()
            .filter_map(|c| if let egui::Shape::Text(t) = &c.shape { Some((t.galley.text().to_string(), t.visual_bounding_rect())) } else { None })
            .collect()
    }

    /// A read-out whose value is taller than a line (a matrix) has its name centred on
    /// the value, not at the top of the row.
    #[test]
    fn a_readouts_name_is_centred_on_a_tall_value() {
        let ctx = with_fonts();
        let shapes = painted(&ctx, |ui| readout(ui, "the matrix", r"$\begin{bmatrix} 1 \\ 2 \\ 3 \end{bmatrix}$"));
        let find = |what: &str| shapes.iter().find(|(t, _)| t.contains(what)).map(|(_, r)| *r).unwrap_or_else(|| panic!("{what} in {shapes:?}"));
        let (name, value) = (find("the matrix"), find("1"));
        assert!(value.height() > 2.0 * name.height(), "{value:?} {name:?}");
        assert!((name.center().y - value.center().y).abs() < 3.0, "the name at {:?}, the value at {:?}", name.center(), value.center());
    }

    /// A "see also" link with a formula in it wraps to the panel's width, as words do,
    /// rather than running past it on one line.
    #[test]
    fn a_long_link_wraps_to_the_panel() {
        let ctx = with_fonts();
        let long = r"SVD, step by step: $A = U \Sigma V^\top$ as a rotation, a stretch and another rotation";
        for width in [120.0f32, 180.0, 260.0] {
            let shapes = painted(&ctx, |ui| {
                ui.allocate_ui(egui::vec2(width, 800.0), |ui| see_also(ui, &[("svd", long), ("lsq", "the same problem as a projection")]));
            });
            let right = shapes.iter().map(|(_, r)| r.right()).fold(0.0f32, f32::max);
            assert!(right <= width + 1.0, "in {width}: text reaches {right}");
        }
    }

    /// The labels of the formula boxes are written as formulas.
    #[test]
    fn formula_boxes_have_typeset_labels() {
        for (plain, formula) in [
            ("f(x) =", "$f(x) =$"),
            ("z =", "$z =$"),
            ("loss =", r"$\text{loss} =$"),
            ("target =", r"$\text{target} =$"),
            ("dx/dt =", r"$\mathrm{d}x/\mathrm{d}t =$"),
            ("dy/dt =", r"$\mathrm{d}y/\mathrm{d}t =$"),
            ("", ""),
            ("$g(x) =$", "$g(x) =$"),
        ] {
            assert_eq!(field_label(plain), formula);
            assert!(typeset::inspect(&field_label(plain)).error.is_none());
        }
    }

    /// A read-out in a row of other things keeps its number off its name when the
    /// number needs a line of its own.
    #[test]
    fn a_readout_in_a_row_keeps_its_number_off_its_name() {
        let ctx = with_fonts();
        let shapes = painted(&ctx, |ui| {
            ui.allocate_ui(egui::vec2(160.0, 400.0), |ui| {
                ui.horizontal(|ui| readout(ui, "a name long enough to need a line", "123456.789"));
            });
        });
        let find = |what: &str| shapes.iter().find(|(t, _)| t.contains(what)).map(|(_, r)| *r).unwrap_or_else(|| panic!("{what} in {shapes:?}"));
        let (name, value) = (find("a name"), find("123456"));
        assert!(!name.intersects(value), "the name at {name:?}, the value at {value:?}");
    }

    #[test]
    fn links_are_collected_only_when_asked() {
        let show = || {
            let _ = egui::Context::default().run_ui(egui::RawInput::default(), |ui| {
                see_also(ui, &[("surface", "the surface"), ("nowhere", "a dead end")]);
                see_also(ui, &[("surface", "again")]);
            });
        };
        show();
        assert!(collect_links(true).is_empty(), "nothing was being collected");
        show();
        assert_eq!(collect_links(false), ["surface", "nowhere"]);
        show();
        assert!(collect_links(false).is_empty());
    }

    /// The scene check also asks whether a panel has its standard parts.
    #[test]
    fn the_parts_of_a_panel_are_noted() {
        collect_panel(true);
        let _ = egui::Context::default().run_ui(egui::RawInput::default(), |ui| {
            let _ = experiment(ui, "one", "look here");
            let _ = experiment(ui, "two", "and here");
            see_also(ui, &[("surface", "the surface")]);
        });
        assert_eq!(collect_panel(true), PanelParts { experiments: 2, key: false, links: vec!["surface"] });
        // a tool presses the second experiment, once
        press_experiment(1);
        let mut pressed = Vec::new();
        for _ in 0..2 {
            let _ = egui::Context::default().run_ui(egui::RawInput::default(), |ui| pressed.push([experiment(ui, "one", ""), experiment(ui, "two", "")]));
            collect_panel(true);
        }
        assert_eq!(pressed, [[false, true], [false, false]]);
        let _ = egui::Context::default().run_ui(egui::RawInput::default(), |ui| key(ui, &[(pal::CYAN, "a curve")]));
        assert_eq!(collect_panel(false), PanelParts { experiments: 0, key: true, links: vec![] });
    }
}

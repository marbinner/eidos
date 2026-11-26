//! Probabilists' Hermite polynomials: the family, why they are orthogonal under the
//! Gaussian, expansions in them, and what correlation does to them.

use crate::math::hermite::{coefficients, gauss_hermite, gauss_mean, h, h_all, he, he_formula, phi, zeros};
use crate::scenes::prelude::*;

// The scenes of this topic, in the order they appear in the catalogue.
crate::scenes::topic!(family, orth, expand, noise, dirs);

fn factorial(n: usize) -> f64 {
    (1..=n).map(|k| k as f64).product()
}

/// (2k - 1)!! = E[X^(2k)].
fn double_factorial_odd(k: usize) -> f64 {
    (1..=k).map(|j| (2 * j - 1) as f64).product()
}

fn degree_color(k: usize, of: usize) -> Color {
    Cmap::Turbo.color(0.1 + 0.8 * k as f32 / of.max(1) as f32)
}

/// The monic polynomial of degree `k`, as markup (to go between dollars): He₄.
fn he_name(k: usize) -> String {
    format!(r"\mathrm{{He}}_{{{k}}}")
}

// (a checkbox whose label may have mathematics in it: egui's own sets its label as
// plain text)
use crate::ui::checkbox;

/// The Gaussian weight as a soft filled hump standing on `base`.
fn weight_hump(f: &mut Frame, base: f32, scale: f32, x_scale: f32, x_off: f32, color: Color) {
    let n = 160;
    let hump = (0..=n).map(|i| -5.5 + 11.0 * i as f32 / n as f32).map(|x| Vec2::new(x * x_scale + x_off, base + phi(x as f64) as f32 * scale));
    fill_under(f, hump, base, color.a(0.16));
    f.curve(-5.5, 5.5, n, 1.3, color.a(0.7), |x| Vec3::new(x * x_scale + x_off, base + phi(x as f64) as f32 * scale, 0.0));
}

/// The words of a text, each formula between dollars one word however many spaces
/// it has, so that a line is never broken inside one.
fn words_of(text: &str) -> Vec<&str> {
    let mut words = Vec::new();
    let (mut start, mut inside, mut escaped) = (0, false, false);
    for (i, c) in text.char_indices() {
        match c {
            _ if escaped => escaped = false,
            '\\' => escaped = true,
            '$' => inside = !inside,
            ' ' if !inside => {
                words.push(&text[start..i]);
                start = i + 1;
            }
            _ => {}
        }
    }
    words.push(&text[start..]);
    words
}

/// The weight's name beside its hump, up and to the left of `at` (logical pixels), on a
/// plate: every curve passes through here. On a phone it is as wide as the left half
/// of the view, and is moved right as far as it must be to stay in it.
fn weight_caption(f: &mut Frame, at: Vec2) {
    let text = r"Gaussian weight $\varphi(x)$, not to scale";
    let right = (at.x - 10.0).max(f.text_size(text, 12.0).x + 10.0);
    f.caption_px(Vec2::new(right, at.y - 4.0), text, 12.0, pal::PURPLE, Align2::RIGHT_BOTTOM);
}

/// A caption's parts, joined by `gap` on one line where `room` (logical pixels) allows,
/// otherwise on as many lines as it takes (on a phone held upright a single line runs
/// off the screen); a part too long for a line of its own is broken between words.
fn caption_lines(f: &Frame, parts: &[&str], gap: &str, size: f32, room: f32) -> Vec<String> {
    // (a text with a formula in it as the frame measures it, which keeps the layout of
    // what it measured: worked out afresh in every frame, the formulas cost more than
    // the rest of the scene)
    let fits = |text: &str| if crate::typeset::has_math(text) { f.text_size(text, size).x } else { text_width(text, size) } <= room;
    let mut lines: Vec<String> = Vec::new();
    for part in parts {
        if let Some(last) = lines.last_mut() {
            let joined = format!("{last}{gap}{part}");
            if fits(&joined) {
                *last = joined;
                continue;
            }
        }
        let mut line = String::new();
        for word in words_of(part) {
            if !line.is_empty() && !fits(&format!("{line} {word}")) {
                lines.push(std::mem::take(&mut line));
            }
            if !line.is_empty() {
                line.push(' ');
            }
            line.push_str(word);
        }
        lines.push(line);
    }
    lines
}

/// The key at the foot of the picture, on as many lines as the view's width needs.
fn bottom_caption(f: &mut Frame, parts: &[&str], size: f32) {
    let view = f.view_size();
    let lines = caption_lines(f, parts, "      ", size, view.x - 36.0);
    let count = lines.len();
    for (k, line) in lines.into_iter().enumerate() {
        let up = (count - 1 - k) as f32 * (size + 7.0);
        f.caption_px(Vec2::new(18.0, view.y - 18.0 - up), line, size, pal::DIM, Align2::LEFT_BOTTOM);
    }
}

/// Highest degree the expansion scenes carry.
const EXP_NMAX: usize = 40;

const EXPANSION_PRESETS: [(&str, &str); 10] = [
    ("ReLU", "relu(x)"),
    ("tanh", "tanh(x)"),
    ("sign", "sign(x)"),
    ("$|x|$", "abs(x)"),
    ("sigmoid", "sigmoid(2x)"),
    ("sin", "sin(2x)"),
    ("$x^3$", "x^3"),
    ("exp", "exp(x/2)"),
    ("bump", "exp(-2 (x-1)^2)"),
    ("ReLU$^2$", "relu(x)^2"),
];

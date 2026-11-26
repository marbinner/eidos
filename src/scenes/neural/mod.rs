//! Neural networks: what they compute, how they learn, and why depth helps.

use crate::math::nn::{Act, Loss, Mlp, Optim};
use crate::scenes::prelude::*;

// The scenes of this topic, in the order they appear in the catalogue.
crate::scenes::topic!(mlp, race, particles, signal, fold, fit1d, landscape);

// (Superposition is written here, among the networks it trains, and listed under a
// later heading, interpretability, which it opens: the catalogue puts the
// backpropagation course between the two.)
mod superposition;

/// The scene of this topic that opens the interpretability section.
pub fn opening_interpretability() -> Vec<SceneDef> {
    vec![superposition::DEF]
}

// Not scenes: the parts the playground is made of, and the models two of the
// scenes draw.
mod arith;
mod plane;
mod session;
mod table;
mod toy;
mod training;
mod wide;

pub use mlp::arithmetic;
pub use session::benchmark as training_benchmark;
pub use training::start_afresh;

/// The service that trains networks in the background.
pub fn trainer() -> Box<dyn crate::background::Service> {
    Box::<session::Trainer>::default()
}

pub const CLASS: [Color; 2] = [Color::hex(0x4F9DFF), Color::hex(0xFF8A3D)];

/// The scenes that sort points in the plane draw the square from -DATA_R to DATA_R.
const DATA_R: f32 = 1.15;

/// The picture of a network's answer over that square: `n` by `n` values, coloured
/// from blue (negative: the first class) to orange, with white lines where the
/// caller's choice of `style.contour` puts them. The caller sets `style.range`, the
/// contours and `style.model` (where the square goes), and fills in the values.
fn answer_field(gfx: &Gfx, n: usize) -> Surface {
    let mut field = Surface::new(gfx, n as u32, n as u32);
    field.fill(|u, v| [u - 0.5, v - 0.5, 0.0, 0.0]);
    field.style.lit = 0.0;
    field.style.cmap = Some(Cmap::CoolWarm);
    field.style.contour_alpha = 0.9;
    // (with a contour spacing of 1: one line only, the boundary, where the answer is zero)
    field.style.contour_reach = 0.5;
    field.style.line_color = pal::WHITE;
    field
}

/// A loss as a read-out or a label shows it: typeset, so that a small or a large one is
/// a power of ten (2.5 × 10⁻⁴), or as it is when it is not a number (a run that has
/// diverged).
fn loss_number(v: f32) -> String {
    if v.is_finite() {
        format!("${}$", fmt(v))
    } else {
        fmt(v)
    }
}

/// The share of the points a network with one output puts on the right side of zero.
fn share_right(outputs: &[f32], y: &[f32]) -> f32 {
    let hits = outputs.iter().zip(y).filter(|(o, t)| (**o > 0.0) == (**t > 0.5)).count();
    hits as f32 / y.len().max(1) as f32
}

/// Decides when pictures that are costly to recompute are brought up to date: once
/// the step count has moved on, but only after the rest their own cost calls for
/// (see [`crate::clock::Rest`]), so that a slow recomputation never sets the frame
/// rate.
#[derive(Default)]
struct Redraw {
    /// The step count the pictures show (none yet, to begin with).
    shown: Option<u64>,
    rest: crate::clock::Rest,
}

impl Redraw {
    /// Should the pictures be recomputed this frame? Yes if `force` says something
    /// other than training has changed them.
    fn due(&mut self, steps: u64, force: bool) -> bool {
        if force || self.shown.is_none() {
            return true;
        }
        if self.shown == Some(steps) {
            return false;
        }
        self.rest.over()
    }

    /// They are up to date again; recomputing them took `seconds`.
    fn done(&mut self, steps: u64, seconds: f64) {
        self.shown = Some(steps);
        self.rest = crate::clock::Rest::after(seconds);
    }

    /// Start afresh (a new network: whatever is shown is of the old one).
    fn reset(&mut self) {
        *self = Redraw::default();
    }
}

/// Labelled 2D points in roughly [-1, 1]².
#[derive(Clone)]
pub struct Data {
    pub x: Vec<f32>,
    pub y: Vec<f32>,
}

impl Data {
    pub fn len(&self) -> usize {
        self.y.len()
    }

    pub fn point(&self, i: usize) -> Vec2 {
        Vec2::new(self.x[2 * i], self.x[2 * i + 1])
    }
}

pub const DATASETS: [&str; 13] =
    ["spirals", "circles", "moons", "xor", "blobs", "line", "stripes", "checkerboard", "rings", "sectors", "wave", "four blobs", "pinwheel"];

/// For the datasets defined by a rule: which class a point belongs to.
fn rule(kind: usize, p: Vec2) -> bool {
    let parity = |v: f32| (v.floor() as i32).rem_euclid(2) == 1;
    match kind {
        5 => 0.8 * p.x + p.y > 0.1,
        6 => parity((p.x + p.y) * 1.8),
        7 => parity(p.x * 1.6) != parity(p.y * 1.6),
        8 => parity(p.length() * 3.4),
        9 => parity((p.y.atan2(p.x) + PI) / (PI / 3.0)),
        10 => p.y > 0.45 * (3.6 * p.x).sin(),
        _ => parity((p.y.atan2(p.x) + PI + 2.6 * p.length()) / (PI / 3.0)),
    }
}

pub fn make_data(kind: usize, n: usize, noise: f32, rng: &mut Rng) -> Data {
    let mut d = Data { x: Vec::with_capacity(n * 2), y: Vec::with_capacity(n) };
    if kind >= 5 && kind != 11 {
        // rule-based sets: points anywhere in the square, labelled by where they fall,
        // then jittered so that noise blurs the boundary
        for _ in 0..n {
            let p = Vec2::new(rng.range(-0.95, 0.95), rng.range(-0.95, 0.95));
            let label = rule(kind, p);
            let q = p + Vec2::new(rng.normal(), rng.normal()) * noise * 0.12;
            d.x.push(q.x);
            d.x.push(q.y);
            d.y.push(if label { 1.0 } else { 0.0 });
        }
        return d;
    }
    for i in 0..n {
        let c = (i % 2) as f32;
        let t = rng.f32();
        let p = match kind {
            0 => {
                let r = 0.1 + 0.85 * t;
                let a = 1.75 * t * TAU + c * PI;
                Vec2::new(r * a.cos(), r * a.sin())
            }
            1 => Vec2::from_angle(t * TAU) * if c > 0.5 { 0.85 } else { 0.32 * rng.f32().sqrt() + 0.05 },
            2 => {
                let a = t * PI;
                if c > 0.5 {
                    Vec2::new(a.cos() * 0.62 - 0.3, a.sin() * 0.62 - 0.22)
                } else {
                    Vec2::new(0.3 - a.cos() * 0.62, 0.3 - a.sin() * 0.62)
                }
            }
            3 => {
                let q = Vec2::new(rng.range(0.12, 0.9), rng.range(0.12, 0.9));
                let s = if rng.bool(0.5) { 1.0 } else { -1.0 };
                if c > 0.5 {
                    Vec2::new(q.x * s, q.y * s)
                } else {
                    Vec2::new(q.x * s, -q.y * s)
                }
            }
            11 => {
                // four clusters, diagonal pairs sharing a class
                let side = if rng.bool(0.5) { 1.0 } else { -1.0 };
                let centre = if c > 0.5 { Vec2::new(0.5 * side, 0.5 * side) } else { Vec2::new(0.5 * side, -0.5 * side) };
                centre + Vec2::new(rng.normal(), rng.normal()) * 0.16
            }
            _ => Vec2::new(rng.normal(), rng.normal()) * 0.2 + if c > 0.5 { Vec2::new(0.45, 0.4) } else { Vec2::new(-0.45, -0.4) },
        };
        let p = p + Vec2::new(rng.normal(), rng.normal()) * noise * 0.12;
        d.x.push(p.x);
        d.x.push(p.y);
        d.y.push(c);
    }
    d
}

/// Layer widths, input first: `hidden` layers of `width` neurons between the inputs
/// and the outputs.
pub fn layer_sizes(inputs: usize, hidden: usize, width: usize, outputs: usize) -> Vec<usize> {
    std::iter::once(inputs).chain(std::iter::repeat_n(width, hidden)).chain(std::iter::once(outputs)).collect()
}

/// Row-major grid of input points covering [-r, r]².
fn input_grid(n: usize, r: f32) -> Vec<f32> {
    let mut g = Vec::with_capacity(n * n * 2);
    for j in 0..n {
        for i in 0..n {
            g.push(-r + 2.0 * r * i as f32 / (n - 1) as f32);
            g.push(-r + 2.0 * r * j as f32 / (n - 1) as f32);
        }
    }
    g
}

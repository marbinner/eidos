//! Signal propagation: what depth does to the correlation between two inputs.
//!
//! In a wide network with random weights, the pre-activations of a layer are Gaussian.
//! Two inputs then have just two numbers that matter: the common length `q` of their
//! pre-activation vectors (their mean square) and the correlation `c` between them.
//! One layer with weight scale `sw` and bias scale `sb` maps them to
//!
//! ```text
//!     q' = sw² E[f(u)²] + sb²
//!     c' = (sw² E[f(u) f(v)] + sb²) / q'
//! ```
//!
//! where `(u, v)` are Gaussian with variance `q` and correlation `c`, and `f` is the
//! activation. Written in Hermite polynomials, `f(√q z) = Σ aₙ hₙ(z)`, the expectation
//! of the product is `Σ aₙ² cⁿ`: the map on correlations is a power series whose
//! coefficients are the activation's squared Hermite coefficients.

use super::*;
use crate::clock::Budget;
use crate::math::hermite;

pub const DEF: SceneDef = SceneDef {
    id: "signal",
    title: "Signal propagation",
    group: "Neural networks",
    blurb: "What depth does to the angle between two inputs, and how the activation decides it.",
    make: |g| Box::new(Signal::new(g)),
};

/// Expectations over standard normal variables by the trapezoid rule on a regular
/// grid, which copes with the kinks of ReLU where Gaussian quadrature does not.
struct Gauss {
    /// A fine grid for one variable, a coarser one per axis for two.
    fine: (Vec<f32>, Vec<f32>),
    coarse: (Vec<f32>, Vec<f32>),
    arg: Vec<f32>,
    val: Vec<f32>,
    side: Vec<f32>,
}

impl Gauss {
    fn new() -> Self {
        let grid = |n: usize, reach: f32| {
            let z: Vec<f32> = (0..n).map(|i| -reach + 2.0 * reach * i as f32 / (n - 1) as f32).collect();
            let mut w: Vec<f32> = z.iter().map(|z| (-0.5 * z * z).exp()).collect();
            let total: f32 = w.iter().sum();
            w.iter_mut().for_each(|w| *w /= total);
            (z, w)
        };
        // even counts, so that no point sits exactly on a kink or a jump at zero
        Gauss { fine: grid(384, 8.0), coarse: grid(96, 6.0), arg: Vec::new(), val: Vec::new(), side: Vec::new() }
    }

    /// `E[f(√q Z)²]`.
    fn square(&mut self, act: Act, q: f32) -> f32 {
        let s = q.sqrt();
        self.fine.0.iter().zip(&self.fine.1).map(|(&z, &w)| w * act.f(s * z).powi(2)).sum()
    }

    /// `E[f'(√q Z)²]`.
    fn slope_square(&mut self, act: Act, q: f32) -> f32 {
        let s = q.sqrt();
        self.fine.0.iter().zip(&self.fine.1).map(|(&z, &w)| w * act.df(s * z, act.f(s * z)).powi(2)).sum()
    }

    /// `E[f(u) f(v)]` for `u`, `v` of variance `q` and correlation `c`.
    fn product(&mut self, act: Act, q: f32, c: f32) -> f32 {
        let (z, w) = (&self.coarse.0, &self.coarse.1);
        let (n, s, c) = (z.len(), q.sqrt(), c.clamp(-1.0, 1.0));
        let rest = (1.0 - c * c).max(0.0).sqrt();
        self.arg.clear();
        for &a in z {
            self.arg.extend(z.iter().map(|&b| s * (c * a + rest * b)));
        }
        self.val.resize(n * n, 0.0);
        act.apply(&self.arg, &mut self.val);
        self.side.clear();
        self.side.extend(z.iter().map(|&a| act.f(s * a)));
        let mut total = 0.0;
        for (i, row) in self.val.chunks_exact(n).enumerate() {
            let inner: f32 = row.iter().zip(w).map(|(v, w)| v * w).sum();
            total += w[i] * self.side[i] * inner;
        }
        total
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
struct Params {
    act: Act,
    /// Standard deviation of a weight times the square root of the layer's width.
    sw: f32,
    /// Standard deviation of a bias.
    sb: f32,
}

/// What becomes of the length of the activity, layer after layer.
#[derive(Clone, Copy, Debug, PartialEq)]
enum Fate {
    /// It settles at a fixed value.
    Settles,
    /// It grows without bound.
    Grows,
    /// It shrinks to nothing.
    Dies,
}

/// Lengths beyond these count as grown without bound, or as gone: the map of
/// correlations is drawn at them, as the limit it tends to.
const GROWN: f32 = 1e4;
const GONE: f32 = 1e-6;

/// The length the layers settle at, starting from 1, and what becomes of it.
fn settled_length(g: &mut Gauss, p: Params) -> (f32, Fate) {
    let (mut q, mut change) = (1.0f32, 0.0f32);
    for _ in 0..120 {
        let next = p.sw * p.sw * g.square(p.act, q) + p.sb * p.sb;
        if !next.is_finite() || next > GROWN {
            return (GROWN, Fate::Grows);
        }
        if next < GONE {
            return (GONE, Fate::Dies);
        }
        change = next - q;
        q = next;
        if change.abs() < 1e-5 * q {
            return (q, Fate::Settles);
        }
    }
    // Still moving after more layers than any network here has, and slowly: near
    // the border between two kinds of behaviour, where iterating on could take
    // thousands of layers. Instead, find where the iteration is heading: the first
    // length in the direction it moves that the layers keep (q' = q). There is none
    // if the length is on its way to infinity, or to nothing.
    let up = change > 0.0;
    let gap = |g: &mut Gauss, x: f32| p.sw * p.sw * g.square(p.act, x) + p.sb * p.sb - x;
    let (step, end, fate) = if up { (1.5f32, GROWN, Fate::Grows) } else { (1.0 / 1.5, GONE, Fate::Dies) };
    let mut near = q;
    loop {
        let far = near * step;
        if (up && far > end) || (!up && far < end) {
            return (end, fate);
        }
        if (gap(g, far) > 0.0) != up {
            // it lies between `near` and `far`: halve the interval (in proportion)
            let (mut a, mut b) = (near, far);
            for _ in 0..24 {
                let mid = (a * b).sqrt();
                if (gap(g, mid) > 0.0) == up {
                    a = mid;
                } else {
                    b = mid;
                }
            }
            return ((a * b).sqrt(), Fate::Settles);
        }
        near = far;
    }
}

/// The correlation after one layer, for inputs of length `q` and correlation `c`.
fn next_correlation(g: &mut Gauss, p: Params, q: f32, c: f32) -> f32 {
    let (signal, bias) = (p.sw * p.sw, p.sb * p.sb);
    let q_next = signal * g.square(p.act, q) + bias;
    ((signal * g.product(p.act, q, c) + bias) / q_next.max(1e-12)).clamp(-1.0, 1.0)
}

/// The factor by which a gradient's mean square grows or shrinks per layer on the way
/// back, for inputs of length `q`: below one, order; above one, chaos. At the settled
/// length it is also the slope of the map at c = 1.
fn slope_at_one(g: &mut Gauss, p: Params, q: f32) -> f32 {
    p.sw * p.sw * g.slope_square(p.act, q)
}

/// The slope at c = 1 of the map for inputs of length `q`, whatever the length:
/// `sw² q E[f'²] / q'`. (The map divides by the next layer's length `q'`; where the
/// length has not settled, that is not `q`, and this is not [`slope_at_one`].)
fn map_slope(g: &mut Gauss, p: Params, q: f32) -> f32 {
    let next = p.sw * p.sw * g.square(p.act, q) + p.sb * p.sb;
    slope_at_one(g, p, q) * q / next.max(1e-30)
}

/// The weight scale at the edge between order and chaos for bias scale `sb`: where
/// the slope at one is exactly one (found by bisection between 0.2 and 3).
fn edge(g: &mut Gauss, act: Act, sb: f32) -> f32 {
    let (mut lo, mut hi) = (0.2f32, 3.0f32);
    for _ in 0..22 {
        let mid = 0.5 * (lo + hi);
        let p = Params { act, sw: mid, sb };
        let (q, fate) = settled_length(g, p);
        if fate == Fate::Grows || (fate == Fate::Settles && slope_at_one(g, p, q) > 1.0) {
            hi = mid;
        } else {
            lo = mid;
        }
    }
    0.5 * (lo + hi)
}

/// The map's power series: `s[n]` is the weight on `cⁿ`, from the activation's
/// Hermite coefficients at length `q` (with the bias joining the constant term).
/// They sum to one when `q` is the settled length, up to the degrees left out.
fn series(p: Params, q: f32, degrees: usize) -> Vec<f32> {
    let s = (q as f64).sqrt();
    let a = hermite::coefficients(|x| p.act.f((s * x) as f32) as f64, degrees);
    let q_next = (p.sw * p.sw) as f64 * hermite::gauss_mean(|x| (p.act.f((s * x) as f32) as f64).powi(2)) + (p.sb * p.sb) as f64;
    a.iter().enumerate().map(|(n, a)| (((p.sw * p.sw) as f64 * a * a + if n == 0 { (p.sb * p.sb) as f64 } else { 0.0 }) / q_next.max(1e-30)) as f32).collect()
}

/// Two inputs carried through a real random network, layer by layer: the check on
/// the theory. Weights are drawn as they are needed and never stored.
struct Probe {
    width: usize,
    a: Vec<f32>,
    b: Vec<f32>,
    fa: Vec<f32>,
    fb: Vec<f32>,
    /// The correlation measured after each layer so far (index 0: the inputs), or
    /// NaN where the network's numbers have run out of range.
    seen: Vec<f32>,
    rng: Rng,
    /// The next row of the layer under way (0: none is).
    row: usize,
}

impl Probe {
    fn new(width: usize, q: f32, c: f32, seed: u64) -> Self {
        let mut rng = Rng::new(seed);
        let (s, rest) = (q.sqrt(), (1.0 - c * c).max(0.0).sqrt());
        let (mut a, mut b) = (Vec::with_capacity(width), Vec::with_capacity(width));
        for _ in 0..width {
            let (g1, g2) = (rng.normal(), rng.normal());
            a.push(s * g1);
            b.push(s * (c * g1 + rest * g2));
        }
        let mut probe = Probe { width, a, b, fa: vec![0.0; width], fb: vec![0.0; width], seen: Vec::new(), rng, row: 0 };
        probe.seen.push(probe.correlation());
        probe
    }

    /// The cosine between the two patterns of activity. (Summed in f64: in a network
    /// whose lengths grow, the squared lengths pass the largest f32 long before the
    /// activity does. NaN once even that is out of range.)
    fn correlation(&self) -> f32 {
        let (mut ab, mut aa, mut bb) = (0.0f64, 0.0f64, 0.0f64);
        for (&x, &y) in self.a.iter().zip(&self.b) {
            let (x, y) = (x as f64, y as f64);
            (ab, aa, bb) = (ab + x * y, aa + x * x, bb + y * y);
        }
        let c = ab / (aa.sqrt() * bb.sqrt());
        if c.is_finite() {
            c.clamp(-1.0, 1.0) as f32
        } else {
            f32::NAN
        }
    }

    /// Up to `rows` more rows of the next layer, finishing it if they reach its end.
    /// The weights are uniform with the right variance: over a wide layer only the
    /// variance matters. (A layer is done a part at a time so that a wide one fits
    /// in the frame's time allowance.)
    fn rows(&mut self, p: Params, rows: usize) {
        let n = self.width;
        if self.row == 0 {
            p.act.apply(&self.a, &mut self.fa);
            p.act.apply(&self.b, &mut self.fb);
        }
        let spread = p.sw * (3.0 / n as f32).sqrt();
        let end = (self.row + rows).min(n);
        for i in self.row..end {
            let (mut za, mut zb) = (0.0, 0.0);
            for j in 0..n {
                let w = (self.rng.f32() * 2.0 - 1.0) * spread;
                za += w * self.fa[j];
                zb += w * self.fb[j];
            }
            let bias = self.rng.normal() * p.sb;
            self.a[i] = za + bias;
            self.b[i] = zb + bias;
        }
        self.row = end % n;
        if end == n {
            self.seen.push(self.correlation());
        }
    }

    /// The rest of the layer under way, or one more whole layer.
    #[cfg(test)]
    fn layer(&mut self, p: Params) {
        self.rows(p, self.width - self.row);
    }
}

const CURVE: usize = 65;
/// The phase diagram's colour for settings under which lengths grow without bound.
const GREY: Color = Color::hex(0x3A3F4B);
const PHASE_N: usize = 44;
const PHASE_SW: f32 = 3.0;
const PHASE_SB: f32 = 2.0;
const DEGREES: usize = 12;

struct Signal {
    p: Params,
    act: usize,
    start: f32,
    depth: usize,
    width: usize,
    seed: u64,
    gauss: Gauss,
    /// Everything derived from the settings, recomputed when they change: the
    /// length the map is drawn at and what becomes of the length, the factor for
    /// gradients there, and the slope of the map drawn at c = 1.
    solved: Option<Params>,
    q: f32,
    fate: Fate,
    chi: f32,
    slope: f32,
    curve: Vec<f32>,
    rest: f32,
    rest_slope: f32,
    spectrum: Vec<f32>,
    theory: Vec<f32>,
    probe: Option<Probe>,
    /// The phase diagram for the current activation, filled a few pixels per frame
    /// (row by row: `phase_done` is how many), and whether any of them is grey.
    phase: Image,
    phase_for: Option<Act>,
    phase_done: usize,
    phase_grey: bool,
}

impl Signal {
    fn new(gfx: &Gfx) -> Self {
        let mut phase = Image::new(gfx, PHASE_N as u32, PHASE_N as u32);
        phase.smooth = true;
        Signal {
            p: Params { act: Act::Tanh, sw: 1.6, sb: 0.3 },
            act: 0,
            // two nearly equal inputs: in the chaotic regime they drift apart
            start: 0.97,
            depth: 24,
            width: 512,
            seed: 1,
            gauss: Gauss::new(),
            solved: None,
            q: 1.0,
            fate: Fate::Settles,
            chi: 1.0,
            slope: 1.0,
            curve: Vec::new(),
            rest: 1.0,
            rest_slope: 0.0,
            spectrum: Vec::new(),
            theory: Vec::new(),
            probe: None,
            phase,
            phase_for: None,
            phase_done: 0,
            phase_grey: false,
        }
    }

    /// The tabulated map at correlation `c` (between its samples, a straight line).
    fn map(&self, c: f32) -> f32 {
        let x = (c.clamp(-1.0, 1.0) + 1.0) * 0.5 * (CURVE - 1) as f32;
        let i = (x.floor() as usize).min(CURVE - 2);
        self.curve[i] + (self.curve[i + 1] - self.curve[i]) * (x - i as f32)
    }

    fn solve(&mut self) {
        let p = self.p;
        (self.q, self.fate) = settled_length(&mut self.gauss, p);
        self.chi = slope_at_one(&mut self.gauss, p, self.q);
        self.slope = map_slope(&mut self.gauss, p, self.q);
        self.curve = (0..CURVE).map(|i| next_correlation(&mut self.gauss, p, self.q, -1.0 + 2.0 * i as f32 / (CURVE - 1) as f32)).collect();
        self.spectrum = series(p, self.q, DEGREES);
        // where correlations come to rest: follow the map from the middle
        let mut c = 0.5;
        for _ in 0..2000 {
            c = self.map(c);
        }
        self.rest = c;
        self.rest_slope = ((self.map((c + 0.02).min(1.0)) - self.map(c - 0.02)) / ((c + 0.02).min(1.0) - (c - 0.02))).abs();
        self.solved = Some(p);
        self.retrace();
    }

    /// The path of the chosen pair of inputs: in theory, and through a fresh network.
    fn retrace(&mut self) {
        self.theory.clear();
        let mut c = self.start;
        self.theory.push(c);
        for _ in 0..self.depth {
            c = self.map(c);
            self.theory.push(c);
        }
        self.probe = Some(Probe::new(self.width, self.q, self.start, self.seed));
    }

    /// A few more pixels of the phase diagram: for every weight and bias scale, is the
    /// factor for gradients below or above one? (Grey: the length grows without
    /// bound. Where it dies out, the factor is the one for vanishing activity.)
    fn paint_phase(&mut self, budget: &Budget) {
        if self.phase_for != Some(self.p.act) {
            self.phase_for = Some(self.p.act);
            (self.phase_done, self.phase_grey) = (0, false);
            self.phase.pixels_mut().fill(Color::hex(0x10141E));
        }
        let heat = Cmap::CoolWarm.table();
        while self.phase_done < PHASE_N * PHASE_N && !budget.spent() {
            let (row, col) = (self.phase_done / PHASE_N, self.phase_done % PHASE_N);
            let sb = PHASE_SB * row as f32 / (PHASE_N - 1) as f32;
            let p = Params { act: self.p.act, sw: PHASE_SW * (col as f32 + 0.5) / PHASE_N as f32, sb };
            let (q, fate) = settled_length(&mut self.gauss, p);
            let color = if fate == Fate::Grows {
                self.phase_grey = true;
                GREY
            } else {
                let chi = slope_at_one(&mut self.gauss, p, q).max(1e-6);
                heat[table_index(0.5 + 0.5 * (1.3 * chi.ln()).tanh())]
            };
            self.phase.pixels_mut()[self.phase_done] = color;
            self.phase_done += 1;
        }
    }

    /// Set up one of the experiments in `TRIALS`.
    fn set_up(&mut self, k: usize) {
        let (act, sw, sb, start) = match k {
            0 => (Act::Tanh, 0.9, 0.3, 0.0),
            1 => (Act::Tanh, 2.5, 0.3, 0.99),
            2 => (Act::Tanh, edge(&mut self.gauss, Act::Tanh, 0.3), 0.3, 0.6),
            3 => (Act::Relu, std::f32::consts::SQRT_2, 0.0, 0.0),
            _ => (Act::Hermite2, 1.0, 0.0, 0.9),
        };
        self.p = Params { act, sw, sb };
        self.act = Act::ALL.iter().position(|&a| a == act).unwrap_or(0);
        (self.start, self.depth) = (start, 24);
        // (the next frame sees the new settings and works everything out again)
        self.solved = None;
    }

    fn regime(&self) -> &'static str {
        if self.fate == Fate::Grows {
            "lengths grow without bound: no depth of this network is stable"
        } else if self.fate == Fate::Dies || self.q < 1e-4 {
            "the signal dies out: its length shrinks to nothing"
        } else if self.chi < 0.97 {
            "ordered: every pair of inputs ends up perfectly correlated, so depth erases what told them apart"
        } else if self.chi > 1.03 {
            "chaotic: even nearly equal inputs drift apart, to the resting correlation"
        } else {
            "at the edge between the two: correlations change slowly, so signals survive many layers"
        }
    }
}

/// Where the four parts of the picture go, in the rectangle `w` wide and five high
/// that the canvas shows (the "stage" of `scenes/kit.rs`): the map (a square), the correlation layer
/// by layer, the diagram of all settings (a square) and the bars. Each is given by
/// its bottom-left corner and its size; room for titles and tick labels is left
/// around them. The bottom 0.3 is kept for one line of text.
#[derive(Clone, Copy, Debug)]
struct Layout {
    map: (Vec2, f32),
    depth: (Vec2, Vec2),
    phase: (Vec2, f32),
    bars: (Vec2, Vec2),
}

impl Layout {
    fn of(w: f32) -> Layout {
        let (left, right, top, floor): (f32, f32, f32, f32) = (-w * 0.5 + 0.42, w * 0.5 - 0.16, 2.5 - 0.44, -2.5 + 0.3);
        // three arrangements; the one that gives the largest map is used
        // (each needs a certain size to have room for its other parts at all)
        let wide = Some((top - floor - 0.42).min(w * 0.5 - 0.5)).filter(|m| *m >= 2.6).unwrap_or(0.0);
        let medium = Some((w - 2.45).min(2.05)).filter(|m| *m >= 1.2).unwrap_or(0.0);
        let tall_phase = ((w - 0.58 - 0.5) * 0.5).min(0.95);
        let tall = (w - 0.58).min(top - 0.06 - floor - 0.42 - 0.9 - 0.3 - 0.64 - tall_phase);
        if wide >= medium && wide >= tall {
            let m = wide;
            let y = floor + 0.42 + 0.5 * (top - floor - 0.42 - m);
            let column = left + m + 0.6;
            let depth_h = (m * 0.36).clamp(0.8, 1.6);
            let p = (m - depth_h - 0.8).min((right - column) * 0.45).max(0.5);
            Layout {
                map: (Vec2::new(left, y), m),
                depth: (Vec2::new(column, y + m - depth_h), Vec2::new(right - column, depth_h)),
                phase: (Vec2::new(column, y), p),
                bars: (Vec2::new(column + p + 0.5, y), Vec2::new(right - column - p - 0.5, (p - 0.16).max(0.3))),
            }
        } else if medium >= tall {
            let m = medium;
            let p = (top - floor - 0.42 - 0.42 - 0.38 - m).clamp(0.6, 1.5);
            let y = top - m - 0.5 * (top - floor - 0.42 - 0.42 - 0.38 - m - p).max(0.0);
            let column = left + m + 0.55;
            let depth_h = m.min(1.7);
            Layout {
                map: (Vec2::new(left, y), m),
                depth: (Vec2::new(column, y + m - depth_h), Vec2::new(right - column, depth_h)),
                phase: (Vec2::new(left, y - 0.42 - 0.38 - p), p),
                bars: (Vec2::new(left + p + 0.55, y - 0.42 - 0.38 - p), Vec2::new(right - left - p - 0.55, (p - 0.16).max(0.3))),
            }
        } else {
            let (m, p) = (tall.max(0.6), tall_phase);
            let x = (-m * 0.5 + 0.13).max(left);
            let top = top - 0.06;
            let depth_y = top - m - 0.42 - 0.3 - 0.6;
            let y = depth_y - 0.3 - 0.34 - p;
            Layout {
                map: (Vec2::new(x, top - m), m),
                depth: (Vec2::new(left, depth_y), Vec2::new(right - left, 0.6)),
                phase: (Vec2::new(left, y), p),
                bars: (Vec2::new(left + p + 0.5, y), Vec2::new(right - left - p - 0.5, (p - 0.16).max(0.3))),
            }
        }
    }
}

/// Rows of the test network worked out between looks at the clock.
const PROBE_ROWS: usize = 32;

/// Experiments: (button, what to look for). `Signal::set_up` holds what each sets.
const TRIALS: [(&str, &str); 5] = [
    ("order: small weights", "Two unrelated inputs ($c = 0$) climb the staircase to $c = 1$: after enough layers the network cannot tell them apart."),
    ("chaos: large weights", "Two nearly equal inputs ($c = 0.99$) come apart, down to the resting value: differences too small to matter are blown up."),
    ("the edge between them", "The curve only just touches the diagonal at $c = 1$, so the staircase creeps. Signals, and gradients on the way back, survive many layers: this is where deep networks are best started."),
    ("ReLU at its edge", r#"With ReLU the edge is at weight scale $\sqrt{2}$ (the usual "He" starting scale). Lengths are kept, yet the curve lies above the diagonal: inputs are still pulled together, slowly."#),
    ("an activation that squares $c$", r"$(x^2 - 1)/\sqrt{2}$ is the second Hermite polynomial. One layer squares the correlation, $c \mapsto c^2$, and there is a single bar, at power 2."),
];

impl Scene for Signal {
    fn view(&self) -> View {
        STAGE_VIEW
    }

    fn settings(&self, s: &mut ViewSettings) {
        s.grid = false;
    }

    /// Nothing moves here except while results are still being worked out.
    fn animated(&self) -> bool {
        self.solved != Some(self.p) || self.phase_done < PHASE_N * PHASE_N || self.phase_for != Some(self.p.act) || self.probe.as_ref().is_some_and(|p| p.seen.len() <= self.depth)
    }

    fn ui(&mut self, ui: &mut egui::Ui, _: &Gfx) {
        ui::note(ui, "Before any training, a deep network with random weights already does something to its inputs: layer after layer it pulls them together or pushes them apart. Which of the two decides whether the network can be trained at all, and it can be read off one curve.");
        ui::term(ui, "Correlation $c$:", "how alike the network's activity is for two inputs. 1: identical, 0: unrelated, −1: opposite. (It is the cosine of the angle between the two patterns of activity.)");

        ui::section(ui, "Things to try");
        for (k, (label, look_for)) in TRIALS.iter().enumerate() {
            if ui::experiment(ui, label, look_for) {
                self.set_up(k);
            }
        }

        ui::section(ui, "Activation");
        let names: Vec<&str> = Act::ALL.iter().map(|a| activation_name(*a)).collect();
        if ui::choice(ui, &names, &mut self.act) {
            self.p.act = Act::ALL[self.act];
        }
        ui::section(ui, "Random weights");
        ui::slider(ui, r"weight scale $\sigma_w$", &mut self.p.sw, 0.05..=PHASE_SW);
        ui::slider(ui, r"bias scale $\sigma_b$", &mut self.p.sb, 0.0..=PHASE_SB);
        ui::note(ui, r"How large the random weights and biases are: a weight has standard deviation $\sigma_w / \sqrt{\text{width}}$, a bias $\sigma_b$. Or drag the white point in the diagram of all settings.");
        ui::section(ui, "The two inputs");
        let mut moved = ui::slider(ui, "starting correlation", &mut self.start, -1.0..=1.0);
        moved |= ui::slider_int(ui, "layers", &mut self.depth, 4..=60);
        moved |= ui::slider_int_log(ui, "width of the test network", &mut self.width, 32..=1024);
        if ui.button("another random network").clicked() {
            self.seed += 1;
            moved = true;
        }
        if moved && self.solved.is_some() {
            self.retrace();
        }
        ui::note(ui, "The line is the theory for an infinitely wide network. The dots are one real random network of the chosen width. Where they part ways, the network is too narrow or the balance of lengths is unstable. Where they stop, its numbers have grown too large (or too small) for the computer to hold.");
        ui::section(ui, "Reading the picture");
        ui::key(
            ui,
            &[
                (pal::CYAN, "the map: the correlation after one layer, for each correlation before it"),
                (pal::YELLOW, "the staircase, and the line in the chart: your two inputs, layer after layer"),
                (pal::PINK, "where correlations come to rest"),
                (pal::WHITE, "white dots: the same, measured in one real random network"),
                (GREY, "grey, in the diagram of all settings: lengths grow without bound"),
                (pal::PURPLE, "bars: how much each power of $c$ counts in the map"),
            ],
        );
        ui::note(ui, "The staircase goes up to the curve (one layer), across to the dashed diagonal (its output becomes the next input), and so on. Where the curve is above the diagonal, a layer raises the correlation.");
        ui::term(ui, "Order:", "every pair of inputs ends up at $c = 1$. Depth erases what told them apart.");
        ui::term(ui, "Chaos:", "even nearly equal inputs are driven apart. Depth scrambles them.");
        ui::term(ui, "The edge:", "the border between the two, where signals travel furthest.");

        ui::section(ui, "Numbers");
        let length = match self.fate {
            Fate::Settles => format!("${}$", fmt(self.q)),
            Fate::Grows => "none: it grows without bound".to_string(),
            Fate::Dies => "none: it shrinks to nothing".to_string(),
        };
        ui::readout(ui, "length the layers settle at", length);
        ui::readout(ui, "slope at $c = 1$", format!("{:.3}", self.slope));
        ui::readout(ui, "correlations come to rest at", format!("{:.3}", self.rest));
        if self.rest_slope > 1e-4 && self.rest_slope < 0.999 {
            ui::readout(ui, "layers to close 90 % of the gap", format!("{:.1}", 10f32.ln() / -self.rest_slope.ln()));
        }
        ui.label(egui::RichText::new(self.regime()).size(12.5).color(pal::YELLOW.egui()));
        ui::note(ui, "\"Length\" is the mean square of a layer's activity. Where the length settles, the slope of the curve at $c = 1$ decides the regime: below 1 is order, above 1 is chaos. It is then also the factor by which a gradient's mean square grows or shrinks at each layer on its way back (its size, by the square root of that). Where the length does not settle, the curve is drawn for the length it tends to. The slope always equals the average power among the bars. A pure Hermite polynomial of degree $k$ as activation raises the correlation to the power $k$.");
        ui::see_also(ui, &[("lyapunov", "Lyapunov exponents: the same story as the growth of a nudge"), ("coherence", "what averaging Jacobians keeps"), ("hermite_expand", "Hermite expansions"), ("mlp", "train a network")]);
    }

    fn frame(&mut self, f: &mut Frame) {
        if self.solved != Some(self.p) {
            self.solve();
        }
        let budget = Budget::new(0.003);
        // carry the test network a few layers further
        let (depth, p) = (self.depth, self.p);
        if let Some(probe) = &mut self.probe {
            while probe.seen.len() <= depth && !budget.spent() {
                probe.rows(p, PROBE_ROWS);
            }
        }
        self.paint_phase(&budget);

        let (w, ppu) = (stage_width(f), pixels_per_unit(f));
        let lay = Layout::of(w);
        let pick = |options: &[&'static str], room: f32, size: f32| options.iter().copied().find(|t| text_width(t, size) < room * ppu).unwrap_or(options[options.len() - 1]);

        // ----- the map c -> c', with the staircase of repeated layers
        let (lo, side) = lay.map;
        let at = |c: f32, v: f32| lo + Vec2::new(c + 1.0, v + 1.0) * (side * 0.5);
        inset(f, lo - Vec2::splat(0.06), lo + Vec2::splat(side + 0.06));
        for t in [-1.0f32, -0.5, 0.0, 0.5, 1.0] {
            let c = if t == 0.0 { pal::DIM.a(0.6) } else { pal::FAINT.a(0.5) };
            f.line(at(t, -1.0), at(t, 1.0), 1.0, c);
            f.line(at(-1.0, t), at(1.0, t), 1.0, c);
            if side * ppu > 230.0 || t.fract() == 0.0 {
                f.text(at(t, -1.0) - Vec2::new(0.0, 0.07), fmt_tick(t, 0.5), 11.0, pal::DIM, Align2::CENTER_TOP);
                f.text(at(-1.0, t) - Vec2::new(0.07, 0.0), fmt_tick(t, 0.5), 11.0, pal::DIM, Align2::RIGHT_CENTER);
            }
        }
        f.dashed(at(-1.0, -1.0), at(1.0, 1.0), 1.4, pal::DIM);
        f.path((0..CURVE).map(|i| at(-1.0 + 2.0 * i as f32 / (CURVE - 1) as f32, self.curve[i])), 3.0, pal::CYAN);
        let mut stairs = vec![at(self.start, self.start)];
        for pair in self.theory.windows(2) {
            stairs.push(at(pair[0], pair[1]));
            stairs.push(at(pair[1], pair[1]));
        }
        f.path(stairs, 1.6, pal::YELLOW.a(0.9));
        f.dot(at(self.rest, self.rest), 9.0, pal::PINK);
        // the resting point is named below the diagonal, in as many words as fit
        // between it and the edge of the square
        let rest_at = at(self.rest, self.rest);
        let room = (lo.x + side - rest_at.x - 0.1) * ppu;
        if text_width("comes to rest here", 11.5) < room {
            f.text(rest_at + Vec2::new(0.08, -0.08), "comes to rest here", 11.5, pal::PINK, Align2::LEFT_TOP);
        } else if text_width("rest", 11.5) < room {
            f.text(rest_at + Vec2::new(0.08, -0.08), "rest", 11.5, pal::PINK, Align2::LEFT_TOP);
        } else {
            f.text(rest_at + Vec2::new(0.02, -0.12), "rest", 11.5, pal::PINK, Align2::RIGHT_TOP);
        }
        let title = pick(&["one layer's map: the correlation after it (up) against the correlation before it (across)", "one layer: correlation after (up) against before (across)", "one layer: $c$ after against $c$ before"], side + 0.3, 12.5);
        f.text(lo + Vec2::new(-0.3, side + 0.12), title, 12.5, pal::TEXT, Align2::LEFT_BOTTOM);
        f.text(at(1.0, -1.0) + Vec2::new(0.0, -0.24), "correlation before the layer →", 11.0, pal::DIM, Align2::RIGHT_TOP);
        f.text(at(-1.0, 1.0) + Vec2::new(0.1, -0.1), format!("slope at $c = 1$:  {:.2}", self.slope), 12.0, pal::CYAN, Align2::LEFT_TOP);
        if side * ppu > 300.0 {
            f.text(at(-0.52, -0.48), "diagonal: no change", 10.5, pal::DIM, Align2::LEFT_TOP);
        }
        let mut handle = at(self.start, self.start);
        if f.drag2(1, &mut handle, pal::YELLOW) {
            self.start = ((handle.x - lo.x) / (side * 0.5) - 1.0).clamp(-1.0, 1.0);
            self.retrace();
        }
        // the handle's name stays inside the square
        let (start_off, start_align) = if self.start > 0.35 { (Vec2::new(-0.1, 0.03), Align2::RIGHT_BOTTOM) } else { (Vec2::new(0.1, -0.02), Align2::LEFT_CENTER) };
        f.text(at(self.start, self.start) + start_off, "your two inputs start here (drag)", 11.5, pal::YELLOW, start_align);

        // ----- correlation against depth: theory and one real network
        let (o, size) = lay.depth;
        let floor = self.theory.iter().chain(self.probe.iter().flat_map(|p| p.seen.iter())).fold(0.0f32, |m, &v| m.min(v)).min(0.0);
        let floor = if floor < 0.0 { -1.0 } else { 0.0 };
        let spot = |layer: usize, c: f32| o + Vec2::new(size.x * layer as f32 / self.depth as f32, size.y * (c - floor) / (1.0 - floor));
        inset(f, o - Vec2::splat(0.06), o + size + Vec2::splat(0.06));
        f.line(spot(0, floor), spot(self.depth, floor), 1.0, pal::DIM.a(0.8));
        f.line(spot(0, 1.0), spot(self.depth, 1.0), 1.0, pal::FAINT.a(0.6));
        if floor < 0.0 {
            f.line(spot(0, 0.0), spot(self.depth, 0.0), 1.0, pal::FAINT.a(0.4));
        }
        f.path(self.theory.iter().enumerate().map(|(l, &c)| spot(l, c)), 2.2, pal::YELLOW);
        if let Some(probe) = &self.probe {
            let dot = (size.x * ppu / self.depth as f32 * 0.5).clamp(3.0, 6.0);
            let seen = &probe.seen[..probe.seen.len().min(self.depth + 1)];
            for (l, &c) in seen.iter().enumerate().filter(|(_, c)| c.is_finite()) {
                f.dot(spot(l, c), dot, pal::WHITE);
            }
            // where the real network's numbers left the range a computer holds, the
            // dots stop: say so there
            if let Some(l) = seen.iter().position(|c| !c.is_finite()) {
                f.dashed(spot(l, floor), spot(l, 1.0), 1.0, pal::DIM);
                let room = o.x + size.x - spot(l, 0.0).x - 0.08;
                let note = pick(&["no dots from here: the network's numbers are out of range", "numbers out of range", "out of range", ""], room, 10.5);
                f.text(spot(l, 0.5 * (1.0 + floor)) + Vec2::new(0.06, 0.0), note, 10.5, pal::DIM, Align2::LEFT_CENTER);
            }
        }
        let title = pick(&["the correlation of your two inputs, layer by layer", "the correlation, layer by layer", "$c$, layer by layer"], size.x, 12.5);
        f.text(o + Vec2::new(0.0, size.y + 0.1), title, 12.5, pal::TEXT, Align2::LEFT_BOTTOM);
        // what the line and the dots are: beside the title if there is room, or else
        // under the chart between its two end labels
        let legend = format!("line: theory    dots: a network {} wide", self.width);
        let last = format!("layer {}", self.depth);
        if text_width(title, 12.5) + text_width(&legend, 11.0) + 24.0 < size.x * ppu {
            f.text(o + size + Vec2::new(0.0, 0.1), legend, 11.0, pal::DIM, Align2::RIGHT_BOTTOM);
        } else if text_width(&legend, 11.0) + text_width(&last, 11.0) + 60.0 < size.x * ppu {
            f.text(o + Vec2::new(size.x * 0.5, -0.07), legend, 11.0, pal::DIM, Align2::CENTER_TOP);
        }
        f.text(spot(0, floor) - Vec2::new(0.06, 0.0), fmt_tick(floor, 1.0), 11.0, pal::DIM, Align2::RIGHT_CENTER);
        f.text(spot(0, 1.0) - Vec2::new(0.06, 0.0), "1", 11.0, pal::DIM, Align2::RIGHT_CENTER);
        f.text(o + Vec2::new(0.0, -0.07), "input", 11.0, pal::DIM, Align2::LEFT_TOP);
        f.text(o + Vec2::new(size.x, -0.07), last, 11.0, pal::DIM, Align2::RIGHT_TOP);

        // ----- every weight and bias scale at once
        let (pm, ps) = lay.phase;
        f.image(&mut self.phase, pm, pm + Vec2::splat(ps), -0.01);
        frame_square(f, pm + Vec2::splat(ps * 0.5), ps * 0.5, 1.2, pal::DIM);
        let mut knob = pm + Vec2::new(self.p.sw / PHASE_SW, self.p.sb / PHASE_SB) * ps;
        if f.drag2(2, &mut knob, pal::WHITE) {
            let t = ((knob - pm) / ps).clamp(Vec2::ZERO, Vec2::ONE);
            self.p.sw = (t.x * PHASE_SW).max(0.05);
            self.p.sb = t.y * PHASE_SB;
        }
        let title = if self.phase_grey {
            pick(&["every setting at once: order (blue), chaos (red), lengths unbounded (grey)", "order (blue), chaos (red), unbounded (grey)", "blue order, red chaos, grey unbounded", "all settings"], ps + 0.45, 12.0)
        } else {
            pick(&["every setting at once: order (blue), chaos (red)", "all settings: order (blue), chaos (red)", "order (blue), chaos (red)", "all settings"], ps + 0.45, 12.0)
        };
        f.text(pm + Vec2::new(0.0, ps + 0.08), title, 12.0, pal::TEXT, Align2::LEFT_BOTTOM);
        f.text(pm + Vec2::new(ps * 0.5, -0.06), "weight scale →", 11.0, pal::DIM, Align2::CENTER_TOP);
        f.text(pm + Vec2::new(0.0, -0.06), "0", 10.5, pal::DIM, Align2::CENTER_TOP);
        f.text(pm + Vec2::new(ps, -0.06), fmt_tick(PHASE_SW, 1.0), 10.5, pal::DIM, Align2::CENTER_TOP);
        f.text(pm + Vec2::new(-0.05, ps * 0.5), "bias", 11.0, pal::DIM, Align2::RIGHT_BOTTOM);
        f.text(pm + Vec2::new(-0.05, ps * 0.5), "scale ↑", 11.0, pal::DIM, Align2::RIGHT_TOP);
        f.text(pm + Vec2::new(-0.05, 0.0), "0", 10.5, pal::DIM, Align2::RIGHT_CENTER);
        f.text(pm + Vec2::new(-0.05, ps), fmt_tick(PHASE_SB, 1.0), 10.5, pal::DIM, Align2::RIGHT_CENTER);
        if ps * ppu > 150.0 {
            // on a plate: the diagram behind it runs from deep blue through white to red
            if let Some(at) = on_screen(f, knob.extend(0.0)) {
                f.caption_px(at + Vec2::new(0.0, -13.0), "these settings (drag)", 10.5, pal::WHITE, Align2::CENTER_BOTTOM);
            }
        }

        // ----- the map as a power series: the activation's Hermite spectrum
        let (so, ss) = lay.bars;
        inset(f, so - Vec2::splat(0.06), so + ss + Vec2::splat(0.06));
        f.line(so, so + Vec2::new(ss.x, 0.0), 1.0, pal::DIM.a(0.8));
        let heights: Vec<f32> = self.spectrum.iter().map(|s| s.clamp(0.0, 1.0) * ss.y).collect();
        let bar = ss.x / heights.len() as f32;
        bars(f, so.x, bar, &heights, so.y, pal::PURPLE);
        let every = if bar * ppu > 15.0 { 1 } else { 2 };
        for n in (0..heights.len()).step_by(every) {
            f.text(so + Vec2::new((n as f32 + 0.5) * bar, -0.05), n.to_string(), 10.5, pal::DIM, Align2::CENTER_TOP);
        }
        let shown: f32 = self.spectrum.iter().sum();
        let title = pick(&["the map as a sum of powers of $c$: how much each power counts", "how much each power of $c$ counts", "powers of $c$"], ss.x, 12.0);
        f.text(so + Vec2::new(0.0, ss.y + 0.1), title, 12.0, pal::TEXT, Align2::LEFT_BOTTOM);
        // what the bars shown leave out (said only if it is something, or there is room to spare)
        let rest = (1.0 - shown).max(0.0) * 100.0;
        if rest >= 0.5 || ss.x * ppu > 260.0 {
            let tail = pick(&["higher powers", "higher"], ss.x * 0.6, 10.5);
            f.text(so + ss - Vec2::new(0.04, 0.04), format!("{tail}: {rest:.0} %"), 10.5, pal::DIM, Align2::RIGHT_TOP);
        }
        f.text(so + Vec2::new(ss.x, -0.2), "power of $c$ →", 10.5, pal::DIM, Align2::RIGHT_TOP);

        // ----- the verdict, in words
        wrapped(f, Vec2::new(-w * 0.5 + 0.16, -2.5 + 0.27), w - 0.32, self.regime(), 12.0, pal::YELLOW);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn p(act: Act, sw: f32, sb: f32) -> Params {
        Params { act, sw, sb }
    }

    #[test]
    fn the_relu_map_is_the_known_one() {
        // E[relu(u) relu(v)] / E[relu(u)²] = (sin t + (pi - t) cos t) / pi, with c = cos t
        let mut g = Gauss::new();
        for q in [0.5f32, 1.0, 4.0] {
            for i in 0..=20 {
                let c = -1.0 + 0.1 * i as f32;
                let t = c.clamp(-1.0, 1.0).acos();
                let exact = (t.sin() + (std::f32::consts::PI - t) * c) / std::f32::consts::PI;
                let ours = next_correlation(&mut g, p(Act::Relu, 1.3, 0.0), q, c);
                assert!((ours - exact).abs() < 6e-3, "q={q} c={c}: {ours} vs {exact}");
            }
        }
        // and its slope at one is sw²/2: the edge is at sw = sqrt(2)
        assert!((slope_at_one(&mut g, p(Act::Relu, 2f32.sqrt(), 0.0), 1.0) - 1.0).abs() < 1e-3);
    }

    #[test]
    fn a_hermite_polynomial_raises_the_correlation_to_its_degree() {
        let mut g = Gauss::new();
        for (act, degree) in [(Act::Hermite2, 2), (Act::Hermite3, 3)] {
            // at length one these activations are exactly h_2 and h_3
            for i in 0..=10 {
                let c = -1.0 + 0.2 * i as f32;
                let ours = next_correlation(&mut g, p(act, 1.0, 0.0), 1.0, c);
                assert!((ours - c.powi(degree)).abs() < 2e-3, "{act:?} c={c}: {ours}");
            }
            let s = series(p(act, 1.0, 0.0), 1.0, 6);
            assert!((s[degree as usize] - 1.0).abs() < 1e-3, "{s:?}");
        }
    }

    #[test]
    fn tanh_has_order_and_chaos_and_the_slope_is_the_average_degree() {
        let mut g = Gauss::new();
        let slope = |g: &mut Gauss, sw: f32, sb: f32| {
            let (q, fate) = settled_length(g, p(Act::Tanh, sw, sb));
            assert_eq!(fate, Fate::Settles);
            (q, slope_at_one(g, p(Act::Tanh, sw, sb), q))
        };
        // (without biases, small weights let the activity die out: ordered too)
        let (q, fate) = settled_length(&mut g, p(Act::Tanh, 0.9, 0.0));
        assert!(fate == Fate::Dies && slope_at_one(&mut g, p(Act::Tanh, 0.9, 0.0), q) < 1.0);
        assert!(slope(&mut g, 1.0, 0.5).1 < 1.0);
        let (q, chi) = slope(&mut g, 1.8, 0.3);
        assert!(chi > 1.0);
        // the series sums to one at the settled length, and its average degree is the slope
        let s = series(p(Act::Tanh, 1.8, 0.3), q, 60);
        let total: f32 = s.iter().sum();
        let mean_degree: f32 = s.iter().enumerate().map(|(n, s)| n as f32 * s).sum();
        assert!((total - 1.0).abs() < 5e-3, "sum {total}");
        assert!((mean_degree - chi).abs() < 0.03 * chi, "average degree {mean_degree} vs slope {chi}");
        // the power series reproduces the map
        for c in [-0.6f32, 0.0, 0.5, 0.9] {
            let from_series: f32 = s.iter().enumerate().map(|(n, s)| s * c.powi(n as i32)).sum();
            assert!((from_series - next_correlation(&mut g, p(Act::Tanh, 1.8, 0.3), q, c)).abs() < 5e-3);
        }
    }

    #[test]
    fn a_real_random_network_follows_the_theory() {
        let mut g = Gauss::new();
        for (params, start) in [(p(Act::Tanh, 1.6, 0.3), 0.6f32), (p(Act::Tanh, 0.9, 0.4), 0.1), (p(Act::Relu, 1.2, 0.2), -0.3)] {
            let (q, fate) = settled_length(&mut g, params);
            assert_eq!(fate, Fate::Settles);
            let mut probe = Probe::new(3000, q, start, 5);
            let mut c = start;
            for layer in 1..=8 {
                c = next_correlation(&mut g, params, q, c);
                probe.layer(params);
                let seen = probe.seen[layer];
                assert!((seen - c).abs() < 0.06, "{params:?} layer {layer}: measured {seen}, theory {c}");
            }
        }
    }

    /// The experiments' captions, held to account.
    #[test]
    fn the_experiments_show_what_their_captions_say() {
        let mut g = Gauss::new();
        let follow = |g: &mut Gauss, params: Params, start: f32, layers: usize| {
            let (q, fate) = settled_length(g, params);
            assert_eq!(fate, Fate::Settles);
            let mut c = start;
            for _ in 0..layers {
                c = next_correlation(g, params, q, c);
            }
            (c, slope_at_one(g, params, q))
        };
        // order: unrelated inputs become one
        let (c, chi) = follow(&mut g, p(Act::Tanh, 0.9, 0.3), 0.0, 24);
        assert!(chi < 1.0 && c > 0.999, "order: slope {chi}, ends at {c}");
        // chaos: nearly equal inputs come apart
        let (c, chi) = follow(&mut g, p(Act::Tanh, 2.5, 0.3), 0.99, 24);
        assert!(chi > 1.0 && c < 0.9, "chaos: slope {chi}, ends at {c}");
        // the edge: the slope is one there, and the staircase creeps
        let sw = edge(&mut g, Act::Tanh, 0.3);
        let (c, chi) = follow(&mut g, p(Act::Tanh, sw, 0.3), 0.6, 24);
        assert!((chi - 1.0).abs() < 0.01 && c > 0.6 && c < 0.999, "edge at {sw}: slope {chi}, ends at {c}");
        // ReLU at the square root of two: lengths are kept, inputs still pulled together
        let relu = p(Act::Relu, std::f32::consts::SQRT_2, 0.0);
        let (q, _) = settled_length(&mut g, relu);
        assert!((q - 1.0).abs() < 1e-3, "ReLU length {q}");
        for c in [-0.8f32, -0.3, 0.0, 0.4, 0.9] {
            assert!(next_correlation(&mut g, relu, q, c) > c);
        }
        // the squaring activation keeps its length of one, where it is exactly h_2
        let (q, fate) = settled_length(&mut g, p(Act::Hermite2, 1.0, 0.0));
        assert!(fate == Fate::Settles && (q - 1.0).abs() < 1e-3, "Hermite length {q}");
    }

    /// The parts of the picture stay on the canvas and clear of each other, whatever
    /// its shape.
    #[test]
    fn the_layout_fits_every_canvas() {
        for w in [2.0f32, 2.44, 3.0, 3.6, 4.2, 5.2, 6.0, 6.6, 8.0, 10.0, 12.0] {
            let l = Layout::of(w);
            // each part with the room its labels take: (left, bottom, right, top)
            let rects = [
                (l.map.0.x - 0.3, l.map.0.y - 0.4, l.map.0.x + l.map.1, l.map.0.y + l.map.1 + 0.3),
                (l.depth.0.x - 0.2, l.depth.0.y - 0.25, l.depth.0.x + l.depth.1.x, l.depth.0.y + l.depth.1.y + 0.28),
                (l.phase.0.x - 0.3, l.phase.0.y - 0.25, l.phase.0.x + l.phase.1, l.phase.0.y + l.phase.1 + 0.26),
                (l.bars.0.x - 0.08, l.bars.0.y - 0.3, l.bars.0.x + l.bars.1.x, l.bars.0.y + l.bars.1.y + 0.3),
            ];
            for (i, a) in rects.iter().enumerate() {
                assert!(a.2 - a.0 > 0.4 && a.3 - a.1 > 0.4, "w={w}: part {i} has no size: {a:?}");
                assert!(a.0 >= -w * 0.5 - 1e-3 && a.2 <= w * 0.5 + 1e-3 && a.1 >= -2.5 + 0.25 && a.3 <= 2.5 - 0.05, "w={w}: part {i} leaves the canvas: {a:?}");
                for (j, b) in rects.iter().enumerate().skip(i + 1) {
                    let apart = a.2 <= b.0 + 1e-3 || b.2 <= a.0 + 1e-3 || a.3 <= b.1 + 1e-3 || b.3 <= a.1 + 1e-3;
                    assert!(apart, "w={w}: parts {i} and {j} overlap: {a:?} {b:?}");
                }
            }
        }
    }

    /// What the panel says about the length is what the length does: the reviewer's
    /// cases, which used to be called settled (ReLU just above √2 grew too slowly to
    /// pass the cap in 120 layers; tanh with small weights and no bias was held at
    /// the floor), and a slow approach to a true fixed point.
    #[test]
    fn lengths_that_do_not_settle_are_called_what_they_are() {
        let mut g = Gauss::new();
        for sw in [1.415f32, 1.42, 1.44, 1.45, 1.46] {
            for sb in [0.0f32, 0.3] {
                assert_eq!(settled_length(&mut g, p(Act::Relu, sw, sb)).1, Fate::Grows, "ReLU sw={sw} sb={sb}");
            }
        }
        for sw in [1.38f32, 1.40, 1.41, 1.414] {
            assert_eq!(settled_length(&mut g, p(Act::Relu, sw, 0.0)).1, Fate::Dies, "ReLU sw={sw}");
        }
        // with a bias the same weights settle, at sb² / (1 - sw²/2), slowly
        for (sw, exact) in [(1.40f32, 4.5f32), (1.41, 0.09 / (1.0 - 1.41 * 1.41 / 2.0))] {
            let (q, fate) = settled_length(&mut g, p(Act::Relu, sw, 0.3));
            assert!(fate == Fate::Settles && (q - exact).abs() < 0.01 * exact, "ReLU sw={sw} sb=0.3: {fate:?} at {q}, not {exact}");
        }
        // the edge itself keeps its length, as the experiment says
        let (q, fate) = settled_length(&mut g, p(Act::Relu, std::f32::consts::SQRT_2, 0.0));
        assert!(fate == Fate::Settles && (q - 1.0).abs() < 1e-3);
        // tanh without biases: small weights let the signal die (and quickly: well
        // before the 120 layers it is followed for)
        assert_eq!(settled_length(&mut g, p(Act::Tanh, 0.9, 0.0)), (GONE, Fate::Dies));
        assert_eq!(settled_length(&mut g, p(Act::Tanh, 1.0, 0.0)).1, Fate::Dies);
        let (q, fate) = settled_length(&mut g, p(Act::Tanh, 1.05, 0.0));
        assert!(fate == Fate::Settles && (q - 0.0533).abs() < 1e-3, "tanh 1.05: {fate:?} at {q}");
    }

    /// The slope written in the square is the slope of the curve drawn there: the
    /// same as the factor for gradients where the length settles, and not where it
    /// does not (tanh 0.9 without bias draws the diagonal, slope 1, while gradients
    /// shrink by 0.81 a layer). (Smooth activations only: ReLU's map has a kink-like
    /// end at c = 1 that a difference quotient approaches too slowly to check this way.)
    #[test]
    fn the_slope_shown_is_the_slope_drawn() {
        let mut g = Gauss::new();
        for (act, sw, sb) in [(Act::Tanh, 0.9f32, 0.0f32), (Act::Tanh, 1.6, 0.3), (Act::Tanh, 2.5, 0.3), (Act::Hermite2, 1.6, 0.3), (Act::Sin, 1.5, 0.3)] {
            let params = p(act, sw, sb);
            let (q, fate) = settled_length(&mut g, params);
            let h = 0.005;
            let drawn = (next_correlation(&mut g, params, q, 1.0) - next_correlation(&mut g, params, q, 1.0 - h)) / h;
            let shown = map_slope(&mut g, params, q);
            assert!((shown - drawn).abs() < 0.01 * shown, "{params:?} ({fate:?}): shown {shown}, drawn {drawn}");
            if fate == Fate::Settles {
                assert!((shown - slope_at_one(&mut g, params, q)).abs() < 1e-3 * shown);
            }
        }
        let (q, _) = settled_length(&mut g, p(Act::Tanh, 0.9, 0.0));
        assert!((map_slope(&mut g, p(Act::Tanh, 0.9, 0.0), q) - 1.0).abs() < 1e-3);
        assert!((slope_at_one(&mut g, p(Act::Tanh, 0.9, 0.0), q) - 0.81).abs() < 1e-3);
    }

    /// Where lengths grow without bound the test network starts at the length cap
    /// and grows by sw²/2 a layer. Its correlations are still measured (in f32 the
    /// product of the squared lengths overflowed from layer 20, which read as 0).
    #[test]
    fn a_network_whose_lengths_grow_is_still_measured() {
        let mut g = Gauss::new();
        let params = p(Act::Relu, 3.0, 0.3);
        let (q, fate) = settled_length(&mut g, params);
        assert_eq!(fate, Fate::Grows);
        for seed in 1..=3 {
            let mut probe = Probe::new(512, q, 0.97, seed);
            let mut c = 0.97;
            for layer in 1..=24 {
                c = next_correlation(&mut g, params, q, c);
                probe.layer(params);
                let seen = probe.seen[layer];
                assert!((seen - c).abs() < 0.02, "seed {seed} layer {layer}: measured {seen}, theory {c}");
            }
        }
        // a network whose numbers leave the range of f32 altogether measures nothing
        // (and the scene draws no dot there), rather than a false 0
        let params = p(Act::Hermite2, 1.6, 0.3);
        let (q, _) = settled_length(&mut g, params);
        let mut probe = Probe::new(512, q, 0.97, 1);
        for _ in 0..12 {
            probe.layer(params);
        }
        assert!(probe.seen[..5].iter().all(|c| c.is_finite() && c.abs() <= 1.0), "{:?}", probe.seen);
        assert!(probe.seen[5..].iter().all(|c| c.is_nan()), "{:?}", probe.seen);
    }

    /// A layer of the test network worked out a block of rows at a time (so that a
    /// wide one fits in a frame) is the same layer.
    #[test]
    fn a_layer_in_parts_is_the_same_layer() {
        let params = p(Act::Tanh, 1.6, 0.3);
        let (mut whole, mut parts) = (Probe::new(100, 1.2, 0.5, 4), Probe::new(100, 1.2, 0.5, 4));
        for _ in 0..3 {
            whole.layer(params);
        }
        while parts.seen.len() < 4 {
            parts.rows(params, 7);
        }
        assert_eq!(whole.seen, parts.seen);
        assert_eq!((whole.a, whole.b), (parts.a, parts.b));
    }
}


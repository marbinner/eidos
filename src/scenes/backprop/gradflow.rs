//! Gradients through depth: vanishing and exploding.
//!
//! A plain deep network, built and measured with `math::arch`: the size of the signal
//! at every layer and of the gradient that comes back to it, beside the prediction for
//! infinitely wide layers; the singular values of the Jacobian of the whole stack; and
//! a short run of training at the present setting.
//!
//! The prediction (the "mean field"): a unit of a layer whose weights have variance
//! σ²/n receives a sum of n terms, which for a wide layer is a bell of mean 0 and mean
//! square q = σ² × (the mean square of what the layer reads). The layer's output then
//! has mean square E[φ(√q Z)²], and on the way back the gradient's mean square is
//! multiplied by σ² E[φ'(√q Z)²] (Z a standard normal number). Both are worked out
//! example by example, the forward one from the data and the backward one from the
//! gradient measured at the last layer.

use super::*;
use crate::clock::{Budget, Pace, Rest};
use crate::math::arch::{spirals, Init, Kind, Loss, MlpSpec, Model, Norm, Place, Stage};
use crate::math::autograd::{Act, Tape, Var};
use crate::scenes::stats::plot::{self, Glide, Plot, Rect, AXES, BUTTONS, OWN};

pub const DEF: SceneDef = SceneDef {
    id: "gradflow",
    title: "Gradients through depth: vanishing and exploding",
    group: GROUP,
    blurb: "Fifty layers multiply fifty Jacobians: how the scale of the weights and the activation decide whether the gradient dies, explodes or arrives whole.",
    make: |_| Box::new(Gradflow::create()),
};

// ----- the network and what is measured on it ------------------------------------------

/// The activations offered.
const ACTS: [Act; 3] = [Act::Relu, Act::Tanh, Act::Identity];
/// The starting schemes offered. (Xavier's is LeCun's for the square layers here.)
const INITS: [Init; 3] = [Init::Lecun, Init::He, Init::Orthogonal];
/// The widths offered.
const WIDTHS: [usize; 4] = [8, 16, 32, 64];
/// The shallowest and deepest networks. (The stack whose Jacobian is shown is layers
/// 2 to the last: it needs two.)
const LEAST_DEPTH: usize = 2;
const MOST_DEPTH: usize = 50;
/// The range of the gain.
const LEAST_GAIN: f64 = 0.25;
const MOST_GAIN: f64 = 4.0;
/// Examples in the batch whose signals and gradients are measured, and trained on.
const BATCH: usize = 16;
/// The seed of the data, and of the weights the scene opens with.
const DATA_SEED: u64 = 7;
const OPENING_SEED: u64 = 2;
/// A run of training: steps, steps a second of each run, and the most runs kept.
const STEPS: usize = 300;
const RATE: f32 = 150.0;
const KEPT_RUNS: usize = 4;
/// The length the gradient is clipped to, when it is.
const CLIP: f64 = 1.0;
/// A singular value below this share of the largest cannot be told from rounding
/// error in double precision (`linalg::svd` makes those below 1e-13 of it zero).
const RESOLVED: f64 = 1e-12;

/// Depth, width, activation, scheme and seed: what decides a draw of the weights at
/// gain 1.
type Draw = (usize, usize, Act, Init, u64);

/// What decides the starting network.
#[derive(Clone, Copy, Debug, PartialEq)]
struct Setting {
    depth: usize,
    width: usize,
    act: Act,
    init: Init,
    gain: f64,
    /// The draw of the weights.
    seed: u64,
}

impl Setting {
    /// The same with every number in its range (a NaN gain taken as 1).
    fn kept(self) -> Setting {
        let gain = if self.gain.is_nan() { 1.0 } else { self.gain.max(LEAST_GAIN).min(MOST_GAIN) };
        Setting { depth: self.depth.max(LEAST_DEPTH).min(MOST_DEPTH), width: self.width.max(1).min(64), gain, ..self }
    }

    fn spec(&self) -> MlpSpec {
        MlpSpec {
            inputs: 2,
            width: self.width,
            depth: self.depth,
            outputs: 3,
            act: self.act,
            norm: Norm::None,
            place: Place::Pre,
            residual: false,
            residual_scale: 1.0,
            init: self.init,
            gain: self.gain,
            bias: true,
        }
    }

    /// σ² of a layer: its number of inputs times the variance of one of its weights.
    /// Then what a unit receives has σ² times the mean square of what the layer reads.
    fn sigma2(&self, fan_in: usize, fan_out: usize) -> f64 {
        fan_in as f64 * self.init.variance(fan_in, fan_out) * self.gain * self.gain
    }

    /// σ² of the hidden layers after the first (square: width in, width out).
    fn hidden_sigma2(&self) -> f64 {
        self.sigma2(self.width, self.width)
    }

    /// The gain at which the square layers keep the gradient's mean square: σ² = 2 for
    /// ReLU, whose average squared slope is 1/2; σ² = 1 for the identity, and for tanh
    /// at a small signal, where its slope is 1.
    fn critical_gain(&self) -> f64 {
        let wanted = if self.act == Act::Relu { 2.0 } else { 1.0 };
        let at_one = self.width as f64 * self.init.variance(self.width, self.width);
        (wanted / at_one).sqrt()
    }

    /// What tells the starting weights apart, but for the gain: networks that differ
    /// only in their gain are one network scaled.
    fn draw(&self) -> Draw {
        (self.depth, self.width, self.act, self.init, self.seed)
    }
}

/// The mean square of what an activation gives, and of its slope, for an input drawn
/// from a bell of mean 0 and mean square q: E[φ(√q Z)²] and E[φ'(√q Z)²]. Exact for
/// ReLU (q/2 and 1/2) and the identity (q and 1); for tanh from a table.
fn moments(act: Act, q: f64) -> (f64, f64) {
    match act {
        Act::Relu => (0.5 * q, 0.5),
        Act::Identity => (q, 1.0),
        _ => tanh_table().at(q),
    }
}

/// E[tanh²(√q Z)] and E[sech⁴(√q Z)] tabulated against ln q (as their logarithms), read
/// between the entries by cubic interpolation, and by their series beyond the table's
/// ends.
struct TanhTable {
    square: Vec<f64>,
    slope: Vec<f64>,
}

const TABLE_LO: f64 = -18.0;
const TABLE_HI: f64 = 18.0;
const TABLE_N: usize = 721;
/// Entries beyond each end, so that the cubic has its neighbours there too.
const TABLE_PAD: usize = 2;

fn tanh_table() -> &'static TanhTable {
    static TABLE: std::sync::OnceLock<TanhTable> = std::sync::OnceLock::new();
    TABLE.get_or_init(TanhTable::new)
}

impl TanhTable {
    fn step() -> f64 {
        (TABLE_HI - TABLE_LO) / (TABLE_N - 1) as f64
    }

    fn new() -> TanhTable {
        let n = TABLE_N + 2 * TABLE_PAD;
        let (mut square, mut slope) = (Vec::with_capacity(n), Vec::with_capacity(n));
        for i in 0..n {
            let t = TABLE_LO + Self::step() * (i as f64 - TABLE_PAD as f64);
            let (m, s) = tanh_moments_direct(t.exp());
            // (their logarithms: nearly straight in ln q at both ends, so that the cubic
            // between the entries keeps their digits there)
            square.push(m.ln());
            slope.push(s.ln());
        }
        TanhTable { square, slope }
    }

    fn at(&self, q: f64) -> (f64, f64) {
        if q.is_nan() {
            return (f64::NAN, f64::NAN);
        }
        if q <= 0.0 {
            return (0.0, 1.0);
        }
        let t = q.ln();
        if t < TABLE_LO {
            // tanh² x = x² − 2x⁴/3 + …, sech⁴ x = 1 − 2x² + 5x⁴/3 − …, and E Z⁴ = 3
            return (q - 2.0 * q * q, 1.0 - 2.0 * q + 5.0 * q * q);
        }
        if t > TABLE_HI {
            // so wide a bell is flat where tanh turns: ∫ sech² = 2 and ∫ sech⁴ = 4/3,
            // times the bell's height there, 1/√(2πq)
            let h = 1.0 / (std::f64::consts::TAU * q).sqrt();
            return (1.0 - 2.0 * h, 4.0 / 3.0 * h);
        }
        let x = (t - TABLE_LO) / Self::step() + TABLE_PAD as f64;
        let last = TABLE_N + 2 * TABLE_PAD - 1;
        let i = (x.floor().max(1.0) as usize).min(last - 2);
        let u = x - i as f64;
        let cubic = |v: &[f64]| {
            let p = |k: isize| v[(i as isize + k).max(0).min(last as isize) as usize];
            let (a, b, c, d) = (p(-1), p(0), p(1), p(2));
            // Catmull-Rom: the cubic through b and c with slopes (c − a)/2 and (d − b)/2
            b + 0.5 * u * (c - a + u * (2.0 * a - 5.0 * b + 4.0 * c - d + u * (3.0 * (b - c) + d - a)))
        };
        (cubic(&self.square).exp(), cubic(&self.slope).exp())
    }
}

/// E[tanh²(√q Z)] and E[sech⁴(√q Z)] by the trapezoid rule, in the variable u = √q Z
/// (so that a very wide bell, where sech is a narrow bump, is sampled as finely as a
/// narrow one): u from −R to R, R eight standard deviations or 40, whichever is less
/// (beyond 40, sech² is below 1e-34). The integrands are smooth, for which the
/// trapezoid rule converges very fast: 400 steps are good to rounding.
fn tanh_moments_direct(q: f64) -> (f64, f64) {
    let sd = q.sqrt();
    let reach = (8.0 * sd).min(40.0);
    let n = 400;
    let h = 2.0 * reach / n as f64;
    let (mut mass, mut tanh2, mut sech2, mut sech4) = (0.0, 0.0, 0.0, 0.0);
    for k in 0..=n {
        let u = -reach + h * k as f64;
        let w = (-0.5 * (u / sd).powi(2)).exp() * if k == 0 || k == n { 0.5 } else { 1.0 };
        // tanh |u| = (1 − e)/(1 + e) and sech² u = 4e/(1 + e)², e = exp(−2|u|), from one
        // exponential and without cancellation
        let e = (-2.0 * u.abs()).exp();
        let (t, s2) = ((1.0 - e) / (1.0 + e), 4.0 * e / ((1.0 + e) * (1.0 + e)));
        mass += w;
        tanh2 += w * t * t;
        sech2 += w * s2;
        sech4 += w * s2 * s2;
    }
    if reach < 8.0 * sd {
        // a bell wider than ±40: tanh² is 1 beyond, so E tanh² = 1 − E sech², and the
        // shares are of the bell's full mass, √(2π) sd / h in these units
        let mass = std::f64::consts::TAU.sqrt() * sd / h;
        return (1.0 - sech2 / mass, sech4 / mass);
    }
    // (the whole bell inside: tanh² summed itself, which keeps its digits when small)
    (tanh2 / mass, sech4 / mass)
}

/// One network as measured at its start.
#[derive(Clone, Debug, Default)]
struct Measured {
    /// The root mean square of each layer's output over the batch: layer 0 (the data)
    /// to the last hidden layer.
    forward: Vec<f64>,
    /// The root mean square of the gradient of the loss with respect to each layer's
    /// output: layers 1 to the last (entry 0 is layer 1).
    backward: Vec<f64>,
    /// The predictions for infinitely wide layers, in the same places.
    predicted_forward: Vec<f64>,
    predicted_backward: Vec<f64>,
    /// The prediction for the root mean square of the stack's singular values: the
    /// gradient's factor from the last layer back to the first, for the first example.
    predicted_stack: f64,
    /// The loss at the start.
    loss: f64,
    /// For ReLU: the fewest units open for the first example in a layer of the stack
    /// (layers 2 to the last), and that layer.
    fewest_open: Option<(usize, usize)>,
}

impl Measured {
    fn depth(&self) -> usize {
        self.backward.len()
    }

    /// The fitted factors a layer, forward and backward (see [`fitted_factor`]).
    fn factors(&self) -> (Option<f64>, Option<f64>) {
        let forward = self.forward.get(1..).and_then(fitted_factor);
        let backward = fitted_factor(&self.backward).map(|k| 1.0 / k);
        (forward, backward)
    }

    /// The same for the predictions: the factor a layer, on average over the layers.
    fn predicted_factors(&self) -> (Option<f64>, Option<f64>) {
        let forward = self.predicted_forward.get(1..).and_then(fitted_factor);
        let backward = fitted_factor(&self.predicted_backward).map(|k| 1.0 / k);
        (forward, backward)
    }

    /// What reaches the first layer, as a share of what the last layer got: measured
    /// and predicted.
    fn left(&self) -> (f64, f64) {
        let share = |v: &[f64]| match (v.first(), v.last()) {
            (Some(a), Some(b)) if v.len() > 1 => a / b,
            _ => f64::NAN,
        };
        (share(&self.backward), share(&self.predicted_backward))
    }
}

/// The data: `BATCH` points of the spirals, three arms.
fn data() -> (Vec<f64>, Vec<f64>) {
    spirals(BATCH, 3, &mut Rng::new(DATA_SEED))
}

/// The loss of a network that ignores its input and gives each class its share of the
/// data: the cross-entropy of the classes' frequencies.
fn guess_loss(y: &[f64]) -> f64 {
    let n = y.len().max(1) as f64;
    (0..3).map(|c| y.iter().filter(|&&t| t == c as f64).count() as f64 / n).filter(|&p| p > 0.0).map(|p| -p * p.ln()).sum()
}

/// The starting network of a setting, drawn at gain 1 (see [`scaled`]).
fn build_base(s: &Setting) -> Model {
    Model::mlp(&Setting { gain: 1.0, ..*s }.spec(), &mut Rng::new(s.seed))
}

/// A network drawn at gain 1, with every weight matrix times `gain`: the network the
/// library draws at that gain (to rounding), without drawing it again.
fn scaled(base: &Model, gain: f64) -> Model {
    let mut model = base.clone();
    for g in model.groups_mut().iter_mut().filter(|g| g.kind == Kind::Weight) {
        g.value.iter_mut().for_each(|v| *v *= gain);
    }
    model
}

/// Set `model`'s weight matrices to `base`'s times `gain` (the same network, drawn at
/// gain 1, scaled), without allocating.
fn rescale(model: &mut Model, base: &Model, gain: f64) {
    for (g, b) in model.groups_mut().iter_mut().zip(base.groups()) {
        if g.kind == Kind::Weight {
            for (v, w) in g.value.iter_mut().zip(&b.value) {
                *v = w * gain;
            }
        }
    }
}

/// Measure a network: one forward and backward pass of the cross-entropy loss on the
/// data, and the predictions.
fn measure(s: &Setting, model: &Model, tape: &mut Tape, x: &[f64], y: &[f64]) -> Measured {
    let trace = model.loss_mlp(tape, x, y, Loss::CrossEntropy);
    tape.backward(trace.loss);
    let depth = s.depth;
    let mut m = Measured { loss: tape.value(trace.loss).first().copied().unwrap_or(f64::NAN), ..Measured::default() };
    // (what `arch::Probe` gives as `rms` and `grad_rms`, for these stages alone)
    let rms = |v: &[f64]| if v.is_empty() { f64::NAN } else { (v.iter().map(|x| x * x).sum::<f64>() / v.len() as f64).sqrt() };
    for l in 0..=depth {
        let var = trace.get(if l == 0 { Stage::Input } else { Stage::Output }, l);
        m.forward.push(var.map_or(f64::NAN, |v| rms(tape.value(v))));
        if l > 0 {
            m.backward.push(var.map_or(f64::NAN, |v| rms(tape.grad(v))));
        }
    }
    // the units open for the first example, layer by layer, for ReLU
    if s.act == Act::Relu {
        m.fewest_open = (2..=depth)
            .filter_map(|l| {
                let z = tape.value(trace.get(Stage::PreAct, l)?);
                Some((z.iter().take(s.width).filter(|&&v| v > 0.0).count(), l))
            })
            .min();
    }
    // the predictions, example by example: the forward one from the data, the backward
    // one from the gradient measured at the last layer
    let top = trace.get(Stage::Output, depth).map(|v| tape.grad(v).to_vec()).unwrap_or_default();
    let (first, hidden) = (s.sigma2(2, s.width), s.hidden_sigma2());
    let n = x.len() / 2;
    let w = s.width;
    let mut fwd = vec![0.0; depth + 1];
    let mut back = vec![0.0; depth];
    let mut slopes = vec![0.0; depth + 1];
    for e in 0..n {
        let ms0 = 0.5 * (x[2 * e] * x[2 * e] + x[2 * e + 1] * x[2 * e + 1]);
        fwd[0] += ms0;
        let mut ms = ms0;
        for l in 1..=depth {
            // what layer l's units receive, and what they give
            let q = if l == 1 { first } else { hidden } * ms;
            let (square, slope) = moments(s.act, q);
            ms = square;
            slopes[l] = slope;
            fwd[l] += ms;
        }
        let mut g = if top.len() >= (e + 1) * w { top[e * w..(e + 1) * w].iter().map(|g| g * g).sum::<f64>() / w as f64 } else { f64::NAN };
        back[depth - 1] += g;
        let mut stack = 1.0;
        for l in (1..depth).rev() {
            // from layer l + 1's output back to layer l's: its gates, then its weights
            g *= slopes[l + 1] * hidden;
            stack *= slopes[l + 1] * hidden;
            back[l - 1] += g;
        }
        if e == 0 {
            m.predicted_stack = stack.sqrt();
        }
    }
    let n = n.max(1) as f64;
    m.predicted_forward = fwd.iter().map(|v| (v / n).sqrt()).collect();
    m.predicted_backward = back.iter().map(|v| (v / n).sqrt()).collect();
    m
}

/// The factor a step of a fit of a straight line to the logarithms of `values` against
/// their places (by least squares, over those finite and above 0; none if fewer than
/// two are).
fn fitted_factor(values: &[f64]) -> Option<f64> {
    let pts: Vec<(f64, f64)> = values.iter().enumerate().filter(|(_, v)| v.is_finite() && **v > 0.0).map(|(i, v)| (i as f64, v.ln())).collect();
    if pts.len() < 2 {
        return None;
    }
    let n = pts.len() as f64;
    let (mx, my) = (pts.iter().map(|p| p.0).sum::<f64>() / n, pts.iter().map(|p| p.1).sum::<f64>() / n);
    let sxx: f64 = pts.iter().map(|p| (p.0 - mx).powi(2)).sum();
    let sxy: f64 = pts.iter().map(|p| (p.0 - mx) * (p.1 - my)).sum();
    let k = (sxy / sxx).exp();
    (sxx > 0.0 && k.is_finite()).then_some(k)
}

// ----- the Jacobian of the stack and its singular values, a slice at a time ------------

/// The Jacobian of the last hidden layer's output with respect to layer 2's input (the
/// stack of square layers 2 to the last) for the first example, worked out a row at a
/// time as `arch::input_output_jacobian` works it out (one backward sweep per row,
/// seeded at one unit of the output: the same arithmetic, so the same numbers), and
/// then its singular values by one-sided Jacobi rotations, a sweep at a time: so that
/// the work, tens of milliseconds for 64 units, can be spread over frames.
struct SpectrumJob {
    setting: Setting,
    tape: Tape,
    input: Var,
    output: Var,
    rows: usize,
    cols: usize,
    next_row: usize,
    seed: Vec<f64>,
    jac: Vec<f64>,
    jacobi: Option<Jacobi>,
}

impl SpectrumJob {
    fn new(s: &Setting, model: &Model, x: &[f64]) -> Option<SpectrumJob> {
        let mut tape = Tape::new();
        let trace = model.forward_mlp(&mut tape, x.get(..2)?);
        let (input, output) = (trace.get(Stage::Input, 2)?, trace.get(Stage::Output, s.depth)?);
        let (rows, cols) = (tape.cols(output), tape.cols(input));
        let seed = vec![0.0; tape.value(output).len()];
        Some(SpectrumJob { setting: *s, tape, input, output, rows, cols, next_row: 0, seed, jac: vec![0.0; rows * cols], jacobi: None })
    }

    /// One more row of the Jacobian, or one more sweep of rotations. Returns the
    /// singular values once there is nothing more to do.
    fn step(&mut self) -> Option<Vec<f64>> {
        if self.next_row < self.rows {
            let k = self.next_row;
            self.seed[k] = 1.0;
            self.tape.backward_seeded(self.output, &self.seed);
            self.seed[k] = 0.0;
            self.jac[k * self.cols..(k + 1) * self.cols].copy_from_slice(&self.tape.grad(self.input)[..self.cols]);
            self.next_row += 1;
            return None;
        }
        let jacobi = self.jacobi.get_or_insert_with(|| Jacobi::new(&self.jac, self.rows, self.cols));
        if jacobi.done {
            return Some(jacobi.values());
        }
        jacobi.sweep();
        jacobi.done.then(|| jacobi.values())
    }

    /// Work until done or until the allowance is spent (at least one step). Returns the
    /// singular values when done.
    fn advance(&mut self, budget: &Budget) -> Option<Vec<f64>> {
        loop {
            if let Some(values) = self.step() {
                return Some(values);
            }
            if budget.spent() {
                return None;
            }
        }
    }

    /// The whole of it at once.
    fn finish(&mut self) -> Vec<f64> {
        loop {
            if let Some(values) = self.step() {
                return values;
            }
        }
    }
}

/// One-sided Jacobi rotations (Hestenes' method, as `linalg::svd` does them): the
/// columns of the matrix are rotated in pairs until they are at right angles to one
/// another; their lengths are then the singular values. One difference: a column
/// shorter than 1e-13 of the longest is left alone, being rounding error and nothing
/// else (rotating such columns against each other never converges, and took
/// `linalg::svd` all of its 60 sweeps, 90 ms for 64 columns); `linalg::svd` reports
/// them as zero, and so does this.
struct Jacobi {
    w: Vec<f64>,
    m: usize,
    n: usize,
    sweeps: usize,
    done: bool,
}

impl Jacobi {
    fn new(a: &[f64], m: usize, n: usize) -> Jacobi {
        // (columns are the short side: a wide matrix is worked on as its transpose)
        if m < n {
            let mut t = vec![0.0; m * n];
            for i in 0..m {
                for j in 0..n {
                    t[j * m + i] = a[i * n + j];
                }
            }
            return Jacobi { w: t, m: n, n: m, sweeps: 0, done: m == 0 };
        }
        Jacobi { w: a[..m * n].to_vec(), m, n, sweeps: 0, done: n == 0 }
    }

    fn norms2(&self) -> Vec<f64> {
        (0..self.n).map(|j| (0..self.m).map(|i| self.w[i * self.n + j].powi(2)).sum()).collect()
    }

    fn sweep(&mut self) {
        let (m, n) = (self.m, self.n);
        let longest = self.norms2().into_iter().fold(0.0f64, f64::max);
        let tiny = 1e-26 * longest;
        let w = &mut self.w;
        let mut off = 0.0f64;
        for p in 0..n {
            for q in p + 1..n {
                let (mut alpha, mut beta, mut gamma) = (0.0, 0.0, 0.0);
                for i in 0..m {
                    let (x, y) = (w[i * n + p], w[i * n + q]);
                    alpha += x * x;
                    beta += y * y;
                    gamma += x * y;
                }
                if alpha <= tiny || beta <= tiny || gamma == 0.0 || gamma.abs() <= 1e-15 * (alpha * beta).sqrt() {
                    continue;
                }
                off = off.max(gamma.abs() / (alpha * beta).sqrt().max(1e-300));
                let zeta = (beta - alpha) / (2.0 * gamma);
                let t = zeta.signum() / (zeta.abs() + (1.0 + zeta * zeta).sqrt());
                let c = 1.0 / (1.0 + t * t).sqrt();
                let s = c * t;
                for i in 0..m {
                    let (x, y) = (w[i * n + p], w[i * n + q]);
                    w[i * n + p] = c * x - s * y;
                    w[i * n + q] = s * x + c * y;
                }
            }
        }
        self.sweeps += 1;
        // (a matrix that is not finite never converges: 60 sweeps, as linalg::svd)
        self.done = off < 1e-13 || off.is_nan() || self.sweeps >= 60 || !longest.is_finite();
    }

    /// The singular values, largest first; those below 1e-13 of the largest are 0.
    fn values(&self) -> Vec<f64> {
        let mut s: Vec<f64> = self.norms2().into_iter().map(f64::sqrt).collect();
        s.sort_by(|a, b| b.total_cmp(a));
        let nothing = 1e-13 * s.first().copied().unwrap_or(0.0);
        s.iter_mut().filter(|v| **v <= nothing).for_each(|v| *v = 0.0);
        s
    }
}

// ----- training ---------------------------------------------------------------------------

/// A short run of plain gradient descent on the data from a setting's start.
struct Run {
    setting: Setting,
    lr: f64,
    clip: bool,
    model: Model,
    /// The loss before each step, and after the last.
    losses: Vec<f64>,
    /// Layer 1's and the last layer's weights at the start, to say how far they moved.
    first: Vec<f64>,
    last: Vec<f64>,
    /// The scene's seconds since the run began, and how many steps are drawn: as many
    /// as the rate has asked for by now, or as have been taken if fewer, followed
    /// through an ease (the steps themselves come two or three to a frame).
    clock: f32,
    shown: Glide,
}

impl Run {
    fn new(setting: Setting, model: Model, lr: f64, clip: bool) -> Run {
        let weights = |l: usize| model.group(&format!("layer {l} weights")).map(|g| g.value.clone()).unwrap_or_default();
        let (first, last) = (weights(1), weights(setting.depth));
        Run { setting, lr, clip, model, losses: Vec::with_capacity(STEPS + 1), first, last, clock: 0.0, shown: Glide::default() }
    }

    fn done(&self) -> bool {
        self.losses.len() > STEPS || self.losses.last().is_some_and(|l| !l.is_finite())
    }

    /// The loss now, and (unless the run is over) one step of gradient descent.
    fn step(&mut self, tape: &mut Tape, x: &[f64], y: &[f64]) {
        if self.done() {
            return;
        }
        let trace = self.model.loss_mlp(tape, x, y, Loss::CrossEntropy);
        let loss = tape.value(trace.loss).first().copied().unwrap_or(f64::NAN);
        self.losses.push(loss);
        if self.done() {
            return;
        }
        tape.backward(trace.loss);
        self.model.zero_grads();
        self.model.accumulate(tape, &trace);
        if self.clip {
            self.model.clip(CLIP);
        }
        self.model.step_sgd(self.lr);
    }

    /// How far layer 1's and the last layer's weights have moved, as shares of their
    /// length at the start.
    fn moved(&self) -> (f64, f64) {
        let share = |l: usize, start: &[f64]| {
            let Some(now) = self.model.group(&format!("layer {l} weights")) else { return f64::NAN };
            let gone = now.value.iter().zip(start).map(|(a, b)| (a - b).powi(2)).sum::<f64>().sqrt();
            gone / start.iter().map(|v| v * v).sum::<f64>().sqrt()
        };
        (share(1, &self.first), share(self.setting.depth, &self.last))
    }

    /// The step at which the loss stopped being a number, if it did.
    fn blew_up(&self) -> Option<usize> {
        self.losses.iter().position(|l| !l.is_finite())
    }
}

// ----- the experiments --------------------------------------------------------------------

/// One run an experiment starts: the scheme, the gain and whether the gradient is
/// clipped (the rest of its setting is the experiment's).
type RunAt = (Init, f64, bool);

struct Trial {
    button: &'static str,
    caption: &'static str,
    setting: Setting,
    /// The runs it starts, the last drawn bright.
    runs: &'static [RunAt],
    /// A scheme whose spectrum is shown faint beside this one's (the rest of the
    /// setting the experiment's).
    ghost: Option<Init>,
}

const fn at(depth: usize, act: Act, init: Init, gain: f64) -> Setting {
    Setting { depth, width: 32, act, init, gain, seed: OPENING_SEED }
}

const TRIALS: [Trial; 5] = [
    Trial {
        button: "too small a start",
        caption: r"Weights of variance $1/n$ (LeCun's start at gain 1), right for the identity. But ReLU passes on only the positive half of what its units receive, and on the way back about half its gates are shut: each layer halves the mean square, forward and back, $\chi = 1 \times \tfrac{1}{2}$. Both shrink by $\sqrt{1/2} = 0.707$ a layer, the signal going forward and the gradient going back (so the pink line falls towards layer 1); fitted to this network's points, by 0.682 and 0.684. Layer 1 gets $6.3 \times 10^{-9}$ of the gradient at layer 50 (the line: $4.2 \times 10^{-8}$). The white mark on the scale is the gain that keeps the size: 1.41.",
        setting: at(50, Act::Relu, Init::Lecun, 1.0),
        runs: &[],
        ghost: None,
    },
    Trial {
        button: "too large a start",
        caption: r"The same weights at gain 2: $\sigma^2 = 4$, $\chi = 2$, and each layer doubles the mean square. Both grow by $\sqrt{2} = 1.41$ a layer, the signal forward and the gradient back (fitted: 1.36 and 1.38), and layer 1 gets $6.3 \times 10^{6}$ times the gradient at layer 50. The outputs are so large that the loss starts at $2.5 \times 10^{6}$. Without clipping (faint), one step of gradient descent at rate 0.05 takes it to $4.8 \times 10^{245}$ and the next to no number at all. Clipped to length 1 (bright), every step is small, and in 300 steps the loss comes down to 115.",
        setting: at(50, Act::Relu, Init::Lecun, 2.0),
        runs: &[(Init::Lecun, 2.0, false), (Init::Lecun, 2.0, true)],
        ghost: None,
    },
    Trial {
        button: "the edge: He's start",
        caption: r"He's start draws variance $2/n$: $\sigma^2 = 2$ and $\chi = 2 \times \tfrac{1}{2} = 1$. The lines are level: with infinitely wide layers, every layer keeps both sizes. These layers of 32 keep a little less (fitted: 0.965 and 0.968 a layer), and layer 1 gets 0.15 of the gradient at layer 50. In a layer of finite width the size wanders from layer to layer, and typically drifts down: of 40 draws, 35 fitted below 1. Narrow layers wander more: with 8 units this draw's gradient falls by 0.89 a layer and layer 1 gets $7.2 \times 10^{-3}$; with 64 units, 0.97 and 0.30.",
        setting: at(50, Act::Relu, Init::He, 1.0),
        runs: &[],
        ghost: None,
    },
    Trial {
        button: "the same average, a wild spread",
        caption: r"A deep linear network (the identity: every gate open) at gain 1. Both kinds of weights keep the gradient's mean square, $\chi = 1$: Gaussian ones on average, orthogonal ones exactly. The stack's singular values, on the right, are another matter. Gaussian weights (grey): from 1.9 down to $5.8 \times 10^{-12}$, and 11 of the 32 smaller still; the largest alone makes 85 % of the sum of their squares, the three largest 99.9 %, so a gradient going in along a random direction comes out almost entirely along three. Orthogonal weights (pink): all 32 exactly 1, every direction kept. With $\tanh$ an orthogonal start keeps them between 0.82 and 0.95; with ReLU it does not help, as about half the gates are shut.",
        setting: at(50, Act::Identity, Init::Orthogonal, 1.0),
        runs: &[],
        ghost: Some(Init::Lecun),
    },
    Trial {
        button: "thirty layers: no, and yes",
        caption: r"Thirty ReLU layers, trained for 300 steps each (rate 0.05, clipped at 1). From LeCun's start (faint) the signal reaching layer 30 is $5.0 \times 10^{-6}$ of the data's, so the output hardly depends on the input: the loss settles at 1.095, what a guess of the classes' shares gives, and layer 1's weights move by $2.2 \times 10^{-5}$ of their length. From He's start (bright) the same thirty layers bring the loss down to $8.1 \times 10^{-4}$.",
        setting: at(30, Act::Relu, Init::He, 1.0),
        runs: &[(Init::Lecun, 1.0, true), (Init::He, 1.0, true)],
        ghost: None,
    },
];

/// The scene opens as the first experiment sets it up.
const OPENING: usize = 0;
/// The learning rate the experiments use.
const LR: f64 = 0.05;

// ----- the scene ----------------------------------------------------------------------

pub struct Gradflow {
    setting: Setting,
    lr: f32,
    clip: bool,
    x: Vec<f64>,
    y: Vec<f64>,
    tape: Tape,
    /// The last network drawn at gain 1, and the present start.
    base: Option<(Draw, Model)>,
    start: Option<(Setting, Model)>,
    measured: Option<(Setting, Measured)>,
    job: Option<SpectrumJob>,
    spectrum: Option<(Setting, Vec<f64>)>,
    /// The same network's spectrum with another scheme, drawn faint.
    ghost: Option<(Setting, Vec<f64>)>,
    /// After a costly recomputation, how long to wait before the next.
    rest: Rest,
    runs: Vec<Run>,
    pace: Pace,
    look: Look,
    /// The window last drawn in (a new one is drawn at once, without gliding).
    canvas: Option<Vec2>,
    dragging: bool,
}

impl Gradflow {
    /// (Nothing here needs the graphics device, so the tests can make one too.)
    fn create() -> Gradflow {
        let (x, y) = data();
        let mut s = Gradflow {
            setting: TRIALS[OPENING].setting,
            lr: LR as f32,
            clip: true,
            x,
            y,
            tape: Tape::new(),
            base: None,
            start: None,
            measured: None,
            job: None,
            spectrum: None,
            ghost: None,
            rest: Rest::default(),
            runs: Vec::new(),
            pace: Pace::default(),
            look: Look::default(),
            canvas: None,
            dragging: false,
        };
        s.set_up(OPENING);
        s
    }

    /// Set up one of the experiments in `TRIALS`: every setting, the faint spectrum it
    /// compares with, and the runs it starts.
    fn set_up(&mut self, k: usize) {
        let t = &TRIALS[k];
        self.lr = LR as f32;
        self.clip = true;
        self.runs.clear();
        self.ghost = None;
        if let Some(init) = t.ghost {
            let other = Setting { init, ..t.setting };
            let model = self.start_of(&other);
            if let Some(mut job) = SpectrumJob::new(&other, &model, &self.x) {
                self.ghost = Some((other, job.finish()));
            }
        }
        self.setting = t.setting;
        for &(init, gain, clip) in t.runs {
            let s = Setting { init, gain, ..t.setting };
            self.start_run(s, LR, clip);
        }
        self.refresh(None);
    }

    /// Make `start` the starting network of a setting: the base, drawn at gain 1 and
    /// kept, scaled (in place, when only the gain has changed).
    fn ready_start(&mut self, s: &Setting) {
        if self.base.as_ref().is_none_or(|(key, _)| *key != s.draw()) {
            self.base = Some((s.draw(), build_base(s)));
            self.start = None;
        }
        let Some((_, base)) = &self.base else { return };
        match &mut self.start {
            Some((old, model)) if old.draw() == s.draw() => {
                if old.gain != s.gain {
                    rescale(model, base, s.gain);
                }
                *old = *s;
            }
            _ => self.start = Some((*s, scaled(base, s.gain))),
        }
    }

    /// The starting network of a setting, a copy.
    fn start_of(&mut self, s: &Setting) -> Model {
        self.ready_start(s);
        match &self.start {
            Some((_, model)) => model.clone(),
            None => scaled(&build_base(s), s.gain),
        }
    }

    /// A run from the start of `s`, kept with the last few.
    fn start_run(&mut self, s: Setting, lr: f64, clip: bool) {
        let model = self.start_of(&s);
        self.runs.push(Run::new(s, model, lr, clip));
        if self.runs.len() > KEPT_RUNS {
            self.runs.remove(0);
        }
    }

    /// Bring what is measured up to date with the setting: the measurement at once (if
    /// the last costly one has had its rest), the spectrum a slice at a time within
    /// `budget` (all of it with none).
    fn refresh(&mut self, budget: Option<&Budget>) {
        self.setting = self.setting.kept();
        let s = self.setting;
        // a faint spectrum belongs to the same network with another scheme
        if self.ghost.as_ref().is_some_and(|(g, _)| Setting { init: s.init, ..*g } != s || g.init == s.init) {
            self.ghost = None;
        }
        let stale = self.measured.as_ref().is_none_or(|(m, _)| *m != s);
        let measuring = stale && (budget.is_none() || self.rest.over());
        if measuring {
            let began = crate::clock::now();
            self.ready_start(&s);
            if let Some((_, model)) = &self.start {
                let m = measure(&s, model, &mut self.tape, &self.x, &self.y);
                self.job = SpectrumJob::new(&s, model, &self.x);
                self.measured = Some((s, m));
            }
            self.rest = Rest::after(crate::clock::now() - began);
        }
        if let Some(job) = &mut self.job {
            let values = match budget {
                // (a measurement that used up this frame's time leaves the slices to the next)
                Some(b) if measuring && b.spent() => None,
                Some(b) => job.advance(b),
                None => Some(job.finish()),
            };
            if let Some(values) = values {
                self.spectrum = Some((job.setting, values));
                self.job = None;
            }
        }
    }

    /// The scheme changed by a choice of the visitor's: the spectrum shown so far stays,
    /// faint, for comparison.
    fn choose_init(&mut self, init: Init) {
        if init == self.setting.init {
            return;
        }
        if let Some((s, values)) = &self.spectrum {
            if *s == self.setting {
                self.ghost = Some((*s, values.clone()));
            }
        }
        self.setting.init = init;
    }

    /// A run at the present setting.
    fn train(&mut self) {
        self.start_run(self.setting, self.lr as f64, self.clip);
    }

    /// Training steps owed to the runs that are not over, within the frame's allowance.
    fn work(&mut self, dt: f32, real_dt: f32) {
        let active = self.runs.iter().filter(|r| !r.done()).count();
        if active == 0 || dt <= 0.0 {
            return;
        }
        for run in self.runs.iter_mut().filter(|r| !r.done()) {
            run.clock += dt;
        }
        let mut turn = self.pace.begin(RATE * active as f32, dt, real_dt, 0.004);
        let mut next = 0;
        while turn.next() {
            // the runs in turn, one step each
            let Some(i) = (0..self.runs.len()).map(|k| (next + k) % self.runs.len()).find(|&i| !self.runs[i].done()) else { break };
            self.runs[i].step(&mut self.tape, &self.x, &self.y);
            next = i + 1;
        }
        self.pace.end(turn);
    }

    /// Every run to its end at once (the tests of this file).
    #[cfg(test)]
    fn finish_runs(&mut self) {
        for run in &mut self.runs {
            while !run.done() {
                run.step(&mut self.tape, &self.x, &self.y);
            }
        }
    }

    fn measured(&self) -> Option<&Measured> {
        self.measured.as_ref().filter(|(s, _)| *s == self.setting).map(|(_, m)| m)
    }

    fn running(&self) -> bool {
        self.runs.iter().any(|r| !r.done())
    }
}

// ----- names ---------------------------------------------------------------------------

fn act_name(act: Act) -> &'static str {
    match act {
        Act::Relu => "ReLU",
        Act::Tanh => r"$\tanh$",
        _ => "identity",
    }
}

fn init_name(init: Init) -> &'static str {
    match init {
        Init::Lecun => "LeCun",
        Init::He => "He",
        Init::Xavier => "Xavier",
        Init::Orthogonal => "orthogonal",
    }
}

/// A positive number for a label: three figures, as a power of ten beyond a thousand
/// or below a hundredth.
fn sci(v: f64) -> String {
    if !v.is_finite() {
        return if v.is_nan() { "none".into() } else { r"$\infty$".into() };
    }
    if v == 0.0 {
        return "0".into();
    }
    let e = v.abs().log10().floor();
    if (-2.0..3.0).contains(&e) {
        let decimals = (2.0 - e).max(0.0) as usize;
        return format!("${}$", num(v, decimals));
    }
    let mut mant = v / 10f64.powf(e);
    let mut e = e as i32;
    if (mant * 10.0).round() / 10.0 >= 10.0 {
        mant /= 10.0;
        e += 1;
    }
    format!(r"${} \times 10^{{{}}}$", num(mant, 1), e)
}

/// A factor for a label: ×0.707.
fn times(k: Option<f64>) -> String {
    match k {
        Some(k) if k.is_finite() && k < 1000.0 => format!(r"$\times{}$", num(k, 3)),
        Some(k) if k.is_finite() => format!(r"$\times{}$", sci(k).trim_matches('$')),
        _ => "none".into(),
    }
}

/// A power of ten for an axis: 1, 10, or 10ⁿ.
fn decade(d: i32) -> String {
    match d {
        0 => "1".into(),
        1 => "10".into(),
        _ => format!("$10^{{{d}}}$"),
    }
}

// ----- the layout ----------------------------------------------------------------------

/// Where everything goes in a window.
#[derive(Clone, Copy, Debug)]
struct Layout {
    view: Vec2,
    narrow: bool,
    /// Text sizes: headings, words, small print.
    big: f32,
    text: f32,
    small: f32,
    /// The line saying what the network is.
    head: Vec2,
    /// The chart of the layers, and its key above it.
    layers: Rect,
    layers_key: Vec2,
    /// The chart of singular values, and the words above it (as wide as the chart).
    spectrum: Rect,
    spectrum_key: Vec2,
    /// The gain's scale (its two ends), its title (none on a phone), the scheme
    /// buttons and the button that draws the weights again.
    knob: (Vec2, Vec2),
    knob_title: Option<Vec2>,
    schemes: [Rect; 3],
    redraw: Rect,
    /// How to read the picture, where there is room for it.
    note: Rect,
    /// The chart of the loss, its title, and its buttons.
    loss: Rect,
    loss_key: Vec2,
    train: Rect,
    clip: Rect,
    /// Only the chart of the layers: the window is too small for the rest.
    compact: bool,
}

impl Layout {
    fn of(view: Vec2, inset: f32) -> Layout {
        let view = view.max(Vec2::splat(1.0));
        let narrow = view.x < 760.0;
        let (big, text, small) = if narrow { (13.0, 11.5, 10.5) } else { (15.0, 12.5, 11.5) };
        let side = if narrow { 10.0 } else { 22.0 };
        let top = inset.max(0.0) + if narrow { 8.0 } else { 14.0 };
        let r = |x0: f32, y0: f32, x1: f32, y1: f32| (Vec2::new(x0, y0), Vec2::new(x1.max(x0 + 1.0), y1.max(y0 + 1.0)));
        let none = r(0.0, 0.0, 0.0, 0.0);
        let compact = view.y - top < if narrow { 600.0 } else { 500.0 } || view.x < 300.0;
        let mut lay = Layout {
            view,
            narrow,
            big,
            text,
            small,
            head: Vec2::new(side, top),
            layers: none,
            layers_key: Vec2::ZERO,
            spectrum: none,
            spectrum_key: Vec2::ZERO,
            knob: (Vec2::ZERO, Vec2::ZERO),
            knob_title: None,
            schemes: [none; 3],
            redraw: none,
            note: none,
            loss: none,
            loss_key: Vec2::ZERO,
            train: none,
            clip: none,
            compact,
        };
        // (the key of the layers: three rows, then the y axis's name, then the chart)
        let key_rows = 3.0 * (small + if narrow { 5.0 } else { 6.0 });
        if compact {
            lay.narrow = true;
            lay.layers_key = Vec2::new(side, top + 22.0);
            lay.layers = r(side + 44.0, lay.layers_key.y + key_rows + 22.0, view.x - side - 10.0, view.y - 34.0);
            return lay;
        }
        let button_h = if narrow { 26.0 } else { 28.0 };
        let x1 = view.x - side;
        let h = view.y - top;
        if narrow {
            // the layers, the scale, the buttons, the spectrum, the loss: one under another
            lay.layers_key = Vec2::new(side, top + 22.0);
            let y0 = lay.layers_key.y + key_rows + 20.0;
            lay.layers = r(side + 40.0, y0, x1 - 6.0, y0 + h * 0.27);
            let knob_y = lay.layers.1.y + 58.0;
            lay.knob = (Vec2::new(side + 48.0, knob_y), Vec2::new(x1 - 22.0, knob_y));
            let row = plot::even_row(4, side, x1, knob_y + 28.0, button_h, 6.0);
            lay.schemes = [row[0], row[1], row[2]];
            lay.redraw = row[3];
            // (the spectrum's words: a title and two wrapped lines)
            lay.spectrum_key = Vec2::new(side, lay.redraw.1.y + 12.0);
            let y0 = lay.spectrum_key.y + (small + 5.0) + 2.0 * (small + 3.0) + 12.0;
            lay.spectrum = r(side + 40.0, y0, x1 - 6.0, y0 + h * 0.13);
            let row = plot::even_row(2, side, x1, view.y - 8.0 - button_h, button_h, 6.0);
            (lay.train, lay.clip) = (row[0], row[1]);
            lay.loss_key = Vec2::new(side, lay.spectrum.1.y + 26.0);
            lay.loss = r(side + 40.0, lay.loss_key.y + small + 14.0, x1 - 6.0, lay.train.0.y - 34.0);
        } else {
            // the layers and the spectrum side by side; under the layers the scale, the
            // buttons and how to read it, under the spectrum the loss
            let split = side + (view.x - 2.0 * side) * 0.60;
            lay.layers_key = Vec2::new(side, top + 28.0);
            // (above the charts: the layers' three rows, or the spectrum's title and
            // three wrapped lines, whichever is taller, then the y axes' names)
            let y0 = lay.layers_key.y + key_rows.max(text + 8.0 + 3.0 * (small + 3.0)) + 26.0;
            // under the charts, the loss needs its words, 80 pixels of chart, its axis
            // and its buttons: the charts above give way to it, down to 160 pixels
            let loss_needs = 48.0 + (text + 6.0 + 2.0 * (small + 3.0) + 14.0) + 80.0 + 38.0 + button_h + 12.0;
            let y1 = (top + h * 0.64).min(view.y - loss_needs).max(y0 + 160.0);
            lay.layers = r(side + 48.0, y0, split - 34.0, y1);
            lay.spectrum_key = Vec2::new(split + 6.0, top + 28.0);
            lay.spectrum = r(split + 52.0, y0, x1 - 10.0, y1);
            let row2 = y1 + 48.0;
            lay.knob_title = Some(Vec2::new(side, row2));
            let knob_y = row2 + 54.0;
            lay.knob = (Vec2::new(side + 48.0, knob_y), Vec2::new(split - 70.0, knob_y));
            let row = plot::even_row(4, side, split - 34.0, knob_y + 34.0, button_h, 8.0);
            lay.schemes = [row[0], row[1], row[2]];
            lay.redraw = row[3];
            lay.note = r(side, lay.redraw.1.y + 16.0, split - 34.0, view.y - 12.0);
            lay.loss_key = Vec2::new(split + 6.0, row2);
            let row = plot::even_row(2, split + 6.0, x1, view.y - 12.0 - button_h, button_h, 8.0);
            (lay.train, lay.clip) = (row[0], row[1]);
            lay.loss = r(split + 52.0, row2 + text + 6.0 + 2.0 * (small + 3.0) + 14.0, x1 - 10.0, lay.train.0.y - 38.0);
        }
        lay
    }
}

// ----- what is drawn, gliding -----------------------------------------------------------

// The numbers of the chart of layers, in the order the glide keeps them: the last
// layer, the range of the y axis (powers of ten), then for every layer 0 to the most
// the measured forward and backward sizes and the predicted ones (powers of ten).
const X_MAX: usize = 0;
const Y_LO: usize = 1;
const Y_HI: usize = 2;
const FWD: usize = 3;
const BACK: usize = FWD + MOST_DEPTH + 1;
const PFWD: usize = BACK + MOST_DEPTH + 1;
const PBACK: usize = PFWD + MOST_DEPTH + 1;
const LAYERS_LEN: usize = PBACK + MOST_DEPTH + 1;

/// Where a size that is not there (0, or not a number) is put on a chart: far below.
const NOTHING: f32 = -400.0;

#[derive(Clone, Debug, Default)]
struct Look {
    layers: Glide,
    /// The range of the spectrum's axis and the values (powers of ten).
    spectrum: Glide,
    /// The range of the loss's axis.
    loss: Glide,
    /// The gain and the critical gain (natural logarithms).
    knob: Glide,
    moving: bool,
}

/// A size as a power of ten for a chart, or `NOTHING`.
fn lg(v: f64) -> f32 {
    if v > 0.0 && v.is_finite() {
        (v.log10() as f32).max(-350.0).min(350.0)
    } else {
        NOTHING
    }
}

/// A range of powers of ten round the values that are there: a little beyond them, at
/// least `least` wide.
fn range_of(values: impl Iterator<Item = f32>, least: f32) -> (f32, f32) {
    let (mut lo, mut hi) = (f32::MAX, f32::MIN);
    for v in values.filter(|v| *v > NOTHING + 1.0) {
        lo = lo.min(v);
        hi = hi.max(v);
    }
    if lo > hi {
        return (-1.0, 1.0);
    }
    let (lo, hi) = (lo - 0.35, hi + 0.35);
    if hi - lo < least {
        let mid = 0.5 * (lo + hi);
        return (mid - 0.5 * least, mid + 0.5 * least);
    }
    (lo, hi)
}

/// The chart of layers' numbers for a measurement, in the order of `X_MAX` to `PBACK`.
fn layers_goal(m: &Measured) -> Vec<f32> {
    let depth = m.depth();
    let mut g = vec![0.0f32; LAYERS_LEN];
    // (beyond the last layer, the last layer's numbers: a deeper network grows from there)
    let fill = |g: &mut [f32], from: usize, values: &[f64], first: usize| {
        for l in 0..=MOST_DEPTH {
            let i = l.max(first).min(depth).saturating_sub(first);
            g[from + l] = values.get(i).map_or(NOTHING, |&v| lg(v));
        }
    };
    fill(&mut g, FWD, &m.forward, 0);
    fill(&mut g, BACK, &m.backward, 1);
    fill(&mut g, PFWD, &m.predicted_forward, 0);
    fill(&mut g, PBACK, &m.predicted_backward, 1);
    let shown = (0..=depth).flat_map(|l| [g[FWD + l], g[PFWD + l]]).chain((1..=depth).flat_map(|l| [g[BACK + l], g[PBACK + l]]));
    let (lo, hi) = range_of(shown, 2.0);
    // (a size of 0, below every power of ten, at the bottom: so that every number glides)
    g[FWD..].iter_mut().filter(|v| **v <= NOTHING + 1.0).for_each(|v| *v = lo);
    g[X_MAX] = depth as f32;
    g[Y_LO] = lo;
    g[Y_HI] = hi;
    g
}

/// How many of the singular values (largest first) can be told from rounding error.
fn resolved_count(v: &[f64]) -> usize {
    let top = v.first().copied().unwrap_or(0.0);
    v.iter().filter(|&&s| s > RESOLVED * top && s.is_finite()).count()
}

/// The spectrum's numbers: its axis's range and the level of the values' root mean
/// square, then the values and where each is across the chart, then the same for the
/// faint one if it is as long. Those too small to tell from rounding error are put on
/// the smallest that can be told, place and all: the line ends there, and as more can
/// be told (or fewer) they glide out of it (or into it).
fn spectrum_goal(values: &[f64], ghost: Option<&[f64]>) -> Vec<f32> {
    let shown = |v: &[f64]| -> Vec<f32> {
        let seen = resolved_count(v);
        let last = if seen > 0 { lg(v[seen - 1]) } else { 0.0 };
        let ys = v.iter().enumerate().map(|(i, &s)| if i < seen { lg(s) } else { last });
        let xs = (0..v.len()).map(|i| (i.min(seen.max(1) - 1) + 1) as f32);
        ys.chain(xs).collect()
    };
    let mine = shown(values);
    let other = ghost.filter(|g| g.len() == values.len()).map(shown).unwrap_or_default();
    let n = values.len();
    let ys = mine[..n].iter().chain(other.get(..n).unwrap_or_default());
    let (lo, hi) = range_of(ys.copied(), 1.5);
    let rms = (values.iter().map(|v| v * v).sum::<f64>() / n.max(1) as f64).sqrt();
    let level = if rms > 0.0 && rms.is_finite() { lg(rms) } else { lo };
    let mut g = vec![lo, hi, level];
    g.extend(mine);
    g.extend(other);
    g
}

/// Where the spectrum's values start in its glide (after the range and the level).
const SPEC_AT: usize = 3;

// ----- colours and kinds of line --------------------------------------------------------

/// The faint spectrum: the same network with the scheme before.
const GHOST: Color = pal::DIM;
/// The highest loss the chart shows (a power of ten).
const LOSS_TOP: f32 = 8.0;
/// The critical gain's mark.
const EDGE: Color = pal::WHITE;

/// A batch of lines each, for each series of the charts, the dashed level of the
/// spectrum, the scale, the guess, and each run (`RUNS` to `RUNS + KEPT_RUNS − 1`).
/// (Each keeps its number of points from frame to frame, so that `eidos motion` can
/// follow them point by point.)
const FWD_LINE: u8 = OWN;
const BACK_LINE: u8 = OWN + 1;
const SPEC_LINE: u8 = OWN + 2;
const GHOST_LINE: u8 = OWN + 3;
const LEVEL: u8 = OWN + 4;
const SCALE: u8 = OWN + 5;
const GUESS: u8 = OWN + 6;
const RUNS: u8 = OWN + 7;

/// Draw a point in every batch the scene uses at the start of every frame, so that no
/// batch drops out (which would move every later one up a place, and `eidos motion`
/// would take one batch for another).
fn hold_batches(f: &mut Frame) {
    let corner = f.hud(Vec2::ZERO);
    let kinds = [AXES, BUTTONS, FWD_LINE, BACK_LINE, SPEC_LINE, GHOST_LINE, SCALE];
    for kind in kinds.into_iter().chain((0..KEPT_RUNS as u8).map(|k| RUNS + k)) {
        plot::apart(f, kind, false).segment(corner, corner, 1.0, Color::CLEAR);
    }
    for kind in [LEVEL, GUESS] {
        plot::apart(f, kind, true).segment(corner, corner, 1.0, Color::CLEAR);
    }
    plot::disc(f, Vec2::ZERO, 1.0, Color::CLEAR);
}

impl Scene for Gradflow {
    fn view(&self) -> View {
        View::Flat { center: Vec2::ZERO, half_height: 1.0 }
    }

    fn settings(&self, s: &mut ViewSettings) {
        s.grid = false;
        s.bloom = s.bloom.min(0.2);
    }

    fn animated(&self) -> bool {
        self.look.moving || self.running() || self.job.is_some() || self.dragging || self.measured().is_none()
    }

    fn pointer_hint(&self) -> Option<&'static str> {
        Some("yellow dot on the scale: drag the gain    buttons on the picture: click")
    }

    fn ui(&mut self, ui: &mut egui::Ui, _: &Gfx) {
        ui::note(
            ui,
            r"A plain deep network, with no normalisation and no skip connections, hands the gradient back through one layer after another, and each multiplies it by its weights (transposed) and by its gates, the slopes of its activation. Fifty layers, fifty factors: unless each keeps the gradient's size on average, what reaches the first layers dies away or explodes, geometrically with depth.",
        );
        ui::note(
            ui,
            r"For weights of variance $\sigma^2/n$, $n$ the number of inputs of a unit, each layer multiplies the gradient's mean square by about",
        );
        ui::equation(ui, r"\chi = \sigma^2 \, \mathbb{E}\big[\phi'(z)^2\big],");
        ui::note(
            ui,
            r"$\sigma^2$ times the average squared slope of the activation $\phi$ at what its units receive. For ReLU that average is $\tfrac{1}{2}$, so $\sigma^2 = 2$ keeps the size (He's start): less and it dies, more and it explodes. The signal going forward has a factor of its own. Both are predictions for very wide layers, in which every draw of the weights behaves alike: the lines are that prediction, the points this network.",
        );

        ui::section(ui, "Things to try");
        for (k, trial) in TRIALS.iter().enumerate() {
            if ui::experiment(ui, trial.button, trial.caption) {
                self.set_up(k);
            }
        }

        ui::section(ui, "The network");
        let s = &mut self.setting;
        ui::slider_int(ui, "layers", &mut s.depth, LEAST_DEPTH..=MOST_DEPTH);
        let mut w = WIDTHS.iter().position(|&w| w == s.width).unwrap_or(2);
        if ui::choice(ui, &["8", "16", "32", "64"], &mut w) {
            s.width = WIDTHS[w];
        }
        ui::note(ui, "Units in each layer: the prediction is for very wide layers, and narrow ones scatter about it.");
        let mut a = ACTS.iter().position(|&a| a == s.act).unwrap_or(0);
        if ui::choice(ui, &ACTS.map(act_name), &mut a) {
            s.act = ACTS[a];
        }
        let mut i = INITS.iter().position(|&i| i == s.init).unwrap_or(0);
        if ui::choice(ui, &INITS.map(init_name), &mut i) {
            self.choose_init(INITS[i]);
        }
        let s = &mut self.setting;
        let mut gain = s.gain as f32;
        if ui::slider_log(ui, "gain", &mut gain, LEAST_GAIN as f32..=MOST_GAIN as f32) {
            s.gain = gain as f64;
        }
        ui::note(
            ui,
            r"The starting weights: LeCun's are drawn with variance $1/n$, He's with $2/n$, each times the gain squared; orthogonal ones are a random orthogonal matrix, which keeps every length, times the gain. So $\sigma^2$ is the gain squared, twice that for He's. The white mark on the scale is the gain where $\chi = 1$.",
        );
        if ui.button("draw the weights again").clicked() {
            self.setting.seed = self.setting.seed.wrapping_add(1);
        }

        ui::section(ui, "Training");
        ui::slider_log(ui, "learning rate", &mut self.lr, 0.001..=0.3);
        ui::checkbox(ui, &mut self.clip, "clip the gradient's length at 1");
        ui.horizontal(|ui| {
            if ui.button(format!("train {STEPS} steps")).clicked() {
                self.train();
            }
            if !self.runs.is_empty() && ui.button("clear the runs").clicked() {
                self.runs.clear();
            }
        });
        ui::note(ui, "Plain gradient descent on the 16 points: each step moves every weight against its gradient, times the rate. Clipped, a gradient longer than 1 is shortened to length 1 first.");

        ui::section(ui, "Reading the picture");
        ui::key(
            ui,
            &[
                (FORWARD, "cyan: the signal going forward, the root mean square of each layer's output over the 16 examples"),
                (BACKWARD, r"pink: the gradient coming back, the root mean square of $\partial L/\partial h_l$, the loss's gradient with respect to layer $l$'s output; on the right, the stack's singular values"),
                (pal::TEXT, "points: this network; lines: the prediction for infinitely wide layers"),
                (WEIGHT, r"yellow: the gain of the starting weights, on its scale"),
                (EDGE, r"white mark: the gain where $\chi = 1$"),
                (GHOST, "grey: the same network with the scheme chosen before"),
                (LOSS, "orange: the loss during training, the last run bright and earlier ones faint"),
            ],
        );
        ui::section(ui, "Words");
        ui::term(ui, "Mean square, root mean square:", r"the average of the squares of a vector's entries, and its square root: the entries' typical size, whatever their signs.");
        ui::term(ui, "Gate:", r"the slope $\phi'(z)$ of a unit's activation at what it received: on the way back the unit multiplies its gradient by it.");
        ui::term(ui, "Gain:", r"the factor on every starting weight; $\sigma^2$ is the variance of a weight times its number of inputs.");
        ui::term(ui, "LeCun, He, orthogonal:", r"three ways to draw the starting weights: variance $1/n$, variance $2/n$, a random orthogonal matrix. (Xavier's, $2/(n_\text{in} + n_\text{out})$, is LeCun's for these square layers.)");
        ui::term(ui, "Jacobian of the stack:", r"the matrix of how each output of the last layer moves with each input of layer 2. The gradient goes back through its transpose.");
        ui::term(ui, "Singular values:", r"how much a matrix stretches the directions it stretches most, next most, and so on. Their mean square is the average squared stretch of a random direction.");
        ui::term(ui, "Clipping:", "shortening the whole gradient to a set length when it is longer, before the step.");
        ui::term(ui, "Loss:", r"cross-entropy: minus the logarithm of the probability the network gives the right class, averaged over the examples.");

        ui::section(ui, "Numbers");
        let s = self.setting;
        ui::readout(ui, r"$\sigma^2$ of the square layers", num(s.hidden_sigma2(), 3));
        ui::readout(ui, r"gain where $\chi = 1$", num(s.critical_gain(), 3));
        // (the latest measured: while a control moves, a frame or two behind it)
        if let Some((_, m)) = &self.measured {
            let (f, b) = m.factors();
            let (pf, pb) = m.predicted_factors();
            ui::readout_colored(ui, "signal: factor a layer, fit (line)", format!("{} ({})", times(f), times(pf)), FORWARD);
            ui::readout_colored(ui, "gradient: factor a layer back, fit (line)", format!("{} ({})", times(b), times(pb)), BACKWARD);
            let (left, pleft) = m.left();
            ui::readout_colored(ui, "gradient at layer 1 over the last (line)", format!("{} ({})", sci(left), sci(pleft)), BACKWARD);
            ui::readout(ui, "loss at the start", sci(m.loss));
        }
        if let Some((_, v)) = &self.spectrum {
            let top = v.first().copied().unwrap_or(0.0);
            let seen = v.iter().filter(|&&x| x > RESOLVED * top).count();
            ui::readout(ui, "largest singular value", sci(top));
            ui::readout(ui, "smallest that can be resolved", sci(v.get(seen.max(1) - 1).copied().unwrap_or(0.0)));
            ui::readout(ui, "too small to resolve", format!("{} of {}", v.len() - seen, v.len()));
        }
        if let Some(run) = self.runs.last() {
            let (first, last) = run.moved();
            ui::readout_colored(ui, "last run: loss now", sci(run.losses.last().copied().unwrap_or(f64::NAN)), LOSS);
            ui::readout_colored(ui, "  layer 1's weights moved by", sci(first), LOSS);
            ui::readout_colored(ui, "  the last layer's by", sci(last), LOSS);
        }
        ui::see_also(
            ui,
            &[
                ("signal", "the forward maps of length and correlation, and the edge of chaos"),
                ("lyapunov", "products of many random matrices, and how their stretches spread"),
                ("normgrad", "normalisation: every layer brought back to one size"),
                ("residual", "skip connections: a straight road for the gradient"),
            ],
        );
    }

    fn frame(&mut self, f: &mut Frame) {
        hold_batches(f);
        let lay = Layout::of(f.view_size(), f.top_inset);
        // (a canvas squeezed to a sliver is one the controls have opened over, on a phone)
        let tiny = lay.view.x < 160.0 || lay.view.y < 160.0;
        let was_dragging = self.dragging;
        self.dragging = false;
        if !tiny && !lay.compact {
            self.buttons(f, &lay);
            self.drag(f, &lay);
        }
        let budget = crate::clock::frame_budget(f.real_dt, 0.004);
        self.refresh(Some(&budget));
        self.work(f.dt, f.real_dt);
        let snap = self.canvas != Some(lay.view);
        if snap && !tiny {
            self.canvas = Some(lay.view);
        }
        self.follow(f.real_dt, snap, was_dragging || self.dragging);
        if tiny {
            return;
        }
        self.draw_head(f, &lay);
        self.draw_layers(f, &lay);
        if !lay.compact {
            self.draw_knob(f, &lay);
            self.draw_note(f, &lay);
            self.draw_spectrum(f, &lay);
            self.draw_loss(f, &lay);
        }
    }
}

impl Gradflow {
    /// The buttons over the picture.
    fn buttons(&mut self, f: &mut Frame, lay: &Layout) {
        let size = lay.small;
        for (k, init) in INITS.into_iter().enumerate() {
            if plot::button(f, lay.schemes[k], init_name(init), size, self.setting.init == init) {
                self.choose_init(init);
            }
        }
        let (long, short) = ("draw the weights again", "new draw");
        let fits = text_width(long, size) + 16.0 <= lay.redraw.1.x - lay.redraw.0.x;
        if plot::button(f, lay.redraw, if fits { long } else { short }, size, false) {
            self.setting.seed = self.setting.seed.wrapping_add(1);
        }
        if plot::button(f, lay.train, &format!("train {STEPS} steps"), size, false) {
            self.train();
        }
        let clip = if self.clip { "clipped at 1" } else { "not clipped" };
        if plot::button(f, lay.clip, clip, size, self.clip) {
            self.clip = !self.clip;
        }
    }

    /// The gain, dragged along its scale.
    fn drag(&mut self, f: &mut Frame, lay: &Layout) {
        let (a, b) = lay.knob;
        if b.x - a.x < 20.0 {
            return;
        }
        let shown = self.look.knob.get(0) as f64;
        let gain = if self.look.knob.shown.is_empty() { self.setting.gain } else { shown.exp() };
        let at = Vec2::new(a.x + (b.x - a.x) * knob_share(gain), a.y);
        if let Some(to) = plot::screen_handle(f, 1, at, WEIGHT) {
            let t = ((to.x - a.x) / (b.x - a.x)).max(0.0).min(1.0) as f64;
            let gain = (LEAST_GAIN.ln() + t * (MOST_GAIN.ln() - LEAST_GAIN.ln())).exp();
            if gain.is_finite() {
                self.setting.gain = gain;
                self.dragging = true;
            }
        }
    }

    /// Bring the picture `dt` seconds nearer to what it shows. (`snap`: there at once.)
    fn follow(&mut self, dt: f32, snap: bool, held: bool) {
        let layers = self.measured.as_ref().map(|(_, m)| layers_goal(m));
        let spectrum = self.spectrum.as_ref().map(|(s, v)| {
            let ghost = self.ghost.as_ref().filter(|(g, _)| Setting { init: s.init, ..*g } == *s).map(|(_, v)| v.as_slice());
            spectrum_goal(v, ghost)
        });
        let (lo, hi) = self.loss_range();
        let knob = [self.setting.gain.ln() as f32, self.setting.critical_gain().ln() as f32];
        let look = &mut self.look;
        let mut moving = false;
        if let Some(goal) = layers {
            moving |= follow_or_snap(&mut look.layers, &goal, dt, snap);
        }
        if let Some(goal) = spectrum {
            // (a spectrum of another length is another picture: it is not glided to)
            let other = look.spectrum.shown.len() != goal.len();
            moving |= follow_or_snap(&mut look.spectrum, &goal, dt, snap || other);
        }
        moving |= follow_or_snap(&mut look.loss, &[lo, hi], dt, snap);
        // (the scale answers the hand at once)
        moving |= follow_or_snap(&mut look.knob, &knob, dt, snap || held);
        for run in &mut self.runs {
            let steps = run.losses.len().saturating_sub(1) as f32;
            let due = if run.done() { steps } else { steps.min(run.clock * RATE) };
            moving |= follow_or_snap(&mut run.shown, &[due], dt, snap);
        }
        look.moving = moving;
    }

    /// The range of the loss's axis (powers of ten): the guess, a loss of 0.001, where
    /// each run began, and all of each finished run. (A run going on is not followed,
    /// or the axis would move in fits as it reaches lower; what goes below waits at
    /// the edge until the run is over.)
    fn loss_range(&self) -> (f32, f32) {
        let guess = lg(guess_loss(&self.y));
        let values = self.runs.iter().flat_map(|r| {
            let all = if r.done() { r.losses.as_slice() } else { &r.losses[..r.losses.len().min(1)] };
            all.iter().map(|&l| lg(l))
        });
        let (lo, hi) = range_of(values.chain([guess, -3.0, 0.5]), 1.0);
        // (a run that blew up goes off the top: the rest would be squashed flat)
        (lo.max(-6.0), hi.min(LOSS_TOP))
    }

    fn draw_head(&self, f: &mut Frame, lay: &Layout) {
        let s = self.setting;
        let (act, init, gain, sigma2) = (act_name(s.act), init_name(s.init), num(s.gain, 2), num(s.hidden_sigma2(), 2));
        // the longest of these that fits across the window
        let texts = [
            format!(r"{} plain layers of {} units, {act}, on 16 points of the spirals; starting weights: {init} $\times {gain}$, so $\sigma^2 = {sigma2}$", s.depth, s.width),
            format!(r"{} plain layers of {} units, {act}; {init} $\times {gain}$, so $\sigma^2 = {sigma2}$", s.depth, s.width),
            format!(r"{} plain layers of {}, {act}; {init} $\times {gain}$", s.depth, s.width),
            format!(r"{} layers, {act}; {init} $\times {gain}$", s.depth),
        ];
        let room = lay.view.x - 2.0 * lay.head.x;
        // (`text_width` keeps what it measured: the longer texts, not drawn, are asked for
        // in every frame)
        let Some(text) = texts.iter().find(|t| text_width(t, lay.big) <= room).or(texts.last()) else { return };
        f.text_px(lay.head, text.as_str(), lay.big, pal::TEXT, Align2::LEFT_TOP);
    }

    /// The layers' chart as it is drawn now (its ranges gliding).
    fn layers_plot(&self, lay: &Layout) -> Option<Plot> {
        let g = &self.look.layers;
        if g.shown.len() < LAYERS_LEN {
            return None;
        }
        let p = Plot::new(lay.layers, [0.0, g.get(X_MAX) as f64], [g.get(Y_LO) as f64, g.get(Y_HI) as f64]);
        (!p.tiny()).then_some(p)
    }

    fn draw_layers(&self, f: &mut Frame, lay: &Layout) {
        let Some(p) = self.layers_plot(lay) else { return };
        let Some(m) = self.measured.as_ref().map(|(_, m)| m) else { return };
        let g = &self.look.layers;
        let depth = m.depth();
        p.backing(f, 46.0, 34.0, 8.0, 12.0);
        // the key, above the chart
        let (size, small) = (lay.text, lay.small);
        let (fit_f, fit_b) = m.factors();
        let (pred_f, pred_b) = m.predicted_factors();
        let (left, pleft) = m.left();
        let key = lay.layers_key;
        let row = small + if lay.narrow { 5.0 } else { 6.0 };
        let lines: [(Color, String); 3] = if lay.narrow {
            [
                (FORWARD, format!("signal, forward: {} a layer (line {})", times(fit_f), times(pred_f))),
                (BACKWARD, format!(r"gradient, back: {} a layer (line {})", times(fit_b), times(pred_b))),
                (BACKWARD, format!("layer 1 gets {} of layer {}'s (line {})", sci(left), depth, sci(pleft))),
            ]
        } else {
            [
                (FORWARD, format!(r"the signal going forward, each layer's output $h_l$: {} a layer (the line: {})", times(fit_f), times(pred_f))),
                (BACKWARD, format!(r"the gradient coming back, $\partial L/\partial h_l$: {} a layer (the line: {})", times(fit_b), times(pred_b))),
                (BACKWARD, format!("at layer 1 the gradient is {} of what it is at layer {} (the line: {})", sci(left), depth, sci(pleft))),
            ]
        };
        for (k, (color, text)) in lines.iter().enumerate() {
            let at = key + Vec2::new(0.0, k as f32 * row);
            if k < 2 {
                hud_rect(f, at + Vec2::new(0.0, 2.0), at + Vec2::new(small - 2.0, small), *color);
            }
            f.text_px(at + Vec2::new(small + 6.0, 0.0), text.as_str(), small, if k < 2 { pal::TEXT } else { pal::DIM }, Align2::LEFT_TOP);
        }
        f.text_px(Vec2::new(p.rect.0.x - 44.0, p.rect.0.y - 8.0), "root mean square (log scale)", small, pal::DIM, Align2::LEFT_BOTTOM);
        // the axes: whole layers along the bottom, powers of ten up the side
        let (min, max) = p.rect;
        plot::line_of(f, AXES, Vec2::new(min.x, max.y), max, 1.0, pal::DIM.a(0.8));
        plot::line_of(f, AXES, min, Vec2::new(min.x, max.y), 1.0, pal::DIM.a(0.8));
        let ticks = plot::Ticks::of(0.0, depth as f64, p.width() * 0.8);
        for v in ticks.values() {
            if v > p.x[1] + 1e-9 {
                continue;
            }
            let x = p.px_x(v);
            tick(f, Vec2::new(x, max.y), Vec2::new(x, max.y + 4.0), pal::DIM.a(0.8));
            f.text_px(Vec2::new(x, max.y + 6.0), ticks.name(v), size, pal::DIM, Align2::CENTER_TOP);
        }
        f.text_px(Vec2::new(max.x, max.y + 8.0 + size), "layer", size, pal::DIM, Align2::RIGHT_TOP);
        decade_axis(f, &p, size);
        // the predictions as lines, the measurements as points
        let line = |f: &mut Frame, kind: u8, from: usize, at: usize, color: Color| {
            let pts: Vec<Vec2> = (from..=depth).map(|l| inside(&p, l as f64, g.get(at + l) as f64)).collect();
            plot::path_of(f, kind, &pts, 1.6, color.a(0.75));
        };
        line(f, FWD_LINE, 0, PFWD, FORWARD);
        line(f, BACK_LINE, 1, PBACK, BACKWARD);
        let dot_size = if depth > 30 || lay.narrow { 5.5 } else { 7.0 };
        // (a size of exactly 0 has no place on a logarithmic axis: it waits at the
        // bottom, faint)
        let there = |v: Option<&f64>| v.is_some_and(|v| *v > 0.0 && v.is_finite());
        for l in 0..=depth {
            let faint = |c: Color, ok: bool| if ok { c } else { c.a(0.3) };
            dot_kept(f, &p, l as f64, g.get(FWD + l) as f64, dot_size, faint(FORWARD, there(m.forward.get(l))));
            if l > 0 {
                dot_kept(f, &p, l as f64, g.get(BACK + l) as f64, dot_size, faint(BACKWARD, there(m.backward.get(l - 1))));
            }
        }
    }

    fn draw_knob(&self, f: &mut Frame, lay: &Layout) {
        let (a, b) = lay.knob;
        if b.x - a.x < 20.0 || self.look.knob.shown.len() < 2 {
            return;
        }
        let size = lay.small;
        let above = if lay.knob_title.is_some() { 58.0 } else { 34.0 };
        hud_panel(f, a - Vec2::new(44.0, above), b + Vec2::new(36.0, 24.0));
        if let Some(at) = lay.knob_title {
            f.text_px(at, "the gain of the starting weights", lay.text, pal::TEXT, Align2::LEFT_TOP);
        } else {
            f.text_px(Vec2::new(a.x - 40.0, a.y), "gain", size, pal::TEXT, Align2::LEFT_CENTER);
        }
        let x_of = |gain: f64| a.x + (b.x - a.x) * knob_share(gain);
        plot::line_of(f, SCALE, a, b, 2.0, pal::DIM.a(0.7));
        for gain in [0.25, 0.5, 1.0, 2.0, 4.0] {
            let x = x_of(gain);
            plot::line_of(f, SCALE, Vec2::new(x, a.y - 4.0), Vec2::new(x, a.y + 4.0), 1.0, pal::DIM.a(0.8));
            f.text_px(Vec2::new(x, a.y + 7.0), num(gain, if gain < 1.0 { 2 } else { 0 }), size, pal::DIM, Align2::CENTER_TOP);
        }
        let gain = (self.look.knob.get(0) as f64).exp();
        let edge = (self.look.knob.get(1) as f64).exp();
        let xe = x_of(edge);
        let xg = x_of(gain);
        plot::line_of(f, SCALE, Vec2::new(xe, a.y - 12.0), Vec2::new(xe, a.y + 12.0), 2.0, EDGE);
        let s = &self.setting;
        let mark = format!(r"$\chi = 1$: {}", num(s.critical_gain(), 2));
        let label = format!(r"{}: $\sigma^2 = {}$", num(s.gain, 2), num(s.hidden_sigma2(), 2));
        // (the two labels go on the sides away from each other, so they never meet)
        let (mark_align, label_offset, label_align) = if xg <= xe {
            (Align2::LEFT_BOTTOM, Vec2::new(-9.0, -9.0), Align2::RIGHT_BOTTOM)
        } else {
            (Align2::RIGHT_BOTTOM, Vec2::new(9.0, -9.0), Align2::LEFT_BOTTOM)
        };
        f.text_px(Vec2::new(xe, a.y - 14.0), mark, size, EDGE, mark_align);
        plot::disc(f, Vec2::new(xg, a.y), 13.0, WEIGHT);
        f.text_px(Vec2::new(xg, a.y) + label_offset, label, size, WEIGHT, label_align);
    }

    /// How to read the picture, under the scale where there is room.
    fn draw_note(&self, f: &mut Frame, lay: &Layout) {
        let (min, max) = lay.note;
        if max.x - min.x < 200.0 || max.y - min.y < 60.0 {
            return;
        }
        let note = r"How to read it. The points are this network, measured on its 16 examples: the root mean square of each layer's output (cyan) and of the loss's gradient with respect to it (pink). The lines are the prediction for infinitely wide layers, started from the data and from the gradient at the last layer. On this logarithmic axis a fixed factor a layer is a straight line: tilted down, the size dies away with depth; tilted up, it explodes; level, it is kept. On the right: even where the average is kept, the stack may stretch a few directions of the gradient far more than the rest.";
        // (where the whole of it fits, or not at all)
        if wrapped_height_px(f, max.x - min.x, note, lay.small) <= max.y - min.y {
            wrapped_px(f, min, max.x - min.x, note, lay.small, pal::DIM);
        }
    }

    fn draw_spectrum(&self, f: &mut Frame, lay: &Layout) {
        let g = &self.look.spectrum;
        let Some((s, values)) = self.spectrum.as_ref() else { return };
        let n = values.len();
        if g.shown.len() < SPEC_AT + 2 * n || n == 0 {
            return;
        }
        let p = Plot::new(lay.spectrum, [0.5, n as f64 + 0.5], [g.get(0) as f64, g.get(1) as f64]);
        if p.tiny() {
            return;
        }
        let (size, small) = (lay.text, lay.small);
        p.backing(f, 48.0, 34.0, 8.0, 12.0);
        let key = lay.spectrum_key;
        let width = lay.spectrum.1.x - key.x + 4.0;
        let top = values[0];
        let seen = values.iter().filter(|&&v| v > RESOLVED * top).count();
        let rms = (values.iter().map(|v| v * v).sum::<f64>() / n as f64).sqrt();
        let title = if lay.narrow { "the stack's singular values, largest first".to_string() } else { format!("singular values of the Jacobian of layers 2 to {}", s.depth) };
        f.text_px(key, title, if lay.narrow { small } else { size }, pal::TEXT, Align2::LEFT_TOP);
        // what is too small to draw
        let rest = n - seen;
        let open = self.measured.as_ref().filter(|(ms, _)| ms == s).and_then(|(_, m)| m.fewest_open).filter(|_| s.act == Act::Relu && rest > 0);
        let unseen = match open {
            _ if seen == 0 && values[0] == 0.0 => match open {
                Some((0, layer)) => format!("all {n} are 0: layer {layer} has no unit open for this example, and passes nothing back"),
                _ => format!("all {n} are 0"),
            },
            Some((open, layer)) if open <= seen => format!("the other {rest} are 0: layer {layer} has only {open} units open for this example"),
            Some((open, _)) if lay.narrow => format!(r"{rest} more below $10^{{-12}}$ of the largest (at most {open} are not 0)"),
            Some((open, layer)) => format!(r"{rest} more below $10^{{-12}}$ of the largest; at most {open} are not 0, as layer {layer} has {open} units open for this example"),
            None if rest > 0 => format!(r"{rest} more below $10^{{-12}}$ of the largest"),
            None => String::new(),
        };
        let words = if lay.narrow {
            format!("dashed: their root mean square, {}", sci(rms))
        } else {
            format!("Largest first: how much the stack stretches each of {n} directions of the gradient. Dashed: their root mean square, {}.", sci(rms))
        };
        let at = key + Vec2::new(0.0, if lay.narrow { small + 5.0 } else { size + 8.0 });
        wrapped_px(f, at, width, &words, small, pal::DIM);
        if !lay.narrow {
            f.text_px(Vec2::new(p.rect.0.x - 44.0, p.rect.0.y - 8.0), "singular value", small, pal::DIM, Align2::LEFT_BOTTOM);
        }
        let (min, max) = p.rect;
        plot::line_of(f, AXES, Vec2::new(min.x, max.y), max, 1.0, pal::DIM.a(0.8));
        plot::line_of(f, AXES, min, Vec2::new(min.x, max.y), 1.0, pal::DIM.a(0.8));
        let ticks = plot::Ticks::of(1.0, n as f64, p.width() * 0.7);
        for v in [1.0].into_iter().chain(ticks.values().filter(|&v| v > 1.5)) {
            let x = p.px_x(v);
            tick(f, Vec2::new(x, max.y), Vec2::new(x, max.y + 4.0), pal::DIM.a(0.8));
            f.text_px(Vec2::new(x, max.y + 6.0), ticks.name(v), size, pal::DIM, Align2::CENTER_TOP);
        }
        decade_axis(f, &p, size);
        // the faint one first, then this one's
        let ghost = self.ghost.as_ref().filter(|(gs, v)| Setting { init: s.init, ..*gs } == *s && v.len() == n);
        // (the values too small to resolve lie on the last that can be: the line ends
        // there, and their dots are clear)
        let series = |f: &mut Frame, from: usize, seen: usize, kind: u8, color: Color, dot: f32| {
            let all: Vec<Vec2> = (0..n).map(|i| inside(&p, g.get(from + n + i) as f64, g.get(from + i) as f64)).collect();
            plot::path_of(f, kind, &all, 1.2, color.a(0.6));
            for (i, at) in all.into_iter().enumerate() {
                plot::disc(f, at, dot, if i < seen { color } else { Color::CLEAR });
            }
        };
        if let Some((gs, gv)) = ghost.filter(|_| g.shown.len() >= SPEC_AT + 4 * n) {
            let gseen = resolved_count(gv);
            series(f, SPEC_AT + 2 * n, gseen, GHOST_LINE, GHOST, 4.5);
            let last = gseen.max(1) - 1;
            p.text(f, g.get(SPEC_AT + 3 * n + last) as f64, g.get(SPEC_AT + 2 * n + last) as f64, Vec2::new(6.0, 2.0), init_name(gs.init), small, GHOST, Align2::LEFT_TOP);
        } else {
            // (the faint line's batch, with nothing in it but as many points, so that
            // the batches after it keep their places)
            let corner = inside(&p, 1.0, p.y[0]);
            plot::path_of(f, GHOST_LINE, &vec![corner; n], 1.2, Color::CLEAR);
        }
        series(f, SPEC_AT, seen, SPEC_LINE, BACKWARD, 5.5);
        let level = inside(&p, 1.0, g.get(2) as f64).y;
        plot::dashed_of(f, LEVEL, Vec2::new(min.x, level), Vec2::new(max.x, level), 1.2, BACKWARD.a(0.7));
        // what could not be drawn, in the corner the line has left empty
        if !unseen.is_empty() {
            let room = (max.x - p.px_x(seen as f64 + 1.0)).min(p.width() * 0.62).max(80.0);
            let at = Vec2::new(max.x - room - 2.0, max.y - 6.0 - wrapped_height_px(f, room, &unseen, small));
            wrapped_px(f, at, room, &unseen, small, pal::DIM);
        }
    }

    fn draw_loss(&self, f: &mut Frame, lay: &Layout) {
        let g = &self.look.loss;
        if g.shown.len() < 2 {
            return;
        }
        let p = Plot::new(lay.loss, [0.0, STEPS as f64], [g.get(0) as f64, g.get(1) as f64]);
        if p.tiny() {
            return;
        }
        let (size, small) = (lay.text, lay.small);
        p.backing(f, 46.0, 34.0, 8.0, 12.0);
        let guess = guess_loss(&self.y);
        if lay.narrow {
            let words = format!("the loss in training, rate {}; dashed: {}, a guess", num(self.lr as f64, 3), num(guess, 3));
            f.text_px(lay.loss_key, words, small, pal::TEXT, Align2::LEFT_TOP);
        } else {
            f.text_px(lay.loss_key, "the loss in training", size, pal::TEXT, Align2::LEFT_TOP);
            let width = lay.loss.1.x - lay.loss_key.x + 4.0;
            let (rate, guess) = (num(self.lr as f64, 3), num(guess, 3));
            let long = format!("{STEPS} steps of plain gradient descent on the 16 points at rate {rate}; dashed: {guess}, the loss of guessing each class's share");
            // (two lines are kept for it: a narrower window has the shorter words)
            let words = if wrapped_height_px(f, width, &long, small) <= 2.0 * (small + 3.0) + 1.0 { long } else { format!("gradient descent, rate {rate}; dashed: {guess}, guessing") };
            wrapped_px(f, lay.loss_key + Vec2::new(0.0, size + 6.0), width, &words, small, pal::DIM);
        }
        let (min, max) = p.rect;
        plot::line_of(f, AXES, Vec2::new(min.x, max.y), max, 1.0, pal::DIM.a(0.8));
        plot::line_of(f, AXES, min, Vec2::new(min.x, max.y), 1.0, pal::DIM.a(0.8));
        let ticks = plot::Ticks::of(0.0, STEPS as f64, p.width() * 0.8);
        for v in ticks.values() {
            let x = p.px_x(v);
            tick(f, Vec2::new(x, max.y), Vec2::new(x, max.y + 4.0), pal::DIM.a(0.8));
            f.text_px(Vec2::new(x, max.y + 6.0), ticks.name(v), size, pal::DIM, Align2::CENTER_TOP);
        }
        f.text_px(Vec2::new(max.x, max.y + 8.0 + size), "step", size, pal::DIM, Align2::RIGHT_TOP);
        decade_axis(f, &p, size);
        p.hline(f, GUESS, lg(guess) as f64, 1.0, pal::DIM.a(0.8), true);
        if self.runs.is_empty() {
            let hint = if lay.narrow { "train: a run from this start" } else { "“train” runs gradient descent from this start" };
            f.text_px((min + max) * 0.5, hint, small, pal::DIM, Align2::CENTER_CENTER);
        }
        let count = self.runs.len();
        let mut labels = Vec::with_capacity(count);
        for (k, run) in self.runs.iter().enumerate() {
            let bright = k + 1 == count;
            let color = if bright { LOSS } else { LOSS.mix(pal::DIM, 0.45).a(0.6) };
            let shown = run.shown.get(0).max(0.0);
            let whole = shown.floor() as usize;
            // the loss at a step, or at the last step that has one (a run that blew up
            // ends where its loss stopped being a number)
            let finite = run.losses.iter().rposition(|l| l.is_finite()).unwrap_or(0);
            let at = |k: usize| run.losses.get(k.min(finite)).map_or(p.y[0], |&l| lg(l) as f64);
            let tip = {
                let t = (shown - whole as f32) as f64;
                (shown as f64, at(whole) + (at(whole + 1) - at(whole)) * t)
            };
            // every step's point, those not reached yet gathered at the tip: the same
            // number of points in every frame
            let pts: Vec<Vec2> = (0..=STEPS).map(|k| if k as f32 <= shown { inside(&p, k as f64, at(k)) } else { inside(&p, tip.0, tip.1) }).collect();
            plot::path_of(f, RUNS + k as u8, &pts, if bright { 2.2 } else { 1.4 }, color);
            let Some(&end) = pts.last() else { continue };
            let (x, y) = (p.x_at(end.x), p.y_at(end.y));
            let label = self.run_label(run);
            let text = match run.blew_up() {
                Some(at) if whole + 1 >= at => format!("{label}: no number after {at} steps"),
                _ => label,
            };
            labels.push((p.at(x, y), text, color));
        }
        // each run named just under the end of its line (where a falling loss has left
        // room), to its left near the right edge, and moved apart where two would meet
        let (right, left): (Vec<_>, Vec<_>) = labels.into_iter().partition(|(at, _, _)| at.x > p.px_x(STEPS as f64 * 0.55));
        let under = |v: Vec<(Vec2, String, Color)>, dx: f32| v.into_iter().map(|(at, t, c)| (at + Vec2::new(dx, 6.0), t, c)).collect::<Vec<_>>();
        spread_labels(f, under(right, -2.0), small, Align2::RIGHT_TOP);
        spread_labels(f, under(left, 6.0), small, Align2::LEFT_TOP);
    }

    /// A run's name: what of its setting differs from the present one, or its scheme
    /// and gain.
    fn run_label(&self, run: &Run) -> String {
        let (r, s) = (run.setting, self.setting);
        let mut parts = vec![format!(r"{} $\times {}$", init_name(r.init), num(r.gain, 2))];
        if r.depth != s.depth {
            parts.push(format!("{} layers", r.depth));
        }
        if r.width != s.width {
            parts.push(format!("{} wide", r.width));
        }
        if r.act != s.act {
            parts.push(act_name(r.act).to_string());
        }
        if !run.clip {
            parts.push("not clipped".into());
        }
        if (run.lr - self.lr as f64).abs() > 1e-9 {
            parts.push(format!("rate {}", num(run.lr, 3)));
        }
        parts.join(", ")
    }
}

/// Where a gain sits along its scale, from 0 to 1.
fn knob_share(gain: f64) -> f32 {
    let t = (gain.max(LEAST_GAIN).min(MOST_GAIN).ln() - LEAST_GAIN.ln()) / (MOST_GAIN.ln() - LEAST_GAIN.ln());
    if t.is_finite() {
        t as f32
    } else {
        0.5
    }
}

/// A tick or a grid line, along or across: a thin rectangle one pixel wide. (Their
/// number changes as an axis glides; as fills they keep out of the batches of lines,
/// whose points `eidos motion` follows from frame to frame by their places.)
fn tick(f: &mut Frame, a: Vec2, b: Vec2, color: Color) {
    let (min, max) = (a.min(b) - Vec2::splat(0.5), a.max(b) + Vec2::splat(0.5));
    if min.is_finite() && max.is_finite() {
        hud_rect(f, min, max, color);
    }
}

/// Where (x, y) is on a chart's rectangle, kept inside it. (Lines are drawn through
/// such points rather than cut at the edge: the same number of points in every frame,
/// so that `eidos motion` can follow them while an axis glides.)
fn inside(p: &Plot, x: f64, y: f64) -> Vec2 {
    let at = p.at(x, y);
    if at.is_finite() {
        at.max(p.rect.0).min(p.rect.1)
    } else {
        Vec2::new(p.rect.0.x, p.rect.1.y)
    }
}

/// A dot at (x, y) of a chart, in every frame: where it is outside the chart (while an
/// axis glides) it is drawn clear at the edge, so that the number of dots drawn stays
/// the same from frame to frame.
fn dot_kept(f: &mut Frame, p: &Plot, x: f64, y: f64, size: f32, color: Color) {
    let at = p.at(x, y);
    let (min, max) = p.rect;
    let inside = at.is_finite() && plot::within(at, p.rect, 0.5);
    let at = if at.is_finite() { at.max(min).min(max) } else { min };
    plot::disc(f, at, size, if inside { color } else { Color::CLEAR });
}

/// Follow a glide, or be there at once.
fn follow_or_snap(g: &mut Glide, goal: &[f32], dt: f32, snap: bool) -> bool {
    if snap {
        g.snap(goal);
        false
    } else {
        g.follow(goal, dt, false)
    }
}

/// The y axis of a chart in powers of ten: a tick at every power (or every second,
/// fifth, ...) inside its range.
fn decade_axis(f: &mut Frame, p: &Plot, size: f32) {
    let (min, max) = p.rect;
    let span = p.y[1] - p.y[0];
    let room = (p.height() / (size + 12.0)).max(1.0) as f64;
    let step = [1, 2, 3, 5, 10, 20, 25, 50, 100].into_iter().find(|&s| span / s as f64 <= room).unwrap_or(100);
    let first = (p.y[0] / step as f64).ceil() as i64 * step;
    let mut d = first;
    while (d as f64) <= p.y[1] && d < first + 200 * step {
        let y = p.px_y(d as f64);
        tick(f, Vec2::new(min.x - 4.0, y), Vec2::new(min.x, y), pal::DIM.a(0.8));
        tick(f, Vec2::new(min.x, y), Vec2::new(max.x, y), pal::DIM.a(0.12));
        f.text_px(Vec2::new(min.x - 6.0, y), decade(d as i32), size, pal::DIM, Align2::RIGHT_CENTER);
        d += step;
    }
}

/// How tall `wrapped_px` draws a text.
fn wrapped_height_px(f: &Frame, width: f32, text: &str, size: f32) -> f32 {
    wrapped_height(f, width / pixels_per_unit(f).max(1e-6), text, size)
}

/// Text broken into lines no wider than `width`, from `at` down, in logical pixels of
/// the screen (`kit::wrapped` takes the plane's own units, and keeps its lines).
fn wrapped_px(f: &mut Frame, at: Vec2, width: f32, text: &str, size: f32, color: Color) {
    let ppu = pixels_per_unit(f).max(1e-6);
    let Some(world) = f.cam.pick_ground(at * f.px) else { return };
    wrapped(f, world.truncate(), width / ppu, text, size, color);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::math::arch::{input_output_jacobian, Span};
    use crate::math::linalg;

    fn setting(depth: usize, width: usize, act: Act, init: Init, gain: f64, seed: u64) -> Setting {
        Setting { depth, width, act, init, gain, seed }
    }

    /// A setting's starting network, as the scene makes it.
    fn start(s: &Setting) -> Model {
        scaled(&build_base(s), s.gain)
    }

    /// A short run, as the scene trains.
    fn train(model: Model, s: Setting, lr: f64, clip: bool) -> Run {
        let (x, y) = data();
        let mut tape = Tape::new();
        let mut run = Run::new(s, model, lr, clip);
        while !run.done() {
            run.step(&mut tape, &x, &y);
        }
        run
    }

    fn close(a: f64, b: f64, tol: f64) -> bool {
        (a - b).abs() <= tol
    }

    /// An experiment as its button sets it up, with its runs taken to their ends.
    fn trial(k: usize) -> Gradflow {
        let mut scene = Gradflow::create();
        scene.set_up(k);
        scene.finish_runs();
        scene
    }

    fn spectrum(scene: &Gradflow) -> &[f64] {
        scene.spectrum.as_ref().filter(|(s, _)| *s == scene.setting).map(|(_, v)| v.as_slice()).expect("a spectrum for the setting")
    }

    /// How many singular values can be told from rounding error.
    fn resolved(v: &[f64]) -> usize {
        v.iter().filter(|&&x| x > RESOLVED * v[0]).count()
    }

    /// The loss of the network with layer `l`'s output replaced by `h` (examples by
    /// units), worked out by hand from the model's weights: what the gradient the scene
    /// shows at layer l is the derivative of.
    fn loss_from(model: &Model, s: &Setting, l: usize, h: &[f64], y: &[f64]) -> f64 {
        let (h, n) = (hidden_from(model, s, l, h, s.depth), y.len());
        let (w, b) = (model.group("read-out weights").unwrap(), model.group("read-out bias").unwrap());
        let mut total = 0.0;
        for e in 0..n {
            let logits: Vec<f64> = (0..w.rows).map(|j| b.value[j] + (0..w.cols).map(|i| w.value[j * w.cols + i] * h[e * w.cols + i]).sum::<f64>()).collect();
            let top = logits.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
            let sum: f64 = logits.iter().map(|z| (z - top).exp()).sum();
            total += sum.ln() + top - logits[y[e] as usize];
        }
        total / n as f64
    }

    /// Layer `to`'s output with layer `l`'s replaced by `h`, by hand.
    fn hidden_from(model: &Model, s: &Setting, l: usize, h: &[f64], to: usize) -> Vec<f64> {
        let mut h = h.to_vec();
        for k in l + 1..=to {
            let (w, b) = (model.group(&format!("layer {k} weights")).unwrap(), model.group(&format!("layer {k} bias")).unwrap());
            let n = h.len() / w.cols;
            h = (0..n).flat_map(|e| (0..w.rows).map(move |j| (e, j))).map(|(e, j)| s.act.f(b.value[j] + (0..w.cols).map(|i| w.value[j * w.cols + i] * h[e * w.cols + i]).sum::<f64>())).collect();
        }
        h
    }

    /// The gradients the chart of layers shows, ∂L/∂h at every layer, are the
    /// derivatives of the loss: central differences of the loss worked out by hand from
    /// that layer on, at a few entries of each layer, for every activation (with the
    /// biases moved off 0, so that they count too).
    #[test]
    fn the_gradients_shown_are_differences_of_the_loss() {
        let (x, y) = data();
        let mut rng = Rng::new(5);
        for act in ACTS {
            for init in INITS {
                let s = setting(6, 8, act, init, 1.2, 3);
                let mut model = start(&s);
                for g in model.groups_mut().iter_mut().filter(|g| g.kind == Kind::Bias) {
                    g.value.iter_mut().for_each(|v| *v = 0.2 * rng.normal() as f64);
                }
                let mut tape = Tape::new();
                let trace = model.loss_mlp(&mut tape, &x, &y, Loss::CrossEntropy);
                tape.backward(trace.loss);
                for l in 1..=s.depth {
                    let v = trace.get(Stage::Output, l).unwrap();
                    let (h, grad) = (tape.value(v).to_vec(), tape.grad(v).to_vec());
                    for _ in 0..4 {
                        let k = rng.below(h.len());
                        let eps = 1e-6 * (1.0 + h[k].abs());
                        let (mut up, mut down) = (h.clone(), h.clone());
                        up[k] += eps;
                        down[k] -= eps;
                        let (a, b) = (loss_from(&model, &s, l, &up, &y), loss_from(&model, &s, l, &down, &y));
                        let fd = (a - b) / (2.0 * eps);
                        // (the difference of two losses loses their last digits: that much, and
                        // a millionth of the gradient for the truncation)
                        let rounding = 8.0 * f64::EPSILON * a.abs().max(b.abs()) / (2.0 * eps);
                        assert!((fd - grad[k]).abs() <= 1e-6 * grad[k].abs().max(fd.abs()) + rounding, "{} {} layer {l} entry {k}: tape {}, differences {fd}", act.name(), init.name(), grad[k]);
                    }
                }
                // and the chart's numbers are those gradients' root mean squares
                let rms: Vec<f64> = (1..=s.depth)
                    .map(|l| {
                        let g = tape.grad(trace.get(Stage::Output, l).unwrap());
                        (g.iter().map(|v| v * v).sum::<f64>() / g.len() as f64).sqrt()
                    })
                    .collect();
                let m = measure(&s, &model, &mut tape, &x, &y);
                assert_eq!(m.backward, rms);
            }
        }
    }

    /// The stack's Jacobian is the library's, worked out a row at a time, and its
    /// entries are differences of the stack's output; its singular values are
    /// `linalg::svd`'s.
    #[test]
    fn the_spectrum_is_the_librarys_jacobian_and_svd() {
        let (x, _) = data();
        for (act, init, gain, width, depth) in [
            (Act::Relu, Init::He, 1.0, 8, 6),
            (Act::Relu, Init::Lecun, 1.0, 32, 50),
            (Act::Tanh, Init::Lecun, 1.0, 16, 20),
            (Act::Tanh, Init::Orthogonal, 1.0, 32, 50),
            (Act::Identity, Init::Lecun, 1.0, 32, 50),
            (Act::Identity, Init::Orthogonal, 1.0, 64, 30),
            (Act::Relu, Init::Lecun, 4.0, 16, 50),
            (Act::Tanh, Init::He, 0.25, 8, 2),
        ] {
            let s = setting(depth, width, act, init, gain, 2);
            let model = start(&s);
            let mut job = SpectrumJob::new(&s, &model, &x).unwrap();
            // a row at a time, as the frames do it
            let spent = Budget::new(0.0);
            let mut values = None;
            let mut steps = 0;
            while values.is_none() {
                values = job.advance(&spent);
                steps += 1;
                assert!(steps < 10_000);
            }
            let values = values.unwrap();
            let (jac, rows, cols) = input_output_jacobian(&model, &x[..2], Span::Layers(2, depth));
            assert_eq!((rows, cols), (width, width));
            assert_eq!(job.jac, jac, "{} {} width {width}: the Jacobian", act.name(), init.name());
            let theirs = linalg::svd(&jac, rows, cols).1;
            assert_eq!(values.len(), theirs.len());
            for (k, (a, b)) in values.iter().zip(&theirs).enumerate() {
                if *b > 1e-10 * theirs[0] {
                    // (each is good to a few roundings of the largest)
                    assert!((a - b).abs() <= 1e-9 * b + 1e-14 * theirs[0], "{} {} width {width}: singular value {k}: {a} against {b}", act.name(), init.name());
                } else {
                    assert!(*a <= 1e-9 * theirs[0], "{} {} width {width}: singular value {k}: {a} where linalg::svd has {b}", act.name(), init.name());
                }
            }
            // a column of the Jacobian by differences of the stack's output, for a
            // small stack (the long ones underflow or overflow a difference)
            if depth <= 20 {
                let mut tape = Tape::new();
                let trace = model.forward_mlp(&mut tape, &x[..2]);
                let h1 = tape.value(trace.get(Stage::Output, 1).unwrap()).to_vec();
                for j in [0, width / 2, width - 1] {
                    let eps = 1e-6;
                    let (mut up, mut down) = (h1.clone(), h1.clone());
                    up[j] += eps;
                    down[j] -= eps;
                    let (a, b) = (hidden_from(&model, &s, 1, &up, depth), hidden_from(&model, &s, 1, &down, depth));
                    let scale = jac.iter().fold(0.0f64, |m, v| m.max(v.abs()));
                    for k in 0..width {
                        let fd = (a[k] - b[k]) / (2.0 * eps);
                        assert!((fd - jac[k * width + j]).abs() <= 1e-6 * scale + 1e-12, "{} entry ({k}, {j}): {} against {fd}", act.name(), jac[k * width + j]);
                    }
                }
            }
        }
    }

    /// Scaling a draw at gain 1 gives the library's draw at that gain, to rounding.
    #[test]
    fn a_scaled_draw_is_the_librarys_draw_at_that_gain() {
        for init in INITS {
            for gain in [0.25, 1.37, 4.0] {
                let s = setting(5, 16, Act::Tanh, init, gain, 9);
                let ours = scaled(&build_base(&s), gain);
                let theirs = Model::mlp(&s.spec(), &mut Rng::new(s.seed));
                for (a, b) in ours.groups().iter().zip(theirs.groups()) {
                    assert_eq!(a.name, b.name);
                    for (u, v) in a.value.iter().zip(&b.value) {
                        assert!((u - v).abs() <= 1e-13 * v.abs().max(1e-300), "{} at {gain}: {u} against {v}", a.name);
                    }
                }
            }
        }
    }

    /// The predictions: for ReLU and the identity a fixed factor a layer, √(σ²/2) and
    /// σ, forward and back, exactly 1 at the critical gain; and they are the averages
    /// over draws of the weights (64 units, 10 layers, 120 draws), within what so many
    /// draws can tell.
    #[test]
    fn the_lines_are_the_average_over_draws() {
        let (x, y) = data();
        let mut tape = Tape::new();
        for init in INITS {
            for (act, per) in [(Act::Relu, 0.5f64), (Act::Identity, 1.0)] {
                for gain in [0.6, 1.0, 1.7] {
                    let s = setting(12, 16, act, init, gain, 1);
                    let m = measure(&s, &start(&s), &mut tape, &x, &y);
                    let want = (s.hidden_sigma2() * per).sqrt();
                    let (f, b) = m.predicted_factors();
                    assert!(close(f.unwrap(), want, 1e-12) && close(b.unwrap(), want, 1e-12), "{} {}: {f:?} {b:?} against {want}", act.name(), init.name());
                }
                let s = setting(12, 16, act, init, setting(12, 16, act, init, 1.0, 1).critical_gain(), 1);
                let m = measure(&s, &scaled(&build_base(&s), s.gain), &mut tape, &x, &y);
                let (f, b) = m.predicted_factors();
                assert!(close(f.unwrap(), 1.0, 1e-12) && close(b.unwrap(), 1.0, 1e-12), "{} {} at the critical gain: {f:?} {b:?}", act.name(), init.name());
            }
        }
        for (act, init, gain) in [(Act::Relu, Init::He, 1.0), (Act::Relu, Init::Lecun, 1.6), (Act::Tanh, Init::Lecun, 1.3), (Act::Identity, Init::Lecun, 0.9)] {
            let (mut fwd, mut back) = (0.0, 0.0);
            let mut predicted = (0.0, 0.0);
            let draws = 400;
            for seed in 0..draws {
                let s = setting(6, 64, act, init, gain, 100 + seed);
                let m = measure(&s, &start(&s), &mut tape, &x, &y);
                // the mean square at the last layer as a share of layer 1's, forward, and
                // at layer 1 as a share of the last layer's, back
                fwd += (m.forward[6] / m.forward[1]).powi(2);
                back += (m.backward[0] / m.backward[5]).powi(2);
                predicted = ((m.predicted_forward[6] / m.predicted_forward[1]).powi(2), (m.predicted_backward[0] / m.predicted_backward[5]).powi(2));
            }
            let (fwd, back) = (fwd / draws as f64, back / draws as f64);
            assert!((fwd / predicted.0 - 1.0).abs() < 0.12, "{} {}: forward {fwd} against {}", act.name(), init.name(), predicted.0);
            assert!((back / predicted.1 - 1.0).abs() < 0.15, "{} {}: backward {back} against {}", act.name(), init.name(), predicted.1);
        }
    }

    /// The captions of the "things to try", held to account.
    #[test]
    fn the_experiments_show_what_their_captions_say() {
        // too small a start: 0.707 a layer, fitted 0.682 and 0.684; 6.3e-9 at layer 1 (line 4.2e-8)
        let scene = trial(0);
        let m = scene.measured().unwrap();
        let (f, b) = m.factors();
        let (pf, pb) = m.predicted_factors();
        let (left, pleft) = m.left();
        assert!(close(pf.unwrap(), 0.5f64.sqrt(), 1e-12) && close(pb.unwrap(), 0.5f64.sqrt(), 1e-12));
        assert!(close(f.unwrap(), 0.682, 5e-4) && close(b.unwrap(), 0.684, 5e-4), "{f:?} {b:?}");
        assert!(close(left, 6.3e-9, 0.05e-9) && close(pleft, 4.2e-8, 0.05e-8), "{left} {pleft}");
        assert!(close(scene.setting.critical_gain(), 1.41, 0.005));
        assert!(scene.runs.is_empty() && scene.ghost.is_none());
        // too large: 1.41 a layer, fitted 1.36 and 1.38; 6.3e6 at layer 1; loss 2.5e6; unclipped
        // 4.8e245 then NaN; clipped down to 115
        let scene = trial(1);
        let m = scene.measured().unwrap();
        let (f, b) = m.factors();
        let (pf, _) = m.predicted_factors();
        assert!(close(pf.unwrap(), 2f64.sqrt(), 1e-12) && close(scene.setting.hidden_sigma2(), 4.0, 1e-12));
        assert!(close(f.unwrap(), 1.36, 0.005) && close(b.unwrap(), 1.38, 0.005), "{f:?} {b:?}");
        assert!(close(m.left().0, 6.3e6, 0.05e6) && close(m.loss, 2.5e6, 0.05e6), "{} {}", m.left().0, m.loss);
        let (wild, tame) = (&scene.runs[0], &scene.runs[1]);
        assert!(!wild.clip && tame.clip);
        assert!(close(wild.losses[1] / 1e245, 4.8, 0.05) && wild.losses[2].is_nan() && wild.blew_up() == Some(2), "{:?}", wild.losses);
        let last = *tame.losses.last().unwrap();
        assert!(tame.losses.len() == STEPS + 1 && close(last, 115.0, 0.5), "{last}");
        // the edge: level lines; fitted 0.965 and 0.968; 0.15 at layer 1; 35 of 40 draws below 1;
        // 8 units: 0.89 and 7.2e-3; 64 units: 0.97 and 0.30
        let scene = trial(2);
        let m = scene.measured().unwrap();
        let (f, b) = m.factors();
        assert!(close(m.predicted_factors().0.unwrap(), 1.0, 1e-12) && close(m.predicted_factors().1.unwrap(), 1.0, 1e-12));
        assert!(close(f.unwrap(), 0.965, 5e-4) && close(b.unwrap(), 0.968, 5e-4) && close(m.left().0, 0.15, 0.005), "{f:?} {b:?} {}", m.left().0);
        let (x, y) = data();
        let mut tape = Tape::new();
        let below = (1..=40u64)
            .filter(|&seed| {
                let s = Setting { seed, ..scene.setting };
                measure(&s, &start(&s), &mut tape, &x, &y).factors().1.unwrap() < 1.0
            })
            .count();
        assert_eq!(below, 35);
        for (width, factor, at_one, tol) in [(8, 0.89, 7.2e-3, 0.05e-3), (64, 0.97, 0.30, 0.005)] {
            let s = Setting { width, ..scene.setting };
            let m = measure(&s, &start(&s), &mut tape, &x, &y);
            assert!(close(m.factors().1.unwrap(), factor, 0.005) && close(m.left().0, at_one, tol), "{width}: {:?} {}", m.factors(), m.left().0);
        }
        // the same average, a wild spread: orthogonal all exactly 1; Gaussian 1.9 down to
        // 5.8e-12, 11 smaller; the largest 85 %, the three largest 99.9 %; tanh 0.82 to
        // 0.95; ReLU spread
        let scene = trial(3);
        let mine = spectrum(&scene);
        assert!(mine.len() == 32 && mine.iter().all(|&v| close(v, 1.0, 1e-12)), "{mine:?}");
        let (g, theirs) = scene.ghost.as_ref().unwrap();
        assert!(g.init == Init::Lecun && Setting { init: Init::Orthogonal, ..*g } == scene.setting);
        let seen = resolved(theirs);
        assert!(close(theirs[0], 1.9, 0.05) && close(theirs[seen - 1], 5.8e-12, 0.05e-12) && theirs.len() - seen == 11, "{theirs:?}");
        let total: f64 = theirs.iter().map(|v| v * v).sum();
        assert!(close(theirs[0] * theirs[0] / total, 0.85, 0.005) && close(theirs[..3].iter().map(|v| v * v).sum::<f64>() / total, 0.999, 0.0005));
        let m = scene.measured().unwrap();
        assert!(close(m.factors().1.unwrap(), 1.0, 1e-9) && close(m.predicted_factors().1.unwrap(), 1.0, 1e-12));
        let mut other = trial(3);
        other.setting.act = Act::Tanh;
        other.refresh(None);
        let v = spectrum(&other);
        assert!(resolved(v) == 32 && close(v[0], 0.95, 0.005) && close(v[31], 0.82, 0.005), "{} {}", v[0], v[31]);
        other.setting = Setting { act: Act::Relu, gain: other.setting.critical_gain(), ..other.setting };
        other.setting.gain = 2f64.sqrt();
        other.refresh(None);
        let v = spectrum(&other);
        assert!(resolved(v) <= 12 && v[0] / v[resolved(v) - 1] > 1e6, "ReLU, orthogonal: {v:?}");
        // thirty layers: LeCun's signal at layer 30 is 5.0e-6 of the data's, its loss settles
        // at 1.095 and layer 1 moves by 2.2e-5; He's comes down to 8.1e-4
        let scene = trial(4);
        let (no, yes) = (&scene.runs[0], &scene.runs[1]);
        assert!(no.setting.init == Init::Lecun && yes.setting.init == Init::He && no.setting.depth == 30 && no.clip && yes.clip);
        assert!(close(no.lr, 0.05, 1e-12) && close(yes.lr, 0.05, 1e-12));
        let s = no.setting;
        let m = measure(&s, &start(&s), &mut tape, &x, &y);
        assert!(close(m.forward[30] / m.forward[0], 5.0e-6, 0.05e-6), "{}", m.forward[30] / m.forward[0]);
        let guess = guess_loss(&y);
        assert!(close(guess, 1.095, 5e-4) && close(*no.losses.last().unwrap(), guess, 5e-4), "{} against {guess}", no.losses.last().unwrap());
        assert!(no.losses.iter().all(|&l| l > guess - 1e-4 && l < 1.0990), "{:?}", no.losses.iter().fold(f64::INFINITY, |a, &b| a.min(b)));
        assert_eq!(sci(no.moved().0), r"$2.2 \times 10^{-5}$");
        assert!(close(*yes.losses.last().unwrap(), 8.1e-4, 0.05e-4), "{}", yes.losses.last().unwrap());
    }

    /// Too large a start without clipping is no number after two steps at every seed
    /// tried, and LeCun's thirty layers settle at the guess at every seed: what the
    /// captions show is not one draw's luck.
    #[test]
    fn the_training_claims_hold_for_other_draws() {
        for seed in 1..=6u64 {
            let wild = Setting { seed, ..TRIALS[1].setting };
            let run = train(start(&wild), wild, LR, false);
            assert!(run.blew_up().is_some_and(|at| at <= 2), "seed {seed}: {:?}", &run.losses[..run.losses.len().min(4)]);
            let s = Setting { seed, init: Init::Lecun, ..TRIALS[4].setting };
            let run = train(start(&s), s, LR, true);
            assert!(close(*run.losses.last().unwrap(), guess_loss(&data().1), 0.01), "seed {seed}: {}", run.losses.last().unwrap());
            let s = Setting { seed, ..TRIALS[4].setting };
            let run = train(start(&s), s, LR, true);
            assert!(*run.losses.last().unwrap() < 0.1, "seed {seed}: He's start ends at {}", run.losses.last().unwrap());
        }
    }

    /// Every number drawn is finite at the ends of every control, and nothing panics:
    /// every activation and scheme, the narrowest and widest, the shallowest and
    /// deepest, the smallest and largest gain, and a draw of 8 units in which a whole
    /// layer is shut for every example.
    #[test]
    fn the_ends_of_the_controls_are_fine() {
        let (x, y) = data();
        let mut tape = Tape::new();
        let finite = |v: &[f32]| v.iter().all(|x| x.is_finite());
        for act in ACTS {
            for init in INITS {
                for width in [8, 64] {
                    for depth in [LEAST_DEPTH, MOST_DEPTH] {
                        for gain in [LEAST_GAIN, MOST_GAIN] {
                            let s = setting(depth, width, act, init, gain, 4);
                            let model = start(&s);
                            let m = measure(&s, &model, &mut tape, &x, &y);
                            assert!(finite(&layers_goal(&m)), "{s:?}");
                            let v = SpectrumJob::new(&s, &model, &x).unwrap().finish();
                            assert!(v.len() == width && v.iter().all(|v| v.is_finite() && *v >= 0.0), "{s:?}: {v:?}");
                            assert!(finite(&spectrum_goal(&v, Some(&v))), "{s:?}");
                            let _ = (m.factors(), m.predicted_factors(), m.left(), sci(m.loss), times(m.factors().0));
                        }
                    }
                }
            }
        }
        // a shut layer: no gradient below it, and nothing breaks
        let shut = (1..200u64).find_map(|seed| {
            let s = setting(50, 8, Act::Relu, Init::He, 1.0, seed);
            let m = measure(&s, &start(&s), &mut tape, &x, &y);
            (m.backward[0] == 0.0).then_some((s, m))
        });
        let (s, m) = shut.expect("a draw with a shut layer");
        assert!(finite(&layers_goal(&m)), "{s:?}");
        let _ = (times(m.factors().1), sci(m.left().0));
        // a stack with a layer shut for the first example: its Jacobian is 0, every
        // singular value 0, and drawn as such
        let dead = (1..400u64)
            .map(|seed| setting(50, 8, Act::Relu, Init::He, 1.0, seed))
            .find(|s| measure(s, &start(s), &mut tape, &x, &y).fewest_open.is_some_and(|(open, _)| open == 0))
            .expect("a draw with a layer shut for the first example");
        let v = SpectrumJob::new(&dead, &start(&dead), &x).unwrap().finish();
        assert!(v.iter().all(|&v| v == 0.0) && resolved_count(&v) == 0, "{v:?}");
        assert!(finite(&spectrum_goal(&v, None)));
        // the scene itself, set to every end and drawn the numbers of
        let mut scene = Gradflow::create();
        for (depth, width, gain) in [(LEAST_DEPTH, 8, LEAST_GAIN), (MOST_DEPTH, 64, MOST_GAIN), (0, 0, f64::NAN), (usize::MAX, usize::MAX, f64::INFINITY)] {
            scene.setting = Setting { depth, width, gain, ..scene.setting };
            scene.refresh(None);
            scene.follow(0.016, true, false);
            assert!(finite(&scene.look.layers.shown) && finite(&scene.look.spectrum.shown) && finite(&scene.look.knob.shown), "{:?}", scene.setting);
        }
    }

    /// Choosing another scheme keeps the last spectrum, faint, while the rest of the
    /// setting stays; changing anything else lets it go.
    #[test]
    fn the_faint_spectrum_is_the_scheme_before() {
        let mut scene = Gradflow::create();
        scene.refresh(None);
        scene.choose_init(Init::Orthogonal);
        scene.refresh(None);
        let (g, _) = scene.ghost.as_ref().expect("the scheme before");
        assert!(g.init == Init::Lecun && scene.setting.init == Init::Orthogonal);
        scene.setting.gain = 1.5;
        scene.refresh(None);
        assert!(scene.ghost.is_none());
    }

    /// The runs keep to their number, take their steps from the pace while the scene
    /// runs, and none while it is paused. (A small network: a step of the opening's
    /// fifty layers can take longer than a frame's allowance on a busy machine, and the
    /// pace then waits real time between steps, which calls in a row never give it.)
    #[test]
    fn training_is_paced() {
        let mut scene = Gradflow::create();
        scene.setting = Setting { depth: 4, width: 8, ..scene.setting };
        for _ in 0..KEPT_RUNS + 2 {
            scene.train();
        }
        assert_eq!(scene.runs.len(), KEPT_RUNS);
        scene.work(0.0, 0.016);
        assert!(scene.runs.iter().all(|r| r.losses.is_empty()));
        for _ in 0..30 {
            scene.work(1.0 / 60.0, 1.0 / 60.0);
        }
        assert!(scene.runs.iter().all(|r| !r.losses.is_empty()) && scene.running());
    }

    /// The table of tanh's moments is their integrals: against the trapezoid rule on a
    /// fine grid (where that resolves the bell), against the direct sum between the
    /// table's entries, and against the limit for very wide bells.
    #[test]
    fn the_tanh_table_is_its_integrals() {
        for q in [2e-8, 1e-6, 1e-3, 0.0123, 0.1, 0.37, 1.0, 2.5, 10.0, 77.0, 1e3, 4e4, 3e6, 1e7] {
            let (m, s) = tanh_table().at(q);
            let (m2, s2) = tanh_moments_direct(q);
            assert!((m - m2).abs() <= 1e-7 * m2 && (s - s2).abs() <= 1e-7 * s2, "q {q}: ({m}, {s}) against ({m2}, {s2})");
        }
        for q in [1e9, 1e12] {
            let (m, s) = tanh_table().at(q);
            let h = 1.0 / (std::f64::consts::TAU * q).sqrt();
            let (m2, s2) = tanh_moments_direct(q);
            assert!((s - s2).abs() <= 1e-6 * s2 && (1.0 - m - 2.0 * h).abs() <= 1e-6 * h, "q {q}: ({m}, {s}) against ({m2}, {s2})");
        }
        for q in [1e-9, 1e-6, 1e-3, 0.01, 0.1, 0.37, 1.0, 2.5, 10.0, 77.0, 1e3] {
            let (m, s) = tanh_table().at(q);
            let sd = q.sqrt();
            let m2 = crate::math::hermite::gauss_mean(|z| (sd * z).tanh().powi(2));
            let s2 = crate::math::hermite::gauss_mean(|z| (1.0 / (sd * z).cosh()).powi(4));
            assert!((m - m2).abs() <= 1e-6 * m2.max(1e-12) + 1e-14, "q {q}: E tanh² {m} against {m2}");
            assert!((s - s2).abs() <= 1e-6 * s2 + 1e-14, "q {q}: E sech⁴ {s} against {s2}");
        }
    }

    /// What the work costs, natively (run with --ignored --nocapture): the tanh table,
    /// drawing a network, measuring it, its spectrum, a training step.
    #[test]
    #[ignore]
    fn the_cost_of_the_work() {
        let ms = |t: std::time::Instant| t.elapsed().as_secs_f64() * 1e3;
        let t = std::time::Instant::now();
        let _ = TanhTable::new();
        println!("tanh table: {:.1} ms", ms(t));
        let (x, y) = data();
        let mut tape = Tape::new();
        for (act, init) in [(Act::Relu, Init::He), (Act::Tanh, Init::Lecun), (Act::Tanh, Init::Orthogonal)] {
            for width in [32, 64] {
                let s = setting(50, width, act, init, 1.0, 1);
                let t = std::time::Instant::now();
                let base = build_base(&s);
                let built = ms(t);
                let t = std::time::Instant::now();
                let model = scaled(&base, 1.1);
                let scaling = ms(t);
                let t = std::time::Instant::now();
                let _ = measure(&s, &model, &mut tape, &x, &y);
                let measuring = ms(t);
                let mut job = SpectrumJob::new(&s, &model, &x).unwrap();
                let t = std::time::Instant::now();
                let mut steps = 0;
                while job.next_row < job.rows {
                    job.step();
                    steps += 1;
                }
                let rows = ms(t);
                let t = std::time::Instant::now();
                let mut sweeps = 0;
                while job.step().is_none() {
                    sweeps += 1;
                }
                let svd = ms(t);
                let mut run = Run::new(s, model, 0.05, true);
                let t = std::time::Instant::now();
                for _ in 0..20 {
                    run.step(&mut tape, &x, &y);
                }
                println!(
                    "{} {} 50 x {width}: draw {built:.2} ms, scale {scaling:.2}, measure {measuring:.2}, Jacobian {rows:.2} ({steps} rows, {:.3} each), singular values {svd:.2} ({sweeps} sweeps), a training step {:.2}",
                    act.name(),
                    init.name(),
                    rows / steps as f64,
                    ms(t) / 20.0
                );
            }
        }
    }

    /// What each experiment shows, through the function its button calls: the numbers
    /// the captions were written from (run with --ignored --nocapture).
    #[test]
    #[ignore]
    fn explore_trials() {
        for k in 0..TRIALS.len() {
            let mut scene = Gradflow::create();
            scene.set_up(k);
            scene.finish_runs();
            let s = scene.setting;
            let m = scene.measured().unwrap().clone();
            let (f, b) = m.factors();
            let (pf, pb) = m.predicted_factors();
            let (left, pleft) = m.left();
            let sv = &scene.spectrum.as_ref().unwrap().1;
            let seen = sv.iter().filter(|v| **v > RESOLVED * sv[0]).count();
            let rms = (sv.iter().map(|v| v * v).sum::<f64>() / sv.len() as f64).sqrt();
            println!("{}: {:?}", TRIALS[k].button, s);
            println!("   factors fwd {f:?} ({pf:?}) back {b:?} ({pb:?}) left {left:.3e} ({pleft:.3e}) loss {:.4e} open {:?}", m.loss, m.fewest_open);
            println!("   forward at 0 {:.3e}, at last {:.3e}; backward at last {:.3e}", m.forward[0], m.forward[s.depth], m.backward[s.depth - 1]);
            println!("   spectrum {:.4e} .. {:.4e}, {seen} of {} resolved, rms {rms:.4e} (pred {:.4e})", sv[0], sv[seen - 1], sv.len(), m.predicted_stack);
            if let Some((g, gv)) = &scene.ghost {
                let gseen = gv.iter().filter(|v| **v > RESOLVED * gv[0]).count();
                let grms = (gv.iter().map(|v| v * v).sum::<f64>() / gv.len() as f64).sqrt();
                println!("   ghost {:?}: {:.4e} .. {:.4e}, {gseen} resolved, rms {grms:.4e}", g.init, gv[0], gv[gseen - 1]);
            }
            for run in &scene.runs {
                let (a, z) = run.moved();
                println!("   run {:?} x{} clip {}: losses {:.4e} -> {:.4e} (min {:.4e}) steps {} blew {:?} moved {a:.2e} {z:.2e}", run.setting.init, run.setting.gain, run.clip, run.losses[0], run.losses.last().unwrap(), run.losses.iter().cloned().fold(f64::INFINITY, f64::min), run.losses.len() - 1, run.blew_up());
            }
            // the same at other widths
            for width in [8, 16, 64] {
                let mut other = Gradflow::create();
                other.set_up(k);
                other.setting.width = width;
                other.refresh(None);
                let m = other.measured().unwrap();
                let (f, b) = m.factors();
                let (left, pleft) = m.left();
                let sv = &other.spectrum.as_ref().unwrap().1;
                let seen = sv.iter().filter(|v| **v > RESOLVED * sv[0]).count();
                println!("   width {width}: fwd {f:?} back {b:?} left {left:.3e} ({pleft:.3e}) spectrum {:.3e}..{:.3e} {seen}/{}", sv[0], sv[seen - 1], sv.len());
            }
        }
        // tanh and ReLU with orthogonal weights, 50 layers
        for (act, gain) in [(Act::Tanh, 1.0), (Act::Relu, 2f64.sqrt()), (Act::Tanh, 1.0)] {
            for init in [Init::Lecun, Init::Orthogonal] {
                let mut scene = Gradflow::create();
                scene.set_up(3);
                scene.setting.act = act;
                scene.setting.init = init;
                scene.setting.gain = gain;
                scene.refresh(None);
                let sv = &scene.spectrum.as_ref().unwrap().1;
                let seen = sv.iter().filter(|v| **v > RESOLVED * sv[0]).count();
                let m = scene.measured().unwrap();
                println!("{} {:?} x{gain:.3}: spectrum {:.4e}..{:.4e} {seen}/{} open {:?} factors {:?} {:?}", act.name(), init, sv[0], sv[seen - 1], sv.len(), m.fewest_open, m.factors(), m.predicted_factors());
            }
        }
    }

    /// How typical the experiments' draw is (the edge at four widths, forty draws each),
    /// the unclipped run's losses, the share of the largest singular values, and how
    /// He's start trains at five depths: the numbers behind the captions' claims.
    #[test]
    #[ignore]
    fn explore_more() {
        let (x, y) = data();
        let mut tape = Tape::new();
        for width in [8, 16, 32, 64] {
            let mut fits = Vec::new();
            let mut lefts = Vec::new();
            for seed in 1..=40u64 {
                let s = setting(50, width, Act::Relu, Init::He, 1.0, seed);
                let m = measure(&s, &start(&s), &mut tape, &x, &y);
                fits.push(m.factors().1.unwrap_or(f64::NAN));
                lefts.push(m.left().0);
            }
            fits.sort_by(f64::total_cmp);
            lefts.sort_by(f64::total_cmp);
            let mean = fits.iter().sum::<f64>() / 40.0;
            let below = fits.iter().filter(|&&f| f < 1.0).count();
            let glog = (lefts.iter().map(|l| l.ln()).sum::<f64>() / 40.0).exp();
            println!("He w{width}: back fit mean {mean:.4}, median {:.4}, {below}/40 below 1; left median {:.3e} geo-mean {glog:.3e} min {:.2e} max {:.2e}", fits[20], lefts[20], lefts[0], lefts[39]);
        }
        let mut scene = Gradflow::create();
        scene.set_up(1);
        scene.finish_runs();
        println!("E2 unclipped losses {:?}", scene.runs[0].losses);
        scene.set_up(3);
        let gv = &scene.ghost.as_ref().unwrap().1;
        let total: f64 = gv.iter().map(|v| v * v).sum();
        println!("E4 gaussian shares: top1 {:.4} top3 {:.4}", gv[0] * gv[0] / total, gv[..3].iter().map(|v| v * v).sum::<f64>() / total);
        scene.set_up(4);
        scene.finish_runs();
        for run in &scene.runs {
            let l = &run.losses;
            println!("E5 {:?}: {:.4} {:.4} {:.4} {:.4} {:.4} {:.4} {:.2e}", run.setting.init, l[0], l[25], l[50], l[100], l[150], l[200], l[300]);
        }
        let lecun = Setting { init: Init::Lecun, ..scene.setting };
        let m = measure(&lecun, &build_base(&lecun), &mut tape, &x, &y);
        println!("E5 LeCun forward at 30 / data: {:.3e}", m.forward[30] / m.forward[0]);
        for depth in [2, 4, 8, 16, 30] {
            let mut line = format!("He depth {depth}:");
            for seed in 1..=4u64 {
                let s = setting(depth, 32, Act::Relu, Init::He, 1.0, seed);
                let run = train(start(&s), s, 0.05, true);
                let l = &run.losses;
                let first = l.iter().position(|&v| v < 0.1).map_or("-".to_string(), |k| k.to_string());
                line += &format!("  [{:.3} {:.3} {:.1e} <0.1 at {first}]", l[50], l[100], l[300]);
            }
            println!("{line}");
        }
    }
}

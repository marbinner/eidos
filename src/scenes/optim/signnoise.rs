//! Normalised steps in noise. Sign descent, Adam and Muon keep a direction and throw
//! away the size of the gradient: per coordinate (its sign) or per matrix (its singular
//! vectors). This scene asks what is left of that direction when the gradient comes
//! from a mini-batch, first for one coordinate whose noise is normal (exact formulas),
//! then for one layer of a small network at fixed weights (measured by sampling).
//!
//! One coordinate: the true gradient g, one example's noise of standard deviation σ
//! (taken as the unit), a batch of B, so the batch's gradient is about normal with
//! standard deviation σ/√B (an approximation, good when B is not small). A running
//! average m ← βm + (1 − β)g of independent batch gradients at fixed weights has the
//! mean g and the variance (1 − β)/(1 + β) times one batch's ([`worth`]): it is worth a
//! batch (1 + β)/(1 − β) times larger. From these, each rule's step in units of the
//! step the whole data would give ([`Coordinate`]).
//!
//! The network: a tanh MLP (2 → 16 → 16 → 16 → 3) on 1024 spiral points, trained by
//! Adam on demand. At its present weights every example's gradient is measured
//! (`terrain::PerExample`), and its second hidden layer, a 16 × 16 matrix, is the
//! layer every rule is compared on, because it is one Muon acts on. Batches are drawn
//! with replacement (as from an endless supply of examples like these), so that a
//! batch's covariance is exactly Σ/B. "The useful part" of a step (defined in
//! [`Tally`]) is measured for each rule over sampled batches ([`Plain`]), with
//! momentum over a long stream of batches at the same weights ([`Stream`]), and given
//! exactly where a formula holds (SGD, and Adam with its second moment at its mean).

use super::*;
use crate::clock::{self, Budget};
use crate::math::arch::{Init, MlpSpec, Model, Norm, Place};
use crate::math::autograd::{Act, Tape};
use crate::math::optim::{self, Hyper, Iteration, Optimizer, Rule, Scratch};
use crate::math::stats::{erf, normal_cdf};
use crate::math::terrain::{self, Draw, PerExample, Problem, Sampler};
use crate::scenes::stats::plot::{self, Glide, Plot, Rect, AXES, MARKS};

pub const DEF: SceneDef = SceneDef {
    id: "signnoise",
    title: "Normalised steps in noise",
    group: GROUP,
    blurb: "Sign descent, Adam and Muon throw away the size of the gradient: what is left when the gradient is mostly noise, and how that changes the batch size worth paying for.",
    make: |_| Box::new(SignNoise::create()),
};

/// Sign descent's colour. The course has none for it (it is not among the seven rules
/// of `descent` and `race`); RMSProp's is borrowed, since RMSProp does not appear here
/// and is its nearest kin: RMSProp with no memory, g/√(g²), is the sign.
const SIGN: Color = RMSPROP;

// ----- the network ---------------------------------------------------------------------

/// The examples: points on three spiral arms.
const EXAMPLES: usize = 1024;
/// The hidden layers' width and number.
const WIDTH: usize = 16;
const DEPTH: usize = 3;
/// The layer every rule is compared on: a hidden layer's weights, WIDTH by WIDTH.
const LAYER: &str = "layer 2 weights";
const SIDE: usize = WIDTH;
const COORDS: usize = SIDE * SIDE;
const DATA_SEED: u64 = 7;
const NET_SEED: u64 = 3;
const TRAIN_SEED: u64 = 11;
const SAMPLE_SEED: u64 = 5;
/// How the network is trained, when it is: Adam (Kingma and Ba's defaults but the
/// step), on batches of 32 dealt without replacement.
const TRAIN_LR: f64 = 0.01;
const TRAIN_BATCH: usize = 32;

fn spec() -> MlpSpec {
    MlpSpec { inputs: 2, width: WIDTH, depth: DEPTH, outputs: 3, act: Act::Tanh, norm: Norm::None, place: Place::Pre, residual: false, init: Init::Xavier, ..MlpSpec::default() }
}

// ----- batch sizes ---------------------------------------------------------------------

/// The batch sizes measured: 1, 2, 4, ..., 4096.
const LEVELS: usize = 13;
const MOST_B: usize = 1 << (LEVELS - 1);
/// Batches sampled for each size (without momentum).
const SAMPLED: usize = 200;
/// Of those, how many also give their singular values (they cost more).
const SPECTRUM_TRIALS: usize = 24;
/// The averages recorded from a stream with momentum, for each size.
const STREAM_SAMPLES: usize = 200;
/// The batches sampled for one coordinate's histogram (without momentum; with it, the
/// averages recorded).
const COORD_SAMPLES: usize = 4000;
const COORD_STREAM_SAMPLES: usize = 2000;
/// A stream for one coordinate longer than this many examples is not run.
const MOST_COORD_DRAWS: usize = 30_000_000;
/// The largest momentum the control allows.
const MOST_BETA: f64 = 0.99;

/// The batch size of level k.
fn size_of(k: usize) -> usize {
    1 << k.min(LEVELS - 1)
}

// ----- one coordinate, normal noise -------------------------------------------------------

/// How much larger a batch a running average with coefficient β is worth, at fixed
/// weights: (1 + β)/(1 − β). Derived: m = (1 − β) Σₖ βᵏ gₜ₋ₖ over independent batch
/// gradients of variance v has variance (1 − β)² Σₖ β²ᵏ v = (1 − β)²/(1 − β²) v = v (1 −
/// β)/(1 + β) once its start is forgotten (test `an_average_is_worth_a_larger_batch`).
fn worth(beta: f64) -> f64 {
    let b = if beta.is_finite() { beta.max(0.0).min(MOST_BETA) } else { 0.0 };
    (1.0 + b) / (1.0 - b)
}

/// The expected sign of g + sZ, Z standard normal: 2Φ(r) − 1 = erf(r/√2) with r = g/s
/// (`noise.md` D5; Balles and Hennig 2018, Eq. 10).
fn expected_sign(r: f64) -> f64 {
    if r.is_nan() {
        return f64::NAN;
    }
    erf(r / std::f64::consts::SQRT_2)
}

/// One coordinate with normal noise: the true gradient g in units of one example's
/// noise σ (so σ = 1), a batch of B, momentum β. Everything about its steps, in units of
/// the step each rule would take with the whole data's gradient.
#[derive(Clone, Copy, Debug, PartialEq)]
struct Coordinate {
    g: f64,
    b: f64,
    beta: f64,
}

impl Coordinate {
    /// r = g√B/σ: the true gradient over one batch's noise.
    fn ratio(&self) -> f64 {
        self.g * self.b.max(1.0).sqrt()
    }

    /// The same for what the step is made from: the average, worth a larger batch.
    fn ratio_used(&self) -> f64 {
        self.ratio() * worth(self.beta).sqrt()
    }

    /// The probability that the sign used is the true one: Φ(r).
    fn right(&self) -> f64 {
        normal_cdf(self.ratio_used())
    }

    /// Sign descent (Signum, with momentum): the mean step and its standard deviation.
    /// The step is ±1; its mean is the expected sign, and its spread √(1 − mean²).
    fn sign(&self) -> (f64, f64) {
        let m = expected_sign(self.ratio_used());
        (m, (1.0 - m * m).max(0.0).sqrt())
    }

    /// SGD (heavy-ball momentum, with β): mean 1 (the whole data's step) and spread
    /// one batch's noise over g, divided by √worth.
    fn sgd(&self) -> (f64, f64) {
        (1.0, 1.0 / self.ratio_used())
    }

    /// Adam at fixed weights with its second moment at its mean, v = g² + σ²/B (it
    /// averages squares of single batches, over about 1/(1 − β₂) steps), ε neglected:
    /// the step m/√v has mean g/√(g² + σ²/B) and spread (σ/√(B·worth))/√(g² + σ²/B)
    /// (`noise.md` D6, McCandlish and co-authors App. E.2, Balles and Hennig Eqs. 7 to
    /// 9). Its whole-data step is ±1, the sign.
    fn adam(&self) -> (f64, f64) {
        let r = self.ratio();
        let root = (1.0 + r * r).sqrt();
        (r / root, 1.0 / (worth(self.beta).sqrt() * root))
    }
}

// ----- the network's layer, measured ---------------------------------------------------

/// The layer's statistics at one set of weights, from every example's gradient.
#[derive(Clone, Debug)]
struct Layer {
    /// The training step these weights are from, and the loss on all the data.
    step: usize,
    loss: f64,
    /// Every example's gradient of the layer, one after another (EXAMPLES × COORDS).
    rows: Vec<f64>,
    /// The true gradient G (the mean of the rows) and each coordinate's variance σᵢ²
    /// over the examples (divisor N: the variance of one example drawn at random).
    g: Vec<f64>,
    var: Vec<f64>,
    /// Each coordinate's own noise scale Bᵢ = σᵢ²/Gᵢ².
    own: Vec<f64>,
    /// The share of examples whose gradient has the true gradient's sign, coordinate by
    /// coordinate: exactly the probability that a batch of one gives the right sign.
    right1: Vec<f64>,
    /// The coordinates by their own noise scale, the least noisy first.
    ranked: Vec<usize>,
    /// The whole-data steps of sign descent (and Adam), and of Muon (the reference's
    /// iteration on G), and their lengths.
    sign_g: Vec<f64>,
    sign_norm: f64,
    muon_g: Vec<f64>,
    muon_norm: f64,
    /// G's singular values, largest first.
    sv: Vec<f64>,
    /// Sign descent's useful part at a batch of one, exactly ([`Layer::sign_one`]).
    sign1: f64,
    /// The layer's noise scales: McCandlish's simple one (B_ℓ2 in Naganuma's naming),
    /// Naganuma and co-authors' ℓ1 one, and their nuclear-norm one read with the noise
    /// covariance of the rows and of the columns.
    simple: f64,
    l1: f64,
    nuclear: [f64; 2],
    /// The same two for every parameter of the network.
    whole_simple: f64,
    whole_l1: f64,
}

/// The noise covariance of the layer's rows, E[D Dᵀ], and of its columns, E[Dᵀ D], over
/// the examples' deviations D = gₙ − G (SIDE by SIDE each), added up a few examples at a
/// time: from them Naganuma and co-authors' nuclear-norm noise scale,
/// ‖C^½‖²_S1 / ‖G‖²_S1, read with either.
#[derive(Clone, Debug)]
struct Covariances {
    step: usize,
    k: usize,
    rows: Vec<f64>,
    cols: Vec<f64>,
    d: Vec<f64>,
}

impl Covariances {
    fn new(step: usize) -> Covariances {
        Covariances { step, k: 0, rows: vec![0.0; SIDE * SIDE], cols: vec![0.0; SIDE * SIDE], d: vec![0.0; COORDS] }
    }

    /// Up to `n` more examples; returns the two noise scales once every example is in.
    fn advance(&mut self, layer: &Layer, n: usize) -> Option<[f64; 2]> {
        let count = layer.rows.len() / COORDS;
        let end = (self.k + n).min(count);
        for k in self.k..end {
            for i in 0..COORDS {
                self.d[i] = layer.rows[k * COORDS + i] - layer.g[i];
            }
            let d = &self.d;
            for a in 0..SIDE {
                for b in a..SIDE {
                    let (mut r, mut c) = (0.0, 0.0);
                    for j in 0..SIDE {
                        r += d[a * SIDE + j] * d[b * SIDE + j];
                        c += d[j * SIDE + a] * d[j * SIDE + b];
                    }
                    self.rows[a * SIDE + b] += r;
                    self.cols[a * SIDE + b] += c;
                }
            }
        }
        self.k = end;
        if self.k < count {
            return None;
        }
        let full = |half: &[f64]| {
            let mut c = vec![0.0; SIDE * SIDE];
            for a in 0..SIDE {
                for b in a..SIDE {
                    c[a * SIDE + b] = half[a * SIDE + b] / count.max(1) as f64;
                    c[b * SIDE + a] = c[a * SIDE + b];
                }
            }
            c
        };
        let nuc: f64 = layer.sv.iter().sum();
        Some([(trace_root(&full(&self.rows), SIDE) / nuc).powi(2), (trace_root(&full(&self.cols), SIDE) / nuc).powi(2)])
    }
}

/// The ℓ1 norm.
fn l1(v: &[f64]) -> f64 {
    v.iter().map(|x| x.abs()).sum()
}

fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

/// Naganuma and co-authors' ℓ1 noise scale (2026, Eq. 13): ‖σ‖₁² / ‖G‖₁², with σ the
/// vector of per-coordinate standard deviations of one example's gradient.
fn l1_scale(g: &[f64], var: &[f64]) -> f64 {
    let sd: f64 = var.iter().map(|v| v.max(0.0).sqrt()).sum();
    (sd / l1(g)).powi(2)
}

/// tr(C^½) for the symmetric positive semi-definite `n` by `n` matrix C: the sum of the
/// roots of its eigenvalues (negative ones, rounding, taken as 0).
fn trace_root(c: &[f64], n: usize) -> f64 {
    optim::sym_eigen(c, n).0.iter().map(|l| l.max(0.0).sqrt()).sum()
}

impl Layer {
    /// From a finished measurement of every example (the model's weights as they were).
    fn of(per: &PerExample, model: &Model, problem: &Problem, tape: &mut Tape, step: usize) -> Layer {
        let span = terrain::span(model, LAYER).unwrap_or(0..0);
        let n = per.len();
        let mut rows = Vec::with_capacity(n * COORDS);
        for k in 0..n {
            let r = per.row(k);
            let piece = r.get(span.clone()).unwrap_or(&[]);
            rows.extend_from_slice(piece);
            rows.resize((k + 1) * COORDS, 0.0);
        }
        let stats = per.noise_in(span.clone());
        let g = if stats.mean.len() == COORDS { stats.mean.clone() } else { vec![0.0; COORDS] };
        let var = if stats.variances.len() == COORDS { stats.variances.clone() } else { vec![0.0; COORDS] };
        let own: Vec<f64> = (0..COORDS).map(|i| if g[i] == 0.0 { f64::INFINITY } else { var[i] / (g[i] * g[i]) }).collect();
        let right1: Vec<f64> = (0..COORDS)
            .map(|i| {
                let s = g[i].signum();
                let agree = (0..n).filter(|&k| g[i] != 0.0 && rows[k * COORDS + i] * s > 0.0).count();
                agree as f64 / n.max(1) as f64
            })
            .collect();
        let mut ranked: Vec<usize> = (0..COORDS).collect();
        ranked.sort_by(|&a, &b| own[a].total_cmp(&own[b]));
        let sign_g = optim::sign(&g);
        let sign_norm = dot(&sign_g, &sign_g).sqrt();
        let mut muon_g = g.clone();
        Iteration::JORDAN.run(&mut muon_g, SIDE, SIDE, &mut Scratch::new());
        let muon_norm = dot(&muon_g, &muon_g).sqrt();
        let sv = optim::singular_values(&g, SIDE, SIDE);
        // (the nuclear-norm noise scales take a pass over every example's matrix: see
        // `Covariances`, worked out in slices afterwards)
        let nuclear = [f64::NAN; 2];
        let whole = per.noise();
        let mut layer = Layer {
            step,
            loss: problem.loss(model, tape),
            simple: stats.simple_noise_scale(),
            l1: l1_scale(&g, &var),
            nuclear,
            whole_simple: whole.simple_noise_scale(),
            whole_l1: l1_scale(&whole.mean, &whole.variances),
            rows,
            g,
            var,
            own,
            right1,
            ranked,
            sign_g,
            sign_norm,
            muon_g,
            muon_norm,
            sv,
            sign1: f64::NAN,
        };
        layer.sign1 = layer.sign_one();
        layer
    }

    /// |G|.
    fn norm(&self) -> f64 {
        dot(&self.g, &self.g).sqrt()
    }

    /// The median own noise scale.
    fn median_scale(&self) -> f64 {
        let k = self.ranked.get(COORDS / 2).copied().unwrap_or(0);
        self.own.get(k).copied().unwrap_or(f64::NAN)
    }

    /// How many coordinates have their own noise scale below `b`.
    fn below(&self, b: f64) -> usize {
        self.own.iter().filter(|&&s| s < b).count()
    }

    /// SGD's useful part, exactly: |G| / √(E|g_B|²) = 1/√(1 + B_simple/B), with
    /// momentum at B·worth (test `sgd_and_adam_formulas_hold_by_sampling`).
    fn sgd_useful(&self, b: f64) -> f64 {
        1.0 / (1.0 + self.simple / b).sqrt()
    }

    /// Adam's useful part at fixed weights with v at its mean (G² + σ²/B, from single
    /// batches) and its first moment worth a batch b·`worth`: E⟨u, sign G⟩ /
    /// (|sign G| √E|u|²), with E uᵢ = Gᵢ/√vᵢ and E uᵢ² = (Gᵢ² + σᵢ²/(b worth))/vᵢ.
    /// Returns it and the root mean square of the step per coordinate (in units of the
    /// learning rate).
    fn adam_useful(&self, b: f64, worth: f64) -> (f64, f64) {
        let (mut along, mut square) = (0.0, 0.0);
        for i in 0..COORDS {
            let (g, s2) = (self.g[i], self.var[i]);
            let v = g * g + s2 / b;
            if v > 0.0 {
                along += g.abs() / v.sqrt();
                square += (g * g + s2 / (b * worth)) / v;
            }
        }
        (along / (self.sign_norm * square.sqrt()), (square / COORDS as f64).sqrt())
    }

    /// Sign descent's useful part at a batch of one, exactly: a coordinate's sign is right
    /// for the share `right1` of the examples and wrong for the rest (an example whose
    /// gradient there is exactly 0 counts as neither, and so does the step's length).
    fn sign_one(&self) -> f64 {
        let n = self.rows.len() / COORDS;
        let (mut along, mut square) = (0.0, 0.0);
        for k in 0..n {
            for i in 0..COORDS {
                let x = self.rows[k * COORDS + i];
                if x != 0.0 {
                    square += 1.0;
                    along += if x * self.g[i] > 0.0 { 1.0 } else if self.g[i] != 0.0 { -1.0 } else { 0.0 };
                }
            }
        }
        if n == 0 || square == 0.0 {
            return f64::NAN;
        }
        (along / n as f64) / (self.sign_norm * (square / n as f64).sqrt())
    }

    /// Sign descent's useful part with each coordinate's batch gradient taken as normal
    /// (the formula of one coordinate, averaged over the layer): Σ erf(√(B / 2Bᵢ)) over
    /// the coordinates with a sign, over |sign G| √n.
    fn sign_normal(&self, b: f64) -> f64 {
        let along: f64 = (0..COORDS).filter(|&i| self.g[i] != 0.0).map(|i| expected_sign((b / self.own[i]).sqrt())).sum();
        along / (self.sign_norm * (COORDS as f64).sqrt())
    }
}

// ----- sampling ---------------------------------------------------------------------------

/// The running sums from which "the useful part" of a rule's step is measured. For a
/// step d computed from a mini-batch and the step d* the same rule takes with the
/// whole data, the useful part is
///
///   u = E⟨d, d*⟩ / (|d*| √(E|d|²)):
///
/// the mean length of d along d*, over d's root mean square length. It lies between −1
/// and 1. With a = E⟨d, d̂*⟩ the mean progress along d*, E|d|² = a² + E|d − a d̂*|²: a
/// share u² of d's mean square is the square of the mean progress, and the rest, 1 − u²,
/// is everything else (the jitter, and any steady part across d*). For a step of fixed
/// length (the sign; Muon's, nearly) it is the mean cosine with d*. For SGD it is
/// exactly 1/√(1 + B_simple/B).
#[derive(Clone, Copy, Debug, Default, PartialEq)]
struct Tally {
    n: usize,
    /// Σ⟨d, d*⟩/|d*|, its square, and Σ|d|².
    along: f64,
    along2: f64,
    square: f64,
}

impl Tally {
    fn push(&mut self, along: f64, square: f64) {
        if along.is_finite() && square.is_finite() {
            self.n += 1;
            self.along += along;
            self.along2 += along * along;
            self.square += square;
        }
    }

    fn useful(&self) -> f64 {
        if self.n == 0 || self.square.is_nan() || self.square <= 0.0 {
            return f64::NAN;
        }
        (self.along / self.n as f64) / (self.square / self.n as f64).sqrt()
    }

    /// Its standard error, from the spread of the lengths along d* (the root mean square
    /// length varies far less).
    fn error(&self) -> f64 {
        if self.n < 2 || self.square.is_nan() || self.square <= 0.0 {
            return f64::NAN;
        }
        let n = self.n as f64;
        let mean = self.along / n;
        let var = ((self.along2 - n * mean * mean) / (n - 1.0)).max(0.0);
        (var / n).sqrt() / (self.square / n).sqrt()
    }
}

/// Sign descent's and Muon's steps from one batch gradient `gb`: their tallies get one
/// more each. `o` and `s` are room to work in.
fn tally_normalised(layer: &Layer, gb: &[f64], sign: &mut Tally, muon: &mut Tally, o: &mut Vec<f64>, s: &mut Scratch) {
    let (mut along, mut square) = (0.0, 0.0);
    for (&x, &t) in gb.iter().zip(&layer.sign_g) {
        let d = if x > 0.0 {
            1.0
        } else if x < 0.0 {
            -1.0
        } else {
            0.0
        };
        along += d * t;
        square += d * d;
    }
    sign.push(along / layer.sign_norm, square);
    o.clear();
    o.extend_from_slice(gb);
    Iteration::JORDAN.run(o, SIDE, SIDE, s);
    muon.push(dot(o, &layer.muon_g) / layer.muon_norm, dot(o, o));
}

/// Mini-batches without momentum, SAMPLED of each size: each trial draws MOST_B examples
/// (with replacement) and its first B make the batch of size B, so that one draw serves
/// every size (a size's batches are independent of one another; two sizes' batches of
/// one trial share examples, which makes their curves smoother but not biased).
#[derive(Clone, Debug)]
struct Plain {
    rng: Rng,
    trials: usize,
    sgd: [Tally; LEVELS],
    sign: [Tally; LEVELS],
    muon: [Tally; LEVELS],
    /// Over the first SPECTRUM_TRIALS trials, the sums of the sorted singular values of
    /// each size's batch gradient and of its noise alone (g_B − G).
    batch_sv: Vec<[f64; SIDE]>,
    noise_sv: Vec<[f64; SIDE]>,
    sum: Vec<f64>,
    gb: Vec<f64>,
    o: Vec<f64>,
    sv: Vec<f64>,
    scratch: Scratch,
}

impl Plain {
    fn new(seed: u64) -> Plain {
        Plain {
            rng: Rng::new(seed),
            trials: 0,
            sgd: [Tally::default(); LEVELS],
            sign: [Tally::default(); LEVELS],
            muon: [Tally::default(); LEVELS],
            batch_sv: vec![[0.0; SIDE]; LEVELS],
            noise_sv: vec![[0.0; SIDE]; LEVELS],
            sum: vec![0.0; COORDS],
            gb: vec![0.0; COORDS],
            o: Vec::with_capacity(COORDS),
            sv: Vec::with_capacity(SIDE),
            scratch: Scratch::new(),
        }
    }

    fn done(&self) -> bool {
        self.trials >= SAMPLED
    }

    /// One more trial.
    fn trial(&mut self, layer: &Layer) {
        if self.done() {
            return;
        }
        let gn = layer.norm();
        self.sum.fill(0.0);
        let mut drawn = 0;
        let spectrum = self.trials < SPECTRUM_TRIALS;
        for k in 0..LEVELS {
            let b = size_of(k);
            while drawn < b {
                let e = self.rng.below(EXAMPLES);
                for (s, r) in self.sum.iter_mut().zip(&layer.rows[e * COORDS..(e + 1) * COORDS]) {
                    *s += r;
                }
                drawn += 1;
            }
            for (o, s) in self.gb.iter_mut().zip(&self.sum) {
                *o = s / b as f64;
            }
            self.sgd[k].push(dot(&self.gb, &layer.g) / gn, dot(&self.gb, &self.gb));
            tally_normalised(layer, &self.gb, &mut self.sign[k], &mut self.muon[k], &mut self.o, &mut self.scratch);
            if spectrum {
                optim::singular_values_into(&self.gb, SIDE, SIDE, &mut self.sv, &mut self.scratch);
                for (a, v) in self.batch_sv[k].iter_mut().zip(&self.sv) {
                    *a += v;
                }
                for (o, g) in self.gb.iter_mut().zip(&layer.g) {
                    *o -= g;
                }
                optim::singular_values_into(&self.gb, SIDE, SIDE, &mut self.sv, &mut self.scratch);
                for (a, v) in self.noise_sv[k].iter_mut().zip(&self.sv) {
                    *a += v;
                }
            }
        }
        self.trials += 1;
    }

    /// The mean sorted singular values at level k: of the batch gradient, of its noise.
    fn spectra(&self, k: usize) -> ([f64; SIDE], [f64; SIDE]) {
        let n = self.trials.min(SPECTRUM_TRIALS).max(1) as f64;
        let k = k.min(LEVELS - 1);
        (self.batch_sv[k].map(|v| v / n), self.noise_sv[k].map(|v| v / n))
    }
}

/// With momentum, at the same weights: a stream of independent batches, each size's
/// average m ← βm + (1 − β)g_B (divided by 1 − βᵗ, as Adam corrects its m, so that
/// it is a weighted mean from the first step), recorded every `every` steps once its
/// start is forgotten. Only the sizes whose average is worth up to 8192 examples are
/// run, and as many as keep the stream to about a million examples drawn.
#[derive(Clone, Debug)]
struct Stream {
    beta: f64,
    levels: usize,
    every: usize,
    burn: usize,
    rng: Rng,
    t: usize,
    /// One average per size, end to end.
    m: Vec<f64>,
    samples: usize,
    sign: [Tally; LEVELS],
    muon: [Tally; LEVELS],
    sum: Vec<f64>,
    mh: Vec<f64>,
    o: Vec<f64>,
    scratch: Scratch,
}

impl Stream {
    /// The steps between two recordings, and the steps before the first: twice and five
    /// times the average's memory 1/(1 − β) (after `every` steps the noise it had is
    /// down to about e⁻², 14 %; after `burn`, its start to e⁻⁵).
    fn spacing(beta: f64) -> (usize, usize) {
        let memory = 1.0 / (1.0 - beta.max(0.0).min(MOST_BETA));
        ((2.0 * memory).round().max(1.0) as usize, (5.0 * memory).ceil() as usize)
    }

    /// The sizes run (levels 0 to this, exclusive) for momentum β.
    fn levels_for(beta: f64) -> usize {
        let (every, burn) = Stream::spacing(beta);
        let steps = burn + STREAM_SAMPLES * every;
        let mut levels = 1;
        while levels < LEVELS && (size_of(levels) as f64 * worth(beta) <= 8192.0) && steps * size_of(levels) <= 1_200_000 {
            levels += 1;
        }
        levels
    }

    fn new(beta: f64, seed: u64) -> Stream {
        let (every, burn) = Stream::spacing(beta);
        let levels = Stream::levels_for(beta);
        Stream {
            beta,
            levels,
            every,
            burn,
            rng: Rng::new(seed),
            t: 0,
            m: vec![0.0; levels * COORDS],
            samples: 0,
            sign: [Tally::default(); LEVELS],
            muon: [Tally::default(); LEVELS],
            sum: vec![0.0; COORDS],
            mh: vec![0.0; COORDS],
            o: Vec::with_capacity(COORDS),
            scratch: Scratch::new(),
        }
    }

    fn done(&self) -> bool {
        self.samples >= STREAM_SAMPLES
    }

    /// The steps the whole stream takes.
    #[cfg(test)]
    fn length(&self) -> usize {
        self.burn + STREAM_SAMPLES * self.every
    }

    /// One step of the stream (and a recording, if it is time).
    fn step(&mut self, layer: &Layer) {
        if self.done() {
            return;
        }
        self.sum.fill(0.0);
        let mut drawn = 0;
        self.t += 1;
        let record = self.t > self.burn && (self.t - self.burn).is_multiple_of(self.every);
        let correct = 1.0 / (1.0 - self.beta.powi(self.t.min(100_000) as i32));
        for k in 0..self.levels {
            let b = size_of(k);
            while drawn < b {
                let e = self.rng.below(EXAMPLES);
                for (s, r) in self.sum.iter_mut().zip(&layer.rows[e * COORDS..(e + 1) * COORDS]) {
                    *s += r;
                }
                drawn += 1;
            }
            let m = &mut self.m[k * COORDS..(k + 1) * COORDS];
            for (m, s) in m.iter_mut().zip(&self.sum) {
                *m = self.beta * *m + (1.0 - self.beta) * (s / b as f64);
            }
            if record {
                for (h, m) in self.mh.iter_mut().zip(m.iter()) {
                    *h = m * correct;
                }
                tally_normalised(layer, &self.mh, &mut self.sign[k], &mut self.muon[k], &mut self.o, &mut self.scratch);
            }
        }
        if record {
            self.samples += 1;
        }
    }
}

/// One coordinate's batch gradients, sampled: its histogram and how often its sign is
/// right, at a batch size and momentum. Values are kept in units of the coordinate's
/// own σ, with the true gradient's sign taken as positive.
#[derive(Clone, Debug)]
struct CoordSample {
    coord: usize,
    b: usize,
    beta: f64,
    rng: Rng,
    values: Vec<f64>,
    right: usize,
    /// The running average (with momentum), the steps taken, and when to record.
    m: f64,
    t: usize,
    every: usize,
    burn: usize,
    target: usize,
    /// Too long a stream to run.
    skipped: bool,
}

impl CoordSample {
    fn new(coord: usize, b: usize, beta: f64, seed: u64) -> CoordSample {
        let (every, burn) = if beta > 0.0 { Stream::spacing(beta) } else { (1, 0) };
        let target = if beta > 0.0 { COORD_STREAM_SAMPLES } else { COORD_SAMPLES };
        let skipped = (burn + target * every).saturating_mul(b) > MOST_COORD_DRAWS;
        CoordSample { coord, b, beta, rng: Rng::new(seed), values: Vec::with_capacity(target), right: 0, m: 0.0, t: 0, every, burn, target, skipped }
    }

    fn done(&self) -> bool {
        self.skipped || self.values.len() >= self.target
    }

    /// Up to `n` more draws of a batch.
    fn advance(&mut self, layer: &Layer, n: usize) {
        let i = self.coord.min(COORDS - 1);
        let (g, sd) = (layer.g[i], layer.var[i].max(0.0).sqrt());
        let unit = if sd > 0.0 { g.signum() / sd } else { 0.0 };
        for _ in 0..n {
            if self.done() {
                return;
            }
            let mut s = 0.0;
            for _ in 0..self.b {
                s += layer.rows[self.rng.below(EXAMPLES) * COORDS + i];
            }
            let gb = s / self.b as f64;
            let x = if self.beta > 0.0 {
                self.t += 1;
                self.m = self.beta * self.m + (1.0 - self.beta) * gb;
                if !(self.t > self.burn && (self.t - self.burn).is_multiple_of(self.every)) {
                    continue;
                }
                self.m / (1.0 - self.beta.powi(self.t.min(100_000) as i32))
            } else {
                gb
            };
            if x * g > 0.0 {
                self.right += 1;
            }
            self.values.push(x * unit);
        }
    }

    /// The share of the samples with the right sign.
    fn share_right(&self) -> f64 {
        if self.values.is_empty() {
            f64::NAN
        } else {
            self.right as f64 / self.values.len() as f64
        }
    }
}

// ----- training ---------------------------------------------------------------------------

/// The network, its data and its training, reproducible from the seeds: the same step
/// count always means the same weights.
struct Trainer {
    problem: Problem,
    model: Model,
    opt: Optimizer,
    sampler: Sampler,
    tape: Tape,
    steps: usize,
}

impl Trainer {
    fn new() -> Trainer {
        let problem = Problem::spirals(EXAMPLES, 3, &mut Rng::new(DATA_SEED));
        let model = Model::mlp(&spec(), &mut Rng::new(NET_SEED));
        let opt = Optimizer::new(&model, Rule::Adam, Hyper { lr: TRAIN_LR, ..Hyper::default_for(Rule::Adam) });
        let sampler = Sampler::new(problem.len(), TRAIN_BATCH, Draw::WithoutReplacement, Rng::new(TRAIN_SEED));
        Trainer { problem, model, opt, sampler, tape: Tape::new(), steps: 0 }
    }

    /// Train towards `target` steps, within the budget (at least one step); true when
    /// there. A target behind the present weights starts again from the first.
    fn work(&mut self, target: usize, budget: &Budget) -> bool {
        if target < self.steps {
            *self = Trainer::new();
        }
        while self.steps < target {
            let Trainer { problem, model, opt, sampler, tape, steps } = self;
            problem.grad(model, tape, sampler.deal());
            opt.step(model, TRAIN_LR);
            *steps += 1;
            if budget.spent() {
                break;
            }
        }
        self.steps == target
    }
}

// ----- what the visitor sets ---------------------------------------------------------------

/// The smallest and largest true gradient of one coordinate the control allows (in
/// units of one example's noise σ).
const LEAST_G: f64 = 0.001;
const MOST_G: f64 = 3.0;
/// One press of the button trains this many steps more; at most this many in all.
const TRAIN_MORE: usize = 500;
const MOST_TRAINED: usize = 20_000;

/// Which charts a narrow window shows.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Page {
    Coordinate,
    Network,
}

/// Everything the picture depends on that the visitor or an experiment sets.
#[derive(Clone, Copy, Debug, PartialEq)]
struct Setup {
    /// The one coordinate's true gradient in units of σ, when it is the visitor's own.
    g: f64,
    /// Or the layer's coordinate of this rank by its own noise scale (0: least noisy).
    pick: Option<usize>,
    /// The batch size: a power of two from 1 to MOST_B.
    b: usize,
    /// The momentum β of the average the steps are made from (0: none).
    beta: f64,
    /// A second batch size marked on the network's chart.
    compare: Option<usize>,
    /// The network's training steps.
    trained: usize,
    page: Page,
}

/// The nearest power of two to `b` on a logarithmic scale, from 1 to MOST_B.
fn power_of_two(b: f64) -> usize {
    if b.is_nan() || b < 1.0 {
        return 1;
    }
    1 << (b.log2().round().max(0.0) as usize).min(LEVELS - 1)
}

impl Setup {
    /// Kept to the controls' ranges (a value that is not a number takes a safe one).
    fn kept(self) -> Setup {
        let g = if self.g.is_finite() { self.g.max(LEAST_G).min(MOST_G) } else { 0.1 };
        let beta = if self.beta.is_finite() { self.beta.max(0.0).min(MOST_BETA) } else { 0.0 };
        Setup {
            g,
            pick: self.pick.map(|r| r.min(COORDS - 1)),
            b: power_of_two(self.b as f64),
            beta,
            compare: self.compare.map(|c| power_of_two(c as f64)),
            trained: self.trained.min(MOST_TRAINED),
            ..self
        }
    }
}

// ----- the experiments ---------------------------------------------------------------------

/// The rank (by own noise scale, least noisy first) of the coordinate the coin-toss
/// experiments take: about nine in ten of the layer's coordinates are less noisy.
const COIN_RANK: usize = 229;
/// Later in training: this many steps.
const LATER: usize = 3000;

struct Trial {
    button: &'static str,
    caption: &'static str,
}

const TRIALS: [Trial; 6] = [
    Trial {
        button: "a sure sign",
        caption: r"Entry $(15, 14)$ of layer 2, the least noisy of its 256 coordinates: its own noise scale $B_i = \sigma^2/g^2$ is 8.6 and the batch is 256, so $g\sqrt{B}/\sigma = 5.4$. The bell sits far to the right of zero: the formula gives the wrong sign a chance of $2.5 \times 10^{-8}$, and all 4 000 sampled batches had it right. Sign descent's step is the whole data's step, with no jitter ($1.00 \pm 0.00$), while SGD's still jitters by 0.18 of its mean step. Yet one example alone has this coordinate's sign right only 68 % of the time: it is the average of 256 that is sure.",
    },
    Trial {
        button: "a coin toss",
        caption: r"Entry $(4, 12)$: nine in ten of the layer's coordinates are less noisy. Its own noise scale is 1 238 and the batch 16, so $g\sqrt{B}/\sigma = 0.11$ and the bell straddles zero: the sign is right with probability 0.545 by the formula, and in 0.535 of 4 000 sampled batches. Sign descent steps a full $\pm 1$ every time, so its mean step is 0.09 and its jitter 1.00: a coin toss of full size, whose mean grows only as $\sqrt{B}$ until $B$ nears 1 238. SGD's mean step is the whole data's at any batch, with a jitter 8.8 times as large that shrinks as $1/\sqrt{B}$. Adam without momentum is as jittery as the sign ($0.11 \pm 0.99$).",
    },
    Trial {
        button: "average first: momentum",
        caption: r"The same entry with momentum $\beta = 0.9$: each step is made from an average over about ten batches, whose noise is that of a batch $(1 + \beta)/(1 - \beta) = 19$ times larger, 304 examples. The bell narrows by $\sqrt{19}$; the sign is right with probability 0.690 (in 0.682 of 2 000 sampled averages), and Signum's step is $0.38 \pm 0.93$. Adam's mean step stays at 0.11 but its jitter falls to 0.23: its numerator is averaged, its denominator (the size of single batches' gradients) is not. At $\beta = 0.99$ the average is worth 199 batches and the sign is right with probability 0.95. This is at fixed weights: in training the average holds gradients of weights that have since moved.",
    },
    Trial {
        button: "batch 4 against batch 256",
        caption: r"Layer 2's useful part, alone and then with momentum 0.9: at a batch of 4, SGD 0.34 and 0.84, sign descent 0.22 and 0.67, Adam 0.27 and 0.71, Muon 0.14 and 0.54; at 256, SGD 0.94 and 1.00, sign descent 0.82 and 0.95, Adam 0.80 and 0.94, Muon 0.76 and 0.98. At 4 every step without momentum is mostly jitter. By 256 SGD's curve has nearly stopped rising (its noise scale $B_{\text{simple}}$ is 31), while sign descent's and Muon's still rise: they give noisy coordinates and directions a full share of the step. Without momentum Muon's is the lowest of the four from a batch of 2 to 512, and second only to SGD's from 1 024.",
    },
    Trial {
        button: "later in training",
        caption: r"The network trained 3 000 steps (Adam, step 0.01, batches of 32): its loss fell from 1.14 to 0.00123 and layer 2's true gradient became 72 times shorter, but the noise did not shrink as much. $B_{\text{simple}}$ rose from 31 to 843, the median own noise scale from 43 to 2 000, and 11 of the 256 coordinates are below a batch of 256 (199 were). Every curve moved right: at 256 the useful part is now 0.48 for SGD, 0.26 for sign descent, 0.35 for Adam and 0.40 for Muon, against 0.94, 0.82, 0.80 and 0.76 before. Without momentum Muon is now above sign descent from a batch of 8 up. McCandlish and co-authors (2018) found the noise scale growing during training on their tasks too, typically tenfold or more.",
    },
    Trial {
        button: "Muon at a small batch",
        caption: r"A batch of 4. In this network one example's gradient of a weight matrix has rank one, since the layer meets one input vector per example (scene backpropmat), so the batch's has rank 4 at most: the cyan line stops at its fourth singular value. Its noise alone has a largest singular value of 0.26, over twice the true gradient's largest, 0.11: no direction of the true gradient stands above it. Muon's step from one such batch has a useful part of 0.14. Averaged first over about ten batches (momentum 0.9; Muon too averages first, with 0.95 in its reference code), it is 0.54, about what one batch of 64 gives. In the NanoGPT benchmark (a GPT-2-small-sized model, batches of half a million tokens) Muon without momentum needed 8 225 steps to reach the target and tuned Muon 3 250: momentum matters there too, at far larger batches, for reasons the benchmark does not settle.",
    },
];

/// The scene opens as this experiment sets it up.
const OPENING: usize = 1;

/// The setup of experiment k.
fn trial(k: usize) -> Setup {
    let base = Setup { g: 0.1, pick: Some(COIN_RANK), b: 16, beta: 0.0, compare: None, trained: 0, page: Page::Coordinate };
    match k {
        0 => Setup { pick: Some(0), b: 256, ..base },
        2 => Setup { beta: 0.9, ..base },
        3 => Setup { b: 4, compare: Some(256), beta: 0.9, page: Page::Network, ..base },
        4 => Setup { trained: LATER, b: 256, beta: 0.9, page: Page::Network, ..base },
        5 => Setup { b: 4, beta: 0.9, page: Page::Network, ..base },
        _ => base,
    }
}

// ----- numbers as the scene writes them ----------------------------------------------------

/// A number with three significant digits (0.0129, 1.96, 152), a true minus sign,
/// thousands set apart.
fn sig(v: f64) -> String {
    if !v.is_finite() {
        return "?".to_string();
    }
    if v == 0.0 {
        return "0".to_string();
    }
    let a = v.abs();
    if a >= 1000.0 {
        let n = count(a.round() as usize);
        return if v < 0.0 { format!("−{n}") } else { n };
    }
    let decimals = (2 - a.log10().floor() as i32).clamp(0, 12) as usize;
    num(v, decimals)
}

/// An own noise scale: one decimal below 10, else whole.
fn scale_text(v: f64) -> String {
    if v.is_finite() && v < 10.0 {
        num(v, 1)
    } else {
        whole(v)
    }
}

/// Two decimals (a share, a useful part).
fn two(v: f64) -> String {
    if v.is_finite() {
        num(v, 2)
    } else {
        "?".to_string()
    }
}

/// A batch size or a count, with thousands set apart.
fn whole(v: f64) -> String {
    if v.is_finite() && v >= 0.0 {
        count(v.round() as usize)
    } else if v.is_infinite() {
        "∞".to_string()
    } else {
        "?".to_string()
    }
}

// ----- the scene ---------------------------------------------------------------------------

pub struct SignNoise {
    setup: Setup,
    trainer: Trainer,
    /// Every example's gradient being measured (for the trainer's present weights).
    per: Option<PerExample>,
    /// The last layer measured, and what has been sampled of it.
    layer: Option<Layer>,
    plain: Option<Plain>,
    stream: Option<Stream>,
    sample: Option<CoordSample>,
    cov: Option<Covariances>,
    look: Look,
    /// The window last drawn in (a new one is drawn at once, without gliding).
    canvas: Option<Vec2>,
    dragging: bool,
}

impl SignNoise {
    /// (Nothing here needs the graphics device, so the tests can make one too.)
    fn create() -> SignNoise {
        let mut s = SignNoise {
            setup: trial(OPENING),
            trainer: Trainer::new(),
            per: None,
            layer: None,
            plain: None,
            stream: None,
            sample: None,
            cov: None,
            look: Look::default(),
            canvas: None,
            dragging: false,
        };
        s.set_up(OPENING);
        s
    }

    /// Set up experiment k.
    fn set_up(&mut self, k: usize) {
        self.setup = trial(k).kept();
    }

    /// Is any work still to be done for what the setup asks?
    fn busy(&self) -> bool {
        let s = &self.setup;
        let Some(layer) = &self.layer else { return true };
        if self.trainer.steps != s.trained || layer.step != s.trained || layer.nuclear[0].is_nan() {
            return true;
        }
        if !self.plain.as_ref().is_some_and(|p| p.done()) {
            return true;
        }
        if s.beta > 0.0 && !self.stream.as_ref().is_some_and(|m| m.beta == s.beta && m.done()) {
            return true;
        }
        if let Some(rank) = s.pick {
            let coord = layer.ranked[rank];
            if !self.sample.as_ref().is_some_and(|c| c.coord == coord && c.b == s.b && c.beta == s.beta && c.done()) {
                return true;
            }
        }
        false
    }

    /// The layer measured for the present setup, if it has been.
    fn current(&self) -> Option<&Layer> {
        self.layer.as_ref().filter(|l| l.step == self.setup.trained && self.trainer.steps == self.setup.trained)
    }

    /// Do what is owed, within the budget: train, measure every example, sample the
    /// chosen coordinate, the batches, and the stream with momentum, in that order.
    fn work(&mut self, budget: &Budget) {
        let s = self.setup;
        if self.trainer.steps != s.trained {
            self.per = None;
            if !self.trainer.work(s.trained, budget) || budget.spent() {
                return;
            }
        }
        if self.layer.as_ref().is_none_or(|l| l.step != s.trained) {
            let Trainer { problem, model, tape, .. } = &mut self.trainer;
            let per = self.per.get_or_insert_with(|| PerExample::new(problem, model));
            if !per.work(problem, model, tape, budget) {
                return;
            }
            self.layer = Some(Layer::of(per, model, problem, tape, s.trained));
            self.per = None;
            self.plain = Some(Plain::new(SAMPLE_SEED));
            self.stream = None;
            self.sample = None;
            if budget.spent() {
                return;
            }
        }
        if let Some(layer) = &mut self.layer {
            if layer.nuclear[0].is_nan() {
                let cov = self.cov.get_or_insert_with(|| Covariances::new(layer.step));
                if cov.step != layer.step {
                    *cov = Covariances::new(layer.step);
                }
                loop {
                    if let Some(n) = cov.advance(layer, 64) {
                        layer.nuclear = n;
                        self.cov = None;
                        break;
                    }
                    if budget.spent() {
                        return;
                    }
                }
            }
        }
        let Some(layer) = &self.layer else { return };
        if let Some(rank) = s.pick {
            let coord = layer.ranked[rank];
            if !self.sample.as_ref().is_some_and(|c| c.coord == coord && c.b == s.b && c.beta == s.beta) {
                self.sample = Some(CoordSample::new(coord, s.b, s.beta, SAMPLE_SEED + 2));
            }
            if let Some(c) = &mut self.sample {
                while !c.done() {
                    c.advance(layer, 64);
                    if budget.spent() {
                        return;
                    }
                }
            }
        }
        let plain = self.plain.get_or_insert_with(|| Plain::new(SAMPLE_SEED));
        while !plain.done() {
            plain.trial(layer);
            if budget.spent() {
                return;
            }
        }
        if s.beta > 0.0 {
            if !self.stream.as_ref().is_some_and(|m| m.beta == s.beta) {
                self.stream = Some(Stream::new(s.beta, SAMPLE_SEED + 1));
            }
            if let Some(m) = &mut self.stream {
                while !m.done() {
                    for _ in 0..8 {
                        m.step(layer);
                    }
                    if budget.spent() {
                        return;
                    }
                }
            }
        }
    }

    /// The one coordinate as the setup makes it: the visitor's own g, or the chosen
    /// coordinate of the layer (once it is measured), g = |Gᵢ|/σᵢ = 1/√Bᵢ.
    fn coordinate(&self) -> Coordinate {
        let s = &self.setup;
        let g = match (s.pick, &self.layer) {
            (Some(rank), Some(layer)) => {
                let own = layer.own[layer.ranked[rank]];
                (1.0 / own.sqrt()).max(LEAST_G).min(MOST_G)
            }
            _ => s.g,
        };
        let g = if g.is_finite() { g } else { LEAST_G };
        Coordinate { g, b: s.b as f64, beta: s.beta }
    }

    /// The chosen coordinate of the layer, as (row, column) from 1, and its index.
    fn picked(&self) -> Option<(usize, usize, usize)> {
        let rank = self.setup.pick?;
        let layer = self.layer.as_ref()?;
        let i = layer.ranked[rank];
        Some((i / SIDE + 1, i % SIDE + 1, i))
    }

    /// The finished sample of the chosen coordinate for the present setup.
    fn sample_now(&self) -> Option<&CoordSample> {
        let (_, _, i) = self.picked()?;
        self.current()?;
        self.sample.as_ref().filter(|c| c.coord == i && c.b == self.setup.b && c.beta == self.setup.beta && !c.values.is_empty())
    }

    /// The measured useful parts at level k: sign descent and Muon, alone and with the
    /// present momentum (NaN where not measured yet, or not at all).
    fn measured(&self, k: usize) -> [f64; 4] {
        let enough = |t: &Tally, n: usize| if t.n >= n { t.useful() } else { f64::NAN };
        let mut v = [f64::NAN; 4];
        if let Some(p) = &self.plain {
            // (at a batch of one, sign descent's is known exactly)
            v[0] = match (&self.layer, k) {
                (Some(l), 0) => l.sign1,
                _ => enough(&p.sign[k], 10),
            };
            v[1] = enough(&p.muon[k], 10);
        }
        if let Some(m) = self.stream.as_ref().filter(|m| m.beta == self.setup.beta && self.setup.beta > 0.0 && k < m.levels) {
            v[2] = enough(&m.sign[k], 10);
            v[3] = enough(&m.muon[k], 10);
        }
        v
    }
}

// ----- the layout ------------------------------------------------------------------------------

/// Where everything goes in a window (logical pixels from its top-left corner).
#[derive(Clone, Copy, Debug, PartialEq)]
struct Layout {
    view: Vec2,
    narrow: bool,
    text: f32,
    small: f32,
    /// Train more, back to the start; the two pages (narrow windows only).
    train: Rect,
    again: Rect,
    pages: [Rect; 2],
    /// Where the status line starts (its left end, its middle height).
    status: Vec2,
    /// The charts' own rectangles (titles, ticks and names go outside them).
    bell: Rect,
    steps: Rect,
    useful: Rect,
    strip: Rect,
    spectrum: Rect,
    /// Which halves are drawn.
    one: bool,
    net: bool,
}

/// Room left of a chart (tick numbers), under it (ticks, the axis's name, a legend
/// line) and over it (its title).
const LEFT: f32 = 44.0;
const BELOW: f32 = 40.0;
/// Room for a legend line under a chart.
const LEGEND: f32 = 20.0;
const ABOVE: f32 = 30.0;

impl Layout {
    fn of(view: Vec2, inset: f32, page: Page) -> Layout {
        let view = view.max(Vec2::splat(1.0));
        let narrow = view.x < 760.0;
        let (text, small) = if narrow { (12.0, 10.5) } else { (13.5, 11.5) };
        let side = if narrow { 8.0 } else { 18.0 };
        let top = inset.max(0.0) + if narrow { 6.0 } else { 10.0 };
        let r = |x0: f32, y0: f32, x1: f32, y1: f32| (Vec2::new(x0, y0), Vec2::new(x1.max(x0 + 1.0), y1.max(y0 + 1.0)));
        let none = r(0.0, 0.0, 0.0, 0.0);
        let button_h = if narrow { 24.0 } else { 26.0 };
        let mut lay = Layout {
            view,
            narrow,
            text,
            small,
            train: none,
            again: none,
            pages: [none; 2],
            status: Vec2::ZERO,
            bell: none,
            steps: none,
            useful: none,
            strip: none,
            spectrum: none,
            one: false,
            net: false,
        };
        let wide = view.x >= 900.0 && view.y >= 560.0;
        let mut y = top;
        if !wide {
            // the two pages, side by side
            let row = plot::even_row(2, side, view.x - side, y, button_h, 6.0);
            lay.pages = [row[0], row[1]];
            y += button_h + 6.0;
        }
        let bw = if narrow { 112.0 } else { 150.0 };
        lay.train = r(side, y, side + bw, y + button_h);
        lay.again = r(side + bw + 6.0, y, side + 2.0 * bw - 20.0, y + button_h);
        lay.status = Vec2::new(lay.again.1.x + 10.0, y + 0.5 * button_h);
        y += button_h + 8.0;
        let bottom = view.y - 6.0;
        let chart = |x0: f32, y0: f32, x1: f32, y1: f32| r(x0 + LEFT, y0 + ABOVE, x1 - 6.0, y1 - BELOW);
        if wide {
            let column = (view.x - 3.0 * side) * 0.5;
            let (x0, x1) = (side, side + column);
            let h = bottom - y;
            let split = y + h * 0.42;
            lay.bell = chart(x0, y, x1, split);
            lay.steps = chart(x0, split + 8.0, x1, bottom - LEGEND);
            let (x0, x1) = (2.0 * side + column, view.x - side);
            Layout::network(&mut lay, x0, x1, y, bottom);
            lay.one = true;
            lay.net = true;
        } else {
            let h = bottom - y;
            match page {
                Page::Coordinate => {
                    let split = y + h * 0.45;
                    lay.bell = chart(side, y, view.x - side, split);
                    lay.steps = chart(side, split + 6.0, view.x - side, bottom - LEGEND);
                    lay.one = true;
                }
                Page::Network => {
                    Layout::network(&mut lay, side, view.x - side, y, bottom);
                    lay.net = true;
                }
            }
        }
        lay
    }

    /// The network's charts between `x0` and `x1`, from `y` down to `bottom`: the useful
    /// parts, the strip of own noise scales under them (sharing their axis), the spectra.
    fn network(lay: &mut Layout, x0: f32, x1: f32, y: f32, bottom: f32) {
        let r = |x0: f32, y0: f32, x1: f32, y1: f32| (Vec2::new(x0, y0), Vec2::new(x1.max(x0 + 1.0), y1.max(y0 + 1.0)));
        let h = bottom - y;
        let low = y + h * 0.46;
        lay.useful = r(x0 + LEFT, y + ABOVE, x1 - 6.0, low);
        let strip_h = if lay.narrow { 28.0 } else { 34.0 };
        lay.strip = r(lay.useful.0.x, low + 22.0, lay.useful.1.x, low + 22.0 + strip_h);
        lay.spectrum = r(x0 + LEFT, lay.strip.1.y + 22.0 + ABOVE, x1 - 6.0, bottom - BELOW - LEGEND);
    }

    /// Is the window too small for the charts?
    fn tiny(&self) -> bool {
        self.view.x < 160.0 || self.view.y < 200.0
    }
}

// ----- what is drawn, gliding ----------------------------------------------------------------

/// The points each curve of the one coordinate's chart and each formula of the network's
/// chart is drawn with.
const LINE_N: usize = 64;
/// The bins of a coordinate's histogram, and of the layer's own noise scales (eighths
/// of a decade from 1 to MOST_B, and one for those above).
const HIST_BINS: usize = 48;
const STRIP_PER_DECADE: f64 = 8.0;
/// log₁₀ of the largest batch.
fn most_x() -> f64 {
    (MOST_B as f64).log10()
}
fn strip_bins() -> usize {
    (most_x() * STRIP_PER_DECADE).ceil() as usize
}

/// A value that is not a number glides as this (off every chart) and is not drawn.
const NOWHERE: f32 = -30.0;

// The network chart's numbers, in the order the glide keeps them: the measured curves
// (sign descent and Muon, alone and with momentum, LEVELS each), then the formulas (SGD
// and Adam alone and with momentum, and sign descent's normal formula, LINE_N each).
const MEASURED: usize = 4;
const FORMULAS: usize = 5;
fn net_len() -> usize {
    MEASURED * LEVELS + FORMULAS * LINE_N
}

/// What is drawn, gliding to where it is going.
#[derive(Clone, Debug, Default)]
struct Look {
    /// The one coordinate: ln g, ln B, ln of the batch the average is worth; ln of the
    /// compared batch; how far the measured histogram rises over the bell's peak.
    one: Glide,
    /// Its histogram (density in units of σ, HIST_BINS bars).
    hist: Glide,
    /// The network's chart, its strip of own noise scales, its spectra (log₁₀ of the true
    /// gradient's, the batch's and the noise's singular values).
    net: Glide,
    /// Where the network's numbers have a value, and what they were worked out for
    /// (they read every coordinate at every point: worked out again only on a change).
    net_mask: Vec<bool>,
    net_key: Option<NetKey>,
    net_goal: Vec<f32>,
    strip: Glide,
    spectrum: Glide,
    moving: bool,
}

/// Follow `to` as `Glide::follow` does, except that a number which appears (from
/// NOWHERE) or goes (to NOWHERE) is there at once: a curve that gains a point does not
/// rise into the chart from far below it. (Such a change restarts the eases of the
/// others, which bends their glide but does not move them.)
fn follow_appearing(g: &mut Glide, to: &[f32], dt: f32) -> bool {
    let gone = |v: f32| v <= NOWHERE + 1.0;
    if g.shown.len() == to.len() && g.shown.iter().zip(to).any(|(&a, &b)| gone(a) != gone(b)) {
        let mixed: Vec<f32> = g.shown.iter().zip(to).map(|(&a, &b)| if gone(a) != gone(b) { b } else { a }).collect();
        g.snap(&mixed);
    }
    g.follow(to, dt, false)
}

/// What the network chart's numbers depend on: the layer's step, the batches sampled,
/// the momentum and the stream's recordings.
type NetKey = (usize, usize, u64, usize);

/// The x of point k of LINE_N across the network's chart (log₁₀ B).
fn across(k: usize) -> f64 {
    most_x() * k as f64 / (LINE_N - 1) as f64
}

/// A number for a glide (one that is not a number goes NOWHERE).
fn keep(v: f64) -> f32 {
    if v.is_finite() {
        (v as f32).max(-1e6).min(1e6)
    } else {
        NOWHERE
    }
}

impl SignNoise {
    /// The bell chart's range for a coordinate: x round the true gradient and zero, y over
    /// the peak of the density and of the histogram.
    fn bell_goal(&self, c: &Coordinate, hist_top: f64) -> [f32; 3] {
        // (across, one batch's bell, so that momentum narrows the bell in a frame that
        // stays; up, the bell drawn, the average's)
        let one = 1.0 / c.b.max(1.0).sqrt();
        let used = one / worth(c.beta).sqrt();
        let half = 3.6 * one;
        // (the smaller of g − half and 0, and the larger of g + half and 0, with the
        // corner rounded: a range that glides through it does not stop dead there)
        let k = 0.6 * one;
        let smooth = |a: f64, b: f64, side: f64| 0.5 * (a + b + side * ((a - b).powi(2) + k * k).sqrt());
        let (mut lo, mut hi) = (smooth(c.g - half, 0.0, -1.0), smooth(c.g + half, 0.0, 1.0));
        let pad = 0.08 * (hi - lo);
        lo -= pad;
        hi += pad;
        let peak = 1.0 / (used * (2.0 * std::f64::consts::PI).sqrt());
        [lo as f32, hi as f32, (1.4 * peak.max(hist_top)) as f32]
    }

    /// The chosen coordinate's histogram over the range `lo..hi` (density, per unit of
    /// σ), and its tallest bar.
    fn hist_goal(&self, lo: f64, hi: f64) -> (Vec<f32>, f64) {
        let mut bins = vec![0.0f64; HIST_BINS];
        let Some(c) = self.sample_now() else { return (vec![0.0; HIST_BINS], 0.0) };
        let width = (hi - lo) / HIST_BINS as f64;
        if width.is_nan() || width <= 0.0 {
            return (vec![0.0; HIST_BINS], 0.0);
        }
        for &x in &c.values {
            let k = ((x - lo) / width).floor();
            if k >= 0.0 && (k as usize) < HIST_BINS {
                bins[k as usize] += 1.0;
            }
        }
        let n = c.values.len().max(1) as f64;
        let dens: Vec<f64> = bins.iter().map(|b| b / (n * width)).collect();
        let top = dens.iter().copied().fold(0.0, f64::max);
        (dens.iter().map(|&d| d as f32).collect(), top)
    }

    /// The network chart's numbers as they are to be (see `net_len`), and where each has
    /// a value.
    fn net_goal(&self) -> (Vec<f32>, Vec<bool>) {
        let mut v = Vec::with_capacity(net_len());
        let mut mask = Vec::with_capacity(net_len());
        for r in 0..MEASURED {
            for k in 0..LEVELS {
                let m = self.measured(k)[r];
                v.push(keep(m));
                mask.push(m.is_finite());
            }
        }
        let beta = self.setup.beta;
        let w = worth(beta);
        for r in 0..FORMULAS {
            for k in 0..LINE_N {
                let b = 10f64.powf(across(k));
                let value = match (&self.layer, r) {
                    (Some(l), 0) => l.sgd_useful(b),
                    (Some(l), 1) if beta > 0.0 => l.sgd_useful(b * w),
                    (Some(l), 2) => l.adam_useful(b, 1.0).0,
                    (Some(l), 3) if beta > 0.0 => l.adam_useful(b, w).0,
                    (Some(l), 4) => l.sign_normal(b),
                    _ => f64::NAN,
                };
                v.push(keep(value));
                mask.push(value.is_finite());
            }
        }
        (v, mask)
    }

    /// The strip's bars: how many coordinates have their own noise scale in each eighth
    /// of a decade, and (last) above the largest batch.
    fn strip_goal(&self) -> Vec<f32> {
        let n = strip_bins();
        let mut bins = vec![0.0f32; n + 1];
        if let Some(l) = &self.layer {
            for &s in &l.own {
                let x = s.max(1.0).log10() * STRIP_PER_DECADE;
                let k = if s.is_finite() && x < n as f64 { x.floor().max(0.0) as usize } else { n };
                bins[k.min(n)] += 1.0;
            }
        }
        bins
    }

    /// The spectra: log₁₀ of the true gradient's singular values, of the batch's (the
    /// mean over SPECTRUM_TRIALS batches of each sorted value) and of the noise's, at the
    /// present batch size. Values below 10⁻⁷ of the largest of their kind are rounding
    /// (`optim::singular_values_into` does not resolve them) and are NOWHERE.
    fn spectrum_goal(&self) -> Vec<f32> {
        let mut v = vec![NOWHERE; 3 * SIDE];
        if let (Some(l), Some(p)) = (&self.layer, &self.plain) {
            if p.trials > 0 {
                let k = power_of_two(self.setup.b as f64).trailing_zeros() as usize;
                let (batch, noise) = p.spectra(k);
                for (j, set) in [&l.sv[..], &batch[..], &noise[..]].into_iter().enumerate() {
                    let top = set.iter().copied().fold(0.0, f64::max);
                    for (i, &x) in set.iter().enumerate().take(SIDE) {
                        if top > 0.0 && x > 1e-7 * top {
                            v[j * SIDE + i] = x.log10() as f32;
                        }
                    }
                }
            }
        }
        // the chart's range, whole decades round what is drawn (glided with the rest)
        let (lo, hi) = v.iter().filter(|&&x| x > NOWHERE + 1.0).fold((f32::INFINITY, f32::NEG_INFINITY), |(a, b), &x| (a.min(x), b.max(x)));
        let (lo, hi) = if lo.is_finite() { ((lo - 0.3).floor(), (hi + 0.3).ceil()) } else { (-6.0, 0.0) };
        v.push(lo);
        v.push(hi.max(lo + 1.0));
        v
    }

    /// Bring the picture `dt` seconds nearer to what it shows. (`snap`: there at once.)
    fn follow(&mut self, dt: f32, snap: bool) {
        let s = self.setup;
        let c = self.coordinate();
        let compare = s.compare.unwrap_or(s.b) as f64;
        let rough = self.bell_goal(&c, 0.0);
        let (hist, top) = self.hist_goal(rough[0] as f64, rough[1] as f64);
        // (how far the measured histogram rises over the bell's peak: glided with the
        // coordinate, so that the bell, its histogram and its frame move as one)
        let peak = rough[2] as f64 / 1.4;
        let over = if peak > 0.0 { (top / peak).max(1.0) } else { 1.0 };
        let one = vec![c.g.ln() as f32, (c.b.ln()) as f32, worth(c.beta).ln() as f32, compare.ln() as f32, over as f32];
        let key = (
            self.layer.as_ref().map_or(usize::MAX, |l| l.step),
            self.plain.as_ref().map_or(0, |p| p.trials),
            s.beta.to_bits(),
            self.stream.as_ref().filter(|m| m.beta == s.beta).map_or(0, |m| m.samples),
        );
        if self.look.net_key != Some(key) {
            let (goal, mask) = self.net_goal();
            self.look.net_goal = goal;
            self.look.net_mask = mask;
            self.look.net_key = Some(key);
        }
        let net = self.look.net_goal.clone();
        let strip = self.strip_goal();
        let spectrum = self.spectrum_goal();
        let look = &mut self.look;
        if snap {
            for (g, to) in [(&mut look.one, &one), (&mut look.hist, &hist), (&mut look.net, &net), (&mut look.strip, &strip), (&mut look.spectrum, &spectrum)] {
                g.snap(to);
            }
            look.moving = false;
            return;
        }
        let mut moving = false;
        for (g, to) in [(&mut look.one, &one), (&mut look.hist, &hist), (&mut look.strip, &strip)] {
            moving |= g.follow(to, dt, false);
        }
        moving |= follow_appearing(&mut look.net, &net, dt);
        moving |= follow_appearing(&mut look.spectrum, &spectrum, dt);
        look.moving = moving;
    }

    /// The one coordinate as it is drawn now (following the glide).
    fn shown_coordinate(&self) -> Coordinate {
        let one = &self.look.one;
        if one.shown.len() < 3 {
            return self.coordinate();
        }
        Coordinate { g: (one.get(0) as f64).exp(), b: (one.get(1) as f64).exp(), beta: beta_of(((one.get(2) as f64).exp()).max(1.0)) }
    }
}

/// The momentum whose average is worth a batch `w` times larger: β = (w − 1)/(w + 1).
fn beta_of(w: f64) -> f64 {
    ((w - 1.0) / (w + 1.0)).max(0.0).min(MOST_BETA)
}

// ----- drawing ---------------------------------------------------------------------------------

// Lines whose number of points changes (they are cut at a chart's edge), each in a batch of
// its own.
const BELL: u8 = plot::OWN;
const BELL_ONE: u8 = BELL + 1;
const HIST: u8 = BELL_ONE + 1;
const STEP: u8 = HIST + 1;
const NET: u8 = STEP + 6;
const SPECTRUM: u8 = NET + MEASURED as u8 + FORMULAS as u8;
const MARKER: u8 = SPECTRUM + 3;
const GUIDE: u8 = MARKER + 1;

/// The point batches: the dots on the step chart, on the network's chart, on the spectra.
fn dot_batch<'b>(f: &'b mut Frame<'_>, k: usize) -> &'b mut crate::gfx::draw::PointBatch {
    f.dl.points_with(PointKind::Overlay, PointStyle { outline_px: 0.0, size_scale: 1.0 + 1e-4 * (k + 1) as f32, ..Default::default() })
}

/// Every batch the scene draws in, made in the same order in every frame, so that one
/// that is empty in a frame does not move the later ones up a place (see `eidos motion`).
fn hold_batches(f: &mut Frame) {
    let corner = f.hud(Vec2::ZERO);
    for kind in [AXES, MARKS, plot::BUTTONS].into_iter().chain(BELL..=GUIDE) {
        plot::apart(f, kind, false).segment(corner, corner, 1.0, Color::CLEAR);
    }
    for kind in [BELL_ONE, STEP + 3, STEP + 4, STEP + 5, NET + 2, NET + 3, NET + 5, NET + 7, MARKER, GUIDE] {
        plot::apart(f, kind, true).segment(corner, corner, 1.0, Color::CLEAR);
    }
    for k in 0..3 {
        dot_batch(f, k);
    }
}

/// Points (in the plot's units) as a dashed line kept inside the plot.
fn dashed_path(f: &mut Frame, p: &Plot, kind: u8, pts: &[(f64, f64)], width: f32, color: Color) {
    let px: Vec<Vec2> = pts.iter().map(|&(x, y)| p.at(x, y)).collect();
    let cut = plot::clip(&px, p.rect);
    for piece in cut.split(|q| !q.is_finite()).filter(|piece| piece.len() >= 2) {
        let points: Vec<Vec3> = piece.iter().map(|q| f.hud(*q)).collect();
        plot::apart(f, kind, true).polyline(points, width, color);
    }
}

/// Dots in screen space, in batch k.
fn dots_px(f: &mut Frame, k: usize, pts: &[(Vec2, Color)], size: f32) {
    let at: Vec<(Vec3, Color)> = pts.iter().filter(|(p, _)| p.is_finite()).map(|(p, c)| (f.hud(*p), *c)).collect();
    let b = dot_batch(f, k);
    for (p, c) in at {
        b.push(p, size, c);
    }
}

/// The batch sizes named on an axis: powers of four.
fn batch_ticks() -> impl Iterator<Item = (f64, String)> {
    (0..LEVELS).step_by(2).map(|k| ((size_of(k) as f64).log10(), count(size_of(k))))
}

/// A log₁₀ B axis along the bottom of a plot, with its name.
fn batch_axis(f: &mut Frame, p: &Plot, size: f32, name: &str) {
    let (min, max) = p.rect;
    plot::line_of(f, AXES, Vec2::new(min.x, max.y), max, 1.0, pal::DIM.a(0.8));
    for (x, label) in batch_ticks() {
        let at = p.px_x(x);
        plot::line_of(f, AXES, Vec2::new(at, max.y), Vec2::new(at, max.y + 4.0), 1.0, pal::DIM.a(0.8));
        f.text_px(Vec2::new(at, max.y + 5.0), label, size, pal::DIM, Align2::CENTER_TOP);
    }
    f.text_px(Vec2::new(max.x, max.y + 7.0 + size), name, size, pal::DIM, Align2::RIGHT_TOP);
}

/// A chart's title over its top-left corner.
fn title(f: &mut Frame, p: &Plot, text: &str, size: f32) {
    f.text_px(Vec2::new(p.rect.0.x - LEFT + 4.0, p.rect.0.y - 10.0), text, size, pal::TEXT, Align2::LEFT_BOTTOM);
}

/// A legend: coloured entries in a row from `at`, wrapping to `right`; dashed entries
/// marked so. Returns the height used.
fn legend(f: &mut Frame, at: Vec2, right: f32, entries: &[(Color, bool, &str)], size: f32) -> f32 {
    let (mut x, mut y) = (at.x, at.y);
    let line = size + 5.0;
    for &(c, dashed, text) in entries {
        let w = f.text_size(text, size).x + 26.0;
        if x + w > right && x > at.x {
            x = at.x;
            y += line;
        }
        let (a, b) = (Vec2::new(x, y + 0.5 * size), Vec2::new(x + 16.0, y + 0.5 * size));
        if dashed {
            plot::dashed_of(f, MARKS, a, b, 2.0, c);
        } else {
            plot::line_of(f, MARKS, a, b, 2.4, c);
        }
        f.text_px(Vec2::new(x + 20.0, y), text, size, c, Align2::LEFT_TOP);
        x += w + 10.0;
    }
    y + line - at.y
}

impl SignNoise {
    /// The buttons over the picture; true if anything changed.
    fn buttons(&mut self, f: &mut Frame, lay: &Layout) -> bool {
        let mut changed = false;
        let size = lay.small;
        let train = if lay.narrow { format!("train {TRAIN_MORE}") } else { format!("train {TRAIN_MORE} more steps") };
        if plot::button(f, lay.train, &train, size, false) && self.setup.trained < MOST_TRAINED {
            self.setup.trained = (self.setup.trained + TRAIN_MORE).min(MOST_TRAINED);
            changed = true;
        }
        if plot::button(f, lay.again, if lay.narrow { "restart" } else { "back to the start" }, size, false) && self.setup.trained != 0 {
            self.setup.trained = 0;
            changed = true;
        }
        if !lay.one || !lay.net {
            for (k, (name, page)) in [("one coordinate", Page::Coordinate), ("a layer of the network", Page::Network)].into_iter().enumerate() {
                if plot::button(f, lay.pages[k], name, size, self.setup.page == page) && self.setup.page != page {
                    self.setup.page = page;
                    changed = true;
                }
            }
        }
        changed
    }

    /// The batch size, dragged along either chart's axis.
    fn drag(&mut self, f: &mut Frame, lay: &Layout) {
        // (the dot is where the picture is: it glides to each power of two in turn as the
        // pointer passes it, rather than snapping)
        let shown = self.look.one.get(1) as f64;
        let x = if self.look.one.shown.len() > 1 && shown.is_finite() { shown.exp().log10() } else { (self.setup.b as f64).log10() };
        for (id, rect, lo) in [(1u32, lay.steps, -1.2), (2, lay.useful, 0.0)] {
            let on = if id == 1 { lay.one } else { lay.net };
            if !on {
                continue;
            }
            let p = Plot::new(rect, [0.0, most_x()], [lo, lo + 1.0]);
            if p.tiny() {
                continue;
            }
            if let Some((to, _)) = p.handle(f, id, x, lo, pal::WHITE) {
                if to.is_finite() {
                    self.setup.b = power_of_two(10f64.powf(to));
                    self.dragging = true;
                }
            }
        }
    }

    /// The status line: what the network is, and what is being worked out.
    fn status(&self, f: &mut Frame, lay: &Layout) {
        let s = &self.setup;
        let mut text = match self.current() {
            Some(l) => format!("trained {} steps, loss {}", count(l.step), sig(l.loss)),
            None => format!("trained {} steps", count(self.trainer.steps)),
        };
        if self.trainer.steps != s.trained {
            text += &format!(" · training to {}", count(s.trained));
        } else if self.current().is_none() {
            text += " · measuring every example's gradient";
        } else if !self.plain.as_ref().is_some_and(|p| p.done()) {
            text += &format!(" · sampling batches: {} of {SAMPLED}", self.plain.as_ref().map_or(0, |p| p.trials));
        } else if s.beta > 0.0 && !self.stream.as_ref().is_some_and(|m| m.done()) {
            text += &format!(" · with momentum: {} of {STREAM_SAMPLES}", self.stream.as_ref().map_or(0, |m| m.samples));
        }
        let room = lay.view.x - lay.status.x - 6.0;
        if room > 60.0 {
            f.text_px(lay.status, text, lay.small, pal::DIM, Align2::LEFT_CENTER);
        }
    }

    fn draw_bell(&self, f: &mut Frame, lay: &Layout) {
        // (the range follows the shown coordinate itself, so that the bell and its frame
        // glide as one)
        let c = self.shown_coordinate();
        let r = self.bell_goal(&c, 0.0);
        let r = [r[0], r[1], r[2] * self.look.one.get(4).max(1.0)];
        let p = Plot::new(lay.bell, [r[0] as f64, r[1] as f64], [0.0, (r[2] as f64).max(1e-6)]);
        if p.tiny() {
            return;
        }
        let size = lay.small;
        p.backing(f, LEFT, BELOW - 6.0, ABOVE, 8.0);
        let w = worth(c.beta);
        let s = 1.0 / c.b.max(1.0).sqrt();
        let used = s / w.sqrt();
        let g = c.g;
        let head = match (self.picked(), lay.narrow) {
            (Some((i, j, _)), false) => format!(r"One coordinate, entry $({i}, {j})$ of layer 2: its gradient from a batch of $B = {}$", whole(c.b)),
            (Some((i, j, _)), true) => format!(r"Entry $({i}, {j})$ of layer 2, batch $B = {}$", whole(c.b)),
            (None, false) => format!(r"One coordinate: its gradient from a batch of $B = {}$", whole(c.b)),
            (None, true) => format!(r"One coordinate, batch $B = {}$", whole(c.b)),
        };
        title(f, &p, &head, lay.text);
        let density = move |x: f64, sd: f64| (-0.5 * ((x - g) / sd).powi(2)).exp() / (sd * (2.0 * std::f64::consts::PI).sqrt());
        // the measured batches, behind
        let hist = &self.look.hist.shown;
        if hist.len() == HIST_BINS && hist.iter().any(|&h| h > 0.0) {
            let heights: Vec<f64> = hist.iter().map(|&h| h as f64).collect();
            let width = (p.x[1] - p.x[0]) / HIST_BINS as f64;
            p.bars(f, p.x[0], width, &heights, 0.0, pal::DIM.a(0.16));
            let mut outline = Vec::with_capacity(2 * HIST_BINS + 2);
            outline.push((p.x[0], 0.0));
            for (k, &h) in heights.iter().enumerate() {
                outline.push((p.x[0] + k as f64 * width, h));
                outline.push((p.x[0] + (k + 1) as f64 * width, h));
            }
            outline.push((p.x[1], 0.0));
            p.path(f, HIST, outline, 1.2, pal::DIM.mix(pal::TEXT, 0.4));
        }
        // the two sides of zero
        p.area(f, p.x[0], 0.0, 0.0, 90, NOISE.a(0.42), |x| density(x, used));
        p.area(f, 0.0, p.x[1], 0.0, 90, SIGNAL.a(0.10), |x| density(x, used));
        if w > 1.0 + 1e-6 {
            let pts: Vec<(f64, f64)> = (0..160).map(|k| p.x[0] + (p.x[1] - p.x[0]) * k as f64 / 159.0).map(|x| (x, density(x, s))).collect();
            dashed_path(f, &p, BELL_ONE, &pts, 1.3, pal::TEXT.a(0.6));
        }
        p.curve(f, BELL, 160, 2.0, pal::TEXT, |x| density(x, used));
        p.vline(f, MARKS, 0.0, 1.0, pal::DIM, false);
        p.vline(f, GUIDE, g, 1.8, SIGNAL, false);
        let (min, max) = p.rect;
        let gx = p.px_x(g);
        if gx >= min.x && gx <= max.x {
            // (on the side away from the plate of shares, at the top right)
            let (dx, align) = if gx > 0.5 * (min.x + max.x) { (-4.0, Align2::RIGHT_TOP) } else { (4.0, Align2::LEFT_TOP) };
            f.text_px(Vec2::new(gx + dx, min.y + 4.0), r"true gradient $g$", size, SIGNAL, align);
        }
        // the shares either side
        let right = normal_cdf(g / used);
        let mut lines = vec![(format!("right sign {}", two(right)), SIGNAL), (format!("wrong sign {}", two(1.0 - right)), NOISE)];
        if let Some(sample) = self.sample_now() {
            lines.push((format!("measured: right in {} of {}", two(sample.share_right()), count(sample.values.len())), pal::DIM.mix(pal::TEXT, 0.5)));
        }
        for (k, (t, col)) in lines.iter().enumerate() {
            f.caption_px(Vec2::new(max.x - 6.0, min.y + 4.0 + k as f32 * (size + 6.0)), t.clone(), size, *col, Align2::RIGHT_TOP);
        }
        // the axis: zero and the true gradient marked, and one batch's spread σ/√B (or the
        // average's) as a bracket round g, instead of ticks that would slide as it zooms
        plot::line_of(f, AXES, Vec2::new(min.x, max.y), max, 1.0, pal::DIM.a(0.8));
        // (the two names side by side, away from each other, when the marks are close)
        let close = (p.px_x(g) - p.px_x(0.0)).abs() < 24.0;
        for (x, name, side) in [(0.0, "0", -1.0f32), (g, "$g$", 1.0)] {
            let at = p.px_x(x).max(min.x).min(max.x);
            plot::line_of(f, AXES, Vec2::new(at, max.y), Vec2::new(at, max.y + 4.0), 1.0, pal::DIM.a(0.8));
            let (dx, align) = match (close, side < 0.0) {
                (false, _) => (0.0, Align2::CENTER_TOP),
                (true, true) => (-2.0, Align2::RIGHT_TOP),
                (true, false) => (2.0, Align2::LEFT_TOP),
            };
            f.text_px(Vec2::new(at + dx, max.y + 5.0), name, size, pal::DIM, align);
        }
        let (a, b) = (p.px_x(g - used).max(min.x), p.px_x(g + used).min(max.x));
        let y = max.y - 10.0;
        plot::line_of(f, AXES, Vec2::new(a, y), Vec2::new(b, y), 1.2, pal::TEXT.a(0.7));
        for x in [a, b] {
            plot::line_of(f, AXES, Vec2::new(x, y - 3.0), Vec2::new(x, y + 3.0), 1.2, pal::TEXT.a(0.7));
        }
        let spread = if w > 1.0 + 1e-6 { r"$\pm\,\sigma/\sqrt{B \cdot 19}$".replace("19", &short(w)) } else { r"$\pm\,\sigma/\sqrt{B}$".to_string() };
        f.text_px(Vec2::new(b + 4.0, y), spread, size, pal::TEXT.a(0.8), Align2::LEFT_CENTER);
        let name = if w > 1.0 + 1e-6 {
            r"the average over about $1/(1 - \beta)$ batches, in units of one example's noise $\sigma$ (dashed: one batch)".to_string()
        } else {
            r"the batch's gradient, in units of one example's noise $\sigma$".to_string()
        };
        let name = if lay.narrow { r"gradient in units of $\sigma$".to_string() } else { name };
        f.text_px(Vec2::new(max.x, max.y + 7.0 + size), name, size, pal::DIM, Align2::RIGHT_TOP);
    }

    fn draw_steps(&self, f: &mut Frame, lay: &Layout) {
        let p = Plot::new(lay.steps, [0.0, most_x()], [0.0, 2.1]);
        if p.tiny() {
            return;
        }
        let size = lay.small;
        p.backing(f, LEFT, BELOW + LEGEND, ABOVE, 8.0);
        let c = self.shown_coordinate();
        title(f, &p, if lay.narrow { "Its step against the batch size" } else { "Its step against the batch size, in units of the whole data's step" }, lay.text);
        let (min, max) = p.rect;
        // the axis up: 0, 1 (the whole data's step) and 2
        plot::line_of(f, AXES, min, Vec2::new(min.x, max.y), 1.0, pal::DIM.a(0.8));
        for (y, name) in [(0.0, "0"), (0.5, "0.5"), (1.0, "1"), (1.5, "1.5"), (2.0, "2")] {
            let at = p.px_y(y);
            plot::line_of(f, AXES, Vec2::new(min.x - 4.0, at), Vec2::new(min.x, at), 1.0, pal::DIM.a(0.8));
            plot::line_of(f, AXES, Vec2::new(min.x, at), Vec2::new(max.x, at), 1.0, pal::DIM.a(if y == 1.0 { 0.4 } else { 0.1 }));
            f.text_px(Vec2::new(min.x - 6.0, at), name, size, pal::DIM, Align2::RIGHT_CENTER);
        }
        // the coordinate's own noise scale, and the two regimes either side, over the
        // line of the whole data's step (where only SGD's jitter runs)
        let own = 1.0 / (c.g * c.g);
        let worth_now = worth(c.beta);
        let ox = (own / worth_now).log10();
        let high = p.px_y(1.12);
        if ox > p.x[0] && ox < p.x[1] {
            let at = p.px_x(ox);
            plot::dashed_of(f, GUIDE, Vec2::new(at, min.y), Vec2::new(at, max.y), 1.2, pal::DIM);
            let name = if worth_now > 1.0 + 1e-6 { format!(r"$B_i/{} = {}$", short(worth_now), scale_text(own / worth_now)) } else { format!(r"$B_i = \sigma^2/g^2 = {}$", scale_text(own)) };
            let wide = f.text_size(&name, size).x;
            let align = if at + wide + 8.0 < max.x { Align2::LEFT_BOTTOM } else { Align2::RIGHT_BOTTOM };
            let dx = if align == Align2::LEFT_BOTTOM { 4.0 } else { -4.0 };
            f.text_px(Vec2::new(at + dx, p.px_y(1.5)), name, size, pal::DIM, align);
        }
        let left = (own / (4.0 * worth_now)).log10().min(p.x[1]);
        if left > p.x[0] + 0.35 {
            f.text_px(Vec2::new(p.px_x(left) - 4.0, high), "coin toss", size, NOISE, Align2::RIGHT_BOTTOM);
        }
        let right = (own * 4.0 / worth_now).log10().max(p.x[0]);
        if right < p.x[1] - 0.35 {
            f.text_px(Vec2::new(p.px_x(right) + 4.0, high), "sure sign", size, SIGNAL, Align2::LEFT_BOTTOM);
        }
        // each rule's mean step (solid) and its standard deviation (dashed)
        let rules = step_rules(c.beta);
        let at = |x: f64| Coordinate { b: 10f64.powf(x), ..c };
        for (k, (color, _, rule)) in rules.into_iter().enumerate() {
            // (SGD's jitter, 1/(g√(B·worth)), falls from far over the top: it is drawn from
            // where it comes in, with as many points in every frame, so that it is never cut)
            let from = if k == 0 { (1.0 / (c.g * c.g * worth(c.beta) * p.y[1] * p.y[1])).log10().max(p.x[0]).min(p.x[1]) } else { p.x[0] };
            let pts: Vec<(f64, f64)> = (0..LINE_N).map(|i| from + (p.x[1] - from) * i as f64 / (LINE_N - 1) as f64).map(|x| (x, rule(&at(x)).1.min(p.y[1]))).collect();
            dashed_path(f, &p, STEP + 3 + k as u8, &pts, 1.6, color);
        }
        for (k, (color, _, rule)) in rules.into_iter().enumerate() {
            p.curve(f, STEP + k as u8, LINE_N, 2.4, color, |x| rule(&at(x)).0);
        }
        // the batch size now
        let bx = c.b.log10();
        p.vline(f, MARKER, bx, 1.2, pal::WHITE.a(0.7), false);
        // (a dot over the top is kept, unseen and off the picture, so that the batch holds
        // the same dots in every frame)
        let mut dots: Vec<(Vec2, Color)> = Vec::new();
        for (color, _, rule) in rules {
            let (m, sd) = rule(&c);
            for v in [m, sd] {
                let inside = v >= p.y[0] && v <= p.y[1];
                dots.push(if inside { (p.at(bx, v), color) } else { (Vec2::new(-1000.0, -1000.0), Color::CLEAR) });
            }
        }
        dots_px(f, 0, &dots, if lay.narrow { 6.0 } else { 7.0 });
        batch_axis(f, &p, size, "batch size $B$ (drag the white dot)");
        let entries: Vec<(Color, bool, &str)> = rules.iter().map(|(c, name, _)| (*c, false, *name)).collect();
        legend(f, Vec2::new(min.x - LEFT + 4.0, max.y + 24.0 + size), max.x, &entries, size);
        // the numbers at the batch size, on a plate in the top-left corner (where only
        // SGD's jitter would be, far over the top for small batches)
        let mut lines = vec![(format!(r"at $B = {}$: mean step (solid) $\pm$ jitter (dashed)", whole(c.b)), pal::TEXT)];
        for (color, name, rule) in rules {
            let (m, sd) = rule(&c);
            lines.push((format!(r"{name}: ${} \pm {}$", two(m), if sd < 10.0 { two(sd) } else { sig(sd) }), color));
        }
        for (k, (t, col)) in lines.iter().enumerate() {
            f.caption_px(Vec2::new(min.x + 6.0, min.y + 4.0 + k as f32 * (size + 6.0)), t.clone(), size, *col, Align2::LEFT_TOP);
        }
    }

    fn draw_useful(&self, f: &mut Frame, lay: &Layout) {
        let p = Plot::new(lay.useful, [0.0, most_x()], [-0.05, 1.04]);
        if p.tiny() {
            return;
        }
        let size = lay.small;
        p.backing(f, LEFT, 8.0, ABOVE, 8.0);
        let s = &self.setup;
        let head = if lay.narrow { "Layer 2: useful part against the batch size $B$" } else { "Layer 2 of the network: the useful part of each rule's step, against the batch size $B$" };
        title(f, &p, head, lay.text);
        let (min, max) = p.rect;
        plot::line_of(f, AXES, min, Vec2::new(min.x, max.y), 1.0, pal::DIM.a(0.8));
        for y in [0.0, 0.25, 0.5, 0.75, 1.0] {
            let at = p.px_y(y);
            plot::line_of(f, AXES, Vec2::new(min.x - 4.0, at), Vec2::new(min.x, at), 1.0, pal::DIM.a(0.8));
            plot::line_of(f, AXES, Vec2::new(min.x, at), Vec2::new(max.x, at), 1.0, pal::DIM.a(0.1));
            f.text_px(Vec2::new(min.x - 6.0, at), num(y, 2), size, pal::DIM, Align2::RIGHT_CENTER);
        }
        // the axis's ticks (named under the strip)
        for (x, _) in batch_ticks() {
            let at = p.px_x(x);
            plot::line_of(f, AXES, Vec2::new(at, max.y), Vec2::new(at, max.y + 4.0), 1.0, pal::DIM.a(0.8));
        }
        let net = &self.look.net.shown;
        let mask = &self.look.net_mask;
        if net.len() != net_len() || mask.len() != net_len() {
            return;
        }
        let measured_x = |k: usize| (size_of(k) as f64).log10();
        // the formulas: SGD and Adam alone and with momentum, sign descent's normal formula
        let formulas: [(Color, bool, f32); FORMULAS] = [(SGD, false, 2.0), (MOMENTUM, true, 2.0), (ADAM, false, 2.0), (ADAM, true, 2.0), (SIGN.a(0.45), false, 1.2)];
        for (r, (color, dashed, width)) in formulas.into_iter().enumerate() {
            let base = MEASURED * LEVELS + r * LINE_N;
            let pts: Vec<(f64, f64)> = (0..LINE_N).map(|k| if mask[base + k] { (across(k), net[base + k] as f64) } else { (f64::NAN, f64::NAN) }).collect();
            let kind = NET + MEASURED as u8 + r as u8;
            if dashed {
                dashed_path(f, &p, kind, &pts, width, color);
            } else {
                p.path(f, kind, pts, width, color);
            }
        }
        // the measured: sign descent and Muon, alone and with momentum
        let measured: [(Color, bool); MEASURED] = [(SIGN, false), (MUON, false), (SIGN, true), (MUON, true)];
        let mut dots: Vec<(Vec2, Color)> = Vec::new();
        for (r, (color, dashed)) in measured.into_iter().enumerate() {
            // (a measured value a little below zero is drawn on the bottom edge)
            let pts: Vec<(f64, f64)> = (0..LEVELS)
                .map(|k| if mask[r * LEVELS + k] { (measured_x(k), (net[r * LEVELS + k] as f64).max(p.y[0]).min(p.y[1])) } else { (f64::NAN, f64::NAN) })
                .collect();
            if dashed {
                dashed_path(f, &p, NET + r as u8, &pts, 2.0, color);
            } else {
                p.path(f, NET + r as u8, pts.iter().copied(), 2.0, color);
                for &(x, y) in &pts {
                    if x.is_finite() && y >= p.y[0] && y <= p.y[1] {
                        dots.push((p.at(x, y), color));
                    }
                }
            }
        }
        dots_px(f, 1, &dots, if lay.narrow { 4.0 } else { 5.0 });
        // the batch sizes marked
        let one = &self.look.one;
        let bx = (one.get(1) as f64).exp().log10();
        p.vline(f, MARKER, bx, 1.2, pal::WHITE.a(0.7), false);
        if s.compare.is_some() {
            let cx = (one.get(3) as f64).exp().log10();
            plot::dashed_of(f, MARKER, Vec2::new(p.px_x(cx), min.y), Vec2::new(p.px_x(cx), max.y), 1.2, pal::WHITE.a(0.5));
        }
        // the numbers at the marked sizes, on a plate in the top-left corner
        self.useful_table(f, &p, lay);
    }

    /// The table of useful parts at the marked batch sizes: the legend of the chart too.
    fn useful_table(&self, f: &mut Frame, p: &Plot, lay: &Layout) {
        let s = &self.setup;
        let size = lay.small;
        let (min, _) = p.rect;
        let sizes: Vec<usize> = match s.compare {
            Some(c) if c != s.b => {
                let mut v = vec![s.b, c];
                v.sort();
                v
            }
            _ => vec![s.b],
        };
        let rows: [(Color, &str); 4] = [(SGD, "SGD"), (SIGN, if s.beta > 0.0 { "sign descent, Signum" } else { "sign descent" }), (ADAM, "Adam"), (MUON, "Muon")];
        let mut lines: Vec<(String, Color)> = Vec::new();
        let names: Vec<String> = sizes.iter().map(|b| format!(r"$B = {}$", count(*b))).collect();
        let head = if s.beta > 0.0 {
            format!(r"at {}, alone $\to$ with momentum {} (dashed):", names.join(" and "), short(s.beta))
        } else {
            format!("at {}:", names.join(" and "))
        };
        lines.push((head, pal::TEXT));
        for (r, &(c, name)) in rows.iter().enumerate() {
            let mut cells: Vec<String> = Vec::new();
            for &b in &sizes {
                let (alone, with) = self.useful_at(r, b);
                cells.push(if s.beta > 0.0 { format!(r"${} \to {}$", two(alone), two(with)) } else { format!("${}$", two(alone)) });
            }
            lines.push((format!("{name}: {}", cells.join(",  ")), c));
        }
        if !lay.narrow {
            lines.push(("faint purple: the bell's formula for sign descent".to_string(), SIGN.a(0.7)));
        }
        for (k, (t, c)) in lines.iter().enumerate() {
            f.caption_px(Vec2::new(min.x + 6.0, min.y + 4.0 + k as f32 * (size + 6.0)), t.clone(), size, *c, Align2::LEFT_TOP);
        }
    }

    /// Rule r's useful part at batch size `b`, alone and with the present momentum (NaN
    /// where not known).
    fn useful_at(&self, r: usize, b: usize) -> (f64, f64) {
        let Some(l) = &self.layer else { return (f64::NAN, f64::NAN) };
        let k = power_of_two(b as f64).trailing_zeros() as usize;
        let bf = b as f64;
        let w = worth(self.setup.beta);
        let m = self.measured(k);
        match r {
            0 => (l.sgd_useful(bf), l.sgd_useful(bf * w)),
            1 => (m[0], m[2]),
            2 => (l.adam_useful(bf, 1.0).0, l.adam_useful(bf, w).0),
            _ => (m[1], m[3]),
        }
    }

    fn draw_strip(&self, f: &mut Frame, lay: &Layout) {
        let (min, max) = lay.strip;
        if !(max.x - min.x > 40.0 && max.y - min.y > 12.0) {
            return;
        }
        let size = lay.small;
        let p = Plot::new(lay.strip, [0.0, most_x()], [0.0, 1.0]);
        hud_panel(f, min - Vec2::new(LEFT, 20.0), max + Vec2::new(8.0, 20.0));
        let bars = &self.look.strip.shown;
        let n = strip_bins();
        let b = (self.look.one.get(1) as f64).exp();
        if bars.len() == n + 1 {
            let top = bars.iter().copied().fold(1.0f32, f32::max) as f64;
            for (k, &h) in bars.iter().take(n).enumerate() {
                let (x0, x1) = (k as f64 / STRIP_PER_DECADE, ((k + 1) as f64 / STRIP_PER_DECADE).min(most_x()));
                let below = 10f64.powf(x1) <= b * 1.0001;
                p.bar(f, x0, x1, 0.0, h as f64 / top, 0.5, if below { SIGNAL.a(0.75) } else { NOISE.a(0.7) });
            }
        }
        plot::line_of(f, AXES, Vec2::new(min.x, max.y), max, 1.0, pal::DIM.a(0.8));
        for (x, label) in batch_ticks() {
            let at = p.px_x(x);
            plot::line_of(f, AXES, Vec2::new(at, max.y), Vec2::new(at, max.y + 4.0), 1.0, pal::DIM.a(0.8));
            f.text_px(Vec2::new(at, max.y + 5.0), label, size, pal::DIM, Align2::CENTER_TOP);
        }
        let name = if lay.narrow {
            r"coordinates by own noise scale $B_i$; white: $B_i < B$"
        } else {
            r"layer 2's 256 coordinates by their own noise scale $B_i = \sigma_i^2/G_i^2$ (white: below the batch size)"
        };
        f.text_px(Vec2::new(min.x - LEFT + 4.0, min.y - 3.0), name, size, pal::DIM, Align2::LEFT_BOTTOM);
        // the coordinates beyond the largest batch: counted at the end of that line
        let over = bars.get(n).map_or(0, |v| v.round() as usize);
        if over > 0 {
            let text = format!("and {over} above {}", count(MOST_B));
            let room = max.x - (min.x - LEFT + 4.0) - f.text_size(name, size).x;
            if f.text_size(&text, size).x + 12.0 < room {
                f.text_px(Vec2::new(max.x, min.y - 3.0), text, size, NOISE, Align2::RIGHT_BOTTOM);
            }
        }
        // the noise scales of SGD and of sign descent, named on either side of their marks
        if let Some(l) = self.current() {
            let marks = [(l.simple, SGD, r"$B_{\text{simple}}$"), (l.l1, SIGN, r"$B_{\ell_1}$")];
            let first_left = marks[0].0 <= marks[1].0;
            for (k, (v, c, name)) in marks.into_iter().enumerate() {
                let x = v.log10();
                if x > p.x[0] && x < p.x[1] {
                    let at = p.px_x(x);
                    plot::line_of(f, MARKS, Vec2::new(at, min.y - 2.0), Vec2::new(at, max.y), 2.0, c);
                    let left = (k == 0) == first_left;
                    let (dx, align) = if left { (-3.0, Align2::RIGHT_TOP) } else { (3.0, Align2::LEFT_TOP) };
                    f.caption_px(Vec2::new(at + dx, min.y + 1.0), name, size, c, align);
                }
            }
        }
        p.vline(f, MARKER, b.log10(), 1.2, pal::WHITE.a(0.7), false);
    }

    fn draw_spectrum(&self, f: &mut Frame, lay: &Layout) {
        let v = &self.look.spectrum.shown;
        if v.len() != 3 * SIDE + 2 {
            return;
        }
        let (lo, hi) = (v[3 * SIDE] as f64, v[3 * SIDE + 1] as f64);
        let p = Plot::new(lay.spectrum, [0.5, SIDE as f64 + 0.5], [lo, hi.max(lo + 1.0)]);
        if p.tiny() {
            return;
        }
        let size = lay.small;
        p.backing(f, LEFT, BELOW, ABOVE, 8.0);
        let b = self.setup.b;
        title(f, &p, if lay.narrow { "Layer 2's gradient: its singular values" } else { "Layer 2's gradient: its 16 singular values, largest first (logarithmic)" }, lay.text);
        let (min, max) = p.rect;
        plot::line_of(f, AXES, min, Vec2::new(min.x, max.y), 1.0, pal::DIM.a(0.8));
        plot::line_of(f, AXES, Vec2::new(min.x, max.y), max, 1.0, pal::DIM.a(0.8));
        let mut d = p.y[0].ceil() as i32;
        while (d as f64) <= p.y[1] + 1e-9 {
            let at = p.px_y(d as f64);
            plot::line_of(f, AXES, Vec2::new(min.x - 4.0, at), Vec2::new(min.x, at), 1.0, pal::DIM.a(0.8));
            plot::line_of(f, AXES, Vec2::new(min.x, at), Vec2::new(max.x, at), 1.0, pal::DIM.a(0.08));
            if (p.y[1] - p.y[0]) < 9.0 || d % 2 == 0 {
                f.text_px(Vec2::new(min.x - 6.0, at), format!("$10^{{{d}}}$"), size, pal::DIM, Align2::RIGHT_CENTER);
            }
            d += 1;
        }
        for i in [1usize, 4, 8, 12, 16] {
            let at = p.px_x(i as f64);
            f.text_px(Vec2::new(at, max.y + 5.0), i.to_string(), size, pal::DIM, Align2::CENTER_TOP);
        }
        f.text_px(Vec2::new(max.x, max.y + 7.0 + size), "which singular value", size, pal::DIM, Align2::RIGHT_TOP);
        let sets: [(Color, f32); 3] = [(SIGNAL, 2.2), (SGD, 1.8), (NOISE, 1.8)];
        let mut dots = Vec::new();
        for (j, (c, width)) in sets.into_iter().enumerate() {
            let pts: Vec<(f64, f64)> =
                (0..SIDE).map(|i| v[j * SIDE + i]).enumerate().map(|(i, y)| if y > NOWHERE + 1.0 { ((i + 1) as f64, y as f64) } else { (f64::NAN, f64::NAN) }).collect();
            p.path(f, SPECTRUM + j as u8, pts.iter().copied(), width, c);
            for &(x, y) in &pts {
                if x.is_finite() {
                    dots.push((p.at(x, y), c));
                }
            }
        }
        dots_px(f, 2, &dots, if lay.narrow { 5.0 } else { 6.0 });
        let names: [(Color, bool, &str); 3] = [(SIGNAL, false, "the true gradient"), (SGD, false, "a batch's gradient"), (NOISE, false, "its noise alone")];
        legend(f, Vec2::new(min.x - LEFT + 4.0, max.y + 22.0 + size), max.x, &names, size);
        // how many of the true gradient's directions stand above the noise
        if let (Some(l), Some(pl)) = (self.current(), &self.plain) {
            if pl.trials >= SPECTRUM_TRIALS.min(SAMPLED) {
                let k = power_of_two(b as f64).trailing_zeros() as usize;
                let (_, noise) = pl.spectra(k);
                let above = l.sv.iter().filter(|&&x| x > noise[0]).count();
                let text = format!(r"at $B = {}$: {} of the true gradient's 16 stand above the noise's largest", count(b), above);
                f.caption_px(Vec2::new(max.x - 6.0, min.y + 4.0), text, size, pal::TEXT, Align2::RIGHT_TOP);
            }
        }
    }
}

/// The rules of the one coordinate's chart, with momentum β or without: colour, name,
/// and its step's mean and spread.
type StepRule = (Color, &'static str, fn(&Coordinate) -> (f64, f64));
fn step_rules(beta: f64) -> [StepRule; 3] {
    if beta > 0.0 {
        [(MOMENTUM, "momentum", Coordinate::sgd), (SIGN, "Signum", Coordinate::sign), (ADAM, "Adam", Coordinate::adam)]
    } else {
        [(SGD, "SGD", Coordinate::sgd), (SIGN, "sign descent", Coordinate::sign), (ADAM, r"Adam, $\beta_1 = 0$", Coordinate::adam)]
    }
}

/// A setting as a label writes it: as few digits as it needs.
fn short(v: f64) -> String {
    let text = sig(v);
    if text.contains('.') {
        text.trim_end_matches('0').trim_end_matches('.').to_string()
    } else {
        text
    }
}

impl Scene for SignNoise {
    fn view(&self) -> View {
        View::Flat { center: Vec2::ZERO, half_height: 1.0 }
    }

    fn settings(&self, s: &mut ViewSettings) {
        s.grid = false;
        s.bloom = s.bloom.min(0.2);
    }

    fn animated(&self) -> bool {
        self.busy() || self.look.moving || self.dragging
    }

    fn pointer_hint(&self) -> Option<&'static str> {
        Some("white dots on the charts: drag the batch size    buttons at the top: click")
    }

    fn ui(&mut self, ui: &mut egui::Ui, _: &Gfx) {
        self.panel(ui);
    }

    fn frame(&mut self, f: &mut Frame) {
        hold_batches(f);
        let lay = Layout::of(f.view_size(), f.top_inset, self.setup.page);
        let tiny = lay.tiny();
        self.dragging = false;
        if !tiny {
            self.buttons(f, &lay);
            self.drag(f, &lay);
        }
        self.setup = self.setup.kept();
        let budget = clock::frame_budget(f.real_dt, 0.006);
        self.work(&budget);
        let snap = self.canvas != Some(lay.view) && !tiny;
        if snap {
            self.canvas = Some(lay.view);
        }
        self.follow(f.real_dt, snap);
        if tiny {
            return;
        }
        self.status(f, &lay);
        if lay.one {
            self.draw_bell(f, &lay);
            self.draw_steps(f, &lay);
        }
        if lay.net {
            self.draw_useful(f, &lay);
            self.draw_strip(f, &lay);
            self.draw_spectrum(f, &lay);
        }
    }
}

impl SignNoise {
    fn panel(&mut self, ui: &mut egui::Ui) {
        ui::note(
            ui,
            r"Sign descent, Adam and Muon keep a direction and throw away the size of the gradient: sign descent keeps each coordinate's sign, Muon the singular vectors of a weight matrix's gradient (of its momentum), with every singular value pushed towards 1. Why that can help with the whole data's gradient is for the scenes adam and muon. A mini-batch's gradient is the true gradient plus noise (scene minibatch), and a step that throws away sizes gives the noise's part as full a size as the signal's. How much of such a step still points the right way, and what do a larger batch, or an average over steps, buy? First one coordinate whose noise is a bell; then one layer of a small network, measured.",
        );

        ui::section(ui, "Things to try");
        for (k, t) in TRIALS.iter().enumerate() {
            if ui::experiment(ui, t.button, t.caption) {
                self.set_up(k);
            }
        }

        ui::section(ui, "One coordinate");
        let s = &mut self.setup;
        let mut which = s.pick.is_none() as usize;
        if ui::choice(ui, &["a coordinate of layer 2", "my own"], &mut which) {
            s.pick = if which == 0 { Some(COIN_RANK) } else { None };
        }
        match s.pick {
            Some(rank) => {
                let mut r = rank + 1;
                if ui::slider_int(ui, "its rank by own noise scale", &mut r, 1..=COORDS) {
                    s.pick = Some(r.clamp(1, COORDS) - 1);
                }
                ui::note(ui, "1 is the least noisy of the layer's 256 coordinates, 256 the noisiest.");
            }
            None => {
                let mut g = s.g as f32;
                if ui::slider_log(ui, r"true gradient $g$, in units of $\sigma$", &mut g, LEAST_G as f32..=MOST_G as f32) {
                    s.g = g as f64;
                }
            }
        }
        ui::note(
            ui,
            r"A batch's gradient of one coordinate averages $B$ examples' gradients, which have mean $g$ (the true gradient) and standard deviation $\sigma$ over the data. So it has mean $g$ and spread $\sigma/\sqrt{B}$, and is near a bell (a normal distribution) once $B$ is not small: an approximation, which the measured outline tests. Its sign is then right with probability",
        );
        ui::equation(ui, r"\Phi\!\left(\frac{|g|\sqrt{B}}{\sigma}\right), \quad \mathbb{E}\,\operatorname{sign} = \operatorname{erf}\!\left(\frac{g\sqrt{B}}{\sqrt{2}\,\sigma}\right) \approx 0.80\,\frac{g\sqrt{B}}{\sigma}");
        ui::note(
            ui,
            r"($\Phi$ the share of a standard bell below a point; the approximation for $g\sqrt{B} \ll \sigma$). The coordinate's own noise scale $B_i = \sigma^2/g^2$ is the batch at which $g\sqrt{B}/\sigma = 1$. Well below it, sign descent's step has full size and a nearly random sign: its mean grows only as $\sqrt{B}$ and its jitter stays near 1. SGD's mean step is right at any batch, and its jitter shrinks as $1/\sqrt{B}$. Balles and Hennig (2018) give the same formula for the sign.",
        );

        ui::section(ui, "Batch size and momentum");
        let mut b = s.b;
        if ui::slider_int_log(ui, "batch size $B$", &mut b, 1..=MOST_B) {
            s.b = power_of_two(b as f64);
        }
        ui::note(ui, "Powers of two, as the network is measured at. The white dots on the charts drag it too.");
        if s.compare.is_some() && ui::buttons(ui, &["remove the second batch size"]).is_some() {
            s.compare = None;
        }
        let mut beta = s.beta as f32;
        if ui::slider(ui, r"momentum $\beta$", &mut beta, 0.0..=MOST_BETA as f32) {
            s.beta = (beta as f64 * 100.0).round() / 100.0;
        }
        ui::note(ui, r"The step is made from $m \leftarrow \beta m + (1 - \beta)\,g_B$, as Signum, Adam's $\hat m$ and Muon make theirs. With independent batches at fixed weights, its variance is a fraction of one batch's:");
        ui::equation(ui, r"\operatorname{Var} m = (1-\beta)^2 \sum_{k \ge 0} \beta^{2k}\,\frac{\sigma^2}{B} = \frac{1-\beta}{1+\beta}\cdot\frac{\sigma^2}{B}");
        ui::note(
            ui,
            r"so it is worth a batch $(1 + \beta)/(1 - \beta)$ times larger: 19 times at $\beta = 0.9$, 199 at 0.99. That holds at fixed weights. In training the weights move while it averages, and its gradients are on average $\beta/(1 - \beta)$ steps old.",
        );

        ui::section(ui, "The network");
        ui.horizontal_wrapped(|ui| {
            if ui.button(ui::label_text(ui, &format!("train {TRAIN_MORE} more steps"), 12.5, None, false)).clicked() {
                self.setup.trained = (self.setup.trained + TRAIN_MORE).min(MOST_TRAINED);
            }
            if ui.button(ui::label_text(ui, "back to the start", 12.5, None, false)).clicked() {
                self.setup.trained = 0;
            }
        });
        ui::note(
            ui,
            r"A tanh network ($2 \to 16 \to 16 \to 16 \to 3$) classifying 1024 points on three spiral arms, trained when asked by Adam (step 0.01, batches of 32). At its present weights every example's gradient is computed, and the rules are compared on layer 2's weights, a $16 \times 16$ matrix of the kind Muon acts on. Batches are drawn with replacement, as from an endless supply of such examples: 200 of each size from 1 to 4096. With momentum, a stream of independent batches at the same weights, 200 averages recorded $2/(1 - \beta)$ steps apart, for the smaller sizes only (up to 256 at $\beta = 0.9$).",
        );
        ui::note(ui, r"The useful part of a step: with $d$ the step a rule takes from a mini-batch, and $d^*$ the step it takes from the whole data,");
        ui::equation(ui, r"u = \frac{\mathbb{E}\,\langle d, d^*\rangle}{|d^*|\,\sqrt{\mathbb{E}\,|d|^2}}");
        ui::note(
            ui,
            r"the mean length of $d$ along $d^*$, over $d$'s typical length. Of $d$'s mean square, a share $u^2$ is the square of its mean progress along $d^*$, and $1 - u^2$ is the rest: jitter, and any steady part across $d^*$. For SGD it is exactly $1/\sqrt{1 + B_{\text{simple}}/B}$, with $B_{\text{simple}}$ the noise scale of scene minibatch, and is drawn from that. Sign descent's and Muon's are measured (the dots; sign descent's at a batch of one exactly), and wobble by up to about 0.03 (their standard errors are under Numbers); the faint purple line is sign descent's from the bell's formula, coordinate by coordinate.",
        );
        ui::note(
            ui,
            r"Measured here, at the start: in 73 of the layer's 256 coordinates fewer than half of the examples have the true gradient's sign (a minority of larger gradients outweighs a majority of the other sign). A batch of one is then wrong more often than right, and sign descent's useful part at a batch of one is 0.07 where the bell's formula gives 0.12. After 3000 steps of training the formula is above the measured line at every batch up to 1024.",
        );
        ui::note(
            ui,
            r"Adam here divides by $\sqrt{v}$ with $v$ at its long-run mean $G^2 + \sigma^2/B$ (it averages the squares of single batches over about 1000 steps; $\varepsilon$ is left out), so its curve is a formula too; with the whole data its step is the sign. Muon here is its reference's iteration (five rounds of its polynomial) on the gradient or on the average, without Nesterov's variant or the shape factor.",
        );

        ui::section(ui, "What has been published");
        ui::note(
            ui,
            r"Kunstner and co-authors (2023) tuned the step for each batch size on five tasks, from a small convolutional network on MNIST to Transformer-XL on WikiText-2, and measured the training loss: sign descent did poorly with small batches and much better with large ones, and with the full batch it closed most of the gap between SGD and Adam.",
        );
        ui::note(
            ui,
            r"The convergence theorem for signSGD of Bernstein and co-authors (2018) takes a batch that grows with the number of steps; with a small fixed batch, Karimireddy and co-authors (2019) gave convex problems on which signSGD does not reach the minimum. Bernstein and co-authors also point out that when the noise is very skewed or has two humps, the chance of the right sign can be arbitrarily small.",
        );
        ui::note(
            ui,
            r"For Adam and RMSProp, Malladi and co-authors (2022) proved that when the noise is much larger than the mean gradient (with further conditions on the noise, and for Adam after a warm-start phase), a batch $\kappa$ times larger keeps training about the same per epoch if the step is multiplied by $\sqrt{\kappa}$, $1 - \beta_1$ and $1 - \beta_2$ by $\kappa$, and $\varepsilon$ divided by $\sqrt{\kappa}$. For SGD the corresponding rule is linear, within limits: Goyal and co-authors (2017) multiplied the step by the batch's factor, with a warm-up, and kept ResNet-50's accuracy on ImageNet with momentum SGD up to batches of about 8000, beyond which accuracy fell. Measured best steps for Adam have not always followed the square root: McCandlish and co-authors (2018) found them growing as $B$ to a power between 0.5 and 1 and then levelling off, and Marek and co-authors (2025), going from batch 1 to 1024 with language models of 30 million parameters, found that a step about 3 times larger worked better than the 32 times the square root gives.",
        );
        ui::note(
            ui,
            r"For Muon, Essential AI (2025) trained language models of 100 million to 4 billion parameters with AdamW and with Muon: for a 1-billion-parameter model, the tokens AdamW needed to reach a loss, over those Muon needed, stayed above 1 and did not fall as the batch grew. Sato and co-authors (2025) derived a bound in which a larger momentum gives Muon a smaller batch that uses the fewest gradients; they say it shows how the settings govern that batch, not its value. At small batches the evidence is about a different rule: Tuddenham and co-authors (2022), who orthogonalised each gradient before averaging (Muon averages first), found it worse than SGD with momentum at very small batches and starting to do better at a batch of 16, for a ResNet-18 on CIFAR-10.",
        );
        ui::note(
            ui,
            r"Naganuma and co-authors (2026) defined noise scales for sign descent and for spectral descent (Muon without momentum) in their own norms: $B_{\ell_1} = \|\sigma\|_1^2/\|G\|_1^2$, with $\sigma$ the coordinates' standard deviations over the examples, and $B_{S_1} = \|C_{\text{row}}^{1/2}\|_{S_1}^2/\|G\|_{S_1}^2$ with nuclear norms (sums of singular values) and $C_{\text{row}}$ a covariance of the noise. From a lower bound on one step's progress, they find the bound reaching a share $\kappa$ of its whole-data value at a batch of $\kappa/(1 - \kappa)$ times their $\ell_2$ noise scale (the simple one, $B_{\text{simple}}$) for SGD, and of $(1 - \sqrt{\kappa})^{-2} B_{\ell_1}$ for sign descent: for half, $B_{\text{simple}}$ and $11.7\,B_{\ell_1}$. They caution that the constants are loose. Their values for this network are under Numbers, read with $C_{\text{row}}$ as $\mathbb{E}\,DD^\top$, the covariance between the rows of one example's deviation $D = g_n - G$, or as $\mathbb{E}\,D^\top D$, between its columns.",
        );

        ui::section(ui, "Reading the picture");
        ui::key(
            ui,
            &[
                (SIGNAL, r"white: the true gradient ($g$, or $G$ for the layer); the right-sign side of the bell; coordinates less noisy than the batch"),
                (NOISE, "red: the wrong-sign side of the bell; coordinates noisier than the batch; the noise's singular values"),
                (pal::DIM.mix(pal::TEXT, 0.4), "grey outline: the chosen coordinate's sampled batch gradients"),
                (SGD, "cyan: SGD, whose step is the batch's gradient"),
                (MOMENTUM, "pink: SGD with momentum"),
                (SIGN, "purple: sign descent; with momentum, Signum"),
                (ADAM, "orange: Adam"),
                (MUON, "blue: Muon"),
                (pal::WHITE.a(0.7), "white line and dot: the batch size (drag it); dashed white: a second one"),
            ],
        );
        ui::section(ui, "Words");
        ui::term(ui, "True gradient:", r"the gradient of the loss on all the data: the mean of every example's gradient.");
        ui::term(ui, r"Mini-batch, batch size $B$:", "the examples whose gradients a step averages. Its gradient's noise is what it differs from the true gradient by.");
        ui::term(ui, "Sign descent, Signum:", "every coordinate moves by the step size, against the sign of its gradient; Signum takes the sign of a running average (momentum) instead.");
        ui::term(ui, "Adam:", r"steps by $\eta\,\hat m/(\sqrt{\hat v} + \varepsilon)$, running averages of the gradient and of its square (scene adam).");
        ui::term(ui, "Muon:", "a weight matrix steps along its momentum with the singular values pushed towards 1 (scene muon).");
        ui::term(ui, "Singular values:", "how much a matrix stretches along each of its own directions, largest first (scene svd).");
        ui::term(ui, r"Own noise scale $B_i$:", r"a coordinate's $\sigma_i^2/G_i^2$: the batch at which its noise and its true gradient are the same size.");
        ui::term(ui, r"Noise scale $B_{\text{simple}}$:", r"the trace of the gradients' covariance over $|G|^2$: the batch at which half a batch gradient's mean square is noise (scene minibatch).");
        ui::term(ui, "Jitter:", "the standard deviation of a step about its mean.");
        ui::term(ui, "Useful part:", "defined above, under The network.");

        ui::section(ui, "Numbers");
        self.numbers(ui);
        ui::see_also(
            ui,
            &[
                ("minibatch", "a mini-batch gradient: signal and noise"),
                ("adam", "Adam, taken apart"),
                ("muon", "Muon: an orthogonal step"),
                ("noisescale", "how large a batch is worth it"),
            ],
        );
    }

    fn numbers(&self, ui: &mut egui::Ui) {
        let s = &self.setup;
        let c = self.coordinate();
        match self.picked() {
            Some((i, j, _)) => ui::readout(ui, "the coordinate", format!("entry $({i}, {j})$ of layer 2")),
            None => ui::readout(ui, "the coordinate", "my own"),
        }
        ui::readout(ui, r"$g/\sigma$", sig(c.g));
        ui::readout(ui, r"own noise scale $B_i = \sigma^2/g^2$", scale_text(1.0 / (c.g * c.g)));
        ui::readout(ui, r"$g\sqrt{B}/\sigma$", sig(c.ratio()));
        if s.beta > 0.0 {
            ui::readout(ui, "the average is worth a batch of", whole(c.b * worth(s.beta)));
        }
        ui::readout_colored(ui, r"right sign: the bell's $\Phi$", num(c.right(), 3), SIGNAL);
        if let Some(sample) = self.sample_now() {
            ui::readout(ui, &format!("right sign: measured, {} draws", count(sample.values.len())), num(sample.share_right(), 3));
        }
        if let (Some((_, _, i)), Some(l)) = (self.picked(), self.current()) {
            ui::readout(ui, "right sign with one example (exact)", num(l.right1[i], 3));
        }
        for (color, name, rule) in step_rules(s.beta) {
            let (m, sd) = rule(&c);
            ui::readout_colored(ui, &format!("{name}: mean step $\\pm$ jitter"), format!(r"${} \pm {}$", two(m), if sd < 10.0 { two(sd) } else { sig(sd) }), color);
        }
        let Some(l) = self.current() else {
            ui::readout(ui, "layer 2", "being measured");
            return;
        };
        ui::readout(ui, "training steps", count(l.step));
        ui::readout(ui, "loss on all the data", sig(l.loss));
        ui::readout(ui, r"layer 2's $|G|$", format!("${}$", sig(l.norm())));
        ui::readout_colored(ui, r"$B_{\text{simple}}$: layer 2, all 643 weights", format!("{}, {}", whole(l.simple), whole(l.whole_simple)), SGD);
        ui::readout_colored(ui, r"$B_{\ell_1}$: layer 2, all 643 weights", format!("{}, {}", whole(l.l1), whole(l.whole_l1)), SIGN);
        ui::readout_colored(ui, r"$B_{S_1}$ of layer 2: rows, columns", format!("{}, {}", whole(l.nuclear[0]), whole(l.nuclear[1])), MUON);
        ui::readout(ui, "median own noise scale", whole(l.median_scale()));
        ui::readout(ui, &format!("coordinates with $B_i < {}$", count(s.b)), format!("{} of 256", l.below(s.b as f64)));
        ui::readout(ui, "wrong more often than right with one example", format!("{} of 256", l.right1.iter().filter(|&&r| r < 0.5).count()));
        let rows = [(SGD, "SGD"), (SIGN, "sign descent"), (ADAM, "Adam"), (MUON, "Muon")];
        let k = s.b.trailing_zeros() as usize;
        for (r, (color, name)) in rows.into_iter().enumerate() {
            let (alone, with) = self.useful_at(r, s.b);
            // (the sampled ones with their standard errors)
            let error = |t: Option<&Tally>| t.map(|t| t.error()).filter(|e| e.is_finite()).map_or(String::new(), |e| format!(r" $\pm$ {}", two(e)));
            let (ea, ew) = match r {
                1 => (error(self.plain.as_ref().map(|p| &p.sign[k])), error(self.stream.as_ref().filter(|m| k < m.levels).map(|m| &m.sign[k]))),
                3 => (error(self.plain.as_ref().map(|p| &p.muon[k])), error(self.stream.as_ref().filter(|m| k < m.levels).map(|m| &m.muon[k]))),
                _ => (String::new(), String::new()),
            };
            let text = if s.beta > 0.0 { format!(r"{}{ea}, with momentum {}{ew}", two(alone), two(with)) } else { format!("{}{ea}", two(alone)) };
            ui::readout_colored(ui, &format!("useful part at $B = {}$: {name}", count(s.b)), text, color);
        }
        let (_, rms) = l.adam_useful(s.b as f64, worth(s.beta));
        ui::readout_colored(ui, "Adam's r.m.s. step, in step sizes", two(rms), ADAM);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The layer measured at `steps` steps of training.
    fn layer_at(steps: usize) -> Layer {
        let mut t = Trainer::new();
        t.work(steps, &Budget::new(1e9));
        let mut per = PerExample::new(&t.problem, &t.model);
        per.advance(&t.problem, &t.model, &mut t.tape, usize::MAX);
        let mut layer = Layer::of(&per, &t.model, &t.problem, &mut t.tape, steps);
        layer.nuclear = Covariances::new(steps).advance(&layer, usize::MAX).unwrap();
        layer
    }

    fn plain(layer: &Layer) -> Plain {
        let mut p = Plain::new(SAMPLE_SEED);
        while !p.done() {
            p.trial(layer);
        }
        p
    }

    fn stream(layer: &Layer, beta: f64) -> Stream {
        let mut s = Stream::new(beta, SAMPLE_SEED + 1);
        while !s.done() {
            s.step(layer);
        }
        s
    }

    /// An experiment as its button sets it up, with all its work done.
    fn run(k: usize) -> SignNoise {
        let mut s = SignNoise::create();
        s.set_up(k);
        let budget = Budget::new(1e9);
        let mut rounds = 0;
        while s.busy() {
            s.work(&budget);
            rounds += 1;
            assert!(rounds < 100, "experiment {k} never finishes");
        }
        s
    }

    fn close(a: f64, b: f64, tol: f64) -> bool {
        (a - b).abs() <= tol
    }

    /// A running average of independent draws has (1 − β)/(1 + β) of their variance once
    /// its start is forgotten: the derivation of `worth`, by simulation.
    #[test]
    fn an_average_is_worth_a_larger_batch() {
        for beta in [0.5, 0.9, 0.99] {
            let mut rng = Rng::new(3);
            let (mut m, mut sum, mut sq, mut n) = (0.0f64, 0.0, 0.0, 0usize);
            for t in 0..400_000 {
                m = beta * m + (1.0 - beta) * rng.normal() as f64;
                if t > 2000 {
                    sum += m;
                    sq += m * m;
                    n += 1;
                }
            }
            let var = sq / n as f64 - (sum / n as f64).powi(2);
            let want = (1.0 - beta) / (1.0 + beta);
            assert!((var / want - 1.0).abs() < 0.06, "{beta}: {var} against {want}");
            assert!(close(worth(beta), 1.0 / want, 1e-12));
        }
        assert!(close(worth(0.9), 19.0, 1e-9) && close(worth(0.99), 199.0, 1e-9) && worth(0.0) == 1.0);
        // the control's ends and nonsense stay finite
        assert!(worth(f64::NAN) == 1.0 && worth(5.0).is_finite() && worth(-1.0) == 1.0);
        assert!(close(beta_of(worth(0.9)), 0.9, 1e-12));
    }

    /// The one coordinate's formulas: the expected sign against sampling of a normal
    /// variable, its small-ratio slope √(2/π) ≈ 0.80, and Adam's mean and spread against
    /// the rule itself run on a stream (Kingma and Ba's form, β₂ = 0.999, ε = 1e-8).
    #[test]
    fn one_coordinate_formulas_hold() {
        let mut rng = Rng::new(11);
        for r in [0.05, 0.3, 1.0, 2.5] {
            let n = 200_000;
            let right = (0..n).filter(|_| r + rng.normal() as f64 > 0.0).count() as f64 / n as f64;
            assert!(close(right, normal_cdf(r), 0.004), "{r}: {right}");
            assert!(close(2.0 * right - 1.0, expected_sign(r), 0.008));
        }
        assert!(close(expected_sign(0.01) / 0.01, (2.0 / std::f64::consts::PI).sqrt(), 1e-4));
        assert!(close((2.0 / std::f64::consts::PI).sqrt(), 0.80, 0.003));
        // sign: mean² + spread² = 1; SGD: spread one batch's noise over g
        let c = Coordinate { g: 0.1, b: 16.0, beta: 0.0 };
        let (m, sd) = c.sign();
        assert!(close(m * m + sd * sd, 1.0, 1e-12) && close(c.sgd().1, 1.0 / (0.1 * 4.0), 1e-12));
        // Adam on a stream g + s z, s = 1/√B: noise far above the mean, and far below
        for (g, b, beta) in [(0.1, 4.0, 0.9), (0.1, 4.0, 0.0), (3.0, 1.0, 0.9)] {
            let c = Coordinate { g, b, beta };
            let s = 1.0 / f64::sqrt(b);
            let (mut m1, mut v, mut sum, mut sq, mut n) = (0.0f64, 0.0f64, 0.0, 0.0, 0usize);
            for t in 1..=300_000i32 {
                let x = g + s * rng.normal() as f64;
                m1 = beta * m1 + (1.0 - beta) * x;
                v = 0.999 * v + 0.001 * x * x;
                let u = (m1 / (1.0 - beta.powi(t.min(10_000)))) / ((v / (1.0 - 0.999f64.powi(t.min(100_000)))).sqrt() + 1e-8);
                if t > 20_000 {
                    sum += u;
                    sq += u * u;
                    n += 1;
                }
            }
            let mean = sum / n as f64;
            let spread = (sq / n as f64 - mean * mean).sqrt();
            let (fm, fs) = c.adam();
            assert!(close(mean, fm, 0.02 + 0.02 * fm.abs()), "{g} {b} {beta}: mean {mean} against {fm}");
            assert!(close(spread, fs, 0.03 * fs + 0.005), "{g} {b} {beta}: spread {spread} against {fs}");
        }
    }

    /// The useful part's definition: 1 for steps along d*, −1 against it, and E|d|² = a² +
    /// E|d − a d̂*|² (u² the share of the square of the mean progress).
    #[test]
    fn the_useful_part_is_what_its_definition_says() {
        let mut t = Tally::default();
        for _ in 0..5 {
            t.push(2.0, 4.0);
        }
        assert!(close(t.useful(), 1.0, 1e-12));
        // steps along d* of unequal lengths: less than 1, their mean over their r.m.s.
        let mut t = Tally::default();
        for k in 1..5 {
            t.push(k as f64, (k * k) as f64);
        }
        assert!(close(t.useful(), 2.5 / 7.5f64.sqrt(), 1e-12));
        let mut t = Tally::default();
        t.push(-2.0, 4.0);
        assert!(close(t.useful(), -1.0, 1e-12));
        // random steps in the plane about a fixed mean, d* along x
        let mut rng = Rng::new(4);
        let (mut t, mut dev) = (Tally::default(), 0.0);
        let steps: Vec<[f64; 2]> = (0..20_000).map(|_| [0.7 + rng.normal() as f64, 0.3 + 0.5 * rng.normal() as f64]).collect();
        for d in &steps {
            t.push(d[0], d[0] * d[0] + d[1] * d[1]);
        }
        let a = t.along / t.n as f64;
        for d in &steps {
            dev += (d[0] - a).powi(2) + d[1] * d[1];
        }
        let square = t.square / t.n as f64;
        assert!(close(square, a * a + dev / t.n as f64, 1e-9));
        assert!(close(t.useful() * t.useful(), a * a / square, 1e-12));
        assert!(t.error() > 0.0 && t.error() < 0.01);
        // nothing pushed: not a number, and never a panic
        assert!(Tally::default().useful().is_nan() && Tally::default().error().is_nan());
    }

    /// SGD's useful part is exactly 1/√(1 + B_simple/B); Adam's with v at its mean is its
    /// formula: both against sampled batches on the layer (four standard errors).
    #[test]
    fn sgd_and_adam_formulas_hold_by_sampling() {
        let layer = layer_at(0);
        let p = plain(&layer);
        let mut rng = Rng::new(17);
        let mut adam = [Tally::default(); LEVELS];
        let mut gb = vec![0.0; COORDS];
        for k in 0..LEVELS {
            let b = size_of(k);
            let want = layer.sgd_useful(b as f64);
            let (got, se) = (p.sgd[k].useful(), p.sgd[k].error());
            assert!((got - want).abs() < 4.0 * se + 0.005, "B {b}: SGD {got} ± {se} against {want}");
            if b <= 64 {
                for _ in 0..400 {
                    gb.fill(0.0);
                    for _ in 0..b {
                        let e = rng.below(EXAMPLES);
                        for (o, r) in gb.iter_mut().zip(&layer.rows[e * COORDS..(e + 1) * COORDS]) {
                            *o += r / b as f64;
                        }
                    }
                    let u: Vec<f64> = (0..COORDS).map(|i| gb[i] / (layer.g[i] * layer.g[i] + layer.var[i] / b as f64).sqrt()).collect();
                    adam[k].push(dot(&u, &layer.sign_g) / layer.sign_norm, dot(&u, &u));
                }
                let want = layer.adam_useful(b as f64, 1.0).0;
                let (got, se) = (adam[k].useful(), adam[k].error());
                assert!((got - want).abs() < 4.0 * se + 0.01, "B {b}: Adam {got} ± {se} against {want}");
            }
        }
        // without momentum Adam's step has mean square 1 per coordinate, exactly
        assert!(close(layer.adam_useful(16.0, 1.0).1, 1.0, 1e-12));
    }

    /// At fixed weights an average with momentum β is worth a batch (1 + β)/(1 − β) times
    /// larger for sign descent and Muon too: the measured stream against the bell's
    /// formula at B·19 (sign) and against plain batches near B·19 (Muon).
    #[test]
    fn momentum_is_a_larger_batch_on_the_layer() {
        let layer = layer_at(0);
        let p = plain(&layer);
        let m = stream(&layer, 0.9);
        assert_eq!(m.levels, 9);
        for k in 0..m.levels {
            let b = size_of(k) as f64;
            let sign = m.sign[k].useful();
            let want = layer.sign_normal(b * worth(0.9));
            assert!((sign - want).abs() < 4.0 * m.sign[k].error() + 0.01, "B {b}: Signum {sign} against {want}");
            // Muon at B with momentum lies between plain Muon at the two sizes about 19 B
            if k + 5 < LEVELS {
                let (lo, hi) = (p.muon[k + 4].useful(), p.muon[k + 5].useful());
                let muon = m.muon[k].useful();
                assert!(muon > lo - 0.04 && muon < hi + 0.04, "B {b}: Muon {muon} not between {lo} and {hi}");
            }
        }
    }

    /// Where examples' gradients are skewed, the bell's formula is wrong for small
    /// batches: on this layer at its start, sign descent's measured useful part at a batch
    /// of one is well below the formula's, and agrees from 16 up.
    #[test]
    fn the_bell_is_wrong_for_small_batches() {
        let layer = layer_at(0);
        let p = plain(&layer);
        // (at a batch of one the useful part is known exactly: each coordinate's sign is
        // right for the share right1 of the examples)
        let exact = layer.sign_one();
        println!("exact {exact} sampled {} ± {} normal {}", p.sign[0].useful(), p.sign[0].error(), layer.sign_normal(1.0));
        assert!((p.sign[0].useful() - exact).abs() < 4.0 * p.sign[0].error());
        assert!(exact < layer.sign_normal(1.0) - 0.02, "{exact} {}", layer.sign_normal(1.0));
        for k in 4..LEVELS {
            let b = size_of(k) as f64;
            assert!((p.sign[k].useful() - layer.sign_normal(b)).abs() < 4.0 * p.sign[k].error() + 0.015, "B {b}");
        }
        assert_eq!(layer.right1.iter().filter(|&&r| r < 0.5).count(), 73);
        // what the panel says of it
        let note = "in 73 of the layer's 256 coordinates fewer than half";
        let source = include_str!("signnoise.rs");
        assert!(source.contains(note));
        assert!(source.contains(&format!("at a batch of one is {} where the bell's formula gives {}", two(exact), two(layer.sign_normal(1.0)))));
        let later = layer_at(LATER);
        let p = plain(&later);
        for k in 0..=10 {
            let measured = if k == 0 { later.sign_one() } else { p.sign[k].useful() };
            assert!(later.sign_normal(size_of(k) as f64) > measured, "level {k}");
        }
        assert!(source.contains(&format!("After {} steps of training the formula is above the measured line at every batch up to {}", LATER, size_of(10))));
    }

    /// What the primer says of sign descent on this layer: at every batch measured, at the
    /// start and later in training, its useful part is below SGD's, so it needs a larger
    /// batch for the same useful part.
    #[test]
    fn sign_descent_needs_larger_batches_here() {
        for steps in [0, LATER] {
            let layer = layer_at(steps);
            let p = plain(&layer);
            for k in 0..LEVELS {
                assert!(p.sign[k].useful() < layer.sgd_useful(size_of(k) as f64), "step {steps}, level {k}");
            }
        }
        assert!(include_str!("../../primers/signnoise.md").contains("on this layer it needs a larger batch than SGD's for the same useful part"));
    }

    /// The numbers each experiment's caption says, from the scene as the button sets it up.
    #[test]
    fn the_experiments_show_what_their_captions_say() {
        let says = |k: usize, what: &str| assert!(TRIALS[k].caption.contains(what), "experiment {k} should say \"{what}\": {}", TRIALS[k].caption);
        let entry = |s: &SignNoise| s.picked().map(|(i, j, _)| format!("Entry $({i}, {j})$")).unwrap();
        // a sure sign
        let s = run(0);
        let (l, c, (_, _, i)) = (s.current().unwrap(), s.coordinate(), s.picked().unwrap());
        says(0, &entry(&s));
        assert_eq!(s.setup.pick, Some(0));
        says(0, &format!("own noise scale $B_i = \\sigma^2/g^2$ is {} and the batch is {}", scale_text(l.own[i]), s.setup.b));
        says(0, &format!(r"$g\sqrt{{B}}/\sigma = {}$", num(c.ratio(), 1)));
        let wrong = normal_cdf(-c.ratio());
        assert!(wrong > 2.45e-8 && wrong < 2.55e-8, "{wrong}");
        says(0, r"a chance of $2.5 \times 10^{-8}$");
        let sample = s.sample_now().unwrap();
        assert_eq!((sample.right, sample.values.len()), (COORD_SAMPLES, COORD_SAMPLES));
        says(0, &format!("all {} sampled batches had it right", count(COORD_SAMPLES)));
        let (sm, ss) = c.sign();
        says(0, &format!(r"(${} \pm {}$)", two(sm), two(ss)));
        says(0, &format!("SGD's still jitters by {} of its mean step", two(c.sgd().1)));
        says(0, &format!("only {} % of the time", num(100.0 * l.right1[i], 0)));
        // a coin toss
        let s = run(1);
        let (l, c, (_, _, i)) = (s.current().unwrap(), s.coordinate(), s.picked().unwrap());
        says(1, &entry(&s));
        assert!(close(COIN_RANK as f64 / COORDS as f64, 0.9, 0.01));
        says(1, &format!("Its own noise scale is {} and the batch {}", scale_text(l.own[i]), s.setup.b));
        says(1, &format!(r"$g\sqrt{{B}}/\sigma = {}$", num(c.ratio(), 2)));
        let sample = s.sample_now().unwrap();
        says(1, &format!("probability {} by the formula, and in {} of {} sampled batches", num(c.right(), 3), num(sample.share_right(), 3), count(sample.values.len())));
        // (the two agree within sampling error: a standard error of √(p(1 − p)/n))
        assert!((sample.share_right() - c.right()).abs() < 2.0 * (0.25 / sample.values.len() as f64).sqrt());
        let (sm, ss) = c.sign();
        says(1, &format!("its mean step is {} and its jitter {}", two(sm), two(ss)));
        says(1, &format!(r"until $B$ nears {}", scale_text(l.own[i])));
        says(1, &format!("with a jitter {} times as large", num(c.sgd().1, 1)));
        let (am, asd) = c.adam();
        says(1, &format!(r"(${} \pm {}$)", two(am), two(asd)));
        assert!(close(sm, 0.80 * c.ratio(), 0.002));
        // momentum
        let s = run(2);
        let c = s.coordinate();
        says(2, &format!(r"$(1 + \beta)/(1 - \beta) = {}$ times larger, {} examples", short(worth(0.9)), whole(16.0 * worth(0.9))));
        let sample = s.sample_now().unwrap();
        says(2, &format!("probability {} (in {} of {} sampled averages)", num(c.right(), 3), num(sample.share_right(), 3), count(sample.values.len())));
        assert!((sample.share_right() - c.right()).abs() < 2.5 * (0.25 / sample.values.len() as f64).sqrt());
        let (sm, ss) = c.sign();
        says(2, &format!(r"Signum's step is ${} \pm {}$", two(sm), two(ss)));
        let (am, asd) = c.adam();
        says(2, &format!("Adam's mean step stays at {} but its jitter falls to {}", two(am), two(asd)));
        let c99 = Coordinate { beta: 0.99, ..c };
        says(2, &format!("worth {} batches and the sign is right with probability {}", short(worth(0.99)), two(c99.right())));
        // batch 4 against batch 256
        let s = run(3);
        let at = |b: usize| (0..4).map(|r| s.useful_at(r, b)).collect::<Vec<_>>();
        let (four, big) = (at(4), at(256));
        let names = ["SGD", "sign descent", "Adam", "Muon"];
        for (r, name) in names.iter().enumerate() {
            says(3, &format!("{name} {} and {}", two(four[r].0), two(four[r].1)));
            says(3, &format!("{name} {} and {}", two(big[r].0), two(big[r].1)));
        }
        says(3, &format!("$B_{{\\text{{simple}}}}$ is {}", whole(s.current().unwrap().simple)));
        let p = s.plain.as_ref().unwrap();
        let l = s.current().unwrap();
        for k in 1..LEVELS {
            let b = size_of(k) as f64;
            let muon = p.muon[k].useful();
            let others = [l.sgd_useful(b), p.sign[k].useful(), l.adam_useful(b, 1.0).0];
            if b <= 512.0 {
                assert!(others.iter().all(|&o| muon < o), "B {b}: Muon {muon} is not the lowest of {others:?}");
            } else {
                assert!(muon < others[0] && others[1..].iter().all(|&o| muon > o), "B {b}: Muon {muon} is not second to SGD: {others:?}");
            }
        }
        // later in training
        let early = run(3);
        let s = run(4);
        let (l0, l) = (early.current().unwrap(), s.current().unwrap());
        says(4, &format!("trained {} steps", count(LATER)));
        says(4, &format!("its loss fell from {} to {}", sig(l0.loss), sig(l.loss)));
        says(4, &format!("became {} times shorter", num(l0.norm() / l.norm(), 0)));
        says(4, &format!("rose from {} to {}", whole(l0.simple), whole(l.simple)));
        says(4, &format!("the median own noise scale from {} to {}", whole(l0.median_scale()), whole(l.median_scale())));
        says(4, &format!("and {} of the 256 coordinates are below a batch of 256 ({} were)", l.below(256.0), l0.below(256.0)));
        let now: Vec<String> = (0..4).map(|r| two(s.useful_at(r, 256).0)).collect();
        let then: Vec<String> = (0..4).map(|r| two(early.useful_at(r, 256).0)).collect();
        says(4, &format!("{} for SGD, {} for sign descent, {} for Adam and {} for Muon, against {}, {}, {} and {} before", now[0], now[1], now[2], now[3], then[0], then[1], then[2], then[3]));
        let p = s.plain.as_ref().unwrap();
        for k in 3..LEVELS {
            assert!(p.muon[k].useful() > p.sign[k].useful(), "level {k}");
        }
        assert!(p.muon[2].useful() < p.sign[2].useful());
        assert!(l.simple / l0.simple > 10.0);
        // Muon at a small batch
        let s = run(5);
        let l = s.current().unwrap();
        let p = s.plain.as_ref().unwrap();
        let (batch, noise) = p.spectra(2);
        let top = batch[0];
        assert!(batch[..4].iter().all(|&v| v > 1e-7 * top) && batch[4..].iter().all(|&v| v < 1e-7 * top), "{batch:?}");
        let goal = s.spectrum_goal();
        assert!(goal[SIDE..SIDE + 4].iter().all(|&v| v > NOWHERE) && goal[SIDE + 4..2 * SIDE].iter().all(|&v| v == NOWHERE));
        says(5, &format!("a largest singular value of {}, over twice the true gradient's largest, {}", two(noise[0]), two(l.sv[0])));
        assert!(noise[0] > 2.0 * l.sv[0] && l.sv.iter().all(|&v| v < noise[0]));
        let (alone, with) = s.useful_at(3, 4);
        says(5, &format!("has a useful part of {}", two(alone)));
        says(5, &format!("it is {}, about what one batch of 64 gives", two(with)));
        assert!((p.muon[6].useful() - with).abs() < 0.02, "{} {}", p.muon[6].useful(), with);
    }

    /// The constants the panel quotes: what a momentum's average is worth, and the factor
    /// in Naganuma and co-authors' batch for half the progress of sign descent.
    #[test]
    fn the_panel_quotes_its_constants_right() {
        let source = include_str!("signnoise.rs");
        assert!(close((1.0 / (1.0 - 0.5f64.sqrt())).powi(2), 11.66, 0.005) && source.contains(r"$11.7\,B_{\ell_1}$"));
        assert!(source.contains(&format!(r"{} times at $\beta = 0.9$, {} at 0.99", short(worth(0.9)), short(worth(0.99)))));
        // with momentum 0.9 the stream records an average every 20 steps: 2/(1 − β)
        assert_eq!(Stream::spacing(0.9).0, 20);
        assert!(size_of(Stream::levels_for(0.9) - 1) == 256 && source.contains(r"(up to 256 at $\beta = 0.9$)"));
    }

    /// The scene opens as the experiment OPENING sets it up.
    #[test]
    fn the_scene_opens_on_an_experiment() {
        let opened = SignNoise::create();
        let mut pressed = SignNoise::create();
        pressed.set_up(4);
        pressed.set_up(OPENING);
        assert_eq!(opened.setup, pressed.setup);
        assert_eq!(opened.setup, trial(OPENING).kept());
    }

    /// Settings at and beyond their ends, and nonsense, are kept to the controls'
    /// ranges, and every number the picture glides to stays finite.
    #[test]
    fn degenerate_settings_are_kept_finite() {
        let mut s = run(1);
        for setup in [
            Setup { g: f64::NAN, pick: None, b: 0, beta: f64::NAN, compare: Some(0), trained: 0, page: Page::Network },
            Setup { g: 1e9, pick: None, b: usize::MAX, beta: 5.0, compare: None, trained: 0, page: Page::Coordinate },
            Setup { g: LEAST_G, pick: Some(10_000), b: 1, beta: MOST_BETA, compare: Some(MOST_B), trained: 0, page: Page::Coordinate },
            Setup { g: 0.0, pick: Some(COORDS - 1), b: MOST_B, beta: 0.0, compare: None, trained: 0, page: Page::Coordinate },
        ] {
            s.setup = setup.kept();
            let k = s.setup;
            assert!(k.g.is_finite() && k.g >= LEAST_G && k.g <= MOST_G && k.b >= 1 && k.b <= MOST_B && k.b.is_power_of_two() && (0.0..=MOST_BETA).contains(&k.beta), "{k:?}");
            assert!(k.pick.is_none_or(|r| r < COORDS) && k.compare.is_none_or(|c| c.is_power_of_two() && c <= MOST_B));
            let budget = Budget::new(1e9);
            let mut rounds = 0;
            while s.busy() && rounds < 100 {
                s.work(&budget);
                rounds += 1;
            }
            assert!(!s.busy(), "{k:?}");
            s.follow(0.016, true);
            let c = s.shown_coordinate();
            assert!(c.g.is_finite() && c.b.is_finite() && c.beta.is_finite());
            for f in [Coordinate::sign, Coordinate::sgd, Coordinate::adam] {
                let (m, sd) = f(&c);
                assert!(m.is_finite() && sd.is_finite(), "{k:?}");
            }
            let look = &s.look;
            for g in [&look.one, &look.hist, &look.net, &look.strip, &look.spectrum] {
                assert!(g.shown.iter().all(|v| v.is_finite()), "{k:?}: {:?}", g.shown);
            }
            let r = s.bell_goal(&c, 0.0);
            assert!(r.iter().all(|v| v.is_finite()) && r[1] > r[0] && r[2] > 0.0, "{k:?}: {r:?}");
        }
        // the noisiest coordinate: an own noise scale of millions is still a finite picture
        let l = s.current().unwrap();
        assert!(l.own[l.ranked[COORDS - 1]] > 1e5);
        assert!(power_of_two(f64::NAN) == 1 && power_of_two(0.3) == 1 && power_of_two(3.0) == 4 && power_of_two(1e12) == MOST_B);
    }

    /// The layout keeps every chart in the window and apart, from a phone to a desktop.
    #[test]
    fn the_layout_fits_every_window() {
        for (w, h) in [(1070.0, 1000.0), (1600.0, 1000.0), (1220.0, 700.0), (900.0, 560.0), (390.0, 800.0), (360.0, 640.0), (700.0, 500.0), (300.0, 600.0), (500.0, 400.0), (240.0, 180.0)] {
            for page in [Page::Coordinate, Page::Network] {
                let lay = Layout::of(Vec2::new(w, h), 40.0, page);
                if lay.tiny() {
                    continue;
                }
                let inside = |(a, b): Rect| a.x >= 0.0 && a.y >= 0.0 && b.x <= w + 0.5 && b.y <= h + 0.5;
                let apart = |(a, b): Rect, (c, d): Rect| b.x <= c.x || d.x <= a.x || b.y <= c.y || d.y <= a.y;
                let mut boxes: Vec<Rect> = vec![lay.train, lay.again];
                if lay.one {
                    boxes.extend([lay.bell, lay.steps]);
                }
                if lay.net {
                    boxes.extend([lay.useful, lay.strip, lay.spectrum]);
                }
                if !(lay.one && lay.net) {
                    boxes.extend(lay.pages);
                }
                assert!(lay.one || lay.net, "{w}x{h}");
                for (i, a) in boxes.iter().enumerate() {
                    assert!(inside(*a), "{w}x{h} {page:?}: box {i} {a:?}");
                    for b in &boxes[i + 1..] {
                        assert!(apart(*a, *b), "{w}x{h} {page:?}: {a:?} and {b:?} overlap");
                    }
                }
                if h >= 560.0 {
                    for r in [lay.bell, lay.steps, lay.useful, lay.spectrum] {
                        if r.1.x - r.0.x > 1.5 {
                            assert!(r.1.y - r.0.y > 60.0 && r.1.x - r.0.x > 150.0, "{w}x{h} {page:?}: {r:?}");
                        }
                    }
                }
            }
        }
    }

    /// After an experiment the work is done and the picture comes to rest.
    #[test]
    fn the_picture_comes_to_rest() {
        let mut s = SignNoise::create();
        for k in [1, 5, 0, 2] {
            s.set_up(k);
            let mut frames = 0;
            while s.animated() {
                s.work(&Budget::new(0.004));
                s.follow(1.0 / 60.0, false);
                frames += 1;
                assert!(frames < 6000, "experiment {k} still moving");
            }
            assert!(!s.busy() && !s.look.moving);
            let before = s.look.net.shown.clone();
            s.follow(1.0 / 60.0, false);
            assert_eq!(before, s.look.net.shown);
        }
    }

    #[test]
    #[ignore]
    fn explore_costs() {
        let mut t = Trainer::new();
        let mut per = PerExample::new(&t.problem, &t.model);
        let t0 = std::time::Instant::now();
        per.advance(&t.problem, &t.model, &mut t.tape, usize::MAX);
        let t1 = std::time::Instant::now();
        let layer = Layer::of(&per, &t.model, &t.problem, &mut t.tape, 0);
        let t2 = std::time::Instant::now();
        let mut cov = Covariances::new(0);
        let _ = cov.advance(&layer, 64);
        let t3 = std::time::Instant::now();
        println!("per-example {:?}, layer {:?}, 64 examples of covariances {:?}", t1 - t0, t2 - t1, t3 - t2);
        let mut s = run(1);
        s.follow(0.016, true);
        let t4 = std::time::Instant::now();
        for _ in 0..100 {
            s.follow(0.016, false);
        }
        println!("follow at rest {:?}", (std::time::Instant::now() - t4) / 100);
        let mut p = Plain::new(1);
        let t5 = std::time::Instant::now();
        p.trial(s.current().unwrap());
        println!("one trial {:?}", std::time::Instant::now() - t5);
        let mut m = Stream::new(0.9, 1);
        let t6 = std::time::Instant::now();
        for _ in 0..8 {
            m.step(s.current().unwrap());
        }
        println!("eight stream steps {:?}", std::time::Instant::now() - t6);
        let mut tr = Trainer::new();
        let t7 = std::time::Instant::now();
        tr.work(100, &Budget::new(1e9));
        println!("100 training steps {:?}", std::time::Instant::now() - t7);
    }

    #[test]
    #[ignore]
    fn explore_experiments() {
        for k in 0..TRIALS.len() {
            let s = run(k);
            let l = s.current().unwrap();
            let c = s.coordinate();
            let su = s.setup;
            println!("== {k} {} {:?}", TRIALS[k].button, su);
            if let Some((i, j, idx)) = s.picked() {
                let sample = s.sample_now().unwrap();
                println!(
                    "   coord ({i},{j}) own {:.1} g {:.4} r {:.3} r_used {:.3} P {:.4} measured {:.4} (n {}) right1 {:.3} sign {:?} sgd {:?} adam {:?}",
                    l.own[idx], c.g, c.ratio(), c.ratio_used(), c.right(), sample.share_right(), sample.values.len(), l.right1[idx], c.sign(), c.sgd(), c.adam()
                );
            }
            println!("   loss {:.4} |G| {:.4e} simple {:.1} l1 {:.1} nuclear {:.1} {:.1} whole {:.1} {:.1} median {:.1} below(B) {} right1<.5 {}", l.loss, l.norm(), l.simple, l.l1, l.nuclear[0], l.nuclear[1], l.whole_simple, l.whole_l1, l.median_scale(), l.below(su.b as f64), l.right1.iter().filter(|&&r| r < 0.5).count());
            for b in [Some(su.b), su.compare].into_iter().flatten() {
                let rows: Vec<String> = (0..4).map(|r| { let (a, w) = s.useful_at(r, b); format!("{a:.3}/{w:.3}") }).collect();
                let (_, rms) = l.adam_useful(b as f64, worth(su.beta));
                println!("   at B {b}: sgd sign adam muon = {:?}; adam rms with momentum {:.3}", rows, rms);
            }
            let p = s.plain.as_ref().unwrap();
            let kk = su.b.trailing_zeros() as usize;
            let (bs, ns) = p.spectra(kk);
            println!("   above {} ; noise top {:.3e} G top {:?} batch {:?}", l.sv.iter().filter(|&&x| x > ns[0]).count(), ns[0], &l.sv[..3], &bs[..5]);
        }
    }

    #[test]
    #[ignore]
    fn explore() {
        let _ = (MOST_B, Coordinate { g: 1.0, b: 1.0, beta: 0.0 }.right());
        for steps in [0usize, 3000] {
            let t0 = std::time::Instant::now();
            let layer = layer_at(steps);
            let t1 = std::time::Instant::now();
            let p = plain(&layer);
            let t2 = std::time::Instant::now();
            let s = stream(&layer, 0.9);
            let t3 = std::time::Instant::now();
            println!(
                "step {steps}: loss {:.4} |G| {:.3e} simple {:.1} l1 {:.1} nuclear {:?} whole {:.1} {:.1}; median own {:.1}",
                layer.loss,
                layer.norm(),
                layer.simple,
                layer.l1,
                layer.nuclear,
                layer.whole_simple,
                layer.whole_l1,
                layer.median_scale()
            );
            println!("   times: layer {:?} plain {:?} stream {:?} ({} levels, {} steps)", t1 - t0, t2 - t1, t3 - t2, s.levels, s.length());
            println!("   right1 below half: {}", layer.right1.iter().filter(|&&r| r < 0.5).count());
            for k in 0..LEVELS {
                let b = size_of(k) as f64;
                let (a0, _) = layer.adam_useful(b, 1.0);
                let (a9, rms) = layer.adam_useful(b, worth(0.9));
                println!(
                    "   B {:5}: sgd {:.3}/{:.3} sign {:.3}±{:.3} (normal {:.3}) muon {:.3}±{:.3} adam0 {a0:.3} adam.9 {a9:.3} (rms {rms:.3}) | momentum: sgd {:.3} signum {:.3} (normal {:.3}) muon {:.3} | below {}",
                    size_of(k),
                    p.sgd[k].useful(),
                    layer.sgd_useful(b),
                    p.sign[k].useful(),
                    p.sign[k].error(),
                    layer.sign_normal(b),
                    p.muon[k].useful(),
                    p.muon[k].error(),
                    layer.sgd_useful(b * worth(0.9)),
                    s.sign[k].useful(),
                    layer.sign_normal(b * worth(0.9)),
                    s.muon[k].useful(),
                    layer.below(b)
                );
            }
            let (bs, ns) = p.spectra(2);
            println!("   sv G {:?}", layer.sv.iter().map(|x| format!("{x:.1e}")).collect::<Vec<_>>());
            println!("   sv B=4 {:?}", bs.iter().map(|x| format!("{x:.1e}")).collect::<Vec<_>>());
            println!("   sv N=4 {:?}", ns.iter().map(|x| format!("{x:.1e}")).collect::<Vec<_>>());
            let mut c = CoordSample::new(layer.ranked[COORDS / 2], 16, 0.0, 9);
            c.advance(&layer, usize::MAX);
            println!("   median coordinate at B 16: right {:.3}", c.share_right());
        }
    }
}


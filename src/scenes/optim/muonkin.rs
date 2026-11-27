//! Muon's relatives: what fifteen optimisers do to one gradient matrix. A layer of a
//! small network (a 16 by 16 weight matrix of a tanh MLP on three spirals, its weights
//! frozen part of the way through training) hands the same stream of mini-batch
//! gradients to every rule of the family the course's library implements, each in its
//! published form; each takes its step from the same weights, so only the rule and the
//! history differ. The tree sorts them by what they do to G = U S Vᵀ (keep U and V and
//! change S; act on entries; whiten with remembered statistics) and lights one; beside
//! it, the singular values of that rule's step and the gradient's, and the angles
//! between its step and Muon's, the gradient's and the whole data's orthogonal step.
//! Under them, a race of the working members on the same network, each at a rate a
//! stated search chose, from a table that the ignored test
//! `the_stored_race_is_what_the_search_finds` regenerates and compares.
//!
//! The rules are `optim::Rule` and `optim::Hyper` as the library gives them (their
//! sources and forms are named there and in the panel); "exactly" here replaces the
//! reference's polynomial by the exact polar factor (`optim::Ortho::Exact`). Each
//! published claim the panel quotes is attributed there to its authors, with its scope
//! (checked against the papers in the course notes, muon.md §§2 to 6 and §6.3).

use super::*;
use crate::clock::Budget;
use crate::math::arch::{MlpSpec, Model};
use crate::math::autograd::{Act, Tape};
use crate::math::optim::{self, Hyper, Iteration, Optimizer, Ortho, Rule, Scratch};
use crate::math::terrain::{Draw, Problem, Sampler};
use crate::scenes::stats::plot::{self, Glide, Plot, Rect};

pub const DEF: SceneDef = SceneDef {
    id: "muonkin",
    title: "Muon's relatives",
    group: GROUP,
    blurb: "Sign descent, Shampoo, SOAP and Muon's own variants at work on one gradient matrix: what each does to its singular values, and what they share.",
    make: |_| Box::new(MuonKin::create()),
};

// ----- the network and its data ------------------------------------------------------------

/// Units in each of the network's three hidden layers: the matrix looked at is WIDTH by
/// WIDTH.
const WIDTH: usize = 16;
const DEPTH: usize = 3;
/// Points on three spiral arms, from a seed of their own.
const POINTS: usize = 192;
const DATA_SEED: u64 = 7;
/// The weights' seed (the race runs two more).
const NET_SEED: u64 = 1;
/// The matrix whose gradient the scene looks at: the second hidden layer's weights, the
/// first that Muon takes (the reference gives the layer that reads the data to Adam).
const LAYER_NAME: &str = "layer 2 weights";
/// Where the weights are frozen: after this many steps of Adam (rate 0.01, batches of
/// 16), a network part of the way through learning the spirals.
const WARM: usize = 300;
const WARM_LR: f64 = 0.01;
const WARM_BATCH: usize = 16;

fn spec() -> MlpSpec {
    MlpSpec { inputs: 2, width: WIDTH, depth: DEPTH, outputs: 3, act: Act::Tanh, ..MlpSpec::default() }
}

fn problem() -> Problem {
    Problem::spirals(POINTS, 3, &mut Rng::new(DATA_SEED))
}

/// The network after `steps` steps of Adam from the seed's weights.
fn warmed(steps: usize, seed: u64, problem: &Problem) -> Model {
    let mut model = Model::mlp(&spec(), &mut Rng::new(seed));
    let mut tape = Tape::new();
    let mut opt = Optimizer::new(&model, Rule::Adam, Hyper { lr: WARM_LR, ..Hyper::default_for(Rule::Adam) });
    let mut batches = Sampler::new(problem.len(), WARM_BATCH, Draw::WithoutReplacement, Rng::new(seed.wrapping_add(100)));
    for _ in 0..steps {
        let idx = batches.deal().to_vec();
        problem.grad(&mut model, &mut tape, &idx);
        opt.step(&mut model, WARM_LR);
    }
    model
}

// ----- the members -------------------------------------------------------------------------

/// The working members of the family: the rules the course's library implements, each
/// in the form its source gives (see `optim::Rule` and `optim::Hyper`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
enum Member {
    Gd,
    Sign,
    Signum,
    Lion,
    Adam,
    /// Muon with its momentum switched off: the orthogonalised gradient.
    Ortho,
    /// Shampoo with no memory and no damping: a fresh Shampoo at every step.
    Fresh,
    Spectral,
    Muon,
    OSgdm,
    Shampoo,
    Soap,
    NorMuon,
    AdaMuon,
    Scion,
}

impl Member {
    const ALL: [Member; 15] = [
        Member::Gd,
        Member::Sign,
        Member::Signum,
        Member::Lion,
        Member::Adam,
        Member::Ortho,
        Member::Fresh,
        Member::Spectral,
        Member::Muon,
        Member::OSgdm,
        Member::Shampoo,
        Member::Soap,
        Member::NorMuon,
        Member::AdaMuon,
        Member::Scion,
    ];

    fn index(self) -> usize {
        Member::ALL.iter().position(|&m| m == self).unwrap_or(0)
    }

    fn name(self) -> &'static str {
        match self {
            Member::Gd => "gradient descent",
            Member::Sign => "sign descent",
            Member::Signum => "Signum",
            Member::Lion => "Lion",
            Member::Adam => "Adam",
            Member::Ortho => "Muon, no momentum",
            Member::Fresh => "Shampoo, no memory",
            Member::Spectral => "spectral descent",
            Member::Muon => "Muon",
            Member::OSgdm => "Orthogonal-SGDM",
            Member::Shampoo => "Shampoo",
            Member::Soap => "SOAP",
            Member::NorMuon => "NorMuon",
            Member::AdaMuon => "AdaMuon",
            Member::Scion => "Scion",
        }
    }

    /// The rule and its settings, as the library gives the published form; `exact`
    /// replaces the reference's polynomial by the exact orthogonal factor where the
    /// member uses it.
    fn rule(self, exact: bool) -> (Rule, Hyper) {
        let ortho = if exact { Ortho::Exact } else { Ortho::Polynomial(Iteration::JORDAN) };
        match self {
            Member::Gd => (Rule::Sgd, Hyper::default_for(Rule::Sgd)),
            Member::Sign => (Rule::SignSgd, Hyper::default_for(Rule::SignSgd)),
            Member::Signum => (Rule::Signum, Hyper::default_for(Rule::Signum)),
            Member::Lion => (Rule::Lion, Hyper::default_for(Rule::Lion)),
            Member::Adam => (Rule::Adam, Hyper::default_for(Rule::Adam)),
            Member::Ortho => (Rule::Muon, Hyper { momentum: 0.0, ortho, ..Hyper::muon_reference() }),
            Member::Fresh => (Rule::Shampoo, Hyper { damping: 0.0, ..Hyper::shampoo() }),
            Member::Spectral => (Rule::SpectralDescent, Hyper::spectral_descent()),
            Member::Muon => (Rule::Muon, Hyper { ortho, ..Hyper::muon_reference() }),
            Member::OSgdm => (Rule::Muon, Hyper::orthogonal_sgdm()),
            Member::Shampoo => (Rule::Shampoo, Hyper::shampoo()),
            Member::Soap => (Rule::Soap, Hyper::soap()),
            Member::NorMuon => (Rule::NorMuon, Hyper { ortho, ..Hyper::normuon() }),
            Member::AdaMuon => (Rule::AdaMuon, Hyper { ortho, ..Hyper::adamuon() }),
            Member::Scion => (Rule::Scion, Hyper::scion()),
        }
    }
}

// ----- matrices --------------------------------------------------------------------------

/// The cosine of the angle between two matrices, as vectors of their entries: ⟨A, B⟩ /
/// (‖A‖_F ‖B‖_F). NaN if either is zero.
fn cosine(a: &[f64], b: &[f64]) -> f64 {
    let d: f64 = a.iter().zip(b).map(|(x, y)| x * y).sum();
    let (na, nb) = (optim::frobenius(a), optim::frobenius(b));
    if na > 0.0 && nb > 0.0 && d.is_finite() {
        (d / (na * nb)).clamp(-1.0, 1.0)
    } else {
        f64::NAN
    }
}

/// The angle in degrees for a cosine (NaN stays NaN).
fn degrees(c: f64) -> f64 {
    c.acos().to_degrees()
}

/// Singular values below this share of the largest are not resolved here (they are
/// found from the eigenvalues of A Aᵀ, whose rounding is at 1e-16 of the largest's
/// square) and are drawn as "below" it.
const RESOLVED: f64 = 1e-6;

// ----- the stream: one layer's gradients at frozen weights ---------------------------------

/// How long a history the stream runs to.
const T_MAX: usize = 64;

/// The batch sizes offered: one example, a few, more, and all the data.
const BATCHES: [usize; 5] = [1, 4, 16, 64, POINTS];

/// What the stream depends on.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Setup {
    /// Examples per gradient.
    batch: usize,
    /// Orthogonalise exactly, in place of the reference's polynomial.
    exact: bool,
    /// The stream's batches' seed.
    seed: u64,
}

impl Setup {
    fn base() -> Setup {
        Setup { batch: POINTS, exact: true, seed: 1 }
    }
}

/// What one member's step at one point of the stream is, on the layer looked at.
#[derive(Clone, Copy, Debug)]
struct Step {
    /// Its singular values over the largest (largest first; 0 where below `RESOLVED`).
    sv: [f64; WIDTH],
    /// The cosines of the angle between its direction (minus the update) and: Muon's
    /// direction at the same step; the present gradient; the orthogonal step of the
    /// whole data's gradient (exact or the polynomial, as the stream's setup says).
    cos_muon: f64,
    cos_grad: f64,
    cos_whole: f64,
    /// The update's spectral norm at learning rate 1.
    spectral: f64,
    /// Its stable rank.
    stable: f64,
    /// Whether it moved at all.
    moved: bool,
}

impl Step {
    const NONE: Step = Step { sv: [0.0; WIDTH], cos_muon: f64::NAN, cos_grad: f64::NAN, cos_whole: f64::NAN, spectral: 0.0, stable: 0.0, moved: false };
}

/// What the gradient is at one point of the stream.
#[derive(Clone, Copy, Debug)]
struct Grad {
    sv: [f64; WIDTH],
    stable: f64,
    /// The cosine with the whole data's gradient.
    cos_whole: f64,
    /// Its rank as the exact orthogonal factor counts it, and as Shampoo's inverse roots
    /// do (singular values above 4 n ε of the largest, and above √(4 n ε) of it).
    rank_polar: usize,
    rank_roots: usize,
}

/// The members' steps on a stream of mini-batch gradients of one layer, all at the same
/// frozen weights: step t of the stream hands every member the gradient of the t-th
/// batch (the same batches for all), and each takes its step at learning rate 1 from
/// those weights (they are put back after each step, so that only the history differs).
/// Worked out a slice at a time, as far as a budget allows.
struct Stream {
    setup: Setup,
    problem: Problem,
    frozen: Model,
    layer: usize,
    rows: usize,
    cols: usize,
    work: Model,
    tape: Tape,
    sampler: Sampler,
    members: Vec<(Member, Model, Optimizer)>,
    scratch: Scratch,
    /// The whole data's gradient of the layer, and its orthogonal step.
    whole: Vec<f64>,
    whole_step: Vec<f64>,
    /// Per step of the stream: the gradient, and each member's step.
    grads: Vec<Grad>,
    steps: Vec<Vec<Step>>,
    /// Per step: the largest entry of Shampoo-without-memory's step minus the exact
    /// orthogonal factor of the gradient, over the largest entry of the latter.
    bridge: Vec<f64>,
    /// Per step: the cosine between the exact orthogonal factor of the gradient and the
    /// reference's polynomial applied to it.
    polynomial: Vec<f64>,
    /// Muon with Orthogonal-SGDM's momentum (0.9, heavy ball, no Nesterov), orthogonalised
    /// exactly: the order of the two operations is then all that differs between them.
    matched: (Model, Optimizer),
    /// Per step: its cosine with the whole data's orthogonal step.
    matched_whole: Vec<f64>,
    /// The latest update of each member (the layer's), and the gradient.
    last: Vec<Vec<f64>>,
    last_grad: Vec<f64>,
}

impl Stream {
    fn new(setup: Setup, problem: &Problem, frozen: &Model) -> Stream {
        let layer = frozen.index_of(LAYER_NAME).unwrap_or(0);
        let (rows, cols) = frozen.groups().get(layer).map_or((1, 1), |g| (g.rows, g.cols));
        let mut tape = Tape::new();
        let mut full = frozen.clone();
        problem.full_grad(&mut full, &mut tape);
        let whole = full.groups().get(layer).map_or(Vec::new(), |g| g.grad.clone());
        let mut scratch = Scratch::new();
        let mut whole_step = vec![0.0; rows * cols];
        if setup.exact {
            optim::polar_into(&whole, rows, cols, &mut whole_step, &mut scratch);
        } else {
            whole_step.copy_from_slice(&whole);
            Iteration::JORDAN.run(&mut whole_step, rows, cols, &mut scratch);
        }
        let matched_hyper = Hyper { momentum: 0.9, convention: optim::Convention::Sum, nesterov: false, ortho: Ortho::Exact, ..Hyper::muon_reference() };
        let matched = (frozen.clone(), Optimizer::new(frozen, Rule::Muon, matched_hyper));
        let members = Member::ALL
            .iter()
            .map(|&m| {
                let (rule, hyper) = m.rule(setup.exact);
                let model = frozen.clone();
                let opt = Optimizer::new(&model, rule, hyper);
                (m, model, opt)
            })
            .collect();
        Stream {
            setup,
            problem: problem.clone(),
            frozen: frozen.clone(),
            layer,
            rows,
            cols,
            work: frozen.clone(),
            tape,
            sampler: Sampler::new(problem.len(), setup.batch, Draw::WithoutReplacement, Rng::new(setup.seed.wrapping_mul(0x9E37_79B9).wrapping_add(5))),
            members,
            scratch,
            whole,
            whole_step,
            grads: Vec::with_capacity(T_MAX),
            steps: Vec::with_capacity(T_MAX),
            bridge: Vec::with_capacity(T_MAX),
            polynomial: Vec::with_capacity(T_MAX),
            matched,
            matched_whole: Vec::with_capacity(T_MAX),
            last: vec![Vec::new(); Member::ALL.len()],
            last_grad: Vec::new(),
        }
    }

    fn done(&self) -> usize {
        self.steps.len()
    }

    fn finished(&self) -> bool {
        self.done() >= T_MAX
    }

    /// Relative singular values, largest first, zero below `RESOLVED`.
    fn spectrum(a: &[f64], rows: usize, cols: usize, scratch: &mut Scratch, out: &mut Vec<f64>) -> ([f64; WIDTH], f64, f64) {
        optim::singular_values_into(a, rows, cols, out, scratch);
        let top = out.first().copied().unwrap_or(0.0);
        let mut sv = [0.0; WIDTH];
        for (k, s) in out.iter().take(WIDTH).enumerate() {
            let r = if top > 0.0 { s / top } else { 0.0 };
            sv[k] = if r >= RESOLVED { r } else { 0.0 };
        }
        let stable = if top > 0.0 { out.iter().map(|s| s * s).sum::<f64>() / (top * top) } else { 0.0 };
        (sv, top, stable)
    }

    /// One more step of the stream.
    fn step(&mut self) {
        if self.finished() {
            return;
        }
        let idx = self.sampler.deal().to_vec();
        self.problem.grad(&mut self.work, &mut self.tape, &idx);
        let (r, c, layer) = (self.rows, self.cols, self.layer);
        let g = self.work.groups()[layer].grad.clone();
        let mut out = Vec::with_capacity(WIDTH);
        let (sv, _, stable) = Self::spectrum(&g, r, c, &mut self.scratch, &mut out);
        // the rank the two counts give: the exact factor's and the roots'
        let top = out.first().copied().unwrap_or(0.0);
        let eps = f64::EPSILON;
        let n = r.max(c) as f64;
        let mut polar = vec![0.0; r * c];
        optim::polar_into(&g, r, c, &mut polar, &mut self.scratch);
        let rank_polar = optim::singular_values(&polar, r, c).iter().filter(|&&s| s > 0.5).count();
        let rank_roots = out.iter().filter(|&&s| s > (4.0 * n * eps).sqrt() * top).count();
        self.grads.push(Grad { sv, stable, cos_whole: cosine(&g, &self.whole), rank_polar, rank_roots });
        let mut updates: Vec<Vec<f64>> = Vec::with_capacity(self.members.len());
        let frozen = &self.frozen;
        for (m, model, opt) in self.members.iter_mut() {
            for ((dst, src), base) in model.groups_mut().iter_mut().zip(self.work.groups()).zip(frozen.groups()) {
                dst.grad.copy_from_slice(&src.grad);
                dst.value.copy_from_slice(&base.value);
            }
            if *m == Member::Fresh {
                // no memory: a new Shampoo for every step
                opt.reset();
            }
            opt.step(model, 1.0);
            updates.push(opt.update(layer).to_vec());
        }
        let muon = Member::Muon.index();
        let dirs: Vec<Vec<f64>> = updates.iter().map(|u| u.iter().map(|x| -x).collect()).collect();
        let mut row = Vec::with_capacity(self.members.len());
        for d in &dirs {
            let moved = d.iter().any(|&x| x != 0.0);
            if !moved {
                row.push(Step::NONE);
                continue;
            }
            let (sv, spectral, stable) = Self::spectrum(d, r, c, &mut self.scratch, &mut out);
            row.push(Step {
                sv,
                cos_muon: cosine(d, &dirs[muon]),
                cos_grad: cosine(d, &g),
                cos_whole: cosine(d, &self.whole_step),
                spectral,
                stable,
                moved,
            });
        }
        let fresh = &dirs[Member::Fresh.index()];
        let big = optim::max_abs(&polar).max(1e-300);
        self.bridge.push(fresh.iter().zip(&polar).map(|(a, b)| (a - b).abs()).fold(0.0, f64::max) / big);
        let mut poly = g.clone();
        Iteration::JORDAN.run(&mut poly, r, c, &mut self.scratch);
        self.polynomial.push(cosine(&poly, &polar));
        let (model, opt) = &mut self.matched;
        for ((dst, src), base) in model.groups_mut().iter_mut().zip(self.work.groups()).zip(frozen.groups()) {
            dst.grad.copy_from_slice(&src.grad);
            dst.value.copy_from_slice(&base.value);
        }
        opt.step(model, 1.0);
        let d: Vec<f64> = opt.update(layer).iter().map(|x| -x).collect();
        self.matched_whole.push(cosine(&d, &self.whole_step));
        self.steps.push(row);
        self.last = dirs;
        self.last_grad = g;
    }

    /// Steps of the stream until `upto` (at most `T_MAX`) or until the budget is spent
    /// (at least one, if any is owed).
    fn work(&mut self, upto: usize, budget: &Budget) {
        let upto = upto.min(T_MAX);
        let mut first = true;
        while self.done() < upto && (first || !budget.spent()) {
            self.step();
            first = false;
        }
    }

    /// Member m's step at stream step t (from 1).
    fn at(&self, t: usize, m: Member) -> Option<&Step> {
        self.steps.get(t.checked_sub(1)?)?.get(m.index())
    }

    fn grad_at(&self, t: usize) -> Option<&Grad> {
        self.grads.get(t.checked_sub(1)?)
    }
}

// ----- the race --------------------------------------------------------------------------

/// The members that race (Shampoo without memory takes the step of Muon without
/// momentum, orthogonalised exactly, on every hidden matrix, so it is left out).
#[cfg(test)]
const RACERS: [Member; 14] = [
    Member::Gd,
    Member::Sign,
    Member::Signum,
    Member::Lion,
    Member::Adam,
    Member::Ortho,
    Member::Spectral,
    Member::Muon,
    Member::OSgdm,
    Member::Shampoo,
    Member::Soap,
    Member::NorMuon,
    Member::AdaMuon,
    Member::Scion,
];

/// Steps of a race, and every how many the whole data's loss is noted.
const RACE_STEPS: usize = 1000;
const RACE_EVERY: usize = 20;
const RACE_POINTS: usize = RACE_STEPS / RACE_EVERY + 1;
const RACE_BATCH: usize = 16;
/// The seeds raced (each its own starting weights and its own order of batches; every
/// member gets the same ones).
const RACE_SEEDS: [u64; 3] = [1, 2, 3];
/// The learning rates searched: 10^(k/4) for k from `GRID_LO` to `GRID_HI`, 10⁻⁵ to
/// 100, four to a factor of ten.
const GRID_LO: i32 = -20;
const GRID_HI: i32 = 8;
const GRID_N: usize = (GRID_HI - GRID_LO + 1) as usize;
/// Scion's radii searched (its sources give none for a network like this one).
#[cfg(test)]
const RADII: [f64; 7] = [1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0];
/// The rate of the Adam that the groups a matrix rule leaves (the input layer, the
/// read-out, the biases) take in the race: Adam's own choice in the search, 10^(−7/4).
/// (The library ties that rate to the matrix rule's; fixed, the race compares the
/// matrix rules and not how fast the read-out learns.)
const AUX_K: i32 = -7;

/// Scion's best score at each radius of `RADII` (over every rate; the search's).
#[cfg(test)]
const SCION_BY_RADIUS: [f64; RADII.len()] = [9.03714e-1, 6.64016e-1, 8.58099e-2, 2.51060e-2, 1.15411e-2, 8.12487e-3, 5.97054e-3];

/// The race as the search left it (`search`; the ignored test
/// `the_stored_race_is_what_the_search_finds` runs it again and compares).
#[rustfmt::skip]
const RACE: [Raced; 14] = [
    Raced { member: Member::Gd, k: -2, rho: 1.0, finals: [9.31800e-2, 2.91575e-1, 3.21196e-2],
        curve: [1.11852e0, 1.09543e0, 1.07596e0, 1.06990e0, 1.10979e0, 1.04596e0, 1.03076e0, 1.04817e0, 1.02973e0, 1.01266e0, 1.01036e0, 9.97102e-1, 9.90534e-1, 1.02010e0, 1.00161e0, 9.42740e-1, 1.04879e0, 9.22859e-1, 8.88297e-1, 8.99089e-1, 8.95410e-1, 8.55773e-1, 1.01559e0, 7.28447e-1, 7.72797e-1, 6.91427e-1, 6.17103e-1, 6.42244e-1, 6.12742e-1, 5.74352e-1, 1.01341e0, 5.91582e-1, 4.99417e-1, 4.76541e-1, 7.14012e-1, 5.05015e-1, 3.26853e-1, 1.29583e0, 3.72914e-1, 7.58457e-1, 2.05860e-1, 6.32870e-1, 1.99504e-1, 3.59164e-1, 3.67412e-1, 1.65377e0, 1.28834e-1, 9.82392e-2, 7.31311e-2, 9.18873e-2, 7.38087e-2],
        scores: [1.20618e0, 1.20410e0, 1.20048e0, 1.19431e0, 1.18413e0, 1.16831e0, 1.14611e0, 1.11999e0, 1.09672e0, 1.08165e0, 1.07150e0, 1.06079e0, 1.04548e0, 1.01934e0, 9.68204e-1, 8.58454e-1, 6.02638e-1, 2.54389e-1, 1.38958e-1, 4.03810e-1, 1.09230e0, 5.55482e0, 1.97934e1, 3.41849e1, 6.38010e1, 9.78631e1, 1.85076e2, 4.24462e2, 4.96087e2] },
    Raced { member: Member::Sign, k: -7, rho: 1.0, finals: [5.81947e-2, 7.21876e-2, 4.86899e-2],
        curve: [1.11852e0, 1.08369e0, 1.10786e0, 1.03139e0, 1.09137e0, 9.62365e-1, 9.21682e-1, 8.74121e-1, 8.57772e-1, 7.50247e-1, 7.43114e-1, 6.77690e-1, 6.43917e-1, 6.34052e-1, 5.85638e-1, 5.49809e-1, 5.23248e-1, 5.57987e-1, 4.37769e-1, 4.26256e-1, 4.27560e-1, 3.99464e-1, 3.31787e-1, 3.32142e-1, 2.54864e-1, 3.13238e-1, 2.33466e-1, 4.30386e-1, 1.73606e-1, 2.36974e-1, 3.06234e-1, 1.15519e-1, 1.42589e-1, 1.01380e-1, 9.92091e-2, 1.92562e-1, 6.75450e-2, 8.85315e-2, 2.86539e-1, 1.01559e-1, 8.89243e-2, 1.24029e-1, 7.02815e-2, 7.25384e-2, 1.20946e-1, 2.01146e-1, 2.91414e-2, 9.06260e-2, 1.15650e-2, 2.54003e-2, 1.34241e-1],
        scores: [1.16977e0, 1.14527e0, 1.11347e0, 1.08495e0, 1.06959e0, 1.05796e0, 1.04021e0, 1.00879e0, 9.57780e-1, 8.70381e-1, 6.92879e-1, 3.65102e-1, 1.14355e-1, 5.96907e-2, 2.37622e-1, 4.55949e-1, 6.55877e-1, 1.27950e0, 2.25050e0, 4.74873e0, 1.21925e1, 1.94872e1, 3.35948e1, 6.29011e1, 1.32580e2, 1.67001e2, 3.31731e2, 5.79604e2, 1.07297e3] },
    Raced { member: Member::Signum, k: -10, rho: 1.0, finals: [3.30005e-2, 6.24494e-3, 7.97967e-3],
        curve: [1.11852e0, 1.07523e0, 1.05608e0, 1.03058e0, 1.03244e0, 1.03259e0, 9.76142e-1, 9.68280e-1, 9.57349e-1, 9.04294e-1, 8.75955e-1, 8.31637e-1, 7.70167e-1, 7.93138e-1, 7.79879e-1, 6.78353e-1, 6.85136e-1, 6.32763e-1, 6.32271e-1, 5.39276e-1, 4.69846e-1, 4.26004e-1, 4.34007e-1, 3.72306e-1, 3.40520e-1, 3.08669e-1, 2.88758e-1, 2.75771e-1, 2.24021e-1, 1.84223e-1, 1.37734e-1, 1.81003e-1, 1.53029e-1, 1.00855e-1, 1.30356e-1, 2.05367e-1, 1.48104e-1, 4.25653e-2, 4.34853e-2, 3.56294e-2, 7.63991e-2, 5.59465e-2, 1.93513e-2, 1.44064e-2, 1.99652e-2, 9.00113e-3, 9.71409e-3, 6.12324e-2, 5.65506e-2, 2.20327e-2, 1.54729e-2],
        scores: [1.15118e0, 1.11838e0, 1.08026e0, 1.05364e0, 1.02573e0, 9.75888e-1, 8.87004e-1, 7.24981e-1, 4.34809e-1, 7.72647e-2, 1.57417e-2, 4.22286e-2, 1.35183e-1, 4.80493e-1, 1.27259e0, 1.65967e0, 2.16145e0, 3.62601e0, 5.85579e0, 1.13885e1, 1.94219e1, 3.15836e1, 4.99215e1, 1.21005e2, 1.94840e2, 2.74839e2, 5.50838e2, 1.01279e3, 1.76003e3] },
    Raced { member: Member::Lion, k: -10, rho: 1.0, finals: [2.43865e-2, 6.60475e-3, 2.77709e-2],
        curve: [1.11852e0, 1.06703e0, 1.05673e0, 1.03404e0, 1.02701e0, 9.97183e-1, 9.69798e-1, 9.43051e-1, 9.15568e-1, 8.87991e-1, 8.60150e-1, 8.36324e-1, 7.71695e-1, 7.63812e-1, 7.09669e-1, 6.54561e-1, 6.16597e-1, 5.68576e-1, 5.06354e-1, 4.42496e-1, 3.84668e-1, 3.44370e-1, 3.02857e-1, 2.53159e-1, 2.02715e-1, 1.69881e-1, 1.30555e-1, 1.01771e-1, 7.13803e-2, 4.98176e-2, 3.49235e-2, 3.04107e-2, 1.86943e-2, 1.00060e-2, 5.96903e-3, 3.58857e-3, 2.58033e-3, 1.87766e-3, 2.28174e-3, 1.55171e-2, 5.69040e-3, 3.91610e-3, 7.79412e-3, 1.32420e-2, 2.36374e-2, 5.01945e-3, 4.56371e-3, 6.77707e-2, 6.60625e-3, 2.37098e-2, 1.92821e-2],
        scores: [1.15089e0, 1.11798e0, 1.07977e0, 1.05312e0, 1.02443e0, 9.74433e-1, 8.86236e-1, 7.11749e-1, 3.30357e-1, 2.47661e-2, 1.95874e-2, 8.77544e-2, 3.11594e-1, 8.28331e-1, 1.11904e0, 1.31898e0, 1.50893e0, 2.39462e0, 2.84822e0, 6.55251e0, 1.28950e1, 1.83756e1, 4.42488e1, 5.49380e1, 1.09485e2, 1.86554e2, 3.24435e2, 6.01497e2, 1.00051e3] },
    Raced { member: Member::Adam, k: -7, rho: 1.0, finals: [6.67915e-3, 1.18452e-2, 3.76509e-3],
        curve: [1.11852e0, 1.07479e0, 1.04787e0, 1.01724e0, 1.00780e0, 9.97029e-1, 9.43261e-1, 9.09971e-1, 8.95711e-1, 8.01660e-1, 7.52695e-1, 6.54728e-1, 5.74386e-1, 5.06891e-1, 4.26952e-1, 3.46939e-1, 2.95182e-1, 4.18944e-1, 4.58392e-1, 1.98572e-1, 2.20691e-1, 2.32365e-1, 1.51475e-1, 9.98828e-2, 1.10888e-1, 8.69040e-2, 6.77989e-2, 5.92473e-2, 5.69352e-2, 7.19067e-2, 8.75920e-2, 7.22989e-2, 4.56884e-2, 3.33613e-2, 3.93912e-2, 4.01599e-2, 3.38314e-2, 2.51293e-2, 1.69835e-2, 1.46750e-2, 1.25442e-2, 1.18590e-2, 1.02173e-2, 9.43774e-3, 8.62713e-3, 8.05003e-3, 7.55385e-3, 7.02603e-3, 6.64055e-3, 6.26410e-3, 5.91120e-3],
        scores: [1.16976e0, 1.14558e0, 1.11392e0, 1.08433e0, 1.06743e0, 1.05408e0, 1.03222e0, 9.91674e-1, 9.18859e-1, 7.75916e-1, 4.87521e-1, 1.64688e-1, 2.46736e-2, 7.42982e-3, 4.85966e-2, 5.11482e-1, 1.11007e0, 1.19156e0, 1.38695e0, 1.90320e0, 2.33780e0, 4.29431e0, 9.33093e0, 1.22855e1, 3.40194e1, 4.83352e1, 7.76061e1, 2.24443e2, 3.81490e2] },
    Raced { member: Member::Ortho, k: -6, rho: 1.0, finals: [1.59713e-2, 3.63869e-2, 8.01464e-3],
        curve: [1.11852e0, 1.07251e0, 1.03748e0, 1.01228e0, 9.91816e-1, 9.46887e-1, 8.94634e-1, 8.44965e-1, 7.82384e-1, 7.13557e-1, 6.36799e-1, 5.66034e-1, 4.78073e-1, 4.16939e-1, 3.59282e-1, 2.88637e-1, 2.70507e-1, 2.07641e-1, 2.13648e-1, 1.62142e-1, 1.47732e-1, 1.12922e-1, 9.91603e-2, 8.86194e-2, 6.96420e-2, 6.73763e-2, 5.22854e-2, 5.41703e-2, 5.44078e-2, 4.83628e-2, 2.97740e-2, 3.14530e-2, 3.20819e-2, 2.38616e-2, 1.90698e-2, 2.09608e-2, 4.47789e-2, 4.35154e-2, 2.15985e-2, 2.13235e-2, 1.32623e-2, 1.31334e-2, 1.24275e-2, 2.41973e-2, 1.05722e-2, 1.06453e-2, 1.99200e-2, 1.18075e-2, 2.00962e-2, 1.84475e-2, 9.58514e-3],
        scores: [5.29872e-1, 5.29241e-1, 5.28115e-1, 5.26103e-1, 5.22482e-1, 5.15874e-1, 5.03475e-1, 4.79540e-1, 4.34432e-1, 3.55573e-1, 2.39587e-1, 1.11694e-1, 4.60032e-2, 2.23144e-2, 2.01243e-2, 5.49859e-2, 2.45165e-2, 3.58965e-2, 7.14896e-2, 1.51439e-1, 1.61440e-1, 5.41441e-1, 7.91350e-1, 1.01123e0, 9.74780e-1, 9.63056e-1, 9.73353e-1, 9.80383e-1, 1.00172e0] },
    Raced { member: Member::Spectral, k: 4, rho: 1.0, finals: [4.66426e-3, 8.35248e-3, 1.14727e-2],
        curve: [1.11852e0, 1.07186e0, 1.05654e0, 1.02836e0, 1.04550e0, 9.90424e-1, 9.28231e-1, 8.94104e-1, 8.99808e-1, 7.20499e-1, 6.40006e-1, 5.38910e-1, 6.95034e-1, 4.17138e-1, 3.73673e-1, 2.61180e-1, 2.60434e-1, 2.07672e-1, 1.74135e-1, 3.40622e-1, 1.69591e-1, 8.03979e-2, 9.15534e-2, 1.79855e-1, 7.33391e-2, 6.52528e-2, 6.80977e-2, 3.76533e-2, 8.18406e-2, 3.02020e-2, 2.65941e-2, 2.45133e-2, 1.56244e-2, 1.26201e-2, 1.49602e-2, 1.10777e-2, 1.24339e-2, 9.04057e-3, 8.11553e-3, 1.06041e-2, 7.14339e-3, 7.12853e-3, 8.94564e-3, 6.04950e-3, 5.39462e-3, 5.16372e-3, 5.23301e-3, 4.87577e-3, 4.49740e-3, 4.52264e-3, 4.19250e-3],
        scores: [1.04975e0, 1.04975e0, 1.04975e0, 1.04974e0, 1.04974e0, 1.04972e0, 1.04970e0, 1.04967e0, 1.04962e0, 1.04955e0, 1.04950e0, 1.04925e0, 1.04812e0, 1.04613e0, 1.04286e0, 1.03720e0, 1.02464e0, 1.01242e0, 9.82986e-1, 9.23654e-1, 7.70438e-1, 3.84285e-1, 8.01755e-2, 2.46230e-2, 8.16314e-3, 1.71055e-1, 6.74150e-1, 2.03119e0, 5.61604e0] },
    Raced { member: Member::Muon, k: -9, rho: 1.0, finals: [2.98226e-3, 1.21784e-3, 1.85739e-3],
        curve: [1.11852e0, 1.08196e0, 1.05253e0, 1.03883e0, 1.02469e0, 1.00193e0, 9.69969e-1, 9.73928e-1, 9.31090e-1, 8.77725e-1, 8.45649e-1, 8.22302e-1, 7.57139e-1, 7.51592e-1, 6.83330e-1, 6.17760e-1, 5.70674e-1, 5.53817e-1, 4.69425e-1, 4.06834e-1, 3.55519e-1, 3.12780e-1, 2.76736e-1, 2.48312e-1, 2.03419e-1, 1.86690e-1, 1.47486e-1, 1.31397e-1, 1.07790e-1, 9.79939e-2, 7.88654e-2, 7.59521e-2, 6.41929e-2, 5.31506e-2, 4.54370e-2, 3.39489e-2, 3.06816e-2, 2.24816e-2, 2.03893e-2, 1.64370e-2, 1.37758e-2, 1.05377e-2, 8.52721e-3, 6.94693e-3, 5.82634e-3, 4.82869e-3, 4.29529e-3, 3.36268e-3, 2.75005e-3, 2.44721e-3, 2.05609e-3],
        scores: [5.27397e-1, 5.24812e-1, 5.20148e-1, 5.11628e-1, 4.95919e-1, 4.67742e-1, 4.19091e-1, 3.33721e-1, 2.16361e-1, 8.41464e-2, 1.71095e-2, 2.01917e-3, 5.38606e-3, 2.49061e-2, 2.85736e-2, 6.18919e-2, 1.96876e-1, 6.96744e-1, 9.67166e-1, 1.01235e0, 1.00018e0, 9.99382e-1, 9.71009e-1, 9.59610e-1, 9.51187e-1, 9.31464e-1, 9.77936e-1, 9.07673e-1, 9.22032e-1] },
    Raced { member: Member::OSgdm, k: -10, rho: 1.0, finals: [4.10504e-2, 8.50404e-3, 3.95549e-4],
        curve: [1.11852e0, 1.07696e0, 1.04073e0, 1.01886e0, 9.92058e-1, 9.68953e-1, 9.22440e-1, 8.90749e-1, 8.35789e-1, 7.74249e-1, 7.17167e-1, 6.93765e-1, 5.88006e-1, 5.25486e-1, 4.85215e-1, 4.09572e-1, 3.86726e-1, 3.38512e-1, 2.86861e-1, 2.89760e-1, 2.10117e-1, 1.87013e-1, 1.76179e-1, 1.85820e-1, 1.49355e-1, 1.21001e-1, 1.12043e-1, 1.09124e-1, 9.42611e-2, 8.63532e-2, 9.47588e-2, 9.54530e-2, 6.50107e-2, 6.75265e-2, 5.84208e-2, 6.51273e-2, 6.52925e-2, 4.89015e-2, 7.71240e-2, 4.66184e-2, 5.06457e-2, 4.57458e-2, 5.59120e-2, 8.21350e-2, 4.70011e-2, 5.77564e-2, 4.52270e-2, 3.98967e-2, 3.80995e-2, 4.04249e-2, 4.16037e-2],
        scores: [1.04580e0, 1.04185e0, 1.03535e0, 1.01675e0, 9.93038e-1, 9.52894e-1, 8.62471e-1, 6.26642e-1, 2.54772e-1, 5.10257e-2, 1.66500e-2, 2.40567e-1, 2.14708e-1, 3.71598e-1, 3.83719e-1, 6.72348e-1, 1.06331e0, 1.52360e0, 2.23800e0, 3.39966e0, 5.19245e0, 1.00212e1, 1.50904e1, 3.46721e1, 5.46549e1, 9.21549e1, 1.85452e2, 2.65373e2, 6.23224e2] },
    Raced { member: Member::Shampoo, k: 0, rho: 1.0, finals: [8.21045e-3, 1.54379e-2, 6.95479e-3],
        curve: [1.11852e0, 1.06002e0, 1.00771e0, 9.14481e-1, 8.60414e-1, 5.79794e-1, 5.21411e-1, 4.85945e-1, 3.64604e-1, 2.94215e-1, 2.87404e-1, 2.20518e-1, 2.06461e-1, 1.47552e-1, 1.63224e-1, 1.13430e-1, 1.78638e-1, 7.92561e-2, 1.08251e-1, 5.68002e-2, 5.63276e-2, 6.03843e-2, 4.92834e-2, 3.92651e-2, 4.44394e-2, 2.78718e-2, 2.48072e-2, 2.20576e-2, 2.15124e-2, 1.93908e-2, 1.88486e-2, 1.72858e-2, 1.68943e-2, 1.51030e-2, 1.44454e-2, 1.37291e-2, 1.29932e-2, 1.25091e-2, 1.20532e-2, 1.13958e-2, 1.10252e-2, 1.06460e-2, 1.01573e-2, 9.70389e-3, 9.37432e-3, 9.06233e-3, 8.76584e-3, 8.47821e-3, 8.15871e-3, 7.97147e-3, 7.67803e-3],
        scores: [1.04975e0, 1.04974e0, 1.04975e0, 1.04978e0, 1.04989e0, 1.05004e0, 1.04963e0, 1.04878e0, 1.04740e0, 1.04435e0, 1.03603e0, 1.02993e0, 1.01454e0, 9.87542e-1, 9.33438e-1, 8.00998e-1, 5.11127e-1, 1.65126e-1, 4.11886e-2, 1.79246e-2, 1.02010e-2, 2.25787e-2, 2.57361e-1, 9.38659e-1, 1.06933e0, 1.23520e0, 1.53865e0, 2.43802e0, 4.34058e0] },
    Raced { member: Member::Soap, k: -8, rho: 1.0, finals: [8.40093e-7, 1.99178e-7, 6.46150e-8],
        curve: [1.11852e0, 1.05803e0, 1.02632e0, 9.97771e-1, 9.60684e-1, 9.27574e-1, 8.61286e-1, 8.06797e-1, 7.56112e-1, 6.64491e-1, 5.60245e-1, 4.74381e-1, 3.71961e-1, 2.97671e-1, 2.41957e-1, 1.84150e-1, 1.47613e-1, 1.11835e-1, 7.57697e-2, 6.42393e-2, 4.14469e-2, 3.03383e-2, 2.12677e-2, 1.49930e-2, 1.04860e-2, 7.15658e-3, 4.87564e-3, 3.44642e-3, 2.31860e-3, 1.58650e-3, 1.10890e-3, 7.82484e-4, 4.97304e-4, 3.52763e-4, 2.15822e-4, 1.41456e-4, 9.28399e-5, 6.05533e-5, 3.96298e-5, 2.63610e-5, 1.73524e-5, 1.16554e-5, 7.99737e-6, 5.41172e-6, 3.63516e-6, 2.40771e-6, 1.57139e-6, 1.09761e-6, 7.19719e-7, 4.85378e-7, 3.26371e-7],
        scores: [1.05049e0, 1.04975e0, 1.04893e0, 1.04718e0, 1.03749e0, 1.02345e0, 9.77015e-1, 9.03294e-1, 7.38016e-1, 3.75762e-1, 5.00504e-2, 4.70048e-4, 3.67962e-7, 2.11035e-3, 8.77155e-2, 1.08303e-2, 1.40593e-1, 4.33469e-1, 1.03319e0, 1.35971e0, 1.60335e0, 2.53480e0, 3.41574e0, 4.96216e0, 6.57218e0, 6.35489e0, 8.06982e0, 6.90260e0, 1.09871e0] },
    Raced { member: Member::NorMuon, k: -8, rho: 1.0, finals: [3.25985e-3, 5.86526e-3, 2.26367e-3],
        curve: [1.11852e0, 1.08042e0, 1.04844e0, 1.03349e0, 1.01355e0, 9.91810e-1, 9.53632e-1, 9.47331e-1, 9.06459e-1, 8.46625e-1, 8.13711e-1, 7.73158e-1, 7.12885e-1, 7.13462e-1, 6.12283e-1, 5.25512e-1, 5.10880e-1, 4.48840e-1, 4.30510e-1, 3.31026e-1, 3.15442e-1, 2.56004e-1, 2.17903e-1, 1.85783e-1, 1.54615e-1, 1.20961e-1, 9.66575e-2, 7.83153e-2, 6.32684e-2, 6.43571e-2, 4.62595e-2, 4.87460e-2, 4.48409e-2, 3.06572e-2, 2.46583e-2, 1.79427e-2, 1.99969e-2, 1.38961e-2, 1.03852e-2, 8.74866e-3, 9.54833e-3, 9.64996e-3, 8.78931e-3, 6.49102e-3, 6.02166e-3, 4.67556e-3, 4.78313e-3, 3.66620e-3, 2.76909e-3, 2.70361e-3, 2.37721e-3],
        scores: [5.28086e-1, 5.26039e-1, 5.22335e-1, 5.15551e-1, 5.03040e-1, 4.80885e-1, 4.42804e-1, 3.75009e-1, 2.78623e-1, 1.48874e-1, 5.44633e-2, 1.27976e-2, 3.79626e-3, 1.97888e-2, 8.50801e-2, 6.83318e-2, 2.48739e-1, 3.59184e-1, 6.89238e-1, 8.81919e-1, 9.59695e-1, 1.02963e0, 1.05663e0, 1.05157e0, 1.09185e0, 1.09704e0, f64::INFINITY, f64::INFINITY, f64::INFINITY] },
    Raced { member: Member::AdaMuon, k: -8, rho: 1.0, finals: [4.39181e-2, 6.24440e-3, 5.02288e-3],
        curve: [1.11852e0, 1.08406e0, 1.05406e0, 1.04218e0, 1.02252e0, 1.00169e0, 9.64518e-1, 9.62330e-1, 9.24705e-1, 8.72761e-1, 8.38292e-1, 8.00407e-1, 7.41939e-1, 7.31622e-1, 6.45345e-1, 5.55887e-1, 5.36463e-1, 4.88419e-1, 5.08204e-1, 3.80185e-1, 3.54454e-1, 3.01046e-1, 2.71123e-1, 2.43303e-1, 2.09452e-1, 1.92886e-1, 1.35283e-1, 1.30011e-1, 9.95162e-2, 1.10662e-1, 1.00642e-1, 1.12242e-1, 7.27860e-2, 6.30340e-2, 5.15458e-2, 4.95759e-2, 4.30674e-2, 6.73313e-2, 3.42129e-2, 2.86043e-2, 1.78956e-2, 1.59697e-2, 1.59025e-2, 1.60845e-2, 7.58650e-2, 1.12403e-1, 4.01924e-2, 6.06109e-2, 2.48039e-2, 5.71686e-2, 3.68147e-2],
        scores: [5.28250e-1, 5.26336e-1, 5.22855e-1, 5.16445e-1, 5.04847e-1, 4.84221e-1, 4.47597e-1, 3.90874e-1, 3.11181e-1, 1.97616e-1, 6.70529e-2, 4.31571e-2, 1.83951e-2, 2.26402e-2, 4.78279e-2, 7.54658e-2, 1.22369e-1, 5.73272e-1, 9.53348e-1, 9.84915e-1, 1.04578e0, 1.01803e0, 1.02260e0, 9.99071e-1, 1.05740e0, 1.06383e0, 1.05519e0, 1.03434e0, 1.05513e0] },
    Raced { member: Member::Scion, k: -16, rho: 64.0, finals: [3.63887e-3, 1.04228e-2, 3.84999e-3],
        curve: [1.11852e0, 1.08255e0, 1.05124e0, 1.03593e0, 1.01880e0, 9.99498e-1, 9.62242e-1, 9.65869e-1, 9.27353e-1, 8.64634e-1, 8.29984e-1, 7.87686e-1, 7.16480e-1, 6.80454e-1, 6.19170e-1, 5.65934e-1, 4.95520e-1, 4.72485e-1, 3.98043e-1, 3.12304e-1, 2.67166e-1, 2.31148e-1, 1.94783e-1, 1.80874e-1, 1.40412e-1, 1.45402e-1, 1.08258e-1, 8.58915e-2, 7.44761e-2, 6.03594e-2, 4.50301e-2, 4.35679e-2, 3.36193e-2, 2.67538e-2, 2.10299e-2, 1.85801e-2, 1.67095e-2, 1.44968e-2, 1.36168e-2, 9.68085e-3, 7.53599e-3, 7.45223e-3, 5.30301e-3, 5.66199e-3, 4.13702e-3, 3.59932e-3, 6.88707e-3, 3.14186e-3, 2.51374e-3, 2.79295e-3, 2.85874e-3],
        scores: [5.84999e-1, 3.64223e-1, 1.01712e-1, 1.56600e-2, 5.97054e-3, 2.22655e-2, 7.54380e-2, 1.47863e-1, 4.28440e-1, 8.52114e-1, 1.05971e0, 1.07006e0, 1.07229e0, 1.08153e0, 1.07761e0, 1.09091e0, 1.10645e0, 1.10820e0, 1.11584e0, 1.11719e0, 1.11210e0, 1.12538e0, f64::INFINITY, f64::INFINITY, f64::INFINITY, f64::INFINITY, f64::INFINITY, f64::INFINITY, f64::INFINITY] },
];

fn grid_rate(k: i32) -> f64 {
    10f64.powf(k as f64 / 4.0)
}

/// The rule a member races with at rate `lr` (and, for Scion, radius `rho`): its
/// published form (the reference's polynomial where it has one). A matrix rule's other
/// groups (the biases; for the Muon-like rules also the input layer and the read-out)
/// all take Adam with Kingma and Ba's defaults at the rate 10^(AUX_K/4), whatever the
/// rule's own rate: so the race compares the matrix rules, not how their fall-backs
/// were set.
#[cfg(test)]
fn racing(m: Member, lr: f64, rho: f64) -> (Rule, Hyper) {
    let (rule, h) = m.rule(false);
    if !rule.is_matrix_rule() {
        return (rule, Hyper { lr, ..h });
    }
    let aux = optim::Aux::adam(grid_rate(AUX_K));
    let scale = if m == Member::Scion { optim::Scale::Scion(rho) } else { h.scale };
    (rule, Hyper { lr, aux, scale, ..h })
}

/// One race: member m at learning rate lr from the seed's weights, the whole data's
/// loss at the start and every `RACE_EVERY` steps; a loss that is not a number (the
/// run has blown up) is infinite, and so is everything after it.
#[cfg(test)]
fn race(m: Member, lr: f64, rho: f64, seed: u64, problem: &Problem) -> Vec<f64> {
    let mut model = Model::mlp(&spec(), &mut Rng::new(seed));
    let (rule, hyper) = racing(m, lr, rho);
    let mut opt = Optimizer::new(&model, rule, hyper);
    let mut tape = Tape::new();
    let mut batches = Sampler::new(problem.len(), RACE_BATCH, Draw::WithoutReplacement, Rng::new(seed.wrapping_add(200)));
    let mut out = Vec::with_capacity(RACE_POINTS);
    out.push(problem.loss(&model, &mut tape));
    for t in 1..=RACE_STEPS {
        let idx = batches.deal().to_vec();
        problem.grad(&mut model, &mut tape, &idx);
        opt.step(&mut model, lr);
        if t % RACE_EVERY == 0 {
            let l = problem.loss(&model, &mut tape);
            if !l.is_finite() {
                out.resize(RACE_POINTS, f64::INFINITY);
                break;
            }
            out.push(l);
        }
    }
    out
}

/// What the search measures of a race: the mean of its last five noted losses (steps
/// 920 to 1000); infinite if it blew up.
#[cfg(test)]
fn tail_score(curve: &[f64]) -> f64 {
    let tail = &curve[curve.len().saturating_sub(5)..];
    let s = tail.iter().sum::<f64>() / tail.len().max(1) as f64;
    if s.is_finite() {
        s
    } else {
        f64::INFINITY
    }
}

/// One member's race as the search left it.
#[derive(Clone, Copy, Debug)]
struct Raced {
    member: Member,
    /// The rate chosen, 10^(k/4).
    k: i32,
    /// Scion's radius chosen (1 for the others, which have none).
    rho: f64,
    /// Each seed's score at the chosen settings.
    finals: [f64; 3],
    /// The first seed's whole-data loss at the start and every `RACE_EVERY` steps.
    curve: [f64; RACE_POINTS],
    /// The mean score over the seeds at every rate of the grid (at the chosen radius).
    scores: [f64; GRID_N],
}

impl Raced {
    fn rate(&self) -> f64 {
        grid_rate(self.k)
    }

    fn mean(&self) -> f64 {
        self.finals.iter().sum::<f64>() / self.finals.len() as f64
    }
}

/// The race of member m, from the table.
fn raced(m: Member) -> Option<&'static Raced> {
    RACE.iter().find(|r| r.member == m)
}

/// The whole search: every member at every rate (Scion at every radius too) on every
/// seed; the settings chosen are those with the lowest mean score over the seeds.
/// Work spread over `threads` threads. (About ten seconds on 24 threads.)
#[cfg(test)]
fn search(threads: usize) -> (Vec<Raced>, [f64; RADII.len()]) {
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::Mutex;
    let p = problem();
    let ks: Vec<i32> = (GRID_LO..=GRID_HI).collect();
    // (member, radius, rate, seed)
    let mut jobs: Vec<(Member, usize, usize, usize)> = Vec::new();
    for m in RACERS {
        let radii = if m == Member::Scion { RADII.len() } else { 1 };
        for r in 0..radii {
            for k in 0..ks.len() {
                for s in 0..RACE_SEEDS.len() {
                    jobs.push((m, r, k, s));
                }
            }
        }
    }
    let results: Mutex<Vec<Vec<f64>>> = Mutex::new(vec![Vec::new(); jobs.len()]);
    let next = AtomicUsize::new(0);
    std::thread::scope(|sc| {
        for _ in 0..threads.max(1) {
            sc.spawn(|| loop {
                let j = next.fetch_add(1, Ordering::Relaxed);
                let Some(&(m, r, k, s)) = jobs.get(j) else { break };
                let curve = race(m, grid_rate(ks[k]), RADII[r], RACE_SEEDS[s], &p);
                if let Ok(mut all) = results.lock() {
                    all[j] = curve;
                }
            });
        }
    });
    let results = results.into_inner().unwrap_or_default();
    let curve_of = |m: Member, r: usize, k: usize, s: usize| -> &Vec<f64> {
        let j = jobs.iter().position(|&job| job == (m, r, k, s)).unwrap_or(0);
        &results[j]
    };
    let mut by_radius = [f64::INFINITY; RADII.len()];
    for (r, best) in by_radius.iter_mut().enumerate() {
        for k in 0..ks.len() {
            let mean = (0..RACE_SEEDS.len()).map(|s| tail_score(curve_of(Member::Scion, r, k, s))).sum::<f64>() / RACE_SEEDS.len() as f64;
            *best = best.min(mean);
        }
    }
    let raced = RACERS
        .iter()
        .map(|&m| {
            let radii = if m == Member::Scion { RADII.len() } else { 1 };
            let mean = |r: usize, k: usize| (0..RACE_SEEDS.len()).map(|s| tail_score(curve_of(m, r, k, s))).sum::<f64>() / RACE_SEEDS.len() as f64;
            let mut best = (0, 0);
            for r in 0..radii {
                for k in 0..ks.len() {
                    if mean(r, k) < mean(best.0, best.1) {
                        best = (r, k);
                    }
                }
            }
            let (r, k) = best;
            let mut finals = [0.0; 3];
            for (s, f) in finals.iter_mut().enumerate() {
                *f = tail_score(curve_of(m, r, k, s));
            }
            let mut curve = [0.0; RACE_POINTS];
            curve.copy_from_slice(curve_of(m, r, k, 0));
            let mut scores = [0.0; GRID_N];
            for (i, sc) in scores.iter_mut().enumerate() {
                *sc = mean(r, i);
            }
            Raced { member: m, k: ks[k], rho: RADII[r], finals, curve, scores }
        })
        .collect();
    (raced, by_radius)
}

// ----- how the members look ----------------------------------------------------------------

impl Member {
    /// Its colour: Muon's, Adam's and gradient descent's are the course's, the rest are
    /// shades of their family's (blue for those that keep U and V, orange for the sign
    /// rules, AdaGrad's green for Shampoo's, whose memory is AdaGrad's for matrices).
    fn color(self) -> Color {
        match self {
            Member::Gd => SGD,
            Member::Muon => MUON,
            Member::Ortho => MUON.mix(pal::WHITE, 0.45),
            Member::Spectral => MUON.mix(pal::PURPLE, 0.6),
            Member::OSgdm => MUON.mix(pal::CYAN, 0.35).mix(pal::WHITE, 0.15),
            Member::NorMuon => MUON.mix(pal::GREEN, 0.4),
            Member::AdaMuon => MUON.mix(ADAM, 0.4),
            Member::Scion => MUON.mix(pal::PINK, 0.45),
            Member::Adam => ADAM,
            Member::Sign => ADAM.mix(pal::WHITE, 0.45),
            Member::Signum => ADAM.mix(pal::YELLOW, 0.55),
            Member::Lion => ADAM.mix(pal::PINK, 0.5),
            Member::Shampoo => ADAGRAD,
            Member::Fresh => ADAGRAD.mix(pal::WHITE, 0.45),
            Member::Soap => ADAGRAD.mix(ADAM, 0.5),
        }
    }

    /// Its name in two short lines, for a small window.
    fn short(self) -> [&'static str; 2] {
        match self {
            Member::Gd => ["gradient", "descent"],
            Member::Sign => ["sign", "descent"],
            Member::Ortho => ["Muon, no", "momentum"],
            Member::Fresh => ["Shampoo,", "no memory"],
            Member::Spectral => ["spectral", "descent"],
            Member::OSgdm => ["Orthog.-", "SGDM"],
            Member::NorMuon => ["Nor-", "Muon"],
            Member::AdaMuon => ["Ada-", "Muon"],
            m => [m.name(), ""],
        }
    }

    /// Where it sits in the tree: (column, row), columns 0 to 5, rows 0 to 3.
    fn spot(self) -> (f32, f32) {
        match self {
            Member::Gd => (2.0, 0.0),
            Member::Ortho => (1.0, 1.0),
            Member::Fresh => (3.0, 1.0),
            Member::Sign => (4.5, 1.0),
            Member::Muon => (0.0, 2.0),
            Member::OSgdm => (1.0, 2.0),
            Member::Spectral => (2.0, 2.0),
            Member::Shampoo => (3.0, 2.0),
            Member::Adam => (4.0, 2.0),
            Member::Signum => (5.0, 2.0),
            Member::NorMuon => (0.0, 3.0),
            Member::AdaMuon => (1.0, 3.0),
            Member::Scion => (2.0, 3.0),
            Member::Soap => (3.0, 3.0),
            Member::Lion => (5.0, 3.0),
        }
    }

    /// What it steps along, in a line (G the present gradient, M a running average of
    /// the gradients).
    fn along(self) -> &'static str {
        match self {
            Member::Gd => r"$G$ itself",
            Member::Sign => r"the sign of each entry of $G$",
            Member::Signum => r"the sign of each entry of $M$",
            Member::Lion => r"the signs of a mix of $M$ and $G$",
            Member::Adam => r"$\hat M / \sqrt{\hat V}$, entry by entry",
            Member::Ortho => r"$UV^\top$ of $G$",
            Member::Fresh => r"$(GG^\top)^{-1/4}\,G\,(G^\top G)^{-1/4}$",
            Member::Spectral => r"$UV^\top$ of $G$, times $\|G\|_{\mathrm{nuc}}/mn$",
            Member::Muon => r"$UV^\top$ of a mix of $M$ and $G$",
            Member::OSgdm => r"a running sum of the $UV^\top$ of each $G$",
            Member::Shampoo => r"$L^{-1/4}\,G\,R^{-1/4}$, $L$ and $R$ summed over the history",
            Member::Soap => r"Adam's step in the eigenvectors of Shampoo's $L$ and $R$",
            Member::NorMuon => r"$UV^\top$ of $M$, each row divided by its running size",
            Member::AdaMuon => r"$UV^\top$ of the signs of $M$, each entry divided by its running size",
            Member::Scion => r"$UV^\top$ of $M$, with the weights shrunk towards a ball",
        }
    }
}

/// The edges of the tree: what turns one member into another, and where its words go
/// (how far along the edge, and on which side).
struct Edge {
    from: Member,
    to: Member,
    words: &'static str,
    along: f32,
    side: Side,
    look: EdgeLook,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Side {
    Left,
    Right,
    Above,
    Below,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum EdgeLook {
    Line,
    /// The same step: a double line.
    Same,
}

const EDGES: [Edge; 11] = [
    Edge { from: Member::Gd, to: Member::Ortho, words: r"every $\sigma$ set to 1", along: 0.5, side: Side::Left, look: EdgeLook::Line },
    Edge { from: Member::Gd, to: Member::Sign, words: "act on each entry instead", along: 0.5, side: Side::Above, look: EdgeLook::Line },
    Edge { from: Member::Ortho, to: Member::Fresh, words: "the same step", along: 0.5, side: Side::Below, look: EdgeLook::Same },
    Edge { from: Member::Ortho, to: Member::Muon, words: "momentum before", along: 0.5, side: Side::Left, look: EdgeLook::Line },
    Edge { from: Member::Ortho, to: Member::OSgdm, words: "momentum after", along: 0.35, side: Side::Right, look: EdgeLook::Line },
    Edge { from: Member::Ortho, to: Member::Spectral, words: r"times $\|G\|_{\mathrm{nuc}}$", along: 0.75, side: Side::Right, look: EdgeLook::Line },
    Edge { from: Member::Fresh, to: Member::Shampoo, words: "remember the past", along: 0.3, side: Side::Right, look: EdgeLook::Line },
    Edge { from: Member::Shampoo, to: Member::Soap, words: "Adam in its eigenbasis", along: 0.5, side: Side::Right, look: EdgeLook::Line },
    Edge { from: Member::Sign, to: Member::Adam, words: "divide by averages", along: 0.72, side: Side::Left, look: EdgeLook::Line },
    Edge { from: Member::Sign, to: Member::Signum, words: "momentum before", along: 0.5, side: Side::Right, look: EdgeLook::Line },
    Edge { from: Member::Signum, to: Member::Lion, words: "two averages", along: 0.5, side: Side::Right, look: EdgeLook::Line },
];

/// Muon's variants, hung from it on a comb.
const VARIANTS: [Member; 3] = [Member::NorMuon, Member::AdaMuon, Member::Scion];

/// An angle as the scene writes it: one decimal below 10°, none above; "?" for none.
fn angle(cos: f64) -> String {
    let d = degrees(cos);
    if !d.is_finite() {
        return "?".to_string();
    }
    if d < 9.95 {
        format!("{}°", num(d, 1))
    } else {
        format!("{}°", num(d, 0))
    }
}

/// A number with three significant digits, a true minus sign, and the form 1.2e-5
/// between dollars (which the typesetting writes as 1.2 × 10⁻⁵) below 0.001.
fn sig(v: f64) -> String {
    sig_n(v, 3)
}

/// The same with `digits` significant digits (at least 1).
fn sig_n(v: f64, digits: i32) -> String {
    if !v.is_finite() {
        return "?".to_string();
    }
    if v == 0.0 {
        return "0".to_string();
    }
    let a = v.abs();
    let digits = digits.max(1);
    if a < 1e-3 {
        let e = a.log10().floor() as i32;
        let m = v / 10f64.powi(e);
        return format!("${}e{}$", num(m, (digits - 1) as usize), e);
    }
    let decimals = (digits - 1 - a.log10().floor() as i32).clamp(0, 6) as usize;
    num(v, decimals)
}

// ----- the experiments ---------------------------------------------------------------------

/// How the race chart reads: the loss against the step, or the search's score against
/// the learning rate.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum RaceView {
    Steps,
    Rates,
}

struct Trial {
    button: &'static str,
    caption: &'static str,
}

const TRIALS: [Trial; 6] = [
    Trial {
        button: "one gradient, three answers",
        caption: r"The gradient of all 192 points, with no history yet. Its singular values (grey) fall from 1 to $7.8 \times 10^{-6}$ of the largest: a step along it moves the layer almost only along two directions. The rules give three answers. Those that keep $U$ and $V$ set every singular value to 1 and step along $U V^\top$: 6 boxes at 0.0° from Muon. The sign rules step along the sign of each entry (with one gradient, Signum, Lion and Adam are sign descent): 72°. Gradient descent steps along $G$ itself: 63°. Shampoo is 5.0° off only through its damping $\epsilon = 10^{-12}$; AdaMuon, which takes the sign first, is 74° off; SOAP takes no step on its first call.",
    },
    Trial {
        button: "the sign of each entry",
        caption: r"The same gradient, sign descent lit. Its step is not orthogonal: its singular values still fall, to 0.078 of the largest, and 3 of the 16 are below a millionth of it. Its stable rank is 2.28, against the gradient's 1.86 and $U V^\top$'s 16: the sign spreads the step a little, nowhere near evenly. It is 50° from the gradient's step and 72° from Muon's. Each is a steepest step for its own ruler: the sign when a step's size is its largest entry, $U V^\top$ when it is the most it stretches a vector (Bernstein and Newhouse, 2024).",
    },
    Trial {
        button: "Shampoo without memory",
        caption: r"Shampoo with no memory and no damping builds its two matrices from this gradient alone, and for a matrix of full rank $(G G^\top)^{-1/4}\, G\, (G^\top G)^{-1/4} = U V^\top$: every entry of its step matches the exact $U V^\top$ to within $5.9 \times 10^{-8}$ of the largest (the roots come from $G G^\top$, whose eigenvalues span ten decades). That is Muon's step with the momentum off, orthogonalised exactly, as Bernstein and Newhouse (2024) pointed out. Muon's own polynomial (scene muon) lands 32° away: it raises the 10 largest singular values to between 0.73 and 1.16 and leaves the 6 smallest, each under a thousandth of the gradient's size, between 0.003 and 0.33.",
    },
    Trial {
        button: "Shampoo with memory",
        caption: r"Batches of 16 now, 64 of them in turn at the same weights. Shampoo adds up every $G G^\top$ and $G^\top G$, and its step stops being flat: after 64, its stable rank is 1.92 (Muon's is 16), and it lies 31° from the latest gradient and 74° from Muon's step. Its matrices sum over all the batches while its step follows this batch's gradient, much of whose shape is left. With all 192 points in every batch the gradient never changes: Shampoo keeps the direction $U V^\top$ (0.2° from it after 64), and only its length falls, as $1/\sqrt{t}$: to $1/8$ at $t = 64$.",
    },
    Trial {
        button: "momentum before or after",
        caption: r"Batches of 16, 64 of them, Orthogonal-SGDM lit. Muon averages the gradients, then orthogonalises; Orthogonal-SGDM (Tuddenham and co-authors, 2022) orthogonalises each, then averages. From the orthogonal step of the whole data's gradient, Muon's step is 31° away, Orthogonal-SGDM's 61°, and Muon's without momentum 78°. Orthogonalising first gives each batch's noise directions full size before the average; averaging first lets the noise partly cancel. With the same momentum for both (0.9, no Nesterov mix) Muon is 43° away, still nearer, and the order was the same for 8 sets of batches. Jordan and co-authors (2024) report that the momentum first works better in their runs.",
    },
    Trial {
        button: "the race",
        caption: r"Fourteen rules train this network on the spirals, SOAP lit: the same starting weights and the same 1000 batches of 16 for each, each at the rate a search over 29 chose (Scion over radii too; the panel says how). The loss over the last 80 steps, the mean of three seeds: SOAP $3.7 \times 10^{-7}$, Muon 0.0020, NorMuon 0.0038, Scion 0.0060, Adam 0.0074; Muon without momentum 0.020, ten times Muon's; gradient descent 0.14. One small task and one search: it ranks nothing beyond itself. In the NanoGPT speedrun's optimisation benchmark (a GPT-2-small-sized model), Muon without momentum needed 8225 steps to the target where tuned Muon needed 3250.",
    },
];

/// The scene opens as the first experiment sets it up.
const OPENING: usize = 0;

/// What experiment k sets: the stream, the member lit, the step of the stream shown
/// (played from the start when it is not the first), and the race's view.
fn trial(k: usize) -> (Setup, Member, usize, RaceView) {
    let all = Setup::base();
    let batches = Setup { batch: 16, ..all };
    match k {
        1 => (all, Member::Sign, 1, RaceView::Steps),
        2 => (all, Member::Fresh, 1, RaceView::Steps),
        3 => (batches, Member::Shampoo, T_MAX, RaceView::Steps),
        4 => (batches, Member::OSgdm, T_MAX, RaceView::Steps),
        5 => (all, Member::Soap, 1, RaceView::Steps),
        _ => (all, Member::Muon, 1, RaceView::Steps),
    }
}

/// How fast the history plays: steps of the stream a second.
const PLAY_RATE: f32 = 10.0;

// ----- the scene ---------------------------------------------------------------------------

pub struct MuonKin {
    setup: Setup,
    chosen: Member,
    /// The step of the stream shown (1 to T_MAX), and where the film is (it plays up to
    /// `target`).
    t: usize,
    target: usize,
    clock: f32,
    playing: bool,
    /// While the history plays, how far the film is between step `t` and the next (0
    /// to 1): the bars are drawn that far towards the next step's, so that they move
    /// evenly however the steps fall into frames.
    phase: f32,
    race_view: RaceView,
    problem: Problem,
    frozen: Model,
    stream: Stream,
    look: Look,
    /// The window last drawn in (a new one is drawn at once, without gliding).
    canvas: Option<Vec2>,
    placed: Vec<Placed>,
    labels_moving: bool,
}

/// What is drawn, gliding to where it is going.
#[derive(Clone, Debug, Default)]
struct Look {
    /// The bars: log₁₀ of the gradient's singular values (relative), then the chosen
    /// member's.
    bars: Glide,
    /// The race's curves: log₁₀ of each racer's points (or of its scores); and the
    /// chart's ranges.
    race: Glide,
    race_frame: Glide,
    moving: bool,
}

impl MuonKin {
    fn create() -> MuonKin {
        let problem = problem();
        let frozen = warmed(WARM, NET_SEED, &problem);
        let setup = Setup::base();
        let stream = Stream::new(setup, &problem, &frozen);
        let mut s = MuonKin {
            setup,
            chosen: Member::Muon,
            t: 1,
            target: 1,
            clock: 0.0,
            playing: false,
            phase: 0.0,
            race_view: RaceView::Steps,
            problem,
            frozen,
            stream,
            look: Look::default(),
            canvas: None,
            placed: Vec::new(),
            labels_moving: false,
        };
        s.set_up(OPENING);
        s
    }

    /// Set up experiment k.
    fn set_up(&mut self, k: usize) {
        let (setup, chosen, t, view) = trial(k);
        self.setup = setup;
        self.chosen = chosen;
        self.race_view = view;
        self.refresh();
        if t > 1 {
            self.play_to(t);
        } else {
            self.t = 1;
            self.target = 1;
            self.playing = false;
        }
    }

    /// Play the history from its start up to step `to`.
    fn play_to(&mut self, to: usize) {
        self.target = to.clamp(1, T_MAX);
        self.t = 1;
        self.clock = 0.0;
        self.playing = self.target > 1;
    }

    /// A stream for the present setup (a new one if it has changed).
    fn refresh(&mut self) {
        if self.stream.setup != self.setup {
            self.stream = Stream::new(self.setup, &self.problem, &self.frozen);
        }
        self.t = self.t.clamp(1, T_MAX);
    }

    /// The film and the stream's work, `dt` seconds of the scene's time on.
    fn advance(&mut self, dt: f32, budget: &Budget) {
        // the stream is worked out ahead of what is shown, as far as the budget goes
        let want = if self.playing { self.target } else { self.t };
        self.stream.work(want, budget);
        self.phase = 0.0;
        if self.playing {
            self.clock += dt.max(0.0);
            let at = 1.0 + self.clock * PLAY_RATE;
            let goal = at.floor() as usize;
            // (the film waits for the stream)
            let ready = self.stream.done().max(1);
            self.t = goal.min(self.target).min(ready);
            if self.t >= self.target {
                self.playing = false;
            } else if goal == self.t && self.t < ready {
                self.phase = (at - goal as f32).clamp(0.0, 1.0);
            } else {
                // (waiting for the stream: the clock waits too)
                self.clock = (self.t as f32 - 1.0) / PLAY_RATE;
            }
        }
    }

    /// The step shown, as far as the stream has got.
    fn shown(&self) -> usize {
        self.t.min(self.stream.done()).max(1)
    }

    fn step_shown(&self) -> Option<&Step> {
        self.stream.at(self.shown(), self.chosen)
    }
}

// ----- the layout --------------------------------------------------------------------------

/// Where everything goes in a window (logical pixels from its top-left corner).
#[derive(Clone, Copy, Debug, PartialEq)]
struct Layout {
    view: Vec2,
    narrow: bool,
    /// Text sizes: titles, small print, the boxes' names.
    text: f32,
    small: f32,
    name: f32,
    tree_head: Vec2,
    tree: Rect,
    /// The chosen member's lines, under the tree (nothing on a small window).
    about: Rect,
    /// The buttons: the five batch sizes, and play.
    buttons: [Rect; 6],
    spec_head: Vec2,
    spec: Rect,
    race_head: Vec2,
    race: Rect,
    race_buttons: [Rect; 2],
    /// Only the tree: the window is too small for the rest.
    compact: bool,
}

impl Layout {
    fn of(view: Vec2, inset: f32) -> Layout {
        let view = view.max(Vec2::splat(1.0));
        let narrow = view.x < 760.0;
        let (text, small, name) = if narrow { (12.0, 10.0, 9.5) } else { (13.5, 11.5, 11.0) };
        let side = if narrow { 8.0 } else { 20.0 };
        let top = inset.max(0.0) + if narrow { 6.0 } else { 12.0 };
        let r = |x0: f32, y0: f32, x1: f32, y1: f32| (Vec2::new(x0, y0), Vec2::new(x1.max(x0 + 1.0), y1.max(y0 + 1.0)));
        let none = r(0.0, 0.0, 0.0, 0.0);
        let head = text + 10.0;
        let mut lay = Layout {
            view,
            narrow,
            text,
            small,
            name,
            tree_head: Vec2::new(side, top + text),
            tree: none,
            about: none,
            buttons: [none; 6],
            spec_head: Vec2::ZERO,
            spec: none,
            race_head: Vec2::ZERO,
            race: none,
            race_buttons: [none; 2],
            compact: false,
        };
        let button_h = if narrow { 22.0 } else { 26.0 };
        let wide = view.x >= 900.0 && view.x >= 1.05 * view.y;
        if wide {
            let tw = ((view.x - 3.0 * side) * 0.57).floor();
            let x1 = side + tw;
            let (bh, header) = lay.box_size();
            let tree_h = (view.y - top - head - 200.0).min(3.0 * 150.0 + bh + 14.0 + header).max(160.0);
            lay.tree = r(side, top + head, x1, top + head + tree_h);
            lay.about = r(side, lay.tree.1.y + 22.0, x1, view.y - 14.0);
            let x0 = x1 + side;
            let x2 = view.x - side;
            let row = plot::even_row(6, x0, x2, top, button_h, 6.0);
            for (k, b) in row.into_iter().enumerate().take(6) {
                lay.buttons[k] = b;
            }
            let y = top + button_h + 14.0;
            lay.spec_head = Vec2::new(x0, y + text);
            let spec_top = y + head + 3.0 * (small + 4.0) + 6.0;
            let spec_h = ((view.y - spec_top) * 0.44).max(80.0);
            lay.spec = r(x0 + 44.0, spec_top, x2 - 6.0, spec_top + spec_h);
            let ry = lay.spec.1.y + 46.0;
            lay.race_head = Vec2::new(x0, ry + text);
            let rb = plot::even_row(2, x0, x0 + 230.0, ry + text + 8.0, button_h - 4.0, 6.0);
            lay.race_buttons = [rb[0], rb[1]];
            lay.race = r(x0 + 44.0, rb[0].1.y + 16.0, x2 - 6.0, view.y - 40.0);
            return lay;
        }
        let avail = view.y - top;
        if avail < 560.0 || view.x < 250.0 {
            lay.compact = true;
            lay.tree = r(side, top + head, view.x - side, view.y - 10.0);
            return lay;
        }
        let tree_h = (avail * 0.40).min(view.x * 0.95);
        lay.tree = r(side, top + head, view.x - side, top + head + tree_h);
        let by = lay.tree.1.y + 8.0;
        let row = plot::even_row(6, side, view.x - side, by, button_h, 4.0);
        for (k, b) in row.into_iter().enumerate().take(6) {
            lay.buttons[k] = b;
        }
        let y = by + button_h + 8.0;
        lay.spec_head = Vec2::new(side, y + text);
        let spec_top = y + head + 3.0 * (small + 4.0);
        let left = side + 36.0;
        let rest = view.y - spec_top;
        let spec_h = (rest * 0.42 - 30.0).max(50.0);
        lay.spec = r(left, spec_top, view.x - side - 8.0, spec_top + spec_h);
        let ry = lay.spec.1.y + 34.0;
        lay.race_head = Vec2::new(side, ry + text);
        let rb = plot::even_row(2, view.x - side - 120.0, view.x - side, ry - 4.0, button_h - 4.0, 4.0);
        lay.race_buttons = [rb[0], rb[1]];
        lay.race = r(left, ry + head + 4.0, view.x - side - 14.0, view.y - 30.0);
        lay
    }

    /// Where a member's box is centred: rows evenly from the top of the tree, with
    /// room at the bottom for the families' names.
    fn spot(&self, m: Member) -> Vec2 {
        let (c, row) = m.spot();
        let (min, max) = self.tree;
        let col = (max.x - min.x) / 6.0;
        let (bh, header) = self.box_size();
        let top = min.y + 0.5 * bh + 2.0;
        let pitch = (((max.y - min.y) - bh - 14.0 - header) / 3.0).max(1.0);
        Vec2::new(min.x + (c + 0.5) * col, top + row * pitch)
    }

    /// Whether the boxes fit in the tree at all (rows apart, columns wide enough for a
    /// short name).
    fn tree_fits(&self) -> bool {
        let (min, max) = self.tree;
        let (bh, header) = self.box_size();
        let pitch = ((max.y - min.y) - bh - 14.0 - header) / 3.0;
        pitch >= bh + 6.0 && (max.x - min.x) / 6.0 >= 44.0
    }

    /// A box's height, and the room under the first row for the families' names.
    fn box_size(&self) -> (f32, f32) {
        if self.narrow {
            (2.0 * self.name + 9.0, self.small + 6.0)
        } else {
            (self.name + self.small + 14.0, self.small + 10.0)
        }
    }

    /// The box of member m: its corners. (A box with no neighbour beside it in its row
    /// may be wider than a column.)
    fn box_of(&self, m: Member, width: f32) -> Rect {
        let c = self.spot(m);
        let (bh, _) = self.box_size();
        let col = (self.tree.1.x - self.tree.0.x) / 6.0;
        let room = match m {
            Member::Gd | Member::Ortho => 1.7 * col,
            Member::Fresh | Member::Sign => 1.4 * col,
            _ => col - 4.0,
        };
        let w = width.min(room).max(10.0);
        (c - Vec2::new(0.5 * w, 0.5 * bh), c + Vec2::new(0.5 * w, 0.5 * bh))
    }
}

// ----- what is drawn, gliding --------------------------------------------------------------

/// The bars' floor (log₁₀ of a relative singular value): below `RESOLVED`, a value is
/// drawn as a stub a little over it.
const FLOOR: f64 = -6.4;
const STUB: f64 = -6.25;

/// log₁₀ of a relative singular value as a bar shows it.
fn bar_height(v: f64) -> f64 {
    if v > 0.0 && v.is_finite() {
        v.log10().max(STUB)
    } else {
        STUB
    }
}

/// The race chart's points: for each racer, `RACE_POINTS` places across, each as (x,
/// y) between 0 and 1 of the chart's ranges, and whether it has a value. Against the
/// step: its noted losses. Against the rate: its scores, read off the grid at as many
/// rates spread evenly over it (straight between the grid's points in the logarithms).
struct RaceFrame {
    /// log₁₀ ranges across and up.
    x: [f64; 2],
    y: [f64; 2],
}

fn race_frame(view: RaceView) -> RaceFrame {
    let mut lo = f64::INFINITY;
    let mut hi = f64::NEG_INFINITY;
    for r in RACE.iter() {
        let values: &[f64] = match view {
            RaceView::Steps => &r.curve,
            RaceView::Rates => &r.scores,
        };
        for &v in values {
            if v.is_finite() && v > 0.0 {
                lo = lo.min(v.log10());
                hi = hi.max(v.log10());
            }
        }
    }
    if !(lo.is_finite() && hi.is_finite()) {
        (lo, hi) = (-3.0, 0.0);
    }
    match view {
        RaceView::Steps => RaceFrame { x: [0.0, RACE_STEPS as f64], y: [lo.floor() - 0.1, hi.min(0.4) + 0.12] },
        RaceView::Rates => RaceFrame { x: [GRID_LO as f64 / 4.0, GRID_HI as f64 / 4.0], y: [lo.floor() - 0.1, 0.6] },
    }
}

/// One racer's points, as (x, y) in [0, 1] of the frame, and where it has a value.
fn race_points(r: &Raced, view: RaceView, frame: &RaceFrame) -> (Vec<f32>, Vec<bool>) {
    let mut pts = Vec::with_capacity(2 * RACE_POINTS);
    let mut mask = Vec::with_capacity(RACE_POINTS);
    let ny = |v: f64| ((v - frame.y[0]) / (frame.y[1] - frame.y[0])) as f32;
    for i in 0..RACE_POINTS {
        let u = i as f64 / (RACE_POINTS - 1) as f64;
        let value = match view {
            RaceView::Steps => r.curve[i],
            RaceView::Rates => {
                let at = u * (GRID_N - 1) as f64;
                let (a, b) = (at.floor() as usize, (at.ceil() as usize).min(GRID_N - 1));
                let (va, vb) = (r.scores[a], r.scores[b]);
                if va.is_finite() && vb.is_finite() && va > 0.0 && vb > 0.0 {
                    let w = at - a as f64;
                    10f64.powf(va.log10() * (1.0 - w) + vb.log10() * w)
                } else {
                    f64::INFINITY
                }
            }
        };
        let ok = value.is_finite() && value > 0.0 && value.log10() <= frame.y[1];
        pts.push(u as f32);
        pts.push(if ok { ny(value.log10()) } else { f32::NAN });
        mask.push(ok);
    }
    // (a point with no value glides as its neighbour does, and is not drawn)
    let mut last = None;
    for i in 0..RACE_POINTS {
        if pts[2 * i + 1].is_finite() {
            last = Some(pts[2 * i + 1]);
        } else if let Some(v) = last {
            pts[2 * i + 1] = v;
        }
    }
    let first = (0..RACE_POINTS).find(|&i| pts[2 * i + 1].is_finite()).map(|i| pts[2 * i + 1]).unwrap_or(1.0);
    for i in 0..RACE_POINTS {
        if !pts[2 * i + 1].is_finite() {
            pts[2 * i + 1] = first;
        }
    }
    (pts, mask)
}

impl MuonKin {
    /// The bars as they are to be: the gradient's, then the chosen member's (while the
    /// history plays, `phase` of the way to the next step's).
    fn bars_goal(&self) -> Vec<f32> {
        let heights = |t: usize| {
            let mut v = Vec::with_capacity(2 * WIDTH);
            let g = self.stream.grad_at(t);
            let s = self.stream.at(t, self.chosen);
            for k in 0..WIDTH {
                v.push(bar_height(g.map_or(0.0, |g| g.sv[k])) as f32);
            }
            for k in 0..WIDTH {
                v.push(bar_height(s.map_or(0.0, |s| s.sv[k])) as f32);
            }
            v
        };
        let t = self.shown();
        let now = heights(t);
        if self.phase > 0.0 && t < self.stream.done() {
            // (through the steps on either side too, a Catmull–Rom curve: the bars'
            // speed then changes smoothly from one step of the history to the next)
            let (before, next) = (heights(t.saturating_sub(1).max(1)), heights(t + 1));
            let after = heights((t + 2).min(self.stream.done()));
            let u = self.phase;
            return (0..now.len())
                .map(|i| {
                    let (p0, p1, p2, p3) = (before[i], now[i], next[i], after[i]);
                    0.5 * (2.0 * p1 + (p2 - p0) * u + (2.0 * p0 - 5.0 * p1 + 4.0 * p2 - p3) * u * u + (3.0 * p1 - p0 - 3.0 * p2 + p3) * u * u * u)
                })
                .collect();
        }
        now
    }

    fn race_goal(&self) -> (Vec<f32>, Vec<Vec<bool>>) {
        let frame = race_frame(self.race_view);
        let mut all = Vec::with_capacity(RACE.len() * 2 * RACE_POINTS);
        let mut masks = Vec::with_capacity(RACE.len());
        for r in RACE.iter() {
            let (p, m) = race_points(r, self.race_view, &frame);
            all.extend(p);
            masks.push(m);
        }
        (all, masks)
    }

    fn race_frame_goal(&self) -> Vec<f32> {
        let f = race_frame(self.race_view);
        vec![f.x[0] as f32, f.x[1] as f32, f.y[0] as f32, f.y[1] as f32]
    }

    /// Bring the picture `dt` seconds nearer to what it shows (`snap`: there at once).
    fn follow(&mut self, dt: f32, snap: bool) {
        let bars = self.bars_goal();
        let (race, _) = self.race_goal();
        let frame = self.race_frame_goal();
        let look = &mut self.look;
        if snap {
            look.bars.snap(&bars);
            look.race.snap(&race);
            look.race_frame.snap(&frame);
            look.moving = false;
            return;
        }
        let mut moving = look.bars.follow(&bars, dt, false);
        moving |= look.race.follow(&race, dt, false);
        moving |= look.race_frame.follow(&frame, dt, false);
        look.moving = moving;
    }
}

// ----- drawing -------------------------------------------------------------------------------

// Lines whose number of points changes, each in a batch of its own.
const EDGE: u8 = plot::OWN;
const BOX: u8 = EDGE + 1;
const LIT: u8 = BOX + 1;
const RACE_LINE: u8 = LIT + 1;
const RACE_LIT: u8 = RACE_LINE + 1;
const BAR_TOP: u8 = RACE_LIT + 1;

/// Every batch the scene draws in, made in the same order in every frame.
fn hold_batches(f: &mut Frame) {
    let corner = f.hud(Vec2::ZERO);
    for kind in [plot::AXES, plot::MARKS, plot::BUTTONS].into_iter().chain(EDGE..=BAR_TOP) {
        plot::apart(f, kind, false).segment(corner, corner, 1.0, Color::CLEAR);
    }
    plot::apart(f, EDGE, true).segment(corner, corner, 1.0, Color::CLEAR);
}

/// The gradient's bars.
const GRADIENT: Color = Color::hex(0xB8C0D0);

/// The dark of a box, and of a button.
const BOX_DARK: Color = Color::hex(0x161C2B);

/// The point where the segment from the centre of `r` towards `to` leaves the
/// rectangle `r`.
fn border_point((min, max): Rect, to: Vec2) -> Vec2 {
    let c = (min + max) * 0.5;
    let d = to - c;
    let half = (max - min) * 0.5;
    let sx = if d.x.abs() > 1e-6 { half.x / d.x.abs() } else { f32::INFINITY };
    let sy = if d.y.abs() > 1e-6 { half.y / d.y.abs() } else { f32::INFINITY };
    let s = sx.min(sy);
    if s.is_finite() {
        c + d * s
    } else {
        c
    }
}

impl MuonKin {
    /// The width a box needs for its name (and, on a large window, its second line).
    fn box_width(&self, f: &Frame, lay: &Layout, m: Member) -> f32 {
        if lay.narrow {
            let [a, b] = m.short();
            f.text_size(a, lay.name).x.max(f.text_size(b, lay.name).x) + 8.0
        } else {
            f.text_size(m.name(), lay.name).x.max(f.text_size("72° from Muon", lay.small).x) + 14.0
        }
    }

    /// The tree: families, edges, boxes. Clicking a box chooses its member.
    fn draw_tree(&mut self, f: &mut Frame, lay: &Layout) {
        let (min, max) = lay.tree;
        if !lay.tree_fits() {
            f.caption_px(Vec2::new(min.x, min.y), "a larger window shows the family tree", lay.small, pal::DIM, Align2::LEFT_TOP);
            return;
        }
        hud_panel(f, min - Vec2::new(8.0, 8.0), max + Vec2::new(8.0, 6.0));
        let title = if lay.narrow {
            r"Muon's family: what each rule does to $G = USV^\top$"
        } else {
            r"Muon's family: what each rule steps along, given a gradient $G = USV^\top$"
        };
        f.text_px(lay.tree_head + Vec2::new(0.0, -2.0), title, lay.text, pal::TEXT, Align2::LEFT_BOTTOM);
        let col = (max.x - min.x) / 6.0;
        let (bh, header) = lay.box_size();
        // the families' bands, and their names under them
        let row1 = lay.spot(Member::Ortho).y - 0.5 * bh - 8.0;
        let bands: [(f32, f32, Color, &str, &str); 3] = [
            (-0.5, 2.5, MUON, r"keep $U$ and $V$, change $S$", r"keep $U$, $V$"),
            (2.5, 3.5, ADAGRAD, "whiten, with memory", "memory"),
            (3.5, 5.5, ADAM, "act on each entry", "entries"),
        ];
        let foot = max.y - header;
        for (a, b, c, wide, short) in bands {
            let (x0, x1) = (min.x + (a + 0.5) * col + 2.0, min.x + (b + 0.5) * col - 2.0);
            // (colours mix in linear light: a little goes a long way)
            hud_rect(f, Vec2::new(x0, row1), Vec2::new(x1, max.y), Color::hex(0x080B14).mix(c, 0.022));
            let words = if lay.narrow { short } else { wide };
            f.text_px(Vec2::new((x0 + x1) * 0.5, foot + 0.5 * header), words, lay.small, c.mix(pal::TEXT, 0.35), Align2::CENTER_CENTER);
        }
        // the boxes' places
        let widths: Vec<f32> = Member::ALL.iter().map(|&m| self.box_width(f, lay, m)).collect();
        let boxes: Vec<Rect> = Member::ALL.iter().zip(&widths).map(|(&m, &w)| lay.box_of(m, w)).collect();
        let rect = |m: Member| boxes[m.index()];
        let lit = self.chosen;
        // the edges
        for e in EDGES.iter() {
            let (a, b) = (rect(e.from), rect(e.to));
            let (ca, cb) = ((a.0 + a.1) * 0.5, (b.0 + b.1) * 0.5);
            let (p, q) = (border_point(a, cb), border_point(b, ca));
            let into = e.to == lit;
            let color = if into { lit.color() } else { pal::DIM.a(0.75) };
            let width = if into { 1.8 } else { 1.1 };
            match e.look {
                EdgeLook::Line => {
                    plot::line_of(f, EDGE, p, q, width, color);
                    // an arrowhead at the child
                    let d = (q - p).normalize_or_zero();
                    let n = Vec2::new(-d.y, d.x);
                    plot::line_of(f, EDGE, q, q - d * 6.0 + n * 3.5, width, color);
                    plot::line_of(f, EDGE, q, q - d * 6.0 - n * 3.5, width, color);
                }
                EdgeLook::Same => {
                    let d = (q - p).normalize_or_zero();
                    let n = Vec2::new(-d.y, d.x) * 2.0;
                    plot::line_of(f, EDGE, p + n, q + n, width, color);
                    plot::line_of(f, EDGE, p - n, q - n, width, color);
                }
            }
            if !lay.narrow || e.look == EdgeLook::Same {
                let at = p + (q - p) * e.along;
                let words = if lay.narrow { "$=$" } else { e.words };
                let (offset, align) = match e.side {
                    Side::Left => (Vec2::new(-5.0, 0.0), Align2::RIGHT_CENTER),
                    Side::Right => (Vec2::new(5.0, 0.0), Align2::LEFT_CENTER),
                    Side::Above => (Vec2::new(0.0, -3.0), Align2::CENTER_BOTTOM),
                    Side::Below if lay.narrow => (Vec2::new(0.0, -3.0), Align2::CENTER_BOTTOM),
                    Side::Below => (Vec2::new(0.0, 0.5 * bh + 3.0), Align2::CENTER_TOP),
                };
                f.text_px(at + offset, words, lay.small, if into { lit.color().mix(pal::TEXT, 0.4) } else { pal::DIM }, align);
            }
        }
        // the comb from Muon to its variants
        let muon = rect(Member::Muon);
        let tops: Vec<Vec2> = VARIANTS.iter().map(|&m| Vec2::new((rect(m).0.x + rect(m).1.x) * 0.5, rect(m).0.y)).collect();
        let bus = (muon.1.y + tops[0].y) * 0.5;
        let from = Vec2::new((muon.0.x + muon.1.x) * 0.5, muon.1.y);
        let comb = pal::DIM.a(0.75);
        plot::line_of(f, EDGE, from, Vec2::new(from.x, bus), 1.1, comb);
        plot::line_of(f, EDGE, Vec2::new(tops[0].x, bus), Vec2::new(tops[tops.len() - 1].x, bus), 1.1, comb);
        for (k, t) in tops.iter().enumerate() {
            let into = VARIANTS[k] == lit;
            plot::line_of(f, EDGE, Vec2::new(t.x, bus), *t, if into { 1.8 } else { 1.1 }, if into { lit.color() } else { comb });
        }
        if !lay.narrow {
            f.text_px(Vec2::new(tops[tops.len() - 1].x + 6.0, bus), "its variants", lay.small, pal::DIM, Align2::LEFT_CENTER);
        }
        // the boxes
        let t = self.shown();
        let pointer = plot::pointer(f);
        let mut picked = None;
        for (i, &m) in Member::ALL.iter().enumerate() {
            let (a, b) = boxes[i];
            let on = m == lit;
            let over = pointer.is_some_and(|p| plot::within(p, (a, b), 0.0));
            let c = m.color();
            hud_rect_at(f, 2, a, b, if on { BOX_DARK.mix(c, 0.32) } else if over { BOX_DARK.mix(c, 0.14) } else { BOX_DARK });
            let edge = [a, Vec2::new(b.x, a.y), b, Vec2::new(a.x, b.y), a];
            plot::path_of(f, if on { LIT } else { BOX }, &edge, if on { 1.8 } else { 1.0 }, c.a(if on { 1.0 } else if over { 0.85 } else { 0.55 }));
            let centre = (a + b) * 0.5;
            let ink = if on { pal::WHITE } else { pal::TEXT.a(0.9) };
            if lay.narrow {
                let [l1, l2] = m.short();
                if l2.is_empty() {
                    f.text_px(centre, l1, lay.name, ink, Align2::CENTER_CENTER);
                } else {
                    f.text_px(centre - Vec2::new(0.0, 0.5 * lay.name + 0.5), l1, lay.name, ink, Align2::CENTER_CENTER);
                    f.text_px(centre + Vec2::new(0.0, 0.5 * lay.name + 0.5), l2, lay.name, ink, Align2::CENTER_CENTER);
                }
            } else {
                f.text_px(Vec2::new(centre.x, a.y + 5.0), m.name(), lay.name, ink, Align2::CENTER_TOP);
                let second = match (m, self.stream.at(t, m)) {
                    (Member::Muon, _) => "the reference".to_string(),
                    (_, Some(s)) if s.moved => format!("{} from Muon", angle(s.cos_muon)),
                    (_, Some(_)) => "no step yet".to_string(),
                    (_, None) => "…".to_string(),
                };
                f.text_px(Vec2::new(centre.x, b.y - 4.0), second, lay.small, c.mix(pal::TEXT, 0.45), Align2::CENTER_BOTTOM);
            }
            if over {
                f.handles.claim_clicks();
                f.handles.capture();
                if f.ptr.clicked {
                    f.ptr.clicked = false;
                    picked = Some(m);
                }
            }
        }
        if let Some(m) = picked {
            self.chosen = m;
        }
    }

    /// The chosen member's rule under the tree, on a large window: its name, its
    /// formulas, what they say, its source.
    fn draw_about(&self, f: &mut Frame, lay: &Layout) {
        let (min, max) = lay.about;
        if lay.narrow || max.y - min.y < 60.0 {
            return;
        }
        let m = self.chosen;
        let width = max.x - min.x;
        hud_panel(f, min - Vec2::new(8.0, 8.0), max + Vec2::new(8.0, 0.0));
        let mut y = min.y;
        f.text_px(Vec2::new(min.x, y), format!("{}: steps along {}", m.name(), m.along()), lay.text, m.color().mix(pal::TEXT, 0.3), Align2::LEFT_TOP);
        y += lay.text + 12.0;
        for eq in m.equations() {
            for line in eq.split(r"\\") {
                if y + 2.0 * lay.text > max.y {
                    return;
                }
                f.text_px(Vec2::new(min.x + 0.5 * width, y), format!("${}$", line.trim()), lay.text, pal::TEXT, Align2::CENTER_TOP);
                y += 1.9 * lay.text;
            }
        }
        y += 4.0;
        for words in [m.gloss(), m.source()] {
            let size = lay.small + 0.5;
            if y + wrapped_height_px(f, width, words, size) > max.y {
                return;
            }
            y += wrapped_px(f, Vec2::new(min.x, y), width, words, size, pal::TEXT.a(0.85)) + 8.0;
        }
    }

    /// The buttons over the picture; true if anything changed.
    fn buttons(&mut self, f: &mut Frame, lay: &Layout) -> bool {
        let mut changed = false;
        let size = lay.small;
        for (k, &b) in BATCHES.iter().enumerate() {
            let words = match (k, lay.narrow) {
                (0, false) => "batch 1".to_string(),
                (0, true) => "$B = 1$".to_string(),
                (4, _) => format!("all {POINTS}"),
                _ => format!("{b}"),
            };
            if plot::button(f, lay.buttons[k], &words, size, self.setup.batch == b) && self.setup.batch != b {
                self.setup.batch = b;
                self.refresh();
                changed = true;
            }
        }
        let play = if self.playing { "❚❚ pause" } else { "▶ history" };
        if plot::button(f, lay.buttons[5], play, size, false) {
            if self.playing {
                self.playing = false;
            } else {
                let to = if self.target > 1 { self.target } else { T_MAX };
                self.play_to(to);
            }
            changed = true;
        }
        for (k, (words, view)) in [("against the step", RaceView::Steps), ("against the rate", RaceView::Rates)].into_iter().enumerate() {
            let words = if lay.narrow { ["steps", "rates"][k] } else { words };
            if plot::button(f, lay.race_buttons[k], words, size, self.race_view == view) && self.race_view != view {
                self.race_view = view;
                changed = true;
            }
        }
        changed
    }

    /// The singular values of the chosen member's step beside the gradient's.
    fn draw_spectrum(&self, f: &mut Frame, lay: &Layout) {
        let p = Plot::new(lay.spec, [0.5, WIDTH as f64 + 0.5], [FLOOR, 0.12]);
        if p.tiny() {
            return;
        }
        let (min, max) = p.rect;
        let m = self.chosen;
        let t = self.shown();
        let size = lay.small;
        p.backing(f, if lay.narrow { 36.0 } else { 44.0 }, 34.0, 3.0 * (size + 4.0) + lay.text + 14.0, 8.0);
        // (the longest title that fits)
        let titles = [
            format!("{}: its step's singular values (its colour) and the gradient's (grey)", m.name()),
            format!("{}: its step's singular values and the gradient's (grey)", m.name()),
            format!("{}: its step and the gradient (grey)", m.name()),
        ];
        let room = max.x + 6.0 - lay.spec_head.x;
        let title = titles.iter().find(|t| f.text_size(t, lay.text).x <= room).unwrap_or(&titles[2]);
        f.text_px(lay.spec_head, title.as_str(), lay.text, pal::TEXT, Align2::LEFT_BOTTOM);
        // what the stream is at, and the numbers
        let batch = match (self.setup.batch >= POINTS, t == 1) {
            (true, _) => format!("all {POINTS} points"),
            (false, true) => format!("a batch of {}", self.setup.batch),
            (false, false) => format!("batches of {}", self.setup.batch),
        };
        let ortho = match self.setup.exact {
            true => "orthogonalised exactly",
            false => "Muon's polynomial",
        };
        let history = if t == 1 { "one gradient".to_string() } else { format!("{t} gradients") };
        let mut lines = vec![(format!("after {history} of {batch}, at fixed weights; {ortho}"), pal::DIM)];
        let ink = m.color().mix(pal::TEXT, 0.35);
        match self.step_shown() {
            Some(s) if s.moved => {
                let g = self.stream.grad_at(t);
                let mut angles = format!("{} from Muon's step, {} from the gradient", angle(s.cos_muon), angle(s.cos_grad));
                if self.setup.batch < POINTS {
                    angles += &format!(", {} from the whole data's orthogonal step", angle(s.cos_whole));
                }
                lines.push((angles, ink));
                lines.push((format!("stable rank {} (the gradient's {})", num(s.stable, 2), g.map_or("?".to_string(), |g| num(g.stable, 2))), ink));
            }
            Some(_) => lines.push(("no step yet: SOAP's first call only sets up its rotation".to_string(), ink)),
            None => lines.push(("working it out…".to_string(), ink)),
        }
        let y1 = lay.spec_head.y + 4.0;
        for (k, (line, color)) in lines.into_iter().enumerate() {
            f.text_px(Vec2::new(lay.spec_head.x, y1 + k as f32 * (size + 4.0)), line, size, color, Align2::LEFT_TOP);
        }
        // the axes
        plot::line_of(f, plot::AXES, Vec2::new(min.x, max.y), max, 1.0, pal::DIM.a(0.8));
        plot::line_of(f, plot::AXES, min, Vec2::new(min.x, max.y), 1.0, pal::DIM.a(0.8));
        for d in [0i32, -2, -4, -6] {
            let y = p.px_y(d as f64);
            plot::line_of(f, plot::AXES, Vec2::new(min.x - 4.0, y), Vec2::new(min.x, y), 1.0, pal::DIM.a(0.8));
            plot::line_of(f, plot::AXES, Vec2::new(min.x, y), Vec2::new(max.x, y), 1.0, pal::DIM.a(0.1));
            let name = if d == 0 { "1".to_string() } else { format!("$10^{{{d}}}$") };
            f.text_px(Vec2::new(min.x - 6.0, y), name, size, pal::DIM, Align2::RIGHT_CENTER);
        }
        for k in [1usize, 4, 8, 12, 16] {
            let x = p.px_x(k as f64);
            plot::line_of(f, plot::AXES, Vec2::new(x, max.y), Vec2::new(x, max.y + 4.0), 1.0, pal::DIM.a(0.8));
            f.text_px(Vec2::new(x, max.y + 6.0), format!("{k}"), size, pal::DIM, Align2::CENTER_TOP);
        }
        let across = if lay.narrow { "$k$-th largest" } else { "the $k$-th largest singular value, over the largest" };
        f.text_px(Vec2::new(max.x, max.y + 8.0 + size), across, size, pal::DIM, Align2::RIGHT_TOP);
        // the floor
        let floor = p.px_y(-6.0);
        plot::fill(f, (Vec2::new(min.x, floor), max), pal::DIM.a(0.08));
        // the bars
        let bars = &self.look.bars.shown;
        if bars.len() >= 2 * WIDTH {
            for k in 0..WIDTH {
                let x = (k + 1) as f64;
                for (a, b, h, c) in [(x - 0.42, x, bars[k] as f64, GRADIENT), (x, x + 0.42, bars[WIDTH + k] as f64, m.color())] {
                    p.bar(f, a, b, FLOOR, h, 1.0, c.a(0.85));
                    // (a crisp top edge)
                    let y = p.px_y(h).max(min.y).min(max.y);
                    plot::line_of(f, BAR_TOP, Vec2::new(p.px_x(a) + 1.0, y), Vec2::new(p.px_x(b) - 1.0, y), 1.6, c);
                }
            }
        }
        f.text_px(Vec2::new(min.x - 6.0, 0.5 * (floor + max.y)), "0", size, pal::DIM, Align2::RIGHT_CENTER);
    }

    /// The race: the loss against the step, or the search's score against the rate.
    fn draw_race(&mut self, f: &mut Frame, lay: &Layout) {
        let fr = &self.look.race_frame;
        let frame = RaceFrame { x: [fr.get(0) as f64, fr.get(1) as f64], y: [fr.get(2) as f64, fr.get(3) as f64] };
        let p = Plot::new(lay.race, frame.x, frame.y);
        if p.tiny() {
            return;
        }
        let (min, max) = p.rect;
        let size = lay.small;
        p.backing(f, if lay.narrow { 36.0 } else { 44.0 }, 34.0, lay.text + 14.0, 8.0);
        let title = match (self.race_view, lay.narrow) {
            (RaceView::Steps, _) => "the race (seed 1 of 3)",
            (RaceView::Rates, _) => "the search (3 seeds)",
        };
        f.text_px(lay.race_head, title, lay.text, pal::TEXT, Align2::LEFT_BOTTOM);
        if !lay.narrow {
            // (the longest words that fit beside the buttons)
            let choices: &[&str] = match self.race_view {
                RaceView::Steps => &["the loss on all 192 points; each rule at the rate the search chose", "each rule at the rate the search chose", "each at its chosen rate"],
                RaceView::Rates => &["each rule's loss over steps 920 to 1000 at every rate tried; the dot, its choice", "the loss at every rate tried; the dot, the choice", "the dot: the choice"],
            };
            let at = Vec2::new(lay.race_buttons[1].1.x + 10.0, (lay.race_buttons[1].0.y + lay.race_buttons[1].1.y) * 0.5);
            let room = lay.race.1.x + 6.0 - at.x;
            if let Some(sub) = choices.iter().find(|c| f.text_size(c, size).x <= room) {
                f.text_px(at, *sub, size, pal::DIM, Align2::LEFT_CENTER);
            }
        }
        // axes: log₁₀ up; steps or rates across
        plot::line_of(f, plot::AXES, Vec2::new(min.x, max.y), max, 1.0, pal::DIM.a(0.8));
        plot::line_of(f, plot::AXES, min, Vec2::new(min.x, max.y), 1.0, pal::DIM.a(0.8));
        let goal = race_frame(self.race_view);
        let (lo, hi) = (goal.y[0].ceil() as i32, goal.y[1].floor() as i32);
        let every = if hi - lo > 4 { 2 } else { 1 };
        for d in (lo..=hi).filter(|d| (d - hi) % every == 0) {
            let y = p.px_y(d as f64);
            if y < min.y - 0.5 || y > max.y + 0.5 {
                continue;
            }
            plot::line_of(f, plot::AXES, Vec2::new(min.x - 4.0, y), Vec2::new(min.x, y), 1.0, pal::DIM.a(0.8));
            plot::line_of(f, plot::AXES, Vec2::new(min.x, y), Vec2::new(max.x, y), 1.0, pal::DIM.a(0.1));
            let name = match d {
                0 => "1".to_string(),
                _ => format!("$10^{{{d}}}$"),
            };
            f.text_px(Vec2::new(min.x - 6.0, y), name, size, pal::DIM, Align2::RIGHT_CENTER);
        }
        match self.race_view {
            RaceView::Steps => {
                for s in [0usize, 250, 500, 750, 1000] {
                    let x = p.px_x(s as f64);
                    plot::line_of(f, plot::AXES, Vec2::new(x, max.y), Vec2::new(x, max.y + 4.0), 1.0, pal::DIM.a(0.8));
                    f.text_px(Vec2::new(x, max.y + 6.0), format!("{s}"), size, pal::DIM, Align2::CENTER_TOP);
                }
                f.text_px(Vec2::new(max.x, max.y + 8.0 + size), "step", size, pal::DIM, Align2::RIGHT_TOP);
            }
            RaceView::Rates => {
                for d in [-5i32, -4, -3, -2, -1, 0, 1, 2] {
                    let x = p.px_x(d as f64);
                    plot::line_of(f, plot::AXES, Vec2::new(x, max.y), Vec2::new(x, max.y + 4.0), 1.0, pal::DIM.a(0.8));
                    if lay.narrow && d % 2 != 0 {
                        continue;
                    }
                    let name = match d {
                        0 => "1".to_string(),
                        1 => "10".to_string(),
                        2 => "100".to_string(),
                        _ => format!("$10^{{{d}}}$"),
                    };
                    f.text_px(Vec2::new(x, max.y + 6.0), name, size, pal::DIM, Align2::CENTER_TOP);
                }
                f.text_px(Vec2::new(max.x, max.y + 8.0 + size), r"learning rate $\eta$", size, pal::DIM, Align2::RIGHT_TOP);
            }
        }
        f.text_px(Vec2::new(min.x + 6.0, min.y + 4.0), "loss", size, pal::DIM, Align2::LEFT_TOP);
        // the curves: the others faint, the chosen one bright
        let (_, masks) = self.race_goal();
        let pts = &self.look.race.shown;
        let at = |i: usize, j: usize| -> Option<Vec2> {
            let (u, v) = (*pts.get(i * 2 * RACE_POINTS + 2 * j)?, *pts.get(i * 2 * RACE_POINTS + 2 * j + 1)?);
            Some(Vec2::new(min.x + u * (max.x - min.x), max.y - v * (max.y - min.y)))
        };
        let lit = RACE.iter().position(|r| r.member == self.chosen);
        // (the lit rule's name first, so that it keeps its place; then Muon's)
        let mut labels: Vec<ChartLabel> = Vec::new();
        let mut reference_label: Option<ChartLabel> = None;
        for (i, r) in RACE.iter().enumerate() {
            let on = Some(i) == lit;
            let line: Vec<Vec2> = (0..RACE_POINTS).map(|j| if masks[i][j] { at(i, j).unwrap_or(Vec2::NAN) } else { Vec2::NAN }).collect();
            let c = r.member.color();
            let clipped = plot::clip(&line, p.rect);
            // (Muon, the scene's reference, a little brighter than the rest)
            let reference = r.member == Member::Muon;
            let (width, ink) = if on {
                (2.4, c)
            } else if reference {
                (1.5, c.a(0.75))
            } else {
                (1.1, c.a(0.38))
            };
            plot::path_of(f, if on { RACE_LIT } else { RACE_LINE }, &clipped, width, ink);
            if reference && !on {
                if let Some(j) = (0..RACE_POINTS).rev().find(|&j| masks[i][j]) {
                    if let Some(end) = at(i, j) {
                        reference_label = Some(ChartLabel { at: Vec2::new(end.x.min(max.x - 4.0), end.y + 4.0), align: Align2::RIGHT_TOP, text: "Muon".to_string(), color: c });
                    }
                }
            }
            if on {
                if let Some(j) = (0..RACE_POINTS).rev().find(|&j| masks[i][j]) {
                    if let Some(end) = at(i, j) {
                        let name = match self.race_view {
                            RaceView::Steps => format!("{}, rate {}", r.member.name(), sig(r.rate())),
                            RaceView::Rates => r.member.name().to_string(),
                        };
                        labels.push(ChartLabel { at: Vec2::new(end.x.min(max.x - 4.0), end.y - 4.0), align: Align2::RIGHT_BOTTOM, text: name, color: c });
                    }
                }
                // its chosen rate, on the search's chart
                if self.race_view == RaceView::Rates {
                    let x = (r.k - GRID_LO) as f64 / (GRID_N - 1) as f64;
                    let score = r.scores[(r.k - GRID_LO) as usize];
                    if score.is_finite() && score > 0.0 {
                        let y = (score.log10() - frame.y[0]) / (frame.y[1] - frame.y[0]);
                        let q = Vec2::new(min.x + x as f32 * (max.x - min.x), max.y - y as f32 * (max.y - min.y));
                        if plot::within(q, p.rect, 0.5) {
                            plot::disc(f, q, 7.0, c);
                        }
                    }
                }
            }
        }
        if lit.is_none() {
            let words = if lay.narrow { "this rule does not race" } else { "Shampoo without memory does not race: on every hidden matrix its step is that of Muon without momentum, orthogonalised exactly" };
            f.caption_px(Vec2::new(min.x + 8.0, max.y - 6.0), words, size, pal::DIM, Align2::LEFT_BOTTOM);
        }
        labels.extend(reference_label);
        self.labels_moving |= place(f, labels, size, p.rect, &mut self.placed);
    }
}

/// Text broken into lines no wider than `width`, from `at` down, in logical pixels of
/// the screen (`kit::wrapped` takes the plane's own units). Returns the height taken.
fn wrapped_px(f: &mut Frame, at: Vec2, width: f32, text: &str, size: f32, color: Color) -> f32 {
    let ppu = pixels_per_unit(f).max(1e-6);
    let Some(world) = f.cam.pick_ground(at * f.px) else { return 0.0 };
    let w = width / ppu;
    wrapped(f, world.truncate(), w, text, size, color);
    wrapped_height(f, w, text, size)
}

/// How tall `wrapped_px` draws a text.
fn wrapped_height_px(f: &Frame, width: f32, text: &str, size: f32) -> f32 {
    wrapped_height(f, width / pixels_per_unit(f).max(1e-6), text, size)
}

/// A label of the chart: where it wants to be (the point its alignment anchors),
/// what it says, its colour.
struct ChartLabel {
    at: Vec2,
    align: Align2,
    text: String,
    color: Color,
}

/// Where a label of the chart was put: its words, the slot it took (above or below
/// its own place, in its heights) and how far from its place it is drawn now.
#[derive(Clone, Debug, Default)]
struct Placed {
    text: String,
    slot: i32,
    drawn: f32,
}

/// The labels on plates, each at its own place if that is clear of those placed before
/// it, else the nearest clear place above or below it, inside the rectangle; a label
/// keeps the slot it had while that is still clear, and glides to a new one (`memo`
/// remembers them; true while one glides). (As the scene noisefloor places its own.)
fn place(f: &mut Frame, tags: Vec<ChartLabel>, size: f32, (min, max): Rect, memo: &mut Vec<Placed>) -> bool {
    let mut taken: Vec<Rect> = Vec::new();
    let mut now: Vec<Placed> = Vec::with_capacity(tags.len());
    let ease = 1.0 - (-f.real_dt.max(0.0) / 0.08).exp();
    let mut gliding = false;
    for t in tags {
        let w = f.text_size(&t.text, size) + Vec2::new(8.0, 4.0);
        let corner = |at: Vec2| {
            let x = match t.align.x() {
                egui::Align::Min => at.x,
                egui::Align::Center => at.x - 0.5 * w.x,
                egui::Align::Max => at.x - w.x,
            };
            let y = match t.align.y() {
                egui::Align::Min => at.y,
                egui::Align::Center => at.y - 0.5 * w.y,
                egui::Align::Max => at.y - w.y,
            };
            Vec2::new(x, y)
        };
        let fits = |c: Vec2| c.x >= min.x - 1.0 && c.y >= min.y - 1.0 && c.x + w.x <= max.x + 1.0 && c.y + w.y <= max.y + 1.0;
        let clear = |c: Vec2, taken: &[Rect]| taken.iter().all(|(a, b)| c.x + w.x <= a.x || b.x <= c.x || c.y + w.y <= a.y || b.y <= c.y);
        let step = w.y + 1.0;
        let before = memo.iter().find(|m| m.text == t.text).cloned();
        let slots = before.as_ref().map(|b| b.slot).into_iter().chain((0..13).map(|k| if k % 2 == 1 { -(k + 1) / 2 } else { k / 2 }));
        let found = slots.into_iter().find(|&k| {
            let c = corner(t.at + Vec2::new(0.0, k as f32 * step));
            fits(c) && clear(c, &taken)
        });
        let Some(slot) = found else { continue };
        let c = corner(t.at + Vec2::new(0.0, slot as f32 * step));
        taken.push((c, c + w));
        let goal = slot as f32 * step;
        let drawn = match before {
            Some(b) if (b.drawn - goal).abs() > 0.5 => {
                gliding = true;
                b.drawn + (goal - b.drawn) * ease
            }
            _ => goal,
        };
        f.caption_px(t.at + Vec2::new(0.0, drawn), t.text.clone(), size, t.color, t.align);
        now.push(Placed { text: t.text, slot, drawn });
    }
    *memo = now;
    gliding
}

// ----- what the panel says of each member ---------------------------------------------------

impl Member {
    /// Its source, and which form of it this is.
    fn source(self) -> &'static str {
        match self {
            Member::Gd => "Plain gradient descent, the root of the family.",
            Member::Sign => "signSGD, as Bernstein and co-authors (2018) give it.",
            Member::Signum => r"Signum's rule as Bernstein and co-authors (2018) give it; its $\beta = 0.9$ is chosen here.",
            Member::Lion => r"Lion, as Chen and co-authors (2023) give it, with $\beta_1 = 0.9$, $\beta_2 = 0.99$.",
            Member::Adam => r"Adam as Kingma and Ba give it, with their defaults $\beta_1 = 0.9$, $\beta_2 = 0.999$, $\epsilon = 10^{-8}$.",
            Member::Ortho => "Muon as its reference code has it (Jordan and co-authors, 2024), with the momentum set to 0.",
            Member::Fresh => "Shampoo as Gupta, Koren and Singer (2018) give it, with its two matrices made from the present gradient alone and no damping.",
            Member::Spectral => "Stochastic spectral descent as Carlson, Cevher and Carin (2015) give it: an exact decomposition and no momentum.",
            Member::Muon => r"Muon as its reference code has it (Jordan and co-authors, 2024): momentum $\beta = 0.95$, mixed with the gradient in Nesterov's way, then five rounds of its polynomial, or the exact $UV^\top$, as chosen. On this square layer its shape factor is 1.",
            Member::OSgdm => r"Orthogonal-SGDM as Tuddenham, Prügel-Bennett and Hare (2022) give it: the exact $UV^\top$ of each gradient, then momentum $\gamma = 0.9$, as in their experiments.",
            Member::Shampoo => r"Shampoo as Gupta, Koren and Singer (2018) give it in their Algorithm 1; its damping $\epsilon = 10^{-12}$ is chosen here.",
            Member::Soap => "SOAP as its authors' code has it (Vyas and co-authors, 2024): its first call only sets the rotation up, and the rotation is refreshed every 10 steps.",
            Member::NorMuon => r"NorMuon's rule as Li and co-authors (2025) give it, with Muon's $\beta = 0.95$; its second average's $\beta_2 = 0.95$ is chosen here.",
            Member::AdaMuon => r"AdaMuon's rule as Si, Zhang and Shen (2025) give it; its $\beta = 0.95$ is chosen here.",
            Member::Scion => r"Scion's constrained form, as Pethick and co-authors (2025) give it; its radius 1 is chosen here. The picture shows its step, not the shrinking of the weights.",
        }
    }

    /// Its rule, as display formulas (η the learning rate, G the gradient, M a running
    /// average of it).
    fn equations(self) -> &'static [&'static str] {
        match self {
            Member::Gd => &[r"\Delta W = -\eta\, G"],
            Member::Sign => &[r"\Delta W = -\eta \sgn G"],
            Member::Signum => &[r"M \leftarrow \beta M + (1 - \beta) G, \quad \Delta W = -\eta \sgn M"],
            Member::Lion => &[r"\Delta W = -\eta \sgn\big(\beta_1 M + (1 - \beta_1) G\big), \\ M \leftarrow \beta_2 M + (1 - \beta_2) G"],
            Member::Adam => &[r"M \leftarrow \beta_1 M + (1 - \beta_1) G, \quad V \leftarrow \beta_2 V + (1 - \beta_2) G^2, \\ \Delta W = -\eta\, \hat M / (\sqrt{\hat V} + \epsilon)"],
            Member::Ortho => &[r"\Delta W = -\eta\, U V^\top \quad \text{for } G = U S V^\top"],
            Member::Fresh => &[r"\Delta W = -\eta\, (G G^\top)^{-1/4}\, G\, (G^\top G)^{-1/4} = -\eta\, U V^\top"],
            Member::Spectral => &[r"\Delta W = -\eta\, \frac{\|G\|_{\mathrm{nuc}}}{mn}\, U V^\top"],
            Member::Muon => &[r"M \leftarrow \beta M + (1 - \beta) G, \\ \Delta W = -\eta \operatorname{orth}\big(\beta M + (1 - \beta) G\big)"],
            Member::OSgdm => &[r"V \leftarrow \gamma V + \eta \operatorname{orth}(G), \quad \Delta W = -V"],
            Member::Shampoo => &[r"L \leftarrow L + G G^\top, \quad R \leftarrow R + G^\top G, \\ \Delta W = -\eta\, L^{-1/4}\, G\, R^{-1/4}"],
            Member::Soap => &[r"G' = Q_L^\top G Q_R, \quad \Delta W = -\eta\, Q_L\, \frac{\hat M'}{\sqrt{\hat V'} + \epsilon}\, Q_R^\top"],
            Member::NorMuon => &[r"O = \operatorname{orth}(M), \quad v_i \leftarrow \beta_2 v_i + (1 - \beta_2) \operatorname{mean}_j O_{ij}^2, \\ \Delta W_{ij} \propto -O_{ij} / \sqrt{v_i}"],
            Member::AdaMuon => &[r"O = \operatorname{orth}(\sgn M), \quad V \leftarrow \beta V + (1 - \beta) O^2, \\ \Delta W_{ij} \propto -O_{ij} / \sqrt{V_{ij}}"],
            Member::Scion => &[r"D \leftarrow (1 - \alpha) D + \alpha G, \\ W \leftarrow (1 - \eta) W - \eta \rho \operatorname{orth}(D)"],
        }
    }

    /// The words under its formulas: what the symbols are, and what it does with a
    /// history.
    fn gloss(self) -> &'static str {
        match self {
            Member::Gd => r"It keeps $U$, $V$ and $S$: its step is as unequal as the gradient's singular values.",
            Member::Sign => r"Every entry moves by $\eta$, up or down. It is the steepest step when a step's size is its largest entry (Bernstein and Newhouse, 2024); $U V^\top$ is the steepest when its size is the most it stretches a vector.",
            Member::Signum => r"The sign of a running average of the gradients. With one gradient it is sign descent.",
            Member::Lion => r"Two averages: one mixed with $G$ for the step, another kept as the memory. Muon's momentum is this mixing with $\beta_1 = \beta^2$ and $\beta_2 = \beta$, and $U V^\top$ in place of the sign.",
            Member::Adam => r"$\hat M$ and $\hat V$ are the averages corrected for starting at 0. With both averages switched off it is sign descent (Bernstein and Newhouse, 2024), and so is its first step, to within $\epsilon$.",
            Member::Ortho => r"It keeps $U$ and $V$ and sets every singular value to 1: for a gradient of full rank, of all steps that stretch no vector by more than $\eta$, the one that lowers the loss most to first order (scene steepest).",
            Member::Fresh => r"For a matrix of full rank the two roots undo $S$ exactly: $(GG^\top)^{-1/4} G (G^\top G)^{-1/4} = U S^{-1/2} S S^{-1/2} V^\top$. Bernstein and Newhouse (2024) pointed this out. With the exponent $1/2$ used by SOAP's authors it gives $U S^{-1} V^\top$ instead.",
            Member::Spectral => r"The same direction as $U V^\top$, but its length grows with the nuclear norm $\|G\|_{\mathrm{nuc}} = \sum \sigma_i$ (divided by the number of entries, $mn$): the steepest step under the spectral norm when its size is a penalty rather than a bound (Bernstein and Newhouse, 2024). Lau, Long and Su (2025, PolarGrad) scale by $\|G\|_{\mathrm{nuc}}$ alone, so that the step shrinks as the gradient does, which Muon's does not.",
            Member::Muon => r"$\operatorname{orth}$ is $U V^\top$ of what it is given (or the reference's polynomial for it). The momentum comes first, then the orthogonalisation.",
            Member::OSgdm => r"The orthogonalisation comes first, then the momentum: an average of matrices each with every singular value 1, which is not itself one.",
            Member::Shampoo => r"$L$ and $R$ start at $\epsilon I$ and add up the gradients' $G G^\top$ and $G^\top G$ for ever, as AdaGrad adds up squares. With one gradient and no damping it is $U V^\top$.",
            Member::Soap => r"$Q_L$ and $Q_R$ are the eigenvectors of running averages of $G G^\top$ and $G^\top G$; in that basis it runs Adam, then turns back. With $Q_L = Q_R = I$ it is Adam.",
            Member::NorMuon => r"Muon's step with each row (each neuron's weights) divided by a running average of its size, then scaled so that the step's root mean square is $0.2\eta$.",
            Member::AdaMuon => r"The sign first, then the orthogonalisation, then each entry divided by a running average of its size; scaled to a root mean square of $0.2\eta$. Its first step is, to within its $\epsilon$, a matrix of signs: those of $\operatorname{orth}(\sgn G)$.",
            Member::Scion => r"Each step is a weighted average of the weights and a matrix of spectral norm $\rho$, so weights inside that ball stay inside it (Pethick and co-authors, 2025). Without the shrinking it is Muon without Nesterov's mixing.",
        }
    }

    /// What its authors (or others) report of it, with the scope.
    fn reports(self) -> Option<&'static str> {
        Some(match self {
            Member::Ortho => "In the NanoGPT speedrun's optimisation benchmark (a GPT-2-small-sized model, 12 layers 768 wide, batches of half a million tokens), Muon without momentum needed 8225 steps to the target loss where tuned Muon needed 3250.",
            Member::Muon => "Its published record against Adam is in scene muonadam.",
            Member::OSgdm => "On a ResNet-20 for CIFAR-10, with the settings tuned for SGD with momentum, Tuddenham and co-authors found it behind SGD with momentum (90.2 % against 91.2 % accuracy, five runs each); on a ResNet-18 for CIFAR-10 they report that it starts to outperform SGD with momentum at a batch of 16, and does worse at extremely small batches. Jordan and co-authors (2024) moved the momentum before the orthogonalisation, which they found works better.",
            Member::Shampoo => "Gupta and co-authors proved a bound on its regret in online convex optimisation; the exponent $1/4$ comes from that analysis. The Shampoo entries of the NanoGPT speedrun's optimisation benchmark use Meta's DistributedShampoo instead, which averages where this sums, takes momentum first and rescales its step to Adam's: another algorithm.",
            Member::Soap => "On language models of 360 and 660 million parameters its authors report, in the large-batch regime, over 40 % fewer iterations and over 35 % less wall-clock time than AdamW, and about 20 % fewer of each than Shampoo.",
            Member::NorMuon => "On models of 1.1 and 5.4 billion parameters trained on 50 billion tokens, its authors report 21.7 % and 13.9 % fewer steps than AdamW to reach AdamW's final loss, against 10.4 % and 6.1 % for Muon.",
            Member::AdaMuon => "Its authors' abstract reports more than 40 % better training efficiency than Adam in large-scale training.",
            Member::Scion => "In its authors' runs of nanoGPT models, its best step size carried over from one width to another, where Adam's and Muon's did not; at 3 billion parameters, with settings tuned at 124 million, its validation loss was 2.890, Muon's 2.909 and Adam's 3.024.",
            Member::Spectral => "Carlson and co-authors trained restricted Boltzmann machines with it, and found the decomposition's cost essentially free beside that of estimating their gradients.",
            _ => return None,
        })
    }
}

// ----- the scene's trait ---------------------------------------------------------------------

/// The variants of Muon published since, a sentence each: who, what was changed, what
/// they report and on what.
const VARIANT_NOTES: [(&str, &str); 8] = [
    ("Weight decay and the update's scale:", "Liu and co-authors (Moonshot AI, 2025) added decoupled weight decay and multiplied the step by $0.2\\sqrt{\\max(A, B)}$ for an $A$ by $B$ matrix, so that its root mean square is $0.2\\eta$, about what AdamW's usually is. Reusing AdamW's tuned settings, in compute-optimal runs of dense models of 0.4 to 1.5 billion parameters, they report AdamW's loss with about 52 % of its training FLOPs."),
    ("A better polynomial:", "Amsel and co-authors (2025, Polar Express) chose each round's polynomial to be the best possible over a range of singular values, so that the iteration converges; training GPT-2 Small and Large on a billion tokens, it gave lower losses than the reference's polynomial at every learning rate they tried."),
    ("Fewer directions:", "Ahn and co-authors (2025, Dion) make an orthonormal step of a chosen lower rank, by one step of power iteration with error feedback, to save communication when a matrix is spread over many machines; at 3 billion parameters they report half or a quarter of the directions on par with full Muon. (Not run here.)"),
    ("Constraints on the weights:", "Pethick and co-authors (2025, Scion) give every layer a norm and take each step as the best point of a ball in it; with weight decay the weights stay inside a ball."),
    ("An adaptive scale per neuron:", "Li and co-authors (2025, NorMuon) divide each row of Muon's step by a running average of its size."),
    ("An adaptive scale per entry:", "Si, Zhang and Shen (2025, AdaMuon) take the sign first and divide each entry by a running average of its size."),
    ("A step that shrinks with the gradient:", "Lau, Long and Su (2025, PolarGrad) scale $UV^\\top$ by the nuclear norm."),
    ("Clipping attention:", "The Kimi K2 team (2025, MuonClip) found attention logits growing large with Muon at scale and added a clip of a head's query and key weights after each step; with it they pre-trained on 15.5 trillion tokens without a loss spike."),
];

impl Scene for MuonKin {
    fn view(&self) -> View {
        View::Flat { center: Vec2::ZERO, half_height: 1.0 }
    }

    fn settings(&self, s: &mut ViewSettings) {
        s.grid = false;
        s.bloom = s.bloom.min(0.2);
    }

    fn animated(&self) -> bool {
        self.playing || self.look.moving || self.labels_moving || self.stream.done() < self.t.min(T_MAX)
    }

    fn pointer_hint(&self) -> Option<&'static str> {
        Some("a box: show that rule    the buttons: batch size, the history, the race's view")
    }

    fn ui(&mut self, ui: &mut egui::Ui, _: &Gfx) {
        ui::note(
            ui,
            r"Muon (scene muon) is one answer to a question many rules answer: given a layer's gradient matrix $G = U S V^\top$ (scene svd) and the gradients before it, what matrix does the step go along? Here one layer of a small network (a 16 by 16 matrix) hands its gradients to fifteen rules, all at the same fixed weights, and the answers sort into three kinds: keep $U$ and $V$ and change $S$; act on each entry; whiten with statistics remembered from the past.",
        );

        ui::section(ui, "Things to try");
        for (k, t) in TRIALS.iter().enumerate() {
            if ui::experiment(ui, t.button, t.caption) {
                self.set_up(k);
            }
        }

        ui::section(ui, "The rule shown");
        let names: Vec<&str> = Member::ALL.iter().map(|m| m.name()).collect();
        let mut k = self.chosen.index();
        if ui::choice(ui, &names, &mut k) {
            self.chosen = Member::ALL[k.min(Member::ALL.len() - 1)];
        }
        ui::note(ui, "Or click its box in the tree.");

        ui::section(ui, "The gradients");
        let mut b = BATCHES.iter().position(|&x| x == self.setup.batch).unwrap_or(4);
        if ui::choice(ui, &["1", "4", "16", "64", "all 192"], &mut b) {
            self.setup.batch = BATCHES[b.min(BATCHES.len() - 1)];
        }
        ui::note(ui, r"Examples per batch. A batch of $B$ gives a gradient of rank at most $B$: with one example every rule that keeps $U$ and $V$ steps along $G$ itself (scene backpropmat).");
        let mut t = self.t;
        if ui::slider_int(ui, r"after $t$ gradients", &mut t, 1..=T_MAX) && t != self.t {
            self.t = t;
            self.target = t;
            self.playing = false;
        }
        ui::note(ui, "Each a new batch at the same weights, so that only the history differs. With all the data in every batch the gradient never changes.");
        ui.horizontal_wrapped(|ui| {
            if ui.button(ui::label_text(ui, if self.playing { "pause" } else { "play the history" }, 12.5, None, false)).clicked() {
                if self.playing {
                    self.playing = false;
                } else {
                    self.play_to(T_MAX);
                }
            }
            if ui.button(ui::label_text(ui, "new batches", 12.5, None, false)).clicked() {
                self.setup.seed = self.setup.seed.wrapping_add(1);
            }
        });
        let mut e = (!self.setup.exact) as usize;
        if ui::choice(ui, &["orthogonalise exactly", "Muon's polynomial"], &mut e) {
            self.setup.exact = e == 0;
        }
        ui::note(ui, r"How Muon, its momentum-free form, NorMuon and AdaMuon make $U V^\top$: exactly, or by five rounds of the reference's polynomial (scene muon), which leaves singular values below about a thousandth of the matrix's size small. The others orthogonalise exactly, as their sources do.");
        self.refresh();

        ui::section(ui, self.chosen.name());
        let m = self.chosen;
        for eq in m.equations() {
            ui::equation(ui, eq);
        }
        ui::note(ui, m.gloss());
        ui::note(ui, m.source());
        if let Some(r) = m.reports() {
            ui::note(ui, r);
        }

        ui::section(ui, "The race");
        ui::note(
            ui,
            &format!(
                r"Each rule trains this network on the spirals from the same weights, on the same {RACE_STEPS} batches of {RACE_BATCH} points, at the rate (of {GRID_N}, from $10^{{-5}}$ to 100, four to a factor of ten) whose loss over steps 920 to {RACE_STEPS} was lowest on average over {} seeds. The groups a matrix rule leaves (the biases, and for the Muon-like rules the first layer and the read-out) take Adam at {}, the rate the search chose for Adam. Every other setting is the rule's published default. One small task, one search: the order says nothing beyond it.",
                RACE_SEEDS.len(),
                sig(grid_rate(AUX_K)),
            ),
        );
        ui::note(
            ui,
            r"Scion's radius, which its sources give no value for here, was searched too (1 to 64): it did best at the largest, 64, where its weights end far inside the ball (spectral norms below 9). At radius 1 its loss stayed above 0.9 at every rate: the ball holds the weights too small for these spirals.",
        );
        let mut v = (self.race_view == RaceView::Rates) as usize;
        if ui::choice(ui, &["against the step", "against the rate"], &mut v) {
            self.race_view = if v == 1 { RaceView::Rates } else { RaceView::Steps };
        }

        ui::section(ui, "Since Muon");
        for (term, words) in VARIANT_NOTES {
            ui::term(ui, term, words);
        }

        ui::section(ui, "Reading the picture");
        ui::key(
            ui,
            &[
                (MUON, "blue boxes: keep $U$ and $V$; Muon the deepest blue"),
                (ADAM, "orange boxes: act on each entry; Adam the deepest orange"),
                (ADAGRAD, "green boxes: whiten with remembered statistics"),
                (SGD, "cyan box: gradient descent"),
                (pal::DIM, "arrows: what turns one rule into the next; the lit box's arrow in its colour"),
                (GRADIENT, "grey bars: the gradient's own singular values; beside them, the lit rule's step's"),
                (pal::DIM.a(0.5), r"the band at the bottom of the bars: below $10^{-6}$ of the largest, or zero"),
                (MUON, "the race: a line for each rule in its box's colour, the lit rule's bright, Muon's a little brighter than the rest"),
            ],
        );
        ui::section(ui, "Words");
        ui::term(ui, r"Singular values, $G = U S V^\top$:", r"a matrix stretches the directions of $V$'s columns by $\sigma_1 \ge \sigma_2 \ge \dots \ge 0$ and turns them into $U$'s (scene svd).");
        ui::term(ui, r"$U V^\top$, orthogonalised:", "the same singular vectors with every singular value 1 (for a matrix that has lost rank, every one that is not 0): the polar factor (scene polar).");
        ui::term(ui, "Angle between two steps:", r"between them as lists of their 256 entries: $\cos\theta = \langle A, B\rangle / (\|A\|_F \|B\|_F)$.");
        ui::term(ui, "Stable rank:", r"$\sum_i \sigma_i^2 / \sigma_1^2$: about how many singular values are near the largest. 16 for $U V^\top$ here, 1 for a matrix of rank 1.");
        ui::term(ui, "Nuclear norm:", r"$\|G\|_{\mathrm{nuc}} = \sum_i \sigma_i$.");
        ui::term(ui, "Momentum:", "a running average of the gradients, stepped along in place of the latest.");
        ui::term(ui, "Whitening, preconditioner:", r"a matrix the gradient is multiplied by so that every direction counts alike: Shampoo's $L^{-1/4}$ and $R^{-1/4}$.");
        ui::term(ui, "Mini-batch:", "the examples whose mean gradient a step uses.");
        ui::term(ui, "Weight decay:", "shrinking the weights a little at every step; the picture shows each rule's step without it.");

        ui::section(ui, "Numbers");
        self.numbers(ui);
        ui::see_also(
            ui,
            &[
                ("muon", "Muon's own rule, step by step"),
                ("steepest", "which step is steepest depends on the ruler"),
                ("adam", "Adam and its averages"),
                ("polar", r"the polar factor $U V^\top$"),
                ("muonadam", "Muon against Adam on one network"),
            ],
        );
    }

    fn frame(&mut self, f: &mut Frame) {
        hold_batches(f);
        let lay = Layout::of(f.view_size(), f.top_inset);
        // (a canvas squeezed to a sliver is one the controls have opened over, on a phone)
        let tiny = lay.view.x < 160.0 || lay.view.y < 120.0;
        if !tiny && !lay.compact && self.buttons(f, &lay) {
            self.refresh();
        }
        let budget = crate::clock::frame_budget(f.real_dt, 0.004);
        self.advance(f.dt, &budget);
        let snap = self.canvas != Some(lay.view) && !tiny;
        if snap {
            self.canvas = Some(lay.view);
        }
        self.follow(f.real_dt, snap);
        if tiny {
            return;
        }
        self.labels_moving = false;
        self.draw_tree(f, &lay);
        if lay.compact {
            let at = Vec2::new(lay.tree.0.x, lay.view.y - 6.0);
            f.caption_px(at, "a taller window shows the steps' singular values and the race", lay.small, pal::DIM, Align2::LEFT_BOTTOM);
        } else {
            self.draw_about(f, &lay);
            self.draw_spectrum(f, &lay);
            self.draw_race(f, &lay);
        }
    }
}

impl MuonKin {
    /// The read-outs.
    fn numbers(&self, ui: &mut egui::Ui) {
        let t = self.shown();
        let m = self.chosen;
        ui::readout(ui, "gradients so far", format!("{t} of {T_MAX}"));
        if let Some(g) = self.stream.grad_at(t) {
            ui::readout_colored(ui, "the gradient's stable rank", num(g.stable, 2), SGD);
            ui::readout(ui, "  its angle to the whole data's", angle(g.cos_whole));
        }
        match self.step_shown() {
            Some(s) if s.moved => {
                ui::readout_colored(ui, "angle to Muon's step", angle(s.cos_muon), m.color());
                ui::readout_colored(ui, "angle to the gradient", angle(s.cos_grad), m.color());
                ui::readout_colored(ui, "angle to the whole data's orthogonal step", angle(s.cos_whole), m.color());
                ui::readout_colored(ui, "its step's stable rank", num(s.stable, 2), m.color());
                ui::readout_colored(ui, r"its largest stretch at $\eta = 1$", sig(s.spectral), m.color());
            }
            Some(_) => ui::readout(ui, "its step", "none yet"),
            None => ui::readout(ui, "its step", "working it out…"),
        }
        if matches!(m, Member::Fresh | Member::Ortho) {
            if let Some(b) = self.stream.bridge.get(t - 1) {
                ui::readout(ui, r"Shampoo without memory against the exact $U V^\top$", format!("entries within {}", sig(*b)));
            }
            if let Some(c) = self.stream.polynomial.get(t - 1) {
                ui::readout(ui, r"Muon's polynomial against the exact $U V^\top$", angle(*c));
            }
            if let Some(g) = self.stream.grad_at(t) {
                if g.rank_polar != g.rank_roots {
                    ui::readout(
                        ui,
                        "  why they differ",
                        format!("{} singular values count for the exact factor, {} for the roots", g.rank_polar, g.rank_roots),
                    );
                }
            }
        }
        if matches!(m, Member::Muon | Member::OSgdm) && self.setup.exact {
            if let Some(c) = self.stream.matched_whole.get(t - 1) {
                ui::readout(ui, r"Muon at Orthogonal-SGDM's momentum (0.9): angle to the whole data's orthogonal step", angle(*c));
            }
        }
        if let Some(r) = raced(m) {
            ui::readout_colored(ui, "in the race: its rate", sig(r.rate()), m.color());
            if m == Member::Scion {
                ui::readout(ui, "  its radius", num(r.rho, 0));
            }
            ui::readout_colored(ui, "  its loss over steps 920–1000, each seed", format!("{}, {}, {}", sig(r.finals[0]), sig(r.finals[1]), sig(r.finals[2])), m.color());
            ui::readout_colored(ui, "  their mean", sig(r.mean()), m.color());
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Experiment k as its button sets it up, with its stream worked out to the end and
    /// the history shown where the film stops.
    fn run(k: usize) -> MuonKin {
        let mut s = MuonKin::create();
        s.set_up(k);
        s.stream.work(T_MAX, &Budget::new(1e9));
        s.t = s.target;
        s.playing = false;
        s
    }

    fn step(s: &MuonKin, m: Member) -> Step {
        *s.stream.at(s.shown(), m).unwrap()
    }

    /// A stream worked out to the end.
    fn stream(setup: Setup) -> Stream {
        let p = problem();
        let frozen = warmed(WARM, NET_SEED, &p);
        let mut st = Stream::new(setup, &p, &frozen);
        st.work(T_MAX, &Budget::new(1e9));
        st
    }

    fn says(k: usize, what: &str) {
        assert!(TRIALS[k].caption.contains(what), "experiment {k} should say \"{what}\": {}", TRIALS[k].caption);
    }

    /// The numbers each experiment's caption says, from the scene as its button sets it
    /// up.
    #[test]
    fn the_experiments_show_what_their_captions_say() {
        // one gradient: three answers
        let s = run(0);
        assert_eq!((s.setup.batch, s.shown(), s.chosen), (POINTS, 1, Member::Muon));
        let g = *s.stream.grad_at(1).unwrap();
        let smallest = g.sv.iter().cloned().filter(|&v| v > 0.0).fold(1.0, f64::min);
        assert_eq!(g.sv.iter().filter(|&&v| v > 0.0).count(), WIDTH);
        says(0, r"$7.8 \times 10^{-6}$ of the largest");
        assert!((smallest - 7.8e-6).abs() < 0.05e-6, "{smallest}");
        let flat: Vec<Member> = Member::ALL.iter().copied().filter(|&m| m != Member::Muon && angle(step(&s, m).cos_muon) == "0.0°").collect();
        assert_eq!(flat, vec![Member::Ortho, Member::Fresh, Member::Spectral, Member::OSgdm, Member::NorMuon, Member::Scion]);
        says(0, &format!("{} boxes at 0.0° from Muon", flat.len()));
        for m in [Member::Ortho, Member::Fresh, Member::Spectral, Member::Muon, Member::OSgdm, Member::NorMuon, Member::Scion] {
            assert!(step(&s, m).sv.iter().all(|&v| (v - 1.0).abs() < 1e-6), "{m:?}");
        }
        for m in [Member::Sign, Member::Signum, Member::Lion, Member::Adam] {
            assert_eq!(angle(step(&s, m).cos_muon), "72°", "{m:?}");
        }
        says(0, "are sign descent): 72°.");
        says(0, &format!("$G$ itself: {}", angle(step(&s, Member::Gd).cos_muon)));
        says(0, &format!("Shampoo is {} off", angle(step(&s, Member::Shampoo).cos_muon)));
        // (its damping is the whole difference: without it, Shampoo's first step is UVᵀ)
        assert_eq!(angle(step(&s, Member::Fresh).cos_muon), "0.0°");
        says(0, &format!("AdaMuon, which takes the sign first, is {} off", angle(step(&s, Member::AdaMuon).cos_muon)));
        assert!(!step(&s, Member::Soap).moved);

        // the sign of each entry
        let s = run(1);
        assert_eq!(s.chosen, Member::Sign);
        let st = step(&s, Member::Sign);
        let nonzero: Vec<f64> = st.sv.iter().cloned().filter(|&v| v > 0.0).collect();
        says(1, &format!("to {} of the largest, and {} of the 16 are below a millionth of it", num(*nonzero.last().unwrap(), 3), WIDTH - nonzero.len()));
        says(1, &format!("Its stable rank is {}, against the gradient's {}", num(st.stable, 2), num(g.stable, 2)));
        says(1, &format!("It is {} from the gradient's step and {} from Muon's", angle(st.cos_grad), angle(st.cos_muon)));

        // Shampoo without memory: the bridge, and the polynomial beside it
        let s = run(2);
        assert_eq!(s.chosen, Member::Fresh);
        let bridge = s.stream.bridge[0];
        assert!(bridge < 5.95e-8 && bridge > 5.85e-8, "{bridge}");
        says(2, r"to within $5.9 \times 10^{-8}$ of the largest");
        let (a, b) = (s.stream.grads[0].sv[0], s.stream.grads[0].sv[WIDTH - 1]);
        assert!((2.0 * (a / b).log10()).floor() == 10.0, "the eigenvalues of G Gᵀ span {} decades", 2.0 * (a / b).log10());
        says(2, &format!("lands {} away", angle(s.stream.polynomial[0])));
        let mut poly = s.stream.whole.clone();
        Iteration::JORDAN.run(&mut poly, WIDTH, WIDTH, &mut Scratch::new());
        let sv = optim::singular_values(&poly, WIDTH, WIDTH);
        let frob = optim::frobenius(&s.stream.whole);
        let gsv = optim::singular_values(&s.stream.whole, WIDTH, WIDTH);
        let small: Vec<usize> = (0..WIDTH).filter(|&k| gsv[k] < 1e-3 * frob).collect();
        assert_eq!(small, (10..16).collect::<Vec<_>>());
        says(2, &format!("raises the 10 largest singular values to between {} and {}", num(sv[9], 2), num(sv[0], 2)));
        says(2, &format!("leaves the 6 smallest, each under a thousandth of the gradient's size, between {} and {}", num(sv[15], 3), num(sv[10], 2)));

        // Shampoo with memory
        let s = run(3);
        assert_eq!((s.setup.batch, s.shown(), s.chosen), (16, T_MAX, Member::Shampoo));
        let st = step(&s, Member::Shampoo);
        says(3, &format!("after 64, its stable rank is {} (Muon's is 16)", num(st.stable, 2)));
        says(3, &format!("it lies {} from the latest gradient and {} from Muon's step", angle(st.cos_grad), angle(st.cos_muon)));
        let full = stream(Setup::base());
        let sh = full.at(T_MAX, Member::Shampoo).unwrap();
        says(3, &format!("({} from it after 64)", angle(sh.cos_muon)));
        for t in [1usize, 4, 16, 64] {
            let len = full.at(t, Member::Shampoo).unwrap().spectral;
            assert!((len * (t as f64).sqrt() - 1.0).abs() < 2e-3, "t {t}: {len}");
        }

        // momentum before or after
        let s = run(4);
        assert_eq!((s.setup.batch, s.shown(), s.chosen), (16, T_MAX, Member::OSgdm));
        let whole = |m: Member| angle(step(&s, m).cos_whole);
        says(4, &format!("Muon's step is {} away, Orthogonal-SGDM's {}, and Muon's without momentum {}", whole(Member::Muon), whole(Member::OSgdm), whole(Member::Ortho)));
        assert!(s.stream.setup.exact);
        says(4, &format!("(0.9, no Nesterov mix) Muon is {} away", angle(s.stream.matched_whole[T_MAX - 1])));
        let sets = 8u64;
        says(4, &format!("the order was the same for {sets} sets of batches"));
        for seed in 1..=sets {
            let st = stream(Setup { batch: 16, exact: true, seed });
            let d = |m: Member| degrees(st.at(T_MAX, m).unwrap().cos_whole);
            let matched = degrees(st.matched_whole[T_MAX - 1]);
            assert!(d(Member::Muon) < matched && matched < d(Member::OSgdm) && d(Member::OSgdm) < d(Member::Ortho), "seed {seed}");
        }

        // the race
        let s = run(5);
        assert_eq!(s.race_view, RaceView::Steps);
        let mean = |m: Member| raced(m).unwrap().mean();
        says(5, &format!("the same 1000 batches of {RACE_BATCH}"));
        assert_eq!(RACE_STEPS, 1000);
        says(5, &format!("a search over {GRID_N} chose"));
        assert_eq!(sig_n(mean(Member::Soap), 2), r"$3.7e-7$");
        says(5, &format!("Muon {}, NorMuon {}, Scion {}, Adam {}", sig_n(mean(Member::Muon), 2), sig_n(mean(Member::NorMuon), 2), sig_n(mean(Member::Scion), 2), sig_n(mean(Member::Adam), 2)));
        assert_eq!(s.chosen, Member::Soap);
        says(5, r"SOAP $3.7 \times 10^{-7}$");
        let mut order: Vec<Member> = RACERS.to_vec();
        order.sort_by(|a, b| mean(*a).total_cmp(&mean(*b)));
        assert_eq!(&order[..5], &[Member::Soap, Member::Muon, Member::NorMuon, Member::Scion, Member::Adam]);
        // (the score: the noted losses at steps 920, 940, ..., 1000)
        assert_eq!(RACE_STEPS - (RACE_POINTS - 5) * RACE_EVERY, 80);
        says(5, "The loss over the last 80 steps, the mean of three seeds");
        says(5, &format!("Muon without momentum {}, ten times Muon's", sig_n(mean(Member::Ortho), 2)));
        let ratio = mean(Member::Ortho) / mean(Member::Muon);
        assert!((9.5..10.5).contains(&ratio), "{ratio}");
        says(5, &format!("gradient descent {}", sig_n(mean(Member::Gd), 2)));
        assert_eq!(*order.last().unwrap(), Member::Gd);
    }

    /// What the panel says of the race: the fall-back's rate is Adam's choice; Scion's
    /// radius was the largest tried, it ends far inside its ball, and at radius 1 it
    /// cannot fit the spirals at any rate.
    #[test]
    fn the_race_is_as_the_panel_says() {
        assert_eq!(raced(Member::Adam).unwrap().k, AUX_K);
        let scion = raced(Member::Scion).unwrap();
        assert_eq!(scion.rho, RADII[RADII.len() - 1]);
        const { assert!(SCION_BY_RADIUS[0] > 0.9) };
        // every racer's choice is inside its grid, not at an edge
        for r in RACE.iter() {
            assert!(r.k > GRID_LO && r.k < GRID_HI, "{:?} at the edge of the grid", r.member);
            assert!(r.finals.iter().all(|v| v.is_finite()) && r.curve.iter().all(|v| v.is_finite()), "{:?}", r.member);
            assert!((r.mean() - r.scores[(r.k - GRID_LO) as usize]).abs() < 1e-4 * r.mean(), "{:?}", r.member);
        }
        // Scion's weights at the end of its race (seed 1): far inside the ball of radius 64
        let p = problem();
        let mut model = Model::mlp(&spec(), &mut Rng::new(RACE_SEEDS[0]));
        let (rule, hyper) = racing(Member::Scion, scion.rate(), scion.rho);
        let mut opt = Optimizer::new(&model, rule, hyper);
        let mut tape = Tape::new();
        let mut batches = Sampler::new(p.len(), RACE_BATCH, Draw::WithoutReplacement, Rng::new(RACE_SEEDS[0].wrapping_add(200)));
        for _ in 0..RACE_STEPS {
            let idx = batches.deal().to_vec();
            p.grad(&mut model, &mut tape, &idx);
            opt.step(&mut model, scion.rate());
        }
        for g in model.groups() {
            if opt.as_matrix(model.index_of(&g.name).unwrap()) {
                let n = optim::spectral_norm(&g.value, g.rows, g.cols);
                assert!(n < 9.0, "{}: {n}", g.name);
            }
        }
        assert!(p.loss(&model, &mut tape) > 0.0);
    }

    /// Three racers run again (seed 1, at their chosen rates): their curves are the
    /// table's to the digits kept.
    #[test]
    fn the_stored_curves_are_what_a_race_gives() {
        let p = problem();
        for m in [Member::Muon, Member::Adam, Member::Gd] {
            let r = raced(m).unwrap();
            let curve = race(m, r.rate(), r.rho, RACE_SEEDS[0], &p);
            for (a, b) in curve.iter().zip(&r.curve) {
                assert!((a - b).abs() <= 1e-4 * b.abs(), "{m:?}: {a} against {b}");
            }
        }
    }

    /// The identities the panel states, on this layer's gradients.
    #[test]
    fn the_relations_the_panel_states() {
        let p = problem();
        let frozen = warmed(WARM, NET_SEED, &p);
        let layer = frozen.index_of(LAYER_NAME).unwrap();
        let mut tape = Tape::new();
        let mut model = frozen.clone();
        let mut batches = Sampler::new(p.len(), 16, Draw::WithoutReplacement, Rng::new(9));
        // Muon's momentum mixes as Lion does, with β1 = β² and β2 = β
        let beta = 0.95;
        let mut muon = Optimizer::new(&model, Rule::Muon, Hyper::muon_reference());
        let mut lion_m = vec![0.0; WIDTH * WIDTH];
        for _ in 0..12 {
            let idx = batches.deal().to_vec();
            p.grad(&mut model, &mut tape, &idx);
            let g = model.groups()[layer].grad.clone();
            muon.step(&mut model, 0.0);
            let c: Vec<f64> = (0..g.len()).map(|i| beta * beta * lion_m[i] + (1.0 - beta * beta) * g[i]).collect();
            for i in 0..g.len() {
                lion_m[i] = beta * lion_m[i] + (1.0 - beta) * g[i];
            }
            assert!(cosine(muon.ortho_in(layer), &c) > 1.0 - 1e-12);
            let ratio = optim::frobenius(muon.ortho_in(layer)) / optim::frobenius(&c);
            assert!((ratio - 1.0).abs() < 1e-12, "{ratio}");
        }
        // Moonshot's and torch's forms step the same way as the reference's on this
        // square layer; the blog's (no Nesterov) does not
        for (name, h, same) in [("Moonshot", Hyper::muon_moonshot(), true), ("torch", Hyper::muon_torch(), true), ("blog", Hyper::muon_blog(), false)] {
            let mut m1 = frozen.clone();
            let mut m2 = frozen.clone();
            let mut a = Optimizer::new(&m1, Rule::Muon, Hyper::muon_reference());
            let mut b = Optimizer::new(&m2, Rule::Muon, h);
            let mut batches = Sampler::new(p.len(), 16, Draw::WithoutReplacement, Rng::new(3));
            let mut worst = 1.0f64;
            for _ in 0..10 {
                let idx = batches.deal().to_vec();
                p.grad(&mut m1, &mut tape, &idx);
                for (d, s) in m2.groups_mut().iter_mut().zip(m1.groups()) {
                    d.grad.copy_from_slice(&s.grad);
                }
                a.step(&mut m1, 1e-9);
                b.step(&mut m2, 1e-9);
                worst = worst.min(cosine(a.update(layer), b.update(layer)));
            }
            if same {
                assert!(worst > 1.0 - 1e-9, "{name}: {worst}");
            } else {
                assert!(worst < 0.999, "{name}: {worst}");
            }
        }
        // AdaMuon's first step is a matrix of signs; Adam's is sign descent's to within ε
        let st = stream(Setup::base());
        let mut model = frozen.clone();
        p.full_grad(&mut model, &mut tape);
        let g = model.groups()[layer].grad.clone();
        let mut ada = Optimizer::new(&model, Rule::AdaMuon, Hyper::adamuon());
        ada.step(&mut model.clone(), 1.0);
        let u = ada.update(layer);
        let size = u[0].abs();
        assert!(size > 0.0 && u.iter().all(|x| (x.abs() - size).abs() < 1e-4 * size), "AdaMuon's first step");
        let adam = st.at(1, Member::Adam).unwrap();
        let sign = st.at(1, Member::Sign).unwrap();
        assert_eq!(angle(adam.cos_muon), angle(sign.cos_muon));
        let mut first = Optimizer::new(&model, Rule::Adam, Hyper::default_for(Rule::Adam));
        first.step(&mut model.clone(), 1.0);
        for (u, gi) in first.update(layer).iter().zip(&g) {
            if gi.abs() > 1e-6 {
                assert!((u + gi.signum()).abs() < 0.01, "Adam's first step: {u} for {gi}");
            }
        }
        // Shampoo without memory is the exact UVᵀ wherever the gradient's rank is clear
        for batch in [1usize, 4, POINTS] {
            let st = stream(Setup { batch, ..Setup::base() });
            for t in 1..=T_MAX {
                let gr = st.grad_at(t).unwrap();
                assert_eq!(gr.rank_polar, gr.rank_roots, "B {batch} t {t}");
                assert!(st.bridge[t - 1] < 1e-6, "B {batch} t {t}: {}", st.bridge[t - 1]);
            }
        }
        // with one example every rule that keeps U and V steps along G itself
        let st = stream(Setup { batch: 1, ..Setup::base() });
        for m in [Member::Ortho, Member::Fresh, Member::Spectral, Member::Muon, Member::OSgdm, Member::Shampoo, Member::Scion] {
            assert!(st.at(1, m).unwrap().cos_grad > 1.0 - 1e-9, "{m:?}");
        }
        assert!(st.grad_at(1).unwrap().sv[1] == 0.0);
    }

    /// The scene opens as the first experiment sets it up.
    #[test]
    fn the_scene_opens_on_the_first_experiment() {
        let opened = MuonKin::create();
        let mut pressed = MuonKin::create();
        pressed.set_up(3);
        pressed.set_up(OPENING);
        assert_eq!((opened.setup, opened.chosen, opened.t, opened.race_view), (pressed.setup, pressed.chosen, pressed.t, pressed.race_view));
    }

    /// Every batch size, both ways of orthogonalising, every member, every step of the
    /// history: every number drawn is finite (or the step is honestly "none yet").
    #[test]
    fn every_setting_draws_finite_numbers() {
        for batch in BATCHES {
            for exact in [true, false] {
                let mut s = MuonKin::create();
                s.setup = Setup { batch, exact, seed: 2 };
                s.refresh();
                s.stream.work(T_MAX, &Budget::new(1e9));
                for t in [1usize, 2, 17, T_MAX] {
                    s.t = t;
                    for m in Member::ALL {
                        s.chosen = m;
                        let st = step(&s, m);
                        if !st.moved {
                            assert!(m == Member::Soap && t == 1, "{m:?} B {batch} t {t}");
                            continue;
                        }
                        assert!(st.cos_muon.is_finite() && st.cos_grad.is_finite() && st.cos_whole.is_finite() && st.stable.is_finite() && st.spectral.is_finite(), "{m:?} B {batch} t {t}");
                        assert!(st.sv.iter().all(|v| v.is_finite() && *v >= 0.0 && *v <= 1.0 + 1e-12));
                        s.follow(0.016, true);
                        assert!(s.look.bars.shown.iter().all(|v| v.is_finite()));
                    }
                }
            }
        }
        for view in [RaceView::Steps, RaceView::Rates] {
            let f = race_frame(view);
            assert!(f.x[1] > f.x[0] && f.y[1] > f.y[0]);
            for r in RACE.iter() {
                let (pts, mask) = race_points(r, view, &f);
                assert!(pts.iter().all(|v| v.is_finite()) && mask.iter().any(|&m| m), "{:?}", r.member);
            }
        }
    }

    /// The layout keeps everything in the window and apart, from a phone to a desktop.
    #[test]
    fn the_layout_fits_every_window() {
        for (w, h) in [(1080.0, 1000.0), (1220.0, 1000.0), (1600.0, 1000.0), (1600.0, 700.0), (390.0, 800.0), (360.0, 640.0), (700.0, 500.0), (300.0, 600.0), (500.0, 400.0), (240.0, 180.0)] {
            let lay = Layout::of(Vec2::new(w, h), 40.0);
            let inside = |(a, b): Rect| a.x >= 0.0 && a.y >= 0.0 && b.x <= w + 0.5 && b.y <= h + 0.5 && a.is_finite() && b.is_finite();
            let apart = |(a, b): Rect, (c, d): Rect| b.x <= c.x || d.x <= a.x || b.y <= c.y || d.y <= a.y;
            assert!(inside(lay.tree), "{w}x{h}: {:?}", lay.tree);
            assert!(lay.tree_fits() || (w < 300.0 && h < 300.0), "{w}x{h}: no room for the tree");
            if !lay.tree_fits() {
                continue;
            }
            // the boxes, at their widest, inside the tree and apart
            let boxes: Vec<Rect> = Member::ALL.iter().map(|&m| lay.box_of(m, 1e9)).collect();
            for (i, a) in boxes.iter().enumerate() {
                assert!(a.0.x >= lay.tree.0.x - 1.0 && a.1.x <= lay.tree.1.x + 1.0 && a.0.y >= lay.tree.0.y - 1.0 && a.1.y <= lay.tree.1.y + 1.0, "{w}x{h}: box {i}");
                for b in &boxes[i + 1..] {
                    assert!(apart(*a, *b), "{w}x{h}: boxes {a:?} and {b:?} overlap");
                }
            }
            if lay.compact {
                continue;
            }
            let rects: Vec<Rect> = [vec![lay.tree, lay.spec, lay.race], lay.buttons.to_vec(), lay.race_buttons.to_vec()].concat();
            for (i, a) in rects.iter().enumerate() {
                assert!(inside(*a), "{w}x{h}: {i} {a:?}");
                for b in &rects[i + 1..] {
                    assert!(apart(*a, *b), "{w}x{h}: {a:?} and {b:?} overlap");
                }
            }
            assert!(lay.spec.1.y - lay.spec.0.y > 50.0 && lay.race.1.y - lay.race.0.y > 70.0, "{w}x{h}");
        }
    }

    /// The history plays to its end and the picture comes to rest; a change glides and
    /// stops.
    #[test]
    fn the_history_plays_and_the_picture_rests() {
        let mut s = MuonKin::create();
        s.follow(0.016, true);
        s.set_up(3);
        let (mut frames, mut played) = (0, 0);
        let mut last = 0;
        while s.animated() {
            s.advance(1.0 / 60.0, &Budget::new(1.0));
            s.follow(1.0 / 60.0, false);
            assert!(s.shown() >= last, "the history went back");
            last = s.shown();
            frames += 1;
            if s.playing {
                played = frames;
            }
            assert!(frames < 60 * 20, "still playing");
        }
        assert_eq!(s.shown(), T_MAX);
        // as long as the rate says, and the bars come to rest soon after
        let want = (T_MAX - 1) as f32 / PLAY_RATE * 60.0;
        assert!((played as f32 - want).abs() < 3.0, "{played} frames, {want} wanted");
        assert!(frames - played < 90, "{} frames to rest", frames - played);
        for k in [0usize, 5, 1, 2] {
            s.set_up(k);
            s.race_view = if k == 5 { RaceView::Rates } else { RaceView::Steps };
            let mut frames = 0;
            s.advance(1.0 / 60.0, &Budget::new(1.0));
            s.follow(1.0 / 60.0, false);
            while s.animated() {
                s.advance(1.0 / 60.0, &Budget::new(1.0));
                s.follow(1.0 / 60.0, false);
                frames += 1;
                assert!(frames < 400, "experiment {k} still moving");
            }
        }
    }

    /// The numbers behind the captions (run with --ignored --nocapture).
    #[test]
    #[ignore]
    fn explore() {
        for k in 0..TRIALS.len() {
            let s = run(k);
            let t = s.shown();
            let g = s.stream.grad_at(t).unwrap();
            println!("== {k} {} t {t} B {} exact {}", TRIALS[k].button, s.setup.batch, s.setup.exact);
            println!("   grad: stable {:.3} cos whole {:.3} ranks {} {}", g.stable, g.cos_whole, g.rank_polar, g.rank_roots);
            println!("   bridge {:.3e} polynomial {:.3} matched {:.3}", s.stream.bridge[t - 1], degrees(s.stream.polynomial[t - 1]), degrees(s.stream.matched_whole[t - 1]));
            for m in Member::ALL {
                let st = step(&s, m);
                println!("   {:>18}: muon {:6.2} grad {:6.2} whole {:6.2} stable {:6.3} spec {:.4e} moved {}", m.name(), degrees(st.cos_muon), degrees(st.cos_grad), degrees(st.cos_whole), st.stable, st.spectral, st.moved);
            }
        }
    }
}

#[cfg(test)]
mod race_table {
    use super::*;

    /// A number as the table writes it: six significant digits.
    fn n(v: f64) -> String {
        if v.is_infinite() {
            "f64::INFINITY".into()
        } else {
            format!("{:.5e}", v)
        }
    }

    fn near(a: f64, b: f64) -> bool {
        if a.is_infinite() || b.is_infinite() {
            return a == b;
        }
        (a - b).abs() <= 1e-4 * b.abs().max(1e-300)
    }

    /// The whole search, run again (on 24 threads, about ten seconds): it chooses what
    /// the table says, and its numbers are the table's to the six digits kept. Prints
    /// the table, to paste in if the race is ever changed.
    #[test]
    #[ignore]
    fn the_stored_race_is_what_the_search_finds() {
        let (all, by_radius) = search(24);
        println!("#[rustfmt::skip]\nconst RACE: [Raced; {}] = [", all.len());
        for r in &all {
            println!("    Raced {{ member: Member::{:?}, k: {}, rho: {:.1}, finals: [{}],", r.member, r.k, r.rho, r.finals.iter().map(|v| n(*v)).collect::<Vec<_>>().join(", "));
            println!("        curve: [{}],", r.curve.iter().map(|v| n(*v)).collect::<Vec<_>>().join(", "));
            println!("        scores: [{}] }},", r.scores.iter().map(|v| n(*v)).collect::<Vec<_>>().join(", "));
        }
        println!("];");
        println!("const SCION_BY_RADIUS: [f64; {}] = [{}];", by_radius.len(), by_radius.iter().map(|v| n(*v)).collect::<Vec<_>>().join(", "));
        assert_eq!(all.len(), RACE.len());
        for (got, want) in all.iter().zip(RACE.iter()) {
            assert_eq!((got.member, got.k), (want.member, want.k), "{:?}", got.member);
            assert_eq!(got.rho, want.rho, "{:?}", got.member);
            for (a, b) in got.finals.iter().zip(&want.finals).chain(got.curve.iter().zip(&want.curve)).chain(got.scores.iter().zip(&want.scores)) {
                assert!(near(*a, *b), "{:?}: {a} against {b}", got.member);
            }
        }
        for (a, b) in by_radius.iter().zip(&SCION_BY_RADIUS) {
            assert!(near(*a, *b), "Scion by radius: {a} against {b}");
        }
    }
}

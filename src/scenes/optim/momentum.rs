//! Momentum: a ball with weight. Gradient descent forgets its last step; heavy-ball
//! momentum keeps a share β of it. On a bowl each of the bowl's own directions still
//! moves on its own, but its state is now two numbers (how far it is from the bottom and
//! how fast it moves) and a step multiplies them by a 2 × 2 matrix. That matrix's two
//! eigenvalues say everything, as 1 − ηλ did without momentum: whether the direction
//! creeps in, rings, flips or flies off, and by how much it shrinks per step. The map of
//! them over the step and the momentum is the picture; the bowl's directions are dots on
//! it, and a valley seen from above and the losses show what the map predicts.

use super::*;
use crate::scenes::stats::plot::{self, Plot, Rect};

pub const DEF: SceneDef = SceneDef {
    id: "momentum",
    title: "Momentum: a ball with weight",
    group: GROUP,
    blurb: "Keeping part of the last step turns a crawl along the valley floor into a glide: which steps and momenta converge, which ring, and which fly off.",
    make: |_| Box::new(Momentum::new()),
};

// ----- the mathematics, in f64 -------------------------------------------------------
//
// A bowl L(w) = ½ Σ λᵢ xᵢ², xᵢ the distance from the bottom along the bowl's i-th own
// direction (an eigenvector of its Hessian) and λᵢ the curvature there. Each rule below
// is linear, so every direction moves on its own; the sharpest curvature is 1 here, so
// that the step η and ηλ_max are the same number.

/// The most steps a run takes.
const MAX: usize = 3000;
/// Every run starts this far from the bottom along the flattest direction, and `REST`
/// along each of the others.
const FLAT_START: f64 = -1.0;
const REST_START: f64 = 0.4;
/// The share of its starting loss a run is counted down to.
const TARGET: f64 = 1e-6;
/// A distance or velocity past this is a run that has flown off: it is held there, so
/// that every number stays finite (it is far outside every chart).
const BIG: f64 = 1e12;
/// How near a boundary of the map counts as on it (relative).
const EDGE: f64 = 1e-9;

/// The ranges of the knobs.
const ETA_MIN: f64 = 0.001;
const ETA_MAX: f64 = 4.4;
const BETA_MAX: f64 = 0.995;
const KAPPA_MAX: f64 = 1000.0;
const D_MIN: usize = 2;
const D_MAX: usize = 8;

/// The rule a run is made with.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Rule {
    /// Gradient descent: w ← w − η∇L.
    Plain,
    /// Heavy ball, as PyTorch's SGD writes it with dampening 0: v ← βv + ∇L(w),
    /// w ← w − ηv.
    Heavy,
    /// Nesterov's, as PyTorch writes it: b ← βb + g, w ← w − η(g + βb).
    Nesterov,
}

impl Rule {
    fn index(self) -> usize {
        match self {
            Rule::Plain => 0,
            Rule::Heavy => 1,
            Rule::Nesterov => 2,
        }
    }

    fn name(self) -> &'static str {
        match self {
            Rule::Plain => "plain descent",
            Rule::Heavy => "heavy ball",
            Rule::Nesterov => "Nesterov",
        }
    }

    fn color(self) -> Color {
        match self {
            Rule::Plain => SGD,
            Rule::Heavy => MOMENTUM,
            Rule::Nesterov => NESTEROV,
        }
    }
}

/// The bowl's curvatures, flattest first: d of them, evenly spread on a logarithmic
/// scale from 1/κ to 1.
fn curvatures(kappa: f64, d: usize) -> Vec<f64> {
    let d = d.max(1);
    let kappa = kappa.max(1.0);
    if d == 1 {
        return vec![1.0];
    }
    (0..d).map(|i| kappa.powf(-((d - 1 - i) as f64) / (d - 1) as f64)).collect()
}

/// Where a run starts, along each direction.
fn start(d: usize) -> Vec<f64> {
    (0..d).map(|i| if i == 0 { FLAT_START } else { REST_START }).collect()
}

/// A complex number (real part, imaginary part).
type C = (f64, f64);

fn modulus(z: C) -> f64 {
    z.0.hypot(z.1)
}

/// The two roots of z² − pz + q, the larger in size first.
fn roots(p: f64, q: f64) -> [C; 2] {
    let disc = p * p - 4.0 * q;
    if disc >= 0.0 {
        let r = disc.sqrt();
        // (the larger in size from p's own sign, the other from the product q, so that
        // nothing is lost to cancellation)
        let big = 0.5 * (p + if p >= 0.0 { r } else { -r });
        let small = if big != 0.0 { q / big } else { 0.0 };
        [(big, 0.0), (small, 0.0)]
    } else {
        let im = 0.5 * (-disc).sqrt();
        [(0.5 * p, im), (0.5 * p, -im)]
    }
}

/// The trace and determinant of one direction's step, with a = ηλ: its eigenvalues are
/// the roots of z² − (trace) z + det. (Plain descent's step is the number 1 − a; it is
/// written as the pair 1 − a and 0.)
fn trace_det(rule: Rule, a: f64, beta: f64) -> (f64, f64) {
    match rule {
        Rule::Plain => (1.0 - a, 0.0),
        Rule::Heavy => (1.0 + beta - a, beta),
        Rule::Nesterov => ((1.0 + beta) * (1.0 - a), beta * (1.0 - a)),
    }
}

/// One direction's step as a matrix, acting on (x, v): x its distance, v the momentum
/// buffer (for plain descent, a buffer that is never used). (The tests hold the
/// eigenvalues, the regions and the simulation to it.)
#[cfg(test)]
fn step_matrix(rule: Rule, eta: f64, lambda: f64, beta: f64) -> [[f64; 2]; 2] {
    let a = eta * lambda;
    match rule {
        Rule::Plain => [[1.0 - a, 0.0], [0.0, 0.0]],
        // v' = βv + λx, x' = x − ηv'
        Rule::Heavy => [[1.0 - a, -eta * beta], [lambda, beta]],
        // b' = βb + λx, x' = x − η(λx + βb')
        Rule::Nesterov => [[1.0 - a * (1.0 + beta), -eta * beta * beta], [lambda, beta]],
    }
}

fn eigenvalues(rule: Rule, a: f64, beta: f64) -> [C; 2] {
    let (t, d) = trace_det(rule, a, beta);
    roots(t, d)
}

/// The larger eigenvalue's size: what a direction is multiplied by per step, in the long run.
fn rate(rule: Rule, a: f64, beta: f64) -> f64 {
    modulus(eigenvalues(rule, a, beta)[0])
}

/// The edges of the map's regions, as values of a = ηλ for the momentum β: where the two
/// eigenvalues become equal and positive (critical), where they become equal again (heavy
/// ball: both −√β; Nesterov: both 0), and the limit, past which a direction grows.
#[derive(Clone, Copy, Debug)]
struct Edges {
    critical: f64,
    second: f64,
    limit: f64,
}

fn edges(rule: Rule, beta: f64) -> Edges {
    let r = beta.max(0.0).sqrt();
    match rule {
        Rule::Plain => Edges { critical: 1.0, second: 1.0, limit: 2.0 },
        Rule::Heavy => Edges { critical: (1.0 - r).powi(2), second: (1.0 + r).powi(2), limit: 2.0 + 2.0 * beta },
        Rule::Nesterov => Edges { critical: ((1.0 - beta) / (1.0 + beta)).powi(2), second: 1.0, limit: (2.0 + 2.0 * beta) / (1.0 + 2.0 * beta) },
    }
}

/// How a direction moves.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Kind {
    /// Two real eigenvalues, both positive: it slides in without crossing the bottom.
    Creeps,
    /// The two equal and positive: the edge between creeping and ringing.
    Critical,
    /// A complex pair: it overshoots, swings back and shrinks.
    Rings,
    /// The larger eigenvalue negative: it changes side at every step.
    Flips,
    /// Both eigenvalues 0: it is at the bottom after a step or two.
    Gone,
    /// The larger has size 1: it neither shrinks nor grows.
    OnLimit,
    /// The larger has size above 1: it grows.
    FliesOff,
}

impl Kind {
    fn word(self) -> &'static str {
        match self {
            Kind::Creeps => "creeps in",
            Kind::Critical => "critical",
            Kind::Rings => "rings",
            Kind::Flips => "flips",
            Kind::Gone => "gone at once",
            Kind::OnLimit => "on the limit",
            Kind::FliesOff => "flies off",
        }
    }
}

fn kind(rule: Rule, a: f64, beta: f64) -> Kind {
    let e = edges(rule, beta);
    let near = |x: f64| (a - x).abs() <= EDGE * (1.0 + x.abs());
    let [z1, z2] = eigenvalues(rule, a, beta);
    if modulus(z1) < 1e-12 && modulus(z2) < 1e-12 {
        Kind::Gone
    } else if near(e.limit) {
        Kind::OnLimit
    } else if a > e.limit {
        Kind::FliesOff
    } else if near(e.critical) {
        Kind::Critical
    } else if a < e.critical {
        Kind::Creeps
    } else if a < e.second && !near(e.second) {
        Kind::Rings
    } else {
        Kind::Flips
    }
}

/// The best fixed step and momentum for curvatures from 1/κ to 1, and the rate they
/// give every direction: β* = ((√κ − 1)/(√κ + 1))², η* = 4/(√L + √μ)², rate √β*.
fn best_pair(kappa: f64) -> (f64, f64, f64) {
    let s = kappa.max(1.0).sqrt();
    let r = (s - 1.0) / (s + 1.0);
    (4.0 / (1.0 + 1.0 / s).powi(2), r * r, r)
}

/// Plain descent's best fixed step for the same curvatures, 2/(μ + L), and its rate
/// (κ − 1)/(κ + 1).
fn best_plain(kappa: f64) -> (f64, f64) {
    let k = kappa.max(1.0);
    (2.0 / (1.0 + 1.0 / k), (k - 1.0) / (k + 1.0))
}

/// The first step from which `values` stay at or below `share` times their start, if
/// they get there within the run.
fn steps_to(values: &[f64], share: f64) -> Option<usize> {
    let first = *values.first()?;
    let line = share * first;
    let last = *values.last()?;
    if last.is_nan() || last > line {
        return None;
    }
    let above = values.iter().rposition(|v| v.is_nan() || *v > line);
    Some(above.map_or(0, |k| k + 1))
}

/// One rule's run on the bowl: the distance along each direction at each step,
/// `x[t * d + i]`, the loss at each step, and the steps it takes to bring the loss to
/// a millionth of its start.
#[derive(Clone, Debug, Default)]
struct Path {
    x: Vec<f64>,
    loss: Vec<f64>,
    count: Option<usize>,
}

impl Path {
    fn at(&self, t: usize, i: usize, d: usize) -> f64 {
        self.x.get(t.min(MAX) * d + i).copied().unwrap_or(0.0)
    }
}

/// `MAX` steps of `rule` from the start, along every direction.
fn simulate(rule: Rule, eta: f64, beta: f64, lams: &[f64], out: &mut Path) {
    let d = lams.len();
    out.x.clear();
    out.x.resize((MAX + 1) * d, 0.0);
    let x0 = start(d);
    for (i, (&lambda, &x)) in lams.iter().zip(&x0).enumerate() {
        let (mut x, mut v) = (x, 0.0f64);
        out.x[i] = x;
        for t in 1..=MAX {
            let g = lambda * x;
            match rule {
                Rule::Plain => x -= eta * g,
                Rule::Heavy => {
                    v = beta * v + g;
                    x -= eta * v;
                }
                Rule::Nesterov => {
                    v = beta * v + g;
                    x -= eta * (g + beta * v);
                }
            }
            (x, v) = (x.max(-BIG).min(BIG), v.max(-BIG).min(BIG));
            out.x[t * d + i] = x;
        }
    }
    out.loss.clear();
    out.loss.extend((0..=MAX).map(|t| 0.5 * lams.iter().enumerate().map(|(i, l)| l * out.x[t * d + i].powi(2)).sum::<f64>()));
    out.count = steps_to(&out.loss, TARGET);
}

/// The three rules' runs on one bowl, at one step and momentum.
#[derive(Clone, Debug, Default)]
struct Run {
    lams: Vec<f64>,
    eta: f64,
    beta: f64,
    plain_eta: f64,
    paths: [Path; 3],
}

impl Run {
    fn new(knobs: &Knobs, d: usize) -> Run {
        let mut run = Run::default();
        run.redo(knobs, d);
        run
    }

    fn redo(&mut self, knobs: &Knobs, d: usize) {
        self.lams = curvatures(knobs.kappa, d);
        (self.eta, self.beta, self.plain_eta) = (knobs.eta, knobs.beta, knobs.plain_eta);
        let [plain, heavy, nesterov] = &mut self.paths;
        simulate(Rule::Plain, knobs.plain_eta, 0.0, &self.lams, plain);
        simulate(Rule::Heavy, knobs.eta, knobs.beta, &self.lams, heavy);
        simulate(Rule::Nesterov, knobs.eta, knobs.beta, &self.lams, nesterov);
    }

    fn d(&self) -> usize {
        self.lams.len()
    }

    fn path(&self, rule: Rule) -> &Path {
        &self.paths[rule.index()]
    }

    /// ηλ of direction i, for a rule.
    fn a(&self, rule: Rule, i: usize) -> f64 {
        let eta = if rule == Rule::Plain { self.plain_eta } else { self.eta };
        eta * self.lams.get(i).copied().unwrap_or(1.0)
    }

    fn beta_of(&self, rule: Rule) -> f64 {
        if rule == Rule::Plain {
            0.0
        } else {
            self.beta
        }
    }

    fn rate(&self, rule: Rule, i: usize) -> f64 {
        rate(rule, self.a(rule, i), self.beta_of(rule))
    }

    fn kind(&self, rule: Rule, i: usize) -> Kind {
        kind(rule, self.a(rule, i), self.beta_of(rule))
    }

    /// The largest rate of a rule's directions, and those within a hair of it (the
    /// slowest, which set the run's rate).
    fn slowest(&self, rule: Rule) -> (f64, Vec<usize>) {
        let rates: Vec<f64> = (0..self.d()).map(|i| self.rate(rule, i)).collect();
        let most = rates.iter().copied().fold(0.0f64, f64::max);
        (most, // (a hair: where two eigenvalues meet, rounding parts them by about the square root
        // of the machine's precision)
        (0..self.d()).filter(|&i| rates[i] >= most - 1e-6 * most.max(1.0)).collect())
    }
}

// ----- the map: a picture of the rate, and where it is a given number ----------------

/// The map shows s = √(ηλ) across, from 0 to `S_MAX` (ηλ = 4.41), and β up from 0 to 1.
const S_MAX: f64 = 2.1;
const B_MAX: f64 = 1.0;
/// The picture's pixels.
const IMG_W: u32 = 336;
const IMG_H: u32 = 160;
/// The rates the map has contour lines for.
const LEVELS: [f64; 3] = [0.5, 0.7, 0.9];
/// How dark the darkest blue is: rates of 1 − 10⁻³ and more.
const V_MAX: f64 = 3.0;
/// The colour of a direction that grows.
const FLY: Color = Color::hex(0x3E1828);

/// The map's colour for a rate: pale for a direction gone at once, dark blue for one that
/// barely shrinks (by −log₁₀(1 − |z|), so that 0.9, 0.99 and 0.999 are evenly apart), wine
/// red for one that grows.
fn shade(rho: f64) -> Color {
    if rho.is_nan() || rho >= 1.0 {
        return FLY;
    }
    let v = -(1.0 - rho).max(1e-12).log10();
    let t = 1.0 - (v / V_MAX).min(1.0) as f32;
    Cmap::Slate.table()[table_index(t)]
}

/// Where the rate is `level`, as short segments in the map's own units (s, β):
/// marching squares on a fine grid of the rate.
fn contour(rule: Rule, level: f64) -> Segments {
    const NX: usize = 252;
    // (not a whole number of rows to any β = c² of `LEVELS`: a contour along a row of
    // the grid itself would be cut both ways)
    const NY: usize = 181;
    let s_at = |i: usize| S_MAX * i as f64 / NX as f64;
    let b_at = |j: usize| B_MAX * j as f64 / NY as f64;
    let mut v = vec![0.0; (NX + 1) * (NY + 1)];
    for j in 0..=NY {
        for i in 0..=NX {
            v[j * (NX + 1) + i] = rate(rule, s_at(i).powi(2), b_at(j)) - level;
        }
    }
    let mut out = Vec::new();
    for j in 0..NY {
        for i in 0..NX {
            let corners = [(i, j), (i + 1, j), (i + 1, j + 1), (i, j + 1)];
            let val = corners.map(|(i, j)| v[j * (NX + 1) + i]);
            let pos = corners.map(|(i, j)| (s_at(i), b_at(j)));
            let mut cuts: Vec<(f64, f64)> = Vec::with_capacity(4);
            for e in 0..4 {
                let (p, q) = (e, (e + 1) % 4);
                if (val[p] < 0.0) != (val[q] < 0.0) {
                    let t = val[p] / (val[p] - val[q]);
                    cuts.push((pos[p].0 + (pos[q].0 - pos[p].0) * t, pos[p].1 + (pos[q].1 - pos[p].1) * t));
                }
            }
            if cuts.len() >= 2 {
                out.push([cuts[0], cuts[1]]);
            }
            if cuts.len() == 4 {
                out.push([cuts[2], cuts[3]]);
            }
        }
    }
    out
}

/// Short segments in the map's own units (s, β).
type Segments = Vec<[(f64, f64); 2]>;

/// What the map of a rule is drawn with, made once for each rule.
struct Chart {
    contours: Vec<(f64, Segments)>,
}

impl Chart {
    fn of(rule: Rule) -> Chart {
        Chart { contours: LEVELS.iter().map(|&c| (c, contour(rule, c))).collect() }
    }
}

// ----- the settings, and how they glide ----------------------------------------------

/// The settings that glide from one experiment to the next: the step, the momentum, the
/// condition number and plain descent's own step.
#[derive(Clone, Copy, Debug, PartialEq)]
struct Knobs {
    eta: f64,
    beta: f64,
    kappa: f64,
    plain_eta: f64,
}

impl Knobs {
    fn to_vec(self) -> [f64; 4] {
        [self.eta, self.beta, self.kappa.max(1.0).ln(), self.plain_eta]
    }

    fn from_vec(v: [f64; 4]) -> Knobs {
        Knobs { eta: v[0], beta: v[1], kappa: v[2].exp(), plain_eta: v[3] }
    }
}

/// How fast settings glide: each of two eases one after the other covers two thirds of
/// what is left in this many seconds.
const GLIDE: f64 = 0.09;

/// Settings shown on their way to the ones asked for, through two eases; exactly there
/// in the end.
#[derive(Clone, Debug)]
struct Ease {
    target: Knobs,
    between: [f64; 4],
    shown: [f64; 4],
}

impl Ease {
    fn at(k: Knobs) -> Ease {
        Ease { target: k, between: k.to_vec(), shown: k.to_vec() }
    }

    fn snap(&mut self, k: Knobs) {
        *self = Ease::at(k);
    }

    fn arrived(&self) -> bool {
        self.shown == self.target.to_vec() && self.between == self.target.to_vec()
    }

    /// The settings as drawn: the target itself once arrived, not a copy through logs.
    fn now(&self) -> Knobs {
        if self.arrived() {
            self.target
        } else {
            Knobs::from_vec(self.shown)
        }
    }

    fn step(&mut self, dt: f64) {
        let to = self.target.to_vec();
        let k = 1.0 - (-dt.max(0.0) / GLIDE).exp();
        let mut far = false;
        for i in 0..4 {
            self.between[i] += (to[i] - self.between[i]) * k;
            self.shown[i] += (self.between[i] - self.shown[i]) * k;
            let near = 1e-6 * to[i].abs().max(1.0);
            far |= !((to[i] - self.shown[i]).abs() <= near && (to[i] - self.between[i]).abs() <= near);
        }
        if !far {
            (self.between, self.shown) = (to, to);
        }
    }
}

// ----- the experiments -----------------------------------------------------------------

struct Trial {
    label: &'static str,
    look_for: &'static str,
    rule: Rule,
    eta: f64,
    beta: f64,
    kappa: f64,
    d: usize,
    plain_best: bool,
    /// The direction followed in the chart of one direction (None: the slowest).
    follow: Option<usize>,
}

const KAPPA: f64 = 100.0;
const DIRS: usize = 5;

const TRIALS: [Trial; 6] = [
    Trial {
        label: "plain descent on a long valley",
        look_for: r"Momentum 0, so heavy ball is plain descent and the runs are one. The step $\eta = 1.9$ is just under plain descent's limit 2 (the sharpest curvature is 1). The five directions sit on the bottom of the map: the sharpest, at $\eta\lambda = 1.9$, flips side every step and shrinks by 0.9 (the zigzag across the valley); the flattest, at 0.019, creeps in by 0.981 a step (the crawl along it). The flattest carries the white ring: it sets the count, 278 steps to bring the loss to a millionth of its start.",
        rule: Rule::Heavy,
        eta: 1.9,
        beta: 0.0,
        kappa: KAPPA,
        d: DIRS,
        plain_best: false,
        follow: None,
    },
    Trial {
        label: "the same step with momentum 0.9",
        look_for: r"The same $\eta = 1.9$ with $\beta = 0.9$: every direction is now in the band where the eigenvalues are a complex pair, and every one shrinks by exactly $\sqrt{0.9} = 0.949$ a step, whatever its curvature (all five carry the white ring). The sharpest no longer flips: it rings, and for longer than plain descent's zigzag, which shrank by 0.9. The flattest speeds up from 0.981 to 0.949 a step, and the loss reaches a millionth in 131 steps instead of 278.",
        rule: Rule::Heavy,
        eta: 1.9,
        beta: 0.9,
        kappa: KAPPA,
        d: DIRS,
        plain_best: false,
        follow: None,
    },
    Trial {
        label: "the best pair for this bowl",
        look_for: r"For curvatures from $\lambda_{\min} = 0.01$ to $\lambda_{\max} = 1$ the best fixed pair is $\beta = ((\sqrt\kappa - 1)/(\sqrt\kappa + 1))^2 = 0.669$ and $\eta = 4/(\sqrt{\lambda_{\max}} + \sqrt{\lambda_{\min}})^2 = 3.31$. It puts the flattest direction on the band's left edge and the sharpest on its right edge: every direction shrinks by $\sqrt\beta = 0.818$ a step. Plain descent takes its own best step here, $2/(\lambda_{\min} + \lambda_{\max}) = 1.98$, and shrinks its worst two by 0.980. An error then shrinks $e$-fold in about $\sqrt\kappa/2 = 5$ steps against $\kappa/2 = 50$; to a millionth, 57 steps against 337. Not ten times fewer: momentum's loss first rises ninefold (at step 4) and is above plain descent's until step 19. The best pair gives the best rate in the long run, not the fastest start.",
        rule: Rule::Heavy,
        // (set from the formulas by `set_up`)
        eta: 0.0,
        beta: 0.0,
        kappa: KAPPA,
        d: DIRS,
        plain_best: true,
        follow: None,
    },
    Trial {
        label: "too much momentum",
        look_for: r"$\beta = 0.99$ at the same step: every direction rings, all at $\sqrt{0.99} = 0.995$ a step, slower than plain descent's slowest (0.981). In the band the rate is $\sqrt\beta$ whatever the step, so more momentum is slower there. 1409 steps to a millionth against 278, and on the way the loss climbs back above its start (to 1.27 times it, at step 12).",
        rule: Rule::Heavy,
        eta: 1.9,
        beta: 0.99,
        kappa: KAPPA,
        d: DIRS,
        plain_best: false,
        follow: None,
    },
    Trial {
        label: "a step plain descent cannot bear",
        look_for: r"$\eta = 3$: plain descent multiplies its sharpest direction by $1 - 3 = -2$ every step (the chart of one direction follows it): it flips and doubles, and the loss flies off. With $\beta = 0.9$ the limit is $\eta\lambda < 2 + 2\beta = 3.8$, every direction is in the band at 0.949 a step, and the loss reaches a millionth in 141 steps. But the first step is a plain one (nothing is kept yet), so the loss first rises 2.7-fold: momentum can raise the loss while it converges.",
        rule: Rule::Heavy,
        eta: 3.0,
        beta: 0.9,
        kappa: KAPPA,
        d: DIRS,
        plain_best: false,
        follow: Some(DIRS - 1),
    },
    Trial {
        label: "Nesterov at the same settings",
        look_for: r"Nesterov's momentum takes the gradient a step ahead. At $\eta = 1.9$ and $\beta = 0.9$, where heavy ball (faint) converges in 131 steps, its limit is $\eta\lambda < (2 + 2\beta)/(1 + 2\beta) = 1.357$: lower even than plain descent's 2, at the same step. Its sharpest direction (1.9) flies off, growing 2.10-fold a step. Below its limit its band of complex pairs ends at $\eta\lambda = 1$, and there it shrinks by $\sqrt{\beta(1 - \eta\lambda)}$, faster where the curvature is larger: its other four directions shrink by 0.940 down to 0.599 a step, against heavy ball's 0.949. Drag the step down to 1.2 and both converge: Nesterov in 81 steps, heavy ball in 132.",
        rule: Rule::Nesterov,
        eta: 1.9,
        beta: 0.9,
        kappa: KAPPA,
        d: DIRS,
        plain_best: false,
        follow: Some(DIRS - 1),
    },
];

/// The experiment the scene opens with, at rest.
const OPENING: usize = 1;

// ----- the scene -------------------------------------------------------------------

/// How a film of a run plays: its first `SLOW_STEPS` steps at `SLOW_RATE` a second, a
/// ball tracing each (they are the long ones, where the paths zigzag), then the rest of
/// the run in `FAST` seconds, speeding up as the steps shorten (see `film_step`); and how
/// long the run drawn takes to fade out before a film.
const SLOW_STEPS: f64 = 10.0;
const SLOW_RATE: f64 = 4.0;
const FAST: f64 = 3.5;
const FADE: f64 = 0.35;
/// How fast the step drawn follows the end of the run, at rest.
const FOLLOW: f64 = 0.07;

/// How long a film of `span` steps lasts.
fn film_length(span: f64) -> f64 {
    let slow = span.max(0.0).min(SLOW_STEPS);
    slow / SLOW_RATE + if span > SLOW_STEPS { FAST } else { 0.0 }
}

/// Where a film of `span` steps is after `tau` seconds: `SLOW_RATE` steps a second for
/// the first `SLOW_STEPS`, then a rate that grows exponentially from there so as to end
/// at `span` after `FAST` seconds more (evenly, if the slow rate is enough).
fn film_step(span: f64, tau: f64) -> f64 {
    let span = span.max(0.0);
    let slow = span.min(SLOW_STEPS);
    let t = tau.max(0.0);
    if t * SLOW_RATE <= slow {
        return t * SLOW_RATE;
    }
    let (rest, t) = (span - slow, (t - slow / SLOW_RATE).min(FAST));
    if rest.is_nan() || rest <= SLOW_RATE * FAST {
        return slow + rest * t / FAST;
    }
    // r c (e^{T/c} − 1) = rest, for the time constant c (it falls as c grows)
    let reach = |c: f64| SLOW_RATE * c * ((FAST / c).exp() - 1.0);
    let (mut lo, mut hi) = (0.05f64, 1e4f64);
    for _ in 0..80 {
        let mid = (lo * hi).sqrt();
        if reach(mid) > rest {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    let c = (lo * hi).sqrt();
    slow + (SLOW_RATE * c * ((t / c).exp() - 1.0)).min(rest)
}

/// The run drawn when a film was asked for, kept as it was while it fades out: what
/// is remade under it as the settings glide would otherwise move with them.
#[derive(Clone, Debug)]
struct Frozen {
    run: Run,
    span: usize,
    rules: Vec<Rule>,
    shown: f64,
    follow: usize,
}

/// Where the film is.
#[derive(Clone, Copy, Debug, PartialEq)]
enum Film {
    /// The whole run drawn (the step drawn follows its end).
    Rest,
    /// The run that was drawn fading out before the film starts (as it was: see
    /// `Frozen`), with this many seconds left.
    Fade { left: f64 },
    /// No run drawn, while the settings and the charts' ranges arrive, so that the
    /// film is drawn on still axes and its span is theirs.
    Settle,
    /// Playing a run of `span` steps, `tau` seconds in.
    Play { tau: f64, span: usize },
    /// Stopped part of the way.
    Paused { tau: f64, span: usize },
}

/// How long after the last change by hand the charts take their new steps (seconds).
const HOLD: f64 = 0.3;

/// The handle's id.
const HANDLE: u32 = 1;

pub struct Momentum {
    rule: Rule,
    d: usize,
    plain_best: bool,
    ease: Ease,
    run: Run,
    /// The settings the run was made with, and its number of directions.
    made: (Knobs, usize),
    /// The direction followed (None: the slowest).
    follow: Option<usize>,
    /// The film, the run that was drawn when it was asked for, and the step drawn (at
    /// rest, through two eases).
    film: Film,
    frozen: Option<Frozen>,
    between: f64,
    shown: f64,
    /// The steps the charts run to and the chart of one direction's height, as drawn:
    /// they glide to new ones.
    span: plot::Glide,
    height: plot::Glide,
    /// While settings are being changed by hand: the steps the charts keep, and for how
    /// many seconds more.
    held: Option<(usize, f64)>,
    /// Where the steps to a millionth are marked on the chart of the losses, for each
    /// rule (past its end where there is no count), gliding there.
    marks: plot::Glide,
    /// The map's picture and its lines, for the rule they were made for.
    image: Option<Image>,
    painted: Option<Rule>,
    chart: Chart,
}

impl Momentum {
    fn new() -> Momentum {
        let knobs = Knobs { eta: 1.0, beta: 0.5, kappa: KAPPA, plain_eta: 1.0 };
        let mut scene = Momentum {
            rule: Rule::Heavy,
            d: DIRS,
            plain_best: false,
            ease: Ease::at(knobs),
            run: Run::new(&knobs, DIRS),
            made: (knobs, DIRS),
            follow: None,
            film: Film::Rest,
            frozen: None,
            between: 0.0,
            shown: 0.0,
            span: plot::Glide::default(),
            height: plot::Glide::default(),
            marks: plot::Glide::default(),
            held: None,
            image: None,
            painted: None,
            chart: Chart::of(Rule::Heavy),
        };
        scene.set_up(OPENING);
        // (it opens at rest, the whole run drawn)
        scene.ease.snap(scene.ease.target);
        scene.remake();
        scene.film = Film::Rest;
        let span = scene.span_wanted();
        (scene.between, scene.shown) = (span as f64, span as f64);
        scene.span.snap(&[span as f32]);
        scene.height.snap(&[scene.height_wanted()]);
        scene.marks.snap(&scene.marks_wanted());
        scene
    }

    /// The settings asked for, with plain descent's step worked out.
    fn knobs(&self, eta: f64, beta: f64, kappa: f64) -> Knobs {
        let eta = eta.max(ETA_MIN).min(ETA_MAX);
        let beta = beta.max(0.0).min(BETA_MAX);
        let kappa = kappa.max(1.0).min(KAPPA_MAX);
        let plain_eta = if self.plain_best { best_plain(kappa).0 } else { eta };
        Knobs { eta, beta, kappa, plain_eta }
    }

    /// Experiment `i`: every setting it depends on, then its film from the start.
    fn set_up(&mut self, i: usize) {
        self.freeze();
        let t = &TRIALS[i.min(TRIALS.len() - 1)];
        let (mut eta, mut beta) = (t.eta, t.beta);
        if t.label == TRIALS[2].label {
            let (e, b, _) = best_pair(t.kappa);
            (eta, beta) = (e, b);
        }
        self.set_rule(t.rule);
        self.d = t.d;
        self.plain_best = t.plain_best;
        self.follow = t.follow;
        self.ease.target = self.knobs(eta, beta, t.kappa);
        self.play_from_start();
    }

    fn set_rule(&mut self, rule: Rule) {
        if rule != self.rule {
            self.rule = rule;
            self.chart = Chart::of(rule);
        }
    }

    /// A setting changed by hand: there at once, and the whole run drawn.
    fn by_hand(&mut self, eta: f64, beta: f64, kappa: f64) {
        let k = self.knobs(eta, beta, kappa);
        self.ease.snap(k);
        self.stop_film();
        // the charts keep their steps while the hand is moving, and take the new ones
        // when it rests: axes that changed with every move would run to and fro under it
        let span = self.held.map_or_else(|| self.span_wanted(), |(span, _)| span);
        self.held = Some((span, HOLD));
    }

    /// A setting changed by a switch: the picture glides to it, the whole run drawn.
    fn by_switch(&mut self, eta: f64, beta: f64, kappa: f64) {
        self.ease.target = self.knobs(eta, beta, kappa);
        self.stop_film();
    }

    /// Back to the whole run.
    fn stop_film(&mut self) {
        self.film = Film::Rest;
        self.frozen = None;
    }

    fn play_from_start(&mut self) {
        if self.frozen.is_none() {
            self.freeze();
        }
        self.film = Film::Fade { left: FADE };
    }

    /// Keep the run drawn as it is, to fade out.
    fn freeze(&mut self) {
        self.frozen = Some(Frozen { run: self.run.clone(), span: self.span_wanted(), rules: self.drawn_rules(), shown: self.shown, follow: self.followed() });
    }

    /// The run to draw, the steps it is drawn to, the rules drawn and the direction
    /// followed: the frozen one while it fades out, none while the axes settle.
    fn drawn(&self) -> Option<(&Run, usize, Vec<Rule>, f64, usize)> {
        match (&self.film, &self.frozen) {
            (Film::Fade { .. }, Some(fr)) => Some((&fr.run, fr.span, fr.rules.clone(), fr.shown, fr.follow)),
            (Film::Settle, _) => None,
            _ => Some((&self.run, self.span_wanted(), self.drawn_rules(), self.shown, self.followed())),
        }
    }

    /// Whether the film is in its slow start, where a ball traces each step.
    fn ball(&self) -> bool {
        matches!(self.film, Film::Play { .. } | Film::Paused { .. }) && self.shown < SLOW_STEPS
    }

    /// How strongly the runs are drawn: fading out before a film.
    fn fade(&self) -> f32 {
        match self.film {
            Film::Fade { left } => {
                let u = (left / FADE).max(0.0).min(1.0) as f32;
                u * u * (3.0 - 2.0 * u)
            }
            _ => 1.0,
        }
    }

    fn playing(&self) -> bool {
        matches!(self.film, Film::Fade { .. } | Film::Settle | Film::Play { .. })
    }

    /// Play or pause; play again from the start once the film is over.
    fn toggle_play(&mut self) {
        self.film = match self.film {
            Film::Play { tau, span } => Film::Paused { tau, span },
            Film::Paused { tau, span } => Film::Play { tau, span },
            Film::Fade { .. } | Film::Settle => {
                self.frozen = None;
                Film::Paused { tau: 0.0, span: self.span_wanted() }
            }
            Film::Rest => {
                self.freeze();
                Film::Fade { left: FADE }
            }
        };
    }

    /// The run again, if the settings drawn have moved since it was made.
    fn remake(&mut self) {
        let now = (self.ease.now(), self.d);
        if now != self.made {
            self.run.redo(&now.0, now.1);
            self.made = now;
        }
        if self.follow.is_some_and(|i| i >= self.d) {
            self.follow = None;
        }
    }

    /// The direction the chart of one direction follows: the one chosen, or else the
    /// slowest (the flattest of them, if several are as slow).
    fn followed(&self) -> usize {
        match self.follow {
            Some(i) if i < self.run.d() => i,
            _ => self.run.slowest(self.rule).1.first().copied().unwrap_or(0),
        }
    }

    /// The steps the charts show: past where the slower of the two runs drawn reaches a
    /// millionth of its start, rounded up.
    fn span_wanted(&self) -> usize {
        if let Some((span, _)) = self.held {
            return span;
        }
        self.span_of_run()
    }

    /// The span the run itself asks for.
    fn span_of_run(&self) -> usize {
        let counts = [self.run.path(Rule::Plain).count, self.run.path(self.rule).count];
        match counts.iter().flatten().max() {
            Some(&n) => round_up((n as f64 * 1.12).ceil() as usize).max(30).min(MAX),
            None => 60,
        }
    }

    /// Where the counts are marked: at each rule's count, or past the chart's end.
    fn marks_wanted(&self) -> Vec<f32> {
        let past = self.span_wanted() as f32 * 1.1 + 1.0;
        [Rule::Plain, Rule::Heavy, Rule::Nesterov].iter().map(|r| self.run.path(*r).count.map_or(past, |n| n as f32)).collect()
    }

    /// How high the chart of one direction goes: the largest share of its start the
    /// runs drawn reach within the span, at least 1.05 and at most 3.2.
    fn height_wanted(&self) -> f32 {
        let (d, i, span) = (self.run.d(), self.followed(), self.span_wanted());
        let x0 = start(d)[i].abs().max(1e-12);
        let mut most = 1.0f64;
        for rule in self.drawn_rules() {
            let path = self.run.path(rule);
            for t in 0..=span {
                most = most.max(path.at(t, i, d).abs() / x0);
            }
        }
        (most * 1.08).max(1.05).min(3.2) as f32
    }

    /// The rules whose runs are drawn: the rule chosen (and heavy ball beside Nesterov),
    /// and plain descent, thinner, on top (so that both show where they are one, with
    /// no momentum).
    fn drawn_rules(&self) -> Vec<Rule> {
        match self.rule {
            Rule::Nesterov => vec![Rule::Heavy, Rule::Nesterov, Rule::Plain],
            _ => vec![self.rule, Rule::Plain],
        }
    }

    /// Move everything on: `dt` seconds of the scene's time (the film) and `real_dt` of
    /// the clock's (the glides, which answer the visitor and go on while time stands still).
    fn advance(&mut self, dt: f32, real_dt: f32) {
        let (dt, real) = (dt.max(0.0) as f64, real_dt.max(0.0) as f64);
        self.held = self.held.map(|(span, left)| (span, left - real)).filter(|(_, left)| *left > 0.0);
        self.ease.step(real);
        self.remake();
        let span = self.span_wanted();
        match self.film {
            Film::Fade { left } => {
                // the run that was drawn fades out as it was (a path run backwards would
                // send its head racing to and fro along every zigzag)
                let left = (left - real).max(0.0);
                if left == 0.0 {
                    (self.film, self.frozen) = (Film::Settle, None);
                    (self.between, self.shown) = (0.0, 0.0);
                } else {
                    self.film = Film::Fade { left };
                }
            }
            Film::Settle => {
                let arrived = self.ease.arrived()
                    && self.span.shown == [span as f32]
                    && self.height.shown == [self.height_wanted()];
                if arrived {
                    self.film = Film::Play { tau: 0.0, span };
                }
            }
            Film::Play { tau, span: film_span } => {
                let tau = tau + dt;
                let at = film_step(film_span as f64, tau);
                (self.between, self.shown) = (at, at);
                self.film = if tau >= film_length(film_span as f64) { Film::Rest } else { Film::Play { tau, span: film_span } };
            }
            Film::Paused { tau, span: film_span } => {
                let at = film_step(film_span as f64, tau);
                (self.between, self.shown) = (at, at);
            }
            Film::Rest => {
                let to = span as f64;
                let e = 1.0 - (-real / FOLLOW).exp();
                self.between += (to - self.between) * e;
                self.shown += (self.between - self.shown) * e;
                if ((to - self.between).abs() < 1e-3 && (to - self.shown).abs() < 1e-3) || !self.shown.is_finite() || !self.between.is_finite() {
                    (self.between, self.shown) = (to, to);
                }
            }
        }
        // (the charts' ranges stay as they were while the old run fades out in them)
        if !matches!(self.film, Film::Fade { .. }) {
            self.span.follow(&[span as f32], real_dt, false);
            self.height.follow(&[self.height_wanted()], real_dt, false);
        }
        self.marks.follow(&self.marks_wanted(), real_dt, false);
    }

    fn moving(&self) -> bool {
        self.held.is_some()
            || self.playing()
            || !self.ease.arrived()
            || (self.film == Film::Rest && (self.shown != self.span_wanted() as f64 || self.between != self.span_wanted() as f64))
            || self.span.shown != [self.span_wanted() as f32]
            || self.height.shown != [self.height_wanted()]
            || self.marks.shown != self.marks_wanted()
    }
}

/// The part of the valley drawn, the same for every run so that runs can be set side by
/// side (u from, to: along the flattest direction; w from, to: along the sharpest): round
/// the start and the bottom, with room for an overshoot. A run that goes further leaves
/// the picture.
const VALLEY: [f64; 4] = [-1.3, 1.1, -1.05, 1.05];

/// A number of steps rounded up to a round one.
fn round_up(n: usize) -> usize {
    let m = match n {
        0..=100 => 10,
        101..=500 => 50,
        501..=1500 => 100,
        _ => 250,
    };
    n.div_ceil(m) * m
}

// ----- where things go -----------------------------------------------------------------

#[derive(Clone, Debug)]
struct Layout {
    narrow: bool,
    text: f32,
    small: f32,
    /// Play, from the start, heavy ball, Nesterov.
    buttons: Vec<Rect>,
    map: Option<Rect>,
    bowl: Option<Rect>,
    dist: Option<Rect>,
    loss: Option<Rect>,
}

/// The room round the map for its title, colour key and axes.
const MAP_ABOVE: f32 = 76.0;
const MAP_BELOW: f32 = 40.0;
const MAP_LEFT: f32 = 40.0;
/// The room above a chart for its title and key, and below it for its axis.
const CHART_ABOVE: f32 = 46.0;
const CHART_BELOW: f32 = 30.0;

const BUTTON_WORDS: [&str; 4] = ["▶ play the run", "↺ from the start", "heavy ball", "Nesterov"];

impl Layout {
    fn of(f: &Frame, view: Vec2, inset: f32) -> Layout {
        let view = view.max(Vec2::ONE);
        let narrow = view.x < 640.0;
        let (text, small) = if narrow { (11.5, 10.5) } else { (12.5, 11.5) };
        let side = if narrow { 8.0 } else { 16.0 };
        let top = inset.max(0.0) + if narrow { 6.0 } else { 12.0 };
        let h = if narrow { 26.0 } else { 28.0 };
        let mut lay = Layout { narrow, text, small, buttons: Vec::new(), map: None, bowl: None, dist: None, loss: None };
        if view.x < 240.0 || view.y < 180.0 {
            return lay;
        }
        lay.buttons = if narrow {
            plot::even_row(4, side, view.x - side, top, h, 5.0)
        } else {
            let mut rects = plot::button_row(f, &BUTTON_WORDS[..2], side, top, h, text, 8.0);
            let x = rects.last().map_or(side, |r| r.1.x) + 24.0;
            rects.extend(plot::button_row(f, &BUTTON_WORDS[2..], x, top, h, text, 4.0));
            rects
        };
        let body = top + h + if narrow { 10.0 } else { 14.0 };
        let ok = |r: Rect| (r.1.x - r.0.x >= 60.0 && r.1.y - r.0.y >= 40.0).then_some(r);
        if narrow {
            // a phone: the map, the valley, and the two charts side by side under them
            let room = view.y - body - 4.0;
            let map_h = (room * 0.36).min((view.x - 2.0 * side) * 0.8).floor();
            let map = (Vec2::new(side + MAP_LEFT - 8.0, body + MAP_ABOVE - 8.0), Vec2::new(view.x - side - 6.0, body + MAP_ABOVE - 8.0 + map_h));
            lay.map = ok(map);
            let bowl_top = map.1.y + MAP_BELOW + CHART_ABOVE - 10.0;
            let bowl_h = (room * 0.2).floor();
            let bowl = (Vec2::new(side + 6.0, bowl_top), Vec2::new(view.x - side - 6.0, bowl_top + bowl_h));
            lay.bowl = ok(bowl);
            let charts_top = bowl.1.y + 14.0 + CHART_ABOVE;
            let bottom = view.y - CHART_BELOW + 4.0;
            let half = ((view.x - 2.0 * side) * 0.5).floor();
            lay.dist = ok((Vec2::new(side + 30.0, charts_top), Vec2::new(side + half - 8.0, bottom)));
            lay.loss = ok((Vec2::new(side + half + 32.0, charts_top), Vec2::new(view.x - side - 4.0, bottom)));
            return lay;
        }
        // wider: the map over the chart of one direction on the left, the valley over
        // the losses on the right (the valley as tall as its long shape wants)
        let left_w = (view.x * 0.56).floor();
        let room = view.y - body;
        let top_h = (room * 0.6).floor();
        lay.map = ok((Vec2::new(side + MAP_LEFT + 6.0, body + MAP_ABOVE), Vec2::new(left_w - 20.0, body + top_h - MAP_BELOW)));
        lay.dist = ok((Vec2::new(side + MAP_LEFT + 6.0, body + top_h + 10.0 + CHART_ABOVE), Vec2::new(left_w - 20.0, view.y - CHART_BELOW - 10.0)));
        let (x0, x1) = (left_w + 12.0, view.x - side - 8.0);
        let bowl_h = ((x1 - x0) * 0.62).min(room * 0.48).floor();
        let bowl = (Vec2::new(x0, body + CHART_ABOVE), Vec2::new(x1, body + CHART_ABOVE + bowl_h));
        lay.bowl = ok(bowl);
        lay.loss = ok((Vec2::new(left_w + 50.0, bowl.1.y + 18.0 + CHART_ABOVE), Vec2::new(x1, view.y - CHART_BELOW - 10.0)));
        lay
    }
}

/// Kinds of line with a batch each (see `plot::apart`), so that one whose number of
/// points changes leaves the others to be followed.
const OUTLINES: u8 = plot::OWN;
const CONTOURS: u8 = plot::OWN + 1;
const LIMITS: u8 = plot::OWN + 2;
const PLAIN_LIMIT: u8 = plot::OWN + 3;
const LINE: u8 = plot::OWN + 4;
const RINGS: u8 = plot::OWN + 5;
const DROP: u8 = plot::OWN + 6;
const LEVEL_LINES: u8 = plot::OWN + 7;
const BOWL_PATHS: u8 = plot::OWN + 8;
const DIST_SERIES: u8 = plot::OWN + 11;
const LOSS_SERIES: u8 = plot::OWN + 14;
const LOSS_MARKS: u8 = plot::OWN + 17;

/// Every kind of line, in the order its batch is made in every frame. (Each is given an
/// unseen point off the picture: the drawing list lets a batch go that is left empty
/// for a frame and makes it again at the end of the list, which would put the batches
/// in another order whenever a run's lines come or go.)
fn in_order(f: &mut Frame) {
    let off = f.hud(Vec2::splat(-40.0));
    let solid = [plot::AXES, plot::CURVES, plot::MARKS, OUTLINES, CONTOURS, LIMITS, LINE, RINGS, LEVEL_LINES];
    let runs = (0..3).flat_map(|n| [BOWL_PATHS + n, DIST_SERIES + n, LOSS_SERIES + n]);
    for kind in solid.into_iter().chain(runs) {
        let batch = plot::apart(f, kind, false);
        batch.push(off, 1.0, Color::CLEAR);
        batch.brk();
    }
    for kind in [PLAIN_LIMIT, DROP, LOSS_MARKS] {
        let batch = plot::apart(f, kind, true);
        batch.push(off, 1.0, Color::CLEAR);
        batch.brk();
    }
}

/// The world rectangle under a rectangle of the screen (logical pixels).
fn world_rect(f: &Frame, (min, max): Rect) -> (Vec2, Vec2) {
    let (lo, hi) = f.cam.flat_bounds();
    let view = f.view_size().max(Vec2::ONE);
    let to = |p: Vec2| Vec2::new(lo.x + p.x / view.x * (hi.x - lo.x), hi.y - p.y / view.y * (hi.y - lo.y));
    let (a, b) = (to(min), to(max));
    (Vec2::new(a.x, b.y), Vec2::new(b.x, a.y))
}

/// The dark backing round a picture drawn on the plane (which lies behind every panel):
/// the outer rectangle less the hole.
fn backing_round(f: &mut Frame, (o0, o1): Rect, (h0, h1): Rect) {
    let (h0, h1) = (h0.max(o0), h1.min(o1));
    if !(h1.x > h0.x && h1.y > h0.y) {
        hud_panel(f, o0, o1);
        return;
    }
    hud_panel(f, o0, Vec2::new(o1.x, h0.y));
    hud_panel(f, Vec2::new(o0.x, h1.y), o1);
    hud_panel(f, Vec2::new(o0.x, h0.y), Vec2::new(h0.x, h1.y));
    hud_panel(f, Vec2::new(h1.x, h0.y), Vec2::new(o1.x, h1.y));
}

/// A circle in the batch of its kind.
fn ring(f: &mut Frame, kind: u8, centre: Vec2, radius: f32, width: f32, color: Color) {
    let pts: Vec<Vec2> = (0..=40).map(|i| centre + Vec2::from_angle(i as f32 * TAU / 40.0) * radius).collect();
    plot::path_of(f, kind, &pts, width, color);
}

/// A number of steps as words.
fn steps(n: usize) -> String {
    if n == 1 {
        "1 step".to_string()
    } else {
        format!("{n} steps")
    }
}

/// A count as words: "131 steps", or "more than 3000 steps", or that it flies off.
fn count_words(path: &Path) -> String {
    match path.count {
        Some(n) => steps(n),
        None if path.loss.last().is_some_and(|l| *l > path.loss[0]) => "never: it flies off".to_string(),
        None => format!("more than {MAX} steps"),
    }
}

impl Scene for Momentum {
    fn view(&self) -> View {
        View::Flat { center: Vec2::ZERO, half_height: 1.0 }
    }

    fn settings(&self, s: &mut ViewSettings) {
        s.grid = false;
        s.bloom = s.bloom.min(0.2);
    }

    fn animated(&self) -> bool {
        self.moving()
    }

    fn pointer_hint(&self) -> Option<&'static str> {
        Some("drag the large dot on the map    click a dot to follow it    buttons: click")
    }

    fn ui(&mut self, ui: &mut egui::Ui, _: &Gfx) {
        ui::note(ui, r"Gradient descent forgets its last step. Momentum keeps a share $\beta$ of it: the steps build up in a velocity $v$, as a heavy ball gathers speed rolling downhill. Here it is written as PyTorch's SGD writes it (heavy-ball momentum, no dampening), with step $\eta$:");
        ui::equation(ui, r"v_{t+1} = \beta v_t + \nabla L(w_t), \qquad w_{t+1} = w_t - \eta\, v_{t+1}");
        ui::note(ui, r"On a bowl each of its own directions (an eigenvector of its Hessian, of curvature $\lambda$) still moves on its own, but with two numbers, its distance $x$ from the bottom and its velocity $v$, which a step multiplies by a matrix:");
        ui::equation(ui, r"\begin{bmatrix} x_{t+1} \\ v_{t+1} \end{bmatrix} = \begin{bmatrix} 1 - \eta\lambda & -\eta\beta \\ \lambda & \beta \end{bmatrix} \begin{bmatrix} x_t \\ v_t \end{bmatrix}");
        ui::note(ui, r"Its two eigenvalues $z_1$, $z_2$ say everything, as $1 - \eta\lambda$ did without momentum. They depend only on $\eta\lambda$ and $\beta$, and after many steps the direction shrinks by $|z|$, the larger one's size, a step. The map colours every $\eta\lambda$ and $\beta$ by $|z|$; the bowl's directions are its dots.");
        ui::section(ui, "Things to try");
        for (i, t) in TRIALS.iter().enumerate() {
            if ui::experiment(ui, t.label, t.look_for) {
                self.set_up(i);
            }
        }
        ui::section(ui, "Momentum");
        let mut choice = if self.rule == Rule::Nesterov { 1 } else { 0 };
        if ui::choice(ui, &["heavy ball", "Nesterov"], &mut choice) {
            self.set_rule(if choice == 1 { Rule::Nesterov } else { Rule::Heavy });
            self.stop_film();
        }
        ui::note(ui, r"Nesterov's momentum takes the gradient a step ahead. PyTorch writes it in a form that moves the same way on a bowl, $b \leftarrow \beta b + g$, $w \leftarrow w - \eta\,(g + \beta b)$, and that is the form used here. With it, heavy ball at the same settings is drawn faint.");
        let t = self.ease.target;
        let mut eta = t.eta as f32;
        if ui::slider_log(ui, r"step $\eta$", &mut eta, ETA_MIN as f32..=ETA_MAX as f32) {
            self.by_hand(eta as f64, t.beta, t.kappa);
        }
        let t = self.ease.target;
        let mut beta = t.beta as f32;
        if ui::slider(ui, r"momentum $\beta$", &mut beta, 0.0..=BETA_MAX as f32) {
            self.by_hand(t.eta, beta as f64, t.kappa);
        }
        ui::note(ui, r"Or drag the large dot on the map: across for $\eta$ (the line of dots stretches from 0), up and down for $\beta$.");
        let mut best = self.plain_best;
        if ui::checkbox(ui, &mut best, "plain descent at its own best step").changed() {
            self.plain_best = best;
            let t = self.ease.target;
            self.by_switch(t.eta, t.beta, t.kappa);
        }
        ui::note(ui, r"Off, plain descent takes the same $\eta$, so that the two differ only by momentum; on, it takes the best fixed step for this bowl, $2/(\lambda_{\min} + \lambda_{\max})$.");
        ui::section(ui, "The bowl");
        let t = self.ease.target;
        let mut kappa = t.kappa as f32;
        if ui::slider_log(ui, r"$\kappa = \lambda_{\max} / \lambda_{\min}$", &mut kappa, 1.0..=KAPPA_MAX as f32) {
            self.by_hand(t.eta, t.beta, kappa as f64);
        }
        let mut d = self.d;
        if ui::slider_int(ui, "directions", &mut d, D_MIN..=D_MAX) {
            self.d = d;
            self.stop_film();
        }
        ui::note(ui, r"The curvatures are spread evenly on a log scale from $1/\kappa$ to 1. Every run starts 1 from the bottom along the flattest direction and 0.4 along each of the others.");

        ui::section(ui, "Reading the picture");
        ui::key(
            ui,
            &[
                (MOMENTUM, r"pink: heavy ball. On the map, its directions on a line at height $\beta$ (the large dot, the sharpest, can be dragged); below, its path and its loss"),
                (NESTEROV, "yellow dots and lines: Nesterov's momentum, the same"),
                (SGD, r"cyan: plain descent; on the map its directions lie along the bottom, at $\beta = 0$"),
                (pal::WHITE, "white ring: the slowest direction, which sets the run's rate. White lines: the edges of the regions. Dashed: the direction the chart of one direction follows (click a dot for another)"),
                (LIMIT, r"yellow line: the limit, $|z| = 1$; dashed, plain descent's, $\eta\lambda = 2$"),
                (shade(0.9), r"the map's colour: $|z|$, pale where a direction is gone at once, dark where it barely shrinks, with thin lines through equal values; wine red: $|z| > 1$, it flies off"),
            ],
        );

        ui::section(ui, "Words");
        ui::term(ui, "Eigenvalues of a step:", r"the numbers $z$ with $Mu = zu$ for the matrix $M$ of a step. Here they solve $z^2 - (1 + \beta - \eta\lambda)\,z + \beta = 0$, and over many steps a direction's distance shrinks, or grows, by $|z|$ a step, the larger one's size (a complex pair turns it too, so it swings about that rate).");
        ui::term(ui, "Creeps in, rings, flips:", r"two real eigenvalues, both positive: it slides in without crossing the bottom. A complex pair $r(\cos\theta \pm i \sin\theta)$: it turns by $\theta$ a step as it shrinks by $r$, overshooting and swinging back. The larger one negative: it changes side every step.");
        ui::term(ui, "Critical:", r"the two eigenvalues equal, at $\eta\lambda = (1 - \sqrt\beta)^2$, the edge between creeping in and ringing. They are equal again at $(1 + \sqrt\beta)^2$, both $-\sqrt\beta$.");
        ui::term(ui, r"Condition number $\kappa$:", r"the sharpest curvature over the flattest, $\lambda_{\max}/\lambda_{\min}$.");
        ui::term(ui, "Momentum as an average:", r"$v_{t+1} = g_t + \beta g_{t-1} + \beta^2 g_{t-2} + \cdots$ with $g_t = \nabla L(w_t)$, the recent gradients weighted by powers of $\beta$: about the last $1/(1 - \beta)$ of them count. On a steady slope $g$ it builds up to $g/(1 - \beta)$, a step $1/(1 - \beta)$ times a plain one. With noisy gradients the averaging matters too (later scenes).");
        ui::term(ui, "The other way of writing it:", r"as an average, $m \leftarrow \beta m + (1 - \beta)\,g$, $w \leftarrow w - \eta' m$, the form of Adam's first moment. It takes the same steps when $\eta' = \eta/(1 - \beta)$, so its limit reads $\eta'\lambda < (2 + 2\beta)/(1 - \beta)$: 38 for $\beta = 0.9$.");

        ui::section(ui, "Numbers");
        let run = &self.run;
        let rule = self.rule;
        ui::readout(ui, r"$\eta$, $\beta$, $\kappa$", format!("${}$, ${}$, ${}$", num(run.eta, 3), num(run.beta, 3), trim_big(self.ease.now().kappa)));
        for i in 0..run.d() {
            ui::readout_colored(
                ui,
                &format!(r"$\eta\lambda_{} = {}$", i + 1, num(run.a(rule, i), 4)),
                format!(r"{}, $|z| = {}$", run.kind(rule, i).word(), num(run.rate(rule, i), 4)),
                rule.color(),
            );
        }
        let limit = edges(rule, run.beta).limit;
        ui::readout(ui, &format!("{}'s limit", rule.name()), format!(r"$\eta\lambda < {}$", num(limit, 3)));
        for r in self.drawn_rules() {
            ui::readout_colored(ui, &format!("{}: to a millionth", r.name()), count_words(run.path(r)), r.color());
        }
        ui::readout(ui, "plain descent's step", format!("${}$", num(run.plain_eta, 3)));
        let kappa = self.ease.now().kappa;
        let (e, b, r) = best_pair(kappa);
        ui::readout(ui, "the best pair for this bowl", format!(r"$\eta = {}$, $\beta = {}$: ${}$", num(e, 3), num(b, 3), num(r, 3)));
        let (e, r) = best_plain(kappa);
        ui::readout(ui, "plain descent's best step", format!(r"$\eta = {}$: ${}$", num(e, 3), num(r, 3)));
        ui::see_also(
            ui,
            &[
                ("stepsize", "the step and the curvature, without momentum"),
                ("descent", "momentum on surfaces of two variables"),
                ("noisefloor", "momentum when the gradients are noisy"),
                ("signnoise", "averaging noisy gradients, and their signs"),
                ("adam", "Adam: the average form, and a step for each weight"),
            ],
        );
    }

    fn frame(&mut self, f: &mut Frame) {
        let view = f.view_size();
        // (a canvas squeezed to a sliver is one the controls have opened over, on a phone)
        let covered = view.x < 160.0;
        let lay = Layout::of(f, view, f.top_inset);
        if !covered {
            // the picture is for its buttons and its handle: no moving the view
            f.handles.capture();
            f.handles.claim_clicks();
            self.buttons(f, &lay);
        }
        self.advance(f.dt, f.real_dt);
        if covered {
            return;
        }
        // (the handle after the glides, so that it is drawn where the other dots are)
        self.hand(f, &lay);
        self.remake();
        in_order(f);
        self.draw_map(f, &lay);
        self.draw_bowl(f, &lay);
        self.draw_dist(f, &lay);
        self.draw_loss(f, &lay);
    }
}

impl Momentum {
    fn buttons(&mut self, f: &mut Frame, lay: &Layout) {
        let size = lay.text;
        if let [play, again, heavy, nesterov] = lay.buttons[..] {
            let words = if lay.narrow {
                [if self.playing() { "pause" } else { "play" }, "again", "heavy ball", "Nesterov"]
            } else {
                [if self.playing() { "❚❚ pause" } else { "▶ play the run" }, "↺ from the start", "heavy ball", "Nesterov"]
            };
            if plot::button(f, play, words[0], size, self.playing()) {
                self.toggle_play();
            }
            if plot::button(f, again, words[1], size, false) {
                self.play_from_start();
            }
            if plot::button(f, heavy, words[2], size, self.rule == Rule::Heavy) {
                self.set_rule(Rule::Heavy);
                self.stop_film();
            }
            if plot::button(f, nesterov, words[3], size, self.rule == Rule::Nesterov) {
                self.set_rule(Rule::Nesterov);
                self.stop_film();
            }
        }
    }

    /// The map's plot: s = √(ηλ) across, β up.
    fn map_plot(rect: Rect) -> Plot {
        Plot::new(rect, [0.0, S_MAX], [0.0, B_MAX])
    }

    /// The handle on the map (the sharpest direction, which the line of dots stretches
    /// from 0 to), and clicks on the dots.
    fn hand(&mut self, f: &mut Frame, lay: &Layout) {
        let Some(rect) = lay.map else { return };
        let p = Self::map_plot(rect);
        if p.tiny() {
            return;
        }
        let k = self.ease.now();
        let lam_max = self.run.lams.last().copied().unwrap_or(1.0);
        let s = (k.eta * lam_max).sqrt();
        if let Some((s, b)) = p.handle(f, HANDLE, s, k.beta, self.rule.color()) {
            let t = self.ease.target;
            self.by_hand(s * s / lam_max, b, t.kappa);
        }
        if f.ptr.clicked && !f.handles.dragging() {
            if let Some(at) = plot::pointer(f) {
                let mut best: Option<(usize, f32)> = None;
                for (rule, beta) in [(self.rule, k.beta), (Rule::Plain, 0.0)] {
                    for i in 0..self.run.d() {
                        let dot = p.at(self.run.a(rule, i).sqrt(), beta);
                        let dist = dot.distance(at);
                        if dist < 11.0 && best.is_none_or(|(_, b)| dist < b) {
                            best = Some((i, dist));
                        }
                    }
                }
                if let Some((i, _)) = best {
                    self.follow = Some(i);
                }
            }
        }
    }

    /// The map's picture of the rate, painted again only when the rule changes.
    fn paint(&mut self, f: &mut Frame) {
        if self.painted == Some(self.rule) && self.image.is_some() {
            return;
        }
        let image = self.image.get_or_insert_with(|| {
            let mut image = Image::new(f.gfx, IMG_W, IMG_H);
            image.smooth = true;
            image
        });
        image.ensure_size(f.gfx, IMG_W, IMG_H);
        let [w, h] = image.size();
        let rule = self.rule;
        let pixels = image.pixels_mut();
        for j in 0..h {
            for i in 0..w {
                let s = S_MAX * (i as f64 + 0.5) / w as f64;
                let b = B_MAX * (j as f64 + 0.5) / h as f64;
                if let Some(px) = pixels.get_mut((j * w + i) as usize) {
                    *px = shade(rate(rule, s * s, b));
                }
            }
        }
        self.painted = Some(rule);
    }

    /// (a) The map: ηλ across, β up, coloured by the rate, the regions outlined and
    /// named, the bowl's directions as dots on the line at the momentum's height.
    fn draw_map(&mut self, f: &mut Frame, lay: &Layout) {
        let Some(rect) = lay.map else { return };
        let p = Self::map_plot(rect);
        if p.tiny() {
            return;
        }
        let (min, max) = rect;
        let (text, small) = (lay.text, lay.small);
        let left = if lay.narrow { MAP_LEFT - 10.0 } else { MAP_LEFT };
        backing_round(f, (min - Vec2::new(left + 2.0, MAP_ABOVE - 2.0), max + Vec2::new(8.0, MAP_BELOW - 4.0)), rect);
        self.paint(f);
        if let Some(image) = &mut self.image {
            let (lo, hi) = world_rect(f, rect);
            f.image(image, lo, hi, 0.0);
        }
        let rule = self.rule;
        // the contours of equal rate, faint, each named where it is nearest the middle
        let wanted_s = if rule == Rule::Nesterov { 0.2 } else { 1.0 };
        for (level, segs) in &self.chart.contours {
            for [a, b] in segs {
                plot::line_of(f, CONTOURS, p.at(a.0, a.1), p.at(b.0, b.1), 1.0, pal::WHITE.a(0.32));
            }
            let aim = p.at(wanted_s, level * level);
            let near = segs.iter().map(|[a, b]| p.at(0.5 * (a.0 + b.0), 0.5 * (a.1 + b.1))).min_by(|x, y| x.distance(aim).total_cmp(&y.distance(aim)));
            if let Some(at) = near {
                if !lay.narrow || *level > 0.6 {
                    f.text_px(at + Vec2::new(0.0, -2.0), format!("$|z| = {}$", trim(*level)), small - 1.5, pal::WHITE.a(0.75), Align2::CENTER_BOTTOM);
                }
            }
        }
        // the edges of the regions, and the limit
        let curve = |g: &dyn Fn(f64) -> f64| -> Vec<(f64, f64)> {
            (0..=120)
                .map(|j| {
                    let b = B_MAX * j as f64 / 120.0;
                    (g(b).max(0.0).sqrt(), b)
                })
                .collect()
        };
        p.path(f, OUTLINES, curve(&|b| edges(rule, b).critical), 1.4, pal::WHITE.a(0.8));
        p.path(f, OUTLINES, curve(&|b| edges(rule, b).second), 1.4, pal::WHITE.a(0.8));
        p.path(f, LIMITS, curve(&|b| edges(rule, b).limit), 2.2, LIMIT);
        p.vline(f, PLAIN_LIMIT, 2f64.sqrt(), 1.2, LIMIT.a(0.7), true);
        // the regions' names
        let names: [(f64, f64, &str); 4] = match rule {
            Rule::Nesterov => [(0.27, 0.22, "creeps in"), (0.68, 0.62, "rings"), (1.19, 0.05, "flips"), (1.75, 0.75, "flies off")],
            _ => [(0.26, 0.22, "creeps in"), (1.0, 0.37, "rings"), (1.335, 0.06, "flips"), (1.92, 0.3, "flies off")],
        };
        for (s, b, name) in names {
            f.caption_px(p.at(s, b), name, text, pal::TEXT, Align2::CENTER_CENTER);
        }
        let crit_b = 0.1;
        let crit = p.at(edges(rule, crit_b).critical.sqrt(), crit_b);
        f.caption_px(crit, if lay.narrow { "critical" } else { "critical: $z_1 = z_2$" }, small - 0.5, pal::TEXT.a(0.9), Align2::CENTER_CENTER);
        let (limit_b, limit_words) = match rule {
            Rule::Nesterov => (0.45, r"limit: $\eta\lambda = (2 + 2\beta)/(1 + 2\beta)$"),
            _ => (0.56, r"limit: $\eta\lambda = 2 + 2\beta$"),
        };
        let limit_at = p.at(edges(rule, limit_b).limit.sqrt(), limit_b);
        if rule == Rule::Nesterov {
            f.caption_px(limit_at + Vec2::new(6.0, 0.0), if lay.narrow { "limit" } else { limit_words }, small - 0.5, LIMIT, Align2::LEFT_CENTER);
        } else {
            f.caption_px(limit_at - Vec2::new(6.0, 0.0), if lay.narrow { "limit" } else { limit_words }, small - 0.5, LIMIT, Align2::RIGHT_CENTER);
        }
        let plain_at = p.at(2f64.sqrt(), 0.17);
        f.caption_px(plain_at - Vec2::new(5.0, 0.0), if lay.narrow { "plain's limit" } else { r"plain descent's limit, $\eta\lambda = 2$" }, small - 1.0, LIMIT.a(0.85), Align2::RIGHT_CENTER);
        // the axes: round values of ηλ on the square-root scale
        line_of_axis(f, &p);
        let ticks: &[f64] = if lay.narrow { &[0.0, 0.1, 0.5, 1.0, 2.0, 3.0, 4.0] } else { &[0.0, 0.1, 0.25, 0.5, 1.0, 2.0, 3.0, 4.0] };
        for &a in ticks {
            let x = p.px_x(a.sqrt());
            plot::line_of(f, plot::AXES, Vec2::new(x, max.y), Vec2::new(x, max.y + 4.0), 1.0, pal::DIM.a(0.8));
            f.text_px(Vec2::new(x, max.y + 6.0), trim(a), small - 0.5, pal::DIM, Align2::CENTER_TOP);
        }
        let x_name = if lay.narrow { r"step times curvature, $\eta\lambda$ (square-root scale)" } else { r"the step times the curvature, $\eta\lambda$ (on a square-root scale)" };
        f.text_px(Vec2::new((min.x + max.x) * 0.5, max.y + 8.0 + small), x_name, small - 0.5, pal::DIM, Align2::CENTER_TOP);
        p.y_axis(f, "", small - 0.5);
        f.text_px(Vec2::new(min.x - left + 4.0, min.y - 5.0), r"momentum $\beta$", small - 0.5, pal::DIM, Align2::LEFT_BOTTOM);
        // the title and the colour key
        let title = if lay.narrow {
            format!(r"{}: $|z|$, what a direction is multiplied by per step", rule.name())
        } else {
            format!(r"{}: how fast a direction shrinks, $|z|$ a step (the larger eigenvalue's size)", rule.name())
        };
        let title_at = Vec2::new(min.x - left + 4.0, min.y - MAP_ABOVE + 6.0);
        fit_text(f, title_at, max.x - title_at.x, &title, text, pal::TEXT);
        self.colour_key(f, Vec2::new(min.x - left + 4.0, title_at.y + text + 10.0), small, lay.narrow);
        // the bowl's directions
        self.dots(f, &p, small);
    }

    /// The colour key of the map: a strip from |z| = 0 to 0.999, and the colour of a
    /// direction that grows.
    fn colour_key(&self, f: &mut Frame, at: Vec2, small: f32, narrow: bool) {
        let lead = r"$|z|$:";
        f.text_px(at + Vec2::new(0.0, 4.0), lead, small, pal::DIM, Align2::LEFT_CENTER);
        let x0 = at.x + f.text_size(lead, small).x + 8.0;
        let width = if narrow { 120.0 } else { 170.0 };
        let strips = 40;
        for n in 0..strips {
            let (a, b) = (n as f32 / strips as f32, (n + 1) as f32 / strips as f32);
            // (evenly in −log₁₀(1 − |z|), as the map is coloured)
            let v = ((a + b) * 0.5) as f64 * V_MAX;
            hud_rect(f, Vec2::new(x0 + width * a, at.y), Vec2::new(x0 + width * b + 0.5, at.y + 8.0), shade(1.0 - 10f64.powf(-v)));
        }
        for (v, name) in [(0.0, "0"), (1.0, "0.9"), (2.0, "0.99"), (3.0, "0.999")] {
            let x = x0 + width * (v / V_MAX) as f32;
            plot::line_of(f, plot::AXES, Vec2::new(x, at.y + 8.0), Vec2::new(x, at.y + 11.0), 1.0, pal::DIM);
            let align = if v == 0.0 { Align2::LEFT_TOP } else if v == V_MAX { Align2::RIGHT_TOP } else { Align2::CENTER_TOP };
            f.text_px(Vec2::new(x, at.y + 11.0), name, small - 1.5, pal::DIM, align);
        }
        let x1 = x0 + width + 14.0;
        hud_rect(f, Vec2::new(x1, at.y), Vec2::new(x1 + 16.0, at.y + 8.0), FLY);
        f.text_px(Vec2::new(x1 + 21.0, at.y + 4.0), if narrow { "above 1" } else { "above 1: it flies off" }, small - 1.0, pal::DIM, Align2::LEFT_CENTER);
    }

    /// The bowl's directions on the map: the rule's on the line at height β (the
    /// sharpest is the handle), plain descent's along the bottom, the slowest ringed and
    /// the one followed below dropped to the axis.
    fn dots(&self, f: &mut Frame, p: &Plot, small: f32) {
        let rule = self.rule;
        let run = &self.run;
        let d = run.d();
        let beta = run.beta;
        let color = rule.color();
        let s_of = |rule: Rule, i: usize| run.a(rule, i).max(0.0).sqrt();
        plot::line_of(f, LINE, p.at(0.0, beta), p.at(s_of(rule, d - 1), beta), 1.6, color.a(0.6));
        for i in 0..d {
            plot::disc(f, p.at(s_of(Rule::Plain, i), 0.0), 7.0, SGD);
        }
        for i in 0..d.saturating_sub(1) {
            plot::disc(f, p.at(s_of(rule, i), beta), 9.0, color);
        }
        let (most, slow) = run.slowest(rule);
        for &i in &slow {
            ring(f, RINGS, p.at(s_of(rule, i), beta), 10.5, 1.6, pal::WHITE);
        }
        // what the ring says, above the line (below it near the top)
        if let Some(&first) = slow.first() {
            let from = p.at(s_of(rule, first), beta);
            let words = if most > 1.0 + 1e-9 {
                format!(r"grows: $|z| = {}$", num(most, 3))
            } else if slow.len() == d && d > 1 {
                format!(r"all as slow: $|z| = {}$", num(most, 3))
            } else {
                format!(r"slowest: $|z| = {}$", num(most, 3))
            };
            let w = f.text_size(&words, small).x;
            let up = from.y - p.rect.0.y > 30.0;
            // (kept inside the map across)
            let cx = (from.x).max(p.rect.0.x + w * 0.5 + 2.0).min(p.rect.1.x - w * 0.5 - 2.0);
            let offset = Vec2::new(cx - from.x, if up { -22.0 } else { 22.0 });
            f.text_placed("slowest", 0, from, offset, words, small, pal::WHITE);
        }
        // β, beyond the line's end (or before its start, where there is no room after it)
        let words = format!(r"$\beta = {}$", num(beta, 3));
        let w = f.text_size(&words, small).x;
        let end = p.at(s_of(rule, d - 1), beta);
        let up = if end.y - p.rect.0.y > 26.0 { -18.0 } else { 18.0 };
        let offset = if end.x + 16.0 + w <= p.rect.1.x - 4.0 { Vec2::new(16.0 + 0.5 * w, 0.0) } else { Vec2::new(-0.5 * w, up) };
        f.text_placed("beta", 0, end, offset, words, small, color.mix(pal::WHITE, 0.3));
        // the direction followed in the chart below, dropped to the axis
        let i = self.followed();
        let s = s_of(rule, i);
        plot::dashed_of(f, DROP, p.at(s, beta), p.at(s, 0.0), 1.0, pal::WHITE.a(0.55));
    }

    /// (c) The valley from above: the flattest direction across, the sharpest up, the
    /// level lines of the loss there, and the paths of the runs drawn.
    fn draw_bowl(&mut self, f: &mut Frame, lay: &Layout) {
        let Some(rect) = lay.bowl else { return };
        let (min, max) = rect;
        let (text, small) = (lay.text, lay.small);
        hud_panel(f, min - Vec2::new(8.0, CHART_ABOVE - 4.0), max + Vec2::new(8.0, 8.0));
        let run = &self.run;
        let d = run.d();
        let title = if lay.narrow {
            "the valley from above".to_string()
        } else if d == 2 {
            "the valley from above: its flat direction across, its steep one up".to_string()
        } else {
            format!("the bowl from above, in the flattest (across) and sharpest (up) of its {d} directions")
        };
        let title_at = Vec2::new(min.x - 2.0, min.y - CHART_ABOVE + 6.0);
        fit_text(f, title_at, max.x - title_at.x, &title, text, pal::TEXT);
        self.legend(f, Vec2::new(min.x - 2.0, title_at.y + text + 12.0), small, true);
        let [u0, u1, w0, w1] = VALLEY;
        let scale = (((max.x - min.x) as f64 / (u1 - u0).max(1e-6)).min((max.y - min.y) as f64 / (w1 - w0).max(1e-6))) as f32;
        let (cu, cw) = (0.5 * (u0 + u1), 0.5 * (w0 + w1));
        let mid = (min + max) * 0.5;
        // (a run that flies off is nowhere once it is well outside the window: a step
        // from far above to far below, as a run that flips and grows takes, would
        // otherwise be drawn straight across the window, and it stands for nothing)
        let (du, dw) = (0.5 * (u1 - u0), 0.5 * (w1 - w0));
        let to = |u: f64, w: f64| {
            if !(u >= u0 - du && u <= u1 + du && w >= w0 - dw && w <= w1 + dw) {
                return Vec2::NAN;
            }
            mid + Vec2::new(((u - cu) as f32) * scale, -((w - cw) as f32) * scale)
        };
        // the level lines of the loss in this plane, through the start and round it
        let (lf, ls) = (run.lams[0], run.lams[d - 1]);
        let at_start = 0.5 * (lf * FLAT_START * FLAT_START + ls * REST_START * REST_START);
        for n in -2..=5 {
            let c = at_start * 4f64.powi(-n);
            let (ru, rw) = ((2.0 * c / lf).sqrt(), (2.0 * c / ls).sqrt());
            let pts: Vec<Vec2> = (0..=96).map(|j| j as f64 * std::f64::consts::TAU / 96.0).map(|th| to(ru * th.cos(), rw * th.sin())).collect();
            plot::path_of(f, LEVEL_LINES, &plot::clip(&pts, rect), 1.0, pal::DIM.a(if n == 0 { 0.55 } else { 0.3 }));
        }
        // the bottom
        let o = to(0.0, 0.0);
        if plot::within(o, rect, 0.0) {
            plot::line_of(f, plot::MARKS, o - Vec2::new(5.0, 0.0), o + Vec2::new(5.0, 0.0), 1.2, pal::TEXT.a(0.8));
            plot::line_of(f, plot::MARKS, o - Vec2::new(0.0, 5.0), o + Vec2::new(0.0, 5.0), 1.2, pal::TEXT.a(0.8));
        }
        // the paths, as far as the step drawn; and during the film's slow start, a ball
        // on each
        let s = to(FLAT_START, REST_START);
        if plot::within(s, rect, 0.0) {
            plot::disc(f, s, 9.0, pal::WHITE);
        }
        let Some((run, span, rules, upto, _)) = self.drawn() else { return };
        let d = run.d();
        // the runs that leave for good, named in the corner
        let mut y = min.y + 6.0;
        for rule in rules.iter().filter(|r| run.path(**r).loss.last().is_some_and(|l| *l > run.path(**r).loss[0])) {
            let words = format!("{} flies off", rule.name());
            f.caption_px(Vec2::new(max.x - 6.0, y), words, small - 0.5, rule.color().mix(pal::WHITE, 0.2).a(self.fade()), Align2::RIGHT_TOP);
            y += small + 8.0;
        }
        for (n, rule) in rules.into_iter().enumerate() {
            let path = run.path(rule);
            let pts: Vec<Vec2> = (0..=span).map(|t| to(path.at(t, 0, d), path.at(t, d - 1, d))).collect();
            let faint = self.rule == Rule::Nesterov && rule == Rule::Heavy;
            let color = rule.color().a(if faint { 0.45 } else { 1.0 } * self.fade());
            reveal(f, BOWL_PATHS + n as u8, &pts, rect, upto, if rule == Rule::Plain { 1.5 } else { 1.8 }, color);
            if self.ball() {
                let ball = along(&pts, upto);
                if plot::within(ball, rect, 0.0) {
                    plot::disc(f, ball, if faint { 6.0 } else { 8.0 }, color);
                }
            }
        }
    }

    /// The key of the runs drawn: plain descent, the rule (and heavy ball beside
    /// Nesterov), and the start where the valley is drawn.
    fn legend(&self, f: &mut Frame, at: Vec2, small: f32, with_start: bool) {
        let mut x = at.x;
        if with_start {
            plot::disc(f, Vec2::new(x + 4.0, at.y), 8.0, pal::WHITE);
            f.text_px(Vec2::new(x + 12.0, at.y), "start", small - 0.5, pal::DIM, Align2::LEFT_CENTER);
            x += 12.0 + f.text_size("start", small - 0.5).x + 14.0;
        }
        for rule in self.drawn_rules() {
            let faint = self.rule == Rule::Nesterov && rule == Rule::Heavy;
            let color = rule.color().a(if faint { 0.5 } else { 1.0 });
            hud_rect(f, Vec2::new(x, at.y - 1.0), Vec2::new(x + 14.0, at.y + 1.0), color);
            let name = rule.name();
            f.text_px(Vec2::new(x + 18.0, at.y), name, small - 0.5, pal::DIM, Align2::LEFT_CENTER);
            x += 18.0 + f.text_size(name, small - 0.5).x + 14.0;
        }
    }

    /// (b) One direction's distance from the bottom against the step.
    fn draw_dist(&mut self, f: &mut Frame, lay: &Layout) {
        let Some(rect) = lay.dist else { return };
        let (text, small) = (lay.text, lay.small);
        let span = self.span.get(0).max(1.0) as f64;
        let top = self.height.get(0).max(1.0) as f64;
        let p = Plot::new(rect, [0.0, span], [-top, top]);
        if p.tiny() {
            return;
        }
        let (min, max) = rect;
        p.backing(f, if lay.narrow { 28.0 } else { 40.0 }, CHART_BELOW - 4.0, CHART_ABOVE - 4.0, 8.0);
        let run = &self.run;
        let (d, i) = (run.d(), self.followed());
        let which = if d > 1 && i == 0 {
            ", the flattest"
        } else if d > 1 && i == d - 1 {
            ", the sharpest"
        } else {
            ""
        };
        let title = if lay.narrow { format!("direction {} of {d}{which}", i + 1) } else { format!("direction {} of {d}{which}: its distance from the bottom, a share of its start", i + 1) };
        let title_at = Vec2::new(min.x - if lay.narrow { 26.0 } else { 36.0 }, min.y - CHART_ABOVE + 6.0);
        fit_text(f, title_at, max.x - title_at.x, &title, text, pal::TEXT);
        let rule = self.rule;
        let words = if lay.narrow {
            format!(r"$\eta\lambda = {}$: $|z| = {}$; plain ${}$", num(run.a(rule, i), 3), num(run.rate(rule, i), 3), num(run.rate(Rule::Plain, i), 3))
        } else {
            format!(
                r"$\eta\lambda = {}$: {} {}, $|z| = {}$; plain descent {}, ${}$",
                num(run.a(rule, i), 3),
                rule.name(),
                run.kind(rule, i).word(),
                num(run.rate(rule, i), 3),
                run.kind(Rule::Plain, i).word(),
                num(run.rate(Rule::Plain, i), 3)
            )
        };
        fit_text(f, title_at + Vec2::new(0.0, text + 7.0), max.x - title_at.x, &words, small - 0.5, pal::DIM);
        p.hline(f, plot::AXES, 0.0, 1.0, pal::DIM.a(0.5), false);
        gliding_axes(f, &p, small - 1.0, true);
        f.text_px(Vec2::new(max.x, max.y + 7.0 + small), "step", small - 1.0, pal::DIM, Align2::RIGHT_TOP);
        let Some((run, span, rules, upto, i)) = self.drawn() else { return };
        let d = run.d();
        let x0 = start(d)[i];
        for (n, rule) in rules.into_iter().enumerate() {
            let path = run.path(rule);
            // (nowhere once well outside the chart: see `draw_bowl`)
            let y = |t: usize| Some(path.at(t, i, d) / x0).filter(|y| y.abs() <= 2.0 * top).unwrap_or(f64::NAN);
            let pts: Vec<Vec2> = (0..=span).map(|t| p.at(t as f64, y(t))).collect();
            let faint = self.rule == Rule::Nesterov && rule == Rule::Heavy;
            reveal(f, DIST_SERIES + n as u8, &pts, rect, upto, if rule == Rule::Plain { 1.3 } else { 1.7 }, rule.color().a(if faint { 0.45 } else { 1.0 } * self.fade()));
        }
    }

    /// (c) The losses against the step, on a logarithmic axis.
    fn draw_loss(&mut self, f: &mut Frame, lay: &Layout) {
        let Some(rect) = lay.loss else { return };
        let (text, small) = (lay.text, lay.small);
        let span = self.span.get(0).max(1.0) as f64;
        let p = Plot::new(rect, [0.0, span], [LOSS_FLOOR, LOSS_TOP]);
        if p.tiny() {
            return;
        }
        let (min, max) = rect;
        p.backing(f, if lay.narrow { 30.0 } else { 44.0 }, CHART_BELOW - 4.0, CHART_ABOVE - 4.0, 8.0);
        let title = if lay.narrow { "the loss / its start" } else { "the loss, a share of its start (log scale), and the steps to a millionth" };
        let title_at = Vec2::new(min.x - if lay.narrow { 28.0 } else { 40.0 }, min.y - CHART_ABOVE + 6.0);
        fit_text(f, title_at, max.x - title_at.x, title, text, pal::TEXT);
        // the counts, as a key
        let run = &self.run;
        let mut x = title_at.x;
        let y = title_at.y + text + 12.0;
        for rule in self.drawn_rules() {
            let faint = self.rule == Rule::Nesterov && rule == Rule::Heavy;
            let color = rule.color().a(if faint { 0.5 } else { 1.0 });
            let path = run.path(rule);
            let words = match (lay.narrow, path.count) {
                (true, Some(n)) => format!("{n}"),
                (true, None) => "never".to_string(),
                (false, _) => format!("{}: {}", rule.name(), count_words(path)),
            };
            hud_rect(f, Vec2::new(x, y - 1.0), Vec2::new(x + 12.0, y + 1.0), color);
            f.text_px(Vec2::new(x + 16.0, y), words.as_str(), small - 0.5, pal::DIM, Align2::LEFT_CENTER);
            x += 16.0 + f.text_size(&words, small - 0.5).x + 12.0;
        }
        // the powers of ten up the side, and a millionth
        for e in (LOSS_FLOOR as i32)..=(LOSS_TOP as i32) {
            let yy = p.px_y(e as f64);
            let named = e % 3 == 0;
            plot::line_of(f, plot::AXES, Vec2::new(min.x, yy), Vec2::new(max.x, yy), 1.0, pal::FAINT.a(if named { 0.5 } else { 0.18 }));
            if named {
                let label = if e == 0 { "$1$".to_string() } else { format!("$10^{{{e}}}$") };
                f.text_px(Vec2::new(min.x - 5.0, yy), label, small - 1.0, pal::DIM, Align2::RIGHT_CENTER);
            }
        }
        plot::line_of(f, plot::AXES, Vec2::new(min.x, min.y), Vec2::new(min.x, max.y), 1.0, pal::DIM.a(0.8));
        p.hline(f, LOSS_MARKS, TARGET.log10(), 1.2, pal::WHITE.a(0.5), true);
        gliding_axes(f, &p, small - 1.0, false);
        f.text_px(Vec2::new(max.x, max.y + 7.0 + small), "step", small - 1.0, pal::DIM, Align2::RIGHT_TOP);
        // where each reaches a millionth for good (gliding there: as the settings glide
        // the counts change a step at a time)
        // (drawn in every frame, unseen at the chart's end when past it, so that the
        // marks keep their places in their batch)
        for rule in self.drawn_rules() {
            let faint = self.rule == Rule::Nesterov && rule == Rule::Heavy;
            let at = self.marks.get(rule.index()) as f64;
            let x = p.px_x(at.max(0.0).min(span));
            let alpha = if at <= span { if faint { 0.35 } else { 0.6 } } else { 0.0 };
            plot::dashed_of(f, LOSS_MARKS, Vec2::new(x, min.y), Vec2::new(x, max.y), 1.0, rule.color().a(alpha));
        }
        let Some((run, span, rules, upto, _)) = self.drawn() else { return };
        for (n, rule) in rules.into_iter().enumerate() {
            let path = run.path(rule);
            let l0 = path.loss[0].max(1e-300);
            let y = |t: usize| (path.loss[t.min(MAX)] / l0).max(1e-300).log10().max(LOSS_FLOOR - 5.0);
            let y = |t: usize| Some(y(t)).filter(|y| *y <= LOSS_TOP + 5.0).unwrap_or(f64::NAN);
            let pts: Vec<Vec2> = (0..=span).map(|t| p.at(t as f64, y(t))).collect();
            let faint = self.rule == Rule::Nesterov && rule == Rule::Heavy;
            reveal(f, LOSS_SERIES + n as u8, &pts, rect, upto, if rule == Rule::Plain { 1.4 } else { 1.8 }, rule.color().a(if faint { 0.45 } else { 1.0 } * self.fade()));
        }
    }
}

/// The range of the loss chart, in powers of ten of the loss over its start.
const LOSS_FLOOR: f64 = -9.0;
const LOSS_TOP: f64 = 1.5;

/// The axes of a chart whose ranges glide: the lines along the bottom and up the left,
/// and round numbers along them (of the steps, and with `up` of the height), with no
/// tick marks: as a range glides the round numbers change, and marks would come and go
/// with them.
fn gliding_axes(f: &mut Frame, p: &Plot, size: f32, up: bool) {
    if p.tiny() {
        return;
    }
    let (min, max) = p.rect;
    plot::line_of(f, plot::AXES, Vec2::new(min.x, max.y), max, 1.0, pal::DIM.a(0.8));
    let ticks = plot::Ticks::of(p.x[0], p.x[1], p.width());
    for v in ticks.values() {
        f.text_px(Vec2::new(p.px_x(v), max.y + 5.0), ticks.name(v), size, pal::DIM, Align2::CENTER_TOP);
    }
    if up {
        plot::line_of(f, plot::AXES, min, Vec2::new(min.x, max.y), 1.0, pal::DIM.a(0.8));
        let ticks = plot::Ticks::of(p.y[0], p.y[1], p.height() * 2.2);
        for v in ticks.values() {
            f.text_px(Vec2::new(min.x - 6.0, p.px_y(v)), ticks.name(v), size, pal::DIM, Align2::RIGHT_CENTER);
        }
    }
}

/// The line along the bottom of a chart.
fn line_of_axis(f: &mut Frame, p: &Plot) {
    let (min, max) = p.rect;
    plot::line_of(f, plot::AXES, Vec2::new(min.x, max.y), max, 1.0, pal::DIM.a(0.8));
}

/// The point a share of the way along a path of steps: `at` steps from its start.
fn along(pts: &[Vec2], at: f64) -> Vec2 {
    let Some(last) = pts.len().checked_sub(1) else { return Vec2::NAN };
    let at = at.max(0.0).min(last as f64);
    let t0 = (at.floor() as usize).min(last);
    let t1 = (t0 + 1).min(last);
    pts[t0].lerp(pts[t1], (at - t0 as f64) as f32)
}

/// A run's path (`pts[t]` the point at step t, on the screen) drawn as far as step
/// `upto`, kept inside `rect`. Every step's point is drawn in its place, those not yet
/// reached transparent, so that nothing moves while the film plays: only what can be
/// seen grows. During the slow start the line ends at the ball, a share of the way
/// through a step (two points more, at the ball); after it, each new step fades in as it
/// is reached (one point more, unseen, where the slow start ended: the same picture at
/// the change, its points in another order, and so a number of points of its own).
fn reveal(f: &mut Frame, kind: u8, pts: &[Vec2], rect: Rect, upto: f64, width: f32, color: Color) {
    let Some(last) = pts.len().checked_sub(1) else { return };
    let upto = upto.max(0.0).min(last as f64);
    let t0 = (upto.floor() as usize).min(last);
    let frac = (upto - t0 as f64) as f32;
    let on = |c: f32| color.a(color.0[3] as f32 / 255.0 * c);
    let mut pieces: Vec<Vec<(Vec2, Color)>> = Vec::with_capacity(3);
    let seen: Vec<(Vec2, Color)> = pts[..=t0].iter().map(|p| (*p, on(1.0))).collect();
    if upto < SLOW_STEPS {
        // the line seen, up to the ball; and the rest, unseen, from the ball on
        let ball = along(pts, upto);
        let mut first = seen;
        first.push((ball, on(1.0)));
        let mut rest = vec![(ball, on(0.0))];
        rest.extend(pts[t0 + 1..].iter().map(|p| (*p, on(0.0))));
        pieces.push(first);
        pieces.push(rest);
    } else {
        let mut line = seen;
        line.extend(pts[t0 + 1..].iter().enumerate().map(|(k, p)| (*p, on(if k == 0 { frac } else { 0.0 }))));
        let mark = pts[(SLOW_STEPS as usize).min(last)];
        pieces.push(line);
        pieces.push(vec![(mark, on(0.0))]);
    }
    let placed: Vec<Vec<(Vec3, Color)>> = pieces.iter().map(|piece| clip_fixed(piece, rect).iter().map(|(p, c)| (f.hud(*p), *c)).collect()).collect();
    let batch = plot::apart(f, kind, false);
    for run in placed {
        batch.brk();
        for (p, c) in run {
            batch.push(p, width, c);
        }
        batch.brk();
    }
}

/// A line with a colour at each point, kept to the part inside a rectangle with the same
/// number of points whatever it crosses: each segment becomes four points, the part of
/// it inside (its colours taken in proportion) between two unseen points at its ends,
/// and a segment wholly outside four unseen points where it comes nearest the
/// rectangle, pressed against its edge (where an inside part shrinks to nothing as the
/// segment leaves, so that nothing jumps); a segment with an end that is not a number
/// (a run gone far off the chart) off the picture altogether. A line cut where it leaves
/// would gain and lose points as a run's end crosses the edge, and could not be
/// followed from frame to frame.
fn clip_fixed(pts: &[(Vec2, Color)], (min, max): Rect) -> Vec<(Vec2, Color)> {
    let (lo, hi) = (min - Vec2::splat(0.5), max + Vec2::splat(0.5));
    // (a segment with an end that is nowhere is kept off the picture, unseen)
    let nowhere = Vec2::splat(-40.0);
    let mut out = Vec::with_capacity(4 * pts.len());
    for pair in pts.windows(2) {
        let ((a, ca), (b, cb)) = (pair[0], pair[1]);
        let d = b - a;
        let mut range = (a.is_finite() && b.is_finite()).then_some((0.0f32, 1.0f32));
        for (p, q) in [(-d.x, a.x - lo.x), (d.x, hi.x - a.x), (-d.y, a.y - lo.y), (d.y, hi.y - a.y)] {
            range = range.and_then(|(t0, t1)| {
                let r = if p == 0.0 {
                    (q >= 0.0).then_some((t0, t1))
                } else if p < 0.0 {
                    Some((t0.max(q / p), t1))
                } else {
                    Some((t0, t1.min(q / p)))
                };
                r.filter(|(t0, t1)| t0 <= t1)
            });
        }
        let press = |p: Vec2| if p.is_finite() { p.max(lo).min(hi) } else { lo };
        // (how far a point is from the rectangle: convex along the segment, so a search
        // by thirds finds where the segment comes nearest)
        let nearest = || {
            let gap = |t: f32| (a + d * t - press(a + d * t)).length_squared();
            let (mut t0, mut t1) = (0.0f32, 1.0f32);
            for _ in 0..24 {
                let (m0, m1) = (t0 + (t1 - t0) / 3.0, t1 - (t1 - t0) / 3.0);
                if gap(m0) <= gap(m1) {
                    t1 = m1;
                } else {
                    t0 = m0;
                }
            }
            press(a + d * (0.5 * (t0 + t1)))
        };
        match range {
            Some((t0, t1)) => {
                let (p, q) = (a + d * t0, a + d * t1);
                let (cp, cq) = (ca.mix(cb, t0), ca.mix(cb, t1));
                out.extend([(p, cp.a(0.0)), (p, cp), (q, cq), (q, cq.a(0.0))]);
            }
            None => {
                let p = if a.is_finite() && b.is_finite() { nearest() } else { nowhere };
                out.extend([(p, ca.a(0.0)), (p, ca.a(0.0)), (p, cb.a(0.0)), (p, cb.a(0.0))]);
            }
        }
    }
    out
}

/// A line with a colour at each point, cut to the part inside a rectangle (with half a
/// pixel's grace), the colours between taken in proportion; where it leaves the
/// rectangle it is broken (a point that is not a number).
#[cfg(test)]
fn clip_colored(pts: &[(Vec2, Color)], (min, max): Rect) -> Vec<(Vec2, Color)> {
    let (min, max) = (min - Vec2::splat(0.5), max + Vec2::splat(0.5));
    let mut out: Vec<(Vec2, Color)> = Vec::with_capacity(pts.len() + 4);
    let mut open = false;
    if let [only] = pts {
        if only.0.cmpge(min).all() && only.0.cmple(max).all() {
            out.push(*only);
        }
        return out;
    }
    for pair in pts.windows(2) {
        let ((a, ca), (b, cb)) = (pair[0], pair[1]);
        let d = b - a;
        let mut range = Some((0.0f32, 1.0f32));
        if !(a.is_finite() && b.is_finite()) {
            range = None;
        }
        for (p, q) in [(-d.x, a.x - min.x), (d.x, max.x - a.x), (-d.y, a.y - min.y), (d.y, max.y - a.y)] {
            if let Some((t0, t1)) = range {
                range = if p == 0.0 {
                    (q >= 0.0).then_some((t0, t1))
                } else if p < 0.0 {
                    Some((t0.max(q / p), t1))
                } else {
                    Some((t0, t1.min(q / p)))
                };
                range = range.filter(|(t0, t1)| t0 <= t1);
            }
        }
        match range {
            Some((t0, t1)) => {
                let (p, q) = (a + d * t0, a + d * t1);
                let (cp, cq) = (ca.mix(cb, t0), ca.mix(cb, t1));
                if !(open && t0 == 0.0) {
                    if !out.is_empty() {
                        out.push((Vec2::NAN, cp));
                    }
                    out.push((p, cp));
                }
                out.push((q, cq));
                open = t1 == 1.0;
            }
            None => open = false,
        }
    }
    out
}

/// A large number with the decimals it is worth: 100, 31.6, 2.15.
fn trim_big(v: f64) -> String {
    num(v, if v >= 99.5 { 0 } else if v >= 9.95 { 1 } else { 2 })
}

/// A number with as few decimals as it needs, up to three.
fn trim(v: f64) -> String {
    let s = num(v, 3);
    let s = s.trim_end_matches('0').trim_end_matches('.');
    if s.is_empty() || s == "−" {
        "0".to_string()
    } else {
        s.to_string()
    }
}

/// A line of words drawn smaller if it does not fit the width.
fn fit_text(f: &mut Frame, at: Vec2, width: f32, text: &str, size: f32, color: Color) {
    let w = f.text_size(text, size).x;
    let s = if w > width && w > 0.0 { (size * width / w).max(8.0) } else { size };
    f.text_px(at, text, s, color, Align2::LEFT_TOP);
}

#[cfg(test)]
mod tests {
    use super::*;

    const RULES: [Rule; 3] = [Rule::Plain, Rule::Heavy, Rule::Nesterov];
    const BETAS: [f64; 8] = [0.0, 0.05, 0.3, 0.5, 0.669, 0.9, 0.95, 0.99];

    /// One direction's distances for `n` steps of `rule`, from `x0` at rest.
    fn one(rule: Rule, eta: f64, beta: f64, lambda: f64, x0: f64, n: usize) -> Vec<f64> {
        let mut path = Path::default();
        let lams = [lambda];
        simulate(rule, eta, beta, &lams, &mut path);
        // (`simulate` starts at the flattest direction's start; scale to x0)
        path.x.iter().take(n + 1).map(|x| x / FLAT_START * x0).collect()
    }

    /// The size of the largest eigenvalue of a 2 × 2 matrix, measured: how fast its
    /// powers grow, on a generic vector, renormalised as it goes.
    fn measured_rate(m: [[f64; 2]; 2], steps: usize) -> f64 {
        let (mut v, mut log) = ([1.0, 0.37], 0.0);
        for _ in 0..steps {
            v = [m[0][0] * v[0] + m[0][1] * v[1], m[1][0] * v[0] + m[1][1] * v[1]];
            let n = v[0].hypot(v[1]);
            if n == 0.0 {
                return 0.0;
            }
            log += n.ln();
            v = [v[0] / n, v[1] / n];
        }
        (log / steps as f64).exp()
    }

    /// The step's matrix is what the rules do, as PyTorch writes them: its powers give
    /// the runs, its trace and determinant are the ones the eigenvalues are taken from,
    /// and its powers grow at the rate of the larger eigenvalue.
    #[test]
    fn the_step_matrix_is_the_rule_and_has_the_eigenvalues_drawn() {
        for rule in RULES {
            for &beta in &BETAS {
                let beta = if rule == Rule::Plain { 0.0 } else { beta };
                for eta in [0.05, 0.4, 1.0, 1.3, 1.9, 3.0, 4.3] {
                    for lambda in [1.0, 0.1, 0.01] {
                        let m = step_matrix(rule, eta, lambda, beta);
                        let a = eta * lambda;
                        let (t, d) = trace_det(rule, a, beta);
                        assert!((m[0][0] + m[1][1] - t).abs() < 1e-12 && (m[0][0] * m[1][1] - m[0][1] * m[1][0] - d).abs() < 1e-12, "{rule:?} {eta} {lambda} {beta}");
                        // the eigenvalues solve z² − tz + d = 0
                        for z in eigenvalues(rule, a, beta) {
                            let re = z.0 * z.0 - z.1 * z.1 - t * z.0 + d;
                            let im = 2.0 * z.0 * z.1 - t * z.1;
                            assert!(re.hypot(im) < 1e-9, "{rule:?} {a} {beta}: {z:?}");
                        }
                        // its powers are the run: (x, v) from (x₀, 0)
                        let run = one(rule, eta, beta, lambda, 0.7, 60);
                        let mut v = [0.7, 0.0];
                        for (k, x) in run.iter().enumerate() {
                            if x.abs() < 1e6 {
                                assert!((v[0] - x).abs() <= 1e-9 * x.abs().max(1.0), "{rule:?} {eta} {lambda} {beta} step {k}: {} {}", v[0], x);
                            }
                            v = [m[0][0] * v[0] + m[0][1] * v[1], m[1][0] * v[0] + m[1][1] * v[1]];
                        }
                        // and they grow, or shrink, at the larger eigenvalue's size
                        let r = rate(rule, a, beta);
                        let measured = measured_rate(m, 4000);
                        assert!((measured - r).abs() < 0.01 * r.max(0.05), "{rule:?} {eta} {lambda} {beta}: {measured} against {r}");
                    }
                }
            }
        }
    }

    /// The recurrences of the dossier (curvature.md A2, A3) hold for the runs:
    /// heavy ball xₜ₊₁ = (1 + β − a)xₜ − βxₜ₋₁, Nesterov xₜ₊₁ = (1 + β)(1 − a)xₜ − β(1 − a)xₜ₋₁.
    #[test]
    fn the_runs_follow_the_dossiers_recurrences() {
        for &beta in &BETAS {
            for a in [0.01, 0.3, 1.0, 1.7, 3.1] {
                let h = one(Rule::Heavy, a, beta, 1.0, 1.0, 40);
                let n = one(Rule::Nesterov, a, beta, 1.0, 1.0, 40);
                // (while the runs are well short of where they are held)
                for t in (1..39).filter(|&t| h[t + 1].abs().max(n[t + 1].abs()) < 1e10) {
                    let scale = h[t].abs().max(h[t - 1].abs()).max(1.0);
                    assert!((h[t + 1] - ((1.0 + beta - a) * h[t] - beta * h[t - 1])).abs() < 1e-9 * scale, "heavy {a} {beta} {t}");
                    let scale = n[t].abs().max(n[t - 1].abs()).max(1.0);
                    assert!((n[t + 1] - ((1.0 + beta) * (1.0 - a) * n[t] - beta * (1.0 - a) * n[t - 1])).abs() < 1e-9 * scale, "Nesterov {a} {beta} {t}");
                }
                // and the first step of either is a plain one: nothing is kept yet
                assert!((h[1] - (1.0 - a)).abs() < 1e-12);
            }
        }
    }

    /// Heavy ball's map is the dossier's (curvature.md A2): the eigenvalues are equal at
    /// ηλ = (1 ± √β)², a complex pair of size exactly √β between, two positive below,
    /// the larger negative above, and the size reaches 1 at ηλ = 2 + 2β.
    #[test]
    fn heavy_balls_regions_are_the_dossiers() {
        for &beta in &BETAS[1..] {
            let e = edges(Rule::Heavy, beta);
            let r = beta.sqrt();
            assert!((e.critical - (1.0 - r).powi(2)).abs() < 1e-15 && (e.second - (1.0 + r).powi(2)).abs() < 1e-15 && (e.limit - (2.0 + 2.0 * beta)).abs() < 1e-15);
            let disc = |a: f64| (1.0 + beta - a).powi(2) - 4.0 * beta;
            assert!(disc(e.critical).abs() < 1e-12 && disc(e.second).abs() < 1e-12);
            for n in 1..40 {
                let a = e.critical + (e.second - e.critical) * n as f64 / 40.0;
                let [z, _] = eigenvalues(Rule::Heavy, a, beta);
                assert!(z.1 != 0.0 && (modulus(z) - r).abs() < 1e-12, "{a} {beta}");
                assert_eq!(kind(Rule::Heavy, a, beta), Kind::Rings);
            }
            for a in [0.5 * e.critical, 0.99 * e.critical] {
                let [z1, z2] = eigenvalues(Rule::Heavy, a, beta);
                assert!(z1.1 == 0.0 && z1.0 > 0.0 && z2.0 > 0.0 && z1.0 < 1.0);
                assert_eq!(kind(Rule::Heavy, a, beta), Kind::Creeps);
            }
            for a in [0.01, 0.5, 0.99].map(|u| e.second + (e.limit - e.second) * u) {
                let [z1, _] = eigenvalues(Rule::Heavy, a, beta);
                assert!(z1.1 == 0.0 && z1.0 < 0.0 && z1.0 > -1.0, "{a} {beta}");
                assert_eq!(kind(Rule::Heavy, a, beta), Kind::Flips);
            }
            assert_eq!(kind(Rule::Heavy, e.critical, beta), Kind::Critical);
            assert!((rate(Rule::Heavy, e.critical, beta) - r).abs() < 1e-7);
            assert_eq!(kind(Rule::Heavy, e.limit, beta), Kind::OnLimit);
            assert!((rate(Rule::Heavy, e.limit, beta) - 1.0).abs() < 1e-12);
            assert!(rate(Rule::Heavy, e.limit * 1.001, beta) > 1.0 && kind(Rule::Heavy, e.limit * 1.001, beta) == Kind::FliesOff);
            // the runs agree: just inside the limit a direction dies, just outside it grows
            let inside = one(Rule::Heavy, e.limit * 0.97, beta, 1.0, 1.0, MAX);
            let outside = one(Rule::Heavy, e.limit * 1.03, beta, 1.0, 1.0, 400);
            assert!(inside[MAX].abs() < 1e-3 && outside[400].abs() > 10.0, "{beta}");
        }
        // with no momentum it is plain descent: the factor 1 − ηλ, the limit 2
        for a in [0.1, 0.9, 1.5, 1.99] {
            assert!((rate(Rule::Heavy, a, 0.0) - (1.0 - a).abs()).abs() < 1e-12);
            assert!((rate(Rule::Plain, a, 0.0) - (1.0 - a).abs()).abs() < 1e-12);
        }
        assert_eq!(kind(Rule::Plain, 1.0, 0.0), Kind::Gone);
        assert_eq!(edges(Rule::Plain, 0.0).limit, 2.0);
    }

    /// Nesterov's map (curvature.md A3 for the limit; the rest derived): a complex pair
    /// for ((1 − β)/(1 + β))² < ηλ < 1, of size √(β(1 − ηλ)), which falls with the
    /// curvature; both eigenvalues 0 at ηλ = 1; one of each sign beyond, the negative
    /// the larger; and the limit (2 + 2β)/(1 + 2β), below plain descent's 2 for every
    /// β > 0 and far below heavy ball's.
    #[test]
    fn nesterovs_regions_and_its_lower_limit() {
        for &beta in &BETAS[1..] {
            let e = edges(Rule::Nesterov, beta);
            assert!((e.limit - (2.0 + 2.0 * beta) / (1.0 + 2.0 * beta)).abs() < 1e-15);
            assert!(e.limit < 2.0 && e.limit < edges(Rule::Heavy, beta).limit);
            let disc = |a: f64| ((1.0 + beta) * (1.0 - a)).powi(2) - 4.0 * beta * (1.0 - a);
            assert!(disc(e.critical).abs() < 1e-12, "{beta}");
            for n in 1..40 {
                let a = e.critical + (1.0 - e.critical) * n as f64 / 40.0;
                let [z, _] = eigenvalues(Rule::Nesterov, a, beta);
                assert!(z.1 != 0.0 && (modulus(z) - (beta * (1.0 - a)).sqrt()).abs() < 1e-12, "{a} {beta}");
                assert_eq!(kind(Rule::Nesterov, a, beta), Kind::Rings);
            }
            assert_eq!(kind(Rule::Nesterov, 1.0, beta), Kind::Gone);
            for a in [1.001, 0.5 * (1.0 + e.limit), e.limit * 0.999] {
                let [z1, z2] = eigenvalues(Rule::Nesterov, a, beta);
                assert!(z1.1 == 0.0 && z1.0 < 0.0 && z2.0 > 0.0 && z1.0 > -1.0, "{a} {beta}");
                assert_eq!(kind(Rule::Nesterov, a, beta), Kind::Flips);
            }
            assert!((rate(Rule::Nesterov, e.limit, beta) - 1.0).abs() < 1e-12);
            let inside = one(Rule::Nesterov, e.limit * 0.97, beta, 1.0, 1.0, MAX);
            let outside = one(Rule::Nesterov, e.limit * 1.03, beta, 1.0, 1.0, 400);
            assert!(inside[MAX].abs() < 1e-3 && outside[400].abs() > 10.0, "{beta}");
        }
        assert_eq!(num(edges(Rule::Nesterov, 0.9).limit, 3), "1.357");
        assert_eq!(num(edges(Rule::Heavy, 0.9).limit, 1), "3.8");
    }

    /// The best pair (curvature.md A2, from Goh 2017): every curvature from μ to L shrinks
    /// by (√κ − 1)/(√κ + 1) or less, the two extremes on the band's edges; a search over
    /// steps and momenta finds nothing better; and plain descent's best step 2/(μ + L)
    /// gives (κ − 1)/(κ + 1), which no other step beats.
    #[test]
    fn the_best_pair_and_the_best_step_are_the_dossiers() {
        for kappa in [4.0, 25.0, 100.0, 1000.0] {
            let (eta, beta, r) = best_pair(kappa);
            let s = kappa.sqrt();
            assert!((r - (s - 1.0) / (s + 1.0)).abs() < 1e-15 && (beta.sqrt() - r).abs() < 1e-15);
            let mu = 1.0 / kappa;
            assert!((eta * mu - (1.0 - r).powi(2)).abs() < 1e-12 && (eta - (1.0 + r).powi(2)).abs() < 1e-12);
            let worst = |rule: Rule, eta: f64, beta: f64| (0..=400).map(|n| mu * kappa.powf(n as f64 / 400.0)).map(|l| rate(rule, eta * l, beta)).fold(0.0f64, f64::max);
            assert!((worst(Rule::Heavy, eta, beta) - r).abs() < 1e-6, "{kappa}");
            let mut best_found = f64::INFINITY;
            for i in 1..=120 {
                for j in 0..=99 {
                    let (e, b) = (4.4 * i as f64 / 120.0, 0.99 * j as f64 / 99.0);
                    best_found = best_found.min(worst(Rule::Heavy, e, b));
                }
            }
            assert!(best_found >= r - 1e-6, "{kappa}: {best_found} below {r}");
            let (step, plain) = best_plain(kappa);
            assert!((plain - (kappa - 1.0) / (kappa + 1.0)).abs() < 1e-15);
            assert!((worst(Rule::Plain, step, 0.0) - plain).abs() < 1e-12);
            assert!((1..=400).map(|i| worst(Rule::Plain, 2.2 * i as f64 / 400.0, 0.0)).all(|w| w >= plain - 1e-12));
        }
        // e-fold in about √κ/2 and κ/2 steps, for κ = 100
        let fold = |r: f64| -1.0 / r.ln();
        assert_eq!((fold(best_pair(KAPPA).2).round(), fold(best_plain(KAPPA).1).round()), (5.0, 50.0));
    }

    /// On a steady slope the velocity builds up to g/(1 − β), and a direction of small
    /// curvature moves as plain descent would with the step η/(1 − β) (curvature.md A2,
    /// from source 17): the slow eigenvalue is about 1 − ηλ/(1 − β).
    #[test]
    fn momentum_on_a_steady_slope_is_a_longer_step() {
        for beta in [0.5f64, 0.9, 0.99] {
            let mut v = 0.0f64;
            for _ in 0..5000 {
                v = beta * v + 1.0;
            }
            assert!((v - 1.0 / (1.0 - beta)).abs() < 1e-9);
            let a = 1e-3 * (1.0 - beta).powi(2);
            let slow = eigenvalues(Rule::Heavy, a, beta)[0].0;
            assert!(((1.0 - slow) / (a / (1.0 - beta)) - 1.0).abs() < 0.01, "{beta}");
        }
    }

    /// Momentum written as an average, m ← βm + (1 − β)g, w ← w − η′m, takes the same
    /// steps when η′ = η/(1 − β) (curvature.md A4), and its limit is then
    /// η′λ < (2 + 2β)/(1 − β): 38 for β = 0.9.
    #[test]
    fn the_average_form_is_the_same_rule() {
        for beta in [0.5, 0.9] {
            for a in [0.02, 0.7, 2.5] {
                let heavy = one(Rule::Heavy, a, beta, 1.0, 1.0, 50);
                let eta2 = a / (1.0 - beta);
                let (mut x, mut m) = (1.0f64, 0.0f64);
                for (t, h) in heavy.iter().enumerate().skip(1) {
                    m = beta * m + (1.0 - beta) * x;
                    x -= eta2 * m;
                    assert!((x - h).abs() < 1e-9 * h.abs().max(1.0), "{beta} {a} {t}");
                }
            }
        }
        assert_eq!(num(edges(Rule::Heavy, 0.9).limit / (1.0 - 0.9), 6), "38.000000");
    }

    /// Where the rate is a given value: flat across heavy ball's band (it is √β there,
    /// whatever ηλ), and not across Nesterov's.
    #[test]
    fn heavy_balls_contours_are_flat_across_the_band() {
        for c in LEVELS {
            let beta = c * c;
            let segs = contour(Rule::Heavy, c);
            // (the pieces inside the band at their own height, well away from its edges)
            let inside = |(s, b): (f64, f64)| {
                let e = edges(Rule::Heavy, b);
                s > e.critical.sqrt() + 0.05 && s < e.second.sqrt() - 0.05
            };
            let across: Vec<_> = segs.iter().filter(|[a, b]| inside(*a) && inside(*b)).collect();
            assert!(across.len() > 40, "{c}");
            assert!(across.iter().all(|[a, b]| (a.1 - beta).abs() < 0.0056 && (b.1 - beta).abs() < 0.0056), "{c}");
            let ns = contour(Rule::Nesterov, c);
            let band: Vec<f64> = ns.iter().filter(|[a, _]| a.0 > 0.3 && a.0 < 0.95).map(|[a, _]| a.1).collect();
            let (lo, hi) = band.iter().fold((f64::MAX, f64::MIN), |(l, h), b| (l.min(*b), h.max(*b)));
            assert!(hi - lo > 0.1, "{c}: Nesterov's from {lo} to {hi}");
        }
        for rho in [0.0, 0.3, 0.9, 0.999, 0.9999999, 1.0, 1.5, f64::NAN] {
            let c = shade(rho);
            assert!(rho < 1.0 || c == FLY);
        }
    }

    /// A film plays its first ten steps at four a second, then the rest of the run in
    /// three and a half seconds, never going back, and ends exactly at its span; the
    /// rate does not jump where the slow start ends.
    #[test]
    fn a_film_starts_slowly_and_ends_on_time() {
        for span in [6.0, 10.0, 20.0, 150.0, 350.0, 1750.0, MAX as f64] {
            let length = film_length(span);
            let mut last = 0.0;
            for n in 0..=600 {
                let at = film_step(span, length * n as f64 / 600.0);
                assert!(at >= last && at <= span, "{span} {n}");
                last = at;
            }
            assert!((film_step(span, length) - span).abs() < 1e-6 * span, "{span}");
            assert_eq!(film_step(span, 1.0), SLOW_RATE.min(span));
            let slow_end = SLOW_STEPS.min(span) / SLOW_RATE;
            if span > SLOW_STEPS + SLOW_RATE * FAST {
                let before = (film_step(span, slow_end) - film_step(span, slow_end - 0.01)) / 0.01;
                let after = (film_step(span, slow_end + 0.01) - film_step(span, slow_end)) / 0.01;
                assert!((before - SLOW_RATE).abs() < 1e-6 && (after - SLOW_RATE).abs() < 0.1, "{span}: {before} {after}");
            }
        }
        assert_eq!((film_length(6.0), film_length(350.0)), (1.5, 6.0));
    }

    /// A line cut to a rectangle keeps its colours where it is cut, in proportion.
    #[test]
    fn a_coloured_line_is_cut_at_the_edge() {
        let r = (Vec2::new(0.0, 0.0), Vec2::new(10.0, 10.0));
        let (white, clear) = (pal::WHITE, pal::WHITE.a(0.0));
        let cut = clip_colored(&[(Vec2::new(-10.0, 5.0), white), (Vec2::new(10.0, 5.0), clear), (Vec2::new(10.0, 30.0), clear)], r);
        assert_eq!(cut.len(), 3, "{cut:?}");
        assert!((cut[0].0.x - -0.5).abs() < 1e-4 && (cut[0].1 .0[3] as f32 - 255.0 * 10.5 / 20.0).abs() <= 1.0, "{cut:?}");
        assert!(cut[1].0 == Vec2::new(10.0, 5.0) && cut[1].1 .0[3] == 0);
        assert!(clip_colored(&[(Vec2::new(-5.0, -5.0), white), (Vec2::new(-1.0, -9.0), white)], r).is_empty());
        assert_eq!(clip_colored(&[(Vec2::new(1.0, 1.0), white)], r).len(), 1);
        // with four points a segment, whatever is inside: the same picture
        for line in [
            vec![(Vec2::new(-10.0, 5.0), white), (Vec2::new(10.0, 5.0), white), (Vec2::new(10.0, 30.0), white)],
            vec![(Vec2::new(-5.0, -5.0), white), (Vec2::new(-1.0, -9.0), white)],
            vec![(Vec2::new(1.0, 1.0), white), (Vec2::new(2.0, 3.0), white), (Vec2::new(30.0, 3.0), white)],
        ] {
            let fixed = clip_fixed(&line, r);
            assert_eq!(fixed.len(), 4 * (line.len() - 1));
            assert!(fixed.iter().all(|(p, _)| p.is_finite() && within_by(*p, r, 0.5)));
            // what is seen is what the cut line shows
            let seen: Vec<Vec2> = fixed.windows(2).filter(|w| w[0].1 .0[3] > 0 || w[1].1 .0[3] > 0).flat_map(|w| [w[0].0, w[1].0]).collect();
            let cut: Vec<Vec2> = clip_colored(&line, r).into_iter().map(|(p, _)| p).filter(|p| p.is_finite()).collect();
            assert!(cut.iter().all(|c| seen.iter().any(|s| (*s - *c).length() < 1e-4)), "{seen:?} {cut:?}");
        }
    }

    fn within_by(p: Vec2, (min, max): Rect, by: f32) -> bool {
        p.cmpge(min - Vec2::splat(by)).all() && p.cmple(max + Vec2::splat(by)).all()
    }

    #[test]
    fn steps_are_counted_from_where_the_loss_stays_down() {
        assert_eq!(steps_to(&[1.0, 0.1, 1e-7, 2e-6, 1e-7, 1e-8], 1e-6), Some(4));
        assert_eq!(steps_to(&[1.0, 1e-7], 1e-6), Some(1));
        assert_eq!(steps_to(&[1.0, 0.5, 2.0], 1e-6), None);
        assert_eq!(steps_to(&[1.0, f64::NAN, 1e-9], 1e-6), Some(2));
        assert_eq!(steps_to(&[], 1e-6), None);
        assert_eq!((round_up(131 * 112 / 100 + 1), round_up(7), round_up(1578)), (150, 10, 1750));
    }

    /// The scene as an experiment's button leaves it, run at 60 frames a second until
    /// it comes to rest.
    fn trial(i: usize) -> Momentum {
        let mut scene = Momentum::new();
        scene.set_up(i);
        let mut frames = 0;
        while scene.moving() {
            scene.advance(1.0 / 60.0, 1.0 / 60.0);
            frames += 1;
            assert!(frames < 60 * 30, "experiment {i} never comes to rest");
        }
        assert!(scene.film == Film::Rest && scene.shown == scene.span_wanted() as f64);
        scene
    }

    fn rates(s: &Momentum, rule: Rule) -> Vec<String> {
        (0..s.run.d()).map(|i| num(s.run.rate(rule, i), 3)).collect()
    }

    fn kinds(s: &Momentum, rule: Rule) -> Vec<Kind> {
        (0..s.run.d()).map(|i| s.run.kind(rule, i)).collect()
    }

    fn top_loss(p: &Path) -> (f64, usize) {
        p.loss.iter().enumerate().fold((0.0, 0), |(m, at), (t, l)| if *l > m { (*l, t) } else { (m, at) })
    }

    #[test]
    fn plain_descent_does_what_its_caption_says() {
        let s = trial(0);
        let r = &s.run;
        assert!(r.beta == 0.0 && r.eta == 1.9 && r.lams[r.d() - 1] == 1.0 && r.d() == 5 && s.rule == Rule::Heavy);
        assert_eq!(r.path(Rule::Heavy).x, r.path(Rule::Plain).x);
        assert_eq!(edges(Rule::Plain, 0.0).limit, 2.0);
        assert_eq!((r.kind(Rule::Plain, 4), num(r.rate(Rule::Plain, 4), 1), num(r.a(Rule::Plain, 4), 1)), (Kind::Flips, "0.9".into(), "1.9".into()));
        assert_eq!((r.kind(Rule::Plain, 0), num(r.rate(Rule::Plain, 0), 3), num(r.a(Rule::Plain, 0), 3)), (Kind::Creeps, "0.981".into(), "0.019".into()));
        assert_eq!(r.slowest(Rule::Heavy).1, vec![0]);
        assert_eq!(r.path(Rule::Plain).count, Some(278));
    }

    #[test]
    fn the_same_step_with_momentum_does_what_its_caption_says() {
        let s = trial(1);
        let r = &s.run;
        assert!(r.beta == 0.9 && r.eta == 1.9 && s.rule == Rule::Heavy);
        assert!(kinds(&s, Rule::Heavy).iter().all(|k| *k == Kind::Rings));
        assert!(rates(&s, Rule::Heavy).iter().all(|v| v == "0.949") && num(0.9f64.sqrt(), 3) == "0.949");
        assert_eq!(r.slowest(Rule::Heavy).1, vec![0, 1, 2, 3, 4]);
        // the sharpest rings for longer than plain descent's zigzag; the flattest speeds up
        assert!(r.rate(Rule::Heavy, 4) > r.rate(Rule::Plain, 4) && num(r.rate(Rule::Plain, 4), 1) == "0.9");
        assert_eq!(num(r.rate(Rule::Plain, 0), 3), "0.981");
        assert_eq!((r.path(Rule::Heavy).count, r.path(Rule::Plain).count), (Some(131), Some(278)));
    }

    #[test]
    fn the_best_pair_does_what_its_caption_says() {
        let s = trial(2);
        let r = &s.run;
        assert!(s.plain_best && s.rule == Rule::Heavy);
        assert_eq!((num(r.lams[0], 2), num(r.lams[r.d() - 1], 0)), ("0.01".into(), "1".into()));
        assert_eq!((num(r.beta, 3), num(r.eta, 2), num(r.plain_eta, 2)), ("0.669".into(), "3.31".into(), "1.98".into()));
        // the flattest on the band's left edge, the sharpest on its right edge, all at √β
        let e = edges(Rule::Heavy, r.beta);
        assert!((r.a(Rule::Heavy, 0) - e.critical).abs() < 1e-12 && (r.a(Rule::Heavy, 4) - e.second).abs() < 1e-12);
        assert_eq!(kinds(&s, Rule::Heavy), vec![Kind::Critical, Kind::Rings, Kind::Rings, Kind::Rings, Kind::Flips]);
        assert!(rates(&s, Rule::Heavy).iter().all(|v| v == "0.818"));
        assert_eq!(r.slowest(Rule::Heavy).1, vec![0, 1, 2, 3, 4]);
        // plain descent's worst two
        assert_eq!(r.slowest(Rule::Plain).1, vec![0, 4]);
        assert_eq!(num(r.slowest(Rule::Plain).0, 3), "0.980");
        assert_eq!((r.path(Rule::Heavy).count, r.path(Rule::Plain).count), (Some(57), Some(337)));
        // the loss first rises ninefold, at step 4, and is above plain descent's until step 19
        let (top, at) = top_loss(r.path(Rule::Heavy));
        assert!(at == 4 && (top / r.path(Rule::Heavy).loss[0]).round() == 9.0, "{top} {at}");
        let (h, p) = (&r.path(Rule::Heavy).loss, &r.path(Rule::Plain).loss);
        assert!(h[0] == p[0] && (1..=19).all(|t| h[t] > p[t]) && (20..=MAX).all(|t| h[t] < p[t]));
    }

    #[test]
    fn too_much_momentum_does_what_its_caption_says() {
        let s = trial(3);
        let r = &s.run;
        assert!(r.beta == 0.99 && r.eta == 1.9);
        assert!(kinds(&s, Rule::Heavy).iter().all(|k| *k == Kind::Rings));
        assert!(rates(&s, Rule::Heavy).iter().all(|v| v == "0.995") && num(0.99f64.sqrt(), 3) == "0.995");
        assert!(r.slowest(Rule::Heavy).0 > r.slowest(Rule::Plain).0 && num(r.slowest(Rule::Plain).0, 3) == "0.981");
        assert_eq!((r.path(Rule::Heavy).count, r.path(Rule::Plain).count), (Some(1409), Some(278)));
        let (top, at) = top_loss(r.path(Rule::Heavy));
        assert_eq!((num(top / r.path(Rule::Heavy).loss[0], 2), at), ("1.27".into(), 12));
        // the whole run is drawn, inside the most steps there are
        assert!(s.span_wanted() >= 1409 && s.span_wanted() <= MAX);
    }

    #[test]
    fn a_step_plain_descent_cannot_bear_does_what_its_caption_says() {
        let s = trial(4);
        let r = &s.run;
        assert!(r.beta == 0.9 && r.eta == 3.0 && r.plain_eta == 3.0);
        assert_eq!(s.followed(), 4);
        let [z, _] = eigenvalues(Rule::Plain, r.a(Rule::Plain, 4), 0.0);
        assert_eq!((z.0, r.kind(Rule::Plain, 4)), (-2.0, Kind::FliesOff));
        assert!(r.path(Rule::Plain).count.is_none() && count_words(r.path(Rule::Plain)) == "never: it flies off");
        assert_eq!(num(edges(Rule::Heavy, 0.9).limit, 1), "3.8");
        assert!(kinds(&s, Rule::Heavy).iter().all(|k| *k == Kind::Rings) && rates(&s, Rule::Heavy).iter().all(|v| v == "0.949"));
        assert_eq!(r.path(Rule::Heavy).count, Some(141));
        let (h, p) = (r.path(Rule::Heavy), r.path(Rule::Plain));
        assert!((0..r.d()).all(|i| h.at(1, i, r.d()) == p.at(1, i, r.d())));
        let (top, at) = top_loss(h);
        assert_eq!((num(top / h.loss[0], 1), at), ("2.7".into(), 1));
    }

    #[test]
    fn nesterov_does_what_its_caption_says() {
        let mut s = trial(5);
        assert!(s.rule == Rule::Nesterov && s.followed() == 4);
        let r = &s.run;
        assert!(r.beta == 0.9 && r.eta == 1.9);
        assert_eq!(r.path(Rule::Heavy).count, Some(131));
        assert!(r.path(Rule::Nesterov).count.is_none() && count_words(r.path(Rule::Nesterov)) == "never: it flies off");
        assert_eq!((r.kind(Rule::Nesterov, 4), num(r.rate(Rule::Nesterov, 4), 2)), (Kind::FliesOff, "2.10".into()));
        let others: Vec<f64> = (0..4).map(|i| r.rate(Rule::Nesterov, i)).collect();
        assert_eq!((num(others[0], 3), num(others[3], 3)), ("0.940".into(), "0.599".into()));
        assert!(others.windows(2).all(|w| w[0] > w[1]) && others.iter().all(|v| *v < r.rate(Rule::Heavy, 0)));
        assert!((0..4).all(|i| r.kind(Rule::Nesterov, i) == Kind::Rings));
        // at a step both can bear
        s.by_hand(1.2, 0.9, KAPPA);
        s.advance(1.0 / 60.0, 1.0 / 60.0);
        assert_eq!((s.run.path(Rule::Nesterov).count, s.run.path(Rule::Heavy).count), (Some(81), Some(132)));
    }

    /// The scene opens at rest on the second experiment, drawn whole; a film comes
    /// exactly to rest at any display's rate.
    #[test]
    fn it_opens_at_rest_and_a_film_comes_to_rest() {
        let s = Momentum::new();
        assert!(!s.moving() && s.run.beta == 0.9 && s.run.eta == 1.9 && s.shown == s.span_wanted() as f64);
        for hz in [30.0f32, 60.0, 144.0, 240.0] {
            let mut s = Momentum::new();
            s.toggle_play();
            // the run drawn stays put as it fades out; the film then starts at step 0 and
            // only goes forwards
            let (mut frames, mut last, mut rewound) = (0, s.shown, false);
            while s.moving() {
                s.advance(1.0 / hz, 1.0 / hz);
                if rewound {
                    assert!(s.shown >= last && s.fade() == 1.0, "{hz}: back from {last} to {}", s.shown);
                } else if matches!(s.film, Film::Fade { .. }) {
                    assert!(s.shown == last && s.fade() < 1.0, "{hz}: the run drawn moved while it faded");
                } else {
                    rewound = s.shown < 1.0;
                    assert!(rewound, "{hz}: the film started at {}", s.shown);
                }
                last = s.shown;
                frames += 1;
                assert!((frames as f64 / hz as f64) < film_length(MAX as f64) + FADE + 2.0, "still moving at {hz} Hz");
            }
            assert!(rewound && s.shown == s.span_wanted() as f64);
        }
    }

    /// Whatever is pressed or dragged, in whatever order, with frames of any length,
    /// everything drawn from stays finite and in range.
    #[test]
    fn any_input_leaves_everything_finite() {
        let mut rng = Rng::new(5);
        let mut s = Momentum::new();
        for n in 0..1500 {
            match rng.below(10) {
                0 => s.set_up(rng.below(TRIALS.len() + 2)),
                1 => s.by_hand(rng.range(-1.0, 9.0) as f64, rng.range(-1.0, 2.0) as f64, rng.range(0.0, 2000.0) as f64),
                2 => s.by_hand([ETA_MIN, ETA_MAX][rng.below(2)], [0.0, BETA_MAX][rng.below(2)], [1.0, KAPPA_MAX][rng.below(2)]),
                3 => s.set_rule([Rule::Heavy, Rule::Nesterov][rng.below(2)]),
                4 => s.d = D_MIN + rng.below(D_MAX - D_MIN + 1),
                5 => s.toggle_play(),
                6 => s.follow = Some(rng.below(12)),
                7 => {
                    s.plain_best = !s.plain_best;
                    let t = s.ease.target;
                    s.by_switch(t.eta, t.beta, t.kappa);
                }
                _ => {}
            }
            let dt = [0.0, 1.0 / 240.0, 1.0 / 60.0, 0.5, 4.0][rng.below(5)];
            s.advance(dt, dt);
            let r = &s.run;
            assert!(r.paths.iter().all(|p| p.x.iter().chain(&p.loss).all(|v| v.is_finite())), "{n}");
            assert!(s.shown.is_finite() && s.shown >= 0.0 && s.shown <= MAX as f64 && s.followed() < r.d(), "{n}");
            assert!(s.span.shown.iter().chain(&s.height.shown).all(|v| v.is_finite()), "{n}");
            for i in 0..r.d() {
                assert!(r.rate(s.rule, i).is_finite() && r.a(s.rule, i) > 0.0, "{n}");
            }
        }
        // a round bowl: every direction the same
        s.by_hand(1.0, 0.5, 1.0);
        s.advance(0.1, 0.1);
        assert!(s.run.lams.iter().all(|l| *l == 1.0) && s.run.slowest(s.rule).1.len() == s.run.d());
    }
}

#[cfg(test)]
mod numbers {
    use super::*;

    /// Prints what each experiment shows, for writing the captions.
    #[test]
    #[ignore]
    fn print_numbers() {
        for i in 0..TRIALS.len() {
            let mut s = Momentum::new();
            s.set_up(i);
            s.ease.snap(s.ease.target);
            s.remake();
            let r = &s.run;
            println!("== {} : eta {} beta {} kappa {} plain_eta {}", TRIALS[i].label, r.eta, r.beta, s.ease.target.kappa, r.plain_eta);
            for rule in [Rule::Plain, Rule::Heavy, Rule::Nesterov] {
                let p = r.path(rule);
                let l0 = p.loss[0];
                let (mx, at) = p.loss.iter().enumerate().fold((0.0f64, 0), |(m, a), (t, l)| if *l > m { (*l, t) } else { (m, a) });
                println!("  {:?}: count {:?}  max loss/start {:.4} at {}  slowest {:?}", rule, p.count, mx / l0, at, r.slowest(rule));
                for k in 0..r.d() {
                    println!("     dir {}: a {:.5} rate {:.5} kind {:?} eig {:?}", k + 1, r.a(rule, k), r.rate(rule, k), r.kind(rule, k), eigenvalues(rule, r.a(rule, k), r.beta_of(rule)));
                }
                println!("     loss first: {:?}", p.loss[..8].iter().map(|l| format!("{:.4}", l / l0)).collect::<Vec<_>>());
            }
            println!("  span {}  followed {}", s.span_wanted(), s.followed());
        }
        println!("best pair {:?} best plain {:?}", best_pair(KAPPA), best_plain(KAPPA));
    }
}




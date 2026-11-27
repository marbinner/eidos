//! The noise scale and the critical batch size. How large should a batch be? A larger
//! batch gives a cleaner gradient and so allows a longer step, until the curvature, not
//! the noise, is what limits the step. Four charts:
//!
//! (a) one step of SGD in the model of McCandlish, Kaplan and Amodei (2018, §2.2): the
//! expected change of the loss against the step, a parabola for each batch size, their
//! lowest points on one straight line;
//! (b) the best gain of one step against the batch size: proportional to B below the
//! noise scale, a ceiling above it;
//! (c) and (d) a whole run: the steps to a target against the batch size, and the steps
//! against the examples, from one of three sources: the one-step model with its noise
//! scale held fixed (the authors' relation between steps and examples), the noisy
//! quadratic model of Zhang and co-authors (2019) for three rules, each at its best
//! step (exact recursions, [`terrain::Nqm`]), and this scene's own small network,
//! trained by SGD at every batch size at its best step (a measurement).
//!
//! The three numbers of the one-step model, |G|², Gᵀ H G and tr(H Σ), come from that
//! network at a moment of its full-batch run (measured exactly with Hessian-vector
//! products, [`terrain::FullNoise`]) or are set by hand. Everything that takes longer
//! than a frame (the noisy quadratic model's search for the best step at each batch
//! size, the network's runs and its noise scale along a run) was computed beforehand:
//! the tables below, regenerated and compared by the ignored tests at the end.

use super::*;
use crate::math::terrain::{self, OneStep};
use crate::scenes::stats::plot::{self, Glide, Plot, Rect, AXES, MARKS};

pub const DEF: SceneDef = SceneDef {
    id: "noisescale",
    title: "The noise scale and the critical batch size",
    group: GROUP,
    blurb: "Bigger batches allow longer steps until curvature, not noise, is the limit: how many steps a larger batch saves, and the batch size beyond which it saves almost none.",
    make: |_| Box::new(NoiseScale::create()),
};

// ----- the one-step model -------------------------------------------------------------------

/// The largest batch the controls reach (2²⁴).
const MOST_BATCH: f64 = 16_777_216.0;
/// The ranges of the three numbers when they are set by hand.
const HAND_G2: [f32; 2] = [1e-6, 10.0];
const HAND_GHG: [f32; 2] = [1e-7, 10.0];
const HAND_TRHS: [f32; 2] = [1e-5, 1e4];

/// A batch size as the controls keep it: a whole number from 1 to `MOST_BATCH` (a value
/// that is not a number becomes 1).
fn batch_kept(b: f64) -> f64 {
    if b.is_nan() {
        1.0
    } else {
        b.round().max(1.0).min(MOST_BATCH)
    }
}

/// The noise scale B_noise = tr(H Σ) / Gᵀ H G when it is a batch size (positive and
/// finite), else None: Gᵀ H G ≤ 0 (the loss does not curve up along the gradient), or
/// no noise.
fn noise_scale(one: &OneStep) -> Option<f64> {
    let b = one.noise_scale();
    (b.is_finite() && b > 0.0 && one.ghg > 0.0).then_some(b)
}

/// The best one-step gain at batch b, when the model has one (c(B) = Gᵀ H G + tr(H Σ)/B
/// > 0 and G ≠ 0); `b` may be infinite (the true gradient).
fn best_gain(one: &OneStep, b: f64) -> Option<f64> {
    let g = one.best_gain(b);
    (g.is_finite() && g > 0.0).then_some(g)
}

/// The batch sizes drawn for context in chart (a): powers of 4 round the noise scale
/// (round the chosen batch if there is no noise scale), at least 1.
fn context(one: &OneStep, b: f64) -> [f64; 5] {
    let centre = noise_scale(one).unwrap_or(b).max(1.0).min(MOST_BATCH);
    let base = 2f64.powf(centre.log2().round());
    [base / 16.0, base / 4.0, base, base * 4.0, base * 16.0].map(|v| v.max(1.0).min(MOST_BATCH))
}

// ----- a whole run: McCandlish's relation, fitted -----------------------------------------------

/// The relation between steps and examples of McCandlish and co-authors (2018, Eqs. 2.11
/// to 2.12; course notes, noise.md §1.4 and D4) fitted to runs at fixed batch sizes:
/// S(B) = S_min (1 + B_crit / B), so that E(B) = B S(B) = E_min (1 + B / B_crit) with
/// E_min = S_min B_crit, and (S/S_min − 1)(E/E_min − 1) = 1.
#[derive(Clone, Copy, Debug, PartialEq)]
struct Fit {
    s_min: f64,
    b_crit: f64,
    /// The root mean square of the residuals of ln S.
    rms: f64,
}

impl Fit {
    fn e_min(&self) -> f64 {
        self.s_min * self.b_crit
    }

    /// The fitted steps at batch b (b infinite: S_min).
    fn steps(&self, b: f64) -> f64 {
        self.s_min * (1.0 + self.b_crit / b)
    }
}

/// The least-squares fit of ln S = ln S_min + ln(1 + B_crit / B) to the points (B, S)
/// (B infinite for the true gradient), done "in log-space" as its source fits it (App.
/// A.3). For a given B_crit the best ln S_min is the mean of ln S − ln(1 + B_crit/B), so
/// only B_crit is searched: on its logarithm, a scan of 241 points from 10⁻³ to 10⁹ and
/// then golden sections between the neighbours of the best. None with fewer than two
/// usable points.
fn fit(points: &[(f64, f64)]) -> Option<Fit> {
    let pts: Vec<(f64, f64)> =
        points.iter().filter(|(b, s)| *b > 0.0 && *s > 0.0 && s.is_finite()).map(|&(b, s)| (if b.is_finite() { 1.0 / b } else { 0.0 }, s.ln())).collect();
    if pts.len() < 2 {
        return None;
    }
    let n = pts.len() as f64;
    let cost = |lc: f64| -> (f64, f64) {
        let c = lc.exp();
        let mean = pts.iter().map(|(inv, ls)| ls - (c * inv).ln_1p()).sum::<f64>() / n;
        let ss = pts.iter().map(|(inv, ls)| (ls - (c * inv).ln_1p() - mean).powi(2)).sum::<f64>();
        (ss, mean)
    };
    let (lo, hi) = (1e-3f64.ln(), 1e9f64.ln());
    let grid = |k: usize| lo + (hi - lo) * k as f64 / 240.0;
    let best = (0..=240).min_by(|&a, &b| cost(grid(a)).0.total_cmp(&cost(grid(b)).0))?;
    let (mut a, mut b) = (grid(best.saturating_sub(1)), grid((best + 1).min(240)));
    const PHI: f64 = 0.618_033_988_749_895;
    for _ in 0..80 {
        let (x1, x2) = (b - PHI * (b - a), a + PHI * (b - a));
        if cost(x1).0 < cost(x2).0 {
            b = x2;
        } else {
            a = x1;
        }
    }
    let lc = 0.5 * (a + b);
    let (ss, mean) = cost(lc);
    Some(Fit { s_min: mean.exp(), b_crit: lc.exp(), rms: (ss / n).sqrt() })
}

// ----- the noisy quadratic model: its table -------------------------------------------------------

/// One batch size of the noisy quadratic model at its best step: the steps to the
/// target, and the learning rate and momentum that give them.
#[derive(Clone, Copy, Debug, PartialEq)]
struct NqmRow {
    /// B (infinite: the true gradient, no noise).
    batch: f64,
    steps: u64,
    lr: f64,
    momentum: f64,
}

/// The batch sizes of the model's table: 1, 2, 4, ..., 2²⁰, and the true gradient.
const NQM_ROWS: usize = 22;
/// The preconditioner's power: P = H^½.
#[cfg(test)]
const NQM_POWER: f64 = 0.5;

/// The three rules of the noisy quadratic model the scene compares.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Rule {
    Sgd,
    Momentum,
    Preconditioned,
}

impl Rule {
    const ALL: [Rule; 3] = [Rule::Sgd, Rule::Momentum, Rule::Preconditioned];

    fn table(self) -> &'static [NqmRow; NQM_ROWS] {
        match self {
            Rule::Sgd => &NQM_SGD,
            Rule::Momentum => &NQM_MOMENTUM,
            Rule::Preconditioned => &NQM_PRECONDITIONED,
        }
    }

    fn color(self) -> Color {
        match self {
            Rule::Sgd => SGD,
            Rule::Momentum => MOMENTUM,
            Rule::Preconditioned => PRECONDITIONED,
        }
    }

    fn name(self) -> &'static str {
        match self {
            Rule::Sgd => "SGD",
            Rule::Momentum => "momentum",
            Rule::Preconditioned => r"preconditioned by $H^{1/2}$",
        }
    }

    fn button(self) -> &'static str {
        match self {
            Rule::Sgd => "SGD",
            Rule::Momentum => "momentum",
            Rule::Preconditioned => "preconditioned",
        }
    }

    /// McCandlish's relation fitted to the rule's steps at every batch size of the table.
    fn fit(self) -> Option<Fit> {
        fit(&self.table().iter().map(|r| (r.batch, r.steps as f64)).collect::<Vec<_>>())
    }
}

/// The noisy quadratic model's preconditioned rule: not a rule of the course, so a tint
/// of its own (the course's purple is RMSProp's).
const PRECONDITIONED: Color = Color::hex(0xC9B8FF);

// ----- the network: its tables ---------------------------------------------------------------------

/// One batch size of the network's measurement: the step size that reached the target in
/// the fewest steps (their mean over the seeds), and each seed's steps there.
#[derive(Clone, Copy, Debug, PartialEq)]
struct Row {
    /// B, drawn with replacement; 0 for the true gradient (gradient descent).
    batch: u32,
    /// The step size: point k of the grid, `lr_of(k)`.
    k: i32,
    steps: [u32; SEEDS],
}

impl Row {
    fn lr(&self) -> f64 {
        lr_of(self.k)
    }

    fn b(&self) -> f64 {
        if self.batch == 0 {
            f64::INFINITY
        } else {
            self.batch as f64
        }
    }

    fn mean(&self) -> f64 {
        self.steps.iter().map(|&s| s as f64).sum::<f64>() / SEEDS as f64
    }

    fn least(&self) -> f64 {
        self.steps.iter().copied().min().unwrap_or(0) as f64
    }

    fn most(&self) -> f64 {
        self.steps.iter().copied().max().unwrap_or(0) as f64
    }
}

/// The three numbers of the one-step model measured on the network at one step of its
/// full-batch run, and what else was measured there.
#[derive(Clone, Copy, Debug, PartialEq)]
struct Moment {
    /// Steps of the run taken.
    step: u32,
    /// The loss on all the data.
    loss: f64,
    /// |G|², Gᵀ H G, tr(H Σ) and tr Σ (Σ of divisor N).
    g2: f64,
    ghg: f64,
    trhs: f64,
    trace: f64,
}

impl Moment {
    fn one_step(&self) -> OneStep {
        OneStep { g2: self.g2, ghg: self.ghg, trhs: self.trhs }
    }

    /// B_simple = tr Σ / |G|².
    fn simple(&self) -> f64 {
        self.trace / self.g2
    }
}

/// What was measured at every step of a full-batch run (not only the moments kept).
#[derive(Clone, Copy, Debug, PartialEq)]
struct Summary {
    /// The mean over the steps of B_noise (McCandlish's B_crit ≈ ∫B ds / ∫ds) and of
    /// B_simple.
    mean_noise: f64,
    mean_simple: f64,
    /// Their quartiles over the steps.
    noise_quartiles: [f64; 3],
    simple_quartiles: [f64; 3],
    /// Steps at which Gᵀ H G was not positive (no noise scale there).
    curving_down: u32,
    /// The run's step size over the one-step model's best full-batch step ηmax: its
    /// quartiles over the steps after the first tenth of the run (where Gᵀ H G > 0).
    overstep: [f64; 3],
}

/// The network trained to one target.
#[derive(Clone, Copy, Debug)]
struct Target {
    /// The loss on all the data to reach.
    loss: f64,
    /// Every batch size at its best step size; the true gradient last.
    rows: &'static [Row],
    /// The full-batch run at its best step size (the last row), every few steps.
    moments: &'static [Moment],
    summary: Summary,
}

impl Target {
    fn full(&self) -> &Row {
        &self.rows[self.rows.len() - 1]
    }

    fn fit(&self) -> Option<Fit> {
        fit(&self.rows.iter().map(|r| (r.b(), r.mean())).collect::<Vec<_>>())
    }

    /// McCandlish's prediction of a whole run from the noise scale averaged over the
    /// full-batch run: S(B) = S_min (1 + B̄_noise / B), with S_min the full-batch run's.
    fn predicted(&self, b: f64) -> f64 {
        self.full().mean() * (1.0 + self.summary.mean_noise / b)
    }
}

/// The seeds of the mini-batches at every batch size.
const SEEDS: usize = 5;

// ----- what is shown ---------------------------------------------------------------------------

/// Where the three numbers of the one-step model come from.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Source {
    /// The network at a moment of its full-batch run.
    Network,
    /// Set by hand.
    Hand,
}

/// What the charts of a whole run show.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Whole {
    /// The one-step model, its noise scale held fixed through the run.
    Model,
    /// The noisy quadratic model, three rules.
    Nqm,
    /// The network, measured.
    Network,
}

impl Whole {
    const ALL: [Whole; 3] = [Whole::Model, Whole::Nqm, Whole::Network];

    fn button(self, narrow: bool) -> &'static str {
        match (self, narrow) {
            (Whole::Model, false) => "if the noise scale stayed",
            (Whole::Model, true) => "model",
            (Whole::Nqm, false) => "noisy quadratic model",
            (Whole::Nqm, true) => "quadratic",
            (Whole::Network, false) => "this network, measured",
            (Whole::Network, true) => "network",
        }
    }
}

/// In a narrow window the charts come two at a time.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Page {
    /// (a) and (b).
    Step,
    /// (c) and (d).
    Run,
}

/// Everything the picture depends on.
#[derive(Clone, Copy, Debug, PartialEq)]
struct Setup {
    source: Source,
    /// The network's target (an index of `TARGETS`) and the moment of its full-batch run.
    target: usize,
    moment: usize,
    /// The three numbers set by hand.
    hand: OneStep,
    /// The batch size B, and a second one drawn beside it.
    batch: f64,
    second: Option<f64>,
    /// Another moment of the run, whose best gain is drawn faint in (b).
    earlier: Option<usize>,
    whole: Whole,
    rule: Rule,
    page: Page,
}

/// The target the scene opens on.
const TARGET: usize = 1;

impl Setup {
    /// What the scene opens with: the network at its start, a batch at its noise scale.
    fn opening() -> Setup {
        let m = TARGETS[TARGET].moments[0];
        Setup {
            source: Source::Network,
            target: TARGET,
            moment: 0,
            hand: m.one_step(),
            batch: batch_kept(m.one_step().noise_scale()),
            second: None,
            earlier: None,
            whole: Whole::Model,
            rule: Rule::Sgd,
            page: Page::Step,
        }
    }

    /// Kept to the controls' ranges.
    fn kept(self) -> Setup {
        let target = self.target.min(TARGETS.len() - 1);
        let last = TARGETS[target].moments.len() - 1;
        let fix = |v: f64, [lo, hi]: [f32; 2], zero: bool| {
            if v.is_finite() {
                if zero && v <= 0.0 {
                    0.0
                } else {
                    v.max(lo as f64).min(hi as f64)
                }
            } else {
                lo as f64
            }
        };
        Setup {
            target,
            moment: self.moment.min(last),
            hand: OneStep { g2: fix(self.hand.g2, HAND_G2, false), ghg: fix(self.hand.ghg, HAND_GHG, true), trhs: fix(self.hand.trhs, HAND_TRHS, true) },
            batch: batch_kept(self.batch),
            second: self.second.map(batch_kept),
            earlier: self.earlier.map(|e| e.min(last)),
            ..self
        }
    }

    fn target(&self) -> &'static Target {
        &TARGETS[self.target.min(TARGETS.len() - 1)]
    }

    /// The moment of the network the numbers come from (when they do).
    fn moment(&self) -> &'static Moment {
        let ms = self.target().moments;
        &ms[self.moment.min(ms.len() - 1)]
    }

    /// The three numbers of the one-step model.
    fn one(&self) -> OneStep {
        match self.source {
            Source::Network => self.moment().one_step(),
            Source::Hand => self.hand,
        }
    }

    /// B_simple, when the numbers are the network's (by hand there is no tr Σ).
    fn simple(&self) -> Option<f64> {
        let s = self.moment().simple();
        (self.source == Source::Network && s.is_finite() && s > 0.0).then_some(s)
    }

    /// The earlier moment's numbers, if one is drawn.
    fn earlier(&self) -> Option<(&'static Moment, OneStep)> {
        let ms = self.target().moments;
        self.earlier.filter(|_| self.source == Source::Network).map(|e| &ms[e.min(ms.len() - 1)]).map(|m| (m, m.one_step()))
    }

    /// The step size of the network's full-batch run, when the numbers are the network's.
    fn run_lr(&self) -> Option<f64> {
        (self.source == Source::Network).then(|| self.target().full().lr())
    }

    /// The larger and the smaller of the batches drawn.
    fn big(&self) -> f64 {
        self.second.map_or(self.batch, |s| s.max(self.batch))
    }

    fn small(&self) -> f64 {
        self.second.map_or(self.batch, |s| s.min(self.batch))
    }
}

// ----- the experiments ---------------------------------------------------------------------------

struct Trial {
    button: &'static str,
    caption: &'static str,
}

const TRIALS: [Trial; 5] = [
    Trial { button: "far below the noise scale, and twice it", caption: r"The network at its start, where $B_{\text{noise}} = 499$, with batches of 16 and of 32 (faint), both far below it. The larger batch's best step is 1.94 times as long and gains 1.94 times as much: on the gain chart both lie on the straight rise. If the noise scale stayed the same, a run with batches of 32 would take 1.94 times fewer steps, for 3.1 % more examples: below the noise scale a larger batch costs hardly any more examples. Shallue and co-authors (2019), tuning the step at every batch size, found this regime, which they call perfect scaling, in every workload they trained; it ended anywhere from a batch of $2^4$ to one of $2^{13}$." },
    Trial { button: "far above it", caption: r"The same numbers, with batches of 8 192 and 16 384, 16 and 33 times the noise scale. Their best steps are within 6 % of the true gradient's, $\eta_{\max} = 3.70$: the curvature, not the noise, now sets the step. Twice the batch gains 3.0 % more in a step; a run would take 2.9 % fewer steps, for 1.94 times the examples. Shallue and co-authors (2019) found that beyond some batch size the steps stopped falling altogether (maximal data parallelism, in their words), and never saw them rise as the batch grew." },
    Trial { button: "momentum and a preconditioner", caption: r"The noisy quadratic model of Zhang and co-authors (2019), each rule at its best learning rate (and momentum) at each batch size. At small batches momentum is no faster than SGD: at $B = 1$, 6 022 909 steps against 6 024 935, with the same effective step $\eta/(1 - \beta) = 0.0034$. With the true gradient it needs 165 steps against SGD's 6 673, so its curve bends far later: McCandlish and co-authors' critical batch size, fitted to each curve, is 36 700 against 863. The preconditioner is 4.3 times faster even at $B = 1$, and bends at 18 300. The model's authors drew the same conclusions from it, and in their networks (image classifiers and a Transformer) found larger critical batch sizes with the preconditioned rules they tried, Adam among them, than with momentum SGD." },
    Trial { button: "the network early and late", caption: r"The network's noise scale at the start of its full-batch run (faint) and at its end, at a loss of 0.30: $B_{\text{noise}}$ fell from 499 to 72.4, $B_{\text{simple}}$ from 178 to 105. In between it was far from steady: near step 20, where the true gradient almost vanished, it reached 302 000; over the middle half of the run's steps it lay between 22.6 and 61.0. McCandlish and co-authors found the noise scale growing during training in their experiments, typically tenfold or more; here, up to this target, it did not. For the middle half of its steps after the first tenth, the run's own step was between 1.8 and 2.1 times $\eta_{\max}$: McCandlish and co-authors saw the like with a fixed step size, training settling where the best step along the update was almost exactly half the step taken." },
    Trial { button: "the measured curve against the model's", caption: r"Plain SGD on the network, each batch at its best step size of the grid, 5 seeds (dots: their mean; bars: the least and the most). The steps nearly halve with each doubling of the batch up to 8 (by 1.8 to 2.0 times), fall by less from there, and stop falling beyond 256. McCandlish and co-authors' relation fits the means within 11 %, with $B_{\text{crit}} = 36.9$. The noise scale measured along the full-batch run lay between 22.6 and 61.0 for the middle half of its steps: the bend is near it. But its mean over the run, 7 710, which the one-step model takes as its prediction (red), is 209 times too large: 48 of its first 65 steps, where the true gradient almost vanished, make 99 % of it. Across their tasks McCandlish and co-authors found the simple noise scale predicting the critical batch size within about an order of magnitude; Merrill and co-authors (2025), measuring the critical batch size directly for language models trained with Adam (OLMo, 1 and 7 billion parameters), found the noise scale below it by several orders of magnitude." },
];

/// The setup of experiment k (anything else: the opening).
fn trial(k: usize) -> Setup {
    let open = Setup::opening();
    let last = TARGETS[TARGET].moments.len() - 1;
    match k {
        0 => Setup { batch: 16.0, second: Some(32.0), ..open },
        1 => Setup { batch: 8192.0, second: Some(16384.0), ..open },
        2 => Setup { batch: 1024.0, whole: Whole::Nqm, rule: Rule::Momentum, page: Page::Run, ..open },
        3 => Setup { moment: last, earlier: Some(0), batch: 64.0, ..open },
        4 => Setup { batch: 64.0, whole: Whole::Network, page: Page::Run, ..open },
        _ => open,
    }
}

// ----- numbers as the scene writes them ---------------------------------------------------------

/// A number with three significant digits, as text: 0.0334, 499, 41 700 (thousands set
/// apart); in a formula (dollars and all) when it is below 0.001 or from a million up:
/// $6.71 \times 10^{-5}$. A true minus sign. "?" for a value that is not a number.
fn sig(v: f64) -> String {
    let bare = sig_bare(v);
    if bare.contains('e') {
        format!("${bare}$")
    } else {
        bare
    }
}

/// The same for use inside a formula (no dollars): 6.71e-5 is set as 6.71 × 10⁻⁵ there.
fn sig_bare(v: f64) -> String {
    if v.is_nan() {
        return "?".to_string();
    }
    if v.is_infinite() {
        return if v > 0.0 { r"\infty".to_string() } else { r"-\infty".to_string() };
    }
    if v == 0.0 {
        return "0".to_string();
    }
    let a = v.abs();
    if !(1e-3..999_500.0).contains(&a) {
        let e = a.log10().floor() as i32;
        let mut m = a / 10f64.powi(e);
        let mut e = e;
        if m >= 9.995 {
            m /= 10.0;
            e += 1;
        }
        let ms = num(m, 2);
        let ms = ms.trim_end_matches('0').trim_end_matches('.');
        return format!("{}{ms}e{e}", if v < 0.0 { "−" } else { "" });
    }
    if a >= 999.5 {
        let unit = 10f64.powi(a.log10().floor() as i32 - 2);
        let n = count(((a / unit).round() * unit) as usize);
        return if v < 0.0 { format!("−{n}") } else { n };
    }
    let decimals = (2 - a.log10().floor() as i32).clamp(0, 12) as usize;
    num(v, decimals)
}

/// A batch size: a whole number, thousands set apart; ∞ for the true gradient.
fn batch_name(b: f64) -> String {
    if b.is_infinite() {
        r"$\infty$".to_string()
    } else {
        count(b.max(0.0).round() as usize)
    }
}

/// A ratio near 1 with three decimals, else three significant digits.
fn ratio(v: f64) -> String {
    if v.is_finite() && (0.9..1.1).contains(&v) {
        num(v, 3)
    } else {
        sig(v)
    }
}

// ----- the layout --------------------------------------------------------------------------------

/// The four charts, in the order of `Layout::plots`.
const A: usize = 0;
const B: usize = 1;
const C: usize = 2;
const D: usize = 3;

/// Where everything goes in a window (logical pixels from its top-left corner).
#[derive(Clone, Copy, Debug, PartialEq)]
struct Layout {
    view: Vec2,
    /// Small text (a phone).
    narrow: bool,
    /// Two charts at a time, chosen by the page buttons.
    paged: bool,
    text: f32,
    small: f32,
    /// The charts' plots, where their titles go, and whether each is shown.
    plots: [Rect; 4],
    heads: [Vec2; 4],
    shown: [bool; 4],
    /// The buttons: the pages (paged windows only), the sources of a whole run, the rules
    /// (the noisy quadratic model only).
    pages: [Rect; 2],
    wholes: [Rect; 3],
    rules: [Rect; 3],
    /// Only chart (b): the window is too small for more.
    compact: bool,
}

impl Layout {
    fn of(view: Vec2, inset: f32, page: Page, whole: Whole) -> Layout {
        let view = view.max(Vec2::splat(1.0));
        let narrow = view.x < 700.0;
        let paged = view.x < 760.0 || view.y < 600.0;
        let (text, small) = if narrow { (12.0, 10.5) } else { (13.0, 11.5) };
        let side = if narrow { 10.0 } else { 18.0 };
        let top = inset.max(0.0) + if paged { 6.0 } else { 10.0 };
        let none = (Vec2::ZERO, Vec2::ZERO);
        let mut lay = Layout {
            view,
            narrow,
            paged,
            text,
            small,
            plots: [none; 4],
            heads: [Vec2::ZERO; 4],
            shown: [false; 4],
            pages: [none; 2],
            wholes: [none; 3],
            rules: [none; 3],
            compact: false,
        };
        let head = text + 8.0;
        let (left, right, below) = (if narrow { 46.0 } else { 54.0 }, 10.0, 34.0);
        // a cell: its title at the top, its plot inside the room for the numbers round it
        let cell = |lay: &mut Layout, k: usize, x0: f32, y0: f32, x1: f32, y1: f32| {
            lay.heads[k] = Vec2::new(x0, y0 + text + 2.0);
            let (a, b) = (Vec2::new(x0 + left, y0 + head + 10.0), Vec2::new(x1 - right, y1 - below));
            lay.plots[k] = (a, b.max(a + Vec2::splat(1.0)));
            lay.shown[k] = true;
        };
        let bh = if paged { 24.0 } else { 26.0 };
        if view.x < 260.0 || view.y < 300.0 {
            lay.compact = true;
            cell(&mut lay, B, side, top, view.x - side, view.y - 4.0);
            return lay;
        }
        let three = |x0: f32, x1: f32, y: f32| -> [Rect; 3] {
            let row = plot::even_row(3, x0, x1, y, bh, 6.0);
            [row[0], row[1], row[2]]
        };
        if !paged {
            let gap = 26.0;
            let col = (view.x - 2.0 * side - gap) * 0.5;
            let row = (view.y - top - 8.0 - bh - 18.0) * 0.5;
            let (x0, x1) = (side, side + col + gap);
            cell(&mut lay, A, x0, top, x0 + col, top + row);
            cell(&mut lay, B, x1, top, x1 + col, top + row);
            let by = top + row + 6.0;
            lay.wholes = three(x0, x0 + col, by);
            if whole == Whole::Nqm {
                lay.rules = three(x1, x1 + col, by);
            }
            let y2 = by + bh + 12.0;
            cell(&mut lay, C, x0, y2, x0 + col, view.y - 8.0);
            cell(&mut lay, D, x1, y2, x1 + col, view.y - 8.0);
        } else {
            let row = plot::even_row(2, side, view.x - side, top, bh, 6.0);
            lay.pages = [row[0], row[1]];
            let mut y = top + bh + 8.0;
            let (k0, k1) = match page {
                Page::Step => (A, B),
                Page::Run => (C, D),
            };
            if page == Page::Run {
                lay.wholes = three(side, view.x - side, y);
                y += bh + 6.0;
                if whole == Whole::Nqm {
                    lay.rules = three(side, view.x - side, y);
                    y += bh + 6.0;
                }
            }
            let h = (view.y - y - 6.0) * 0.5;
            cell(&mut lay, k0, side, y, view.x - side, y + h);
            cell(&mut lay, k1, side, y + h, view.x - side, view.y - 6.0);
        }
        lay
    }
}

// ----- what is drawn, gliding ------------------------------------------------------------------------

/// Points of each curve of charts (a), (b) and (c).
const A_N: usize = 72;
const B_N: usize = 96;
const C_N: usize = 96;
/// The batch sizes of chart (a)'s curves: the context, the true gradient, the second,
/// the chosen.
const A_CURVES: usize = 8;
/// Where the true gradient (B = ∞) is drawn across a chart of whole runs: this far
/// right of the largest batch of the table (in log₁₀ B).
const INF_GAP: f64 = 0.6;

/// The batch sizes of chart (a)'s curves, in the order drawn.
fn a_batches(s: &Setup) -> [f64; A_CURVES] {
    let c = context(&s.one(), s.batch);
    [c[0], c[1], c[2], c[3], c[4], f64::INFINITY, s.second.unwrap_or(s.batch), s.batch]
}

/// log₁₀ of x at point k of n across a range.
fn across(k: usize, n: usize, lo: f64, hi: f64) -> f64 {
    lo + (hi - lo) * k as f64 / (n - 1).max(1) as f64
}

/// The steps of a whole run against B in the one-step model with its noise scale held
/// fixed, over the fewest: 1 + B_noise / B (`terrain::steps_over_min`).
fn model_steps(bn: f64, b: f64) -> f64 {
    terrain::steps_over_min(b, bn)
}

/// The examples likewise: 1 + B / B_noise.
fn model_examples(bn: f64, b: f64) -> f64 {
    terrain::examples_over_min(b, bn)
}

/// The largest batch of a whole run's table and where the true gradient goes across.
fn table_reach(whole: Whole, target: &Target) -> (f64, f64) {
    let top = match whole {
        Whole::Nqm => NQM_SGD[NQM_ROWS - 2].batch,
        _ => target.rows.iter().filter(|r| r.batch > 0).map(|r| r.batch as f64).fold(1.0, f64::max),
    };
    (top, top.log10() + INF_GAP)
}

/// x across a chart of whole runs for a batch (the true gradient at `inf_at`).
fn x_of(b: f64, inf_at: f64) -> f64 {
    if b.is_finite() {
        b.max(1e-300).log10()
    } else {
        inf_at
    }
}

/// What the charts show, as it is to be: each chart's ranges (log₁₀ where its axes are
/// logarithmic), its content (curves and marks), and which of the content has a value
/// (a mark that has none is not drawn; its number is a stand-in that glides).
struct Goal {
    frames: [Vec<f32>; 4],
    content: [Vec<f32>; 4],
    present: [Vec<bool>; 4],
}

/// A value for the glide (f32) of a number that may not have one: (stand-in, present).
fn val(v: Option<f64>, or: f64) -> (f32, bool) {
    match v {
        Some(x) if x.is_finite() => (x as f32, true),
        _ => (or as f32, false),
    }
}

fn lg(v: f64) -> Option<f64> {
    (v.is_finite() && v > 0.0).then(|| v.log10())
}

impl Setup {
    /// Chart (a)'s ranges: [step at the right, change at the bottom, at the top].
    fn a_frame(&self) -> [f64; 3] {
        let one = self.one();
        let big = self.big();
        let e = one.best_step(big);
        if let (Some(g), true) = (best_gain(&one, big), e.is_finite() && e > 0.0) {
            return [2.4 * e, -1.25 * g, 1.05 * g];
        }
        // no lowest point for the batches chosen (the loss falls ever faster as the step
        // grows, or there is no gradient): a range that shows the line as it is
        let x = if one.ghg != 0.0 && one.g2 > 0.0 { one.g2 / one.ghg.abs() } else { 1.0 };
        let x = if x.is_finite() && x > 0.0 { x.min(1e6) } else { 1.0 };
        let low = one.change(x, big);
        if low.is_finite() && low < 0.0 {
            [x, 1.25 * low, -0.6 * low]
        } else if low.is_finite() && low > 0.0 {
            [x, -0.6 * low, 1.25 * low]
        } else {
            [x, -1.0, 1.0]
        }
    }

    /// Chart (b)'s ranges, log₁₀: [B from, to, gain from, to].
    fn b_frame(&self) -> [f64; 4] {
        let one = self.one();
        let mut far = [lg(self.big()), noise_scale(&one).and_then(|b| lg(b * 300.0)), self.simple().and_then(|b| lg(b * 4.0))]
            .into_iter()
            .flatten()
            .fold(3.0, f64::max);
        let mut gains: Vec<f64> = vec![best_gain(&one, 1.0), best_gain(&one, f64::INFINITY), best_gain(&one, self.big())].into_iter().flatten().collect();
        if let Some((_, e)) = self.earlier() {
            far = far.max(noise_scale(&e).and_then(|b| lg(b * 300.0)).unwrap_or(0.0));
            gains.extend([best_gain(&e, 1.0), best_gain(&e, f64::INFINITY)].into_iter().flatten());
        }
        let far = far.min(MOST_BATCH.log10() + 0.3);
        let lo = gains.iter().copied().fold(f64::INFINITY, f64::min);
        let hi = gains.iter().copied().fold(0.0, f64::max);
        let (y0, y1) = if lo.is_finite() && lo > 0.0 && hi > 0.0 { (lo.log10() - 0.35, hi.log10() + 0.4) } else { (-3.0, 0.0) };
        [0.0, far, y0, y1.max(y0 + 0.8)]
    }

    /// Chart (c)'s ranges, log₁₀: [B from, to, steps from, to].
    fn c_frame(&self) -> [f64; 4] {
        match self.whole {
            Whole::Model => {
                let b = self.b_frame();
                let top = noise_scale(&self.one()).map_or(1.0, |bn| model_steps(bn, 1.0).log10());
                [b[0], b[1], -0.12, top.max(1.0) + 0.25]
            }
            Whole::Nqm | Whole::Network => {
                let (_, inf_at) = table_reach(self.whole, self.target());
                let steps = self.whole_steps();
                let lo = steps.iter().copied().fold(f64::INFINITY, f64::min);
                let hi = steps.iter().copied().fold(0.0, f64::max);
                [-0.15, inf_at + 0.3, lo.max(1.0).log10() - 0.25, hi.max(10.0).log10() + 0.25]
            }
        }
    }

    /// Every number of steps chart (c) draws (for its range).
    fn whole_steps(&self) -> Vec<f64> {
        match self.whole {
            Whole::Model => Vec::new(),
            Whole::Nqm => Rule::ALL.iter().flat_map(|r| r.table().iter().map(|row| row.steps as f64)).collect(),
            Whole::Network => {
                let t = self.target();
                let mut v: Vec<f64> = t.rows.iter().flat_map(|r| [r.least(), r.most()]).collect();
                // (the prediction from the averaged noise scale, up to two decades above the data)
                let most = v.iter().copied().fold(0.0, f64::max);
                v.push(t.predicted(1.0).min(most * 100.0));
                v
            }
        }
    }

    /// Chart (d)'s ranges, log₁₀: [examples from, to, steps from, to] (in the model, over
    /// their fewest).
    fn d_frame(&self) -> [f64; 4] {
        match self.whole {
            Whole::Model => {
                let (x, y) = match noise_scale(&self.one()) {
                    Some(bn) => (model_examples(bn, self.big()).log10(), model_steps(bn, self.small()).log10()),
                    None => (2.0, 2.0),
                };
                [-0.12, x.max(2.0).min(8.0) + 0.25, -0.12, y.max(2.0).min(8.0) + 0.25]
            }
            Whole::Nqm | Whole::Network => {
                let pts = self.whole_points();
                let (mut x0, mut x1, mut y0, mut y1) = (f64::INFINITY, 0.0f64, f64::INFINITY, 0.0f64);
                for (e, s) in pts {
                    x0 = x0.min(e);
                    x1 = x1.max(e);
                    y0 = y0.min(s);
                    y1 = y1.max(s);
                }
                if !(x0.is_finite() && y0.is_finite() && x1 > 0.0 && y1 > 0.0) {
                    return [0.0, 1.0, 0.0, 1.0];
                }
                [x0.log10() - 0.2, x1.log10() + 0.2, y0.log10() - 0.25, y1.log10() + 0.25]
            }
        }
    }

    /// The (examples, steps) of every run with a finite batch that chart (d) draws.
    fn whole_points(&self) -> Vec<(f64, f64)> {
        match self.whole {
            Whole::Model => Vec::new(),
            Whole::Nqm => Rule::ALL.iter().flat_map(|r| r.table().iter().filter(|row| row.batch.is_finite()).map(|row| (row.batch * row.steps as f64, row.steps as f64))).collect(),
            Whole::Network => self.target().rows.iter().filter(|r| r.batch > 0).map(|r| (r.b() * r.mean(), r.mean())).collect(),
        }
    }

    /// Everything the charts show, as it is to be.
    fn goal(&self) -> Goal {
        let one = self.one();
        let bn = noise_scale(&one);
        let mut frames: [Vec<f32>; 4] = Default::default();
        let mut content: [Vec<f32>; 4] = Default::default();
        let mut present: [Vec<bool>; 4] = Default::default();
        let mut push = |k: usize, (v, p): (f32, bool)| {
            content[k].push(v);
            present[k].push(p);
        };

        // (a): the line of the lowest points, the run's own step, each batch's parabola
        // and its lowest point
        // (its ranges glide as a zoom: the logarithms of the step at the right and of the
        // scale of the changes, and the changes in that scale; so does what is drawn in it)
        let fa = self.a_frame();
        let scale = fa[1].abs().max(fa[2].abs()).max(1e-300);
        frames[A] = vec![fa[0].max(1e-300).ln() as f32, scale.ln() as f32, (fa[1] / scale) as f32, (fa[2] / scale) as f32];
        let (lo, hi) = (fa[1] / scale, fa[2] / scale);
        let (y_floor, y_ceil) = (lo - 3.0 * (hi - lo), hi + 3.0 * (hi - lo));
        push(A, val(Some(-0.5 * one.g2 * fa[0] / scale), 0.0));
        push(A, val(self.run_lr().map(|lr| lr / fa[0]), 2.0));
        let batches = a_batches(self);
        for &b in &batches {
            for i in 0..A_N {
                let x = fa[0] * i as f64 / (A_N - 1) as f64;
                let y = one.change(x, b) / scale;
                push(A, val(Some(y.max(y_floor).min(y_ceil)), 0.0));
            }
        }
        for &b in &batches {
            let (e, g) = (one.best_step(b), best_gain(&one, b));
            let there = e.is_finite() && g.is_some();
            push(A, val(there.then_some(e / fa[0]), 0.0));
            push(A, val(g.filter(|_| there).map(|g| -g / scale), 0.0));
        }

        // (b): the best gain against B, and the earlier moment's; the marks
        let fb = self.b_frame();
        frames[B] = fb.iter().map(|&v| v as f32).collect();
        let gain_at = |o: &OneStep, x: f64| best_gain(o, 10f64.powf(x)).map(|g| g.log10());
        let earlier = self.earlier().map(|e| e.1);
        for o in [Some(one), earlier] {
            for i in 0..B_N {
                let x = across(i, B_N, fb[0], fb[1]);
                push(B, val(o.and_then(|o| gain_at(&o, x)), fb[2]));
            }
        }
        let max = best_gain(&one, f64::INFINITY);
        push(B, val(bn.and_then(lg), fb[1]));
        push(B, val(max.map(|m| (0.5 * m).log10()), fb[2]));
        push(B, val(self.simple().and_then(lg), fb[1]));
        for b in [Some(self.batch), self.second] {
            push(B, val(b.and_then(lg), 0.0));
            push(B, val(b.and_then(|b| best_gain(&one, b)).and_then(lg), fb[2]));
        }
        push(B, val(max.and_then(lg), fb[3]));
        let e_bn = earlier.as_ref().and_then(noise_scale);
        let e_max = earlier.as_ref().and_then(|e| best_gain(e, f64::INFINITY));
        push(B, val(e_bn.and_then(lg), fb[1]));
        push(B, val(e_max.map(|m| (0.5 * m).log10()), fb[2]));
        push(B, val(e_max.and_then(lg), fb[3]));

        // (c): steps against B
        let fc = self.c_frame();
        frames[C] = fc.iter().map(|&v| v as f32).collect();
        if self.whole == Whole::Model {
            for i in 0..C_N {
                let x = across(i, C_N, fc[0], fc[1]);
                push(C, val(bn.map(|bn| model_steps(bn, 10f64.powf(x)).log10()), 0.0));
            }
            push(C, val(bn.and_then(lg), fc[1]));
            for b in [Some(self.batch), self.second] {
                push(C, val(b.and_then(lg), 0.0));
                push(C, val(b.and_then(|b| bn.map(|bn| model_steps(bn, b).log10())), 0.0));
            }
        } else {
            push(C, val(lg(self.batch), 0.0));
        }

        // (d): steps against examples
        let fd = self.d_frame();
        frames[D] = fd.iter().map(|&v| v as f32).collect();
        if self.whole == Whole::Model {
            for b in [Some(self.batch), self.second] {
                push(D, val(b.and_then(|b| bn.map(|bn| model_examples(bn, b).log10())), 0.0));
                push(D, val(b.and_then(|b| bn.map(|bn| model_steps(bn, b).log10())), 0.0));
            }
        }
        Goal { frames, content, present }
    }
}

/// What is drawn, gliding to what it is to be.
#[derive(Clone, Debug, Default)]
struct Look {
    frames: [Glide; 4],
    content: [Glide; 4],
    present: [Vec<bool>; 4],
    moving: bool,
}

impl Look {
    /// Bring the picture `dt` seconds nearer to the goal (`snap`: there at once).
    fn follow(&mut self, goal: &Goal, dt: f32, snap: bool) {
        let mut moving = false;
        for k in 0..4 {
            if snap {
                self.frames[k].snap(&goal.frames[k]);
                self.content[k].snap(&goal.content[k]);
            } else {
                moving |= self.frames[k].follow(&goal.frames[k], dt, false);
                moving |= self.content[k].follow(&goal.content[k], dt, false);
            }
            self.present[k].clone_from(&goal.present[k]);
        }
        self.moving = moving;
    }

    /// Content value i of chart k as shown, if it has one.
    fn get(&self, k: usize, i: usize) -> Option<f64> {
        self.present[k].get(i).copied().unwrap_or(false).then(|| self.content[k].get(i) as f64)
    }

    fn frame(&self, k: usize, i: usize) -> f64 {
        self.frames[k].get(i) as f64
    }
}

// ----- drawing: axes, labels, batches -------------------------------------------------------------

// Lines, each curve that can be cut at its chart's edge in a batch of its own (its number
// of points changes as it moves across the edge, and in a shared batch that would seem
// to move every line after it).
/// The parabolas of the first chart (+ 0 to 7).
const K_PARA: u8 = plot::OWN;
/// The best gain against B, and the earlier moment's.
const K_GAIN: u8 = plot::OWN + 8;
const K_GAIN_EARLIER: u8 = plot::OWN + 9;
/// The steps against B (+ the rule: 0 to 2), and their fits.
const K_RUN: u8 = plot::OWN + 10;
const K_RUN_FIT: u8 = plot::OWN + 13;
/// The prediction from the averaged noise scale.
const K_PREDICT: u8 = plot::OWN + 16;
/// The steps against the examples (+ the rule), and their fits.
const K_TRADE: u8 = plot::OWN + 17;
const K_TRADE_FIT: u8 = plot::OWN + 20;
/// Straight lines of two points (cut to the chart), solid or dashed.
const K_SEG: u8 = plot::OWN + 23;

/// A curve given in a chart's units, cut to the chart, and drawn with a fixed number of
/// points in each piece of it that lies inside (`n`, evenly along the piece's length): as
/// the curve slides across the chart's edge its points slide along with it, where cut at
/// its sample points they would come and go (which `eidos motion` sees as the rest of the
/// batch jumping). A point that is not a number breaks the curve.
fn path_even(f: &mut Frame, p: &Plot, kind: u8, pts: impl IntoIterator<Item = (f64, f64)>, n: usize, width: f32, color: Color) {
    let screen: Vec<Vec2> = pts.into_iter().map(|(x, y)| p.at(x, y)).collect();
    let cut = plot::clip(&screen, p.rect);
    let mut out: Vec<Vec2> = Vec::new();
    for piece in cut.split(|q| !q.is_finite()).filter(|piece| piece.len() >= 2) {
        let mut along = vec![0.0f32; piece.len()];
        for i in 1..piece.len() {
            along[i] = along[i - 1] + (piece[i] - piece[i - 1]).length();
        }
        let total = along[piece.len() - 1];
        if total.is_nan() || total <= 0.0 {
            continue;
        }
        if !out.is_empty() {
            out.push(Vec2::NAN);
        }
        let mut j = 0;
        for k in 0..n.max(2) {
            let d = total * k as f32 / (n.max(2) - 1) as f32;
            while j + 2 < piece.len() && along[j + 1] < d {
                j += 1;
            }
            let span = (along[j + 1] - along[j]).max(1e-6);
            let t = ((d - along[j]) / span).clamp(0.0, 1.0);
            out.push(piece[j] + (piece[j + 1] - piece[j]) * t);
        }
    }
    plot::path_of(f, kind, &out, width, color);
}

/// The x of a chart's width at k of n points, from `lo` to `hi`.
fn spread(k: usize, n: usize, lo: f64, hi: f64) -> f64 {
    lo + (hi - lo) * k as f64 / (n - 1).max(1) as f64
}

/// A line of chart (d) in the tabled sources: its batch (the rule's place), colour, whether
/// it is the one chosen, its (B, steps), and McCandlish's relation fitted to it.
type Series = (u8, Color, bool, Vec<(f64, f64)>, Option<Fit>);

/// The place of a rule in `Rule::ALL`, for its batch.
fn rule_index(r: Rule) -> u8 {
    Rule::ALL.iter().position(|&q| q == r).unwrap_or(0) as u8
}

/// A straight line from a to b in a chart's units, cut to the chart, in the batch of
/// straight lines.
#[allow(clippy::too_many_arguments)]
fn seg(f: &mut Frame, p: &Plot, a: (f64, f64), b: (f64, f64), width: f32, color: Color, dashed: bool) {
    let cut = plot::clip(&[p.at(a.0, a.1), p.at(b.0, b.1)], p.rect);
    if let [u, v] = cut[..] {
        if dashed {
            plot::dashed_of(f, K_SEG, u, v, width, color);
        } else {
            plot::line_of(f, K_SEG, u, v, width, color);
        }
    }
}

/// The point batches: one for each chart's dots.
fn dot_batch<'b>(f: &'b mut Frame<'_>, k: usize) -> &'b mut crate::gfx::draw::PointBatch {
    f.dl.points_with(PointKind::Overlay, PointStyle { outline_px: 0.0, size_scale: 1.0 + 1e-4 * (k + 1) as f32, ..Default::default() })
}

/// Every batch the scene draws in, made in the same order in every frame, so that one
/// that is empty in a frame does not move the later ones up a place (see `eidos motion`).
fn hold_batches(f: &mut Frame) {
    let corner = f.hud(Vec2::ZERO);
    for kind in [AXES, plot::CURVES, MARKS, plot::BUTTONS].into_iter().chain(K_PARA..=K_SEG) {
        plot::apart(f, kind, false).segment(corner, corner, 1.0, Color::CLEAR);
    }
    for kind in [MARKS, K_SEG] {
        plot::apart(f, kind, true).segment(corner, corner, 1.0, Color::CLEAR);
    }
    for k in 0..4 {
        dot_batch(f, k);
    }
}

/// A dot in screen space, in chart k's batch (nothing if it is not finite).
fn dot(f: &mut Frame, k: usize, at: Vec2, size: f32, color: Color) {
    if at.is_finite() {
        let p = f.hud(at);
        dot_batch(f, k).push(p, size, color);
    }
}

/// A ring: a dot with a darker middle.
fn ring(f: &mut Frame, k: usize, at: Vec2, size: f32, color: Color) {
    dot(f, k, at, size, color);
    dot(f, k, at, size * 0.5, Color::hex(0x0B0F1A));
}

/// The ticks of a logarithmic axis from lo to hi (log₁₀): the whole decades (named),
/// and 2 and 5 of each where there are few decades.
fn log_ticks(lo: f64, hi: f64) -> Vec<(f64, bool)> {
    let mut v = Vec::new();
    if !(lo.is_finite() && hi.is_finite() && hi > lo) {
        return v;
    }
    let few = hi - lo < 1.6;
    let mut d = lo.floor() as i32;
    while (d as f64) <= hi + 1e-9 && v.len() < 80 {
        for (m, main) in [(1.0f64, true), (2.0, false), (5.0, false)] {
            let x = d as f64 + m.log10();
            if x >= lo - 1e-9 && x <= hi + 1e-9 && (main || few) {
                v.push((x, main || few));
            }
        }
        d += 1;
    }
    v
}

/// A tick of a logarithmic axis as it is written: 1, 10, 100, 1 000, 10 000, then
/// powers of ten (10⁵, 10⁻³); 2 and 5 times a power as numbers.
fn power_name(x: f64) -> String {
    let d = x.round();
    if (x - d).abs() < 1e-6 {
        return match d as i32 {
            0 => "1".to_string(),
            1 => "10".to_string(),
            2 => "100".to_string(),
            3 => count(1000),
            4 => count(10_000),
            e => format!("$10^{{{e}}}$"),
        };
    }
    sig(10f64.powf(x))
}

/// A tick of a linear axis whose ticks are `step` apart, with the digits the spacing
/// needs (in powers of ten when they are very small or very large).
fn lin_name(v: f64, step: f64) -> String {
    if v == 0.0 {
        return "0".to_string();
    }
    if step >= 1e-8 && v.abs() < 1e6 {
        let decimals = (-(step * (1.0 + 1e-9)).log10().floor()).clamp(0.0, 8.0) as usize;
        return num(v, decimals);
    }
    sig(v)
}

/// The axis along the bottom of a chart: its line, ticks and their numbers (logarithmic
/// or linear; the ticks of the range it is gliding to, `goal`, so that they move with
/// the glide instead of being chosen anew in every frame), its name at the right; for a chart of whole runs, the true gradient's
/// place (∞) at `inf_at`, set apart by a break.
fn axis_x(f: &mut Frame, p: &Plot, goal: &Plot, size: f32, log: bool, name: &str, inf_at: Option<f64>) {
    let (min, max) = p.rect;
    plot::line_of(f, AXES, Vec2::new(min.x, max.y), max, 1.0, pal::DIM.a(0.8));
    let mut ticks: Vec<(f64, String)> = Vec::new();
    if log {
        let reach = inf_at.map_or(goal.x[1], |i| (i - 0.35).min(goal.x[1]));
        for (x, named) in log_ticks(goal.x[0], reach) {
            ticks.push((x, if named { power_name(x) } else { String::new() }));
        }
    } else {
        let t = plot::Ticks::of(goal.x[0], goal.x[1], p.width());
        for v in t.values() {
            ticks.push((v, lin_name(v, t.step)));
        }
    }
    let mut last_right = f32::NEG_INFINITY;
    for (x, label) in ticks {
        if x < p.x[0] - 1e-9 || x > p.x[1] + 1e-9 {
            continue;
        }
        let at = p.px_x(x);
        plot::line_of(f, AXES, Vec2::new(at, max.y), Vec2::new(at, max.y + if label.is_empty() { 2.0 } else { 4.0 }), 1.0, pal::DIM.a(0.8));
        if !label.is_empty() {
            // (a number that would run into the one before is left out)
            let w = f.text_size(&label, size).x;
            if at - 0.5 * w > last_right + 4.0 {
                f.text_px(Vec2::new(at, max.y + 6.0), label, size, pal::DIM, Align2::CENTER_TOP);
                last_right = at + 0.5 * w;
            }
        }
    }
    if let Some(i) = inf_at {
        if i >= p.x[0] && i <= p.x[1] {
            let at = p.px_x(i);
            plot::line_of(f, AXES, Vec2::new(at, max.y), Vec2::new(at, max.y + 4.0), 1.0, pal::DIM.a(0.8));
            f.text_px(Vec2::new(at, max.y + 6.0), r"$\infty$", size, pal::DIM, Align2::CENTER_TOP);
            let gap = p.px_x(i - 0.3);
            for dx in [-2.0, 2.0] {
                plot::line_of(f, AXES, Vec2::new(gap + dx - 2.0, max.y + 3.0), Vec2::new(gap + dx + 2.0, max.y - 3.0), 1.0, pal::DIM);
            }
        }
    }
    f.text_px(Vec2::new(max.x, max.y + 8.0 + size), name, size, pal::DIM, Align2::RIGHT_TOP);
}

/// The axis up the left of a chart, with faint lines across at its named ticks; its name
/// at the top, inside, is returned as a label to be placed first.
fn axis_y(f: &mut Frame, p: &Plot, goal: &Plot, size: f32, log: bool, name: &str) -> ChartLabel {
    let (min, max) = p.rect;
    plot::line_of(f, AXES, min, Vec2::new(min.x, max.y), 1.0, pal::DIM.a(0.8));
    let ticks: Vec<(f64, String)> = if log {
        log_ticks(goal.y[0], goal.y[1]).into_iter().filter(|t| t.1).map(|(y, _)| (y, power_name(y))).collect()
    } else {
        let t = plot::Ticks::of(goal.y[0], goal.y[1], p.height() * 1.4);
        t.values().map(|v| (v, lin_name(v, t.step))).collect()
    };
    let mut last_top = f32::INFINITY;
    for (y, label) in ticks {
        if y < p.y[0] - 1e-9 || y > p.y[1] + 1e-9 {
            continue;
        }
        let at = p.px_y(y);
        plot::line_of(f, AXES, Vec2::new(min.x - 4.0, at), Vec2::new(min.x, at), 1.0, pal::DIM.a(0.8));
        plot::line_of(f, AXES, Vec2::new(min.x, at), Vec2::new(max.x, at), 1.0, pal::DIM.a(0.09));
        if at < last_top - size - 2.0 {
            f.text_px(Vec2::new(min.x - 6.0, at), label, size, pal::DIM, Align2::RIGHT_CENTER);
            last_top = at;
        }
    }
    label(Vec2::new(min.x + 2.0, min.y + 2.0), Align2::LEFT_TOP, name, pal::DIM)
}

/// A label of a chart: where it wants to be (the point its alignment anchors), what it
/// says, its colour; and another place to try if there is no room at the first (the
/// other side of its point, say).
struct ChartLabel {
    at: Vec2,
    align: Align2,
    text: String,
    color: Color,
    alt: Option<(Vec2, Align2)>,
    /// What the label is remembered by from frame to frame: its text, or a name of its
    /// own for one whose numbers change while it is dragged along.
    key: Option<&'static str>,
}

fn label(at: Vec2, align: Align2, text: impl Into<String>, color: Color) -> ChartLabel {
    ChartLabel { at, align, text: text.into(), color, alt: None, key: None }
}

impl ChartLabel {
    /// Remember the label by this name, not by its text.
    fn keyed(mut self, key: &'static str) -> ChartLabel {
        self.key = Some(key);
        self
    }

    /// The place to try if there is no room at the first.
    fn or(mut self, at: Vec2, align: Align2) -> ChartLabel {
        self.alt = Some((at, align));
        self
    }
}

/// Where a label of a chart was put: its words, which of its places and the slot it
/// took (above or below that place, in its heights), and how far from the place it is
/// drawn now (gliding to that slot when it changes).
#[derive(Clone, Debug, Default)]
struct Placed {
    key: String,
    alt: bool,
    slot: i32,
    drawn: f32,
}

/// The labels on plates, each at its own place if that is clear of those placed before
/// it, else the nearest clear place above or below it (its first place, then its other),
/// inside the rectangle, in the order given: the first keep their places; one with no
/// room is left out. A label keeps the place and slot it had while they are still
/// clear, and glides to a new slot (`memo` remembers them; true while one glides).
/// (After the noise floor's.)
fn place(f: &mut Frame, tags: Vec<ChartLabel>, size: f32, (min, max): Rect, memo: &mut Vec<Placed>) -> bool {
    let mut taken: Vec<Rect> = Vec::new();
    let mut now: Vec<Placed> = Vec::with_capacity(tags.len());
    let ease = 1.0 - (-f.real_dt.max(0.0) / 0.08).exp();
    let mut gliding = false;
    for t in tags {
        if !t.at.is_finite() {
            continue;
        }
        let w = f.text_size(&t.text, size) + Vec2::new(8.0, 4.0);
        let corner = |at: Vec2, align: Align2| {
            let x = match align.x() {
                egui::Align::Min => at.x,
                egui::Align::Center => at.x - 0.5 * w.x,
                egui::Align::Max => at.x - w.x,
            };
            let y = match align.y() {
                egui::Align::Min => at.y,
                egui::Align::Center => at.y - 0.5 * w.y,
                egui::Align::Max => at.y - w.y,
            };
            Vec2::new(x, y)
        };
        let fits = |c: Vec2| c.x >= min.x - 1.0 && c.y >= min.y - 1.0 && c.x + w.x <= max.x + 1.0 && c.y + w.y <= max.y + 1.0;
        let clear = |c: Vec2, taken: &[Rect]| taken.iter().all(|(a, b)| c.x + w.x <= a.x || b.x <= c.x || c.y + w.y <= a.y || b.y <= c.y);
        let step = w.y + 1.0;
        let spot = |alt: bool| if alt { t.alt.unwrap_or((t.at, t.align)) } else { (t.at, t.align) };
        let key = t.key.map_or_else(|| t.text.clone(), str::to_string);
        let before = memo.iter().find(|m| m.key == key).cloned();
        let mut tries: Vec<(bool, i32)> = before.as_ref().map(|b| (b.alt, b.slot)).into_iter().collect();
        for k in 0..13 {
            let slot = if k % 2 == 1 { -(k + 1) / 2 } else { k / 2 };
            tries.push((false, slot));
            if t.alt.is_some() {
                tries.push((true, slot));
            }
        }
        let found = tries.into_iter().find(|&(alt, k)| {
            let (at, align) = spot(alt);
            let c = corner(at + Vec2::new(0.0, k as f32 * step), align);
            fits(c) && clear(c, &taken)
        });
        let Some((alt, slot)) = found else { continue };
        let (at, align) = spot(alt);
        let c = corner(at + Vec2::new(0.0, slot as f32 * step), align);
        taken.push((c, c + w));
        let goal = slot as f32 * step;
        let drawn = match before {
            Some(b) if b.alt == alt && (b.drawn - goal).abs() > 0.5 => {
                gliding = true;
                b.drawn + (goal - b.drawn) * ease
            }
            _ => goal,
        };
        f.caption_px(at + Vec2::new(0.0, drawn), t.text.clone(), size, t.color, align);
        now.push(Placed { key, alt, slot, drawn });
    }
    *memo = now;
    gliding
}

// ----- the scene ----------------------------------------------------------------------------------

pub struct NoiseScale {
    setup: Setup,
    look: Look,
    /// The window last drawn in (a new one is drawn at once, without gliding).
    canvas: Option<Vec2>,
    /// Where each chart's labels were put, and whether one is gliding to a new place.
    placed: [Vec<Placed>; 4],
    labels_moving: bool,
    /// The batch is being dragged on chart (b).
    dragging: bool,
    /// McCandlish's relation fitted to each rule of the noisy quadratic model and to the
    /// network at each target (worked out once).
    nqm_fits: [Option<Fit>; 3],
    net_fits: Vec<Option<Fit>>,
    /// The charts' ranges as they are to be (their ticks are chosen from these).
    goal_frames: [Vec<f32>; 4],
}

impl NoiseScale {
    /// (Nothing here needs the graphics device, so the tests can make one too.)
    fn create() -> NoiseScale {
        let setup = Setup::opening();
        let mut s = NoiseScale {
            setup,
            look: Look::default(),
            canvas: None,
            placed: Default::default(),
            labels_moving: false,
            dragging: false,
            nqm_fits: Rule::ALL.map(Rule::fit),
            net_fits: TARGETS.iter().map(Target::fit).collect(),
            goal_frames: Default::default(),
        };
        s.look.follow(&setup.goal(), 0.0, true);
        s
    }

    /// Set up experiment k.
    fn set_up(&mut self, k: usize) {
        self.setup = trial(k).kept();
    }

    fn net_fit(&self) -> Option<Fit> {
        self.net_fits.get(self.setup.target).copied().flatten()
    }

    fn nqm_fit(&self, r: Rule) -> Option<Fit> {
        self.nqm_fits[Rule::ALL.iter().position(|&q| q == r).unwrap_or(0)]
    }

    /// The buttons over the picture; true if anything changed.
    fn buttons(&mut self, f: &mut Frame, lay: &Layout) -> bool {
        let size = lay.small;
        let mut changed = false;
        if lay.paged && !lay.compact {
            for (k, (name, page)) in [("one step", Page::Step), ("a whole run", Page::Run)].into_iter().enumerate() {
                if plot::button(f, lay.pages[k], name, size, self.setup.page == page) && self.setup.page != page {
                    self.setup.page = page;
                    changed = true;
                }
            }
        }
        if lay.shown[C] || lay.shown[D] {
            for (k, w) in Whole::ALL.into_iter().enumerate() {
                if plot::button(f, lay.wholes[k], w.button(lay.narrow), size, self.setup.whole == w) && self.setup.whole != w {
                    self.setup.whole = w;
                    changed = true;
                }
            }
            if self.setup.whole == Whole::Nqm {
                for (k, r) in Rule::ALL.into_iter().enumerate() {
                    if plot::button(f, lay.rules[k], r.button(), size, self.setup.rule == r) && self.setup.rule != r {
                        self.setup.rule = r;
                        changed = true;
                    }
                }
            }
        }
        changed
    }

    /// The batch, dragged along chart (b).
    fn drag(&mut self, f: &mut Frame, lay: &Layout) {
        if !lay.shown[B] {
            return;
        }
        let p = self.b_plot(lay);
        let m = 2 * B_N;
        let (Some(x), y) = (self.look.get(B, m + 3), self.look.get(B, m + 4)) else { return };
        let y = y.unwrap_or(p.y[0]);
        if let Some((x, _)) = p.handle(f, 1, x, y.max(p.y[0]).min(p.y[1]), SGD) {
            if x.is_finite() {
                self.setup.batch = batch_kept(10f64.powf(x));
                self.dragging = true;
            }
        }
    }

    /// Chart k with the ranges it is gliding to.
    fn goal_plot(&self, k: usize, lay: &Layout) -> Plot {
        let g = |i: usize| self.goal_frames[k].get(i).copied().unwrap_or(0.0) as f64;
        if k == A {
            let (x, scale) = (g(0).exp(), g(1).exp());
            Plot::new(lay.plots[A], [0.0, x], [g(2) * scale, g(3) * scale])
        } else {
            Plot::new(lay.plots[k], [g(0), g(1)], [g(2), g(3)])
        }
    }

    /// The step at the right of chart (a) and the scale of its changes, as shown.
    fn a_scale(&self) -> (f64, f64) {
        (self.look.frame(A, 0).exp(), self.look.frame(A, 1).exp())
    }

    fn a_plot(&self, lay: &Layout) -> Plot {
        let (x, scale) = self.a_scale();
        Plot::new(lay.plots[A], [0.0, x], [self.look.frame(A, 2) * scale, self.look.frame(A, 3) * scale])
    }

    fn b_plot(&self, lay: &Layout) -> Plot {
        let g = |i| self.look.frame(B, i);
        Plot::new(lay.plots[B], [g(0), g(1)], [g(2), g(3)])
    }

    fn c_plot(&self, lay: &Layout) -> Plot {
        let g = |i| self.look.frame(C, i);
        Plot::new(lay.plots[C], [g(0), g(1)], [g(2), g(3)])
    }

    fn d_plot(&self, lay: &Layout) -> Plot {
        let g = |i| self.look.frame(D, i);
        Plot::new(lay.plots[D], [g(0), g(1)], [g(2), g(3)])
    }

    /// Chart (a): one step's expected change of the loss against the step size.
    fn draw_a(&mut self, f: &mut Frame, lay: &Layout) {
        let p = self.a_plot(lay);
        if p.tiny() {
            return;
        }
        let g = self.goal_plot(A, lay);
        let s = self.setup;
        let one = s.one();
        let size = lay.small;
        p.backing(f, if lay.narrow { 46.0 } else { 54.0 }, 32.0, 8.0, 8.0);
        let title = if lay.narrow {
            "one step: the expected change of the loss"
        } else {
            r"One step of SGD: the expected change of the loss against the step size $\eta$"
        };
        f.text_px(lay.heads[A], title, lay.text, pal::TEXT, Align2::LEFT_BOTTOM);
        axis_x(f, &p, &g, size, false, r"step size $\eta$", None);
        let axis = axis_y(f, &p, &g, size, false, "change of the loss");
        let (min, max) = p.rect;
        // no change
        p.hline(f, AXES, 0.0, 1.0, pal::DIM.a(0.7), false);
        let mut labels: Vec<ChartLabel> = Vec::new();
        // the line through the lowest points
        let (_, scale) = self.a_scale();
        if let Some(end) = self.look.get(A, 0).map(|e| e * scale) {
            seg(f, &p, (0.0, 0.0), (p.x[1], end), 1.2, pal::DIM.a(0.8), false);
            let at = p.at(p.x[1], end);
            let inside = at.y.max(min.y + 4.0).min(max.y - 4.0);
            if !lay.narrow {
                labels.push(label(Vec2::new(max.x - 4.0, inside), Align2::RIGHT_BOTTOM, r"the lowest points: on the line $-\tfrac12|G|^2\eta$", pal::DIM));
            }
        }
        // the parabolas: the context faint, the true gradient white, the batches chosen
        let batches = a_batches(&s);
        let order = [0usize, 1, 2, 3, 4, 5, 6, 7];
        for &c in &order {
            if c == 6 && s.second.is_none() {
                continue;
            }
            // (a batch of the context that is one of those chosen is drawn as that one)
            if c < 5 && (batches[c] == s.batch || Some(batches[c]) == s.second) {
                continue;
            }
            let (color, width) = match c {
                5 => (SIGNAL, 1.6),
                6 => (SGD.a(0.6), 1.8),
                7 => (SGD, 2.6),
                _ => (pal::DIM.a(0.45), 1.0),
            };
            let pts: Vec<(f64, f64)> = (0..A_N).map(|i| (p.x[1] * i as f64 / (A_N - 1) as f64, self.look.get(A, 2 + c * A_N + i).map_or(f64::NAN, |y| y * scale))).collect();
            // (where the curve last is inside the chart, for a name when its lowest point is not)
            // (the lowest place of the curve where it first runs inside the chart: its lowest
            // point, or where it leaves the chart below; the true gradient is named there,
            // which moves smoothly from the one to the other as the chart's ranges glide)
            let inside = |&&(x, y): &&(f64, f64)| y >= p.y[0] && y <= p.y[1] && x <= p.x[1];
            let lowest_in = pts.iter().skip_while(|q| !inside(q)).take_while(inside).min_by(|a, b| a.1.total_cmp(&b.1)).copied();
            path_even(f, &p, K_PARA + c as u8, pts, A_N, width, color);
            if c == 5 {
                if let Some((x, y)) = lowest_in {
                    let q = p.at(x, y);
                    labels.insert(0, label(q + Vec2::new(-6.0, 2.0), Align2::RIGHT_TOP, "the true gradient", SIGNAL).or(q + Vec2::new(6.0, 2.0), Align2::LEFT_TOP));
                }
            }
            // its lowest point
            let at = 2 + A_CURVES * A_N + 2 * c;
            let lowest = match (self.look.get(A, at), self.look.get(A, at + 1)) {
                (Some(x), Some(y)) => Some(p.at(x * p.x[1], y * scale)),
                _ => None,
            };
            if let Some(q) = lowest {
                if plot::within(q, p.rect, 0.5) {
                    dot(f, 0, q, if c >= 5 { 6.0 } else { 3.5 }, if c < 5 { pal::DIM } else { color });
                    if c != 5 {
                        let keys = ["a0", "a1", "a2", "a3", "a4", "a5", "second", "batch"];
                        let tag = label(q + Vec2::new(5.0, 3.0), Align2::LEFT_TOP, format!("$B = {}$", batch_name(batches[c])), if c < 5 { pal::DIM } else { color }).keyed(keys[c]);
                        if c > 5 {
                            labels.insert(0, tag);
                        } else {
                            labels.push(tag);
                        }
                    }
                }
            }
        }
        // the longest step that still lowers the expected loss, for the batch chosen
        let at = 2 + A_CURVES * A_N + 2 * 7;
        if let Some(x) = self.look.get(A, at) {
            let x2 = 2.0 * x * p.x[1];
            if x2 > p.x[0] && x2 < p.x[1] {
                let q = p.at(x2, 0.0);
                plot::line_of(f, MARKS, q - Vec2::new(0.0, 6.0), q + Vec2::new(0.0, 6.0), 2.0, LIMIT);
                let name = if lay.narrow { r"$2\eta_{\text{opt}}$" } else { r"$2\eta_{\text{opt}}$: longer, and the loss rises" };
                labels.push(label(q + Vec2::new(-4.0, -6.0), Align2::RIGHT_BOTTOM, name, LIMIT).or(q + Vec2::new(4.0, -6.0), Align2::LEFT_BOTTOM));
            }
        }
        // the step the network's full-batch run takes
        if let Some(x) = self.look.get(A, 1).map(|t| t * p.x[1]) {
            if x > p.x[0] && x < p.x[1] {
                p.vline(f, MARKS, x, 1.0, pal::TEXT.a(0.35), true);
                labels.push(label(Vec2::new(p.px_x(x) + 4.0, min.y + 18.0), Align2::LEFT_TOP, format!(r"the run's own step, $\eta = {}$", sig_bare(x)), pal::TEXT.a(0.8)).keyed("run"));
            }
        }
        if one.g2 == 0.0 {
            labels.insert(0, label(Vec2::new(min.x + 8.0, min.y + 20.0), Align2::LEFT_TOP, "no gradient: no step changes the loss to first order", pal::TEXT));
        } else if best_gain(&one, s.big()).is_none() {
            labels.insert(
                0,
                label(Vec2::new(min.x + 8.0, min.y + 20.0), Align2::LEFT_TOP, r"no lowest point: $G^\top H G + \operatorname{tr}(H\Sigma)/B \le 0$", NOISE),
            );
        }
        labels.insert(0, axis);
        self.labels_moving |= place(f, labels, size, p.rect, &mut self.placed[A]);
    }

    /// Chart (b): the best gain of one step against the batch size.
    fn draw_b(&mut self, f: &mut Frame, lay: &Layout) {
        let p = self.b_plot(lay);
        if p.tiny() {
            return;
        }
        let g = self.goal_plot(B, lay);
        let s = self.setup;
        let size = lay.small;
        let (min, max) = p.rect;
        p.backing(f, if lay.narrow { 46.0 } else { 54.0 }, 32.0, 8.0, 8.0);
        let title = if lay.narrow { r"the best step's gain against $B$" } else { r"The best step's gain against the batch size $B$" };
        f.text_px(lay.heads[B], title, lay.text, pal::TEXT, Align2::LEFT_BOTTOM);
        axis_x(f, &p, &g, size, true, "batch size $B$", None);
        let axis = axis_y(f, &p, &g, size, true, "gain in one step");
        let m = 2 * B_N;
        let get = |i: usize| self.look.get(B, m + i);
        let mut labels: Vec<ChartLabel> = Vec::new();
        // the ceiling and the rising line
        if let Some(top) = get(7) {
            seg(f, &p, (p.x[0], top), (p.x[1], top), 1.2, SIGNAL.a(0.7), false);
            let name = if lay.narrow { r"the true gradient's" } else { r"the true gradient's best, $\Delta L_{\max}$" };
            labels.push(label(Vec2::new(max.x - 4.0, p.px_y(top) - 3.0), Align2::RIGHT_BOTTOM, name, SIGNAL));
            if let Some(bn) = get(0) {
                let from = p.x[0];
                let to = (bn + 0.5).min(p.x[1]);
                seg(f, &p, (from, top - bn + from), (to, top - bn + to), 1.2, pal::DIM.a(0.9), true);
                let name = if lay.narrow { r"$\propto B$" } else { r"$\propto B$: twice the batch, twice the gain" };
                let x = from + 0.15 * (bn - from).max(0.4);
                labels.push(label(p.at(x, top - bn + x) + Vec2::new(6.0, 2.0), Align2::LEFT_TOP, name, pal::DIM));
            }
        }
        // the earlier moment's curve, faint
        if let Some((em, _)) = s.earlier() {
            let pts: Vec<(f64, f64)> = (0..B_N).map(|i| (across(i, B_N, p.x[0], p.x[1]), self.look.get(B, B_N + i).unwrap_or(f64::NAN))).collect();
            path_even(f, &p, K_GAIN_EARLIER, pts, B_N, 1.6, SGD.a(0.35));
            if let (Some(bn), Some(half)) = (get(8), get(9)) {
                let q = p.at(bn, half);
                if plot::within(q, p.rect, 0.5) {
                    ring(f, 1, q, 7.0, NOISE.a(0.5));
                    labels.push(label(q + Vec2::new(6.0, 4.0), Align2::LEFT_TOP, format!("at step {}", count(em.step as usize)), NOISE.a(0.7)));
                }
            }
            if let Some(top) = get(10) {
                seg(f, &p, (p.x[0], top), (p.x[1], top), 1.0, SIGNAL.a(0.3), false);
            }
        }
        // the curve
        let pts: Vec<(f64, f64)> = (0..B_N).map(|i| (across(i, B_N, p.x[0], p.x[1]), self.look.get(B, i).unwrap_or(f64::NAN))).collect();
        path_even(f, &p, K_GAIN, pts, B_N, 2.4, SGD);
        // the noise scale and the simple one
        if let (Some(bn), Some(half)) = (get(0), get(1)) {
            if bn >= p.x[0] && bn <= p.x[1] {
                seg(f, &p, (bn, p.y[0]), (bn, half), 1.4, NOISE, true);
                let q = p.at(bn, half);
                // (not over the handle, when the batch is at the noise scale)
                let handle = get(3).zip(get(4)).map(|(x, y)| p.at(x, y));
                if handle.is_none_or(|h| (h - q).length() > 12.0) {
                    ring(f, 1, q, 9.0, NOISE);
                }
                let name = if lay.narrow {
                    format!(r"$B_{{\text{{noise}}}} = {}$", sig_bare(10f64.powf(bn)))
                } else {
                    format!(r"noise scale $B_{{\text{{noise}}}} = {}$: half the ceiling", sig_bare(10f64.powf(bn)))
                };
                labels.insert(0, label(q + Vec2::new(7.0, 3.0), Align2::LEFT_TOP, name, NOISE).or(q + Vec2::new(-7.0, 3.0), Align2::RIGHT_TOP));
            }
        }
        if let Some(bs) = get(2) {
            if bs >= p.x[0] && bs <= p.x[1] {
                seg(f, &p, (bs, p.y[0]), (bs, p.y[1]), 1.0, NOISE.a(0.45), true);
                labels.push(label(Vec2::new(p.px_x(bs) - 4.0, min.y + 20.0), Align2::RIGHT_TOP, format!(r"$B_{{\text{{simple}}}} = {}$", sig_bare(10f64.powf(bs))), NOISE.a(0.8)));
            }
        }
        // the batches chosen (the first is the handle)
        if let (Some(x), Some(y)) = (get(5), get(6)) {
            let q = p.at(x, y);
            if plot::within(q, p.rect, 0.5) {
                dot(f, 1, q, 8.0, SGD.a(0.7));
                labels.push(label(q + Vec2::new(-7.0, -5.0), Align2::RIGHT_BOTTOM, format!("$B = {}$", batch_name(s.second.unwrap_or(10f64.powf(x)))), SGD.a(0.85)).keyed("second"));
            }
        }
        if let Some(x) = get(3) {
            let q = get(4).map(|y| p.at(x, y));
            if let Some(q) = q.filter(|q| plot::within(*q, p.rect, 0.5)) {
                labels.insert(0, label(q + Vec2::new(-8.0, -6.0), Align2::RIGHT_BOTTOM, format!("$B = {}$", batch_name(s.batch)), SGD).keyed("batch"));
            } else if x >= p.x[0] && x <= p.x[1] {
                p.vline(f, MARKS, x, 1.0, SGD.a(0.6), true);
                labels.insert(0, label(Vec2::new(p.px_x(x) + 4.0, min.y + 4.0), Align2::LEFT_TOP, format!("$B = {}$: no best step", batch_name(s.batch)), SGD).keyed("batch"));
            }
        }
        labels.insert(0, axis);
        self.labels_moving |= place(f, labels, size, p.rect, &mut self.placed[B]);
    }
}

impl NoiseScale {
    /// The perfect-scaling line through a point, on log axes: steps falling as 1/B, cut
    /// to the chart. Returns a place along it for its name.
    fn perfect_line(f: &mut Frame, p: &Plot, x0: f64, y0: f64, color: Color) -> Option<Vec2> {
        let x1 = x0 + (y0 - p.y[0]) + 1.0;
        let cut = plot::clip(&[p.at(x0, y0), p.at(x1, y0 - (x1 - x0))], p.rect);
        let [a, b] = cut[..] else { return None };
        plot::dashed_of(f, K_SEG, a, b, 1.2, color);
        Some(a + (b - a) * 0.35)
    }

    /// Chart (c): the steps of a whole run against the batch size.
    fn draw_c(&mut self, f: &mut Frame, lay: &Layout) {
        let p = self.c_plot(lay);
        if p.tiny() {
            return;
        }
        let g = self.goal_plot(C, lay);
        let s = self.setup;
        let size = lay.small;
        let (min, max) = p.rect;
        p.backing(f, if lay.narrow { 46.0 } else { 54.0 }, 32.0, 8.0, 8.0);
        let t = s.target();
        let title = match (s.whole, lay.narrow) {
            (Whole::Model, false) => r"A whole run, if $B_{\text{noise}}$ stayed the same: steps over the fewest".to_string(),
            (Whole::Model, true) => r"a whole run: steps over the fewest".to_string(),
            (Whole::Nqm, false) => "The noisy quadratic model: steps to the target, each batch at its best step".to_string(),
            (Whole::Nqm, true) => "the quadratic model: steps to the target".to_string(),
            (Whole::Network, false) => format!("This network: steps to a loss of {}, each batch at its best step", short(t.loss)),
            (Whole::Network, true) => format!("the network: steps to a loss of {}", short(t.loss)),
        };
        f.text_px(lay.heads[C], title, lay.text, pal::TEXT, Align2::LEFT_BOTTOM);
        let mut labels: Vec<ChartLabel> = Vec::new();
        let axis: Option<ChartLabel>;
        match s.whole {
            Whole::Model => {
                axis_x(f, &p, &g, size, true, "batch size $B$", None);
                axis = Some(axis_y(f, &p, &g, size, true, r"$S/S_{\min}$"));
                let pts: Vec<(f64, f64)> = (0..C_N).map(|i| (across(i, C_N, p.x[0], p.x[1]), self.look.get(C, i).unwrap_or(f64::NAN))).collect();
                path_even(f, &p, K_RUN, pts, C_N, 2.4, SGD);
                // the fewest steps, and the perfect-scaling line
                seg(f, &p, (p.x[0], 0.0), (p.x[1], 0.0), 1.2, SIGNAL.a(0.7), false);
                labels.push(label(Vec2::new(max.x - 4.0, p.px_y(0.0) - 3.0), Align2::RIGHT_BOTTOM, r"the fewest steps, $S_{\min}$", SIGNAL));
                if let Some(bn) = self.look.get(C, C_N) {
                    if let Some(q) = Self::perfect_line(f, &p, p.x[0], bn - p.x[0], pal::DIM.a(0.9)) {
                        if !lay.narrow {
                            labels.push(label(q + Vec2::new(6.0, 0.0), Align2::LEFT_CENTER, "twice the batch, half the steps", pal::DIM));
                        }
                    }
                    let q = p.at(bn, 2f64.log10());
                    if plot::within(q, p.rect, 0.5) {
                        ring(f, 2, q, 9.0, NOISE);
                        let name = if lay.narrow { r"$B_{\text{crit}} = B_{\text{noise}}$" } else { r"$B_{\text{crit}} = B_{\text{noise}}$: twice the fewest steps" };
                        labels.insert(0, label(q + Vec2::new(7.0, -3.0), Align2::LEFT_BOTTOM, name, NOISE));
                    }
                } else {
                    labels.insert(0, label(Vec2::new(min.x + 8.0, min.y + 20.0), Align2::LEFT_TOP, r"no noise scale here: $B_{\text{noise}}$ is not a batch size", NOISE));
                }
                for (k, (c, sz)) in [(SGD, 9.0), (SGD.a(0.6), 7.0)].into_iter().enumerate() {
                    if let (Some(x), Some(y)) = (self.look.get(C, C_N + 1 + 2 * k), self.look.get(C, C_N + 2 + 2 * k)) {
                        let q = p.at(x, y);
                        if plot::within(q, p.rect, 0.5) && (k == 0 || s.second.is_some()) {
                            dot(f, 2, q, sz, c);
                        }
                    }
                }
            }
            Whole::Nqm => {
                let (top, inf_at) = table_reach(Whole::Nqm, t);
                axis_x(f, &p, &g, size, true, "batch size $B$", Some(inf_at));
                axis = Some(axis_y(f, &p, &g, size, true, "steps"));
                let sgd1 = NQM_SGD[0].steps as f64;
                if let Some(q) = Self::perfect_line(f, &p, 0.0, sgd1.log10(), pal::DIM.a(0.9)) {
                    if !lay.narrow {
                        labels.push(label(q + Vec2::new(6.0, 0.0), Align2::LEFT_CENTER, "twice the batch, half the steps", pal::DIM));
                    }
                }
                let order: Vec<Rule> = Rule::ALL.into_iter().filter(|&r| r != s.rule).chain([s.rule]).collect();
                for r in order {
                    let on = r == s.rule;
                    let color = if on { r.color() } else { r.color().a(0.45) };
                    let rows = r.table();
                    let pts: Vec<(f64, f64)> = rows.iter().filter(|row| row.batch.is_finite()).map(|row| (row.batch.log10(), (row.steps as f64).log10())).collect();
                    p.path(f, K_RUN + rule_index(r), pts.iter().copied(), if on { 2.2 } else { 1.4 }, color);
                    let last = rows[NQM_ROWS - 1];
                    let tail = rows[NQM_ROWS - 2];
                    seg(f, &p, (tail.batch.log10(), (tail.steps as f64).log10()), (inf_at, (last.steps as f64).log10()), 1.0, color, true);
                    for &(x, y) in pts.iter().chain([(inf_at, (last.steps as f64).log10())].iter()) {
                        let q = p.at(x, y);
                        if plot::within(q, p.rect, 0.5) {
                            dot(f, 2, q, if on { 4.5 } else { 3.0 }, color);
                        }
                    }
                    if let Some(fit) = self.nqm_fit(r) {
                        let end = top.log10() + 0.05;
                        path_even(f, &p, K_RUN_FIT + rule_index(r), (0..64).map(|k| spread(k, 64, 0.0, end)).map(|x| (x, fit.steps(10f64.powf(x)).log10())), 64, 1.0, color.a(0.6));
                        let q = p.at(fit.b_crit.log10(), (2.0 * fit.s_min).log10());
                        if plot::within(q, p.rect, 0.5) {
                            ring(f, 2, q, if on { 9.0 } else { 7.0 }, color);
                            let tag = label(q + Vec2::new(7.0, -3.0), Align2::LEFT_BOTTOM, format!(r"$B_{{\text{{crit}}}} \approx {}$", sig_bare(fit.b_crit)), color);
                            if on {
                                labels.insert(0, tag);
                            } else {
                                labels.push(tag);
                            }
                        }
                    }
                }
                if let Some(x) = self.look.get(C, 0) {
                    p.vline(f, MARKS, x, 1.0, pal::TEXT.a(0.3), false);
                }
            }
            Whole::Network => {
                let (top, inf_at) = table_reach(Whole::Network, t);
                axis_x(f, &p, &g, size, true, "batch size $B$", Some(inf_at));
                axis = Some(axis_y(f, &p, &g, size, true, "steps"));
                let rows = t.rows;
                let first = rows[0].mean();
                if let Some(q) = Self::perfect_line(f, &p, 0.0, first.log10(), pal::DIM.a(0.9)) {
                    if !lay.narrow {
                        labels.push(label(q + Vec2::new(6.0, 0.0), Align2::LEFT_CENTER, "twice the batch, half the steps", pal::DIM));
                    }
                }
                // the noise scale during the full-batch run: the middle half of its steps
                let [q1, _, q3] = t.summary.noise_quartiles;
                if q1 > 0.0 && q3 > q1 {
                    let (a, b) = (p.px_x(q1.log10()).max(min.x), p.px_x(q3.log10()).min(max.x));
                    if b > a {
                        plot::fill(f, (Vec2::new(a, max.y - 9.0), Vec2::new(b, max.y)), NOISE.a(0.5));
                        let name = if lay.narrow { r"$B_{\text{noise}}$, middle half" } else { r"$B_{\text{noise}}$ during the full-batch run: its middle half" };
                        labels.push(label(Vec2::new(a, max.y - 11.0), Align2::LEFT_BOTTOM, name, NOISE));
                    }
                }
                // the one-step model's prediction from the averaged noise scale
                let mean = t.summary.mean_noise;
                let end = top.log10() + 0.05;
                path_even(f, &p, K_PREDICT, (0..64).map(|k| spread(k, 64, 0.0, end)).map(|x| (x, t.predicted(10f64.powf(x)).log10())), 64, 1.4, NOISE.a(0.8));
                let q = p.at(mean.log10(), (2.0 * t.full().mean()).log10());
                if plot::within(q, p.rect, 0.5) && mean <= top {
                    ring(f, 2, q, 7.0, NOISE.a(0.8));
                }
                let name = if lay.narrow {
                    format!(r"the model, averaged $B_{{\text{{noise}}}} = {}$", sig_bare(mean))
                } else {
                    format!(r"the model with $B_{{\text{{noise}}}}$ averaged over the full-batch run, ${}$", sig_bare(mean))
                };
                let at = p.at(0.15, t.predicted(10f64.powf(0.15)).log10());
                labels.push(label(Vec2::new(at.x.max(min.x + 4.0), at.y.max(min.y + 4.0)), Align2::LEFT_TOP, name, NOISE));
                // the measurement: each batch's mean over the seeds, and their spread
                let pts: Vec<(f64, f64)> = rows.iter().filter(|r| r.batch > 0).map(|r| (r.b().log10(), r.mean().log10())).collect();
                p.path(f, K_RUN, pts.iter().copied(), 2.2, SGD);
                let full = t.full();
                let tail = rows[rows.len() - 2];
                seg(f, &p, (tail.b().log10(), tail.mean().log10()), (inf_at, full.mean().log10()), 1.0, SGD, true);
                for r in rows {
                    let x = x_of(r.b(), inf_at);
                    plot::line_of(f, plot::CURVES, p.at(x, r.least().log10()), p.at(x, r.most().log10()), 1.4, SGD.a(0.6));
                    let q = p.at(x, r.mean().log10());
                    if plot::within(q, p.rect, 0.5) {
                        dot(f, 2, q, 5.0, SGD);
                    }
                }
                if let Some(fit) = self.net_fit() {
                    let end = top.log10() + 0.05;
                    path_even(f, &p, K_RUN_FIT, (0..64).map(|k| spread(k, 64, 0.0, end)).map(|x| (x, fit.steps(10f64.powf(x)).log10())), 64, 1.0, SGD.a(0.55));
                    let q = p.at(fit.b_crit.log10(), (2.0 * fit.s_min).log10());
                    if plot::within(q, p.rect, 0.5) {
                        ring(f, 2, q, 9.0, SGD);
                        labels.insert(0, label(q + Vec2::new(7.0, -3.0), Align2::LEFT_BOTTOM, format!(r"$B_{{\text{{crit}}}} \approx {}$ (fitted)", sig_bare(fit.b_crit)), SGD));
                    }
                }
                if let Some(x) = self.look.get(C, 0) {
                    p.vline(f, MARKS, x, 1.0, pal::TEXT.a(0.3), false);
                }
            }
        }
        if let Some(a) = axis {
            labels.insert(0, a);
        }
        self.labels_moving |= place(f, labels, size, p.rect, &mut self.placed[C]);
    }

    /// Chart (d): the steps of a whole run against its examples.
    fn draw_d(&mut self, f: &mut Frame, lay: &Layout) {
        let p = self.d_plot(lay);
        if p.tiny() {
            return;
        }
        let g = self.goal_plot(D, lay);
        let s = self.setup;
        let size = lay.small;
        let (min, max) = p.rect;
        p.backing(f, if lay.narrow { 46.0 } else { 54.0 }, 32.0, 8.0, 8.0);
        let title = match (s.whole, lay.narrow) {
            (Whole::Model, false) => "Steps against examples, each over its fewest: the trade-off",
            (Whole::Model, true) => "steps against examples",
            (_, false) => r"Steps against examples ($B$ a step)",
            (_, true) => "steps against examples",
        };
        f.text_px(lay.heads[D], title, lay.text, pal::TEXT, Align2::LEFT_BOTTOM);
        let mut labels: Vec<ChartLabel> = Vec::new();
        let axis: Option<ChartLabel>;
        match s.whole {
            Whole::Model => {
                axis_x(f, &p, &g, size, true, r"examples $E/E_{\min}$", None);
                axis = Some(axis_y(f, &p, &g, size, true, r"steps $S/S_{\min}$"));
                // (S/S_min − 1)(E/E_min − 1) = 1, through B/B_crit from 10⁻⁴ to 10⁴
                let pts: Vec<(f64, f64)> = (0..=160).map(|i| -4.0 + 8.0 * i as f64 / 160.0).map(|u| ((10f64.powf(u)).ln_1p() / std::f64::consts::LN_10, (10f64.powf(-u)).ln_1p() / std::f64::consts::LN_10)).collect();
                path_even(f, &p, K_TRADE, pts, 161, 2.4, SGD);
                seg(f, &p, (0.0, p.y[0]), (0.0, p.y[1]), 1.2, pal::DIM.a(0.9), false);
                seg(f, &p, (p.x[0], 0.0), (p.x[1], 0.0), 1.2, SIGNAL.a(0.7), false);
                let (a, b) = if lay.narrow {
                    (r"fewest examples, $E_{\min}$", r"fewest steps, $S_{\min}$")
                } else {
                    (r"the fewest examples, $E_{\min}$: $B$ far below $B_{\text{crit}}$", r"the fewest steps, $S_{\min}$: $B$ far above $B_{\text{crit}}$")
                };
                labels.push(label(Vec2::new(p.px_x(0.0) + 5.0, min.y + 20.0), Align2::LEFT_TOP, a, pal::DIM));
                labels.push(label(Vec2::new(max.x - 4.0, p.px_y(0.0) - 3.0), Align2::RIGHT_BOTTOM, b, SIGNAL));
                let q = p.at(2f64.log10(), 2f64.log10());
                ring(f, 3, q, 9.0, NOISE);
                let name = if lay.narrow { r"$B = B_{\text{crit}}$" } else { r"$B = B_{\text{crit}}$: twice each" };
                labels.insert(0, label(q + Vec2::new(7.0, -3.0), Align2::LEFT_BOTTOM, name, NOISE));
                for (k, (c, sz)) in [(SGD, 9.0), (SGD.a(0.6), 7.0)].into_iter().enumerate() {
                    if let (Some(x), Some(y)) = (self.look.get(D, 2 * k), self.look.get(D, 2 * k + 1)) {
                        let q = p.at(x, y);
                        let b = if k == 0 { Some(s.batch) } else { s.second };
                        if let (true, Some(b)) = (plot::within(q, p.rect, 0.5), b) {
                            dot(f, 3, q, sz, c);
                            labels.push(label(q + Vec2::new(7.0, 3.0), Align2::LEFT_TOP, format!("$B = {}$", batch_name(b)), c).keyed(["batch", "second"][k]));
                        }
                    }
                }
                if noise_scale(&s.one()).is_none() {
                    labels.insert(0, label(Vec2::new(max.x - 8.0, min.y + 20.0), Align2::RIGHT_TOP, r"no noise scale here", NOISE));
                }
            }
            Whole::Nqm | Whole::Network => {
                axis_x(f, &p, &g, size, true, "examples $E$", None);
                axis = Some(axis_y(f, &p, &g, size, true, "steps $S$"));
                let t = s.target();
                let (top, _) = table_reach(s.whole, t);
                let series: Vec<Series> = match s.whole {
                    Whole::Nqm => Rule::ALL
                        .into_iter()
                        .filter(|&r| r != s.rule)
                        .chain([s.rule])
                        .map(|r| {
                            let on = r == s.rule;
                            let pts = r.table().iter().filter(|row| row.batch.is_finite()).map(|row| (row.batch, row.steps as f64)).collect();
                            (rule_index(r), if on { r.color() } else { r.color().a(0.45) }, on, pts, self.nqm_fit(r))
                        })
                        .collect(),
                    _ => vec![(0, SGD, true, t.rows.iter().filter(|r| r.batch > 0).map(|r| (r.b(), r.mean())).collect(), self.net_fit())],
                };
                for (k, color, on, pts, fit) in series {
                    let xy: Vec<(f64, f64)> = pts.iter().map(|&(b, st)| ((b * st).log10(), st.log10())).collect();
                    p.path(f, K_TRADE + k, xy.iter().copied(), if on { 2.2 } else { 1.4 }, color);
                    for (i, &(x, y)) in xy.iter().enumerate() {
                        let q = p.at(x, y);
                        if plot::within(q, p.rect, 0.5) {
                            // (the point of the batch chosen, larger)
                            let chosen = on && (pts[i].0.log2() - s.batch.log2()).abs() < 0.5;
                            dot(f, 3, q, if chosen { 9.0 } else if on { 4.5 } else { 3.0 }, color);
                        }
                    }
                    if let Some(fit) = fit {
                        let curve: Vec<(f64, f64)> =
                            (0..=96).map(|i| 10f64.powf(-0.3 + (top.log10() + 0.6) * i as f64 / 96.0)).map(|b| ((b * fit.steps(b)).log10(), fit.steps(b).log10())).collect();
                        path_even(f, &p, K_TRADE_FIT + k, curve, 97, 1.0, color.a(0.6));
                        if on {
                            let (e, sm) = (fit.e_min().log10(), fit.s_min.log10());
                            seg(f, &p, (e, p.y[0]), (e, p.y[1]), 1.0, color.a(0.7), true);
                            seg(f, &p, (p.x[0], sm), (p.x[1], sm), 1.0, color.a(0.7), true);
                            let (a, b) = if lay.narrow {
                                (format!(r"$E_{{\min}} \approx {}$", sig_bare(fit.e_min())), format!(r"$S_{{\min}} \approx {}$", sig_bare(fit.s_min)))
                            } else {
                                (format!(r"fewest examples $E_{{\min}} \approx {}$", sig_bare(fit.e_min())), format!(r"fewest steps $S_{{\min}} \approx {}$", sig_bare(fit.s_min)))
                            };
                            labels.push(label(Vec2::new(p.px_x(e) + 5.0, min.y + 20.0), Align2::LEFT_TOP, a, color));
                            labels.push(label(Vec2::new(max.x - 4.0, p.px_y(sm) - 3.0), Align2::RIGHT_BOTTOM, b, color));
                            let q = p.at((2.0 * fit.e_min()).log10(), (2.0 * fit.s_min).log10());
                            if plot::within(q, p.rect, 0.5) {
                                ring(f, 3, q, 9.0, color);
                                labels.insert(0, label(q + Vec2::new(7.0, -3.0), Align2::LEFT_BOTTOM, r"$B = B_{\text{crit}}$: twice each", color));
                            }
                        }
                    }
                }
            }
        }
        if let Some(a) = axis {
            labels.insert(0, a);
        }
        self.labels_moving |= place(f, labels, size, p.rect, &mut self.placed[D]);
    }
}

/// A setting as a label writes it: as few digits as it needs (0.3, 0.6).
fn short(v: f64) -> String {
    let text = sig(v);
    if text.contains('.') && !text.contains('$') {
        text.trim_end_matches('0').trim_end_matches('.').to_string()
    } else {
        text
    }
}

impl Scene for NoiseScale {
    fn view(&self) -> View {
        View::Flat { center: Vec2::ZERO, half_height: 1.0 }
    }

    fn settings(&self, s: &mut ViewSettings) {
        s.grid = false;
        s.bloom = s.bloom.min(0.2);
    }

    fn animated(&self) -> bool {
        self.look.moving || self.labels_moving || self.dragging
    }

    fn pointer_hint(&self) -> Option<&'static str> {
        Some("cyan dot on the gain chart: drag the batch size    buttons: click")
    }

    fn ui(&mut self, ui: &mut egui::Ui, _: &Gfx) {
        ui::note(
            ui,
            r"How large should a batch be? A step of SGD costs $B$ examples. A larger batch gives a gradient with less noise, and so allows a longer step, but only until the noise is no longer what limits the step: from there the curvature does, and more examples buy almost nothing. McCandlish, Kaplan and Amodei (2018) made this exact in a model of one step, with the loss replaced by its quadratic approximation. A step $\eta$ against the mean gradient of $B$ independent examples changes the loss, on average, by",
        );
        ui::equation(ui, r"\mathbb{E}\,\Delta L = -\eta\,|G|^2 + \tfrac12\eta^2\Big(\textcolor{green}{G^\top H G} + \frac{\textcolor{red}{\operatorname{tr}(H\Sigma)}}{B}\Big)");
        ui::note(
            ui,
            r"$G$ is the true gradient (of the loss on all the data), $H$ the Hessian, $\Sigma$ the covariance of one example's gradient. The best step and its gain shrink by one factor:",
        );
        ui::equation(ui, r"\eta_{\text{opt}} = \frac{\eta_{\max}}{1 + B_{\text{noise}}/B},\quad \Delta L_{\text{opt}} = \frac{\Delta L_{\max}}{1 + B_{\text{noise}}/B},\quad B_{\text{noise}} = \frac{\operatorname{tr}(H\Sigma)}{G^\top H G}");
        ui::note(
            ui,
            r"with $\eta_{\max} = |G|^2/G^\top H G$ and $\Delta L_{\max} = \tfrac12|G|^4/G^\top H G$ the true gradient's. Far below the noise scale $B_{\text{noise}}$ the gain grows in proportion to $B$: twice the batch, half the steps, the same examples. Far above it the gain hardly grows, and the extra examples are wasted. The model assumes one greedy step, the quadratic approximation, plain SGD (no momentum, no preconditioner), the best step at every batch size, and $G$, $H$, $\Sigma$ taken at one point; it is about the training loss only. Its authors call their assumptions unfounded, and find the result a helpful guide all the same.",
        );

        ui::section(ui, "Things to try");
        for (k, t) in TRIALS.iter().enumerate() {
            if ui::experiment(ui, t.button, t.caption) {
                self.set_up(k);
            }
        }

        ui::section(ui, "One step: its three numbers");
        let s = &mut self.setup;
        let mut source = (s.source == Source::Hand) as usize;
        if ui::choice(ui, &["from the network", "set by hand"], &mut source) {
            if source == 1 && s.source == Source::Network {
                s.hand = s.moment().one_step();
            }
            s.source = if source == 1 { Source::Hand } else { Source::Network };
        }
        match s.source {
            Source::Network => {
                let names: Vec<String> = TARGETS.iter().map(|t| format!("trained to a loss of {}", short(t.loss))).collect();
                let names: Vec<&str> = names.iter().map(|n| n.as_str()).collect();
                let mut target = s.target;
                if TARGETS.len() > 1 && ui::choice(ui, &names, &mut target) {
                    let step = s.moment().step;
                    s.target = target.min(TARGETS.len() - 1);
                    // (the moment nearest the same step of the new run)
                    let ms = s.target().moments;
                    s.moment = (0..ms.len()).min_by_key(|&i| (ms[i].step as i64 - step as i64).abs()).unwrap_or(0);
                    s.earlier = None;
                }
                let last = s.target().moments.len() - 1;
                let mut at = s.moment;
                ui::slider_int(ui, "moment of its full-batch run", &mut at, 0..=last);
                s.moment = at.min(last);
                ui::note(
                    ui,
                    &format!(
                        r"Step {} of {} of gradient descent with the whole data set, $\eta = {}$ (the run marked $\infty$ on the chart of the network's steps). Measured exactly: every example's gradient, and the Hessian through its products with vectors.",
                        count(s.moment().step as usize),
                        count(s.target().full().steps[0] as usize),
                        short(s.target().full().lr())
                    ),
                );
            }
            Source::Hand => {
                let mut g2 = s.hand.g2 as f32;
                if ui::slider_log(ui, r"$|G|^2$", &mut g2, HAND_G2[0]..=HAND_G2[1]) {
                    s.hand.g2 = g2 as f64;
                }
                let mut ghg = s.hand.ghg as f32;
                if ui::slider_log_or_zero(ui, r"$G^\top H G$", &mut ghg, HAND_GHG[0], HAND_GHG[1]) {
                    s.hand.ghg = ghg as f64;
                }
                let mut trhs = s.hand.trhs as f32;
                if ui::slider_log_or_zero(ui, r"$\operatorname{tr}(H\Sigma)$", &mut trhs, HAND_TRHS[0], HAND_TRHS[1]) {
                    s.hand.trhs = trhs as f64;
                }
                ui::note(ui, r"The curvature along the gradient times $|G|^2$, and the noise weighed by the curvature. At 0 there is no lowest point, or no noise.");
            }
        }
        let mut b = s.batch.round() as usize;
        if ui::slider_int_log(ui, "batch size $B$", &mut b, 1..=MOST_BATCH as usize) {
            s.batch = b as f64;
            s.second = None;
        }
        ui::note(ui, "Or drag the cyan dot on the chart of the gain. The examples are drawn independently (with replacement).");

        ui::section(ui, "A whole run");
        let mut whole = Whole::ALL.iter().position(|&w| w == s.whole).unwrap_or(0);
        if ui::choice(ui, &["if the noise scale stayed", "noisy quadratic model", "this network"], &mut whole) {
            s.whole = Whole::ALL[whole.min(2)];
        }
        match s.whole {
            Whole::Model => ui::note(
                ui,
                r"McCandlish and co-authors add up such steps over a run: each step's worth of the true gradient's progress takes $1 + B_{\text{noise}}/B$ steps of batch $B$. With $B_{\text{noise}}$ the same all along, a run takes $S = S_{\min}(1 + B_{\text{noise}}/B)$ steps and $E = BS = E_{\min}(1 + B/B_{\text{noise}})$ examples.",
            ),
            Whole::Nqm => {
                let mut rule = Rule::ALL.iter().position(|&r| r == s.rule).unwrap_or(0);
                if ui::choice(ui, &["SGD", "momentum", r"preconditioned"], &mut rule) {
                    s.rule = Rule::ALL[rule.min(2)];
                }
                ui::note(
                    ui,
                    r"Zhang and co-authors' (2019) model: $10^4$ independent directions of curvature $h_i = 1/i$, gradient noise of variance $h_i/B$ in each, the loss $\tfrac12\sum h_i\theta_i^2$ from $\mathbb{E}\,\theta_i^2 = 1$ down to 0.01. Exact expected losses; at each batch the best learning rate (and for momentum the best $\beta$ of ten, up to 0.999). The preconditioner divides by $H^{1/2}$: the model's stand-in for a preconditioned rule, not Adam.",
                );
            }
            Whole::Network => ui::note(
                ui,
                &format!(
                    r"A tanh network 2–{w}–{w}–3 ({p} weights) on {n} points of three spirals, cross-entropy, from one start. At each batch, plain SGD at the step sizes of a grid, $\eta = 2^{{k/4}}/256$; the one that reaches the target in the fewest steps, on average over {SEEDS} seeds of the batches, is kept. A run's steps: until the loss on all the data, checked after every step, is first at or below the target.",
                    w = NET_WIDTH,
                    p = net_params(),
                    n = NET_POINTS
                ),
            ),
        }

        ui::section(ui, "Reading the picture");
        ui::key(
            ui,
            &[
                (SGD, "cyan: plain SGD; among the parabolas, the batch chosen (faint: a second); dots: runs measured or computed"),
                (pal::DIM, "grey parabolas: batches four times apart, for comparison"),
                (SIGNAL, r"white: the true gradient (no noise): its parabola, its best gain, the fewest steps"),
                (NOISE, r"red: the noise scale $B_{\text{noise}}$ (faint: $B_{\text{simple}}$); the model's prediction"),
                (LIMIT, r"yellow tick: twice the best step, beyond which the expected loss rises"),
                (MOMENTUM, "pink: heavy-ball momentum, in the noisy quadratic model"),
                (PRECONDITIONED, r"lilac: SGD preconditioned by $H^{1/2}$, in the noisy quadratic model"),
                (pal::DIM, r"dashed grey across the steps: perfect scaling, steps falling as $1/B$"),
            ],
        );
        ui::section(ui, "Words");
        ui::term(ui, "SGD:", "stochastic gradient descent: a step against the gradient of a mini-batch instead of the whole data's.");
        ui::term(ui, r"Batch size $B$:", "the examples whose gradients one step averages; a step costs that many examples.");
        ui::term(ui, r"True gradient $G$:", "the gradient of the loss on all the data: what a mini-batch's gradient estimates.");
        ui::term(ui, r"Hessian $H$:", r"the matrix of second derivatives of the loss; $G^\top H G/|G|^2$ is the curvature along the gradient.");
        ui::term(ui, r"Covariance $\Sigma$:", r"of one example's gradient over the data (divisor $N$). A batch of $B$ independent examples has noise of covariance $\Sigma/B$.");
        ui::term(ui, r"Noise scale $B_{\text{noise}}$:", r"$\operatorname{tr}(H\Sigma)/G^\top HG$ (McCandlish and co-authors' Eq. 2.8): the batch at which the best step gains half what the true gradient's would.");
        ui::term(
            ui,
            r"Simple noise scale $B_{\text{simple}}$:",
            r"$\operatorname{tr}\Sigma/|G|^2$, the batch at which a mini-batch gradient's expected squared length is half noise (scene minibatch). It equals $B_{\text{noise}}$ when $H$ is a multiple of the identity, which the authors call an unrealistic assumption; they report that in practice the two differ by a small factor, particularly with training that improves the conditioning.",
        );
        ui::term(ui, r"Steps $S$, examples $E$:", r"to reach a target loss; at a fixed batch $E = BS$. $S_{\min}$ and $E_{\min}$: the fewest possible.");
        ui::term(ui, "Perfect scaling:", "Shallue and co-authors' (2019) name for the regime where twice the batch takes half the steps.");
        ui::term(ui, "Expected:", "averaged over all the ways the batch could come out.");
        ui::term(ui, "Noisy quadratic model:", r"a loss that is a sum of independent bowls, $\tfrac12\sum_i h_i\theta_i^2$, whose gradient comes with noise that does not depend on where the weights are: simple enough that every expected loss is exact.");
        ui::term(ui, "Momentum:", r"heavy ball: $v \leftarrow \beta v + g$, then $\theta \leftarrow \theta - \eta v$; each gradient goes on acting for about $1/(1 - \beta)$ steps.");
        ui::term(ui, "Preconditioner:", r"a fixed matrix $P$ whose inverse multiplies the gradient before the step, $\theta \leftarrow \theta - \eta P^{-1} g$; here $P = H^{1/2}$.");

        ui::section(ui, "Critical batch size: whose?");
        ui::note(ui, "The phrase means different things in different papers. A number from one cannot be compared with a number from another without converting.");
        ui::note(ui, r"McCandlish and co-authors (2018), and Kaplan and co-authors (2020) after them: $B_{\text{crit}} = E_{\min}/S_{\min}$ for a chosen target loss, fitted to runs that follow $(S/S_{\min} - 1)(E/E_{\min} - 1) = 1$; at that batch a run takes twice the fewest steps and twice the fewest examples. This scene's fits use it.");
        ui::note(ui, r"Shallue and co-authors (2019) give no single number: the end of perfect scaling, and a maximum useful batch size beyond which the steps stop falling.");
        ui::note(ui, "Zhang and co-authors (2019), for the noisy quadratic model: the phase transition between the regime of perfect scaling and the one insensitive to the batch size.");
        ui::note(ui, r"Ma, Bassily and Belkin (2018), for least squares that fit every point exactly: about $\max_i |x_i|^2/\lambda_1$, with $\lambda_1$ the largest eigenvalue of $\tfrac1n\sum_i x_i x_i^\top$; set by the data, not by the noise at a point.");
        ui::note(ui, "Zhang and co-authors (2025): the largest batch, above the best one, that needs at most 20 % more steps than perfect scaling to reach the best loss of one pass over the data.");
        ui::note(ui, "Merrill and co-authors (2025): the largest multiple of the batch whose runs, branched from a checkpoint with the learning rate scaled up too, reach after 2 billion tokens a loss no more than 0.01 above each smaller batch's.");
        ui::note(ui, "Sato and co-authors (2025), for Muon: the batch that minimises the total number of example gradients in a convergence bound.");
        ui::note(ui, "Essential AI (2025): the largest batch that reaches a given loss with the fewest tokens.");
        ui::note(ui, "Naganuma and co-authors (2026): the batch at which a lower bound on one step's progress reaches a chosen fraction of the full batch's.");

        ui::section(ui, "Numbers");
        self.numbers(ui);
        ui::see_also(
            ui,
            &[
                ("minibatch", "a mini-batch gradient: signal and noise"),
                ("noisefloor", "what a constant step cannot get under"),
                ("stepsize", "the step size on a bowl, without noise"),
                ("signnoise", "sign descent and Adam in noise"),
            ],
        );
    }

    fn frame(&mut self, f: &mut Frame) {
        hold_batches(f);
        let lay = Layout::of(f.view_size(), f.top_inset, self.setup.page, self.setup.whole);
        // (a canvas squeezed to a sliver is one the controls have opened over, on a phone)
        let tiny = lay.view.x < 160.0 || lay.view.y < 120.0;
        self.dragging = false;
        if !tiny {
            self.buttons(f, &lay);
            self.drag(f, &lay);
        }
        self.setup = self.setup.kept();
        // (the buttons may have changed what is shown)
        let lay = Layout::of(f.view_size(), f.top_inset, self.setup.page, self.setup.whole);
        let snap = self.canvas != Some(lay.view) && !tiny;
        if snap {
            self.canvas = Some(lay.view);
        }
        let goal = self.setup.goal();
        self.look.follow(&goal, f.real_dt, snap);
        self.goal_frames = goal.frames;
        if tiny {
            return;
        }
        self.labels_moving = false;
        if lay.shown[A] {
            self.draw_a(f, &lay);
        }
        if lay.shown[B] {
            self.draw_b(f, &lay);
        }
        if lay.shown[C] {
            self.draw_c(f, &lay);
        }
        if lay.shown[D] {
            self.draw_d(f, &lay);
        }
    }
}

impl NoiseScale {
    /// The read-outs.
    fn numbers(&self, ui: &mut egui::Ui) {
        let s = &self.setup;
        let one = s.one();
        if s.source == Source::Network {
            let m = s.moment();
            ui::readout(ui, &format!("the network at step {}: its loss", count(m.step as usize)), sig(m.loss));
        }
        ui::readout(ui, r"$|G|^2$", sig(one.g2));
        ui::readout_colored(ui, r"$G^\top H G$", sig(one.ghg), CURVATURE);
        ui::readout_colored(ui, r"$\operatorname{tr}(H\Sigma)$", sig(one.trhs), NOISE);
        match noise_scale(&one) {
            Some(bn) => ui::readout_colored(ui, r"$B_{\text{noise}}$", sig(bn), NOISE),
            None => ui::readout_colored(ui, r"$B_{\text{noise}}$", format!("{}: not a batch size", sig(one.noise_scale())), NOISE),
        }
        if let Some(bs) = s.simple() {
            ui::readout_colored(ui, r"$B_{\text{simple}} = \operatorname{tr}\Sigma/|G|^2$", sig(bs), NOISE);
            if let Some(bn) = noise_scale(&one) {
                ui::readout(ui, r"  $B_{\text{noise}}/B_{\text{simple}}$", ratio(bn / bs));
            }
        }
        let (em, gm) = (one.max_step(), one.max_gain());
        if em.is_finite() && em > 0.0 {
            ui::readout(ui, r"$\eta_{\max}$, $\Delta L_{\max}$", format!("{}, {}", sig(em), sig(gm)));
            if let Some(lr) = s.run_lr() {
                ui::readout(ui, r"  the run's own step over $\eta_{\max}$", ratio(lr / em));
            }
        } else {
            ui::readout(ui, r"$\eta_{\max}$", r"none: $G^\top H G \le 0$");
        }
        let b = s.batch;
        let (e, g) = (one.best_step(b), best_gain(&one, b));
        match g {
            Some(g) if e.is_finite() => {
                ui::readout_colored(ui, &format!(r"at $B = {}$: $\eta_{{\text{{opt}}}}$, $\Delta L_{{\text{{opt}}}}$", batch_name(b)), format!("{}, {}", sig(e), sig(g)), SGD);
                if gm.is_finite() && gm > 0.0 {
                    ui::readout(ui, "  of the true gradient's gain", format!("{} %", num(100.0 * g / gm, 1)));
                }
            }
            _ => ui::readout_colored(ui, &format!(r"at $B = {}$", batch_name(b)), "no best step", SGD),
        }
        if let Some(bn) = noise_scale(&one) {
            ui::readout(ui, r"  if $B_{\text{noise}}$ stayed: $S/S_{\min}$, $E/E_{\min}$", format!("{}, {}", ratio(model_steps(bn, b)), ratio(model_examples(bn, b))));
        }
        match s.whole {
            Whole::Model => {}
            Whole::Nqm => {
                let r = s.rule;
                if let Some(fit) = self.nqm_fit(r) {
                    ui::readout_colored(ui, &format!(r"{}: $B_{{\text{{crit}}}}$ (fitted)", r.name()), sig(fit.b_crit), r.color());
                    ui::readout(ui, r"  $S_{\min}$, $E_{\min}$ (fitted)", format!("{}, {}", sig(fit.s_min), sig(fit.e_min())));
                }
                let rows = r.table();
                if let Some(row) = rows.iter().find(|row| (row.batch.log2() - b.log2()).abs() < 0.5) {
                    let mut text = format!(r"{} steps, $\eta = {}$", count(row.steps as usize), sig_bare(row.lr));
                    if r == Rule::Momentum {
                        text += &format!(r", $\beta = {}$", row.momentum);
                    }
                    ui::readout(ui, &format!(r"  at $B = {}$", batch_name(row.batch)), text);
                }
                ui::readout(ui, "  with the true gradient", format!("{} steps", count(rows[NQM_ROWS - 1].steps as usize)));
            }
            Whole::Network => {
                let t = s.target();
                if let Some(fit) = self.net_fit() {
                    ui::readout_colored(ui, r"$B_{\text{crit}}$ (fitted)", sig(fit.b_crit), SGD);
                    ui::readout(ui, r"  $S_{\min}$, $E_{\min}$ (fitted)", format!("{}, {}", sig(fit.s_min), sig(fit.e_min())));
                }
                let sm = &t.summary;
                ui::readout_colored(ui, r"$B_{\text{noise}}$ over the full-batch run: mean", sig(sm.mean_noise), NOISE);
                ui::readout(ui, "  its quartiles", format!("{}, {}, {}", sig(sm.noise_quartiles[0]), sig(sm.noise_quartiles[1]), sig(sm.noise_quartiles[2])));
                ui::readout(ui, r"$B_{\text{simple}}$ over it: mean", sig(sm.mean_simple));
                if let Some(row) = t.rows.iter().find(|r| r.batch > 0 && (r.b().log2() - b.log2()).abs() < 0.5) {
                    ui::readout(
                        ui,
                        &format!(r"  at $B = {}$: steps (seeds), $\eta$", batch_name(row.b())),
                        format!("{} ({}–{}), {}", sig(row.mean()), count(row.least() as usize), count(row.most() as usize), short(row.lr())),
                    );
                }
                ui::readout(ui, "  with the true gradient", format!(r"{} steps, $\eta = {}$", count(t.full().steps[0] as usize), short(t.full().lr())));
            }
        }
    }
}

// ----- the tables ------------------------------------------------------------------------------

// The noisy quadratic model: the standard model of `terrain::Nqm::standard` (10⁴ directions,
// target 0.01), each rule's best learning rate (and, for momentum, the best of
// `terrain::MOMENTA`) at each batch size, found by `terrain::Nqm::best_lr_momentum`.
// Regenerated and compared by the ignored test `regenerate_the_quadratic_tables`.

const NQM_SGD: [NqmRow; NQM_ROWS] = [
    NqmRow { batch: 1.0, steps: 6024935, lr: 0.003398230766683988, momentum: 0.0 },
    NqmRow { batch: 2.0, steps: 3013330, lr: 0.006794891255633155, momentum: 0.0 },
    NqmRow { batch: 4.0, steps: 1507530, lr: 0.0135795569883999, momentum: 0.0 },
    NqmRow { batch: 8.0, steps: 754635, lr: 0.027124524981521955, momentum: 0.0 },
    NqmRow { batch: 16.0, steps: 378197, lr: 0.054093964391166434, momentum: 0.0 },
    NqmRow { batch: 32.0, steps: 189998, lr: 0.10757366655074947, momentum: 0.0 },
    NqmRow { batch: 64.0, steps: 95942, lr: 0.21258082135705808, momentum: 0.0 },
    NqmRow { batch: 128.0, steps: 49009, lr: 0.4139910318161185, momentum: 0.0 },
    NqmRow { batch: 256.0, steps: 25778, lr: 0.7759432856542603, momentum: 0.0 },
    NqmRow { batch: 512.0, steps: 14796, lr: 1.284459152442473, momentum: 0.0 },
    NqmRow { batch: 1024.0, steps: 10354, lr: 1.6411620994207063, momentum: 0.0 },
    NqmRow { batch: 2048.0, steps: 8632, lr: 1.7923882578047217, momentum: 0.0 },
    NqmRow { batch: 4096.0, steps: 7832, lr: 1.8664638623110426, momentum: 0.0 },
    NqmRow { batch: 8192.0, steps: 7401, lr: 1.9101592821269135, momentum: 0.0 },
    NqmRow { batch: 16384.0, steps: 7147, lr: 1.9386855582792042, momentum: 0.0 },
    NqmRow { batch: 32768.0, steps: 6989, lr: 1.9569182549123252, momentum: 0.0 },
    NqmRow { batch: 65536.0, steps: 6887, lr: 1.9699343477734437, momentum: 0.0 },
    NqmRow { batch: 131072.0, steps: 6819, lr: 1.9786598043369201, momentum: 0.0 },
    NqmRow { batch: 262144.0, steps: 6773, lr: 1.9851070171985317, momentum: 0.0 },
    NqmRow { batch: 524288.0, steps: 6742, lr: 1.989498490304295, momentum: 0.0 },
    NqmRow { batch: 1048576.0, steps: 6720, lr: 1.992859821601115, momentum: 0.0 },
    NqmRow { batch: f64::INFINITY, steps: 6673, lr: 1.9989549629404553, momentum: 0.0 },
];

const NQM_MOMENTUM: [NqmRow; NQM_ROWS] = [
    NqmRow { batch: 1.0, steps: 6022909, lr: 1.699910174427947e-5, momentum: 0.995 },
    NqmRow { batch: 2.0, steps: 3010059, lr: 6.80015427309119e-6, momentum: 0.999 },
    NqmRow { batch: 4.0, steps: 1504255, lr: 1.3601550725320874e-5, momentum: 0.999 },
    NqmRow { batch: 8.0, steps: 751351, lr: 2.7205586035826777e-5, momentum: 0.999 },
    NqmRow { batch: 16.0, steps: 374897, lr: 5.444453555841314e-5, momentum: 0.999 },
    NqmRow { batch: 32.0, steps: 186664, lr: 0.00010900183110644454, momentum: 0.999 },
    NqmRow { batch: 64.0, steps: 92536, lr: 0.00021850588737239377, momentum: 0.999 },
    NqmRow { batch: 128.0, steps: 45448, lr: 0.00043921658776525593, momentum: 0.999 },
    NqmRow { batch: 256.0, steps: 21850, lr: 0.000888273775073874, momentum: 0.999 },
    NqmRow { batch: 512.0, steps: 10095, lr: 0.0018317181680661686, momentum: 0.999 },
    NqmRow { batch: 1024.0, steps: 5557, lr: 0.017662565260073072, momentum: 0.995 },
    NqmRow { batch: 2048.0, steps: 2595, lr: 0.03615902441556677, momentum: 0.995 },
    NqmRow { batch: 4096.0, steps: 1305, lr: 0.1431508767066063, momentum: 0.99 },
    NqmRow { batch: 8192.0, steps: 668, lr: 0.5583180773389497, momentum: 0.98 },
    NqmRow { batch: 16384.0, steps: 372, lr: 0.9086037536008844, momentum: 0.98 },
    NqmRow { batch: 32768.0, steps: 261, lr: 3.118529463115172, momentum: 0.95 },
    NqmRow { batch: 65536.0, steps: 209, lr: 3.4332280446951096, momentum: 0.95 },
    NqmRow { batch: 131072.0, steps: 187, lr: 3.6576330058494753, momentum: 0.95 },
    NqmRow { batch: 262144.0, steps: 179, lr: 3.599612491624644, momentum: 0.95 },
    NqmRow { batch: 524288.0, steps: 171, lr: 3.7432611355712524, momentum: 0.95 },
    NqmRow { batch: 1048576.0, steps: 170, lr: 3.712256015650217, momentum: 0.95 },
    NqmRow { batch: f64::INFINITY, steps: 165, lr: 3.8498698987162303, momentum: 0.95 },
];

const NQM_PRECONDITIONED: [NqmRow; NQM_ROWS] = [
    NqmRow { batch: 1.0, steps: 1391819, lr: 0.00017113234251791968, momentum: 0.0 },
    NqmRow { batch: 2.0, steps: 695912, lr: 0.00034215151771629807, momentum: 0.0 },
    NqmRow { batch: 4.0, steps: 347958, lr: 0.0006843655359966209, momentum: 0.0 },
    NqmRow { batch: 8.0, steps: 173982, lr: 0.0013688560845382218, momentum: 0.0 },
    NqmRow { batch: 16.0, steps: 86993, lr: 0.0027370796386865697, momentum: 0.0 },
    NqmRow { batch: 32.0, steps: 43499, lr: 0.005474113859565388, momentum: 0.0 },
    NqmRow { batch: 64.0, steps: 21752, lr: 0.010945698203126195, momentum: 0.0 },
    NqmRow { batch: 128.0, steps: 10878, lr: 0.021881978025793905, momentum: 0.0 },
    NqmRow { batch: 256.0, steps: 5442, lr: 0.043753844677391386, momentum: 0.0 },
    NqmRow { batch: 512.0, steps: 2723, lr: 0.08739624230770866, momentum: 0.0 },
    NqmRow { batch: 1024.0, steps: 1364, lr: 0.17434899421082492, momentum: 0.0 },
    NqmRow { batch: 2048.0, steps: 685, lr: 0.3468990886419854, momentum: 0.0 },
    NqmRow { batch: 4096.0, steps: 345, lr: 0.6858794961105109, momentum: 0.0 },
    NqmRow { batch: 8192.0, steps: 177, lr: 1.3280125316968108, momentum: 0.0 },
    NqmRow { batch: 16384.0, steps: 107, lr: 1.8949484260293088, momentum: 0.0 },
    NqmRow { batch: 32768.0, steps: 91, lr: 1.944615008189412, momentum: 0.0 },
    NqmRow { batch: 65536.0, steps: 86, lr: 1.948916904477895, momentum: 0.0 },
    NqmRow { batch: 131072.0, steps: 84, lr: 1.949933832120103, momentum: 0.0 },
    NqmRow { batch: 262144.0, steps: 83, lr: 1.949933832120103, momentum: 0.0 },
    NqmRow { batch: 524288.0, steps: 82, lr: 1.949933832120103, momentum: 0.0 },
    NqmRow { batch: 1048576.0, steps: 82, lr: 1.949933832120103, momentum: 0.0 },
    NqmRow { batch: f64::INFINITY, steps: 82, lr: 1.949933832120103, momentum: 0.0 },
];

// The network (see `measure`): every batch size at its best step size, the true gradient
// last; its full-batch run measured at every step, the moments kept; and the summary over
// every step. Regenerated and compared by the ignored test `regenerate_the_network_tables`.

const NET_0_6_ROWS: [Row; 13] = [
    Row { batch: 1, k: 14, steps: [11252, 11982, 12331, 12715, 13293] },
    Row { batch: 2, k: 18, steps: [5774, 5982, 6078, 6358, 6747] },
    Row { batch: 4, k: 22, steps: [2917, 3062, 3027, 3267, 3323] },
    Row { batch: 8, k: 26, steps: [1608, 1778, 1591, 1691, 2079] },
    Row { batch: 16, k: 29, steps: [946, 1044, 868, 985, 965] },
    Row { batch: 32, k: 32, steps: [639, 685, 595, 610, 645] },
    Row { batch: 64, k: 34, steps: [461, 454, 430, 426, 438] },
    Row { batch: 128, k: 34, steps: [379, 368, 371, 376, 367] },
    Row { batch: 256, k: 35, steps: [339, 311, 320, 320, 342] },
    Row { batch: 512, k: 35, steps: [317, 333, 311, 324, 303] },
    Row { batch: 1024, k: 35, steps: [302, 318, 316, 331, 330] },
    Row { batch: 2048, k: 35, steps: [324, 324, 310, 315, 334] },
    Row { batch: 0, k: 36, steps: [316, 316, 316, 316, 316] },
];

const NET_0_6_MOMENTS: [Moment; 42] = [
    Moment { step: 0, loss: 1.1146533586921057, g2: 0.01802931933547777, ghg: 0.004872302923251781, trhs: 2.430190257310111, trace: 3.202576585053132 },
    Moment { step: 8, loss: 1.0852032523474935, g2: 0.051750019112997606, ghg: 0.04414076129999971, trhs: 1.692449559393965, trace: 2.716509942893222 },
    Moment { step: 16, loss: 1.0491470046402938, g2: 0.00046107098366492187, ghg: 5.6382916361739354e-5, trhs: 1.9611291880239416, trace: 2.8489613022529183 },
    Moment { step: 24, loss: 1.0421293344467832, g2: 0.0005247309406804838, ghg: 5.654310678311301e-7, trhs: 2.4655339350875916, trace: 3.1709106482949787 },
    Moment { step: 32, loss: 1.0318381097278881, g2: 0.0009374427554315423, ghg: 0.0001240615148363747, trhs: 3.1642489102679856, trace: 3.5994267778173916 },
    Moment { step: 40, loss: 1.0339568079513235, g2: 0.04623923533401496, ghg: 0.057281967630973714, trhs: 3.756023817697574, trace: 4.066909428603296 },
    Moment { step: 48, loss: 1.06456144332599, g2: 0.09524987423017814, ghg: 0.08553212752063982, trhs: 2.7619657421599526, trace: 3.7636765748189736 },
    Moment { step: 56, loss: 0.9967327211875353, g2: 0.012395657237985764, ghg: 0.01236245849863767, trhs: 3.3509759724438646, trace: 4.0840623471063315 },
    Moment { step: 64, loss: 1.0543113124259607, g2: 0.12549915274823792, ghg: 0.10095020906741034, trhs: 3.694238835461165, trace: 4.694806416462461 },
    Moment { step: 72, loss: 1.017433145455313, g2: 0.08965321647150805, ghg: 0.08250516497755483, trhs: 3.8468264213094847, trace: 4.75011888718985 },
    Moment { step: 80, loss: 1.0065696447620276, g2: 0.077678993245252, ghg: 0.06741728592952734, trhs: 3.9457137589103324, trace: 4.842735769675063 },
    Moment { step: 88, loss: 1.024403901369855, g2: 0.13044448915365867, ghg: 0.11424427010798073, trhs: 4.405302010002327, trace: 5.275455304251684 },
    Moment { step: 96, loss: 1.0053092250561289, g2: 0.1073698340814784, ghg: 0.09443263638261822, trhs: 4.496255477911874, trace: 5.314830034209771 },
    Moment { step: 104, loss: 1.0027006681552406, g2: 0.12497569755053728, ghg: 0.11755727652558072, trhs: 4.783001610249426, trace: 5.567873571621248 },
    Moment { step: 112, loss: 0.9967385109988633, g2: 0.128040165658296, ghg: 0.12043830476990633, trhs: 4.873213746561923, trace: 5.633718308070867 },
    Moment { step: 120, loss: 0.9873732138834046, g2: 0.1277472167303701, ghg: 0.1207114701184908, trhs: 4.976004321533836, trace: 5.694333295739113 },
    Moment { step: 128, loss: 0.983109459310291, g2: 0.13673846307343684, ghg: 0.12718499294501065, trhs: 5.087702230576674, trace: 5.7944461258385145 },
    Moment { step: 136, loss: 0.971909087668079, g2: 0.1344060202687401, ghg: 0.12234176973885033, trhs: 5.17048726293876, trace: 5.894055573361665 },
    Moment { step: 144, loss: 0.9614520567043101, g2: 0.14491104212043293, ghg: 0.14003954815149877, trhs: 5.485547060014879, trace: 6.0952738837508464 },
    Moment { step: 152, loss: 0.9774368686097137, g2: 0.2554285750401729, ghg: 0.31526260721911264, trhs: 5.250402942456148, trace: 6.194137175907406 },
    Moment { step: 160, loss: 0.97242090720605, g2: 0.17778149626266082, ghg: 0.14483127428542034, trhs: 4.130787592682235, trace: 5.594043600140386 },
    Moment { step: 168, loss: 0.9388191184039473, g2: 0.15932440265014308, ghg: 0.14662792473225134, trhs: 4.344421948941654, trace: 5.655254030648933 },
    Moment { step: 176, loss: 0.9214518119420988, g2: 0.158386682782998, ghg: 0.14497657478119516, trhs: 4.408551525551229, trace: 5.69237431115557 },
    Moment { step: 184, loss: 0.8982204923208666, g2: 0.14803088701573164, ghg: 0.12765750847879917, trhs: 4.1907138923764, trace: 5.475021575991769 },
    Moment { step: 192, loss: 0.8906693382550883, g2: 0.15201586771782788, ghg: 0.09086480860306413, trhs: 4.163430314156488, trace: 5.584632878829676 },
    Moment { step: 200, loss: 1.0128661222669815, g2: 0.4010370711471225, ghg: 0.31148488123818097, trhs: 3.8541532851725413, trace: 6.1676678590199305 },
    Moment { step: 208, loss: 0.9006622093722405, g2: 0.22258143606302092, ghg: 0.17204727280847612, trhs: 3.825057321943007, trace: 5.902734211880424 },
    Moment { step: 216, loss: 0.8659063197017197, g2: 0.2070210046062638, ghg: 0.2008088432443375, trhs: 4.021523234634981, trace: 5.825170394611463 },
    Moment { step: 224, loss: 0.8894965637568071, g2: 0.2993334259493687, ghg: 0.3083784674983123, trhs: 4.357795910086181, trace: 6.229373975302435 },
    Moment { step: 232, loss: 0.8553741924935988, g2: 0.25418723896758, ghg: 0.2694659498522705, trhs: 4.294253424832176, trace: 5.98153073157499 },
    Moment { step: 240, loss: 0.8141257563312679, g2: 0.22895899652026214, ghg: 0.2440953792220225, trhs: 4.547076618892605, trace: 6.340256607492026 },
    Moment { step: 248, loss: 0.7738230058608687, g2: 0.21278369222681026, ghg: 0.24372757440420928, trhs: 4.487189346582499, trace: 6.019571856788177 },
    Moment { step: 256, loss: 0.9938199907014666, g2: 0.6329000205730196, ghg: 0.6525558723754736, trhs: 5.915727623287968, trace: 8.887734254392221 },
    Moment { step: 264, loss: 0.781226868810725, g2: 0.2774486310005388, ghg: 0.22868169937944968, trhs: 3.874929180317193, trace: 6.648368816649995 },
    Moment { step: 272, loss: 0.792927452275624, g2: 0.3157429766265293, ghg: 0.2353117885771757, trhs: 4.694415147155882, trace: 7.3424984055651334 },
    Moment { step: 280, loss: 0.7125771907724913, g2: 0.21668298588474125, ghg: 0.1525243652625285, trhs: 4.019284962473308, trace: 6.608126072513343 },
    Moment { step: 288, loss: 0.7383498341519686, g2: 0.3228020712579054, ghg: 0.24214475207928474, trhs: 4.877531410967731, trace: 7.812493881415437 },
    Moment { step: 296, loss: 0.6775086351301294, g2: 0.23406892809013657, ghg: 0.1672326008999293, trhs: 4.47170216420683, trace: 7.106581793554931 },
    Moment { step: 304, loss: 0.6749173898750882, g2: 0.2896547791526503, ghg: 0.23740387117639594, trhs: 5.158085200740828, trace: 7.756041606032614 },
    Moment { step: 312, loss: 0.8239527227606137, g2: 0.6421579849842437, ghg: 0.3903212990301002, trhs: 7.435492491872695, trace: 13.310296765798636 },
    Moment { step: 314, loss: 0.9660935596499126, g2: 0.558199821801663, ghg: -0.09569784470548591, trhs: 1.6357170465735582, trace: 13.01832158730688 },
    Moment { step: 316, loss: 0.5919473232025484, g2: 0.15867831965862647, ghg: 0.1613016307201915, trhs: 4.458455263679176, trace: 5.846922632569017 },
];

const NET_0_6_SUMMARY: Summary = Summary { mean_noise: 45463.323345260585, mean_simple: 434.01315667098453, noise_quartiles: [19.313272955995895, 29.576866939825543, 40.57444396296698], simple_quartiles: [25.32026953362572, 35.08396103553255, 43.877453504199394], curving_down: 1, overstep: [1.6726228102916314, 1.8293946547280364, 1.9917756480782072] };

const NET_0_3_ROWS: [Row; 13] = [
    Row { batch: 1, k: 14, steps: [21153, 18627, 17980, 17766, 18695] },
    Row { batch: 2, k: 17, steps: [9493, 10491, 9434, 9267, 10691] },
    Row { batch: 4, k: 22, steps: [4596, 5795, 4637, 4456, 5376] },
    Row { batch: 8, k: 25, steps: [2493, 2970, 2545, 2834, 2835] },
    Row { batch: 16, k: 27, steps: [1732, 1981, 1697, 1718, 1847] },
    Row { batch: 32, k: 31, steps: [945, 1057, 1078, 1052, 1292] },
    Row { batch: 64, k: 33, steps: [667, 738, 691, 749, 803] },
    Row { batch: 128, k: 34, steps: [651, 564, 637, 607, 670] },
    Row { batch: 256, k: 35, steps: [476, 538, 484, 479, 560] },
    Row { batch: 512, k: 35, steps: [515, 518, 572, 592, 531] },
    Row { batch: 1024, k: 35, steps: [519, 666, 552, 544, 580] },
    Row { batch: 2048, k: 33, steps: [551, 563, 544, 563, 578] },
    Row { batch: 0, k: 34, steps: [465, 465, 465, 465, 465] },
];

const NET_0_3_MOMENTS: [Moment; 41] = [
    Moment { step: 0, loss: 1.1146533586921057, g2: 0.01802931933547777, ghg: 0.004872302923251781, trhs: 2.430190257310111, trace: 3.202576585053132 },
    Moment { step: 12, loss: 1.053050958095371, g2: 0.0006028417527308305, ghg: 3.71029539207495e-5, trhs: 2.9512312896834443, trace: 3.3630710028410524 },
    Moment { step: 20, loss: 1.047531840650666, g2: 0.00045877231438663465, ghg: 1.1228989102961255e-5, trhs: 3.387541414013562, trace: 3.594440867014685 },
    Moment { step: 24, loss: 1.0449407022025365, g2: 0.0004905352925772385, ghg: 1.8533432690481305e-5, trhs: 3.620925161336011, trace: 3.7094994365149017 },
    Moment { step: 36, loss: 1.0357834018587595, g2: 0.002742226700929763, ghg: 0.0033250316251310285, trhs: 4.445514580181773, trace: 4.114193013915687 },
    Moment { step: 48, loss: 1.0322840365300916, g2: 0.025035974926853884, ghg: 0.028727092752537015, trhs: 3.515202448322698, trace: 3.9116797088598663 },
    Moment { step: 60, loss: 1.002886276494977, g2: 0.0023205617959787413, ghg: 0.0015077774580230469, trhs: 4.697461550523309, trace: 4.537798574726274 },
    Moment { step: 72, loss: 1.0157202173515503, g2: 0.09711420018305018, ghg: 0.12964461641125483, trhs: 5.828628070763348, trace: 5.380458220767757 },
    Moment { step: 84, loss: 0.9782341862838574, g2: 0.02574757410029844, ghg: 0.032902920480349825, trhs: 5.8945998204368655, trace: 5.4400124645575785 },
    Moment { step: 96, loss: 0.9898855625076525, g2: 0.100088354785128, ghg: 0.1446969723284759, trhs: 6.695508884493913, trace: 6.011118055564911 },
    Moment { step: 108, loss: 0.974239499103888, g2: 0.08995490379842865, ghg: 0.12206039991873485, trhs: 7.52865834570042, trace: 6.492547884226143 },
    Moment { step: 120, loss: 0.9576566551976431, g2: 0.07631590187350143, ghg: 0.10597336461881375, trhs: 7.717266479925076, trace: 6.554088975867611 },
    Moment { step: 132, loss: 0.9673556342547687, g2: 0.15090407093302044, ghg: 0.25954910807863535, trhs: 6.657276275307018, trace: 6.240114300342269 },
    Moment { step: 144, loss: 0.9409032704261171, g2: 0.08425732718071009, ghg: 0.1135811213160809, trhs: 5.867271921320124, trace: 5.787335721525286 },
    Moment { step: 156, loss: 0.9279329639122552, g2: 0.09680424688976454, ghg: 0.14101228483362105, trhs: 6.097590223266025, trace: 5.934066003991475 },
    Moment { step: 168, loss: 0.9175562706985344, g2: 0.11465820156002485, ghg: 0.16591673975369214, trhs: 6.077873324922629, trace: 6.027443673770685 },
    Moment { step: 180, loss: 0.8998182278583029, g2: 0.11787489357414455, ghg: 0.1660014870314248, trhs: 6.118946393712326, trace: 6.125016722490886 },
    Moment { step: 192, loss: 0.8867823186428594, g2: 0.13755655335876735, ghg: 0.18608709344660918, trhs: 6.158253033983381, trace: 6.302230187811249 },
    Moment { step: 204, loss: 0.8719651981045352, g2: 0.1515951721425908, ghg: 0.1868959071983925, trhs: 6.200384392617125, trace: 6.517812095695124 },
    Moment { step: 216, loss: 0.8611334249898702, g2: 0.1767569155719574, ghg: 0.19226513655820537, trhs: 6.40207499442193, trace: 6.834193553548485 },
    Moment { step: 228, loss: 0.8401247730429678, g2: 0.20140631268453313, ghg: 0.2414997138969019, trhs: 7.206654455867856, trace: 7.28946062596853 },
    Moment { step: 240, loss: 0.8707206856553787, g2: 0.3512823168137595, ghg: 0.3848893012139401, trhs: 7.738007659346915, trace: 8.180048537570057 },
    Moment { step: 252, loss: 0.8569382258762986, g2: 0.3328467644092254, ghg: 0.3726364029699518, trhs: 7.335926356322759, trace: 8.189391088562235 },
    Moment { step: 264, loss: 0.8024235829499105, g2: 0.28522502292688273, ghg: 0.3601417793993875, trhs: 8.074640522980717, trace: 8.55098920743366 },
    Moment { step: 276, loss: 0.7871855001240767, g2: 0.31168676729488926, ghg: 0.3802470467135269, trhs: 9.091959257137495, trace: 9.221639026866074 },
    Moment { step: 288, loss: 0.7332435569322799, g2: 0.2613756963084778, ghg: 0.3384301925344097, trhs: 8.018277504912135, trace: 8.497524037831345 },
    Moment { step: 300, loss: 0.7080369740527253, g2: 0.3048298098293643, ghg: 0.4450869598979674, trhs: 8.439760277214367, trace: 8.457658110927811 },
    Moment { step: 312, loss: 0.7439466191284665, g2: 0.5051296595345232, ghg: 0.719420633970109, trhs: 11.97759368157249, trace: 11.342648222973777 },
    Moment { step: 324, loss: 0.7531769722591228, g2: 0.54108681807948, ghg: 0.8559003088269495, trhs: 12.039169196133837, trace: 11.713636395763027 },
    Moment { step: 336, loss: 0.5987749476700034, g2: 0.20912334497756113, ghg: 0.3246740923656199, trhs: 7.728410028266374, trace: 7.552146499747654 },
    Moment { step: 348, loss: 0.6399098510009286, g2: 0.39329503279581013, ghg: 0.603167567013189, trhs: 11.524401046971846, trace: 10.50271656117689 },
    Moment { step: 360, loss: 0.5002884673304914, g2: 0.012608780645877683, ghg: 0.011918525985014826, trhs: 3.9738591121596243, trace: 4.788054321582422 },
    Moment { step: 372, loss: 0.47301552168856437, g2: 0.03337039520303629, ghg: 0.04168218318795989, trhs: 4.711732172692748, trace: 5.514849915824677 },
    Moment { step: 384, loss: 0.48235373686946226, g2: 0.12409901839172427, ghg: 0.13487369731508564, trhs: 4.800813968947124, trace: 6.510215949414317 },
    Moment { step: 396, loss: 0.44844278301235935, g2: 0.12379193381210817, ghg: 0.1563661251064704, trhs: 5.9658639899723696, trace: 6.621914824092839 },
    Moment { step: 408, loss: 0.4124275256110443, g2: 0.10617687714911835, ghg: 0.1558872285722218, trhs: 6.108814133890388, trace: 5.921391508852274 },
    Moment { step: 420, loss: 0.4017511912384135, g2: 0.14909448130754013, ghg: 0.24144152298865065, trhs: 7.005925788916843, trace: 6.348358637796456 },
    Moment { step: 432, loss: 0.37673236400987725, g2: 0.13148923504530458, ghg: 0.21064235816356505, trhs: 6.57576523612912, trace: 5.923674623695795 },
    Moment { step: 444, loss: 0.3604035218168628, g2: 0.14635651130279884, ghg: 0.2564155180496196, trhs: 7.1341982709701055, trace: 6.096837759620501 },
    Moment { step: 456, loss: 0.37037779081624067, g2: 0.2158632296381438, ghg: 0.3879642921627859, trhs: 8.635964456888441, trace: 7.296196046324284 },
    Moment { step: 465, loss: 0.29928274808104155, g2: 0.03314696026587483, ghg: 0.041038669722255325, trhs: 2.969939777417764, trace: 3.4702105349903203 },
];

const NET_0_3_SUMMARY: Summary = Summary { mean_noise: 7711.972783650977, mean_simple: 495.27707740380515, noise_quartiles: [22.61747665296803, 33.416803480149206, 61.01274671562575], simple_quartiles: [30.512207366561153, 46.45769756557346, 77.26817930255321], curving_down: 0, overstep: [1.7667158978580377, 1.9131541530297504, 2.0637180807096436] };

const TARGETS: [Target; 2] = [
    Target { loss: 0.6, rows: &NET_0_6_ROWS, moments: &NET_0_6_MOMENTS, summary: NET_0_6_SUMMARY },
    Target { loss: 0.3, rows: &NET_0_3_ROWS, moments: &NET_0_3_MOMENTS, summary: NET_0_3_SUMMARY },
];

// ----- how the tables were made ---------------------------------------------------------------------

/// The network's problem and how it was measured. (Only the tests run it: the scene shows
/// the tables.)
const NET_POINTS: usize = 256;
#[cfg(test)]
const NET_DATA_SEED: u64 = 5;
#[cfg(test)]
const NET_INIT_SEED: u64 = 7;
const NET_WIDTH: usize = 8;
const NET_DEPTH: usize = 2;
/// The batch sizes measured (the true gradient besides).
#[cfg(test)]
const NET_BATCHES: [u32; 12] = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048];
/// The weights of the network 2 → w → … → w → 3 with biases.
fn net_params() -> usize {
    3 * NET_WIDTH + (NET_DEPTH - 1) * (NET_WIDTH * NET_WIDTH + NET_WIDTH) + 3 * NET_WIDTH + 3
}

/// The losses to reach.
#[cfg(test)]
const NET_TARGETS: [f64; 2] = [0.6, 0.3];

/// Step size k of the grid: 2^(k/4) / 256.
fn lr_of(k: i32) -> f64 {
    2f64.powf(k as f64 / 4.0) / 256.0
}

#[cfg(test)]
mod measure {
    use super::*;
    use crate::math::arch::{Init, MlpSpec, Model, Norm, Place};
    use crate::math::autograd::{Act, Tape};
    use crate::math::terrain::{Draw, FullNoise, Problem, Sampler};

    /// 256 points on three spiral arms.
    pub fn problem() -> Problem {
        Problem::spirals(NET_POINTS, 3, &mut Rng::new(NET_DATA_SEED))
    }

    /// The tanh network 2 → 8 → 8 → 3 at its start (Xavier's, from one seed).
    pub fn model() -> Model {
        let spec = MlpSpec {
            inputs: 2,
            width: NET_WIDTH,
            depth: NET_DEPTH,
            outputs: 3,
            act: Act::Tanh,
            norm: Norm::None,
            place: Place::Pre,
            residual: false,
            init: Init::Xavier,
            ..MlpSpec::default()
        };
        Model::mlp(&spec, &mut Rng::new(NET_INIT_SEED))
    }

    /// One step: SGD with a batch of b drawn with replacement (0: the true gradient).
    fn step(problem: &Problem, model: &mut Model, tape: &mut Tape, sampler: &mut Sampler, b: u32, lr: f64) {
        if b == 0 {
            problem.full_grad(model, tape);
        } else {
            problem.grad(model, tape, sampler.deal());
        }
        model.step_sgd(lr);
    }

    /// The steps of SGD with batches of b (from the mini-batch seed `seed`) at step size
    /// lr until the loss on all the data is first at or below the target, checked after
    /// every step, at most `cap`; None if not by then, or if the loss blows up (over 50,
    /// or not a number).
    pub fn steps_to(problem: &Problem, start: &Model, b: u32, lr: f64, target: f64, cap: u64, seed: u64) -> Option<u64> {
        let mut model = start.clone();
        let mut tape = Tape::new();
        let mut sampler = Sampler::new(problem.len(), b.max(1) as usize, Draw::WithReplacement, Rng::new(seed));
        if problem.loss(&model, &mut tape) <= target {
            return Some(0);
        }
        for t in 1..=cap {
            step(problem, &mut model, &mut tape, &mut sampler, b, lr);
            let l = problem.loss(&model, &mut tape);
            if l.is_nan() || l > 50.0 {
                return None;
            }
            if l <= target {
                return Some(t);
            }
        }
        None
    }

    /// The mean steps over the seeds at grid point k, with each seed's; None if a seed
    /// fails, or once their sum passes `budget` times the number of seeds (it cannot then
    /// beat a mean of `budget`).
    fn mean_at(problem: &Problem, start: &Model, b: u32, k: i32, target: f64, budget: Option<f64>) -> Option<(f64, [u32; SEEDS])> {
        let seeds = if b == 0 { 1 } else { SEEDS };
        let mut all = [0u32; SEEDS];
        let mut sum = 0u64;
        for (i, slot) in all.iter_mut().enumerate().take(seeds) {
            let cap = match budget {
                Some(bud) => ((bud * seeds as f64).floor() as u64).saturating_sub(sum),
                None => 2_000_000,
            };
            if cap == 0 {
                return None;
            }
            let s = steps_to(problem, start, b, lr_of(k), target, cap, i as u64 + 1)?;
            sum += s;
            *slot = s as u32;
        }
        if b == 0 {
            all = [all[0]; SEEDS];
        }
        Some((sum as f64 / seeds as f64, all))
    }

    /// The grid point with the fewest mean steps for batch b, searched outwards from
    /// `start`: down the grid and up it, each way until four points in a row have not
    /// improved on the best (a point whose seeds' sum passes the best's is not finished).
    pub fn best(problem: &Problem, start_model: &Model, b: u32, target: f64, start: i32) -> (i32, f64, [u32; SEEDS]) {
        let mut best: Option<(i32, f64, [u32; SEEDS])> = None;
        let mut tried: Vec<i32> = Vec::new();
        let eval = |k: i32, best: &mut Option<(i32, f64, [u32; SEEDS])>, tried: &mut Vec<i32>| -> bool {
            tried.push(k);
            match mean_at(problem, start_model, b, k, target, best.as_ref().map(|b| b.1)) {
                Some((m, all)) if best.as_ref().is_none_or(|b| m < b.1) => {
                    *best = Some((k, m, all));
                    true
                }
                _ => false,
            }
        };
        let mut k = start;
        eval(k, &mut best, &mut tried);
        while best.is_none() && k > -40 {
            k -= 2;
            eval(k, &mut best, &mut tried);
        }
        for dir in [-1i32, 1] {
            let mut worse = 0;
            let mut k = best.as_ref().map_or(start, |b| b.0);
            while worse < 4 && k > -40 && k < 60 {
                k += dir;
                if tried.contains(&k) {
                    continue;
                }
                if eval(k, &mut best, &mut tried) {
                    worse = 0;
                } else {
                    worse += 1;
                }
            }
        }
        best.expect("some step size reaches the target")
    }

    /// The measurement of one target: every batch at its best step size (the true
    /// gradient last), the full-batch run measured at every step, the moments kept, and
    /// the summary.
    pub fn target(loss: f64) -> (Vec<Row>, Vec<Moment>, Summary) {
        let problem = problem();
        let start = model();
        let mut rows = Vec::new();
        let mut k = 32;
        for b in std::iter::once(0).chain(NET_BATCHES.iter().rev().copied()) {
            let (bk, _, steps) = best(&problem, &start, b, loss, k);
            k = bk;
            rows.push(Row { batch: b, k: bk, steps });
        }
        rows.rotate_left(1);
        rows.reverse();
        rows.rotate_left(1);
        // (now the batches in increasing order, the true gradient last)
        let full = rows[rows.len() - 1];
        let total = full.steps[0] as usize;
        let mut model = start.clone();
        let mut tape = Tape::new();
        let mut all: Vec<Moment> = Vec::with_capacity(total + 1);
        for s in 0..=total {
            if s > 0 {
                problem.full_grad(&mut model, &mut tape);
                model.step_sgd(full.lr());
            }
            let mut noise = FullNoise::new(&problem, &model);
            noise.advance(&problem, &mut model, &mut tape, usize::MAX);
            let n = noise.per_example().noise();
            all.push(Moment { step: s as u32, loss: problem.loss(&model, &mut tape), g2: n.g2, ghg: noise.ghg(), trhs: noise.trace_hs(), trace: n.trace });
        }
        // the summary, over the points at which the run took its steps (0 to S − 1)
        let taken = &all[..total];
        let quartiles = |mut v: Vec<f64>| -> [f64; 3] {
            v.sort_by(f64::total_cmp);
            let at = |q: f64| {
                let x = q * (v.len() - 1) as f64;
                let (i, frac) = (x.floor() as usize, x - x.floor());
                v[i] + (v[(i + 1).min(v.len() - 1)] - v[i]) * frac
            };
            [at(0.25), at(0.5), at(0.75)]
        };
        let noise: Vec<f64> = taken.iter().filter(|m| m.ghg > 0.0).map(|m| m.trhs / m.ghg).collect();
        let simple: Vec<f64> = taken.iter().map(|m| m.simple()).collect();
        let tenth = total / 10;
        let over: Vec<f64> = taken[tenth..].iter().filter(|m| m.ghg > 0.0).map(|m| full.lr() / (m.g2 / m.ghg)).collect();
        let summary = Summary {
            mean_noise: noise.iter().sum::<f64>() / noise.len() as f64,
            mean_simple: simple.iter().sum::<f64>() / simple.len() as f64,
            noise_quartiles: quartiles(noise.clone()),
            simple_quartiles: quartiles(simple),
            curving_down: (taken.len() - noise.len()) as u32,
            overstep: quartiles(over),
        };
        // the moments kept: every few steps, the end, the largest noise scale, and the
        // most negative curvature along the gradient (if any)
        let every = total.div_ceil(40).max(1);
        let mut keep: Vec<usize> = (0..=total).filter(|s| s % every == 0).collect();
        keep.push(total);
        if let Some(i) = (0..total).filter(|&i| all[i].ghg > 0.0).max_by(|&a, &b| (all[a].trhs / all[a].ghg).total_cmp(&(all[b].trhs / all[b].ghg))) {
            keep.push(i);
        }
        if let Some(i) = (0..total).filter(|&i| all[i].ghg <= 0.0).min_by(|&a, &b| all[a].ghg.total_cmp(&all[b].ghg)) {
            keep.push(i);
        }
        keep.sort();
        keep.dedup();
        (rows, keep.into_iter().map(|i| all[i]).collect(), summary)
    }

    /// The tables as Rust, to paste.
    pub fn print_target(name: &str, rows: &[Row], moments: &[Moment], summary: &Summary) {
        println!("const {name}_ROWS: [Row; {}] = [", rows.len());
        for r in rows {
            println!("    Row {{ batch: {}, k: {}, steps: {:?} }},", r.batch, r.k, r.steps);
        }
        println!("];");
        println!("const {name}_MOMENTS: [Moment; {}] = [", moments.len());
        for m in moments {
            println!("    Moment {{ step: {}, loss: {:?}, g2: {:?}, ghg: {:?}, trhs: {:?}, trace: {:?} }},", m.step, m.loss, m.g2, m.ghg, m.trhs, m.trace);
        }
        println!("];");
        println!("const {name}_SUMMARY: Summary = {summary:?};");
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::math::terrain::{Nqm, MOMENTA, NQM_DIM, NQM_TARGET};

    fn close(a: f64, b: f64, rel: f64) -> bool {
        (a - b).abs() <= rel * a.abs().max(b.abs())
    }

    /// The noisy quadratic model's tables, computed again: the standard model, each
    /// rule's best learning rate (and momentum, from `MOMENTA`) at each batch size, by
    /// the library's search; printed as Rust and compared with the tables. (About a
    /// minute and a half.)
    #[test]
    #[ignore]
    fn regenerate_the_quadratic_tables() {
        let nqm = Nqm::standard(NQM_DIM);
        let batches: Vec<f64> = (0..NQM_ROWS - 1).map(|k| 2f64.powi(k as i32)).chain([f64::INFINITY]).collect();
        for r in Rule::ALL {
            let (momenta, power, name) = match r {
                Rule::Sgd => (vec![0.0], 0.0, "NQM_SGD"),
                Rule::Momentum => (MOMENTA.to_vec(), 0.0, "NQM_MOMENTUM"),
                Rule::Preconditioned => (vec![0.0], NQM_POWER, "NQM_PRECONDITIONED"),
            };
            println!("const {name}: [NqmRow; NQM_ROWS] = [");
            for (i, &b) in batches.iter().enumerate() {
                let t = nqm.best_lr_momentum(NQM_TARGET, b, &momenta, power, 0.0, 1 << 40).unwrap();
                let row = NqmRow { batch: b, steps: t.steps, lr: t.lr, momentum: t.momentum };
                let b_text = if b.is_finite() { format!("{b:?}") } else { "f64::INFINITY".to_string() };
                println!("    NqmRow {{ batch: {b_text}, steps: {}, lr: {:?}, momentum: {:?} }},", row.steps, row.lr, row.momentum);
                let stored = r.table()[i];
                assert!(stored.batch == row.batch && stored.steps == row.steps && close(stored.lr, row.lr, 1e-9) && stored.momentum == row.momentum, "{r:?} {stored:?} {row:?}");
            }
            println!("];");
        }
    }

    /// The network's tables, measured again (several minutes): printed as Rust and
    /// compared with the tables.
    #[test]
    #[ignore]
    fn regenerate_the_network_tables() {
        let measured: Vec<_> = NET_TARGETS
            .iter()
            .enumerate()
            .map(|(i, &loss)| {
                let m = measure::target(loss);
                measure::print_target(&format!("NET_{}", ["0_6", "0_3"][i]), &m.0, &m.1, &m.2);
                m
            })
            .collect();
        for (i, (rows, moments, summary)) in measured.into_iter().enumerate() {
            let loss = NET_TARGETS[i];
            let t = &TARGETS[i];
            assert_eq!(t.loss, loss);
            assert_eq!(t.rows, &rows[..], "target {loss}: the rows");
            assert_eq!(t.moments.len(), moments.len());
            for (a, b) in t.moments.iter().zip(&moments) {
                assert!(a.step == b.step && close(a.loss, b.loss, 1e-9) && close(a.g2, b.g2, 1e-9) && close(a.ghg, b.ghg, 1e-7) && close(a.trhs, b.trhs, 1e-9), "{a:?} {b:?}");
            }
            let (a, b) = (t.summary, summary);
            assert!(close(a.mean_noise, b.mean_noise, 1e-7) && close(a.mean_simple, b.mean_simple, 1e-9) && a.curving_down == b.curving_down, "{a:?} {b:?}");
        }
    }

    /// The fit recovers McCandlish's relation from exact data, is a least-squares minimum
    /// (no nearby B_crit does better), and needs two points.
    #[test]
    fn the_fit_recovers_the_relation() {
        for (s_min, b_crit) in [(300.0, 40.0), (6_565.0, 863.0), (82.0, 18_000.0), (1.0, 0.01)] {
            let pts: Vec<(f64, f64)> = (0..12).map(|k| 2f64.powi(k)).chain([f64::INFINITY]).map(|b| (b, s_min * (1.0 + b_crit / b))).collect();
            let fit = fit(&pts).unwrap();
            assert!(close(fit.b_crit, b_crit, 1e-6) && close(fit.s_min, s_min, 1e-6) && fit.rms < 1e-6, "{fit:?}");
            assert!(close(fit.e_min(), s_min * b_crit, 1e-6));
            // the identities: at B_crit twice the fewest steps and examples
            assert!(close(fit.steps(b_crit), 2.0 * s_min, 1e-9));
            assert!(close(terrain::steps_over_min(3.0, 7.0) * 3.0 / 7.0, terrain::examples_over_min(3.0, 7.0) * 3.0 / 7.0 * 7.0 / 3.0, 1e-12));
        }
        // a least-squares minimum on noisy data
        let pts: Vec<(f64, f64)> = TARGETS[1].rows.iter().map(|r| (r.b(), r.mean())).collect();
        let f0 = fit(&pts).unwrap();
        let cost = |bc: f64| {
            let r: Vec<f64> = pts.iter().map(|&(b, s)| s.ln() - (bc / b).ln_1p()).collect();
            let m = r.iter().sum::<f64>() / r.len() as f64;
            r.iter().map(|x| (x - m).powi(2)).sum::<f64>()
        };
        for k in [0.9, 0.99, 1.01, 1.1] {
            assert!(cost(f0.b_crit * k) > cost(f0.b_crit), "{k}");
        }
        assert!(fit(&[(4.0, 100.0)]).is_none() && fit(&[]).is_none() && fit(&[(1.0, f64::NAN), (2.0, -1.0)]).is_none());
    }

    /// The one-step model as the charts draw it: the lowest point of every parabola lies
    /// on the line −½|G|²η; twice the best step gives no change; at B = B_noise the gain is
    /// half the true gradient's; the gain is proportional to B far below it.
    #[test]
    fn the_one_step_model_as_drawn() {
        for m in TARGETS.iter().flat_map(|t| t.moments.iter()).filter(|m| m.ghg > 0.0) {
            let one = m.one_step();
            for b in [1.0, 7.0, 64.0, 1e4, f64::INFINITY] {
                let (e, g) = (one.best_step(b), best_gain(&one, b).unwrap());
                assert!(close(-g, -0.5 * one.g2 * e, 1e-12), "the lowest point on the line");
                assert!(close(one.change(e, b), -g, 1e-12));
                assert!(one.change(2.0 * e, b).abs() < 1e-12 * g.max(1e-300) * 10.0);
            }
            let bn = noise_scale(&one).unwrap();
            assert!(close(best_gain(&one, bn).unwrap(), 0.5 * one.max_gain(), 1e-12));
            let tiny = bn * 1e-4;
            assert!(close(best_gain(&one, 2.0 * tiny).unwrap() / best_gain(&one, tiny).unwrap(), 2.0, 1e-3));
        }
    }

    /// The noisy quadratic model's table is the library's: the steps at each stored
    /// learning rate (and momentum) are the stored steps, for every row of every rule; and
    /// the search finds the stored rate again at one batch size.
    #[test]
    fn the_quadratic_table_is_reproduced() {
        let nqm = Nqm::standard(NQM_DIM);
        for r in Rule::ALL {
            let power = if r == Rule::Preconditioned { NQM_POWER } else { 0.0 };
            for row in r.table() {
                let rule = terrain::Rule { lr: row.lr, momentum: row.momentum, power, average: 0.0 };
                assert_eq!(nqm.steps_to(NQM_TARGET, &rule, row.batch, 1 << 40), Some(row.steps), "{r:?} {row:?}");
            }
        }
        let best = nqm.best_lr(NQM_TARGET, 64.0, 0.0, 0.0, 0.0, 1 << 40).unwrap();
        let row = NQM_SGD[6];
        assert_eq!(row.batch, 64.0);
        assert!(best.steps == row.steps && close(best.lr, row.lr, 1e-12), "{best:?} {row:?}");
    }

    /// The network's table where it is quick to check: the network has the weights the
    /// panel says; the first moment of each run measured again; the full-batch run of the
    /// opening target and one seed of one batch size take the stored steps.
    #[test]
    fn the_network_table_is_reproduced() {
        use crate::math::autograd::Tape;
        use crate::math::terrain::FullNoise;
        let problem = measure::problem();
        let start = measure::model();
        assert_eq!(start.n_params(), net_params());
        assert_eq!(net_params(), 123);
        let mut tape = Tape::new();
        let mut m = start.clone();
        let mut full = FullNoise::new(&problem, &m);
        full.advance(&problem, &mut m, &mut tape, usize::MAX);
        for t in &TARGETS {
            let at = t.moments[0];
            assert_eq!(at.step, 0);
            assert!(close(at.g2, full.per_example().noise().g2, 1e-12) && close(at.ghg, full.ghg(), 1e-9) && close(at.trhs, full.trace_hs(), 1e-9));
            assert!(close(at.loss, problem.loss(&start, &mut tape), 1e-12));
        }
        let t = &TARGETS[TARGET];
        let gd = t.full();
        assert_eq!(measure::steps_to(&problem, &start, 0, gd.lr(), t.loss, 10_000, 1), Some(gd.steps[0] as u64));
        let row = t.rows.iter().find(|r| r.batch == 64).unwrap();
        assert_eq!(measure::steps_to(&problem, &start, 64, row.lr(), t.loss, 100_000, 1), Some(row.steps[0] as u64));
    }

    /// The mean noise scale of the opening target's full-batch run is the table's, and the
    /// steps early in the run where it is over 1 000 (where the true gradient almost
    /// vanished) make nearly all of it, as experiment 5 says: measured again at every step
    /// (a minute).
    #[test]
    #[ignore]
    fn a_few_steps_make_most_of_the_mean_noise_scale() {
        use crate::math::autograd::Tape;
        use crate::math::terrain::FullNoise;
        let problem = measure::problem();
        let mut model = measure::model();
        let mut tape = Tape::new();
        let t = &TARGETS[TARGET];
        let full = t.full();
        let mut scales = Vec::new();
        for _ in 0..full.steps[0] {
            let mut noise = FullNoise::new(&problem, &model);
            noise.advance(&problem, &mut model, &mut tape, usize::MAX);
            scales.push((noise.ghg() > 0.0).then(|| noise.noise_scale()));
            problem.full_grad(&mut model, &mut tape);
            model.step_sgd(full.lr());
        }
        let defined: Vec<(usize, f64)> = scales.iter().enumerate().filter_map(|(i, v)| v.map(|v| (i, v))).collect();
        let total: f64 = defined.iter().map(|v| v.1).sum();
        assert!(close(total / defined.len() as f64, t.summary.mean_noise, 1e-9));
        let big: Vec<&(usize, f64)> = defined.iter().filter(|v| v.1 > 1000.0).collect();
        let share = big.iter().map(|v| v.1).sum::<f64>() / total;
        let last = big.iter().map(|v| v.0).max().unwrap();
        println!("{} steps over 1 000, the last at step {last}, make {share:.4} of the sum", big.len());
        let says = |what: &str| assert!(TRIALS[4].caption.contains(what), "experiment 4 should say \"{what}\"");
        says(&format!("{} of its first {} steps", big.len(), last + 1));
        says(&format!("make {} % of it", num(100.0 * share, 0)));
    }

    /// The scene opens on the network at its start, with a batch at its noise scale.
    #[test]
    fn the_scene_opens_at_the_noise_scale() {
        let s = NoiseScale::create();
        let one = s.setup.one();
        let bn = noise_scale(&one).unwrap();
        assert!((s.setup.batch - bn).abs() <= 0.5, "{} {bn}", s.setup.batch);
        assert!((best_gain(&one, s.setup.batch).unwrap() / one.max_gain() - 0.5).abs() < 1e-3);
        assert_eq!(s.setup.whole, Whole::Model);
    }

    /// Degenerate numbers: no gradient, no noise, no curvature along the gradient, the
    /// network's moment where it curves down, the ends of every range, every source of a
    /// whole run: everything drawn is finite and every range is a range.
    #[test]
    fn degenerate_settings_are_told_honestly() {
        let down = TARGETS.iter().flat_map(|t| t.moments.iter().enumerate().map(move |(i, m)| (t, i, m))).find(|(_, _, m)| m.ghg <= 0.0);
        let mut cases: Vec<Setup> = Vec::new();
        let open = Setup::opening();
        for hand in [
            OneStep { g2: 0.0, ghg: 0.0, trhs: 0.0 },
            OneStep { g2: 1.0, ghg: 0.0, trhs: 0.0 },
            OneStep { g2: 1.0, ghg: 0.0, trhs: 5.0 },
            OneStep { g2: 1.0, ghg: 2.0, trhs: 0.0 },
            OneStep { g2: HAND_G2[0] as f64, ghg: HAND_GHG[1] as f64, trhs: HAND_TRHS[1] as f64 },
            OneStep { g2: HAND_G2[1] as f64, ghg: HAND_GHG[0] as f64, trhs: HAND_TRHS[0] as f64 },
            OneStep { g2: f64::NAN, ghg: -3.0, trhs: f64::INFINITY },
        ] {
            for b in [1.0, MOST_BATCH, f64::NAN, -4.0] {
                cases.push(Setup { source: Source::Hand, hand, batch: b, second: Some(b * 2.0), ..open });
            }
        }
        if let Some((t, i, _)) = down {
            let target = TARGETS.iter().position(|q| std::ptr::eq(q, t)).unwrap();
            for b in [1.0, 4.0, 64.0, MOST_BATCH] {
                cases.push(Setup { target, moment: i, batch: b, earlier: Some(0), ..open });
            }
        }
        cases.push(Setup { target: 99, moment: 999, earlier: Some(999), ..open });
        let n = cases.len();
        for (k, c) in cases.into_iter().enumerate() {
            for whole in Whole::ALL {
                let s = Setup { whole, ..c }.kept();
                assert!(s.batch >= 1.0 && s.batch <= MOST_BATCH && s.hand.g2.is_finite() && s.hand.ghg >= 0.0 && s.hand.trhs >= 0.0, "{s:?}");
                let g = s.goal();
                for i in 0..4 {
                    assert!(g.frames[i].iter().all(|v| v.is_finite()), "case {k} of {n} {whole:?}: frame {i} {:?}", g.frames[i]);
                    assert!(g.frames[i].len() >= 3);
                    assert!(g.content[i].iter().all(|v| v.is_finite()), "case {k} {whole:?}: content {i}");
                    assert_eq!(g.content[i].len(), g.present[i].len());
                }
                let fa = s.a_frame();
                assert!(fa[0] > 0.0 && fa[2] > fa[1], "{fa:?}");
                for fr in [s.b_frame(), s.c_frame(), s.d_frame()] {
                    assert!(fr[1] > fr[0] && fr[3] > fr[2], "{fr:?}");
                }
            }
        }
        // the moment that curves down says so
        if let Some((t, i, m)) = down {
            let target = TARGETS.iter().position(|q| std::ptr::eq(q, t)).unwrap();
            let s = Setup { target, moment: i, batch: 1e4, ..open };
            assert!(noise_scale(&s.one()).is_none() && best_gain(&s.one(), 1e4).is_none() && m.ghg <= 0.0);
        }
    }

    /// The layout keeps the charts and the buttons in the window and apart, from a phone
    /// to a desktop.
    #[test]
    fn the_layout_fits_every_window() {
        let sizes = [(1080.0, 1000.0), (1600.0, 1000.0), (1100.0, 700.0), (560.0, 700.0), (390.0, 800.0), (360.0, 640.0), (700.0, 500.0), (300.0, 600.0), (500.0, 400.0), (240.0, 180.0)];
        for (w, h) in sizes {
            for page in [Page::Step, Page::Run] {
                for whole in Whole::ALL {
                    let lay = Layout::of(Vec2::new(w, h), 40.0, page, whole);
                    let inside = |(a, b): Rect| a.x >= 0.0 && a.y >= 0.0 && b.x <= w + 0.5 && b.y <= h + 0.5;
                    let apart = |(a, b): Rect, (c, d): Rect| b.x <= c.x || d.x <= a.x || b.y <= c.y || d.y <= a.y;
                    let mut boxes: Vec<Rect> = (0..4).filter(|&k| lay.shown[k]).map(|k| lay.plots[k]).collect();
                    boxes.extend(lay.pages.iter().chain(&lay.wholes).chain(&lay.rules).filter(|r| r.1.x > r.0.x));
                    assert!(!boxes.is_empty());
                    for (i, a) in boxes.iter().enumerate() {
                        assert!(inside(*a), "{w}x{h} {page:?} {whole:?}: box {i} {a:?}");
                        for b in &boxes[i + 1..] {
                            assert!(apart(*a, *b), "{w}x{h} {page:?} {whole:?}: {a:?} and {b:?}");
                        }
                    }
                    for k in (0..4).filter(|&k| lay.shown[k]) {
                        let (a, b) = lay.plots[k];
                        if !lay.compact && h >= 600.0 {
                            assert!(b.x - a.x > 200.0 && b.y - a.y > 150.0, "{w}x{h}: chart {k} {a:?} {b:?}");
                        }
                        assert!(lay.heads[k].y < a.y, "{w}x{h}: the title of {k} over its chart");
                    }
                }
            }
        }
    }

    /// The numbers each experiment's caption says, from the scene as its button sets it up.
    #[test]
    fn the_experiments_show_what_their_captions_say() {
        let says = |k: usize, what: &str| assert!(TRIALS[k].caption.contains(what), "experiment {k} should say \"{what}\": {}", TRIALS[k].caption);
        let pct = |v: f64| format!("{} %", num(100.0 * v, 1));
        // far below the noise scale, and twice it
        let s = trial(0).kept();
        let one = s.one();
        let bn = noise_scale(&one).unwrap();
        let (b1, b2) = (s.batch, s.second.unwrap());
        assert_eq!((b1, b2), (16.0, 32.0));
        says(0, &format!(r"$B_{{\text{{noise}}}} = {}$", sig_bare(bn)));
        assert!(b2 < bn / 10.0);
        let gain = best_gain(&one, b2).unwrap() / best_gain(&one, b1).unwrap();
        assert!(close(one.best_step(b2) / one.best_step(b1), gain, 1e-12));
        says(0, &format!("{} times as long", num(gain, 2)));
        assert!(close(model_steps(bn, b1) / model_steps(bn, b2), gain, 1e-12));
        says(0, &format!("{} times fewer steps", num(gain, 2)));
        says(0, &format!("for {} more examples", pct(model_examples(bn, b2) / model_examples(bn, b1) - 1.0)));
        // far above it
        let s = trial(1).kept();
        let (b1, b2) = (s.batch, s.second.unwrap());
        says(1, &format!("{} and {}", count(b1 as usize), count(b2 as usize)));
        says(1, &format!("{} and {} times the noise scale", num(b1 / bn, 0), num(b2 / bn, 0)));
        let short_by = 1.0 - one.best_step(b1) / one.max_step();
        assert!(short_by < 0.06 && short_by > 0.05, "{short_by}");
        says(1, &format!(r"$\eta_{{\max}} = {}$", num(one.max_step(), 2)));
        let gain = best_gain(&one, b2).unwrap() / best_gain(&one, b1).unwrap();
        says(1, &format!("{} more in a step", pct(gain - 1.0)));
        says(1, &format!("{} fewer steps", pct(1.0 - model_steps(bn, b2) / model_steps(bn, b1))));
        says(1, &format!("{} times the examples", num(model_examples(bn, b2) / model_examples(bn, b1), 2)));
        // momentum and a preconditioner, in the noisy quadratic model
        let s = trial(2).kept();
        assert_eq!(s.whole, Whole::Nqm);
        let (sgd, mom, pre) = (NQM_SGD, NQM_MOMENTUM, NQM_PRECONDITIONED);
        says(2, &format!("{} steps against {}", count(mom[0].steps as usize), count(sgd[0].steps as usize)));
        let eff = mom[0].lr / (1.0 - mom[0].momentum);
        assert!(close(eff, sgd[0].lr, 0.01), "{eff} {}", sgd[0].lr);
        says(2, &format!(r"$\eta/(1 - \beta) = {}$", num(eff, 4)));
        says(2, &format!("{} steps against SGD's {}", count(mom[NQM_ROWS - 1].steps as usize), count(sgd[NQM_ROWS - 1].steps as usize)));
        let fits = Rule::ALL.map(|r| r.fit().unwrap());
        says(2, &format!("{} against {}", sig(fits[1].b_crit), sig(fits[0].b_crit)));
        says(2, &format!("{} times faster even at $B = 1$, and bends at {}", num(sgd[0].steps as f64 / pre[0].steps as f64, 1), sig(fits[2].b_crit)));
        assert!(pre[NQM_ROWS - 1].steps < mom[NQM_ROWS - 1].steps);
        // the network early and late
        let s = trial(3).kept();
        let t = s.target();
        let (start, end) = (t.moments[0], *s.moment());
        assert_eq!(end.step, t.full().steps[0]);
        assert_eq!(s.earlier, Some(0));
        let bn_end = end.one_step().noise_scale();
        says(3, &format!("fell from {} to {}", sig(start.one_step().noise_scale()), sig(bn_end)));
        says(3, &format!("from {} to {}", sig(start.simple()), sig(end.simple())));
        says(3, &format!("a loss of {}", num(end.loss, 2)));
        let spike = t.moments.iter().filter(|m| m.ghg > 0.0).max_by(|a, b| a.one_step().noise_scale().total_cmp(&b.one_step().noise_scale())).unwrap();
        says(3, &format!("near step {}", spike.step));
        says(3, &format!("it reached {}", sig(spike.one_step().noise_scale())));
        let q = t.summary.noise_quartiles;
        says(3, &format!("between {} and {}", sig(q[0]), sig(q[2])));
        let o = t.summary.overstep;
        says(3, &format!("between {} and {} times", num(o[0], 1), num(o[2], 1)));
        // the measured curve against the model's
        let s = trial(4).kept();
        assert_eq!(s.whole, Whole::Network);
        let t = s.target();
        let fit = t.fit().unwrap();
        says(4, &format!(r"$B_{{\text{{crit}}}} = {}$", sig(fit.b_crit)));
        let worst = t.rows.iter().map(|r| (r.mean() / fit.steps(r.b()) - 1.0).abs()).fold(0.0, f64::max);
        says(4, &format!("within {} %", num(100.0 * worst, 0)));
        says(4, &format!("between {} and {}", sig(q[0]), sig(q[2])));
        // the steps nearly halve up to 8, then fall by less, and stop falling beyond 256
        let means: Vec<f64> = t.rows.iter().filter(|r| r.batch > 0).map(Row::mean).collect();
        let halving: Vec<f64> = (0..3).map(|i| means[i] / means[i + 1]).collect();
        let (lo, hi) = (halving.iter().copied().fold(f64::INFINITY, f64::min), halving.iter().copied().fold(0.0, f64::max));
        says(4, &format!("up to 8 (by {} to {} times)", num(lo, 1), num(hi, 1)));
        assert!((3..7).all(|i| means[i] / means[i + 1] < lo), "less from there");
        let at256 = t.rows.iter().position(|r| r.batch == 256).unwrap();
        assert!(t.rows[at256..t.rows.len() - 1].iter().all(|r| r.mean() >= t.rows[at256].mean()), "no fewer steps beyond 256");
        says(4, "stop falling beyond 256");
        assert!(fit.b_crit > q[0] && fit.b_crit < q[2], "the bend within the middle half of the noise scale");
        // where the noise scale is over 1 000, the true gradient is small (what experiments 4
        // and 5 say)
        let mut g2: Vec<f64> = t.moments.iter().map(|m| m.g2).collect();
        g2.sort_by(f64::total_cmp);
        let median = g2[g2.len() / 2];
        for m in t.moments.iter().filter(|m| m.one_step().noise_scale() > 1000.0) {
            assert!(m.step < 65 && m.g2 < 0.05 * median, "{m:?}");
        }
        let mean = t.summary.mean_noise;
        says(4, &format!("{}, which the one-step model", sig(mean)));
        says(4, &format!("{} times too large", num(mean / fit.b_crit, 0)));
    }

    /// After every experiment the picture glides to rest.
    #[test]
    fn glides_arrive_and_stop() {
        let mut s = NoiseScale::create();
        for k in [0, 1, 2, 3, 4, 0] {
            s.set_up(k);
            let mut frames = 0;
            while {
                s.look.follow(&s.setup.goal(), 1.0 / 60.0, false);
                s.look.moving
            } {
                frames += 1;
                assert!(frames < 400, "experiment {k} still moving");
            }
            assert!(frames > 3, "experiment {k}: {frames}");
        }
    }
}

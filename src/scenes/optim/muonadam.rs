//! Muon against Adam on one network. Three rules (SGD with momentum, AdamW, Muon) train
//! the same small network from the same starting weights on the same batches, each at
//! twelve step sizes, with three seeds and two batch sizes; what they did is recorded
//! beforehand in the tables at the end of this file and only read here. The picture
//! shows the loss against the step, the final loss against the step size, the shape of
//! each rule's step on a chosen layer, the singular values of that layer's weights, and
//! how large each layer's step is beside its weights.
//!
//! The runs are those of `tests::record::everything`, with the library's rules
//! (`math::optim`) and problems (`math::terrain`); the ignored test
//! `the_tables_are_what_training_gives` makes them again and compares. Everything the
//! captions say is computed from the tables by the tests at the end.

use super::*;
use crate::scenes::stats::plot::{self, Glide, Plot, Rect, AXES, MARKS, OWN};

pub const DEF: SceneDef = SceneDef {
    id: "muonadam",
    title: "Muon against Adam on one network",
    group: GROUP,
    blurb: "The same small network trained by both: the loss, the shape of each update, the learning rates each tolerates, and what changes with the batch size.",
    make: |_| Box::new(MuonAdam::create()),
};

// ----- what was recorded -------------------------------------------------------------------

/// The three rules, in the order they are drawn and stored.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
enum Who {
    Sgd,
    AdamW,
    Muon,
}

impl Who {
    const ALL: [Who; 3] = [Who::Sgd, Who::AdamW, Who::Muon];

    fn index(self) -> usize {
        self as usize
    }

    /// The step sizes tried: twelve, each twice the one before.
    fn grid(self) -> [f64; GRID] {
        let lo = match self {
            Who::Sgd => 0.0025,
            Who::AdamW => 1e-4,
            Who::Muon => 1.25e-4,
        };
        std::array::from_fn(|k| lo * 2f64.powi(k as i32))
    }

    /// Its colour: the course's colour for the rule (heavy-ball momentum, Adam, Muon).
    fn color(self) -> Color {
        match self {
            Who::Sgd => MOMENTUM,
            Who::AdamW => ADAM,
            Who::Muon => MUON,
        }
    }

    fn name(self) -> &'static str {
        match self {
            Who::Sgd => "SGD with momentum",
            Who::AdamW => "AdamW",
            Who::Muon => "Muon",
        }
    }

    /// Its name where there is little room.
    fn short(self) -> &'static str {
        match self {
            Who::Sgd => "momentum",
            Who::AdamW => "AdamW",
            Who::Muon => "Muon",
        }
    }
}

/// Step sizes tried for each rule.
const GRID: usize = 12;
/// Steps in every run.
const STEPS: usize = 2000;
/// The steps after which the loss on all the data was recorded (0: the start).
const LOSS_AT: [usize; 57] = [
    0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 16, 18, 20, 22, 25, 28, 32, 35, 40, 45, 50, 56, 63, 71, 79, 89, 100, 112, 126, 141, 158, 178, 200, 224, 251, 282, 316,
    355, 398, 447, 501, 562, 631, 708, 794, 891, 1000, 1122, 1259, 1413, 1585, 1778, 2000,
];
/// The steps after which the singular values of every weight matrix and of its last
/// step were recorded (at 0 there has been no step yet).
const SPECTRA_AT: [usize; 12] = [0, 1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000];
/// The seeds of the starting weights (and of the order of the batches).
const SEEDS: [u64; 3] = [1, 2, 3];
/// The batch sizes recorded.
const BATCHES: [usize; 2] = [8, 128];
/// The points of the data set.
const POINTS: usize = 256;
/// Its seed: the same data for every run.
#[cfg(test)]
const DATA_SEED: u64 = 11;
/// The network: 3 hidden layers of 32 tanh units.
const WIDTH: usize = 32;
const DEPTH: usize = 3;
/// The loss the batch experiment counts the steps to.
const TARGET: f64 = 0.01;

/// One weight matrix of the network.
#[derive(Clone, Copy, Debug)]
struct Matrix {
    name: &'static str,
    rows: usize,
    cols: usize,
    /// Whether Muon's run trains it with Muon (else with AdamW).
    muon: bool,
}

impl Matrix {
    /// Its number of singular values.
    const fn k(&self) -> usize {
        if self.rows < self.cols {
            self.rows
        } else {
            self.cols
        }
    }
}

/// The network's weight matrices, in order: one row per output unit, as in PyTorch.
const MATRICES: [Matrix; 4] = [
    Matrix { name: "layer 1", rows: WIDTH, cols: 2, muon: false },
    Matrix { name: "layer 2", rows: WIDTH, cols: WIDTH, muon: true },
    Matrix { name: "layer 3", rows: WIDTH, cols: WIDTH, muon: true },
    Matrix { name: "read-out", rows: 3, cols: WIDTH, muon: false },
];
/// The most singular values a matrix has.
const MOST_K: usize = WIDTH;

/// The numbers recorded at one step for one run: for each matrix, the singular values
/// of its last step, then those of its weights.
const PER_SPECTRUM: usize = 2 * (MATRICES[0].k() + MATRICES[1].k() + MATRICES[2].k() + MATRICES[3].k());

/// Where matrix m's numbers start in a spectrum record.
fn matrix_offset(m: usize) -> usize {
    MATRICES.iter().take(m).map(|x| 2 * x.k()).sum()
}

// ----- the tables' encoding ----------------------------------------------------------------

/// The 64 characters a table is written in, two to a number.
const ALPHABET: &[u8; 64] = b"0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz+/";
/// Codes per decade: a number is kept to 1.2 % (a two-hundredth of a decade).
const PER_DECADE: f64 = 200.0;
/// The smallest number kept (smaller ones, and 0, are kept as 0).
const LEAST_LOG: f64 = -12.0;
/// The code of a number that is not finite or above 10⁸ (a run that blew up).
const BLOWN: u16 = 4095;

/// The code of a number: 0 for 0 (or anything below 10⁻¹²), `BLOWN` for one that is not
/// finite or above 10⁸, else 200 (log₁₀ v + 12), rounded.
#[cfg(test)]
fn encode(v: f64) -> u16 {
    if v.is_nan() || v == f64::INFINITY || v > 1e8 {
        BLOWN
    } else if v <= 10f64.powf(LEAST_LOG) {
        0
    } else {
        (((v.log10() - LEAST_LOG) * PER_DECADE).round() as i64).clamp(1, BLOWN as i64 - 1) as u16
    }
}

/// The number a code stands for.
fn decode(c: u16) -> f64 {
    match c {
        0 => 0.0,
        BLOWN => f64::INFINITY,
        _ => 10f64.powf(c as f64 / PER_DECADE + LEAST_LOG),
    }
}

/// A table's text into its numbers (characters that are not of the alphabet are
/// skipped).
fn unpack(text: &str) -> Vec<f32> {
    let mut place = [u8::MAX; 256];
    for (i, &a) in ALPHABET.iter().enumerate() {
        place[a as usize] = i as u8;
    }
    let digits: Vec<u16> = text.bytes().map(|b| place[b as usize]).filter(|&d| d != u8::MAX).map(u16::from).collect();
    digits.chunks_exact(2).map(|p| decode(p[0] * 64 + p[1]) as f32).collect()
}

#[cfg(test)]
fn pack(values: &[f64]) -> String {
    let mut s = String::with_capacity(2 * values.len());
    for &v in values {
        let c = encode(v);
        s.push(ALPHABET[(c / 64) as usize] as char);
        s.push(ALPHABET[(c % 64) as usize] as char);
    }
    s
}

/// The recorded numbers, decoded.
struct Record {
    loss: Vec<f32>,
    spectra: Vec<f32>,
}

impl Record {
    fn loss_index(b: usize, s: usize, who: Who, k: usize) -> usize {
        (((b * SEEDS.len() + s) * 3 + who.index()) * GRID + k) * LOSS_AT.len()
    }

    fn spectrum_index(b: usize, s: usize, who: Who, c: usize, m: usize) -> usize {
        (((b * SEEDS.len() + s) * 3 + who.index()) * SPECTRA_AT.len() + c) * PER_SPECTRUM + matrix_offset(m)
    }

    /// The loss curve of batch b, seed s, `who` at its k-th step size (at `LOSS_AT`).
    fn curve(&self, b: usize, s: usize, who: Who, k: usize) -> &[f32] {
        let at = Record::loss_index(b, s, who, k);
        self.loss.get(at..at + LOSS_AT.len()).unwrap_or(&[])
    }

    /// The loss at the end of that run.
    fn final_loss(&self, b: usize, s: usize, who: Who, k: usize) -> f64 {
        self.curve(b, s, who, k).last().map_or(f64::NAN, |&v| v as f64)
    }

    /// The mean over the seeds of the final loss (infinite if any run blew up).
    fn mean_final(&self, b: usize, who: Who, k: usize) -> f64 {
        (0..SEEDS.len()).map(|s| self.final_loss(b, s, who, k)).sum::<f64>() / SEEDS.len() as f64
    }

    /// The best step size of `who` at batch b: the lowest mean final loss.
    fn best(&self, b: usize, who: Who) -> usize {
        (0..GRID).min_by(|&x, &y| self.mean_final(b, who, x).total_cmp(&self.mean_final(b, who, y))).unwrap_or(0)
    }

    /// The singular values (largest first) of matrix m's last step (`step`) or of its
    /// weights, at the c-th of `SPECTRA_AT`, in the run of `who` at its best step.
    fn spectrum(&self, b: usize, s: usize, who: Who, c: usize, m: usize, step: bool) -> &[f32] {
        let k = MATRICES.get(m).map_or(0, |x| x.k());
        let at = Record::spectrum_index(b, s, who, c, m) + if step { 0 } else { k };
        self.spectra.get(at..at + k).unwrap_or(&[])
    }

    /// The first recorded step at which the loss is at or below `target` (None if never).
    fn reaches(&self, b: usize, s: usize, who: Who, k: usize, target: f64) -> Option<usize> {
        self.curve(b, s, who, k).iter().position(|&v| (v as f64) <= target).map(|i| LOSS_AT[i])
    }

    /// The step sizes of the grid whose runs end, on average over the seeds, below
    /// `TARGET`: the first and the last (None if there are none).
    #[cfg(test)]
    fn tolerated(&self, b: usize, who: Who) -> Option<(usize, usize)> {
        let ok: Vec<usize> = (0..GRID).filter(|&k| self.mean_final(b, who, k) < TARGET).collect();
        Some((*ok.first()?, *ok.last()?))
    }
}

/// The tables, decoded once.
fn record() -> &'static Record {
    static R: std::sync::OnceLock<Record> = std::sync::OnceLock::new();
    R.get_or_init(|| Record { loss: unpack(LOSS_TABLE), spectra: unpack(SPECTRA_TABLE) })
}

/// The stable rank of a matrix from its singular values: Σσ² / σ₁² (0 for none, or all 0).
fn stable_rank(sv: &[f32]) -> f64 {
    let top = sv.iter().fold(0.0f64, |m, &v| m.max(v as f64));
    if !(top > 0.0 && top.is_finite()) {
        return 0.0;
    }
    sv.iter().map(|&v| (v as f64 / top).powi(2)).sum::<f64>()
}

/// The largest singular value over their mean (0 for none, or all 0).
fn top_over_mean(sv: &[f32]) -> f64 {
    let top = sv.iter().fold(0.0f64, |m, &v| m.max(v as f64));
    let mean = sv.iter().map(|&v| v as f64).sum::<f64>() / sv.len().max(1) as f64;
    if mean > 0.0 && top.is_finite() {
        top / mean
    } else {
        0.0
    }
}

/// The Frobenius norm, from the singular values.
fn frob(sv: &[f32]) -> f64 {
    sv.iter().map(|&v| (v as f64) * (v as f64)).sum::<f64>().sqrt()
}

// ----- numbers as the scene writes them -----------------------------------------------------

/// A loss or a ratio with two significant figures: written plainly from 0.001 to 9999,
/// as a power of ten in a formula otherwise ($7.9e-5$ is set as 7.9 × 10⁻⁵).
fn g(v: f64) -> String {
    if v == 0.0 {
        return "0".to_string();
    }
    if !v.is_finite() {
        return "blown up".to_string();
    }
    let a = v.abs();
    if (100.0..9999.5).contains(&a) {
        let unit = 10f64.powi(a.log10().floor() as i32 - 1);
        return num((v / unit).round() * unit, 0);
    }
    if (0.001..100.0).contains(&a) {
        let d = (1 - a.log10().floor() as i32).max(0) as usize;
        let s = num(v, d);
        // (0.0099 rounding up to 0.010 is written 0.01)
        return if s.contains('.') { s.trim_end_matches('0').trim_end_matches('.').to_string() } else { s };
    }
    let mut e = a.log10().floor() as i32;
    let mut m = v / 10f64.powi(e);
    if (m.abs() * 10.0).round() >= 100.0 {
        e += 1;
        m = v / 10f64.powi(e);
    }
    format!("${}e{}$", num(m, 1), e)
}

/// A step size of the grid, with the digits it has.
fn lr_text(v: f64) -> String {
    let s = format!("{v:.6}");
    let s = s.trim_end_matches('0').trim_end_matches('.');
    s.to_string()
}

/// A count of steps, or "never" in a run's length.
fn steps_text(v: Option<usize>) -> String {
    match v {
        Some(s) => count(s),
        None => "not within 2000".to_string(),
    }
}

// ----- the experiments -----------------------------------------------------------------------

/// What the picture shows.
#[derive(Clone, Copy, Debug, PartialEq)]
struct Setup {
    /// Which batch size (an index into `BATCHES`).
    batch: usize,
    /// Which seed (an index into `SEEDS`).
    seed: usize,
    /// Which weight matrix (an index into `MATRICES`).
    layer: usize,
    /// For each rule, how many grid points its step size is above (or below) its best.
    shift: [i32; 3],
    /// The step the bars and the weights are read at (1 to 2000).
    step: f64,
    /// The other batch size's curves, faint.
    other: bool,
}

impl Setup {
    fn base() -> Setup {
        Setup { batch: 0, seed: 0, layer: 1, shift: [0; 3], step: STEPS as f64, other: false }
    }

    /// Kept to what was recorded.
    fn kept(self) -> Setup {
        let step = if self.step.is_finite() { self.step.max(1.0).min(STEPS as f64) } else { STEPS as f64 };
        Setup {
            batch: self.batch.min(BATCHES.len() - 1),
            seed: self.seed.min(SEEDS.len() - 1),
            layer: self.layer.min(MATRICES.len() - 1),
            shift: self.shift.map(|s| s.clamp(-(GRID as i32), GRID as i32)),
            step,
            ..self
        }
    }

    /// The grid point of `who`'s step size shown.
    fn pick(&self, who: Who) -> usize {
        let best = record().best(self.batch, who) as i32;
        (best + self.shift[who.index()]).clamp(0, GRID as i32 - 1) as usize
    }

    /// Whether every rule is at its best step.
    fn at_best(&self) -> bool {
        Who::ALL.iter().all(|&w| self.pick(w) == record().best(self.batch, w))
    }

    /// The recorded spectra nearest at or before the step shown (at least the first step).
    fn checkpoint(&self) -> usize {
        SPECTRA_AT.iter().rposition(|&s| s as f64 <= self.step + 1e-9).unwrap_or(1).max(1)
    }
}

struct Trial {
    button: &'static str,
    caption: &'static str,
}

const TRIALS: [Trial; 5] = [
    Trial {
        button: "each rule at its best step",
        caption: r"Batches of 8, each rule at the step size that gave it the lowest final loss on average over the seeds (the ringed dots on the right): SGD with momentum 0.02, AdamW 0.0128, Muon 0.008. For about the first 400 steps all three (seed 1) stay above 0.8, near where they started (1.1). After 2000 steps, on average over the three seeds: SGD with momentum at 0.22, AdamW at 0.0085 and Muon at $7.9e-5$; in every seed Muon ended lowest, 21 to 650 times below AdamW. The faint lines are the other two seeds.",
    },
    Trial {
        button: "every step four times too long",
        caption: r"Each rule at four times its best step (two dots to the right of its ring): SGD with momentum 0.08, AdamW 0.0512, Muon 0.032. Muon's three runs still end between 0.0074 and 0.069; SGD's end between 0.87 and 2.8 and AdamW's between 0.66 and 1.4, against 1.1 at the start: they learnt little or nothing. With batches of 128 too: four times the best step leaves SGD between 6.9 and 79, AdamW between 1 and 1.1, and Muon between $6.0e-7$ and $5.2e-6$. On the right-hand chart, the step sizes whose runs end below 0.01 on average span a factor of 4 for Muon, one step size for AdamW and none for SGD at batches of 8; at batches of 128 a factor of 16 for SGD, 64 for AdamW, and for Muon every step size from the smallest tried to 0.128, a factor of 1024.",
    },
    Trial {
        button: "the step's shape, early and late",
        caption: r"Layer 2's step, played from step 1 to 2000 (batches of 8, seed 1). The steps of SGD with momentum and of AdamW stay close to rank one throughout: one to three singular values stand above the rest (at half the largest or more), and their stable ranks stay between 1.0 and 2.0. Muon's step has between 5 and 16 of its 32 singular values at half the largest or more, the rest smaller, with stable ranks from 3.3 to 9.7: far from flat. Five rounds of its iteration bring a singular value of the matrix it is handed (the momentum, blended with the newest gradient) to between 0.68 and 1.2 only if it is at least about 0.0015 of that matrix's size (its Frobenius norm); here the others are smaller than that (scene muon).",
    },
    Trial {
        button: "a small batch against a large one",
        caption: r"Batches of 128, with batches of 8 dashed (scene signnoise shows what mini-batch noise does to steps that are normalised). The best steps of SGD with momentum and AdamW grew, SGD with momentum's 16 times (0.02 to 0.32) and AdamW's 4 times (0.0128 to 0.0512), but Muon's not at all (0.008). The loss first fell below 0.01 by steps 178 to 251 for SGD with momentum, by steps 112 to 126 for AdamW and by steps 224 to 282 for Muon (over the three seeds); with batches of 8 it never did for SGD, did by steps 1585 and 2000 in two seeds of three for AdamW, and by steps 1122 to 1259 for Muon. So the larger batch saved Muon the fewest steps: 4.8 times fewer, against at least 15 for AdamW and at least 9.5 for SGD with momentum (a run that never got there counted as 2000). Yet Muon ended lowest, on average $8.6e-8$ against $1.0e-4$ for SGD with momentum and $1.8e-4$ for AdamW, after falling to between $7.9e-10$ and $1.1e-9$ by steps 794 to 1000 and climbing back. Its step keeps its size as the gradient shrinks (scene muon); with layers 2 and 3 held still from step 900, the loss stayed near its lowest.",
    },
    Trial {
        button: "the weights at the end",
        caption: r"Layer 2's weights after 2000 steps with batches of 128 (seed 1), against where they started (white, dashed: stable rank 9.1). Under every rule the largest singular value grew most: from 1.9 to 6.8 under SGD with momentum, 9.3 under AdamW and 14 under Muon. But Muon raised nearly all of them: its middle one grew from 0.77 to 3.1, while SGD's ended at 0.89 and AdamW's fell to 0.49. Over the three seeds Muon raised 30 to 32 of the 32 singular values, AdamW 5 to 7. So Muon's weights end the flattest: stable rank 5.9, against 3.3 and 2.9. In all three seeds and at both batch sizes, Muon's layer 2 ended with the highest stable rank: 4.3 to 6.7, against 2.7 to 3.8 for the other two. In layer 3 the same holds at batches of 8, but at batches of 128 the three end close together (3.3 to 4.5).",
    },
];

/// The scene opens as the first experiment sets it up.
const OPENING: usize = 0;

/// The setup of experiment k, and whether its film plays.
fn trial(k: usize) -> (Setup, bool) {
    let base = Setup::base();
    match k {
        1 => (Setup { shift: [2; 3], ..base }, false),
        2 => (Setup { step: 1.0, ..base }, true),
        3 => (Setup { batch: 1, other: true, ..base }, false),
        4 => (Setup { batch: 1, ..base }, false),
        _ => (base, false),
    }
}

// ----- the film -------------------------------------------------------------------------------

/// The run played through on a logarithmic clock: step 2000^(clock / length), so that
/// each decade of steps takes the same time.
#[derive(Clone, Copy, Debug, Default)]
struct Film {
    clock: f64,
    playing: bool,
}

impl Film {
    /// Seconds from step 1 to step 2000.
    const LENGTH: f64 = 8.0;

    fn step_at(clock: f64) -> f64 {
        (STEPS as f64).powf((clock / Film::LENGTH).clamp(0.0, 1.0))
    }

    fn clock_of(step: f64) -> f64 {
        Film::LENGTH * step.max(1.0).ln() / (STEPS as f64).ln()
    }
}

// ----- the scene -----------------------------------------------------------------------------

pub struct MuonAdam {
    setup: Setup,
    film: Film,
    look: Look,
    /// The window last drawn in (a new one is drawn at once, without gliding).
    canvas: Option<Vec2>,
    /// Where the charts' labels were put, and whether one is gliding to a new place.
    placed: Vec<Placed>,
    labels_moving: bool,
    /// The step is being dragged on the loss chart.
    dragging: bool,
}

/// What is drawn, gliding to where it is going.
#[derive(Clone, Debug, Default)]
struct Look {
    /// The ranges up the charts (log₁₀ of the loss chart's bottom and top, of the
    /// step-size chart's, of the weights chart's, of the chart of the steps beside the
    /// weights).
    ranges: Glide,
    /// The loss curves at each of `LOSS_AT`, log₁₀: the three rules, the other seeds'
    /// (two each), the other batch size's.
    curves: Glide,
    /// The final losses, log₁₀: each rule's mean at each step size, then each seed's.
    finals: Glide,
    /// The step's singular values over its largest, each rule's, padded with 0.
    bars: Glide,
    /// The weights' singular values, log₁₀: each rule's, then the start's.
    weights: Glide,
    /// The step beside the weights, log₁₀ of ‖ΔW‖ / ‖W‖, for each rule and matrix.
    rel: Glide,
    /// Where the ranges are going (the axes' ticks are chosen from these, so that their
    /// number does not change while a range glides).
    goal: Vec<f32>,
    moving: bool,
}

/// What a value that is not a number glides as: off the top of every chart.
const ABOVE: f32 = 12.0;
/// And a value of 0, on a logarithmic chart: off the bottom.
const BELOW: f32 = -14.0;

fn log_or(v: f64) -> f32 {
    if v.is_nan() || v == f64::INFINITY {
        ABOVE
    } else if v <= 0.0 {
        BELOW
    } else {
        (v.log10() as f32).max(BELOW).min(ABOVE)
    }
}

impl MuonAdam {
    /// (Nothing here needs the graphics device, so the tests can make one too.)
    fn create() -> MuonAdam {
        let mut s = MuonAdam { setup: Setup::base(), film: Film::default(), look: Look::default(), canvas: None, placed: Vec::new(), labels_moving: false, dragging: false };
        s.set_up(OPENING);
        s.follow(0.0, true);
        s
    }

    /// Set up experiment k.
    fn set_up(&mut self, k: usize) {
        let (setup, film) = trial(k);
        self.setup = setup.kept();
        self.film = Film { clock: Film::clock_of(self.setup.step), playing: film };
    }

    /// Play the run from the first step (or pause it).
    fn play(&mut self) {
        if self.film.playing {
            self.film.playing = false;
        } else {
            if self.setup.step >= STEPS as f64 - 1e-9 {
                self.setup.step = 1.0;
            }
            self.film = Film { clock: Film::clock_of(self.setup.step), playing: true };
        }
    }

    /// Move the film on by `dt` seconds of the scene's time.
    fn advance(&mut self, dt: f32) {
        if !self.film.playing {
            return;
        }
        self.film.clock += dt.max(0.0) as f64;
        self.setup.step = Film::step_at(self.film.clock);
        if self.film.clock >= Film::LENGTH {
            self.film.playing = false;
            self.setup.step = STEPS as f64;
        }
    }

    // ----- what the pictures are to show ---------------------------------------------------

    /// The loss curves, log₁₀ at each of `LOSS_AT`: the three rules', the other seeds',
    /// the other batch size's (the same seed, each rule at its best for that batch).
    fn curves_goal(&self) -> Vec<f32> {
        let r = record();
        let s = &self.setup;
        let n = LOSS_AT.len();
        let mut v = Vec::with_capacity(12 * n);
        let mut push = |c: &[f32]| v.extend((0..n).map(|i| log_or(c.get(i).copied().unwrap_or(f32::NAN) as f64)));
        for w in Who::ALL {
            push(r.curve(s.batch, s.seed, w, s.pick(w)));
        }
        for w in Who::ALL {
            for seed in (0..SEEDS.len()).filter(|&x| x != s.seed) {
                push(r.curve(s.batch, seed, w, s.pick(w)));
            }
        }
        let ob = 1 - s.batch;
        for w in Who::ALL {
            push(r.curve(ob, s.seed, w, r.best(ob, w)));
        }
        v
    }

    /// The final losses, log₁₀: each rule's mean over the seeds at each step size, then
    /// each seed's.
    fn finals_goal(&self) -> Vec<f32> {
        let r = record();
        let b = self.setup.batch;
        let mut v = Vec::with_capacity(4 * 3 * GRID);
        for w in Who::ALL {
            v.extend((0..GRID).map(|k| log_or(r.mean_final(b, w, k))));
        }
        for w in Who::ALL {
            for s in 0..SEEDS.len() {
                v.extend((0..GRID).map(|k| log_or(r.final_loss(b, s, w, k))));
            }
        }
        v
    }

    /// The bars: each rule's step's singular values over its largest, padded with 0.
    fn bars_goal(&self) -> Vec<f32> {
        let r = record();
        let s = &self.setup;
        let c = s.checkpoint();
        let mut v = Vec::with_capacity(3 * MOST_K);
        for w in Who::ALL {
            let sv = r.spectrum(s.batch, s.seed, w, c, s.layer, true);
            let top = sv.first().copied().unwrap_or(0.0);
            v.extend((0..MOST_K).map(|i| match sv.get(i) {
                Some(&x) if top > 0.0 => x / top,
                _ => 0.0,
            }));
        }
        v
    }

    /// The weights' singular values, log₁₀: each rule's, then the start's.
    fn weights_goal(&self) -> Vec<f32> {
        let r = record();
        let s = &self.setup;
        let c = s.checkpoint();
        let mut v = Vec::with_capacity(4 * MOST_K);
        let mut push = |sv: &[f32]| v.extend((0..MOST_K).map(|i| sv.get(i).map_or(BELOW, |&x| log_or(x as f64))));
        for w in Who::ALL {
            push(r.spectrum(s.batch, s.seed, w, c, s.layer, false));
        }
        push(r.spectrum(s.batch, s.seed, Who::Sgd, 0, s.layer, false));
        v
    }

    /// The step beside the weights, log₁₀ ‖ΔW‖_F / ‖W‖_F, each rule and matrix.
    fn rel_goal(&self) -> Vec<f32> {
        let r = record();
        let s = &self.setup;
        let c = s.checkpoint();
        let mut v = Vec::with_capacity(3 * MATRICES.len());
        for w in Who::ALL {
            for m in 0..MATRICES.len() {
                let (d, wt) = (frob(r.spectrum(s.batch, s.seed, w, c, m, true)), frob(r.spectrum(s.batch, s.seed, w, c, m, false)));
                v.push(if wt > 0.0 { log_or(d / wt) } else { BELOW });
            }
        }
        v
    }

    /// The charts' ranges up: from below the lowest value shown to above the highest
    /// (each chart's own, log₁₀), from goals already worked out.
    fn ranges_goal(&self, curves: &[f32], finals: &[f32], weights: &[f32], rel: &[f32]) -> Vec<f32> {
        let span = |vals: &mut dyn Iterator<Item = f32>, cap: f32| {
            let (mut lo, mut hi) = (f32::INFINITY, f32::NEG_INFINITY);
            for v in vals.filter(|v| *v > BELOW + 0.5 && *v < ABOVE - 0.5) {
                lo = lo.min(v);
                hi = hi.max(v.min(cap));
            }
            if !(lo.is_finite() && hi.is_finite()) {
                (-1.0, 0.5)
            } else {
                (lo, hi)
            }
        };
        let n = LOSS_AT.len();
        let shown = if self.setup.other { 12 * n } else { 9 * n };
        let start = log_or(record().curve(self.setup.batch, self.setup.seed, Who::Sgd, 0).first().copied().unwrap_or(1.0) as f64);
        let (lo, hi) = span(&mut curves.iter().take(shown).copied().chain(std::iter::once(start)), 2.0);
        let (flo, fhi) = span(&mut finals.iter().copied().chain(std::iter::once(start)), 2.0);
        let (wlo, whi) = span(&mut weights.iter().copied(), 4.0);
        let (rlo, rhi) = span(&mut rel.iter().copied(), 1.0);
        vec![lo - 0.35, (hi + 0.3).max(lo + 1.0), flo - 0.45, (fhi + 0.45).max(flo + 1.0), wlo - 0.15, (whi + 0.15).max(wlo + 0.5), rlo - 0.3, (rhi + 0.3).max(rlo + 1.0)]
    }

    /// Bring the picture `dt` seconds nearer to what it shows. (`snap`: there at once.)
    fn follow(&mut self, dt: f32, snap: bool) {
        let (curves, finals, bars, weights, rel) = (self.curves_goal(), self.finals_goal(), self.bars_goal(), self.weights_goal(), self.rel_goal());
        let ranges = self.ranges_goal(&curves, &finals, &weights, &rel);
        let l = &mut self.look;
        l.goal.clone_from(&ranges);
        let mut moving = false;
        for (g, to) in [(&mut l.ranges, &ranges), (&mut l.curves, &curves), (&mut l.finals, &finals), (&mut l.bars, &bars), (&mut l.weights, &weights), (&mut l.rel, &rel)] {
            if snap {
                g.snap(to);
            } else {
                moving |= g.follow(to, dt, false);
            }
        }
        l.moving = moving;
    }
}

// ----- the layout ------------------------------------------------------------------------

/// Where everything goes in a window (logical pixels from its top-left corner). Each
/// chart's rectangle is where its data goes; its title, ticks and names are round it.
#[derive(Clone, Copy, Debug, PartialEq)]
struct Layout {
    view: Vec2,
    narrow: bool,
    text: f32,
    small: f32,
    /// The title's place (its left end, on its baseline).
    head: Vec2,
    /// The buttons: batch 8, batch 128, seeds 1 to 3, best steps, four times, play.
    buttons: [Rect; 8],
    /// The network's boxes: the inputs, the four matrices, the outputs.
    boxes: [Rect; 6],
    /// The chart of each layer's step beside its weights (under the boxes).
    rel: Rect,
    loss: Rect,
    lr: Rect,
    /// The three charts of the step's singular values, side by side.
    bars: [Rect; 3],
    weights: Rect,
    /// Only the loss chart: the window is too small for the rest.
    compact: bool,
}

/// Room left round a chart for its title and axes: left, above, right, below (less on a
/// narrow window).
const ROOM: [f32; 4] = [46.0, 38.0, 10.0, 30.0];
const ROOM_NARROW: [f32; 4] = [40.0, 32.0, 8.0, 24.0];

fn room(narrow: bool) -> [f32; 4] {
    if narrow {
        ROOM_NARROW
    } else {
        ROOM
    }
}

impl Layout {
    fn of(view: Vec2, inset: f32) -> Layout {
        let view = view.max(Vec2::splat(1.0));
        let narrow = view.x < 700.0;
        let (text, small) = if narrow { (12.0, 10.5) } else { (13.5, 11.5) };
        let side = if narrow { 8.0 } else { 16.0 };
        let top = inset.max(0.0) + if narrow { 6.0 } else { 8.0 };
        let r = |x0: f32, y0: f32, x1: f32, y1: f32| (Vec2::new(x0, y0), Vec2::new(x1.max(x0 + 1.0), y1.max(y0 + 1.0)));
        let none = r(0.0, 0.0, 0.0, 0.0);
        let bh = if narrow { 24.0 } else { 26.0 };
        let mut lay = Layout {
            view,
            narrow,
            text,
            small,
            head: Vec2::new(side, top + text + 4.0),
            buttons: [none; 8],
            boxes: [none; 6],
            rel: none,
            loss: none,
            lr: none,
            bars: [none; 3],
            weights: none,
            compact: false,
        };
        // the buttons: beside the title on a wide window, two rows under it on a narrow one
        let rows_y = if narrow {
            let y = top + text + 12.0;
            let a = plot::even_row(4, side, view.x - side, y, bh, 5.0);
            let b = plot::even_row(4, side, view.x - side, y + bh + 5.0, bh, 5.0);
            lay.buttons[..4].copy_from_slice(&a[..4]);
            lay.buttons[4..].copy_from_slice(&b[..4]);
            y + 2.0 * bh + 5.0
        } else {
            let x0 = (view.x * 0.38).max(side + 340.0).min(view.x - side - 8.0 * 60.0);
            let row = plot::even_row(8, x0, view.x - side, top, bh, 5.0);
            for (k, b) in row.into_iter().enumerate().take(8) {
                lay.buttons[k] = b;
            }
            top + bh
        };
        let compact = view.x < 300.0 || view.y < 420.0;
        if compact {
            lay.compact = true;
            let y0 = rows_y + 8.0;
            let room = room(narrow);
            lay.loss = r(side + room[0], y0 + room[1], view.x - side - room[2], view.y - room[3]);
            return lay;
        }
        // the network: six boxes across, and under the four matrices the chart of their steps
        let box_h = if narrow { 30.0 } else { 40.0 };
        let y0 = rows_y + if narrow { 10.0 } else { 14.0 };
        let gap = if narrow { 8.0 } else { 26.0 };
        let row = plot::even_row(6, side, view.x - side, y0, box_h, gap);
        for (k, b) in row.into_iter().enumerate().take(6) {
            lay.boxes[k] = b;
        }
        let rel_top = y0 + box_h + if narrow { 32.0 } else { 40.0 };
        // (a short phone has no room for the chart of the steps beside the weights)
        let rel_h = if narrow && view.y < 700.0 {
            0.0
        } else if narrow {
            40.0
        } else {
            66.0
        };
        let charts_top = if rel_h > 0.0 {
            lay.rel = r(lay.boxes[1].0.x, rel_top, lay.boxes[4].1.x, rel_top + rel_h);
            rel_top + rel_h + if narrow { 18.0 } else { 22.0 }
        } else {
            y0 + box_h + 20.0
        };
        let room = room(narrow);
        let in_cell = |(a, b): Rect| r(a.x + room[0], a.y + room[1], b.x - room[2], b.y - room[3]);
        let bars_in = |lay: &mut Layout, (a, b): Rect| {
            // three panels side by side, each with room for its title and a little axis
            let inner = r(a.x + 22.0, a.y + room[1] + 18.0, b.x - room[2], b.y - 12.0);
            let row = plot::even_row(3, inner.0.x, inner.1.x, inner.0.y, inner.1.y - inner.0.y, if narrow { 22.0 } else { 30.0 });
            for (k, c) in row.into_iter().enumerate().take(3) {
                lay.bars[k] = c;
            }
        };
        if narrow {
            // one chart under another
            let h = ((view.y - charts_top - 4.0) / 4.0).max(40.0);
            let cell = |k: usize| r(side, charts_top + k as f32 * h, view.x - side, charts_top + (k + 1) as f32 * h);
            lay.loss = in_cell(cell(0));
            lay.lr = in_cell(cell(1));
            bars_in(&mut lay, cell(2));
            lay.weights = in_cell(cell(3));
        } else {
            let mid = side + (view.x - 2.0 * side) * 0.56;
            let h = ((view.y - charts_top - 6.0) / 2.0).max(60.0);
            let (y1, y2) = (charts_top + h, charts_top + 2.0 * h);
            lay.loss = in_cell(r(side, charts_top, mid - 6.0, y1));
            lay.lr = in_cell(r(mid + 6.0, charts_top, view.x - side, y1));
            bars_in(&mut lay, r(side, y1, mid - 6.0, y2));
            lay.weights = in_cell(r(mid + 6.0, y1, view.x - side, y2));
        }
        lay
    }
}

// ----- drawing ---------------------------------------------------------------------------------

// Lines whose number of points can change, each in a batch of its own (see `plot`).
const LOSS: u8 = OWN;
/// (three rules, six other seeds, three of the other batch size)
const FINALS: u8 = LOSS + 12;
const WEIGHTS: u8 = FINALS + 3;
const REL: u8 = WEIGHTS + 4;
const NOW: u8 = REL + 3;

/// The point batches, each its own (dots of the step-size chart, of the weights, of the
/// steps beside the weights, the marks).
fn dot_batch<'b>(f: &'b mut Frame<'_>, k: usize) -> &'b mut crate::gfx::draw::PointBatch {
    f.dl.points_with(PointKind::Overlay, PointStyle { outline_px: 0.0, size_scale: 1.0 + 1e-4 * (k + 1) as f32, ..Default::default() })
}

fn dots_px(f: &mut Frame, k: usize, pts: &[(Vec2, f32, Color)]) {
    let at: Vec<(Vec3, f32, Color)> = pts.iter().filter(|p| p.0.is_finite()).map(|(p, s, c)| (f.hud(*p), *s, *c)).collect();
    let b = dot_batch(f, k);
    for (p, s, c) in at {
        b.push(p, s, c);
    }
}

/// Every batch the scene draws in, made in the same order in every frame, so that one
/// that is empty in a frame does not move the later ones up a place (see `eidos motion`).
fn hold_batches(f: &mut Frame) {
    let corner = f.hud(Vec2::ZERO);
    for kind in [AXES, plot::CURVES, MARKS, plot::BUTTONS].into_iter().chain(LOSS..=NOW) {
        plot::apart(f, kind, false).segment(corner, corner, 1.0, Color::CLEAR);
    }
    for kind in [MARKS, LOSS + 9, LOSS + 10, LOSS + 11, WEIGHTS + 3] {
        plot::apart(f, kind, true).segment(corner, corner, 1.0, Color::CLEAR);
    }
    for k in 0..5 {
        dot_batch(f, k);
    }
}

/// log₁₀ ticks: the whole decades in a range (every `every`-th named), and where there
/// are few, 2 and 5 of each.
fn log_ticks(lo: f64, hi: f64, every: i32) -> Vec<(f64, bool)> {
    let mut v = Vec::new();
    if !(lo.is_finite() && hi.is_finite() && hi > lo) {
        return v;
    }
    let few = hi - lo < 1.6;
    let mut d = lo.floor() as i32;
    while (d as f64) <= hi + 1e-9 && v.len() < 80 {
        for (m, main) in [(1.0f64, true), (2.0, false), (5.0, false)] {
            let x = d as f64 + m.log10();
            let named = (main && d.rem_euclid(every.max(1)) == 0) || few;
            if x >= lo - 1e-9 && x <= hi + 1e-9 && (main || few) {
                v.push((x, named));
            }
        }
        d += 1;
    }
    v
}

/// A power of ten as an axis writes it.
fn power_name(x: f64) -> String {
    let d = x.round() as i32;
    if (x - d as f64).abs() < 1e-6 {
        return match d {
            0 => "1".to_string(),
            1 => "10".to_string(),
            2 => "100".to_string(),
            3 => "1000".to_string(),
            -1 => "0.1".to_string(),
            -2 => "0.01".to_string(),
            -3 => "0.001".to_string(),
            _ => format!("$10^{{{d}}}$"),
        };
    }
    g(10f64.powf(x))
}

/// Ticks up a logarithmic axis (chosen from where its range is going), and faint lines
/// across.
fn log_axis_y(f: &mut Frame, p: &Plot, size: f32, goal: (f64, f64)) {
    let (min, max) = p.rect;
    plot::line_of(f, AXES, min, Vec2::new(min.x, max.y), 1.0, pal::DIM.a(0.8));
    let per = p.height() / ((goal.1 - goal.0) as f32).max(0.1);
    let every = if per > 16.0 {
        1
    } else if per > 8.0 {
        2
    } else {
        3
    };
    for (y, named) in log_ticks(goal.0, goal.1, every) {
        if y < p.y[0] - 1e-6 || y > p.y[1] + 1e-6 || !named {
            continue;
        }
        let at = p.px_y(y);
        plot::line_of(f, AXES, Vec2::new(min.x - 4.0, at), Vec2::new(min.x, at), 1.0, pal::DIM.a(0.8));
        plot::line_of(f, AXES, Vec2::new(min.x, at), Vec2::new(max.x, at), 1.0, pal::DIM.a(0.1));
        f.text_px(Vec2::new(min.x - 6.0, at), power_name(y), size, pal::DIM, Align2::RIGHT_CENTER);
    }
}

/// Ticks along a logarithmic axis.
fn log_axis_x(f: &mut Frame, p: &Plot, size: f32) {
    let (min, max) = p.rect;
    plot::line_of(f, AXES, Vec2::new(min.x, max.y), max, 1.0, pal::DIM.a(0.8));
    let every = if p.width() / ((p.x[1] - p.x[0]) as f32).max(0.1) > 60.0 { 1 } else { 2 };
    for (x, named) in log_ticks(p.x[0], p.x[1], every) {
        if x < p.x[0] - 1e-6 || x > p.x[1] + 1e-6 {
            continue;
        }
        let at = p.px_x(x);
        plot::line_of(f, AXES, Vec2::new(at, max.y), Vec2::new(at, max.y + if named { 4.0 } else { 2.0 }), 1.0, pal::DIM.a(0.8));
        if named {
            f.text_px(Vec2::new(at, max.y + 6.0), power_name(x), size, pal::DIM, Align2::CENTER_TOP);
        }
    }
}

/// What the two charts of singular values say when other step sizes are shown: only the
/// runs at the best step sizes were recorded in full.
const BEST_ONLY: &str = "each rule at its best step size";

/// A chart's backing, its title above it and the name of its y axis.
fn chart_frame(f: &mut Frame, lay: &Layout, p: &Plot, title: &str, name: &str) {
    let rm = room(lay.narrow);
    let (min, _) = p.rect;
    p.backing(f, rm[0] - 4.0, rm[3] - 2.0, rm[1] - 4.0, rm[2] - 2.0);
    let left = min.x - rm[0] + 6.0;
    f.text_px(Vec2::new(left, min.y - rm[1] + 4.0 + lay.text), title, lay.text, pal::TEXT, Align2::LEFT_BOTTOM);
    f.text_px(Vec2::new(left, min.y - 5.0), name, lay.small, pal::DIM, Align2::LEFT_BOTTOM);
}

/// The x of the loss chart: the step.
const STEP_RANGE: [f64; 2] = [0.0, STEPS as f64];

impl MuonAdam {
    fn range(&self, i: usize) -> f64 {
        self.look.ranges.get(i) as f64
    }

    /// Where the range of chart i is going (not where it is).
    fn range_goal(&self, i: usize) -> f64 {
        self.look.goal.get(i).copied().unwrap_or(0.0) as f64
    }

    fn loss_plot(&self, lay: &Layout) -> Plot {
        Plot::new(lay.loss, STEP_RANGE, [self.range(0), self.range(1)])
    }

    /// The step-size chart: across, every rule's grid.
    fn lr_plot(&self, lay: &Layout) -> Plot {
        let lo = Who::ALL.iter().map(|w| w.grid()[0]).fold(f64::INFINITY, f64::min).log10() - 0.15;
        let hi = Who::ALL.iter().map(|w| w.grid()[GRID - 1]).fold(0.0, f64::max).log10() + 0.15;
        Plot::new(lay.lr, [lo, hi], [self.range(2), self.range(3)])
    }

    fn weights_plot(&self, lay: &Layout) -> Plot {
        let k = MATRICES[self.setup.layer].k() as f64;
        Plot::new(lay.weights, [0.5, k + 0.5], [self.range(4), self.range(5)])
    }

    /// The buttons over the picture.
    fn buttons(&mut self, f: &mut Frame, lay: &Layout) {
        let s = self.setup;
        let size = lay.small;
        let words = ["batch 8", "batch 128", "seed 1", "seed 2", "seed 3", "best steps", r"steps $\times 4$"];
        for (k, w) in words.iter().enumerate() {
            let on = match k {
                0 | 1 => s.batch == k,
                2..=4 => s.seed == k - 2,
                5 => s.shift == [0; 3],
                _ => s.shift == [2; 3],
            };
            if plot::button(f, lay.buttons[k], w, size, on) {
                match k {
                    0 | 1 => self.setup.batch = k,
                    2..=4 => self.setup.seed = k - 2,
                    5 => self.setup.shift = [0; 3],
                    _ => self.setup.shift = [2; 3],
                }
            }
        }
        let play = if self.film.playing { "❚❚ pause" } else { "▶ play" };
        if plot::button(f, lay.buttons[7], play, size, false) {
            self.play();
        }
    }

    /// Clicks on the network's matrices (to choose one) and on the step-size chart's dots
    /// (to show that run).
    fn clicks(&mut self, f: &mut Frame, lay: &Layout) {
        let Some(at) = plot::pointer(f) else { return };
        for m in 0..MATRICES.len() {
            if plot::within(at, lay.boxes[m + 1], 2.0) {
                f.handles.claim_clicks();
                if f.ptr.clicked {
                    self.setup.layer = m;
                    f.ptr.clicked = false;
                }
                return;
            }
        }
        let p = self.lr_plot(lay);
        if p.tiny() || !plot::within(at, p.rect, 8.0) {
            return;
        }
        // the nearest dot of a rule's mean, if near enough
        let mut best: Option<(f32, Who, usize)> = None;
        for w in Who::ALL {
            for k in 0..GRID {
                let y = (self.look.finals.get(w.index() * GRID + k) as f64).max(p.y[0]).min(p.y[1]);
                let d = (p.at(w.grid()[k].log10(), y) - at).length();
                if d < 12.0 && best.is_none_or(|b| d < b.0) {
                    best = Some((d, w, k));
                }
            }
        }
        if let Some((_, w, k)) = best {
            f.handles.claim_clicks();
            if f.ptr.clicked {
                self.setup.shift[w.index()] = k as i32 - record().best(self.setup.batch, w) as i32;
                f.ptr.clicked = false;
            }
        }
    }

    /// The step, dragged along the loss chart.
    fn drag(&mut self, f: &mut Frame, lay: &Layout) {
        let p = self.loss_plot(lay);
        if p.tiny() {
            return;
        }
        let at = Vec2::new(p.px_x(self.setup.step), p.rect.1.y);
        if let Some(to) = plot::screen_handle(f, 1, at, pal::WHITE) {
            let step = p.x_at(to.x).clamp(1.0, STEP_RANGE[1]);
            if step.is_finite() {
                self.setup.step = step;
                self.film.playing = false;
                self.dragging = true;
            }
        }
    }

    fn draw_head(&self, f: &mut Frame, lay: &Layout) {
        let s = &self.setup;
        let title = if lay.narrow {
            format!("Three rules, one network · batch {} · seed {}", BATCHES[s.batch], s.seed + 1)
        } else {
            format!("Three rules train one network: batches of {}, seed {}", BATCHES[s.batch], s.seed + 1)
        };
        f.text_px(lay.head, title, lay.text, pal::TEXT, Align2::LEFT_BOTTOM);
    }

    /// The network: its boxes, the rule each matrix takes in Muon's run, and under them
    /// each matrix's step beside its weights.
    fn draw_network(&mut self, f: &mut Frame, lay: &Layout) {
        let s = self.setup;
        let size = lay.small;
        let names: [(&str, String); 6] = [
            ("data", if lay.narrow { "2 numbers".into() } else { "2 numbers a point".into() }),
            ("layer 1", format!(r"${WIDTH} \times 2$")),
            ("layer 2", format!(r"${WIDTH} \times {WIDTH}$")),
            ("layer 3", format!(r"${WIDTH} \times {WIDTH}$")),
            ("read-out", format!(r"$3 \times {WIDTH}$")),
            ("3 classes", if lay.narrow { "softmax".into() } else { "softmax, loss".into() }),
        ];
        for (k, (a, b)) in lay.boxes.iter().enumerate() {
            let matrix = (1..=4).contains(&k);
            let chosen = matrix && s.layer == k - 1;
            let over = plot::pointer(f).is_some_and(|p| matrix && plot::within(p, (*a, *b), 2.0));
            let fill = if chosen {
                Color::hex(0x2B3D66)
            } else if over {
                Color::hex(0x2A3350)
            } else if matrix {
                Color::hex(0x161C2B)
            } else {
                Color::hex(0x10141E)
            };
            hud_rect(f, *a, *b, fill);
            let edge = [*a, Vec2::new(b.x, a.y), *b, Vec2::new(a.x, b.y), *a];
            let line = if matrix { pal::LINK.a(if chosen { 0.95 } else { 0.45 }) } else { pal::DIM.a(0.4) };
            plot::path_of(f, plot::BUTTONS, &edge, if chosen { 1.6 } else { 1.0 }, line);
            let mid = (*a + *b) * 0.5;
            let (up, down, big) = if lay.narrow { (6.0, 7.0, size - 0.5) } else { (8.0, 9.0, size + 0.5) };
            f.text_px(Vec2::new(mid.x, mid.y - up), names[k].0, big, pal::TEXT, Align2::CENTER_CENTER);
            f.text_px(Vec2::new(mid.x, mid.y + down), names[k].1.clone(), big - 1.5, pal::DIM, Align2::CENTER_CENTER);
            if k + 1 < lay.boxes.len() {
                let (from, to) = (Vec2::new(b.x + 2.0, mid.y), Vec2::new(lay.boxes[k + 1].0.x - 2.0, mid.y));
                if to.x - from.x > 6.0 {
                    hud_arrow(f, from, to, 1.2, pal::DIM.a(0.7));
                }
            }
            // which rule Muon's run gives it
            if matrix {
                let m = MATRICES[k - 1];
                let (w, c) = if m.muon { ("Muon's run: Muon", MUON) } else { ("Muon's run: AdamW", ADAM) };
                let w = if lay.narrow {
                    if m.muon {
                        "Muon"
                    } else {
                        "AdamW"
                    }
                } else {
                    w
                };
                f.text_px(Vec2::new(mid.x, b.y + 3.0), w, size - 1.0, c.a(0.9), Align2::CENTER_TOP);
            }
        }
        // the step beside the weights
        let p = Plot::new(lay.rel, [0.5, 4.5], [self.range(6), self.range(7)]);
        if p.tiny() {
            return;
        }
        let (min, _) = p.rect;
        p.backing(f, 46.0, 6.0, 22.0, 8.0);
        let title = if lay.narrow {
            format!(r"each layer's step over its weights, $\|\Delta W\| / \|W\|$, step {}", count(SPECTRA_AT[s.checkpoint()]))
        } else {
            format!(r"each layer's step beside its weights, $\|\Delta W\| / \|W\|$ (Frobenius norms), at step {}", count(SPECTRA_AT[s.checkpoint()]))
        };
        f.text_px(Vec2::new(min.x - 40.0, min.y - 7.0), title, size, pal::TEXT, Align2::LEFT_BOTTOM);
        let goal = (self.range_goal(6), self.range_goal(7));
        log_axis_y(f, &p, size - 1.0, goal);
        let mut dots = Vec::new();
        for w in Who::ALL {
            let pts: Vec<Vec2> = (0..MATRICES.len())
                .map(|m| {
                    // (under each box's middle)
                    let (a, b) = lay.boxes[m + 1];
                    let x = (a.x + b.x) * 0.5 + (w.index() as f32 - 1.0) * 3.0;
                    let y = (self.look.rel.get(w.index() * MATRICES.len() + m) as f64).max(p.y[0]).min(p.y[1]);
                    Vec2::new(x, p.px_y(y))
                })
                .collect();
            plot::path_of(f, REL + w.index() as u8, &pts, 1.4, w.color().a(0.7));
            dots.extend(pts.into_iter().map(|q| (q, if lay.narrow { 4.5 } else { 6.0 }, w.color())));
        }
        dots_px(f, 2, &dots);
    }

    /// The loss against the step.
    fn draw_loss(&mut self, f: &mut Frame, lay: &Layout, labels: &mut Vec<ChartLabel>) {
        let p = self.loss_plot(lay);
        if p.tiny() {
            return;
        }
        let s = self.setup;
        let r = record();
        let size = lay.small;
        let (min, max) = p.rect;
        let title = if lay.narrow {
            "loss on all the points, against the step".to_string()
        } else if s.at_best() {
            format!("the loss on all {POINTS} points against the step, each rule at its best step size")
        } else {
            format!("the loss on all {POINTS} points against the step, at the step sizes chosen")
        };
        chart_frame(f, lay, &p, &title, "loss");
        let goal = (self.range_goal(0), self.range_goal(1));
        log_axis_y(f, &p, size, goal);
        plot::line_of(f, AXES, Vec2::new(min.x, max.y), max, 1.0, pal::DIM.a(0.8));
        let every = if p.width() > 300.0 { 500 } else { 1000 };
        for t in (0..=STEPS).step_by(every) {
            let x = p.px_x(t as f64);
            plot::line_of(f, AXES, Vec2::new(x, max.y), Vec2::new(x, max.y + 4.0), 1.0, pal::DIM.a(0.8));
            f.text_px(Vec2::new(x, max.y + 6.0), count(t), size, pal::DIM, Align2::CENTER_TOP);
        }
        if !lay.narrow {
            f.text_px(Vec2::new(max.x, max.y + 8.0 + size), "step", size, pal::DIM, Align2::RIGHT_TOP);
        }
        // the target and the start
        let target = TARGET.log10();
        if target > p.y[0] && target < p.y[1] {
            plot::dashed_of(f, MARKS, Vec2::new(min.x, p.px_y(target)), Vec2::new(max.x, p.px_y(target)), 1.0, pal::DIM.a(0.7));
        }
        let start = r.curve(s.batch, s.seed, Who::Sgd, 0).first().copied().unwrap_or(1.0) as f64;
        let ys = start.log10();
        if ys > p.y[0] && ys < p.y[1] {
            f.text_px(Vec2::new(min.x + 4.0, p.px_y(ys) - 3.0), format!("start {}", g(start)), size - 1.0, pal::DIM, Align2::LEFT_BOTTOM);
        }
        let n = LOSS_AT.len();
        let xs: Vec<f64> = LOSS_AT.iter().map(|&t| t as f64).collect();
        let curve = |k: usize| -> Vec<(f64, f64)> { (0..n).map(|i| (xs[i], (self.look.curves.get(k * n + i) as f64).min(p.y[1] + 0.5))).collect() };
        // the other batch size's, dashed; the other seeds', thin and faint; this one's
        if s.other {
            for w in Who::ALL {
                let pts: Vec<Vec2> = plot::clip(&curve(9 + w.index()).iter().map(|&(x, y)| p.at(x, y)).collect::<Vec<_>>(), p.rect);
                for piece in pts.split(|q| !q.is_finite()).filter(|piece| piece.len() >= 2) {
                    let points: Vec<Vec3> = piece.iter().map(|q| f.hud(*q)).collect();
                    plot::apart(f, LOSS + 9 + w.index() as u8, true).polyline(points, 1.3, w.color().a(0.6));
                }
            }
        }
        for w in Who::ALL {
            for j in 0..2 {
                p.path(f, LOSS + 3 + (2 * w.index() + j) as u8, curve(3 + 2 * w.index() + j), 1.0, w.color().a(0.28));
            }
        }
        // (the curves' names, the lowest placed first: the others then make room above it)
        let mut ends = Vec::with_capacity(3);
        for w in Who::ALL {
            let pts = curve(w.index());
            p.path(f, LOSS + w.index() as u8, pts.clone(), 2.2, w.color());
            // its name at its end, or where it left the chart
            let k = s.pick(w);
            let c = r.curve(s.batch, s.seed, w, k);
            let blown = c.iter().position(|v| !v.is_finite()).map(|i| LOSS_AT[i]);
            let last = pts.iter().rposition(|q| q.1 <= p.y[1] && q.1 >= p.y[0]).unwrap_or(n - 1);
            let at = p.at(pts[last].0, pts[last].1.max(p.y[0]).min(p.y[1]));
            let mut text = format!("{} {}", if lay.narrow { w.short() } else { w.name() }, lr_text(w.grid()[k]));
            if let Some(b) = blown {
                text += &format!(": blew up by step {}", count(b));
            }
            // (or, if there is no room there, two thirds of the way along)
            let i = LOSS_AT.iter().position(|&t| t >= 1300).unwrap_or(n - 1);
            let mid = p.at(pts[i].0, pts[i].1.max(p.y[0]).min(p.y[1]));
            ends.push(ChartLabel {
                at: Vec2::new(at.x.min(max.x - 2.0).max(min.x + 60.0), at.y - 3.0),
                align: Align2::RIGHT_BOTTOM,
                text,
                color: w.color(),
                rect: p.rect,
                alt: Some(Vec2::new(mid.x, mid.y - 3.0)),
            });
        }
        ends.sort_by(|a, b| b.at.y.total_cmp(&a.at.y));
        labels.extend(ends);
        if s.other && !lay.narrow {
            f.text_px(Vec2::new(min.x + 4.0, max.y - 4.0), format!("dashed: batches of {}, each at its own best step", BATCHES[1 - s.batch]), size - 1.0, pal::DIM, Align2::LEFT_BOTTOM);
        }
        // the step the bars are read at
        let x = p.px_x(s.step);
        plot::line_of(f, NOW, Vec2::new(x, min.y), Vec2::new(x, max.y), 1.0, pal::TEXT.a(0.45));
    }

    /// The final loss against the step size.
    fn draw_lr(&mut self, f: &mut Frame, lay: &Layout, labels: &mut Vec<ChartLabel>) {
        let p = self.lr_plot(lay);
        if p.tiny() {
            return;
        }
        let s = self.setup;
        let r = record();
        let size = lay.small;
        let (min, max) = p.rect;
        let title = if lay.narrow { "final loss against the step size" } else { "the loss after 2000 steps against the step size: mean of 3 seeds, and each" };
        chart_frame(f, lay, &p, title, "final loss");
        let goal = (self.range_goal(2), self.range_goal(3));
        log_axis_y(f, &p, size, goal);
        log_axis_x(f, &p, size);
        if !lay.narrow {
            f.text_px(Vec2::new(max.x, max.y + 8.0 + size), "step size", size, pal::DIM, Align2::RIGHT_TOP);
        }
        let start = r.curve(s.batch, s.seed, Who::Sgd, 0).first().copied().unwrap_or(1.0) as f64;
        let ys = start.log10();
        if ys > p.y[0] && ys < p.y[1] {
            plot::dashed_of(f, MARKS, Vec2::new(min.x, p.px_y(ys)), Vec2::new(max.x, p.px_y(ys)), 1.0, pal::DIM.a(0.6));
        }
        if !lay.narrow {
            f.text_px(Vec2::new(min.x + 4.0, min.y + 4.0), "dashed: the loss at the start", size - 1.0, pal::DIM, Align2::LEFT_TOP);
        }
        let mut seeds = Vec::new();
        let mut means = Vec::new();
        let mut marks = Vec::new();
        for w in Who::ALL {
            let at = |k: usize, v: f32| p.at(w.grid()[k].log10(), (v as f64).max(p.y[0]).min(p.y[1]));
            let pts: Vec<Vec2> = (0..GRID).map(|k| at(k, self.look.finals.get(w.index() * GRID + k))).collect();
            plot::path_of(f, FINALS + w.index() as u8, &pts, 1.6, w.color().a(0.8));
            for (k, q) in pts.iter().enumerate() {
                let top = self.look.finals.get(w.index() * GRID + k) as f64 >= p.y[1];
                means.push((*q, if top { 4.0 } else { 6.0 }, if top { w.color().a(0.5) } else { w.color() }));
                for sd in 0..SEEDS.len() {
                    let v = self.look.finals.get(3 * GRID + (w.index() * SEEDS.len() + sd) * GRID + k);
                    seeds.push((at(k, v), 3.0, w.color().a(0.45)));
                }
            }
            // the best, ringed; the one shown, white at its heart
            let best = r.best(s.batch, w);
            marks.push((pts[best], 13.0, w.color()));
            marks.push((pts[best], 9.5, Color::hex(0x0B0F18)));
            marks.push((pts[best], 6.0, w.color()));
            marks.push((pts[s.pick(w)], 3.5, pal::WHITE));
            let tag = if lay.narrow { lr_text(w.grid()[best]) } else { format!("best {}", lr_text(w.grid()[best])) };
            labels.push(ChartLabel { at: pts[best] + Vec2::new(0.0, 9.0), align: Align2::CENTER_TOP, text: tag, color: w.color(), rect: p.rect, alt: None });
        }
        dots_px(f, 0, &seeds);
        dots_px(f, 1, &means);
        dots_px(f, 3, &marks);
        if !lay.narrow {
            f.text_px(Vec2::new(max.x, min.y - 6.0), "click a dot to show that run", size - 1.0, pal::DIM, Align2::RIGHT_BOTTOM);
        }
    }

    /// The singular values of each rule's step, over its largest.
    fn draw_bars(&mut self, f: &mut Frame, lay: &Layout) {
        let s = self.setup;
        let r = record();
        let size = lay.small;
        let m = MATRICES[s.layer];
        let k = m.k();
        let c = s.checkpoint();
        let (a, _) = lay.bars[0];
        let (_, b) = lay.bars[2];
        if b.x - a.x < 60.0 || b.y - a.y < 20.0 {
            return;
        }
        let rm = room(lay.narrow);
        hud_panel(f, a - Vec2::new(26.0, rm[1] + 18.0), b + Vec2::new(8.0, 10.0));
        let title = if lay.narrow {
            format!("{}'s step: singular values over the largest, step {}", m.name, count(SPECTRA_AT[c]))
        } else {
            format!("{}'s step: its singular values, each over the largest, at step {}", m.name, count(SPECTRA_AT[c]))
        };
        f.text_px(Vec2::new(a.x - 20.0, a.y - rm[1] - 18.0 + 4.0 + lay.text), title, lay.text, pal::TEXT, Align2::LEFT_BOTTOM);
        if !s.at_best() {
            f.text_px(Vec2::new(b.x, a.y - rm[1] - 18.0 + 4.0 + lay.text), BEST_ONLY, size - 1.0, LIMIT, Align2::RIGHT_BOTTOM);
        }
        for w in Who::ALL {
            let p = Plot::new(lay.bars[w.index()], [0.5, k as f64 + 0.5], [0.0, 1.08]);
            if p.tiny() {
                continue;
            }
            let (min, max) = p.rect;
            plot::line_of(f, AXES, Vec2::new(min.x, max.y), max, 1.0, pal::DIM.a(0.8));
            plot::line_of(f, AXES, min, Vec2::new(min.x, max.y), 1.0, pal::DIM.a(0.8));
            for (v, name) in [(0.0, "0"), (1.0, "1")] {
                plot::line_of(f, AXES, Vec2::new(min.x - 3.0, p.px_y(v)), Vec2::new(min.x, p.px_y(v)), 1.0, pal::DIM.a(0.8));
                f.text_px(Vec2::new(min.x - 5.0, p.px_y(v)), name, size - 1.0, pal::DIM, Align2::RIGHT_CENTER);
            }
            let sv = r.spectrum(s.batch, s.seed, w, c, s.layer, true);
            let given = w == Who::Muon && !m.muon;
            let rule = match (lay.narrow, given) {
                (true, true) => "(AdamW)",
                (true, false) => w.short(),
                (false, true) => "Muon's run: AdamW here",
                (false, false) => w.name(),
            };
            f.text_px(Vec2::new(min.x, min.y - 16.0), rule, size, w.color(), Align2::LEFT_BOTTOM);
            let facts = if lay.narrow {
                format!("rank {}", num(stable_rank(sv), 1))
            } else {
                format!("stable rank {} · largest/mean {}", num(stable_rank(sv), 1), num(top_over_mean(sv), 1))
            };
            f.text_px(Vec2::new(min.x, min.y - 3.0), facts, size - 1.0, pal::DIM, Align2::LEFT_BOTTOM);
            for i in 0..k {
                let h = self.look.bars.get(w.index() * MOST_K + i) as f64;
                p.bar(f, i as f64 + 0.5, i as f64 + 1.5, 0.0, h.clamp(0.0, 1.08), if k > 8 { 0.5 } else { 3.0 }, w.color().a(0.85));
            }
            if k > 3 {
                f.text_px(Vec2::new(max.x, max.y + 3.0), format!("{k}"), size - 1.5, pal::DIM, Align2::RIGHT_TOP);
                f.text_px(Vec2::new(min.x, max.y + 3.0), "1", size - 1.5, pal::DIM, Align2::LEFT_TOP);
            }
        }
    }

    /// The weights' singular values.
    fn draw_weights(&mut self, f: &mut Frame, lay: &Layout) {
        let p = self.weights_plot(lay);
        if p.tiny() {
            return;
        }
        let s = self.setup;
        let r = record();
        let size = lay.small;
        let m = MATRICES[s.layer];
        let k = m.k();
        let c = s.checkpoint();
        let (min, max) = p.rect;
        let title = if lay.narrow {
            format!("{}'s weights: singular values, step {}", m.name, count(SPECTRA_AT[c]))
        } else {
            format!("{}'s weights: their singular values at step {}", m.name, count(SPECTRA_AT[c]))
        };
        chart_frame(f, lay, &p, &title, if lay.narrow { "" } else { "singular value" });
        if !s.at_best() && !lay.narrow {
            f.text_px(Vec2::new(max.x, min.y - 5.0), BEST_ONLY, size - 1.0, LIMIT, Align2::RIGHT_BOTTOM);
        }
        let goal = (self.range_goal(4), self.range_goal(5));
        log_axis_y(f, &p, size, goal);
        plot::line_of(f, AXES, Vec2::new(min.x, max.y), max, 1.0, pal::DIM.a(0.8));
        for i in [1usize, k] {
            f.text_px(Vec2::new(p.px_x(i as f64), max.y + 4.0), format!("{i}"), size, pal::DIM, Align2::CENTER_TOP);
        }
        if !lay.narrow {
            f.text_px(Vec2::new(max.x, max.y + 8.0 + size), "largest first", size, pal::DIM, Align2::RIGHT_TOP);
        }
        let mut dots = Vec::new();
        for (j, color) in [(3usize, pal::WHITE.a(0.7)), (0, Who::Sgd.color()), (1, Who::AdamW.color()), (2, Who::Muon.color())] {
            let pts: Vec<(f64, f64)> = (0..k).map(|i| (i as f64 + 1.0, (self.look.weights.get(j * MOST_K + i) as f64).max(p.y[0]).min(p.y[1]))).collect();
            if j == 3 {
                let q: Vec<Vec2> = pts.iter().map(|&(x, y)| p.at(x, y)).collect();
                for pair in q.windows(2) {
                    plot::dashed_of(f, WEIGHTS + 3, pair[0], pair[1], 1.2, color);
                }
            } else {
                p.path(f, WEIGHTS + j as u8, pts.clone(), 1.5, color);
                dots.extend(pts.iter().map(|&(x, y)| (p.at(x, y), if k > 8 { 3.5 } else { 6.0 }, color)));
            }
        }
        dots_px(f, 4, &dots);
        // the stable ranks, as a key: in the corner the curves leave empty, or on a
        // narrow window in a row over the chart
        let start = r.spectrum(s.batch, s.seed, Who::Sgd, 0, s.layer, false);
        let mut key: Vec<(String, Color)> = vec![(format!("{} {}", if lay.narrow { "start" } else { "start: stable rank" }, num(stable_rank(start), 1)), pal::WHITE.a(0.8))];
        for w in Who::ALL {
            let sv = r.spectrum(s.batch, s.seed, w, c, s.layer, false);
            let text = if lay.narrow { format!("{} {}", w.short(), num(stable_rank(sv), 1)) } else { format!("{}: {}", w.name(), num(stable_rank(sv), 1)) };
            key.push((text, w.color()));
        }
        if lay.narrow {
            let mut x = min.x - room(true)[0] + 6.0;
            f.text_px(Vec2::new(x, min.y - 5.0), "stable rank:", size - 1.0, pal::DIM, Align2::LEFT_BOTTOM);
            x += f.text_size("stable rank:", size - 1.0).x + 6.0;
            for (text, color) in key {
                f.text_px(Vec2::new(x, min.y - 5.0), text.clone(), size - 1.0, color, Align2::LEFT_BOTTOM);
                x += f.text_size(&text, size - 1.0).x + 7.0;
            }
        } else {
            let mut y = max.y - 4.0 - 3.0 * (size + 4.0);
            for (text, color) in key {
                f.caption_px(Vec2::new(min.x + 6.0, y), text, size - 1.0, color, Align2::LEFT_BOTTOM);
                y += size + 4.0;
            }
        }
    }
}

// ----- labels that keep clear of one another -----------------------------------------------

/// A label of a chart: where it wants to be (the point its alignment anchors), what it
/// says, its colour, and the chart it must stay inside.
struct ChartLabel {
    at: Vec2,
    align: Align2,
    text: String,
    color: Color,
    rect: Rect,
    /// Another place to try if there is no room near the first.
    alt: Option<Vec2>,
}

/// Where a label was put: its words, which of its places it took, the slot there (above
/// or below the place, in its heights) and where it is drawn now (gliding to that slot
/// when it changes).
#[derive(Clone, Debug, Default)]
struct Placed {
    text: String,
    alt: bool,
    slot: i32,
    /// Two eases one after the other, as `Glide` has them: the first follows the slot,
    /// the second (where the label is drawn) follows the first, so a move starts softly.
    between: Vec2,
    drawn: Vec2,
}

/// The labels on plates, each at its own place if that is clear of those placed before
/// it, else the nearest clear place above or below it, inside its chart (in the order
/// given: the first keep their places). A label keeps the slot it had while that is
/// still clear, and glides to a new one (`memo` remembers them; true while one glides).
fn place(f: &mut Frame, tags: Vec<ChartLabel>, size: f32, memo: &mut Vec<Placed>) -> bool {
    let mut taken: Vec<Rect> = Vec::new();
    let mut now: Vec<Placed> = Vec::with_capacity(tags.len());
    let ease = 1.0 - (-f.real_dt.max(0.0) / 0.08).exp();
    let mut gliding = false;
    for t in tags {
        let (min, max) = t.rect;
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
        // (the slot it had first, if it is still free; then the nearest, near its own
        // place and then near the other)
        let places = [(false, t.at)].into_iter().chain(t.alt.map(|a| (true, a)));
        let tries = places.flat_map(|(alt, at)| (0..13).map(move |k| (alt, at, if k % 2 == 1 { -(k + 1) / 2 } else { k / 2 })));
        let first = before.as_ref().map(|b| (b.alt, if b.alt { t.alt.unwrap_or(t.at) } else { t.at }, b.slot));
        let found = first.into_iter().chain(tries).find(|&(_, at, k)| {
            let c = corner(at + Vec2::new(0.0, k as f32 * step));
            fits(c) && clear(c, &taken)
        });
        let Some((alt, at, slot)) = found else { continue };
        let c = corner(at + Vec2::new(0.0, slot as f32 * step));
        taken.push((c, c + w));
        let goal = at + Vec2::new(0.0, slot as f32 * step);
        let (between, drawn) = match before {
            Some(b) if b.drawn.is_finite() && b.between.is_finite() && ((b.drawn - goal).length() > 0.5 || (b.between - goal).length() > 0.5) => {
                gliding = true;
                let between = b.between + (goal - b.between) * ease;
                (between, b.drawn + (between - b.drawn) * ease)
            }
            _ => (goal, goal),
        };
        f.caption_px(drawn, t.text.clone(), size, t.color, t.align);
        now.push(Placed { text: t.text, alt, slot, between, drawn });
    }
    *memo = now;
    gliding
}

// ----- the scene's contract ------------------------------------------------------------------

impl Scene for MuonAdam {
    fn view(&self) -> View {
        View::Flat { center: Vec2::ZERO, half_height: 1.0 }
    }

    fn settings(&self, s: &mut ViewSettings) {
        s.grid = false;
        s.bloom = s.bloom.min(0.2);
    }

    fn animated(&self) -> bool {
        self.film.playing || self.look.moving || self.dragging || self.labels_moving
    }

    fn pointer_hint(&self) -> Option<&'static str> {
        Some("white dot under the loss chart: drag the step    boxes: choose a layer    dots on the right: show that run")
    }

    fn ui(&mut self, ui: &mut egui::Ui, _: &Gfx) {
        self.panel(ui);
    }

    fn frame(&mut self, f: &mut Frame) {
        hold_batches(f);
        let lay = Layout::of(f.view_size(), f.top_inset);
        // (a canvas squeezed to a sliver is one the controls have opened over, on a phone)
        let tiny = lay.view.x < 160.0 || lay.view.y < 120.0;
        self.dragging = false;
        if !tiny {
            self.buttons(f, &lay);
            if !lay.compact {
                self.clicks(f, &lay);
            }
            self.drag(f, &lay);
        }
        self.setup = self.setup.kept();
        self.advance(f.dt);
        let snap = self.canvas != Some(lay.view) && !tiny;
        if snap {
            self.canvas = Some(lay.view);
            // (labels are not glided across a new window)
            self.placed.clear();
        }
        self.follow(f.real_dt, snap);
        if tiny {
            return;
        }
        self.labels_moving = false;
        let mut labels = Vec::new();
        self.draw_head(f, &lay);
        self.draw_loss(f, &lay, &mut labels);
        if !lay.compact {
            self.draw_network(f, &lay);
            self.draw_lr(f, &lay, &mut labels);
            self.draw_bars(f, &lay);
            self.draw_weights(f, &lay);
        }
        self.labels_moving |= place(f, labels, lay.small, &mut self.placed);
    }
}

// ----- the panel ---------------------------------------------------------------------------

impl MuonAdam {
    fn panel(&mut self, ui: &mut egui::Ui) {
        ui::note(
            ui,
            "Scenes adam and muon took the two rules apart. Here they, and SGD with momentum, train the same small network: the same starting weights, the same batches, each rule at the best of twelve step sizes, three times over with three seeds. The question is not which is better but how their training differs: how fast the loss falls, how wide a range of step sizes each bears, what shape each step has, what becomes of the weights, and what a larger batch changes. Every run was recorded beforehand; everything shown is this network's, for the step sizes and seeds tried, and says nothing on its own about other networks.",
        );

        ui::section(ui, "Things to try");
        for (k, t) in TRIALS.iter().enumerate() {
            if ui::experiment(ui, t.button, t.caption) {
                self.set_up(k);
            }
        }

        ui::section(ui, "The run shown");
        let s = &mut self.setup;
        let mut batch = s.batch;
        if ui::choice(ui, &["batches of 8", "batches of 128"], &mut batch) {
            s.batch = batch;
        }
        let mut seed = s.seed;
        if ui::choice(ui, &["seed 1", "seed 2", "seed 3"], &mut seed) {
            s.seed = seed;
        }
        ui::note(ui, "A seed fixes the starting weights and the order of the batches; every rule gets the same.");
        let mut layer = s.layer;
        if ui::choice(ui, &["layer 1", "layer 2", "layer 3", "read-out"], &mut layer) {
            s.layer = layer;
        }
        let mut steps = if s.shift == [0; 3] {
            0
        } else if s.shift == [2; 3] {
            1
        } else if s.shift == [-2; 3] {
            2
        } else {
            3
        };
        if ui::choice(ui, &["best step sizes", r"$\times 4$", r"$\div 4$"], &mut steps) {
            s.shift = [[0; 3], [2; 3], [-2; 3]][steps.min(2)];
        }
        ui::note(ui, "Each rule's step size, from its best; a click on a dot of the step-size chart chooses one rule's.");
        let mut step = s.step as f32;
        if ui::slider_log(ui, "the step the bars are read at", &mut step, 1.0..=STEPS as f32) {
            s.step = step as f64;
            self.film.playing = false;
        }
        ui::note(ui, "The singular values were recorded after steps 1, 2, 5, 10, 20, 50, 100, 200, 500, 1000 and 2000: the bars show the last of these at or before this step.");
        ui::checkbox(ui, &mut s.other, "the other batch size's runs, dashed");
        if ui.button(ui::label_text(ui, if self.film.playing { "pause" } else { "play the run" }, 12.5, None, false)).clicked() {
            self.play();
        }

        ui::section(ui, "The set-up");
        ui::note(ui, &format!("The network: 2 inputs, {DEPTH} hidden layers of {WIDTH} tanh units, 3 outputs, 2307 numbers in all. The data: {POINTS} points on three spiral arms, the same for every run; the loss is the cross-entropy on all {POINTS} points, and it only measures how well the network fits them (no new points are tried). An MLP and not the course's small transformer: tried in the same way on the transformer's repeated sequences (batches of 8, seed 1, six step sizes each), the three rules ended within a factor of 1.5 of one another, and all below 0.71, the least any network can reach on sequences it has not seen, so that its loss mixed learning with memorising."));
        ui::note(ui, "Every run: 2000 steps at a constant step size, from the seed's starting weights, through the seed's batches (drawn without putting back, a fresh order each pass through the data).");
        ui::note(ui, r"SGD with momentum as PyTorch has it: $b \leftarrow 0.9\,b + g$, $\theta \leftarrow \theta - \eta b$. AdamW as PyTorch has it: $\beta_1 = 0.9$, $\beta_2 = 0.999$, $\varepsilon = 10^{-8}$, weight decay $0.01\,\eta\,\theta$ each step.");
        ui::note(ui, "Muon as its reference code has it (Jordan and co-authors, 2024): momentum 0.95 with Nesterov's form, five rounds of the polynomial iteration with $(3.4445, -4.7750, 2.0315)$, no weight decay, on the hidden layers 2 and 3. Layer 1, which reads the data, the read-out and every bias take AdamW, as its authors prescribe; here exactly the AdamW above, at AdamW's best step for that batch size, so that the two runs differ only on layers 2 and 3.");
        ui::note(ui, "The step sizes: twelve for each rule, each twice the last (SGD with momentum from 0.0025, AdamW from 0.0001, Muon from 0.000125). A rule's best is the one whose three runs end, on average, at the lowest loss.");

        ui::section(ui, "Reading the picture");
        ui::key(
            ui,
            &[
                (MOMENTUM, "pink: SGD with momentum"),
                (ADAM, "orange: AdamW"),
                (MUON, "blue: Muon"),
                (pal::DIM, "faint lines: the other two seeds, at the same step size; dashed: the other batch size"),
                (pal::WHITE, "ringed dot: a rule's best step size; white heart: the one shown; white dashed line: the weights at the start"),
            ],
        );
        ui::note(ui, "The three rules' step sizes are not one scale: a step of 0.01 means different things to each. Compare the widths of their dips on the step-size chart, not where they lie.");

        ui::section(ui, "Words");
        ui::term(ui, "Step size:", r"the learning rate $\eta$: what a rule's step is multiplied by.");
        ui::term(ui, "SGD with momentum:", "each step goes along a running sum of the recent mini-batch gradients (scene momentum).");
        ui::term(ui, "AdamW:", "Adam, which divides each weight's step by the size of its recent gradients, with decoupled weight decay (scene adam).");
        ui::term(ui, "Muon:", "for a weight matrix: the momentum, with every singular value pushed towards 1 (scene muon).");
        ui::term(ui, "The step:", r"what a rule adds to a weight matrix in one step, $\Delta W$ (weight decay apart).");
        ui::term(ui, "Singular values:", "how much a matrix stretches each of its own perpendicular directions, largest first (scene svd).");
        ui::term(ui, "Stable rank:", r"$\sum_i \sigma_i^2 / \sigma_1^2$: about how many singular values are close to the largest; between 1 and the number of them.");
        ui::term(ui, r"$\|\Delta W\| / \|W\|$:", "how large the step is beside the weights: the Frobenius norms, the square roots of the sums of the squares of the entries.");
        ui::term(ui, "Batch, seed:", "the points one step's gradient is averaged over; the draw of the starting weights and of the batches' order.");

        ui::section(ui, "Numbers");
        self.numbers(ui);

        ui::section(ui, "What published work reports");
        ui::note(ui, "On language models of hundreds of millions of parameters or more, not networks like this one:");
        for line in PUBLISHED {
            ui::note(ui, line);
        }

        ui::section(ui, "Why they might differ");
        for line in WHY {
            ui::note(ui, line);
        }
        ui::note(ui, &why_here());

        ui::see_also(
            ui,
            &[
                ("muon", "Muon's step taken apart"),
                ("adam", "Adam taken apart"),
                ("signnoise", "normalised steps in mini-batch noise"),
                ("race", "seven optimisers on one network, live"),
                ("optlab", "any two rules side by side"),
            ],
        );
    }

    /// The read-outs.
    fn numbers(&self, ui: &mut egui::Ui) {
        let s = self.setup;
        let r = record();
        for w in Who::ALL {
            let k = s.pick(w);
            ui::readout_colored(ui, w.name(), format!("step size {}", lr_text(w.grid()[k])), w.color());
            ui::readout(ui, "  final loss, this seed", g(r.final_loss(s.batch, s.seed, w, k)));
            ui::readout(ui, "  mean of the three seeds", g(r.mean_final(s.batch, w, k)));
            ui::readout(ui, "  first below 0.01 at step", steps_text(r.reaches(s.batch, s.seed, w, k, TARGET)));
        }
        let c = s.checkpoint();
        let m = MATRICES[s.layer];
        ui::readout(ui, &format!("{}, after step {}", m.name, count(SPECTRA_AT[c])), "");
        for w in Who::ALL {
            let (d, wt) = (r.spectrum(s.batch, s.seed, w, c, s.layer, true), r.spectrum(s.batch, s.seed, w, c, s.layer, false));
            let rel = if frob(wt) > 0.0 { frob(d) / frob(wt) } else { f64::NAN };
            ui::readout_colored(
                ui,
                &format!("  {}", w.short()),
                format!("step: rank {}, ratio {}, size {} · weights: rank {}", num(stable_rank(d), 1), num(top_over_mean(d), 1), g(rel), num(stable_rank(wt), 1)),
                w.color(),
            );
        }
        ui::note(ui, r"(rank: stable rank; ratio: largest singular value over their mean; size: $\|\Delta W\| / \|W\|$.)");
    }
}

/// The lowest and highest of a measure of layer 2's step at batches of 8, over every
/// recorded step after the start and every seed.
fn range_of(w: Who, f: fn(&[f32]) -> f64) -> (f64, f64) {
    let r = record();
    let vals: Vec<f64> = (0..SEEDS.len()).flat_map(|s| (1..SPECTRA_AT.len()).map(move |c| (s, c))).map(|(s, c)| f(r.spectrum(0, s, w, c, 1, true))).collect();
    (vals.iter().copied().fold(f64::INFINITY, f64::min), vals.iter().copied().fold(0.0, f64::max))
}

/// What this network's numbers say beside the explanations (batches of 8, layer 2).
fn why_here() -> String {
    let (sa, sb) = range_of(Who::Sgd, stable_rank);
    let (aa, ab) = range_of(Who::AdamW, stable_rank);
    let (ra, rb) = range_of(Who::AdamW, top_over_mean);
    let (ma, mb) = range_of(Who::Muon, top_over_mean);
    format!(
        "Here, on layer 2 with batches of 8 (every recorded step, three seeds), the steps of SGD with momentum and of AdamW had stable ranks of {} to {} and {} to {}: close to rank one, much as the first account describes for transformers. The largest singular value of AdamW's step was {} to {} times their mean, and of Muon's {} to {} times, not about 1 (see the third experiment).",
        num(sa, 1),
        num(sb, 1),
        num(aa, 1),
        num(ab, 1),
        num(ra, 0),
        num(rb, 0),
        num(ma, 1),
        num(mb, 1)
    )
}

/// What published work reports, each attributed to its authors with its scope (as the
/// course's fact-checked notes give them: muon.md §6.3, §9 and §10, items 7 to 10 and 18).
const PUBLISHED: [&str; 10] = [
    "Jordan and co-authors (2024), who introduced Muon: a 1.5-billion-parameter transformer reached GPT-2 XL's score on a benchmark (HellaSwag) in 10 hours on eight GPUs with Muon and 13.3 with AdamW, both tuned. In the NanoGPT speed-run's comparison of a 124-million-parameter GPT, each optimiser tuned in about 20 tries and run once, the validation loss ended at 3.276 with Muon, 3.275 with SOAP and 3.341 with Adam.",
    "Liu and co-authors at Moonshot AI (2025): in runs of 0.4 to 1.5 billion parameters, each trained for its compute-optimal length, Muon matched AdamW's loss with about 52 % of the computation. Muon reused the settings tuned for AdamW; it was not tuned on its own.",
    "Wen and co-authors (2025), tuning every optimiser separately from 0.1 to 1.2 billion parameters: Muon's speed-up over AdamW was at most about 1.4 (at 130 million), falling to about 1.1 at 1.2 billion; their fitted law predicts Muon slightly worse than AdamW at 7 billion parameters trained on the amount of data a common rule (Chinchilla's) gives for that size, an extrapolation, not a measurement.",
    "Li and co-authors (2025), at 1.1 and 5.4 billion parameters, with a step size shared between the rules: Muon needed 10 % and 6 % fewer steps than AdamW to reach AdamW's final loss. Frans, Abbeel and Levine (2025), each method tuned on a 162-million-parameter GPT-2: Muon needed about 80 % of Adam's steps, and another method, SOAP, fewer still.",
    "Semenov and co-authors (2025), tuning each method from 124 to 720 million parameters: at 720 million with batches of a million tokens two other optimisers, AdEMAMix and MARS, did best.",
    "Shah and co-authors at Essential AI (2025): for a 1-billion-parameter model, AdamW needed more data than Muon to reach the same loss, and the ratio did not fall as the batch grew; over runs of 500 million parameters with batches from 128 thousand to 16 million tokens, Muon gave the better trade-offs of time against computation. Khona and co-authors (2026, known here from the abstract) report Muon and SOAP staying stable at batches of up to 100 million tokens where AdamW degraded.",
    "Liu and co-authors (on their large Moonlight model) and Wang and co-authors (2025, on a 160-million-parameter GPT) measured flatter singular-value spectra in Muon's weight matrices than in AdamW's or Adam's; such matrices can still be strongly low-rank (Dolatabadi and co-authors, 2026, from the abstract).",
    "Reported where Muon did no better, or worse: fine-tuning a model pre-trained with AdamW (Liu and co-authors); attention scores growing too large, more often with Muon than with AdamW, in the Kimi K2 team's very large models (they added a correction); a matched small vision transformer in which the ranking reversed (Beneventano and co-authors, 2026, from the abstract); and in the speed-run's benchmark Muon with its momentum switched off took 8225 steps to the target against 3250 for tuned Muon.",
    "An earlier rule that makes the gradient orthogonal before averaging it, Tuddenham and co-authors' (2022), did worse than SGD with momentum at batches below 16 on an image task: a finding about that rule, not about Muon. Whether Muon's best step size stays the same as a network grows is disputed: it did not across widths in one study (Pethick and co-authors, 2025), and was about the same across four model sizes in another (Ahn and co-authors, 2025).",
    "Muon's authors count its extra arithmetic at under 1 % of a step for language models trained on large batches: a count of operations, not a timing. One speed-run log has 142 ms a step with Muon against 139 ms with Adam.",
];

/// The explanations offered, each attributed with the evidence it rests on (as the
/// course's notes collect them, muon.md §6.4).
const WHY: [&str; 8] = [
    "No account of why Muon trains faster, where it does, is settled. Those published rest on different evidence and partly disagree:",
    "Rare directions. Jordan and co-authors (2024) wrote, from looking at them, that the steps of SGD with momentum and of Adam on a transformer's matrices are nearly of low rank, and speculated that making them orthogonal strengthens the rare directions. A speculation, with no numbers given.",
    "The right ruler. Bernstein (2025) derives Muon's step as the one that lowers the loss fastest, to first order, among steps of a given size, when a step's size is how much it can stretch an input (scene steepest); Bernstein and Newhouse (2024) found the same direction with the size as a penalty, which makes the step's length grow with the gradient. The bound behind that choice is proved for a linear model with squared error and normalised inputs; that it speeds training rests on experiments.",
    "Inputs of low stable rank. Davis and Drusvyatskiy (2025) show that the decrease of the loss a spectral step guarantees on a layer beats a gradient step's when the gradient's 'nuclear rank' exceeds the stable rank of the layer's input, and prove that such inputs are common at the start of training. Their comparison is with gradient descent, not Adam, and they give a case where spectral steps lose.",
    "Step sizes along singular directions. Wu and co-authors (2026) measured, on language models of 64 and 124 million parameters, that the few strongest directions of the momentum bear, on held-out data, a step nearly ten times smaller than the rest; Muon gives every direction the step the strongest bear, which lifts the rest. Measured, not proved.",
    "Rare classes. Wang and co-authors (2025) prove, for one step from zero of a one-layer linear memory, with Muon orthogonalising exactly and Adam replaced by sign descent, that Muon learns rare and common items about equally whatever the embeddings, while sign descent can be badly unbalanced; in a 160-million-parameter model they found its gains came mostly from the attention value and output matrices and the MLPs.",
    "Not flatness alone. Frans, Abbeel and Levine (2025) measured the ratio of the largest to the average singular value of a step: about 12 for Adam and about 1 for Muon on their GPT-2. Yet SOAP, whose steps were less flat (2 to 3), did better than Muon, and adding an Adam-style per-weight scaling to Muon helped almost as much again.",
    "Where a run ends. Fan, Schmidt and Thrampoulidis (2025) prove that, for a linear classifier on separable data, an idealised Muon tends to the one with the largest margin measured in the spectral norm; a 2026 paper (Wu, Mansouri and Zhu, known here from its abstract) states that Muon as implemented tends instead to gradient descent's.",
];

// TABLES (made by `tests::the_tables_are_what_training_gives`; two characters a number, see `unpack`)
const LOSS_TABLE: &str = concat!(
    "bibibibibibhbhbhbhbhbgbgbgbfbfbebebebdbcbcbbbbbbbbbabababababababZbZbZbabZbZbZbZbZbZbZbZbZbZbZbZbYbYbYbYbYbXbXbWbWbibibibhbhbhbhbgbgbgbfbfbe",
    "bebdbdbcbcbbbababababababababababZbZbZbZbZbZbZbZbZbZbZbZbZbZbYbYbYbYbYbYbYbXbWbWbUbSbQbLbibibibhbgbgbgbfbfbfbebdbcbcbcbbbbbababababbbbbbbaba",
    "babZbZbZbZbZbZbZbZbabZbZbZbZbZbYbZbYbYbYbYbXbWbWbVbTbPbKbBaxaUbibibhbgbfbfbfbfbfbebebdbdbcbcbcbcbbbababbbdbdbbbabababababZbababZbZbZbbbbbZbZ",
    "bZbZbZbZbYbYbZbYbXbXbXbRbWbMbCaoaPZhbibibhbfbebfbfbgbhbgbfbfbfbfbebebbbbbcbdbebebcbababababababZbbbababZbZbdbebababZbZbabZbZbZbabYbYbYbTbMbN",
    "bEasa/bRZzbibibgbfbfbibjblblbjbhbgbgbebcbcbfbgbebcbcbdbcbhbcbibkbibbbfbfbebZbabibdbabdbjbVbYbhbmbdbfbWbibdbfbNbabKbQc7cPbVbKbibibgbhbkbpbqbp",
    "blbfbbbcbgbdbcbfbcbfbhbjbgbfbdbmbxboblbbbibgbpbcc1c5cLcCbkcAc6cgc4cfdLcNcdd2bldWd+dGb/cLdEeJc9cacvbibjblbrbubvbtbubvbobzc1btc9cTcEcwcZdedkex",
    "dTdpdddceBefdafEf6dzcdeVf7dsewf7eveYeLfbeXf4e/dFbodseceifEeje2e0fbdff6fbbibmbvcCc0cPcVcrbjcyd6dzdReBeweoehefdweCf7eCe6f7fZfmgLgcfxepgTeyevep",
    "dpdhfGdCeifkdqeYgZexgdg4fbgih0gEf6e+gAdeggeigWbibyc8cickdGdfeldZewfLeTeYg3gtgfhBehg0ftfYgOgyfhhzg4hReWfsg7gMgJe+f9gKgXcOfggIgRhYhsgZfEglgdg6",
    "hNf7f/gAh6fci2hyh4hMbicNcdd5fQfYgfgfgZfwhohwfNhghdguhqhugyhmhuhmhliHinhuiCh4hKgTgUi9hTfZgchXh3i1gtgsixiOgvhRgviYhchaicishMeNg/i7hpiGiQbiczdD",
    "gKduhGglgqi9hpiriphbiFjRiPiRjchZjFjZjljDkGihk2kWiritjHhkiEiaiQfxiNjBiqhhjSi9iJgwjTibigh6iHgiiJiRhqiijXg1i5hdbibibibibibibibibibhbhbhbhbhbhbh",
    "bhbhbgbgbgbgbgbfbfbfbfbebebebdbdbcbcbcbbbbbbbbbabababababababZbZbZbZbZbZbZbZbZbZbZbibibibibibhbhbhbhbhbhbhbgbgbgbgbgbgbfbfbfbebebebdbdbcbcbc",
    "bbbbbbbababababababababZbZbZbZbZbZbZbZbZbZbZbZbZbYbYbYbYbibibibhbhbhbhbgbgbgbgbgbfbfbfbebebebdbdbcbcbbbbbbbbbabababababababZbZbZbZbZbZbZbZbZ",
    "bZbZbZbZbZbZbYbYbYbYbXbXbWbVbUbibibhbhbgbgbfbfbfbfbebebdbdbdbcbcbcbbbbbabababababababababZbZbZbZbZbZbZbZbZbZbZbZbZbYbYbYbYbYbXbXbWbVbUbTbQbN",
    "bLbIbibibhbgbfbebebebebdbdbcbcbcbbbbbbbabababababababababZbZbZbZbZbZbZbZbZbZbZbZbZbYbYbYbYbXbWbWbWbWbTbSbQbMbIbEb8b1atbibhbgbfbebdbdbdbdbdbd",
    "bdbdbcbcbcbbbababababbbcbbbababZbZbZbZbZbZbZbYbZbabZbYbYbXbXbWbWbVbUbUbTbUbPbObMbIbFa+aeaCZKbibibhbfbebebebfbfbebebebdbcbbbababZbabbbbbcbbbZ",
    "bZbZbZbabZbZbZbZbYbYbYbabbbXbXbVbVbVbVbSbRbRbLbMbEb7aZaEZWYLXmX/WfbibkbmbjbhbgbgbhbgbfbdbbbabababbbbbcbcbbbbbcbabZbababbbbbababbbZbZbYbXbabc",
    "bZbfbWbVbUbRbJb8bBapabaGZlYRYLW5VMUnUJT/bibububoblblbmbrbrbmbhbebgbdbbbbbcbdbdbebebfbebfbdbiblblbcbebcbpbXbXbZbPbSbPbPbKbabFbKb8aoauaUaca0aA",
    "aca/aRaNa1aBZVbicEbxc3bwbyc3c8c2bobjb+c5bpbgblbfbZbYbhbcbfbpbdbdbkbabTbNbdbjbXbvbgbibablbqbUbSbgbabnbPc7bgbcbabibebPbYbMbNbpbjbHbicgc3c7cOcD",
    "c0bvbxc0brc0cVcUbrcNcLbzc6c1bxc6c0brcGbebvb+bpbibhbxcAbfbkbecEbrbkbpcCbmbrbvbgbobxbhc2cAbvbgbvbucDbkcUbidAcXctd5d1c3cBcad7cec4cidBbscvd3c/cn",
    "d1c9chcec6bpcEcvc4cdcAc2bwcIcMcQbkb/btbocRcTbzbpbjbvbhb+cRcHcqbic9cSbscZbtblbibibibibibkbkblbkbibhbfbdbcbcbbbbbbbabababcbcbbbbbababZbZbZbabZ",
    "bZbYbYbbbbbYbYbXbXbYbXbXbWbVbUbVbVbSbKbLbBb7auadaLbibibibibibkbkblbkbibhbfbdbcbcbbbbbbbabababcbcbbbbbababZbZbZbabZbZbYbYbbbbbYbYbXbXbYbXbXbW",
    "bVbTbUbUbRbIbIb6a/akaOZ/bibibibibibkbkblbkbibhbfbdbcbcbbbbbbbabababcbcbbbbbababZbZbZbabZbZbYbYbbbbbYbYbXbXbYbWbXbVbUbSbSbSbObCb9awajaMZoZHbi",
    "bibibibibkbkblbkbibhbfbdbcbcbbbbbbbabababcbcbbbbbababZbZbZbabZbZbYbYbbbbbYbYbXbXbYbVbWbTbSbQbObLbCa/anaWZ/ZZYZXQbibibibibibjbkblbkbibhbfbdbc",
    "bcbbbbbbbabababcbcbbbbbababZbZbZbabZbZbYbYbabbbYbXbWbVbXbTbTbRbPbJbAb8apaRZrZ5XrWSVBTVbibibibibibjbkblbkbibgbfbdbcbcbbbbbbbababbbcbcbbbababa",
    "bZbZbZbZbZbZbYbYbababXbWbUbSbVbQbQbMbHb9axaca7ZQYkXKV3THRPPZbibibibibibjbkbkbjbibgbfbdbcbcbcbbbbbababbbcbdbbbabababZbZbYbZbZbYbXbXbZbYbVbTbP",
    "bMbQbIbEb5araRZsYwYNX0WvUWVjTgROPtbibibibhbhbjbjbkbjbhbgbfbdbdbcbcbbbbbababcbdbdbbbabababZbZbYbYbYbXbVbUbVbTbPbRbHbBbCazajZ/ZIXiWRWeVDS/R+QE",
    "PMa5UQRWbibibhbgbgbibibjbibhbfbfbebdbdbcbbbababbbdbfbbbZbZbZbZbYbXbVbWbUbSbPbLbObGb4b5aoaha0ZzYmYUY7XLYGV8YBVSYoWxWjQxWZXabibhbgbfbfbhbhbjbi",
    "bhbfbfbfbdbbbZbZbbbdbcbabcbYbYbXbYbYbVbUbWbRbNbObEbKa/agaLada3ZyZQZtY6Y6YwXCYuXTZ1Y0Y9WuXbX1YmZtbibibgbebfbkbibkbibfbcbbbabZbZbabcbbbabYbYbb",
    "bWbVbWbXbebUbRbQbObLbFbAb2avagaiajaBa7aea0ZPZYZpZTZdYNZ7YYaNaza5a8a0aHbibkbhbibfbibgbnblbfbbbbbbbdbYbbbbbZbYbWbZbabUbWbTbVbVbUbUbRbQbRbJbKb7",
    "bCbDbHb6a/b7a/aoapararapbOb6awbEbWbJbHbJbBbIbdbdbdbdbcbcbcbcbcbcbcbcbcbcbcbbbbbbbbbbbabababababababZbZbZbZbZbZbZbZbZbZbZbZbZbZbYbYbYbYbYbYbY",
    "bXbXbXbWbWbUbTbRbQbdbdbcbcbcbcbcbcbcbcbcbcbbbbbbbbbababababZbZbZbababZbZbZbZbZbZbZbZbZbZbZbZbYbYbYbYbYbYbYbXbXbXbWbUbTbRbPbQbObKbIbFbdbcbcbc",
    "bcbcbcbcbcbcbbbbbbbababababZbZbZbZbababababZbZbZbZbabZbZbZbZbYbZbYbYbYbYbYbXbYbXbVbVbVbSbPbPbLbKbIbGazajaGbdbcbcbcbcbbbbbbbbbbbbbabababababa",
    "bababababbbbbbbababZbZbZbbbZbbbZbZbYbZbZbYbYbYbXbYbYbWbUbVbTbQbPbLbCb6b8abZfYwXCbdbcbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbabZbabbbcbcbcbbbababbbbbe",
    "bgbbbabZbZbabbbZbabZbYbababYbYbZbXbZbWbVbSbSbLarbla8aTbdbcbbbabbbbbcbdbdbdbdbcbcbcbbbbbabababbbcbcbdbdbabbbfbgbbbjbmbkbabbbfblbfbabfbbbdbdbd",
    "bcbcbcbcbZbXbdbVbWbabWbRbXbZbdbbbababbbfbibibibkbjbebdbcbcbbbcbebgbfbobfbgbmbfbqbpbybebrcJblbibecRchczcHebdGdWcqdadSbtdkeKd3c/dEdrdectdGdXdI",
    "bzbdbdbabbbgbkbibhbsbqbmbsboblblbkbwbpbzcLcedPdHe3eHdqeveseGe/e0e7dbeFeseqd2btdJd3e1dxeadYc8e+fadueddfeefUeeeJfCdgdnbdbmbibuc7bfcDb/c4cHblcr",
    "cweAd/ftfBgMgOf6g3fCfSf4e1eJfdfbeuetekgjeDfxghg5fhfSg3fgfhgPfIh3gfghevg4gbcpetd1gaekfNeVeebdc8cUdNfNe+eNgQgng4gUgUgAedftgEgGfEdifLhNgThJhtfC",
    "h6fsgvgzh7gRewfIgxi9hlhHfafnfQhygpevhIgfgJg5hkhhgGg8hqh9h6fKgKgqbdcgePfWgFgQgvh5gOgSgthGe/hWhGhCftfWhNf9dqfBgph5iQgFiNh/h8hph9gjdphGiphKitgt",
    "iDi/idhUjCgShhiGiXgPhZisg4jKfpiWhriRhHbddFfIhBiMiFf/hXhEi4hQgpi8iuhTiLiripiah8jejUjoikicfvgWi5hGiYfzgAhUj4jBi7injmj1jBinj8inisjIjCjCjKjMjYi4",
    "ini9jDg3jPgZbdbdbdbcbcbcbcbcbcbcbcbcbcbcbcbcbcbcbcbcbcbcbcbcbbbbbbbbbbbbbabababababababababZbZbZbZbZbZbZbZbZbZbZbZbZbZbYbYbYbYbdbcbcbcbcbcbc",
    "bcbcbcbcbcbcbcbcbcbcbcbbbbbbbbbbbbbbbbbababababababZbZbZbZbZbZbZbZbZbZbZbZbZbZbZbYbYbYbYbYbYbYbXbXbXbdbcbcbcbcbcbcbcbcbcbcbcbcbbbbbbbbbbbbba",
    "bababababababZbZbZbZbZbZbZbZbZbZbZbZbZbZbZbYbYbYbYbYbYbYbXbXbXbWbVbUbTbSbQbdbcbcbcbcbcbcbcbbbbbbbbbbbbbabababababZbZbZbZbZbZbZbZbZbZbZbZbZbZ",
    "bZbZbZbYbYbYbYbYbYbYbXbXbXbWbVbUbTbRbQbPbNbLbKbIbdbcbbbbbbbbbbbbbbbbbbbababababZbZbZbZbZbZbZbZbabZbZbZbZbZbabZbZbZbYbYbZbYbYbYbYbXbXbWbWbUbU",
    "bTbRbObNbLbJbHbHbAb7b2bdbbbababababbbbbbbbbbbabababZbZbZbZbZbZbZbZbabababZbZbZbZbabZbZbZbYbYbZbZbYbYbXbXbWbWbUbTbUbSbObLbJbFbAb4ataOZ+Yybdba",
    "babababbbdbebdbcbbbbbabababababZbZbabababbbbbabZbZbZbabbbcbabZbYbYbZbabZbYbYbXbXbXbVbXbVbSbMbEaxaVZxZ8YFX2W5VUbdbcbbbbbcbebhbgbebcbcbbbbbbba",
    "bababZbababababcbcbababbbbbcbdbibcbabZbZbZbabZbXbZbWbVbRbRbWbSbHbJb0azaaa1ZobMXxVqVAbdbmbibebhbjbkbibebebebebdbdbcbabababcbcbhbfbfbbbfbbbbbf",
    "bbbhbobnbfbbbkbwbabebfbRbZbYbPbabXbBbFbWbLbKbJatazb3alaIYjbdc5bpbrbwbkbibgbnbobmbjbcbabdbbbabdbWbWbibRbhbZbUbXbfbPbObYbmbTbbbSc5bfbWbibebic2",
    "bbbVbRbhbObQbcbZbTbnc0bfc0bibPaybdcRcEcjcec4c5c3cAbvbhbqbtbnbmbdbZbjbdbrbjcAc5bobibnbvc9bmbrcNbzcFc4cocIc2bmcCbkcBbjcFbmbkc3c0bzbZbjbgcvc2c7",
    "bhbocAbdcbdNdjdKcscBcud4cocbcfcpd4cjbrcQcAcFb/cJc9cAcgcVcfbkbxcbbdcibxcGcTd2bxc6cRcDc2b+c1cRcLcTcdcNb+cXbncEcdc7bncKbrbmbdbcbbbbbbbbbbbbbbbb",
    "bbbbbbbbbbbbbabZbZbZbababababababZbZbYbbbabababYbYbYbYbXbXbXbWbYbVbUbTbUbRbSbObObIbLbKbJbAb0atbdbcbbbbbbbbbbbbbbbbbbbbbbbbbbbbbabZbZbZbababa",
    "babababZbZbYbbbabababYbYbYbYbXbXbXbWbYbVbUbTbUbRbSbObNbHbJbHbJb8avaobdbcbbbbbbbbbbbbbbbbbbbbbbbbbbbbbabZbZbZbababababababZbZbYbbbZbababYbYbY",
    "bYbXbXbXbWbXbVbTbTbTbQbQbMbLbFbGbAbBaraOa5bdbcbbbbbbbbbbbbbbbbbbbbbbbbbbbbbabZbZbZbababababababZbZbYbbbZbZbabYbYbYbXbXbXbXbWbXbUbSbRbRbNbNbI",
    "bHb8bBapaUa7ZLYTbdbcbbbbbbbbbbbbbbbbbbbbbbbbbbbbbabZbZbZbababababababZbZbYbbbZbZbabYbYbYbXbXbXbWbVbVbTbPbNbMbIbGbBb5arasa6ZPYcWzWKbdbcbbbbbb",
    "bbbbbbbbbbbbbbbbbbbbbababZbZbZbababababababYbZbYbbbZbababYbYbYbXbWbWbVbTbRbPbKbGbFb5avagaBZOYZXKW7UASqRSbdbcbbbbbbbbbbbbbbbbbbbbbbbbbabababZ",
    "bZbZbabababababZbYbZbYbbbZbabZbYbXbXbWbSbTbRbPbNbGb6b2aoaBZRYqXcV+VEVNSaQEOYNJbdbcbbbbbbbbbbbbbbbbbbbbbbbababababZbZbZbababababZbZbYbZbZbbba",
    "babZbWbVbUbUbPbObJbEb4anaWZvYfXCWUVNXoXNSnYfU7SkT9OQbdbcbbbbbbbbbcbcbbbbbbbababababababZbZbZbabababbbZbYbZbabZbabbbYbWbTbSbPbSbFbBa/a/aKZaYK",
    "X/WUWaUoX/XNTVTKVOVcZDVHUsbdbbbabbbcbdbfbdbcbbbababababbbababZbZbZbababbbZbXbXbabYbWbbbgbWbTbLbKbEbDb7aQa0aCa3ZXYbY9YZXdZrXcYwYBXvYqYBZ8XkW/",
    "bdbbbbbcbebebgbebbbbbbbbbabbbbbZbYbYbababbbZbcbYbXbWbdbVbVbWbdbUbMbEbIb3b2b2aZaEaiaRaTa7a8aUZ0a3Z6a0ZzYeaUa5ZlaMaTbdbcbebibebdbcbcbabcbcbcbZ",
    "bcbdbZbYbWbcbZbVbUbYbSbXbYbVbRbLbRbTbGbEbBbEb9bHbBbFbEb3b0aua+a+bGbCbRbHbQbSbNbJbIbCbPbUbebebebebebebebebdbdbdbdbdbdbcbcbcbbbbbbbabababZbZbZ",
    "bZbZbZbZbZbZbZbZbZbZbZbZbZbZbZbYbYbYbYbYbYbYbYbXbXbXbXbWbWbVbVbebebebebebebdbdbdbdbdbdbcbcbbbbbbbababZbZbZbZbZbZbZbZbZbZbZbZbZbZbZbYbZbZbYbZ",
    "bYbYbYbYbYbYbYbXbXbWbWbWbVbVbSbRbObNbebebebebdbdbdbdbdbcbcbcbbbbbbbbbababZbZbababababZbZbZbZbZbZbZbZbYbYbYbZbZbYbYbYbYbXbXbXbXbWbVbVbTbTbPbP",
    "bNbEb4ayarbebebebdbdbdbdbdbcbcbbbbbcbcbdbcbbbabbbbbcbcbbbababZbZbZbZbZbYbZbYbZbYbZbabYbYbZbYbXbXbWbVbTbSbSbRbKb6b6b7agaGZga8bebebebdbdbdbcbc",
    "bbbbbbbcbhbhbhbebbbcbfbdbcbdbdbcbabababababZbZbabZbabYbabbbYbZbabbbZbZbYbabXbWbZbUbQbXbZbVbHbSagaVbebfbfbdbcbcbbbabababdbhbpbnbhbcbcbgbibdbb",
    "bbbibabebcbcbcbfbbbbbbbcbZbYbbbcbVbZbabmbgbUbTbYbPbdbSbSbKbebEbObWbHbScwbebhbgbebcbabababbbdbhbobsbkbcbebkbmbgbcbgbjbzbubmblbncBcEbrbocHb/cR",
    "e0bqcUdydLc3dcctdlcTb/c4ctdkbtccdLcnd7eEd7eheCbebobkbebcblbrbtbub/cAcLc1c0cKcUcRcFc3cBdycTdTcNeCdocmepfDe8dYf4dHesdecbe+f2ekfpezfAeTdffWcCfT",
    "eYdreZdQd3e9cIfWbpfnbec3blc2cYeBcGcCcidldpcodkdvend1dkeng4f9exgXfkfhfqfMdLh2gie6fcfegHdhf+gUgOh7d+gSf/g7fTfNfPeLg4gbeddfeDgvfvgVezeYgFbecWcA",
    "dReFf5freLf+ffe8f/e/gKfUfod/gGfdggfohLhBhKhag+gJhQhOi1gzecgCglgOfWgnezglf/gEerhWgYhogMg4gch6gth2gSgtgShfgeh9bec+d+gbh6gjf+gsgseYgmhhhshwhsi7",
    "ighbiEimibj6i3hhiGhxhEhdhsh+gGhdiBhrgQflhSh0hwhTj6ili7iQhzfghIglhBhti6hBgvi9iYhpiTbedQg7hOi7iCgRiFiOjEjMiqjDiSjiieiai+i/gphBixgYg0iMiuiSimiL",
    "jKiBj8j1ioiojEimhjh+isirj6jdish/hTiYiFjqiuimiZgVgrjLhzkCbebebebebebebebebebebebebdbdbdbdbdbdbdbdbdbdbcbcbcbcbcbbbbbbbbbbbabababababZbZbZbZbZ",
    "bZbZbZbZbZbZbZbZbZbYbYbYbYbYbYbebebebebebebebebebdbdbdbdbdbdbdbdbdbcbcbcbcbbbbbbbbbabababababZbZbZbZbZbZbZbZbZbZbZbZbZbZbZbYbYbYbYbYbYbYbYbY",
    "bYbXbebebebebebebebdbdbdbdbdbdbdbcbcbcbcbbbbbabababababZbZbZbZbZbZbZbZbZbZbZbZbZbZbZbYbYbYbYbYbYbYbYbYbYbXbXbXbXbWbWbVbebebebebdbebdbdbdbdbd",
    "bcbcbcbbbbbbbabababZbZbZbZbZbZbZbZbZbZbZbZbZbZbYbZbYbYbYbYbYbYbYbYbYbYbXbXbXbXbWbVbVbTbSbPbNbebfbebebdbdbdbdbdbcbcbbbbbbbabababZbZbZbZbZbZba",
    "bZbZbZbZbZbZbYbZbYbYbYbYbYbYbYbYbYbXbYbXbXbXbWbWbVbUbUbSbSbMbObIbHbebgbfbebebdbdbcbcbbbbbabbbbbbbbbbbabZbZbababababZbZbZbZbYbZbYbZbYbYbYbYbZ",
    "bYbYbYbYbXbXbXbXbWbVbWbUbUbSbSbSbKbIb4aYbebjbibgbebdbbbbbabababbbebebfbdbbbabababbbabbbababZbabZbZbZbZbZbYbZbYbabZbYbZbYbYbYbYbXbXbUbTbSbTbJ",
    "b2adaFZPYoWnW2bebrbobhbdbbbababababbbdbibibfbbbabdbhbebbbabdbbbababbbabbbZbZbabZbabZbababYbZbZbZbXbVbUbYbNbNbGa+ama0ZjY/a0YjWbVYbec7btbdbabe",
    "bibkbjbhbhbhbibgbebgbkbnblbfbdbabfbbbcbcbcbbbhbabZbcbYbVbUbYbYbebTbbbXbebebQbXbTbfbObYbObFb6bBb9b8aoaxbecUbpb+brcAcCcCc3bpbfbjbzbxbhbnbwbnbs",
    "brbhbgbpbgbabibsblbhbhbababgbfbkbsbfbZbgbbbmbubbbmbmbsbYbYbYbbbUbYbfbdbibcbxbeclcMcFcCcRcBc9bsblc2cKc1bwcJbnbwc0bvbebsblbrbrcFc0blcMc8cNbgcK",
    "cNbkbvbpbwbhbpbjbhc7bobnb/bjbmbhbnboc9bnbvbhcLcDccbecodRcVdAcwczcvctcncXcXbwb/cDcJbpbzbgbfcObsbucgc1c6bocZcGcPbrcLc5b/cxbncEc7bwcMbkcjbmb/cM",
    "cZcRcGbhc0cgchbvcPcOcfcnbebfbfbfbebebdbdbdbcbcbbbbbbbbbbbbbabZbZbabbbbbabZbabZbZbYbYbYbZbZbYbXbYbZbXbYbZbYbXbWbVbWbUbTbVbSbSbNbTbNbQbHbEb9be",
    "bfbfbfbebebdbdbdbcbcbbbbbbbbbbbbbabZbZbabbbbbabZbabZbZbYbYbYbZbZbYbXbYbZbXbXbZbYbXbWbVbWbTbSbUbRbSbMbTbMbObGbCb6bebfbfbfbebebdbdbcbcbcbbbbbb",
    "bbbbbbbabZbZbabbbbbabZbabZbZbYbYbYbZbZbYbXbYbZbXbXbZbYbWbWbVbVbTbSbTbRbRbLbRbJbIb9b2asbebfbfbfbebebdbdbcbcbcbbbbbbbbbbbbbabZbZbabbbbbabZbabZ",
    "bZbYbYbYbZbZbYbXbYbYbXbXbZbYbWbVbUbVbSbRbRbPbObHbMbBb3apaYaAbebfbfbfbebebdbdbcbcbbbbbbbbbbbbbbbabZbZbabbbbbabZbabZbZbYbYbYbZbYbYbXbYbYbWbXbZ",
    "bYbWbVbTbTbPbObMbLbGb3awaPZ+YrXmWJbebfbfbfbebebdbdbcbcbbbbbbbbbbbbbbbabZbZbabbbbbabZbZbZbZbYbYbYbZbYbYbXbXbYbWbWbYbXbUbSbPbMbGbBazahaKZTYbXn",
    "WmUeSpQwbebfbfbebebebdbcbcbbbbbbbbbbbbbbbbbabZbZbabbbbbabZbZbZbYbYbYbYbYbYbYbWbXbXbUbUbUbUbNbIbCb1amaOa4Z1Y1WxUyTkSIQIOBMlbebgbfbebebdbcbcbc",
    "bbbbbabbbbbbbbbbbababababbbbbabYbYbZbYbYbYbXbYbXbXbVbWbVbPbPbLbGbAaraWZ+ZWYNXfVcUyUYVVSqPlPyMnLhbebgbfbebdbdbbbbbbbabababbbcbdbcbbbabababbbb",
    "bababZbYbabZbZbYbXbYbYbVbRbUbSbLbGb6a+aha7aEZPYrX7X6X3XRW+X1V+YTYqXXXubebhbfbdbdbcbbbbbababbbbbebfbfbcbababbbbbbbbbcbabZbabcbZbbbWbWbUbWbObH",
    "bIbMb3aha/aAZlZ/YuYyYHX0ZXY3XHWqWvY2Z6WTYhZubebkbfbbbcbcbababZbbbdbfbjbhbcbbbebhbibdbbbZbcbWbZbbbZbVbXbObKb9b1anaRakbIaOaVZma2ZwZ8Ywb2ZXa7Zu",
    "ZOZxbGaAaZZuadaXaMbebqbgbabdbfbcbcbdbfbkboblbabZbibhbcbcbhbYbWbXbRbXbXbXbSbSbSbQbObBbDb3a/b3azb1atbAb0bAazb3bFbAb0b0bBb3bJbCbEbDb6azbibibibi",
    "bibhbhbhbhbgbgbgbfbfbfbebebdbdbcbcbbbbbbbabababababababababZbZbZbZbZbZbZbZbZbZbZbZbZbZbZbYbYbYbYbXbXbWbVbUbibibibibhbhbgbgbgbfbfbebdbdbcbcbb",
    "bbbbbababababababababZbZbZbZbZbZbZbZbZbZbZbZbZbZbZbYbYbYbYbYbXbXbWbUbTbRbObKbEb5bibibhbhbgbgbfbebebdbdbcbbbbbbbabababababababZbZbZbZbZbZbZbZ",
    "bZbZbZbZbZbZbYbYbYbYbYbXbXbWbVbTbSbPbLbGb8ayaca5ZPYXXcbibibhbgbfbebdbcbbbbbbbababababababababZbZbZbZbZbZbZbZbZbZbZbZbYbYbYbYbYbXbXbWbUbTbRbO",
    "bJbCb2aoaKZiYuX/XCWXVxVRU/UbbibhbgbfbdbcbbbababababbbbbabababababZbZbZbZbZbZbZbZbYbYbYbYbYbXbXbWbUbTbQbNbJbBb1aiaFZXYdXlWyWIVkVGUqUSU6TnTUTB",
    "SvbibhbebcbbbabbbbbbbbbbbbbababababZbZbZbZbZbZbYbYbYbYbXbXbWbVbTbSbObKbJb2anaJZdYeXYWfV/VSV0UdUGTxTeTLT4SpSZSJS4RrRcbibgbcbabbbcbcbbbbbababa",
    "babZbZbZbZbZbYbYbYbYbYbXbXbVbTbTbRbObFb5avaaZiYWX2VzVHUlULTzTdTJT1SlSVSER/RmRXRJR5QtQfQSQFbibebabbbcbcbbbababbbababababZbZbZbZbZbYbYbXbXbXbV",
    "bQbLbDbKb2araDYvX1VPUWU0TfTKT0SkSSSBRxRhRSREQ/QnQaQMQ9PyPlPZPMPAbibcbcbebbbabbbbbabZbababababZbZbZbYbZbYbXbabVbTbPbQbYbLavbxbTanakZyaCaIZpaL",
    "Z4bZcZbNbMZ/b6bsbecDfYdmdieZfbgTe/etfxbibcbibhbdbdbcbabebabebibcbebvbgbgbhcDcneNerfXfqfdfre3fefogzgkfcftgZgbgAetg4hLgHgLh9gHgvgahBf1fih7gLhS",
    "gEgFg6hNgefcbibmcAbhcOdEd6eKeoeefehPd4gdgUg/gRhUgaezgfgSh2hngsgigng7eng4gwhFhjfAi/iEgegvg/hmgThGgBg8hFg2hUgWhZhLh4gZi3h4gDhEdEbic4cmfIg3hNhL",
    "gYhJiXichaiMjAj5iJjBj1hXjjjXiohChBgyiFgHgghTiAiDj5i+hThnhAijhtgIgKh2i9i+iXiRhzfPiqjJivj4jHiCici6ici5bibibibibibhbhbhbhbhbhbhbhbgbgbgbgbgbfbf",
    "bfbfbebebebdbdbcbcbcbbbbbbbabababababZbZbZbZbZbZbZbZbZbZbZbYbYbYbYbXbWbVbTbibibibhbhbhbhbgbgbgbgbgbfbfbfbebebebdbdbdbcbcbbbbbbbababababababZ",
    "bZbZbZbZbZbZbZbZbZbZbYbYbYbYbXbWbVbUbRbPbMbJbFbAbibibhbhbgbgbgbfbfbebebebdbdbcbcbcbbbbbbbabababababZbZbZbZbZbZbZbZbZbZbZbZbYbYbYbYbXbXbWbUbS",
    "bQbNbKbGbAb4ayanaaaIZubibhbgbgbfbebdbdbcbcbcbbbbbabababababababZbZbZbZbZbZbZbZbZbZbZbZbYbYbYbYbXbWbVbUbRbObLbHbCb6a+arafaPa0ZWYvYGXeX0WObibg",
    "bfbebcbcbbbababababababababababZbZbZbZbZbZbZbZbZbYbYbYbYbYbXbXbWbUbSbPbLbHbBb5ayaoabaKZwZNYjY2XNWkW8VaV3UbU9TjbibfbcbbbababababababababababZ",
    "bZbZbZbZbZbZbZbYbYbYbYbYbXbWbVbUbSbPbKbFb7b0arafaRa5ZbYzYFXYWuWIVlVGUpUOT+TcTFSvSaSFbibdbababbbbbbbabababZbZbZbZbZbZbZbZbYbYbYbYbXbXbWbVbUbS",
    "bPbLbFb9b1awaiaOZvZCYSXkW/WLVpVKUvUWU8TpTUTBSuScSKS3RoRYRGbibbbcbcbbbabababababZbZbZbZbZbYbZbYbYbYbXbXbVbVbSbPbLbHb7a/agaGZgYwXvWuW4VQUuUQU0",
    "TeTJS/SjSSSCRzRkRWRJR6QvQiQVQHQ3bibfbfbdbcbbbabababZbZbZbZbZbYbYbXbXbWbUbSbQbMbIbDb7a+ajaUa3Z6XeWdVoVFUpUSU8TsTZTIT2SpSaSNSARzRnRbRPRER3QuQi",
    "QWQKQ6bibwbfbjbdbhbbbcbdbbbZbbbZbXbWbUbSbQbKbCb0auaaaNa7ZaZ8YQXOWPVUUrUMU0TiTRTDT1SrSgSVSKSAS0RtRkRbRSRIR8Q/QrQfQSQGQ3PsbicLcGbobvbhbubqbibj",
    "bdbZbYbUbTbNbIbIbFb6asamadaGZtZkZWZ6YyYga3apYPXoY8WaUST/TQT1SsSmSjScSVSSSLSDS5RyRmRcRNR8QuQfQRbicmcsbvcdcScYbwcEcAbfc7blblbmbbbZbYbXbQbJbSbJ",
    "bCbHb4b0bTbHb4b1awbJbibcbPbQbTbYbSbNbVbLbebIbJbQbKbPb7bHbdbGbLb8bFbYbibhbabcbdbcbababbbbbabababZbZbZbZbYbYbYbXbWbVbUbTbTbPbMbLbNbHbBb6b4arac",
    "aMaFZqZcZ6YpYNXxXMWvWXWAVuVZVIV1UtUZUJU2TnbibhbabcbdbcbababbbbbabababZbZbZbZbYbYbXbXbWbVbUbTbTbPbMbLbNbHbBb6b4apabaKaDZlZTYzYUYJXVWzWZW6VjVL",
    "V0UhUOU5ToTVTBSsbibhbabcbdbcbababbbbbabababZbZbZbZbYbYbXbXbWbVbUbSbSbPbMbKbNbGbBb5b4apaXaGa2ZaZBYgYBXfWyWRVrVKUrUNTwTWT8SmSRS7RpRVbibhbabcbd",
    "bcbababbbbbabababZbZbZbZbYbYbXbXbWbVbUbSbSbPbLbKbObGbDb4b3akaPa5ZvZGYqYAXaWpWCVPUjU1TMSiS4RVR0QcQHQ4PuPkbibhbabcbdbcbababbbbbabababZbZbZbYbY",
    "bYbXbXbWbVbUbSbRbObKbJbNbFbCb1auaaa8ZlZEYbXpWuVyV1U5T5S6R5Q3O/NzM+M9LWL7L5LOM4bibhbabcbdbcbababbbbbabababZbZbZbYbYbYbXbWbWbVbTbRbRbMbJbHbLbB",
    "b3asada6ZZYvY3X2VvUjTSS3QWOqMyKxIvH7FqEHDRC+D6DjEhGDbibhbabcbdbcbabZbbbbbabZbabZbZbZbYbYbYbXbWbVbVbSbQbNbJbFbBbBawagaFZiY+X+WhVFTeRpPnNULAIm",
    "GREJCJAV9R949B9lAUAvCBDTFAbibhbabcbdbbbZbZbbbbbabZbZbZbZbYbYbYbXbWbVbVbSbQbMbHbCb2atafa7ZUYWXLVcTjRrPnNoLdJaHNFRDzDFCRB5AeAvBdCeCDD0EIFtHfJM",
    "bibhbabdbdbbbZbabbbbbZbZbZbZbYbYbYbXbWbVbTbTbNbHb9a/aqaTa0ZNXuWCUISaQvP4NQL+KSJ6HiGXFhFIFCFWG6GcHHH1HLIJJRKLKVKkL0bibhbabdbcbabZbabababZbZbZ",
    "bYbYbXbXbWbUbRbMbMbBb0apaRZqYsXeVyU5SYREQCPDOMNjN6MjMGLHLQLCLTLRLBKXKeKeKrKQKMKpLILfLULEbibgbbbdbbbabbbZbZbZbYbabXbYbVbTbQbPbKbCb0ataQZzZ8Xs",
    "VwUBS+R+RCQdQ7PnPPP6OnOhOCO4NrNgNdNDNOZUXnXNWdUyY8W4SAUQUXSHT3bibhbdbdbbbbbabZbXbXbVbZbVbSbMbIbDb4apaVZtZSYbXVWVUuT+T6SWR/RXR1QhR0aUYMXMWdXy",
    "VdYKZna4b0YxYeaRa7Zba6YqZJWdWHaIa3aTbdbdbdbdbcbcbcbcbcbcbcbcbcbcbbbbbbbbbbbbbababababababZbZbZbZbZbZbZbZbZbZbZbZbZbZbZbYbYbYbYbYbYbYbXbXbWbW",
    "bVbTbRbPbNbdbdbdbcbcbcbcbcbcbcbcbbbbbbbbbabababababZbZbZbZbZbZbZbZbZbZbZbZbZbZbZbYbYbYbYbYbYbYbXbXbWbWbVbTbSbQbNbLbIbFbBb6a/bdbdbcbcbcbcbcbb",
    "bbbbbbbbbababababZbZbZbZbZbZbZbZbZbZbZbZbZbZbYbYbYbYbYbYbXbXbXbWbVbUbSbQbObMbJbGbCb7b1avafa8ZLYQXQbdbcbcbcbcbbbbbbbababababZbZbZbZbZbZbZbZbZ",
    "bZbZbZbZbYbYbYbYbYbYbXbXbXbWbVbTbRbPbNbKbHbDb9b3azanaPZlYqXmWqW4VUU/UaUBbdbcbcbcbbbabababZbZbZbZbZbZbZbZbZbZbZbZbYbYbYbYbYbYbXbXbXbWbVbUbRbP",
    "bNbKbHbCb7b1b0akaRZsYoXoWwWCVaV3UeUFTuTaTHS/SkbdbcbbbbbabZbZbZbababababZbZbZbZbYbYbYbYbYbYbXbXbWbWbVbTbRbObLbKbFbFb8b5anaPZuYiXNWVVpVHUsUUU7",
    "TpTWTESzSiSSSDR+RmRXbdbcbababZbZbababababZbZbZbYbZbYbYbYbYbXbXbWbVbTbRbPbNbNbLbBb7ataeaGZdYbWzVyVJUpUPU2TjTPT7SsScSMS7RvRhRTRGR3QsQfQSbdbbbZ",
    "bZbabababZbZbZbZbZbZbYbYbYbYbXbWbVbTbSbPbObLbHbFb1awaRa/aNZOaZXkVUU0TKSzSfSOS8RuRfRQRCQ+QmQZQMQ9PzPmPaPOPDP1bdbababababZbZbZbZbZbYbYbYbXbXbV",
    "bVbUbUbebTbVbUbLbQbMb7bRbRb1bRcQa+bnaMaLafaSapaxascbbMcYe3eDeGdie2fXeDevg1gAeXdtdmbdbabcbbbZbabababYbcbYbcbabbbXbYbnbhbhbVbSbkdhemcsg9gQgdgX",
    "gRfielc+fDfph0fzgFhNhMhFflhEf+fygtfndagPeFgegze+g3fHhKhSbdbfbgbnbdbwc/dEcceqg6gdgChcgUhUiRgbiIi9hPhMhFgbhWhVgTisggf6hhhvdsh9gthNhwgwi6gJhigE",
    "fnf4i5f5iWg1hkh9hxfpgzh/h0h6hQbdbycFdneehHhrhLhVh+i0hni6iGiahyj8jFhriJhgiBi7hlj1jBjNi8hnipiSirhjgMgkj2hojTjYiRggiAidk1iWiSi4hYiNiWiNglhjhqj0",
    "jKi/bdbdbcbcbcbcbcbcbcbcbcbcbcbcbcbcbbbbbbbbbbbbbbbabababababababZbZbZbZbZbZbZbZbZbZbZbZbYbYbYbYbYbYbXbXbWbVbUbTbRbQbObdbcbcbcbcbcbcbcbcbcbb",
    "bbbbbbbbbbbbbababababababZbZbZbZbZbZbZbZbZbZbZbZbZbYbYbYbYbYbYbXbXbWbVbUbTbRbQbObMbKbIbGbDb9bdbcbcbcbcbbbbbbbbbbbbbababababababZbZbZbZbZbZbZ",
    "bZbZbZbZbZbZbYbYbYbYbYbYbXbXbWbWbVbTbSbQbObNbLbIbGbDb9b4a/atakaZaLbdbcbcbbbbbbbabababababZbZbZbZbZbZbZbZbZbZbZbZbZbYbYbYbYbYbYbXbXbXbWbVbUbT",
    "bRbPbNbLbJbHbDbAb6b0avamabaNa6ZkZIYqYHXibdbcbbbabababZbZbZbZbZbZbZbZbZbZbZbZbYbYbYbYbYbYbYbXbXbXbWbVbUbSbQbPbNbLbIbGbDb9b4a/aualaZaKa0ZbZ2YR",
    "XpXDWfW6VbV6UebdbbbabZbZbZbZbZbZbZbZbZbZbZbYbYbYbYbYbYbXbXbXbWbVbUbTbRbPbNbLbJbGbEbAb5a/awapafaSa9ZnZ7YQXgW+WNVqVKUtUSU2TgTJSyScbdbabZbabZbZ",
    "bZbZbZbZbYbYbYbYbYbYbXbXbXbWbVbUbTbRbPbNbLbIbHbCb8b5azawaiaVaFZqZHYbXtX9WVVvVPUwUVU7TmTQT6SoSWSERyRhRQbdbabcbabZbZbZbZbYbYbYbYbXbXbXbWbWbVbU",
    "bTbRbQbNbLbKbGbDb6b3avakaZa2ZaYiY0WwWCVfVBUoUQU5TnTVTDSzSjSUSGS2RrReRRRER1QpbdbfbabbbababZbZbYbYbYbXbWbWbVbVbTbSbQbRbLbKbFbAb6auanaTa2ZdZAYB",
    "X2VyV6UYU9TqTXTGT1SoSbSPSDS2RtRjRZRORER4QvQkQYQMQ9bdbtbbbkbZbdbebbbZbcbYbcbWbXbUbTbQbPbNbIbAb4akaNa6ZVY+YeXpWkVwVAUdUDTuTZTIT4SuSjSXSLSAR/Rs",
    "RiRZRPRER5QxQnQdQSQEQ1Pqbdc8cFbbc7bmbhbZbZbYbUbTbQbJbHbEb6a+avagaMaMZvZXZSa0Z5YBWXVSUZU5ToTeTXTRTLTDT7T2SzSsSiSYSSSPSISDS2RrRiRZRORDR1QqQcbd",
    "cPdlcCcDcRc9bqbubebqbbbYbObUbKbJbMbJbBb6b5a+azatbDaya+apasagawbHb3bHb1bBbCbcbQbYbRbQbbbcbgbNbfbibrbkbbbgbubhbnbabdbebZbbbbbZbZbZbZbZbYbZbYbX",
    "bXbXbWbXbXbVbTbTbQbPbNbMbMbNbIbFbDbBb7b6awanaaaMaFZkZWZQYoYGXrXDWhWFVwVZVHV0UkUUUET/TmbdbebZbbbbbZbZbZbZbZbYbZbYbXbXbXbWbXbXbVbTbTbQbPbNbMbM",
    "bNbIbFbDbCb7b6awamaaaLaFZiZRZOYhY7XcWvWKVpVPV0UgUOU4TpTZTHS+bdbebZbbbbbZbZbZbZbZbYbZbYbXbXbXbWbXbXbVbTbTbQbPbNbLbMbNbIbFbCbCb6b5avamaaaIaIZg",
    "ZIZ9YNXjX9WKVjV5UbU7TjTMT0SiSPS5RnbdbebZbbbbbZbZbZbZbZbYbZbYbXbXbWbWbXbWbVbTbTbQbPbMbLbMbMbHbEbAbCb4b2asajaZaHaKZjZ2YtXVWcVpV0UETVSpS9RZR3Qf",
    "QNQAQ1PubdbebZbbbbbZbZbZbZbZbYbZbYbXbXbWbWbXbWbUbTbUbQbObMbKbLbLbGbCb8b8a/awafaSaEZxZbZCYBXHW9V3T/SwRoQgPYOSNQMYLoLIL6LFLobdbebZbbbbbZbZbZbZ",
    "bZbYbZbXbXbXbWbWbWbWbUbTbUbPbObLbJbKbIbEb8b2a+amabaEZtZJYhXjWMV1TeS9QXOqM+L0J2HDFjEBDICsCsDNEPG0bdbebabbbbbZbZbZbZbZbYbZbXbXbXbWbWbWbVbTbTbT",
    "bObNbLbHbGbBb6azaoaaaIZsZBYBWnVDTZRePbNIKpIDG1DoBuAL9X9YA5AwBiCEDLEWGJbdbebabbbabZbZbZbZbYbYbYbXbXbWbVbVbVbUbSbRbPbLbJbHbBb7a+araTZzZGY0WZUq",
    "SzR7PHNOLJJ9G/F5D5BdAZAHAjBKBqCBCjDmEXFpHeILbdbebabbbabZbZbZbYbYbYbYbXbXbVbUbUbTbRbPbMbLbFbCb5axaiaHZhYaWyVBTLRbP+OIMfLAJkIDH6GMG7G7E2DyE5EB",
    "EeF4EAEgFoGwIAJOKUbdbdbbbbbabZbZbZbYbYbXbYbWbWbUbSbRbPbMbKbDbEb0apaTZ/ZAXsW4UESeRQQ7P4O6MyL+L2KQJwJQJAImIfHsHuIMIuIIHJHYHFI2IMJXJzKibdbdbdbb",
    "bababZbZbYbYbXbYbWbUbSbQbPbMbJbEb0b0aUZpZ8XkVlTsSVRSQlQ2PGOgOCNiNIN0MyMpMjMgaTXBX2V3UFUsZxYWU/SdWVVjYqY9UsbdbebbbcbYbbbYbZbXbXbVbUbRbPbKbFbD",
    "b8axahaBZkZ4XMWXVCWSWiXdWWVvVYTkVhWna5YTY1VUY4XUZFVwTyZhaEZgZCYQWCaNYhZqacZ9ZraCbebebebebdbdbdbdbdbdbdbdbcbcbcbcbbbbbbbababababZbZbZbZbZbZbZ",
    "bZbZbZbZbZbZbZbZbZbYbYbYbYbYbYbYbYbYbXbXbXbXbWbWbVbUbTbebebebdbdbdbdbdbcbcbcbcbbbbbbbababababZbZbZbZbZbZbZbZbZbZbZbZbZbZbYbYbYbYbYbYbYbYbYbX",
    "bXbXbXbWbWbVbUbTbSbPbMbJbEb7bebebdbdbdbcbcbcbbbbbbbabababZbZbZbZbZbZbZbZbZbZbZbZbZbYbYbYbYbYbYbYbYbYbYbXbXbXbWbWbVbVbUbSbQbNbKbGbAb0amaJZdYg",
    "XjbebdbdbdbcbbbbbabababZbZbZbZbZbZbZbZbZbZbZbZbZbYbYbYbYbYbYbYbYbXbXbXbXbWbWbVbUbTbSbPbMbJbEb7axaca3ZBYBXIWbW0VXV5UibebdbdbcbbbababZbZbZbZbZ",
    "bZbabZbZbZbZbZbYbYbYbYbYbYbYbXbXbXbXbWbWbVbUbTbSbPbMbIbEb6avaca8ZMYCXBWOVlVEUnUOU1TiTPT7SrbebdbcbababZbZbZbababababZbZbZbZbZbYbYbYbYbYbXbXbX",
    "bXbWbVbVbUbSbQbObKbGb6axabaIZoZ5YDXAWHVdV5UeUETuTZTFSzSiSRSBRyRibebcbabZbZbababababZbZbZbYbZbYbYbYbYbYbXbXbXbWbVbUbTbSbPbNbIbDbAataha5ZWYXXI",
    "WFVUUxUUU5TlTRT8SsSbSKS4RrRdRPRBQ+QnQabebbbZbabbbbbabZbZbZbZbZbYbYbYbXbXbXbWbWbVbVbSbQbebXbQbJb9ayasa4b5ZsZ8YGYIXHULT9SdSFRxRfRPRAQyQkQXQKQ7",
    "PwPkPYPMPAO+bebbbabcbbbZbZbabababZbZbZbZbZbYbXbWbVbVbQbPbKbObCb4asbebdckcDcqctcIbNbcbAbZbJadbAcTcwcyfVcXeLg5fpgNfzeBfPeefJdleDbebfbgbdbebeba",
    "babcbbbZbcbYbbbrbzbfcHe7fngAfrfogPg5eBgSgghCgTgSgOf7gLglg0fZg4hugKfyg8emgegLg2eJg9gNgdh5g5f+hBhDcke8bebubvcxd4f5gKfcfOfff1ggh8gti1gVh7gphchA",
    "fWiAhiiui1hri2iBiGhYhxgdgohwhWgThGfSfMiMgshZhmiThZhdhrhWi0g1g8hLhNfMfTflh8becHcPfpfKh5h9g0ephphMiZhXhXhUeYg+hbhsiQhUfNinioi6grhYiPixiGhthria",
    "iSjEiwhKi3jWjPhOicj7j8iki/gZisiQg/izhRi7h2hBgphEbebebebdbdbdbdbdbdbdbdbdbdbdbcbcbcbcbcbcbbbbbbbbbabababababZbZbZbZbZbZbZbZbZbZbZbYbYbYbYbYbY",
    "bYbYbYbXbXbXbWbWbVbUbTbebebdbdbdbdbdbdbcbcbcbcbcbcbbbbbbbbbabababababZbZbZbZbZbZbZbZbZbZbZbYbYbYbYbYbYbYbYbYbXbXbXbXbWbVbUbTbRbObMbJbGbDbebd",
    "bdbdbcbcbcbcbbbbbbbbbababababZbZbZbZbZbZbZbZbZbZbZbZbYbYbYbYbYbYbYbYbYbXbXbXbXbWbVbUbTbRbPbMbJbGbCb7b1auakaXaJbebdbcbcbbbbbababababZbZbZbZbZ",
    "bZbZbZbZbZbZbZbZbYbYbYbYbYbYbYbYbXbXbXbXbWbWbVbUbSbQbNbKbHbDb8b2awalaYaJa1ZjZLYwYRXubebcbbbababZbZbZbZbZbZbZbZbZbZbZbZbZbYbYbYbYbYbYbYbYbXbX",
    "bXbXbWbWbVbUbSbPbMbKbHbDb8b2avajaUaGa0ZiZOYvYMXkX7WWVyVRUybebbbabZbZbZbZbZbZbZbZbZbZbYbYbYbYbYbYbYbYbYbXbXbXbWbWbVbUbTbRbPbMbJbFbCb7b1awalaT",
    "a7ZkZGYiX/XKWlWIVpVNU+UbUETtTXTBbebabZbabababZbZbZbZbZbZbZbYbYbYbYbYbXbXbXbXbWbVbVbTbSbQbObMbIbIbFbAb4azalaSaDZhZ0YTXhX5WSVyVWV2UeUGTwTbTHS+",
    "SiSQS8bebbbbbbbabZbZbababZbZbZbYbYbYbYbYbYbXbXbWbWbUbSbRbPbMbLbEb8b1asaXa8ZWYhXiWlW2VTU/UbUDTtTZTHS/SkSVSGS1RpRcRORBQzQlbeblbfbbbbbcbcbabZbZ",
    "bababZbYbZbZbYbYbXbWbVbUbRbObNbMbFb6awaba3ZEXvWkVpV7UdUETtTXTFSzSiSSSES0RoRcRPRDR1QrQfQTQGQ2Ppbec4blbqbhbbbcbdbbbZbYbZbYbWbUbTbQbObLbGbCbFb0",
    "aragaSa1Z8YUXRWOVgV2UbUFTxThTSTDS+SmSZSNSDS3RuRjRVRKR7QxQmQbQPQEQ2PqbecMcabrbdbnbcbdbXbZbYbTbNbMbFbAb7b3b0ahabaIa5ZhZ6YcXuWxVtV4UdUIU4TuTkTZ",
    "TOTFT7T0SvSmScSNSES6RzRlRbRPRAQ+QnQZQMQ8PvbecYdbckcJc1b+btbjbkbdbbbgbZbibZbWbUbQbObRbMbGbGbEbKbBb2b0ataranbLbcbBaqaeakakaRb1aybGbMbHb9bQbQbK",
    "bibAbPblbablbebgbebibabcbdbbbababababZbZbZbZbYbYbYbXbWbWbVbUbSbRbQbPbPbObKbIbEbEb9b8b1awajacaBZzZiZHYqYUXrXMW/WTWBVoViVGU/UlUYUIT/bebibabcbd",
    "bbbababababZbZbZbZbYbYbYbXbWbWbVbUbSbRbQbPbObObJbIbEbFb9b8b0avahaba8ZyZdZAYfYNXaX1WVW1VeVGUyUeUMU4ToTWTDbebibabcbdbbbababababZbZbZbZbYbYbYbX",
    "bWbWbVbUbSbRbQbObObObJbHbEbEb9b5a+a+afaZaJZoZSY+YVXiX4WRVvVMUuUSU2TgTKT0SjSQS5bebibabcbdbbbababababZbZbZbZbYbYbYbXbWbWbVbUbSbRbQbObObNbIbGbE",
    "bEb8b2axb7aYaVa8ZbZ4YZXiWsW9VQUmU3TRSsSJRrRSR8QsQdQObebibabcbdbbbababababZbZbZbZbYbYbYbXbWbWbUbUbTbRbQbObObNbIbGbEbCb5a/asavaNaFZgY/Y8XNWGVK",
    "UMTPSWRYQgPrP3OQNvNZNPNRNgbebibabcbdbbbababababZbZbZbZbYbYbYbXbWbVbUbUbTbQbPbObObNbHbFbCb8b1atagaUZwZTYcXNW6UkTGRkQ7OOMbKhIrHHG3ErEOEIEkFdGz",
    "bebibabcbdbbbababababZbZbZbZbYbYbXbXbWbVbUbTbUbQbObNbMbLbGbEb6b0arada6ZKYEWsVITXRgPZNIKlIAFtDYBW9w9H969VA2ASBFCNDcbebibabcbdbbbababababZbZbZ",
    "bZbYbYbXbXbWbVbTbSbTbObMbLbHbHb7a+akaPZpYuXSVdTjRmPiNNKyIHFuDhBcA49E989cAGAyBOAtBdCSDDEVbebibabcbcbbbababababZbZbYbYbXbXbXbWbVbTbRbPbPbHbCb9",
    "azanaFZSYJWmUgScQmOrN5LfK7IbH9FxE+DwDPCeCPCmDWE8EAExG8GbGtGGI3bebibabcbcbabZbabZbZbYbZbYbYbXbWbVbTbRbObIbIbCazamaRZxYqXaVJTSRtQaPSOQNFM2KpJj",
    "IgIEHuHXHGGoGOG2GRG9FaGKHYInJQJmKoLWbebibbbcbbbabZbabZbYbYbYbWbWbUbRbObLbJb8awajaDZXYvXCVCTWSWReQuPsPCOaO5NtNeNSNFNINGNINDNAbZYTUoV3XXUuTqXl",
    "WdThXza7VobebkbdbbbcbZbZbabYbZbXbXbVbUbQbObHbBb0araNZqYhXKWHUoU3T9SLRqRNR4QpQkZMalYjX/X4XZVjX5YOWbYAaFYtVIbEYEaPZ8ZPXvasZPZr",
);

const SPECTRA_TABLE: &str = concat!(
    "0000dddD0000000000000000000000000000000000000000000000000000000000000000cOcJcHc8c5c0b+bxbvbpbhbbbYbPbObKb9b3avaoalahaVaIa2Z+ZgZOYtYfXnXT0000",
    "000000000000000000000000000000000000000000000000000000000000cPcHcEc9c5c0bxbqbobgbdbXbWbJbGbAb3a/avavasaiaUaKaHa2ZpZfZJZ5XqSt000000bnbTbHSKRW",
    "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",
    "IKHmHVGyGjG6G2FuFZEWEMEDCXC5BaAwclcMcKcHc7c0b+b+btbqbnbebYbSbPbObHb2azalaiagaPaGa2ZpZbZFZ2YgYCWqWpUMS8QkPKPDOWO4NoN3LnKsKiKSJuJkIVIGHlHiHJH9",
    "GmGIFxFREvEcEADWCG9jcOcJcGcCc7c4bxbrbqbjbebYbVbTbGbDbAb2auasamahaVaNaGa4ZnZbZHZ7XoSyWXUX6ibublbKViUReLdsWPV+V1TcSxSdSLRaPUP4OVO2NSMuMfM2LTLL",
    "K+KLKHJVIoIGHkHVHJGrF/FJEIC/drdPczcRcJcEcAb/bxbsboblbfbYbTbObJbCb6b3auaiaZaLa8a1ZWZLZIYmXCWLWgW8UcU2TFSuSNRuRORBPgPbO+OsOLN+NSMuMWMDLcLXLRKT",
    "KAJhIHIAHWH1GSDxdQd9cxcVcJcCc3c1b+brbpbjbabZbSbMbEb7b2aqapakaWaUaQa9ZwZdZBY+YEVMWHUk69dNd7bl0000dddD0000000000000000000000000000000000000000",
    "000000000000000000000000cOcJcHc8c5c0b+bxbvbpbhbbbYbPbObKb9b3avaoalahaVaIa2Z+ZgZOYtYfXnXT0000000000000000000000000000000000000000000000000000",
    "000000000000cPcHcEc9c5c0bxbqbobgbdbXbWbJbGbAb3a/avavasaiaUaKaHa2ZpZfZJZ5XqSt000000bnbTbHYAXedddDa2Z8Y6XtXZXKX4WuWhWWWSWNWFVdKHJpItHZGWFiEvC4",
    "C3BLB4AjAWAH98000000cOcJcIc7c6c0b+bybvbpbibbbYbPbObJbBb5avapakahaXaJa3a0ZmZWYyYYXZXIZxZNYEY2XaXLXBW+WjWgWYWWWFW9W0VwVtVhLwJWI0FmFJEiENChCTAP",
    "AOAF0000cPcHcEc9c5c0bwbpbnbgbcbXbWbNbIbEb7a/awasaqalaWaNaDa1ZqZbZHZ4XxPfYQXmWxbmbTbKY3WedddDZuYCXGWsWKV/VdVVVIV4UvUKTxTWT4S9RwRjREQwQaPWPJOm",
    "NYM/M0LbKGJPIfHTcPcJcIc7c6c0b+bzbvbpbibabYbQbObJbEb5avapanaiaYaJa4a0ZoZTYxYUXVXBZrXpXKWpWXW8VqVgVDV0UPUDTjTKSsSeS8RwR6QlQgPnPWPLOlOMNWM/LsLI",
    "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",
    "bCb6a/ayakaiaTaJaBa1ZwZmZPYyYfVpSMSANRdTdEbL0000dddD0000000000000000000000000000000000000000000000000000000000000000cOcJcHc8c5c0b+bxbvbpbhbb",
    "bYbPbObKb9b3avaoalahaVaIa2Z+ZgZOYtYfXnXT0000000000000000000000000000000000000000000000000000000000000000cPcHcEc9c5c0bxbqbobgbdbXbWbJbGbAb3a/",
    "avavasaiaUaKaHa2ZpZfZJZ5XqSt000000bnbTbHYAXedddDVCV5V2UsUTTqRgPl6n6a6R6H5w5q5g5J4s4r4m4Y3u3l3U000000000000000000cOcJcHc8c5c0b+bxbvbpbhbbbYbP",
    "bObKb9b3avaoalahaVaIa2Z/ZgZOYtYfXoXSV3V1UYUTUST+SUPz6V675n5m5O554d4Z3e3V0000000000000000000000000000cPcHcEc9c5c0bxbqbobgbdbXbWbJbGbAb3a/avav",
    "asaiaUaKaHa2ZpZeZJZ5XqT2YQXmWxbmbTbKY6WPdcdDVCU/UmUfUTTfTHSuREQEPyOoNVMVLEKr5u5r5X5H4o0000000000000000000000cOcJcHc8c5c0b+bxbvbpbhbbbYbPbObK",
    "b9b3auaoalahaVaIa3Z/ZgZOYsYfXnXTV9UoUlUUUTT+TXSVRBQFPjOyOJMWLbKr5+5c5I5F4+4C3n3Y0000000000000000cPcHcEc9c5c0bxbqbobgbdbXbWbJbGbAb3a/avavasai",
    "aUaKaHa2ZpZeZJZ6XqStYDW1UvbmbTbMXSX1dcdCVCV3V2U+UaUXUSUDSbRtRJRGR1QHPTP3P2OcNxNZMzMdM7LGKrKbK5JuIRHpGWFicOcJcHc8c5c0b+bxbvbpbhbbbYbPbObKb9b3",
    "auaoakahaVaIa3Z/ZhZOYrYeXoXVVBVAV2UsUoUYUYTgTESIRxRTQNQ5PiPKOvOcO9NsNPMnMPM7LlKUKKJpHiHDFBAacPcHcEc9c5c0bxbqbobgbdbXbWbJbGbAb3b0avavasaiaUaJ",
    "aHa2ZqZfZJZ6XqStXbXCTablbVbPX8WCdbdCV7V3UiUfUaUYUXURTlTOS1RyR9R4QzQHQ4PfPbOsOKNtNLN1MxMDLdKSKHJvJBITcOcJcHc8c6c0b+bxbvbpbgbbbYbPbObKb9b3auao",
    "ajagaVaIa3Z/ZhZNYpYfXtXXVBV2V1UuUhUbUXUJU1TOSPS0RkRgQwQpQNPXPAP4OPO9NlMzMpMeM5LqLEKrJVFJcOcHcEc8c5c0bwbqbobgbdbXbWbIbGbAb4b0avauasajaTaJaHa2",
    "ZqZgZJZ7XqTaXPW9T0blbWbOWYVKdbdCVCV8V3U/UwUrUrUaUSUGTsTASaSFRuRVQ+QjQZPjPcPDP4OXOGNwN0MWMGLlLFKJcOcIcHc7c6b/bzbxbvbpbfbabYbPbObKbAb3ataoaiag",
    "aVaHa3Z+ZiZNYkYfXwXaVAV6V6V5UxUoUhUeUNUDU1TKSkSYS6RjRTR1QaQLQ6PbPHP0OdOANWNTMbLyL2JIcOcHcFc8c5c0bwbpbnbgbcbXbWbHbGbBb5b0avauaqajaSaJaFa1ZrZh",
    "ZIZ6XmUPWfWOShblbYbAVyVIdYdDVCVBV5V5V1UmUmUbUaUWUWUSTvTrT8SlSXS4RfRTRGQhQYPvPaP6OyOPNgNWN2JDcOcGcFc8c5bzbzbxbsbmbdbabYbQbNbGbDb2arajagaZaRaH",
    "a3ZsZfZMYeYXYJX6VBV8V5V4UxUsUfUcUZUXUXUSUAU7ThT9SpSLS8RtRTRER1QXQDPoPIP1OmO7NaKpcNcGcEc6c4b+bvbobmbgbabXbTbHbDbAb0a+avatalafaOaKa9ZyZoZYZOZ2",
    "XeUaWZVhSZbkbFanVpTxdXdDVCV7V3V0UuUsUrUfUaUZUWUHTsTjTUT7SeSPRzRqR0QtQjQ7PyPGOvOFNpN5LmL6cLcHcDc6c0bxbxbsbmbfbcbYbUbPbObCb7b2axadaaaSaOa7a5Zy",
    "ZnZ4YpYCY4VKVDVBV8V3V2V1UxUtUiUgUcUUUTU7TpTASvSiSRRgRWR3QZQGPvPSPFP1OTO0MnMIcFcCc7c3c0bxbrbobibdbYbVbMbEb9b4b2avasapafaXaNa1ZwZtZfZcZ8YiXHWo",
    "VyUyR2bjbNa/WXV5dbdNVCV6V5V5UzUsUdUaURUNUATiTbT1ScSIR+ROQyQmQBPoPaPKOtOIO9NoNeMWLcKacUcNcGcFc1bzbvbobmbibebabTbQbLbBbAb0auaqaeaNaMZyZqZmZjZ4",
    "YrY9XGUwVDV7V2V2UtUmUaUZUXURUJUFTWT8SiSLS6RoRkR7QjQYQBPjPIO/OlOINQMkLxLHcGcDc2c0bwbsbqbjbibcbXbNbIbFb6b3a+atamaeaUaPaFa1ZrZdZPZDZ3YeXmVsWzVH",
    "RnbkbVb4VZV4e1dvV8V5V2U/U+U+UrUoUgUfUaUaURURURTzTUT0SUSDRtRHQrQXQCPSPHP2OLNKMEM4d7d2cqcdcKcFc7c6bqbobfbdbZbUbPbGb7b0ayaqapaeadaCa9ZrZZZ8YtYD",
    "XrVeVDV6V3V1U+UsUpUpUoUjUiUVUVUVUUTuTUTFSZS8RvRfRCQtQiPwPrOrO4M/MYLfcrcocOcGcAc0bvbpbkbababRbKbHbCb6b2auaoafaUaLaCa4ZxZoZdZCYtYPXwXDWGVkS3cT",
    "cHbaU0TkeNeIVDV8V8V5V1UpUoUjUbUaUXUWUSUSURURUGUFTWSdSNRvRgRGQcQ9PeOxOcNpMfK7drdddXdHdFd8cychcUcMcIbzbobhbbbWbMbCa+axasabaWaMa6Z/ZcZHY/XsXTUY",
    "VBVAV7V4V3V1V0U/U+UrUqUqUcUaUZUUUSUSURUKUHTaSvSYS0RRRIQnQ5P1NuM+dkdZdHdDd0cycecacScLc8c5c0bibgbYbRbMbHb6avaqaWaTaPa5ZtZXZ3YuYdUZUCT+PodmdPbY",
    "PuN4eOeHVDVCV7V7V0UoUkUXUWUWUVUSURTSSlS9R2QSQGPdOyOiN5MtLuLKL5KVKQIzHxF8eVeOeJd+dpdidRdNdDczcgcgcTcBc4bcbSbNbCb1auauafaTZuZpZaZDZCYGXJVWVDVC",
    "V8V7V7V4V0UyUyUvUuUtUpUoUnUiUaUaUaUXUVUSUSU8TrTkTTSlRvR5QdOreSeIeFeAdwdrdndfdWdRdHdCd3d1c/ctcfcZcTc4bxbkbabKb7aeaZZxZpZNYmWCQOPdMYdldVbM0000",
    "dWd50000000000000000000000000000000000000000000000000000000000000000cMcJcAc5c1bxbvbrbjbbbZbYbUbHbFb9b5b1auahaeaUaPaOa6ZnZZZJZ4YfY4Tb00000000",
    "00000000000000000000000000000000000000000000000000000000cQcLcFc8c7c3bybxbsbjbabZbQbMbJbFbAb6b0auamalaWaLaHZ+ZfZSYyYdX0Vr000000bZbRb6S/SLdWd5",
    "VJU1OoOUM0KyHSDb5+5p5U5N4s4Q4O3y3u3o0000000000000000000000000000cMcJcAc5c1bxbvbrbjbbbZbYbUbHbFb9b5b1auahaeaUaPaOa6ZnZZZIZ4YfY4TiVITfOKN9LMKA",
    "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",
    "IPILHVGxGNFlELBlctcecRcKcDc8c3bybxbrbmbfbZbSbMbFbAb6b4aoajacaZaHaFZuZcZ7YxYrXESQWcVM7RcVcFbKTST7e6e3UZUPTlR/RKQtQFQ1PqP6NaNCMvMcLRL2KnKSJnJF",
    "J7I5HTHBGpGOFJE/EVE4DcBkdndkdMdAcYcIc6b/bvbtbpbhbcbSbMbIbEb5b2awakaYaYaKa4ZmZhZPZHYVXtTBUwUETYSpRwRkR5QzQZQJPoPUP0OdOPNxNtMaMBLsLZKyKHJhJYJC",
    "IiHhGtGDErEYdfdbdKcwcacQcGc6c2bybubmbgbXbSbObEb9b2ayarahacaYaOa8ZwZcZCYtYRUuUFTy00dvdfbM0000dWd500000000000000000000000000000000000000000000",
    "00000000000000000000cMcJcAc5c1bxbvbrbjbbbZbYbUbHbFb9b5b1auahaeaUaPaOa6ZnZZZJZ4YfY4Tb00000000000000000000000000000000000000000000000000000000",
    "00000000cQcLcFc8c7c3bybxbsbjbabZbQbMbJbFbAb6b0auamalaWaLaHZ+ZfZSYyYdX0Vr000000bZbRb6X/XvdXd5ZzZLYPXnXVXGX6WxWqWdWOWHW7W2V+VlVjGkGTFBD8CvBVAP",
    "A3A1998/00000000cMcKcBc5c2bxbvbubjbbbZbYbTbIbGb8b4b4auajaYaXaPaKa5ZsZbZJZ2YZX+VRZ/ZIYEY0XQXJWpWbWaWHW5VsHXHJGKG4FgFGERDtDaBzBBAcAbAJ9w9k9U00",
    "0000cQcLcFcAc7c4bybwbtbjbbbYbRbNbJbEbAb8b2atamaiaVaHaEZzZfZRZ6YoX4UwYQXrWjbZbRb6XdX7dWd6ZSY0XIWaWFVjVGVBUeUHTqTKSjSVS9S1RnRaRCQ9Q1PJO/OUNuNc",
    "NQMcM4KIJeIhcNcKcBc5c3bxbwbubjbcbZbYbUbIbGb8b6b3auakaXaVaOaIa6ZqZaZLZ0YYXwVjZSYFXfX2WiWAV+VdVXU+UbUOUNTiTaSySAS4RkR3QrQDPlPTPGOVN/NbNKMQLgKD",
    "cQcMcFc9c7c5bybubtbjbbbXbRbObJbDbCb9b2atamajaVaHaFZyZgZNZ8YtXATdXcXXVkbYbPb7WtV/dVd5YbXoVzVWUXU9ThTRSzSxS6RuRnRMR1QvQUQ2Q0PRPKOwOSO0NbNKMdLs",
    "LdLDKsHqcNcJc8c6c3bybwbubjbdbYbWbTbJbEb8b7b1atajacaTaKaFa6ZmZYZMY/YZXxUPYjY6WoVkVMUlUbUDTmTdTJT4SkShSUSNRiRFRAQhQJPvPhPSP8ObN+NMMULeKwJpcPcM",
    "cGc8c7c4bybsbqbjbabXbQbPbKbEbCb8b2atakajaVaIaCZwZhZJZAYyXEVBXLWVU4bXbIb3WqVpdTd2YQXOVjUYU8TRTCSYSNSARkRZR2QzQOQBP+PvPqP1OqOcOQOANYNCMuLpLbL4",
    "KDJKcLcIc7c4c2bxbububjbdbYbVbQbJbDb7b3b0auaiadaSaKaCa1ZXZUZLYyYeXuUdYPXUX5VrVBUkTcTTTFT8SnScSMR/RdRPR7QoQcQ5Q0PjPQPHP0OhODNwNnMUMKINcPcLcGc9",
    "c6c1bxbrbnbkbabZbRbNbKbEb9b7b0ataladaVaHaCZyZdZIZ4YrXbWoX9WcUdbWb8axWdU+dSc/Y2WuUsUXTTTDTASPRzReRJR3QbQMQFQCPsPpPLO/OnOgOLNzNZNIMvMMLtLVKfId",
    "cMcIc7c3c3bxbvbpblbdbZbVbPbKbBb8b4a+avafaVaRaQaEZgZXZOZ5YzYhXbW6XoWfVyVGUGUATbTHS+SkSSSIRzRuRJRDQzQoQeQ7PyPvPYPGP8OvOFN/MwMuMEL1cPcLcGcAc7c0",
    "bwbrbnbkbdbZbPbMbKbGbAb5b1asaoaZaUaKaEZ/ZfZKZAYjYcWsWRUnTTbWa+auVkVbdPcxXVX7U5TYT0SvRxRnRcR4QwQYQIPtPlPXPKPCOnOfOVORO2NvNMNEMZMFLJL5JWIWcKcH",
    "c6c4c1bxbwbobibcbYbVbPbKbCb9b5a/atagaQaPaHaEZpZZZFZCYyYbYNWNXVWOUzUHToTfTCSzSgSQSIS0RiRIRBQtQiQKQCQ7PqPUPMPDOuOYOQO0NvNTMKLNcNcMcIc8c6b+btbp",
    "bkbhbbbYbPbMbKbEbAb0auaralaYaTaNa8ZzZsZLZIYsYQWsWAUaSCbWapaLWWVXdLcwXyWuWBTxThT9StSXSDRpRnReRYQnQgQZQ9PlPcPWP9P4OrOJOBNuNdNQMlMDLRJLcKcIcGc0",
    "b/bvbubsblbjbdbVbObNbDb8b5b2axajaaaVaMaAZ+ZsZLZ9Z4YLY6WPY0XJWqVUUMU5TzTeT8SwSoSiSJS4R/RvRoReRQRJQnQTQLQDQ4PqPWO+OOO0NaL/cPcIcEc9c4bybtbnbjbh",
    "bZbUbSbMbHbGb9b4asalakadaWaHa7a0ZjZXYoYoYeWwWPW8SabXahaUVDU4dLcjWnVbTxTRSTSHRpR9QfQLQBQ5PwPdP7O/OxOVORO2NzNgNZNQN1MyMiMTLZL0ItHUcMcIcBc2bybv",
    "btbrbnbebdbUbQbLbHb9b3a+avagaaaTaLaFZvZkZSZBYlYBXMWGW/VcUmU8TrTOSkSUSAS2RgRQR2QqQjQYQQQ8Q4PuPdPTPEO/OYOLO9NUN9MqLUFUcPcGc8b+bxbwbrbkbhbfbXbU",
    "bMbMbFbBa+axavanaiaYaOa9a2ZeZWZGYvYSXFWXVbUoQdbUaSZ/W/VVdgcrXiXDWOUWUATcTLS+SYSUSJRhRORBR2QvQgQVQCP+PaPQPFP2OdOTO8NgNcNDL/LBdFcwcScOb/bybpbn",
    "bkbhbcbZbTbObFb7b1azavamacaYaIa9Z/ZkZRZJYZXtXFVNY1XLWfVNV1UbUDTaTWT9T6SsSZSRS9RlRdRIRBR1QpQdQYQKQ1PgPLP7OjOdNVKbdDc/cUc8bybubtbobbbVbQbObIbE",
    "b9b6a+auanaiadaTaCa4ZuZiZZZ9YgXyWWVbWtVzRNbbbWbOWMVre3dMYQXPWjVfV0UfURU6T/TLTGT4ShSZSPSFS4RjRGQyQiQUQGQ4PhPLP4OoOXOTMcKse4d/dqdCcZc1brbpbobc",
    "bXbVbObNbBb5b3ayaqahacaVaQaHZvZfZVZRZ0Y4WeWHYHXxWlWRW6VnVMV3UtUgULU1TgTWTMTCS+SfSRSJSFS4RgRLR6QgQaQGPmOpNpMyd+dudUdNdEcrcNcCc3bsbmbjbWbSbHbG",
    "b4araiadaTaMaHaCZtZqZdZZZAYdY9VkXJVhQ+cuccbJS4QDe6dPTST2SbQiQdQNPxPbPXPKOoOeOHOANtNkNUN5M+MnMZMMLvLiLQL1KzKcK7JgJMGZeQeDeCdtcqcHcAbzbtbfbabO",
    "bLbDbCb6b1akaeaYaRaPaHa6ZzZnZKZ1YhXzXGSLUlU6TLTASES2RiRXRDRBQtQZQPQ4PyPkPQPGP4OmODO4O3NlNYNMN0MuMFLVKWJveWeDe9dydidPczcncXcKcFc1btbibbbSbMbH",
    "bAawauafaWaWZ/ZpZbZOZ3YrXwX9T1ScNFdYdRb60000dWd50000000000000000000000000000000000000000000000000000000000000000cMcJcAc5c1bxbvbrbjbbbZbYbUbH",
    "bFb9b5b1auahaeaUaPaOa6ZnZZZJZ4YfY4Tb0000000000000000000000000000000000000000000000000000000000000000cQcLcFc8c7c3bybxbsbjbabZbQbMbJbFbAb6b0au",
    "amalaWaLaHZ+ZfZSYyYdX0Vr000000bZbRb6X/XvdXd5V1UzUcUYTwSuPPLY6g6M685r5q5p5c5B4z4s4q3q2C0000000000000000000000cMcJcAc5c1bxbvbrbjbbbZbYbUbHbFb9",
    "b5b1auahaeaUaPaOa6ZnZZZJZ4YfY5TaV3UwUrUaTKSAPbKz675i5d4r4m4Z4Y4W4Q4F443v3l372n000000000000000000cQcLcFc8c7c3bybxbsbjbabYbQbMbJbFbAb6b0auamal",
    "aWaLaGZ+ZeZSYyYdW+VsYQXrWjbZbRb6XtW0dWd5V9UrUpUmUiUWUDTsSDROQROkOCMEJJG76F5+5f504b4A40000000000000000000cMcJcAc5c1bxbvbrbjbbbZbYbUbHbFb9b5b1",
    "auahaeaUaPaNa6ZnZZZJZ4YfY5TOVDUwUtUaUWUSU8TwSERRQSOGNjLpJSGK605l5J5I5C4V42000000000000000000cQcLcFc8c7c3bybxbsbjbabYbQbMbJbFbAb6b0auamalaWaL",
    "aGZ/ZfZSYyYdWxVsY3WbVUbZbQb7XGW8dWd5VCV5U/UYUYUXUSUJTGR+RpQwPfP8OYO6O1M/MjMTM7LeKnKTJrJXIbHiH2GYECBkcMcJcAc5c1bxbvbrbibbbZbYbUbHbFb9b5b2auag",
    "aeaTaPaNa5ZmZZZJZ4YfY8RrV9UzUkUaUVURULTbTKS8R2QkPiP9ORNtNPNJMkMVM7LlLBK8JuJZIVHuHLGyDNBbcQcLcEc8c7c4bybxbsbjbabYbQbLbJbFbAb6b0auamalaWaKaGZ/",
    "ZfZSYyYbWnVzXXVyU0bYbPb8WEVjdWd4V8V0UtUpUmUeUbUTTsTGSjSNRmROQoQFPyPSPEOqOTO4NXNHN2MwL+LHL0KDJfGZcMcJcAc5c1bxbvbrbibbbZbYbUbHbGb9b5b2auagadaT",
    "aQaNa4ZlZYZJZ3YgY9TLV6V5UzUtUbUaUaUVTyTcTDR+RMR4QiQSPnPXOsOiOaOBNyNNN0MZMDLsLGKRI9FDcPcLcEc7c7c3bybxbsbibabYbPbLbIbFbAb6b0atalakaWaKaGZ/ZfZR",
    "Y+YaWYVzWVW7TPbYbMbAVyV6dWd3VAV6V6U/U+UrUbUTUSU3TZT8SSS6RvRTR3QpQZPjPLOqOjO8NxN7MoMJLuLCKjHwcMcIcAc5c1bwbubqbibbbYbXbTbHbFb9b5b2atagaaaTaRaM",
    "a2ZlZYZJZ0YiYATJVDV6V4V1UzUtUlUaUVUQU1TVSqSWS4RxQpQdQLQ3PaPKOfODO4NpNfMsMjM1L8IqcPcMcEc7c6c2bybwbrbibabXbObLbIbFbAb6a/asalakaVaJaDZ/ZcZOYyYX",
    "WnW7WPVXSqbXbIbEVpVDdVc+V7V6V4V3V0UdUaUaUZUYUXUTUFTyTCSlS2RvRqRBQnQXQ9Q0PSOxOaOLNhN2M3LjcMcFc9c6b/bwbubqbhbdbZbVbRbHbDb8b3b1aqagaWaSaJaIZwZd",
    "ZUZHY/YiYDTmVAV5V3V3V0U+U+UrUpUeUcUaUVUBTOTBSjSJS4RjQ/QkQBP+PYP9OhOVNuNJLoKDcNcKcEc6c2b+bybsbqbgbYbVbQbNbHbCb8b4ayarajahaTaDa6Z+ZZZKZ3YPX9Wa",
    "VnVRRUbWbGapWrVkdXd0VCV4U/UyUoUnUjUjUVUSURU6TsTBS/SXSRSBS1RAR0QZQTPtP7OcNuNdNSMtM3KlcJcEc8c7b+bvbsbobhbebXbRbRbIb9b6b5b0aladaWaUaHa6ZlZdZOZD",
    "YqYfXGV6V9V8V7UvUqUnUmUiUXURUQUATzTQTGSuSiSKS5RuRWQxQKPvPnPZP5OaOANDMcJDcMcIcDc3c1bybubtbjbfbVbTbQbJbDb8b4b0asalahaaaSaDa9ZqZVZPYuYJXHVrX4Wb",
    "T9bVbDb1VuVPdedAVCV9V5U/UzUuUtUhUeUcUVUUUSURTpTfTST7SjSHRhRPQrQaQ9PSPOPDOENQMKMAcRcLcGc3c1bybpbkbjbgbZbTbPbGbEb4b2asahabaYaSaKa4ZuZZZWZ1YqYP",
    "X9WxVCV9V5V3V2UxUvUuUtUoUaUUUSURUPTrTST4SIS8R/ReRGQjQNQJPvPNOnONMrKDcPcEcBcAb+btbqblbgbYbUbQbObEbCb3axawaoabaWaQaMa8ZwZnZMZ1YsY1XtVfVPV5QsbV",
    "bJawXAW2e5dtVCV6V6V2V1UzUtUsUsUoUnUUUMU7TYTESuSFRxQoQQQ0PkPHP3OqOZNsM+MXL9JRd1czcmcVcNcGcBc2brbqbkbcbRbMbIbEb5b1b0atadabaIa7ZwZhZXZJZ1YNY8WI",
    "VCV7V6V2UrUiUgUgUeUaUVUSURUJUHTBSxSPS8RbRCQkQ8PkPbOrOXN/NcMrLSJtcxclcXcRcKc6b/bmbjbcbWbSbPbCb5b3azataoaiaWaKa7a6ZxZtZhZ7YkYDXJVtWUVaSbcecCbN",
    "U9TteQeFVDV9V7V2U/UyUtUjUcUaUYUXUSURUMTFT1SCRoRHQzQaQMPWPFOeO3NTN2M/LyKGdsdidOdLd7c+cucrcZc4c0btbjbcbUbPbFb1avapafaYaTaAa0ZbZTYsYSYHX+VgVDVB",
    "V5V4V4U/U+UwUsUpUkUkUkUeUbUaUTUSUQU3TTS+SNS2RWQzQlPPP6OvMyL/dldddLdDd6d3ckcXcTcJc4brbibWbPbMbBb4auamaaaaaOaFa1ZqZeZ9Y+YnY3VhUZTdQ3dqdWbINFMR",
    "eIe8VBVBV7V7V5V2UxUuUlUkUYUATdTGRtRkQjQ7Q1OkOJNhMMLmKxJvIkIPHlHOGMFCedeYeMe3dvdudZdZdHd2cncYcVcBbhbYbUbIb7atamahabaSa1ZmZLZ8YlX/X0WkVBVAV8V7",
    "V7V6V4V3V1V0V0UsUnUhUgUcUaUaUaUaUZUWUUURTfT9SgSTSORmQiPgeXeOeEeDe5dxdpdjdfdQdIdCcyctcocccYcTcGcBc1blbZbTb4apaWa5ZmZfYTXjO3NoJqdodZb80000dTd5",
    "0000000000000000000000000000000000000000000000000000000000000000cHcEcBc8c4c3bybwbtbnbdbabZbTbPbLbIb5a/avajaeaZaUaCZvZhZTZFYFXcVr000000000000",
    "0000000000000000000000000000000000000000000000000000cLcJcFcCc3b/bubsblbibcbVbUbPbJbBb7b1axaqakaUaRaOaGZ/ZlZVZDYsXoXF000000bjbbbQTjR/dTd5VWT2",
    "PGLyLCKCFwE+5j5I5B564b4C48464440321d000000000000000000000000cHcEcBc8c4c3bzbwbtbnbdbabZbTbObLbIb5a/avajaeaZaUaCZvZhZTZEYFXcVrV2S+OWLNKlJxFbE/",
    "6C5a5G4z3w3Z2x2o1M000000000000000000000000000000cLcJcFcCc3b/bubsblbibcbVbUbPbJbBb7b1axaqakaUaRaOaGZ/ZlZVZDYsXoXEUwSw00bjbbbQTTT6dTd5VBUjOhNc",
    "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",
    "KWJ/J5IxIuINHfGhFL9xdxdSdLdAcEc6c6c3c1bxbqbmbhbZbXbQbMbDb8ayavafaeaZaPaDZvZdYgXpW+UtWfVqUzUOTQSySbSOS3RJR1QAPkPNP9OYOGN/NuN1MVM1LwLYL1KhKDJl",
    "JDIuI2FvdxdLc/cfcdcNcEc2b/bybrblbjbgbdbSbNbGb9b0axahaaaSaJaCZyZiZdYuYHXuV/Uq6jdpdZbe0000dTd5000000000000000000000000000000000000000000000000",
    "0000000000000000cHcEcBc8c4c3bybwbtbnbdbabZbTbPbLbIb5a/avajaeaZaUaCZvZhZTZFYFXcVr000000000000000000000000000000000000000000000000000000000000",
    "0000cLcJcFcCc3b/bubsblbibcbVbUbPbJbBb7b1axaqakaUaRaOaGZ/ZlZVZDYsXoXF000000bjbbbQX/XvdTd5a4Z0Y6X/XKXDX1WuWnWYWRWBVqVlJhIEFaFICMAmA09q9l9i9H90",
    "000000000000cIcFcCc8c4c3b+bybsbnbebabZbUbNbKbGb4a+awakadaVaTaFZyZcZSZKYNXbUqa5Z2YDXWXJXEWwWpWJWEVzKQJ3HSH7FvFFDTBIBCAhAb9o9S9L9K958Q72000000",
    "cMcJcGcDc2b/bvbtbkbibYbWbWbQbLbCb8b1axapakaZaRaPaJZ/ZnZOZBYtXzVrYPXpWybibabQXfWvdTd6ZDYwW/WhWEVeVEV3UBTxTcT5T2SRRSR0QtQNPpPSPGOaNrNeN6MTLaKZ",
    "K1IxI4HWcJcFcCc8c4c3b/bybsbnbfbabZbUbMbLbGb3azavakadaUaRaHZyZcZSZQYJXiTfZMYxX4WhWDVvVgUwUhUWUATrTdT3SpSHS7RVRRQkQFQ2PbP6OdO7NbMiMJL8JsIycLcJ",
    "cFcDc1b+bvbtblbhbZbXbVbQbMbCb8b2ayaqakacaSaOaJa3ZpZNZ8YlY2UaY1X3UqbhbYbQWtWMdSd5YnY1V8UpUUU8T3SiSLRzRZRCR8QyQVPtPTP5OyOOO2NfNNMrMZM4LzLPKlKQ",
    "IhHBcJcFcCc7c4c2bzbxbrbnbdbabYbTbNbKbHb3azarafaeaVaKaHZzZaZUZLYLXpV4YwY7W5ViVDV3UGTdTTStSOS8RwRsRMRDR2QrQJQDPzPfO+OxOKOINUNKMzMaK7I3cLcJcEcC",
    "c1bybvbqblbgbbbXbRbPbLbBb8b2ayaralacaRaKaFa6ZpZPZ8YkX/X0XYWxV3bgbSbKWNVgdRd5YJXZUzUWU1TEStShRzRMRAQyQAP/PWPCOpOgOOOANbNRNKMqMgMOLsLaL6KUJ4Fv",
    "cJcFcDc7c4c2bybwbrbmbcbabXbTbObKbHb2azaoagaeaVaHa3ZxZfZZZOYLXyWkYCY1W1VJUnTvTpTDSwSjSYS4RyRrROR4QuQpQSQ0PNPHO+OdOJO2NLMpMTLvLVIncKcIcDcBc1b+",
    "bvbpbmbibZbXbQbMbKbCb8b2auaqalafaRaKaEa1ZnZVZ6YcYAXRW+WeTvbgbNbDWqVGdOd5YrXJUYTzTwTFSiSLRrRMR3QWQPQFPzPdPNPCP2OeOcODNwN7MrMhMELzLHL2KIJZcJcF",
    "cDc7c4c2bzbubsbnbcbbbYbUbRbKbFb2azapaiacaXaDa0Z+ZcZWZIYJY4VVYaXYVuV8UuUOU1TmTWSxSgSISGRvRaRYRJRFQpQUQAPuPTPLP0OZO4NxMqMMM0J0cJcHcDc9c2bybwbr",
    "blbkbabWbObJbIbDb8b3awauamaeaQaLaJZ/ZnZaZ5YkXwXaXHW7TNbgbEb9V/UodId4XpWOULTcS+SUS4S1RFR3QtQcQIQ9PiPUP0ObOYOJN/NpNaNIM/MnMHLtLLKwKgIrcJcHcBc7",
    "c4c2bzbubtbobbbbbZbWbPbLbEb2a+aqajafaYaBa2ZvZZZRZ4YeY7XTXYWCW3VGUmTjT5SmSRSLRwRtRdRXRPRHQwQqQaQKQIQAPpPfPFOzOKNaNON2MHKdcMcHc9c7c0bzbubpbkbg",
    "bbbQbNbKbGbCb8b3b1a+aradaQaPa7ZtZfZYZCYgXvW/WEVMSebfazaqVuUadGcyXKW2U5SvSXS3RqRNQyQlQfQ1PlPUPFP0OoOhOPOANnNWNJNCMyMmMMLjLLL5JJGecJcEcAc7c4c2",
    "bzbvbpbobcbabXbUbQbJbBb4ayanakaaaWaKa1ZuZcZWZMYaYAXMX5ViUTTqTiS+SnSXSTR/RWRLR9QoQmQgQPQ8Q0PdPHP9OnOcOMOBNmNdNRMjM5JhcJcEcAc5b+btbsbobjbabUbT",
    "bPbJbGbBb8b4b0azanabaSaQa3ZuZdZCY/YSYFX5VlULR1beanaWWEUhdBcnXvVnVHTQTGSjRsRkRaQxQFQ4P+PkPVPRP7OsOmOhOBNyNnNVNPNBMqMVLeLPKMJJcMcFc9c8c3c1bybu",
    "bnbkbbbWbWbRbObHb7b3axapahaYaTa6Z/ZmZUZMYoYNY8XeXRVgVVU+TdSxSjSTSHRyRnRaRSRDQwQdQXQHQAPuPiPMPKOzOlOMOJNoNRMjLyI4cBc9c6b/btbsbnbjbcbXbUbPbMbF",
    "bCb8b6b1a+awaiaZaTaOa5ZrZcZGYxYRY9WVWnUoRqbdagaCWPVJd9cjY7X4VzUjUSTiT3SgSKS8R/RSRHQoQcQVQ6PtPmPYPHP1OxOXONNzNhNNM/MrL/Ksd1clcEc3c1bzbubqbnbl",
    "bhbdbQbPbMbHbBa+asaoacaUaRaGa3ZfZaZ0YoY2XoV/XvX6WuWWUNT+TlTRTBSxSoShSMS8RvRnRiReRLRHR2QxQZQFQ3PvPhPEP4NxN4KccucTc1bybqbfbdbZbUbQbNbJbCb5b5az",
    "avamahaWaUaIa7ZvZtZaZTZ0YgYZWLU6WSVmRrbabIajVoU4dnd7YmWkVnV9UfUIU0TeTJSxSPS2RoRYRKR5QzQcQUQPQ4Q0PqPOOzOwOUOHNqNJMbKie7dydadCcHc4bvbqblbebabX",
    "bQbKbIbCb5ataqakacaVaQaEZ/ZmZWZJYuYaWuVnYOXSWkVvVfVHV4UiUVT+TpTjTWTLSlSaSER/RmRZRHRAR2QZQ7PgP5P3OjOBNrKxeGdpdXdGcyctc1bubibVbUbPbHbDb2ayaqao",
    "alafaSaKaBa2ZrZmZNZAYnYIXvW6W4UtQac/cdbSS/RpdtdAUpTyT1RJQtQNQ3PkPVP9OmOfOSONOGNkNVN9N6MyMoMFM0LtLoLQKxKZJuJOIkGGeQeFe1dicNbybrbkbcbUbQbMbKbD",
    "b9b3azaraqaZaYaIa9a1ZyZhZJYwYoXkXDWTVCUlU7T3SbS9RyRfRLRCQvQoQQQPQ2PnPdPRPKP8P0OZOROKO6NxNHMkMXLiKRI8eeeDe4dodgdbd9cdcGcAb+bjbebXbLbEb7b5a/aq",
    "ahaUaQaOa4ZmZbZHZAYcXlViTOSfNsdfdLbF0000dTd50000000000000000000000000000000000000000000000000000000000000000cHcEcBc8c4c3bybwbtbnbdbabZbTbPbL",
    "bIb5a/avajaeaZaUaCZvZhZTZFYFXcVr0000000000000000000000000000000000000000000000000000000000000000cLcJcFcCc3b/bubsblbibcbVbUbPbJbBb7b1axaqakaU",
    "aRaOaGZ/ZlZVZDYsXoXF000000bjbbbQX/XvdTd5U/UVUTTlT0S0NkMp5j5Z504m4X413d2s2P000000000000000000000000000000cHcEcBc8c4c3bzbwbtbnbdbabZbTbObLbIb5",
    "a/avajaeaZaUaCZvZhZTZEYFXcVsUaUXUSTcT0SCNsNG6E5f5O535100000000000000000000000000000000000000cLcJcFcCc3b/bubsblbibcbWbUbPbJbBb7b1axaqakaUaRaO",
    "aGZ/ZlZVZDYsXoXFYPXpWybibabQXYWzdTd6VCV2UbUZUXUJTQRoQHNxNYN0MWJgHtGi6G685j5g5c564e370000000000000000cHcEcBc8c4c3bzbwbtbnbdbabZbTbObLbIb5a/av",
    "ajaeaZaUaCZvZiZTZFYFXbVoV9V1UrUkUYU7SsSBR1OiNhNBMuJdJ2G4625b5V5M4z4Y42000000000000000000cLcJcFcCc3b/bvbsblbibcbWbVbPbJbBb7b1axaqakaUaRaOaGa0",
    "ZmZUZEYsXpXDXxX3UwbibabQVzV1dTd6VAUxUsUkUjUWUSTzSqRwRjQrQ0PeOfOTNxNZMgM6LFK7JrJjItIZHkHCGXEwEOApcHcEcBc8c4c3bybwbtbobdbabZbTbPbLbIb5a/avajae",
    "aaaUaCZwZhZTZFYGXaVlV5UwUsUsUiUYUYU4TZSYSLRfQxQZPMOtOCNRNCMWMLM1LdKSK5JqIbIIHmGqEh9ocLcKcFcCc3b/bvbsblbibcbWbVbPbJbCb7b1axaqakaUaQaOaGa0ZmZU",
    "ZFYrXqXEWIVaT6bibZbPW+UKdTd6VCV4V2UqUhUfUTUKSHS1QsQUPWOxObOKOBNdNJMlMCLkLdKrKUJxJLIzIEHgGyFxcHcEcBc8c4c2bybwbtbobdbabYbTbObLbIb5a+avaiaeaaaU",
    "aCZwZiZTZFYGXOVkVCV3U/UzUvUVUUURTGSQRlQpQKPtPhPPPGOyO9NjNQMzLxLJLBKoKhJQJDIeHZGicLcKcEcCc3b+bvbsblbibcbWbUbPbJbCb6b1axaqakaUaQaNaGa1ZmZUZFYq",
    "XtXBXFVRTZbhbabLWNVYdSd7VDVCV4UsUeUVUTUSUBTxSxS8RwRGQZQ1PiPYOkOWOTNwNlNLMjMFLvLLKpKHK2H3cIcEcBc7c5c1bybvbsbobdbbbZbTbObLbJb5azauajaeaaaTaDZx",
    "ZkZSZHYGX4VSV3UxUtUrUmUmUWUUUUTyTSSfSARfQyQVQAPvPJP1OOODNzNgNQMkMTLYKlKFJRFBcKcJcEcBc3b+bvbsblbhbbbWbUbObJbCb6b1axarakaTaPaMaFa2ZmZTZGYqXyWz",
    "WlVYT9bibebLVnU9dQd6V9V6V5UxUiUeUeUYUXUTUBTKTDSMR/RvRdR8QQP/PcPTPLP9OZOBNVN1MQLTKpK3cIcFcBc6c3b/bzbvbqbmbcbZbYbTbNbLbKb3awauahaeaXaOaCZwZlZN",
    "ZIYBWMUMVDVBV0V0UqUjUdUZUZUYU4TwTqT9SmSERzRiRFQvQSP+PkPCP7OlO9O5NkM+MhKxcIcHcBcAc0bybvbsblbfbbbUbTbNbHbCb6b0avaqakaUaOaFa9a0ZpZSZCYrY9VAW8UM",
    "SFbfbRbMU/UldRd6VCV5V4V4V2V2UfUeUaUZUXUSU5TWTPT5SoSNRXRLR5QvQAPjPbP6OcOINmN5N2LpcHcGcBc4c3bybwbubobhbebXbUbQbNbKbJb1auatadaZaPaAa6ZyZoZTYsYA",
    "XYTzVCVBV7V7V7UuUqUmUlUYUUUSURUNUBTWTQSrSOS3RWRDQzQjQUPrPNP3OXN4MyEQcHcDc8c5b+btbsbqbmbdbabUbPbDbBb8b0ayasakaeaXaKaFa0ZqZeZNZHZ0Y0XFVFUnQzbe",
    "bVbEWcUwdXdCV9V5UuUqUlUaUYUTUTURU1ThTTT4SmRyRpR7QeQYPqPhPJOsOeN/NmNNNEMaLgJMcRcKcGc8c4b+bvbubqbjbcbYbTbSbQbJbCb2a+aragaTaOaCa2ZtZgZPYxYJWyUn",
    "VDV5UrUpUkUhUaUYUSUSUSULU0TXSnSAR+RrRRRIQkQMP+PpPVP7OWO8NhN2MWLNcLcBc5c0bzbvbrblbgbZbYbWbQbGbDb4b1arapaladaTaSaCZ/ZjZbZSYyYrYKVZWDVBRbbdbWbK",
    "XYWIe2dlVDVAV7V5V5UxUoUhUeUZUYUSUCTpTXT2S+SbS8RkR4QiQ9PYPIOkOZNtN5MwL8Gid6cycncicZcHc6c2b/btbobkbfbbbLbKbEbAb2atapahaRaMa1ZoZiYwYlXKXBWFVCVA",
    "V7V7V1U+UyUtUrUnUfUXUWUSU2TXTOSXS4RPR3QfQ1PcPTOyObOGNbMyMKJVcvcqcjcMcDc7c3bubobkbdbYbObKbCbBb1azanajaZaTaMa7a2ZfZKZ1YnYZXWXLWMVsRecWcRbYTvTN",
    "eKeCVBV5V3V3V1V0U/UrUhUeUcUWUTUSURU2TfTMSZS5RkRdQqQkQ3PpPPP0O0NRLoJMdrdmdIdAd3clcjcdcZcKbybvblbhbbbXbQbEbAazawadaQaLa3a0ZUZMZ6YhXaWrVBV8V7V7",
    "V6V0U/U+U+U+U+UyUyUrUjUiUaUaUVUTU4TiTVSsSAS2R8R2QLPEOqMzdgdbdJdAczcvcicccTcFcBc3bnbebabVbMb7a+awavacaUaIa5ZqZZZVZFYbY3X2UAToPbdqdfbUO9MheDe5",
    "V9V8V7UzUzUtUeUaUWUSURUETqTbTNRQQYQ5PBOgOONSMSLUKcJyJrIeIMHBG7F1egeZeNe4dvdcdWdRdAd1cpcjcRcEbwbgbbbWbPbCavanaZaYaPa4ZLZDZ6YZYCVtVCVAV9V7V5V4",
    "V2U+UzUtUkUhUgUdUdUVUVUUUTUTUSUETsTUTDSxSeRxRRRAQoPOeWeJeEe7d+dpdndddWdQdJdCd8d2cucpcmcZcBc7bvbnbNb9a/anaZZ+ZmZLYiWnOGNVJRdpdebK0000dddD0000",
    "000000000000000000000000000000000000000000000000000000000000cOcJcHc8c5c0b+bxbvbpbhbbbYbPbObKb9b3avaoalahaVaIa2Z+ZgZOYtYfXnXT0000000000000000",
    "000000000000000000000000000000000000000000000000cPcHcEc9c5c0bxbqbobgbdbXbWbJbGbAb3a/avavasaiaUaKaHa2ZpZfZJZ5XqSt000000bnbTbHWZVLdddDYDWzRYRR",
    "Q6Q2N7LzLEKtKCIwGKFtFFE1D6BcB5A7928j8i8J8H8A7R6z00000000cOcJcHc8c6c0b+bxbvbpbhbbbYbPbObKb9b3auaoakahaWaJa2Z+ZgZMYtYfXnXVY3WVR2QHPyOpNQLTLEKk",
    "JmIiHDFnFUElDhDGCOBfAk9n998l8j8d807M7G4t4c00cPcHcEc9c5c0bxbpbobgbdbWbWbJbGbBb4b0avavasaiaUaKaIa2ZoZfZIZ6XpSzY0WY00bnbSbHXBWHdddDYsXuSWRnRVQh",
    "N/NPMTLhLUL7KVJHHvHEGsGAFoFXEYEMD/CXBzBXBGAtAo8u8t00cOcJcHc7c6c0b+bxbvbpbgbbbYbPbPbKbAb3avaoakahaWaIa1Z+ZdZJYtYfXmXUYfXcRuRQQfPpOwOKMuM1LjLJ",
    "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",
    "EgD8e5djdYdJdEcecUcNcCc1bxbwbobcbYbUbRbOb8b3apakabaTaHa2ZgZAYsYgXSVXQEQ600eQe8bk0000dddD0000000000000000000000000000000000000000000000000000",
    "000000000000cOcJcHc8c5c0b+bxbvbpbhbbbYbPbObKb9b3avaoalahaVaIa2Z+ZgZOYtYfXnXT0000000000000000000000000000000000000000000000000000000000000000",
    "cPcHcEc9c5c0bxbqbobgbdbXbWbJbGbAb3a/avavasaiaUaKaHa2ZpZfZJZ5XqSt000000bnbTbHa2ZWdcdCbpbHaCZuZXZPYxYvYcYbYIY1X/XhQALkKSJiJUJLIxInIMHQF1CgCDBv",
    "BT000000cRcMcKcAc8c5c2bzbybpbhbebbbVbPbMbJbCb6a+anaiaTaFaAZjZbZIYXYBXJWMbqbHaCZkZPZCYqYeYdYDY4Y2XtXpXjXbOYKtJ4IiH/HqGeEiCxCEBHA+00000000cPcL",
    "cEcDc8c5bybwbrbobfbZbYbQbIbCb9b3awataiaVaNaFa6ZzZtZYZLZ9XEWtaBZsYybobXbIZgYVdZdCbGaWZrY/YjYVYKY0XhXWXOXGWxWeWVWAVtVkVFV5UnURU7TiTPSxS9REQEOb",
    "N9JRcXcNcLcDc9c6c1bzbybnbmbgbabTbNbLbHbGb3ataoajaVa9a1ZiZYZSYQYEX5VqbSaOZkZ4YQYEXpXhXXXHXAX2WVWFW2VkVZVHVFUYU0TpTJT6T3S6RLQiPGOVNiMMcPcLcIcD",
    "c9c6c1bxbsbnbebYbUbSbIbCb5a/avataaaVaPaFa6ZuZnZYZHYuXQUZZvZ4XVbobLb8YnYDdUd8aIZqZaYfXqXSXOX1WoWbWFW3VqVdVRVIVAUvUbULT+TtThTRSuSeS0RZR6QEOqMi",
    "c+cQcJcFcCc9c1b+bxbrbpblbdbVbNbMbIbCb2avakaiaUaIZxZnZaZMYGXoXLUaZyZcY/YcYFXHWtWcWQWHV+VpVaVJV7V3UkUbUVURU9TwTcTPSxSrSRSBRkQvQDOJcicXcJcDcAc8",
    "c3bzbwbmbgbZbYbNbKbCb8b5a+atanaXaOaIa3ZtZkZaZOZCXoXAZGYZWbbqb2apYSXddNd6ZKY/YoYHXaXKWTWOW2W0VwVqVRV7UxUlUPU1TwToTOTJTASuSmSGRwRXRPR2QQMEdIcj",
    "cNcIcFcEc6b+bzbvbtbmbebYbObNbKbDb4atalagaYaPa2ZqZTZ6YcXqXfVvZOZ0YLY9XxXKWsWkWLW8VvVgVUVHV2UwUcURUHTyTwTYTMTCSwSoSIRlRXQTPkPIcxckcQcDcAc7c2b/",
    "bzbobkbgbabLbLbEb9b6b2awapagaMaLaCZmZdZUZGY/XiXBYgXbVJbrajaKXeWudLd2ZBYsYdXMWmWXWAW5VyVhVTVLV5UvUiUXUIU5TyThTKTBSvSmSgSCRhRFQXQ1OINUdZdAchcL",
    "cKcFcBc6c1bwbubibfbZbQbObJbAb2auatanaWaOa8ZzZbZJYrY5XWWzZAYtYOXhXVWvWkWeWLW3VnVjVbVFVBUsUkUWUSU5U1TuTWTBSuSeSOS1RhRIQLPjdGc/cjcOcHcAc3c2c0bv",
    "bmbfbbbSbObLbIb7b1awatafaYaJaCZyZpZCZ7YVXlXaXMXGU2bsb4anXnW+dXdPZiZ0YYXaXDX0WmWMW6VtVqVbVWV3V1UtUoUUUJT/TuTfTJT1SXSGS7RvRAQTPpOpeIe9dycbcScN",
    "cNcKcFcBb+bsbibdbcbXbMbBb2a/awarafabaPZyZmYxYpYMXfUTZBYcYRY0XpXmXIW/WyWmWTWHW8W5VzVjVWVFUyUsUiUGTyTfTNTDShSYS6RsRhNMe3e0dJdBd6crcgcMc5c2bzbo",
    "bfbdbabWbJbCb9b5ayalabaaa9a4ZvZRZGYiWPTRY5XhTicocibmVcUzdjdYX9X0WbV6V4UkUaUAU7TaTWTFT8SxSsSeSWSOS5RxRnRMRIQvQSQGPuPWPAOTNqMpeeeVeScvcmcdcScR",
    "cKcIc5bqbnbgbdbWbMbIb5a/ayaoajaaaIa3ZpZTZ6YMXqUnX1WoWjWWV/VsVlVWVRVEVAUzUnUYUPUCT/TcTRTLS+SzSkSUR/RyRYRFR5QoPjNMeReLdudndWd9d3cucccXcIc8c0bx",
    "bkbhbfbbbYbLbIb3auapaaaSZxZsZgZCYPXbWVW9S0dddQbkTFSwdkdZUGU7U4SwShSMS9RhRZRTQ/QyQmQUQQQBQ1PqPcPaPJO+OrOlOPOCNPN7MoMgLJKdeeeWeUcxcncecRcQcHcF",
    "c1bnblbdbbbUbJbDb2axasalagaUaEa0ZpZPY/YSXZV+UrUTUPU5ToTYTWTQTMT7S+SnSgSMSFRzRmRYRMR9R6QrQKQ1PkPeP6OuOANyMtLdeQeJdxdndXdEd5cycfcacMcKb/bwblbf",
    "bcbWbUbMbGbBaramaYaHa3ZuZWYnYKXHUKUAPxdmdabgTISldhdVUCT/TrSpSbRzRvRYRJREQ+QnQaQOQLQAQ5PyPoPTPLO/OcONNzNQNEN8MSMGLXL0eZeSeQcuclcYcJcHc9c4brbc",
    "babSbQbJbBb2atanajaeaRaIa2ZpZhZLYoYYXMWLUBTxTZTMTFSzStSoSXSUSOS5RyRsRZRIR2Q+QpQhQGQCPyPMPAOpOUOKO6NHLZK+eHeAdtdidWdEd1ctcYcUcScGbvbsbgbZbTbO",
    "bGbDb7b3aiaYaHa4ZuZdZLZ4YEWITmTdPPdodgbUTJT5dedRUKUAU0T1SqS1R+RiRcRJR4QwQgQeQMQIQ0PzPnPXPROqOkOZN/NUNFN6MjMHLjJ3eXeQeLcucncNc9c6bzbrbYbRbKbH",
    "b6b5azarakadaTaOaDa1ZnZkZHZ2YdYNXHSsU7T+TZT7SvScSYSQSJSFS4RtRnRWR8R3Q+QuQnQPQ8PsPePOOyOtOTO7NZN1M8KWe8e4dmdhdOd5cwcfcScPcGcCbtbhbabRbHbDb5az",
    "ataoafaHa4ZqZdZLYsYYYJX3TcTSP4dodibATNTCdedRUVUAT/TCSvS8RyRmRkRRR2Q+QiQYQVQGPvPnPgPTPDP3OuOVO9NoNfN3MnMcLQIKeYeReJd2cvcFc5bpblbYbTbEb4ayaoae",
    "aYaXaQaMa7Z+ZnZhZSZ4YxYkYZXtWTQzU9U2TRT5SeSXSRSMSAS7RxRjRWRORBR2QtQoQSQGP+PjPbPHOrOZOGNwNGN6MhLhe7e1dodWd6cvcocVcScCc9bybkbPbIb9b5b4araWaJa8",
    "a6a3ZzZQZPYsYOXzW+VyTVTOP4dmdhaa0000dddD0000000000000000000000000000000000000000000000000000000000000000cOcJcHc8c5c0b+bxbvbpbhbbbYbPbObKb9b3",
    "avaoalahaVaIa2Z+ZgZOYtYfXnXT0000000000000000000000000000000000000000000000000000000000000000cPcHcEc9c5c0bxbqbobgbdbXbWbJbGbAb3a/avavasaiaUaK",
    "aHa2ZpZfZJZ5XqSt000000bnbTbHa2ZWdcdCV1UwUvUtUqUYS8Q/QGPvPENyLMKvKHJ3I8GjG9E2CfA/AdA88J8A7d5t5W5K4g00cOcJcHc8c5c0b+bxbvbpbhbbbYbPbObLb9b3avao",
    "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",
    "T6S/SrSgSTSOSISARtRmRYRAQoQaQJQAQ1PfPROyOdNnNbN8M8JSeLeJdsdldjdidZdVdSdMdFdCdAc/cxcrcecZcOcIc4bxbsblbabGawaRZwZqZcVMDoDYAOeIe7bdIRH+drdiVDV7",
    "V2UvUrUpUgUdUbUbUSUJTuTiSySJRJQUPXPOOiNGMjM5L4JGH/HXHBFXEWCdf6e+eveheXeVeJeCeBe0dydsdhdXdGd6d1czcocgcWcRbrbbbTb3ataVa2ZhZ1WSVBVBVAV9V7V7V2Uw",
    "UsUmUlUfUbUWUVURUFT/TuTbTITGSzSiSVSOS2RNQQQIPoOffAf8eReMdzdxdwdldhdTdMdFdBc+ctcscocacTcDbybwbcbXb9axafaPa3ZGYpXPLGKMHjdcdRaw0000dWd500000000",
    "00000000000000000000000000000000000000000000000000000000cMcJcAc5c1bxbvbrbjbbbZbYbUbHbFb9b5b1auahaeaUaPaOa6ZnZZZJZ4YfY4Tb00000000000000000000",
    "00000000000000000000000000000000000000000000cQcLcFc8c7c3bybxbsbjbabZbQbMbJbFbAb6b0auamalaWaLaHZ+ZfZSYyYdX0Vr000000bZbRb6VaTddWd5XJVkSKRXQFOv",
    "N9LdL4J+JQHiG/ELDSCGBzBUAO8E887C7C6t6q5+5Y0000000000cMcJcAc5c1bxbvbrbjbbbZbYbUbHbFb9b5b1auahadaUaPaOa6ZnZZZIZ4YfY4TUXUVKRAQxOwO5MjLAKmJpIVHw",
    "GdFbDuDYChCPAnAR9X8L857y7q7P7L7I7G5q0000cQcLcFc8c7c4bybxbsbjbabYbQbMbJbFbAb6b0aualalaWaKaHZ/ZdZSYyYdX0VtXTVN7UbZbRb7VzV5dWd5XsX6SOR+RBQFOTMy",
    "LRL1K+KIJIHhGdG6FSEvESEOD9CpBeAwAO9u9f8Q887b6y00cMcJcAc5c1bxbubsbjbbbZbYbTbHbGb9b5b2auaiabaUaPaNa6ZnZZZIZ4YfY3TfY5WgR+R7QFPWO9MYMCL7KVKAJVIT",
    "IGHZGnGAFNF4EEDyDDCqB+BVB0AqA48w7v6rcQcLcFc8c7c4bzbxbsbjbabYbQbMbJbFbAb7b1atamakaWaKaHZ/ZbZQYzYeX0VwY2Wl00bZbRb8VSVNdWd5XEWiVPTJT7QzPmOrNgNF",
    "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",
    "bLb6b3avaqaoaeaZaNaDa8a3ZkZ9Z0YCX8VJUIU7TmThTPTKS+SrSlScSMSCRvRfRcRAR9QzQcQNQBPuPiPVPGOoOZNpNgNNMSKveIe9e6djdbdNd1cpcicacQcEc4bvbjbhbcbTbIb3",
    "b2asalaYaFZ/ZtZXZWYiYNU2TjTZPgdtdcbSTIT5dpdPUBU2TzT2R/RqRfRXRUR6QbQXQIQ3P/PjPXPJO/OyOhOHNzNKNAMxMnMCLoK+KnGGeZeReIdKcFbzbybqbebabNbKbDbAb5ar",
    "aqabaaaSaPaHa9ZwZrZlZVZ4YmYCW5VpUIT/TuTXTPT4SjSZSVSNS3RwRlRWRRRAQ/QqQYQLQ5PrPWPQP6OlOUOPNbMULsK+eHe0d+dhdWdId6cecXcMcEc8bubobabWbTbFb5a/alae",
    "aRaGa5ZpZjZIYtYZXsWbTYTOPZdrdcbATTTGdmdQUHUBU4TGS8RlRfRURAR9QiQPQNQ3PwPZPTPOPBP0OZOQOGNtNiNWMsMALzLxKeJMeWePeIdLc9bvbpbibdbMb2atapalahaSaJaA",
    "a3Z+ZrZpZfZKZDZ1YsYcYQY7WkT9UHU4TjTUTESrSZSXSHS6RyRuRaRER5Q+QpQaQAQ5PtPjPaPEOsOgOBNjM+LmLgKWeKe0dqdfdOdDd1cdcIc8c4c1bhbZbIb7b0awaqadaZaMa8Zp",
    "ZkZIZCYlYcYAXKWeTUTIPZdndbag0000dWd50000000000000000000000000000000000000000000000000000000000000000cMcJcAc5c1bxbvbrbjbbbZbYbUbHbFb9b5b1auah",
    "aeaUaPaOa6ZnZZZJZ4YfY4Tb0000000000000000000000000000000000000000000000000000000000000000cQcLcFc8c7c3bybxbsbjbabZbQbMbJbFbAb6b0auamalaWaLaHZ+",
    "ZfZSYyYdX0Vr000000bZbRb6aHXydXd4V7V3UtUpUkUaT7RcR3PzPPNhMzKKJRIFHyHTGNE1CiBwBSAV9A7T7M6t634h0000cMcJcAc5c1bxbvbrbjbbbZbYbUbHbFb9b5b1auahaeaU",
    "aPaOa6ZnZZZJZ4YgY5TuUeUaUXUXURTvSZR1QdPfOLNmMULSJlJPIXIFGdGIFXEDDSCRC5AoAN9K7A6U5x5EcQcLcFc8c7c3bybxbsbjbabYbQbMbJbFbAb6b0auamalaWaLaGZ+ZeZS",
    "YyYdX0VtaRZHYUbZbSbIZgZBdYd3V8V6V0V0UeUZUWUSU0TmSkSKRmRGQTPPOrNON5MJLfL6K3IcIKHVHBF/F8DBCN7xcMcJcAc5c1bxbvbrbjbbbZbYbUbHbFb9b5b1auahaeaUaPaO",
    "a6ZnZZZIZ4YfY6UEV9V5UvUmUlUhUbUGTgTYS+SCRHQgPZP0OcOUO2NWMlLwLPL2KdKPJVIZIPI7FZFRcQcLcFc8c7c3bybxbsbjbabYbQbMbJbFbAb6b0auamalaWaKaGZ+ZeZSYyYc",
    "W/VxZxZLXtbZbYbGYoY2dXd1VDVAV6V5V1UpUjUcUaUTUSU7TtT8T0SeSKRxReRIQuQXQ9Q1PfNyNEN4MYLhKSJYcMcJcAc5c1bxbvbrbjbbbZbYbTbHbGb9b5b1auahacaTaQaOa5Zm",
    "ZYZHZ5YfY7UEVBV6V2UzUwUuUkUjUgUUUUUATyToT2SnS9S2RpRXQuQHQ4PqPXOaOKNUNGMPMFKpcQcLcFc8c7c3bybwbsbjbabYbQbMbIbFbAb6b0atamalaWaKaFZ/ZeZRYzYcWvVx",
    "Z5YaWVbfbYb9Y6XzdWd0VCV8V7V3V1V0V0U/UvUvUuUuUsUSUSUQU7TrTJT2ScSPS6RBQPQDPqPdPHOVNxNJcMcIc9c5c1bxbvbrbibcbZbYbSbHbGb9b5b1atahaZaSaQaNa3ZkZVZI",
    "Z4YfY4UEVCV9V8V7V5V5V4UuUmUhUfUaUZUXUUUDT/TXTKSySiSRRcRIRFQYPvPgOzOpNPMucPcLcEc7c6c2bybwbrbjbabYbPbMbIbFbAb6b0atamakaXaJaEa0ZgZQY/YaWrVvXuXn",
    "VIbkbWbHYEXsdZd3VBV8V8V7V4V2V2UxUuUtUpUmUgUaUXUWUUU8TxTcTKSwSqSBRyRdRRQWQEPKOeNVcLcGc8c5c1bybvbrbibebZbXbQbHbFb9b5b0arahaVaRaNaKa2ZgZQZKZ3Yh",
    "XlUzVDVAV8V8V7V0UuUkUiUfUfUaUVUUUSURUQUPU2TUSmSWSARxRKR0QfQJPtOyNXL7cPcLcFc8c6c1bybubqbjbbbXbObMbHbEb8b5a/atamakaWaIaBa1ZiZRZ7YbWrVpY6XFUOb/",
    "bmbRYAY2e3daVCV8V6V5V4V4UyUxUuUsUsUhUaUZUYUXUTUSURU8TyTgT6SsSaR/RdRBQlQiPlObcLcFcBc8c3b/bvbtbmbgbabYbMbHbDb9b4b2aqadaTaRaKaEa0ZWZHZ4Y/YmX0UL",
    "VDVBVAV7V7V6V1U/UxUuUuUrUqUjUhUgUeUcUCU1TnTbTNSySmSCS3R8R3QKPwNHcRcMcHc6c4b/bybsbobibdbTbQbMbJbDbBb2a+auanagaTaKa7a2ZjZQZJYTXFWLYAXUUTc/cYbj",
    "YXXweZe7V9V8V7V5V3V1V1UtUpUpUmUjUiUgUdUcUXUUURUATnTJT7SaSVRjRAR5QzQTPPOjcScMcIcEcAc3c1bwbtbmbfbebRbObIbCbBawapajaZaRaOaIaAZgZaZFZ8Y/Y9VGVDVC",
    "VBV1V0U/UyUvUoUjUjUcUaUaUZUVUVUVUNTyTuTMT7StSkSMS5QpQYPqO+KvcXcQcPcDc7c2c0b/btblbdbcbUbTbKbIbEb3a+atapaaaSaPaMa0ZzZZYvYNXcWJYOW/UKdvdJbmX0WB",
    "f4esVDVBV7V7V2V1V1UxUwUrUmUlUiUhUfUdUVUUUTUSUSURURToTXTESaS3RaQhPBOkclchcccXcScKcGc9c7c1bsbrbobgbfbZbObIbAa/atahacaTa8a0ZiZQYyYcYCWqVDVAV7V5",
    "V2V1U/U/U/UvUuUqUpUpUoUbUXUTUTUTUSURURUNTwTiTJScSURrRLPOcncicbcZcLcJcEcEc5c2bzbubmbhbXbUbKbGbCb9b3atajabaSa4ZwZXZNYxXfWfX7WlTiepeGbxIdIAewek",
    "VBVAV6V2V2UqUpUiUgUeUaUZUYUXUWUVUUUTUTUSURURUEU2TRSmS0RsROQ7ODFudfdXdUdPdNdHd9d8d6d2cycwcrcicgcbcWcScRcFc9c2bybrbObBaVaOZbYrY7XIVCVBV7V5U+Uy",
    "UwUuUqUoUmUlUlUjUjUfUaUZUYUWUUUSUSUSURUKU9U1TZT4QONodadVdNdIdFdAd7d0c/cvcnckcjcecYcUcQcOcGcBbzbvbkbWbPbKb3alaZZqYZXwJeJFGCejeFboCfCKeZeOUjUT",
    "UQUIT+TrTWSvSrSjSSSCRpRLQzQ2PpOmOeNgN5MOLQKtKKJwJIIuIUHfF1DTeJeBe6e4dydvdndhdfdbdUdSdKd8d3c/cycscicYcVcMcCc2bSbIafaNZcYHW+UQV8V5UMTzTwTYTOTG",
    "T3StSVSPSGS8RnRfRWR4R2QwQaQKQ7PsPXPJP4ONNTMsLbGjeKeGdydxdtdedZdWdTdNdKdBd8c/cvcpcmcgcYcScKcCb+bjbMbHb2akaTa4ZfY8FUDaBQeNdubSJII+dtdiV9V8V5U/",
    "UtUsUrUrUmUkUgUfUVUDTvSlSORlQlQDPIOZNeMgMVLNK6JRJ3IKGxDsf5e/ezeqekeZeXeQeFe9e1dodYdNdLd6d0cvcjcicScPc1b+bUb5aaaFZIYjYPXmVCVAV8V5V4V4V4U/UyUr",
    "UlUjUiUhUhUXUWUUUTURUGUDUAU0TcTLT3SRRxRRPqPHf9f4eeeWeOeFe2dxdvdgdZdQdNdAc/cucocicbcQcLb/bxblbhbOb3aVZ/ZuYbWHMdLWIldhdCal0000dTd5000000000000",
    "0000000000000000000000000000000000000000000000000000cHcEcBc8c4c3bybwbtbnbdbabZbTbPbLbIb5a/avajaeaZaUaCZvZhZTZFYFXcVr000000000000000000000000",
    "0000000000000000000000000000000000000000cLcJcFcCc3b/bubsblbibcbVbUbPbJbBb7b1axaqakaUaRaOaGZ/ZlZVZDYsXoXF000000bjbbbQWMUsdTd5Y1WVR6QpPzOrN0MB",
    "JoJfIxH/EODbD4CoBMAiA89f9Z857/7j7a74675L00000000cHcEcCc8c4c3bybwbtbobdbabYbTbPbLbJb5a/avajaeaZaUaCZvZjZTZIYEXbVhXeWXQbQ7PTOAN1MKJlJEIqHvFQER",
    "DECMCDBVAk9k8c8b8H7i736J000000000000cLcKcFcCc3b/bubsblbibcbWbUbPbJbBb7b1axaqakaVaQaOaGa0ZlZVZDYsXpXAXbWS81bjbbbQWOW7dTd5YJXcS7RIR9PlP3NALZLT",
    "K/JpJfItIDHXGZG5FXErDwD7CkChBZA+A+AS9X8a8C6WcIcEcCc8c4c2bybxbtbobebabYbSbPbLbJb5a/avajagaZaTaCZwZlZUZMYCXaU+Y5XSRvQoQXPxOvO9M5M2LaKhJXJGIpIK",
    "H/GTFvFeFNExDWDCCjCCBGA7A69H8l87cLcKcFcCc3b/bubsblbibcbWbUbPbIbBb7b1ayarakaVaPaNaGa1ZmZUZDYrXqWnY1XK00bjbbbQVmVidTd5XXWvU/TxRyQcPgP4NQN9MnMH",
    "LpL7KmJrJVJCILI7HKGqGGG1FJEjE7D+D2BdA/AncIcFcCc9c4c2bybwbsbnbebabXbSbPbLbJb4a+atalagaXaSaCZwZmZWZQY9XXUKXWWPV1TrRbQ+QuPrORNoNFMqMOLtLBKsKFK7",
    "JdJCIpIPHeGxGYG3FlFOEfDTCPBIcLcKcEcCc3b/bubrblbhbdbWbVbQbIbCb8b2azasakaVaPaKaGa3ZoZQZDYrXtRsXbWM8vbjbabSWCVAdSd4XTWBVnUXSzRYRLPCOxONNwNQMjMH",
    "LsLQLLKpKkKDJlJFIlIHHjGZFzFXFPEeDUC6cHcEcBc9c4c2bybwbsbnbdbabXbTbObKbJb4ayasahafaRaPaBZwZjZUZPY6XSVMX3W2VEVCT0RNQwQUQCPcO/OENVNCMwMJLvLlLIKr",
    "KUK4JvJJIqIcH/HYG1FcEDDHcLcIcDcBc2b+bubrblbhbbbWbUbPbHbBb6b1ayapakaUaOaIaFa3ZmZOZEYsXvUjXfVw00bgbUb8VrVZdRd3WeWGViULTZRuRLQoPfPdP6OgOPN/NPMO",
    "M4LrLPKzKlK1JuJNJLINI5HrHSGlFqCjcIcFcBc9c5c2bzbwbsbnbbbbbXbUbMbLbKb6axarajafaPaMaAZxZmZUZMY3XJWoWZVvUdU4TnT9SORwQrQEPuPNP7OGO1NLM+MbMCLqLUL1",
    "KoKQJbJFI+IPIFHaGxDKcLcIcCcBc2b+bvbqblbibdbYbUbObGbBb7b3a+aqakaVaOaHaDa6ZrZJZGYuY1XSWnVZ70bgbVbJXPWudWdBYXXeX7VcVDTWTGSoRuRYQwQePYPOOhOPNlNj",
    "NFMwMVM8LeLAL7KUJyJjJBIrHRFrcbcRcGcBcAc4c2bybubsbibbbabYbNbLbJb9b0awamadaaaNaAa1ZqZgZPYAXMWcYwWYW8V4URTmSdRbRZR3QgQEQ3PvPHP1OiOQO6NdN4M/MWMU",
    "LqLiLPL9KgIkIgHbcfcNcGcBc9c4b/brbobmbgbcbUbPbJbHb9b5azavakaUaNaJaFa6ZwZPZGYsYCXlZ2Xc8rcGbybdYGXOdudZYhXzX0WdVFUpUeTkSbSDRtRTRFQnQNQLQ7PMPLP2",
    "OgOQN+NBN1MUMJM6LCL5JfHcdVcrcacKcFc9c8c5bzbsbqbmbgbZbXbPbNbLbCb2aoafaaaTaGa3ZuZpZVYKXAWlYLY0XJWsWHVhVcUSUEToTfT3SXSGRjRWRMRAQtQNQBPpPdP8OqOE",
    "NZNQNBM9LBKsdYcmcRcIcBc4c1btbrblbhbfbZbUbNbGbDb3b1atakaaaSaEaDa3ZtZjZBYdXtX1YNY900dXchbdWWV7ePe4XFWVW5V7T+TcTLSXSNRrRWR0QvQ8PoPTPGOnO9O0NhNL",
    "N9MuL/LWLRKkKXJtIxEfe7e4dzd5cQcFcCc9c2c0bvboblbZbVbPbLbCbAb8arajabaXaDaAZsZkZ7YfXEW2WyWoW5VsVUUyUtUTU8T/TdTJT8SjSPSLSBRvRiRWR3QmQVQEPpP5OtOG",
    "NhN7MjHmd/dvdTdMd8cscqcccIc9b+btbmbhbbbXbObAb5b0anakabaKaCZzZxZoZVZJXkV/XwVm8deEdrbdQJQ9eQe7RORDR4QbPWO/OjOXNyNbN5MvMULqLHKtKLK8JiJ6ISHbHQGK",
    "G5FHF6DCC6BeAL8meAe7e0dHchcJcDc9c5c1bvbpbmbZbUbPbNbEbAb9asaeaYaUaEaAZsZbZ0YzXdV/R+RQRJR3QwQgQXQLQHP+PePMPAP0OmOcOONvNSMlMWMKM3LbKxKHJyJSI+Im",
    "GwFDe5dwdcdSd8c+cuchcNcFcAbxbnbmbfbYbQbFb8b2avadaZaQaHaDZ+ZfZTYyXMU4SZRy00eMeAbdP8OueRe7Q4PxPePOOLNlNRNDMsMSLvLhLNKQJvJpJ9IxIRHxHGGOG8F9EgE9",
    "DkCBBK9t9I6keBe8e1dIchcJcDc9c5c1bvbpbnbZbUbPbNbEbAbAasafaYaUaEaBZsZbZ5YzXlVqQbQDQ1PnPiPOPCP7OxOgOKODNyNjNXNKNBMsMNLdLNKsKTKKK1JaJ6I6HoGPFIE6",
    "e6dwdedUd8d0cvcicOcGcAbxbobmbgbZbPbHb8b3awaeaZaTaJaFZ+ZcZSYrX7USRGQp00ePeDbdNwNteRe8O/OhObOQNFMqMMM0LoLTKwKhKOJJIvIiI8HtHHGyGIFfEqEBDXD2CnAb",
    "AB918R7PeCe8e2dKcicJcDc9c5c2bvbpbnbZbUbPbObEbAbAatafaYaUaEaBZsZbZ7YyXqVhPWPAP1OiOTOGO4N+NtNVNHN8M+MZMWM9M5LoLKL2KVKDJuJDIuI9H+HBGmFxFDBRe7dx",
    "dfdVd9d1cwcjcPcHcAbxbpblbhbabPbIb7b4axaeaaaVaLaGZ+ZaZRYmW+V6QAPm00eSeGbd0000dTd5000000000000000000000000000000000000000000000000000000000000",
    "0000cHcEcBc8c4c3bybwbtbnbdbabZbTbPbLbIb5a/avajaeaZaUaCZvZhZTZFYFXcVr0000000000000000000000000000000000000000000000000000000000000000cLcJcFcC",
    "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",
    "aWaOaKaDZoZiZfYvYTWpTlTcPhdrdjbTTNT4dndbUHUBTwSoSMS6RfRRROQwQlQZQVQ9Q7PhPNPGP1OgOZOCO2NsNXNRMqMPLTLMK9JmeaeOe8cucVc6c2bybtbkbdbMbIb9b5b0azat",
    "apaaaRaIa7a4ZxZTZMYyYfXUXHUuU3TqTaTRT8SvSeSTSHSAR+RrRhRVRMRFQxQtQiQcQAP/PqPZP8P4OgOENwNLMJJke6d/dpdSdGdCd4crcqcJcCc1b/bibZbWbGbDb9axaoaiabaT",
    "aGa0ZvZRZ9YdYDWOTTTJPJdodjbATaTCdmdYUUUITySvSMR/RtRqRNR0QuQXQTQHQ3PxPJPIP0OkOUOHN+NqNKNCM/MHLZKzKgJFeYeQe8cycQc2b+btbjbPbEb5a+alafaVaRaOaAa2",
    "Z+ZnZaZRZHZ7YhYBXfXXVTUQU6TlTSTGSsSjSWSHS5RkRdRWRQR9R2QvQpQWQPQIQ5PpPdPSPJP2OoOPNeNPMNHae5dvdldRd8cwcpcXcOcAc9btbkbdbJbCb8azalaaaYaQaJaAZzZY",
    "ZGZBYxYFXYUYTMTLPOdjdead0000dTd50000000000000000000000000000000000000000000000000000000000000000cHcEcBc8c4c3bybwbtbnbdbabZbTbPbLbIb5a/avajae",
    "aZaUaCZvZhZTZFYFXcVr0000000000000000000000000000000000000000000000000000000000000000cLcJcFcCc3b/bubsblbibcbVbUbPbJbBb7b1axaqakaUaRaOaGZ/ZlZV",
    "ZDYsXoXF000000bjbbbQaBZ2dUd7VBV6V1UzUaU2SGRRP2OvOCNGJeIrIKI2GcFyFXFBCFBXAp8f8O6W645o5R420000cHcEcBc8c4c3bybwbtbobdbabZbTbPbLbIb5a/avajaeaZaU",
    "aCZvZiZTZFYEXbVoVBV8V5UsUUTkSdRwPMOqOQNWL1K2IqHzHpH5GLFSEBDQCiBxAVA98c7E5s5h5c00cLcJcFcCc3b/bubsblbibcbWbUbPbJbBb7b1axaqakaUaRaOaGZ/ZlZVZDYs",
    "XqXDaGZrYNbjbfbPZYYndVd8V0UxUtUqUhUaUSU3TrTaSVS3RHPrPLOhODMlMfM8LxLdL2KMJJHdH4GhE7DzCeBycHcEcBc8c4c2bybwbtbobdbabYbTbPbLbIb5a/avajaeaZaUaCZv",
    "ZiZTZFYEXaVnVBV4V0UvUZUYUSTxTpTSTHRxQxQhPYOdOCNbNQMxMPM0LjL4KlJKIoHxHCGYG0DocLcJcFcCc3b/bubsblbibcbVbVbPbJbBb7b1axaqakaUaQaNaGZ/ZmZVZDYsXqXB",
    "ZgZYXjbfbbbVYSXedTd9V8V7V6UuUuUrUkUjUVUVUTU0TnT7SZS8S3RrRDRBQwQaQDP5OiOcO6N2MZM7L4JWcHcEcBc8c4c2bzbwbtbnbdbabYbTbObLbIb5a/avajaeaZaTaDZvZhZT",
    "ZFYDXWVeV9V1U/UwUmUkUkUaUZUXUWUKTpTKSqSCS1ReRCQjQZQ4PnPaOtOgOKO8NkMUM5IfcLcJcFcCc2b/bubsblbibcbVbUbPbJbCb7b1axaqakaUaQaNaGa0ZnZUZEYuXqX4Z9X+",
    "WebkbbbHYKY5dQdBVDVAV6V4U+UzUvUqUfUbUWUVUSUQUJTkTVTJSsSQRuRpRJR7QaPqPXP5OmOOM/LtcHcEcBc7c3c2bzbwbtbnbdbabYbSbObLbIb5a+avajaeaYaTaEZvZiZQZFYB",
    "XSVPVCVAV7V7V6U/UuUsUqUkUeUYUXURU8TrTUTJT7SVSORgQvQqQKPtPgPLO/OMNXLbcLcIcEcCc2b+bubsblbibbbVbUbObJbCb7b0axaqalaTaQaNaGa1ZoZSZDYuXoWvYSXEUzbj",
    "bYbBXxXmdQdDVCV8V7V5V3V2V0UwUuUtUrUnUiUhUZUXUVUSUET/TZTESrSjRqRkRLQbPsPFOqNFcGcEcBc6c2c1bybwbtblbcbbbWbRbPbLbHb4azawakaeaWaTaEZxZiZNZHX+XTUO",
    "VDV9V8V8V7V5V2U/UuUiUhUeUcUbUaUZUXUTUTU4TlTVTES+SJRaR6QjQ6PjPQMWcKcIcEcAc1bybubsbmbhbabUbSbNbJbCb6a+ayaqakaSaRaMaCa3ZqZMZCYpXqVoXfXCUUbzbbbL",
    "Y9XRdsdSVCVAV9V7V6V5U/UxUxUwUvUtUqUpUnUkUTURUGTqTjTST9SvSQRnRRR0Q+PtOXNCcGcDcBc3c1c0bzbvbubjbebYbVbQbMbJbHb2a+auamagaUaNa6ZuZjZfYyYBWpVNVCVB",
    "V9V8V5V4V2UzUtUtUqUjUiUhUcUaUZUWUTUPT/TVT8SRSLRoRSQ/QtQFPPOPcMcHcEc6b/bwbvbqbnbgbZbSbObJbHbBb5a/axaladaXaQaLa8a0ZkZNZ9YoYBX3XxW+UVd4cGbaY4XY",
    "eAdvV9V9V6V1V0UxUxUtUcUbUXUUUTUTUTUMU6TbTRTDSiSgS4RSR8QoQgQHQ1P3OdMAcKcIcFc9c6c4c2bubsbqbkbZbYbSbNbJbDb9b3awapafaZaNa9a5ZiZeZ0YXXNRTVDV7V5V3",
    "V2UsUrUmUmUhUaUXUWUVUUUUUEU6TvTWT0SfSVSGS0RnQxQiQNPyPLNwcScOcHc9c3bzbxbtbnblbcbUbQbLbIbEb8b4ayarahaZaTaPaBZ/ZsZfZTZ6YQXRY9XSU5dYdCbaXJX5eqeb",
    "VCV6V4V4V3V0U+U+UyUtUrUrUqUoUiUbUXUUUTUSUSUQUETiTcSeSXRwRWRAQKMBcdcacVcTcOcKcEcAc8c1bwbpbobebcbWbTbLbDb8b0avalabaMaHa5ZzZKZ6YSWSVCVCVAV7V5V1",
    "UzUvUsUkUiUhUhUgUfUZUYUWUVUUUTUTUSUQToTJTBSaRvRcR6QhckcfcacScHcGcAc4bzbxbubrbebabQbPbObLbGb9araoahaaaPaMa1ZgZXZHY2WzXhXTU8eiePboL2KDekebVCV6",
    "V5V5V1U+UxUwUnUfUdUbUZUZUYUYUWUWUTUTUSUSURUGTSSpSHRUQgQJPrNudWdNdIdHdEdAd9d3d2cycucmclcgcccUcQcOcFc7c2bwbibVbJasaeaQZmZ2Y5VpVCVBVAV9V7V7V7V4",
    "V2V1V0U+UzUxUxUoUlUkUjUfUbUWUUUSUSULU2TESlSHRWMQdadYdMdFd8d4d1c+cvcqcnclcecXcTcPcLcHcAb/btbkbYbPbEb0ajaLZFYmXgWoLWLIHrejeZbiCFB+eOeFV2UZUTUD",
    "TxTmTWTHStSaSSRvRbR0QiQEPZOjOYN+NUMlM9LzLKKBK3JPH+H0GdEVeCe8e6e1dvdtdsdmdddddXdUdMdHdBd9d7c+cucecVcIc0bgbXbOb3acaIZyX/TKVDVCTuTPTLT8T1SpSmSc",
    "SZSNSDS5RwRlRcRYRFR4QxQgQFQAPqPaP8ORNRMkLpFneJeHe1drdpdhdedZdVdQdMdGd9d5d1cxcrcncicZcUcJc6bjbhbYbCaxZ/YbXETIDoDF9feNeDbLHJGvdhdYVDVCVAUzUrUq",
    "UhUTUSUQU0TFSqS4RMRBQqPXOaNoNgN0MZLgLGKqJVJ3HmG0FdEKf0e+etejeceYeTeLe3dtdodddVdRdKdHd7d4c+cpcbcJc2btbMbLataca5ZtXKV7VAVAV7V7V2UzUuUsUrUhUhUc",
    "UaUSURUJUAU8TtTaTASySrSZSDRoRBQiQJQ6OSMdfBfAeUeLe7e6e1dpdfdcdUdOdLd8d0cwcuckchcWcMcGc6bcbDb6atanaGa3ZZYWJWJ8FTdgdWaf",
);
// END OF TABLES

#[cfg(test)]
mod tests {
    use super::*;

    /// Training runs exactly as the tables were made.
    pub mod record {
        use super::super::*;
        use crate::math::arch::{Init, MlpSpec, Model, Norm, Place};
        use crate::math::autograd::{Act, Tape};
        use crate::math::optim::{self, Aux, Hyper, Optimizer, Rule};
        use crate::math::terrain::{Draw, Problem, Sampler};

        pub fn problem() -> Problem {
            Problem::spirals(POINTS, 3, &mut Rng::new(DATA_SEED))
        }

        pub fn spec() -> MlpSpec {
            MlpSpec {
                inputs: 2,
                width: WIDTH,
                depth: DEPTH,
                outputs: 3,
                act: Act::Tanh,
                norm: Norm::None,
                place: Place::Pre,
                residual: false,
                residual_scale: 1.0,
                init: Init::Lecun,
                gain: 1.0,
                bias: true,
            }
        }

        /// The rule and its settings; `adam_lr` is the step the AdamW run found best,
        /// which Muon's run gives the parameters Muon does not take.
        pub fn hyper(who: Who, lr: f64, adam_lr: f64) -> (Rule, Hyper) {
            match who {
                Who::Sgd => (Rule::Momentum, Hyper { lr, ..Hyper::default_for(Rule::Momentum) }),
                Who::AdamW => (Rule::Adam, Hyper { lr, ..Hyper::adamw() }),
                Who::Muon => (Rule::Muon, Hyper { lr, aux: Aux::adamw(adam_lr), ..Hyper::muon_reference() }),
            }
        }

        /// What one run records.
        #[derive(Clone, Debug, Default)]
        pub struct Run {
            /// The loss on all the data at each of `LOSS_AT` (infinite once blown up).
            pub loss: Vec<f64>,
            /// At each of `SPECTRA_AT` (with `spectra`): `PER_SPECTRUM` numbers.
            pub spectra: Vec<f64>,
            /// For Muon's hidden matrices at each of `SPECTRA_AT` after the first: the
            /// singular values of the matrix handed to the iteration, over its
            /// Frobenius norm plus the iteration's 1e-7 (what the iteration starts from).
            pub inputs: Vec<Vec<f64>>,
        }

        pub fn run(who: Who, lr: f64, adam_lr: f64, batch: usize, seed: u64, spectra: bool) -> Run {
            let p = problem();
            let mut m = Model::mlp(&spec(), &mut Rng::new(seed));
            let (rule, h) = hyper(who, lr, adam_lr);
            let mut opt = Optimizer::new(&m, rule, h);
            let mut tape = Tape::new();
            let mut batches = Sampler::new(p.len(), batch, Draw::WithoutReplacement, Rng::new(seed ^ 0xBA7C_5EED));
            let mats: Vec<usize> = m.groups().iter().enumerate().filter(|(_, g)| g.rows > 1 && g.cols > 1).map(|(i, _)| i).collect();
            assert_eq!(mats.len(), MATRICES.len());
            for (g, x) in mats.iter().zip(&MATRICES) {
                let gr = &m.groups()[*g];
                assert!(gr.rows == x.rows && gr.cols == x.cols && gr.name == format!("{} weights", x.name));
                assert_eq!(opt.as_matrix(*g), who == Who::Muon && x.muon, "{}", x.name);
            }
            let mut out = Run::default();
            let mut sc = optim::Scratch::new();
            let mut sv = Vec::new();
            let mut blown = false;
            for t in 0..=STEPS {
                if t > 0 && !blown {
                    let idx = batches.deal().to_vec();
                    p.grad(&mut m, &mut tape, &idx);
                    opt.step(&mut m, lr);
                }
                if LOSS_AT.contains(&t) {
                    let l = if blown { f64::INFINITY } else { p.loss(&m, &mut tape) };
                    blown |= !(l.is_finite() && l <= 1e8);
                    out.loss.push(if blown { f64::INFINITY } else { l });
                }
                if spectra && SPECTRA_AT.contains(&t) {
                    for (&g, x) in mats.iter().zip(&MATRICES) {
                        let gr = &m.groups()[g];
                        if t == 0 {
                            out.spectra.extend(std::iter::repeat_n(0.0, x.k()));
                        } else {
                            optim::singular_values_into(opt.update(g), gr.rows, gr.cols, &mut sv, &mut sc);
                            out.spectra.extend_from_slice(&sv);
                        }
                        optim::singular_values_into(&gr.value, gr.rows, gr.cols, &mut sv, &mut sc);
                        out.spectra.extend_from_slice(&sv);
                        if t > 0 && who == Who::Muon && x.muon {
                            let u = opt.ortho_in(g);
                            optim::singular_values_into(u, gr.rows, gr.cols, &mut sv, &mut sc);
                            let d = optim::frobenius(u) + 1e-7;
                            out.inputs.push(sv.iter().map(|s| s / d).collect());
                        }
                    }
                }
            }
            out
        }

        /// Every run of the scene: for each batch size, the sweeps of SGD with momentum
        /// and AdamW, then Muon's (whose other parameters take AdamW at its best step for
        /// that batch), then each rule's run at its best step again with its spectra.
        /// Returns the two tables' numbers, in the order `Record` reads them.
        pub fn everything() -> (Vec<f64>, Vec<f64>) {
            let n = LOSS_AT.len();
            let mut loss = vec![0.0; BATCHES.len() * SEEDS.len() * 3 * GRID * n];
            let mut spectra = vec![0.0; BATCHES.len() * SEEDS.len() * 3 * SPECTRA_AT.len() * PER_SPECTRUM];
            for (b, &batch) in BATCHES.iter().enumerate() {
                let mut best = [0usize; 3];
                for who in Who::ALL {
                    let adam_lr = Who::AdamW.grid()[best[Who::AdamW.index()]];
                    let runs: Vec<(usize, usize, Run)> = std::thread::scope(|sc| {
                        let hs: Vec<_> = (0..GRID)
                            .flat_map(|k| (0..SEEDS.len()).map(move |s| (k, s)))
                            .map(|(k, s)| sc.spawn(move || (k, s, run(who, who.grid()[k], adam_lr, batch, SEEDS[s], false))))
                            .collect();
                        hs.into_iter().map(|h| h.join().unwrap()).collect()
                    });
                    for (k, s, r) in runs {
                        let at = Record::loss_index(b, s, who, k);
                        loss[at..at + n].copy_from_slice(&r.loss);
                    }
                    // (the best step from the table as it will be read, rounding and all)
                    let rec = Record { loss: unpack(&pack(&loss)), spectra: Vec::new() };
                    best[who.index()] = rec.best(b, who);
                }
                let adam_lr = Who::AdamW.grid()[best[Who::AdamW.index()]];
                let runs: Vec<(usize, Who, Run)> = std::thread::scope(|sc| {
                    let hs: Vec<_> = Who::ALL
                        .into_iter()
                        .flat_map(|w| (0..SEEDS.len()).map(move |s| (w, s)))
                        .map(|(w, s)| sc.spawn(move || (s, w, run(w, w.grid()[best[w.index()]], adam_lr, batch, SEEDS[s], true))))
                        .collect();
                    hs.into_iter().map(|h| h.join().unwrap()).collect()
                });
                for (s, w, r) in runs {
                    let at = Record::spectrum_index(b, s, w, 0, 0);
                    spectra[at..at + SPECTRA_AT.len() * PER_SPECTRUM].copy_from_slice(&r.spectra);
                }
            }
            (loss, spectra)
        }
    }

    /// A caption says this.
    fn says(k: usize, what: &str) {
        assert!(TRIALS[k].caption.contains(what), "experiment {k} should say \"{what}\": {}", TRIALS[k].caption);
    }

    /// The scene as experiment k's button sets it up.
    fn scene(k: usize) -> MuonAdam {
        let mut s = MuonAdam::create();
        s.set_up(k);
        s.follow(0.0, true);
        s
    }

    /// The bars at least half as tall as the first.
    fn tall(sv: &[f32]) -> usize {
        let top = sv.first().copied().unwrap_or(0.0);
        sv.iter().filter(|&&v| top > 0.0 && v >= 0.5 * top).count()
    }

    /// The lowest and highest of some numbers, written as the captions write them.
    fn span(vals: impl Iterator<Item = f64>, write: impl Fn(f64) -> String) -> (String, String) {
        let v: Vec<f64> = vals.collect();
        let lo = v.iter().copied().fold(f64::INFINITY, f64::min);
        let hi = v.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        (write(lo), write(hi))
    }

    /// The tables hold what they should: every run of a seed starts from the same loss
    /// (1.1 to two figures), and nothing is out of place.
    #[test]
    fn the_tables_are_whole() {
        let r = record();
        assert_eq!(r.loss.len(), BATCHES.len() * SEEDS.len() * 3 * GRID * LOSS_AT.len());
        assert_eq!(r.spectra.len(), BATCHES.len() * SEEDS.len() * 3 * SPECTRA_AT.len() * PER_SPECTRUM);
        for b in 0..BATCHES.len() {
            for s in 0..SEEDS.len() {
                for w in Who::ALL {
                    for k in 0..GRID {
                        let c = r.curve(b, s, w, k);
                        assert_eq!(c[0], r.curve(0, s, Who::Sgd, 0)[0], "{b} {s} {w:?} {k}: every run of a seed starts alike");
                        assert!(g(c[0] as f64) == "1.1", "{b} {s} {w:?} {k}: starts at {}", c[0]);
                        // (a run that blew up stays blown up)
                        if let Some(i) = c.iter().position(|v| !v.is_finite()) {
                            assert!(c[i..].iter().all(|v| v.is_infinite()));
                        }
                    }
                    for c in 0..SPECTRA_AT.len() {
                        for m in 0..MATRICES.len() {
                            let (d, wt) = (r.spectrum(b, s, w, c, m, true), r.spectrum(b, s, w, c, m, false));
                            assert!(d.iter().chain(wt).all(|v| v.is_finite() && *v >= 0.0));
                            assert!(wt.windows(2).all(|p| p[0] >= p[1]), "largest first");
                            assert_eq!(c == 0, d.iter().all(|v| *v == 0.0), "a step after the start, none before");
                        }
                    }
                }
            }
            // every rule's best is inside its grid, with a worse step on either side
            for w in Who::ALL {
                let k = r.best(b, w);
                assert!(k > 0 && k + 2 < GRID, "{b} {w:?}: best at {k}");
            }
        }
        // the same start for every rule of a seed
        for s in 0..SEEDS.len() {
            let w0 = r.spectrum(0, s, Who::Sgd, 0, 1, false);
            for w in Who::ALL {
                assert_eq!(w0, r.spectrum(1, s, w, 0, 1, false));
            }
        }
    }

    /// The encoding keeps a number to within 0.6 % of itself, and 0 and infinity as they are.
    #[test]
    fn the_encoding_keeps_numbers_to_a_hundredth() {
        for v in [1e-11, 3.3e-9, 0.0123, 0.5, 1.0986, 7.0, 123.4, 9.9e7] {
            let back = unpack(&pack(&[v]))[0] as f64;
            assert!((back / v - 1.0).abs() < 0.006, "{v} -> {back}");
        }
        let back = unpack(&pack(&[0.0, 1e-13, f64::INFINITY, f64::NAN, 2e8]));
        assert_eq!(back[..2], [0.0, 0.0]);
        assert!(back[2..].iter().all(|v| v.is_infinite()));
        assert_eq!(g(0.0085), "0.0085");
        assert_eq!(g(0.2216), "0.22");
        assert_eq!(g(7.857e-5), "$7.9e-5$");
        assert_eq!(g(9.96e-5), "$1.0e-4$");
        assert_eq!(g(653.0), "650");
        assert_eq!(g(1.035), "1");
        assert_eq!(lr_text(0.0128), "0.0128");
        assert_eq!(lr_text(1.25e-4), "0.000125");
    }

    /// The set-up the panel states: the network's size, and which matrices each rule's
    /// run gives to which rule.
    #[test]
    fn the_set_up_is_as_stated() {
        use crate::math::arch::Model;
        use crate::math::optim::{Optimizer, Rule};
        let m = Model::mlp(&record::spec(), &mut Rng::new(1));
        assert_eq!(m.n_params(), 2307);
        let (rule, h) = record::hyper(Who::Muon, 0.008, 0.0128);
        let opt = Optimizer::new(&m, rule, h);
        let taken: Vec<&str> = m.groups().iter().enumerate().filter(|(i, _)| opt.as_matrix(*i)).map(|(_, g)| g.name.as_str()).collect();
        assert_eq!(taken, ["layer 2 weights", "layer 3 weights"]);
        assert!(m.groups().iter().enumerate().all(|(i, _)| opt.as_matrix(i) || opt.rule_of(i) == Rule::Adam));
        assert_eq!(h.aux.lr, 0.0128);
        assert_eq!(STEPS, *LOSS_AT.last().unwrap());
        assert_eq!(STEPS, *SPECTRA_AT.last().unwrap());
    }

    #[test]
    fn each_rule_at_its_best_step() {
        let s = scene(0);
        let r = record();
        assert_eq!((s.setup.batch, s.setup.seed, s.setup.shift, s.setup.step), (0, 0, [0; 3], STEPS as f64));
        for w in Who::ALL {
            says(0, &format!("{} {}", w.name(), lr_text(w.grid()[r.best(0, w)])));
            says(0, &format!("{} at {}", w.name(), g(r.mean_final(0, w, r.best(0, w)))));
        }
        // all three above 0.8 up to step 400, one under it by step 501
        let curves: Vec<&[f32]> = Who::ALL.iter().map(|&w| r.curve(0, 0, w, r.best(0, w))).collect();
        for (i, &t) in LOSS_AT.iter().enumerate() {
            if t <= 400 {
                assert!(curves.iter().all(|c| c[i] > 0.8), "step {t}");
            }
        }
        let at501 = LOSS_AT.iter().position(|&t| t == 501).unwrap();
        assert!(curves.iter().any(|c| c[at501] < 0.8));
        says(0, "For about the first 400 steps all three (seed 1) stay above 0.8, near where they started (1.1)");
        // Muon lowest in every seed, by these factors over AdamW
        let (km, ka, ks) = (r.best(0, Who::Muon), r.best(0, Who::AdamW), r.best(0, Who::Sgd));
        for sd in 0..SEEDS.len() {
            let (m, a, sg) = (r.final_loss(0, sd, Who::Muon, km), r.final_loss(0, sd, Who::AdamW, ka), r.final_loss(0, sd, Who::Sgd, ks));
            assert!(m < a && a < sg, "seed {sd}");
        }
        let (lo, hi) = span((0..SEEDS.len()).map(|sd| r.final_loss(0, sd, Who::AdamW, ka) / r.final_loss(0, sd, Who::Muon, km)), g);
        says(0, &format!("in every seed Muon ended lowest, {lo} to {hi} times below AdamW"));
    }

    #[test]
    fn every_step_four_times_too_long() {
        let s = scene(1);
        let r = record();
        assert_eq!(s.setup.shift, [2; 3]);
        for w in Who::ALL {
            assert_eq!(s.setup.pick(w), r.best(0, w) + 2);
            says(1, &format!("{} {}", w.name(), lr_text(w.grid()[r.best(0, w) + 2])));
        }
        let finals = |b: usize, w: Who| span((0..SEEDS.len()).map(|sd| r.final_loss(b, sd, w, r.best(b, w) + 2)), g);
        let start = r.curve(0, 0, Who::Sgd, 0)[0] as f64;
        let (a, b) = finals(0, Who::Muon);
        says(1, &format!("Muon's three runs still end between {a} and {b}"));
        let (a, b) = finals(0, Who::Sgd);
        let (c, d) = finals(0, Who::AdamW);
        says(1, &format!("SGD's end between {a} and {b} and AdamW's between {c} and {d}, against {} at the start", g(start)));
        for w in [Who::Sgd, Who::AdamW] {
            assert!((0..SEEDS.len()).all(|sd| r.final_loss(0, sd, w, r.best(0, w) + 2) > 0.5 * r.curve(0, sd, w, 0)[0] as f64), "{w:?}");
        }
        let (a, b) = finals(1, Who::Sgd);
        let (c, d) = finals(1, Who::AdamW);
        let (e, h) = finals(1, Who::Muon);
        says(1, &format!("four times the best step leaves SGD between {a} and {b}, AdamW between {c} and {d}, and Muon between {e} and {h}"));
        // the range of step sizes that end below 0.01 on average
        for b in 0..BATCHES.len() {
            for w in Who::ALL {
                if let Some((lo, hi)) = r.tolerated(b, w) {
                    assert!((lo..=hi).all(|k| r.mean_final(b, w, k) < TARGET), "{b} {w:?}: a gap");
                }
            }
        }
        let factor = |b: usize, w: Who| r.tolerated(b, w).map(|(lo, hi)| 2usize.pow((hi - lo) as u32));
        assert_eq!(factor(0, Who::Sgd), None);
        assert_eq!(factor(0, Who::AdamW), Some(1));
        says(1, &format!("span a factor of {} for Muon, one step size for AdamW and none for SGD at batches of 8", factor(0, Who::Muon).unwrap()));
        says(1, &format!("at batches of 128 a factor of {} for SGD, {} for AdamW", factor(1, Who::Sgd).unwrap(), factor(1, Who::AdamW).unwrap()));
        let (lo, hi) = r.tolerated(1, Who::Muon).unwrap();
        assert_eq!(lo, 0, "Muon's range reaches the smallest step tried");
        says(1, &format!("for Muon every step size from the smallest tried to {}, a factor of {}", lr_text(Who::Muon.grid()[hi]), 2usize.pow(hi as u32)));
    }

    /// What five rounds of the reference's iteration do to one singular value s (as a
    /// share of the matrix's Frobenius norm): every s from 0.0016 to 1 ends between 0.68
    /// and 1.21, and those below 0.0013 end lower (the polynomial of Jordan and co-authors,
    /// 2024; computed in the course's notes, muon.md §8.3, where the largest s that ends
    /// below 0.68 is about 0.0015).
    #[test]
    fn the_iteration_lifts_only_what_is_large_enough() {
        use crate::math::optim::Iteration;
        let it = Iteration::JORDAN;
        let (mut lo, mut hi) = (f64::INFINITY, 0.0f64);
        for i in 0..=200_000 {
            let s = 0.0016 + (1.0 - 0.0016) * i as f64 / 200_000.0;
            let v = it.scalar(s);
            lo = lo.min(v);
            hi = hi.max(v);
        }
        assert!(lo >= 0.68 && hi <= 1.21, "{lo} {hi}");
        for s in [1e-4, 5e-4, 1e-3, 0.0013] {
            assert!(it.scalar(s) < 0.68, "{s}");
        }
    }

    #[test]
    fn the_steps_shape_early_and_late() {
        let s = scene(2);
        let r = record();
        assert!(s.film.playing && s.setup.step == 1.0 && s.setup.layer == 1 && s.setup.batch == 0 && s.setup.seed == 0);
        let cs = 1..SPECTRA_AT.len();
        let step = |w: Who, c: usize| r.spectrum(0, 0, w, c, 1, true);
        for w in [Who::Sgd, Who::AdamW] {
            assert!(cs.clone().all(|c| tall(step(w, c)) <= 3), "{w:?}");
        }
        says(2, "one to three singular values stand above the rest (at half the largest or more)");
        let (a, b) = span(cs.clone().flat_map(|c| [stable_rank(step(Who::Sgd, c)), stable_rank(step(Who::AdamW, c))]), |v| num(v, 1));
        says(2, &format!("their stable ranks stay between {a} and {b}"));
        let (a, b) = span(cs.clone().map(|c| tall(step(Who::Muon, c)) as f64), |v| num(v, 0));
        says(2, &format!("Muon's step has between {a} and {b} of its 32 singular values at half the largest or more"));
        let (a, b) = span(cs.clone().map(|c| stable_rank(step(Who::Muon, c))), |v| num(v, 1));
        says(2, &format!("with stable ranks from {a} to {b}"));
        says(2, "to between 0.68 and 1.2 only if it is at least about 0.0015 of that matrix's size");
    }

    #[test]
    fn a_small_batch_against_a_large_one() {
        let s = scene(3);
        let r = record();
        assert!(s.setup.batch == 1 && s.setup.other && s.setup.shift == [0; 3]);
        for w in Who::ALL {
            let (a, b) = (w.grid()[r.best(0, w)], w.grid()[r.best(1, w)]);
            let grew = (b / a).round() as usize;
            let what = if grew == 1 { format!("not at all ({})", lr_text(a)) } else { format!("{grew} times ({} to {})", lr_text(a), lr_text(b)) };
            says(3, &format!("{}'s {what}", w.name()));
        }
        let reach = |b: usize, w: Who| -> Vec<Option<usize>> { (0..SEEDS.len()).map(|sd| r.reaches(b, sd, w, r.best(b, w), TARGET)).collect() };
        let mean = |v: &[Option<usize>]| v.iter().map(|x| x.unwrap_or(STEPS) as f64).sum::<f64>() / v.len() as f64;
        for w in Who::ALL {
            let v = reach(1, w);
            assert!(v.iter().all(|x| x.is_some()));
            let (a, b) = span(v.iter().map(|x| x.unwrap() as f64), |x| format!("{x}"));
            says(3, &format!("by steps {a} to {b} for {}", w.name()));
        }
        assert!(reach(0, Who::Sgd).iter().all(|x| x.is_none()));
        let adam = reach(0, Who::AdamW);
        let got: Vec<String> = adam.iter().flatten().map(|x| format!("{x}")).collect();
        assert_eq!(got.len(), 2);
        says(3, &format!("did by steps {} and {} in two seeds of three for AdamW", got[0], got[1]));
        let (a, b) = span(reach(0, Who::Muon).iter().map(|x| x.unwrap() as f64), |x| format!("{x}"));
        says(3, &format!("by steps {a} to {b} for Muon"));
        let gain = |w: Who| mean(&reach(0, w)) / mean(&reach(1, w));
        let (gm, ga, gs) = (gain(Who::Muon), gain(Who::AdamW), gain(Who::Sgd));
        assert!(gm < ga && gm < gs);
        let f = |w: Who| g(r.mean_final(1, w, r.best(1, w)));
        says(3, &format!("Yet Muon ended lowest, on average {} against {} for SGD with momentum and {} for AdamW", f(Who::Muon), f(Who::Sgd), f(Who::AdamW)));
        says(3, &format!("So the larger batch saved Muon the fewest steps: {} times fewer, against at least {} for AdamW and at least {} for SGD with momentum", num(gm, 1), num(ga.floor(), 0), num((gs * 10.0).floor() / 10.0, 1)));
        // Muon's lowest loss, where, and the climb after it
        let km = r.best(1, Who::Muon);
        let lows: Vec<(f64, usize)> = (0..SEEDS.len())
            .map(|sd| {
                let c = r.curve(1, sd, Who::Muon, km);
                let i = (0..c.len()).min_by(|&a, &b| c[a].total_cmp(&c[b])).unwrap();
                (c[i] as f64, LOSS_AT[i])
            })
            .collect();
        for (sd, (low, _)) in lows.iter().enumerate() {
            assert!(r.final_loss(1, sd, Who::Muon, km) > 10.0 * low, "seed {sd} climbs");
        }
        let (a, b) = span(lows.iter().map(|l| l.0), g);
        let (c, d) = span(lows.iter().map(|l| l.1 as f64), |x| format!("{x}"));
        says(3, &format!("after falling to between {a} and {b} by steps {c} to {d} and climbing back"));
        assert!(r.mean_final(1, Who::Muon, r.best(1, Who::Muon)) < r.mean_final(1, Who::Sgd, r.best(1, Who::Sgd)).min(r.mean_final(1, Who::AdamW, r.best(1, Who::AdamW))));
    }

    #[test]
    fn the_weights_at_the_end() {
        let s = scene(4);
        let r = record();
        assert!(s.setup.layer == 1 && s.setup.step == STEPS as f64 && s.setup.batch == 1 && s.setup.seed == 0);
        let end = SPECTRA_AT.len() - 1;
        let w_at = |b: usize, sd: usize, w: Who, c: usize, m: usize| r.spectrum(b, sd, w, c, m, false);
        let start = w_at(1, 0, Who::Sgd, 0, 1);
        says(4, &format!("white, dashed: stable rank {}", num(stable_rank(start), 1)));
        let z = |w: Who| w_at(1, 0, w, end, 1);
        // here the largest grew by the most under every rule; in every seed and at both
        // batch sizes it grew more than 2.5 times, and the stable rank fell
        for w in Who::ALL {
            let grew: Vec<f64> = (0..WIDTH).map(|i| z(w)[i] as f64 - start[i] as f64).collect();
            assert!(grew[1..].iter().all(|&x| x < grew[0]), "{w:?}");
        }
        for b in 0..2 {
            for sd in 0..SEEDS.len() {
                for w in Who::ALL {
                    let (a, e) = (w_at(b, sd, w, 0, 1), w_at(b, sd, w, end, 1));
                    assert!(e[0] > 2.5 * a[0] && stable_rank(e) < 0.8 * stable_rank(a), "{b} {sd} {w:?}");
                }
            }
        }
        says(4, &format!("the largest singular value grew most: from {} to {} under SGD with momentum, {} under AdamW and {} under Muon", g(start[0] as f64), g(z(Who::Sgd)[0] as f64), g(z(Who::AdamW)[0] as f64), g(z(Who::Muon)[0] as f64)));
        let mid = WIDTH / 2;
        says(4, &format!("its middle one grew from {} to {}, while SGD's ended at {} and AdamW's fell to {}", g(start[mid] as f64), g(z(Who::Muon)[mid] as f64), g(z(Who::Sgd)[mid] as f64), g(z(Who::AdamW)[mid] as f64)));
        assert!(z(Who::AdamW)[mid] < start[mid] && z(Who::Muon)[mid] > 2.0 * start[mid]);
        let rose = |w: Who| span((0..SEEDS.len()).map(|sd| (0..WIDTH).filter(|&i| w_at(1, sd, w, end, 1)[i] > w_at(1, sd, w, 0, 1)[i]).count() as f64), |v| num(v, 0));
        let (a, b) = rose(Who::Muon);
        let (c, d) = rose(Who::AdamW);
        says(4, &format!("Muon raised {a} to {b} of the 32 singular values, AdamW {c} to {d}"));
        says(4, &format!("stable rank {}, against {} and {}", num(stable_rank(z(Who::Muon)), 1), num(stable_rank(z(Who::Sgd)), 1), num(stable_rank(z(Who::AdamW)), 1)));
        // Muon's layer 2 highest everywhere
        let all = |w: Who| (0..2).flat_map(move |b| (0..SEEDS.len()).map(move |sd| (b, sd))).map(move |(b, sd)| stable_rank(w_at(b, sd, w, end, 1)));
        let muon_low = all(Who::Muon).fold(f64::INFINITY, f64::min);
        let others_high = all(Who::Sgd).chain(all(Who::AdamW)).fold(0.0, f64::max);
        assert!(muon_low > others_high);
        let (a, b) = span(all(Who::Muon), |v| num(v, 1));
        let (c, d) = span(all(Who::Sgd).chain(all(Who::AdamW)), |v| num(v, 1));
        says(4, &format!("Muon's layer 2 ended with the highest stable rank: {a} to {b}, against {c} to {d} for the other two"));
        // layer 3: highest at batches of 8, not in every seed at 128
        for sd in 0..SEEDS.len() {
            let m = stable_rank(w_at(0, sd, Who::Muon, end, 2));
            assert!(Who::ALL[..2].iter().all(|&w| stable_rank(w_at(0, sd, w, end, 2)) < m));
        }
        let at128: Vec<f64> = Who::ALL.iter().flat_map(|&w| (0..SEEDS.len()).map(move |sd| (w, sd))).map(|(w, sd)| stable_rank(w_at(1, sd, w, end, 2))).collect();
        assert!((0..SEEDS.len()).any(|sd| at128[6 + sd] < at128[sd].max(at128[3 + sd])));
        let (a, b) = span(at128.iter().copied(), |v| num(v, 1));
        says(4, &format!("at batches of 128 the three end close together ({a} to {b})"));
    }

    /// What the panel says this network shows beside the published explanations.
    #[test]
    fn what_this_network_says_beside_the_explanations() {
        let (_, sb) = range_of(Who::Sgd, stable_rank);
        let (_, ab) = range_of(Who::AdamW, stable_rank);
        assert!(sb < 2.5 && ab < 2.5, "close to rank one: {sb} {ab}");
        let (ma, _) = range_of(Who::Muon, top_over_mean);
        assert!(ma > 2.0, "not about 1: {ma}");
        let (ra, _) = range_of(Who::AdamW, top_over_mean);
        assert!(ra > 5.0);
        // (the primer: the steps of SGD with momentum on layer 2 have stable ranks below 3,
        // at every recorded step, seed and batch size)
        let r = record();
        for b in 0..BATCHES.len() {
            for sd in 0..SEEDS.len() {
                for c in 1..SPECTRA_AT.len() {
                    assert!(stable_rank(r.spectrum(b, sd, Who::Sgd, c, 1, true)) < 3.0, "{b} {sd} {c}");
                }
            }
        }
        let text = why_here();
        assert!(text.contains("close to rank one") && !text.contains("NaN") && !text.contains('?'), "{text}");
    }

    /// The scene opens as the first experiment sets it up.
    #[test]
    fn the_scene_opens_on_the_first_experiment() {
        let opened = MuonAdam::create();
        let mut pressed = MuonAdam::create();
        pressed.set_up(3);
        pressed.set_up(OPENING);
        assert_eq!(opened.setup, pressed.setup);
        assert!(!opened.film.playing);
    }

    /// Every setting a control can reach, and some none can: every number drawn is finite.
    #[test]
    fn every_setting_draws_finite_numbers() {
        let mut s = MuonAdam::create();
        for batch in [0, 1, 7] {
            for seed in [0, 2, 9] {
                for layer in [0, 1, 2, 3, 9] {
                    for shift in [[0; 3], [2; 3], [-2; 3], [-99, 99, 5], [i32::MIN, i32::MAX, 0]] {
                        for step in [1.0, 1.5, 999.0, 2000.0, f64::NAN, -5.0, 1e30] {
                            s.setup = Setup { batch, seed, layer, shift, step, other: batch == 1 }.kept();
                            s.follow(0.016, true);
                            let l = &s.look;
                            for g in [&l.ranges, &l.curves, &l.finals, &l.bars, &l.weights, &l.rel] {
                                assert!(g.shown.iter().all(|v| v.is_finite()), "{:?}", s.setup);
                            }
                            assert!((1..SPECTRA_AT.len()).contains(&s.setup.checkpoint()));
                            let r = &l.ranges.shown;
                            assert!(r[1] > r[0] && r[3] > r[2] && r[5] > r[4] && r[7] > r[6], "{r:?}");
                        }
                    }
                }
            }
        }
    }

    /// The layout keeps everything in the window and apart, from a phone to a desktop.
    #[test]
    fn the_layout_fits_every_window() {
        for (w, h) in [(1220.0, 1000.0), (1600.0, 1000.0), (1100.0, 700.0), (900.0, 600.0), (390.0, 800.0), (360.0, 640.0), (700.0, 500.0), (300.0, 600.0), (500.0, 400.0), (240.0, 180.0)] {
            let lay = Layout::of(Vec2::new(w, h), 40.0);
            let inside = |(a, b): Rect| a.x >= 0.0 && a.y >= 0.0 && b.x <= w + 0.5 && b.y <= h + 0.5 && a.is_finite() && b.is_finite();
            let apart = |(a, b): Rect, (c, d): Rect| b.x <= c.x || d.x <= a.x || b.y <= c.y || d.y <= a.y;
            assert!(inside(lay.loss), "{w}x{h}: {:?}", lay.loss);
            let buttons: Vec<Rect> = lay.buttons.to_vec();
            for (i, a) in buttons.iter().enumerate() {
                for b in &buttons[i + 1..] {
                    assert!(apart(*a, *b), "{w}x{h}: buttons {a:?} {b:?}");
                }
            }
            if lay.compact {
                continue;
            }
            // (each chart with the room round it for its title and axes)
            let rm = room(lay.narrow);
            let grown = |(a, b): Rect| (a - Vec2::new(rm[0] - 6.0, rm[1] - 4.0), b + Vec2::new(rm[2] - 4.0, rm[3] - 6.0));
            let bars = (lay.bars[0].0 - Vec2::new(20.0, rm[1] + 14.0), lay.bars[2].1 + Vec2::new(4.0, 6.0));
            let mut boxes: Vec<Rect> = vec![grown(lay.loss), grown(lay.lr), grown(lay.weights), bars];
            if lay.rel.1.x - lay.rel.0.x > 2.0 {
                boxes.push((lay.rel.0 - Vec2::new(40.0, 20.0), lay.rel.1));
            }
            boxes.extend(lay.boxes.iter().copied());
            boxes.extend(buttons.iter().copied());
            for (i, a) in boxes.iter().enumerate() {
                assert!(inside(*a), "{w}x{h}: box {i} {a:?}");
                for b in &boxes[i + 1..] {
                    assert!(apart(*a, *b), "{w}x{h}: {a:?} and {b:?} overlap");
                }
            }
            for c in [lay.loss, lay.lr, lay.weights, lay.bars[0]] {
                assert!(c.1.y - c.0.y > 40.0 && c.1.x - c.0.x > 60.0, "{w}x{h}: {c:?}");
            }
        }
    }

    /// The film plays from step 1 to 2000 in its length, and the picture comes to rest.
    #[test]
    fn the_film_plays_to_the_end_and_rests() {
        assert!((Film::step_at(0.0) - 1.0).abs() < 1e-12 && (Film::step_at(Film::LENGTH) - STEPS as f64).abs() < 1e-9);
        for c in [0.5, 3.0, 7.9] {
            assert!((Film::clock_of(Film::step_at(c)) - c).abs() < 1e-9);
        }
        let mut s = scene(2);
        let (mut frames, mut last) = (0, 0.0);
        while s.animated() {
            s.advance(1.0 / 60.0);
            s.follow(1.0 / 60.0, false);
            assert!(s.setup.step >= last, "the film went back");
            last = s.setup.step;
            frames += 1;
            assert!(frames < 60 * 30, "still playing");
        }
        assert_eq!(s.setup.step, STEPS as f64);
        assert!((frames as f64 / 60.0 - Film::LENGTH).abs() < 1.0, "{frames}");
    }

    /// After every experiment's button the glides arrive and stop.
    #[test]
    fn glides_arrive_and_stop() {
        let mut s = MuonAdam::create();
        for k in [3, 0, 1, 4, 2, 3] {
            s.set_up(k);
            s.film.playing = false;
            let mut frames = 0;
            s.follow(1.0 / 60.0, false);
            while s.look.moving {
                s.follow(1.0 / 60.0, false);
                frames += 1;
                assert!(frames < 300, "experiment {k} still moving");
            }
        }
    }

    /// The panel's reason for an MLP: the course's small transformer (2 blocks 16 wide,
    /// 2 heads) on 256 of the library's repeated sequences of 12 tokens from 8, trained
    /// the same way at batches of 8 (seed 1, each rule's best of six step sizes, Muon's
    /// other parameters at AdamW's best), ends with the three rules' best final losses
    /// within a factor of 1.5, and below the least possible on new sequences: (ln 7 + ln
    /// 6 + ln 5 + ln 4 + ln 3) / 11 = 0.712 (the first copy's tokens after the first are
    /// drawn from those not yet used; the rest can be predicted). (Run with --ignored.)
    #[test]
    #[ignore]
    fn the_transformer_shows_less() {
        use crate::math::arch::{Model, Norm, Place, TransformerSpec};
        use crate::math::autograd::{Act, Tape};
        use crate::math::optim::Optimizer;
        use crate::math::terrain::{Draw, Problem, Sampler};
        let floor = (2..=7).map(|k| (k as f64).ln()).sum::<f64>() / 11.0 - 2f64.ln() / 11.0;
        assert!((floor - 0.712).abs() < 5e-4, "{floor}");
        let p = Problem::repeats(256, 12, 8, &mut Rng::new(DATA_SEED));
        let spec = TransformerSpec { vocab: 8, context: 12, d_model: 16, heads: 2, layers: 2, d_mlp: 32, act: Act::Gelu, norm: Norm::Layer, place: Place::Pre, residual: true, init_gain: 1.0, tied_unembed: false };
        let train = |who: Who, lr: f64, adam: f64| {
            let mut m = Model::transformer(&spec, &mut Rng::new(SEEDS[0]));
            let (rule, h) = record::hyper(who, lr, adam);
            let mut opt = Optimizer::new(&m, rule, h);
            let mut tape = Tape::new();
            let mut batches = Sampler::new(p.len(), BATCHES[0], Draw::WithoutReplacement, Rng::new(SEEDS[0] ^ 0xBA7C_5EED));
            for _ in 0..STEPS {
                let idx = batches.deal().to_vec();
                p.grad(&mut m, &mut tape, &idx);
                opt.step(&mut m, lr);
            }
            p.loss(&m, &mut tape)
        };
        let sweep = |who: Who, lo: f64, adam: f64| -> (f64, f64) {
            std::thread::scope(|sc| {
                let hs: Vec<_> = (0..6).map(|k| lo * 2f64.powi(k)).map(|lr| sc.spawn(move || (train(who, lr, adam), lr))).collect();
                hs.into_iter().map(|h| h.join().unwrap()).filter(|v| v.0.is_finite()).fold((f64::INFINITY, 0.0), |a, b| if b.0 < a.0 { b } else { a })
            })
        };
        let sgd = sweep(Who::Sgd, 0.005, 0.0);
        let adam = sweep(Who::AdamW, 4e-4, 0.0);
        let muon = sweep(Who::Muon, 5e-4, adam.1);
        let best = [sgd.0, adam.0, muon.0];
        let (lo, hi) = (best.iter().copied().fold(f64::INFINITY, f64::min), best.iter().copied().fold(0.0, f64::max));
        assert!(hi < 1.5 * lo && hi < floor, "{best:?}");
        // on the MLP, by contrast, AdamW's best ends far above Muon's
        let r = record();
        assert!(r.mean_final(0, Who::AdamW, r.best(0, Who::AdamW)) > 20.0 * r.mean_final(0, Who::Muon, r.best(0, Who::Muon)));
    }

    /// The fourth experiment's last sentence: Muon's run at batches of 128 (seed 1, its
    /// best step) made again, with layers 2 and 3 held still from step 900 (a new
    /// optimiser whose Muon step is 1e-12, the rest AdamW at its step as before): the
    /// loss at step 2000 stays within a factor 1.5 of the loss at step 900, where the
    /// recorded run climbed more than ten-fold. (A few seconds: run with --ignored.)
    #[test]
    #[ignore]
    fn muons_climb_is_its_own_steps() {
        use crate::math::arch::Model;
        use crate::math::autograd::Tape;
        use crate::math::optim::{Aux, Hyper, Optimizer, Rule};
        use crate::math::terrain::{Draw, Sampler};
        let r = record();
        let (km, ka) = (r.best(1, Who::Muon), r.best(1, Who::AdamW));
        let (lr, adam) = (Who::Muon.grid()[km], Who::AdamW.grid()[ka]);
        let p = record::problem();
        let mut m = Model::mlp(&record::spec(), &mut Rng::new(SEEDS[0]));
        let (rule, h) = record::hyper(Who::Muon, lr, adam);
        let mut opt = Optimizer::new(&m, rule, h);
        let mut step_lr = lr;
        let mut tape = Tape::new();
        let mut batches = Sampler::new(p.len(), BATCHES[1], Draw::WithoutReplacement, Rng::new(SEEDS[0] ^ 0xBA7C_5EED));
        let mut at900 = 0.0;
        for t in 1..=STEPS {
            if t == 901 {
                at900 = p.loss(&m, &mut tape);
                let still = Hyper { lr: 1e-12, aux: Aux::adamw(adam), ..Hyper::muon_reference() };
                opt = Optimizer::new(&m, Rule::Muon, still);
                step_lr = still.lr;
            }
            let idx = batches.deal().to_vec();
            p.grad(&mut m, &mut tape, &idx);
            opt.step(&mut m, step_lr);
        }
        let end = p.loss(&m, &mut tape);
        assert!(end < 1.5 * at900 && end > 0.0, "{at900} -> {end}");
        let recorded = r.curve(1, 0, Who::Muon, km);
        let i900 = LOSS_AT.iter().position(|&t| t == 891).unwrap();
        assert!(*recorded.last().unwrap() > 10.0 * recorded[i900]);
    }

    /// Muon's bars are what its iteration does to the momentum's singular values: in a
    /// run made again (batches of 8, seed 1, its best step), at every recorded step the
    /// singular values of layer 2's step are the step size times five rounds of the
    /// reference's polynomial applied to the momentum's singular values (each over its
    /// Frobenius norm plus 1e-7), to 1e-5 of the step size; and every bar below half the
    /// first comes from a
    /// singular value under 0.0016 of the momentum's size. (A few seconds: run with
    /// --ignored.)
    #[test]
    #[ignore]
    fn muons_bars_are_what_its_iteration_does() {
        use crate::math::optim::Iteration;
        let r = record();
        let (km, ka) = (r.best(0, Who::Muon), r.best(0, Who::AdamW));
        let lr = Who::Muon.grid()[km];
        let run = record::run(Who::Muon, lr, Who::AdamW.grid()[ka], BATCHES[0], SEEDS[0], true);
        for (c, inputs) in run.inputs.chunks(2).enumerate() {
            let at = (c + 1) * PER_SPECTRUM + matrix_offset(1);
            let step: Vec<f64> = run.spectra[at..at + WIDTH].to_vec();
            let mut want: Vec<(f64, f64)> = inputs[0].iter().map(|&s| (lr * Iteration::JORDAN.scalar(s), s)).collect();
            want.sort_by(|a, b| b.0.total_cmp(&a.0));
            for (got, (w, _)) in step.iter().zip(&want) {
                // (below about 1e-5 of the step size a singular value is rounding, here amplified)
                assert!((got - w).abs() <= 1e-5 * lr, "step {}: {got} against {w}", SPECTRA_AT[c + 1]);
            }
            let top = step[0];
            for (w, s) in &want {
                if *w < 0.5 * top {
                    assert!(*s < 0.0016, "step {}: a short bar from {s}", SPECTRA_AT[c + 1]);
                }
            }
        }
    }

    /// The tables, made again from nothing, are the ones in this file (run with
    /// `--ignored`; set MUONADAM_TABLES to a file's path to have the new tables written
    /// there, as the source text to put between the lines `// TABLES` and `// END OF
    /// TABLES`).
    #[test]
    #[ignore]
    fn the_tables_are_what_training_gives() {
        let (loss, spectra) = record::everything();
        let (new_loss, new_spectra) = (pack(&loss), pack(&spectra));
        if let Ok(path) = std::env::var("MUONADAM_TABLES") {
            let lines = |s: &str| s.as_bytes().chunks(140).map(|c| format!("    \"{}\",\n", std::str::from_utf8(c).unwrap_or(""))).collect::<String>();
            let text = format!("const LOSS_TABLE: &str = concat!(\n{});\n\nconst SPECTRA_TABLE: &str = concat!(\n{});\n", lines(&new_loss), lines(&new_spectra));
            std::fs::write(&path, text).expect("could not write the tables");
        }
        let old: String = LOSS_TABLE.chars().filter(|c| ALPHABET.contains(&(*c as u8))).collect();
        let old_s: String = SPECTRA_TABLE.chars().filter(|c| ALPHABET.contains(&(*c as u8))).collect();
        assert!(old == new_loss, "the loss table has changed");
        assert!(old_s == new_spectra, "the spectra table has changed");
    }
}

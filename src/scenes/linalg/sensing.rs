//! Compressed sensing: a sparse signal recovered exactly from far fewer measurements
//! than it has entries, by choosing the answer with the least 1-norm.

use super::*;
use crate::clock::{Budget, Pace};
use crate::math::sparse::{self, Check, Homotopy, Shortest, Solution, Status};
use std::cell::{OnceCell, RefCell};
use std::collections::HashMap;

pub const DEF: SceneDef = SceneDef {
    id: "sensing",
    title: "Compressed sensing",
    group: "Linear algebra: data and dimensions",
    blurb: "Recover a sparse signal from far fewer measurements than it has unknowns.",
    make: |_| Box::new(Sensing::new()),
};

/// The sizes the signal can have.
const N_MIN: usize = 32;
const N_MAX: usize = 128;
/// A reconstruction whose error is below this share of the signal's length counts as
/// exact. (Where recovery works the error is rounding, 10⁻¹² or less; where it fails,
/// per cents: `exact_and_wrong_are_far_apart`.)
const EXACT: f64 = 1e-4;
/// Trials behind each point of the recovery curves.
const TRIALS: u32 = 50;
/// Points of the curves measured per second (one trial of each kind at one m): the
/// curves fill in over a few seconds, within a time allowance per frame. (Headless
/// renders, which have no time limit, take a sixtieth of that per frame.)
const TRIAL_RATE: f32 = 500.0;
/// How closely the curves as drawn follow the trials, in seconds of the scene's time
/// (see `Drawn`).
const FOLLOW: f32 = 0.06;
/// A point of a curve as drawn has caught up when it is this close to the share
/// measured (a hundredth of a pixel on the tallest chart).
const CAUGHT_UP: f32 = 2e-5;
/// The most steps of the solver one frame may take, whatever its time allowance says.
const MOST_STEPS: usize = 400;
/// How long a replay of the solve takes, in seconds.
const REPLAY: f32 = 4.0;

/// The colours: one thing each.
const TRUTH: Color = pal::WHITE;
const L1: Color = pal::GREEN;
const L2: Color = pal::PINK;
const ENTRIES: Color = pal::PURPLE;
const M_MARK: Color = pal::YELLOW;
const HEAT: Cmap = Cmap::IceFire;

/// How the measurements are taken.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Kind {
    /// Each a sum of all the entries with random weights.
    Random,
    /// The first m entries, read directly (A = the first m rows of the identity).
    Entries,
}

/// Everything an experiment sets.
#[derive(Clone, Copy, Debug)]
struct Setup {
    n: usize,
    k: usize,
    m: usize,
    kind: Kind,
    /// Seeds of the signal and of the matrix.
    signal: u64,
    matrix: u64,
}

const fn setup(n: usize, k: usize, m: usize, kind: Kind) -> Setup {
    Setup { n, k, m, kind, signal: 1, matrix: 1 }
}

/// The scene opens on the first of these. Every claim here is checked by a test of
/// the same setting (and of other signals and matrices where chance is involved).
const TRY: [(&str, &str, Setup); 5] = [
    (
        "enough measurements",
        "40 measurements of 100 unknowns, 8 of them not zero. The green answer lands exactly on the signal: the same 8 positions, the same values, an error of $10^{-15}$ or so, which is rounding. On the curve, $m = 40$ is past the step: signals like this come back nearly every time.",
        setup(100, 8, 40, Kind::Random),
    ),
    (
        "too few",
        "The same signal and the first 20 of the same measurements. The green answer still agrees with all 20, and its 1-norm is smaller than the signal's, but it is a different spray of 20 spikes: wrong. On the curve, $m = 20$ is before the step.",
        setup(100, 8, 20, Kind::Random),
    ),
    (
        "not sparse enough",
        "40 measurements again, but now 20 entries are not zero: too many for 40. The green answer is wrong, a spray of 40 spikes, and the step on the curve has moved right, to about $m = 50$.",
        setup(100, 20, 40, Kind::Random),
    ),
    (
        "the shortest answer is a smear",
        "90 measurements of 100 unknowns. The pink answer, the shortest that agrees with them, is still wrong in every one of the 100 entries: it spreads the signal out, and misses it by about a quarter of its length. The green answer is exact.",
        setup(100, 8, 90, Kind::Random),
    ),
    (
        "read single entries instead",
        "$A$ now reads the first 40 entries directly. Those come back right, but nothing at all is learned about the other 60, so the non-zero entries among them are lost, whatever the method. The purple curve: this works only if all 8 non-zero entries happen to be among those read, which takes nearly all 100.",
        setup(100, 8, 40, Kind::Entries),
    ),
];

/// The values of m at which the recovery curves are measured.
fn grid(n: usize) -> Vec<usize> {
    let step = n.div_ceil(50).max(1);
    let mut ms: Vec<usize> = (step..=n).step_by(step).collect();
    if ms.last() != Some(&n) {
        ms.push(n);
    }
    ms
}

/// The chance of exact recovery against m, measured by trials: for both kinds of
/// measurement, at every m of the grid, so many trials of a fresh random signal (and,
/// for random sums, a fresh random matrix).
struct Curves {
    n: usize,
    k: usize,
    ms: Vec<usize>,
    /// (exact recoveries, trials) at each m.
    random: Vec<[u32; 2]>,
    entries: Vec<[u32; 2]>,
    rng: Rng,
    /// Where the next trial is in the grid, and how many rounds of the grid are done.
    next: usize,
    rounds: u32,
    /// Trials the solver could not finish (none, for these matrices: the tests).
    unsolved: u32,
}

impl Curves {
    fn new(n: usize, k: usize) -> Curves {
        let ms = grid(n);
        Curves { n, k, random: vec![[0, 0]; ms.len()], entries: vec![[0, 0]; ms.len()], ms, rng: Rng::new(7 + 1000 * n as u64 + k as u64), next: 0, rounds: 0, unsolved: 0 }
    }

    fn complete(&self) -> bool {
        self.rounds >= TRIALS
    }

    /// One trial of each kind at the next m.
    fn trial(&mut self) {
        let (n, k) = (self.n, self.k);
        let Some(&m) = self.ms.get(self.next) else {
            self.next = 0;
            return;
        };
        let a = sparse::gaussian_matrix(m, n, &mut self.rng);
        let random = recovers(&a, m, n, k, &mut self.rng);
        let entries = Some(reads_all(m, &sparse::sparse_signal(n, k, &mut self.rng)));
        for (counts, outcome) in [(self.random.get_mut(self.next), random), (self.entries.get_mut(self.next), entries)] {
            if let Some(counts) = counts {
                counts[0] += (outcome == Some(true)) as u32;
                counts[1] += 1;
            }
            self.unsolved += outcome.is_none() as u32;
        }
        self.next += 1;
        if self.next >= self.ms.len() {
            self.next = 0;
            self.rounds += 1;
        }
    }

    /// The measured chance at `m` (between the two nearest points of the grid) and
    /// the fewest trials behind it.
    fn chance(&self, kind: Kind, m: usize) -> Option<(f32, u32)> {
        let counts = if kind == Kind::Random { &self.random } else { &self.entries };
        let at = |i: usize| counts.get(i).filter(|c| c[1] > 0).map(|c| (c[0] as f32 / c[1] as f32, c[1]));
        let above = self.ms.iter().position(|&g| g >= m)?;
        if self.ms[above] == m || above == 0 {
            return at(above);
        }
        let (lo, hi) = (at(above - 1)?, at(above)?);
        let t = (m - self.ms[above - 1]) as f32 / (self.ms[above] - self.ms[above - 1]) as f32;
        Some((lo.0 + (hi.0 - lo.0) * t, lo.1.min(hi.1)))
    }
}

/// The recovery curves as they are drawn. The trials come in whole, eight or nine in
/// a frame at 60 frames a second, each moving one point of each curve by up to one
/// trial's share, and in one frame of every six only points that stay at 0 or 100 %;
/// drawn as they came, the curves jerked and stood still by turns. So each point is
/// drawn following its measured share through two eases of `FOLLOW`, one after the
/// other (as `follow` in `packing.rs` does), exactly there in the end. `Curves` is
/// what was measured, and every number and word says that; this is only the picture.
#[derive(Default)]
struct Drawn {
    /// For random sums and for single entries, at each m of the grid: the share drawn,
    /// and the one on the way to it from the share measured (None before any trial).
    shown: [Vec<Option<f32>>; 2],
    between: [Vec<Option<f32>>; 2],
}

impl Drawn {
    /// The share measured at each m (None before any trial there).
    fn measured(counts: &[[u32; 2]]) -> impl Iterator<Item = Option<f32>> + '_ {
        counts.iter().map(|c| (c[1] > 0).then(|| c[0] as f32 / c[1] as f32))
    }

    fn caught_up(&self, curves: &Curves) -> bool {
        [&curves.random, &curves.entries].into_iter().zip(&self.shown).all(|(counts, shown)| shown.len() == counts.len() && Self::measured(counts).eq(shown.iter().copied()))
    }

    /// Bring the curves as drawn `dt` seconds of the scene's time nearer to the
    /// trials. A point measured for the first time is drawn where it was measured.
    fn follow(&mut self, curves: &Curves, dt: f32) {
        let k = 1.0 - (-dt.max(0.0) / FOLLOW).exp();
        for (kind, counts) in [&curves.random, &curves.entries].into_iter().enumerate() {
            let (shown, between) = (&mut self.shown[kind], &mut self.between[kind]);
            if shown.len() != counts.len() || between.len() != counts.len() {
                *shown = vec![None; counts.len()];
                *between = vec![None; counts.len()];
            }
            for ((to, shown), between) in Self::measured(counts).zip(shown.iter_mut()).zip(between.iter_mut()) {
                (*shown, *between) = match (to, *shown, *between) {
                    (Some(to), Some(s), Some(b)) => {
                        let b = b + (to - b) * k;
                        let s = s + (b - s) * k;
                        // (exactly there in the end, not nearly: the picture then stops changing)
                        if (to - s).abs().max((to - b).abs()) < CAUGHT_UP {
                            (Some(to), Some(to))
                        } else {
                            (Some(s), Some(b))
                        }
                    }
                    _ => (to, to),
                };
            }
        }
    }
}

/// Does basis pursuit recover a fresh k-sparse signal measured by `a` exactly? (None
/// if it could not be solved.)
fn recovers(a: &[f64], m: usize, n: usize, k: usize, rng: &mut Rng) -> Option<bool> {
    let x = sparse::sparse_signal(n, k, rng);
    let mut y = vec![0.0; m];
    sparse::apply(a, m, n, &x, &mut y);
    sparse::basis_pursuit(a, m, n, &y).map(|s| sparse::relative_error(&s.x, &x) < EXACT)
}

/// Is a signal measured by reading its first m entries recovered exactly? Those
/// entries are y itself, and nothing constrains the rest, so the least-1-norm answer
/// is y followed by zeros (`reading_entries_gives_back_what_was_read` in
/// `math/sparse.rs`, and `reading_entries_needs_no_solver` here): exact if and only if
/// every non-zero entry is among the first m. (Worked out so, it costs nothing beside
/// a trial of random sums.)
fn reads_all(m: usize, x: &[f64]) -> bool {
    x.iter().skip(m).all(|v| *v == 0.0)
}

/// Row i of the random matrix with seed `seed`: the same numbers whatever m is (one
/// more measurement adds a row and changes no other), and as many as there are
/// columns. Standard normal; the matrix divides them by √m.
fn random_row(seed: u64, i: usize, n: usize) -> Vec<f64> {
    let mut rng = Rng::new(seed.wrapping_mul(1_000_003).wrapping_add(i as u64));
    let mut row = vec![0.0; n];
    sparse::normals(&mut row, &mut rng);
    row
}

/// The rows of the random matrix with one seed and n, kept as they are drawn.
#[derive(Default)]
struct Rows {
    seed: u64,
    n: usize,
    /// Rows 0, 1, … one after another, n numbers each.
    drawn: Vec<f64>,
}

impl Rows {
    /// The first m rows for this seed and n, drawing those not drawn yet (all of them,
    /// for another seed or n).
    fn first(&mut self, seed: u64, m: usize, n: usize) -> &[f64] {
        if (seed, n) != (self.seed, self.n) {
            (self.seed, self.n) = (seed, n);
            self.drawn.clear();
        }
        let have = self.drawn.len() / n.max(1);
        for i in have..m {
            self.drawn.extend(random_row(seed, i, n));
        }
        &self.drawn[..m * n]
    }
}

pub struct Sensing {
    n: usize,
    k: usize,
    m: usize,
    kind: Kind,
    signal_seed: u64,
    matrix_seed: u64,
    show_l1: bool,
    show_l2: bool,
    /// The problem: A (m × n) as it is used, the numbers it is drawn from (for the
    /// picture), the signal x and its measurements y = A x.
    a: Vec<f64>,
    raw: Vec<f64>,
    x: Vec<f64>,
    y: Vec<f64>,
    /// The rows of random weights drawn so far for this matrix seed and n, as many as
    /// any m has needed. A row does not depend on m (`random_row`), so a new m (the
    /// handle on the chart, dragged) only draws the rows it adds.
    rows: Rows,
    /// The shortest solution, and what finds it for any right-hand side.
    shortest: Vec<f64>,
    projector: Option<Shortest>,
    /// The coherence of A, worked out when something first shows it: a phone's
    /// picture does not, and while m is dragged it would cost as much as the rest of
    /// the new problem together. Emptied with every new problem (`rebuild`).
    mu: OnceCell<Option<f64>>,
    /// Basis pursuit for this problem, whether it is still being solved, and its
    /// answer (None if it could not be solved).
    solver: Homotopy,
    solving: bool,
    answer: Option<Solution>,
    check: Option<Check>,
    /// How far a replay of the solve has got, from 0 to 1 (None when not replaying).
    replay: Option<f32>,
    /// Asked to replay before the answer was ready.
    replay_wanted: bool,
    /// The recovery curves as measured, and as drawn (following the measurements).
    curves: Curves,
    drawn: Drawn,
    pace: Pace,
    /// A, and y beside it, as a picture; repainted when the problem changes.
    heat: Option<Image>,
    heat_stale: bool,
    /// The words on coherence under the chart, broken into lines, and the labels'
    /// widths.
    note: Broken,
    widths: Widths,
}

impl Sensing {
    fn new() -> Sensing {
        let first = TRY[0].2;
        let mut s = Sensing {
            n: first.n,
            k: first.k,
            m: first.m,
            kind: first.kind,
            signal_seed: first.signal,
            matrix_seed: first.matrix,
            show_l1: true,
            show_l2: true,
            a: Vec::new(),
            raw: Vec::new(),
            x: Vec::new(),
            y: Vec::new(),
            rows: Rows::default(),
            shortest: Vec::new(),
            projector: None,
            mu: OnceCell::new(),
            solver: Homotopy::new(&[], 0, 0, &[]),
            solving: false,
            answer: None,
            check: None,
            replay: None,
            replay_wanted: false,
            curves: Curves::new(first.n, first.k),
            drawn: Drawn::default(),
            pace: Pace::default(),
            heat: None,
            heat_stale: true,
            note: Broken::default(),
            widths: Widths::default(),
        };
        s.apply(first);
        s
    }

    /// Set an experiment up.
    fn apply(&mut self, s: Setup) {
        (self.n, self.k, self.m, self.kind, self.signal_seed, self.matrix_seed) = (s.n, s.k, s.m, s.kind, s.signal, s.matrix);
        (self.show_l1, self.show_l2) = (true, true);
        self.rebuild();
    }

    /// Keep the settings within their ranges (N first: the others depend on it).
    fn tidy(&mut self) {
        self.n = self.n.clamp(N_MIN, N_MAX);
        self.k = self.k.clamp(1, self.n / 2);
        self.m = self.m.clamp(1, self.n);
    }

    /// A new problem for the settings: the matrix, the signal, the measurements, the
    /// shortest answer, and the start of the solve.
    fn rebuild(&mut self) {
        self.tidy();
        let (n, m, k) = (self.n, self.m, self.k);
        match self.kind {
            Kind::Random => {
                self.raw = self.rows.first(self.matrix_seed, m, n).to_vec();
                let scale = 1.0 / (m as f64).sqrt();
                self.a = self.raw.iter().map(|v| v * scale).collect();
            }
            Kind::Entries => {
                self.a = sparse::first_rows(m, n);
                // (drawn as bright as a large random weight)
                self.raw = self.a.iter().map(|v| v * 2.5).collect();
            }
        }
        self.x = sparse::sparse_signal(n, k, &mut Rng::new(self.signal_seed));
        self.y = vec![0.0; m];
        sparse::apply(&self.a, m, n, &self.x, &mut self.y);
        self.projector = Shortest::of(&self.a, m, n);
        self.shortest = match &self.projector {
            Some(p) => p.solve(&self.a, &self.y),
            None => vec![0.0; n],
        };
        self.mu = OnceCell::new();
        self.solver = Homotopy::new(&self.a, m, n, &self.y).recording();
        (self.solving, self.answer, self.check, self.replay) = (true, None, None, None);
        if self.curves.n != n || self.curves.k != k {
            self.curves = Curves::new(n, k);
        }
        self.heat_stale = true;
    }

    /// Carry the solve on, within `budget`; the answer once it is found.
    fn solve(&mut self, budget: &Budget) {
        if !self.solving {
            return;
        }
        let mut steps = 0;
        while self.solver.status() == Status::Running && steps < MOST_STEPS && !(steps > 0 && budget.spent()) {
            self.solver.step();
            steps += 1;
        }
        if self.solver.status() == Status::Running {
            return;
        }
        self.solving = false;
        self.answer = self.solver.solution();
        self.check = self.answer.as_ref().map(|s| Check::of(&self.a, self.m, self.n, &self.y, s));
        if std::mem::take(&mut self.replay_wanted) && self.answer.is_some() {
            self.replay = Some(0.0);
        }
    }

    /// Run the solve to the end at once (the tests).
    #[cfg(test)]
    fn settle(&mut self) {
        while self.solving {
            self.solve(&Budget::new(1.0));
        }
    }

    /// Where the solve is a share `t` of the way along: the penalty λ on the 1-norm
    /// a share t of the way from where every entry is zero to none. Returns the point
    /// of the path there (which is exact: between events it is a straight line).
    fn path_at(&self, t: f32) -> Option<Vec<f64>> {
        let n = self.n;
        let (path, lambdas) = (self.solver.path(), self.solver.lambdas());
        let first = *lambdas.first()?;
        let lambda = first * (1.0 - t.clamp(0.0, 1.0) as f64);
        // the last event at or above that λ, and the next
        let i = lambdas.iter().rposition(|&l| l >= lambda).unwrap_or(0);
        let j = (i + 1).min(lambdas.len() - 1);
        let u = if lambdas[i] > lambdas[j] { (lambdas[i] - lambda) / (lambdas[i] - lambdas[j]) } else { 0.0 };
        let (p, q) = (path.get(i * n..(i + 1) * n)?, path.get(j * n..(j + 1) * n)?);
        Some(p.iter().zip(q).map(|(p, q)| p + (q - p) * u.clamp(0.0, 1.0)).collect())
    }

    /// The answer a share `t` of the way through the solve, made to agree with every
    /// measurement: the entries chosen so far, plus the shortest correction that makes
    /// up what they leave of y. At the start (nothing chosen) it is the shortest
    /// answer; at the end, the least-1-norm one.
    fn along(&self, t: f32) -> Option<Vec<f64>> {
        let (n, m) = (self.n, self.m);
        let x = self.path_at(t)?;
        let mut ax = vec![0.0; m];
        sparse::apply(&self.a, m, n, &x, &mut ax);
        let rest: Vec<f64> = self.y.iter().zip(&ax).map(|(y, v)| y - v).collect();
        let fix = self.projector.as_ref()?.solve(&self.a, &rest);
        Some(x.iter().zip(&fix).map(|(x, f)| x + f).collect())
    }

    /// How many entries the solve has chosen a share `t` of the way through.
    fn chosen(&self, t: f32) -> usize {
        self.path_at(t).map_or(0, |x| x.iter().filter(|v| **v != 0.0).count())
    }

    /// What the green row shows: the replay's answer while replaying, else the answer.
    fn green(&self) -> Option<Vec<f64>> {
        match self.replay {
            Some(s) => self.along(s),
            None => self.answer.as_ref().map(|a| a.x.clone()),
        }
    }

    /// The coherence of A (None if a column of A is all zeros).
    fn mu(&self) -> Option<f64> {
        *self.mu.get_or_init(|| sparse::coherence(&self.a, self.m, self.n))
    }

    fn error_l1(&self) -> Option<f64> {
        self.answer.as_ref().map(|a| sparse::relative_error(&a.x, &self.x))
    }

    /// Paint A (and y in the column before it, one empty column between) into the picture.
    fn paint(&mut self, gfx: &Gfx) {
        let (n, m) = (self.n, self.m);
        let image = self.heat.get_or_insert_with(|| Image::new(gfx, (n + 2) as u32, m as u32));
        image.ensure_size(gfx, (n + 2) as u32, m as u32);
        let [w, h] = image.size();
        if (w as usize, h as usize) != (n + 2, m) {
            return;
        }
        let table = HEAT.table();
        let biggest = self.y.iter().fold(0.0f64, |b, v| b.max(v.abs())).max(1e-12);
        let pixels = image.pixels_mut();
        for i in 0..m {
            // (pixel rows count from the bottom, matrix rows from the top)
            let row = &mut pixels[(m - 1 - i) * (n + 2)..(m - i) * (n + 2)];
            row[0] = table[heat_index(self.y[i] / biggest)];
            row[1] = Color::CLEAR;
            for j in 0..n {
                row[j + 2] = table[heat_index(self.raw[i * n + j] / 2.5)];
            }
        }
        self.heat_stale = false;
    }
}

/// How many entries are not zero (to working precision: a billionth of the largest).
fn nonzero(x: &[f64]) -> usize {
    let big = x.iter().fold(0.0f64, |b, v| b.max(v.abs()));
    x.iter().filter(|v| v.abs() > 1e-9 * big).count()
}

/// Where a value between −1 and 1 goes in the heat map's colours: not all the way to
/// the ends, which are bright enough to shout.
fn heat_index(v: f64) -> usize {
    table_index(0.5 + 0.4 * v.clamp(-1.0, 1.0) as f32)
}

/// Where everything goes, in the units of the stage (five high, as wide as the canvas
/// makes it, centred on 0).
struct Layout {
    /// Text size, and the height of a line of it in units.
    size: f32,
    line: f32,
    /// Top of the equation block (its title's top), and the left edge of the column of y,
    /// with its width.
    top: f32,
    y_left: f32,
    y_w: f32,
    /// Left edge of A, the width of one of its columns, its top, and the height of one
    /// of its rows (square cells while they fit the room kept for A, flatter beyond).
    a_left: f32,
    pitch: f32,
    a_top: f32,
    row_h: f32,
    /// The three rows of stems: the signal, the least-1-norm answer, the shortest
    /// answer. Their baselines, and units per unit value.
    rows: [f32; 3],
    amp: f32,
    /// The recovery chart: bottom-left corner and size.
    chart: Vec2,
    chart_size: Vec2,
    /// The two norms of each answer, compared (top-left and width; none: no room).
    norms: Vec2,
    norms_w: f32,
    /// Where the words on coherence go (top-left), and how wide (none: no room).
    note: Vec2,
    note_w: f32,
}

impl Layout {
    fn of(w: f32, ppu: f32, top_inset: f32, n: usize, m: usize) -> Layout {
        let px = 1.0 / ppu.max(1e-3);
        let wide = w >= 4.4;
        let size = if wide { 12.5 } else { 10.5 };
        let line = (size + 6.0) * px;
        let top = 2.5 - top_inset * px - if wide { 14.0 } else { 6.0 } * px;
        let bottom = -2.5 + 6.0 * px;
        let (left, right) = if wide { (-w * 0.5 + 0.22, -w * 0.5 + 0.22 + 0.6 * w - 0.3) } else { (-w * 0.5 + 0.1, w * 0.5 - 0.1) };
        // the column of y (at least 8 pixels wide), a gap with "=", then n columns of A
        let gap = 18.0 * px;
        let pitch = ((right - left - gap - 8.0 * px) / n as f32).min((right - left - gap) / (n as f32 + 1.0)).max(1e-4);
        let y_w = pitch.max(8.0 * px);
        let a_left = left + y_w + gap;
        // two lines of words above A, and the ticks that mark the signal's columns
        let a_top = top - 2.0 * line - 20.0 * px;
        // room for A: as tall as 40 % of its width at most (so that 40 of 100 rows
        // have square cells), and no more than the stage can spare
        let room = (0.4 * n as f32 * pitch).min(if wide { 0.26 } else { 0.17 } * (top - bottom));
        let row_h = (room / m.max(1) as f32).min(pitch);
        let a_bottom = a_top - room;
        let (rows_bottom, chart, chart_size, note, note_w, norms, norms_w);
        if wide {
            rows_bottom = bottom + 1.2 * line;
            // the chart in a column of its own: three lines of words above it, the axis
            // and a key of two lines under it, then the words on coherence
            let (cl, cr) = (right + 0.62, w * 0.5 - 0.28);
            let chart_top = top - 3.3 * line;
            let ch = ((chart_top - bottom) * 0.6).min((cr - cl) * 1.1).max(0.3);
            chart = Vec2::new(cl, chart_top - ch);
            chart_size = Vec2::new((cr - cl).max(0.3), ch);
            // (each left out where the view is too short for it: a phone on its side)
            norms = Vec2::new(cl - 0.42, chart.y - 4.4 * line);
            norms_w = if norms.y - 12.5 * line > bottom { cr - cl + 0.42 } else { 0.0 };
            note = Vec2::new(cl - 0.42, norms.y - 13.4 * line);
            note_w = if norms_w > 0.0 && note.y - 6.0 * line > bottom { cr - cl + 0.42 } else { 0.0 };
        } else {
            // the chart under everything else, then its axis and key
            let (cl, cr) = (left + 34.0 * px, right - 0.05);
            let ch = ((top - bottom) * 0.2).clamp(0.3, 1.5);
            chart = Vec2::new(cl, bottom + 3.6 * line);
            chart_size = Vec2::new((cr - cl).max(0.3), ch);
            rows_bottom = chart.y + ch + 4.6 * line;
            (note, note_w, norms, norms_w) = (Vec2::ZERO, 0.0, Vec2::ZERO, 0.0);
        }
        // each row: its label, then stems up to 1.1 either way
        let band = ((a_bottom - 0.4 * line - rows_bottom) / 3.0).max(0.05);
        let amp = ((band - 1.3 * line) * 0.5 / 1.1).max(0.01);
        let rows = [0, 1, 2].map(|r| a_bottom - 0.4 * line - band * r as f32 - 1.3 * line - amp * 1.1);
        Layout { size, line, top, y_left: left, y_w, a_left, pitch, a_top, row_h, rows, amp, chart, chart_size, norms, norms_w, note, note_w }
    }

    /// The x of entry j's stem (under column j of A).
    fn at(&self, j: usize) -> f32 {
        self.a_left + (j as f32 + 0.5) * self.pitch
    }
}

/// Labels' widths, kept from frame to frame: most labels are what they were the frame
/// before. A width no frame has asked for since the one before is forgotten.
#[derive(Default)]
struct Widths {
    now: RefCell<HashMap<(String, u32), f32>>,
    last: HashMap<(String, u32), f32>,
}

impl Widths {
    /// A new frame: what the last one asked for is kept for this one.
    fn next_frame(&mut self) {
        self.last = std::mem::take(self.now.get_mut());
    }

    /// How many logical pixels `text` takes at `size`: one with a formula in it as it
    /// is drawn (`drawn_width`), plain words estimated by their number of characters,
    /// as `text_width` does.
    fn of(&self, f: &Frame, text: &str, size: f32) -> f32 {
        let key = (text.to_string(), size.to_bits());
        let mut now = self.now.borrow_mut();
        if let Some(&w) = now.get(&key) {
            return w;
        }
        let measure = || if crate::typeset::has_math(text) { drawn_width(f, text, size) } else { text_width(text, size) };
        let w = self.last.get(&key).copied().unwrap_or_else(measure);
        now.insert(key, w);
        w
    }

    /// The first of `options` that fits in `room` units at this zoom (the last if none
    /// does).
    fn pick<'a>(&self, f: &Frame, options: &[&'a str], room: f32, ppu: f32, size: f32) -> &'a str {
        options.iter().copied().find(|t| self.of(f, t, size) < room * ppu).unwrap_or(options[options.len() - 1])
    }
}

/// How wide a label with formulas in it is drawn: its words as the interface font sets
/// them, its formulas from the fonts' metrics. (The whole label is not laid out to
/// measure it, as `Frame::text_size` would: that takes a few tenths of a millisecond
/// for a label never measured before, and some labels here change with every round of
/// trials, wordings that are measured and then not drawn among them.)
fn drawn_width(f: &Frame, text: &str, size: f32) -> f32 {
    match crate::typeset::pieces(text) {
        Ok(parts) => parts
            .iter()
            .map(|part| match part {
                crate::typeset::Piece::Text(words) => f.text_size(words, size).x,
                crate::typeset::Piece::Math(formula) => text_width(&format!("${formula}$"), size),
            })
            .sum(),
        Err(_) => text_width(text, size),
    }
}

/// The words of a text, each formula whole however many spaces it has (a formula is
/// not broken across lines).
fn words(text: &str) -> Vec<&str> {
    let bytes = text.as_bytes();
    let (mut words, mut start, mut inside, mut i) = (Vec::new(), 0, false, 0);
    while i < bytes.len() {
        match bytes[i] {
            // (an escaped dollar is no edge of a formula)
            b'\\' => i += 1,
            b'$' => inside = !inside,
            b' ' if !inside => {
                words.push(&text[start..i]);
                start = i + 1;
            }
            _ => {}
        }
        i += 1;
    }
    words.push(&text[start.min(text.len())..]);
    words
}

/// A text broken into lines, kept with what it was broken for. The words on coherence
/// are drawn every frame on a wide screen, and a formula is worked out to be measured,
/// so they are broken into lines (between words, as `kit::wrapped` breaks them) only
/// when they change.
#[derive(Default)]
struct Broken {
    text: String,
    size: f32,
    room: f32,
    lines: Vec<String>,
}

impl Broken {
    /// The lines of `text` at `size`, none wider than `room` logical pixels (but a
    /// word wider by itself), each word as wide as `width` says: the lines kept, if
    /// they were broken for the same, else broken afresh and kept.
    fn lines(&mut self, text: &str, size: f32, room: f32, width: impl Fn(&str) -> f32) -> &[String] {
        if (self.text.as_str(), self.size, self.room) != (text, size, room) {
            let (mut lines, mut line) = (Vec::new(), String::new());
            // (the line's width so far: each word is measured once)
            let mut so_far = 0.0;
            for word in words(text) {
                let w = width(word);
                if line.is_empty() {
                    so_far = w;
                } else if so_far + size * 0.47 + w > room {
                    lines.push(std::mem::take(&mut line));
                    so_far = w;
                } else {
                    line.push(' ');
                    so_far += size * 0.47 + w;
                }
                line.push_str(word);
            }
            lines.push(line);
            *self = Broken { text: text.to_string(), size, room, lines };
        }
        &self.lines
    }
}

/// A small number written as a power of ten, as markup to go between dollars:
/// `3 \times 10^{-16}`.
fn tiny(v: f64) -> String {
    if v.is_nan() || v <= 0.0 {
        return "0".to_string();
    }
    let e = v.log10().floor();
    let mantissa = (v / 10f64.powf(e)).round() as i64;
    let (mantissa, e) = if mantissa >= 10 { (1, e + 1.0) } else { (mantissa, e) };
    format!(r"{mantissa} \times 10^{{{}}}", e as i64)
}

/// An error as a share of the signal: a power of ten when tiny (a formula), else per
/// cent.
fn error_text(e: f64) -> String {
    if e < 1e-4 {
        format!("${}$", tiny(e))
    } else if e < 0.01 {
        format!("{:.2} %", e * 100.0)
    } else {
        format!("{:.0} %", e * 100.0)
    }
}

impl Sensing {
    /// One row of stems: `values`, over faint bars of the signal (if `truth` is given).
    fn stems(f: &mut Frame, lay: &Layout, base: f32, values: &[f64], truth: Option<&[f64]>, color: Color, ppu: f32) {
        let n = values.len();
        let pitch_px = lay.pitch * ppu;
        let reach = lay.amp * 1.1;
        let y_of = |v: f64| base + (v as f32 * lay.amp).clamp(-reach, reach);
        let left = lay.a_left;
        f.line((left, base), (left + n as f32 * lay.pitch, base), 1.0, pal::DIM.a(0.6));
        if let Some(truth) = truth {
            let bar = (0.75 * pitch_px).clamp(2.0, 9.0);
            for (j, &t) in truth.iter().enumerate() {
                if t != 0.0 {
                    f.line((lay.at(j), base), (lay.at(j), y_of(t)), bar, TRUTH.a(0.24));
                }
            }
        }
        let biggest = values.iter().fold(0.0f64, |b, v| b.max(v.abs()));
        let (stem, head) = ((0.3 * pitch_px).clamp(1.0, 2.0), (0.8 * pitch_px).clamp(3.0, 6.5));
        for (j, &v) in values.iter().enumerate() {
            // (entries that are zero to working precision are zero)
            if v.abs() <= 1e-9 * biggest.max(1e-300) {
                continue;
            }
            let (x, y) = (lay.at(j), y_of(v));
            f.line((x, base), (x, y), stem, color);
            f.dot((x, y), head, color);
            if (v as f32 * lay.amp).abs() > reach {
                // beyond the row: a bar across says it goes on
                f.line((x - 0.6 * lay.pitch, y), (x + 0.6 * lay.pitch, y), 2.0, color);
            }
        }
    }

    fn draw_equation(&mut self, f: &mut Frame, lay: &Layout, ppu: f32) {
        let (n, m, k) = (self.n, self.m, self.k);
        let (size, line) = (lay.size, lay.line);
        let a_bottom = lay.a_top - m as f32 * lay.row_h;
        let a_right = lay.a_left + n as f32 * lay.pitch;
        if self.heat_stale {
            self.paint(f.gfx);
        }
        if let Some(image) = &mut self.heat {
            if image.size() == [(n + 2) as u32, m as u32] {
                f.image_part(image, [0, 0, 1, m as u32], Vec2::new(lay.y_left, a_bottom), Vec2::new(lay.y_left + lay.y_w, lay.a_top), 0.0);
                f.image_part(image, [2, 0, n as u32, m as u32], Vec2::new(lay.a_left, a_bottom), Vec2::new(a_right, lay.a_top), 0.0);
            }
        }
        frame_rect(f, Vec2::new(lay.a_left, a_bottom), Vec2::new(a_right, lay.a_top), 1.0, pal::DIM.a(0.7));
        frame_rect(f, Vec2::new(lay.y_left, a_bottom), Vec2::new(lay.y_left + lay.y_w, lay.a_top), 1.0, pal::DIM.a(0.7));
        let mid = 0.5 * (a_bottom + lay.a_top);
        f.text(((lay.y_left + lay.y_w + lay.a_left) * 0.5, mid), "$=$", size + 3.0, pal::TEXT, Align2::CENTER_CENTER);
        f.text((lay.y_left + 0.5 * lay.y_w, lay.a_top + 3.0 / ppu), "$y$", size + 1.0, pal::TEXT, Align2::CENTER_BOTTOM);
        // the columns that x picks: y is the sum of these, each times its entry of x
        let tick = (0.6 * lay.pitch * ppu).clamp(1.5, 4.0);
        for (j, &v) in self.x.iter().enumerate() {
            if v != 0.0 {
                let x = lay.at(j);
                f.line((x, lay.a_top + 2.0 / ppu), (x, lay.a_top + 8.0 / ppu), tick, TRUTH);
            }
        }
        // what it is, above
        let room = a_right - lay.y_left;
        let title = match self.kind {
            Kind::Random => {
                let long = format!("$y = Ax$:  {m} measurements of {n} unknowns, {k} of which are not zero");
                let short = format!("$y = Ax$:  {m} measurements, {n} unknowns");
                self.widths.pick(f, &[&long, &short], room, ppu, size + 1.0).to_string()
            }
            Kind::Entries => {
                let long = format!("$y = Ax$:  {m} measurements that read the first {m} of {n} entries");
                let short = format!("$y = Ax$:  reading {m} of {n} entries");
                self.widths.pick(f, &[&long, &short], room, ppu, size + 1.0).to_string()
            }
        };
        f.text((lay.y_left, lay.top), title, size + 1.0, pal::TEXT, Align2::LEFT_TOP);
        let sub = match self.kind {
            Kind::Random => self.widths.pick(
                f,
                &[
                    "each row of $A$: one measurement, a random weight for every entry (blue below zero, red above, brighter for larger).   White ticks: the columns the signal's non-zero entries pick",
                    "each row of $A$: one measurement, a random weight for every entry (blue below zero, red above, brighter for larger)",
                    r"rows of $A$: random weights ($\text{blue} < 0 < \text{red}$)",
                ],
                room,
                ppu,
                size - 1.5,
            ),
            Kind::Entries => self.widths.pick(
                f,
                &["each row of $A$: all zeros (dark) but a $1$ (bright) at the entry it reads.   White ticks: the columns the signal's non-zero entries pick", "each row of $A$: zeros, and a $1$ at the entry it reads", "rows of $A$: a $1$ at the entry read"],
                room,
                ppu,
                size - 1.5,
            ),
        };
        f.text((lay.y_left, lay.top - line - 1.0 / ppu), sub, size - 1.5, pal::DIM, Align2::LEFT_TOP);
    }

    fn draw_rows(&self, f: &mut Frame, lay: &Layout, ppu: f32) {
        let (n, k) = (self.n, self.k);
        let size = lay.size;
        let label_y = |r: usize| lay.rows[r] + lay.amp * 1.1 + 0.3 * lay.line;
        let room = n as f32 * lay.pitch;
        let right = lay.a_left + room;
        // the signal
        let long = format!("$x$, the signal: {k} of its {n} entries are not zero (nobody knows which)");
        let name = format!("$x$, the signal: {k} of its {n} entries are not zero");
        let short = format!("$x$: {k} of {n} not zero");
        f.text((lay.a_left, label_y(0)), self.widths.pick(f, &[&long, &name, &short], room, ppu, size), size, TRUTH, Align2::LEFT_BOTTOM);
        f.text((lay.y_left + 0.5 * lay.y_w, lay.rows[0]), "$x$", lay.size + 1.0, TRUTH, Align2::CENTER_CENTER);
        Self::stems(f, lay, lay.rows[0], &self.x, None, TRUTH, ppu);
        // the least 1-norm
        if self.show_l1 {
            // (how many entries the replay has chosen, counted once for both labels)
            let chosen = self.replay.map(|s| self.chosen(s));
            let (label, verdict) = match chosen {
                Some(chosen) => (format!("being found: {chosen} entries chosen, the shortest fix makes up the rest of $y$"), String::new()),
                None => match self.error_l1() {
                    Some(e) if e < EXACT => (r"least 1-norm $\sum |x_i|$ (basis pursuit)".to_string(), format!("exact: error {}", error_text(e))),
                    Some(e) => (r"least 1-norm $\sum |x_i|$ (basis pursuit)".to_string(), format!("wrong: error {}", error_text(e))),
                    None if self.solving => (r"least 1-norm $\sum |x_i|$ (basis pursuit): being found".to_string(), String::new()),
                    None => ("least 1-norm: could not be found (this matrix is degenerate)".to_string(), String::new()),
                },
            };
            let verdict_w = self.widths.of(f, &verdict, size) / ppu + 0.15;
            let label = if let Some(chosen) = chosen {
                let short = format!("being found: {chosen} chosen");
                self.widths.pick(f, &[&label, &short], room, ppu, size).to_string()
            } else {
                self.widths.pick(f, &[&label, "least 1-norm (basis pursuit)", "least 1-norm"], room - verdict_w, ppu, size).to_string()
            };
            f.text((lay.a_left, label_y(1)), label, size, L1, Align2::LEFT_BOTTOM);
            f.text((right, label_y(1)), verdict, size, L1, Align2::RIGHT_BOTTOM);
            if let Some(x) = self.green() {
                Self::stems(f, lay, lay.rows[1], &x, Some(&self.x), L1, ppu);
            }
        }
        // the shortest
        if self.show_l2 {
            // (exact too, now and then: with m = N there is only one answer, and reading
            // entries that hold every non-zero one leaves nothing to choose)
            let e = sparse::relative_error(&self.shortest, &self.x);
            let verdict = format!("{}: error {}", if e < EXACT { "exact" } else { "wrong" }, error_text(e));
            let verdict_w = self.widths.of(f, &verdict, size) / ppu + 0.15;
            let label = self.widths.pick(f, &[r"least 2-norm $\sqrt{\sum x_i^2}$, the shortest answer", "least 2-norm (shortest)", "shortest"], room - verdict_w, ppu, size);
            f.text((lay.a_left, label_y(2)), label, size, L2, Align2::LEFT_BOTTOM);
            f.text((right, label_y(2)), verdict, size, L2, Align2::RIGHT_BOTTOM);
            Self::stems(f, lay, lay.rows[2], &self.shortest, Some(&self.x), L2, ppu);
        }
        let below = lay.rows[2] - lay.amp * 1.1 - 0.25 * lay.line;
        let caption = self.widths.pick(
            f,
            &[
                "Under each answer, faint white bars: the signal, to compare. Every answer drawn agrees with all the measurements $y$.",
                "Faint bars: the signal. Each answer agrees with every measurement.",
                "Faint bars: the signal.",
            ],
            room + 0.2,
            ppu,
            size - 1.5,
        );
        f.text((lay.a_left, below), caption, size - 1.5, pal::DIM, Align2::LEFT_TOP);
    }

    /// The chance of exact recovery against m, with the current m marked (and a
    /// handle to move it).
    fn draw_chart(&mut self, f: &mut Frame, lay: &Layout, ppu: f32) {
        let (n, k) = (self.n, self.k);
        let (o, s) = (lay.chart, lay.chart_size);
        let (size, line) = (lay.size, lay.line);
        let at = |m: f32, p: f32| o + Vec2::new(s.x * m / n as f32, s.y * p);
        inset(f, o - Vec2::new(34.0 / ppu, 3.3 * line), o + s + Vec2::new(10.0 / ppu, 3.1 * line));
        for p in [0.0f32, 0.5, 1.0] {
            f.line(at(0.0, p), at(n as f32, p), 1.0, if p == 0.0 { pal::DIM.a(0.8) } else { pal::FAINT.a(0.6) });
            f.text(at(0.0, p) - Vec2::new(5.0 / ppu, 0.0), format!("{:.0} %", p * 100.0), size - 2.0, pal::DIM, Align2::RIGHT_CENTER);
        }
        for m in [0, n / 4, n / 2, 3 * n / 4, n] {
            f.text(at(m as f32, 0.0) - Vec2::new(0.0, 5.0 / ppu), m.to_string(), size - 2.0, pal::DIM, Align2::CENTER_TOP);
        }
        f.text(at(n as f32, 0.0) - Vec2::new(0.0, 1.0 * line), "measurements $m$ →", size - 1.5, pal::DIM, Align2::RIGHT_TOP);
        // below m = k nothing can work: even knowing where the non-zero entries are, k
        // unknowns cannot be pinned down by fewer than k equations
        f.dashed(at(k as f32, 0.0), at(k as f32, 1.0), 1.0, pal::DIM.a(0.8));
        // (the curves as drawn, which follow the trials: see `Drawn`)
        let ms = &self.curves.ms;
        for (shown, color) in [(&self.drawn.shown[1], ENTRIES), (&self.drawn.shown[0], L1)] {
            let pts: Vec<Vec2> = ms.iter().zip(shown).filter_map(|(&m, p)| p.map(|p| at(m as f32, p))).collect();
            f.path(pts.iter().copied(), 2.0, color);
            let dot = (s.x * ppu / ms.len().max(1) as f32 * 0.55).clamp(3.0, 6.0);
            for p in pts {
                f.dot(p, dot, color);
            }
        }
        // the current m: a line and a handle to drag along the axis
        f.line(at(self.m as f32, 0.0), at(self.m as f32, 1.0), 1.6, M_MARK.a(0.9));
        let mut knob = at(self.m as f32, 0.0);
        if f.drag2(1, &mut knob, M_MARK) {
            let new_m = (((knob.x - o.x) / s.x.max(1e-6)) * n as f32).round().max(1.0).min(n as f32) as usize;
            if new_m != self.m {
                self.m = new_m;
                self.rebuild();
            }
        }
        // what it reads there, on a plate just above the chart
        let here = match self.curves.chance(self.kind, self.m) {
            Some((p, _)) => format!("$m = {}$: exact {:.0} % of the time", self.m, p * 100.0),
            None => format!("$m = {}$", self.m),
        };
        if let Some(mark) = on_screen(f, at(self.m as f32, 1.0).extend(0.0)) {
            let fits = self.widths.of(f, &here, size - 1.0) * 0.5 + 6.0;
            let (lo, hi) = (on_screen(f, at(0.0, 1.0).extend(0.0)), on_screen(f, at(n as f32, 1.0).extend(0.0)));
            let x = match (lo, hi) {
                (Some(lo), Some(hi)) => mark.x.max(lo.x + fits - 30.0).min(hi.x - fits + 8.0),
                _ => mark.x,
            };
            f.caption_px(Vec2::new(x, mark.y - 0.35 * line * ppu), here, size - 1.0, M_MARK, Align2::CENTER_BOTTOM);
        }
        if s.x * ppu > 180.0 {
            f.text(at(k as f32, 1.0) + Vec2::new(4.0 / ppu, -4.0 / ppu), format!("$m = k = {k}$"), size - 2.5, pal::DIM, Align2::LEFT_TOP);
        }
        // titles above, and what the two curves are under the axis
        let title = self.widths.pick(f, &["How often is the least 1-norm exact?", "Chance of exact recovery"], s.x + 30.0 / ppu, ppu, size + 1.0);
        f.text(o + Vec2::new(-30.0 / ppu, s.y + 2.75 * line), title, size + 1.0, pal::TEXT, Align2::LEFT_BOTTOM);
        let done = self.curves.rounds;
        let trials = if done >= TRIALS {
            format!("each point: {TRIALS} trials of a fresh random signal ({k} of {n} entries not zero) and, for random sums, a fresh $A$")
        } else {
            format!("measuring: {done} of {TRIALS} trials at each $m$ so far, each a fresh random signal ({k} of {n} not zero)")
        };
        let middle = if done >= TRIALS { format!("each point: {TRIALS} trials, each a fresh random signal") } else { format!("measuring: {done} of {TRIALS} trials at each $m$ so far") };
        let short = if done >= TRIALS { format!("{TRIALS} fresh random trials at each $m$") } else { format!("measuring: {done} of {TRIALS} at each $m$") };
        f.text(o + Vec2::new(-30.0 / ppu, s.y + 1.75 * line), self.widths.pick(f, &[&trials, &middle, &short], s.x + 30.0 / ppu, ppu, size - 2.0), size - 2.0, pal::DIM, Align2::LEFT_BOTTOM);
        let key = [(L1, "random sums (the green answer)"), (ENTRIES, "reading single entries")];
        let key_y = o.y - 2.1 * line;
        let mut x = o.x - 30.0 / ppu;
        for (color, name) in key {
            f.dot((x + 4.0 / ppu, key_y), 7.0, color);
            f.text((x + 12.0 / ppu, key_y), name, size - 1.5, color, Align2::LEFT_CENTER);
            x += (text_width(name, size - 1.5) + 34.0) / ppu;
        }
    }

    /// On a wide screen, under the chart: each answer's two norms beside the signal's,
    /// so that each can be seen to win its own contest.
    fn draw_norms(&self, f: &mut Frame, lay: &Layout, ppu: f32) {
        if lay.norms_w <= 0.0 {
            return;
        }
        let (size, line) = (lay.size, lay.line);
        let (o, w) = (lay.norms, lay.norms_w);
        f.text(o, self.widths.pick(f, &["Each answer has the least of its own norm", "The two norms"], w, ppu, size + 1.0), size + 1.0, pal::TEXT, Align2::LEFT_TOP);
        let green = self.answer.as_ref().map(|a| a.x.as_slice());
        let answers: [(&str, Color, Option<&[f64]>); 3] = [("signal", TRUTH, Some(&self.x)), ("green", L1, green), ("pink", L2, Some(&self.shortest))];
        let name_w = 70.0 / ppu;
        let label_w = 64.0 / ppu;
        let bar_w = (w - name_w - label_w - 0.05).max(0.1);
        for (g, (title, norm)) in [(r"1-norm  $\sum |x_i|$", sparse::norm1 as fn(&[f64]) -> f64), (r"2-norm  $\sqrt{\sum x_i^2}$", sparse::norm2)].into_iter().enumerate() {
            let top = o.y - 1.6 * line - g as f32 * 4.5 * line;
            f.text((o.x, top), title, size - 1.0, pal::TEXT, Align2::LEFT_TOP);
            let values: Vec<Option<f64>> = answers.iter().map(|(_, _, x)| x.map(norm)).collect();
            let most = values.iter().flatten().fold(0.0f64, |b, v| b.max(*v)).max(1e-12);
            for (i, ((name, color, _), value)) in answers.iter().zip(&values).enumerate() {
                let y = top - (i as f32 + 1.6) * line;
                f.text((o.x + name_w - 6.0 / ppu, y), *name, size - 1.5, *color, Align2::RIGHT_CENTER);
                if let Some(v) = value {
                    let x0 = o.x + name_w;
                    let x1 = x0 + bar_w * (*v / most) as f32;
                    f.line((x0, y), (x1.max(x0 + 1.0 / ppu), y), (0.55 * line * ppu).clamp(3.0, 9.0), color.a(0.85));
                    f.text((x1 + 8.0 / ppu, y), format!("{v:.2}"), size - 1.5, *color, Align2::LEFT_CENTER);
                }
            }
        }
        let verdict = match (self.answer.as_ref(), self.error_l1()) {
            (Some(_), Some(e)) if e < EXACT => "Green's 1-norm is the signal's: here the least 1-norm is the signal.",
            (Some(_), Some(_)) => "Green's 1-norm is less than the signal's: the least 1-norm is not the signal, so it cannot be found this way.",
            _ => "",
        };
        wrapped(f, Vec2::new(o.x, o.y - 10.9 * line), w, verdict, size - 1.5, pal::DIM);
    }

    /// On a wide screen, under the chart: what coherence promises against what works.
    fn draw_note(&mut self, f: &mut Frame, lay: &Layout, ppu: f32) {
        if lay.note_w <= 0.0 {
            return;
        }
        let size = lay.size - 1.0;
        let text = self.coherence_words();
        let lines = self.note.lines(&text, size, lay.note_w * ppu, |word| text_width(word, size));
        // (left out where its last lines would run off the foot of the stage: the room
        // kept for it is less than it takes on a short screen, and its formulas,
        // which are not broken across lines, can make it a line longer than the
        // letters they replaced)
        if lay.note.y - lines.len() as f32 * (size + 3.0) / ppu < -0.5 * STAGE_H {
            return;
        }
        for (i, line) in lines.iter().enumerate() {
            f.text_off(lay.note, Vec2::new(0.0, i as f32 * (size + 3.0)), line.as_str(), size, pal::DIM, Align2::LEFT_TOP);
        }
    }

    /// The coherence of this A, what it guarantees, and what the trials say.
    fn coherence_words(&self) -> String {
        let (n, m, k) = (self.n, self.m, self.k);
        match self.mu() {
            None => format!("Coherence: none to speak of. {} of $A$'s columns are all zeros: those entries are never measured, so they cannot be recovered, whatever the method.", n.saturating_sub(m)),
            Some(mu) => {
                let sure = sparse::guaranteed(mu, n);
                let welch = sparse::welch_bound(m, n);
                let best = sparse::guaranteed(welch, n);
                let mut text = format!(
                    r"Coherence $\mu = {mu:.2}$: the two most alike columns of $A$, scaled to length 1, have a dot product of $\pm {mu:.2}$. The classical guarantee, every signal with $k < (1 + 1/\mu)/2$ is recovered, covers $k \le {sure}$ here. Even the best possible ${m} \times {n}$ matrix ($\mu = {welch:.2}$) would be guaranteed only $k \le {best}$."
                );
                if let Some((p, _)) = self.curves.chance(self.kind, m) {
                    if k > sure && p >= 0.5 {
                        text += &format!(" Yet with $k = {k}$ the trials succeed {:.0} % of the time: the guarantee is far more cautious than what really happens.", p * 100.0);
                    }
                }
                text
            }
        }
    }
}

impl Scene for Sensing {
    fn view(&self) -> View {
        STAGE_VIEW
    }

    fn settings(&self, s: &mut ViewSettings) {
        s.grid = false;
        s.bloom = s.bloom.min(0.2);
    }

    /// Moving: while the answer is being found or replayed, and while the trials behind
    /// the curves are still being run and the curves as drawn catch up with them. Then
    /// nothing changes until the settings do.
    fn animated(&self) -> bool {
        self.solving || self.replay.is_some() || !self.curves.complete() || !self.drawn.caught_up(&self.curves)
    }

    fn ui(&mut self, ui: &mut egui::Ui, _: &Gfx) {
        ui::note(ui, "A signal $x$ has $N$ entries, but only $k$ of them are not zero, and nobody knows which. Instead of reading the entries one by one, take $m$ measurements, far fewer than $N$, each a sum of all the entries with random weights: $y = Ax$. With fewer equations than unknowns, infinitely many $x$ agree with $y$. The shortest of them is a smear. The one with the least 1-norm, once $m$ is large enough, is exactly $x$.");

        ui::section(ui, "Things to try");
        for (label, look_for, setup) in TRY {
            if ui::experiment(ui, label, look_for) {
                self.apply(setup);
            }
        }

        ui::section(ui, "The signal and the measurements");
        let mut changed = ui::slider_int(ui, "unknowns $N$", &mut self.n, N_MIN..=N_MAX);
        let half = self.n / 2;
        changed |= ui::slider_int(ui, "entries not zero, $k$", &mut self.k, 1..=half.max(1));
        changed |= ui::slider_int(ui, "measurements $m$", &mut self.m, 1..=self.n);
        ui::note(ui, "Or drag the yellow handle on the chart. One more measurement adds a row to $A$ and leaves the others as they were.");
        let mut kind = (self.kind == Kind::Entries) as usize;
        if ui::choice(ui, &["random sums", "read single entries"], &mut kind) {
            self.kind = if kind == 1 { Kind::Entries } else { Kind::Random };
            changed = true;
        }
        ui::note(ui, "Random sums: each row of $A$ holds $N$ random numbers. Read single entries: row $i$ is all zeros but a $1$ at entry $i$, so $y$ is the first $m$ entries of $x$.");
        match ui::buttons(ui, &["new signal", "new matrix", "show how it is found"]) {
            Some(0) => {
                self.signal_seed += 1;
                changed = true;
            }
            Some(1) => {
                self.matrix_seed += 1;
                changed = true;
            }
            Some(2) => {
                if self.answer.is_some() {
                    self.replay = Some(0.0);
                } else {
                    self.replay_wanted = self.solving;
                }
                self.show_l1 = true;
            }
            _ => {}
        }
        if changed {
            self.rebuild();
        }
        ui::note(ui, r"How it is found: the 1-norm is given a price $\lambda$, which falls from the level at which no entry is worth having to zero. On the way, entries are chosen one at a time (now and then one is dropped again), each the one whose column of $A$ is most like what is still unexplained. Shown with the shortest correction for the rest of $y$ added, every moment agrees with every measurement: it starts as the shortest answer and ends as the least-1-norm one.");
        ui.horizontal_wrapped(|ui| {
            ui.checkbox(&mut self.show_l1, egui::RichText::new("least 1-norm").color(L1.egui()));
            ui.checkbox(&mut self.show_l2, egui::RichText::new("least 2-norm").color(L2.egui()));
        });

        ui::section(ui, "Reading the picture");
        ui::key(
            ui,
            &[
                (HEAT.table()[heat_index(-0.8)], "$A$ and $y$: blue cells are below zero, red ones above (brighter for larger), dark ones zero"),
                (TRUTH, "white: the signal $x$; the ticks above $A$ mark the columns its non-zero entries pick"),
                (L1, "green: the answer with the least 1-norm, and how often that answer is exact (the curve)"),
                (L2, "pink: the shortest answer (least 2-norm)"),
                (ENTRIES, "purple curve: how often the least 1-norm is exact when $A$ reads single entries"),
                (M_MARK, "yellow: the number of measurements $m$ now (drag it)"),
            ],
        );
        ui::note(ui, "Each answer is drawn over faint white bars of the signal: an exact answer covers them and nothing else.");
        let (n, k) = (self.n as f64, self.k as f64);
        let rule = 1.5 * k * (n / k).ln();
        let middle = if 4 * self.k <= self.n {
            format!(r"Here the middle of that climb is near $1.5\,k \ln(N/k) = {rule:.0}$: doubling $N$ moves it only a few measurements, doubling $k$ moves it about one and a half times as far.")
        } else {
            r"For a sparse signal, $k$ up to about $N/4$, the middle of that climb sits near $1.5\,k \ln(N/k)$; with this many non-zero entries it comes later.".to_string()
        };
        ui::note(
            ui,
            &format!(
                "The curves: at each $m$, {TRIALS} trials, each with a fresh random signal with $k$ non-zero entries (and, for random sums, a fresh random $A$); a point is the share in which the least 1-norm was exact. The green one climbs from almost never to almost always within a narrow range of $m$, a few times $k$ and far below $N$. {middle} Below the dashed line $m = k$ nothing could work: even knowing where the $k$ non-zero entries are, fewer than $k$ equations cannot pin them down."
            ),
        );

        ui::section(ui, "Words");
        ui::term(ui, "Sparse:", "most entries are zero.");
        ui::term(ui, "1-norm:", r"$\sum |x_i|$, the sum of the sizes of the entries.");
        ui::term(ui, "2-norm:", r"$\sqrt{\sum x_i^2}$, the length. The shortest answer is the one with the least.");
        ui::term(ui, "Basis pursuit:", "choosing, among all $x$ with $Ax = y$, the one with the least 1-norm.");
        ui::term(ui, "Exact recovery:", "the answer differs from the signal by less than $10^{-4}$ of the signal's length. (When it works the difference is rounding, $10^{-12}$ or less; when it fails, it is per cents.)");
        ui::term(ui, r"Coherence $\mu$:", r"the largest $|\text{dot product}|$ between two columns of $A$, each scaled to length 1: how easily two entries can be mistaken for each other.");

        ui::section(ui, "Numbers");
        let l1 = self.answer.as_ref().map(|a| sparse::norm1(&a.x));
        ui::readout(ui, "1-norm of the signal", format!("{:.3}", sparse::norm1(&self.x)));
        if let Some(l1) = l1 {
            ui::readout_colored(ui, "1-norm of the green answer", format!("{l1:.3}"), L1);
        }
        ui::readout_colored(ui, "1-norm of the pink answer", format!("{:.3}", sparse::norm1(&self.shortest)), L2);
        ui::readout(ui, "2-norm of the signal", format!("{:.3}", sparse::norm2(&self.x)));
        ui::readout_colored(ui, "2-norm of the pink answer", format!("{:.3}", sparse::norm2(&self.shortest)), L2);
        if let Some(a) = &self.answer {
            ui::readout_colored(ui, "entries not zero, green", format!("{} of {}", nonzero(&a.x), self.n), L1);
        }
        ui::readout_colored(ui, "entries not zero, pink", format!("{} of {}", nonzero(&self.shortest), self.n), L2);
        if let Some(e) = self.error_l1() {
            ui::readout_colored(ui, "error of the green answer", error_text(e), L1);
        }
        ui::readout_colored(ui, "error of the pink answer", error_text(sparse::relative_error(&self.shortest, &self.x)), L2);
        if let Some(c) = &self.check {
            ui::readout(ui, "proof that nothing has a smaller 1-norm", if c.proven(1e-8) { "holds" } else { "fails" });
        }
        ui::note(ui, r"Errors: the distance from the signal as a share of its length. The proof is a vector $\nu$ with $|(A^\top \nu)_i| \le 1$ for every $i$ and $(A^\top \nu)_i = \operatorname{sign}(x_i)$ wherever the green answer is not zero: then any $x$ with $Ax = y$ has");
        ui::equation(ui, r"\sum_i |x_i| \ge \sum_i x_i (A^\top \nu)_i = y^\top \nu,");
        ui::note(ui, "which is the green answer's 1-norm.");
        match self.mu() {
            Some(mu) => {
                ui::readout(ui, r"coherence $\mu$ of this $A$", format!("{mu:.3}"));
                ui::readout(ui, "guaranteed for every signal with", format!(r"$k \le {}$", sparse::guaranteed(mu, self.n)));
                ui::readout(ui, r"lowest $\mu$ any matrix this shape has", format!("{:.3}", sparse::welch_bound(self.m, self.n)));
            }
            None => ui::readout(ui, r"coherence $\mu$ of this $A$", "none: some columns are all zeros"),
        }
        if let Some((p, trials)) = self.curves.chance(self.kind, self.m) {
            ui::readout(ui, "chance of exact recovery at this $m$", format!("{:.0} % ({trials} trials)", p * 100.0));
        }
        ui::note(ui, &self.coherence_words());
        ui::see_also(
            ui,
            &[
                ("underdet", "too few equations: why the 1-norm picks a corner, in two and three unknowns"),
                ("packing", "the lowest coherence any matrix can have"),
                ("sae", "sparse autoencoders: the same 1-norm pulls features out of a network"),
                ("superposition", "superposition: more features than dimensions"),
            ],
        );
    }

    fn frame(&mut self, f: &mut Frame) {
        // the solve answers the settings, so it goes on while paused; the trials are
        // the scene's own work and wait while it is paused
        self.solve(&Budget::new(0.006));
        if f.dt > 0.0 && !self.curves.complete() {
            let mut turn = self.pace.begin(TRIAL_RATE, f.dt, f.real_dt, 0.003);
            while !self.curves.complete() && turn.next() {
                self.curves.trial();
            }
            self.pace.end(turn);
        }
        self.drawn.follow(&self.curves, f.dt);
        if let Some(t) = self.replay {
            let next = t + f.dt / REPLAY;
            self.replay = (next < 1.0).then_some(next);
        }

        let (w, ppu) = (stage_width(f), pixels_per_unit(f));
        let lay = Layout::of(w, ppu, f.top_inset, self.n, self.m);
        self.widths.next_frame();
        self.draw_equation(f, &lay, ppu);
        self.draw_rows(f, &lay, ppu);
        self.draw_chart(f, &lay, ppu);
        self.draw_norms(f, &lay, ppu);
        self.draw_note(f, &lay, ppu);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The scene set up as an experiment (with other seeds, if given), solved.
    fn run(k: usize, seeds: (u64, u64)) -> Sensing {
        let mut s = Sensing::new();
        s.apply(Setup { signal: seeds.0, matrix: seeds.1, ..TRY[k].2 });
        s.settle();
        s
    }

    /// The same with its recovery curves measured.
    fn measured(k: usize) -> Sensing {
        let mut s = run(k, (1, 1));
        while !s.curves.complete() {
            s.curves.trial();
        }
        s
    }

    /// The seeds the captions are also checked with, beyond the experiment's own.
    const SEEDS: [(u64, u64); 5] = [(1, 1), (2, 1), (1, 2), (3, 3), (4, 5)];

    fn green(s: &Sensing) -> &[f64] {
        &s.answer.as_ref().expect("solved").x
    }

    fn proven(s: &Sensing) -> bool {
        s.check.is_some_and(|c| c.proven(1e-8))
    }

    /// Where a curve first reaches one half.
    fn middle(s: &Sensing, kind: Kind) -> usize {
        (1..=s.n).find(|&m| s.curves.chance(kind, m).is_some_and(|(p, _)| p >= 0.5)).unwrap_or(s.n + 1)
    }

    /// The opening picture: exact recovery from 40 measurements of 100 unknowns, the
    /// shortest answer a smear, and the proof that nothing has a smaller 1-norm.
    #[test]
    fn the_opening_picture_is_an_exact_recovery() {
        let s = Sensing::new();
        assert_eq!((s.n, s.k, s.m, s.kind), (100, 8, 40, Kind::Random));
        let mut s = s;
        s.settle();
        assert!(s.error_l1().unwrap() < 1e-14 && proven(&s));
        assert_eq!(nonzero(green(&s)), 8);
        assert!(green(&s).iter().zip(&s.x).all(|(g, x)| (*g == 0.0) == (*x == 0.0)), "the same positions");
        assert_eq!(nonzero(&s.shortest), 100);
        assert!(sparse::relative_error(&s.shortest, &s.x) > 0.7);
        // every answer drawn agrees with the measurements
        for x in [green(&s), &s.shortest] {
            let mut ax = vec![0.0; s.m];
            sparse::apply(&s.a, s.m, s.n, x, &mut ax);
            assert!(sparse::relative_error(&ax, &s.y) < 1e-12);
        }
    }

    /// "Enough measurements": exact (the error is rounding, 10⁻¹⁵ or so), and past the
    /// step: signals like this come back nearly every time.
    #[test]
    fn enough_measurements() {
        for seeds in SEEDS {
            let s = run(0, seeds);
            let e = s.error_l1().unwrap();
            assert!(e < 1e-14 && proven(&s), "{seeds:?}: {e}");
            assert_eq!(nonzero(green(&s)), 8);
        }
        let s = measured(0);
        let (p, trials) = s.curves.chance(Kind::Random, 40).unwrap();
        assert!(p >= 0.96 && trials == TRIALS, "{p}");
        assert!(middle(&s, Kind::Random) < 40);
    }

    /// "Too few": the first 20 of the same measurements. The answer agrees with all
    /// 20, has a smaller 1-norm than the signal, and is a spray of 20 spikes; m = 20
    /// is before the step.
    #[test]
    fn too_few() {
        let first = run(0, (1, 1));
        for seeds in SEEDS {
            let s = run(1, seeds);
            assert_eq!(s.m, 20);
            let e = s.error_l1().unwrap();
            assert!(e > 0.3 && proven(&s), "{seeds:?}: {e}");
            assert!(sparse::norm1(green(&s)) < sparse::norm1(&s.x) * 0.99);
            assert_eq!(nonzero(green(&s)), 20, "{seeds:?}");
        }
        // the same signal, and the first 20 rows of the same matrix
        let s = run(1, (1, 1));
        assert_eq!(s.x, first.x);
        assert_eq!(s.raw[..], first.raw[..20 * 100]);
        let s = measured(1);
        let (p, _) = s.curves.chance(Kind::Random, 20).unwrap();
        assert!(p <= 0.1 && middle(&s, Kind::Random) > 20, "{p}");
    }

    /// "Not sparse enough": 20 non-zero entries are too many for 40 measurements; the
    /// answer is a spray of 40, and the step has moved to about m = 50.
    #[test]
    fn not_sparse_enough() {
        for seeds in SEEDS {
            let s = run(2, seeds);
            let e = s.error_l1().unwrap();
            assert!(e > 0.3 && proven(&s), "{seeds:?}: {e}");
            assert_eq!(nonzero(green(&s)), 40, "{seeds:?}");
        }
        let s = measured(2);
        let (p, _) = s.curves.chance(Kind::Random, 40).unwrap();
        let half = middle(&s, Kind::Random);
        assert!(p <= 0.05 && (46..=56).contains(&half), "{p} at 40, half way at {half}");
    }

    /// "The shortest answer is a smear": 90 measurements, and the pink answer is
    /// still wrong in all 100 entries, by about a quarter of the signal's length; the
    /// green one is exact.
    #[test]
    fn the_shortest_answer_is_a_smear() {
        let s = run(3, (1, 1));
        let e = sparse::relative_error(&s.shortest, &s.x);
        assert!((0.22..0.32).contains(&e), "{e}");
        for seeds in SEEDS {
            let s = run(3, seeds);
            assert_eq!(s.m, 90);
            assert_eq!(nonzero(&s.shortest), 100);
            assert!(s.shortest.iter().zip(&s.x).all(|(p, x)| (p - x).abs() > 1e-9), "{seeds:?}: wrong in every entry");
            assert!(sparse::relative_error(&s.shortest, &s.x) > 0.2);
            assert!(s.error_l1().unwrap() < 1e-14 && proven(&s));
        }
    }

    /// "Read single entries instead": the first 40 come back, nothing after them,
    /// so the non-zero entries beyond 40 are lost; on the purple curve this works only
    /// when all 8 happen to be among those read, which takes nearly all 100.
    #[test]
    fn reading_single_entries() {
        for seeds in SEEDS {
            let s = run(4, seeds);
            let g = green(&s);
            assert!((0..40).all(|j| g[j] == s.x[j]) && (40..100).all(|j| g[j] == 0.0));
            let lost = (40..100).filter(|&j| s.x[j] != 0.0).count();
            assert!(lost > 0 && s.error_l1().unwrap() > 0.3, "{seeds:?}");
            assert!(s.mu().is_none() && proven(&s));
        }
        let s = measured(4);
        assert_eq!(s.curves.chance(Kind::Entries, 40).unwrap().0, 0.0);
        let half = middle(&s, Kind::Entries);
        assert!(half >= 85, "{half}");
        // which is what counting says: all 8 among the m read, C(m, 8) / C(100, 8)
        let choose = |a: usize, b: usize| (0..b).map(|i| (a - i) as f64 / (b - i) as f64).product::<f64>();
        for (&m, c) in s.curves.ms.iter().zip(&s.curves.entries) {
            let p = if m >= 8 { choose(m, 8) / choose(100, 8) } else { 0.0 };
            let seen = c[0] as f64 / c[1] as f64;
            assert!((seen - p).abs() < 0.2, "at m = {m}: {seen} against {p}");
        }
        // and at the same m random sums do recover
        assert!(s.curves.chance(Kind::Random, 40).unwrap().0 >= 0.96);
    }

    /// The curve for reading entries is worked out without the solver; the solver
    /// says the same, signal after signal.
    #[test]
    fn reading_entries_needs_no_solver() {
        let mut rng = Rng::new(17);
        let (mut yes, mut no) = (0, 0);
        for trial in 0..300 {
            let (n, k) = ([32, 100, 128][trial % 3], 1 + trial % 6);
            let m = 1 + rng.below(n);
            let a = sparse::first_rows(m, n);
            let x = sparse::sparse_signal(n, k, &mut rng);
            let mut y = vec![0.0; m];
            sparse::apply(&a, m, n, &x, &mut y);
            let solved = sparse::relative_error(&sparse::basis_pursuit(&a, m, n, &y).unwrap().x, &x) < EXACT;
            assert_eq!(solved, reads_all(m, &x), "N {n} k {k} m {m}");
            if solved {
                yes += 1;
            } else {
                no += 1;
            }
        }
        assert!(yes > 30 && no > 30, "{yes} recovered, {no} not");
    }

    /// The step: from almost never to almost always over a narrow range of m, a few
    /// times k, with its middle near 1.5 k ln(N/k) for k up to N/4 (and later than that
    /// for more); doubling N moves it a few measurements, doubling k about one and a
    /// half times as far. Never below m = k. (Fewer trials than the scene runs, at fewer
    /// values of m, to keep this quick.)
    #[test]
    fn the_step_is_sharp_and_grows_like_k_log_n_over_k() {
        let chance = |n: usize, k: usize, m: usize, trials: usize| {
            let mut rng = Rng::new(99 + (n * 1000 + k * 10 + m) as u64);
            let wins = (0..trials).filter(|_| recovers(&sparse::gaussian_matrix(m, n, &mut rng), m, n, k, &mut rng) == Some(true)).count();
            wins as f64 / trials as f64
        };
        // (by halving the range: the chance climbs with m)
        let middle = |n: usize, k: usize| {
            let (mut lo, mut hi) = (k, n);
            while hi - lo > 1 {
                let mid = (lo + hi) / 2;
                if chance(n, k, mid, 60) >= 0.5 {
                    hi = mid;
                } else {
                    lo = mid;
                }
            }
            hi
        };
        let (low, high) = (chance(100, 8, 18, 60), chance(100, 8, 36, 60));
        assert!(low <= 0.1 && high >= 0.9, "{low} at m = 18, {high} at m = 36");
        let mut found = Vec::new();
        for (n, k) in [(64usize, 4usize), (128, 4), (128, 8), (128, 16), (100, 8), (100, 20), (32, 1), (32, 8), (100, 25), (100, 40), (64, 32)] {
            let half = middle(n, k);
            let rule = 1.5 * k as f64 * (n as f64 / k as f64).ln();
            if 4 * k <= n {
                // (within a fifth, or within a measurement or so where k is so small that
                // m is counted in ones)
                assert!((half as f64 - rule).abs() < (0.2 * rule).max(1.5), "N {n} k {k}: half way at {half}, 1.5 k ln(N/k) = {rule:.1}");
            } else {
                assert!(half as f64 > 1.2 * rule, "N {n} k {k}: half way at {half}, 1.5 k ln(N/k) = {rule:.1}");
            }
            assert!(chance(n, k, k - 1, 20) == 0.0 || k == 1, "N {n} k {k}: below m = k");
            found.push(half as f64);
        }
        assert!(found[1] - found[0] <= 7.0, "doubling N: {found:?}");
        for pair in [(1, 2), (2, 3)] {
            let ratio = found[pair.1] / found[pair.0];
            assert!((1.3..1.9).contains(&ratio), "doubling k: {found:?}");
        }
    }

    /// "Exact" and "wrong" are far apart, so the threshold of 10⁻⁴ is not a judgement
    /// call: at the middle of the step, where both happen, every error is either
    /// rounding (below 10⁻¹²) or per cents.
    #[test]
    fn exact_and_wrong_are_far_apart() {
        let mut rng = Rng::new(5);
        let (n, k, m) = (100, 8, 28);
        let (mut exact, mut wrong) = (0, 0);
        for _ in 0..80 {
            let a = sparse::gaussian_matrix(m, n, &mut rng);
            let x = sparse::sparse_signal(n, k, &mut rng);
            let mut y = vec![0.0; m];
            sparse::apply(&a, m, n, &x, &mut y);
            let e = sparse::relative_error(&sparse::basis_pursuit(&a, m, n, &y).unwrap().x, &x);
            assert!(!(1e-12..0.01).contains(&e), "{e}");
            if e < 1e-12 {
                exact += 1;
            } else {
                wrong += 1;
            }
        }
        assert!(exact > 15 && wrong > 15, "{exact} exact, {wrong} wrong");
    }

    /// The coherence of the opening matrix guarantees only k ≤ 1, and even the best
    /// possible matrix of its shape only k ≤ 4, yet k = 8 works nearly every time.
    #[test]
    fn the_guarantee_is_cautious() {
        let s = measured(0);
        let mu = s.mu().unwrap();
        assert!((0.55..0.7).contains(&mu), "{mu}");
        assert_eq!(sparse::guaranteed(mu, 100), 1);
        assert_eq!(sparse::guaranteed(sparse::welch_bound(40, 100), 100), 4);
        let words = s.coherence_words();
        assert!(words.contains(r"covers $k \le 1$") && words.contains(r"only $k \le 4$") && words.contains("Yet with $k = 8$"), "{words}");
        assert_eq!(crate::typeset::inspect(&words), Default::default(), "{words}");
        // reading entries: columns of zeros, no coherence to speak of
        let s = run(4, (1, 1));
        assert!(s.coherence_words().starts_with("Coherence: none to speak of. 60 of $A$'s columns"));
    }

    /// Every text the scene works out is markup that parses, with no word set in
    /// italic: an error of every size, and the words on coherence for every experiment
    /// and at the ends of the sliders.
    #[test]
    fn every_text_is_markup_that_parses() {
        let fine = |text: &str| crate::typeset::inspect(text) == Default::default();
        for e in [0.0, 1e-300, 3.2e-16, 9.96e-13, 9.6e-5, 1e-4, 0.005, 0.25, 1.0, 12.0] {
            let text = error_text(e);
            assert!(fine(&text), "{text}");
            assert!(fine(&format!("exact: error {text}")));
        }
        assert_eq!(error_text(6.2e-16), r"$6 \times 10^{-16}$");
        assert_eq!(error_text(9.6e-13), r"$1 \times 10^{-12}$");
        for (label, look_for, _) in TRY {
            assert!(fine(label) && fine(look_for), "{look_for}");
        }
        for k in 0..TRY.len() {
            let words = run(k, (1, 1)).coherence_words();
            assert!(fine(&words), "{words}");
        }
        for (n, k, m, kind) in [(32, 16, 1, Kind::Random), (128, 64, 128, Kind::Random), (128, 1, 1, Kind::Entries)] {
            let mut s = Sensing::new();
            s.apply(Setup { n, k, m, kind, signal: 3, matrix: 4 });
            let words = s.coherence_words();
            assert!(fine(&words), "{words}");
        }
    }

    /// "How it is found": every moment of the replay agrees with every measurement;
    /// it starts at the shortest answer and ends at the green one; entries are chosen
    /// along the way.
    #[test]
    fn the_replay_goes_from_the_shortest_to_the_least_one_norm() {
        for k in [0, 1, 2] {
            let s = run(k, (1, 1));
            let mut ax = vec![0.0; s.m];
            for i in 0..=40 {
                let t = i as f32 / 40.0;
                let p = s.along(t).unwrap();
                sparse::apply(&s.a, s.m, s.n, &p, &mut ax);
                assert!(sparse::relative_error(&ax, &s.y) < 1e-10, "experiment {k} at {t}");
            }
            assert!(sparse::relative_error(&s.along(0.0).unwrap(), &s.shortest) < 1e-12);
            assert!(sparse::relative_error(&s.along(1.0).unwrap(), green(&s)) < 1e-12);
            assert_eq!(s.chosen(0.0), 0);
            assert_eq!(s.chosen(1.0), nonzero(green(&s)));
        }
    }

    /// The solver finishes on every matrix the scene makes, at the extremes of every
    /// slider too (a few rounds of trials at each).
    #[test]
    fn the_solver_always_finishes() {
        for (n, k) in [(32usize, 1usize), (32, 16), (128, 64), (100, 50), (128, 1)] {
            let mut c = Curves::new(n, k);
            for _ in 0..3 * c.ms.len() {
                c.trial();
            }
            assert_eq!(c.unsolved, 0, "N {n} k {k}");
        }
    }

    /// The rows kept from one m to the next, and the coherence worked out when first
    /// asked for, never leave an old problem on show: after every kind of change (m up
    /// and down, as the handle on the chart moves it; a new matrix; a new signal;
    /// another N; reading entries and back; an experiment) the problem is exactly the
    /// one a scene opened afresh on the same settings makes, bit for bit.
    #[test]
    fn a_changed_problem_is_the_one_made_afresh() {
        let mut s = Sensing::new();
        let changes: [fn(&mut Sensing); 10] = [
            |s| s.m = 60,
            |s| s.m = 25,
            |s| s.m = 61,
            |s| s.matrix_seed += 1,
            |s| s.m = 30,
            |s| s.signal_seed += 1,
            |s| s.n = 64,
            |s| s.kind = Kind::Entries,
            |s| s.kind = Kind::Random,
            |s| s.apply(TRY[3].2),
        ];
        for (i, change) in changes.into_iter().enumerate() {
            // (asked for first, so that a coherence kept from before would be seen)
            let _ = s.mu();
            change(&mut s);
            s.rebuild();
            let mut fresh = Sensing::new();
            fresh.apply(Setup { n: s.n, k: s.k, m: s.m, kind: s.kind, signal: s.signal_seed, matrix: s.matrix_seed });
            assert!(s.a == fresh.a && s.raw == fresh.raw && s.x == fresh.x && s.y == fresh.y && s.shortest == fresh.shortest, "change {i}");
            assert_eq!(s.mu(), fresh.mu(), "change {i}");
            assert_eq!(s.mu(), sparse::coherence(&s.a, s.m, s.n), "change {i}");
        }
    }

    /// The words on coherence kept broken into lines are never old ones: after every
    /// kind of change (other words, as the trials come in or the settings change; a
    /// text size; a width) they are exactly the lines broken afresh. They are broken
    /// between words, never inside a formula, and no line is wider than the room but
    /// for a word wider by itself.
    #[test]
    fn the_lines_kept_are_the_ones_broken_afresh() {
        let (s, opened) = (measured(0), run(4, (1, 1)));
        let texts = [s.coherence_words(), opened.coherence_words(), run(2, (1, 1)).coherence_words()];
        let mut kept = Broken::default();
        for (text, size, room) in [(&texts[0], 11.5, 330.0), (&texts[1], 11.5, 330.0), (&texts[1], 11.5, 200.0), (&texts[2], 11.5, 200.0), (&texts[2], 10.5, 200.0), (&texts[0], 11.5, 40.0)] {
            let width = |word: &str| text_width(word, size);
            let lines = kept.lines(text, size, room, width).to_vec();
            assert_eq!(lines, Broken::default().lines(text, size, room, width), "{text}");
            assert_eq!(lines.join(" "), *text);
            for line in &lines {
                assert_eq!(crate::typeset::inspect(line), Default::default(), "{line}");
                assert!(text_width(line, size) <= room + 1e-3 || words(line).len() == 1, "{line}");
            }
        }
        assert_eq!(words(r"a $k \le 1$, b \$ c $x$"), ["a", r"$k \le 1$,", "b", r"\$", "c", "$x$"]);
    }

    /// Settings at their ends, and every window: nothing panics, and every number on
    /// show is finite.
    #[test]
    fn the_ends_of_the_sliders_are_fine() {
        for (n, k, m, kind) in [(32, 16, 1, Kind::Random), (32, 1, 32, Kind::Random), (128, 64, 128, Kind::Random), (128, 64, 1, Kind::Entries), (32, 16, 32, Kind::Entries), (500, 900, 0, Kind::Random)] {
            let mut s = Sensing::new();
            s.apply(Setup { n, k, m, kind, signal: 3, matrix: 4 });
            s.settle();
            assert!(s.n >= N_MIN && s.k <= s.n / 2 && s.m >= 1 && s.m <= s.n);
            assert!(s.shortest.iter().chain(green(&s)).all(|v| v.is_finite()));
            assert!(proven(&s), "N {n} k {k} m {m} {kind:?}");
            if s.m == s.n {
                // one answer only: the shortest is the signal too (and so labelled exact)
                assert!(sparse::relative_error(&s.shortest, &s.x) < 1e-10);
            }
            assert!(s.mu().is_none_or(|mu| mu.is_finite()));
            for t in [0.0, 0.5, 1.0] {
                assert!(s.along(t).unwrap().iter().all(|v| v.is_finite()));
            }
            for (w, ppu, inset) in [(8.9, 200.0, 0.0), (2.44, 160.0, 46.0), (6.67, 36.0, 46.0), (2.0, 1.0, 46.0)] {
                let lay = Layout::of(w, ppu, inset, s.n, s.m);
                let numbers = [lay.size, lay.line, lay.top, lay.y_left, lay.y_w, lay.a_left, lay.pitch, lay.a_top, lay.row_h, lay.amp, lay.note_w, lay.norms_w];
                assert!(numbers.iter().chain(&lay.rows).all(|v| v.is_finite()) && lay.chart.is_finite() && lay.chart_size.is_finite());
            }
        }
    }

    /// The numbers behind the captions (run with --ignored --nocapture).
    #[test]
    #[ignore]
    fn explore() {
        for (k, (label, _, _)) in TRY.iter().enumerate() {
            for seeds in SEEDS {
                let s = run(k, seeds);
                let g = green(&s);
                println!(
                    "{label:<32} seeds {seeds:?}: green error {:.2e}, {} not zero, 1-norm {:.3} against {:.3}; pink error {:.3}, {} not zero; μ {:?}, {} steps",
                    s.error_l1().unwrap(),
                    nonzero(g),
                    sparse::norm1(g),
                    sparse::norm1(&s.x),
                    sparse::relative_error(&s.shortest, &s.x),
                    nonzero(&s.shortest),
                    s.mu().map(|m| (m * 1000.0).round() / 1000.0),
                    s.solver.steps()
                );
            }
        }
        for (n, k) in [(100usize, 8usize), (100, 20), (128, 4), (64, 4), (128, 8), (128, 16), (128, 64), (32, 4)] {
            let mut c = Curves::new(n, k);
            let start = std::time::Instant::now();
            while !c.complete() {
                c.trial();
            }
            let took = start.elapsed().as_secs_f64();
            let row = |counts: &Vec<[u32; 2]>| c.ms.iter().zip(counts).map(|(m, c)| format!("{m}:{}", c[0])).collect::<Vec<_>>().join(" ");
            println!("N {n} k {k} ({took:.2} s):\n  random  {}\n  entries {}", row(&c.random), row(&c.entries));
        }
    }
}

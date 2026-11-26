//! Counts and the chi-square test.
//!
//! Counts in categories against the counts a hypothesis expects, cell by cell: each
//! cell gives (O − E)²/E, drawn as a square of that area, and the squares' areas laid
//! end to end make the statistic, marked on the chi-square curve that it follows
//! (nearly) when the hypothesis is true. Two uses, one at a time: does a die roll fair
//! (one set of counts against given shares), and do two ways of sorting the same
//! people have anything to do with each other (a table, against the counts
//! independence expects). Many more samples made the same way build the statistic's
//! real distribution under the curve: that is how the scene shows where the curve is
//! right, and where it is not.

use super::plot::{button, fill, line_of, path_of, screen_handle, Glide, Plot, Rect, AXES, MARKS, OWN};
use super::*;
use crate::math::stats::{normal_cdf, unit, Dist};

pub const DEF: SceneDef = SceneDef {
    id: "chisquare",
    title: "Counts and the chi-square test",
    group: TESTING,
    blurb: "Counts observed against counts expected: does a die roll fair, and do two ways of sorting people have anything to do with each other?",
    make: |_| Box::new(ChiSquare::create()),
};

// Lines by kind (see `plot::apart`): the curve is cut where it leaves its chart, and so
// is the histogram's outline, so each has a kind of its own; the shapes of a mode (the
// squares, the bars' tops, the mosaic) keep their number of points while it is shown.
const CURVE: u8 = OWN;
const HIST: u8 = OWN + 1;
const SHAPES: u8 = OWN + 2;

// ----- the arithmetic ------------------------------------------------------------------

/// The share of the chi-square curve with `df` degrees of freedom beyond `x`: the
/// p-value of a statistic `x`, by the curve. (1 where there is nothing to test.)
fn tail(x: f64, df: usize) -> f64 {
    if df == 0 || x.is_nan() {
        return 1.0;
    }
    (1.0 - Dist::ChiSquare { df: df as f64 }.cdf(x.max(0.0))).max(0.0).min(1.0)
}

/// The value with a share `alpha` of the curve beyond it.
fn critical(df: usize, alpha: f64) -> f64 {
    if df == 0 {
        return f64::INFINITY;
    }
    if alpha == 0.05 {
        return curve_facts(df)[1];
    }
    Dist::ChiSquare { df: df as f64 }.quantile(1.0 - alpha)
}

/// A chi-square test of observed counts against expected ones.
#[derive(Clone, Debug, Default, PartialEq)]
struct Test {
    observed: Vec<u32>,
    expected: Vec<f64>,
    /// Each cell's part of the statistic, (O − E)²/E: 0 where nothing is expected (such
    /// a cell holds nobody either, and drops out).
    parts: Vec<f64>,
    stat: f64,
    /// The degrees of freedom: 0 when there is nothing to test.
    df: usize,
}

impl Test {
    fn of(observed: Vec<u32>, expected: Vec<f64>, df: usize) -> Test {
        let parts: Vec<f64> = observed.iter().zip(&expected).map(|(&o, &e)| if e > 0.0 { (o as f64 - e).powi(2) / e } else { 0.0 }).collect();
        let stat = parts.iter().sum();
        Test { observed, expected, parts, stat, df }
    }

    /// The p-value, by the curve; none without a test.
    fn p(&self) -> Option<f64> {
        (self.df > 0).then(|| tail(self.stat, self.df))
    }

    /// The smallest expected count of a cell that is in the test.
    fn smallest_expected(&self) -> Option<f64> {
        self.expected.iter().copied().filter(|e| *e > 0.0).reduce(f64::min)
    }

    /// Did cell `i` get more than expected?
    fn more(&self, i: usize) -> bool {
        match (self.observed.get(i), self.expected.get(i)) {
            (Some(&o), Some(&e)) => o as f64 > e,
            _ => false,
        }
    }
}

// ----- the die -------------------------------------------------------------------------

const FACES: usize = 6;
/// A fair die's share of each face: the hypothesis.
const FAIR: f64 = 1.0 / 6.0;
/// The most rolls in a set.
const MOST_ROLLS: usize = 1200;

/// The true die: a six comes up a share `six` of the time, the other faces share the
/// rest equally.
fn die_shares(six: f64) -> [f64; FACES] {
    let six = if six.is_finite() { six.max(0.0).min(1.0) } else { FAIR };
    let rest = (1.0 - six) / 5.0;
    [rest, rest, rest, rest, rest, six]
}

/// `n` rolls of a die with these shares: how often each face came up.
fn roll(rng: &mut Rng, shares: &[f64; FACES], n: usize) -> [u32; FACES] {
    let mut edges = [0.0; FACES];
    let mut sum = 0.0;
    for (edge, share) in edges.iter_mut().zip(shares) {
        sum += share;
        *edge = sum;
    }
    let mut counts = [0u32; FACES];
    for _ in 0..n {
        let u = unit(rng);
        let face = edges[..FACES - 1].iter().position(|&e| u < e).unwrap_or(FACES - 1);
        counts[face] += 1;
    }
    counts
}

/// The rolls the scene shows: `n` rolls of the true die from `seed`. The same seed
/// gives the same draws, so that a longer set adds rolls to a shorter one, and loading
/// the die moves some rolls to a six without rolling the others again.
fn die_sample(seed: u64, n: usize, six: f64) -> [u32; FACES] {
    roll(&mut Rng::new(seed), &die_shares(six), n)
}

/// The counts of a die against a fair one: the goodness of fit.
fn die_test(counts: &[u32; FACES]) -> Test {
    let n: u32 = counts.iter().sum();
    Test::of(counts.to_vec(), vec![n as f64 * FAIR; FACES], if n > 0 { FACES - 1 } else { 0 })
}

/// The same statistic without the bookkeeping (for the many repeats).
fn die_stat(counts: &[u32; FACES]) -> f64 {
    let e = counts.iter().sum::<u32>() as f64 * FAIR;
    if e <= 0.0 {
        return 0.0;
    }
    counts.iter().map(|&o| (o as f64 - e).powi(2) / e).sum()
}

// ----- the table -----------------------------------------------------------------------

const COLS: usize = 4;
const DRINKS: [&str; COLS] = ["tea", "coffee", "juice", "water"];
const PLACES: [&str; 2] = ["north", "south"];
/// The most people a table holds.
const MOST_PEOPLE: usize = 20_000;

/// People sorted two ways: where they live (the rows) and the drink they like best (the
/// first `cols` columns; the counts beyond are 0).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Table {
    cols: usize,
    n: [[u32; COLS]; 2],
}

impl Table {
    const fn of(cols: usize, north: [u32; COLS], south: [u32; COLS]) -> Table {
        Table { cols, n: [north, south] }
    }

    fn row(&self, i: usize) -> u32 {
        self.n[i][..self.cols].iter().sum()
    }

    fn col(&self, j: usize) -> u32 {
        self.n[0][j] + self.n[1][j]
    }

    fn total(&self) -> u32 {
        self.row(0) + self.row(1)
    }

    /// The counts, the north's row and then the south's.
    fn cells(&self) -> Vec<u32> {
        self.n.iter().flat_map(|row| row[..self.cols].iter().copied()).collect()
    }

    /// How many rows and columns hold anybody.
    fn filled(&self) -> (usize, usize) {
        ((0..2).filter(|&i| self.row(i) > 0).count(), (0..self.cols).filter(|&j| self.col(j) > 0).count())
    }

    /// The test of independence: a cell expects its row's total times its column's,
    /// over everybody; an empty row or column drops out, with its degrees of freedom.
    fn test(&self) -> Test {
        let total = self.total() as f64;
        let mut expected = Vec::with_capacity(2 * self.cols);
        for i in 0..2 {
            for j in 0..self.cols {
                expected.push(if total > 0.0 { self.row(i) as f64 * self.col(j) as f64 / total } else { 0.0 });
            }
        }
        let (rows, cols) = self.filled();
        Test::of(self.cells(), expected, rows.saturating_sub(1) * cols.saturating_sub(1))
    }

    /// Cramér's V: √(χ² / (N (k − 1))), k the smaller of the numbers of rows and columns
    /// that hold anybody. From 0 (no association) to 1, whatever the number of people.
    fn cramers_v(&self, test: &Test) -> Option<f64> {
        if test.df == 0 {
            return None;
        }
        let (rows, cols) = self.filled();
        let k = (rows.min(cols) - 1) as f64;
        Some((test.stat / (self.total() as f64 * k)).sqrt().min(1.0))
    }

    /// For two rows and two columns: the share of each row in the first column, and the
    /// z statistic for the difference between the two shares (its spread from the
    /// shares pooled). None where a row or a column is empty.
    fn two_shares(&self) -> Option<TwoShares> {
        if self.cols != 2 {
            return None;
        }
        let (r1, r2, c1, n) = (self.row(0) as f64, self.row(1) as f64, self.col(0) as f64, self.total() as f64);
        if r1 == 0.0 || r2 == 0.0 || c1 == 0.0 || c1 == n {
            return None;
        }
        let (first, second) = (self.n[0][0] as f64 / r1, self.n[1][0] as f64 / r2);
        let pooled = c1 / n;
        let spread = (pooled * (1.0 - pooled) * (1.0 / r1 + 1.0 / r2)).sqrt();
        Some(TwoShares { first, second, z: (first - second) / spread })
    }
}

/// Two shares compared by a z statistic.
#[derive(Clone, Copy, Debug, PartialEq)]
struct TwoShares {
    first: f64,
    second: f64,
    z: f64,
}

impl TwoShares {
    /// The two-sided p-value of z, by the normal curve.
    fn p(&self) -> f64 {
        2.0 * normal_cdf(-self.z.abs())
    }
}

/// The table with `people` people in the same shares as `from`, as near as whole people
/// allow (the largest remainders get the odd people; ties to the first cell). With
/// nobody to go by, the people are shared out evenly. A whole multiple of the people
/// gives exactly that multiple of every count.
fn rescale(from: &Table, people: u32) -> Table {
    let cells = from.cells();
    let k = cells.len().max(1);
    let weights: Vec<u64> = if from.total() == 0 { vec![1; k] } else { cells.iter().map(|&c| c as u64).collect() };
    let sum: u64 = weights.iter().sum::<u64>().max(1);
    let people = people as u64;
    let mut counts: Vec<u64> = weights.iter().map(|w| w * people / sum).collect();
    let mut rest: Vec<(u64, usize)> = weights.iter().enumerate().map(|(i, w)| (w * people % sum, i)).collect();
    rest.sort_by(|a, b| b.0.cmp(&a.0).then(a.1.cmp(&b.1)));
    let short = people.saturating_sub(counts.iter().sum::<u64>()) as usize;
    for &(_, i) in rest.iter().take(short) {
        counts[i] += 1;
    }
    let mut out = Table { cols: from.cols, n: [[0; COLS]; 2] };
    for (i, c) in counts.iter().enumerate() {
        let (row, col) = (i / from.cols.max(1), i % from.cols.max(1));
        if row < 2 && col < COLS {
            out.n[row][col] = *c as u32;
        }
    }
    out
}

// ----- many more samples ---------------------------------------------------------------

/// What new samples are made from: rolls of the true die (each set tested against a
/// fair die), or the table's people with their columns dealt out again at random (the
/// hypothesis made true, the totals kept, so the expected counts are the table's own).
#[derive(Clone, Debug, PartialEq)]
enum Source {
    Die { shares: [f64; FACES], n: usize },
    Table { cols: usize, rows: [u32; 2], columns: [u32; COLS], expected: Vec<f64> },
}

/// Histogram bins: this many to the 0.999 point of the curve, three times that in all.
const BIN_SCALE: usize = 32;
const BINS: usize = 3 * BIN_SCALE;
/// The most new samples kept count of.
const MOST: usize = 10_000;
/// Every count of new samples starts from this seed, so that it is the same each time.
const REPEAT_SEED: u64 = 2024;

/// The χ² of many new samples, made one at a time.
struct Repeats {
    source: Option<Source>,
    df: usize,
    /// The value with 5 % of the curve beyond it: a new sample beyond it would be called
    /// significant at the 5 % level.
    critical: f64,
    /// The histogram's bins (each `width` wide, from 0), and the samples' count, sum,
    /// sum of squares, and how many were beyond the critical value.
    width: f64,
    bins: Vec<u32>,
    done: usize,
    sum: f64,
    squares: f64,
    beyond: usize,
    /// The bin the newest sample went into (if it went into one): the histogram before
    /// it is the histogram now without it.
    newest: Option<usize>,
    rng: Rng,
    /// The table's people, each by the column they are in (dealt out afresh each time).
    pool: Vec<u8>,
}

impl Repeats {
    fn new() -> Repeats {
        Repeats {
            source: None,
            df: 0,
            critical: f64::INFINITY,
            width: 1.0,
            bins: vec![0; BINS],
            done: 0,
            sum: 0.0,
            squares: 0.0,
            beyond: 0,
            newest: None,
            rng: Rng::new(REPEAT_SEED),
            pool: Vec::new(),
        }
    }

    /// Start again, from `source`, with `df` degrees of freedom.
    fn restart(&mut self, source: Source, df: usize) {
        self.pool.clear();
        if let Source::Table { cols, columns, .. } = &source {
            for (j, &c) in columns.iter().enumerate().take(*cols) {
                self.pool.extend(std::iter::repeat_n(j as u8, c as usize));
            }
        }
        self.source = Some(source);
        self.df = df;
        self.critical = critical(df, 0.05);
        self.width = bin_width(df);
        self.bins.iter_mut().for_each(|b| *b = 0);
        (self.done, self.sum, self.squares, self.beyond, self.newest) = (0, 0.0, 0.0, 0, None);
        self.rng = Rng::new(REPEAT_SEED);
    }

    /// Is there a new sample to make?
    fn can_step(&self) -> bool {
        self.df > 0 && self.done < MOST && self.source.is_some()
    }

    fn step(&mut self) {
        let Repeats { source, rng, pool, .. } = self;
        let stat = match source {
            Some(Source::Die { shares, n }) => die_stat(&roll(rng, shares, *n)),
            Some(Source::Table { cols, rows, columns, expected }) => shuffled(rng, pool, *cols, *rows, columns, expected),
            None => return,
        };
        self.done += 1;
        self.sum += stat;
        self.squares += stat * stat;
        if stat > self.critical {
            self.beyond += 1;
        }
        let bin = (stat / self.width).floor();
        self.newest = (bin >= 0.0 && bin < BINS as f64).then_some(bin as usize);
        if let Some(b) = self.newest {
            self.bins[b] += 1;
        }
    }

    fn mean(&self) -> Option<f64> {
        (self.done > 0).then(|| self.sum / self.done as f64)
    }

    fn share_beyond(&self) -> Option<f64> {
        (self.done > 0).then(|| self.beyond as f64 / self.done as f64)
    }

    /// About as far as the samples reach: their mean and three spreads beyond it.
    fn reach(&self) -> f64 {
        if self.done < 30 {
            return 0.0;
        }
        let mean = self.sum / self.done as f64;
        let var = (self.squares / self.done as f64 - mean * mean).max(0.0);
        mean + 3.0 * var.sqrt()
    }

    /// Each bin's height as a density (its share of the samples over its width), so
    /// that the bars stand on the same scale as the curve: a share `share` of the way
    /// from the histogram before the newest sample to the histogram with it. (Samples
    /// come whole, one in a frame and two in the next; drawn as they come, the bars
    /// would move in fits and starts. See `Phase`.)
    fn heights(&self, share: f64) -> Vec<f32> {
        let density = |count: f64, done: usize| if done > 0 { count / (done as f64 * self.width) } else { 0.0 };
        let share = share.max(0.0).min(1.0);
        self.bins
            .iter()
            .enumerate()
            .map(|(i, &b)| {
                let now = density(b as f64, self.done);
                let before = density(b as f64 - (self.newest == Some(i)) as u8 as f64, self.done.saturating_sub(1));
                let before = if self.done > 1 { before } else { now };
                (before + (now - before) * share) as f32
            })
            .collect()
    }
}

/// The χ² of the table's people with their columns dealt out again at random: the
/// smaller row's people are a random draw from everybody (a partial shuffle of
/// `pool`), and the other row has the rest of each column.
fn shuffled(rng: &mut Rng, pool: &mut [u8], cols: usize, rows: [u32; 2], columns: &[u32; COLS], expected: &[f64]) -> f64 {
    let total = pool.len();
    let first_smaller = rows[0] <= rows[1];
    let deal = (if first_smaller { rows[0] } else { rows[1] } as usize).min(total);
    for k in 0..deal {
        let j = k + rng.below(total - k);
        pool.swap(k, j);
    }
    let mut dealt = [0u32; COLS];
    for &c in &pool[..deal] {
        dealt[(c as usize).min(COLS - 1)] += 1;
    }
    let mut stat = 0.0;
    for j in 0..cols.min(COLS) {
        let north = if first_smaller { dealt[j] } else { columns[j] - dealt[j].min(columns[j]) };
        let south = columns[j] - north.min(columns[j]);
        for (o, e) in [(north, expected.get(j)), (south, expected.get(cols + j))] {
            if let Some(&e) = e {
                if e > 0.0 {
                    stat += (o as f64 - e).powi(2) / e;
                }
            }
        }
    }
    stat
}

/// What the charts need to know of the curve with `df` degrees of freedom: its 0.999
/// point (where a chart of it can end), its 5 % point, and how high its density gets
/// as the histogram's bins would show it (its average over each bin) and at its peak.
/// Worked out once for each `df` (a few hundred evaluations of the incomplete gamma
/// function: most of what a frame cost before they were kept).
fn curve_facts(df: usize) -> [f64; 3] {
    thread_local! {
        static KEPT: std::cell::RefCell<Vec<(usize, [f64; 3])>> = const { std::cell::RefCell::new(Vec::new()) };
    }
    if let Some(facts) = KEPT.with(|k| k.borrow().iter().find(|(d, _)| *d == df).map(|(_, f)| *f)) {
        return facts;
    }
    let facts = if df == 0 {
        [10.0, f64::INFINITY, 1.0]
    } else {
        let d = Dist::ChiSquare { df: df as f64 };
        let reach = d.quantile(0.999);
        let w = reach / BIN_SCALE as f64;
        let mut top = (0..BINS).map(|i| (d.cdf((i + 1) as f64 * w) - d.cdf(i as f64 * w)) / w).fold(0.0, f64::max);
        if df >= 2 {
            top = top.max(d.pdf((df - 2) as f64));
        }
        [reach, d.quantile(0.95), top]
    };
    KEPT.with(|k| k.borrow_mut().push((df, facts)));
    facts
}

fn reach_of(df: usize) -> f64 {
    curve_facts(df)[0]
}

fn bin_width(df: usize) -> f64 {
    reach_of(df) / BIN_SCALE as f64
}

fn curve_top(df: usize) -> f64 {
    curve_facts(df)[2]
}

/// The least round number at or above `v` (above 0): 1, 1.2, 1.5, 2, 2.5, 3, 4, 5, 6 or
/// 8 times a power of ten.
fn nice_ceil(v: f64) -> f64 {
    let v = if v.is_finite() { v.max(1e-9) } else { 1.0 };
    let p = 10f64.powf(v.log10().floor());
    for m in [1.0, 1.2, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0] {
        if m * p >= v * (1.0 - 1e-12) {
            return m * p;
        }
    }
    10.0 * p
}

/// A p-value as the words write it: two figures, or "less than" for a tiny one.
fn p_text(p: f64) -> String {
    if !p.is_finite() {
        "?".to_string()
    } else if p < 0.0001 {
        r"< 0.0001".to_string()
    } else if p < 0.001 {
        format!("= {}", num(p, 4))
    } else if p < 0.1 {
        format!("= {}", num(p, 3))
    } else {
        format!("= {}", num(p, 2))
    }
}

/// A statistic or a count as the picture writes it.
fn stat_text(v: f64) -> String {
    if v.abs() >= 100.0 {
        num(v, 1)
    } else {
        num(v, 2)
    }
}

// ----- the experiments -----------------------------------------------------------------

/// Experiments: (button, what to look for). Each is set up by [`ChiSquare::set_up`].
const TRIALS: [(&str, &str); 6] = [
    (
        "a fair die",
        r"Sixty rolls of a fair die. The counts wobble round the 10 expected of each face, and here $\chi^2 = 4.40$ ($p = 0.49$). The green histogram fills with the $\chi^2$ of new sets of 60 rolls and settles on the purple curve: their average comes to 5, the degrees of freedom (the $\chi^2$ of a fair die averages exactly 5, however many rolls), and about 5 % of them reach $p < 0.05$, every one a false alarm.",
    ),
    (
        "a loaded die",
        r"Sixty rolls of a die that comes up six one time in four, not one in six. Here the six came up 17 times against the 10 a fair die expects, yet $p = 0.29$; and of new sets of 60 rolls only about 22 % reach $p < 0.05$: most sets this small do not show the loading. Set 600 rolls and about 99 % do.",
    ),
    (
        "totals kept",
        r"North and south like the three drinks in the same shares: each observed tile fills its expected outline, every part is 0, and so is $\chi^2$. The totals are kept: drag the split in the tea or the coffee column, and the juice column's split moves the other way. Two counts are free, the rest follow: $(2 - 1)(3 - 1) = 2$ degrees of freedom.",
    ),
    (
        "ten times the people",
        r"Now the north likes tea more than the south does. With these 100 people $\chi^2 = 4.76$ and $p = 0.092$: not much evidence. Watch them grow to 1 000 in the same shares: the mosaic stays, and so does Cramér's $V = 0.22$, while $\chi^2$ grows to 47.62, ten times as large, and $p$ falls below 0.0001. $\chi^2$ measures the evidence, not the strength.",
    ),
    (
        "two by two",
        r"Tea or coffee: 60 % of the north drink tea, against 36 % of the south. The $z$ statistic for the difference between two shares is $z = 2.40$; its square is $\chi^2 = 5.77$, and both give $p = 0.016$. With two rows and two columns, $\chi^2$ is always $z^2$, with the same $p$-value.",
    ),
    (
        "small counts",
        r"Eight people, every expected count 1. By the curve $\chi^2 = 8.00$ gives $p = 0.046$. But shuffle the people (the hypothesis made true): the green histogram is three spikes, nothing like the curve, and 8.6 % of the shuffles (6 in 70, exactly) reach $p < 0.05$, not 5 %. The usual advice, expected counts of at least 5, is about this.",
    ),
];

/// The tables of the experiments.
const SAME_SHARES: Table = Table::of(3, [15, 21, 14, 0], [15, 21, 14, 0]);
const NORTH_LIKES_TEA: Table = Table::of(3, [20, 18, 12, 0], [10, 24, 16, 0]);
const TWO_BY_TWO: Table = Table::of(2, [30, 20, 0, 0], [18, 32, 0, 0]);
const SMALL: Table = Table::of(4, [2, 2, 0, 0], [0, 0, 2, 2]);
/// The seeds of the experiments' rolls: sets about as typical as can be found.
const FAIR_SEED: u64 = 12;
const LOADED_SEED: u64 = 24;
/// How long the people take to grow ten times (seconds of the scene's time).
const GROW_SECONDS: f32 = 3.0;

// ----- the scene -----------------------------------------------------------------------

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Mode {
    Die,
    Table,
}

/// What the visitor's hand holds.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Held {
    /// The six's level: how often the die comes up six.
    Six,
    /// The split between north and south in a column.
    Split(usize),
    /// The edge after a column.
    Edge(usize),
}

/// The people growing to a new number in the same shares.
#[derive(Clone, Copy, Debug)]
struct Grow {
    from: Table,
    to: u32,
    seconds: f32,
    /// How many people there are now, as a number that grows smoothly (the counts are
    /// whole; the picture follows this).
    now: f32,
}

/// What the picture glides to.
struct Goals {
    /// Die: the six counts; table: the eight cells (north's four, then south's).
    counts: Vec<f32>,
    /// Die: the expected count, then each face's level for the true die.
    levels: Vec<f32>,
    /// See `Y_COUNTS` and the rest.
    axes: Vec<f32>,
    hist: Vec<f32>,
}

const Y_COUNTS: usize = 0;
const X_CHI: usize = 1;
const Y_CHI: usize = 2;
const SCALE: usize = 3;

/// The picture as drawn: each part glides to its goal.
#[derive(Default)]
struct Look {
    counts: Glide,
    levels: Glide,
    axes: Glide,
    hist: Glide,
    moving: bool,
}

impl Look {
    fn follow(&mut self, g: &Goals, dt: f32, snap: bool) {
        if snap {
            self.counts.snap(&g.counts);
            self.levels.snap(&g.levels);
            self.axes.snap(&g.axes);
            self.hist.snap(&g.hist);
            self.moving = false;
            return;
        }
        let a = self.counts.follow(&g.counts, dt, true);
        let b = self.levels.follow(&g.levels, dt, true);
        let c = self.axes.follow(&g.axes, dt, true);
        let d = self.hist.follow(&g.hist, dt, true);
        self.moving = a || b || c || d;
    }
}

/// A text broken into lines: the text, the width and size it was broken for (their
/// bits), and the lines.
type Wrapped = (String, u32, u32, std::rc::Rc<Vec<String>>);

/// Rolls of the die: by seed, number and loading (its bits), how often each face came up.
type Rolled = ((u64, usize, u64), [u32; FACES]);

pub struct ChiSquare {
    mode: Mode,
    // the die
    rolls: usize,
    six: f64,
    seed: u64,
    // the table
    table: Table,
    keep_totals: bool,
    /// The people control's value, and the table it rescales (kept while it is moved, so
    /// that rescaling again and again does not drift; any other change lets it go).
    people: usize,
    scaled_from: Option<Table>,
    grow: Option<Grow>,
    held: Option<Held>,
    /// The table as it was when an edge was taken hold of, and the count chart's height
    /// while the six's level is held (so that the chart does not move under the hand).
    held_table: Option<Table>,
    held_height: Option<f32>,
    // many more samples
    repeating: bool,
    rate: usize,
    repeats: Repeats,
    pace: Pace,
    /// How far the scene's time has gone towards the next new sample (see `Phase`).
    phase: Phase,
    share: f32,
    // the picture
    look: Look,
    /// The canvas and the mode last drawn (a new one is drawn at once, without gliding).
    drawn_as: Option<(Vec2, Mode, usize)>,
    /// What decides which labels and ticks there are, and how many of each the last
    /// frame drew (see [`ChiSquare::keep_apart`]).
    signature: u64,
    drawn_last: (usize, usize),
    /// Lines of words already broken to a width.
    wraps: std::cell::RefCell<Vec<Wrapped>>,
    /// The rolls shown, by their seed, number and loading.
    rolled: std::cell::Cell<Option<Rolled>>,
    /// The new samples' numbers as last said (how many, their average and the share
    /// beyond the 5 % point), and how long ago (see [`ChiSquare::tell`]).
    told: (usize, Option<(f64, f64)>),
    tell_clock: f32,
}

impl ChiSquare {
    /// (Nothing here needs the graphics device, so the tests can make one too.)
    fn create() -> ChiSquare {
        let mut s = ChiSquare {
            mode: Mode::Die,
            rolls: 60,
            six: FAIR,
            seed: FAIR_SEED,
            table: NORTH_LIKES_TEA,
            keep_totals: false,
            people: NORTH_LIKES_TEA.total() as usize,
            scaled_from: None,
            grow: None,
            held: None,
            held_table: None,
            held_height: None,
            repeating: true,
            rate: 300,
            repeats: Repeats::new(),
            pace: Pace::default(),
            phase: Phase::default(),
            share: 1.0,
            look: Look::default(),
            drawn_as: None,
            signature: 0,
            drawn_last: (0, 0),
            wraps: Default::default(),
            rolled: Default::default(),
            told: (0, None),
            tell_clock: 0.0,
        };
        s.set_up(0);
        s
    }

    /// Set up one of the experiments in `TRIALS`.
    fn set_up(&mut self, k: usize) {
        (self.held, self.held_table, self.held_height, self.grow, self.scaled_from) = (None, None, None, None, None);
        self.rate = 300;
        match k {
            0 | 1 => {
                self.mode = Mode::Die;
                self.rolls = 60;
                (self.six, self.seed) = if k == 0 { (FAIR, FAIR_SEED) } else { (0.25, LOADED_SEED) };
                self.repeating = true;
            }
            _ => {
                self.mode = Mode::Table;
                let (table, keep, repeat) = match k {
                    2 => (SAME_SHARES, true, false),
                    3 => (NORTH_LIKES_TEA, false, false),
                    4 => (TWO_BY_TWO, false, false),
                    _ => (SMALL, false, true),
                };
                self.table = table;
                self.keep_totals = keep;
                self.repeating = repeat;
                if k == 3 {
                    self.grow = Some(Grow { from: table, to: 10 * table.total(), seconds: 0.0, now: table.total() as f32 });
                }
            }
        }
        self.people = self.table.total() as usize;
        // (a new experiment counts its new samples afresh, whatever came before)
        self.repeats.source = None;
        self.refresh();
    }

    /// The rolls shown (rolled once for each seed, number and loading, and kept).
    fn die_counts(&self) -> [u32; FACES] {
        let key = (self.seed, self.rolls, self.six.to_bits());
        if let Some((k, counts)) = self.rolled.get() {
            if k == key {
                return counts;
            }
        }
        let counts = die_sample(self.seed, self.rolls, self.six);
        self.rolled.set(Some((key, counts)));
        counts
    }

    /// The test of what is shown.
    fn test(&self) -> Test {
        match self.mode {
            Mode::Die => die_test(&self.die_counts()),
            Mode::Table => self.table.test(),
        }
    }

    /// What the new samples are made from now.
    fn source(&self) -> (Source, usize) {
        match self.mode {
            Mode::Die => (Source::Die { shares: die_shares(self.six), n: self.rolls }, if self.rolls > 0 { FACES - 1 } else { 0 }),
            Mode::Table => {
                let t = &self.table;
                let test = t.test();
                let mut columns = [0; COLS];
                for (j, c) in columns.iter_mut().enumerate().take(t.cols) {
                    *c = t.col(j);
                }
                (Source::Table { cols: t.cols, rows: [t.row(0), t.row(1)], columns, expected: test.expected }, test.df)
            }
        }
    }

    /// Start the new samples afresh if what they are made from has changed.
    fn refresh(&mut self) {
        self.rolls = self.rolls.min(MOST_ROLLS);
        self.people = self.people.max(2).min(MOST_PEOPLE);
        let (source, df) = self.source();
        if self.repeats.source.as_ref() != Some(&source) {
            self.repeats.restart(source, df);
            self.pace.reset();
            self.phase.reset();
            self.share = 1.0;
        }
    }

    /// The new samples asked for, paced: so many a second of the scene's time, within
    /// the frame's allowance (none while something is held or the people grow).
    fn work(&mut self, dt: f32, real_dt: f32) {
        if !self.repeating || self.held.is_some() || self.grow.is_some() || dt <= 0.0 {
            // (the histogram as it is, not part of the way to a sample that is not coming)
            if !self.repeating || self.held.is_some() || self.grow.is_some() {
                self.share = 1.0;
            }
            return;
        }
        if !self.repeats.can_step() {
            self.share = 1.0;
            // (all that were asked for are in, or there is nothing to make them from)
            if self.repeats.done >= MOST {
                self.repeating = false;
            }
            return;
        }
        let mut turn = scene_turn(&mut self.pace, self.rate as f32, dt, real_dt);
        while self.repeats.can_step() && turn.next() {
            self.repeats.step();
        }
        let taken = self.pace.end(turn);
        self.share = self.phase.after(self.rate as f32, dt, taken);
    }

    /// The numbers of the new samples, as the words and the panel say them: a few times
    /// a second while they come (numbers that change in every frame cannot be read, and
    /// each change sets its line of words again), at once when they start afresh or stop.
    fn tell(&mut self, real_dt: f32) {
        let r = &self.repeats;
        let now = (r.done, r.mean().zip(r.share_beyond()));
        self.tell_clock += real_dt.max(0.0);
        let running = self.repeating && r.can_step();
        if self.tell_clock >= 0.25 || !running || r.done < self.told.0 {
            self.told = now;
            self.tell_clock = 0.0;
        }
    }

    /// The people growing, by the scene's time.
    fn advance_growth(&mut self, dt: f32) {
        let Some(mut g) = self.grow else { return };
        g.seconds += dt.max(0.0);
        let t = (g.seconds / GROW_SECONDS).min(1.0);
        // (eased in and out: the statistic starts and ends its slide gently)
        let s = t * t * (3.0 - 2.0 * t);
        let from = g.from.total() as f32;
        g.now = from + (g.to as f32 - from) * s;
        let now = if t >= 1.0 { g.to } else { g.now.round() as u32 };
        self.table = rescale(&g.from, now);
        self.people = now as usize;
        self.grow = (t < 1.0).then_some(g);
    }

    /// Switch between the die and the table.
    fn switch(&mut self, mode: Mode) {
        if self.mode != mode {
            self.mode = mode;
            (self.held, self.grow) = (None, None);
            self.refresh();
        }
    }

    /// Start or stop the new samples (starting again when all asked for are in).
    fn toggle_repeats(&mut self) {
        if self.repeating {
            self.repeating = false;
        } else {
            if self.repeats.done >= MOST {
                self.repeats.source = None;
                self.refresh();
            }
            self.repeating = true;
        }
    }

    /// The north's count in column `j` set to `north` (its column keeps its total). With
    /// the totals kept, the last column takes up the difference, within what it holds.
    fn set_split(&mut self, j: usize, north: u32) {
        let t = &mut self.table;
        if j >= t.cols {
            return;
        }
        let cj = t.col(j);
        let north = north.min(cj);
        if self.keep_totals {
            let last = t.cols - 1;
            if j == last {
                return;
            }
            let r1 = t.row(0) as i64;
            let others: i64 = (0..last).filter(|&k| k != j).map(|k| t.n[0][k] as i64).sum();
            let c_last = t.col(last) as i64;
            // the last column's north count, r1 − others − north, must be within its column
            let north = (north as i64).max(r1 - others - c_last).min(r1 - others).max(0).min(cj as i64);
            let rest = r1 - others - north;
            if rest < 0 || rest > c_last {
                return;
            }
            t.n[0][j] = north as u32;
            t.n[1][j] = cj - north as u32;
            t.n[0][last] = rest as u32;
            t.n[1][last] = (c_last - rest) as u32;
        } else {
            t.n[0][j] = north;
            t.n[1][j] = cj - north;
        }
        self.scaled_from = None;
        self.people = self.table.total() as usize;
    }

    /// The edge after column `j` moved to a share `share` of everybody: people move
    /// between columns `j` and `j + 1` only, and each keeps its rows' shares as they were
    /// when the edge was taken hold of (`from`).
    fn set_edge(&mut self, j: usize, share: f64, from: &Table) {
        if j + 1 >= from.cols {
            return;
        }
        let n = from.total();
        let before: u32 = (0..j).map(|k| from.col(k)).sum();
        let pair = from.col(j) + from.col(j + 1);
        let cut = ((share.max(0.0).min(1.0) * n as f64).round() as i64 - before as i64).max(0).min(pair as i64) as u32;
        let north_of = |col: usize, total: u32| -> u32 {
            let c = from.col(col);
            let share = if c > 0 {
                from.n[0][col] as f64 / c as f64
            } else if n > 0 {
                from.row(0) as f64 / n as f64
            } else {
                0.5
            };
            ((share * total as f64).round() as u32).min(total)
        };
        for (col, total) in [(j, cut), (j + 1, pair - cut)] {
            let north = north_of(col, total);
            self.table.n[0][col] = north;
            self.table.n[1][col] = total - north;
        }
        self.scaled_from = None;
    }

    /// Two to four columns: the columns kept keep their people; a new one gets an even
    /// share of them, split as the whole table is.
    fn set_columns(&mut self, cols: usize) {
        let cols = cols.max(2).min(COLS);
        let old = self.table;
        if cols == old.cols {
            return;
        }
        let mut t = old;
        for j in cols..COLS {
            t.n[0][j] = 0;
            t.n[1][j] = 0;
        }
        let each = (old.total() as f64 / old.cols as f64).round() as u32;
        let north_share = if old.total() > 0 { old.row(0) as f64 / old.total() as f64 } else { 0.5 };
        for j in old.cols..cols {
            let north = (north_share * each as f64).round() as u32;
            t.n[0][j] = north.min(each);
            t.n[1][j] = each - north.min(each);
        }
        t.cols = cols;
        self.table = t;
        self.scaled_from = None;
        self.grow = None;
        self.people = self.table.total().max(2) as usize;
    }

    /// The people control moved: the same shares, this many people.
    fn set_people(&mut self, people: usize) {
        let from = *self.scaled_from.get_or_insert(self.table);
        self.grow = None;
        self.table = rescale(&from, people.max(2).min(MOST_PEOPLE) as u32);
    }

    /// What the picture glides to.
    fn goals(&self) -> Goals {
        let test = self.test();
        let (counts, levels, count_top) = match self.mode {
            Mode::Die => {
                let c = self.die_counts();
                let e = self.rolls as f64 * FAIR;
                let truth = die_shares(self.six).map(|s| s * self.rolls as f64);
                let most = c.iter().map(|&v| v as f64).chain(truth).fold(e, f64::max);
                let top = self.held_height.map_or(nice_ceil((most * 1.12).max(4.0)) as f32, |h| h);
                (c.iter().map(|&v| v as f32).collect(), std::iter::once(e as f32).chain(truth.iter().map(|&v| v as f32)).collect(), top)
            }
            Mode::Table => {
                // (while the people grow, the picture follows them smoothly: the same
                // shares of a number that is not yet whole)
                let (from, times) = match self.grow {
                    Some(g) if g.from.total() > 0 => (g.from, g.now / g.from.total() as f32),
                    _ => (self.table, 1.0),
                };
                (from.n.iter().flat_map(|row| row.iter().map(move |&v| v as f32 * times)).collect(), Vec::new(), 1.0)
            }
        };
        let df = test.df;
        let reach = reach_of(df);
        // (while the people grow, the chart and the squares are scaled for where they are
        // going, so that they make one glide there rather than many)
        let aim = match self.grow {
            Some(g) => rescale(&g.from, g.to).test(),
            None => test,
        };
        let far = aim.stat * 1.12;
        let x_top = nice_ceil(reach.max(far.min(3.0 * reach)).max(self.repeats.reach().min(3.0 * reach)));
        let heights = if self.repeats.df == df { self.repeats.heights(self.share as f64) } else { vec![0.0; BINS] };
        let mut y_top = curve_top(df);
        if self.repeats.done >= 300 && self.repeats.df == df {
            // (the bins on the chart, that is: where they reach well above the curve, as
            // they do when the statistic takes only a few values, the chart makes room, in
            // round steps so that it does not follow every new sample; a bar only a little
            // above the curve, by chance, is cut at the top)
            let shown = (x_top / self.repeats.width).ceil() as usize;
            let highest = heights.iter().take(shown).fold(0.0f64, |m, &h| m.max(h as f64));
            if highest > 1.6 * y_top {
                y_top = nice_ceil(highest);
            }
        }
        let scale = aim.parts.iter().copied().fold(4.0, f64::max);
        Goals { counts, levels, axes: vec![count_top, x_top as f32, (1.12 * y_top) as f32, scale as f32], hist: heights }
    }
}

// ----- the layout ----------------------------------------------------------------------

/// Where everything goes on a canvas.
#[derive(Clone, Copy, Debug, PartialEq)]
struct Layout {
    view: Vec2,
    narrow: bool,
    /// Text sizes: headings, words, small print.
    big: f32,
    text: f32,
    small: f32,
    side: f32,
    /// The row of buttons over the picture: its top and height.
    top: f32,
    button_h: f32,
    /// The counts (die: their bars; table: the mosaic), and what is under them (the
    /// squares; the table of numbers); the rectangles the data fill, with their names
    /// and numbers round them.
    first: Rect,
    under: Rect,
    /// The chi-square chart, and the words under it.
    chart: Rect,
    words: Rect,
    /// Too small for the charts: the numbers alone.
    compact: bool,
}

impl Layout {
    fn of(view: Vec2, inset: f32, mode: Mode) -> Layout {
        let view = Vec2::new(view.x.max(1.0), view.y.max(1.0));
        let narrow = view.x < 640.0 || view.x < 0.62 * view.y;
        let (big, text, small) = if narrow { (13.0, 11.5, 10.5) } else { (15.0, 12.5, 11.5) };
        let side = if narrow { 12.0 } else { 24.0 };
        let top = inset.max(0.0) + if narrow { 6.0 } else { 12.0 };
        let button_h = if narrow { 26.0 } else { 28.0 };
        let content = top + button_h + if narrow { 10.0 } else { 18.0 };
        // room left of the first picture: its count numbers, or the rows' names
        let margin = match (mode, narrow) {
            (Mode::Die, true) => 30.0,
            (Mode::Die, false) => 40.0,
            (Mode::Table, true) => 56.0,
            (Mode::Table, false) => 74.0,
        };
        // a picture's title and the line of names over it
        let head = big + small + 20.0;
        // under the first picture: the faces' or the columns' names, and a title
        let names = match mode {
            Mode::Die => text + 12.0,
            Mode::Table => 2.0 * (small + 3.0) + 12.0,
        };
        let under_h = match (mode, narrow) {
            (Mode::Die, true) => 40.0,
            (Mode::Die, false) => 66.0,
            (Mode::Table, true) => 86.0,
            (Mode::Table, false) => 120.0,
        };
        let under_gap = names + small + 12.0;
        // under the squares, their numbers
        let under_foot = if mode == Mode::Die { small + 8.0 } else { 6.0 };
        // under the chart: its tick numbers, and the parts laid end to end
        let chart_foot = small + 30.0;
        let chart_margin = if narrow { 14.0 } else { 20.0 };
        if narrow {
            let words_h = 6.0 * (small + 4.0) + 6.0;
            let fixed = head + under_gap + under_h + under_foot + head + chart_foot + words_h;
            let flex = view.y - 6.0 - content - fixed;
            let h1 = (flex * 0.5).max(1.0);
            let h2 = (flex - h1).max(1.0);
            let x1 = view.x - side;
            let first = (Vec2::new(side + margin, content + head), Vec2::new(x1, content + head + h1));
            let uy = first.1.y + under_gap;
            let under = (Vec2::new(side + margin, uy), Vec2::new(x1, uy + under_h));
            let cy = under.1.y + under_foot + head;
            let chart = (Vec2::new(side + chart_margin, cy), Vec2::new(x1, cy + h2));
            let words = (Vec2::new(side, chart.1.y + chart_foot), Vec2::new(x1, view.y - 4.0));
            let compact = flex < 130.0 || x1 - side - margin < 150.0;
            Layout { view, narrow, big, text, small, side, top, button_h, first, under, chart, words, compact }
        } else {
            let gap = 56.0;
            let colw = (view.x - 2.0 * side - gap) * 0.5;
            let (l, r) = (side, side + colw + gap);
            let bottom = view.y - 14.0;
            // the left: the counts, and under them the squares or the table of numbers
            let h1 = bottom - content - head - under_gap - under_h - under_foot;
            let first = (Vec2::new(l + margin, content + head), Vec2::new(l + colw, content + head + h1.max(1.0)));
            let uy = first.1.y + under_gap;
            let under = (Vec2::new(l + margin, uy), Vec2::new(l + colw, uy + under_h));
            // the right: the chart, as tall as the counts, and the words under it
            let words_h = 7.0 * (text + 6.0);
            let h2 = (bottom - content - head - chart_foot - words_h).min(h1).max(1.0);
            let chart = (Vec2::new(r + chart_margin, content + head), Vec2::new(r + colw, content + head + h2));
            let words = (Vec2::new(r, chart.1.y + chart_foot), Vec2::new(r + colw, bottom));
            let compact = h1 < 120.0 || colw < 220.0;
            Layout { view, narrow, big, text, small, side, top, button_h, first, under, chart, words, compact }
        }
    }
}

/// The width of a rectangle, and its height.
fn width((a, b): Rect) -> f32 {
    b.x - a.x
}

fn height((a, b): Rect) -> f32 {
    b.y - a.y
}

/// The first of `options` that fits in `room` logical pixels at `size` (the last if none does).
fn pick<'a>(f: &Frame, options: &[&'a str], room: f32, size: f32) -> &'a str {
    options.iter().copied().find(|t| f.text_size(t, size).x <= room).or(options.last().copied()).unwrap_or("")
}

/// The words of a text, each with its formulas whole.
fn words_of(text: &str) -> Vec<&str> {
    let (mut words, mut start, mut inside) = (Vec::new(), 0, false);
    let bytes = text.as_bytes();
    let mut i = 0;
    while i < bytes.len() {
        match bytes[i] {
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

impl ChiSquare {
    /// `text` broken into lines no wider than `room` pixels at `size`: worked out once
    /// for each text, width and size, and kept.
    fn lines(&self, f: &Frame, text: &str, room: f32, size: f32) -> std::rc::Rc<Vec<String>> {
        let key = (room.to_bits(), size.to_bits());
        let mut kept = self.wraps.borrow_mut();
        if let Some((_, _, _, lines)) = kept.iter().find(|(t, r, s, _)| t == text && (*r, *s) == key) {
            return lines.clone();
        }
        let space = f.text_size("a b", size).x - f.text_size("ab", size).x;
        let mut lines: Vec<String> = Vec::new();
        let mut used = 0.0;
        for word in words_of(text) {
            let w = f.text_size(word, size).x;
            match lines.last_mut() {
                Some(line) if used + space + w <= room => {
                    line.push(' ');
                    line.push_str(word);
                    used += space + w;
                }
                _ => {
                    lines.push(word.to_string());
                    used = w;
                }
            }
        }
        let lines = std::rc::Rc::new(lines);
        if kept.len() >= 96 {
            kept.clear();
        }
        kept.push((text.to_string(), key.0, key.1, lines.clone()));
        lines
    }

    /// Words from `at` down, no wider than `room`: the height they took.
    #[allow(clippy::too_many_arguments)]
    fn wrap(&self, f: &mut Frame, at: Vec2, room: f32, text: &str, size: f32, color: Color, lowest: f32) -> f32 {
        let lines = self.lines(f, text, room, size);
        let step = size + 4.0;
        let mut y = at.y;
        for line in lines.iter() {
            if y + step > lowest + 1.0 {
                break;
            }
            f.text_px(Vec2::new(at.x, y), line.as_str(), size, color, Align2::LEFT_TOP);
            y += step;
        }
        y - at.y
    }
}

/// A key's entry on one line: a short line (dashed, or a block) in a colour, and words.
#[derive(Clone, Copy)]
enum Swatch {
    Line,
    Dashed,
    Block,
}

/// Entries of a key side by side, ending at `right` on the line whose middle is at `y`.
fn key_line(f: &mut Frame, right: f32, y: f32, entries: &[(Swatch, Color, &str)], size: f32) {
    let mut x = right;
    for (swatch, color, text) in entries.iter().rev() {
        let w = f.text_size(text, size).x;
        f.text_px(Vec2::new(x, y), *text, size, *color, Align2::RIGHT_CENTER);
        x -= w + 6.0;
        let (a, b) = (Vec2::new(x - 16.0, y), Vec2::new(x, y));
        match swatch {
            Swatch::Line => line_of(f, MARKS, a, b, 2.0, *color),
            Swatch::Dashed => plot::dashed_of(f, MARKS, a, b, 1.6, *color),
            Swatch::Block => fill(f, (Vec2::new(x - 12.0, y - 5.0), Vec2::new(x - 2.0, y + 5.0)), color.a(0.6)),
        }
        x -= 22.0;
    }
}

/// How wide a key line is.
fn key_width(f: &Frame, entries: &[(Swatch, Color, &str)], size: f32) -> f32 {
    entries.iter().map(|(_, _, t)| f.text_size(t, size).x + 28.0).sum()
}

/// A square of side `side` centred at `at`: filled for a cell with more than expected,
/// hollow for one with fewer.
fn square(f: &mut Frame, at: Vec2, side: f32, more: bool, color: Color) {
    // (a part of nothing is no square: drawn clear, so that the outlines keep their number)
    let color = if side >= 1.0 { color } else { Color::CLEAR };
    let h = Vec2::splat(side.max(0.0) * 0.5);
    let (lo, hi) = (at - h, at + h);
    if more && side >= 1.0 {
        fill(f, (lo, hi), color.a(0.55));
    }
    path_of(f, SHAPES, &[lo, Vec2::new(hi.x, lo.y), hi, Vec2::new(lo.x, hi.y), lo], if more { 1.2 } else { 1.6 }, color);
}

/// Names along a line kept at least `gap` apart (and inside `lo` to `hi`), each as
/// near as it can be to where it wants to be: (where it wants to be, its width).
fn spread(wanted: &[(f32, f32)], lo: f32, hi: f32, gap: f32) -> Vec<f32> {
    let mut at: Vec<f32> = wanted.iter().map(|w| w.0).collect();
    let n = at.len();
    for _ in 0..8 {
        for i in 1..n {
            let need = (wanted[i - 1].1 + wanted[i].1) * 0.5 + gap;
            if at[i] - at[i - 1] < need {
                let push = (need - (at[i] - at[i - 1])) * 0.5;
                at[i - 1] -= push;
                at[i] += push;
            }
        }
        if let (Some(first), Some(last)) = (wanted.first(), wanted.last()) {
            at[0] = at[0].max(lo + first.1 * 0.5);
            at[n - 1] = at[n - 1].min(hi - last.1 * 0.5);
        }
    }
    at
}

// ----- the scene's frame ---------------------------------------------------------------

impl Scene for ChiSquare {
    fn view(&self) -> View {
        View::Flat { center: Vec2::ZERO, half_height: 1.0 }
    }

    fn settings(&self, s: &mut ViewSettings) {
        s.grid = false;
        s.bloom = s.bloom.min(0.2);
    }

    fn animated(&self) -> bool {
        self.look.moving || self.grow.is_some() || self.held.is_some() || (self.repeating && self.repeats.can_step())
    }

    fn pointer_hint(&self) -> Option<&'static str> {
        Some(match self.mode {
            Mode::Die => "green dot over the six: drag up or down to load the die    buttons over the picture: click",
            Mode::Table => "cyan dots: drag to move people between rows or between columns    buttons over the picture: click",
        })
    }

    fn ui(&mut self, ui: &mut egui::Ui, _: &Gfx) {
        ui::note(ui, r"Counts in categories (the faces a die came up, people sorted by where they live and what they drink) are compared with the counts a hypothesis expects, cell by cell. Each cell gives its observed count $O$ less its expected count $E$, squared, divided by $E$; their sum is the chi-square statistic:");
        ui::equation(ui, r"\chi^2 = \sum_{\text{cells}} \frac{(O - E)^2}{E}");
        ui::note(ui, r"If the hypothesis is true, and no expected count is small, $\chi^2$ follows nearly the chi-square curve with as many degrees of freedom as there are counts free to vary once the totals are fixed. A $\chi^2$ far out in the curve's tail is evidence against the hypothesis; the share of the curve beyond it is the $p$-value.");

        ui::section(ui, "Things to try");
        for (k, trial) in TRIALS.iter().enumerate() {
            if ui::experiment(ui, trial.0, trial.1) {
                self.set_up(k);
            }
        }

        ui::section(ui, "The data");
        let mut mode = if self.mode == Mode::Die { 0 } else { 1 };
        if ui::choice(ui, &["a die", "a table"], &mut mode) {
            self.switch(if mode == 0 { Mode::Die } else { Mode::Table });
        }
        match self.mode {
            Mode::Die => {
                ui::slider_int_log(ui, "rolls", &mut self.rolls, 0..=MOST_ROLLS);
                let mut six = self.six as f32;
                if ui::slider(ui, r"how often a six comes up, $P(\text{six})$", &mut six, 0.0..=1.0) {
                    self.six = six as f64;
                }
                ui::note(ui, r"The die rolled. The hypothesis tested is always a fair die, $P = 1/6$ for every face; the other five faces of this one share what the six leaves equally.");
                match ui::buttons(ui, &["make it fair", "roll again"]) {
                    Some(0) => self.six = FAIR,
                    Some(_) => self.seed = self.seed.wrapping_add(1),
                    None => {}
                }
            }
            Mode::Table => {
                let mut cols = self.table.cols - 2;
                if ui::choice(ui, &["two drinks", "three", "four"], &mut cols) {
                    self.set_columns(cols + 2);
                }
                if ui::slider_int_log(ui, "people", &mut self.people, 2..=MOST_PEOPLE) {
                    self.set_people(self.people);
                }
                ui::note(ui, "The same shares, more or fewer people.");
                ui::checkbox(ui, &mut self.keep_totals, "keep the totals");
                ui::note(ui, r"With the totals of the rows and the columns kept, only $(\text{rows} - 1)(\text{columns} - 1)$ counts can be chosen: the rest are forced. That number is the degrees of freedom.");
            }
        }

        ui::section(ui, "Many more samples");
        let label = if self.mode == Mode::Die { "roll many more sets" } else { "shuffle the people many times" };
        let mut on = self.repeating;
        if ui::checkbox(ui, &mut on, label).changed() {
            self.toggle_repeats();
        }
        ui::slider_int_log(ui, "samples a second", &mut self.rate, 1..=2000);
        ui::note(
            ui,
            match self.mode {
                Mode::Die => "New sets of as many rolls of the same die, each tested against a fair die. When the die is fair, that is the hypothesis at work: their histogram is what the curve stands in for.",
                Mode::Table => "The same people with the drinks they like dealt out again at random: whatever went with what is gone, the totals stay. That makes the hypothesis true, and shows how the statistic is really spread for these totals.",
            },
        );
        self.refresh();

        ui::section(ui, "Reading the picture");
        ui::key(
            ui,
            &[
                (DATA, "cyan: the observed counts (bars for the die; for the table, tiles whose areas are the counts)"),
                (ASSUMED, "purple: what the hypothesis expects (dashed), and the chi-square curve it implies for the statistic"),
                (ESTIMATE, r"yellow: each cell's part of $\chi^2$, $(O - E)^2/E$, as the area of a square (filled: more than expected; hollow: fewer), the parts laid end to end, and $\chi^2$; in the mosaic, where the observed tiles go beyond the expected ones"),
                (ALARM, "pink: the curve beyond the statistic, whose area is the $p$-value"),
                (TRUTH, "green: the true die (each face's count on average), and the statistic of many new samples, as a histogram"),
            ],
        );
        ui::section(ui, "Words");
        ui::term(ui, "Observed count:", "how many fell in a cell.");
        ui::term(ui, "Expected count:", r"how many the hypothesis says fall there on average: the total times the cell's share (a sixth, for a fair die); in a table, $\text{row total} \times \text{column total} / N$, with $N$ people in all.");
        ui::term(ui, "Chi-square statistic:", r"$\chi^2$, the sum of $(O - E)^2/E$ over the cells.");
        ui::term(ui, "Degrees of freedom:", r"how many counts can vary once the totals the expected counts use are fixed: the number of cells less one for one set of counts; $(\text{rows} - 1)(\text{columns} - 1)$ for a table.");
        ui::term(ui, "Goodness of fit:", "the test of one set of counts against given shares (is the die fair?).");
        ui::term(ui, "Test of independence:", "the test of whether two ways of sorting the same people have anything to do with each other: if not, every row has its people in the same shares across the columns.");
        ui::term(ui, "Contingency table:", "counts of people sorted two ways at once, one along the rows, the other along the columns.");
        ui::term(ui, r"Cramér's $V$:", r"$\sqrt{\chi^2 / (N(k - 1))}$, with $N$ people and $k$ the smaller of the numbers of rows and columns: the strength of the association, from 0 (none) to 1 (with two rows: each column's people all from one row). Unlike $\chi^2$, it does not grow with $N$.");
        ui::term(ui, r"$p$-value:", r"the share of the curve beyond the observed $\chi^2$: how often the hypothesis would give a $\chi^2$ at least as large.");

        ui::section(ui, "Numbers");
        let test = self.test();
        match self.mode {
            Mode::Die => ui::readout(ui, "rolls", self.rolls.to_string()),
            Mode::Table => ui::readout(ui, r"people, $N$", self.table.total().to_string()),
        }
        ui::readout(ui, r"$\chi^2$", format!("${}$", num(test.stat, 3)));
        ui::readout(ui, "degrees of freedom", test.df.to_string());
        ui::readout(ui, r"$p$-value", test.p().map_or("no test".to_string(), |p| format!("$p {}$", p_text(p))));
        ui::readout(ui, "smallest expected count", test.smallest_expected().map_or("none".to_string(), |e| num(e, 2)));
        if self.mode == Mode::Table {
            ui::readout(ui, r"Cramér's $V$", self.table.cramers_v(&test).map_or("no test".to_string(), |v| num(v, 3)));
            if let Some(z) = self.table.two_shares() {
                ui::readout(ui, r"$z$, two shares", format!("${}$", num(z.z, 3)));
                ui::readout(ui, r"$z^2$", format!("${}$", num(z.z * z.z, 3)));
                ui::readout(ui, r"$p$ from $z$, two-sided", format!("$p {}$", p_text(z.p())));
            }
        }
        let (done, said) = self.told;
        ui::readout(ui, "new samples", done.to_string());
        if let Some((mean, share)) = said {
            ui::readout(ui, r"their average $\chi^2$", num(mean, 2));
            ui::readout(ui, r"their share with $p < 0.05$", format!("{} %", num(100.0 * share, 1)));
        }
        ui::see_also(
            ui,
            &[
                ("chance", "independence as areas: the mosaic"),
                ("testing", r"the logic of a test: $p$-values, errors, power"),
                ("pvalues", r"many tests, and what $p$-values do then"),
                ("entropy", "mutual information: another measure of how far a table is from independence"),
                ("distributions", "the chi-square distribution, a sum of squared normal numbers"),
            ],
        );
    }

    fn frame(&mut self, f: &mut Frame) {
        let lay = Layout::of(f.view_size(), f.top_inset, self.mode);
        // (a canvas squeezed to a sliver is one the controls have opened over, on a phone)
        let tiny = lay.view.x < 160.0 || lay.view.y < 140.0;
        if !tiny {
            self.buttons(f, &lay);
        }
        if tiny || lay.compact {
            (self.held, self.held_table, self.held_height) = (None, None, None);
        } else {
            self.handles(f, &lay);
        }
        self.advance_growth(f.dt);
        self.refresh();
        self.work(f.dt, f.real_dt);
        self.tell(f.real_dt);

        let goals = self.goals();
        let now = (lay.view, self.mode, self.table.cols);
        let snap = self.drawn_as.map(|(v, m, c)| (v, m, if m == Mode::Table { c } else { 0 })) != Some((now.0, now.1, if now.1 == Mode::Table { now.2 } else { 0 }));
        // (a new histogram for other degrees of freedom is drawn at once: its bins mean other values)
        let hist_snap = self.look.hist.shown.len() != goals.hist.len();
        if snap && !tiny {
            self.drawn_as = Some(now);
        }
        // (by the clock: the picture answers the hand and the buttons also while the scene's time stands still)
        self.look.follow(&goals, f.real_dt, snap || hist_snap);
        if self.held == Some(Held::Six) {
            // (the level the hand holds is drawn where the hand puts it; the counts it
            // changes, whole numbers, glide)
            self.look.levels.snap(&goals.levels);
        }
        if tiny || lay.compact {
            self.draw_compact(f, &lay);
            return;
        }
        let mut signature = Vec::new();
        match self.mode {
            Mode::Die => self.draw_die(f, &lay, &mut signature),
            Mode::Table => self.draw_table(f, &lay),
        }
        self.draw_chart(f, &lay, &mut signature);
        self.draw_words(f, &lay);
        signature.extend([self.mode as u64, self.table.cols as u64]);
        self.keep_apart(f, &signature);
    }
}

impl ChiSquare {
    /// When which labels and ticks there are has changed (other round values on an
    /// axis, another mode), the frame draws a different number of them than the last,
    /// so that nothing takes the new ones for the old ones moved (as `eidos motion`
    /// would: it pairs them by their places in their batches).
    fn keep_apart(&mut self, f: &mut Frame, signature: &[u64]) {
        let sig = signature.iter().fold(0xcbf2_9ce4_8422_2325u64, |h, v| (h ^ v).wrapping_mul(0x1000_0000_01b3));
        let count = |f: &mut Frame| (plot::apart(f, AXES, false).pts.iter().filter(|p| p.width >= 0.0).count(), f.dl.labels.len());
        if sig != self.signature {
            let (axes, labels) = count(f);
            if axes == self.drawn_last.0 {
                let far = f.hud(Vec2::ZERO);
                plot::apart(f, AXES, false).segment(far, far, 1.0, Color::CLEAR);
            }
            if labels == self.drawn_last.1 {
                f.text_px(Vec2::ZERO, "", 1.0, Color::CLEAR, Align2::LEFT_TOP);
            }
        }
        self.signature = sig;
        self.drawn_last = count(f);
    }

    /// The buttons over the picture: the mode, and what makes new samples.
    fn buttons(&mut self, f: &mut Frame, lay: &Layout) {
        let (size, h, gap) = (lay.text, lay.button_h, 6.0);
        let width = |f: &Frame, words: &[&str]| words.iter().map(|w| f.text_size(w, size).x).fold(0.0, f32::max) + 18.0;
        let mut x = lay.side;
        for (mode, name) in [(Mode::Die, "a die"), (Mode::Table, "a table")] {
            let w = width(f, &[name]);
            if button(f, (Vec2::new(x, lay.top), Vec2::new(x + w, lay.top + h)), name, size, self.mode == mode) {
                self.switch(mode);
            }
            x += w + gap;
        }
        let mut right = lay.view.x - lay.side;
        let (start, stop) = if self.mode == Mode::Die { ("roll many more", "stop") } else { ("shuffle many times", "stop") };
        let w = width(f, &[start, stop]);
        let label = if self.repeating { stop } else { start };
        if right - w > x + 8.0 && button(f, (Vec2::new(right - w, lay.top), Vec2::new(right, lay.top + h)), label, size, self.repeating) {
            self.toggle_repeats();
        }
        right -= w + gap;
        if self.mode == Mode::Die {
            let w = width(f, &["roll again"]);
            if right - w > x + 8.0 && button(f, (Vec2::new(right - w, lay.top), Vec2::new(right, lay.top + h)), "roll again", size, false) {
                self.seed = self.seed.wrapping_add(1);
            }
        }
    }

    /// The chart of the die's counts, as drawn.
    fn count_plot(&self, lay: &Layout) -> Plot {
        Plot::new(lay.first, [0.5, 6.5], [0.0, self.look.axes.get(Y_COUNTS).max(1.0) as f64])
    }

    /// The mosaic as drawn: each column's left edge and width, and where its split is,
    /// and where the expected split is (screen pixels).
    fn mosaic(&self, lay: &Layout) -> Mosaic {
        let c = &self.look.counts;
        let cols = self.table.cols;
        let (min, max) = lay.first;
        let (w, h) = (max.x - min.x, max.y - min.y);
        let north: Vec<f32> = (0..cols).map(|j| c.get(j).max(0.0)).collect();
        let south: Vec<f32> = (0..cols).map(|j| c.get(COLS + j).max(0.0)).collect();
        let totals: Vec<f32> = (0..cols).map(|j| north[j] + south[j]).collect();
        let all: f32 = totals.iter().sum();
        let r1: f32 = north.iter().sum();
        let expected = if all > 0.0 { min.y + h * r1 / all } else { min.y + h * 0.5 };
        let mut left = min.x;
        let mut columns = Vec::with_capacity(cols);
        for j in 0..cols {
            let cw = if all > 0.0 { w * totals[j] / all } else { w / cols as f32 };
            let split = if totals[j] > 0.0 { min.y + h * north[j] / totals[j] } else { expected };
            columns.push((left, cw, split));
            left += cw;
        }
        Mosaic { columns, expected }
    }

    /// The handles: the six's level (die), the splits and the edges between columns (table).
    fn handles(&mut self, f: &mut Frame, lay: &Layout) {
        let mut held = None;
        match self.mode {
            Mode::Die => {
                if self.rolls > 0 {
                    let p = self.count_plot(lay);
                    let level = self.look.levels.get(FACES) as f64;
                    if let Some((_, y)) = p.handle(f, 1, 6.0, level, TRUTH) {
                        if self.held_height.is_none() {
                            self.held_height = Some(self.look.axes.get(Y_COUNTS));
                        }
                        self.six = (y / self.rolls as f64).max(0.0).min(1.0);
                        held = Some(Held::Six);
                    }
                }
            }
            Mode::Table => {
                let m = self.mosaic(lay);
                let (min, max) = lay.first;
                let (w, h) = ((max.x - min.x).max(1.0), (max.y - min.y).max(1.0));
                let t = self.table;
                for j in 0..t.cols {
                    let forced = self.keep_totals && j + 1 == t.cols;
                    let (left, cw, split) = m.columns[j];
                    if forced || t.col(j) == 0 || cw < 2.0 {
                        continue;
                    }
                    if let Some(to) = screen_handle(f, 10 + j as u32, Vec2::new(left + cw * 0.5, split), DATA) {
                        let share = ((to.y - min.y) / h).max(0.0).min(1.0) as f64;
                        self.set_split(j, (share * t.col(j) as f64).round() as u32);
                        held = Some(Held::Split(j));
                    }
                }
                if !self.keep_totals {
                    for j in 0..t.cols - 1 {
                        let (left, cw, _) = m.columns[j];
                        if let Some(to) = screen_handle(f, 20 + j as u32, Vec2::new(left + cw, max.y), DATA) {
                            if self.held != Some(Held::Edge(j)) || self.held_table.is_none() {
                                self.held_table = Some(self.table);
                            }
                            let from = self.held_table.unwrap_or(self.table);
                            self.set_edge(j, ((to.x - min.x) / w) as f64, &from);
                            held = Some(Held::Edge(j));
                        }
                    }
                }
                if held.is_some() {
                    self.people = self.table.total() as usize;
                }
            }
        }
        self.held = held;
        if held.is_none() {
            (self.held_table, self.held_height) = (None, None);
        }
    }
}

/// The mosaic as drawn: for each column (left edge, width, where its split is), and the
/// expected split.
struct Mosaic {
    columns: Vec<(f32, f32, f32)>,
    expected: f32,
}

// ----- drawing -------------------------------------------------------------------------

impl ChiSquare {
    /// A chart's title and, on the line under it, a key at its right.
    fn head(&self, f: &mut Frame, lay: &Layout, rect: Rect, left: f32, titles: &[&str], key: &[(Swatch, Color, &str)]) {
        let room = rect.1.x - left;
        let title = pick(f, titles, room, lay.big);
        f.text_px(Vec2::new(left, rect.0.y - lay.small - 16.0), title, lay.big, pal::TEXT, Align2::LEFT_BOTTOM);
        if !key.is_empty() && key_width(f, key, lay.small) <= room {
            key_line(f, rect.1.x, rect.0.y - lay.small * 0.5 - 6.0, key, lay.small);
        }
    }

    fn draw_die(&self, f: &mut Frame, lay: &Layout, signature: &mut Vec<u64>) {
        let test = self.test();
        let p = self.count_plot(lay);
        let (min, max) = lay.first;
        let e = self.look.levels.get(0) as f64;
        let loaded = (self.six - FAIR).abs() > 1e-9;
        p.backing(f, if lay.narrow { 30.0 } else { 40.0 }, lay.text + 10.0, lay.big + lay.small + 22.0, 8.0);
        let titles = [
            &format!("{} rolls of the die: how often each face came up", self.rolls)[..],
            &format!("{} rolls: how often each face came up", self.rolls)[..],
            &format!("{} rolls", self.rolls)[..],
        ];
        let expected = format!("expected if fair, $E = {}$", num(self.rolls as f64 * FAIR, if self.rolls.is_multiple_of(6) { 0 } else { 1 }));
        let key_fair = [(Swatch::Block, TRUTH, "drag: load the die"), (Swatch::Dashed, ASSUMED, &expected[..])];
        let key_loaded = [(Swatch::Line, TRUTH, "this die, on average (drag)"), (Swatch::Dashed, ASSUMED, &expected[..])];
        self.head(f, lay, lay.first, p.rect.0.x - if lay.narrow { 30.0 } else { 40.0 }, &titles, if loaded { &key_loaded } else { &key_fair });
        p.y_grid(f);
        p.y_axis(f, "", lay.small);
        signature.push(crate::scenes::stats::plot::Ticks::of(p.y[0], p.y[1], p.height() * 1.6).step.to_bits());
        line_of(f, AXES, Vec2::new(min.x, max.y), max, 1.0, pal::DIM.a(0.8));
        // the bars, their tops, their counts, and the faces under them
        for i in 0..FACES {
            let x = (i + 1) as f64;
            let count = self.look.counts.get(i) as f64;
            p.bar(f, x - 0.36, x + 0.36, 0.0, count, 0.0, DATA.a(0.3));
            let top = p.px_y(count.min(p.y[1]));
            line_of(f, SHAPES, Vec2::new(p.px_x(x - 0.36), top), Vec2::new(p.px_x(x + 0.36), top), 2.0, DATA);
            let said = test.observed.get(i).copied().unwrap_or(0);
            f.text_px(Vec2::new(p.px_x(x), (top - 3.0).max(min.y + lay.small)), said.to_string(), lay.small, DATA, Align2::CENTER_BOTTOM);
            f.text_px(Vec2::new(p.px_x(x), max.y + 5.0), (i + 1).to_string(), lay.text, pal::DIM, Align2::CENTER_TOP);
            // the true die's level for this face (drawn clear when it is the fair one's)
            let level = self.look.levels.get(1 + i) as f64;
            let y = p.px_y(level.min(p.y[1]));
            line_of(f, SHAPES, Vec2::new(p.px_x(x - 0.46), y), Vec2::new(p.px_x(x + 0.46), y), 2.0, if loaded { TRUTH } else { Color::CLEAR });
        }
        if self.rolls > 0 {
            p.hline(f, MARKS, e, 1.6, ASSUMED, true);
        } else {
            f.text_px((min + max) * 0.5, "no rolls: nothing to compare", lay.text, pal::DIM, Align2::CENTER_CENTER);
        }

        // the squares: each face's part of the statistic as an area
        let (umin, umax) = lay.under;
        let title = pick(f, &[r"each face's part of $\chi^2$: $(O - E)^2/E$, the area of its square", r"each face's part of $\chi^2$, $(O - E)^2/E$", r"parts of $\chi^2$"], umax.x - lay.side, lay.small);
        f.text_px(Vec2::new(lay.side, umin.y - 6.0), title, lay.small, ESTIMATE, Align2::LEFT_BOTTOM);
        let parts = self.shown_parts();
        let slot = width(lay.under) / 6.0;
        let biggest = (umax.y - umin.y).min(slot - 6.0).max(4.0);
        let scale = self.look.axes.get(SCALE).max(1e-3);
        let mid = (umin.y + umax.y) * 0.5;
        for (i, part) in parts.iter().enumerate().take(FACES) {
            let side = biggest * (*part as f32 / scale).max(0.0).sqrt().min(1.0);
            let x = p.px_x((i + 1) as f64);
            square(f, Vec2::new(x, mid), side, test.more(i), ESTIMATE);
            let value = test.parts.get(i).copied().unwrap_or(0.0);
            f.text_px(Vec2::new(x, umax.y + 3.0), num(value, 2), lay.small, ESTIMATE, Align2::CENTER_TOP);
        }
        // a square of 1, to read the others by
        let one = biggest * (1.0 / scale).sqrt().min(1.0);
        let at = Vec2::new(umin.x - (if lay.narrow { 15.0 } else { 20.0 }), mid);
        path_of(
            f,
            SHAPES,
            &[at - Vec2::splat(one * 0.5), at + Vec2::new(one * 0.5, -one * 0.5), at + Vec2::splat(one * 0.5), at + Vec2::new(-one * 0.5, one * 0.5), at - Vec2::splat(one * 0.5)],
            1.0,
            pal::DIM,
        );
        f.text_px(Vec2::new(at.x, umax.y + 3.0), "area 1", lay.small, pal::DIM, Align2::CENTER_TOP);
    }

    /// Each cell's part of the statistic as drawn (from the counts as they glide).
    fn shown_parts(&self) -> Vec<f64> {
        let c = &self.look.counts;
        match self.mode {
            Mode::Die => {
                let e = (0..FACES).map(|i| c.get(i) as f64).sum::<f64>() * FAIR;
                (0..FACES).map(|i| if e > 0.0 { (c.get(i) as f64 - e).powi(2) / e } else { 0.0 }).collect()
            }
            Mode::Table => {
                let cols = self.table.cols;
                let cell = |i: usize, j: usize| c.get(i * COLS + j).max(0.0) as f64;
                let rows = [0, 1].map(|i| (0..cols).map(|j| cell(i, j)).sum::<f64>());
                let all = rows[0] + rows[1];
                let mut parts = Vec::with_capacity(2 * cols);
                for (i, row) in rows.iter().enumerate() {
                    for j in 0..cols {
                        let e = if all > 0.0 { row * (cell(0, j) + cell(1, j)) / all } else { 0.0 };
                        parts.push(if e > 0.0 { (cell(i, j) - e).powi(2) / e } else { 0.0 });
                    }
                }
                parts
            }
        }
    }

    fn draw_table(&self, f: &mut Frame, lay: &Layout) {
        let t = &self.table;
        let test = t.test();
        let m = self.mosaic(lay);
        let (min, max) = lay.first;
        let margin = min.x - lay.side;
        hud_panel(f, min - Vec2::new(margin, lay.big + lay.small + 22.0), max + Vec2::new(8.0, 2.0 * (lay.small + 3.0) + 8.0));
        let n = t.total();
        let titles = [
            &format!("{n} people, by place and favourite drink: tile areas are counts")[..],
            &format!("{n} people, by place and drink: areas are counts")[..],
            &format!("{n} people, by place and drink")[..],
        ];
        self.head(f, lay, lay.first, lay.side, &titles, &[(Swatch::Block, ESTIMATE, "beyond expected"), (Swatch::Dashed, ASSUMED, "expected if independent")]);
        let cols = t.cols;
        // the tiles, and where they go beyond what independence expects
        for (j, &(left, cw, split)) in m.columns.iter().enumerate() {
            let (l, r) = (left + 1.0, (left + cw - 1.0).max(left + 1.0));
            fill(f, (Vec2::new(l, min.y), Vec2::new(r, split)), DATA.a(0.3));
            fill(f, (Vec2::new(l, split), Vec2::new(r, max.y)), DATA.a(0.08));
            if (split - m.expected).abs() > 0.5 && t.col(j) > 0 {
                fill(f, (Vec2::new(l, split.min(m.expected)), Vec2::new(r, split.max(m.expected))), ESTIMATE.a(0.42));
            }
            // the split as a line, and the column's right edge
            line_of(f, SHAPES, Vec2::new(l, split), Vec2::new(r, split), 2.0, DATA);
            line_of(f, SHAPES, Vec2::new(left + cw, min.y), Vec2::new(left + cw, max.y), 1.0, pal::DIM.a(0.6));
        }
        let edge = [min, Vec2::new(max.x, min.y), max, Vec2::new(min.x, max.y), min];
        path_of(f, SHAPES, &edge, 1.0, pal::DIM.a(0.8));
        if n > 0 {
            plot::dashed_of(f, MARKS, Vec2::new(min.x, m.expected), Vec2::new(max.x, m.expected), 2.6, ASSUMED.mix(pal::WHITE, 0.15));
        }
        // the counts in their tiles (clear where a tile is too small for its number)
        for (j, &(left, cw, split)) in m.columns.iter().enumerate() {
            for (i, (top, bottom)) in [(min.y, split), (split, max.y)].into_iter().enumerate() {
                let count = t.n[i][j];
                let room = cw >= f.text_size(&count.to_string(), lay.text).x + 6.0 && bottom - top >= lay.text + 4.0;
                let color = if room { pal::TEXT } else { Color::CLEAR };
                f.caption_px(Vec2::new(left + cw * 0.5, (top + bottom) * 0.5), count.to_string(), lay.text, color, Align2::CENTER_CENTER);
            }
            // a forced split (with the totals kept) is marked, not offered to the hand
            if self.keep_totals && j + 1 == cols && cw > 2.0 {
                hud_circle(f, Vec2::new(left + cw * 0.5, split), 6.0, 1.6, DATA);
            }
        }
        if self.keep_totals {
            for (j, &(left, cw, split)) in m.columns.iter().enumerate() {
                let word = if j + 1 == cols { "forced" } else { "free" };
                let below = split + 9.0 < max.y - lay.small - 4.0;
                let y = if below { split + 9.0 } else { split - 9.0 };
                let align = if below { Align2::CENTER_TOP } else { Align2::CENTER_BOTTOM };
                f.caption_px(Vec2::new(left + cw * 0.5, y), word, lay.small, if j + 1 == cols { pal::DIM } else { DATA }, align);
            }
        }
        // the rows' names and totals at the left, by the expected split
        for (i, (top, bottom)) in [(min.y, m.expected), (m.expected, max.y)].into_iter().enumerate() {
            let y = ((top + bottom) * 0.5).max(min.y + lay.small).min(max.y - lay.small);
            f.text_px(Vec2::new(min.x - 6.0, y - 1.0), PLACES[i], lay.small, pal::TEXT, Align2::RIGHT_BOTTOM);
            f.text_px(Vec2::new(min.x - 6.0, y + 1.0), t.row(i).to_string(), lay.small, pal::DIM, Align2::RIGHT_TOP);
        }
        // the columns' names and totals under them, kept apart
        let wanted: Vec<(f32, f32)> = m
            .columns
            .iter()
            .enumerate()
            .map(|(j, &(left, cw, _))| (left + cw * 0.5, f.text_size(DRINKS[j], lay.small).x.max(f.text_size(&t.col(j).to_string(), lay.small).x)))
            .collect();
        let xs = spread(&wanted, min.x - margin * 0.5, max.x + 6.0, 6.0);
        for (j, x) in xs.iter().enumerate() {
            f.text_px(Vec2::new(*x, max.y + 4.0), DRINKS[j], lay.small, pal::TEXT, Align2::CENTER_TOP);
            f.text_px(Vec2::new(*x, max.y + lay.small + 7.0), t.col(j).to_string(), lay.small, pal::DIM, Align2::CENTER_TOP);
        }

        // the table of numbers: observed, expected, and each cell's part as a square
        let (umin, umax) = lay.under;
        hud_panel(f, Vec2::new(lay.side, umin.y - lay.small - 12.0), umax + Vec2::new(8.0, 6.0));
        let title = pick(
            f,
            &[
                r"observed (cyan), expected (purple), and each cell's part of $\chi^2$, $(O - E)^2/E$, as a square",
                r"observed (cyan), expected (purple), part of $\chi^2$ (yellow)",
                r"observed, expected, part of $\chi^2$",
            ],
            umax.x - lay.side,
            lay.small,
        );
        f.text_px(Vec2::new(lay.side, umin.y - 6.0), title, lay.small, pal::DIM, Align2::LEFT_BOTTOM);
        let header = lay.small + 6.0;
        let cw = (umax.x - umin.x) / cols as f32;
        let ch = (umax.y - umin.y - header) * 0.5;
        let parts = self.shown_parts();
        let scale = self.look.axes.get(SCALE).max(1e-3);
        let biggest = (ch - lay.small - 8.0).min(cw * 0.45).max(4.0);
        for j in 0..cols {
            let x = umin.x + j as f32 * cw;
            f.text_px(Vec2::new(x + cw * 0.5, umin.y), DRINKS[j], lay.small, pal::DIM, Align2::CENTER_TOP);
            line_of(f, SHAPES, Vec2::new(x, umin.y + header - 2.0), Vec2::new(x, umax.y), 1.0, pal::DIM.a(0.25));
        }
        for i in 0..2 {
            let y = umin.y + header + i as f32 * ch;
            f.text_px(Vec2::new(umin.x - 6.0, y + ch * 0.5), PLACES[i], lay.small, pal::TEXT, Align2::RIGHT_CENTER);
            line_of(f, SHAPES, Vec2::new(umin.x, y), Vec2::new(umax.x, y), 1.0, pal::DIM.a(0.25));
            for j in 0..cols {
                let k = i * cols + j;
                let x = umin.x + j as f32 * cw;
                let left = x + 6.0;
                f.text_px(Vec2::new(left, y + ch * 0.5 - 1.0), t.n[i][j].to_string(), lay.text, DATA, Align2::LEFT_BOTTOM);
                let e = test.expected.get(k).copied().unwrap_or(0.0);
                f.text_px(Vec2::new(left, y + ch * 0.5 + 1.0), num(e, 1), lay.small, ASSUMED, Align2::LEFT_TOP);
                let part = parts.get(k).copied().unwrap_or(0.0);
                let side = biggest * (part as f32 / scale).max(0.0).sqrt().min(1.0);
                let at = Vec2::new(x + cw - biggest * 0.5 - 6.0, y + 3.0 + biggest * 0.5);
                square(f, at, side, test.more(k), ESTIMATE);
                let value = test.parts.get(k).copied().unwrap_or(0.0);
                f.text_px(Vec2::new(at.x, y + biggest + 5.0), num(value, 2), lay.small, ESTIMATE, Align2::CENTER_TOP);
            }
        }
    }

    /// The chi-square chart: the curve, the histogram of new samples, the statistic with
    /// the tail beyond it, and the parts laid end to end under it.
    fn draw_chart(&self, f: &mut Frame, lay: &Layout, signature: &mut Vec<u64>) {
        let test = self.test();
        let df = test.df;
        let x_top = self.look.axes.get(X_CHI).max(1.0) as f64;
        let y_top = self.look.axes.get(Y_CHI).max(1e-3) as f64;
        let p = Plot::new(lay.chart, [0.0, x_top], [0.0, y_top]);
        let (min, max) = lay.chart;
        let left = lay.chart.0.x - if lay.narrow { 14.0 } else { 20.0 };
        p.backing(f, min.x - left, lay.small + 26.0, lay.big + lay.small + 22.0, 8.0);
        let what = if self.mode == Mode::Die { "the die is fair" } else { "place and drink are independent" };
        let d = if df == 1 { "1 degree of freedom".to_string() } else { format!("{df} degrees of freedom") };
        let long = format!(r"$\chi^2$ if {what}: the chi-square curve, {d}");
        let medium = format!(r"$\chi^2$ if {what}: {d}");
        let short = format!(r"$\chi^2$ if {what}");
        let none = r"no test: nothing to compare".to_string();
        let titles: Vec<&str> = if df == 0 { vec![&none] } else { vec![&long, &medium, &short] };
        let sample = if self.mode == Mode::Die { "new sets of rolls" } else { "shuffles" };
        let key = [(Swatch::Block, TRUTH, sample), (Swatch::Line, ASSUMED, "the curve")];
        self.head(f, lay, lay.chart, left, &titles, if df == 0 { &[] } else { &key });
        p.x_axis(f, "", lay.small);
        signature.push(crate::scenes::stats::plot::Ticks::of(p.x[0], p.x[1], p.width()).step.to_bits());
        line_of(f, AXES, min, Vec2::new(min.x, max.y), 1.0, pal::DIM.a(0.8));
        f.text_px(Vec2::new(max.x - 2.0, max.y - lay.small - 8.0), r"$\chi^2$", lay.small, pal::DIM, Align2::RIGHT_BOTTOM);
        if df == 0 {
            let why = match self.mode {
                Mode::Die => "no rolls",
                Mode::Table => "only one row or one column holds anybody",
            };
            f.text_px((min + max) * 0.5, why, lay.text, pal::DIM, Align2::CENTER_CENTER);
            return;
        }
        // the histogram of new samples, bars and outline
        let w = self.repeats.width;
        let shown_bins = ((x_top / w).ceil() as usize).min(BINS);
        let heights = &self.look.hist;
        for i in 0..shown_bins {
            let h = heights.get(i) as f64;
            if h > 0.0 {
                p.bar(f, i as f64 * w, (i + 1) as f64 * w, 0.0, h, 0.0, TRUTH.a(0.05));
            }
        }
        // (kept inside the chart, the outline and the curve are never cut at its edges,
        // and keep their number of points from frame to frame)
        let outline = (0..shown_bins).flat_map(|i| {
            let h = (heights.get(i) as f64).min(y_top);
            [((i as f64 * w).min(x_top), h), (((i + 1) as f64 * w).min(x_top), h)]
        });
        p.path(f, HIST, outline, 1.2, TRUTH.a(0.75));
        // the curve, and the tail beyond the statistic
        let dist = Dist::ChiSquare { df: df as f64 };
        let stat = self.shown_parts().iter().sum::<f64>();
        let density = |x: f64| {
            let d = dist.pdf(x);
            if d.is_nan() {
                0.0
            } else {
                d.min(y_top)
            }
        };
        p.area(f, stat, x_top, 0.0, 90, ALARM.a(0.24), density);
        p.curve(f, CURVE, 180, 2.2, ASSUMED, density);
        // the 5 % point
        let crit = critical(df, 0.05);
        p.vline(f, MARKS, crit, 1.0, pal::DIM.a(0.7), true);
        if crit < x_top {
            // (on the side away from the statistic's line, gliding across when it crosses)
            let half = f.text_size("5 % beyond", lay.small) * 0.5;
            let room_left = p.px_x(crit) - 2.0 * half.x - 6.0 >= min.x;
            let side = if test.stat > crit && room_left { -1.0 } else { 1.0 };
            let from = Vec2::new(p.px_x(crit), min.y + 2.0);
            f.text_placed("five", 0, from, Vec2::new(side * (half.x + 3.0), half.y), "5 % beyond", lay.small, pal::DIM);
        }
        // the statistic: a line up the chart, or an arrow at its edge if it is beyond
        let said = format!(r"$\chi^2 = {}$", stat_text(test.stat));
        let pv = test.p().unwrap_or(1.0);
        let p_said = format!("$p {}$", p_text(pv));
        let size = f.text_size(&said, lay.text);
        let mid = min.y + height(lay.chart) * 0.32;
        if stat <= x_top {
            let x = p.px_x(stat);
            p.vline(f, MARKS, stat, 2.2, ESTIMATE, false);
            let lx = (x + 5.0).min(max.x - size.x - 2.0);
            f.caption_px(Vec2::new(lx, min.y + lay.small + 8.0), said, lay.text, ESTIMATE, Align2::LEFT_TOP);
            let px_size = f.text_size(&p_said, lay.text);
            let py = (x + 5.0).min(max.x - px_size.x - 2.0);
            f.caption_px(Vec2::new(py, mid + lay.text + 6.0), p_said, lay.text, ALARM, Align2::LEFT_TOP);
        } else {
            let y = min.y + lay.small + 8.0 + size.y * 0.5;
            hud_arrow(f, Vec2::new(max.x - 34.0, y + size.y + 6.0), Vec2::new(max.x - 4.0, y + size.y + 6.0), 2.0, ESTIMATE);
            f.caption_px(Vec2::new(max.x - 4.0, y), said, lay.text, ESTIMATE, Align2::RIGHT_CENTER);
            f.caption_px(Vec2::new(max.x - 4.0, mid + lay.text + 6.0), p_said, lay.text, ALARM, Align2::RIGHT_TOP);
        }
        // the parts laid end to end along the foot of the chart, from 0 to the statistic
        let parts = self.shown_parts();
        let strip = lay.small + 4.0;
        let (y0, y1) = (max.y - strip - 1.0, max.y - 1.0);
        let end = p.px_x(stat.min(x_top));
        hud_rect_at(f, 2, Vec2::new(min.x, y0 - 1.0), Vec2::new(end + 1.0, y1), Color::hex(0x080B14));
        let mut from = 0.0;
        for (i, part) in parts.iter().enumerate() {
            let to = from + part;
            let (a, b) = (p.px_x(from.min(x_top)), p.px_x(to.min(x_top)));
            let (lo, hi) = (Vec2::new(a, y0), Vec2::new(b, y1));
            let more = test.more(i);
            if more {
                hud_rect_at(f, 2, lo, hi, ESTIMATE.mix(Color::hex(0x080B14), 0.35));
            }
            path_of(f, SHAPES, &[lo, Vec2::new(hi.x, lo.y), hi, Vec2::new(lo.x, hi.y), lo], 1.0, ESTIMATE);
            if self.mode == Mode::Die {
                // (each part named by its face where it is wide enough)
                let room = b - a >= lay.small * 0.8;
                let color = match (room, more) {
                    (false, _) => Color::CLEAR,
                    (true, true) => Color::hex(0x080B14),
                    (true, false) => ESTIMATE,
                };
                f.text_px(Vec2::new((a + b) * 0.5, (y0 + y1) * 0.5), (i + 1).to_string(), lay.small * 0.9, color, Align2::CENTER_CENTER);
            }
            from = to;
        }
        let name = "the parts, end to end";
        let room = max.x - end - 8.0 >= f.text_size(name, lay.small).x && stat <= x_top;
        f.text_px(Vec2::new(end + 5.0, (y0 + y1) * 0.5), name, lay.small, if room { ESTIMATE } else { Color::CLEAR }, Align2::LEFT_CENTER);
    }

    /// What the numbers say, under the chart.
    fn draw_words(&self, f: &mut Frame, lay: &Layout) {
        let (min, max) = lay.words;
        let room = max.x - min.x;
        let size = if lay.narrow { lay.small } else { lay.text };
        let mut y = min.y;
        let test = self.test();
        let say = |f: &mut Frame, text: &str, color: Color, y: &mut f32| {
            *y += self.wrap(f, Vec2::new(min.x, *y), room, text, size, color, max.y) + 3.0;
        };
        if test.df == 0 {
            let why = match self.mode {
                Mode::Die => "No rolls: nothing to compare.".to_string(),
                Mode::Table => "No test: with only one row or one column holding anybody, there is nothing to compare.".to_string(),
            };
            say(f, &why, pal::DIM, &mut y);
            return;
        }
        let pv = test.p().unwrap_or(1.0);
        let df = match self.mode {
            Mode::Die => format!(r"$6 - 1 = 5$ degrees of freedom (six counts that must add up to {})", self.rolls),
            Mode::Table => {
                let (rows, cols) = self.table.filled();
                let word = if test.df == 1 { "degree" } else { "degrees" };
                format!(r"$({rows} - 1)({cols} - 1) = {}$ {word} of freedom", test.df)
            }
        };
        say(f, &format!(r"$\chi^2 = {}$ with {df}: $p {}$.", stat_text(test.stat), p_text(pv)), pal::TEXT, &mut y);
        let share = if pv < 0.0001 { "less than 0.01 %".to_string() } else { format!("about {} %", num(100.0 * pv, if pv < 0.1 { 1 } else { 0 })) };
        let meaning = match self.mode {
            Mode::Die => format!(r"If the die is fair, sets of {} rolls give a $\chi^2$ at least this large {share} of the time (by the curve).", self.rolls),
            Mode::Table => format!(r"If place and drink had nothing to do with each other, a $\chi^2$ at least this large would come up {share} of the time (by the curve)."),
        };
        say(f, &meaning, pal::DIM, &mut y);
        if self.mode == Mode::Table {
            if let Some(z) = self.table.two_shares() {
                say(
                    f,
                    &format!(
                        r"Tea: {} % of the north, {} % of the south. $z = {}$, and $z^2 = {} = \chi^2$: the same $p$.",
                        num(100.0 * z.first, 0),
                        num(100.0 * z.second, 0),
                        num(z.z, 2),
                        num(z.z * z.z, 2)
                    ),
                    pal::TEXT,
                    &mut y,
                );
            }
            if let Some(v) = self.table.cramers_v(&test) {
                say(f, &format!(r"Cramér's $V = {}$: the strength of the association, from 0 (none) to 1; it does not grow with the number of people.", num(v, 2)), pal::TEXT, &mut y);
            }
        }
        if let Some(e) = test.smallest_expected().filter(|e| *e < 5.0) {
            say(f, &format!("Expected counts as small as {}: below about 5, the curve can be a poor guide.", num(e, 1)), ALARM, &mut y);
        }
        if let (done, Some((mean, share))) = self.told {
            let what = match self.mode {
                Mode::Die => format!("{} new sets of {} rolls of this die", count(done), self.rolls),
                Mode::Table => format!("{} shuffles", count(done)),
            };
            say(f, &format!(r"{what}: average $\chi^2$ {}, and {} % reach $p < 0.05$.", num(mean, 2), num(100.0 * share, 1)), TRUTH, &mut y);
        }
    }

    /// On a canvas too small for the charts: the numbers alone.
    fn draw_compact(&self, f: &mut Frame, lay: &Layout) {
        let test = self.test();
        // (under the buttons over the picture, when there is room for them)
        let at = Vec2::new(8.0, if lay.view.x < 160.0 || lay.view.y < 140.0 { lay.top } else { lay.top + lay.button_h + 6.0 });
        let line = match test.p() {
            Some(p) => format!(r"$\chi^2 = {}$, {} d.f., $p {}$", stat_text(test.stat), test.df, p_text(p)),
            None => "no test: nothing to compare".to_string(),
        };
        f.caption_px(at, line, lay.small, pal::TEXT, Align2::LEFT_TOP);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn close(a: f64, b: f64, tol: f64) -> bool {
        (a - b).abs() <= tol * (1.0 + b.abs())
    }

    /// The scene as an experiment leaves it, its new samples made to `samples`.
    fn run(k: usize, samples: usize) -> ChiSquare {
        let mut s = ChiSquare::create();
        s.set_up(k);
        while let Some(g) = s.grow {
            let _ = g;
            s.advance_growth(0.25);
        }
        s.refresh();
        while s.repeats.can_step() && s.repeats.done < samples {
            s.repeats.step();
        }
        s
    }

    /// The statistic is the sum of the cells' parts, each (O − E)²/E, with the expected
    /// counts by definition: a sixth of the rolls each; row total × column total / N.
    #[test]
    fn the_statistic_is_the_sum_of_its_parts() {
        let t = die_test(&[8, 12, 9, 11, 6, 14]);
        assert_eq!(t.expected, vec![10.0; 6]);
        let by_hand = (4.0 + 4.0 + 1.0 + 1.0 + 16.0 + 16.0) / 10.0;
        assert!(close(t.stat, by_hand, 1e-12) && t.df == 5, "{t:?}");
        assert!(close(t.parts.iter().sum::<f64>(), t.stat, 1e-12));
        assert!(t.more(1) && !t.more(0));
        // a table: rows 50 and 50, columns 30, 42 and 28 of 100
        let t = NORTH_LIKES_TEA.test();
        assert_eq!(t.expected, vec![15.0, 21.0, 14.0, 15.0, 21.0, 14.0]);
        let by_hand = 25.0 / 15.0 * 2.0 + 9.0 / 21.0 * 2.0 + 4.0 / 14.0 * 2.0;
        assert!(close(t.stat, by_hand, 1e-12) && t.df == 2, "{t:?}");
        assert_eq!(stat_text(t.stat), "4.76");
        // the same with a die_stat, and the quick statistic of the repeats
        assert!(close(die_stat(&[8, 12, 9, 11, 6, 14]), 4.2, 1e-12));
        assert_eq!(die_stat(&[0; 6]), 0.0);
    }

    /// p-values are the tail of `Dist::ChiSquare` (and agree with the incomplete gamma
    /// function's own tail); the degrees of freedom follow the cells that are in.
    #[test]
    fn p_values_and_degrees_of_freedom() {
        for (x, df) in [(4.2, 5), (11.07, 5), (3.841, 1), (5.991, 2), (0.0, 3), (60.0, 2)] {
            let q = crate::math::stats::gamma_q(df as f64 / 2.0, x / 2.0);
            assert!((tail(x, df) - q).abs() < 1e-12, "{x} {df}");
        }
        assert!(close(critical(5, 0.05), 11.070_497_693_516_351, 1e-9));
        assert!(close(critical(1, 0.05), 3.841_458_820_694_124, 1e-9));
        // an empty column drops out with its degree of freedom
        let t = Table::of(3, [5, 0, 7, 0], [9, 0, 2, 0]);
        let test = t.test();
        assert_eq!((test.df, test.expected[1], test.parts[1]), (1, 0.0, 0.0));
        assert!(test.stat.is_finite() && test.p().is_some());
        // an empty row: one row, nothing to compare
        let one_row = Table::of(3, [5, 6, 7, 0], [0, 0, 0, 0]);
        let test = one_row.test();
        assert!(test.df == 0 && test.p().is_none() && test.stat == 0.0 && one_row.cramers_v(&test).is_none() && one_row.two_shares().is_none());
        // nobody at all; no rolls
        let nobody = Table::of(2, [0; 4], [0; 4]).test();
        assert!(nobody.df == 0 && nobody.stat == 0.0 && nobody.smallest_expected().is_none());
        let none = die_test(&[0; 6]);
        assert!(none.df == 0 && none.p().is_none() && none.parts.iter().all(|p| *p == 0.0));
        assert_eq!(p_text(0.00001), "< 0.0001");
        assert_eq!(p_text(0.0123), "= 0.012");
        assert_eq!(p_text(0.523), "= 0.52");
    }

    /// For two rows and two columns, χ² is the square of the z statistic for two shares
    /// (pooled), and the two p-values are the same.
    #[test]
    fn two_by_two_chi_square_is_z_squared() {
        let mut rng = Rng::new(5);
        let mut checked = 0;
        for _ in 0..500 {
            let n = [0, 1].map(|_| [rng.below(60) as u32, rng.below(60) as u32, 0, 0]);
            let t = Table::of(2, n[0], n[1]);
            let test = t.test();
            if let Some(z) = t.two_shares() {
                assert!(close(z.z * z.z, test.stat, 1e-9), "{t:?}: z² {} against χ² {}", z.z * z.z, test.stat);
                assert!((z.p() - test.p().unwrap_or(f64::NAN)).abs() < 1e-9, "{t:?}");
                checked += 1;
            } else {
                assert_eq!(test.df, 0, "{t:?}");
            }
        }
        assert!(checked > 480);
        // the experiment's table
        let s = run(4, 0);
        let (test, z) = (s.table.test(), s.table.two_shares().expect("two shares"));
        assert_eq!((num(100.0 * z.first, 0), num(100.0 * z.second, 0)), ("60".to_string(), "36".to_string()));
        assert_eq!((num(z.z, 2), num(test.stat, 2), num(z.z * z.z, 2)), ("2.40".to_string(), "5.77".to_string(), "5.77".to_string()));
        assert_eq!((p_text(z.p()), p_text(test.p().unwrap_or(1.0))), ("= 0.016".to_string(), "= 0.016".to_string()));
    }

    /// Ten times the people in the same shares: every count exactly ten times, χ² exactly
    /// ten times, V the same; and the experiment's numbers.
    #[test]
    fn ten_times_the_people() {
        let mut s = ChiSquare::create();
        s.set_up(3);
        let before = s.table;
        let (t0, v0) = (before.test(), before.cramers_v(&before.test()).expect("a test"));
        assert_eq!((stat_text(t0.stat), p_text(t0.p().unwrap_or(1.0)), num(v0, 2)), ("4.76".to_string(), "= 0.092".to_string(), "0.22".to_string()));
        // the growth, frame by frame at 60 a second
        let mut frames = 0;
        while s.grow.is_some() {
            s.advance_growth(1.0 / 60.0);
            frames += 1;
            assert!(frames < 400);
        }
        assert!(frames as f32 >= GROW_SECONDS * 60.0 - 1.0);
        for i in 0..2 {
            for j in 0..COLS {
                assert_eq!(s.table.n[i][j], 10 * before.n[i][j]);
            }
        }
        let t1 = s.table.test();
        assert!((t1.stat / t0.stat - 10.0).abs() < 1e-12, "{} against {}", t1.stat, t0.stat);
        let v1 = s.table.cramers_v(&t1).expect("a test");
        assert!((v1 - v0).abs() < 1e-12);
        assert_eq!((stat_text(t1.stat), p_text(t1.p().unwrap_or(1.0))), ("47.62".to_string(), "< 0.0001".to_string()));
        // and rescaling in general keeps the number of people exactly
        for people in [2, 7, 99, 1001, 20_000] {
            assert_eq!(rescale(&NORTH_LIKES_TEA, people).total(), people);
        }
        assert_eq!(rescale(&Table::of(2, [0; 4], [0; 4]), 9).total(), 9);
    }

    /// The die experiments: the rolls shown, and what many new sets of rolls do.
    #[test]
    fn the_die_experiments_say_what_their_captions_say() {
        // a fair die: χ² = 4.40, p = 0.49; new sets average 5, about 5 % beyond
        let s = run(0, MOST);
        let t = s.test();
        assert_eq!(s.die_counts().iter().sum::<u32>(), 60);
        assert_eq!((stat_text(t.stat), p_text(t.p().unwrap_or(1.0))), ("4.40".to_string(), "= 0.49".to_string()));
        let (mean, share) = (s.repeats.mean().unwrap_or(0.0), s.repeats.share_beyond().unwrap_or(0.0));
        assert!((mean - 5.0).abs() < 0.08, "average {mean}");
        assert!((share - 0.05).abs() < 0.006, "share {share}");
        // a loaded die: 17 sixes, p = 0.29 here; about 22 % of sets of 60 reach p < 0.05
        let s = run(1, MOST);
        let t = s.test();
        assert_eq!((s.die_counts()[5], p_text(t.p().unwrap_or(1.0))), (17, "= 0.29".to_string()));
        let share = s.repeats.share_beyond().unwrap_or(0.0);
        assert!((share - 0.22).abs() < 0.012, "share {share}");
        // and 600 rolls: about 99 %
        let mut s = run(1, 0);
        s.rolls = 600;
        s.refresh();
        for _ in 0..4000 {
            s.repeats.step();
        }
        let share = s.repeats.share_beyond().unwrap_or(0.0);
        assert!(share > 0.98 && share < 0.996, "share {share}");
    }

    /// Under the hypothesis, the statistic averages its degrees of freedom whatever the
    /// number of rolls (the caption says "for any number of rolls").
    #[test]
    fn a_fair_die_averages_five_for_any_number_of_rolls() {
        for n in [1, 6, 30, 600] {
            let mut r = Repeats::new();
            r.restart(Source::Die { shares: die_shares(FAIR), n }, 5);
            for _ in 0..MOST {
                r.step();
            }
            let mean = r.mean().unwrap_or(0.0);
            // (the spread of the average of 10 000 is about 0.03 here; at 1 roll, 0)
            assert!((mean - 5.0).abs() < 0.12, "{n} rolls: average {mean}");
        }
    }

    /// The table experiments: same shares; small counts.
    #[test]
    fn the_table_experiments_say_what_their_captions_say() {
        // same shares, totals kept: χ² = 0; the tea and coffee splits are free, juice's forced
        let mut s = run(2, 0);
        let t = s.table.test();
        assert!(t.stat == 0.0 && t.df == 2 && t.parts.iter().all(|p| *p == 0.0) && s.keep_totals);
        let rows = [s.table.row(0), s.table.row(1)];
        let cols: Vec<u32> = (0..3).map(|j| s.table.col(j)).collect();
        s.set_split(0, 20);
        assert_eq!(s.table.n[0][0], 20);
        assert_eq!([s.table.row(0), s.table.row(1)], rows);
        assert_eq!((0..3).map(|j| s.table.col(j)).collect::<Vec<_>>(), cols);
        assert_eq!(s.table.n[0][2], 14 - 5, "the juice column took up the difference");
        // as far as the juice column allows, and no further
        s.set_split(1, 0);
        assert_eq!((s.table.n[0][1], s.table.n[0][2]), (2, 28));
        // the forced split cannot be set
        let was = s.table;
        s.set_split(2, 0);
        assert_eq!(s.table, was);

        // small counts: every expected count 1; by the curve χ² = 8.00, p = 0.046; the
        // shuffles: three values, 8.6 % beyond the 5 % point (exactly 6/70 = 0.0857)
        let s = run(5, MOST);
        let t = s.table.test();
        assert!(t.expected.iter().all(|e| *e == 1.0) && t.df == 3);
        assert_eq!((stat_text(t.stat), p_text(t.p().unwrap_or(1.0))), ("8.00".to_string(), "= 0.046".to_string()));
        let share = s.repeats.share_beyond().unwrap_or(0.0);
        assert!((share - 6.0 / 70.0).abs() < 0.008, "share {share}");
        let filled = s.repeats.bins.iter().filter(|b| **b > 0).count();
        assert_eq!(filled, 3, "three spikes");
    }

    /// The small table's false alarm rate exactly: every way of dealing its people out
    /// again with the totals kept, weighted by how many shuffles give it (the
    /// hypergeometric counts), and the share of them that the curve calls significant.
    #[test]
    fn small_counts_false_alarms_are_six_in_seventy() {
        let t = SMALL;
        let columns = [0, 1, 2, 3].map(|j| t.col(j));
        let (north, n) = (t.row(0), t.total());
        let choose = |n: u32, k: u32| -> f64 { (0..k).map(|i| (n - i) as f64 / (i + 1) as f64).product() };
        let crit = critical(3, 0.05);
        let (mut all, mut alarms, mut mean) = (0.0, 0.0, 0.0);
        for a in 0..=columns[0] {
            for b in 0..=columns[1] {
                for c in 0..=columns[2] {
                    for d in 0..=columns[3] {
                        if a + b + c + d != north {
                            continue;
                        }
                        let ways = choose(columns[0], a) * choose(columns[1], b) * choose(columns[2], c) * choose(columns[3], d);
                        let dealt = Table::of(4, [a, b, c, d], [columns[0] - a, columns[1] - b, columns[2] - c, columns[3] - d]);
                        let stat = dealt.test().stat;
                        all += ways;
                        mean += ways * stat;
                        if stat > crit {
                            alarms += ways;
                        }
                    }
                }
            }
        }
        assert_eq!(all, choose(n, north));
        assert!((alarms / all - 6.0 / 70.0).abs() < 1e-12, "{}", alarms / all);
        assert_eq!(num(100.0 * alarms / all, 1), "8.6");
        // (the mean, (rows − 1)(columns − 1) N/(N − 1), as for any table with its totals kept)
        assert!((mean / all - 3.0 * 8.0 / 7.0).abs() < 1e-12);
    }

    /// Edges between columns move people between two columns only, each keeping its
    /// rows' shares; an empty column can be refilled; nothing is ever negative.
    #[test]
    fn edges_move_people_between_neighbours() {
        let mut s = ChiSquare::create();
        s.set_up(3);
        s.grow = None;
        s.table = NORTH_LIKES_TEA;
        let from = s.table;
        // the edge after tea to 20 % of everybody: tea 20, coffee 52
        s.set_edge(0, 0.2, &from);
        assert_eq!((s.table.col(0), s.table.col(1), s.table.col(2)), (20, 52, 28));
        assert_eq!(s.table.total(), 100);
        assert_eq!(s.table.n[0][0], 13, "tea keeps two thirds north");
        // to the far left: tea empty, and back
        s.set_edge(0, -1.0, &from);
        assert_eq!(s.table.col(0), 0);
        let empty = s.table;
        s.set_edge(0, 0.1, &empty);
        assert_eq!(s.table.col(0), 10);
        // the north's share of the whole table for a column that was empty
        assert_eq!(s.table.n[0][0], (10.0 * empty.row(0) as f64 / 100.0).round() as u32);
        // columns: two to four
        s.set_columns(4);
        assert_eq!(s.table.cols, 4);
        assert!(s.table.col(3) > 0);
        s.set_columns(2);
        assert_eq!((s.table.cols, s.table.n[0][2], s.table.n[1][3]), (2, 0, 0));
    }

    /// Whatever the settings, every number the picture is drawn from is finite, and the
    /// layout fits every window.
    #[test]
    fn degenerate_settings_stay_finite() {
        let windows = [(1075.0, 1000.0, 0.0), (1400.0, 760.0, 0.0), (390.0, 756.0, 44.0), (578.0, 700.0, 0.0), (700.0, 500.0, 0.0), (240.0, 180.0, 0.0), (1.0, 1.0, 0.0)];
        let mut s = ChiSquare::create();
        let check = |s: &ChiSquare| {
            let g = s.goals();
            assert!(g.counts.iter().chain(&g.levels).chain(&g.axes).chain(&g.hist).all(|v| v.is_finite()), "{:?} {:?}", s.mode, s.table);
            assert!(g.axes[X_CHI] > 0.0 && g.axes[Y_CHI] > 0.0 && g.axes[Y_COUNTS] > 0.0);
            let t = s.test();
            assert!(t.stat.is_finite() && t.parts.iter().all(|p| p.is_finite()));
            assert!(s.shown_parts().iter().all(|p| p.is_finite()));
        };
        for (rolls, six) in [(0, FAIR), (1, 0.0), (1, 1.0), (1200, 1.0), (6, 0.5)] {
            s.set_up(0);
            (s.rolls, s.six) = (rolls, six);
            s.refresh();
            for _ in 0..50 {
                if s.repeats.can_step() {
                    s.repeats.step();
                }
            }
            check(&s);
        }
        for t in [Table::of(2, [0; 4], [0; 4]), Table::of(3, [5, 0, 0, 0], [0, 0, 0, 0]), Table::of(4, [1, 0, 0, 0], [0, 0, 0, 1]), Table::of(2, [20_000, 0, 0, 0], [0, 0, 0, 0])] {
            s.set_up(4);
            s.table = t;
            s.refresh();
            for _ in 0..50 {
                if s.repeats.can_step() {
                    s.repeats.step();
                }
            }
            check(&s);
        }
        for (w, h, inset) in windows {
            for mode in [Mode::Die, Mode::Table] {
                let lay = Layout::of(Vec2::new(w, h), inset, mode);
                let rects = [lay.first, lay.under, lay.chart, lay.words];
                assert!(rects.iter().all(|(a, b)| a.is_finite() && b.is_finite()), "{w}x{h}");
                if w >= 390.0 && h >= 700.0 {
                    assert!(!lay.compact, "{w}x{h} {mode:?}");
                    assert!(rects.iter().all(|(a, b)| a.x >= 0.0 && a.y >= inset && b.x <= w && b.y <= h + 0.5 && b.x > a.x && b.y > a.y), "{w}x{h} {mode:?}: {rects:?}");
                    assert!(lay.first.1.y < lay.under.0.y && height(lay.first) > 100.0 && height(lay.chart) > 100.0, "{w}x{h} {mode:?}: {lay:?}");
                    if lay.narrow {
                        assert!(lay.under.1.y < lay.chart.0.y && lay.chart.1.y < lay.words.0.y && height(lay.words) > 50.0, "{w}x{h} {mode:?}: {lay:?}");
                    }
                }
            }
        }
    }

    /// After a change the picture glides to the new one and comes to rest exactly there,
    /// at any frame rate.
    #[test]
    fn glides_arrive_and_stop() {
        for hz in [30.0f32, 60.0, 144.0] {
            let mut s = ChiSquare::create();
            s.repeating = false;
            let g = s.goals();
            s.look.follow(&g, 0.0, true);
            assert!(!s.look.moving);
            s.seed += 1;
            let mut frames = 0;
            loop {
                let g = s.goals();
                s.look.follow(&g, 1.0 / hz, false);
                frames += 1;
                if !s.look.moving {
                    assert_eq!(s.look.counts.shown, g.counts);
                    break;
                }
                assert!((frames as f32) < 2.0 * hz, "still gliding after two seconds at {hz} Hz");
            }
            assert!(frames as f32 > 0.1 * hz, "arrived in {frames} frames at {hz} Hz: no glide");
        }
    }

    /// The numbers behind the captions (run with --ignored --nocapture).
    #[test]
    #[ignore]
    fn explore() {
        for seed in 0..40u64 {
            let fair = die_test(&die_sample(seed, 60, FAIR));
            let loaded = die_test(&die_sample(seed, 60, 0.25));
            println!("seed {seed:>2}: fair χ² {:.2} p {:.3}   loaded χ² {:.2} p {:.3} {:?}", fair.stat, fair.p().unwrap_or(1.0), loaded.stat, loaded.p().unwrap_or(1.0), die_sample(seed, 60, 0.25));
        }
        for k in 0..TRIALS.len() {
            let s = run(k, MOST);
            let t = s.test();
            println!(
                "{:<22} χ² {:.3} df {} p {:.4}  repeats {} mean {:.3} beyond {:.4}",
                TRIALS[k].0,
                t.stat,
                t.df,
                t.p().unwrap_or(1.0),
                s.repeats.done,
                s.repeats.mean().unwrap_or(0.0),
                s.repeats.share_beyond().unwrap_or(0.0)
            );
        }
    }
}

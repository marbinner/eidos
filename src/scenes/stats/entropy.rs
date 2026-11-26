//! Entropy and information: how much an outcome tells (its surprise), how much a
//! distribution holds on average (its entropy, and the yes/no questions it takes to
//! learn an outcome), what planning for the wrong distribution costs (the cross-entropy
//! and the Kullback–Leibler divergence), how much one variable tells about another (the
//! mutual information), and the average log-likelihood of a model as minus a
//! cross-entropy.
//!
//! Everything is drawn in screen space with the course's chart kit (`plot.rs`), laid
//! out for the window: side by side on a desktop, one under another on a phone. The
//! numbers are worked out in bits and shown in bits or nats.

use super::plot::{self, Glide, Plot, Rect, AXES, MARKS, OWN};
use super::*;
use crate::math::stats::unit;

pub const DEF: SceneDef = SceneDef {
    id: "entropy",
    title: "Entropy and information",
    group: MODELS,
    blurb: "Surprise, entropy, cross-entropy and the Kullback-Leibler divergence: how uncertain a distribution is, and how far one distribution is from another.",
    make: |_| Box::new(Info::create()),
};

// ----- the mathematics ----------------------------------------------------------------

/// The most outcomes a distribution here has.
const MOST: usize = 8;
/// The two variables of the table take this many values each.
const SIDE: usize = 3;
const CELLS: usize = SIDE * SIDE;
/// The nodes of a tree of questions: the outcomes, and at most one question fewer.
const NODES: usize = 2 * MOST - 1;

/// The surprise of an outcome of probability `p`, in bits: log₂(1/p). Nothing for a
/// certainty, one bit for a side of a fair coin, infinite for what cannot happen.
fn surprise(p: f64) -> f64 {
    if p > 0.0 {
        (-p.log2()).max(0.0)
    } else {
        f64::INFINITY
    }
}

/// The entropy of `p` in bits: its average surprise. (An outcome of probability 0 adds
/// nothing: 0 log 0 is taken as 0, the limit of p log p.)
fn entropy(p: &[f64]) -> f64 {
    p.iter().filter(|&&x| x > 0.0).map(|&x| x * surprise(x)).sum::<f64>().max(0.0)
}

/// The cross-entropy of `p` with `q` in bits: the average surprise, as `q` measures it,
/// of outcomes that follow `p`. Infinite if `q` rules out something `p` does not.
fn cross_entropy(p: &[f64], q: &[f64]) -> f64 {
    p.iter().zip(q).filter(|(&a, _)| a > 0.0).map(|(&a, &b)| a * surprise(b)).sum()
}

/// The Kullback–Leibler divergence from `p` to `q` in bits: Σ p log₂(p/q), the
/// cross-entropy less the entropy, worked out term by term. Infinite if `q` rules out
/// something `p` does not. (Never below 0; rounding can leave a sum a hair below.)
fn divergence(p: &[f64], q: &[f64]) -> f64 {
    let d: f64 = p.iter().zip(q).filter(|(&a, _)| a > 0.0).map(|(&a, &b)| if b > 0.0 { a * (a / b).log2() } else { f64::INFINITY }).sum();
    d.max(0.0)
}

/// The total of each row (the values of X) and of each column (Y) of a table.
fn margins(t: &[f64; CELLS]) -> ([f64; SIDE], [f64; SIDE]) {
    let (mut rows, mut cols) = ([0.0; SIDE], [0.0; SIDE]);
    for r in 0..SIDE {
        for c in 0..SIDE {
            rows[r] += t[r * SIDE + c];
            cols[c] += t[r * SIDE + c];
        }
    }
    (rows, cols)
}

/// The table two variables with these margins would have if they were independent:
/// each cell the product of its row's total and its column's.
fn product(t: &[f64; CELLS]) -> [f64; CELLS] {
    let (rows, cols) = margins(t);
    std::array::from_fn(|k| rows[k / SIDE] * cols[k % SIDE])
}

/// The mutual information of the table's two variables in bits: its divergence from
/// the table as if independent.
fn information(t: &[f64; CELLS]) -> f64 {
    divergence(t, &product(t))
}

/// Set entry `i` of the distribution `p` to `v` (kept from 0 to 1); the others keep
/// their proportions to each other and give way, so that all still add up to 1. (If
/// they are all 0, they share what is left equally.)
fn give_way(p: &mut [f64], i: usize, v: f64) {
    let n = p.len();
    if i >= n || n < 2 {
        return;
    }
    let v = if v.is_finite() { v.max(0.0).min(1.0) } else { p[i] };
    let rest: f64 = p.iter().enumerate().filter(|(j, _)| *j != i).map(|(_, x)| *x).sum();
    for (j, x) in p.iter_mut().enumerate() {
        if j != i {
            *x = if rest > 1e-12 { *x * (1.0 - v) / rest } else { (1.0 - v) / (n - 1) as f64 };
        }
    }
    p[i] = v;
    normalise(p);
}

/// Scale `p` to add up to 1 (all alike if it adds up to nothing).
fn normalise(p: &mut [f64]) {
    for x in p.iter_mut() {
        if !(x.is_finite() && *x > 0.0) {
            *x = 0.0;
        }
    }
    let sum: f64 = p.iter().sum();
    let n = p.len().max(1) as f64;
    for x in p.iter_mut() {
        *x = if sum > 0.0 { *x / sum } else { 1.0 / n };
    }
}

/// The best yes/no questions for a distribution, made by Huffman's method: join the
/// two least likely outcomes (or groups of outcomes) into one, again and again; each
/// joining is a question that tells the two apart, and an outcome takes as many
/// questions as joinings lie above it. An outcome that cannot happen needs none and is
/// left out.
#[derive(Clone, Debug, PartialEq)]
struct Questions {
    /// How many questions find each outcome (None: it cannot happen).
    depth: [Option<usize>; MOST],
    /// Node `k < MOST` is outcome `k`; node `MOST + j` is the `j`-th question made.
    /// Each node's question above it, and each question's "yes" and "no" branches
    /// ("yes" the likelier of the two, or the one taken first when they are alike).
    parent: [Option<usize>; NODES],
    branches: [Option<(usize, usize)>; NODES],
    /// The first question (or the only outcome, when one is certain).
    root: Option<usize>,
}

impl Questions {
    fn of(p: &[f64]) -> Questions {
        let mut qs = Questions { depth: [None; MOST], parent: [None; NODES], branches: [None; NODES], root: None };
        let mut weight = [0.0f64; NODES];
        let mut pool: Vec<usize> = (0..p.len().min(MOST)).filter(|&i| p[i] > 0.0).collect();
        for &i in &pool {
            weight[i] = p[i];
        }
        let mut next = MOST;
        while pool.len() > 1 && next < NODES {
            // (ties go to the node made first: an outcome before a question)
            pool.sort_by(|&a, &b| weight[a].total_cmp(&weight[b]).then(a.cmp(&b)));
            let (light, heavy) = (pool[0], pool[1]);
            pool.drain(..2);
            weight[next] = weight[light] + weight[heavy];
            let (yes, no) = if weight[heavy] > weight[light] { (heavy, light) } else { (light, heavy) };
            qs.branches[next] = Some((yes, no));
            qs.parent[yes] = Some(next);
            qs.parent[no] = Some(next);
            pool.push(next);
            next += 1;
        }
        qs.root = pool.first().copied();
        for i in 0..p.len().min(MOST) {
            if p[i] > 0.0 {
                let (mut d, mut at) = (0, i);
                while let Some(up) = qs.parent[at] {
                    d += 1;
                    at = up;
                }
                qs.depth[i] = Some(d);
            }
        }
        qs
    }

    /// The average number of questions for outcomes that follow `p`.
    fn average(&self, p: &[f64]) -> f64 {
        p.iter().zip(&self.depth).map(|(&x, d)| d.map_or(0.0, |d| x * d as f64)).sum()
    }

    /// The most questions any outcome takes.
    fn deepest(&self) -> usize {
        self.depth.iter().flatten().copied().max().unwrap_or(0)
    }

    /// The outcomes under node `k`, left to right as drawn ("yes" before "no").
    fn under(&self, k: usize, out: &mut Vec<usize>) {
        match self.branches.get(k).copied().flatten() {
            Some((yes, no)) => {
                self.under(yes, out);
                self.under(no, out);
            }
            None => out.push(k),
        }
    }

    /// Where each node is drawn: across from 0 to 1, each outcome in a column of its
    /// own in the order drawn and each question midway between its two branches; and
    /// down, by the number of questions above it. None for a node not in the tree.
    fn places(&self) -> [Option<(f32, f32)>; NODES] {
        let mut out = [None; NODES];
        let Some(root) = self.root else { return out };
        let mut leaves = Vec::with_capacity(MOST);
        self.under(root, &mut leaves);
        let m = leaves.len().max(1) as f32;
        let mut across = [0.0f32; NODES];
        for (k, &leaf) in leaves.iter().enumerate() {
            across[leaf] = (k as f32 + 0.5) / m;
        }
        // (questions are made after their branches: in the order made, each finds its
        // branches placed)
        for k in MOST..NODES {
            if let Some((yes, no)) = self.branches[k] {
                across[k] = 0.5 * (across[yes] + across[no]);
            }
        }
        for k in 0..NODES {
            let live = if k < MOST { self.depth[k].is_some() } else { self.branches[k].is_some() };
            if live {
                let (mut d, mut at) = (0, k);
                while let Some(up) = self.parent[at] {
                    d += 1;
                    at = up;
                }
                out[k] = Some((across[k], d as f32));
            }
        }
        out
    }
}

// ----- numbers as the picture writes them ------------------------------------------------

/// The units information is shown in.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Unit {
    Bits,
    Nats,
}

impl Unit {
    /// What one bit is in these units.
    fn scale(self) -> f64 {
        match self {
            Unit::Bits => 1.0,
            Unit::Nats => std::f64::consts::LN_2,
        }
    }

    fn name(self) -> &'static str {
        match self {
            Unit::Bits => "bits",
            Unit::Nats => "nats",
        }
    }

    /// The logarithm that counts in these units, as a formula writes it.
    fn log(self) -> &'static str {
        match self {
            Unit::Bits => r"\log_2",
            Unit::Nats => r"\ln",
        }
    }
}

/// An amount of information given in bits, in `unit`, as markup to go between
/// dollars: two decimals, or ∞.
fn amount(v: f64, unit: Unit) -> String {
    if v.is_nan() {
        "?".to_string()
    } else if v == f64::INFINITY {
        r"\infty".to_string()
    } else if v == f64::NEG_INFINITY {
        r"-\infty".to_string()
    } else {
        num(v * unit.scale(), 2)
    }
}

/// A probability as the picture writes it: up to three decimals, no trailing zeros.
fn share(p: f64) -> String {
    let text = format!("{:.3}", p.max(0.0).min(1.0));
    let text = text.trim_end_matches('0').trim_end_matches('.');
    if text.is_empty() {
        "0".to_string()
    } else {
        text.to_string()
    }
}

/// The name of outcome `i`, as markup to go between dollars.
fn outcome(i: usize) -> String {
    const NAMES: [&str; MOST] = ["A", "B", "C", "D", "E", "F", "G", "H"];
    format!(r"\mathrm{{{}}}", NAMES[i.min(MOST - 1)])
}

/// "A", "A or C", "A, B or C": the outcomes of a list, as a text.
fn listed(list: &[usize]) -> String {
    let names: Vec<String> = list.iter().map(|&i| format!("${}$", outcome(i))).collect();
    match names.split_last() {
        None => String::new(),
        Some((last, [])) => last.clone(),
        Some((last, rest)) => format!("{} or {}", rest.join(", "), last),
    }
}

// ----- the distributions the scene starts from -------------------------------------------

/// ½, ¼, ⅛, ... and the last two alike: probabilities whose surprises are whole
/// numbers of bits.
fn halving(n: usize) -> [f64; MOST] {
    let n = n.max(2).min(MOST);
    std::array::from_fn(|i| {
        if i + 1 < n {
            0.5f64.powi(i as i32 + 1)
        } else if i + 1 == n {
            0.5f64.powi(n as i32 - 1)
        } else {
            0.0
        }
    })
}

fn even(n: usize) -> [f64; MOST] {
    let n = n.max(1).min(MOST);
    std::array::from_fn(|i| if i < n { 1.0 / n as f64 } else { 0.0 })
}

/// Nine times in ten the first outcome, the rest shared by the others.
fn nine_in_ten(n: usize) -> [f64; MOST] {
    let n = n.max(2).min(MOST);
    std::array::from_fn(|i| {
        if i == 0 {
            0.9
        } else if i < n {
            0.1 / (n - 1) as f64
        } else {
            0.0
        }
    })
}

fn certain(n: usize) -> [f64; MOST] {
    let _ = n;
    std::array::from_fn(|i| if i == 0 { 1.0 } else { 0.0 })
}

/// A distribution drawn at random, every one of the `n` outcomes' shares as likely as
/// every other (a flat Dirichlet distribution).
fn random(n: usize, seed: u64) -> [f64; MOST] {
    let n = n.max(2).min(MOST);
    let mut rng = Rng::new(seed);
    let mut p: [f64; MOST] = std::array::from_fn(|i| if i < n { -unit(&mut rng).ln() } else { 0.0 });
    normalise(&mut p[..n]);
    p
}

/// `p` with `to` outcomes in place of its `from`: the ones in common keep their
/// proportions, a new one gets as much as an outcome had on average.
fn resized(p: &[f64; MOST], from: usize, to: usize) -> [f64; MOST] {
    let (from, to) = (from.max(1).min(MOST), to.max(2).min(MOST));
    let mut out: [f64; MOST] = std::array::from_fn(|i| if i < to { if i < from { p[i] } else { 1.0 / from as f64 } } else { 0.0 });
    normalise(&mut out[..to]);
    out
}

/// The tables the scene offers: in each, row r is X = r + 1 and column c is Y = c + 1.
/// A table in between: X and Y mostly agree.
const BETWEEN: [f64; CELLS] = [0.30, 0.05, 0.00, 0.05, 0.20, 0.05, 0.00, 0.10, 0.25];
/// Each decides the other: one cell in each row and each column.
const DECIDES: [f64; CELLS] = [0.0, 0.5, 0.0, 0.0, 0.0, 0.3, 0.2, 0.0, 0.0];
/// Nothing at all in the last row: X never takes its third value.
const EMPTY_ROW: [f64; CELLS] = [0.40, 0.10, 0.00, 0.10, 0.30, 0.10, 0.00, 0.00, 0.00];

/// Independent: each cell the product of its row's and its column's total.
fn independent() -> [f64; CELLS] {
    let (rows, cols) = ([0.5, 0.3, 0.2], [0.2, 0.5, 0.3]);
    std::array::from_fn(|k| rows[k / SIDE] * cols[k % SIDE])
}

// ----- a sample -------------------------------------------------------------------------

/// How many outcomes a sample has, and how many a second they are drawn at.
const SAMPLE: usize = 200;
const DRAW_RATE: f32 = 100.0;
/// The sample of the "things to try".
const SAMPLE_SEED: u64 = 13;

/// Outcomes drawn one after another from a distribution, counted.
#[derive(Clone, Debug)]
struct Sample {
    /// What they are drawn from, over how many outcomes, and how many are wanted (0:
    /// no sample).
    from: [f64; MOST],
    n: usize,
    wanted: usize,
    counts: [u32; MOST],
    drawn: usize,
    rng: Rng,
}

impl Sample {
    fn none() -> Sample {
        Sample { from: [0.0; MOST], n: 0, wanted: 0, counts: [0; MOST], drawn: 0, rng: Rng::new(0) }
    }

    fn start(from: &[f64; MOST], n: usize, seed: u64) -> Sample {
        Sample { from: *from, n: n.min(MOST), wanted: SAMPLE, counts: [0; MOST], drawn: 0, rng: Rng::new(seed) }
    }

    fn exists(&self) -> bool {
        self.wanted > 0
    }

    fn pending(&self) -> bool {
        self.drawn < self.wanted
    }

    /// One more outcome: the first whose running total passes a uniform number.
    fn draw_one(&mut self) {
        if !self.pending() || self.n == 0 {
            return;
        }
        let u = unit(&mut self.rng);
        let mut total = 0.0;
        // (should rounding leave the totals short of the number, the last that can happen)
        let mut pick = (0..self.n).rev().find(|&i| self.from[i] > 0.0).unwrap_or(0);
        for i in 0..self.n {
            total += self.from[i];
            if u < total && self.from[i] > 0.0 {
                pick = i;
                break;
            }
        }
        self.counts[pick] += 1;
        self.drawn += 1;
    }

    /// Each outcome's share of the sample so far (all 0 before the first).
    fn shares(&self) -> [f64; MOST] {
        let n = self.drawn.max(1) as f64;
        std::array::from_fn(|i| if self.drawn > 0 { self.counts[i] as f64 / n } else { 0.0 })
    }

    /// The average log-likelihood of the model `q`, per outcome, in bits:
    /// (1/N) Σ log₂ q(xₖ), straight from the outcomes' counts.
    fn log_likelihood(&self, q: &[f64]) -> f64 {
        if self.drawn == 0 {
            return 0.0;
        }
        let mut sum = 0.0;
        for (i, &c) in self.counts.iter().enumerate() {
            if c > 0 {
                let qi = q.get(i).copied().unwrap_or(0.0);
                if qi <= 0.0 {
                    return f64::NEG_INFINITY;
                }
                sum += c as f64 * qi.log2();
            }
        }
        sum / self.drawn as f64
    }
}

// ----- the scene ------------------------------------------------------------------------

/// Which picture is shown.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Look {
    /// One distribution: its surprises, its entropy, its questions.
    One,
    /// Two distributions: the cross-entropy and the divergence, and a sample.
    Two,
    /// Two variables: the mutual information.
    Table,
}

const LOOKS: [Look; 3] = [Look::One, Look::Two, Look::Table];
const LOOK_NAMES: [&str; 3] = ["one distribution", "two distributions", "two variables"];
const UNITS: [Unit; 2] = [Unit::Bits, Unit::Nats];

/// What the buttons on the picture offer, besides the views and the units.
const P_PRESETS: [&str; 7] = ["even", "halving", "nine in ten", "certain", "random", "fewer", "more"];
const Q_PRESETS: [&str; 2] = [r"$q$ even", r"$q = p$"];
const TABLE_PRESETS: [&str; 4] = ["independent", "each decides the other", "in between", "an empty row"];

/// The course's colours as this scene uses them.
const P: Color = TRUTH;
const Q: Color = ASSUMED;
const EXTRA: Color = ALARM;
const SHARES: Color = DATA;
const LIKELIHOOD: Color = ESTIMATE;

/// What the pictures show, written under them where there is room (a desktop).
const TREE_NOTE: &str = r"Each white dot asks a question: is it one of the outcomes named? Each green dot is an outcome, as many questions down as it takes to reach it. Huffman's method joins the two least likely outcomes into one, again and again, so a rarer outcome never takes fewer questions than a likelier one. No questions do better on average than the entropy; these do worse by less than one.";
const PLAN_NOTE: &str = r"Planning for $q$ means paying for each outcome its surprise under $q$, $\log_2(1/q)$ bits: what the best questions for $q$ take, exactly so when its probabilities are halves, quarters, eighths and so on. Outcomes that really follow $p$ cost $H(p, q)$ bits on average, the cross-entropy: never less than $H(p)$, and more by the divergence $D(p \,\|\, q)$, which is 0 only when $q = p$.";
const TABLE_NOTE: &str = r"Each square's area is how often its pair happens. The table as if independent keeps the totals of the rows and of the columns, and makes every row in proportion to the column totals: then knowing $X$ says nothing about $Y$. The mutual information is 0 exactly when the two tables are the same, and it can be no more than the smaller of $H(X)$ and $H(Y)$.";

/// The "things to try": (button, what to look for). Each is set up by `Info::set_up`.
const TRIALS: [(&str, &str); 5] = [
    (
        "a coin",
        r"A coin that lands heads ($\mathrm{A}$) nine times in ten. Heads surprises little ($0.15$ bits), tails a lot ($3.32$), but tails is rare: on average the coin holds $0.47$ bits. Drag the bar to $\tfrac12$: a fair coin holds 1 bit, the most two outcomes can. Drag it to 1: the coin holds nothing; the surprise of tails is then $\infty$, but tails never happens and adds 0. One question settles any coin that can land either way; a certain one needs none.",
    ),
    (
        "the wrong model",
        r"Outcomes follow $p = (\tfrac12, \tfrac14, \tfrac18, \tfrac18)$, but the questions are planned for $q$, four outcomes alike: two questions each. That takes $H(p, q) = 2$ bits on average where $H(p) = 1.75$ would do: the divergence $D(p \,\|\, q) = 0.25$ bits is the price of the wrong model. Under the outcomes, their extra bits: $\mathrm{A}$ costs one more, $\mathrm{C}$ and $\mathrm{D}$ one less, $0.25$ on average over outcomes from $p$. The other way round, $D(q \,\|\, p)$ is $0.25$ too, by coincidence; the next experiment shows how far apart the two can be.",
    ),
    (
        "ruled out",
        r"Now $q$ agrees with $p$ on $\mathrm{A}$ and $\mathrm{B}$ and calls $\mathrm{D}$ impossible. But $\mathrm{D}$ happens one time in eight, its surprise under $q$ is $\infty$, and so are the cross-entropy and the divergence: no amount of good guesses elsewhere makes up for ruling out what then happens. The other way round, $D(q \,\|\, p)$ is only $0.25$ bits: the divergence is not a distance.",
    ),
    (
        "two variables",
        r"A table of how often two variables $X$ and $Y$ take each pair of values, and beside it the table they would have if they were independent, with the same totals. The mutual information is how far apart the two tables are, $D = 0.70$ bits, and it is also $H(X) + H(Y) - H(X, Y)$: the overlap of the bars below. The button independent gives 0; each decides the other gives all of $H(X) = 1.49$ bits.",
    ),
    (
        "fitting is the same thing",
        r"200 outcomes are drawn from $p$; the cyan marks are their shares $\hat p$. The yellow number is the average log-likelihood of the model $q$ per outcome, $(1/N) \sum_k \log_2 q(x_k)$, and it is $-H(\hat p, q)$. With $q$ even it is $-2$ bits whatever the sample. Drag the purple bars onto the cyan marks, or press set $q$ to the shares: it rises to its highest, $-H(\hat p) = -1.77$. Maximum likelihood picks the model closest to the sample; the true $p$ itself scores a little less, $-1.78$.",
    ),
];

struct Info {
    look: Look,
    unit: Unit,
    /// The number of outcomes, and the two distributions over them (0 beyond `n`).
    n: usize,
    p: [f64; MOST],
    q: [f64; MOST],
    /// The table of two variables: row r is X = r + 1, column c is Y = c + 1.
    table: [f64; CELLS],
    sample: Sample,
    /// Seeds for the random preset and for new samples.
    seeds: u64,
    pace: crate::clock::Pace,
    /// What is drawn, gliding to what is: the number of outcomes, the bars of p and of
    /// q, the sample's shares, the cells of the table, and the tree of questions.
    slots: Glide,
    bars_p: Glide,
    bars_q: Glide,
    marks: Glide,
    cells: Glide,
    tree: Glide,
    /// Where the tree was last going (see `follow`).
    tree_goal: Vec<f32>,
    /// The scale under the tree: the entropy, the average number of questions, and
    /// the end of the scale. (The average jumps when an outcome's probability reaches
    /// 0, and its mark glides there.)
    scale: Glide,
    moving: bool,
    /// The view and number of outcomes drawn last: the frame after a change of either
    /// draws something new, not the old picture moved.
    drawn: Option<(Look, usize)>,
}

impl Info {
    fn create() -> Info {
        let mut s = Info {
            look: Look::One,
            unit: Unit::Bits,
            n: 4,
            p: halving(4),
            q: even(4),
            table: BETWEEN,
            sample: Sample::none(),
            seeds: 0,
            pace: Default::default(),
            slots: Glide::default(),
            bars_p: Glide::default(),
            bars_q: Glide::default(),
            marks: Glide::default(),
            cells: Glide::default(),
            tree: Glide::default(),
            tree_goal: Vec::new(),
            scale: Glide::default(),
            moving: false,
            drawn: None,
        };
        s.follow(0.0);
        s
    }

    /// Set up one of the "things to try": every setting its lesson depends on.
    fn set_up(&mut self, k: usize) {
        self.unit = Unit::Bits;
        self.sample = Sample::none();
        self.table = BETWEEN;
        self.n = 4;
        self.p = halving(4);
        self.q = even(4);
        match k {
            0 => {
                self.look = Look::One;
                self.n = 2;
                self.p = nine_in_ten(2);
                self.q = even(2);
            }
            1 => self.look = Look::Two,
            2 => {
                self.look = Look::Two;
                self.q = [0.5, 0.25, 0.25, 0.0, 0.0, 0.0, 0.0, 0.0];
            }
            3 => self.look = Look::Table,
            _ => {
                self.look = Look::Two;
                self.sample = Sample::start(&self.p, self.n, SAMPLE_SEED);
            }
        }
        self.pace.reset();
    }

    fn set_n(&mut self, n: usize) {
        let n = n.max(2).min(MOST);
        if n != self.n {
            self.p = resized(&self.p, self.n, n);
            self.q = resized(&self.q, self.n, n);
            self.n = n;
            self.sample = Sample::none();
        }
    }

    /// p has changed: a sample drawn from it as it was is no longer a sample from it.
    fn p_changed(&mut self) {
        self.sample = Sample::none();
    }

    fn preset_p(&mut self, k: usize) {
        match k {
            0 => self.p = even(self.n),
            1 => self.p = halving(self.n),
            2 => self.p = nine_in_ten(self.n),
            3 => self.p = certain(self.n),
            4 => {
                self.seeds += 1;
                self.p = random(self.n, 1000 + self.seeds);
            }
            5 => self.set_n(self.n.saturating_sub(1)),
            _ => self.set_n(self.n + 1),
        }
        self.p_changed();
    }

    fn preset_q(&mut self, k: usize) {
        match k {
            0 => self.q = even(self.n),
            1 => self.q = self.p,
            _ => {
                if self.sample.exists() && self.sample.drawn > 0 {
                    self.q = self.sample.shares();
                }
            }
        }
    }

    fn preset_table(&mut self, k: usize) {
        self.table = match k {
            0 => independent(),
            1 => DECIDES,
            2 => BETWEEN,
            _ => EMPTY_ROW,
        };
    }

    fn new_sample(&mut self) {
        self.seeds += 1;
        self.sample = Sample::start(&self.p, self.n, SAMPLE_SEED + 7919 * self.seeds);
        self.pace.reset();
    }

    /// The sample's outcomes, at their rate of scene time.
    fn work(&mut self, dt: f32, real_dt: f32) {
        if !self.sample.pending() || dt <= 0.0 {
            return;
        }
        let mut turn = scene_turn(&mut self.pace, DRAW_RATE, dt, real_dt);
        while turn.next() {
            self.sample.draw_one();
        }
        self.pace.end(turn);
    }

    /// Draw everything that moves `dt` seconds nearer to where it is going.
    fn follow(&mut self, dt: f32) {
        let mut moving = self.slots.follow(&[self.n as f32], dt, false);
        let wide = |v: &[f64; MOST]| -> Vec<f32> { v.iter().map(|&x| x as f32).collect() };
        moving |= self.bars_p.follow(&wide(&self.p), dt, false);
        moving |= self.bars_q.follow(&wide(&self.q), dt, false);
        moving |= self.marks.follow(&wide(&self.sample.shares()), dt, false);
        moving |= self.cells.follow(&self.table.map(|x| x as f32), dt, false);
        let mut tree = tree_target(&Questions::of(&self.p[..self.n]), self.n);
        // a node leaving the tree goes on to where it was going, fading, and its line
        // with it; one coming back glides from there
        if self.tree_goal.len() == tree.len() {
            for k in 0..NODES {
                let (i, e) = (TREE_NODES + 4 * k, TREE_LINES + 4 * k);
                if tree[i + 2] == 0.0 {
                    for j in [i, i + 1, i + 3, e, e + 1, e + 2, e + 3] {
                        tree[j] = self.tree_goal[j];
                    }
                }
            }
        }
        moving |= self.tree.follow(&tree, dt, false);
        self.tree_goal = tree;
        let p = &self.p[..self.n];
        let top = (self.n as f64).log2().ceil() + 1.0;
        moving |= self.scale.follow(&[entropy(p) as f32, Questions::of(p).average(p) as f32, top as f32], dt, false);
        self.moving = moving;
    }

    fn shown(glide: &Glide) -> [f64; MOST] {
        std::array::from_fn(|i| glide.get(i).max(0.0) as f64)
    }
}

// ----- the tree as drawn -----------------------------------------------------------------

/// Where the tree of questions is drawn, as numbers that can glide: the number of rows
/// (the most questions any outcome takes); where the words "yes" and "no" go, beside
/// the first question's two branches; then for each node across (0 to 1), down (in
/// questions), how much of it shows, and on which side of it its question is written
/// (−1 left, 1 right); then for each node its line up to the question above it (from,
/// to). A node not in the tree waits unseen: an outcome that cannot happen at the
/// bottom of its column, a question at the top. (Once the tree has been drawn, one
/// that leaves it keeps on its last course instead: see `Info::follow`.)
fn tree_target(qs: &Questions, n: usize) -> Vec<f32> {
    let places = qs.places();
    // (0 when one outcome is certain: no question is asked)
    let rows = qs.deepest() as f32;
    let top = qs.root.and_then(|r| places[r]).unwrap_or((0.5, 0.0));
    let hidden = |k: usize| if k < MOST { (((k as f32 + 0.5) / n.max(1) as f32).min(1.0), rows) } else { top };
    let mut out = Vec::with_capacity(TREE_LEN);
    out.push(rows);
    // the words: a little way down each of the first question's branches
    let (yes, no) = match qs.root.and_then(|r| qs.branches[r]) {
        Some((y, n)) => (places[y].unwrap_or(top), places[n].unwrap_or(top)),
        None => (top, top),
    };
    for (x, y) in [yes, no] {
        out.extend([top.0 + (x - top.0) * 0.45, top.1 + (y - top.1) * 0.45]);
    }
    for k in 0..NODES {
        let (at, seen) = places[k].map_or((hidden(k), 0.0), |p| (p, 1.0));
        // (a question is written on the side away from the line up to its own question)
        let side = match qs.parent[k].and_then(|up| places[up]) {
            Some(up) if up.0 > at.0 => -1.0,
            _ => 1.0,
        };
        out.extend([at.0, at.1, seen, side]);
    }
    for k in 0..NODES {
        let at = places[k].unwrap_or(hidden(k));
        let up = qs.parent[k].and_then(|u| places[u]).filter(|_| places[k].is_some()).unwrap_or(at);
        out.extend([at.0, at.1, up.0, up.1]);
    }
    out
}

/// The length of `tree_target`'s list, and where its parts start.
const TREE_LEN: usize = 5 + 4 * NODES + 4 * NODES;
const TREE_NODES: usize = 5;
const TREE_LINES: usize = 5 + 4 * NODES;

// ----- the layout ------------------------------------------------------------------------

/// Text sizes and margins for a canvas.
#[derive(Clone, Copy, Debug, PartialEq)]
struct Sizes {
    big: f32,
    head: f32,
    text: f32,
    small: f32,
    side: f32,
    button: f32,
}

impl Sizes {
    fn of(view: Vec2) -> Sizes {
        if view.x < 700.0 {
            Sizes { big: 17.0, head: 12.5, text: 11.5, small: 10.5, side: 8.0, button: 24.0 }
        } else {
            Sizes { big: 22.0, head: 14.0, text: 12.5, small: 11.5, side: 20.0, button: 26.0 }
        }
    }
}

/// Where everything goes on a canvas, for the view shown. (The rectangles of the other
/// views are left empty.)
#[derive(Clone, Debug, PartialEq)]
struct Layout {
    view: Vec2,
    s: Sizes,
    /// Too small a canvas for the pictures: a line of numbers alone.
    tiny: bool,
    /// The buttons that choose the view, the units, and this view's others.
    tabs: Vec<Rect>,
    units: Vec<Rect>,
    presets: Vec<Rect>,
    /// One distribution: its probabilities, each outcome's part of the entropy, the
    /// parts added up, and the questions.
    probs: Rect,
    parts: Rect,
    sum: Rect,
    tree: Rect,
    /// Two: p and q, the bits on average each way, and the sample.
    pq: Rect,
    sums: Rect,
    fit: Rect,
    /// Two variables: the table, the table as if independent, the side of a cell, and
    /// the bars of their information.
    joint: Vec2,
    product: Vec2,
    cell: f32,
    venn: Rect,
}

const NO_RECT: Rect = (Vec2::ZERO, Vec2::ZERO);

/// Buttons side by side from `x0`, each as wide as its words (`width`) and on a new
/// row when the next would pass `x1`. Returns them and the bottom of the last row.
fn flow(words: &[&str], x0: f32, x1: f32, y: f32, h: f32, gap: f32, width: &dyn Fn(&str) -> f32) -> (Vec<Rect>, f32) {
    let (mut x, mut y) = (x0, y);
    let mut out = Vec::with_capacity(words.len());
    for w in words {
        let wide = width(w) + 18.0;
        if x > x0 && x + wide > x1 {
            x = x0;
            y += h + gap;
        }
        out.push((Vec2::new(x, y), Vec2::new(x + wide, y + h)));
        x += wide + gap;
    }
    (out, if words.is_empty() { y - gap } else { y + h })
}

/// The top of a chart of probabilities: 1, and a hair more, so that the tick at 1 is
/// drawn (`plot::Ticks` counts its steps in single precision, and 1 / 0.2 comes to a
/// hair under 5 there).
const ONE: f64 = 1.0 + 1e-6;

/// The rows of text under a chart of bars, and how far apart they are.
fn line(s: &Sizes) -> f32 {
    s.text + 5.0
}

impl Layout {
    /// `presets` are this view's buttons; `measure` says how wide a text is at a size.
    fn of(view: Vec2, inset: f32, look: Look, presets: &[&str], measure: &dyn Fn(&str, f32) -> f32) -> Layout {
        let view = view.max(Vec2::splat(1.0));
        let s = Sizes::of(view);
        let mut lay = Layout {
            view,
            s,
            tiny: view.x < 240.0 || view.y < 260.0,
            tabs: Vec::new(),
            units: Vec::new(),
            presets: Vec::new(),
            probs: NO_RECT,
            parts: NO_RECT,
            sum: NO_RECT,
            tree: NO_RECT,
            pq: NO_RECT,
            sums: NO_RECT,
            fit: NO_RECT,
            joint: Vec2::ZERO,
            product: Vec2::ZERO,
            cell: 0.0,
            venn: NO_RECT,
        };
        if lay.tiny {
            return lay;
        }
        let (x0, x1) = (s.side, view.x - s.side);
        let width = |t: &str| measure(t, s.text);
        let top = inset.max(0.0) + if view.x < 700.0 { 6.0 } else { 12.0 };
        let (tabs, bottom) = flow(&LOOK_NAMES, x0, x1, top, s.button, 6.0, &width);
        lay.tabs = tabs;
        // this view's buttons, and the units at the end of their rows
        let mut words: Vec<&str> = presets.to_vec();
        words.extend(UNITS.map(Unit::name));
        let (mut all, bottom) = flow(&words, x0, x1, bottom + 6.0, s.button, 6.0, &width);
        lay.units = all.split_off(presets.len());
        lay.presets = all;
        let content = bottom + if view.x < 700.0 { 10.0 } else { 16.0 };
        let wide = view.x >= 700.0 && view.x > 0.8 * view.y;
        let avail = (view.y - content - 8.0).max(40.0);
        let l = line(&s);
        match look {
            Look::One => {
                // (above each chart: its title; below the first, the names, the surprises
                // and what they are; below the sum, its axis)
                let (above_a, below_a, above_b, below_b, above_c, bar_c, below_c, gap) = (40.0, 3.0 * l + 8.0, 30.0, l, s.big + 16.0, 22.0, 22.0, if wide { 16.0 } else { 8.0 });
                let fixed = above_a + below_a + above_b + below_b + above_c + bar_c + below_c + 2.0 * gap;
                if wide {
                    let split = (view.x * 0.58).round();
                    let rest = (avail - fixed).max(40.0);
                    let (ha, hb) = ((rest * 0.62).min(380.0), (rest * 0.38).min(230.0));
                    let gap = gap + ((rest - ha - hb) / 3.0).max(0.0).min(30.0);
                    let (l0, r0) = (x0 + 40.0, split - 16.0);
                    let mut y = content + above_a;
                    lay.probs = (Vec2::new(l0, y), Vec2::new(r0, y + ha));
                    y += ha + below_a + gap + above_b;
                    lay.parts = (Vec2::new(l0, y), Vec2::new(r0, y + hb));
                    y += hb + below_b + gap + above_c;
                    lay.sum = (Vec2::new(l0, y), Vec2::new(r0, y + bar_c));
                    let tree_bottom = (y + bar_c + below_c).max(content + 300.0).min(view.y - 8.0);
                    lay.tree = (Vec2::new(split + 30.0, content + 26.0), Vec2::new(x1, tree_bottom - 60.0));
                } else {
                    let (above_t, below_t) = (22.0, 2.0 * l + 40.0);
                    let rest = avail - fixed - above_t - below_t - gap;
                    let (l0, r0) = (x0 + 30.0, x1 - 4.0);
                    let mut y = content + above_a;
                    if rest >= 170.0 {
                        let (ha, hb, ht) = (rest * 0.42, rest * 0.18, rest * 0.4);
                        lay.probs = (Vec2::new(l0, y), Vec2::new(r0, y + ha));
                        y += ha + below_a + gap + above_b;
                        lay.parts = (Vec2::new(l0, y), Vec2::new(r0, y + hb));
                        y += hb + below_b + gap + above_c;
                        lay.sum = (Vec2::new(l0, y), Vec2::new(r0, y + bar_c));
                        y += bar_c + below_c + gap + above_t;
                        lay.tree = (Vec2::new(x0 + 4.0, y), Vec2::new(x1 - 4.0, y + ht));
                    } else {
                        // a short canvas: the probabilities and the entropy they add up to,
                        // and no room for the rest
                        let ha = (avail - above_a - below_a - gap - above_c - bar_c - below_c).max(40.0);
                        lay.probs = (Vec2::new(l0, y), Vec2::new(r0, y + ha));
                        y += ha + below_a + gap + above_c;
                        lay.sum = (Vec2::new(l0, y), Vec2::new(r0, y + bar_c));
                    }
                }
            }
            Look::Two => {
                let (above_a, below_a, gap) = (40.0, 3.0 * l + 8.0, if wide { 18.0 } else { 10.0 });
                // three rows of bars: a line of words over each
                let sums_h = 3.0 * (l + 26.0) + 20.0;
                if wide {
                    let split = (view.x * 0.6).round();
                    let (l0, r0) = (x0 + 40.0, split - 16.0);
                    let ha = (avail - above_a - below_a - gap - 24.0 - sums_h).max(60.0).min(380.0);
                    let mut y = content + above_a;
                    lay.pq = (Vec2::new(l0, y), Vec2::new(r0, y + ha));
                    y += ha + below_a + gap + 24.0;
                    lay.sums = (Vec2::new(x0 + 8.0, y), Vec2::new(r0, y + sums_h));
                    lay.fit = (Vec2::new(split + 30.0, content + 4.0), Vec2::new(x1, (content + 420.0).min(view.y - 8.0)));
                } else {
                    let fit_h = 168.0;
                    let ha = (avail - above_a - below_a - 2.0 * gap - 22.0 - sums_h - fit_h).max(60.0);
                    let (l0, r0) = (x0 + 30.0, x1 - 4.0);
                    let mut y = content + above_a;
                    lay.pq = (Vec2::new(l0, y), Vec2::new(r0, y + ha));
                    y += ha + below_a + gap + 22.0;
                    lay.sums = (Vec2::new(x0 + 4.0, y), Vec2::new(r0, y + sums_h));
                    y += sums_h + gap;
                    lay.fit = (Vec2::new(x0 + 4.0, y), Vec2::new(x1 - 4.0, y + fit_h));
                }
            }
            Look::Table => {
                // a block: its title and the names of the columns above the cells, the
                // names of the rows to their left, the totals right of and below them
                let (title, names_top, names_left, right, below) = (26.0, 18.0, if wide { 50.0 } else { 44.0 }, 64.0, 38.0);
                // (the bars, their axis, and two lines of words under them)
                let vrow = if wide { 26.0 } else { 22.0 };
                let venn_h = 24.0 + 3.0 * vrow + 26.0 + 2.0 * l + 12.0;
                let gap = if wide { 40.0 } else { 12.0 };
                let block = |cell: f32| names_left + 3.0 * cell + right;
                let high = |cell: f32| title + names_top + 3.0 * cell + below;
                let across = ((x1 - x0 - gap) * 0.5 - names_left - right) / 3.0;
                let side_by_side = across.min((avail - venn_h - gap - title - names_top - below) / 3.0);
                let stacked = ((x1 - x0) - names_left - right).min(3.0 * 200.0) / 3.0;
                let stacked = stacked.min((avail - venn_h - 2.0 * gap - 2.0 * (title + names_top + below)) / 6.0);
                let cell = if side_by_side >= 0.85 * stacked { side_by_side } else { stacked };
                let cell = cell.max(14.0).min(120.0);
                lay.cell = cell;
                if side_by_side >= 0.85 * stacked {
                    let used = 2.0 * block(cell) + gap;
                    let left = x0 + ((x1 - x0) - used).max(0.0) * 0.5;
                    lay.joint = Vec2::new(left, content);
                    lay.product = Vec2::new(left + block(cell) + gap, content);
                    let y = content + high(cell) + gap;
                    lay.venn = (Vec2::new(x0 + 8.0, y + 24.0), Vec2::new(x1 - 8.0, y + 24.0 + 3.0 * vrow));
                } else {
                    let left = x0 + ((x1 - x0) - block(cell)).max(0.0) * 0.5;
                    lay.joint = Vec2::new(left, content);
                    lay.product = Vec2::new(left, content + high(cell) + gap);
                    let y = content + 2.0 * high(cell) + 2.0 * gap;
                    lay.venn = (Vec2::new(x0 + 4.0, y + 24.0), Vec2::new(x1 - 4.0, y + 24.0 + 3.0 * vrow));
                }
            }
        }
        lay
    }
}

/// A paragraph on the screen: `text` broken into lines no wider than `width` pixels,
/// from `at` (its top-left corner, in pixels) down. Returns its height in pixels.
fn paragraph(f: &mut Frame, at: Vec2, width: f32, text: &str, size: f32, color: Color) -> f32 {
    let ppu = pixels_per_unit(f).max(1e-6);
    let Some(world) = f.cam.pick_ground(at * f.px) else { return 0.0 };
    let width = width.max(20.0) / ppu;
    wrapped(f, world.truncate(), width, text, size, color);
    wrapped_height(f, width, text, size)
}

/// How tall `paragraph` draws `text`.
fn paragraph_height(f: &Frame, width: f32, text: &str, size: f32) -> f32 {
    wrapped_height(f, width.max(20.0) / pixels_per_unit(f).max(1e-6), text, size)
}

/// A paragraph that says what a picture shows, under it, if there is room for it
/// between `y` and the bottom of the canvas.
fn note_below(f: &mut Frame, x0: f32, x1: f32, y: f32, text: &str, size: f32) {
    let width = x1 - x0;
    if width > 200.0 && f.view_size().y - 10.0 - y >= paragraph_height(f, width, text, size) {
        paragraph(f, Vec2::new(x0, y), width, text, size, pal::DIM);
    }
}

/// A line of four pixels' worth of nothing, in each kind of line the scene draws, and an
/// empty label: drawn in the first frame of a new picture so that nothing in it is
/// taken for something of the old one moved (`eidos motion` follows lines and labels
/// by their places in their batches).
fn mark_new_picture(f: &mut Frame) {
    for kind in [AXES, MARKS, plot::CURVES, plot::BUTTONS, OWN, OWN + 1, OWN + 2] {
        plot::line_of(f, kind, Vec2::ZERO, Vec2::new(0.0, 4.0), 1.0, Color::CLEAR);
    }
    f.text_px(Vec2::ZERO, "", 1.0, Color::CLEAR, Align2::LEFT_TOP);
}

impl Info {
    /// This view's buttons on the picture (besides the views and the units).
    fn presets(&self) -> &'static [&'static str] {
        match self.look {
            Look::One => &P_PRESETS,
            Look::Two => &Q_PRESETS,
            Look::Table => &TABLE_PRESETS,
        }
    }

    fn layout(&self, f: &Frame) -> Layout {
        Layout::of(f.view_size(), f.top_inset, self.look, self.presets(), &|t, size| f.text_size(t, size).x)
    }

    /// The buttons on the picture, and what their clicks do.
    fn buttons(&mut self, f: &mut Frame, lay: &Layout) {
        let size = lay.s.text;
        for (k, r) in lay.tabs.clone().into_iter().enumerate() {
            if plot::button(f, r, LOOK_NAMES[k], size, self.look == LOOKS[k]) {
                self.look = LOOKS[k];
            }
        }
        for (k, r) in lay.units.clone().into_iter().enumerate() {
            if plot::button(f, r, UNITS[k].name(), size, self.unit == UNITS[k]) {
                self.unit = UNITS[k];
            }
        }
        let words = self.presets();
        for (k, r) in lay.presets.clone().into_iter().enumerate() {
            if plot::button(f, r, words.get(k).copied().unwrap_or(""), size, false) {
                match self.look {
                    Look::One => self.preset_p(k),
                    Look::Two => self.preset_q(k),
                    Look::Table => self.preset_table(k),
                }
            }
        }
    }

    // ----- one distribution ------------------------------------------------------------

    fn plots_one(&self, lay: &Layout) -> (Plot, Plot) {
        let slots = self.slots.get(0).max(1.0) as f64;
        let top = 0.6 * self.unit.scale();
        (Plot::new(lay.probs, [0.0, slots], [0.0, ONE]), Plot::new(lay.parts, [0.0, slots], [0.0, top + 1e-6]))
    }

    /// The handles on the tops of p's bars. Says whether one is held.
    fn handles_one(&mut self, f: &mut Frame, lay: &Layout) -> bool {
        let (a, _) = self.plots_one(lay);
        let shown = Info::shown(&self.bars_p);
        let mut held = false;
        for i in 0..self.n {
            if let Some((_, y)) = a.handle(f, 100 + i as u32, i as f64 + 0.5, shown[i], P) {
                give_way(&mut self.p[..self.n], i, y);
                held = true;
            }
        }
        if held {
            self.p_changed();
        }
        held
    }

    fn draw_one(&self, f: &mut Frame, lay: &Layout) {
        let (s, n, unit) = (lay.s, self.n, self.unit);
        let (a, b) = self.plots_one(lay);
        let shown = Info::shown(&self.bars_p);
        let slots = a.x[1];
        let l = line(&s);
        let gap = (a.width() / slots as f32 * 0.2).max(1.0);

        // the probabilities, each outcome's name and surprise under its bar
        a.backing(f, if lay.view.x < 700.0 { 30.0 } else { 40.0 }, 3.0 * l + 8.0, 40.0, 10.0);
        a.y_grid(f);
        a.y_axis(f, "", s.small);
        a.title(f, &format!(r"how likely each outcome is: ${}$", tint(P, "p")), s.head);
        for i in 0..MOST {
            a.bar(f, i as f64, i as f64 + 1.0, 0.0, shown[i], gap, P.a(0.62));
        }
        for i in 0..n {
            let x = a.px_x(i as f64 + 0.5);
            let top = a.px_y(shown[i]).max(a.rect.0.y);
            f.text_px(Vec2::new(x, top - 9.0), share(self.p[i]), s.small, pal::TEXT, Align2::CENTER_BOTTOM);
            f.text_px(Vec2::new(x, a.rect.1.y + 8.0), format!("${}$", tint(P, &outcome(i))), s.text, P, Align2::CENTER_TOP);
            f.text_px(Vec2::new(x, a.rect.1.y + 8.0 + l), format!("${}$", amount(surprise(self.p[i]), unit)), s.small, pal::TEXT, Align2::CENTER_TOP);
        }
        let under = format!(r"under each: its surprise ${}(1/p)$, in {}", unit.log(), unit.name());
        f.text_px(Vec2::new(a.rect.0.x, a.rect.1.y + 8.0 + 2.0 * l + 1.0), under, s.small, pal::DIM, Align2::LEFT_TOP);

        // each outcome's part: probability times surprise (left out of a canvas too
        // short for it: the sum below still shows the parts)
        if !b.tiny() {
            self.draw_parts(f, lay, &b, &shown, gap);
        }

        // the parts added up: the entropy, against the most n outcomes can hold
        let part = |p: f64| if p > 0.0 { p * surprise(p) * unit.scale() } else { 0.0 };
        self.draw_sum(f, lay, &shown, part);
        self.draw_tree(f, lay);
    }

    fn draw_parts(&self, f: &mut Frame, lay: &Layout, b: &Plot, shown: &[f64; MOST], gap: f32) {
        let (s, n, unit) = (lay.s, self.n, self.unit);
        let l = line(&s);
        b.backing(f, if lay.view.x < 700.0 { 30.0 } else { 40.0 }, l, 26.0, 10.0);
        b.y_grid(f);
        b.y_axis(f, "", s.small);
        b.title(f, &format!(r"each outcome's part of the average: $p \, {}(1/p)$", unit.log()), s.head);
        let part = |p: f64| if p > 0.0 { p * surprise(p) * unit.scale() } else { 0.0 };
        for i in 0..MOST {
            b.bar(f, i as f64, i as f64 + 1.0, 0.0, part(shown[i]), gap, P.a(0.38));
        }
        for i in 0..n {
            let x = b.px_x(i as f64 + 0.5);
            let top = b.px_y(part(shown[i])).max(b.rect.0.y);
            f.text_px(Vec2::new(x, top - 3.0), num(part(self.p[i]), 2), s.small, pal::TEXT, Align2::CENTER_BOTTOM);
            f.text_px(Vec2::new(x, b.rect.1.y + 3.0), format!("${}$", outcome(i)), s.small, pal::DIM, Align2::CENTER_TOP);
        }
    }

    fn draw_sum(&self, f: &mut Frame, lay: &Layout, shown: &[f64; MOST], part: impl Fn(f64) -> f64) {
        let (s, unit) = (lay.s, self.unit);
        let slots = self.slots.get(0).max(1.0) as f64;
        let most = slots.log2().max(0.5) * unit.scale();
        // (a hair beyond the end, so that its tick is drawn: see `ONE`)
        let c = Plot::new(lay.sum, [0.0, most + 1e-6], [0.0, 1.0]);
        c.backing(f, if lay.view.x < 700.0 { 30.0 } else { 40.0 }, 24.0, s.big + 14.0, 10.0);
        let h = entropy(&self.p[..self.n]);
        f.text_px(
            Vec2::new(c.rect.0.x - 30.0f32.min(c.rect.0.x), c.rect.0.y - 8.0),
            format!(r"entropy ${} = {}$ {}", tint(P, "H(p)"), amount(h, unit), unit.name()),
            s.big,
            pal::TEXT,
            Align2::LEFT_BOTTOM,
        );
        // (as much of what the end of the bar means as there is room for beside the entropy)
        let formula = format!(r"${}\,{} = {}$", unit.log(), self.n, amount((self.n as f64).log2(), unit));
        let room = c.rect.1.x - c.rect.0.x - f.text_size(&format!(r"entropy $H(p) = {}$ {}", amount(h, unit), unit.name()), s.big).x;
        let most_text = [format!("most for {} outcomes: {formula}", self.n), format!("most: {formula}")].into_iter().find(|t| f.text_size(t, s.small).x + 16.0 < room);
        if let Some(most_text) = most_text {
            f.text_px(Vec2::new(c.rect.1.x, c.rect.0.y - 8.0), most_text, s.small, pal::DIM, Align2::RIGHT_BOTTOM);
        }
        // an empty bar the whole way: as much as there could be
        hud_rect(f, c.rect.0, c.rect.1, pal::FAINT.a(0.22));
        let mut from = 0.0;
        for i in 0..MOST {
            let w = part(shown[i]);
            if w <= 0.0 {
                continue;
            }
            let color = if i % 2 == 0 { P.a(0.75) } else { P.a(0.45) };
            c.bar(f, from, from + w, 0.0, 1.0, 0.0, color);
            let (x0, x1) = (c.px_x(from), c.px_x(from + w).min(c.rect.1.x));
            if i < self.n && x1 - x0 > 14.0 {
                f.text_px(Vec2::new((x0 + x1) * 0.5, (c.rect.0.y + c.rect.1.y) * 0.5), format!("${}$", outcome(i)), s.small, pal::WHITE, Align2::CENTER_CENTER);
            }
            from += w;
        }
        gliding_axis(f, &c, (self.n as f64).log2() * unit.scale() + 1e-6, OWN + 2, s.small);
    }

    fn draw_tree(&self, f: &mut Frame, lay: &Layout) {
        let (s, r) = (lay.s, lay.tree);
        if r.1.x - r.0.x < 60.0 || r.1.y - r.0.y < 40.0 {
            return;
        }
        let l = line(&s);
        let t = &self.tree.shown;
        let at = |k: usize| t.get(k).copied().unwrap_or(0.0);
        let asked = at(0);
        let rows = asked.max(1.0);
        // room at the left for the rows' names, above for the first question, below for the names
        let (left, pad_top, pad_bottom) = (if lay.view.x < 700.0 { 58.0 } else { 74.0 }, 22.0, 26.0);
        let row = ((r.1.y - r.0.y - pad_top - pad_bottom) / rows).min(64.0).max(4.0);
        // (as tall as the tree, which glides as it grows or shrinks)
        let bottom = r.0.y + pad_top + rows * row + pad_bottom;
        hud_panel(f, r.0 - Vec2::new(6.0, 24.0), Vec2::new(r.1.x + 6.0, bottom + 2.0 * l + 40.0));
        f.text_px(r.0 + Vec2::new(0.0, -6.0), "the best yes/no questions (a Huffman code)", s.head, pal::TEXT, Align2::LEFT_BOTTOM);
        let inner = (r.0.x + left, r.1.x - 10.0);
        let px = |x: f32, y: f32| Vec2::new(inner.0 + x * (inner.1 - inner.0), r.0.y + pad_top + y * row);
        // a faint line for each number of questions
        for d in 1..MOST {
            let seen = (asked - d as f32 + 1.0).max(0.0).min(1.0);
            let y = px(0.0, d as f32).y;
            plot::dashed_of(f, MARKS, Vec2::new(r.0.x + 4.0, y), Vec2::new(r.1.x, y), 1.0, pal::DIM.a(0.25 * seen));
            if seen > 0.05 {
                let words = if d == 1 { "1 question".to_string() } else { format!("{d} questions") };
                f.text_px(Vec2::new(r.0.x + 4.0, y - 2.0), words, s.small, pal::DIM.a(seen), Align2::LEFT_BOTTOM);
            }
        }
        // the lines from each node up to its question
        for k in 0..NODES {
            let e = TREE_LINES + 4 * k;
            let seen = at(TREE_NODES + 4 * k + 2);
            plot::line_of(f, OWN, px(at(e), at(e + 1)), px(at(e + 2), at(e + 3)), 1.6, pal::DIM.a(0.9 * seen));
        }
        // "yes" and "no" on the first question's branches
        if self.n >= 2 && self.p.iter().filter(|&&x| x > 0.0).count() >= 2 {
            f.text_px(px(at(1), at(2)) + Vec2::new(-6.0, 0.0), "yes", s.small, pal::DIM, Align2::RIGHT_CENTER);
            f.text_px(px(at(3), at(4)) + Vec2::new(6.0, 0.0), "no", s.small, pal::DIM, Align2::LEFT_CENTER);
        }
        let qs = Questions::of(&self.p[..self.n]);
        for k in 0..NODES {
            let i = TREE_NODES + 4 * k;
            let (p, seen, side) = (px(at(i), at(i + 1)), at(i + 2), at(i + 3));
            // (every node is drawn in every frame, unseen ones clear: the batches keep
            // their sizes, which is how `eidos motion` follows them)
            if k < MOST {
                plot::disc(f, p, 12.0, P.a(seen));
                f.text_px(p + Vec2::new(0.0, 8.0), format!("${}$", outcome(k)), s.text, P.a(seen), Align2::CENTER_TOP);
            } else {
                plot::disc(f, p, 7.0, pal::TEXT.a(seen));
                // the question: is it one of the outcomes down "yes"?
                if let Some((yes, _)) = qs.branches[k].filter(|_| seen > 0.02) {
                    let mut list = Vec::new();
                    qs.under(yes, &mut list);
                    let words = format!("{}?", listed(&list));
                    let w = f.text_size(&words, s.small).x;
                    if w < (inner.1 - inner.0) * 0.55 {
                        f.text_px(p + Vec2::new(side * (8.0 + 0.5 * w), -5.0), words, s.small, pal::TEXT.a(seen), Align2::CENTER_BOTTOM);
                    }
                }
            }
        }
        if qs.root.is_some_and(|root| root < MOST) {
            f.text_px(px(0.5, 0.0) + Vec2::new(0.0, 26.0), "certain: no question is needed", s.small, pal::DIM, Align2::CENTER_TOP);
        }
        self.draw_questions_against_entropy(f, lay, &qs, Vec2::new(r.0.x, bottom + 4.0), r.1.x - r.0.x);
        note_below(f, r.0.x, r.1.x, bottom + 2.0 * l + 56.0, TREE_NOTE, s.text);
    }

    /// The average number of questions against the entropy in bits (a question's
    /// answer is at most a bit): never below it, and less than one above it.
    fn draw_questions_against_entropy(&self, f: &mut Frame, lay: &Layout, qs: &Questions, at: Vec2, width: f32) {
        let s = lay.s;
        let p = &self.p[..self.n];
        let (h, avg) = (entropy(p), qs.average(p));
        let words = format!(r"on average $L = {}$ questions; entropy $H = {}$ bits", num(avg, 2), num(h, 2));
        f.text_px(at, words, s.text, pal::TEXT, Align2::LEFT_TOP);
        // a scale from 0 to the most the questions can take, with H to H + 1 marked
        let y = at.y + line(&s) + 20.0;
        // (the marks where the numbers are going: see `Info::scale`)
        let (h, avg, top) = (self.scale.get(0) as f64, self.scale.get(1) as f64, self.scale.get(2).max(1.0) as f64);
        let scale = Plot::new((Vec2::new(at.x + 10.0, y - 9.0), Vec2::new(at.x + width.min(360.0) - 10.0, y + 9.0)), [0.0, top], [0.0, 1.0]);
        if scale.tiny() {
            return;
        }
        plot::line_of(f, AXES, Vec2::new(scale.rect.0.x, y), Vec2::new(scale.rect.1.x, y), 1.0, pal::DIM.a(0.6));
        fill_rect(f, (Vec2::new(scale.px_x(h), y - 5.0), Vec2::new(scale.px_x((h + 1.0).min(top)), y + 5.0)), P.a(0.25));
        plot::line_of(f, MARKS, Vec2::new(scale.px_x(h), y - 7.0), Vec2::new(scale.px_x(h), y + 7.0), 1.6, P);
        f.text_px(Vec2::new(scale.px_x(h), y + 8.0), "$H$", s.small, P, Align2::CENTER_TOP);
        f.text_px(Vec2::new(scale.px_x((h + 1.0).min(top)), y + 8.0), "$H + 1$", s.small, P.a(0.7), Align2::CENTER_TOP);
        plot::disc(f, Vec2::new(scale.px_x(avg), y), 9.0, pal::WHITE);
        f.text_px(Vec2::new(scale.px_x(avg), y - 7.0), "$L$", s.small, pal::WHITE, Align2::CENTER_BOTTOM);
    }

    // ----- two distributions ------------------------------------------------------------

    fn plot_two(&self, lay: &Layout) -> Plot {
        Plot::new(lay.pq, [0.0, self.slots.get(0).max(1.0) as f64], [0.0, ONE])
    }

    fn handles_two(&mut self, f: &mut Frame, lay: &Layout) -> (bool, bool) {
        let a = self.plot_two(lay);
        let (sp, sq) = (Info::shown(&self.bars_p), Info::shown(&self.bars_q));
        let (mut held_p, mut held_q) = (false, false);
        for i in 0..self.n {
            if let Some((_, y)) = a.handle(f, 100 + i as u32, i as f64 + 0.3, sp[i], P) {
                give_way(&mut self.p[..self.n], i, y);
                held_p = true;
            }
            if let Some((_, y)) = a.handle(f, 200 + i as u32, i as f64 + 0.7, sq[i], Q) {
                give_way(&mut self.q[..self.n], i, y);
                held_q = true;
            }
        }
        if held_p {
            self.p_changed();
        }
        (held_p, held_q)
    }

    fn draw_two(&mut self, f: &mut Frame, lay: &Layout) {
        let (s, n, unit) = (lay.s, self.n, self.unit);
        let a = self.plot_two(lay);
        let (sp, sq, marks) = (Info::shown(&self.bars_p), Info::shown(&self.bars_q), Info::shown(&self.marks));
        let l = line(&s);
        let slot = a.width() / a.x[1] as f32;
        let gap = (slot * 0.06).max(0.5);
        a.backing(f, if lay.view.x < 700.0 { 30.0 } else { 40.0 }, 3.0 * l + 8.0, 40.0, 10.0);
        a.y_grid(f);
        a.y_axis(f, "", s.small);
        a.title(f, &format!(r"what happens, ${}$, and what is planned for, ${}$", tint(P, "p"), tint(Q, "q")), s.head);
        for i in 0..MOST {
            let x = i as f64;
            a.bar(f, x + 0.1, x + 0.5, 0.0, sp[i], gap, P.a(0.62));
            a.bar(f, x + 0.5, x + 0.9, 0.0, sq[i], gap, Q.a(0.62));
        }
        // the sample's shares, a mark across each slot
        if self.sample.exists() {
            for i in 0..MOST {
                let y = a.px_y(marks[i]).max(a.rect.0.y);
                let (x0, x1) = (a.px_x(i as f64 + 0.04), a.px_x(i as f64 + 0.96).min(a.rect.1.x));
                if x1 > x0 && i < n {
                    plot::line_of(f, OWN + 1, Vec2::new(x0, y), Vec2::new(x1, y), 2.4, SHARES);
                }
            }
        }
        let small_slots = slot < 44.0;
        for i in 0..n {
            let x = a.px_x(i as f64 + 0.5);
            if !small_slots {
                for (v, shown, dx, color) in [(self.p[i], sp[i], 0.3, P), (self.q[i], sq[i], 0.7, Q)] {
                    let top = a.px_y(shown).max(a.rect.0.y);
                    f.text_px(Vec2::new(a.px_x(i as f64 + dx), top - 9.0), share(v), s.small, color, Align2::CENTER_BOTTOM);
                }
            }
            f.text_px(Vec2::new(x, a.rect.1.y + 8.0), format!("${}$", outcome(i)), s.text, pal::TEXT, Align2::CENTER_TOP);
            let (pi, qi) = (self.p[i], self.q[i]);
            // what planning for q costs this outcome beyond planning for p
            let (extra, color) = if pi <= 0.0 {
                ("never".to_string(), pal::DIM)
            } else if qi <= 0.0 {
                (format!("${}$", amount(f64::INFINITY, unit)), EXTRA)
            } else {
                let v = (pi / qi).log2();
                let text = if v * unit.scale() >= 0.005 { format!("$+{}$", amount(v, unit)) } else { format!("${}$", amount(v, unit)) };
                (text, if v > 0.0 { EXTRA } else { pal::TEXT })
            };
            f.text_px(Vec2::new(x, a.rect.1.y + 8.0 + l), extra, s.small, color, Align2::CENTER_TOP);
        }
        let under = format!(r"under each: the extra {} it costs when planned for $q$, ${}(p/q)$", unit.name(), unit.log());
        f.text_px(Vec2::new(a.rect.0.x, a.rect.1.y + 8.0 + 2.0 * l + 1.0), under, s.small, pal::DIM, Align2::LEFT_TOP);

        self.draw_sums(f, lay);
        self.draw_fit(f, lay);
    }

    /// The bits on average each way: entropy and divergence end to end, making the
    /// cross-entropy.
    fn draw_sums(&self, f: &mut Frame, lay: &Layout) {
        let (s, n, unit) = (lay.s, self.n, self.unit);
        let r = lay.sums;
        if r.1.x - r.0.x < 80.0 {
            return;
        }
        let l = line(&s);
        hud_panel(f, r.0 - Vec2::new(8.0, 26.0), r.1 + Vec2::new(10.0, 4.0));
        f.text_px(r.0 + Vec2::new(0.0, -6.0), format!("{} per outcome on average, each way", unit.name()), s.head, pal::TEXT, Align2::LEFT_BOTTOM);
        let (p, q) = (&self.p[..n], &self.q[..n]);
        let mut rows: Vec<(String, f64, f64, Color, String, &str)> = vec![
            (
                format!(r"outcomes from ${}$, planned for ${}$: $D(p \,\|\, q) = {}$ extra", tint(P, "p"), tint(Q, "q"), amount(divergence(p, q), unit)),
                entropy(p),
                divergence(p, q),
                P,
                format!(r"$H(p, q) = {}$", amount(cross_entropy(p, q), unit)),
                "H(p)",
            ),
            (
                format!(r"outcomes from ${}$, planned for ${}$: $D(q \,\|\, p) = {}$ extra", tint(Q, "q"), tint(P, "p"), amount(divergence(q, p), unit)),
                entropy(q),
                divergence(q, p),
                Q,
                format!(r"$H(q, p) = {}$", amount(cross_entropy(q, p), unit)),
                "H(q)",
            ),
        ];
        let sampled = self.sample.exists();
        if sampled {
            let shares = self.sample.shares();
            let sh = &shares[..n];
            rows.push((
                format!(r"the sample's shares ${}$, planned for ${}$: $D(\hat p \,\|\, q) = {}$ extra", tint(SHARES, r"\hat p"), tint(Q, "q"), amount(divergence(sh, q), unit)),
                entropy(sh),
                divergence(sh, q),
                SHARES,
                format!(r"$H(\hat p, q) = {}$", amount(cross_entropy(sh, q), unit)),
                r"H(\hat p)",
            ));
        }
        // (the scale is the same for every number of outcomes' worth: up to one bit more
        // than the most n outcomes can hold)
        let slots = self.slots.get(0).max(1.0) as f64;
        let most = (slots.log2() + 1.0) * unit.scale();
        let total_w = f.text_size(r"$H(p, q) = 0.00$", s.small).x + 14.0;
        let scale = Plot::new((Vec2::new(r.0.x, r.0.y), Vec2::new((r.1.x - total_w).max(r.0.x + 20.0), r.1.y - 20.0)), [0.0, most], [0.0, 1.0]);
        for (k, (words, h, d, color, total, name)) in rows.iter().enumerate() {
            let y = r.0.y + k as f32 * (l + 26.0);
            f.text_px(Vec2::new(r.0.x, y), words.as_str(), s.small, pal::DIM, Align2::LEFT_TOP);
            let (y0, y1) = (y + l, y + l + 18.0);
            let (h, d) = (h * unit.scale(), d * unit.scale());
            let ends = |x: f64| scale.px_x(x.min(most)).min(scale.rect.1.x);
            fill_rect(f, (Vec2::new(ends(0.0), y0), Vec2::new(ends(h), y1)), color.a(0.6));
            let inside = format!("${} = {}$", name, amount(h / unit.scale(), unit));
            if f.text_size(&inside, s.small).x + 12.0 < ends(h) - ends(0.0) {
                f.text_px(Vec2::new(ends(0.0) + 6.0, (y0 + y1) * 0.5), inside, s.small, pal::WHITE, Align2::LEFT_CENTER);
            }
            let end = if d.is_finite() { ends(h + d) } else { scale.rect.1.x };
            fill_rect(f, (Vec2::new(ends(h), y0), Vec2::new(end, y1)), EXTRA.a(0.75));
            if !d.is_finite() || h + d > most {
                // off the scale: an arrow onwards
                hud_arrow(f, Vec2::new(scale.rect.1.x - 2.0, (y0 + y1) * 0.5), Vec2::new(scale.rect.1.x + 8.0, (y0 + y1) * 0.5), 1.6, EXTRA);
            }
            f.text_px(Vec2::new(end + if d.is_finite() && h + d <= most { 6.0 } else { 14.0 }, (y0 + y1) * 0.5), total.as_str(), s.small, pal::TEXT, Align2::LEFT_CENTER);
        }
        if !sampled {
            let words = r"a sample's shares $\hat p$, planned for $q$: no sample drawn yet";
            f.text_px(Vec2::new(r.0.x, r.0.y + 2.0 * (l + 26.0)), words, s.small, pal::DIM.a(0.7), Align2::LEFT_TOP);
        }
        gliding_axis(f, &scale, ((n as f64).log2() + 1.0) * unit.scale(), OWN + 2, s.small);
    }

    /// The sample, and the average log-likelihood of q.
    fn draw_fit(&mut self, f: &mut Frame, lay: &Layout) {
        let (s, n, unit) = (lay.s, self.n, self.unit);
        let r = lay.fit;
        let width = r.1.x - r.0.x;
        if width < 80.0 {
            return;
        }
        let words = if self.sample.exists() { "draw another 200" } else { "draw 200 outcomes from $p$" };
        let bw = f.text_size(words, s.text).x + 18.0;
        if plot::button(f, (r.0, r.0 + Vec2::new(bw, s.button)), words, s.text, false) {
            self.new_sample();
        }
        let mut y = r.0.y + s.button + 8.0;
        if !self.sample.exists() {
            let text = r"A model is fitted by making the data as likely as it can be. Draw a sample from $p$ to see what that has to do with the cross-entropy.";
            y += paragraph(f, Vec2::new(r.0.x, y), width, text, s.text, pal::DIM);
        } else {
            let q = &self.q[..n];
            let shares = self.sample.shares();
            let sh = &shares[..n];
            let ll = self.sample.log_likelihood(q);
            let said = format!("a sample of {} outcomes from $p$; their shares are the cyan marks", self.sample.drawn);
            y += paragraph(f, Vec2::new(r.0.x, y), width, &said, s.small, pal::DIM) + 4.0;
            f.text_px(Vec2::new(r.0.x, y), r"average log-likelihood of $q$, per outcome:", s.text, pal::TEXT, Align2::LEFT_TOP);
            y += line(&s) + 2.0;
            f.text_px(Vec2::new(r.0.x, y), format!("${}$ {}", amount(ll, unit), unit.name()), s.big, LIKELIHOOD, Align2::LEFT_TOP);
            y += s.big + 8.0;
            let best = format!(r"$= -H(\hat p, q)$; at most $-H(\hat p) = {}$, when $q = \hat p$", amount(-entropy(sh), unit));
            y += paragraph(f, Vec2::new(r.0.x, y), width, &best, s.small, pal::DIM) + 6.0;
            let words = r"set $q$ to the shares";
            let bw = f.text_size(words, s.text).x + 18.0;
            if plot::button(f, (Vec2::new(r.0.x, y), Vec2::new(r.0.x + bw, y + s.button)), words, s.text, false) && self.sample.drawn > 0 {
                self.q = self.sample.shares();
            }
            y += s.button;
        }
        hud_panel(f, r.0 - Vec2::new(8.0, 6.0), Vec2::new(r.1.x + 8.0, y + 8.0));
        if lay.fit.0.y < lay.sums.0.y {
            note_below(f, r.0.x, r.1.x, y + 30.0, PLAN_NOTE, s.text);
        }
    }

    // ----- two variables ----------------------------------------------------------------

    /// The centre of cell (row, column) of the block whose top-left corner is `o`.
    fn cell_centre(lay: &Layout, o: Vec2, k: usize) -> Vec2 {
        let names_left = if lay.view.x >= 700.0 { 50.0 } else { 44.0 };
        let c0 = o + Vec2::new(names_left, 26.0 + 18.0);
        c0 + Vec2::new(((k % SIDE) as f32 + 0.5) * lay.cell, ((k / SIDE) as f32 + 0.5) * lay.cell)
    }

    /// The side of the square that shows a probability in a cell: its area is the
    /// probability's share of the cell's.
    fn square(lay: &Layout, v: f64) -> f32 {
        0.92 * lay.cell * (v.max(0.0).min(1.0) as f32).sqrt()
    }

    fn handles_table(&mut self, f: &mut Frame, lay: &Layout) -> bool {
        let mut held = false;
        for k in 0..CELLS {
            let v = self.cells.get(k) as f64;
            let c = Info::cell_centre(lay, lay.joint, k);
            let at = c - Vec2::new(0.0, 0.5 * Info::square(lay, v));
            if let Some(to) = plot::screen_handle(f, 300 + k as u32, at, P) {
                let side = (2.0 * (c.y - to.y) / (0.92 * lay.cell)).max(0.0).min(1.0) as f64;
                give_way(&mut self.table, k, side * side);
                held = true;
            }
        }
        held
    }

    fn draw_table(&self, f: &mut Frame, lay: &Layout) {
        let (s, unit) = (lay.s, self.unit);
        let shown: [f64; CELLS] = std::array::from_fn(|k| self.cells.get(k).max(0.0) as f64);
        self.draw_block(f, lay, lay.joint, &shown, P, &format!(r"how often each pair happens: ${}$", tint(P, "p(x, y)")));
        self.draw_block(f, lay, lay.product, &product(&shown), Q, &format!(r"as if independent: ${}$", tint(Q, r"p(x)\,p(y)")));

        // the information each holds, as bars on one scale
        let r = lay.venn;
        if r.1.x - r.0.x < 80.0 {
            return;
        }
        let t = &self.table;
        let (rows, cols) = margins(t);
        let (hx, hy, hxy) = (entropy(&rows), entropy(&cols), entropy(t));
        let mi = information(t);
        let l = line(&s);
        hud_panel(f, r.0 - Vec2::new(8.0, 26.0), r.1 + Vec2::new(8.0, 2.0 * l + 30.0));
        f.text_px(r.0 + Vec2::new(0.0, -6.0), format!("how much each holds, in {}", unit.name()), s.head, pal::TEXT, Align2::LEFT_BOTTOM);
        let names = if lay.view.x < 700.0 { 58.0 } else { 70.0 };
        let most = (CELLS as f64).log2() * unit.scale();
        let scale = Plot::new((Vec2::new(r.0.x + names, r.0.y), Vec2::new(r.1.x - 70.0, r.1.y)), [0.0, most], [0.0, 1.0]);
        let row_h = (scale.height() / 3.0).max(6.0);
        let (a, b, c) = (hx * unit.scale(), hy * unit.scale(), hxy * unit.scale());
        let bars: [(&str, f64, f64); 3] = [(r"$H(X)$", 0.0, a), (r"$H(Y)$", c - b, c), (r"$H(X, Y)$", 0.0, c)];
        // (the overlap of H(X) and H(Y) is the mutual information: drawn in its own colour
        // on both bars, not laid over them)
        let (o0, o1) = (c - b, a.max(c - b));
        for (k, (name, from, to)) in bars.iter().enumerate() {
            let y0 = r.0.y + k as f32 * row_h + 3.0;
            let y1 = y0 + row_h - 6.0;
            f.text_px(Vec2::new(r.0.x, (y0 + y1) * 0.5), *name, s.text, pal::TEXT, Align2::LEFT_CENTER);
            let span = |a: f64, b: f64, color: Color, f: &mut Frame| fill_rect(f, (Vec2::new(scale.px_x(a), y0), Vec2::new(scale.px_x(b), y1)), color);
            if k == 2 {
                span(*from, *to, P.a(0.3), f);
            } else {
                span(*from, o0.max(*from).min(*to), P.a(0.5), f);
                span(o0.max(*from).min(*to), o1.min(*to).max(*from), EXTRA.a(0.7), f);
                span(o1.min(*to).max(*from), *to, P.a(0.5), f);
            }
            f.text_px(Vec2::new(scale.px_x(*to) + 6.0, (y0 + y1) * 0.5), amount(to - from, Unit::Bits), s.small, pal::TEXT, Align2::LEFT_CENTER);
        }
        let (x0, x1, y0, y1) = (scale.px_x(o0), scale.px_x(o1), r.0.y, r.0.y + 2.0 * row_h);
        plot::line_of(f, MARKS, Vec2::new(x0, y0), Vec2::new(x0, y1), 1.4, EXTRA);
        plot::line_of(f, MARKS, Vec2::new(x1, y0), Vec2::new(x1, y1), 1.4, EXTRA);
        let named = format!("$I = {}$", amount(mi, unit));
        if f.text_size(&named, s.small).x + 8.0 < x1 - x0 {
            f.caption_px(Vec2::new((x0 + x1) * 0.5, y0 + row_h), named, s.small, pal::WHITE, Align2::CENTER_CENTER);
        }
        scale.x_axis(f, "", s.small);
        let said = format!(
            r"mutual information ${} = H(X) + H(Y) - H(X, Y) = {}$ {}",
            tint(EXTRA, "I(X; Y)"),
            amount(hx + hy - hxy, unit),
            unit.name()
        );
        let at = Vec2::new(r.0.x, r.1.y + 26.0);
        f.text_px(at, said, s.text, pal::TEXT, Align2::LEFT_TOP);
        let said = format!(r"$= D(p(x, y) \,\|\, p(x)\,p(y)) = {}$: the pink overlap", amount(mi, unit));
        f.text_px(at + Vec2::new(0.0, l + 4.0), said, s.text, pal::TEXT, Align2::LEFT_TOP);
        note_below(f, r.0.x, r.1.x, at.y + 2.0 * l + 30.0, TABLE_NOTE, s.text);
    }

    /// One table: its cells as squares whose areas are their probabilities, the names
    /// of the values, and the totals of the rows and the columns.
    fn draw_block(&self, f: &mut Frame, lay: &Layout, o: Vec2, t: &[f64; CELLS], color: Color, title: &str) {
        let (s, cell) = (lay.s, lay.cell);
        let names_left = if lay.view.x >= 700.0 { 50.0 } else { 44.0 };
        let c0 = o + Vec2::new(names_left, 26.0 + 18.0);
        let grid = 3.0 * cell;
        hud_panel(f, o - Vec2::new(6.0, 0.0), c0 + Vec2::new(grid + 64.0, grid + 38.0));
        f.text_px(o + Vec2::new(0.0, 20.0), title, s.head, pal::TEXT, Align2::LEFT_BOTTOM);
        let (rows, cols) = margins(t);
        f.text_px(Vec2::new(c0.x + grid + 8.0, c0.y - 3.0), "totals", s.small, pal::DIM, Align2::LEFT_BOTTOM);
        for r in 0..SIDE {
            let y = c0.y + (r as f32 + 0.5) * cell;
            f.text_px(Vec2::new(c0.x - 6.0, y), format!("$X = {}$", r + 1), s.small, pal::DIM, Align2::RIGHT_CENTER);
            // the row's total
            let w = 44.0 * rows[r] as f32;
            fill_rect(f, (Vec2::new(c0.x + grid + 8.0, y - 4.0), Vec2::new(c0.x + grid + 8.0 + w, y + 4.0)), color.a(0.5));
            f.text_px(Vec2::new(c0.x + grid + 8.0, y + 6.0), share(rows[r]), s.small, pal::TEXT, Align2::LEFT_TOP);
        }
        for c in 0..SIDE {
            let x = c0.x + (c as f32 + 0.5) * cell;
            f.text_px(Vec2::new(x, c0.y - 3.0), format!("$Y = {}$", c + 1), s.small, pal::DIM, Align2::CENTER_BOTTOM);
            let h = 18.0 * cols[c] as f32;
            fill_rect(f, (Vec2::new(x - 4.0, c0.y + grid + 6.0), Vec2::new(x + 4.0, c0.y + grid + 6.0 + h)), color.a(0.5));
            f.text_px(Vec2::new(x + 7.0, c0.y + grid + 6.0), share(cols[c]), s.small, pal::TEXT, Align2::LEFT_TOP);
        }
        for k in 0..CELLS {
            let centre = Info::cell_centre(lay, o, k);
            let half = Vec2::splat(0.5 * cell - 1.0);
            plot::path_of(f, AXES, &[centre - half, Vec2::new(centre.x + half.x, centre.y - half.y), centre + half, Vec2::new(centre.x - half.x, centre.y + half.y), centre - half], 1.0, pal::DIM.a(0.3));
            let side = Info::square(lay, t[k]);
            if side > 0.5 {
                fill_rect(f, (centre - Vec2::splat(0.5 * side), centre + Vec2::splat(0.5 * side)), color.a(0.55));
            }
            if cell >= 30.0 {
                f.text_px(centre + Vec2::new(0.0, 0.5 * cell - 3.0), share(t[k]), s.small, pal::TEXT.a(0.85), Align2::CENTER_BOTTOM);
            }
        }
    }

    /// A canvas too small for the pictures: the main number alone.
    fn draw_tiny(&self, f: &mut Frame) {
        let view = f.view_size();
        let (n, unit) = (self.n, self.unit);
        let text = match self.look {
            Look::One => format!("$H(p) = {}$ {}", amount(entropy(&self.p[..n]), unit), unit.name()),
            Look::Two => format!(r"$D(p \,\|\, q) = {}$ {}", amount(divergence(&self.p[..n], &self.q[..n]), unit), unit.name()),
            Look::Table => format!("$I(X; Y) = {}$ {}", amount(information(&self.table), unit), unit.name()),
        };
        f.text_px(view * 0.5, text, 12.0, pal::TEXT, Align2::CENTER_CENTER);
    }
}

/// The axis under a chart whose range glides (`plot` as drawn now) to `to`: the round
/// values are chosen from where it is going and slide with it while it glides, in a
/// batch of their own (`kind`), so that none comes or goes on the way. One beyond the
/// end as drawn waits at the end, unseen.
fn gliding_axis(f: &mut Frame, plot: &Plot, to: f64, kind: u8, size: f32) {
    if plot.tiny() {
        return;
    }
    let (min, max) = plot.rect;
    plot::line_of(f, AXES, Vec2::new(min.x, max.y), max, 1.0, pal::DIM.a(0.8));
    let ticks = plot::Ticks::of(plot.x[0], to, plot.width());
    for v in ticks.values() {
        let x = plot.px_x(v);
        let seen = if x <= max.x + 0.5 { 1.0 } else { 0.0 };
        let x = x.min(max.x);
        plot::line_of(f, kind, Vec2::new(x, max.y), Vec2::new(x, max.y + 4.0), 1.0, pal::DIM.a(0.8 * seen));
        f.text_px(Vec2::new(x, max.y + 6.0), ticks.name(v), size, pal::DIM.a(seen), Align2::CENTER_TOP);
    }
}

/// A translucent rectangle in screen space, kept the right way round.
fn fill_rect(f: &mut Frame, (a, b): Rect, color: Color) {
    let (min, max) = (a.min(b), a.max(b));
    if max.x - min.x > 0.0 && max.y - min.y > 0.0 {
        plot::fill(f, (min, max), color);
    }
}

impl Scene for Info {
    fn view(&self) -> View {
        View::Flat { center: Vec2::ZERO, half_height: 1.0 }
    }

    fn settings(&self, s: &mut ViewSettings) {
        s.grid = false;
        s.bloom = s.bloom.min(0.2);
    }

    fn animated(&self) -> bool {
        self.moving || self.sample.pending()
    }

    fn pointer_hint(&self) -> Option<&'static str> {
        Some("dots on the bars and in the cells: drag    buttons on the picture: click")
    }

    fn ui(&mut self, ui: &mut egui::Ui, _: &Gfx) {
        ui::note(
            ui,
            r"How much does an outcome tell you? One you were sure of tells you nothing; the toss of a fair coin tells you one bit, the answer to one yes/no question. An outcome of probability $p$ tells $\log_2(1/p)$ bits: its surprise. The entropy of a distribution is its average surprise:",
        );
        ui::equation(ui, r"H(p) = \sum_x p(x) \log_2 \frac{1}{p(x)}");
        ui::note(ui, r"It is also how many yes/no questions it takes, on average, to learn an outcome: the best questions take at least $H$ and less than $H + 1$. Planning for the wrong distribution $q$ costs more: the cross-entropy, which is the entropy plus the Kullback–Leibler divergence. Drag the dots on top of the bars; the other bars give way, so that they still add up to 1.");

        ui::section(ui, "Things to try");
        for (k, (name, caption)) in TRIALS.iter().enumerate() {
            if ui::experiment(ui, name, caption) {
                self.set_up(k);
            }
        }

        ui::section(ui, "The picture");
        let mut look = LOOKS.iter().position(|l| *l == self.look).unwrap_or(0);
        if ui::choice(ui, &LOOK_NAMES, &mut look) {
            self.look = LOOKS[look];
        }
        match self.look {
            Look::One | Look::Two => {
                let mut n = self.n;
                if ui::slider_int(ui, "outcomes", &mut n, 2..=MOST) {
                    self.set_n(n);
                    self.p_changed();
                }
                if let Some(k) = ui::buttons(ui, &P_PRESETS[..5]) {
                    self.preset_p(k);
                }
                ui::note(ui, r"Shapes for $p$: halving is $\tfrac12, \tfrac14, \tfrac18, \ldots$, whose surprises are whole numbers of bits.");
                if self.look == Look::Two {
                    let mut words: Vec<&str> = Q_PRESETS.to_vec();
                    words.push(r"set $q$ to the shares");
                    if let Some(k) = ui::buttons(ui, &words) {
                        self.preset_q(k);
                    }
                    if ui::buttons(ui, &[r"draw 200 outcomes from $p$"]).is_some() {
                        self.new_sample();
                    }
                    ui::note(ui, r"A sample is drawn from $p$ as it is then; changing $p$ afterwards drops it.");
                }
            }
            Look::Table => {
                if let Some(k) = ui::buttons(ui, &TABLE_PRESETS) {
                    self.preset_table(k);
                }
            }
        }
        let mut u = UNITS.iter().position(|x| *x == self.unit).unwrap_or(0);
        if ui::choice(ui, &["bits", "nats"], &mut u) {
            self.unit = UNITS[u];
        }
        ui::note(ui, r"Nats count with the natural logarithm $\ln$ instead of $\log_2$: 1 nat is $1/\ln 2 = 1.44$ bits. The questions are counted in bits whatever is chosen: a yes/no answer holds at most one.");

        ui::section(ui, "Reading the picture");
        ui::key(
            ui,
            &[
                (P, r"green: $p$, how often each outcome really happens; the table of two variables"),
                (Q, r"purple: $q$, the distribution planned for; the table as if independent"),
                (EXTRA, "pink: the extra bits of planning for the wrong distribution (the divergence); the mutual information"),
                (SHARES, "cyan: the shares of the outcomes in a sample"),
                (LIKELIHOOD, r"yellow: the average log-likelihood of $q$, worked out from the sample"),
                (pal::TEXT, "white: the questions; the dot on the scale under them, their average number"),
            ],
        );
        ui::section(ui, "Words");
        ui::term(ui, "Surprise:", r"$\log_2(1/p)$ bits for an outcome of probability $p$: 0 for a certainty, more the rarer it is, and $\infty$ for what cannot happen.");
        ui::term(ui, "Bit:", "the information in the answer to a yes/no question whose two answers are equally likely.");
        ui::term(ui, "Entropy:", r"the average surprise; $n$ outcomes hold at most $\log_2 n$ bits, when all are equally likely.");
        ui::term(ui, "Huffman code:", "the best yes/no questions for a distribution, found by joining the two least likely outcomes into one, again and again.");
        ui::term(ui, "Cross-entropy:", r"$H(p, q) = \sum_x p(x) \log_2(1/q(x))$, the average surprise, as $q$ measures it, of outcomes that follow $p$.");
        ui::term(ui, "Kullback–Leibler divergence:", r"$D(p \,\|\, q) = H(p, q) - H(p)$: never below 0, 0 only when $q = p$, in general not the same both ways, and $\infty$ when $q$ rules out what $p$ allows.");
        ui::term(ui, "Mutual information:", r"$I(X; Y) = H(X) + H(Y) - H(X, Y)$, how many bits one variable tells about the other: 0 exactly when they are independent.");
        ui::term(ui, "Log-likelihood:", "the logarithm of the probability a model gives the data that were seen.");

        ui::section(ui, "Numbers");
        let (n, unit) = (self.n, self.unit);
        let (p, q) = (&self.p[..n], &self.q[..n]);
        let v = |x: f64| format!("${}$ {}", amount(x, unit), unit.name());
        match self.look {
            Look::One => {
                ui::readout(ui, r"entropy $H(p)$", v(entropy(p)));
                ui::readout(ui, "questions on average", num(Questions::of(p).average(p), 2));
                ui::readout(ui, format!("the most {n} outcomes can hold").as_str(), v((n as f64).log2()));
            }
            Look::Two => {
                ui::readout(ui, r"entropy $H(p)$", v(entropy(p)));
                ui::readout(ui, r"cross-entropy $H(p, q)$", v(cross_entropy(p, q)));
                ui::readout(ui, r"divergence $D(p \,\|\, q)$", v(divergence(p, q)));
                ui::readout(ui, r"divergence $D(q \,\|\, p)$", v(divergence(q, p)));
                if self.sample.exists() {
                    let shares = self.sample.shares();
                    ui::readout(ui, "outcomes drawn", self.sample.drawn.to_string());
                    ui::readout(ui, r"average log-likelihood of $q$", v(self.sample.log_likelihood(q)));
                    ui::readout(ui, r"its highest, $-H(\hat p)$", v(-entropy(&shares[..n])));
                }
            }
            Look::Table => {
                let t = &self.table;
                let (rows, cols) = margins(t);
                ui::readout(ui, r"$H(X)$", v(entropy(&rows)));
                ui::readout(ui, r"$H(Y)$", v(entropy(&cols)));
                ui::readout(ui, r"$H(X, Y)$", v(entropy(t)));
                ui::readout(ui, r"mutual information $I(X; Y)$", v(information(t)));
            }
        }
        ui::see_also(
            ui,
            &[
                ("likelihood", "maximum likelihood, which minimises a cross-entropy"),
                ("logistic", "cross-entropy as the loss of a classifier"),
                ("chisquare", "another measure of how far a table is from independent"),
                ("mlp", "a network trained by lowering a cross-entropy"),
            ],
        );
    }

    fn frame(&mut self, f: &mut Frame) {
        let lay = self.layout(f);
        if lay.tiny {
            self.follow(f.real_dt);
            self.draw_tiny(f);
            return;
        }
        if self.drawn != Some((self.look, self.n)) {
            if self.drawn.is_some() {
                mark_new_picture(f);
            }
            self.drawn = Some((self.look, self.n));
        }
        self.buttons(f, &lay);
        // (a button may have changed the view: the layout is then the new one's)
        let lay = if self.drawn != Some((self.look, self.n)) { self.layout(f) } else { lay };
        let (mut held_p, mut held_q, mut held_t) = (false, false, false);
        match self.look {
            Look::One => held_p = self.handles_one(f, &lay),
            Look::Two => (held_p, held_q) = self.handles_two(f, &lay),
            Look::Table => held_t = self.handles_table(f, &lay),
        }
        self.work(f.dt, f.real_dt);
        self.follow(f.real_dt);
        // (what is held is drawn where the hand puts it)
        if held_p {
            self.bars_p.snap(&self.p.map(|x| x as f32));
        }
        if held_q {
            self.bars_q.snap(&self.q.map(|x| x as f32));
        }
        if held_t {
            self.cells.snap(&self.table.map(|x| x as f32));
        }
        match self.look {
            Look::One => self.draw_one(f, &lay),
            Look::Two => self.draw_two(f, &lay),
            Look::Table => self.draw_table(f, &lay),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const DYADIC: [f64; 4] = [0.5, 0.25, 0.125, 0.125];

    /// A distribution drawn at random over `n` outcomes, some of them sometimes 0.
    fn any(n: usize, rng: &mut Rng, zeros: bool) -> Vec<f64> {
        let mut p: Vec<f64> = (0..n).map(|_| if zeros && rng.bool(0.2) { 0.0 } else { -unit(rng).ln() }).collect();
        normalise(&mut p);
        p
    }

    #[test]
    fn the_definitions_hold() {
        // surprise: nothing for a certainty, a bit for a coin, infinite for the impossible
        assert_eq!(surprise(1.0), 0.0);
        assert_eq!(surprise(0.5), 1.0);
        assert_eq!(surprise(0.125), 3.0);
        assert_eq!(surprise(0.0), f64::INFINITY);
        assert_eq!(entropy(&DYADIC), 1.75);
        assert_eq!(entropy(&[1.0, 0.0, 0.0]), 0.0);
        assert!((entropy(&even(8)) - 3.0).abs() < 1e-12);
        // 0 log 0 = 0: an outcome that cannot happen changes nothing
        assert_eq!(entropy(&[0.5, 0.5, 0.0]), entropy(&[0.5, 0.5]));
        assert_eq!(cross_entropy(&DYADIC, &[0.25; 4]), 2.0);
        assert_eq!(divergence(&DYADIC, &[0.25; 4]), 0.25);
        assert_eq!(divergence(&[0.5, 0.5, 0.0], &[0.5, 0.25, 0.25]), 0.0 + 0.5);
        assert_eq!(divergence(&[0.5, 0.5], &[1.0, 0.0]), f64::INFINITY);
        assert_eq!(cross_entropy(&[0.5, 0.5], &[1.0, 0.0]), f64::INFINITY);
        // but q may rule out what p rules out too
        assert!(divergence(&[1.0, 0.0], &[0.5, 0.5]).is_finite());
        let mut rng = Rng::new(5);
        for _ in 0..2000 {
            let n = 2 + rng.below(MOST - 1);
            let (p, q) = (any(n, &mut rng, true), any(n, &mut rng, false));
            let (h, x, d) = (entropy(&p), cross_entropy(&p, &q), divergence(&p, &q));
            // cross-entropy = entropy + divergence; the divergence is never negative, and
            // 0 at q = p; entropy is at most log2 n
            assert!((x - (h + d)).abs() < 1e-9, "{p:?} {q:?}: {x} against {h} + {d}");
            assert!(d >= 0.0 && divergence(&p, &p) < 1e-12 && cross_entropy(&p, &p) - h < 1e-12);
            assert!(h >= 0.0 && h <= (n as f64).log2() + 1e-12);
            // only 0 at q = p
            if p.iter().zip(&q).any(|(a, b)| (a - b).abs() > 1e-3) {
                assert!(d > 0.0, "{p:?} {q:?}");
            }
        }
    }

    #[test]
    fn the_questions_take_between_h_and_h_plus_one() {
        let mut rng = Rng::new(9);
        for _ in 0..5000 {
            let n = 2 + rng.below(MOST - 1);
            let p = any(n, &mut rng, true);
            let qs = Questions::of(&p);
            let (h, l) = (entropy(&p), qs.average(&p));
            assert!(l >= h - 1e-12 && l < h + 1.0, "{p:?}: {l} questions for {h} bits");
            // a prefix code: Kraft's sum is 1 for a full tree of the outcomes that can happen
            let kraft: f64 = qs.depth.iter().flatten().map(|&d| 0.5f64.powi(d as i32)).sum();
            let live = p.iter().filter(|&&x| x > 0.0).count();
            assert!(live < 2 || (kraft - 1.0).abs() < 1e-12, "{p:?}: {kraft}");
            for (i, &x) in p.iter().enumerate() {
                assert_eq!(qs.depth[i].is_some(), x > 0.0);
                // a rarer outcome never takes fewer questions than a likelier one
                for (j, &y) in p.iter().enumerate() {
                    if x > 0.0 && y > x {
                        assert!(qs.depth[i] >= qs.depth[j], "{p:?}: {i} and {j}");
                    }
                }
            }
            // the tree drawn: every node that is in it has a place inside the box
            let places = qs.places();
            for (k, at) in places.iter().enumerate() {
                if let Some((x, y)) = at {
                    assert!((0.0..=1.0).contains(x) && *y >= 0.0 && *y <= qs.deepest() as f32, "{k}");
                }
            }
        }
        // whole numbers of bits: exactly the entropy
        for n in 2..=MOST {
            for p in [halving(n), even(n)] {
                let p = &p[..n];
                if p.iter().all(|&x| x.log2().fract() == 0.0) {
                    let qs = Questions::of(p);
                    assert_eq!(qs.average(p), entropy(p), "{p:?}");
                    for (i, &x) in p.iter().enumerate() {
                        assert_eq!(qs.depth[i], Some(surprise(x) as usize));
                    }
                }
            }
        }
        // one outcome certain: no question; the ½, ¼, ⅛, ⅛ tree asks "A?", then "B?", then "C?"
        assert_eq!(Questions::of(&[0.0, 1.0, 0.0]).average(&[0.0, 1.0, 0.0]), 0.0);
        let qs = Questions::of(&DYADIC);
        let root = qs.root.expect("a tree");
        assert_eq!(qs.branches[root].map(|b| b.0), Some(0));
        assert_eq!(qs.depth[..4], [Some(1), Some(2), Some(3), Some(3)]);
    }

    #[test]
    fn mutual_information_is_the_divergence_from_independence() {
        let mut rng = Rng::new(3);
        for k in 0..2000 {
            let mut t = [0.0; CELLS];
            t.copy_from_slice(&any(CELLS, &mut rng, k % 2 == 0));
            let (rows, cols) = margins(&t);
            let by_entropies = entropy(&rows) + entropy(&cols) - entropy(&t);
            assert!((information(&t) - by_entropies).abs() < 1e-9, "{t:?}");
            assert!(information(&t) >= 0.0);
            // no more than either variable holds
            assert!(information(&t) <= entropy(&rows).min(entropy(&cols)) + 1e-9);
            // and 0 for the product of its margins, which keeps the margins; above 0 for
            // any table that is not that product
            assert!(information(&product(&t)) < 1e-12);
            let (rows2, cols2) = margins(&product(&t));
            assert!(rows.iter().zip(&rows2).chain(cols.iter().zip(&cols2)).all(|(a, b)| (a - b).abs() < 1e-12));
            if t.iter().zip(&product(&t)).any(|(a, b)| (a - b).abs() > 1e-3) {
                assert!(information(&t) > 0.0, "{t:?}");
            }
        }
        assert!(information(&independent()) < 1e-12);
        let (rows, cols) = margins(&DECIDES);
        assert!((information(&DECIDES) - entropy(&rows)).abs() < 1e-12 && (entropy(&rows) - entropy(&cols)).abs() < 1e-12);
        // an empty row: its cells and their products are 0, and everything stays finite
        let (rows, _) = margins(&EMPTY_ROW);
        assert_eq!(rows[2], 0.0);
        assert!(information(&EMPTY_ROW).is_finite() && information(&EMPTY_ROW) > 0.0);
    }

    #[test]
    fn dragging_a_bar_keeps_the_rest_in_proportion() {
        let mut p = DYADIC;
        give_way(&mut p, 0, 0.75);
        assert!((p.iter().sum::<f64>() - 1.0).abs() < 1e-12);
        assert!((p[0] - 0.75).abs() < 1e-12 && (p[1] / p[2] - 2.0).abs() < 1e-9 && (p[2] - p[3]).abs() < 1e-12);
        // all on one, then back: the others share what is left
        give_way(&mut p, 1, 1.0);
        assert_eq!(p, [0.0, 1.0, 0.0, 0.0]);
        give_way(&mut p, 1, 0.4);
        assert!((p[0] - 0.2).abs() < 1e-12 && (p[2] - 0.2).abs() < 1e-12 && (p[1] - 0.4).abs() < 1e-12);
        // out of range and not a number
        give_way(&mut p, 2, 7.0);
        assert_eq!(p[2], 1.0);
        give_way(&mut p, 2, f64::NAN);
        assert!(p.iter().all(|x| x.is_finite()) && (p.iter().sum::<f64>() - 1.0).abs() < 1e-12);
        // more outcomes and fewer
        let wider = resized(&halving(4), 4, 6);
        assert!((wider.iter().sum::<f64>() - 1.0).abs() < 1e-12 && wider[4] > 0.0 && wider[6] == 0.0);
        let fewer = resized(&halving(4), 4, 2);
        assert!((fewer[0] - 2.0 / 3.0).abs() < 1e-12 && fewer[2] == 0.0);
    }

    /// The scene after experiment `k`, its sample (if any) all drawn.
    fn after(k: usize) -> Info {
        let mut s = Info::create();
        s.set_up(k);
        while s.sample.pending() {
            s.sample.draw_one();
        }
        s
    }

    fn two(v: f64) -> String {
        amount(v, Unit::Bits)
    }

    #[test]
    fn the_experiments_show_what_their_captions_say() {
        // a coin: 0.15 and 3.32 bits of surprise, 0.47 on average; fair, 1 bit; certain, 0;
        // one question whatever it is
        let s = after(0);
        assert_eq!((s.look, s.n, s.unit), (Look::One, 2, Unit::Bits));
        let p = &s.p[..2];
        assert_eq!((two(surprise(p[0])), two(surprise(p[1])), two(entropy(p))), ("0.15".into(), "3.32".into(), "0.47".into()));
        assert_eq!(Questions::of(p).average(p), 1.0);
        for (heads, bits) in [(0.5, 1.0), (1.0, 0.0)] {
            let mut p = s.p;
            give_way(&mut p[..2], 0, heads);
            assert_eq!(entropy(&p[..2]), bits);
            assert_eq!(Questions::of(&p[..2]).average(&p[..2]), if heads == 1.0 { 0.0 } else { 1.0 });
        }
        // (at 1 the tails' surprise is written ∞ and its part 0)
        assert_eq!(amount(surprise(0.0), Unit::Bits), r"\infty");

        // the wrong model: 2 bits where 1.75 would do; extra bits +1, 0, −1, −1; and 0.25
        // the other way round too
        let s = after(1);
        let (p, q) = (&s.p[..4], &s.q[..4]);
        assert_eq!((s.look, s.n), (Look::Two, 4));
        assert_eq!((cross_entropy(p, q), entropy(p), divergence(p, q), divergence(q, p)), (2.0, 1.75, 0.25, 0.25));
        let extra: Vec<f64> = (0..4).map(|i| (p[i] / q[i]).log2()).collect();
        assert_eq!(extra, vec![1.0, 0.0, -1.0, -1.0]);
        assert!(Questions::of(q).depth[..4].iter().all(|d| *d == Some(2)));

        // ruled out: q agrees on A and B, calls D impossible; D happens one time in eight;
        // ∞ one way and 0.25 bits the other
        let s = after(2);
        let (p, q) = (&s.p[..4], &s.q[..4]);
        assert!(p[0] == q[0] && p[1] == q[1] && q[3] == 0.0 && p[3] == 0.125);
        assert_eq!((divergence(p, q), cross_entropy(p, q), surprise(q[3])), (f64::INFINITY, f64::INFINITY, f64::INFINITY));
        assert_eq!(divergence(q, p), 0.25);

        // two variables: D = 0.70 bits = H(X) + H(Y) − H(X, Y); independent 0; each
        // deciding the other, all of H(X) = 1.49
        let s = after(3);
        assert_eq!(s.look, Look::Table);
        let t = &s.table;
        let (rows, cols) = margins(t);
        assert_eq!(two(information(t)), "0.70");
        assert!((information(t) - (entropy(&rows) + entropy(&cols) - entropy(t))).abs() < 1e-12);
        let mut s = s;
        s.preset_table(0);
        assert_eq!(two(information(&s.table)), "0.00");
        s.preset_table(1);
        let (rows, _) = margins(&s.table);
        assert_eq!((two(information(&s.table)), two(entropy(&rows))), ("1.49".into(), "1.49".into()));

        // fitting: 200 outcomes; with q even −2 bits; at q = the shares its highest,
        // −H(p̂) = −1.77; the true p scores −1.78; and it is always −H(p̂, q)
        let mut s = after(4);
        assert_eq!((s.look, s.sample.drawn, s.q), (Look::Two, SAMPLE, even(4)));
        let shares = s.sample.shares();
        assert_eq!(s.sample.log_likelihood(&s.q), -2.0);
        assert_eq!(two(-entropy(&shares[..4])), "−1.77");
        assert_eq!(two(s.sample.log_likelihood(&DYADIC)), "−1.78");
        s.preset_q(2);
        assert_eq!(s.q, shares);
        let best = s.sample.log_likelihood(&s.q);
        assert!((best + entropy(&shares[..4])).abs() < 1e-12);
        let mut rng = Rng::new(17);
        for _ in 0..500 {
            let q = any(4, &mut rng, false);
            let ll = s.sample.log_likelihood(&q);
            assert!((ll + cross_entropy(&shares[..4], &q)).abs() < 1e-12);
            assert!(ll < best);
        }
    }

    /// Edge cases: an outcome at 0, all on one, q ruling something out, two outcomes, an
    /// empty row. Every number written is finite or a written ∞, and nothing is NaN.
    #[test]
    fn degenerate_settings_are_told_honestly() {
        for p in [certain(4), [0.5, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], halving(2)] {
            let n = if p[2] == 0.0 && p[1] == 0.0 { 4 } else { 2 };
            let p = &p[..n];
            assert!(entropy(p).is_finite() && Questions::of(p).average(p).is_finite());
            for &x in p {
                assert!(!amount(surprise(x), Unit::Bits).contains('?'));
                let part = if x > 0.0 { x * surprise(x) } else { 0.0 };
                assert!(part.is_finite());
            }
        }
        // q with a zero where p has none: ∞ written; p with a zero where q has none: finite
        assert_eq!(amount(divergence(&[0.5, 0.5], &[1.0, 0.0]), Unit::Nats), r"\infty");
        assert!(divergence(&[1.0, 0.0], &[0.5, 0.5]).is_finite());
        // a sample's log-likelihood under a q that rules out what was drawn
        let mut sample = Sample::start(&even(2), 2, 1);
        while sample.pending() {
            sample.draw_one();
        }
        assert_eq!(sample.log_likelihood(&[1.0, 0.0]), f64::NEG_INFINITY);
        assert_eq!(amount(f64::NEG_INFINITY, Unit::Bits), r"-\infty");
        assert_eq!(sample.counts[..2].iter().sum::<u32>(), SAMPLE as u32);
        // never an outcome that cannot happen
        let mut sample = Sample::start(&[0.0, 0.3, 0.0, 0.7, 0.0, 0.0, 0.0, 0.0], 4, 2);
        while sample.pending() {
            sample.draw_one();
        }
        assert!(sample.counts[0] == 0 && sample.counts[2] == 0);
        // the tree of a certain outcome, and of two
        for p in [&[1.0, 0.0][..], &[0.0, 0.0, 1.0], &[0.3, 0.7]] {
            let target = tree_target(&Questions::of(p), p.len());
            assert!(target.len() == TREE_LEN && target.iter().all(|v| v.is_finite()), "{p:?}");
        }
        // the empty row's table as if independent
        assert!(product(&EMPTY_ROW).iter().all(|v| v.is_finite()));
        assert_eq!(share(1.0), "1");
        assert_eq!(share(0.0), "0");
        assert_eq!(share(0.125), "0.125");
        assert_eq!(share(0.5), "0.5");
    }

    /// The layout of every window: finite, the charts inside the canvas and not on top
    /// of each other, for each view.
    #[test]
    fn the_layout_fits_every_window() {
        let measure = |t: &str, size: f32| text_width(t, size);
        for (w, h, inset) in [(1075.0, 1000.0, 0.0), (1400.0, 760.0, 0.0), (575.0, 700.0, 0.0), (390.0, 756.0, 44.0), (700.0, 500.0, 0.0), (240.0, 180.0, 0.0), (1.0, 1.0, 0.0), (3000.0, 2000.0, 0.0)] {
            for look in LOOKS {
                let presets = match look {
                    Look::One => &P_PRESETS[..],
                    Look::Two => &Q_PRESETS[..],
                    Look::Table => &TABLE_PRESETS[..],
                };
                let lay = Layout::of(Vec2::new(w, h), inset, look, presets, &measure);
                let rects = [lay.probs, lay.parts, lay.sum, lay.tree, lay.pq, lay.sums, lay.fit, lay.venn];
                assert!(rects.iter().all(|(a, b)| a.is_finite() && b.is_finite()) && lay.joint.is_finite() && lay.product.is_finite() && lay.cell.is_finite());
                if lay.tiny {
                    continue;
                }
                let used: Vec<Rect> = rects.iter().copied().filter(|r| r.1 != Vec2::ZERO).collect();
                for (a, b) in &used {
                    assert!(a.x >= 0.0 && a.y >= inset && b.x <= w + 0.5 && b.y <= h + 0.5, "{look:?} {w}x{h}: {a} {b}");
                    assert!(b.x > a.x && b.y > a.y, "{look:?} {w}x{h}: {a} {b}");
                }
                for r in lay.tabs.iter().chain(&lay.units).chain(&lay.presets) {
                    assert!(r.0.x >= 0.0 && r.1.x <= w + 0.5 && r.0.y >= inset, "{look:?} {w}x{h}: a button at {:?}", r);
                }
                if w >= 390.0 && h >= 700.0 {
                    match look {
                        Look::One => {
                            assert!(lay.probs.1.y < lay.parts.0.y && lay.parts.1.y < lay.sum.0.y, "{w}x{h}");
                            assert!(lay.probs.1.y - lay.probs.0.y > 100.0, "{w}x{h}: {:?}", lay.probs);
                        }
                        Look::Two => assert!(lay.pq.1.y < lay.sums.0.y && lay.pq.1.y - lay.pq.0.y > 100.0, "{w}x{h}"),
                        Look::Table => assert!(lay.cell >= 40.0, "{w}x{h}: cells of {}", lay.cell),
                    }
                }
            }
        }
    }

    /// After a change the picture glides to the new one and comes to rest exactly
    /// there, at any frame rate.
    #[test]
    fn glides_arrive_and_stop() {
        for hz in [30.0f32, 60.0, 144.0] {
            let mut s = Info::create();
            assert!(!s.moving);
            s.set_up(0);
            let mut frames = 0;
            loop {
                s.follow(1.0 / hz);
                frames += 1;
                if !s.moving {
                    assert_eq!(s.bars_p.shown, s.p.map(|x| x as f32).to_vec());
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
        let s = after(3);
        let (rows, cols) = margins(&s.table);
        println!("table: H(X) {} H(Y) {} H(X,Y) {} I {}", entropy(&rows), entropy(&cols), entropy(&s.table), information(&s.table));
        let s = after(4);
        let shares = s.sample.shares();
        println!("sample {:?}: -H(shares) {} ll(p) {} ll(even) {}", &shares[..4], -entropy(&shares[..4]), s.sample.log_likelihood(&DYADIC), s.sample.log_likelihood(&even(4)));
        // How far the best model (the shares) is above the true p in other samples: on
        // average about 0.011 bits, (k - 1) / (2 N ln 2). The one shown is typical (0.014)
        // and its two numbers round apart at two decimals.
        for seed in 1..80u64 {
            let mut sample = Sample::start(&halving(4), 4, seed);
            while sample.pending() {
                sample.draw_one();
            }
            let sh = sample.shares();
            println!("seed {seed}: shares {:?} -H {:.4} ll(p) {:.4} gap {:.4}", &sh[..4], -entropy(&sh[..4]), sample.log_likelihood(&DYADIC), divergence(&sh[..4], &DYADIC));
        }
    }
}
